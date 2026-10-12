import torch, numpy as np, networkx as nx #segment coarse-graph construction/uncoarsening: fold degree==2 chains into segments, connect segments sharing branch points
from torch_geometric.graphgym.config import cfg #read unified config dataset.segment.len_scale

def _len_scale(): return float(cfg.dataset.get('segment', {}).get('len_scale', 10.0)) #segment-length normalization scale (unified config)


def _extract_chains(n, adj): #extract continuous chains between degree!=2 branch/endpoints from adj[N,N], returns tensor list (each chain is a sequence of original nodes)
    G = nx.Graph(); G.add_nodes_from(range(n)) #build undirected graph, used for degree statistics
    a = adj.copy(); np.fill_diagonal(a, 0) #remove self-loops
    for i in range(n): #add undirected edges
        for j in range(i + 1, n):
            if a[i, j] > 0 or a[j, i] > 0: G.add_edge(i, j)
    branch = {nd for nd in G.nodes if G.degree(nd) != 2} #branch/endpoints: degree!=2
    chains, visited = [], set()
    for s in sorted(branch): #starting from each branch point, trace the chain along degree==2 nodes
        for nbr in sorted(G.neighbors(s)):
            ek = (min(s, nbr), max(s, nbr))
            if ek in visited: continue
            chain, cur = [s, nbr], nbr; visited.add(ek)
            while cur not in branch:
                nxts = [v for v in G.neighbors(cur) if (min(cur, v), max(cur, v)) not in visited]
                if len(nxts) != 1: break
                nxt = nxts[0]; visited.add((min(cur, nxt), max(cur, nxt))); chain.append(nxt); cur = nxt
            chains.append(chain)
    for u, v in G.edges(): #handle pure cycles that are all degree==2 (no branch point to start from)
        ek = (min(u, v), max(u, v))
        if ek in visited: continue
        chain, cur = [u, v], v; visited.add(ek)
        while cur != chain[0]:
            nxts = [w for w in G.neighbors(cur) if (min(cur, w), max(cur, w)) not in visited]
            if not nxts: break
            nxt = nxts[0]; visited.add((min(cur, nxt), max(cur, nxt))); chain.append(nxt); cur = nxt
        chains.append(chain)
    return chains


def build_segment_graph(pos, adj, ntype): #forward-coarsen the original graph into a coarse graph: pos[N,2] adj[N,N] ntype[N] -> dict of seg_* fields (segment feature = length L + type one-hot)
    n = pos.shape[0]
    adj_np = adj.numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
    nty_np = ntype.numpy() if isinstance(ntype, torch.Tensor) else np.asarray(ntype)
    chains = _extract_chains(n, adj_np)
    if not chains: chains = [[i] for i in range(n)] #degenerate: when there are no chains, each node forms its own segment
    M = len(chains)
    seg_pos = np.zeros((M, 2), dtype=np.float32) #segment midpoint coordinates, for visualization only
    seg_ntype = np.zeros(M, dtype=np.int64) #segment node type: majority vote over all nodes in the chain
    endpoints = [] #endpoint set of each segment (used to build inter-segment adjacency: shared endpoints)
    for k, c in enumerate(chains):
        seg_pos[k] = pos_np[c].mean(axis=0)
        vals, cnts = np.unique(nty_np[c], return_counts=True)
        seg_ntype[k] = int(vals[cnts.argmax()])
        endpoints.append({c[0], c[-1]}) #chain start/end as the segment endpoints
    seg_adj = np.zeros((M, M), dtype=np.int64) #inter-segment adjacency W: connect an edge if the endpoint intersection is non-empty
    for i in range(M):
        for j in range(i + 1, M):
            if endpoints[i] & endpoints[j]:
                seg_adj[i, j] = seg_adj[j, i] = 1
    eidx = np.stack(np.nonzero(seg_adj), axis=0) if seg_adj.sum() > 0 else np.zeros((2, 0), dtype=np.int64)
    seg_edge_index = torch.tensor(eidx, dtype=torch.long)
    seg_edge_attr = torch.ones(seg_edge_index.shape[1], dtype=torch.long)
    seg_adj_flat = torch.tensor(seg_adj.flatten(), dtype=torch.float)
    chain_lens = torch.tensor([len(c) for c in chains], dtype=torch.long) #original node count per chain = ground-truth segment length L (for uncoarsening)
    chain_flat = torch.tensor([v for c in chains for v in c], dtype=torch.long) #flattened original node ids within chains, used with chain_lens to restore
    C = max(int(max(nty_np.max(), 1)) + 1, 2) #number of node_type classes (at least 2)
    seg_nt = torch.tensor(seg_ntype, dtype=torch.long)
    type_oh = torch.zeros(M, C, dtype=torch.float).scatter_(1, seg_nt.unsqueeze(1), 1.0) #type one-hot [M,C]
    l_norm = (chain_lens.float() / _len_scale()).unsqueeze(1) #length normalization [M,1]
    seg_x = torch.cat([l_norm, type_oh], dim=1) #segment feature = [length L_norm(1) + type one-hot(C)], dim 1+C (C=2->3); jointly reconstructed by encoder/diffusion
    return dict(seg_x=seg_x, seg_edge_index=seg_edge_index,
                seg_edge_attr=seg_edge_attr, seg_adj_flat=seg_adj_flat,
                seg_node_type=seg_nt, seg_pos_orig=torch.tensor(seg_pos, dtype=torch.float),
                seg_num_nodes=torch.tensor(M, dtype=torch.long),
                seg_chain_lens=chain_lens, seg_chain_flat=chain_flat)


def split_seg_x(seg_x): #split segment feature [M,1+C] -> (L_norm[M], type_idx[M]); denormalize to get integer segment lengths lens
    sx = seg_x.detach().cpu() if isinstance(seg_x, torch.Tensor) else torch.as_tensor(seg_x)
    l_norm, type_logits = sx[:, 0], sx[:, 1:]
    lens = torch.clamp((l_norm * _len_scale()).round().long(), min=1) #l=max(1,round(L_norm·scale))
    return lens, type_logits.argmax(dim=1)


def uncoarsen_segment_graph(W, lens, seg_type): #uncoarsen: coarse graph (W[M,M] segment connections, lens[M] segment lengths, seg_type[M] segment types) -> node-level full adjacency adj[N,N] + node_type[N]
    M = len(lens)
    lens = [max(1, int(round(float(l)))) for l in (lens.tolist() if isinstance(lens, torch.Tensor) else lens)]
    Wn = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.asarray(W)
    sty = seg_type.tolist() if isinstance(seg_type, torch.Tensor) else list(seg_type)
    starts = np.cumsum([0] + lens); N = int(starts[-1]) #segment i occupies nodes [starts[i],starts[i+1])
    adj = np.zeros((N, N), dtype=np.int64); ntype = np.zeros(N, dtype=np.int64)
    for i in range(M):
        a, b = int(starts[i]), int(starts[i + 1]); ntype[a:b] = int(sty[i])
        for r in range(a, b - 1): adj[r, r + 1] = adj[r + 1, r] = 1 
    G = nx.Graph(); G.add_nodes_from(range(M)) #coarse graph: maximal cliques = shared-branch junctions
    for i in range(M):
        for j in range(i + 1, M):
            if Wn[i, j] > 0 or Wn[j, i] > 0: G.add_edge(i, j)
    seg_ports = [[int(starts[i]), int(starts[i + 1]) - 1] for i in range(M)] #two port nodes per segment (start/end)
    used = [0] * M #number of assigned ports (<=2 ends per segment: two branch points)
    for clq in nx.find_cliques(G): #treat each maximal clique as a junction, each segment in the clique takes one port and edges are added pairwise 
        if len(clq) < 2: continue
        terms = []
        for s in clq:
            p = seg_ports[s][min(used[s], 1)]; used[s] = min(used[s] + 1, 2); terms.append(p)
        for x in range(len(terms)):
            for y in range(x + 1, len(terms)):
                adj[terms[x], terms[y]] = adj[terms[y], terms[x]] = 1 #port copies within a junction are pairwise interconnected
    return torch.tensor(adj, dtype=torch.long), torch.tensor(ntype, dtype=torch.long)
