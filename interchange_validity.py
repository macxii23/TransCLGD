"""Quality/rationality criteria for generated interchange graphs (structure/geometry/attributes, target/distribution-agnostic); depends only on numpy/json (no torch)"""
import json                                                                            #dependency
import numpy as np
from collections import defaultdict

VALID_DEFAULTS = dict(max_nodes=200, min_ramp=2, lcc_frac=0.9, meanang=50.0, kinkfrac=0.25)  #unified default thresholds (calibrated on large graph_data sample): lcc>=0.9 (~0.5% false-reject on real graphs, removes fragmented) + zero tolerance for isolated nodes/dangling ramp clusters (engineering connectivity)


def read_graph(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)


def _xy_nodes(g):
    return np.asarray([[float(n.get('x', 0.0)), float(n.get('y', 0.0))] for n in g.get('nodes', [])], dtype=np.float64)


def _adj_edges(g):
    nodes, edges = g.get('nodes', []), g.get('edges', [])
    id_map = {n.get('id', i): i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for e in edges:
        s, t = id_map.get(e.get('source'), -1), id_map.get(e.get('target'), -1)
        if s >= 0 and t >= 0 and s != t: adj[s, t] = adj[t, s] = 1
    return adj


def _gen_mask(g, generated_labels): #node-level boolean mask: True=generated part (default ramp only), False=kept part (mainline/ramp_kept) — read directly from JSON string field, independent of NODE_TYPE_MAP
    return np.asarray([n.get('node_type', 'mainline') in generated_labels for n in g.get('nodes', [])], dtype=bool)


def _lcc_frac(adj): #largest connected component fraction (union-find over upper-triangular edges): reflects whether the generated graph is fragmented/has islands/dangling ramps
    n = adj.shape[0]
    if n == 0: return 0.0
    par = list(range(n))
    def f(x):
        r = x
        while par[r] != r: r = par[r]
        while par[x] != r: par[x], x = r, par[x]
        return r
    for i, j in zip(*np.triu(adj, 1).nonzero()):
        ri, rj = f(int(i)), f(int(j))
        if ri != rj: par[ri] = rj
    sz = defaultdict(int)
    for i in range(n): sz[f(i)] += 1
    return max(sz.values()) / n


def _ramp_turning_angles(g, gen_labels, adj): #extract chains between degree!=2 branch points, collect internal turning angles of ramp-majority chains (deg: 0=straight, 180=U-turn); reflects ramp smoothness
    nodes = g.get('nodes', []); n = len(nodes)
    if n < 3: return []
    nbr = [set(np.nonzero(adj[i])[0].tolist()) for i in range(n)]; xy = _xy_nodes(g)
    is_ramp = [nd.get('node_type', 'mainline') in gen_labels for nd in nodes]
    branch = {i for i in range(n) if len(nbr[i]) != 2}; seen = set(); angs = []
    def _emit(ch):                                                                     #only ramp-majority chains count toward internal turning angles
        if sum(is_ramp[i] for i in ch) <= len(ch) / 2: return
        for k in range(1, len(ch) - 1):
            u, v = xy[ch[k]] - xy[ch[k - 1]], xy[ch[k + 1]] - xy[ch[k]]
            nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            if nu < 1e-12 or nv < 1e-12: continue
            angs.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, float(np.dot(u, v)) / (nu * nv)))))))
    for s in branch:
        for nb in nbr[s]:
            ek = (min(s, nb), max(s, nb))
            if ek in seen: continue
            ch = [s, nb]; seen.add(ek); cur = nb
            while cur not in branch:
                nxts = [v for v in nbr[cur] if (min(cur, v), max(cur, v)) not in seen]
                if len(nxts) != 1: break
                nx = nxts[0]; seen.add((min(cur, nx), max(cur, nx))); ch.append(nx); cur = nx
            _emit(ch)
    return angs


def _ramp_off_mainline(adj, gen_mask): #whether any generated (ramp) node is unreachable from the mainline (non-generated nodes) -> dangling ramp cluster, no traffic in/out, engineering-infeasible
    n = adj.shape[0]; main = np.nonzero(~gen_mask)[0]
    if len(main) == 0: return bool(gen_mask.any())                                     #no mainline anchor: all ramps dangling
    reached = np.zeros(n, bool); st = main.tolist()
    for s in main: reached[s] = True
    while st:                                                                          #BFS: mainline-reachable set (any edge)
        for v in np.nonzero(adj[st.pop()])[0]:
            if not reached[v]: reached[v] = True; st.append(int(v))
    return bool((gen_mask & ~reached).any())                                           #some ramp lies outside the mainline connected region


def graph_valid(g, gen_labels, max_nodes, min_ramp, lcc_frac, meanang, kinkfrac, allow_isolated=False, require_ramp_on_mainline=True): #structural/geometric/connectivity/attribute validity of a generated graph (target/distribution-agnostic): returns (ok, reason). Decides whether it is a passable, legal interchange
    nodes = g.get('nodes', []); n = len(nodes)
    if n < 3: return False, 'too_few_nodes'                                            #node collapse
    if n > max_nodes: return False, 'too_many_nodes'                                    #node explosion
    xy = _xy_nodes(g)
    if not np.isfinite(xy).all(): return False, 'nonfinite_coord'                      #NaN/Inf coordinate
    if float(xy.std(0).mean()) < 1e-6: return False, 'degenerate_coord'                #all points coincide (degenerate)
    if any(int(nd.get('num_lanes', 1)) < 1 for nd in nodes): return False, 'bad_lane'  #illegal lane count
    adj = _adj_edges(g)
    if int(adj.sum()) == 0: return False, 'no_edge'                                    #no edge
    if not allow_isolated and int((adj.sum(1) == 0).sum()) > 0: return False, 'isolated_node'  #isolated node (degree 0): dangling node, engineering-unreachable (~0.05% false-reject on real graphs)
    if _lcc_frac(adj) < lcc_frac: return False, 'fragmented'                           #fragmented/isolated/ramp not connected to mainline (default lcc>=0.9)
    gm = _gen_mask(g, gen_labels)
    if int(gm.sum()) < min_ramp: return False, 'no_ramp'                               #no ramp generated (degenerated to pure mainline)
    if require_ramp_on_mainline and _ramp_off_mainline(adj, gm): return False, 'ramp_off_mainline'  #dangling ramp cluster (unreachable from mainline): disconnected alignment, no traffic in/out (~1.5% false-reject on real graphs)
    angs = _ramp_turning_angles(g, gen_labels, adj)                                    #ramp smoothness: mean turning angle / near-U-turn fraction (thresholds from real-data p99; a single sharp bend/hairpin allowed, only rejects "twisted throughput, not smooth")
    if angs and (float(np.mean(angs)) > meanang or float(np.mean(np.asarray(angs) > 90.0)) > kinkfrac): return False, 'ramp_not_smooth'
    return True, 'ok'
