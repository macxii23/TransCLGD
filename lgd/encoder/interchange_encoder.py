import torch #interchange node/edge encoders (Geo: continuous xy linear, Attr: discrete lanes+speed Embedding)
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder


@register_node_encoder('InterchangeNode')
class InterchangeNodeEncoder(nn.Module):
    """Linear encoding of continuous node features (2-dim xy coordinates)"""
    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = nn.Linear(cfg.dataset.get('node_encoder_num_types', 2), emb_dim)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x.float())
        x.x = self.encoder(x.x.float())
        return x


@register_node_encoder('InterchangeAttr')
class InterchangeAttrNodeEncoder(nn.Module):
    """Discrete attribute node encoding: Embedding(lanes) + Embedding(speed) → emb_dim, input x=[N,2]=[lanes_cls, speed_cls]"""
    def __init__(self, emb_dim):
        super().__init__()
        n_lanes = cfg.dataset.get('num_lanes_classes', 4)
        n_speed = cfg.dataset.get('num_speed_classes', 10)
        self.lanes_emb = nn.Embedding(n_lanes, emb_dim)
        self.speed_emb = nn.Embedding(n_speed, emb_dim)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.lanes_emb(x[:, 0].long()) + self.speed_emb(x[:, 1].long())
        x.x = self.lanes_emb(x.x[:, 0].long()) + self.speed_emb(x.x[:, 1].long())
        return x


@register_node_encoder('InterchangeUnified')
class InterchangeUnifiedNodeEncoder(nn.Module):
    """Unified split-path node encoding: Linear(xy 2-dim→d) + Embedding(lanes 3-class→d) + Embedding(speed 3-class→d), following GeoLDM heterogeneous-feature split paths"""
    def __init__(self, emb_dim):
        super().__init__()
        self.n_lanes = cfg.dataset.get('num_lanes_classes', 3)
        self.n_speed = cfg.dataset.get('num_speed_classes', 3)
        self.xy_enc = nn.Linear(2, emb_dim)
        self.lanes_emb = nn.Embedding(self.n_lanes, emb_dim)
        self.speed_emb = nn.Embedding(self.n_speed, emb_dim)

    def _encode(self, feat): #feat=[N,8]: xy(2)+lane_oh(3)+speed_oh(3), split-path encode then sum
        return self.xy_enc(feat[:, :2].float()) + self.lanes_emb(feat[:, 2:2+self.n_lanes].argmax(dim=1)) + self.speed_emb(feat[:, 2+self.n_lanes:].argmax(dim=1))

    def forward(self, x):
        if isinstance(x, torch.Tensor): return self._encode(x)
        x.x = self._encode(x.x)
        return x


@register_node_encoder('InterchangeAffinity')
class InterchangeAffinityNodeEncoder(nn.Module):
    """Affinity node encoding: linear mapping of continuous N×K affinity values→emb_dim"""
    def __init__(self, emb_dim):
        super().__init__()
        k = cfg.diffusion.get('affinity_k', cfg.dataset.get('affinity_k', 10))
        self.encoder = nn.Linear(k, emb_dim)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x.float())
        x.x = self.encoder(x.x.float())
        return x


@register_edge_encoder('InterchangeEdge')
class InterchangeEdgeEncoder(nn.Module):
    """Embedding encoding of binary edge features (0/1 adjacency)"""
    def __init__(self, emb_dim):
        super().__init__()
        num_types = cfg.dataset.get('edge_encoder_num_types', 2)
        if cfg.dataset.get('add_virtual_node_edge', False):
            num_types += 10
        self.encoder = nn.Embedding(num_types, emb_dim, padding_idx=0)

    def forward(self, edge_attr):
        if isinstance(edge_attr, torch.Tensor):
            return self.encoder(edge_attr.long().squeeze(-1) if edge_attr.dim() > 1 else edge_attr.long())
        edge_attr.edge_attr = self.encoder(edge_attr.edge_attr.long().squeeze(-1) if edge_attr.edge_attr.dim() > 1 else edge_attr.edge_attr.long())
        return edge_attr
