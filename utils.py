import torch
import random

from torch_geometric.graphgym.config import cfg


def random_mask(data, proportion_node=0.1, proportion_edge=0.1):
    N = data.num_nodes
    m = data.edge_index.shape[1]
    masked_node_idx = random.sample(range(N), int(N * proportion_node))
    masked_edge_idx = random.sample(range(m), int(m * proportion_edge))
    data.x_unmasked = data.x
    data.edge_attr_unmasked = data.edge_attr
    data.x[masked_node_idx] = torch.ones([int(N * proportion_node), 1], dtype=data.x.dtype, device=data.x.device) * (cfg.dataset.node_encoder_num_types + 2)
    data.edge_attr[masked_edge_idx] = torch.ones([int(m * proportion_edge), 1], dtype=data.edge_attr.dtype, device=data.x.device) * (cfg.dataset.edge_encoder_num_types + 1)
    return data, masked_node_idx, masked_edge_idx
