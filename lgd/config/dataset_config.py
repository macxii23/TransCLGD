from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # Number of discrete node / edge types for typed encoders (e.g. Interchange).
    cfg.dataset.node_encoder_num_types = 0
    cfg.dataset.edge_encoder_num_types = 0
    cfg.dataset.add_virtual_node_edge = False

    cfg.dataset.subgraph = False

    cfg.dataset.num_hop = 0
