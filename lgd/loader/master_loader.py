import logging, os, os.path as osp, time #Interchange-only master loader
from functools import partial
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader
from .split_generator import prepare_splits, set_dataset_splits
from lgd.transform.posenc_stats import compute_posenc_stats
from lgd.transform.transforms import (pre_transform_in_memory, segment_rewire,
                                      no_virtual_node_edge, pretransform_pe)


def log_loaded_dataset(dataset, format, name): #log basic dataset stats
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}"); logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")
    n = dataset.data.num_nodes if hasattr(dataset.data, 'num_nodes') else (dataset.data.x.size(0) if hasattr(dataset.data, 'x') else 0)
    logging.info(f"  avg num_nodes/graph: {n // max(len(dataset), 1)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir): #load PyG-Interchange + PE / dense-edge preprocess
    if not format.startswith('PyG-'): raise ValueError(f"Unknown data format: {format}")
    pyg_dataset_id = format.split('-', 1)[1]
    dataset_dir = osp.join(dataset_dir, pyg_dataset_id)
    processed_path = dataset_dir + '/' + name + '/processed.pt'
    if os.path.exists(processed_path):
        return torch.load(processed_path, map_location="cpu")
    if pyg_dataset_id != 'Interchange':
        raise ValueError(f"Unexpected PyG Dataset identifier: {format} (open release supports PyG-Interchange only)")
    print(dataset_dir)
    dataset = preformat_Interchange(dataset_dir, name)
    if cfg.dataset.get('segment', {}).get('enable', False):
        logging.info('[Interchange] segment_rewire enabled: seg_* → main fields, raw_* backup')
        pre_transform_in_memory(dataset, segment_rewire, show_progress=True)
    log_loaded_dataset(dataset, format, name)

    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable and (not key.startswith('posenc_ER')):
            pe_name = key.split('_', 1)[1]; pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel') and pecfg.kernel.times_func:
                pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: {pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: {pe_enabled_list} for all graphs...")
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset, partial(compute_posenc_stats, pe_types=pe_enabled_list, is_undirected=is_undirected, cfg=cfg), show_progress=True)
        logging.info(f"Done! Took {time.perf_counter() - start:.2f}s")

    if cfg.dataset.add_virtual_node_edge: #virtual node/edge augmentation was molecular-only and is not part of this release
        raise NotImplementedError('dataset.add_virtual_node_edge is not supported by the interchange release, keep it False')
    if cfg.dataset.get('aug_edge', True):
        logging.info('Augmenting edges')
        pre_transform_in_memory(dataset, partial(no_virtual_node_edge, format=cfg.dataset.format), show_progress=True)
        logging.info('Preprocessing PE')
        pre_transform_in_memory(dataset, partial(pretransform_pe, add_virtual_node=False), show_progress=True)

    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs); delattr(dataset, 'split_idxs')
    prepare_splits(dataset)
    return dataset


def join_dataset_splits(datasets): #merge train/val/test into one dataset with split_idxs
    assert len(datasets) == 3, "Expecting train, val, test loader"
    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + [datasets[1].get(i) for i in range(n2)] + [datasets[2].get(i) for i in range(n3)]
    datasets[0]._indices = None; datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs = [list(range(n1)), list(range(n1, n1 + n2)), list(range(n1 + n2, n1 + n2 + n3))]
    return datasets[0]


def preformat_Interchange(dataset_dir, name): #load interchange train/val/test and merge
    from .dataset.interchange import InterchangeDataset
    return join_dataset_splits([InterchangeDataset(stage='train', root=dataset_dir),
                                InterchangeDataset(stage='val', root=dataset_dir),
                                InterchangeDataset(stage='test', root=dataset_dir)])
