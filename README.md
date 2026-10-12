# TransCLGD: Intelligent Generative Transportation Design with Cascaded Latent Graph Diffusion

Companion code for the paper. TransCLGD (Intelligent Generative Transportation Design with Cascaded Latent Graph Diffusion) represents interchanges as point-level attributed graphs and generates them through a **cascaded latent graph diffusion** framework with two sequential stages.

- **Topology stage**: the point-level adjacency is equivalently transformed into a segment-level coarse graph `C = (W, L)` (coarsening). A topology autoencoder builds its latent space, a property-conditioned topology denoising network generates a new coarse graph, and reverse uncoarsening recovers the point-level adjacency `A_hat`.
- **Attribute stage**: an attribute autoencoder and a property-conditioned attribute denoising network generate the point-level node attributes (node coordinates, lane configuration, design speed) jointly under the generated topology.
- **Property conditioning**: the target vector `(cost, throughput)` is embedded and injected into both denoising networks through cross-attention.
- **Inference-time guidance**: the same trained model supports three tasks without task-specific retraining — de novo generation (from Gaussian noise), mainline-constrained completion (masked latent replacement), and reference-guided upgrading (partially perturbed reference latent).

---



## 1. Environment

```bash
# Step 1: create the conda environment and install PyTorch (CUDA 11.8)
conda create -n iclgd python=3.9 -y
conda activate iclgd
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: install PyG and its extension libraries
pip install torch_geometric==2.4.0
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Step 3: install the remaining dependencies (--no-deps prevents torch from being overwritten)
pip install yacs pytorch-lightning ogb tensorboardX einops networkx scipy opt_einsum
pip install --no-deps performer-pytorch
pip install --no-deps local-attention axial-positional-embedding hyper-connections torch-einops-utils
pip install wandb torchmetrics fsspec
conda clean --all

# Step 4: verify
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

> If a later `pip install` overwrites CUDA torch with a CPU build, re-run `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118`. The environment name may be changed freely. `requirements.txt` lists the same set for a one-shot `pip install -r requirements.txt` (PyTorch/PyG wheels still need the index URLs above).



## 2. Data

The `graph_data_all/graphs.jsonl.gz` and `graph_data_4direction/graphs.jsonl.gz` in the repository are **demo data** (24 graphs each) for quickly checking that the pipeline runs. The complete dataset is `dataset.zip` (~23 MB) at the repository root. To use it, extract it at the repository root and replace the demo files of the same name:

```bash
unzip -o dataset.zip                                         # Linux / macOS
# Expand-Archive dataset.zip -DestinationPath . -Force       # Windows PowerShell
```

The paths are exactly the `graph_data_dir` values referenced by the configs in `cfg/`, so no config change is needed. After replacing the data, delete any existing `processed/` cache and `_property_bins.json` so they are rebuilt from the complete data.

### 2.1 Contents of `dataset.zip`

```
dataset.zip
├── graph_data_all/
│   └── graphs.jsonl.gz          full augmented dataset (three-leg + four-leg interchanges)
└── graph_data_4direction/
    └── graphs.jsonl.gz          four-leg interchange subset
```

Each `graphs.jsonl.gz` is a gzip-compressed, UTF-8 JSON Lines file: **one line = one interchange graph**, stored as a single JSON object. It can be read without this codebase:

```python
import gzip, json
for line in gzip.open('graph_data_4direction/graphs.jsonl.gz', 'rt', encoding='utf-8'):
    g = json.loads(line)                                         # one interchange
```



### 2.2 Record format

Each interchange is a point-level attributed graph: the road centerlines are discretized into ordered points (nodes), and consecutive points along the same road are connected by edges.

```json
{
  "id": "1",
  "cost": 22390.2,
  "throughput": 13120.0,
  "norm_params": {"diag": 6014.676358},
  "nodes": [
    {"id": 0, "x": 0.105,  "y": -0.0738, "node_type": "mainline", "num_lanes": 3, "speed": 22.22},
    {"id": 1, "x": 0.0968, "y": -0.069,  "node_type": "mainline", "num_lanes": 3, "speed": 22.22}
  ],
  "edges": [
    {"source": 0, "target": 1}
  ]
}
```

**Graph-level fields**


| Field              | Description                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | Unique interchange identifier within the file.                                                                                               |
| `cost`             | Design property 1: construction cost of the interchange, in m.                                                                               |
| `throughput`       | Design property 2: traffic throughput of the interchange, in veh/h.                                                                          |
| `norm_params.diag` | Coordinate normalization scale (the same global value for every graph in both files; also `global_diag` in `cfg/interchange-generate.yaml`). |
| `nodes`            | Point-level nodes (see below).                                                                                                               |
| `edges`            | Point-level edges (see below).                                                                                                               |


The pair `(cost, throughput)` is the target property vector used for conditioning.

**Node fields** (`nodes[i]`)


| Field       | Description                        |
| ----------- | ---------------------------------- |
| `id`        | Node index, referenced by `edges`. |
| `x`, `y`    | Normalized planar coordinates.     |
| `node_type` | `mainline` or `ramp`.              |
| `num_lanes` | Number of lanes at this point.     |
| `speed`     | Design speed in m/s.               |


**Edge fields** (`edges[j]`)


| Field              | Description                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `source`, `target` | Node `id`s of the two endpoints. Edges are treated as undirected (the loader symmetrizes the adjacency). |




### 2.3 How the code uses the data

Besides the packed file, the loader also accepts a folder layout `{id}/graph.json` with the same record format (used for generated outputs); when both exist the packed file is preferred. `meta.json` (optional global params) is not required, and `_property_bins.json` is written on the first training preprocess.

Graphs with more than `max_nodes: 200` nodes are skipped at preprocessing. The remaining graphs are split randomly into train / val / test = 80% / 10% / 10% with a fixed seed (`42`).


| Directory                | Usage                                                                                                                                                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph_data_all/`        | Full augmented dataset (three-leg + four-leg interchanges). Used for **all training**: topology/attribute autoencoders, topology/attribute diffusion models, and the property predictor. Provides the normalization statistics `y_mean/y_std`. |
| `graph_data_4direction/` | Four-leg subset. Used at **inference** to provide task A target properties, task B mainline anchors, and task C existing interchanges to upgrade; it is also the basis for the cost/throughput percentile conditions.                          |


> Normalization is shared through `_property_bins.json`: `y_mean/y_std` come from the training set (`graph_data_all`) while the cost/throughput bins come from the four-leg subset. The file is generated automatically on the first training preprocess and is **not** version-controlled, so delete any stale copy when the data changes.
>
> Datasets are auto-preprocessed and cached on first load. Training caches under its own dataset root; inference builds a separate cache (`dataset_root` in the inference config) from `graph_data_4direction`. **Clear the corresponding** `processed/` **cache and re-preprocess when switching datasets or between the two stages**, since the topology configs set `dataset.segment.enable: True` and their cache carries the extra `seg_`* coarse-graph fields that the attribute configs do not build. (The point-level cap is `max_nodes: 200` in both stages; the topology configs additionally cap the coarse graph at `dataset.segment.max_nodes: 50` segments, applied after coarsening.)
>
> Inference additionally writes `_cond_index.npz` next to the inference dataset (`meta_cache` in the inference config): a scan cache of `(id, cost, throughput, N)` that avoids re-reading every graph on later runs. It is regenerated automatically and not version-controlled.



## 3. Training

Weights are written to `results/<cfg-name>/0/ckpt/<epoch>.ckpt`. Each diffusion config references its autoencoder ckpt via `diffusion.first_stage_config`; **after the autoencoder finishes, set that field to the actual saved ckpt path.** The autoencoders are trained first and frozen during latent diffusion training. The ckpt paths shipped in the configs (`.../0/ckpt/9.ckpt`) are placeholders — no trained weights are released.

### Topology stage

```bash
# 1) pretrain the topology autoencoder (segment coarse graph: connectivity W + segment length/type L)
python pretrain.py --cfg cfg/interchange-topology-autoencoder.yaml
# 2) train the property-conditioned topology diffusion; decoded coarse graph is uncoarsened into the adjacency A_hat
python train_interchange_diffusion.py --cfg cfg/interchange-topology-diffusion.yaml
```



### Attribute stage

The attribute autoencoder and denoising network jointly model node attributes (coordinates + lanes + speed):

```bash
# 1) pretrain the attribute autoencoder
python pretrain.py --cfg cfg/interchange-attribute-autoencoder.yaml
# 2) train the property-conditioned attribute diffusion (generates node attributes under the generated topology)
python train_interchange_diffusion.py --cfg cfg/interchange-attribute-diffusion.yaml
```



### Property predictor (for evaluation)

Graph-transformer predictor trained on `graph_data_all`, used later for property matching. It regresses **throughput** only; cost is recovered from a closed-form geometric formula (`_cost_from_graph` in the evaluator), so no cost head is trained:

```bash
python pretrain.py --cfg cfg/interchange-property-predictor.yaml
```



## 4. Inference (three tasks)

All tasks share `cfg/interchange-generate.yaml` (topology/attribute config + ckpt paths, target grid, inference data source, etc.).

### Task A - Property-conditioned de novo interchange generation

Given a target property vector, the topology stage generates the coarse graph from Gaussian noise (uncoarsened to `A_hat`), then the attribute stage generates node attributes under `A_hat`:

```bash
python generate_interchange_cond.py --cfg cfg/interchange-generate.yaml \
    --targets "21000,15000;30000,18000" --n 20
```



### Task B - Mainline-constrained interchange completion

The prescribed mainline is preserved through masked latent replacement (applied to both stages), while the ramp system is generated:

```bash
python generate_interchange_complete.py --cfg cfg/interchange-generate.yaml \
    --anchor-template <gid> --targets "25000,12000" --k 20
```



### Task C - Reference-guided existing interchange upgrading

The mainline is preserved as in task B, but the attribute latent starts from a partially perturbed reference latent (default 50%) so ramps are refined locally around the reference toward the updated target:

```bash
python generate_interchange_upgrade.py --cfg cfg/interchange-generate.yaml \
    --anchor-template <gid> --perturb-frac 0.5 --k 20
```

Without `--targets` this task defaults to the reference's own properties, which is a sanity check rather than an upgrade — report property errors only for targets that actually differ from the reference.

Tasks B and C also accept `--noise-temp` (default `1.0`): the multiplier applied to the starting Gaussian noise of the ramp part, where values above 1 trade fidelity for diversity. The default reproduces standard sampling exactly.

Each task outputs training-compatible `graph.json` and a `_summary.csv`. Evaluation writes metrics JSON/CSV only (no figure export anywhere in the pipeline).

## 5. Evaluation (multi-stage feasibility)

1. **Quality checking** removes malformed graphs and abnormal node attributes (node collapse or explosion, non-finite or degenerate coordinates, illegal lane counts, no edge).
2. **Rationality checking** verifies interchange-specific connectivity and geometric consistency of what remains (no isolated node, no fragmented graph, ramps actually generated and reachable from the kept mainline, ramp chains smooth). The kept part is defined by `--generated_labels`, so the components fixed by tasks B/C are the ones checked against.
3. **Property matching** estimates the design properties of the samples that passed stages 1-2 and reports their agreement with the target vector: throughput comes from the graph-transformer predictor, cost from the closed-form geometric formula. Reported as MAE and MAPE (mean / median / p90), overall and per target.

The three stages run in that order, so the predictor only sees samples that already passed the checks. Errors are reported exactly as measured: there is no calibration, outlier filtering, sample exclusion, or success-rate style derived metric anywhere in the flow. The evaluator accepts the regression predictor only, since a classification head would compare prediction and target as bin representatives and silently hide the error inside each bin.

Stage 1-2 criteria live in `interchange_validity.py`; the full flow is orchestrated by `eval_generated_interchange_metrics.py` (which calls the predictor):

```bash
python eval_generated_interchange_metrics.py \
    --gen_dir results/generated \
    --predictor_ckpt results/interchange-property-predictor/0/ckpt/<best>.ckpt \
    --graph_data_dir graph_data_4direction
```

It writes `_metrics.csv` (one row per generated sample: the stage that rejected it, or its property errors) and `_metrics_summary.json` (per-stage pass counts with rejection reasons, overall and per-target property errors, and the thresholds used).

Predictor's own benchmark evaluation:

```bash
python eval_property_predictor.py --cfg cfg/interchange-property-predictor.yaml --split test
```

Without `--ckpt` this picks the highest-numbered epoch on disk, not the best-validation one; pass `--ckpt` explicitly to evaluate a specific checkpoint.

## 6. Repository layout

```
cfg/                                     Core configs (topology/attribute autoencoder + diffusion, predictor, inference)
lgd/                                     TransCLGD library (diffusion, graph transformer, interchange dataset, segment coarsening, etc.)
pretrain.py                              autoencoder / property-predictor pretraining entry
train_interchange_diffusion.py           latent diffusion training entry (DDP-capable)
generate_interchange_cond.py             task A inference
generate_interchange_complete.py         task B inference
generate_interchange_upgrade.py          task C inference
eval_generated_interchange_metrics.py    multi-stage feasibility evaluation
eval_property_predictor.py               predictor benchmark evaluation
interchange_validity.py                  quality / rationality criteria
utils.py                                 random node/edge masking shared by the pretraining loops
requirements.txt                         dependency list (see "Environment")
dataset.zip                              complete dataset; extract at the repo root (see "Data")
graph_data_all/  graph_data_4direction/  demo data (24 graphs each); replace with dataset.zip for the full data
```

