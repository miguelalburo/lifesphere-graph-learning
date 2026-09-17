# LifeSphere Graph Learning

Research investigation into whether [LifeSphere](#lifesphere), a multiomics cancer knowledge graph built for query and retrieval, is also a good substrate for **graph learning (GL) / graph neural network (GNN)** methods, using cancer patient survival as the prediction target.

For background and results summarised, watch this [presentation video](https://www.youtube.com/watch?v=SEs1uHdIaTA)

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) (or plain `venv`/`pip`).

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python --torch-backend=cpu

# optional: register a Jupyter kernel for this venv
.venv/bin/python -m ipykernel install --user --name gl-lifesphere --display-name "gl-lifesphere (.venv)"
```

`--torch-backend=cpu` is not optional on Linux: without it the resolver pulls the
~2.5GB CUDA build of torch, which cannot run on a machine with no GPU.
`tests/test_stack.py` fails if a CUDA wheel is installed. See the header of
`requirements.txt` for the plain-`pip` equivalent.

Verify the install with `.venv/bin/python -m pytest` — the suite pins the
dependency properties the three-model comparison relies on and touches nothing
outside the venv.

Add Neo4j credentials to a `.env` file at the repo root (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`) — see [Data](#data).

## Repository layout

```
gl_lifesphere/        importable package (run Python from the repo root)
  extract/            read-only Cypher pulls from the live Neo4j graph
  features/           encoding + missing-data strategies, shared by all models
  constructions/      graph construction builders (subject subgraph, similarity, hetero)
  models/
    baseline/         model 1 — clinical staging/pathology benchmark
    tabular/          model 2 — flattened one-row-per-Subject control
    graph/            model 3 — GL/GNN encoders over the constructions
  survival/           censored targets, losses, and metrics — identical across models
  evaluation/         Study-stratified splits, cross-model comparison, interpretation
experiments/configs/  one config per run (model, construction, endpoint, split, seed)
data/                 raw/ interim/ processed/ — gitignored working data
results/              metrics/ figures/ — gitignored run outputs
notebooks/            exploratory analysis
docs/                 research notes and agent docs
tests/                cast/censoring/split correctness
```

Each package directory's `__init__.py` documents what belongs in it. `pyproject.toml`
carries tool configuration only — the project is not packaged or installed, so a
notebook importing the package still needs the repo root on `sys.path`
(`sys.path.insert(0, "..")`).

## Notebooks

- `notebooks/survival_statistics_TCGA.ipynb` — connects to the live graph via the `neo4j` driver and summarizes schema shape, survival-endpoint censoring, per-Subject fan-out, Study-level imbalance, and a 5-year mortality-horizon feasibility check. Companion to `docs/research/gl-gnn-survival-methods.md`.

## License

TBD.
