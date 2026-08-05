# LifeSphere Graph Learning

Research investigation into whether [LifeSphere](#lifesphere) — a multiomics cancer knowledge graph built for query and retrieval — is also a good substrate for **graph learning (GL) / graph neural network (GNN)** methods, using cancer patient survival as the prediction target.

## Background

Knowledge graphs (KGs) are seeing increasing use in biomedical research to structure data for scientific discovery, offering schema flexibility and search capability that tabular relational databases lack. Public repositories (cBioPortal, CIVIC) and proprietary industry KGs (GSK, AstraZeneca) exist, but public ones typically integrate only clinical data or selected molecular profiles, and working with them requires Cypher fluency — limiting KGs to niche, hypothesis-driven use.

### LifeSphere

LifeSphere is a tool built by our team at the University of Birmingham to address this. It combines a multiomics knowledge graph — currently aggregating pan-cancer TCGA data, storing clinical data alongside raw omics measurements (per-loci variants, per-gene expression, per-CpG methylation beta values) — with a natural-language retrieval system (GraphRAG) that grounds responses in the graph itself.

### This investigation

LifeSphere is designed for retrieval, not modeling. This repo is a separate investigation into whether its underlying graph is also useful as a substrate for predictive modeling. Using **overall survival** (and other survival endpoints) as the prediction target:

- Train several classes of GL/GNN algorithms across **multiple constructions of the graph** — this is a deliberately multi-construction study, not a single fixed schema.
- Compare against the standard clinical staging/pathology-only baseline used in oncology.
- Compare against a flattened/tabular (non-graph) export of the same data, to isolate whether graph structure itself adds predictive value beyond the features.
- Check whether a model trained on LifeSphere's schema surfaces any features or biomarkers beyond known pan-cancer ones.

## Status

Early stage: data exploration and graph-construction design against the live LifeSphere Neo4j instance. No modeling code yet.

## Data

The graph data lives in a Neo4j Enterprise instance, connected to via credentials in a local (gitignored) `.env` file — not included in this repo. This repo does not redistribute LifeSphere's underlying data.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/) (or plain `venv`/`pip`).

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python

# optional: register a Jupyter kernel for this venv
.venv/bin/python -m ipykernel install --user --name gl-lifesphere --display-name "gl-lifesphere (.venv)"
```

Add Neo4j credentials to a `.env` file at the repo root (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`) — see [Data](#data).

## Repository layout

```
gl_lifesphere/        importable package (run Python from the repo root)
  extract/            read-only Cypher pulls from the live Neo4j graph
  features/           encoding + missing-data strategies, shared by all arms
  constructions/      graph construction builders (subject subgraph, similarity, hetero)
  models/
    baseline/         arm 1 — clinical staging/pathology benchmark
    tabular/          arm 2 — flattened one-row-per-Subject control
    graph/            arm 3 — GL/GNN encoders over the constructions
  survival/           censored targets, losses, and metrics — identical across arms
  evaluation/         Study-stratified splits, cross-arm comparison, interpretation
experiments/configs/  one config per run (arm, construction, endpoint, split, seed)
data/                 raw/ interim/ processed/ — gitignored working data
results/              metrics/ figures/ — gitignored run outputs
notebooks/            exploratory analysis
docs/                 research notes and agent docs
tests/                cast/censoring/split correctness
```

Each package directory's `__init__.py` documents what belongs in it. There is no
build config yet, so a notebook importing the package needs the repo root on
`sys.path` (`sys.path.insert(0, "..")`).

## Notebooks

- `notebooks/survival_statistics_TCGA.ipynb` — connects to the live graph via the `neo4j` driver and summarizes schema shape, survival-endpoint censoring, per-Subject fan-out, Study-level imbalance, and a 5-year mortality-horizon feasibility check. Companion to `docs/research/gl-gnn-survival-methods.md`.

## License

TBD.
