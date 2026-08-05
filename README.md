# LifeSphere Graph Learning

Research investigation into whether [LifeSphere](#lifesphere) — a multiomics cancer knowledge graph built for query and retrieval — is also a good substrate for **graph learning (GL) / graph neural network (GNN)** methods, using cancer patient survival as the prediction target.

## Background

Knowledge graphs (KGs) are seeing increasing use in biomedical research to structure data for scientific discovery, offering schema flexibility and search capability that tabular relational databases lack. Public repositories (cBioPortal, CIVIC) and proprietary industry KGs (GSK, AstraZeneca) exist, but public ones typically integrate only clinical data or selected molecular profiles, and working with them requires Cypher fluency — limiting KGs to niche, hypothesis-driven use.

### LifeSphere

LifeSphere is a tool built by our team at the University of Birmingham to address this. It combines a multiomics knowledge graph — currently aggregating pan-cancer TCGA data, storing clinical data alongside raw omics measurements (per-loci variants, per-gene expression, per-CpG methylation beta values) — with a natural-language retrieval system (GraphRAG) that grounds responses in the graph itself.

### This investigation

LifeSphere is designed for retrieval, not modeling. This repo is a separate investigation into whether its underlying graph is also useful as a substrate for predictive modeling. Using **overall survival** (and other survival endpoints) as the prediction target:

- Train several classes of GL/GNN algorithms across **multiple forms/variants of the graph** — this is a deliberately multi-variant study, not a single fixed schema.
- Compare against the standard clinical staging/pathology-only baseline used in oncology.
- Compare against a flattened/tabular (non-graph) export of the same data, to isolate whether graph structure itself adds predictive value beyond the features.
- Check whether a model trained on LifeSphere's schema surfaces any features or biomarkers beyond known pan-cancer ones.

## Status

Early stage: data exploration and graph-variant design against the live LifeSphere Neo4j instance. No modeling code yet.

## Data

The graph data lives in a Neo4j Enterprise instance, connected to via credentials in a local (gitignored) `.env` file — not included in this repo. This repo does not redistribute LifeSphere's underlying data.

## License

TBD.
