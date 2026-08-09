"""Graph construction builders — one module per construction.

A graph construction is a specific transformation/subset of the KG built to test
one model or hypothesis. This project is explicitly multi-construction: no
single shape is "the" LifeSphere graph for modelling purposes, so constructions
are peers here rather than versions of one another.

("Construction" rather than "variant": in this project variant already means the
genomic kind — per-loci Variant / MutationCall.)

Planned/known constructions:

- `subject_subgraph`  One rooted PyG `HeteroData` per Subject (graph-level
  regression). Buildable on the clinical-only schema; thin until omics lands.
  `cache` is its build-and-persist wrapper — the builder stays a pure function
  of its inputs, and everything about slicing, ordering and invalidation lives
  beside it rather than inside it.
- `similarity`        One cohort- or Study-level patient similarity network,
  Subjects as nodes (node-level regression).
- `hetero`            The full multi-label schema as `HeteroData`, node types
  preserved. Worth building once the omics layer adds real heterogeneity.

Built constructions are cached to `data/processed/<construction>/`.
"""
