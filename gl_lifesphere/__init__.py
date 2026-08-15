"""Graph learning on the LifeSphere knowledge graph, with survival as the target.

The package is laid out along the study's three comparison models (clinical
baseline / flattened-tabular representation / graph constructions), which share
a single target formulation, loss, and metric so the models stay comparable:

    extract        Read-only pulls from the live Neo4j graph (Cypher in one place).
    features       Encoding + missing-data strategies shared by every model.
    constructions  Graph construction builders (the multi-construction part).
    models         One subpackage per model: baseline, tabular, graph.
    survival       Censored-target construction, losses, and survival metrics.
    evaluation     Splitting protocol and cross-model comparison reporting.

See `CONTEXT.md` for domain vocabulary and `docs/research/` for the methods
these modules implement.
"""
