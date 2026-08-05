"""Graph learning on the LifeSphere knowledge graph, with survival as the target.

The package is laid out along the study's three comparison arms (clinical
baseline / flattened-tabular representation / graph constructions), which share
a single target formulation, loss, and metric so the arms stay comparable:

    extract        Read-only pulls from the live Neo4j graph (Cypher in one place).
    features       Encoding + missing-data strategies shared by every arm.
    constructions  Graph construction builders (the multi-construction part).
    models         One subpackage per arm: baseline, tabular, graph.
    survival       Censored-target construction, losses, and survival metrics.
    evaluation     Splitting protocol and cross-arm comparison reporting.

See `CONTEXT.md` for domain vocabulary and `docs/research/` for the methods
these modules implement.
"""
