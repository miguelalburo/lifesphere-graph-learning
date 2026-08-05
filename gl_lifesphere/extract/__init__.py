"""Read-only extraction from the live LifeSphere Neo4j graph.

Every Cypher query the project runs belongs here, so that schema quirks are
handled once rather than per-experiment — in particular:

- `Survival.timeToEventDays` / `eventOccurred` are stored as strings and must be
  cast explicitly (string `"9"` sorts before `"10"`).
- ~40% of Diagnosis nodes have no `OF_CONDITION` link.
- The `lifesphere` database is clinical-only; the omics layer currently lives
  disconnected in the `old` database. Anything reading omics must say which
  database it targets.

Connection details come from the gitignored repo-root `.env`
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`).

Outputs land in `data/raw/` (verbatim pulls) and `data/interim/` (typed,
cast, joined frames) so downstream arms never re-query the live instance.
"""
