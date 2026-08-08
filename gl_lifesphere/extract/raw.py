"""Run the Cypher pulls and land them verbatim in `data/raw/`.

This is the only module that touches the live instance. Everything downstream
reads files, so the pipeline is reproducible from `data/raw/` alone and a
re-analysis never needs the graph to be up.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .connection import Neo4jSettings, graph_session, run, settings_from_env
from .cypher import COHORT_STUDIES, PULLS
from .store import RAW_DIR, write_raw


def pull_all(
    *,
    settings: Neo4jSettings | None = None,
    studies: tuple[str, ...] = COHORT_STUDIES,
    directory: Path | None = None,
) -> dict[str, int]:
    """Run every named pull and write it to `data/raw/`. Returns row counts.

    A manifest is written beside the tables recording the instance, the Study
    list and the row counts, so a `data/raw/` directory found later can be
    identified without guessing which graph it came from.
    """
    resolved = settings if settings is not None else settings_from_env()
    target = directory if directory is not None else RAW_DIR

    counts: dict[str, int] = {}
    with graph_session(resolved) as session:
        for name, cypher in PULLS.items():
            rows = run(session, cypher, studies=list(studies))
            write_raw(name, rows, directory=target)
            counts[name] = len(rows)

    write_raw(
        "_manifest",
        [
            {
                "pulled_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "source": resolved.redacted(),
                "studies": list(studies),
                "n_studies": len(studies),
                "row_counts": counts,
            }
        ],
        directory=target,
    )
    return counts


def pull_one(
    name: str,
    *,
    settings: Neo4jSettings | None = None,
    studies: tuple[str, ...] = COHORT_STUDIES,
) -> list[dict[str, Any]]:
    """Run a single named pull and return its rows without writing them.

    Used to record the small fixtures the tests run against.
    """
    if name not in PULLS:
        raise KeyError(f"Unknown pull {name!r}. Known pulls: {sorted(PULLS)}")
    with graph_session(settings) as session:
        return run(session, PULLS[name], studies=list(studies))
