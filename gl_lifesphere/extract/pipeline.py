"""The three stages, wired together: pull → cast → freeze.

Split into stages that each read the previous stage's files, so a cast bug is
fixed and replayed without re-querying the graph, and so `data/raw/` remains a
standing record of what the instance held on the day it was pulled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import guards
from .cast import CASTS, reduce_diagnoses
from .cohort import CohortCounts, freeze
from .connection import Neo4jSettings
from .raw import pull_all
from .store import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, read_raw, read_table, write_table


@dataclass(frozen=True)
class ExtractResult:
    """What one end-to-end run produced."""

    raw_counts: dict[str, int]
    interim_counts: dict[str, int]
    cohort: CohortCounts


def build_interim(*, raw: Path | None = None, destination: Path | None = None) -> dict[str, int]:
    """Cast every raw table and write the typed frames to `data/interim/`.

    Also writes `diagnosis_primary`, the one-row-per-Subject reduction of #4 §1,
    because every arm needs it and re-deriving the fallback rule per arm is how
    two arms end up with different stage columns.
    """
    source = raw if raw is not None else RAW_DIR
    target = destination if destination is not None else INTERIM_DIR

    frames = {}
    counts: dict[str, int] = {}
    for name, cast in CASTS.items():
        frame = cast(read_raw(name, directory=source))
        frames[name] = frame
        write_table(name, frame, directory=target, meta={"source": f"data/raw/{name}.jsonl"})
        counts[name] = len(frame)

    # Schema-drift checks against the real extract, not only against fixtures.
    guards.check_age_units(frames["subjects"], frames["diagnoses"])
    guards.check_one_primary_diagnosis(frames["diagnoses"])
    guards.check_sample_class_is_redundant(frames["samples"])
    for name, frame in frames.items():
        guards.check_missing_tokens(frame, where=f"interim/{name}")

    primary = reduce_diagnoses(frames["diagnoses"])
    write_table(
        "diagnosis_primary",
        primary,
        directory=target,
        meta={"rule": "primary Diagnosis with per-field fallback to the Subject's others (#4 §1)"},
    )
    counts["diagnosis_primary"] = len(primary)
    return counts


def run(
    *,
    settings: Neo4jSettings | None = None,
    skip_pull: bool = False,
) -> ExtractResult:
    """Run the full extract. `skip_pull` replays from an existing `data/raw/`."""
    raw_counts = (
        {name: len(read_raw(name)) for name in CASTS} if skip_pull else pull_all(settings=settings)
    )
    interim_counts = build_interim()
    _, cohort = freeze()
    return ExtractResult(raw_counts=raw_counts, interim_counts=interim_counts, cohort=cohort)


def load_cohort_labels() -> pd.DataFrame:
    """Read the frozen label table back.

    The single entry point arms use to get the Cox target, so no arm reaches
    into `data/processed/` by path and no arm re-derives eligibility.
    """
    return read_table("labels", directory=PROCESSED_DIR / "cohort_os")
