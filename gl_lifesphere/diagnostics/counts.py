"""Per-Subject accrual counts, and the one pull in the project that exists to leak.

The degree-only probe of encoder doc §7 needs `n_samples`, `n_diagnoses` and
`n_interventions`. Two of those are a `groupby` over `data/interim/`. The third
is not in the extract at all: `extract.cypher.DROPPED_NODE_TYPES` records that
`Intervention` is never pulled, because #4 §4 measured its bare presence as an
immortal-time leak (93.5% of Subjects, HR 0.716, p = 8.5e-05) — and closes with
"a ticket that needs them adds a query here". #13 is that ticket, and this is
that query.

**This table is a control's input and must never become a feature.** Three
things keep that true, and none of them is a convention:

- it is pulled by this module rather than by `extract.raw.pull_all`, so it is
  absent from `PULLS`, from `CASTS`, and from the pipeline that builds the
  tables the arms read;
- it is written to `data/interim/diagnostics/` rather than `data/interim/`, so
  neither `features.raw.assemble_raw_frame` nor `constructions.cache` can reach
  it — both name the tables they load;
- `nInterventions` is already in `extract.guards.IMMORTAL_TIME_DERIVED`, so if
  it ever did reach a design matrix, `check_feature_frame` raises. The probe's
  own frame therefore fails the project's own leakage guard **by design**, and
  `tests/test_diagnostics.py` pins exactly that.

**The counts are `count(DISTINCT ...)`, not edge counts.** The Intervention
layer is complete-bipartite within a Subject — every one of their Samples links
to every one of their Interventions, which is how 55,832 nodes carry 821,184
edges — so counting relationships would multiply a Subject's Interventions by
their Samples and measure the product of two of the probe's own covariates.

This is also the only Cypher in the project that aggregates. `cypher.py`'s "no
query casts" rule exists so `data/raw/` is a verbatim copy of the graph; pulling
821,184 edge rows to count them in Python would honour the letter of that at a
cost out of all proportion to a diagnostic, so the aggregate is done in the
query and the table is kept out of `data/raw/` entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..extract.connection import Neo4jSettings, graph_session, run
from ..extract.cypher import COHORT_STUDIES
from ..extract.store import INTERIM_DIR, PROCESSED_DIR, read_table, write_table

# The accrual counts themselves, spelled as the graph would spell them so that
# `guards.IMMORTAL_TIME_DERIVED` recognises `nInterventions` on sight.
COUNT_COLUMNS: tuple[str, ...] = ("nSamples", "nDiagnoses", "nInterventions")

# What the Cox model is actually fit on. `log1p` rather than the raw count is
# encoder doc §7's own specification: accrual is heavily right-skewed (a handful
# of Subjects carry an order of magnitude more Samples than the median), and an
# untransformed count would let those few dominate a three-covariate fit.
PROBE_COLUMNS: tuple[str, ...] = tuple(f"log1p_{column}" for column in COUNT_COLUMNS)

# Not one of §7's three, and never part of the headline probe. This is the one
# accrual channel arm 3 can actually see: mean aggregation hides degree
# (encoder doc §2.3), but #11 split `HAS_DIAGNOSIS` on `isPrimaryDiagnosis`, and
# an empty relation contributes exactly 0 while a non-empty one does not — so
# "this Subject has at least one secondary Diagnosis" is structurally visible to
# the encoder. Fitting it alone turns the gate's claim about arm 3's exposure
# into a measurement instead of an assertion.
SECONDARY_DIAGNOSIS = "has_secondary_diagnosis"
DERIVED_COLUMNS: tuple[str, ...] = (SECONDARY_DIAGNOSIS,)

DIAGNOSTICS_DIR = INTERIM_DIR / "diagnostics"
INTERVENTION_COUNTS = "intervention_counts"

# One row per Subject in the pinned Studies, including those with no
# Intervention at all: the `OPTIONAL MATCH` plus `count(DISTINCT ...)` gives
# them 0 rather than dropping them. Driving from `Study -> Subject` rather than
# from `Intervention` is the same rule `features.raw` follows — join *to* the
# roster, never from the table being counted, or Subjects with none of the thing
# silently leave the cohort.
INTERVENTION_COUNTS_QUERY = """
MATCH (study:Study)-[:HAS_SUBJECT]->(subject:Subject)
WHERE study.studyId IN $studies
OPTIONAL MATCH (subject)-[:PROVIDED_SAMPLE]->(:Sample)-[:UNDERWENT_INTERVENTION]->(i:Intervention)
RETURN subject.subjectId          AS subjectId,
       count(DISTINCT i)          AS nInterventions
ORDER BY subjectId
"""


@dataclass(frozen=True)
class AccrualCounts:
    """One row per cohort Subject, indexed by `subjectId`, holding `COUNT_COLUMNS`."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    def design(
        self,
        subject_ids: frozenset[str] | None = None,
        columns: tuple[str, ...] = PROBE_COLUMNS,
    ) -> pd.DataFrame:
        """The probe's design matrix, `subjectId`-indexed.

        Every `PROBE_COLUMNS` entry is `log(1 + count)`; `SECONDARY_DIAGNOSIS` is
        the 0/1 indicator `nDiagnoses > 1` and is the only column here that is
        not a log-count.

        No standardisation and no imputation. There is nothing to impute — a
        Subject with no Samples has 0, which is a count rather than a missing
        value — and a three-covariate Cox fit is scale-invariant in its ranking,
        so leaving the columns raw keeps each coefficient readable as a hazard
        ratio per unit of `log(1 + count)`.

        `columns` narrows the matrix so the probe can be decomposed covariate by
        covariate. A failing probe is only actionable once you know *which*
        count carries it: a protective hazard ratio is immortal time, while a
        harmful one is disease burden, and those two have opposite implications
        for whether an arm that could see the count is compromised.
        """
        unknown = set(columns) - set(PROBE_COLUMNS) - set(DERIVED_COLUMNS)
        if unknown:
            raise KeyError(f"not probe columns: {sorted(unknown)}")
        frame = self.frame if subject_ids is None else self.frame.loc[sorted(subject_ids)]
        lookup = dict(zip(PROBE_COLUMNS, COUNT_COLUMNS, strict=True))
        return pd.DataFrame(
            {
                column: (
                    (frame["nDiagnoses"].to_numpy() > 1).astype("float64")
                    if column == SECONDARY_DIAGNOSIS
                    else np.log1p(frame[lookup[column]].to_numpy(dtype="float64"))
                )
                for column in columns
            },
            index=frame.index,
        )


def build_accrual_counts(
    *,
    members: pd.DataFrame,
    samples: pd.DataFrame,
    diagnoses: pd.DataFrame,
    intervention_counts: pd.DataFrame,
) -> AccrualCounts:
    """Assemble the three counts from already-loaded tables. Pure, so tests skip disk.

    Every Subject on the roster gets a row and a Subject absent from a table
    gets 0 — `members` is the cohort and the counts are joined to it, never the
    other way round. One cohort Subject has no Diagnosis at all (#4 §8), and
    that Subject having `n_diagnoses = 0` is a fact the probe should see rather
    than a row it should lose.
    """
    roster = pd.Index(sorted(str(subject) for subject in members["subjectId"]), name="subjectId")
    frame = pd.DataFrame(index=roster)
    frame["nSamples"] = _counts(samples, roster)
    frame["nDiagnoses"] = _counts(diagnoses, roster)
    frame["nInterventions"] = (
        intervention_counts.assign(subjectId=lambda f: f["subjectId"].astype(str))
        .set_index("subjectId")["nInterventions"]
        .reindex(roster)
        .fillna(0)
        .astype("int64")
    )
    return AccrualCounts(frame=frame)


def _counts(table: pd.DataFrame, roster: pd.Index) -> pd.Series:
    grouped = table.assign(subjectId=lambda f: f["subjectId"].astype(str)).groupby("subjectId").size()
    return grouped.reindex(roster).fillna(0).astype("int64")


def pull_intervention_counts(
    *, settings: Neo4jSettings | None = None, destination: Path | None = None
) -> pd.DataFrame:
    """Run the one diagnostic query against the live instance and persist the result.

    Written with `write_table` so the dtype sidecar travels with it — an
    `nInterventions` column that came back as `object` and silently `log1p`-ed
    to NaN is exactly the quiet wrong answer `extract.store` exists to prevent.
    """
    with graph_session(settings) as session:
        rows = run(session, INTERVENTION_COUNTS_QUERY, studies=list(COHORT_STUDIES))

    frame = pd.DataFrame(rows).astype({"subjectId": "string", "nInterventions": "int64"})
    write_table(
        INTERVENTION_COUNTS,
        frame,
        directory=destination if destination is not None else DIAGNOSTICS_DIR,
        meta={
            "purpose": "#13's degree-only probe (encoder doc §7). A control's input, never a feature.",
            "query": "count(DISTINCT Intervention) per Subject over the pinned cohort Studies",
            "why_not_in_PULLS": (
                "Intervention is in extract.cypher.DROPPED_NODE_TYPES -- its bare presence is an "
                "immortal-time leak (#4 §4). Kept out of data/interim/ so no arm can load it."
            ),
        },
    )
    return frame


def load_accrual_counts(
    *,
    interim: Path | None = None,
    cohort: Path | None = None,
    diagnostics: Path | None = None,
) -> AccrualCounts:
    """`build_accrual_counts`, reading from `data/interim/` and `data/interim/diagnostics/`.

    Raises a pointed error rather than a bare `FileNotFoundError` when the
    Intervention counts are absent: they need the live instance, so a run on a
    fresh checkout fails here and should say why and what to do.
    """
    interim_dir = interim if interim is not None else INTERIM_DIR
    cohort_dir = cohort if cohort is not None else PROCESSED_DIR / "cohort_os"
    diagnostics_dir = diagnostics if diagnostics is not None else DIAGNOSTICS_DIR

    if not (diagnostics_dir / f"{INTERVENTION_COUNTS}.jsonl").exists():
        raise FileNotFoundError(
            f"{diagnostics_dir / f'{INTERVENTION_COUNTS}.jsonl'} is missing. The degree probe "
            "needs Intervention counts, which the frozen extract deliberately does not carry "
            "(extract.cypher.DROPPED_NODE_TYPES). Run "
            "`python -m gl_lifesphere.diagnostics --pull-counts` against the live instance first."
        )

    return build_accrual_counts(
        members=read_table("members", directory=cohort_dir),
        samples=read_table("samples", directory=interim_dir),
        diagnoses=read_table("diagnoses", directory=interim_dir),
        intervention_counts=read_table(INTERVENTION_COUNTS, directory=diagnostics_dir),
    )
