"""Read-only extraction from the live LifeSphere Neo4j graph.

Every Cypher query the project runs belongs here, so that schema quirks are
handled once rather than per-experiment. Three stages, each reading the
previous stage's files rather than the graph:

    raw      Verbatim Cypher pulls          -> data/raw/
    cast     Explicit typed casts           -> data/interim/
    cohort   The frozen OS eligibility rule -> data/processed/cohort_os/

Run it with `python -m gl_lifesphere.extract`.

**The graph stores everything as strings.** Every property on every node is
`STRING NOT NULL`, so the casts are the load-bearing part of this package and
they all live in `cast.py`. The specific traps, each verified live on
2026-08-08 and each covered by a test in `tests/test_extract.py`:

- All five `:Survival` properties are strings, `timeToEventDays` and
  `eventOccurred` included. `astype(bool)` maps the string `"0"` to `True`, and
  a string `timeToEventDays` sorts `"1000"` before `"999"`.
- The endpoint property is **`survivalType`**, not `endpoint`. Querying
  `endpoint` returns null for all 37,063 records and reads as "no records
  match" rather than raising.
- `isPrimaryDiagnosis` — the schema's only edge property — holds the strings
  `'True'`/`'False'`. A Cypher `= true` comparison matches **zero** rows
  silently.
- `Subject.ageAtIndexDays` holds **years** despite its name (range 14–89, with
  `-1` as a missing sentinel); `Diagnosis.ageAtDiagnosisDays` genuinely holds
  days. The pull renames the first to `ageAtIndexYears` so no reader can trust
  the property name.
- Missing `race`/`ethnicity` arrive as **true nulls today** and were recorded as
  the literal string `"None"` when #4 was written. Both representations are
  handled, because the load has demonstrably been normalised at least once.
- ~40% of Diagnoses have no `OF_CONDITION` link, but that is entirely an
  artifact of secondary diagnoses: all 6,808 cohort primary Diagnoses carry a
  Condition and all 4,867 secondary ones carry none.

The `lifesphere` database is clinical-only; the omics layer currently lives
disconnected in the `old` database. Anything reading omics must say which
database it targets — `graph_session` always passes the database explicitly.

Connection details come from the gitignored repo-root `.env`
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`).
"""

from __future__ import annotations

from .cohort import CohortCounts, CohortMismatchError, assert_locked, freeze, select_os_cohort
from .connection import Neo4jSettings, graph_session, settings_from_env
from .cypher import BASELINE_SAMPLE_TYPES, COHORT_STUDIES, POST_BASELINE_SAMPLE_TYPES
from .guards import LeakageError, check_feature_frame
from .pipeline import ExtractResult, build_interim, load_cohort_labels, run
from .store import read_raw, read_table

__all__ = [
    "BASELINE_SAMPLE_TYPES",
    "COHORT_STUDIES",
    "POST_BASELINE_SAMPLE_TYPES",
    "CohortCounts",
    "CohortMismatchError",
    "ExtractResult",
    "LeakageError",
    "Neo4jSettings",
    "assert_locked",
    "build_interim",
    "check_feature_frame",
    "freeze",
    "graph_session",
    "load_cohort_labels",
    "read_raw",
    "read_table",
    "run",
    "select_os_cohort",
    "settings_from_env",
]
