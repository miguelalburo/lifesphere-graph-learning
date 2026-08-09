"""Building every cohort Subject's rooted subgraph once per fold, and caching it.

#11 deliberately left this out: a prototype should not persist, and the cache is
training infrastructure rather than part of the question "is the construction
correct". So this module is the wrapper #12 adds *around* `subject_subgraph`
without touching it — `build_subgraph` stays a pure function of its inputs, and
everything about slicing, ordering, persistence and invalidation lives here.

**The cache is per fold, because the features are.** The feature contract is
fitted on one fold's training Subjects alone (#4 §3), so the same Subject's
subgraph carries different `x` in fold 0 and fold 3 while its *structure* is
identical. Caching the structure once and re-applying features per fold would be
the smaller artefact, but it would also put a second feature-application path
next to `build_subgraph`'s — and the one thing this project cannot afford is two
routes to a design matrix that are supposed to agree. Rebuilding the whole graph
per fold keeps exactly one.

**Invalidation is by fingerprint, not by mtime.** The key hashes the Subject
set, the build options, the fitted contract in full (every vocabulary, median
and standardisation constant, so a refit on different Subjects misses), and the
source of `subject_subgraph.py` itself (so editing the builder misses). A stale
read is the failure mode that matters: it would silently train on features from
a previous fold's contract, a leak that produces a plausible number rather than
an exception.

The contract is rendered by `_canonical_json` rather than `repr`, because the
opposite failure is just as quiet. `FeatureContract` carries a `frozenset`, and
set iteration order depends on Python's per-process hash randomisation — so a
`repr`-based key changes between runs of identical code on identical data, and
the cache then misses every single time while still behaving like a cache.
Nothing fails and the numbers stay right; the only symptom is that every run
pays the full rebuild.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..extract.store import INTERIM_DIR, PROCESSED_DIR, read_table
from ..features.contract import FeatureContract
from . import subject_subgraph as ss

CONSTRUCTION_DIR = PROCESSED_DIR / "subject_subgraph"


@dataclass(frozen=True)
class BuildOptions:
    """The three switches `build_subgraph` takes, carried as one value.

    Grouped rather than passed individually because they are also part of the
    cache key: an option that changes the graph and does not change the
    fingerprint is exactly the stale-read bug this module exists to prevent.
    """

    split_primary_relation: bool = True
    add_reverse_edges: bool = True
    dedupe_conditions: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "split_primary_relation": self.split_primary_relation,
            "add_reverse_edges": self.add_reverse_edges,
            "dedupe_conditions": self.dedupe_conditions,
        }


# ---------------------------------------------------------------------------
# Slicing the interim tables into per-Subject records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRecords:
    """Every cohort Subject's slice of the interim tables, grouped once.

    `SubjectRecord`'s docstring makes slicing the caller's job so that building
    6,811 subgraphs groups the tables once rather than filtering them 6,811
    times. This is that caller: one `groupby` per table at load, then a dict
    lookup per Subject.
    """

    subject_ids: tuple[str, ...]
    subjects: dict[str, pd.Series]
    diagnoses: dict[str, pd.DataFrame]
    samples: dict[str, pd.DataFrame]
    pathology: dict[str, pd.DataFrame]
    empty_diagnoses: pd.DataFrame
    empty_samples: pd.DataFrame
    empty_pathology: pd.DataFrame

    def record(self, subject_id: str) -> ss.SubjectRecord:
        """One Subject's inputs, with an empty — not absent — frame where a
        table has no row for them.

        The empty frames are zero-row slices of the real tables, so they keep
        the column set and dtypes. A Subject with no Diagnosis then flows
        through the same code as one with four, rather than needing a branch.
        """
        subject = self.subjects[subject_id]
        return ss.SubjectRecord(
            subject_id=subject_id,
            study_id=str(subject["studyId"]),
            subject=subject,
            diagnoses=self.diagnoses.get(subject_id, self.empty_diagnoses),
            samples=self.samples.get(subject_id, self.empty_samples),
            pathology=self.pathology.get(subject_id, self.empty_pathology),
        )

    def roster_digest(self) -> str:
        """A hash of the Subject roster alone — one input to `fingerprint`, not it."""
        return _digest("|".join(self.subject_ids))


def build_subject_records(
    *,
    members: pd.DataFrame,
    subjects: pd.DataFrame,
    diagnoses: pd.DataFrame,
    samples: pd.DataFrame,
    pathology: pd.DataFrame,
) -> SubjectRecords:
    """Group already-loaded tables by Subject. Pure, so tests skip disk entirely.

    Every table is restricted to the cohort roster first. `members` is the
    roster and the other tables are joined *to* it, never the other way round —
    one cohort Subject has no Diagnosis at all (#4 §8), and driving the
    iteration from `diagnoses` would silently drop them and desynchronise the
    Subject set every arm shares (`features.raw`'s docstring makes the same
    point for the flattened arm).
    """
    roster = [str(subject) for subject in sorted(members["subjectId"])]
    in_cohort = set(roster)

    subject_rows = subjects[subjects["subjectId"].isin(in_cohort)]
    missing = in_cohort - set(subject_rows["subjectId"])
    if missing:
        raise ValueError(
            f"{len(missing)} cohort Subject(s) have no row in the subjects table, "
            f"e.g. {sorted(missing)[:5]}"
        )

    return SubjectRecords(
        subject_ids=tuple(roster),
        subjects={str(row["subjectId"]): row for _, row in subject_rows.iterrows()},
        diagnoses=_group(diagnoses, in_cohort),
        samples=_group(samples, in_cohort),
        pathology=_group(pathology, in_cohort),
        empty_diagnoses=diagnoses.iloc[:0],
        empty_samples=samples.iloc[:0],
        empty_pathology=pathology.iloc[:0],
    )


def _group(frame: pd.DataFrame, in_cohort: set[str]) -> dict[str, pd.DataFrame]:
    restricted = frame[frame["subjectId"].isin(in_cohort)]
    return {str(key): group for key, group in restricted.groupby("subjectId")}


def load_subject_records(
    *, interim: Path | None = None, cohort: Path | None = None
) -> SubjectRecords:
    """`build_subject_records`, reading from `data/interim/` and `data/processed/`."""
    interim_dir = interim if interim is not None else INTERIM_DIR
    cohort_dir = cohort if cohort is not None else PROCESSED_DIR / "cohort_os"

    return build_subject_records(
        members=read_table("members", directory=cohort_dir),
        subjects=read_table("subjects", directory=interim_dir),
        diagnoses=read_table("diagnoses", directory=interim_dir),
        samples=read_table("samples", directory=interim_dir),
        pathology=read_table("pathology_details", directory=interim_dir),
    )


# ---------------------------------------------------------------------------
# Building and caching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphSet:
    """One subgraph per cohort Subject, in a fixed Subject order.

    Order is `subject_ids` and nothing else. A design matrix and a
    `SurvivalTarget` are zipped by `subjectId` rather than by position
    (`SurvivalTarget.reorder`), and `select` returns the ids alongside the
    graphs so the caller can do exactly that instead of assuming the two agree.
    """

    subject_ids: tuple[str, ...]
    graphs: tuple[HeteroData, ...]
    fingerprint: str

    def __len__(self) -> int:
        return len(self.graphs)

    def select(self, subject_ids: frozenset[str]) -> tuple[list[HeteroData], list[str]]:
        """The graphs for `subject_ids`, in this set's order, with their ids."""
        missing = subject_ids - set(self.subject_ids)
        if missing:
            raise KeyError(
                f"{len(missing)} requested Subject(s) are not in this construction, "
                f"e.g. {sorted(missing)[:5]}"
            )
        chosen = [
            (graph, subject)
            for graph, subject in zip(self.graphs, self.subject_ids)
            if subject in subject_ids
        ]
        return [graph for graph, _ in chosen], [subject for _, subject in chosen]


def build_subgraphs(
    records: SubjectRecords,
    contract: FeatureContract,
    *,
    options: BuildOptions = BuildOptions(),
) -> SubgraphSet:
    """Every cohort Subject's subgraph under one fold's fitted contract."""
    graphs = tuple(
        ss.build_subgraph(
            records.record(subject_id),
            contract,
            split_primary_relation=options.split_primary_relation,
            add_reverse_edges=options.add_reverse_edges,
            dedupe_conditions=options.dedupe_conditions,
        )
        for subject_id in records.subject_ids
    )
    return SubgraphSet(
        subject_ids=records.subject_ids,
        graphs=graphs,
        fingerprint=fingerprint(records, contract, options),
    )


def without_edges(subgraphs: SubgraphSet) -> SubgraphSet:
    """The same construction with every `edge_index` emptied — #13's structure ablation.

    **Edge types survive; only their contents go.** The encoder sizes itself
    from `reference.edge_types`, so dropping a type would drop its `W_r` and
    change the parameter count — and the ablation would then be measuring
    "fewer parameters" as much as "no message passing", which is precisely the
    confound encoder doc §7 designed this control to avoid. Every relation-sum
    instead evaluates over an empty neighbourhood and contributes exactly `0`
    (the convolutions are bias-free, §2.4), so the root's bare-identity residual
    is the only path left to the readout and `z_G` is a function of the
    Subject's own demographics alone.

    Applied to an already-built set rather than expressed as a `BuildOptions`
    flag, and that is the point: the ablated run reads the *same cache entry*
    as the real one, so the two provably differ in their edges and in nothing
    else — not in the fitted contract, not in a vocabulary, not in a rebuild.

    The fingerprint is prefixed rather than recomputed so an ablated set can
    never be mistaken for, or written over, the construction it came from.
    """
    stripped = []
    for graph in subgraphs.graphs:
        ablated = graph.clone()
        for edge_type in ablated.edge_types:
            ablated[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        stripped.append(ablated)
    return SubgraphSet(
        subject_ids=subgraphs.subject_ids,
        graphs=tuple(stripped),
        fingerprint=f"edges-ablated:{subgraphs.fingerprint}",
    )


def fingerprint(
    records: SubjectRecords, contract: FeatureContract, options: BuildOptions
) -> str:
    """A hash over everything a cached build depends on.

    Four inputs, each covering a different way a cache entry can go stale:

    - the Subject roster, so a re-extract that moves the cohort misses;
    - the build options, so a flag that changes the graph changes the key;
    - the **fitted** contract's `repr`, which carries every vocabulary, median
      and standardisation constant — so refitting on a different fold's training
      Subjects misses even though the column *names* are unchanged. This is the
      one that matters: reusing fold 0's features for fold 3 is a leak that
      produces a plausible C-index rather than an error;
    - the source of `subject_subgraph.py`, so editing the builder misses. A
      construction cache that survives a change to the construction is worse
      than no cache at all.
    """
    builder_source = Path(ss.__file__).read_bytes()
    return _digest(
        "\n".join(
            [
                records.roster_digest(),
                _canonical_json(options.as_dict()),
                _canonical_json(contract),
                hashlib.sha256(builder_source).hexdigest(),
            ]
        )
    )


def _canonical_json(value: object) -> str:
    """A stable string for any fitted-encoder tree, safe to hash across processes.

    **Not `repr`, and that distinction is load-bearing.** `FeatureContract`
    carries a `frozenset` (race's fixed fold), and the iteration order of a set
    of strings depends on Python's per-process hash randomisation — so
    `repr(contract)` differs between runs of the same code on the same data.
    A fingerprint built from it changes on most restarts, and the cache then
    misses every time while still *looking* like a working cache: nothing fails,
    the numbers stay correct, and the only symptom is that every run silently
    pays the full rebuild. Sets are sorted here, and floats go through `repr` so
    they round-trip exactly rather than through JSON's shortest representation.
    """
    return json.dumps(_canonical(value), sort_keys=True)


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (frozenset, set)):
        # Canonicalise first, then sort the *rendered* elements, so a set of
        # mixed or non-comparable types still has one stable order.
        return sorted(json.dumps(_canonical(item), sort_keys=True) for item in value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return repr(value)


def cache_path(fold: int, *, directory: Path | None = None) -> Path:
    base = directory if directory is not None else CONSTRUCTION_DIR
    return base / f"fold_{fold}.pt"


def load_or_build(
    fold: int,
    records: SubjectRecords,
    contract: FeatureContract,
    *,
    options: BuildOptions = BuildOptions(),
    directory: Path | None = None,
    use_cache: bool = True,
) -> SubgraphSet:
    """The fold's subgraphs, from `data/processed/subject_subgraph/` when valid.

    A cache entry whose fingerprint does not match is rebuilt and overwritten
    rather than raising: the entry is not corrupt, it is simply for a different
    build, and the only correct response is the current one.
    """
    target = cache_path(fold, directory=directory)
    expected = fingerprint(records, contract, options)

    if use_cache and target.exists():
        cached = _read(target)
        if cached is not None and cached.fingerprint == expected:
            return cached

    built = build_subgraphs(records, contract, options=options)
    if use_cache:
        _write(target, built)
    return built


def _read(target: Path) -> SubgraphSet | None:
    """Read a cache entry, treating an unreadable one as a miss.

    `weights_only=False` is required — the payload is `HeteroData` objects, not
    a state dict — which is also why an unpickling failure is caught and turned
    into a rebuild rather than propagated: a half-written file from an
    interrupted run should cost time, not a session.
    """
    try:
        payload = torch.load(target, weights_only=False)
    except Exception:  # noqa: BLE001 - any failure to read is a cache miss
        return None
    if not isinstance(payload, SubgraphSet):
        return None
    return payload


def _write(target: Path, subgraphs: SubgraphSet) -> None:
    """Write via a temporary file, so an interrupted write leaves no valid-looking entry."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".pt.partial")
    torch.save(subgraphs, staging)
    staging.replace(target)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
