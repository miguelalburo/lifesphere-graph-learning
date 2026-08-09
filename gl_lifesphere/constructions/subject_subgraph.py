"""One rooted, heterogeneous `HeteroData` per Subject — the raw graph construction.

**The question this answers (#11):** does the schema-faithful rooted subgraph,
once built, actually look like what we think it looks like? The construction is
here; the assertions that answer the question are here too (`check_subgraph`),
because "assert it, do not eyeball it" only works if the assertions travel with
the builder. The hand-driven inspector that pushes both through deliberately
varied Subjects is the throwaway shell in `prototype_subgraph_tui.py`.

**Shape, after #4 reshaped it.** Five node types (Subject, Diagnosis, Condition,
Sample, PathologyDetail) and four relations (`HAS_DIAGNOSIS`, `OF_CONDITION`,
`PROVIDED_SAMPLE`, `HAS_PATHOLOGY`) — not the 7/6 the encoder doc §0 describes.
`Intervention` and `PhenotypeObservation` are excluded as node types, not merely
as features (#4 §4, §5); `Survival` is the response and `Study` is a split key,
so neither is ever a node (encoder doc §0).

**Direction is the load-bearing decision.** Every relation in the LifeSphere
schema points away from the Subject, so a subgraph built with Neo4j's directions
alone leaves the root receiving nothing and the encoder silently reduces to the
Subject's own two demographic fields (encoder doc §2.1). Reverse edges are added
by default and `messages_reaching_root` measures the property directly rather
than trusting the transform.

**Features come from the shared contract, never from a second encoding.** The
same fitted `FeatureContract` the flattened arm calls (#4 §2: "there is no
encoding difference between the arms to negotiate") is sliced into per-node-type
blocks here. What differs between the arms is *which rows* the encoders see: the
flattened arm reduces to one primary Diagnosis with per-field fallback (#4 §1),
whereas every Diagnosis becomes its own node here. That is a structural
difference, which is the only kind allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from ..extract.cypher import BASELINE_SAMPLE_TYPES, POST_BASELINE_SAMPLE_TYPES
from ..features.contract import FeatureContract
from ..features.raw import SAMPLE_PROPORTION_COLUMNS

SUBJECT = "Subject"
DIAGNOSIS = "Diagnosis"
CONDITION = "Condition"
SAMPLE = "Sample"
PATHOLOGY = "PathologyDetail"

NODE_TYPES: tuple[str, ...] = (SUBJECT, DIAGNOSIS, CONDITION, SAMPLE, PATHOLOGY)

# Never a node in this construction, each for a different reason: Survival is
# the response, Study is the split key and a cancer-type confound (encoder doc
# §0), Intervention is an immortal-time leak in its bare presence and
# PhenotypeObservation is a featureless cancer-type channel (#4 §4, §5).
EXCLUDED_NODE_TYPES: tuple[str, ...] = (
    "Survival",
    "Study",
    "Intervention",
    "PhenotypeObservation",
)

HAS_DIAGNOSIS = (SUBJECT, "HAS_DIAGNOSIS", DIAGNOSIS)
HAS_PRIMARY_DIAGNOSIS = (SUBJECT, "HAS_PRIMARY_DIAGNOSIS", DIAGNOSIS)
HAS_SECONDARY_DIAGNOSIS = (SUBJECT, "HAS_SECONDARY_DIAGNOSIS", DIAGNOSIS)
OF_CONDITION = (DIAGNOSIS, "OF_CONDITION", CONDITION)
PROVIDED_SAMPLE = (SUBJECT, "PROVIDED_SAMPLE", SAMPLE)
HAS_PATHOLOGY = (DIAGNOSIS, "HAS_PATHOLOGY", PATHOLOGY)

# PathologyDetail carries no property in the entire cohort (#4 §5:
# `necrosisPercent` is present on 0 of 13,525 rows), but a node type with a
# zero-width `x` has no `Linear(0 -> d)` to learn. One constant column is the
# honest encoding: `Linear(1 -> d)` on a constant input is a learned bias
# vector, which is exactly "an embedding of the single category `exists`".
PATHOLOGY_FEATURE_WIDTH = 1

# Encoder doc §2.2 — every node in the subgraph is within 2 hops of the root,
# which is the same number that fixes L.
SUBGRAPH_RADIUS = 2


@dataclass(frozen=True)
class SubjectRecord:
    """Everything one Subject's subgraph is built from, already sliced.

    Slicing is the caller's job so that building 6,811 subgraphs groups the
    interim tables once rather than filtering them 6,811 times, and so the
    builder stays a pure function of its inputs.
    """

    subject_id: str
    study_id: str
    subject: pd.Series
    diagnoses: pd.DataFrame
    samples: pd.DataFrame
    pathology: pd.DataFrame


@dataclass(frozen=True)
class FeatureBlocks:
    """The contract's columns, partitioned across the node types that carry them.

    The partition is the concrete meaning of "the arms share a feature
    contract": every column the flattened arm puts in one wide row appears
    exactly once somewhere in the graph, and no column appears twice or goes
    missing. `check_partitions_contract` asserts precisely that.
    """

    subject: tuple[str, ...]
    diagnosis: tuple[str, ...]
    condition: tuple[str, ...]
    sample: tuple[str, ...]
    pathology: tuple[str, ...] = ()

    @classmethod
    def from_contract(cls, contract: FeatureContract) -> "FeatureBlocks":
        return cls(
            subject=(*contract.sex.output_columns, *contract.race.output_columns),
            diagnosis=(
                *contract.stage.output_columns,
                *contract.age.output_columns,
                *contract.condition_subtype.output_columns,
            ),
            condition=tuple(contract.condition_name.output_columns),
            sample=tuple(contract.sample_proportions.output_columns),
            pathology=(),
        )

    @property
    def widths(self) -> dict[str, int]:
        return {
            SUBJECT: len(self.subject),
            DIAGNOSIS: len(self.diagnosis),
            CONDITION: len(self.condition),
            SAMPLE: len(self.sample),
            PATHOLOGY: PATHOLOGY_FEATURE_WIDTH,
        }

    def columns(self, node_type: str) -> tuple[str, ...]:
        return {
            SUBJECT: self.subject,
            DIAGNOSIS: self.diagnosis,
            CONDITION: self.condition,
            SAMPLE: self.sample,
            PATHOLOGY: self.pathology,
        }[node_type]

    def check_partitions_contract(self, contract: FeatureContract) -> None:
        """Raise unless the blocks are a true partition of `contract.feature_names`."""
        assigned = [
            *self.subject, *self.diagnosis, *self.condition, *self.sample, *self.pathology
        ]
        duplicated = sorted({c for c in assigned if assigned.count(c) > 1})
        if duplicated:
            raise AssertionError(f"columns assigned to more than one node type: {duplicated}")
        expected, got = set(contract.feature_names), set(assigned)
        if expected != got:
            raise AssertionError(
                f"blocks do not partition the contract: "
                f"missing {sorted(expected - got)}, extra {sorted(got - expected)}"
            )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _one_row(series: pd.Series) -> pd.DataFrame:
    return series.to_frame().T


def _subject_features(record: SubjectRecord, contract: FeatureContract) -> torch.Tensor:
    row = _one_row(record.subject)
    frame = pd.concat(
        [contract.sex.transform(row["sexAtBirth"]), contract.race.transform(row["race"])], axis=1
    )
    return _to_tensor(frame)


def _diagnosis_features(record: SubjectRecord, contract: FeatureContract) -> torch.Tensor:
    """Per-Diagnosis, with the contract's encoders applied to each node's own row.

    The flattened arm's per-field fallback (#4 §1) is deliberately *not* applied
    here: a secondary Diagnosis that records no stage is a node whose stage is
    genuinely absent, and borrowing the primary's value would invent a fact the
    graph does not hold. What each node does inherit is the contract's
    training-fold within-Study median for that absent field, so the imputation
    rule is shared even though the rows are not.

    Age falls back to the Subject's `ageAtIndexYears` exactly as
    `FeatureContract.transform` does, since the two properties are the same
    quantity at different precision (#4 §3).
    """
    diagnoses = record.diagnoses
    study = pd.Series([record.study_id] * len(diagnoses), index=diagnoses.index, dtype="object")

    age = diagnoses["ageAtDiagnosisYears"].astype("Float64")
    age = age.fillna(record.subject.get("ageAtIndexYears", pd.NA))

    frame = pd.concat(
        [
            contract.stage.transform(diagnoses["stageOrdinal"], study),
            contract.age.transform(age, study),
            contract.condition_subtype.transform(diagnoses["conditionSubtype"]),
        ],
        axis=1,
    )
    return _to_tensor(frame)


def _condition_nodes(record: SubjectRecord, *, dedupe: bool) -> pd.DataFrame:
    """The distinct Conditions this Subject's Diagnoses point at.

    Deduplication is on by default because a Condition is a shared vocabulary
    node in the KG — two Diagnoses of the same cancer type point at the *same*
    node, and materialising two copies would be a construction artifact rather
    than something the graph says. `OF_CONDITION` is exactly 0-or-1 per
    Diagnosis (#4 §5), so this only ever merges across Diagnoses, never within.
    """
    linked = record.diagnoses.dropna(subset=["conditionId"])
    nodes = linked[["conditionId", "conditionName"]]
    if dedupe:
        nodes = nodes.drop_duplicates(subset=["conditionId"])
    return nodes.reset_index(drop=True)


def _condition_features(nodes: pd.DataFrame, contract: FeatureContract) -> torch.Tensor:
    return _to_tensor(contract.condition_name.transform(nodes["conditionName"]))


def _sample_nodes(record: SubjectRecord) -> pd.DataFrame:
    """Baseline-type Samples only — the post-baseline types are excluded as *nodes*.

    #4 §4 is explicit that `Metastatic` and the three rarer post-baseline types
    leave the graph construction, not just the feature vector: a metastatic
    sample scores a clinically absurd protective HR because you must survive to
    be biopsied again, and leaving the node in would hand that same bit back
    through the neighbourhood.
    """
    samples = record.samples
    return samples[samples["sampleType"].isin(BASELINE_SAMPLE_TYPES)].reset_index(drop=True)


def _sample_features(nodes: pd.DataFrame, contract: FeatureContract) -> torch.Tensor:
    """One standardised `sampleType` indicator per Sample node.

    Standardised with the *proportion* encoder's own mean and std, which is what
    makes the arms provably equal on this block: standardisation is affine and
    the mean commutes with it, so the mean over a Subject's Sample nodes of
    `(onehot - mu) / sigma` is exactly `(p - mu) / sigma`, the row the flattened
    arm gets. Mean-aggregating this node type reproduces the contract's
    proportion vector to floating-point tolerance — `check_subgraph` asserts it
    rather than leaving it as an argument on paper.
    """
    indicator = pd.DataFrame(
        {
            column: (nodes["sampleType"] == sample_type).astype("float64")
            for column, sample_type in zip(SAMPLE_PROPORTION_COLUMNS, BASELINE_SAMPLE_TYPES)
        },
        index=nodes.index,
    )
    encoder = contract.sample_proportions
    standardised = {
        column: (indicator[column] - encoder.means[column]) / encoder.stds[column]
        for column in SAMPLE_PROPORTION_COLUMNS
    }
    return _to_tensor(pd.DataFrame(standardised, index=nodes.index))


def _to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    if len(frame) == 0:
        return torch.zeros((0, frame.shape[1]), dtype=torch.float32)
    return torch.as_tensor(frame.to_numpy(dtype="float32"))


def _edge_index(source: list[int], target: list[int]) -> torch.Tensor:
    if not source:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([source, target], dtype=torch.long)


def build_subgraph(
    record: SubjectRecord,
    contract: FeatureContract,
    *,
    blocks: FeatureBlocks | None = None,
    split_primary_relation: bool = False,
    add_reverse_edges: bool = True,
    dedupe_conditions: bool = True,
) -> HeteroData:
    """One rooted `HeteroData` for `record`. Node 0 of `Subject` is always the root.

    `split_primary_relation` materialises encoder doc §2.5's proposal —
    `HAS_PRIMARY_DIAGNOSIS` / `HAS_SECONDARY_DIAGNOSIS` as separate relations
    with separate weight matrices — instead of an `edge_attr`. #4 established
    that `isPrimaryDiagnosis` is the *only* edge property in the whole schema,
    so this is the only edge-attribute decision the clinical-only graph will
    ever force, and splitting removes the need for any layer that takes
    `edge_attr` at all.

    `add_reverse_edges=False` exists to be run, once, and seen to fail: it is
    the encoder doc §2.1 bug in its exact form, and the point of building it is
    that `messages_reaching_root` catches it rather than a suspicious C-index
    six weeks later.
    """
    blocks = blocks if blocks is not None else FeatureBlocks.from_contract(contract)

    diagnoses = record.diagnoses.reset_index(drop=True)
    conditions = _condition_nodes(record, dedupe=dedupe_conditions)
    samples = _sample_nodes(record)
    pathology = record.pathology.reset_index(drop=True)

    data = HeteroData()
    data[SUBJECT].x = _subject_features(record, contract)
    data[SUBJECT].subject_id = [record.subject_id]
    data[DIAGNOSIS].x = _diagnosis_features(
        SubjectRecord(
            record.subject_id, record.study_id, record.subject, diagnoses,
            record.samples, record.pathology,
        ),
        contract,
    )
    data[CONDITION].x = _condition_features(conditions, contract)
    data[SAMPLE].x = _sample_features(samples, contract)
    data[PATHOLOGY].x = torch.ones((len(pathology), PATHOLOGY_FEATURE_WIDTH), dtype=torch.float32)

    # Kept for the inspector, and for any downstream check that needs to say
    # *which* Sample or Condition a row came from.
    data[SAMPLE].sample_type = samples["sampleType"].tolist()
    data[CONDITION].condition_id = conditions["conditionId"].tolist()
    data[DIAGNOSIS].diagnosis_id = diagnoses["diagnosisId"].tolist()
    data[DIAGNOSIS].is_primary = [bool(v) if pd.notna(v) else None for v in diagnoses["isPrimaryDiagnosis"]]

    diagnosis_row = {d: i for i, d in enumerate(diagnoses["diagnosisId"])}
    condition_row = {c: i for i, c in enumerate(conditions["conditionId"])}

    if split_primary_relation:
        primary = [i for i, v in enumerate(diagnoses["isPrimaryDiagnosis"]) if v is True or v is np.True_]
        secondary = [i for i in range(len(diagnoses)) if i not in set(primary)]
        data[HAS_PRIMARY_DIAGNOSIS].edge_index = _edge_index([0] * len(primary), primary)
        data[HAS_SECONDARY_DIAGNOSIS].edge_index = _edge_index([0] * len(secondary), secondary)
    else:
        rows = list(range(len(diagnoses)))
        data[HAS_DIAGNOSIS].edge_index = _edge_index([0] * len(rows), rows)

    of_condition_src = [
        diagnosis_row[d]
        for d, c in zip(diagnoses["diagnosisId"], diagnoses["conditionId"])
        if pd.notna(c)
    ]
    of_condition_dst = [condition_row[c] for c in diagnoses["conditionId"] if pd.notna(c)]
    data[OF_CONDITION].edge_index = _edge_index(of_condition_src, of_condition_dst)

    sample_rows = list(range(len(samples)))
    data[PROVIDED_SAMPLE].edge_index = _edge_index([0] * len(sample_rows), sample_rows)

    # A PathologyDetail whose parent Diagnosis is not in this subgraph would be
    # a dangling edge; the interim table carries `subjectId` alongside
    # `diagnosisId`, so the mismatch is possible in principle and dropped
    # explicitly here rather than raising a key error.
    pathology_pairs = [
        (diagnosis_row[d], i)
        for i, d in enumerate(pathology["diagnosisId"])
        if d in diagnosis_row
    ]
    data[HAS_PATHOLOGY].edge_index = _edge_index(
        [src for src, _ in pathology_pairs], [dst for _, dst in pathology_pairs]
    )

    if add_reverse_edges:
        data = T.ToUndirected()(data)
    return data


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def messages_reaching_root(data: HeteroData, *, hops: int = SUBGRAPH_RADIUS) -> dict[str, torch.Tensor]:
    """Per node type, a mask of nodes with a directed path of `<= hops` to the root.

    This is encoder doc §2.1 measured rather than assumed. `u -> v` means `v`
    receives a message from `u`, so a node influences the readout only if some
    directed path leads from it to the root within `L` layers. Built from the
    real `edge_index_dict`, so it catches a missing reverse relation, a relation
    reversed one time too many, and a `ToUndirected()` that quietly skipped a
    type — three failures that all look identical from the outside.
    """
    reached = {t: torch.zeros(data[t].num_nodes, dtype=torch.bool) for t in data.node_types}
    reached[SUBJECT][0] = True
    for _ in range(hops):
        frontier = {t: mask.clone() for t, mask in reached.items()}
        for (src, _, dst), edge_index in data.edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            hits = reached[dst][edge_index[1]]
            frontier[src][edge_index[0][hits]] = True
        reached = frontier
    return reached


def _expected_edge_types(
    *, split_primary_relation: bool, add_reverse_edges: bool
) -> set[tuple[str, str, str]]:
    forward = {OF_CONDITION, PROVIDED_SAMPLE, HAS_PATHOLOGY}
    forward |= (
        {HAS_PRIMARY_DIAGNOSIS, HAS_SECONDARY_DIAGNOSIS}
        if split_primary_relation
        else {HAS_DIAGNOSIS}
    )
    if not add_reverse_edges:
        return forward
    return forward | {(dst, f"rev_{rel}", src) for src, rel, dst in forward}


def check_subgraph(
    data: HeteroData,
    record: SubjectRecord,
    contract: FeatureContract,
    *,
    blocks: FeatureBlocks | None = None,
    split_primary_relation: bool = False,
    add_reverse_edges: bool = True,
) -> list[Check]:
    """Every property #11 asks to be asserted rather than eyeballed."""
    blocks = blocks if blocks is not None else FeatureBlocks.from_contract(contract)
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append(Check(name, passed, detail))

    # --- shape -------------------------------------------------------------
    present = set(data.node_types)
    add(
        "node types are exactly the five #4 left standing",
        present == set(NODE_TYPES),
        f"missing {sorted(set(NODE_TYPES) - present)}, extra {sorted(present - set(NODE_TYPES))}",
    )
    leaked = present & set(EXCLUDED_NODE_TYPES)
    add("Survival, Study, Intervention, PhenotypeObservation absent", not leaked, f"present: {sorted(leaked)}")

    expected_edges = _expected_edge_types(
        split_primary_relation=split_primary_relation, add_reverse_edges=add_reverse_edges
    )
    got_edges = set(data.edge_types)
    add(
        f"relation types are the expected {len(expected_edges)}",
        got_edges == expected_edges,
        f"missing {sorted(expected_edges - got_edges)}, extra {sorted(got_edges - expected_edges)}",
    )

    # --- direction (encoder doc §2.1) --------------------------------------
    reached = messages_reaching_root(data)
    stranded = {
        t: int((~reached[t]).sum()) for t in data.node_types if int((~reached[t]).sum()) > 0
    }
    add(
        f"every node reaches the root within {SUBGRAPH_RADIUS} hops",
        not stranded,
        f"stranded: {stranded}" if stranded else "root receives from all node types present",
    )

    # --- features ----------------------------------------------------------
    try:
        blocks.check_partitions_contract(contract)
        add("per-node-type blocks partition the shared feature contract", True,
            f"{len(contract.feature_names)} contract columns, no duplicates, none dropped")
    except AssertionError as error:
        add("per-node-type blocks partition the shared feature contract", False, str(error))

    widths = blocks.widths
    wrong = {
        t: (int(data[t].x.size(1)), widths[t])
        for t in NODE_TYPES
        if t in present and int(data[t].x.size(1)) != widths[t]
    }
    add(
        "feature widths per node type match the contract's blocks",
        not wrong,
        f"got vs expected: {wrong}" if wrong else ", ".join(f"{t}={widths[t]}" for t in NODE_TYPES),
    )

    nonfinite = {
        t: int((~torch.isfinite(data[t].x)).sum()) for t in present
        if int((~torch.isfinite(data[t].x)).sum()) > 0
    }
    add("no NaN or inf in any feature matrix", not nonfinite, f"non-finite cells: {nonfinite}")

    # --- #4 §4: the post-baseline types leave as nodes, not just as features
    types_present = set(getattr(data[SAMPLE], "sample_type", []))
    banned = types_present & POST_BASELINE_SAMPLE_TYPES
    add(
        "no post-baseline sampleType reached the export",
        not banned,
        f"present: {sorted(banned)}" if banned else f"types: {sorted(types_present) or 'none'}",
    )

    # --- the arms agree on the Sample block --------------------------------
    add(*_check_sample_mean_matches_contract(data, record, contract))

    # --- arity -------------------------------------------------------------
    of_condition = data[OF_CONDITION].edge_index if OF_CONDITION in got_edges else torch.zeros((2, 0))
    per_diagnosis = np.bincount(
        of_condition[0].numpy(), minlength=data[DIAGNOSIS].num_nodes
    ) if of_condition.numel() else np.zeros(data[DIAGNOSIS].num_nodes, dtype=int)
    add(
        "OF_CONDITION is 0-or-1 per Diagnosis",
        bool((per_diagnosis <= 1).all()),
        f"max per Diagnosis: {int(per_diagnosis.max()) if len(per_diagnosis) else 0}",
    )

    return checks


def _check_sample_mean_matches_contract(
    data: HeteroData, record: SubjectRecord, contract: FeatureContract
) -> tuple[str, bool, str]:
    """Mean-aggregating the Sample nodes must reproduce the flattened arm's row.

    If this ever fails, the two arms are encoding the same biospecimens
    differently and any difference in their C-index is partly an encoding
    artifact rather than a structural result — which is the one thing the whole
    three-arm design exists to rule out.
    """
    name = "mean over Sample nodes == the contract's proportion row"
    x = data[SAMPLE].x
    baseline = record.samples[record.samples["sampleType"].isin(BASELINE_SAMPLE_TYPES)]
    if len(baseline) == 0:
        return (
            name,
            x.numel() == 0,
            "no baseline Sample: the graph aggregates an empty set to 0, the flattened arm "
            "gets the training-fold mean standardised — they differ, by design and by a "
            "measurable amount (see the inspector's [c] view)",
        )
    counts = baseline["sampleType"].value_counts()
    proportions = np.array(
        [counts.get(t, 0) / len(baseline) for t in BASELINE_SAMPLE_TYPES], dtype="float64"
    )
    encoder = contract.sample_proportions
    expected = np.array(
        [
            (proportions[i] - encoder.means[c]) / encoder.stds[c]
            for i, c in enumerate(SAMPLE_PROPORTION_COLUMNS)
        ]
    )
    observed = x.mean(dim=0).numpy().astype("float64")
    delta = float(np.abs(observed - expected).max())
    return name, delta < 1e-5, f"max |graph mean - contract row| = {delta:.2e}"


def check_condition_vocabulary(diagnoses: pd.DataFrame) -> list[Check]:
    """Cohort-level: is the Condition node's only feature faithful to its identity?

    Not a per-subgraph property — every Subject has at most one Condition (#4 §1
    makes `OF_CONDITION` a primary-Diagnosis-only relation), so nothing inside a
    single rooted subgraph can reveal that the vocabulary is collapsing. It has
    to be asked of the whole cohort at once.

    #4 §2 locks the Condition node's feature to `conditionName`, and its
    identity is `conditionId`. If several `conditionId`s share a `conditionName`
    then the type-specific projection `f_Condition` hands anatomically distinct
    Conditions the identical embedding row, and the node type is carrying less
    than its cardinality suggests.
    """
    linked = diagnoses.dropna(subset=["conditionId"])
    per_name = linked.groupby("conditionName")["conditionId"].nunique().sort_values(ascending=False)
    collapsed = per_name[per_name > 1]
    worst = ", ".join(f"{name!r} ← {n} codes" for name, n in per_name.head(3).items())
    return [
        Check(
            "conditionName distinguishes every Condition node identity",
            collapsed.empty,
            f"{int(linked['conditionId'].nunique())} conditionIds collapse to "
            f"{int(linked['conditionName'].nunique())} conditionNames; worst: {worst}",
        )
    ]


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    nodes: dict[str, int]
    edges: dict[tuple[str, str, str], int]
    isolated: dict[str, int]


def summarise(data: HeteroData) -> Summary:
    """Node and edge counts, for rendering a subgraph in one screen."""
    return Summary(
        nodes={t: int(data[t].num_nodes) for t in data.node_types},
        edges={t: int(data[t].edge_index.size(1)) for t in data.edge_types},
        isolated={t: c for t, c in _isolated_counts(data).items() if c},
    )


def _isolated_counts(data: HeteroData) -> dict[str, int]:
    touched = {t: torch.zeros(data[t].num_nodes, dtype=torch.bool) for t in data.node_types}
    for (src, _, dst), edge_index in data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        touched[src][edge_index[0]] = True
        touched[dst][edge_index[1]] = True
    return {t: int((~mask).sum()) for t, mask in touched.items()}
