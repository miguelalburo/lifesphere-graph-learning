"""The trust gate (#13) — the three controls encoder doc §7 puts before any result.

**A gate, not a measurement.** Each of the three constructs a model that
*should* fail and checks that it does, so what this package produces is a
verdict rather than a number to quote. The map's destination is a result trusted
even when it is bad, and this is what earns the word.

- `ablation` — retrain model 3 with every `edge_index` emptied. If the full
  encoder does not clearly beat that, message passing contributes nothing.
- `degree_probe` — a Cox fit on nothing but `log(1 + n_samples)`,
  `log(1 + n_diagnoses)`, `log(1 + n_interventions)`. Its C-index is the floor
  a structural result has to clear to be interesting, and #4 §4 excludes those
  counts from every model precisely so it should come out weak.
- `shuffle` — permute the label across Subjects and retrain. C should sit at
  ~0.5; meaningfully above it means leakage.

`gate` holds the thresholds and turns three sets of metrics into three
PASS/FAIL verdicts. The thresholds live in `experiments/configs/trust_gate.json`
so they are declared before the run rather than chosen once the numbers are in
— a gate whose bar moves to accommodate the result is not a gate.

**Two of these controls are designed to score at chance, which the rest of the
codebase treats as a bug.** #3 §6 has every model assert that its own
training-fold risk discriminates, because a sign-inverted score silently
produces `1 - C` rather than raising. Under a permuted label that premise is
inverted, so the shuffle passes `expect_discrimination=False` and the degree
probe never asserts at all. Nothing else in the project may do either.

**One module here is not part of the gate.** `relation_split` is #15, the
follow-up the degree probe's own decomposition raised: whether model 3's +0.0180
over model 2 is the secondary-diagnosis channel the relation split makes visible,
rather than message passing. It lives beside the three because it shares all
their machinery — the recorded-model reader, the comparability guard, the paired
delta — but it is an attribution question rather than a control. It has its own
verdict vocabulary, its own command
(`python -m gl_lifesphere.diagnostics.relation_split`) and its own output
directory, and `gate` has no input path to it. A cheaper follow-up must not be
able to overturn a declared FAIL.
"""

from __future__ import annotations

from .ablation import AblationResult, run_structure_ablation
from .counts import AccrualCounts, COUNT_COLUMNS, PROBE_COLUMNS, load_accrual_counts
from .degree_probe import DegreeProbeResult, run_degree_probe
from .gate import GateConfig, GateReport, Verdict, assess
from .relation_split import (
    Attribution,
    RelationSplitConfig,
    RelationSplitResult,
    assess_attribution,
    run_relation_split,
)
from .shuffle import ShuffleResult, permute_targets, run_label_shuffle

__all__ = [
    "COUNT_COLUMNS",
    "PROBE_COLUMNS",
    "AblationResult",
    "AccrualCounts",
    "Attribution",
    "DegreeProbeResult",
    "GateConfig",
    "GateReport",
    "RelationSplitConfig",
    "RelationSplitResult",
    "ShuffleResult",
    "Verdict",
    "assess",
    "assess_attribution",
    "load_accrual_counts",
    "permute_targets",
    "run_degree_probe",
    "run_label_shuffle",
    "run_relation_split",
    "run_structure_ablation",
]
