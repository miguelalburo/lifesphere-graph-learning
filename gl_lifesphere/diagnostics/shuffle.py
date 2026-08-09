"""Diagnostic three: permute the label across Subjects and retrain (encoder doc §7).

"Permute `timeToEventDays`/`eventOccurred` across Subjects and retrain. C-index
should sit at ~0.5, since permutation destroys the association while preserving
every marginal. Anything meaningfully above indicates leakage — most likely a
Survival property that survived into the feature export, per §0."

**`(time, event)` moves as one pair.** Permuting the two independently would
also destroy their dependence — censoring is not independent of duration — and
the resulting null is a different, weaker one that a leaking feature could clear
without the leak being visible. The pair is carried to a new Subject intact, so
the only thing destroyed is which Subject it belongs to.

**Permuted once for the whole cohort, before the folds are cut.** A permutation
drawn per fold would give the same Subject different labels in the fold that
trains on them and the fold that tests them, which is a data-corruption control
rather than a label-shuffle control.

**Two schemes, and the second one is not redundant.**

- `global` is §7's own specification and the gate's pass/fail: a label may land
  on any Subject in the cohort, so every association is destroyed, Study-level
  ones included.
- `within_study` permutes inside each Study, which destroys exactly what the
  headline within-Study Harrell C measures while *preserving* the fact that
  Studies differ in survival. Arms 2 and 3 carry `Condition` at
  R²_study = 0.948, so under this scheme their pooled C is expected to stay
  **above** 0.5 while their within-Study C falls to chance. That is not a
  leak — it is #4 §6's immunity argument made visible, and the direct
  demonstration that the pooled secondaries are partly a Study read-off while
  the headline metric is not.

**Every arm is shuffled, not only arm 3.** §7 frames the control for the
encoder, but the leak it is looking for lives in the shared feature export
(#4), which all three arms consume. Arms 1 and 2 are cheap, and a leak that
showed in them and not in arm 3 would be missed by a graph-only control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constructions import cache
from ..evaluation.splits import N_SPLITS, fold_split, load_folds
from ..features import assemble_raw_frame
from ..models.baseline import train as baseline_train
from ..models.graph import train as graph_train
from ..models.tabular import train as tabular_train
from ..survival import metrics
from ..survival.targets import SurvivalTarget, load_targets

GLOBAL = "global"
WITHIN_STUDY = "within_study"
SCHEMES: tuple[str, ...] = (GLOBAL, WITHIN_STUDY)


def permute_targets(targets: SurvivalTarget, *, seed: int, scheme: str = GLOBAL) -> SurvivalTarget:
    """A new target with `(time, event)` pairs redistributed across Subjects.

    `subject_id` and `study` stay exactly where they were, so the result is
    still keyed the way `SurvivalTarget.reorder` expects and every arm's
    Subject bookkeeping is untouched. Only the label moves.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown shuffle scheme {scheme!r}; expected one of {list(SCHEMES)}")

    rng = np.random.default_rng(seed)
    order = np.arange(len(targets))
    if scheme == GLOBAL:
        order = rng.permutation(order)
    else:
        for name in np.unique(targets.study):
            positions = np.flatnonzero(targets.study == name)
            order[positions] = rng.permutation(positions)

    return SurvivalTarget(
        subject_id=targets.subject_id,
        study=targets.study,
        time=targets.time[order],
        event=targets.event[order],
    )


@dataclass(frozen=True)
class ArmShuffle:
    """One arm retrained end-to-end on one permuted label."""

    arm: str
    scheme: str
    seed: int
    fold_metrics: tuple[metrics.FoldMetrics, ...]

    @property
    def within_study(self) -> tuple[float, ...]:
        return tuple(float(m.within_study.cindex) for m in self.fold_metrics)

    @property
    def pooled(self) -> tuple[float, ...]:
        return tuple(float(m.pooled_harrell_c) for m in self.fold_metrics)

    def to_dict(self) -> dict[str, object]:
        within = self.within_study
        pooled = self.pooled
        return {
            "arm": self.arm,
            "scheme": self.scheme,
            "seed": self.seed,
            metrics.HEADLINE: {
                "per_fold": list(within),
                "mean": float(np.mean(within)),
                "std": float(np.std(within)),
            },
            "pooled_harrell_c": {
                "per_fold": list(pooled),
                "mean": float(np.mean(pooled)),
                "std": float(np.std(pooled)),
            },
            "folds": [m.to_dict() for m in self.fold_metrics],
        }


@dataclass(frozen=True)
class ShuffleResult:
    runs: tuple[ArmShuffle, ...]

    def run(self, arm: str, scheme: str) -> ArmShuffle:
        for candidate in self.runs:
            if candidate.arm == arm and candidate.scheme == scheme:
                return candidate
        raise KeyError(f"no shuffle run for arm={arm!r} scheme={scheme!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostic": "label_shuffle",
            "description": (
                "Every arm retrained on a label whose (time, event) pairs have been "
                "redistributed across Subjects (encoder doc §7). C should sit at ~0.5; "
                "meaningfully above means a Survival-derived column reached the feature side."
            ),
            "schemes": {
                GLOBAL: "labels may land on any cohort Subject -- §7's specification, and the gate's pass/fail.",
                WITHIN_STUDY: (
                    "labels stay inside their Study. Destroys what within_study_harrell_c "
                    "measures while preserving Study-level survival differences, so a pooled C "
                    "above 0.5 here is the Condition~Study channel (#4 §6), not a leak."
                ),
            },
            "runs": [run.to_dict() for run in self.runs],
        }


def run_label_shuffle(
    *,
    arms: tuple[str, ...] = (baseline_train.ARM, tabular_train.ARM, graph_train.ARM),
    schemes: tuple[str, ...] = SCHEMES,
    seed: int = 0,
    baseline_config: baseline_train.BaselineArmConfig | None = None,
    tabular_config: tabular_train.TabularArmConfig | None = None,
    graph_config: graph_train.GraphArmConfig | None = None,
    records: cache.SubjectRecords | None = None,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
    folds: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> ShuffleResult:
    """Retrain each requested arm on each permutation scheme, across all 5 folds.

    Every input is injectable so the whole control can be exercised against a
    synthetic cohort — a shuffle that could only run on the real 6,811 Subjects
    would have no test of its own wiring, which is the one thing a control
    cannot afford to get wrong quietly.
    """
    raw_frame = raw if raw is not None else assemble_raw_frame()
    real_targets = targets if targets is not None else load_targets()
    fold_assignment = folds if folds is not None else load_folds()
    subject_records = records
    if graph_train.ARM in arms and subject_records is None:
        subject_records = cache.load_subject_records()

    runs: list[ArmShuffle] = []
    for scheme in schemes:
        permuted = permute_targets(real_targets, seed=seed, scheme=scheme)
        for arm in arms:
            fold_metrics = tuple(
                _run_one_fold(
                    arm,
                    outer_fold,
                    raw=raw_frame,
                    targets=permuted,
                    records=subject_records,
                    folds=fold_assignment,
                    baseline_config=baseline_config or baseline_train.BaselineArmConfig(),
                    tabular_config=tabular_config or tabular_train.TabularArmConfig(),
                    graph_config=graph_config or graph_train.GraphArmConfig(),
                    use_cache=use_cache,
                )
                for outer_fold in range(N_SPLITS)
            )
            runs.append(
                ArmShuffle(arm=arm, scheme=scheme, seed=seed, fold_metrics=fold_metrics)
            )
    return ShuffleResult(runs=tuple(runs))


def _run_one_fold(
    arm: str,
    outer_fold: int,
    *,
    raw: pd.DataFrame,
    targets: SurvivalTarget,
    records: cache.SubjectRecords | None,
    folds: pd.DataFrame,
    baseline_config: baseline_train.BaselineArmConfig,
    tabular_config: tabular_train.TabularArmConfig,
    graph_config: graph_train.GraphArmConfig,
    use_cache: bool,
) -> metrics.FoldMetrics:
    """Dispatch to the arm's own `run_fold`, with the #3 §6 self-check turned off.

    Each arm is called through its real entry point rather than a
    shuffle-specific copy, so the control exercises the same contract fitting,
    the same encoder, and the same decoder the reported result came from. A
    shuffle that ran through its own reimplementation could not detect a leak
    that lived in the arm.
    """
    split = fold_split(folds, outer_fold)
    if arm == baseline_train.ARM:
        return baseline_train.run_fold(
            outer_fold,
            raw=raw,
            targets=targets,
            split=split,
            config=baseline_config,
            expect_discrimination=False,
        ).fold_metrics
    if arm == tabular_train.ARM:
        return tabular_train.run_fold(
            outer_fold,
            raw=raw,
            targets=targets,
            split=split,
            config=tabular_config,
            expect_discrimination=False,
        ).fold_metrics
    if arm == graph_train.ARM:
        if records is None:
            raise ValueError("arm 3's shuffle needs `records`; pass them or let the runner load them")
        return graph_train.run_fold(
            outer_fold,
            records=records,
            raw=raw,
            targets=targets,
            split=split,
            config=graph_config,
            use_cache=use_cache,
            expect_discrimination=False,
        ).fold_metrics
    raise ValueError(f"unknown arm {arm!r}")
