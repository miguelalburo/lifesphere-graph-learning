"""Diagnostic two: a Cox model on nothing but three accrual counts (encoder doc §7).

"Fit a Cox model on nothing but `log(1+n_samples)`, `log(1+n_diagnoses)`,
`log(1+n_interventions)`. If this alone reaches a respectable C-index, accrual
is acting as a proxy for follow-up duration (§0's immortal-time bias) and *any*
model with access to those counts — including the encoder via its
neighbourhoods — is partly measuring it. This probe's C-index is the floor a
structural result has to clear to be interesting."

Everything downstream of the design matrix is the arms' own machinery: the same
`decoder.select_penalty`, the same `CoxPHFitter(strata=['study'])`, the same
`metrics.score_fold`, the same persisted folds. Only the covariates differ,
which is what makes the probe's C directly comparable to an arm's.

**Two things this arm-shaped code deliberately does not do.**

It never calls `metrics.assert_discriminates`. Every arm self-checks that its
training-fold risk beats chance (#3 §5, §6), because a sign-inverted score
silently returns `1 - C`. This model is *designed* to fail that check — a probe
that came out weak would crash the run — so the training-fold C is recorded as
a result instead. The sign convention is still safe: the probe never touches a
learned representation, so there is no sign to invert.

And it fits no feature contract. The counts are not features and must never
become them; `counts.py` explains the three separate mechanisms that keep the
table out of the arms, of which `guards.IMMORTAL_TIME_DERIVED` naming
`nInterventions` is the last line.

**The coefficients are the diagnosis, not the C-index.** A weak overall C with a
significantly protective `nInterventions` still says accrual is measuring
follow-up; #4 §4 already measured bare Intervention presence at HR 0.716,
p = 8.5e-05, and this probe is the continuous form of that same measurement, so
the per-covariate hazard ratios are recorded alongside the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.splits import N_SPLITS, FoldSplit, fold_split, load_folds
from ..survival import decoder, metrics
from ..survival.targets import SurvivalTarget, load_targets
from .counts import PROBE_COLUMNS, SECONDARY_DIAGNOSIS, AccrualCounts, load_accrual_counts

ALL = "all"


@dataclass(frozen=True)
class ProbeFold:
    """One outer fold of the probe: the fit, its coefficients, and its held-out score."""

    fold: int
    chosen_penalizer: float
    train_harrell_c: float
    hazard_ratios: dict[str, dict[str, float]]
    fold_metrics: metrics.FoldMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "chosen_penalizer": self.chosen_penalizer,
            "train_harrell_c": self.train_harrell_c,
            "hazard_ratios": self.hazard_ratios,
            "metrics": self.fold_metrics.to_dict(),
        }


def subset_grid(columns: tuple[str, ...] = PROBE_COLUMNS) -> dict[str, tuple[str, ...]]:
    """The full covariate set, each covariate alone, and each leave-one-out.

    A probe that comes out strong is only actionable once it is decomposed. The
    three counts do not mean the same thing: a *protective* hazard ratio is
    encoder doc §0's immortal-time bias (more of the thing because the Subject
    lived to accrue it), while a *harmful* one is disease burden and is
    ordinary prognostic signal. Which one drives the probe decides whether a
    strong result is a defect to fix or a floor to clear.
    """
    grid = {ALL: tuple(columns)}
    for column in columns:
        grid[f"only:{column}"] = (column,)
    if len(columns) > 2:
        for column in columns:
            grid[f"without:{column}"] = tuple(c for c in columns if c != column)
    # Deliberately outside `ALL`: this is not one of §7's three covariates, so
    # it must not move the headline number. It is here because it is the one
    # accrual channel an arm can actually see (`counts.SECONDARY_DIAGNOSIS`),
    # which is what decides whether a strong probe implicates arm 3 or merely
    # sets a floor.
    grid[f"arm3_visible:{SECONDARY_DIAGNOSIS}"] = (SECONDARY_DIAGNOSIS,)
    return grid


@dataclass(frozen=True)
class DegreeProbeResult:
    """The probe over the full covariate set, plus every decomposition of it."""

    by_subset: dict[str, tuple[ProbeFold, ...]]

    @property
    def folds(self) -> tuple[ProbeFold, ...]:
        return self.by_subset[ALL]

    @property
    def per_fold(self) -> tuple[float, ...]:
        return self._per_fold(ALL)

    def _per_fold(self, subset: str) -> tuple[float, ...]:
        return tuple(
            float(f.fold_metrics.within_study.cindex) for f in self.by_subset[subset]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnostic": "degree_probe",
            "description": (
                "Stratified penalised Cox on log(1+n_samples), log(1+n_diagnoses), "
                "log(1+n_interventions) and nothing else (encoder doc §7). Same folds, same "
                "decoder, same metrics as every arm; the counts are a control's input and are "
                "excluded from every arm's features by #4 §4."
            ),
            "covariates": list(PROBE_COLUMNS),
            "metric": metrics.HEADLINE,
            "per_fold": list(self.per_fold),
            "mean": float(np.mean(self.per_fold)),
            "std": float(np.std(self.per_fold)),
            "decomposition": {
                subset: {
                    "covariates": list(self.by_subset[subset][0].hazard_ratios),
                    "per_fold": list(self._per_fold(subset)),
                    "mean": float(np.mean(self._per_fold(subset))),
                }
                for subset in sorted(self.by_subset)
            },
            "hazard_ratios_fold_0": self.by_subset[ALL][0].hazard_ratios,
            "how_to_read_a_hazard_ratio": (
                "Below 1 on an accrual count is the immortal-time signature (encoder doc §0): "
                "more of the thing, lower hazard, because accruing it required surviving to "
                "accrue it. Above 1 is disease burden and is ordinary prognostic signal, which "
                "raises the floor without implying a defect."
            ),
            "self_check_waived": (
                "metrics.assert_discriminates is deliberately not called: this control is "
                "designed to score at chance, and #3 §6's check would turn a passing probe into "
                "a crash. train_harrell_c is recorded per fold instead."
            ),
            "folds": [fold.to_dict() for fold in self.folds],
        }


def run_fold(
    outer_fold: int,
    *,
    counts: AccrualCounts,
    targets: SurvivalTarget,
    split: FoldSplit,
    columns: tuple[str, ...] = PROBE_COLUMNS,
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID,
) -> ProbeFold:
    """One outer fold: select lambda on the nested slice, fit, score the held-out fold.

    Structured exactly like arm 1's `run_fold` (`models/baseline/train.py`), and
    for the same reason: the probe's number is only a floor for an arm's number
    if it was produced by the same protocol.
    """
    x_train = counts.design(split.train, columns)
    x_val = counts.design(split.val, columns)
    x_test = counts.design(split.test, columns)

    train_target = targets.reorder(x_train.index)
    x_trainval = pd.concat([x_train, x_val])
    trainval_target = targets.reorder(x_trainval.index)
    test_target = targets.reorder(x_test.index)

    selection = decoder.select_penalty(
        x_train,
        train_study=train_target.study,
        train_time=train_target.time,
        train_event=train_target.event,
        trainval=x_trainval,
        trainval_study=trainval_target.study,
        trainval_time=trainval_target.time,
        trainval_event=trainval_target.event,
        grid=penalty_grid,
    )

    fit = decoder.fit_decoder(
        x_trainval,
        study=trainval_target.study,
        time=trainval_target.time,
        event=trainval_target.event,
        penalizer=selection.chosen_penalizer,
    )

    train_risk = decoder.risk_scores(fit, x_trainval, study=trainval_target.study)
    test_risk = decoder.risk_scores(fit, x_test, study=test_target.study)
    horizons = metrics.DEFAULT_HORIZONS_DAYS
    test_survival = decoder.survival_function(
        fit, x_test, study=test_target.study, times=np.asarray(horizons)
    )

    return ProbeFold(
        fold=outer_fold,
        chosen_penalizer=selection.chosen_penalizer,
        train_harrell_c=metrics.harrell_c(
            trainval_target.event, trainval_target.time, train_risk
        ),
        hazard_ratios=_hazard_ratios(fit),
        fold_metrics=metrics.score_fold(
            train_time=trainval_target.time,
            train_event=trainval_target.event,
            test_time=test_target.time,
            test_event=test_target.event,
            test_study=test_target.study,
            risk=test_risk,
            survival_probabilities=test_survival,
            horizons=horizons,
        ),
    )


def _hazard_ratios(fit: decoder.CoxPHFitter) -> dict[str, dict[str, float]]:
    """Per-covariate `exp(coef)` and p-value, in the shape #4 §4 reported its own.

    An HR below 1 on an accrual count is the immortal-time signature: more of
    the thing, lower hazard, because accruing it required surviving to accrue it.
    """
    summary = fit.summary
    return {
        str(covariate): {
            "coef": float(row["coef"]),
            "hazard_ratio": float(row["exp(coef)"]),
            "p": float(row["p"]),
        }
        for covariate, row in summary.iterrows()
    }


def run_degree_probe(
    *,
    counts: AccrualCounts | None = None,
    targets: SurvivalTarget | None = None,
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID,
    subsets: dict[str, tuple[str, ...]] | None = None,
) -> DegreeProbeResult:
    """All 5 outer folds of the locked Study-stratified split (#7), per covariate subset."""
    accrual = counts if counts is not None else load_accrual_counts()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()
    grid = subsets if subsets is not None else subset_grid()

    return DegreeProbeResult(
        by_subset={
            name: tuple(
                run_fold(
                    outer_fold,
                    counts=accrual,
                    targets=survival_targets,
                    split=fold_split(folds, outer_fold),
                    columns=columns,
                    penalty_grid=penalty_grid,
                )
                for outer_fold in range(N_SPLITS)
            )
            for name, columns in grid.items()
        }
    )
