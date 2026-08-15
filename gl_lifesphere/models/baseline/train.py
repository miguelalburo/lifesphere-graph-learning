"""Model 1 — the shared decoder fit directly on clinical covariates (#9, #3 §3, #8).

No stage one. Models 2 and 3 train an encoder and freeze it before refitting the
shared decoder on `z`; model 1 *is* the shared decoder, `lifelines.CoxPHFitter
(penalizer=lambda, l1_ratio=0, strata=['study'])`, fit directly on staging and
pathology covariates — the model class and covariate set #8 settled. Every
other mechanic (fold structure, `lambda` selection, risk-score convention,
scoring) is identical to model 2's stage two, by construction: both call through
`gl_lifesphere.survival.decoder` and `gl_lifesphere.survival.metrics`.

`baseline_columns` is the one piece of logic specific to this model: it takes
`features.FeatureContract`'s full shared design matrix (#4) and selects the
subset `CONTEXT.md`'s Baseline definition ("staging and pathology features
only, no molecular data") allows:

- **`p_*` (Sample-type proportions) excluded.** #4 §4 states explicitly that
  models 2 and 3 carry that ascertainment channel and "model 1 does not" — it is
  a specimen-composition signal, not a staging/pathology one.
- **`condition_*` excluded — a judgement call, not a direct reading of #4.**
  (Read `conditionName` for `conditionId` throughout this paragraph: the
  measurement below was run before #12 re-keyed the feature on identity. The
  exclusion only hardens — keying on `conditionId` raises R²_study from 0.772
  to 0.948 and the level count from ~28 to 121, so both the collinearity
  symptom and the parameter cost get worse, not better.)
  #4 §6's general contract keeps `Condition` for every model
  ("Condition and conditionSubtype are both kept, with Condition flagged"),
  reasoned safe there because the *headline* metric is immune to a pure
  Study proxy by construction. #8's own covariate enumeration for this model,
  though, never names it ("stage ordinal I-IV, age, conditionSubtype,
  histological type"). Measured directly on the frozen cohort with #4 §7's
  own method (5-fold CV, stratified penalised Cox, within-Study Harrell C)
  before deciding between the two readings: including `conditionName` adds
  ~27 parameters, triggers `lifelines`' perfect-separation warning in
  multiple folds (`condition___missing__` "completely determines whether a
  subject dies or not" — the same collinearity-with-Study symptom #4 §7
  diagnosed for `conditionSubtype`/`diagnosisMethod`), and moves within-Study
  C from 0.6556 to 0.6542 — 0.0014, an order of magnitude under the ~0.02
  fold-noise resolution limit #3 §5 established. Excluded on that basis:
  #4 §6's safety argument does not extend to a linear model carrying the risk
  of near-singular coefficients for zero measurable benefit.
- Study is already the Cox stratum (#3 §1) and never a covariate; a
  near-duplicate of the stratum (`conditionName`) fighting into the design
  matrix is the concrete form that would take.

Everything else in the shared contract survives: sex, race, stage, age, and
`conditionSubtype` (histological subtype) are exactly the covariate set #8's
resolution names, drawn from #4.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from ...evaluation.splits import FoldSplit, fold_split, load_folds
from ...features import FeatureContract, assemble_raw_frame, fit_feature_contract
from ...survival import decoder, metrics
from ...survival.targets import SurvivalTarget, load_targets


# Named here rather than only in `__main__` so callers that run this model as one
# of several — #13's label shuffle retrains all three — can address it the same
# way they address models 2 and 3.
MODEL = "model1_baseline"


def baseline_columns(contract: FeatureContract) -> list[str]:
    """The staging/pathology-only subset of one fold's fitted `FeatureContract`."""
    return [
        *contract.sex.output_columns,
        *contract.race.output_columns,
        *contract.stage.output_columns,
        *contract.age.output_columns,
        *contract.condition_subtype.output_columns,
    ]


@dataclass(frozen=True)
class BaselineModelConfig:
    """Model 1 has no learned representation, so `penalty_grid` is the only real
    hyperparameter; `seed` is carried for the config traceability #9 asks for
    even though nothing here is stochastic (the Cox fit is deterministic)."""

    seed: int = 0
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BaselineModelConfig":
        """Ignores keys the dataclass does not declare (e.g. a `_comment`)."""
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        if "penalty_grid" in kwargs:
            kwargs["penalty_grid"] = tuple(kwargs["penalty_grid"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]


def _design_and_target(
    contract: FeatureContract,
    columns: list[str],
    raw: pd.DataFrame,
    subject_ids: frozenset[str],
    targets: SurvivalTarget,
) -> tuple[pd.DataFrame, SurvivalTarget]:
    design = contract.transform(raw, subject_ids)[columns]
    return design, targets.reorder(design.index)


def _drop_zero_variance_training_columns(train_design: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns constant on the *training* fold, dropped before any covariate reaches the decoder.

    Verified against the real cohort: `race`'s one-hot carries `__RARE__` and
    `__MISSING__` sentinel columns unconditionally (`features.contract`), but
    `race` is a closed ~5-category vocabulary encoded at `min_count=1`, so
    `__RARE__` is essentially never populated by any training fold, and
    `conditionSubtype`'s `__MISSING__` bucket is frequently empty too. A
    column that is constant across every training Subject carries no fit
    information and is exactly collinear with any other constant column,
    which singularises the Cox Hessian (`lifelines.exceptions.ConvergenceError:
    delta contains nan value(s)`, reproduced against the frozen cohort's fold
    0). Dropping it is strictly correct for a penalised regression regardless
    of where the constant column came from, so this stays in model 1's own code
    rather than changing the shared contract's column list for every model.
    """
    return [column for column in columns if train_design[column].nunique(dropna=False) > 1]


@dataclass(frozen=True)
class FoldResult:
    fold: int
    config: BaselineModelConfig
    contract: FeatureContract = field(repr=False)
    covariates: tuple[str, ...]
    penalty_selection: decoder.PenaltySelection
    fold_metrics: metrics.FoldMetrics
    # subjectId-indexed test-fold predictions (studyId, risk, durationDays,
    # event) — every fold's test set is disjoint and their union is the whole
    # cohort (#7), so concatenating this across all 5 folds is what the
    # literature sanity check (#8, #9) scores against.
    test_predictions: pd.DataFrame = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "covariates": list(self.covariates),
            "chosen_penalizer": self.penalty_selection.chosen_penalizer,
            "penalty_scores": {str(k): v for k, v in self.penalty_selection.scores.items()},
            "metrics": self.fold_metrics.to_dict(),
        }


def run_fold(
    outer_fold: int,
    *,
    raw: pd.DataFrame,
    targets: SurvivalTarget,
    split: FoldSplit,
    config: BaselineModelConfig,
    expect_discrimination: bool = True,
) -> FoldResult:
    """One outer fold: fit contract on inner-train -> select lambda -> fit -> score.

    The feature contract is fit on `split.train` alone, excluding `split.val`
    — matching model 2's own implementation (`gl_lifesphere/models/tabular/train.py`)
    so the nested validation slice sees only train-fitted statistics on both
    models, not just the one with an encoder to early-stop.

    `expect_discrimination=False` belongs to #13's label shuffle alone; see
    `survival.two_stage.two_stage_score`.
    """
    contract = fit_feature_contract(raw, split.train)
    columns = baseline_columns(contract)
    train_design = contract.transform(raw, split.train)[columns]
    columns = _drop_zero_variance_training_columns(train_design, columns)

    x_train, train_target = train_design[columns], targets.reorder(train_design.index)
    x_val, val_target = _design_and_target(contract, columns, raw, split.val, targets)
    x_test, test_target = _design_and_target(contract, columns, raw, split.test, targets)

    x_trainval = pd.concat([x_train, x_val])
    trainval_target = targets.reorder(x_trainval.index)

    selection = decoder.select_penalty(
        x_train,
        train_study=train_target.study,
        train_time=train_target.time,
        train_event=train_target.event,
        trainval=x_trainval,
        trainval_study=trainval_target.study,
        trainval_time=trainval_target.time,
        trainval_event=trainval_target.event,
        grid=config.penalty_grid,
    )

    fit = decoder.fit_decoder(
        x_trainval,
        study=trainval_target.study,
        time=trainval_target.time,
        event=trainval_target.event,
        penalizer=selection.chosen_penalizer,
    )

    # #3 §6: every model must self-check its training-fold score before trusting
    # the held-out one -- a sign-inverted risk silently produces `1 - C`.
    train_risk = decoder.risk_scores(fit, x_trainval, study=trainval_target.study)
    if expect_discrimination:
        metrics.assert_discriminates(
            trainval_target.event,
            trainval_target.time,
            train_risk,
            where=f"model1 fold {outer_fold} (train+val)",
        )

    test_risk = decoder.risk_scores(fit, x_test, study=test_target.study)
    horizons = metrics.DEFAULT_HORIZONS_DAYS
    test_survival = decoder.survival_function(
        fit, x_test, study=test_target.study, times=np.asarray(horizons)
    )
    fold_metrics = metrics.score_fold(
        train_time=trainval_target.time,
        train_event=trainval_target.event,
        test_time=test_target.time,
        test_event=test_target.event,
        test_study=test_target.study,
        risk=test_risk,
        survival_probabilities=test_survival,
        horizons=horizons,
    )

    test_predictions = pd.DataFrame(
        {
            "studyId": test_target.study,
            "risk": test_risk,
            "durationDays": test_target.time,
            "event": test_target.event,
        },
        index=x_test.index,
    )

    return FoldResult(
        fold=outer_fold,
        config=config,
        contract=contract,
        covariates=tuple(columns),
        penalty_selection=selection,
        fold_metrics=fold_metrics,
        test_predictions=test_predictions,
    )


def run_all_folds(
    config: BaselineModelConfig,
    *,
    raw: pd.DataFrame | None = None,
    targets: SurvivalTarget | None = None,
) -> list[FoldResult]:
    """All 5 outer folds of the locked Study-stratified split (#7)."""
    raw_frame = raw if raw is not None else assemble_raw_frame()
    survival_targets = targets if targets is not None else load_targets()
    folds = load_folds()

    results = []
    for outer_fold in range(5):
        split = fold_split(folds, outer_fold)
        results.append(
            run_fold(outer_fold, raw=raw_frame, targets=survival_targets, split=split, config=config)
        )
    return results
