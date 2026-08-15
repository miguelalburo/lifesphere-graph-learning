"""Stage two: freeze the encoder's `z`, select `lambda`, fit the shared decoder, score.

#3 §3 makes the two-stage wiring the primary result for every neural model — an
encoder trained on the stratified Efron Cox loss, then frozen, then a
`lifelines.CoxPHFitter(strata=['study'])` refit on its `z`. Stage *one* differs
between the models, because that is the representation under test; stage two must
not, because a difference there would land in the comparison as if it were a
structural finding.

This module is that shared half, extracted so it is one implementation rather
than a convention two `train.py` files are each expected to honour. It takes
`z` as a plain frame and knows nothing about where it came from — an MLP over a
design matrix (#10) or an R-GCN over a rooted subgraph (#12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import decoder, metrics
from .targets import SurvivalTarget


@dataclass(frozen=True)
class TwoStageResult:
    """One pass of stage two: select `lambda`, fit the shared decoder, score it."""

    penalty_selection: decoder.PenaltySelection
    fold_metrics: metrics.FoldMetrics
    # The #3 §6 self-check's own number, kept rather than discarded. It is the
    # in-sample C on train+val, so it is not a result — but under #13's label
    # shuffle it is the one place a leak would show up first and loudest, since
    # a permuted label with a leaking covariate still fits in-sample.
    train_harrell_c: float


def two_stage_score(
    *,
    z_train: pd.DataFrame,
    train_target: SurvivalTarget,
    z_trainval: pd.DataFrame,
    trainval_target: SurvivalTarget,
    z_test: pd.DataFrame,
    test_target: SurvivalTarget,
    penalty_grid: tuple[float, ...] = decoder.PENALTY_GRID,
    where: str,
    expect_discrimination: bool = True,
) -> TwoStageResult:
    """Stage two, given a (frozen, already-embedded) `z`.

    Shared by each model's primary trained-encoder pass and by #3 §3's
    random-init-encoder diagnostic — both are "extract `z`, then pass it through
    the identical shared decoder", differing only in where `z` came from.

    `expect_discrimination=False` suppresses the #3 §6 self-check, and **only
    #13's label-shuffle control may pass it**. That check exists because a
    sign-inverted risk silently produces `1 - C` rather than raising; under a
    permuted label the model is *supposed* to score at chance, so the check's
    premise is inverted and leaving it on would turn a passing control into a
    crash. The training-fold C is computed either way and returned on the
    result, so nothing is lost by skipping the raise.
    """
    selection = decoder.select_penalty(
        z_train,
        train_study=train_target.study,
        train_time=train_target.time,
        train_event=train_target.event,
        trainval=z_trainval,
        trainval_study=trainval_target.study,
        trainval_time=trainval_target.time,
        trainval_event=trainval_target.event,
        grid=penalty_grid,
    )

    fit = decoder.fit_decoder(
        z_trainval,
        study=trainval_target.study,
        time=trainval_target.time,
        event=trainval_target.event,
        penalizer=selection.chosen_penalizer,
    )

    # #3 §6: every model must self-check its training-fold score before trusting
    # the held-out one — a sign-inverted risk silently produces `1 - C`.
    train_risk = decoder.risk_scores(fit, z_trainval, study=trainval_target.study)
    train_harrell_c = (
        metrics.assert_discriminates(
            trainval_target.event, trainval_target.time, train_risk, where=where
        )
        if expect_discrimination
        else metrics.harrell_c(trainval_target.event, trainval_target.time, train_risk)
    )

    test_risk = decoder.risk_scores(fit, z_test, study=test_target.study)
    horizons = metrics.DEFAULT_HORIZONS_DAYS
    test_survival = decoder.survival_function(
        fit, z_test, study=test_target.study, times=np.asarray(horizons)
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
    return TwoStageResult(
        penalty_selection=selection,
        fold_metrics=fold_metrics,
        train_harrell_c=train_harrell_c,
    )
