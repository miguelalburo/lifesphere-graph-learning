"""The shared decoder every model is refitted and scored through (#3 §3).

`lifelines.CoxPHFitter(penalizer=lambda, l1_ratio=0, strata=['study'])` is the
single fit — not `sksurv.CoxnetSurvivalAnalysis`, which has no `strata`
parameter and would silently reverse the stratification decision. Two-stage
wiring means every model hands this module a representation matrix (raw clinical
covariates for model 1, a frozen encoder's `z` for models 2 and 3) and gets back
risk scores through the identical call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .losses import stratified_partial_log_likelihood

STUDY_COLUMN = "study"
DURATION_COLUMN = "time"
EVENT_COLUMN = "event"

# 12-point log grid, 1e-4 .. 1e1 (#3 §3), identical for every model and fold.
PENALTY_GRID: tuple[float, ...] = tuple(np.logspace(-4, 1, 12))


def _scored_frame(representation: pd.DataFrame, *, study: np.ndarray) -> pd.DataFrame:
    """`representation` plus the strata column every scoring call needs present."""
    frame = representation.reset_index(drop=True).copy()
    frame[STUDY_COLUMN] = np.asarray(study)
    return frame


def fit_decoder(
    representation: pd.DataFrame,
    *,
    study: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    penalizer: float,
) -> CoxPHFitter:
    """Fit the shared decoder on one representation matrix, at one penalty."""
    frame = _scored_frame(representation, study=study)
    frame[DURATION_COLUMN] = np.asarray(time, dtype="float64")
    frame[EVENT_COLUMN] = np.asarray(event, dtype="bool")

    fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0, strata=[STUDY_COLUMN])
    fitter.fit(frame, duration_col=DURATION_COLUMN, event_col=EVENT_COLUMN)
    return fitter


def risk_scores(fitter: CoxPHFitter, representation: pd.DataFrame, *, study: np.ndarray) -> np.ndarray:
    """The convention-correct risk score: higher = higher hazard = shorter survival.

    `predict_log_partial_hazard`, never `lifelines.utils.concordance_index`'s
    predicted-time convention (#3 §6 bans it — a sign error there is silent).
    `study` must be supplied because the strata column has to be present on any
    frame this fitter scores, including held-out folds.
    """
    frame = _scored_frame(representation, study=study)
    return fitter.predict_log_partial_hazard(frame).to_numpy(dtype="float64")


def survival_function(
    fitter: CoxPHFitter, representation: pd.DataFrame, *, study: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Breslow S(t|x) per Subject at `times`, shape (n_subjects, n_times).

    Fit on training folds only and applied to held-out Subjects, per decoder
    doc §9 — computing `baseline_cumulative_hazard_` on test data would leak
    into calibration metrics.
    """
    frame = _scored_frame(representation, study=study)
    predicted = fitter.predict_survival_function(frame, times=times)
    return predicted.to_numpy(dtype="float64").T


@dataclass(frozen=True)
class PenaltySelection:
    """One fold's penalty grid search, kept in full so every value is recorded (#3 §3)."""

    chosen_penalizer: float
    scores: dict[float, float]


def select_penalty(
    train: pd.DataFrame,
    *,
    train_study: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    trainval: pd.DataFrame,
    trainval_study: np.ndarray,
    trainval_time: np.ndarray,
    trainval_event: np.ndarray,
    grid: tuple[float, ...] = PENALTY_GRID,
) -> PenaltySelection:
    """Pick lambda maximising `ll(train+val) - ll(train)` (#3 §3).

    Both terms are the bare stratified partial log-likelihood at each fit's own
    fitted coefficients — evaluated with `torchsurv`, per #3's amendment, never
    `CoxPHFitter.log_likelihood_`, which nets off the penalty and would bias
    selection toward small `lambda`. The train-only fit's likelihood is scored
    on the train data, and the train+val fit's likelihood is scored on the
    train+val data, so neither term touches a risk set collapsed to just the
    val slice.
    """
    scores: dict[float, float] = {}
    for penalizer in grid:
        train_fit = fit_decoder(
            train, study=train_study, time=train_time, event=train_event, penalizer=penalizer
        )
        train_ll = stratified_partial_log_likelihood(
            risk_scores(train_fit, train, study=train_study),
            train_time,
            train_event,
            train_study,
        )

        trainval_fit = fit_decoder(
            trainval,
            study=trainval_study,
            time=trainval_time,
            event=trainval_event,
            penalizer=penalizer,
        )
        trainval_ll = stratified_partial_log_likelihood(
            risk_scores(trainval_fit, trainval, study=trainval_study),
            trainval_time,
            trainval_event,
            trainval_study,
        )
        scores[penalizer] = trainval_ll - train_ll

    chosen = max(scores, key=lambda p: scores[p])
    return PenaltySelection(chosen_penalizer=chosen, scores=scores)
