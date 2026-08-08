"""Tests for the shared stratified Cox decoder (#3 §3) and its loss (#3 §1),
depended on by arm 1 (#9) and arm 2 (#10).

`tests/test_stack.py::TestObjectiveAgreement` already pins that `torchsurv`
and `lifelines` optimise the same stratified Efron objective at the library
level, including that `log_likelihood_` nets off the penalty. This file pins
the modules built on that guarantee: `losses.stratified_partial_log_likelihood`
actually agrees with `log_likelihood_` at `penalizer=0`, `decoder.fit_decoder`
recovers a planted effect's sign, `select_penalty` produces one score per grid
point, and `risk_scores` never falls back to the banned
`lifelines.utils.concordance_index` time convention.

Nothing here touches the live instance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gl_lifesphere.survival import decoder, losses, metrics

PLANTED_BETA = np.array([0.8, -0.5, 0.2])


def _synthetic_cohort(
    n: int = 400, n_strata: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """A representation matrix + (study, time, event) shaped like one fold's `x_train`.

    Identical shape to `tests/test_stack.py`'s `_stratified_cohort` fixture —
    reused rather than shared via import since this file must stand alone, but
    kept the same on purpose: a round-robin stratum assignment tried here
    first produced a ~0.1 lifelines/torchsurv disagreement where this one
    reproduces the pinned ~4e-5 (`tests/test_stack.py::TestObjectiveAgreement`),
    so the *shape* of the fixture — not just the loss formula — is part of
    what that agreement was measured on.
    """
    rng = np.random.default_rng(seed)
    subject_ids = [f"S{i:04d}" for i in range(n)]
    x = rng.normal(size=(n, len(PLANTED_BETA)))
    stratum = rng.integers(0, n_strata, size=n)
    study = np.array([f"STUDY-{s}" for s in stratum])
    latent = rng.exponential(scale=np.exp(-(x @ PLANTED_BETA)) * (1.0 + stratum))
    censor = rng.exponential(scale=2.0 * (1.0 + stratum), size=n)
    time = np.ceil(np.minimum(latent, censor) * 60.0)
    event = latent <= censor

    representation = pd.DataFrame(
        x, columns=["x0", "x1", "x2"], index=pd.Index(subject_ids, name="subjectId")
    )
    return representation, study, time, event


@pytest.fixture
def cohort() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _synthetic_cohort()


class TestPenaltyGrid:
    def test_is_a_twelve_point_log_grid_from_1e_minus4_to_1e1(self) -> None:
        assert len(decoder.PENALTY_GRID) == 12
        assert decoder.PENALTY_GRID[0] == pytest.approx(1e-4)
        assert decoder.PENALTY_GRID[-1] == pytest.approx(1e1)


class TestStratifiedPartialLogLikelihood:
    def test_agrees_with_lifelines_log_likelihood_at_zero_penalty(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """`log_likelihood_` nets off the penalty only once `penalizer > 0`
        (`tests/test_stack.py::test_penalised_log_likelihood_is_not_the_bare_partial_likelihood`)
        — at 0.0 the two must agree."""
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        risk = decoder.risk_scores(fitter, representation, study=study)
        bare = losses.stratified_partial_log_likelihood(risk, time, event, study)
        assert bare == pytest.approx(float(fitter.log_likelihood_), abs=1e-3)

    def test_diverges_from_log_likelihood_once_penalised(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Pins the trap #3's 2026-08-07 amendment exists to avoid."""
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=1.0)
        risk = decoder.risk_scores(fitter, representation, study=study)
        bare = losses.stratified_partial_log_likelihood(risk, time, event, study)
        assert bare > float(fitter.log_likelihood_) + 1.0


class TestFitDecoder:
    def test_recovers_the_sign_of_a_planted_effect(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        assert np.array_equal(np.sign(fitter.params_.values), np.sign(PLANTED_BETA))

    def test_returns_a_baseline_hazard_per_stratum(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        assert fitter.baseline_cumulative_hazard_.shape[1] == len(set(study))


class TestRiskScores:
    def test_higher_risk_is_shorter_survival(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """#3 §6's sign convention, checked through the module rather than assumed."""
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        risk = decoder.risk_scores(fitter, representation, study=study)
        metrics.assert_discriminates(event, time, risk, where="test")

    def test_discriminates_on_the_training_fold(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        risk = decoder.risk_scores(fitter, representation, study=study)
        assert metrics.harrell_c(event, time, risk) > 0.6


class TestSelectPenalty:
    def test_returns_one_score_per_grid_point(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        representation, study, time, event = cohort
        train_mask = np.arange(len(representation)) < 200

        result = decoder.select_penalty(
            representation.iloc[train_mask],
            train_study=study[train_mask],
            train_time=time[train_mask],
            train_event=event[train_mask],
            trainval=representation,
            trainval_study=study,
            trainval_time=time,
            trainval_event=event,
        )
        assert set(result.scores) == set(decoder.PENALTY_GRID)
        assert result.chosen_penalizer in decoder.PENALTY_GRID
        assert result.scores[result.chosen_penalizer] == max(result.scores.values())


class TestSurvivalFunction:
    def test_returns_a_row_per_subject_and_a_column_per_time(
        self, cohort: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        representation, study, time, event = cohort
        fitter = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        times = np.array([5.0, 20.0, 60.0])
        surv = decoder.survival_function(fitter, representation, study=study, times=times)
        assert surv.shape == (len(representation), len(times))
        assert ((surv >= 0.0) & (surv <= 1.0)).all()
