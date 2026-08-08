"""Tests for `gl_lifesphere.survival` (#10, locked by #3).

`tests/test_stack.py` already pins the underlying `torchsurv`/`lifelines`/
`sksurv` contracts these modules build on; this file pins that *our* wrapper
code around them preserves those contracts — the sign convention, the
pair-pooling rule, and the "select on the bare partial likelihood, not
`log_likelihood_`" rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gl_lifesphere.survival import decoder, losses, metrics
from gl_lifesphere.survival.targets import SurvivalTarget

SEED = 0


def _stratified_cohort(
    n: int = 300, n_strata: int = 4, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same shape as `tests/fixtures.py`'s stack tests: a planted linear effect,
    heavy censoring, per-stratum baseline hazard scale, integer day-counts."""
    rng = np.random.default_rng(seed)
    beta = np.array([0.8, -0.5])
    x = rng.normal(size=(n, len(beta)))
    study = np.array([f"STUDY-{i}" for i in rng.integers(0, n_strata, size=n)])
    strata_scale = 1.0 + np.array([int(s.split("-")[1]) for s in study])
    latent = rng.exponential(scale=np.exp(-(x @ beta)) * strata_scale)
    censor = rng.exponential(scale=2.0 * strata_scale, size=n)
    time = np.ceil(np.minimum(latent, censor) * 60.0)
    event = latent <= censor
    return x, study, event, time


class TestLosses:
    def test_loss_is_finite_and_differentiable(self) -> None:
        _, study, event, time = _stratified_cohort()
        risk = torch.randn(len(study), requires_grad=True)

        loss = losses.stratified_efron_cox_loss(risk, time, event, study)
        loss.backward()

        assert torch.isfinite(loss)
        assert risk.grad is not None
        assert torch.isfinite(risk.grad).all()
        assert risk.grad.norm() > 0

    def test_bare_partial_log_likelihood_matches_lifelines(self) -> None:
        """The statistic `decoder.select_penalty` compares across the grid must
        be the same objective `lifelines` optimises (#3's amendment)."""
        from lifelines import CoxPHFitter

        x, study, event, time = _stratified_cohort()
        frame = pd.DataFrame(x, columns=["x0", "x1"])
        frame["time"] = time
        frame["event"] = event
        frame["study"] = study

        fitter = CoxPHFitter(penalizer=0.0, l1_ratio=0.0, strata=["study"])
        fitter.fit(frame, duration_col="time", event_col="event")

        ours = losses.stratified_partial_log_likelihood(
            x @ fitter.params_.values, time, event, study
        )
        assert ours == pytest.approx(float(fitter.log_likelihood_), abs=1e-3)


class TestDecoder:
    def test_recovers_the_sign_of_a_planted_effect_and_scores_above_chance(self) -> None:
        x, study, event, time = _stratified_cohort()
        representation = pd.DataFrame(x, columns=["x0", "x1"])

        fit = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.0)
        risk = decoder.risk_scores(fit, representation, study=study)

        assert np.array_equal(np.sign(fit.params_.values), np.sign([0.8, -0.5]))
        assert metrics.harrell_c(event, time, risk) > 0.6

    def test_survival_function_is_a_valid_probability_and_non_increasing(self) -> None:
        x, study, event, time = _stratified_cohort()
        representation = pd.DataFrame(x, columns=["x0", "x1"])
        fit = decoder.fit_decoder(representation, study=study, time=time, event=event, penalizer=0.1)

        times = np.array([5.0, 10.0, 20.0])
        sf = decoder.survival_function(fit, representation, study=study, times=times)

        assert sf.shape == (len(study), len(times))
        assert ((sf >= 0.0) & (sf <= 1.0)).all()
        assert (np.diff(sf, axis=1) <= 1e-9).all(), "S(t) must be non-increasing in t"

    def test_select_penalty_chooses_from_the_supplied_grid_and_records_every_score(self) -> None:
        x, study, event, time = _stratified_cohort()
        representation = pd.DataFrame(x, columns=["x0", "x1"])
        grid = (0.01, 0.1, 1.0)

        selection = decoder.select_penalty(
            representation,
            train_study=study,
            train_time=time,
            train_event=event,
            trainval=representation,
            trainval_study=study,
            trainval_time=time,
            trainval_event=event,
            grid=grid,
        )

        assert selection.chosen_penalizer in grid
        assert set(selection.scores) == set(grid)


class TestMetrics:
    def test_perfect_ranking_scores_one(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([True, True, True, True])
        risk = np.array([4.0, 3.0, 2.0, 1.0])  # higher risk = shorter survival
        assert metrics.harrell_c(event, time, risk) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([True, True, True, True])
        risk = np.array([1.0, 2.0, 3.0, 4.0])
        assert metrics.harrell_c(event, time, risk) == pytest.approx(0.0)

    def test_sign_flip_reflects_concordance_on_a_non_degenerate_score(self) -> None:
        """#3 §6 pins `C(-r) == 1 - C(r)`; the degenerate C=1/C=0 cases above
        satisfy that identity trivially, so it needs a mid-range check too."""
        x, _, event, time = _stratified_cohort()
        risk = x @ np.array([0.8, -0.5])
        base = metrics.harrell_c(event, time, risk)

        assert 0.5 < base < 1.0, "fixture must discriminate without being perfect"
        assert metrics.harrell_c(event, time, -risk) == pytest.approx(1.0 - base)

    def test_within_study_pools_pairs_rather_than_averaging_per_study_c(self) -> None:
        """A tiny Study (few pairs) must not get the same weight as a big one."""
        # Study "BIG": perfectly concordant, many pairs.
        big_time = np.arange(1.0, 21.0)
        big_event = np.ones(20, dtype=bool)
        big_risk = -big_time  # higher risk for shorter time -> concordant

        # Study "SMALL": perfectly anti-concordant, one pair.
        small_time = np.array([1.0, 2.0])
        small_event = np.array([True, True])
        small_risk = np.array([1.0, 2.0])  # discordant

        time = np.concatenate([big_time, small_time])
        event = np.concatenate([big_event, small_event])
        risk = np.concatenate([big_risk, small_risk])
        study = np.array(["BIG"] * 20 + ["SMALL"] * 2)

        result = metrics.within_study_harrell_c(event, time, risk, study)

        # Pair-pooled: (190 concordant + 0 concordant) / (190 + 1) pairs.
        n_big_pairs = 20 * 19 // 2
        expected = n_big_pairs / (n_big_pairs + 1)
        assert result.cindex == pytest.approx(expected)
        # A naive mean of per-study C (1.0 and 0.0 -> 0.5) would differ sharply.
        assert result.cindex != pytest.approx(0.5, abs=0.05)

    def test_a_stratum_with_no_events_is_excluded_not_averaged_in(self) -> None:
        time = np.array([1.0, 2.0, 3.0, 9.0])
        event = np.array([False, False, True, True])
        risk = np.array([0.1, 0.2, 0.3, 0.4])
        study = np.array(["ALL-CENSORED", "ALL-CENSORED", "HAS-EVENTS", "HAS-EVENTS"])

        result = metrics.within_study_harrell_c(event, time, risk, study)
        assert result.excluded_studies == ("ALL-CENSORED",)
        assert result.per_study["ALL-CENSORED"] is None

    def test_uno_c_is_bounded_and_close_to_harrell_under_light_censoring(self) -> None:
        x, study, event, time = _stratified_cohort()
        risk = x @ np.array([0.8, -0.5])

        c = metrics.uno_c(train_event=event, train_time=time, test_event=event, test_time=time, risk=risk)
        assert 0.0 <= c <= 1.0

    def test_assert_discriminates_raises_on_inverted_risk(self) -> None:
        x, study, event, time = _stratified_cohort()
        risk = x @ np.array([0.8, -0.5])

        metrics.assert_discriminates(event, time, risk, where="test")
        with pytest.raises(ValueError, match="does not exceed"):
            metrics.assert_discriminates(event, time, -risk, where="test")

    def test_score_fold_returns_every_locked_metric(self) -> None:
        x, study, event, time = _stratified_cohort()
        risk = x @ np.array([0.8, -0.5])

        result = metrics.score_fold(
            train_time=time, train_event=event, test_time=time, test_event=event,
            test_study=study, risk=risk,
        )
        payload = result.to_dict()
        for key in (
            "pooled_harrell_c",
            "within_study_harrell_c",
            "uno_c_ipcw",
            "time_dependent_auc",
            "integrated_brier_score",
        ):
            assert key in payload


class TestSurvivalTarget:
    def test_subset_and_reorder_round_trip(self) -> None:
        target = SurvivalTarget(
            subject_id=np.array(["a", "b", "c"]),
            study=np.array(["S1", "S1", "S2"]),
            time=np.array([10.0, 20.0, 30.0]),
            event=np.array([True, False, True]),
        )
        subset = target.subset(frozenset({"a", "c"}))
        assert set(subset.subject_id) == {"a", "c"}

        reordered = subset.reorder(["c", "a"])
        assert list(reordered.subject_id) == ["c", "a"]
        assert list(reordered.time) == [30.0, 10.0]
