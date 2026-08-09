"""Tests for the shared scoring entry point (#3 §5, §6; implemented for #9).

`tests/test_stack.py` already pins that `sksurv` and `torchsurv` agree with
each other and with the risk-score sign convention at the *library* level.
This file pins the module built on top of them: pair-pooled aggregation
across Studies is a sum-then-divide and not an average, a Study with no
comparable pair is excluded rather than silently dropped or averaged in as
0.5, and the discrimination guard actually raises on an inverted score.

Nothing here touches the live instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from gl_lifesphere.survival import metrics

# Same shape as `tests/test_stack.py`'s cohort: a planted linear effect, a
# per-stratum hazard scale, integer day-count times. Reused (not shared via
# import) because this file must stand alone from the stack's contract tests.
PLANTED_BETA = np.array([0.8, -0.5, 0.2])


def _stratified_cohort(
    n: int = 400, n_strata: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (risk, event, time, study) for a synthetic multi-Study cohort."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(PLANTED_BETA)))
    stratum = rng.integers(0, n_strata, size=n)
    latent = rng.exponential(scale=np.exp(-(x @ PLANTED_BETA)) * (1.0 + stratum))
    censor = rng.exponential(scale=2.0 * (1.0 + stratum), size=n)
    time = np.ceil(np.minimum(latent, censor) * 60.0)
    event = latent <= censor
    risk = x @ PLANTED_BETA
    study = np.array([f"STUDY-{s}" for s in stratum])
    return risk, event, time, study


class TestHarrellC:
    def test_matches_a_hand_countable_example(self) -> None:
        # 4 Subjects, all events, no ties: risk strictly decreasing with time
        # (higher risk -> shorter survival) is the perfectly concordant case.
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([True, True, True, True])
        assert metrics.harrell_c(event, time, risk) == pytest.approx(1.0)

    def test_reversing_risk_gives_the_worst_possible_score(self) -> None:
        risk = np.array([1.0, 2.0, 3.0, 4.0])
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([True, True, True, True])
        assert metrics.harrell_c(event, time, risk) == pytest.approx(0.0)

    def test_scale_and_monotone_transform_invariance(self) -> None:
        risk, event, time, _ = _stratified_cohort()
        base = metrics.harrell_c(event, time, risk)
        assert metrics.harrell_c(event, time, 3.0 * risk + 7.0) == pytest.approx(base)
        assert metrics.harrell_c(event, time, np.exp(risk)) == pytest.approx(base)

    def test_sign_reflects_concordance(self) -> None:
        risk, event, time, _ = _stratified_cohort()
        base = metrics.harrell_c(event, time, risk)
        assert base > 0.6, "fixture must discriminate or the flip below is invisible"
        assert metrics.harrell_c(event, time, -risk) == pytest.approx(1.0 - base)

    def test_no_comparable_pairs_raises(self) -> None:
        event = np.array([False, False])
        time = np.array([1.0, 2.0])
        risk = np.array([0.1, 0.2])
        with pytest.raises(Exception):  # sksurv.exceptions.NoComparablePairException
            metrics.harrell_c(event, time, risk)


class TestWithinStudyPairPooling:
    """The headline metric: sum concordant/comparable pairs, then divide once."""

    def test_pooling_is_not_the_average_of_per_study_c(self) -> None:
        # Study A: 3 comparable pairs, all concordant -> C_A = 1.0
        # Study B: 10 comparable pairs, 4 concordant -> C_B = 0.4
        # Mean of the two Cs is 0.7; pair-pooled is (3+4)/(3+10) ~= 0.538.
        study = np.array(["S-A"] * 3 + ["S-B"] * 5)
        risk = np.array([3.0, 2.0, 1.0] + [1.0, 2.0, 3.0, 4.0, 4.9])
        time = np.array([1.0, 2.0, 3.0] + [1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([True] * 3 + [True] * 5)

        result = metrics.within_study_harrell_c(event, time, risk, study)
        per_study_mean = np.mean([c for c in result.per_study.values() if c is not None])
        assert result.cindex != pytest.approx(per_study_mean)

    def test_cross_study_pairs_are_never_counted(self) -> None:
        """A Subject in Study A must never be compared against one in Study B."""
        study = np.array(["S-A", "S-A", "S-B", "S-B"])
        risk = np.array([10.0, -10.0, 10.0, -10.0])
        time = np.array([1.0, 1000.0, 1.0, 1000.0])
        event = np.array([True, True, True, True])

        result = metrics.within_study_harrell_c(event, time, risk, study)
        # Each 2-Subject Study has exactly 1 comparable pair; a cross-Study
        # count would inflate the total to 6 (4 choose 2).
        assert result.per_study["S-A"] == pytest.approx(1.0)
        assert result.per_study["S-B"] == pytest.approx(1.0)
        assert result.cindex == pytest.approx(1.0)

    def test_a_study_with_no_comparable_pair_is_excluded_not_dropped_silently(self) -> None:
        """The only event happens after every censoring time in the Study."""
        study = np.array(["S-A", "S-A", "S-A", "S-B", "S-B"])
        risk = np.array([0.1, 0.2, 0.3, 0.0, 1.0])
        time = np.array([1.0, 2.0, 9.0, 5.0, 6.0])
        event = np.array([False, False, True, True, False])

        result = metrics.within_study_harrell_c(event, time, risk, study)
        assert "S-A" in result.excluded_studies
        assert result.per_study["S-A"] is None
        assert "S-B" not in result.excluded_studies

    def test_pooled_c_raises_if_nothing_was_scoreable(self) -> None:
        study = np.array(["S-A"])
        risk = np.array([0.1])
        time = np.array([1.0])
        event = np.array([True])
        with pytest.raises(Exception):
            metrics.within_study_harrell_c(event, time, risk, study)


class TestUnoC:
    def test_lower_than_harrell_under_heavy_censoring(self) -> None:
        """#3 §5 measures Harrell 0.013 above Uno on the real cohort; this fixture
        (identical to `test_stack.py`'s) resolves a smaller but real margin (~0.009)."""
        risk, event, time, _ = _stratified_cohort()
        assert 0.3 < 1.0 - event.mean() < 0.6, "fixture must be meaningfully censored"

        harrell = metrics.harrell_c(event, time, risk)
        uno = metrics.uno_c(train_event=event, train_time=time, test_event=event, test_time=time, risk=risk)
        assert harrell - uno > 5e-3

    def test_coincides_with_harrell_when_nothing_is_censored(self) -> None:
        risk, _, time, _ = _stratified_cohort(seed=3)
        event = np.ones_like(time, dtype=bool)
        harrell = metrics.harrell_c(event, time, risk)
        uno = metrics.uno_c(train_event=event, train_time=time, test_event=event, test_time=time, risk=risk)
        assert uno == pytest.approx(harrell, abs=1e-3)


class TestTimeDependentAUC:
    def test_returns_one_value_per_horizon_in_unit_interval(self) -> None:
        risk, event, time, _ = _stratified_cohort()
        # `sksurv` requires every horizon strictly inside the test follow-up
        # range, so these are chosen as quantiles of the fixture's own times
        # rather than the module's day-scale defaults, which this fixture's
        # arbitrary time units do not match.
        horizons = tuple(np.quantile(time, [0.25, 0.5, 0.75]))
        result = metrics.time_dependent_auc(
            train_event=event,
            train_time=time,
            test_event=event,
            test_time=time,
            risk=risk,
            horizons=horizons,
        )
        assert set(result) == set(horizons)
        assert all(0.0 <= value <= 1.0 for value in result.values())

    def test_default_horizons_are_one_two_three_years_in_days(self) -> None:
        assert metrics.DEFAULT_HORIZONS_DAYS == (365.0, 730.0, 1095.0)


class TestIntegratedBrier:
    def test_a_well_calibrated_estimate_beats_a_poorly_calibrated_one(self) -> None:
        rng = np.random.default_rng(5)
        n, n_times = 200, 10
        time = rng.uniform(0.1, 5.0, size=n)
        event = rng.random(n) < 0.6
        horizons = tuple(np.linspace(0.2, 4.5, n_times))

        # "Well calibrated": survival probability actually decays past each
        # Subject's own time. "Poorly calibrated": everyone predicted to
        # survive with near-certainty regardless of their outcome.
        horizon_array = np.array(horizons)
        good = np.clip(1.0 - horizon_array[None, :] / (time[:, None] + 1e-6), 0.0, 1.0)
        bad = np.full((n, n_times), 0.99)

        good_score = metrics.integrated_brier(
            train_event=event,
            train_time=time,
            test_event=event,
            test_time=time,
            survival_probabilities=good,
            horizons=horizons,
        )
        bad_score = metrics.integrated_brier(
            train_event=event,
            train_time=time,
            test_event=event,
            test_time=time,
            survival_probabilities=bad,
            horizons=horizons,
        )
        assert 0.0 <= good_score <= 1.0
        assert good_score < bad_score


class TestAssertDiscriminates:
    def test_passes_for_a_discriminating_score(self) -> None:
        risk, event, time, _ = _stratified_cohort()
        metrics.assert_discriminates(event, time, risk, where="test")

    def test_raises_for_an_inverted_score(self) -> None:
        risk, event, time, _ = _stratified_cohort()
        with pytest.raises(ValueError, match="test"):
            metrics.assert_discriminates(event, time, -risk, where="test")

    def test_raises_for_chance_level_score(self) -> None:
        rng = np.random.default_rng(6)
        n = 200
        risk = rng.normal(size=n)
        time = rng.exponential(size=n)  # independent of risk
        event = rng.random(n) < 0.5
        with pytest.raises(ValueError):
            metrics.assert_discriminates(event, time, risk, where="test", min_c=0.55)


class TestScoreFold:
    def test_bundles_every_metric_for_one_fold(self) -> None:
        risk, event, time, study = _stratified_cohort()
        result = metrics.score_fold(
            train_time=time,
            train_event=event,
            test_time=time,
            test_event=event,
            test_study=study,
            risk=risk,
        )
        assert result.n_test == len(time)
        assert result.n_events_test == int(event.sum())
        assert result.integrated_brier is None  # no survival_probabilities supplied
        payload = result.to_dict()
        assert "within_study_harrell_c" in payload
        assert "uno_c_ipcw" in payload
        td_auc = payload["time_dependent_auc"]
        assert isinstance(td_auc, dict)
        assert set(td_auc) == {"365.0", "730.0", "1095.0"}

    def test_integrated_brier_present_when_survival_probabilities_supplied(self) -> None:
        risk, event, time, study = _stratified_cohort()
        # Rescale time onto a day-like scale so the module's day-based default
        # horizons (365/730/1095) fall inside follow-up.
        time = time * 20.0 + 1.0
        horizons = metrics.DEFAULT_HORIZONS_DAYS
        survival_probabilities = np.tile(
            np.linspace(0.9, 0.1, len(horizons)), (len(time), 1)
        )
        result = metrics.score_fold(
            train_time=time,
            train_event=event,
            test_time=time,
            test_event=event,
            test_study=study,
            risk=risk,
            survival_probabilities=survival_probabilities,
            horizons=horizons,
        )
        assert result.integrated_brier is not None
        assert 0.0 <= result.integrated_brier <= 1.0


class TestIpcwCoverage:
    """The Brier score and the time-dependent AUC weight each held-out Subject
    by `1/G(y_i)`, with `G` fit on the training fold. A Subject who outlives
    that fold's follow-up has no such weight, and sksurv raises rather than
    extrapolating.

    This project sits right on that boundary: with the real labels, fold 0
    already holds a test Subject past its training follow-up and scores anyway,
    purely because the last training observation happens to be censored. #13's
    label shuffle moved the labels and the same fold started raising — so the
    crash was revealed by the permutation, not caused by it, and it would have
    surfaced later on a different endpoint or seed with no warning.
    """

    @staticmethod
    def _train(last_is_event: bool) -> tuple[np.ndarray, np.ndarray]:
        event = np.array([True, False, True, False, last_is_event])
        time = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        return event, time

    def test_a_test_time_inside_the_training_follow_up_is_always_covered(self) -> None:
        event, time = self._train(last_is_event=True)

        assert metrics.brier_ipcw_is_defined(event, time, np.array([10.0, 45.0, 50.0]))

    def test_outliving_the_training_fold_is_uncovered_when_it_ended_in_an_event(self) -> None:
        event, time = self._train(last_is_event=True)

        assert not metrics.brier_ipcw_is_defined(event, time, np.array([10.0, 60.0]))

    def test_outliving_it_is_still_covered_when_the_last_observation_was_censored(self) -> None:
        """`G` has already reached 0 there, so the weight is legitimately 0
        rather than undefined — which is the only reason the real fold 0 scores."""
        event, time = self._train(last_is_event=False)

        assert metrics.brier_ipcw_is_defined(event, time, np.array([10.0, 60.0]))

    def test_the_auc_only_needs_event_times_to_be_covered(self) -> None:
        """`cumulative_dynamic_auc` weights through `predict_ipcw`, which
        evaluates `G` at `time[event]` only — so a *censored* Subject beyond the
        training follow-up costs the AUC nothing, while the Brier score, which
        evaluates every test time, cannot weight them. Gating both on the Brier
        condition would suppress an AUC that is perfectly computable."""
        event, time = self._train(last_is_event=True)
        # One held-out Subject past the training follow-up, censored.
        test_time = np.array([10.0, 60.0])
        test_event = np.array([True, False])

        assert not metrics.brier_ipcw_is_defined(event, time, test_time)
        assert metrics.auc_ipcw_is_defined(event, time, test_event, test_time)

    def test_the_auc_refuses_an_event_beyond_the_training_follow_up(self) -> None:
        event, time = self._train(last_is_event=True)

        assert not metrics.auc_ipcw_is_defined(
            event, time, np.array([True, True]), np.array([10.0, 60.0])
        )

    def test_score_fold_skips_only_the_metric_that_cannot_be_weighted(self) -> None:
        """A *censored* Subject past the training follow-up costs the Brier
        score but not the AUC, and `score_fold` must skip only the one that is
        actually undefined. The ranking metrics that carry every headline in
        this project need no censoring estimate at all, so they are always
        returned."""
        risk, event, time, study = _stratified_cohort()
        time = time * 20.0 + 1.0
        # The training fold ends in an event, so `G` never reaches 0 and a test
        # time beyond it is genuinely unweightable.
        train_event = event.copy()
        train_event[np.argmax(time)] = True
        test_time = time.copy()
        test_time[0] = time.max() + 1000.0
        test_event = event.copy()
        test_event[0] = False

        result = metrics.score_fold(
            train_time=time,
            train_event=train_event,
            test_time=test_time,
            test_event=test_event,
            test_study=study,
            risk=risk,
            survival_probabilities=np.tile(
                np.linspace(0.9, 0.1, len(metrics.DEFAULT_HORIZONS_DAYS)), (len(time), 1)
            ),
        )

        assert result.integrated_brier is None
        assert set(result.td_auc) == set(metrics.DEFAULT_HORIZONS_DAYS)
        assert result.ipcw_note is not None
        assert 0.0 <= result.pooled_harrell_c <= 1.0
        assert result.to_dict()["ipcw_note"] == metrics.IPCW_UNDEFINED

    def test_score_fold_skips_both_when_the_uncovered_subject_had_an_event(self) -> None:
        risk, event, time, study = _stratified_cohort()
        time = time * 20.0 + 1.0
        train_event = event.copy()
        train_event[np.argmax(time)] = True
        test_time = time.copy()
        test_time[0] = time.max() + 1000.0
        test_event = event.copy()
        test_event[0] = True

        result = metrics.score_fold(
            train_time=time, train_event=train_event, test_time=test_time,
            test_event=test_event, test_study=study, risk=risk,
        )

        assert result.td_auc == {}
        assert result.integrated_brier is None
        assert result.ipcw_note is not None

    def test_a_covered_fold_records_no_note(self) -> None:
        risk, event, time, study = _stratified_cohort()
        time = time * 20.0 + 1.0

        result = metrics.score_fold(
            train_time=time, train_event=event, test_time=time, test_event=event,
            test_study=study, risk=risk,
        )

        assert result.ipcw_note is None
        assert "ipcw_note" not in result.to_dict()
