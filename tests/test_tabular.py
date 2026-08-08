"""End-to-end mechanics test for arm 2 (#10).

Not a result test — a small, fast, fully synthetic cohort exercising the whole
two-stage pipeline (`fit_feature_contract` -> encoder training with early
stopping -> penalty selection -> shared decoder -> `score_fold`) so a wiring
bug between any two of those pieces fails here rather than 25 minutes into a
real 5-fold run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gl_lifesphere.evaluation.splits import FoldSplit
from gl_lifesphere.models.tabular import TabularArmConfig, run_fold
from gl_lifesphere.survival.targets import SurvivalTarget

N_PER_STUDY = 30
STUDIES = ("STUDY-A", "STUDY-B", "STUDY-C")


@pytest.fixture
def synthetic_cohort() -> tuple[pd.DataFrame, SurvivalTarget, FoldSplit]:
    rng = np.random.default_rng(0)
    n = N_PER_STUDY * len(STUDIES)

    subject_id = np.array([f"SUBJ-{i:03d}" for i in range(n)])
    study = np.repeat(STUDIES, N_PER_STUDY)

    stage = rng.integers(1, 5, size=n).astype(float)
    age = rng.normal(60, 10, size=n)
    proportions = rng.dirichlet(alpha=(2, 2, 2), size=n)

    raw = pd.DataFrame(
        {
            "subjectId": subject_id,
            "studyId": study,
            "sexAtBirth": rng.choice(["male", "female"], size=n),
            "race": rng.choice(["white", "black or african american", "asian"], size=n),
            "ageAtIndexYears": age,
            "stageOrdinal": stage,
            "ageAtDiagnosisYears": age,
            "conditionSubtype": rng.choice(["subtype-1", "subtype-2"], size=n),
            "conditionName": rng.choice(["condition-1", "condition-2"], size=n),
            "p_primary_tumor": proportions[:, 0],
            "p_blood_derived_normal": proportions[:, 1],
            "p_solid_tissue_normal": proportions[:, 2],
        }
    )

    # A genuine (if weak) signal: higher stage -> shorter time. Scaled to
    # thousands of days, like the real cohort's `timeToEventDays`, so the
    # locked 1/2/3-year horizons (`metrics.DEFAULT_HORIZONS_DAYS`) fall inside
    # this synthetic cohort's follow-up range.
    risk = 0.5 * stage - 0.2 * (age - 60) / 10
    latent = rng.exponential(scale=np.exp(-risk) * 3000)
    censor = rng.exponential(scale=4000, size=n)
    time = np.ceil(np.minimum(latent, censor))
    event = latent <= censor

    target = SurvivalTarget(subject_id=subject_id, study=study, time=time, event=event)

    # A simple contiguous 60/20/20 split, stratified by construction since each
    # Study occupies a fixed contiguous block of `N_PER_STUDY`.
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    for s in range(len(STUDIES)):
        block = subject_id[s * N_PER_STUDY : (s + 1) * N_PER_STUDY]
        train_ids.update(block[:18])
        val_ids.update(block[18:24])
        test_ids.update(block[24:])

    split = FoldSplit(
        train=frozenset(train_ids), val=frozenset(val_ids), test=frozenset(test_ids)
    )
    return raw, target, split


class TestRunFold:
    def test_runs_end_to_end_and_returns_populated_metrics(
        self, synthetic_cohort: tuple[pd.DataFrame, SurvivalTarget, FoldSplit]
    ) -> None:
        raw, target, split = synthetic_cohort
        config = TabularArmConfig(hidden_dims=(8,), d=4, max_epochs=15, patience=5, seed=0)

        result = run_fold(0, raw=raw, targets=target, split=split, config=config)

        assert result.fold_metrics.n_test == len(split.test)
        assert 0.0 <= result.fold_metrics.pooled_harrell_c <= 1.0
        assert 0.0 <= result.fold_metrics.within_study.cindex <= 1.0
        assert result.fold_metrics.integrated_brier is not None
        assert result.penalty_selection.chosen_penalizer in config.penalty_grid

    def test_runs_the_two_locked_diagnostics_alongside_the_primary_result(
        self, synthetic_cohort: tuple[pd.DataFrame, SurvivalTarget, FoldSplit]
    ) -> None:
        """#3 §3 pre-declares two controls as "not as options": a frozen
        randomly-initialised encoder, and the end-to-end score that is a
        by-product of stage one. Both must run on every fold."""
        raw, target, split = synthetic_cohort
        config = TabularArmConfig(hidden_dims=(8,), d=4, max_epochs=15, patience=5, seed=0)

        result = run_fold(0, raw=raw, targets=target, split=split, config=config)

        assert 0.0 <= result.random_init.fold_metrics.pooled_harrell_c <= 1.0
        assert result.random_init.penalty_selection.chosen_penalizer in config.penalty_grid
        # The end-to-end score bypasses the shared decoder's Breslow baseline
        # hazard entirely, so it never carries a Brier score.
        assert 0.0 <= result.end_to_end.pooled_harrell_c <= 1.0
        assert result.end_to_end.integrated_brier is None

    def test_random_init_diagnostic_is_reproducible_across_runs(
        self, synthetic_cohort: tuple[pd.DataFrame, SurvivalTarget, FoldSplit]
    ) -> None:
        """"Requires no training run at all" (#3 §3) — it must be exactly the
        pre-training weights, not an arbitrary untracked initialisation."""
        raw, target, split = synthetic_cohort
        config = TabularArmConfig(hidden_dims=(8,), d=4, max_epochs=5, patience=5, seed=0)

        first = run_fold(0, raw=raw, targets=target, split=split, config=config)
        second = run_fold(0, raw=raw, targets=target, split=split, config=config)

        assert first.random_init.fold_metrics.pooled_harrell_c == pytest.approx(
            second.random_init.fold_metrics.pooled_harrell_c
        )

    def test_result_serialises_to_json_safe_dict(
        self, synthetic_cohort: tuple[pd.DataFrame, SurvivalTarget, FoldSplit]
    ) -> None:
        import json

        raw, target, split = synthetic_cohort
        config = TabularArmConfig(hidden_dims=(8,), d=4, max_epochs=10, patience=5, seed=0)

        result = run_fold(0, raw=raw, targets=target, split=split, config=config)
        # `default=str` covers the numpy/study-key edge cases; this just pins
        # that nothing raises TypeError on an un-serialisable object.
        json.dumps(result.to_dict(), default=str)

    def test_is_deterministic_given_a_fixed_seed(
        self, synthetic_cohort: tuple[pd.DataFrame, SurvivalTarget, FoldSplit]
    ) -> None:
        raw, target, split = synthetic_cohort
        config = TabularArmConfig(hidden_dims=(8,), d=4, max_epochs=10, patience=5, seed=0)

        first = run_fold(0, raw=raw, targets=target, split=split, config=config)
        second = run_fold(0, raw=raw, targets=target, split=split, config=config)

        assert first.fold_metrics.pooled_harrell_c == pytest.approx(
            second.fold_metrics.pooled_harrell_c
        )
