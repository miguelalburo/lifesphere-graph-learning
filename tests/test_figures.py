"""The figure layer's two derivations, which are the only places it can lie.

Drawing is not tested here — a mispositioned label is visible, and the last step
of `gl_lifesphere/figures/` is looking at the rendered file. What is not visible
is a Study whose plotted mean silently folded in a `None`: an excluded fold
(#3 §4's "no comparable pair" rule) read as 0.0 would drag a Study toward chance
and look exactly like a real result, which is the class `tests/README.md` puts
first. Both tests below are that class.
"""

from __future__ import annotations

import pytest

from gl_lifesphere.diagnostics.recorded import RecordedModel
from gl_lifesphere.figures import inputs


def _model(per_study_by_fold: list[dict[str, float | None]]) -> RecordedModel:
    """A recorded model carrying nothing but the per-Study block each fold wrote."""
    return RecordedModel(
        model="model3_graph",
        folds=tuple({"metrics": {"per_study_harrell_c": block}} for block in per_study_by_fold),
        config={},
    )


class TestPerStudyMeans:
    def test_averages_only_the_folds_that_scored_the_study(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `None` fold is dropped, not read as a zero.

        TCGA-TGCT is the live instance of this: 4 events across 133 Subjects, so
        some folds hold no comparable pair. Averaging over 3 of 5 is right;
        averaging 0.0 into the other 2 would print ~0.4 and read as a Study the
        model actively gets wrong.
        """
        monkeypatch.setattr(
            inputs.recorded,
            "load_model",
            lambda model: _model(
                [
                    {"TCGA-TGCT": 0.8, "TCGA-BRCA": 0.7},
                    {"TCGA-TGCT": None, "TCGA-BRCA": 0.8},
                    {"TCGA-TGCT": 0.6, "TCGA-BRCA": 0.9},
                ]
            ),
        )

        means = inputs.load_per_study("model3_graph")

        assert means["TCGA-TGCT"] == pytest.approx(0.7)  # not (0.8 + 0 + 0.6) / 3
        assert means["TCGA-BRCA"] == pytest.approx(0.8)

    def test_a_study_scored_in_no_fold_is_absent_rather_than_defaulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            inputs.recorded,
            "load_model",
            lambda model: _model([{"TCGA-KICH": None}, {"TCGA-KICH": None}]),
        )

        assert "TCGA-KICH" not in inputs.load_per_study("model3_graph")


class TestFoldsScoringStudy:
    def test_counts_the_folds_behind_each_mean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fig. 4 marks a Study whose mean is over fewer than 5 folds."""
        monkeypatch.setattr(
            inputs.recorded,
            "load_model",
            lambda model: _model(
                [
                    {"TCGA-TGCT": 0.8, "TCGA-BRCA": 0.7},
                    {"TCGA-TGCT": None, "TCGA-BRCA": 0.8},
                    {"TCGA-TGCT": 0.6, "TCGA-BRCA": 0.9},
                ]
            ),
        )

        counted = inputs.folds_scoring_study("model3_graph")

        assert counted == {"TCGA-TGCT": 2, "TCGA-BRCA": 3}
