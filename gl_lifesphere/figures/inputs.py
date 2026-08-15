"""Reading the plotted numbers back out of `results/metrics/`, and nothing else.

Every figure in this package is a view of a file that already exists. Nothing
here trains, scores, or re-derives a metric: a model's per-fold numbers come from
`recorded.load_model`, the same reader the trust gate compares against, so a figure
and a verdict cannot disagree about what a model scored. If a plotted number is
wrong, it is wrong in `results/metrics/`, and that is the only place to fix it.

The one quantity assembled here rather than read is a **per-Study mean**: each
fold records `per_study_harrell_c` over its own held-out fifth, and Fig. 4 wants
one number per Study, so the folds are averaged. `None` entries are dropped
rather than treated as zero — a Study with no comparable pair in a fold (#3 §4's
exclusion rule; TCGA-TGCT carries 4 events across 133 Subjects) contributes no
value to the mean, and a Study excluded in every fold is reported as excluded
rather than plotted at some default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..diagnostics import recorded
from ..evaluation.compare import PairedDelta, paired_delta
from ..extract.connection import REPO_ROOT
from ..extract.pipeline import load_cohort_labels
from ..survival import metrics as survival_metrics

METRICS_DIR = REPO_ROOT / "results" / "metrics"
HEADLINE = survival_metrics.HEADLINE

# #3 §5: below this, 5 folds on this cohort cannot separate two models however
# tight the paired interval looks. Drawn on every delta panel for that reason.
RESOLUTION_LIMIT = 0.02


@dataclass(frozen=True)
class ModelScores:
    """One model's headline metric across the folds, as recorded."""

    key: str
    per_fold: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.per_fold))

    @property
    def std(self) -> float:
        return float(np.std(self.per_fold, ddof=0))


def load_model_scores(model: str, *, metric: str = HEADLINE) -> ModelScores:
    """A model's per-fold metric, in fold order."""
    return ModelScores(key=model, per_fold=tuple(recorded.load_model(model).metric(metric)))


def load_per_study(model: str) -> dict[str, float]:
    """Per-Study Harrell C, averaged over the folds that scored the Study."""
    folds = recorded.load_model(model).folds
    collected: dict[str, list[float]] = {}
    for fold in folds:
        fold_metrics = fold["metrics"]
        assert isinstance(fold_metrics, dict)
        per_study = fold_metrics["per_study_harrell_c"]
        assert isinstance(per_study, dict)
        for study, value in per_study.items():
            if value is None:
                continue
            collected.setdefault(study, []).append(float(value))
    return {study: float(np.mean(values)) for study, values in collected.items()}


def folds_scoring_study(model: str) -> dict[str, int]:
    """How many of the 5 folds contributed a value for each Study.

    Fewer than 5 means the Study held no comparable pair in some fold, which is
    what a Study with almost no events looks like (TCGA-TGCT, 4 events across
    133 Subjects). Fig. 4 marks those Studies rather than plotting a mean over
    an unstated number of folds.
    """
    counted: dict[str, int] = {}
    for fold in recorded.load_model(model).folds:
        fold_metrics = fold["metrics"]
        assert isinstance(fold_metrics, dict)
        per_study = fold_metrics["per_study_harrell_c"]
        assert isinstance(per_study, dict)
        for study, value in per_study.items():
            counted[study] = counted.get(study, 0) + (value is not None)
    return counted


def study_sizes() -> dict[str, tuple[int, int]]:
    """`{study: (n_subjects, n_events)}` over the frozen OS cohort."""
    labels = load_cohort_labels()
    grouped = labels.groupby("studyId")["event"]
    return {
        str(study): (int(grouped.size()[study]), int(grouped.sum()[study]))
        for study in grouped.size().index
    }


def delta(treatment: str, reference: str, *, metric: str = HEADLINE) -> PairedDelta:
    """`treatment - reference`, paired fold by fold, with #3 §5's caveat attached."""
    return paired_delta(
        list(load_model_scores(treatment, metric=metric).per_fold),
        list(load_model_scores(reference, metric=metric).per_fold),
    )


def read_json(*parts: str) -> dict:
    """One results file, by path relative to `results/metrics/`."""
    path = Path(METRICS_DIR, *parts)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — the figure it feeds cannot be drawn. "
            f"Produce it first (see results/README.md for which command writes it)."
        )
    with path.open() as handle:
        return json.load(handle)
