"""Train arm 1 across all 5 folds, write metrics, and run the literature sanity check.

    python -m gl_lifesphere.models.baseline [--config PATH] [--out DIR]

Every run should be traceable to a config in `experiments/configs/`
(`experiments/README.md`); results land in `results/metrics/`, gitignored by
default per `results/README.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...extract.connection import REPO_ROOT
from .sanity_check import pooled_out_of_sample_predictions, run_sanity_check
from .train import BaselineArmConfig, FoldResult, run_all_folds

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "arm1_baseline.json"
DEFAULT_OUT = REPO_ROOT / "results" / "metrics" / "arm1_baseline"

_SUMMARY_KEYS = (
    "pooled_harrell_c",
    "within_study_harrell_c",
    "uno_c_ipcw",
    "integrated_brier_score",
)


def _summarise(results: list[FoldResult]) -> dict[str, object]:
    """Mean/std across folds for the headline metrics, plus the raw per-fold values."""
    summary: dict[str, object] = {}
    for key in _SUMMARY_KEYS:
        values = [r.fold_metrics.to_dict()[key] for r in results]
        present = [float(v) for v in values if isinstance(v, (int, float))]
        summary[key] = {
            "mean": float(np.mean(present)) if present else None,
            "std": float(np.std(present)) if present else None,
            "per_fold": values,
        }
    return summary


def main(config_path: Path = DEFAULT_CONFIG, out_dir: Path = DEFAULT_OUT) -> list[FoldResult]:
    payload = json.loads(config_path.read_text())
    config = BaselineArmConfig.from_dict(payload)
    results = run_all_folds(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (out_dir / f"fold_{result.fold}.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str) + "\n"
        )

    sanity = run_sanity_check(
        pooled_out_of_sample_predictions([r.test_predictions for r in results])
    )

    summary = {
        "arm": "arm1_baseline",
        "config": payload,
        "folds": _summarise(results),
        "sanity_check": [s.to_dict() for s in sanity],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    for check in sanity:
        if check.in_band is None:
            flag = "UNSCOREABLE"
        elif check.in_band:
            flag = "OK"
        else:
            flag = "OUT OF BAND"
        uno_display = f"{check.observed_c:.4f}" if check.observed_c is not None else "n/a"
        harrell_display = f"{check.harrell_c:.4f}" if check.harrell_c is not None else "n/a"
        print(
            f"sanity check {check.study}: Uno C={uno_display} (Harrell C={harrell_display}) "
            f"band={check.band} n={check.n_subjects} events={check.n_events} -> {flag}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train arm 1 (clinical baseline) across all 5 folds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.config, args.out)
