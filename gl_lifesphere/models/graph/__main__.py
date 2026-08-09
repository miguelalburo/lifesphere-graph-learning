"""Train arm 3 across all 5 folds and write metrics.

    python -m gl_lifesphere.models.graph [--config PATH] [--out DIR] [--no-cache]

Every run should be traceable to a config in `experiments/configs/`
(`experiments/README.md`); results land in `results/metrics/`, gitignored by
default per `results/README.md`. `--no-cache` forces the construction to be
rebuilt from `data/interim/` rather than read from
`data/processed/subject_subgraph/` — the cache invalidates itself on the
contract, the options and the builder's own source, so this is a belt-and-braces
switch rather than the normal way to get a correct run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...extract.connection import REPO_ROOT
from .train import FoldResult, GraphArmConfig, iter_folds

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "arm3_graph.json"
DEFAULT_OUT = REPO_ROOT / "results" / "metrics" / "arm3_graph"

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


def main(
    config_path: Path = DEFAULT_CONFIG, out_dir: Path = DEFAULT_OUT, *, use_cache: bool = True
) -> list[FoldResult]:
    """Train all 5 folds, writing each fold's metrics as soon as it finishes.

    Arm 2 writes everything at the end, which is fine at its runtime. This arm
    builds a construction per fold before it trains one, so a run is long enough
    that losing four completed folds to an interruption in the fifth is a real
    cost — and the fold files are the expensive artefact, not the summary, which
    is a pure function of them.
    """
    payload = json.loads(config_path.read_text())
    config = GraphArmConfig.from_dict(payload)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[FoldResult] = []
    for result in iter_folds(config, use_cache=use_cache):
        (out_dir / f"fold_{result.fold}.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str) + "\n"
        )
        results.append(result)

    summary = {"arm": "arm3_graph", "config": payload, "folds": _summarise(results)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train arm 3 (GL/GNN over the raw graph construction) across all 5 folds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    main(args.config, args.out, use_cache=not args.no_cache)
