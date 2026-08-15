"""Train model 2 across all 5 folds and write metrics.

    python -m gl_lifesphere.models.tabular [--config PATH] [--out DIR]

Every run should be traceable to a config in `experiments/configs/`
(`experiments/README.md`); results land in `results/metrics/`, gitignored by
default per `results/README.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...extract.connection import REPO_ROOT
from ..training import summarise_folds
from .train import MODEL, FoldResult, TabularModelConfig, run_all_folds

DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "model2_tabular.json"
DEFAULT_OUT = REPO_ROOT / "results" / "metrics" / "model2_tabular"

def main(config_path: Path = DEFAULT_CONFIG, out_dir: Path = DEFAULT_OUT) -> list[FoldResult]:
    payload = json.loads(config_path.read_text())
    config = TabularModelConfig.from_dict(payload)
    results = run_all_folds(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (out_dir / f"fold_{result.fold}.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str) + "\n"
        )

    summary = {
        "model": MODEL,
        "config": payload,
        "folds": summarise_folds([r.fold_metrics for r in results]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model 2 (flattened/tabular control) across all 5 folds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.config, args.out)
