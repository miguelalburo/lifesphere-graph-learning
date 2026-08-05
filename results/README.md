# results/

Experiment outputs. Contents are gitignored by default; promote a specific file
with `git add -f` when it is one the write-up actually cites.

| Directory  | Holds                                                                    |
| ---------- | ------------------------------------------------------------------------ |
| `metrics/` | Per-run scores (C-index, AUC, Brier), pooled and broken down by Study.    |
| `figures/` | Plots generated from those metrics.                                       |

Each run should be traceable to a config in `experiments/configs/`.
