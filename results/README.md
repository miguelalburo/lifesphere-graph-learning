# results/

Experiment outputs. Contents are gitignored by default; promote a specific file
with `git add -f` when it is one the write-up actually cites.

| Directory  | Holds                                                                    |
| ---------- | ------------------------------------------------------------------------ |
| `metrics/` | Per-run scores (C-index, AUC, Brier), pooled and broken down by Study.    |
| `figures/` | Plots generated from those metrics.                                       |

Each run should be traceable to a config in `experiments/configs/`.

## `metrics/trust_gate/` — read this before any arm's number

`metrics/arm{1,2,3}_*/` hold the arms. `metrics/trust_gate/` holds #13's three
controls, and it sits beside them deliberately: encoder doc §7 puts these checks
"before reading any C-index as a result", so a table quoted without them is a
table nobody has earned yet.

| File                      | Holds                                                                 |
| ------------------------- | --------------------------------------------------------------------- |
| `structure_ablation.json` | Arm 3 retrained with every `edge_index` emptied.                      |
| `degree_probe.json`       | Cox on accrual counts alone, decomposed covariate by covariate.       |
| `label_shuffle.json`      | Every arm retrained on a permuted label, under two permutation schemes. |
| `gate.json`               | The three PASS/FAIL verdicts, the declared thresholds, and the reading each verdict implies. |

`gate.json` is a pure function of the other three and is rebuilt from them by
`python -m gl_lifesphere.diagnostics --only gate`, so a verdict can be re-read
without retraining anything. **A PASS means a control failed the way a control
is supposed to — it says nothing about whether a result is good.**

## `metrics/relation_split/` — the follow-up the gate raised, not a fourth verdict

`relation_split.json` is #15: arm 3 re-run with `split_primary_relation=False`
and paired fold-for-fold against the recorded split run and against arm 2. It
answers whether arm 3's +0.0180 margin is the secondary-diagnosis channel #11's
relation split makes structurally visible, or message passing.

It sits in its own directory on purpose. `gate.json` is assembled from exactly
the three files above, so a follow-up cannot rewrite the verdict that prompted
it — and this probe's vocabulary (`ATTRIBUTED` / `NOT ATTRIBUTED` / `REVERSED`)
is deliberately not PASS/FAIL for the same reason. Produced by
`python -m gl_lifesphere.diagnostics.relation_split`.
