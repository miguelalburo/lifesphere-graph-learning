"""Arm 2 — the flattened/tabular representation.

One row per Subject over the same features the graph arm sees (Subject
demographics, primary-Diagnosis stage/age/Condition, Sample-type proportions),
with the edges removed — Intervention is dropped entirely, from every arm, as
an immortal-time leak (#4). Not a baseline in this project's vocabulary — it
is the control that separates "the features carry the signal" from "the
structure carries the signal" (rung 0 of encoder doc §4's ablation ladder).

Rung 0 is "flattened + plain neural network": `network.TabularEncoder` maps
the flattened design matrix to `z`, trained end-to-end on the stratified Efron
Cox loss and then frozen, so `z` passes through the identical
`gl_lifesphere.survival.decoder` every arm is scored through (#3 §3, §8).
"""

from __future__ import annotations

from .network import TabularEncoder
from .train import FoldResult, TabularArmConfig, run_all_folds, run_fold, train_encoder

__all__ = [
    "FoldResult",
    "TabularArmConfig",
    "TabularEncoder",
    "run_all_folds",
    "run_fold",
    "train_encoder",
]
