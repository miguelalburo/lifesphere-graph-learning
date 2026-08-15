"""Feature encoding shared across all three models.

Kept separate from `constructions` and `models` on purpose: the study's claim is
about *structure*, so the graph model and the flattened/tabular model must draw on
the same encoded features. Anything that differs between them should be a
deliberate structural difference, not an incidental encoding one.

Covers categorical encoding (Subject demographics, Diagnosis/Condition,
sampleType), Subject-level aggregation of Sample fan-out into a proportion
vector, and training-fold-only imputation/standardisation. Locked by #4;
Intervention is dropped entirely rather than encoded (immortal-time leak).
"""

from __future__ import annotations

from .contract import FeatureContract, fit_feature_contract
from .raw import assemble_raw_frame, build_raw_frame

__all__ = [
    "FeatureContract",
    "assemble_raw_frame",
    "build_raw_frame",
    "fit_feature_contract",
]
