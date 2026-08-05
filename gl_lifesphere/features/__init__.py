"""Feature encoding shared across all three arms.

Kept separate from `constructions` and `models` on purpose: the study's claim is
about *structure*, so the graph arm and the flattened/tabular arm must draw on
the same encoded features. Anything that differs between them should be a
deliberate structural difference, not an incidental encoding one.

Covers categorical encoding (Condition, sampleType, interventionType),
Subject-level aggregation of Sample/Intervention fan-out, and the explicit
missing-data strategy for unlinked Condition.
"""
