"""Evaluation protocol and cross-arm reporting.

- splits      Study-stratified splitting. A naive random split is not adequate:
              Study size varies ~21.5x (BRCA 1,098 vs. CHOL 51), so a pan-cancer
              number is otherwise dominated by the largest Studies.
- compare     Assembles per-arm, per-construction, per-endpoint results into the
              comparison the study is actually about, including per-Study
              breakdowns alongside the pooled figure.
- interpret   Feature/biomarker attribution — the "does LifeSphere's schema
              surface anything beyond known pan-cancer features" question.

Writes to `results/metrics/` and `results/figures/`.
"""
