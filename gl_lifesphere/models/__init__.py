"""The three comparison arms, one subpackage each.

Every model here has the same contract: consume its own representation, emit a
risk score per Subject. The survival loss and metric live in
`gl_lifesphere.survival` and are identical across arms.

- `baseline`  Clinical staging/pathology-only benchmark (classical Cox, RSF).
- `tabular`   Flattened one-row-per-Subject representation, no edges — the
              control that isolates whether graph structure adds anything.
- `graph`     GL/GNN encoders over `gl_lifesphere.constructions`.
"""
