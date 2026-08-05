"""Arm 3 — GL/GNN models over the graph constructions.

Encoders produce a per-Subject embedding; the risk head and survival loss sit
downstream in `gl_lifesphere.survival`. Which encoder is valid depends on the
construction's task family — graph-level readout (GIN/GraphSAGE + pooling) for
per-Subject subgraphs, node-level for the patient similarity network,
type-aware (HGT) for the heterogeneous construction.
"""
