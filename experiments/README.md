# experiments/

One config per run in `configs/`, naming the arm, the construction (for graph runs),
the survival endpoint, the split, and the seed. The point of the study is the
comparison across these, so a run is only useful if it is pinned down enough to
line up against the others.

A comparison is only apples-to-apples when the arms share endpoint, split, and
survival loss — vary the representation, hold the rest fixed.
