# tests/

Priority here is the quiet-wrong-answer class of bug rather than coverage:
string-to-numeric casts on `Survival` properties, censoring bookkeeping in
target construction, missing-`Condition` handling, and Study-stratified splits
not leaking Subjects across folds.

Tests must not require the live Neo4j instance — fixture off small recorded
extracts.
