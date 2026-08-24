# P8.1.5 — Round-Trip-Consistent Full-SMILES Policy

P8.1.5 tests the video-consistency analogy as a training constraint, not an
inference reranker.  A single checkpoint, autoregressive decoder, softmax, and
full-molecule SMILES vocabulary maps either an empty source plus property
program to a new molecule or a source molecule plus property program to its
edit.  There is no router, materializer, output-property reranker, or alternate
interpreter.

The source-free condition layout is exactly the P1 Direct-SMILES layout.  All
P1 tensors are frozen, and only source-conditioned residual modules train, so
P1 de-novo behavior is protected by construction and audited after evaluation.

R1 and R2 independently initialize the same source modules from the same P1
Group-RL checkpoint and use the same seed, optimizer, training steps, raw
decoding, and frozen evaluation subsets.  R1 uses forward supervision only
(`cycle_weight=0`).  R2's sole scientific change is inverse-program
round-trip supervision (`cycle_weight=1`): for each edit `x --c--> y`, the
same decoder also learns `y --inverse(c)--> x`.  R2 is queued with `afterany`
and therefore is not cancelled by a poor R1.

Both arms report the P6 hard de-novo strata and MolEdit Table1 edit subset with
raw `k=1,8,20`, candidate-level success, validity, uniqueness, and identity-copy
fraction.  The audit also reports strict non-identity success and the raw
source-Tanimoto distribution, so an apparent consistency gain caused by exact
copying cannot pass as successful editing.  Candidate order is untouched and
no property-based selection is performed; R1 is the matched no-cycle raw20
control.
