# P8.1.8: Unified Masked Molecule Policy

P8.1.8 tests a single masked SELFIES denoiser over one fixed molecular canvas.
De-novo generation starts from an all-mask canvas; editing starts from the same
canvas initialized by the source SELFIES with a corrupted span.  Both modes use
the same vocabulary, bidirectional denoiser, output head, iterative refinement
rule, checkpoint, and SELFIES decoder.  There is no task router, output-family
switch, external molecule library, or property-aware inference reranking.

R1 trains once at seed 7 and evaluates raw `k=1,8,20` on the frozen P6 hard
6p/7p and MolEdit Table 1 subsets.  R2 reuses the identical checkpoint and
changes only the source-span corruption fraction: high R1 identity increases
the editable span, otherwise poor source retention decreases it.  R2 always
runs.

