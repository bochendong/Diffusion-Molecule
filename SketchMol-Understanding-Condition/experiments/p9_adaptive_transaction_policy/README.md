# P9: Adaptive Molecular Transaction Policy

P9 addresses P6--P8's falsified assumption that both empty-graph construction
and source-graph editing should use the same low-level trajectory length. One
decoder instead learns a state-adaptive transaction language:

- from an empty graph it emits a compact molecular literal (SMILES);
- from a source graph it emits a short, typed local transaction.

This is not a model router: there is one checkpoint, decoder, output softmax,
and property-program conditioner. The P1 Group-RL literal path is frozen
parameter-for-parameter. Only newly introduced source-memory and transaction
token parameters are trained, giving exact de-novo retention by construction.

The single-seed gate evaluates the same P6 hard 6p/7p and Table 1 subsets with
raw `k=1,8,20`, without property-aware inference reranking.

P9 follows the mandatory two-round protocol. R1 uses instruction-conditioned
grammar enumeration followed by checkpoint ranking for edits. R2 keeps the
trained checkpoint and every evaluation input fixed, and changes exactly one
factor: edits are sampled directly from the checkpoint's source-only
transaction distribution. Thus a negative R1 still receives a second attempt,
while the comparison also reveals whether R1 depends on an implicit candidate
planner.
