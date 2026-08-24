# P8.1.6 — Macro Group-RL on one full-SMILES policy

P8.1.3 showed that a learned macro payload vocabulary has zero train-only
reachability on held-out de-novo molecules; BRICS factorization further hurt
exact reconstruction. P8.1.6 therefore adds no output token, macro vocabulary,
interpreter, or materializer. A macro is only one terminal reward assigned to
the complete full-SMILES trajectory.

Both rounds start from the same P8.1.4-R1 checkpoint and use the same 256 mixed
rows, seed, Group-RL optimizer and sampling protocol. R1 diagnoses sparse
conjunctive credit with `joint_bottleneck`. R2 runs unconditionally after R1
and changes one factor only: reward aggregation becomes `dense_softmin`, a
smooth worst-property objective. We call this the distributionally robust
property objective, not a new optimizer.

Both checkpoints retain one decoder, one softmax and one SMILES vocabulary.
Evaluation uses the P6 hard de-novo and Table1 edit subsets, 20 candidates in
raw generation order, k=1/8/20 and candidate-level validity/success/identity,
with no property reranking.
