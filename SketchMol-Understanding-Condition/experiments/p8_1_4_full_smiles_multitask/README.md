# P8.1.4 — Full-SMILES Multitask Policy

The strict claim is one checkpoint, one autoregressive decoder, one softmax,
and one full-molecule SMILES language. `GENERATE` versus `EDIT` is represented
only by the explicit mode token in the `unified` condition layout; the edit
row additionally observes source-SMILES tokens. There is no router, graph
interpreter, materializer, property reranker, or alternate output family.

R1 freezes every parameter inherited from the P1 de-novo Group-RL checkpoint
and trains only small source-gated residual modules on a balanced de-novo/edit
set. It intentionally disables the pointer-generator: preserving a source is
not counted as solving an edit, and the audit reports identity-copy rate.

R2 always runs after R1. Its only scientific change is adding one source-only
Group-RL stage. The preregistered Group-RL reward includes an identity-copy
penalty; this is part of that single objective change, not a second ablation.
Architecture, data language, checkpoint lineage, raw candidate order, decoding,
and evaluation remain fixed.
Both rounds use the P6 hard de-novo and Table1 edit subsets, exactly 20 raw
candidates per condition, `k=1,8,20`, candidate-level metrics, and no property
reranking.
