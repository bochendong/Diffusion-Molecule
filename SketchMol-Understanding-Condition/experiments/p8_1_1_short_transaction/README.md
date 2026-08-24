# P8.1.1: Short State-Adaptive Transactions

This gate tests one deliberately narrow claim: a unified molecular policy is
most reliable when its output length follows the current molecular state.  The
same property conditioner, autoregressive decoder, vocabulary softmax and
checkpoint emit:

- an ordinary molecular literal from an empty state; and
- one short executable transaction from a supplied molecular state.

The legacy P1 Group-RL path is frozen exactly.  Only source-memory parameters
and newly appended transaction-token embeddings are trained.  Editing uses a
source-only executable grammar; the instruction affects probabilities only
through the learned checkpoint.  Twenty edit candidates are sampled without
replacement from that distribution.  No target molecule, property oracle or
property reranker is used at inference.

The single-seed gate reports raw `k=1,8,20` on the frozen P6 hard 6p/7p and
MolEdit Table 1 subsets.  `audit_unified_checkpoint.py` fails the run if the
legacy de-novo parameters moved, train/eval IDs overlap, or the two evaluation
arms did not use the same checkpoint hash.

