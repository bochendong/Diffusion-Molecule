# P8.1.11 — Group-Relative RL for the Unified SELFIES Transducer

P8.1.11 starts from the clean P8.1.2-R1 P6-warmstart checkpoint, which already
produces 100% valid de-novo candidates.  It does not retrain SFT and does not
switch representations: one decoder emits the same `TRANSDUCE / KEEP / DELETE /
INSERT / STOP` program language from either an empty or source SELFIES state,
and one interpreter reconstructs the complete molecule.

The available full-SMILES GRPO runner cannot honestly optimize this constrained
program sampler.  P8.1.11 therefore implements **group-relative REINFORCE**, not
GRPO: four grammar-constrained programs are sampled per prompt, molecular
property rewards are standardized within the group, and their differentiable
token log-probabilities receive one policy-gradient update.  A sampled squared
reference log-ratio penalty and a small oracle-program SFT anchor stabilize the
update; there is no PPO ratio or clipped surrogate.

Training uses only 6p/7p de-novo training rows and non-assay edit training rows.
The reward explicitly blanks `target_smiles`; evaluation rows and targets never
enter training.  R1 uses a hard joint bottleneck over validity, requested
properties, and edit similarity.  R2 independently restarts from the same base
and changes only the aggregation to dense softmin.  It is queued with `afterany`.

Both arms run raw `k=1,8,20` on the frozen P6 hard de-novo and MolEdit Table1
subsets and report candidate validity, uniqueness, identity, and strict
non-identity success.  There is no property reranking, router, or alternate
materializer.
