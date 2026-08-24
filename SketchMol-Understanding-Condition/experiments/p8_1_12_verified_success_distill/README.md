# P8.1.12 — Verified-success outcome distillation

P8.1.12 fixes the causal failure of P8.1.9: the transaction teacher is no
longer distilled merely because it assigns an outcome high likelihood.  On a
strictly train-only partition, all source-executable outcomes are first scored
with the official MolEdit Table-1 property predicate and the 0.65 source
similarity gate.  Only valid, strict-success, non-identity outcomes enter the
success set.  The highest-likelihood teacher outcome is selected **within that
set**.

Rows with no verified outcome are dropped and coverage is reported by task and
requested-property count.  ID, canonical source, original target, and selected
pseudo-target overlap against the frozen Table-1 evaluation set are fail-closed.
Existing P8.1.1 evaluation candidates are never training inputs.

The student is the single full-SMILES P8.1.7/P8.1.4 checkpoint.  Only its
source-conditioned modules train; the de-novo parameter path stays bitwise
frozen.  The transaction teacher and interpreter are absent at inference.

- R1: uniform verified-success outcome SFT.
- R2: same base, verified outcomes, steps, and seed; only teacher-likelihood
  confidence weighting changes.

```bash
bash SketchMol-Understanding-Condition/experiments/p8_1_12_verified_success_distill/submit_queue.sh
```
