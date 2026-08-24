# P8.1.13 — Verified counterfactual DPO

P8.1.13 turns P8.1.12's train-only verified-success outcomes into explicit
same-prompt preferences.  The chosen outcome is an official strict-success,
nonidentity edit.  Its hard negative is a valid strict failure produced under
the same source and property program, chosen by source similarity and then
property success fraction.  The deployed model is one source-aware decoder
which always emits a complete molecule SMILES.

R1 uses uniform, reference-based, length-normalized DPO. R2 changes exactly
one scientific factor: the per-pair loss weight is proportional to the
transaction teacher's positive-vs-negative confidence. Both rounds are
mandatory and separately restart from the same full-SMILES base checkpoint.

`run_precompute.sh` requires the P8.1.12 PRE marker and reuses its verified
positive CSV. It only re-enumerates the counterfactual failures that P8.1.12
did not preserve. `submit_queue.sh` accepts `P8112_PRE_JOB_ID` when that shared
artifact is still queued, avoiding a duplicate P8.1.12 precompute.
