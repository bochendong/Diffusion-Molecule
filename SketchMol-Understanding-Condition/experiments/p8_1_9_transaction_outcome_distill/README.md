# P8.1.9 — Transaction-outcome distillation into one full-SMILES policy

P8.1.9 uses the P8.1.1 short-transaction policy only as an **offline
training-data teacher**.  The teacher is rerun on the instruction-v2 training
partition, its executable transaction is committed to a complete SMILES, and
that SMILES is used as an SFT target for the P8.1.7/P8.1.4 shared full-SMILES
student.  The deployed student has one checkpoint, decoder, vocabulary, and
output head.  It does not contain the transaction interpreter.

The overlap audit is fail-closed.  Existing P8.1.1 Table-1 candidates are
evaluation artifacts and are never accepted as training data.  Training rows
are removed if an identifier, canonical source, original target, or generated
pseudo-target overlaps the frozen Table-1 evaluation molecules.

Two mandatory seed-7 rounds restart from the same student checkpoint:

- R1: one uniformly weighted deterministic teacher outcome per train row.
- R2: the same outcomes and update budget, with teacher-confidence weighting
  implemented by deterministic row replication.  This is the only changed
  scientific factor.

Both rounds report raw k=1/8/20 P6-hard de-novo and MolEdit Table-1 metrics,
candidate validity, identity, strict non-identity success, and source
similarity.  No property reranking or source-copy candidate is enabled.

```bash
bash SketchMol-Understanding-Condition/experiments/p8_1_9_transaction_outcome_distill/submit_queue.sh
```
