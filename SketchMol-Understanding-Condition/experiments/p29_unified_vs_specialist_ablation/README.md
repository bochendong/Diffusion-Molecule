# Unified versus specialist continuation ablation

This experiment tests whether the balanced unified MolProgram continuation
matches two task-specific continuations from the same aligned checkpoint.

The frozen arms are:

- unified: 488,490 construction plus 569,905 editing examples;
- construction specialist: the exact 488,490 construction examples used by
  the unified arm;
- editing specialist: the exact 569,905 editing examples used by the unified
  arm.

Every active property-count bucket contributes 81,415 examples.  The sampler
materializes all thirteen frozen permutations before applying the mode filter,
so each specialist reuses the corresponding rows and within-bucket order from
the already-trained unified arm.  The specialists inherit the same aligned
24k adapter and use the same learning rate, effective batch size, seed, and
one-pass schedule.  No calibration or benchmark-based checkpoint selection is
allowed.

The unified and construction-specialist adapters are compared under the native
Table 1 Best-of-40 protocol.  The unified and editing-specialist adapters are
compared under the native Table 2 sampled-once Raw@1 protocol.  The final
collector creates one two-row table without a cross-task composite score.

Run on Nibi with:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition
bash experiments/p29_unified_vs_specialist_ablation/submit_ablation.sh
```

This is a shared-initialization continuation ablation.  It must not be
described as two specialists trained independently from the base model.
