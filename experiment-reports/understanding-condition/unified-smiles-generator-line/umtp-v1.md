# Unified Molecular Transformation Policy v1

| Field | Value |
| --- | --- |
| Status | implementation ready; cluster run pending |
| Scope | one goal-conditioned policy for de novo generation and source-conditioned editing |
| Input contract | `(goal, source_or_null) -> target molecule` |
| Primary checkpoint | `outputs/unified_molecular_transformation_policy_v1/seed_<seed>/policy/unified_smiles_generator.pt` |

## Method contract

UMTP v1 removes the Direct/Unified inference split. Both tasks use the same
condition encoder and shared molecular decoder:

```text
goal / property program -> goal memory -----+
                                               -> shared decoder -> target SMILES
source molecule / NULL  -> source memory ---+
```

- De novo rows have no source tokens, so the source-aware residual is disabled.
- Edit rows route source-marked condition tokens through a dedicated molecular
  memory and a source-conditioned output residual.
- The `transformation` condition layout preserves the legacy de novo goal
  representation exactly and adds source memory only for edit rows.
- A legacy de novo checkpoint initializes the shared decoder; new source-aware
  modules are initialized separately.

## Retention constraint

The frozen de novo teacher is no longer controlled only by a fixed KL weight.
UMTP supports dual-ascent control:

```text
lambda <- clip(lambda + dual_lr * (observed_de_novo_KL - target_KL))
```

This makes de novo retention a declared constraint while leaving edit learning
room when the policy is already within the target KL budget.

## Verifier-guided search distillation

Search distillation is train-only and leakage-audited:

1. Build a task-balanced pool from `unified_joint_train_rows.csv`.
2. Sample one fixed candidate pool per train condition.
3. Keep property-feasible candidates; edit winners must additionally satisfy
   the configured source-Tanimoto threshold and must not be source copies.
4. Mix verifier winners with original source replay rows.
5. Fine-tune the same source-aware policy for one policy-improvement stage.

Formal test rows never enter this loop.

## Cluster entrypoints

Smoke/pilot, one train seed and one evaluation seed:

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_v1_pipeline.sh
```

Paper matrix:

```bash
git pull --ff-only
UMTP_TRAIN_SEEDS=7,17,27 \
UMTP_EVAL_SEEDS=101,202,303 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_v1_pipeline.sh
```

The pipeline submits:

```text
source-aware joint SFT
  -> train-only verifier search and distillation
  -> 2p7p / OOD / Table1 formal evaluation
  -> umtp_v1_runs.csv / umtp_v1_aggregate.csv
```

Candidate budgets `1,20,128,256` share one maximum candidate pool. `raw` and
`finalizer` are derived from identical prefixes.

## Success criteria

The first run is useful only if it improves the policy itself, not just search:

- preserve de novo validation within the declared retention budget;
- improve Table1 `Acc_all(0.65)` rather than only validity/`Acc_all(0.15)`;
- improve raw@1 and finalizer@20 relative to Joint v2;
- keep source-copy augmentation disabled for formal evaluation;
- keep formal test rows out of checkpoint selection and search distillation.
