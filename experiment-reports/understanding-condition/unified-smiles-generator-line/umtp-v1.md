# Unified Molecular Transformation Policy v1

| Field | Value |
| --- | --- |
| Status | seed-7 SFT/search distillation complete; formal eval queued; short RL pilot completed (`stop`) |
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

## Fast source-aware RL go/no-go

The short RL pilot is deliberately smaller than a paper run. It answers one
question before more H100 time is committed: can the current UMTP checkpoint
improve strict source-conditioned editing at raw `n=1` without losing held-out
de novo generation?

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_v1_rl_pilot.sh
```

One 20 GB H100 MIG runs a paired protocol in a single job:

```text
fixed stratified edit/de novo validation subsets + fixed eval seed
  -> baseline raw/finalizer n=1,8
  -> 1 epoch GRPO on 24 rows per train group, 8 rollouts per prompt
  -> post-RL raw/finalizer n=1,8 on the identical candidate protocol
  -> automatic go/stop report with checkpoint hashes
```

The pilot is `go` only when Table1 raw `n=1` improves by at least 2 points at
`Acc_all(0.65)` or 5 points at `Acc_all(0.15)`, while held-out de novo raw
`n=1` strict success drops by no more than 2 points. Formal Table1/OOD test
rows are not used for this decision. This is a diagnostic for the current
source-aware checkpoint, not a claim that similarity-reward GRPO is a new
method.

Result: Nibi job `19073785` completed in **9m25s**. Table1-style raw
`Acc@0.15` moved from **10.0% to 12.5%**, strict `Acc@0.65` remained **0**,
and held-out de novo raw strict moved from **5.83% to 7.50%**. The automatic
decision was **stop**; see [`umtp-v1-rl-pilot.md`](umtp-v1-rl-pilot.md).

## Success criteria

The first run is useful only if it improves the policy itself, not just search:

- preserve de novo validation within the declared retention budget;
- improve Table1 `Acc_all(0.65)` rather than only validity/`Acc_all(0.15)`;
- improve raw@1 and finalizer@20 relative to Joint v2;
- keep source-copy augmentation disabled for formal evaluation;
- keep formal test rows out of checkpoint selection and search distillation.
