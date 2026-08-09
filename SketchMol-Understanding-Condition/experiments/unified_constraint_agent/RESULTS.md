# Unified Constraint Agent Results

## Common-LLM 1.5B method-signal pilot

Date: 2026-08-09

### Protocol

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adaptation: LoRA rank 16, alpha 32, one epoch
- Stable compute: FP32/TF32, learning rate `5e-5`
- Training rows after tokenization: 4,485
- Held-out rows: 144, separated during train-only data preparation
- Decoding: deterministic first action for this format/control diagnostic
- Training job: `19372638` (`COMPLETED`, 12 minutes 25 seconds)
- Evaluation job: `19373690` (`COMPLETED`, 3 minutes 1 second)

The saved adapter contains 18,464,768 parameters. Both the training gate and
an independent safetensors scan reported zero non-finite parameters.

### Held-out action-control results

| Scope | Model | JSON parse | Correct action type | Executable action | Exact action |
|---|---:|---:|---:|---:|---:|
| All (144) | Base | 66.0% | 27.1% | 0.7% | 0.0% |
| All (144) | LoRA | **97.9%** | **97.2%** | **37.5%** | 0.7% |
| De novo (95) | Base | 85.3% | 40.0% | 1.1% | 0.0% |
| De novo (95) | LoRA | **100.0%** | **100.0%** | 9.5% | 0.0% |
| Table1 edit (23) | Base | 30.4% | 4.3% | 0.0% | 0.0% |
| Table1 edit (23) | LoRA | **87.0%** | **82.6%** | **82.6%** | 4.3% |
| MuMO edit (26) | Base | 26.9% | 0.0% | 0.0% | 0.0% |
| MuMO edit (26) | LoRA | **100.0%** | **100.0%** | **100.0%** | 0.0% |

Artifacts:

```text
/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/
  outputs/unified_constraint_agent_common_llm_pilot_v1/
    model/seed_1703/training_summary.json
    eval/seed_1703/base/summary.json
    eval/seed_1703/tuned/summary.json
```

### Interpretation

The pilot passes the method-signal gate for shared constraint parsing and typed
action control: the LoRA model reliably selects the de novo versus edit action
space and makes most edit actions executable. It does **not** establish a
paper-level molecular-design improvement.

Greedy de novo output collapsed to five unique strings across 95 conditions;
one invalid string was emitted for 83 conditions. Edit outputs were more
diverse, but executable GraphEditDSL does not imply that the requested
properties improve or that the similarity constraint is met. Exact-action
matching is also too strict for underdetermined molecular tasks, so final
claims must use molecular property, similarity, validity, and strict-success
metrics.

### Method decision

Do not scale this direct action-SFT recipe to 7B yet. Keep the common LLM as the
shared constraint and revision controller, while retaining typed chemical
executors:

1. De novo: let the chemical generator produce candidates; let the common LLM
   produce a structured design/revision plan and select verifier-guided actions.
2. Editing: constrain GraphEditDSL generation to source-valid sites and use the
   executor plus vector verifier feedback during revision.
3. Evaluate both branches with sampled `n=20`, deterministic verifier selection,
   and the official strict property/similarity metrics.

This preserves one common reasoning model without requiring it to serve as an
unconstrained low-level molecular decoder for every task.

### Invalid earlier attempts

- Job `19371280` completed at the scheduler level but every saved adapter
  parameter was non-finite. It must not be used.
- Job `19372184` used the finite-value gate and failed at step 50 after BF16
  instability. It is audit evidence only.

## Executable GraphEditDSL ranking diagnostic

Date: 2026-08-09

Job `19387681` completed in 2 minutes 38 seconds. This diagnostic replaced
free-form edit generation with LLM likelihood ranking over at most 20
RDKit-executable one-step actions. It used the same 49 train-only held-out edit
rows as the format diagnostic: 23 Table1 and 26 MuMO.

| Scope | Model | Strict selected@1 | Any strict@20 | Mean property success | Mean source similarity |
|---|---:|---:|---:|---:|---:|
| All | Base | 2.0% | 18.4% | 20.4% | 0.793 |
| All | LoRA | **4.1%** | 18.4% | **28.0%** | 0.795 |
| Table1 | Base | 4.3% | 26.1% | 27.8% | 0.775 |
| Table1 | LoRA | 4.3% | 26.1% | **38.6%** | **0.810** |
| MuMO | Base | 0.0% | 11.5% | 13.8% | **0.810** |
| MuMO | LoRA | **3.8%** | 11.5% | **18.6%** | 0.782 |

All rows had at least one executable action, but post-filter pools averaged
only 11.6 unique molecules. Only 9/49 rows contained any strict-success action,
and the target teacher action appeared in the pool for 8/49 rows.

The LoRA signal is clearer below top-1. Among the nine reachable strict cases,
the best strict action's median rank improved from 9 to 4. LoRA placed five
strict actions in the global top-4 (10.2% of all rows), eight in top-7 (16.3%),
and all nine in top-10 (18.4%). Base placed only one in top-7 (2.0%). Thus the
common LLM learned useful action ordering, but the current exact-action SFT loss
is not a strong enough ranking objective.

### Decision after constrained ranking

Do not launch the official benchmark with this checkpoint. The next pilot must
address both independent limits:

1. Candidate support: produce 20 actual valid, diverse actions and add compact
   multi-step revisions so `any-hit@20` and teacher-action recall rise.
2. Selection: train directly on train-only strict-positive versus hard-negative
   action preferences, rather than relying on free-generation cross-entropy.
3. Keep the paper contract fixed: Common LLM ranks typed actions, RDKit executes
   them, and the vector verifier returns feedback for one bounded revision.

The failed predecessor job `19377452` stopped safely because four prompts were
longer than 512 tokens. Commit `de9e23b` raised the limit to 1024 and preserves
the complete action target under truncation; it did not change the evaluation
rows or candidate budget.
