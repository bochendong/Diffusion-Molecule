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
