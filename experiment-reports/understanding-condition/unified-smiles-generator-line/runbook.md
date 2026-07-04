# Unified Generator Runbook

This runbook defines the intended rollout. It is deliberately aligned with existing benchmark exporters/evaluators so that the new line can be compared without inventing new metrics.

## Stage 0: data and row unification

Create a single training/evaluation row schema:

```text
condition_id
task_mode                 # de_novo | edit
instruction
prompt
source_smiles             # empty for de_novo
source_image              # optional; render source_smiles when absent
target_smiles             # train/eval reference when available
condition_properties
property_count
external_task_properties
external_property_directions_json
target_<property>
<property>_direction
```

Reuse current exporters as sources:

| Source rows | Existing exporter/script |
| --- | --- |
| 2p-7p de novo | `export_denovo_2p7p_benchmark_rows.py` |
| de novo OOD | `export_denovo_ood_benchmark_rows.py` |
| MolEdit/Table1-style edit rows | existing MolEdit/Table1 exports and `run_direct_smiles_moledit_table1_group_rl.sh` inputs |
| MuMO/C-MuMO external rows | `export_external_multiproperty_benchmark_rows.py` |

## Stage 1: unified SFT

Train one `ConditionedSmilesDecoder` with explicit mode tokens.

Training mix:

```text
de_novo rows:
  condition = [VLM query tokens; <DE_NOVO>; property tokens]
  target = target/generated training SMILES

edit rows:
  condition = [VLM query tokens; <EDIT>; source tokens; property tokens]
  target = target edited SMILES
```

The standalone implementation now makes `task_mode` and mode-token insertion explicit inside the new experiment folder instead of routing through the old direct-SMILES condition-mixing flags.

Input-modality ablation:

```text
with_image:
  condition_feature_variant = full
  frozen condition features = instruction/text + source molecule image when available

no_image:
  condition_feature_variant = text_only
  frozen condition features = instruction/text only
```

For edit rows, no-image still keeps the explicit source-SMILES token path in
the generator. It removes visual VLM evidence, not source structure altogether.
Removing source tokens is a separate ablation.

## Stage 2: unified group-relative RL

Continue from the unified SFT checkpoint with one policy and task-aware rewards.

```text
same generator, mixed mini-batch
  row.task_mode = de_novo -> property_strict reward
  row.task_mode = edit    -> source-preserving edit reward
```

RL loop:

```text
condition pack
  -> sample G candidates per prompt
  -> score candidates with per-row reward router
  -> normalize rewards within each prompt group
  -> update the same SMILES decoder with policy-gradient + SFT anchor + reference penalty
```

Default reward components:

| Mode | Reward signal |
| --- | --- |
| `de_novo` | validity + strict property success - property distance |
| `edit` | validity + edit-direction success + source similarity - source-copy penalty - edit/property distance |

Current standalone entrypoint:

```text
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_group_rl.sh
```

The default `SUCC_UNIFIED_RL_REWARD_MODE=auto` keeps both task lines in one RL
run. Forced `property_strict` or `table1_edit` modes are only for ablations.

## Stage 3: unified sampling and benchmark bridge

One benchmark runner should emit both selected and candidate outputs:

```text
unified_smiles_predictions.csv
unified_smiles_candidate_predictions.csv
```

Sampling defaults:

| Benchmark | Initial budget |
| --- | ---: |
| De novo fair-budget | 40 |
| De novo scaling | 128 / 256 |
| MolEdit Table1 fair row | 20 / 256 |
| MolEdit Table1 assisted row | 2048, only if explicitly labeled |
| MuMO | 20 and 40 |
| C-MuMO | 20 and 40 |

Candidate-generation modes:

```text
sample:
  stochastic decoding from the same condition; useful for diversity.

beam:
  deterministic beam search; useful for high-probability candidates and ablations.

sample_beam:
  union of stochastic samples and beam candidates before dedup/rerank.
```

For top-40 external benchmarks, keep generation budget aligned:

```text
top-40 sample run: num_samples >= 40
top-40 beam run: beam_size >= 40
top-40 mixed run: num_samples + beam_size >= 40, but report both settings
```

Benchmark bridge:

```text
run_unified_smiles_generator_benchmark_suite.sh
  optional: sample checkpoint -> selected/candidate CSVs
  denovo_2p7p            -> existing de novo evaluator
  denovo_ood             -> existing OOD de novo evaluator
  external_multiproperty -> existing MuMO/C-MuMO style evaluator
  moledit_table1         -> unified candidate budget selection -> existing MolEdit Table1 evaluator
```

For mixed unified CSVs, the bridge writes evaluator-specific filtered input
CSVs first, so de novo rows do not leak into edit metrics and edit rows do not
leak into de novo strict-success metrics.

Run benchmark reports separately for:

```text
unified_smiles_generator_with_image
unified_smiles_generator_no_image
```

The comparison answers whether the rendered/source molecule image adds signal
over instruction text plus explicit source-SMILES tokens.

The current suite scripts cover the default grid:

```text
sample + with_image
sample + no_image
beam   + with_image
beam   + no_image
```

Keep first server runs at top-20/top-40 budgets. Increase to `n=128/256` only
after the four-way grid produces valid candidate CSVs and stable metrics.

## Stage 4: SMILES finalizer

The finalizer chooses the final SMILES from generated candidates but does not change the generator architecture.

```text
de_novo finalizer:
  RDKit validity
  property strict success
  property distance

edit finalizer:
  RDKit validity
  source Tanimoto / scaffold preservation
  property success / distance
  optional ADMET prior without oracle lookup
```

Rules:

- If a benchmark has an official candidate-level evaluator, keep the full candidate CSV.
- If a reranker directly optimizes a benchmark's hidden/objective success labels, mark it assisted.
- Do not use generated/source oracle CSVs for test-time selection.

## Stage 5: code entrypoints and benchmark suite

Current standalone code entrypoints:

```text
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/unified_smiles_generator.py
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/unified_benchmark_runner.py
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/prepare_condition_feature_variants.py
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_train.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_group_rl.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_sample.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_benchmark_suite.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_feature_variants.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_experiment_suite.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_smiles_generator_feature_variants.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_smiles_generator_experiment_suite.sh
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/gpu_memory_plan.md
SketchMol-Understanding-Condition/experiments/unified_smiles_generator/examples/smoke_rows.csv
```

Still to add after first server run:

```text
collect_unified_smiles_generator_results.py
```

The suite should produce one report table with separate sections:

```text
1. De novo 2p-7p
2. De novo OOD
3. MolEditRL / MolEdit-Instruct Table1
4. MuMO
5. C-MuMO
6. Ablations
```

## Required ablations

| Ablation | Purpose |
| --- | --- |
| no mode token | Check whether explicit task routing matters |
| no source tokens | Verify edit mode needs source-SMILES conditioning |
| text-only VLM / no-image input | Separate instruction text from rendered molecule image |
| image-only VLM for edit | Test source image contribution |
| no property tokens | Measure how much is carried by VLM prompt alone |
| shared generator vs branch-specific checkpoints | Prove the unified-generator claim |

## Reporting rules

- Keep candidate budget in every table title.
- Do not compare top-20 to top-40 without labeling the budget.
- Do not mix de novo strict success with edit SR.
- For MuMO/C-MuMO, report `SR`, `Sim(success)`, and `RI(success)` using the existing official-style evaluator.
- For MolEditRL/Table1, split fair generation and assisted selection rows.
- For de novo, report property-count buckets and overall strict.
