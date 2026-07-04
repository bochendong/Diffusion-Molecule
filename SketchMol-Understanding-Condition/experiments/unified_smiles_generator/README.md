# Unified SMILES Generator Code Line

This folder contains the standalone code for the unified-generator experiment
line.  It intentionally avoids importing repo-internal training modules; the
SMILES tokenizer, vocabulary, Transformer decoder, condition packer, training
loop, sampler, and finalizer are copied into `unified_smiles_generator.py`.

## Train

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_train.sh
```

Important environment variables:

```text
SUCC_UNIFIED_TRAIN_CSV
SUCC_UNIFIED_EVAL_CSV
SUCC_UNIFIED_OUTPUT_DIR
SUCC_UNIFIED_TRAIN_FEATURES_DIR
SUCC_UNIFIED_EVAL_FEATURES_DIR
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT  # full | text_only | image_only
SUCC_UNIFIED_INPUT_MODALITY             # with_image | no_image; optional label
SUCC_UNIFIED_NUM_SAMPLES
SUCC_UNIFIED_DECODING_MODE      # sample | beam | sample_beam
SUCC_UNIFIED_BEAM_SIZE
SUCC_UNIFIED_BEAM_EXPAND_SIZE
SUCC_UNIFIED_TOP_K_CANDIDATES
```

## Sample

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_sample.sh
```

Important environment variables:

```text
SUCC_UNIFIED_CHECKPOINT
SUCC_UNIFIED_EVAL_CSV
SUCC_UNIFIED_OUTPUT_DIR
SUCC_UNIFIED_EVAL_FEATURES_DIR
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT
SUCC_UNIFIED_INPUT_MODALITY
```

## Group RL

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_group_rl.sh
```

This continues from one unified SFT checkpoint and keeps the two task lines
inside the same policy:

```text
de_novo row -> property_strict reward
edit row    -> table1_edit/source-preserving reward
```

Important environment variables:

```text
SUCC_UNIFIED_RL_TRAIN_CSV
SUCC_UNIFIED_RL_EVAL_CSV
SUCC_UNIFIED_RL_OUTPUT_DIR
SUCC_UNIFIED_RL_RESUME_CHECKPOINT
SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT
SUCC_UNIFIED_INPUT_MODALITY
SUCC_UNIFIED_RL_REWARD_MODE        # auto | property_strict | table1_edit
SUCC_UNIFIED_RL_SFT_WEIGHT
SUCC_UNIFIED_RL_REFERENCE_KL_WEIGHT
SUCC_UNIFIED_RL_REWARD_SOURCE_SIMILARITY_WEIGHT
SUCC_UNIFIED_RL_REWARD_SOURCE_COPY_PENALTY
```

The default `auto` reward mode routes each row by `task_mode`, so mixed
de novo/edit batches are valid.

## Benchmark Suite

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_benchmark_suite.sh
```

This connects unified selected/candidate CSVs to the existing benchmark
evaluators:

```text
denovo_2p7p              -> evaluate_univideo_image_benchmark.py
denovo_ood               -> evaluate_univideo_image_benchmark.py
external_multiproperty   -> evaluate_external_multiproperty_predictions.py
moledit_table1           -> evaluate_moledit_table_metrics.py
```

Important environment variables:

```text
SUCC_UNIFIED_BENCHMARK_TASKS          # comma list or all
SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE     # 1 runs sampler before evaluation
SUCC_UNIFIED_BENCHMARK_PREDICTION_CSV
SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV
SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR
SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV
SUCC_UNIFIED_MOLEDIT_BUDGETS          # e.g. 20,256
SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV
SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV
```

When input CSVs contain mixed tasks, the runner writes evaluator-specific
filtered inputs under the benchmark output directory before running metrics.

## Input Modality Ablation

Run the same generator recipe with different frozen condition-feature variants:

```text
with_image:
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT=full
  SUCC_UNIFIED_INPUT_MODALITY=with_image

no_image:
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT=text_only
  SUCC_UNIFIED_INPUT_MODALITY=no_image
```

`full` means the frozen HF/VLM condition features may use the source molecule
image. If `source_image` is absent but `source_smiles` exists, the exporter can
render a molecule image before VLM encoding. `text_only` means the frozen VLM
condition features use the instruction/prompt only. In edit mode, both settings
still append source-SMILES tokens to the unified generator; use a separate
`no_source_tokens` ablation if the goal is to remove source structure entirely.

Prediction rows include:

```text
condition_feature_variant
input_modality
method
```

With the default method name, rows are tagged as
`unified_smiles_generator_with_image` or `unified_smiles_generator_no_image`.

## Experiment Suite

The suite expands the main ablation grid:

```text
decoding: sample | beam
input:    with_image | no_image
```

Inference/benchmark-only from existing checkpoints:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_experiment_suite.sh
```

Slurm submit:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_smiles_generator_experiment_suite.sh
```

Default suite behavior is intentionally conservative:

```text
SUCC_UNIFIED_SUITE_RUN_FEATURE_EXPORT=0
SUCC_UNIFIED_SUITE_RUN_TRAIN=0
SUCC_UNIFIED_SUITE_RUN_RL=0
SUCC_UNIFIED_SUITE_RUN_BENCHMARK=1
SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES=40
SUCC_UNIFIED_SUITE_BEAM_SIZE=40
```

Set modality checkpoints explicitly when not running training/RL:

```text
SUCC_UNIFIED_WITH_IMAGE_CHECKPOINT
SUCC_UNIFIED_NO_IMAGE_CHECKPOINT
```

Frozen feature export for both input variants:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_smiles_generator_feature_variants.sh
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_smiles_generator_feature_variants.sh
```

GPU memory guidance is in `gpu_memory_plan.md`.

## Output contract

The sampler writes:

```text
unified_smiles_predictions.csv
unified_smiles_candidate_predictions.csv
```

These preserve benchmark-facing fields such as `condition_id`,
`source_smiles`, `condition_properties`, `external_task_properties`, and
`external_property_directions_json`, while adding:

```text
task_mode
method
generated_smiles
candidate_rank
candidate_selected
valid_smiles
unified_finalizer_score
unified_property_success_fraction
unified_property_distance
source_tanimoto
source_similarity_success
candidate_generation_mode
```

Reward/finalizer scoring uses RDKit descriptors when available and falls back
to optional TDC oracles for assay-style properties such as `DRD2`, `GSK3B`, and
`JNK3`.

## Candidate generation modes

```text
sample:
  repeat the same condition `num_samples` times and use stochastic decoding.

beam:
  run deterministic beam search with `beam_size` beams.

sample_beam:
  union stochastic samples and beam candidates, then deduplicate and rerank.
```

For MuMO/C-MuMO top-40 style runs, set either:

```bash
SUCC_UNIFIED_DECODING_MODE=sample SUCC_UNIFIED_NUM_SAMPLES=40
```

or:

```bash
SUCC_UNIFIED_DECODING_MODE=beam SUCC_UNIFIED_BEAM_SIZE=40
```
