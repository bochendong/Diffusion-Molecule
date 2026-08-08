# Unified SMILES Generator Code Line

This folder contains the standalone code for the unified-generator experiment
line.  It intentionally avoids importing repo-internal training modules; the
SMILES tokenizer, vocabulary, Transformer decoder, condition packer, training
loop, sampler, and finalizer are copied into `unified_smiles_generator.py`.

## Unified Molecular Transformation Policy v1

UMTP v1 is the clean successor to the compatibility-based UJV2 path. It uses
one `(goal, source_or_null) -> molecule` contract:

- `--source-aware` separates source-marked condition tokens into a dedicated
  molecular memory while retaining one shared decoder;
- `--condition-layout transformation` keeps the proven de novo goal layout and
  adds source memory only when a source molecule exists;
- de novo rows contain no source memory and therefore bypass the edit residual;
- `--distill-control adaptive` treats frozen-teacher retention as a KL
  constraint with a dual variable instead of a fixed penalty;
- verifier search is run only on a balanced train pool and is distilled back
  into the same policy.

One-seed cluster pipeline:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_v1_pipeline.sh
```

Full paper matrix:

```bash
UMTP_TRAIN_SEEDS=7,17,27 UMTP_EVAL_SEEDS=101,202,303 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_v1_pipeline.sh
```

Primary entrypoints:

```text
run_umtp_v1_train.sh
run_umtp_v1_search_distill.sh
run_umtp_v1_eval_one.sh
submit_umtp_v1_pipeline.sh
prepare_transformation_search_pool.py
build_transformation_search_distillation_rows.py
collect_umtp_v1_results.py
```

## Unified Joint v2 fair protocol

Joint v2 compares only Unified U0/U1/U2. Historical SketchMol, Direct,
UniVideo, Phase1, and Fair v1 numbers are reference-only and are not used for
direct superiority claims.

The formal protocol uses train seeds `7,17,27`, evaluation seeds
`101,202,303`, and candidate budgets `1,20,128,256`. Each
checkpoint/benchmark/evaluation-seed combination samples one maximum pool of
256 candidates. `raw` takes the first generated candidate in each prefix;
`finalizer` applies the RDKit/TDC property scorer to the same prefix. Source
copy augmentation is disabled unless explicitly requested for diagnostics.

Prepare data, train U1/U2, evaluate every epoch on validation@20, and select a
checkpoint without reading formal test metrics:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_joint_v2_train_matrix.sh
```

Validation@20 is averaged across evaluation seeds `101,202,303`. The forgetting
gate requires de novo macro strict success to remain within
2 percentage points of U0. Eligible epochs are ranked by mean Table1
`Acc_all(0.15)`. A seed with no eligible epoch is written as
`forgetting_failure` and receives no formal-test checkpoint symlink.

After all six U1/U2 seed jobs have selected checkpoints, submit the formal
test matrix:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_joint_v2_eval_suite.sh
```

The task jobs run independently and a dependent collector writes:

```text
unified_joint_v2_runs.csv
unified_joint_v2_aggregate.csv
unified_joint_v2_paired_deltas.csv
unified_joint_v2_summary.json
```

Each run also has `joint_v2_run_metadata.json` with checkpoint/input/candidate
SHA256, seed, modality, budgets, selection modes, candidate-pool counts, and a
source-copy audit. The dataset manifest records row/group counts, SHA256,
target-molecule overlap, source-target edit-pair overlap, and Bemis-Murcko
scaffold overlap. Formal input modalities are `text_property` for de novo/OOD
and `source_structure_text` for Table1.

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
SUCC_UNIFIED_RL_OBJECTIVE           # group_pg | grpo
SUCC_UNIFIED_RL_GRPO_CLIP_EPS       # default: 0.2
SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS  # reuse each rollout group for clipped GRPO updates
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT
SUCC_UNIFIED_INPUT_MODALITY
SUCC_UNIFIED_RL_REWARD_MODE        # auto | property_strict | table1_edit
SUCC_UNIFIED_RL_REWARD_AGGREGATION # mean | joint_bottleneck
SUCC_UNIFIED_RL_REWARD_JOINT_BONUS_WEIGHT
SUCC_UNIFIED_RL_REWARD_BOTTLENECK_WEIGHT
SUCC_UNIFIED_RL_SFT_WEIGHT
SUCC_UNIFIED_RL_REFERENCE_KL_WEIGHT
SUCC_UNIFIED_RL_REWARD_SOURCE_SIMILARITY_WEIGHT
SUCC_UNIFIED_RL_REWARD_SOURCE_COPY_PENALTY
```

`joint_bottleneck` keeps the partial-success signal, adds an explicit bonus
only when every requested property succeeds, and subtracts the largest
normalized property violation. Missing oracle values count as failed
constraints. The legacy `mean` behavior remains the default for old runs.

The first fixed-budget single-seed pilot uses `n=20` and one H100 20 GB MIG:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_joint_bottleneck_pilot.sh
```

## Protected GraphEditDSL paper-budget reconciliation

The protected common-decoder action policy writes one maximum candidate pool.
Reuse that immutable pool for the paper-facing `n=20` result without requesting
another GPU or rerunning candidate generation:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_graph_action_budget_reconcile.sh
```

The default prefix budgets are `1,8,20,64,256`. The reconciliation job is CPU
only, requires the existing candidate CSV and summary, and fails instead of
silently regenerating them when either artifact is missing.

## Instruction-aligned GraphEditDSL v2 pilot

The v1 action labels prioritize paired-target similarity and only treat local
RDKit properties as supported supervision. The v2 pilot instead ranks action
labels by the official source-relative instruction predicate, includes cached
TDC/SA scorers such as GSK3B and DRD2, and records per-task one-step oracle
reachability before training.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_umtp_graph_action_instruction_pilot.sh
```

The single-seed run is validation-only. It compares the protected v1 and v2
action checkpoints at `n=1,8,20`, freezes the legacy de novo path, and stops
before training unless GSK3B is at least 95% oracle-evaluable with at least 5%
strict one-step reachability. Formal Table1 test rows are not used for the gate.

The default `auto` reward mode routes each row by `task_mode`, so mixed
de novo/edit batches are valid.

`SUCC_UNIFIED_RL_OBJECTIVE=group_pg` keeps the original group-relative
REINFORCE-style loss:

```text
loss = - A_i log pi_theta(y_i | x)
```

`SUCC_UNIFIED_RL_OBJECTIVE=grpo` switches to a clipped GRPO-style surrogate
using rollout-time log-probabilities from the same policy:

```text
ratio = exp(log pi_theta(y_i | x) - log pi_old(y_i | x))
loss  = - min(ratio * A_i, clip(ratio, 1-eps, 1+eps) * A_i)
```

Use `SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS>1` when you want clipping to matter
within each sampled rollout group.

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

## Direct Warm-start GRPO Beam Rescue

The first-pass unified smoke is intentionally weak. To test whether the unified
wrapper can recover benchmark-level de novo performance, reuse the strong
direct-SMILES group-RL checkpoint, keep the direct-compatible condition layout,
train only de novo rows with GRPO, and evaluate with beam@40:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_smiles_generator_direct_warmstart_grpo_beam.sh
```

Key defaults:

```text
SUCC_UNIFIED_DIRECT_WARMSTART_CHECKPOINT=.../direct_smiles_generator_rl.pt
SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT=direct_compat
SUCC_UNIFIED_RL_OBJECTIVE=grpo
SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE=40
SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE=128
```

The script writes to
`SketchMol-Understanding-Condition/outputs/unified_smiles_generator_direct_warmstart_grpo_beam_v1/`
and runs both a warm-start beam sanity check and the GRPO beam benchmark.

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

## Joint v2: one protected Unified checkpoint

`Joint v2` is the first training protocol in this folder that puts de novo
2p-7p rows and Table1 source-edit rows into the same checkpoint. It deliberately
excludes OOD rows from training and writes a hashed data manifest before the
trainer starts.

Stages:

```text
u0  frozen Direct 2p-7p SFT checkpoint; evaluation baseline only
u1  task-balanced joint SFT from the same u0 checkpoint
u2  task-balanced joint SFT plus de novo-only frozen-teacher KL protection
```

The balanced sampler gives de novo and edit equal row probability. Within de
novo it cycles uniformly over property counts; within edit it cycles uniformly
over instruction task groups. The default 24,000 samples/epoch therefore
replays the full 12k de novo set while oversampling the smaller edit set.

Train U1 or U2:

```bash
SUCC_UNIFIED_JOINT_STAGE=u1 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_joint_v2_train.sh

SUCC_UNIFIED_JOINT_STAGE=u2 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_joint_v2_train.sh
```

Slurm:

```bash
SUCC_UNIFIED_JOINT_STAGE=u1 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_joint_v2_train.sh

SUCC_UNIFIED_JOINT_STAGE=u2 \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/submit_unified_joint_v2_train.sh
```

The default protected stage uses `distill_weight=0.3`, `direct_compat`, four
SFT epochs, and a fresh optimizer at `1e-4`. U1 and U2 both start from the same
base checkpoint so their difference is attributable to teacher protection.

Full evaluation uses all 2p-7p rows, OOD rows, and Table1 rows at identical
candidate budgets. It reports both property-finalized and raw generation runs:

```bash
SUCC_UNIFIED_JOINT_STAGE=u2 \
SUCC_UNIFIED_JOINT_EVAL_BUDGETS=20,128 \
SUCC_UNIFIED_JOINT_SELECTION_MODES=finalizer,raw \
bash SketchMol-Understanding-Condition/experiments/unified_smiles_generator/run_unified_joint_v2_eval_suite.sh
```

All per-task summary rows are collected into
`eval/<stage>/unified_joint_v2_summary.csv` after the matrix completes.

The data preparation step defaults to `drop_train` for any train/eval molecule
or edit-pair collision and records all removals in
`unified_joint_manifest.json`. Set
`SUCC_UNIFIED_JOINT_OVERLAP_POLICY=fail` when auditing a newly rebuilt corpus.

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
