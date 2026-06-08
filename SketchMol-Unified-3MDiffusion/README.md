# SketchMol Unified 3M-Diffusion Stream

This folder is the standalone 3M-style unified experiment line. It is intentionally
separate from `SketchMol-Understanding-Condition` so changes to this prototype do
not alter the main understanding stream or UniVideo-style generation stream.

## Scope

This project explores a unified training path inspired by 3M-Diffusion:

```text
3M-style molecule description rows
  + SketchMol multi-property edit rows
    -> unified condition JSONL
    -> molecule-language / image-language alignment
    -> edit-aware condition tokens
    -> latent diffusion over target molecular vectors
```

The current implementation is a self-contained smoke/prototype stack. It does
not import from `sketchmol_understanding_condition`. The only shared inputs are
data artifacts such as:

```text
Research/Molecule Generation/3M-Diffusion/data/*
SketchMol-MultiProperty-EditDataset/outputs/.../diffusion_edit_manifest.csv
```

## Layout

```text
sketchmol_unified_3m_diffusion/
  unified_condition_dataset.py
  unified_featurization.py
  edit_condition_tokens.py
  latent_diffusion_generation.py
  encoders.py
  chem.py
  image_features.py
  text_features.py
  benchmark_export.py

scripts/
  preflight_unified_3m.py
  export_unified_condition_dataset.py
  train_alignment_pretraining.py
  train_edit_condition_tokens.py
  train_latent_diffusion_generation.py
  evaluate_latent_diffusion_generation.py
  export_latent_benchmark_inputs.py
  run_unified_materialized_benchmark.sh
  run_unified_generation_smoke.sh
  submit_unified_generation_pipeline.sh
```

## Local Smoke

From the repository root:

```bash
SMU3M_3M_ROOT="Research/Molecule Generation/3M-Diffusion" \
SMU3M_EDIT_MANIFEST="SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/diffusion_edit_manifest.csv" \
bash SketchMol-Unified-3MDiffusion/scripts/run_unified_generation_smoke.sh
```

The script uses the isolated `SMU3M_*` environment prefix:

```text
SMU3M_PYTHON_BIN
SMU3M_3M_ROOT
SMU3M_AUTO_CLONE_3M
SMU3M_EDIT_MANIFEST
SMU3M_OUTPUT_DIR
SMU3M_DESCRIPTION_LIMIT
SMU3M_EDIT_LIMIT
SMU3M_TRAIN_LIMIT
SMU3M_BATCH_SIZE
SMU3M_EPOCHS
SMU3M_EVAL_LIMIT
SMU3M_MAX_EVAL_PER_PROPERTY_COUNT
SMU3M_EVAL_SAMPLE_STEPS
SMU3M_ALIGNMENT_SEED
SMU3M_EDIT_SEED
SMU3M_DIFFUSION_SEED
SMU3M_EVAL_SEED
SMU3M_DIFFUSION_TIMESTEPS
SMU3M_DEVICE
SMU3M_REQUIRE_CUDA
SMU3M_CHECKPOINT_EVERY
SMU3M_RESUME
SMU3M_INCLUDE_PUBCHEM
SMU3M_INCLUDE_KV
SMU3M_MIN_EDIT_SOURCE_TANIMOTO
SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS
SMU3M_REQUIRE_EVAL_ORACLE_STRICT
```

Default output:

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_smoke/
  dataset/summary.json
  alignment/alignment_model.pt
  edit_condition_tokens/edit_condition_connector.pt
  latent_diffusion/latent_diffusion_generation.pt
  eval_latent/metrics.json
  eval_latent/generated_latents.npy
  eval_latent/prior_latents.npy
  eval_latent/target_latents.npy
  eval_latent/edit_latent_predictions.npy
  eval_latent/edit_latent_fingerprints.npy
  eval_latent/index.csv
```

## Slurm

On a Slurm login node:

```bash
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

By default this pipeline now builds and trains on the source-neighbor edit
dataset:

```text
SMMED_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1
SMMED_PAIRING_STRATEGY=source_neighbor
SMU3M_EDIT_MANIFEST=.../multiproperty_source_neighbor_v1/diffusion_edit_manifest.csv
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_v1
SMU3M_MIN_EDIT_SOURCE_TANIMOTO=0.4
SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS=1
SMU3M_REQUIRE_EVAL_ORACLE_STRICT=1
```

Set `SMMED_OUTPUT_DIR`, `SMU3M_EDIT_MANIFEST`, and
`SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS=0` explicitly only when rerunning older
`multiproperty_100k_v1` experiments.

The submit script defaults to an economical server run. It does not request a
40GB GPU unless you ask for that profile.

```text
SMU3M_GPU_PROFILE=h100_20gb_mig
SMU3M_RESOURCE_PROFILE=economy
SMU3M_DESCRIPTION_LIMIT=5000
SMU3M_EDIT_LIMIT=50000
SMU3M_TRAIN_LIMIT=50000
SMU3M_BATCH_SIZE=512
SMU3M_EPOCHS=5
SMU3M_DIFFUSION_TIMESTEPS=100
SMU3M_DIFFUSION_OBJECTIVE=pred_x0
SMU3M_DIFFUSION_TARGET=residual
SMU3M_EVAL_SAMPLE_ETA=0.0
SMU3M_REQUIRE_CUDA=1
SMU3M_RESUME=1
SMU3M_CHECKPOINT_EVERY=1
SMU3M_INCLUDE_PUBCHEM=1
SMU3M_INCLUDE_KV=0
SMU3M_SLURM_CPUS=2
SMU3M_SLURM_MEM=16G
```

Every training stage writes `checkpoints/latest.pt` after each epoch. If
`SMU3M_RESUME=1`, the run script resumes from those latest checkpoints when
they exist. For latent diffusion, the script only resumes checkpoints whose
stored `diffusion_objective` and `diffusion_target` match the current settings;
old collapsed `pred_noise` / full-target checkpoints are skipped and retrained.

The current default trains Stage 3 as residual diffusion:

```text
connector prior = target fingerprint/property/delta heads from edit connector
diffusion target = normalized target latent - connector prior
sampling = deterministic DDIM, then add connector prior back
```

This tests whether Stage 3 can improve a learned edit prior rather than generate
the whole molecular vector from pure noise.

In `eval_latent/metrics.json`, check the prior baseline before reading the
generated metrics:

```text
prior_target_fingerprint_cosine
prior_target_property_mae
generated_minus_prior_latent_mae
latent_block_summary.fingerprint/generated_target_mae
latent_block_summary.properties/generated_target_mae
```

If `prior_*` is already poor, the edit connector is the bottleneck. If prior is
reasonable but generated is worse or barely moves from prior, Stage 3 diffusion
still needs work.

If you intentionally request a 40GB H100 MIG, use the throughput profile so the
job increases batch/model size instead of wasting the larger allocation:

```bash
SMU3M_GPU_PROFILE=h100_40gb_mig \
SMU3M_RESOURCE_PROFILE=throughput_40gb \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

That profile defaults to:

```text
SMU3M_BATCH_SIZE=4096
SMU3M_EVAL_BATCH_SIZE=4096
SMU3M_NUM_WORKERS=2
SMU3M_NUM_QUERIES=32
SMU3M_ALIGNMENT_HIDDEN_DIM=1024
SMU3M_EDIT_HIDDEN_DIM=1024
SMU3M_DIFFUSION_HIDDEN_DIM=1024
SMU3M_DIFFUSION_DEPTH=6
SMU3M_SLURM_CPUS=4
SMU3M_SLURM_MEM=40G
```

If Slurm efficiency reports say the job used less than one core or less than
10% memory, keep `SMU3M_RESOURCE_PROFILE=economy` or lower
`SMU3M_SLURM_CPUS` / `SMU3M_SLURM_MEM` rather than repeating the same request.

The submit script still uses `SMMED_*` only for optional multi-property dataset
building. Unified training/job controls use `SMU3M_*`, including:

```text
SMU3M_SLURM_ACCOUNT
SMU3M_SLURM_TIME
SMU3M_SLURM_MEM
SMU3M_SLURM_CPUS
SMU3M_SLURM_JOB_NAME
SMU3M_LOG_DIR
SMU3M_GPU_PROFILE
SMU3M_SLURM_GPUS
```

## Materialized Benchmark

Latent-space metrics are only diagnostics. To compare against SketchMol-style
paper tables, materialize the Unified 3M eval latents through the shared
multi-property benchmark:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_BENCHMARK_SHARDS=5 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

The wrapper uses `eval_latent/generated_latents.npy` plus
`dataset/unified_condition_eval.jsonl`. If the eval directory was produced
before benchmark export existed, it automatically writes:

```text
eval_latent/edit_latent_predictions.npy     # target props / deltas / active / directions
eval_latent/edit_latent_fingerprints.npy    # Unified target fingerprint block
eval_latent/index.csv                       # condition_id alignment for the benchmark
```

It then runs `benchmark_multiproperty_retrieval.py`. The default benchmark
profile is `primary_fast`, which writes to `benchmark_materialized_primary_fast/`
and runs the source-similarity main line plus the essential baselines. Use
`SMU3M_BENCHMARK_PROFILE=full` only when you need the full comparison table.
The primary proposed row is
`edit_latent_source_similarity_rerank`: it does not require scaffold identity,
and instead ranks candidates by predicted edit latent plus source Tanimoto. By
default `SMU3M_BENCHMARK_FINGERPRINT_WEIGHT=1.0` and
`SMU3M_BENCHMARK_SOURCE_SIMILARITY_WEIGHT=1.0`, so candidate ranking uses
predicted properties/deltas, the Unified fingerprint block, and source
similarity. To avoid spending hours computing RDKit Tanimoto for every global
candidate, `primary_fast` first keeps the top edit-latent candidates and then
computes source Tanimoto for `SMU3M_SOURCE_SIMILARITY_RERANK_CANDIDATES`
(default `256`). Set it to `0` for exact legacy reranking. The report to read is:

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/benchmark_materialized/
  benchmark_report.md
  benchmark_summary.csv
  benchmark_decoded.csv
```

For the default fast profile, the directory is:

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/benchmark_materialized_primary_fast/
```

With `SMU3M_BENCHMARK_SHARDS=5`, the submit script calls `sbatch` five times by
default, so each shard gets its own job id. Each shard writes to
`benchmark_materialized_primary_fast/shards/shard_<i>_of_5/`. After all shards
finish, merge the shard outputs:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_BENCHMARK_PROFILE=primary_fast \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/merge_unified_materialized_benchmark_shards.sh
```

Set `SMU3M_BENCHMARK_SUBMIT_MODE=array` if you prefer one Slurm array job
instead of five separate `sbatch` calls.

The materialized benchmark is CPU-serial by default. The submit wrapper now
requests 1 CPU and modest memory for all profiles (`8G` for `primary_fast` /
`scaffold`, `16G` for `full`) so Slurm efficiency reports do not flag the run
as a 4-core serial job. Override `SMU3M_BENCHMARK_SLURM_CPUS`,
`SMU3M_BENCHMARK_SLURM_MEM`, or `SMU3M_BENCHMARK_SLURM_TIME` only when a larger
shard actually needs it.

Those files contain the comparison-ready numbers: `strict@Tanimoto>=0.4/0.6/0.8`
as the primary edit metric, plus 2p-7p strict success and scaffold diagnostics.

## Source-Aware Connector Objective

The edit-token connector now trains with two source-aware fingerprint losses in
addition to the original target property, delta, active-property, direction,
target fingerprint, and similarity-bin heads:

```text
SMU3M_SOURCE_SIMILARITY_LOSS_WEIGHT=0.15
SMU3M_HARD_NEGATIVE_LOSS_WEIGHT=0.05
SMU3M_SOURCE_AWARE_TEMPERATURE=0.07
SMU3M_HARD_NEGATIVE_MARGIN=0.2
SMU3M_SOURCE_AWARE_SHARED_GRADIENT=0
```

`source_similarity_mse` matches the predicted target fingerprint's soft
Tanimoto to the source-target Tanimoto. `source_aware_hard_negative` mines
in-batch targets that are active-property/delta-close but source-incompatible
and pushes them below the true target. By default these source-aware losses
update only the fingerprint head from a detached pooled representation, so the
shared connector trunk remains driven by target property, delta, active-property,
direction, fingerprint, and similarity-bin supervision. Set
`SMU3M_SOURCE_AWARE_SHARED_GRADIENT=1` only as an ablation; the stronger v1-style
shared-gradient setup improved fingerprint cosine but hurt property/delta
control.

For short source-aware direction sweeps, do not submit one Slurm job per
configuration. Pack them into one allocation with GLOST:

```bash
SMU3M_SWEEP_OUTPUT_ROOT=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2 \
SMU3M_SWEEP_LAUNCHER=glost \
SMU3M_SWEEP_PRESET=extended \
SMU3M_SWEEP_CONCURRENCY=1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_sourceaware_sweep.sh
```

The default `extended` sweep runs 31 short directions in one submitted job:

```text
baseline
sim002_head, sim005_head, sim010_head, sim015_head, sim020_head, sim030_head, sim045_head
hard001_head, hard002_head, hard005_head, hard010_head, hard015_head
balanced_005_001, balanced_005_002, balanced_010_002, balanced_015_003, balanced_015_005
balanced_020_005, balanced_030_005, balanced_030_010
temp_cool_015_005, temp_warm_015_005
margin_low_015_005, margin_high_015_005
shared_tiny, shared_low, shared_mid
strong_045_015, strong_sim_cool, hard_cool
```

Each task writes an isolated output directory under `SMU3M_SWEEP_OUTPUT_ROOT`.
The runner creates `tasks/sourceaware_sweep.tasks` for `glost_launch` and writes
`sweep_summary.md` / `sweep_summary.csv` after all tasks finish. The summary
includes fingerprint cosine, source-target fingerprint baseline, gain over that
baseline, property MAE, delta MAE, and the corresponding prior deltas. Set
`SMU3M_SWEEP_PRESET=compact` to run the original nine-direction sweep, or set
`SMU3M_SWEEP_CONFIGS` directly with semicolon-separated
`label:source_weight:hard_weight:shared_gradient:margin:temperature` entries.
Increase `SMU3M_SWEEP_CONCURRENCY` only when the subtasks are CPU-only or you
are sure that concurrent subtasks can share the requested GPU safely.

After job `15729225`, the short sweep found two Pareto-improving configurations
over the in-job baseline: `hard002_head` and `balanced_005_001`. To follow up,
run the packed winner/full and diffusion-prior ablations:

```bash
SMU3M_FOLLOWUP_OUTPUT_ROOT=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1 \
SMU3M_SWEEP_OUTPUT_ROOT=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2 \
SMU3M_FOLLOWUP_PLAN=all \
SMU3M_FOLLOWUP_LAUNCHER=glost \
SMU3M_FOLLOWUP_CONCURRENCY=1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_sourceaware_followup.sh
```

The default follow-up runs:

```text
baseline_full_s11
hard002_full_s11, hard002_full_s23, hard002_full_s37
balanced005001_full_s11
sharedtiny_full_s11
hard002_freeze_p000_steps1, hard002_freeze_p000_steps5, hard002_freeze_p000_steps20
hard002_joint_p025_steps20, hard002_joint_p050_steps20, hard002_joint_p100_steps20
```

The first group reruns the best connector settings with full latent eval
(`SMU3M_EVAL_LIMIT=0`). The second group starts from the pulled
`hard002_head` connector artifacts and tests whether generation quality is lost
by the diffusion sampler. It writes `followup_summary.md` /
`followup_summary.csv`.

To materialize the current sweep winners into the shared benchmark, submit:

```bash
SMU3M_SWEEP_OUTPUT_ROOT=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2 \
SMU3M_WINNER_LABELS=hard002_head,balanced_005_001 \
SMU3M_BENCHMARK_PROFILE=primary_fast \
SMU3M_BENCHMARK_SHARDS=5 \
SMU3M_BENCHMARK_SUBMIT_MODE=array \
SMU3M_BENCHMARK_PRIOR_ONLY=1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_sourceaware_winner_benchmarks.sh
```

`SMU3M_WINNER_LABELS` also accepts explicit output specs as
`label=output_dir`, so follow-up outputs can be benchmarked with the same
runner. Run this on the machine that still has each output directory's
`eval_latent/generated_latents.npy` and `eval_latent/prior_latents.npy`. To add
the current best full-eval follow-up:

```bash
SMU3M_SWEEP_OUTPUT_ROOT=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2 \
SMU3M_WINNER_LABELS='hard002_head,balanced_005_001,sharedtiny_full_s11=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/sharedtiny_full_s11' \
SMU3M_BENCHMARK_PROFILE=primary_fast \
SMU3M_BENCHMARK_SHARDS=5 \
SMU3M_BENCHMARK_SUBMIT_MODE=array \
SMU3M_BENCHMARK_PRIOR_ONLY=1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_sourceaware_winner_benchmarks.sh
```

With `SMU3M_BENCHMARK_PRIOR_ONLY=1`, each winner gets two benchmark lines: the
sampled generated latents and a prior-only materialization using
`eval_latent/prior_latents.npy`. This directly tests whether the connector prior
is stronger than the latent diffusion sample.

To test whether the candidate library contains source-similar property matches,
run a source-first/oracle benchmark. Start with a capped CPU run before full
eval because it computes RDKit source Tanimoto against sampled candidates:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/sharedtiny_full_s11 \
SMU3M_BENCHMARK_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/sharedtiny_full_s11/benchmark_source_first_primary_fast_1k \
SMU3M_BENCHMARK_METHODS=source_identity,source_tanimoto_property_oracle,edit_latent_source_first_rerank,edit_latent_source_similarity_rerank,target_oracle \
SMMED_LIMIT_EVAL_ROWS=1000 \
SMMED_MAX_GLOBAL_CANDIDATES=5000 \
SMU3M_SOURCE_FIRST_MIN_TANIMOTO=0.4 \
SMU3M_SOURCE_FIRST_CANDIDATES=0 \
SMU3M_BENCHMARK_SHARDS=5 \
SMU3M_BENCHMARK_SUBMIT_MODE=array \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

`source_tanimoto_property_oracle` is the candidate-library upper bound inside
the sampled pool: filter by source Tanimoto first, then choose the closest
property match. `edit_latent_source_first_rerank` applies the learned edit-latent
scorer after that source-first filter. If the oracle is still weak, the
candidate library/retrieval formulation is the bottleneck; if the oracle is
strong but edit-latent source-first is weak, the learned scorer is the bottleneck.

## Diffusion Refine

After the residual Stage 3 runs, continue with joint connector + diffusion
refine and latent eval:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_v1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_diffusion_refine.sh
```

The default refine path now jointly fine-tunes the edit connector and diffusion
from the latest checkpoint, then evaluates a balanced 2p-7p sample. It does not
run the materialized benchmark inside the same GPU job; submit that separately
after the latent eval artifacts are written. This avoids the frozen-prior
bottleneck seen in job `15694324` without risking an 8-hour timeout during the
CPU benchmark stage.

Useful overrides:

```text
SMU3M_DIFFUSION_EXTRA_EPOCHS=100
SMU3M_DIFFUSION_LR=3e-4
SMU3M_TRAIN_DIFFUSION_CONNECTOR=1
SMU3M_PRIOR_LOSS_WEIGHT=0.25
SMU3M_MAX_EVAL_PER_PROPERTY_COUNT=250
SMU3M_RUN_MATERIALIZED_BENCHMARK=0
```

Then run the materialized benchmark as a separate job:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_BENCHMARK_PROFILE=primary_fast \
SMU3M_BENCHMARK_SHARDS=5 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

For a longer stability run with the full eval set (9455 edit rows), disable the
balanced eval cap:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_DIFFUSION_EXTRA_EPOCHS=200 \
SMU3M_PRIOR_LOSS_WEIGHT=0.25 \
SMU3M_EVAL_LIMIT=0 \
SMU3M_MAX_EVAL_PER_PROPERTY_COUNT=0 \
SMU3M_RUN_MATERIALIZED_BENCHMARK=0 \
SMU3M_SLURM_TIME=08:00:00 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_diffusion_refine.sh
```

To sweep `prior_loss_weight` without overwriting the current best checkpoint,
write each run to separate diffusion/eval directories while resuming
from the base `latent_diffusion` checkpoint:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_BASE_DIFFUSION_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/latent_diffusion \
SMU3M_DIFFUSION_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/latent_diffusion_prior050 \
SMU3M_EVAL_LATENT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/eval_latent_prior050_full \
SMU3M_DIFFUSION_EXTRA_EPOCHS=100 \
SMU3M_PRIOR_LOSS_WEIGHT=0.5 \
SMU3M_EVAL_LIMIT=0 \
SMU3M_MAX_EVAL_PER_PROPERTY_COUNT=0 \
SMU3M_RUN_MATERIALIZED_BENCHMARK=0 \
SMU3M_SLURM_TIME=08:00:00 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_diffusion_refine.sh
```

After that sweep job writes `eval_latent_prior050_full`, benchmark it separately:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2 \
SMU3M_EVAL_LATENT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/eval_latent_prior050_full \
SMU3M_BENCHMARK_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/benchmark_materialized_prior050_primary_fast \
SMU3M_BENCHMARK_PROFILE=primary_fast \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

## Latest Run Results

Artifacts live under `outputs/unified_generation_3m_edit_v2/`.

### Current best: job `15695020` (Jun 6 2026, ~9 min)

Joint connector + diffusion refine from epoch 150 to 250 with
`train_diffusion_connector=1`, `prior_loss_weight=0.25`, balanced eval
(1307 samples, up to 250 per property count), and materialized benchmark.
Log: `logs/smu3m-diff-refine-15695020.log`.

| Stage | Epochs | Final train loss | Status |
| --- | ---: | ---: | --- |
| Latent diffusion + connector | 151–250 | 0.162 → **0.021** | converged |
| `diffusion_target_mae` | 151–250 | 0.147 → **0.041** | improved |
| `prior_mse` | 151–250 | 0.304 → **0.060** | improved |

Eval on 1307 edit samples:

| Metric | `15694324` (ep150) | `15695020` (ep250) |
| --- | ---: | ---: |
| `prior_target_property_mae` | 7.43 | **6.74** |
| `target_property_mae` | 7.27 | **6.75** |
| `target_fingerprint_cosine` | 0.381 | 0.386 |
| `generated_minus_prior_latent_mae` | 0.071 | **0.020** |

Materialized benchmark joint strict all (1307 samples, 2p–7p):

| Method | `15694324` | `15695020` |
| --- | ---: | ---: |
| `scaffold_property_retrieval` | 0.593 (1000, 2p-only table) | **0.317** |
| `edit_latent_scaffold_source_rerank` | 0.453 | **0.297** |
| `edit_latent_scaffold_retrieval` | 0.453 | **0.299** |
| `source_identity` | 0.446 | 0.152 |

Per-property strict for `edit_latent_scaffold_source_rerank` vs SketchMol reference:

| | 2p | 3p | 4p | 5p |
| --- | ---: | ---: | ---: | ---: |
| edit latent | 0.640 | 0.420 | 0.248 | 0.168 |
| scaffold retrieval | 0.656 | 0.460 | 0.272 | 0.172 |
| SketchMol reference | 0.804 | 0.768 | 0.736 | 0.716 |

Joint connector training fixed the frozen-prior bottleneck. Edit-latent scaffold
retrieval now tracks the scaffold-property baseline (~0.30 joint strict) but
remains below SketchMol structured reference, especially on 3p+ tasks.
`edit_latent_global_retrieval` has high per-property strict with scaffold all = 0,
so it is not a valid source-conditioned method.

### Frozen-prior plateau: job `15694324` (Jun 6 2026, ~3.7 min)

Resumed residual diffusion from epoch 50 to 150 with `lr=3e-4`, then reran
eval and materialized benchmark. Log: `logs/smu3m-diff-refine-15694324.log`.

| Stage | Epochs | Final train loss | Status |
| --- | ---: | ---: | --- |
| Latent diffusion | 51–150 | 0.067 → **0.034** | loss down |
| `diffusion_target_mae` | 51–150 | ~0.146 | flat |

Eval on 1000 edit samples:

| Metric | `15692318` (ep50) | `15694324` (ep150) |
| --- | ---: | ---: |
| `target_fingerprint_cosine` | 0.392 | 0.381 |
| `target_property_mae` | 7.25 | 7.27 |
| `generated_minus_prior_latent_mae` | 0.045 | **0.071** |
| `prior_target_property_mae` | 7.43 | 7.43 |

Materialized benchmark joint strict (2p):

| Method | `15692318` | `15694324` |
| --- | ---: | ---: |
| `scaffold_property_retrieval` | 0.593 | 0.593 |
| `edit_latent_scaffold_source_rerank` | 0.454 | 0.453 |
| `source_identity` | 0.446 | 0.446 |

Extra diffusion epochs lowered MSE loss but did not improve retrieval metrics.
The frozen connector prior still dominates (`prior_target_property_mae` ≈ 7.43).
Next step: joint connector + diffusion fine-tune (`SMU3M_TRAIN_DIFFUSION_CONNECTOR=1`).

### Residual fix: job `15692318` (Jun 6 2026, ~2.5 min)

Resumed alignment/connector from epoch 50, retrained Stage 3 diffusion with the
new defaults (`pred_x0`, `residual` target, DDIM `sample_eta=0.0`). Log:
`logs/smu3m-unified-15692318.log`.

| Stage | Epochs | Final train loss | Status |
| --- | ---: | ---: | --- |
| Alignment | 50 | resumed @ ep50 | skipped |
| Edit connector | 50 | resumed @ ep50 | skipped |
| Latent diffusion | 50 | 0.268 → **0.068** | converged |

Eval on 1000 edit samples (`eval_latent/metrics.json`):

| Metric | Value |
| --- | ---: |
| `source_fingerprint_cosine` | 0.371 |
| `target_fingerprint_cosine` | 0.392 |
| `source_target_fingerprint_cosine` (GT pairs) | 0.799 |
| `target_property_mae` | 7.25 |
| `latent_mae` / `latent_mse` | 0.214 / 5.54 |
| `prior_target_fingerprint_cosine` | 0.409 |
| `generated_minus_prior_latent_mae` | 0.045 |

Diffusion collapse is fixed. Generated latents are fingerprint-aligned with
source/target again. Residual diffusion still adds only a small delta on top of
the source prior (`generated_minus_prior_latent_mae` ≈ 0.045), so property edit
signal remains weak.

### Baseline collapse: job `15690752` (Jun 6 2026, ~4 min)

Full 3×50-epoch run before the residual Stage 3 fix. Log:
`logs/smu3m-unified-15690752.log`.

| Metric | Value |
| --- | ---: |
| `source_fingerprint_cosine` | 0.008 |
| `target_fingerprint_cosine` | 0.009 |
| `target_property_mae` | 554.7 |
| `latent_mae` / `latent_mse` | 19.8 / 30027 |

Stage 3 loss stuck near 1.0; generated latents were unrelated to source/target.
This run motivated the current residual defaults and prior-vs-generated diagnostics.

## Validation

```bash
cd SketchMol-Unified-3MDiffusion
python -m pytest -q
```
