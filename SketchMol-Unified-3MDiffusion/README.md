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

MolEdit smoke (preferred):

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/run_unified_moledit_smoke.sh
```

Legacy multi-property smoke:

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

Primary path: MolEdit-Instruct enhanced splits on `$DM_DATA_ROOT`:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_pipeline.sh
```

After the training/eval job has written `eval_latent/generated_latents.npy`, run
the MolEdit benchmark wrapper:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_benchmark.sh
```

That wrapper exports the matching
`dataset/moledit_benchmark_condition_rows.csv`, submits the materialized
benchmark, and for a single-shard run submits MolEdit table metrics with an
`afterok` dependency.

Legacy multi-property / source-neighbor runs still use
`submit_unified_generation_pipeline.sh`. Override `SMMED_OUTPUT_DIR`,
`SMU3M_EDIT_MANIFEST`, and `SMU3M_DATASET_MODE=multiproperty` only when
rerunning older SketchMol image-manifest experiments.

## MolEdit-Instruct Training

The pipeline can now train directly from the enhanced MolEdit-Instruct splits
built under `$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/`. This is the
path to use when aligning with MolEditRL-style text/SMILES editing tasks rather
than the older SketchMol multi-property manifest:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_pipeline.sh
```

For a small interactive run:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/run_unified_moledit_smoke.sh
```

These wrappers set:

```text
SMU3M_DATASET_MODE=moledit
SMU3M_MOLEDIT_TRAIN_SPLIT=$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv
SMU3M_MOLEDIT_EVAL_SPLIT=$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv
SMU3M_MIN_EDIT_SOURCE_TANIMOTO=0.0
SMMED_SUBMIT_DATASET_BUILD=0
```

The MolEdit adapter maps enhanced split columns into the same
`unified_condition_train.jsonl` / `unified_condition_eval.jsonl` schema used by
the existing connector and latent-diffusion stages. RDKit properties
(`MW`, `LogP`, `QED`, `TPSA`, `HBD`, `HBA`, `RB`) feed the numeric heads, while
non-RDKit task tags such as `GSK3B`, `DRD2`, and `SA` are preserved in metadata
for table-style evaluation.

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
multi-property benchmark. For MolEdit-Instruct outputs, use the dedicated
wrapper so `condition_rows.csv` is exported from the MolEdit eval split instead
of falling back to legacy source-neighbor rows:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_benchmark.sh
```

The wrapper uses `eval_latent/generated_latents.npy` plus
`dataset/unified_condition_eval.jsonl`. If the eval directory was produced
before benchmark export existed, it automatically writes:

```text
eval_latent/edit_latent_predictions.npy     # target props / deltas / active / directions
eval_latent/edit_latent_fingerprints.npy    # Unified target fingerprint block
eval_latent/index.csv                       # condition_id alignment for the benchmark
```

For MolEdit mode it also writes:

```text
dataset/moledit_benchmark_condition_rows.csv
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
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/benchmark_materialized/
  benchmark_report.md
  benchmark_summary.csv
  benchmark_decoded.csv
```

For the default fast profile, the directory is:

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/benchmark_materialized_primary_fast/
```

Use the generic materialized submit script directly only for legacy or custom
multi-property outputs:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v1 \
SMU3M_DATASET_MODE=multiproperty \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

With `SMU3M_BENCHMARK_SHARDS=5`, the submit script calls `sbatch` five times by
default, so each shard gets its own job id. Each shard writes to
`benchmark_materialized_primary_fast/shards/shard_<i>_of_5/`. After all shards
finish, merge the shard outputs:

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1 \
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

## MolEditRL Table Metrics

The MolEditRL paper table reports per-task `Validity`, `Acc_all(0.65)`,
`Acc_valid(0.65)`, `Acc_all(0.15)`, `Acc_valid(0.15)`, and `FCD`. Use the
MolEdit table metric script on any prediction CSV with `example_id` or
`condition_id` plus `predicted_smiles` or `generated_smiles`. It also accepts the
existing materialized benchmark `benchmark_decoded.csv` format with `method`.

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_table_metrics.sh
```

`MW`, `LogP`, `QED`, `HBA`, and rotatable-bond tasks are scored with RDKit.
`GSK3B`, `DRD2`, and `SA` use `tdc.Oracle` when available; install TDC in the
cluster environment for full Table 1 coverage. The script writes:

```text
moledit_table_summary.csv
moledit_table_summary.json
moledit_table_summary.md
```

## Legacy multiproperty experiments

Source-aware connector sweeps, source-anchor sweeps, and diffusion-refine wrappers
from the old multi-property line live under `scripts/archive/legacy/`. Their
outputs were removed from `outputs/` to free disk space.

For new work, use **MolEdit-Instruct Training** above and evaluate with
**MolEditRL Table Metrics**. Legacy multiproperty training remains available via
`submit_unified_generation_pipeline.sh` when explicitly needed.

## Validation

```bash
cd SketchMol-Unified-3MDiffusion
python -m pytest -q
```
