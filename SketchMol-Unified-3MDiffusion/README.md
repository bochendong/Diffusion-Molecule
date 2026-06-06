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

scripts/
  preflight_unified_3m.py
  export_unified_condition_dataset.py
  train_alignment_pretraining.py
  train_edit_condition_tokens.py
  train_latent_diffusion_generation.py
  evaluate_latent_diffusion_generation.py
  run_unified_generation_smoke.sh
  submit_unified_generation_pipeline.sh
```

## Local Smoke

From the repository root:

```bash
SMU3M_3M_ROOT="Research/Molecule Generation/3M-Diffusion" \
SMU3M_EDIT_MANIFEST="SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/diffusion_edit_manifest.csv" \
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
SMU3M_EVAL_SAMPLE_STEPS
SMU3M_DIFFUSION_TIMESTEPS
SMU3M_DEVICE
SMU3M_REQUIRE_CUDA
SMU3M_CHECKPOINT_EVERY
SMU3M_RESUME
SMU3M_INCLUDE_PUBCHEM
SMU3M_INCLUDE_KV
```

Default output:

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_smoke/
  dataset/summary.json
  alignment/alignment_model.pt
  edit_condition_tokens/edit_condition_connector.pt
  latent_diffusion/latent_diffusion_generation.pt
  eval_latent/metrics.json
```

## Slurm

On a Slurm login node:

```bash
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

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
they exist.

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

## Latest Run Results

Server job `15690752` (`smu3m-unified`, Jun 6 2026) completed the full pipeline on
H100 MIG 3g.40gb in ~4 minutes. Artifacts live under
`outputs/unified_generation_3m_edit_v2/`; log:
`logs/smu3m-unified-15690752.log`.

| Stage | Epochs | Final train loss | Status |
| --- | ---: | ---: | --- |
| Alignment | 50 | 0.485 | converged |
| Edit connector | 50 | 0.416 | converged |
| Latent diffusion | 50 | ~1.0 (spike 9.7 @ ep31) | not learning |

Eval on 1000 edit samples (`eval_latent/metrics.json`):

| Metric | Value |
| --- | ---: |
| `source_fingerprint_cosine` | 0.008 |
| `target_fingerprint_cosine` | 0.009 |
| `source_target_fingerprint_cosine` (GT pairs) | 0.799 |
| `target_property_mae` | 554.7 |
| `latent_mae` / `latent_mse` | 19.8 / 30027 |

Alignment and connector stages trained normally, but latent diffusion collapsed
(loss stuck near 1.0). Generated latents are unrelated to source/target fingerprints.
Next step: debug Stage 3 diffusion (latent normalization, loss target, training schedule).

## Validation

```bash
cd SketchMol-Unified-3MDiffusion
python -m pytest -q
```
