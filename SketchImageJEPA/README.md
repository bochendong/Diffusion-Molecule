# SketchImage-JEPA

Standalone project for testing whether a JEPA-style molecular planner can
replace the fragile image-generation-to-OCR middle of SketchMol-style molecule
generation and editing.

The project is intentionally independent. It does not import code from
SketchMol, MolPilot, PhysTabMol, or the local paper repositories. Those projects
are used only as research references and benchmark inspiration.

## Research Question

SketchMol-style pipelines are attractive because they unify molecule generation,
fragment growing, and inpainting in an image space. The weak point is that the
final molecule must be recovered from generated pixels:

```text
condition / source / mask
  -> molecular image latent diffusion
  -> generated molecule image
  -> OCR or image-to-structure recognition
  -> RDKit validation and ranking
```

SketchImage-JEPA tests a different route:

```text
condition / source / mask / optional image statistics
  -> context encoder features
  -> JEPA-style target latent predictor
  -> molecular latent decoder or retrieval baseline
  -> RDKit-compatible verification and ranking
```

The first implementation is deliberately small: a numpy ridge JEPA predictor,
hashed molecular/context features, a retrieval decoder, deterministic
train/eval splitting, optional RDKit-rendered image context, and a toy
SketchMol-aligned benchmark. This gives us a fast runnable surface before
moving to a torch model and larger ChEMBL/PubChem runs.

## Quick Start

```bash
cd SketchImageJEPA
python3 -m pip install -e .
bash scripts/run_sketchmol_aligned.sh
```

If your default Python does not have `numpy`, run the script with an explicit
interpreter:

```bash
PYTHON_BIN=/path/to/python3 bash scripts/run_smoke.sh
```

The one-click scripts follow the same convention:

```bash
SKETCHIMAGE_PYTHON_BIN=/path/to/python3 bash scripts/run_sketchmol_aligned.sh
```

## GPU Backend

The default `ridge` backend is a CPU benchmark harness. For a real GPU run,
use the PyTorch latent denoising backend:

If the selected venv cannot import torch, install it once:

```bash
MODULE_RDKIT=rdkit/2025.09.4 \
VENV_DIR=/scratch/bdong/venvs/sketchimage-rdkit \
bash scripts/setup_torch_venv.sh
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_BACKEND=torch_denoiser \
SKETCHIMAGE_PYTHON_BIN=/path/to/python-with-torch \
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
bash scripts/run_torch_denoiser.sh
```

On a Slurm login node, submit the GPU job:

```bash
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_torch_50k_10k_v1 \
SKETCHIMAGE_PYTHON_BIN=/path/to/python-with-torch \
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
SKETCHIMAGE_TORCH_EPOCHS=25 \
bash scripts/submit_torch_denoiser.sh
```

Default GPU request:

```text
h100 10GB MIG = 1
cpus-per-task = 8
mem = 64G
time = 8h
```

The submit helper defaults to `SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig` and tries
both GPU names used by neighboring project scripts:
`nvidia_h100_80gb_hbm3_1g.10gb:1` and `h100_1g.10gb:1`. For a larger slice,
set `SKETCHIMAGE_GPU_PROFILE=h100_20gb_mig` or
`SKETCHIMAGE_SLURM_GPUS=<cluster_gpu_name>:1`.

The torch backend trains a conditional latent denoising model and writes the
same `metrics.json`, `predictions.csv`, `task_type_summary.csv`, and
`run_config.json` artifacts as the CPU run. The decoder still uses the shared
RDKit/property evaluation path, so CPU and GPU runs are directly comparable.

The smoke run writes:

```text
outputs/smoke/metrics.json
outputs/smoke/predictions.csv
outputs/smoke/run_config.json
outputs/smoke/train_examples.csv
outputs/smoke/eval_examples.csv
outputs/smoke/model/
```

The default comparison run uses SketchMol-aligned parameters:

```text
condition/context dim = 256
molecular latent dim = 32 x 32 x 4 = 4096
candidates per condition = 8
image size reference = 256
SketchMol DDIM reference = 250 steps, eta 1.0, scale 2, scale_pro 4
```

It writes a timestamped directory under `outputs/runs/` and stores the full
reference knob set in `run_config.json`. Useful overrides:

Build tasks from a molecule CSV and run:

```bash
SKETCHIMAGE_MOLECULE_CSV=data/example_molecules.csv \
SKETCHIMAGE_RUN_NAME=first_server_check \
bash scripts/run_sketchmol_aligned.sh
```

The molecule CSV can use `smiles`, `SMILES`, or `canonical_smiles` as the SMILES
column. The generated task CSV is saved under `outputs/tasks/`.

Run from an already-built task CSV:

```bash
SKETCHIMAGE_DATASET_CSV=data/example_tasks.csv \
SKETCHIMAGE_RUN_NAME=first_server_check \
SKETCHIMAGE_RENDER_IMAGE_CONTEXT=1 \
bash scripts/run_sketchmol_aligned.sh
```

For explicit splits:

```bash
SKETCHIMAGE_TRAIN_CSV=data/train.csv \
SKETCHIMAGE_EVAL_CSV=data/eval.csv \
bash scripts/run_sketchmol_aligned.sh
```

For a tiny health check that does not use the 4096-dimensional aligned latent,
use:

```bash
bash scripts/run_smoke.sh
```

On a Slurm login node, submit instead of running directly:

```bash
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
bash scripts/submit_sketchmol_aligned.sh
```

Run the bundled CSV example:

```bash
cd SketchImageJEPA
PYTHONPATH=. python3 -m sketchimage_jepa.experiment \
  --dataset-csv data/example_tasks.csv \
  --output-dir outputs/example_csv \
  --train-fraction 0.67 \
  --render-image-context
```

For server data, provide one combined CSV with `--dataset-csv` or explicit
splits with `--train-csv` and `--eval-csv`. The CSV columns are:

```text
task_id,task_type,source_smiles,target_smiles,instruction,mask_hint,image_path,goals
```

`task_type` must be one of `de_novo`, `edit`, `inpaint`, or `fragment_grow`.
`goals` uses semicolon-separated tags. `image_path` is optional; when
`--render-image-context` is enabled and RDKit is installed, missing image paths
are filled with rendered source molecule images. De novo rows without a source
are skipped unless they already provide an `image_path`, because target images
would leak the answer. Examples with a `mask_hint` also get a deterministic mask
overlay as a first proxy for SketchMol-style inpainting context.

Run tests:

```bash
cd SketchImageJEPA
python3 -m pip install -e .
PYTHONPATH=. python3 -m unittest discover -s tests
```

## Current Tasks

The toy benchmark covers the task shapes we want to compare against
SketchMol-style evaluations:

- `de_novo`: property-guided generation without a source molecule.
- `edit`: source molecule plus text instruction to target a nearby molecule.
- `inpaint`: source molecule plus a mask hint to preserve part of the molecule.
- `fragment_grow`: source fragment plus instruction to grow toward a target.

## Metrics

The smoke benchmark reports:

- `top1_validity`
- `top1_target_tanimoto`
- `top1_scaffold_match`
- `topk_target_hit`
- `mean_best_tanimoto`

When RDKit is unavailable, the project falls back to deterministic string-level
descriptors and similarities so the smoke test still runs. Server runs should
install RDKit for real chemistry metrics.

The model is trained only on `train_examples.csv` and evaluated on
`eval_examples.csv`. Target SMILES are used as prediction labels and metrics,
not as context features.

## Next Milestones

1. Replace the ridge predictor with a torch JEPA model using smooth latent and
   delta losses.
2. Add a larger real molecule CSV builder from ChEMBL/PubChem-like sources.
3. Replace the deterministic mask overlay with atom/region-aware RDKit masks.
4. Add source-guided graph edits and SELFIES/graph decoder candidates.
5. Run a SketchMol-aligned benchmark with validity, property hit, fragment
   growth success, scaffold preservation, novelty, and local edit success.
