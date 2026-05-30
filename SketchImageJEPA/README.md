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
RDKit Morgan-fingerprint molecular latents with deterministic fallbacks,
hashed context features, a retrieval decoder, deterministic train/eval
splitting, optional RDKit-rendered image context, and a toy SketchMol-aligned
benchmark. This gives us a fast runnable surface before moving to a larger
learned decoder and ChEMBL/PubChem runs.

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

If torch is already installed in a different venv, list candidates:

```bash
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" bash scripts/find_torch_python.sh
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

For faster iteration, submit a three-variant GPU sweep instead of waiting for
one run at a time:

```bash
SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v10_contrastive_temp \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
bash scripts/submit_torch_sweep.sh
```

The sweep submits `contrastive_cool`, `contrastive_cooler`, and
`contrastive_cold` variants around the current best contrastive-temperature
setting.
Compare them after completion:

```bash
SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v10_contrastive_temp \
bash scripts/summarize_torch_sweep.sh
```

If the best sweep run has good top-k recall but weak top-1 ordering, run the
CPU-only rerank diagnostic:

```bash
bash scripts/rerank_run.sh outputs/runs/sketchmol_aligned_torch_50k_10k_v10_contrastive_temp_contrastive_cool
```

For property-transform runs, the same command also sweeps
`property_delta_weight` when `predictions.csv` contains `property_delta_mae`.
It writes `best_by_objective.csv` so target, property-delta, property, and
balanced objectives can be inspected separately. Override the grid with
`SKETCHIMAGE_RERANK_PROPERTY_DELTA_WEIGHTS` if needed.

## Phase 1 Decoder Check

Before adding the JEPA planner, test whether a decoder can generate molecules
from an oracle molecular latent. This follows the staged SketchMol logic:
first make the decoder capable, then connect the planner.

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_RUN_NAME=phase1_oracle_latent_diffusion_seed7 \
bash scripts/submit_oracle_latent_diffusion.sh
```

This run now defaults to a latent-conditioned autoregressive token decoder,
because the first denoising-only pilot produced zero valid SMILES. The
denoising objective is still available with
`SKETCHIMAGE_ORACLE_OBJECTIVE=denoising`, but it should be treated as a failed
diagnostic until the decoder is redesigned around image or graph latents.

```text
target molecule SMILES
  -> oracle molecular latent
  -> latent-conditioned token decoder
  -> sampled SMILES candidates
  -> RDKit validity, Tanimoto, exact-match, and train-pool diagnostics
```

It is not the final SketchMol-style image decoder yet. It is a Phase 1.0
capability check: if this oracle-latent decoder is weak, adding a JEPA planner
cannot fix the generation bottleneck.

## Phase 2A Planned Decoder

After Phase 1 proves the frozen decoder can use oracle molecular latents, Phase
2A trains the JEPA planner to predict that decoder latent and then samples
through the frozen Phase 1 model:

```text
condition / source / mask / image context
  -> JEPA planner predicted molecular latent
  -> frozen Phase 1 latent-conditioned token decoder
  -> sampled SMILES candidates
  -> RDKit verification and train-pool diagnostics
```

Start with the 2048-dimensional Phase 1 decoder because it was close to the
4096-dimensional decoder but cheaper to train and decode:

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase2_planned_decoder_2048_seed7 \
bash scripts/submit_phase2_planned_decoder.sh
```

This writes the normal run artifacts plus `planner_latent_mse` and
`planner_latent_cosine`, which separate planner failure from decoder failure.

## Phase 2B Robust Decoder

Phase 2A showed that a decoder trained only on oracle latents can collapse when
fed JEPA-predicted latents. Phase 2B starts from the Phase 1 decoder weights and
fine-tunes the decoder on mixed conditions:

```text
target SMILES paired with:
  oracle target latent
  planner-predicted latent
  noisy target latent
  interpolation between oracle and planner latents
```

Run this after Phase 2A or directly after Phase 1 plus the hard split:

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase2_robust_decoder_2048_seed7 \
bash scripts/submit_phase2_robust_decoder.sh
```

Read this against Phase 2A first: if validity recovers, the main issue was
decoder robustness to off-manifold planner latents. If validity stays low, the
planner latent distribution or molecular latent representation needs a deeper
change.

## Phase 2C Latent Calibration

Phase 2B can recover validity while still leaving target similarity weak. Phase
2C keeps the robust decoder fixed and trains a small adapter:

```text
planner-predicted latent
  -> residual ridge calibration adapter
  -> calibrated latent closer to oracle molecular latent
  -> frozen robust decoder
  -> sampled SMILES candidates
```

Run it after Phase 2B:

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_DECODER_DIR=outputs/runs/phase2_robust_decoder_2048_seed7/decoder \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase2_calibrated_decoder_2048_seed7 \
bash scripts/submit_phase2_calibrated_decoder.sh
```

Compare `planner_latent_cosine` with `calibrated_latent_cosine`, then compare
Phase 2B against Phase 2C on validity, `top1_target_tanimoto`, and
`mean_best_tanimoto`.

## Latent Sensitivity Diagnostic

After Phase 2C, run the diagnostic that keeps the decoder fixed and compares
controlled latent sources:

```text
oracle target latent
noisy oracle latent at matched cosine levels
planner-predicted latent
calibrated planner latent
target/planner interpolations
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_DECODER_DIR=outputs/runs/phase2_robust_decoder_2048_seed7/decoder \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_PLANNER_RUN_DIR=outputs/runs/phase2_robust_decoder_2048_seed7 \
SKETCHIMAGE_CALIBRATED_RUN_DIR=outputs/runs/phase2_calibrated_decoder_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase2_latent_sensitivity_2048_seed7 \
bash scripts/submit_latent_sensitivity.sh
```

Read `outputs/runs/<run_name>/source_summary.csv` first. If noisy oracle
latents fail at the same cosine as planner latents, the planner needs stronger
latent alignment. If noisy oracle stays strong but planner/calibrated fail, the
decoder needs planner-distribution conditioning rather than more oracle-only
training.

## Phase 2D Oracle-Anchored Robust Decoder

Phase 2D tries to keep the strong Phase 1 oracle control while adding light
planner-latent robustness. It starts from the Phase 1 decoder, then fine-tunes
with many oracle target rows and only small exposure to noisy, planner,
calibrated, and interpolation latents:

```text
mostly oracle target latent
  + small noisy oracle basin
  + light planner/calibrated/interpolation exposure
  -> oracle-anchored robust decoder
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_PLANNER_RUN_DIR=outputs/runs/phase2_robust_decoder_2048_seed7 \
SKETCHIMAGE_CALIBRATED_RUN_DIR=outputs/runs/phase2_calibrated_decoder_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase2_oracle_anchored_decoder_2048_seed7 \
bash scripts/submit_phase2_oracle_anchored_decoder.sh
```

Read `source_summary.csv` first. The target is to keep `oracle_target`
substantially stronger than Phase 2B while improving `planner_predicted` or
`calibrated_predicted` over the prior robust decoder.

## Phase 3A Decoder-Compatible Planner

Phase 2D suggests decoder-only tuning is not enough: oracle latents decode
well, but planner latents remain decoder-hostile. Phase 3A keeps the Phase 1
decoder frozen and changes the planner objective. It adds optional norm and
decoder-compatibility losses so the predicted latent is pushed toward the
oracle/noisy-oracle basin that the decoder can read.

```text
condition / source / mask / image context
  -> JEPA planner with norm + cosine-floor compatibility loss
  -> frozen Phase 1 latent-conditioned token decoder
  -> sampled SMILES candidates
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase3_decoder_compatible_planner_2048_seed7 \
bash scripts/submit_phase3_decoder_compatible_planner.sh
```

Read `metrics.json` first. The new planner diagnostics are
`planner_latent_cosine`, `planner_cosine_ge_margin`, and the
`planner_eval_*_norm_*` fields.

## Phase 4A Edit-Action Planner

Phase 4A changes the interface instead of asking the planner to guess the full
target latent. It filters to source-conditioned tasks (`edit`, `inpaint`, and
`fragment_grow`), learns an action latent, and composes candidates by moving
from the source latent:

```text
source molecule latent + task condition
  -> predicted edit/action latent
  -> normalize(source latent + alpha * action latent)
  -> frozen Phase 1 latent-conditioned token decoder
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase4_edit_action_planner_2048_seed7 \
bash scripts/submit_phase4_edit_action_planner.sh
```

Read `metrics.json` and `alpha_summary.csv` first. This run reports excluded
de novo rows separately, plus `action_latent_cosine` and
`composed_latent_cosine` so we can tell whether the edit action or the decoder
is limiting quality.

## Phase 4B Normalized Edit-Action Planner

Phase 4B keeps the Phase 4 interface, but fixes a failure mode where the
planner predicts action vectors with much larger norm than the true edit. It
trains the planner on the edit direction and restores the step size from the
training action-norm distribution before composing the latent:

```text
source molecule latent + task condition
  -> predicted unit edit direction
  -> median train edit step * direction
  -> normalize(source latent + alpha * scaled direction)
  -> frozen Phase 1 latent-conditioned token decoder
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase4b_normalized_action_planner_2048_seed7 \
bash scripts/submit_phase4b_normalized_action_planner.sh
```

Read `action_eval_step_*`, `action_train_step_*`, `action_latent_cosine`, and
`composed_latent_cosine` first. If Phase 4A was limited by over-large action
norms, Phase 4B should improve composed latent cosine and the alpha-beam
target metrics without changing the frozen decoder.

## Phase 4C Retrieval-Guided Edit-Action Planner

Phase 4C targets the remaining Phase 4B failure mode: the action step is now
well calibrated, but the edit direction still generalizes poorly. It predicts
the normalized edit direction, retrieves nearby training edits in
condition-plus-source latent space, and blends the predicted direction with
neighbor target directions before composing the candidate latent.

```text
source molecule latent + task condition
  -> predicted unit edit direction
  -> nearest train edit directions
  -> blended/corrected edit direction
  -> median train edit step * corrected direction
  -> frozen Phase 1 latent-conditioned token decoder
```

```bash
cd SketchImageJEPA
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=phase4c_retrieval_action_planner_2048_seed7 \
bash scripts/submit_phase4c_retrieval_action_planner.sh
```

Read the corrected and uncorrected diagnostics side by side:
`action_latent_cosine` vs `uncorrected_action_latent_cosine`,
`composed_latent_cosine` vs `uncorrected_composed_latent_cosine`, and the
`action_eval_correction_*` fields. If retrieval helps, the corrected cosine
and target metrics should move together.

## Paper Track

The paper direction is documented in `docs/research_questions.md`. Instead of
continuing open-ended temperature sweeps, run a controlled ablation matrix:

```bash
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
bash scripts/submit_paper_matrix.sh
```

The default pilot matrix runs one seed for `ridge_baseline`, `planner_best`,
`planner_v2`, `no_contrastive`, and `no_image_context`. `planner_v2` adds an
explicit edit-delta loss and hard-negative loss. For a three-seed paper table,
set:

```bash
SKETCHIMAGE_PAPER_MODE=full bash scripts/submit_paper_matrix.sh
```

Summarize completed matrix runs:

```bash
SKETCHIMAGE_PAPER_MODE=full \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/summarize_paper_matrix.sh
```

Audit an existing run for train-set shortcut difficulty:

```bash
bash scripts/submit_audit_benchmark.sh outputs/runs/sketchmol_aligned_paper_pilot_ridge_baseline_seed7
```

Build a hard split before running the matrix:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
SKETCHIMAGE_HARD_SPLIT_NAME=sketchmol_hard_seed7 \
bash scripts/submit_hard_split.sh
```

Then run the paper matrix on the hard split:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_paper_pilot \
bash scripts/submit_paper_matrix.sh
```

New Slurm logs written by submit helpers go to `outputs/logs/`.

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

## Stage-C Generative Decoder Prototype

The retrieval decoder can only rank molecules from the training target pool.
For hard scaffold splits this creates a ceiling: if the desired scaffold is not
already in the pool, top-k target hits collapse. The first Stage-C prototype
keeps the current torch planner but switches the candidate surface to a hybrid
generative decoder. It retrieves planner seeds, locally mutates them with
RDKit-compatible edits, and ranks generated candidates using latent, source,
property, and scaffold cues without using the hidden target.

Submit the prototype on a Slurm login node:

```bash
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_stage_c_generative_seed7 \
bash scripts/submit_generative_decoder.sh
```

For the hard split, reuse the CPU-built train/eval CSVs:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_stage_c_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_generative planner_generative_only no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

The next stronger decoder learns edit transforms from the training source-target
pairs before applying them to eval sources. This is the first prototype aimed
at preserving source-related structure rather than making generic mutations:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_learned_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_learned_transform_decoder.sh
```

To compare learned transforms against the mutation decoder and the best hard
diagnostic in one matrix:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_learned_transform_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_learned_transform planner_learned_transform_only planner_generative no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

For source-conditioned tasks, the scaffold-preserving transform decoder is the
next stricter variant. It applies learned transforms to the eval source and
filters/reranks candidates by source core retention. The run writes
`source_scaffold_retained` in `predictions.csv` plus aggregate retention
metrics in `metrics.json`.

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_scaffold_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_scaffold_transform_decoder.sh
```

Compact comparison:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_scaffold_transform_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_scaffold_transform planner_scaffold_transform_only planner_learned_transform no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

The property-conditioned transform decoder is the next stricter variant. It
keeps source-core retention, but selects learned transforms whose training
property delta matches the requested edit direction. The run writes
`property_delta_mae` and `property_delta_success` in `predictions.csv`, plus
aggregate delta metrics in `metrics.json` and `task_type_summary.csv`.

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_property_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_property_transform_decoder.sh
```

Compact comparison:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_property_transform_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_property_transform planner_property_transform_only planner_scaffold_transform no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

The latent-conditioned transform beam decoder addresses the failure mode where
property-delta candidates exist in top-k but are structurally far from the
target. It expands source edits as a beam and scores every expansion with the
predicted molecular latent, property-delta match, and source-core retention.

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_latent_beam_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_latent_beam_transform_decoder.sh
```

Compact comparison:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_latent_beam_transform_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_latent_beam_transform planner_latent_beam_transform_only planner_property_transform no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

The supervised edit-policy transform decoder tests the next research
hypothesis: the planner should not only rank heuristic mutations, but should
select learned molecular edit actions supervised by source-target pairs. It
learns a lightweight transform policy from training edits, then uses the
predicted latent plan to rank edit actions during generation.

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_edit_policy_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_edit_policy_transform_decoder.sh
```

Compact comparison:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_edit_policy_transform_pilot \
SKETCHIMAGE_PAPER_VARIANTS="planner_edit_policy_transform planner_edit_policy_transform_only planner_latent_beam_transform planner_property_transform no_contrastive" \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

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
