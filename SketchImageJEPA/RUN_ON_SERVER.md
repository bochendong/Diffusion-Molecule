# Run SketchImage-JEPA On Server

From the server checkout:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA
bash scripts/run_sketchmol_aligned.sh
```

Do not run a large experiment directly on a login node. Use the login node for
`git pull` and `sbatch`; run the experiment on a Slurm compute node.

Clean old root-level Slurm logs if they are no longer needed:

```bash
rm -f sketchimage-gpu-*.log sketchimage-jepa-*.log sketchimage-cpu-*.log
mkdir -p outputs/logs
```

New submit helpers write logs under `outputs/logs/` instead of cluttering the
project root.

Submit from the login node:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
bash scripts/submit_sketchmol_aligned.sh
```

## GPU Torch Backend

The CPU script is still useful for fast checks. To run the GPU-capable
conditional latent denoiser, use a Python environment with `torch`, `numpy`,
and RDKit available, then submit:

If your current `sketchimage-rdkit` venv does not have torch yet, install it
once from the login node:

```bash
cd "/path/to/Diffusion Molecule/SketchImageJEPA"
MODULE_RDKIT=rdkit/2025.09.4 \
VENV_DIR=/scratch/bdong/venvs/sketchimage-rdkit \
bash scripts/setup_torch_venv.sh
```

If torch already exists in another venv, find it first:

```bash
cd "/path/to/Diffusion Molecule/SketchImageJEPA"
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" bash scripts/find_torch_python.sh
```

Then submit with the matching interpreter:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/<torch-venv>/bin/python \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
bash scripts/submit_torch_denoiser.sh
```

To avoid waiting on one hyperparameter guess at a time, submit the three-job
torch sweep. It launches `contrastive_cool`, `contrastive_cooler`, and
`contrastive_cold` variants as separate 10GB MIG jobs:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v10_contrastive_temp \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
bash scripts/submit_torch_sweep.sh
```

After the jobs finish:

```bash
SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v10_contrastive_temp \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/summarize_torch_sweep.sh
```

If a run has high `mean_best_tanimoto` but low `top1_target_tanimoto`, sweep
reranking weights on the completed `predictions.csv` without using more GPU:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/rerank_run.sh outputs/runs/sketchmol_aligned_torch_50k_10k_v10_contrastive_temp_contrastive_cool
```

For property-transform runs, use the same CPU command. It now includes a
`property_delta_weight` sweep when `predictions.csv` has `property_delta_mae`:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/rerank_run.sh outputs/runs/sketchmol_hard_property_transform_seed7
```

This writes:

```text
outputs/runs/<run_name>/rerank_diagnostics/rerank_sweep_summary.csv
outputs/runs/<run_name>/rerank_diagnostics/best_by_objective.csv
outputs/runs/<run_name>/rerank_diagnostics/best_task_type_summary.csv
outputs/runs/<run_name>/rerank_diagnostics/best_reranked_predictions.csv
```

## Paper Matrix

For the top-conference track, switch from open-ended hyperparameter sweeps to
the controlled ablation matrix:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
bash scripts/submit_paper_matrix.sh
```

The default pilot matrix submits one seed for `ridge_baseline`,
`planner_best`, `planner_v2`, `no_contrastive`, and `no_image_context`.
`planner_v2` adds explicit edit-delta and hard-negative losses. Use the full
matrix when the pilot trend is clean:

```bash
SKETCHIMAGE_PAPER_MODE=full \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
bash scripts/submit_paper_matrix.sh
```

Summarize matrix results:

```bash
SKETCHIMAGE_PAPER_MODE=full \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/summarize_paper_matrix.sh
```

Audit an existing run to quantify nearest-neighbor shortcut difficulty. This is
a CPU job, so submit it to a CPU compute node:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_audit_benchmark.sh outputs/runs/sketchmol_aligned_paper_pilot_ridge_baseline_seed7
```

Build a hard scaffold/nearest-neighbor-controlled split. This also runs on a
CPU compute node:

```bash
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
SKETCHIMAGE_HARD_SPLIT_NAME=sketchmol_hard_seed7 \
bash scripts/submit_hard_split.sh
```

Run the paper matrix on that hard split:

```bash
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_PAPER_MATRIX_NAME=sketchmol_hard_paper_pilot \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_paper_matrix.sh
```

## Phase 1 Oracle Latent Decoder

This is the first step toward replacing the current heuristic/generative
decoder with a learned decoder. It removes the JEPA planner and gives the
decoder the target molecule latent directly. The first denoising-only pilot
returned zero valid SMILES, so this script now defaults to an autoregressive
SMILES objective while preserving `SKETCHIMAGE_ORACLE_OBJECTIVE=denoising` as a
diagnostic.

```text
oracle target latent -> latent-conditioned SMILES decoder -> sampled molecules
```

Run it from the login node with a 10GB MIG GPU:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase1_oracle_latent_diffusion_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_MOLECULE_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_ORACLE_EPOCHS=20 \
SKETCHIMAGE_ORACLE_BATCH_SIZE=128 \
bash scripts/submit_oracle_latent_diffusion.sh
```

Outputs:

```text
outputs/runs/phase1_oracle_latent_diffusion_seed7/metrics.json
outputs/runs/phase1_oracle_latent_diffusion_seed7/predictions.csv
outputs/runs/phase1_oracle_latent_diffusion_seed7/task_type_summary.csv
outputs/runs/phase1_oracle_latent_diffusion_seed7/run_config.json
```

Resource default:

```text
h100 10GB MIG = 1
cpus-per-task = 4
mem = 32G
time = 4h
```

For a quick sanity check before a full run:

```bash
SKETCHIMAGE_MOLECULE_CSV=data/example_molecules.csv \
SKETCHIMAGE_RUN_NAME=phase1_oracle_latent_diffusion_smoke \
SKETCHIMAGE_ORACLE_EPOCHS=2 \
SKETCHIMAGE_MOLECULE_LIMIT=100 \
bash scripts/submit_oracle_latent_diffusion.sh
```

## Phase 2A Planned Decoder

Use this after a Phase 1 oracle decoder exists. The point is to test the next
link in the chain:

```text
task condition/source/image context -> JEPA planner latent -> frozen Phase 1 decoder -> SMILES
```

Submit the 2048-latent version from the login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase2_planned_decoder_2048_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
bash scripts/submit_phase2_planned_decoder.sh
```

Outputs:

```text
outputs/runs/phase2_planned_decoder_2048_seed7/metrics.json
outputs/runs/phase2_planned_decoder_2048_seed7/predictions.csv
outputs/runs/phase2_planned_decoder_2048_seed7/task_type_summary.csv
outputs/runs/phase2_planned_decoder_2048_seed7/run_config.json
```

Key metrics to read first:

```text
planner_latent_mse
planner_latent_cosine
top1_target_tanimoto
mean_best_tanimoto
topk_target_hit
top1_decoder_train_pool_member
candidate_decoder_train_pool_member_fraction
```

If `planner_latent_cosine` is low, the planner is the bottleneck. If the
planner cosine is high but target metrics are weak, the frozen decoder or
latent representation is the bottleneck.

## Phase 2B Robust Decoder

Phase 2A can fail because the frozen decoder only learned oracle target latents.
Phase 2B starts from that Phase 1 decoder and fine-tunes it on oracle, noisy,
planner-predicted, and interpolated latents paired to the same target SMILES.
The goal is to make the decoder robust to realistic planner error.

Submit from the login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase2_robust_decoder_2048_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
bash scripts/submit_phase2_robust_decoder.sh
```

Default Phase 2B resource request:

```text
h100 10GB MIG = 1
cpus-per-task = 8
mem = 64G
time = 10h
```

Default robust fine-tune mix:

```text
oracle repeats = 1
planner-predicted repeats = 2
noisy oracle repeats = 1
oracle/planner interpolation repeats = 1
noise std = 0.05
decoder fine-tune epochs = 8
decoder fine-tune lr = 1e-4
```

Primary comparison:

```text
Phase 2A frozen decoder validity/top1/mean_best
vs.
Phase 2B robust decoder validity/top1/mean_best
```

If validity recovers but target Tanimoto remains weak, tune the planner or
condition representation. If validity remains low, the latent-conditioned
token decoder itself is still not robust enough.

## Phase 2C Latent Calibration

Phase 2B made the decoder valid again, but target similarity can still be weak
because planner latents remain off the oracle latent manifold. Phase 2C trains
a small residual ridge adapter:

```text
planner latent -> calibrated latent -> fixed robust decoder -> SMILES
```

Submit from the login node after Phase 2B has completed:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase2_calibrated_decoder_2048_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_DECODER_DIR=outputs/runs/phase2_robust_decoder_2048_seed7/decoder \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
bash scripts/submit_phase2_calibrated_decoder.sh
```

Primary metrics:

```text
planner_latent_cosine
calibrated_latent_cosine
top1_validity
top1_target_tanimoto
mean_best_tanimoto
topk_target_hit
top1_decoder_train_pool_member
```

Default adapter:

```text
mode = residual_ridge
ridge = 0.01
blend = 1.0
normalize = true
```

If calibration improves cosine but hurts validity, compare against the original
Phase 2B robust decoder. If calibration improves neither cosine nor Tanimoto,
the planner representation needs a stronger alignment loss rather than another
decoder-side fix.

## Latent Sensitivity Diagnostic

This is the next run after Phase 2C. It does not train a new planner or decoder.
It freezes the same decoder and decodes several latent sources:

```text
oracle_target
noisy_oracle_c0_32
noisy_oracle_c0_38
noisy_oracle_c0_63
noisy_oracle_c0_78
planner_predicted
calibrated_predicted
target/planner interpolations
```

Submit from the login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase2_latent_sensitivity_2048_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_DECODER_DIR=outputs/runs/phase2_robust_decoder_2048_seed7/decoder \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_PLANNER_RUN_DIR=outputs/runs/phase2_robust_decoder_2048_seed7 \
SKETCHIMAGE_CALIBRATED_RUN_DIR=outputs/runs/phase2_calibrated_decoder_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
bash scripts/submit_latent_sensitivity.sh
```

The first file to read is:

```text
outputs/runs/phase2_latent_sensitivity_2048_seed7/source_summary.csv
```

Interpretation:

```text
oracle_target strong, noisy_oracle_c0_38 weak:
  planner latent accuracy is not high enough for this decoder.

oracle_target strong, noisy_oracle_c0_38 strong, planner/calibrated weak:
  planner latents are directionally off-distribution, not just low-cosine.

oracle_target weak:
  robust decoder fine-tuning damaged oracle control or the decoder is not
  sensitive enough to molecular latents.
```

## Phase 2D Oracle-Anchored Robust Decoder

Phase 1 has strong oracle control, while Phase 2B recovers validity but weakens
that control. Phase 2D starts again from the Phase 1 decoder and fine-tunes with
a much stronger oracle anchor:

```text
many oracle target rows
small noisy-oracle basin
light planner/calibrated/interpolation exposure
low learning rate
```

Submit from the login node after Phase 2B, Phase 2C, and the hard split exist:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_RUN_NAME=phase2_oracle_anchored_decoder_2048_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHIMAGE_ORACLE_DECODER_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_DECODER_POOL_DIR=outputs/runs/phase1_oracle_latent_ar_2048_seed7 \
SKETCHIMAGE_PLANNER_RUN_DIR=outputs/runs/phase2_robust_decoder_2048_seed7 \
SKETCHIMAGE_CALIBRATED_RUN_DIR=outputs/runs/phase2_calibrated_decoder_2048_seed7 \
SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
bash scripts/submit_phase2_oracle_anchored_decoder.sh
```

Default decoder fine-tune recipe:

```text
epochs = 4
lr = 0.00002
oracle_repeats = 8
noisy_cosines = 0.78,0.90
planner_repeats = 1
calibrated_repeats = 1
interpolation_alphas = 0.10,0.25
```

Read these first:

```text
outputs/runs/phase2_oracle_anchored_decoder_2048_seed7/source_summary.csv
outputs/runs/phase2_oracle_anchored_decoder_2048_seed7/metrics.json
```

Success criteria:

```text
oracle_target mean_best_tanimoto stays much closer to Phase 1 than Phase 2B.
planner_predicted or calibrated_predicted improves over Phase 2B/2C.
validity does not collapse.
```

## Stage-C Generative Decoder Prototype

This is the next experiment after the hard split shows the retrieval ceiling.
It keeps the current torch planner, but sets
`SKETCHIMAGE_DECODER_MODE=hybrid_generative`, so the decoder can mutate
retrieved/source/template seeds into SMILES that are not restricted to the
training target pool.

Run the hard-split pilot from the login node:

```bash
cd "/scratch/bdong/projects/Diffusion-Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_stage_c_generative_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_generative_decoder.sh
```

Or submit it as a matrix variant against the strongest hard-split diagnostic:

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

The learned-transform decoder is the next Stage-C implementation. It learns
source-target edit fragments from training tasks, then applies those transforms
to eval sources:

```bash
cd "/scratch/bdong/projects/Diffusion-Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_learned_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_learned_transform_decoder.sh
```

For a compact comparison:

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

Run the scaffold-preserving transform decoder when the goal is specifically to
improve edit/inpaint/fragment-grow source-core retention:

```bash
cd "/scratch/bdong/projects/Diffusion-Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_scaffold_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_scaffold_transform_decoder.sh
```

Or compare scaffold-preserving against the previous learned-transform run:

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

Run the property-conditioned transform decoder when scaffold retention works
but property success collapses. It selects learned source edits by matching the
requested property delta:

```bash
cd "/scratch/bdong/projects/Diffusion-Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_TRAIN_CSV=outputs/tasks/sketchmol_hard_seed7_train.csv \
SKETCHIMAGE_EVAL_CSV=outputs/tasks/sketchmol_hard_seed7_eval.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_hard_property_transform_seed7 \
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/submit_property_transform_decoder.sh
```

Or compare it against scaffold-only and the no-contrastive diagnostic:

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

Run the latent-conditioned transform beam decoder when property-delta rerank
can find directionally correct candidates but target Tanimoto stays low:

```bash
cd "/scratch/bdong/projects/Diffusion-Molecule"
git pull --rebase origin main
cd SketchImageJEPA

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

Run the supervised edit-policy transform decoder when testing whether a learned
source-target edit policy improves the candidate generator beyond heuristic
transform ranking:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull --rebase origin main
cd SketchImageJEPA

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

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_torch_50k_10k_v1 \
SKETCHIMAGE_PYTHON_BIN=/path/to/python-with-torch \
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
SKETCHIMAGE_MOLECULE_LIMIT=50000 \
SKETCHIMAGE_MAX_TASKS=10000 \
SKETCHIMAGE_TORCH_EPOCHS=25 \
bash scripts/submit_torch_denoiser.sh
```

Default resource request:

```text
h100 10GB MIG = 1
cpus-per-task = 8
mem = 64G
time = 8h
```

The submit helper defaults to `SKETCHIMAGE_GPU_PROFILE=h100_10gb_mig`. It first
tries `nvidia_h100_80gb_hbm3_1g.10gb:1`, then `h100_1g.10gb:1`, matching the
GPU names used by the neighboring PhysTabMol and MolPilot scripts. Override
with `SKETCHIMAGE_GPU_PROFILE=h100_20gb_mig` or
`SKETCHIMAGE_SLURM_GPUS=<available_gpu_name>:1` if the scheduler requires a
different name.

If your cluster requires modules before the venv is active, provide them as a
space-separated list:

```bash
SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4 cuda" \
SKETCHIMAGE_PYTHON_BIN=/path/to/python-with-torch \
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
bash scripts/submit_torch_denoiser.sh
```

Check the job:

```bash
squeue -u "$USER"
tail -f sketchimage-jepa-<jobid>.log
```

One-command pull and run:

```bash
cd "/path/to/Diffusion Molecule"
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

Run with a molecule CSV. The script will first build task rows, then run the
SketchMol-aligned experiment:

```bash
cd "/path/to/Diffusion Molecule"
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

For a built-in tiny molecule example:

```bash
cd "/path/to/Diffusion Molecule"
SKETCHIMAGE_MOLECULE_CSV=data/example_molecules.csv \
SKETCHIMAGE_RUN_NAME=molecule_builder_smoke \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

Run with an already-built task CSV:

```bash
cd "/path/to/Diffusion Molecule"
SKETCHIMAGE_DATASET_CSV=/path/to/tasks.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

If the server Python is not the right one:

```bash
SKETCHIMAGE_PYTHON_BIN=/path/to/python3 \
SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

If you already pulled and only want to run:

```bash
SKETCHIMAGE_SKIP_PULL=1 bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

The run writes:

```text
SketchImageJEPA/outputs/tasks/<run_name>_tasks.csv
SketchImageJEPA/outputs/tasks/<run_name>_tasks.summary.json
SketchImageJEPA/outputs/runs/<run_name>/metrics.json
SketchImageJEPA/outputs/runs/<run_name>/predictions.csv
SketchImageJEPA/outputs/runs/<run_name>/run_config.json
SketchImageJEPA/outputs/runs/<run_name>/train_examples.csv
SketchImageJEPA/outputs/runs/<run_name>/eval_examples.csv
```

CSV format:

```text
task_id,task_type,source_smiles,target_smiles,instruction,mask_hint,image_path,goals
```

`task_type` is one of `de_novo`, `edit`, `inpaint`, or `fragment_grow`.
