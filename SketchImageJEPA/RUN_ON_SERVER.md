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

This writes:

```text
outputs/runs/<run_name>/rerank_diagnostics/rerank_sweep_summary.csv
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
