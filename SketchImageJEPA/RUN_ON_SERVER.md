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
torch sweep. It launches `latent_heavy`, `contrastive_strong`, and `batch256`
variants as separate 10GB MIG jobs:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA

SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v8_contrastive_focus \
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
SKETCHIMAGE_SWEEP_NAME=sketchmol_aligned_torch_50k_10k_v8_contrastive_focus \
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/summarize_torch_sweep.sh
```

If a run has high `mean_best_tanimoto` but low `top1_target_tanimoto`, sweep
reranking weights on the completed `predictions.csv` without using more GPU:

```bash
SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/rerank_run.sh outputs/runs/sketchmol_aligned_torch_50k_10k_v8_contrastive_focus_latent_heavy
```

This writes:

```text
outputs/runs/<run_name>/rerank_diagnostics/rerank_sweep_summary.csv
outputs/runs/<run_name>/rerank_diagnostics/best_task_type_summary.csv
outputs/runs/<run_name>/rerank_diagnostics/best_reranked_predictions.csv
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
