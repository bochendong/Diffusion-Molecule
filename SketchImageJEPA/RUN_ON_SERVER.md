# Run SketchImage-JEPA On Server

From the server checkout:

```bash
cd "/path/to/Diffusion Molecule"
git pull --rebase origin main
cd SketchImageJEPA
bash scripts/run_sketchmol_aligned.sh
```

One-command pull and run:

```bash
cd "/path/to/Diffusion Molecule"
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

Run with your dataset:

```bash
cd "/path/to/Diffusion Molecule"
SKETCHIMAGE_DATASET_CSV=/path/to/tasks.csv \
SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

If the server Python is not the right one:

```bash
SKETCHIMAGE_PYTHON_BIN=/path/to/python3 \
SKETCHIMAGE_DATASET_CSV=/path/to/tasks.csv \
bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

If you already pulled and only want to run:

```bash
SKETCHIMAGE_SKIP_PULL=1 bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
```

The run writes:

```text
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
