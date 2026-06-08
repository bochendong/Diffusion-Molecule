# Dataset setup

This directory is the dataset registry for the online/cluster checkout. The local repo should keep only this README and setup notes here; large raw files, processed manifests, model outputs, and benchmark artifacts should stay outside git.

## Canonical data root

Use one canonical data root on the online machine:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
mkdir -p "$DM_DATA_ROOT"/{raw,processed,derived,external,cache}
```

Recommended layout:

```text
$DM_DATA_ROOT/
  raw/
    multiproperty/
      train_table.csv
    3m-diffusion/
      ChEBI-20_data/
      PubChem324k/
      kv_data/
    moledit-instruct/
      MolEdit-Instruct_3034459.txt
  processed/
    multiproperty_100k_v1/
      molecule_database.csv
      edit_pairs.csv
      condition_rows.csv
      baseline_variants.csv
      diffusion_edit_manifest.csv
    moledit-instruct/
      moledit_instruct.csv
      moledit_instruct.jsonl
      enhanced_v1/
        molecule_cache/
        enhanced_pairs/
        splits/
  derived/
    unified_generation_3m_edit_v2/
      unified_condition_train.jsonl
      unified_condition_eval.jsonl
    smiles_dualstream/
      large_train.jsonl
  external/
    README.md
  cache/
```

`raw/` is for source datasets that should not be rewritten. `processed/` is for reusable benchmark tables produced from raw data. `derived/` is for train/eval manifests produced by a pipeline and safe to regenerate. `cache/` is disposable.

## Current dataset inventory

| Dataset | Current local source | Online canonical path | Status |
| --- | --- | --- | --- |
| Multi-property edit table | `SketchMol-MultiProperty-EditDataset/data/train_table.csv` | `$DM_DATA_ROOT/raw/multiproperty/train_table.csv` | Present locally, about 95k rows / 37 MB |
| 3M-Diffusion text datasets | `Research/Molecule Generation/3M-Diffusion/data/` | `$DM_DATA_ROOT/raw/3m-diffusion/` | Present locally through the ignored external checkout; ChEBI-20, PubChem324k, kv_data |
| MolEdit-Instruct raw | `Research/Molecule Generation/MolEditRL/data/raw/` (symlink) | `$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt` | Downloaded on cluster (2026-06-08); 3,034,459 rows / 733 MB |
| MolEdit-Instruct manifests | n/a (outside git) | `$DM_DATA_ROOT/processed/moledit-instruct/moledit_instruct.{csv,jsonl}` | Converted from raw txt; 736 MB + 953 MB |
| MolEdit-Instruct enhanced | `Research/Molecule Generation/MolEditRL/data/processed/enhanced_v1/` (symlink) | `$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/` | Built on cluster (2026-06-08); train 2,984,459 / eval_balanced 50,000 / smoke 1,000 |
| Multi-property edit table | `SketchMol-MultiProperty-EditDataset/data/train_table.csv` (symlink) | `$DM_DATA_ROOT/raw/multiproperty/train_table.csv` | Symlink present; canonical file still missing on cluster |
| Unified condition train/eval JSONL | pipeline `outputs/.../dataset/` | `$DM_DATA_ROOT/derived/unified_generation_3m_edit_v2/` | Derived; regenerate or copy from a completed run |
| SMILES-DualStream large manifest | `SMILES-DualStream-EditorAtomas/outputs/manifests/` | `$DM_DATA_ROOT/derived/smiles_dualstream/` | Derived; regenerate from multi-property table |

Not datasets:

- `Research/Molecule Generation/Atomas/dataset/` is code, not data.
- `Research/Molecule Generation/smi-editor/smi_dict_token.txt` is a vocabulary/resource file, not the Uni-Mol pretraining dataset.
- `predictions.csv`, `benchmark_decoded.csv`, `benchmark_summary.csv`, `image_path.csv`, and `metrics.json` under `outputs/` are run artifacts or evaluation outputs, not canonical source data.

## Compatibility links for existing scripts

Most scripts currently default to repo-relative paths. On the online machine, keep those paths working with symlinks or explicit environment variables.

From repo root:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule

mkdir -p "SketchMol-MultiProperty-EditDataset/data"
ln -sfn "$DM_DATA_ROOT/raw/multiproperty/train_table.csv" \
  "SketchMol-MultiProperty-EditDataset/data/train_table.csv"

mkdir -p "Research/Molecule Generation/3M-Diffusion"
ln -sfn "$DM_DATA_ROOT/raw/3m-diffusion" \
  "Research/Molecule Generation/3M-Diffusion/data"

mkdir -p "Research/Molecule Generation/MolEditRL/data/raw"
ln -sfn "$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt" \
  "Research/Molecule Generation/MolEditRL/data/raw/MolEdit-Instruct_3034459.txt"

mkdir -p "Research/Molecule Generation/MolEditRL/data/processed"
ln -sfn "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1" \
  "Research/Molecule Generation/MolEditRL/data/processed/enhanced_v1"
```

If symlinks are inconvenient, use the existing environment overrides where available:

```bash
SMMED_INPUT_CSV="$DM_DATA_ROOT/raw/multiproperty/train_table.csv" \
SMMED_OUTPUT_DIR="$DM_DATA_ROOT/processed/multiproperty_100k_v1" \
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/run_build_dataset.sh
```

For Unified 3M, the preflight currently checks `Research/Molecule Generation/3M-Diffusion/data`, so the 3M-Diffusion data symlink is the least invasive option.

## Fetch MolEdit-Instruct

The MolEditRL folder contains the public-asset fetch and manifest conversion helpers.

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
mkdir -p "$DM_DATA_ROOT/raw/moledit-instruct" "$DM_DATA_ROOT/processed/moledit-instruct"

curl -L --fail --continue-at - \
  -o "$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt" \
  "https://huggingface.co/datasets/FanSiLeC/MolEdit-Instruct/resolve/main/MolEdit-Instruct_3034459.txt"

python3 "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_instruct_manifest.py" \
  --input "$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt" \
  --output-csv "$DM_DATA_ROOT/processed/moledit-instruct/moledit_instruct.csv" \
  --output-jsonl "$DM_DATA_ROOT/processed/moledit-instruct/moledit_instruct.jsonl"
```

Known caveat: the public Hugging Face file has only `example_id`, `instruction`, `source_smiles`, and `target_smiles`. The MolEditRL paper uses structured task labels for reward-oracle selection, but those labels are not exposed in the public txt file.

## Enhance MolEdit-Instruct

The full MolEdit-Instruct dataset should be enhanced once on the online machine, then reused by training and benchmark jobs. This avoids repeated RDKit canonicalization, property scoring, scaffold extraction, and fingerprint/Tanimoto work.

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
export MOLEDIT_RAW_INPUT="$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt"
export MOLEDIT_OUTPUT_DIR="$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1"
export MOLEDIT_SHARDS=64
export MOLEDIT_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python

bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh" --dry-run
bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh"
```

Local serial smoke (2 shards, 1000 rows):

```bash
MOLEDIT_LOCAL=1 MOLEDIT_SHARDS=2 MOLEDIT_LIMIT=1000 \
MOLEDIT_OUTPUT_DIR="$DM_DATA_ROOT/processed/moledit-instruct/enhanced_smoke" \
MOLEDIT_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh"
```

The wrapper submits dependent Slurm arrays when `sbatch` exists:

```text
normalize-pairs -> molecule-cache -> pair-features -> finalize
```

Final reusable manifests:

```text
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_hard.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/smoke_1000.jsonl
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/summary.json
```

See `Research/Molecule Generation/MolEditRL/README.md` for split schema, benchmark usage, and caveats.

### Online cluster build status (2026-06-08)

MolEdit-Instruct enhancement has been built once on `/scratch/bdong/datasets/Diffusion-Molecule`:

```text
input_rows:        3,034,459
train_rows:        2,984,459   (splits/train.csv, ~2.4 GB)
eval_balanced_rows:   50,000   (splits/eval_balanced.csv, ~41 MB)
smoke_rows:            1,000   (splits/smoke_1000.jsonl)
eval_hard_rows:            0   (header only; current finalize logic did not select a hard split)
invalid_or_missing:        2
summary:           splits/summary.json
```

Prefer `splits/eval_balanced.csv` or `splits/smoke_1000.jsonl` for benchmark runs. Use `splits/train.csv` only when you explicitly need the full precomputed feature table; it is large.

Slurm job chain used for the full build: normalize `15803639` -> cache `15803641` -> pair-features `15803642` -> finalize `15803643`.

## Build order on the online machine

1. Put or link raw datasets under `$DM_DATA_ROOT/raw`.
2. Build the enhanced MolEdit-Instruct benchmark cache/splits if the raw file is available:

```bash
MOLEDIT_OUTPUT_DIR="$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1" \
bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh"
```

3. Build reusable multi-property processed tables:

```bash
SMMED_INPUT_CSV="$DM_DATA_ROOT/raw/multiproperty/train_table.csv" \
SMMED_OUTPUT_DIR="$DM_DATA_ROOT/processed/multiproperty_100k_v1" \
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/run_build_dataset.sh
```

4. For scripts that expect the old output location, either pass explicit env vars or link the processed folder:

```bash
mkdir -p SketchMol-MultiProperty-EditDataset/outputs
ln -sfn "$DM_DATA_ROOT/processed/multiproperty_100k_v1" \
  "SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1"
```

5. Export Unified 3M datasets and train from the pipeline scripts. If you want the dataset JSONL to live under `$DM_DATA_ROOT/derived`, copy or link the run's `dataset/` directory after export.

6. Prepare the pure-SMILES manifest only after the multi-property table is available:

```bash
SDEA_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SMILES-DualStream-EditorAtomas/scripts/prepare_large_manifest.sh --overwrite-manifest
```

Then copy or link `SMILES-DualStream-EditorAtomas/outputs/manifests/large_train.jsonl` into `$DM_DATA_ROOT/derived/smiles_dualstream/`.

## Quick checks

```bash
wc -l "$DM_DATA_ROOT/raw/multiproperty/train_table.csv"
find "$DM_DATA_ROOT/raw/3m-diffusion" -maxdepth 2 -type f -name '*.txt' -print
test -s "$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt"
wc -l "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv"
wc -l "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv"
python3 -m json.tool "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/summary.json" | head
```

Expected rough sizes:

```text
multi-property train_table.csv: ~95,001 lines including header
3M-Diffusion raw text folder: ~28 MB in the current local snapshot
MolEdit-Instruct raw txt: ~733 MB / 3,034,459 lines
MolEdit-Instruct enhanced train.csv: ~2,984,460 lines including header
MolEdit-Instruct enhanced eval_balanced.csv: ~50,001 lines including header
```

## Git policy

Do not commit files under `datasets/` except this README and `.gitignore`. Large data should be tracked by the online storage/release mechanism, not this source repo. Generated run outputs should stay in pipeline `outputs/` or `$DM_DATA_ROOT/derived`, depending on whether they are meant to be reused.
