# SketchMol Multi-Property Edit Dataset

This folder owns the large dataset build for the Understanding-Condition stream.
It is intentionally separate from model experiments so the molecule database,
edit-pair database, generated condition rows, and Slurm build workflow stay in
one place.

## Goal

Build a benchmark-aligned multi-property scaffold-preserving edit dataset:

```text
source molecule image + natural-language multi-property edit instruction
    -> target molecule / property deltas / SketchMol-compatible condition fields
```

The dataset is designed to scale beyond the current small
`mixed_objective_dataset_8k_strict_v2` proof-of-concept and align with
SketchMol-style 2-7 property control.

## Outputs

The default one-click build writes:

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/
  molecule_database.csv
  edit_pairs.csv
  condition_rows.csv
  baseline_variants.csv
  summary.json
  images/
```

`molecule_database.csv` stores canonical SMILES, scaffold, and 7 SketchMol
properties:

```text
MW, LogP, QED, TPSA, HBD, HBA, RB
```

`edit_pairs.csv` stores scaffold-preserving source-target pairs and full
property delta vectors.

`condition_rows.csv` samples 2-7 active properties from each pair and renders a
natural-language instruction.

`baseline_variants.csv` expands each condition into:

```text
full, text_only, image_only, random_query, caption_bottleneck
```

## One-Click Slurm Build

Run this on a Slurm login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

SMMED_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

The submitted job runs on a compute node. Logs are written to:

```text
SketchMol-MultiProperty-EditDataset/logs/
```

## Useful Overrides

```bash
SMMED_INPUT_CSV=PhysTabMol/runs/20260601_070814_sketchmol_compare_structure_seed7/tables/train_table.csv
SMMED_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_200k_v1
SMMED_LIMIT=200000
SMMED_MAX_PAIRS=100000
SMMED_MAX_PAIRS_PER_SCAFFOLD=300
SMMED_CONDITIONS_PER_PAIR=4
SMMED_MIN_ACTIVE_PROPERTIES=2
SMMED_MAX_CONDITION_PROPERTIES=7
```

For a faster dry run:

```bash
SMMED_LIMIT=10000 \
SMMED_MAX_PAIRS=5000 \
SMMED_MAX_PAIRS_PER_SCAFFOLD=50 \
SMMED_CONDITIONS_PER_PAIR=2 \
SMMED_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/dry_run_10k \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

## Build Steps

The workflow is:

1. Build molecule database from a PhysTabMol/SketchMol table.
2. Mine large scaffold-preserving edit pairs with 2+ active property changes.
3. Generate 2-7 property instruction rows and SketchMol-compatible condition
   columns.
4. Expand rows into baseline variants for Understanding-Condition ablations.

## Current Recommendation

Use this dataset before running more encoder ablations. The older few-hundred
pair dataset is enough for pipeline validation, but not enough for a main
multi-property report.
