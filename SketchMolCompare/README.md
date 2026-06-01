# SketchMolCompare

This folder is the comparison-facing workspace for large-scale experiments against SketchMol-style baselines. It is intentionally separate from `SketchImageJEPA`, `SketchSMILES`, and `PhysTabMol` so the publication comparison layer does not get mixed with exploratory model development.

## Goal

We want to test the practical claim:

> If the molecule sketch image is already available, we do not need to generate another image and then OCR it. We can decode directly to SMILES or molecular structure from the image/latent condition, then verify that the decoded molecule renders consistently.

That makes the comparison with SketchMol concrete:

- SketchMol-style pipeline: condition -> generated sketch image -> OCR/structure extraction -> molecule.
- Our OCR-free pipeline: condition image/latent -> SMILES/molecule -> deterministic RDKit render for visual consistency checks.

## Tracks

1. `sketchsmiles_ocr_free`
   Summarizes SketchSMILES runs such as Phase 5A-4 and Phase 5C. Key metrics are exact SMILES match, top-k exact match, target Tanimoto, scaffold match, validity, and image-render consistency.

2. `sketchmol_aligned`
   Summarizes PhysTabMol SketchMol-aligned benchmark CSVs. Key metrics are SketchMol success rates, validity, uniqueness, novelty, drug-likeness, and diversity.

## Quick Commands

Collect already-finished runs into one report:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_COMPARE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash SketchMolCompare/scripts/run_compare_existing.sh
```

Submit the large OCR-free best current model:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_sketchsmiles_5a4_transformer.sh
```

Submit the image-conditioned OCR-free model:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_sketchsmiles_5c_image_decoder.sh
```

Submit the image-conditioned model with fingerprint auxiliary reranking:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_sketchsmiles_5d_image_fingerprint.sh
```

Submit the PhysTabMol SketchMol-aligned structure benchmark:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

bash SketchMolCompare/scripts/submit_sketchmol_structure_benchmark.sh
```

## Outputs

`run_compare_existing.sh` writes:

- `SketchMolCompare/outputs/comparisons/current/comparison_rows.csv`
- `SketchMolCompare/outputs/comparisons/current/comparison_rows.json`
- `SketchMolCompare/outputs/comparisons/current/comparison_report.md`

Use those files as the first table-building surface for paper comparisons.
