# SketchMolCompare

This folder is the comparison-facing workspace for large-scale experiments against the real SketchMol baseline under `Research/Molecule Generation/SketchMol`. It is intentionally separate from `SketchImageJEPA`, `SketchSMILES`, and the original SketchMol repo so the publication comparison layer does not get mixed with exploratory model development.

## Goal

We want to test the practical claim:

> If the molecule sketch image is already available, we do not need to generate another image and then OCR it. We can decode directly to SMILES or molecular structure from the image/latent condition, then verify that the decoded molecule renders consistently.

That makes the comparison with SketchMol concrete:

- SketchMol-style pipeline: condition -> generated sketch image -> OCR/structure extraction -> molecule.
- Our OCR-free pipeline: condition image/latent -> SMILES/molecule -> deterministic RDKit render for visual consistency checks.

## Tracks

1. `sketchsmiles_ocr_free`
   Summarizes SketchSMILES runs such as Phase 5A-4 and Phase 5C. Key metrics are exact SMILES match, top-k exact match, target Tanimoto, scaffold match, validity, and image-render consistency.

2. `real_sketchmol_plus_ocr`
   Summarizes materialized outputs from the original SketchMol diffusion image generator followed by its MolScribe/OCR recognition step. Key metrics are OCR SMILES presence, RDKit validity, property success, and MolScribe score.

3. `token_diffusion_ocr_free`
   Route A. Summarizes masked token diffusion runs that denoise molecular
   structure tokens directly into SMILES, then validate/render with RDKit.

4. `joint_diffusion_ocr_free`
   Route B. Summarizes shared diffusion runs with both a SMILES token head and
   learned sketch image head, using RDKit render consistency instead of OCR.

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

Submit the randomized-SMILES augmented oracle-fingerprint model:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_sketchsmiles_5a6_randomized.sh
```

Submit the broader candidate-generation matrix:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_sketchsmiles_candidate_matrix.sh
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

Submit the real SketchMol + OCR benchmark:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_CKPT=/absolute/path/to/sketchmol/model.ckpt \
SKETCHMOL_MOLSCRIBE_MODEL=/absolute/path/to/swin_base_char_aux_200k.pth \
SKETCHMOL_PRESET_STR="MW:400" \
bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh
```

Submit Route A, direct token diffusion:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_TOKEN_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_TOKEN_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_token_diffusion.sh
```

Submit Route B, joint image + SMILES diffusion:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_JOINT_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_JOINT_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolCompare/scripts/submit_joint_diffusion.sh
```

Replace both `/absolute/path/to/...` examples with real checkpoint files on the
cluster filesystem. The submit script validates those files before it calls
`sbatch`.

Materialize an already-finished real SketchMol + OCR CSV into the standalone benchmark folder:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_BENCHMARK_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_BENCHMARK_SOURCE_CSV=/path/to/sketchmol/image_path.csv \
bash SketchMolBenchmark/scripts/materialize_current.sh
```

## Outputs

`run_compare_existing.sh` writes:

- `SketchMolCompare/outputs/comparisons/current/comparison_rows.csv`
- `SketchMolCompare/outputs/comparisons/current/comparison_rows.json`
- `SketchMolCompare/outputs/comparisons/current/comparison_report.md`

Use those files as the first table-building surface for paper comparisons.

By default, `run_compare_existing.sh` reads `SketchMolBenchmark/outputs/current/benchmark_summary.csv` when it exists. It does not use the older PhysTabMol proxy path as a SketchMol baseline.
