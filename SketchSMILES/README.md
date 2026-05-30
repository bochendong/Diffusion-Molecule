# SketchSMILES

OCR-free molecular sketch generation through synchronized image and SMILES
decoding.

## Why This Is Separate

`SketchImageJEPA/` studies planner-to-decoder latent routing for
SketchMol-aligned molecular generation and editing. `SketchSMILES/` is a
different research direction: it targets the image-to-molecule recognition
bottleneck in SketchMol-style systems.

The core question is not whether a generated molecular image can later be
recognized by OCR. The question is whether a model can jointly emit:

- a human-readable molecular sketch image
- a machine-readable SMILES string
- a consistency score showing that both outputs describe the same molecule

```text
condition / source / instruction
        |
 shared molecular representation
       / \
  sketch   SMILES
       \ /
 consistency verifier
```

## Research Question

Can molecular sketch generation avoid the expensive image-to-SMILES OCR step by
jointly producing a visual sketch and a canonical molecular string, then
verifying cross-modal consistency?

## Hypothesis

A synchronized image-SMILES model can preserve the visual interpretability of
SketchMol-style image generation while making inference faster and more
reliable, because SMILES is emitted directly rather than recovered through a
separate recognizer.

## Proposed Phases

1. **Phase 0: Dataset and verifier contract**
   Build paired `(SMILES, rendered image)` manifests and define validity,
   renderability, and pair-consistency metrics.

2. **Phase 5A: Oracle paired decoder**
   Given an oracle molecule latent, generate both SMILES and a molecular sketch.
   This proves the paired-output interface before adding instruction planning.

3. **Phase 5B: Conditional paired generator**
   Map `condition/source/instruction` into the shared representation and decode
   synchronized image + SMILES outputs.

4. **Phase 5C: Consistency-guided filtering**
   Add a verifier that rejects outputs where the generated image and SMILES do
   not agree.

## Baselines

- SketchMol-style image pipeline: condition -> image -> OCR/recognition -> SMILES
- Direct conditional SMILES generator: condition/source/instruction -> SMILES
- RDKit oracle pair: SMILES -> RDKit-rendered image, used as a consistency
  control rather than as the final model

## Quick Smoke

```bash
cd SketchSMILES
python3 -m unittest discover -s tests
```

On the server, load the same RDKit module used by the SketchImageJEPA jobs:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/run_smoke.sh
```

## Phase 0 Pairs

Build a paired SMILES/rendered-image manifest:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_INPUT_CSV=/scratch/bdong/projects/Diffusion-Molecule/PhysTabMol/data/molecules.csv \
SKETCHSMILES_OUTPUT_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_LIMIT=50000 \
bash scripts/run_phase0_pairs.sh
```

Audit the paired manifest and create a visual sample sheet:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_SAMPLE_COUNT=64 \
bash scripts/run_phase0_audit.sh
```

The audit writes:

```text
outputs/pairs/phys_50k/audit_summary.json
outputs/pairs/phys_50k/audit_rows.csv
outputs/pairs/phys_50k/sample_pairs.csv
outputs/pairs/phys_50k/sample_contact_sheet.png
```
