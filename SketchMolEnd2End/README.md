# SketchMolEnd2End

End-to-end molecular sketch recognition/generation experiments that bypass the
SketchMol image -> MolScribe/OCR -> SMILES pipeline.

The first runnable baseline is deliberately simple:

```text
paired RDKit sketch image -> image encoder -> structure decoder -> SMILES
                                              -> RDKit render/check metrics
```

This keeps the image as the model input, but makes the molecular structure the
direct supervised output. OCR is only a comparison baseline, not part of this
route.

## Current Baseline

`image_to_structure` wraps the existing `SketchSMILES` Phase 5C
image-conditioned decoder and writes outputs under:

```text
SketchMolEnd2End/outputs/runs/<run_name>
```

The important metrics are:

- `top1_exact_match_fraction`
- `topk_exact_match_fraction`
- `top1_valid_fraction`
- `mean_best_tanimoto`
- `top1_scaffold_match_fraction`
- `image_mse_mean` between input sketch and RDKit render of the predicted
  molecule

## Commands

Smoke run:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
bash SketchMolEnd2End/scripts/run_smoke.sh
```

Full run on an allocated node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_E2E_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_E2E_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolEnd2End/scripts/run_image_to_structure.sh
```

Submit to Slurm:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_E2E_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_E2E_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolEnd2End/scripts/submit_image_to_structure.sh
```

Offline rerank diagnostic for a completed run:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_E2E_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_E2E_RUN_DIR=SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7 \
bash SketchMolEnd2End/scripts/run_rerank_diagnostic.sh
```

This reuses the saved beam candidates in `predictions.csv` and compares:

- `beam`: original sequence-score order.
- `predicted_fingerprint`: rerank by the predicted fingerprint score saved by the
  fingerprint-auxiliary run.
- `render_mse`: render each candidate and choose the image closest to the input
  sketch.
- `oracle_tanimoto`: target-aware upper bound for how much top-1 can improve if
  candidate selection were solved.

Train an image-molecule contrastive reranker and apply it to saved candidates:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_E2E_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_E2E_RUN_DIR=SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7 \
bash SketchMolEnd2End/scripts/run_contrastive_reranker.sh
```

For a quick smoke run:

```bash
SKETCHMOL_E2E_TRAIN_LIMIT=512 \
SKETCHMOL_E2E_EVAL_LIMIT=64 \
SKETCHMOL_E2E_EPOCHS=1 \
SKETCHMOL_E2E_DEVICE=cpu \
bash SketchMolEnd2End/scripts/run_contrastive_reranker.sh
```

This trains a lightweight CLIP-style reranker: the image tower embeds the input
sketch, the molecule tower embeds Morgan fingerprints, and in-batch negatives
teach paired image/structure matching. It is meant as the first learned
candidate-selection baseline against `beam`, `predicted_fingerprint`,
`render_mse`, and `oracle_tanimoto`.

Submit the contrastive reranker to Slurm:

```bash
SKETCHMOL_E2E_RUN_DIR=SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7 \
bash SketchMolEnd2End/scripts/submit_contrastive_reranker.sh
```

Re-evaluate a saved image-to-structure model with a larger candidate beam:

```bash
SKETCHMOL_E2E_RUN_DIR=SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7 \
SKETCHMOL_E2E_BEAM_SIZE=32 \
bash SketchMolEnd2End/scripts/run_saved_image_to_structure_eval.sh
```

Submit the same candidate expansion to Slurm:

```bash
SKETCHMOL_E2E_RUN_DIR=SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7 \
SKETCHMOL_E2E_BEAM_SIZE=32 \
bash SketchMolEnd2End/scripts/submit_saved_image_to_structure_eval.sh
```

After it finishes, run the offline rerank diagnostic on the expanded
`predictions.csv`, especially `render_mse` and `oracle_tanimoto`, to measure
whether the larger candidate pool raises the exact-match upper bound.

## Next Step

Before training another decoder, run the offline rerank diagnostic. If
`oracle_tanimoto` is much better than `beam`, the next paper-worthy target is
image-aware candidate selection. If the oracle upper bound is also weak, the
next target should be the decoder itself, such as image-to-SELFIES or an
image-conditioned graph/token diffusion decoder.
