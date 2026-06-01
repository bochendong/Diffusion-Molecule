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
   Decode synchronized image + SMILES outputs from a shared molecular
   representation.

4. **Phase 5C: Image-conditioned SMILES decoder**
   Replace the oracle molecular fingerprint with the molecular sketch image and
   test whether direct image-to-SMILES decoding can bypass OCR.

5. **Phase 5D: Consistency-guided filtering**
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

## Phase 5A-0 Oracle Paired Baseline

Run the oracle baseline that emits canonical SMILES and an RDKit-rendered sketch
from the same molecule, then verifies the paired-output contract:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a0_oracle_baseline_seed7 \
SKETCHSMILES_SAMPLE_COUNT=64 \
bash scripts/run_phase5a0_oracle_baseline.sh
```

The run writes:

```text
outputs/runs/phase5a0_oracle_baseline_seed7/metrics.json
outputs/runs/phase5a0_oracle_baseline_seed7/oracle_predictions.csv
outputs/runs/phase5a0_oracle_baseline_seed7/train_pairs.csv
outputs/runs/phase5a0_oracle_baseline_seed7/eval_pairs.csv
outputs/runs/phase5a0_oracle_baseline_seed7/sample_contact_sheet.png
```

## Phase 5A-1 Learned SMILES Decoder

Train a learned oracle-conditioned SMILES decoder. The model consumes an RDKit
Morgan fingerprint, emits SMILES directly, renders the top generated SMILES back
to a sketch, and evaluates both molecular accuracy and paired-output
consistency:

Submit the full run to a GPU node:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a1_learned_smiles_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
bash scripts/submit_phase5a1_learned_smiles_decoder.sh
```

Inside an already allocated GPU node, run:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a1_learned_smiles_decoder_seed7 \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_DEVICE=auto \
bash scripts/run_phase5a1_learned_smiles_decoder.sh
```

For a short GPU sanity run, add `SKETCHSMILES_LIMIT=2000`,
`SKETCHSMILES_EPOCHS=2`, and set a separate `SKETCHSMILES_RUN_NAME`.

The run writes:

```text
outputs/runs/phase5a1_learned_smiles_decoder_seed7/metrics.json
outputs/runs/phase5a1_learned_smiles_decoder_seed7/predictions.csv
outputs/runs/phase5a1_learned_smiles_decoder_seed7/model.pt
outputs/runs/phase5a1_learned_smiles_decoder_seed7/vocab.json
outputs/runs/phase5a1_learned_smiles_decoder_seed7/train_history.json
outputs/runs/phase5a1_learned_smiles_decoder_seed7/sample_contact_sheet.png
```

The submit helper defaults to `SKETCHSMILES_GPU_PROFILE=h100_10gb_mig` and
tries `nvidia_h100_80gb_hbm3_1g.10gb:1`, then `h100_1g.10gb:1`. For a larger
slice, set `SKETCHSMILES_GPU_PROFILE=h100_20gb_mig` or
`SKETCHSMILES_GPU_PROFILE=h100_40gb_mig`.

## Phase 5A-2 Tokenized Beam Decoder

Run the stronger oracle-conditioned decoder with SMILES tokenization and beam
search:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a2_tokenized_beam_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_BEAM_SIZE=8 \
bash scripts/submit_phase5a2_tokenized_beam_decoder.sh
```

For a short sanity run, add `SKETCHSMILES_LIMIT=2000`,
`SKETCHSMILES_EPOCHS=2`, and set a separate `SKETCHSMILES_RUN_NAME`.

## Phase 5A-3 Condition-Attentive Transformer Decoder

Run the same oracle-conditioned paired-output task with a Transformer decoder.
The molecular fingerprint is projected into condition tokens, and each SMILES
token attends to those condition tokens during autoregressive beam decoding:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a3_transformer_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_BEAM_SIZE=8 \
SKETCHSMILES_TRANSFORMER_LAYERS=4 \
SKETCHSMILES_ATTENTION_HEADS=8 \
SKETCHSMILES_CONDITION_TOKENS=8 \
bash scripts/submit_phase5a3_transformer_decoder.sh
```

For a short sanity run, add `SKETCHSMILES_LIMIT=2000`,
`SKETCHSMILES_EPOCHS=2`, and set a separate `SKETCHSMILES_RUN_NAME`.

## Phase 5A-4 Reranked Transformer Decoder

Run the 5A-3 Transformer decoder with a wider beam and condition-fingerprint
reranking. The reranker scores each valid beam candidate by Morgan fingerprint
Tanimoto to the conditioning fingerprint, then renders the selected top-1
SMILES:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a4_reranked_transformer_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_BEAM_SIZE=16 \
SKETCHSMILES_RERANK_MODE=condition_fingerprint \
bash scripts/submit_phase5a4_reranked_transformer_decoder.sh
```

This is the fastest way to test whether 5A-3 already places the correct
molecule in the beam but under-ranks it.

If the training epochs finish but Slurm kills the job during the expensive
eval/rerank/render step, resume from the saved `model.pt` without retraining:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5a4_reranked_transformer_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_SLURM_TIME=06:00:00 \
bash scripts/submit_phase5a4_eval_only.sh
```

Summarize the Phase 5A chain into one comparison table:

```bash
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash scripts/run_phase5_summary.sh
```

## Phase 5B Joint SMILES/Sketch Decoder

Run a shared-latent two-head model. The model consumes the same oracle molecular
fingerprint, then jointly decodes a SMILES string and a learned sketch image.
The run also renders the generated SMILES with RDKit so the learned image can be
checked against both the target sketch and the generated molecule sketch:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5b_joint_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_IMAGE_SIZE=128 \
SKETCHSMILES_BEAM_SIZE=8 \
bash scripts/submit_phase5b_joint_decoder.sh
```

The sample contact sheet shows three columns per example:
target sketch, learned sketch, and RDKit-rendered generated SMILES. For a short
sanity run, add `SKETCHSMILES_LIMIT=2000`, `SKETCHSMILES_EPOCHS=2`, and set a
separate `SKETCHSMILES_RUN_NAME`.

## Phase 5C Image-Conditioned SMILES Decoder

Run a direct sketch-image-to-SMILES decoder. The model encodes the rendered
molecular sketch with a CNN, exposes image tokens to a Transformer SMILES
decoder, then renders the generated SMILES for pair-consistency evaluation:

```bash
SKETCHSMILES_MODULES="gcc rdkit/2025.09.4" \
SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHSMILES_PAIR_DIR=outputs/pairs/phys_50k \
SKETCHSMILES_RUN_NAME=phase5c_image_smiles_decoder_seed7 \
SKETCHSMILES_GPU_PROFILE=h100_10gb_mig \
SKETCHSMILES_EPOCHS=20 \
SKETCHSMILES_BATCH_SIZE=128 \
SKETCHSMILES_IMAGE_SIZE=128 \
SKETCHSMILES_BEAM_SIZE=8 \
bash scripts/submit_phase5c_image_smiles_decoder.sh
```

For a quick sanity run, add `SKETCHSMILES_LIMIT=2000`,
`SKETCHSMILES_EPOCHS=2`, and set a separate `SKETCHSMILES_RUN_NAME`.
