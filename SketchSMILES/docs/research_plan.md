# OCR-Free Joint Sketch-SMILES Generation

## Problem

SketchMol-style systems make molecular images central to generation, but the
generated image still needs to be converted back into a molecular graph or
SMILES string. That image-recognition step is slow, brittle, and can become the
real bottleneck even when the image itself looks plausible.

## Key Idea

Generate the image and the SMILES as paired outputs from a shared molecular
representation. The image is retained for interpretability, while the SMILES is
available immediately for cheminformatics evaluation.

## What Makes It Different From Direct SMILES Generation

The model is not only optimized to emit a valid SMILES. It must also emit a
visual molecular sketch that matches the SMILES. The image remains an explicit
output, and consistency becomes a first-class metric.

## Metrics

- SMILES validity: RDKit parse success.
- SMILES quality: novelty, uniqueness, property success, target similarity.
- Image renderability: valid molecular depiction surface with enough visible
  atoms/bonds.
- Pair consistency: generated sketch matches generated SMILES.
- Efficiency: inference latency compared with image -> OCR -> SMILES.

## First Minimal Experiment

Use RDKit-rendered images as the oracle visual target and train/evaluate a
paired-output interface on molecules from the existing PhysTabMol CSV. This
establishes the data contract and verifier before introducing a learned image
decoder.
