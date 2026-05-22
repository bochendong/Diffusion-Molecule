# SketchImage-JEPA Blueprint

## Goal

Build a SketchMol-aligned molecular generation and editing project that keeps
the useful visual-task framing but removes the generated-image-to-OCR recovery
step from the core path.

## Baseline Pipeline

```text
source molecule / fragment / mask / text condition
  -> context features
  -> JEPA target latent prediction
  -> retrieval or learned molecular decoder
  -> verifier-ranked candidates
```

## Why JEPA Here

The planner should learn the latent edit implied by the context instead of
reconstructing molecule pixels. This makes the model answer the molecular
question directly: which target representation should satisfy the prompt,
source constraints, and local edit mask?

## Experiment Ladder

1. Numpy ridge JEPA plus retrieval decoder on toy fixtures.
2. Real CSV loader with train/test split and RDKit metrics.
3. RDKit-rendered molecule image statistics as optional context.
4. Torch JEPA with target, delta, contrastive, and variance regularization
   losses.
5. Source-guided graph edits and SELFIES/graph decoder.
6. SketchMol-aligned benchmark comparison.

## Paper-Facing Metrics

- Validity
- Uniqueness
- Novelty
- Property hit rate
- Scaffold preservation
- Fragment growth success
- Local edit success
- Mean best Tanimoto to reference

## Current Run Contract

Each run directory writes:

- `metrics.json`: aggregate evaluation metrics.
- `predictions.csv`: ranked eval candidates and verifier scores.
- `run_config.json`: input paths, dimensions, split seed, image-context stats,
  and model history.
- `train_examples.csv`: exact training rows used by the run.
- `eval_examples.csv`: exact evaluation rows used by the run.
- `model/`: saved JEPA predictor config and weights.

## SketchMol-Aligned Preset

The comparison preset intentionally follows the SketchMol / PhysTabMol
SketchMol-aligned settings:

- Input molecule image reference size: `256`.
- Latent shape reference: `32 x 32 x 4`, recorded as `latent_dim=4096`.
- Conditioning/context dimension: `256`.
- Candidate count per condition: `8`.
- Single-property benchmark reference: `125 conditions x 8 samples = 1000
  molecules per target`.
- Multi-property benchmark reference: `1000` conditions.
- Optimization benchmark reference: `100` conditions.
- SketchMol sampling reference: DDIM `250` steps, `eta=1.0`, valid guidance
  `scale=2`, property guidance `scale_pro=4`.

The numpy baseline is not a diffusion sampler, so these diffusion knobs are
stored as comparison metadata rather than consumed by the ridge predictor.
