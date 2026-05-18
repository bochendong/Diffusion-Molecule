# MolPilot

Independent prototype for unified multimodal molecular generation.

This folder is intentionally separate from `PhysTabMol`. It is a new research
line inspired by two local codebases:

- SketchMol: property-conditioned molecular image diffusion, repaint-style
  inpainting, and property/protein prompt experiments.
- UniVideo: an understanding stream that turns text/image/video context into
  conditioning tokens for a diffusion generator, with separate conditioning
  branches for unconditional, text-only, and multimodal guidance.

MolPilot adapts those ideas to molecules without importing either project.

## Target Tasks

MolPilot uses one unified request schema for three modes:

1. `edit`: source molecule plus instruction -> edited molecule.
2. `inpaint`: source/scaffold plus atom or region mask plus instruction -> completed molecule.
3. `de_novo`: natural-language profile prompt -> molecule from scratch.

The minimum input is a SMILES string for editing and inpainting. If no molecule
image is provided, the understanding stream can render one from the SMILES when
RDKit is available.

## Architecture

```text
text instruction + source SMILES + optional molecule image + optional mask
        |
        v
Understanding stream
  - text/objective grounding
  - molecule descriptors
  - image statistics or rendered molecule image
  - condition branches: uncond, text/spec, multimodal
        |
        v
Conditional molecular latent diffusion
  - task token: edit / inpaint / de_novo
  - source latent and mask conditioning when present
  - outputs molecular latent candidates
        |
        v
Molecule decoder
  - first scaffold uses nearest latent retrieval as a baseline
  - intended next step: learned graph/SELFIES decoder
        |
        v
Verifier
  - hard RDKit checks for valid, MW, LogP, QED, TPSA, HBA/HBD, RB, scaffold,
    similarity, novelty
  - learned predictors can be added for solubility, hERG, BBB, activity
  - disease-level claims are not counted as verified unless grounded to a
    target/predictor
```

## Why This Is Not PhysTabMol

PhysTabMol currently plans descriptor rows and uses a source-aware fragment
decoder. MolPilot is designed for direct molecular latent generation from
multimodal prompts. The early code keeps a retrieval decoder only as a sanity
baseline while the learned molecular decoder is built.

## First Commands

Run the lightweight end-to-end smoke test:

```bash
cd MolPilot
bash scripts/run_smoke.sh
```

Run the staged training smoke test:

```bash
cd MolPilot
bash scripts/run_staged_smoke.sh
```

Run on a server with a real molecule CSV:

```bash
cd MolPilot
MOLPILOT_DATA=/path/to/molecules.csv \
MOLPILOT_RUN_NAME=molpilot_edit_inpaint_denovo_v1 \
bash scripts/run_staged_server.sh
```

Submit the staged run to Slurm:

```bash
cd MolPilot
MOLPILOT_DATA=data/molecules.csv \
MOLPILOT_RUN_NAME=molpilot_staged_v1 \
sbatch scripts/run_staged_server.slurm.sh
```

Expected CSV columns for real data:

```text
smiles
```

Optional columns such as `name`, `MW`, `LogP`, `QED`, `TPSA`, `HBD`, `HBA`,
`RB`, and `SA` are used when present; otherwise RDKit computes them.

## Verification Boundary

MolPilot must not claim that a molecule treats a disease unless the prompt has
been grounded to a verifiable target or assay proxy. A prompt like
"make this a lung cancer drug" is treated as hypothesis-level. It can produce a
case study, but it does not enter the main verified benchmark.

## Staged Training

MolPilot follows a staged design because both reference systems do:

- SketchMol first trains an autoencoder/latent image space, then trains
  conditional diffusion, then performs bad-sample feedback and filtering. We
  borrow this because diffusion training is more stable when the generator
  works in a compact latent space rather than raw molecules from day one.
- UniVideo separates the multimodal understanding stream from the diffusion
  generator. We borrow this because user input is multimodal and messy: text,
  SMILES, rendered molecule images, optional masks, and possibly reference
  molecules need to become a clean conditioning representation before
  generation.

The current staged scripts are:

```text
Stage 1: train molecular autoencoder
  scripts/run_staged_*.sh -> molpilot.train_autoencoder

Stage 2: train understanding alignment
  text/SMILES/image/spec condition -> target molecular latent

Stage 3: train conditional molecular latent diffusion
  source/prompt condition latent -> target molecular latent

Stage 4: sample and verify
  generated candidates -> RDKit/proxy verifier
```

Stage 1 now supports two codec modes:

```text
sequence: SELFIES autoencoder when `selfies` is installed; otherwise SMILES tokens
feature: deterministic descriptor/hash feature autoencoder baseline
```

The staged scripts default to `sequence`. Nearest-latent retrieval remains only
as a fallback and secondary candidate source. The next major modeling step is a
stronger graph decoder and task-specific inpainting mask loss.

Useful ablation:

```bash
MOLPILOT_CODEC=feature bash scripts/run_staged_server.sh
MOLPILOT_CODEC=sequence bash scripts/run_staged_server.sh
```
