# SketchMolTokenDiffusion

Route A for the SketchMol comparison:

```text
condition / image / sketch / property -> masked token diffusion -> SMILES -> RDKit validation/render
```

This folder is separate from `SketchSMILES` because the research claim is
different. The decoder is not an autoregressive SMILES model; it starts from
masked structure tokens and iteratively denoises them into canonical SMILES
candidates.

## Current Experiment

The first runnable experiment uses the paired manifest from `SketchSMILES` and
an oracle Morgan fingerprint condition. That gives a clean upper-bound test for
the core question:

```text
Can diffusion generate molecular structure tokens directly while keeping the
SketchMol-style multi-candidate sampling interface?
```

Outputs are written under:

```text
SketchMolTokenDiffusion/outputs/runs/<run_name>
```

`SketchMolCompare` can summarize these runs through their `metrics.json`.

## Commands

Run locally or inside an allocated node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_TOKEN_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_TOKEN_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolTokenDiffusion/scripts/run_masked_token_diffusion.sh
```

Submit on a login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_TOKEN_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_TOKEN_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolTokenDiffusion/scripts/submit_masked_token_diffusion.sh
```

Smoke test:

```bash
bash SketchMolTokenDiffusion/scripts/run_smoke.sh
```
