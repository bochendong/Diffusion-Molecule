# SketchMolJointDiffusion

Route B for the SketchMol comparison:

```text
condition / sketch / latent
  -> shared masked-token diffusion backbone
  -> SMILES token head
  -> image head
  -> consistency metrics between learned sketch and RDKit(SMILES) render
```

This route keeps the diffusion-style multi-step sampling, but avoids SketchMol's
OCR dependency. The model directly emits a machine-readable SMILES candidate and
also emits an image for human inspection and self-consistency checks.

The current implementation reuses the Route A denoising backbone from
`SketchMolTokenDiffusion` and turns on the image head/loss. That makes Route A
and Route B comparable as a clean ablation:

- A: token diffusion only.
- B: token diffusion + image head + render consistency metrics.

## Commands

Run inside an allocated GPU node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_JOINT_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_JOINT_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolJointDiffusion/scripts/run_joint_diffusion.sh
```

Submit from a login node:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_JOINT_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SKETCHMOL_JOINT_PAIR_DIR=SketchSMILES/outputs/pairs/phys_50k \
bash SketchMolJointDiffusion/scripts/submit_joint_diffusion.sh
```

Smoke test:

```bash
bash SketchMolJointDiffusion/scripts/run_smoke.sh
```
