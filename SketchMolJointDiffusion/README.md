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
`SketchMolTokenDiffusion` and turns on the image head/loss. The backbone first
compresses the oracle condition into a shared latent; the token decoder and
image decoder both consume that latent. An optional CLIP-style contrastive loss
aligns the token-side latent, condition latent, and image latent.

That makes Route A and Route B comparable as a clean ablation:

- A: token diffusion only.
- B: token diffusion + image head + render consistency metrics.
- B-CLIP: shared-latent joint diffusion + image/token contrastive alignment.

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

Submit the latent-aligned CLIP-style variant:

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SKETCHMOL_JOINT_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash SketchMolCompare/scripts/submit_joint_diffusion_latent_clip.sh
```

Smoke test:

```bash
bash SketchMolJointDiffusion/scripts/run_smoke.sh
```
