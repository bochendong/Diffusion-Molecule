# MolPilot Design Notes

## What We Borrow Conceptually

SketchMol contributes the experiment shape:

- property-conditioned generation with explicit numeric prompts
- target/profile-conditioned generation
- inpainting by keeping part of an input image/latent and repainting the rest
- post-generation filtering with molecular recognition and RDKit checks
- guidance scales for validity and property control

MolPilot should avoid SketchMol's weak point: generating molecule pixels and
then depending on OCR. The generator should output a molecular representation
directly.

UniVideo contributes the architecture shape:

- an understanding stream accepts text plus optional visual context
- the understanding stream creates conditioning embeddings for diffusion
- visual conditions can also be encoded into the latent stream
- editing/in-context generation is handled by task-specific instructions and
  condition branches instead of hard-coded edit rules

MolPilot maps this to molecules:

```text
understanding stream:
  text + source SMILES + rendered/source image + mask
  -> condition tokens / latent + executable objective spec

generation stream:
  source molecular latent + task token + mask + condition latent
  -> edited/inpainted/de novo molecular latent

verification stream:
  deterministic hard checks + optional learned predictor checks
```

## Task Modes

### Editing

Input:

```text
source_smiles + optional source image + natural-language issue/goal
```

Examples:

- "This lead has poor solubility; improve it without changing the core."
- "LogP is too high; lower it while keeping MW similar."
- "TPSA is too high for CNS exposure; make a smaller local edit."

### Inpainting

Input:

```text
source_smiles or scaffold + atom/substructure/image mask + instruction
```

Examples:

- "Keep the scaffold and fill the masked R-group to improve solubility."
- "Preserve this pharmacophore and complete the linker."

### De Novo

Input:

```text
natural-language profile prompt + optional target/profile context
```

Examples:

- "Generate CNS-like drug-like molecules with MW < 450 and TPSA < 90."
- "Generate soluble lead-like molecules with high QED."

Disease-level prompts are allowed as qualitative prompts, but not as verified
main-table objectives unless grounded to a target or predictor.

## Verification Boundary

Hard verified:

- validity
- MW, LogP, QED, TPSA, HBD, HBA, RB
- Lipinski/Veber-like drug-likeness
- scaffold preservation
- similarity and novelty
- inpainting mask preservation/completion once atom masks are implemented

Proxy verified:

- solubility with ESOL/AqSolDB predictor
- hERG with classifier
- BBB/permeability with classifier
- activity with target-specific QSAR or docking

Not verified:

- "treat cancer"
- "cure disease"
- clinical efficacy
- target activity without a target-specific model

## First Paper-Facing Ablations

1. text/spec conditioning only
2. source SMILES conditioning only
3. source rendered image conditioning only
4. full understanding stream
5. no source latent
6. no mask conditioning for inpainting
7. direct latent MLP vs latent diffusion
8. retrieval decoder vs learned molecular decoder
9. RDKit-only hard verification vs predictor-augmented verification

## Immediate Engineering Gaps

- Strengthen the new sequence decoder with a larger SELFIES vocabulary and
  validity-aware decoding.
- Add a graph decoder baseline.
- Add atom/substructure mask construction for inpainting.
- Add RDKit-rendered image branch to the dataset builder.
- Add scaffold split and prompt split.
- Add predictor hooks for solubility/hERG/BBB/activity.
- Add server-scale scripts with checkpointing and resumable evaluation.

## Training Stages Borrowed From Prior Code

### Why Borrow SketchMol's Staging

SketchMol does not train one monolithic model from prompt to molecule. Its
workflow is roughly:

```text
image generation data -> image autoencoder -> conditional latent diffusion
-> sampling/filtering feedback
```

MolPilot borrows this because molecule generation also needs a stable latent
space. The difference is that MolPilot's latent should represent molecules
directly rather than molecular images, avoiding OCR as a bottleneck.

### Why Borrow UniVideo's Understanding Stream

UniVideo uses an MLLM to consume text plus visual conditions, then passes hidden
states or metaquery tokens to a diffusion transformer. MolPilot borrows this
separation because medicinal chemistry prompts are messy:

```text
"This lead has poor solubility; keep the core"
source SMILES
rendered structure
optional masked region
optional target/protein context
```

These inputs should be grounded into both a continuous condition latent for
diffusion and an executable verifier spec for evaluation.

### MolPilot Staged Training

```text
Stage 1: molecular autoencoder
  SMILES/SELFIES/features -> z_mol -> sequence decoder

Stage 2: understanding alignment
  instruction + source SMILES + optional image/mask -> z_condition
  train with MSE + contrastive alignment to target z_mol

Stage 3: conditional molecular diffusion
  z_condition + source/task/mask context -> denoise target z_mol

Stage 4: verifier-guided sampling
  sample candidates, check hard RDKit/proxy constraints, save failures
```

Stage 4 is the place to later add SketchMol-style feedback or DPO-style
preference training from verifier successes/failures.

## Current Decoder Status

MolPilot now has a first sequence molecular decoder:

```text
SMILES -> SELFIES if available -> sequence encoder -> z_mol
z_mol -> sequence decoder -> SELFIES/SMILES -> candidate molecule
```

If `selfies` is unavailable, it falls back to a SMILES tokenizer. If PyTorch is
unavailable, the sequence codec keeps the same interface but uses nearest
latent retrieval so smoke tests still run. Server experiments should install
`selfies` and use PyTorch, which is the default intended path.
