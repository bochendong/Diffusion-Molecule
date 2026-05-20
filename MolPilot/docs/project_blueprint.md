# MolPilot Blueprint

## Research Question

Can a molecular design model translate natural-language and multimodal context
into chemically verifiable molecule edits without relying on image-to-SMILES OCR
recovery?

## Positioning

MolPilot is not meant to be another generic text-to-SMILES generator. The core
claim is verified closed-loop molecular editing:

1. Build source-target editing tasks automatically from molecule collections.
2. Ground natural-language instructions into verifiable objectives.
3. Predict an edit/target latent with a JEPA-style context-to-target model.
4. Generate candidates with conditional latent diffusion.
5. Use deterministic chemistry verifiers for ranking, feedback, and reporting.

## What We Borrow

- From SketchMol: staged latent training and the need to compare against an
  image/OCR route.
- From UniVideo: separate the understanding stream from the generation stream.
- From JEPA: predict target embeddings from context instead of reconstructing
  raw pixels or raw SMILES.
- From medicinal chemistry workflows: source-conditioned lead optimization is
  more realistic than unconstrained de novo generation.

The JEPA implementation follows the project-level idea exposed by
`keon/jepa`'s minimal examples, especially masked/context-to-target latent
prediction and action-conditioned latent dynamics. MolPilot does not import that
code; it adapts the idea to molecular source/instruction/target latents.

## Unified Architecture

```text
source SMILES + optional image/mask + instruction
        |
        v
Understanding stream
  - language/spec grounding
  - source descriptors and rendered image features
  - task branch: edit / inpaint / de novo
        |
        v
Molecular JEPA planner
  - context = source latent + instruction/multimodal condition
  - target = verified target molecule latent
  - loss = target latent prediction + edit-delta prediction + contrastive loss
        |
        v
Conditional latent diffusion
  - samples molecular latents around the JEPA-predicted target/edit latent
        |
        v
Molecule decoder
  - current: sequence decoder plus nearest-latent fallback
  - next: stronger source-aware graph/SELFIES editor
        |
        v
Verifier loop
  - validity, property goals, scaffold/similarity/MW constraints
  - request-level top-k metrics and failure reasons
  - next: preference/RL feedback for verifier-guided training
```

## Current Evidence

The verified data filter fixed the random-target issue. On the 10k ChEMBL run,
candidate-level success improved from 0.0164 to 0.2066 after verified pair
construction. Verifier-aware ranking then showed that candidates often contain
valid solutions, but de novo dominates the aggregate number while edit/inpaint
remain weak.

The main technical gap is therefore not "more prompts"; it is source-aware
editing. The next paper-relevant step is to move verifier usage from post-hoc
ranking into training feedback.

## Main Experiments

1. Verified instruction editing:
   - edit overall@1/5/10
   - inpaint overall@1/5/10
   - constraint success, scaffold preservation, similarity, MW drift
   - macro task average so de novo generation cannot hide edit/inpaint failure
2. JEPA planner ablation:
   - no Stage 2 predictor
   - alignment MLP
   - JEPA predictor
   - JEPA + verifier preference feedback
3. OCR comparison:
   - SketchMol-style image/OCR route
   - direct molecule generation route
4. Generalization:
   - unseen paraphrases
   - unseen scaffolds
   - unseen objective combinations
5. Optional 3D extension:
   - conformer success
   - shape descriptors
   - no disease/docking claim unless a reliable proxy is added

## Acceptance Target

The project is not paper-ready until edit/inpaint request-level metrics improve.
The near-term target is:

```text
edit overall@10 >= 0.35
inpaint overall@10 >= 0.25
constraint success >= 0.60
exact train hit / trivial retrieval rate low enough for novelty claims
```
