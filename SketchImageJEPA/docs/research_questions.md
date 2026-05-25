# SketchImage-JEPA Paper Track

## Main Claim

SketchImage-JEPA should be developed as a molecular design planner, not as a
temperature-tuned retrieval baseline. The paper-facing claim is:

> A predictive JEPA latent planner can align sketch/image, source-molecule, and
> task context signals into a target molecular design space that improves
> controllable molecule generation and editing.

The current retrieval decoder is a diagnostic scaffold. It is useful because it
shows whether the planner predicts a target region of chemical space, but it is
not the final top-conference contribution by itself.

## Research Questions

### RQ1: Planner Representation

Can a JEPA-style molecular planner learn a transferable latent edit target from
source structure, visual context, and task instruction?

Minimum evidence:

- The same planner improves de novo, edit, inpaint, and fragment-grow tasks.
- Gains hold across multiple split seeds.
- The planner improves both top-1 quality and best-of-k candidate quality.
- Task-type breakdowns show where the representation transfers and where it
  fails.

### RQ2: Contrastive Objective

Does hard-negative contrastive alignment create a better molecular design space
than regression/cosine prediction alone?

Minimum evidence:

- Best planner versus no-contrastive ablation.
- Weak versus strong contrastive alignment.
- Training history records active contrastive losses.
- Improvements are not explainable only by post-hoc reranking.

### RQ3: Visual/Sketch Context

Does rendered sketch/image context provide information beyond text and source
SMILES context?

Minimum evidence:

- Best planner with image context versus the same planner without image context.
- Task-type analysis, especially inpaint and fragment-grow rows.
- Qualitative examples where the image/mask-like context changes the candidate
  ranking in a chemically plausible way.

### RQ4: Generative Decoder

Can the JEPA planner guide a true molecular decoder instead of stopping at
retrieval?

Minimum evidence:

- Replace retrieval with a graph, SELFIES, diffusion, or flow decoder conditioned
  on the predicted planner latent.
- Compare direct generation versus JEPA-planned generation.
- Report validity, novelty, diversity, property success, and local-edit
  preservation.

### RQ5: Benchmark Contract

Do current SketchMol-style systems solve realistic local-control molecule
editing, or do they mostly solve nearest-neighbor similarity?

Minimum evidence:

- Fixed train/eval split construction.
- Strong internal baselines and, when available, official external baselines.
- Multiple seeds.
- No target-oracle ranking.
- Metrics reported overall and by task type.

## Immediate Experiment Priority

Run a paper ablation matrix before further temperature sweeps:

1. `ridge_baseline`: existing SketchMol-aligned CPU baseline.
2. `planner_best`: best current contrastive planner.
3. `planner_v2`: stronger edit planner with explicit delta and hard-negative
   losses.
4. `no_contrastive`: same planner without contrastive loss.
5. `weak_contrastive`: weaker contrastive setting.
6. `no_image_context`: best planner without rendered image context.

The pilot matrix uses one seed to check direction. The full matrix uses three
seeds and is the first paper-facing comparison table.

## Stop Criteria

Stop treating hyperparameter tuning as the main work once:

- `planner_best` beats `ridge_baseline` across seeds.
- `no_contrastive` is clearly worse than `planner_best`.
- `no_image_context` quantifies whether the sketch/image branch matters.

After that, move effort to the generative decoder and stronger external
benchmarks.

## Current Implementation Plan

1. Audit the easy split to quantify nearest-neighbor shortcuts.
2. Build a scaffold/nearest-neighbor controlled hard split.
3. Run the paper matrix on both the easy split and the hard split.
4. Promote `planner_v2` only if it improves over `planner_best` on the hard
   split, not just on the easy split.
