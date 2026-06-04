# Latent Edit Trajectory Attention

## Core Idea

Existing molecular diffusion models usually generate molecules in a single pass, or condition the next edit only on the current molecular state:

```text
p(z_{t+1} | z_t, target)
```

This project treats molecular design as a trajectory-level latent editing problem. Instead of letting the diffusion editor modify molecules in a memoryless way, the model attends over the full edit history:

```text
p(z_{t+1} | z_0, z_1, ..., z_t, target)
```

The key contribution is a memory-aware molecular diffusion model: each future latent edit is guided by previous molecular states, property changes, and optional expert feedback.

## Current Prototype

This folder now contains a minimal trainable prototype:

```text
latent_edit_trajectory_attention/
  models.py   # trajectory transformer + history-conditioned diffusion editor
  data.py     # synthetic latent optimization trajectories
  train.py    # CLI training entry point
tests/
  test_trajectory_attention.py
scripts/
  run_smoke.sh
```

Run the smoke test:

```bash
cd "/home/bdong/scratch/projects/Diffusion-Molecule/Latent Edit Trajectory Attention"
bash scripts/run_smoke.sh
```

Run a synthetic training job:

```bash
python3 -m latent_edit_trajectory_attention.train \
  --output-dir outputs/runs/synthetic_trajectory_attention_seed7 \
  --examples 256 \
  --history-length 6 \
  --epochs 5
```

Run with the project script and the `phystabmol` venv:

```bash
bash scripts/run_synthetic_training.sh
```

Submit to Slurm:

```bash
bash scripts/submit_synthetic_training.sh
```

Train on the original SketchMol optimization examples:

```bash
bash scripts/run_sketchmol_opt_training.sh
```

Submit the SketchMol opt-pair comparison job:

```bash
bash scripts/submit_sketchmol_opt_training.sh
```

SketchMol opt-pair jobs are CPU jobs by default because the dataset is small. Request a GPU only for larger sweeps:

```bash
LETA_USE_GPU=1 bash scripts/submit_sketchmol_opt_training.sh
```

Submit the current-state-only baseline:

```bash
bash scripts/submit_sketchmol_current_only_baseline.sh
```

Build a development multi-step trajectory file from SketchMol `opt_examples`:

```bash
bash scripts/bootstrap_sketchmol_trajectories.sh
```

Run the four trajectory-memory baselines:

```bash
bash scripts/run_baseline_suite.sh
```

Summarize paper metrics:

```bash
bash scripts/summarize_paper_metrics.sh
```

Useful overrides:

```bash
LETA_EPOCHS=50 LETA_EXAMPLES=8192 bash scripts/submit_synthetic_training.sh
LETA_EPOCHS=50 LETA_LATENT_DIM=512 bash scripts/submit_sketchmol_opt_training.sh
LETA_EPOCHS=50 bash scripts/submit_sketchmol_current_only_baseline.sh
LETA_EPOCHS=50 LETA_SUITE_NAME=paper_suite_v1 bash scripts/run_baseline_suite.sh
```

The current prototype supports both synthetic latent trajectories and SketchMol before/after optimization pairs. Real loaders should emit the same batch keys:

```text
z_history, property_delta, edit_type_ids, history_mask, next_z, target
```

For the SketchMol opt-pair path, `Before_opt_smiles` is encoded as `z_history[:, 0]`, `After_opt_smiles` is encoded as `next_z`, and available score/property deltas become `property_delta`.

For the publishable trajectory-memory path, use a trajectory JSONL with one row per optimization step:

```text
trajectory_id, step, smiles, parent_smiles, image_path, condition,
properties, delta_properties, reward, validity, molscribe_score,
failure_reason, selected_next_action, edit_type
```

The main comparison is:

```text
history model:
  epsilon_theta(z_next_noisy, noise_step, z_0...z_t, property_delta, edit_type, target)

current-only baseline:
  epsilon_theta(z_next_noisy, noise_step, z_t, target)

ablations:
  no_reward_history      # zero out reward/property trajectory tokens
  shuffled_history       # shuffle historical order before attention
```

## Motivation

Lead optimization is rarely a one-step decision. A useful edit may improve one property while hurting another, and the next edit often needs to compensate for previous trade-offs.

Example trajectory:

```text
Step 1: add CF3
  -> LogP improves and binding improves, but solubility drops

Step 2: add OH
  -> solubility recovers, but binding weakens

Step 3: replace with OMe
  -> balances binding and solubility
```

A current-state-only diffusion editor sees only the latest molecule. It does not know why CF3 was added, which edits failed, or which property trade-offs have already been explored. A trajectory attention module can learn that some edits improved binding, some edits harmed activity, and the next diffusion step should avoid repeating unproductive directions.

## Architecture

```text
Molecule / Sketch
        |
      Encoder
        |
       z_0
        |
 Diffusion Edit Step
        |
       z_1
        |
 Expert / Property Score r_1
        |
 [z_0, z_1, r_1, Delta r_1, edit_type_1]
        |
 Trajectory Transformer
        |
 history context h_t
        |
 Diffusion Editor
        |
      z_{t+1}
```

At each edit step, the trajectory is represented as a sequence of tokens:

```text
Token_i = [z_i ; Delta property_i ; edit_type_i]
```

The trajectory transformer summarizes the edit history:

```text
h_t = TrajectoryTransformer(Token_0, ..., Token_t)
```

The diffusion editor then conditions denoising on the current latent state, diffusion noise step, target objective, and history context:

```text
epsilon_theta(z_t, noise_step, h_t, target)
```

## Model Components

1. Encoder

Maps an input molecule or molecular sketch into a latent representation.

Possible inputs:

- molecular graph
- SMILES
- rendered molecular sketch
- SketchMol image embedding

2. Latent Edit Diffusion

Learns to generate the next latent molecular state from a noisy latent input. The editor is not only conditioned on the current molecule, but also on a trajectory context vector.

3. Trajectory Transformer

Applies attention over previous latent states and edit outcomes. This module gives the model memory over what has already been tried.

4. Property / Expert Feedback Tokens

Each step can include property feedback:

```text
(z_i, r_i, Delta r_i)
```

or a richer token:

```text
Token_i = [z_i ; Delta property_i ; edit_type_i]
```

This allows the model to learn not just which molecules appeared in the trajectory, but which edits produced useful or harmful changes.

## Training Signal

The model can be trained from molecular optimization trajectories:

```text
(x_0, x_1, ..., x_T)
```

with property scores:

```text
(r_0, r_1, ..., r_T)
```

Training objective:

- encode each molecule or sketch into latent state `z_i`
- build trajectory tokens from `z_i`, property deltas, and edit metadata
- predict the next latent state with a diffusion denoising loss
- optionally add auxiliary losses for property improvement, validity, similarity, and synthesizability

Core denoising objective:

```text
L_diff = E[||epsilon - epsilon_theta(z_t^noisy, noise_step, h_t, target)||^2]
```

Optional trajectory objective:

```text
L_rank = encourage edits with positive Delta property to receive higher attention or higher generation probability
```

## Connection To SketchMol

This direction naturally extends SketchMol-style image-to-structure modeling:

- the initial state can be a hand-drawn or generated molecular sketch
- the sketch encoder provides `z_0`
- the diffusion editor proposes structure-level modifications
- the trajectory attention module remembers previous visual/structural edits
- generated molecules can be rendered back to sketches for closed-loop validation

This turns SketchMol from a one-shot sketch-to-molecule model into an iterative molecular design system.

## Connection To Agent And Expert Feedback

The trajectory can be produced by an agent loop:

```text
generate edit -> score molecule -> store result -> attend over history -> generate next edit
```

The expert can be:

- docking score model
- QSAR predictor
- property predictor
- synthetic accessibility estimator
- human chemist preference
- RLME-style expert feedback module

This makes the method compatible with agentic molecular optimization, where each generation step updates the memory used by the next step.

## Main Claim

Do not let diffusion perform memoryless molecular edits. Let it remember the latent edit trajectory, use attention to summarize which previous edits helped or failed, and condition future diffusion steps on that history.

One-sentence version:

> We model molecular optimization as history-conditioned latent diffusion, where a trajectory transformer attends over previous molecular edits and expert feedback to guide the next molecular generation step.

## Why This Is A Strong Direction

- It gives a clear structural innovation beyond a standard diffusion model.
- It matches how real lead optimization works: iterative, feedback-driven, and path-dependent.
- It connects naturally to SketchMol through sketch/image encoders.
- It connects naturally to molecular agents through iterative scoring and memory.
- It can use RLME-style expert feedback as trajectory tokens.
- It creates a clean paper contribution: memory-aware molecular diffusion or history-conditioned molecular editing.

## Initial Risks

- Good trajectory data may be harder to obtain than single-molecule data.
- The model needs a meaningful definition of `edit_type_i`.
- Property deltas must be normalized carefully across objectives.
- Attention over long trajectories may overfit to dataset-specific optimization paths.
- Evaluation should compare against current-state-only diffusion to prove that memory helps.

## Baselines

Useful comparisons:

- one-shot molecular diffusion
- current-state-conditioned molecular diffusion
- graph-based molecular editing without trajectory attention
- reinforcement learning molecular optimization
- agentic optimization with non-neural memory

## Evaluation

Primary metrics:

- property improvement
- molecular validity
- novelty
- diversity
- similarity to starting molecule
- success rate under multi-objective constraints

Trajectory-specific metrics:

- improvement per edit step
- recovery from bad edits
- ability to avoid repeated failed edits
- final score versus path length
- attention alignment with helpful property changes

