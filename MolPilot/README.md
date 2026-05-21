# MolPilot

Independent prototype for verified JEPA-guided molecular editing and generation.

This folder is intentionally separate from `PhysTabMol`. It is a new research
line inspired by two local codebases:

- SketchMol: property-conditioned molecular image diffusion, repaint-style
  inpainting, and property/protein prompt experiments.
- UniVideo: an understanding stream that turns text/image/video context into
  conditioning tokens for a diffusion generator, with separate conditioning
  branches for unconditional, text-only, and multimodal guidance.

MolPilot adapts those ideas to molecules without importing either project.

The current research direction is summarized in
[`docs/project_blueprint.md`](docs/project_blueprint.md): verified
closed-loop molecular editing, with a JEPA-style latent planner and a chemical
verifier used for data construction, ranking, and future training feedback.

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
Molecular JEPA planner
  - predicts target/edit molecule latent from source latent + instruction
  - learns latent edits instead of hand-written edit actions
        |
        v
Conditional molecular latent diffusion
  - task token: edit / inpaint / de_novo
  - conditioned by JEPA-predicted target/edit latent
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
MOLPILOT_DATA=../PhysTabMol/data/molecules.csv \
MOLPILOT_RUN_NAME=molpilot_edit_inpaint_denovo_v1 \
bash scripts/run_staged_server.sh
```

Submit the staged run to Slurm:

```bash
cd MolPilot
MOLPILOT_DATA=../PhysTabMol/data/molecules.csv \
MOLPILOT_RUN_NAME=molpilot_staged_v1 \
sbatch scripts/run_staged_server.slurm.sh
```

Or use the one-command ChEMBL submit helper. It defaults to a 10k run:

```bash
cd MolPilot
bash scripts/submit_chembl_staged.sh
```

The helper defaults to a 40GB H100 MIG profile with `--gpus=h100_3g.40gb:1`
and `--mem-per-cpu=4096M`. For smaller slices, set
`MOLPILOT_GPU_PROFILE=h100_10gb_mig` or `h100_20gb_mig`; for a whole H100, set
`MOLPILOT_GPU_PROFILE=h100_full`. You can also override training knobs such as
`MOLPILOT_AE_BATCH_SIZE`, `MOLPILOT_AE_HIDDEN_DIM`,
`MOLPILOT_DIFFUSION_BATCH_SIZE`, and `MOLPILOT_DIFFUSION_HIDDEN_DIM`. If Slurm
uses a different resource name, override it with `MOLPILOT_SLURM_GPUS`.
The default staged Slurm time is 2 hours; for larger 100k/full runs, override
with `MOLPILOT_SLURM_TIME=08:00:00` or another appropriate value.
The submit helper defaults to `MOLPILOT_STAGE2_MODEL=jepa` and uses
`/scratch/bdong/venvs/phystabmol/bin/python` when that venv exists. If needed,
override either value with `MOLPILOT_STAGE2_MODEL=alignment` or `PYTHON_BIN=...`.
Stage 4 uses task-balanced evaluation by default: it builds requests from
`MOLPILOT_EVAL_MOLECULE_LIMIT` molecules, then caps each task family with
`MOLPILOT_MAX_REQUESTS_PER_TASK` or `MOLPILOT_EVAL_LIMIT`.

Resample an existing trained stage without retraining:

```bash
cd MolPilot
MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260519_190716 \
bash scripts/resample_existing_stage.sh
```

Or submit the same resampling step to Slurm:

```bash
cd MolPilot
MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260519_190716 \
sbatch --export=ALL scripts/resample_existing_stage.slurm.sh
```

This writes ranked samples to `stage4_samples_ranked`, including
`request_metrics.csv` for `overall@1/5/10` and `failure_reasons.csv` for
constraint debugging. Set `MOLPILOT_DISABLE_VERIFIER_RANKING=1` to produce a
no-ranking ablation.

For `edit` and `inpaint`, Stage 4 now enables source-guided sampling by
default. It decodes raw diffusion samples, source-anchored latent variants,
local source-neighborhood candidates, graph-level R-group edits, and same
scaffold analog candidates, then records `candidate_origin` plus `origin_*`
metrics. Use `MOLPILOT_DISABLE_SOURCE_GUIDANCE=1` for a full source-guidance
ablation, `MOLPILOT_DISABLE_GRAPH_EDITOR=1` to remove graph-level edits,
`MOLPILOT_SOURCE_EDIT_STRENGTHS=0.20,0.40,0.65` to tune how much generated
delta is kept, `MOLPILOT_GRAPH_EDIT_LIMIT=128` to widen RDKit local edits, and
`MOLPILOT_SCAFFOLD_LIBRARY_K=0` to disable same-scaffold analog retrieval.

Task-balanced resampling can be controlled with:

```bash
MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260519_190716 \
MOLPILOT_EVAL_MOLECULE_LIMIT=10000 \
MOLPILOT_MAX_REQUESTS_PER_TASK=1000 \
MOLPILOT_EVAL_TASKS=edit,inpaint,de_novo \
bash scripts/resample_existing_stage.sh
```

Overnight source-decoder ablations can be submitted in one command:

```bash
MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260520_015836 \
MOLPILOT_ABLATION_TIME=08:00:00 \
bash scripts/run_overnight_resample_ablation.sh
```

This requests an 8-hour walltime for each job, but jobs exit early when the
sweep is done. The default `overnight` profile submits quick isolation runs
plus heavier edit/inpaint sweeps: `full_graph`, `graph_only`,
`scaffold_library_only`, `latent_only`, `diffusion_only`, `full_no_library`,
`full_wide`, `graph_heavy`, `scaffold_library_heavy`, `full_heavy`, and
`full_heavy_seed17`. Use `MOLPILOT_ABLATION_PROFILE=quick` if you only want the
seven shorter runs. Summarize them after the jobs finish:

```bash
MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260520_015836 \
bash scripts/summarize_resample_ablation.sh
```

For the full 100k run:

```bash
cd MolPilot
MOLPILOT_LIMIT=100000 \
MOLPILOT_EVAL_LIMIT=5000 \
MOLPILOT_RUN_NAME=molpilot_sequence_100k_v1 \
bash scripts/submit_chembl_staged.sh
```

The server script intentionally fails if `MOLPILOT_DATA` does not exist. Use
`scripts/run_staged_smoke.sh` when you want the built-in 6-molecule debug run.

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
  source latent + text/SMILES/image/spec condition -> target/edit molecular latent
  default model: JEPA-style predictor

Stage 3: train conditional molecular latent diffusion
  JEPA-predicted condition latent -> target molecular latent

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
MOLPILOT_STAGE2_MODEL=alignment bash scripts/run_staged_server.sh
MOLPILOT_STAGE2_MODEL=jepa bash scripts/run_staged_server.sh
MOLPILOT_CODEC=feature bash scripts/run_staged_server.sh
MOLPILOT_CODEC=sequence bash scripts/run_staged_server.sh
```
