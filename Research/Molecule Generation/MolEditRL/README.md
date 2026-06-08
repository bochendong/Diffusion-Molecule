# MolEditRL benchmark note

This folder tracks public assets for:

```text
MolEditRL: Structure-Preserving Molecular Editing via Discrete Diffusion and Reinforcement Learning
arXiv:2505.20131
ICLR 2026 Poster
```

## Current code status

I searched the public paper page, OpenReview record, CatalyzeX, Hugging Face, and GitHub for an official MolEditRL implementation. As of 2026-06-08, I did not find a public official code repository.

What is public:

- Paper: https://arxiv.org/abs/2505.20131
- OpenReview: https://openreview.net/forum?id=40QphlZ9fY
- Dataset: https://huggingface.co/datasets/FanSiLeC/MolEdit-Instruct
- Dataset file: `MolEdit-Instruct_3034459.txt`, 3,034,459 rows, about 768 MB

The arXiv source package contains LaTeX and figures only. The OpenReview camera-ready record has a `supplementary_material` field in its presentation schema, but the public content does not expose a downloadable supplement. The paper source checklist says the authors planned to release code and pretrained models after acceptance, so this folder should be revisited if an official repository appears.

## Why this is relevant for us

MolEditRL is aligned with our main direction: source-conditioned molecular editing under natural language property instructions while preserving structure. Its public MolEdit-Instruct dataset is directly useful as a text/SMILES benchmark:

```text
example_id<TAB>instruction<TAB>source_smiles<TAB>target_smiles
```

This complements our current benchmark stack because it is not image-first. It can test whether a method can follow edit instructions and keep source similarity when the source molecule is given as SMILES.

## Fetch public assets

Dry-run metadata check:

```bash
bash "Research/Molecule Generation/MolEditRL/scripts/fetch_public_assets.sh" --dry-run
```

Download paper assets and the full public dataset:

```bash
MOLEDITRL_DOWNLOAD_DATASET=1 \
bash "Research/Molecule Generation/MolEditRL/scripts/fetch_public_assets.sh"
```

Downloaded files are ignored by git:

```text
paper/arxiv-2505.20131.pdf
paper/arxiv-2505.20131-source.tar.gz
data/raw/MolEdit-Instruct_3034459.txt
```

## Convert dataset into benchmark manifests

After downloading the Hugging Face text file:

```bash
python3 "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_instruct_manifest.py" \
  --input "Research/Molecule Generation/MolEditRL/data/raw/MolEdit-Instruct_3034459.txt" \
  --output-csv "Research/Molecule Generation/MolEditRL/data/processed/moledit_instruct.csv" \
  --output-jsonl "Research/Molecule Generation/MolEditRL/data/processed/moledit_instruct.jsonl"
```

For a quick local smoke subset:

```bash
python3 "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_instruct_manifest.py" \
  --input "Research/Molecule Generation/MolEditRL/data/raw/MolEdit-Instruct_3034459.txt" \
  --output-csv "Research/Molecule Generation/MolEditRL/data/processed/moledit_instruct_head1000.csv" \
  --limit 1000
```

The parser intentionally preserves the original instruction text and SMILES strings without canonicalization. Canonicalization and property scoring should happen in the benchmark runner so it can share the same RDKit/TDC environment as the rest of this repo.

## Pre-enhance MolEdit-Instruct for faster reuse

For the full 3.0M-row dataset, use the shardable enhancement pipeline instead of recomputing RDKit features in every experiment. It writes only to the online data root:

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
export MOLEDIT_RAW_INPUT="$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt"
export MOLEDIT_OUTPUT_DIR="$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1"
export MOLEDIT_SHARDS=64

bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh" --dry-run
bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh"
```

The job wrapper submits four dependent stages when `sbatch` is available:

```text
normalize-pairs -> molecule-cache -> pair-features -> finalize
```

`molecule-cache` is the only RDKit-dependent stage. It canonicalizes each unique raw SMILES, computes common properties, Murcko scaffold, and Morgan fingerprint on-bit indices once. `pair-features` joins those cache rows back to edit pairs and computes source-target Tanimoto, scaffold match, active property deltas, instruction-task hints, and difficulty buckets.

Expected reusable outputs:

```text
$MOLEDIT_OUTPUT_DIR/
  pairs/pairs_shard_00000_of_00064.csv
  smiles/smiles_shard_00000_of_00064.txt
  molecule_cache/molecule_cache_shard_00000_of_00064.csv
  enhanced_pairs/enhanced_pairs_shard_00000_of_00064.csv
  splits/train.csv
  splits/eval_balanced.csv
  splits/eval_hard.csv
  splits/smoke_1000.jsonl
  splits/summary.json
```

For a small serial smoke run on a machine with RDKit:

```bash
MOLEDIT_LOCAL=1 \
MOLEDIT_SHARDS=2 \
MOLEDIT_LIMIT=1000 \
MOLEDIT_OUTPUT_DIR="$DM_DATA_ROOT/processed/moledit-instruct/enhanced_smoke" \
bash "Research/Molecule Generation/MolEditRL/scripts/prepare_moledit_enhancement_jobs.sh"
```

The generated `splits/train.csv`, `splits/eval_balanced.csv`, and `splits/eval_hard.csv` are the intended benchmark manifests. Prefer these over the plain `moledit_instruct.csv` once the enhanced build is available.

## Benchmark integration notes

Suggested minimal comparison surface:

```text
input: source_smiles + instruction
prediction: predicted_target_smiles
reference: target_smiles
metrics: validity, property success, source Tanimoto, strict success at Tanimoto >= 0.4/0.6/0.8
```

One caveat: OpenReview author replies say MolEditRL uses a structured task label to select the RL reward oracle, but the public Hugging Face txt file exposes only `example_id`, `instruction`, `source_smiles`, and `target_smiles`. For property-success evaluation, infer the task from instruction templates or recompute property deltas from source/target molecules in the same RDKit/TDC environment used by the rest of this repo.

For our current reporting style, put source-similarity metrics before scaffold-only diagnostics. This benchmark should help distinguish real source-conditioned editing from property-only retrieval.
