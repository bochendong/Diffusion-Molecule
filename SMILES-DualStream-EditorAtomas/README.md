# SMILES DualStream Editor-Atomas

This is a separate pure-SMILES experiment. It intentionally does not import
the existing SketchMol image, VLM, OCR, or diffusion packages, so changes here
do not affect other experiments in this repository.

## Why This Exists

The current main repo uses a two-stream design:

```text
understanding stream: source molecule image + instruction -> edit condition
generation stream: source-conditioned generation -> target molecule
```

This folder keeps the two-stream idea but removes images completely:

```text
SMI-Editor-style edit stream:
  corrupted/source SMILES -> target SMILES edit reconstruction

Atomas-style hierarchy stream:
  source/corrupted SMILES <-> target SMILES alignment at token, fragment,
  and molecule levels
```

The external ideas are used as design references, not vendored code:

- SMI-Editor official repo: https://github.com/zhengkangjie/smi-editor
- Atomas official repo: https://github.com/yikunpku/Atomas

## Design

### Stream A: Edit Stream

Inspired by SMI-Editor, this stream builds fragment-level edit supervision from
SMILES only. For self-supervised rows, it disrupts a clean SMILES by masking,
deleting, or shuffling fragment spans and trains toward the original string.
For source-target edit rows, it treats the source SMILES as the edit input and
the target SMILES as the reconstruction target.

### Stream B: Hierarchical Alignment Stream

Inspired by Atomas, this stream builds three aligned views from SMILES tokens:

```text
token / atom level
fragment level
molecule level
```

Because this experiment is pure SMILES, the hierarchy aligns
`source_or_corrupted_smiles` against `target_smiles`, instead of molecule-image
or molecule-text pairs.

## Quick Start

Run the standard-library smoke path:

```bash
cd /Users/dongpochen/Github/Diffusion\ Molecule
python3 SMILES-DualStream-EditorAtomas/scripts/run_smoke.py
```

Prepare JSONL training examples from a CSV with `source_smiles,target_smiles`:

```bash
python3 SMILES-DualStream-EditorAtomas/scripts/prepare_manifest.py \
  --input-csv /path/to/diffusion_edit_manifest.csv \
  --output-jsonl SMILES-DualStream-EditorAtomas/outputs/train.jsonl \
  --source-column source_smiles \
  --target-column target_smiles
```

Prepare self-supervised rows from a plain SMILES CSV:

```bash
python3 SMILES-DualStream-EditorAtomas/scripts/prepare_manifest.py \
  --input-csv SketchMol-MultiProperty-EditDataset/data/train_table.csv \
  --output-jsonl SMILES-DualStream-EditorAtomas/outputs/pretrain.jsonl \
  --smiles-column smiles \
  --limit 1000
```

The model training entry is optional and requires PyTorch:

```bash
python3 SMILES-DualStream-EditorAtomas/scripts/train_dual_stream.py \
  --train-jsonl SMILES-DualStream-EditorAtomas/outputs/train.jsonl \
  --output-dir SMILES-DualStream-EditorAtomas/outputs/model_smoke
```

## Large-Scale Training

The large run is config-driven and resumable. It prepares the pure-SMILES
manifest, trains with edit reconstruction plus molecule/token/fragment
alignment losses, writes `latest_checkpoint.pt` after every epoch, and appends
metrics to `train_log.jsonl`.

Dry-run the server plan first:

```bash
SDEA_DRY_RUN=1 \
bash SMILES-DualStream-EditorAtomas/scripts/submit_large_train.sh
```

Submit the default 20GB H100 MIG run:

```bash
bash SMILES-DualStream-EditorAtomas/scripts/submit_large_train.sh
```

Use a 40GB profile when you want a wider/faster run:

```bash
SDEA_SLURM_GPUS=h100_3g.40gb:1 \
SDEA_SLURM_CPUS=8 \
SDEA_SLURM_MEM=64G \
SDEA_SLURM_TIME=48:00:00 \
bash SMILES-DualStream-EditorAtomas/scripts/submit_large_train.sh
```

Prepare only the JSONL manifest:

```bash
bash SMILES-DualStream-EditorAtomas/scripts/prepare_large_manifest.sh --overwrite-manifest
```

Run locally in an already provisioned shell:

```bash
SDEA_LOCAL_RUN=1 \
SDEA_PYTHON_BIN=/path/to/python-with-torch \
bash SMILES-DualStream-EditorAtomas/scripts/submit_large_train.sh
```

Default large config:

```text
SMILES-DualStream-EditorAtomas/configs/large.yaml
```

Important outputs:

```text
SMILES-DualStream-EditorAtomas/outputs/manifests/large_train.jsonl
SMILES-DualStream-EditorAtomas/outputs/runs/large/latest_checkpoint.pt
SMILES-DualStream-EditorAtomas/outputs/runs/large/checkpoint_epoch_XXXX.pt
SMILES-DualStream-EditorAtomas/outputs/runs/large/train_log.jsonl
SMILES-DualStream-EditorAtomas/outputs/runs/large/summary.json
```

On this local machine PyTorch and RDKit may be absent, so the dependency-light
parts can be validated locally, while actual model training should run in the
server environment.

## Outputs

`prepare_manifest.py` writes rows like:

```json
{
  "sample_id": "row_000001",
  "mode": "pair_edit",
  "source_smiles": "CCO",
  "corrupted_smiles": "CCO",
  "target_smiles": "CCN",
  "edit": {"levenshtein": 1, "replace": 1},
  "alignment": {"token_jaccard": 0.5, "fragment_jaccard": 0.0}
}
```

No image paths are produced or consumed.
