# Unified SMILES Generator GPU Memory Plan

These are conservative planning estimates for the first server runs. Actual
memory depends on the HF/VLM checkpoint, sequence length, row count, and
PyTorch/transformers versions.

## Stages

| Stage | Default script | GPU memory estimate | Suggested GPU | Notes |
| --- | --- | ---: | --- | --- |
| Frozen HF/VLM feature export, `full` with source image/render | `run_unified_smiles_generator_feature_variants.sh` | 26-38 GB | H100 40GB MIG or full H100 | Qwen2.5-VL-7B class models dominate memory. Keep `SUCC_HF_BATCH_SIZE=1`. |
| Frozen HF/VLM feature export, `text_only` | `run_unified_smiles_generator_feature_variants.sh` | 18-30 GB | H100 40GB MIG | Usually lighter than image mode, but same VLM weights are loaded. |
| Unified SFT decoder | `run_unified_smiles_generator_train.sh` | 4-10 GB | 20GB MIG is usually enough | Small Transformer decoder: d=256, 4 layers. |
| Unified group RL | `run_unified_smiles_generator_group_rl.sh` | 8-18 GB | 20GB MIG ok; 40GB safer | Memory scales with `batch_size * rollouts_per_prompt * max_new_tokens`. |
| Sampling, stochastic `sample` | `run_unified_smiles_generator_benchmark_suite.sh` | 4-12 GB | 20GB MIG | Memory scales with `parallel_samples`; runtime scales with `num_samples`. |
| Sampling, `beam` | `run_unified_smiles_generator_benchmark_suite.sh` | 4-10 GB | 20GB MIG | Often lower memory than large parallel sampling, but slower because beam is row-wise. |
| RDKit/TDC benchmark evaluation | `unified_benchmark_runner.py` | mostly CPU | CPU or same job | TDC oracles may be slow but not GPU-heavy. |

## First-Pass Settings

Use these for the first real run:

```bash
export SUCC_HF_BATCH_SIZE=1
export SUCC_UNIFIED_EPOCHS=1
export SUCC_UNIFIED_RL_EPOCHS=1
export SUCC_UNIFIED_RL_BATCH_SIZE=4
export SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT=8
export SUCC_UNIFIED_NUM_SAMPLES=20
export SUCC_UNIFIED_BEAM_SIZE=20
export SUCC_UNIFIED_TOP_K_CANDIDATES=20
```

After smoke metrics look sane:

```bash
export SUCC_UNIFIED_RL_BATCH_SIZE=8
export SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT=16
export SUCC_UNIFIED_NUM_SAMPLES=40
export SUCC_UNIFIED_BEAM_SIZE=40
export SUCC_UNIFIED_TOP_K_CANDIDATES=40
```

Hold `n=128/256` and MolEdit assisted `n=2048` until the top-20/top-40 suite is
stable.

## Queue Strategy

Run feature export separately from decoder training and benchmarking:

```text
1. feature export: full,text_only
2. SFT/group-RL: with_image and no_image checkpoints
3. benchmark suite: sample and beam decoding
```

This avoids putting Qwen-VL feature export and all benchmark inference inside
one long job. The default experiment-suite submit script does not export
features or train unless explicitly enabled.

## Expected Wait-Time Risk

| Choice | Risk | Recommendation |
| --- | --- | --- |
| `h100_full` for every stage | Long queue | Use only for VLM export if 40GB MIG fails. |
| `h100_40gb_mig` | Balanced | Good default for VLM export and RL. |
| `nvidia_h100_80gb_hbm3_2g.20gb` | Shorter queue but tighter | Good for SFT/sampling; may be tight for Qwen-VL full image export. |
| `a100:1` | Available on some clusters | Usually fine for decoder, may be tight or slower for VLM export. |

If a job is waiting too long, first reduce `SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT`
or sampling budget before requesting a larger GPU.
