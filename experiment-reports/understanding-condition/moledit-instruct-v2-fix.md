# SUCC UniVideo × MolEdit-Instruct v2 Fix

| 字段 | 值 |
| --- | --- |
| **状态** | ready to launch |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `scripts/submit_univideo_moledit_v2_fix_pipeline.sh` |
| **基线** | v1: [moledit-instruct-v1.md](moledit-instruct-v1.md) |

## 目的

修复 v1 暴露的两个问题：

1. latent 太贴 source：`source latent cosine≈0.991`，edit signal 不够。
2. MolEdit table metrics 落后 MolEditRL，尤其 MW↑ 当前为 0。

本轮不改成 OCR 路线，仍然保持 MolEdit-Instruct enhanced_v1 +
OCR-free materialized benchmark，与 Unified 3M x MolEdit 的 testing 口径对齐。

## 主要改动

| 模块 | 改动 |
| --- | --- |
| dataset export | 支持 `--moledit-balanced-train-per-task` / `--moledit-balanced-eval-per-task` |
| dataset export | 支持 Table1-only balanced subset，默认 v2 train 每 task 5000、eval 每 task 100 |
| properties | SUCC `PROPERTY_COLUMNS` 加入 `SA`；有 RDKit SA scorer 时自动补 source/target/delta SA |
| train sampler | 新增 `--sampling-strategy weighted`，按 Table1 task / active property 加权采样 |
| aux loss | target property / delta / direction loss 默认只监督 active instruction properties |
| aux loss | 新增 `--aux-property-weights`，v2 默认 `MW=3,SA=3,RB=2` |
| pipeline | 新增 v2 wrapper，不覆盖 v1 输出 |

## 默认 v2 配置

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v2_fix_pipeline.sh
```

Wrapper 默认关键变量：

| 变量 | 值 |
| --- | --- |
| `SUCC_UNIFIED_OUTPUT_DIR` | `SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix` |
| `SUCC_CONDITION_FEATURES_DIR` | `SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v2_fix` |
| `SUCC_MOLEDIT_TABLE1_TASKS_ONLY` | `1` |
| `SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK` | `5000` |
| `SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK` | `100` |
| `SUCC_SAMPLING_STRATEGY` | `weighted` |
| `SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS` | `MW=4,SA=3,RB=2` |
| `SUCC_AUX_PROPERTY_WEIGHTS` | `MW=3,SA=3,RB=2` |
| `SUCC_AUX_LOSS_WEIGHT` | `0.50` |
| `SUCC_RESIDUAL_SAMPLE_SCALE` | `1.25` |
| `SUCC_CONNECTOR_LATENT_BLEND` | `0.10` |
| OCR | off |

## 要看的指标

优先看：

1. `univideo_molecule/eval_latent/metrics.json`
   - `source_latent_cosine` 是否从 v1 的 0.991 降下来。
   - `target_latent_cosine` 是否从 v1 的 0.380 上升。
2. `univideo_molecule/benchmark_materialized_primary_fast/benchmark_report.md`
   - 主看 `edit_latent_source_similarity_rerank`，不要主推 `source_first_rerank`。
   - mean source Tani 希望保持在 0.4 左右或更高。
3. `univideo_molecule/moledit_table_metrics/moledit_table_summary.md`
   - MW↑ 是否从 0 起步。
   - Rotbonds↓ / SA↓ 是否高于 v1。

## 对照目标

| metric | v1 SUCC | Unified 3M | MolEditRL overlap baseline | v2 目标 |
| --- | ---: | ---: | ---: | ---: |
| overlap Acc_all(0.65) mean | 0.222 | 0.000 | 0.555 | >0.30 |
| overlap Acc_all(0.15) mean | 0.445 | 0.111 | 0.838 | >0.55 |
| MW↑ Acc_all(0.65) | 0.000 | 0.000 | 0.404 | >0 |
| MW↑ Acc_all(0.15) | 0.000 | 0.000 | 0.856 | >0 |

## 风险

- GSK3B / DRD2 仍需要 PyTDC oracle，没装时 table metrics 会按
  `skip-task` 跳过。
- 本轮把 `SA` 纳入 SUCC property head，但如果服务器 RDKit 没有
  `SA_Score/sascorer.py`，SA 数值监督会退化为 direction/text 信号。
- Table1-only 训练更利于和 MolEditRL 对齐，但可能牺牲 broader MolEdit
  enhanced split 上的泛化；需要和 v1 并排读。
