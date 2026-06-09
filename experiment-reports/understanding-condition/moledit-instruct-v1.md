# SUCC UniVideo × MolEdit-Instruct v1

| 字段 | 值 |
| --- | --- |
| **状态** | ready to launch（MolEdit 数据 + OCR-free benchmark 已接线） |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Understanding-Condition` |
| **对齐对象** | `SketchMol-Unified-3MDiffusion` × MolEdit-Instruct v1 |

## 目的

将 `SketchMol-Understanding-Condition` 从 source-neighbor / OCR 评估主线切到
`MolEdit-Instruct enhanced_v1`，训练和 benchmark 口径与 Unified 3M × MolEdit 对齐。

## 数据

```text
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv
```

## 新入口

训练 + latent eval + dependent materialized benchmark：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_pipeline.sh
```

只重跑 materialized benchmark + MolEdit table metrics：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_benchmark.sh
```

## 关键配置

| 变量 | 值 |
| --- | --- |
| `SUCC_DATASET_MODE` | `moledit` |
| `SUCC_UNIFIED_OUTPUT_DIR` | `SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v1/` |
| `SUCC_CONDITION_FEATURES_DIR` | `SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v1/` |
| `SUCC_RUN_MOLSCRIBE_OCR` | `0` |
| `SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK` | `0` |
| `SUCC_DECODE_EVAL_IMAGES` | `0` |
| `SUCC_LATENT_BACKEND` | `image_vae` |
| `SUCC_LATENT_TARGET_MODE` | `residual` |
| `SUCC_DIFFUSION_OBJECTIVE` | `pred_x0` |

## Benchmark 对齐

Materialized benchmark 不再依赖 `image_path.csv` 或 MolScribe。默认从：

```text
dataset/univideo_edit_eval.jsonl
univideo_molecule/eval_latent/predictions.csv
univideo_molecule/eval_latent/generated_latents.npy
univideo_molecule/eval_latent/target_latents.npy
```

导出：

```text
univideo_molecule/benchmark_condition_rows.csv
```

再运行 `edit_latent_source_similarity_rerank` 等 Unified-style retrieval 方法，并可接
MolEditRL table metrics。

## 与旧线区别

1. 不使用 `multiproperty_source_neighbor_v1/condition_rows.csv` 作为主数据。
2. 不跑 `submit_succ_ocr_benchmark.sh` / MolScribe OCR。
3. 旧 source-neighbor v2 保留为历史对照；新结果应写入本报告。
