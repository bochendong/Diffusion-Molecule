# SUCC Direct SMILES De Novo v1 Sampled Rerank

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-15） |
| **最后更新** | 2026-06-15（n=128 推理复用） |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_benchmark.sh` / `submit_direct_smiles_denovo_ood_benchmark.sh` |
| **目的** | 修复 direct SMILES v0 的 autoregressive mode collapse，同时保持无 candidate-library retrieval |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16079726` | `succ-direct-smiles-2p7p`（n=32 训练+推理） | 1h50m | 0 | `direct_smiles_denovo_2p7p_v1_sampled_rerank/` |
| `16079727` | `succ-direct-smiles-ood`（n=32 训练+推理） | 23m | 0 | `direct_smiles_denovo_ood_v1_sampled_rerank/` |
| `16104381` | `succ-direct-smiles-2p7p-n64`（n=64 推理-only，复用 v1 ckpt） | 3h21m | 0 | `direct_smiles_denovo_2p7p_v1_sampled_rerank_n64/` |
| `16104383` | `succ-direct-smiles-ood-n64`（n=64 推理-only，复用 v1 ckpt） | 38m | 0 | `direct_smiles_denovo_ood_v1_sampled_rerank_n64/` |
| `16143574` | `succ-direct-smiles-2p7p-n128`（n=128 推理-only，复用 v1 ckpt） | 6h35m | 0 | `direct_smiles_denovo_2p7p_v1_sampled_rerank_n128/` |
| `16143575` | `succ-direct-smiles-ood-n128`（n=128 推理-only，复用 v1 ckpt） | 1h17m | 0 | `direct_smiles_denovo_ood_v1_sampled_rerank_n128/` |

## 背景

v0 在 teacher forcing 下 loss/perplexity 正常下降，但推理阶段严重塌缩：

| Benchmark | Strict | Validity | Unique valid |
| --- | ---: | ---: | ---: |
| 2p-7p v0 | 0.0007 | 1.000 | 1 / 6000 |
| OOD v0 | 0.000 | 0.094 | 1 / 1000 |

## v1 方法

不引入 molecule DB、`property_nearest`、`latent_property_rerank`。每个 prompt 从 decoder 采样 32 条 direct SMILES，再按 RDKit validity + absolute property rerank 选最优。

默认推理：temperature 0.85、top-k 40、top-p 0.95、repetition penalty 1.15、no-repeat ngram 6、max_new_tokens 96。

## 2p7p 结果（n=32，job `16079726`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.624 | 0.358 | 0.230 | 0.133 | 0.071 | 0.048 | **0.244** | 1.000 | **5967 / 6000** |

采样诊断（每 prompt 32 candidates）：mean valid **11.4**；mean selected strict fraction **0.696**。

## 2p7p 结果（n=64，job `16104381`，推理-only）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.759** | 0.556 | 0.363 | 0.233 | 0.128 | 0.084 | **0.354** | 1.000 | **5970 / 6000** |

| vs n=32 | Δ overall strict | mean valid candidates |
| --- | ---: | ---: |
| n=64 | **+11.0pp**（0.244→0.354） | **22.8**（was 11.4） |

2p strict **75.9%** 已接近 SketchMol ref **80.4%**。

## OOD 结果（n=32，job `16079727`）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: | ---: |
| **0.390** | **0.954** | **954 / 1000** | 0.243 | 0.050 |

采样诊断：mean valid candidates **3.3** / 32。

## OOD 结果（n=64，job `16104383`，推理-only）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: | ---: |
| **0.545** | **0.998** | **998 / 1000** | 0.413 | 0.070 |

| vs n=32 | Δ overall strict | mean valid candidates |
| --- | ---: | ---: |
| n=64 | **+15.5pp**（0.390→0.545） | **6.8**（was 3.3） |

## 2p7p 结果（n=128，job `16143574`，推理-only）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.850** | 0.674 | 0.508 | 0.346 | 0.218 | 0.160 | **0.459** | 1.000 | **5974 / 6000** |

| vs n=64 | Δ overall strict | mean valid candidates |
| --- | ---: | ---: |
| n=128 | **+10.5pp**（0.354→0.459） | **45.8**（was 22.8） |

2p strict **85.0%** 超过 SketchMol ref **80.4%**。

## OOD 结果（n=128，job `16143575`，推理-only）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: | ---: |
| **0.709** | **1.000** | **1000 / 1000** | 0.637 | 0.100 |

| vs n=64 | Δ overall strict | mean valid candidates |
| --- | ---: | ---: |
| n=128 | **+16.4pp**（0.545→0.709） | **13.7**（was 6.8） |

## num_samples 缩放（同一 v1 ckpt，推理-only）

| num_samples | 2p7p strict | OOD strict | mean valid（2p7p / OOD） |
| ---: | ---: | ---: | --- |
| 32 | 0.244 | 0.390 | 11.4 / 3.3 |
| 64 | 0.354 | 0.545 | 22.8 / 6.8 |
| **128** | **0.459** | **0.709** | **45.8 / 13.7** |

strict 随采样数单调上升；每翻倍约 +10–16pp，尚未饱和。

## 与 v0 / hybrid 对照

| 方法 | 2p7p strict | OOD strict | retrieval? |
| --- | ---: | ---: | --- |
| direct v0 greedy | 0.0007 | 0.000 | 否 |
| direct v1 n=32 | 0.244 | 0.390 | 否 |
| direct v1 n=64 | 0.354 | 0.545 | 否 |
| **direct v1 n=128** | **0.459** | **0.709** | 否 |
| dualmode latent_nearest | 0.039 | 0.127 | 是 |
| hybrid latent_property_rerank | 0.970 | 0.985 | 是 |
| SketchMol ref（2p） | 0.804 | — | — |

## 结论

1. **Collapse 已打破**（n=32 即证实）；**仅增大 `num_samples`、不重训**可持续提升 strict。
2. **n=128 当前 best direct**：2p7p **45.9%**、OOD **70.9%**；相对 n=32 提升 **+21.5pp / +31.9pp**。
3. **2p 已超过 SketchMol ref**（85.0% vs 80.4%）；高 property count（6p/7p）仍明显偏低（16–22%）。
4. **仍低于 retrieval hybrid**（97%/98.5%），但 OOD **70.9%** 已接近 random_property_rerank@k=64（66%）量级，证明是真实 decoder 采样而非 DB 检索。
5. **下一步**：试 n=256 看是否饱和；reward fine-tuning 提升 decoder 质量（减少靠堆采样）；7p / rare_combo 专项。
6. **运维备注**：首次 n=64 提交（`16102594`/`16102596`）因 PyTorch 2.6 `weights_only` 加载失败；`8167eef` 修复后成功。

## 提交命令（归档）

### n=32 全量（训练+推理，默认）

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_benchmark.sh

export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_benchmark.sh
```

### n=64 推理-only（复用 v1 checkpoint）

```bash
SUCC_DIRECT_DENOVO_RUN_TRAIN=0 \
SUCC_DIRECT_DENOVO_NUM_SAMPLES=64 \
SUCC_DIRECT_DENOVO_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank_n64 \
SUCC_DIRECT_DENOVO_RESUME_CHECKPOINT=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/direct_smiles_model/direct_smiles_generator.pt \
SUCC_DIRECT_DENOVO_TRAIN_ROWS_CSV=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/denovo_2p7p_train_rows.csv \
SUCC_DIRECT_DENOVO_EVAL_ROWS_CSV=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/denovo_2p7p_eval_rows.csv \
SUCC_DIRECT_DENOVO_TRAIN_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/train_condition_features_hf_vlm \
SUCC_DIRECT_DENOVO_EVAL_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/eval_condition_features_hf_vlm \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_benchmark.sh
```

OOD 侧将 `SUCC_DIRECT_DENOVO_*` 换为 `SUCC_DIRECT_OOD_*`，路径指向 `*_ood_*`。

产出：
- `outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/`（n=32）
- `outputs/direct_smiles_denovo_ood_v1_sampled_rerank/`（n=32）
- `outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank_n64/`
- `outputs/direct_smiles_denovo_ood_v1_sampled_rerank_n64/`
- `outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank_n128/`
- `outputs/direct_smiles_denovo_ood_v1_sampled_rerank_n128/`
