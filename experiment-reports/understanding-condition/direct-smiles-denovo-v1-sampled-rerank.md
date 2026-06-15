# SUCC Direct SMILES De Novo v1 Sampled Rerank

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-15） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_benchmark.sh` / `submit_direct_smiles_denovo_ood_benchmark.sh` |
| **目的** | 修复 direct SMILES v0 的 autoregressive mode collapse，同时保持无 candidate-library retrieval |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16079726` | `succ-direct-smiles-2p7p` | 1h50m | 0 | `direct_smiles_denovo_2p7p_v1_sampled_rerank/` |
| `16079727` | `succ-direct-smiles-ood` | 23m | 0 | `direct_smiles_denovo_ood_v1_sampled_rerank/` |

## 背景

v0 在 teacher forcing 下 loss/perplexity 正常下降，但推理阶段严重塌缩：

| Benchmark | Strict | Validity | Unique valid |
| --- | ---: | ---: | ---: |
| 2p-7p v0 | 0.0007 | 1.000 | 1 / 6000 |
| OOD v0 | 0.000 | 0.094 | 1 / 1000 |

## v1 方法

不引入 molecule DB、`property_nearest`、`latent_property_rerank`。每个 prompt 从 decoder 采样 32 条 direct SMILES，再按 RDKit validity + absolute property rerank 选最优。

默认推理：temperature 0.85、top-k 40、top-p 0.95、repetition penalty 1.15、no-repeat ngram 6、max_new_tokens 96。

## 2p7p 结果（job `16079726`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.624 | 0.358 | 0.230 | 0.133 | 0.071 | 0.048 | **0.244** | 1.000 | **5967 / 6000** |

采样诊断（每 prompt 32 candidates）：

| 指标 | 值 |
| --- | ---: |
| mean valid candidates | 11.4 |
| mean unique valid candidates | 11.4 |
| mean selected strict fraction | 0.696 |
| unique generated (final) | 5967 |

## OOD 结果（job `16079727`）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: | ---: |
| **0.390** | **0.954** | **954 / 1000** | 0.243 | 0.050 |

采样诊断：mean valid candidates **3.3** / 32；final unique **1000**。

## 与 v0 / hybrid 对照

| 方法 | 2p7p strict | OOD strict | retrieval? |
| --- | ---: | ---: | --- |
| direct v0 greedy | 0.0007 | 0.000 | 否 |
| **direct v1 sampled rerank** | **0.244** | **0.390** | 否 |
| dualmode latent_nearest | 0.039 | 0.127 | 是 |
| hybrid latent_property_rerank | 0.970 | 0.985 | 是 |
| SketchMol ref（2p） | 0.804 | — | — |

## 结论

1. **Collapse 已打破**：v1 将 unique valid 从 1 提升到 5967/6000（2p7p）和 954/1000（OOD）；validity 从 9.4% 提升到 95.4%（OOD）。
2. **Real direct generation 已可用但仍弱于 retrieval hybrid**：2p7p strict **24.4%**、OOD **39%**，约为 hybrid 的 1/4–1/3；property count 越高 strict 越低（7p 仅 4.8%）。
3. **采样 rerank 是关键**：decoder 每 prompt 平均能产出 ~11（2p7p）/ ~3（OOD）个有效候选，rerank 从中选出相对满足属性的分子；无此步骤 v0 完全失败。
4. **仍远低于 SketchMol structured ref**（2p 62.4% vs 80.4%），说明条件→SMILES 对齐仍有较大空间。
5. **下一步**：增大 `num_samples`（64/128）；reward fine-tuning / SELFIES decoder；OOD rare_combo 专项；与 hybrid 差距需靠更强 decoder 而非 candidate DB。

## 提交命令（归档）

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_benchmark.sh

export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_benchmark.sh
```

产出：
- `outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/benchmark_direct_smiles/benchmark_report.md`
- `outputs/direct_smiles_denovo_ood_v1_sampled_rerank/benchmark_direct_smiles/benchmark_report.md`
