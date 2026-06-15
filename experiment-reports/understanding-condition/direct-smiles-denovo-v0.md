# SUCC Direct SMILES De Novo v0

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-15） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_benchmark.sh` / `submit_direct_smiles_denovo_ood_benchmark.sh` |
| **目的** | 建立不依赖 candidate retrieval / property rerank 的 real direct de novo baseline |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16079256` | `succ-direct-smiles-2p7p` | 9m51s | 0 | `direct_smiles_denovo_2p7p_v0/` |
| `16079257` | `succ-direct-smiles-ood` | 4m19s | 0 | `direct_smiles_denovo_ood_v0/` |

## 动机

`latent_property_rerank` 在 2p7p/OOD 上达到 0.970/0.985，但 random-shortlist sanity
显示 `random_property_rerank@4096` 也能达到 0.978/0.980。因此该高分主要来自
candidate library + property rerank，不能作为 direct de novo generation claim。

Direct SMILES v0 路径：

```text
condition text / property prompt
  -> Qwen/HF VLM query tokens
  -> Transformer SMILES decoder
  -> generated_smiles
  -> direct-SMILES evaluator
```

不使用 candidate library，不使用 `property_nearest`，不使用 materializer rerank。

## 方法

| 参数 | 2p7p | OOD |
| --- | ---: | ---: |
| Train rows | 12000（2000/property-count） | 8000（800/spec） |
| Eval rows | 6000 | 1000 |
| Epochs | 12 | 12 |
| Model | 4-layer Transformer decoder | 4-layer Transformer decoder |
| Condition | Qwen `query_tokens.npy`（dim=256） | 同左 |
| Vocab size | 47 | 47 |

训练 loss 正常下降（2p7p eval perplexity 4.02→2.36；OOD 4.63→2.40），但生成严重塌缩。

## 2p7p 结果（job `16079256`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.0007** | 1.000 | **1 / 6000** |

| 对照 | overall strict |
| --- | ---: |
| direct_smiles_mllm v0 | **0.0007** |
| dualmode v1 `latent_nearest` | 0.039 |
| hybrid `latent_property_rerank` | 0.970 |
| `property_nearest` 上界 | 0.980 |
| SketchMol structured ref（2p） | 0.804 |

**Mode collapse**：6000 行全部生成同一超长烷烃链 SMILES（`CCCC...`，约 120 个 C），与 property prompt 无关。`prediction_summary.unique_generated = 1`。

## OOD 结果（job `16079257`）

| Overall strict | validity | unique valid | unique generated |
| ---: | ---: | ---: | ---: |
| **0.000** | **0.094** | **1 / 1000** | 10 |

仅 94/1000 行 RDKit valid；有效行也几乎同一分子。2p bucket strict 0%；7p bucket validity 94% 但 strict 0%。

## 结论

1. **Real direct generation 目前远不可用**：v0 在去掉 candidate rerank 后 strict 接近 0，与 hybrid 97%/98.5% 形成鲜明对比，印证后者不能代表生成能力。
2. **训练 perplexity 下降 ≠ 条件生成成功**：teacher-forcing loss 收敛，但 autoregressive 推理塌缩到单一超长 SMILES；条件（Qwen query tokens）未有效调制输出。
3. **与 prior collapse 一致**：dualmode `latent_nearest` unique ~7% 已很差；direct v0 更极端（unique 0.017%）。
4. **下一步（按优先级）**：
   - 推理：temperature/top-k、重复惩罚、max length 约束
   - 训练：property-aware loss、contrastive condition、更大 decoder / 更长训练
   - 数据：检查 tokenizer（vocab=47 过小？）、SMILES 长度分布、condition 对齐
   - 仍保留 hybrid 作 retrieval upper bound，但论文 claim 应基于 direct 或明确标注 retrieval-assisted

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
- `outputs/direct_smiles_denovo_2p7p_v0/benchmark_direct_smiles/benchmark_report.md`
- `outputs/direct_smiles_denovo_ood_v0/benchmark_direct_smiles/benchmark_report.md`
