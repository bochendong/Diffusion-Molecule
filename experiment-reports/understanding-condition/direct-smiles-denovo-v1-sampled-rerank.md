# SUCC Direct SMILES De Novo v1 Sampled Rerank

| 字段 | 值 |
| --- | --- |
| **状态** | **待运行** |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_benchmark.sh` / `submit_direct_smiles_denovo_ood_benchmark.sh` |
| **目的** | 修复 direct SMILES v0 的 autoregressive mode collapse，同时保持无 candidate-library retrieval |

## 背景

v0 在 teacher forcing 下 loss/perplexity 正常下降，但推理阶段严重塌缩：

| Benchmark | Strict | Validity | Unique valid |
| --- | ---: | ---: | ---: |
| 2p-7p v0 | 0.0007 | 1.000 | 1 / 6000 |
| OOD v0 | 0.000 | 0.094 | 1 / 1000 |

主要问题不是 evaluator 或 materializer，而是 direct decoder 在 autoregressive generation
阶段几乎只输出单一长链 SMILES。

## v1 改动

v1 不引入 molecule DB nearest search，不使用 `property_nearest`，也不使用
`latent_property_rerank`。它只在每个 prompt 下从 decoder 自己采样出的候选里选择最优 SMILES：

```text
condition prompt
  -> Qwen/HF VLM query tokens
  -> Transformer SMILES decoder
  -> N direct SMILES samples
  -> RDKit validity + absolute target property rerank
  -> generated_smiles
```

默认推理参数：

| 参数 | 默认 |
| --- | ---: |
| `num_samples` | 32 |
| `max_new_tokens` | 96 |
| `temperature` | 0.85 |
| `top_k` | 40 |
| `top_p` | 0.95 |
| `repetition_penalty` | 1.15 |
| `no_repeat_ngram_size` | 6 |
| `min_new_tokens` | 6 |

Rerank score 使用 evaluator 同一组 strict tolerance：

```text
score = 10 * valid + 100 * strict_fraction - 10 * normalized_property_distance
```

输出 CSV 会额外记录：

```text
direct_candidate_count
direct_unique_candidate_count
direct_valid_candidate_count
direct_unique_valid_candidate_count
direct_best_candidate_rank
direct_best_score
direct_best_strict_fraction
direct_best_property_distance
```

## 服务器命令

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_benchmark.sh

export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_benchmark.sh
```

默认输出：

```text
SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank/
SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v1_sampled_rerank/
```

## 接受标准

第一目标不是立刻超过 hybrid，而是验证 collapse 是否被打破：

| 指标 | 期望 |
| --- | --- |
| `unique_valid_smiles` | 明显高于 v0 的 1 |
| `validity` | OOD 明显高于 v0 的 0.094 |
| `strict` | 明显高于 v0 的 0 / 0.0007 |
| `direct_valid_candidate_count` | 证明 decoder 本身能采样出有效分子 |

如果 v1 仍然 strict 接近 0，但 candidate validity/uniqueness 上来，下一轮应转向
SELFIES/graph decoder 和 reward fine-tuning；如果 candidate 本身仍然无效或单一，
说明需要换 decoder/tokenization，而不是继续加 rerank。
