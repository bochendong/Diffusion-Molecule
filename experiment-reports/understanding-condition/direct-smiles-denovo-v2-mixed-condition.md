# SUCC Direct SMILES De Novo v2 Mixed Condition

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-21） |
| **最后更新** | 2026-06-21（2p7p + OOD v2 首轮训练） |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_v2_benchmark.sh` / `submit_direct_smiles_denovo_ood_v2_benchmark.sh` |
| **目的** | 在 direct SMILES decoder 上引入 mixed conditioning + property-count curriculum，提升高 property count（6p/7p）strict |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16472651` | `succ-direct-smiles-2p7p-v2`（首轮；epoch 9/12 时 Node Fail） | 17m | **Node Fail** | `direct_smiles_denovo_2p7p_v2_mixed_condition/`（`latest_checkpoint.pt` epoch 9） |
| `16472652` | `succ-direct-smiles-ood-v2` | ~1h28m | 0 | `direct_smiles_denovo_ood_v2_mixed_condition/` |
| `16477041` | `succ-direct-smiles-2p7p-v2`（从 epoch 9 resume） | 3h17m | 0 | 同上（补完 epoch 10–12 + n=64 评测） |

## v2 方法（相对 v1）

| 变量 | v1 | v2 |
| --- | --- | --- |
| condition mixing | `features_only` | **`append_property_program`** |
| property-count curriculum | 关 | **采样 + loss 加权均开**（power=1.0，baseline=2.0） |
| 推理 num_samples | 32（默认）/ 64–256 缩放 | **64**（首轮） |
| 训练 | 12 epoch SFT | 12 epoch SFT（同 lr/batch） |

仍无 molecule DB retrieval；推理仍为 sampled rerank（temperature 0.85、top-k 40、top-p 0.95 等，与 v1 一致）。

## 2p7p 结果（n=64，job `16477041`，resume 自 `16472651`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid | mean valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.924** | **0.830** | **0.747** | **0.611** | **0.523** | **0.449** | **0.681** | **0.986** | **5914 / 6000** | **18.7** |

训练末 epoch（12）：`eval_loss` **0.806**，`eval_perplexity` **2.24**；`mean_property_count` **5.16**（curriculum 生效）。

### vs v1（同 n=64 或 best n=256）

| 对比 | overall strict | 6p strict | 7p strict |
| --- | ---: | ---: | ---: |
| v1 n=64（推理-only） | 0.354 | 0.128 | 0.084 |
| v1 n=256（推理-only） | 0.562 | 0.314 | 0.258 |
| **v2 n=64（重训）** | **0.681** | **0.523** | **0.449** |

v2 在 **仅 n=64 推理** 下 overall **+11.9pp** vs v1 best n=256；7p **+19.1pp**（0.258→0.449）。2p/3p/4p 亦全面超过 SketchMol ref。

## OOD 结果（n=64，job `16472652`）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict | mean valid |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.531** | **0.888** | **881 / 1000** | **0.330** | **0.510** | **5.1** |

| vs v1 OOD | overall strict | 7p bucket strict | validity |
| --- | ---: | ---: | ---: |
| v1 n=64 | 0.545 | 0.070 | 0.998 |
| v1 n=256 | 0.801 | 0.190 | 1.000 |
| **v2 n=64** | **0.531** | **0.510** | **0.888** |

OOD overall 略低于 v1 n=64（-1.4pp），但 **7p bucket 大幅提升**（7.0%→51.0%）；validity 降至 88.8%，2p bucket 仅 33.0%。

## 与 v1 / hybrid 对照

| 方法 | 2p7p strict | OOD strict | retrieval? |
| --- | ---: | ---: | --- |
| direct v1 n=256 | 0.562 | 0.801 | 否 |
| direct v1 DPO n=256 | 0.521 | — | 否 |
| **direct v2 n=64** | **0.681** | **0.531** | 否 |
| hybrid latent_property_rerank | 0.970 | 0.985 | 是 |
| SketchMol ref（2p） | 0.804 | — | — |

## 结论

1. **mixed condition + curriculum 有效**：2p7p 在 n=64 下已达 **68.1%**，显著优于 v1 全部采样规模；高 property count 是主要增益来源（6p/7p +19–21pp vs v1 n=256）。
2. **OOD 分化**：7p rare combo 从 v1 的 ~7–19% 跳到 **51%**；但 overall 与 2p bucket 未同步提升，validity 88.8% 需排查。
3. **Node Fail 可恢复**：`16472651` epoch 9 中断后，`16477041` resume 3h17m 补完；feature/数据导出无需重跑。
4. **下一步**：2p7p 试 n=128/256 推理-only（复用 v2 ckpt）；OOD 提 validity / 2p；与 preference DPO 结合（v2 conditioning 已兼容 builder）。

## 提交命令（归档）

```bash
# 2p7p v2（默认 mixed condition + curriculum，n=64 推理）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_benchmark.sh

# OOD v2
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_benchmark.sh

# 2p7p resume（Node Fail 后）
SUCC_DIRECT_DENOVO_RESUME_CHECKPOINT=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/direct_smiles_model/latest_checkpoint.pt \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_benchmark.sh
```

产出：
- `outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/`
- `outputs/direct_smiles_denovo_ood_v2_mixed_condition/`
