# Unified SMILES Phase1 De Novo (direct_compat) v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed |
| **最后更新** | 2026-07-12 |
| **Jobs** | `17274038` (2p7p), `17288282` (OOD) |
| **Checkpoint** | Direct de novo 2p7p SFT warm-start（`direct_smiles_denovo_2p7p_v2` 线） |
| **Condition layout** | `direct_compat`（`<DE_NOVO>` + property program + frozen VLM features） |
| **解码** | sample ×128，`top_k=40`，`temperature=0.85` |

## 结论摘要

1. **In-distribution 2p7p 很强**：主 profile `direct_compat_with_image` 在 500 行 eval 上 **97.0%** 2p strict、**100%** validity，明显高于 SketchMol structured reference（80.4% 2p）。
2. **VLM / feature store 对 2p7p 几乎无差别**：`with_image` / `text_only` / `direct_feature_store` 均为 97.0%；仅 `property_program_only` 降至 90.4%（validity 97.8%）。
3. **OOD 明显更难**：1000 行 OOD eval 上 **74.3%** 2p strict、**45.0%** 7p strict（validity 100%），7p 仍低于 SketchMol reference（68.5%）。
4. **与 Table1 edit 对比**：de novo direct_compat 已可用；Table1 需专用 RL 训练（见 [`table1-edit-v1.md`](table1-edit-v1.md)）。

## 主结果：2p7p in-distribution（500 eval rows）

Profile：`direct_compat_with_image`。产物根目录：`outputs/unified_smiles_generator_phase1_direct_compat_v1/direct_compat_with_image/`

| 指标 | Unified | SketchMol ref |
| --- | ---: | ---: |
| validity all | **100%** | — |
| 2p strict | **97.0%** | 80.4% |

本地 benchmark 报告：
`.../benchmark_warmstart_beam/denovo_2p7p/benchmark_report.md`

## Condition / modality 消融（2p7p，500 rows）

| Profile | validity | 2p strict | 说明 |
| --- | ---: | ---: | --- |
| `direct_compat_with_image` | 100% | **97.0%** | 主配置 |
| `direct_compat_text_only` | 100% | **97.0%** | 无 query image |
| `direct_feature_store` | 100% | **97.0%** | direct 2p7p feature dirs |
| `property_program_only` | 97.8% | 90.4% | 无 frozen VLM features |

产物根目录：`outputs/unified_smiles_generator_phase1_direct_compat_v1/{profile}/`

## OOD de novo（1000 eval rows）

Profile：`direct_compat_with_image`。Job `17288282`。

| 指标 | Unified | SketchMol ref |
| --- | ---: | ---: |
| validity all | **100%** | — |
| 2p strict | **74.3%** | 80.4% |
| 7p strict | **45.0%** | 68.5% |

产物根目录：`outputs/unified_smiles_generator_phase1_ood_direct_compat_v1/`

本地 benchmark 报告：
`.../benchmark_warmstart_beam/denovo_ood/benchmark_report.md`

## 代码 / 脚本

| 脚本 | 用途 |
| --- | --- |
| `run_unified_direct_warmstart_diagnostics.sh` | Phase1 2p7p 四 profile 诊断 / 消融 |
| `submit_unified_direct_warmstart_diagnostics.sh` | Slurm 提交（`SUCC_UNIFIED_DIRECT_COMPAT_PHASE=phase1`） |
| `run_unified_phase1_ood_eval.sh` | 全量 OOD direct_compat eval |
| `submit_unified_phase1_ood_eval.sh` | Slurm 提交 OOD eval |

## 下一步

1. Fair v1（U0 eval / U1 de novo SFT）完成后与 Phase1 数字对齐对比。
2. OOD 7p 差距：考虑更多采样预算或 OOD 专用 SFT/RL。
3. Table1 edit：优先报 Direct RL 主表；Unified 需对齐 train/inference condition layout。
