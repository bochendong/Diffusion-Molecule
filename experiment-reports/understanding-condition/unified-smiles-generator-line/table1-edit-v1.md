# Unified SMILES Table1 Edit v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed (metrics) |
| **最后更新** | 2026-07-12 |
| **Jobs** | `17431689`/`17477533`/`17483305` (train), `17428762` (metrics resume), `17497827` (unified eval) |

## 结论摘要

1. **Metrics/oracle 修复有效**：Table1 TDC oracle 预检 + `module load rdkit` 后 10/10 task 均可 measured。
2. **Table1 edit 需要专用训练**：de novo SFT ckpt 仅 **8.6%** mean Acc@0.15 @n20。
3. **Direct Table1 group RL 有效**：同 pipeline 下 **57.3%** @n20 / **80.6%** @n256 Acc@0.15。
4. **Unified generator 仍有 train/inference gap**：加载同一 RL ckpt + `direct_edit_compat` 仅 **26.9%** @n20 / **24.7%** @n256（Direct 原生 eval 约一半）。

## Direct Table1 group RL（job `17483305`）

Checkpoint：`outputs/direct_smiles_moledit_table1_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt`

Warm-start：de novo 2p7p SFT（group RL ckpt 不存在）。Condition：`append_source_property_program`。

| Budget | mean Acc@0.15 | mean Acc@0.65 | Validity |
| ---: | ---: | ---: | ---: |
| n=20 | **57.3%** | 0% | 100% |
| n=256 | **80.6%** | 0% | 100% |

参考：Univideo fair similarity rerank n=20 **56.9%** Acc@0.15。

产物：
- `.../benchmark_group_rl/moledit_table_metrics_n20/moledit_table_summary.md`
- `.../benchmark_group_rl/moledit_table_metrics_n256/moledit_table_summary.md`

## Unified Phase1 Table1（de novo ckpt, job `17301127` + resume `17428762`）

Layout：`direct_edit_compat`。Checkpoint：de novo 2p7p SFT fallback。

| Budget | mean Acc@0.15 | source copy (selected) |
| ---: | ---: | ---: |
| n=20 | **8.6%** | 429/1000 (43%) |
| n=256 | 7.1% | — |

产物：`outputs/unified_smiles_generator_phase1_table1_direct_compat_v1/`

Selection 消融（同 predictions，CPU）：
- `unified_score` + 过滤 source copy → **12.8%**
- `rank` + 过滤 source copy → **22.7%**

## Unified + Table1 RL ckpt（job `17497827`）

Checkpoint：同上 Table1 RL ckpt。Layout：`direct_edit_compat`。

| Budget | mean Acc@0.15 | vs Direct same ckpt |
| ---: | ---: | ---: |
| n=20 | **26.9%** | −30.4pp |
| n=256 | **24.7%** | −55.9pp |

产物：`outputs/unified_smiles_generator_phase1_table1_table1_rl_v1/`

## 代码/脚本

- Oracle 预检：`experiments/unified_smiles_generator/ensure_moledit_oracle_env.sh`
- Metrics resume：`run_unified_phase1_table1_benchmark_resume.sh`
- Feature export fix：`scripts/export_condition_features.py`（Table1 rows 无 `variant` 列）
- RL 训练 fallback ckpt：`scripts/run_direct_smiles_moledit_table1_group_rl.sh`

## 相关报告

- De novo Phase1（direct_compat）：[`phase1-denovo-direct-compat-v1.md`](phase1-denovo-direct-compat-v1.md) — 2p7p **97%** 2p strict，OOD **74%** / **45%**。

## 下一步

1. 主表 Table1 数字优先报 **Direct Table1 RL**（57%/81%）。
2. Unified 侧对齐训练 condition（`append_source_property_program`）或做 Unified-native Table1 SFT/RL。
3. Selection：过滤 source-copy + 试 `rank` selection。
