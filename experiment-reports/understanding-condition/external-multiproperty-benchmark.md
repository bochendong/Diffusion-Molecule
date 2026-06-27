# External Multi-property Benchmark Port

| 字段 | 值 |
| --- | --- |
| **状态** | **MuMO source-edit SFT+RL 完成**；agentic revise 入口已加入；oracle CSV 待回填 |
| **最后更新** | 2026-06-27（agentic revise v1） |
| **代码范围** | `SketchMol-Understanding-Condition` |
| **目标** | 把 SUCC direct-SMILES/LLM-conditioned 线接到 GeLLMO/MuMOInstruct 和 GeLLMO-C/C-MuMOInstruct 风格的 source-conditioned multi-property IND/OOD benchmark |

## 关键判断

MuMO / C-MuMO 不是 zero-source absolute target design。它们的任务形式是：

1. 输入 source molecule。
2. 按任务要求 increase / decrease 多个性质。
3. 保持结构变化尽量小。
4. 对 5 IND + 5 OOD task 分别统计 multi-property success。

所以这一版不应该复用 2p/7p 的 zero-source CSV，而应该走 source-conditioned edit CSV。

## 已实现

新增 exporter：

```bash
SketchMol-Understanding-Condition/scripts/export_external_multiproperty_benchmark_rows.py
```

覆盖 task specs：

| Suite | IND tasks | OOD tasks |
| --- | --- | --- |
| MuMO | BDP, BDQ, BPQ, DPQ, BDPQ | MPQ, BDMQ, BHMQ, BMPQ, HMPQ |
| C-MuMO | BPQ, ELQ, ACEP, BDPQ, DHMQ | CDE, ABMP, BCMQ, BDEQ, HLMPQ |

新增 evaluator：

```bash
SketchMol-Understanding-Condition/scripts/evaluate_external_multiproperty_predictions.py
```

它支持两层评估：

1. 本地可算的 proxy properties：QED、LogP、SA、pLogP proxy。
2. 外部 generated-property CSV：用于 BBBP、HIA、mutagenicity、hERG、DILI、PAMPA 等 ADMET/oracle 性质。

新增 server 入口：

```bash
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_benchmark.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_group_rl.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_source_edit_sft.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh
SketchMol-Understanding-Condition/scripts/build_external_agentic_revise_predictions.py
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_agentic_revise.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
SketchMol-Understanding-Condition/scripts/run_external_multiproperty_source_copy_sanity.sh
SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh
```

one-shot 默认是 pilot/diagnostic：

- `RUN_TRAIN=0`
- 使用现有 direct-SMILES checkpoint。
- `NUM_SAMPLES=20`，对齐 GeLLMO 论文推理设置。
- `DISABLE_PROPERTY_RERANK=1`，避免第一版变成 output-side rerank。
- 如果没有 generated-property CSV，报告会把 missing oracle properties 明确列出来。

group-RL 入口默认：

- 从当前最强的 direct-SMILES group-RL checkpoint warm start。
- 使用 source-conditioned external rows 训练；`SFT_WEIGHT=0.15`，避免 target placeholder 过度鼓励复制 source。
- reward 仍使用本地可算 proxy strict/distance，同时新增可关闭的 source Tanimoto reward（默认权重 `0.5`，阈值 `0.4`）。
- benchmark 阶段默认 `NUM_SAMPLES=20`、`DISABLE_PROPERTY_RERANK=1`。
- ADMET/oracle 性质仍需要 generated-property CSV 才能公平报告；没有 oracle CSV 时，group-RL 只优化 proxy subset + source conservation。

## Slurm 运行记录

| Job | 名称 | 数据 | 输出 |
| --- | --- | --- | --- |
| `16774795` | `succ-external-mumo-group-rl` | MuMO train/test（HuggingFace 官方 JSON） | `direct_smiles_external_mumo_group_rl_v1/` |
| `16779361` | `succ-external-mumo-one-shot` | MuMO test（one-shot baseline） | `direct_smiles_external_mumo_one_shot_v1/` |
| `16779362` | `succ-external-mumo-source-copy` | MuMO test（source-copy sanity） | `external_mumo_source_copy_sanity/` |
| `16800837` | `succ-external-mumo-source-edit-sft` | MuMO train/test（source-edit SFT warm-start） | `direct_smiles_external_mumo_source_edit_sft_v1/` |
| `16806903` | `succ-external-mumo-source-edit-rl` | MuMO train/test（source-edit SFT → group-RL） | `direct_smiles_external_mumo_source_edit_sft_group_rl_v1/` |

集群数据路径（已下载）：

- train：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json`（828k rows）
- test：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json`（44k rows）

## MuMO group-RL pilot（job `16774795`）

入口：`submit_direct_smiles_external_multiproperty_group_rl.sh`（`16742f8`）。MuMO official train/test；10 tasks × 200 train rows（2000 total）；eval 1992 rows（HMPQ 192）；warm-start 2p7p group-RL ckpt；1 epoch group-relative RL（`SFT_WEIGHT=0.15`，`source_similarity_weight=0.5`）；benchmark n=20，`DISABLE_PROPERTY_RERANK=1`；**无 generated-property oracle CSV**。

训练：`eval_mean_reward` **-0.337**；`mean_reward` **-0.660**；`sft_loss` **1.964**。

评测（proxy-only；strict 需 oracle CSV 才能算）：

| Split | Valid | Proxy eval prop frac | Strict | Sim ≥0.4 |
| --- | ---: | ---: | ---: | ---: |
| IND（5 tasks） | **83.6%** | **42.2%** | 0 | 0 |
| OOD（5 tasks） | **86.8%** | **39.3%** | 0 | 0 |
| **all** | **85.1%** | **40.7%** | **0** | **0** |

分 task proxy eval prop frac（最高 / 最低）：

| Task | Split | Valid | Proxy eval prop frac |
| --- | --- | ---: | ---: |
| BPQ | ind | 93.5% | **62.3%** |
| MPQ | ood | 95.5% | **63.7%** |
| BDQ | ind | 69.5% | **23.2%** |
| BHMQ | ood | 71.0% | **17.8%** |

**结论**（初版；见下方 diagnostics 更新）：
1. **端到端 pipeline 跑通**：export → feature → group-RL → benchmark；10-task coverage OK。
2. **无 oracle CSV 时 strict=0 是预期行为**（BBBP/DRD2/HIA/mutagenicity 缺 external scorer）；proxy subset `eval prop frac` **~41%** 仅作 plumbing 诊断。
3. **source similarity 全为 0**（Sim ≥0.4）：初判为 source-edit 能力问题；已由 source-copy sanity 排除 eval 链路 bug。

## MuMO diagnostics（jobs `16779361` / `16779362`）

同口径 eval：MuMO test，10 tasks × 200，n=20，无 rerank，无 oracle CSV。

| Job | 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| `16779362` | **source-copy sanity** | **100%** | 46.7% | **100%** | 0 |
| `16779361` | **one-shot baseline** | **90.7%** | **42.5%** | **0** | 0 |
| `16774795` | group-RL | 85.1% | 40.7% | 0 | 0 |

**诊断结论**：
1. **source-copy Sim@0.4 = 100%** → `source_smiles` 字段与 Tanimoto evaluator **正常**；group-RL / one-shot 的 Sim=0 **不是数据或评估链路 bug**。
2. **one-shot proxy 42.5% ≈ group-RL 40.7%** → 短 RL 未带来增益，validity 还略降（90.7% vs 85.1%）；瓶颈在 **de novo ckpt 不具备 source-conditioned edit 能力**，而非 RL 超参。
3. **strict 全为 0**（三方一致）→ 仍需 ADMET/TDC generated-property CSV 才能报 official strict；source-copy strict=0 也符合预期（复制 source 不改善性质）。
4. **下一步**：先跑 source-edit SFT warm-start，再用该 checkpoint 接 source-edit group-RL；ADMET oracle 回填；C-MuMO 复跑。

## Source-edit repair v1

当前 direct-SMILES checkpoint 是 zero-source de novo 训练出来的；MuMO/C-MuMO 则要求输入 source molecule 后做小改动。因此这版先修生成主干，而不是继续调 rerank：

1. 新增 `append_source_property_program` condition mode：在原 MLLM feature / fallback numeric feature 后，显式追加 source-SMILES token features，再追加 property program tokens。模型结构不变，旧 checkpoint 可 warm start。
2. 新增 source-edit SFT warm-start：默认从最强 2p7p group-RL checkpoint 起步，使用 MuMO train/test rows，关闭 property rerank，先把 Sim≥0.4 拉回来。
3. 新增 SFT checkpoint 上的 source-edit group-RL：使用同一 source-aware condition mode，加大 source-similarity reward，仍关闭 property rerank，目标是同时提升 source similarity 与 proxy property success。

这条线和 de novo 2p-7p/OOD 主线互不覆盖；Table1 edit 主链也不改。

## MuMO source-edit SFT（job `16800837`）

入口：`submit_direct_smiles_external_mumo_source_edit_sft.sh`（`e7b755e`）。从 2p7p group-RL ckpt warm-start；`append_source_property_program` + `reset_training_state=1`；MuMO train 2000 / eval 1992 rows；1 epoch SFT；benchmark n=20，无 rerank，无 oracle CSV。输出 `direct_smiles_external_mumo_source_edit_sft_v1/`。

训练：1 epoch，`eval_loss` **1.275**；`eval_perplexity` **3.58**；`reset_training_state` **true**。

评测（同 diagnostics 口径）：

| 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | ---: | ---: | ---: | ---: |
| one-shot（`16779361`） | 90.7% | 42.5% | 0 | 0 |
| group-RL de novo（`16774795`） | 85.1% | 40.7% | 0 | 0 |
| **source-edit SFT（`16800837`）** | **97.3%** | **45.4%** | **0** | 0 |
| source-copy sanity（`16779362`） | 100% | 46.7% | **100%** | 0 |

**结论**：source-aware conditioning + 1 epoch MuMO SFT **提升了 validity（+6.6pp vs one-shot）和 proxy（+2.9pp）**，但 **Sim@0.4 仍为 0**——模型仍未学会保持 source scaffold。1 epoch / 2000 rows 可能不够；下一步接 **source-edit group-RL**（加大 `source_similarity_weight=2.0`）验证 RL 能否拉回 Sim。

## MuMO source-edit group-RL（job `16806903`）

入口：`submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh`。从 source-edit SFT ckpt（`16800837`）warm-start；`append_source_property_program`；MuMO train 2000 / eval 1992 rows；1 epoch group-relative RL（`SFT_WEIGHT=0.5`，`source_similarity_weight=2.0`，`source_similarity_threshold=0.4`）；benchmark n=20，无 rerank，无 oracle CSV。输出 `direct_smiles_external_mumo_source_edit_sft_group_rl_v1/`。

训练：`eval_mean_reward` **-0.791**；`mean_reward` **-0.874**；`sft_loss` **0.992**；`pg_loss` **-0.058**。

评测（同 diagnostics 口径）：

| 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | ---: | ---: | ---: | ---: |
| one-shot（`16779361`） | 90.7% | 42.5% | 0 | 0 |
| group-RL de novo（`16774795`） | 85.1% | 40.7% | 0 | 0 |
| source-edit SFT（`16800837`） | 97.3% | 45.4% | 0 | 0 |
| **source-edit group-RL（`16806903`）** | **96.4%** | **45.0%** | **0** | **0** |
| source-copy sanity（`16779362`） | 100% | 46.7% | **100%** | 0 |

分 split：

| Split | Valid | Proxy eval prop frac | Sim ≥0.4 |
| --- | ---: | ---: | ---: |
| IND（5 tasks） | 95.1% | 41.3% | 0 |
| OOD（5 tasks） | 97.7% | 48.9% | 0 |

**结论**：在 SFT 基础上加大 source-similarity reward（权重 2.0）**未能拉回 Sim@0.4**；validity 略降（-0.9pp），proxy 基本持平（-0.4pp）。说明瓶颈不在 RL 超参，而在 **source-edit 监督/数据量不足**（1 epoch × 2000 rows）或 **生成范式**（direct SMILES 从零解码 vs 显式 edit/revise）。下一步优先：**加长 SFT epoch / 扩大 train rows**、ADMET oracle CSV 回填 strict、或接 **agentic revise loop**。

## MuMO agentic revise v1（待跑）

入口：`submit_direct_smiles_external_mumo_agentic_revise.sh`。流程：

1. 复用 source-edit SFT checkpoint 生成 direct proposal（`append_source_property_program`，n=20，无 property rerank）。
2. 对每个 source molecule 执行 source-preserving local edit actions（add/replace/remove terminal atoms），形成 source-neighbor candidate pool。
3. 按本地可算性质成功、source Tanimoto、property distance 选择 revised molecule。
4. 用同一个 external evaluator 输出 Valid / Sim≥0.4 / Prop all / Strict。

这条是 **agentic/assisted source-edit**，不与 one-shot direct generation 混报；目标是先验证显式 revise 能不能把 Sim 从 0 拉回来。

## 服务器命令

MuMO one-shot baseline（同口径对照，推荐先跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh
```

MuMO source-copy sanity（不算模型结果，只验证 source similarity / evaluator）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh
```

最小 one-shot pilot（自定义数据路径）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_or_cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

group-RL 公平版（同一个官方文件里含 train/test split）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_FILE=/path/to/mumo_or_cmumo_all_splits.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

group-RL 公平版（train/test 是两个独立文件）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_group_rl_v1 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

MuMO source-edit SFT warm-start（推荐下一条先跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh
```

MuMO source-edit SFT + group-RL（等上一条完成后跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh
```

MuMO agentic revise（下一条推荐跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
```

只跑 MuMO OOD：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=ood \
SUCC_EXTERNAL_MULTIPROP_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/external_mumo_ood_pilot \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

如果已经有 ADMET/TDC generated-property CSV：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=/path/to/generated_admet_props.csv \
SUCC_EXTERNAL_MULTIPROP_SUITE=cmumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/external_cmumo_pilot \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

## 当前限制

这版是 benchmark port，不是最终 claim。

1. BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等性质需要外部 generated-property CSV 才能公平评估。
2. `plogp` 当前给 SUCC numeric token 的本地 proxy 是 LogP；正式报告需要用外部/官方 pLogP scorer。
3. direct-SMILES 现有 checkpoint 主要是 zero-source de novo 训练出来的。source-conditioned external benchmark 最终应该接一版 source-edit SFT/RL 或 agentic revise loop。
4. 当前 external group-RL 的训练 reward 只能直接优化本地 proxy subset（QED / pLogP proxy / SA when available）和 source similarity；BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等 oracle 性质需要后续 predictor-in-loop / agentic oracle loop。

## 接下来

1. ~~跑 external group-RL P1 pilot~~ → **`16774795` 完成**；proxy **40.7%**，Sim=0。
2. ~~one-shot baseline~~ → **`16779361` 完成**；proxy **42.5%**，Sim=0；≈ group-RL，RL 无增益。
3. ~~source-copy sanity~~ → **`16779362` 完成**；Sim@0.4 **100%**；排除 eval 链路 bug。
4. ~~**source-edit SFT warm-start**~~ → **`16800837` 完成**；validity **97.3%**，proxy **45.4%**，Sim 仍 **0**。
5. ~~**source-edit group-RL**~~ → **`16806903` 完成**；validity **96.4%**，proxy **45.0%**，Sim 仍 **0**；RL 无 Sim 增益。
6. **agentic revise v1** → 入口已加入；下一步跑 `direct_smiles_external_mumo_agentic_revise_v1/`，检查 Sim@0.4 是否从 0 拉回。
7. **加长 source-edit SFT**（更多 epoch / rows）；ADMET/TDC generated-property CSV 回填 strict；C-MuMO 复跑。

## 外部来源

- GeLLMO official repo: https://github.com/ninglab/GeLLMO
- GeLLMO-C official repo: https://github.com/ninglab/GeLLMO-C
- GeLLMO ACL page: https://aclanthology.org/2025.acl-long.1225/
