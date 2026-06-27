# External Multi-property Benchmark Port

| 字段 | 值 |
| --- | --- |
| **状态** | adapter v1 implemented |
| **最后更新** | 2026-06-26 |
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
```

默认是 pilot/diagnostic：

- `RUN_TRAIN=0`
- 使用现有 direct-SMILES checkpoint。
- `NUM_SAMPLES=20`，对齐 GeLLMO 论文推理设置。
- `DISABLE_PROPERTY_RERANK=1`，避免第一版变成 output-side rerank。
- 如果没有 generated-property CSV，报告会把 missing oracle properties 明确列出来。

## 服务器命令

最小 pilot：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_or_cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
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

## 接下来

1. 先用官方 HF test split 或本地转换后的 CSV 跑 pilot，确认行数、task coverage 和生成速度。
2. 跑一遍 generated molecules 的 ADMET/TDC oracle，回填 generated-property CSV。
3. 再决定是否把 source-conditioned external rows 接到 group RL。

## 外部来源

- GeLLMO official repo: https://github.com/ninglab/GeLLMO
- GeLLMO-C official repo: https://github.com/ninglab/GeLLMO-C
- GeLLMO ACL page: https://aclanthology.org/2025.acl-long.1225/
