# Understanding Rerank Experiment Notes

## Research Question

我们的核心问题不是单纯找到满足目标性质的分子，而是：

```text
given source molecule + multi-property edit instruction
  -> generate/select target molecule
  -> preserve the source scaffold as much as possible
  -> satisfy 2-7 active property constraints
```

因此，`strict success` 只能说明性质是否满足；`scaffold_match` 是同等关键的指标。
如果一个方法 strict 很高但 scaffold match 接近 0，它只能说明属性检索能力强，
不能回答 scaffold-preserving edit 这个 research question。

## Experiment Goal

这轮实验尝试把 Understanding-Condition stream 接到 SketchMolBenchmark 的思路里，
先做一个轻量的 rerank 原型：

```text
property-conditioned candidate stream
  -> top-K candidate molecules
  -> Understanding-Condition feature rerank
  -> SketchMol-style strict success evaluation
```

这里的 property-conditioned candidate stream 目前还不是真正的 SketchMol diffusion
采样输出，而是用 train molecule pool 模拟候选集合。所以这轮实验只能验证
understanding stream 是否能改善候选选择，不能作为最终生成模型结果。

## Implemented Methods

代码入口：

```text
SketchMol-MultiProperty-EditDataset/scripts/benchmark_multiproperty_retrieval.py
```

新增方法：

```text
global_property_vlm_rerank
scaffold_property_vlm_rerank
```

新增参数：

```text
--rerank-candidates
--rerank-property-weight
```

`global_property_vlm_rerank` 的流程是：

```text
1. 从全局 train candidate pool 里按 active property target 选 top-K。
2. 用 Understanding-Condition dual connector feature 计算 query-candidate 相似度。
3. 用 feature_score - property_weight * property_error 做最终 rerank。
4. 选 top-1 后按 SketchMol strict tolerance 评估。
```

`scaffold_property_vlm_rerank` 的流程相同，但候选池优先限制到同 scaffold。
如果 eval scaffold 在 train pool 里不存在，则 fallback 到 global pool。

## Feature Stream

本轮使用的 understanding feature 是 dual-stream connector 输出：

```text
SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_dual_connector_50k_e5/
```

它由 frozen Qwen2.5-VL pooled feature 经过 MLP connector 训练得到，并拼入 source
molecule feature stream。训练目标是显式预测 multi-property condition vector，
包括 target values、active masks 和 directions。

## Main Smoke Results

评估设置：

```text
condition rows:
  SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv

eval split:
  eval

eval rows:
  200 per property count, 1200 total

eval target candidates:
  excluded from retrieval pool

rerank candidates:
  64

rerank property weight:
  0.5
```

结果文件：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hybrid_rerank_w05_smoke/benchmark_report.md
```

关键 strict success：

```text
source_identity:
  2p 0.490
  3p 0.200
  4p 0.075
  5p 0.015
  6p 0.000
  7p 0.000
  scaffold all 1.000

vlm_feature_retrieval:
  2p 0.305
  3p 0.210
  4p 0.105
  5p 0.035
  6p 0.020
  7p 0.050
  scaffold all 0.000

global_property_vlm_rerank, property_weight=0.5:
  2p 1.000
  3p 0.975
  4p 0.890
  5p 0.770
  6p 0.630
  7p 0.465
  scaffold all 0.001

SketchMol structured reference:
  2p 0.804
  3p 0.768
  4p 0.736
  5p 0.716
  6p 0.678
  7p 0.685
```

Scaffold-filtered smoke result：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hybrid_scaffold_rerank_smoke/benchmark_report.md
```

关键 strict success：

```text
scaffold_property_retrieval:
  2p 1.000
  3p 1.000
  4p 0.955
  5p 0.905
  6p 0.840
  7p 0.710
  scaffold all 0.005

scaffold_property_vlm_rerank, property_weight=0.0:
  2p 0.975
  3p 0.900
  4p 0.725
  5p 0.530
  6p 0.430
  7p 0.260
  scaffold all 0.005
```

## Full HF VLM Run: 15639850

完整 workflow 日志：

```text
SketchMol-Understanding-Condition/logs/succ-hf-vlm-15639850.log
```

本次完整运行成功完成：

```text
HF VLM feature export:
  rows 300000
  pooled shape 300000 x 3584
  query tokens shape 300000 x 32 x 256

benchmark eval rows:
  22452

candidate source:
  molecule_database

candidate molecules:
  87107

scaffold fallback mode:
  source_identity
```

Frozen HF VLM benchmark：

```text
report:
  SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm/benchmark_report.md

joint success all:
  source_identity                  0.155
  global_property_retrieval        0.003
  scaffold_property_retrieval      0.194
  vlm_feature_retrieval            0.000
  vlm_scaffold_feature_retrieval   0.155
  global_property_vlm_rerank       0.001
  scaffold_property_vlm_rerank     0.193
  target_oracle                    1.000
```

`global_property_vlm_rerank` 的 strict property success 很高：

```text
2p 0.999
3p 0.986
4p 0.922
5p 0.806
6p 0.687
7p 0.535
```

但它的 scaffold all 只有 0.001，joint all 也只有 0.001。因此它仍然只是
property retrieval / selector sanity check，不是 scaffold-preserving edit。

Connector benchmark：

```text
report:
  SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm_edit_connector/benchmark_report.md

joint success all:
  source_identity                       0.155
  global_property_retrieval             0.003
  scaffold_property_retrieval           0.194
  edit_latent_global_retrieval          0.000
  edit_latent_scaffold_retrieval        0.188
  edit_latent_scaffold_source_rerank    0.187
  target_oracle                         1.000
```

Connector training itself was stable:

```text
train rows 50000
eval rows 59880
epochs 5
eval MSE 0.323 -> 0.292
```

However, the connector did not beat the scaffold-property retrieval baseline on
the actual joint edit metric.

The key diagnostic is fallback rate:

```text
scaffold_property_retrieval source fallback:           0.841
vlm_scaffold_feature_retrieval source fallback:        0.996
scaffold_property_vlm_rerank source fallback:          0.841
edit_latent_scaffold_retrieval source fallback:        0.841
edit_latent_scaffold_source_rerank source fallback:    0.841
```

This means most eval scaffolds do not have same-scaffold candidates in the
retrieval candidate library. The main bottleneck is therefore not the reranker;
it is the candidate stream. To improve scaffold-preserving edit, the next method
needs a source-conditioned visual candidate generator, rather than global
database retrieval.

## Interpretation

The good news:

```text
Understanding rerank is useful as a selector.
```

Pure VLM feature retrieval is weak: overall strict success is about 0.121 in the
200-per-count smoke. After adding a property-conditioned top-K stage and a
property penalty in reranking, the hybrid method reaches 0.788 overall strict
success. This means the understanding feature is not useless; it can help choose
among candidates once the candidate stream is already near the target property
region.

The important negative result:

```text
The current global rerank does not solve scaffold-preserving edit.
```

`global_property_vlm_rerank` has good strict success but `scaffold all` is only
0.001. This is unacceptable for our research question. It is closer to
"property retrieval from a database" than "edit this source molecule while
preserving scaffold".

The scaffold-filtered run also shows a limitation:

```text
Our train candidate pool rarely contains the same eval scaffold.
```

Even methods named `scaffold_property_*` only reach scaffold match around 0.005
because most eval scaffolds are unseen in the train candidate pool and the script
falls back to global retrieval. Therefore this retrieval-style prototype is not
a fair substitute for a scaffold-preserving generator.

## Current Conclusion

This experiment supports a narrower claim:

```text
An Understanding-Condition stream can improve candidate selection after a
property-conditioned candidate generator has produced plausible molecules.
```

It does not yet support the main research claim:

```text
Understanding-Condition improves scaffold-preserving multi-property molecular edit.
```

To answer the actual research question, the next experiment must provide a
candidate stream that already preserves source scaffold with high probability.
Then understanding rerank should choose among those scaffold-preserving samples.

## Recommended Next Experiment

Replace the current train-pool candidate stream with a real or scaffold-preserving
candidate stream:

```text
source molecule + property condition
  -> SketchMol / diffusion / scaffold-constrained generator samples K candidates
  -> parse candidates to SMILES
  -> compute scaffold and properties
  -> rerank with Understanding-Condition feature + property penalty
  -> evaluate strict success and scaffold match jointly
```

The target result should be reported with both metrics:

```text
strict success by property count:
  2p, 3p, 4p, 5p, 6p, 7p

scaffold match:
  per property count and overall

joint success:
  strict success AND scaffold match
```

The most important number for the paper should be joint success, not strict
success alone.

## Commands Used

Global hybrid rerank with property penalty:

```bash
/home/bdong/.venvs/molscribe_overlay/bin/python \
  SketchMol-MultiProperty-EditDataset/scripts/benchmark_multiproperty_retrieval.py \
  --condition-rows-csv SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv \
  --output-dir SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hybrid_rerank_w05_smoke \
  --methods source_identity,global_property_retrieval,vlm_feature_retrieval,global_property_vlm_rerank,target_oracle \
  --condition-features-dir SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_dual_connector_50k_e5 \
  --condition-feature-array pooled \
  --condition-feature-variant full \
  --max-eval-per-property-count 200 \
  --max-feature-candidates 5000 \
  --max-global-candidates 20000 \
  --rerank-candidates 64 \
  --rerank-property-weight 0.5
```

Scaffold-filtered hybrid rerank:

```bash
/home/bdong/.venvs/molscribe_overlay/bin/python \
  SketchMol-MultiProperty-EditDataset/scripts/benchmark_multiproperty_retrieval.py \
  --condition-rows-csv SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv \
  --output-dir SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hybrid_scaffold_rerank_smoke \
  --methods source_identity,scaffold_property_retrieval,vlm_scaffold_feature_retrieval,scaffold_property_vlm_rerank,target_oracle \
  --condition-features-dir SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_dual_connector_50k_e5 \
  --condition-feature-array pooled \
  --condition-feature-variant full \
  --max-eval-per-property-count 200 \
  --max-feature-candidates 5000 \
  --max-global-candidates 20000 \
  --rerank-candidates 64
```
