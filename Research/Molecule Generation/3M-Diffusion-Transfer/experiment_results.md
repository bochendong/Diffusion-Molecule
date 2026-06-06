# Experiment Results

## succ-hf-vlm-15652638

日志：

```text
SketchMol-Understanding-Condition/logs/succ-hf-vlm-15652638.log
```

这次 HF VLM workflow 完整跑完：

```text
HF VLM feature export:
  rows: 300000
  pooled: 300000 x 3584
  query_tokens: 300000 x 32 x 256

connector:
  train rows: 50000
  eval rows: 59880
  eval MSE: 0.323 -> 0.292
```

Frozen VLM benchmark：

```text
report:
  SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm/benchmark_report.md

global_property_vlm_rerank strict property success:
  2p 0.999
  3p 0.986
  4p 0.922
  5p 0.806
  6p 0.687
  7p 0.535

global_property_vlm_rerank source-preserving metrics:
  scaffold joint all: 0.001
  strict@Tanimoto>=0.4: 0.009
  strict@Tanimoto>=0.6: 0.001
  strict@Tanimoto>=0.8: 0.000
```

Connector benchmark：

```text
report:
  SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm_edit_connector/benchmark_report.md

scaffold_property_retrieval:
  joint all: 0.194
  strict@0.4: 0.181
  strict@0.6: 0.165
  strict@0.8: 0.135

edit_latent_scaffold_retrieval:
  joint all: 0.188
  strict@0.4: 0.178
  strict@0.6: 0.163
  strict@0.8: 0.136

edit_latent_scaffold_source_rerank:
  joint all: 0.187
  strict@0.4: 0.178
  strict@0.6: 0.164
  strict@0.8: 0.137
```

结论：

```text
VLM/connector 链路可跑通，但没有提升 source-preserving edit。
global_property_vlm_rerank 的 property success 很高，但 source preservation 很差；
edit-latent scaffold rerank 没超过 scaffold_property_retrieval baseline。
```

## succ-unified-gen-15654655

日志：

```text
SketchMol-Understanding-Condition/logs/succ-unified-gen-15654655.log
```

这次 unified Understanding + latent diffusion smoke 完整跑完：

```text
dataset rows: 60266
train rows: 45545
eval rows: 14721
description_pretrain rows: 10266
edit_generation rows: 50000
unique source SMILES: 3197
unique target SMILES: 13461
```

训练曲线：

```text
alignment pretraining loss:
  2.170 -> 0.499

edit condition token loss:
  2.865 -> 1.854

latent diffusion loss:
  0.997 -> 0.923
```

产物：

```text
SketchMol-Understanding-Condition/outputs/unified_generation_3m_edit_v1/
  dataset/summary.json
  alignment/alignment_model.pt
  edit_condition_tokens/edit_condition_connector.pt
  edit_condition_tokens/query_tokens.npy
  edit_condition_tokens/pooled.npy
  latent_diffusion/latent_diffusion_generation.pt
```

后来补跑的 latent-space smoke eval：

```text
eval:
  SketchMol-Understanding-Condition/outputs/unified_generation_3m_edit_v1/eval_latent_smoke/metrics.json

rows: 200
sample steps: 20

generated -> target fingerprint cosine: 0.105
generated -> source fingerprint cosine: 0.106
source -> target fingerprint cosine baseline: 0.824

generated target property MAE: 41.1
source target property MAE baseline: 7.3
```

结论：

```text
Unified pipeline 成功打通了 dataset -> alignment -> edit condition tokens ->
latent diffusion 的训练链路，但当前 latent diffusion 采样还没有学到有效的
target/source structure latent。

它目前是 pipeline smoke，不是有效生成结果。下一步需要改 generation objective
或 decoder/latent 表示，而不是只延长训练。
```
