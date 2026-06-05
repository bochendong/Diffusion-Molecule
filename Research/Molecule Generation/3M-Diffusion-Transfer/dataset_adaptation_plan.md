# 数据集改造方案

## 为什么不能直接用 3M-Diffusion 数据集

3M-Diffusion 的 ChEBI-20/PubChem 风格数据是：

```text
molecule description -> molecule structure
```

它没有 source molecule，也没有 source-target edit pair，因此不能直接回答我们的核心问题：

```text
给定 source molecule image 和多性质编辑指令，模型能否生成满足性质约束、
同时和 source 保持结构相似的 target molecule？
```

所以 3M 数据适合作为 pretraining / auxiliary alignment，不适合作为主 benchmark。

## 我们应该构建的 unified schema

建议后续统一导出 JSONL 或 CSV，每条样本都带 `task_type`：

```text
task_type=description_pretrain
  molecule_id
  molecule_smiles
  molecule_image_runtime_source
  description
  target_smiles

task_type=edit_generation
  source_smiles
  source_image_runtime_source
  instruction
  active_properties
  source_property_values
  target_property_values
  property_deltas
  source_tanimoto
  target_smiles
```

这样一个数据集可以同时支持：

```text
understanding pretraining
edit-aware connector training
diffusion condition training
SketchMol-style benchmark export
```

## 数据 split 原则

为了让结果更可信，建议 split 不能只按 row random split。需要至少保留两个 split：

```text
random pair split:
  用于快速迭代和 sanity check

source/scaffold-disjoint split:
  用于报告主结果，避免模型记住同 scaffold 的 target 候选
```

如果我们最终主指标改成 source Tanimoto，而不是硬 scaffold match，也仍然要避免
同一个 source molecule 或完全相同 target molecule 出现在 train/eval 两边。

## 借鉴 3M 的数据统计习惯

建议每次数据构建都输出：

```text
row count
unique SMILES count
split sizes
description/instruction length distribution
SMILES length distribution
active property count distribution
source Tanimoto distribution
valid molecule ratio
```

本文件夹里的 `scripts/profile_3m_datasets.py` 先对 3M-Diffusion 自带数据做同类统计，
方便我们之后把自己的数据集统计表做成同一种格式。

## 最小下一步

短期内最值得做的是新增一个导出脚本：

```text
export_unified_condition_dataset.py
```

输入：

```text
3M-Diffusion/data/ChEBI-20_data/*.txt
SketchMol-MultiProperty-EditDataset/outputs/.../diffusion_edit_manifest.csv
```

输出：

```text
unified_condition_train.jsonl
unified_condition_eval.jsonl
summary.json
```

这会把 3M 的 description pretraining 和我们的 edit generation 数据连接起来，
但 benchmark 主结果仍然只在我们的 edit_generation eval rows 上算。

