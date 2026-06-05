# Diffusion Molecule

这个仓库现在只保留和当前论文主线直接相关的部分：

```text
SketchMolBenchmark
SketchMol-Understanding-Condition
SketchMol-MultiProperty-EditDataset
Research/Molecule Generation/SketchMol/SketchMol-v1-main
```

## 当前主线

我们的目标是做 source-conditioned multi-property molecular editing：

```text
source molecule image + natural-language instruction
    -> source-similar target molecule satisfying 2-7 property constraints
```

核心设计保持为双流：

```text
Understanding stream: image + instruction -> edit condition / edit latent
Generation stream: source-conditioned diffusion -> target molecule
```

## 目录

`SketchMol-MultiProperty-EditDataset`
负责大数据集构建、source-target edit pair、SketchMol-compatible condition
columns、`diffusion_edit_manifest.csv`，以及 `strict@Tanimoto>=t` 评估。

`SketchMol-Understanding-Condition`
负责 multimodal understanding、HF VLM feature export、connector/rerank baseline，
以及一键提交大模型 understanding benchmark。

`SketchMolBenchmark`
负责真实 SketchMol + MolScribe/OCR baseline，以及把直接结构预测 materialize 成
统一 benchmark summary。

`Research/Molecule Generation/SketchMol/SketchMol-v1-main`
是官方 SketchMol vendored repo，仅作为 `SketchMolBenchmark` 的外部依赖保留。
不要在这里继续扩展新方法。

## 一键大实验

在 Nibi/Slurm login node 上：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
bash SketchMol-Understanding-Condition/scripts/submit_hf_vlm_multiproperty_pipeline.sh
```

这个 pipeline 会：

```text
1. 构建 MultiProperty edit dataset
2. 导出 diffusion_edit_manifest.csv
3. 跑 HF VLM understanding features
4. 训练 edit connector
5. 输出 SketchMol-style strict success 和 source Tanimoto constrained results
```

关键报告路径：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm/benchmark_report.md
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm_edit_connector/benchmark_report.md
```

## 数据输入

默认输入表已经迁到：

```text
SketchMol-MultiProperty-EditDataset/data/train_table.csv
```

需要换成更大的外部数据源时，用：

```bash
SMMED_INPUT_CSV=/path/to/train_table.csv
```

## 主结果口径

不要只看 property strict success，也不要只看 scaffold match。当前主结果应该同时报告：

```text
2p-7p strict success
mean / median source Tanimoto
strict@Tanimoto>=0.4 / 0.6 / 0.8
validity
```

如果一个方法 strict success 高但 source Tanimoto 低，它更像 property retrieval，
不能算真正的 source-conditioned editing。
