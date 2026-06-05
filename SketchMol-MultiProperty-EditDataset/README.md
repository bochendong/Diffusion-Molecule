# SketchMol Multi-Property Edit Dataset

这个文件夹专门放 Understanding-Condition 后续大实验需要的数据集构建、
数据产物和 benchmark 脚本。它和模型实验文件夹分开，避免把数据工程、
Slurm 入口、SketchMol 风格评估混在一起。

## 目标

构建一个能和 SketchMol 多性质控制结果对齐的大数据集：

```text
source molecule image + natural-language multi-property edit instruction
    -> target molecule / property deltas / SketchMol-compatible condition fields
```

核心任务是 scaffold-preserving multi-property edit，也就是在保留核心骨架的
前提下同时控制 2-7 个性质。

## 默认输出

默认输出目录：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/
```

里面会生成：

```text
molecule_database.csv
edit_pairs.csv
condition_rows.csv
baseline_variants.csv
summary.json
images/                       # optional; only created when SMMED_RENDER_IMAGES=1
benchmark_scaffold_retrieval/
```

`molecule_database.csv` 存 canonical SMILES、scaffold 和 7 个 SketchMol 性质：

```text
MW, LogP, QED, TPSA, HBD, HBA, RB
```

`edit_pairs.csv` 存 scaffold-preserving source-target molecule pairs。

`condition_rows.csv` 从每个 pair 采样 2-7 个 active properties，并生成自然语言
instruction 和 SketchMol-compatible condition columns。

`baseline_variants.csv` 展开 Understanding-Condition ablation 需要的输入形式：

```text
full, text_only, image_only, random_query, caption_bottleneck
```

`benchmark_scaffold_retrieval/benchmark_report.md` 是最重要的 report-ready 结果：
它直接给出 2p-7p strict success table，并附上 SketchMol reference row。报告也会
输出 `joint success`：

```text
joint success = strict property success AND scaffold match
```

strict success 用来和 SketchMol 的多性质控制表对齐；joint success 用来判断模型
是否真的在保留 source scaffold 的前提下完成编辑。

## 一键跑完整 benchmark

如果你现在想看到接近 SketchMol 论文表格的结果，优先跑这个：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

SMMED_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

这个命令会提交 Slurm job，在 compute node 上完成：

1. 构建 molecule database。
2. 构建 scaffold-preserving edit pair database。
3. 生成 2-7 property condition rows。
4. 跑 SketchMol-style strict success benchmark。
5. 输出可直接写进 report 的 markdown 表格。

日志目录：

```text
SketchMol-MultiProperty-EditDataset/logs/
```

结果目录：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_scaffold_retrieval/
  benchmark_report.md
  benchmark_summary.csv
  benchmark_decoded.csv
  metrics.json
```

默认 benchmark rows：

```text
source_identity
scaffold_property_retrieval
target_oracle
SketchMol structured reference
```

其中 `scaffold_property_retrieval` 是当前最关键的强 baseline：它优先只从
train pool 里检索同 scaffold、active-property values 最接近 target 的
molecule。默认会把 eval target 从 retrieval candidate pool 排除，避免直接
拿到答案。

默认 `SMMED_SCAFFOLD_FALLBACK_MODE=source_identity`。如果某个 eval scaffold
在 train candidate pool 里完全没有同 scaffold 候选，benchmark 会退回 source
molecule，而不是偷偷改成 global retrieval。这样 strict success 可能更低，但
scaffold/joint 指标更诚实。需要复现旧的全局 fallback 行为时可以设置：

```bash
SMMED_SCAFFOLD_FALLBACK_MODE=global
```

`target_oracle` 是上界，不是可发表方法；它只用于确认数据和 strict-success
评估脚本是否正常。

## 只构建数据集

如果只想先生成数据，不跑 benchmark：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

The run script loads `StdEnv/2023`, `python/3.11`, and `rdkit/2025.09.4`
before building the dataset. Override `SMMED_PYTHON_BIN` only if that Python
can import RDKit.

默认不会预渲染 molecule PNG：

```text
SMMED_RENDER_IMAGES=0
```

这是为了避免大数据集生成海量小 PNG，打爆集群 inode/file-count quota。需要调试少量
图像时再显式设置 `SMMED_RENDER_IMAGES=1`。大 VLM workflow 可以直接从
`source_smiles` 在内存中渲染分子图，不依赖预生成 PNG 文件。

## 常用参数

扩大数据集：

```bash
SMMED_INPUT_CSV=PhysTabMol/runs/20260601_070814_sketchmol_compare_structure_seed7/tables/train_table.csv
SMMED_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_200k_v1
SMMED_LIMIT=200000
SMMED_MAX_PAIRS=200000
SMMED_MAX_PAIRS_PER_SCAFFOLD=500
SMMED_CONDITIONS_PER_PAIR=4
```

快速 dry run：

```bash
SMMED_LIMIT=10000 \
SMMED_MAX_PAIRS=5000 \
SMMED_MAX_PAIRS_PER_SCAFFOLD=50 \
SMMED_CONDITIONS_PER_PAIR=2 \
SMMED_MAX_EVAL_PER_PROPERTY_COUNT=500 \
SMMED_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/dry_run_10k \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

如果要额外跑 global retrieval baseline：

```bash
SMMED_BENCHMARK_METHODS=source_identity,global_property_retrieval,scaffold_property_retrieval,target_oracle \
SMMED_MAX_GLOBAL_CANDIDATES=5000 \
SMMED_MAX_EVAL_PER_PROPERTY_COUNT=1000 \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

如果要看 scaffold fallback 对结论的影响，可以同时跑两组：

```bash
SMMED_SCAFFOLD_FALLBACK_MODE=source_identity \
SMMED_BENCHMARK_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_scaffold_honest \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh

SMMED_SCAFFOLD_FALLBACK_MODE=global \
SMMED_BENCHMARK_OUTPUT_DIR=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_scaffold_global_fallback \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

## 当前建议

后面 report 的主表先用完整 benchmark 输出的 2p-7p strict success table。
旧的几百条 Understanding-Condition 数据适合做 pipeline validation，不适合作为主实验结论。
