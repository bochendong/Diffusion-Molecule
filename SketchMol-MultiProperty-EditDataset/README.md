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

核心任务是 source-conditioned multi-property edit：给定 source molecule image
和性质编辑指令，生成/找到一个同时满足 2-7 个性质约束、并且和 source molecule
保持足够结构相似的 target molecule。Scaffold match 仍然保留为诊断指标，但主
source-preservation 指标改成 Morgan fingerprint Tanimoto(source, generated)。

## 默认输出

默认输出目录：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/
```

默认输入表：

```text
SketchMol-MultiProperty-EditDataset/data/train_table.csv
```

这张表由旧实验迁移而来，保留 canonical molecule rows 和 SketchMol 风格 7 个
性质列，避免数据集构建继续依赖已经清理掉的历史项目目录。需要换更大的外部
数据源时，设置 `SMMED_INPUT_CSV=/path/to/train_table.csv`。

里面会生成：

```text
molecule_database.csv
edit_pairs.csv
condition_rows.csv
baseline_variants.csv
diffusion_edit_manifest.csv
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

`diffusion_edit_manifest.csv` 是后续 Understanding stream + diffusion generation
stream 训练要用的 handoff 表。它保留：

```text
source_smiles / target_smiles
source_image / target_image
instruction / prompt
condition_properties / property_count
source_tanimoto / source_similarity_bin
SketchMol-compatible property setting columns
```

默认只导出 `source_tanimoto >= 0.4` 的样本，避免 diffusion 训练集退化成
“无关 source 的性质检索”。需要调整时设置：

```bash
SMMED_DIFFUSION_MIN_SOURCE_TANIMOTO=0.3
SMMED_DIFFUSION_MAX_SOURCE_TANIMOTO=1.0
```

`benchmark_scaffold_retrieval/benchmark_report.md` 是最重要的 report-ready 结果：
它直接给出 2p-7p strict success table，并附上 SketchMol reference row。报告也会
输出 source-similarity-constrained success：

```text
strict@Tanimoto>=t = strict property success AND Tanimoto(source, generated) >= t
```

这个指标比硬 scaffold match 更适合作为我们现在的主结果，因为它允许合理的局部
结构变化，同时仍然惩罚完全脱离 source 的 property retrieval。报告也保留
`joint success` 作为硬 scaffold 诊断：

```text
joint success = strict property success AND scaffold match
```

strict success 用来和 SketchMol 的多性质控制表对齐；`strict@Tanimoto>=t` 用来
判断模型是否真的在 source-conditioned edit，而不是只在数据库里找满足性质的分子。

## 一键跑完整 benchmark

如果你现在想看到接近 SketchMol 论文表格的结果，优先跑这个：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

这个命令会提交 Slurm job，在 compute node 上完成：

1. 构建 molecule database。
2. 构建 scaffold-preserving edit pair database。
3. 生成 2-7 property condition rows。
4. 导出 diffusion/edit 训练 manifest。
5. 跑 SketchMol-style strict success + source Tanimoto benchmark。
6. 输出可直接写进 report 的 markdown 表格。

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

其中 `scaffold_property_retrieval` 是当前最关键的强 baseline：它默认从
`molecule_database.csv` 这个大候选库里检索同 scaffold、active-property
values 最接近 target 的 molecule。默认会把 eval target 从 retrieval
candidate pool 排除，避免直接拿到答案。这个设置比只从 train target rows
取候选更适合我们现在要验证的“大数据库 + scaffold-preserving edit ranking”
方向。

但 report 主结论不要只看 scaffold/joint。当前更应该看：

```text
2p-7p strict success
mean/median source Tanimoto
strict@Tanimoto>=0.4 / 0.6 / 0.8
```

如果某个方法 strict success 高、但 source Tanimoto 低，它更像 property retrieval，
不能算真正的 source-conditioned editing。

默认 `SMMED_SCAFFOLD_FALLBACK_MODE=source_identity`。如果某个 eval scaffold
在 candidate pool 里完全没有同 scaffold 候选，benchmark 会退回 source
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

构建脚本会先加载 `StdEnv/2023`、`python/3.11` 和 `rdkit/2025.09.4`。只有当你
指定的 Python 能 import RDKit 时，才覆盖 `SMMED_PYTHON_BIN`。

默认不会预渲染 molecule PNG：

```text
SMMED_RENDER_IMAGES=0
```

这是为了避免大数据集生成海量小 PNG，打爆集群 inode/file-count quota。需要调试少量
图像时再显式设置 `SMMED_RENDER_IMAGES=1`。大 VLM workflow 可以直接从
`source_smiles` 在内存中渲染分子图，不依赖预生成 PNG 文件。

## Source-neighbor 数据集

如果当前目标是训练真正的 source-conditioned edit，而不是 property retrieval，
优先构建新的 source-neighbor 版本：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

bash SketchMol-MultiProperty-EditDataset/scripts/submit_source_neighbor_dataset.sh
```

Slurm 默认资源按串行 RDKit/CSV 构建收紧：

```text
SMMED_SLURM_CPUS=1
SMMED_SLURM_MEM=8G
SMMED_SLURM_TIME=02:00:00
```

脚本也会设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`
和 `NUMEXPR_NUM_THREADS=1`，避免串行 job 申请多核但实际只跑一个线程。

默认输出目录：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/
```

这个路径默认：

```text
SMMED_PAIRING_STRATEGY=source_neighbor
SMMED_RENDER_IMAGES=0
SMMED_SOURCE_NEIGHBOR_MIN_TANIMOTO=0.4
SMMED_SOURCE_NEIGHBOR_MAX_TANIMOTO=0.95
SMMED_MIN_SOURCE_NEIGHBORS_T04=2
SMMED_CONDITION_CANDIDATE_DIAGNOSTICS=1
SMMED_MIN_STRICT_CANDIDATES_T04=1
SMMED_ORACLE_FILTER_SPLITS=eval
```

也就是说，pair 挖掘时优先选同 scaffold 且 `source_tanimoto>=0.4` 的
source-local edit；`source_tanimoto>0.95` 的近 no-op pair 会被排除。主 CSV
不存图片，只保留空的 `source_image` / `target_image` 兼容列。

新增的 pair/condition 质量列包括：

```text
source_tanimoto
source_similarity_bin
source_scaffold / target_scaffold / same_scaffold / scaffold_relation
pair_quality_tier
same_scaffold_neighbor_count
source_neighbor_count_t04 / t05 / t06
target_neighbor_rank_by_tanimoto
candidate_pool_size_t04 / t05 / t06
strict_candidate_count_t04 / t05 / t06
oracle_candidate_smiles_t04
oracle_source_tanimoto_t04
oracle_strict_success_t04
source_identity_strict_success
instruction_template_id / instruction_style / preservation_constraint
property_constraints_json
```

`SMMED_ORACLE_FILTER_SPLITS=eval` 会过滤掉 eval 中没有 source-neighbor strict
candidate 的 condition row，避免 benchmark 再测到“候选库没有可行答案”的假失败。
train split 默认保留更多样本，因为真实 target 本身仍然是有效 edit 监督。

## 常用参数

扩大数据集：

```bash
SMMED_INPUT_CSV=SketchMol-MultiProperty-EditDataset/data/train_table.csv
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

后面 report 的主表先用完整 benchmark 输出的 2p-7p strict success table 和
`strict@Tanimoto>=0.4/0.6/0.8`。旧的几百条 Understanding-Condition 数据适合做
pipeline validation，不适合作为主实验结论。下一步大实验应该用
`diffusion_edit_manifest.csv` 接 Understanding stream + diffusion generation
stream，而不是继续扩大纯检索模型。
