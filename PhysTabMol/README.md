# PhysTabMol

PhysTabMol 是一个面向 **图像条件分子设计的、具备物理先验的对比式表格扩散** 的快速概念验证。

本目录为独立原型，**不会**从 SketchMol、ImagiChem、UniVideo 或 `Research/` 下任何论文仓库导入代码；这些项目仅作为方法设计的研究启发。

原型实现了 SketchMol + ImagiChem 讨论中的设想：

- 图像特征作为条件信号；
- 分子在描述符 / 片段计数表格空间中生成，而非像素空间；
- 轻量对比对齐器学习图像与表格的一致性；
- 条件表格扩散模型采样候选设计行；
- 保守的、具备物理先验的解码器将行转为类 SMILES 候选，并用药物化学规则过滤；
- 受 UniVideo 启发的上下文界面将查询图像、参考图像/参考分子与文本意图合并为单一生成条件。

## 项目动机

SketchMol 表达力强，但图像生成加 OCR 恢复可能不稳定。ImagiChem 可复现且化学约束明确，但图像到化学的映射大量依赖手写规则。UniVideo 提供第三种思路：不同任务可共享同一上下文条件界面，由媒介、参考与指令构成。PhysTabMol 介于其间：学习紧凑的图像条件表格分布，用上下文参考引导生成/编辑，再在解码与过滤阶段施加基于规则的化学先验。

## 服务器实验入口

真实实验请优先使用 `phystabmol.experiment`：

```bash
python3 -m phystabmol.experiment \
  --data /path/to/molecules.csv \
  --smiles-column smiles \
  --image-column image_path \
  --backend auto \
  --run-name pubchem_pilot \
  --samples-per-condition 32 \
  --decode-top-k 5
```

`--image-column` 是可选项。若不提供，PhysTabMol 会从 SMILES 生成确定性的代理分子图像，使图像条件与对比学习流程仍可运行。若服务器装有 RDKit，描述符、有效性与多样性计算会自动走 RDKit。

每次实验会创建独立的时间戳目录：

```text
runs/YYYYMMDD_HHMMSS_run_name/
  config.json
  environment.json
  metrics.json
  summary.txt
  models/
    contrastive_aligner.pkl
    diffusion.pt 或 diffusion.pkl
  tables/
    train_table.csv
    test_table.csv
    understanding_stream_train.csv
    understanding_stream_eval.csv
    generated_table_rows.csv
    decoded_candidates.csv
    sketchmol_benchmark/
      sketchmol_benchmark_summary.csv
      sketchmol_benchmark_decoded.csv
      sketchmol_distribution_matching.json
```

在 GPU 服务器上，`--backend auto` 会优先使用 PyTorch；如果想强制使用 PyTorch：

```bash
python3 -m phystabmol.experiment --data /path/to/molecules.csv --backend torch
```

## 输入 CSV

最低要求：

```csv
smiles
CC(=O)Nc1ccccc1
COc1ccncc1
```

若已有分子图像，可提供：

```csv
smiles,image_path
CC(=O)Nc1ccccc1,/abs/path/mol_001.png
COc1ccncc1,/abs/path/mol_002.png
```

## 自动下载数据

第一次建议直接在服务器下载 ChEMBL：

```bash
cd PhysTabMol
bash scripts/download_chembl_100k.sh
```

默认会生成：

```text
data/molecules.csv
data/molecules.manifest.txt
```

`data/` 下的大型下载文件默认会被 `.gitignore` 排除，适合 push 代码后直接在服务器上运行下载脚本。

也可以手动选择数据源：

```bash
# ChEMBL，默认推荐
python3 scripts/download_dataset.py --source chembl --limit 100000 --out data/molecules_chembl.csv

# PubChem，官方 CID-SMILES，先抽 100k
python3 scripts/download_dataset.py --source pubchem --limit 100000 --out data/molecules_pubchem_100k.csv

# ZINC20 ML smiles chunks，文件较大
python3 scripts/download_dataset.py --source zinc20 --zinc-chunks 1 --limit 100000 --out data/molecules_zinc20_100k.csv
```

如果服务器装了 RDKit，可以加过滤：

```bash
python3 scripts/download_dataset.py \
  --source chembl \
  --limit 100000 \
  --rdkit-filter \
  --out data/molecules.csv
```

## 快速冒烟测试

在本目录下执行：

```bash
python3 -m phystabmol.run_demo
```

演示会写入：

```text
outputs/phystabmol_demo.csv
```

也可对外部图像做条件生成：

```bash
python3 -m phystabmol.run_demo --image /path/to/image.png --samples 12
```

UniVideo 风格的上下文生成/编辑：

```bash
python3 -m phystabmol.run_demo \
  --reference-smiles "CC(=O)Nc1ccccc1" \
  --intent increase_qed \
  --samples 12
```

支持的轻量文本意图：

- `increase_qed`
- `increase_logp`
- `decrease_logp`
- `lower_sa`
- `more_polar`
- `less_polar`

## Understanding Stream

PhysTabMol 现在显式包含一条受 UniVideo 启发的 **understanding stream**：

```text
query/reference/intent
  -> understanding_stream
  -> structured semantic tags + summary + numeric understanding embedding
  -> tabular diffusion condition
  -> physics-aware decoding
```

它会为每个样本保存：

- `understanding_summary`：可读的语义理解结果；
- `understanding_tags`：如 `high_visual_contrast`、`druglike_window`、`reference_guided`、`increase_qed`；
- `u_*` 数值列：拼接进 diffusion 条件向量。

输出文件：

```text
tables/understanding_stream_train.csv
tables/understanding_stream_eval.csv
```

如果要做 ablation，可以关闭这条流：

```bash
python3 -m phystabmol.experiment \
  --data /path/to/molecules.csv \
  --disable-understanding-stream
```

## SketchMol-Aligned Benchmark

因为主要比较对象是 SketchMol，服务器实验可以直接打开对齐 benchmark：

```bash
python3 -m phystabmol.experiment \
  --data /path/to/molecules.csv \
  --backend torch \
  --understanding-backbone clip \
  --run-sketchmol-benchmark \
  --benchmark-samples-per-condition 1000 \
  --benchmark-multi-conditions 1000 \
  --benchmark-optimization-conditions 100
```

当前 benchmark 对齐 SketchMol 的这些实验口径：

- 单属性约束：LogP、QED、MW、TPSA、HBD、HBA、RB；
- OOD 约束：LogP/TPSA/HBA/RB/MW 的外推 target；
- 多属性约束：2 到 7 个属性同时约束；
- 性质优化：LogP +2.5、QED +0.3、TPSA -45；
- 分布匹配：LogP、QED、MW、TPSA 的 1D Wasserstein 近似；
- 基础指标：Validity、Uniqueness、Novelty、Success Rate in Valid Mols、MAE。

SketchMol 的 EP4/AKT1/ROCK1 activity 与 docking 实验需要额外的 activity predictor 和 docking workflow；当前框架先把属性约束、优化和 3D conformer 评估打通，activity/docking scorer 应作为下一步外部模块接入。

输出位置：

```text
tables/sketchmol_benchmark/
```

## 恢复未完成的 Slurm Run

如果作业已经生成了 `tables/generated_table_rows.csv`，但因为 time limit 没来得及生成
`decoded_candidates.csv / metrics.json / summary.txt`，可以不用重训，直接恢复解码评估：

```bash
python3 -m phystabmol.decode_run \
  --run-dir runs/20260501_145202_slurm_13110897 \
  --max-conditions 5000 \
  --samples-per-condition 8 \
  --decode-top-k 2
```

默认会写回：

```text
tables/decoded_candidates.csv
metrics.json
summary.txt
postprocess_config.json
```

## GPU 使用率调节

默认 Slurm 脚本申请的是 10GB H100 MIG，不是整张 80GB H100。若 allocation 允许，可以在提交时覆盖
GPU slice，并放大 PyTorch batch/model：

```bash
PHYSTABMOL_TORCH_BATCH_SIZE=2048 \
PHYSTABMOL_TORCH_HIDDEN_DIM=1536 \
PHYSTABMOL_TORCH_LAYERS=8 \
sbatch scripts/run_phystabmol_gpu.slurm.sh
```

如果使用更大的 MIG/整卡，可以用 Slurm 命令行覆盖脚本里的 `#SBATCH --gpus=...`：

```bash
sbatch --gpus=h100:1 --mem=160G scripts/run_phystabmol_gpu.slurm.sh
```

每次 run 的 `environment.json` 会记录 `cuda_max_memory_allocated_mb` 和
`cuda_max_memory_reserved_mb`，用来判断显存是否真的吃满。注意：molecular decoding/evaluation 主要是
CPU/RDKit 工作，增加显存只能加速/放大训练阶段，不能直接解决解码超时。

## Verified Instruction-Guided Editing

项目现在也包含一个**无人工标注**的 instruction-guided molecular editing benchmark：

```text
source molecule + natural-language instruction
  -> edited molecule candidates
  -> RDKit / deterministic verifier
```

LLM 只允许用于把结构化 spec 改写成自然语言；化学事实是否完成，全部由 `instruction_spec_json`
里的可执行规则验证。RDKit 验不了的模糊药化目标不进入主表。

本地自检：

```bash
bash scripts/smoke_instruction_editing.sh
```

构建 verified instruction dataset：

```bash
cd PhysTabMol
bash scripts/build_instruction_dataset.sh
```

默认读取 `data/molecules.csv`，输出：

```text
data/instruction_editing.csv
data/instruction_editing.jsonl
```

每条样本包含：

```text
source_smiles,target_smiles,instruction_text,instruction_spec_json,
reference_smiles,reference_role,property_delta_json,edit_tags,split
```

先跑 deterministic baselines：

```bash
bash scripts/evaluate_instruction_baselines.sh
```

会评估：

- `no_edit`
- `random_target`
- `rule_retrieval`
- `oracle_target`

主指标：

- `validity`
- `goal_success_rate`
- `constraint_success_rate`
- `edit_success_rate`
- `overall_instruction_success_rate`
- `similarity_to_source`
- `novelty`
- `druglike_rate`

训练 instruction-guided tabular diffusion edit planner：

```bash
python3 -m phystabmol.instruction_experiment \
  --dataset data/instruction_editing.csv \
  --backend torch \
  --run-name instruction_pilot \
  --samples-per-instruction 8 \
  --decode-top-k 2
```

多模态输入 ablation：

```bash
python3 -m phystabmol.instruction_experiment \
  --dataset data/instruction_editing.csv \
  --backend torch \
  --run-name instruction_full_multimodal \
  --multimodal-context full
```

支持的 `--multimodal-context`：

- `none`：只用 source structure + instruction spec；
- `source_image`：追加 source molecule 的 2D rendered/proxy image features；
- `source_reference`：追加 source image、reference image 与 visual delta；
- `source_3d`：追加 source 的 RDKit 3D conformer descriptors；
- `full`：source/reference image + source/reference 3D descriptors。

跑完整对照：

```bash
bash scripts/run_instruction_multimodal_ablation.sh
```

`source_reference/full` 需要新数据集里的 `reference_smiles` 列。若想用 `target_smiles` 当显式 reference，
可以加 `--allow-target-reference`，但这应标注为 oracle-reference setting。

或直接提交 Slurm：

```bash
sbatch scripts/run_instruction_editing_gpu.slurm.sh
```

LLM paraphrase 工作流是离线的。先导出 prompt：

```bash
bash scripts/export_instruction_paraphrase_prompts.sh
```

外部 LLM 返回 `pair_id,instruction_text` 后，用确定性语言/spec consistency check 过滤：

```bash
python3 -m phystabmol.instruction_paraphrases filter \
  --dataset data/instruction_editing.csv \
  --paraphrases llm_paraphrases.jsonl \
  --out data/instruction_editing_llm_verified.csv
```

这一步只检查语言是否引入了 spec 外目标，不做化学裁判；最终化学评估仍由 verifier 完成。

## 3D Molecule Support

虽然不做视频，项目已经预留 3D 分子评估。若安装 RDKit，可开启：

```bash
python3 -m phystabmol.experiment \
  --data /path/to/molecules.csv \
  --enable-3d \
  --save-3d-sdf
```

它会用 ETKDG 生成 conformer，并保存：

- `3d_embed_success`
- `3d_radius_gyration`
- `3d_asphericity`
- `3d_eccentricity`
- `3d_npr1 / 3d_npr2`
- 可选 SDF 文件

## 依赖

本地冒烟测试使用：

- numpy
- pandas
- pillow
- scikit-learn

服务器训练建议安装：

```bash
pip install -r requirements-server.txt
conda install -c conda-forge rdkit
```

RDKit 对论文级有效性、描述符和 Tanimoto 多样性评估很重要。

## 原型模块

- `features.py`：图像统计与表格行构造。
- `context.py`：受 UniVideo 启发的查询/参考/指令条件。
- `understanding.py`：受 UniVideo 启发的显式理解流，输出语义摘要、标签和 understanding embedding。
- `contrastive.py`：NumPy 版 InfoNCE 风格图像/表格对齐。
- `diffusion.py`：基于 `sklearn.neural_network.MLPRegressor` 的条件表格扩散。
- `torch_diffusion.py`：用于 GPU 服务器训练的 PyTorch 后端。
- `experiment.py`：保存配置、模型、表格结果和指标的服务器实验入口。
- `decoder.py`：具备物理先验的骨架与官能团模板解码器。
- `evaluate.py`：有效性、唯一性、新颖性、类药性、性质误差与多样性等指标。
- `sketchmol_benchmark.py`：对齐 SketchMol 的单属性、多属性、OOD 与优化 benchmark。
- `geometry3d.py`：RDKit-backed 3D conformer 与形状指标。
- `instruction_dataset.py`：自动构建 source/target/edit spec instruction 数据。
- `instruction_verifier.py`：RDKit/规则验证 goal、constraint、edit 是否完成。
- `instruction_evaluate.py`：instruction editing 主指标评估。
- `instruction_multimodal.py`：source/reference molecule image 与可选 3D context 特征。
- `instruction_baselines.py`：no-edit、random、rule-retrieval、oracle baselines。
- `instruction_experiment.py`：instruction-guided tabular diffusion edit planner。
- `instruction_paraphrases.py`：LLM paraphrase prompt 导出与 deterministic consistency 过滤。
- `run_demo.py`：端到端冒烟实验。

## 论文实验建议

以当前演示为最小可行流水线，再将起始分子替换为 PubChem/ZINC/ChEMBL 子集。

建议消融：

- template-only instructions vs verified LLM paraphrases；
- unseen paraphrase test 与 unseen edit-combination test；
- no edit / random / rule retrieval / no-instruction planner；
- tabular diffusion without instruction vs instruction-guided tabular diffusion；
- 扩散 + 物理先验解码器 + 对比对齐；
- 扩散但无对比对齐；
- 扩散但无物理先验解码器；
- 仅规则的 ImagiChem 风格基线；
- 图像统计条件 vs 学习到的图像嵌入条件；
- 无上下文参考 vs 参考分子/图像条件。

建议指标：

- 有效性、唯一性、新颖性；
- Lipinski / Veber / SA 过滤通过率；
- MW、LogP、QED、TPSA 等目标性质的 MAE；
- 成对 Tanimoto 多样性；
- 小幅图像扰动下的稳定性。
