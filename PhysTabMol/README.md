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
    generated_table_rows.csv
    decoded_candidates.csv
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
- `contrastive.py`：NumPy 版 InfoNCE 风格图像/表格对齐。
- `diffusion.py`：基于 `sklearn.neural_network.MLPRegressor` 的条件表格扩散。
- `torch_diffusion.py`：用于 GPU 服务器训练的 PyTorch 后端。
- `experiment.py`：保存配置、模型、表格结果和指标的服务器实验入口。
- `decoder.py`：具备物理先验的骨架与官能团模板解码器。
- `evaluate.py`：有效性、唯一性、新颖性、类药性、性质误差与多样性等指标。
- `run_demo.py`：端到端冒烟实验。

## 论文实验建议

以当前演示为最小可行流水线，再将起始分子替换为 PubChem/ZINC/ChEMBL 子集。

建议消融：

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
