# 实验报告

本目录存放各实验的可更新报告（Markdown）。每个子文件夹对应一条实验线或一类任务；文件夹内的 `.md` 文件按实验命名，随跑数、改配置或出新结果持续更新。

## 目录约定

```text
experiment-reports/
  README.md                 # 本说明
  unified-3m/               # SketchMol-Unified-3MDiffusion
  understanding-condition/  # SketchMol-Understanding-Condition
  multiproperty/            # SketchMol-MultiProperty-EditDataset
  smiles-dualstream/        # SMILES-DualStream-EditorAtomas
```

每个实验 md 建议包含：

- **状态**：planned / running / completed / blocked
- **目的**：一句话说明要验证什么
- **配置**：关键环境变量、数据路径、Slurm job
- **产出**：checkpoint、metrics、benchmark 路径
- **结果摘要**：表格或要点（从 `outputs/` 或 benchmark_report 抄录）
- **结论与下一步**：待填

## 当前实验索引

| 文件夹 | 报告文件 | 说明 |
| --- | --- | --- |
| `unified-3m/` | [moledit-instruct-v1.md](unified-3m/moledit-instruct-v1.md) | MolEdit-Instruct enhanced splits 上的 Unified 3M 训练 |
| `understanding-condition/` | [univideo-v2-residual-ink.md](understanding-condition/univideo-v2-residual-ink.md) | UniVideo image-to-structure 完整 pipeline（canonical v2） |
| `understanding-condition/` | [materialized-benchmark-v2-primary-fast.md](understanding-condition/materialized-benchmark-v2-primary-fast.md) | v2 OCR-free materialized benchmark（**已完成**，job `15810260`，12 min） |
| `multiproperty/` | [source-neighbor-v1.md](multiproperty/source-neighbor-v1.md) | source-neighbor multi-property 数据集构建 |

新增实验时：在对应子文件夹下新建 `{实验名}.md`，并在此表加一行链接。
