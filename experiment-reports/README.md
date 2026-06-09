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

每个实验 md 建议包含：**状态**、**目的**、**配置**、**产出路径**、**结果摘要**、**结论与下一步**。

## 当前实验索引

| 文件夹 | 报告文件 | 说明 |
| --- | --- | --- |
| `unified-3m/` | [moledit-instruct-v1.md](unified-3m/moledit-instruct-v1.md) | MolEdit Unified 3M 训练完成（`15810199`）；benchmark 首次空跑（`15822025`/`15822029`） |
| `understanding-condition/` | [source-neighbor-v2-residual-ink.md](understanding-condition/source-neighbor-v2-residual-ink.md) | **新主线** source-neighbor 重训 + benchmark（`15821981`/`15821983`） |
| `understanding-condition/` | [univideo-v2-residual-ink.md](understanding-condition/univideo-v2-residual-ink.md) | 旧 canonical v2（`multiproperty_100k_v1`，历史对照） |
| `understanding-condition/` | [materialized-benchmark-v2-primary-fast.md](understanding-condition/materialized-benchmark-v2-primary-fast.md) | 旧 v2 materialized benchmark（`15810260`） |
| `multiproperty/` | [source-neighbor-v1.md](multiproperty/source-neighbor-v1.md) | source-neighbor 数据集构建 |

## 近期 Slurm Job 一览（2026-06-08 ~ 09）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15810199` | `smu3m-unified` | 完成（3m45s） | [moledit-instruct-v1](unified-3m/moledit-instruct-v1.md) |
| `15821981` | `succ-univideo-mol` | 完成（54m35s） | [source-neighbor-v2](understanding-condition/source-neighbor-v2-residual-ink.md) |
| `15821983` | `succ-univideo-bench` | 完成（12m28s） | [source-neighbor-v2](understanding-condition/source-neighbor-v2-residual-ink.md) |
| `15822025` | `smu3m-diff-bench` | 完成但空结果 | [moledit-instruct-v1](unified-3m/moledit-instruct-v1.md) |
| `15822029` | `smu3m-moledit-table` | 完成但空结果 | [moledit-instruct-v1](unified-3m/moledit-instruct-v1.md) |
| `15810260` | `succ-univideo-bench`（旧 v2） | 完成 | [materialized-benchmark-v2](understanding-condition/materialized-benchmark-v2-primary-fast.md) |

新增实验时：在对应子文件夹下新建 `{实验名}.md`，并在此表加一行链接。
