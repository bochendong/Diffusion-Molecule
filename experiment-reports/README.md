# 实验报告

本目录存放各实验的可更新报告（Markdown）。

## 当前实验索引

| 文件夹 | 报告文件 | 说明 |
| --- | --- | --- |
| `.` | [moleditrl-baseline-comparison.md](moleditrl-baseline-comparison.md) | MolEditRL vs SUCC v1/v2 / Unified 对照 |
| `unified-3m/` | [moledit-instruct-v1.md](unified-3m/moledit-instruct-v1.md) | Unified 3M MolEdit v1（`15810199`/`15837897`） |
| `unified-3m/` | [moledit-sourceanchored-v2.md](unified-3m/moledit-sourceanchored-v2.md) | **完成** source-anchored v2（`15866599`/`15884514-21`/`15884578`） |
| `understanding-condition/` | [moledit-instruct-v1.md](understanding-condition/moledit-instruct-v1.md) | SUCC MolEdit v1（`15838733`/`15838734`） |
| `understanding-condition/` | [moledit-instruct-v2-fix.md](understanding-condition/moledit-instruct-v2-fix.md) | **完成** SUCC v2 fix（`15866598`/`15866600`/`15866601`） |
| `understanding-condition/` | [source-neighbor-v2-residual-ink.md](understanding-condition/source-neighbor-v2-residual-ink.md) | source-neighbor 对照（`15821981`/`15821983`） |

## 近期 Slurm Job 一览（2026-06-09 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15866598` | `succ-univideo-mol`（v2 fix） | 完成（27m16s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866600` | `succ-univideo-bench`（v2 fix） | 完成（2m37s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866601` | `succ-moledit-table`（v2 fix） | 完成（5s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866599` | `smu3m-srcanch-v2` | 完成（3m17s） | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884514`–`15884521` | `smu3m-diff-bench`（5 shards） | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884578` | `smu3m-moledit-table` | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |

新增实验时：在对应子文件夹下新建 `{实验名}.md`，并在此表加一行链接。
