# SUCC Direct SMILES De Novo v0

| 字段 | 值 |
| --- | --- |
| **状态** | 待运行 |
| **创建** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_direct_smiles_denovo_2p7p_benchmark.sh` / `submit_direct_smiles_denovo_ood_benchmark.sh` |
| **目的** | 建立不依赖 candidate retrieval / property rerank 的 real direct de novo baseline |

## 动机

`latent_property_rerank` 在 2p7p/OOD 上达到 0.970/0.985，但 random-shortlist sanity
显示 `random_property_rerank@4096` 也能达到 0.978/0.980。因此该高分主要来自
candidate library + property rerank，不能作为 direct de novo generation claim。

Direct SMILES v0 改为：

```text
condition text / property prompt
  -> Qwen/HF VLM query tokens
  -> Transformer SMILES decoder
  -> generated_smiles
  -> direct-SMILES evaluator
```

不使用 candidate library，不使用 `property_nearest`，不使用 materializer rerank。

## 方法

新增：

| 文件 | 说明 |
| --- | --- |
| `sketchmol_understanding_condition/direct_smiles_generation.py` | SMILES tokenizer + MLLM-conditioned Transformer decoder |
| `scripts/train_direct_smiles_generator.py` | 训练/评估 direct SMILES generator，输出 `direct_smiles_predictions.csv` |
| `scripts/run_direct_smiles_denovo_2p7p_benchmark.sh` | 导出 2p7p train/eval rows、Qwen features、训练并评估 |
| `scripts/run_direct_smiles_denovo_ood_benchmark.sh` | 导出 OOD train/eval rows、Qwen features、训练并评估 |

## 提交命令

2p7p：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_benchmark.sh
```

OOD：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_benchmark.sh
```

默认配置：

| 参数 | 2p7p | OOD |
| --- | ---: | ---: |
| Train rows | 2000 / property-count | 800 / spec |
| Eval rows | 1000 / property-count | 100 / spec |
| Epochs | 12 | 12 |
| Model | 4-layer Transformer decoder | 4-layer Transformer decoder |
| Condition | Qwen `query_tokens.npy` | Qwen `query_tokens.npy` |

## 结果

待填。
