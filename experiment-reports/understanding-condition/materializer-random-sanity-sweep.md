# SUCC Materializer Random-Shortlist Sanity Sweep

| 字段 | 值 |
| --- | --- |
| **状态** | 待运行 |
| **创建** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_denovo_materializer_sanity_sweep.sh` |
| **目的** | 检查 hybrid materializer 的 0.970/0.985 是否主要来自 property rerank 本身 |

## 设计

固定 checkpoint：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1/
```

每个 shortlist size 同时跑 2p7p 和 OOD：

| Method | 说明 |
| --- | --- |
| `latent_property_rerank` | generated latent top-k shortlist，再按 absolute property rerank |
| `random_property_rerank` | 随机 top-k shortlist，再按同一套 absolute property rerank；不使用 generated latent |
| `property_nearest` | 全候选库属性最近邻上界/诊断 |

默认 shortlist size：

```text
64 128 256 512 1024 4096
```

## 判读标准

| 现象 | 解释 |
| --- | --- |
| `random_property_rerank` 接近 `latent_property_rerank` | 结果主要来自 candidate library + property rerank，模型贡献弱 |
| `latent_property_rerank` 显著高于 random，尤其在小 k 下 | generated latent 能有效缩小候选空间，模型贡献站得住 |
| 小 k 下 latent 优势明显，大 k 下 random 追上 | 模型贡献主要是 shortlist efficiency，而非最终属性 oracle |

## 提交命令

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_denovo_materializer_sanity_sweep.sh
```

可缩小 sweep：

```bash
SUCC_SANITY_PROPERTY_RERANK_CANDIDATES_LIST="64 256 4096" \
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_denovo_materializer_sanity_sweep.sh
```

## 结果

待填。
