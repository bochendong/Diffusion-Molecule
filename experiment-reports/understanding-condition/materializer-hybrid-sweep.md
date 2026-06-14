# SUCC Zero-Source Materializer Hybrid Sweep

| 字段 | 值 |
| --- | --- |
| **状态** | ready to submit |
| **最后更新** | 2026-06-14 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_denovo_materializer_sweep.sh` |
| **目的** | 判断 zero-source strict 低是否主要来自 `generated_latent -> SMILES` materializer |

## 方法

固定复用已训练 checkpoint，不重新训练：

| Label | Checkpoint |
| --- | --- |
| v1 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1/` |
| v3 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded/` |
| v4 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1/` |

每个 checkpoint 同时跑 2p7p 和 OOD，并输出三种 materializer：

| Method | 说明 |
| --- | --- |
| `latent_nearest` | 当前默认，按 generated latent 最近邻取 candidate |
| `latent_property_rerank` | latent top-4096 shortlist，再按 absolute property strict/distance rerank |
| `property_nearest` | 候选库属性最近邻上界/诊断，不作为最终模型结果 |

默认 hybrid 打分：

```text
score = 100 * strict_fraction - 10 * normalized_property_distance + 1 * latent_score
```

## 提交命令

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_denovo_materializer_sweep.sh
```

## 待回填：2p7p

| Checkpoint | Method | Strict | Unique valid | Target latent cos | Hybrid vs latent | Gap to property_nearest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | latent_nearest | TBD | TBD | TBD | — | — |
| v1 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v1 | property_nearest | TBD | TBD | — | — | — |
| v3 | latent_nearest | TBD | TBD | TBD | — | — |
| v3 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v3 | property_nearest | TBD | TBD | — | — | — |
| v4 | latent_nearest | TBD | TBD | TBD | — | — |
| v4 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v4 | property_nearest | TBD | TBD | — | — | — |

## 待回填：OOD

| Checkpoint | Method | Overall strict | Unique valid | Target latent cos | Hybrid vs latent | Gap to property_nearest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | latent_nearest | TBD | TBD | TBD | — | — |
| v1 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v1 | property_nearest | TBD | TBD | — | — | — |
| v3 | latent_nearest | TBD | TBD | TBD | — | — |
| v3 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v3 | property_nearest | TBD | TBD | — | — | — |
| v4 | latent_nearest | TBD | TBD | TBD | — | — |
| v4 | latent_property_rerank | TBD | TBD | TBD | TBD | TBD |
| v4 | property_nearest | TBD | TBD | — | — | — |

## 判读

- 如果 `latent_property_rerank` 明显超过 `latent_nearest`，说明主要瓶颈是 materializer。
- 如果 `latent_property_rerank` 接近 `property_nearest`，说明当前模型 latent 已足以 shortlist，下一步应优化 rerank/agentic policy。
- 如果 `latent_property_rerank` 仍然低，下一步转向 candidate-in-loop contrastive training 或 RL。
