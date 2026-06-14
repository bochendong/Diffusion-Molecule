# SUCC MolEdit-Instruct Dual-Mode v4 Warmstart-v1

| 字段 | 值 |
| --- | --- |
| **状态** | planned / ready to submit |
| **最后更新** | 2026-06-14 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh`（当前默认 v4_warmstart_v1） |
| **前序** | [dualmode-v1](moledit-instruct-dualmode-v1.md) / [dualmode-v3-guarded](moledit-instruct-dualmode-v3-guarded.md) |

## 动机

v3 guarded 保住了 Table1 guard，并把 OOD / 2p7p target latent cosine 拉到
0.57-0.58；但 strict success 和 unique valid 仍低于 v1。v1 的 zero-source
diversity / reverse stimulation 更强，所以 v4 不再从随机初始化开始，而是从
dualmode v1 checkpoint warm-start，再用更小学习率吸收 v3 的 mixed-latent
和 source-free augmentation 经验。

## 默认配置

| 变量 | v3 guarded | v4 warmstart-v1 |
| --- | ---: | ---: |
| `SUCC_INIT_CHECKPOINT` | none | `...dualmode_v1/.../univideo_molecule_generation.pt` |
| `SUCC_INIT_STATS_FROM_CHECKPOINT` | 0 | **1** |
| `SUCC_LR` | 1e-3 | **3e-4** |
| `SUCC_STAGE1/2/3_EPOCHS` | 4 / 14 / 6 | **1 / 8 / 4** |
| `SUCC_LATENT_TARGET_MODE` | mixed | mixed |
| `SUCC_SOURCE_DROPOUT` | 0.12 | 0.12 |
| `SUCC_DENOVO_SAMPLE_WEIGHT` | 4.0 | **3.0** |
| `SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT` | 0.05 | **0.08** |
| `SUCC_SOURCE_FREE_AUGMENT_WEIGHT` | 0.50 | **0.30** |
| `SUCC_SOURCE_FREE_AUGMENT_PROB` | 0.40 | **0.35** |
| `SUCC_TABLE1_GUARD_MIN_ACC065` | 0.794 | 0.794 |

输出路径：

```text
outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1/
outputs/denovo_ood_ours_dualmode_v4_warmstart_v1/benchmark_ours/
outputs/denovo_2p7p_ours_dualmode_v4_warmstart_v1/benchmark_ours/
```

## 提交命令

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_dualmode_pipeline.sh
```

## 观察指标

| 指标 | 目标 |
| --- | --- |
| Table1 10-task `Acc_all(0.65)` mean | >= 0.794，guard pass |
| OOD overall strict | 优先超过 v1 0.127；至少超过 v3 0.101 |
| OOD unique valid | 接近或超过 v1 0.199 |
| OOD target latent cosine | 保持接近 v3 0.571 |
| 2p7p overall strict | 优先超过 v1 0.039；至少超过 v3 0.0285 |
| 2p7p unique valid | 接近或超过 v1 0.068 |

## 待回填

| Job | 名称 | 墙钟时间 | Exit | 说明 |
| --- | --- | --- | ---: | --- |
| TBD | `succ-univideo-mol` | TBD | TBD | v4 warm-start training |
| TBD | `succ-table1-guard` | TBD | TBD | Table1 guard |
| TBD | `succ-denovo-ood-ours` | TBD | TBD | OOD |
| TBD | `succ-denovo-2p7p-ours` | TBD | TBD | de novo 2p7p |
