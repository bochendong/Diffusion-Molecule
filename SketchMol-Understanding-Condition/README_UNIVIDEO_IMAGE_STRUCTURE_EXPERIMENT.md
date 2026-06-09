# UniVideo Image-to-Structure 完整实验 README

这份文档保留历史 image-to-structure / MolScribe OCR 实验说明。当前主线已经切到
MolEdit-Instruct enhanced_v1 + OCR-free materialized benchmark：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_pipeline.sh
```

下面的 MolScribe/OCR 流程仅作为 legacy 诊断：

```text
source molecule image / SMILES + instruction
  -> HF VLM understanding stream
  -> connector 得到 diffusion condition tokens
  -> source-conditioned latent diffusion
  -> generated molecule image
  -> MolScribe OCR
  -> generated SMILES
  -> RDKit / SketchMol-style 2p-7p benchmark
```

最终要看的不是 latent MSE，而是：

```text
2p-7p strict success
validity
OCR SMILES present rate
mean / median source Tanimoto
strict@Tanimoto>=0.4 / 0.6 / 0.8
uniqueness / exact-target / source-identity diagnostics
```

## 1. 运行前检查

先进入项目：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main
```

检查数据是否存在：

```bash
test -f SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/condition_rows.csv
```

如果这条失败，说明 multi-property 数据集还没在当前 checkout 里生成或路径不对，需要设置：

```bash
export SUCC_CONDITION_ROWS=/path/to/condition_rows.csv
```

检查 Python 环境。默认脚本使用：

```bash
export SUCC_PYTHON_BIN=${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}
```

这个 Python 至少要能 import：

```bash
$SUCC_PYTHON_BIN - <<'PY'
import torch
import numpy
import PIL
import rdkit
import transformers
print("basic env ok")
PY
```

如果要跑最终 image-to-structure benchmark，还要检查 MolScribe：

```bash
export SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth
export SUCC_MOLSCRIBE_WORKDIR=${SUCC_MOLSCRIBE_WORKDIR:-"Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"}
test -f "$SUCC_MOLSCRIBE_MODEL"

PYTHONPATH="$SUCC_MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}" $SUCC_PYTHON_BIN - <<'PY'
from molscribe import MolScribe
print("molscribe import ok")
PY
```

如果 MolScribe 不是 pip 包，而是在某个 checkout 里，额外设置到包含
`molscribe/` 包的目录；对本仓库的 SketchMol checkout，默认就是：

```bash
export SUCC_MOLSCRIBE_WORKDIR="Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"
```

检查大 VLM 模型路径：

```bash
export SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct
test -d "$SUCC_HF_MODEL_NAME_OR_PATH"
```

如果 VLM features 已经导出过，可以不再检查/加载大模型，直接跳过 feature export：

```bash
export SUCC_RUN_FEATURE_EXPORT=0
export SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_source_neighbor_v1
test -f "$SUCC_CONDITION_FEATURES_DIR/query_tokens.npy"
test -f "$SUCC_CONDITION_FEATURES_DIR/index.csv"
```

检查 quota，尤其是 inode：

```bash
df -h /scratch/bdong
df -i /scratch/bdong
```

这条 pipeline 默认不会生成海量训练 PNG；只有 eval 生成图像会落盘，数量由 `SUCC_MAX_DECODE_IMAGES` 控制，默认等于 `SUCC_EVAL_LIMIT`。

## 2. 最推荐的一键运行

更新到 source-neighbor 数据集后，直接跑 v2-style source-neighbor launcher：

```bash
bash SketchMol-Understanding-Condition/scripts/submit_univideo_source_neighbor_v2_pipeline.sh
```

默认设置：

```text
SUCC_LATENT_BACKEND=image_vae
SUCC_LATENT_TARGET_MODE=residual
SUCC_DIFFUSION_OBJECTIVE=pred_x0
SUCC_IMAGE_VAE_INK_LOSS_WEIGHT=4.0
SUCC_RUN_IMAGE_VAE_TRAIN=0 if old v2 image VAE exists, otherwise auto
SUCC_RUN_FEATURE_EXPORT=auto
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=prepare
SUCC_RUN_MOLSCRIBE_OCR=0
SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER=1
SUCC_EVAL_LIMIT=1000
SUCC_MAX_DECODE_IMAGES=$SUCC_EVAL_LIMIT
SUCC_SAMPLE_ETA=0.0
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_source_neighbor_v2_residual_ink
```

含义是：

```text
如果没有 condition features，就导出 HF VLM features；
如果旧 v2 ink-aware image VAE checkpoint 存在，就复用它，否则自动训练 VAE；
训练 UniVideo-style generator；
decode eval latents 为 molecule images；
只准备 image_path.csv；
训练 job 成功后，自动提交 OCR-free materialized benchmark。
```

这版优先修 blank-image collapse。SUCC-local VAE 训练会提高黑色分子骨架像素权重，
并额外加入 ink/stroke loss，metrics 里会记录 foreground/background/blank-canvas/ink 指标。
UniVideo diffusion 默认使用 `pred_x0` 目标、residual edit latent
(`target_latent - source_latent`) 和 deterministic DDIM sampling (`SUCC_SAMPLE_ETA=0.0`)。
residual 模式会在采样后把预测 edit residual 加回 source latent；这样生成 latent 更容易留在
source molecule 的 VAE 流形上，避免 latent cosine 看起来不错但 decoder 输出白图。

如果要做 ablation，把 generation stream 切回旧的 absolute target latent：

```bash
SUCC_LATENT_TARGET_MODE=absolute \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

## 3. 如果已经有 VLM features

可以跳过大模型 feature export，节省时间：

```bash
SUCC_RUN_FEATURE_EXPORT=0 \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

## 4. 如果要接真实 SketchMol VAE

默认的 `image_vae` 会训练 SUCC-local VAE。若服务器上有原始 SketchMol VAE 或完整 SketchMol diffusion checkpoint，可以切到：

```bash
SUCC_LATENT_BACKEND=sketchmol_vae \
SUCC_SKETCHMOL_ROOT="Research/Molecule Generation/SketchMol/SketchMol-v1-main" \
SUCC_SKETCHMOL_VAE_CONFIG="Research/Molecule Generation/SketchMol/SketchMol-v1-main/configs/autoencoder/autoencoder_kl_pubchem400w_32x32x4.yaml" \
SUCC_SKETCHMOL_VAE_CHECKPOINT=/scratch/bdong/models/sketchmol/autoencoder_kl_pubchem400w_32x32x4.ckpt \
SUCC_RUN_FEATURE_EXPORT=0 \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

运行前检查：

```bash
test -d "$SUCC_SKETCHMOL_ROOT"
test -f "$SUCC_SKETCHMOL_VAE_CONFIG"
test -f "$SUCC_SKETCHMOL_VAE_CHECKPOINT"

SKETCHMOL_ROOT_FOR_CHECK="$SUCC_SKETCHMOL_ROOT" "$SUCC_PYTHON_BIN" - <<'PY'
import os
import sys
root = os.environ["SKETCHMOL_ROOT_FOR_CHECK"]
sys.path.insert(0, root)
import omegaconf
import ldm.models.autoencoder
print("sketchmol vae env ok")
PY
```

`SUCC_SKETCHMOL_VAE_CHECKPOINT` 可以是单独 AutoencoderKL checkpoint，也可以是完整 SketchMol diffusion checkpoint；adapter 会自动抽取 `first_stage_model.*` 权重。

## 5. 只准备 image_path.csv，不跑 OCR

如果 MolScribe 环境还没准备好，先生成给 MolScribe 用的 CSV：

```bash
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=prepare \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

输出：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
  univideo_molecule/image_structure_benchmark/image_path.csv
```

之后可以单独跑 OCR 和评估：

```bash
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
IMAGE_CSV=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/univideo_molecule/image_structure_benchmark/image_path.csv

$SUCC_PYTHON_BIN SketchMol-Understanding-Condition/scripts/run_molscribe_ocr.py \
  --model-path /scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
  --image-csv "$IMAGE_CSV" \
  --batch-size 16 \
  --device cuda

$SUCC_PYTHON_BIN SketchMol-Understanding-Condition/scripts/evaluate_univideo_image_benchmark.py \
  --image-csv "$IMAGE_CSV" \
  --output-dir SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/univideo_molecule/image_structure_benchmark \
  --method univideo_image_vae \
  --source-tanimoto-thresholds 0.4,0.6,0.8
```

## 6. OCR 跑不通时对齐 Unified 3M materialized benchmark

如果服务器 MolScribe / OCR 环境暂时跑不通，推荐直接跑 OCR-free
materialized benchmark。这个流程和 `SketchMol-Unified-3MDiffusion` 的
testing 方法对齐：当前默认先从 MolEdit eval JSONL + `predictions.csv` 导出
`benchmark_condition_rows.csv`，再结合 eval latent materialize `generated_smiles`，
最后用同一个 image benchmark evaluator 按 `method` 分组输出
`benchmark_report.md` / `benchmark_summary.csv` / `benchmark_decoded.csv`。

默认 `primary_fast` profile 会一次跑这些对照：

```text
source_identity
source_tanimoto_property_oracle
edit_latent_source_first_rerank
edit_latent_source_similarity_rerank
target_oracle
```

本地或交互节点直接跑：

```bash
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1 \
SUCC_MATERIALIZED_BENCHMARK_PROFILE=primary_fast \
bash SketchMol-Understanding-Condition/scripts/run_univideo_materialized_benchmark.sh
```

服务器上单独提交 CPU benchmark job：

```bash
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1 \
SUCC_MATERIALIZED_BENCHMARK_PROFILE=primary_fast \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_materialized_benchmark.sh
```

输出默认在：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/univideo_molecule/benchmark_materialized_primary_fast/
```

如果当前产物只有 benchmark condition rows，还没有 `generated_latents.npy` /
`target_latents.npy`，先用 `oracle` profile 做数据流 sanity check。旧的
`image_path.csv` 仍可通过 `SUCC_SOURCE_CSV` 显式指定，但不再是 MolEdit 主线默认：

```bash
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1 \
SUCC_MATERIALIZED_BENCHMARK_PROFILE=oracle \
bash SketchMol-Understanding-Condition/scripts/run_univideo_materialized_benchmark.sh
```

注意 `target_oracle` 会直接复制 row 里的 `target_smiles`，是上界/校验，不是模型预测。
`source_tanimoto_property_oracle` 是 candidate-library upper bound，用 source
Tanimoto 过滤后再按 active target properties 找最近 molecule。

需要只跑指定 method 时，用 `SUCC_MATERIALIZED_METHODS` 覆盖：

```bash
SUCC_MATERIALIZED_METHODS=source_identity,edit_latent_source_similarity_rerank,target_oracle \
bash SketchMol-Understanding-Condition/scripts/run_univideo_materialized_benchmark.sh
```

## 7. 输出文件怎么看

主输出目录：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/
```

关键文件：

```text
dataset/summary.json
  数据集导出摘要。

molecule_image_vae/molecule_image_vae.pt
molecule_image_vae/metrics.json
molecule_image_vae/reconstructions/*.png
  SUCC-local VAE 的 checkpoint、重建指标和重建样例。

univideo_molecule/univideo_molecule_generation.pt
univideo_molecule/metrics.json
  UniVideo-style generator checkpoint 和训练配置/history。

univideo_molecule/eval_latent/metrics.json
univideo_molecule/eval_latent/predictions.csv
univideo_molecule/eval_latent/generated_images/*.png
  latent-space 评估和生成图片。

univideo_molecule/eval_latent/source_oracle_images/*.png
univideo_molecule/eval_latent/target_oracle_images/*.png
  直接 decode source / target VAE latents 的诊断图片；用于判断 VAE 本身是否能画出结构。

univideo_molecule/image_structure_benchmark/image_path.csv
  MolScribe 输入/输出 CSV；OCR 后会包含 SMILES 和 molscribe_score。

univideo_molecule/image_structure_benchmark/benchmark_decoded.csv
  每条样本的 generated SMILES、validity、property error、source Tanimoto。

univideo_molecule/image_structure_benchmark/benchmark_summary.csv
  可用于画表的汇总 CSV。

univideo_molecule/image_structure_benchmark/benchmark_report.md
  最终报告主表。
```

报告里优先看：

```text
Main 2p-7p Table:
  validity all
  2p strict ... 7p strict

Source-Similarity-Constrained Success:
  mean source Tani
  median source Tani
  strict@0.4 / strict@0.6 / strict@0.8

Diagnostics:
  molecule_image_vae eval_target_ink_fraction
  molecule_image_vae eval_reconstruction_ink_fraction
  eval_latent decoded_image_quality.target_oracle.nonwhite_fraction
  eval_latent decoded_image_quality.generated.nonwhite_fraction
  OCR present
  valid
  source scaffold
  unique valid
  exact target
  source identity
```

注意：`strict` 是按所有生成样本算的，OCR 失败或 RDKit invalid 都算失败；`success_rate_strict_in_valid_mols` 只在 valid 分子内部算，用于诊断，不作为唯一主结论。

先看 VAE/oracle 诊断，再看 OCR：如果 `target_oracle.nonwhite_fraction` 仍接近 0，
说明 VAE 还在白图坍缩；如果 target oracle 有结构但 generated 仍接近 0，问题转向
UniVideo diffusion；只有 generated images 有明显非白像素后，OCR success 才有解释意义。

## 当前状态（精简）

Canonical 输出目录：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
```

历史 job 对比表已从本 README 移除。跑分看
`univideo_molecule/image_structure_benchmark/benchmark_report.md` 或
`univideo_molecule/benchmark_materialized_primary_fast/benchmark_report.md`。

OCR 相关修复仍适用：默认用 wrapper/custom OCR + `onmt220` overlay。

## 8. 查看 Slurm 状态

提交后脚本会打印 job id：

```text
Pipeline submitted.
  univideo_molecule_job=...
```

查看队列：

```bash
squeue -u $USER
```

查看日志：

```bash
tail -f SketchMol-Understanding-Condition/logs/succ-univideo-mol-<job_id>.log
```

如果 job pending 很久，先看：

```bash
squeue -j <job_id> -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

## 9. 常见问题

如果报 `SUCC_HF_MODEL_NAME_OR_PATH is required`：

```text
原因：condition features 不存在，同时没有提供 Qwen2.5-VL 路径。
处理：设置 SUCC_HF_MODEL_NAME_OR_PATH，或设置 SUCC_RUN_FEATURE_EXPORT=0 并确认 features 已存在。
```

如果报 `MolScribe model missing`：

```text
原因：要跑最终 image-to-structure benchmark，但找不到 MolScribe checkpoint。
处理：设置 SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth。
```

如果报 `from molscribe import MolScribe` 失败：

```text
原因：当前 Python 环境没有 MolScribe，或 MolScribe checkout 不在 PYTHONPATH。
处理：设置 SUCC_MOLSCRIBE_WORKDIR=Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate，
或换成已经安装 MolScribe 的 Python。
```

如果 `generated_images/` 很少：

```text
原因：SUCC_MAX_DECODE_IMAGES 太小，或 eval rows 太少。
处理：设置 SUCC_EVAL_LIMIT 和 SUCC_MAX_DECODE_IMAGES，例如 1000。
```

如果 2p-7p strict 很差：

```text
先看 generated_images 是否像分子图；
再看 OCR present / validity；
如果图像差，优先调 VAE/decoder；
如果图像可读但性质差，再调 condition connector / diffusion training。
```

如果 strict 高但 source Tanimoto 很低：

```text
说明模型更像 property generation/retrieval，不是 source-conditioned edit。
报告时要同时列 strict@Tanimoto>=0.4/0.6/0.8。
```

## 10. 最小推荐配置

第一次完整跑建议：

```bash
SUCC_EVAL_LIMIT=1000 \
SUCC_MAX_DECODE_IMAGES=1000 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_source_neighbor_v2_pipeline.sh
```

如果只是快速检查链路，不想等太久：

```bash
SUCC_EDIT_LIMIT=2000 \
SUCC_TRAIN_LIMIT=2000 \
SUCC_EVAL_LIMIT=64 \
SUCC_MAX_DECODE_IMAGES=64 \
SUCC_STAGE1_EPOCHS=1 \
SUCC_STAGE2_EPOCHS=1 \
SUCC_STAGE3_EPOCHS=0 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_source_neighbor_v2_pipeline.sh
```

快速检查只用来确认 pipeline，不用于报告主结果。
