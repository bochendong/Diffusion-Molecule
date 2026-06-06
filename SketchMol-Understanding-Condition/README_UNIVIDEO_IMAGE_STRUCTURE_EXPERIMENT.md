# UniVideo Image-to-Structure 完整实验 README

这份文档只说明当前主线实验怎么跑：

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
test -f SketchMol-Understanding-Condition/outputs/multiproperty_100k_v1/condition_rows.csv
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
test -f "$SUCC_MOLSCRIBE_MODEL"

$SUCC_PYTHON_BIN - <<'PY'
from molscribe import MolScribe
print("molscribe import ok")
PY
```

如果 MolScribe 不是 pip 包，而是在某个 checkout 里，额外设置：

```bash
export SUCC_MOLSCRIBE_WORKDIR=/path/to/MolScribe
```

检查大 VLM 模型路径：

```bash
export SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct
test -d "$SUCC_HF_MODEL_NAME_OR_PATH"
```

如果 VLM features 已经导出过，可以不再检查/加载大模型，直接跳过 feature export：

```bash
export SUCC_RUN_FEATURE_EXPORT=0
export SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm
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

如果使用 SUCC-local molecule-image VAE，直接跑：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

默认设置：

```text
SUCC_LATENT_BACKEND=image_vae
SUCC_RUN_IMAGE_VAE_TRAIN=auto
SUCC_RUN_FEATURE_EXPORT=auto
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=auto
SUCC_EVAL_LIMIT=1000
SUCC_MAX_DECODE_IMAGES=$SUCC_EVAL_LIMIT
```

含义是：

```text
如果没有 condition features，就导出 HF VLM features；
如果没有 molecule_image_vae.pt，就先训练 VAE；
训练 UniVideo-style generator；
decode eval latents 为 molecule images；
跑 MolScribe OCR；
跑 RDKit / SketchMol-style benchmark。
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
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/
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

## 6. 输出文件怎么看

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
  OCR present
  valid
  source scaffold
  unique valid
  exact target
  source identity
```

注意：`strict` 是按所有生成样本算的，OCR 失败或 RDKit invalid 都算失败；`success_rate_strict_in_valid_mols` 只在 valid 分子内部算，用于诊断，不作为唯一主结论。

## 7. 查看 Slurm 状态

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

## 8. 常见问题

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
处理：设置 SUCC_MOLSCRIBE_WORKDIR=/path/to/MolScribe，或换成 molscribe_overlay Python。
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

## 9. 最小推荐配置

第一次完整跑建议：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
SUCC_EVAL_LIMIT=1000 \
SUCC_MAX_DECODE_IMAGES=1000 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

如果只是快速检查链路，不想等太久：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
SUCC_EDIT_LIMIT=2000 \
SUCC_TRAIN_LIMIT=2000 \
SUCC_EVAL_LIMIT=64 \
SUCC_MAX_DECODE_IMAGES=64 \
SUCC_STAGE1_EPOCHS=1 \
SUCC_STAGE2_EPOCHS=1 \
SUCC_STAGE3_EPOCHS=0 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

快速检查只用来确认 pipeline，不用于报告主结果。
