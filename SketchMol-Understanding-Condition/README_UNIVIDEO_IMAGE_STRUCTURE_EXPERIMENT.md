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
test -f SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv
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
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct \
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
SUCC_IMAGE_VAE_EPOCHS=10
SUCC_IMAGE_VAE_FOREGROUND_WEIGHT=8.0
SUCC_IMAGE_VAE_INK_LOSS_WEIGHT=4.0
SUCC_IMAGE_VAE_INK_FRACTION_WEIGHT=2.0
SUCC_DIFFUSION_OBJECTIVE=pred_x0
SUCC_LATENT_TARGET_MODE=residual
SUCC_SAMPLE_ETA=0.0
```

含义是：

```text
如果没有 condition features，就导出 HF VLM features；
如果没有 molecule_image_vae.pt，就先训练 VAE；
如果已有旧版 molecule_image_vae.pt 但 metrics 里没有 ink_loss_weight，会自动重训 VAE；
训练 UniVideo-style generator；
decode eval latents 为 molecule images；
跑 MolScribe OCR；
跑 RDKit / SketchMol-style benchmark。
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

## Latest Run Results

Artifacts live under `outputs/univideo_molecule_generation_v2_residual_ink/` (current)
and `outputs/univideo_molecule_generation_v1/` (previous ablation).

### Current best: job `15697119` (Jun 6 2026, ~2 h)

Residual edit latent (`SUCC_LATENT_TARGET_MODE=residual`), ink-aware VAE
(`foreground_weight=8.0`, `ink_loss_weight=4.0`, `ink_fraction_weight=2.0`),
`pred_x0` diffusion, full pipeline through MolScribe OCR benchmark.
Log: `logs/succ-univideo-mol-15697119.log`.

| Stage | Status |
| --- | --- |
| Image VAE retrain (10 ep) | eval loss 0.053 → **0.031**; ink L1 tracked |
| Stage1 connector | aux loss 1.63 → **1.47** |
| Stage2 diffusion | diffusion loss 1.54 → **0.94** |
| Stage3 multitask | diffusion loss **0.85** |

Latent eval on 1000 samples (`univideo_molecule/eval_latent/metrics.json`):

| Metric | `15692057` (absolute) | `15697119` (residual) |
| --- | ---: | ---: |
| `source_latent_cosine` | 0.620 | **0.980** |
| `target_latent_cosine` | 0.620 | **0.384** |
| `source_target_latent_cosine` | 0.463 | 0.389 |
| `latent_mae` / `latent_mse` | 0.528 / 1.057 | 0.752 / 2.135 |

Decoded image quality:

| | generated | target oracle |
| --- | ---: | ---: |
| `nonwhite_fraction` | **0.043** | 0.042 |
| `mean_intensity` | **0.976** | 0.976 |
| `dark_fraction` | **0.032** | 0.031 |

All 1000 decoded PNGs now contain visible molecular structure (median ~11 KB each;
v1 median ~802 bytes). Blank-image collapse is fixed. Residual mode keeps generated
latents near source (`source_latent_cosine` 0.98) but not near target (0.38 ≈
source-target baseline), so the model mostly copies source rather than performing
property edits.

Image-to-structure benchmark (`image_structure_benchmark/benchmark_report.md`):

| Metric | `15692057` | `15697119` |
| --- | ---: | ---: |
| OCR SMILES present | 0 / 1000 | 0 / 1000 |
| validity all | 0.0 | 0.0 |
| strict success (2p–7p) | 0.0 | 0.0 |
| MolScribe confidence (mean / max) | 0.075 / 0.45 | 0.073 / 0.47 |

Decode quality is solved; next bottlenecks are edit signal (raise `target_latent_cosine`)
and MolScribe-readable rendering.

Follow-up code fix after this run:

- `run_molscribe_ocr.py` now supports a vendored MolScribe raw-token fallback.
  If graph-to-SMILES conversion returns empty/invalid while the decoder token
  SMILES is non-empty, the script writes the raw token SMILES and records
  `molscribe_decode_source=raw_token_fallback`.
- Residual generation now trains the connector latent auxiliary head against the
  same edit residual used by diffusion (`target_latent - source_latent`) instead
  of the absolute target latent. This removes the previous auxiliary/diffusion
  mismatch that encouraged source-copy behavior.
- New sampling knobs: `SUCC_RESIDUAL_SAMPLE_SCALE` and
  `SUCC_CONNECTOR_LATENT_BLEND`.

OCR-only rerun on the existing `15697119` generated images:

```bash
IMAGE_CSV=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/univideo_molecule/image_structure_benchmark/image_path.csv

PYTHONPATH="Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate${PYTHONPATH:+:$PYTHONPATH}" \
$SUCC_PYTHON_BIN SketchMol-Understanding-Condition/scripts/run_molscribe_ocr.py \
  --model-path /scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
  --image-csv "$IMAGE_CSV" \
  --batch-size 16 \
  --device cuda

$SUCC_PYTHON_BIN SketchMol-Understanding-Condition/scripts/evaluate_univideo_image_benchmark.py \
  --image-csv "$IMAGE_CSV" \
  --output-dir SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/univideo_molecule/image_structure_benchmark \
  --method univideo_image_vae \
  --source-tanimoto-thresholds 0.4,0.6,0.8
```

Next training run for the residual auxiliary fix:

```bash
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v3_residual_aux \
SUCC_RUN_IMAGE_VAE_TRAIN=0 \
SUCC_IMAGE_VAE_CHECKPOINT=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt \
SUCC_RUN_FEATURE_EXPORT=0 \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm \
SUCC_LATENT_TARGET_MODE=residual \
SUCC_RESIDUAL_SAMPLE_SCALE=1.25 \
SUCC_CONNECTOR_LATENT_BLEND=0.1 \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

### Residual-ink ablation: job `15692057` (Jun 6 2026, ~2 h)

Foreground-aware VAE retrain (`foreground_weight=8.0`), absolute target latent,
`pred_x0` diffusion, full pipeline through MolScribe OCR benchmark.
Log: `logs/succ-univideo-mol-15692057.log`.
Output: `outputs/univideo_molecule_generation_v1/`.

| Stage | Status |
| --- | --- |
| Image VAE retrain (10 ep) | eval loss 0.017 → **0.011**; foreground L1 0.104 → **0.066** |
| Stage1 connector | aux loss 1.60 → **1.44** |
| Stage2 diffusion | diffusion loss 0.74 → **0.23** |
| Stage3 multitask | diffusion loss **0.21** |

Latent eval on 1000 samples (`univideo_molecule/eval_latent/metrics.json`):

| Metric | `15688319` | `15692057` |
| --- | ---: | ---: |
| `source_latent_cosine` | 0.003 | **0.620** |
| `target_latent_cosine` | 0.003 | **0.620** |
| `latent_mae` / `latent_mse` | 0.393 / 0.375 | 0.528 / 1.057 |

Decoded image quality:

| | generated | target oracle |
| --- | ---: | ---: |
| `nonwhite_fraction` | 0.004 | 0.042 |
| `mean_intensity` | 0.999 | 0.976 |
| `dark_fraction` | 0.0015 | 0.031 |

~85% of decoded PNGs are still near-blank (~774–802 bytes). 151 images have
>1% nonwhite pixels; 5 exceed 5% and show recognizable molecular skeletons.
MolScribe OCR ran successfully but produced no SMILES.

Image-to-structure benchmark (`image_structure_benchmark/benchmark_report.md`):

| Metric | Result |
| --- | ---: |
| OCR SMILES present | 0 / 1000 |
| validity all | 0.0 |
| strict success (2p–7p) | 0.0 |
| SketchMol reference strict (2p–7p) | 0.68–0.80 |
| MolScribe confidence (mean / max) | 0.075 / 0.45 |

Latent alignment improved dramatically, but VAE decode still collapses to white
for most samples. End-to-end strict success remains 0; bottleneck is now decode
quality rather than latent conditioning or OCR infrastructure.

### Upstream training (`15688319`, `succ-univideo-mol`)

Completed dataset export, image VAE, and 3-stage UniVideo training. Latent eval on
1000 samples (`univideo_molecule/eval_latent/metrics.json`):

| Metric | Value |
| --- | ---: |
| `source_latent_cosine` | 0.002 |
| `target_latent_cosine` | 0.003 |
| `latent_mae` / `latent_mse` | 0.393 / 0.375 |

Image VAE reconstruction loss stayed near 0.047; diffusion loss stayed near 1.0.
All 1000 decoded images are blank white 256×256 PNGs (~758 bytes each).

MolScribe OCR failed in this job (`ModuleNotFoundError: No module named 'molscribe'`)
because `SUCC_MOLSCRIBE_WORKDIR` was not set.

### OCR retry (`15690165`, `succ-univideo-mol-ocr-retry`)

MolScribe infrastructure fix verified: import OK, 1000 images OCR'd in ~3 min.
Log: `logs/succ-univideo-mol-ocr-retry-15690165.log`.

Benchmark report (`image_structure_benchmark/benchmark_report.md`):

| Metric | Result |
| --- | ---: |
| OCR SMILES present | 0 / 1000 |
| validity all | 0.0 |
| strict success (2p–7p) | 0.0 |
| SketchMol reference strict (2p–7p) | 0.68–0.80 |

OCR pipeline works, but benchmark is all zeros because upstream images are blank.
MolScribe confidence scores are ~0.076 with empty SMILES. Root cause is upstream
diffusion/VAE collapse, not OCR.

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

## 9. 最小推荐配置

第一次完整跑建议：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct \
SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=1 \
SUCC_EVAL_LIMIT=1000 \
SUCC_MAX_DECODE_IMAGES=1000 \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

如果只是快速检查链路，不想等太久：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct \
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
