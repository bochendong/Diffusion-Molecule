# Diffusion Molecule

这个仓库用于 source-conditioned multi-property molecular editing。目标是从
source molecule image / SMILES 和自然语言编辑目标出发，生成一个满足 2-7 个性质约束、
同时仍然和 source 结构相似的 target molecule。

## 目录结构

```text
SketchMol-MultiProperty-EditDataset
  构建 source-target edit pair、condition rows、diffusion_edit_manifest.csv，
  并做 multi-property / source Tanimoto 评估。

SketchMol-Understanding-Condition
  主线 understanding stream：HF VLM feature export、connector、rerank baseline、
  UniVideo-style molecule generation pipeline。

SketchMol-Unified-3MDiffusion
  独立 3M-style unified 路线：description pretraining + edit-generation rows
  -> alignment -> edit tokens -> latent diffusion。

SketchMolBenchmark
  真实 SketchMol + MolScribe/OCR baseline，以及直接预测结果 materialization。

Research/Molecule Generation/SketchMol/SketchMol-v1-main
  官方 SketchMol vendored repo，只作为 benchmark 外部依赖保留。
```

## 研究主线

当前主线是双流：

```text
Understanding stream:
  source molecule image + instruction -> edit condition / edit latent

Generation stream:
  source-conditioned diffusion -> target molecule
```

主结果不要只看 property strict success，也不要只看 scaffold match。报告时至少同时看：

```text
2p-7p strict success
mean / median source Tanimoto
strict@Tanimoto>=0.4 / 0.6 / 0.8
validity
```

如果 strict success 高但 source Tanimoto 低，它更像 property retrieval，不是真正的
source-conditioned editing。

## 服务器入口

在 Nibi/Slurm login node 上：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main
```

## Python 环境与 venv

这个仓库没有单一 venv 覆盖所有任务。按 pipeline 选对 Python，比盲目统一到一个
interpreter 更省事。

### 1. `molscribe_overlay`（主线默认）

```bash
export SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
export SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
```

用途：

```text
SketchMol-Understanding-Condition 的 HF VLM / UniVideo / benchmark export
SketchMol-Unified-3MDiffusion 的 unified 3M training
SketchMolBenchmark 的 MolScribe OCR（配合 SKETCHMOL_MOLSCRIBE_WORKDIR）
```

要求能同时 import `torch`、`transformers`、`PIL`、`rdkit`。HF VLM 和
UniVideo submit 脚本会在 job 开始时做早期检查。

注意：不要把当前大 VLM pipeline 指到没有 RDKit 的 `phystabmol` venv。

### 2. `phystabmol`（Torch 训练 / SketchMol 采样）

```bash
export SKETCHMOL_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python
export SKETCHMOL_BENCHMARK_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python
```

用途：

```text
SketchMol diffusion 采样（真实 SketchMol baseline）
部分旧版 CNN / fusion encoder 训练
SketchMolBenchmark materialization / smoke
```

特点：有 Torch，但没有 RDKit。pair mining、scaffold-preservation evaluation、
dataset build 不要单独依赖它。

### 3. Compute Canada module 环境（RDKit 数据构建）

`SketchMol-MultiProperty-EditDataset` 和不少 understanding 脚本会先加载：

```text
StdEnv/2023
python/3.11
rdkit/2025.09.4
```

只有当你指定的 Python 已经能 import RDKit 时，才覆盖 `SMMED_PYTHON_BIN` 或
`SUCC_PYTHON_BIN`。

### 4. MolScribe 不是 pip 包

MolScribe 类来自 vendored SketchMol checkout，不在 venv 里 pip install。跑 OCR 时
要把 `evaluate/` 加进 `PYTHONPATH`：

```bash
export SUCC_MOLSCRIBE_WORKDIR="Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"
export SKETCHMOL_MOLSCRIBE_WORKDIR="$SUCC_MOLSCRIBE_WORKDIR"
export SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth
```

UniVideo pipeline 现在会自动检测上面的 SketchMol `evaluate/` 目录；SketchMolBenchmark
paper repro 也使用同一 workdir。

### 5. 快速自检

主线 Python：

```bash
$SUCC_PYTHON_BIN - <<'PY'
import torch, transformers, PIL, rdkit
print("molscribe_overlay ok")
PY
```

MolScribe OCR：

```bash
PYTHONPATH="$SUCC_MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}" $SUCC_PYTHON_BIN - <<'PY'
from molscribe import MolScribe
print("molscribe import ok")
PY
```

SketchMol 采样 Python：

```bash
$SKETCHMOL_PYTHON_BIN - <<'PY'
import torch
print("phystabmol torch ok")
PY
```

### 6. 环境变量对照

```text
SUCC_PYTHON_BIN              Understanding-Condition 主线
SMU3M_PYTHON_BIN             Unified-3MDiffusion 主线
SKETCHMOL_PYTHON_BIN         SketchMol diffusion 采样
SKETCHMOL_MOLSCRIBE_PYTHON_BIN   MolScribe OCR；默认可与 SUCC_PYTHON_BIN 相同
SKETCHMOL_BENCHMARK_PYTHON_BIN   benchmark materialization / decode 评估
SMMED_PYTHON_BIN             multi-property dataset build
SUCC_MOLSCRIBE_WORKDIR       UniVideo / understanding OCR 的 MolScribe PYTHONPATH
SKETCHMOL_MOLSCRIBE_WORKDIR  SketchMolBenchmark OCR 的 MolScribe PYTHONPATH
```

## Slurm 资源怎么选

先用短时间交互节点做环境测试，测试完 `Ctrl+C` 或 `exit` 退出节点，再用 `sbatch`
或本仓库的 `submit_*.sh` 正式提交。

```bash
salloc \
  --account=def-hup-ab \
  --gpus=h100_2g.20gb:1 \
  --cpus-per-task=16 \
  --mem-per-cpu=4G \
  --time=00:30:00
```

参数参考：

```text
--account
  rg-hup      funded resources
  def-hup-ab  standard resources

--gpus
  h100_80gb:1       one whole H100 80G
  h100_1g.10gb:1    1/8 H100, 10G
  h100_2g.20gb:1    1/4 H100, 20G
  h100_3g.40gb:1    1/2 H100, 40G

--cpus-per-task
  16 is usually enough for one standard GPU job.
  If preprocessing is CPU-heavy, run that as a CPU job first.

--mem-per-cpu
  4096M or 4G is usually enough on this cluster.

--time
  For tests, keep it short.
```

如果 Slurm efficiency report 提示 `Less than 1 core was used`、`Less than 10% memory
was used`，下次不要继续申请同样资源。对 serial / light preprocessing job，优先降到：

```text
CPU: 1-2
RAM: 4G-16G
time: 按实际运行时长留余量
```

各 pipeline 默认资源大致如下，提交前先看 `submit_*.sh` 打印的参数，再按 efficiency
report 往下调：

```text
Understanding / UniVideo (SUCC_*)
  默认: 8 CPU, 80G RAM, h100_40gb_mig
  覆盖: SUCC_SLURM_CPUS, SUCC_SLURM_MEM, SUCC_SLURM_TIME, SUCC_GPU_PROFILE, SUCC_SLURM_GPUS

Unified 3M (SMU3M_*)
  默认: economy profile + h100_20gb_mig
  覆盖: SMU3M_SLURM_CPUS, SMU3M_SLURM_MEM, SMU3M_SLURM_TIME, SMU3M_GPU_PROFILE, SMU3M_RESOURCE_PROFILE

Dataset build / smoke (SMMED_* / succ-smoke)
  默认: 1-2 CPU, 8G RAM, 无 GPU 或短 GPU
  覆盖: SMMED_SLURM_CPUS, SMMED_SLURM_MEM, SUCC_SLURM_CPUS, SUCC_SLURM_MEM
```

如果申请 40GB GPU，就应该用对应的 throughput profile，把 batch 和模型宽度调大；
否则优先用 20GB MIG，避免拿 40GB 跑小 batch。Understanding / UniVideo 因为还要跑
7B VLM feature export，通常保留 40GB；只有纯训练、纯 OCR resume 或明确 smoke 时，
再考虑降到 20GB MIG。

脚本里通常可以用环境变量覆盖资源。例如省资源跑法：

```bash
SMU3M_GPU_PROFILE=h100_20gb_mig \
SMU3M_RESOURCE_PROFILE=economy \
SMU3M_SLURM_TIME=08:00:00 \
SMU3M_SLURM_CPUS=2 \
SMU3M_SLURM_MEM=16G \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

40GB 吞吐跑法：

```bash
SMU3M_GPU_PROFILE=h100_40gb_mig \
SMU3M_RESOURCE_PROFILE=throughput_40gb \
SMU3M_SLURM_CPUS=4 \
SMU3M_SLURM_MEM=40G \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

也可以直接指定 GPU request：

```bash
SMU3M_SLURM_GPUS=h100_2g.20gb:1 \
SMU3M_RESOURCE_PROFILE=economy \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

## 常用 Pipeline

### 1. 只构建 multi-property edit dataset

默认输入表：

```text
SketchMol-MultiProperty-EditDataset/data/train_table.csv
```

大规模构建时不要渲染海量 pair PNG，避免 inode / quota 爆掉：

```bash
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

如果只是当前 shell 里直接跑：

```bash
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/run_build_dataset.sh
```

换外部数据源：

```bash
SMMED_INPUT_CSV=/path/to/train_table.csv \
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

### 2. 跑主线 HF VLM understanding pipeline

这个 pipeline 会构建数据集、导出 `diffusion_edit_manifest.csv`、跑 HF VLM features、
训练 edit connector，并输出 SketchMol-style report。

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
bash SketchMol-Understanding-Condition/scripts/submit_hf_vlm_multiproperty_pipeline.sh
```

关键输出：

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm/benchmark_report.md
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm_edit_connector/benchmark_report.md
```

### 3. 跑 UniVideo-style generation stream

如果已经有 condition rows / VLM features，可以提交 source-conditioned molecule generation：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
SUCC_CONDITION_ROWS=SketchMol-Understanding-Condition/outputs/multiproperty_100k_v1/condition_rows.csv \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

如果 VLM features 已经导出过：

```bash
SUCC_RUN_FEATURE_EXPORT=0 \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm \
SUCC_CONDITION_ROWS=SketchMol-Understanding-Condition/outputs/multiproperty_100k_v1/condition_rows.csv \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

如果前面的训练和图片生成已经完成，只需要续跑 MolScribe OCR / image benchmark，
直接跑 OCR 和评估脚本，不要重提整条 pipeline（否则会重新训练 UniVideo）：

```bash
export SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
export SUCC_MOLSCRIBE_WORKDIR="Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"
IMAGE_CSV=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1/univideo_molecule/image_structure_benchmark/image_path.csv

PYTHONPATH="$SUCC_MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}" \
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

### 4. 跑 unified 3M training

这是独立的 3M-style unified 路线。默认会做 preflight、要求 CUDA、按 epoch 保存
checkpoint，并在重新运行时自动 resume。默认 resource profile 是 `economy`，也就是
20GB MIG + 大 batch，而不是浪费 40GB GPU。

```bash
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

如果明确要用 40GB GPU，就用 throughput profile：

```bash
SMU3M_GPU_PROFILE=h100_40gb_mig \
SMU3M_RESOURCE_PROFILE=throughput_40gb \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

如果服务器上没有 `PubChem324k` 数据：

```bash
SMU3M_INCLUDE_PUBCHEM=0 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

默认输出：

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v1/
  dataset/summary.json
  alignment/alignment_model.pt
  alignment/checkpoints/latest.pt
  edit_condition_tokens/edit_condition_connector.pt
  edit_condition_tokens/checkpoints/latest.pt
  latent_diffusion/latent_diffusion_generation.pt
  latent_diffusion/checkpoints/latest.pt
  eval_latent/metrics.json
```

### 5. 跑 SketchMol baseline / OCR benchmark

真实 SketchMol baseline 依赖 vendored SketchMol repo 和 MolScribe 环境。提交前确认 checkpoint
路径不是占位符。

```bash
bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh
```

如果 diffusion 已经生成了 `image_path.csv`，只需要继续 OCR / materialization：

```bash
bash SketchMolBenchmark/scripts/resume_real_sketchmol_ocr.sh
```

### 6. 汇总或复用已有结果

Multi-property dataset 自带 full benchmark wrapper：

```bash
bash SketchMol-MultiProperty-EditDataset/scripts/submit_full_benchmark.sh
```

Understanding-condition 结果导出到 SketchMolBenchmark：

```bash
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
SUCC_VARIANT=full \
bash SketchMol-Understanding-Condition/scripts/run_benchmark_export.sh
```

## 本地检查

本机不一定有 `torch` / `rdkit`，所以本地主要做语法和脚本检查：

```bash
python3 -m py_compile SketchMol-Unified-3MDiffusion/sketchmol_unified_3m_diffusion/*.py \
  SketchMol-Unified-3MDiffusion/scripts/*.py

bash -n SketchMol-Unified-3MDiffusion/scripts/run_unified_generation_smoke.sh \
  SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

真正的 CUDA、Torch、3M data、MolScribe 和 manifest 检查在服务器 job 的 preflight
或 submit wrapper 里完成。
