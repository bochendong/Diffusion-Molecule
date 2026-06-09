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

SMILES-DualStream-EditorAtomas
  独立纯 SMILES 双流路线：SMI-Editor-style edit reconstruction +
  Atomas-style token/fragment/molecule hierarchical alignment，不使用图像。

SketchMolBenchmark
  真实 SketchMol + MolScribe/OCR baseline，以及直接预测结果 materialization。

Research/Molecule Generation/SketchMol/SketchMol-v1-main
  官方 SketchMol vendored repo，只作为 benchmark 外部依赖保留。

Research/Molecule Generation/MolEditRL
  MolEditRL / MolEdit-Instruct benchmark 资产登记；cluster 上 raw + enhanced_v1 已构建，详见 MolEditRL/README.md 与 datasets/README.md。

datasets
  线上数据集布置说明；本地只提交 README，不提交大数据。
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
export SDEA_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
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

### 4. MolScribe OCR 环境（vendored + onmt220）

MolScribe 类来自 vendored SketchMol checkout，不在 venv 里 pip install。`molscribe_overlay`
还会继承 `phystabmol` 自带的旧版 OpenNMT；**如果不额外 prepend onmt220 overlay，graph→SMILES
会全部返回空字符串**（OCR present = 0%）。

仓库里唯一确认成功的 OCR run 输出目录
`SketchMolBenchmark/outputs/paper_repro_mw400_real_official_ocr`（OCR present 100%，validity 90%）。
它使用的组合是：

```text
Python:   /home/bdong/.venvs/molscribe_overlay/bin/python
ONMT:     /scratch/bdong/python_overlays/onmt220          # 必须在 PYTHONPATH 最前
MolScribe: SketchMol-v1-main/evaluate/molscribe           # vendored，不是 pip 包
OCR 脚本:  evaluate/predict_csv.py                         # 原版，不做额外二值化
GPU:      必须（login 节点 CPU 上不可靠）
```

推荐环境变量：

```bash
export SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
export SKETCHMOL_MOLSCRIBE_PYTHON_BIN="$SUCC_PYTHON_BIN"
export SUCC_MOLSCRIBE_WORKDIR="Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"
export SKETCHMOL_MOLSCRIBE_WORKDIR="$SUCC_MOLSCRIBE_WORKDIR"
export SUCC_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth
export SUCC_ONMT_OVERLAY=/scratch/bdong/python_overlays/onmt220
export SKETCHMOL_ONMT_OVERLAY="$SUCC_ONMT_OVERLAY"
```

Understanding-Condition 和 SketchMolBenchmark 的 OCR shell 脚本会 source
`SketchMol-Understanding-Condition/scripts/molscribe_env.sh`，自动把 **onmt220 → evaluate/**
 按顺序 prepend 到 `PYTHONPATH`。

不要对 SketchMol 扩散图使用 Understanding-Condition 里额外的 `--preprocess-images`
二值化；默认应与 `predict_csv.py` 一致（`--no-preprocess-images`）。

如果 diffusion 已经采样完、只需要重跑 OCR：

```bash
SKETCHMOL_BENCHMARK_SOURCE_CSV=/path/to/image_path.csv \
SKETCHMOL_MOLSCRIBE_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
SKETCHMOL_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
bash SketchMolBenchmark/scripts/resume_real_sketchmol_ocr.sh
```

UniVideo 的 OCR rerun 默认走 wrapper/custom/raw-token fallback；详见
`SketchMol-Understanding-Condition/README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md`。

### 5. 快速自检

主线 Python：

```bash
$SUCC_PYTHON_BIN - <<'PY'
import torch, transformers, PIL, rdkit
print("molscribe_overlay ok")
PY
```

MolScribe OCR（必须同时检查 vendored molscribe 和 onmt220 overlay）：

```bash
# shellcheck source=SketchMol-Understanding-Condition/scripts/molscribe_env.sh
source SketchMol-Understanding-Condition/scripts/molscribe_env.sh
PYTHON_BIN="$SUCC_PYTHON_BIN" prepend_molscribe_pythonpath
PYTHON_BIN="$SUCC_PYTHON_BIN" check_molscribe_import
```

期望输出包含：

```text
molscribe import ok (.../SketchMol-v1-main/evaluate/molscribe/__init__.py)
onmt import ok (/scratch/bdong/python_overlays/onmt220/onmt/__init__.py)
```

如果 `onmt` 来自 `phystabmol/lib/python3.11/site-packages/onmt/`，OCR 很可能全是空 SMILES。

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
SUCC_ONMT_OVERLAY              MolScribe OCR 用的 OpenNMT overlay（默认 onmt220）
SKETCHMOL_ONMT_OVERLAY         同上，SketchMolBenchmark 别名
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
  默认: 1 CPU, 8G-12G RAM, 无 GPU 或短 GPU
  覆盖: SMMED_SLURM_CPUS, SMMED_SLURM_MEM, SUCC_SLURM_CPUS, SUCC_SLURM_MEM

Materialized benchmark / OCR resume
  默认: 1 CPU, 8G-16G RAM, 1-4h
  覆盖: SMU3M_BENCHMARK_SLURM_CPUS, SMU3M_BENCHMARK_SLURM_MEM, SUCC_SLURM_CPUS, SKETCHMOL_SLURM_CPUS
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

主线已切换到 **MolEdit-Instruct**（text/SMILES source-conditioned editing）。
Image / multi-property pipeline 仍保留作并行对照。数据布局见 `datasets/README.md`。

### 1. MolEdit-Instruct：Unified 3M 训练 + 表格指标（默认）

数据已在 cluster 上构建：

```text
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv
```

本地 smoke：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/run_unified_moledit_smoke.sh
```

集群训练：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_pipeline.sh
```

默认输出：

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/
```

训练后跑 MolEditRL 表格指标：

```bash
python SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py \
  --reference "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv" \
  --predictions SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/benchmark_materialized_primary_fast/benchmark_decoded.csv \
  --method edit_latent_source_similarity_rerank \
  --output-dir SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/moledit_table_metrics
```

详见 `SketchMol-Unified-3MDiffusion/README.md` 与 `Research/Molecule Generation/MolEditRL/README.md`。

### 2. UniVideo image pipeline + OCR-free materialized benchmark

完整运行手册：
`SketchMol-Understanding-Condition/README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md`

Canonical 输出：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
```

OCR 跑不通时：

```bash
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink \
SUCC_MATERIALIZED_BENCHMARK_PROFILE=primary_fast \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_materialized_benchmark.sh
```

### 3. Legacy：multi-property dataset + HF VLM + 旧 Unified 3M

Multi-property 数据集（image / VLM 仍依赖）：

```bash
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

HF VLM understanding pipeline：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
bash SketchMol-Understanding-Condition/scripts/submit_hf_vlm_multiproperty_pipeline.sh
```

旧 Unified 3M multi-property 训练：

```bash
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_generation_pipeline.sh
```

Materialized benchmark（multi-property eval）：

```bash
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1 \
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh
```

### 4. SketchMol baseline / OCR

真实 SketchMol baseline 依赖 vendored SketchMol、**onmt220 overlay** 和
`molscribe_overlay` venv：

```bash
SKETCHMOL_MOLSCRIBE_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
SKETCHMOL_MOLSCRIBE_MODEL=/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth \
bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh
```

UniVideo OCR rerun（wrapper/custom）：

```bash
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
SUCC_MOLSCRIBE_RUNNER=wrapper \
SUCC_MOLSCRIBE_BACKEND=custom \
SUCC_RAW_SMILES_FALLBACK=1 \
bash SketchMol-Understanding-Condition/scripts/submit_succ_ocr_benchmark.sh
```

### 5. Pure-SMILES Editor-Atomas

```bash
SDEA_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SMILES-DualStream-EditorAtomas/scripts/submit_large_train.sh
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
