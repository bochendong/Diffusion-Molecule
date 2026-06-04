# SketchMol Understanding-Condition Stream

## 0. 核心判断

这个方向可以做，但不能把故事讲成“旁边加一个 LLM”。比较稳的研究切入点是：

> 用 MLLM 把分子图像 + 自然语言编辑目标解析成一组可被 diffusion cross-attention 使用的 condition tokens，从而提升分子编辑里的 scaffold 保留、局部替换、性质优化和目标蛋白活性控制。

换句话说，大模型不负责直接生成 SMILES，也不只是输出一段 caption；它负责生成 conditional latent / condition token sequence。diffusion generator 仍然负责生成分子图像，MolScribe 或后处理再把图像还原成结构。

这和 UniVideo 的思路最像：

- UniVideo: image/video/text -> Qwen-VL hidden states/metaqueries -> video diffusion transformer。
- SketchMol: molecule image/sketch/text/property -> molecular understanding tokens -> latent diffusion UNet。

SketchMol 现有条件流已经是 cross-attention：`MixedEmbedderV2` 把离散/连续性质变成条件 token，配置里的 `context_dim` 是 256。因此最自然的改法不是推翻 SketchMol，而是把 MLLM condition tokens 接到同一个 cross-attention 接口里。

## 0.1 当前主线：不用小模型，改用大 VLM

当前主实验不再以小 CNN / hashed text proxy 作为方法主体。那些结果只保留为
pipeline sanity check 和 ablation 负控。

新的主线是：

```text
source molecule image + multi-property instruction
  -> frozen HuggingFace VLM hidden states
  -> molecular condition features / query tokens
  -> scaffold-aware multi-property retrieval benchmark
  -> SketchMol-style 2p-7p strict success table
```

一键提交大模型 workflow：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule
git pull origin main

SUCC_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SUCC_HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash SketchMol-Understanding-Condition/scripts/submit_hf_vlm_multiproperty_workflow.sh
```

如果集群不能直接联网下载 HuggingFace 模型，把
`SUCC_HF_MODEL_NAME_OR_PATH` 换成本地模型目录，例如：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct
```

默认输出：

```text
SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm/
  pooled.npy
  query_tokens.npy
  index.csv
  summary.json

SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/benchmark_hf_vlm/
  benchmark_report.md
  benchmark_summary.csv
  benchmark_decoded.csv
  metrics.json
```

`benchmark_report.md` 会直接包含：

```text
source_identity
vlm_scaffold_feature_retrieval
target_oracle
SketchMol structured reference
```

这里的 `vlm_scaffold_feature_retrieval` 才是大模型主结果：它用 frozen VLM
对 `source image + instruction` 的表示，在 train condition features 里检索同
scaffold target molecule，然后用 SketchMol 风格的 2-7 性质 strict success
评估。

快速 dry run 可以限制导出行数：

```bash
SUCC_LIMIT=2000 \
SMMED_LIMIT_EVAL_ROWS=200 \
SMMED_MAX_EVAL_PER_PROPERTY_COUNT=200 \
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/models/Qwen2.5-VL-7B-Instruct \
bash SketchMol-Understanding-Condition/scripts/submit_hf_vlm_multiproperty_workflow.sh
```

## 代码入口

这个目录现在是独立于 `Research` 的原型工程，不直接改动别人代码。

```text
SketchMol-Understanding-Condition/
  sketchmol_understanding_condition/
    encoders.py        # MLLM hidden -> SketchMol condition tokens
    pair_mining.py     # 同 scaffold source-target edit pair mining
    instructions.py    # 自然语言编辑指令模板
    evaluation.py      # validity / scaffold preservation / property success
    chem.py            # RDKit helper，按需导入
  scripts/
    build_edit_dataset.py
    mine_edit_pairs.py # 从 CSV 构造编辑训练对
    run_server_smoke.sh
    submit_server_smoke.sh
  configs/
    adapter_smoke.yaml # adapter-only 第一阶段实验配置
    compute_canada_env.sh
  tests/
    test_condition_encoder.py
```

第一版代码先固定最关键的接口：`MolecularQueryProjector` 输出 `[B, Nq, 256]`，`HybridConditionEncoder` 可以把它和 SketchMol 原来的 property tokens 拼接。后续接入 SketchMol 时，优先把这个 encoder 放进新的 `cond_stage_config`，而不是直接改 UNet 主体。

### Compute Canada 环境

RDKit 相关逻辑使用 Compute Canada 自带 module，不默认从 pip 安装。当前推荐固定版本：

```bash
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
```

已在集群上验证：

```text
Python: 3.11.5
RDKit: 2025.09.4
```

模型侧 encoder smoke 可以用已有 venv：

```bash
PYTHONPATH=SketchMol-Understanding-Condition \
  /home/bdong/scratch/venvs/phystabmol/bin/python -c "import torch; print(torch.__version__)"
```

注意：`phystabmol` 里有 Torch，但没有 RDKit；pair mining / scaffold preservation evaluation 应使用上面的 Compute Canada RDKit module 环境。

提交服务器 smoke job：

```bash
SketchMol-Understanding-Condition/scripts/submit_server_smoke.sh
```

如果 Slurm controller 暂时不可用，可以直接运行同一套检查：

```bash
SketchMol-Understanding-Condition/scripts/run_server_smoke.sh
```

构造第一阶段 scaffold-preserving edit 数据集：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/build_edit_dataset.py \
  --input-csv SketchImageJEPA/data/example_molecules.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/smoke_edit_dataset \
  --max-pairs 12 \
  --min-similarity 0.1 \
  --max-similarity 0.99 \
  --eval-fraction 0.25
```

注意这里必须把项目路径追加到 `PYTHONPATH`，不能覆盖 RDKit module 注入的 Python path。

构造第一批 QED-improvement 论文数据接口：

```bash
SketchMol-Understanding-Condition/scripts/run_qed_dataset_smoke.sh
```

或提交到 Slurm：

```bash
SketchMol-Understanding-Condition/scripts/submit_qed_dataset_smoke.sh
```

该脚本从 `PhysTabMol/runs/20260601_070814_sketchmol_compare_structure_seed7/tables/train_table.csv` 读取 5000 个分子，挖 200 个同 scaffold、QED 提升的 source-target edit pairs，并生成：

```text
outputs/qed_edit_dataset_5k/edit_pairs.csv
outputs/qed_edit_dataset_5k/train.csv
outputs/qed_edit_dataset_5k/eval.csv
outputs/qed_edit_dataset_5k/baseline_variants.csv
outputs/qed_edit_dataset_5k/summary.json
```

`baseline_variants.csv` 会把每个 edit pair 展开成五个消融变体：`full`、`text_only`、`image_only`、`random_query`、`caption_bottleneck`。这是证明 understanding stream 不是装饰的第一组实验接口。

### 数据质量检测

参考 `SketchMolBenchmark` 和 `SketchMolEnd2End` 的经验，第一阶段不能只报告生成图片或 OCR 成败，还要检查 edit pair 本身是否适合做论文评估。新增诊断入口：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/diagnose_edit_dataset.py \
  --edit-pairs-csv SketchMol-Understanding-Condition/outputs/qed_edit_dataset_5k_diverse_strict/edit_pairs.csv \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/qed_edit_dataset_5k_diverse_strict/baseline_variants.csv \
  --output-json SketchMol-Understanding-Condition/outputs/qed_edit_dataset_5k_diverse_strict/diagnostics.json \
  --property-name QED
```

当前推荐使用 strict split 数据集：

```text
outputs/qed_edit_dataset_5k_diverse_strict/
```

诊断结果：

```text
edit_pairs: 200
train/eval: 161 / 39
baseline_rows: 1000
unique_scaffolds: 57
source_valid_fraction: 1.0
target_valid_fraction: 1.0
image_exists_fraction: 1.0
scaffold_match_fraction: 1.0
property_success_fraction: 1.0
train_eval_overlap_molecules: 0
```

这比最早的 smoke 数据更适合论文实验：不再只有单一 benzene scaffold，并且 eval 分子不泄漏到 train。

### 第一版 Retrieval Baseline

在接完整 SketchMol diffusion 之前，先用轻量 retrieval baseline 检查消融协议是否有信号：

```bash
SketchMol-Understanding-Condition/scripts/run_retrieval_baseline.sh
```

或提交到 Slurm：

```bash
SketchMol-Understanding-Condition/scripts/submit_retrieval_baseline.sh
```

如果要把 understanding stream 和 `SketchMolBenchmark` 放到同一张表里，不要只报告
retrieval/probe 指标。先把某个 condition variant 的预测导出成 direct
structure CSV，再用 benchmark materializer 计算同口径的 validity / property
success / MAE：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SUCC_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SUCC_BASELINE_VARIANTS_CSV=SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
SUCC_VARIANT=full \
SketchMol-Understanding-Condition/scripts/run_benchmark_export.sh
```

如果要评估某个已经导出的 condition encoder feature，而不是默认的非神经
condition feature，把 `SUCC_CONDITION_FEATURES_DIR` 指向
`export_condition_features.py` 产物目录：

```bash
SUCC_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SUCC_BASELINE_VARIANTS_CSV=SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
SUCC_VARIANT=full \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_fusion_v2_contrastive_e12 \
SUCC_BENCHMARK_EXPORT_DIR=SketchMol-Understanding-Condition/outputs/benchmark_export_fusion_v2_contrastive_e12 \
SUCC_BENCHMARK_OUTPUT_DIR=SketchMolBenchmark/outputs/understanding_condition_fusion_v2_contrastive_e12 \
SketchMol-Understanding-Condition/scripts/run_benchmark_export.sh
```

这会写出：

```text
SketchMol-Understanding-Condition/outputs/benchmark_export_full/benchmark_predictions.csv
SketchMolBenchmark/outputs/understanding_condition_full/benchmark_summary.csv
```

默认候选库只用 train target pool，避免 eval target 泄漏。`full` 应作为主结果；
`text_only`、`image_only`、`caption_bottleneck` 作为 ablation 单独导出并比较。

当前输出：

```text
outputs/retrieval_baseline_qed_strict/metrics.csv
outputs/retrieval_baseline_qed_strict/metrics.json
```

当前 ridge baseline 结果：

```text
variant             top1   top5   top10  mean_rank
full                0.026  0.154  0.333  18.67
text_only           0.026  0.128  0.256  20.00
image_only          0.026  0.154  0.333  18.67
random_query        0.026  0.128  0.256  19.51
caption_bottleneck  0.026  0.128  0.205  20.54
```

这个结果是有用的负控：在 QED-only 数据里，所有 instruction 基本都是 “increase QED”，所以 `full` 和 `image_only` 几乎相同，语言分支没有足够变化可学。下一步要构造 mixed-objective edit 数据集，例如 `increase QED`、`decrease LogP`、`increase TPSA`、`decrease TPSA`、`reduce MolWt` 混合，才能真正测试 understanding stream 是否利用自然语言目标。

### Mixed-Objective 数据集

构造 mixed-objective 数据集：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/build_mixed_objective_dataset.py \
  --input-csv PhysTabMol/runs/20260601_070814_sketchmol_compare_structure_seed7/tables/train_table.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict \
  --limit 8000 \
  --pairs-per-objective 80 \
  --max-pairs-per-scaffold 5 \
  --eval-fraction 0.2
```

当前数据质量：

```text
edit_pairs: 232
train/eval: 189 / 43
baseline_rows: 1160
unique_scaffolds: 29
source_valid_fraction: 1.0
target_valid_fraction: 1.0
image_exists_fraction: 1.0
scaffold_match_fraction: 1.0
train_eval_overlap_molecules: 0
```

运行 mixed retrieval：

```bash
SketchMol-Understanding-Condition/scripts/run_mixed_retrieval_baseline.sh
```

当前 ridge retrieval 结果：

```text
variant             top1   top5   top10  mean_rank
full                0.023  0.116  0.256  21.42
text_only           0.023  0.116  0.233  21.98
image_only          0.023  0.116  0.256  21.44
random_query        0.023  0.140  0.233  22.42
caption_bottleneck  0.023  0.093  0.233  22.05
```

这个结果说明 mixed 数据已经解决了 instruction 单一的问题，但 ridge + hashed text 仍然太弱，`full` 和 `image_only` 只有很小差距。下一步应做 objective-direction classifier 或接入真实 text/VLM encoder，而不是继续调 ridge retrieval。

### Objective-Direction Classifier

Retrieval 任务太难，先用 objective-direction 分类器验证语言目标是否可检测：

```bash
SketchMol-Understanding-Condition/scripts/run_objective_classifier.sh
```

或提交到 Slurm：

```bash
SketchMol-Understanding-Condition/scripts/submit_objective_classifier.sh
```

当前推荐数据集：

```text
outputs/mixed_objective_dataset_8k_strict_v2/
```

该版本保留不同 objective 下的同一 source-target pair 作为不同 instruction 样本，但每条样本只有一个标签，例如 `QED_increase` 或 `LogP_decrease`。全局 split 仍按 molecule connected components 做，避免 train/eval molecule leakage。

当前分类结果：

```text
variant             accuracy  macro_f1
full                1.000     1.000
text_only           1.000     1.000
image_only          0.324     0.282
random_query        0.183     0.177
caption_bottleneck  0.986     0.983
```

解释：语言目标本身已经可被可靠识别，`image_only` 和 `random_query` 明显低。这说明 mixed instruction stream 是有效的；但它还没有证明 multimodal understanding 对生成有帮助。下一步要把 objective classifier 扩展为 structure-aware 任务，例如在同一 instruction 下预测 target property delta bucket / target fingerprint retrieval，并用真实 text/VLM encoder 替换当前 hashed text proxy。

### Delta-Bucket Classifier

为了让任务同时依赖 instruction 和 source structure，进一步预测 `objective + delta magnitude bucket`，例如 `QED_increase_small`、`LogP_decrease_large`。bucket 阈值只用 train split 估计，避免 eval 分布泄漏。

运行：

```bash
SketchMol-Understanding-Condition/scripts/run_delta_bucket_classifier.sh
```

或提交到 Slurm：

```bash
SketchMol-Understanding-Condition/scripts/submit_delta_bucket_classifier.sh
```

当前结果：

```text
variant             accuracy  macro_f1
full                0.366     0.325
text_only           0.324     0.156
image_only          0.155     0.160
random_query        0.085     0.074
caption_bottleneck  0.352     0.292
```

这是目前最接近论文 claim 的正向检测：`full` 最好，说明图像/结构特征加上语言目标比只看文本更能预测编辑幅度；`image_only` 和 `random_query` 明显低，说明目标语言不是装饰。Slurm 复现：`15491795 COMPLETED 0:0`。

### Condition Encoder v0

当前统一 condition 接口输出：

```text
row + variant -> pooled [768], query_tokens [16, 256]
```

已有 `proxy` encoder 使用 `source Morgan fingerprint + hashed text`，适合快速验证接口，但结构分支仍然直接看 SMILES。新增 `multimodal_v0` encoder 把结构分支换成 OCR-free rendered image statistics：

```text
source_image -> grayscale / edge / histogram / patch statistics
prompt       -> hashed n-gram text vector
fusion       -> pooled [768] -> query_tokens [16, 256]
```

导出：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/export_condition_features.py \
  --encoder multimodal_v0 \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_v0
```

已导出：

```text
outputs/condition_features_mixed_v2_multimodal_v0/index.csv
outputs/condition_features_mixed_v2_multimodal_v0/pooled.npy          [2000, 768]
outputs/condition_features_mixed_v2_multimodal_v0/query_tokens.npy    [2000, 16, 256]
```

分类器现在可以直接读取导出的 condition features：

```bash
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_v0 \
SUCC_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/delta_bucket_classifier_multimodal_v0 \
SketchMol-Understanding-Condition/scripts/run_delta_bucket_classifier.sh
```

`multimodal_v0` objective-direction 结果：

```text
variant             accuracy  macro_f1
full                1.000     1.000
text_only           1.000     1.000
image_only          0.225     0.217
random_query        0.183     0.177
caption_bottleneck  0.972     0.975
```

`multimodal_v0` delta-bucket 结果：

```text
variant             accuracy  macro_f1
full                0.352     0.294
text_only           0.380     0.180
image_only          0.085     0.063
random_query        0.085     0.074
caption_bottleneck  0.282     0.278
```

解释：objective-direction 仍然主要由文本决定，所以 `full` 和 `text_only` 都接近满分。delta-bucket 更有价值：`multimodal_v0 full` 的 macro-F1 明显高于 `text_only`，说明真实 rendered-image 统计特征已经提供了编辑幅度相关信号；但它低于 Morgan fingerprint proxy 的 `full`（0.325 macro-F1），说明当前 OCR-free 图像统计还只是弱视觉理解分支。

### Condition Encoder v1

`multimodal_cnn_v1` 尝试把 image branch 从全局统计升级为固定卷积/patch-layout 特征：

```text
source_image -> Sobel/Laplacian/diagonal filters + 8x8 patch stats + 16x16 ink map
prompt       -> hashed n-gram text vector
fusion       -> pooled [768] -> query_tokens [16, 256]
```

导出：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/export_condition_features.py \
  --encoder multimodal_cnn_v1 \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_cnn_v1
```

已导出：

```text
outputs/condition_features_mixed_v2_multimodal_cnn_v1/index.csv
outputs/condition_features_mixed_v2_multimodal_cnn_v1/pooled.npy          [2000, 768]
outputs/condition_features_mixed_v2_multimodal_cnn_v1/query_tokens.npy    [2000, 16, 256]
```

`multimodal_cnn_v1` objective-direction 结果：

```text
variant             accuracy  macro_f1
full                1.000     1.000
text_only           1.000     1.000
image_only          0.169     0.152
random_query        0.183     0.177
caption_bottleneck  0.972     0.975
```

`multimodal_cnn_v1` delta-bucket 结果：

```text
variant             accuracy  macro_f1
full                0.296     0.263
text_only           0.380     0.180
image_only          0.056     0.033
random_query        0.085     0.074
caption_bottleneck  0.282     0.278
```

结论：固定卷积/patch 特征没有超过 v0 的全局统计（`full` macro-F1 0.263 vs 0.294），但仍然高于 `text_only` 的 macro-F1 0.180。这说明“只堆手工视觉特征”不是最优路线；下一步应训练一个 image encoder，例如用 source image 预测 Morgan/property delta 或用 pair contrastive loss 学结构 embedding，再接入同一个 condition feature export/probe 流程。

### Condition Encoder v2

`multimodal_trained_v2` 使用一个可训练的小 CNN image encoder。训练分两步：先在 RDKit module 环境导出监督标签，再在 Torch venv 里训练 CNN。

导出 image encoder 监督标签：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/export_image_encoder_v2_targets.py \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/image_encoder_v2_targets_mixed_v2
```

当前 targets：

```text
rows: 400
train/eval: 329 / 71
fingerprint_bits: 512
properties: MolWt, LogP, QED, TPSA, HBD, HBA, rotatable
```

训练 3-epoch smoke CNN：

```bash
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
/home/bdong/scratch/venvs/phystabmol/bin/python \
  SketchMol-Understanding-Condition/scripts/train_image_encoder_v2.py \
  --targets-npz SketchMol-Understanding-Condition/outputs/image_encoder_v2_targets_mixed_v2/image_encoder_v2_targets.npz \
  --output-dir SketchMol-Understanding-Condition/outputs/image_encoder_v2_mixed_v2_e3 \
  --epochs 3 \
  --batch-size 32 \
  --image-size 64
```

训练曲线显示 CNN 确实学到 source image 到结构监督的信号：

```text
epoch  eval_fp_bce  eval_property_mse  train_loss
1      0.556        1.210              0.769
2      0.358        1.325              0.378
3      0.331        1.384              0.277
```

导出 trained condition features：

```bash
/home/bdong/scratch/venvs/phystabmol/bin/python \
  SketchMol-Understanding-Condition/scripts/export_condition_features.py \
  --encoder multimodal_trained_v2 \
  --image-encoder-checkpoint SketchMol-Understanding-Condition/outputs/image_encoder_v2_mixed_v2_e3/image_encoder_v2.pt \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_trained_v2_e3
```

已导出：

```text
outputs/condition_features_mixed_v2_multimodal_trained_v2_e3/pooled.npy        [2000, 768]
outputs/condition_features_mixed_v2_multimodal_trained_v2_e3/query_tokens.npy  [2000, 16, 256]
```

`multimodal_trained_v2_e3` objective-direction 结果：

```text
variant             accuracy  macro_f1
full                1.000     1.000
text_only           1.000     1.000
image_only          0.183     0.165
random_query        0.183     0.177
caption_bottleneck  0.972     0.975
```

`multimodal_trained_v2_e3` delta-bucket 结果：

```text
variant             accuracy  macro_f1
full                0.254     0.213
text_only           0.380     0.180
image_only          0.056     0.042
random_query        0.085     0.074
caption_bottleneck  0.282     0.278
```

结论：短训练 CNN 能学 source fingerprint，但接到 edit delta-bucket 后并没有超过 v0/v1；`full` macro-F1 仍高于 `text_only`，但只有 0.213。这个负结果说明“只让 image encoder 预测 source fingerprint/property”不够贴近编辑任务。下一步应把 v2 训练目标改成 pair-aware：输入 source image，监督 target-source property delta、delta bucket，或者做 source/target contrastive embedding，而不是只做 source reconstruction。

### Condition Encoder v2.1

`multimodal_pairaware_v2` 把训练目标改成 pair-aware：输入仍然只看 `source_image`，但监督信号变成 `objective_direction_bucket` 分类和 target-source property delta 回归。

导出 pair-aware targets：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/export_pairaware_image_targets.py \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/pairaware_image_targets_mixed_v2
```

当前 targets：

```text
rows: 400
train/eval: 329 / 71
classes: 15 objective_direction_bucket labels
properties: MolWt, LogP, QED, TPSA, HBD, HBA, rotatable deltas
```

训练 e8 pair-aware CNN：

```bash
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
/home/bdong/scratch/venvs/phystabmol/bin/python \
  SketchMol-Understanding-Condition/scripts/train_pairaware_image_encoder.py \
  --targets-npz SketchMol-Understanding-Condition/outputs/pairaware_image_targets_mixed_v2/pairaware_image_targets.npz \
  --output-dir SketchMol-Understanding-Condition/outputs/pairaware_image_encoder_mixed_v2_e8 \
  --epochs 8 \
  --batch-size 32 \
  --image-size 64
```

训练现象：

```text
train_loss: 3.013 -> 2.088
eval_accuracy: best 0.070
eval_delta_mse: best 0.726
```

导出 pair-aware condition features：

```bash
/home/bdong/scratch/venvs/phystabmol/bin/python \
  SketchMol-Understanding-Condition/scripts/export_condition_features.py \
  --encoder multimodal_pairaware_v2 \
  --image-encoder-checkpoint SketchMol-Understanding-Condition/outputs/pairaware_image_encoder_mixed_v2_e8/pairaware_image_encoder.pt \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_pairaware_v2_e8
```

已导出：

```text
outputs/condition_features_mixed_v2_multimodal_pairaware_v2_e8/pooled.npy        [2000, 768]
outputs/condition_features_mixed_v2_multimodal_pairaware_v2_e8/query_tokens.npy  [2000, 16, 256]
```

`multimodal_pairaware_v2_e8` objective-direction 结果：

```text
variant             accuracy  macro_f1
full                1.000     1.000
text_only           1.000     1.000
image_only          0.169     0.163
random_query        0.183     0.177
caption_bottleneck  0.972     0.975
```

`multimodal_pairaware_v2_e8` delta-bucket 结果：

```text
variant             accuracy  macro_f1
full                0.282     0.249
text_only           0.380     0.180
image_only          0.056     0.032
random_query        0.085     0.074
caption_bottleneck  0.282     0.278
```

结论：pair-aware 训练比 source-reconstruction v2 有提升（`full` macro-F1 0.249 vs 0.213），说明训练目标更贴近 edit task 是有效方向；但它仍低于 `multimodal_cnn_v1` 0.263 和 `multimodal_v0` 0.294。主要问题是当前 pair-aware encoder 只看 source image，却被要求预测 objective-specific bucket，缺少 prompt 条件。下一步应训练 image+text fusion encoder：`source_image + prompt -> objective_direction_bucket + property_delta`，而不是 image-only pair-aware encoder。

### Condition Encoder v2.2

`multimodal_fusion_v2` 进一步把 prompt 放进训练过程：`source_image + prompt -> objective_direction_bucket + property_delta`。这更接近 understanding stream，但也测试了一个关键风险：监督头训练好的 fusion embedding 是否能直接作为 downstream condition feature。

导出 fusion targets：

```bash
module purge
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
python SketchMol-Understanding-Condition/scripts/export_fusion_image_text_targets.py \
  --baseline-variants-csv SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
  --output-dir SketchMol-Understanding-Condition/outputs/fusion_image_text_targets_mixed_v2
```

训练 e8 fusion encoder：

```bash
export PYTHONPATH="SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"
/home/bdong/scratch/venvs/phystabmol/bin/python \
  SketchMol-Understanding-Condition/scripts/train_fusion_image_text_encoder.py \
  --targets-npz SketchMol-Understanding-Condition/outputs/fusion_image_text_targets_mixed_v2/fusion_image_text_targets.npz \
  --output-dir SketchMol-Understanding-Condition/outputs/fusion_image_text_encoder_mixed_v2_e8 \
  --epochs 8 \
  --batch-size 32 \
  --image-size 64 \
  --text-dim 256
```

训练现象：

```text
train_loss: 2.970 -> 2.063
eval_accuracy: best 0.127
eval_delta_mse: best 0.723
```

导出时对 `multimodal_fusion_v2` 加了 batched `encode_rows()`，否则逐行 CPU 前向太慢。导出结果：

```text
outputs/condition_features_mixed_v2_multimodal_fusion_v2_e8/pooled.npy        [2000, 768]
outputs/condition_features_mixed_v2_multimodal_fusion_v2_e8/query_tokens.npy  [2000, 16, 256]
```

`multimodal_fusion_v2_e8` objective-direction 结果：

```text
variant             accuracy  macro_f1
full                0.676     0.652
text_only           1.000     1.000
image_only          0.141     0.110
random_query        0.183     0.174
caption_bottleneck  1.000     1.000
```

`multimodal_fusion_v2_e8` delta-bucket 结果：

```text
variant             accuracy  macro_f1
full                0.239     0.157
text_only           0.268     0.139
image_only          0.085     0.042
random_query        0.070     0.059
caption_bottleneck  0.324     0.268
```

结论：fusion 训练本身比 image-only pair-aware 更能学到 objective 信息（训练 eval accuracy 最高 0.127 vs 0.070），但把中间 fusion embedding 直接导出给 downstream linear probe 效果反而变差，`full` 不如简单 text/caption condition。这说明当前问题不是“是否加入 prompt”，而是 condition representation 的训练目标不对：分类头能用 embedding，但 embedding 本身不一定线性保留 edit 信息。下一步应改成直接训练可导出的 condition feature，例如 metric learning / supervised contrastive，使同一 delta-bucket 的 `pooled/query_tokens` 靠近、不同 bucket 分离；或者把 downstream probe head 和 condition encoder 联合训练，而不是先训 head 再丢掉 head。

### 下一轮主实验：Contrastive Fusion

这里建议优先继续 `multimodal_fusion_v2`，但把训练目标改成可导出的表示目标：
分类头和 property-delta regression 继续保留，同时加 supervised contrastive loss，让同一
objective-direction bucket 的 fusion embeddings 靠近、不同 bucket 分离。

训练 contrastive fusion encoder：

```bash
cd /scratch/bdong/projects/Diffusion-Molecule

SUCC_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SUCC_BASELINE_CSV=SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv \
SUCC_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/fusion_image_text_encoder_mixed_v2_contrastive_e12 \
SUCC_EPOCHS=12 \
SUCC_BATCH_SIZE=64 \
SUCC_CONTRASTIVE_WEIGHT=0.2 \
SUCC_CONTRASTIVE_TEMPERATURE=0.2 \
SketchMol-Understanding-Condition/scripts/run_contrastive_fusion_encoder.sh
```

导出 condition feature；这里的 `SUCC_IMAGE_ENCODER_CHECKPOINT` 是历史参数名，
对 `multimodal_fusion_v2` 实际传入 fusion checkpoint：

```bash
SUCC_ENCODER=multimodal_fusion_v2 \
SUCC_IMAGE_ENCODER_CHECKPOINT=SketchMol-Understanding-Condition/outputs/fusion_image_text_encoder_mixed_v2_contrastive_e12/fusion_image_text_encoder.pt \
SUCC_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_fusion_v2_contrastive_e12 \
SketchMol-Understanding-Condition/scripts/run_condition_encoder_export.sh
```

先跑 probe，确认表示本身是否变好：

```bash
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_fusion_v2_contrastive_e12 \
SUCC_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/delta_bucket_classifier_multimodal_fusion_v2_contrastive_e12 \
SketchMol-Understanding-Condition/scripts/run_delta_bucket_classifier.sh
```

再把同一批 exported features 接到 `SketchMolBenchmark` direct-prediction
materializer，避免只在 internal probe 上自洽：

```bash
SUCC_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
SUCC_VARIANT=full \
SUCC_CONDITION_FEATURES_DIR=SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_fusion_v2_contrastive_e12 \
SUCC_BENCHMARK_EXPORT_DIR=SketchMol-Understanding-Condition/outputs/benchmark_export_fusion_v2_contrastive_e12 \
SUCC_BENCHMARK_OUTPUT_DIR=SketchMolBenchmark/outputs/understanding_condition_fusion_v2_contrastive_e12 \
SketchMol-Understanding-Condition/scripts/run_benchmark_export.sh
```

继续推进的门槛：`full` 的 delta-bucket macro-F1 至少要回到并超过
`multimodal_v0` 的 0.294，同时 benchmark summary 里的 target similarity /
property success 不能弱于默认 ridge bridge。若只提升 objective accuracy，但
benchmark 不动，说明 embedding 仍然更像分类器中间层，不适合继续接 diffusion。

## 1. 从 UniVideo 迁移什么

UniVideo 值得迁移的不是视频模型本身，而是三个设计原则。

### 1.1 MLLM 不是输出文本，而是输出 hidden condition

UniVideo 的 `mllm_encoder.py` 用 Qwen2.5-VL，配置里有 `num_metaqueries`。tokenizer 会在 prompt 后追加 `<begin_of_img><img0>...<imgN><end_of_img>` 这类 metaquery tokens，然后取 MLLM hidden states 作为 generation stream 的条件。

这对 SketchMol 的启发是：不要让 LLM 输出“把 logP 降低一点”这种文本后再人工解析；而是让模型学习一组固定长度的 molecular edit query tokens，例如 16 或 32 个 query token，投影到 SketchMol UNet 的 `context_dim=256`。

### 1.2 Understanding stream 必须看输入图像

UniVideo 配置里区分 `mllm_use_ref_img` 和 `mllm_use_cond_pixels`，说明 generation stream 的条件可以同时来自文本指令和参考图像/像素内容。

对应到 SketchMol：

- 输入：原始分子图像或手绘 sketch。
- 输入：自然语言编辑目标，例如“保留 core scaffold，将 para 位氯替换为甲氧基，同时降低 LogP”。
- 输出：一组 condition tokens，供 diffusion inpainting / editing 使用。

这里的关键是 MLLM 必须参与理解原始分子图像里的结构约束，而不是只读文本目标。

### 1.3 生成流只消费 condition，不关心语言细节

UniVideo 的 transformer forward 接受 `encoder_hidden_states` 和 `encoder_attention_mask`，先用 projection/refiner 处理，再进入 cross-attention。

SketchMol 也类似：当前 `MixedEmbedderV2` 输出 `[B, condition_len, 256]`，UNet 通过 `conditioning_key: crossattn` 消费。我们只需要让新的 MLLM condition encoder 输出同样形状，就能最小侵入接入。

## 2. 推荐研究问题

不要一开始做“任意自然语言分子设计”。风险太大，数据也不一定够。

更好的问题是：

> 给定分子图像和编辑指令，生成满足指令的分子变体，同时尽量保留 scaffold，并改善一个或多个目标性质。

任务可以拆成三类，逐级推进：

1. Scaffold-preserving functional group replacement  
   例如保留 core scaffold，只替换 side chain 或特定位点的 functional group。

2. Property-directed molecular editing  
   例如降低 LogP、提高 QED、降低 TPSA、提高 solubility proxy。

3. Protein-aware optimization  
   例如增强 AKT1/EP4/ROCK1 binding，同时控制 LogP/QED/SA。

其中第 1 类最适合作为 proof of understanding，因为可以很明确地评估“是否真的理解了 scaffold 和编辑位点”。

## 3. 最低可行系统

### 3.1 模型结构

建议先做一个三段式结构：

```text
input molecule image + edit instruction
        |
        v
Molecular MLLM Encoder
        |
        v
condition tokens: [B, Nq, 256]
        |
        v
SketchMol latent diffusion UNet
        |
        v
edited molecule image -> MolScribe/RDKit validation
```

具体模块：

- `MolecularMLLMConditionEncoder`
  - backbone 可以先用冻结的 Qwen2.5-VL 或更轻的 vision-language encoder；
  - 增加 learnable molecular query tokens；
  - 输出 query hidden states；
  - 用 MLP 投影到 `context_dim=256`。

- `HybridConditionEncoder`
  - 保留 SketchMol 原来的 property tokens；
  - 拼接 MLLM tokens；
  - 输出 `[property_tokens + mllm_tokens]` 给 UNet cross-attention。

- `SketchMol diffusion generator`
  - 第一阶段不改 VAE；
  - 第一阶段不改 UNet 主体；
  - 只改 condition stage，降低工程风险。

### 3.2 两种接入方式

最稳的是从轻到重做。

#### A. Adapter-only

冻结 SketchMol diffusion 和 MLLM，只训练一个 projector / adapter，把 MLLM hidden states 映射到 256 维 condition tokens。

优点：

- 训练成本低；
- 可以快速验证 MLLM 信息是否可用；
- 适合数据较少的情况。

缺点：

- 能力上限有限；
- 如果 SketchMol 原模型没见过这类条件 token，效果可能不稳定。

#### B. Condition-stage finetune

冻结 VAE，微调 UNet cross-attention 和新的 condition encoder。

优点：

- 更容易让 generator 学会使用新 token；
- 适合论文主实验。

缺点：

- 需要更多数据；
- 必须做严格消融，否则容易被质疑只是多参数带来的收益。

## 4. 数据怎么构造

这个方向成败主要取决于数据，不取决于模型名字。

### 4.1 自监督编辑对

从已有分子库构造 `(source molecule, target molecule, edit instruction)`：

1. 用 Bemis-Murcko scaffold 或 MCS 找 source/target 的共同 scaffold。
2. 找 side chain / functional group 差异。
3. 自动生成 instruction：
   - “keep the scaffold and replace chloro with methoxy”
   - “preserve the core ring system and reduce LogP”
   - “modify the side chain to improve QED while keeping molecular weight similar”
4. source 渲染成图像，target 渲染成监督图像。
5. target 计算性质，作为评估标签或辅助 condition。

这样不需要人工标注大规模文本，但 instruction 和结构变化之间有真实对应关系。

### 4.2 性质驱动 pair mining

对同 scaffold 分子按性质差异配对：

- source 和 target scaffold 相同或 MCS 高；
- target 的 LogP 更低 / QED 更高 / TPSA 更合理；
- structural distance 不要太大，避免变成 de novo generation。

这类数据适合证明“自然语言性质目标 + 图像理解”能指导局部优化。

### 4.3 蛋白目标数据

SketchMol 已有 EP4/AKT1/ROCK1 这类 protein condition 使用方式。可以把 protein-aware task 放在第二阶段：

- instruction 写成 “improve AKT1 binding while preserving the scaffold”；
- condition 里同时给 protein token / predicted activity label；
- 评估用 docking score、QSAR predictor 或已有活性标签。

不要把蛋白 binding 作为第一阶段主卖点。它的噪声更大，证明链条更长。

## 5. 怎么证明 understanding stream 真的有用

这是论文最关键的部分。必须设计消融，让评审无法说“LLM 只是装饰”。

### 5.1 必做 baseline

1. SketchMol original condition  
   只用原始 property/protein condition，不用 MLLM。

2. Text-only LLM condition  
   MLLM 只读 instruction，不看 molecule image。

3. Image-only condition  
   MLLM 只看 molecule image，不读 instruction。

4. Random/frozen query tokens  
   用同样数量的 query tokens，但不来自 MLLM。

5. Caption bottleneck  
   让 MLLM 先生成文本 caption，再用文本 encoder 作为 condition。这个 baseline 用来证明 hidden/metaquery 比“LLM 写一句话”更强。

6. Oracle property condition  
   直接给 target property 数值，检验 MLLM 是否只是学到了性质标签。

如果 full model 只比 1 好，但不比 2/4/6 好，这个故事就不成立。

### 5.2 必做反事实实验

1. Instruction swap  
   同一个 source image，交换不同编辑指令。输出应随指令改变。

2. Image swap  
   同一个 instruction，换 source image。输出应保留各自 scaffold，而不是生成同一种结果。

3. Scaffold mask test  
   把 scaffold 区域遮住或打乱，性能应明显下降。否则模型可能没有真的看结构。

4. Functional group target test  
   指定替换基团，评估生成结果里目标 group 出现率和非目标区域保留率。

5. Attention / token attribution  
   检查 diffusion cross-attention 是否使用 MLLM query tokens，尤其是在被编辑区域附近。

### 5.3 指标

基础分子指标：

- validity
- uniqueness
- novelty
- reconstruction / parse success
- property success rate

编辑任务指标：

- scaffold preservation rate
- MCS similarity
- Tanimoto similarity range
- edit locality score
- target functional group hit rate
- non-target region preservation

性质优化指标：

- delta LogP
- delta QED
- delta TPSA
- synthetic accessibility
- multi-objective success rate

蛋白任务指标：

- predicted activity improvement
- docking score improvement
- activity-success under scaffold constraint

最重要的不是单个指标最高，而是 full model 在“需要图像理解 + 指令理解同时成立”的指标上显著优于 baseline。

## 6. 训练目标

第一阶段可以不做复杂 RL 或 docking-in-the-loop。

建议训练目标：

1. Diffusion denoising loss  
   标准 latent diffusion 噪声预测。

2. Condition dropout / classifier-free guidance  
   随机 drop instruction、image、property condition，用于训练可控 guidance 和消融。

3. Optional contrastive alignment  
   让 source + correct instruction 的 condition tokens 比 source + wrong instruction 更接近 target edit embedding。

4. Optional scaffold consistency loss  
   用 RDKit/MolScribe 后验评估做离线筛选，不建议一开始端到端反传。

第一版重点应放在数据和消融，不要把训练目标堆太多。

## 7. 推荐实验路线

### Phase 1: 不改大模型，证明接口可行

- 冻结 SketchMol；
- 冻结 MLLM；
- 训练 MLLM hidden -> 256 condition projector；
- 数据只做 scaffold-preserving replacement；
- 对比 original condition、text-only、random-query。

成功标准：

- validity 不明显下降；
- scaffold preservation 高于 de novo/property-only；
- functional group hit rate 高于 text-only/random-query。

### Phase 2: 微调 condition + cross-attention

- 解冻 SketchMol UNet 里的 cross-attention；
- 加 condition dropout；
- 加 instruction/image swap evaluation；
- 扩展到 property-directed editing。

成功标准：

- 同 scaffold 性质优化成功率提升；
- image swap / instruction swap 都有明显响应；
- target 和 non-target 区域的权衡优于原 SketchMol。

### Phase 3: 蛋白目标故事

- 加 AKT1/EP4/ROCK1 数据；
- instruction 里加入 binding target；
- 与原 single-protein condition baseline 比较；
- 用 docking/QSAR 做二级筛选。

成功标准：

- 在 scaffold preservation 约束下，activity/docking 指标改善；
- 不是简单牺牲 drug-likeness 换 binding score。

## 8. 最大风险和对应处理

### 风险 1: MLLM 不懂分子图

普通 VLM 对分子图像的化学语义可能很弱。处理办法：

- 先用 MolScribe / RDKit 解析 source image，给 MLLM 辅助文本：SMILES、atom labels、scaffold summary；
- 或者预训练一个轻量 molecular image encoder；
- 不要完全依赖通用 VLM 的化学知识。

### 风险 2: 数据规模不够

处理办法：

- 用自监督 pair mining 自动造数据；
- 从简单编辑开始；
- 冻结大模型，只训 adapter；
- 把“能否理解指令和图像”作为主贡献，而不是追求 SOTA de novo generation。

### 风险 3: LLM 变装饰

处理办法：

- baseline 必须强；
- 做 image/text swap；
- 做 random query；
- 做 caption bottleneck；
- 做 attention/token attribution；
- 评估必须包含编辑局部性和 scaffold 保留，而不只是 property 变好。

### 风险 4: 生成结果不可解析

处理办法：

- 保留 SketchMol 原来的图像生成和解析路径；
- 先做 inpainting/editing，不做完全开放生成；
- 后处理用 MolScribe + RDKit validity；
- 对无效样本单独报告，不要只筛选后报成功率。

## 9. 最适合写成论文的故事

标题方向可以是：

> Understanding-Conditioned Molecular Diffusion for Instruction-Guided Scaffold-Preserving Editing

故事线：

1. 现有分子 diffusion 可以按性质生成，但对自然语言编辑目标和图像中具体结构关系理解不足。
2. UniVideo 类 unified understanding-generation 架构启发我们把 MLLM hidden states 作为 generation condition。
3. 我们提出 molecular understanding stream，将 molecule image + instruction 编码成 conditional query tokens。
4. diffusion stream 在 latent image space 生成编辑后分子。
5. 通过严格消融证明：只有同时使用图像理解和指令理解，才能在 scaffold 保留、functional group 命中和性质优化上稳定提升。

这比“LLM + diffusion 生成分子”更容易成立，因为贡献点清楚：MLLM 负责理解和条件构造，diffusion 负责生成，评估专门证明理解流的必要性。

## 10. 近期可执行 TODO

1. 写一个 pair mining 脚本，从同 scaffold 分子构造 source-target edit pairs。
2. 定义 5-10 个模板 instruction，覆盖 scaffold 保留、functional group replacement、LogP/QED/TPSA 调整。
3. 实现 `MolecularMLLMConditionEncoder`，输出 `[B, Nq, 256]`。
4. 实现 `HybridConditionEncoder`，拼接原 SketchMol property tokens 和 MLLM query tokens。
5. 先跑 adapter-only 小实验，只比较 full / text-only / image-only / random-query。
6. 建立评估脚本：validity、scaffold preservation、functional group hit、property delta。
