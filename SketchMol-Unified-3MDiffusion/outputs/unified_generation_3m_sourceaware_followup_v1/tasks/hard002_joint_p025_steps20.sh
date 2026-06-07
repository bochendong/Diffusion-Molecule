#!/usr/bin/env bash
set -euo pipefail
cd "/home/bdong/scratch/projects/Diffusion-Molecule"
mkdir -p "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/logs" "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/hard002_joint_p025_steps20"
exec > >(tee "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/logs/hard002_joint_p025_steps20.log") 2>&1
echo "source-aware follow-up diffusion task: hard002_joint_p025_steps20"
echo "  base_output_dir=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head"
echo "  prior_loss_weight=0.25"
echo "  train_diffusion_connector=1"
echo "  sample_steps=20"
echo "  extra_epochs=20"
echo "  output_dir=SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/hard002_joint_p025_steps20"
for required in "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head/dataset/unified_condition_train.jsonl" "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head/dataset/unified_condition_eval.jsonl" "SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head/edit_condition_tokens/edit_condition_connector.pt"; do
  if [ ! -f "$required" ]; then
    echo "Missing required base artifact: $required" >&2
    exit 2
  fi
done
export SMU3M_OUTPUT_DIR="SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head"
export SMU3M_DIFFUSION_DIR="SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/hard002_joint_p025_steps20/latent_diffusion"
export SMU3M_BASE_DIFFUSION_DIR="SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head/latent_diffusion"
export SMU3M_EVAL_LATENT_DIR="SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/hard002_joint_p025_steps20/eval_latent"
export SMU3M_PRIOR_LOSS_WEIGHT="0.25"
export SMU3M_TRAIN_DIFFUSION_CONNECTOR="1"
export SMU3M_EVAL_SAMPLE_STEPS="20"
export SMU3M_DIFFUSION_EXTRA_EPOCHS="20"
export SMU3M_DIFFUSION_SEED="13"
export SMU3M_EVAL_SEED="14"
export SMU3M_RESUME="0"
export SMU3M_RUN_MATERIALIZED_BENCHMARK="0"
export SMU3M_EVAL_LIMIT="1000"
export SMU3M_MAX_EVAL_PER_PROPERTY_COUNT="0"
bash "/home/bdong/scratch/projects/Diffusion-Molecule/SketchMol-Unified-3MDiffusion/scripts/run_unified_diffusion_refine.sh"
