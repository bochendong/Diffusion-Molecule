#!/usr/bin/env bash
# MuMO-train support audit -> B41 warm-start training/freeze -> oracle -> science gate.

set -euo pipefail

STAGE="${1:?usage: run_property_aligned_valid_terminal_mumo_v2.sh prepare|trainfreeze|oracle|gate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_CODE_DIR="$(cd "$PROJECT_CODE_DIR/.." && pwd)"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  if [[ "$STAGE" == "trainfreeze" ]]; then
    module load cuda/12.6 >/dev/null 2>&1 || true
  fi
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_B_MUMO_V2_ROOT:-$PROJECT_DIR/outputs/property_aligned_valid_terminal_mumo_v2_vocab_expanded_max64/seed_2211}"
DATA_DIR="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711/data"
PREVIOUS_ROOT="$PROJECT_DIR/outputs/b_series_external_mumo_transfer_v1/seed_2191"
JOINT_DATASET="$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
B37_DIR="$PROJECT_DIR/outputs/source_clamped_region_graph_diffusion_v37/seed_1983"
B38_DIR="$PROJECT_DIR/outputs/source_clamped_latent_graph_jump_process_v38/seed_1985"
B39_DIR="$PROJECT_DIR/outputs/latent_cardinality_graph_jump_bridge_v39/seed_1987"
B40_DIR="$PROJECT_DIR/outputs/valence_constrained_latent_particle_bridge_v40/seed_1989"
B41_DIR="$PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
VALID_TERMINAL_DIR="$PROJECT_DIR/outputs/valid_terminal_molecule_latent_jump_v1/seed_1991"
PREREG="$SCRIPT_DIR/property_aligned_valid_terminal_mumo_v2_preregistration.json"
RUNNER="$SCRIPT_DIR/property_aligned_valid_terminal_mumo_v2.py"
PREPARE_DIR="$RUN_ROOT/prepare"
FREEZE_DIR="$RUN_ROOT/trainfreeze"
ORACLE_DIR="$RUN_ROOT/oracle"
EVAL_DIR="$RUN_ROOT/evaluation"
GATE_DIR="$RUN_ROOT/gate"
CANDIDATES="$FREEZE_DIR/frozen_candidates.csv"
ORACLE="$ORACLE_DIR/generated_properties.csv"

export PYTHONPATH="$PROJECT_CODE_DIR:$PROJECT_CODE_DIR/experiments/unified_latent_external_transfer:$PROJECT_CODE_DIR/experiments/unified_latent_flow:$PROJECT_CODE_DIR/experiments/unified_latent_table1:$PROJECT_CODE_DIR/experiments/unified_constraint_agent:$PROJECT_CODE_DIR/scripts:$REPO_CODE_DIR/SketchMol-Unified-3MDiffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PREPARE_DIR" "$FREEZE_DIR" "$ORACLE_DIR" "$EVAL_DIR" "$GATE_DIR"

COMMON_B_ARGS=(
  --train-csv "$JOINT_DATASET/unified_joint_train_rows.csv"
  --validation-csv "$JOINT_DATASET/unified_joint_validation_rows.csv"
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt"
  --representation-summary "$REPRESENTATION_DIR/summary.json"
  --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt"
  --b22-summary "$B22_DIR/summary.json"
  --b36-summary "$B36_DIR/summary.json"
  --b37-summary "$B37_DIR/summary.json"
  --b38-checkpoint "$B38_DIR/source_clamped_latent_graph_jump_process.pt"
  --b38-summary "$B38_DIR/summary.json"
  --b39-checkpoint "$B39_DIR/latent_cardinality_graph_jump_bridge.pt"
  --b39-summary "$B39_DIR/summary.json"
  --b39-evaluated-candidates "$B39_DIR/evaluated_train_only_dev_candidates.csv"
  --b40-summary "$B40_DIR/summary.json"
  --b40-evaluated-candidates "$B40_DIR/evaluated_train_only_dev_candidates.csv"
  --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt"
  --b41-summary "$B41_DIR/summary.json"
  --b41-protocol-manifest "$PROJECT_CODE_DIR/experiments/unified_latent_flow/viability_preserving_interacting_particle_transport_v41_preregistration.json"
  --valid-terminal-summary "$VALID_TERMINAL_DIR/summary.json"
)

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$RUNNER" prepare \
      --preregistration "$PREREG" \
      --data-dir "$DATA_DIR" \
      --previous-conditions "$PREVIOUS_ROOT/prepare/generation_conditions.jsonl" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
      --output-dir "$PREPARE_DIR" \
      --workers "${SLURM_CPUS_PER_TASK:-1}"
    ;;
  trainfreeze)
    "$PYTHON_BIN" "$RUNNER" trainfreeze \
      --preregistration "$PREREG" \
      --prepare-summary "$PREPARE_DIR/prepare_summary.json" \
      --fit-pairs "$PREPARE_DIR/fit_pairs.pkl" \
      --calibration-pairs "$PREPARE_DIR/calibration_pairs.pkl" \
      --generation-conditions "$PREPARE_DIR/generation_conditions.jsonl" \
      --output-dir "$FREEZE_DIR" \
      --device auto \
      "${COMMON_B_ARGS[@]}"
    ;;
  oracle)
    if "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("execution_skipped") else 1)' "$FREEZE_DIR/freeze_summary.json"; then
      printf '{"protocol":"property_aligned_valid_terminal_mumo_v2","stage":"oracle","execution_skipped":true}\n' > "$ORACLE_DIR/oracle_summary.json"
      exit 0
    fi
    [[ -s "$CANDIDATES" ]] || { echo "ERROR: frozen exact-n20 candidates missing" >&2; exit 2; }
    SUCC_PYTHON_BIN="$PYTHON_BIN" \
    SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$CANDIDATES" \
    SUCC_ORACLE_OUTPUT_CSV="$ORACLE" \
    SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
    bash "$PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$CANDIDATES" \
      --output-dir "$EVAL_DIR" \
      --generated-properties-csv "$ORACLE" \
      --source-properties-csv "$ORACLE" \
      --group-column condition_id \
      --min-source-tanimoto 0.4 \
      --report-title "Property-aligned valid-terminal MuMO prospective OOD exact n=20"
    ;;
  gate)
    "$PYTHON_BIN" "$RUNNER" gate \
      --preregistration "$PREREG" \
      --prepare-summary "$PREPARE_DIR/prepare_summary.json" \
      --freeze-summary "$FREEZE_DIR/freeze_summary.json" \
      --candidates "$CANDIDATES" \
      --evaluation-detail "$EVAL_DIR/external_multiproperty_detail.csv" \
      --output-json "$GATE_DIR/summary.json"
    ;;
  *)
    echo "ERROR: unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
