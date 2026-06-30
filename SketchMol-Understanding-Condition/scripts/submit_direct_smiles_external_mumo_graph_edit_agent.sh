#!/usr/bin/env bash
# Submit MuMO source-conditioned GraphEditDSL agent benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

BASE_AGENTIC_OUTPUT_DIR="${SUCC_EXTERNAL_GRAPH_EDIT_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1}"
export SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV:-$BASE_AGENTIC_OUTPUT_DIR/direct_smiles_proposals.csv}"
export SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS="${SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS:-1}"

if [[ ! -f "$SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV" && "$SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS" == "1" ]]; then
  echo "ERROR: missing direct proposal CSV: $SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV" >&2
  echo "Run submit_direct_smiles_external_mumo_agentic_revise.sh first, or set SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="${SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
export SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="${SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_graph_edit_agent_v1}"
export SUCC_EXTERNAL_GRAPH_EDIT_SUITE="${SUCC_EXTERNAL_GRAPH_EDIT_SUITE:-mumo}"
export SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT="${SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT:-all}"
export SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK:-200}"
export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE="${SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE:-heuristic_graph_dsl}"
export SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE="${SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE:-similarity_first}"
export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS="${SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS:-1}"
export SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE="${SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE:-64}"
export SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT="${SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT:-32}"
export SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY:-160}"
export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT:-256}"
export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW:-4096}"
export SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="${SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES:-20}"
export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME:-succ-external-mumo-graph-edit}"

ACCOUNT="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_TIME:-${SUCC_SLURM_TIME:-12:00:00}}"
MEM="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_MEM:-${SUCC_SLURM_MEM:-64G}}"
CPUS="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_CPUS:-${SUCC_SLURM_CPUS:-8}}"
JOB_NAME="$SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
GPU_PROFILE="${SUCC_EXTERNAL_GRAPH_EDIT_GPU_PROFILE:-${SUCC_GPU_PROFILE:-none}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "$SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE" || ! -f "$SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE" ]]; then
  echo "ERROR: source file not found: $SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE" >&2
  exit 2
fi

if [[ -n "${SUCC_EXTERNAL_GRAPH_EDIT_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_EXTERNAL_GRAPH_EDIT_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "none" || "$GPU_PROFILE" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting MuMO GraphEditDSL agent benchmark"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  source_file=$SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE"
echo "  output_dir=$SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR"
echo "  direct_prediction_csv=$SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV"
echo "  require_direct_proposals=$SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS"
echo "  task_split=$SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT"
echo "  max_rows_per_task=$SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK"
echo "  planner_mode=$SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE"
echo "  selection_mode=$SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE"
echo "  planner_steps=$SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS"
echo "  beam_size=$SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE"
echo "  site_limit=$SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT"
echo "  max_plans_per_property=$SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY"
echo "  max_candidates_per_parent=$SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT"
echo "  max_candidates_per_row=$SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW"
echo "  top_k_candidates=$SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-none}"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  if [[ -n "$GPU_REQUEST" ]]; then
    echo "Trying sbatch with --gpus=$GPU_REQUEST"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_external_mumo_graph_edit_agent.sh'")"; then
      continue
    fi
  else
    echo "Trying sbatch without GPU request"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_external_mumo_graph_edit_agent.sh'")"; then
      continue
    fi
  fi
  echo "$output"
  job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit MuMO GraphEditDSL agent benchmark." >&2
  exit 1
fi

echo "external_mumo_graph_edit_agent_job=$job_id"
