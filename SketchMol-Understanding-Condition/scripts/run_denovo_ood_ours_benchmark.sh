#!/usr/bin/env bash
# Run de novo OOD property-design benchmark through the trained SUCC UniVideo model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SUCC_OOD_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_ood_ours_v1}"
MODEL_OUTPUT_DIR="${SUCC_OOD_MODEL_OUTPUT_DIR:-${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v3_attack}}"
RESUME_CHECKPOINT="${SUCC_OOD_RESUME_CHECKPOINT:-${SUCC_RESUME_CHECKPOINT:-$MODEL_OUTPUT_DIR/univideo_molecule/univideo_molecule_generation.pt}}"
MOLECULE_DB="${SUCC_OOD_MOLECULE_DB_CSV:-$SMMED_DEFAULT_MOLECULE_DB}"
SPEC_JSON="${SUCC_OOD_SPEC_JSON:-}"
ROWS_CSV="${SUCC_OOD_ROWS_CSV:-$OUTPUT_DIR/denovo_ood_rows.csv}"
NEGATIVE_ROWS_CSV="${SUCC_OOD_NEGATIVE_ROWS_CSV:-$OUTPUT_DIR/denovo_ood_negative_rows.csv}"
CANDIDATE_CSV="${SUCC_OOD_CANDIDATE_CSV:-$OUTPUT_DIR/denovo_ood_candidate_rows.csv}"
EVAL_JSONL="${SUCC_OOD_EVAL_JSONL:-$OUTPUT_DIR/denovo_ood_eval.jsonl}"
NEGATIVE_JSONL="${SUCC_OOD_NEGATIVE_JSONL:-$OUTPUT_DIR/denovo_ood_negative_eval.jsonl}"
CANDIDATE_JSONL="${SUCC_OOD_CANDIDATE_JSONL:-$OUTPUT_DIR/denovo_ood_candidate_rows.jsonl}"
FEATURES_DIR="${SUCC_OOD_CONDITION_FEATURES_DIR:-$OUTPUT_DIR/condition_features_hf_vlm}"
NEGATIVE_FEATURES_DIR="${SUCC_OOD_NEGATIVE_CONDITION_FEATURES_DIR:-$OUTPUT_DIR/negative_condition_features_hf_vlm}"
OURS_EVAL_DIR="${SUCC_OOD_OURS_EVAL_DIR:-$OUTPUT_DIR/ours_eval_latent}"
CANDIDATE_LATENTS="${SUCC_OOD_CANDIDATE_LATENTS_NPY:-$OUTPUT_DIR/denovo_ood_candidate_target_latents.npy}"
ROWS_PER_SPEC="${SUCC_OOD_ROWS_PER_SPEC:-100}"
SEED="${SUCC_OOD_SEED:-17}"
CANDIDATE_LIMIT="${SUCC_OOD_CANDIDATE_LIMIT:-0}"
METHODS="${SUCC_OOD_MATERIALIZED_METHODS:-latent_nearest}"
METHOD_LABEL="${SUCC_OOD_METHOD_LABEL:-denovo_ood_succ_univideo_guided}"
GUIDANCE_SCALE="${SUCC_OOD_NEGATIVE_GUIDANCE_SCALE:-2.0}"
BENCHMARK_OUTPUT_DIR="${SUCC_OOD_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_ours}"
DIRECT_CSV="${SUCC_OOD_TARGET_MOLECULES_DIRECT_CSV:-$BENCHMARK_OUTPUT_DIR/target_molecules_direct.csv}"
TOP_K="${SUCC_OOD_TARGET_FINDER_TOP_K:-5}"
PROPERTY_RERANK_CANDIDATES="${SUCC_OOD_PROPERTY_RERANK_CANDIDATES:-4096}"
PROPERTY_RERANK_WEIGHT="${SUCC_OOD_PROPERTY_RERANK_WEIGHT:-10}"
STRICT_RERANK_WEIGHT="${SUCC_OOD_STRICT_RERANK_WEIGHT:-100}"
LATENT_RERANK_WEIGHT="${SUCC_OOD_LATENT_RERANK_WEIGHT:-1}"
FORCE_EXPORT="${SUCC_OOD_FORCE_EXPORT:-0}"
RUN_FEATURE_EXPORT="${SUCC_OOD_RUN_FEATURE_EXPORT:-auto}"
USE_CONDITION_FEATURES="${SUCC_OOD_USE_CONDITION_FEATURES:-1}"
HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
HF_ATTN_IMPLEMENTATION="${SUCC_HF_ATTN_IMPLEMENTATION:-}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"
LATENT_BACKEND="${SUCC_OOD_LATENT_BACKEND:-${SUCC_LATENT_BACKEND:-image_vae}}"
IMAGE_VAE_CHECKPOINT="${SUCC_IMAGE_VAE_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt}"
SKETCHMOL_ROOT="${SUCC_SKETCHMOL_ROOT:-SketchMol}"
SKETCHMOL_VAE_CONFIG="${SUCC_SKETCHMOL_VAE_CONFIG:-}"
SKETCHMOL_VAE_CHECKPOINT="${SUCC_SKETCHMOL_VAE_CHECKPOINT:-}"
SKETCHMOL_SCALE_FACTOR="${SUCC_SKETCHMOL_SCALE_FACTOR:-1.0}"
IMAGE_SIZE="${SUCC_IMAGE_SIZE:-256}"
VAE_BATCH_SIZE="${SUCC_VAE_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-48}"
SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
DEVICE="${SUCC_DEVICE:-auto}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

echo "De novo OOD SUCC UniVideo benchmark"
echo "  python=$PYTHON_BIN"
echo "  molecule_db=$MOLECULE_DB"
echo "  output_dir=$OUTPUT_DIR"
echo "  model_output_dir=$MODEL_OUTPUT_DIR"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  rows_csv=$ROWS_CSV"
echo "  negative_rows_csv=$NEGATIVE_ROWS_CSV"
echo "  candidate_csv=$CANDIDATE_CSV"
echo "  guidance_scale=$GUIDANCE_SCALE"
echo "  latent_backend=$LATENT_BACKEND"
echo "  methods=$METHODS"
echo "  property_rerank_candidates=$PROPERTY_RERANK_CANDIDATES"
echo "  property_rerank_weight=$PROPERTY_RERANK_WEIGHT"
echo "  strict_rerank_weight=$STRICT_RERANK_WEIGHT"
echo "  latent_rerank_weight=$LATENT_RERANK_WEIGHT"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"

if [[ ! -f "$MOLECULE_DB" ]]; then
  echo "ERROR: molecule database not found: $MOLECULE_DB" >&2
  exit 2
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "ERROR: missing trained SUCC checkpoint: $RESUME_CHECKPOINT" >&2
  exit 2
fi

if [[ "$FORCE_EXPORT" == "1" || ! -f "$ROWS_CSV" || ! -f "$NEGATIVE_ROWS_CSV" || ! -f "$CANDIDATE_CSV" ]]; then
  EXPORT_ARGS=(
    --molecule-db-csv "$MOLECULE_DB"
    --output-csv "$ROWS_CSV"
    --negative-output-csv "$NEGATIVE_ROWS_CSV"
    --candidate-output-csv "$CANDIDATE_CSV"
    --rows-per-spec "$ROWS_PER_SPEC"
    --seed "$SEED"
    --candidate-limit "$CANDIDATE_LIMIT"
  )
  if [[ -n "$SPEC_JSON" ]]; then
    EXPORT_ARGS+=(--spec-json "$SPEC_JSON")
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_ood_benchmark_rows.py" "${EXPORT_ARGS[@]}"
fi

if [[ "$FORCE_EXPORT" == "1" || ! -f "$EVAL_JSONL" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_2p7p_eval_jsonl.py" \
    --input-csv "$ROWS_CSV" \
    --output-jsonl "$EVAL_JSONL" \
    --split eval
fi
if [[ "$FORCE_EXPORT" == "1" || ! -f "$NEGATIVE_JSONL" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_2p7p_eval_jsonl.py" \
    --input-csv "$NEGATIVE_ROWS_CSV" \
    --output-jsonl "$NEGATIVE_JSONL" \
    --split eval
fi
if [[ "$FORCE_EXPORT" == "1" || ! -f "$CANDIDATE_JSONL" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_2p7p_eval_jsonl.py" \
    --input-csv "$CANDIDATE_CSV" \
    --output-jsonl "$CANDIDATE_JSONL" \
    --split candidate
fi

export_features() {
  local baseline_csv="$1"
  local output_dir="$2"
  local should_export=0
  if [[ "$RUN_FEATURE_EXPORT" == "1" || "$FORCE_EXPORT" == "1" ]]; then
    should_export=1
  elif [[ "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$output_dir/query_tokens.npy" ]]; then
    should_export=1
  fi
  if [[ "$should_export" != "1" ]]; then
    return
  fi
  if [[ -z "$HF_MODEL_NAME_OR_PATH" ]]; then
    echo "ERROR: SUCC_HF_MODEL_NAME_OR_PATH is required to export OOD HF VLM features." >&2
    exit 2
  fi
  export SUCC_ENCODER=hf_vlm
  export SUCC_VARIANTS=full
  export SUCC_BASELINE_CSV="$baseline_csv"
  export SUCC_OUTPUT_DIR="$output_dir"
  export SUCC_POOLED_DIM="$POOLED_DIM"
  export SUCC_NUM_QUERIES="$NUM_QUERIES"
  export SUCC_QUERY_DIM="$QUERY_DIM"
  export SUCC_HF_MODEL_NAME_OR_PATH="$HF_MODEL_NAME_OR_PATH"
  export SUCC_HF_DEVICE_MAP="$HF_DEVICE_MAP"
  export SUCC_HF_DTYPE="$HF_DTYPE"
  export SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE"
  export SUCC_HF_MAX_LENGTH="$HF_MAX_LENGTH"
  export SUCC_HF_RENDER_IMAGE_SIZE="$HF_RENDER_IMAGE_SIZE"
  if [[ -n "$HF_ATTN_IMPLEMENTATION" ]]; then
    export SUCC_HF_ATTN_IMPLEMENTATION="$HF_ATTN_IMPLEMENTATION"
  fi
  bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"
}

if [[ "$USE_CONDITION_FEATURES" == "1" ]]; then
  export_features "$ROWS_CSV" "$FEATURES_DIR"
  export_features "$NEGATIVE_ROWS_CSV" "$NEGATIVE_FEATURES_DIR"
fi

LATENT_BACKEND_ARGS=()
TARGET_LATENT_ARGS=()
if [[ "$LATENT_BACKEND" == "image_vae" ]]; then
  if [[ ! -f "$IMAGE_VAE_CHECKPOINT" ]]; then
    echo "ERROR: missing image VAE checkpoint: $IMAGE_VAE_CHECKPOINT" >&2
    exit 2
  fi
  LATENT_BACKEND_ARGS=(
    --latent-backend image_vae
    --image-vae-checkpoint "$IMAGE_VAE_CHECKPOINT"
    --image-size "$IMAGE_SIZE"
    --vae-batch-size "$VAE_BATCH_SIZE"
  )
  TARGET_LATENT_ARGS=(
    --latent-backend image_vae
    --image-vae-checkpoint "$IMAGE_VAE_CHECKPOINT"
    --image-size "$IMAGE_SIZE"
    --batch-size "$VAE_BATCH_SIZE"
  )
elif [[ "$LATENT_BACKEND" == "fingerprint_property_vector" ]]; then
  LATENT_BACKEND_ARGS=(--latent-backend fingerprint_property_vector)
  TARGET_LATENT_ARGS=(--latent-backend fingerprint_property_vector)
elif [[ "$LATENT_BACKEND" == "sketchmol_vae" ]]; then
  if [[ -z "$SKETCHMOL_VAE_CHECKPOINT" || -z "$SKETCHMOL_VAE_CONFIG" ]]; then
    echo "ERROR: SUCC_SKETCHMOL_VAE_CONFIG and SUCC_SKETCHMOL_VAE_CHECKPOINT are required for sketchmol_vae." >&2
    exit 2
  fi
  LATENT_BACKEND_ARGS=(
    --latent-backend sketchmol_vae
    --sketchmol-root "$SKETCHMOL_ROOT"
    --sketchmol-vae-config "$SKETCHMOL_VAE_CONFIG"
    --sketchmol-vae-checkpoint "$SKETCHMOL_VAE_CHECKPOINT"
    --sketchmol-scale-factor "$SKETCHMOL_SCALE_FACTOR"
    --image-size "$IMAGE_SIZE"
    --vae-batch-size "$VAE_BATCH_SIZE"
  )
  TARGET_LATENT_ARGS=(
    --latent-backend sketchmol_vae
    --sketchmol-root "$SKETCHMOL_ROOT"
    --sketchmol-vae-config "$SKETCHMOL_VAE_CONFIG"
    --sketchmol-vae-checkpoint "$SKETCHMOL_VAE_CHECKPOINT"
    --sketchmol-scale-factor "$SKETCHMOL_SCALE_FACTOR"
    --image-size "$IMAGE_SIZE"
    --batch-size "$VAE_BATCH_SIZE"
  )
else
  echo "ERROR: unsupported latent backend: $LATENT_BACKEND" >&2
  exit 2
fi

FEATURE_ARGS=()
NEGATIVE_FEATURE_ARGS=()
if [[ "$USE_CONDITION_FEATURES" == "1" ]]; then
  FEATURE_ARGS=(--condition-features-dir "$FEATURES_DIR" --condition-feature-array query_tokens --condition-feature-variant full)
  NEGATIVE_FEATURE_ARGS=(--negative-condition-features-dir "$NEGATIVE_FEATURES_DIR")
fi

mkdir -p "$OURS_EVAL_DIR"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_univideo_molecule_generation.py" \
  --train-jsonl "$EVAL_JSONL" \
  --eval-jsonl "$EVAL_JSONL" \
  --negative-eval-jsonl "$NEGATIVE_JSONL" \
  --negative-guidance-scale "$GUIDANCE_SCALE" \
  "${FEATURE_ARGS[@]}" \
  "${NEGATIVE_FEATURE_ARGS[@]}" \
  --output-dir "$OURS_EVAL_DIR" \
  --limit 1 \
  --eval-limit 100000000 \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$SAMPLE_STEPS" \
  --sample-eta "$SAMPLE_ETA" \
  --device "$DEVICE" \
  --eval-only \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  "${LATENT_BACKEND_ARGS[@]}"

if [[ "$FORCE_EXPORT" == "1" || ! -f "$CANDIDATE_LATENTS" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_univideo_target_latents.py" \
    --jsonl "$CANDIDATE_JSONL" \
    --output-npy "$CANDIDATE_LATENTS" \
    --device "$DEVICE" \
    "${TARGET_LATENT_ARGS[@]}"
fi

mkdir -p "$BENCHMARK_OUTPUT_DIR"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/materialize_univideo_target_molecules.py" \
  --source-csv "$ROWS_CSV" \
  --candidate-csv "$CANDIDATE_CSV" \
  --output-csv "$DIRECT_CSV" \
  --methods "$METHODS" \
  --generated-latents-npy "$OURS_EVAL_DIR/eval_latent/generated_latents.npy" \
  --candidate-latents-npy "$CANDIDATE_LATENTS" \
  --top-k "$TOP_K" \
  --property-rerank-candidates "$PROPERTY_RERANK_CANDIDATES" \
  --property-rerank-weight "$PROPERTY_RERANK_WEIGHT" \
  --strict-rerank-weight "$STRICT_RERANK_WEIGHT" \
  --latent-rerank-weight "$LATENT_RERANK_WEIGHT"

if [[ -n "$METHOD_LABEL" && "$METHODS" == "latent_nearest" ]]; then
  "$PYTHON_BIN" - "$DIRECT_CSV" "$METHOD_LABEL" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
method_label = sys.argv[2]
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
fieldnames = list(rows[0].keys()) if rows else []
for row in rows:
    row["method"] = method_label
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
  --image-csv "$DIRECT_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --method "$METHOD_LABEL" \
  --smiles-column generated_smiles \
  --report-title "SUCC UniVideo De Novo OOD Property-Design Benchmark" \
  --benchmark-family "denovo_ood_property_design" \
  --benchmark-task "denovo_ood_property_design" \
  --accept-direct-smiles \
  --hide-source-similarity-section

echo
echo "SUCC de novo OOD benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo "  direct_csv=$DIRECT_CSV"
echo "  generated_latents=$OURS_EVAL_DIR/eval_latent/generated_latents.npy"
echo "  candidate_latents=$CANDIDATE_LATENTS"
echo
sed -n '1,100p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
