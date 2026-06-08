#!/usr/bin/env bash
# Run packed source-anchored fingerprint diffusion refinements.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

BASE_OUTPUT_DIR="${SMU3M_BASE_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_sourceguard_v1}"
SWEEP_OUTPUT_ROOT="${SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1}"
SWEEP_LAUNCHER="${SMU3M_SOURCEANCHOR_SWEEP_LAUNCHER:-glost}"
SWEEP_CONCURRENCY="${SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY:-1}"
SWEEP_DRY_RUN="${SMU3M_SOURCEANCHOR_SWEEP_DRY_RUN:-0}"
SWEEP_CONFIGS="${SMU3M_SOURCEANCHOR_SWEEP_CONFIGS:-}"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-python3}"

join_configs() {
  local IFS=';'
  printf '%s' "$*"
}

default_configs() {
  local configs=(
    "blend070_guard025_p005:0.70:0.25:0.05:30:11"
    "blend085_guard050_p005:0.85:0.50:0.05:30:11"
    "blend095_guard050_p005:0.95:0.50:0.05:30:11"
    "blend085_guard100_p005:0.85:1.00:0.05:30:11"
    "blend085_guard050_p025:0.85:0.50:0.25:30:11"
    "blend095_guard100_p025:0.95:1.00:0.25:30:11"
  )
  join_configs "${configs[@]}"
}

if (( SWEEP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY must be positive, got $SWEEP_CONCURRENCY" >&2
  exit 2
fi
if [[ -z "$SWEEP_CONFIGS" ]]; then
  SWEEP_CONFIGS="$(default_configs)"
fi

for required in \
  "$BASE_OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  "$BASE_OUTPUT_DIR/dataset/unified_condition_eval.jsonl" \
  "$BASE_OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  "$BASE_OUTPUT_DIR/latent_diffusion/checkpoints/latest.pt"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required base artifact: $required" >&2
    exit 2
  fi
done

mkdir -p "$SWEEP_OUTPUT_ROOT/tasks" "$SWEEP_OUTPUT_ROOT/logs"
TASK_FILE="$SWEEP_OUTPUT_ROOT/tasks/sourceanchor_sweep.tasks"
MANIFEST_CSV="$SWEEP_OUTPUT_ROOT/tasks/sourceanchor_sweep_manifest.csv"
: > "$TASK_FILE"
printf 'label,output_dir,source_fingerprint_prior_blend,fingerprint_guard_loss_weight,prior_loss_weight,extra_epochs,seed\n' > "$MANIFEST_CSV"

echo "Running Unified 3M source-anchor sweep"
echo "  base_output_dir=$BASE_OUTPUT_DIR"
echo "  output_root=$SWEEP_OUTPUT_ROOT"
echo "  launcher=$SWEEP_LAUNCHER"
echo "  concurrency=$SWEEP_CONCURRENCY"
echo "  dry_run=$SWEEP_DRY_RUN"

task_scripts=()

seed_plus() {
  local seed="$1"
  local offset="$2"
  echo "$((seed + offset))"
}

write_task() {
  local config="$1"
  IFS=':' read -r label source_fp_blend fingerprint_guard prior_weight extra_epochs seed <<< "$config"
  if [[ -z "${label:-}" || -z "${source_fp_blend:-}" || -z "${fingerprint_guard:-}" || -z "${prior_weight:-}" ]]; then
    echo "ERROR: invalid source-anchor config '$config'; expected label:source_fp_blend:fingerprint_guard:prior_weight:extra_epochs:seed" >&2
    exit 2
  fi
  extra_epochs="${extra_epochs:-30}"
  seed="${seed:-11}"
  local safe_label task_script task_log task_output_dir
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"
  task_script="$SWEEP_OUTPUT_ROOT/tasks/${safe_label}.sh"
  task_log="$SWEEP_OUTPUT_ROOT/logs/${safe_label}.log"
  task_output_dir="$SWEEP_OUTPUT_ROOT/$safe_label"
  cat > "$task_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
mkdir -p "$(dirname "$task_log")" "$task_output_dir"
exec > >(tee "$task_log") 2>&1
echo "source-anchor diffusion task: $safe_label"
echo "  base_output_dir=$BASE_OUTPUT_DIR"
echo "  source_fingerprint_prior_blend=$source_fp_blend"
echo "  fingerprint_guard_loss_weight=$fingerprint_guard"
echo "  prior_loss_weight=$prior_weight"
echo "  extra_epochs=$extra_epochs"
echo "  output_dir=$task_output_dir"
export SMU3M_OUTPUT_DIR="$BASE_OUTPUT_DIR"
export SMU3M_BASE_DIFFUSION_DIR="$BASE_OUTPUT_DIR/latent_diffusion"
export SMU3M_DIFFUSION_DIR="$task_output_dir/latent_diffusion"
export SMU3M_EVAL_LATENT_DIR="$task_output_dir/eval_latent"
export SMU3M_SOURCE_FINGERPRINT_PRIOR_BLEND="$source_fp_blend"
export SMU3M_FINGERPRINT_GUARD_LOSS_WEIGHT="$fingerprint_guard"
export SMU3M_FINGERPRINT_GUARD_MARGIN="${SMU3M_SOURCEANCHOR_FINGERPRINT_GUARD_MARGIN:-0.02}"
export SMU3M_PRIOR_LOSS_WEIGHT="$prior_weight"
export SMU3M_SOURCE_REGRET_LOSS_WEIGHT="${SMU3M_SOURCEANCHOR_SOURCE_REGRET_LOSS_WEIGHT:-0.35}"
export SMU3M_SOURCE_RADIUS_LOSS_WEIGHT="${SMU3M_SOURCEANCHOR_SOURCE_RADIUS_LOSS_WEIGHT:-0.10}"
export SMU3M_TRAIN_DIFFUSION_CONNECTOR="${SMU3M_SOURCEANCHOR_TRAIN_DIFFUSION_CONNECTOR:-1}"
export SMU3M_DIFFUSION_EXTRA_EPOCHS="$extra_epochs"
export SMU3M_DIFFUSION_SEED="$(seed_plus "$seed" 2)"
export SMU3M_EVAL_SEED="$(seed_plus "$seed" 3)"
export SMU3M_RESUME="1"
export SMU3M_ALLOW_INCOMPATIBLE_RESUME_WEIGHTS="1"
export SMU3M_RUN_MATERIALIZED_BENCHMARK="0"
export SMU3M_EVAL_LIMIT="${SMU3M_SOURCEANCHOR_EVAL_LIMIT:-0}"
export SMU3M_MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_SOURCEANCHOR_MAX_EVAL_PER_PROPERTY_COUNT:-250}"
bash "$PROJECT_DIR/scripts/run_unified_diffusion_refine.sh"
EOF
  chmod +x "$task_script"
  task_scripts+=("$task_script")
  printf 'bash %q # %s\n' "$task_script" "$safe_label" >> "$TASK_FILE"
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$safe_label" "$task_output_dir" "$source_fp_blend" "$fingerprint_guard" "$prior_weight" "$extra_epochs" "$seed" >> "$MANIFEST_CSV"
}

IFS=';' read -r -a configs <<< "$SWEEP_CONFIGS"
for config in "${configs[@]}"; do
  [[ -z "$config" ]] && continue
  write_task "$config"
done

if (( ${#task_scripts[@]} == 0 )); then
  echo "ERROR: no source-anchor tasks were generated." >&2
  exit 2
fi
echo "  tasks=${#task_scripts[@]}"
echo "  task_file=$TASK_FILE"
echo "  manifest=$MANIFEST_CSV"

if [[ "$SWEEP_DRY_RUN" == "1" ]]; then
  sed -n '1,160p' "$TASK_FILE"
  exit 0
fi

run_with_parallel_fallback() {
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\0' "${task_scripts[@]}" | parallel -0 -j "$SWEEP_CONCURRENCY" bash {}
  elif command -v xargs >/dev/null 2>&1; then
    printf '%s\0' "${task_scripts[@]}" | xargs -0 -n 1 -P "$SWEEP_CONCURRENCY" bash
  else
    for task_script in "${task_scripts[@]}"; do
      bash "$task_script"
    done
  fi
}

case "$SWEEP_LAUNCHER" in
  glost)
    if command -v module >/dev/null 2>&1; then
      module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 glost/0.3.1 >/dev/null 2>&1 \
        || module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 glost/0.3.1 >/dev/null 2>&1 \
        || module load glost >/dev/null 2>&1 \
        || true
    fi
    if command -v glost_launch >/dev/null 2>&1; then
      if command -v srun >/dev/null 2>&1; then
        srun glost_launch "$TASK_FILE"
      else
        glost_launch "$TASK_FILE"
      fi
    else
      echo "GLOST not available; falling back to GNU parallel/xargs."
      run_with_parallel_fallback
    fi
    ;;
  parallel)
    run_with_parallel_fallback
    ;;
  serial)
    for task_script in "${task_scripts[@]}"; do
      bash "$task_script"
    done
    ;;
  *)
    echo "ERROR: unsupported SMU3M_SOURCEANCHOR_SWEEP_LAUNCHER=$SWEEP_LAUNCHER" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" - "$SWEEP_OUTPUT_ROOT" "$MANIFEST_CSV" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
manifest = {}
if manifest_path.exists():
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            manifest[row["label"]] = row


def _float_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _diff_or_blank(left, right):
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return ""
    return left_float - right_float


def _cell(value):
    value_float = _float_or_none(value)
    if value_float is None:
        return value if value != "" else ""
    return f"{value_float:.4g}"


rows = []
for metrics_path in sorted(root.glob("*/eval_latent/metrics.json")):
    label = metrics_path.parents[1].name
    eval_metrics = json.loads(metrics_path.read_text())
    overall = eval_metrics.get("overall", {})
    config = manifest.get(label, {})
    row = {
        "label": label,
        "source_fingerprint_prior_blend": config.get(
            "source_fingerprint_prior_blend", overall.get("source_fingerprint_prior_blend", "")
        ),
        "fingerprint_guard_loss_weight": config.get("fingerprint_guard_loss_weight", ""),
        "prior_loss_weight": config.get("prior_loss_weight", ""),
        "extra_epochs": config.get("extra_epochs", ""),
        "rows": eval_metrics.get("rows", overall.get("rows", "")),
        "target_fingerprint_cosine": overall.get("target_fingerprint_cosine", ""),
        "source_target_fingerprint_cosine": overall.get("source_target_fingerprint_cosine", ""),
        "prior_target_fingerprint_cosine": overall.get("prior_target_fingerprint_cosine", ""),
        "target_fp_gain_vs_source_target": _diff_or_blank(
            overall.get("target_fingerprint_cosine", ""),
            overall.get("source_target_fingerprint_cosine", ""),
        ),
        "target_fp_gain_vs_prior": _diff_or_blank(
            overall.get("target_fingerprint_cosine", ""),
            overall.get("prior_target_fingerprint_cosine", ""),
        ),
        "fingerprint_cosine_gain_vs_source": overall.get("fingerprint_cosine_gain_vs_source", ""),
        "fingerprint_beats_source": overall.get("fingerprint_beats_source", ""),
        "target_property_mae": overall.get("target_property_mae", ""),
        "source_target_property_mae": overall.get("source_target_property_mae", ""),
        "prior_target_property_mae": overall.get("prior_target_property_mae", ""),
        "property_mae_gain_vs_source": overall.get("property_mae_gain_vs_source", ""),
        "property_mae_beats_source": overall.get("property_mae_beats_source", ""),
        "generated_minus_prior_latent_mae": overall.get("generated_minus_prior_latent_mae", ""),
        "eval_metrics": str(metrics_path),
    }
    rows.append(row)

fieldnames = [
    "label",
    "source_fingerprint_prior_blend",
    "fingerprint_guard_loss_weight",
    "prior_loss_weight",
    "extra_epochs",
    "rows",
    "target_fingerprint_cosine",
    "source_target_fingerprint_cosine",
    "prior_target_fingerprint_cosine",
    "target_fp_gain_vs_source_target",
    "target_fp_gain_vs_prior",
    "fingerprint_cosine_gain_vs_source",
    "fingerprint_beats_source",
    "target_property_mae",
    "source_target_property_mae",
    "prior_target_property_mae",
    "property_mae_gain_vs_source",
    "property_mae_beats_source",
    "generated_minus_prior_latent_mae",
    "eval_metrics",
]

csv_path = root / "sourceanchor_sweep_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "sourceanchor_sweep_summary.md"
with md_path.open("w", encoding="utf-8") as handle:
    handle.write("# Unified 3M Source-Anchor Sweep Summary\n\n")
    handle.write(f"- output root: `{root}`\n")
    handle.write(f"- tasks completed: `{len(rows)}`\n\n")
    handle.write(
        "| label | fp blend | fp guard | prior w | target fp | source-target fp | "
        "prior fp | fp gain vs source | fp beats source | property MAE | prop gain vs source |\n"
    )
    handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        handle.write(
            "| {label} | {source_fingerprint_prior_blend} | {fingerprint_guard_loss_weight} | "
            "{prior_loss_weight} | {target_fingerprint_cosine} | {source_target_fingerprint_cosine} | "
            "{prior_target_fingerprint_cosine} | {fingerprint_cosine_gain_vs_source} | "
            "{fingerprint_beats_source} | {target_property_mae} | {property_mae_gain_vs_source} |\n".format(
                **{key: _cell(value) for key, value in row.items()}
            )
        )

print(f"sourceanchor_sweep_summary_csv={csv_path}")
print(f"sourceanchor_sweep_summary_md={md_path}")
PY

echo
echo "Source-anchor sweep finished:"
echo "  output_root=$SWEEP_OUTPUT_ROOT"
echo "  manifest=$MANIFEST_CSV"
echo "  summary=$SWEEP_OUTPUT_ROOT/sourceanchor_sweep_summary.md"
