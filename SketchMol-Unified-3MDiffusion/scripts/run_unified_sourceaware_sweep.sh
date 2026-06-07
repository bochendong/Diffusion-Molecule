#!/usr/bin/env bash
# Run a packed sweep of short Unified 3M source-aware connector experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

SWEEP_OUTPUT_ROOT="${SMU3M_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2}"
SWEEP_LAUNCHER="${SMU3M_SWEEP_LAUNCHER:-glost}"
SWEEP_CONCURRENCY="${SMU3M_SWEEP_CONCURRENCY:-1}"
SWEEP_RESUME="${SMU3M_SWEEP_RESUME:-0}"
SWEEP_DRY_RUN="${SMU3M_SWEEP_DRY_RUN:-0}"
SWEEP_PRESET="${SMU3M_SWEEP_PRESET:-extended}"
SWEEP_CONFIGS="${SMU3M_SWEEP_CONFIGS:-}"

join_configs() {
  local IFS=';'
  printf '%s' "$*"
}

default_sweep_configs() {
  local preset="$1"
  local configs=()
  case "$preset" in
    compact)
      configs=(
        "baseline:0:0:0:0.2:0.07"
        "sim005_head:0.05:0:0:0.2:0.07"
        "sim015_head:0.15:0:0:0.2:0.07"
        "sim030_head:0.30:0:0:0.2:0.07"
        "hard005_head:0:0.05:0:0.2:0.07"
        "hard010_head:0:0.10:0:0.2:0.07"
        "balanced_head:0.15:0.05:0:0.2:0.07"
        "strong_head:0.30:0.10:0:0.2:0.07"
        "shared_low:0.05:0.02:1:0.2:0.07"
      )
      ;;
    extended)
      configs=(
        "baseline:0:0:0:0.2:0.07"
        "sim002_head:0.02:0:0:0.2:0.07"
        "sim005_head:0.05:0:0:0.2:0.07"
        "sim010_head:0.10:0:0:0.2:0.07"
        "sim015_head:0.15:0:0:0.2:0.07"
        "sim020_head:0.20:0:0:0.2:0.07"
        "sim030_head:0.30:0:0:0.2:0.07"
        "sim045_head:0.45:0:0:0.2:0.07"
        "hard001_head:0:0.01:0:0.2:0.07"
        "hard002_head:0:0.02:0:0.2:0.07"
        "hard005_head:0:0.05:0:0.2:0.07"
        "hard010_head:0:0.10:0:0.2:0.07"
        "hard015_head:0:0.15:0:0.2:0.07"
        "balanced_005_001:0.05:0.01:0:0.2:0.07"
        "balanced_005_002:0.05:0.02:0:0.2:0.07"
        "balanced_010_002:0.10:0.02:0:0.2:0.07"
        "balanced_015_003:0.15:0.03:0:0.2:0.07"
        "balanced_015_005:0.15:0.05:0:0.2:0.07"
        "balanced_020_005:0.20:0.05:0:0.2:0.07"
        "balanced_030_005:0.30:0.05:0:0.2:0.07"
        "balanced_030_010:0.30:0.10:0:0.2:0.07"
        "temp_cool_015_005:0.15:0.05:0:0.2:0.05"
        "temp_warm_015_005:0.15:0.05:0:0.2:0.10"
        "margin_low_015_005:0.15:0.05:0:0.1:0.07"
        "margin_high_015_005:0.15:0.05:0:0.4:0.07"
        "shared_tiny:0.02:0.01:1:0.2:0.07"
        "shared_low:0.05:0.02:1:0.2:0.07"
        "shared_mid:0.10:0.03:1:0.2:0.07"
        "strong_045_015:0.45:0.15:0:0.2:0.07"
        "strong_sim_cool:0.45:0:0:0.2:0.05"
        "hard_cool:0:0.05:0:0.2:0.05"
      )
      ;;
    *)
      echo "ERROR: unsupported SMU3M_SWEEP_PRESET=$preset" >&2
      echo "Use compact, extended, or set SMU3M_SWEEP_CONFIGS directly." >&2
      return 2
      ;;
  esac
  join_configs "${configs[@]}"
}

if [[ -z "$SWEEP_CONFIGS" ]]; then
  if ! SWEEP_CONFIGS="$(default_sweep_configs "$SWEEP_PRESET")"; then
    exit 2
  fi
  SWEEP_CONFIG_SOURCE="preset:$SWEEP_PRESET"
else
  SWEEP_CONFIG_SOURCE="custom"
fi

if (( SWEEP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_SWEEP_CONCURRENCY must be positive, got $SWEEP_CONCURRENCY" >&2
  exit 2
fi

mkdir -p "$SWEEP_OUTPUT_ROOT/tasks" "$SWEEP_OUTPUT_ROOT/logs"
TASK_FILE="$SWEEP_OUTPUT_ROOT/tasks/sourceaware_sweep.tasks"
: > "$TASK_FILE"

echo "Running Unified 3M source-aware sweep"
echo "  output_root=$SWEEP_OUTPUT_ROOT"
echo "  launcher=$SWEEP_LAUNCHER"
echo "  concurrency=$SWEEP_CONCURRENCY"
echo "  resume=$SWEEP_RESUME"
echo "  dry_run=$SWEEP_DRY_RUN"
echo "  config_source=$SWEEP_CONFIG_SOURCE"
echo "  configs=$SWEEP_CONFIGS"

task_scripts=()
IFS=';' read -r -a configs <<< "$SWEEP_CONFIGS"
for config in "${configs[@]}"; do
  [[ -z "$config" ]] && continue
  IFS=':' read -r label source_weight hard_weight shared_gradient margin temperature <<< "$config"
  if [[ -z "${label:-}" || -z "${source_weight:-}" || -z "${hard_weight:-}" ]]; then
    echo "ERROR: invalid sweep config '$config'; expected label:source_weight:hard_weight:shared_gradient:margin:temperature" >&2
    exit 2
  fi
  shared_gradient="${shared_gradient:-0}"
  margin="${margin:-0.2}"
  temperature="${temperature:-0.07}"
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"
  task_script="$SWEEP_OUTPUT_ROOT/tasks/${safe_label}.sh"
  task_log="$SWEEP_OUTPUT_ROOT/logs/${safe_label}.log"
  task_output_dir="$SWEEP_OUTPUT_ROOT/$safe_label"
  cat > "$task_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
mkdir -p "$(dirname "$task_log")"
exec > >(tee "$task_log") 2>&1
echo "source-aware sweep task: $safe_label"
echo "  source_similarity_loss_weight=$source_weight"
echo "  hard_negative_loss_weight=$hard_weight"
echo "  source_aware_shared_gradient=$shared_gradient"
echo "  hard_negative_margin=$margin"
echo "  source_aware_temperature=$temperature"
echo "  output_dir=$task_output_dir"
export SMU3M_OUTPUT_DIR="$task_output_dir"
export SMU3M_SOURCE_SIMILARITY_LOSS_WEIGHT="$source_weight"
export SMU3M_HARD_NEGATIVE_LOSS_WEIGHT="$hard_weight"
export SMU3M_SOURCE_AWARE_SHARED_GRADIENT="$shared_gradient"
export SMU3M_HARD_NEGATIVE_MARGIN="$margin"
export SMU3M_SOURCE_AWARE_TEMPERATURE="$temperature"
export SMU3M_RESUME="$SWEEP_RESUME"
bash "$PROJECT_DIR/scripts/run_unified_generation_smoke.sh"
EOF
  chmod +x "$task_script"
  task_scripts+=("$task_script")
  printf 'bash %q # %s\n' "$task_script" "$safe_label" >> "$TASK_FILE"
done

if (( ${#task_scripts[@]} == 0 )); then
  echo "ERROR: no sweep tasks were generated." >&2
  exit 2
fi
echo "  tasks=${#task_scripts[@]}"

if [[ "$SWEEP_DRY_RUN" == "1" ]]; then
  echo "Dry run requested; generated task file without executing tasks."
  echo "  task_file=$TASK_FILE"
  sed -n '1,120p' "$TASK_FILE"
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
    echo "ERROR: unsupported SMU3M_SWEEP_LAUNCHER=$SWEEP_LAUNCHER" >&2
    echo "Use glost, parallel, or serial." >&2
    exit 2
    ;;
esac

"${SMU3M_PYTHON_BIN:-python3}" - "$SWEEP_OUTPUT_ROOT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []


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


for metrics_path in sorted(root.glob("*/eval_latent/metrics.json")):
    label = metrics_path.parents[1].name
    eval_metrics = json.loads(metrics_path.read_text())
    edit_metrics_path = metrics_path.parents[1] / "edit_condition_tokens" / "metrics.json"
    edit_metrics = json.loads(edit_metrics_path.read_text()) if edit_metrics_path.exists() else {}
    overall = eval_metrics.get("overall", {})
    config = edit_metrics.get("config", {})
    last_edit = (edit_metrics.get("history") or [{}])[-1]
    row = {
        "label": label,
        "source_similarity_loss_weight": config.get("source_similarity_loss_weight", ""),
        "hard_negative_loss_weight": config.get("hard_negative_loss_weight", ""),
        "source_aware_shared_gradient": config.get("source_aware_shared_gradient", ""),
        "rows": eval_metrics.get("rows", overall.get("rows", "")),
        "target_fingerprint_cosine": overall.get("target_fingerprint_cosine", ""),
        "source_fingerprint_cosine": overall.get("source_fingerprint_cosine", ""),
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
        "target_property_mae": overall.get("target_property_mae", ""),
        "prior_target_property_mae": overall.get("prior_target_property_mae", ""),
        "property_mae_minus_prior": _diff_or_blank(
            overall.get("target_property_mae", ""),
            overall.get("prior_target_property_mae", ""),
        ),
        "delta_mae": overall.get("delta_mae", ""),
        "prior_delta_mae": overall.get("prior_delta_mae", ""),
        "delta_mae_minus_prior": _diff_or_blank(
            overall.get("delta_mae", ""),
            overall.get("prior_delta_mae", ""),
        ),
        "latent_mae": overall.get("latent_mae", ""),
        "generated_minus_prior_latent_mae": overall.get("generated_minus_prior_latent_mae", ""),
        "train_source_similarity_mse": last_edit.get("train_source_similarity_mse", ""),
        "train_source_aware_hard_negative": last_edit.get("train_source_aware_hard_negative", ""),
        "eval_metrics": str(metrics_path),
    }
    rows.append(row)

fieldnames = [
    "label",
    "source_similarity_loss_weight",
    "hard_negative_loss_weight",
    "source_aware_shared_gradient",
    "rows",
    "target_fingerprint_cosine",
    "source_fingerprint_cosine",
    "source_target_fingerprint_cosine",
    "prior_target_fingerprint_cosine",
    "target_fp_gain_vs_source_target",
    "target_fp_gain_vs_prior",
    "target_property_mae",
    "prior_target_property_mae",
    "property_mae_minus_prior",
    "delta_mae",
    "prior_delta_mae",
    "delta_mae_minus_prior",
    "latent_mae",
    "generated_minus_prior_latent_mae",
    "train_source_similarity_mse",
    "train_source_aware_hard_negative",
    "eval_metrics",
]
csv_path = root / "sweep_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "sweep_summary.md"
with md_path.open("w", encoding="utf-8") as handle:
    handle.write("# Unified 3M Source-Aware Sweep Summary\n\n")
    handle.write(f"- output root: `{root}`\n")
    handle.write(f"- tasks completed: `{len(rows)}`\n\n")
    handle.write(
        "| label | source w | hard w | shared grad | target fp cos | source fp cos | "
        "source-target fp | target fp gain | property MAE | prop vs prior | delta MAE | delta vs prior |\n"
    )
    handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        handle.write(
            "| {label} | {source_similarity_loss_weight} | {hard_negative_loss_weight} | "
            "{source_aware_shared_gradient} | {target_fingerprint_cosine} | "
            "{source_fingerprint_cosine} | {source_target_fingerprint_cosine} | "
            "{target_fp_gain_vs_source_target} | {target_property_mae} | "
            "{property_mae_minus_prior} | {delta_mae} | {delta_mae_minus_prior} |\n".format(
                **{key: _cell(value) for key, value in row.items()}
            )
        )

print(f"sweep_summary_csv={csv_path}")
print(f"sweep_summary_md={md_path}")
PY

echo "Source-aware sweep finished."
echo "  task_file=$TASK_FILE"
echo "  summary=$SWEEP_OUTPUT_ROOT/sweep_summary.md"
