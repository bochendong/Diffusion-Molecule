#!/usr/bin/env bash
# Run packed follow-up experiments for source-aware Unified 3M winners.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

FOLLOWUP_OUTPUT_ROOT="${SMU3M_FOLLOWUP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1}"
SWEEP_OUTPUT_ROOT="${SMU3M_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2}"
FOLLOWUP_PLAN="${SMU3M_FOLLOWUP_PLAN:-all}"
FOLLOWUP_LAUNCHER="${SMU3M_FOLLOWUP_LAUNCHER:-glost}"
FOLLOWUP_CONCURRENCY="${SMU3M_FOLLOWUP_CONCURRENCY:-1}"
FOLLOWUP_DRY_RUN="${SMU3M_FOLLOWUP_DRY_RUN:-0}"
FOLLOWUP_RESUME="${SMU3M_FOLLOWUP_RESUME:-0}"
FULL_CONFIGS="${SMU3M_FOLLOWUP_FULL_CONFIGS:-}"
DIFFUSION_CONFIGS="${SMU3M_FOLLOWUP_DIFFUSION_CONFIGS:-}"

join_configs() {
  local IFS=';'
  printf '%s' "$*"
}

default_full_configs() {
  local configs=(
    "baseline_full_s11:0:0:0:0.2:0.07:11"
    "hard002_full_s11:0:0.02:0:0.2:0.07:11"
    "hard002_full_s23:0:0.02:0:0.2:0.07:23"
    "hard002_full_s37:0:0.02:0:0.2:0.07:37"
    "balanced005001_full_s11:0.05:0.01:0:0.2:0.07:11"
    "sharedtiny_full_s11:0.02:0.01:1:0.2:0.07:11"
  )
  join_configs "${configs[@]}"
}

default_diffusion_configs() {
  local configs=(
    "hard002_freeze_p000_steps1:hard002_head:0:0:1:5:11"
    "hard002_freeze_p000_steps5:hard002_head:0:0:5:5:11"
    "hard002_freeze_p000_steps20:hard002_head:0:0:20:5:11"
    "hard002_joint_p025_steps20:hard002_head:0.25:1:20:20:11"
    "hard002_joint_p050_steps20:hard002_head:0.5:1:20:20:11"
    "hard002_joint_p100_steps20:hard002_head:1.0:1:20:20:11"
  )
  join_configs "${configs[@]}"
}

case "$FOLLOWUP_PLAN" in
  all | full | diffusion) ;;
  *)
    echo "ERROR: unsupported SMU3M_FOLLOWUP_PLAN=$FOLLOWUP_PLAN" >&2
    echo "Use all, full, or diffusion." >&2
    exit 2
    ;;
esac
if (( FOLLOWUP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_FOLLOWUP_CONCURRENCY must be positive, got $FOLLOWUP_CONCURRENCY" >&2
  exit 2
fi
if [[ -z "$FULL_CONFIGS" ]]; then
  FULL_CONFIGS="$(default_full_configs)"
fi
if [[ -z "$DIFFUSION_CONFIGS" ]]; then
  DIFFUSION_CONFIGS="$(default_diffusion_configs)"
fi

mkdir -p "$FOLLOWUP_OUTPUT_ROOT/tasks" "$FOLLOWUP_OUTPUT_ROOT/logs"
TASK_FILE="$FOLLOWUP_OUTPUT_ROOT/tasks/sourceaware_followup.tasks"
MANIFEST_CSV="$FOLLOWUP_OUTPUT_ROOT/tasks/sourceaware_followup_manifest.csv"
: > "$TASK_FILE"
printf 'label,kind,output_dir,base_output_dir,config\n' > "$MANIFEST_CSV"

echo "Running Unified 3M source-aware follow-up"
echo "  output_root=$FOLLOWUP_OUTPUT_ROOT"
echo "  sweep_output_root=$SWEEP_OUTPUT_ROOT"
echo "  plan=$FOLLOWUP_PLAN"
echo "  launcher=$FOLLOWUP_LAUNCHER"
echo "  concurrency=$FOLLOWUP_CONCURRENCY"
echo "  resume=$FOLLOWUP_RESUME"
echo "  dry_run=$FOLLOWUP_DRY_RUN"

task_scripts=()

seed_plus() {
  local seed="$1"
  local offset="$2"
  echo "$((seed + offset))"
}

write_full_task() {
  local config="$1"
  IFS=':' read -r label source_weight hard_weight shared_gradient margin temperature seed <<< "$config"
  if [[ -z "${label:-}" || -z "${source_weight:-}" || -z "${hard_weight:-}" || -z "${seed:-}" ]]; then
    echo "ERROR: invalid full config '$config'; expected label:source_weight:hard_weight:shared_gradient:margin:temperature:seed" >&2
    exit 2
  fi
  shared_gradient="${shared_gradient:-0}"
  margin="${margin:-0.2}"
  temperature="${temperature:-0.07}"
  local safe_label task_script task_log task_output_dir
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"
  task_script="$FOLLOWUP_OUTPUT_ROOT/tasks/${safe_label}.sh"
  task_log="$FOLLOWUP_OUTPUT_ROOT/logs/${safe_label}.log"
  task_output_dir="$FOLLOWUP_OUTPUT_ROOT/$safe_label"
  cat > "$task_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
mkdir -p "$(dirname "$task_log")"
exec > >(tee "$task_log") 2>&1
echo "source-aware follow-up full task: $safe_label"
echo "  source_similarity_loss_weight=$source_weight"
echo "  hard_negative_loss_weight=$hard_weight"
echo "  source_aware_shared_gradient=$shared_gradient"
echo "  seed=$seed"
echo "  output_dir=$task_output_dir"
export SMU3M_OUTPUT_DIR="$task_output_dir"
export SMU3M_SOURCE_SIMILARITY_LOSS_WEIGHT="$source_weight"
export SMU3M_HARD_NEGATIVE_LOSS_WEIGHT="$hard_weight"
export SMU3M_SOURCE_AWARE_SHARED_GRADIENT="$shared_gradient"
export SMU3M_HARD_NEGATIVE_MARGIN="$margin"
export SMU3M_SOURCE_AWARE_TEMPERATURE="$temperature"
export SMU3M_ALIGNMENT_SEED="$(seed_plus "$seed" 0)"
export SMU3M_EDIT_SEED="$(seed_plus "$seed" 1)"
export SMU3M_DIFFUSION_SEED="$(seed_plus "$seed" 2)"
export SMU3M_EVAL_SEED="$(seed_plus "$seed" 3)"
export SMU3M_RESUME="$FOLLOWUP_RESUME"
export SMU3M_EVAL_LIMIT="${SMU3M_FOLLOWUP_FULL_EVAL_LIMIT:-0}"
export SMU3M_MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_FOLLOWUP_FULL_MAX_EVAL_PER_PROPERTY_COUNT:-0}"
bash "$PROJECT_DIR/scripts/run_unified_generation_smoke.sh"
EOF
  chmod +x "$task_script"
  task_scripts+=("$task_script")
  printf 'bash %q # %s\n' "$task_script" "$safe_label" >> "$TASK_FILE"
  printf '%s,full,%s,,%s\n' "$safe_label" "$task_output_dir" "$config" >> "$MANIFEST_CSV"
}

write_diffusion_task() {
  local config="$1"
  IFS=':' read -r label base_label prior_weight train_connector sample_steps extra_epochs seed <<< "$config"
  if [[ -z "${label:-}" || -z "${base_label:-}" || -z "${prior_weight:-}" || -z "${train_connector:-}" ]]; then
    echo "ERROR: invalid diffusion config '$config'; expected label:base_label:prior_weight:train_connector:sample_steps:extra_epochs:seed" >&2
    exit 2
  fi
  sample_steps="${sample_steps:-20}"
  extra_epochs="${extra_epochs:-5}"
  seed="${seed:-11}"
  local safe_label task_script task_log task_output_dir base_output_dir
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"
  task_script="$FOLLOWUP_OUTPUT_ROOT/tasks/${safe_label}.sh"
  task_log="$FOLLOWUP_OUTPUT_ROOT/logs/${safe_label}.log"
  task_output_dir="$FOLLOWUP_OUTPUT_ROOT/$safe_label"
  base_output_dir="$SWEEP_OUTPUT_ROOT/$base_label"
  cat > "$task_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
mkdir -p "$(dirname "$task_log")" "$task_output_dir"
exec > >(tee "$task_log") 2>&1
echo "source-aware follow-up diffusion task: $safe_label"
echo "  base_output_dir=$base_output_dir"
echo "  prior_loss_weight=$prior_weight"
echo "  train_diffusion_connector=$train_connector"
echo "  sample_steps=$sample_steps"
echo "  extra_epochs=$extra_epochs"
echo "  output_dir=$task_output_dir"
for required in "$base_output_dir/dataset/unified_condition_train.jsonl" "$base_output_dir/dataset/unified_condition_eval.jsonl" "$base_output_dir/edit_condition_tokens/edit_condition_connector.pt"; do
  if [ ! -f "\$required" ]; then
    echo "Missing required base artifact: \$required" >&2
    exit 2
  fi
done
export SMU3M_OUTPUT_DIR="$base_output_dir"
export SMU3M_DIFFUSION_DIR="$task_output_dir/latent_diffusion"
export SMU3M_BASE_DIFFUSION_DIR="$base_output_dir/latent_diffusion"
export SMU3M_EVAL_LATENT_DIR="$task_output_dir/eval_latent"
export SMU3M_PRIOR_LOSS_WEIGHT="$prior_weight"
export SMU3M_TRAIN_DIFFUSION_CONNECTOR="$train_connector"
export SMU3M_EVAL_SAMPLE_STEPS="$sample_steps"
export SMU3M_DIFFUSION_EXTRA_EPOCHS="$extra_epochs"
export SMU3M_DIFFUSION_SEED="$(seed_plus "$seed" 2)"
export SMU3M_EVAL_SEED="$(seed_plus "$seed" 3)"
export SMU3M_RESUME="0"
export SMU3M_RUN_MATERIALIZED_BENCHMARK="0"
export SMU3M_EVAL_LIMIT="${SMU3M_FOLLOWUP_DIFFUSION_EVAL_LIMIT:-1000}"
export SMU3M_MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_FOLLOWUP_DIFFUSION_MAX_EVAL_PER_PROPERTY_COUNT:-0}"
bash "$PROJECT_DIR/scripts/run_unified_diffusion_refine.sh"
EOF
  chmod +x "$task_script"
  task_scripts+=("$task_script")
  printf 'bash %q # %s\n' "$task_script" "$safe_label" >> "$TASK_FILE"
  printf '%s,diffusion,%s,%s,%s\n' "$safe_label" "$task_output_dir" "$base_output_dir" "$config" >> "$MANIFEST_CSV"
}

if [[ "$FOLLOWUP_PLAN" == "all" || "$FOLLOWUP_PLAN" == "full" ]]; then
  IFS=';' read -r -a full_configs <<< "$FULL_CONFIGS"
  for config in "${full_configs[@]}"; do
    [[ -z "$config" ]] && continue
    write_full_task "$config"
  done
fi
if [[ "$FOLLOWUP_PLAN" == "all" || "$FOLLOWUP_PLAN" == "diffusion" ]]; then
  IFS=';' read -r -a diffusion_configs <<< "$DIFFUSION_CONFIGS"
  for config in "${diffusion_configs[@]}"; do
    [[ -z "$config" ]] && continue
    write_diffusion_task "$config"
  done
fi

if (( ${#task_scripts[@]} == 0 )); then
  echo "ERROR: no follow-up tasks were generated." >&2
  exit 2
fi
echo "  tasks=${#task_scripts[@]}"
echo "  task_file=$TASK_FILE"
echo "  manifest=$MANIFEST_CSV"

if [[ "$FOLLOWUP_DRY_RUN" == "1" ]]; then
  echo "Dry run requested; generated task file without executing tasks."
  sed -n '1,160p' "$TASK_FILE"
  exit 0
fi

run_with_parallel_fallback() {
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\0' "${task_scripts[@]}" | parallel -0 -j "$FOLLOWUP_CONCURRENCY" bash {}
  elif command -v xargs >/dev/null 2>&1; then
    printf '%s\0' "${task_scripts[@]}" | xargs -0 -n 1 -P "$FOLLOWUP_CONCURRENCY" bash
  else
    for task_script in "${task_scripts[@]}"; do
      bash "$task_script"
    done
  fi
}

case "$FOLLOWUP_LAUNCHER" in
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
    echo "ERROR: unsupported SMU3M_FOLLOWUP_LAUNCHER=$FOLLOWUP_LAUNCHER" >&2
    echo "Use glost, parallel, or serial." >&2
    exit 2
    ;;
esac

"${SMU3M_PYTHON_BIN:-python3}" - "$FOLLOWUP_OUTPUT_ROOT" "$MANIFEST_CSV" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])


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


manifest = {}
if manifest_path.exists():
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            manifest[row["label"]] = row

rows = []
for metrics_path in sorted(root.glob("*/eval_latent/metrics.json")):
    label = metrics_path.parents[1].name
    eval_metrics = json.loads(metrics_path.read_text())
    overall = eval_metrics.get("overall", {})
    prior_target = overall.get("prior_target_fingerprint_cosine", "")
    row = {
        "label": label,
        "kind": manifest.get(label, {}).get("kind", ""),
        "rows": eval_metrics.get("rows", overall.get("rows", "")),
        "target_fingerprint_cosine": overall.get("target_fingerprint_cosine", ""),
        "source_fingerprint_cosine": overall.get("source_fingerprint_cosine", ""),
        "source_target_fingerprint_cosine": overall.get("source_target_fingerprint_cosine", ""),
        "prior_target_fingerprint_cosine": prior_target,
        "target_fp_minus_prior": _diff_or_blank(overall.get("target_fingerprint_cosine", ""), prior_target),
        "target_property_mae": overall.get("target_property_mae", ""),
        "prior_target_property_mae": overall.get("prior_target_property_mae", ""),
        "property_mae_minus_prior": _diff_or_blank(
            overall.get("target_property_mae", ""),
            overall.get("prior_target_property_mae", ""),
        ),
        "delta_mae": overall.get("delta_mae", ""),
        "prior_delta_mae": overall.get("prior_delta_mae", ""),
        "delta_mae_minus_prior": _diff_or_blank(overall.get("delta_mae", ""), overall.get("prior_delta_mae", "")),
        "latent_mae": overall.get("latent_mae", ""),
        "generated_minus_prior_latent_mae": overall.get("generated_minus_prior_latent_mae", ""),
        "sample_steps": eval_metrics.get("sample_steps", ""),
        "seed": eval_metrics.get("seed", ""),
        "eval_metrics": str(metrics_path),
    }
    rows.append(row)

fieldnames = [
    "label",
    "kind",
    "rows",
    "target_fingerprint_cosine",
    "source_fingerprint_cosine",
    "source_target_fingerprint_cosine",
    "prior_target_fingerprint_cosine",
    "target_fp_minus_prior",
    "target_property_mae",
    "prior_target_property_mae",
    "property_mae_minus_prior",
    "delta_mae",
    "prior_delta_mae",
    "delta_mae_minus_prior",
    "latent_mae",
    "generated_minus_prior_latent_mae",
    "sample_steps",
    "seed",
    "eval_metrics",
]
csv_path = root / "followup_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "followup_summary.md"
with md_path.open("w", encoding="utf-8") as handle:
    handle.write("# Unified 3M Source-Aware Follow-up Summary\n\n")
    handle.write(f"- output root: `{root}`\n")
    handle.write(f"- tasks completed: `{len(rows)}`\n\n")
    handle.write("| label | kind | rows | target fp | source fp | prior target fp | gen-prior fp | property MAE | delta MAE | move from prior |\n")
    handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        cells = {key: _cell(value) for key, value in row.items()}
        handle.write(
            "| {label} | {kind} | {rows} | {target_fingerprint_cosine} | {source_fingerprint_cosine} | "
            "{prior_target_fingerprint_cosine} | {target_fp_minus_prior} | {target_property_mae} | "
            "{delta_mae} | {generated_minus_prior_latent_mae} |\n".format(**cells)
        )

print(f"followup_summary_csv={csv_path}")
print(f"followup_summary_md={md_path}")
PY

echo "Source-aware follow-up finished."
echo "  task_file=$TASK_FILE"
echo "  summary=$FOLLOWUP_OUTPUT_ROOT/followup_summary.md"
