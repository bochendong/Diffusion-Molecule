#!/usr/bin/env bash
# Collect existing SketchSMILES and real SketchMol+OCR artifacts into one table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_COMPARE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SKETCHMOL_COMPARE_OUT:-SketchMolCompare/outputs/comparisons/current}"

DEFAULT_SKETCHSMILES_RUNS=(
  SketchSMILES/outputs/runs/phase5a1_learned_smiles_decoder_seed7
  SketchSMILES/outputs/runs/phase5a2_tokenized_beam_decoder_seed7
  SketchSMILES/outputs/runs/phase5a4_reranked_transformer_decoder_seed7
  SketchSMILES/outputs/runs/phase5a6_randomized_smiles_decoder_seed7
  SketchSMILES/outputs/runs/phase5c_image_smiles_decoder_seed7
  SketchSMILES/outputs/runs/phase5d_image_fingerprint_decoder_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a4_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a4_beam32_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a4_sample64_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a4_large_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a6_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a6_aug1_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a6_aug2_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a6_aug4_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5a6_aug8_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5c_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5d_seed7
  SketchSMILES/outputs/runs/sketchmol_compare_phase5d_strong_seed7
  SketchMolTokenDiffusion/outputs/runs/sketchmol_compare_token_diffusion_seed7
  SketchMolTokenDiffusion/outputs/runs/sketchmol_compare_token_diffusion_selfies_seed7
  SketchMolJointDiffusion/outputs/runs/sketchmol_compare_joint_diffusion_seed7
  SketchMolJointDiffusion/outputs/runs/sketchmol_compare_joint_latent_clip_seed7
  SketchMolJointDiffusion/outputs/runs/sketchmol_compare_joint_selfies_latent_clip_seed7
  SketchMolJointDiffusion/outputs/runs/sketchmol_compare_joint_selfies_image_only_seed7
  SketchMolJointDiffusion/outputs/runs/sketchmol_compare_joint_selfies_light_aux_seed7
)
DEFAULT_SKETCHMOL_SUMMARIES=()
if [[ -f SketchMolBenchmark/outputs/current/benchmark_summary.csv ]]; then
  DEFAULT_SKETCHMOL_SUMMARIES+=(SketchMolBenchmark/outputs/current/benchmark_summary.csv)
fi

if [[ -n "${SKETCHMOL_COMPARE_SKETCHSMILES_RUNS:-}" ]]; then
  # shellcheck disable=SC2206
  SKETCHSMILES_RUNS=($SKETCHMOL_COMPARE_SKETCHSMILES_RUNS)
else
  SKETCHSMILES_RUNS=("${DEFAULT_SKETCHSMILES_RUNS[@]}")
fi

if [[ -n "${SKETCHMOL_COMPARE_SKETCHMOL_SUMMARIES:-}" ]]; then
  # shellcheck disable=SC2206
  SKETCHMOL_SUMMARIES=($SKETCHMOL_COMPARE_SKETCHMOL_SUMMARIES)
else
  SKETCHMOL_SUMMARIES=("${DEFAULT_SKETCHMOL_SUMMARIES[@]}")
fi

ARGS=(--output-dir "$OUTPUT_DIR")
INPUTS=0

for run_dir in "${SKETCHSMILES_RUNS[@]}"; do
  if [[ -f "$run_dir/metrics.json" ]]; then
    ARGS+=(--sketchsmiles-run "$run_dir")
    INPUTS=$((INPUTS + 1))
  else
    echo "warning: skipping missing SketchSMILES run: $run_dir" >&2
  fi
done

for summary_csv in "${SKETCHMOL_SUMMARIES[@]}"; do
  if [[ -f "$summary_csv" ]]; then
    ARGS+=(--sketchmol-summary "$summary_csv")
    INPUTS=$((INPUTS + 1))
  else
    echo "warning: skipping missing SketchMol summary: $summary_csv" >&2
  fi
done

if [[ "$INPUTS" -eq 0 ]]; then
  echo "ERROR: no existing comparison inputs found." >&2
  echo "Set SKETCHMOL_COMPARE_SKETCHSMILES_RUNS and/or SKETCHMOL_COMPARE_SKETCHMOL_SUMMARIES." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolCompare existing-run summary"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  inputs=$INPUTS"

"$PYTHON_BIN" -m sketchmol_compare.collect_metrics "${ARGS[@]}"

echo
echo "SketchMolCompare summary finished: $OUTPUT_DIR"
echo "  csv=$OUTPUT_DIR/comparison_rows.csv"
echo "  json=$OUTPUT_DIR/comparison_rows.json"
echo "  report=$OUTPUT_DIR/comparison_report.md"
