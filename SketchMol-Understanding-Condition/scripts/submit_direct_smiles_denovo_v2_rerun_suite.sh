#!/usr/bin/env bash
# Submit the direct-SMILES v2 follow-up variants as one rerun suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

SUITE_TAG="${SUCC_DIRECT_V2_SUITE_TAG:-v2_rerun_suite}"
SUITE_ROOT="${SUCC_DIRECT_V2_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_${SUITE_TAG}}"
SUITE_LABEL="${SUCC_DIRECT_V2_SUITE_LABEL:-SUCC Direct SMILES De Novo v2 Rerun Suite [$SUITE_TAG]}"
JOB_PREFIX="${SUCC_DIRECT_V2_SUITE_JOB_PREFIX:-succ-dsm-v2-rerun}"

BASE_2P7P_OUTPUT_DIR="${SUCC_DIRECT_DENOVO_V2_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
BASE_OOD_OUTPUT_DIR="${SUCC_DIRECT_OOD_V2_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition}"

mkdir -p "$SUITE_ROOT"

submit_variant() {
  local label="$1"
  local script_path="$2"
  shift 2

  echo
  echo "[$label] submitting via $(basename "$script_path")"

  local output
  if ! output="$(env "$@" bash "$script_path" 2>&1)"; then
    printf '%s\n' "$output" >&2
    echo "ERROR: failed to submit suite variant '$label'." >&2
    exit 1
  fi
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1
}

JOB_2P7P_N128="$(
  submit_variant \
    "2p7p_n128" \
    "$SCRIPT_DIR/submit_direct_smiles_denovo_2p7p_v2_n128_benchmark.sh" \
    "SUCC_DIRECT_DENOVO_V2_BASE_OUTPUT_DIR=$BASE_2P7P_OUTPUT_DIR" \
    "SUCC_DIRECT_DENOVO_SLURM_JOB_NAME=${JOB_PREFIX}-2p7p-n128" \
    "SUCC_DIRECT_DENOVO_OUTPUT_DIR=$SUITE_ROOT/2p7p_n128" \
    "SUCC_DIRECT_DENOVO_REPORT_TITLE=${SUITE_LABEL} :: 2p7p n=128"
)"

JOB_OOD_N128="$(
  submit_variant \
    "ood_n128" \
    "$SCRIPT_DIR/submit_direct_smiles_denovo_ood_v2_n128_benchmark.sh" \
    "SUCC_DIRECT_OOD_V2_BASE_OUTPUT_DIR=$BASE_OOD_OUTPUT_DIR" \
    "SUCC_DIRECT_OOD_SLURM_JOB_NAME=${JOB_PREFIX}-ood-n128" \
    "SUCC_DIRECT_OOD_OUTPUT_DIR=$SUITE_ROOT/ood_n128" \
    "SUCC_DIRECT_OOD_REPORT_TITLE=${SUITE_LABEL} :: OOD n=128"
)"

JOB_OOD_VALIDITY="$(
  submit_variant \
    "ood_validity_repair_n128" \
    "$SCRIPT_DIR/submit_direct_smiles_denovo_ood_v2_validity_repair_benchmark.sh" \
    "SUCC_DIRECT_OOD_V2_BASE_OUTPUT_DIR=$BASE_OOD_OUTPUT_DIR" \
    "SUCC_DIRECT_OOD_SLURM_JOB_NAME=${JOB_PREFIX}-ood-validity" \
    "SUCC_DIRECT_OOD_OUTPUT_DIR=$SUITE_ROOT/ood_validity_repair_n128" \
    "SUCC_DIRECT_OOD_REPORT_TITLE=${SUITE_LABEL} :: OOD validity-repair n=128"
)"

JOB_OOD_BALANCED="$(
  submit_variant \
    "ood_balanced" \
    "$SCRIPT_DIR/submit_direct_smiles_denovo_ood_v2_balanced_benchmark.sh" \
    "SUCC_DIRECT_OOD_SLURM_JOB_NAME=${JOB_PREFIX}-ood-balanced" \
    "SUCC_DIRECT_OOD_OUTPUT_DIR=$SUITE_ROOT/ood_balanced" \
    "SUCC_DIRECT_OOD_REPORT_TITLE=${SUITE_LABEL} :: OOD balanced retrain"
)"

MANIFEST_PATH="$SUITE_ROOT/suite_manifest.md"
cat >"$MANIFEST_PATH" <<EOF
# $SUITE_LABEL

- suite tag: \`$SUITE_TAG\`
- suite root: \`$SUITE_ROOT\`
- base 2p7p output: \`$BASE_2P7P_OUTPUT_DIR\`
- base OOD output: \`$BASE_OOD_OUTPUT_DIR\`

## Submitted jobs

| Variant | Job ID | Output dir |
| --- | --- | --- |
| 2p7p n=128 | \`${JOB_2P7P_N128:-missing}\` | \`$SUITE_ROOT/2p7p_n128\` |
| OOD n=128 | \`${JOB_OOD_N128:-missing}\` | \`$SUITE_ROOT/ood_n128\` |
| OOD validity-repair n=128 | \`${JOB_OOD_VALIDITY:-missing}\` | \`$SUITE_ROOT/ood_validity_repair_n128\` |
| OOD balanced retrain | \`${JOB_OOD_BALANCED:-missing}\` | \`$SUITE_ROOT/ood_balanced\` |

## Collect results

\`\`\`bash
python3 SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_v2_suite_results.py \\
  --suite-root "$SUITE_ROOT"
\`\`\`
EOF

echo
echo "Direct-SMILES v2 rerun suite submitted."
echo "  suite_root=$SUITE_ROOT"
echo "  manifest=$MANIFEST_PATH"
echo "  2p7p_n128_job=${JOB_2P7P_N128:-missing}"
echo "  ood_n128_job=${JOB_OOD_N128:-missing}"
echo "  ood_validity_repair_job=${JOB_OOD_VALIDITY:-missing}"
echo "  ood_balanced_job=${JOB_OOD_BALANCED:-missing}"
echo "  collect_results=python3 SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_v2_suite_results.py --suite-root \"$SUITE_ROOT\""
