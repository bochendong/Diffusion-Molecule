#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P251_SCRIPT_DIR:?P251_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P251_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="${P251_P23_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
P25="${P251_P25_ROOT:-$PROJECT/outputs/p25_p23_joint_group_rl/seed_2525}"
OUT="${P251_OUTPUT_ROOT:-$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125}"
D5="$P23/eval_denovo_table1_fill/data/denovo_5p.prompts.jsonl"
D67="$P23/eval_corrected_prompts/prompts/denovo_6p7p_p19.jsonl"
EDIT="$P23/eval_moledit_table1_500/data/table1_500.prompts.jsonl"
for path in "$P23/data/train.sft.jsonl" "$P23/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$P25/data/frozen_gate.jsonl" "$D5" "$D67" "$EDIT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25.1 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
mkdir -p "$OUT/data"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$SCRIPT_DIR/build_dual_gates.py" \
  --denovo-5p "$D5" --denovo-6p7p "$D67" --edit-table2 "$EDIT" \
  --exclude-gate "$P25/data/frozen_gate.jsonl" --output-dir "$OUT/data/gates" \
  --per-bucket 20 --seed 251250
sha256sum "$P23/data/train.sft.jsonl" "$P23/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$P25/data/frozen_gate.jsonl" "$D5" "$D67" "$EDIT" > "$OUT/data/input.sha256"
touch "$OUT/PREPARED"
