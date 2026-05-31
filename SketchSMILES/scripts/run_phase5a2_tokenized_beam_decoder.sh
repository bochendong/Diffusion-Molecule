#!/usr/bin/env bash
# Run Phase 5A-2 tokenized SMILES decoder with beam search.

set -euo pipefail

export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5a2_tokenized_beam_decoder_seed${SKETCHSMILES_SEED:-7}}"
export SKETCHSMILES_TOKENIZATION="${SKETCHSMILES_TOKENIZATION:-smiles_token}"
export SKETCHSMILES_DECODING="${SKETCHSMILES_DECODING:-beam}"
export SKETCHSMILES_BEAM_SIZE="${SKETCHSMILES_BEAM_SIZE:-8}"
export SKETCHSMILES_LENGTH_PENALTY="${SKETCHSMILES_LENGTH_PENALTY:-0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_phase5a1_learned_smiles_decoder.sh"
