from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_flow"
MODEL_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_atom_state_valence_grammar_motif_graph_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_atom_state_valence_grammar_motif_graph_flow_pilot.sh"


def test_b16_uses_one_train_supported_joint_atom_state() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def build_train_atom_state_grammar" in source
    assert "for example in (pair.source, pair.target)" in source
    assert "def decode_train_supported_atom_states" in source
    assert "log_softmax(dim=-1)" in source
    assert "joint_score.argmax(dim=-1)" in source
    assert "budget = torch.minimum(budget, state_capacity)" in source
    assert "atom_state_valence_grammar_motif_graph_flow_pilot_v16" in source


def test_b16_records_reproducible_train_only_support_without_repair() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert '"train_atom_state_support_sha256"' in source
    assert '"train_only_atom_state_support"' in source
    assert '"joint_atom_state_decode_before_graph_assembly"' in source
    assert '"atom_state_conditioned_valence_capacity"' in source
    assert '"training_dynamics_exactly_b15"' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "property_oracle" not in sample_source
    assert "SanitizeMol" not in sample_source


def test_b16_runner_is_matched_and_uses_a_small_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_ATOM_STATE_GRAMMAR_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_ATOM_STATE_GRAMMAR_VALIDATION_LIMIT:-20}"' in run_source
    assert '--epochs "${SUCC_ATOM_STATE_GRAMMAR_EPOCHS:-8}"' in run_source
    assert "--atom-state-valence-grammar" in run_source
    assert "--property-interaction-latents" not in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_ATOM_STATE_GRAMMAR_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "SUCC_ATOM_STATE_GRAMMAR_TIME:-00:10:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
