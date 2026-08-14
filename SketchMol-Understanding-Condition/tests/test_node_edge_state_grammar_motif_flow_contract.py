from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_flow"
MODEL_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_node_edge_state_grammar_motif_graph_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_node_edge_state_grammar_motif_graph_flow_pilot.sh"


def test_b17_builds_full_and_coarse_train_only_bond_support() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"bond_support"' in source
    assert '"coarse_bond_support"' in source
    assert '"bond_support_sha256"' in source
    assert "full_supported[1:].any()" in source
    assert "coarse_supported[1:].any()" in source
    assert "if supported is None:" in source
    assert "node_edge_state_grammar_motif_graph_flow_pilot_v17" in source


def test_b17_uses_fresh_disjoint_validation_and_paired_b16_evaluation() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "--validation-selection-seed" in source
    assert "--validation-exclusion-seed" in source
    assert '"historical_validation_source_overlap"' in source
    assert '"historical_validation_pair_overlap"' in source
    assert '"matched_b16_evaluation"' in source
    assert '"matched_b16_same_training_and_latent_samples"' in source
    assert 'checks["matched_b16_validity_delta"]' in source
    assert 'checks["matched_b16_3p_strict_any20_delta"]' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "property_oracle" not in sample_source
    assert "SanitizeMol" not in sample_source


def test_b17_runner_locks_fresh_split_exact_n20_and_small_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "--node-edge-state-grammar" in run_source
    assert 'SUCC_NODE_EDGE_GRAMMAR_EXCLUSION_SEED:-1742' in run_source
    assert 'SUCC_NODE_EDGE_GRAMMAR_VALIDATION_SEED:-2719' in run_source
    assert '--train-limit "${SUCC_NODE_EDGE_GRAMMAR_TRAIN_LIMIT:-1500}"' in run_source
    assert '--epochs "${SUCC_NODE_EDGE_GRAMMAR_EPOCHS:-8}"' in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_NODE_EDGE_GRAMMAR_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "SUCC_NODE_EDGE_GRAMMAR_TIME:-00:12:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
