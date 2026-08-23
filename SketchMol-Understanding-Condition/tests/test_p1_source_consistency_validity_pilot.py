from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_DIR / "experiments" / "unified_smiles_generator" / "umtp_graph_action_policy.py"
RUN_PATH = (
    PROJECT_DIR
    / "experiments"
    / "p1_property_program_group_rl"
    / "run_p1_source_consistency_validity_pilot.sh"
)
SUBMIT_PATH = RUN_PATH.with_name("submit_p1_source_consistency_validity_pilot.sh")


def load_policy_module():
    spec = importlib.util.spec_from_file_location("p1_consistency_policy", POLICY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consistency_score_is_source_only_and_penalizes_large_displacement() -> None:
    policy = load_policy_module()
    compact = {"fingerprint": 0.8, "scaffold": 1.0, "edit_magnitude": 1.0}
    displaced = {"fingerprint": 0.4, "scaffold": 0.0, "edit_magnitude": 6.0}
    compact_score = policy.consistency_augmented_policy_score(
        -1.0,
        compact,
        fingerprint_weight=1.5,
        scaffold_weight=0.75,
        edit_magnitude_weight=0.1,
    )
    displaced_score = policy.consistency_augmented_policy_score(
        -1.0,
        displaced,
        fingerprint_weight=1.5,
        scaffold_weight=0.75,
        edit_magnitude_weight=0.1,
    )
    assert compact_score > displaced_score


def test_pilot_combines_consistency_ranking_with_grammar_valid_decoding() -> None:
    source = RUN_PATH.read_text(encoding="utf-8")
    assert "evaluate_edit_variant policy 0.0 0.0 0.0" in source
    assert "evaluate_edit_variant consistent" in source
    assert "evaluate_edit_variant strong_consistent" in source
    assert "evaluate_denovo_variant baseline 0 1.15 6" in source
    assert "evaluate_denovo_variant grammar_valid 1 1.05 0" in source
    assert "SUCC_UNIFIED_SMILES_GRAMMAR_CONSTRAINT" in source
    assert "--consistency-fingerprint-weight" in source
    assert "--consistency-scaffold-weight" in source
    assert "--consistency-edit-magnitude-weight" in source


def test_submitter_is_short_single_seed_gpu_gate() -> None:
    source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert 'TIME="${P1_CONSISTENCY_SLURM_TIME:-02:00:00}"' in source
    assert 'SEED="${P1_CONSISTENCY_SEED:-7}"' in source
    assert 'job-name="p1-consistency-s${SEED}"' in source
