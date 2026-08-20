from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "fresh_graph_jump_language_confirmation.py"
PREREGISTRATION = EXPERIMENT / "fresh_graph_jump_language_confirmation_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_fresh_graph_jump_language_confirmation.sh"
SUBMITTER = EXPERIMENT / "submit_fresh_graph_jump_language_confirmation.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_exact_n20_and_all_causal_arms() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "fresh_graph_jump_language_causal_confirmation_v1"
    assert manifest["exact_raw_attempts_per_condition"] == 20
    assert manifest["fresh_condition_count"] == 60
    assert sum(manifest["fresh_task_quotas"].values()) == 60
    assert manifest["model_training"] is False
    assert manifest["generation_target_access"] is False
    assert manifest["official_test_target_access"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["retry_or_resampling"] is False
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert manifest["arms"] == [
        "numeric_b41",
        "numeric_canonical",
        "numeric_d3_grpo",
        "language_template",
        "language_paraphrase",
        "language_keyword",
        "language_scrambled",
        "language_reversed",
    ]


def test_generation_process_cannot_accept_the_sealed_target_path() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze_parser = source.split('freeze = subparsers.add_parser("freeze")', 1)[1].split(
        'evaluate = subparsers.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze_parser
    assert 'freeze.add_argument("--generation-conditions"' in freeze_parser
    runner = RUNNER.read_text(encoding="utf-8")
    freeze_case = runner.split('freeze)', 1)[1].split('evaluate)', 1)[0]
    assert "sealed_evaluation_targets" not in freeze_case
    assert "generation_conditions.json" in freeze_case


def test_evaluation_waits_for_both_frozen_groups() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert 'SUCC_FRESH_CONFIRM_ARM_GROUP=graph' in source
    assert 'SUCC_FRESH_CONFIRM_ARM_GROUP=language' in source
    assert '--dependency="afterok:$graph_job:$language_job"' in source
    assert '--gres="gpu:$GPU_REQUEST"' in source


def test_decision_keeps_language_and_graph_evidence_separate() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "advance_canonical_graph_jump_with_language" in source
    assert "advance_canonical_graph_jump_without_language" in source
    assert "stop_fresh_confirmation_without_retuning" in source
    assert "language_semantic_margin" in source
    assert "language_reversed_drop" in source
    assert "canonical_gain_vs_b41" in source
