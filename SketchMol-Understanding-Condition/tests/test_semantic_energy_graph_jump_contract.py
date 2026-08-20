from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "semantic_energy_graph_jump_v1.py"
PREREGISTRATION = EXPERIMENT / "semantic_energy_graph_jump_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_semantic_energy_graph_jump_v1.sh"
SUBMITTER = EXPERIMENT / "submit_semantic_energy_graph_jump_v1.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_language_energy_and_exact_n20() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "train_only_semantic_energy_graph_jump_v1"
    assert manifest["common_llm_prompt_contains_source"] is False
    assert manifest["explicit_semantic_hard_negatives"] == [
        "reversed",
        "scrambled",
        "property_swap",
    ]
    assert manifest["frozen_canonical_graph_jump"] is True
    assert manifest["canonical_training"] is False
    assert manifest["exact_raw_attempts_per_condition"] == 20
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["generation_target_access"] is False
    assert manifest["fit_probe_split"] == "canonical_source_group_exact_condition_budget"
    assert manifest["source_group_split_seed"] == 2041
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)


def test_prepare_replaces_leaky_critic_split_with_source_group_split() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    helper = source.split("def source_group_split_indices", 1)[1].split(
        "def specs_for_row", 1
    )[0]
    prepare = source.split("def run_prepare", 1)[1].split(
        "def constraint_only_chat", 1
    )[0]
    assert "canonical_pair_source" in helper
    assert "probe_conditions" in helper
    assert "source_group_split_indices(" in prepare
    prepare_lines = {line.strip() for line in prepare.splitlines()}
    assert 'train_indices = list(bundle["train_indices"])' not in prepare_lines
    assert 'validation_indices = list(bundle["validation_indices"])' not in prepare_lines
    assert "legacy_fit_probe_source_overlap" in prepare


def test_constraint_prompt_has_no_source_identity_shortcut() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    prompt = source.split("def constraint_only_chat", 1)[1].split(
        "@torch.no_grad()", 1
    )[0]
    assert "source_smiles" not in prompt
    assert "constraint_only_chat(text)" in source
    assert "semantic_margin_loss" in source
    assert "token_margin_loss" in source
    assert "reversed_flow_loss" in source


def test_freeze_process_cannot_accept_sealed_targets() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze = source.split('freeze = stages.add_parser("freeze")', 1)[1].split(
        'evaluate = stages.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze
    runner = RUNNER.read_text(encoding="utf-8")
    freeze_case = runner.split("freeze)", 1)[1].split("evaluate)", 1)[0]
    assert "sealed_evaluation_targets" not in freeze_case
    assert "generation_conditions.json" in freeze_case


def test_representation_gate_precedes_generation() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert 'return 0 if passed else 3' in source
    assert 'if not bool(dict(train_summary["representation_gate"])["passed"])' in source
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "SUCC_SEMANTIC_ENERGY_STAGE=train" in submitter
    assert "SUCC_SEMANTIC_ENERGY_STAGE=freeze" in submitter
    assert "&&" in submitter


def test_submitter_uses_one_bounded_20gb_mig_signal() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in source
    assert "--time=02:00:00" in source
    assert '--dependency="afterok:$prepare_job"' in source
    assert '--dependency="afterok:$train_freeze_job"' in source
    assert "nvidia_h100_80gb_hbm3:1" not in source
