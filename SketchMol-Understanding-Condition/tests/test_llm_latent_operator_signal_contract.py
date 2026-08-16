import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "llm_latent_operator_signal.py"
MERGER = EXPERIMENT / "merge_llm_latent_operator_signal.py"
PREREGISTRATION = EXPERIMENT / "llm_latent_operator_signal_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_llm_latent_operator_signal.sh"
MERGE_RUNNER = EXPERIMENT / "run_merge_llm_latent_operator_signal.sh"
SUBMITTER = EXPERIMENT / "submit_llm_latent_operator_signal.sh"


def test_llm_latent_operator_preregisters_a_direct_hidden_state_intervention():
    prereg = json.loads(PREREGISTRATION.read_text())
    assert prereg["protocol"] == "train_only_common_llm_latent_operator_signal_v1"
    assert prereg["status"] == "preregistered_before_first_run"
    assert prereg["arms"] == [
        "property_mlp",
        "base_frozen",
        "sft_frozen",
        "sft_lora",
    ]
    assert prereg["controller_fit_property_counts"] == [2]
    assert prereg["composition_ood_property_counts"] == [3]
    assert prereg["common_llm_emits_text_or_actions"] is False
    assert prereg["common_llm_hidden_state_controls_latent_dynamics"] is True
    assert prereg["frozen_b41_checkpoint"] is True
    assert prereg["b41_training"] is False
    assert prereg["exact_raw_attempts_per_condition"] == 20
    assert prereg["implementation_sha256"] == hashlib.sha256(
        IMPLEMENTATION.read_bytes()
    ).hexdigest()
    assert "PLACEHOLDER" not in PREREGISTRATION.read_text()


def test_generation_and_selection_contract_is_target_and_ranking_free():
    prereg = json.loads(PREREGISTRATION.read_text())
    for key in (
        "molecular_candidate_ranking",
        "oracle_selection",
        "retry_or_resampling",
        "posthoc_molecule_repair",
        "generation_target_access",
        "generation_property_oracle_access",
        "b26_heldout_access",
        "b33_fresh_source_access",
        "moledit_table1_benchmark_access",
        "official_test_access",
    ):
        assert prereg[key] is False

    tree = ast.parse(IMPLEMENTATION.read_text())
    prompt_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generation_safe_prompt"
    )
    prompt_source = ast.unparse(prompt_function)
    assert "pair.target_smiles" not in prompt_source
    assert "pair.target" not in prompt_source
    assert "source_smiles" in prompt_source
    assert "constraints" in prompt_source


def test_llm_hidden_residual_enters_both_frozen_latent_fields():
    source = IMPLEMENTATION.read_text()
    assert "modified_tokens = add_global_residual(tokens, residual)" in source
    assert "model.transport_velocity(" in source
    assert "model.route_condition(modified_tokens)" in source
    assert "model.denoiser(" in source
    assert "model.eval().requires_grad_(False)" in source
    assert "common_llm_emits_text_or_actions" in source


def test_cross_arm_gate_requires_llm_to_beat_non_sft_controllers():
    source = MERGER.read_text()
    prereg = json.loads(PREREGISTRATION.read_text())
    assert "sft_frozen" in source and "sft_lora" in source
    assert "property_mlp" in source and "base_frozen" in source
    assert "llm_gain_vs_non_sft_comparators" in source
    assert prereg["signal_gates"]["validity_gain_vs_valid_terminal"] > 0
    assert prereg["signal_gates"]["horizon_reduction_vs_valid_terminal"] > 0
    assert prereg["signal_gates"]["strict_delta_vs_valid_terminal"] <= 0


def test_cluster_entrypoints_keep_arms_parallel_and_merge_dependent():
    runner = RUNNER.read_text()
    merger = MERGE_RUNNER.read_text()
    submitter = SUBMITTER.read_text()
    assert "TRANSFORMERS_OFFLINE=1" in runner
    assert "--arm \"$ARM\"" in runner
    assert "--valid-terminal-summary" in runner
    assert "--sft-adapter-dir" in runner
    for arm in ("property_mlp", "base_frozen", "sft_frozen", "sft_lora"):
        assert f"submit_arm {arm}" in submitter
    assert "afterany:" in submitter
    assert "--kill-on-invalid-dep=yes" in submitter
    assert "merge_llm_latent_operator_signal.py" in merger
