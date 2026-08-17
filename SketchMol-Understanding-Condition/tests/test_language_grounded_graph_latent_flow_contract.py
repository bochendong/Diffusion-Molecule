import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "language_grounded_graph_latent_flow.py"
MERGER = EXPERIMENT / "merge_language_grounded_graph_latent_flow.py"
PREREGISTRATION = EXPERIMENT / "language_grounded_graph_latent_flow_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_language_grounded_graph_latent_flow.sh"
SUBMITTER = EXPERIMENT / "submit_language_grounded_graph_latent_flow.sh"


def test_preregisters_end_to_end_state_dependent_transport_adapter():
    prereg = json.loads(PREREGISTRATION.read_text())
    assert prereg["protocol"] == "train_only_language_grounded_graph_latent_flow_v1"
    assert prereg["status"] == "preregistered_before_first_run"
    assert prereg["arms"] == ["property_memory", "common_llm_memory"]
    assert prereg["state_dependent_transport_adapter"] is True
    assert prereg["paired_flow_matching_supervision"] is True
    assert prereg["terminal_reachability_is_auxiliary_training_loss"] is True
    assert prereg["inference_classifier_gradient_guidance"] is False
    assert prereg["seed"] == 1991
    assert prereg["generation_seed_matches_valid_terminal_baseline"] is True
    assert prereg["implementation_sha256"] == hashlib.sha256(
        IMPLEMENTATION.read_bytes()
    ).hexdigest()
    assert "PLACEHOLDER" not in PREREGISTRATION.read_text()


def test_current_state_queries_memory_and_directly_changes_velocity():
    source = IMPLEMENTATION.read_text()
    tree = ast.parse(source)
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LanguageGroundedTransportAdapter"
    )
    adapter_source = ast.unparse(adapter)
    for name in ("latent", "source_pool", "flow_time", "memory", "memory_mask"):
        assert name in adapter_source
    assert "adapted_velocity = base_velocity + residual" in source
    assert "target_velocity = endpoint - noise" in source
    assert "F.mse_loss(adapted_velocity, target_velocity)" in source
    assert "torch.autograd.grad" not in source


def test_both_arms_have_equal_trainable_adapter_memory_dimension():
    prereg = json.loads(PREREGISTRATION.read_text())
    source = IMPLEMENTATION.read_text()
    assert prereg["memory_adapter_dim"] == 64
    assert "equalize_memory_dimension" in source
    assert "fixed_non_trainable_llm_memory_projection" in source
    assert "adapter_initialization_seed" in prereg


def test_generation_is_target_free_and_has_no_molecular_selector():
    prereg = json.loads(PREREGISTRATION.read_text())
    for key in (
        "common_llm_emits_text_or_actions",
        "inference_classifier_gradient_guidance",
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
    assert prereg["exact_raw_attempts_per_condition"] == 20
    assert prereg["fit_property_counts"] == [2]
    assert prereg["composition_diagnostic_property_counts"] == [3]


def test_merge_requires_absolute_and_llm_specific_downstream_gain():
    prereg = json.loads(PREREGISTRATION.read_text())
    source = MERGER.read_text()
    gates = prereg["signal_gates"]
    assert gates["relative_flow_mse_reduction"] > 0
    assert gates["strict_gain_vs_valid_terminal"] > 0
    assert gates["horizon_reduction_vs_valid_terminal"] > 0
    assert gates["llm_strict_gain_vs_property_memory"] > 0
    assert "strict_gain_vs_valid_terminal" in source
    assert "llm_strict_gain_vs_property_memory" in source
    assert "stop_language_grounded_graph_latent_flow_without_gate_changes" in source


def test_cluster_runs_only_two_single_seed_mig_arms_in_parallel():
    runner = RUNNER.read_text()
    submitter = SUBMITTER.read_text()
    assert "TRANSFORMERS_OFFLINE=1" in runner
    assert "--trajectory-dataset" in runner
    assert "--state-guidance-summary" in runner
    assert "submit_arm property_memory" in submitter
    assert "submit_arm common_llm_memory" in submitter
    assert "afterany:$property_job:$llm_job" in submitter
    assert "nvidia_h100_80gb_hbm3_1g.10gb" in submitter
    assert "nvidia_h100_80gb_hbm3_2g.20gb" in submitter
    assert "3g.40gb" not in submitter
