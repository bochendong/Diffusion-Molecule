import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "common_llm_state_viability_guidance.py"
MERGER = EXPERIMENT / "merge_common_llm_state_viability_guidance.py"
PREREGISTRATION = EXPERIMENT / "common_llm_state_viability_guidance_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_common_llm_state_viability_guidance.sh"
SUBMITTER = EXPERIMENT / "submit_common_llm_state_viability_guidance.sh"


def test_preregisters_state_dependent_common_llm_latent_guidance():
    prereg = json.loads(PREREGISTRATION.read_text())
    assert prereg["protocol"] == "train_only_common_llm_state_viability_guidance_v1"
    assert prereg["status"] == "preregistered_before_first_run"
    assert prereg["arms"] == ["property_memory", "common_llm_memory"]
    assert prereg["fit_property_counts"] == [2]
    assert prereg["composition_diagnostic_property_counts"] == [3]
    assert prereg["current_latent_queries_constraint_memory_each_flow_step"] is True
    assert prereg["terminal_reachability_gradient_guides_latent_vector_field"] is True
    assert prereg["common_llm_emits_text_or_actions"] is False
    assert prereg["exact_raw_attempts_per_condition"] == 20
    assert prereg["implementation_sha256"] == hashlib.sha256(
        IMPLEMENTATION.read_bytes()
    ).hexdigest()
    assert "PLACEHOLDER" not in PREREGISTRATION.read_text()


def test_common_llm_memory_is_queried_by_current_state_not_emitted_as_action():
    source = IMPLEMENTATION.read_text()
    tree = ast.parse(source)
    critic = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StateViabilityCritic"
    )
    critic_source = ast.unparse(critic)
    assert "latent" in critic_source
    assert "source_pool" in critic_source
    assert "flow_time" in critic_source
    assert "memory" in critic_source
    assert "attention" not in critic_source.lower() or "softmax" in critic_source
    assert "torch.autograd.grad" in source
    assert "F.logsigmoid(reachability)" in source
    assert "model.transport_velocity(" in source
    assert "operator.generation_safe_prompt" not in source


def test_generation_contract_has_no_molecular_ranking_or_oracle_selection():
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
    assert prereg["development_is_reused_method_development_split"] is True
    assert prereg["development_is_formal_fresh_ood"] is False


def test_fit_trajectory_labels_and_common_llm_control_are_separated():
    source = IMPLEMENTATION.read_text()
    assert "fit_only_trajectory_collection" in source
    assert "trajectory_pairs" in source
    assert "compressed_llm_token_memories" in source
    assert "property_memory" in source
    assert "common_llm_memory" in source
    assert "validation_terminal_class_counts" in source
    assert "state_auc" in source


def test_cluster_dag_prepares_once_then_runs_critics_in_parallel():
    runner = RUNNER.read_text()
    submitter = SUBMITTER.read_text()
    assert "TRANSFORMERS_OFFLINE=1" in runner
    assert "--stage \"$STAGE\"" in runner
    assert "--trajectory-dataset" in runner
    assert "SUCC_STATE_GUIDE_STAGE=prepare" in submitter
    assert "SUCC_STATE_GUIDE_STAGE=property_memory" in submitter
    assert "SUCC_STATE_GUIDE_STAGE=common_llm_memory" in submitter
    assert submitter.count("afterok:$prepare_job") == 2
    assert "afterany:$property_job:$llm_job" in submitter
    assert "nvidia_h100_80gb_hbm3_1g.10gb" in submitter
    assert "nvidia_h100_80gb_hbm3_2g.20gb" in submitter


def test_merge_gate_requires_absolute_and_llm_specific_signal():
    prereg = json.loads(PREREGISTRATION.read_text())
    source = MERGER.read_text()
    assert prereg["critic_gates"]["state_auc"] >= 0.65
    assert prereg["signal_gates"]["validity_gain_vs_valid_terminal"] > 0
    assert prereg["signal_gates"]["horizon_reduction_vs_valid_terminal"] > 0
    assert prereg["signal_gates"]["llm_gain_vs_property_memory"] > 0
    assert "llm_validity_gain_vs_property_memory" in source
    assert "llm_horizon_reduction_vs_property_memory" in source
    assert "stop_state_viability_guidance_without_gate_changes" in source
