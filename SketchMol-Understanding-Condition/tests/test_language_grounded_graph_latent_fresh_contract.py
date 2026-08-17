import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "language_grounded_graph_latent_fresh_confirmation.py"
PREREGISTRATION = EXPERIMENT / "language_grounded_graph_latent_fresh_v2_preregistration.json"
RUNNER = EXPERIMENT / "run_language_grounded_graph_latent_fresh_confirmation.sh"
SUBMITTER = EXPERIMENT / "submit_language_grounded_graph_latent_fresh_confirmation.sh"


def test_preregisters_direction_only_prospective_confirmation():
    prereg = json.loads(PREREGISTRATION.read_text())
    assert prereg["protocol"] == "direction_only_language_grounded_graph_latent_fresh_v2"
    assert prereg["status"] == "preregistered_before_first_run"
    assert prereg["direction_only_generation_conditions"] is True
    assert prereg["numeric_target_property_access_during_generation"] is False
    assert prereg["fresh_target_process_isolation"] is True
    assert prereg["fresh_property_counts"] == [2, 3, 4, 5, 6, 7]
    assert sum(prereg["fresh_property_count_quotas"].values()) == 64
    assert prereg["exact_raw_attempts_per_condition"] == 20
    assert prereg["implementation_sha256"] == hashlib.sha256(
        IMPLEMENTATION.read_bytes()
    ).hexdigest()
    assert "PLACEHOLDER" not in PREREGISTRATION.read_text()


def test_generation_row_discards_numeric_target_and_delta_fields():
    source = IMPLEMENTATION.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "direction_only_row"
    )
    function_source = ast.unparse(function)
    assert "target_MW" not in function_source
    assert "target_QED" not in function_source
    assert "delta_MW" not in function_source
    assert "source_smiles" in function_source
    assert "instruction_tasks" in function_source
    assert "increase" in function_source and "decrease" in function_source


def test_freeze_arm_parser_cannot_accept_evaluation_targets():
    source = IMPLEMENTATION.read_text()
    tree = ast.parse(source)
    arm = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_arm"
    )
    arm_source = ast.unparse(arm)
    assert "evaluation_targets" not in arm_source
    assert "sealed_evaluation_targets" not in arm_source
    assert "generation_conditions" in arm_source
    assert "freeze_adapted_candidates" in arm_source


def test_evaluation_opens_targets_only_after_both_frozen_arms():
    source = IMPLEMENTATION.read_text()
    submitter = SUBMITTER.read_text()
    assert "post_freeze_evaluation_targets" in source
    assert 'afterok:$property_job:$llm_job' in submitter
    assert "SUCC_LANG_FRESH_STAGE=evaluate" in submitter
    assert "generation_target_path_accepted" in source


def test_no_ranking_retry_or_oracle_selection_and_exact_n20():
    prereg = json.loads(PREREGISTRATION.read_text())
    for key in (
        "molecular_candidate_ranking",
        "oracle_selection",
        "retry_or_resampling",
        "posthoc_molecule_repair",
        "official_test_access",
    ):
        assert prereg[key] is False
    assert prereg["exact_raw_attempts_per_condition"] == 20
    source = IMPLEMENTATION.read_text()
    assert "repeat_on_same_fresh_sources" in source
    assert "stop_direction_only_language_grounded_flow_without_fresh_retuning" in source


def test_cluster_uses_cpu_prepare_parallel_migs_and_cpu_evaluation():
    runner = RUNNER.read_text()
    submitter = SUBMITTER.read_text()
    assert "TRANSFORMERS_OFFLINE=1" in runner
    assert "SUCC_LANG_FRESH_STAGE=prepare" in submitter
    assert "nvidia_h100_80gb_hbm3_1g.10gb" in submitter
    assert "nvidia_h100_80gb_hbm3_2g.20gb" in submitter
    assert "3g.40gb" not in submitter
    assert "--mail-type=END,FAIL" in submitter
