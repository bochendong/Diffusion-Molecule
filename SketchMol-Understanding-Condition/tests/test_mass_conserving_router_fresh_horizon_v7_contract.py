import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXP / "mass_conserving_router_fresh_horizon_v7.py"
HORIZON = EXP / "horizon_closed_graph_jump.py"
PREREG = EXP / "mass_conserving_router_fresh_horizon_v7_preregistration.json"
RUNNER = EXP / "run_mass_conserving_router_fresh_horizon_v7.sh"
SUBMITTER = EXP / "submit_mass_conserving_router_fresh_horizon_v7.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v7_preregistered_fresh_exact_n20_contract():
    payload = json.loads(PREREG.read_text())
    assert payload["fresh_condition_count"] == 48
    assert payload["fresh_property_count_quotas"] == {"2": 24, "3": 24}
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["candidate_pool_before_selection"] == 20
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["repeat_v6_conditions"] is False


def test_v7_implementation_and_horizon_are_digest_locked():
    payload = json.loads(PREREG.read_text())
    assert payload["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert payload["horizon_closure_sha256"] == sha256(HORIZON)


def test_v7_e1_manifest_digest_matches_the_frozen_v6_lineage():
    payload = json.loads(PREREG.read_text())
    e1 = EXP / "e1_nl_condition_head_preregistration.json"
    assert payload["locked_inputs"]["e1_manifest_sha256"] == sha256(e1)


def test_v7_freeze_process_cannot_accept_targets_or_oracles():
    text = IMPLEMENTATION.read_text()
    freeze_block = text.split('freeze = stages.add_parser("freeze")', 1)[1].split(
        'evaluate = stages.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze_block
    assert "property-oracle" not in freeze_block
    runner_freeze = RUNNER.read_text().split("  freeze)", 1)[1].split("  evaluate)", 1)[0]
    assert "sealed_evaluation_targets" not in runner_freeze


def test_v7_horizon_closure_is_an_in_process_terminal_transition():
    text = HORIZON.read_text()
    assert "checkpoint_node" in text
    assert "checkpoint_edge" in text
    assert "final_transition = jump_index + 1 == max_jumps" in text
    assert "legal[close, 0] = True" in text
    assert "Horizon-closed graph jump emitted a non-materializable state" in text
    assert "property_success" not in text
    assert "target_smiles" not in text


def test_v7_execution_and_science_gate_are_separate_jobs():
    text = SUBMITTER.read_text()
    assert "uca-v7-fresh-prepare" in text
    assert "uca-v7-fresh-freeze" in text
    assert "uca-v7-fresh-eval" in text
    assert "uca-v7-fresh-scigate" in text
    assert 'dependency="afterok:$evaluate_job"' in text
    assert "scientific_stop_is_a_complete_gate_artifact_not_a_failed_job" in text


def test_v7_science_stop_exits_zero_and_forbids_same_fresh_retuning():
    text = IMPLEMENTATION.read_text()
    gate_block = text.split("def run_gate", 1)[1].split("def main", 1)[0]
    assert '"repeat_on_same_fresh_sources_for_retuning": False' in gate_block
    assert '"lower_gate_after_result": False' in gate_block
    assert "return 0" in gate_block
