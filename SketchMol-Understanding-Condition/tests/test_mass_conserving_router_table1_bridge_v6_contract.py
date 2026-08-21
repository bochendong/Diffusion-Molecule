import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXP / "mass_conserving_router_table1_bridge_v6.py"
PREREG = EXP / "mass_conserving_router_table1_bridge_v6_preregistration.json"
RUNNER = EXP / "run_mass_conserving_router_table1_bridge_v6.sh"
SUBMITTER = EXP / "submit_mass_conserving_router_table1_bridge_v6.sh"


def test_v6_preregistered_exact_n20_target_isolation():
    payload = json.loads(PREREG.read_text())
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["candidate_pool_before_selection"] == 20
    assert payload["generation_target_access"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["training"] is False
    assert payload["official_test_access"] is False


def test_v6_freeze_parser_cannot_accept_evaluation_targets():
    text = IMPLEMENTATION.read_text()
    freeze_block = text.split('freeze = stages.add_parser("freeze")', 1)[1].split(
        'evaluate = stages.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze_block
    assert "sealed_evaluation_targets" not in RUNNER.read_text().split("  freeze)", 1)[1].split("  evaluate)", 1)[0]


def test_v6_execution_and_science_gate_are_separate_jobs():
    text = SUBMITTER.read_text()
    assert "uca-v6-bridge-freeze" in text
    assert "uca-v6-bridge-eval" in text
    assert "uca-v6-bridge-scigate" in text
    assert 'dependency="afterok:$evaluate_job"' in text
    assert "scientific_stop_is_a_complete_gate_artifact_not_a_failed_job" in text


def test_v6_gate_does_not_raise_or_exit_nonzero_on_scientific_stop():
    text = IMPLEMENTATION.read_text()
    gate_block = text.split("def run_gate", 1)[1].split("def main", 1)[0]
    assert '"scientific_stop_exits_zero": True' in gate_block
    assert "return 0" in gate_block
