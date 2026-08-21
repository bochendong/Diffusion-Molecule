import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXP / "paired_signed_set_calibration_v8.py"
PREREG = EXP / "paired_signed_set_calibration_v8_preregistration.json"
RUNNER = EXP / "run_paired_signed_set_calibration_v8.sh"
SUBMITTER = EXP / "submit_paired_signed_set_calibration_v8.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v8_preregisters_paired_heavy_development_ablation():
    payload = json.loads(PREREG.read_text())
    assert payload["protocol"] == "train_only_paired_signed_set_calibration_v8"
    assert payload["task_role"] == (
        "train_only_development_calibration_not_fresh_confirmation"
    )
    assert payload["replicate_seeds"] == [2111, 2112, 2113]
    assert payload["paired_common_random_numbers"] is True
    assert payload["conditions"] == 48
    assert payload["token_count"] == 18
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["candidate_pool_before_selection"] == 20
    assert len(payload["arms"]) == 5
    assert 3 * 5 * 48 * 20 == 14400


def test_v8_preserves_target_isolation_and_never_reuses_v7_fresh():
    payload = json.loads(PREREG.read_text())
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["v7_fresh_source_access"] is False
    source = IMPLEMENTATION.read_text()
    freeze_parser = source.split('freeze = stages.add_parser("freeze")', 1)[1].split(
        'evaluate = stages.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze_parser
    runner_freeze = RUNNER.read_text().split("  freeze)", 1)[1].split(
        "  evaluate)", 1
    )[0]
    assert "sealed_evaluation_targets" not in runner_freeze


def test_v8_signed_vertex_and_sqrt_projection_are_locked():
    payload = json.loads(PREREG.read_text())
    assert payload["calibration_arms"] == [
        "language_signed_vertex",
        "language_sqrt_sharpened",
    ]
    source = IMPLEMENTATION.read_text()
    assert 'coefficients >= 0' in source
    assert 'coefficients.abs().clamp_min(1e-8).sqrt()' in source
    assert 'return signed * magnitude' in source
    assert 'return coefficients * support_float' in source


def test_v8_uses_identical_particle_seed_across_arms():
    source = IMPLEMENTATION.read_text()
    assert "paired_particle_seed = seed * 100000 + index" in source
    sample_block = source.split("for arm in ARMS:", 1)[1].split(
        "root_summary.write_text", 1
    )[0]
    assert "ARMS.index" not in sample_block
    assert "paired_particle_seed," in sample_block


def test_v8_code_and_horizon_are_digest_locked():
    payload = json.loads(PREREG.read_text())
    assert payload["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert payload["horizon_closure_sha256"] == sha256(
        EXP / "horizon_closed_graph_jump.py"
    )
    assert payload["v6_implementation_sha256"] == sha256(
        EXP / "mass_conserving_router_table1_bridge_v6.py"
    )


def test_v8_slurm_is_bounded_parallel_and_science_stop_is_complete():
    submitter = SUBMITTER.read_text()
    source = IMPLEMENTATION.read_text()
    assert "--array=0-2%2" in submitter
    assert "--array=0-2%3" in submitter
    assert "--time=01:30:00" in submitter
    assert 'dependency="afterok:$freeze_job"' in submitter
    assert 'dependency="afterok:$evaluate_job"' in submitter
    assert "scientific_stop_exits_zero" in submitter
    gate_block = source.split("def run_gate", 1)[1].split("def main", 1)[0]
    assert '"scientific_stop_exits_zero": True' in gate_block
    assert "return 0" in gate_block
