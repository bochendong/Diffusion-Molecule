from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "semantic_intervention_fresh_v9.py"
PREREGISTRATION = EXPERIMENT / "semantic_intervention_fresh_v9_preregistration.json"
RUNNER = EXPERIMENT / "run_semantic_intervention_fresh_v9.sh"
SUBMITTER = EXPERIMENT / "submit_semantic_intervention_fresh_v9.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v9_preregistration_locks_candidate_level_semantic_intervention() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["protocol"] == "prospective_semantic_intervention_fresh_v9"
    assert payload["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert payload["fresh_condition_count"] == 20
    assert payload["fresh_property_count_quotas"] == {"3": 20}
    assert payload["protocol_amendment"]["failed_prepare_job"] == 20229890
    assert payload["protocol_amendment"]["candidate_generation_started"] is False
    assert payload["protocol_amendment"]["property_oracle_accessed"] is False
    assert payload["protocol_amendment"]["science_gates_changed"] is False
    assert payload["replicate_seeds"] == [2131, 2132, 2133]
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["candidate_pool_before_selection"] == 20
    assert payload["primary_metric_family"] == "candidate_level_distributional_success"
    assert payload["any20_role"] == "secondary_capability_metric"
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["official_test_access"] is False
    assert payload["arms"] == [
        "numeric_canonical",
        "language_full",
        "language_reversed",
        "language_no_lora",
        "language_no_token_slots",
        "language_no_composition",
    ]


def test_v9_generation_and_science_decision_are_physically_separate() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze_parser = source.split('freeze = stages.add_parser("freeze")', 1)[1].split(
        'evaluate = stages.add_parser("evaluate")', 1
    )[0]
    assert "evaluation-targets" not in freeze_parser
    assert "target-smiles" not in freeze_parser
    assert "property-oracle" not in freeze_parser
    assert 'if args.stage == "freeze"' in source
    assert 'if args.stage == "evaluate"' in source
    assert 'if args.stage == "gate"' in source
    assert '"scientific_stop_exits_zero": True' in source
    assert "raise SystemExit(1)" not in source


def test_v9_runner_and_submitter_preserve_pairing_and_exact_n20() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert 'SUCC_V9_STAGE=prepare' in submitter
    assert 'SUCC_V9_STAGE=freeze' in submitter
    assert 'SUCC_V9_STAGE=evaluate' in submitter
    assert 'SUCC_V9_STAGE=gate' in submitter
    assert "--array=0-2%2" in submitter
    assert 'dependency="afterok:$prepare_job"' in submitter
    assert 'dependency="afterok:$freeze_job"' in submitter
    assert 'dependency="afterok:$evaluate_job"' in submitter
    assert "3_replicates_x_6_arms_x_20_fresh_3p_conditions_x_exact_20" in submitter
    assert "candidate_property_success_candidate_strict_success_property_fraction" in submitter
    assert runner.count("--known-source") == 7
    assert "mass_conserving_router_fresh_horizon_v7" in runner


def test_v9_gate_uses_paired_candidate_metrics_not_saturated_any20() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert '"candidate_property_success"' in source
    assert '"candidate_strict_success"' in source
    assert '"mean_property_fraction"' in source
    assert "bootstrap_interval" in source
    assert "numeric_noninferior_property" in source
    assert 'CONTROLS = (' in source
    assert '"language_reversed"' in source
    assert '"language_no_lora"' in source
    assert '"language_no_token_slots"' in source
    assert '"language_no_composition"' in source
    assert 'checks[f"{short}_property_ci"]' in source
    assert 'checks[f"{short}_strict_ci"]' in source
    assert "any@20 is reported but is not a causal gate" in source
