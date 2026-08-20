from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "mass_conserving_property_set_router_v5.py"
GATE = EXPERIMENT / "gate_mass_conserving_property_set_router_v5.py"
PREREGISTRATION = (
    EXPERIMENT / "mass_conserving_property_set_router_v5_preregistration.json"
)
RUNNER = EXPERIMENT / "run_mass_conserving_property_set_router_v5.sh"
GATE_RUNNER = EXPERIMENT / "run_mass_conserving_property_set_router_v5_gate.sh"
SUBMITTER = EXPERIMENT / "submit_mass_conserving_property_set_router_v5.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_structural_set_router_and_fresh_probe() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "train_only_mass_conserving_property_set_router_v5"
    assert (
        manifest["mechanism"]
        == "mass_conserving_inclusion_energy_exact_topk_set_router"
    )
    assert manifest["categorical_cardinality_head"] is False
    assert manifest["support_threshold_search"] is False
    assert manifest["fresh_probe_excludes_v4_fit_and_probe"] is True
    assert manifest["fresh_probe_specs_per_k"] == 32
    assert manifest["full_fit_examples"] == manifest["no_composition_fit_examples"]
    assert manifest["arm_seed_stride"] == 0
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert manifest["gate_implementation_sha256"] == sha256(GATE)


def test_cardinality_is_inclusion_mass_not_a_categorical_head() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "class MassConservingPropertySetRouter" in source
    assert "del self.cardinality" in source
    assert "soft_cardinality = torch.sigmoid(support_logits).sum" in source
    assert "soft_cardinality.round().long()" in source
    assert "F.smooth_l1_loss" in source
    assert "support_separation_loss" in source
    assert "support_threshold_search" in source
    assert "generated_smiles" not in source
    assert "target_smiles" not in source


def test_fresh_probe_excludes_all_v4_composition_signatures() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "forbidden_property_sets" in source
    assert "fresh_v4_fit_overlap" in source
    assert "fresh_v4_probe_overlap" in source
    assert "fresh_v4_fit_property_set_overlap" in source
    assert "fresh_v4_probe_property_set_overlap" in source
    assert "sample_property_set_disjoint_specs" in source
    assert "fresh_primary_science_gate" in source
    assert "reused_v4_development_regression_only" in source
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["v4_composition_fit_seed"] == 2062
    assert manifest["v4_composition_probe_seed"] == 2063
    assert manifest["fresh_composition_probe_seed"] == 2073


def test_all_three_attribution_arms_and_equal_exposure_are_locked() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["arms"] == [
        "full",
        "no_lora",
        "no_token_slots",
        "no_composition",
    ]
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert 'arm == "no_lora"' in source
    assert 'arm == "no_composition"' in source
    assert "MassConservingPropertySetRouter" in source
    assert manifest["no_composition_unique_fit_examples"] == 136


def test_slurm_keeps_execution_and_science_outcomes_separate() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    gate_source = GATE.read_text(encoding="utf-8")
    assert "--array=0-3%2" in submitter
    assert "uca-set-router-v5-exec" in submitter
    assert "uca-set-router-v5-scigate" in submitter
    assert 'dependency="afterok:$execution_job_id"' in submitter
    assert RUNNER.name in submitter
    assert GATE_RUNNER.name in submitter
    assert "return 0 if passed else 3" in gate_source


def test_exact_n20_remains_locked_until_the_fresh_gate_passes() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    contract = manifest["generation_unlock_contract"]
    assert contract["exact_raw_attempts_per_condition"] == 20
    assert contract["candidate_pool_before_selection"] == 20
    assert contract["generation_target_access"] is False
    assert contract["oracle_selection"] is False
    assert contract["molecular_candidate_ranking"] is False
    assert contract["required_replays"] == [
        "denovo_2p7p",
        "moledit_table1",
        "mumo",
    ]


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("set_router_v5_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_summary(arm: str) -> dict[str, object]:
    full = arm == "full"
    no_lora = arm == "no_lora"
    no_slots = arm == "no_token_slots"
    no_composition = arm == "no_composition"
    return {
        "protocol": "train_only_mass_conserving_property_set_router_v5",
        "execution_status": "completed",
        "arm": arm,
        "contract": {
            "support_threshold_search": False,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
            "categorical_cardinality_head": False,
            "mass_conserving_cardinality": True,
            "fresh_probe_excludes_v4_fit_and_probe": True,
            "reused_graph_probe_is_regression_only": True,
        },
        "fresh_probe_lineage": {
            "fresh_v4_fit_overlap": 0,
            "fresh_v4_probe_overlap": 0,
            "fresh_v4_fit_property_set_overlap": 0,
            "fresh_v4_probe_property_set_overlap": 0,
            "fresh_probe_sha256": "locked-fresh-probe",
        },
        "multicardinality_probe": {
            "exact_support_rate": 0.96 if not no_composition else 0.10,
            "cardinality_exact_rate": 0.97,
            "active_sign_accuracy": 0.98,
            "exact_signed_support_rate": (
                0.92 if full else 0.40 if no_lora else 0.65 if no_slots else 0.05
            ),
        },
        "graph_probe_routing": {
            "matched": {
                "exact_support_rate": 0.98,
                "exact_signed_support_rate": 0.98,
            }
        },
        "graph_probe_tokens": {"language_mse_ratio_vs_intercept": 0.10},
        "graph_probe_flow": {
            "language_flow_ratio_vs_intercept": 0.30,
            "matched_flow_advantage": 0.02,
        },
    }


def _write_arms(root: Path, *, weaken_full: bool = False) -> None:
    for arm in ("full", "no_lora", "no_token_slots", "no_composition"):
        payload = _arm_summary(arm)
        if arm == "full" and weaken_full:
            payload["multicardinality_probe"]["exact_support_rate"] = 0.50
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_gate_unlocks_only_a_complete_positive_fresh_result(tmp_path: Path) -> None:
    module = _load_gate_module()
    arms_root = tmp_path / "arms"
    _write_arms(arms_root)
    gate_dir = tmp_path / "gate"
    result = module.main(
        [
            "--protocol-manifest",
            str(PREREGISTRATION),
            "--arms-root",
            str(arms_root),
            "--output-dir",
            str(gate_dir),
        ]
    )
    assert result == 0
    unlock = json.loads(
        (gate_dir / "generation_unlock.json").read_text(encoding="utf-8")
    )
    assert unlock["exact_raw_attempts_per_condition"] == 20
    assert unlock["candidate_pool_before_selection"] == 20


def test_negative_fresh_result_stops_only_the_gate_job(tmp_path: Path) -> None:
    module = _load_gate_module()
    arms_root = tmp_path / "arms"
    _write_arms(arms_root, weaken_full=True)
    gate_dir = tmp_path / "gate"
    result = module.main(
        [
            "--protocol-manifest",
            str(PREREGISTRATION),
            "--arms-root",
            str(arms_root),
            "--output-dir",
            str(gate_dir),
        ]
    )
    assert result == 3
    gate = json.loads((gate_dir / "gate_summary.json").read_text(encoding="utf-8"))
    assert gate["science_gate"]["passed"] is False
    assert gate["decision"] == "stop_before_molecule_generation"
    assert not (gate_dir / "generation_unlock.json").exists()
