from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "structured_sparse_property_router_v4.py"
GATE = EXPERIMENT / "gate_structured_sparse_property_router_v4.py"
PREREGISTRATION = EXPERIMENT / "structured_sparse_property_router_v4_preregistration.json"
RUNNER = EXPERIMENT / "run_structured_sparse_property_router_v4.sh"
GATE_RUNNER = EXPERIMENT / "run_structured_sparse_property_router_v4_gate.sh"
SUBMITTER = EXPERIMENT / "submit_structured_sparse_property_router_v4.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_structured_router_and_ablation_arms() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "train_only_structured_sparse_property_router_v4"
    assert manifest["mechanism"] == "explicit_cardinality_exact_topk_property_router"
    assert manifest["support_threshold_search"] is False
    assert manifest["max_instruction_cardinality"] == 7
    assert manifest["arm_seed_stride"] == 0
    assert manifest["full_fit_examples"] == manifest["no_composition_fit_examples"]
    assert manifest["no_composition_unique_fit_examples"] == 136
    assert manifest["arms"] == [
        "full",
        "no_lora",
        "no_token_slots",
        "no_composition",
    ]
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)
    assert manifest["gate_implementation_sha256"] == sha256(GATE)


def test_router_is_cardinality_constrained_without_threshold_search() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "class StructuredSparsePropertyRouter" in source
    assert "def exact_topk_support" in source
    assert "cardinality_logits.argmax" in source
    assert "torch.topk" in source
    assert "support_threshold_search" in source
    assert "return 0\n" in source
    assert "generated_smiles" not in source
    assert "target_smiles" not in source
    assert "oracle_selection" in source


def test_ablation_changes_only_declared_mechanisms() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert 'use_lora = args.arm != "no_lora"' in source
    assert 'use_token_slots=args.arm != "no_token_slots"' in source
    assert 'if arm == "no_composition"' in source
    assert "Preserve the full arm's number of optimizer updates" in source
    assert 'arm_seed_stride' in source
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["single_seed"] is True
    assert manifest["language_fit_excludes_graph_probe_property_pairs"] is True


def test_slurm_separates_execution_success_from_science_gate() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    gate_runner = GATE_RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    gate_source = GATE.read_text(encoding="utf-8")
    assert "--array=0-3%2" in submitter
    assert "uca-sparse-v4-exec" in submitter
    assert "uca-sparse-v4-scigate" in submitter
    assert 'dependency="afterok:$execution_job_id"' in submitter
    assert "run_structured_sparse_property_router_v4.sh" in submitter
    assert "run_structured_sparse_property_router_v4_gate.sh" in submitter
    assert "structured_sparse_property_router_v4.py" in runner
    assert "gate_structured_sparse_property_router_v4.py" in gate_runner
    assert "return 0 if passed else 3" in gate_source


def test_generation_is_locked_to_exact_n20_after_gate() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    contract = manifest["generation_unlock_contract"]
    assert contract["exact_raw_attempts_per_condition"] == 20
    assert contract["candidate_pool_before_selection"] == 20
    assert contract["generation_target_access"] is False
    assert contract["oracle_selection"] is False
    assert contract["molecular_candidate_ranking"] is False
    gate_source = GATE.read_text(encoding="utf-8")
    assert "unlock_target_isolated_exact_n20_generation" in gate_source
    assert "unlocked_not_executed" in gate_source


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("structured_router_v4_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_summary(arm: str) -> dict[str, object]:
    return {
        "protocol": "train_only_structured_sparse_property_router_v4",
        "execution_status": "completed",
        "arm": arm,
        "contract": {
            "support_threshold_search": False,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
            "explicit_cardinality": True,
            "exact_topk_support": True,
        },
        "graph_probe_routing": {
            "matched": {
                "exact_support_rate": 0.98 if arm != "no_token_slots" else 0.80,
                "support_precision": 0.98,
                "support_recall": 0.98,
            }
        },
        "multicardinality_probe": {
            "exact_support_rate": 0.90 if arm != "no_composition" else 0.70,
            "cardinality_exact_rate": 0.95,
            "active_sign_accuracy": 0.95,
        },
        "graph_probe_tokens": {
            "language_mse_ratio_vs_intercept": 0.10 if arm != "no_lora" else 0.30
        },
        "graph_probe_flow": {
            "language_flow_ratio_vs_intercept": 0.30,
            "matched_flow_advantage": 0.02,
            "oracle_canonical_flow_relative_error": 0.01,
        },
    }


def test_science_gate_is_a_distinct_post_execution_decision(tmp_path: Path) -> None:
    module = _load_gate_module()
    arms_root = tmp_path / "arms"
    for arm in ("full", "no_lora", "no_token_slots", "no_composition"):
        arm_dir = arms_root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "summary.json").write_text(
            json.dumps(_arm_summary(arm)), encoding="utf-8"
        )
    gate_dir = tmp_path / "gate"
    assert (
        module.main(
            [
                "--protocol-manifest",
                str(PREREGISTRATION),
                "--arms-root",
                str(arms_root),
                "--output-dir",
                str(gate_dir),
            ]
        )
        == 0
    )
    gate = json.loads((gate_dir / "gate_summary.json").read_text(encoding="utf-8"))
    unlock = json.loads(
        (gate_dir / "generation_unlock.json").read_text(encoding="utf-8")
    )
    assert gate["science_gate"]["passed"] is True
    assert unlock["exact_raw_attempts_per_condition"] == 20
    assert unlock["candidate_pool_before_selection"] == 20
    assert unlock["molecular_candidate_ranking"] is False


def test_negative_science_result_stops_only_the_gate_job(tmp_path: Path) -> None:
    module = _load_gate_module()
    arms_root = tmp_path / "arms"
    for arm in ("full", "no_lora", "no_token_slots", "no_composition"):
        payload = _arm_summary(arm)
        if arm == "full":
            payload["graph_probe_routing"]["matched"]["exact_support_rate"] = 0.50
        arm_dir = arms_root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    gate_dir = tmp_path / "gate"
    assert (
        module.main(
            [
                "--protocol-manifest",
                str(PREREGISTRATION),
                "--arms-root",
                str(arms_root),
                "--output-dir",
                str(gate_dir),
            ]
        )
        == 3
    )
    gate = json.loads((gate_dir / "gate_summary.json").read_text(encoding="utf-8"))
    assert gate["science_gate"]["passed"] is False
    assert gate["decision"] == "stop_before_molecule_generation"
    assert not (gate_dir / "generation_unlock.json").exists()
