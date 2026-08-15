from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
AUDIT = EXPERIMENT_DIR / "audit_table1_assay_latent_action_support.py"
MERGE = EXPERIMENT_DIR / "merge_table1_assay_latent_action_support.py"
ORACLES = EXPERIMENT_DIR / "pinned_table1_assay_oracles.py"
MANIFEST = EXPERIMENT_DIR / "table1_assay_latent_action_support_v30_r1_preregistration.json"
RUN = EXPERIMENT_DIR / "run_table1_assay_latent_action_support.sh"
SUBMIT = EXPERIMENT_DIR / "submit_table1_assay_latent_action_support.sh"


def test_support_audit_is_exhaustive_target_free_and_not_a_selector() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "for token in vocabulary" in source
    assert '"exhaustive_train_only_vocabulary": True' in source
    assert '"generation_target_access": False' in source
    assert '"moledit_target_access": False' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"selected_prediction_output": False' in source
    assert "target_smiles" not in source
    assert 'selection.get("target_columns_used") != 0' in source
    assert "target_values" not in source
    assert "finite_descriptor_score" in source
    assert 'if prop in {"GSK3B", "DRD2"}' in source
    assert "load_pinned_oracles" in source
    assert '"oracle_failure_policy": "raise"' in source
    assert '"oracle_preflight_passed"' in source


def test_manifest_fixes_assay_tasks_and_decision_thresholds() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_repair_run"
    assert manifest["repair_of_protocol"] == "target_free_table1_assay_latent_action_support_v30"
    assert manifest["diagnostic_only"] is True
    assert manifest["exhaustive_train_only_vocabulary"] is True
    assert manifest["shards"] == 8
    assert manifest["similarity_prefilter"] == 0.15
    assert manifest["oracle_failure_policy"] == "raise"
    assert manifest["oracle_preflight_required"] is True
    assert manifest["oracle_batch_size"] == 256
    assert manifest["oracles"]["GSK3B"]["bytes"] == 30865235
    assert manifest["oracles"]["DRD2"]["bytes"] == 35417609
    assert len(manifest["oracles"]["GSK3B"]["sha256"]) == 64
    assert len(manifest["oracles"]["DRD2"]["sha256"]) == 64
    assert manifest["tasks"] == [
        "GSK3B:increase",
        "DRD2:decrease+MW:decrease+SA:decrease",
    ]
    assert manifest["gates"]["minimum_task_assay_support_rate_t0_15"] == 0.5
    assert manifest["gates"]["minimum_task_full_support_rate_t0_15"] == 0.25


def test_merge_chooses_joint_latent_or_grammar_expansion_without_lowering_gate() -> None:
    source = MERGE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "train_property_conditioned_joint_site_token_latent" in source
    assert "expand_to_connected_region_latent_action_grammar" in source
    assert "minimum_task_assay_support_rate_t0_15" in source
    assert '"molecular_candidate_ranking": False' in source


def test_runner_uses_eight_cpu_shards_and_dependent_merge() -> None:
    run_source = RUN.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in run_source
    assert "nearest_token_candidates.csv" in run_source
    assert "gsk3b_legacy_sklearn_compatible.pkl" in run_source
    assert "drd2_graph2graph_svc_py36.pkl" in run_source
    assert '--array="0-7%8"' in submit_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--cpus-per-task=1" in submit_source
    assert "--dependency=\"afterok:$array_job\"" in submit_source
    assert "--kill-on-invalid-dep=yes" in submit_source
    assert "--gres" not in submit_source


def test_pinned_oracles_validate_artifacts_and_fail_closed() -> None:
    source = ORACLES.read_text(encoding="utf-8")
    ast.parse(source)
    assert "sha256_file" in source
    assert "expected_size" in source
    assert "expected_digest" in source
    assert "known_active" in source
    assert "known_negative" in source
    assert "probe_range" in source
    assert "predict_proba" in source
    assert "except Exception" not in source
    assert "return 0.0" not in source
    assert "GetMorganFingerprintAsBitVect" in source
    assert "useCounts=True, useFeatures=True" in source
