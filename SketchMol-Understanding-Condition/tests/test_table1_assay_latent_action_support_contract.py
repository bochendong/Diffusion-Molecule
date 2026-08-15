from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
AUDIT = EXPERIMENT_DIR / "audit_table1_assay_latent_action_support.py"
MERGE = EXPERIMENT_DIR / "merge_table1_assay_latent_action_support.py"
MANIFEST = EXPERIMENT_DIR / "table1_assay_latent_action_support_v30_preregistration.json"
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


def test_manifest_fixes_assay_tasks_and_decision_thresholds() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["diagnostic_only"] is True
    assert manifest["exhaustive_train_only_vocabulary"] is True
    assert manifest["shards"] == 8
    assert manifest["similarity_prefilter"] == 0.15
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
    assert '--array="0-7%8"' in submit_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--cpus-per-task=1" in submit_source
    assert "--dependency=\"afterok:$array_job\"" in submit_source
    assert "--kill-on-invalid-dep=yes" in submit_source
    assert "--gres" not in submit_source
