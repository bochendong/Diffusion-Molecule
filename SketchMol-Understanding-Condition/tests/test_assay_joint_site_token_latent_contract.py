from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
MODEL = EXPERIMENT_DIR / "assay_joint_site_token_latent.py"
MANIFEST = EXPERIMENT_DIR / "assay_joint_site_token_latent_v31_preregistration.json"
RUN = EXPERIMENT_DIR / "run_assay_joint_site_token_latent.sh"
SUBMIT = EXPERIMENT_DIR / "submit_assay_joint_site_token_latent.sh"


def test_joint_latent_samples_states_without_molecule_ranking() -> None:
    source = MODEL.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def joint_actions(" in source
    assert "torch.multinomial" in source
    assert "logits.reshape(-1)" in source
    assert "site_index, token_index = divmod" in source
    assert "join_fragments(site.core, token)" in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"oracle_selection": False' in source
    assert '"one_joint_latent_state_one_raw_molecule": True' in source
    assert "sorted_candidates" not in source
    assert "topk(" not in source


def test_training_is_train_only_and_table1_is_gate_guarded() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "reconstruct_b24_train_pairs" in source
    assert "attachment_site_eligible_sources" in source
    assert "sources_without_attachment_site" in source
    assert "if kernel.source_sites(str(pair.source_smiles), site_config)" in source
    assert '"training_source_selection": source_selection' in source
    assert '"training_labels_from_b24_train_sources_only": True' in source
    assert '"evaluation_source_training_access": False' in source
    assert "if internal_gate_passed:" in source
    assert '"table1_eval_rows_read"] = 0' in source
    assert "load_pinned_oracles" in source
    assert "B30-r1" in source
    assert "target_smiles" not in source


def test_preregistration_locks_budget_split_and_scientific_gates() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["exact_raw_attempts_per_condition"] == 20
    assert manifest["train_source_limit"] == 512
    assert manifest["actions_per_condition"] == 96
    assert manifest["dev_fraction"] == 0.2
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["evaluation_source_training_access"] is False
    assert manifest["tasks"] == [
        "GSK3B:increase",
        "DRD2:decrease+MW:decrease+SA:decrease",
    ]
    assert manifest["gates"]["minimum_task_any20_t0_15"] == 0.7
    assert manifest["gates"]["strict_auc"] == 0.7


def test_runner_is_cpu_only_single_seed_and_uses_pinned_oracles() -> None:
    run_source = RUN.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert "gsk3b_legacy_sklearn_compatible.pkl" in run_source
    assert "drd2_graph2graph_svc_py36.pkl" in run_source
    assert "--device cpu" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--cpus-per-task=8" in submit_source
    assert "--mem=32G" in submit_source
    assert "--gres" not in submit_source
    assert "--array" not in submit_source
