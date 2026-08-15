from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
MODEL = EXPERIMENT_DIR / "structure_constrained_joint_latent.py"
MANIFEST = EXPERIMENT_DIR / "structure_constrained_joint_latent_v32_preregistration.json"
RUN = EXPERIMENT_DIR / "run_structure_constrained_joint_latent.sh"
SUBMIT = EXPERIMENT_DIR / "submit_structure_constrained_joint_latent.sh"


def test_constraint_is_applied_in_latent_distribution_before_molecule_creation() -> None:
    source = MODEL.read_text(encoding="utf-8")
    ast.parse(source)
    assert "structure_shortfall = torch.relu(-structure_margin)" in source
    assert "structure_dual_weight" in source
    assert "feasibility_log_probability" in source
    assert "torch.multinomial" in source
    assert "site_index, token_index = divmod" in source
    assert "kernel.fragments.join_fragments(site.core, token)" in source
    assert source.index("torch.multinomial") < source.index(
        '"smiles": kernel.fragments.join_fragments(site.core, token)'
    )
    assert "sorted_candidates" not in source
    assert "topk(" not in source


def test_b31_assay_energy_is_frozen_and_only_structure_head_is_trained() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "def load_b31_energy(" in source
    assert "parameter.requires_grad_(False)" in source
    assert "def train_structure_model(" in source
    assert '"b31_assay_energy_frozen": True' in source
    assert '"structure_head_only_trainable": True' in source
    assert '"property_oracle_generation_access": False' in source
    assert "load_pinned_oracles" in source


def test_preregistration_locks_strict_similarity_and_budget() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["exact_raw_attempts_per_condition"] == 20
    assert manifest["structure_similarity_threshold"] == 0.65
    assert manifest["gates"]["overall_any20_t0_15"] == 0.9
    assert manifest["gates"]["overall_any20_t0_65"] == 0.5
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["evaluation_target_access"] is False
    assert manifest["moledit_table1_access"] is False


def test_runner_is_cpu_only_single_seed_and_pins_b31() -> None:
    run_source = RUN.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert "assay_joint_site_token_energy.pt" in run_source
    assert "gsk3b_legacy_sklearn_compatible.pkl" in run_source
    assert "drd2_graph2graph_svc_py36.pkl" in run_source
    assert "--device cpu" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--cpus-per-task=8" in submit_source
    assert "--mem=32G" in submit_source
    assert "--gres" not in submit_source
    assert "--array" not in submit_source
