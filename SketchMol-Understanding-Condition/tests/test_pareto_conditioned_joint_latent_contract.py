from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
MODEL = EXPERIMENT_DIR / "pareto_conditioned_joint_latent.py"
MANIFEST = EXPERIMENT_DIR / "pareto_conditioned_joint_latent_v33_preregistration.json"
RUN = EXPERIMENT_DIR / "run_pareto_conditioned_joint_latent.sh"
SUBMIT = EXPERIMENT_DIR / "submit_pareto_conditioned_joint_latent.sh"


def test_all_twenty_latent_states_freeze_before_product_construction() -> None:
    source = MODEL.read_text(encoding="utf-8")
    ast.parse(source)
    assert "frozen_draws" in source
    assert "torch.multinomial" in source
    assert "B33 did not freeze exactly 20 latent states" in source
    assert source.index("for level_index, level") < source.index("for level, flat_index")
    assert source.index("frozen_draws.append") < source.index(
        "kernel.fragments.join_fragments(site.core, token)"
    )
    assert "sorted_candidates" not in source
    assert "topk(" not in source


def test_b31_b32_are_frozen_and_no_new_model_is_trained() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "def load_b32_structure_energy(" in source
    assert source.count("parameter.requires_grad_(False)") >= 1
    assert "torch.optim" not in source
    assert '"model_training": False' in source
    assert '"b31_assay_energy_frozen": True' in source
    assert '"b32_structure_energy_frozen": True' in source


def test_fresh_sources_exclude_every_b31_b32_energy_source() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "def select_fresh_energy_sources(" in source
    assert "source not in excluded" in source
    assert "selected_sources & excluded" in source
    assert '"fresh_b31_b32_source_overlap"' in source
    assert '"evaluation_target_access": False' in source
    assert '"moledit_table1_access": False' in source
    assert '"official_test_access": False' in source


def test_preregistration_locks_pareto_schedule_budget_and_gates() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    levels = manifest["structure_preference_levels"]
    assert levels == [
        {"name": "property", "scale": 0.0, "attempts": 5},
        {"name": "balanced", "scale": 1.0, "attempts": 5},
        {"name": "structure", "scale": 1.5, "attempts": 5},
        {"name": "strict", "scale": 2.0, "attempts": 5},
    ]
    assert sum(level["attempts"] for level in levels) == 20
    assert manifest["gates"]["overall_any20_t0_15"] == 0.9
    assert manifest["gates"]["overall_any20_t0_65"] == 0.7
    assert manifest["gates"]["minimum_task_any20_t0_15"] == 0.85
    assert manifest["gates"]["minimum_task_any20_t0_65"] == 0.6
    assert manifest["gates"]["mean_source_tanimoto"] == 0.6
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False


def test_runner_is_cpu_only_small_single_job_and_pins_b32() -> None:
    run_source = RUN.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert "structure_feasibility_energy.pt" in run_source
    assert "--b32-summary" in run_source
    assert "--device cpu" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert '--time="00:30:00"' in submit_source
    assert "--cpus-per-task=4" in submit_source
    assert "--mem=16G" in submit_source
    assert "--gres" not in submit_source
    assert "--array" not in submit_source
