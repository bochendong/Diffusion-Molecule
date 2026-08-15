from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
MODEL = EXPERIMENT_DIR / "continuous_pareto_latent_transport.py"
MANIFEST = EXPERIMENT_DIR / "continuous_pareto_latent_transport_v34_preregistration.json"
RUN = EXPERIMENT_DIR / "run_continuous_pareto_latent_transport.sh"
SUBMIT = EXPERIMENT_DIR / "submit_continuous_pareto_latent_transport.sh"


def test_continuous_preference_is_a_trainable_latent_energy_input() -> None:
    source = MODEL.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class ContinuousParetoTransport" in source
    assert "preference * structure_margin" in source
    assert "(1.0 - preference) * assay_margin" in source
    assert "def continuous_target(" in source
    assert "torch.optim.AdamW" in source
    assert '"continuous_transport_only_trainable": True' in source
    assert '"b31_assay_energy_frozen": True' in source
    assert '"b32_structure_energy_frozen": True' in source


def test_all_twenty_states_freeze_before_any_molecule_is_constructed() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "frozen_draws" in source
    assert "torch.multinomial" in source
    assert "B34 did not freeze exactly 20 latent states" in source
    assert source.index("frozen_draws.append") < source.index(
        "kernel.fragments.join_fragments(site.core, token)"
    )
    assert "sorted_candidates" not in source
    assert "topk(" not in source


def test_preregistration_locks_target_access_budget_and_unchanged_gates() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["exact_raw_attempts_per_condition"] == 20
    assert manifest["generation_preference_min"] == 0.1
    assert manifest["generation_preference_max"] == 1.0
    assert manifest["continuous_similarity_min"] == 0.15
    assert manifest["continuous_similarity_max"] == 0.65
    assert manifest["gates"]["overall_any20_t0_15"] == 0.9
    assert manifest["gates"]["overall_any20_t0_65"] == 0.7
    assert manifest["gates"]["mean_source_tanimoto"] == 0.6
    assert manifest["b33_fresh_source_access"] is False
    assert manifest["evaluation_target_access"] is False
    assert manifest["moledit_table1_access"] is False
    assert manifest["official_test_access"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False


def test_runner_is_bounded_cpu_only_and_does_not_read_b33() -> None:
    run_source = RUN.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert "structure_feasibility_energy.pt" in run_source
    assert "assay_joint_site_token_energy.pt" in run_source
    assert "pareto_conditioned_joint_latent_v33" not in run_source
    assert "--device cpu" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert '--time="00:45:00"' in submit_source
    assert "--cpus-per-task=4" in submit_source
    assert "--mem=20G" in submit_source
    assert "--gres" not in submit_source
    assert "--array" not in submit_source
