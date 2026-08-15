from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "two_step_residual_fragment_rollout.py"
RUN_PATH = EXPERIMENT_DIR / "run_two_step_residual_fragment_rollout.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_two_step_residual_fragment_rollout.sh"


def test_second_step_reencodes_intermediate_without_property_decision() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def second_step" in source
    assert "graph.molecule_example" in source
    assert "encode_one_source(representation, example, device)" in source
    assert "if int(pair.property_count) < 3:" in source
    assert '"three_property_fragment_steps": 2' in source
    assert '"oracle_free_residual_property_slots": True' in source
    assert '"property_based_early_stop": False' in source


def test_final_exact_n20_is_frozen_before_evaluation_without_oracle_retry() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "frozen = freeze_rollouts(" in source
    assert "rows, metrics = evaluate(frozen, args)" in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"generation_rdkit_validity_feedback": False' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"failed_attachment_retry": False' in source
    assert '"exact_raw_attempts_per_condition": 20' in source


def test_runner_is_cpu_only_bounded_matched_ablation() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "latent_fragment_attachment_kernel_v24/cpu_seed_1761" in run_source
    assert "--gate-3p-strict-delta 0.14" in run_source
    assert "--num-attempts 20" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=4G" in submit_source
    assert "00:08:00" in submit_source
    assert "--gres" not in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
