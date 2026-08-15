from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "table1_energy_tilted_latent_transfer.py"
MANIFEST_PATH = EXPERIMENT_DIR / "table1_energy_tilted_latent_transfer_v29_preregistration.json"
RUN_PATH = EXPERIMENT_DIR / "run_table1_energy_tilted_latent_transfer.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_table1_energy_tilted_latent_transfer.sh"


def test_table1_transfer_is_target_free_and_zero_training() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def source_only_row" in source
    assert '"moledit_target_access": False' in source
    assert '"generation_target_access": False' in source
    assert '"model_training": False' in source
    assert '"models_frozen": True' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"failed_attachment_retry": False' in source
    assert '"second_edit": False' in source
    assert "target_smiles" not in source_only_projection_body(source)


def source_only_projection_body(source: str) -> str:
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_only_row"
    )
    return ast.get_source_segment(source, function) or ""


def test_preregistration_fixes_small_table1_subset_and_b28_sampler() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["moledit_target_access"] is False
    assert manifest["model_training"] is False
    assert manifest["num_attempts"] == 20
    assert manifest["per_task"] == 4
    assert len(manifest["table1_tasks"]) == 5
    assert manifest["distance_temperature"] == 0.03
    assert manifest["energy_weight"] == 1.25
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["second_edit"] is False


def test_transfer_reports_per_attempt_and_any20_as_distinct_metrics() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert '"acc_all_t0_15_per_attempt"' in source
    assert '"acc_all_t0_65_per_attempt"' in source
    assert '"acc_any20_t0_15"' in source
    assert '"acc_any20_t0_65"' in source
    assert "is not paper Acc_all" in source


def test_runner_is_bounded_cpu_only() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "eval_balanced.csv" in run_source
    assert "latent_fragment_attachment_kernel.pt" in run_source
    assert "latent_property_energy.pt" in run_source
    assert "--device cpu" in run_source
    assert "completed B29 result exists" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert '--time="00:30:00"' in submit_source
    assert "--cpus-per-task=4" in submit_source
    assert "--mem=12G" in submit_source
    assert "--gres" not in submit_source
