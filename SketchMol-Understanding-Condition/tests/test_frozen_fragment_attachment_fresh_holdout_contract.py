from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "frozen_fragment_attachment_fresh_holdout.py"
MANIFEST_PATH = EXPERIMENT_DIR / "fresh_holdout_v26_preregistration.json"
RUN_PATH = EXPERIMENT_DIR / "run_frozen_fragment_attachment_fresh_holdout.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_frozen_fragment_attachment_fresh_holdout.sh"


def test_fresh_holdout_loads_frozen_b24_without_training_or_search() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "model.load_state_dict" in source
    assert "kernel.evaluate(" in source
    assert "train_model(" not in source
    assert '"model_training": False' in source
    assert '"hyperparameter_search": False' in source
    assert '"repeat_after_scientific_failure": False' in source


def test_fresh_split_excludes_old_dev_and_reconstructed_train() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "historical_sources | reused_sources | train_sources" in source
    assert "historical_keys | reused_keys | train_keys" in source
    assert '"fresh_train_source_overlap"' in source
    assert '"fresh_reused_dev_pair_overlap"' in source
    assert '"all_split_overlaps_zero"' in source


def test_preregistration_locks_once_only_n20_and_top_conference_gates() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["frozen_commit"] == "b9d5f5e"
    assert manifest["prior_validation_selection_seeds"] == [1742, 2719]
    assert manifest["fresh_validation_selection_seed"] == 4099
    assert manifest["num_attempts"] == 20
    assert manifest["evaluation_target_access"] is False
    assert manifest["official_test_access"] is False
    assert manifest["model_training"] is False
    assert manifest["hyperparameter_search"] is False
    assert manifest["repeat_after_scientific_failure"] is False
    assert manifest["gates"] == {
        "mean_source_tanimoto": 0.4,
        "mean_unique_valid": 12.0,
        "minimum_conditions_per_property_count": 5,
        "strict_any20": 0.65,
        "three_property_strict_any20": 0.5,
        "two_property_strict_any20": 0.8,
        "validity": 0.95,
    }


def test_runner_is_cpu_only_bounded_and_cannot_overwrite_completed_result() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "once-only fresh-heldout result already exists" in run_source
    assert "--device cpu" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--time=\"00:15:00\"" in submit_source
    assert "--cpus-per-task=2" in submit_source
    assert "--mem=8G" in submit_source
    assert "--gres" not in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
