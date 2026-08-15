from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "latent_property_energy_guidance.py"
MANIFEST_PATH = EXPERIMENT_DIR / "latent_property_energy_v27_preregistration.json"
RUN_PATH = EXPERIMENT_DIR / "run_latent_property_energy_guidance.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_latent_property_energy_guidance.sh"


def test_energy_is_trained_on_source_relative_train_only_labels() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class LatentPropertyEnergy" in source
    assert "def source_relative_label" in source
    assert "unified.score_property(pair.source_smiles, prop)" in source
    assert "unified.score_property(canonical, prop)" in source
    assert '"energy_fit_labels_train_only": True' in source
    assert '"b26_heldout_access": False' in source
    assert '"official_test_access": False' in source


def test_guidance_changes_one_latent_before_one_token_decode() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "torch.autograd.grad(objective.sum(), guided)" in source
    assert "selected_tokens = distances.argmin" in source
    assert "kernel.fragments.join_fragments(site.core, target_fragment)" in source
    assert '"one_latent_one_token_one_raw_molecule": True' in source
    assert '"candidate_ranking": False' in source
    assert '"failed_attachment_retry": False' in source
    assert '"second_edit": False' in source
    assert '"exact_raw_attempts_per_condition": 20' in source


def test_fit_internal_dev_is_grouped_and_generation_is_target_blind() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "def grouped_fit_dev_split" in source
    assert "pairs[index].source_smiles" in source
    assert '"fit_internal_dev_source_overlap"' in source
    assert "every condition has frozen all 20 direct latent decodes" in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"post_freeze_internal_dev_oracle_access": True' in source


def test_preregistration_locks_single_structural_hypothesis() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["b26_heldout_access"] is False
    assert manifest["official_test_access"] is False
    assert manifest["num_attempts"] == 20
    assert manifest["hard_negatives_per_pair"] == 6
    assert manifest["guidance_steps"] == 4
    assert manifest["candidate_ranking"] is False
    assert manifest["second_edit"] is False
    assert manifest["gates"]["strict_any20_delta"] == 0.05
    assert manifest["gates"]["three_property_strict_delta"] == 0.1


def test_runner_is_cpu_only_and_bounded() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "--device cpu" in run_source
    assert "completed B27 result exists" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--time=\"00:30:00\"" in submit_source
    assert "--cpus-per-task=4" in submit_source
    assert "--mem=12G" in submit_source
    assert "--gres" not in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
