from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "energy_tilted_vq_fragment_sampling.py"
MANIFEST_PATH = EXPERIMENT_DIR / "energy_tilted_vq_v28_preregistration.json"
RUN_PATH = EXPERIMENT_DIR / "run_energy_tilted_vq_fragment_sampling.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_energy_tilted_vq_fragment_sampling.sh"


def test_quantizer_samples_one_latent_token_without_molecule_ranking() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def token_energy" in source
    assert "logits = -distance_offset" in source
    assert "selected_tokens = torch.multinomial" in source
    assert "kernel.fragments.join_fragments(site.core, target_fragment)" in source
    assert "selected_tokens = distances.argmin" not in source
    assert '"one_latent_one_sampled_token_one_raw_molecule": True' in source
    assert '"molecular_candidate_materialization": False' in source
    assert '"molecular_candidate_ranking": False' in source


def test_energy_and_b24_are_frozen_and_no_oracle_enters_generation() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "parameter.requires_grad_(False)" in source
    assert '"model_training": False' in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert "all latent-token" in source
    assert '"failed_attachment_retry": False' in source
    assert '"second_edit": False' in source
    assert '"exact_raw_attempts_per_condition": 20' in source


def test_preregistration_fixes_energy_tilt_before_first_run() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "preregistered_before_first_run"
    assert manifest["b26_heldout_access"] is False
    assert manifest["official_test_access"] is False
    assert manifest["num_attempts"] == 20
    assert manifest["distance_temperature"] == 0.03
    assert manifest["energy_weight"] == 1.25
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["gates"]["strict_any20_delta"] == 0.05
    assert manifest["gates"]["three_property_strict_delta"] == 0.1


def test_runner_is_zero_training_cpu_only_and_bounded() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "latent_property_energy.pt" in run_source
    assert "--device cpu" in run_source
    assert "completed B28 result exists" in run_source
    assert "--account=def-hup-ab_cpu" in submit_source
    assert "--time=\"00:15:00\"" in submit_source
    assert "--cpus-per-task=4" in submit_source
    assert "--mem=12G" in submit_source
    assert "--gres" not in submit_source
