from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "token_slot_lora_property_compiler_v3.py"
PREREGISTRATION = EXPERIMENT / "token_slot_lora_property_compiler_v3_preregistration.json"
RUNNER = EXPERIMENT / "run_token_slot_lora_property_compiler_v3.sh"
SUBMITTER = EXPERIMENT / "submit_token_slot_lora_property_compiler_v3.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_token_slot_lora_probe() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "train_only_token_slot_lora_property_compiler_v3"
    assert manifest["single_mechanism_change"] == "token_slot_lora_property_compiler"
    assert manifest["common_llm_prompt_contains_source"] is False
    assert manifest["language_fit_excludes_graph_probe_property_pairs"] is True
    assert manifest["molecule_generation"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["generation_target_access"] is False
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)


def test_compiler_uses_token_slots_and_live_lora() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "class TokenPropertySlotDecoder" in source
    assert "span_targets" in source
    assert "slot_attention" in source
    assert "latent_lora=True" in source
    assert "language_fit_excludes_graph_probe_property_pairs" in source
    assert "heldout_pair_overlap" in source
    assert "last_hidden_state" in source
    assert "source_smiles" not in source
    assert "generated_smiles" not in source


def test_runner_exposes_no_generation_or_oracle_surface() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "evaluation-target" not in runner
    assert "oracle" not in runner.lower()
    assert "candidate" not in runner.lower()
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submitter
    assert "--time=00:45:00" in submitter
    assert "dependency" not in submitter.lower()
