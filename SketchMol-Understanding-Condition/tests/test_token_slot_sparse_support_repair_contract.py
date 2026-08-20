from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "evaluate_token_slot_sparse_support_repair_v3.py"
PREREGISTRATION = EXPERIMENT / "token_slot_sparse_support_repair_v3_preregistration.json"
RUNNER = EXPERIMENT / "run_token_slot_sparse_support_repair_v3.sh"
SUBMITTER = EXPERIMENT / "submit_token_slot_sparse_support_repair_v3.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_frozen_evaluation_only_repair() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "frozen_token_slot_sparse_support_repair_v3"
    assert manifest["single_mechanism_change"] == "exact_zero_inactive_property_slots"
    assert manifest["training"] is False
    assert manifest["support_probability_threshold"] == 0.5
    assert manifest["threshold_search"] is False
    assert manifest["molecule_generation"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)


def test_repair_uses_frozen_support_and_exact_zero_slots() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "is_trainable=False" in source
    assert "torch.where(support, raw, torch.zeros_like(raw))" in source
    assert "tokens[:, 1:, :]" in source
    assert "threshold_search" in source
    assert "oracle_canonical_velocity_max_abs" in source
    assert "oracle_canonical_flow_mse_ratio" in source
    assert "oracle_canonical_flow_relative_error" in source
    assert "optimizer" not in source
    assert "loss.backward" not in source
    assert "generated_smiles" not in source


def test_runner_exposes_no_generation_or_oracle_surface() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "evaluation-target" not in runner
    assert "oracle" not in runner.lower()
    assert "candidate" not in runner.lower()
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submitter
    assert "--time=00:20:00" in submitter
    assert "dependency" not in submitter.lower()
