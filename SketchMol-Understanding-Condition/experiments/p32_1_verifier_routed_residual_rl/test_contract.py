#!/usr/bin/env python3
"""CPU-only source contracts for P32.1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_collect():
    spec = importlib.util.spec_from_file_location(
        "p321_collect", ROOT / "collect_residual_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_locks_residual_routing_before_submission():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    assert prereg["status"] == "locked_before_submission"
    assert prereg["routing"]["strict_direct"] == "hard accept and return unchanged"
    assert prereg["training"]["policy_update"].startswith("exact categorical")
    assert prereg["evaluation"]["property_reranking_within_residual_actions"] is False


def test_protocol_hard_accepts_strict_direct_and_starts_failures_from_direct():
    source = (ROOT / "residual_protocol.py").read_text()
    assert "def hard_accept_direct" in source
    assert 'return direct_smiles(record) or "C"' in source
    assert '"kind": "hard_accept_direct"' in source
    assert "initial_verifier_feedback" in source


def test_training_is_failed_only_exact_rl_with_pcgrad():
    source = (ROOT / "train_residual_graph_rl.py").read_text()
    assert "select_failed" in source
    assert "exact_action_value_backward" in source
    assert "assign_paired_pcgrad" in source
    assert "reward-ranked" not in source.lower()


def test_gate_requires_rl_gain_in_both_modes():
    collect = load_collect()
    direct = {
        mode: {"strict_macro": 0.4, "relaxed_macro": 0.5, "valid_macro": 0.9}
        for mode in collect.MODES
    }
    step0 = {
        mode: {"strict_macro": 0.41, "relaxed_macro": 0.5, "valid_macro": 0.9}
        for mode in collect.MODES
    }
    good = {
        mode: {"strict_macro": 0.42, "relaxed_macro": 0.5, "valid_macro": 0.9}
        for mode in collect.MODES
    }
    _checks, passed = collect.assess(
        good, direct, step0, {mode: 1 for mode in collect.MODES}
    )
    assert passed
    bad = {mode: dict(values) for mode, values in good.items()}
    bad["edit"]["strict_macro"] = 0.41
    _checks, passed = collect.assess(
        bad, direct, step0, {mode: 1 for mode in collect.MODES}
    )
    assert not passed


def test_runners_pin_both_benchmark_oracles():
    for name in ("run_audit.sh", "run_train.sh", "run_eval.sh"):
        source = (ROOT / name).read_text()
        assert "gsk3b_legacy_sklearn_compatible.pkl" in source
        assert "drd2_graph2graph_svc_py36.pkl" in source
