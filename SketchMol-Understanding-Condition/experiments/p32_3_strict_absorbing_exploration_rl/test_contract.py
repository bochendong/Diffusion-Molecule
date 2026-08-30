#!/usr/bin/env python3
"""CPU-only source contracts for P32.3."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_collector():
    spec = importlib.util.spec_from_file_location("p323_collect", ROOT / "collect_gate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_locks_sparse_editing_rl_before_submission():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    assert prereg["status"] == "locked_before_submission"
    assert prereg["support_curriculum"]["minimum_unique_edit_rows"] == 5
    assert prereg["training"]["strict_success"] == "absorbing terminal state"
    assert prereg["training"]["editing_trajectories"].startswith("all first actions")


def test_training_uses_policy_gradient_and_absorbing_strict_feedback():
    source = (ROOT / "train_strict_absorbing_rl.py").read_text()
    assert "first_action.terminal or first_feedback.strict_success" in source
    assert "trajectory_group_backward" in source
    assert "assign_paired_pcgrad" in source
    assert "reward_guided" not in source
    assert "dpo" not in source.lower()
    assert "raft" not in source.lower()


def test_inference_stops_on_strict_feedback():
    source = (ROOT / "evaluate_strict_absorbing_checkpoint.py").read_text()
    assert "if action.terminal or feedback.strict_success" in source
    assert "property_reranking" not in source


def test_gate_requires_edit_gain_and_denovo_retention():
    collect = load_collector()
    direct = {
        "de_novo": {"strict_macro": 0.39, "relaxed_macro": 0.39, "valid_macro": 0.99},
        "edit": {"strict_macro": 0.66, "relaxed_macro": 0.78, "valid_macro": 0.98},
    }
    step0 = {
        "de_novo": {"strict_macro": 0.54, "relaxed_macro": 0.54, "valid_macro": 1.0},
        "edit": {"strict_macro": 0.66, "relaxed_macro": 0.78, "valid_macro": 1.0},
    }
    passing = {
        "de_novo": {"strict_macro": 0.54, "relaxed_macro": 0.54, "valid_macro": 1.0},
        "edit": {"strict_macro": 0.68, "relaxed_macro": 0.80, "valid_macro": 1.0},
    }
    _checks, passed = collect.assess(passing, direct, step0, {"de_novo": 18, "edit": 1})
    assert passed
    failing = {mode: dict(values) for mode, values in passing.items()}
    failing["de_novo"]["strict_macro"] = 0.53
    assert not collect.assess(failing, direct, step0, {"de_novo": 18, "edit": 1})[1]


def test_runners_keep_pinned_oracles():
    for name in ("run_preflight.sh", "run_support_audit.sh", "run_train.sh", "run_eval.sh"):
        source = (ROOT / name).read_text()
        assert "gsk3b_legacy_sklearn_compatible.pkl" in source
        assert "drd2_graph2graph_svc_py36.pkl" in source
