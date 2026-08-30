#!/usr/bin/env python3
"""CPU-only source contracts for P32.2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_math():
    spec = importlib.util.spec_from_file_location(
        "p322_math", ROOT / "trajectory_math.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_is_locked_to_terminal_credit():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    assert prereg["status"] == "locked_before_submission"
    assert prereg["training"]["trajectory_group_size"] == 8
    assert "terminal advantage" in prereg["training"]["credit_assignment"]
    assert prereg["training"]["terminal_return"]["stopped_failed_direct"] == 0.0


def test_terminal_return_prefers_strict_and_does_not_reward_stop():
    source = (ROOT / "train_multistep_trajectory_rl.py").read_text()
    assert "def terminal_return" in source
    assert "if feedback.strict_success" in source
    assert 'changed = any(action.kind != "stop"' in source
    assert "return 0.0" in source


def test_group_uses_distinct_first_actions_and_terminal_credit():
    source = (ROOT / "train_multistep_trajectory_rl.py").read_text()
    assert "weighted_without_replacement" in source
    assert "trajectory_group_backward" in source
    assert "trajectory.terminal_return" in source
    assert "exact_action_value_backward" not in source
    assert "assign_paired_pcgrad" in source


def test_centered_advantages_have_zero_mean():
    trajectory_math = load_math()
    values = trajectory_math.centered_advantages([0.0, 0.02, 0.2, 1.0])
    assert abs(sum(values)) < 1e-9
    assert trajectory_math.centered_advantages([0.0, 0.0]) == [0.0, 0.0]


def test_runner_keeps_pinned_oracles_and_frozen_data():
    for name in ("run_preflight.sh", "run_train.sh", "run_eval.sh"):
        source = (ROOT / name).read_text()
        assert "gsk3b_legacy_sklearn_compatible.pkl" in source
        assert "drd2_graph2graph_svc_py36.pkl" in source
