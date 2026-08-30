#!/usr/bin/env python3
"""CPU-only source and preregistration contracts for P32."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_collect():
    spec = importlib.util.spec_from_file_location("p32_collect_gate", ROOT / "collect_gate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_is_dual_mode_and_locked():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    assert prereg["status"] == "locked_before_submission"
    assert prereg["training"]["task_gradient_merge"] == "paired symmetric PCGrad"
    assert prereg["evaluation"]["property_reranking"] is False
    assert prereg["promotion"]["both_modes_strict_above_P24_direct"] is True


def test_train_uses_online_support_and_paired_pcgrad():
    source = (ROOT / "train_shared_graph_repair_rl.py").read_text()
    assert "protocol.online_supports" in source
    assert "exact_action_value_backward" in source
    assert "assign_paired_pcgrad" in source
    assert "reward-ranked" not in source.lower()


def test_eval_is_greedy_without_property_reranking():
    source = (ROOT / "evaluate_checkpoint.py").read_text()
    assert "protocol.greedy_rollout" in source
    assert '"property_reranking": False' in source


def test_runners_pin_both_benchmark_oracles():
    for name in ("run_audit.sh", "run_train.sh", "run_eval.sh"):
        source = (ROOT / name).read_text()
        assert "gsk3b_legacy_sklearn_compatible.pkl" in source
        assert "drd2_graph2graph_svc_py36.pkl" in source


def test_gate_requires_both_modes_to_improve():
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
        mode: {"strict_macro": 0.42, "relaxed_macro": 0.5, "valid_macro": 0.89}
        for mode in collect.MODES
    }
    _checks, passed = collect.assess(good, direct, step0)
    assert passed
    bad = {mode: dict(values) for mode, values in good.items()}
    bad["edit"]["strict_macro"] = 0.41
    _checks, passed = collect.assess(bad, direct, step0)
    assert not passed


def test_protocol_mean_accepts_generator_source():
    source = (ROOT / "graph_repair_protocol.py").read_text()
    assert "materialized = [float(value) for value in values]" in source
