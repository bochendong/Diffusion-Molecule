from __future__ import annotations

import json
from pathlib import Path

import pytest

from rloo_math import rloo_advantages, scalar_reward


HERE = Path(__file__).resolve().parent


def details(**updates):
    value = {
        "valid": True,
        "canonical": True,
        "strict": False,
        "property_strict": False,
        "property_fraction": 0.5,
        "mean_satisfaction": 0.5,
        "bottleneck": 0.5,
        "source_similarity": 0.2,
        "copy": False,
    }
    value.update(updates)
    return value


def test_rloo_is_leave_one_out_and_zero_sum():
    values = rloo_advantages([1.0, 2.0, 4.0])
    assert values == pytest.approx([-2.0, -0.5, 2.5])
    assert sum(values) == pytest.approx(0.0)


def test_strict_reward_ordering_and_invalid_floor():
    channels = {}
    invalid = scalar_reward(channels, details(valid=False), "de_novo")
    partial = scalar_reward(channels, details(), "de_novo")
    denovo_strict = scalar_reward(
        channels, details(strict=True, property_strict=True), "de_novo"
    )
    edit_property_only = scalar_reward(
        channels, details(property_strict=True, source_similarity=0.4), "edit"
    )
    edit_strict = scalar_reward(
        channels,
        details(strict=True, property_strict=True, source_similarity=0.7),
        "edit",
    )
    assert invalid < partial < denovo_strict
    assert partial < edit_property_only < edit_strict


def test_preregistered_online_rl_contract():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    amendment = json.loads((HERE / "amendment_01_gate_feasibility.json").read_text())
    trainer = (HERE / "train_online_rloo.py").read_text()
    submit = (HERE / "submit_frontier_rloo.sh").read_text()
    assert prereg["training"]["online_fresh_rollouts"] is True
    assert prereg["training"]["advantage"].startswith("leave-one-out")
    assert prereg["training"]["sft_anchor_during_rl"] is False
    assert prereg["evaluation"]["both_modes_at_every_checkpoint"] is True
    assert amendment["recorded_before"] == "any P31.1 GPU training"
    assert amendment["gpu_training_started_before_amendment"] is False
    assert "20 conditions per arity" in amendment["amended_de_novo_gate"]
    assert "completion token log-probability sum" in prereg["training"]["policy_log_probability"]
    assert "decoupled_advantages" not in trainer
    assert "chosen_sft_loss" not in trainer
    assert "temperature=1.0" in trainer
    assert "top_p=1.0" in trainer
    assert "top_k=0" in trainer
    assert "for mode in de_novo edit" in submit
    assert "for step in 025 050 100" in submit
