from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from collect_gate import assess
from hard_boundary_reward import INELIGIBLE_FLOOR, hard_boundary_reward


HERE = Path(__file__).resolve().parent
P31 = HERE.parent / "p31_1_frontier_online_rloo"
sys.path.insert(0, str(P31))
from rloo_math import rloo_advantages  # noqa: E402


def details(**updates):
    value = {
        "valid": True,
        "canonical": True,
        "property_strict": False,
        "strict": False,
        "relaxed": False,
        "property_fraction": 0.5,
        "mean_satisfaction": 0.5,
        "bottleneck": 0.5,
        "source_similarity": 0.7,
        "copy": False,
    }
    value.update(updates)
    return value


def summary(strict=0.58, relaxed=0.75, valid=0.98, bucket=0.58):
    return {
        "aggregate": {
            "edit_strict_065_macro": strict,
            "edit_relaxed_015_macro": relaxed,
            "edit_valid_macro": valid,
        },
        "buckets": {
            f"edit:task-{index}": {"strict_rate": bucket}
            for index in range(10)
        },
    }


def test_hard_boundary_and_strict_ordering():
    invalid = hard_boundary_reward({}, details(valid=False), "edit")
    low_similarity = hard_boundary_reward({}, details(source_similarity=0.649), "edit")
    copy = hard_boundary_reward({}, details(source_similarity=1.0, copy=True), "edit")
    feasible = hard_boundary_reward({}, details(source_similarity=0.65), "edit")
    strict = hard_boundary_reward(
        {}, details(source_similarity=0.65, property_strict=True, strict=True), "edit"
    )
    assert invalid == low_similarity == copy == INELIGIBLE_FLOOR
    assert INELIGIBLE_FLOOR < feasible < strict
    with pytest.raises(ValueError):
        hard_boundary_reward({}, details(), "de_novo")


def test_ineligible_candidates_have_negative_advantage_in_updated_group():
    strict = hard_boundary_reward(
        {}, details(property_strict=True, strict=True, source_similarity=0.7), "edit"
    )
    feasible = hard_boundary_reward({}, details(source_similarity=0.7), "edit")
    returns = [strict, feasible, *([INELIGIBLE_FLOOR] * 14)]
    advantages = rloo_advantages(returns)
    assert advantages[0] > 0
    assert all(value < 0 for value in advantages[2:])


def test_gate_and_frozen_protocol():
    assert assess(summary(), summary(strict=0.60, relaxed=0.74, valid=0.97, bucket=0.60))["passed"]
    prereg = json.loads((HERE / "preregistration.json").read_text())
    trainer = (HERE / "train_hard_boundary_rloo.py").read_text()
    submit = (HERE / "submit_p34.sh").read_text()
    assert prereg["status"] == "frozen before GPU training"
    assert prereg["reward"]["ineligible_dense_similarity_credit"] is False
    assert prereg["reward"]["ineligible_reward"] == INELIGIBLE_FLOOR
    assert "p31.scalar_reward = hard_boundary_reward" in trainer
    assert "for step in 005 010 020" in submit
    assert "p34-e000" not in submit
