from __future__ import annotations

import json
from pathlib import Path

import pytest

from collect_gate import assess
from source_reward import source_constrained_reward


HERE = Path(__file__).resolve().parent


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
        "source_similarity": 0.2,
        "copy": False,
    }
    value.update(updates)
    return value


def summary(strict=0.60, relaxed=0.85, valid=0.95, bucket_strict=0.60):
    return {
        "aggregate": {
            "edit_strict_065_macro": strict,
            "edit_relaxed_015_macro": relaxed,
            "edit_valid_macro": valid,
        },
        "buckets": {
            f"edit:task-{index}": {"strict_rate": bucket_strict}
            for index in range(10)
        },
    }


def test_reward_is_source_dense_strict_dominant_and_copy_aware():
    channels = {"source_aligned": 0.5}
    invalid = source_constrained_reward(channels, details(valid=False), "edit")
    low_source = source_constrained_reward(channels, details(source_similarity=0.10), "edit")
    high_source = source_constrained_reward(channels, details(source_similarity=0.60), "edit")
    property_only = source_constrained_reward(
        channels, details(property_strict=True, relaxed=True, source_similarity=0.60), "edit"
    )
    strict = source_constrained_reward(
        channels,
        details(property_strict=True, relaxed=True, strict=True, source_similarity=0.70),
        "edit",
    )
    copied = source_constrained_reward(
        channels, details(property_strict=True, strict=True, relaxed=True, source_similarity=1.0, copy=True), "edit"
    )
    noncopy = source_constrained_reward(
        channels, details(property_strict=True, strict=True, relaxed=True, source_similarity=1.0), "edit"
    )
    assert invalid < low_source < high_source < property_only < strict
    assert copied < noncopy
    with pytest.raises(ValueError):
        source_constrained_reward(channels, details(), "de_novo")


def test_small_gate_contract():
    baseline = summary()
    passing = summary(strict=0.63, relaxed=0.84, valid=0.94, bucket_strict=0.63)
    result = assess(baseline, passing)
    assert result["passed"] is True
    failing = summary(strict=0.61, relaxed=0.85, valid=0.95, bucket_strict=0.61)
    assert assess(baseline, failing)["passed"] is False


def test_frozen_experiment_contract():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    trainer = (HERE / "train_edit_source_rloo.py").read_text()
    run_train = (HERE / "run_train.sh").read_text()
    submit = (HERE / "submit_p32_4.sh").read_text()
    assert prereg["status"] == "frozen before GPU training"
    assert prereg["training"]["algorithm"] == "online sequence RLOO"
    assert prereg["training"]["mode"] == "edit only"
    assert prereg["initialization"]["construction_policy"].startswith("frozen")
    assert "p31.scalar_reward = source_constrained_reward" in trainer
    assert "editing_specialist/adapter" in run_train
    assert "--target-updates 20" in run_train
    assert "for step in 005 010 020" in submit
    assert "P324_STEP=000" in submit
