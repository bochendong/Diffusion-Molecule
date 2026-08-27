from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_decoupled_joint_rl as trainer  # noqa: E402


def test_decoupled_advantage_preserves_dense_signal_when_strict_is_constant():
    rows = [
        {"validity": 1.0, "property_bottleneck": value, "property_strict": 0.0}
        for value in (0.1, 0.2, 0.8, 0.4)
    ]
    advantages, record = trainer.decoupled_advantages(
        rows,
        {"validity": 0.5, "property_bottleneck": 1.0, "property_strict": 1.0},
    )
    assert advantages[2] == max(advantages)
    assert record["active_channels"] == ["property_bottleneck"]
    assert record["zero_signal"] is False


def test_each_mode_has_independent_reward_channels():
    assert "source_aligned" not in trainer.CHANNEL_WEIGHTS["de_novo"]
    assert "source_aligned" in trainer.CHANNEL_WEIGHTS["edit"]
    assert "property_bottleneck" in trainer.CHANNEL_WEIGHTS["de_novo"]
    assert "property_bottleneck" in trainer.CHANNEL_WEIGHTS["edit"]


def test_preregistration_matches_runner_and_keeps_target_out_of_reward():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    runner = (HERE / "run_train.sh").read_text()
    assert prereg["training"]["group_size"] == 16
    assert prereg["training"]["gradient_surgery"] == "symmetric two-task PCGrad when cosine is negative"
    assert prereg["training"]["reward_target_smiles_access"] is False
    assert "--group-size 16" in runner
    assert "--gradient-surgery pcgrad" in runner
    assert "gradient_cosine" in (HERE / "train_decoupled_joint_rl.py").read_text()
