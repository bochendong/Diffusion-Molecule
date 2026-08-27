from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_p23_joint_grpo as rl  # noqa: E402


def row(mode: str, bucket: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        count = int(bucket[0])
        conditions = [
            {"property": prop, "goal": {"around": float(index + offset)}}
            for offset, prop in enumerate(rl.protocol.PROPERTIES[:count])
        ]
        task_key = "+".join(f"{item['property']}:around" for item in conditions)
        source = "<EMPTY>"
        plan = "BUILD"
    else:
        task_key = bucket
        conditions = [
            {"property": part.split(":")[0], "goal": part.split(":")[1]}
            for part in bucket.split("+")
        ]
        source = "CCO"
        plan = "MODIFY"
    return {
        "example_id": f"{mode}:{bucket}:{index}",
        "task_mode": mode,
        "task_key": task_key,
        "target_smiles": "forbidden-to-reward",
        "messages": [
            {"role": "system", "content": rl.protocol.SYSTEM},
            {"role": "user", "content": json.dumps({"conditions": conditions, "source": source})},
            {"role": "assistant", "content": json.dumps({"plan": plan, "smiles": "CCN"}, separators=(",", ":"))},
        ],
    }


def test_selection_is_exactly_three_complete_target_rounds():
    rows = []
    for count in (5, 6, 7):
        rows.extend(row("de_novo", f"{count}p", index) for index in range(5))
    for task in sorted(rl.TARGET_EDIT_TASKS):
        rows.extend(row("edit", task, index) for index in range(5))
    selected = rl.select_training_rows(rows, rounds=3, seed=2525)
    assert len(selected) == 39
    assert Counter(rl.target_bucket(item) for item in selected) == {
        bucket: 3 for bucket in rl.TARGET_BUCKETS
    }
    for start in range(0, 39, 13):
        assert {rl.target_bucket(item) for item in selected[start : start + 13]} == set(rl.TARGET_BUCKETS)


def test_group_advantages_are_centered_and_zero_for_ties():
    assert rl.group_advantages([2.0, 2.0, 2.0, 2.0]) == [0.0] * 4
    values = rl.group_advantages([0.0, 1.0, 2.0, 3.0])
    assert sum(values) == pytest.approx(0.0)


def test_runners_pin_both_assay_oracles_and_checkpoint_each_round():
    train = (HERE / "run_train.sh").read_text()
    evaluate = (HERE / "run_eval.sh").read_text()
    assert "SUCC_GSK3B_ORACLE_PATH" in train + evaluate
    assert "SUCC_DRD2_ORACLE_PATH" in train + evaluate
    assert "--checkpoint-every 13" in train
    assert "--sft-anchor-weight 1.0" in train


def test_preregistration_pins_p23_sha_and_target_blind_reward():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["initial_policy"]["adapter_sha256"] == (
        "843953150e3481c4112a64aca0225c5738853054ce3fea1bcdff99e58e099e40"
    )
    assert prereg["training"]["reward_target_smiles_access"] is False
