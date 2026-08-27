from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_mode_paired_grpo as trainer  # noqa: E402


def row(mode: str, bucket: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        count = int(bucket[0])
        conditions = [
            {"property": prop, "goal": {"around": float(index + offset)}}
            for offset, prop in enumerate(trainer.p25.protocol.PROPERTIES[:count])
        ]
        source = "<EMPTY>"
        task_key = bucket
        plan = "BUILD"
    else:
        conditions = [
            {"property": part.split(":")[0], "goal": part.split(":")[1]}
            for part in bucket.split("+")
        ]
        source = "CCO"
        task_key = bucket
        plan = "MODIFY"
    return {
        "example_id": f"{mode}:{bucket}:{index}",
        "task_mode": mode,
        "task_key": task_key,
        "messages": [
            {"role": "system", "content": trainer.p25.protocol.SYSTEM},
            {"role": "user", "content": json.dumps({"conditions": conditions, "source": source})},
            {"role": "assistant", "content": json.dumps({"plan": plan, "smiles": "CCN"}, separators=(",", ":"))},
        ],
    }


def test_schedule_pairs_modes_one_to_one_and_balances_target_buckets():
    rows = []
    for count in (5, 6, 7):
        rows.extend(row("de_novo", f"{count}p", index) for index in range(15))
    for task in sorted(trainer.p25.TARGET_EDIT_TASKS):
        rows.extend(row("edit", task, index) for index in range(5))
    schedule = trainer.select_mode_pairs(rows, pairs=30, seed=25125)
    assert len(schedule) == 30
    assert all(a["task_mode"] == "de_novo" and b["task_mode"] == "edit" for a, b in schedule)
    assert Counter(trainer.p25.target_bucket(a) for a, _ in schedule) == {
        "de_novo:5p": 10, "de_novo:6p": 10, "de_novo:7p": 10,
    }
    assert set(Counter(trainer.p25.target_bucket(b) for _, b in schedule).values()) == {3}


def test_preregistration_requires_real_reference_kl_and_two_gates():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["training"]["reference_kl_weight"] == 0.05
    assert prereg["training"]["groups_per_step"] == {"de_novo": 1, "edit": 1}
    assert prereg["evaluation"]["dev_and_final_disjoint"] is True
    assert prereg["evaluation"]["dev_gate_must_promote_before_final_gate_is_run"] is True


def test_runner_uses_group8_and_both_assay_oracles():
    runner = (HERE / "run_train.sh").read_text()
    assert "--group-size 8" in runner
    assert "--reference-kl-weight 0.05" in runner
    assert "SUCC_GSK3B_ORACLE_PATH" in runner
    assert "SUCC_DRD2_ORACLE_PATH" in runner
