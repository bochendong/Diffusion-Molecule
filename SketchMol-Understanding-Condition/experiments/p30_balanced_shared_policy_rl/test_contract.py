from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_balanced_shared_rl as trainer  # noqa: E402


def row(mode: str, bucket: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        count = int(bucket.removeprefix("de_novo:").removesuffix("p"))
        task_key = ""
        conditions = [
            {"property": "MW", "goal": {"around": 100.0 + offset}}
            for offset in range(count)
        ]
    else:
        task_key = bucket.removeprefix("edit:")
        conditions = [{"property": "MW", "goal": "increase"}]
    return {
        "example_id": f"{mode}-{bucket}-{index}",
        "task_mode": mode,
        "task_key": task_key,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps({"conditions": conditions, "source": "<EMPTY>" if mode == "de_novo" else "CC"})},
            {"role": "assistant", "content": "{}"},
        ],
    }


def test_balanced_selector_covers_all_six_and_ten_buckets_equally():
    rows = []
    for bucket in trainer.DE_NOVO_BUCKETS:
        rows.extend(row("de_novo", bucket, index) for index in range(12))
    for bucket in trainer.EDIT_BUCKETS:
        rows.extend(row("edit", bucket, index) for index in range(8))
    pairs = trainer.select_balanced_pairs(rows, 60, 30001)
    assert len(pairs) == 60
    denovo_counts = Counter(trainer.balanced_bucket(left) for left, _ in pairs)
    edit_counts = Counter(trainer.balanced_bucket(right) for _, right in pairs)
    assert set(denovo_counts) == set(trainer.DE_NOVO_BUCKETS)
    assert set(edit_counts) == set(trainer.EDIT_BUCKETS)
    assert set(denovo_counts.values()) == {10}
    assert set(edit_counts.values()) == {6}


def test_preregistration_and_runner_lock_balanced_contract():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    runner = (HERE / "run_train.sh").read_text()
    submit = (HERE / "submit_balanced_rl.sh").read_text()
    assert prereg["training"]["paired_optimizer_steps"] == 60
    assert prereg["training"]["group_size"] == 16
    assert prereg["training"]["reward_target_smiles_access"] is False
    assert "--pairs 60 --group-size 16" in runner
    assert "checkpoint-$step" in submit
    assert "010 020 030 040 050 060" in submit


def test_selector_does_not_read_final_gate():
    source = (HERE / "select_dev_checkpoint.py").read_text()
    assert "--final-comparison" not in source
    assert '"selection_uses_final_gate": False' in source
    assert "min(de_novo, edit)" in source


def test_bisector_is_common_descent_for_nonopposed_gradients():
    torch = pytest.importorskip("torch")
    first = [torch.tensor([2.0, 0.0])]
    second = [torch.tensor([0.0, 5.0])]
    merged, record = trainer.balanced_bisector_gradients(first, second)
    assert record["common_descent"] is True
    assert record["merged_dot_denovo"] > 0
    assert record["merged_dot_edit"] > 0
    assert merged[0][0] == pytest.approx(merged[0][1])
