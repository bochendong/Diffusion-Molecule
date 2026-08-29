from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_small_raw1_gate as builder  # noqa: E402


def prompt(count: int, condition_id: str) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": json.dumps({
                    "source": "<EMPTY>",
                    "conditions": [
                        {"property": "MW", "goal": {"around": 100 + index}}
                        for index in range(count)
                    ],
                }),
            },
        ],
    }


def test_property_count_reads_structured_prompt():
    assert builder.property_count(prompt(6, "six")) == 6


def test_contract_starts_from_alignment_refresh_and_stops_after_small_gate():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    train = (HERE / "run_train.sh").read_text()
    submit = (HERE / "submit_alignment_raw1_rl.sh").read_text()
    evaluate = (HERE / "evaluate_small_raw1_gate.py").read_text()
    assert prereg["training"]["paired_optimizer_steps"] == 30
    assert prereg["screen"]["total_conditions"] == 120
    assert "alignment_refresh/model/adapter" in train
    assert "--pairs 30 --group-size 16" in train
    assert "do_sample=False" in evaluate
    assert "best_of_40" not in submit
    assert "run_small_gate.sh" in submit


def test_checkpoint_screen_only_evaluates_saved_early_steps():
    submit = (HERE / "submit_checkpoint_screens.sh").read_text()
    selector = (HERE / "select_small_gate_checkpoint.py").read_text()
    runner = (HERE / "run_small_gate.sh").read_text()
    assert "for step in 010 020" in submit
    assert '30: "rl"' in selector
    assert 'P301_EVAL_ADAPTER' in runner
    assert "RUN_FULL_BUDGET_CURVE" in selector
    assert "best_of_40" not in submit
