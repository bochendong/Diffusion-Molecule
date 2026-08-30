from __future__ import annotations

import json
from pathlib import Path

from collect_pilot import summarize
from prepare_pilot import select_train


HERE = Path(__file__).resolve().parent


def row(mode, identity, count=2, task_key=""):
    return {
        "example_id": identity,
        "task_mode": mode,
        "condition_program": [{} for _ in range(count)],
        "task_key": task_key,
        "messages": [{"role": "user", "content": "{}"}, {"role": "assistant", "content": "x"}],
    }


def test_exact_matched_training_subsets():
    from prepare_pilot import EDIT_TASKS

    rows = []
    for count in range(2, 8):
        rows.extend(row("de_novo", f"d-{count}-{i}", count=count) for i in range(500))
    for task in EDIT_TASKS:
        rows.extend(row("edit", f"e-{task}-{i}", count=1, task_key=task) for i in range(300))
    de_novo, editing, joint, quotas = select_train(rows, 33001)
    assert len(de_novo) == len(editing) == 3000
    assert len(joint) == 6000
    assert {item["example_id"] for item in de_novo}.issubset(
        {item["example_id"] for item in joint}
    )
    assert {item["example_id"] for item in editing}.issubset(
        {item["example_id"] for item in joint}
    )
    assert set(quotas.values()) == {300, 500}


def test_collector_decisions_and_parameter_efficiency():
    def evaluation(de=0.50, edit=0.60, de_valid=0.95, edit_valid=0.94):
        return {"aggregate": {
            "denovo_strict_macro": de,
            "denovo_valid_macro": de_valid,
            "edit_strict_065_macro": edit,
            "edit_valid_macro": edit_valid,
            "edit_relaxed_015_macro": 0.80,
        }}

    evals = {
        "joint": evaluation(de=0.53, edit=0.60),
        "denovo": evaluation(de=0.50),
        "edit": evaluation(edit=0.60),
    }
    trains = {arm: {"trainable_parameters": 100} for arm in evals}
    result = summarize(evals, trains)
    assert result["decision"] == "SUPPORT_UNIFIED_POSITIVE_TRANSFER_PILOT"
    assert result["efficiency"]["joint_over_separate_adapter_parameter_ratio"] == 0.5
    evals["joint"] = evaluation(de=0.50, edit=0.55)
    assert summarize(evals, trains)["decision"] == "ASYMMETRIC_INTERFERENCE_PILOT"


def test_preregistered_clean_initialization_and_submission():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    trainer = (HERE / "train_arm.py").read_text()
    submit = (HERE / "submit_p33.sh").read_text()
    assert prereg["status"] == "frozen before GPU training"
    assert prereg["common_initialization"]["prior_joint_training"] is False
    assert prereg["common_initialization"]["prior_molecular_adapter"] is None
    assert prereg["arms"]["joint"]["de_novo_rows"] == 3000
    assert prereg["arms"]["joint"]["editing_rows"] == 3000
    assert "get_peft_model" in trainer
    assert "PeftModel.from_pretrained" not in trainer
    assert "for arm in joint denovo edit" in submit
