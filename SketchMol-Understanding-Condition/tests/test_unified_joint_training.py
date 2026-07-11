from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_smiles_generator"
    / "unified_smiles_generator.py"
)
PREPARE_PATH = UNIFIED_PATH.with_name("prepare_unified_joint_rows.py")
COLLECT_PATH = UNIFIED_PATH.with_name("collect_unified_joint_v2_results.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


unified = load_module(UNIFIED_PATH, "unified_joint_training_module")
prepare = load_module(PREPARE_PATH, "prepare_unified_joint_rows_module")
collect = load_module(COLLECT_PATH, "collect_unified_joint_v2_results_module")


def dataset_item(mode: str, group: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        row = {"property_count": group, "target_smiles": "CCO"}
    else:
        row = {
            "instruction_tasks": f'[{ {"property": group, "direction": "increase"} }]'.replace("'", '"'),
            "source_smiles": "CC",
            "target_smiles": "CCC",
        }
    return {"task_mode": mode, "row": row, "index": index}


def test_task_balanced_order_balances_modes_and_groups() -> None:
    dataset = []
    dataset.extend(dataset_item("de_novo", "2", idx) for idx in range(8))
    dataset.extend(dataset_item("de_novo", "3", 20 + idx) for idx in range(2))
    dataset.extend(dataset_item("edit", "MW", 40 + idx) for idx in range(3))
    dataset.extend(dataset_item("edit", "QED", 60 + idx) for idx in range(1))

    order = unified.build_epoch_order(
        dataset,
        sampling_mode="task_balanced",
        samples_per_epoch=40,
        seed=7,
    )
    selected = [dataset[index] for index in order]
    mode_counts = unified.task_mode_counts(selected)
    group_counts = unified.training_group_counts(selected)

    assert mode_counts == {"de_novo": 20, "edit": 20}
    assert group_counts == {
        "de_novo:2p": 10,
        "de_novo:3p": 10,
        "edit:MW:+1": 10,
        "edit:QED:+1": 10,
    }


def test_de_novo_distillation_masks_edit_rows() -> None:
    config = {
        "vocab_size": 7,
        "condition_dim": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "pad_id": 0,
        "max_length": 16,
    }
    student = unified.ConditionedSmilesDecoder(**config)
    teacher = unified.ConditionedSmilesDecoder(**config)
    teacher.load_state_dict(student.state_dict())
    rows = [
        {
            "condition": np.ones((2, 4), dtype=np.float32),
            "decoder_input_ids": np.asarray([1, 4], dtype=np.int64),
            "target_ids": np.asarray([4, 2], dtype=np.int64),
            "task_mode": "de_novo",
        },
        {
            "condition": np.ones((2, 4), dtype=np.float32),
            "decoder_input_ids": np.asarray([1, 5], dtype=np.int64),
            "target_ids": np.asarray([5, 2], dtype=np.int64),
            "task_mode": "edit",
        },
    ]
    batch = unified.collate_batch(rows, pad_id=0)
    student_logits = student(
        batch["condition"],
        batch["decoder_input_ids"],
        condition_mask=batch["condition_mask"],
    )
    loss, token_count = unified.de_novo_distillation_loss(
        student_logits,
        teacher,
        batch,
        pad_id=0,
        temperature=1.0,
    )
    assert token_count == 2
    assert float(loss.item()) < 1e-6


def test_prepare_rows_drops_train_eval_overlap() -> None:
    eval_row = {
        "sample_id": "eval-1",
        "task_mode": "de_novo",
        "property_count": "2",
        "target_smiles": "CCO",
    }
    train_rows = [
        {
            "sample_id": "train-overlap",
            "task_mode": "de_novo",
            "property_count": "2",
            "target_smiles": "CCO",
        },
        {
            "sample_id": "train-clean",
            "task_mode": "de_novo",
            "property_count": "2",
            "target_smiles": "CCN",
        },
    ]
    kept, dropped = prepare.remove_train_eval_overlap(train_rows, [eval_row], policy="drop_train")
    assert [row["sample_id"] for row in kept] == ["train-clean"]
    assert len(dropped) == 1


def test_collector_reads_stage_metadata() -> None:
    stage_root = Path("/tmp/eval/u2")
    path = stage_root / "table1" / "at128" / "raw" / "moledit_table1" / "n128" / "metrics" / "moledit_table_summary.csv"
    assert collect.metadata(path, stage_root) == {
        "task": "table1",
        "budget": "128",
        "selection": "raw",
        "source_summary": str(path),
    }
