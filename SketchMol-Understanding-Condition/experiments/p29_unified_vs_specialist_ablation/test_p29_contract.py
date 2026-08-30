from __future__ import annotations

import csv
import json
import importlib.util
import random
import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "p29_specialist_trainer", HERE / "train_specialist_sft.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "p29_ablation_collector", HERE / "collect_ablation.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_freezes_matched_exposures():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["matched_training"]["rows_per_active_bucket"] == 81415
    assert prereg["arms"]["unified"]["construction_examples"] == 488490
    assert prereg["arms"]["unified"]["editing_examples"] == 569905
    assert prereg["arms"]["construction_specialist"]["editing_examples"] == 0
    assert prereg["arms"]["editing_specialist"]["construction_examples"] == 0
    assert prereg["matched_training"]["calibration"] is False


def test_training_runner_keeps_shared_initialization_and_exact_rows():
    runner = (HERE / "run_specialist_train.sh").read_text()
    assert "seed_2323_full24k_aligned/model/stage1_v2/adapter" in runner
    assert "train_specialist_sft.py" in runner
    assert "--rows-per-task 81415" in runner
    assert "--expected-examples \"$expected_examples\"" in runner
    assert "--gradient-accumulation 65" in runner
    assert "--learning-rate 1e-5" in runner
    assert "--seed 24003" in runner
    assert "--task-mode \"$task_mode\"" in runner


def test_specialist_sampler_reuses_joint_rows_exactly():
    trainer = load_trainer()

    class FakeGenerator:
        def manual_seed(self, seed):
            self.rng = random.Random(seed)

    def randperm(size, generator):
        values = list(range(size))
        generator.rng.shuffle(values)
        return values

    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace(
        Generator=FakeGenerator,
        randperm=randperm,
    )

    class FakeDataset:
        bucket_indices = {
            **{f"de_novo:{count}p": list(range(10 * count, 10 * count + 7)) for count in range(2, 8)},
            **{f"edit:{count}p": list(range(100 + 10 * count, 100 + 10 * count + 7)) for count in range(1, 8)},
        }

    try:
        construction = list(
            trainer.ExactSpecialistSampler(
                FakeDataset(), seed=24003, task_mode="de_novo", rows_per_task=5,
            )
        )
        editing = list(
            trainer.ExactSpecialistSampler(
                FakeDataset(), seed=24003, task_mode="edit", rows_per_task=5,
            )
        )
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    assert len(construction) == 6 * 5
    assert len(editing) == 7 * 5
    assert not set(construction) & set(editing)


def test_submission_runs_only_matched_native_evaluations():
    submit = (HERE / "submit_ablation.sh").read_text()
    assert "run_table1_generate.sh" in submit
    assert "run_table1_finalize.sh" in submit
    assert "run_table2_generate.sh" in submit
    assert "run_table2_score.sh" in submit
    assert "construction_specialist" in submit
    assert "editing_specialist" in submit
    assert "P24_EVAL_ADAPTER" in submit
    assert "P24_EVAL_REQUIRED" in submit
    assert "afterok:$unified_t1s:$unified_t2s:$construction_t1s:$editing_t2s" in submit


def test_collector_reports_raw1_macro_without_replacing_best40(tmp_path):
    collector = load_collector()
    results = tmp_path / "results"
    table = results / "table1"
    table.mkdir(parents=True)
    (table / "p24_table1.json").write_text(
        json.dumps(
            {
                "average_2p_7p": 0.9,
                "strict_success": {f"{count}p": 0.9 for count in range(2, 8)},
            }
        )
    )
    fields = [
        "setting", "property_count", "conditions", "candidate_budget",
        "validity", "strict_success_rate",
    ]
    split_counts = {
        "denovo_2p4p": range(2, 5),
        "denovo_5p": range(5, 6),
        "denovo_6p7p": range(6, 8),
    }
    for split, counts in split_counts.items():
        path = results / split / "budget_sweep_summary.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for count in counts:
                writer.writerow(
                    {
                        "setting": "raw_at_1",
                        "property_count": count,
                        "conditions": 100,
                        "candidate_budget": 1,
                        "validity": 0.99,
                        "strict_success_rate": count / 10,
                    }
                )
            writer.writerow(
                {
                    "setting": "raw_at_1",
                    "property_count": "all",
                    "conditions": len(counts) * 100,
                    "candidate_budget": 1,
                    "validity": 0.99,
                    "strict_success_rate": 0.45,
                }
            )

    metrics = collector.table1(results)
    assert metrics["de_novo_raw1_avg_2p_7p"] == 0.45
    assert metrics["de_novo_best40_avg_2p_7p"] == 0.9
