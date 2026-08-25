from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module("p19_builder", ROOT / "build_expanded_subsets.py")
aggregate = load_module("p19_aggregate", ROOT / "aggregate_expanded.py")


def test_preregistration_is_frozen_test_only():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    assert prereg["created_before_expanded_generation"] is True
    assert prereg["training_or_parameter_updates"] is False
    assert prereg["benchmark_tuning_allowed"] is False
    assert prereg["subset"]["table1"]["total_rows"] == 100
    assert prereg["subset"]["hard_denovo"]["total_rows"] == 40
    uncertainty = prereg["reporting"]["table1_uncertainty"]
    assert uncertainty["paired_condition_bootstrap_seed"] == 1919
    assert uncertainty["paired_condition_bootstrap_replicates"] == 10000


def test_mandatory_rows_are_retained_and_ranked():
    pool = [{"condition_id": f"r{i}"} for i in range(12)]
    mandatory = [pool[8], pool[11]]
    chosen = builder.choose_with_mandatory(pool, mandatory, 10, 1717)
    assert len(chosen) == 10
    assert {"r8", "r11"} <= {row["condition_id"] for row in chosen}
    assert chosen == sorted(chosen, key=lambda row: builder.stable_rank(row, 1717))


def test_wilson_interval_and_bootstrap_are_deterministic():
    interval = aggregate.wilson(0, 10)
    assert interval["n"] == 10
    assert interval["low"] == 0.0
    assert 0.27 < interval["high"] < 0.29
    left = {
        "a": {"task": "t", "validity": 0.0, "property_anyk": 0.0, "acc_0.15": 0.0, "strict_acc_0.65": 0.0, "mean_best_source_similarity": 0.1},
        "b": {"task": "t", "validity": 1.0, "property_anyk": 1.0, "acc_0.15": 1.0, "strict_acc_0.65": 1.0, "mean_best_source_similarity": 0.2},
    }
    right = {
        "a": {"task": "t", "validity": 1.0, "property_anyk": 1.0, "acc_0.15": 1.0, "strict_acc_0.65": 1.0, "mean_best_source_similarity": 0.2},
        "b": {"task": "t", "validity": 1.0, "property_anyk": 1.0, "acc_0.15": 1.0, "strict_acc_0.65": 1.0, "mean_best_source_similarity": 0.2},
    }
    first = aggregate.paired_bootstrap(left, right, seed=1919, replicates=100)
    second = aggregate.paired_bootstrap(left, right, seed=1919, replicates=100)
    assert first == second
    assert first["strict_acc_0.65"]["delta"] == 0.5
