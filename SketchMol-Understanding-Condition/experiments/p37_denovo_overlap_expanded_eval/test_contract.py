from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collector = load_module("p37_collect", HERE / "collect_overlap.py")
evaluator = load_module("p37_evaluate", HERE / "evaluate_overlap_arm.py")


def test_preregistration_freezes_expanded_target_disjoint_raw1_gate():
    payload = json.loads((HERE / "preregistration.json").read_text())
    assert payload["status"] == "frozen before expanded GPU evaluation"
    assert payload["gate"]["primary_2p4p_per_arity_per_group"] == 100
    assert payload["gate"]["secondary_5p_per_group"] == 40
    assert payload["gate"]["total_conditions"] == 680
    assert payload["gate"]["exclude_prior_gate"] is True
    assert payload["gate"]["exclude_100k_training_target_hashes"] is True
    assert payload["decoding"]["candidate_budget"] == 1
    assert payload["decoding"]["property_reranking"] is False


def test_property_overlap_classification_is_shared_vocabulary_based():
    assert evaluator.group_for_task_key("MW:around=1+QED:around=1") == "shared_only"
    assert (
        evaluator.group_for_task_key("MW:around=1+TPSA:around=1")
        == "contains_denovo_only"
    )


def test_exact_mcnemar_handles_paired_discordance():
    assert collector.exact_mcnemar(6, 1) == 0.125
    assert collector.exact_mcnemar(0, 0) == 1.0


def test_submission_reuses_four_existing_adapters_without_training():
    submit = (HERE / "submit_expanded_overlap_eval.sh").read_text()
    run_eval = (HERE / "run_eval.sh").read_text()
    assert "for scale in 10000 100000" in submit
    assert "for arm in joint denovo" in submit
    assert "run_train" not in submit
    assert "p33_joint_vs_separate_10k_single_seed" in run_eval
    assert "p35_joint_vs_separate_scale_sweep" in run_eval
