from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_pins_corrected_gate_and_accelerated_full_contract():
    text = (HERE / "submit_train.sh").read_text()
    assert "max_steps=500" in text and "batch_size=1" in text and "accumulation=26" in text
    assert 'walltime="2-00:00:00"' in text
    assert "max_steps=16283" in text and "batch_size=5" in text and "accumulation=13" in text
    assert 16283 * 5 * 13 == 1_058_395
    assert "P24_BATCH_SIZE=\"$batch_size\"" in text
    assert "P24_GRADIENT_ACCUMULATION=\"$accumulation\"" in text
    runner = (HERE / "run_train.sh").read_text()
    assert 'output="$OUTPUT_ROOT/gate_13k"' in runner
    assert '--per-device-batch-size "$batch_size"' in runner


def test_training_summary_counts_physical_batch_in_example_budget():
    trainer = (HERE / "train_indexed_sft.py").read_text()
    assert '"per_device_batch_size": args.per_device_batch_size' in trainer
    assert "args.max_steps * args.per_device_batch_size * args.gradient_accumulation" in trainer
    assert "range(0, self.per_bucket, self.batch_size)" in trainer
    assert "for position in range(start, start + self.batch_size)" in trainer
    assert "TaskBalancedSampler(selected, args.seed, args.per_device_batch_size)" in trainer
    assert "available_per_bucket - (available_per_bucket % self.batch_size)" in trainer
    assert '"balanced_sampler_dropped_rows_per_task"' in trainer
    assert '"sampler_physical_batches_task_homogeneous": True' in trainer


def test_gate_validation_is_target_blind_and_fail_closed():
    builder = (HERE / "build_gate_eval.py").read_text()
    validator = (HERE / "validate_gate_outputs.py").read_text()
    runner = (HERE / "run_gate_validation.sh").read_text()
    assert '"messages": messages[:-1]' in builder
    assert "target_smiles" not in builder
    assert "exact_training_contract" in validator
    assert "maximum-edit-copy-rate" in validator
    assert "GATE_VALIDATION_COMPLETE" in runner


def test_table1_preregistration_and_chain_cover_every_cell():
    prereg = json.loads((HERE / "p24_table1_preregistration.json").read_text())
    assert prereg["evaluation"]["conditions"] == {
        "2p": 100, "3p": 100, "4p": 100, "5p": 100, "6p": 20, "7p": 20,
    }
    generation = (HERE / "run_table1_generate.sh").read_text()
    finalizer = (HERE / "run_table1_finalize.sh").read_text()
    for name in ("denovo_2p4p", "denovo_5p", "denovo_6p7p"):
        assert name in generation and name in finalizer
    assert "collect_table1.py" in finalizer


def test_table2_reuses_frozen_edit500_and_pins_oracles():
    prereg = json.loads((HERE / "p24_table2_preregistration.json").read_text())
    assert prereg["frozen_data"]["total_outputs"] == 5000
    assert prereg["generation"] == {
        "outputs_per_source": 1,
        "sampling_seed": 23501,
        "temperature": 0.8,
        "top_p": 0.95,
        "greedy": False,
        "property_reranking": False,
        "any_at_k": False,
        "target_access": False,
    }
    generation = (HERE / "run_table2_generate.sh").read_text()
    scoring = (HERE / "run_table2_score.sh").read_text()
    assert "table1_500.prompts.jsonl" in generation
    assert "--seed 23501" in generation
    assert "alignment_refresh/model/adapter" in generation
    assert "EXPECTED_GSK3B=cd8ee8a58" in scoring
    assert "EXPECTED_DRD2=dbc473fca" in scoring
    assert "--missing-oracle-policy fail" in scoring
    assert "--require-exact-candidate-count" in scoring


def test_collector_rejects_missing_cells(tmp_path: Path):
    collector = load("collect_table1")
    try:
        collector.cells(tmp_path / "missing.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing summary must fail closed")
