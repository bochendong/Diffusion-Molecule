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


def test_submission_pins_corrected_gate_and_long_full_walltime():
    text = (HERE / "submit_train.sh").read_text()
    assert "max_steps=500" in text and "accumulation=26" in text
    assert 'walltime="3-00:00:00"' in text
    assert "P24_GRADIENT_ACCUMULATION=\"$accumulation\"" in text
    assert 'output="$OUTPUT_ROOT/gate_13k"' in (HERE / "run_train.sh").read_text()


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


def test_collector_rejects_missing_cells(tmp_path: Path):
    collector = load("collect_table1")
    try:
        collector.cells(tmp_path / "missing.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing summary must fail closed")
