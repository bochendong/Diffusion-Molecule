from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "unified_latent_table1" / "summarize_d3_table1_aggregations.py"


def load_module():
    spec = importlib.util.spec_from_file_location("d3_aggregation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_reports_macro_and_candidate_weighted_micro():
    module = load_module()
    rows = [
        {
            "n": 20,
            "valid_n": 20,
            "selection": "candidate-level@n=20",
            "Validity": 1.0,
            "Acc_all(0.65)": 0.2,
            "Acc_valid(0.65)": 0.2,
            "Acc_all(0.15)": 0.4,
            "Acc_valid(0.15)": 0.4,
        },
        {
            "n": 40,
            "valid_n": 20,
            "selection": "candidate-level@n=20",
            "Validity": 0.5,
            "Acc_all(0.65)": 0.5,
            "Acc_valid(0.65)": 1.0,
            "Acc_all(0.15)": 0.5,
            "Acc_valid(0.15)": 1.0,
        },
    ]
    result = module.aggregate(rows)
    assert result["task_count"] == 2
    assert result["row_count"] == 60
    assert result["macro"]["Acc_all(0.65)"] == 0.35
    assert result["micro"]["Acc_all(0.65)"] == 0.4


def test_cli_writes_fair_comparison_gate(tmp_path, monkeypatch):
    module = load_module()
    candidate = tmp_path / "candidate.json"
    anyk = tmp_path / "any.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    base = {
        "n": 20,
        "valid_n": 20,
        "selection": "candidate-level@n=20",
        "Validity": 1.0,
        "Acc_all(0.65)": 0.46,
        "Acc_valid(0.65)": 0.46,
        "Acc_all(0.15)": 0.73,
        "Acc_valid(0.15)": 0.73,
    }
    candidate.write_text(json.dumps([base]), encoding="utf-8")
    any_row = dict(base, selection="any@20", **{"Acc_all(0.65)": 0.8, "Acc_all(0.15)": 0.9})
    anyk.write_text(json.dumps([any_row]), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--candidate-json",
            str(candidate),
            "--any-json",
            str(anyk),
            "--output-json",
            str(output),
            "--output-markdown",
            str(markdown),
            "--model-name",
            "D3 test",
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["fair_comparison_gate"]["strict_beats_moleditrl"] is True
    assert "Any@20" in markdown.read_text(encoding="utf-8")
