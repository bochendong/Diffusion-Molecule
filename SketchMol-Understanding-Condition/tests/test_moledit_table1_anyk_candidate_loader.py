from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_moledit_table1_anyk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("moledit_anyk", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_loader_preserves_empty_raw_output(tmp_path):
    module = load_module()
    path = tmp_path / "candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "candidate_index", "generated_smiles"])
        writer.writeheader()
        writer.writerow({"condition_id": "row-1", "candidate_index": 1, "generated_smiles": "CC"})
        writer.writerow({"condition_id": "row-1", "candidate_index": 2, "generated_smiles": ""})
    loaded = module.load_candidates(path, method_filter=None, candidate_limit=20)
    assert loaded["row-1"] == ["CC", ""]
