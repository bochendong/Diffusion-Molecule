from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path("SketchMol-Understanding-Condition/experiments/p2_validity_edit_repair")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_p2_entrypoints_exist_and_shell_contract_mentions_raw_constraints():
    for name in (
        "prepare_p2_eval_subsets.py",
        "run_p2_denovo_validity_benchmark.sh",
        "evaluate_p2_validity_edit_repair.py",
        "finalize_p2_validity_edit_repair.sh",
        "submit_p2_validity_edit_repair.sh",
        "README.md",
    ):
        assert (ROOT / name).exists()
    run_text = (ROOT / "run_p2_denovo_validity_benchmark.sh").read_text(encoding="utf-8")
    assert "--smiles-grammar-constraint" in run_text
    assert "raw_candidates_n${NUM_SAMPLES}.csv" in run_text
    assert "diagnostic_property_reranked_selected.csv" in run_text
    assert "--no-repeat-ngram-size \"$no_repeat_ngram\"" in run_text


def test_p2_subset_preparation_is_stratified_and_deterministic(tmp_path):
    module = load_module("prepare_p2_eval_subsets_test", ROOT / "prepare_p2_eval_subsets.py")
    two_p = []
    for count in range(2, 8):
        for index in range(3):
            two_p.append({"condition_id": f"p{count}_{index}", "property_count": str(count)})
    ood = []
    for bucket in ("forward_extreme", "rare_combo", "reverse_stimulation"):
        for index in range(3):
            ood.append({"condition_id": f"{bucket}_{index}", "ood_bucket": bucket})
    two_p_path = tmp_path / "two_p.csv"
    ood_path = tmp_path / "ood.csv"
    write_rows(two_p_path, two_p)
    write_rows(ood_path, ood)

    args = [
        "--two-p-seven-p-csv", str(two_p_path),
        "--ood-csv", str(ood_path),
        "--output-dir", str(tmp_path / "out"),
        "--per-property-count", "2",
        "--per-ood-bucket", "2",
        "--seed", "7",
    ]
    assert module.main(args) == 0
    first = (tmp_path / "out" / "denovo_2p7p_eval.csv").read_text(encoding="utf-8")
    assert module.main(args) == 0
    assert (tmp_path / "out" / "denovo_2p7p_eval.csv").read_text(encoding="utf-8") == first
    assert len(list(csv.DictReader((tmp_path / "out" / "denovo_2p7p_eval.csv").open()))) == 12
    assert len(list(csv.DictReader((tmp_path / "out" / "denovo_ood_eval.csv").open()))) == 6
