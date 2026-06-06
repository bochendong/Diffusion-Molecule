import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_prepare_univideo_molscribe_csv_smoke(tmp_path):
    predictions = tmp_path / "predictions.csv"
    eval_jsonl = tmp_path / "eval.jsonl"
    images = tmp_path / "images"
    output = tmp_path / "image_path.csv"
    images.mkdir()
    (images / "generated_00000.png").write_bytes(b"path smoke")

    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "condition_id", "source_smiles", "target_smiles", "property_count", "latent_mse"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "edit:test:c1",
                "condition_id": "c1",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "property_count": "2",
                "latent_mse": "0.1",
            }
        )

    eval_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "edit:test:c1",
                "task_type": "edit_generation",
                "split": "eval",
                "prompt": "edit",
                "target_smiles": "CCN",
                "source_smiles": "CCO",
                "instruction": "edit",
                "condition_properties": "MW,QED",
                "property_count": "2",
                "source_tanimoto": "0.5",
                "source_properties": {"MW": 46, "QED": 0.4},
                "target_properties": {"MW": 45, "QED": 0.5},
                "property_deltas": {"MW": -1, "QED": 0.1},
                "active_properties": {"MW": True, "QED": True},
                "directions": {"MW": "decrease", "QED": "increase"},
                "metadata": {"condition_id": "c1", "pair_id": "p1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_univideo_molscribe_csv.py",
            "--predictions-csv",
            str(predictions),
            "--eval-jsonl",
            str(eval_jsonl),
            "--generated-images-dir",
            str(images),
            "--output-csv",
            str(output),
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["image_path"].endswith("generated_00000.png")
    assert rows[0]["condition_properties"] == "MW,QED"
    assert rows[0]["target_QED"] == "0.5"


def test_image_benchmark_summary_counts_invalid_as_strict_failure():
    module = _load_script("evaluate_univideo_image_benchmark.py")
    rows = [
        {
            "method": "m",
            "property_count": 2,
            "valid": True,
            "strict_success": True,
            "source_scaffold_match": True,
            "source_tanimoto": 0.7,
            "target_tanimoto": 0.9,
            "generated_smiles": "CCN",
            "image_path_exists": True,
            "ocr_smiles_present": True,
            "exact_target_match": False,
            "source_identity": False,
        },
        {
            "method": "m",
            "property_count": 2,
            "valid": False,
            "strict_success": False,
            "source_scaffold_match": False,
            "source_tanimoto": "",
            "target_tanimoto": "",
            "generated_smiles": "",
            "image_path_exists": True,
            "ocr_smiles_present": False,
            "exact_target_match": False,
            "source_identity": False,
        },
    ]

    summary = module._summarize(rows, thresholds=[0.6])
    by_label = {row["benchmark_label"]: row for row in summary}

    assert by_label["2_properties"]["strict_success_rate"] == 0.5
    assert by_label["2_properties"]["success_rate_strict_in_valid_mols"] == 1.0
    assert by_label["2_properties"]["validity"] == 0.5
    assert by_label["2_properties"]["strict_success_at_source_tanimoto_ge_0_6"] == 0.5


def _load_script(name: str):
    path = Path("SketchMol-Understanding-Condition/scripts") / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
