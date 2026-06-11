import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


RDKit_MISSING = importlib.util.find_spec("rdkit") is None


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


def test_export_univideo_benchmark_rows_without_images(tmp_path):
    predictions = tmp_path / "predictions.csv"
    eval_jsonl = tmp_path / "eval.jsonl"
    output = tmp_path / "benchmark_condition_rows.csv"

    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "condition_id", "source_smiles", "target_smiles", "latent_mse"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "edit:moledit_instruct:m1",
                "condition_id": "m1",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "latent_mse": "0.1",
            }
        )

    eval_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "edit:moledit_instruct:m1",
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
                "metadata": {"condition_id": "m1", "pair_hash": "p1", "moledit_task_key": "MW:decrease+QED:increase"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_univideo_benchmark_rows.py",
            "--predictions-csv",
            str(predictions),
            "--eval-jsonl",
            str(eval_jsonl),
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
    assert rows[0]["condition_id"] == "m1"
    assert rows[0]["source_smiles"] == "CCO"
    assert rows[0]["condition_properties"] == "MW,QED"
    assert rows[0]["target_QED"] == "0.5"
    assert rows[0]["moledit_task_key"] == "MW:decrease+QED:increase"


def test_export_denovo_2p7p_benchmark_rows_and_property_nearest(tmp_path):
    molecule_db = tmp_path / "molecule_database.csv"
    rows_csv = tmp_path / "denovo_2p7p_rows.csv"
    candidates_csv = tmp_path / "denovo_candidate_rows.csv"
    direct_csv = tmp_path / "direct.csv"
    _write_molecule_database(molecule_db, count=8)

    export_result = subprocess.run(
        [
            sys.executable,
            "scripts/export_denovo_2p7p_benchmark_rows.py",
            "--molecule-db-csv",
            str(molecule_db),
            "--output-csv",
            str(rows_csv),
            "--candidate-output-csv",
            str(candidates_csv),
            "--rows-per-property-count",
            "2",
            "--min-properties",
            "2",
            "--max-properties",
            "3",
            "--seed",
            "7",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert export_result.returncode == 0, export_result.stderr
    rows = list(csv.DictReader(rows_csv.open(newline="", encoding="utf-8")))
    candidates = list(csv.DictReader(candidates_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 4
    assert len(candidates) == 4
    assert sorted({row["property_count"] for row in rows}) == ["2", "3"]
    assert all(row["source_smiles"] == "" for row in rows)
    assert all(row["task_type"] == "de_novo_design" for row in rows)
    assert all(row["sketchmol_preset_str"] for row in rows)
    assert all(len(row["condition_properties"].split(",")) == int(row["property_count"]) for row in rows)
    assert not ({row["molecule_id"] for row in rows} & {row["molecule_id"] for row in candidates})

    materialize_result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_univideo_target_molecules.py",
            "--source-csv",
            str(rows_csv),
            "--candidate-csv",
            str(candidates_csv),
            "--output-csv",
            str(direct_csv),
            "--mode",
            "property_nearest",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert materialize_result.returncode == 0, materialize_result.stderr
    direct_rows = list(csv.DictReader(direct_csv.open(newline="", encoding="utf-8")))
    assert len(direct_rows) == 4
    assert {row["method"] for row in direct_rows} == {"property_nearest"}
    assert all(row["generated_smiles"] for row in direct_rows)


@pytest.mark.skipif(RDKit_MISSING, reason="RDKit is required for SMILES validation")
def test_decode_row_ignores_raw_token_fallback_for_validity():
    module = _load_script("evaluate_univideo_image_benchmark.py")
    row = {
        "SMILES": "CCO",
        "molscribe_decode_source": "raw_token_fallback",
        "source_smiles": "CCN",
        "target_smiles": "CCO",
        "condition_properties": "MW",
        "property_count": "1",
        "target_MW": "46",
        "MW_active": "True",
    }
    decoded = module._decode_row(row, method="m", smiles_column="SMILES")
    assert decoded["valid"] is False
    assert decoded["ocr_smiles_present"] is False
    assert decoded["raw_fallback_valid"] is True
    assert decoded["graph_decoded"] is False


@pytest.mark.skipif(RDKit_MISSING, reason="RDKit is required for SMILES validation")
def test_decode_row_accepts_sketchmol_graph_decode():
    module = _load_script("evaluate_univideo_image_benchmark.py")
    row = {
        "SMILES": "CCO",
        "molscribe_decode_source": "sketchmol_graph",
        "source_smiles": "CCN",
        "target_smiles": "CCO",
        "condition_properties": "MW",
        "property_count": "1",
        "target_MW": "46",
        "MW_active": "True",
    }
    decoded = module._decode_row(row, method="m", smiles_column="SMILES")
    assert decoded["valid"] is True
    assert decoded["ocr_smiles_present"] is True
    assert decoded["graph_decoded"] is True


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


def test_materialize_univideo_target_molecules_target_oracle(tmp_path):
    source = tmp_path / "image_path.csv"
    output = tmp_path / "direct.csv"
    _write_simple_target_rows(
        source,
        [
            {
                "sample_id": "s0",
                "condition_id": "c0",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "condition_properties": "MW",
                "target_MW": "45",
                "MW_active": "True",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_univideo_target_molecules.py",
            "--source-csv",
            str(source),
            "--output-csv",
            str(output),
            "--mode",
            "target_oracle",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert rows[0]["generated_smiles"] == "CCN"
    assert rows[0]["target_finder_mode"] == "target_oracle"
    assert rows[0]["exact_target_match"] == "True"


def test_materialize_univideo_target_molecules_multiple_methods(tmp_path):
    source = tmp_path / "image_path.csv"
    output = tmp_path / "direct.csv"
    _write_simple_target_rows(
        source,
        [
            {
                "sample_id": "s0",
                "condition_id": "c0",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "condition_properties": "MW",
                "target_MW": "45",
                "MW_active": "True",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_univideo_target_molecules.py",
            "--source-csv",
            str(source),
            "--output-csv",
            str(output),
            "--methods",
            "source_identity,target_oracle",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    by_method = {row["method"]: row for row in rows}
    assert sorted(by_method) == ["source_identity", "target_oracle"]
    assert by_method["source_identity"]["generated_smiles"] == "CCO"
    assert by_method["target_oracle"]["generated_smiles"] == "CCN"
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["methods"] == ["source_identity", "target_oracle"]
    assert summary["rows"] == 2


def test_materialize_univideo_target_molecules_latent_nearest(tmp_path):
    source = tmp_path / "image_path.csv"
    output = tmp_path / "direct.csv"
    generated = tmp_path / "generated_latents.npy"
    targets = tmp_path / "target_latents.npy"
    _write_simple_target_rows(
        source,
        [
            {"sample_id": "s0", "condition_id": "c0", "source_smiles": "CCO", "target_smiles": "CCN"},
            {"sample_id": "s1", "condition_id": "c1", "source_smiles": "CCC", "target_smiles": "CCCl"},
        ],
    )
    np.save(generated, np.asarray([[0.05, 1.0], [1.0, 0.05]], dtype=np.float32))
    np.save(targets, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_univideo_target_molecules.py",
            "--source-csv",
            str(source),
            "--output-csv",
            str(output),
            "--mode",
            "latent_nearest",
            "--generated-latents-npy",
            str(generated),
            "--candidate-latents-npy",
            str(targets),
            "--top-k",
            "2",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert rows[0]["generated_smiles"] == "CCCl"
    assert rows[0]["matched_condition_id"] == "c1"
    assert rows[1]["generated_smiles"] == "CCN"
    assert rows[1]["matched_condition_id"] == "c0"


def test_materialize_univideo_target_molecules_property_nearest(tmp_path):
    source = tmp_path / "image_path.csv"
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "direct.csv"
    _write_simple_target_rows(
        source,
        [
            {
                "sample_id": "q0",
                "condition_id": "q0",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "condition_properties": "MW,LogP",
                "target_MW": "100",
                "target_LogP": "2.0",
                "MW_active": "True",
                "LogP_active": "True",
            }
        ],
    )
    _write_simple_target_rows(
        candidates,
        [
            {
                "sample_id": "k0",
                "condition_id": "k0",
                "target_smiles": "CCCC",
                "target_MW": "150",
                "target_LogP": "3.0",
            },
            {
                "sample_id": "k1",
                "condition_id": "k1",
                "target_smiles": "CCCO",
                "target_MW": "101",
                "target_LogP": "2.1",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_univideo_target_molecules.py",
            "--source-csv",
            str(source),
            "--candidate-csv",
            str(candidates),
            "--output-csv",
            str(output),
            "--mode",
            "property_nearest",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert rows[0]["generated_smiles"] == "CCCO"
    assert rows[0]["matched_condition_id"] == "k1"


def _write_simple_target_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "condition_id",
        "pair_id",
        "image_path",
        "source_smiles",
        "target_smiles",
        "condition_properties",
        "target_MW",
        "target_LogP",
        "target_QED",
        "target_TPSA",
        "target_HBD",
        "target_HBA",
        "target_RB",
        "MW_active",
        "LogP_active",
        "QED_active",
        "TPSA_active",
        "HBD_active",
        "HBA_active",
        "RB_active",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_molecule_database(path: Path, *, count: int) -> None:
    fieldnames = ["mol_id", "canonical_smiles", "scaffold", "MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "mol_id": f"mol_{index}",
                    "canonical_smiles": "C" * (index + 2),
                    "scaffold": "",
                    "MW": 80 + index * 20,
                    "LogP": 0.5 + index * 0.2,
                    "QED": 0.3 + index * 0.03,
                    "TPSA": 20 + index * 5,
                    "HBD": index % 3,
                    "HBA": 1 + (index % 4),
                    "RB": index % 5,
                }
            )


def _load_script(name: str):
    path = Path("SketchMol-Understanding-Condition/scripts") / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
