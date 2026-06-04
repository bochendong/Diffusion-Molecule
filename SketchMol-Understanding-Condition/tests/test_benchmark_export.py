import csv
from pathlib import Path

import numpy as np

from sketchmol_understanding_condition import benchmark_export
from sketchmol_understanding_condition.benchmark_export import (
    BenchmarkExportConfig,
    export_ridge_benchmark_predictions,
)
import sketchmol_understanding_condition.retrieval_data as retrieval_data


def test_export_ridge_benchmark_predictions_writes_direct_rows(tmp_path, monkeypatch):
    def fake_fp(smiles, n_bits=4, **_kwargs):
        table = {
            "CCO": [1.0, 0.0, 0.0, 0.0],
            "CCC": [0.0, 1.0, 0.0, 0.0],
            "CCN": [1.0, 0.0, 0.0, 0.0],
        }
        return table.get(smiles, [0.0] * n_bits)

    def fake_props(_smiles):
        return {"MolWt": 46.07, "LogP": 1.0, "QED": 0.5, "TPSA": 20.0, "HBD": 1.0, "HBA": 1.0, "rotatable": 0.0}

    monkeypatch.setattr(benchmark_export, "canonical_smiles", lambda smiles: smiles)
    monkeypatch.setattr(benchmark_export, "morgan_fingerprint_bits", fake_fp)
    monkeypatch.setattr(benchmark_export, "molecular_properties", fake_props)
    monkeypatch.setattr(benchmark_export, "morgan_tanimoto", lambda a, b: 1.0 if a == b else 0.25)
    monkeypatch.setattr(benchmark_export, "scaffold_smiles", lambda smiles: smiles[:2] if smiles else "")
    monkeypatch.setattr(retrieval_data, "morgan_fingerprint_bits", fake_fp)
    monkeypatch.setattr(retrieval_data, "molecular_properties", fake_props)

    source_image = tmp_path / "source.png"
    target_image = tmp_path / "target.png"
    source_image.write_bytes(b"fake")
    target_image.write_bytes(b"fake")
    rows = [
        _row("train_1", "train", "CCN", "CCO", source_image, target_image),
        _row("train_2", "train", "CCO", "CCC", source_image, target_image),
        _row("eval_1", "eval", "CCN", "CCO", source_image, target_image),
    ]
    baseline_csv = tmp_path / "baseline_variants.csv"
    with baseline_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_csv = tmp_path / "benchmark_predictions.csv"
    summary = export_ridge_benchmark_predictions(
        baseline_variants_csv=baseline_csv,
        output_csv=output_csv,
        config=BenchmarkExportConfig(variant="full", fingerprint_bits=4),
    )

    assert summary["eval_rows"] == 1
    exported = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert exported[0]["generated_smiles"] == "CCO"
    assert exported[0]["image_path"] == str(source_image)
    assert exported[0]["MolWt_None"] == "False"
    assert exported[0]["MolWt_setting"] != ""

    feature_dir = tmp_path / "condition_features"
    feature_dir.mkdir()
    np.save(
        feature_dir / "pooled.npy",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    with (feature_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id"])
        writer.writeheader()
        writer.writerows({"variant_id": row["variant_id"]} for row in rows)

    feature_output_csv = tmp_path / "feature_benchmark_predictions.csv"
    feature_summary = export_ridge_benchmark_predictions(
        baseline_variants_csv=baseline_csv,
        output_csv=feature_output_csv,
        config=BenchmarkExportConfig(variant="full", fingerprint_bits=4, condition_features_dir=feature_dir),
    )

    assert feature_summary["eval_rows"] == 1
    assert feature_summary["condition_features_dir"] == str(feature_dir)
    assert list(csv.DictReader(feature_output_csv.open(newline="", encoding="utf-8")))


def _row(pair_id: str, split: str, source: str, target: str, source_image: Path, target_image: Path) -> dict[str, str]:
    return {
        "variant_id": f"{pair_id}:full",
        "pair_id": pair_id,
        "split": split,
        "variant": "full",
        "condition_mode": "mllm_image_text",
        "source_image": str(source_image),
        "target_image": str(target_image),
        "source_smiles": source,
        "target_smiles": target,
        "instruction": "increase molecular weight",
        "prompt": "increase molecular weight",
        "property_name": "MolWt",
        "property_delta": "1.0",
        "objective": "MolWt",
        "direction": "increase",
    }
