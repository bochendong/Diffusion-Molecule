import csv
import json
import tempfile
import unittest
from pathlib import Path

from sketchmol_benchmark.materialize import materialize


class MaterializeTests(unittest.TestCase):
    def test_materializes_real_sketchmol_ocr_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "mol.png"
            image.write_bytes(b"fake-png")
            source_csv = root / "image_path.csv"
            with source_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "SMILES",
                        "molscribe_score",
                        "MolWt_setting",
                        "MolWt_None",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(image),
                        "SMILES": "CCO",
                        "molscribe_score": "0.95",
                        "MolWt_setting": "46.07",
                        "MolWt_None": "False",
                    }
                )

            output_dir = root / "SketchMolBenchmark" / "outputs" / "current"
            metrics = materialize(
                source_csv=source_csv,
                output_dir=output_dir,
                benchmark_name="real_sketchmol_smoke",
                sketchmol_repo=root / "SketchMol-v1-main",
            )

            self.assertEqual(metrics["rows"], 1)
            summary = metrics["summary"]
            self.assertEqual(summary["benchmark_task"], "sketchmol_plus_ocr")
            self.assertEqual(summary["benchmark_label"], "real_sketchmol_smoke")
            self.assertEqual(summary["image_path_exists_fraction"], 1.0)
            self.assertEqual(summary["ocr_smiles_present_rate"], 1.0)
            self.assertAlmostEqual(summary["molscribe_score_mean"], 0.95)
            self.assertTrue((output_dir / "benchmark_summary.csv").exists())
            self.assertTrue((output_dir / "benchmark_decoded.csv").exists())
            self.assertTrue((output_dir / "benchmark_report.md").exists())

            manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_name"], "real_sketchmol_smoke")
            self.assertEqual(manifest["benchmark_kind"], "real_sketchmol_plus_ocr")

    def test_materializes_direct_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sketch.png"
            image.write_bytes(b"fake-png")
            source_csv = root / "direct_predictions.csv"
            with source_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "generated_smiles",
                        "MolWt_setting",
                        "MolWt_None",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(image),
                        "generated_smiles": "CCO",
                        "MolWt_setting": "46.07",
                        "MolWt_None": "False",
                    }
                )

            output_dir = root / "SketchMolBenchmark" / "outputs" / "direct_structure_current"
            metrics = materialize(
                source_csv=source_csv,
                output_dir=output_dir,
                benchmark_name="direct_smoke",
                benchmark_kind="direct_structure_prediction",
                benchmark_task="direct_structure_prediction",
                smiles_column="generated_smiles",
                present_rate_field="predicted_smiles_present_rate",
                present_detail_field="predicted_smiles_present",
                source_copy_name="source_direct_predictions.csv",
                report_title="Direct Structure Prediction Benchmark",
                report_description="Direct decoder smoke.",
                present_label="predicted SMILES present",
            )

            self.assertEqual(metrics["benchmark_kind"], "direct_structure_prediction")
            self.assertEqual(metrics["rows"], 1)
            summary = metrics["summary"]
            self.assertEqual(summary["benchmark_task"], "direct_structure_prediction")
            self.assertEqual(summary["benchmark_label"], "direct_smoke")
            self.assertEqual(summary["image_path_exists_fraction"], 1.0)
            self.assertEqual(summary["predicted_smiles_present_rate"], 1.0)
            self.assertTrue((output_dir / "benchmark_summary.csv").exists())
            self.assertTrue((output_dir / "benchmark_decoded.csv").exists())
            self.assertTrue((output_dir / "source_direct_predictions.csv").exists())

            manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_kind"], "direct_structure_prediction")
            self.assertEqual(manifest["smiles_column"], "generated_smiles")


if __name__ == "__main__":
    unittest.main()
