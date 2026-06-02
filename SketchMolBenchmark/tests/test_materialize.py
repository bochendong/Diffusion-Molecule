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


if __name__ == "__main__":
    unittest.main()
