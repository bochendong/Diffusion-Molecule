import csv
import tempfile
import unittest
from pathlib import Path

from sketch_smiles.audit_pairs import audit_pair_manifest
from sketch_smiles.build_pairs import PairRecord, build_pair_manifest, summarize_pairs
from sketch_smiles.phase5a0_oracle_baseline import run_oracle_paired_baseline


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except Exception:
        return False


class SketchSMILESTests(unittest.TestCase):
    def test_summarize_pairs_counts_valid_records(self):
        records = [
            PairRecord(pair_id="a", input_smiles="CCO", canonical_smiles="CCO", valid=True, image_path=""),
            PairRecord(pair_id="b", input_smiles="bad", canonical_smiles="", valid=False, image_path=""),
        ]
        summary = summarize_pairs(records)
        self.assertEqual(summary["molecules"], 2.0)
        self.assertEqual(summary["valid_smiles"], 1.0)
        self.assertEqual(summary["valid_fraction"], 0.5)

    @unittest.skipUnless(_rdkit_available(), "RDKit is not installed")
    def test_build_pair_manifest_writes_csv_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                writer.writerow({"smiles": "CCO"})
                writer.writerow({"smiles": "not_a_smiles"})

            output_dir = Path(tmp, "pairs")
            records = build_pair_manifest(input_csv=input_csv, output_dir=output_dir)
            self.assertEqual(len(records), 2)
            self.assertTrue(Path(output_dir, "pairs.csv").exists())
            self.assertTrue(Path(output_dir, "summary.json").exists())
            self.assertTrue(Path(records[0].image_path).exists())
            self.assertTrue(records[0].valid)
            self.assertFalse(records[1].valid)

    def test_audit_pair_manifest_writes_summary_without_optional_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair_dir = Path(tmp, "pairs")
            pair_dir.mkdir()
            pairs_csv = pair_dir / "pairs.csv"
            with pairs_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pair_id", "input_smiles", "canonical_smiles", "valid", "image_path", "error"])
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "mol_000000",
                        "input_smiles": "CCO",
                        "canonical_smiles": "CCO",
                        "valid": "True",
                        "image_path": "missing.png",
                        "error": "",
                    }
                )

            summary = audit_pair_manifest(pair_dir=pair_dir, sample_count=1)
            self.assertEqual(summary["pairs"], 1.0)
            self.assertEqual(summary["image_exists"], 0.0)
            self.assertTrue(Path(pair_dir, "audit_summary.json").exists())
            self.assertTrue(Path(pair_dir, "audit_rows.csv").exists())
            self.assertTrue(Path(pair_dir, "sample_pairs.csv").exists())

    @unittest.skipUnless(_rdkit_available(), "RDKit is not installed")
    def test_oracle_paired_baseline_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                writer.writerow({"smiles": "CCO"})
                writer.writerow({"smiles": "CCN"})
                writer.writerow({"smiles": "CCC"})
                writer.writerow({"smiles": "COC"})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=128)
            run_dir = Path(tmp, "run")
            metrics = run_oracle_paired_baseline(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.5,
                seed=3,
                image_size=128,
                sample_count=2,
                contact_sheet_cols=2,
                contact_thumb_size=96,
            )
            self.assertEqual(metrics["pairs"], 4.0)
            self.assertEqual(metrics["eval_pairs"], 2.0)
            self.assertEqual(metrics["smiles_valid_fraction"], 1.0)
            self.assertEqual(metrics["paired_output_success_fraction"], 1.0)
            self.assertEqual(metrics["image_compared_fraction"], 1.0)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "run_config.json").exists())
            self.assertTrue(Path(run_dir, "train_pairs.csv").exists())
            self.assertTrue(Path(run_dir, "eval_pairs.csv").exists())
            self.assertTrue(Path(run_dir, "oracle_predictions.csv").exists())
            self.assertTrue(Path(run_dir, "sample_predictions.csv").exists())
            self.assertTrue(Path(run_dir, "sample_contact_sheet.png").exists())


if __name__ == "__main__":
    unittest.main()
