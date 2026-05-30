import csv
import tempfile
import unittest
from pathlib import Path

from sketch_smiles.build_pairs import PairRecord, build_pair_manifest, summarize_pairs


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


if __name__ == "__main__":
    unittest.main()
