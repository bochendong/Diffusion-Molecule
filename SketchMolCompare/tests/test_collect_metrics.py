import csv
import json
import tempfile
import unittest
from pathlib import Path

from sketchmol_compare.collect_metrics import collect_rows, write_outputs


class CollectMetricsTests(unittest.TestCase):
    def test_collects_sketchsmiles_and_sketchmol_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "SketchSMILES" / "outputs" / "runs" / "phase5a4"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "phase": "phase5a4_condition_reranked_transformer_decoder",
                        "eval_pairs": 10,
                        "train_pairs": 40,
                        "top1_exact_match_fraction": 0.5,
                        "topk_exact_match_fraction": 0.6,
                        "top1_target_tanimoto": 0.8,
                        "top1_valid_fraction": 0.9,
                    }
                ),
                encoding="utf-8",
            )

            summary = root / "PhysTabMol" / "runs" / "abc" / "tables" / "sketchmol_benchmark" / "sketchmol_benchmark_summary.csv"
            summary.parent.mkdir(parents=True)
            with summary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "benchmark_task",
                        "benchmark_label",
                        "success_rate_in_valid_mols",
                        "success_rate_sketchmol_tolerance_in_valid_mols",
                        "n",
                        "validity",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "benchmark_task": "single_property",
                        "benchmark_label": "LogP",
                        "success_rate_in_valid_mols": "0.25",
                        "success_rate_sketchmol_tolerance_in_valid_mols": "0.75",
                        "n": "2",
                        "validity": "1.0",
                    }
                )
                writer.writerow(
                    {
                        "benchmark_task": "single_property",
                        "benchmark_label": "QED",
                        "success_rate_in_valid_mols": "0.75",
                        "success_rate_sketchmol_tolerance_in_valid_mols": "1.0",
                        "n": "6",
                        "validity": "0.5",
                    }
                )

            rows = collect_rows([run_dir], [summary])
            self.assertEqual(rows[0]["family"], "sketchsmiles_ocr_free")
            overall = [row for row in rows if row["family"] == "sketchmol_aligned" and row["benchmark_task"] == "overall"][0]
            self.assertAlmostEqual(overall["success_rate_in_valid_mols"], 0.625)
            self.assertAlmostEqual(overall["success_rate_sketchmol_tolerance_in_valid_mols"], 0.9375)
            self.assertAlmostEqual(overall["validity"], 0.625)

            output_dir = root / "out"
            write_outputs(rows, output_dir)
            self.assertTrue((output_dir / "comparison_rows.csv").exists())
            self.assertTrue((output_dir / "comparison_rows.json").exists())
            self.assertTrue((output_dir / "comparison_report.md").exists())


if __name__ == "__main__":
    unittest.main()
