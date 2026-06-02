import csv
import contextlib
import tempfile
import unittest
from pathlib import Path

from sketch_smiles.audit_pairs import audit_pair_manifest
from sketch_smiles.build_pairs import PairRecord, build_pair_manifest, summarize_pairs
from sketch_smiles.phase5a0_oracle_baseline import run_oracle_paired_baseline
from sketch_smiles.phase5a1_learned_smiles_decoder import (
    _fingerprint_tanimoto,
    _tokenize_smiles,
    evaluate_saved_smiles_decoder,
    run_learned_smiles_decoder,
)
from sketch_smiles.phase5b_joint_decoder import run_joint_paired_decoder
from sketch_smiles.phase5_summary import summarize_runs
from sketch_smiles.phase5c_image_smiles_decoder import run_image_conditioned_smiles_decoder
from sketch_smiles.phase5a5_retrieval_diagnostic import run_retrieval_diagnostic


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@contextlib.contextmanager
def _quiet_rdkit_errors():
    try:
        from rdkit import RDLogger
    except Exception:
        yield
        return

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")


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

    def test_smiles_tokenizer_keeps_common_multi_char_tokens(self):
        tokens = _tokenize_smiles("CC(Cl)Br[NH4+]", tokenization="smiles_token")
        self.assertEqual(tokens, ["C", "C", "(", "Cl", ")", "Br", "[NH4+]"])

    def test_fingerprint_tanimoto_scores_binary_overlap(self):
        try:
            np = __import__("numpy")
        except Exception:
            self.skipTest("NumPy is not installed")
        self.assertAlmostEqual(_fingerprint_tanimoto([1, 1, 0, 0], [1, 0, 1, 0], np=np), 1.0 / 3.0)
        self.assertEqual(_fingerprint_tanimoto([1, 1, 0, 0], None, np=np), 0.0)

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
            with _quiet_rdkit_errors():
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

    def test_phase5_summary_writes_comparison_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp, "run_a")
            run_dir.mkdir()
            Path(run_dir, "metrics.json").write_text(
                '{"phase":"phase5a","top1_exact_match_fraction":0.5,"topk_exact_match_fraction":0.7}\n',
                encoding="utf-8",
            )
            output_dir = Path(tmp, "summary")
            rows = summarize_runs([run_dir], output_dir=output_dir)
            self.assertEqual(rows[0]["run_name"], "run_a")
            self.assertEqual(rows[0]["top1_exact_match_fraction"], 0.5)
            self.assertTrue(Path(output_dir, "phase5_summary.csv").exists())
            self.assertTrue(Path(output_dir, "phase5_summary.json").exists())

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

    @unittest.skipUnless(_rdkit_available() and _torch_available(), "RDKit and PyTorch are not installed")
    def test_learned_smiles_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=128)
            run_dir = Path(tmp, "run")
            metrics = run_learned_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=5,
                fingerprint_bits=128,
                max_length=24,
                hidden_dim=32,
                embedding_dim=16,
                epochs=1,
                batch_size=2,
                samples_per_condition=2,
                sample_top_k=4,
                tokenization="smiles_token",
                decoding="beam",
                beam_size=2,
                image_size=128,
                sample_count=2,
                device="cpu",
            )
            self.assertEqual(metrics["pairs"], 6.0)
            self.assertGreater(metrics["train_examples"], 0.0)
            self.assertGreater(metrics["eval_examples"], 0.0)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "model.pt").exists())
            self.assertTrue(Path(run_dir, "vocab.json").exists())
            self.assertTrue(Path(run_dir, "predictions.csv").exists())
            self.assertTrue(Path(run_dir, "train_history.json").exists())

    @unittest.skipUnless(_rdkit_available() and _torch_available(), "RDKit and PyTorch are not installed")
    def test_transformer_smiles_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=128)
            run_dir = Path(tmp, "run")
            metrics = run_learned_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=6,
                fingerprint_bits=128,
                max_length=24,
                hidden_dim=32,
                embedding_dim=16,
                epochs=1,
                batch_size=2,
                samples_per_condition=2,
                sample_top_k=4,
                tokenization="smiles_token",
                decoding="beam",
                beam_size=2,
                rerank_mode="condition_fingerprint",
                model_type="transformer",
                transformer_layers=1,
                attention_heads=2,
                condition_tokens=2,
                dropout=0.0,
                image_size=128,
                sample_count=2,
                device="cpu",
            )
            self.assertEqual(metrics["phase"], "phase5a4_condition_reranked_transformer_decoder")
            self.assertEqual(metrics["model_type"], "transformer")
            self.assertEqual(metrics["rerank_mode"], "condition_fingerprint")
            self.assertGreater(metrics["train_examples"], 0.0)
            self.assertGreater(metrics["eval_examples"], 0.0)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "model.pt").exists())
            self.assertTrue(Path(run_dir, "predictions.csv").exists())

            eval_metrics = evaluate_saved_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=6,
                tokenization="smiles_token",
                decoding="beam",
                beam_size=2,
                rerank_mode="condition_fingerprint",
                image_size=128,
                sample_count=2,
                device="cpu",
            )
            self.assertTrue(eval_metrics["eval_only"])
            self.assertEqual(eval_metrics["phase"], "phase5a4_condition_reranked_transformer_decoder")
            self.assertTrue(Path(run_dir, "sample_contact_sheet.png").exists())

    @unittest.skipUnless(_rdkit_available() and _torch_available(), "RDKit and PyTorch are not installed")
    def test_joint_paired_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=64)
            run_dir = Path(tmp, "run")
            metrics = run_joint_paired_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=7,
                fingerprint_bits=128,
                max_length=24,
                hidden_dim=64,
                latent_dim=32,
                embedding_dim=16,
                epochs=1,
                batch_size=2,
                samples_per_condition=2,
                sample_top_k=4,
                beam_size=2,
                image_size=64,
                sample_count=2,
                device="cpu",
            )
            self.assertEqual(metrics["phase"], "phase5b_shared_latent_smiles_sketch_decoder")
            self.assertEqual(metrics["pairs"], 6.0)
            self.assertGreater(metrics["train_examples"], 0.0)
            self.assertGreater(metrics["eval_examples"], 0.0)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "model.pt").exists())
            self.assertTrue(Path(run_dir, "vocab.json").exists())
            self.assertTrue(Path(run_dir, "predictions.csv").exists())
            self.assertTrue(Path(run_dir, "sample_contact_sheet.png").exists())

    @unittest.skipUnless(_rdkit_available() and _torch_available(), "RDKit and PyTorch are not installed")
    def test_image_conditioned_smiles_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=64)
            run_dir = Path(tmp, "run")
            metrics = run_image_conditioned_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=8,
                max_length=24,
                hidden_dim=32,
                embedding_dim=16,
                encoder_channels=8,
                image_token_grid=2,
                transformer_layers=1,
                attention_heads=2,
                dropout=0.0,
                epochs=1,
                batch_size=2,
                samples_per_condition=2,
                sample_top_k=4,
                beam_size=2,
                image_size=64,
                sample_count=2,
                device="cpu",
            )
            self.assertEqual(metrics["phase"], "phase5c_image_conditioned_smiles_decoder")
            self.assertEqual(metrics["pairs"], 6.0)
            self.assertGreater(metrics["train_examples"], 0.0)
            self.assertGreater(metrics["eval_examples"], 0.0)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "model.pt").exists())
            self.assertTrue(Path(run_dir, "vocab.json").exists())
            self.assertTrue(Path(run_dir, "predictions.csv").exists())
            self.assertTrue(Path(run_dir, "sample_contact_sheet.png").exists())

    @unittest.skipUnless(_rdkit_available() and _torch_available(), "RDKit and PyTorch are not installed")
    def test_image_fingerprint_reranked_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=64)
            run_dir = Path(tmp, "run")
            metrics = run_image_conditioned_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=9,
                max_length=24,
                hidden_dim=32,
                embedding_dim=16,
                encoder_channels=8,
                image_token_grid=2,
                fingerprint_bits=128,
                fingerprint_loss_weight=0.1,
                rerank_mode="predicted_fingerprint",
                transformer_layers=1,
                attention_heads=2,
                dropout=0.0,
                epochs=1,
                batch_size=2,
                samples_per_condition=2,
                sample_top_k=4,
                beam_size=2,
                image_size=64,
                sample_count=2,
                device="cpu",
            )
            self.assertEqual(metrics["phase"], "phase5d_image_fingerprint_reranked_smiles_decoder")
            self.assertEqual(metrics["fingerprint_bits"], 128.0)
            self.assertEqual(metrics["rerank_mode"], "predicted_fingerprint")
            self.assertIn("mean_predicted_target_fingerprint_tanimoto", metrics)
            self.assertTrue(Path(run_dir, "metrics.json").exists())
            self.assertTrue(Path(run_dir, "model.pt").exists())
            self.assertTrue(Path(run_dir, "predictions.csv").exists())

    @unittest.skipUnless(_rdkit_available(), "RDKit is not installed")
    def test_retrieval_diagnostic_writes_source_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp, "molecules.csv")
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "COC", "CCCl", "CCBr"]:
                    writer.writerow({"smiles": smiles})

            pair_dir = Path(tmp, "pairs")
            build_pair_manifest(input_csv=input_csv, output_dir=pair_dir, image_size=64)
            run_dir = Path(tmp, "source_run")
            metrics = run_learned_smiles_decoder(
                pair_dir=pair_dir,
                output_dir=run_dir,
                train_fraction=0.67,
                seed=10,
                fingerprint_bits=128,
                max_length=24,
                hidden_dim=32,
                embedding_dim=16,
                epochs=1,
                batch_size=2,
                tokenization="smiles_token",
                decoding="beam",
                beam_size=2,
                rerank_mode="condition_fingerprint",
                model_type="transformer",
                transformer_layers=1,
                attention_heads=2,
                condition_tokens=2,
                dropout=0.0,
                sample_count=2,
                device="cpu",
            )
            self.assertGreater(metrics["eval_examples"], 0.0)
            diagnostic_dir = Path(tmp, "diagnostic")
            diagnostic_metrics = run_retrieval_diagnostic(
                run_dir=run_dir,
                output_dir=diagnostic_dir,
                fingerprint_bits=128,
                retrieval_top_k=2,
            )
            self.assertEqual(diagnostic_metrics["phase"], "phase5a5_oracle_retrieval_diagnostic")
            self.assertTrue(Path(diagnostic_dir, "source_summary.csv").exists())
            self.assertTrue(Path(diagnostic_dir, "retrieval_diagnostic_rows.csv").exists())


if __name__ == "__main__":
    unittest.main()
