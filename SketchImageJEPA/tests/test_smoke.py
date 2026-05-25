import tempfile
import unittest
import json
import csv
from pathlib import Path

import numpy as np

from sketchimage_jepa.dataset import toy_examples
from sketchimage_jepa.benchmark_audit import audit_split
from sketchimage_jepa.decoder import RetrievalDecoder
from sketchimage_jepa.experiment import run_experiment
from sketchimage_jepa.features import MOLECULE_LATENT_VERSION, context_vector, matrix_from_examples, molecule_latent
from sketchimage_jepa.image_context import attach_rendered_image_context
from sketchimage_jepa.jepa import JEPAConfig, SketchImageJEPAPredictor
from sketchimage_jepa.hard_split import build_hard_split
from sketchimage_jepa.paper_matrix import summarize_matrix
from sketchimage_jepa.property_guidance import parse_property_targets
from sketchimage_jepa.report import summarize_prediction_rows
from sketchimage_jepa.rerank_predictions import rerank_predictions_csv
from sketchimage_jepa.schema import BenchmarkExample, Candidate, TaskType
from sketchimage_jepa.task_builder import build_tasks_from_molecules, load_molecule_rows
from sketchimage_jepa.torch_denoiser import TorchDenoiserConfig
from sketchimage_jepa.verifier import score_candidates


class SketchImageJEPATests(unittest.TestCase):
    def test_predictor_save_load_interface(self):
        examples = toy_examples()
        conditions, targets, sources = matrix_from_examples(examples, feature_dim=32, latent_dim=16)
        model = SketchImageJEPAPredictor(JEPAConfig(feature_dim=32, latent_dim=16))
        model.fit(conditions, targets, sources)
        pred = model.predict(conditions, sources)
        self.assertEqual(pred.shape, targets.shape)
        self.assertTrue(np.isfinite(pred).all())
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = SketchImageJEPAPredictor.load(tmp)
            pred2 = loaded.predict(conditions, sources)
            self.assertEqual(pred2.shape, targets.shape)

    def test_smoke_experiment_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(output_dir=tmp, feature_dim=32, latent_dim=16, top_k=3, train_fraction=0.67, seed=7, render_image_context=True)
            self.assertIn("mean_best_tanimoto", metrics)
            self.assertEqual(metrics["train_tasks"], 4.0)
            self.assertEqual(metrics["eval_tasks"], 2.0)
            self.assertTrue(Path(tmp, "metrics.json").exists())
            self.assertTrue(Path(tmp, "predictions.csv").exists())
            self.assertTrue(Path(tmp, "task_type_summary.csv").exists())
            self.assertTrue(Path(tmp, "task_type_summary.json").exists())
            self.assertTrue(Path(tmp, "run_config.json").exists())
            self.assertTrue(Path(tmp, "train_examples.csv").exists())
            self.assertTrue(Path(tmp, "eval_examples.csv").exists())
            self.assertTrue(Path(tmp, "model", "config.json").exists())

    def test_sketchmol_aligned_preset_records_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(output_dir=tmp, feature_dim=64, latent_dim=64, top_k=4, train_fraction=0.67, seed=7, preset="sketchmol_aligned")
            self.assertEqual(metrics["train_tasks"], 4.0)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["preset"], "sketchmol_aligned")
            self.assertEqual(config["backend"], "ridge")
            self.assertEqual(config["molecule_latent_version"], MOLECULE_LATENT_VERSION)
            self.assertEqual(config["sketchmol_reference"]["condition_dim"], 256)
            self.assertEqual(config["sketchmol_reference"]["latent_dim"], 4096)
            self.assertEqual(config["sketchmol_reference"]["samples_per_condition"], 8)

    def test_context_does_not_leak_target_smiles(self):
        left = BenchmarkExample(
            task_id="same_context",
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            target_smiles="CCN",
            instruction="Change the terminal hetero atom.",
            goals=("small_edit",),
        )
        right = BenchmarkExample(
            task_id="same_context",
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            target_smiles="CCCCCCCC",
            instruction="Change the terminal hetero atom.",
            goals=("small_edit",),
        )
        np.testing.assert_allclose(context_vector(left, 32), context_vector(right, 32))

    def test_image_context_does_not_render_denovo_target(self):
        example = BenchmarkExample(
            task_id="denovo_no_source",
            task_type=TaskType.DE_NOVO,
            target_smiles="CCO",
            instruction="Generate ethanol.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            updated, meta = attach_rendered_image_context([example], tmp)
            self.assertIsNone(updated[0].image_path)
            self.assertEqual(meta["rendered_images"], 0)

    def test_task_builder_creates_task_csv_rows(self):
        smiles = [
            "c1ccccc1",
            "Cc1ccccc1",
            "Oc1ccccc1",
            "Nc1ccccc1",
            "CCOc1ccccc1",
            "CCOc1ccc(O)cc1",
            "CCN(CC)CC",
            "CCN(CCO)CCO",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "molecules.csv")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smi in smiles:
                    writer.writerow({"smiles": smi})
            molecules = load_molecule_rows(path)
            tasks = build_tasks_from_molecules(molecules, max_tasks=12, pairs_per_source=2, pair_candidates=8, seed=3)
            task_types = {task.task_type.value for task in tasks}
            self.assertIn("de_novo", task_types)
            self.assertIn("edit", task_types)
            self.assertIn("inpaint", task_types)
            self.assertLessEqual(len(tasks), 12)

    def test_report_groups_metrics_by_task_type(self):
        rows = [
            {
                "task_id": "d1",
                "task_type": "de_novo",
                "rank": "1",
                "valid": "True",
                "target_tanimoto": "0.20",
                "scaffold_match": "False",
            },
            {
                "task_id": "d1",
                "task_type": "de_novo",
                "rank": "2",
                "valid": "True",
                "target_tanimoto": "0.70",
                "scaffold_match": "False",
            },
            {
                "task_id": "e1",
                "task_type": "edit",
                "rank": "1",
                "valid": "True",
                "target_tanimoto": "0.80",
                "scaffold_match": "True",
            },
        ]
        summary = {row["task_type"]: row for row in summarize_prediction_rows(rows)}
        self.assertEqual(summary["overall"]["n"], 2)
        self.assertEqual(summary["de_novo"]["n"], 1)
        self.assertAlmostEqual(summary["de_novo"]["top1_target_tanimoto"], 0.20)
        self.assertAlmostEqual(summary["de_novo"]["mean_best_tanimoto"], 0.70)
        self.assertAlmostEqual(summary["edit"]["top1_scaffold_match"], 1.0)

    def test_paper_matrix_summarizes_seeded_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, top1 in ((7, 0.4), (13, 0.6)):
                run_dir = Path(tmp, f"paper_planner_best_seed{seed}")
                run_dir.mkdir(parents=True)
                with Path(run_dir, "task_type_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "task_type",
                            "n",
                            "top1_target_tanimoto",
                            "mean_best_tanimoto",
                            "topk_target_hit",
                            "top1_scaffold_match",
                            "top1_property_success",
                            "topk_property_success",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "task_type": "overall",
                            "n": "100",
                            "top1_target_tanimoto": str(top1),
                            "mean_best_tanimoto": "0.7",
                            "topk_target_hit": "0.5",
                            "top1_scaffold_match": "0.25",
                            "top1_property_success": "0.55",
                            "topk_property_success": "0.75",
                        }
                    )
            rows, missing = summarize_matrix("paper", ["planner_best"], [7, 13, 23], run_root=tmp)
            overall = next(row for row in rows if row["variant"] == "planner_best" and row["task_type"] == "overall")
            self.assertEqual(overall["seeds_done"], 2)
            self.assertEqual(overall["missing_seeds"], "23")
            self.assertAlmostEqual(overall["top1_target_tanimoto_mean"], 0.5)
            self.assertEqual(len(missing), 1)

    def test_benchmark_audit_detects_train_nearest_shortcut(self):
        examples = toy_examples()
        train = examples[:4]
        eval_examples = examples[4:]
        rows, summary = audit_split(train, eval_examples)
        self.assertEqual(len(rows), 2)
        overall = next(row for row in summary if row["task_type"] == "overall")
        self.assertIn("mean_nearest_train_target_tanimoto", overall)
        self.assertGreaterEqual(overall["mean_nearest_train_target_tanimoto"], 0.0)

    def test_hard_split_writes_non_overlapping_eval(self):
        train, eval_examples, summary = build_hard_split(toy_examples(), eval_fraction=0.33, seed=5, max_train_target_tanimoto=0.95)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(eval_examples), 0)
        self.assertIn("audit_summary", summary)
        train_ids = {example.task_id for example in train}
        eval_ids = {example.task_id for example in eval_examples}
        self.assertFalse(train_ids & eval_ids)

    def test_denovo_decoder_uses_property_guidance(self):
        smiles = ["CCCCCCCC", "CCO", "c1ccccc1"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        example = BenchmarkExample(
            task_id="denovo_property",
            task_type=TaskType.DE_NOVO,
            target_smiles="CCO",
            instruction="Generate a molecule with MW around 40, LogP around 0.3, QED around 0.5, and TPSA around 17.",
        )
        decoder = RetrievalDecoder(smiles, latents)
        candidates = decoder.decode(np.zeros((1, 16), dtype=np.float32), [None], top_k=1, examples=[example])
        self.assertEqual(candidates[0][0].smiles, "CCO")
        self.assertEqual(candidates[0][0].origin, "property_guided_retrieval")

    def test_property_parser_accepts_toward_language(self):
        targets = parse_property_targets("Edit the source molecule to increase MW toward 45.08.")
        self.assertAlmostEqual(targets["MW"], 45.08)

    def test_source_conditioned_decoder_uses_task_guidance(self):
        smiles = ["CCCCCCCC", "CCN"]
        latents = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        source_latent = np.asarray([[0.0, 1.0]], dtype=np.float32)
        pred_latent = np.asarray([[1.0, 0.0]], dtype=np.float32)
        example = BenchmarkExample(
            task_id="edit_task_guidance",
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            target_smiles="CCN",
            instruction="Edit the source molecule to increase MW toward 45.08. Keep the molecule structurally related to the source.",
        )
        decoder = RetrievalDecoder(smiles, latents, source_rerank_weight=1.0, property_rerank_weight=0.5)
        candidates = decoder.decode(pred_latent, ["CCO"], top_k=1, examples=[example], source_latents=source_latent)
        self.assertEqual(candidates[0][0].smiles, "CCN")
        self.assertEqual(candidates[0][0].origin, "task_guided_retrieval")

    def test_decoder_does_not_return_source_as_top_candidate(self):
        smiles = ["CCO", "CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = RetrievalDecoder(smiles, latents)
        candidates = decoder.decode(latents[:1], ["CCO"], top_k=1)
        self.assertEqual(candidates[0][0].smiles, "CCN")

    def test_scoring_preserves_model_rank_without_target_oracle(self):
        example = BenchmarkExample(
            task_id="rank_order",
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            target_smiles="CCN",
            instruction="Change the terminal hetero atom.",
        )
        candidates = [
            Candidate(smiles="CCCCCCCC", origin="model", score=0.9, rank=1),
            Candidate(smiles="CCN", origin="model", score=0.1, rank=2),
        ]
        scores = score_candidates(example, candidates)
        self.assertEqual(scores[0].smiles, "CCCCCCCC")
        self.assertGreater(scores[1].target_tanimoto, scores[0].target_tanimoto)

    def test_molecule_latent_shape_and_norm(self):
        latent = molecule_latent("CCO", 64)
        self.assertEqual(latent.shape, (64,))
        self.assertTrue(np.isfinite(latent).all())
        self.assertGreater(float(np.linalg.norm(latent)), 0.0)

    def test_rerank_predictions_can_promote_property_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "predictions.csv")
            fieldnames = [
                "task_id",
                "task_type",
                "instruction",
                "source_smiles",
                "target_smiles",
                "rank",
                "candidate_smiles",
                "origin",
                "valid",
                "target_tanimoto",
                "scaffold_match",
                "score",
                "property_mae",
                "property_success",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "d1",
                        "task_type": "de_novo",
                        "instruction": "Generate a molecule with MW around 40.",
                        "source_smiles": "",
                        "target_smiles": "CCO",
                        "rank": "1",
                        "candidate_smiles": "CCCCCCCC",
                        "origin": "model",
                        "valid": "True",
                        "target_tanimoto": "0.1",
                        "scaffold_match": "False",
                        "score": "1.0",
                        "property_mae": "1.0",
                        "property_success": "False",
                    }
                )
                writer.writerow(
                    {
                        "task_id": "d1",
                        "task_type": "de_novo",
                        "instruction": "Generate a molecule with MW around 40.",
                        "source_smiles": "",
                        "target_smiles": "CCO",
                        "rank": "2",
                        "candidate_smiles": "CCO",
                        "origin": "model",
                        "valid": "True",
                        "target_tanimoto": "1.0",
                        "scaffold_match": "True",
                        "score": "0.0",
                        "property_mae": "0.0",
                        "property_success": "True",
                    }
                )
            records = rerank_predictions_csv(
                path,
                out_dir=Path(tmp, "rerank"),
                base_weights=[0.0],
                source_weights=[0.0],
                property_weights=[1.0],
                scaffold_weights=[0.0],
            )
            self.assertEqual(records[0]["top1_target_tanimoto"], 1.0)

    def test_torch_denoiser_config_has_contrastive_defaults(self):
        config = TorchDenoiserConfig()
        self.assertGreater(config.contrastive_loss_weight, 0.0)
        self.assertGreater(config.contrastive_temperature, 0.0)
        self.assertGreater(config.delta_loss_weight, 0.0)
        self.assertEqual(config.hard_negative_loss_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
