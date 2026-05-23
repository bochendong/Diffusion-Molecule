import tempfile
import unittest
import json
import csv
from pathlib import Path

import numpy as np

from sketchimage_jepa.dataset import toy_examples
from sketchimage_jepa.decoder import RetrievalDecoder
from sketchimage_jepa.experiment import run_experiment
from sketchimage_jepa.features import MOLECULE_LATENT_VERSION, context_vector, matrix_from_examples, molecule_latent
from sketchimage_jepa.image_context import attach_rendered_image_context
from sketchimage_jepa.jepa import JEPAConfig, SketchImageJEPAPredictor
from sketchimage_jepa.report import summarize_prediction_rows
from sketchimage_jepa.schema import BenchmarkExample, Candidate, TaskType
from sketchimage_jepa.task_builder import build_tasks_from_molecules, load_molecule_rows
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


if __name__ == "__main__":
    unittest.main()
