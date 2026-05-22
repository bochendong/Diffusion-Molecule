import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from sketchimage_jepa.dataset import toy_examples
from sketchimage_jepa.experiment import run_experiment
from sketchimage_jepa.features import context_vector, matrix_from_examples
from sketchimage_jepa.image_context import attach_rendered_image_context
from sketchimage_jepa.jepa import JEPAConfig, SketchImageJEPAPredictor
from sketchimage_jepa.schema import BenchmarkExample, TaskType


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


if __name__ == "__main__":
    unittest.main()
