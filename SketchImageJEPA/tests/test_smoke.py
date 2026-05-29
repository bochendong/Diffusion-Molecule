import tempfile
import unittest
import json
import csv
from pathlib import Path

import numpy as np

from sketchimage_jepa.dataset import toy_examples, write_examples_csv
from sketchimage_jepa.benchmark_audit import audit_split
from sketchimage_jepa.chem import _configure_rdkit_logging
from sketchimage_jepa.decoder import RetrievalDecoder
from sketchimage_jepa.experiment import run_experiment
from sketchimage_jepa.features import MOLECULE_LATENT_VERSION, context_vector, matrix_from_examples, molecule_latent
from sketchimage_jepa.generative_decoder import (
    EditPolicyTransformDecoder,
    GenerativeMutationDecoder,
    LatentConditionedTransformBeamDecoder,
    LearnedTransformDecoder,
    PropertyConditionedTransformDecoder,
    ScaffoldPreservingTransformDecoder,
    property_delta_mae,
)
from sketchimage_jepa.image_context import attach_rendered_image_context
from sketchimage_jepa.jepa import JEPAConfig, SketchImageJEPAPredictor
from sketchimage_jepa.hard_split import build_hard_split
from sketchimage_jepa.latent_sensitivity import run_latent_sensitivity_diagnostic
from sketchimage_jepa.oracle_latent_diffusion import OracleLatentDiffusionConfig, SmilesVocabulary, run_oracle_latent_diffusion
from sketchimage_jepa.paper_matrix import summarize_matrix
from sketchimage_jepa.phase2_calibrated_decoder import run_phase2_calibrated_decoder
from sketchimage_jepa.phase2_oracle_anchored_decoder import run_phase2_oracle_anchored_decoder
from sketchimage_jepa.phase2_planned_decoder import run_phase2_planned_decoder
from sketchimage_jepa.phase2_robust_decoder import run_phase2_robust_decoder
from sketchimage_jepa.property_guidance import parse_property_targets
from sketchimage_jepa.report import summarize_prediction_rows
from sketchimage_jepa.rerank_predictions import rerank_predictions_csv
from sketchimage_jepa.schema import BenchmarkExample, Candidate, TaskType
from sketchimage_jepa.task_builder import build_tasks_from_molecules, load_molecule_rows
from sketchimage_jepa.torch_denoiser import TorchDenoiserConfig
from sketchimage_jepa.verifier import score_candidates


def _torch_available():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


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

    def test_generative_decoder_experiment_records_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_generative",
                generative_seed_count=4,
                generative_candidates_per_seed=3,
            )
            self.assertIn("mean_best_tanimoto", metrics)
            self.assertIn("candidate_generated_fraction", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_generative")
            self.assertTrue(config["decoder"]["can_leave_training_pool"])
            with Path(tmp, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("train_pool_member", rows[0])

    def test_learned_transform_decoder_experiment_records_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_learned_transform",
                generative_seed_count=4,
                generative_candidates_per_seed=3,
            )
            self.assertIn("mean_best_tanimoto", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_learned_transform")
            self.assertEqual(config["decoder"]["source_conditioned"], "source_conditioned_learned_transform")

    def test_scaffold_transform_experiment_records_retention_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_scaffold_transform",
                generative_seed_count=4,
                generative_candidates_per_seed=3,
            )
            self.assertIn("top1_source_scaffold_retained", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_scaffold_transform")
            self.assertEqual(config["decoder"]["source_conditioned"], "source_scaffold_preserving_transform")
            with Path(tmp, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("source_scaffold_retained", rows[0])

    def test_property_transform_experiment_records_delta_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_property_transform",
                generative_seed_count=4,
                generative_candidates_per_seed=3,
            )
            self.assertIn("top1_property_delta_mae", metrics)
            self.assertIn("mean_best_property_delta_mae", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_property_transform")
            self.assertEqual(config["decoder"]["source_conditioned"], "source_property_delta_transform")
            with Path(tmp, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("property_delta_mae", rows[0])

    def test_latent_beam_transform_experiment_records_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_latent_beam_transform",
                generative_seed_count=4,
                generative_mutation_rounds=2,
                generative_candidates_per_seed=3,
            )
            self.assertIn("candidate_generated_fraction", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_latent_beam_transform")
            self.assertEqual(config["decoder"]["source_conditioned"], "latent_conditioned_transform_beam")

    def test_edit_policy_transform_experiment_records_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(
                output_dir=tmp,
                feature_dim=32,
                latent_dim=16,
                top_k=3,
                train_fraction=0.67,
                seed=7,
                decoder_mode="hybrid_edit_policy_transform",
                generative_seed_count=4,
                generative_mutation_rounds=2,
                generative_candidates_per_seed=3,
            )
            self.assertIn("candidate_generated_fraction", metrics)
            config = json.loads(Path(tmp, "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["decoder_mode"], "hybrid_edit_policy_transform")
            self.assertEqual(config["decoder"]["source_conditioned"], "supervised_edit_policy_transform")

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

    def test_rdkit_batch_logs_are_disabled_by_default(self):
        class FakeRDKitLogger:
            def __init__(self):
                self.channels = []

            def DisableLog(self, channel):
                self.channels.append(channel)

        rd_logger = FakeRDKitLogger()
        rd_base = FakeRDKitLogger()
        _configure_rdkit_logging(rd_logger, rd_base, enabled=False)
        self.assertIn("rdApp.error", rd_logger.channels)
        self.assertIn("rdApp.*", rd_base.channels)

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

    def test_generative_decoder_can_leave_train_pool(self):
        smiles = ["CCO", "CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = GenerativeMutationDecoder(
            smiles,
            latents,
            seed_count=2,
            candidates_per_seed=4,
            include_retrieval=False,
        )
        candidates = decoder.decode(latents[:1], ["CCO"], top_k=3)
        train_pool = set(smiles)
        self.assertTrue(any(candidate.smiles not in train_pool for candidate in candidates[0]))
        self.assertTrue(all(candidate.origin in {"generated_mutation", "generative_fallback_retrieval"} for candidate in candidates[0]))

    def test_learned_transform_decoder_applies_train_edit(self):
        train_examples = [
            BenchmarkExample(
                task_id="learn_replace",
                task_type=TaskType.EDIT,
                source_smiles="CCO",
                target_smiles="CCN",
                instruction="Change the terminal hetero atom.",
            )
        ]
        smiles = ["CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = LearnedTransformDecoder(
            smiles,
            latents,
            train_examples=train_examples,
            seed_count=4,
            candidates_per_seed=4,
            include_retrieval=False,
            include_mutation_fallback=False,
        )
        eval_example = BenchmarkExample(
            task_id="apply_replace",
            task_type=TaskType.EDIT,
            source_smiles="CCCO",
            target_smiles="CCCN",
            instruction="Change the terminal hetero atom.",
        )
        candidates = decoder.decode(np.stack([molecule_latent("CCCN", 16)]), ["CCCO"], top_k=3, examples=[eval_example])
        self.assertTrue(candidates[0])
        self.assertTrue(any(candidate.origin == "learned_transform" for candidate in candidates[0]))
        self.assertTrue(any(candidate.smiles not in set(smiles) for candidate in candidates[0]))

    def test_scaffold_transform_decoder_preserves_source_core(self):
        train_examples = [
            BenchmarkExample(
                task_id="learn_replace",
                task_type=TaskType.EDIT,
                source_smiles="CCO",
                target_smiles="CCN",
                instruction="Change the terminal hetero atom.",
            )
        ]
        smiles = ["CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = ScaffoldPreservingTransformDecoder(
            smiles,
            latents,
            train_examples=train_examples,
            seed_count=4,
            candidates_per_seed=4,
            include_retrieval=False,
            include_mutation_fallback=False,
        )
        eval_example = BenchmarkExample(
            task_id="apply_replace",
            task_type=TaskType.EDIT,
            source_smiles="CCCO",
            target_smiles="CCCN",
            instruction="Change the terminal hetero atom.",
        )
        candidates = decoder.decode(np.stack([molecule_latent("CCCN", 16)]), ["CCCO"], top_k=3, examples=[eval_example])
        self.assertTrue(candidates[0])
        self.assertTrue(all(candidate.origin == "scaffold_preserving_transform" for candidate in candidates[0]))
        self.assertTrue(any(candidate.smiles not in set(smiles) for candidate in candidates[0]))

    def test_property_transform_decoder_tracks_requested_delta(self):
        train_examples = [
            BenchmarkExample(
                task_id="learn_replace",
                task_type=TaskType.EDIT,
                source_smiles="CCO",
                target_smiles="CCN",
                instruction="Edit the source molecule to increase MW toward 45.08. Keep the molecule structurally related to the source.",
            )
        ]
        smiles = ["CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = PropertyConditionedTransformDecoder(
            smiles,
            latents,
            train_examples=train_examples,
            seed_count=4,
            candidates_per_seed=4,
            include_retrieval=False,
            include_mutation_fallback=False,
        )
        eval_example = BenchmarkExample(
            task_id="apply_replace",
            task_type=TaskType.EDIT,
            source_smiles="CCCO",
            target_smiles="CCCN",
            instruction="Edit the source molecule to increase MW toward 57.10. Keep the molecule structurally related to the source.",
        )
        candidates = decoder.decode(np.stack([molecule_latent("CCCN", 16)]), ["CCCO"], top_k=3, examples=[eval_example])
        self.assertTrue(candidates[0])
        self.assertTrue(all(candidate.origin == "property_conditioned_transform" for candidate in candidates[0]))
        self.assertTrue(any(property_delta_mae("CCCO", candidate.smiles, eval_example) <= 1.0 for candidate in candidates[0]))

    def test_latent_beam_transform_decoder_uses_predicted_latent(self):
        train_examples = [
            BenchmarkExample(
                task_id="learn_replace",
                task_type=TaskType.EDIT,
                source_smiles="CCO",
                target_smiles="CCN",
                instruction="Edit the source molecule to increase MW toward 45.08. Keep the molecule structurally related to the source.",
            )
        ]
        smiles = ["CCN"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = LatentConditionedTransformBeamDecoder(
            smiles,
            latents,
            train_examples=train_examples,
            seed_count=4,
            mutation_rounds=2,
            candidates_per_seed=4,
            include_retrieval=False,
            include_mutation_fallback=False,
        )
        eval_example = BenchmarkExample(
            task_id="apply_replace",
            task_type=TaskType.EDIT,
            source_smiles="CCCO",
            target_smiles="CCCN",
            instruction="Edit the source molecule to increase MW toward 57.10. Keep the molecule structurally related to the source.",
        )
        target_latent = molecule_latent("CCCN", 16)
        candidates = decoder.decode(np.stack([target_latent]), ["CCCO"], top_k=8, examples=[eval_example])
        self.assertTrue(candidates[0])
        by_smiles = {candidate.smiles: candidate for candidate in candidates[0]}
        self.assertIn("CCCN", by_smiles)
        self.assertEqual(by_smiles["CCCN"].origin, "latent_beam_transform")
        self.assertGreater(
            decoder._beam_score("CCCN", target_latent, "CCCO", eval_example),
            decoder._beam_score("CCCN", molecule_latent("CCC", 16), "CCCO", eval_example),
        )

    def test_edit_policy_transform_decoder_uses_supervised_policy(self):
        train_examples = [
            BenchmarkExample(
                task_id="learn_replace",
                task_type=TaskType.EDIT,
                source_smiles="CCO",
                target_smiles="CCN",
                instruction="Edit the source molecule to increase MW toward 45.08. Keep the molecule structurally related to the source.",
            ),
            BenchmarkExample(
                task_id="learn_trim",
                task_type=TaskType.EDIT,
                source_smiles="CCCC",
                target_smiles="CCC",
                instruction="Edit the source molecule to decrease MW toward 44.10. Keep the molecule structurally related to the source.",
            ),
        ]
        smiles = ["CCN", "CCC"]
        latents = np.stack([molecule_latent(smiles_value, 16) for smiles_value in smiles])
        decoder = EditPolicyTransformDecoder(
            smiles,
            latents,
            train_examples=train_examples,
            seed_count=4,
            mutation_rounds=2,
            candidates_per_seed=4,
            include_retrieval=False,
            include_mutation_fallback=False,
        )
        eval_example = BenchmarkExample(
            task_id="apply_replace",
            task_type=TaskType.EDIT,
            source_smiles="CCCO",
            target_smiles="CCCN",
            instruction="Edit the source molecule to increase MW toward 57.10. Keep the molecule structurally related to the source.",
        )
        candidates = decoder.decode(np.stack([molecule_latent("CCCN", 16)]), ["CCCO"], top_k=8, examples=[eval_example])
        self.assertTrue(candidates[0])
        by_smiles = {candidate.smiles: candidate for candidate in candidates[0]}
        self.assertIn("CCCN", by_smiles)
        self.assertEqual(by_smiles["CCCN"].origin, "edit_policy_transform")

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

    def test_smiles_vocabulary_round_trip(self):
        vocab = SmilesVocabulary.build(["CCO", "CCN"])
        encoded = vocab.encode("CCO", max_length=8)
        self.assertEqual(vocab.decode(encoded), "CCO")
        self.assertEqual(len(encoded), 8)

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_oracle_latent_diffusion_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=3,
            )
            output_dir = Path(tmp, "run")
            metrics = run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=output_dir, config=config)
            self.assertIn("top1_target_tanimoto", metrics)
            self.assertIn("top1_exact_match", metrics)
            self.assertTrue(Path(output_dir, "metrics.json").exists())
            self.assertTrue(Path(output_dir, "predictions.csv").exists())
            self.assertTrue(Path(output_dir, "task_type_summary.csv").exists())
            self.assertTrue(Path(output_dir, "model", "model.pt").exists())

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_phase2_planned_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            oracle_dir = Path(tmp, "oracle")
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=3,
            )
            run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=oracle_dir, config=config)
            train_examples = [
                BenchmarkExample(
                    task_id="train_edit_1",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCN",
                    instruction="Change the terminal hetero atom.",
                ),
                BenchmarkExample(
                    task_id="train_edit_2",
                    task_type=TaskType.EDIT,
                    source_smiles="CCC",
                    target_smiles="CCCl",
                    instruction="Replace one terminal atom with chlorine.",
                ),
                BenchmarkExample(
                    task_id="train_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCO",
                    instruction="Generate a small polar molecule.",
                ),
            ]
            eval_examples = [
                BenchmarkExample(
                    task_id="eval_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCC",
                    instruction="Edit the source molecule to increase MW toward 44.10.",
                ),
                BenchmarkExample(
                    task_id="eval_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCCl",
                    instruction="Generate a small molecule with chlorine.",
                ),
            ]
            train_csv = Path(tmp, "train.csv")
            eval_csv = Path(tmp, "eval.csv")
            write_examples_csv(train_csv, train_examples)
            write_examples_csv(eval_csv, eval_examples)

            phase2_dir = Path(tmp, "phase2")
            metrics = run_phase2_planned_decoder(
                oracle_decoder_dir=oracle_dir,
                output_dir=phase2_dir,
                train_csv=train_csv,
                eval_csv=eval_csv,
                feature_dim=16,
                top_k=2,
                backend="ridge",
                render_image_context=False,
                decoder_device="cpu",
                seed=3,
            )
            self.assertIn("planner_latent_mse", metrics)
            self.assertIn("planner_latent_cosine", metrics)
            self.assertTrue(Path(phase2_dir, "metrics.json").exists())
            self.assertTrue(Path(phase2_dir, "predictions.csv").exists())
            self.assertTrue(Path(phase2_dir, "task_type_summary.csv").exists())
            self.assertTrue(Path(phase2_dir, "planner", "config.json").exists())
            with Path(phase2_dir, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["origin"], "phase2_jepa_planned_decoder")
            self.assertIn("decoder_train_pool_member", rows[0])

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_phase2_robust_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            oracle_dir = Path(tmp, "oracle")
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=5,
            )
            run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=oracle_dir, config=config)
            train_examples = [
                BenchmarkExample(
                    task_id="train_edit_1",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCN",
                    instruction="Change the terminal hetero atom.",
                ),
                BenchmarkExample(
                    task_id="train_edit_2",
                    task_type=TaskType.EDIT,
                    source_smiles="CCC",
                    target_smiles="CCCl",
                    instruction="Replace one terminal atom with chlorine.",
                ),
                BenchmarkExample(
                    task_id="train_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCO",
                    instruction="Generate a small polar molecule.",
                ),
            ]
            eval_examples = [
                BenchmarkExample(
                    task_id="eval_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCC",
                    instruction="Edit the source molecule to increase MW toward 44.10.",
                ),
                BenchmarkExample(
                    task_id="eval_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCCl",
                    instruction="Generate a small molecule with chlorine.",
                ),
            ]
            train_csv = Path(tmp, "train.csv")
            eval_csv = Path(tmp, "eval.csv")
            write_examples_csv(train_csv, train_examples)
            write_examples_csv(eval_csv, eval_examples)

            phase2_dir = Path(tmp, "phase2b")
            metrics = run_phase2_robust_decoder(
                oracle_decoder_dir=oracle_dir,
                output_dir=phase2_dir,
                train_csv=train_csv,
                eval_csv=eval_csv,
                feature_dim=16,
                top_k=2,
                backend="ridge",
                render_image_context=False,
                decoder_device="cpu",
                decoder_finetune_epochs=1,
                decoder_finetune_batch_size=2,
                decoder_finetune_lr=1e-4,
                decoder_oracle_repeats=1,
                decoder_planner_repeats=1,
                decoder_noise_repeats=1,
                decoder_interpolation_repeats=1,
                seed=5,
            )
            self.assertIn("planner_latent_mse", metrics)
            self.assertIn("decoder_finetune_rows", metrics)
            self.assertGreater(metrics["decoder_finetune_rows"], 0.0)
            self.assertTrue(Path(phase2_dir, "metrics.json").exists())
            self.assertTrue(Path(phase2_dir, "predictions.csv").exists())
            self.assertTrue(Path(phase2_dir, "task_type_summary.csv").exists())
            self.assertTrue(Path(phase2_dir, "planner", "config.json").exists())
            self.assertTrue(Path(phase2_dir, "decoder", "model.pt").exists())
            self.assertTrue(Path(phase2_dir, "planner_eval_latents.npy").exists())
            with Path(phase2_dir, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["origin"], "phase2_jepa_planned_robust_decoder")

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_phase2_calibrated_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            decoder_run = Path(tmp, "oracle")
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=11,
            )
            run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=decoder_run, config=config)
            train_examples = [
                BenchmarkExample(
                    task_id="train_edit_1",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCN",
                    instruction="Change the terminal hetero atom.",
                ),
                BenchmarkExample(
                    task_id="train_edit_2",
                    task_type=TaskType.EDIT,
                    source_smiles="CCC",
                    target_smiles="CCCl",
                    instruction="Replace one terminal atom with chlorine.",
                ),
                BenchmarkExample(
                    task_id="train_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCO",
                    instruction="Generate a small polar molecule.",
                ),
            ]
            eval_examples = [
                BenchmarkExample(
                    task_id="eval_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCC",
                    instruction="Edit the source molecule to increase MW toward 44.10.",
                ),
                BenchmarkExample(
                    task_id="eval_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCCl",
                    instruction="Generate a small molecule with chlorine.",
                ),
            ]
            train_csv = Path(tmp, "train.csv")
            eval_csv = Path(tmp, "eval.csv")
            write_examples_csv(train_csv, train_examples)
            write_examples_csv(eval_csv, eval_examples)

            phase2_dir = Path(tmp, "phase2c")
            metrics = run_phase2_calibrated_decoder(
                decoder_dir=decoder_run,
                decoder_pool_dir=decoder_run,
                output_dir=phase2_dir,
                train_csv=train_csv,
                eval_csv=eval_csv,
                feature_dim=16,
                top_k=2,
                backend="ridge",
                render_image_context=False,
                decoder_device="cpu",
                calibration_mode="residual_ridge",
                calibration_ridge=0.1,
                calibration_blend=1.0,
                calibration_normalize=True,
                seed=11,
            )
            self.assertIn("calibrated_latent_cosine", metrics)
            self.assertIn("planner_latent_cosine", metrics)
            self.assertTrue(Path(phase2_dir, "metrics.json").exists())
            self.assertTrue(Path(phase2_dir, "predictions.csv").exists())
            self.assertTrue(Path(phase2_dir, "task_type_summary.csv").exists())
            self.assertTrue(Path(phase2_dir, "calibrator", "config.json").exists())
            self.assertTrue(Path(phase2_dir, "calibrator", "weights.npy").exists())
            self.assertTrue(Path(phase2_dir, "calibrated_eval_latents.npy").exists())
            with Path(phase2_dir, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["origin"], "phase2_jepa_calibrated_decoder")

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_latent_sensitivity_diagnostic_writes_source_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            decoder_run = Path(tmp, "oracle")
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=13,
            )
            run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=decoder_run, config=config)
            train_examples = [
                BenchmarkExample(
                    task_id="train_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCN",
                    instruction="Change the terminal hetero atom.",
                ),
                BenchmarkExample(
                    task_id="train_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCO",
                    instruction="Generate a small polar molecule.",
                ),
            ]
            eval_examples = [
                BenchmarkExample(
                    task_id="eval_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCC",
                    instruction="Edit the source molecule to increase MW toward 44.10.",
                ),
                BenchmarkExample(
                    task_id="eval_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCCl",
                    instruction="Generate a small molecule with chlorine.",
                ),
            ]
            train_csv = Path(tmp, "train.csv")
            eval_csv = Path(tmp, "eval.csv")
            write_examples_csv(train_csv, train_examples)
            write_examples_csv(eval_csv, eval_examples)
            _, eval_targets, _ = matrix_from_examples(eval_examples, feature_dim=16, latent_dim=16)
            planner_dir = Path(tmp, "planner")
            planner_dir.mkdir()
            calibrated_dir = Path(tmp, "calibrated")
            calibrated_dir.mkdir()
            np.save(planner_dir / "planner_eval_latents.npy", eval_targets.astype(np.float32))
            np.save(calibrated_dir / "calibrated_eval_latents.npy", eval_targets.astype(np.float32))

            output_dir = Path(tmp, "sensitivity")
            summary = run_latent_sensitivity_diagnostic(
                decoder_dir=decoder_run,
                decoder_pool_dir=decoder_run,
                planner_run_dir=planner_dir,
                calibrated_run_dir=calibrated_dir,
                output_dir=output_dir,
                train_csv=train_csv,
                eval_csv=eval_csv,
                feature_dim=16,
                top_k=2,
                noisy_cosines=(0.5,),
                interpolation_alphas=(0.5,),
                decoder_device="cpu",
                seed=13,
            )
            self.assertTrue(Path(output_dir, "source_summary.csv").exists())
            self.assertTrue(Path(output_dir, "source_summary.json").exists())
            self.assertTrue(Path(output_dir, "run_config.json").exists())
            names = {row["latent_source"] for row in summary}
            self.assertIn("oracle_target", names)
            self.assertIn("noisy_oracle_c0_50", names)
            self.assertIn("planner_predicted", names)
            self.assertIn("calibrated_predicted", names)
            self.assertTrue(Path(output_dir, "oracle_target", "predictions.csv").exists())

    @unittest.skipUnless(_torch_available(), "PyTorch is not installed")
    def test_phase2_oracle_anchored_decoder_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            molecule_csv = Path(tmp, "molecules.csv")
            with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["smiles"])
                writer.writeheader()
                for smiles in ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CC(=O)O"]:
                    writer.writerow({"smiles": smiles})
            decoder_run = Path(tmp, "oracle")
            config = OracleLatentDiffusionConfig(
                condition_dim=16,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                max_length=24,
                epochs=1,
                batch_size=2,
                sample_steps=2,
                samples_per_condition=2,
                sample_multiplier=1,
                sample_batch_size=4,
                device="cpu",
                seed=17,
            )
            run_oracle_latent_diffusion(molecule_csv=molecule_csv, output_dir=decoder_run, config=config)
            train_examples = [
                BenchmarkExample(
                    task_id="train_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCN",
                    instruction="Change the terminal hetero atom.",
                ),
                BenchmarkExample(
                    task_id="train_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCO",
                    instruction="Generate a small polar molecule.",
                ),
            ]
            eval_examples = [
                BenchmarkExample(
                    task_id="eval_edit",
                    task_type=TaskType.EDIT,
                    source_smiles="CCO",
                    target_smiles="CCC",
                    instruction="Edit the source molecule to increase MW toward 44.10.",
                ),
                BenchmarkExample(
                    task_id="eval_denovo",
                    task_type=TaskType.DE_NOVO,
                    target_smiles="CCCl",
                    instruction="Generate a small molecule with chlorine.",
                ),
            ]
            train_csv = Path(tmp, "train.csv")
            eval_csv = Path(tmp, "eval.csv")
            write_examples_csv(train_csv, train_examples)
            write_examples_csv(eval_csv, eval_examples)
            _, train_targets, _ = matrix_from_examples(train_examples, feature_dim=16, latent_dim=16)
            _, eval_targets, _ = matrix_from_examples(eval_examples, feature_dim=16, latent_dim=16)
            planner_dir = Path(tmp, "planner")
            planner_dir.mkdir()
            calibrated_dir = Path(tmp, "calibrated")
            calibrated_dir.mkdir()
            np.save(planner_dir / "planner_train_latents.npy", train_targets.astype(np.float32))
            np.save(planner_dir / "planner_eval_latents.npy", eval_targets.astype(np.float32))
            np.save(calibrated_dir / "calibrated_train_latents.npy", train_targets.astype(np.float32))
            np.save(calibrated_dir / "calibrated_eval_latents.npy", eval_targets.astype(np.float32))

            output_dir = Path(tmp, "phase2d")
            metrics = run_phase2_oracle_anchored_decoder(
                oracle_decoder_dir=decoder_run,
                planner_run_dir=planner_dir,
                calibrated_run_dir=calibrated_dir,
                decoder_pool_dir=decoder_run,
                output_dir=output_dir,
                train_csv=train_csv,
                eval_csv=eval_csv,
                feature_dim=16,
                top_k=2,
                decoder_device="cpu",
                decoder_finetune_epochs=1,
                decoder_finetune_batch_size=2,
                decoder_oracle_repeats=1,
                decoder_noisy_repeats=0,
                decoder_planner_repeats=1,
                decoder_calibrated_repeats=1,
                decoder_interpolation_repeats=0,
                seed=17,
            )
            self.assertIn("oracle_target_mean_best_tanimoto", metrics)
            self.assertIn("planner_predicted_mean_best_tanimoto", metrics)
            self.assertTrue(Path(output_dir, "decoder", "model.pt").exists())
            self.assertTrue(Path(output_dir, "source_summary.csv").exists())
            self.assertTrue(Path(output_dir, "predictions.csv").exists())
            with Path(output_dir, "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["origin"], "phase2_jepa_oracle_anchored_decoder")

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

    def test_rerank_predictions_can_promote_property_delta_match(self):
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
                "property_delta_mae",
                "property_delta_success",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "e1",
                        "task_type": "edit",
                        "instruction": "Edit the source molecule to increase MW toward 57.10.",
                        "source_smiles": "CCCO",
                        "target_smiles": "CCCN",
                        "rank": "1",
                        "candidate_smiles": "CCCCCCCC",
                        "origin": "model",
                        "valid": "True",
                        "target_tanimoto": "0.1",
                        "scaffold_match": "False",
                        "score": "1.0",
                        "property_mae": "1.0",
                        "property_success": "False",
                        "property_delta_mae": "3.0",
                        "property_delta_success": "False",
                    }
                )
                writer.writerow(
                    {
                        "task_id": "e1",
                        "task_type": "edit",
                        "instruction": "Edit the source molecule to increase MW toward 57.10.",
                        "source_smiles": "CCCO",
                        "target_smiles": "CCCN",
                        "rank": "2",
                        "candidate_smiles": "CCCN",
                        "origin": "model",
                        "valid": "True",
                        "target_tanimoto": "1.0",
                        "scaffold_match": "True",
                        "score": "0.0",
                        "property_mae": "0.0",
                        "property_success": "True",
                        "property_delta_mae": "0.1",
                        "property_delta_success": "True",
                    }
                )
            records = rerank_predictions_csv(
                path,
                out_dir=Path(tmp, "rerank"),
                base_weights=[0.0],
                source_weights=[0.0],
                property_weights=[0.0],
                scaffold_weights=[0.0],
                property_delta_weights=[1.0],
            )
            self.assertEqual(records[0]["top1_property_delta_success"], 1.0)
            with Path(tmp, "rerank", "best_reranked_predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["candidate_smiles"], "CCCN")
            with Path(tmp, "rerank", "best_by_objective.csv").open() as handle:
                objective_rows = list(csv.DictReader(handle))
            objectives = {row["objective"] for row in objective_rows}
            self.assertIn("property_delta", objectives)
            self.assertTrue(Path(tmp, "rerank", "best_property_delta_task_type_summary.csv").exists())

    def test_torch_denoiser_config_has_contrastive_defaults(self):
        config = TorchDenoiserConfig()
        self.assertGreater(config.contrastive_loss_weight, 0.0)
        self.assertGreater(config.contrastive_temperature, 0.0)
        self.assertGreater(config.delta_loss_weight, 0.0)
        self.assertEqual(config.hard_negative_loss_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
