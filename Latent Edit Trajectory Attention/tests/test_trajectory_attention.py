import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from latent_edit_trajectory_attention.models import TrajectoryDiffusionConfig


class TrajectoryAttentionConfigTests(unittest.TestCase):
    def test_config_validates_attention_divisibility(self):
        config = TrajectoryDiffusionConfig(hidden_dim=130, attention_heads=8)
        with self.assertRaises(ValueError):
            config.validate()

    def test_config_accepts_default_values(self):
        TrajectoryDiffusionConfig().validate()


class TrajectoryAttentionTorchTests(unittest.TestCase):
    def test_forward_shapes(self):
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is not installed.")

        from latent_edit_trajectory_attention.data import SyntheticTrajectoryConfig, SyntheticTrajectoryDataset
        from latent_edit_trajectory_attention.models import CurrentStateDiffusionEditor, TrajectoryConditionedDiffusionEditor

        config = TrajectoryDiffusionConfig(
            latent_dim=16,
            property_dim=2,
            target_dim=2,
            hidden_dim=32,
            transformer_layers=1,
            attention_heads=4,
            diffusion_steps=8,
            max_history=4,
            dropout=0.0,
        )
        dataset = SyntheticTrajectoryDataset(
            model_config=config,
            data_config=SyntheticTrajectoryConfig(examples=3, history_length=4),
        )
        batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
        model = TrajectoryConditionedDiffusionEditor(config)
        noise_step = torch.tensor([1, 3])
        predicted_noise, context = model(
            noisy_next_z=batch["next_z"],
            noise_step=noise_step,
            z_history=batch["z_history"],
            property_delta=batch["property_delta"],
            edit_type_ids=batch["edit_type_ids"],
            history_mask=batch["history_mask"],
            target=batch["target"],
        )
        self.assertEqual(tuple(predicted_noise.shape), (2, 16))
        self.assertEqual(tuple(context.shape), (2, 32))

        current_model = CurrentStateDiffusionEditor(config)
        current_noise, current_context = current_model(
            noisy_next_z=batch["next_z"],
            noise_step=noise_step,
            z_history=batch["z_history"],
            property_delta=batch["property_delta"],
            edit_type_ids=batch["edit_type_ids"],
            history_mask=batch["history_mask"],
            target=batch["target"],
        )
        self.assertEqual(tuple(current_noise.shape), (2, 16))
        self.assertEqual(tuple(current_context.shape), (2, 32))

    def test_sketchmol_opt_pair_dataset_loads_examples(self):
        try:
            import torch  # noqa: F401
            import rdkit  # noqa: F401
        except Exception:
            self.skipTest("PyTorch and RDKit are required.")

        from latent_edit_trajectory_attention.data import SketchMolOptPairConfig, SketchMolOptPairDataset

        config = TrajectoryDiffusionConfig(
            latent_dim=64,
            property_dim=4,
            target_dim=4,
            hidden_dim=32,
            transformer_layers=1,
            attention_heads=4,
            diffusion_steps=8,
            max_history=4,
            dropout=0.0,
        )
        dataset = SketchMolOptPairDataset(
            model_config=config,
            data_config=SketchMolOptPairConfig(max_examples=8),
        )
        example = dataset[0]
        self.assertGreater(len(dataset), 0)
        self.assertEqual(tuple(example["z_history"].shape), (1, 64))
        self.assertEqual(tuple(example["next_z"].shape), (64,))
        self.assertEqual(tuple(example["property_delta"].shape), (1, 4))
        self.assertIn("before_smiles", example)

    def test_trajectory_dataset_and_metrics(self):
        try:
            import torch
            import rdkit  # noqa: F401
        except Exception:
            self.skipTest("PyTorch and RDKit are required.")

        from latent_edit_trajectory_attention.data import (
            SketchMolTrajectoryConfig,
            SketchMolTrajectoryDataset,
            collate_trajectory_batch,
        )
        from latent_edit_trajectory_attention.metrics import summarize_trajectories
        from latent_edit_trajectory_attention.schema import write_jsonl
        from latent_edit_trajectory_attention.trajectory_generator import steps_from_smiles_sequence

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "traj.jsonl"
            steps = steps_from_smiles_sequence(
                trajectory_id="toy_0",
                smiles_sequence=["CCO", "CCCO", "CCCCO"],
                task_name="qed",
                source="unit_test",
            )
            write_jsonl(trajectory_path, steps)
            config = TrajectoryDiffusionConfig(
                latent_dim=64,
                property_dim=5,
                target_dim=4,
                hidden_dim=32,
                transformer_layers=1,
                attention_heads=4,
                diffusion_steps=8,
                max_history=4,
                dropout=0.0,
            )
            dataset = SketchMolTrajectoryDataset(
                model_config=config,
                data_config=SketchMolTrajectoryConfig(trajectory_path=str(trajectory_path)),
            )
            self.assertEqual(len(dataset), 2)
            batch = collate_trajectory_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["z_history"].shape), (2, 2, 64))
            self.assertEqual(tuple(batch["property_delta"].shape), (2, 2, 5))
            self.assertTrue(torch.is_tensor(batch["history_mask"]))
            summary = summarize_trajectories(trajectory_path)
            self.assertEqual(summary["trajectory_count"], 1)
            self.assertEqual(summary["step_count"], 3)


if __name__ == "__main__":
    unittest.main()

