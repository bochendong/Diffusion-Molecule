import csv
import json
import subprocess
import sys

import numpy as np
import torch

from sketchmol_understanding_condition.unified_condition_dataset import UnifiedConditionSample, write_jsonl
from sketchmol_understanding_condition.molecule_image_vae import MoleculeImageVAE, vae_loss
from sketchmol_understanding_condition.univideo_molecule import (
    FrozenConditionFeatureStore,
    SourceConditionedEditDenoiser,
    SourceConditionedGaussianLatentDiffusion,
    UniVideoMoleculeConnector,
    univideo_connector_alignment_loss,
    univideo_training_arrays,
)


def test_molecule_image_vae_uses_sketchmol_style_latent_shape():
    model = MoleculeImageVAE(image_size=32, latent_channels=4, latent_size=4, base_channels=8)
    images = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)

    output = model(images, sample=False)
    loss, logs = vae_loss(output, images)

    assert output.latent.shape == (2, 4, 4, 4)
    assert output.reconstruction.shape == images.shape
    assert torch.isfinite(loss)
    assert "reconstruction_l1" in logs


def test_univideo_connector_and_source_conditioned_diffusion_shapes():
    connector = UniVideoMoleculeConnector(
        input_hidden_dim=8,
        latent_dim=20,
        context_dim=16,
        num_queries=4,
        hidden_dim=32,
    )
    hidden = torch.randn(3, 5, 8)
    output = connector(hidden)

    assert output.tokens.shape == (3, 4, 16)
    assert output.target_latent.shape == (3, 20)

    aux_loss, logs = univideo_connector_alignment_loss(
        output,
        target_latent=torch.randn(3, 20),
        target_properties=torch.randn(3, 7),
        property_deltas=torch.randn(3, 7),
        active_mask=torch.rand(3, 7),
        direction_labels=torch.ones(3, 7, dtype=torch.long),
        similarity_bin=torch.zeros(3, dtype=torch.long),
    )
    assert torch.isfinite(aux_loss)
    assert "target_latent_mse" in logs

    denoiser = SourceConditionedEditDenoiser(latent_dim=20, context_dim=16, hidden_dim=32, depth=2)
    diffusion = SourceConditionedGaussianLatentDiffusion(denoiser, timesteps=8)
    diffusion_loss = diffusion.loss(
        torch.randn(3, 20),
        torch.randn(3, 20),
        output.tokens,
        output.attention_mask,
    )
    assert torch.isfinite(diffusion_loss)
    assert diffusion.sample(torch.randn(3, 20), output.tokens, output.attention_mask, steps=2).shape == (3, 20)


def test_frozen_feature_store_feeds_univideo_training_arrays(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "query_tokens.npy", np.random.randn(2, 4, 8).astype(np.float32))
    with (feature_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "variant"])
        writer.writeheader()
        writer.writerow({"condition_id": "c1", "variant": "full"})
        writer.writerow({"condition_id": "c2", "variant": "full"})

    store = FrozenConditionFeatureStore(feature_dir, array_name="query_tokens", variant="full")
    sample = _sample("c1", split="train")
    row = univideo_training_arrays(sample, feature_store=store, fingerprint_dim=32)

    assert row["mllm_hidden"].shape == (4, 8)
    assert row["source_latent"].shape == row["target_latent"].shape


def test_train_univideo_molecule_generation_script_smoke(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    eval_jsonl = tmp_path / "eval.jsonl"
    train_samples = [_sample(f"train_{idx}", split="train") for idx in range(4)]
    eval_samples = [_sample(f"eval_{idx}", split="eval") for idx in range(2)]
    write_jsonl(train_jsonl, train_samples)
    write_jsonl(eval_jsonl, eval_samples)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    all_ids = [sample.metadata["condition_id"] for sample in [*train_samples, *eval_samples]]
    np.save(feature_dir / "query_tokens.npy", np.random.randn(len(all_ids), 3, 12).astype(np.float32))
    with (feature_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "variant"])
        writer.writeheader()
        for condition_id in all_ids:
            writer.writerow({"condition_id": condition_id, "variant": "full"})

    output_dir = tmp_path / "out"
    script = "scripts/train_univideo_molecule_generation.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--train-jsonl",
            str(train_jsonl),
            "--eval-jsonl",
            str(eval_jsonl),
            "--condition-features-dir",
            str(feature_dir),
            "--output-dir",
            str(output_dir),
            "--fingerprint-dim",
            "32",
            "--context-dim",
            "16",
            "--num-queries",
            "3",
            "--connector-hidden-dim",
            "32",
            "--denoiser-hidden-dim",
            "32",
            "--denoiser-depth",
            "2",
            "--timesteps",
            "4",
            "--stage1-epochs",
            "1",
            "--stage2-epochs",
            "1",
            "--stage3-epochs",
            "0",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--eval-limit",
            "2",
            "--sample-steps",
            "2",
            "--export-condition-tokens",
        ],
        cwd="SketchMol-Understanding-Condition",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "univideo_molecule_generation.pt").exists()
    assert (output_dir / "eval_latent" / "metrics.json").exists()
    assert (output_dir / "condition_tokens_train" / "query_tokens.npy").exists()


def test_export_univideo_edit_dataset_from_condition_rows(tmp_path):
    condition_rows = tmp_path / "condition_rows.csv"
    with condition_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition_id",
                "split",
                "source_smiles",
                "target_smiles",
                "instruction",
                "condition_properties",
                "property_count",
                "similarity",
                "source_QED",
                "target_QED",
                "delta_QED",
                "QED_active",
                "QED_direction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition_id": "c1",
                "split": "train",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "instruction": "Increase QED while preserving source similarity.",
                "condition_properties": "QED",
                "property_count": "1",
                "similarity": "0.5",
                "source_QED": "0.4",
                "target_QED": "0.6",
                "delta_QED": "0.2",
                "QED_active": "True",
                "QED_direction": "increase",
            }
        )

    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_univideo_edit_dataset.py",
            "--condition-rows-csv",
            str(condition_rows),
            "--output-dir",
            str(output_dir),
            "--variants",
            "full,text_only",
        ],
        cwd="SketchMol-Understanding-Condition",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    row = json.loads((output_dir / "univideo_edit_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["task_type"] == "edit_generation"
    assert row["target_properties"]["QED"] == 0.6
    with (output_dir / "baseline_variants.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"full", "text_only"}


def _sample(condition_id: str, *, split: str) -> UnifiedConditionSample:
    return UnifiedConditionSample(
        sample_id=f"edit:{condition_id}",
        task_type="edit_generation",
        split=split,
        prompt="Increase QED while preserving source similarity.",
        source_smiles="CCO",
        target_smiles="CCN",
        molecule_smiles="CCO",
        instruction="Increase QED while preserving source similarity.",
        condition_properties="QED",
        property_count="1",
        source_tanimoto="0.5",
        source_similarity_bin="medium_similarity",
        source_properties={"QED": 0.4, "MW": 46.0, "LogP": 0.0, "TPSA": 20.0, "HBD": 1.0, "HBA": 1.0, "RB": 0.0},
        target_properties={"QED": 0.6, "MW": 45.0, "LogP": 0.1, "TPSA": 18.0, "HBD": 1.0, "HBA": 1.0, "RB": 0.0},
        property_deltas={"QED": 0.2, "MW": -1.0, "LogP": 0.1, "TPSA": -2.0, "HBD": 0.0, "HBA": 0.0, "RB": 0.0},
        active_properties={"QED": True},
        directions={"QED": "increase"},
        metadata={"condition_id": condition_id},
    )
