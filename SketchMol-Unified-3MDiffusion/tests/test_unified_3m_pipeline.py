import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sketchmol_unified_3m_diffusion.benchmark_export import write_edit_latent_benchmark_inputs
from sketchmol_unified_3m_diffusion.edit_condition_tokens import (
    EditConditionTokenConnector,
    edit_condition_loss,
)
from sketchmol_unified_3m_diffusion.latent_diffusion_generation import (
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (
    EDIT_GENERATION,
    UnifiedConditionSample,
    read_3m_description_samples,
    read_edit_generation_samples,
    split_samples,
)


def test_unified_dataset_reads_description_and_edit_rows(tmp_path):
    description_tsv = tmp_path / "train.txt"
    description_tsv.write_text(
        "CID\tSMILES\tdescription\n"
        "1\tCCO\tThe molecule is ethanol.\n",
        encoding="utf-8",
    )
    edit_csv = tmp_path / "manifest.csv"
    with edit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "source_smiles",
                "target_smiles",
                "instruction",
                "property_count",
                "source_tanimoto",
                "source_similarity_bin",
                "target_QED",
                "delta_QED",
                "QED_active",
                "QED_direction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "e1",
                "split": "train",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "instruction": "Increase QED while preserving source similarity.",
                "property_count": "1",
                "source_tanimoto": "0.5",
                "source_similarity_bin": "medium_similarity",
                "target_QED": "0.6",
                "delta_QED": "0.1",
                "QED_active": "1",
                "QED_direction": "increase",
            }
        )

    samples = []
    samples.extend(read_3m_description_samples(description_tsv, split="train", dataset_name="toy"))
    samples.extend(read_edit_generation_samples(edit_csv))
    summary = split_samples(
        samples,
        train_output=tmp_path / "train.jsonl",
        eval_output=tmp_path / "eval.jsonl",
    )

    assert summary["rows"] == 2
    assert summary["tasks"]["description_pretrain"] == 1
    assert summary["tasks"][EDIT_GENERATION] == 1
    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert row["target_properties"]["QED"] == 0.6
    assert row["active_properties"]["QED"] is True


def test_edit_condition_connector_and_diffusion_loss_shapes():
    connector = EditConditionTokenConnector(
        input_hidden_dim=32,
        context_dim=16,
        num_queries=4,
        hidden_dim=32,
        fingerprint_dim=64,
    )
    hidden = torch.randn(3, 5, 32)
    output = connector(hidden)

    assert output.tokens.shape == (3, 4, 16)
    assert output.target_fingerprint_logits.shape == (3, 64)

    loss, logs = edit_condition_loss(
        output,
        target_properties=torch.randn(3, 7),
        property_deltas=torch.randn(3, 7),
        active_mask=torch.rand(3, 7),
        direction_labels=torch.ones(3, 7, dtype=torch.long),
        target_fingerprint=torch.rand(3, 64),
        similarity_bin=torch.zeros(3, dtype=torch.long),
    )
    assert torch.isfinite(loss)
    assert "fingerprint_bce" in logs

    denoiser = EditLatentDenoiser(latent_dim=20, context_dim=16, hidden_dim=32, depth=2)
    diffusion = GaussianLatentDiffusion(denoiser, timesteps=8)
    diffusion_loss = diffusion.loss(torch.randn(3, 20), output.tokens, output.attention_mask)

    assert torch.isfinite(diffusion_loss)


def test_benchmark_export_writes_condition_aligned_latents(tmp_path):
    sample = UnifiedConditionSample(
        sample_id="edit:multiproperty_edit:pair_1_cond_00_2p",
        task_type=EDIT_GENERATION,
        split="eval",
        prompt="",
        target_smiles="CCN",
        source_smiles="CCO",
        property_count="2",
        source_tanimoto="0.5",
        source_similarity_bin="medium_similarity",
        metadata={"condition_id": "pair_1_cond_00_2p"},
    )
    fingerprint_dim = 4
    latent_dim = fingerprint_dim + 64 + 7
    latent = np.zeros((1, latent_dim), dtype=np.float32)
    latent[0, :fingerprint_dim] = [1.0, 0.0, 1.0, 0.0]
    latent[0, fingerprint_dim : fingerprint_dim + 7] = np.arange(7, dtype=np.float32)
    latent[0, fingerprint_dim + 32 : fingerprint_dim + 39] = np.arange(7, dtype=np.float32) - 3.0
    latent[0, fingerprint_dim + 64 : fingerprint_dim + 71] = 1.0

    summary = write_edit_latent_benchmark_inputs([sample], latent, tmp_path, fingerprint_dim=fingerprint_dim)

    exported = np.load(tmp_path / "edit_latent_predictions.npy")
    fingerprints = np.load(tmp_path / "edit_latent_fingerprints.npy")
    index = list(csv.DictReader((tmp_path / "index.csv").open(newline="", encoding="utf-8")))

    assert summary["rows"] == 1
    assert exported.shape == (1, 28)
    assert fingerprints.tolist() == [[1.0, 0.0, 1.0, 0.0]]
    assert index[0]["condition_id"] == "pair_1_cond_00_2p"
    assert exported[0, 0] == 0.0
    assert exported[0, 7] == -3.0
    assert exported[0, 21] == -1.0


def test_export_latent_benchmark_inputs_aligns_subset(tmp_path, monkeypatch):
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": f"edit:multiproperty_edit:pair_{idx}_cond_00_2p",
                        "task_type": EDIT_GENERATION,
                        "split": "eval",
                        "target_smiles": "CCN",
                        "source_smiles": "CCO",
                        "property_count": "2",
                        "source_tanimoto": "0.5",
                        "source_similarity_bin": "medium_similarity",
                        "metadata": {"condition_id": f"pair_{idx}_cond_00_2p"},
                    }
                )
                for idx in range(5)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    latents_path = tmp_path / "generated_latents.npy"
    np.save(latents_path, np.zeros((2, 80), dtype=np.float32))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"rows": 2, "fingerprint_dim": 4}), encoding="utf-8")
    predictions_path = tmp_path / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerow({"sample_id": "edit:multiproperty_edit:pair_1_cond_00_2p"})
        writer.writerow({"sample_id": "edit:multiproperty_edit:pair_0_cond_00_2p"})

    argv = [
        "export_latent_benchmark_inputs.py",
        "--eval-jsonl",
        str(eval_jsonl),
        "--latents-npy",
        str(latents_path),
        "--output-dir",
        str(tmp_path / "export"),
        "--metrics-json",
        str(metrics_path),
        "--predictions-csv",
        str(predictions_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_latent_benchmark_inputs.py"
    spec = importlib.util.spec_from_file_location("export_latent_benchmark_inputs", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    index = list(csv.DictReader((tmp_path / "export" / "index.csv").open(newline="", encoding="utf-8")))
    assert len(index) == 2
    assert index[0]["sample_id"] == "edit:multiproperty_edit:pair_1_cond_00_2p"
    assert index[1]["sample_id"] == "edit:multiproperty_edit:pair_0_cond_00_2p"
