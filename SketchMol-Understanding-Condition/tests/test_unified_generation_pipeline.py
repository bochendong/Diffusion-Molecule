import csv
import json

import torch

from sketchmol_understanding_condition.edit_condition_tokens import (
    EditConditionTokenConnector,
    edit_condition_loss,
)
from sketchmol_understanding_condition.latent_diffusion_generation import (
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_understanding_condition.unified_condition_dataset import (
    EDIT_GENERATION,
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

