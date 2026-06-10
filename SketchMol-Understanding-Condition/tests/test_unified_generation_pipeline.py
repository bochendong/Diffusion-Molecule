import csv
import json
import subprocess
import sys

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
    read_moledit_generation_samples,
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


def test_unified_dataset_reads_moledit_enhanced_rows(tmp_path):
    moledit_csv = tmp_path / "eval_balanced.csv"
    _write_moledit_split(moledit_csv)

    samples = read_moledit_generation_samples(moledit_csv, split="eval", table1_tasks_only=True)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "edit:moledit_instruct:m1"
    assert sample.split == "eval"
    assert sample.condition_properties == "MW,HBA,QED"
    assert sample.property_count == "3"
    assert sample.source_tanimoto == "0.55"
    assert sample.metadata["moledit_task_key"] == "HBA:increase+MW:increase+QED:decrease"


def test_export_univideo_edit_dataset_from_moledit_splits(tmp_path):
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval_balanced.csv"
    _write_moledit_split(train_csv)
    _write_moledit_split(eval_csv, example_id="m2")
    output_dir = tmp_path / "dataset"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_univideo_edit_dataset.py",
            "--moledit-train-split",
            str(train_csv),
            "--moledit-eval-split",
            str(eval_csv),
            "--output-dir",
            str(output_dir),
            "--variants",
            "full,text_only",
            "--moledit-table1-tasks-only",
        ],
        cwd="SketchMol-Understanding-Condition",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    train_row = json.loads((output_dir / "univideo_edit_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    eval_row = json.loads((output_dir / "univideo_edit_eval.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train_row["metadata"]["condition_id"] == "m1"
    assert eval_row["metadata"]["condition_id"] == "m2"
    assert eval_row["condition_properties"] == "MW,HBA,QED"
    with (output_dir / "baseline_variants.csv").open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == 4
    assert {row["variant"] for row in baseline_rows} == {"full", "text_only"}
    assert {row["condition_id"] for row in baseline_rows} == {"m1", "m2"}


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


def _write_moledit_split(path, *, example_id: str = "m1") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "instruction",
                "source_smiles",
                "target_smiles",
                "pair_hash",
                "instruction_tasks",
                "instruction_task_properties",
                "instruction_task_directions",
                "source_target_tanimoto",
                "difficulty_bucket",
                "pair_quality",
                "computed_active_properties",
                "computed_active_count",
                "source_MW",
                "target_MW",
                "delta_MW",
                "MW_active",
                "MW_direction",
                "source_QED",
                "target_QED",
                "delta_QED",
                "QED_active",
                "QED_direction",
                "source_HBA",
                "target_HBA",
                "delta_HBA",
                "HBA_active",
                "HBA_direction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "example_id": example_id,
                "instruction": "Increase hydrogen bond acceptors and molecular weight, decrease QED.",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "pair_hash": f"pair-{example_id}",
                "instruction_tasks": json.dumps(
                    [
                        {"property": "MW", "direction": "increase"},
                        {"property": "Haccept", "direction": "↑"},
                        {"property": "QED", "direction": "↓"},
                    ]
                ),
                "instruction_task_properties": "Haccept|MW|QED",
                "instruction_task_directions": '{"Haccept":"↑","MW":"increase","QED":"↓"}',
                "source_target_tanimoto": "0.55",
                "difficulty_bucket": "medium_similarity",
                "pair_quality": "cross_scaffold_medium_similarity",
                "computed_active_properties": "MW|QED|HBA",
                "computed_active_count": "3",
                "source_MW": "46.0",
                "target_MW": "45.0",
                "delta_MW": "-1.0",
                "MW_active": "1",
                "MW_direction": "increase",
                "source_QED": "0.5",
                "target_QED": "0.4",
                "delta_QED": "-0.1",
                "QED_active": "1",
                "QED_direction": "decrease",
                "source_HBA": "1",
                "target_HBA": "2",
                "delta_HBA": "1",
                "HBA_active": "1",
                "HBA_direction": "increase",
            }
        )
