import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sketchmol_unified_3m_diffusion.benchmark_export import write_edit_latent_benchmark_inputs
from sketchmol_unified_3m_diffusion.edit_condition_tokens import (
    EditConditionTokenConnector,
    edit_condition_loss,
    source_aware_fingerprint_losses,
)
from sketchmol_unified_3m_diffusion.latent_diffusion_generation import (
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (
    EDIT_GENERATION,
    EDIT_QUALITY_METADATA_FIELDS,
    UnifiedConditionSample,
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
                *EDIT_QUALITY_METADATA_FIELDS,
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
                "pair_quality_tier": "same_scaffold_medium_plus",
                "strict_candidate_count_t04": "2",
                "oracle_strict_success_t04": "True",
                "preservation_constraint": "keep_same_scaffold_or_source_tanimoto_ge_0_4",
                "target_QED": "0.6",
                "delta_QED": "0.1",
                "QED_active": "1",
                "QED_direction": "increase",
            }
        )

    samples = []
    samples.extend(read_3m_description_samples(description_tsv, split="train", dataset_name="toy"))
    samples.extend(
        read_edit_generation_samples(
            edit_csv,
            min_source_tanimoto=0.4,
            require_quality_columns=True,
            require_eval_oracle_strict=True,
        )
    )
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
    assert row["metadata"]["pair_quality_tier"] == "same_scaffold_medium_plus"
    assert summary["pair_quality_tiers"]["same_scaffold_medium_plus"] == 1
    assert summary["edit_source_tanimoto"]["min"] == 0.5


def test_unified_dataset_reads_moledit_enhanced_rows(tmp_path):
    moledit_csv = tmp_path / "eval_balanced.csv"
    with moledit_csv.open("w", newline="", encoding="utf-8") as handle:
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
                "example_id": "m1",
                "instruction": "Increase hydrogen bond acceptors and molecular weight, decrease QED.",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "pair_hash": "abc",
                "instruction_tasks": json.dumps(
                    [
                        {"property": "MW", "direction": "increase"},
                        {"property": "HBA", "direction": "increase"},
                        {"property": "QED", "direction": "decrease"},
                    ]
                ),
                "instruction_task_properties": "HBA|MW|QED",
                "instruction_task_directions": '{"HBA":"increase","MW":"increase","QED":"decrease"}',
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

    samples = read_moledit_generation_samples(moledit_csv, split="eval", table1_tasks_only=True)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "edit:moledit_instruct:m1"
    assert sample.split == "eval"
    assert sample.condition_properties == "MW,HBA,QED"
    assert sample.property_count == "3"
    assert sample.source_tanimoto == "0.55"
    assert sample.metadata["moledit_task_key"] == "HBA:increase+MW:increase+QED:decrease"


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
        source_fingerprint=torch.rand(3, 64),
        source_tanimoto=torch.tensor([0.7, 0.5, float("nan")]),
        similarity_bin=torch.zeros(3, dtype=torch.long),
        weights={"source_similarity_mse": 0.5, "source_aware_hard_negative": 0.25},
    )
    assert torch.isfinite(loss)
    assert "fingerprint_bce" in logs
    assert "source_similarity_mse" in logs
    assert "source_aware_hard_negative" in logs

    denoiser = EditLatentDenoiser(latent_dim=20, context_dim=16, hidden_dim=32, depth=2)
    diffusion = GaussianLatentDiffusion(denoiser, timesteps=8)
    diffusion_loss = diffusion.loss(torch.randn(3, 20), output.tokens, output.attention_mask)
    diffusion_loss_with_pred, pred_x0 = diffusion.loss_and_pred_x0(torch.randn(3, 20), output.tokens, output.attention_mask)

    assert torch.isfinite(diffusion_loss)
    assert torch.isfinite(diffusion_loss_with_pred)
    assert pred_x0.shape == (3, 20)


def test_source_guard_losses_penalize_over_editing():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_latent_diffusion_generation.py"
    spec = importlib.util.spec_from_file_location("train_latent_diffusion_generation", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fingerprint_dim = 4
    latent_dim = fingerprint_dim + 80
    target = torch.zeros(2, latent_dim)
    source = torch.zeros(2, latent_dim)
    target[:, :fingerprint_dim] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    source[:, :fingerprint_dim] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    source[:, fingerprint_dim : fingerprint_dim + 7] = 0.2
    pred = target.clone()
    pred[:, :fingerprint_dim] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pred[:, fingerprint_dim : fingerprint_dim + 7] = 0.8
    latent_mean = torch.zeros(1, latent_dim)
    latent_std = torch.ones(1, latent_dim)

    losses = module.source_guard_losses(
        pred,
        target,
        source,
        torch.tensor([0.8, 0.5]),
        fingerprint_dim=fingerprint_dim,
        source_regret_margin=0.0,
        source_radius_margin=0.0,
        source_similarity_weight_floor=0.25,
        fingerprint_guard_margin=0.0,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    assert losses["source_property_regret"] > 0
    assert losses["source_radius_regret"] > 0
    assert losses["source_property_worse_rate"].item() == 1.0
    assert losses["source_fingerprint_regret"] > 0
    assert losses["source_fingerprint_worse_rate"].item() == 1.0


def test_condition_prior_latent_can_source_anchor_fingerprint():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_latent_diffusion_generation.py"
    spec = importlib.util.spec_from_file_location("train_latent_diffusion_generation", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fingerprint_dim = 4
    latent_dim = fingerprint_dim + 80
    num_props = 7
    condition = SimpleNamespace(
        target_fingerprint_logits=torch.zeros(1, fingerprint_dim),
        target_properties=torch.zeros(1, num_props),
        property_deltas=torch.zeros(1, num_props),
        active_logits=torch.zeros(1, num_props),
    )
    connector_config = {
        "property_mean": [[0.0] * num_props],
        "property_std": [[1.0] * num_props],
        "delta_mean": [[0.0] * num_props],
        "delta_std": [[1.0] * num_props],
    }
    latent_mean = torch.zeros(1, latent_dim)
    latent_std = torch.ones(1, latent_dim)
    source_latent = torch.zeros(1, latent_dim)
    source_latent[:, :fingerprint_dim] = torch.tensor([1.0, 0.0, 1.0, 0.0])

    prior = module.condition_prior_latent(
        condition,
        connector_config=connector_config,
        latent_mean=latent_mean,
        latent_std=latent_std,
        fingerprint_dim=fingerprint_dim,
        latent_dim=latent_dim,
        source_latent=source_latent,
        source_tanimoto=torch.tensor([0.95]),
        source_fingerprint_prior_blend=1.0,
        source_similarity_weight_floor=1.0,
    )

    assert torch.equal(prior[:, :fingerprint_dim], source_latent[:, :fingerprint_dim])


def test_source_aware_fingerprint_losses_backprop_to_logits():
    logits = torch.randn(4, 16, requires_grad=True)
    losses = source_aware_fingerprint_losses(
        logits,
        target_properties=torch.randn(4, 7),
        property_deltas=torch.randn(4, 7),
        active_mask=torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        target_fingerprint=torch.rand(4, 16),
        source_fingerprint=torch.rand(4, 16),
        source_tanimoto=torch.tensor([0.7, 0.6, 0.5, float("nan")]),
        temperature=0.07,
        hard_negative_margin=0.2,
    )

    total = losses["source_similarity_mse"] + losses["source_aware_hard_negative"]
    total.backward()

    assert torch.isfinite(total)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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
    assert "oracle_strict_success_t04" in index[0]
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


def test_materialized_benchmark_runner_creates_prior_only_export_dir(tmp_path):
    output_dir = tmp_path / "unified_run"
    source_eval_dir = output_dir / "eval_latent"
    prior_eval_dir = output_dir / "eval_latent_prior_only"
    benchmark_dir = output_dir / "benchmark_materialized_prior_only_primary_fast"
    source_eval_dir.mkdir(parents=True)

    eval_jsonl = output_dir / "dataset" / "unified_condition_eval.jsonl"
    eval_jsonl.parent.mkdir(parents=True)
    eval_jsonl.write_text("{}\n", encoding="utf-8")
    (source_eval_dir / "prior_latents.npy").write_bytes(b"fake")
    (source_eval_dir / "metrics.json").write_text(json.dumps({"rows": 1}), encoding="utf-8")
    (source_eval_dir / "predictions.csv").write_text("sample_id\nsample-1\n", encoding="utf-8")
    condition_rows = tmp_path / "condition_rows.csv"
    condition_rows.write_text("condition_id\ncondition-1\n", encoding="utf-8")
    molecule_db = tmp_path / "molecule_database.csv"
    molecule_db.write_text("canonical_smiles\nCCO\n", encoding="utf-8")

    fake_python = tmp_path / "fake_python.py"
    fake_python.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

def value(flag):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except ValueError:
        raise SystemExit(f"missing {flag}")

script = sys.argv[1]
if script.endswith("export_latent_benchmark_inputs.py"):
    out = Path(value("--output-dir"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "edit_latent_predictions.npy").write_bytes(b"fake")
    (out / "edit_latent_fingerprints.npy").write_bytes(b"fake")
    (out / "index.csv").write_text("condition_id,sample_id\\ncondition-1,sample-1\\n", encoding="utf-8")
elif script.endswith("benchmark_multiproperty_retrieval.py"):
    out = Path(value("--output-dir"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_report.md").write_text("# report\\n", encoding="utf-8")
    (out / "benchmark_summary.csv").write_text("method,n\\nfake,1\\n", encoding="utf-8")
    (out / "benchmark_decoded.csv").write_text("condition_id\\ncondition-1\\n", encoding="utf-8")
else:
    raise SystemExit(f"unexpected script: {script}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_unified_materialized_benchmark.sh"
    env = {
        **os.environ,
        "SMU3M_PYTHON_BIN": str(fake_python),
        "SMU3M_OUTPUT_DIR": str(output_dir),
        "SMU3M_EVAL_LATENT_DIR": str(prior_eval_dir),
        "SMU3M_GENERATED_LATENTS": str(source_eval_dir / "prior_latents.npy"),
        "SMU3M_EVAL_METRICS": str(source_eval_dir / "metrics.json"),
        "SMU3M_EVAL_PREDICTIONS": str(source_eval_dir / "predictions.csv"),
        "SMU3M_EVAL_JSONL": str(eval_jsonl),
        "SMMED_CONDITION_ROWS": str(condition_rows),
        "SMMED_MOLECULE_DB_CSV": str(molecule_db),
        "SMU3M_BENCHMARK_OUTPUT_DIR": str(benchmark_dir),
        "SMU3M_BENCHMARK_PROFILE": "primary_fast",
        "SMMED_LIMIT_EVAL_ROWS": "1",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Waiting for benchmark export lock" not in result.stdout
    assert (prior_eval_dir / "edit_latent_predictions.npy").exists()
    assert (prior_eval_dir / "edit_latent_fingerprints.npy").exists()
    assert (prior_eval_dir / "index.csv").exists()
    assert (benchmark_dir / "benchmark_report.md").exists()


def test_eval_sampling_keeps_multiple_property_counts():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_latent_diffusion_generation.py"
    spec = importlib.util.spec_from_file_location("evaluate_latent_diffusion_generation", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    samples = [
        UnifiedConditionSample(
            sample_id=f"s{count}_{idx}",
            task_type=EDIT_GENERATION,
            split="eval",
            prompt="",
            target_smiles="CCN",
            property_count=str(count),
        )
        for count in (2, 3, 4)
        for idx in range(5)
    ]

    selected = module._sample_by_property_count(samples, 2, seed=7)
    by_count = {}
    for sample in selected:
        by_count[sample.property_count] = by_count.get(sample.property_count, 0) + 1

    assert by_count == {"2": 2, "3": 2, "4": 2}


def test_preflight_rejects_low_source_tanimoto_and_eval_oracle_failures(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_unified_3m.py"
    spec = importlib.util.spec_from_file_location("preflight_unified_3m", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = tmp_path / "manifest.csv"
    fieldnames = [
        "split",
        "source_smiles",
        "target_smiles",
        "source_tanimoto",
        "source_similarity_bin",
        *EDIT_QUALITY_METADATA_FIELDS,
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "eval",
                "source_smiles": "CCO",
                "target_smiles": "CCN",
                "source_tanimoto": "0.5",
                "source_similarity_bin": "medium_similarity",
                "pair_quality_tier": "same_scaffold_medium_plus",
                "strict_candidate_count_t04": "1",
                "oracle_strict_success_t04": "True",
                "preservation_constraint": "keep_same_scaffold_or_source_tanimoto_ge_0_4",
            }
        )

    summary = module._check_edit_manifest(
        manifest,
        min_source_tanimoto=0.4,
        require_quality_columns=True,
        require_eval_oracle_strict=True,
    )
    assert summary["usable_rows"] == 1
    assert summary["min_source_tanimoto"] == 0.5

    bad_manifest = tmp_path / "bad_manifest.csv"
    with bad_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "eval",
                "source_smiles": "CCO",
                "target_smiles": "CCCCCCCC",
                "source_tanimoto": "0.2",
                "source_similarity_bin": "exploratory_low_similarity",
                "pair_quality_tier": "rejected_too_distant",
                "strict_candidate_count_t04": "0",
                "oracle_strict_success_t04": "False",
                "preservation_constraint": "keep_source_tanimoto_ge_0_4",
            }
        )

    try:
        module._check_edit_manifest(
            bad_manifest,
            min_source_tanimoto=0.4,
            require_quality_columns=True,
            require_eval_oracle_strict=True,
        )
    except SystemExit as exc:
        assert "source-neighbor floor" in str(exc)
    else:
        raise AssertionError("Expected low-source-tanimoto manifest to fail preflight")
