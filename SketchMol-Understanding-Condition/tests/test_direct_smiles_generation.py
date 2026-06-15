from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def test_direct_smiles_entrypoints_exist():
    assert Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_2p7p_benchmark.sh").exists()


def test_smiles_tokenization_roundtrip():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import direct_smiles_generation")
    from sketchmol_understanding_condition.direct_smiles_generation import detokenize_smiles, tokenize_smiles

    smiles = "CC(Cl)Br"
    tokens = tokenize_smiles(smiles)
    assert tokens == ["C", "C", "(", "Cl", ")", "Br"]
    assert detokenize_smiles(tokens) == smiles


def test_conditioned_smiles_decoder_forward_and_generate():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the direct SMILES decoder")
    from sketchmol_understanding_condition.direct_smiles_generation import ConditionedSmilesDecoder, build_vocabulary, tokenize_smiles

    vocab = build_vocabulary(["CCO", "CCN"])
    model = ConditionedSmilesDecoder(
        vocab_size=len(vocab.token_to_id),
        condition_dim=16,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        pad_id=vocab.pad_id,
        max_length=16,
    )
    condition = torch.randn(2, 3, 16)
    decoder_input = torch.tensor(
        [
            vocab.encode(tokenize_smiles("CCO"), add_bos=True),
            vocab.encode(tokenize_smiles("CCN"), add_bos=True),
        ],
        dtype=torch.long,
    )
    logits = model(condition, decoder_input)
    assert logits.shape == (2, decoder_input.shape[1], len(vocab.token_to_id))
    generated = model.generate(condition, bos_id=vocab.bos_id, eos_id=vocab.eos_id, max_new_tokens=5)
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 2


def test_train_direct_smiles_generator_smoke(tmp_path):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the direct SMILES training smoke test")

    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"
    prediction_csv = tmp_path / "predictions.csv"
    rows = [
        {
            "sample_id": "a",
            "condition_id": "a",
            "target_smiles": "CCO",
            "condition_properties": "MW,LogP",
            "property_count": "2",
            "target_MW": "46",
            "target_LogP": "0.0",
            "MW_active": "True",
            "LogP_active": "True",
        },
        {
            "sample_id": "b",
            "condition_id": "b",
            "target_smiles": "CCN",
            "condition_properties": "MW,LogP",
            "property_count": "2",
            "target_MW": "45",
            "target_LogP": "0.1",
            "MW_active": "True",
            "LogP_active": "True",
        },
    ]
    _write_rows(train_csv, rows)
    _write_rows(eval_csv, rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_direct_smiles_generator.py",
            "--train-csv",
            str(train_csv),
            "--eval-csv",
            str(eval_csv),
            "--output-dir",
            str(output_dir),
            "--prediction-csv",
            str(prediction_csv),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--d-model",
            "32",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--dim-feedforward",
            "64",
            "--max-smiles-length",
            "16",
            "--max-new-tokens",
            "8",
            "--device",
            "cpu",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "direct_smiles_generator.pt").exists()
    prediction_rows = list(csv.DictReader(prediction_csv.open(newline="", encoding="utf-8")))
    assert len(prediction_rows) == 2
    assert "generated_smiles" in prediction_rows[0]
    assert prediction_rows[0]["method"] == "direct_smiles_mllm"


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
