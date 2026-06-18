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
    assert Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator_rl.py").exists()
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
    generated = model.generate(
        condition,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        max_new_tokens=5,
        temperature=0.9,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=1,
        suppress_ids=[vocab.token_to_id["<unk>"]],
    )
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 2


def test_direct_smiles_property_rerank_prefers_strict_candidate(monkeypatch):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")

    module = _load_train_module()
    monkeypatch.setattr(module, "canonical_smiles", lambda value: str(value or "") or None)

    def fake_properties(smiles: str):
        if smiles == "far":
            return {"MolWt": 200.0, "LogP": 5.0}
        if smiles == "near":
            return {"MolWt": 101.0, "LogP": 2.1}
        return None

    monkeypatch.setattr(module, "molecular_properties", fake_properties)
    selected = module.select_generated_candidate(
        {
            "condition_properties": "MW,LogP",
            "target_MW": "100",
            "target_LogP": "2.0",
            "MW_active": "True",
            "LogP_active": "True",
        },
        ["far", "near"],
    )

    assert selected["generated_smiles"] == "near"
    assert selected["valid_candidate_count"] == 2
    assert selected["strict_fraction"] == 1.0


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
            "--num-samples",
            "2",
            "--temperature",
            "0.9",
            "--top-k",
            "4",
            "--top-p",
            "0.95",
            "--repetition-penalty",
            "1.1",
            "--no-repeat-ngram-size",
            "3",
            "--min-new-tokens",
            "1",
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
    assert prediction_rows[0]["method"] == "direct_smiles_mllm_sampled_rerank"
    assert "direct_valid_candidate_count" in prediction_rows[0]


def test_rl_sample_rollouts_preserves_prompt_grouping():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL rollout test")

    module = _load_rl_module()

    class FakeModel:
        def __init__(self) -> None:
            self.call_index = 0

        def generate(self, condition_tokens, **_: object):
            self.call_index += 1
            row_ids = condition_tokens[:, 0, 0].to(dtype=torch.long).cpu()
            call_ids = torch.full_like(row_ids, self.call_index)
            bos = torch.zeros_like(row_ids)
            return torch.stack([bos, row_ids, call_ids], dim=1)

    batch = {
        "condition": torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32),
        "condition_mask": torch.tensor([[True], [True]]),
    }
    generated = module.sample_rollouts(
        FakeModel(),
        batch,
        bos_id=0,
        eos_id=99,
        rollouts_per_prompt=3,
        parallel_samples=2,
        max_parallel_sequences=8,
        max_new_tokens=4,
        temperature=0.9,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=1,
        suppress_ids=[],
    )

    assert generated[:, 1].tolist() == [1, 1, 1, 2, 2, 2]
    assert generated[:, 2].tolist() == [1, 1, 2, 1, 1, 2]


def test_rl_reward_prefers_strict_and_valid_candidates(monkeypatch):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL reward test")

    module = _load_rl_module()
    monkeypatch.setattr(module, "_safe_canonical_smiles", lambda smiles: "" if smiles == "bad" else smiles)
    monkeypatch.setattr(
        module,
        "property_score_components",
        lambda _row, smiles: (1.0, 0.2) if smiles == "good" else (0.0, 0.1),
    )

    good = module.reward_for_smiles({}, "good", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)
    weak = module.reward_for_smiles({}, "weak", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)
    bad = module.reward_for_smiles({}, "bad", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)

    assert good > weak
    assert bad == -1.0


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


def _load_train_module():
    path = Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator.py")
    spec = importlib.util.spec_from_file_location("train_direct_smiles_generator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rl_module():
    path = Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator_rl.py")
    spec = importlib.util.spec_from_file_location("train_direct_smiles_generator_rl", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
