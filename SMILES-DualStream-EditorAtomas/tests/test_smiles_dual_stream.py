from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.config import parse_simple_yaml
from smiles_dual_stream.data import read_smiles_pairs
from smiles_dual_stream.edit_corruption import levenshtein_distance, make_edit_example
from smiles_dual_stream.featurize import build_dual_stream_example
from smiles_dual_stream.hierarchy import adaptive_polymerization_levels, hierarchy_alignment
from smiles_dual_stream.tokenization import detokenize_smiles, tokenize_smiles


class SmilesDualStreamTests(unittest.TestCase):
    def test_simple_yaml_config_parser(self) -> None:
        config = parse_simple_yaml(
            """
            train:
              epochs: 50
              lr: 0.0005
              resume: true
              limit: null
            """
        )
        self.assertEqual(config["train"]["epochs"], 50)
        self.assertEqual(config["train"]["lr"], 0.0005)
        self.assertIs(config["train"]["resume"], True)
        self.assertIsNone(config["train"]["limit"])

    def test_tokenizer_keeps_common_atom_units(self) -> None:
        tokens = tokenize_smiles("Clc1cc(Br)ccc1")
        self.assertIn("Cl", tokens)
        self.assertIn("Br", tokens)
        self.assertEqual(detokenize_smiles(tokens), "Clc1cc(Br)ccc1")

    def test_fragment_corruption_has_edit_supervision(self) -> None:
        example = make_edit_example("CC(=O)O", seed=3, identity_probability=0.0, policies=("mask",))
        self.assertNotEqual(example.clean_smiles, example.corrupted_smiles)
        self.assertGreaterEqual(levenshtein_distance(example.corrupted_tokens, example.clean_tokens), 1)
        self.assertTrue(example.operations)

    def test_hierarchy_alignment_is_dependency_free(self) -> None:
        levels = adaptive_polymerization_levels("c1ccccc1O")
        self.assertIn("atom", levels)
        self.assertIn("fragment", levels)
        alignment = hierarchy_alignment("CCO", "CCN")
        self.assertGreaterEqual(alignment["token_jaccard"], 0.0)
        self.assertLessEqual(alignment["token_jaccard"], 1.0)

    def test_pair_csv_and_dual_stream_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "pairs.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "source_smiles",
                        "target_smiles",
                        "instruction",
                        "source_image",
                        "img_mean",
                        "hist_0",
                        "patch_mean_max",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "x1",
                        "source_smiles": "CCO",
                        "target_smiles": "CCN",
                        "instruction": "replace O with N",
                        "source_image": "legacy.png",
                        "img_mean": "0.5",
                        "hist_0": "0.1",
                        "patch_mean_max": "0.9",
                    }
                )
            pairs = read_smiles_pairs(csv_path)
            self.assertNotIn("source_image", pairs[0].metadata)
            self.assertNotIn("img_mean", pairs[0].metadata)
            self.assertNotIn("hist_0", pairs[0].metadata)
            self.assertNotIn("patch_mean_max", pairs[0].metadata)
            example = build_dual_stream_example(pairs[0], seed=5)
            self.assertEqual(example.mode, "pair_edit")
            self.assertEqual(example.corrupted_smiles, "CCO")
            self.assertEqual(example.target_smiles, "CCN")
            self.assertIn("fragment_jaccard", example.alignment)

    def test_optimizer_step_updates_weights_with_fp16_autocast(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from smiles_dual_stream.model import SmilesDualStreamModel
        from smiles_dual_stream.train import _collate, _optimizer_step
        from smiles_dual_stream.tokenization import BOS, EOS, PAD

        pad_id = 0
        bos_id = 1
        eos_id = 2
        model = SmilesDualStreamModel(8, embed_dim=16, hidden_dim=32, pad_id=pad_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        batch = _collate(
            [
                {
                    "input_ids": [3, 4, eos_id],
                    "decoder_input_ids": [bos_id, 5, 6],
                    "target_ids": [5, 6, eos_id],
                }
            ],
            pad_id=pad_id,
            device=device,
        )
        before = next(model.parameters()).detach().clone()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(
                batch["input_ids"],
                batch["decoder_input_ids"],
                batch["target_ids"],
                reconstruction_loss_weight=1.0,
                alignment_loss_weight=0.3,
                molecule_alignment_weight=1.0,
                token_alignment_weight=0.1,
                fragment_alignment_weight=0.2,
            )
            loss = output["loss"]
        scaler.scale(loss).backward()
        skipped = _optimizer_step(model, optimizer, scaler, grad_clip=1.0)
        after = next(model.parameters()).detach()
        self.assertEqual(skipped, 0)
        self.assertFalse(torch.equal(before.to(after.dtype), after.cpu() if after.device.type != "cpu" else after))


if __name__ == "__main__":
    unittest.main()
