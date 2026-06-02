import unittest

from sketchmol_token_diffusion.masked_token_diffusion import (
    MASK,
    _decode_ids,
    _decode_ids_to_smiles,
    _decode_length_limit_for_example,
    _ensure_mask_token,
    _normalize_decode_length_mode,
    _normalize_generation_tokenization,
)


class MaskedTokenDiffusionTests(unittest.TestCase):
    def test_adds_mask_token_once(self):
        stoi, itos = _ensure_mask_token({"<pad>": 0, "<bos>": 1, "<eos>": 2, "C": 3}, ["<pad>", "<bos>", "<eos>", "C"])
        self.assertIn(MASK, stoi)
        self.assertEqual(itos[stoi[MASK]], MASK)
        stoi2, itos2 = _ensure_mask_token(stoi, itos)
        self.assertEqual(stoi2, stoi)
        self.assertEqual(itos2, itos)

    def test_decode_ids_stops_at_eos_and_ignores_specials(self):
        itos = ["<pad>", "<bos>", "<eos>", "C", "O", MASK]
        self.assertEqual(_decode_ids([1, 3, 4, 2, 3, 5], itos), "CO")

    def test_decode_ids_to_smiles_keeps_smiles_tokens(self):
        itos = ["<pad>", "<bos>", "<eos>", "C", "O", MASK]
        self.assertEqual(_decode_ids_to_smiles([1, 3, 4, 2, 3, 5], itos, tokenization="smiles_token"), "CO")

    def test_normalizes_new_tokenization_and_decode_modes(self):
        self.assertEqual(_normalize_generation_tokenization("SELFIES"), "selfies")
        self.assertEqual(_normalize_generation_tokenization("smiles"), "smiles_token")
        self.assertEqual(_normalize_decode_length_mode("median"), "train_median")
        self.assertEqual(_normalize_decode_length_mode("oracle_target"), "oracle")

    def test_decode_length_limit_modes(self):
        example = {"target_length": 13}
        self.assertIsNone(_decode_length_limit_for_example(example, mode="free", train_decode_length=7))
        self.assertEqual(_decode_length_limit_for_example(example, mode="train_median", train_decode_length=7), 7)
        self.assertEqual(_decode_length_limit_for_example(example, mode="oracle", train_decode_length=7), 13)


if __name__ == "__main__":
    unittest.main()
