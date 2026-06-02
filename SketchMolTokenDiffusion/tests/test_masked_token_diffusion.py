import unittest

from sketchmol_token_diffusion.masked_token_diffusion import MASK, _decode_ids, _ensure_mask_token


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


if __name__ == "__main__":
    unittest.main()
