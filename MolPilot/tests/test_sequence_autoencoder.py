import tempfile
import unittest

import numpy as np

from molpilot.autoencoder import load_autoencoder
from molpilot.sequence_autoencoder import MolecularSequenceAutoencoder, SequenceAutoencoderConfig


class SequenceAutoencoderTests(unittest.TestCase):
    def test_sequence_codec_round_trip_interface(self):
        smiles = ["CCO", "CCN", "c1ccccc1"]
        codec = MolecularSequenceAutoencoder(
            SequenceAutoencoderConfig(representation="smiles", latent_dim=16, hidden_dim=32, embedding_dim=16, epochs=1, max_length=32)
        )
        codec.fit(smiles)
        latents = codec.encode_many(smiles)
        self.assertEqual(latents.shape, (3, 16))
        decoded = codec.decode(latents[0], top_k=3)
        self.assertGreaterEqual(len(decoded), 1)

    def test_sequence_codec_save_load_factory(self):
        smiles = ["CCO", "CCN"]
        codec = MolecularSequenceAutoencoder(
            SequenceAutoencoderConfig(representation="smiles", latent_dim=12, hidden_dim=24, embedding_dim=12, epochs=1, max_length=24)
        )
        codec.fit(smiles)
        with tempfile.TemporaryDirectory() as tmp:
            codec.save(tmp)
            loaded = load_autoencoder(tmp)
            z = loaded.encode_many(smiles)
            self.assertEqual(z.shape, (2, 12))
            self.assertTrue(hasattr(loaded, "decode"))


if __name__ == "__main__":
    unittest.main()

