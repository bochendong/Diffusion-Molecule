import tempfile
import unittest

import numpy as np

from molpilot.condition_model import load_condition_model, predict_condition_latents
from molpilot.jepa import JEPAConfig, MolecularJEPAPredictor
from molpilot.schema import GenerationRequest, TaskType


class DummyAutoencoder:
    def encode_many(self, smiles):
        out = np.zeros((len(smiles), 4), dtype=np.float32)
        for idx, smi in enumerate(smiles):
            if smi:
                out[idx, idx % 4] = 1.0
        return out


class JEPATests(unittest.TestCase):
    def test_jepa_predictor_save_load_interface(self):
        conditions = np.eye(5, 6, dtype=np.float32)
        source = np.zeros((5, 4), dtype=np.float32)
        source[:3, 0] = 1.0
        target = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
        model = MolecularJEPAPredictor(JEPAConfig(hidden_dim=16, layers=1, epochs=1, batch_size=2))
        model.fit(conditions, target, source)
        pred = model.predict(conditions, source)
        self.assertEqual(pred.shape, target.shape)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = load_condition_model(tmp)
            pred2 = loaded.predict(conditions, source)
            self.assertEqual(pred2.shape, target.shape)

    def test_predict_condition_latents_uses_source_latents_for_jepa(self):
        conditions = np.eye(2, 6, dtype=np.float32)
        source = np.zeros((2, 4), dtype=np.float32)
        target = np.ones((2, 4), dtype=np.float32)
        model = MolecularJEPAPredictor(JEPAConfig(hidden_dim=16, layers=1, epochs=1, batch_size=2))
        model.fit(conditions, target, source)
        pairs = [
            (GenerationRequest(task_type=TaskType.EDIT, source_smiles="CCO", instruction="Lower LogP."), "CCN"),
            (GenerationRequest(task_type=TaskType.DE_NOVO, instruction="Generate a drug-like molecule."), "CCO"),
        ]
        pred = predict_condition_latents(model, conditions, pairs, DummyAutoencoder())
        self.assertEqual(pred.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
