from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("rdkit")


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "SketchMol-Unified-3MDiffusion" / "scripts" / "evaluate_moledit_table_metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("moledit_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureModel:
    def __init__(self):
        self.features = None

    def predict_proba(self, features):
        self.features = np.asarray(features)
        return np.asarray([[0.25, 0.75]], dtype=np.float64)


def test_drd2_uses_graph2graph_feature_count_fingerprint():
    module = load_module()
    oracle = object.__new__(module.PinnedMorganClassifierOracle)
    oracle.prop = "DRD2"
    oracle.model = CaptureModel()
    score = oracle("CCOc1ccccc1")
    assert score == 0.75
    assert oracle.model.features.shape == (1, 2048)
    assert oracle.model.features.dtype == np.float32
    assert float(oracle.model.features.sum()) > 0.0
    assert float(oracle.model.features.max()) >= 1.0


def test_gsk3b_keeps_ecfp4_bit_vector_features():
    module = load_module()
    oracle = object.__new__(module.PinnedMorganClassifierOracle)
    oracle.prop = "GSK3B"
    oracle.model = CaptureModel()
    oracle("CCOc1ccccc1")
    assert set(np.unique(oracle.model.features)).issubset({0.0, 1.0})
