import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sketchmol_understanding_condition.molscribe_images import preprocess_image_for_molscribe


def _load_run_molscribe_ocr_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_molscribe_ocr.py"
    spec = importlib.util.spec_from_file_location("run_molscribe_ocr", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preprocess_image_for_molscribe_binarizes_soft_render():
    soft = np.full((32, 32, 3), 250, dtype=np.uint8)
    soft[8:24, 8:24] = 20
    processed = preprocess_image_for_molscribe(soft)

    assert processed.shape == (32, 32, 3)
    unique = {tuple(px) for px in processed.reshape(-1, 3)}
    assert unique.issubset({(0, 0, 0), (255, 255, 255)})


def test_preprocess_image_for_molscribe_maps_colored_atoms_to_black():
    colored = np.full((32, 32, 3), 255, dtype=np.uint8)
    colored[10:14, 10:14] = (0, 0, 255)
    colored[18:22, 18:22] = (255, 0, 0)
    processed = preprocess_image_for_molscribe(colored)

    assert np.any(processed == 0)
    assert np.all(processed[processed != 0] == 255)


def test_postprocess_smiles_sketchmol_filters_broken_and_keeps_low_score():
    module = _load_run_molscribe_ocr_module()

    smiles, broken_rate, low_score_rate = module._postprocess_smiles_sketchmol(
        ["CCO", "CC.CC", "CCN", "invalid"],
        [0.90, 0.90, 0.80, 0.90],
    )

    assert smiles == ["CCO", "", "CCN", ""]
    assert broken_rate == 0.25
    assert low_score_rate == 0.25


def test_prepend_sys_path_ordered_keeps_onmt_overlay_first():
    module = _load_run_molscribe_ocr_module()
    original = list(sys.path)
    overlay = Path("/tmp/onmt220")
    evaluate = Path("/tmp/SketchMol/evaluate")
    root = Path("/tmp/SketchMol")

    try:
        sys.path[:] = [str(evaluate), "/tmp/old"]
        module._prepend_sys_path_ordered([overlay, evaluate, root])

        assert sys.path[:3] == [str(overlay), str(evaluate), str(root)]
        assert sys.path.count(str(evaluate)) == 1
    finally:
        sys.path[:] = original


def test_select_smiles_prefers_graph_smiles():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("CCO", "CCN", allow_raw_fallback=True)

    assert smiles == "CCO"
    assert source == "graph"


def test_select_smiles_falls_back_to_valid_raw_token_smiles():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("<invalid>", "CCN", allow_raw_fallback=True)

    assert smiles == "CCN"
    assert source == "raw_token_fallback"


def test_select_smiles_rejects_invalid_raw_token_smiles_even_with_fallback():
    module = _load_run_molscribe_ocr_module()

    garbage = "C)C)C)C)C)"
    smiles, source = module._select_smiles("", garbage, allow_raw_fallback=True)

    assert smiles == ""
    assert source == "empty"


def test_select_smiles_returns_empty_when_both_paths_fail():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("", "<invalid>", allow_raw_fallback=True)

    assert smiles == ""
    assert source == "empty"


def test_predict_sketchmol_uses_path_reader_when_not_preprocessing():
    module = _load_run_molscribe_ocr_module()
    calls: list[str] = []

    class FakeModel:
        def predict_images_from_csv(self, paths, batch_size):
            calls.append("from_csv")
            return (["CCO"], ["molblock"], [0.9])

        def predict_imagespredict_images_from_csv_helper(self, input_images, batch_size):
            calls.append("helper")
            return (["CCN"], ["molblock"], [0.8])

    smiles, scores, diagnostics = module._predict_sketchmol(
        FakeModel(),
        ["/tmp/fake.png"],
        batch_size=1,
        preprocess_images=False,
    )

    assert calls == ["from_csv"]
    assert smiles == ["CCO"]
    assert scores == [0.9]
    assert diagnostics[0]["molscribe_decode_source"] == "sketchmol_graph"


def test_onmt220_attention_mask_broadcasts_for_batch_larger_than_heads():
    import torch

    from sketchmol_understanding_condition.molscribe_onmt_compat import (
        format_attention_mask_onmt220,
    )

    class FakeAttention:
        def forward(self, key, value, query, mask=None, layer_cache=None, attn_type=None):
            return None

    module = FakeAttention()
    batch_size = 16
    heads = 8
    src_len = 144
    src_pad_mask = torch.zeros(batch_size, 1, src_len, dtype=torch.bool)
    formatted = format_attention_mask_onmt220(module, src_pad_mask)
    assert formatted.shape == (batch_size, 1, src_len)

    scores = torch.zeros(batch_size, heads, 1, src_len)
    broadcast_mask = formatted.unsqueeze(1)
    torch.testing.assert_close(
        scores.masked_fill(broadcast_mask, -1e18),
        scores,
    )

    broken = src_pad_mask.squeeze(1).unsqueeze(1)
    try:
        scores.masked_fill(broken, -1e18)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_predict_sketchmol_uses_helper_when_preprocessing():
    module = _load_run_molscribe_ocr_module()
    calls: list[str] = []

    class FakeModel:
        def predict_images_from_csv(self, paths, batch_size):
            calls.append("from_csv")
            return (["CCO"], ["molblock"], [0.9])

        def predict_imagespredict_images_from_csv_helper(self, input_images, batch_size):
            calls.append("helper")
            return (["CCN"], ["molblock"], [0.8])

    with patch.object(
        module,
        "load_preprocessed_rgb_image",
        return_value=np.zeros((8, 8, 3), dtype=np.uint8),
    ):
        smiles, scores, _ = module._predict_sketchmol(
            FakeModel(),
            ["/tmp/fake.png"],
            batch_size=1,
            preprocess_images=True,
        )

    assert calls == ["helper"]
    assert smiles == ["CCN"]
    assert scores == [0.8]
