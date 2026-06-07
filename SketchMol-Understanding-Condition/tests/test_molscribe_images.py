import importlib.util
from pathlib import Path

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
