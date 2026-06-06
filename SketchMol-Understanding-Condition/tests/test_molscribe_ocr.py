import importlib.util
from pathlib import Path


def _load_run_molscribe_ocr_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_molscribe_ocr.py"
    spec = importlib.util.spec_from_file_location("run_molscribe_ocr", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_smiles_prefers_graph_smiles():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("CCO", "CCN")

    assert smiles == "CCO"
    assert source == "graph"


def test_select_smiles_falls_back_to_raw_token_smiles():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("<invalid>", "CCN")

    assert smiles == "CCN"
    assert source == "raw_token_fallback"


def test_select_smiles_returns_empty_when_both_paths_fail():
    module = _load_run_molscribe_ocr_module()

    smiles, source = module._select_smiles("", "<invalid>")

    assert smiles == ""
    assert source == "empty"
