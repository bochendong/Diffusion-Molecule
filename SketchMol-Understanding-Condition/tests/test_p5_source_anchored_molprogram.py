from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "SketchMol-Understanding-Condition"
EXPERIMENT = PROJECT / "experiments" / "p5_source_anchored_molprogram"
UNIFIED = PROJECT / "experiments" / "unified_smiles_generator" / "unified_smiles_generator.py"


def test_p5_python_entrypoints_parse() -> None:
    for path in EXPERIMENT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_p5_is_one_direct_smiles_policy_without_inference_assistance() -> None:
    runner = (EXPERIMENT / "run_p5_source_anchored_molprogram.sh").read_text(encoding="utf-8")
    assert "--source-copy-aware" in runner
    assert "--source-adapter-layers 2" in runner
    assert runner.count("--trainable-scope source_only") == 2
    assert runner.count("--disable-finalizer") == 1
    assert "--num-samples 20" in runner
    assert "for budget in 1 8 20" in runner
    assert "--include-source-copy-candidate" not in runner
    assert "materializer" not in runner.lower()
    assert "router" not in runner.lower()


def test_p5_preregisters_raw_candidate_and_retention_gates() -> None:
    protocol = json.loads((EXPERIMENT / "p5_preregistration.json").read_text(encoding="utf-8"))
    assert protocol["seed"] == 7
    assert protocol["inference"]["single_policy"] is True
    assert protocol["inference"]["direct_smiles"] is True
    assert protocol["inference"]["property_reranking"] is False
    assert protocol["evaluation"]["candidate_budgets"] == [1, 8, 20]
    assert protocol["evaluation"]["report_candidate_level_n20"] is True
    assert protocol["gates"]["require_de_novo_path_bit_identical"] is True


def test_pointer_modules_are_edit_only_and_source_ids_are_threaded_through_generation() -> None:
    source = UNIFIED.read_text(encoding="utf-8")
    prefixes = source.split("SOURCE_ONLY_PREFIXES = (", 1)[1].split(")\n\n", 1)[0]
    for prefix in ("source_adapters.", "source_copy_query.", "source_copy_key.", "source_copy_gate."):
        assert prefix in prefixes
    assert "source_token_ids_for_condition" in source
    assert "copy_probs.scatter_add_" in source
    assert "source_present[:, None, None]" in source
    assert 'for key in ("condition", "condition_mask", "source_token_ids")' in source

