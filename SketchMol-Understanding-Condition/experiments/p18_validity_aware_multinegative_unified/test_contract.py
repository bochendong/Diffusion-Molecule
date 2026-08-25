from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_multinegatives as build  # noqa: E402

RDLogger.DisableLog("rdApp.error")


def row(example_id: str, condition: str, target: str, source: str = ""):
    return {
        "example_id": example_id, "condition_hash": condition,
        "target_hash": f"hash:{target}", "target_smiles": target,
        "source_smiles": source,
    }


def test_preregistered_unified_contract_and_frozen_gate():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["frozen_before_training"] is True
    assert prereg["model_contract"]["base_count"] == 1
    assert prereg["model_contract"]["adapter_count"] == 1
    assert prereg["model_contract"]["mode_router"] is False
    assert prereg["training"]["source_copy_negative"] == {"modes": ["edit"], "margin": 0.10, "weight": 0.12}
    assert prereg["operational_gate"]["edit_greedy_validity"] == ">=0.875"


@pytest.mark.parametrize("mode,smiles", [("de_novo", "CCO"), ("edit", "c1ccccc1")])
def test_invalid_negative_is_strict_json_but_rdkit_invalid(mode, smiles):
    text = build.response(mode, smiles + "(")
    payload = json.loads(text)
    assert set(payload) == {"plan", "smiles"}
    assert Chem.MolFromSmiles(payload["smiles"]) is None


def test_mismatch_donor_is_different_target_condition_and_not_edit_source():
    current = row("a", "condition-a", "CCN", source="CCO")
    pool = [
        current,
        row("same-target", "condition-b", "CCN"),
        row("source-copy", "condition-c", "CCO"),
        row("valid", "condition-d", "CCC"),
    ]
    donor = build.donor_for(current, pool)
    assert donor["condition_hash"] != current["condition_hash"]
    assert donor["target_hash"] != current["target_hash"]
    assert donor["target_smiles"] != current["source_smiles"]


def test_one_negative_forward_contract_is_encoded_in_trainer():
    source = (HERE / "train_multinegative.py").read_text()
    assert source.count("rejected_out = model(**rejected)") == 1
    assert "negative_forwards_per_step\": 1" in source


def test_exact_p17_pilot_seeds_and_budgets_are_locked():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["pilot_benchmark"]["raw_budgets"] == [1, 4, 8]
    assert prereg["pilot_benchmark"]["generation_seeds"] == {"table1": 2717, "denovo": 3717}
    assert len(prereg["locked_p17_input_sha256"]) == 7
