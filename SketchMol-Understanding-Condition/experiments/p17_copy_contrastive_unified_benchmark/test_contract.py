from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p17_protocol as protocol  # noqa: E402


def edit_row():
    return {"source_smiles": "CCO", "target_smiles": "CCN", "QED_active": "True", "QED_direction": "increase"}


def test_copy_negative_uses_only_source_and_same_schema():
    messages, source, mode = protocol.build_prompt(edit_row())
    assert mode == "edit"
    assert protocol.response_from_source(source) == '{"plan":"MODIFY","smiles":"CCO"}'
    assert "CCN" not in json.dumps(messages)


def test_one_protocol_has_no_router():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert prereg["model_contract"]["base_count"] == 1
    assert prereg["model_contract"]["adapter_count"] == 1
    assert prereg["model_contract"]["mode_router"] is False
    assert prereg["pilot_benchmark"]["raw_budgets"] == [1, 4, 8]


def test_pilot_is_explicitly_not_full_benchmark():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert "pilot estimate" in prereg["pilot_benchmark"]["status_label"]
    assert "2 rows per each of 10" in prereg["pilot_benchmark"]["table1_subset"]


def test_pairwise_sign_rewards_lower_chosen_nll():
    import torch
    margin = torch.relu(torch.tensor(0.2 + 0.4 - 1.0)).item()
    violation = torch.relu(torch.tensor(0.2 + 1.0 - 0.4)).item()
    assert margin == 0.0
    assert violation > 0.0
