from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("rdkit")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import p16_protocol as protocol  # noqa: E402


def row(source: str = "") -> dict[str, str]:
    return {
        "source_smiles": source,
        "target_smiles": "CCO",
        "QED_active": "True",
        "QED_direction": "increase" if source else "",
        "QED_setting": "" if source else "0.61",
    }


def test_empty_source_naturally_expresses_generation_without_router():
    messages, source, mode = protocol.build_prompt(row())
    assert source == "<EMPTY>"
    assert mode == "de_novo"
    assert "task_mode" not in messages[-1]["content"]
    assert "target_smiles" not in messages[-1]["content"]


def test_populated_source_naturally_expresses_editing():
    messages, source, mode = protocol.build_prompt(row("C(C)O"))
    assert source == "CCO"
    assert mode == "edit"
    assert '"source":"CCO"' in messages[-1]["content"]


def test_response_schema_is_identical_and_strict():
    assert protocol.response("C(C)O", "de_novo") == '{"plan":"BUILD","smiles":"CCO"}'
    assert protocol.response("C(C)O", "edit") == '{"plan":"MODIFY","smiles":"CCO"}'
    assert protocol.parse_response('{"plan":"BUILD","smiles":"CCO"}', "de_novo")["valid"]
    assert not protocol.parse_response('answer: {"plan":"BUILD","smiles":"CCO"}', "de_novo")["strict_parse"]
    assert not protocol.parse_response('{"plan":"MODIFY","smiles":"CCO"}', "de_novo")["strict_parse"]
    assert not protocol.parse_response('{"plan":"BUILD","smiles":"not_smiles"}', "de_novo")["valid"]


def test_condition_builder_is_allowlist_only():
    dirty = row()
    dirty.update({"prompt": "SECRET", "instruction": "SECRET", "policy_target_smiles": "N#N", "oracle": "SECRET"})
    messages, _, _ = protocol.build_prompt(dirty)
    text = messages[-1]["content"]
    assert "SECRET" not in text
    assert "N#N" not in text


def test_split_hash_is_deterministic_and_source_sensitive():
    condition = protocol.condition_hash(row())
    assert protocol.split_score(condition, "a", 1616) == protocol.split_score(condition, "a", 1616)
    assert protocol.split_score(condition, "a", 1616) != protocol.split_score(condition, "b", 1616)
