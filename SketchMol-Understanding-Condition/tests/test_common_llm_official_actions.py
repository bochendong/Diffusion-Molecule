from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
POLICY_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (SCRIPT_DIR, POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = load_module("evaluate_common_llm_official_actions", SCRIPT_DIR / "evaluate_common_llm_official_actions.py")
external = load_module("select_external_verifier_prefix", SCRIPT_DIR / "select_external_verifier_prefix.py")


def test_prompt_uses_constraint_ir_without_target_smiles() -> None:
    messages = official.prompt_messages(
        {
            "condition_id": "x1",
            "benchmark_task": "moledit_table1",
            "source_smiles": "CCO",
            "target_smiles": "do-not-leak",
            "condition_properties": "QED",
            "QED_direction": "increase",
            "instruction": "increase QED",
        }
    )
    payload = json.loads(messages[1]["content"])
    assert payload["constraint_ir"]["source_smiles"] == "CCO"
    assert "do-not-leak" not in messages[1]["content"]
    assert payload["constraint_ir"]["constraints"][0]["direction"] == 1


def test_local_verifier_prefers_strict_candidate_without_target_fields() -> None:
    weak = {
        "candidate_rank": 1,
        "table1_strict_success": "False",
        "table1_instruction_success": "True",
        "source_similarity_success": "False",
        "unified_property_success_fraction": 1.0,
        "source_tanimoto": 0.3,
        "target_smiles": "CCO",
    }
    strict = {
        "candidate_rank": 2,
        "table1_strict_success": "True",
        "table1_instruction_success": "True",
        "source_similarity_success": "True",
        "unified_property_success_fraction": 1.0,
        "source_tanimoto": 0.7,
        "target_smiles": "unrelated",
    }
    assert max([weak, strict], key=official.verifier_key) is strict


def test_external_selector_honors_prefix_and_official_vector() -> None:
    rows = [
        {
            "condition_id": "a",
            "candidate_rank": "1",
            "external_strict_success": "False",
            "external_all_property_success": "False",
            "external_source_similarity_success": "True",
            "external_property_success_json": '{"bbbp": true, "qed": false}',
        },
        {
            "condition_id": "a",
            "candidate_rank": "2",
            "external_strict_success": "True",
            "external_all_property_success": "True",
            "external_source_similarity_success": "True",
            "external_property_success_json": '{"bbbp": true, "qed": true}',
        },
    ]
    at_one = external.select_rows(rows, budget=1, selection_mode="verifier", group_column="condition_id")
    at_two = external.select_rows(rows, budget=2, selection_mode="verifier", group_column="condition_id")
    assert at_one[0]["candidate_rank"] == "1"
    assert at_two[0]["candidate_rank"] == "2"
    assert at_two[0]["oracle_call_type"] == "admet_ai_tdc_vector"


def test_candidate_pool_widens_only_until_full(monkeypatch) -> None:
    calls = []

    class Policy:
        @staticmethod
        def enumerate_action_candidates(row, *, site_limit, max_actions_per_row):
            calls.append(max_actions_per_row)
            count = 12 if max_actions_per_row == 64 else 25
            return [(object(), f"C{idx}", ["x"]) for idx in range(count)]

    monkeypatch.setattr(official.constrained, "policy_module", lambda: Policy)
    pool, used = official.candidate_pool(
        {"source_smiles": "CC"},
        candidate_budget=20,
        initial_attempt_budget=64,
        max_attempt_budget=512,
        site_limit=32,
    )
    assert calls == [64, 128]
    assert used == 128
    assert len(pool) == 20
