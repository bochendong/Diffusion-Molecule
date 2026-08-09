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
existing = load_module(
    "rerank_common_llm_existing_action_plans",
    SCRIPT_DIR / "rerank_common_llm_existing_action_plans.py",
)


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


def test_reconstructs_two_step_candidate_plan() -> None:
    first = {
        "step": 1,
        "parent_smiles": "CC",
        "generated_smiles": "CCO",
        "action": {"op": "add_atom", "site": 1, "atom": "O", "prop": "qed"},
    }
    second = {
        "step": 2,
        "parent_smiles": "CCO",
        "generated_smiles": "OCCF",
        "action": {"op": "add_atom", "site": 2, "atom": "F", "prop": "bbbp"},
    }
    plans = existing.reconstruct_condition_plans(
        ["CCO", "OCCF"],
        {"CCO": first, "OCCF": second},
    )
    assert [action["atom"] for action in plans["CCO"]] == ["O"]
    assert [action["atom"] for action in plans["OCCF"]] == ["O", "F"]


def test_existing_pool_summary_separates_llm_and_verifier() -> None:
    rows = []
    for condition_id, split in (("a", "ind"), ("b", "ood")):
        rows.extend(
            [
                {
                    "condition_id": condition_id,
                    "external_task_split": split,
                    "candidate_rank": 1,
                    "external_official_success": "False",
                    "external_strict_success": "False",
                    "external_source_similarity_success": "True",
                    "external_property_success_json": '{"qed": false}',
                },
                {
                    "condition_id": condition_id,
                    "external_task_split": split,
                    "candidate_rank": 2,
                    "external_official_success": "True",
                    "external_strict_success": "True",
                    "external_source_similarity_success": "True",
                    "external_property_success_json": '{"qed": true}',
                    "external_source_tanimoto": 0.8,
                },
            ]
        )
    summary = existing.summarize(rows, rows, verifier_k=2, candidate_budget=2)
    selections = summary["selections"]
    assert selections["llm_at_1"]["all"]["success_rate"] == 0.0
    assert selections["original_heuristic_verifier_at_2"]["all"]["success_rate"] == 1.0
    assert selections["llm_verifier_at_2"]["all"]["success_rate"] == 1.0
    assert summary["verifier_recovery_of_reachable"] == 1.0
    assert summary["original_verifier_recovery_of_reachable"] == 1.0
    paired = summary["paired_comparisons"]["llm_at_1_vs_original_heuristic_at_1"]
    assert paired["external_official_success"]["delta"] == 0.0


def test_joint_plan_scoring_uses_one_payload_per_candidate(monkeypatch) -> None:
    encoded_payloads = []

    monkeypatch.setattr(existing.plan_protocol, "plan_prompt_messages", lambda row: [{"role": "user", "content": "x"}])

    def encoded_action(tokenizer, messages, payload, *, max_length):
        encoded_payloads.append(payload)
        return {"input_ids": [1], "attention_mask": [1], "labels": [1]}

    monkeypatch.setattr(existing.constrained, "encoded_action", encoded_action)
    monkeypatch.setattr(
        existing.constrained,
        "score_encoded_actions",
        lambda model, tokenizer, encoded, *, batch_size: [0.1, 0.9],
    )
    rows = [
        {"condition_id": "x", "generated_smiles": "CCN", "candidate_rank": "1"},
        {"condition_id": "x", "generated_smiles": "CCF", "candidate_rank": "2"},
    ]
    plans = {
        ("x", "CCN"): [
            {"op": "add_atom", "site": 1, "atom": "C"},
            {"op": "replace_atom", "site": 2, "atom": "N"},
        ],
        ("x", "CCF"): [{"op": "replace_atom", "site": 1, "atom": "F"}],
    }

    ranked = existing.score_condition(
        rows,
        plans,
        model=object(),
        tokenizer=object(),
        batch_size=2,
        max_length=128,
        score_mode="joint_plan_logprob",
        variant="test_joint_plan",
    )

    assert len(encoded_payloads) == 2
    assert encoded_payloads[0]["action_type"] == "graph_edit_plan"
    assert len(encoded_payloads[0]["value"]["steps"]) == 2
    assert ranked[0]["generated_smiles"] == "CCF"
    assert ranked[0]["method"] == "test_joint_plan"
