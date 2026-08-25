from __future__ import annotations

import json

import train_unified_grpo as p21


def row(mode: str, conditions, source: str = "<EMPTY>"):
    payload = {"conditions": conditions, "source": source}
    return {
        "example_id": f"test:{mode}",
        "task_mode": mode,
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": json.dumps(payload)},
            {"role": "assistant", "content": '{"plan":"BUILD","smiles":"CCO"}'},
        ],
    }


def test_group_advantages_are_zero_mean_and_rank_rewards():
    advantages = p21.group_advantages([0.0, 1.0, 2.0, 3.0])
    assert abs(sum(advantages)) < 1e-7
    assert advantages == sorted(advantages)


def test_scorer_row_never_contains_target_smiles():
    item = row("de_novo", [{"property": "MW", "goal": {"around": 46.069}}])
    scored = p21.scorer_row(p21.prompt_payload(item), "de_novo")
    assert scored["target_MW"] == "46.069"
    assert "target_smiles" not in scored


def test_valid_program_match_beats_invalid_response():
    item = row("de_novo", [{"property": "MW", "goal": {"around": 46.069}}])
    valid, details = p21.reward_response(item, '{"plan":"BUILD","smiles":"CCO"}')
    invalid, invalid_details = p21.reward_response(item, '{"plan":"BUILD","smiles":"not-a-smiles"}')
    assert details["valid"] is True
    assert invalid_details["valid"] is False
    assert valid > invalid


def test_edit_reward_requires_modify_schema():
    item = row("edit", [{"property": "MW", "goal": "increase"}], source="CC")
    wrong, details = p21.reward_response(item, '{"plan":"BUILD","smiles":"CCC"}')
    assert wrong == -1.0
    assert details["valid"] is False
