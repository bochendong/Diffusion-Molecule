from __future__ import annotations

import json
from pathlib import Path

from merge_audit import summarize_group


HERE = Path(__file__).resolve().parent


def candidate(*, strict: bool, valid: bool = True, advantage: float | None = 0.0):
    return {
        "strict": strict,
        "property_strict": strict,
        "valid": valid,
        "canonical": valid,
        "bottleneck": float(strict),
        "mean_satisfaction": float(strict),
        "source_similarity": 0.7 if strict else 0.2,
        "copy": False,
        "scalar_reward": 2.0 if strict else 0.5,
        "advantage": advantage,
        "raw": "success" if strict else "failure",
    }


def test_audit_contract_and_distillable_group():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    submit = (HERE / "submit_support_audit.sh").read_text()
    assert prereg["data"]["total_prompts"] == 960
    assert prereg["generation"]["sampled_candidates_per_prompt"] == 16
    assert prereg["training"] == "none"
    assert "for shard in 0 1 2 3" in submit
    record = {
        "sampled_advantage": {"zero_signal": False},
        "candidates": [
            candidate(strict=False, advantage=None),
            candidate(strict=True, advantage=1.0),
            *[candidate(strict=False, advantage=-1.0) for _ in range(15)],
        ],
    }
    summary = summarize_group(record)
    assert summary["any16_strict"] is True
    assert summary["sample1_strict"] is True
    assert summary["distillable"] is True
    assert summary["top_reward_strict"] is True
