#!/usr/bin/env python3
"""Merge P31 support shards and export an audited distillation preview."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / max(len(values), 1)


def strict(candidate: Mapping[str, object]) -> bool:
    return bool(candidate.get("strict"))


def chosen_key(candidate: Mapping[str, object]):
    return (
        int(strict(candidate)),
        int(bool(candidate.get("property_strict"))),
        int(bool(candidate.get("valid"))),
        float(candidate.get("bottleneck", 0.0)),
        float(candidate.get("mean_satisfaction", 0.0)),
        float(candidate.get("source_similarity", 0.0) or 0.0),
        int(bool(candidate.get("canonical"))),
        -int(bool(candidate.get("copy"))),
        float(candidate.get("scalar_reward", 0.0)),
    )


def summarize_group(record: Mapping[str, object]) -> dict[str, object]:
    candidates = list(record["candidates"])
    greedy = candidates[0]
    sampled = candidates[1:]
    successes = [candidate for candidate in sampled if strict(candidate)]
    top_reward = max(
        sampled,
        key=lambda candidate: (
            float(candidate.get("advantage", 0.0) or 0.0),
            float(candidate.get("scalar_reward", 0.0)),
        ),
    )
    zero_signal = bool(record.get("sampled_advantage", {}).get("zero_signal"))
    return {
        "greedy_strict": strict(greedy),
        "greedy_valid": bool(greedy.get("valid")),
        "sample1_strict": strict(sampled[0]),
        "sample1_valid": bool(sampled[0].get("valid")),
        "any16_strict": bool(successes),
        "sampled_strict_count": len(successes),
        "sampled_valid_rate": mean(bool(candidate.get("valid")) for candidate in sampled),
        "frontier": 0 < len(successes) < len(sampled),
        "zero_signal": zero_signal,
        "top_reward_strict": strict(top_reward),
        "reward_inversion": bool(successes) and not strict(top_reward),
        "distillable": bool(successes) and not strict(greedy),
        "chosen": max(successes, key=chosen_key) if successes else None,
        "rejected": greedy,
    }


def read_jsonl(paths: Sequence[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    records = list(read_jsonl(args.inputs))
    grouped: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    pairs = []
    for record in records:
        summary = summarize_group(record)
        grouped[str(record["bucket"])].append((record, summary))
        if summary["distillable"]:
            pairs.append({
                "protocol": "p31_reward_ranked_pair_preview_v1",
                "example_id": record["example_id"],
                "bucket": record["bucket"],
                "mode": record["mode"],
                "messages": record["prompt_messages"],
                "chosen_assistant": summary["chosen"]["raw"],
                "rejected_assistant": summary["rejected"]["raw"],
                "chosen_strict": True,
                "rejected_strict": False,
                "target_access": False,
            })

    bucket_metrics = {}
    for bucket, items in sorted(grouped.items()):
        summaries = [item[1] for item in items]
        support_prompts = [item for item in summaries if item["any16_strict"]]
        metric = {
            "prompts": len(items),
            "greedy_strict": mean(item["greedy_strict"] for item in summaries),
            "greedy_valid": mean(item["greedy_valid"] for item in summaries),
            "sample1_strict": mean(item["sample1_strict"] for item in summaries),
            "sample1_valid": mean(item["sample1_valid"] for item in summaries),
            "any16_strict": mean(item["any16_strict"] for item in summaries),
            "sampled_strict_rate": mean(
                item["sampled_strict_count"] / 16.0 for item in summaries
            ),
            "sampled_valid_rate": mean(item["sampled_valid_rate"] for item in summaries),
            "frontier_rate": mean(item["frontier"] for item in summaries),
            "zero_signal_rate": mean(item["zero_signal"] for item in summaries),
            "distillable_rate": mean(item["distillable"] for item in summaries),
            "reward_top_strict_given_support": mean(
                item["top_reward_strict"] for item in support_prompts
            ) if support_prompts else 0.0,
            "reward_inversion_rate_given_support": mean(
                item["reward_inversion"] for item in support_prompts
            ) if support_prompts else 0.0,
        }
        metric["any16_minus_greedy"] = metric["any16_strict"] - metric["greedy_strict"]
        metric["support_limited"] = metric["any16_strict"] < 0.20
        metric["distillation_opportunity"] = metric["any16_minus_greedy"] >= 0.15
        metric["reward_misaligned"] = (
            metric["any16_strict"] > 0.0
            and metric["reward_top_strict_given_support"] < 0.70
        )
        bucket_metrics[bucket] = metric

    mode_buckets = {
        "de_novo": [value for key, value in bucket_metrics.items() if key.startswith("de_novo:")],
        "edit": [value for key, value in bucket_metrics.items() if key.startswith("edit:")],
    }
    mode_metrics = {}
    for mode, metrics in mode_buckets.items():
        mode_metrics[mode] = {
            "greedy_strict_macro": mean(item["greedy_strict"] for item in metrics),
            "sample1_strict_macro": mean(item["sample1_strict"] for item in metrics),
            "any16_strict_macro": mean(item["any16_strict"] for item in metrics),
            "support_gap_macro": mean(item["any16_minus_greedy"] for item in metrics),
            "distillable_rate_macro": mean(item["distillable_rate"] for item in metrics),
            "support_limited_buckets": sum(bool(item["support_limited"]) for item in metrics),
            "reward_misaligned_buckets": sum(bool(item["reward_misaligned"]) for item in metrics),
        }
    limited = [key for key, value in bucket_metrics.items() if value["support_limited"]]
    distill = [key for key, value in bucket_metrics.items() if value["distillation_opportunity"]]
    misaligned = [key for key, value in bucket_metrics.items() if value["reward_misaligned"]]
    if distill and limited:
        decision = "MIXED_DISTILLATION_AND_ACTION_SUPPORT"
    elif distill:
        decision = "BUILD_SUCCESS_DISTILLATION"
    elif limited:
        decision = "IMPROVE_ACTION_SUPPORT_BEFORE_POLICY_OPTIMIZATION"
    else:
        decision = "REVISIT_REWARD_AND_DATA"
    result = {
        "protocol": "p31_p24_reward_support_audit_v1",
        "policy": "P24 alignment refresh",
        "prompts": len(records),
        "candidates": len(records) * 17,
        "target_access": False,
        "frozen_evaluation_rows_used": 0,
        "mode_metrics": mode_metrics,
        "bucket_metrics": bucket_metrics,
        "distillable_pairs": len(pairs),
        "support_limited_buckets": limited,
        "distillation_opportunity_buckets": distill,
        "reward_misaligned_buckets": misaligned,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "distillable_pairs.preview.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, sort_keys=True) + "\n")
    lines = [
        "# P31 P24 reward-support audit",
        "",
        f"- prompts: {len(records)}",
        f"- candidates: {len(records) * 17}",
        f"- de novo greedy / Any@16: {100 * mode_metrics['de_novo']['greedy_strict_macro']:.1f}% / {100 * mode_metrics['de_novo']['any16_strict_macro']:.1f}%",
        f"- editing sampled Raw@1 / Any@16: {100 * mode_metrics['edit']['sample1_strict_macro']:.1f}% / {100 * mode_metrics['edit']['any16_strict_macro']:.1f}%",
        f"- distillable prompt pairs: {len(pairs)}",
        f"- support-limited buckets: {', '.join(limited) if limited else 'none'}",
        f"- reward-misaligned buckets: {', '.join(misaligned) if misaligned else 'none'}",
        f"- decision: {decision}",
        "",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
