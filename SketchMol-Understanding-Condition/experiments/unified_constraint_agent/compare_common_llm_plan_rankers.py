#!/usr/bin/env python3
"""Compare held-out plan rankers and enforce the source-preserving go/no-go gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--verifier-k", type=int, default=5)
    parser.add_argument("--min-primary-gain", type=float, default=0.01)
    parser.add_argument("--max-source-similarity-drop", type=float, default=0.01)
    parser.add_argument("--fail-on-stop", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: Mapping[str, object], selection: str, name: str) -> float:
    selections = summary.get("selections", {})
    if not isinstance(selections, Mapping):
        raise ValueError("Summary is missing selections")
    selected = selections.get(selection, {})
    if not isinstance(selected, Mapping):
        raise ValueError(f"Summary is missing selection {selection}")
    all_rows = selected.get("all", {})
    if not isinstance(all_rows, Mapping):
        raise ValueError(f"Summary is missing all-scope metrics for {selection}")
    return float(all_rows[name])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = read_json(args.baseline_summary)
    candidate = read_json(args.candidate_summary)
    verifier = f"llm_verifier_at_{int(args.verifier_k)}"
    comparisons = {}
    for selection in ("llm_at_1", verifier):
        for name in ("success_rate", "strict_success_rate", "source_similarity_success_rate"):
            baseline_value = metric(baseline, selection, name)
            candidate_value = metric(candidate, selection, name)
            comparisons[f"{selection}:{name}"] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
            }
    raw_sr_delta = comparisons["llm_at_1:success_rate"]["delta"]
    raw_strict_delta = comparisons["llm_at_1:strict_success_rate"]["delta"]
    raw_similarity_delta = comparisons["llm_at_1:source_similarity_success_rate"]["delta"]
    verifier_sr_delta = comparisons[f"{verifier}:success_rate"]["delta"]
    verifier_strict_delta = comparisons[f"{verifier}:strict_success_rate"]["delta"]
    verifier_similarity_delta = comparisons[f"{verifier}:source_similarity_success_rate"]["delta"]
    primary_gain = max(raw_sr_delta, raw_strict_delta, verifier_sr_delta, verifier_strict_delta)
    checks = {
        "positive_primary_gain": primary_gain >= float(args.min_primary_gain),
        "raw_strict_non_decrease": raw_strict_delta >= 0.0,
        "verifier_sr_non_decrease": verifier_sr_delta >= 0.0,
        "verifier_strict_non_decrease": verifier_strict_delta >= 0.0,
        "raw_similarity_preserved": raw_similarity_delta >= -float(args.max_source_similarity_drop),
        "verifier_similarity_preserved": verifier_similarity_delta >= -float(args.max_source_similarity_drop),
    }
    decision = "go" if all(checks.values()) else "stop"
    result = {
        "protocol": "common_llm_two_step_plan_preference_gate_v3",
        "decision": decision,
        "baseline_summary": str(args.baseline_summary),
        "candidate_summary": str(args.candidate_summary),
        "verifier_k": int(args.verifier_k),
        "min_primary_gain": float(args.min_primary_gain),
        "max_source_similarity_drop": float(args.max_source_similarity_drop),
        "primary_gain": primary_gain,
        "checks": checks,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Common LLM two-step plan preference gate",
        "",
        f"Decision: **{decision}**",
        "",
        "| Selection | Metric | Baseline | Candidate | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, record in comparisons.items():
        selection, name = key.split(":", 1)
        lines.append(
            f"| {selection} | {name} | {record['baseline']:.4f} | "
            f"{record['candidate']:.4f} | {record['delta']:+.4f} |"
        )
    lines.extend(["", "Checks:"])
    for name, passed in checks.items():
        lines.append(f"- {name}: `{'pass' if passed else 'fail'}`")
    lines.append("")
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if args.fail_on_stop and decision != "go" else 0


if __name__ == "__main__":
    raise SystemExit(main())
