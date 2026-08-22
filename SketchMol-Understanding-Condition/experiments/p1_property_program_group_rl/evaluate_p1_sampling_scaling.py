#!/usr/bin/env python3
"""Evaluate P1 Direct-SMILES SFT vs Group-RL sampling/complexity scaling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_BUDGETS = (1, 4, 8, 20, 32, 64, 128, 256)
MODELS = ("sft", "group_rl")


@dataclass(frozen=True)
class Candidate:
    index: int
    valid: bool
    strict: bool
    canonical_smiles: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-p-seven-p-eval-csv", required=True, type=Path)
    parser.add_argument("--ood-eval-csv", required=True, type=Path)
    parser.add_argument("--two-p-seven-p-sft-candidates", required=True, type=Path)
    parser.add_argument("--two-p-seven-p-group-rl-candidates", required=True, type=Path)
    parser.add_argument("--ood-sft-candidates", required=True, type=Path)
    parser.add_argument("--ood-group-rl-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--protocol", type=Path, default=None)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_key(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def parse_budgets(text: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(item) for item in text.split(",") if item.strip()}))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must contain positive integers")
    return budgets


def candidate_from_row(row: Mapping[str, object]) -> Candidate:
    index = int(float(row.get("direct_candidate_index") or row.get("candidate_index") or 0))
    canonical = str(row.get("direct_candidate_canonical_smiles") or "").strip()
    strict_fraction = _float(row.get("direct_candidate_strict_fraction"))
    valid = bool(canonical)
    strict = valid and math.isclose(strict_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9)
    return Candidate(index=index, valid=valid, strict=strict, canonical_smiles=canonical)


def load_candidate_groups(path: Path, *, required_count: int) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for row in read_rows(path):
        key = condition_key(row)
        if key:
            grouped[key].append(candidate_from_row(row))
    for key, values in grouped.items():
        values.sort(key=lambda item: item.index)
        indices = [item.index for item in values[:required_count]]
        if len(values) < required_count:
            raise RuntimeError(f"{path}: {key} has {len(values)} candidates; expected at least {required_count}")
        if indices != list(range(required_count)):
            raise RuntimeError(f"{path}: {key} does not contain ordered candidate indices 0..{required_count - 1}")
        grouped[key] = values[:required_count]
    return dict(grouped)


def estimated_pass_at_k(total: int, successes: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c,k) / C(n,k)."""
    if total <= 0 or k <= 0:
        return 0.0
    k = min(k, total)
    successes = max(0, min(successes, total))
    failures = total - successes
    if failures < k:
        return 1.0
    miss_probability = 1.0
    for offset in range(k):
        miss_probability *= (failures - offset) / (total - offset)
    return 1.0 - miss_probability


def condition_metrics(candidates: Sequence[Candidate], budgets: Sequence[int]) -> list[dict[str, object]]:
    total = len(candidates)
    total_successes = sum(item.strict for item in candidates)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        prefix = candidates[:budget]
        valid = [item for item in prefix if item.valid]
        strict_count = sum(item.strict for item in prefix)
        unique_valid = len({item.canonical_smiles for item in valid})
        rows.append(
            {
                "candidate_budget": budget,
                "raw_success_fraction": strict_count / budget,
                "empirical_prefix_pass_at_k": float(strict_count > 0),
                "estimated_pass_at_k": estimated_pass_at_k(total, total_successes, budget),
                "validity_fraction": len(valid) / budget,
                "unique_valid_count": unique_valid,
                "unique_valid_fraction": unique_valid / budget,
            }
        )
    return rows


def strata_for(benchmark: str, row: Mapping[str, str]) -> list[tuple[str, str]]:
    out = [("overall", "all")]
    property_count = str(int(float(row.get("property_count") or 0)))
    if property_count != "0":
        out.append(("property_count", property_count))
    if benchmark == "ood":
        bucket = str(row.get("ood_bucket") or "unknown").strip()
        out.append(("ood_bucket", bucket))
    return out


def build_condition_table(
    benchmark: str,
    model: str,
    eval_rows: Sequence[Mapping[str, str]],
    candidates: Mapping[str, Sequence[Candidate]],
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    eval_by_key = {condition_key(row): row for row in eval_rows}
    missing = sorted(set(eval_by_key) - set(candidates))
    extra = sorted(set(candidates) - set(eval_by_key))
    if missing or extra:
        raise RuntimeError(
            f"{benchmark}/{model} condition mismatch: missing={missing[:3]} ({len(missing)}), "
            f"extra={extra[:3]} ({len(extra)})"
        )
    out: list[dict[str, object]] = []
    for key, eval_row in eval_by_key.items():
        for metric_row in condition_metrics(candidates[key], budgets):
            out.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "condition_id": key,
                    "property_count": int(float(eval_row.get("property_count") or 0)),
                    "ood_bucket": str(eval_row.get("ood_bucket") or ""),
                    **metric_row,
                }
            )
    return out


METRICS = (
    "raw_success_fraction",
    "empirical_prefix_pass_at_k",
    "estimated_pass_at_k",
    "validity_fraction",
    "unique_valid_count",
    "unique_valid_fraction",
)


def summarize_condition_table(
    condition_rows: Sequence[Mapping[str, object]],
    eval_lookup: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in condition_rows:
        key = (str(row["benchmark"]), str(row["condition_id"]))
        for group_type, group in strata_for(str(row["benchmark"]), eval_lookup[key]):
            grouped[(str(row["benchmark"]), str(row["model"]), group_type, group, int(row["candidate_budget"]))].append(row)
    summary: list[dict[str, object]] = []
    for (benchmark, model, group_type, group, budget), rows in sorted(grouped.items()):
        item: dict[str, object] = {
            "benchmark": benchmark,
            "model": model,
            "group_type": group_type,
            "group": group,
            "candidate_budget": budget,
            "conditions": len(rows),
        }
        for metric in METRICS:
            item[metric] = statistics.fmean(float(row[metric]) for row in rows)
        summary.append(item)
    return summary


def stable_seed(*parts: object, base: int) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return base + int.from_bytes(digest[:4], "big")


def paired_bootstrap_ci(deltas: Sequence[float], *, resamples: int, seed: int) -> tuple[float, float]:
    if not deltas:
        return math.nan, math.nan
    if len(deltas) == 1 or resamples <= 0:
        value = float(deltas[0])
        return value, value
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [statistics.fmean(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(resamples)]
    estimates.sort()
    low = estimates[max(0, int(0.025 * resamples) - 1)]
    high = estimates[min(resamples - 1, int(0.975 * resamples))]
    return low, high


def build_paired_deltas(
    condition_rows: Sequence[Mapping[str, object]],
    eval_lookup: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    indexed = {
        (str(row["benchmark"]), str(row["model"]), str(row["condition_id"]), int(row["candidate_budget"])): row
        for row in condition_rows
    }
    grouped_ids: dict[tuple[str, str, str, int], list[str]] = defaultdict(list)
    for benchmark, model, condition_id, budget in indexed:
        if model != "sft":
            continue
        eval_row = eval_lookup[(benchmark, condition_id)]
        for group_type, group in strata_for(benchmark, eval_row):
            grouped_ids[(benchmark, group_type, group, budget)].append(condition_id)
    out: list[dict[str, object]] = []
    for (benchmark, group_type, group, budget), condition_ids in sorted(grouped_ids.items()):
        item: dict[str, object] = {
            "benchmark": benchmark,
            "group_type": group_type,
            "group": group,
            "candidate_budget": budget,
            "paired_conditions": len(condition_ids),
        }
        for metric in METRICS:
            deltas = []
            for condition_id in condition_ids:
                sft = indexed[(benchmark, "sft", condition_id, budget)]
                group_rl = indexed.get((benchmark, "group_rl", condition_id, budget))
                if group_rl is None:
                    raise RuntimeError(f"Missing paired group_rl row for {benchmark}/{condition_id}/k={budget}")
                deltas.append(float(group_rl[metric]) - float(sft[metric]))
            mean_delta = statistics.fmean(deltas)
            low, high = paired_bootstrap_ci(
                deltas,
                resamples=resamples,
                seed=stable_seed(benchmark, group_type, group, budget, metric, base=seed),
            )
            item[f"delta_{metric}"] = mean_delta
            item[f"delta_{metric}_ci95_low"] = low
            item[f"delta_{metric}_ci95_high"] = high
        out.append(item)
    return out


def gate_result(deltas: Sequence[Mapping[str, object]]) -> dict[str, object]:
    indexed = {
        (str(row["benchmark"]), str(row["group_type"]), str(row["group"]), int(row["candidate_budget"])): row
        for row in deltas
    }
    checks: dict[str, bool] = {}
    for budget in (8, 20):
        for benchmark in ("two_p_to_seven_p", "ood"):
            row = indexed[(benchmark, "overall", "all", budget)]
            for metric in ("raw_success_fraction", "empirical_prefix_pass_at_k"):
                checks[f"{benchmark}_k{budget}_{metric}_positive"] = float(row[f"delta_{metric}"]) > 0.0
        for count in ("6", "7"):
            row = indexed[("two_p_to_seven_p", "property_count", count, budget)]
            for metric in ("raw_success_fraction", "empirical_prefix_pass_at_k"):
                checks[f"hard_{count}p_k{budget}_{metric}_positive"] = float(row[f"delta_{metric}"]) > 0.0

    point_gate = all(checks.values())
    ci_checks = {}
    for budget in (8, 20):
        row = indexed[("two_p_to_seven_p", "overall", "all", budget)]
        ci_checks[f"two_p_to_seven_p_k{budget}_raw_ci_positive"] = float(
            row["delta_raw_success_fraction_ci95_low"]
        ) > 0.0
    row_k8 = indexed[("two_p_to_seven_p", "overall", "all", 8)]
    ci_checks["two_p_to_seven_p_k8_pass_ci_positive"] = float(
        row_k8["delta_empirical_prefix_pass_at_k_ci95_low"]
    ) > 0.0

    if point_gate and all(ci_checks.values()):
        verdict = "strong_single_seed_signal"
    elif point_gate:
        verdict = "promising_single_seed_signal"
    else:
        low_budget_positive = any(checks.values())
        high_budget_rows = [
            indexed[(benchmark, "overall", "all", 256)] for benchmark in ("two_p_to_seven_p", "ood")
        ]
        high_budget_wins = all(float(row["delta_empirical_prefix_pass_at_k"]) > 0.0 for row in high_budget_rows)
        verdict = "sampling_heavy_only" if high_budget_wins and not low_budget_positive else "mixed_or_negative"
    return {"verdict": verdict, "point_checks": checks, "confidence_checks": ci_checks}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    summary: Sequence[Mapping[str, object]],
    deltas: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> str:
    summary_index = {
        (str(row["benchmark"]), str(row["model"]), str(row["group_type"]), str(row["group"]), int(row["candidate_budget"])): row
        for row in summary
    }
    delta_index = {
        (str(row["benchmark"]), str(row["group_type"]), str(row["group"]), int(row["candidate_budget"])): row
        for row in deltas
    }
    lines = [
        "# P1 Direct SMILES + Property Program + Group-RL: Single-seed gate",
        "",
        f"Verdict: **{gate['verdict']}**.",
        "",
        "All metrics use raw candidates in seeded generation order. `k=1` is the first raw draw. The generator's property-reranked selected molecule is diagnostic-only and is never reported as one-shot. `empirical prefix pass@k` is any strict success among the first k draws; `estimated pass@k` is computed from all n=256 raw outcomes.",
        "",
        "## Overall sampling scaling",
        "",
        "| benchmark | model | k | raw success | empirical pass@k | estimated pass@k | validity | unique valid / k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark in ("two_p_to_seven_p", "ood"):
        for model in MODELS:
            for budget in DEFAULT_BUDGETS:
                key = (benchmark, model, "overall", "all", budget)
                if key not in summary_index:
                    continue
                row = summary_index[key]
                lines.append(
                    f"| {benchmark} | {model} | {budget} | {float(row['raw_success_fraction']):.4f} | "
                    f"{float(row['empirical_prefix_pass_at_k']):.4f} | {float(row['estimated_pass_at_k']):.4f} | "
                    f"{float(row['validity_fraction']):.4f} | {float(row['unique_valid_fraction']):.4f} |"
                )
    lines.extend(
        [
            "",
            "## Paired Group-RL minus SFT at the primary budgets",
            "",
            "| benchmark | stratum | k | delta raw | 95% CI | delta empirical pass@k | 95% CI |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    primary_keys = []
    for budget in (8, 20):
        primary_keys.append(("two_p_to_seven_p", "overall", "all", budget))
        for count in ("2", "3", "4", "5", "6", "7"):
            primary_keys.append(("two_p_to_seven_p", "property_count", count, budget))
        primary_keys.append(("ood", "overall", "all", budget))
        for bucket in ("forward_extreme", "rare_combo", "reverse_stimulation"):
            primary_keys.append(("ood", "ood_bucket", bucket, budget))
    for key in primary_keys:
        row = delta_index.get(key)
        if row is None:
            continue
        stratum = "overall" if key[1] == "overall" else f"{key[1]}={key[2]}"
        lines.append(
            f"| {key[0]} | {stratum} | {key[3]} | {float(row['delta_raw_success_fraction']):+.4f} | "
            f"[{float(row['delta_raw_success_fraction_ci95_low']):+.4f}, {float(row['delta_raw_success_fraction_ci95_high']):+.4f}] | "
            f"{float(row['delta_empirical_prefix_pass_at_k']):+.4f} | "
            f"[{float(row['delta_empirical_prefix_pass_at_k_ci95_low']):+.4f}, {float(row['delta_empirical_prefix_pass_at_k_ci95_high']):+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- `strong_single_seed_signal`: all preregistered low-budget point gates pass, and the overall 2p-7p raw-success CIs at k=8/20 plus empirical pass@8 CI exclude zero.",
            "- `promising_single_seed_signal`: every low-budget point gate passes, but one-seed confidence evidence is not yet uniformly decisive.",
            "- `sampling_heavy_only`: low-budget evidence fails while Group-RL only wins at k=256.",
            "- This run is a one-seed gate, not a final paper claim. A positive gate must be followed by multi-seed confirmation and external baseline reproduction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(args.budgets)
    max_budget = budgets[-1]
    if max_budget != 256:
        raise ValueError("P1 preregistration requires a maximum candidate budget of 256")
    eval_sets = {
        "two_p_to_seven_p": read_rows(args.two_p_seven_p_eval_csv),
        "ood": read_rows(args.ood_eval_csv),
    }
    candidate_paths = {
        ("two_p_to_seven_p", "sft"): args.two_p_seven_p_sft_candidates,
        ("two_p_to_seven_p", "group_rl"): args.two_p_seven_p_group_rl_candidates,
        ("ood", "sft"): args.ood_sft_candidates,
        ("ood", "group_rl"): args.ood_group_rl_candidates,
    }
    condition_rows: list[dict[str, object]] = []
    for (benchmark, model), path in candidate_paths.items():
        condition_rows.extend(
            build_condition_table(
                benchmark,
                model,
                eval_sets[benchmark],
                load_candidate_groups(path, required_count=max_budget),
                budgets,
            )
        )
    eval_lookup = {
        (benchmark, condition_key(row)): row for benchmark, rows in eval_sets.items() for row in rows
    }
    summary = summarize_condition_table(condition_rows, eval_lookup)
    deltas = build_paired_deltas(
        condition_rows,
        eval_lookup,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    gate = gate_result(deltas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p1_condition_metrics.csv", condition_rows)
    write_csv(args.output_dir / "p1_scaling_summary.csv", summary)
    write_csv(args.output_dir / "p1_paired_deltas.csv", deltas)
    (args.output_dir / "p1_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(summary, deltas, gate)
    (args.output_dir / "p1_report.md").write_text(report, encoding="utf-8")
    if args.protocol is not None:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        (args.output_dir / "p1_protocol_snapshot.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
