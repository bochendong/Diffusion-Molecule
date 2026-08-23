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
    raw_smiles: str
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
    parser.add_argument(
        "--allow-condition-intersection",
        action="store_true",
        help="Evaluate only conditions complete in both models. Marks every output as interim.",
    )
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
    raw = str(
        row.get("direct_candidate_raw_smiles")
        or row.get("generated_smiles")
        or row.get("direct_candidate_canonical_smiles")
        or ""
    ).strip()
    canonical = str(row.get("direct_candidate_canonical_smiles") or "").strip()
    strict_fraction = _float(row.get("direct_candidate_strict_fraction"))
    valid = bool(canonical)
    strict = valid and math.isclose(strict_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9)
    return Candidate(index=index, raw_smiles=raw, valid=valid, strict=strict, canonical_smiles=canonical)


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
                # Historical Direct-SMILES tables evaluate the single molecule
                # selected from k candidates. Because every valid candidate
                # outranks an invalid one, its validity is equivalent to this
                # condition-level any-valid rate, not raw candidate validity.
                "selected_validity_at_k": float(bool(valid)),
                "validity_fraction": len(valid) / budget,
                "empty_raw_fraction": sum(not item.raw_smiles for item in prefix) / budget,
                "nonempty_invalid_fraction": sum(bool(item.raw_smiles) and not item.valid for item in prefix) / budget,
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
    "selected_validity_at_k",
    "validity_fraction",
    "empty_raw_fraction",
    "nonempty_invalid_fraction",
    "unique_valid_count",
    "unique_valid_fraction",
)


def build_validity_audit(summary: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Make the selected-vs-raw validity distinction explicit for paper tables."""
    out: list[dict[str, object]] = []
    for row in summary:
        raw_success = float(row["raw_success_fraction"])
        raw_validity = float(row["validity_fraction"])
        out.append(
            {
                "benchmark": row["benchmark"],
                "model": row["model"],
                "group_type": row["group_type"],
                "group": row["group"],
                "candidate_budget": row["candidate_budget"],
                "conditions": row["conditions"],
                "raw_candidate_validity": raw_validity,
                "selected_validity_at_k": row["selected_validity_at_k"],
                "raw_strict_success": raw_success,
                "strict_success_given_valid_candidate": raw_success / raw_validity if raw_validity else 0.0,
                "empty_raw_fraction": row["empty_raw_fraction"],
                "nonempty_rdkit_invalid_fraction": row["nonempty_invalid_fraction"],
                "unique_valid_fraction": row["unique_valid_fraction"],
            }
        )
    return out


def build_paper_table(
    summary: Sequence[Mapping[str, object]],
    deltas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join absolute SFT/Group-RL metrics and paired uncertainty in one table."""
    summary_index = {
        (
            str(row["benchmark"]),
            str(row["model"]),
            str(row["group_type"]),
            str(row["group"]),
            int(row["candidate_budget"]),
        ): row
        for row in summary
    }
    out: list[dict[str, object]] = []
    for delta in deltas:
        benchmark = str(delta["benchmark"])
        group_type = str(delta["group_type"])
        group = str(delta["group"])
        budget = int(delta["candidate_budget"])
        sft = summary_index[(benchmark, "sft", group_type, group, budget)]
        group_rl = summary_index[(benchmark, "group_rl", group_type, group, budget)]
        out.append(
            {
                "benchmark": benchmark,
                "group_type": group_type,
                "group": group,
                "candidate_budget": budget,
                "paired_conditions": delta["paired_conditions"],
                "sft_raw_success": sft["raw_success_fraction"],
                "group_rl_raw_success": group_rl["raw_success_fraction"],
                "delta_raw_success": delta["delta_raw_success_fraction"],
                "delta_raw_ci95_low": delta["delta_raw_success_fraction_ci95_low"],
                "delta_raw_ci95_high": delta["delta_raw_success_fraction_ci95_high"],
                "sft_empirical_pass_at_k": sft["empirical_prefix_pass_at_k"],
                "group_rl_empirical_pass_at_k": group_rl["empirical_prefix_pass_at_k"],
                "delta_empirical_pass_at_k": delta["delta_empirical_prefix_pass_at_k"],
                "delta_pass_ci95_low": delta["delta_empirical_prefix_pass_at_k_ci95_low"],
                "delta_pass_ci95_high": delta["delta_empirical_prefix_pass_at_k_ci95_high"],
                "sft_raw_candidate_validity": sft["validity_fraction"],
                "group_rl_raw_candidate_validity": group_rl["validity_fraction"],
                "sft_selected_validity_at_k": sft["selected_validity_at_k"],
                "group_rl_selected_validity_at_k": group_rl["selected_validity_at_k"],
                "sft_unique_valid_fraction": sft["unique_valid_fraction"],
                "group_rl_unique_valid_fraction": group_rl["unique_valid_fraction"],
            }
        )
    return out


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
            row = indexed.get((benchmark, "overall", "all", budget))
            for metric in ("raw_success_fraction", "empirical_prefix_pass_at_k"):
                checks[f"{benchmark}_k{budget}_{metric}_positive"] = bool(
                    row is not None and float(row[f"delta_{metric}"]) > 0.0
                )
        for count in ("6", "7"):
            row = indexed.get(("two_p_to_seven_p", "property_count", count, budget))
            for metric in ("raw_success_fraction", "empirical_prefix_pass_at_k"):
                checks[f"hard_{count}p_k{budget}_{metric}_positive"] = bool(
                    row is not None and float(row[f"delta_{metric}"]) > 0.0
                )

    point_gate = all(checks.values())
    ci_checks = {}
    for budget in (8, 20):
        row = indexed.get(("two_p_to_seven_p", "overall", "all", budget))
        ci_checks[f"two_p_to_seven_p_k{budget}_raw_ci_positive"] = bool(
            row is not None and float(row["delta_raw_success_fraction_ci95_low"]) > 0.0
        )
    row_k8 = indexed.get(("two_p_to_seven_p", "overall", "all", 8))
    ci_checks["two_p_to_seven_p_k8_pass_ci_positive"] = bool(
        row_k8 is not None and float(row_k8["delta_empirical_prefix_pass_at_k_ci95_low"]) > 0.0
    )

    if point_gate and all(ci_checks.values()):
        verdict = "strong_single_seed_signal"
    elif point_gate:
        verdict = "promising_single_seed_signal"
    else:
        low_budget_positive = any(checks.values())
        high_budget_rows = [
            indexed.get((benchmark, "overall", "all", 256)) for benchmark in ("two_p_to_seven_p", "ood")
        ]
        high_budget_wins = all(
            row is not None and float(row["delta_empirical_prefix_pass_at_k"]) > 0.0
            for row in high_budget_rows
        )
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
    ]
    coverage = dict(gate.get("coverage") or {})
    for benchmark in ("two_p_to_seven_p", "ood"):
        item = dict(coverage.get(benchmark) or {})
        if item:
            lines.append(
                f"Coverage `{benchmark}`: {int(item['paired_intersection_conditions']):,} / "
                f"{int(item['full_conditions']):,} paired conditions."
            )
    lines.extend(
        [
            "",
            "All metrics use raw candidates in seeded generation order. `k=1` is the first raw draw. The generator's property-reranked selected molecule is diagnostic-only and is never reported as one-shot. `empirical prefix pass@k` is any strict success among the first k draws; `estimated pass@k` is computed from all n=256 raw outcomes.",
            "",
            "## Overall sampling scaling",
            "",
            "| benchmark | model | k | raw success | empirical pass@k | estimated pass@k | raw candidate validity | selected validity@k | unique valid / k |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
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
                    f"{float(row['validity_fraction']):.4f} | {float(row['selected_validity_at_k']):.4f} | "
                    f"{float(row['unique_valid_fraction']):.4f} |"
                )
    lines.extend(
        [
            "",
            "## Validity metric alignment",
            "",
            "`raw candidate validity` is the fraction of all unselected draws that RDKit can parse. "
            "`selected validity@k` is the fraction of conditions with at least one valid molecule among the first k draws; "
            "it is the validity quantity comparable to historical property-reranked best-of-k tables. "
            "The two metrics must not be compared as if they were identical.",
            "",
            "| benchmark | model | k | raw candidate validity | selected validity@k | strict among valid candidates | empty raw | nonempty RDKit-invalid |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for benchmark in ("two_p_to_seven_p", "ood"):
        for model in MODELS:
            for budget in (1, 20, 256):
                row = summary_index.get((benchmark, model, "overall", "all", budget))
                if row is None:
                    continue
                validity = float(row["validity_fraction"])
                strict_in_valid = float(row["raw_success_fraction"]) / validity if validity else 0.0
                lines.append(
                    f"| {benchmark} | {model} | {budget} | {validity:.4f} | "
                    f"{float(row['selected_validity_at_k']):.4f} | {strict_in_valid:.4f} | "
                    f"{float(row['empty_raw_fraction']):.4f} | {float(row['nonempty_invalid_fraction']):.4f} |"
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
            "- This run deliberately remains a single-seed result and does not estimate between-seed variance. External baseline reproduction is a separate paper-facing gate.",
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
    candidate_groups = {
        key: load_candidate_groups(path, required_count=max_budget) for key, path in candidate_paths.items()
    }
    coverage: dict[str, dict[str, int]] = {}
    if args.allow_condition_intersection:
        for benchmark in eval_sets:
            original_eval = eval_sets[benchmark]
            eval_keys = {condition_key(row) for row in original_eval}
            common = set(eval_keys)
            model_counts = {}
            for model in MODELS:
                groups = candidate_groups[(benchmark, model)]
                model_counts[f"{model}_complete_conditions"] = len(groups)
                common &= set(groups)
            eval_sets[benchmark] = [row for row in original_eval if condition_key(row) in common]
            for model in MODELS:
                groups = candidate_groups[(benchmark, model)]
                candidate_groups[(benchmark, model)] = {key: groups[key] for key in common}
            coverage[benchmark] = {
                "full_conditions": len(original_eval),
                "paired_intersection_conditions": len(common),
                **model_counts,
            }
            if not common:
                raise RuntimeError(f"{benchmark}: no complete paired conditions for interim evaluation")
    condition_rows: list[dict[str, object]] = []
    for (benchmark, model), path in candidate_paths.items():
        condition_rows.extend(
            build_condition_table(
                benchmark,
                model,
                eval_sets[benchmark],
                candidate_groups[(benchmark, model)],
                budgets,
            )
        )
    eval_lookup = {
        (benchmark, condition_key(row)): row for benchmark, rows in eval_sets.items() for row in rows
    }
    summary = summarize_condition_table(condition_rows, eval_lookup)
    validity_audit = build_validity_audit(summary)
    deltas = build_paired_deltas(
        condition_rows,
        eval_lookup,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paper_table = build_paper_table(summary, deltas)
    gate = gate_result(deltas)
    gate["coverage"] = coverage or {
        benchmark: {
            "full_conditions": len(rows),
            "paired_intersection_conditions": len(rows),
            "sft_complete_conditions": len(rows),
            "group_rl_complete_conditions": len(rows),
        }
        for benchmark, rows in eval_sets.items()
    }
    gate["interim"] = bool(args.allow_condition_intersection)
    if args.allow_condition_intersection:
        gate["verdict"] = f"interim_{gate['verdict']}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p1_condition_metrics.csv", condition_rows)
    write_csv(args.output_dir / "p1_scaling_summary.csv", summary)
    write_csv(args.output_dir / "p1_paired_deltas.csv", deltas)
    write_csv(args.output_dir / "p1_validity_audit.csv", validity_audit)
    write_csv(args.output_dir / "p1_paper_main_table.csv", paper_table)
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
