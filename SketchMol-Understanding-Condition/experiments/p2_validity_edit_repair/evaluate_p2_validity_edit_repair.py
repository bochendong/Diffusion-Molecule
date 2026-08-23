#!/usr/bin/env python3
"""Combine paired de novo validity repair with the P1 edit-consistency gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P1_DIR = SCRIPT_DIR.parent / "p1_property_program_group_rl"
if str(P1_DIR) not in sys.path:
    sys.path.insert(0, str(P1_DIR))

import evaluate_p1_sampling_scaling as p1  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--edit-metrics-csv", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8,20")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_denovo(root: Path, budgets: Sequence[int], resamples: int, seed: int):
    condition_rows = []
    eval_lookup = {}
    for benchmark, eval_name in (("two_p_to_seven_p", "denovo_2p7p_eval.csv"), ("ood", "denovo_ood_eval.csv")):
        eval_rows = read_rows(root / "data" / eval_name)
        for row in eval_rows:
            eval_lookup[(benchmark, p1.condition_key(row))] = row
        for arm, model in (("legacy", "sft"), ("syntax_safe", "group_rl")):
            candidate_path = root / "denovo" / benchmark / arm / "raw_candidates_n20.csv"
            groups = p1.load_candidate_groups(candidate_path, required_count=max(budgets))
            condition_rows.extend(p1.build_condition_table(benchmark, model, eval_rows, groups, budgets))
    summary = p1.summarize_condition_table(condition_rows, eval_lookup)
    deltas = p1.build_paired_deltas(condition_rows, eval_lookup, resamples=resamples, seed=seed)
    for row in summary:
        row["arm"] = "legacy" if row.pop("model") == "sft" else "syntax_safe"
    return summary, deltas


def index_overall(summary: Sequence[Mapping[str, object]]):
    return {
        (str(row["benchmark"]), str(row["arm"]), int(row["candidate_budget"])): row
        for row in summary
        if str(row["group_type"]) == "overall" and str(row["group"]) == "all"
    }


def choose_edit(edit_rows: Sequence[Mapping[str, str]]) -> tuple[str, Mapping[str, str], Mapping[str, str]]:
    rows = [row for row in edit_rows if row.get("task") == "edit" and row.get("budget") == "1" and row.get("selection") == "raw"]
    indexed = {str(row["variant"]): row for row in rows}
    baseline = indexed["policy"]
    candidates = [row for name, row in indexed.items() if name != "policy"]
    best = max(candidates, key=lambda row: (float(row.get("acc_all_0_65") or 0), float(row.get("acc_all_0_15") or 0)))
    return str(best["variant"]), baseline, best


def make_decision(summary, edit_rows):
    overall = index_overall(summary)
    denovo_checks = {}
    observed = {}
    for benchmark in ("two_p_to_seven_p", "ood"):
        base1 = overall[(benchmark, "legacy", 1)]
        safe1 = overall[(benchmark, "syntax_safe", 1)]
        base8 = overall[(benchmark, "legacy", 8)]
        safe8 = overall[(benchmark, "syntax_safe", 8)]
        validity_delta = float(safe1["validity_fraction"]) - float(base1["validity_fraction"])
        raw_delta = float(safe1["raw_success_fraction"]) - float(base1["raw_success_fraction"])
        pass8_delta = float(safe8["empirical_prefix_pass_at_k"]) - float(base8["empirical_prefix_pass_at_k"])
        observed[f"{benchmark}_raw_validity_delta"] = validity_delta
        observed[f"{benchmark}_raw_strict_delta"] = raw_delta
        observed[f"{benchmark}_pass8_delta"] = pass8_delta
        denovo_checks[f"{benchmark}_validity"] = validity_delta >= 0.20
        denovo_checks[f"{benchmark}_raw_strict"] = raw_delta >= -0.01
        denovo_checks[f"{benchmark}_pass8"] = pass8_delta >= -0.02

    best_name, edit_base, edit_best = choose_edit(edit_rows)
    edit_strict_delta = float(edit_best.get("acc_all_0_65") or 0) - float(edit_base.get("acc_all_0_65") or 0)
    edit_relaxed_delta = float(edit_best.get("acc_all_0_15") or 0) - float(edit_base.get("acc_all_0_15") or 0)
    edit_go = edit_strict_delta >= 0.02 or edit_relaxed_delta >= 0.03
    observed.update({"edit_raw_acc_0_65_delta": edit_strict_delta, "edit_raw_acc_0_15_delta": edit_relaxed_delta})
    return {
        "decision": "go" if all(denovo_checks.values()) and edit_go else "stop",
        "best_edit_variant": best_name,
        "denovo_checks": denovo_checks,
        "edit_go": edit_go,
        "observed": observed,
    }


def pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return ""


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary, edit_rows, decision) -> str:
    overall = index_overall(summary)
    lines = [
        "# P2 Syntax-Safe De Novo + Source-Anchored Edit Repair",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "The de novo arm changes the raw autoregressive decoder, not post-hoc selection. The edit arm uses source-only consistency diagnostics over executable GraphEditDSL candidates; target molecules and output property oracles do not enter raw ranking.",
        "",
        "## De novo paired validation",
        "",
        "| Benchmark | Arm | k | Raw validity | Raw strict | Pass@k | Unique valid / k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark in ("two_p_to_seven_p", "ood"):
        for arm in ("legacy", "syntax_safe"):
            for budget in (1, 8, 20):
                row = overall[(benchmark, arm, budget)]
                lines.append(
                    f"| {benchmark} | {arm} | {budget} | {pct(row['validity_fraction'])} | "
                    f"{pct(row['raw_success_fraction'])} | {pct(row['empirical_prefix_pass_at_k'])} | "
                    f"{pct(row['unique_valid_fraction'])} |"
                )
    lines.extend(
        [
            "",
            "## Edit paired validation",
            "",
            "| Variant | n | Selection | Validity | Acc@0.65 | Acc@0.15 |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in edit_rows:
        if row.get("task") != "edit":
            continue
        lines.append(
            f"| {row['variant']} | {row['budget']} | {row['selection']} | {pct(row.get('validity'))} | "
            f"{pct(row.get('acc_all_0_65'))} | {pct(row.get('acc_all_0_15'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- De novo: each benchmark needs at least +20pp raw validity, no worse than -1pp raw strict, and no worse than -2pp pass@8.",
            "- Edit: raw n=1 Acc@0.65 must gain 2pp or Acc@0.15 must gain 3pp.",
            f"- Best edit variant: `{decision['best_edit_variant']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = p1.parse_budgets(str(args.budgets))
    summary, deltas = collect_denovo(args.output_root, budgets, int(args.bootstrap_resamples), int(args.seed))
    edit_rows = read_rows(args.edit_metrics_csv)
    verdict = make_decision(summary, edit_rows)
    final_dir = args.output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_csv(final_dir / "p2_denovo_metrics.csv", summary)
    write_csv(final_dir / "p2_denovo_paired_deltas.csv", deltas)
    payload = {
        "protocol": "p2_syntax_safe_denovo_source_anchored_edit_repair_v1",
        "decision": verdict,
        "denovo_metrics": summary,
        "denovo_paired_deltas": deltas,
        "edit_metrics_csv": str(args.edit_metrics_csv),
    }
    (final_dir / "p2_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(summary, edit_rows, verdict)
    (final_dir / "p2_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
