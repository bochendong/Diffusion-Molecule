"""Export paper-ready summaries for verified instruction editing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "run_name",
    "planner_mode",
    "split_column",
    "backend",
    "overall_success_at_1_by_instruction",
    "overall_success_at_5_by_instruction",
    "overall_success_at_10_by_instruction",
    "overall_instruction_success_rate",
    "goal_success_rate",
    "constraint_success_rate",
    "edit_success_rate",
    "sketchmol_local_edit_success_rate",
    "validity",
    "druglike_rate",
    "novelty",
    "exact_train_hit_rate",
    "sampled_novelty_at_tanimoto_0_90",
    "fragment_growth_candidate_fraction",
    "two_step_fragment_candidate_fraction",
    "mmp_pair_target_candidate_fraction",
    "train_size",
    "eval_size",
    "n",
    "run_dir",
]


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.runs] if args.runs else sorted(Path(args.runs_root).glob("*instruction*"))
    summary = build_summary(run_dirs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    if args.latex_out:
        latex_path = Path(args.latex_out)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(_to_latex(summary), encoding="utf-8")
    if args.breakdown_out:
        breakdown = build_breakdown(run_dirs)
        breakdown_path = Path(args.breakdown_out)
        breakdown_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown.to_csv(breakdown_path, index=False)
    print(f"Wrote {len(summary)} run rows -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize verified instruction editing runs.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--out", default="outputs/instruction_paper_summary.csv")
    parser.add_argument("--latex-out", default="outputs/instruction_paper_summary.tex")
    parser.add_argument("--breakdown-out", default="outputs/instruction_failure_breakdown.csv")
    return parser.parse_args()


def build_summary(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        row = {key: metrics.get(key, "") for key in METRIC_COLUMNS}
        row["run_name"] = run_dir.name
        rows.append(row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_breakdown(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        details_path = run_dir / "tables" / "verified_instruction_candidates.csv"
        if not details_path.exists():
            continue
        df = pd.read_csv(details_path)
        for group_col in ("difficulty", "goal_combo_key", "edit_combo_key", "language_source"):
            if group_col not in df.columns:
                continue
            for label, frame in df.groupby(group_col, dropna=False):
                rows.append(
                    {
                        "run_name": run_dir.name,
                        "group": group_col,
                        "label": label,
                        "n": float(len(frame)),
                        "overall_instruction_success_rate": _mean_bool(frame.get("verify_overall_success", [])),
                        "goal_success_rate": _mean_bool(frame.get("verify_goal_success", [])),
                        "constraint_success_rate": _mean_bool(frame.get("verify_constraint_success", [])),
                        "edit_success_rate": _mean_bool(frame.get("verify_edit_success", [])),
                    }
                )
    return pd.DataFrame(rows)


def _mean_bool(values) -> float:
    if len(values) == 0:
        return 0.0
    return float(pd.Series(values).astype(bool).mean())


def _to_latex(summary: pd.DataFrame) -> str:
    display_cols = [
        "run_name",
        "overall_success_at_10_by_instruction",
        "constraint_success_rate",
        "edit_success_rate",
        "exact_train_hit_rate",
        "sampled_novelty_at_tanimoto_0_90",
        "fragment_growth_candidate_fraction",
    ]
    frame = summary[[col for col in display_cols if col in summary.columns]].copy()
    for col in frame.columns:
        if col != "run_name":
            frame[col] = pd.to_numeric(frame[col], errors="coerce").round(4)
    return frame.to_latex(index=False, escape=True)


if __name__ == "__main__":
    main()
