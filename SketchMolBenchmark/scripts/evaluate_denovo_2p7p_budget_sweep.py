#!/usr/bin/env python3
"""Evaluate raw, average, and best-of-K de novo candidate pools."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
PROJECT_DIR = REPO_ROOT / "SketchMol-Understanding-Condition"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS  # noqa: E402


STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
}
PROPERTY_VALUE_KEYS = {
    "MW": "MolWt",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "rotatable",
    "SA": "SA",
}
SKETCHMOL_REFERENCE = {2: 0.804, 3: 0.768, 4: 0.736, 5: 0.716, 6: 0.678, 7: 0.685}


@dataclass(frozen=True)
class CandidateScore:
    rank: int
    valid: bool
    strict_fraction: float
    distance: float
    selection_score: float

    @property
    def strict(self) -> bool:
        return self.valid and math.isclose(self.strict_fraction, 1.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shards-dir", type=Path, help="SketchMol shard root containing shard_candidates.csv files.")
    source.add_argument("--candidate-csv", type=Path, help="Direct-SMILES candidate CSV emitted by --candidate-output-csv.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="1,2,4,8,20,40")
    parser.add_argument("--report-title", default="De Novo 2p-7p Candidate-Budget Sweep")
    parser.add_argument(
        "--candidate-description",
        default="generated candidates",
        help="Short source description included in the Markdown report.",
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_key(row: Mapping[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "").strip()


def candidate_index(row: Mapping[str, str]) -> int:
    value = row.get("direct_candidate_index")
    if value in {None, ""}:
        value = row.get("candidate_index")
    return int(value or 0)


def candidate_smiles(row: Mapping[str, str]) -> str:
    return str(
        row.get("direct_candidate_raw_smiles")
        or row.get("SMILES")
        or row.get("generated_smiles")
        or ""
    ).strip()


def selected_properties(row: Mapping[str, str]) -> list[str]:
    selected = [item.strip() for item in str(row.get("condition_properties") or "").split(",") if item.strip()]
    selected = [prop for prop in selected if prop in PROPERTY_COLUMNS]
    if selected:
        return selected
    return [prop for prop in PROPERTY_COLUMNS if _truthy(row.get(f"{prop}_active"))]


@lru_cache(maxsize=300_000)
def molecule_record(smiles: str) -> tuple[bool, tuple[tuple[str, float], ...]]:
    text = str(smiles or "").strip()
    if not text:
        return False, tuple()
    try:
        canonical = canonical_smiles(text) or ""
        props = molecular_properties(canonical) if canonical else None
    except RuntimeError:
        return False, tuple()
    if not canonical or not props:
        return False, tuple()
    values = tuple(
        sorted(
            (prop, float(props.get(key, math.nan)))
            for prop, key in PROPERTY_VALUE_KEYS.items()
            if key in props
        )
    )
    return True, values


def score_candidate(row: Mapping[str, str], smiles: str, rank: int) -> CandidateScore:
    valid, prop_items = molecule_record(str(smiles or ""))
    if not valid:
        return CandidateScore(rank, False, 0.0, math.inf, -1_000_000.0 - rank * 1e-6)
    props = dict(prop_items)
    successes: list[bool] = []
    distances: list[float] = []
    for prop in selected_properties(row):
        target = _float(row.get(f"target_{prop}"))
        actual = _float(props.get(prop))
        tolerance = STRICT_TOLERANCE[prop]
        if math.isnan(target) or math.isnan(actual):
            successes.append(False)
            distances.append(1e6)
            continue
        error = abs(actual - target)
        successes.append(error <= tolerance)
        distances.append(error / tolerance)
    strict_fraction = sum(successes) / len(successes) if successes else 0.0
    distance = sum(distances) / len(distances) if distances else 0.0
    selection_score = 10.0 + 100.0 * strict_fraction - 10.0 * distance - rank * 1e-6
    return CandidateScore(rank, True, strict_fraction, distance, selection_score)


def summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    se = statistics.stdev(values) / math.sqrt(len(values))
    return mean, max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = sorted({int(value) for value in args.budgets.split(",") if value.strip()})
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must contain positive integers")

    eval_rows = read_rows(args.eval_csv)
    eval_by_key = {condition_key(row): row for row in eval_rows}
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_paths = (
        [args.candidate_csv]
        if args.candidate_csv is not None
        else sorted(args.shards_dir.glob("shard_*/shard_candidates.csv"))
    )
    for candidate_path in candidate_paths:
        for row in read_rows(candidate_path):
            key = condition_key(row)
            if key:
                candidates[key].append(row)

    detail: list[dict[str, object]] = []
    max_budget = budgets[-1]
    average_setting = f"average_of_{max_budget}"
    for key, eval_row in eval_by_key.items():
        group = sorted(candidates.get(key, []), key=candidate_index)
        if len(group) < max_budget:
            raise RuntimeError(f"{key}: expected at least {max_budget} candidates, found {len(group)}")
        scored = [score_candidate(eval_row, candidate_smiles(item), rank) for rank, item in enumerate(group)]
        property_count = int(float(eval_row.get("property_count") or len(selected_properties(eval_row))))

        detail.append(
            {
                "setting": average_setting,
                "condition_id": key,
                "property_count": property_count,
                "candidate_budget": max_budget,
                "validity_value": sum(item.valid for item in scored[:max_budget]) / max_budget,
                "strict_value": sum(item.strict for item in scored[:max_budget]) / max_budget,
                "selected_rank": "",
            }
        )
        for budget in budgets:
            best = max(scored[:budget], key=lambda item: item.selection_score)
            detail.append(
                {
                    "setting": "raw_at_1" if budget == 1 else f"best_of_{budget}",
                    "condition_id": key,
                    "property_count": property_count,
                    "candidate_budget": budget,
                    "validity_value": float(best.valid),
                    "strict_value": float(best.strict),
                    "selected_rank": best.rank,
                }
            )

    settings = [average_setting, "raw_at_1"] + [f"best_of_{value}" for value in budgets if value != 1]
    summary_rows: list[dict[str, object]] = []
    for setting in settings:
        setting_rows = [row for row in detail if row["setting"] == setting]
        for label in [2, 3, 4, 5, 6, 7, "all"]:
            rows = setting_rows if label == "all" else [row for row in setting_rows if row["property_count"] == label]
            validity, validity_low, validity_high = summarize([float(row["validity_value"]) for row in rows])
            strict, strict_low, strict_high = summarize([float(row["strict_value"]) for row in rows])
            summary_rows.append(
                {
                    "setting": setting,
                    "property_count": label,
                    "conditions": len(rows),
                    "candidate_budget": rows[0]["candidate_budget"] if rows else "",
                    "validity": validity,
                    "validity_ci95_low": validity_low,
                    "validity_ci95_high": validity_high,
                    "strict_success_rate": strict,
                    "strict_ci95_low": strict_low,
                    "strict_ci95_high": strict_high,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "budget_sweep_condition_detail.csv", detail)
    write_csv(args.output_dir / "budget_sweep_summary.csv", summary_rows)

    by_key = {(row["setting"], row["property_count"]): row for row in summary_rows}
    lines = [
        f"# {args.report_title}",
        "",
        f"All settings reuse the same {len(eval_rows):,} conditions and {sum(len(group) for group in candidates.values()):,} {args.candidate_description}.",
        f"`{average_setting}` scores every generated candidate; `best_of_K` applies the property-aware finalizer.",
        "",
        "| setting | validity | overall strict | 2p | 3p | 4p | 5p | 6p | 7p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting in settings:
        overall = by_key[(setting, "all")]
        bucket_values = [float(by_key[(setting, count)]["strict_success_rate"]) for count in range(2, 8)]
        lines.append(
            f"| {setting} | {float(overall['validity']):.3f} | {float(overall['strict_success_rate']):.3f} | "
            + " | ".join(f"{value:.3f}" for value in bucket_values)
            + " |"
        )
    reference_avg = statistics.fmean(SKETCHMOL_REFERENCE.values())
    lines.append(
        f"| SketchMol paper reference |  | {reference_avg:.3f} | "
        + " | ".join(f"{SKETCHMOL_REFERENCE[count]:.3f}" for count in range(2, 8))
        + " |"
    )
    lines.extend(
        [
            "",
            "## Overall 95% confidence intervals",
            "",
            "| setting | strict | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for setting in settings:
        row = by_key[(setting, "all")]
        lines.append(
            f"| {setting} | {float(row['strict_success_rate']):.4f} | "
            f"[{float(row['strict_ci95_low']):.4f}, {float(row['strict_ci95_high']):.4f}] |"
        )
    (args.output_dir / "budget_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
