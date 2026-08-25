#!/usr/bin/env python3
"""Collect official budget-sweep summaries for paired frozen P17/P18."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p17-summary", required=True, type=Path)
    p.add_argument("--p18-summary", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    settings = ("average_of_40", "raw_at_1", "best_of_4", "best_of_8", "best_of_20", "best_of_40")
    table = []
    payload = {"protocol": "p20_r2_frozen_denovo_2p4p_fair_budget40_v1", "primary_setting": "best_of_40", "models": {}}
    for model, path in (("p17", args.p17_summary), ("p18", args.p18_summary)):
        rows = read(path)
        by_key = {(row["setting"], row["property_count"]): row for row in rows}
        payload["models"][model] = {}
        for setting in settings:
            all_row = by_key[(setting, "all")]
            item = {
                "conditions": int(all_row["conditions"]),
                "candidate_budget": int(all_row["candidate_budget"]),
                "validity": float(all_row["validity"]),
                "overall_strict": float(all_row["strict_success_rate"]),
                "overall_strict_ci95": [float(all_row["strict_ci95_low"]), float(all_row["strict_ci95_high"])],
            }
            for count in (2, 3, 4):
                row = by_key[(setting, str(count))]
                item[f"{count}p_strict"] = float(row["strict_success_rate"])
                item[f"{count}p_successes"] = round(float(row["strict_success_rate"]) * int(row["conditions"]))
            payload["models"][model][setting] = item
            table.append({"model": model, "setting": setting, **item})
    payload["primary_fill_table"] = {model: payload["models"][model]["best_of_40"] for model in ("p17", "p18")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "setting", "conditions", "candidate_budget", "validity", "overall_strict", "overall_strict_ci95",
              "2p_strict", "2p_successes", "3p_strict", "3p_successes", "4p_strict", "4p_successes"]
    with (args.output_dir / "fair_table_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(table)
    (args.output_dir / "aggregate_r2.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["primary_fill_table"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
