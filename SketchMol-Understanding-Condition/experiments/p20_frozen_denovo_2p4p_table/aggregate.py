#!/usr/bin/env python3
"""Apply the official SketchMol strict predicate to paired raw candidate prefixes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent.parent


def load_official():
    path = PROJECT / "scripts" / "evaluate_univideo_image_benchmark.py"
    spec = importlib.util.spec_from_file_location("official_denovo_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--p17-candidates", required=True, type=Path)
    parser.add_argument("--p18-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != manifest["locked_sha256"]["reference"]:
        raise AssertionError("locked reference hash changed")
    refs = {str(row["condition_id"]): row for row in read_csv(args.reference)}
    official = load_official()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": manifest["protocol"],
        "status_label": manifest["status_label"],
        "conditions": len(refs),
        "n_per_property_count": 100,
        "candidate_budgets": [1, 4, 8],
        "official_strict_tolerances": official.SKETCHMOL_STRICT_TOLERANCE,
        "models": {},
    }
    table_rows = []
    for model, path in (("p17", args.p17_candidates), ("p18", args.p18_candidates)):
        raw = read_csv(path)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        detail = []
        for candidate in raw:
            key = str(candidate["condition_id"])
            if key not in refs:
                raise AssertionError(f"unknown condition {key}")
            merged = dict(refs[key])
            merged.update(candidate)
            merged["SMILES"] = candidate.get("direct_candidate_canonical_smiles", "")
            merged["method"] = model
            decoded = official._decode_row_with_options(
                merged, method=model, smiles_column="SMILES", accept_direct_smiles=True
            )
            decoded["candidate_rank"] = int(float(candidate["candidate_rank"]))
            grouped[key].append(decoded)
            detail.append(decoded)
        if len(raw) != len(refs) * 8 or any(len(rows) != 8 for rows in grouped.values()):
            raise AssertionError(f"{model}: expected exactly eight candidates per condition")
        detail_path = args.output_dir / f"{model}.candidate_details.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            fields = []
            seen = set()
            for row in detail:
                for key in row:
                    if key not in seen:
                        seen.add(key); fields.append(key)
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(detail)
        model_metrics = {}
        for k in (1, 4, 8):
            by_count = {}
            all_prefix = []
            for count in (2, 3, 4):
                conditions = [rows for rows in grouped.values() if int(rows[0]["property_count"]) == count]
                prefixes = [sorted(rows, key=lambda row: row["candidate_rank"])[:k] for rows in conditions]
                all_prefix.extend(prefixes)
                hits = sum(any(truthy(row["strict_success"]) for row in rows) for rows in prefixes)
                valid = sum(any(truthy(row["valid"]) for row in rows) for rows in prefixes)
                by_count[f"{count}p"] = {
                    "n": len(prefixes), "strict_successes": hits, "strict_rate": hits / len(prefixes),
                    "valid_conditions": valid, "validity_anyk": valid / len(prefixes),
                }
            valid_conditions = sum(any(truthy(row["valid"]) for row in rows) for rows in all_prefix)
            unique_valid = {
                str(row["generated_smiles"]) for rows in all_prefix for row in rows if truthy(row["valid"])
            }
            valid_candidates = sum(sum(truthy(row["valid"]) for row in rows) for rows in all_prefix)
            model_metrics[str(k)] = {
                "by_property_count": by_count,
                "validity_all_anyk": valid_conditions / len(all_prefix),
                "unique_valid_smiles": len(unique_valid),
                "uniqueness_in_valid_candidates": len(unique_valid) / valid_candidates if valid_candidates else 0.0,
                "valid_candidate_rate": valid_candidates / (len(all_prefix) * k),
            }
            row = {"model": model, "k": k, "validity_all": model_metrics[str(k)]["validity_all_anyk"]}
            for count in (2, 3, 4):
                row[f"{count}p_strict"] = by_count[f"{count}p"]["strict_rate"]
                row[f"{count}p_successes"] = by_count[f"{count}p"]["strict_successes"]
            row["unique_valid_smiles"] = len(unique_valid)
            row["uniqueness_in_valid_candidates"] = model_metrics[str(k)]["uniqueness_in_valid_candidates"]
            table_rows.append(row)
        payload["models"][model] = model_metrics
    table = args.output_dir / "table_rows.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0])); writer.writeheader(); writer.writerows(table_rows)
    payload["historical_r1_k1_ablation_not_fill_table"] = {
        model: payload["models"][model]["1"] for model in ("p17", "p18")
    }
    (args.output_dir / "aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["primary_fill_table"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
