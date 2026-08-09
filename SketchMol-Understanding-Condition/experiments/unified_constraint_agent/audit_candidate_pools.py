#!/usr/bin/env python3
"""Audit raw, support, and system-selection success from fixed candidate prefixes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RunSpec:
    name: str
    suite: str
    candidate_csv: Path
    budget: int = 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def number(value: object, default: float | None = None) -> float | None:
    try:
        parsed = float(str(value or "").strip())
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_id(row: Mapping[str, object], index: int = 0) -> str:
    for key in ("condition_id", "sample_id", "example_id", "variant_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return f"row_{index:08d}"


def generated_smiles(row: Mapping[str, object]) -> str:
    return str(
        row.get("generated_smiles", "")
        or row.get("predicted_smiles", "")
        or row.get("candidate_smiles", "")
        or ""
    ).strip()


def valid_candidate(row: Mapping[str, object]) -> bool:
    for key in ("external_valid", "valid_smiles", "is_valid", "valid"):
        if str(row.get(key, "") or "").strip():
            return truthy(row.get(key))
    return bool(generated_smiles(row))


def property_fraction(row: Mapping[str, object]) -> float:
    for key in ("unified_property_success_fraction", "strict_fraction", "property_success_fraction"):
        value = number(row.get(key))
        if value is not None:
            return max(0.0, min(1.0, value))
    payload = row.get("external_property_success_json")
    if str(payload or "").strip():
        try:
            parsed = json.loads(str(payload))
        except (ValueError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, Mapping) and parsed:
            return sum(truthy(value) for value in parsed.values()) / len(parsed)
    if str(row.get("external_all_property_success", "") or "").strip():
        return 1.0 if truthy(row.get("external_all_property_success")) else 0.0
    return 0.0


def strict_success(row: Mapping[str, object]) -> bool:
    for key in ("external_official_success", "strict_success", "success_strict"):
        if str(row.get(key, "") or "").strip():
            return truthy(row.get(key))
    return valid_candidate(row) and property_fraction(row) >= 1.0 - 1e-9


def generation_rank(row: Mapping[str, object], index: int) -> tuple[float, int]:
    for key in ("generation_rank", "sample_rank", "raw_rank"):
        value = number(row.get(key))
        if value is not None:
            return value, index
    value = number(row.get("candidate_rank"))
    return (value if value is not None else float(index + 1)), index


def selection_rank(row: Mapping[str, object], index: int) -> tuple[float, int]:
    if truthy(row.get("candidate_selected")):
        return 0.0, index
    for key in ("candidate_rank", "unified_selected_candidate_rank", "selection_rank"):
        value = number(row.get(key))
        if value is not None:
            return value, index
    return generation_rank(row, index)


def property_distance(row: Mapping[str, object]) -> float | None:
    for key in ("unified_property_distance", "normalized_property_distance", "property_distance"):
        value = number(row.get(key))
        if value is not None:
            return value
    return None


def source_similarity(row: Mapping[str, object]) -> float | None:
    for key in ("external_source_tanimoto", "source_tanimoto", "similarity"):
        value = number(row.get(key))
        if value is not None:
            return value
    return None


def split_value(row: Mapping[str, object]) -> str:
    return str(row.get("external_task_split", "") or row.get("split", "") or "all").strip().lower()


def property_count_value(row: Mapping[str, object]) -> str:
    value = str(row.get("property_count", "") or "").strip()
    return value if value else "all"


def load_specs(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_runs = payload.get("runs", []) if isinstance(payload, Mapping) else []
    specs = []
    for item in raw_runs:
        if not isinstance(item, Mapping):
            continue
        specs.append(
            RunSpec(
                name=str(item["name"]),
                suite=str(item.get("suite", item["name"])),
                candidate_csv=Path(str(item["candidate_csv"])),
                budget=max(1, int(item.get("budget", 20))),
            )
        )
    if not specs:
        raise ValueError("Audit config must contain at least one run")
    return specs


def summarize_groups(groups: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, float | int]:
    total = len(groups)
    raw = 0
    any_hit = 0
    selected = 0
    valid_any = 0
    best_fraction = 0.0
    mean_valid = 0.0
    for items in groups:
        raw += strict_success(items[0])
        any_hit += any(strict_success(item) for item in items)
        chosen = min(enumerate(items), key=lambda pair: selection_rank(pair[1], pair[0]))[1]
        selected += strict_success(chosen)
        valid_any += any(valid_candidate(item) for item in items)
        best_fraction += max((property_fraction(item) for item in items), default=0.0)
        mean_valid += sum(valid_candidate(item) for item in items)
    denominator = max(total, 1)
    return {
        "groups": total,
        "raw_at_1": raw / denominator,
        "any_hit_at_k": any_hit / denominator,
        "selected_at_k": selected / denominator,
        "valid_any_at_k": valid_any / denominator,
        "mean_best_property_fraction": best_fraction / denominator,
        "mean_valid_candidates": mean_valid / denominator,
        "support_gain": (any_hit - raw) / denominator,
        "selection_gain": (selected - raw) / denominator,
        "selection_miss": (any_hit - selected) / denominator,
    }


def audit_run(spec: RunSpec) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_rows(spec.candidate_csv)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_id(row, index)].append((index, row))
    prefixes = []
    for indexed in grouped.values():
        ordered = [row for _, row in sorted(indexed, key=lambda pair: generation_rank(pair[1], pair[0]))]
        prefixes.append(ordered[: spec.budget])

    summary_rows = []
    strata = {
        ("all", "all"): prefixes,
    }
    for items in prefixes:
        split = split_value(items[0])
        count = property_count_value(items[0])
        if split != "all":
            strata.setdefault((split, "all"), []).append(items)
        if count != "all":
            strata.setdefault(("all", count), []).append(items)
    for (split, property_count), groups in sorted(strata.items()):
        values = summarize_groups(groups)
        summary_rows.append(
            {
                "run": spec.name,
                "suite": spec.suite,
                "budget": spec.budget,
                "split": split,
                "property_count": property_count,
                **values,
            }
        )
    manifest = {
        **asdict(spec),
        "candidate_csv": str(spec.candidate_csv),
        "candidate_sha256": sha256(spec.candidate_csv),
        "candidate_rows": len(rows),
        "condition_groups": len(prefixes),
        "complete_budget_groups": sum(len(items) >= spec.budget for items in prefixes),
    }
    return summary_rows, manifest


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    overall = [row for row in rows if row["split"] == "all" and row["property_count"] == "all"]
    lines = [
        "# Unified Constraint Agent Candidate Audit",
        "",
        "| Run | Suite | k | Groups | raw@1 | any-hit@k | selected@k | Support gain | Selection miss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            "| {run} | {suite} | {budget} | {groups} | {raw_at_1:.4f} | {any_hit_at_k:.4f} | "
            "{selected_at_k:.4f} | {support_gain:.4f} | {selection_miss:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "`raw@1` measures first-shot policy control. `any-hit@k` measures candidate-support reachability. ",
            "`selected@k` measures the existing system selection encoded in candidate rank/selection fields.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    manifests = []
    for spec in load_specs(args.config_json):
        rows, manifest = audit_run(spec)
        summary_rows.extend(rows)
        manifests.append(manifest)
    write_csv(args.output_dir / "candidate_audit.csv", summary_rows)
    write_report(args.output_dir / "candidate_audit.md", summary_rows)
    payload = {
        "protocol": "unified_constraint_agent_candidate_audit_v1",
        "config_json": str(args.config_json),
        "runs": manifests,
        "summary_rows": summary_rows,
    }
    (args.output_dir / "candidate_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
