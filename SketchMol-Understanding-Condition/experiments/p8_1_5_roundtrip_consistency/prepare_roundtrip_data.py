#!/usr/bin/env python3
"""Prepare matched forward-only and forward-plus-cycle training supports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ID_FIELDS = ("variant_id", "condition_id", "sample_id", "pair_id")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    return next((str(row.get(key, "")).strip() for key in ID_FIELDS if str(row.get(key, "")).strip()), "")


def balanced(rows: list[dict[str, str]], limit: int, seed: int, *, edit: bool) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row["task_mode"] = "edit" if edit else "de_novo"
        source = str(row.get("source_smiles", "") or "").strip()
        target = str(row.get("target_smiles", "") or "").strip()
        if not target or (edit and (not source or source == target)):
            continue
        group = str(row.get("benchmark_task", "") or row.get("instruction", "") or "edit") if edit else f"{int(float(row.get('property_count') or 0))}p"
        groups[group].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    chosen: list[dict[str, str]] = []
    names = sorted(groups)
    cursor = {name: 0 for name in names}
    while names and len(chosen) < limit:
        alive = []
        for name in names:
            if cursor[name] < len(groups[name]) and len(chosen) < limit:
                chosen.append(groups[name][cursor[name]])
                cursor[name] += 1
            if cursor[name] < len(groups[name]):
                alive.append(name)
        names = alive
    return chosen


def invert_direction(value: object) -> object:
    raw = str(value or "").strip()
    lower = raw.lower()
    pairs = {
        "increase": "decrease", "decrease": "increase", "up": "down", "down": "up",
        "higher": "lower", "lower": "higher", "maximize": "minimize", "minimize": "maximize",
        "1": "-1", "+1": "-1", "-1": "1", "↑": "↓", "↓": "↑",
    }
    return pairs.get(lower, value)


def invert_json(value: str) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value

    def walk(item):
        if isinstance(item, list):
            return [walk(value) for value in item]
        if isinstance(item, dict):
            return {key: invert_direction(value) if key in {"direction", "operation", "trend"} else walk(value) for key, value in item.items()}
        return invert_direction(item)

    return json.dumps(walk(payload), sort_keys=True)


def invert_text(value: str) -> str:
    placeholders = {
        "increase": "__P815_DEC__", "decrease": "__P815_INC__",
        "higher": "__P815_LOW__", "lower": "__P815_HIGH__",
        "maximize": "__P815_MIN__", "minimize": "__P815_MAX__",
        "↑": "__P815_DOWN__", "↓": "__P815_UP__",
    }
    out = str(value or "")
    for source, marker in placeholders.items():
        out = re.sub(rf"\b{source}\b" if source.isalpha() else re.escape(source), marker, out, flags=re.IGNORECASE)
    replacements = {
        "__P815_DEC__": "decrease", "__P815_INC__": "increase",
        "__P815_LOW__": "lower", "__P815_HIGH__": "higher",
        "__P815_MIN__": "minimize", "__P815_MAX__": "maximize",
        "__P815_DOWN__": "↓", "__P815_UP__": "↑",
    }
    for marker, target in replacements.items():
        out = out.replace(marker, target)
    return out


def cycle_row(forward: dict[str, str]) -> dict[str, str]:
    row = dict(forward)
    row["source_smiles"] = str(forward.get("target_smiles", "") or "").strip()
    row["target_smiles"] = str(forward.get("source_smiles", "") or "").strip()
    row["task_mode"] = "edit"
    row["roundtrip_role"] = "cycle_inverse"
    # Absolute property targets must describe the molecule reconstructed by
    # the inverse leg (the original source), not the forward target.  The
    # public Table1 pack already carries both source_* and target_* values.
    for key in list(forward):
        if not key.startswith("target_") or key == "target_smiles":
            continue
        suffix = key[len("target_") :]
        source_key = f"source_{suffix}"
        if source_key in forward and str(forward.get(source_key, "") or "").strip():
            row[key] = str(forward[source_key])
            row[source_key] = str(forward.get(key, "") or "")
    for key in list(row):
        if key.endswith("_direction"):
            row[key] = str(invert_direction(row[key]))
        elif key in {"instruction_tasks_json", "external_property_directions_json"}:
            row[key] = invert_json(row[key])
        elif key in {"instruction", "prompt", "text", "condition_text"}:
            row[key] = invert_text(row[key])
    for key in ID_FIELDS:
        if str(row.get(key, "") or "").strip():
            row[key] = f"{row[key]}__cycle"
    return row


def load_features(path: Path):
    index = read_csv(path / "index.csv")
    query = np.load(path / "query_tokens.npy")
    pooled = np.load(path / "pooled.npy")
    lookup = {}
    for idx, row in enumerate(index):
        for key in ID_FIELDS:
            value = str(row.get(key, "") or "").strip()
            if value and value not in lookup:
                lookup[value] = idx
    return index, query, pooled, lookup


def merge_forward_features(output: Path, rows: list[dict[str, str]], feature_dirs: list[Path]) -> None:
    stores = [load_features(path) for path in feature_dirs]
    index_rows, query_rows, pooled_rows = [], [], []
    missing = []
    for row in rows:
        hit = None
        for store in stores:
            for key in ID_FIELDS:
                value = str(row.get(key, "") or "").strip()
                if value and value in store[3]:
                    hit = store, store[3][value]
                    break
            if hit:
                break
        if not hit:
            missing.append(row_key(row))
            continue
        store, idx = hit
        record = dict(store[0][idx])
        record["row_index"] = str(len(index_rows))
        index_rows.append(record)
        query_rows.append(np.asarray(store[1][idx]))
        pooled_rows.append(np.asarray(store[2][idx]))
    if missing:
        raise SystemExit(f"Missing frozen forward features for {len(missing)} rows: {missing[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "index.csv", index_rows)
    np.save(output / "query_tokens.npy", np.stack(query_rows))
    np.save(output / "pooled.npy", np.stack(pooled_rows))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--denovo-features", required=True, type=Path)
    parser.add_argument("--edit-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-mode", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    denovo = balanced(read_csv(args.denovo_csv), args.per_mode, args.seed, edit=False)
    edit = balanced(read_csv(args.edit_csv), args.per_mode, args.seed + 1, edit=True)
    forward: list[dict[str, str]] = []
    for idx in range(max(len(denovo), len(edit))):
        if idx < len(denovo):
            denovo[idx]["roundtrip_role"] = "forward"
            forward.append(denovo[idx])
        if idx < len(edit):
            edit[idx]["roundtrip_role"] = "forward"
            forward.append(edit[idx])
    cycles = [cycle_row(row) for row in edit]
    r2 = forward + cycles
    random.Random(args.seed + 2).shuffle(r2)
    write_csv(args.output_dir / "r1_forward.csv", forward)
    write_csv(args.output_dir / "r2_forward_cycle.csv", r2)
    write_csv(args.output_dir / "cycle_rows.csv", cycles)
    merge_forward_features(args.output_dir / "features", forward, [args.denovo_features, args.edit_features])
    payload = {
        "protocol": "p8_1_5_roundtrip_support_v1",
        "seed": args.seed,
        "denovo_forward_rows": len(denovo),
        "edit_forward_rows": len(edit),
        "cycle_rows": len(cycles),
        "r1_rows": len(forward),
        "r2_rows": len(r2),
        "identity_forward_edits": sum(row["source_smiles"] == row["target_smiles"] for row in edit),
        "identity_cycles": sum(row["source_smiles"] == row["target_smiles"] for row in cycles),
        "r1_sha256": digest(args.output_dir / "r1_forward.csv"),
        "cycle_sha256": digest(args.output_dir / "cycle_rows.csv"),
        "causal_factor": {"r1_cycle_weight": 0.0, "r2_cycle_weight": 1.0},
        "contract": {
            "checkpoint_count": 1, "decoder_count": 1, "output_language": "full SMILES",
            "external_router": False, "materializer": False, "property_rerank": False,
        },
    }
    (args.output_dir / "support_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
