#!/usr/bin/env python3
"""Build the small balanced P8.1.4 mixed set and its frozen feature store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key_for(row: dict[str, str]) -> str:
    for key in ("variant_id", "condition_id", "sample_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def balanced(rows: list[dict[str, str]], limit: int, seed: int, *, edit: bool) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        row["task_mode"] = "edit" if edit else "de_novo"
        target = str(row.get("target_smiles", "") or "").strip()
        source_smiles = str(row.get("source_smiles", "") or "").strip()
        if not target or (edit and (not source_smiles or source_smiles == target)):
            continue
        if edit:
            group = str(row.get("benchmark_task", "") or row.get("instruction", "") or "edit")
        else:
            group = f"{int(float(row.get('property_count') or 0))}p"
        groups[group].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    names = sorted(groups)
    chosen: list[dict[str, str]] = []
    cursors = {name: 0 for name in names}
    while names and len(chosen) < limit:
        next_names = []
        for name in names:
            cursor = cursors[name]
            if cursor < len(groups[name]) and len(chosen) < limit:
                chosen.append(groups[name][cursor])
                cursors[name] += 1
            if cursors[name] < len(groups[name]):
                next_names.append(name)
        names = next_names
    return chosen


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_dir(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, int]]:
    rows = read_rows(path / "index.csv")
    query = np.load(path / "query_tokens.npy")
    pooled = np.load(path / "pooled.npy")
    lookup: dict[str, int] = {}
    for idx, row in enumerate(rows):
        for key in ("variant_id", "condition_id", "sample_id", "pair_id"):
            value = str(row.get(key, "") or "").strip()
            if value and value not in lookup:
                lookup[value] = idx
    return rows, query, pooled, lookup


def merge_features(
    output: Path,
    selected: list[dict[str, str]],
    sources: list[tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, int]]],
) -> dict[str, object]:
    index_rows: list[dict[str, str]] = []
    query_rows: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []
    missing: list[str] = []
    for row in selected:
        candidates = [
            str(row.get(name, "") or "").strip()
            for name in ("variant_id", "condition_id", "sample_id", "pair_id")
        ]
        hit = None
        for src in sources:
            for candidate in candidates:
                if candidate and candidate in src[3]:
                    hit = (src, src[3][candidate])
                    break
            if hit is not None:
                break
        if hit is None:
            missing.append(key_for(row))
            continue
        src, idx = hit
        record = dict(src[0][idx])
        record["row_index"] = str(len(index_rows))
        record["task_mode"] = str(row.get("task_mode", ""))
        index_rows.append(record)
        query_rows.append(np.asarray(src[1][idx]))
        pooled_rows.append(np.asarray(src[2][idx]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "index.csv", index_rows)
    if not query_rows:
        raise ValueError("No selected rows had frozen condition features")
    np.save(output / "query_tokens.npy", np.stack(query_rows, axis=0))
    np.save(output / "pooled.npy", np.stack(pooled_rows, axis=0))
    return {"selected": len(selected), "feature_rows": len(index_rows), "missing": missing[:20]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--denovo-features", required=True, type=Path)
    parser.add_argument("--edit-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-mode", type=int, default=2000)
    parser.add_argument("--r2-edit-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    denovo = balanced(read_rows(args.denovo_csv), args.per_mode, args.seed, edit=False)
    edit = balanced(read_rows(args.edit_csv), args.per_mode, args.seed + 1, edit=True)
    mixed: list[dict[str, str]] = []
    for idx in range(max(len(denovo), len(edit))):
        if idx < len(denovo):
            mixed.append(denovo[idx])
        if idx < len(edit):
            mixed.append(edit[idx])
    r2_edit = balanced(edit, min(args.r2_edit_limit, len(edit)), args.seed + 2, edit=True)
    write_csv(args.output_dir / "mixed_train.csv", mixed)
    write_csv(args.output_dir / "r2_edit_train.csv", r2_edit)
    sources = [load_feature_dir(args.denovo_features), load_feature_dir(args.edit_features)]
    feature_audit = merge_features(args.output_dir / "mixed_features", mixed, sources)
    payload = {
        "protocol": "p8_1_4_full_smiles_mixed_support_v1",
        "denovo_rows": len(denovo),
        "edit_rows": len(edit),
        "mixed_rows": len(mixed),
        "r2_edit_rows": len(r2_edit),
        "identity_edit_rows": sum(row.get("source_smiles") == row.get("target_smiles") for row in edit),
        "feature_audit": feature_audit,
        "task_contract": {
            "condition_layout": "unified",
            "task_tokens": ["GENERATE", "EDIT"],
            "decoder": "one shared autoregressive full-SMILES decoder",
            "output_language": "one shared SMILES vocabulary",
            "router": False,
            "materializer": False,
            "property_rerank": False,
        },
    }
    digest = hashlib.sha256((args.output_dir / "mixed_train.csv").read_bytes()).hexdigest()
    payload["mixed_train_sha256"] = digest
    (args.output_dir / "support_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if feature_audit["feature_rows"] != len(mixed):
        raise SystemExit("Frozen feature coverage is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
