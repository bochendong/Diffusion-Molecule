#!/usr/bin/env python3
"""Prepare leak-free 6p/7p de-novo plus edit training support for P8.1.11."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
P812_DIR = SCRIPT_DIR.parent / "p8_1_2_unified_transduction"
sys.path.insert(0, str(P812_DIR))
import transduction_oracle as transduction  # noqa: E402


IDS = ("variant_id", "condition_id", "sample_id", "pair_id")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def denovo_program(smiles: str) -> list[str]:
    tokens = transduction.sf_tokens(transduction.canonical(smiles))
    return [transduction.START, transduction.INSERT, *tokens, transduction.INSERT_END, transduction.STOP]


def select_denovo(rows: list[dict[str, str]], per_count: int, seed: int) -> list[dict[str, str]]:
    output = []
    for count in (6, 7):
        pool = [dict(row) for row in rows if int(float(row.get("property_count") or 0)) == count and row.get("target_smiles")]
        random.Random(seed + count).shuffle(pool)
        for row in pool[:per_count]:
            row["task_mode"] = "de_novo"
            row["policy_target_tokens_json"] = json.dumps(denovo_program(row["target_smiles"]))
            row["p811_train_origin"] = f"train_denovo_{count}p"
            output.append(row)
    return output


def select_edit(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    # Avoid expensive assay oracles in the online reward. This is still a
    # train-only edit support and spans the RDKit-computable Table1 properties.
    banned = ("gsk3", "drd2", "binding affinity")
    pool = []
    for raw in rows:
        row = dict(raw)
        if not row.get("source_smiles") or not row.get("policy_target_tokens_json"):
            continue
        text = " ".join(str(row.get(key, "") or "") for key in ("instruction", "benchmark_task")).lower()
        if any(term in text for term in banned):
            continue
        row["task_mode"] = "edit"
        row["p811_train_origin"] = "train_edit_nonassay"
        pool.append(row)
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def load_features(path: Path):
    index = read(path / "index.csv"); query = np.load(path / "query_tokens.npy"); pooled = np.load(path / "pooled.npy")
    lookup = {}
    for idx, row in enumerate(index):
        for key in IDS:
            value = str(row.get(key, "") or "").strip()
            if value and value not in lookup: lookup[value] = idx
    return index, query, pooled, lookup


def merge_features(output: Path, rows: list[dict[str, str]], dirs: list[Path]) -> list[str]:
    stores = [load_features(path) for path in dirs]
    index_rows, queries, pooled, missing = [], [], [], []
    for row in rows:
        hit = None
        for store in stores:
            for key in IDS:
                value = str(row.get(key, "") or "").strip()
                if value and value in store[3]: hit = store, store[3][value]; break
            if hit: break
        if not hit:
            # This exactly matches P8.1.2-R1 training: FeatureStore falls back
            # to deterministic structured condition features when a synthetic
            # transduction row has no frozen VLM id.
            missing.append(str(next((row.get(k) for k in IDS if row.get(k)), "?")))
            continue
        store, idx = hit; record = dict(store[0][idx]); record["row_index"] = str(len(index_rows))
        index_rows.append(record); queries.append(np.asarray(store[1][idx])); pooled.append(np.asarray(store[2][idx]))
    output.mkdir(parents=True, exist_ok=True); write(output / "index.csv", index_rows)
    np.save(output / "query_tokens.npy", np.stack(queries)); np.save(output / "pooled.npy", np.stack(pooled))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denovo-csv", required=True, type=Path); parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--denovo-features", required=True, type=Path); parser.add_argument("--edit-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--per-denovo-count", type=int, default=32)
    parser.add_argument("--edit-limit", type=int, default=64); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    denovo = select_denovo(read(args.denovo_csv), args.per_denovo_count, args.seed)
    edit = select_edit(read(args.edit_csv), args.edit_limit, args.seed + 11)
    rows = []
    for idx in range(max(len(denovo), len(edit))):
        if idx < len(denovo): rows.append(denovo[idx])
        if idx < len(edit): rows.append(edit[idx])
    write(args.output_dir / "train_rows.csv", rows)
    missing_features = merge_features(args.output_dir / "features", rows, [args.denovo_features, args.edit_features])
    payload = {
        "protocol": "p8_1_11_train_only_support_v1", "seed": args.seed, "rows": len(rows),
        "denovo_6p": sum(row["p811_train_origin"] == "train_denovo_6p" for row in rows),
        "denovo_7p": sum(row["p811_train_origin"] == "train_denovo_7p" for row in rows),
        "edit_nonassay": sum(row["p811_train_origin"] == "train_edit_nonassay" for row in rows),
        "eval_rows_used": 0, "eval_targets_used": 0,
        "deterministic_feature_fallback_rows": len(missing_features),
        "deterministic_feature_fallback_ids": missing_features,
        "sha256": hashlib.sha256((args.output_dir / "train_rows.csv").read_bytes()).hexdigest(),
    }
    (args.output_dir / "support_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["denovo_6p"] != args.per_denovo_count or payload["denovo_7p"] != args.per_denovo_count or not edit:
        raise SystemExit("P8.1.11 training support incomplete")
    return 0


if __name__ == "__main__": raise SystemExit(main())
