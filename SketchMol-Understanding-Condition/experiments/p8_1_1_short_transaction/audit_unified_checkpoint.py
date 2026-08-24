#!/usr/bin/env python3
"""Fail-closed audit for one-checkpoint P8.1.1 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = set()
    for row in rows:
        value = next((str(row.get(k, "")).strip() for k in ("variant_id", "condition_id", "sample_id", "pair_id") if str(row.get(k, "")).strip()), "")
        if value:
            out.add(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--denovo-eval-csv", required=True, type=Path)
    parser.add_argument("--edit-eval-csv", required=True, type=Path)
    parser.add_argument("--denovo-summary", type=Path)
    parser.add_argument("--edit-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_vocab = dict(base["vocab"])
    trained_vocab = dict(trained["vocab"])
    checks: dict[str, bool] = {}
    checks["legacy_vocab_ids_exact"] = all(trained_vocab.get(token) == idx for token, idx in base_vocab.items())
    base_state = base["model_state"]
    trained_state = trained["model_state"]
    protected_exact = True
    mismatches: list[str] = []
    for name, tensor in base_state.items():
        if name not in trained_state:
            protected_exact = False
            mismatches.append(f"missing:{name}")
            continue
        candidate = trained_state[name]
        if name in {"token_embedding.weight", "output.weight", "output.bias"}:
            slices = tuple(slice(0, size) for size in tensor.shape)
            candidate = candidate[slices]
        if tensor.shape != candidate.shape or not torch.equal(tensor.cpu(), candidate.cpu()):
            protected_exact = False
            mismatches.append(name)
    checks["legacy_denovo_parameters_bit_exact"] = protected_exact
    checks["source_aware_single_decoder"] = bool(trained["model_config"].get("source_aware", False))
    train_ids = ids(args.train_csv)
    denovo_ids = ids(args.denovo_eval_csv)
    edit_ids = ids(args.edit_eval_csv)
    checks["no_train_denovo_id_overlap"] = not bool(train_ids & denovo_ids)
    checks["no_train_edit_id_overlap"] = not bool(train_ids & edit_ids)
    checkpoint_hash = sha256(args.checkpoint)
    summary_hashes = []
    for summary_path in (args.denovo_summary, args.edit_summary):
        if summary_path:
            summary_hashes.append(json.loads(summary_path.read_text(encoding="utf-8"))["checkpoint_sha256"])
    checks["same_checkpoint_for_both_arms"] = not summary_hashes or all(value == checkpoint_hash for value in summary_hashes)
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "protected_parameter_mismatches": mismatches,
        "train_denovo_overlap_count": len(train_ids & denovo_ids),
        "train_edit_overlap_count": len(train_ids & edit_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

