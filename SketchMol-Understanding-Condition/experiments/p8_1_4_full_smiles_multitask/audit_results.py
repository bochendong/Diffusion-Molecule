#!/usr/bin/env python3
"""Audit one-checkpoint/full-SMILES claims and source-copy collapse."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--edit-candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_state = base["model_state"]
    trained_state = trained["model_state"]
    common = sorted(set(base_state) & set(trained_state))
    changed = [name for name in common if not torch.equal(base_state[name], trained_state[name])]
    candidates = rows(args.edit_candidates)
    identity = 0
    nonempty = 0
    for row in candidates:
        generated = str(row.get("direct_candidate_canonical_smiles") or row.get("generated_smiles") or "").strip()
        source = str(row.get("source_smiles") or "").strip()
        if generated:
            nonempty += 1
            identity += int(generated == source)
    config = dict(trained.get("model_config", {}))
    payload = {
        "protocol": "p8_1_4_one_checkpoint_audit_v1",
        "one_checkpoint": True,
        "one_shared_decoder": "decoder.layers.0.self_attn.in_proj_weight" in trained_state,
        "one_shared_output_head": "output.weight" in trained_state,
        "same_smiles_vocabulary": base.get("vocab") == trained.get("vocab"),
        "condition_layout": str(trained.get("args", {}).get("condition_layout", "")),
        "source_copy_pointer": bool(config.get("source_copy_aware", False)),
        "router": False,
        "interpreter": False,
        "materializer": False,
        "property_rerank": False,
        "common_base_tensors": len(common),
        "changed_base_tensors": changed,
        "base_path_bitwise_protected": not changed,
        "candidate_rows": len(candidates),
        "nonempty_candidates": nonempty,
        "identity_copy_candidates": identity,
        "identity_copy_rate": identity / max(nonempty, 1),
        "noncopy_candidate_rate": (nonempty - identity) / max(len(candidates), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if config.get("source_copy_aware", False) or changed:
        raise SystemExit("Protected-adapter audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
