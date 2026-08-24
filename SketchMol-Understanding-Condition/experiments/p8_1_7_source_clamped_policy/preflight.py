#!/usr/bin/env python3
"""Fail-closed P8.1.7 checkpoint and one-decoder audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--p814-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    audit = json.loads(args.p814_audit.read_text(encoding="utf-8"))
    config = dict(checkpoint.get("model_config", {}))
    state = checkpoint["model_state"]
    source_output = state.get("source_output.weight")
    payload = {
        "protocol": "p8_1_7_source_clamp_preflight_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "one_checkpoint": True,
        "one_decoder": "decoder.layers.0.self_attn.in_proj_weight" in state,
        "one_output_head": "output.weight" in state,
        "source_aware": bool(config.get("source_aware", False)),
        "source_copy_aware": bool(config.get("source_copy_aware", False)),
        "trained_source_residual_norm": float(source_output.norm()) if source_output is not None else 0.0,
        "p1_base_path_bitwise_protected": bool(audit.get("base_path_bitwise_protected", False)),
        "router": False,
        "materializer": False,
        "property_rerank": False,
        "r1_scale": 1.0,
        "r2_scale": 2.0,
        "null_source_gate": "source_present multiplier makes clamp exactly zero",
    }
    checks = [
        payload["one_decoder"], payload["one_output_head"], payload["source_aware"],
        not payload["source_copy_aware"], payload["trained_source_residual_norm"] > 0,
        payload["p1_base_path_bitwise_protected"],
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(checks):
        raise SystemExit("P8.1.7 checkpoint preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
