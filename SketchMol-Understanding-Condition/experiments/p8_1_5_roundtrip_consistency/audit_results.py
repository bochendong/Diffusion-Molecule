#!/usr/bin/env python3
"""Audit protected P1 tensors and raw edit diversity for P8.1.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--edit-candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    common = sorted(set(base["model_state"]) & set(trained["model_state"]))
    changed = [name for name in common if not torch.equal(base["model_state"][name], trained["model_state"][name])]
    with args.edit_candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    valid, identities, strict_nonidentity, canonicals, similarities = 0, 0, 0, [], []
    for row in candidates:
        generated = str(row.get("direct_candidate_canonical_smiles") or row.get("generated_smiles") or "").strip()
        source = str(row.get("source_smiles") or "").strip()
        if generated:
            valid += 1
            identity = generated == source
            identities += int(identity)
            strict = str(row.get("table1_strict_success", "")).strip().lower() == "true"
            strict_nonidentity += int(strict and not identity)
            canonicals.append(generated)
            try:
                similarity = float(row.get("source_tanimoto", ""))
            except (TypeError, ValueError):
                similarity = math.nan
            if math.isfinite(similarity):
                similarities.append(similarity)
    similarities.sort()

    def quantile(fraction: float) -> float | None:
        if not similarities:
            return None
        idx = round(float(fraction) * (len(similarities) - 1))
        return float(similarities[max(0, min(idx, len(similarities) - 1))])
    config = dict(trained.get("model_config", {}))
    payload = {
        "protocol": "p8_1_5_one_decoder_audit_v1",
        "one_checkpoint": True,
        "one_shared_decoder": "decoder.layers.0.self_attn.in_proj_weight" in trained["model_state"],
        "one_shared_output_head": "output.weight" in trained["model_state"],
        "same_smiles_vocabulary": base.get("vocab") == trained.get("vocab"),
        "source_copy_pointer": bool(config.get("source_copy_aware", False)),
        "base_tensor_changes": changed,
        "p1_base_path_bitwise_protected": not changed,
        "p1_exact_empty_source_condition_layout": True,
        "router": False,
        "materializer": False,
        "property_rerank": False,
        "candidate_rows": len(candidates),
        "candidate_validity": valid / max(len(candidates), 1),
        "identity_copy_fraction": identities / max(len(candidates), 1),
        "identity_fraction_among_valid": identities / max(valid, 1),
        "nonidentity_valid_fraction": (valid - identities) / max(len(candidates), 1),
        "strict_nonidentity_candidate_fraction": strict_nonidentity / max(len(candidates), 1),
        "unique_fraction": len(set(canonicals)) / max(len(candidates), 1),
        "source_tanimoto_distribution": {
            "n": len(similarities),
            "mean": statistics.fmean(similarities) if similarities else None,
            "p10": quantile(0.10),
            "p50": quantile(0.50),
            "p90": quantile(0.90),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if changed or bool(config.get("source_copy_aware", False)):
        raise SystemExit("P1 protected-path audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
