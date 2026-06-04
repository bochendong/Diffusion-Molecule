#!/usr/bin/env python
"""Export pooled and query-token condition features for baseline rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sketchmol_understanding_condition.condition_encoder import build_condition_encoder
from sketchmol_understanding_condition.retrieval_data import read_variant_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-variants-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--encoder", default="proxy")
    parser.add_argument("--variants", default="")
    parser.add_argument("--pooled-dim", type=int, default=768)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--query-dim", type=int, default=256)
    parser.add_argument("--image-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--hf-model-name-or-path", default=None)
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-dtype", default="auto")
    parser.add_argument("--hf-batch-size", type=int, default=1)
    parser.add_argument("--hf-max-length", type=int, default=2048)
    parser.add_argument("--hf-attn-implementation", default=None)
    parser.add_argument("--hf-prompt-style", default="auto")
    parser.add_argument("--hf-trust-remote-code", dest="hf_trust_remote_code", action="store_true")
    parser.add_argument("--hf-no-trust-remote-code", dest="hf_trust_remote_code", action="store_false")
    parser.set_defaults(hf_trust_remote_code=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_variant_rows(args.baseline_variants_csv)
    if args.variants:
        keep_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
        rows = [row for row in rows if row.get("variant") in keep_variants]
    if args.limit is not None:
        rows = rows[: args.limit]

    encoder_kwargs = {
        "pooled_dim": args.pooled_dim,
        "num_queries": args.num_queries,
        "query_dim": args.query_dim,
    }
    if args.image_encoder_checkpoint is not None:
        encoder_kwargs["image_encoder_checkpoint"] = str(args.image_encoder_checkpoint)
    if args.encoder == "hf_vlm":
        if not args.hf_model_name_or_path:
            raise ValueError("--hf-model-name-or-path is required for --encoder hf_vlm")
        encoder_kwargs.update(
            {
                "hf_model_name_or_path": args.hf_model_name_or_path,
                "hf_device_map": args.hf_device_map,
                "hf_dtype": args.hf_dtype,
                "hf_batch_size": args.hf_batch_size,
                "hf_max_length": args.hf_max_length,
                "hf_trust_remote_code": args.hf_trust_remote_code,
                "hf_attn_implementation": args.hf_attn_implementation,
                "hf_prompt_style": args.hf_prompt_style,
            }
        )
    encoder = build_condition_encoder(args.encoder, **encoder_kwargs)

    if hasattr(encoder, "encode_rows"):
        encoded_rows = encoder.encode_rows(rows)
    else:
        encoded_rows = [encoder.encode_row(row) for row in rows]

    pooled = []
    query_tokens = []
    index_rows = []
    for idx, (row, encoded) in enumerate(zip(rows, encoded_rows)):
        pooled.append(encoded.pooled)
        query_tokens.append(encoded.query_tokens)
        index_rows.append(
            {
                "row_index": idx,
                "variant_id": row.get("variant_id", ""),
                "pair_id": row.get("pair_id", ""),
                "split": row.get("split", ""),
                "variant": row.get("variant", ""),
                "condition_mode": encoded.condition_mode,
                "condition_id": row.get("condition_id", ""),
                "condition_properties": row.get("condition_properties", ""),
                "property_count": row.get("property_count", ""),
                "objective": row.get("objective", row.get("property_name", "")),
                "direction": row.get("direction", ""),
                "source_smiles": row.get("source_smiles", ""),
                "target_smiles": row.get("target_smiles", ""),
            }
        )

    pooled_arr = np.stack(pooled).astype(np.float32)
    query_arr = np.stack(query_tokens).astype(np.float32)
    np.save(args.output_dir / "pooled.npy", pooled_arr)
    np.save(args.output_dir / "query_tokens.npy", query_arr)
    _write_index(args.output_dir / "index.csv", index_rows)

    summary = {
        "baseline_variants_csv": str(args.baseline_variants_csv),
        "encoder": args.encoder,
        "variants": args.variants or "all",
        "image_encoder_checkpoint": str(args.image_encoder_checkpoint) if args.image_encoder_checkpoint else None,
        "hf_model_name_or_path": args.hf_model_name_or_path if args.encoder == "hf_vlm" else None,
        "hf_device_map": args.hf_device_map if args.encoder == "hf_vlm" else None,
        "hf_dtype": args.hf_dtype if args.encoder == "hf_vlm" else None,
        "hf_batch_size": args.hf_batch_size if args.encoder == "hf_vlm" else None,
        "hf_prompt_style": args.hf_prompt_style if args.encoder == "hf_vlm" else None,
        "rows": len(rows),
        "pooled_shape": list(pooled_arr.shape),
        "query_tokens_shape": list(query_arr.shape),
        "index_csv": str(args.output_dir / "index.csv"),
        "pooled_npy": str(args.output_dir / "pooled.npy"),
        "query_tokens_npy": str(args.output_dir / "query_tokens.npy"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
