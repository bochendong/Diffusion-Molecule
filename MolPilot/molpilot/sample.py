"""Sample MolPilot candidates from staged artifacts."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .alignment import UnderstandingAlignment
from .artifacts import ensure_dir, save_json, write_csv
from .autoencoder import load_autoencoder
from .diffusion import MolecularLatentDiffusion
from .stage_data import build_condition_table, load_smiles_and_pairs
from .verifier import verify_candidate


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    autoencoder = load_autoencoder(args.autoencoder_dir)
    alignment = UnderstandingAlignment.load(args.alignment_dir)
    diffusion = MolecularLatentDiffusion.load(args.diffusion_dir, codec=autoencoder)
    _, pairs = load_smiles_and_pairs(args.data, limit=args.limit)
    raw_conditions, _, bundles, request_rows = build_condition_table(
        pairs,
        condition_dim=args.condition_dim,
        render_missing_images=args.render_missing_images,
        render_dir=str(out_dir / "rendered_inputs"),
    )
    conditions = alignment.predict(raw_conditions)
    sampled = diffusion.sample_smiles(conditions, n_per_condition=args.samples_per_request, top_k=args.decode_top_k)
    rows = []
    overall = []
    hard = []
    for request_idx, (((request, target), bundle), candidates) in enumerate(zip(zip(pairs, bundles), sampled)):
        for rank, candidate in enumerate(candidates[: args.samples_per_request * args.decode_top_k]):
            result = verify_candidate(request.source_smiles, candidate, bundle.objective)
            rows.append(
                {
                    "request_id": request_idx,
                    "rank": rank,
                    "task_type": request.task_type.value,
                    "source_smiles": request.source_smiles or "",
                    "target_smiles": target,
                    "candidate_smiles": candidate,
                    "instruction": request.instruction,
                    "objective_json": json.dumps(bundle.objective.to_dict(), sort_keys=True),
                    "notes": "|".join(bundle.notes),
                    **result.to_dict(),
                }
            )
            overall.append(float(result.overall_success))
            if result.hard_verifiable:
                hard.append(float(result.overall_success))
    write_csv(rows, out_dir / "tables" / "candidates.csv")
    write_csv(request_rows, out_dir / "tables" / "requests.csv")
    metrics = {
        "stage": "stage4_sample_verify",
        "requests": float(len(pairs)),
        "candidates": float(len(rows)),
        "overall_success": float(np.mean(overall)) if overall else 0.0,
        "hard_verified_success": float(np.mean(hard)) if hard else 0.0,
    }
    save_json(metrics, out_dir / "metrics.json")
    print("Stage 4 sampling/evaluation complete")
    print(f"requests={len(pairs)} candidates={len(rows)} hard_verified_success={metrics['hard_verified_success']:.4f}")
    print(f"sample_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and verify MolPilot candidates.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--autoencoder-dir", default="outputs/stages/default/stage1_autoencoder")
    parser.add_argument("--alignment-dir", default="outputs/stages/default/stage2_understanding")
    parser.add_argument("--diffusion-dir", default="outputs/stages/default/stage3_diffusion")
    parser.add_argument("--output-dir", default="outputs/stages/default/stage4_samples")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--samples-per-request", type=int, default=8)
    parser.add_argument("--decode-top-k", type=int, default=4)
    parser.add_argument("--render-missing-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
