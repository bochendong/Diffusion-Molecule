"""Train and evaluate the first MolPilot scaffold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import build_smoke_requests, load_smiles_csv
from .diffusion import DiffusionConfig, MolecularLatentDiffusion
from .schema import GenerationRequest, TaskType
from .understanding import UnderstandingConfig, UnderstandingStream
from .verifier import verify_candidate


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.run_name
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    smiles = load_smiles_csv(args.data, limit=args.limit)
    pairs = build_smoke_requests(smiles)

    stream = UnderstandingStream(
        UnderstandingConfig(
            condition_dim=args.condition_dim,
            render_missing_images=not args.disable_render_missing_images,
            render_dir=str(out_dir / "rendered_inputs"),
        )
    )
    bundles = [stream.encode(req) for req, _ in pairs]
    conditions = np.asarray([bundle.branches["multimodal"].vector for bundle in bundles], dtype=np.float32)
    target_smiles = [target for _, target in pairs]

    model = MolecularLatentDiffusion(
        DiffusionConfig(
            latent_dim=args.latent_dim,
            condition_dim=conditions.shape[1],
            epochs=args.epochs,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
    )
    model.fit(target_smiles, conditions)
    model.save_history(out_dir / "models" / "training_history.json")
    generated = model.sample_smiles(conditions, n_per_condition=args.samples_per_request, top_k=args.decode_top_k)

    rows = []
    overall = []
    hard_rows = []
    for idx, ((request, target), bundle, candidates) in enumerate(zip(pairs, bundles, generated)):
        for rank, candidate in enumerate(candidates[: args.decode_top_k * args.samples_per_request]):
            result = verify_candidate(request.source_smiles, candidate, bundle.objective)
            row = {
                "request_id": idx,
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
            rows.append(row)
            overall.append(float(result.overall_success))
            if result.hard_verifiable:
                hard_rows.append(float(result.overall_success))
    write_csv(out_dir / "tables" / "candidates.csv", rows)
    metrics = {
        "requests": float(len(pairs)),
        "candidates": float(len(rows)),
        "overall_success": float(np.mean(overall)) if overall else 0.0,
        "hard_verified_success": float(np.mean(hard_rows)) if hard_rows else 0.0,
        "torch_backend": str(model.history[-1].get("backend", "unknown") if model.history else "unknown"),
        "latent_dim": float(args.latent_dim),
        "condition_dim": float(args.condition_dim),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.txt").write_text(_summary(metrics), encoding="utf-8")
    print(_summary(metrics))
    print(f"run_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolPilot multimodal molecular generation scaffold.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--samples-per-request", type=int, default=2)
    parser.add_argument("--decode-top-k", type=int, default=2)
    parser.add_argument("--disable-render-missing-images", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            "MolPilot experiment complete",
            f"requests={metrics['requests']:.0f}",
            f"candidates={metrics['candidates']:.0f}",
            f"overall_success={metrics['overall_success']:.4f}",
            f"hard_verified_success={metrics['hard_verified_success']:.4f}",
            f"backend={metrics['torch_backend']}",
        ]
    )


if __name__ == "__main__":
    main()
