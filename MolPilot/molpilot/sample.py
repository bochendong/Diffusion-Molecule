"""Sample MolPilot candidates from staged artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json

import numpy as np

from .artifacts import ensure_dir, save_json, write_csv
from .autoencoder import load_autoencoder
from .condition_model import load_condition_model, predict_condition_latents
from .diffusion import MolecularLatentDiffusion
from .repair_verifier import verify_repair
from .schema import TaskType
from .stage_data import build_condition_table, load_smiles_and_pairs
from .source_guidance import Candidate, decode_source_guided_candidates, parse_strengths
from .verifier import verify_candidate


def main() -> None:
    args = parse_args()
    if args.task_mode == "repair" and args.tasks == "edit,inpaint,de_novo":
        args.tasks = "repair"
    elif args.task_mode == "mixed" and args.tasks == "edit,inpaint,de_novo":
        args.tasks = "edit,inpaint,de_novo,repair"
    out_dir = ensure_dir(args.output_dir)
    autoencoder = load_autoencoder(args.autoencoder_dir)
    alignment = load_condition_model(args.alignment_dir)
    diffusion = MolecularLatentDiffusion.load(args.diffusion_dir, codec=autoencoder)
    _, pairs = load_smiles_and_pairs(
        args.data,
        limit=args.limit,
        task_mode=args.task_mode,
        repair_corruption_types=args.repair_corruptions,
        repair_corruptions_per_molecule=args.repair_corruptions_per_molecule,
        seed=args.seed,
    )
    pairs = _select_pairs_by_task(
        pairs,
        max_per_task=args.max_requests_per_task,
        tasks=args.tasks,
        seed=args.seed,
    )
    if not pairs:
        write_csv([], out_dir / "tables" / "candidates.csv")
        write_csv([], out_dir / "tables" / "requests.csv")
        write_csv([], out_dir / "tables" / "request_metrics.csv")
        metrics = {
            "stage": "stage4_sample_verify",
            "requests": 0.0,
            "candidates": 0.0,
            "task_mode": args.task_mode,
            "overall_success": 0.0,
            "hard_verified_success": 0.0,
        }
        save_json(metrics, out_dir / "metrics.json")
        print("Stage 4 sampling/evaluation complete")
        print("requests=0 candidates=0 hard_verified_success=0.0000")
        print(f"sample_dir={out_dir}")
        return
    raw_conditions, _, bundles, request_rows = build_condition_table(
        pairs,
        condition_dim=args.condition_dim,
        render_missing_images=args.render_missing_images,
        render_dir=str(out_dir / "rendered_inputs"),
    )
    conditions = predict_condition_latents(alignment, raw_conditions, pairs, autoencoder)
    sampled_latents = diffusion.sample_latents(conditions, n_per_condition=args.samples_per_request)
    source_edit_strengths = parse_strengths(args.source_edit_strengths)
    known_smiles = set(getattr(autoencoder, "train_smiles", []) or [])
    rows = []
    request_metric_rows = []
    overall = []
    hard = []
    failure_counts: Counter[str] = Counter()
    latent_cursor = 0
    for request_idx, ((request, target), bundle) in enumerate(zip(pairs, bundles)):
        request_latents = sampled_latents[latent_cursor : latent_cursor + args.samples_per_request]
        latent_cursor += args.samples_per_request
        direct_condition_candidates = _condition_latent_candidates(
            autoencoder,
            conditions[request_idx],
            top_k=args.condition_decode_top_k,
        )
        candidates = decode_source_guided_candidates(
            autoencoder,
            request,
            request_latents,
            objective=bundle.objective,
            top_k=args.decode_top_k,
            source_edit_strengths=source_edit_strengths,
            source_neighborhood_k=args.source_neighborhood_k,
            graph_edit_limit=args.graph_edit_limit,
            scaffold_library_k=args.scaffold_library_k,
            enable_source_guidance=not args.disable_source_guidance,
            include_diffusion_candidates=not args.disable_diffusion_candidates,
            enable_latent_source_guidance=not args.disable_latent_source_guidance,
            enable_graph_editor=not args.disable_graph_editor,
        )
        candidates = _dedupe_candidates(direct_condition_candidates + candidates)
        if request.task_type == TaskType.REPAIR and not args.disable_repair_baselines:
            candidates = _dedupe_candidates(
                _repair_baseline_candidates(autoencoder, request.source_smiles, top_k=args.repair_nearest_k)
                + candidates
            )
        scored = []
        for raw_rank, candidate in enumerate(candidates):
            if request.task_type == TaskType.REPAIR:
                result = verify_repair(
                    request.source_smiles,
                    target,
                    candidate.smiles,
                    known_smiles=known_smiles,
                )
            else:
                result = verify_candidate(request.source_smiles, candidate.smiles, bundle.objective)
            for reason in result.reasons:
                failure_counts[reason] += 1
            scored.append((raw_rank, candidate, result, _ranking_score(result)))
        if not args.disable_verifier_ranking:
            scored.sort(key=lambda item: (-item[3], item[0]))

        request_overall = []
        request_goal = []
        request_constraint = []
        request_exact_recovery = []
        request_scaffold_recovery = []
        request_novel_repair = []
        for rank, (raw_rank, candidate, result, score) in enumerate(scored):
            rows.append(
                {
                    "request_id": request_idx,
                    "rank": rank,
                    "raw_rank": raw_rank,
                    "ranking_score": f"{score:.6f}",
                    "candidate_origin": candidate.origin,
                    "task_type": request.task_type.value,
                    "source_smiles": request.source_smiles or "",
                    "target_smiles": target,
                    "candidate_smiles": candidate.smiles,
                    "instruction": request.instruction,
                    "objective_json": json.dumps(bundle.objective.to_dict(), sort_keys=True),
                    "notes": "|".join(bundle.notes),
                    **result.to_dict(),
                }
            )
            overall.append(float(result.overall_success))
            request_overall.append(float(result.overall_success))
            request_goal.append(float(result.goal_success))
            request_constraint.append(float(result.constraint_success))
            request_exact_recovery.append(float(getattr(result, "exact_recovery", False)))
            request_scaffold_recovery.append(float(getattr(result, "scaffold_recovery", False)))
            request_novel_repair.append(float(getattr(result, "novel_verified_success", False)))
            if result.hard_verifiable:
                hard.append(float(result.overall_success))
        request_metrics = {
            "request_id": request_idx,
            "task_type": request.task_type.value,
            "source_smiles": request.source_smiles or "",
            "target_smiles": target,
            "instruction": request.instruction,
            "n_candidates": len(scored),
            **_topk_metrics(request_overall, "overall"),
            **_topk_metrics(request_goal, "goal"),
            **_topk_metrics(request_constraint, "constraint"),
        }
        if request.task_type == TaskType.REPAIR:
            request_metrics.update(
                {
                    **_topk_metrics(request_exact_recovery, "exact_recovery"),
                    **_topk_metrics(request_scaffold_recovery, "scaffold_recovery"),
                    **_topk_metrics(request_novel_repair, "novel_repair_success"),
                }
            )
        request_metric_rows.append(request_metrics)
    write_csv(rows, out_dir / "tables" / "candidates.csv")
    write_csv(request_rows, out_dir / "tables" / "requests.csv")
    write_csv(request_metric_rows, out_dir / "tables" / "request_metrics.csv")
    if any(row.get("task_type") == TaskType.REPAIR.value for row in request_metric_rows):
        write_csv(request_metric_rows, out_dir / "tables" / "repair_request_metrics.csv")
        write_csv(_novel_verified_summary(request_metric_rows), out_dir / "tables" / "novel_verified_summary.csv")
    write_csv(
        [
            {"reason": reason, "count": count, "fraction": count / max(1, len(rows))}
            for reason, count in failure_counts.most_common()
        ],
        out_dir / "tables" / "failure_reasons.csv",
    )
    metrics = {
        "stage": "stage4_sample_verify",
        "requests": float(len(pairs)),
        "candidates": float(len(rows)),
        "verifier_ranking": not args.disable_verifier_ranking,
        "source_guidance": not args.disable_source_guidance,
        "diffusion_candidates": not args.disable_diffusion_candidates,
        "latent_source_guidance": not args.disable_latent_source_guidance,
        "graph_editor": not args.disable_graph_editor,
        "source_edit_strengths": args.source_edit_strengths,
        "source_neighborhood_k": float(args.source_neighborhood_k),
        "graph_edit_limit": float(args.graph_edit_limit),
        "scaffold_library_k": float(args.scaffold_library_k),
        "condition_decode_top_k": float(args.condition_decode_top_k),
        "max_requests_per_task": float(args.max_requests_per_task),
        "task_mode": args.task_mode,
        "repair_corruptions_per_molecule": float(args.repair_corruptions_per_molecule),
        "overall_success": float(np.mean(overall)) if overall else 0.0,
        "hard_verified_success": float(np.mean(hard)) if hard else 0.0,
        **_aggregate_request_metrics(request_metric_rows),
        **{f"failure_reason_{reason}": float(count) for reason, count in failure_counts.most_common(20)},
    }
    save_json(metrics, out_dir / "metrics.json")
    if any(row.get("task_type") == TaskType.REPAIR.value for row in request_metric_rows):
        save_json(metrics, out_dir / "repair_metrics.json")
    print("Stage 4 sampling/evaluation complete")
    print(f"requests={len(pairs)} candidates={len(rows)} hard_verified_success={metrics['hard_verified_success']:.4f}")
    print(
        "request_topk "
        f"overall@1={metrics.get('request_overall_at_1', 0.0):.4f} "
        f"overall@5={metrics.get('request_overall_at_5', 0.0):.4f} "
        f"overall@10={metrics.get('request_overall_at_10', 0.0):.4f}"
    )
    print(f"sample_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and verify MolPilot candidates.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--autoencoder-dir", default="outputs/stages/default/stage1_autoencoder")
    parser.add_argument("--alignment-dir", default="outputs/stages/default/stage2_understanding")
    parser.add_argument("--diffusion-dir", default="outputs/stages/default/stage3_diffusion")
    parser.add_argument("--output-dir", default="outputs/stages/default/stage4_samples")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-mode", choices=["verified", "repair", "mixed"], default="verified")
    parser.add_argument("--repair-corruptions", default=None)
    parser.add_argument("--repair-corruptions-per-molecule", type=int, default=2)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--samples-per-request", type=int, default=8)
    parser.add_argument("--decode-top-k", type=int, default=4)
    parser.add_argument("--render-missing-images", action="store_true")
    parser.add_argument("--disable-verifier-ranking", action="store_true")
    parser.add_argument("--disable-source-guidance", action="store_true")
    parser.add_argument("--disable-diffusion-candidates", action="store_true")
    parser.add_argument("--disable-latent-source-guidance", action="store_true")
    parser.add_argument("--disable-graph-editor", action="store_true")
    parser.add_argument("--disable-repair-baselines", action="store_true")
    parser.add_argument("--source-edit-strengths", default="0.25,0.50")
    parser.add_argument("--source-neighborhood-k", type=int, default=32)
    parser.add_argument("--graph-edit-limit", type=int, default=96)
    parser.add_argument("--scaffold-library-k", type=int, default=32)
    parser.add_argument("--condition-decode-top-k", type=int, default=4)
    parser.add_argument("--repair-nearest-k", type=int, default=8)
    parser.add_argument("--max-requests-per-task", type=int, default=0)
    parser.add_argument("--tasks", default="edit,inpaint,de_novo")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _condition_latent_candidates(autoencoder, condition_latent, top_k: int = 4) -> list[Candidate]:
    if top_k <= 0:
        return []
    try:
        return [Candidate(smiles, "condition_direct") for smiles in autoencoder.decode(condition_latent, top_k=top_k)]
    except Exception:
        return []


def _repair_baseline_candidates(autoencoder, corrupted_smiles: str | None, top_k: int = 8) -> list[Candidate]:
    out: list[Candidate] = []
    if corrupted_smiles:
        out.append(Candidate(str(corrupted_smiles), "no_repair"))
    if corrupted_smiles and hasattr(autoencoder, "encode"):
        try:
            latent = autoencoder.encode(corrupted_smiles)
            for smiles in autoencoder.decode(latent, top_k=max(1, top_k)):
                out.append(Candidate(smiles, "nearest_valid_retrieval"))
        except Exception:
            pass
    return _dedupe_candidates(out)


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: dict[str, int] = {}
    for candidate in candidates:
        if candidate.smiles in seen:
            existing = out[seen[candidate.smiles]]
            origins = existing.origin.split("+")
            if candidate.origin not in origins:
                existing.origin = existing.origin + "+" + candidate.origin
            continue
        seen[candidate.smiles] = len(out)
        out.append(candidate)
    return out


def _select_pairs_by_task(pairs, max_per_task: int = 0, tasks: str = "edit,inpaint,de_novo", seed: int = 7):
    allowed = [task.strip() for task in str(tasks).split(",") if task.strip()]
    allowed_set = set(allowed)
    grouped: dict[str, list] = {task: [] for task in allowed}
    for pair in pairs:
        task = pair[0].task_type.value
        if task in allowed_set:
            grouped.setdefault(task, []).append(pair)
    rng = np.random.default_rng(seed)
    selected = []
    for task in allowed:
        rows = list(grouped.get(task, []))
        if max_per_task > 0 and len(rows) > max_per_task:
            chosen = rng.choice(len(rows), size=max_per_task, replace=False)
            rows = [rows[int(idx)] for idx in sorted(chosen)]
        selected.extend(rows)
    return selected


def _ranking_score(result) -> float:
    score = 0.0
    score += 1000.0 if result.overall_success else 0.0
    score += 100.0 if result.constraint_success else 0.0
    score += 20.0 if result.goal_success else 0.0
    score += 5.0 if result.valid else 0.0
    penalties = {
        "scaffold_changed": 12.0,
        "low_similarity": 8.0,
        "mw_drift": 6.0,
        "druglike_failed": 4.0,
        "cns_profile_failed": 4.0,
        "invalid_smiles": 100.0,
        "scaffold_not_recovered": 8.0,
        "low_similarity_to_clean": 8.0,
        "property_drift_from_clean": 4.0,
    }
    for reason in result.reasons:
        score -= penalties.get(reason, 1.0)
    return score


def _topk_metrics(values: list[float], prefix: str) -> dict[str, float]:
    out = {}
    for k in (1, 5, 10):
        out[f"{prefix}_at_{k}"] = float(max(values[:k])) if values else 0.0
    return out


def _aggregate_request_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    out = {}
    for prefix in ("overall", "goal", "constraint"):
        for k in (1, 5, 10):
            key = f"{prefix}_at_{k}"
            values = [float(row.get(key, 0.0)) for row in rows]
            out[f"request_{key}"] = float(np.mean(values)) if values else 0.0
    tasks = sorted({str(row.get("task_type", "unknown")) for row in rows})
    for task in tasks:
        task_rows = [row for row in rows if str(row.get("task_type", "unknown")) == task]
        task_key = _safe_metric_key(task)
        out[f"task_{task_key}_requests"] = float(len(task_rows))
        for prefix in ("overall", "goal", "constraint"):
            for k in (1, 5, 10):
                key = f"{prefix}_at_{k}"
                values = [float(row.get(key, 0.0)) for row in task_rows]
                out[f"task_{task_key}_request_{key}"] = float(np.mean(values)) if values else 0.0
    for prefix in ("overall", "goal", "constraint"):
        for k in (1, 5, 10):
            values = []
            for task in tasks:
                task_key = _safe_metric_key(task)
                metric_key = f"task_{task_key}_request_{prefix}_at_{k}"
                if metric_key in out:
                    values.append(out[metric_key])
            out[f"macro_task_request_{prefix}_at_{k}"] = float(np.mean(values)) if values else 0.0
    return out


def _novel_verified_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        task = str(row.get("task_type", "unknown"))
        grouped.setdefault(task, []).append(row)
    out = []
    metric_keys = [f"{prefix}_at_{k}" for prefix in ("overall", "novel_repair_success", "exact_recovery", "scaffold_recovery") for k in (1, 5, 10)]
    for task, task_rows in sorted(grouped.items()):
        summary = {"task_type": task, "requests": len(task_rows)}
        for key in metric_keys:
            summary[key] = float(np.mean([float(row.get(key, 0.0)) for row in task_rows])) if any(key in row for row in task_rows) else 0.0
        out.append(summary)
    return out


def _safe_metric_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_") or "unknown"


if __name__ == "__main__":
    main()
