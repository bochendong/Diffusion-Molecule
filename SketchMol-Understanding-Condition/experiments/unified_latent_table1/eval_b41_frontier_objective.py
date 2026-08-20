#!/usr/bin/env python3
"""Frontier vs singleton next-event labels. Same B39 warm start, 2-epoch fine-tune."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
WORKTREE_LATENT = PROJECT_DIR / "experiments" / "unified_latent_flow"
WORKTREE_PROJECT = PROJECT_DIR
C_DIR = PROJECT_DIR / "experiments" / "unified_action_categorical"
for path in (WORKTREE_LATENT, WORKTREE_PROJECT, C_DIR, PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_d0_b41_table1 as d0b  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402
from dead_end_safe_support import DeadEndSafeSupport  # noqa: E402

base = b41.base
delta = b41.delta
graph = b41.graph
hierarchical = b41.hierarchical
unified = b41.unified
b37 = b41.b37
b39 = b41.b39
b40 = b41.b40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--b22-checkpoint", required=True, type=Path)
    parser.add_argument("--b22-summary", required=True, type=Path)
    parser.add_argument("--b36-summary", required=True, type=Path)
    parser.add_argument("--b37-summary", required=True, type=Path)
    parser.add_argument("--b38-checkpoint", required=True, type=Path)
    parser.add_argument("--b38-summary", required=True, type=Path)
    parser.add_argument("--b39-checkpoint", required=True, type=Path)
    parser.add_argument("--b39-summary", required=True, type=Path)
    parser.add_argument("--b39-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b40-summary", required=True, type=Path)
    parser.add_argument("--b40-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-checkpoint", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--b41-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-protocol-manifest", required=True, type=Path)
    parser.add_argument("--audit-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def collapse_target_next(target_next: torch.Tensor, label_mode: str, seed: int) -> torch.Tensor:
    if label_mode == "frontier" or bool(target_next[0]):
        return target_next
    indices = torch.nonzero(target_next, as_tuple=False).flatten().tolist()
    if not indices:
        return target_next
    if label_mode == "canonical":
        pick = min(indices)
    elif label_mode == "random_singleton":
        pick = random.Random(int(seed) + 17).choice(indices)
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")
    collapsed = torch.zeros_like(target_next)
    collapsed[pick] = True
    return collapsed


def install_label_mode(label_mode: str):
    original = b41.b38.random_topological_prefix

    def wrapped(events, layout, *, seed, completion_probability):
        node, edge, target_next, length = original(
            events, layout, seed=seed, completion_probability=completion_probability
        )
        return node, edge, collapse_target_next(target_next, label_mode, int(seed)), length

    b41.b38.random_topological_prefix = wrapped
    return original


def restore_prefix(original) -> None:
    b41.b38.random_topological_prefix = original


def variant_complete(path: Path, expected_rows: int) -> bool:
    if expected_rows <= 0 or not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        n = max(0, sum(1 for _ in handle) - 1)
    return n >= int(expected_rows)


def unique_smiles_mean(rows: list[dict[str, object]]) -> float:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition_id", "")), set()).add(
            str(row.get("generated_smiles", "") or "").strip()
        )
    sizes = [len(values - {""}) for values in grouped.values()]
    if not sizes:
        return 0.0
    return float(sum(sizes) / len(sizes))


def build_model(representation_config, b41_prereg, vocabulary, device):
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    return model


def sample_table1(
    *,
    model,
    representation,
    vocabulary,
    support,
    support_tensors,
    conditions,
    b41_prereg,
    audit,
    device,
    exact_support,
    variant,
    candidate_path,
):
    attempts = int(audit["exact_raw_attempts_per_condition"])
    safe_support = DeadEndSafeSupport(exact_support)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = safe_support
    rows: list[dict[str, object]] = []
    skipped = 0
    sample_started = time.perf_counter()
    try:
        for index, condition in enumerate(conditions):
            try:
                generated = b41.sample_from_source(
                    model,
                    representation,
                    vocabulary,
                    support,
                    support_tensors,
                    condition.source,
                    np.asarray(condition.condition),
                    b41_prereg,
                    device,
                    int(audit["eval_seed"]) * 100000 + index,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "sample_failed",
                            "variant": variant,
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                generated = [{"generated_smiles": ""}] * attempts
                skipped += 1
            if len(generated) != attempts:
                generated = (list(generated) + [{"generated_smiles": ""}] * attempts)[:attempts]
                skipped += 1
            for attempt, candidate in enumerate(generated, start=1):
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "task": condition.task,
                        "source_smiles": condition.source_smiles,
                        "generated_smiles": candidate.get("generated_smiles", "") or "",
                        "sample_index": attempt,
                        "candidate_index": attempt,
                        "method": audit["protocol"],
                        "family": "b41_frontier_objective",
                        "op": "latent_graph_jump",
                        "variant": variant,
                    }
                )
            if (index + 1) % 20 == 0 or index + 1 == len(conditions):
                elapsed = time.perf_counter() - sample_started
                done = index + 1
                sec_per = elapsed / done
                print(
                    json.dumps(
                        {
                            "stage": "sampled",
                            "variant": variant,
                            "done": done,
                            "total": len(conditions),
                            "elapsed_sec": round(elapsed, 1),
                            "sec_per_condition": round(sec_per, 3),
                            "eta_sec": round(sec_per * (len(conditions) - done), 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.viability_event_mask = original_support
    d0b.write_rows(candidate_path, rows)
    return rows, skipped, time.perf_counter() - sample_started, safe_support.manifest()


def main() -> int:
    args = parse_args()
    audit = json.loads(args.audit_protocol_manifest.read_text(encoding="utf-8"))
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    (
        _b22_summary,
        b22_checkpoint,
        _b36_summary,
        _b37_summary,
        b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_prereg)

    selected_pairs = d0b.reconstruct_support_pairs(args, b41_prereg, b22_checkpoint)
    fit_pairs, _development_pairs, _split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    for pair in fit_pairs:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(b41_prereg["condition_dim"])
        )
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    model = build_model(representation_config, b41_prereg, vocabulary, device)
    b39_state = dict(b39_checkpoint["model_state"])

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    attempts = int(audit["exact_raw_attempts_per_condition"])
    expected_rows = len(conditions) * attempts
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    train_prereg = dict(b41_prereg)
    train_prereg["epochs"] = int(audit["epochs"])
    train_prereg["seed"] = int(audit["seed"])

    print(
        json.dumps(
            {
                "stage": "ready",
                "fit_pairs": len(fit_pairs),
                "conditions": len(conditions),
                "label_modes": list(audit["label_modes"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for label_mode in audit["label_modes"]:
        variant_dir = args.output_dir / str(label_mode)
        variant_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = variant_dir / f"b41_{label_mode}_table1_n20_candidates.csv"
        checkpoint_path = variant_dir / f"b41_{label_mode}_event_kernel.pt"
        if variant_complete(candidate_path, expected_rows):
            print(
                json.dumps({"stage": "skip_existing", "variant": label_mode, "path": str(candidate_path)}, sort_keys=True),
                flush=True,
            )
            continue
        model.load_state_dict(b39_state, strict=True)
        history = []
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(dict(payload["model_state"]), strict=True)
            history = list(payload.get("history") or [])
            print(
                json.dumps({"stage": "resume_checkpoint", "variant": label_mode, "path": str(checkpoint_path)}, sort_keys=True),
                flush=True,
            )
        else:
            original_prefix = install_label_mode(str(label_mode))
            try:
                print(
                    json.dumps(
                        {
                            "stage": "train_start",
                            "variant": label_mode,
                            "pairs": len(fit_pairs),
                            "epochs": train_prereg["epochs"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                train_started = time.perf_counter()
                history = b41.fine_tune_event_kernel(
                    model,
                    representation,
                    fit_pairs,
                    vocabulary,
                    support,
                    support_tensors,
                    train_prereg,
                    device,
                )
                print(
                    json.dumps(
                        {
                            "stage": "train_done",
                            "variant": label_mode,
                            "elapsed_sec": round(time.perf_counter() - train_started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                restore_prefix(original_prefix)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "protocol": audit["protocol"],
                    "label_mode": label_mode,
                    "history": history,
                    "fit_pairs": len(fit_pairs),
                },
                checkpoint_path,
            )
        model.eval().requires_grad_(False)
        rows, skipped, sample_sec, support_manifest = sample_table1(
            model=model,
            representation=representation,
            vocabulary=vocabulary,
            support=support,
            support_tensors=support_tensors,
            conditions=conditions,
            b41_prereg=b41_prereg,
            audit=audit,
            device=device,
            exact_support=exact_support,
            variant=str(label_mode),
            candidate_path=candidate_path,
        )
        sampling = {
            "protocol": audit["protocol"],
            "variant": label_mode,
            "device": str(device),
            "eval_csv": str(args.eval_csv),
            "loaded_conditions": len(conditions),
            "candidate_rows": len(rows),
            "attempts_per_condition": attempts,
            "skipped_count": skipped,
            "candidate_csv": str(candidate_path),
            "checkpoint": str(checkpoint_path),
            "mean_unique_smiles": unique_smiles_mean(rows),
            "molecular_candidate_ranking": False,
            "task_router": False,
            "oracle_in_environment": False,
            "sample_sec": round(sample_sec, 1),
            "elapsed_sec": round(time.perf_counter() - started, 1),
            "history": history,
            "exact_stop_support": support_manifest,
        }
        (variant_dir / "sampling_summary.json").write_text(
            json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"stage": "variant_done", "variant": label_mode, "sample_sec": sampling["sample_sec"]}, sort_keys=True), flush=True)

    print(json.dumps({"stage": "done", "elapsed_sec": round(time.perf_counter() - started, 1)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
