#!/usr/bin/env python3
"""Frozen B41 particle-coverage inference ablation. No retraining."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
graph = b41.graph
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


def iid_latent_particles(
    attempts: int,
    dimension: int,
    generator: torch.Generator,
    device: torch.device,
    scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    particles = (
        torch.randn(attempts, dimension, generator=generator, device=device, dtype=torch.float32)
        * float(scale)
    )
    normalized = F.normalize(particles, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, device=device, dtype=torch.bool)
    return particles, {
        "initial_particle_mean_abs_cosine": float(cosine[off_diagonal].abs().mean().detach().cpu()),
        "initial_particle_max_abs_cosine": float(cosine[off_diagonal].abs().max().detach().cpu()),
    }


@torch.no_grad()
def independent_transport_particles(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: torch.nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Condition-flow Euler only. No repulsion, no RMS spread floor."""
    attempts = particles.shape[0]
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], attempts, axis=0)).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        source_node, _ = representation.encode(source)
    latent = particles.float()
    steps = int(preregistration["flow_steps"])
    for flow_index in range(steps):
        velocities = []
        for start in range(0, attempts, chunk):
            count = min(chunk, attempts - start)
            flow_time = torch.full(
                (count,),
                (flow_index + 0.5) / steps,
                device=device,
                dtype=source_node.dtype,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                velocity = model.transport_velocity(
                    latent[start : start + count],
                    flow_time,
                    source_node.expand(count, -1, -1),
                    source["node_mask"].expand(count, -1),
                    tokens[start : start + count],
                )
            velocities.append(velocity.float())
        latent = latent + torch.cat(velocities, dim=0) / steps
    normalized = F.normalize(latent, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool, device=device)
    final_centered_rms = float(
        (latent - latent.mean(dim=0, keepdim=True)).norm(dim=1).square().mean().sqrt().detach().cpu()
    )
    return latent, {
        "final_particle_mean_abs_cosine": float(cosine[off_diagonal].abs().mean().detach().cpu()),
        "final_particle_max_abs_cosine": float(cosine[off_diagonal].abs().max().detach().cpu()),
        "final_particle_centered_rms": final_centered_rms,
        "minimum_transport_particle_rms": final_centered_rms,
    }


@contextmanager
def particle_sampling_mode(init_mode: str, interact_mode: str):
    original_init = b40.orthogonal_latent_particles
    original_transport = b41.interacting_transport_particles
    if init_mode == "iid":
        b40.orthogonal_latent_particles = iid_latent_particles
    if interact_mode == "independent":
        b41.interacting_transport_particles = independent_transport_particles
    try:
        yield
    finally:
        b40.orthogonal_latent_particles = original_init
        b41.interacting_transport_particles = original_transport


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


def variant_complete(path: Path, expected_rows: int) -> bool:
    if expected_rows <= 0 or not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        n = max(0, sum(1 for _ in handle) - 1)
    return n >= int(expected_rows)


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
        _b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_prereg)
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)

    selected_pairs = d0b.reconstruct_support_pairs(args, b41_prereg, b22_checkpoint)
    fit_pairs, _development_pairs, _split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = b41.delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(b41.unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    attempts = int(audit["exact_raw_attempts_per_condition"])
    expected_rows = len(conditions) * attempts
    exact_support = DeadEndSafeSupport(valid_terminal.ExactMoleculeStopSupport(vocabulary))
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
        for spec in audit["variants"]:
            name = str(spec["name"])
            variant_dir = args.output_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = variant_dir / f"b41_{name}_table1_n20_candidates.csv"
            if variant_complete(candidate_path, expected_rows):
                print(json.dumps({"stage": "skip_existing", "variant": name, "path": str(candidate_path)}, sort_keys=True), flush=True)
                continue
            rows: list[dict[str, object]] = []
            skipped = 0
            sample_started = time.perf_counter()
            with particle_sampling_mode(str(spec["init"]), str(spec["interact"])):
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
                            int(audit["seed"]) * 100000 + index,
                        )
                    except Exception as exc:
                        print(
                            json.dumps(
                                {
                                    "stage": "sample_failed",
                                    "variant": name,
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
                                "family": "b41_particle_coverage",
                                "op": "latent_graph_jump",
                                "variant": name,
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
                                    "variant": name,
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
            d0b.write_rows(candidate_path, rows)
            sampling = {
                "protocol": audit["protocol"],
                "variant": name,
                "init": spec["init"],
                "interact": spec["interact"],
                "device": str(device),
                "eval_csv": str(args.eval_csv),
                "loaded_conditions": len(conditions),
                "candidate_rows": len(rows),
                "attempts_per_condition": attempts,
                "skipped_count": skipped,
                "candidate_csv": str(candidate_path),
                "mean_unique_smiles": unique_smiles_mean(rows),
                "molecular_candidate_ranking": False,
                "task_router": False,
                "oracle_in_environment": False,
                "sample_sec": round(time.perf_counter() - sample_started, 1),
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "exact_stop_support": exact_support.manifest(),
            }
            (variant_dir / "sampling_summary.json").write_text(
                json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps({"stage": "variant_done", **sampling}, sort_keys=True), flush=True)
    finally:
        b41.viability_event_mask = original_support
    print(json.dumps({"stage": "done", "elapsed_sec": round(time.perf_counter() - started, 1)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
