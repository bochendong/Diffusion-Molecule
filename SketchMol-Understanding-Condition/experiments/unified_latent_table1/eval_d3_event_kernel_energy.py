#!/usr/bin/env python3
"""D3: train Table1 energy into the B41 event kernel.

Not D1: no MCS, no frozen B31 fragment goal. The same jump process is fine-tuned
on in-ball edit pairs (including GSK3B) then GRPO-scored at STOP. Inference stays
n=20, one decode, no ranking, no oracle.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
C_DIR = PROJECT_DIR / "experiments" / "unified_action_categorical"
UNIFIED_SCRIPTS = REPO_DIR / "SketchMol-Unified-3MDiffusion" / "scripts"
for path in (LATENT_DIR, C_DIR, PROJECT_DIR, UNIFIED_SCRIPTS, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_d0_b41_table1 as d0b  # noqa: E402
import table1_energy_tilted_latent_transfer as b29  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402
from evaluate_moledit_table_metrics import Chemistry, evaluate_prediction, task_specs_for_reference  # noqa: E402

b40 = b41.b40
b39 = b41.b39
b38 = b41.b38
b37 = b41.b37
base = b41.base
delta = b41.delta
graph = b41.graph
full_graph = b41.full_graph
hierarchical = b41.hierarchical
unified = b41.unified

REAL_TASKS = {
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "MW:increase",
    "RB:decrease",
    "SA:decrease",
}


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
    parser.add_argument("--d3-protocol-manifest", required=True, type=Path)
    parser.add_argument(
        "--frozen-model-checkpoint",
        type=Path,
        default=None,
        help="Load a completed D3 checkpoint and skip supervised/GRPO updates while generating a new train-only teacher pool.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d3 = json.loads(args.d3_protocol_manifest.read_text(encoding="utf-8"))
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
    eval_sources = {
        graph.canonical_smiles(row.get("source_smiles", ""))
        for row in base.read_rows(args.eval_csv)
    }
    eval_sources.discard("")

    print(json.dumps({"stage": "build_energy_pairs"}, sort_keys=True), flush=True)
    pair_started = time.perf_counter()
    energy_pairs = build_inball_pairs(
        args.train_csv,
        b41_prereg,
        representation_config,
        eval_sources,
        min_tanimoto=float(d3["min_source_tanimoto"]),
        allowed_counts={int(value) for value in d3["property_counts"]},
        limit=int(d3["train_pair_limit"]),
        seed=int(d3["seed"]),
        gsk3b_upsample=int(d3["gsk3b_upsample"]),
    )
    print(
        json.dumps(
            {
                "stage": "pairs",
                "original_fit": len(fit_pairs),
                "energy_pairs": len(energy_pairs),
                "elapsed_sec": round(time.perf_counter() - pair_started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(energy_pairs) < 32:
        raise ValueError(f"Need at least 32 in-ball pairs, found {len(energy_pairs)}")

    fit_pairs = keep_vocabulary_pairs(fit_pairs, vocabulary, "original_fit")
    energy_pairs = keep_vocabulary_pairs(energy_pairs, vocabulary, "energy")
    if len(energy_pairs) < 32:
        raise ValueError(f"Need at least 32 in-vocabulary energy pairs, found {len(energy_pairs)}")
    if len(fit_pairs) < 32:
        raise ValueError(f"Need at least 32 in-vocabulary original fit pairs, found {len(fit_pairs)}")

    support = b40.build_support(list(fit_pairs) + list(energy_pairs), vocabulary)
    support_tensors = b40._device_support(support, device)
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
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)

    supervised_prereg = dict(b41_prereg)
    supervised_prereg["epochs"] = int(d3["supervised_epochs"])
    supervised_prereg["batch_size"] = int(d3["supervised_batch_size"])
    supervised_prereg["seed"] = int(d3["seed"])
    train_pairs = list(fit_pairs) + list(energy_pairs)
    supervised_path = args.frozen_model_checkpoint or (args.output_dir / "d3_event_kernel_energy_supervised.pt")
    frozen_payload = None
    if args.frozen_model_checkpoint is not None:
        frozen_payload = torch.load(args.frozen_model_checkpoint, map_location="cpu", weights_only=False)
    if frozen_payload is not None or (bool(d3.get("resume_supervised")) and supervised_path.exists()):
        payload = frozen_payload or torch.load(supervised_path, map_location="cpu", weights_only=False)
        model.load_state_dict(dict(payload["model_state"]), strict=True)
        supervised_history = list(payload.get("supervised_history") or [])
        print(
            json.dumps(
                {
                    "stage": "frozen_teacher_resume" if frozen_payload is not None else "supervised_resume",
                    "pairs": len(train_pairs),
                    "path": str(supervised_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {"stage": "supervised_start", "pairs": len(train_pairs), "epochs": supervised_prereg["epochs"]},
                sort_keys=True,
            ),
            flush=True,
        )
        supervised_started = time.perf_counter()
        supervised_history = b41.fine_tune_event_kernel(
            model, representation, train_pairs, vocabulary, support, support_tensors, supervised_prereg, device
        )
        print(
            json.dumps(
                {"stage": "supervised_done", "elapsed_sec": round(time.perf_counter() - supervised_started, 1)},
                sort_keys=True,
            ),
            flush=True,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": b41_checkpoint.get("model_config"),
                "protocol": d3["protocol"],
                "supervised_history": supervised_history,
            },
            supervised_path,
        )

    skip_grpo = (
        args.frozen_model_checkpoint is not None
        or bool(d3.get("skip_grpo"))
        or int(d3.get("grpo_epochs", 0)) <= 0
    )
    grpo_conditions: list = []
    grpo_history: list[dict[str, float]] = list((frozen_payload or {}).get("grpo_history") or [])
    if not skip_grpo:
        chem = Chemistry()
        grpo_conditions = select_grpo_conditions(
            args.train_csv,
            eval_sources,
            condition_dim=int(b41_prereg["condition_dim"]),
            fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
            max_atoms=int(representation_config["max_atoms"]),
            limit=int(d3["grpo_conditions"]),
            seed=int(d3["seed"]),
        )
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
        if skip_grpo:
            print(
                json.dumps(
                    {
                        "stage": "grpo_skipped",
                        "reason": (
                            "frozen teacher checkpoint"
                            if args.frozen_model_checkpoint is not None
                            else "protocol disables GRPO"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            print(
                json.dumps({"stage": "grpo_start", "conditions": len(grpo_conditions)}, sort_keys=True),
                flush=True,
            )
            grpo_started = time.perf_counter()
            grpo_history = run_grpo(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                grpo_conditions,
                b41_prereg,
                d3,
                chem,
                device,
            )
            print(
                json.dumps(
                    {"stage": "grpo_done", "elapsed_sec": round(time.perf_counter() - grpo_started, 1)},
                    sort_keys=True,
                ),
                flush=True,
            )

        checkpoint_path = args.frozen_model_checkpoint or (args.output_dir / "d3_event_kernel_energy.pt")
        if args.frozen_model_checkpoint is None:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": b41_checkpoint.get("model_config"),
                    "protocol": d3["protocol"],
                    "supervised_history": supervised_history,
                    "grpo_history": grpo_history,
                },
                checkpoint_path,
            )

        eval_prereg = copy.deepcopy(b41_prereg)
        eval_prereg["exact_raw_attempts_per_condition"] = int(d3["exact_raw_attempts_per_condition"])
        conditions = d0b.load_table1_conditions(
            args.eval_csv,
            limit=int(args.eval_limit),
            condition_dim=int(b41_prereg["condition_dim"]),
            graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
            max_atoms=int(representation_config["max_atoms"]),
        )
        rows: list[dict[str, object]] = []
        skipped = 0
        sample_started = time.perf_counter()
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
                    eval_prereg,
                    device,
                    int(d3["seed"]) * 100000 + index,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "sample_failed",
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                generated = [{"generated_smiles": ""}] * int(d3["exact_raw_attempts_per_condition"])
                skipped += 1
            if len(generated) != int(d3["exact_raw_attempts_per_condition"]):
                generated = (list(generated) + [{"generated_smiles": ""}] * 20)[
                    : int(d3["exact_raw_attempts_per_condition"])
                ]
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
                        "method": d3["protocol"],
                        "family": "b41_event_kernel_energy",
                        "op": "latent_graph_jump",
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
        b41.viability_event_mask = original_mask

    candidate_path = args.output_dir / "d3_event_kernel_energy_table1_n20_candidates.csv"
    d0b.write_rows(candidate_path, rows)
    sampling = {
        "protocol": d3["protocol"],
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": int(d3["exact_raw_attempts_per_condition"]),
        "skipped_count": skipped,
        "candidate_csv": str(candidate_path),
        "checkpoint": str(checkpoint_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
        "oracle_in_training_reward": bool(grpo_history),
        "family_mixer": False,
        "mcs_inference_glue": False,
        "supervised_pairs": len(train_pairs),
        "energy_pairs": len(energy_pairs),
        "skip_grpo": skip_grpo,
        "frozen_teacher_checkpoint": str(args.frozen_model_checkpoint) if args.frozen_model_checkpoint else None,
        "supervised_history": supervised_history,
        "grpo_history": grpo_history,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "sample_sec": round(time.perf_counter() - sample_started, 1),
        "exact_stop_support": exact_support.manifest(),
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


def build_inball_pairs(
    train_csv: Path,
    b41_prereg: dict,
    representation_config: dict,
    eval_sources: set[str],
    *,
    min_tanimoto: float,
    allowed_counts: set[int],
    limit: int,
    seed: int,
    gsk3b_upsample: int,
) -> list:
    rows = []
    for row in base.read_rows(train_csv):
        source = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
        target = graph.canonical_smiles(str(row.get("target_smiles", "") or ""))
        if not source or not target or source == target or source in eval_sources:
            continue
        raw = str(row.get("source_tanimoto", "") or "").strip()
        if raw:
            try:
                if float(raw) < min_tanimoto:
                    continue
            except ValueError:
                continue
        rows.append(row)
    pairs, _counts = base.build_pairs(
        rows,
        max_atoms=int(representation_config["max_atoms"]),
        fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        allowed_counts=allowed_counts,
        timeout=int(b41_prereg["mcs_timeout"]),
        min_common_fraction=float(b41_prereg["min_common_fraction"]),
        limit=int(limit),
        seed=int(seed),
        forbidden_sources=eval_sources,
    )
    kept = []
    for pair in pairs:
        similarity = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles)
        if similarity is None or float(similarity) < min_tanimoto:
            continue
        specs = base.task_specs(pair.row)
        pair.condition = hierarchical.property_latent_slot_tokens(
            b29.source_only_row(pair.row, specs), int(b41_prereg["condition_dim"])
        )
        kept.append(pair)
        if "GSK3B" in str(pair.task) and gsk3b_upsample > 1:
            kept.extend([pair] * (int(gsk3b_upsample) - 1))
    return kept


def pair_in_vocabulary(pair, vocabulary) -> bool:
    try:
        collated = base.pair_collate([pair])
        delta.delta_action_targets(collated["source"], collated["target"], vocabulary)
    except (ValueError, RuntimeError):
        return False
    return True


def keep_vocabulary_pairs(pairs, vocabulary, label: str):
    kept = []
    dropped = 0
    for pair in pairs:
        if pair_in_vocabulary(pair, vocabulary):
            kept.append(pair)
        else:
            dropped += 1
    print(
        json.dumps(
            {
                "dropped": dropped,
                "kept": len(kept),
                "set": label,
                "stage": "vocab_filter",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return kept


def select_grpo_conditions(
    train_csv: Path,
    eval_sources: set[str],
    *,
    condition_dim: int,
    fingerprint_bits: int,
    max_atoms: int,
    limit: int,
    seed: int,
) -> list:
    loaded = d0b.load_table1_conditions(
        train_csv,
        limit=0,
        condition_dim=condition_dim,
        graph_fingerprint_bits=fingerprint_bits,
        max_atoms=max_atoms,
    )
    by_task: dict[str, list] = defaultdict(list)
    for condition in loaded:
        if condition.source_smiles in eval_sources:
            continue
        if condition.task not in REAL_TASKS:
            continue
        by_task[condition.task].append(condition)
    rng = random.Random(int(seed))
    selected = []
    gsk = list(by_task.get("GSK3B:increase", []))
    rng.shuffle(gsk)
    selected.extend(gsk[: max(1, int(limit) // 2)])
    others = []
    for task, items in by_task.items():
        if task == "GSK3B:increase":
            continue
        others.extend(items)
    rng.shuffle(others)
    selected.extend(others[: max(0, int(limit) - len(selected))])
    rng.shuffle(selected)
    return selected[: int(limit)]


def run_grpo(
    model,
    representation,
    vocabulary,
    support,
    support_tensors,
    conditions,
    b41_prereg,
    d3,
    chem: Chemistry,
    device: torch.device,
) -> list[dict[str, float]]:
    group = int(d3["grpo_group_size"])
    model.requires_grad_(False)
    model.denoiser.requires_grad_(True)
    trainable = [parameter for parameter in model.denoiser.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("D3 GRPO: denoiser has no trainable parameters after unfreeze")
    print(
        json.dumps(
            {
                "stage": "grpo_optimizer",
                "trainable_tensors": len(trainable),
                "trainable_params": int(sum(item.numel() for item in trainable)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(d3["grpo_lr"]),
        weight_decay=float(b41_prereg["weight_decay"]),
    )
    sample_prereg = copy.deepcopy(b41_prereg)
    sample_prereg["exact_raw_attempts_per_condition"] = group
    history: list[dict[str, float]] = []
    for epoch in range(1, int(d3["grpo_epochs"]) + 1):
        totals: defaultdict[str, float] = defaultdict(float)
        used = 0
        model.denoiser.train()
        for index, condition in enumerate(conditions):
            optimizer.zero_grad(set_to_none=True)
            try:
                smiles, logprob = sample_group_logprob(
                    model,
                    representation,
                    vocabulary,
                    support,
                    support_tensors,
                    condition.source,
                    np.asarray(condition.condition),
                    sample_prereg,
                    device,
                    int(d3["seed"]) * 1000 + epoch * 100 + index,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "grpo_sample_failed",
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            rewards = torch.tensor(
                [terminal_reward(condition.row, item, chem) for item in smiles],
                dtype=torch.float32,
                device=device,
            )
            if float(rewards.max() - rewards.min()) < 1e-6:
                continue
            advantage = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
            loss = -(advantage.detach() * logprob).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), float(b41_prereg["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["reward"] += float(rewards.mean().detach())
            totals["success065"] += float(sum(item >= 1.0 for item in rewards.tolist()) / max(1, len(rewards)))
            used += 1
            del smiles, logprob, rewards, advantage, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if used == 0:
            raise RuntimeError(f"D3 GRPO epoch {epoch}: every group failed or had zero advantage")
        row = {"epoch": epoch, "groups": used, **{name: value / max(1, used) for name, value in totals.items()}}
        history.append(row)
        print(json.dumps({"stage": "grpo_epoch", **row}, sort_keys=True), flush=True)
    model.eval().requires_grad_(False)
    return history


def sample_group_logprob(
    model,
    representation,
    vocabulary,
    support,
    support_tensors,
    source_example,
    condition_tokens: np.ndarray,
    preregistration: dict,
    device: torch.device,
    seed: int,
) -> tuple[list[str], torch.Tensor]:
    """Sample the GRPO group together so particles interact, matching B41 eval."""
    attempts = int(preregistration["exact_raw_attempts_per_condition"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        particles, _ = b41.b40.orthogonal_latent_particles(
            attempts,
            model.transport_dim,
            generator,
            device,
            float(preregistration["latent_noise_scale"]),
        )
        particles, _ = b41.interacting_transport_particles(
            model,
            representation,
            source_example,
            condition_tokens,
            particles,
            preregistration,
            device,
        )
    source = base.move_graph_batch(graph.collate([source_example] * attempts), device)
    tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], attempts, axis=0)).to(device)
    logprob = torch.zeros(attempts, device=device)
    dummy = torch.zeros((), device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        with torch.no_grad():
            source_node, source_edge = representation.encode(source)
            condition = model.route_condition(tokens)
            cardinality_logits = model.cardinality_logits(
                source_node, source["node_mask"].bool(), condition, particles
            ).float()
            cardinality_probability = torch.softmax(
                cardinality_logits / float(preregistration["cardinality_temperature"]), dim=1
            )
            predicted_cardinality = torch.multinomial(
                cardinality_probability, 1, generator=generator
            ).squeeze(1)
        working = full_graph.working_node_mask(
            source["node_mask"], int(preregistration["birth_capacity"])
        )
        node_actions = torch.full_like(source["atomic_number"], delta.NODE_KEEP)
        edge_actions = torch.full_like(source["bond"], delta.EDGE_KEEP)
        stopped = torch.zeros(attempts, dtype=torch.bool, device=device)
        event_counts = torch.zeros(attempts, dtype=torch.long, device=device)

        def _denoiser_step(gate, node_state, edge_state, jump_time, remaining_mass):
            logits = model.denoiser(
                node_state,
                edge_state,
                source_node,
                source_edge,
                source["node_mask"].bool(),
                working,
                jump_time,
                condition,
                particles,
                remaining_mass,
            )
            return logits + gate.to(dtype=logits.dtype) * 0

        for _ in range(int(preregistration["max_jumps"])):
            jump_time = event_counts.float() / float(preregistration["max_jumps"])
            remaining_mass = (
                predicted_cardinality.float() - event_counts.float()
            ) / float(preregistration["max_jumps"])
            logits = torch.utils.checkpoint.checkpoint(
                _denoiser_step,
                dummy,
                node_actions,
                edge_actions,
                jump_time,
                remaining_mass,
                use_reentrant=False,
            ).float()
            with torch.no_grad():
                legal, _diagnostics = b41.viability_event_mask(
                    model.denoiser,
                    source,
                    node_actions,
                    edge_actions,
                    working,
                    support,
                    support_tensors,
                )
                if bool(stopped.any()):
                    legal = legal.clone()
                    legal[stopped] = False
                    legal[stopped, 0] = True
            temperature = float(preregistration["event_temperature"])
            masked = logits.masked_fill(~legal, -1e9)
            log_pi = F.log_softmax(masked / temperature, dim=1)
            probability = torch.softmax(masked.detach() / temperature, dim=1).clamp_min(1e-8)
            sampled = torch.multinomial(probability, 1, generator=generator).squeeze(1)
            step_logprob = log_pi.gather(1, sampled[:, None]).squeeze(1)
            logprob = logprob + step_logprob * (~stopped).float()
            node_actions = node_actions.clone()
            edge_actions = edge_actions.clone()
            stopped = stopped.clone()
            event_counts = event_counts.clone()
            for index in range(attempts):
                if bool(stopped[index]):
                    continue
                if b38.execute_flat_event(
                    int(sampled[index]),
                    model.denoiser.layout,
                    node_actions,
                    edge_actions,
                    index,
                ):
                    stopped[index] = True
                else:
                    event_counts[index] += 1
            if bool(stopped.all()):
                break
        result = delta.apply_delta_actions(source, node_actions, edge_actions, vocabulary)
    prediction = {key: value.detach().cpu().numpy() for key, value in result.items()}
    smiles = []
    for index in range(attempts):
        decoded, _ = graph.graph_to_smiles(prediction, index)
        smiles.append(graph.canonical_smiles(decoded or "") or "")
    return smiles, logprob


def terminal_reward(row: dict[str, str], smiles: str, chem: Chemistry) -> float:
    if not smiles:
        return 0.0
    scored = evaluate_prediction(
        row,
        smiles,
        task_specs_for_reference(row),
        chem=chem,
        thresholds=[0.65, 0.15],
    )
    tanimoto = scored.get("source_tanimoto")
    return (
        float(bool(scored.get("success_t0.65")))
        + 0.5 * float(bool(scored.get("property_success")))
        + 0.25 * (0.0 if tanimoto is None else float(tanimoto))
    )


if __name__ == "__main__":
    raise SystemExit(main())
