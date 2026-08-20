#!/usr/bin/env python3
"""D5a: property-conditioned global soft preservation on frozen B41.

Diagnostic only. Learns eta from train-pair Tanimoto using property tokens,
then subtracts eta from non-STOP event logits. No spatial mask, no GNN on x.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
import eval_d3_event_kernel_energy as d3  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402

base = b41.base
delta = b41.delta
graph = b41.graph
b37 = b41.b37
b39 = b41.b39
b40 = b41.b40


class PropertyPreservationHead(nn.Module):
    """p → predicted source Tanimoto. No graph."""

    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(condition_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.score(tokens.float()).squeeze(-1)


class PreservationEventBias(nn.Module):
    """Subtract eta * cost from event logits. STOP cost is 0.

    Proxy remaining denoiser attributes so B41 support can read pair_left etc.
    """

    def __init__(self, denoiser: nn.Module, cost: torch.Tensor) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.register_buffer("cost", cost)
        self.eta = 0.0

    @property
    def layout(self):
        return self.denoiser.layout

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            denoiser = self._modules.get("denoiser")
            if denoiser is None:
                raise
            return getattr(denoiser, name)

    def forward(self, *args, **kwargs):
        logits = self.denoiser(*args, **kwargs)
        return logits - logits.new_tensor(float(self.eta)) * self.cost.to(dtype=logits.dtype)


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
    parser.add_argument("--d5-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def pack_condition(tokens) -> torch.Tensor:
    packed = torch.as_tensor(np.asarray(tokens), dtype=torch.float32)
    return packed.reshape(-1)


def event_cost_vector(layout, *, stop_cost: float, edit_cost: float) -> torch.Tensor:
    cost = torch.full((int(layout.total_events),), float(edit_cost))
    cost[0] = float(stop_cost)
    return cost


def eta_from_prediction(
    predicted: float,
    train_mean: float,
    train_std: float,
    d5: dict,
) -> float:
    scale = float(d5["eta_scale"])
    z = (float(predicted) - float(train_mean)) / max(float(train_std), 1e-6)
    return float(min(float(d5["eta_max"]), max(float(d5["eta_min"]), scale * z)))


def main() -> int:
    args = parse_args()
    d5 = json.loads(args.d5_protocol_manifest.read_text(encoding="utf-8"))
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
    representation.eval().requires_grad_(False)
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    eval_sources = {
        graph.canonical_smiles(row.get("source_smiles", ""))
        for row in base.read_rows(args.eval_csv)
    }
    eval_sources.discard("")
    print(json.dumps({"stage": "build_train_pairs"}, sort_keys=True), flush=True)
    train_pairs = d3.build_inball_pairs(
        args.train_csv,
        b41_prereg,
        representation_config,
        eval_sources,
        min_tanimoto=float(d5["min_source_tanimoto"]),
        allowed_counts={int(value) for value in d5["property_counts"]},
        limit=int(d5["train_pair_limit"]),
        seed=int(d5["seed"]),
        gsk3b_upsample=int(d5["gsk3b_upsample"]),
    )
    items = []
    for pair in train_pairs:
        similarity = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles)
        if similarity is None:
            continue
        items.append({"tokens": pack_condition(pair.condition), "tanimoto": float(similarity)})
    print(json.dumps({"kept": len(items), "stage": "train_labels"}, sort_keys=True), flush=True)
    if len(items) < 32:
        raise ValueError(f"Need at least 32 train pairs with Tanimoto, found {len(items)}")
    condition_dim = int(items[0]["tokens"].numel())
    if any(int(item["tokens"].numel()) != condition_dim for item in items):
        raise ValueError("Condition slot packing is not uniform")
    labels = torch.tensor([item["tanimoto"] for item in items], dtype=torch.float32)
    train_mean = float(labels.mean())
    train_std = float(labels.std().clamp_min(1e-6))
    head = PropertyPreservationHead(condition_dim).to(device)
    history = train_head(head, items, d5, device)
    torch.save(
        {
            "model_state": head.state_dict(),
            "protocol": d5["protocol"],
            "history": history,
            "train_mean": train_mean,
            "train_std": train_std,
            "condition_dim": condition_dim,
        },
        args.output_dir / "d5a_preservation_strength.pt",
    )

    support = b40.build_support(fit_pairs, vocabulary)
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
        property_count=len(b41.unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)
    cost = event_cost_vector(
        model.denoiser.layout,
        stop_cost=float(d5["stop_event_cost"]),
        edit_cost=float(d5["edit_event_cost"]),
    ).to(device)
    biased = PreservationEventBias(model.denoiser, cost).to(device)
    model.denoiser = biased

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    property_etas = []
    head.eval()
    with torch.no_grad():
        for condition in conditions:
            tokens = pack_condition(condition.condition).to(device)
            predicted = float(head(tokens[None, :])[0].cpu())
            eta = eta_from_prediction(predicted, train_mean, train_std, d5)
            property_etas.append(
                {"task": condition.task, "predicted_tanimoto": predicted, "eta": eta}
            )
    by_task_eta: dict[str, list[float]] = defaultdict(list)
    for row in property_etas:
        by_task_eta[str(row["task"])].append(float(row["eta"]))
    eta_by_task = {
        task: round(sum(values) / max(1, len(values)), 4) for task, values in sorted(by_task_eta.items())
    }
    print(
        json.dumps(
            {
                "mean_property_eta": round(sum(row["eta"] for row in property_etas) / max(1, len(property_etas)), 4),
                "eta_by_task": eta_by_task,
                "train_tanimoto_mean": round(train_mean, 4),
                "train_tanimoto_std": round(train_std, 4),
                "stage": "eta_ready",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "history": history,
                "train_pairs": len(items),
                "train_mean": train_mean,
                "train_std": train_std,
                "eta_by_task": eta_by_task,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
        for variant in d5["variants"]:
            variant_dir = args.output_dir / str(variant)
            variant_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []
            skipped = 0
            sample_started = time.perf_counter()
            for index, condition in enumerate(conditions):
                if str(variant) == "constant_eta":
                    biased.eta = float(d5["constant_eta"])
                else:
                    biased.eta = float(property_etas[index]["eta"])
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
                        int(d5["seed"]) * 100000 + index,
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
                    generated = [{"generated_smiles": ""}] * int(d5["exact_raw_attempts_per_condition"])
                    skipped += 1
                if len(generated) != int(d5["exact_raw_attempts_per_condition"]):
                    generated = (list(generated) + [{"generated_smiles": ""}] * 20)[
                        : int(d5["exact_raw_attempts_per_condition"])
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
                            "method": d5["protocol"],
                            "family": "b41_preservation_eta",
                            "op": "latent_graph_jump",
                            "variant": variant,
                            "eta": biased.eta,
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
            candidate_path = variant_dir / f"d5a_{variant}_table1_n20_candidates.csv"
            d0b.write_rows(candidate_path, rows)
            sampling = {
                "protocol": d5["protocol"],
                "variant": variant,
                "device": str(device),
                "eval_csv": str(args.eval_csv),
                "loaded_conditions": len(conditions),
                "candidate_rows": len(rows),
                "attempts_per_condition": int(d5["exact_raw_attempts_per_condition"]),
                "skipped_count": skipped,
                "candidate_csv": str(candidate_path),
                "molecular_candidate_ranking": False,
                "task_router": False,
                "oracle_in_environment": False,
                "spatial_mask": False,
                "not_ours": True,
                "eta_by_task": eta_by_task if str(variant) == "property_alpha" else {"all": float(d5["constant_eta"])},
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "sample_sec": round(time.perf_counter() - sample_started, 1),
                "exact_stop_support": exact_support.manifest(),
            }
            (variant_dir / "sampling_summary.json").write_text(
                json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(sampling, indent=2, sort_keys=True), flush=True)
    finally:
        b41.viability_event_mask = original_mask
    return 0


def train_head(head, items, d5, device) -> list[dict[str, float]]:
    batch_size = int(d5["head_batch_size"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(d5["head_lr"]))
    history = []
    head.train()
    for epoch in range(1, int(d5["head_epochs"]) + 1):
        order = torch.randperm(len(items)).tolist()
        total = 0.0
        seen = 0
        for start in range(0, len(items), batch_size):
            batch = [items[index] for index in order[start : start + batch_size]]
            tokens = torch.stack([item["tokens"] for item in batch]).to(device)
            target = torch.tensor([item["tanimoto"] for item in batch], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(head(tokens), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        row = {"epoch": epoch, "loss": total / max(1, seen)}
        history.append(row)
        print(json.dumps({"stage": "head_epoch", **row}, sort_keys=True), flush=True)
    head.eval()
    return history


if __name__ == "__main__":
    raise SystemExit(main())
