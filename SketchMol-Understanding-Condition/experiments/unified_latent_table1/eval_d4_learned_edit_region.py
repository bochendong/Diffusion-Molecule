#!/usr/bin/env python3
"""D4b: property-conditioned feasible edit region on frozen B41.

Learns (x,p)→P(EDIT) for source atoms/bonds. Hard mask. B41 still decides
how. Controls are per-molecule matched-count random and position-shuffled
learned. Not a method row unless collect says localization was learned.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
import eval_d3_event_kernel_energy as d3  # noqa: E402
import eval_d4_oracle_spatial_mask as d4a  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402

base = b41.base
delta = b41.delta
graph = b41.graph
b39 = b41.b39
b40 = b41.b40
b37 = b41.b37


class PropertyConditionedEditRegion(nn.Module):
    """Source-side P(EDIT | x, p). Does not predict operations or birth."""

    def __init__(self, node_dim: int, edge_dim: int, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU())
        self.atom = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bond = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_latent, edge_latent, node_mask, tokens):
        cond = self.condition(tokens.float())
        cond_node = cond[:, None, :].expand(-1, node_latent.shape[1], -1)
        atom_logit = self.atom(torch.cat([node_latent, cond_node], dim=-1)).squeeze(-1)
        left = node_latent[:, :, None, :].expand(-1, -1, node_latent.shape[1], -1)
        right = node_latent[:, None, :, :].expand(-1, node_latent.shape[1], -1, -1)
        cond_edge = cond[:, None, None, :].expand(-1, node_latent.shape[1], node_latent.shape[1], -1)
        bond_logit = self.bond(torch.cat([left, right, edge_latent, cond_edge], dim=-1)).squeeze(-1)
        bond_logit = 0.5 * (bond_logit + bond_logit.transpose(1, 2))
        atom_logit = atom_logit.masked_fill(~node_mask.bool(), 0)
        pair_mask = node_mask[:, :, None].bool() & node_mask[:, None, :].bool()
        bond_logit = bond_logit.masked_fill(~pair_mask, 0)
        return atom_logit, bond_logit


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
    parser.add_argument("--d4-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def shuffled_learned_labels(
    source_mask: torch.Tensor,
    atom_edit: torch.Tensor,
    bond_edit: torch.Tensor,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute source atoms; conjugate the bond mask. Counts and shape stay, location does not."""
    source_idx = [int(i) for i in source_mask.nonzero(as_tuple=False).flatten().tolist()]
    perm = list(source_idx)
    rng.shuffle(perm)
    mapping = {old: new for old, new in zip(source_idx, perm)}
    new_atom = torch.zeros_like(atom_edit)
    new_bond = torch.zeros_like(bond_edit)
    for old in source_idx:
        new_atom[mapping[old]] = atom_edit[old]
        for other in source_idx:
            new_bond[mapping[old], mapping[other]] = bond_edit[old, other]
    return new_atom.bool(), new_bond.bool()


def pack_condition(tokens) -> torch.Tensor:
    """Slot tokens are [L, D]; the head reads a flat property-conditioned vector."""
    packed = torch.as_tensor(np.asarray(tokens), dtype=torch.float32)
    return packed.reshape(-1)


def labeled_train_items(pairs, vocabulary, representation, device):
    items = []
    dropped = 0
    with torch.no_grad():
        for pair in pairs:
            labels = d4a.oracle_edit_labels(pair.source, pair.target, vocabulary)
            if labels is None:
                dropped += 1
                continue
            atom_edit, bond_edit = labels
            source = base.move_graph_batch(graph.collate([pair.source]), device)
            node_latent, edge_latent = representation.encode(source)
            items.append(
                {
                    "node_latent": node_latent[0].detach().cpu(),
                    "edge_latent": edge_latent[0].detach().cpu(),
                    "node_mask": source["node_mask"][0].detach().cpu().bool(),
                    "tokens": pack_condition(pair.condition),
                    "atom_edit": atom_edit.cpu().bool(),
                    "bond_edit": bond_edit.cpu().bool(),
                }
            )
    return items, dropped


def train_head(head, items, d4, device) -> list[dict[str, float]]:
    batch_size = int(d4["head_batch_size"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(d4["head_lr"]))
    atom_pos = []
    bond_pos = []
    for item in items:
        mask = item["node_mask"]
        atom_pos.append(float(item["atom_edit"][mask].float().mean()) if bool(mask.any()) else 0.0)
        pairs = mask[:, None] & mask[None, :]
        pairs.fill_diagonal_(False)
        pairs = torch.triu(pairs, diagonal=1)
        if bool(pairs.any()):
            bond_pos.append(float(item["bond_edit"][pairs].float().mean()))
    atom_rate = max(1e-4, min(0.999, sum(atom_pos) / max(1, len(atom_pos))))
    bond_rate = max(1e-4, min(0.999, sum(bond_pos) / max(1, len(bond_pos))))
    atom_weight = torch.tensor((1.0 - atom_rate) / atom_rate, device=device)
    bond_weight = torch.tensor((1.0 - bond_rate) / bond_rate, device=device)
    history = []
    head.train()
    rng = random.Random(int(d4["seed"]))
    for epoch in range(1, int(d4["head_epochs"]) + 1):
        order = list(range(len(items)))
        rng.shuffle(order)
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            chunk = [items[index] for index in order[start : start + batch_size]]
            node_latent = torch.stack([item["node_latent"] for item in chunk]).to(device)
            edge_latent = torch.stack([item["edge_latent"] for item in chunk]).to(device)
            node_mask = torch.stack([item["node_mask"] for item in chunk]).to(device)
            tokens = torch.stack([item["tokens"] for item in chunk]).to(device)
            atom_edit = torch.stack([item["atom_edit"] for item in chunk]).to(device).float()
            bond_edit = torch.stack([item["bond_edit"] for item in chunk]).to(device).float()
            atom_logit, bond_logit = head(node_latent, edge_latent, node_mask, tokens)
            atom_loss = F.binary_cross_entropy_with_logits(
                atom_logit[node_mask], atom_edit[node_mask], pos_weight=atom_weight
            )
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            pair_mask = torch.triu(pair_mask, diagonal=1)
            bond_loss = F.binary_cross_entropy_with_logits(
                bond_logit[pair_mask], bond_edit[pair_mask], pos_weight=bond_weight
            )
            loss = atom_loss + bond_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        row = {
            "atom_pos_rate": atom_rate,
            "bond_pos_rate": bond_rate,
            "epoch": epoch,
            "loss": total / max(1, batches),
        }
        history.append(row)
        print(json.dumps({"stage": "head_epoch", **row}, sort_keys=True), flush=True)
    head.eval()
    return history


def predict_hard_mask(head, representation, source_example, condition_tokens, tau, device):
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = pack_condition(condition_tokens).to(device=device)[None]
    with torch.no_grad():
        node_latent, edge_latent = representation.encode(source)
        atom_logit, bond_logit = head(node_latent, edge_latent, source["node_mask"], tokens)
    source_mask = source["node_mask"][0].bool().cpu()
    atom_edit = (atom_logit[0].sigmoid().cpu() >= float(tau)) & source_mask
    bond_edit = bond_logit[0].sigmoid().cpu() >= float(tau)
    bond_edit = torch.triu(bond_edit, diagonal=1)
    bond_edit = (bond_edit | bond_edit.transpose(0, 1)) & source_mask[:, None] & source_mask[None, :]
    return source_mask, atom_edit, bond_edit


def variant_mask(name: str, source_mask, atom_edit, bond_edit, rng):
    if name == "learned_hard":
        return atom_edit, bond_edit
    if name == "random_matched":
        return d4a.matched_random_labels(source_mask, atom_edit, bond_edit, rng)
    if name == "shuffled_learned":
        return shuffled_learned_labels(source_mask, atom_edit, bond_edit, rng)
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    d4 = json.loads(args.d4_protocol_manifest.read_text(encoding="utf-8"))
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
        min_tanimoto=float(d4["min_source_tanimoto"]),
        allowed_counts={int(value) for value in d4["property_counts"]},
        limit=int(d4["train_pair_limit"]),
        seed=int(d4["seed"]),
        gsk3b_upsample=int(d4["gsk3b_upsample"]),
    )
    items, dropped = labeled_train_items(train_pairs, vocabulary, representation, device)
    print(
        json.dumps(
            {"dropped": dropped, "kept": len(items), "stage": "train_labels"},
            sort_keys=True,
        ),
        flush=True,
    )
    if len(items) < 32:
        raise ValueError(f"Need at least 32 labeled region pairs, found {len(items)}")
    condition_dim = int(items[0]["tokens"].numel())
    if any(int(item["tokens"].numel()) != condition_dim for item in items):
        raise ValueError("Condition slot packing is not uniform across train pairs")
    print(json.dumps({"condition_dim": condition_dim, "stage": "head_init"}, sort_keys=True), flush=True)

    head = PropertyConditionedEditRegion(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=condition_dim,
        hidden_dim=int(d4["head_hidden_dim"]),
    ).to(device)
    head_history = train_head(head, items, d4, device)
    torch.save(
        {"model_state": head.state_dict(), "protocol": d4["protocol"], "history": head_history, "condition_dim": int(items[0]["tokens"].numel())},
        args.output_dir / "d4_learned_edit_region.pt",
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

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    predicted = []
    atom_rates = []
    for condition in conditions:
        source_mask, atom_edit, bond_edit = predict_hard_mask(
            head,
            representation,
            condition.source,
            np.asarray(condition.condition),
            d4["hard_threshold"],
            device,
        )
        predicted.append((source_mask, atom_edit, bond_edit))
        atom_rates.append(float(atom_edit[source_mask].float().mean()) if bool(source_mask.any()) else 0.0)
    print(
        json.dumps(
            {
                "mean_predicted_atom_edit_rate": sum(atom_rates) / max(1, len(atom_rates)),
                "stage": "predicted_masks",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    spatial = d4a.SpatialHardMaskSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = spatial
    try:
        for variant in d4["variants"]:
            variant_dir = args.output_dir / str(variant)
            variant_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []
            skipped = 0
            sample_started = time.perf_counter()
            for index, condition in enumerate(conditions):
                source_mask, atom_edit, bond_edit = predicted[index]
                rng = random.Random(int(d4["seed"]) * 100000 + index)
                atom_mask, bond_mask = variant_mask(str(variant), source_mask, atom_edit, bond_edit, rng)
                spatial.event_allow = d4a.spatial_event_allowance(
                    model.denoiser.layout, source_mask, atom_mask, bond_mask
                )
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
                        int(d4["seed"]) * 100000 + index,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "condition_id": condition.condition_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "stage": "sample_failed",
                                "variant": variant,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    generated = [{"generated_smiles": ""}] * int(d4["exact_raw_attempts_per_condition"])
                    skipped += 1
                if len(generated) != int(d4["exact_raw_attempts_per_condition"]):
                    generated = (list(generated) + [{"generated_smiles": ""}] * 20)[
                        : int(d4["exact_raw_attempts_per_condition"])
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
                            "method": f"{d4['protocol']}/{variant}",
                            "family": "b41_learned_edit_region",
                            "op": "latent_graph_jump",
                            "atom_edit_count": int(atom_mask[source_mask].sum()),
                            "bond_edit_count": int(torch.triu(bond_mask, 1).sum()),
                        }
                    )
                if (index + 1) % 20 == 0 or index + 1 == len(conditions):
                    elapsed = time.perf_counter() - sample_started
                    done = index + 1
                    sec_per = elapsed / done
                    print(
                        json.dumps(
                            {
                                "done": done,
                                "eta_sec": round(sec_per * (len(conditions) - done), 1),
                                "sec_per_condition": round(sec_per, 2),
                                "stage": "sampled",
                                "total": len(conditions),
                                "variant": variant,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            candidate_path = variant_dir / f"d4_{variant}_table1_n20_candidates.csv"
            d0b.write_rows(candidate_path, rows)
            sampling = {
                "protocol": d4["protocol"],
                "variant": variant,
                "not_ours": True,
                "device": str(device),
                "loaded_conditions": len(conditions),
                "candidate_rows": len(rows),
                "attempts_per_condition": int(d4["exact_raw_attempts_per_condition"]),
                "skipped_count": skipped,
                "candidate_csv": str(candidate_path),
                "molecular_candidate_ranking": False,
                "task_router": False,
                "oracle_in_environment": False,
                "mean_predicted_atom_edit_rate": sum(atom_rates) / max(1, len(atom_rates)),
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "sample_sec": round(time.perf_counter() - sample_started, 1),
                "exact_stop_support": spatial.manifest(),
            }
            (variant_dir / "sampling_summary.json").write_text(
                json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(sampling, indent=2, sort_keys=True), flush=True)
    finally:
        b41.viability_event_mask = original_mask
    (args.output_dir / "train_summary.json").write_text(
        json.dumps({"head_history": head_history, "train_pairs": len(items)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
