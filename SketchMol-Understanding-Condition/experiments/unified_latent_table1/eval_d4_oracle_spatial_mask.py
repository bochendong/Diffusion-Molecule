#!/usr/bin/env python3
"""D4a: oracle keep/edit mask on frozen B41. Not a method.

Hypothesis: B41's bottleneck is not knowing where not to edit.
π is (M_keep, M_editable) from aligned source-target labels, never a molecule.
B41 still chooses every WRITE / DELETE / SET. Birth quota is unchanged.
"""

from __future__ import annotations

import argparse
import csv
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

base = b41.base
delta = b41.delta
graph = b41.graph
b38 = b41.b38
b39 = b41.b39
b40 = b41.b40
b37 = b41.b37


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
    parser.add_argument("--variant", required=True, choices=("oracle_hard", "random_matched"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


class SpatialHardMaskSupport:
    """AND a static source-side keep/edit event mask onto exact-STOP legal events."""

    def __init__(self, vocabulary) -> None:
        self.inner = valid_terminal.ExactMoleculeStopSupport(vocabulary)
        self.event_allow: torch.Tensor | None = None

    def __call__(self, field, source, node_actions, edge_actions, working, support, support_tensors):
        legal, diagnostics = self.inner(
            field, source, node_actions, edge_actions, working, support, support_tensors
        )
        if self.event_allow is not None:
            allow = self.event_allow.to(device=legal.device, dtype=torch.bool)
            legal = legal & allow[None, :]
            stuck = ~legal.any(dim=1)
            legal[stuck, 0] = True
            diagnostics = dict(diagnostics)
            diagnostics["spatial_keep_masked"] = ~allow
        return legal, diagnostics

    def manifest(self) -> dict[str, object]:
        return self.inner.manifest()


def pair_endpoints(nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    left: list[int] = []
    right: list[int] = []
    for i in range(nodes - 1):
        for j in range(i + 1, nodes):
            left.append(i)
            right.append(j)
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def spatial_event_allowance(
    layout: b38.EventLayout,
    source_mask: torch.Tensor,
    atom_edit: torch.Tensor,
    bond_edit: torch.Tensor,
) -> torch.Tensor:
    """True = event still allowed. Birth nodes unconstrained; keep atoms cannot attach births."""
    nodes = int(layout.nodes)
    allow = torch.ones(layout.total_events, dtype=torch.bool)
    source_mask = source_mask.bool()
    atom_edit = atom_edit.bool()
    bond_edit = bond_edit.bool()
    keep_atom = source_mask & ~atom_edit
    allow[layout.node_delete_offset : layout.node_write_offset] = ~keep_atom
    write_ok = (~keep_atom).unsqueeze(1).expand(nodes, layout.node_payloads).reshape(-1)
    allow[layout.node_write_offset : layout.edge_delete_offset] = write_ok
    left, right = pair_endpoints(nodes)
    both_source = source_mask[left] & source_mask[right]
    one_source = source_mask[left] ^ source_mask[right]
    bond_keep = both_source & ~bond_edit[left, right]
    keep_to_birth = one_source & torch.where(source_mask[left], keep_atom[left], keep_atom[right])
    forbid_edge = bond_keep | keep_to_birth
    allow[layout.edge_delete_offset : layout.edge_set_offset] = ~forbid_edge
    set_ok = (~forbid_edge).unsqueeze(1).expand(-1, layout.edge_payloads).reshape(-1)
    allow[layout.edge_set_offset :] = set_ok
    allow[0] = True
    return allow


def oracle_edit_labels(source_example, target_example, vocabulary) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        source = graph.collate([source_example])
        target = graph.collate([target_example])
        node_actions, edge_actions = delta.delta_action_targets(source, target, vocabulary)
    except (ValueError, RuntimeError):
        return None
    source_mask = source["node_mask"][0].bool()
    atom_edit = node_actions[0].ne(delta.NODE_KEEP) & source_mask
    edge_edit = edge_actions[0].ne(delta.EDGE_KEEP)
    edge_edit = torch.triu(edge_edit, diagonal=1)
    edge_edit = edge_edit | edge_edit.transpose(0, 1)
    atom_edit = atom_edit | (edge_edit.any(dim=1) & source_mask)
    return atom_edit, edge_edit


def matched_random_labels(
    source_mask: torch.Tensor,
    atom_edit: torch.Tensor,
    bond_edit: torch.Tensor,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_idx = [int(i) for i in source_mask.nonzero(as_tuple=False).flatten().tolist()]
    k_atom = int((atom_edit & source_mask).sum())
    pairs = [(i, j) for i in source_idx for j in source_idx if i < j]
    source_source = torch.zeros_like(bond_edit)
    for i, j in pairs:
        if bool(bond_edit[i, j] | bond_edit[j, i]):
            source_source[i, j] = True
    k_bond = int(source_source.sum())
    random_atom = torch.zeros_like(atom_edit)
    if source_idx and k_atom > 0:
        chosen = rng.sample(source_idx, min(k_atom, len(source_idx)))
        random_atom[chosen] = True
    random_bond = torch.zeros_like(bond_edit)
    if pairs and k_bond > 0:
        chosen_pairs = rng.sample(pairs, min(k_bond, len(pairs)))
        for i, j in chosen_pairs:
            random_bond[i, j] = True
            random_bond[j, i] = True
    return random_atom, random_bond


def load_targets(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition_id = str(row.get("condition_id", "") or row.get("example_id", "") or "")
            target = graph.canonical_smiles(str(row.get("target_smiles", "") or ""))
            if condition_id and target:
                targets[condition_id] = target
    return targets


def main() -> int:
    args = parse_args()
    d4 = json.loads(args.d4_protocol_manifest.read_text(encoding="utf-8"))
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    variant_dir = args.output_dir / str(args.variant)
    variant_dir.mkdir(parents=True, exist_ok=True)
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
    targets = load_targets(args.eval_csv)
    spatial = SpatialHardMaskSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = spatial
    rows: list[dict[str, object]] = []
    skipped = 0
    aligned = 0
    labeled = 0
    atom_edit_rates: list[float] = []
    bond_edit_rates: list[float] = []
    sample_started = time.perf_counter()
    try:
        for index, condition in enumerate(conditions):
            spatial.event_allow = None
            target_smiles = targets.get(str(condition.condition_id), "")
            labels = None
            if target_smiles:
                aligned_pair = base.align_pair(
                    condition.source_smiles,
                    target_smiles,
                    max_atoms=int(representation_config["max_atoms"]),
                    fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
                    timeout=int(b41_prereg["mcs_timeout"]),
                    min_common_fraction=float(b41_prereg["min_common_fraction"]),
                )
                if aligned_pair is not None:
                    aligned += 1
                    aligned_source, target_example, _common = aligned_pair
                    labels = oracle_edit_labels(aligned_source, target_example, vocabulary)
            if labels is not None:
                labeled += 1
                atom_edit, bond_edit = labels
                source_mask = torch.as_tensor(aligned_source.node_mask).bool()
                if str(args.variant) == "random_matched":
                    rng = random.Random(int(d4["seed"]) * 100000 + index)
                    atom_edit, bond_edit = matched_random_labels(source_mask, atom_edit, bond_edit, rng)
                n_source = max(1, int(source_mask.sum()))
                ss = int(source_mask.sum()) * max(0, int(source_mask.sum()) - 1) / 2
                atom_edit_rates.append(float((atom_edit & source_mask).sum()) / n_source)
                bond_edit_rates.append(
                    float(torch.triu(bond_edit, 1).sum()) / max(1.0, ss)
                )
                spatial.event_allow = spatial_event_allowance(
                    model.denoiser.layout, source_mask, atom_edit, bond_edit
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
                            "stage": "sample_failed",
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
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
                        "method": f"{d4['protocol']}/{args.variant}",
                        "family": "b41_oracle_spatial_mask",
                        "op": "latent_graph_jump",
                        "mask_applied": labels is not None,
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
                            "aligned": aligned,
                            "done": done,
                            "eta_sec": round(sec_per * (len(conditions) - done), 1),
                            "labeled": labeled,
                            "sec_per_condition": round(sec_per, 2),
                            "total": len(conditions),
                            "variant": args.variant,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.viability_event_mask = original_mask

    candidate_path = variant_dir / f"d4_{args.variant}_table1_n20_candidates.csv"
    d0b.write_rows(candidate_path, rows)
    sampling = {
        "protocol": d4["protocol"],
        "variant": args.variant,
        "not_ours": True,
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "aligned_count": aligned,
        "labeled_count": labeled,
        "alignment_rate": aligned / max(1, len(conditions)),
        "label_rate": labeled / max(1, len(conditions)),
        "mean_atom_edit_rate": sum(atom_edit_rates) / max(1, len(atom_edit_rates)),
        "mean_bond_edit_rate": sum(bond_edit_rates) / max(1, len(bond_edit_rates)),
        "candidate_rows": len(rows),
        "attempts_per_condition": int(d4["exact_raw_attempts_per_condition"]),
        "skipped_count": skipped,
        "candidate_csv": str(candidate_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
        "oracle_edit_region_from_aligned_target": True,
        "oracle_target_molecule_as_output": False,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "sample_sec": round(time.perf_counter() - sample_started, 1),
        "exact_stop_support": spatial.manifest(),
    }
    (variant_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
