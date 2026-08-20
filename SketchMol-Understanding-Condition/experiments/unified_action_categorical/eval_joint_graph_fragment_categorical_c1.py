#!/usr/bin/env python3
"""C1: sample n=20 from a graph+fragment mixture categorical. No ranking.

Graph family logits are frozen GraphEditDSL policy log-probabilities over
enumerated executable programs. Fragment family logits are the frozen B31
joint (site, token) energy. Each draw first chooses a family with a frozen
equal prior, then samples inside that family. Instruction never selects the
family. SMILES are assembled only after the draw.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
WORKTREE_LATENT = PROJECT_DIR / "experiments" / "unified_latent_flow"
WORKTREE_PROJECT = PROJECT_DIR
WORKTREE_UCA = WORKTREE_PROJECT / "experiments" / "unified_constraint_agent"
for path in (WORKTREE_LATENT, WORKTREE_PROJECT, WORKTREE_UCA, SCRIPT_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import assay_joint_site_token_latent as b31  # noqa: E402
import table1_energy_tilted_latent_transfer as b29  # noqa: E402

b27 = b31.b27
b28 = b31.b28
kernel = b31.kernel
graph = b31.graph
base = b29.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--graph-candidate-csv", required=True, type=Path)
    parser.add_argument("--b31-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--fragment-checkpoint", required=True, type=Path)
    parser.add_argument("--b31-protocol-manifest", required=True, type=Path)
    parser.add_argument("--c1-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    c1 = json.loads(args.c1_protocol_manifest.read_text(encoding="utf-8"))
    preregistration = json.loads(args.b31_protocol_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    representation, _config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    fragment_model, target_fragments, target_endpoints, _frozen = (
        b27.load_frozen_fragment_model(args.fragment_checkpoint, device, preregistration)
    )
    payload = torch.load(args.b31_checkpoint, map_location=device, weights_only=False)
    config = dict(payload["model_config"])
    energy_model = b27.LatentPropertyEnergy(
        endpoint_dim=int(config["endpoint_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    energy_model.load_state_dict(payload["model_state"])
    energy_model.eval()
    for parameter in energy_model.parameters():
        parameter.requires_grad_(False)

    conditions = load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(preregistration["condition_dim"]),
        graph_fingerprint_bits=int(preregistration["graph_fingerprint_bits"]),
    )
    graph_family = load_graph_family(args.graph_candidate_csv)
    source_pairs = [
        SimpleNamespace(source_smiles=row.source_smiles, source=row.source)
        for row in conditions
    ]
    latents = kernel.encode_sources(
        representation,
        source_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    generation_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        num_attempts=int(c1["exact_raw_attempts_per_condition"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        energy_weight=float(preregistration["energy_weight"]),
        seed=int(c1["seed"]),
    )
    family_prior_graph = float(c1["family_prior_graph"])
    attempts = int(c1["exact_raw_attempts_per_condition"])

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    family_counts = {"graph": 0, "fragment": 0}
    for index, condition in enumerate(conditions):
        seed = int(generation_config.seed) * 100000 + index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        graph_actions = graph_family.get(condition.condition_id, [])
        try:
            frag_sites, frag_tokens, frag_probs = fragment_probabilities(
                fragment_model,
                energy_model,
                condition,
                latents[index],
                target_fragments,
                target_endpoints,
                generation_config,
                device,
            )
        except Exception as exc:
            frag_sites, frag_tokens, frag_probs = [], [], None
            skipped.append(
                {
                    "condition_id": condition.condition_id,
                    "stage": "fragment_family",
                    "error": str(exc),
                }
            )
        samples = sample_mixture(
            graph_actions=graph_actions,
            frag_sites=frag_sites,
            frag_tokens=frag_tokens,
            frag_probs=frag_probs,
            attempts=attempts,
            family_prior_graph=family_prior_graph,
            generator=generator,
        )
        if len(samples) != attempts:
            skipped.append(
                {
                    "condition_id": condition.condition_id,
                    "stage": "sample",
                    "error": f"got {len(samples)} samples",
                }
            )
            continue
        for attempt, sample in enumerate(samples, start=1):
            family_counts[str(sample["family"])] += 1
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "task": condition.task,
                    "source_smiles": condition.source_smiles,
                    "generated_smiles": sample["smiles"],
                    "sample_index": attempt,
                    "candidate_index": attempt,
                    "method": c1["protocol"],
                    "family": sample["family"],
                    "op": sample.get("op", ""),
                    "target_fragment_token": sample.get("token", ""),
                    "site_core": sample.get("core", ""),
                }
            )
        if (index + 1) % 20 == 0 or index + 1 == len(conditions):
            print(
                json.dumps(
                    {
                        "stage": "sampled",
                        "done": index + 1,
                        "total": len(conditions),
                        "family_counts": dict(family_counts),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    candidate_path = args.candidate_output or (args.output_dir / "c1_table1_n20_candidates.csv")
    write_rows(candidate_path, rows)
    sampling = {
        "protocol": c1["protocol"],
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": attempts,
        "family_prior_graph": family_prior_graph,
        "family_counts": family_counts,
        "family_fraction_fragment": family_counts["fragment"] / max(1, sum(family_counts.values())),
        "skipped": skipped[:50],
        "skipped_count": len(skipped),
        "candidate_csv": str(candidate_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


def load_table1_conditions(
    path: Path,
    *,
    limit: int,
    condition_dim: int,
    graph_fingerprint_bits: int,
) -> list[b29.TransferCondition]:
    out: list[b29.TransferCondition] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            specs = base.task_specs(raw)
            task = b29.table1_task_key(specs)
            if task == "unknown":
                continue
            row = b29.source_only_row(raw, specs)
            source = graph.canonical_smiles(row["source_smiles"])
            if not source or not row["condition_id"]:
                continue
            source_graph = graph.molecule_example(
                source, max_atoms=64, fingerprint_bits=int(graph_fingerprint_bits)
            )
            if source_graph is None:
                continue
            out.append(
                b29.TransferCondition(
                    row=row,
                    source_smiles=source,
                    source=source_graph,
                    condition=kernel.hierarchical.property_latent_slot_tokens(
                        row, int(condition_dim)
                    ),
                    property_count=len(specs),
                    task=task,
                    condition_id=row["condition_id"],
                )
            )
            if limit > 0 and len(out) >= int(limit):
                break
    return out


def load_graph_family(path: Path) -> dict[str, list[tuple[float, str]]]:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            condition_id = str(raw.get("condition_id", "") or "").strip()
            smiles = str(raw.get("generated_smiles", "") or "").strip()
            logprob = raw.get("graph_action_policy_logprob", "")
            if not condition_id or not smiles or not str(logprob).strip():
                continue
            grouped[condition_id].append((float(logprob), smiles))
    return dict(grouped)


@torch.no_grad()
def fragment_family_scores(
    fragment_model,
    energy_model,
    condition: b29.TransferCondition,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> tuple[list, list[str], torch.Tensor, torch.Tensor]:
    sites, contexts_np, site_logits_np = b31.condition_site_state(
        fragment_model, condition, source_latent, config, device
    )
    contexts = torch.from_numpy(contexts_np).to(device)
    vocabulary = torch.from_numpy(target_endpoints).to(device)
    energy = b28.token_energy(
        energy_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    standardized = (energy - energy.mean()) / energy.std().clamp_min(
        float(config.energy_scale_floor)
    )
    site_logits = torch.from_numpy(site_logits_np).to(device)
    logits = site_logits[:, None] / max(float(config.site_temperature), 1e-6)
    logits = logits.expand_as(standardized).clone()
    logits = logits + float(config.energy_weight) * standardized
    token_lookup = {token: index for index, token in enumerate(target_fragments)}
    for site_index, site in enumerate(sites):
        current_index = token_lookup.get(site.variable)
        if current_index is not None:
            logits[site_index, current_index] = -torch.inf
    flat_logits = logits.reshape(-1).float().cpu()
    probabilities = torch.softmax(flat_logits, dim=0)
    return sites, list(target_fragments), probabilities, flat_logits


@torch.no_grad()
def fragment_probabilities(
    fragment_model,
    energy_model,
    condition: b29.TransferCondition,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> tuple[list, list[str], torch.Tensor]:
    sites, tokens, probabilities, _logits = fragment_family_scores(
        fragment_model,
        energy_model,
        condition,
        source_latent,
        target_fragments,
        target_endpoints,
        config,
        device,
    )
    return sites, tokens, probabilities


def sample_mixture(
    *,
    graph_actions: Sequence[tuple[float, str]],
    frag_sites: Sequence[object],
    frag_tokens: Sequence[str],
    frag_probs: torch.Tensor | None,
    attempts: int,
    family_prior_graph: float,
    generator: torch.Generator,
) -> list[dict[str, object]]:
    graph_ok = bool(graph_actions)
    frag_ok = bool(frag_sites) and frag_probs is not None and int(frag_probs.numel()) > 0
    if not graph_ok and not frag_ok:
        return []
    graph_probs = None
    if graph_ok:
        logits = torch.tensor([item[0] for item in graph_actions], dtype=torch.float32)
        graph_probs = torch.softmax(logits, dim=0)
    output: list[dict[str, object]] = []
    for _ in range(int(attempts)):
        if graph_ok and frag_ok:
            use_graph = bool(
                torch.rand(1, generator=generator).item() < float(family_prior_graph)
            )
        else:
            use_graph = graph_ok
        if use_graph:
            assert graph_probs is not None
            index = int(torch.multinomial(graph_probs, 1, generator=generator).item())
            _logprob, smiles = graph_actions[index]
            output.append({"family": "graph", "smiles": smiles, "op": "graph_edit"})
            continue
        assert frag_probs is not None
        flat_index = int(torch.multinomial(frag_probs, 1, generator=generator).item())
        site_index, token_index = divmod(flat_index, len(frag_tokens))
        site = frag_sites[site_index]
        token = frag_tokens[token_index]
        smiles = kernel.fragments.join_fragments(site.core, token)
        output.append(
            {
                "family": "fragment",
                "smiles": smiles or "",
                "op": "replace_attachment",
                "token": token,
                "core": site.core,
            }
        )
    return output


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
