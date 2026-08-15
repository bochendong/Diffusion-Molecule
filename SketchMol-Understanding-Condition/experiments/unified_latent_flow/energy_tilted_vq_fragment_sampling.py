#!/usr/bin/env python3
"""Sample one fragment token from a learned energy-tilted VQ distribution.

B28 keeps the B24 flow and the B27 train-only energy frozen.  The transported
latent defines a distance prior over the train-only fragment codebook; the
learned property energy tilts that categorical latent distribution.  One token
is sampled directly and assembled once.  Tokens are latent decoder states,
not generated molecule candidates: no molecule is constructed, checked,
scored, sorted, rejected, retried, or repaired before the single draw.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import latent_property_energy_guidance as b27  # noqa: E402


kernel = b27.kernel
base = b27.base
belief = b27.belief
graph = b27.graph
unified = b27.unified

PROTOCOL = "energy_tilted_vq_fragment_sampling_v28"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--energy-checkpoint", type=Path, required=True)
    parser.add_argument("--energy-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_fragment_protocol": kernel.PROTOCOL,
        "frozen_energy_protocol": b27.PROTOCOL,
        "b26_heldout_access": False,
        "official_test_access": False,
        "property_oracle_generation_access": False,
        "molecular_candidate_ranking": False,
        "num_attempts": 20,
        "distance_temperature": 0.03,
        "energy_weight": 1.25,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"B28 preregistration drift: {drift}")
    return payload


def load_energy(
    checkpoint_path: Path,
    summary_path: Path,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[b27.LatentPropertyEnergy, dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != preregistration["frozen_energy_protocol"]:
        raise ValueError("B28 energy summary protocol drift")
    manifest = dict(summary.get("manifest", {}))
    if manifest.get("b26_heldout_access") is not False:
        raise ValueError("B28 refuses an energy checkpoint that accessed B26")
    if manifest.get("official_test_access") is not False:
        raise ValueError("B28 refuses an energy checkpoint that accessed official test")
    calibration = dict(summary.get("calibration", {}))
    prerequisites = dict(preregistration["energy_prerequisites"])
    for key, threshold in prerequisites.items():
        if float(calibration.get(key, 0.0)) < float(threshold):
            raise ValueError(
                f"B28 energy prerequisite failed: {key}="
                f"{calibration.get(key)} < {threshold}"
            )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("stage") != preregistration["frozen_energy_protocol"]:
        raise ValueError("B28 energy checkpoint protocol drift")
    config = dict(payload["model_config"])
    model = b27.LatentPropertyEnergy(
        endpoint_dim=int(config["endpoint_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, summary


@torch.no_grad()
def token_energy(
    energy_model: b27.LatentPropertyEnergy,
    vocabulary: torch.Tensor,
    context: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    attempts = context.shape[0]
    values: list[torch.Tensor] = []
    for start in range(0, vocabulary.shape[0], int(chunk_size)):
        endpoints = vocabulary[start : start + int(chunk_size)]
        count = endpoints.shape[0]
        flat_endpoint = endpoints[None, :, :].expand(attempts, count, -1).reshape(
            attempts * count, -1
        )
        flat_context = context[:, None, :].expand(attempts, count, -1).reshape(
            attempts * count, -1
        )
        margin, strict_logit = energy_model(flat_endpoint, flat_context)
        values.append(
            (margin + 0.25 * torch.tanh(strict_logit)).reshape(attempts, count)
        )
    return torch.cat(values, dim=1)


@torch.no_grad()
def tilted_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: b27.LatentPropertyEnergy,
    pair: object,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    sites = kernel.source_sites(pair.source_smiles, config)
    if not sites:
        return [
            {
                "smiles": "",
                "site_core": "",
                "source_fragment": "",
                "target_fragment_token": "",
                "latent_norm": 0.0,
                "quantization_distance": float("inf"),
                "site_entropy": 0.0,
                "token_energy": 0.0,
                "token_probability": 0.0,
                "token_distribution_entropy": 0.0,
            }
            for _ in range(int(config.num_attempts))
        ]
    attempts = int(config.num_attempts)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    source = torch.from_numpy(np.repeat(source_latent[None, :], attempts, axis=0)).to(
        device
    )
    condition = torch.from_numpy(
        np.repeat(pair.condition[None, ...], attempts, axis=0)
    ).to(device)
    site_matrix = np.stack([site.feature for site in sites]).astype(np.float32)
    site_tensor = torch.from_numpy(
        np.repeat(site_matrix[None, ...], attempts, axis=0)
    ).to(device)
    site_mask = torch.ones(
        attempts, len(sites), dtype=torch.bool, device=device
    )
    request = fragment_model.request_context(source, condition)
    site_logits, site_hidden = fragment_model.site_logits(
        request, site_tensor, site_mask
    )
    site_probabilities = torch.softmax(
        site_logits.float() / max(float(config.site_temperature), 1e-4), dim=-1
    )
    selected_sites = torch.multinomial(
        site_probabilities, 1, replacement=True, generator=generator
    ).squeeze(-1)
    selected_hidden = site_hidden[
        torch.arange(attempts, device=device), selected_sites
    ]
    context = fragment_model.context_for_site(request, selected_hidden)
    latent = torch.randn(
        attempts,
        fragment_model.endpoint_dim,
        generator=generator,
        device=device,
        dtype=context.dtype,
    )
    for step in range(int(config.flow_steps)):
        time = torch.full(
            (attempts,),
            (step + 0.5) / max(1, int(config.flow_steps)),
            device=device,
            dtype=latent.dtype,
        )
        latent = latent + fragment_model.transport_velocity(
            latent, time, context
        ) / max(1, int(config.flow_steps))

    vocabulary = torch.from_numpy(target_endpoints).to(device=device, dtype=latent.dtype)
    distances = ((latent[:, None, :] - vocabulary[None, :, :]) ** 2).mean(dim=-1)
    energies = token_energy(
        energy_model,
        vocabulary,
        context,
        chunk_size=int(config.energy_chunk_size),
    )
    energy_mean = energies.mean(dim=-1, keepdim=True)
    energy_scale = energies.std(dim=-1, keepdim=True).clamp_min(
        float(config.energy_scale_floor)
    )
    standardized_energy = (energies - energy_mean) / energy_scale
    distance_offset = distances - distances.min(dim=-1, keepdim=True).values
    logits = -distance_offset / max(float(config.distance_temperature), 1e-6)
    logits = logits + float(config.energy_weight) * standardized_energy
    for attempt, site_index in enumerate(selected_sites.tolist()):
        current_fragment = sites[int(site_index)].variable
        for token_index, token in enumerate(target_fragments):
            if token == current_fragment:
                logits[attempt, token_index] = -torch.inf
    probabilities = torch.softmax(logits.float(), dim=-1)
    selected_tokens = torch.multinomial(
        probabilities, 1, replacement=True, generator=generator
    ).squeeze(-1)
    token_entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    site_entropy = -(
        site_probabilities * site_probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    output: list[dict[str, object]] = []
    for attempt, (site_index, token_index) in enumerate(
        zip(selected_sites.tolist(), selected_tokens.tolist())
    ):
        site = sites[int(site_index)]
        target_fragment = target_fragments[int(token_index)]
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, target_fragment),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": target_fragment,
                "latent_norm": float(latent[attempt].norm().cpu()),
                "quantization_distance": float(distances[attempt, token_index].cpu()),
                "site_entropy": float(site_entropy[attempt].cpu()),
                "token_energy": float(energies[attempt, token_index].cpu()),
                "token_probability": float(probabilities[attempt, token_index].cpu()),
                "token_distribution_entropy": float(token_entropy[attempt].cpu()),
            }
        )
    return output


def evaluate_tilted(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: b27.LatentPropertyEnergy,
    pairs: Sequence[object],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frozen: list[tuple[object, list[dict[str, object]]]] = []
    for pair_index, pair in enumerate(pairs):
        generated = tilted_actions(
            fragment_model,
            energy_model,
            pair,
            source_latents[pair_index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=int(config.seed) * 100000 + pair_index,
        )
        if len(generated) != int(config.num_attempts):
            raise RuntimeError("B28 did not freeze exactly 20 token samples")
        frozen.append((pair, generated))

    # Targets and property oracles are accessed only after all latent-token
    # samples and their single assembled molecules have been frozen.
    rows: list[dict[str, object]] = []
    from rdkit import Chem

    for pair_index, (pair, generated) in enumerate(frozen):
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"internal_dev_{pair_index:04d}"
        )
        source_copy_target = graph.morgan_tanimoto(
            pair.source_smiles, pair.target_smiles
        ) or 0.0
        for rank, raw in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(str(raw["smiles"] or ""))
            valid = bool(canonical)
            molecule = Chem.MolFromSmiles(canonical) if valid else None
            source_similarity = (
                graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            )
            target_similarity = (
                graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            )
            fraction, _, evaluated, property_success = (
                unified.instruction_success_and_distance(
                    pair.row, canonical or "", task_specs=specs
                )
            )
            similarity_success = bool(
                source_similarity is not None and source_similarity >= 0.4
            )
            rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "site_core": raw["site_core"],
                    "source_fragment": raw["source_fragment"],
                    "target_fragment_token": raw["target_fragment_token"],
                    "latent_norm": raw["latent_norm"],
                    "quantization_distance": raw["quantization_distance"],
                    "site_entropy": raw["site_entropy"],
                    "token_energy": raw["token_energy"],
                    "token_probability": raw["token_probability"],
                    "token_distribution_entropy": raw[
                        "token_distribution_entropy"
                    ],
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(molecule.GetNumAtoms()) if molecule else 0,
                    "valid": valid,
                    "source_tanimoto": float(source_similarity or 0.0),
                    "target_tanimoto": float(target_similarity or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(property_success),
                    "strict_success": bool(property_success and similarity_success),
                    "source_similarity_success": similarity_success,
                }
            )
    metrics = base.summarize_candidates(rows, int(config.num_attempts))
    for name in (
        "token_energy",
        "token_probability",
        "token_distribution_entropy",
        "quantization_distance",
    ):
        values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        metrics[f"mean_{name}"] = float(np.mean(values)) if values else 0.0
    return rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B28 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    representation, _representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(
            args.fragment_checkpoint, device, preregistration
        )
    )
    energy_model, energy_summary = load_energy(
        args.energy_checkpoint, args.energy_summary, preregistration, device
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
    ):
        if frozen_manifest.get(key) != belief.file_sha256(path):
            raise ValueError(f"Frozen B24 input drift: {key}")

    train_pairs, reconstruction = b27.reconstruct_b24_train_pairs(
        args, preregistration
    )
    action_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
    )
    pairs, _actions, coverage = b27.covered_pairs_and_actions(
        train_pairs, action_config
    )
    fit_pairs, dev_pairs = b27.grouped_fit_dev_split(
        pairs,
        seed=int(preregistration["energy_split_seed"]),
        dev_fraction=float(preregistration["energy_dev_fraction"]),
    )
    generation_pairs = b27.generation_subset(
        pairs,
        dev_pairs,
        limit=int(preregistration["internal_dev_generation_limit"]),
        seed=int(preregistration["internal_dev_generation_seed"]),
    )
    source_latents = kernel.encode_sources(
        representation,
        generation_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        num_attempts=int(preregistration["num_attempts"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        distance_temperature=float(preregistration["distance_temperature"]),
        energy_weight=float(preregistration["energy_weight"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        seed=int(preregistration["internal_dev_generation_seed"]),
    )
    baseline_rows, baseline = kernel.evaluate(
        fragment_model,
        generation_pairs,
        source_latents,
        target_fragments,
        target_endpoints,
        config,
        device,
    )
    tilted_rows, tilted = evaluate_tilted(
        fragment_model,
        energy_model,
        generation_pairs,
        source_latents,
        target_fragments,
        target_endpoints,
        config,
        device,
    )
    baseline_two = float(
        baseline["by_property_count"].get("2", {}).get("strict_any20", 0.0)
    )
    baseline_three = float(
        baseline["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    tilted_two = float(
        tilted["by_property_count"].get("2", {}).get("strict_any20", 0.0)
    )
    tilted_three = float(
        tilted["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    gates = dict(preregistration["gates"])
    checks = {
        "exact_attempts": {"value": tilted["attempted_per_condition"], "threshold": 20},
        "validity": {"value": tilted["validity"], "threshold": gates["validity"]},
        "strict_any20": {
            "value": tilted["strict_any20"],
            "threshold": gates["strict_any20"],
        },
        "strict_any20_delta": {
            "value": tilted["strict_any20"] - baseline["strict_any20"],
            "threshold": gates["strict_any20_delta"],
        },
        "three_property_strict_any20": {
            "value": tilted_three,
            "threshold": gates["three_property_strict_any20"],
        },
        "three_property_strict_delta": {
            "value": tilted_three - baseline_three,
            "threshold": gates["three_property_strict_delta"],
        },
        "two_property_strict_delta": {
            "value": tilted_two - baseline_two,
            "threshold": gates["two_property_strict_delta"],
        },
        "mean_unique_valid": {
            "value": tilted["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": tilted["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "fit_internal_dev_source_overlap": {
            "value": b27.overlap_audit(pairs, fit_pairs, dev_pairs)[
                "fit_internal_dev_source_overlap"
            ],
            "threshold": 0,
        },
    }
    exact_checks = {"exact_attempts", "fit_internal_dev_source_overlap"}
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact_checks
            else item["value"] < item["threshold"]
        )
    ]
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "energy_checkpoint_sha256": belief.file_sha256(args.energy_checkpoint),
        "energy_summary_sha256": belief.file_sha256(args.energy_summary),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "representation_protocol": representation_summary.get("protocol"),
        "b26_heldout_access": False,
        "official_test_access": False,
        "model_training": False,
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "post_freeze_internal_dev_oracle_access": True,
        "latent_token_distribution": True,
        "one_latent_one_sampled_token_one_raw_molecule": True,
        "molecular_candidate_materialization": False,
        "molecular_candidate_ranking": False,
        "selector": False,
        "finalizer": False,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "generation_internal_dev_pairs": len(generation_pairs),
        "property_count_breakdown": {
            str(count): sum(pair.property_count == count for pair in generation_pairs)
            for count in (2, 3)
        },
        "reconstruction": reconstruction,
        "coverage": coverage,
        "energy_calibration": energy_summary.get("calibration"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_candidate_rows(args.output_dir / "baseline_candidates.csv", baseline_rows)
    base.write_candidate_rows(args.output_dir / "tilted_candidates.csv", tilted_rows)
    summary = {
        "protocol": PROTOCOL,
        "manifest": run_manifest,
        "baseline_evaluation": baseline,
        "tilted_evaluation": tilted,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": (
            "advance_energy_tilted_latent_decoder_to_cross_task_transfer"
            if not failures
            else "freeze_fragment_latent_line_as_insufficient_for_robust_3p_control"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
