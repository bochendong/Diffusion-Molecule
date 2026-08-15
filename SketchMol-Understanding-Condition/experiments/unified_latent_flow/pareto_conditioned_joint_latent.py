#!/usr/bin/env python3
"""Freeze B31/B32 and stratify direct latent draws across structure preferences.

B31 places probability mass on assay-improving joint ``(site, fragment)``
states, while B32 adds a train-only structure energy.  B33 does not train or
rank another model.  It evaluates one frozen conditional latent distribution
at four preregistered structure-preference values and draws five states from
each value.  All 20 states are frozen before any product molecule is assembled.

The internal confirmation sources are attachment-site-eligible B24 train
sources that were not selected for either fitting or calibrating B31/B32.
Table1, B26 and official-test rows remain unread.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import structure_constrained_joint_latent as b32  # noqa: E402


b31 = b32.b31
b27 = b31.b27
kernel = b31.kernel
base = b31.base
belief = b31.belief
pinned = b31.pinned
PROTOCOL = "pareto_conditioned_joint_latent_v33"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-summary", type=Path, required=True)
    parser.add_argument("--b32-checkpoint", type=Path, required=True)
    parser.add_argument("--b32-summary", type=Path, required=True)
    parser.add_argument("--gsk3b-oracle", type=Path, required=True)
    parser.add_argument("--drd2-oracle", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "b31_protocol": b31.PROTOCOL,
        "b32_protocol": b32.PROTOCOL,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "evaluation_target_access": False,
        "model_training": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "exact_raw_attempts_per_condition": 20,
        "b32_train_source_limit": 512,
        "fresh_source_limit": 24,
        "structure_similarity_threshold": 0.65,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B33 preregistration drift: {drift}")
    if payload.get("tasks") != list(b31.TASK_SPECS):
        raise ValueError("B33 task order drift")
    if set(dict(payload.get("oracles", {}))) != {"GSK3B", "DRD2"}:
        raise ValueError("B33 pinned oracle contract drift")
    levels = list(payload.get("structure_preference_levels", []))
    expected_levels = [
        {"name": "property", "scale": 0.0, "attempts": 5},
        {"name": "balanced", "scale": 1.0, "attempts": 5},
        {"name": "structure", "scale": 1.5, "attempts": 5},
        {"name": "strict", "scale": 2.0, "attempts": 5},
    ]
    if levels != expected_levels:
        raise ValueError("B33 structure-preference schedule drift")
    if sum(int(level["attempts"]) for level in levels) != 20:
        raise ValueError("B33 structure-preference schedule is not exact n=20")
    return payload


def load_b32_structure_energy(
    checkpoint_path: Path,
    summary_path: Path,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[b27.LatentPropertyEnergy, dict[str, object]]:
    if belief.file_sha256(checkpoint_path) != preregistration["b32_checkpoint_sha256"]:
        raise ValueError("B33 frozen B32 checkpoint hash drift")
    if belief.file_sha256(summary_path) != preregistration["b32_summary_sha256"]:
        raise ValueError("B33 frozen B32 summary hash drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != b32.PROTOCOL:
        raise ValueError("B33 refuses a non-B32 structure energy")
    if summary.get("decision") != "stop_structure_constrained_joint_latent_after_single_pilot":
        raise ValueError("B33 B32 decision drift")
    gate = dict(summary.get("internal_gate", {}))
    if gate.get("passed") is not False:
        raise ValueError("B33 expects the preregistered B32 near-miss")
    if set(gate.get("failures", [])) != set(preregistration["b32_expected_failures"]):
        raise ValueError("B33 B32 failure signature drift")
    manifest = dict(summary.get("manifest", {}))
    contract = {
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "exact_raw_attempts_per_condition": 20,
    }
    drift = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in contract.items()
        if manifest.get(key) != expected
    }
    if drift:
        raise ValueError(f"B33 refuses B32 manifest drift: {drift}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("stage") != b32.PROTOCOL:
        raise ValueError("B33 frozen B32 checkpoint protocol drift")
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


def select_fresh_energy_sources(
    train_pairs: Sequence[object],
    preregistration: Mapping[str, object],
    site_config: SimpleNamespace,
) -> tuple[list[object], dict[str, object]]:
    b32_sources, b32_selection = b31.select_train_sources(
        train_pairs,
        limit=int(preregistration["b32_train_source_limit"]),
        seed=int(preregistration["b32_source_selection_seed"]),
        site_config=site_config,
    )
    excluded = {str(pair.source_smiles) for pair in b32_sources}
    by_source: dict[str, object] = {}
    for pair in train_pairs:
        by_source.setdefault(str(pair.source_smiles), pair)
    eligible = [
        pair
        for source, pair in by_source.items()
        if source not in excluded and kernel.source_sites(source, site_config)
    ]
    ordered = sorted(
        eligible,
        key=lambda pair: b27.stable_value(
            int(preregistration["fresh_source_selection_seed"]),
            "fresh-energy-source",
            pair.source_smiles,
        ),
    )
    limit = int(preregistration["fresh_source_limit"])
    if len(ordered) < limit:
        raise ValueError(f"B33 has only {len(ordered)}/{limit} fresh eligible sources")
    selected = ordered[:limit]
    selected_sources = {str(pair.source_smiles) for pair in selected}
    if selected_sources & excluded:
        raise AssertionError("B33 fresh sources overlap B31/B32 energy sources")
    return selected, {
        "reconstructed_unique_sources": len(by_source),
        "b31_b32_source_selection": b32_selection,
        "excluded_b31_b32_sources": len(excluded),
        "remaining_attachment_site_eligible_sources": len(eligible),
        "selected_fresh_sources": len(selected),
        "fresh_b31_b32_source_overlap": len(selected_sources & excluded),
    }


@torch.no_grad()
def pareto_conditioned_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    condition: b31.b29.TransferCondition,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
    *,
    seed: int,
) -> list[dict[str, object]]:
    sites, contexts_np, site_logits_np = b31.condition_site_state(
        fragment_model, condition, source_latent, config, device
    )
    contexts = torch.from_numpy(contexts_np).to(device)
    vocabulary = torch.from_numpy(target_endpoints).to(device)
    assay_margin, assay_logit = b32.grid_predictions(
        assay_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    structure_margin, structure_logit = b32.grid_predictions(
        structure_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    assay_energy = assay_margin + 0.25 * torch.tanh(assay_logit)
    standardized_assay = (assay_energy - assay_energy.mean()) / assay_energy.std().clamp_min(
        float(config.energy_scale_floor)
    )
    structure_shortfall = torch.relu(-structure_margin)
    feasibility_log_probability = F.logsigmoid(
        structure_logit / max(float(config.structure_logit_temperature), 1e-6)
    )
    site_logits = torch.from_numpy(site_logits_np).to(device)
    base_logits = site_logits[:, None] / max(float(config.site_temperature), 1e-6)
    base_logits = base_logits.expand_as(assay_energy).clone()
    base_logits = base_logits + float(config.assay_energy_weight) * standardized_assay
    token_lookup = {token: index for index, token in enumerate(target_fragments)}
    for site_index, site in enumerate(sites):
        current_index = token_lookup.get(site.variable)
        if current_index is not None:
            base_logits[site_index, current_index] = -torch.inf

    # Freeze all categorical states before constructing any product molecule.
    frozen_draws: list[tuple[dict[str, object], int, float, float]] = []
    for level_index, level in enumerate(config.structure_preference_levels):
        scale = float(level["scale"])
        logits = base_logits - scale * float(config.structure_dual_weight) * structure_shortfall
        logits = logits + scale * float(
            config.structure_feasibility_weight
        ) * feasibility_log_probability
        probabilities = torch.softmax(logits.reshape(-1).float(), dim=0)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 104729 * level_index)
        selected = torch.multinomial(
            probabilities,
            int(level["attempts"]),
            replacement=True,
            generator=generator,
        )
        entropy = float(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum().cpu()
        )
        for flat_index in selected.tolist():
            frozen_draws.append(
                (
                    dict(level),
                    int(flat_index),
                    float(probabilities[int(flat_index)].cpu()),
                    entropy,
                )
            )
    if len(frozen_draws) != int(config.num_attempts):
        raise RuntimeError("B33 did not freeze exactly 20 latent states")

    output: list[dict[str, object]] = []
    for level, flat_index, probability, entropy in frozen_draws:
        site_index, token_index = divmod(flat_index, len(target_fragments))
        site = sites[site_index]
        token = target_fragments[token_index]
        predicted_similarity = float(config.structure_similarity_threshold) + float(
            config.structure_similarity_scale
        ) * float(structure_margin[site_index, token_index].cpu())
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, token),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": token,
                "joint_energy": float(assay_energy[site_index, token_index].cpu()),
                "joint_probability": probability,
                "joint_distribution_entropy": entropy,
                "predicted_source_tanimoto": predicted_similarity,
                "predicted_structure_feasible": float(
                    torch.sigmoid(structure_logit[site_index, token_index]).cpu()
                ),
                "structure_preference_name": str(level["name"]),
                "structure_preference_scale": float(level["scale"]),
            }
        )
    return output


def freeze_methods(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    conditions: Sequence[b31.b29.TransferCondition],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> dict[str, list[tuple[b31.b29.TransferCondition, list[dict[str, object]]]]]:
    frozen = {"b31_property": [], "b32_structure": [], "b33_pareto": []}
    for index, condition in enumerate(conditions):
        seed = int(config.seed) * 100000 + index
        b31_actions = b31.joint_actions(
            fragment_model,
            assay_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
            energy_weight=float(config.assay_energy_weight),
        )
        b32_actions = b32.constrained_joint_actions(
            fragment_model,
            assay_model,
            structure_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        b33_actions = pareto_conditioned_actions(
            fragment_model,
            assay_model,
            structure_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        for name, actions in (
            ("b31_property", b31_actions),
            ("b32_structure", b32_actions),
            ("b33_pareto", b33_actions),
        ):
            if len(actions) != int(config.num_attempts):
                raise RuntimeError(f"B33 {name} did not freeze exactly 20 attempts")
            frozen[name].append((condition, actions))
    return frozen


def preference_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["structure_preference_name"])].append(row)
    output: dict[str, object] = {}
    for name, items in sorted(grouped.items()):
        valid = [row for row in items if bool(row["valid"])]
        output[name] = {
            "rows": len(items),
            "validity": len(valid) / max(1, len(items)),
            "success_t0_15_per_attempt": sum(
                bool(row["success_t0_15"]) for row in items
            )
            / max(1, len(items)),
            "success_t0_65_per_attempt": sum(
                bool(row["success_t0_65"]) for row in items
            )
            / max(1, len(items)),
            "mean_source_tanimoto": float(
                np.mean([float(row["source_tanimoto"]) for row in valid])
            )
            if valid
            else 0.0,
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B33 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))

    representation, _representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(args.fragment_checkpoint, device, preregistration)
    )
    assay_model, _b31_summary = b32.load_b31_energy(
        args.b31_checkpoint, args.b31_summary, preregistration, device
    )
    structure_model, b32_summary = load_b32_structure_energy(
        args.b32_checkpoint, args.b32_summary, preregistration, device
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
        (args.fragment_checkpoint, "fragment_checkpoint_sha256"),
    ):
        if dict(b32_summary["manifest"]).get(key) != belief.file_sha256(path):
            raise ValueError(f"B33 frozen B32 input drift: {key}")

    train_pairs, reconstruction = b27.reconstruct_b24_train_pairs(args, preregistration)
    action_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
    )
    fresh_pairs, fresh_selection = select_fresh_energy_sources(
        train_pairs, preregistration, action_config
    )
    conditions = b31.build_conditions(
        fresh_pairs, condition_dim=int(preregistration["condition_dim"])
    )
    source_latents_unique = kernel.encode_sources(
        representation,
        fresh_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    latent_lookup = {
        pair.source_smiles: source_latents_unique[index]
        for index, pair in enumerate(fresh_pairs)
    }
    source_latents = np.stack(
        [latent_lookup[row.source_smiles] for row in conditions]
    ).astype(np.float32)
    generation_config = SimpleNamespace(
        **vars(action_config),
        num_attempts=int(preregistration["exact_raw_attempts_per_condition"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        assay_energy_weight=float(preregistration["assay_energy_weight"]),
        structure_dual_weight=float(preregistration["structure_dual_weight"]),
        structure_feasibility_weight=float(
            preregistration["structure_feasibility_weight"]
        ),
        structure_logit_temperature=float(
            preregistration["structure_logit_temperature"]
        ),
        structure_similarity_threshold=float(
            preregistration["structure_similarity_threshold"]
        ),
        structure_similarity_scale=float(preregistration["structure_similarity_scale"]),
        structure_preference_levels=list(
            preregistration["structure_preference_levels"]
        ),
        seed=int(preregistration["generation_seed"]),
    )
    frozen = freeze_methods(
        fragment_model,
        assay_model,
        structure_model,
        conditions,
        source_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )

    # Evaluation oracles are loaded only after all three exact-n=20 outputs freeze.
    assay_oracles, oracle_provenance = pinned.load_pinned_oracles(
        gsk3b_path=args.gsk3b_oracle,
        drd2_path=args.drd2_oracle,
        specifications=dict(preregistration["oracles"]),
    )
    rows_by_method: dict[str, list[dict[str, object]]] = {}
    metrics_by_method: dict[str, dict[str, object]] = {}
    preference_schedule = [
        str(level["name"])
        for level in preregistration["structure_preference_levels"]
        for _ in range(int(level["attempts"]))
    ]
    for name, values in frozen.items():
        rows, metrics = b31.evaluate_frozen(values, assay_oracles, preregistration)
        if name == "b33_pareto":
            for row in rows:
                schedule_index = (int(row["attempt"]) - 1) % len(preference_schedule)
                preference_name = preference_schedule[schedule_index]
                level = next(
                    item
                    for item in preregistration["structure_preference_levels"]
                    if item["name"] == preference_name
                )
                row["structure_preference_name"] = preference_name
                row["structure_preference_scale"] = float(level["scale"])
        rows_by_method[name] = rows
        metrics_by_method[name] = metrics

    candidate = metrics_by_method["b33_pareto"]
    gates = dict(preregistration["gates"])
    checks: dict[str, dict[str, object]] = {
        "fresh_sources": {
            "value": fresh_selection["selected_fresh_sources"],
            "threshold": gates["minimum_fresh_sources"],
        },
        "fresh_b31_b32_source_overlap": {
            "value": fresh_selection["fresh_b31_b32_source_overlap"],
            "threshold": 0,
        },
        "conditions": {"value": candidate["conditions"], "threshold": 48},
        "exact_attempts": {
            "value": candidate["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": candidate["validity"], "threshold": gates["validity"]},
        "oracle_coverage": {
            "value": candidate["oracle_coverage"],
            "threshold": 1.0,
        },
        "mean_unique_valid": {
            "value": candidate["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": candidate["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "overall_any20_t0_15": {
            "value": candidate["acc_any20_t0_15"],
            "threshold": gates["overall_any20_t0_15"],
        },
        "overall_any20_t0_65": {
            "value": candidate["acc_any20_t0_65"],
            "threshold": gates["overall_any20_t0_65"],
        },
    }
    for task in b31.TASK_SPECS:
        task_metrics = dict(candidate["by_task"])[task]
        checks[f"{task}:any20_t0_15"] = {
            "value": task_metrics["acc_any20_t0_15"],
            "threshold": gates["minimum_task_any20_t0_15"],
        }
        checks[f"{task}:any20_t0_65"] = {
            "value": task_metrics["acc_any20_t0_65"],
            "threshold": gates["minimum_task_any20_t0_65"],
        }
    exact_checks = {
        "fresh_b31_b32_source_overlap",
        "conditions",
        "exact_attempts",
        "oracle_coverage",
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact_checks
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    passed = not failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_method.items():
        b31.write_rows(args.output_dir / f"fresh_internal_{name}_candidates.csv", rows)
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "b31_checkpoint_sha256": belief.file_sha256(args.b31_checkpoint),
        "b31_summary_sha256": belief.file_sha256(args.b31_summary),
        "b32_checkpoint_sha256": belief.file_sha256(args.b32_checkpoint),
        "b32_summary_sha256": belief.file_sha256(args.b32_summary),
        "representation_protocol": representation_summary.get("protocol"),
        "fragment_protocol": frozen_manifest.get("protocol", kernel.PROTOCOL),
        "reconstruction": reconstruction,
        "fresh_source_selection": fresh_selection,
        "b31_assay_energy_frozen": True,
        "b32_structure_energy_frozen": True,
        "model_training": False,
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "property_oracle_generation_access": False,
        "post_freeze_oracle_evaluation_access": True,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "structure_preference_levels": preregistration["structure_preference_levels"],
        "pinned_oracles": oracle_provenance,
    }
    summary = {
        "protocol": PROTOCOL,
        "manifest": run_manifest,
        "b32_decision": b32_summary.get("decision"),
        "fresh_internal": metrics_by_method,
        "preference_diagnostics": preference_diagnostics(
            rows_by_method["b33_pareto"]
        ),
        "gate": {"passed": passed, "checks": checks, "failures": failures},
        "decision": (
            "advance_frozen_pareto_latent_to_once_only_table1_subset"
            if passed
            else "stop_pareto_conditioned_joint_latent_after_single_fresh_pilot"
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
