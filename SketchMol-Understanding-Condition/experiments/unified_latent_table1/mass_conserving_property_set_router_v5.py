#!/usr/bin/env python3
"""Train one arm of a mass-conserving Common-LLM property-set router.

V4 solved false-positive support on the two-property graph replay, but its
categorical 0--7 cardinality classifier under-counted unseen 3--7-property
instructions.  V5 removes that classifier.  Every named property emits an
inclusion probability; their sum is a differentiable set cardinality and its
rounded value constrains a deterministic exact-top-k set at inference.

The primary probe is freshly sampled and signature-disjoint from every V4
composition fit/probe instruction.  The old graph probe is retained only as a
regression check.  A valid arm always exits zero; a separate job applies the
preregistered science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import structured_sparse_property_router_v4 as v4


PROTOCOL = "train_only_mass_conserving_property_set_router_v5"
ARMS = v4.ARMS
V4_ROUTING_METRICS = v4.routing_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = v4.build_parser()
    parser.description = __doc__
    parser.add_argument("--v4-manifest", required=True, type=Path)
    parser.add_argument("--v4-gate-summary", required=True, type=Path)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "mechanism": "mass_conserving_inclusion_energy_exact_topk_set_router",
        "categorical_cardinality_head": False,
        "support_threshold_search": False,
        "fresh_probe_excludes_v4_fit_and_probe": True,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "official_test_access": False,
        "single_seed": True,
        "arms": list(ARMS),
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Mass-conserving router preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Mass-conserving router implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def composition_spec_lineage(
    property_columns: Sequence[str],
    excluded_pairs: set[tuple[str, str]],
    preregistration: Mapping[str, object],
) -> tuple[
    list[list[tuple[str, int]]],
    list[list[tuple[str, int]]],
    list[list[tuple[str, int]]],
]:
    base_specs = v4.base_fit_specs(property_columns, excluded_pairs)
    higher_specs = v4.sample_specs(
        property_columns,
        excluded_pairs,
        range(3, int(preregistration["max_instruction_cardinality"]) + 1),
        int(preregistration["higher_cardinality_fit_specs_per_k"]),
        int(preregistration["v4_composition_fit_seed"]),
    )
    fit_specs = base_specs + higher_specs
    fit_signatures = {v4.specs_signature(specs) for specs in fit_specs}
    old_probe_specs = v4.sample_specs(
        property_columns,
        excluded_pairs,
        range(3, int(preregistration["max_instruction_cardinality"]) + 1),
        int(preregistration["v4_probe_specs_per_k"]),
        int(preregistration["v4_composition_probe_seed"]),
        fit_signatures,
    )
    forbidden_property_sets = {
        tuple(sorted(str(name) for name, _direction in specs))
        for specs in [*fit_specs, *old_probe_specs]
    }
    fresh_probe_specs = sample_property_set_disjoint_specs(
        property_columns,
        excluded_pairs,
        range(3, int(preregistration["max_instruction_cardinality"]) + 1),
        int(preregistration["fresh_probe_specs_per_k"]),
        int(preregistration["fresh_composition_probe_seed"]),
        forbidden_property_sets,
    )
    return fit_specs, old_probe_specs, fresh_probe_specs


def sample_property_set_disjoint_specs(
    property_columns: Sequence[str],
    excluded_pairs: set[tuple[str, str]],
    cardinalities: Sequence[int],
    samples_per_cardinality: int,
    seed: int,
    forbidden_property_sets: set[tuple[str, ...]],
) -> list[list[tuple[str, int]]]:
    rng = random.Random(int(seed))
    seen = set(forbidden_property_sets)
    output: list[list[tuple[str, int]]] = []
    for cardinality in cardinalities:
        accepted = 0
        attempts = 0
        while accepted < int(samples_per_cardinality):
            attempts += 1
            if attempts > int(samples_per_cardinality) * 10000:
                raise ValueError(
                    f"Cannot sample fresh property-set cardinality {cardinality}"
                )
            names = rng.sample(list(map(str, property_columns)), int(cardinality))
            property_set = tuple(sorted(names))
            specs = [(name, rng.choice((-1, 1))) for name in names]
            if property_set in seen or v4.contains_excluded_pair(specs, excluded_pairs):
                continue
            seen.add(property_set)
            output.append(specs)
            accepted += 1
    return output


def training_and_fresh_probe_examples(
    arm: str,
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    excluded_pairs: set[tuple[str, str]],
    preregistration: Mapping[str, object],
) -> tuple[list[v4.v3.TextExample], list[v4.v3.TextExample], list[v4.v3.TextExample]]:
    fit_specs, _old_probe_specs, fresh_probe_specs = composition_spec_lineage(
        property_columns, excluded_pairs, preregistration
    )
    unique_specs = (
        [specs for specs in fit_specs if len(specs) == 1]
        if arm == "no_composition"
        else fit_specs
    )
    unique_examples = v4.examples_from_specs(
        unique_specs,
        property_columns,
        property_names,
        int(preregistration["scramble_seed"]),
        f"fit_v5_{arm}",
    )
    if arm == "no_composition":
        target_count = int(preregistration["no_composition_fit_examples"])
        fit_examples = [
            v4.v3.TextExample(
                text=unique_examples[index % len(unique_examples)].text,
                target=unique_examples[index % len(unique_examples)].target,
                phrases=unique_examples[index % len(unique_examples)].phrases,
                key=(
                    f"{unique_examples[index % len(unique_examples)].key}"
                    f"_exposure_{index:04d}"
                ),
            )
            for index in range(target_count)
        ]
    else:
        fit_examples = unique_examples
    first = [
        v4.v3.make_example(
            specs,
            property_columns,
            property_names,
            "probe_raise_lower",
            f"fresh_multi_probe_{index:04d}_raise_lower",
        )
        for index, specs in enumerate(fresh_probe_specs)
    ]
    second = [
        v4.v3.make_example(
            specs,
            property_columns,
            property_names,
            "probe_more_less",
            f"fresh_multi_probe_{index:04d}_more_less",
        )
        for index, specs in enumerate(fresh_probe_specs)
    ]
    return fit_examples, first, second


class MassConservingPropertySetRouter(v4.StructuredSparsePropertyRouter):
    """Slot-level inclusion energies whose probability mass is the set size."""

    def __init__(
        self,
        llm_hidden_dim: int,
        slot_dim: int,
        property_count: int,
        max_cardinality: int,
        use_token_slots: bool,
    ) -> None:
        super().__init__(
            llm_hidden_dim,
            slot_dim,
            property_count,
            max_cardinality,
            use_token_slots,
        )
        del self.cardinality

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = attention_mask.bool()
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / denominator
        if self.use_token_slots:
            keys = self.key(hidden.float())
            values = self.value(hidden.float())
            attention_logits = (
                torch.einsum("pd,bld->bpl", self.queries, keys) / self.scale
            )
            attention_logits = attention_logits.masked_fill(
                ~mask[:, None, :], torch.finfo(attention_logits.dtype).min
            )
            attention = torch.softmax(attention_logits, dim=-1)
            slot_states = torch.einsum("bpl,bld->bpd", attention, values)
        else:
            shared = self.pooled_value(pooled)
            slot_states = shared[:, None, :] + self.property_embedding[None, :, :]
            attention = mask[:, None, :].float() / denominator[:, None, :]
            attention = attention.expand(-1, self.property_count, -1)
        direction_logits = (
            torch.einsum("bpd,psd->bps", slot_states, self.direction_weights)
            + self.direction_bias
        )
        probabilities = torch.sigmoid(direction_logits)
        raw_coefficients = probabilities[..., 1] - probabilities[..., 0]
        support_logits = (
            torch.einsum("bpd,pd->bp", slot_states, self.support_weights)
            + self.support_bias
        )
        soft_cardinality = torch.sigmoid(support_logits).sum(dim=-1)
        return (
            raw_coefficients,
            direction_logits,
            support_logits,
            soft_cardinality,
            attention,
        )


def exact_mass_conserving_support(
    support_logits: torch.Tensor, soft_cardinality: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cardinality = soft_cardinality.round().long().clamp(
        min=0, max=min(7, int(support_logits.shape[1]))
    )
    support = torch.zeros_like(support_logits, dtype=torch.bool)
    for row, count in enumerate(cardinality.tolist()):
        if count:
            indices = torch.topk(support_logits[row], k=int(count), dim=-1).indices
            support[row, indices] = True
    return support, cardinality


def support_separation_loss(
    support_logits: torch.Tensor, active: torch.Tensor, margin: float
) -> torch.Tensor:
    rows = active.any(dim=-1) & (~active).any(dim=-1)
    if not bool(rows.any()):
        return torch.zeros((), device=support_logits.device)
    positive_floor = support_logits.masked_fill(~active, math.inf).min(dim=-1).values
    negative_ceiling = support_logits.masked_fill(active, -math.inf).max(dim=-1).values
    return F.relu(float(margin) - positive_floor[rows] + negative_ceiling[rows]).mean()


def train_router(
    llm_model: object,
    router: MassConservingPropertySetRouter,
    tokenized: Mapping[str, torch.Tensor],
    arm: str,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    lora_parameters = [
        parameter for parameter in llm_model.parameters() if parameter.requires_grad
    ]
    if arm != "no_lora" and not lora_parameters:
        raise ValueError(f"Arm {arm} expected trainable LoRA parameters")
    if arm == "no_lora" and lora_parameters:
        raise ValueError("no_lora arm unexpectedly has trainable LLM parameters")
    groups = [
        {
            "params": list(router.parameters()),
            "lr": float(preregistration["router_learning_rate"]),
        }
    ]
    if lora_parameters:
        groups.insert(
            0,
            {
                "params": lora_parameters,
                "lr": float(preregistration["lora_learning_rate"]),
            },
        )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(preregistration["weight_decay"])
    )
    parameters = lora_parameters + list(router.parameters())
    batch_size = int(preregistration["training_batch_size"])
    row_count = int(tokenized["input_ids"].shape[0])
    history: list[dict[str, float]] = []
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = list(range(row_count))
        random.Random(int(preregistration["training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        llm_model.train(bool(lora_parameters))
        router.train()
        for start in range(0, row_count, batch_size):
            indices = order[start : start + batch_size]
            input_ids = tokenized["input_ids"][indices].to(device)
            attention_mask = tokenized["attention_mask"][indices].to(device)
            targets = tokenized["targets"][indices].to(device)
            span_targets = tokenized["span_targets"][indices].to(device)
            span_active = tokenized["span_active"][indices].to(device)
            (
                raw,
                direction_logits,
                support_logits,
                soft_cardinality,
                slot_attention,
            ) = v4.model_forward(
                llm_model, router, input_ids, attention_mask, device
            )
            active = targets.ne(0)
            coefficient_loss = (
                (raw[active] - targets[active]).square().mean()
                if bool(active.any())
                else torch.zeros((), device=device)
            )
            inactive_loss = (
                raw[~active].square().mean()
                if bool((~active).any())
                else torch.zeros((), device=device)
            )
            direction_loss = F.binary_cross_entropy_with_logits(
                direction_logits,
                v4.v3.direction_labels(targets),
                pos_weight=torch.full(
                    (2,),
                    float(preregistration["positive_direction_weight"]),
                    device=device,
                ),
            )
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits,
                active.float(),
                pos_weight=torch.tensor(
                    float(preregistration["positive_support_weight"]), device=device
                ),
            )
            cardinality_target = active.sum(dim=-1).float()
            mass_loss = F.smooth_l1_loss(
                soft_cardinality,
                cardinality_target,
                beta=float(preregistration["cardinality_huber_beta"]),
            )
            separation_loss = support_separation_loss(
                support_logits,
                active,
                float(preregistration["support_separation_margin"]),
            )
            if router.use_token_slots and bool(span_active.any()):
                token_log_attention = slot_attention.clamp_min(1e-9).log()
                attention_rows = -(span_targets * token_log_attention).sum(dim=-1)
                attention_loss = attention_rows[span_active].mean()
            else:
                attention_loss = torch.zeros((), device=device)
            loss = (
                float(preregistration["coefficient_loss_weight"]) * coefficient_loss
                + float(preregistration["inactive_loss_weight"]) * inactive_loss
                + float(preregistration["direction_loss_weight"]) * direction_loss
                + float(preregistration["support_loss_weight"]) * support_loss
                + float(preregistration["cardinality_mass_loss_weight"]) * mass_loss
                + float(preregistration["support_separation_loss_weight"])
                * separation_loss
                + float(preregistration["attention_loss_weight"]) * attention_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(preregistration["grad_clip"]))
            optimizer.step()
            for name, value in {
                "loss": loss,
                "coefficient_loss": coefficient_loss,
                "inactive_loss": inactive_loss,
                "direction_loss": direction_loss,
                "support_loss": support_loss,
                "cardinality_mass_loss": mass_loss,
                "support_separation_loss": separation_loss,
                "attention_loss": attention_loss,
                "mean_soft_cardinality": soft_cardinality.mean(),
            }.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite mass-router metrics: {row}")
        history.append(row)
        print(
            json.dumps(
                {"stage": "mass_conserving_set_router_epoch", "arm": arm, **row},
                sort_keys=True,
            ),
            flush=True,
        )
    llm_model.eval()
    router.eval()
    return history


@torch.no_grad()
def predict_examples(
    llm_model: object,
    router: MassConservingPropertySetRouter,
    tokenizer: object,
    examples: Sequence[v4.v3.TextExample],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized = v4.v3.tokenize_examples(
        tokenizer,
        examples,
        int(preregistration["property_count"]),
        int(preregistration["llm_max_length"]),
    )
    coefficient_rows = []
    support_rows = []
    cardinality_rows = []
    batch_size = int(preregistration["probe_batch_size"])
    for start in range(0, len(examples), batch_size):
        raw, _directions, support_logits, soft_cardinality, _attention = (
            v4.model_forward(
                llm_model,
                router,
                tokenized["input_ids"][start : start + batch_size].to(device),
                tokenized["attention_mask"][start : start + batch_size].to(device),
                device,
            )
        )
        support, cardinality = exact_mass_conserving_support(
            support_logits, soft_cardinality
        )
        coefficient_rows.append(
            torch.where(support, raw, torch.zeros_like(raw)).cpu()
        )
        support_rows.append(support.cpu())
        cardinality_rows.append(cardinality.cpu())
    return (
        torch.cat(coefficient_rows, dim=0),
        torch.cat(support_rows, dim=0),
        torch.cat(cardinality_rows, dim=0),
    )


def routing_metrics(
    coefficients: torch.Tensor,
    support: torch.Tensor,
    cardinality: torch.Tensor,
    examples: Sequence[v4.v3.TextExample],
) -> dict[str, float]:
    metrics = V4_ROUTING_METRICS(coefficients, support, cardinality, examples)
    targets = torch.stack([example.target for example in examples])
    target_support = targets.ne(0)
    signed_support = support.eq(target_support).all(dim=-1) & (
        (torch.sign(coefficients) == torch.sign(targets)) | ~target_support
    ).all(dim=-1)
    metrics["exact_signed_support_rate"] = float(signed_support.float().mean())
    return metrics


def lineage_audit(
    preregistration: Mapping[str, object]
) -> dict[str, object]:
    property_columns = [str(name) for name in v4.unified.PROPERTY_COLUMNS]
    excluded_pairs = {
        tuple(sorted(map(str, row)))
        for row in preregistration["heldout_property_pairs"]
    }
    fit, old_probe, fresh_probe = composition_spec_lineage(
        property_columns, excluded_pairs, preregistration
    )
    fit_signatures = {v4.specs_signature(specs) for specs in fit}
    old_signatures = {v4.specs_signature(specs) for specs in old_probe}
    fresh_signatures = {v4.specs_signature(specs) for specs in fresh_probe}
    fit_property_sets = {
        tuple(sorted(str(name) for name, _direction in specs)) for specs in fit
    }
    old_property_sets = {
        tuple(sorted(str(name) for name, _direction in specs)) for specs in old_probe
    }
    fresh_property_sets = {
        tuple(sorted(str(name) for name, _direction in specs)) for specs in fresh_probe
    }
    return {
        "v4_fit_specs": len(fit_signatures),
        "v4_probe_specs": len(old_signatures),
        "v5_fresh_probe_specs": len(fresh_signatures),
        "fresh_v4_fit_overlap": len(fresh_signatures & fit_signatures),
        "fresh_v4_probe_overlap": len(fresh_signatures & old_signatures),
        "fresh_v4_fit_property_set_overlap": len(
            fresh_property_sets & fit_property_sets
        ),
        "fresh_v4_probe_property_set_overlap": len(
            fresh_property_sets & old_property_sets
        ),
        "fresh_probe_sha256": hashlib.sha256(
            json.dumps(
                sorted([list(signature) for signature in fresh_signatures]),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def common_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--protocol-manifest",
        str(args.protocol_manifest),
        "--arm",
        str(args.arm),
        "--v3-summary",
        str(args.v3_summary),
        "--repair-summary",
        str(args.repair_summary),
        "--prepare-summary",
        str(args.prepare_summary),
        "--fit-probe-bundle",
        str(args.fit_probe_bundle),
        "--representation-checkpoint",
        str(args.representation_checkpoint),
        "--representation-summary",
        str(args.representation_summary),
        "--canonical-checkpoint",
        str(args.canonical_checkpoint),
        "--sft-adapter-dir",
        str(args.sft_adapter_dir),
        "--e1-manifest",
        str(args.e1_manifest),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    locked = dict(preregistration["locked_inputs"])
    premise_paths = {
        "v4_implementation_sha256": Path(v4.__file__).resolve(),
        "v4_manifest_sha256": args.v4_manifest,
        "v4_gate_summary_sha256": args.v4_gate_summary,
    }
    premise_hashes = {name: file_sha256(path) for name, path in premise_paths.items()}
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in premise_hashes.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V5 premise drift: {drift}")
    v4_gate = read_json(args.v4_gate_summary)
    if (
        v4_gate.get("decision") != "stop_before_molecule_generation"
        or dict(v4_gate["science_gate"])["passed"] is not False
    ):
        raise ValueError("V5 requires the locked negative V4 science gate")
    failed = {
        name
        for name, passed in dict(dict(v4_gate["science_gate"])["checks"]).items()
        if not bool(passed)
    }
    expected_failures = {
        "full_multicardinality_exact_rate",
        "full_multicardinality_support_rate",
        "token_slot_ablation_delta",
    }
    if failed != expected_failures:
        raise ValueError(f"V5 scientific premise failure-set drift: {sorted(failed)}")

    v4.PROTOCOL = PROTOCOL
    v4.read_preregistration = read_preregistration
    v4.StructuredSparsePropertyRouter = MassConservingPropertySetRouter
    v4.training_and_multicardinality_probe_examples = (
        training_and_fresh_probe_examples
    )
    v4.train_router = train_router
    v4.predict_examples = predict_examples
    v4.routing_metrics = routing_metrics
    result = v4.main(common_argv(args))
    if result != 0:
        return int(result)
    summary_path = args.output_dir / args.arm / "summary.json"
    summary = read_json(summary_path)
    summary["stage"] = "mass_conserving_property_set_router_arm_execution"
    summary["mechanisms"].update(
        {
            "categorical_cardinality_head": False,
            "mass_conserving_cardinality": True,
            "rounded_inclusion_mass_exact_topk": True,
        }
    )
    summary["fresh_probe_lineage"] = lineage_audit(preregistration)
    summary["probe_roles"] = {
        "multicardinality_probe": "fresh_primary_science_gate",
        "graph_probe_routing": "reused_v4_development_regression_only",
        "graph_probe_tokens": "reused_v4_development_regression_only",
        "graph_probe_flow": "reused_v4_development_regression_only",
    }
    summary["artifacts"].update(premise_hashes)
    summary["contract"].update(
        {
            "categorical_cardinality_head": False,
            "mass_conserving_cardinality": True,
            "fresh_probe_excludes_v4_fit_and_probe": True,
            "reused_graph_probe_is_regression_only": True,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "v5_arm_summary_augmented",
                "arm": args.arm,
                "fresh_probe_lineage": summary["fresh_probe_lineage"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
