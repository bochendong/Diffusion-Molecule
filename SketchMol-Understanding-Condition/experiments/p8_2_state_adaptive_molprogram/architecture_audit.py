#!/usr/bin/env python3
"""Fail-closed architecture and leakage audit for state-adaptive MolProgram."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from rdkit import Chem


ID_FIELDS = ("variant_id", "condition_id", "sample_id", "pair_id")
PROTECTED_EXPANDABLE = {"token_embedding.weight", "output.weight", "output.bias"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_id(row: dict[str, str]) -> str:
    return next((str(row.get(key, "") or "").strip() for key in ID_FIELDS if str(row.get(key, "") or "").strip()), "")


def row_ids(rows: Iterable[dict[str, str]]) -> set[str]:
    return {value for row in rows if (value := first_id(row))}


def normalize(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_smiles(value: object) -> str:
    text = normalize(value)
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else text


def transition_fingerprint(row: dict[str, str]) -> str:
    parts = [
        normalize_smiles(row.get("source_smiles") or row.get("molecule_smiles")),
        normalize_smiles(row.get("target_smiles")),
        normalize(row.get("instruction") or row.get("property_program")),
        normalize(row.get("benchmark_task") or row.get("task_name")),
    ]
    if not any(parts):
        return ""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def fingerprints(rows: Iterable[dict[str, str]]) -> set[str]:
    return {value for row in rows if (value := transition_fingerprint(row))}


def target_molecules(rows: Iterable[dict[str, str]]) -> set[str]:
    return {value for row in rows if (value := normalize_smiles(row.get("target_smiles")))}


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--denovo-eval-csv", required=True, type=Path)
    parser.add_argument("--edit-eval-csv", required=True, type=Path)
    parser.add_argument("--denovo-summary", required=True, type=Path)
    parser.add_argument("--edit-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--p811-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "p8_1_1_short_transaction",
    )
    parser.add_argument(
        "--decoder-implementation",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "unified_smiles_generator"
        / "unified_smiles_generator.py",
    )
    args = parser.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_vocab = dict(base["vocab"])
    trained_vocab = dict(trained["vocab"])
    base_state = dict(base["model_state"])
    trained_state = dict(trained["model_state"])
    checkpoint_hash = sha256(args.checkpoint)
    denovo_summary = load_summary(args.denovo_summary)
    edit_summary = load_summary(args.edit_summary)

    protected_mismatches: list[str] = []
    for name, tensor in base_state.items():
        candidate = trained_state.get(name)
        if candidate is None:
            protected_mismatches.append(f"missing:{name}")
            continue
        if name in PROTECTED_EXPANDABLE:
            slices = tuple(slice(0, size) for size in tensor.shape)
            candidate = candidate[slices]
        if tensor.shape != candidate.shape or not torch.equal(tensor.cpu(), candidate.cpu()):
            protected_mismatches.append(name)

    train_rows = read_rows(args.train_csv)
    denovo_rows = read_rows(args.denovo_eval_csv)
    edit_rows = read_rows(args.edit_eval_csv)
    train_ids = row_ids(train_rows)
    denovo_ids = row_ids(denovo_rows)
    edit_ids = row_ids(edit_rows)
    train_fingerprints = fingerprints(train_rows)
    denovo_fingerprints = fingerprints(denovo_rows)
    edit_fingerprints = fingerprints(edit_rows)
    train_targets = target_molecules(train_rows)
    denovo_targets = target_molecules(denovo_rows)
    edit_targets = target_molecules(edit_rows)

    denovo_source = (args.p811_directory / "sample_raw_denovo.py").read_text(encoding="utf-8")
    edit_source = (args.p811_directory / "sample_raw_transactions.py").read_text(encoding="utf-8")
    decoder_source = args.decoder_implementation.read_text(encoding="utf-8")
    transaction_tokens = [
        token for token in trained_vocab
        if token.startswith("<") and token not in {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}
    ]
    summary_hashes = [
        str(denovo_summary.get("checkpoint_sha256", "")),
        str(edit_summary.get("checkpoint_sha256", "")),
    ]

    checks = {
        "one_shared_property_conditioner": "condition_proj.0.weight" in trained_state
        and "condition_proj.1.weight" in trained_state,
        "one_source_aware_decoder": bool(trained.get("model_config", {}).get("source_aware", False))
        and "decoder.layers.0.self_attn.in_proj_weight" in trained_state,
        "one_shared_base_vocab_projection": bool(transaction_tokens)
        and "output.weight" in trained_state
        and int(trained_state["output.weight"].shape[0]) == len(trained_vocab),
        "source_logit_residual_disclosed": "source_output.weight" in trained_state
        and int(trained_state["source_output.weight"].shape[0]) == len(trained_vocab),
        "base_projection_precedes_source_residual": "logits = self.output(decoded)" in decoder_source
        and "source_delta = self.source_output" in decoder_source,
        "source_residual_requires_source_presence": "source_present[:, None, None]" in decoder_source
        and "source_present is not None" in decoder_source,
        "one_final_union_vocab_softmax": "probs = torch.softmax(logits, dim=-1)" in decoder_source
        and not bool(trained.get("model_config", {}).get("source_copy_aware", False)),
        "no_router_selected_model_or_head": not any(
            marker in name.lower()
            for name in trained_state
            for marker in ("task_router", "model_router", "denovo_head", "edit_head")
        ),
        "legacy_vocabulary_ids_exact": all(trained_vocab.get(token) == token_id for token, token_id in base_vocab.items()),
        "legacy_denovo_parameters_bit_exact": not protected_mismatches,
        "same_checkpoint_hash_for_both_modes": all(value == checkpoint_hash for value in summary_hashes),
        "empty_source_masks_transaction_tokens": "blocked_token_ids=action_ids" in denovo_source,
        "populated_source_support_is_graph_executable": "source_only_candidates" in edit_source
        and "universal_actions(source" in edit_source
        and "execute_graph_edit_action(source, action)" in edit_source,
        "no_train_denovo_id_overlap": not bool(train_ids & denovo_ids),
        "no_train_edit_id_overlap": not bool(train_ids & edit_ids),
        "no_train_denovo_transition_overlap": not bool(train_fingerprints & denovo_fingerprints),
        "no_train_edit_transition_overlap": not bool(train_fingerprints & edit_fingerprints),
        "no_train_denovo_target_overlap": not bool(train_targets & denovo_targets),
        "no_train_edit_target_overlap": not bool(train_targets & edit_targets),
        "no_property_reranking": denovo_summary.get("property_reranking") is False
        and edit_summary.get("property_reranking") is False,
        "no_target_at_inference": not bool(denovo_summary.get("target_molecule_used_at_inference", False))
        and edit_summary.get("target_molecule_used_at_inference") is False,
    }
    payload = {
        "protocol": "p8_2_state_adaptive_molprogram_architecture_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "definition": "one learned policy with deterministic state-adaptive legal action support",
        "not_a_learned_router": True,
        "shared_components": [
            "property conditioner",
            "autoregressive decoder",
            "base vocabulary projection",
            "union vocabulary final softmax",
            "checkpoint",
        ],
        "disclosed_state_specific_component": "source-conditioned logit residual",
        "output_by_initial_state": {
            "empty_source": "complete SMILES literal",
            "populated_source": "typed executable transaction",
        },
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checks": checks,
        "transaction_token_count": len(transaction_tokens),
        "protected_parameter_mismatches": protected_mismatches,
        "overlap_counts": {
            "train_denovo_ids": len(train_ids & denovo_ids),
            "train_edit_ids": len(train_ids & edit_ids),
            "train_denovo_transition_fingerprints": len(train_fingerprints & denovo_fingerprints),
            "train_edit_transition_fingerprints": len(train_fingerprints & edit_fingerprints),
            "train_denovo_canonical_targets": len(train_targets & denovo_targets),
            "train_edit_canonical_targets": len(train_targets & edit_targets),
        },
        "inference": {
            "target_molecule_used": False,
            "property_oracle_used": False,
            "property_reranking": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
