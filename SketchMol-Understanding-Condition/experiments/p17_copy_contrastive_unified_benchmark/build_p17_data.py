#!/usr/bin/env python3
"""Build P17 continuation data and two honest expanded development views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import p17_protocol as protocol


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def raw_id(row: Mapping[str, object]) -> str:
    for key in ("condition_id", "sample_id", "pair_id", "example_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return hashlib.sha256(json.dumps(dict(row), sort_keys=True).encode()).hexdigest()[:20]


def convert(row: Mapping[str, object], declared_mode: str) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    if mode != declared_mode:
        raise ValueError(f"mode mismatch: expected={declared_mode} actual={mode}")
    target = protocol.canonical_smiles(
        row.get("target_smiles", "") or row.get("policy_target_smiles", "")
    )
    if not target or (mode == "edit" and target == source):
        raise ValueError("missing, invalid, or identity target")
    chosen = protocol.response(target, mode)
    rejected = protocol.response_from_source(source) if mode == "edit" else ""
    return {
        "example_id": f"p17:{mode}:{raw_id(row)}",
        "task_mode": mode,
        "condition_hash": protocol.condition_hash(row),
        "condition_family_hash": protocol.condition_family_hash(row),
        "source_hash": protocol.source_hash(source),
        "target_hash": hashlib.sha256(target.encode()).hexdigest(),
        "source_smiles": source,
        "target_smiles": target,
        "messages": [*messages, {"role": "assistant", "content": chosen}],
        "rejected_assistant": rejected,
        "pairwise_enabled": mode == "edit",
    }


def clean(rows: Sequence[Mapping[str, object]], mode: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        try:
            item = convert(row, mode)
        except (ValueError, AssertionError):
            continue
        key = (str(item["condition_hash"]), str(item["source_hash"]), str(item["target_hash"]))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def stable_rank(row: Mapping[str, object], seed: int) -> str:
    material = f"{seed}:{row['condition_hash']}:{row['source_hash']}:{row['target_hash']}"
    return hashlib.sha256(material.encode()).hexdigest()


def p16_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, set[str]]:
    result = {key: set() for key in ("condition", "family", "source", "target")}
    for row in rows:
        result["condition"].add(str(row.get("condition_hash", "")))
        source = str(row.get("source_hash", ""))
        if source:
            result["source"].add(source)
        target = protocol.canonical_smiles(row.get("target_smiles", ""))
        if target:
            result["target"].add(hashlib.sha256(target.encode()).hexdigest())
        try:
            user_payload = json.loads(row["messages"][1]["content"])
            family = sorted(
                (str(item["property"]), item["goal"] if isinstance(item["goal"], str) else "around")
                for item in user_payload["conditions"]
            )
            result["family"].add(hashlib.sha256(json.dumps(family, separators=(",", ":")).encode()).hexdigest())
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return result


def choose_dev(
    rows: Sequence[dict[str, object]], prior: Mapping[str, set[str]], per_view: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eligible = [
        row for row in rows
        if str(row["target_hash"]) not in prior["target"]
        and (not row["source_hash"] or str(row["source_hash"]) not in prior["source"])
    ]
    ranked = sorted(eligible, key=lambda row: stable_rank(row, seed))
    # ID means a coarse condition family was observed by P16. Prefer exact condition
    # matches but do not require them; numeric de-novo goals are almost always unique.
    id_pool = [row for row in ranked if str(row["condition_family_hash"]) in prior["family"]]
    id_pool.sort(key=lambda row: (str(row["condition_hash"]) not in prior["condition"], stable_rank(row, seed)))
    id_rows = id_pool[:per_view]
    used = {str(row["example_id"]) for row in id_rows}
    ood_pool = [
        row for row in ranked
        if row["example_id"] not in used and str(row["condition_hash"]) not in prior["condition"]
    ]
    # Maximize strict family OOD first, then fill the explicitly reported exact-condition
    # OOD view if the small source corpus cannot supply enough unseen families.
    ood_pool.sort(key=lambda row: (str(row["condition_family_hash"]) in prior["family"], stable_rank(row, seed + 13)))
    return id_rows, ood_pool[:per_view]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--p16-train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-per-mode", type=int, default=192)
    parser.add_argument("--dev-per-view-mode", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    prior = p16_audit(read_jsonl(args.p16_train_jsonl))
    pools = {
        "de_novo": clean(read_csv(args.denovo_csv), "de_novo"),
        "edit": clean(read_csv(args.edit_csv), "edit"),
    }
    id_rows: list[dict[str, object]] = []
    ood_rows: list[dict[str, object]] = []
    selected_by_mode: dict[str, dict[str, list[dict[str, object]]]] = {}
    for offset, mode in enumerate(("de_novo", "edit")):
        mode_id, mode_ood = choose_dev(pools[mode], prior, args.dev_per_view_mode, args.seed + offset)
        if len(mode_id) < args.dev_per_view_mode or len(mode_ood) < args.dev_per_view_mode:
            raise ValueError(
                f"insufficient expanded {mode} dev: id={len(mode_id)} ood={len(mode_ood)}; "
                "do not silently relax the preregistered view"
            )
        selected_by_mode[mode] = {"id": mode_id, "ood": mode_ood}
        id_rows.extend(mode_id)
        ood_rows.extend(mode_ood)

    held_sources = {str(row["source_hash"]) for row in [*id_rows, *ood_rows] if row["source_hash"]}
    held_targets = {str(row["target_hash"]) for row in [*id_rows, *ood_rows]}
    ood_conditions = {str(row["condition_hash"]) for row in ood_rows}
    strict_ood_families = {
        str(row["condition_family_hash"]) for row in ood_rows
        if str(row["condition_family_hash"]) not in prior["family"]
    }
    train_by_mode: dict[str, list[dict[str, object]]] = {}
    for offset, mode in enumerate(("de_novo", "edit")):
        candidates = [
            row for row in pools[mode]
            if str(row["target_hash"]) not in held_targets
            and (not row["source_hash"] or str(row["source_hash"]) not in held_sources)
            and str(row["condition_hash"]) not in ood_conditions
            and str(row["condition_family_hash"]) not in strict_ood_families
        ]
        candidates.sort(key=lambda row: stable_rank(row, args.seed + 100 + offset))
        train_by_mode[mode] = candidates[: args.train_per_mode]
        if len(train_by_mode[mode]) < args.train_per_mode:
            raise ValueError(f"insufficient {mode} train rows after isolation: {len(train_by_mode[mode])}")

    mixed: list[dict[str, object]] = []
    for left, right in zip(train_by_mode["de_novo"], train_by_mode["edit"]):
        mixed.extend((left, right))
    random.Random(args.seed).shuffle(mixed)
    write_jsonl(args.output_dir / "train.paired.jsonl", mixed)
    write_jsonl(args.output_dir / "dev.id_condition_source_isolated.jsonl", id_rows)
    write_jsonl(args.output_dir / "dev.condition_source_ood.jsonl", ood_rows)

    train_conditions = prior["condition"] | {str(row["condition_hash"]) for row in mixed}
    train_families = prior["family"] | {str(row["condition_family_hash"]) for row in mixed}
    train_sources = prior["source"] | {str(row["source_hash"]) for row in mixed if row["source_hash"]}
    train_targets = prior["target"] | {str(row["target_hash"]) for row in mixed}
    manifest = {
        "protocol": protocol.PROTOCOL,
        "seed": args.seed,
        "locked_inputs": {
            "denovo_train": str(args.denovo_csv),
            "edit_train": str(args.edit_csv),
            "p16_train_manifest": str(args.p16_train_jsonl),
            "table1_pilot_future": "outputs/p6_unified_transition_policy_v1/seed_7/data/edit_table1_gate.csv",
            "denovo_pilot_future": "outputs/p6_unified_transition_policy_v1/seed_7/data/denovo_hard_gate.csv"
        },
        "rows": {
            "train_total": len(mixed), "train_de_novo": len(train_by_mode["de_novo"]),
            "train_edit": len(train_by_mode["edit"]), "id_dev_total": len(id_rows),
            "ood_dev_total": len(ood_rows),
        },
        "dev_per_view_mode": args.dev_per_view_mode,
        "id_view": {
            "condition_family_overlap_by_design": True,
            "exact_condition_overlap": sum(str(row["condition_hash"]) in train_conditions for row in id_rows),
            "source_overlap": sum(bool(row["source_hash"]) and str(row["source_hash"]) in train_sources for row in id_rows),
            "target_overlap": sum(str(row["target_hash"]) in train_targets for row in id_rows),
        },
        "ood_view": {
            "exact_condition_overlap": sum(str(row["condition_hash"]) in train_conditions for row in ood_rows),
            "strict_family_ood_rows": sum(str(row["condition_family_hash"]) not in train_families for row in ood_rows),
            "source_overlap": sum(bool(row["source_hash"]) and str(row["source_hash"]) in train_sources for row in ood_rows),
            "target_overlap": sum(str(row["target_hash"]) in train_targets for row in ood_rows),
            "note": "strict family-OOD count is explicit; the remainder are unseen exact numeric/directional condition identities in known families"
        },
        "pairwise_edit_rows": sum(bool(row["pairwise_enabled"]) for row in mixed),
        "de_novo_rehearsal_rows": sum(row["task_mode"] == "de_novo" for row in mixed),
        "rejected_is_source_copy_only": True,
        "prompt_target_fields": False,
        "mode_router": False,
        "static_candidate_pool": False,
        "property_reranking": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if manifest["id_view"]["source_overlap"] or manifest["id_view"]["target_overlap"]:
        raise SystemExit("P17 ID dev leakage audit failed")
    if manifest["ood_view"]["exact_condition_overlap"] or manifest["ood_view"]["source_overlap"] or manifest["ood_view"]["target_overlap"]:
        raise SystemExit("P17 OOD dev leakage audit failed")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
