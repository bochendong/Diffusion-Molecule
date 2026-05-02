"""Build automatically verified instruction-editing datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chem import canonicalize_smiles, molecular_descriptors, passes_druglike_filters, tanimoto
from .instruction_schema import DEFAULT_THRESHOLDS, normalize_spec, spec_to_json
from .instruction_templates import generate_instruction_texts
from .instruction_verifier import preserves_scaffold, property_delta, score_verification, verify_instruction


def main() -> None:
    args = parse_args()
    rows = build_instruction_dataset(
        data_path=args.data,
        smiles_column=args.smiles_column,
        limit=args.limit,
        max_pairs=args.max_pairs,
        pairs_per_source=args.pairs_per_source,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        instructions_per_spec=args.instructions_per_spec,
        reference_pool_size=args.reference_pool_size,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if args.jsonl_out:
        jsonl_out = Path(args.jsonl_out)
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Built {len(df)} instruction rows from {df['pair_id'].nunique() if not df.empty else 0} verified pairs -> {out}")
    if not df.empty:
        print(df.groupby("split").size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a verified instruction-guided molecular editing dataset.")
    parser.add_argument("--data", required=True, help="Input molecule CSV with a SMILES column.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--out", default="data/instruction_editing.csv")
    parser.add_argument("--jsonl-out", default=None)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--pairs-per-source", type=int, default=12)
    parser.add_argument("--instructions-per-spec", type=int, default=5)
    parser.add_argument("--reference-pool-size", type=int, default=24)
    parser.add_argument("--min-similarity", type=float, default=0.6)
    parser.add_argument("--max-similarity", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_instruction_dataset(
    data_path: str | Path,
    smiles_column: str = "smiles",
    limit: int | None = 100000,
    max_pairs: int = 50000,
    pairs_per_source: int = 12,
    min_similarity: float = 0.6,
    max_similarity: float = 0.95,
    instructions_per_spec: int = 5,
    reference_pool_size: int = 24,
    seed: int = 7,
) -> list[dict[str, Any]]:
    records = _load_records(data_path, smiles_column=smiles_column, limit=limit)
    if len(records) < 2:
        raise ValueError("Need at least two valid molecules to build editing pairs.")

    rng = np.random.default_rng(seed)
    buckets = _bucket_records(records)
    rows: list[dict[str, Any]] = []
    pair_seen: set[str] = set()
    source_order = rng.permutation(len(records))
    for source_idx in source_order:
        if len(pair_seen) >= max_pairs:
            break
        candidates = _candidate_indices(source_idx, records, buckets, rng, pairs_per_source=pairs_per_source)
        for target_idx in candidates:
            if len(pair_seen) >= max_pairs:
                break
            if source_idx == target_idx:
                continue
            source = records[source_idx]
            target = records[target_idx]
            pair_key = _pair_id(source["smiles"], target["smiles"])
            if pair_key in pair_seen:
                continue
            sim = tanimoto(source["smiles"], target["smiles"])
            if sim < min_similarity or sim > max_similarity:
                continue
            spec = _spec_from_pair(source, target, sim, min_similarity)
            if spec is None:
                continue
            check = verify_instruction(source["smiles"], target["smiles"], spec)
            if not check["overall_success"]:
                continue
            reference_smiles, reference_role = _select_reference_smiles(
                source=source,
                target=target,
                records=records,
                candidate_indices=candidates,
                spec=spec,
                rng=rng,
                reference_pool_size=reference_pool_size,
            )
            pair_seen.add(pair_key)
            delta = property_delta(source["descriptors"], target["descriptors"])
            templates = generate_instruction_texts(spec, max_variants=instructions_per_spec)
            for template in templates:
                split = _split_for_key(pair_key, seed=seed)
                rows.append(
                    {
                        "pair_id": pair_key,
                        "source_smiles": source["smiles"],
                        "target_smiles": target["smiles"],
                        "reference_smiles": reference_smiles,
                        "reference_role": reference_role,
                        "instruction_text": template["instruction_text"],
                        "instruction_spec_json": spec_to_json(spec),
                        "property_delta_json": json.dumps(delta, sort_keys=True),
                        "edit_tags": "|".join(spec["goals"] + spec["constraints"] + spec["edits"]),
                        "template_id": template["template_id"],
                        "split": split,
                        "similarity_to_source": float(sim),
                    }
                )
    if not rows:
        raise ValueError("No verified instruction pairs were found. Try lowering --min-similarity or increasing --limit.")
    return rows


def _load_records(data_path: str | Path, smiles_column: str, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_csv(data_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Missing SMILES column '{smiles_column}' in {data_path}")
    if limit:
        df = df.head(limit)
    records = []
    seen = set()
    for raw in df[smiles_column].dropna().astype(str):
        can = canonicalize_smiles(raw)
        if can is None or can in seen:
            continue
        record = molecular_descriptors(can)
        if not record.valid:
            continue
        seen.add(record.smiles)
        records.append({"smiles": record.smiles, "descriptors": record.descriptors})
    return records


def _spec_from_pair(source: dict[str, Any], target: dict[str, Any], similarity: float, min_similarity: float) -> dict[str, Any] | None:
    source_desc = source["descriptors"]
    target_desc = target["descriptors"]
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds["similarity_min"] = float(min_similarity)
    goals = _goal_candidates(source_desc, target_desc)
    edits = _edit_candidates(source_desc, target_desc)
    constraints = ["keep_similarity"]
    if abs(target_desc["MW"] - source_desc["MW"]) <= thresholds["delta_mw_abs_max"]:
        constraints.append("keep_mw_similar")
    if preserves_scaffold(source["smiles"], target["smiles"], similarity_min=min_similarity):
        constraints.append("preserve_scaffold")
    if passes_druglike_filters(target_desc):
        constraints.append("keep_druglike")

    if not goals and not edits:
        return None
    if similarity < min_similarity:
        return None
    spec = normalize_spec({"goals": goals[:2], "constraints": constraints, "edits": edits[:2], "thresholds": thresholds})
    if not spec["goals"] and not spec["edits"]:
        return None
    return spec


def _goal_candidates(source: dict[str, float], target: dict[str, float]) -> list[str]:
    candidates = []
    delta = {key: float(target.get(key, 0.0) - source.get(key, 0.0)) for key in target}
    scored = [
        ("increase_logp", delta.get("LogP", 0.0), DEFAULT_THRESHOLDS["delta_logp_min"]),
        ("decrease_logp", -delta.get("LogP", 0.0), DEFAULT_THRESHOLDS["delta_logp_min"]),
        ("improve_qed", delta.get("QED", 0.0), DEFAULT_THRESHOLDS["delta_qed_min"]),
        ("reduce_tpsa", -delta.get("TPSA", 0.0), DEFAULT_THRESHOLDS["delta_tpsa_min"]),
        ("increase_tpsa", delta.get("TPSA", 0.0), DEFAULT_THRESHOLDS["delta_tpsa_min"]),
        ("increase_mw", delta.get("MW", 0.0), DEFAULT_THRESHOLDS["delta_mw_min"]),
        ("decrease_mw", -delta.get("MW", 0.0), DEFAULT_THRESHOLDS["delta_mw_min"]),
        ("increase_hba", delta.get("HBA", 0.0), DEFAULT_THRESHOLDS["delta_hba_min"]),
        ("increase_hbd", delta.get("HBD", 0.0), DEFAULT_THRESHOLDS["delta_hbd_min"]),
        ("decrease_rb", -delta.get("RB", 0.0), DEFAULT_THRESHOLDS["delta_rb_min"]),
        ("lower_sa", -delta.get("SA", 0.0), DEFAULT_THRESHOLDS["delta_sa_min"]),
    ]
    for name, value, required in sorted(scored, key=lambda item: item[1] / max(item[2], 1e-6), reverse=True):
        if value >= required:
            candidates.append(name)
    return candidates


def _edit_candidates(source: dict[str, float], target: dict[str, float]) -> list[str]:
    edits = []
    if target.get("fg_halogen", 0.0) - source.get("fg_halogen", 0.0) >= DEFAULT_THRESHOLDS["delta_halogen_min"]:
        edits.append("add_halogen")
    if source.get("fg_halogen", 0.0) - target.get("fg_halogen", 0.0) >= DEFAULT_THRESHOLDS["delta_halogen_min"]:
        edits.append("remove_halogen")
    source_hetero = source.get("N", 0.0) + source.get("O", 0.0) + source.get("S", 0.0)
    target_hetero = target.get("N", 0.0) + target.get("O", 0.0) + target.get("S", 0.0)
    if target_hetero - source_hetero >= DEFAULT_THRESHOLDS["delta_heteroatom_min"]:
        edits.append("add_heteroatom")
    if source_hetero - target_hetero >= DEFAULT_THRESHOLDS["delta_heteroatom_min"]:
        edits.append("reduce_heteroatom")
    for group, add_tag, remove_tag in [
        ("fg_ester", "add_ester", "remove_ester"),
        ("fg_amide", "add_amide", "remove_amide"),
        ("fg_amine", "add_amine", "remove_amine"),
        ("fg_alcohol", "add_alcohol", "remove_alcohol"),
    ]:
        diff = target.get(group, 0.0) - source.get(group, 0.0)
        if diff >= DEFAULT_THRESHOLDS["delta_fg_min"]:
            edits.append(add_tag)
        elif -diff >= DEFAULT_THRESHOLDS["delta_fg_min"]:
            edits.append(remove_tag)
    return edits


def _select_reference_smiles(
    source: dict[str, Any],
    target: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_indices: list[int],
    spec: dict[str, Any],
    rng: np.random.Generator,
    reference_pool_size: int,
) -> tuple[str, str]:
    pool = list(dict.fromkeys(candidate_indices))[:reference_pool_size]
    if len(pool) < reference_pool_size:
        extra = rng.choice(len(records), size=min(len(records), reference_pool_size * 2), replace=False).tolist()
        pool.extend(extra)
    best_smiles = source["smiles"]
    best_score = -1.0
    excluded = {source["smiles"], target["smiles"]}
    for idx in dict.fromkeys(pool):
        candidate = records[int(idx)]
        smiles = candidate["smiles"]
        if smiles in excluded:
            continue
        result = verify_instruction(source["smiles"], smiles, spec)
        if not result.get("valid"):
            continue
        score = score_verification(result) + tanimoto(smiles, target["smiles"])
        if result.get("overall_success"):
            score += 20.0
        if score > best_score:
            best_score = score
            best_smiles = smiles
    if best_smiles == source["smiles"]:
        return best_smiles, "source_fallback"
    return best_smiles, "verified_neighbor"


def _bucket_records(records: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[int]]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        desc = record["descriptors"]
        key = (int(desc.get("scaffold_class", 0)), int(desc.get("ring_count", 0)), int(desc.get("MW", 0.0) // 50))
        buckets[key].append(idx)
    return buckets


def _candidate_indices(
    source_idx: int,
    records: list[dict[str, Any]],
    buckets: dict[tuple[int, int, int], list[int]],
    rng: np.random.Generator,
    pairs_per_source: int,
) -> list[int]:
    desc = records[source_idx]["descriptors"]
    scaffold = int(desc.get("scaffold_class", 0))
    rings = int(desc.get("ring_count", 0))
    mw_bin = int(desc.get("MW", 0.0) // 50)
    pool = []
    for ring_offset in (-1, 0, 1):
        for mw_offset in (-1, 0, 1):
            pool.extend(buckets.get((scaffold, rings + ring_offset, mw_bin + mw_offset), []))
    if len(pool) < pairs_per_source * 4:
        pool.extend(rng.choice(len(records), size=min(len(records), pairs_per_source * 12), replace=False).tolist())
    pool = [idx for idx in dict.fromkeys(pool) if idx != source_idx]
    if not pool:
        return []
    sample_size = min(len(pool), max(pairs_per_source * 8, pairs_per_source))
    return rng.choice(pool, size=sample_size, replace=False).tolist()


def _pair_id(source_smiles: str, target_smiles: str) -> str:
    return hashlib.sha1(f"{source_smiles}>{target_smiles}".encode("utf-8")).hexdigest()[:16]


def _split_for_key(key: str, seed: int) -> str:
    value = int(hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest(), 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


if __name__ == "__main__":
    main()
