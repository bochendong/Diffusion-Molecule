#!/usr/bin/env python3
"""Precompute reusable MolEdit-Instruct features in shardable stages.

The raw Hugging Face file is intentionally small in schema but large in row
count. This script keeps each expensive operation resumable:

1. normalize-pairs: split raw rows into deterministic pair shards and SMILES lists.
2. molecule-cache: compute RDKit features once per unique raw SMILES.
3. pair-features: join pair shards to the molecule cache and compute edit features.
4. finalize: build train/eval/hard/smoke manifests from enhanced pair shards.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping


RAW_FIELDS = ["example_id", "instruction", "source_smiles", "target_smiles"]

PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")

DELTA_THRESHOLDS = {
    "MW": 35.0,
    "LogP": 0.5,
    "QED": 0.05,
    "TPSA": 15.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}

CACHE_FIELDS = [
    "raw_smiles",
    "valid",
    "canonical_smiles",
    "scaffold_smiles",
    "fingerprint_bits",
    *PROPERTY_COLUMNS,
]

BASE_PAIR_FIELDS = [
    *RAW_FIELDS,
    "pair_hash",
    "instruction_tasks",
    "instruction_task_properties",
    "instruction_task_directions",
]

ENHANCED_FIELDS = [
    *BASE_PAIR_FIELDS,
    "source_valid",
    "target_valid",
    "source_canonical_smiles",
    "target_canonical_smiles",
    "source_scaffold_smiles",
    "target_scaffold_smiles",
    "scaffold_match",
    "source_target_tanimoto",
    "difficulty_bucket",
    "pair_quality",
    "computed_active_properties",
    "computed_active_count",
]

for _prop in PROPERTY_COLUMNS:
    ENHANCED_FIELDS.extend(
        [
            f"source_{_prop}",
            f"target_{_prop}",
            f"delta_{_prop}",
            f"{_prop}_active",
            f"{_prop}_direction",
        ]
    )

PROPERTY_KEYWORDS = {
    "MW": ("molecular weight", "molecular mass", "mol weight", "weight"),
    "LogP": (
        "logp",
        "lipophilicity",
        "lipophilic",
        "hydrophobicity",
        "hydrophobic",
        "fat solubility",
        "water solubility",
        "hydrophilicity",
        "hydrophilic",
    ),
    "QED": ("qed", "drug-likeness", "drug likeness", "druglike", "drug-like"),
    "TPSA": ("tpsa", "polar surface area", "topological polar surface area"),
    "HBA": (
        "hydrogen bond acceptor",
        "hydrogen-bond acceptor",
        "h-bond acceptor",
        "acceptor count",
    ),
    "HBD": (
        "hydrogen bond donor",
        "hydrogen-bond donor",
        "h-bond donor",
        "donor count",
    ),
    "RB": ("rotatable bond", "rotatable bonds", "rotor", "flexibility"),
    "SA": (
        "synthetic accessibility",
        "synthesis accessibility",
        "synthesizability",
        "synthesise",
        "synthesize",
        "synthesis",
    ),
    "DRD2": ("drd2",),
    "JNK3": ("jnk3",),
    "GSK3B": ("gsk3b", "gsk3 beta", "gsk3beta", "gsk3-beta", "gsk3"),
}

INCREASE_WORDS = (
    "increase",
    "increased",
    "increasing",
    "improve",
    "improved",
    "improving",
    "enhance",
    "enhanced",
    "more",
    "higher",
    "raise",
    "boost",
    "add",
)

DECREASE_WORDS = (
    "decrease",
    "decreased",
    "decreasing",
    "reduce",
    "reduced",
    "reducing",
    "lower",
    "lowered",
    "less",
    "fewer",
    "remove",
    "minimize",
    "minimise",
)


def stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def stable_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def shard_tag(index: int, count: int) -> str:
    return f"{index:05d}_of_{count:05d}"


def ensure_shard_args(num_shards: int, shard_index: int | None) -> int:
    if num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    if shard_index is None:
        raise SystemExit("--shard-index is required for this stage")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num_shards")
    return shard_index


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_raw_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected 4 tab-separated fields, got {len(parts)}"
                )
            yield dict(zip(RAW_FIELDS, parts))


def normalize_instruction(text: str) -> str:
    lowered = text.lower().replace("β", " beta ")
    return re.sub(r"\s+", " ", lowered).strip()


def phrase_present(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def infer_direction(normalized_instruction: str, prop: str) -> str:
    increase = any(phrase_present(normalized_instruction, word) for word in INCREASE_WORDS)
    decrease = any(phrase_present(normalized_instruction, word) for word in DECREASE_WORDS)

    if prop == "LogP":
        water = "water solubility" in normalized_instruction or "hydrophilic" in normalized_instruction
        fat = (
            "fat solubility" in normalized_instruction
            or "lipophilic" in normalized_instruction
            or "hydrophobic" in normalized_instruction
        )
        if water and increase and not decrease:
            return "decrease"
        if water and decrease and not increase:
            return "increase"
        if fat and increase and not decrease:
            return "increase"
        if fat and decrease and not increase:
            return "decrease"

    if prop == "SA":
        easier = (
            "easier to synthesize" in normalized_instruction
            or "easy to synthesize" in normalized_instruction
            or "more synthesizable" in normalized_instruction
            or "improve" in normalized_instruction
        )
        if easier and not decrease:
            return "decrease"

    if increase and not decrease:
        return "increase"
    if decrease and not increase:
        return "decrease"
    return "unknown"


def infer_instruction_tasks(instruction: str) -> list[dict[str, str]]:
    normalized = normalize_instruction(instruction)
    tasks = []
    for prop, keywords in PROPERTY_KEYWORDS.items():
        if any(phrase_present(normalized, keyword) for keyword in keywords):
            tasks.append({"property": prop, "direction": infer_direction(normalized, prop)})
    return tasks


def pair_hash(row: Mapping[str, str]) -> str:
    key = "\t".join(
        [
            row.get("example_id", ""),
            row.get("instruction", ""),
            row.get("source_smiles", ""),
            row.get("target_smiles", ""),
        ]
    )
    return stable_hex(key)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_normalize_pairs(args: argparse.Namespace) -> None:
    shard_index = ensure_shard_args(args.num_shards, args.shard_index)
    output_dir = args.output_dir
    pair_path = output_dir / "pairs" / f"pairs_shard_{shard_tag(shard_index, args.num_shards)}.csv"
    smiles_path = output_dir / "smiles" / f"smiles_shard_{shard_tag(shard_index, args.num_shards)}.txt"
    summary_path = output_dir / "summaries" / f"normalize_pairs_{shard_tag(shard_index, args.num_shards)}.json"

    pair_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    output_rows = 0
    unique_smiles: set[str] = set()
    with pair_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=BASE_PAIR_FIELDS)
        writer.writeheader()

        for row in parse_raw_rows(args.input):
            input_rows += 1
            if stable_hash(row["example_id"]) % args.num_shards != shard_index:
                continue
            tasks = infer_instruction_tasks(row["instruction"])
            properties = sorted({task["property"] for task in tasks})
            directions = {
                task["property"]: task["direction"]
                for task in tasks
                if task["direction"] != "unknown"
            }
            out = {
                **row,
                "pair_hash": pair_hash(row),
                "instruction_tasks": json_dumps(tasks),
                "instruction_task_properties": "|".join(properties),
                "instruction_task_directions": json_dumps(directions),
            }
            writer.writerow(out)
            output_rows += 1
            unique_smiles.add(row["source_smiles"])
            unique_smiles.add(row["target_smiles"])
            if args.limit is not None and output_rows >= args.limit:
                break

    with smiles_path.open("w", encoding="utf-8") as smiles_handle:
        for smiles in sorted(unique_smiles):
            smiles_handle.write(smiles + "\n")

    write_json(
        summary_path,
        {
            "stage": "normalize-pairs",
            "input": str(args.input),
            "pair_path": str(pair_path),
            "smiles_path": str(smiles_path),
            "num_shards": args.num_shards,
            "shard_index": shard_index,
            "input_rows_seen": input_rows,
            "output_rows": output_rows,
            "unique_smiles": len(unique_smiles),
            "limit": args.limit,
        },
    )
    print(f"wrote {output_rows} pair rows to {pair_path}")
    print(f"wrote {len(unique_smiles)} unique SMILES to {smiles_path}")


def import_rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required for --stage molecule-cache. "
            "Install rdkit in the online/cluster Python environment."
        ) from exc
    RDLogger.DisableLog("rdApp.warning")
    return Chem, AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, MurckoScaffold


def morgan_on_bits(mol: object, *, radius: int, n_bits: int, fallback_all_chem: object) -> list[int]:
    try:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    except Exception:
        fp = fallback_all_chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return sorted(int(bit) for bit in fp.GetOnBits())


def iter_smiles_inputs(output_dir: Path) -> Iterable[Path]:
    return sorted((output_dir / "smiles").glob("smiles_shard_*_of_*.txt"))


def stage_molecule_cache(args: argparse.Namespace) -> None:
    shard_index = ensure_shard_args(args.num_shards, args.shard_index)
    Chem, AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, MurckoScaffold = import_rdkit()

    output_dir = args.output_dir
    cache_path = output_dir / "molecule_cache" / f"molecule_cache_shard_{shard_tag(shard_index, args.num_shards)}.csv"
    summary_path = output_dir / "summaries" / f"molecule_cache_{shard_tag(shard_index, args.num_shards)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_inputs = [args.smiles_input] if args.smiles_input else list(iter_smiles_inputs(output_dir))
    if not smiles_inputs:
        raise SystemExit(f"no SMILES shard files found under {output_dir / 'smiles'}")

    assigned_smiles: set[str] = set()
    for smiles_input in smiles_inputs:
        with smiles_input.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                smiles = raw_line.rstrip("\n")
                if not smiles:
                    continue
                if stable_hash(smiles) % args.num_shards == shard_index:
                    assigned_smiles.add(smiles)
                    if args.limit is not None and len(assigned_smiles) >= args.limit:
                        break
        if args.limit is not None and len(assigned_smiles) >= args.limit:
            break

    valid_count = 0
    invalid_count = 0
    with cache_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for smiles in sorted(assigned_smiles):
            row = {field: "" for field in CACHE_FIELDS}
            row["raw_smiles"] = smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                row["valid"] = "0"
                invalid_count += 1
                writer.writerow(row)
                continue

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = ""
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

            row.update(
                {
                    "valid": "1",
                    "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                    "scaffold_smiles": scaffold_smiles,
                    "fingerprint_bits": " ".join(
                        str(bit)
                        for bit in morgan_on_bits(
                            mol,
                            radius=2,
                            n_bits=args.fingerprint_bits,
                            fallback_all_chem=AllChem,
                        )
                    ),
                    "MW": f"{float(Descriptors.MolWt(mol)):.8g}",
                    "LogP": f"{float(Crippen.MolLogP(mol)):.8g}",
                    "QED": f"{float(QED.qed(mol)):.8g}",
                    "TPSA": f"{float(rdMolDescriptors.CalcTPSA(mol)):.8g}",
                    "HBD": f"{float(Lipinski.NumHDonors(mol)):.8g}",
                    "HBA": f"{float(Lipinski.NumHAcceptors(mol)):.8g}",
                    "RB": f"{float(Lipinski.NumRotatableBonds(mol)):.8g}",
                }
            )
            valid_count += 1
            writer.writerow(row)

    write_json(
        summary_path,
        {
            "stage": "molecule-cache",
            "cache_path": str(cache_path),
            "smiles_inputs": [str(path) for path in smiles_inputs],
            "num_shards": args.num_shards,
            "shard_index": shard_index,
            "fingerprint_bits": args.fingerprint_bits,
            "assigned_smiles": len(assigned_smiles),
            "valid": valid_count,
            "invalid": invalid_count,
            "limit": args.limit,
        },
    )
    print(f"wrote {len(assigned_smiles)} molecule rows to {cache_path}")


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bit_set(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(part) for part in value.split() if part}


def tanimoto_from_bits(bits_a: str | None, bits_b: str | None) -> float | None:
    set_a = parse_bit_set(bits_a)
    set_b = parse_bit_set(bits_b)
    if not set_a and not set_b:
        return None
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def source_similarity_bin(value: float | None) -> str:
    if value is None:
        return "invalid_or_missing"
    if value >= 0.7:
        return "high_similarity"
    if value >= 0.5:
        return "medium_similarity"
    if value >= 0.4:
        return "hard_similarity"
    return "too_distant"


def pair_quality_tier(value: float | None, *, same_scaffold: bool) -> str:
    if value is None:
        return "invalid_or_missing"
    if same_scaffold and value >= 0.5:
        return "same_scaffold_medium_plus"
    if same_scaffold and value >= 0.4:
        return "same_scaffold_hard"
    if value >= 0.6:
        return "cross_scaffold_high_similarity"
    if value >= 0.5:
        return "cross_scaffold_medium_similarity"
    if value >= 0.4:
        return "cross_scaffold_hard_similarity"
    return "rejected_too_distant"


def load_needed_cache(output_dir: Path, needed_smiles: set[str]) -> dict[str, dict[str, str]]:
    cache: dict[str, dict[str, str]] = {}
    cache_paths = sorted((output_dir / "molecule_cache").glob("molecule_cache_shard_*_of_*.csv"))
    if not cache_paths:
        raise SystemExit(f"no molecule cache shards found under {output_dir / 'molecule_cache'}")

    for cache_path in cache_paths:
        with cache_path.open("r", encoding="utf-8", newline="") as csv_handle:
            reader = csv.DictReader(csv_handle)
            for row in reader:
                smiles = row.get("raw_smiles", "")
                if smiles in needed_smiles:
                    cache[smiles] = row
        if len(cache) == len(needed_smiles):
            break
    return cache


def blank_cache_row(smiles: str) -> dict[str, str]:
    row = {field: "" for field in CACHE_FIELDS}
    row["raw_smiles"] = smiles
    row["valid"] = "0"
    return row


def stage_pair_features(args: argparse.Namespace) -> None:
    shard_index = ensure_shard_args(args.num_shards, args.shard_index)
    output_dir = args.output_dir
    pair_path = args.pair_input or output_dir / "pairs" / f"pairs_shard_{shard_tag(shard_index, args.num_shards)}.csv"
    enhanced_path = (
        output_dir
        / "enhanced_pairs"
        / f"enhanced_pairs_shard_{shard_tag(shard_index, args.num_shards)}.csv"
    )
    summary_path = output_dir / "summaries" / f"pair_features_{shard_tag(shard_index, args.num_shards)}.json"
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    needed_smiles: set[str] = set()
    with pair_path.open("r", encoding="utf-8", newline="") as csv_handle:
        reader = csv.DictReader(csv_handle)
        for row in reader:
            pair_rows.append(row)
            needed_smiles.add(row["source_smiles"])
            needed_smiles.add(row["target_smiles"])
            if args.limit is not None and len(pair_rows) >= args.limit:
                break

    cache = load_needed_cache(output_dir, needed_smiles)
    missing_cache = len(needed_smiles) - len(cache)
    bucket_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    active_counts: dict[str, int] = {}

    with enhanced_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=ENHANCED_FIELDS)
        writer.writeheader()
        for pair in pair_rows:
            source = cache.get(pair["source_smiles"]) or blank_cache_row(pair["source_smiles"])
            target = cache.get(pair["target_smiles"]) or blank_cache_row(pair["target_smiles"])
            tanimoto = tanimoto_from_bits(source.get("fingerprint_bits"), target.get("fingerprint_bits"))
            source_scaffold = source.get("scaffold_smiles", "")
            target_scaffold = target.get("scaffold_smiles", "")
            same_scaffold = bool(source_scaffold and source_scaffold == target_scaffold)
            bucket = source_similarity_bin(tanimoto)
            quality = pair_quality_tier(tanimoto, same_scaffold=same_scaffold)

            out = {field: "" for field in ENHANCED_FIELDS}
            for field in BASE_PAIR_FIELDS:
                out[field] = pair.get(field, "")
            out.update(
                {
                    "source_valid": source.get("valid", "0"),
                    "target_valid": target.get("valid", "0"),
                    "source_canonical_smiles": source.get("canonical_smiles", ""),
                    "target_canonical_smiles": target.get("canonical_smiles", ""),
                    "source_scaffold_smiles": source_scaffold,
                    "target_scaffold_smiles": target_scaffold,
                    "scaffold_match": "1" if same_scaffold else "0",
                    "source_target_tanimoto": "" if tanimoto is None else f"{tanimoto:.6f}",
                    "difficulty_bucket": bucket,
                    "pair_quality": quality,
                }
            )

            active_props = []
            for prop in PROPERTY_COLUMNS:
                source_value = parse_float(source.get(prop))
                target_value = parse_float(target.get(prop))
                if source_value is not None:
                    out[f"source_{prop}"] = f"{source_value:.8g}"
                if target_value is not None:
                    out[f"target_{prop}"] = f"{target_value:.8g}"
                if source_value is None or target_value is None:
                    continue
                delta = target_value - source_value
                direction = "increase" if delta >= 0 else "decrease"
                is_active = abs(delta) >= DELTA_THRESHOLDS[prop]
                out[f"delta_{prop}"] = f"{delta:.8g}"
                out[f"{prop}_active"] = "1" if is_active else "0"
                out[f"{prop}_direction"] = direction
                if is_active:
                    active_props.append(prop)

            out["computed_active_properties"] = "|".join(active_props)
            out["computed_active_count"] = str(len(active_props))
            writer.writerow(out)

            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            active_counts[str(len(active_props))] = active_counts.get(str(len(active_props)), 0) + 1

    write_json(
        summary_path,
        {
            "stage": "pair-features",
            "pair_path": str(pair_path),
            "enhanced_path": str(enhanced_path),
            "num_shards": args.num_shards,
            "shard_index": shard_index,
            "pair_rows": len(pair_rows),
            "needed_smiles": len(needed_smiles),
            "cache_hits": len(cache),
            "missing_cache": missing_cache,
            "difficulty_buckets": bucket_counts,
            "pair_quality": quality_counts,
            "active_property_counts": active_counts,
            "limit": args.limit,
        },
    )
    print(f"wrote {len(pair_rows)} enhanced pair rows to {enhanced_path}")


def iter_enhanced_paths(output_dir: Path) -> list[Path]:
    paths = sorted((output_dir / "enhanced_pairs").glob("enhanced_pairs_shard_*_of_*.csv"))
    if not paths:
        raise SystemExit(f"no enhanced pair shards found under {output_dir / 'enhanced_pairs'}")
    return paths


def eval_bucket_key(row: Mapping[str, str]) -> str:
    props = row.get("instruction_task_properties") or row.get("computed_active_properties") or "unknown"
    count = row.get("computed_active_count") or "unknown"
    return f"{row.get('difficulty_bucket', 'unknown')}|{props}|active_count={count}"


def row_id(row: Mapping[str, str]) -> str:
    return row.get("pair_hash") or row.get("example_id") or json_dumps(dict(row))


def stage_finalize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    eval_path = split_dir / "eval_balanced.csv"
    hard_path = split_dir / "eval_hard.csv"
    smoke_path = split_dir / f"smoke_{args.smoke_limit}.jsonl"
    summary_path = split_dir / "summary.json"

    enhanced_paths = iter_enhanced_paths(output_dir)
    counts = {
        "input_rows": 0,
        "train_rows": 0,
        "eval_balanced_rows": 0,
        "eval_hard_rows": 0,
        "smoke_rows": 0,
    }
    bucket_counts: dict[str, int] = {}
    selected_eval_buckets: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}

    with ExitStack() as stack:
        train_handle = stack.enter_context(train_path.open("w", encoding="utf-8", newline=""))
        eval_handle = stack.enter_context(eval_path.open("w", encoding="utf-8", newline=""))
        hard_handle = stack.enter_context(hard_path.open("w", encoding="utf-8", newline=""))
        smoke_handle = stack.enter_context(smoke_path.open("w", encoding="utf-8"))

        train_writer = csv.DictWriter(train_handle, fieldnames=ENHANCED_FIELDS)
        eval_writer = csv.DictWriter(eval_handle, fieldnames=ENHANCED_FIELDS)
        hard_writer = csv.DictWriter(hard_handle, fieldnames=ENHANCED_FIELDS)
        for writer in (train_writer, eval_writer, hard_writer):
            writer.writeheader()

        for enhanced_path in enhanced_paths:
            with enhanced_path.open("r", encoding="utf-8", newline="") as csv_handle:
                reader = csv.DictReader(csv_handle)
                for row in reader:
                    counts["input_rows"] += 1
                    key = eval_bucket_key(row)
                    bucket_counts[key] = bucket_counts.get(key, 0) + 1
                    difficulty = row.get("difficulty_bucket", "unknown")
                    quality = row.get("pair_quality", "unknown")
                    difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1

                    if counts["smoke_rows"] < args.smoke_limit:
                        smoke_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts["smoke_rows"] += 1

                    selected_eval = (
                        counts["eval_balanced_rows"] < args.eval_max
                        and selected_eval_buckets.get(key, 0) < args.eval_per_bucket
                    )
                    selected_hard = (
                        not selected_eval
                        and counts["eval_hard_rows"] < args.hard_max
                        and difficulty in {"hard_similarity", "too_distant"}
                    )

                    if selected_eval:
                        eval_writer.writerow({field: row.get(field, "") for field in ENHANCED_FIELDS})
                        selected_eval_buckets[key] = selected_eval_buckets.get(key, 0) + 1
                        counts["eval_balanced_rows"] += 1
                    elif selected_hard:
                        hard_writer.writerow({field: row.get(field, "") for field in ENHANCED_FIELDS})
                        counts["eval_hard_rows"] += 1
                    else:
                        train_writer.writerow({field: row.get(field, "") for field in ENHANCED_FIELDS})
                        counts["train_rows"] += 1

    write_json(
        summary_path,
        {
            "stage": "finalize",
            "output_dir": str(output_dir),
            "enhanced_paths": [str(path) for path in enhanced_paths],
            "paths": {
                "train": str(train_path),
                "eval_balanced": str(eval_path),
                "eval_hard": str(hard_path),
                "smoke": str(smoke_path),
            },
            "config": {
                "eval_max": args.eval_max,
                "eval_per_bucket": args.eval_per_bucket,
                "hard_max": args.hard_max,
                "smoke_limit": args.smoke_limit,
            },
            "counts": counts,
            "difficulty_buckets": difficulty_counts,
            "pair_quality": quality_counts,
            "eval_bucket_counts": bucket_counts,
            "selected_eval_buckets": selected_eval_buckets,
        },
    )
    print(f"wrote train split to {train_path}")
    print(f"wrote balanced eval split to {eval_path}")
    print(f"wrote hard eval split to {hard_path}")
    print(f"wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("normalize-pairs", "molecule-cache", "pair-features", "finalize"),
    )
    parser.add_argument("--input", type=Path, help="Raw MolEdit-Instruct txt file for normalize-pairs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smiles-input", type=Path, help="Optional SMILES list for molecule-cache.")
    parser.add_argument("--pair-input", type=Path, help="Optional pair CSV for pair-features.")
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--eval-max", type=int, default=50000)
    parser.add_argument("--eval-per-bucket", type=int, default=2000)
    parser.add_argument("--hard-max", type=int, default=50000)
    parser.add_argument("--smoke-limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(1024 * 1024 * 1024)
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.fingerprint_bits < 1:
        raise SystemExit("--fingerprint-bits must be positive")
    if args.stage == "normalize-pairs" and args.input is None:
        raise SystemExit("--input is required for --stage normalize-pairs")

    if args.stage == "normalize-pairs":
        stage_normalize_pairs(args)
    elif args.stage == "molecule-cache":
        stage_molecule_cache(args)
    elif args.stage == "pair-features":
        stage_pair_features(args)
    elif args.stage == "finalize":
        stage_finalize(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
