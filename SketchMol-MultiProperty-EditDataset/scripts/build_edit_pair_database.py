#!/usr/bin/env python
"""Mine multi-property source-conditioned edit pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from sketchmol_multiproperty_dataset.common import (
    PROPERTY_COLUMNS,
    active_property_deltas,
    direction_from_delta,
    json_dumps,
    pair_quality_tier,
    source_similarity_bin,
)
from sketchmol_understanding_condition.chem import canonical_smiles, morgan_tanimoto, render_molecule_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-db-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--max-pairs-per-scaffold", type=int, default=300)
    parser.add_argument("--max-pairs-per-source", type=int, default=8)
    parser.add_argument("--max-molecules-per-scaffold", type=int, default=300)
    parser.add_argument("--min-active-properties", type=int, default=2)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--max-similarity", type=float, default=0.9)
    parser.add_argument(
        "--pairing-strategy",
        choices=("scaffold_random", "source_neighbor"),
        default="scaffold_random",
        help="scaffold_random preserves the original behavior; source_neighbor prioritizes useful source-local edits.",
    )
    parser.add_argument(
        "--source-neighbor-min-tanimoto",
        type=float,
        default=0.4,
        help="Minimum source-target Tanimoto for source_neighbor pairs.",
    )
    parser.add_argument(
        "--source-neighbor-max-tanimoto",
        type=float,
        default=0.95,
        help="Maximum source-target Tanimoto for source_neighbor pairs; avoids near no-op edits.",
    )
    parser.add_argument(
        "--min-source-neighbors-t04",
        type=int,
        default=1,
        help="Minimum same-scaffold source neighbors with Tanimoto >= 0.4 required in source_neighbor mode.",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--render-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.image_dir is not None:
        args.image_dir.mkdir(parents=True, exist_ok=True)

    molecules = _read_molecules(args.molecule_db_csv)
    by_scaffold: dict[str, list[dict[str, str]]] = {}
    for row in molecules:
        by_scaffold.setdefault(row["scaffold"], []).append(row)

    if args.pairing_strategy == "source_neighbor":
        pairs, stats = _mine_source_neighbor_pairs(by_scaffold, args=args, rng=rng)
    else:
        pairs, stats = _mine_scaffold_random_pairs(by_scaffold, args=args, rng=rng)

    _assign_component_split(pairs, eval_fraction=args.eval_fraction, seed=args.seed)
    _write_rows(args.output_csv, pairs)

    summary = {
        "molecule_db_csv": str(args.molecule_db_csv),
        "output_csv": str(args.output_csv),
        "molecules": len(molecules),
        "unique_scaffolds": len(by_scaffold),
        "edit_pairs": len(pairs),
        "train_pairs": sum(1 for row in pairs if row["split"] == "train"),
        "eval_pairs": sum(1 for row in pairs if row["split"] == "eval"),
        "pairing_strategy": args.pairing_strategy,
        "max_pairs_per_source": args.max_pairs_per_source,
        "min_active_properties": args.min_active_properties,
        "threshold_scale": args.threshold_scale,
        "min_similarity": args.min_similarity,
        "max_similarity": args.max_similarity,
        "source_neighbor_min_tanimoto": args.source_neighbor_min_tanimoto,
        "source_neighbor_max_tanimoto": args.source_neighbor_max_tanimoto,
        "min_source_neighbors_t04": args.min_source_neighbors_t04,
        "render_images": bool(args.render_images),
        **stats,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_molecules(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        canonical = canonical_smiles(row.get("canonical_smiles", "")) or row.get("canonical_smiles", "")
        if not canonical or not row.get("scaffold"):
            continue
        row = dict(row)
        row["canonical_smiles"] = canonical
        out.append(row)
    return out


def _mine_scaffold_random_pairs(
    by_scaffold: dict[str, list[dict[str, str]]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    scaffold_items = list(by_scaffold.items())
    rng.shuffle(scaffold_items)
    pairs: list[dict[str, object]] = []
    stats = defaultdict(int)

    for scaffold, original_group in scaffold_items:
        if len(pairs) >= args.max_pairs:
            break
        group = _sample_scaffold_group(original_group, args=args, rng=rng)
        if len(group) < 2:
            continue
        neighbor_info = _empty_neighbor_info()
        candidate_indices = [(i, j) for i in range(len(group)) for j in range(len(group)) if i != j]
        rng.shuffle(candidate_indices)
        scaffold_pairs = 0
        for source_idx, target_idx in candidate_indices:
            if scaffold_pairs >= args.max_pairs_per_scaffold or len(pairs) >= args.max_pairs:
                break
            row = _maybe_pair_row(
                group[source_idx],
                group[target_idx],
                pair_id=f"mpair_{len(pairs):08d}",
                scaffold=scaffold,
                neighbor_info=neighbor_info,
                selection_reason="scaffold_random",
                args=args,
                stats=stats,
            )
            if row is None:
                continue
            pairs.append(row)
            scaffold_pairs += 1
    return pairs, dict(stats)


def _mine_source_neighbor_pairs(
    by_scaffold: dict[str, list[dict[str, str]]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    scaffold_items = list(by_scaffold.items())
    rng.shuffle(scaffold_items)
    pairs: list[dict[str, object]] = []
    stats = defaultdict(int)

    for scaffold, original_group in scaffold_items:
        if len(pairs) >= args.max_pairs:
            break
        group = _sample_scaffold_group(original_group, args=args, rng=rng)
        if len(group) < 2:
            continue
        neighbor_info = _same_scaffold_neighbor_info(group)
        candidate_indices = [(i, j) for i in range(len(group)) for j in range(len(group)) if i != j]
        rng.shuffle(candidate_indices)
        per_source_counts: dict[str, int] = defaultdict(int)
        scaffold_pairs = 0

        for source_idx, target_idx in candidate_indices:
            if scaffold_pairs >= args.max_pairs_per_scaffold or len(pairs) >= args.max_pairs:
                break
            source = group[source_idx]
            source_smiles = source["canonical_smiles"]
            counts = neighbor_info["counts"].get(source_smiles, {})
            if int(counts.get("source_neighbor_count_t04", 0)) < args.min_source_neighbors_t04:
                stats["skipped_sparse_source_neighborhood"] += 1
                continue
            if per_source_counts[source_smiles] >= args.max_pairs_per_source:
                stats["skipped_source_pair_cap"] += 1
                continue
            row = _maybe_pair_row(
                source,
                group[target_idx],
                pair_id=f"mpair_{len(pairs):08d}",
                scaffold=scaffold,
                neighbor_info=neighbor_info,
                selection_reason="same_scaffold_source_neighbor",
                args=args,
                stats=stats,
                min_similarity=args.source_neighbor_min_tanimoto,
                max_similarity=args.source_neighbor_max_tanimoto,
            )
            if row is None:
                continue
            pairs.append(row)
            per_source_counts[source_smiles] += 1
            scaffold_pairs += 1
    return pairs, dict(stats)


def _sample_scaffold_group(
    group: list[dict[str, str]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, str]]:
    group = list(group)
    rng.shuffle(group)
    if len(group) > args.max_molecules_per_scaffold:
        group = group[: args.max_molecules_per_scaffold]
    return group


def _same_scaffold_neighbor_info(group: list[dict[str, str]]) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {}
    ranks: dict[tuple[str, str], int] = {}
    similarity_cache: dict[tuple[str, str], float] = {}
    rows_by_source: dict[str, list[tuple[float, str]]] = {}

    for source in group:
        source_smiles = source["canonical_smiles"]
        neighbors: list[tuple[float, str]] = []
        for target in group:
            target_smiles = target["canonical_smiles"]
            if source_smiles == target_smiles:
                continue
            similarity = morgan_tanimoto(source_smiles, target_smiles)
            if similarity is None:
                continue
            similarity = float(similarity)
            similarity_cache[(source_smiles, target_smiles)] = similarity
            neighbors.append((similarity, target_smiles))
        rows_by_source[source_smiles] = sorted(neighbors, reverse=True)

    for source_smiles, neighbors in rows_by_source.items():
        counts[source_smiles] = {
            "same_scaffold_neighbor_count": len(neighbors),
            "source_neighbor_count_t04": sum(1 for similarity, _ in neighbors if similarity >= 0.4),
            "source_neighbor_count_t05": sum(1 for similarity, _ in neighbors if similarity >= 0.5),
            "source_neighbor_count_t06": sum(1 for similarity, _ in neighbors if similarity >= 0.6),
        }
        for rank, (_, target_smiles) in enumerate(neighbors, start=1):
            ranks[(source_smiles, target_smiles)] = rank

    return {"counts": counts, "ranks": ranks, "similarity_cache": similarity_cache}


def _empty_neighbor_info() -> dict[str, object]:
    return {"counts": {}, "ranks": {}, "similarity_cache": {}}


def _maybe_pair_row(
    source: dict[str, str],
    target: dict[str, str],
    *,
    pair_id: str,
    scaffold: str,
    neighbor_info: dict[str, object],
    selection_reason: str,
    args: argparse.Namespace,
    stats: defaultdict[str, int],
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> dict[str, object] | None:
    min_similarity = args.min_similarity if min_similarity is None else min_similarity
    max_similarity = args.max_similarity if max_similarity is None else max_similarity
    source_smiles = source["canonical_smiles"]
    target_smiles = target["canonical_smiles"]
    similarity_cache = neighbor_info["similarity_cache"]
    similarity = similarity_cache.get((source_smiles, target_smiles))
    if similarity is None:
        similarity = morgan_tanimoto(source_smiles, target_smiles)
    if similarity is None or not (min_similarity <= float(similarity) <= max_similarity):
        stats["skipped_similarity"] += 1
        return None

    source_props = _props(source)
    target_props = _props(target)
    active = active_property_deltas(source_props, target_props, threshold_scale=args.threshold_scale)
    if len(active) < args.min_active_properties:
        stats["skipped_delta"] += 1
        return None

    source_scaffold = source.get("scaffold", scaffold)
    target_scaffold = target.get("scaffold", scaffold)
    same_scaffold = source_scaffold == target_scaffold
    counts = neighbor_info["counts"].get(source_smiles, {})
    ranks = neighbor_info["ranks"]
    source_image, target_image = _image_paths(pair_id=pair_id, source=source, target=target, args=args)
    row: dict[str, object] = {
        "pair_id": pair_id,
        "split": "",
        "source_mol_id": source.get("mol_id", ""),
        "target_mol_id": target.get("mol_id", ""),
        "source_smiles": source_smiles,
        "target_smiles": target_smiles,
        "source_image": source_image,
        "target_image": target_image,
        "scaffold": source_scaffold if same_scaffold else "",
        "source_scaffold": source_scaffold,
        "target_scaffold": target_scaffold,
        "same_scaffold": str(bool(same_scaffold)),
        "scaffold_relation": "same_scaffold" if same_scaffold else "different_scaffold",
        "similarity": float(similarity),
        "source_tanimoto": float(similarity),
        "source_similarity_bin": source_similarity_bin(float(similarity)),
        "pair_quality_tier": pair_quality_tier(float(similarity), same_scaffold=same_scaffold),
        "selection_reason": selection_reason,
        "same_scaffold_neighbor_count": counts.get("same_scaffold_neighbor_count", ""),
        "source_neighbor_count_t04": counts.get("source_neighbor_count_t04", ""),
        "source_neighbor_count_t05": counts.get("source_neighbor_count_t05", ""),
        "source_neighbor_count_t06": counts.get("source_neighbor_count_t06", ""),
        "target_neighbor_rank_by_tanimoto": ranks.get((source_smiles, target_smiles), ""),
        "active_properties": ",".join(active.keys()),
        "active_property_count": len(active),
        "active_deltas_json": json_dumps(active),
        "directions_json": json_dumps({prop: direction_from_delta(delta) for prop, delta in active.items()}),
    }
    for prop in PROPERTY_COLUMNS:
        source_value = source_props[prop]
        target_value = target_props[prop]
        delta = target_value - source_value
        row[f"source_{prop}"] = source_value
        row[f"target_{prop}"] = target_value
        row[f"delta_{prop}"] = delta
        row[f"{prop}_direction"] = direction_from_delta(delta)
    return row


def _props(row: dict[str, str]) -> dict[str, float]:
    return {prop: float(row[prop]) for prop in PROPERTY_COLUMNS}


def _image_paths(*, pair_id: str, source: dict[str, str], target: dict[str, str], args: argparse.Namespace) -> tuple[str, str]:
    source_existing = source.get("image_path", "")
    target_existing = target.get("image_path", "")
    if not args.render_images:
        return source_existing, target_existing
    if args.image_dir is None:
        return source_existing, target_existing
    source_path = render_molecule_image(
        source["canonical_smiles"],
        args.image_dir / f"{pair_id}_source.png",
        args.image_size,
    )
    target_path = render_molecule_image(
        target["canonical_smiles"],
        args.image_dir / f"{pair_id}_target.png",
        args.image_size,
    )
    return source_path or source_existing, target_path or target_existing


def _assign_component_split(rows: list[dict[str, object]], *, eval_fraction: float, seed: int) -> None:
    graph: dict[str, set[str]] = {}
    molecule_to_row_ids: dict[str, set[int]] = {}
    for idx, row in enumerate(rows):
        source = str(row["source_smiles"])
        target = str(row["target_smiles"])
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
        molecule_to_row_ids.setdefault(source, set()).add(idx)
        molecule_to_row_ids.setdefault(target, set()).add(idx)

    seen: set[str] = set()
    components: list[set[int]] = []
    for molecule in graph:
        if molecule in seen:
            continue
        stack = [molecule]
        seen.add(molecule)
        component_molecules = set()
        while stack:
            current = stack.pop()
            component_molecules.add(current)
            for neighbor in graph.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_rows = set()
        for molecule in component_molecules:
            component_rows.update(molecule_to_row_ids.get(molecule, set()))
        components.append(component_rows)

    rng = random.Random(seed)
    rng.shuffle(components)
    target_eval = max(1, int(round(len(rows) * eval_fraction))) if rows else 0
    eval_ids: set[int] = set()
    for component in sorted(components, key=len):
        candidate = eval_ids | component
        if len(candidate) <= target_eval or not eval_ids:
            eval_ids = candidate
        if len(eval_ids) >= target_eval:
            break

    for idx, row in enumerate(rows):
        row["split"] = "eval" if idx in eval_ids else "train"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair_id",
        "split",
        "source_mol_id",
        "target_mol_id",
        "source_smiles",
        "target_smiles",
        "source_image",
        "target_image",
        "scaffold",
        "source_scaffold",
        "target_scaffold",
        "same_scaffold",
        "scaffold_relation",
        "similarity",
        "source_tanimoto",
        "source_similarity_bin",
        "pair_quality_tier",
        "selection_reason",
        "same_scaffold_neighbor_count",
        "source_neighbor_count_t04",
        "source_neighbor_count_t05",
        "source_neighbor_count_t06",
        "target_neighbor_rank_by_tanimoto",
        "active_properties",
        "active_property_count",
        "active_deltas_json",
        "directions_json",
    ]
    for prop in PROPERTY_COLUMNS:
        fieldnames.extend([f"source_{prop}", f"target_{prop}", f"delta_{prop}", f"{prop}_direction"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
