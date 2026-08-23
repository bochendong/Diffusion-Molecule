#!/usr/bin/env python3
"""Select strict frozen-D3 candidates and serialize them near their sources."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFMCS


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
for path in (PROJECT_DIR / "scripts", PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_direct_smiles_generator_rl import table1_edit_score_components  # noqa: E402
from sketchmol_understanding_condition.chem import canonical_smiles, morgan_tanimoto  # noqa: E402
from sketchmol_understanding_condition.direct_smiles_generation import tokenize_smiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--teacher-candidates", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--winners-per-condition", type=int, default=2)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.65)
    parser.add_argument("--min-covered-fraction", type=float, default=0.45)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_id(row: dict[str, str]) -> str:
    return str(row.get("condition_id", "") or row.get("sample_id", "") or row.get("pair_id", "")).strip()


def token_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, tokenize_smiles(left), tokenize_smiles(right)).ratio()


def source_aligned_smiles(source_smiles: str, target_smiles: str) -> tuple[str, float, float]:
    source = Chem.MolFromSmiles(source_smiles)
    target = Chem.MolFromSmiles(target_smiles)
    canonical = canonical_smiles(target_smiles)
    baseline = token_similarity(source_smiles, canonical)
    if source is None or target is None:
        return canonical, baseline, baseline
    result = rdFMCS.FindMCS(
        [source, target],
        timeout=3,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=True,
    )
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    if query is None:
        return canonical, baseline, baseline
    source_matches = source.GetSubstructMatches(query, uniquify=True, maxMatches=32)
    target_matches = target.GetSubstructMatches(query, uniquify=True, maxMatches=32)
    best = canonical
    best_score = baseline
    for source_match in source_matches:
        for target_match in target_matches:
            mapping = sorted(zip(source_match, target_match), key=lambda item: item[0])
            mapped_target = [target_idx for _, target_idx in mapping]
            order = mapped_target + [idx for idx in range(target.GetNumAtoms()) if idx not in set(mapped_target)]
            if len(order) != target.GetNumAtoms() or len(set(order)) != len(order):
                continue
            try:
                aligned_mol = Chem.RenumberAtoms(target, order)
                candidate = Chem.MolToSmiles(aligned_mol, canonical=False, isomericSmiles=True)
            except Exception:
                continue
            if canonical_smiles(candidate) != canonical:
                continue
            score = token_similarity(source_smiles, candidate)
            if score > best_score:
                best, best_score = candidate, score
    return best, baseline, best_score


def main() -> int:
    args = parse_args()
    references = {row_id(row): row for row in read_rows(args.reference_csv) if row_id(row)}
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.teacher_candidates):
        identifier = row_id(row)
        if identifier in references:
            candidates[identifier].append(row)
    output: list[dict[str, str]] = []
    covered = 0
    improved_alignment = 0
    task_coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for identifier, ref in references.items():
        task = str(ref.get("p4_teacher_task_key", "") or ref.get("task", ""))
        task_coverage[task][1] += 1
        accepted = []
        source = canonical_smiles(ref.get("source_smiles", ""))
        for candidate in candidates.get(identifier, []):
            generated = canonical_smiles(
                candidate.get("generated_smiles", "")
                or candidate.get("candidate_smiles", "")
                or candidate.get("prediction_smiles", "")
            )
            if not generated or generated == source:
                continue
            similarity = morgan_tanimoto(source, generated)
            if similarity is None or float(similarity) < float(args.min_source_tanimoto):
                continue
            strict_fraction, distance = table1_edit_score_components(ref, generated)
            if float(strict_fraction) < 1.0 - 1e-9:
                continue
            aligned, canonical_score, aligned_score = source_aligned_smiles(source, generated)
            accepted.append((float(similarity), -float(distance), aligned_score, generated, aligned, canonical_score))
        accepted.sort(reverse=True)
        if accepted:
            covered += 1
            task_coverage[task][0] += 1
        for winner_rank, values in enumerate(accepted[: max(1, int(args.winners_per_condition))], start=1):
            similarity, neg_distance, aligned_score, generated, aligned, canonical_score = values
            row = dict(ref)
            row["target_smiles"] = aligned
            row["task_mode"] = "edit"
            row["p4_teacher_generated_smiles"] = generated
            row["p4_teacher_source_tanimoto"] = f"{similarity:.8f}"
            row["p4_teacher_property_distance"] = f"{-neg_distance:.8f}"
            row["p4_canonical_token_similarity"] = f"{canonical_score:.8f}"
            row["p4_aligned_token_similarity"] = f"{aligned_score:.8f}"
            row["p4_teacher_rank"] = str(winner_rank)
            output.append(row)
            improved_alignment += int(aligned_score > canonical_score + 1e-9)
    fraction = covered / max(len(references), 1)
    if fraction < float(args.min_covered_fraction):
        raise ValueError(f"D3 teacher strict coverage {fraction:.3f} is below {args.min_covered_fraction:.3f}")
    fields = list(dict.fromkeys(key for row in output for key in row))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    manifest = {
        "protocol": "p4_event_to_smiles_distillation_rows_v1",
        "reference_rows": len(references),
        "covered_conditions": covered,
        "covered_fraction": fraction,
        "distillation_rows": len(output),
        "winners_per_condition": int(args.winners_per_condition),
        "min_source_tanimoto": float(args.min_source_tanimoto),
        "alignment_improved_rows": improved_alignment,
        "task_coverage": {
            key: {"covered": value[0], "total": value[1]}
            for key, value in sorted(task_coverage.items())
        },
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
