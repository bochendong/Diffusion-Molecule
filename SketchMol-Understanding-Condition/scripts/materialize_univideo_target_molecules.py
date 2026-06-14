#!/usr/bin/env python3
"""Materialize UniVideo target molecules without MolScribe/OCR.

The MolEdit-aligned path consumes benchmark condition rows, generated latents,
and target/candidate latents, then writes a direct-prediction CSV with
`generated_smiles`. This mirrors the Unified 3M x MolEdit materialized
benchmark shape while avoiding any generated-image OCR dependency.

Modes:
  target_oracle   Copy each row's `target_smiles` into `generated_smiles`.
                  This is an upper-bound/sanity-check control.
  source_identity Copy `source_smiles`; useful as a lower-bound diagnostic.
  latent_nearest  Retrieve the nearest candidate target molecule using saved
                  generated and candidate latent arrays.
  latent_property_rerank
                  Retrieve a latent-similarity shortlist, then rerank by
                  absolute target-property strict success and distance.
  property_nearest Retrieve the nearest candidate target molecule using active
                  target property values from the CSV.
  source_tanimoto_property_oracle
                  Unified-3M-style candidate-library upper bound: filter by
                  source Tanimoto, then pick the closest active-property match.
  edit_latent_source_first_rerank
                  Unified-3M-style latent retrieval after source-Tanimoto
                  filtering.
  edit_latent_source_similarity_rerank
                  Unified-3M-style latent retrieval plus source-Tanimoto rerank
                  without requiring scaffold identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS  # noqa: E402

PROPERTY_NORMALIZERS = {
    "MW": 50.0,
    "LogP": 0.5,
    "QED": 0.1,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
}
SKETCHMOL_STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
}
PROPERTY_ALIASES = {
    "gsk3b": "GSK3B",
    "gsk3β": "GSK3B",
    "gsk3": "GSK3B",
    "rb": "RB",
    "rotbonds": "RB",
    "rotbond": "RB",
    "rotatable": "RB",
    "rotatable_bonds": "RB",
    "mw": "MW",
    "molwt": "MW",
    "molecular_weight": "MW",
    "sa": "SA",
    "sas": "SA",
    "synthetic_accessibility": "SA",
    "haccept": "HBA",
    "hacceptor": "HBA",
    "hba": "HBA",
    "logp": "LogP",
    "qed": "QED",
    "drd2": "DRD2",
}
DIRECTION_ALIASES = {
    "increase": "increase",
    "up": "increase",
    "higher": "increase",
    "raise": "increase",
    "↑": "increase",
    "+": "increase",
    "decrease": "decrease",
    "down": "decrease",
    "lower": "decrease",
    "reduce": "decrease",
    "↓": "decrease",
    "-": "decrease",
}
RDKIT_PROPERTY_KEYS = {
    "MW": "MolWt",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "rotatable",
    "SA": "SA",
}
METHODS = (
    "target_oracle",
    "source_identity",
    "latent_nearest",
    "latent_property_rerank",
    "property_nearest",
    "source_tanimoto_property_oracle",
    "source_tanimoto_table_success_oracle",
    "edit_latent_source_first_rerank",
    "edit_latent_source_similarity_rerank",
    "edit_latent_table_success_rerank",
)
_PROPERTY_CACHE: dict[str, dict[str, float]] = {}
_ORACLE_CACHE: dict[str, object | None] = {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        required=True,
        type=Path,
        help="Benchmark condition rows CSV, such as univideo_molecule/benchmark_condition_rows.csv",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=METHODS,
        default="latent_nearest",
        help="Single method to export when --methods is not set.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods, aligned with Unified 3M materialized benchmark names.",
    )
    parser.add_argument("--candidate-csv", type=Path, default=None, help="Candidate rows; defaults to --source-csv")
    parser.add_argument("--candidate-smiles-column", default="target_smiles")
    parser.add_argument("--generated-latents-npy", type=Path, default=None)
    parser.add_argument("--candidate-latents-npy", type=Path, default=None)
    parser.add_argument("--eval-latent-dir", type=Path, default=None, help="Defaults to source_csv parent sibling eval_latent")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--exclude-self", action="store_true", help="Ignore candidates with the same condition/sample id")
    parser.add_argument("--source-first-min-tanimoto", type=float, default=0.4)
    parser.add_argument("--source-first-candidates", type=int, default=0)
    parser.add_argument("--source-similarity-weight", type=float, default=1.0)
    parser.add_argument("--source-similarity-rerank-candidates", type=int, default=256)
    parser.add_argument(
        "--table-success-rerank-candidates",
        type=int,
        default=512,
        help="Number of latent candidates to rerank by Table1-style property-direction success.",
    )
    parser.add_argument("--table-success-weight", type=float, default=100.0)
    parser.add_argument("--table-source-weight", type=float, default=5.0)
    parser.add_argument("--table-latent-weight", type=float, default=1.0)
    parser.add_argument(
        "--property-rerank-candidates",
        type=int,
        default=4096,
        help="Latent shortlist size for latent_property_rerank.",
    )
    parser.add_argument("--property-rerank-weight", type=float, default=10.0)
    parser.add_argument("--strict-rerank-weight", type=float, default=100.0)
    parser.add_argument("--latent-rerank-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows = read_rows(args.source_csv)
    if not source_rows:
        raise ValueError(f"No rows found in {args.source_csv}")
    candidate_csv = args.candidate_csv or args.source_csv
    candidate_rows = read_rows(candidate_csv)
    if not candidate_rows:
        raise ValueError(f"No candidate rows found in {candidate_csv}")

    methods = parse_methods(args.methods, fallback=args.mode)
    out_rows = []
    for method in methods:
        out_rows.extend(materialize_method(args, method, source_rows, candidate_rows))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_csv, out_rows)
    summary = summarize(out_rows, args=args, candidate_csv=candidate_csv, methods=methods)
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_methods(value: str | None, *, fallback: str) -> list[str]:
    methods = [item.strip() for item in str(value or fallback).split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unsupported method(s): {unknown}")
    return methods


def materialize_method(
    args: argparse.Namespace,
    method: str,
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    top_k = max(1, int(args.top_k))
    if method in {
        "source_tanimoto_property_oracle",
        "source_tanimoto_table_success_oracle",
        "edit_latent_source_first_rerank",
        "edit_latent_source_similarity_rerank",
        "edit_latent_table_success_rerank",
    }:
        ensure_source_tanimoto_available()
    if method == "target_oracle":
        out_rows = materialize_column(
            source_rows,
            candidate_rows=source_rows,
            generated_column="target_smiles",
            method=method,
        )
    elif method == "source_identity":
        out_rows = materialize_column(
            source_rows,
            candidate_rows=source_rows,
            generated_column="source_smiles",
            method=method,
        )
    elif method in {
        "latent_nearest",
        "edit_latent_source_first_rerank",
        "edit_latent_source_similarity_rerank",
        "edit_latent_table_success_rerank",
        "latent_property_rerank",
    }:
        generated_latents, candidate_latents = resolve_latents(args, len(source_rows), len(candidate_rows))
        if method == "latent_property_rerank":
            matches = latent_property_rerank_matches(
                generated_latents,
                candidate_latents,
                source_rows=source_rows,
                candidate_rows=candidate_rows,
                metric=args.metric,
                top_k=top_k,
                chunk_size=max(1, int(args.chunk_size)),
                exclude_self=bool(args.exclude_self),
                rerank_candidates=int(args.property_rerank_candidates),
                strict_weight=float(args.strict_rerank_weight),
                property_weight=float(args.property_rerank_weight),
                latent_weight=float(args.latent_rerank_weight),
                candidate_smiles_column=args.candidate_smiles_column,
            )
        else:
            matches = nearest_latent_matches(
                generated_latents,
                candidate_latents,
                source_rows=source_rows,
                candidate_rows=candidate_rows,
                metric=args.metric,
                top_k=top_k,
                chunk_size=max(1, int(args.chunk_size)),
                exclude_self=bool(args.exclude_self),
                source_first_min_tanimoto=(
                    float(args.source_first_min_tanimoto) if method == "edit_latent_source_first_rerank" else None
                ),
                source_first_candidates=int(args.source_first_candidates),
                source_similarity_rerank_candidates=(
                    int(args.source_similarity_rerank_candidates)
                    if method == "edit_latent_source_similarity_rerank"
                    else 0
                ),
                source_similarity_weight=float(args.source_similarity_weight),
                table_success_rerank_candidates=(
                    int(args.table_success_rerank_candidates)
                    if method == "edit_latent_table_success_rerank"
                    else 0
                ),
                table_success_weight=float(args.table_success_weight),
                table_source_weight=float(args.table_source_weight),
                table_latent_weight=float(args.table_latent_weight),
            )
        out_rows = materialize_matches(
            source_rows,
            candidate_rows,
            matches,
            method=method,
            candidate_smiles_column=args.candidate_smiles_column,
        )
    elif method in {"property_nearest", "source_tanimoto_property_oracle"}:
        matches = nearest_property_matches(
            source_rows,
            candidate_rows,
            top_k=top_k,
            exclude_self=bool(args.exclude_self),
            source_first_min_tanimoto=(
                float(args.source_first_min_tanimoto) if method == "source_tanimoto_property_oracle" else None
            ),
            source_first_candidates=int(args.source_first_candidates),
        )
        out_rows = materialize_matches(
            source_rows,
            candidate_rows,
            matches,
            method=method,
            candidate_smiles_column=args.candidate_smiles_column,
        )
    elif method == "source_tanimoto_table_success_oracle":
        matches = table_success_matches(
            source_rows,
            candidate_rows,
            top_k=top_k,
            exclude_self=bool(args.exclude_self),
            source_first_min_tanimoto=float(args.source_first_min_tanimoto),
            source_first_candidates=int(args.source_first_candidates),
            table_success_weight=float(args.table_success_weight),
            table_source_weight=float(args.table_source_weight),
            candidate_smiles_column=args.candidate_smiles_column,
        )
        out_rows = materialize_matches(
            source_rows,
            candidate_rows,
            matches,
            method=method,
            candidate_smiles_column=args.candidate_smiles_column,
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported method: {method}")
    return out_rows


def materialize_column(
    source_rows: list[dict[str, str]],
    *,
    candidate_rows: list[dict[str, str]],
    generated_column: str,
    method: str,
) -> list[dict[str, object]]:
    out_rows = []
    for index, row in enumerate(source_rows):
        generated = row.get(generated_column, "")
        out_rows.append(
            annotate_row(
                row,
                candidate_rows[index] if index < len(candidate_rows) else row,
                generated_smiles=generated,
                method=method,
                rank=1,
                score=1.0 if generated else math.nan,
                distance=0.0 if generated else math.nan,
                top_indices=[index],
                top_scores=[1.0 if generated else math.nan],
                top_distances=[0.0 if generated else math.nan],
                candidate_smiles_column="target_smiles",
            )
        )
    return out_rows


def materialize_matches(
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    matches: list[dict[str, object]],
    *,
    method: str,
    candidate_smiles_column: str,
) -> list[dict[str, object]]:
    out_rows = []
    for row, match in zip(source_rows, matches):
        best_index = int(match["best_index"])
        candidate = candidate_rows[best_index]
        generated = candidate.get(candidate_smiles_column, "")
        out_rows.append(
            annotate_row(
                row,
                candidate,
                generated_smiles=generated,
                method=method,
                rank=1,
                score=float(match.get("score", math.nan)),
                distance=float(match.get("distance", math.nan)),
                top_indices=list(match.get("top_indices", [])),
                top_scores=list(match.get("top_scores", [])),
                top_distances=list(match.get("top_distances", [])),
                candidate_smiles_column=candidate_smiles_column,
            )
        )
    return out_rows


def annotate_row(
    row: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    generated_smiles: str,
    method: str,
    rank: int,
    score: float,
    distance: float,
    top_indices: list[int],
    top_scores: list[float],
    top_distances: list[float],
    candidate_smiles_column: str,
) -> dict[str, object]:
    target_smiles = row.get("target_smiles", "")
    source_smiles = row.get("source_smiles", "")
    out: dict[str, object] = dict(row)
    out.update(
        {
            "generated_smiles": generated_smiles,
            "method": method,
            "target_finder_mode": method,
            "target_finder_rank": rank,
            "target_finder_score": clean_number(score),
            "target_finder_distance": clean_number(distance),
            "matched_sample_id": candidate.get("sample_id", ""),
            "matched_condition_id": candidate.get("condition_id", ""),
            "matched_pair_id": candidate.get("pair_id", ""),
            "matched_source_smiles": candidate.get("source_smiles", ""),
            "matched_target_smiles": candidate.get(candidate_smiles_column, ""),
            "topk_candidate_indices": ";".join(str(idx) for idx in top_indices),
            "topk_scores": ";".join(format_float(value) for value in top_scores),
            "topk_distances": ";".join(format_float(value) for value in top_distances),
            "exact_target_match": bool(generated_smiles and target_smiles and generated_smiles == target_smiles),
            "source_identity": bool(generated_smiles and source_smiles and generated_smiles == source_smiles),
        }
    )
    return out


def resolve_latents(args: argparse.Namespace, source_rows: int, candidate_rows: int) -> tuple[np.ndarray, np.ndarray]:
    latent_dir = args.eval_latent_dir or default_eval_latent_dir(args.source_csv)
    generated_path = args.generated_latents_npy or latent_dir / "generated_latents.npy"
    candidate_path = args.candidate_latents_npy or latent_dir / "target_latents.npy"
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Missing generated latents: {generated_path}. "
            "Use --mode target_oracle/property_nearest, or pass --generated-latents-npy."
        )
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Missing candidate latents: {candidate_path}. "
            "Use --mode target_oracle/property_nearest, or pass --candidate-latents-npy."
        )
    generated = as_2d(np.load(generated_path))
    candidates = as_2d(np.load(candidate_path))
    if generated.shape[0] != source_rows:
        raise ValueError(f"{generated_path} has {generated.shape[0]} rows, but source CSV has {source_rows}")
    if candidates.shape[0] != candidate_rows:
        raise ValueError(f"{candidate_path} has {candidates.shape[0]} rows, but candidate CSV has {candidate_rows}")
    if generated.shape[1] != candidates.shape[1]:
        raise ValueError(f"Latent dimension mismatch: generated={generated.shape}, candidates={candidates.shape}")
    return generated.astype(np.float32), candidates.astype(np.float32)


def default_eval_latent_dir(source_csv: Path) -> Path:
    if source_csv.parent.name == "image_structure_benchmark":
        return source_csv.parent.parent / "eval_latent"
    return source_csv.parent


def nearest_latent_matches(
    generated: np.ndarray,
    candidates: np.ndarray,
    *,
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    metric: str,
    top_k: int,
    chunk_size: int,
    exclude_self: bool,
    source_first_min_tanimoto: float | None,
    source_first_candidates: int,
    source_similarity_rerank_candidates: int,
    source_similarity_weight: float,
    table_success_rerank_candidates: int,
    table_success_weight: float,
    table_source_weight: float,
    table_latent_weight: float,
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    if metric == "cosine":
        candidate_matrix = normalize_rows(candidates)
        for start in range(0, generated.shape[0], chunk_size):
            query = normalize_rows(generated[start : start + chunk_size])
            sims = query @ candidate_matrix.T
            for offset, scores in enumerate(sims):
                row_index = start + offset
                if exclude_self:
                    mask_self(scores, source_rows[row_index], candidate_rows, fill=-np.inf)
                if source_first_min_tanimoto is not None:
                    mask_below_source_tanimoto(scores, source_rows[row_index], candidate_rows, source_first_min_tanimoto, fill=-np.inf)
                    if source_first_candidates > 0:
                        keep_only_top(scores, source_first_candidates, descending=True, fill=-np.inf)
                if source_similarity_rerank_candidates > 0:
                    ranked = top_indices_desc(scores, min(source_similarity_rerank_candidates, scores.shape[0]))
                    rerank_scores = scores.copy()
                    rerank_scores[:] = -np.inf
                    for idx in ranked:
                        source_tanimoto = candidate_source_tanimoto(source_rows[row_index], candidate_rows[int(idx)])
                        source_bonus = 0.0 if math.isnan(source_tanimoto) else float(source_similarity_weight) * source_tanimoto
                        rerank_scores[int(idx)] = float(scores[int(idx)]) + source_bonus
                    if np.any(np.isfinite(rerank_scores)):
                        scores = rerank_scores
                if table_success_rerank_candidates > 0:
                    ranked = top_indices_desc(scores, min(table_success_rerank_candidates, scores.shape[0]))
                    rerank_scores = scores.copy()
                    rerank_scores[:] = -np.inf
                    for idx in ranked:
                        candidate = candidate_rows[int(idx)]
                        table_score = table_success_candidate_score(
                            source_rows[row_index],
                            candidate,
                            candidate_smiles=str(candidate.get("target_smiles", "") or candidate.get("generated_smiles", "")),
                        )
                        if not table_score["valid"]:
                            continue
                        latent_bonus = float(table_latent_weight) * float(scores[int(idx)])
                        rerank_scores[int(idx)] = (
                            float(table_success_weight) * float(table_score["success_fraction"])
                            + float(table_source_weight) * float(table_score["source_tanimoto"])
                            + latent_bonus
                        )
                    if np.any(np.isfinite(rerank_scores)):
                        scores = rerank_scores
                top_indices = top_indices_desc(scores, top_k)
                matches.append(
                    {
                        "best_index": int(top_indices[0]),
                        "score": float(scores[top_indices[0]]),
                        "distance": float(1.0 - scores[top_indices[0]]),
                        "top_indices": [int(idx) for idx in top_indices],
                        "top_scores": [float(scores[idx]) for idx in top_indices],
                        "top_distances": [float(1.0 - scores[idx]) for idx in top_indices],
                    }
                )
        return matches

    for start in range(0, generated.shape[0], chunk_size):
        query = generated[start : start + chunk_size]
        # Squared L2 distance without allocating one array per row.
        q_norm = np.sum(query * query, axis=1, keepdims=True)
        c_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
        distances = np.maximum(q_norm + c_norm - 2.0 * (query @ candidates.T), 0.0)
        for offset, row_distances in enumerate(distances):
            row_index = start + offset
            if exclude_self:
                mask_self(row_distances, source_rows[row_index], candidate_rows, fill=np.inf)
            if source_first_min_tanimoto is not None:
                mask_below_source_tanimoto(row_distances, source_rows[row_index], candidate_rows, source_first_min_tanimoto, fill=np.inf)
                if source_first_candidates > 0:
                    keep_only_top(row_distances, source_first_candidates, descending=False, fill=np.inf)
            if source_similarity_rerank_candidates > 0:
                ranked = top_indices_asc(row_distances, min(source_similarity_rerank_candidates, row_distances.shape[0]))
                rerank_distances = row_distances.copy()
                rerank_distances[:] = np.inf
                for idx in ranked:
                    source_tanimoto = candidate_source_tanimoto(source_rows[row_index], candidate_rows[int(idx)])
                    source_bonus = 0.0 if math.isnan(source_tanimoto) else float(source_similarity_weight) * source_tanimoto
                    rerank_distances[int(idx)] = float(row_distances[int(idx)]) - source_bonus
                if np.any(np.isfinite(rerank_distances)):
                    row_distances = rerank_distances
            if table_success_rerank_candidates > 0:
                ranked = top_indices_asc(row_distances, min(table_success_rerank_candidates, row_distances.shape[0]))
                rerank_distances = row_distances.copy()
                rerank_distances[:] = np.inf
                for idx in ranked:
                    candidate = candidate_rows[int(idx)]
                    table_score = table_success_candidate_score(
                        source_rows[row_index],
                        candidate,
                        candidate_smiles=str(candidate.get("target_smiles", "") or candidate.get("generated_smiles", "")),
                    )
                    if not table_score["valid"]:
                        continue
                    rerank_distances[int(idx)] = (
                        float(table_latent_weight) * float(row_distances[int(idx)])
                        - float(table_success_weight) * float(table_score["success_fraction"])
                        - float(table_source_weight) * float(table_score["source_tanimoto"])
                    )
                if np.any(np.isfinite(rerank_distances)):
                    row_distances = rerank_distances
            top_indices = top_indices_asc(row_distances, top_k)
            best_distance = float(math.sqrt(float(row_distances[top_indices[0]])))
            top_distances = [float(math.sqrt(float(row_distances[idx]))) for idx in top_indices]
            matches.append(
                {
                    "best_index": int(top_indices[0]),
                    "score": float(-best_distance),
                    "distance": best_distance,
                    "top_indices": [int(idx) for idx in top_indices],
                    "top_scores": [float(-value) for value in top_distances],
                    "top_distances": top_distances,
                }
            )
    return matches


def latent_property_rerank_matches(
    generated: np.ndarray,
    candidates: np.ndarray,
    *,
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    metric: str,
    top_k: int,
    chunk_size: int,
    exclude_self: bool,
    rerank_candidates: int,
    strict_weight: float,
    property_weight: float,
    latent_weight: float,
    candidate_smiles_column: str,
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    shortlist_size = max(top_k, int(rerank_candidates))
    if metric == "cosine":
        candidate_matrix = normalize_rows(candidates)
        for start in range(0, generated.shape[0], chunk_size):
            query = normalize_rows(generated[start : start + chunk_size])
            sims = query @ candidate_matrix.T
            for offset, latent_scores in enumerate(sims):
                row_index = start + offset
                if exclude_self:
                    latent_scores = latent_scores.copy()
                    mask_self(latent_scores, source_rows[row_index], candidate_rows, fill=-np.inf)
                ranked = top_indices_desc(latent_scores, min(shortlist_size, latent_scores.shape[0]))
                matches.append(
                    rerank_property_shortlist(
                        source_rows[row_index],
                        candidate_rows,
                        ranked,
                        latent_scores,
                        top_k=top_k,
                        strict_weight=strict_weight,
                        property_weight=property_weight,
                        latent_weight=latent_weight,
                        candidate_smiles_column=candidate_smiles_column,
                    )
                )
        return matches

    for start in range(0, generated.shape[0], chunk_size):
        query = generated[start : start + chunk_size]
        q_norm = np.sum(query * query, axis=1, keepdims=True)
        c_norm = np.sum(candidates * candidates, axis=1, keepdims=True).T
        distances = np.maximum(q_norm + c_norm - 2.0 * (query @ candidates.T), 0.0)
        for offset, row_distances in enumerate(distances):
            row_index = start + offset
            if exclude_self:
                row_distances = row_distances.copy()
                mask_self(row_distances, source_rows[row_index], candidate_rows, fill=np.inf)
            ranked = top_indices_asc(row_distances, min(shortlist_size, row_distances.shape[0]))
            latent_scores = -np.sqrt(np.maximum(row_distances, 0.0))
            matches.append(
                rerank_property_shortlist(
                    source_rows[row_index],
                    candidate_rows,
                    ranked,
                    latent_scores,
                    top_k=top_k,
                    strict_weight=strict_weight,
                    property_weight=property_weight,
                    latent_weight=latent_weight,
                    candidate_smiles_column=candidate_smiles_column,
                )
            )
    return matches


def rerank_property_shortlist(
    row: Mapping[str, str],
    candidate_rows: list[dict[str, str]],
    shortlist_indices: np.ndarray,
    latent_scores: np.ndarray,
    *,
    top_k: int,
    strict_weight: float,
    property_weight: float,
    latent_weight: float,
    candidate_smiles_column: str,
) -> dict[str, object]:
    reranked: list[tuple[float, float, float, int]] = []
    for idx_value in shortlist_indices:
        idx = int(idx_value)
        property_score = absolute_property_candidate_score(
            row,
            candidate_rows[idx],
            candidate_smiles_column=candidate_smiles_column,
        )
        normalized_distance = float(property_score["normalized_property_distance"])
        strict_fraction = float(property_score["strict_fraction"])
        latent_score = float(latent_scores[idx])
        if not math.isfinite(latent_score):
            continue
        if not math.isfinite(normalized_distance):
            normalized_distance = 1e6
        final_score = (
            float(strict_weight) * strict_fraction
            - float(property_weight) * normalized_distance
            + float(latent_weight) * latent_score
        )
        reranked.append((final_score, normalized_distance, latent_score, idx))
    if not reranked:
        fallback = int(shortlist_indices[0]) if len(shortlist_indices) else 0
        fallback_score = float(latent_scores[fallback]) if fallback < latent_scores.shape[0] else math.nan
        return {
            "best_index": fallback,
            "score": fallback_score,
            "distance": math.nan,
            "top_indices": [fallback],
            "top_scores": [fallback_score],
            "top_distances": [math.nan],
        }
    reranked.sort(key=lambda item: (-item[0], item[1], -item[2], item[3]))
    selected = reranked[: max(1, top_k)]
    return {
        "best_index": int(selected[0][3]),
        "score": float(selected[0][0]),
        "distance": float(selected[0][1]),
        "top_indices": [int(item[3]) for item in selected],
        "top_scores": [float(item[0]) for item in selected],
        "top_distances": [float(item[1]) for item in selected],
    }


def nearest_property_matches(
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    top_k: int,
    exclude_self: bool,
    source_first_min_tanimoto: float | None = None,
    source_first_candidates: int = 0,
) -> list[dict[str, object]]:
    candidate_vectors = [property_vector(row, active_from_row=False) for row in candidate_rows]
    matches: list[dict[str, object]] = []
    for row in source_rows:
        query_values, query_mask = property_vector(row, active_from_row=True)
        distances = []
        for candidate, candidate_vector in zip(candidate_rows, candidate_vectors):
            candidate_values, candidate_mask = candidate_vector
            mask = query_mask & candidate_mask
            if not np.any(mask):
                distance = np.inf
            else:
                deltas = np.abs(query_values[mask] - candidate_values[mask])
                normalizers = np.asarray([PROPERTY_NORMALIZERS[prop] for prop, keep in zip(PROPERTY_COLUMNS, mask) if keep])
                distance = float(np.mean(deltas / normalizers))
            distances.append(distance)
        distance_array = np.asarray(distances, dtype=np.float32)
        if exclude_self:
            mask_self(distance_array, row, candidate_rows, fill=np.inf)
        if source_first_min_tanimoto is not None:
            mask_below_source_tanimoto(distance_array, row, candidate_rows, source_first_min_tanimoto, fill=np.inf)
            if source_first_candidates > 0:
                keep_only_top(distance_array, source_first_candidates, descending=False, fill=np.inf)
        top_indices = top_indices_asc(distance_array, top_k)
        best_distance = float(distance_array[top_indices[0]])
        matches.append(
            {
                "best_index": int(top_indices[0]),
                "score": float(-best_distance) if math.isfinite(best_distance) else math.nan,
                "distance": best_distance,
                "top_indices": [int(idx) for idx in top_indices],
                "top_scores": [
                    float(-distance_array[idx]) if math.isfinite(float(distance_array[idx])) else math.nan
                    for idx in top_indices
                ],
                "top_distances": [float(distance_array[idx]) for idx in top_indices],
            }
        )
    return matches


def table_success_matches(
    source_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    top_k: int,
    exclude_self: bool,
    source_first_min_tanimoto: float,
    source_first_candidates: int,
    table_success_weight: float,
    table_source_weight: float,
    candidate_smiles_column: str,
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for row in source_rows:
        scores = np.full(len(candidate_rows), -np.inf, dtype=np.float32)
        source_tanimotos = np.full(len(candidate_rows), np.nan, dtype=np.float32)
        for idx, candidate in enumerate(candidate_rows):
            candidate_smiles = str(candidate.get(candidate_smiles_column, "") or "").strip()
            table_score = table_success_candidate_score(row, candidate, candidate_smiles=candidate_smiles)
            if not table_score["valid"]:
                continue
            source_tanimotos[idx] = float(table_score["source_tanimoto"])
            if source_first_min_tanimoto > 0 and float(table_score["source_tanimoto"]) < source_first_min_tanimoto:
                continue
            scores[idx] = (
                float(table_success_weight) * float(table_score["success_fraction"])
                + float(table_source_weight) * float(table_score["source_tanimoto"])
            )
        if exclude_self:
            mask_self(scores, row, candidate_rows, fill=-np.inf)
        if source_first_candidates > 0:
            keep_only_top(scores, source_first_candidates, descending=True, fill=-np.inf)
        if not np.any(np.isfinite(scores)):
            for idx, value in enumerate(source_tanimotos):
                if math.isfinite(float(value)):
                    scores[idx] = float(value)
        top_indices = top_indices_desc(scores, top_k)
        best = int(top_indices[0])
        best_score = float(scores[best])
        matches.append(
            {
                "best_index": best,
                "score": best_score,
                "distance": float(1.0 - source_tanimotos[best]) if math.isfinite(float(source_tanimotos[best])) else math.nan,
                "top_indices": [int(idx) for idx in top_indices],
                "top_scores": [float(scores[idx]) for idx in top_indices],
                "top_distances": [
                    float(1.0 - source_tanimotos[idx]) if math.isfinite(float(source_tanimotos[idx])) else math.nan
                    for idx in top_indices
                ],
            }
        )
    return matches


def table_success_candidate_score(
    row: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    candidate_smiles: str,
) -> dict[str, object]:
    source_smiles = str(row.get("source_smiles", "") or "").strip()
    candidate_smiles = str(candidate_smiles or "").strip()
    if not source_smiles or not candidate_smiles:
        return {"valid": False, "success_fraction": 0.0, "source_tanimoto": 0.0}
    try:
        source_tanimoto = candidate_source_tanimoto(row, {"target_smiles": candidate_smiles})
    except RuntimeError:
        source_tanimoto = math.nan
    if math.isnan(source_tanimoto):
        return {"valid": False, "success_fraction": 0.0, "source_tanimoto": 0.0}
    task_specs = task_specs_from_row(row)
    if not task_specs:
        return {"valid": True, "success_fraction": 0.0, "source_tanimoto": float(source_tanimoto)}

    successes = []
    for spec in task_specs:
        prop = canonical_property(spec.get("property", ""))
        direction = canonical_direction(spec.get("direction", ""))
        if not prop or direction not in {"increase", "decrease"}:
            successes.append(False)
            continue
        source_value = property_value(source_smiles, prop)
        candidate_value = property_value(candidate_smiles, prop)
        if source_value is None or candidate_value is None:
            successes.append(False)
            continue
        if direction == "increase":
            successes.append(float(candidate_value) > float(source_value))
        else:
            successes.append(float(candidate_value) < float(source_value))
    success_fraction = fraction(successes)
    return {
        "valid": True,
        "success_fraction": float(success_fraction),
        "source_tanimoto": float(source_tanimoto),
    }


def absolute_property_candidate_score(
    row: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    candidate_smiles_column: str,
) -> dict[str, float | bool]:
    query_values, query_mask = property_vector(row, active_from_row=True)
    if not np.any(query_mask):
        return {
            "valid": False,
            "strict_fraction": 0.0,
            "normalized_property_distance": math.inf,
        }

    successes = []
    normalized_errors = []
    for prop, query_value, keep in zip(PROPERTY_COLUMNS, query_values, query_mask):
        if not keep:
            continue
        candidate_value = candidate_property_value(candidate, prop, candidate_smiles_column=candidate_smiles_column)
        tolerance = float(SKETCHMOL_STRICT_TOLERANCE.get(prop, PROPERTY_NORMALIZERS.get(prop, 1.0)))
        if not math.isfinite(float(query_value)) or candidate_value is None or not math.isfinite(float(candidate_value)):
            successes.append(False)
            normalized_errors.append(math.inf)
            continue
        error = abs(float(candidate_value) - float(query_value))
        successes.append(error <= tolerance)
        normalized_errors.append(error / max(tolerance, 1e-8))
    finite_errors = [value for value in normalized_errors if math.isfinite(float(value))]
    if not normalized_errors:
        distance = math.inf
    elif finite_errors:
        missing_penalty = 1e6 * (len(normalized_errors) - len(finite_errors))
        distance = (sum(finite_errors) + missing_penalty) / len(normalized_errors)
    else:
        distance = math.inf
    return {
        "valid": bool(successes),
        "strict_fraction": fraction(successes),
        "normalized_property_distance": float(distance),
    }


def candidate_property_value(
    candidate: Mapping[str, str],
    prop: str,
    *,
    candidate_smiles_column: str,
) -> float | None:
    prop = canonical_property(prop)
    value = to_float(candidate.get(f"target_{prop}"))
    if math.isfinite(value):
        return float(value)
    value = to_float(candidate.get(prop))
    if math.isfinite(value):
        return float(value)
    smiles = str(
        candidate.get(candidate_smiles_column, "")
        or candidate.get("target_smiles", "")
        or candidate.get("generated_smiles", "")
        or ""
    ).strip()
    return property_value(smiles, prop)


def task_specs_from_row(row: Mapping[str, str]) -> list[dict[str, str]]:
    for key in ("moledit_tasks", "instruction_tasks"):
        raw = row.get(key, "")
        if not raw:
            continue
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            specs = []
            for item in parsed:
                if not isinstance(item, Mapping):
                    continue
                prop = canonical_property(item.get("property", ""))
                direction = canonical_direction(item.get("direction", ""))
                if prop:
                    specs.append({"property": prop, "direction": direction})
            if specs:
                return specs
    specs = []
    for prop in selected_props(row.get("condition_properties", "")):
        canonical = canonical_property(prop)
        direction = canonical_direction(row.get(f"{canonical}_direction", "") or row.get(f"{prop}_direction", ""))
        if canonical:
            specs.append({"property": canonical, "direction": direction})
    return specs


def property_value(smiles: str, prop: str) -> float | None:
    prop = canonical_property(prop)
    if not smiles:
        return None
    props = cached_molecular_properties(smiles)
    if prop in RDKIT_PROPERTY_KEYS:
        key = RDKIT_PROPERTY_KEYS[prop]
        value = props.get(key)
        return None if value is None else float(value)
    oracle = tdc_oracle(prop)
    if oracle is None:
        return None
    try:
        return float(oracle(smiles))
    except Exception:
        return None


def cached_molecular_properties(smiles: str) -> dict[str, float]:
    key = str(smiles or "").strip()
    if key not in _PROPERTY_CACHE:
        try:
            from sketchmol_understanding_condition.chem import molecular_properties

            props = molecular_properties(key)
        except RuntimeError:
            props = None
        _PROPERTY_CACHE[key] = dict(props or {})
    return _PROPERTY_CACHE[key]


def tdc_oracle(prop: str):
    prop = canonical_property(prop)
    if prop in _ORACLE_CACHE:
        return _ORACLE_CACHE[prop]
    ensure_rdkit_six_compat()
    try:
        from tdc import Oracle

        _ORACLE_CACHE[prop] = Oracle(name=prop)
    except Exception:
        _ORACLE_CACHE[prop] = None
    return _ORACLE_CACHE[prop]


def ensure_rdkit_six_compat() -> None:
    if "rdkit.six" in sys.modules:
        return
    try:
        from rdkit.six import iteritems  # noqa: F401
    except ModuleNotFoundError:
        six_mod = types.ModuleType("rdkit.six")
        six_mod.iteritems = dict.items
        sys.modules["rdkit.six"] = six_mod


def canonical_property(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PROPERTY_ALIASES.get(text.lower(), text)


def canonical_direction(value: object) -> str:
    text = str(value or "").strip().lower()
    return DIRECTION_ALIASES.get(text, text or "unknown")


def property_vector(row: Mapping[str, str], *, active_from_row: bool) -> tuple[np.ndarray, np.ndarray]:
    values = []
    mask = []
    active_props = selected_props(row.get("condition_properties", ""))
    for prop in PROPERTY_COLUMNS:
        values.append(to_float(row.get(f"target_{prop}")))
        active_flag = truthy(row.get(f"{prop}_active"))
        if active_from_row:
            active = active_flag if active_flag is not None else prop in active_props
        else:
            active = True
        mask.append(bool(active) and math.isfinite(values[-1]))
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=bool)


def selected_props(text: str | None) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def mask_self(values: np.ndarray, row: Mapping[str, str], candidate_rows: list[dict[str, str]], *, fill: float) -> None:
    row_keys = {row.get("condition_id", ""), row.get("sample_id", "")}
    row_keys.discard("")
    if not row_keys:
        return
    for idx, candidate in enumerate(candidate_rows):
        candidate_keys = {candidate.get("condition_id", ""), candidate.get("sample_id", "")}
        candidate_keys.discard("")
        if row_keys & candidate_keys:
            values[idx] = fill


def mask_below_source_tanimoto(
    values: np.ndarray,
    row: Mapping[str, str],
    candidate_rows: list[dict[str, str]],
    threshold: float,
    *,
    fill: float,
) -> None:
    for idx, candidate in enumerate(candidate_rows):
        source_tanimoto = candidate_source_tanimoto(row, candidate)
        if math.isnan(source_tanimoto) or source_tanimoto < float(threshold):
            values[idx] = fill


def candidate_source_tanimoto(row: Mapping[str, str], candidate: Mapping[str, str]) -> float:
    source = str(row.get("source_smiles", "") or "").strip()
    candidate_smiles = str(candidate.get("target_smiles", "") or candidate.get("generated_smiles", "") or "").strip()
    if not source or not candidate_smiles:
        return math.nan
    try:
        from sketchmol_understanding_condition.chem import morgan_tanimoto

        value = morgan_tanimoto(source, candidate_smiles)
    except RuntimeError:
        value = None
    return float(value) if value is not None else math.nan


def ensure_source_tanimoto_available() -> None:
    try:
        from sketchmol_understanding_condition.chem import morgan_tanimoto

        morgan_tanimoto("CCO", "CCO")
    except RuntimeError as exc:
        raise RuntimeError(
            "RDKit is required for source-Tanimoto materialized benchmark methods. "
            "Use --methods source_identity,target_oracle,latent_nearest,property_nearest "
            "or load an RDKit environment."
        ) from exc


def keep_only_top(values: np.ndarray, count: int, *, descending: bool, fill: float) -> None:
    if count <= 0 or count >= values.shape[0]:
        return
    ranked = top_indices_desc(values, count) if descending else top_indices_asc(values, count)
    keep = np.zeros(values.shape[0], dtype=bool)
    keep[ranked] = True
    values[~keep] = fill


def top_indices_desc(scores: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, scores.shape[0])
    if k == scores.shape[0]:
        return np.argsort(-scores)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates])]


def top_indices_asc(distances: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, distances.shape[0])
    if k == distances.shape[0]:
        return np.argsort(distances)
    candidates = np.argpartition(distances, k - 1)[:k]
    return candidates[np.argsort(distances[candidates])]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values
    if values.ndim < 2:
        raise ValueError(f"Expected 2D or higher latent array, got shape {values.shape}")
    return values.reshape(values.shape[0], -1)


def summarize(
    out_rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    candidate_csv: Path,
    methods: list[str],
) -> dict[str, object]:
    exact = [bool(row.get("exact_target_match")) for row in out_rows]
    source_identity = [bool(row.get("source_identity")) for row in out_rows]
    nonempty = [bool(str(row.get("generated_smiles", "")).strip()) for row in out_rows]
    return {
        "source_csv": str(args.source_csv),
        "candidate_csv": str(candidate_csv),
        "output_csv": str(args.output_csv),
        "mode": args.mode,
        "methods": methods,
        "metric": args.metric if any("latent" in method for method in methods) else "",
        "rows": len(out_rows),
        "generated_smiles_present_rate": fraction(nonempty),
        "exact_target_match_rate": fraction(exact),
        "source_identity_rate": fraction(source_identity),
        "top_k": int(args.top_k),
        "exclude_self": bool(args.exclude_self),
        "property_rerank_candidates": int(args.property_rerank_candidates),
        "property_rerank_weight": float(args.property_rerank_weight),
        "strict_rerank_weight": float(args.strict_rerank_weight),
        "latent_rerank_weight": float(args.latent_rerank_weight),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def truthy(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def fraction(values: Sequence[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def clean_number(value: float) -> object:
    return "" if not math.isfinite(float(value)) else float(value)


def format_float(value: float) -> str:
    return "" if not math.isfinite(float(value)) else f"{float(value):.8g}"


if __name__ == "__main__":
    raise SystemExit(main())
