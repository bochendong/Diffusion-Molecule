#!/usr/bin/env python
"""Benchmark multi-property edit conditions with scaffold-aware retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNDERSTANDING_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
for path in (DATASET_DIR, UNDERSTANDING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sketchmol_multiproperty_dataset.common import (
    PROPERTY_COLUMNS,
    SKETCHMOL_REFERENCE_MULTI_PROPERTY,
    SKETCHMOL_STRICT_TOLERANCE,
)
from sketchmol_understanding_condition.chem import (
    canonical_smiles as _canonical_smiles,
    morgan_fingerprint_bits as _morgan_fingerprint_bits,
    morgan_tanimoto as _morgan_tanimoto,
    scaffold_smiles as _scaffold_smiles,
)


METHODS = (
    "source_identity",
    "global_property_retrieval",
    "scaffold_property_retrieval",
    "edit_latent_global_retrieval",
    "edit_latent_scaffold_retrieval",
    "edit_latent_scaffold_source_rerank",
    "vlm_feature_retrieval",
    "vlm_scaffold_feature_retrieval",
    "global_property_vlm_rerank",
    "scaffold_property_vlm_rerank",
    "target_oracle",
)

FEATURE_METHODS = {
    "vlm_feature_retrieval",
    "vlm_scaffold_feature_retrieval",
    "global_property_vlm_rerank",
    "scaffold_property_vlm_rerank",
}
PURE_FEATURE_METHODS = {"vlm_feature_retrieval", "vlm_scaffold_feature_retrieval"}
HYBRID_RERANK_METHODS = {"global_property_vlm_rerank", "scaffold_property_vlm_rerank"}
EDIT_LATENT_METHODS = {
    "edit_latent_global_retrieval",
    "edit_latent_scaffold_retrieval",
    "edit_latent_scaffold_source_rerank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-rows-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate-molecule-db-csv",
        type=Path,
        default=None,
        help=(
            "Optional molecule_database.csv candidate library. When omitted, "
            "retrieval candidates are built from train target rows only."
        ),
    )
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-global-candidates", type=int, default=20000)
    parser.add_argument("--max-feature-candidates", type=int, default=20000)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument("--rerank-property-weight", type=float, default=0.0)
    parser.add_argument(
        "--scaffold-fallback-mode",
        choices=["global", "source_identity", "empty"],
        default="global",
        help=(
            "What scaffold-aware methods should do when no train candidate has "
            "the eval scaffold. `global` preserves the old behavior, "
            "`source_identity` keeps the source molecule as an honest edit "
            "fallback, and `empty` reports no candidate."
        ),
    )
    parser.add_argument("--limit-eval-rows", type=int, default=None)
    parser.add_argument("--max-eval-per-property-count", type=int, default=None)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", choices=["pooled", "query_tokens"], default="pooled")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument("--edit-latent-dir", type=Path, default=None)
    parser.add_argument("--max-edit-latent-candidates", type=int, default=20000)
    parser.add_argument("--edit-latent-property-weight", type=float, default=1.0)
    parser.add_argument("--edit-latent-delta-weight", type=float, default=0.35)
    parser.add_argument("--edit-latent-direction-weight", type=float, default=0.10)
    parser.add_argument("--edit-latent-fingerprint-weight", type=float, default=0.0)
    parser.add_argument("--edit-latent-source-similarity-weight", type=float, default=0.25)
    parser.add_argument("--source-tanimoto-thresholds", default="0.4,0.6,0.8")
    parser.add_argument("--compute-tanimoto", action="store_true")
    parser.add_argument("--allow-eval-target-candidates", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if any(method in FEATURE_METHODS for method in methods) and args.condition_features_dir is None:
        raise ValueError("--condition-features-dir is required for VLM feature retrieval methods")
    if any(method in EDIT_LATENT_METHODS for method in methods) and args.edit_latent_dir is None:
        raise ValueError("--edit-latent-dir is required for edit-latent retrieval methods")
    source_tanimoto_thresholds = _parse_float_list(args.source_tanimoto_thresholds)

    rows = _read_rows(args.condition_rows_csv)
    train_rows = [row for row in rows if row.get("split") != args.eval_split]
    eval_rows = [row for row in rows if row.get("split") == args.eval_split]
    if args.max_eval_per_property_count is not None:
        eval_rows = _sample_eval_rows_by_property_count(eval_rows, args.max_eval_per_property_count, seed=args.seed)
    if args.limit_eval_rows is not None:
        eval_rows = eval_rows[: args.limit_eval_rows]
    excluded_targets = set()
    if not args.allow_eval_target_candidates:
        excluded_targets = {
            _safe_canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
            for row in eval_rows
            if row.get("target_smiles")
        }
    if args.candidate_molecule_db_csv is not None:
        candidates = _candidate_pool_from_molecule_db(args.candidate_molecule_db_csv, excluded_smiles=excluded_targets)
        candidate_source = "molecule_database"
    else:
        candidates = _candidate_pool(train_rows, excluded_smiles=excluded_targets)
        candidate_source = "train_target_rows"
    by_scaffold: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        by_scaffold[str(candidate["scaffold"])].append(candidate)
    feature_context = None
    if args.condition_features_dir is not None:
        feature_context = _build_feature_context(
            train_rows,
            condition_features_dir=args.condition_features_dir,
            array_name=args.condition_feature_array,
            variant=args.condition_feature_variant,
            excluded_smiles=excluded_targets,
        )
    edit_latent_context = None
    if args.edit_latent_dir is not None:
        edit_latent_context = _build_edit_latent_context(
            condition_rows=rows,
            edit_latent_dir=args.edit_latent_dir,
            variant=args.condition_feature_variant,
        )

    decoded_rows = []
    for method in methods:
        for row in eval_rows:
            decoded_rows.append(
                _decode_row(
                    row,
                    method=method,
                    candidates=candidates,
                    by_scaffold=by_scaffold,
                    feature_context=feature_context,
                    edit_latent_context=edit_latent_context,
                    max_global_candidates=args.max_global_candidates,
                    max_feature_candidates=args.max_feature_candidates,
                    max_edit_latent_candidates=args.max_edit_latent_candidates,
                    rerank_candidates=args.rerank_candidates,
                    rerank_property_weight=args.rerank_property_weight,
                    edit_latent_property_weight=args.edit_latent_property_weight,
                    edit_latent_delta_weight=args.edit_latent_delta_weight,
                    edit_latent_direction_weight=args.edit_latent_direction_weight,
                    edit_latent_fingerprint_weight=args.edit_latent_fingerprint_weight,
                    edit_latent_source_similarity_weight=args.edit_latent_source_similarity_weight,
                    scaffold_fallback_mode=args.scaffold_fallback_mode,
                    compute_tanimoto=args.compute_tanimoto or bool(source_tanimoto_thresholds),
                    seed=args.seed,
                )
            )

    summary_rows = _summarize(decoded_rows, source_tanimoto_thresholds=source_tanimoto_thresholds)
    _write_rows(args.output_dir / "benchmark_decoded.csv", decoded_rows)
    _write_rows(args.output_dir / "benchmark_summary.csv", summary_rows)
    _write_report(args.output_dir / "benchmark_report.md", summary_rows, args)
    payload = {
        "condition_rows_csv": str(args.condition_rows_csv),
        "output_dir": str(args.output_dir),
        "methods": methods,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "candidate_molecules": len(candidates),
        "candidate_source": candidate_source,
        "candidate_molecule_db_csv": str(args.candidate_molecule_db_csv) if args.candidate_molecule_db_csv else None,
        "eval_target_candidates_excluded": not args.allow_eval_target_candidates,
        "max_eval_per_property_count": args.max_eval_per_property_count,
        "max_global_candidates": args.max_global_candidates,
        "max_feature_candidates": args.max_feature_candidates,
        "max_edit_latent_candidates": args.max_edit_latent_candidates,
        "rerank_candidates": args.rerank_candidates,
        "rerank_property_weight": args.rerank_property_weight,
        "edit_latent_dir": str(args.edit_latent_dir) if args.edit_latent_dir else None,
        "edit_latent_property_weight": args.edit_latent_property_weight,
        "edit_latent_delta_weight": args.edit_latent_delta_weight,
        "edit_latent_direction_weight": args.edit_latent_direction_weight,
        "edit_latent_fingerprint_weight": args.edit_latent_fingerprint_weight,
        "edit_latent_source_similarity_weight": args.edit_latent_source_similarity_weight,
        "source_tanimoto_thresholds": source_tanimoto_thresholds,
        "scaffold_fallback_mode": args.scaffold_fallback_mode,
        "condition_features_dir": str(args.condition_features_dir) if args.condition_features_dir else None,
        "condition_feature_array": args.condition_feature_array if args.condition_features_dir else None,
        "condition_feature_variant": args.condition_feature_variant if args.condition_features_dir else None,
        "summary_csv": str(args.output_dir / "benchmark_summary.csv"),
        "decoded_csv": str(args.output_dir / "benchmark_decoded.csv"),
        "report": str(args.output_dir / "benchmark_report.md"),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_canonical_smiles(smiles: str) -> str | None:
    try:
        return _canonical_smiles(smiles)
    except RuntimeError:
        text = str(smiles or "").strip()
        return text or None


def _safe_scaffold_smiles(smiles: str) -> str | None:
    try:
        return _scaffold_smiles(smiles)
    except RuntimeError:
        return None


@lru_cache(maxsize=500000)
def _safe_morgan_tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    try:
        return _morgan_tanimoto(smiles_a, smiles_b)
    except RuntimeError:
        return None


@lru_cache(maxsize=500000)
def _safe_morgan_fingerprint_bits(smiles: str, dim: int) -> np.ndarray | None:
    try:
        bits = _morgan_fingerprint_bits(smiles, n_bits=dim)
    except RuntimeError:
        return None
    if bits is None:
        return None
    return np.asarray(bits, dtype=np.float32)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def _candidate_pool(
    rows: Iterable[dict[str, str]],
    *,
    excluded_smiles: set[str] | None = None,
) -> list[dict[str, object]]:
    seen: set[str] = set()
    candidates = []
    excluded_smiles = excluded_smiles or set()
    for row in rows:
        smiles = _safe_canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
        if not smiles or smiles in seen or smiles in excluded_smiles:
            continue
        seen.add(smiles)
        scaffold = row.get("scaffold") or _safe_scaffold_smiles(smiles) or ""
        candidates.append(
            {
                "smiles": smiles,
                "scaffold": scaffold,
                "props": _target_props(row),
            }
        )
    return candidates


def _candidate_pool_from_molecule_db(
    path: Path,
    *,
    excluded_smiles: set[str] | None = None,
) -> list[dict[str, object]]:
    rows = _read_rows(path)
    seen: set[str] = set()
    candidates = []
    excluded_smiles = excluded_smiles or set()
    for row in rows:
        smiles = _safe_canonical_smiles(row.get("canonical_smiles", "")) or row.get("canonical_smiles", "")
        if not smiles or smiles in seen or smiles in excluded_smiles:
            continue
        seen.add(smiles)
        scaffold = row.get("scaffold") or _safe_scaffold_smiles(smiles) or ""
        props = {prop: _to_float(row.get(prop)) for prop in PROPERTY_COLUMNS}
        candidates.append(
            {
                "smiles": smiles,
                "scaffold": scaffold,
                "props": props,
            }
        )
    return candidates


def _sample_eval_rows_by_property_count(rows: list[dict[str, str]], limit: int, *, seed: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    by_count: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_count[int(float(row.get("property_count", 0) or 0))].append(row)
    sampled = []
    for property_count in sorted(by_count):
        group = by_count[property_count]
        if len(group) <= limit:
            sampled.extend(group)
            continue
        rng = random.Random(seed + property_count * 1009)
        indices = sorted(rng.sample(range(len(group)), limit))
        sampled.extend(group[idx] for idx in indices)
    return sampled


def _decode_row(
    row: dict[str, str],
    *,
    method: str,
    candidates: list[dict[str, object]],
    by_scaffold: Mapping[str, list[dict[str, object]]],
    feature_context: dict[str, object] | None,
    edit_latent_context: dict[str, object] | None,
    max_global_candidates: int,
    max_feature_candidates: int,
    max_edit_latent_candidates: int,
    rerank_candidates: int,
    rerank_property_weight: float,
    edit_latent_property_weight: float,
    edit_latent_delta_weight: float,
    edit_latent_direction_weight: float,
    edit_latent_fingerprint_weight: float,
    edit_latent_source_similarity_weight: float,
    scaffold_fallback_mode: str,
    compute_tanimoto: bool,
    seed: int,
) -> dict[str, object]:
    selected_props = _selected_props(row)
    source_props = _source_props(row)
    target_props = _target_props(row)
    if method == "source_identity":
        generated_smiles = _safe_canonical_smiles(row.get("source_smiles", "")) or row.get("source_smiles", "")
        generated_scaffold = row.get("scaffold", "")
        generated_props = source_props
        fallback = ""
    elif method == "target_oracle":
        generated_smiles = _safe_canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
        generated_scaffold = row.get("scaffold", "")
        generated_props = target_props
        fallback = ""
    elif method == "scaffold_property_retrieval":
        pool = by_scaffold.get(row.get("scaffold", ""), [])
        fallback = ""
        if not pool:
            pool, fallback = _scaffold_fallback_pool(
                row,
                candidates=candidates,
                mode=scaffold_fallback_mode,
                max_global_candidates=max_global_candidates,
                seed=seed,
            )
        candidate = _best_candidate(pool, selected_props=selected_props, target_props=target_props)
        generated_smiles = str(candidate.get("smiles", ""))
        generated_scaffold = str(candidate.get("scaffold", ""))
        generated_props = dict(candidate.get("props", {}))
    elif method == "global_property_retrieval":
        pool = _stable_sample(candidates, max_global_candidates, key=row.get("condition_id", ""), seed=seed)
        candidate = _best_candidate(pool, selected_props=selected_props, target_props=target_props)
        generated_smiles = str(candidate.get("smiles", ""))
        generated_scaffold = str(candidate.get("scaffold", ""))
        generated_props = dict(candidate.get("props", {}))
        fallback = ""
    elif method in EDIT_LATENT_METHODS:
        if edit_latent_context is None:
            raise ValueError(f"{method} requires edit latent context")
        scaffold_only = method in {"edit_latent_scaffold_retrieval", "edit_latent_scaffold_source_rerank"}
        use_source_similarity = method == "edit_latent_scaffold_source_rerank"
        if scaffold_only:
            pool = by_scaffold.get(row.get("scaffold", ""), [])
            fallback = ""
            if not pool:
                pool, fallback = _scaffold_fallback_pool(
                    row,
                    candidates=candidates,
                    mode=scaffold_fallback_mode,
                    max_global_candidates=max_global_candidates,
                    seed=seed,
                )
        else:
            pool = _stable_sample(candidates, max_global_candidates, key=row.get("condition_id", ""), seed=seed)
            fallback = ""
        pool = _stable_sample(pool, max_edit_latent_candidates, key=row.get("condition_id", ""), seed=seed)
        candidate, edit_fallback = _best_edit_latent_candidate(
            row,
            pool,
            edit_latent_context=edit_latent_context,
            selected_props=selected_props,
            source_props=source_props,
            property_weight=edit_latent_property_weight,
            delta_weight=edit_latent_delta_weight,
            direction_weight=edit_latent_direction_weight,
            fingerprint_weight=edit_latent_fingerprint_weight,
            source_similarity_weight=edit_latent_source_similarity_weight if use_source_similarity else 0.0,
        )
        fallback = ",".join(item for item in (fallback, edit_fallback) if item)
        generated_smiles = str(candidate.get("smiles", ""))
        generated_scaffold = str(candidate.get("scaffold", ""))
        generated_props = dict(candidate.get("props", {}))
    elif method in PURE_FEATURE_METHODS:
        if feature_context is None:
            raise ValueError(f"{method} requires condition feature context")
        scaffold_only = method == "vlm_scaffold_feature_retrieval"
        candidate, fallback = _best_feature_candidate(
            row,
            feature_context=feature_context,
            scaffold_only=scaffold_only,
            max_candidates=max_feature_candidates,
            scaffold_fallback_mode=scaffold_fallback_mode,
            seed=seed,
        )
        generated_smiles = str(candidate.get("smiles", ""))
        generated_scaffold = str(candidate.get("scaffold", ""))
        generated_props = dict(candidate.get("props", {}))
    elif method in HYBRID_RERANK_METHODS:
        if feature_context is None:
            raise ValueError(f"{method} requires condition feature context")
        scaffold_only = method == "scaffold_property_vlm_rerank"
        if scaffold_only:
            pool = by_scaffold.get(row.get("scaffold", ""), [])
            fallback = ""
            if not pool:
                pool, fallback = _scaffold_fallback_pool(
                    row,
                    candidates=candidates,
                    mode=scaffold_fallback_mode,
                    max_global_candidates=max_global_candidates,
                    seed=seed,
                )
        else:
            pool = _stable_sample(candidates, max_global_candidates, key=row.get("condition_id", ""), seed=seed)
            fallback = ""
        property_pool = _top_property_candidates(
            pool,
            selected_props=selected_props,
            target_props=target_props,
            limit=rerank_candidates,
        )
        candidate, rerank_fallback = _rerank_property_candidates_by_feature(
            row,
            property_pool,
            feature_context=feature_context,
            selected_props=selected_props,
            target_props=target_props,
            property_weight=rerank_property_weight,
        )
        fallback = ",".join(item for item in (fallback, rerank_fallback) if item)
        generated_smiles = str(candidate.get("smiles", ""))
        generated_scaffold = str(candidate.get("scaffold", ""))
        generated_props = dict(candidate.get("props", {}))
    else:
        raise ValueError(f"Unsupported method: {method}")

    property_successes = []
    errors = {}
    for prop in selected_props:
        target = target_props[prop]
        actual = float(generated_props.get(prop, math.nan))
        error = abs(actual - target) if not math.isnan(actual) else math.nan
        errors[prop] = error
        property_successes.append((not math.isnan(error)) and error <= SKETCHMOL_STRICT_TOLERANCE[prop])
    strict_success = bool(property_successes) and all(property_successes)
    scaffold_match = bool(generated_scaffold and row.get("scaffold") and generated_scaffold == row.get("scaffold"))
    joint_success = strict_success and scaffold_match

    out: dict[str, object] = {
        "method": method,
        "condition_id": row.get("condition_id", ""),
        "pair_id": row.get("pair_id", ""),
        "split": row.get("split", ""),
        "property_count": int(float(row.get("property_count", 0) or 0)),
        "condition_properties": ",".join(selected_props),
        "source_smiles": row.get("source_smiles", ""),
        "target_smiles": row.get("target_smiles", ""),
        "generated_smiles": generated_smiles,
        "source_scaffold": row.get("scaffold", ""),
        "generated_scaffold": generated_scaffold,
        "scaffold_match": scaffold_match,
        "strict_success": strict_success,
        "joint_success": joint_success,
        "fallback": fallback,
    }
    if compute_tanimoto:
        source_tanimoto = _safe_morgan_tanimoto(row.get("source_smiles", ""), generated_smiles)
        target_tanimoto = _safe_morgan_tanimoto(row.get("target_smiles", ""), generated_smiles)
        out["source_tanimoto"] = source_tanimoto if source_tanimoto is not None else ""
        out["target_tanimoto"] = target_tanimoto if target_tanimoto is not None else ""
    for prop in PROPERTY_COLUMNS:
        if prop in selected_props:
            out[f"{prop}_target"] = target_props[prop]
            out[f"{prop}_actual"] = generated_props.get(prop, "")
            out[f"{prop}_abs_error"] = errors[prop]
            out[f"{prop}_success"] = prop in selected_props and errors[prop] <= SKETCHMOL_STRICT_TOLERANCE[prop]
        else:
            out[f"{prop}_target"] = ""
            out[f"{prop}_actual"] = ""
            out[f"{prop}_abs_error"] = ""
            out[f"{prop}_success"] = ""
    return out


def _source_candidate(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "smiles": _safe_canonical_smiles(row.get("source_smiles", "")) or row.get("source_smiles", ""),
        "scaffold": row.get("scaffold", ""),
        "props": _source_props(row),
    }


def _empty_candidate() -> dict[str, object]:
    return {"smiles": "", "scaffold": "", "props": {}}


def _scaffold_fallback_pool(
    row: Mapping[str, str],
    *,
    candidates: list[dict[str, object]],
    mode: str,
    max_global_candidates: int,
    seed: int,
) -> tuple[list[dict[str, object]], str]:
    if mode == "global":
        return _stable_sample(candidates, max_global_candidates, key=row.get("condition_id", ""), seed=seed), "global"
    if mode == "source_identity":
        return [_source_candidate(row)], "source_identity"
    if mode == "empty":
        return [_empty_candidate()], "empty_scaffold_pool"
    raise ValueError(f"Unsupported scaffold fallback mode: {mode}")


def _selected_props(row: Mapping[str, str]) -> list[str]:
    props = [prop for prop in (row.get("condition_properties") or "").split(",") if prop]
    return [prop for prop in props if prop in PROPERTY_COLUMNS]


def _source_props(row: Mapping[str, str]) -> dict[str, float]:
    return {prop: _to_float(row.get(f"source_{prop}")) for prop in PROPERTY_COLUMNS}


def _target_props(row: Mapping[str, str]) -> dict[str, float]:
    return {prop: _to_float(row.get(f"target_{prop}")) for prop in PROPERTY_COLUMNS}


def _best_candidate(
    pool: list[dict[str, object]],
    *,
    selected_props: list[str],
    target_props: Mapping[str, float],
) -> dict[str, object]:
    if not pool:
        return {"smiles": "", "scaffold": "", "props": {}}
    best = None
    best_score = float("inf")
    for candidate in pool:
        props = candidate.get("props", {})
        score = 0.0
        for prop in selected_props:
            target = target_props[prop]
            actual = float(props.get(prop, math.nan)) if isinstance(props, Mapping) else math.nan
            if math.isnan(actual):
                score += 1e6
            else:
                score += abs(actual - target) / SKETCHMOL_STRICT_TOLERANCE[prop]
        score /= max(1, len(selected_props))
        if score < best_score:
            best_score = score
            best = candidate
    return best or pool[0]


def _property_score(
    candidate: Mapping[str, object],
    *,
    selected_props: list[str],
    target_props: Mapping[str, float],
) -> float:
    props = candidate.get("props", {})
    score = 0.0
    for prop in selected_props:
        target = target_props[prop]
        actual = float(props.get(prop, math.nan)) if isinstance(props, Mapping) else math.nan
        if math.isnan(actual):
            score += 1e6
        else:
            score += abs(actual - target) / SKETCHMOL_STRICT_TOLERANCE[prop]
    return score / max(1, len(selected_props))


def _top_property_candidates(
    pool: list[dict[str, object]],
    *,
    selected_props: list[str],
    target_props: Mapping[str, float],
    limit: int,
) -> list[dict[str, object]]:
    if not pool:
        return []
    ranked = sorted(
        pool,
        key=lambda candidate: _property_score(candidate, selected_props=selected_props, target_props=target_props),
    )
    return ranked[: max(1, int(limit))]


def _build_feature_context(
    train_rows: list[dict[str, str]],
    *,
    condition_features_dir: Path,
    array_name: str,
    variant: str,
    excluded_smiles: set[str],
) -> dict[str, object]:
    features_by_condition_id = _load_condition_features(condition_features_dir, array_name=array_name, variant=variant)
    candidates = []
    seen: set[tuple[str, str]] = set()
    features_by_smiles: dict[str, np.ndarray] = {}
    for row in train_rows:
        condition_id = row.get("condition_id", "")
        feature = features_by_condition_id.get(condition_id)
        if feature is None:
            continue
        smiles = _safe_canonical_smiles(row.get("target_smiles", "")) or row.get("target_smiles", "")
        if not smiles or smiles in excluded_smiles:
            continue
        key = (condition_id, smiles)
        if key in seen:
            continue
        seen.add(key)
        scaffold = row.get("scaffold") or _safe_scaffold_smiles(smiles) or ""
        candidates.append(
            {
                "condition_id": condition_id,
                "smiles": smiles,
                "scaffold": scaffold,
                "props": _target_props(row),
                "feature": feature,
            }
        )
        features_by_smiles.setdefault(smiles, feature)
    by_scaffold: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        by_scaffold[str(candidate["scaffold"])].append(candidate)
    return {
        "features_by_condition_id": features_by_condition_id,
        "features_by_smiles": features_by_smiles,
        "candidates": candidates,
        "by_scaffold": by_scaffold,
    }


def _load_condition_features(feature_dir: Path, *, array_name: str, variant: str) -> dict[str, np.ndarray]:
    import numpy as np

    array_path = feature_dir / ("query_tokens.npy" if array_name == "query_tokens" else "pooled.npy")
    raw_features = np.load(array_path)
    features = raw_features.reshape(raw_features.shape[0], -1)
    with (feature_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    if features.shape[0] != len(index_rows):
        raise ValueError(
            f"Feature row mismatch: {array_path} has {features.shape[0]} rows, "
            f"index.csv has {len(index_rows)} rows"
        )
    out: dict[str, np.ndarray] = {}
    for index_row, feature in zip(index_rows, features.astype(np.float32)):
        if index_row.get("variant") != variant:
            continue
        condition_id = index_row.get("condition_id", "")
        if not condition_id:
            continue
        norm = float(np.linalg.norm(feature))
        out[condition_id] = feature / norm if norm > 0 else feature
    return out


def _build_edit_latent_context(
    *,
    condition_rows: list[dict[str, str]],
    edit_latent_dir: Path,
    variant: str,
) -> dict[str, object]:
    import numpy as np

    array_path = edit_latent_dir / "edit_latent_predictions.npy"
    index_path = edit_latent_dir / "index.csv"
    if not array_path.exists():
        raise FileNotFoundError(f"Missing edit latent predictions: {array_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing edit latent index: {index_path}")
    predictions = np.load(array_path)
    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    if predictions.shape[0] != len(index_rows):
        raise ValueError(
            f"Edit latent row mismatch: {array_path} has {predictions.shape[0]} rows, "
            f"index.csv has {len(index_rows)} rows"
        )
    fingerprint_path = edit_latent_dir / "edit_latent_fingerprints.npy"
    fingerprints = None
    if fingerprint_path.exists():
        fingerprints = np.load(fingerprint_path)
        if fingerprints.shape[0] != len(index_rows):
            raise ValueError(
                f"Edit latent fingerprint row mismatch: {fingerprint_path} has {fingerprints.shape[0]} rows, "
                f"index.csv has {len(index_rows)} rows"
            )
    known_condition_ids = {row.get("condition_id", "") for row in condition_rows}
    predictions_by_condition_id: dict[str, np.ndarray] = {}
    fingerprints_by_condition_id: dict[str, np.ndarray] = {}
    for row_idx, (index_row, prediction) in enumerate(zip(index_rows, predictions.astype(np.float32))):
        if index_row.get("variant") != variant:
            continue
        condition_id = index_row.get("condition_id", "")
        if not condition_id or condition_id not in known_condition_ids:
            continue
        predictions_by_condition_id.setdefault(condition_id, prediction)
        if fingerprints is not None:
            fingerprints_by_condition_id.setdefault(condition_id, fingerprints[row_idx].astype(np.float32))
    return {
        "predictions_by_condition_id": predictions_by_condition_id,
        "fingerprints_by_condition_id": fingerprints_by_condition_id,
        "fingerprint_dim": int(fingerprints.shape[1]) if fingerprints is not None and fingerprints.ndim == 2 else 0,
        "array_path": str(array_path),
        "fingerprint_path": str(fingerprint_path) if fingerprints is not None else None,
        "index_path": str(index_path),
    }


def _best_edit_latent_candidate(
    row: dict[str, str],
    pool: list[dict[str, object]],
    *,
    edit_latent_context: dict[str, object],
    selected_props: list[str],
    source_props: Mapping[str, float],
    property_weight: float,
    delta_weight: float,
    direction_weight: float,
    fingerprint_weight: float,
    source_similarity_weight: float,
) -> tuple[dict[str, object], str]:
    if not pool:
        return _empty_candidate(), "empty_edit_latent_pool"
    predictions_by_condition_id = edit_latent_context["predictions_by_condition_id"]
    if not isinstance(predictions_by_condition_id, Mapping):
        raise TypeError("Invalid edit latent context: predictions_by_condition_id")
    condition_id = row.get("condition_id", "")
    latent = predictions_by_condition_id.get(condition_id)
    if latent is None:
        return _source_candidate(row), "missing_edit_latent"
    target_values, delta_values, _active_mask, directions = _unpack_edit_latent(np.asarray(latent, dtype=np.float32))
    fingerprints_by_condition_id = edit_latent_context.get("fingerprints_by_condition_id", {})
    predicted_fingerprint = None
    if fingerprint_weight > 0 and isinstance(fingerprints_by_condition_id, Mapping):
        predicted_fingerprint = fingerprints_by_condition_id.get(condition_id)
    fingerprint_dim = int(edit_latent_context.get("fingerprint_dim") or 0)

    scored = []
    for candidate in pool:
        props = candidate.get("props", {})
        if not isinstance(props, Mapping):
            continue
        property_score = 0.0
        delta_score = 0.0
        direction_score = 0.0
        valid = True
        for prop in selected_props:
            prop_idx = PROPERTY_COLUMNS.index(prop)
            actual = _to_float(props.get(prop))
            if math.isnan(actual):
                valid = False
                break
            source_value = float(source_props[prop])
            actual_delta = actual - source_value
            tolerance = SKETCHMOL_STRICT_TOLERANCE[prop]
            property_score += abs(actual - float(target_values[prop_idx])) / tolerance
            delta_score += abs(actual_delta - float(delta_values[prop_idx])) / tolerance
            predicted_direction = float(directions[prop_idx])
            if abs(predicted_direction) > 1e-4 and abs(actual_delta) > 1e-4:
                direction_score += 0.0 if predicted_direction * actual_delta >= 0 else 1.0
        if not valid:
            continue
        denom = max(1, len(selected_props))
        score = (
            float(property_weight) * (property_score / denom)
            + float(delta_weight) * (delta_score / denom)
            + float(direction_weight) * (direction_score / denom)
        )
        if predicted_fingerprint is not None and fingerprint_dim > 0:
            candidate_fingerprint = _safe_morgan_fingerprint_bits(str(candidate.get("smiles", "")), fingerprint_dim)
            if candidate_fingerprint is None:
                score += float(fingerprint_weight) * 1e6
            else:
                score += float(fingerprint_weight) * (
                    1.0 - _cosine_similarity(np.asarray(predicted_fingerprint, dtype=np.float32), candidate_fingerprint)
                )
        if source_similarity_weight > 0:
            similarity = _safe_morgan_tanimoto(row.get("source_smiles", ""), str(candidate.get("smiles", ""))) or 0.0
            score += float(source_similarity_weight) * (1.0 - similarity)
        scored.append((score, candidate))
    if not scored:
        return _empty_candidate(), "empty_edit_latent_scores"
    return min(scored, key=lambda item: item[0])[1], ""


def _unpack_edit_latent(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    prop_count = len(PROPERTY_COLUMNS)
    flat = latent.reshape(-1).astype(np.float32)
    if flat.shape[0] >= 4 * prop_count:
        return (
            flat[:prop_count],
            flat[prop_count : 2 * prop_count],
            flat[2 * prop_count : 3 * prop_count],
            flat[3 * prop_count : 4 * prop_count],
        )
    if flat.shape[0] >= 3 * prop_count:
        target_values = flat[:prop_count]
        active_mask = flat[prop_count : 2 * prop_count]
        directions = flat[2 * prop_count : 3 * prop_count]
        return target_values, np.zeros(prop_count, dtype=np.float32), active_mask, directions
    raise ValueError(f"Edit latent prediction is too small: expected at least {3 * prop_count}, got {flat.shape[0]}")


def _best_feature_candidate(
    row: dict[str, str],
    *,
    feature_context: dict[str, object],
    scaffold_only: bool,
    max_candidates: int,
    scaffold_fallback_mode: str,
    seed: int,
) -> tuple[dict[str, object], str]:
    import numpy as np

    features_by_condition_id = feature_context["features_by_condition_id"]
    if not isinstance(features_by_condition_id, Mapping):
        raise TypeError("Invalid feature context: features_by_condition_id")
    query = features_by_condition_id.get(row.get("condition_id", ""))
    if query is None:
        return {"smiles": "", "scaffold": "", "props": {}}, "missing_query_feature"

    if scaffold_only:
        by_scaffold = feature_context["by_scaffold"]
        if not isinstance(by_scaffold, Mapping):
            raise TypeError("Invalid feature context: by_scaffold")
        pool = list(by_scaffold.get(row.get("scaffold", ""), []))
        fallback = ""
        if not pool:
            if scaffold_fallback_mode == "global":
                pool = list(feature_context["candidates"])
                fallback = "global"
            elif scaffold_fallback_mode == "source_identity":
                return _source_candidate(row), "source_identity"
            elif scaffold_fallback_mode == "empty":
                return _empty_candidate(), "empty_scaffold_pool"
            else:
                raise ValueError(f"Unsupported scaffold fallback mode: {scaffold_fallback_mode}")
    else:
        pool = list(feature_context["candidates"])
        fallback = ""
    if not pool:
        return {"smiles": "", "scaffold": "", "props": {}}, fallback or "empty_pool"
    pool = _stable_sample(pool, max_candidates, key=row.get("condition_id", ""), seed=seed)
    matrix = np.stack([np.asarray(candidate["feature"], dtype=np.float32) for candidate in pool])
    scores = matrix @ np.asarray(query, dtype=np.float32)
    best_idx = int(np.argmax(scores))
    return pool[best_idx], fallback


def _rerank_property_candidates_by_feature(
    row: dict[str, str],
    pool: list[dict[str, object]],
    *,
    feature_context: dict[str, object],
    selected_props: list[str],
    target_props: Mapping[str, float],
    property_weight: float,
) -> tuple[dict[str, object], str]:
    import numpy as np

    if not pool:
        return {"smiles": "", "scaffold": "", "props": {}}, "empty_property_pool"
    features_by_condition_id = feature_context["features_by_condition_id"]
    features_by_smiles = feature_context["features_by_smiles"]
    if not isinstance(features_by_condition_id, Mapping) or not isinstance(features_by_smiles, Mapping):
        raise TypeError("Invalid feature context")
    query = features_by_condition_id.get(row.get("condition_id", ""))
    if query is None:
        return pool[0], "missing_query_feature"

    scored = []
    for candidate in pool:
        smiles = str(candidate.get("smiles", ""))
        feature = features_by_smiles.get(smiles)
        if feature is None:
            continue
        feature_score = float(np.asarray(feature, dtype=np.float32) @ np.asarray(query, dtype=np.float32))
        property_penalty = _property_score(candidate, selected_props=selected_props, target_props=target_props)
        scored.append((feature_score - float(property_weight) * property_penalty, candidate))
    if not scored:
        return pool[0], "missing_candidate_features"
    return max(scored, key=lambda item: item[0])[1], ""


def _stable_sample(items: list[dict[str, object]], limit: int, *, key: str, seed: int) -> list[dict[str, object]]:
    if limit <= 0 or len(items) <= limit:
        return items
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    indices = rng.sample(range(len(items)), limit)
    return [items[idx] for idx in indices]


def _summarize(
    rows: list[dict[str, object]],
    *,
    source_tanimoto_thresholds: list[float] | None = None,
) -> list[dict[str, object]]:
    source_tanimoto_thresholds = source_tanimoto_thresholds or []
    out = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for property_count in range(2, 8):
            selected = [row for row in method_rows if int(row.get("property_count", 0) or 0) == property_count]
            if not selected:
                continue
            strict = _fraction(_to_bool(row.get("strict_success")) for row in selected)
            scaffold = _fraction(_to_bool(row.get("scaffold_match")) for row in selected)
            joint = _fraction(_to_bool(row.get("joint_success")) for row in selected)
            reference = SKETCHMOL_REFERENCE_MULTI_PROPERTY.get(property_count, math.nan)
            fallback_stats = _fallback_stats(selected)
            source_similarity_stats = _source_similarity_stats(
                selected,
                thresholds=source_tanimoto_thresholds,
            )
            summary = {
                "family": "understanding_multiproperty_direct",
                "method": method,
                "benchmark_task": "multi_property_direct",
                "benchmark_label": f"{property_count}_properties",
                "property_count": property_count,
                "n": len(selected),
                "success_rate_strict_in_valid_mols": strict,
                "scaffold_match_rate": scaffold,
                "joint_success_rate": joint,
                **source_similarity_stats,
                **fallback_stats,
                "sketchmol_reference_strict": reference,
                "strict_margin_vs_sketchmol": strict - reference if not math.isnan(reference) else "",
            }
            for prop in PROPERTY_COLUMNS:
                errors = [_to_float(row.get(f"{prop}_abs_error")) for row in selected if row.get(f"{prop}_abs_error") != ""]
                if errors:
                    summary[f"{prop}_mae"] = sum(errors) / len(errors)
            out.append(summary)
        if method_rows:
            strict = _fraction(_to_bool(row.get("strict_success")) for row in method_rows)
            scaffold = _fraction(_to_bool(row.get("scaffold_match")) for row in method_rows)
            joint = _fraction(_to_bool(row.get("joint_success")) for row in method_rows)
            fallback_stats = _fallback_stats(method_rows)
            source_similarity_stats = _source_similarity_stats(
                method_rows,
                thresholds=source_tanimoto_thresholds,
            )
            out.append(
                {
                    "family": "understanding_multiproperty_direct",
                    "method": method,
                    "benchmark_task": "multi_property_direct",
                    "benchmark_label": "all",
                    "property_count": "all",
                    "n": len(method_rows),
                    "success_rate_strict_in_valid_mols": strict,
                    "scaffold_match_rate": scaffold,
                    "joint_success_rate": joint,
                    **source_similarity_stats,
                    **fallback_stats,
                    "sketchmol_reference_strict": "",
                    "strict_margin_vs_sketchmol": "",
                }
            )
    return out


def _write_report(path: Path, summary_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    source_tanimoto_thresholds = _parse_float_list(args.source_tanimoto_thresholds)
    lines = [
        "# Multi-Property Direct Benchmark",
        "",
        (
            "This report compares source-image/property-condition retrieval baselines against "
            "the SketchMol multi-property strict-success reference, then adds source-similarity "
            "metrics for source-conditioned editing."
        ),
        "",
        f"- condition rows: `{args.condition_rows_csv}`",
        f"- eval split: `{args.eval_split}`",
        f"- candidate source: `{'molecule_database' if args.candidate_molecule_db_csv else 'train_target_rows'}`",
        f"- candidate molecule DB: `{args.candidate_molecule_db_csv}`",
        f"- max eval rows per property count: `{args.max_eval_per_property_count}`",
        f"- eval target candidates excluded from retrieval pool: `{not args.allow_eval_target_candidates}`",
        f"- scaffold fallback mode: `{args.scaffold_fallback_mode}`",
        f"- source Tanimoto thresholds: `{','.join(str(value) for value in source_tanimoto_thresholds)}`",
    ]
    if args.condition_features_dir is not None:
        lines.extend(
            [
                f"- condition features: `{args.condition_features_dir}`",
                f"- condition feature array: `{args.condition_feature_array}`",
                f"- condition feature variant: `{args.condition_feature_variant}`",
            ]
        )
    if args.edit_latent_dir is not None:
        lines.extend(
            [
                f"- edit latent predictions: `{args.edit_latent_dir}`",
                (
                    "- edit latent scorer weights: "
                    f"property={args.edit_latent_property_weight}, "
                    f"delta={args.edit_latent_delta_weight}, "
                    f"direction={args.edit_latent_direction_weight}, "
                    f"fingerprint={args.edit_latent_fingerprint_weight}, "
                    f"source_similarity={args.edit_latent_source_similarity_weight}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Strict Property Success",
            "",
            "This table is directly comparable to the SketchMol structured multi-property reference.",
            "",
            "| method | 2p | 3p | 4p | 5p | 6p | 7p | scaffold all | joint all |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    methods = [method for method in METHODS if any(row.get("method") == method for row in summary_rows)]
    rows_by_key = {(row["method"], row["property_count"]): row for row in summary_rows}
    for method in methods:
        values = []
        for count in range(2, 8):
            row = rows_by_key.get((method, count))
            values.append(_fmt(row.get("success_rate_strict_in_valid_mols")) if row else "")
        all_row = rows_by_key.get((method, "all"))
        values.append(_fmt(all_row.get("scaffold_match_rate")) if all_row else "")
        values.append(_fmt(all_row.get("joint_success_rate")) if all_row else "")
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines.extend(
        [
            f"| SketchMol structured reference | {_fmt(0.804)} | {_fmt(0.768)} | {_fmt(0.736)} | {_fmt(0.716)} | {_fmt(0.678)} | {_fmt(0.685)} |  |  |",
            "",
            "## Source-Similarity-Constrained Success",
            "",
            (
                "`strict@Tanimoto>=t` means strict property success AND "
                "Morgan fingerprint Tanimoto(source, generated) >= t. "
                "This is the main source-preservation metric for source-conditioned edit."
            ),
            "",
            "| method | mean source Tani | median source Tani | "
            + " | ".join(f"strict@{threshold:g}" for threshold in source_tanimoto_thresholds)
            + " |",
            "| --- | ---: | ---: | " + " | ".join("---:" for _ in source_tanimoto_thresholds) + " |",
        ]
    )
    for method in methods:
        all_row = rows_by_key.get((method, "all"))
        if not all_row:
            continue
        values = [
            _fmt(all_row.get("mean_source_tanimoto")),
            _fmt(all_row.get("median_source_tanimoto")),
        ]
        for threshold in source_tanimoto_thresholds:
            values.append(_fmt(all_row.get(f"strict_success_at_source_tanimoto_ge_{_threshold_suffix(threshold)}")))
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## Scaffold-Match Diagnostics",
            "",
            "`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.",
            "",
            "| method | 2p | 3p | 4p | 5p | 6p | 7p | all |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        values = []
        for count in range(2, 8):
            row = rows_by_key.get((method, count))
            values.append(_fmt(row.get("joint_success_rate")) if row else "")
        all_row = rows_by_key.get((method, "all"))
        values.append(_fmt(all_row.get("joint_success_rate")) if all_row else "")
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## Fallback Diagnostics",
            "",
            "| method | global fallback | source fallback | empty fallback |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        all_row = rows_by_key.get((method, "all"))
        if not all_row:
            continue
        lines.append(
            f"| {method} | "
            f"{_fmt(all_row.get('global_fallback_fraction'))} | "
            f"{_fmt(all_row.get('source_identity_fallback_fraction'))} | "
            f"{_fmt(all_row.get('empty_fallback_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `target_oracle` is an upper bound because it returns the known target molecule.",
            "- `scaffold_property_retrieval` is the main strong non-generative baseline: it retrieves a candidate with the same scaffold and closest active-property values from the configured candidate library.",
            "- `edit_latent_global_retrieval` ranks candidates using predicted target/delta properties from the learned source-conditioned edit latent.",
            "- `edit_latent_scaffold_retrieval` applies that edit-latent scorer inside the same-scaffold candidate pool.",
            "- `edit_latent_scaffold_source_rerank` is the main proposed retrieval-style method: it uses the learned edit latent plus optional fingerprint/source-similarity reranking inside the scaffold-preserving pool.",
            "- `vlm_scaffold_feature_retrieval` retrieves same-scaffold train targets by nearest frozen VLM condition feature.",
            "- `vlm_feature_retrieval` retrieves train targets by nearest frozen VLM condition feature without scaffold filtering.",
            "- `global_property_vlm_rerank` first keeps the top property-matched candidates, then reranks them with Understanding-Condition features.",
            f"- `scaffold_property_vlm_rerank` applies the same rerank after same-scaffold property filtering. Unseen-scaffold fallback mode is `{args.scaffold_fallback_mode}`.",
            "- `global_property_retrieval` ignores source structure and can satisfy properties while failing source preservation.",
            "- High strict success with low source Tanimoto should be interpreted as property retrieval, not source-conditioned molecular editing.",
            "- A learned Understanding-Condition model should eventually improve strict property success at useful source-Tanimoto thresholds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
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


def _to_float(value: object) -> float:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return math.nan


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y"}


def _fraction(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for item in items if item) / len(items) if items else 0.0


def _fallback_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "global_fallback_fraction": _fraction("global" in str(row.get("fallback", "")).split(",") for row in rows),
        "source_identity_fallback_fraction": _fraction(
            "source_identity" in str(row.get("fallback", "")).split(",") for row in rows
        ),
        "empty_fallback_fraction": _fraction(
            "empty_scaffold_pool" in str(row.get("fallback", "")).split(",") for row in rows
        ),
    }


def _source_similarity_stats(rows: list[dict[str, object]], *, thresholds: list[float]) -> dict[str, float | str]:
    values = [_to_float(row.get("source_tanimoto")) for row in rows if row.get("source_tanimoto") != ""]
    values = [value for value in values if not math.isnan(value)]
    stats: dict[str, float | str] = {
        "mean_source_tanimoto": sum(values) / len(values) if values else "",
        "median_source_tanimoto": _median(values) if values else "",
    }
    for threshold in thresholds:
        suffix = _threshold_suffix(threshold)
        stats[f"source_tanimoto_ge_{suffix}_rate"] = (
            _fraction(value >= threshold for value in values) if values else ""
        )
        stats[f"strict_success_at_source_tanimoto_ge_{suffix}"] = (
            _fraction(
                _to_bool(row.get("strict_success")) and _to_float(row.get("source_tanimoto")) >= threshold
                for row in rows
            )
            if rows
            else ""
        )
    return stats


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _parse_float_list(text: str) -> list[float]:
    out = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0 or value > 1:
            raise ValueError(f"Source Tanimoto threshold must be in [0, 1], got {value}")
        out.append(value)
    return out


def _threshold_suffix(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "_")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
