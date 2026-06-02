"""Phase 5A-5 retrieval diagnostic for oracle-conditioned SMILES decoding.

This diagnostic does not train a new model. It reads an existing Phase 5A-4 run
and compares the neural beam candidates with nearest-neighbor train-pool
retrieval under the same oracle fingerprint condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .audit_pairs import _load_rdkit
from .phase5a0_oracle_baseline import _canonicalize, _fraction, _write_rows
from .phase5a1_learned_smiles_decoder import (
    _canonical_candidate_list,
    _fingerprint_tanimoto,
    _load_numpy,
    _make_fingerprint_fn,
    _scaffold_match,
    _set_rdkit_error_logging,
    _tanimoto,
)


def run_retrieval_diagnostic(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    fingerprint_bits: int = 2048,
    retrieval_top_k: int = 16,
    max_eval: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "retrieval_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs_path = run_dir / "train_pairs.csv"
    predictions_path = run_dir / "predictions.csv"
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"Missing train split from source run: {train_pairs_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions from source run: {predictions_path}")

    rdkit = _load_rdkit()
    if not rdkit:
        raise RuntimeError("RDKit is required for Phase 5A-5 retrieval diagnostics.")
    _set_rdkit_error_logging(enabled=False)
    np = _load_numpy()
    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits)
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(fingerprint_bits))

    train_smiles = _unique_canonical_smiles(_read_rows(train_pairs_path), rdkit=rdkit)
    train_records = _build_train_records(train_smiles, rdkit=rdkit, generator=generator)
    if not train_records:
        raise RuntimeError("No valid train fingerprints available for retrieval diagnostics.")
    train_fps = [record["fp"] for record in train_records]
    train_record_smiles = [record["smiles"] for record in train_records]

    prediction_rows = _read_rows(predictions_path)
    if max_eval is not None:
        prediction_rows = prediction_rows[: int(max_eval)]

    detail_rows: list[dict[str, Any]] = []
    source_rows: dict[str, list[dict[str, Any]]] = {"beam": [], "retrieval": [], "beam_plus_retrieval": []}
    for row_idx, prediction in enumerate(prediction_rows):
        target_smiles = prediction.get("target_smiles", "")
        target_feature = fingerprint_fn(target_smiles)
        target_fp = _fingerprint_bitvect(target_smiles, rdkit=rdkit, generator=generator)
        if target_feature is None:
            continue

        beam_candidates = _candidate_list_from_prediction(prediction, rdkit=rdkit)
        retrieval_candidates, retrieval_scores = _nearest_train_candidates(
            target_fp,
            train_fps=train_fps,
            train_smiles=train_record_smiles,
            data_structs=DataStructs,
            top_k=retrieval_top_k,
        )
        hybrid_candidates = _rerank_by_oracle_fingerprint(
            _dedupe(beam_candidates + retrieval_candidates),
            condition_feature=target_feature,
            fingerprint_fn=fingerprint_fn,
            np=np,
        )

        candidates_by_source = {
            "beam": beam_candidates,
            "retrieval": retrieval_candidates,
            "beam_plus_retrieval": hybrid_candidates,
        }
        for source, candidates in candidates_by_source.items():
            source_rows[source].append(
                _score_candidates(
                    pair_id=prediction.get("pair_id", str(row_idx)),
                    source=source,
                    target_smiles=target_smiles,
                    candidates=candidates,
                    train_smiles=set(train_smiles),
                    rdkit=rdkit,
                )
            )

        detail_rows.append(
            {
                "pair_id": prediction.get("pair_id", str(row_idx)),
                "target_smiles": target_smiles,
                "beam_top1": beam_candidates[0] if beam_candidates else "",
                "retrieval_top1": retrieval_candidates[0] if retrieval_candidates else "",
                "hybrid_top1": hybrid_candidates[0] if hybrid_candidates else "",
                "beam_candidates": "|".join(beam_candidates),
                "retrieval_candidates": "|".join(retrieval_candidates),
                "hybrid_candidates": "|".join(hybrid_candidates),
                "retrieval_condition_tanimotos": "|".join(f"{score:.6f}" for score in retrieval_scores),
            }
        )

    source_summary = [_summarize_source(source, rows) for source, rows in source_rows.items()]
    _write_rows(output_dir / "retrieval_diagnostic_rows.csv", detail_rows)
    _write_rows(output_dir / "source_summary.csv", source_summary)
    (output_dir / "source_summary.json").write_text(json.dumps(source_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {
        "phase": "phase5a5_oracle_retrieval_diagnostic",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "fingerprint_bits": float(fingerprint_bits),
        "retrieval_top_k": float(retrieval_top_k),
        "max_eval": float(max_eval) if max_eval is not None else "",
        "train_pool_size": float(len(train_records)),
        "eval_rows": float(sum(len(rows) for rows in source_rows.values()) / max(1, len(source_rows))),
        "source_summary": str(output_dir / "source_summary.csv"),
        "detail_rows": str(output_dir / "retrieval_diagnostic_rows.csv"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unique_canonical_smiles(rows: list[dict[str, str]], rdkit: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    smiles_list: list[str] = []
    for row in rows:
        raw_smiles = row.get("canonical_smiles") or row.get("input_smiles") or row.get("target_smiles") or ""
        smiles, _error = _canonicalize(raw_smiles, rdkit)
        if smiles and smiles not in seen:
            seen.add(smiles)
            smiles_list.append(smiles)
    return smiles_list


def _build_train_records(smiles_list: list[str], rdkit: dict[str, Any], generator: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for smiles in smiles_list:
        fp = _fingerprint_bitvect(smiles, rdkit=rdkit, generator=generator)
        if fp is not None:
            records.append({"smiles": smiles, "fp": fp})
    return records


def _fingerprint_bitvect(smiles: str, rdkit: dict[str, Any], generator: Any) -> Any:
    mol = rdkit["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


def _candidate_list_from_prediction(prediction: dict[str, str], rdkit: dict[str, Any]) -> list[str]:
    text = prediction.get("canonical_candidates") or prediction.get("beam_canonical_candidates") or prediction.get("generated_smiles") or ""
    raw_candidates = [item for item in text.split("|") if item]
    return _canonical_candidate_list(raw_candidates, rdkit)


def _nearest_train_candidates(
    condition_fp: Any,
    train_fps: list[Any],
    train_smiles: list[str],
    data_structs: Any,
    top_k: int,
) -> tuple[list[str], list[float]]:
    if condition_fp is None:
        return [], []
    similarities = data_structs.BulkTanimotoSimilarity(condition_fp, train_fps)
    scored = [(float(score), index, train_smiles[index]) for index, score in enumerate(similarities)]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    top = scored[: max(1, int(top_k))]
    return [item[2] for item in top], [float(item[0]) for item in top]


def _rerank_by_oracle_fingerprint(
    candidates: list[str],
    condition_feature: Any,
    fingerprint_fn: Any,
    np: Any,
) -> list[str]:
    scored: list[tuple[float, int, str]] = []
    for index, smiles in enumerate(candidates):
        scored.append((_fingerprint_tanimoto(condition_feature, fingerprint_fn(smiles), np=np), -index, smiles))
    scored.sort(reverse=True)
    return [smiles for _score, _neg_index, smiles in scored]


def _score_candidates(
    pair_id: str,
    source: str,
    target_smiles: str,
    candidates: list[str],
    train_smiles: set[str],
    rdkit: dict[str, Any],
) -> dict[str, Any]:
    top1_smiles = candidates[0] if candidates else ""
    target_mol = rdkit["Chem"].MolFromSmiles(target_smiles)
    top1_mol = rdkit["Chem"].MolFromSmiles(top1_smiles) if top1_smiles else None
    tanimotos = [_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidates]
    return {
        "pair_id": pair_id,
        "source": source,
        "target_smiles": target_smiles,
        "top1_smiles": top1_smiles,
        "candidate_count": float(len(candidates)),
        "top1_exact_match": bool(top1_smiles == target_smiles),
        "topk_exact_match": bool(target_smiles in candidates),
        "top1_target_tanimoto": float(_tanimoto(target_mol, top1_mol, rdkit)),
        "mean_best_tanimoto": float(max(tanimotos, default=0.0)),
        "top1_scaffold_match": bool(_scaffold_match(target_smiles, top1_smiles, rdkit)),
        "top1_train_pool_member": bool(top1_smiles in train_smiles),
        "topk_has_train_pool_member": bool(any(smiles in train_smiles for smiles in candidates)),
    }


def _summarize_source(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "source": source,
        "n": float(total),
        "mean_candidate_count": _mean_float(rows, "candidate_count"),
        "top1_exact_match_fraction": _fraction(_count(rows, "top1_exact_match"), total),
        "topk_exact_match_fraction": _fraction(_count(rows, "topk_exact_match"), total),
        "top1_target_tanimoto": _mean_float(rows, "top1_target_tanimoto"),
        "mean_best_tanimoto": _mean_float(rows, "mean_best_tanimoto"),
        "top1_scaffold_match_fraction": _fraction(_count(rows, "top1_scaffold_match"), total),
        "top1_train_pool_member_fraction": _fraction(_count(rows, "top1_train_pool_member"), total),
        "topk_has_train_pool_member_fraction": _fraction(_count(rows, "topk_has_train_pool_member"), total),
    }


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(sum(values) / len(values)) if values else 0.0


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5A-5 retrieval diagnostics on an existing Phase 5A run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--retrieval-top-k", type=int, default=16)
    parser.add_argument("--max-eval", type=int, default=None)
    args = parser.parse_args()
    metrics = run_retrieval_diagnostic(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        fingerprint_bits=args.fingerprint_bits,
        retrieval_top_k=args.retrieval_top_k,
        max_eval=args.max_eval,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
