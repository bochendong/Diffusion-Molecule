"""Evaluate generated molecules with deterministic instruction verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chem import canonicalize_smiles, molecular_descriptors, passes_druglike_filters, tanimoto
from .instruction_local_edit import local_edit_metrics
from .instruction_verifier import verify_instruction
from .io import save_json
from .progress import iter_progress


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    train_smiles = _load_train_smiles(args.train_smiles, args.train_dataset)
    metrics, detailed = evaluate_instruction_candidates(
        candidates,
        train_smiles=train_smiles,
        source_col=args.source_col,
        candidate_col=args.candidate_col,
        spec_col=args.spec_col,
    )
    save_json(metrics, args.out)
    if args.details_out:
        detailed.to_csv(args.details_out, index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate instruction-guided molecular editing candidates.")
    parser.add_argument("--candidates", required=True, help="CSV with source_smiles, candidate_smiles, instruction_spec_json.")
    parser.add_argument("--out", default="outputs/instruction_metrics.json")
    parser.add_argument("--details-out", default=None)
    parser.add_argument("--train-smiles", default=None, help="Optional CSV containing training SMILES for novelty.")
    parser.add_argument("--train-dataset", default=None, help="Optional instruction dataset; train split source/target smiles are used for novelty.")
    parser.add_argument("--source-col", default="source_smiles")
    parser.add_argument("--candidate-col", default="candidate_smiles")
    parser.add_argument("--spec-col", default="instruction_spec_json")
    return parser.parse_args()


def evaluate_instruction_candidates(
    candidates: pd.DataFrame,
    train_smiles: set[str] | None = None,
    source_col: str = "source_smiles",
    candidate_col: str = "candidate_smiles",
    spec_col: str = "instruction_spec_json",
) -> tuple[dict[str, float], pd.DataFrame]:
    required = {source_col, candidate_col, spec_col}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    details = []
    valid_smiles = []
    for idx, row in iter_progress(candidates.iterrows(), total=len(candidates), label="verifying candidates"):
        source = str(row[source_col])
        candidate = str(row[candidate_col])
        result = verify_instruction(source, candidate, row[spec_col])
        out = row.to_dict()
        out.update({f"verify_{key}": value for key, value in result.items()})
        if "target_smiles" in candidates.columns:
            local = local_edit_metrics(source, str(row["target_smiles"]), candidate)
            out.update({f"local_{key}": value for key, value in local.items()})
        out["instruction_row"] = int(idx)
        details.append(out)
        if result["valid"]:
            record = molecular_descriptors(candidate)
            if record.valid:
                valid_smiles.append(record.smiles)

    detailed = pd.DataFrame(details)
    if detailed.empty:
        return {"n": 0.0, "validity": 0.0}, detailed

    valid_mask = detailed["verify_valid"].astype(bool)
    valid = detailed[valid_mask]
    unique_valid = sorted(set(valid_smiles))
    train = train_smiles or set()
    train_canonical = _canonical_smiles_set(train)
    exact_train_hits = [canonicalize_smiles(smi) in train_canonical for smi in detailed[candidate_col].fillna("").astype(str)]
    metrics = {
        "n": float(len(detailed)),
        "validity": float(valid_mask.mean()),
        "goal_success_rate": _mean_bool(detailed["verify_goal_success"]),
        "constraint_success_rate": _mean_bool(detailed["verify_constraint_success"]),
        "edit_success_rate": _mean_bool(detailed["verify_edit_success"]),
        "overall_instruction_success_rate": _mean_bool(detailed["verify_overall_success"]),
        "similarity_to_source": float(valid["verify_similarity_to_source"].mean()) if not valid.empty else 0.0,
        "druglike_rate": _druglike_rate(valid[candidate_col].astype(str).tolist()) if not valid.empty else 0.0,
        "uniqueness": float(len(unique_valid) / max(1, len(valid_smiles))),
        "novelty": float(len([s for s in unique_valid if s not in train_canonical]) / max(1, len(unique_valid))),
        "exact_train_hit_rate": float(np.mean(exact_train_hits)) if exact_train_hits else 0.0,
    }
    metrics.update(_sampled_train_overlap_metrics(unique_valid, train_canonical))
    if "local_target_candidate_similarity" in detailed.columns:
        metrics.update(_local_edit_summary(detailed, valid))
    if not valid.empty:
        metrics.update(
            {
                "goal_success_rate_in_valid_mols": _mean_bool(valid["verify_goal_success"]),
                "constraint_success_rate_in_valid_mols": _mean_bool(valid["verify_constraint_success"]),
                "edit_success_rate_in_valid_mols": _mean_bool(valid["verify_edit_success"]),
                "overall_instruction_success_rate_in_valid_mols": _mean_bool(valid["verify_overall_success"]),
            }
        )
    if "pair_id" in detailed.columns:
        metrics["conditions"] = float(detailed["pair_id"].nunique())
        metrics["mean_candidates_per_condition"] = float(detailed.groupby("pair_id").size().mean())
        metrics["pairs"] = float(detailed["pair_id"].nunique())
        metrics["mean_candidates_per_pair"] = float(detailed.groupby("pair_id").size().mean())
    if "instruction_id" in detailed.columns:
        metrics["instructions"] = float(detailed["instruction_id"].nunique())
        metrics["mean_candidates_per_instruction"] = float(detailed.groupby("instruction_id").size().mean())
        if "conditions" not in metrics:
            metrics["conditions"] = metrics["instructions"]
            metrics["mean_candidates_per_condition"] = metrics["mean_candidates_per_instruction"]
    metrics.update(_topk_metrics(detailed))
    return metrics, detailed


def _load_train_smiles(train_smiles_path: str | None, train_dataset_path: str | None) -> set[str]:
    smiles: set[str] = set()
    if train_smiles_path:
        df = pd.read_csv(train_smiles_path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        smiles.update(df[col].dropna().astype(str).tolist())
    if train_dataset_path:
        df = pd.read_csv(train_dataset_path)
        train = df[df.get("split", "train") == "train"] if "split" in df.columns else df
        for col in ("source_smiles", "target_smiles"):
            if col in train.columns:
                smiles.update(train[col].dropna().astype(str).tolist())
    return smiles


def _canonical_smiles_set(smiles: set[str] | list[str]) -> set[str]:
    out = set()
    for smi in smiles:
        can = canonicalize_smiles(str(smi))
        if can:
            out.add(can)
    return out


def _sampled_train_overlap_metrics(candidate_smiles: list[str], train_smiles: set[str], max_candidates: int = 256, max_train: int = 1024) -> dict[str, float]:
    candidate_set = set()
    for smi in candidate_smiles:
        can = canonicalize_smiles(smi)
        if can:
            candidate_set.add(can)
    candidates = sorted(candidate_set)
    train = sorted(train_smiles)
    if not candidates or not train:
        return {
            "sampled_nearest_train_tanimoto": 0.0,
            "sampled_nearest_train_tanimoto_ge_0_90": 0.0,
            "sampled_novelty_at_tanimoto_0_90": 0.0,
        }
    rng = np.random.default_rng(31)
    if len(candidates) > max_candidates:
        candidates = [candidates[int(i)] for i in rng.choice(len(candidates), size=max_candidates, replace=False)]
    if len(train) > max_train:
        train = [train[int(i)] for i in rng.choice(len(train), size=max_train, replace=False)]
    nearest = []
    for candidate in candidates:
        if candidate in train_smiles:
            nearest.append(1.0)
            continue
        sims = [tanimoto(candidate, train_smi) for train_smi in train if train_smi != candidate]
        nearest.append(max(sims) if sims else 1.0)
    ge_090 = float(np.mean([sim >= 0.90 for sim in nearest])) if nearest else 0.0
    return {
        "sampled_nearest_train_tanimoto": float(np.mean(nearest)) if nearest else 0.0,
        "sampled_nearest_train_tanimoto_ge_0_90": ge_090,
        "sampled_novelty_at_tanimoto_0_90": float(1.0 - ge_090),
    }


def _mean_bool(values: Any) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.asarray(values, dtype=bool).mean())


def _druglike_rate(smiles: list[str]) -> float:
    records = [molecular_descriptors(s) for s in smiles]
    valid = [r for r in records if r.valid]
    return float(np.mean([passes_druglike_filters(r.descriptors) for r in valid])) if valid else 0.0


def _local_edit_summary(detailed: pd.DataFrame, valid: pd.DataFrame) -> dict[str, float]:
    frame = valid if not valid.empty else detailed
    return {
        "target_similarity": float(frame["local_target_candidate_similarity"].mean()) if not frame.empty else 0.0,
        "source_preservation_core_fraction": float(frame["local_source_candidate_core_fraction"].mean()) if not frame.empty else 0.0,
        "local_preservation_success_rate": _mean_bool(detailed["local_local_preservation_success"]),
        "target_recovery_success_rate": _mean_bool(detailed["local_target_recovery_success"]),
        "sketchmol_local_edit_success_rate": _mean_bool(detailed["local_sketchmol_local_edit_success"]),
        "candidate_equals_source_rate": _mean_bool(detailed["local_candidate_equals_source"]),
        "candidate_equals_target_rate": _mean_bool(detailed["local_candidate_equals_target"]),
    }


def _topk_metrics(detailed: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    group_cols = []
    if "instruction_id" in detailed.columns:
        group_cols.append(("instruction", "instruction_id"))
    if "pair_id" in detailed.columns:
        group_cols.append(("pair", "pair_id"))
    success_cols = {
        "overall": "verify_overall_success",
        "goal": "verify_goal_success",
        "constraint": "verify_constraint_success",
        "edit": "verify_edit_success",
    }
    if "local_sketchmol_local_edit_success" in detailed.columns:
        success_cols["local_edit"] = "local_sketchmol_local_edit_success"
    if not group_cols:
        return metrics

    for label, group_col in group_cols:
        ordered = _sort_for_topk(detailed)
        group_sizes = ordered.groupby(group_col).size()
        for k in (1, 5, 10, 20):
            if group_sizes.empty or int(group_sizes.max()) < k:
                continue
            top = ordered.groupby(group_col, sort=False).head(k)
            for metric_name, success_col in success_cols.items():
                if success_col not in top.columns:
                    continue
                value = top.groupby(group_col)[success_col].any().mean()
                metrics[f"{metric_name}_success_at_{k}_by_{label}"] = float(value)
        for metric_name, success_col in success_cols.items():
            if success_col not in ordered.columns:
                continue
            value = ordered.groupby(group_col)[success_col].any().mean()
            metrics[f"{metric_name}_success_at_all_by_{label}"] = float(value)
    return metrics


def _sort_for_topk(detailed: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    ascending = []
    for col in ("sample_idx", "rank"):
        if col in detailed.columns:
            sort_cols.append(col)
            ascending.append(True)
    if "decoder_score" in detailed.columns:
        sort_cols.append("decoder_score")
        ascending.append(True)
    if "instruction_row" in detailed.columns:
        sort_cols.append("instruction_row")
        ascending.append(True)
    if not sort_cols:
        return detailed
    return detailed.sort_values(sort_cols, ascending=ascending, kind="mergesort")


if __name__ == "__main__":
    main()
