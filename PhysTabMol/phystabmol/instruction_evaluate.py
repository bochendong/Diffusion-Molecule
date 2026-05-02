"""Evaluate generated molecules with deterministic instruction verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chem import molecular_descriptors, passes_druglike_filters
from .instruction_verifier import verify_instruction
from .io import save_json


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
    for idx, row in candidates.iterrows():
        source = str(row[source_col])
        candidate = str(row[candidate_col])
        result = verify_instruction(source, candidate, row[spec_col])
        out = row.to_dict()
        out.update({f"verify_{key}": value for key, value in result.items()})
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
        "novelty": float(len([s for s in unique_valid if s not in train]) / max(1, len(unique_valid))),
    }
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
    elif "instruction_id" in detailed.columns:
        metrics["conditions"] = float(detailed["instruction_id"].nunique())
        metrics["mean_candidates_per_condition"] = float(detailed.groupby("instruction_id").size().mean())
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


def _mean_bool(values: Any) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.asarray(values, dtype=bool).mean())


def _druglike_rate(smiles: list[str]) -> float:
    records = [molecular_descriptors(s) for s in smiles]
    valid = [r for r in records if r.valid]
    return float(np.mean([passes_druglike_filters(r.descriptors) for r in valid])) if valid else 0.0


if __name__ == "__main__":
    main()
