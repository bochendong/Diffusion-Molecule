#!/usr/bin/env python3
"""Fail-closed audit of the matched Table1 and 2p--7p inference arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ""


def aggregate_table(path: Path, key: str) -> float:
    rows = read_csv(path)
    total = sum(float(row["n"]) for row in rows)
    return sum(float(row[key]) * float(row["n"]) for row in rows) / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--p811-audit", required=True, type=Path)
    parser.add_argument("--edit-candidates", required=True, type=Path)
    parser.add_argument("--edit-sampling-summary", required=True, type=Path)
    parser.add_argument("--table1-output", required=True, type=Path)
    parser.add_argument("--support-audit", required=True, type=Path)
    parser.add_argument("--denovo-candidates", required=True, type=Path)
    parser.add_argument("--denovo-summaries", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint_hash = sha256(args.checkpoint)
    p811 = read_json(args.p811_audit)
    edit_summary = read_json(args.edit_sampling_summary)
    support = read_json(args.support_audit)
    denovo_summaries = [read_json(path) for path in args.denovo_summaries]
    edit_rows = read_csv(args.edit_candidates)
    denovo_rows = read_csv(args.denovo_candidates)

    edit_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    valid = identity = 0
    unique_by_condition = []
    for row in edit_rows:
        condition_id = str(row.get("condition_id") or row.get("sample_id") or "")
        edit_grouped[condition_id].append(row)
        candidate = canon(row.get("generated_smiles"))
        source = canon(row.get("source_smiles"))
        valid += int(bool(candidate))
        identity += int(bool(candidate and source and candidate == source))
    for rows in edit_grouped.values():
        unique_by_condition.append(len({canon(row.get("generated_smiles")) for row in rows if canon(row.get("generated_smiles"))}) / 20)

    strict_any = {str(k): aggregate_table(args.table1_output / f"any{k}" / "moledit_table_summary.csv", "Acc_all(0.65)") for k in (1, 8, 20)}
    relaxed_any = {str(k): aggregate_table(args.table1_output / f"any{k}" / "moledit_table_summary.csv", "Acc_all(0.15)") for k in (1, 8, 20)}
    validity_any = {str(k): aggregate_table(args.table1_output / f"any{k}" / "moledit_table_summary.csv", "Validity") for k in (1, 8, 20)}
    candidate_table = args.table1_output / "candidate20" / "moledit_table_summary.csv"
    hashes = [
        checkpoint_hash,
        str(p811.get("checkpoint_sha256")),
        str(edit_summary.get("checkpoint_sha256")),
        *[str(item.get("checkpoint_sha256")) for item in denovo_summaries],
    ]
    rerank_flags = [
        bool(edit_summary.get("property_reranking")),
        *[bool(item.get("property_reranking")) for item in denovo_summaries],
    ]
    denovo_counts = Counter(int(float(row.get("property_count") or 0)) for row in denovo_rows)
    forbidden = {"target_smiles", "target_scaffold", "target_image"}
    payload = {
        "protocol": "p8_2_matched_inference_audit_v1",
        "checkpoint_sha256": checkpoint_hash,
        "arm_checkpoint_hashes": hashes,
        "both_arms_checkpoint_sha_exact": len(set(hashes)) == 1,
        "p811_source_aware_single_decoder": bool(p811.get("checks", {}).get("source_aware_single_decoder")),
        "p811_legacy_denovo_parameters_bit_exact": bool(p811.get("checks", {}).get("legacy_denovo_parameters_bit_exact")),
        "property_reranking_flags": rerank_flags,
        "property_reranking": any(rerank_flags),
        "edit_target_molecule_used_at_inference": bool(edit_summary.get("target_molecule_used_at_inference")),
        "denovo_structural_target_columns_present": sorted(forbidden.intersection(denovo_rows[0])),
        "support_eval_target_molecule_used_at_inference": bool(support.get("eval_target_molecule_used_at_inference")),
        "table1_conditions": len(edit_grouped),
        "table1_candidate_rows": len(edit_rows),
        "table1_candidate_validity": valid / max(len(edit_rows), 1),
        "table1_identity_fraction": identity / max(len(edit_rows), 1),
        "table1_unique_fraction": sum(unique_by_condition) / max(len(unique_by_condition), 1),
        "table1_validity_any": validity_any,
        "table1_strict_any": strict_any,
        "table1_relaxed_any": relaxed_any,
        "table1_strict_candidate_fraction": aggregate_table(candidate_table, "Acc_all(0.65)"),
        "table1_relaxed_candidate_fraction": aggregate_table(candidate_table, "Acc_all(0.15)"),
        "denovo_conditions": len({str(row.get("condition_id") or row.get("sample_id") or "") for row in denovo_rows}),
        "denovo_candidate_rows": len(denovo_rows),
        "denovo_property_count_candidates": {str(k): v for k, v in sorted(denovo_counts.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = (
        payload["both_arms_checkpoint_sha_exact"]
        and payload["p811_source_aware_single_decoder"]
        and payload["p811_legacy_denovo_parameters_bit_exact"]
        and not payload["property_reranking"]
        and not payload["edit_target_molecule_used_at_inference"]
        and not payload["support_eval_target_molecule_used_at_inference"]
        and not payload["denovo_structural_target_columns_present"]
        and payload["table1_conditions"] == 200
        and payload["table1_candidate_rows"] == 4000
        and payload["denovo_conditions"] == 6000
        and payload["denovo_candidate_rows"] == 120000
        and payload["denovo_property_count_candidates"] == {str(k): 20000 for k in range(2, 8)}
    )
    if not passed:
        raise SystemExit("P8.2 matched inference audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
