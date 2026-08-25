#!/usr/bin/env python3
"""Add one leakage-free lower-order projection when the 12k pool loses a heldout row."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import p23_protocol as protocol


PROPERTIES = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")


def rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def target(row: dict[str, str]) -> str:
    return protocol.canonical_smiles(
        row.get("target_canonical_smiles") or row.get("target_smiles") or ""
    )


def heldout_targets(paths: Sequence[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        for row in rows(path):
            value = target(row)
            if value:
                output.add(value)
    return output


def project_two_property(row: dict[str, str]) -> dict[str, str]:
    selected = ("MW", "QED")
    output = dict(row)
    output.update({
        "sample_id": "p23_denovo_replacement_2p_000000",
        "condition_id": "p23_denovo_replacement_2p_000000",
        "variant_id": "p23_denovo_replacement_2p_000000:full",
        "split": "train",
        "property_count": "2",
        "condition_properties": ",".join(selected),
        "p23_projection_source": "train_candidate_7p_to_2p",
    })
    for prop in PROPERTIES:
        active = prop in selected
        output[f"{prop}_active"] = "True" if active else "False"
        output[f"{prop}_None"] = "False" if active else "True"
    output["MolWt_None"] = "False"
    output["QED_None"] = "False"
    for alias in ("logp_None", "TPSA_None", "HBD_None", "HBA_None", "rotatable_None"):
        output[alias] = "True"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--fallback", required=True, type=Path)
    parser.add_argument("--heldout", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    primary_rows = list(rows(args.primary))
    if not primary_rows:
        raise ValueError("primary de-novo pool is empty")
    primary_targets = {value for row in primary_rows if (value := target(row))}
    forbidden = primary_targets | heldout_targets(args.heldout)
    candidates = []
    for row in rows(args.fallback):
        value = target(row)
        if not value or value in forbidden:
            continue
        rank = hashlib.sha256(f"2323:2p:{value}".encode()).hexdigest()
        candidates.append((rank, row, value))
    if not candidates:
        raise ValueError("no leakage-free de-novo replacement candidate")
    _rank, donor, replacement_target = min(candidates, key=lambda item: item[0])
    replacement = project_two_property(donor)

    fieldnames = list(primary_rows[0])
    for key in replacement:
        if key not in fieldnames:
            fieldnames.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(primary_rows)
        writer.writerow(replacement)
    print({
        "primary_rows": len(primary_rows),
        "output_rows": len(primary_rows) + 1,
        "replacement_target_hash": protocol.smiles_hash(replacement_target),
        "projection": "7p_to_MW_QED_2p",
        "heldout_overlap": replacement_target in heldout_targets(args.heldout),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
