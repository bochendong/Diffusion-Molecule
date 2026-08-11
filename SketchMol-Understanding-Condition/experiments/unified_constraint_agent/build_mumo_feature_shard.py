#!/usr/bin/env python3
"""Materialize one compact ECFP4/property-label shard from MuMO train rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402
from sketchmol_understanding_condition.chem import canonical_smiles, rdkit_version  # noqa: E402


DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "rotatable",
    "QED",
    "fraction_csp3",
    "ring_count",
    "heavy_atoms",
    "formal_charge",
    "hetero_atoms",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args(argv)


def molecule_features(smiles: str, *, radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray] | None:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    fingerprint = generator.GetFingerprint(molecule)
    bits = np.zeros(int(n_bits), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    descriptors = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            QED.qed(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
            molecule.GetNumHeavyAtoms(),
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
            rdMolDescriptors.CalcNumHeteroatoms(molecule),
        ],
        dtype=np.float32,
    )
    return bits, descriptors


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = protocol.read_jsonl(args.input_jsonl)
    property_index = {prop: index for index, prop in enumerate(protocol.PROPERTIES)}
    fingerprints: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    smiles_values: list[str] = []
    source_groups: list[str] = []
    task_ids: list[str] = []
    partitions: list[str] = []
    roles: list[str] = []
    pair_ids: list[str] = []
    outcomes: Counter[str] = Counter()
    observed_labels: Counter[str] = Counter()

    for raw in rows:
        for role in ("source", "target"):
            smiles = canonical_smiles(str(raw.get(f"{role}_smiles", "") or ""))
            if not smiles:
                outcomes[f"invalid_{role}"] += 1
                continue
            feature = molecule_features(
                smiles,
                radius=int(args.fingerprint_radius),
                n_bits=int(args.fingerprint_bits),
            )
            if feature is None:
                outcomes[f"feature_failure_{role}"] += 1
                continue
            y = np.full(len(protocol.PROPERTIES), np.nan, dtype=np.float32)
            for prop, index in property_index.items():
                value = export.read_property_value(raw, prop, prefix=role)
                if value is not None:
                    y[index] = float(value)
                    observed_labels[prop] += 1
            fingerprints.append(feature[0])
            descriptors.append(feature[1])
            labels.append(y)
            smiles_values.append(smiles)
            source_groups.append(str(raw.get("_uca_source_group", "")))
            task_ids.append(str(raw.get("_uca_task_id", "")))
            partitions.append(str(raw.get("_uca_partition", "")))
            roles.append(role)
            pair_ids.append(str(raw.get("_uca_pair_digest", "")))
            outcomes[f"{role}_feature_rows"] += 1

    if not fingerprints:
        raise ValueError(f"No valid molecular features were built from {args.input_jsonl}")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        fingerprint=np.stack(fingerprints),
        descriptors=np.stack(descriptors),
        labels=np.stack(labels),
        smiles=np.asarray(smiles_values, dtype=str),
        source_group=np.asarray(source_groups, dtype=str),
        task_id=np.asarray(task_ids, dtype=str),
        partition=np.asarray(partitions, dtype=str),
        role=np.asarray(roles, dtype=str),
        pair_id=np.asarray(pair_ids, dtype=str),
        property_names=np.asarray(protocol.PROPERTIES, dtype=str),
        descriptor_names=np.asarray(DESCRIPTOR_NAMES, dtype=str),
    )
    manifest = {
        "protocol": protocol.PROTOCOL_VERSION,
        "stage": "feature_shard",
        "data_role": "fit_and_dev_train_only_labels",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "input": str(args.input_jsonl),
        "input_rows": len(rows),
        "output_rows": len(fingerprints),
        "fingerprint": {"kind": "ECFP4", "radius": int(args.fingerprint_radius), "bits": int(args.fingerprint_bits)},
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "rdkit_version": rdkit_version(),
        "observed_labels": dict(sorted(observed_labels.items())),
        "outcomes": dict(sorted(outcomes.items())),
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
