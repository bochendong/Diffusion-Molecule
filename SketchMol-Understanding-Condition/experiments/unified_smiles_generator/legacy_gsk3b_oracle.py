#!/usr/bin/env python3
"""Prepare and use the benchmark-era TDC GSK3B random-forest oracle.

TDC's original GSK3B pickle was trained with scikit-learn 0.21.3.  Modern TDC
silently selects a different ``gsk3b_current.pkl`` model, while directly
loading the original pickle fails because sklearn's tree ABI changed.  This
module performs one explicit, provenance-recorded conversion:

* preserve every split and leaf count from the official legacy forest;
* add the modern ``missing_go_to_left`` node field with the legacy behavior;
* normalize leaf class counts as sklearn 0.21 did in ``predict_proba``.

The converted artifact is then a regular modern sklearn estimator.  It is
selected only when ``SUCC_GSK3B_ORACLE_PATH`` is set, so benchmark provenance
is explicit rather than dependent on the installed TDC/sklearn version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np


ORACLE_ENV = "SUCC_GSK3B_ORACLE_PATH"
DRD2_ORACLE_ENV = "SUCC_DRD2_ORACLE_PATH"
PINNED_ORACLE_ENVS = {"GSK3B": ORACLE_ENV, "DRD2": DRD2_ORACLE_ENV}
LEGACY_URL = "https://dataverse.harvard.edu/api/access/datafile/4170295"
LEGACY_SHA256 = "18d1cc9bb9498e4bae0755842080558cb2d0444ecb51beffa0ef58a6d760b74b"
KNOWN_ACTIVE = "Nc1nonc1-c1nc2cc(O)ccc2n1C1CCC1"
KNOWN_NEGATIVES = ("CCO", "c1ccccc1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MorganClassifierOracle:
    """Apply an sklearn binary classifier to TDC's ECFP4/2048 features."""

    def __init__(self, model: object):
        self.model = model

    def __call__(self, smiles: str) -> float:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        molecule = Chem.MolFromSmiles(str(smiles or ""))
        if molecule is None:
            raise ValueError("invalid SMILES")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        features = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, features)
        probability = self.model.predict_proba(features.reshape(1, -1))  # type: ignore[attr-defined]
        return float(probability[0, 1])


def load_oracle(path: Path) -> MorganClassifierOracle:
    with path.open("rb") as handle:
        return MorganClassifierOracle(pickle.load(handle))


def configured_oracle() -> MorganClassifierOracle | None:
    return configured_oracle_for("GSK3B")


def configured_oracle_for(prop: str) -> MorganClassifierOracle | None:
    canonical = str(prop or "").strip().upper()
    env_name = PINNED_ORACLE_ENVS.get(canonical)
    configured = str(os.environ.get(env_name, "") or "").strip() if env_name else ""
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return load_oracle(path)


def configured_provenance() -> dict[str, object]:
    configured = str(os.environ.get(ORACLE_ENV, "") or "").strip()
    if not configured:
        return {}
    path = Path(configured).expanduser().resolve()
    result: dict[str, object] = {
        "property": "GSK3B",
        "implementation": "tdc_legacy_sklearn_0.21.3_converted",
        "path": str(path),
        "sha256": sha256_file(path),
        "source_url": LEGACY_URL,
        "source_sha256": LEGACY_SHA256,
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if manifest_path.is_file():
        result["conversion_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return result


def configured_provenance_for(prop: str) -> dict[str, object]:
    canonical = str(prop or "").strip().upper()
    if canonical == "GSK3B":
        return configured_provenance()
    env_name = PINNED_ORACLE_ENVS.get(canonical)
    configured = str(os.environ.get(env_name, "") or "").strip() if env_name else ""
    if not configured:
        return {}
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return {
        "property": canonical,
        "implementation": "pinned_ecfp4_2048_sklearn_classifier",
        "path": str(path),
        "sha256": sha256_file(path),
    }


class _LegacyTree:
    def __new__(cls, *args: object):
        instance = object.__new__(cls)
        instance.args = args
        return instance

    def __setstate__(self, state: dict[str, object]) -> None:
        self.state = state


class _LegacyForestUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[no-untyped-def]
        if module == "sklearn.tree._tree" and name == "Tree":
            return _LegacyTree
        return super().find_class(module, name)


def _fill_missing_defaults(estimator: object, default: object) -> None:
    for key, value in vars(default).items():
        if not hasattr(estimator, key):
            setattr(estimator, key, value)


def convert_legacy_model(source: Path, output: Path) -> dict[str, object]:
    import sklearn
    import sklearn.ensemble._forest as forest_module
    import sklearn.tree._classes as tree_classes
    import sklearn.tree._tree as tree_module
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    source_hash = sha256_file(source)
    if source_hash != LEGACY_SHA256:
        raise ValueError(f"legacy GSK3B SHA256 mismatch: expected {LEGACY_SHA256}, got {source_hash}")

    # The pickle refers to module paths removed after sklearn 0.21.
    sys.modules["sklearn.ensemble.forest"] = forest_module
    sys.modules["sklearn.tree.tree"] = tree_classes
    with source.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _LegacyForestUnpickler(handle).load()

    for estimator in model.estimators_:
        legacy_tree = estimator.tree_
        old_nodes = legacy_tree.state["nodes"]
        nodes = np.zeros(old_nodes.shape, dtype=tree_module.NODE_DTYPE)
        for field_name in old_nodes.dtype.names:
            nodes[field_name] = old_nodes[field_name]

        # sklearn 0.21 normalized class counts per leaf before averaging trees.
        values = legacy_tree.state["values"].astype(float)
        totals = values.sum(axis=2, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
        state = dict(legacy_tree.state, nodes=nodes, values=values)
        tree = tree_module.Tree(*legacy_tree.args)
        tree.__setstate__(state)
        estimator.tree_ = tree
        _fill_missing_defaults(estimator, DecisionTreeClassifier())
        estimator.n_features_in_ = estimator.n_features_

    _fill_missing_defaults(model, RandomForestClassifier())
    model.n_features_in_ = model.n_features_
    model.estimator_ = model.base_estimator_
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(model, handle, protocol=4)

    scores = verify_model(output)
    manifest = {
        "protocol": "tdc_legacy_gsk3b_sklearn_conversion_v1",
        "source_url": LEGACY_URL,
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "source_sklearn_version": "0.21.3",
        "conversion_sklearn_version": sklearn.__version__,
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "sanity_scores": scores,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_model(path: Path) -> dict[str, float]:
    oracle = load_oracle(path)
    scores = {"known_active": oracle(KNOWN_ACTIVE)}
    scores.update({f"known_negative_{index + 1}": oracle(smiles) for index, smiles in enumerate(KNOWN_NEGATIVES)})
    if scores["known_active"] < 0.5:
        raise ValueError(f"GSK3B known-active sanity failed: {scores['known_active']:.6f} < 0.5")
    if max(scores[key] for key in scores if key.startswith("known_negative_")) > 0.1:
        raise ValueError(f"GSK3B known-negative sanity failed: {scores}")
    return scores


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-pickle", required=True, type=Path)
    prepare.add_argument("--output-pickle", required=True, type=Path)
    prepare.add_argument("--download", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--model", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        if not args.source_pickle.is_file():
            if not args.download:
                raise FileNotFoundError(args.source_pickle)
            args.source_pickle.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(LEGACY_URL, args.source_pickle)
        print(json.dumps(convert_legacy_model(args.source_pickle, args.output_pickle), indent=2, sort_keys=True))
    else:
        result = {
            "protocol": "tdc_legacy_gsk3b_oracle_preflight_v1",
            "model": str(args.model.resolve()),
            "sha256": sha256_file(args.model),
            "scores": verify_model(args.model),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
