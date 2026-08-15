#!/usr/bin/env python3
"""Fail-closed benchmark oracles for the Table1 assay support audit.

The public TDC wrapper returns a default score when an evaluator raises.  That
is convenient for interactive use but unsafe for a support audit: a missing or
incompatible pickle can become an apparently complete column of zeros.  This
module instead pins the exact trusted artifacts, validates their digests, runs
known-positive/known-negative probes, and propagates every inference failure.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


GSK3B_ACTIVE = "Nc1nonc1-c1nc2cc(O)ccc2n1C1CCC1"
DRD2_ACTIVE = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
KNOWN_NEGATIVES = ("CCO", "c1ccccc1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifact(path: Path, specification: Mapping[str, object]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned {specification['property']} oracle is missing: {path}")
    size = path.stat().st_size
    expected_size = int(specification["bytes"])
    if size != expected_size:
        raise ValueError(
            f"Pinned {specification['property']} oracle size mismatch: "
            f"expected {expected_size}, got {size}"
        )
    digest = sha256_file(path)
    expected_digest = str(specification["sha256"])
    if digest != expected_digest:
        raise ValueError(
            f"Pinned {specification['property']} oracle SHA256 mismatch: "
            f"expected {expected_digest}, got {digest}"
        )
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest}


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.load(handle)


def _load_legacy_drd2_svc(path: Path) -> object:
    """Load Graph2Graph's sklearn-0.18 SVC under modern sklearn.

    Only compatibility names/attributes used by modern ``predict_proba`` are
    supplied; learned arrays and probabilities remain byte-for-byte those from
    the pinned public pickle.
    """

    import sklearn.svm._classes as svm_classes

    sys.modules.setdefault("sklearn.svm.classes", svm_classes)
    model = _load_pickle(path)
    old = vars(model)
    compatibility = {
        "_n_support": old.get("n_support_"),
        "_probA": old.get("probA_"),
        "_probB": old.get("probB_"),
    }
    for name, value in compatibility.items():
        if not hasattr(model, name) and value is not None:
            setattr(model, name, value)
    if not hasattr(model, "n_features_in_"):
        shape_fit = getattr(model, "shape_fit_", None)
        if not shape_fit or len(shape_fit) != 2:
            raise ValueError("Pinned DRD2 SVC is missing shape_fit_")
        model.n_features_in_ = int(shape_fit[1])
    if not hasattr(model, "break_ties"):
        model.break_ties = False
    if int(model.n_features_in_) != 2048:
        raise ValueError(f"Pinned DRD2 SVC expected 2048 features, got {model.n_features_in_}")
    return model


@dataclass
class PinnedAssayOracle:
    prop: str
    model: object
    artifact: dict[str, object]
    specification: dict[str, object]

    def _features(self, smiles_batch: Sequence[str]) -> np.ndarray:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.warning")
        features = np.zeros((len(smiles_batch), 2048), dtype=np.float32)
        for row_index, smiles in enumerate(smiles_batch):
            molecule = Chem.MolFromSmiles(str(smiles or ""))
            if molecule is None:
                raise ValueError(f"{self.prop} oracle received invalid SMILES: {smiles!r}")
            if self.prop == "GSK3B":
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    molecule, 2, nBits=2048
                )
                DataStructs.ConvertToNumpyArray(fingerprint, features[row_index])
            elif self.prop == "DRD2":
                fingerprint = AllChem.GetMorganFingerprint(
                    molecule, 3, useCounts=True, useFeatures=True
                )
                for index, count in fingerprint.GetNonzeroElements().items():
                    features[row_index, int(index) % 2048] += float(count)
            else:  # pragma: no cover - constructor contract
                raise ValueError(f"Unsupported assay oracle: {self.prop}")
        return features

    def score_many(self, smiles_batch: Sequence[str], *, batch_size: int = 256) -> list[float]:
        if batch_size <= 0:
            raise ValueError("oracle batch_size must be positive")
        scores: list[float] = []
        for start in range(0, len(smiles_batch), batch_size):
            batch = list(smiles_batch[start : start + batch_size])
            probabilities = self.model.predict_proba(self._features(batch))  # type: ignore[attr-defined]
            if probabilities.shape != (len(batch), 2):
                raise ValueError(
                    f"{self.prop} oracle returned shape {probabilities.shape}, "
                    f"expected {(len(batch), 2)}"
                )
            for raw_value in probabilities[:, 1]:
                value = float(raw_value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{self.prop} oracle returned invalid probability: {value}")
                scores.append(value)
        if len(scores) != len(smiles_batch):
            raise RuntimeError(f"{self.prop} oracle coverage failure")
        return scores

    def preflight(self) -> dict[str, object]:
        active = GSK3B_ACTIVE if self.prop == "GSK3B" else DRD2_ACTIVE
        probe_smiles = [active, *KNOWN_NEGATIVES]
        values = self.score_many(probe_smiles, batch_size=len(probe_smiles))
        active_score = values[0]
        negative_scores = values[1:]
        minimum_active = float(self.specification["known_active_min"])
        maximum_negative = float(self.specification["known_negative_max"])
        minimum_range = float(self.specification["minimum_probe_range"])
        probe_range = max(values) - min(values)
        failures = []
        if active_score < minimum_active:
            failures.append("known_active")
        if max(negative_scores) > maximum_negative:
            failures.append("known_negative")
        if probe_range < minimum_range:
            failures.append("probe_range")
        if failures:
            raise ValueError(
                f"Pinned {self.prop} oracle preflight failed {failures}: {values}"
            )
        return {
            "passed": True,
            "known_active_smiles": active,
            "known_active_score": active_score,
            "known_negative_scores": dict(zip(KNOWN_NEGATIVES, negative_scores)),
            "probe_range": probe_range,
            "thresholds": {
                "known_active_min": minimum_active,
                "known_negative_max": maximum_negative,
                "minimum_probe_range": minimum_range,
            },
        }


def load_pinned_oracles(
    *,
    gsk3b_path: Path,
    drd2_path: Path,
    specifications: Mapping[str, object],
) -> tuple[dict[str, PinnedAssayOracle], dict[str, object]]:
    paths = {"GSK3B": gsk3b_path, "DRD2": drd2_path}
    result: dict[str, PinnedAssayOracle] = {}
    provenance: dict[str, object] = {}
    for prop, path in paths.items():
        raw_specification = specifications.get(prop)
        if not isinstance(raw_specification, Mapping):
            raise ValueError(f"Missing preregistered {prop} oracle specification")
        specification = dict(raw_specification)
        if specification.get("property") != prop:
            raise ValueError(f"Preregistered {prop} oracle property drift")
        artifact = _validate_artifact(path, specification)
        model = _load_pickle(path) if prop == "GSK3B" else _load_legacy_drd2_svc(path)
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Pinned {prop} artifact has no predict_proba")
        oracle = PinnedAssayOracle(prop, model, artifact, specification)
        preflight = oracle.preflight()
        result[prop] = oracle
        provenance[prop] = {
            **artifact,
            "implementation": specification["implementation"],
            "source_url": specification["source_url"],
            "source_commit": specification.get("source_commit"),
            "git_blob_sha1": specification.get("git_blob_sha1"),
            "preflight": preflight,
        }
    return result, provenance
