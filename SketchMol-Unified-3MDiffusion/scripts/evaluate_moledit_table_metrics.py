#!/usr/bin/env python3
"""Evaluate MolEdit-Instruct predictions with MolEditRL table-style metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np


def _ensure_rdkit_six_compat() -> None:
    """PyTDC 1.x imports rdkit.six, which RDKit >=2024 removed from easybuild modules."""
    if "rdkit.six" in sys.modules:
        return
    try:
        from rdkit.six import iteritems  # noqa: F401
    except ModuleNotFoundError:
        six_mod = types.ModuleType("rdkit.six")
        six_mod.iteritems = dict.items
        sys.modules["rdkit.six"] = six_mod


_ensure_rdkit_six_compat()


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_unified_3m_diffusion.unified_condition_dataset import TABLE1_TASK_SPECS  # noqa: E402


TASK_LABELS = {
    "GSK3B:increase": "GSK3B↑",
    "RB:decrease": "Rotbonds↓",
    "MW:increase": "MW↑",
    "SA:decrease": "SA↓",
    "HBA:decrease+SA:decrease": "Haccept↓ SA↓",
    "QED:increase+SA:decrease": "QED↑ SA↓",
    "HBA:decrease+LogP:increase": "Haccept↓ LogP↑",
    "HBA:decrease+MW:decrease": "Haccept↓ MW↓",
    "DRD2:decrease+MW:decrease+SA:decrease": "DRD2↓ MW↓ SA↓",
    "HBA:increase+MW:increase+QED:decrease": "Haccept↑ MW↑ QED↓",
}
TABLE1_TASK_ORDER = tuple(TASK_LABELS.keys())
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
    "sascorer": "SA",
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

PREDICTION_SMILES_COLUMNS = (
    "predicted_smiles",
    "generated_smiles",
    "prediction_smiles",
    "candidate_smiles",
    "output_smiles",
    "smiles",
)

ID_COLUMNS = ("example_id", "condition_id", "sample_id", "pair_hash")
PINNED_ORACLE_ENVS = {
    "GSK3B": "SUCC_GSK3B_ORACLE_PATH",
    "DRD2": "SUCC_DRD2_ORACLE_PATH",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PinnedMorganClassifierOracle:
    """Run a provenance-pinned sklearn assay model on its training features."""

    def __init__(self, model_path: Path, prop: str = "GSK3B"):
        self.model_path = model_path.resolve()
        self.prop = canonical_property(prop)
        with self.model_path.open("rb") as handle, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.prop == "DRD2":
                import sklearn.svm._classes as svm_classes

                sys.modules.setdefault("sklearn.svm.classes", svm_classes)
            self.model = pickle.load(handle)
        if self.prop == "DRD2":
            old = vars(self.model)
            for name, value in {
                "_n_support": old.get("n_support_"),
                "_probA": old.get("probA_"),
                "_probB": old.get("probB_"),
            }.items():
                if not hasattr(self.model, name) and value is not None:
                    setattr(self.model, name, value)
            if not hasattr(self.model, "n_features_in_"):
                shape_fit = getattr(self.model, "shape_fit_", None)
                if not shape_fit or len(shape_fit) != 2:
                    raise ValueError("Pinned DRD2 SVC is missing shape_fit_")
                self.model.n_features_in_ = int(shape_fit[1])
            if not hasattr(self.model, "break_ties"):
                self.model.break_ties = False

    def __call__(self, smiles: str) -> float:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        molecule = Chem.MolFromSmiles(str(smiles or ""))
        if molecule is None:
            raise ValueError("invalid SMILES")
        features = np.zeros(2048, dtype=np.float32)
        if self.prop == "DRD2":
            fingerprint = AllChem.GetMorganFingerprint(
                molecule, 3, useCounts=True, useFeatures=True
            )
            for index, count in fingerprint.GetNonzeroElements().items():
                features[int(index) % 2048] += float(count)
        else:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
            DataStructs.ConvertToNumpyArray(fingerprint, features)
        probability = self.model.predict_proba(features.reshape(1, -1))
        return float(probability[0, 1])


def configured_pinned_oracle(prop: str) -> PinnedMorganClassifierOracle | None:
    env_name = PINNED_ORACLE_ENVS.get(canonical_property(prop))
    configured = str(os.environ.get(env_name, "") or "").strip() if env_name else ""
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return PinnedMorganClassifierOracle(path, prop=prop)


def configured_oracle_provenance() -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    for prop, env_name in PINNED_ORACLE_ENVS.items():
        configured = str(os.environ.get(env_name, "") or "").strip()
        if not configured:
            continue
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        provenance[prop] = {
            "implementation": "pinned_ecfp4_2048_sklearn_classifier",
            "env": env_name,
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Enhanced MolEdit split CSV/JSONL.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction CSV or benchmark_decoded.csv.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="Unified3M")
    parser.add_argument("--method", default=None, help="Optional method filter for benchmark_decoded.csv.")
    parser.add_argument("--thresholds", default="0.65,0.15")
    parser.add_argument("--task-filter", choices=("table1", "all"), default="table1")
    parser.add_argument(
        "--include-empty-table1",
        action="store_true",
        help="Write one row for every MolEditRL Table1 task, even when no predictions were evaluated.",
    )
    parser.add_argument(
        "--require-table1-coverage",
        action="store_true",
        help="Fail if any Table1 task lacks references or evaluated prediction rows.",
    )
    parser.add_argument(
        "--missing-oracle-policy",
        choices=("fail", "skip-task", "mark-false"),
        default="fail",
        help=(
            "What to do when TDC oracle tasks such as GSK3B/DRD2/SA are unavailable. "
            "Use skip-task only for smoke/debug runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds:
        raise SystemExit("--thresholds must contain at least one value")

    chem = Chemistry()
    references = load_references(args.reference)
    predictions = load_predictions(args.predictions, model_name=args.model_name, method=args.method)
    rows = []
    reference_counts = count_reference_tasks(references.values(), task_filter=args.task_filter)
    skipped_missing_oracle: dict[str, int] = defaultdict(int)
    coverage_failures: list[str] = []

    model_names = sorted(predictions) or [args.model_name]
    for model_name in model_names:
        pred_rows = predictions.get(model_name, [])
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        generated_by_task: dict[str, list[str]] = defaultdict(list)
        reference_by_task: dict[str, list[str]] = defaultdict(list)
        matched_predictions_by_task: dict[str, int] = defaultdict(int)
        for pred in pred_rows:
            ref = references.get(pred["example_id"])
            if ref is None:
                continue
            task_specs = task_specs_for_reference(ref)
            current_task_key = task_key(task_specs)
            if args.task_filter == "table1" and current_task_key not in TASK_LABELS:
                continue
            matched_predictions_by_task[current_task_key] += 1
            missing_oracles = sorted(chem.missing_oracles(task_specs))
            if missing_oracles and args.missing_oracle_policy == "skip-task":
                skipped_missing_oracle[current_task_key] += 1
                continue
            if missing_oracles and args.missing_oracle_policy == "fail":
                raise SystemExit(
                    f"Missing TDC oracle(s) for task {current_task_key}: {', '.join(missing_oracles)}. "
                    "Install TDC or rerun with --missing-oracle-policy skip-task."
                )
            evaluated = evaluate_prediction(ref, pred["predicted_smiles"], task_specs, chem=chem, thresholds=thresholds)
            evaluated["task_key"] = current_task_key
            grouped[current_task_key].append(evaluated)
            if evaluated["valid"]:
                generated_by_task[current_task_key].append(pred["predicted_smiles"])
                reference_by_task[current_task_key].append(str(ref.get("target_smiles", "")))

        for key in sorted(grouped, key=task_sort_key):
            task_rows = grouped[key]
            summary = summarize_task(
                task_rows,
                thresholds=thresholds,
                fcd=descriptor_fcd(generated_by_task[key], reference_by_task[key], chem=chem),
            )
            summary.update(
                {
                    "model": model_name,
                    "task": TASK_LABELS.get(key, key),
                    "task_key": key,
                    "reference_n": reference_counts.get(key, 0),
                    "prediction_n": matched_predictions_by_task.get(key, 0),
                    "missing_prediction_rows": max(
                        reference_counts.get(key, 0) - matched_predictions_by_task.get(key, 0),
                        0,
                    ),
                    "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                }
            )
            summary["status"] = coverage_status(summary)
            rows.append(summary)
        if args.include_empty_table1 and args.task_filter == "table1":
            present = {row["task_key"] for row in rows if row.get("model") == model_name}
            for key in TABLE1_TASK_ORDER:
                if key in present:
                    continue
                summary = empty_task_summary(thresholds)
                summary.update(
                    {
                        "model": model_name,
                        "task": TASK_LABELS[key],
                        "task_key": key,
                        "reference_n": reference_counts.get(key, 0),
                        "prediction_n": matched_predictions_by_task.get(key, 0),
                        "missing_prediction_rows": max(
                            reference_counts.get(key, 0) - matched_predictions_by_task.get(key, 0),
                            0,
                        ),
                        "missing_oracle_skipped_rows": skipped_missing_oracle.get(key, 0),
                    }
                )
                summary["status"] = coverage_status(summary)
                rows.append(summary)

        if args.require_table1_coverage and args.task_filter == "table1":
            by_task = {
                row["task_key"]: row
                for row in rows
                if row.get("model") == model_name and row.get("task_key") in TASK_LABELS
            }
            for key in TABLE1_TASK_ORDER:
                row = by_task.get(key)
                if row is None:
                    coverage_failures.append(f"{model_name}:{key}:missing-summary-row")
                    continue
                if int(row.get("reference_n") or 0) <= 0:
                    coverage_failures.append(f"{model_name}:{key}:missing-reference")
                if int(row.get("n") or 0) <= 0:
                    coverage_failures.append(f"{model_name}:{key}:{row.get('status', 'not-evaluated')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "moledit_table_summary.csv"
    json_path = args.output_dir / "moledit_table_summary.json"
    md_path = args.output_dir / "moledit_table_summary.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, rows, thresholds=thresholds)
    coverage_path = args.output_dir / "moledit_table_coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "reference_counts": dict(sorted(reference_counts.items(), key=lambda item: task_sort_key(item[0]))),
                "failures": coverage_failures,
                "required": bool(args.require_table1_coverage),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "oracle_provenance.json").write_text(
        json.dumps(configured_oracle_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if coverage_failures:
        message = "Table1 coverage check failed: " + ", ".join(coverage_failures)
        if args.require_table1_coverage:
            raise SystemExit(message)
        print(message, file=sys.stderr)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2, sort_keys=True))


class Chemistry:
    def __init__(self) -> None:
        try:
            from rdkit import Chem
            from rdkit import DataStructs
            from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

            self.Chem = Chem
            self.DataStructs = DataStructs
            self.Crippen = Crippen
            self.Descriptors = Descriptors
            self.Lipinski = Lipinski
            self.QED = QED
            self.rdMolDescriptors = rdMolDescriptors
            self.rdkit_available = True
        except ImportError as exc:
            raise SystemExit("RDKit is required for MolEdit table metrics.") from exc
        self._oracles: dict[str, Callable[[str], float] | None] = {}

    def mol(self, smiles: str):
        if not smiles:
            return None
        return self.Chem.MolFromSmiles(smiles)

    def valid(self, smiles: str) -> bool:
        return self.mol(smiles) is not None

    def tanimoto(self, left: str, right: str) -> float | None:
        left_mol = self.mol(left)
        right_mol = self.mol(right)
        if left_mol is None or right_mol is None:
            return None
        fp_left = self.rdMolDescriptors.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
        fp_right = self.rdMolDescriptors.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
        return float(self.DataStructs.TanimotoSimilarity(fp_left, fp_right))

    def rdkit_properties(self, smiles: str) -> dict[str, float] | None:
        mol = self.mol(smiles)
        if mol is None:
            return None
        return {
            "MW": float(self.Descriptors.MolWt(mol)),
            "LogP": float(self.Crippen.MolLogP(mol)),
            "QED": float(self.QED.qed(mol)),
            "TPSA": float(self.rdMolDescriptors.CalcTPSA(mol)),
            "HBD": float(self.Lipinski.NumHDonors(mol)),
            "HBA": float(self.Lipinski.NumHAcceptors(mol)),
            "RB": float(self.Lipinski.NumRotatableBonds(mol)),
        }

    def score(self, smiles: str, prop: str) -> float | None:
        prop = canonical_property(prop)
        if prop in {"MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB"}:
            props = self.rdkit_properties(smiles)
            return None if props is None else props[prop]
        oracle = self.oracle(prop)
        if oracle is None:
            return None
        try:
            return float(oracle(smiles))
        except Exception:
            return None

    def oracle(self, prop: str) -> Callable[[str], float] | None:
        prop = canonical_property(prop)
        if prop in self._oracles:
            return self._oracles[prop]
        if prop == "SA":
            sa_oracle = self._rdkit_sa_oracle()
            self._oracles[prop] = sa_oracle
            return sa_oracle
        pinned = configured_pinned_oracle(prop)
        if pinned is not None:
            self._oracles[prop] = pinned
            return pinned
        try:
            from tdc import Oracle

            self._oracles[prop] = Oracle(name=prop)
        except Exception:
            self._oracles[prop] = None
        return self._oracles[prop]

    def _rdkit_sa_oracle(self) -> Callable[[str], float] | None:
        import importlib.util
        from pathlib import Path

        from rdkit.Chem import RDConfig

        sascorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        if not sascorer_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("sascorer", sascorer_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def score(smiles: str) -> float:
            mol = self.mol(smiles)
            if mol is None:
                raise ValueError("invalid SMILES")
            return float(module.calculateScore(mol))

        return score

    def missing_oracles(self, task_specs: Iterable[Mapping[str, str]]) -> set[str]:
        missing = set()
        for spec in task_specs:
            prop = canonical_property(spec.get("property", ""))
            if prop not in {"MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB"} and self.oracle(prop) is None:
                missing.add(prop)
        return missing


def load_references(path: Path) -> dict[str, dict[str, object]]:
    refs = {}
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ref_id = normalize_id(row.get("example_id") or row.get("sample_id") or row.get("pair_hash"))
                if ref_id:
                    refs[ref_id] = row
        return refs

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ref_id = normalize_id(row.get("example_id") or row.get("condition_id") or row.get("sample_id") or row.get("pair_hash"))
            if ref_id:
                refs[ref_id] = row
    return refs


def load_predictions(path: Path, *, model_name: str, method: str | None) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_method = row.get("method", "")
            if method is not None and row_method != method:
                continue
            pred_smiles = first_value(row, PREDICTION_SMILES_COLUMNS)
            pred_id = normalize_id(first_value(row, ID_COLUMNS))
            if not pred_id or not pred_smiles:
                continue
            name = row_method or row.get("model", "") or model_name
            out[name].append({"example_id": pred_id, "predicted_smiles": pred_smiles})
    return out


def first_value(row: Mapping[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def normalize_id(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("edit:"):
        return text.split(":")[-1]
    return text


def task_specs_for_reference(row: Mapping[str, object]) -> list[dict[str, str]]:
    raw = row.get("instruction_tasks") or ""
    if not raw and isinstance(row.get("metadata"), Mapping):
        raw = row["metadata"].get("moledit_tasks", "")
    try:
        parsed = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        parsed = []
    specs = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, Mapping):
                continue
            prop = canonical_property(item.get("property", ""))
            direction = canonical_direction(item.get("direction", "") or "unknown")
            if prop:
                specs.append({"property": prop, "direction": direction})
    if specs:
        return specs
    props = str(row.get("instruction_task_properties") or row.get("computed_active_properties") or "").replace(",", "|")
    for prop in [item.strip() for item in props.split("|") if item.strip()]:
        canonical = canonical_property(prop)
        direction = canonical_direction(row.get(f"{canonical}_direction", "") or row.get(f"{prop}_direction", "") or "unknown")
        specs.append({"property": canonical, "direction": direction})
    return specs


def task_key(task_specs: list[dict[str, str]]) -> str:
    pairs = [
        (canonical_property(spec.get("property", "")), canonical_direction(spec.get("direction", "") or "unknown"))
        for spec in task_specs
        if spec.get("property")
    ]
    canonical = TABLE1_TASK_SPECS.get(frozenset(pairs))
    if canonical:
        return canonical
    return "+".join(f"{prop}:{direction}" for prop, direction in sorted(pairs))


def task_sort_key(key: str) -> tuple[int, str]:
    ordered = list(TABLE1_TASK_ORDER)
    return (ordered.index(key), key) if key in TASK_LABELS else (999, key)


def evaluate_prediction(
    ref: Mapping[str, object],
    predicted_smiles: str,
    task_specs: list[dict[str, str]],
    *,
    chem: Chemistry,
    thresholds: list[float],
) -> dict[str, object]:
    valid = chem.valid(predicted_smiles)
    source_smiles = str(ref.get("source_smiles", "") or "")
    tanimoto = chem.tanimoto(source_smiles, predicted_smiles) if valid else None
    property_success = valid and desired_property_success(source_smiles, predicted_smiles, task_specs, chem=chem)
    row = {
        "valid": valid,
        "property_success": property_success,
        "source_tanimoto": tanimoto,
    }
    for threshold in thresholds:
        row[f"success_t{threshold:g}"] = bool(property_success and tanimoto is not None and tanimoto >= threshold)
    return row


def desired_property_success(
    source_smiles: str,
    predicted_smiles: str,
    task_specs: list[dict[str, str]],
    *,
    chem: Chemistry,
) -> bool:
    if not task_specs:
        return False
    for spec in task_specs:
        prop = canonical_property(spec.get("property", ""))
        direction = canonical_direction(spec.get("direction", "") or "unknown")
        source_score = chem.score(source_smiles, prop)
        pred_score = chem.score(predicted_smiles, prop)
        if source_score is None or pred_score is None:
            return False
        if direction == "increase" and not (pred_score > source_score):
            return False
        if direction == "decrease" and not (pred_score < source_score):
            return False
        if direction not in {"increase", "decrease"}:
            return False
    return True


def summarize_task(rows: list[dict[str, object]], *, thresholds: list[float], fcd: float | None) -> dict[str, object]:
    valid_rows = [row for row in rows if row["valid"]]
    out: dict[str, object] = {
        "n": len(rows),
        "valid_n": len(valid_rows),
        "Validity": fraction(row["valid"] for row in rows),
        "FCD": "" if fcd is None or math.isnan(fcd) else fcd,
    }
    for threshold in thresholds:
        key = f"success_t{threshold:g}"
        out[f"Acc_all({threshold:g})"] = fraction(row[key] for row in rows)
        out[f"Acc_valid({threshold:g})"] = fraction(row[key] for row in valid_rows)
    return out


def empty_task_summary(thresholds: list[float]) -> dict[str, object]:
    out: dict[str, object] = {
        "n": 0,
        "valid_n": 0,
        "Validity": "",
        "FCD": "",
    }
    for threshold in thresholds:
        out[f"Acc_all({threshold:g})"] = ""
        out[f"Acc_valid({threshold:g})"] = ""
    return out


def coverage_status(row: Mapping[str, object]) -> str:
    reference_n = int(row.get("reference_n") or 0)
    prediction_n = int(row.get("prediction_n") or 0)
    evaluated_n = int(row.get("n") or 0)
    skipped_oracle = int(row.get("missing_oracle_skipped_rows") or 0)
    if reference_n <= 0:
        return "missing-reference"
    if skipped_oracle > 0 and evaluated_n <= 0:
        return "skipped-missing-oracle"
    if prediction_n <= 0:
        return "missing-prediction"
    if evaluated_n <= 0:
        return "not-evaluated"
    if prediction_n < reference_n:
        return "partial"
    return "measured"


def count_reference_tasks(
    references: Iterable[Mapping[str, object]],
    *,
    task_filter: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ref in references:
        key = task_key(task_specs_for_reference(ref))
        if task_filter == "table1" and key not in TASK_LABELS:
            continue
        counts[key] += 1
    return counts


def fraction(values: Iterable[object]) -> float:
    seq = [bool(value) for value in values]
    return float(sum(seq) / len(seq)) if seq else 0.0


def descriptor_fcd(generated_smiles: list[str], reference_smiles: list[str], *, chem: Chemistry) -> float | None:
    gen = descriptor_matrix(generated_smiles, chem=chem)
    ref = descriptor_matrix(reference_smiles, chem=chem)
    if gen.shape[0] < 2 or ref.shape[0] < 2:
        return None
    return frechet_distance(gen, ref)


def descriptor_matrix(smiles_list: list[str], *, chem: Chemistry) -> np.ndarray:
    rows = []
    for smiles in smiles_list:
        props = chem.rdkit_properties(smiles)
        if props is None:
            continue
        rows.append([props[prop] for prop in ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")])
    return np.asarray(rows, dtype=np.float64)


def frechet_distance(left: np.ndarray, right: np.ndarray) -> float:
    mu_left = np.mean(left, axis=0)
    mu_right = np.mean(right, axis=0)
    cov_left = np.cov(left, rowvar=False)
    cov_right = np.cov(right, rowvar=False)
    diff = mu_left - mu_right
    try:
        covmean = matrix_sqrt(cov_left @ cov_right)
    except np.linalg.LinAlgError:
        return float("nan")
    value = diff.dot(diff) + np.trace(cov_left + cov_right - 2.0 * covmean)
    return float(max(value, 0.0))


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    values = np.where(values.real < 0, 0, values.real)
    root = vectors.real @ np.diag(np.sqrt(values)) @ np.linalg.inv(vectors.real)
    return root.real


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "task",
        "task_key",
        "status",
        "reference_n",
        "prediction_n",
        "missing_prediction_rows",
        "n",
        "valid_n",
        "Validity",
        "Acc_all(0.65)",
        "Acc_valid(0.65)",
        "Acc_all(0.15)",
        "Acc_valid(0.15)",
        "FCD",
        "missing_oracle_skipped_rows",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], *, thresholds: list[float]) -> None:
    threshold_cols = []
    for threshold in thresholds:
        threshold_cols.extend([f"Acc_all({threshold:g})", f"Acc_valid({threshold:g})"])
    columns = ["model", "task", "status", "reference_n", "prediction_n", "Validity", *threshold_cols, "FCD", "n"]
    lines = [
        "# MolEdit Table Metrics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [format_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def canonical_property(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PROPERTY_ALIASES.get(text.lower(), text)


def canonical_direction(value: object) -> str:
    text = str(value or "").strip().lower()
    return DIRECTION_ALIASES.get(text, text or "unknown")


if __name__ == "__main__":
    main()
