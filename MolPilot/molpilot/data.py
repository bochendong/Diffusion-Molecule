"""Dataset construction for edit, inpainting, and de novo smoke experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from .chem import molecular_descriptors, passes_druglike_proxy, scaffold_key
from .schema import GenerationRequest, TaskType
from .understanding import ground_instruction
from .verifier import verify_candidate


SMOKE_SMILES = [
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc(CCN)cc1",
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "CC(C)NCC(O)COc1ccccc1",
    "O=C(Nc1ccccc1)c1ccccc1",
]


def load_smiles_csv(path: str | Path | None, limit: int = 0) -> list[str]:
    if path is None or not str(path).strip() or not Path(path).exists() or Path(path).is_dir():
        return list(SMOKE_SMILES)
    out = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles")
            if not smiles:
                continue
            rec = molecular_descriptors(smiles)
            if rec.valid:
                out.append(rec.smiles)
            if limit and len(out) >= limit:
                break
    return out or list(SMOKE_SMILES)


def build_smoke_requests(smiles: list[str]) -> list[tuple[GenerationRequest, str]]:
    verified = build_verified_requests(smiles)
    if verified:
        return verified
    return build_placeholder_requests(smiles)


def build_verified_requests(smiles: list[str]) -> list[tuple[GenerationRequest, str]]:
    records = [molecular_descriptors(smi) for smi in smiles]
    valid_indices = [idx for idx, rec in enumerate(records) if rec.valid]
    scaffold_groups: dict[str, list[int]] = {}
    for idx in valid_indices:
        key = scaffold_key(records[idx].smiles)
        if key:
            scaffold_groups.setdefault(key, []).append(idx)

    requests: list[tuple[GenerationRequest, str]] = []
    for idx in valid_indices:
        source = records[idx]
        logp_target = _find_same_scaffold_target(idx, records, scaffold_groups, "LogP", maximum_delta=-0.5, keep_mw_similar=True)
        if logp_target is not None:
            requests.extend(
                _verified_pairs(
                    source.smiles,
                    records[logp_target].smiles,
                    [
                        (
                            TaskType.EDIT,
                            "The molecule is too lipophilic. Lower LogP, preserve scaffold, and keep MW similar.",
                            "",
                        ),
                        (
                            TaskType.INPAINT,
                            "Fill the masked substituent to improve solubility while preserving scaffold and keeping MW similar.",
                            "[*:1]",
                        ),
                    ],
                )
            )

        tpsa_target = _find_same_scaffold_target(idx, records, scaffold_groups, "TPSA", maximum_delta=-15.0, keep_mw_similar=True)
        if tpsa_target is not None:
            requests.extend(
                _verified_pairs(
                    source.smiles,
                    records[tpsa_target].smiles,
                    [
                        (
                            TaskType.EDIT,
                            "TPSA is too high. Reduce TPSA, preserve scaffold, and keep MW similar.",
                            "",
                        )
                    ],
                )
            )

        qed_target = _find_same_scaffold_target(idx, records, scaffold_groups, "QED", minimum_delta=0.05, keep_mw_similar=False)
        if qed_target is not None:
            requests.extend(
                _verified_pairs(
                    source.smiles,
                    records[qed_target].smiles,
                    [
                        (
                            TaskType.EDIT,
                            "Improve QED and drug-like score while preserving scaffold.",
                            "",
                        )
                    ],
                )
            )

        if passes_druglike_proxy(source.descriptors):
            instruction = _de_novo_instruction(source.descriptors)
            request = GenerationRequest(task_type=TaskType.DE_NOVO, instruction=instruction)
            result = verify_candidate(None, source.smiles, ground_instruction(instruction))
            if result.overall_success:
                requests.append((request, source.smiles))

    return requests


def build_placeholder_requests(smiles: list[str]) -> list[tuple[GenerationRequest, str]]:
    requests: list[tuple[GenerationRequest, str]] = []
    for idx, smi in enumerate(smiles):
        target = smiles[(idx + 1) % len(smiles)]
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.EDIT,
                    source_smiles=smi,
                    instruction="The molecule is too lipophilic. Lower LogP and keep the core similar.",
                ),
                target,
            )
        )
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.INPAINT,
                    source_smiles=smi,
                    mask_smarts="[*:1]",
                    instruction="Fill the masked substituent to improve solubility while preserving the scaffold.",
                ),
                target,
            )
        )
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.DE_NOVO,
                    instruction="Generate a CNS-like drug-like molecule with MW below 450 and TPSA below 90.",
                ),
                smi,
            )
        )
    return requests


def _verified_pairs(
    source_smiles: str,
    target_smiles: str,
    task_specs: list[tuple[TaskType, str, str]],
) -> list[tuple[GenerationRequest, str]]:
    out: list[tuple[GenerationRequest, str]] = []
    for task_type, instruction, mask_smarts in task_specs:
        request = GenerationRequest(
            task_type=task_type,
            source_smiles=source_smiles,
            instruction=instruction,
            mask_smarts=mask_smarts or None,
        )
        result = verify_candidate(source_smiles, target_smiles, ground_instruction(instruction))
        if result.overall_success:
            out.append((request, target_smiles))
    return out


def _find_same_scaffold_target(
    source_idx: int,
    records,
    scaffold_groups: dict[str, list[int]],
    key: str,
    minimum_delta: float | None = None,
    maximum_delta: float | None = None,
    keep_mw_similar: bool = True,
) -> int | None:
    source = records[source_idx]
    group = scaffold_groups.get(scaffold_key(source.smiles), [])
    best_idx = None
    best_score = float("inf")
    for target_idx in group:
        if target_idx == source_idx:
            continue
        target = records[target_idx]
        delta = target.descriptors.get(key, 0.0) - source.descriptors.get(key, 0.0)
        if minimum_delta is not None and delta < minimum_delta:
            continue
        if maximum_delta is not None and delta > maximum_delta:
            continue
        mw_delta = abs(target.descriptors.get("MW", 0.0) - source.descriptors.get("MW", 0.0))
        if keep_mw_similar and mw_delta > 80.0:
            continue
        score = abs(delta) + 0.01 * mw_delta
        if score < best_score:
            best_score = score
            best_idx = target_idx
    return best_idx


def _de_novo_instruction(desc: dict[str, float]) -> str:
    if desc.get("MW", 999.0) <= 450.0 and desc.get("TPSA", 999.0) <= 90.0 and desc.get("HBD", 99.0) <= 1.0:
        return "Generate a CNS-like drug-like molecule with MW below 450 and TPSA below 90."
    return "Generate a drug-like molecule."
