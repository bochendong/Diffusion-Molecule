#!/usr/bin/env python3
"""Fail-closed prompt and response contract for P16 direct molecular SFT."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

from rdkit import Chem


PROTOCOL = "p16_direct_llm_unified_generation_editing_v1"
PROPERTIES = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA")
SYSTEM = (
    "You are one molecular causal language model for both generation and editing. "
    "Infer the task only from whether source is <EMPTY> or a molecule. Return exactly "
    '{"plan":"BUILD","smiles":"CANONICAL_SMILES"} or '
    '{"plan":"MODIFY","smiles":"CANONICAL_SMILES"}. No prose or markdown.'
)
RESPONSE_RE = re.compile(r'^\{"plan":"(BUILD|MODIFY)","smiles":"([^"\n]+)"\}$')


def canonical_smiles(value: object) -> str:
    text = str(value or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def condition_allowlist(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract only declared property controls; never free-form target/oracle fields."""
    result: list[dict[str, object]] = []
    for prop in PROPERTIES:
        active = str(row.get(f"{prop}_active", "")).strip().lower()
        setting = str(row.get(f"{prop}_setting", "")).strip()
        target = str(row.get(f"target_{prop}", "")).strip()
        direction = str(row.get(f"{prop}_direction", "")).strip().lower()
        none_flag = str(row.get(f"{prop}_None", "")).strip().lower()
        selected = active in {"1", "true", "yes"} or (none_flag == "false")
        if not selected:
            continue
        item: dict[str, object] = {"property": prop}
        if direction in {"increase", "decrease", "preserve"}:
            item["goal"] = direction
        elif setting or target:
            raw = setting or target
            try:
                item["goal"] = {"around": round(float(raw), 6)}
            except ValueError:
                continue
        else:
            continue
        result.append(item)
    if not result:
        raise ValueError("row has no allowlisted active conditions")
    return result


def mode_for_source(source: str) -> str:
    return "de_novo" if source == "<EMPTY>" else "edit"


def build_prompt(row: Mapping[str, object]) -> tuple[list[dict[str, str]], str, str]:
    source_raw = str(row.get("source_smiles", "") or "").strip()
    source = canonical_smiles(source_raw) if source_raw else "<EMPTY>"
    if source_raw and not source:
        raise ValueError("invalid source SMILES")
    payload = {"conditions": condition_allowlist(row), "source": source}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]
    serialized = json.dumps(messages, sort_keys=True)
    for forbidden in ("target_smiles", "policy_target_smiles", "target_scaffold", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"forbidden prompt field: {forbidden}")
    return messages, source, mode_for_source(source)


def response(target: str, mode: str) -> str:
    canonical = canonical_smiles(target)
    if not canonical:
        raise ValueError("invalid target SMILES")
    plan = "BUILD" if mode == "de_novo" else "MODIFY"
    return json.dumps({"plan": plan, "smiles": canonical}, separators=(",", ":"))


def parse_response(text: str, expected_mode: str) -> dict[str, object]:
    raw = str(text).strip()
    match = RESPONSE_RE.fullmatch(raw)
    if match is None:
        return {"strict_parse": False, "valid": False, "canonical": False, "smiles": ""}
    plan, smiles = match.groups()
    expected_plan = "BUILD" if expected_mode == "de_novo" else "MODIFY"
    canonical = canonical_smiles(smiles)
    valid = bool(canonical)
    canonical_form = valid and canonical == smiles
    strict = plan == expected_plan
    return {
        "strict_parse": strict,
        "valid": strict and valid,
        "canonical": strict and canonical_form,
        "smiles": canonical if strict and valid else "",
        "plan": plan,
    }


def condition_hash(row: Mapping[str, object]) -> str:
    payload = json.dumps(condition_allowlist(row), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_hash(source: str) -> str:
    if source == "<EMPTY>":
        return ""
    return hashlib.sha256(source.encode()).hexdigest()


def split_score(condition_digest: str, source_digest: str, seed: int) -> int:
    material = f"{seed}:{condition_digest}:{source_digest}"
    return int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
