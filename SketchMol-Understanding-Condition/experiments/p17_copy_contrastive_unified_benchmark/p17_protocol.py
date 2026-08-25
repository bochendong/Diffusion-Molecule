#!/usr/bin/env python3
"""P17's unchanged P16 prompt/response protocol plus family hashing."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

P16_DIR = Path(__file__).resolve().parent.parent / "p16_direct_llm_unified_generation_editing"
if str(P16_DIR) not in sys.path:
    sys.path.insert(0, str(P16_DIR))
import p16_protocol as p16  # noqa: E402

PROTOCOL = "p17_copy_contrastive_unified_pilot_v1"
canonical_smiles = p16.canonical_smiles
build_prompt = p16.build_prompt
response = p16.response
parse_response = p16.parse_response
condition_hash = p16.condition_hash
source_hash = p16.source_hash
split_score = p16.split_score


def condition_family(row: Mapping[str, object]) -> str:
    """Stable coarse condition signature, excluding numeric target values."""
    family = []
    for item in p16.condition_allowlist(row):
        goal = item["goal"]
        family.append((str(item["property"]), goal if isinstance(goal, str) else "around"))
    return json.dumps(sorted(family), separators=(",", ":"))


def condition_family_hash(row: Mapping[str, object]) -> str:
    return hashlib.sha256(condition_family(row).encode()).hexdigest()


def response_from_source(source: str) -> str:
    if source == "<EMPTY>":
        raise ValueError("de-novo rows have no source-copy negative")
    return response(source, "edit")
