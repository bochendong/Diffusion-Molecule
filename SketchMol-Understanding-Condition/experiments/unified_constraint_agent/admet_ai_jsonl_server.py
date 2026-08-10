#!/usr/bin/env python3
"""Persistent JSONL bridge for the official ADMET-AI property predictors.

The common LLM and ADMET-AI use separate cluster environments.  Loading the
ADMET ensemble for every sampled action is prohibitively slow, so this process
loads it once and serves deterministic batched predictions over stdin/stdout.
Model diagnostics are redirected to stderr to keep stdout machine-readable.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from typing import Mapping


COLUMNS = {
    "bbbp": ("BBB_Martins", "bbb_martins"),
    "hia": ("HIA_Hou", "hia_hou"),
    "mutagenicity": ("AMES", "ames"),
}


def row_mapping(value: object) -> Mapping[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return value if isinstance(value, Mapping) else {}


def finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def main() -> int:
    # ADMET-AI and Lightning print progress for every prediction. Keep the
    # bridge quiet; structured failures still return through the JSON channel.
    with open(os.devnull, "w", encoding="utf-8") as quiet:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            from admet_ai import ADMETModel

            model = ADMETModel()
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("request_id")
                smiles = [str(item) for item in request.get("smiles", []) if str(item).strip()]
                if not smiles:
                    raise ValueError("request has no SMILES")
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    predictions = model.predict(smiles=smiles)
                values = []
                for index, smiles_value in enumerate(smiles):
                    if hasattr(predictions, "iloc"):
                        source = row_mapping(predictions.iloc[index])
                    else:
                        source = row_mapping(predictions)
                    record: dict[str, object] = {"smiles": smiles_value}
                    for prop, candidates in COLUMNS.items():
                        for column in candidates:
                            value = finite_float(source.get(column))
                            if value is not None:
                                record[prop] = value
                                break
                    values.append(record)
                response = {"request_id": request_id, "predictions": values}
            except Exception as exc:  # keep the parent error actionable
                response = {
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(json.dumps(response, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
