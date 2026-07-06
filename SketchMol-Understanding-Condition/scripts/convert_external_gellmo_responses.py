#!/usr/bin/env python3
"""Convert official GeLLMO JSON responses into SUCC external evaluator CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402

import export_external_multiproperty_benchmark_rows as exporter  # noqa: E402


SMILES_TAG_RE = re.compile(r"<SMILES>\s*([A-Za-z0-9@+\-\[\]\(\)\\/%=#$:.]+)\s*</SMILES>")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-json", required=True, type=Path)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--suite", choices=("mumo", "cmumo"), default="mumo")
    parser.add_argument("--task", required=True)
    parser.add_argument("--setting", default="seen", help="Official GeLLMO instr_setting filter: seen, unseen, or all.")
    parser.add_argument("--max-rows", type=int, default=0, help="0 keeps all matching official rows.")
    parser.add_argument("--model-id", default="mistral", help="Official response structure hint: mistral or llama.")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--lora-weights", default="")
    parser.add_argument("--method", default="gellmo_official")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = exporter.select_specs(str(args.suite), "all", str(args.task))
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one {args.suite} task spec for {args.task!r}, got {len(specs)}")
    spec = specs[0]
    source_rows = select_official_rows(
        exporter.read_source_rows(args.source_file),
        spec=spec,
        setting=str(args.setting),
        max_rows=int(args.max_rows),
    )
    responses = read_response_entries(args.response_json)
    output_rows = []
    empty_inputs = 0
    valid_candidates = 0
    for index, raw_row in enumerate(source_rows):
        condition = exporter.build_condition_row(
            raw_row,
            spec=spec,
            raw_index=index,
            local_index=index,
            source_smiles_column=None,
            target_smiles_column=None,
            id_column=None,
        )
        setting_slug = str(args.setting or raw_row.get("instr_setting") or "setting").strip().lower() or "setting"
        for id_key in ("sample_id", "condition_id"):
            condition[id_key] = f"{condition[id_key]}_{setting_slug}"
        condition["variant_id"] = f"{condition['condition_id']}:full"
        response_entry = responses[index] if index < len(responses) else {}
        candidates = extract_candidate_smiles(response_entry, model_id=str(args.model_id))
        if not candidates:
            empty_inputs += 1
            candidates = [""]
        for rank, smiles in enumerate(candidates, start=1):
            row = dict(condition)
            row["generated_smiles"] = smiles
            row["method"] = str(args.method)
            row["candidate_rank"] = rank
            row["candidate_selected"] = "True" if rank == 1 else "False"
            row["official_model_id"] = str(args.model_id)
            row["official_base_model"] = str(args.base_model)
            row["official_lora_weights"] = str(args.lora_weights)
            row["official_task"] = str(args.task)
            row["official_instr_setting"] = str(args.setting)
            row["official_response_index"] = index
            row["official_extracted_candidate_count"] = len([item for item in candidates if item])
            if smiles:
                valid_candidates += 1
            output_rows.append(row)
    write_rows(args.output_csv, output_rows)
    summary = {
        "response_json": str(args.response_json),
        "source_file": str(args.source_file),
        "output_csv": str(args.output_csv),
        "suite": str(args.suite),
        "task": str(args.task),
        "setting": str(args.setting),
        "official_rows": len(source_rows),
        "response_entries": len(responses),
        "output_rows": len(output_rows),
        "empty_inputs": empty_inputs,
        "valid_extracted_candidates": valid_candidates,
        "method": str(args.method),
        "model_id": str(args.model_id),
        "base_model": str(args.base_model),
        "lora_weights": str(args.lora_weights),
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def select_official_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    spec: exporter.ExternalTaskSpec,
    setting: str,
    max_rows: int,
) -> list[dict[str, object]]:
    task_aliases = {
        exporter.normalize_task_name(spec.task_id),
        exporter.normalize_task_name(spec.task_key),
        exporter.normalize_task_name(f"{spec.suite}:{spec.task_id}"),
        exporter.normalize_task_name(f"{spec.suite}:{spec.task_key}"),
    }
    setting = str(setting or "all").strip().lower()
    out = []
    for row in rows:
        row_task = exporter.normalize_task_name(
            exporter.first_value(row, ("task", "task_key", "external_task_key", "external_task_id"))
        )
        if row_task and row_task not in task_aliases:
            continue
        row_setting = str(row.get("instr_setting") or row.get("setting") or "").strip().lower()
        if setting not in {"", "all"} and row_setting != setting:
            continue
        out.append(dict(row))
        if max_rows > 0 and len(out) >= max_rows:
            break
    if not out:
        raise ValueError(f"No source rows matched task={spec.task_id}/{spec.task_key}, setting={setting!r}")
    return out


def read_response_entries(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list response JSON at {path}")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def extract_candidate_smiles(entry: Mapping[str, object], *, model_id: str) -> list[str]:
    text = response_text(entry, model_id=model_id)
    out = []
    seen = set()
    for match in SMILES_TAG_RE.findall(text):
        canonical = canonical_or_blank(match)
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def response_text(entry: Mapping[str, object], *, model_id: str) -> str:
    model_id = str(model_id or "").lower()
    if model_id in {"mistral", "llama", "gellmo"}:
        responses = entry.get("response", [])
        if not isinstance(responses, list):
            responses = [responses]
        cleaned = []
        for response in responses:
            text = str(response or "")
            if "[/INST]\n%%% Response:" in text:
                text = text.split("[/INST]\n%%% Response:", 1)[1].split("<<SYS>>", 1)[0]
            cleaned.append(text)
        return " ".join(cleaned)
    if "output" in entry:
        value = entry.get("output")
        if isinstance(value, list):
            return " ".join(str(item or "") for item in value)
        return str(value or "")
    return json.dumps(dict(entry), sort_keys=True)


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
