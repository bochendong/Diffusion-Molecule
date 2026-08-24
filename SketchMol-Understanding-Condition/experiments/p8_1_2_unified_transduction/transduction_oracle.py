#!/usr/bin/env python3
"""Oracle gate for one empty/source SELFIES sequence-transduction language.

De-novo design is not assigned a special output representation: it is the
same transduction program applied to an empty source sequence.  Editing uses
the same KEEP/DELETE/INSERT interpreter with a non-empty source sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence


START = "<TRANSDUCE>"
INSERT = "<INSERT>"
INSERT_END = "<INSERT_END>"
STOP = "<STOP>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--variant", choices=("r1_canonical", "r2_source_aligned"), required=True)
    parser.add_argument("--max-program-tokens", type=int, default=188)
    parser.add_argument("--mcs-timeout", type=int, default=2)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def modules():
    try:
        import selfies
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
    except ImportError as exc:
        raise RuntimeError("P8.1.2 requires the existing Nibi RDKit+SELFIES environment") from exc
    return selfies, Chem, rdFMCS


def canonical(smiles: str) -> str:
    _selfies, Chem, _rdFMCS = modules()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def sf_tokens(smiles: str) -> list[str]:
    selfies, _Chem, _rdFMCS = modules()
    encoded = selfies.encoder(str(smiles or ""))
    return list(selfies.split_selfies(encoded))


def decode_sf(tokens: Sequence[str]) -> str:
    selfies, _Chem, _rdFMCS = modules()
    return str(selfies.decoder("".join(tokens)))


def source_aligned_smiles(source_smiles: str, target_smiles: str, *, timeout: int) -> str:
    """Renumber target atoms along the source MCS; chemistry is unchanged."""
    _selfies, Chem, rdFMCS = modules()
    source = Chem.MolFromSmiles(source_smiles)
    target = Chem.MolFromSmiles(target_smiles)
    fallback = canonical(target_smiles)
    if source is None or target is None:
        return fallback
    result = rdFMCS.FindMCS(
        [source, target], timeout=max(1, int(timeout)), ringMatchesRingOnly=True,
        completeRingsOnly=True, matchValences=True,
    )
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    if query is None:
        return fallback
    source_matches = source.GetSubstructMatches(query, uniquify=True, maxMatches=16)
    target_matches = target.GetSubstructMatches(query, uniquify=True, maxMatches=16)
    source_seq = sf_tokens(source_smiles)
    best = fallback
    best_equal = -1
    for source_match in source_matches:
        for target_match in target_matches:
            order = [target_idx for _, target_idx in sorted(zip(source_match, target_match))]
            used = set(order)
            order.extend(idx for idx in range(target.GetNumAtoms()) if idx not in used)
            if len(order) != target.GetNumAtoms() or len(set(order)) != len(order):
                continue
            try:
                candidate = Chem.MolToSmiles(Chem.RenumberAtoms(target, order), canonical=False, isomericSmiles=True)
                if canonical(candidate) != fallback:
                    continue
                candidate_seq = sf_tokens(candidate)
            except Exception:
                continue
            equal = sum(block.size for block in SequenceMatcher(None, source_seq, candidate_seq, autojunk=False).get_matching_blocks())
            if equal > best_equal:
                best, best_equal = candidate, equal
    return best


def count_token(kind: str, count: int) -> str:
    if count <= 0:
        raise ValueError("run length must be positive")
    return f"<{kind}_{count}>"


def parse_count(token: str, kind: str) -> int:
    prefix = f"<{kind}_"
    if not token.startswith(prefix) or not token.endswith(">"):
        raise ValueError(f"expected {kind} run, got {token}")
    return int(token[len(prefix):-1])


def transduction_program(source: Sequence[str], target: Sequence[str]) -> list[str]:
    program = [START]
    matcher = SequenceMatcher(None, list(source), list(target), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            program.append(count_token("KEEP", i2 - i1))
        else:
            if tag in {"delete", "replace"} and i2 > i1:
                program.append(count_token("DELETE", i2 - i1))
            if tag in {"insert", "replace"} and j2 > j1:
                program.extend([INSERT, *target[j1:j2], INSERT_END])
    program.append(STOP)
    return program


def execute_program(source: Sequence[str], program: Sequence[str]) -> list[str]:
    if not program or program[0] != START:
        raise ValueError("missing transduction start")
    output: list[str] = []
    cursor = 0
    index = 1
    while index < len(program):
        token = program[index]
        if token == STOP:
            if cursor != len(source):
                raise ValueError("program stopped before consuming source")
            return output
        if token.startswith("<KEEP_"):
            count = parse_count(token, "KEEP")
            output.extend(source[cursor:cursor + count])
            cursor += count
            index += 1
            continue
        if token.startswith("<DELETE_"):
            cursor += parse_count(token, "DELETE")
            index += 1
            continue
        if token == INSERT:
            end = list(program).index(INSERT_END, index + 1)
            output.extend(program[index + 1:end])
            index = end + 1
            continue
        raise ValueError(f"illegal transduction token: {token}")
    raise ValueError("missing stop")


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)])


def mode_for_row(row: Mapping[str, str]) -> str:
    explicit = str(row.get("task_mode", "")).lower()
    if "de" in explicit and "novo" in explicit:
        return "de_novo"
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
    return "editing" if source else "de_novo"


def target_for_row(row: Mapping[str, str], mode: str) -> str:
    if mode == "editing":
        return str(row.get("policy_target_smiles", "") or row.get("target_smiles", "")).strip()
    return str(row.get("target_smiles", "") or row.get("policy_target_smiles", "")).strip()


def main() -> int:
    args = parse_args()
    output: list[dict[str, object]] = []
    skipped = Counter()
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    vocab = {START, INSERT, INSERT_END, STOP}
    for row in read_rows(args.input_csv):
        mode = mode_for_row(row)
        source_raw = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
        target_raw = target_for_row(row, mode)
        try:
            source_smiles = canonical(source_raw) if source_raw else ""
            target_canonical = canonical(target_raw)
            if not target_canonical:
                raise ValueError("invalid_target")
            serialized_target = target_canonical
            if args.variant == "r2_source_aligned" and source_smiles:
                serialized_target = source_aligned_smiles(source_smiles, target_canonical, timeout=args.mcs_timeout)
            source_tokens = sf_tokens(source_smiles) if source_smiles else []
            target_tokens = sf_tokens(serialized_target)
            program = transduction_program(source_tokens, target_tokens)
            reconstructed_tokens = execute_program(source_tokens, program)
            reconstructed = canonical(decode_sf(reconstructed_tokens))
            exact = bool(reconstructed and reconstructed == target_canonical)
            copied = sum(parse_count(token, "KEEP") for token in program if token.startswith("<KEEP_"))
            inserted = sum(
                1 for token in program
                if token.startswith("[") and token.endswith("]")
            )
            record = {
                "mode": mode,
                "program_tokens": len(program),
                "full_target_tokens": len(target_tokens) + 3,
                "fit_budget": len(program) <= int(args.max_program_tokens),
                "exact": exact,
                "valid": bool(reconstructed),
                "copied_tokens": copied,
                "inserted_tokens": inserted,
                "source_tokens": len(source_tokens),
                "target_tokens": len(target_tokens),
            }
            by_mode[mode].append(record)
            vocab.update(program)
            item = dict(row)
            item.update({
                "task_mode": mode,
                "policy_target_tokens_json": json.dumps(program),
                "p812_source_selfies_tokens_json": json.dumps(source_tokens),
                "p812_serialized_target_smiles": serialized_target,
                "p812_reconstructed_smiles": reconstructed,
                "p812_exact_reconstruction": str(exact),
                "p812_program_token_count": len(program),
            })
            output.append(item)
        except Exception as exc:
            skipped[str(exc) or type(exc).__name__] += 1
    write_rows(args.output_csv, output)
    input_rows = len(output) + sum(skipped.values())
    def aggregate(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
        lengths = [int(record["program_tokens"]) for record in records]
        exact = [bool(record["exact"]) for record in records]
        fits = [bool(record["fit_budget"]) for record in records]
        target_lengths = [int(record["full_target_tokens"]) for record in records]
        copied = [int(record["copied_tokens"]) for record in records]
        source_lengths = [int(record["source_tokens"]) for record in records]
        return {
            "rows": len(records),
            "exact_reconstruction": sum(exact) / max(len(exact), 1),
            "fit_budget_fraction": sum(fits) / max(len(fits), 1),
            "mean_program_tokens": mean(lengths) if lengths else math.nan,
            "median_program_tokens": median(lengths) if lengths else math.nan,
            "p95_program_tokens": percentile(lengths, 0.95),
            "mean_full_target_tokens": mean(target_lengths) if target_lengths else math.nan,
            "mean_copy_fraction": mean(
                c / max(s, 1) for c, s in zip(copied, source_lengths)
            ) if records else math.nan,
        }
    summary = {
        "protocol": "p8_1_2_empty_source_selfies_transduction_oracle_v1",
        "variant": args.variant,
        "input_rows": input_rows,
        "output_rows": len(output),
        "coverage": len(output) / max(input_rows, 1),
        "max_program_tokens": int(args.max_program_tokens),
        "vocabulary_size_observed": len(vocab),
        "by_mode": {mode: aggregate(records) for mode, records in sorted(by_mode.items())},
        "skipped": dict(skipped),
        "unification_contract": {
            "decoder_count": 1,
            "checkpoint_count": 1,
            "interpreter_count": 1,
            "output_language_count": 1,
            "task_router": False,
            "de_novo_definition": "the same transducer with an empty source token sequence",
            "editing_definition": "the same transducer with a non-empty source token sequence",
            "property_reranking": False,
        },
        "next_gate": {
            "samples_per_condition": 20,
            "raw_budgets": [1, 8, 20],
            "selection": "generation order only; no property reranking",
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
