#!/usr/bin/env python3
"""E1b gates: keyword vs scrambled vs E1 template vs B41. Diagnostic only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--keyword-summary", required=True, type=Path)
    parser.add_argument("--scrambled-summary", required=True, type=Path)
    parser.add_argument("--template-summary", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def acc(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    if value in ("", None):
        return None
    return float(value)


def pack(summary: dict) -> dict[str, float | None]:
    return {
        "gsk3b_any20_t0_65": acc(summary, "gsk3b_any20_t0_65"),
        "real5_any20_t0_65": acc(summary, "real5_any20_t0_65"),
        "rb_any20_t0_65": acc(summary, "rb_any20_t0_65"),
        "official_gsk3b_any20_t0_65": acc(summary, "official_gsk3b_any20_t0_65"),
        "validity": acc(summary, "validity"),
        "gsk3b_tanimoto": ((summary.get("by_task") or {}).get("GSK3B:increase") or {}).get(
            "mean_best_source_tanimoto"
        ),
    }


def at_least(value: float | None, reference: float | None, margin: float) -> bool:
    return value is not None and reference is not None and value >= reference - float(margin)


def at_most(value: float | None, reference: float | None, margin: float) -> bool:
    return value is not None and reference is not None and value <= reference + float(margin)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    keyword = json.loads(args.keyword_summary.read_text(encoding="utf-8"))
    scrambled = json.loads(args.scrambled_summary.read_text(encoding="utf-8"))
    template = json.loads(args.template_summary.read_text(encoding="utf-8"))
    b41 = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    gates = dict(prereg["gates"])
    template_gsk = acc(template, "gsk3b_any20_t0_65")
    b41_gsk = acc(b41, "gsk3b_any20_t0_65")
    keyword_gsk = acc(keyword, "gsk3b_any20_t0_65")
    scrambled_gsk = acc(scrambled, "gsk3b_any20_t0_65")
    template_margin = float(gates["near_template_gsk_margin"])
    b41_margin = float(gates["near_b41_gsk_margin"])
    keyword_near_template = at_least(keyword_gsk, template_gsk, template_margin)
    scrambled_near_template = at_least(scrambled_gsk, template_gsk, template_margin)
    keyword_near_b41 = at_most(keyword_gsk, b41_gsk, b41_margin)
    scrambled_near_b41 = at_most(scrambled_gsk, b41_gsk, b41_margin)
    validity_ok = (acc(keyword, "validity") or 0.0) >= float(gates["validity"]) and (
        acc(scrambled, "validity") or 0.0
    ) >= float(gates["validity"])
    if keyword_near_template and scrambled_near_b41 and validity_ok:
        decision = "stop_keyword_only"
        claim = "property-name tokens recover the E1 GSK3B jump; scrambled text does not"
    elif keyword_near_template and scrambled_near_template and validity_ok:
        decision = "stop_condition_slot_artifact"
        claim = "destroyed instructions still lift GSK3B; not language"
    elif keyword_near_b41 and scrambled_near_b41 and validity_ok:
        decision = "go_sentence_not_keyword"
        claim = "keywords and scrambled both fall toward B41; E1 needed fluent templates"
    else:
        decision = "stop_inconclusive"
        claim = "ablation did not match a pre-registered pattern; do not call it language"
    payload = {
        "protocol": prereg["protocol"],
        "series": "E",
        "not_ours": True,
        "not_a_method_contribution": True,
        "decision": decision,
        "checks": {
            "keyword_near_template_gsk": keyword_near_template,
            "scrambled_near_template_gsk": scrambled_near_template,
            "keyword_near_b41_gsk": keyword_near_b41,
            "scrambled_near_b41_gsk": scrambled_near_b41,
            "validity": validity_ok,
        },
        "keyword": pack(keyword),
        "scrambled": pack(scrambled),
        "template": pack(template),
        "b41": pack(b41),
        "claim": claim,
        "next": {
            "stop_keyword_only": "Do not call E1 a language head. Optional: real encoder only if composition needs syntax.",
            "stop_condition_slot_artifact": "Stop the NL story. GSK3B jump is a weak-condition / preservation artifact.",
            "go_sentence_not_keyword": "E2 composition is allowed. Still not a method row. No de novo SOTA.",
            "stop_inconclusive": "Do not proceed to E2. Inspect GSK3B Tanimoto and skipped rows first.",
        }[decision],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
