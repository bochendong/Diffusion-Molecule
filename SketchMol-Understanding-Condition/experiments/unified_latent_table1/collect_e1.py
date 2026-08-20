#!/usr/bin/env python3
"""E1 gates: template NL vs paraphrase NL vs frozen B41. Diagnostic only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--template-summary", required=True, type=Path)
    parser.add_argument("--paraphrase-summary", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--train-summary", type=Path, default=None)
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
    }


def within_margin(value: float | None, reference: float | None, margin: float) -> bool:
    return value is not None and reference is not None and value >= reference - float(margin)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    template = json.loads(args.template_summary.read_text(encoding="utf-8"))
    paraphrase = json.loads(args.paraphrase_summary.read_text(encoding="utf-8"))
    b41 = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    train = {}
    if args.train_summary is not None and args.train_summary.exists():
        train = json.loads(args.train_summary.read_text(encoding="utf-8"))
    gates = dict(prereg["gates"])
    template_real5 = acc(template, "real5_any20_t0_65")
    paraphrase_real5 = acc(paraphrase, "real5_any20_t0_65")
    b41_real5 = acc(b41, "real5_any20_t0_65")
    template_ok = within_margin(template_real5, b41_real5, float(gates["template_real5_margin"]))
    paraphrase_ok = within_margin(
        paraphrase_real5, b41_real5, float(gates["paraphrase_real5_margin"])
    )
    validity_ok = (acc(template, "validity") or 0.0) >= float(gates["validity"]) and (
        acc(paraphrase, "validity") or 0.0
    ) >= float(gates["validity"])
    checks = {
        "template_real5_within_margin": template_ok,
        "paraphrase_real5_within_margin": paraphrase_ok,
        "validity": validity_ok,
    }
    if template_ok and paraphrase_ok and validity_ok:
        decision = "go_language_interface"
        claim = "NL projector drives frozen B41 on templates and held-out paraphrases"
    elif template_ok and validity_ok:
        decision = "stop_template_id_only"
        claim = "templates work; paraphrases do not. Text is a task-id API, not language"
    else:
        decision = "stop_head_not_aligned"
        claim = "instruction projector did not keep frozen B41 on Table1 templates"
    payload = {
        "protocol": prereg["protocol"],
        "series": "E",
        "not_ours": True,
        "not_a_method_contribution": True,
        "decision": decision,
        "checks": checks,
        "template": pack(template),
        "paraphrase": pack(paraphrase),
        "b41": pack(b41),
        "train_mse": train.get("train_mse"),
        "paraphrase_token_mse": train.get("paraphrase_token_mse"),
        "claim": claim,
        "next": {
            "go_language_interface": "E2 composition / E3 joint train. Do not claim de novo SOTA.",
            "stop_template_id_only": "Do not call this a language contribution. MiniLM/Qwen only if paraphrase is the gap.",
            "stop_head_not_aligned": "Fix alignment before any larger language model.",
        }[decision],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
