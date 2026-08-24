#!/usr/bin/env python3
"""Choose the direction of the mandatory one-factor R2 from R1 diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--identity-threshold", type=float, default=0.20)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    identity = float(payload["candidate_identity_fraction"])
    if identity >= float(args.identity_threshold):
        fraction, rationale = 0.60, "identity collapse: enlarge the editable source span"
    else:
        fraction, rationale = 0.20, "non-identity dominates: tighten the source anchor"
    print(f"R1 identity={identity:.8f}; {rationale}; R2 mask_fraction={fraction:.2f}", file=sys.stderr)
    print(f"{fraction:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

