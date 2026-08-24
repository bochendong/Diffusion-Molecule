#!/usr/bin/env python3
"""Allow the temperature-only R2 only when R1's primary failure is high entropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--minimum", type=float, default=0.85)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    entropy = float(payload["mean_normalized_policy_entropy"])
    print(f"r1_mean_normalized_policy_entropy={entropy:.8f} threshold={args.minimum:.8f}")
    if entropy < float(args.minimum):
        raise SystemExit(
            "R2 temperature intervention refused: R1 is not a high-entropy policy failure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

