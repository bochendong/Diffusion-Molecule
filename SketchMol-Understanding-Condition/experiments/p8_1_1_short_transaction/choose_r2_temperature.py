#!/usr/bin/env python3
"""Choose the direction of the mandatory temperature-only R2 from R1 entropy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--high-entropy-temperature", type=float, default=0.25)
    parser.add_argument("--low-entropy-temperature", type=float, default=1.5)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    entropy = float(payload["mean_normalized_policy_entropy"])
    high_entropy = entropy >= float(args.threshold)
    temperature = (
        float(args.high_entropy_temperature)
        if high_entropy
        else float(args.low_entropy_temperature)
    )
    direction = "concentrate" if high_entropy else "diversify"
    print(
        f"R1 normalized entropy={entropy:.8f}; R2 will {direction} raw sampling "
        f"with temperature={temperature:.8f}",
        file=sys.stderr,
    )
    # stdout is intentionally machine-readable for the shell driver.
    print(f"{temperature:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

