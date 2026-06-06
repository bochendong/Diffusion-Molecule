#!/usr/bin/env python3
"""Train the optional PyTorch dual-stream model."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.train import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

