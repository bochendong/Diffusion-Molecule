from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "p22_moledit_table1_protocol_sweep" / "analyze_protocol_sweep.py"
SPEC = importlib.util.spec_from_file_location("p22_protocol", MODULE_PATH)
assert SPEC and SPEC.loader
P22 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P22)


def test_reported_moleditrl_macro_is_recovered() -> None:
    macro = P22.target_macro()
    assert abs(macro["Validity"] - 0.9662) < 1e-12
    assert abs(macro["Acc_all(0.65)"] - 0.4502) < 1e-12
    assert abs(macro["Acc_all(0.15)"] - 0.7266) < 1e-12


def test_paper_counts_use_five_hundred_outputs_per_task() -> None:
    assert P22.MOLEDITRL["GSK3B:increase"] == (476 / 500, 171 / 500, 257 / 500)
    assert P22.MOLEDITRL["RB:decrease"] == (492 / 500, 317 / 500, 415 / 500)
