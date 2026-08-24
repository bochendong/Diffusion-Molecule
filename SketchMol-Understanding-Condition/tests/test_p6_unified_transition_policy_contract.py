from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "experiments" / "p6_unified_molecular_transition_policy"
PROGRAM = P6 / "p6_transition_program.py"
RUNNER = P6 / "run_p6_unified_transition_gate.sh"
SUBMIT = P6 / "submit_p6_unified_transition_gate.sh"


def test_p6_sources_parse_and_shells_are_valid() -> None:
    ast.parse(PROGRAM.read_text(encoding="utf-8"))
    for script in (RUNNER, SUBMIT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_p6_has_one_mode_agnostic_program_contract() -> None:
    source = PROGRAM.read_text(encoding="utf-8")
    assert 'PROGRAM = "<MOL_PROGRAM>"' in source
    assert 'INIT_ATOM = "<INIT_ATOM>"' in source
    assert 'ADD_ATOM = "<ADD_ATOM>"' in source
    assert 'ADD_BOND = "<ADD_BOND>"' in source
    assert "def execute_program(initial_smiles" in source
    assert "def source_for_row" in source
    assert "task_router\": False" in source
    assert "task_specific_head\": False" in source
    assert "property_aware_finalizer\": False" in source


def test_p6_is_single_seed_and_reports_honest_low_budget_metrics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'SEED="${P6_SEED:-7}"' in source
    assert "--num-samples 20" in source
    assert "--budgets 1,8,20" in source
    assert "--disable-finalizer" not in source
    assert "evaluate_moledit_table1_anyk.py" in source


def test_p6_preregistration_is_bounded() -> None:
    payload = json.loads((P6 / "p6_preregistration.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["candidate_budgets"] == [1, 8, 20]
    assert payload["architecture"]["decoder_count"] == 1
    assert payload["architecture"]["task_router"] is False
