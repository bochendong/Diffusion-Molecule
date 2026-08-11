from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_constraint_agent"
    / "automation"
    / "experiment_loop.py"
)
SPEC = importlib.util.spec_from_file_location("unified_constraint_experiment_loop", MODULE_PATH)
assert SPEC and SPEC.loader
automation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = automation
SPEC.loader.exec_module(automation)


def make_plan(tmp_path: Path, *, transition: str | None = "round_b") -> dict[str, object]:
    for name in ("submit_a.sh", "submit_b.sh"):
        (tmp_path / name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "project_dir": {"env": "TEST_UCA_PROJECT_DIR", "default": str(tmp_path)},
        "state_path": "state.json",
        "limits": {
            "max_scientific_rounds": 3,
            "max_submissions": 4,
            "max_accelerator_hours": 6.0,
            "fixed_candidate_budget": 20,
            "infra_retry_states": ["NODE_FAIL", "PREEMPTED"],
        },
        "controller": {
            "account": "def-test",
            "job_name": "test-controller",
            "time": "00:05:00",
            "cpus": 1,
            "memory": "1G",
            "mail_user": "test@example.com",
            "log_dir": "logs",
        },
        "rounds": [
            {
                "id": "round_a",
                "submit_argv": ["bash", "submit_a.sh"],
                "job_id_regex": r"job=(\d+)",
                "summary_path": "summary_a.json",
                "requested_accelerator_hours": 1.0,
                "max_attempts": 2,
                "gate": {
                    "decision_path": "decision",
                    "required_equal": {
                        "protocol": "test-v1",
                        "candidate_budget": 20,
                        "split.source_overlap": 0,
                    },
                    "required_at_least": {"metrics.valid_rate": 1.0},
                    "transitions": {"advance": transition, "stop": None},
                },
            },
            {
                "id": "round_b",
                "submit_argv": ["bash", "submit_b.sh"],
                "job_id_regex": r"job=(\d+)",
                "summary_path": "summary_b.json",
                "requested_accelerator_hours": 2.0,
                "max_attempts": 1,
                "gate": {
                    "decision_path": "decision",
                    "required_equal": {"candidate_budget": 20},
                    "transitions": {"advance": None, "stop": None},
                },
            },
        ],
    }


def write_plan(tmp_path: Path, plan: dict[str, object]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def active_state(plan: dict[str, object], *, slurm_job_id: str = "111") -> dict[str, object]:
    state = automation.new_state(plan)
    state.update(
        {
            "status": "running",
            "current_round_id": "round_a",
            "active_job": {
                "job_id": slurm_job_id,
                "round_id": "round_a",
                "attempt": 1,
                "submitted_at": None,
                "controller_job_id": "112",
            },
            "attempts_by_round": {"round_a": 1},
            "scientific_round_ids": ["round_a"],
            "submissions": 1,
            "reserved_accelerator_hours": 1.0,
        }
    )
    return state


def valid_summary(decision: str = "advance") -> dict[str, object]:
    return {
        "protocol": "test-v1",
        "candidate_budget": 20,
        "decision": decision,
        "split": {"source_overlap": 0},
        "metrics": {"valid_rate": 1.0},
    }


def test_plan_rejects_budget_drift_and_transition_cycles(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    automation.validate_plan(plan)

    plan["limits"]["fixed_candidate_budget"] = 21
    with pytest.raises(automation.AutomationError, match="n=20"):
        automation.validate_plan(plan)

    plan["limits"]["fixed_candidate_budget"] = 20
    plan["rounds"][1]["gate"]["transitions"]["advance"] = "round_a"
    with pytest.raises(automation.AutomationError, match="acyclic"):
        automation.validate_plan(plan)


def test_plan_rejects_submit_paths_outside_the_project(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    plan["rounds"][0]["submit_argv"] = ["bash", "../unsafe.sh"]
    with pytest.raises(automation.AutomationError, match="inside the shared project"):
        automation.validate_plan(plan)


def test_code_and_artifact_roots_are_independently_contained(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    artifact_root = tmp_path / "artifacts"
    code_root.mkdir()
    artifact_root.mkdir()
    plan = make_plan(code_root)
    plan["artifact_dir"] = {
        "env": "TEST_UCA_ARTIFACT_DIR",
        "default": str(artifact_root),
    }

    assert automation.resolve_project_path(plan, "submit_a.sh") == code_root / "submit_a.sh"
    assert automation.resolve_artifact_path(plan, "summary_a.json") == artifact_root / "summary_a.json"


def test_gate_requires_protocol_completeness_before_transition(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    round_a = plan["rounds"][0]
    assert automation.evaluate_gate(round_a, valid_summary()) == ("advance", "round_b")

    incomplete = valid_summary()
    incomplete["metrics"]["valid_rate"] = 0.98
    with pytest.raises(automation.AutomationError, match="Gate contract failed"):
        automation.evaluate_gate(round_a, incomplete)


def test_completed_round_advances_once_to_allowlisted_next_round(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    plan_path = write_plan(tmp_path, plan)
    state_path = tmp_path / "state.json"
    state = active_state(plan)
    (tmp_path / "summary_a.json").write_text(json.dumps(valid_summary()), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_executor(argv, *, cwd, env):
        calls.append(list(argv))
        if argv[0] == "bash":
            return "job=222\n"
        assert argv[0] == "sbatch"
        return "333;test-cluster\n"

    automation.reconcile_terminal(
        plan,
        state,
        {"state": "COMPLETED", "exit_code": 0, "elapsed_seconds": 60},
        plan_path=plan_path,
        state_path=state_path,
        executor=fake_executor,
    )

    assert state["status"] == "running"
    assert state["active_job"]["round_id"] == "round_b"
    assert state["active_job"]["job_id"] == "222"
    assert state["active_job"]["controller_job_id"] == "333"
    assert state["submissions"] == 2
    assert state["reserved_accelerator_hours"] == 3.0
    assert [call[0] for call in calls] == ["bash", "sbatch"]


def test_scientific_stop_is_terminal_and_never_submits(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    plan_path = write_plan(tmp_path, plan)
    state_path = tmp_path / "state.json"
    state = active_state(plan)
    (tmp_path / "summary_a.json").write_text(json.dumps(valid_summary("stop")), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("A STOP decision must not submit anything")

    automation.reconcile_terminal(
        plan,
        state,
        {"state": "COMPLETED", "exit_code": 0, "elapsed_seconds": 60},
        plan_path=plan_path,
        state_path=state_path,
        executor=fail_if_called,
    )

    assert state["status"] == "stopped"
    assert state["active_job"] is None
    assert state["submissions"] == 1


def test_only_infrastructure_failures_receive_one_retry(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    plan_path = write_plan(tmp_path, plan)
    state_path = tmp_path / "state.json"
    state = active_state(plan)

    def fake_executor(argv, *, cwd, env):
        return "job=444\n" if argv[0] == "bash" else "445;test-cluster\n"

    automation.reconcile_terminal(
        plan,
        state,
        {"state": "PREEMPTED", "exit_code": 0, "elapsed_seconds": 10},
        plan_path=plan_path,
        state_path=state_path,
        executor=fake_executor,
    )

    assert state["status"] == "running"
    assert state["active_job"]["job_id"] == "444"
    assert state["attempts_by_round"]["round_a"] == 2

    automation.reconcile_terminal(
        plan,
        state,
        {"state": "PREEMPTED", "exit_code": 0, "elapsed_seconds": 10},
        plan_path=plan_path,
        state_path=state_path,
        executor=fake_executor,
    )
    assert state["status"] == "needs_attention"
    assert state["active_job"] is None


def test_code_failure_and_missing_summary_stop_for_attention(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    plan_path = write_plan(tmp_path, plan)
    state_path = tmp_path / "state.json"
    state = active_state(plan)

    automation.reconcile_terminal(
        plan,
        state,
        {"state": "FAILED", "exit_code": 1, "elapsed_seconds": 10},
        plan_path=plan_path,
        state_path=state_path,
        executor=lambda *args, **kwargs: "",
    )
    assert state["status"] == "needs_attention"

    state = active_state(plan)
    automation.reconcile_terminal(
        plan,
        state,
        {"state": "COMPLETED", "exit_code": 0, "elapsed_seconds": 10},
        plan_path=plan_path,
        state_path=state_path,
        executor=lambda *args, **kwargs: "",
    )
    assert state["status"] == "needs_attention"
    assert state["history"][-1]["event"] == "gate_validation_failed"


def test_sacct_parser_ignores_step_rows() -> None:
    output = "111.batch|COMPLETED|0:0|9\n111|COMPLETED|0:0|10\n"
    assert automation.parse_sacct(output, "111") == {
        "state": "COMPLETED",
        "exit_code": 0,
        "elapsed_seconds": 10,
    }


def test_checked_in_plan_declares_only_leakage_safe_v5() -> None:
    plan_path = MODULE_PATH.parent / "experiment_plan.json"
    plan = automation.load_plan(plan_path)

    assert plan["limits"]["fixed_candidate_budget"] == 20
    assert plan["state_path"].endswith("state_v5.json")
    assert [round_["id"] for round_ in plan["rounds"]] == ["retrieved_delta_support_v5"]
    required = plan["rounds"][0]["gate"]["required_equal"]
    assert required["candidate_builder.evaluation_target_access"] is False
    assert required["final_oracle_candidate_budget"] == 20
