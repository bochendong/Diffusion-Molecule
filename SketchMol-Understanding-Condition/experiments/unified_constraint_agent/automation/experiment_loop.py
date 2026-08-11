#!/usr/bin/env python3
"""Run a bounded, manifest-driven Slurm experiment loop.

The controller is intentionally deterministic.  It never invents a training
configuration: every runnable round, gate, transition, and resource budget must
already be present in ``experiment_plan.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
DEFAULT_INFRA_RETRY_STATES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REQUEUE_HOLD"}
SAFE_SUBMIT_EXECUTABLES = {"bash"}


class AutomationError(RuntimeError):
    """Raised when an automation contract or runtime invariant is violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AutomationError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AutomationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutomationError(f"Expected a JSON object in {path}")
    return payload


def dotted_get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise AutomationError(f"Missing required summary field: {dotted_path}")
        current = current[part]
    return current


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AutomationError(f"{label} must stay inside the shared project: {value}")
    return path


def _round_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in plan["rounds"]}


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise AutomationError(f"Unsupported plan schema_version={plan.get('schema_version')!r}")
    limits = plan.get("limits")
    if not isinstance(limits, Mapping):
        raise AutomationError("Plan requires a limits object")
    for key in ("max_scientific_rounds", "max_submissions", "max_accelerator_hours"):
        value = limits.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise AutomationError(f"limits.{key} must be positive")
    if int(limits.get("fixed_candidate_budget", 0)) != 20:
        raise AutomationError("The paper-facing automation contract fixes candidate budget at n=20")

    controller = plan.get("controller")
    if not isinstance(controller, Mapping):
        raise AutomationError("Plan requires a controller object")
    if not str(controller.get("account", "")).strip():
        raise AutomationError("controller.account is required")

    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise AutomationError("Plan requires at least one runnable round")
    identifiers = [str(item.get("id", "")) for item in rounds if isinstance(item, Mapping)]
    if len(identifiers) != len(rounds) or any(not item for item in identifiers):
        raise AutomationError("Every round requires a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise AutomationError("Round ids must be unique")
    known = set(identifiers)
    edges: dict[str, list[str]] = {item: [] for item in identifiers}

    for item in rounds:
        if not isinstance(item, Mapping):
            raise AutomationError("Every round must be a JSON object")
        round_id = str(item["id"])
        argv = item.get("submit_argv")
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(arg, str) for arg in argv):
            raise AutomationError(f"Round {round_id} requires submit_argv as a string list")
        if argv[0] not in SAFE_SUBMIT_EXECUTABLES:
            raise AutomationError(f"Round {round_id} uses disallowed executable: {argv[0]}")
        script_path = _safe_relative_path(argv[1], label=f"round {round_id} submit script")
        if script_path.suffix != ".sh":
            raise AutomationError(f"Round {round_id} must submit a .sh entrypoint")
        _safe_relative_path(str(item.get("summary_path", "")), label=f"round {round_id} summary")
        hours = item.get("requested_accelerator_hours")
        if not isinstance(hours, (int, float)) or float(hours) < 0:
            raise AutomationError(f"Round {round_id} requested_accelerator_hours must be non-negative")
        max_attempts = item.get("max_attempts", 1)
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise AutomationError(f"Round {round_id} max_attempts must be a positive integer")
        gate = item.get("gate")
        if not isinstance(gate, Mapping):
            raise AutomationError(f"Round {round_id} requires a gate object")
        transitions = gate.get("transitions", {})
        if not isinstance(transitions, Mapping):
            raise AutomationError(f"Round {round_id} gate.transitions must be an object")
        for decision, target in transitions.items():
            if target is None:
                continue
            target_id = str(target)
            if target_id not in known:
                raise AutomationError(
                    f"Round {round_id} transition {decision!r} targets unknown round {target_id!r}"
                )
            if target_id == round_id:
                raise AutomationError(f"Round {round_id} cannot transition to itself; use max_attempts for retries")
            edges[round_id].append(target_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise AutomationError("Scientific round transitions must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for target in edges[node]:
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for identifier in identifiers:
        visit(identifier)


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    validate_plan(plan)
    return plan


def project_dir(plan: Mapping[str, Any]) -> Path:
    config = plan.get("project_dir", {})
    env_name = str(config.get("env", "SUCC_UCA_AUTOMATION_REPO_DIR"))
    default = str(config.get("default", ""))
    value = os.environ.get(env_name, default)
    if not value:
        raise AutomationError(f"Set {env_name} or project_dir.default")
    return Path(value).expanduser().resolve()


def artifact_dir(plan: Mapping[str, Any]) -> Path:
    config = plan.get("artifact_dir")
    if not isinstance(config, Mapping):
        return project_dir(plan)
    env_name = str(config.get("env", "SUCC_UCA_SHARED_REPO_DIR"))
    default = str(config.get("default", ""))
    value = os.environ.get(env_name, default)
    if not value:
        raise AutomationError(f"Set {env_name} or artifact_dir.default")
    return Path(value).expanduser().resolve()


def resolve_project_path(plan: Mapping[str, Any], relative: str) -> Path:
    root = project_dir(plan)
    path = (root / _safe_relative_path(relative, label="project path")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AutomationError(f"Resolved path escapes the shared project: {path}") from exc
    return path


def resolve_artifact_path(plan: Mapping[str, Any], relative: str) -> Path:
    root = artifact_dir(plan)
    path = (root / _safe_relative_path(relative, label="artifact path")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AutomationError(f"Resolved artifact path escapes the shared artifact root: {path}") from exc
    return path


def default_state_path(plan: Mapping[str, Any]) -> Path:
    return resolve_artifact_path(plan, str(plan["state_path"]))


def new_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_digest": canonical_digest(plan),
        "status": "idle",
        "current_round_id": None,
        "active_job": None,
        "attempts_by_round": {},
        "scientific_round_ids": [],
        "submissions": 0,
        "reserved_accelerator_hours": 0.0,
        "history": [],
        "updated_at": utc_now(),
    }


def validate_state(plan: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise AutomationError("State schema does not match the controller")
    if state.get("plan_digest") != canonical_digest(plan):
        raise AutomationError(
            "Plan changed after the loop started; review it and initialize a new state file explicitly"
        )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


@contextlib.contextmanager
def locked_state(plan: Mapping[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = load_json(path) if path.exists() else new_state(plan)
        validate_state(plan, state)
        yield state
        state["updated_at"] = utc_now()
        atomic_write_json(path, state)


def append_event(state: MutableMapping[str, Any], event: str, **fields: Any) -> None:
    state.setdefault("history", []).append({"at": utc_now(), "event": event, **fields})


def run_command(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def parse_job_id(output: str, pattern: str | None = None) -> str:
    if pattern:
        matches = re.findall(pattern, output, flags=re.MULTILINE)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            if str(value).isdigit():
                return str(value)
    matches = re.findall(r"(?:Submitted batch job\s+|job(?:_id)?=)(\d+)", output)
    if matches:
        return matches[-1]
    stripped = output.strip()
    match = re.fullmatch(r"(\d+)(?:;[^\s]+)?", stripped)
    if match:
        return match.group(1)
    standalone = re.findall(r"(?m)^\s*(\d+)(?:;[^\s]+)?\s*$", output)
    if standalone:
        return standalone[-1]
    raise AutomationError(f"Could not parse a Slurm job id from output: {output[-500:]}")


def ensure_budget(plan: Mapping[str, Any], state: Mapping[str, Any], round_spec: Mapping[str, Any]) -> None:
    limits = plan["limits"]
    round_id = str(round_spec["id"])
    scientific_ids = list(state.get("scientific_round_ids", []))
    projected_ids = scientific_ids if round_id in scientific_ids else scientific_ids + [round_id]
    if len(projected_ids) > int(limits["max_scientific_rounds"]):
        raise AutomationError("Scientific-round limit reached")
    if int(state.get("submissions", 0)) + 1 > int(limits["max_submissions"]):
        raise AutomationError("Submission limit reached")
    projected_hours = float(state.get("reserved_accelerator_hours", 0.0)) + float(
        round_spec["requested_accelerator_hours"]
    )
    if projected_hours > float(limits["max_accelerator_hours"]) + 1e-12:
        raise AutomationError("Accelerator-hour budget reached")
    attempts = int(state.get("attempts_by_round", {}).get(round_id, 0))
    if attempts >= int(round_spec.get("max_attempts", 1)):
        raise AutomationError(f"Round {round_id} exhausted its attempt limit")


def controller_submit_argv(
    plan: Mapping[str, Any],
    *,
    experiment_job_id: str,
    plan_path: Path,
    state_path: Path,
) -> list[str]:
    controller = plan["controller"]
    account_env = str(controller.get("account_env", "SUCC_UCA_CONTROLLER_ACCOUNT"))
    account = os.environ.get(account_env, str(controller["account"]))
    script = resolve_project_path(
        plan,
        "SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/"
        "run_experiment_controller.sh",
    )
    log_dir = resolve_artifact_path(plan, str(controller["log_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    wrapped = " ".join(
        shlex.quote(str(value))
        for value in ("bash", script, plan_path.resolve(), state_path.resolve())
    )
    argv = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--job-name={controller.get('job_name', 'uca-auto-controller')}",
        f"--time={controller.get('time', '00:05:00')}",
        f"--cpus-per-task={int(controller.get('cpus', 1))}",
        f"--mem={controller.get('memory', '1G')}",
        f"--dependency=afterany:{experiment_job_id}",
        f"--output={log_dir}/%x-%j.log",
    ]
    mail_user = str(controller.get("mail_user", "")).strip()
    if mail_user:
        argv.extend([f"--mail-user={mail_user}", "--mail-type=FAIL"])
    argv.append(f"--wrap={wrapped}")
    return argv


def submit_round(
    plan: Mapping[str, Any],
    state: MutableMapping[str, Any],
    round_id: str,
    *,
    plan_path: Path,
    state_path: Path,
    executor: Callable[..., str] = run_command,
    attach_controller: bool = True,
) -> str:
    if state.get("active_job") is not None:
        raise AutomationError("Refusing to submit while another experiment is active")
    round_spec = _round_map(plan).get(round_id)
    if round_spec is None:
        raise AutomationError(f"Unknown round: {round_id}")
    ensure_budget(plan, state, round_spec)

    root = project_dir(plan)
    script_path = resolve_project_path(plan, round_spec["submit_argv"][1])
    if not script_path.is_file():
        raise AutomationError(f"Allowlisted submit script is missing: {script_path}")
    argv = [round_spec["submit_argv"][0], str(script_path), *round_spec["submit_argv"][2:]]
    state["status"] = "submitting"
    state["current_round_id"] = round_id
    atomic_write_json(state_path, state)

    env = dict(os.environ)
    env["SUCC_UCA_AUTOMATION_ACTIVE"] = "1"
    env.setdefault("SUCC_UCA_MAIL_USER", str(plan["controller"].get("mail_user", "")))
    try:
        output = executor(argv, cwd=root, env=env)
        job_id = parse_job_id(output, round_spec.get("job_id_regex"))
    except Exception as exc:
        state["status"] = "needs_attention"
        append_event(state, "submission_failed", round_id=round_id, error=str(exc))
        raise

    attempts = state.setdefault("attempts_by_round", {})
    attempts[round_id] = int(attempts.get(round_id, 0)) + 1
    scientific_ids = state.setdefault("scientific_round_ids", [])
    if round_id not in scientific_ids:
        scientific_ids.append(round_id)
    state["submissions"] = int(state.get("submissions", 0)) + 1
    state["reserved_accelerator_hours"] = float(state.get("reserved_accelerator_hours", 0.0)) + float(
        round_spec["requested_accelerator_hours"]
    )
    state["active_job"] = {
        "job_id": job_id,
        "round_id": round_id,
        "attempt": attempts[round_id],
        "submitted_at": utc_now(),
        "controller_job_id": None,
    }
    state["status"] = "running"
    append_event(state, "experiment_submitted", round_id=round_id, job_id=job_id)

    if attach_controller:
        try:
            controller_output = executor(
                controller_submit_argv(
                    plan,
                    experiment_job_id=job_id,
                    plan_path=plan_path,
                    state_path=state_path,
                ),
                cwd=root,
                env=env,
            )
            controller_job_id = parse_job_id(controller_output)
            state["active_job"]["controller_job_id"] = controller_job_id
            append_event(
                state,
                "controller_submitted",
                round_id=round_id,
                job_id=job_id,
                controller_job_id=controller_job_id,
            )
        except Exception as exc:
            state["status"] = "running_unwatched"
            append_event(state, "controller_submission_failed", job_id=job_id, error=str(exc))
    return job_id


def normalize_slurm_state(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        return "UNKNOWN"
    return normalized.split("+")[0].split()[0]


def parse_sacct(output: str, job_id: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or fields[0] != str(job_id):
            continue
        exit_code = fields[2].split(":", 1)[0]
        return {
            "state": normalize_slurm_state(fields[1]),
            "exit_code": int(exit_code) if exit_code.isdigit() else None,
            "elapsed_seconds": int(fields[3]) if fields[3].isdigit() else None,
        }
    return None


def query_slurm(job_id: str, *, root: Path, executor: Callable[..., str] = run_command) -> dict[str, Any]:
    sacct_output = executor(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode,ElapsedRaw"],
        cwd=root,
        env=os.environ,
    )
    record = parse_sacct(sacct_output, job_id)
    if record is not None:
        return record
    squeue_output = executor(["squeue", "-h", "-j", job_id, "-o", "%T"], cwd=root, env=os.environ)
    state = normalize_slurm_state(squeue_output) if squeue_output.strip() else "UNKNOWN"
    return {"state": state, "exit_code": None, "elapsed_seconds": None}


def evaluate_gate(round_spec: Mapping[str, Any], summary: Mapping[str, Any]) -> tuple[str, str | None]:
    gate = round_spec["gate"]
    errors: list[str] = []
    for path, expected in gate.get("required_equal", {}).items():
        actual = dotted_get(summary, path)
        if actual != expected:
            errors.append(f"{path}={actual!r}, expected {expected!r}")
    for path, threshold in gate.get("required_at_least", {}).items():
        actual = dotted_get(summary, path)
        if float(actual) < float(threshold):
            errors.append(f"{path}={actual!r}, required >= {threshold!r}")
    for path, threshold in gate.get("required_at_most", {}).items():
        actual = dotted_get(summary, path)
        if float(actual) > float(threshold):
            errors.append(f"{path}={actual!r}, required <= {threshold!r}")
    if errors:
        raise AutomationError("Gate contract failed: " + "; ".join(errors))
    decision_path = str(gate.get("decision_path", "decision"))
    decision = str(dotted_get(summary, decision_path)).strip().lower()
    transitions = {str(key).lower(): value for key, value in gate.get("transitions", {}).items()}
    if decision not in transitions:
        raise AutomationError(f"Gate returned undeclared decision {decision!r}")
    target = transitions[decision]
    return decision, None if target is None else str(target)


def reconcile_terminal(
    plan: Mapping[str, Any],
    state: MutableMapping[str, Any],
    slurm: Mapping[str, Any],
    *,
    plan_path: Path,
    state_path: Path,
    executor: Callable[..., str] = run_command,
    attach_controller: bool = True,
) -> None:
    active = state.get("active_job")
    if not isinstance(active, Mapping):
        raise AutomationError("No active experiment is registered")
    job_id = str(active["job_id"])
    round_id = str(active["round_id"])
    round_spec = _round_map(plan)[round_id]
    slurm_state = normalize_slurm_state(str(slurm.get("state", "UNKNOWN")))
    if slurm_state in ACTIVE_SLURM_STATES:
        state["status"] = "running"
        append_event(state, "experiment_still_active", round_id=round_id, job_id=job_id, slurm_state=slurm_state)
        return

    exit_code = slurm.get("exit_code")
    completed = slurm_state == "COMPLETED" and (exit_code is None or int(exit_code) == 0)
    if not completed:
        append_event(
            state,
            "experiment_failed",
            round_id=round_id,
            job_id=job_id,
            slurm_state=slurm_state,
            exit_code=exit_code,
        )
        state["active_job"] = None
        retry_states = {
            normalize_slurm_state(value)
            for value in plan["limits"].get("infra_retry_states", DEFAULT_INFRA_RETRY_STATES)
        }
        attempts = int(state.get("attempts_by_round", {}).get(round_id, 0))
        if slurm_state in retry_states and attempts < int(round_spec.get("max_attempts", 1)):
            append_event(state, "infra_retry_approved", round_id=round_id, prior_job_id=job_id)
            submit_round(
                plan,
                state,
                round_id,
                plan_path=plan_path,
                state_path=state_path,
                executor=executor,
                attach_controller=attach_controller,
            )
            return
        state["status"] = "needs_attention"
        return

    summary_path = resolve_artifact_path(plan, str(round_spec["summary_path"]))
    try:
        summary = load_json(summary_path)
        decision, target = evaluate_gate(round_spec, summary)
    except Exception as exc:
        append_event(
            state,
            "gate_validation_failed",
            round_id=round_id,
            job_id=job_id,
            summary_path=str(summary_path),
            error=str(exc),
        )
        state["active_job"] = None
        state["status"] = "needs_attention"
        return

    append_event(
        state,
        "gate_decided",
        round_id=round_id,
        job_id=job_id,
        decision=decision,
        next_round_id=target,
        summary_path=str(summary_path),
        elapsed_seconds=slurm.get("elapsed_seconds"),
    )
    state["active_job"] = None
    if target is None:
        state["status"] = "complete" if decision == "advance" else "stopped"
        return
    submit_round(
        plan,
        state,
        target,
        plan_path=plan_path,
        state_path=state_path,
        executor=executor,
        attach_controller=attach_controller,
    )


def cli_parser() -> argparse.ArgumentParser:
    default_plan = Path(__file__).with_name("experiment_plan.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=default_plan)
    parser.add_argument("--state", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    start = subparsers.add_parser("start")
    start.add_argument("--round", required=True)
    adopt = subparsers.add_parser("adopt")
    adopt.add_argument("--round", required=True)
    adopt.add_argument("--job-id", required=True)
    adopt.add_argument("--without-controller", action="store_true")
    attach_controller = subparsers.add_parser("attach-controller")
    attach_controller.add_argument("--job-id", required=True)
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--slurm-state", default=None)
    reconcile.add_argument("--exit-code", type=int, default=None)
    reconcile.add_argument("--elapsed-seconds", type=int, default=None)
    reconcile.add_argument("--without-controller", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = cli_parser().parse_args(argv)
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    state_path = args.state.resolve() if args.state is not None else default_state_path(plan)
    if args.command == "validate":
        print(json.dumps({"decision": "valid", "plan": str(plan_path), "state": str(state_path)}, indent=2))
        return 0
    if args.command == "status":
        state = load_json(state_path) if state_path.exists() else new_state(plan)
        validate_state(plan, state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    with locked_state(plan, state_path) as state:
        if args.command == "start":
            if state.get("status") != "idle":
                raise AutomationError(f"Loop is already initialized with status={state.get('status')}")
            submit_round(
                plan,
                state,
                args.round,
                plan_path=plan_path,
                state_path=state_path,
            )
        elif args.command == "adopt":
            if state.get("status") != "idle":
                raise AutomationError(f"Loop is already initialized with status={state.get('status')}")
            round_spec = _round_map(plan).get(args.round)
            if round_spec is None:
                raise AutomationError(f"Unknown round: {args.round}")
            ensure_budget(plan, state, round_spec)
            attempts = state.setdefault("attempts_by_round", {})
            attempts[args.round] = 1
            state["scientific_round_ids"] = [args.round]
            state["submissions"] = 1
            state["reserved_accelerator_hours"] = float(round_spec["requested_accelerator_hours"])
            state["current_round_id"] = args.round
            state["active_job"] = {
                "job_id": str(args.job_id),
                "round_id": args.round,
                "attempt": 1,
                "submitted_at": None,
                "controller_job_id": None,
            }
            state["status"] = "running"
            append_event(state, "experiment_adopted", round_id=args.round, job_id=str(args.job_id))
            if not args.without_controller:
                output = run_command(
                    controller_submit_argv(
                        plan,
                        experiment_job_id=str(args.job_id),
                        plan_path=plan_path,
                        state_path=state_path,
                    ),
                    cwd=project_dir(plan),
                    env=os.environ,
                )
                state["active_job"]["controller_job_id"] = parse_job_id(output)
        elif args.command == "attach-controller":
            active = state.get("active_job")
            if not isinstance(active, MutableMapping):
                raise AutomationError("No active experiment is registered")
            existing = str(active.get("controller_job_id", "") or "").strip()
            if existing and existing != str(args.job_id):
                raise AutomationError(
                    f"Active experiment already has controller_job_id={existing}"
                )
            active["controller_job_id"] = str(args.job_id)
            if state.get("status") == "running_unwatched":
                state["status"] = "running"
            append_event(
                state,
                "controller_attached",
                round_id=str(active.get("round_id", "")),
                job_id=str(active.get("job_id", "")),
                controller_job_id=str(args.job_id),
            )
        elif args.command == "reconcile":
            active = state.get("active_job")
            if not isinstance(active, Mapping):
                print(json.dumps(state, indent=2, sort_keys=True))
                return 0
            if args.slurm_state is None:
                slurm = query_slurm(str(active["job_id"]), root=project_dir(plan))
            else:
                slurm = {
                    "state": args.slurm_state,
                    "exit_code": args.exit_code,
                    "elapsed_seconds": args.elapsed_seconds,
                }
            reconcile_terminal(
                plan,
                state,
                slurm,
                plan_path=plan_path,
                state_path=state_path,
                attach_controller=not args.without_controller,
            )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AutomationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
