# Bounded Unified-Constraint Experiment Loop

This directory contains a deterministic Slurm controller for scientific
experiment ladders. Email remains a notification channel; Slurm state and the
versioned JSON gate artifact are the only control inputs.

The controller deliberately cannot invent a new experiment. Every submit
script, artifact path, protocol invariant, decision, transition, retry limit,
and accelerator-hour reservation must be declared in `experiment_plan.json`.
Commands are executed without a shell and submit scripts must be relative,
allowlisted Bash entrypoints inside the shared repository.

Code and artifacts use separate roots. `SUCC_UCA_AUTOMATION_REPO_DIR` points
to the clean checkout that supplies controller and experiment code, while
`SUCC_UCA_SHARED_REPO_DIR` points to the persistent checkout or filesystem
that owns checkpoints, outputs, state, and logs. Both roots enforce relative
path containment independently; artifact symlinks cannot weaken the code-path
allowlist.

## Safety contract

- The paper-facing final candidate budget is fixed at `n=20`.
- Only `BOOT_FAIL`, `NODE_FAIL`, `PREEMPTED`, and `REQUEUE_HOLD` are eligible
  for one infrastructure retry. Timeouts, OOMs, code failures, missing
  artifacts, and malformed summaries stop with `needs_attention`.
- A file lock and atomic state writes prevent two controller jobs from
  advancing the same round.
- A plan digest prevents an edited manifest from silently changing an active
  loop. Start a new state file after reviewing a changed plan.
- The default plan permits at most three scientific rounds, four submissions,
  and six reserved accelerator-hours.
- A gate can transition to another round only when that exact round and submit
  entrypoint already exist in the manifest. Transition graphs must be acyclic.

## Nibi lifecycle

Run the initial submission once from a login node:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/submit_experiment_loop.sh
```

The command submits the experiment and a small CPU controller job with
`afterany:<experiment_job_id>`. The controller verifies the Slurm exit status,
loads the exact JSON gate, enforces the protocol invariants, and either stops or
submits the declared next round plus its next controller job. No later SSH or
Duo interaction is required for declared transitions.

The default controller account is `def-hup-ab_cpu`, verified against the Nibi
association list. If the allocation changes, set `SUCC_UCA_CONTROLLER_ACCOUNT`
for the initial command; the value is inherited by the dependent controller
jobs.

On Nibi, a clean permanent worktree can therefore run against the existing
shared artifacts without touching a dirty research checkout:

```bash
export SUCC_UCA_AUTOMATION_REPO_DIR=/scratch/bdong/projects/Diffusion-Molecule-automation
export SUCC_UCA_SHARED_REPO_DIR=/scratch/bdong/projects/Diffusion-Molecule
bash "$SUCC_UCA_AUTOMATION_REPO_DIR/SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/submit_experiment_loop.sh"
```

Inspect the durable state without changing it:

```bash
python3 SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/experiment_loop.py status
```

An already-running job can be adopted once:

```bash
python3 SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/experiment_loop.py \
  adopt --round hierarchical_support_v4 --job-id JOB_ID
```

Use `--without-controller` only for a manual reconciliation. A controller
submission failure leaves the experiment in `running_unwatched`; it never
submits a duplicate experiment.

## Extending the ladder

The checked-in plan contains only the validated v4 support gate. Its
`advance` and `stop` decisions are terminal because the source-preserving delta
proposal experiment does not yet have a reviewed submit entrypoint. Add that
round and then set the desired transition only after its standalone dry run and
gate schema pass. This keeps the automation mechanism separate from scientific
method design.
