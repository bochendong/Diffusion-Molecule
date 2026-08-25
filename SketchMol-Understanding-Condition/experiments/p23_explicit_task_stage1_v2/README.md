# P23: explicit-task Stage 1 v2

P23 is a new versioned continuation of MolProgram Stage 1. It does **not**
rewrite the frozen P18/P19 evidence. The same Qwen2.5-VL-7B checkpoint, tokenizer,
LoRA, prompt schema, and source-based mode expression are retained.

The initial explicit-task gate targets two concrete data failures found after P21:

1. edit conditions come only from `instruction_tasks` or the paired
   `instruction_task_*` fields; target-derived incidental property deltas are
   forbidden;
2. GSK3B, DRD2, and JNK3 are first-class serialized properties rather than being
   dropped by the old eight-property allowlist.

## Stage-1 curriculum

The default run selects 12,000 de-novo and 12,000 edit positives with exact 1:1
mode weight. De-novo rows are balanced across 2p-7p. Edit rows are balanced over
explicit task programs after a source-Tanimoto >= 0.65 filter. The 50k balanced
MolEdit-Instruct evaluation split and any available frozen P19/P20 references are
excluded by canonical source and target hashes.

Sixty percent of the edit budget is reserved for the ten paper task families.
A train row belongs to a paper family when it explicitly contains every clause of
that task; additional explicit clauses remain in the prompt and are never hidden.
The remaining 40% preserves broad instruction coverage.

The completed 6k gate showed that this explicit-only reserve still supplied just
six of the ten paper families. The full paper run therefore enables
`P23_ORACLE_ALIGNED_PAPER_EDITS=1`. In this amended mode, 7,200 edit rows are
split into ten exact quotas of 720. A training source-target pair is projected
onto a paper task only when the benchmark descriptor or pinned assay model
confirms every requested direction. The remaining 4,800 rows retain their
explicit instruction programs. This is train-only label construction: oracle
values never appear in prompts, generation, or candidate selection. Canonical
held-out source/target exclusion runs before projection, pairs cannot be reused
across paper quotas, and a missing quota fails closed.

Training has two continuations inside Stage 1:

1. positive SFT from the frozen P16 mixed adapter (`0.5` epoch by default);
2. normalized contrastive refinement (`0.25` epoch) with source-copy, invalid,
   condition-mismatch, opposite-program, and partial-program negatives when the
   relevant train-only donors exist.

Each unique row contributes total chosen-CE weight 1.0 regardless of its number
of negatives. This is contrastive SFT, not ORPO and not RL.

## Nibi

```bash
cd /scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition
bash experiments/p23_explicit_task_stage1_v2/submit_p23.sh
```

The aligned 24k paper run additionally exports
`P23_ORACLE_ALIGNED_PAPER_EDITS=1`, `P23_PAPER_ROWS_PER_TASK=720`, and a separate
output root. Because frozen-reference exclusion removes one row from the exact
2p source pool, it also sets `P23_DENOVO_FALLBACK_CSV` to the same-source
candidate pool. `augment_denovo_pool.py` selects one heldout-disjoint donor and
projects it to MW+QED, restoring 2,000 rows per property count without restoring
the leaked row. The Nibi job chain is recorded in `RESULTS.md`.

For a smaller plumbing run, set `P23_ROWS_PER_MODE=1000`,
`P23_SFT_EPOCHS=0.25`, and `P23_CONTRASTIVE_EPOCHS=0.1` before submission.
The full adapter is written to
`outputs/p23_explicit_task_stage1_v2/seed_2323/model/stage1_v2/adapter`.

P23 results must be evaluated with prompts rebuilt by `p23_protocol.py`; reusing
the frozen P19 prompt JSONL would reintroduce the protocol bug this experiment is
designed to fix.
