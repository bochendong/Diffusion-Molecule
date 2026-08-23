# P1: Direct SMILES Property Programs + Group-RL

P1 is the numbered, GraphEditDSL-independent paper direction for testing
compositional molecular objective extrapolation. It reuses the frozen Direct
SMILES v2 SFT and Group-RL checkpoints; the first gate performs no new training.

The claim under test is deliberately narrower than the historical n=128/256
best-of-k headline: Group-RL must beat SFT at practical budgets (`k=8/20`) and
in the raw strict-success fraction, especially for 6p/7p and OOD conditions.

## Frozen first gate

- seed: `7` for candidate generation;
- candidates per condition: exactly `256`;
- budgets: `1,4,8,20,32,64,128,256`;
- paired models: existing v2 SFT and existing Group-RL v1;
- paired benchmarks: 2p-7p and OOD;
- OOD decoding is the existing conservative configuration for both models;
- no external molecular library or materializer;
- no new training;
- no property-reranked selected molecule is counted as one-shot.

The raw candidate CSVs retain generation order. The finalizer reports raw
strict-success fraction, empirical prefix pass@k, estimated pass@k from all
n=256 outcomes, validity, and unique-valid yield. It stratifies 2p-7p by
property count and OOD by `forward_extreme`, `rare_combo`, and
`reverse_stimulation`, with paired condition-bootstrap confidence intervals.

The complete machine-readable contract is
`p1_property_program_group_rl_preregistration.json`.

## Nibi run

After pulling the synchronized commit on Nibi:

```bash
bash SketchMol-Understanding-Condition/experiments/p1_property_program_group_rl/submit_p1_sampling_scaling.sh
```

This submits four independent GPU jobs and one `afterok` CPU finalizer. The
default output root is:

```text
SketchMol-Understanding-Condition/outputs/p1_property_program_group_rl_seed7/
```

Primary artifacts:

- `final/p1_report.md`
- `final/p1_gate.json`
- `final/p1_scaling_summary.csv`
- `final/p1_paired_deltas.csv`
- `final/p1_condition_metrics.csv`
- `final/p1_validity_audit.csv`
- `final/p1_paper_main_table.csv`

`p1_validity_audit.csv` separates two quantities that older benchmark tables
often collapse into the word `validity`: raw candidate-level RDKit validity,
and selected validity@k (the fraction of conditions with at least one valid
candidate in the first k draws). Historical best-of-k validity is comparable
to the latter, not to raw candidate validity. The audit also separates empty
decodes from nonempty strings that fail RDKit parsing.

`p1_paper_main_table.csv` joins the absolute SFT and Group-RL values with the
paired deltas and confidence intervals, so paper tables and plots do not need
to reconstruct joins from separate files.

The generator also writes `diagnostic_property_reranked_selected.csv` for
backward compatibility. P1 excludes that file from every reported metric.

GPU jobs default to the Nibi `def-hup-ab_gpu` account; the CPU finalizer uses
`def-hup-ab`. Override them separately with `SUCC_P1_GPU_ACCOUNT` and
`SUCC_P1_CPU_ACCOUNT` when needed.

If a historical Group-RL checkpoint has to be reconstructed from its frozen
SFT parent, `validate_p1_recovered_checkpoint.py` compares its arguments and
one-epoch training history with source jobs `16583941` and `16742519`. The
Group-RL candidate job stops before inference if this provenance guard fails.

For infrastructure-interrupted jobs, the evaluator accepts
`--allow-condition-intersection` to produce a clearly prefixed `interim_*`
verdict from conditions that have all 256 raw candidates in both models. This
diagnostic never replaces the preregistered full-coverage final gate.

When scheduling latency is the bottleneck, run
`submit_p1_fast_hard_gate.sh`. It freezes 128 conditions each from 6p and 7p,
generates only the first 20 raw candidates, and reports k=1/4/8/20. Its
one-hour GPU walltime is designed for backfill and answers the missing
hard-complexity kill question quickly; it is explicitly not the final n=256
P1 result. Re-running its CPU finalizer is sufficient to add the validity audit
to an already completed fast candidate pool; no GPU regeneration is required.

## Source consistency and validity pilot

`submit_p1_source_consistency_validity_pilot.sh` tests a unified correction for
the two current failure modes on validation data only.  It treats a molecular
edit as a frame-to-frame transition: the property program specifies what may
change, while source fingerprint similarity, Bemis--Murcko scaffold retention,
and local graph-edit magnitude measure what should remain consistent.  These
signals rerank executable GraphEditDSL actions without reading target
molecules or output property oracles.  The same protected common decoder is
evaluated on de novo rows with grammar-constrained SMILES decoding,
`no_repeat_ngram_size=0`, and a reduced repetition penalty.

The single-seed gate requires both an edit improvement at raw `n=1` and at
least a 10-point de novo raw-validity gain without losing more than two points
of strict success.  Only a passing validation configuration is eligible for a
frozen full-test run.
