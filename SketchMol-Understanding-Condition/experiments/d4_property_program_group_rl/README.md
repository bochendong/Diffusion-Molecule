# D4: Direct SMILES Property Programs + Group-RL

D4 is the numbered, GraphEditDSL-independent paper direction for testing
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
`d4_property_program_group_rl_preregistration.json`.

## Nibi run

After pulling the synchronized commit on Nibi:

```bash
bash SketchMol-Understanding-Condition/experiments/d4_property_program_group_rl/submit_d4_sampling_scaling.sh
```

This submits four independent GPU jobs and one `afterok` CPU finalizer. The
default output root is:

```text
SketchMol-Understanding-Condition/outputs/d4_property_program_group_rl_seed7/
```

Primary artifacts:

- `final/d4_report.md`
- `final/d4_gate.json`
- `final/d4_scaling_summary.csv`
- `final/d4_paired_deltas.csv`
- `final/d4_condition_metrics.csv`

The generator also writes `diagnostic_property_reranked_selected.csv` for
backward compatibility. D4 excludes that file from every reported metric.
