# P1 Direct SMILES Property Programs + Group-RL

## Status

**Preregistered single-seed gate; no new training.**

P1 is the numbered paper direction for the strongest existing non-GraphEdit
signal: compositional property programs with direct SMILES generation and
group-relative RL. It is deliberately separated from the source-editing and
GraphEditDSL lines.

## Existing positive evidence

The frozen v2 report currently records:

| setting | SFT | Group-RL | delta |
| --- | ---: | ---: | ---: |
| 2p-7p, n=128 | 79.1% | **84.5%** | **+5.4pp** |
| 6p, n=128 | 64.8% | **74.4%** | **+9.6pp** |
| 7p, n=128 | 58.4% | **66.1%** | **+7.7pp** |
| OOD, n=128 | 67.3% | **75.6%** | **+8.3pp** |
| OOD conservative, n=256 | — | **89.4%** | — |

Those results are not yet sufficient for a paper claim because the historical
headline is based on property-aware best-of-k selection.

## P1 first gate

The first experiment reuses the existing SFT and Group-RL checkpoints and
generates four paired raw candidate pools:

1. 2p-7p SFT, seed 7, n=256;
2. 2p-7p Group-RL, seed 7, n=256;
3. OOD SFT, seed 7, n=256, conservative decoding;
4. OOD Group-RL, seed 7, n=256, conservative decoding.

It reports `k=1,4,8,20,32,64,128,256`, raw strict-success fraction, empirical
prefix pass@k, estimated pass@k, validity, and unique-valid yield. Curves are
stratified by 2p→7p complexity and OOD bucket. Group-RL minus SFT confidence
intervals use paired condition bootstrap.

The property-reranked selected molecule is diagnostic-only. It is never called
one-shot; one-shot is exactly raw candidate index 0.

## Decision rule

- **Strong single-seed signal**: Group-RL improves raw success and empirical
  pass@k at k=8/20 overall, for both 6p and 7p, and on OOD, with the primary
  overall confidence checks excluding zero.
- **Promising**: all preregistered point gates pass but confidence evidence is
  not yet uniform; proceed to multi-seed confirmation.
- **Sampling-heavy only**: the low-budget gate fails and the apparent advantage
  appears only near k=256.

Implementation and the machine-readable preregistration live in
`SketchMol-Understanding-Condition/experiments/p1_property_program_group_rl/`.
The Nibi finalizer will write the first result to
`SketchMol-Understanding-Condition/outputs/p1_property_program_group_rl_seed7/final/p1_report.md`.
