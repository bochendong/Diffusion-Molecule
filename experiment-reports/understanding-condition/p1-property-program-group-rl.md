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

## Interim result after Nibi node failures (2026-08-22)

Three of the four generation jobs ended in Nibi `NODE_FAIL`, after writing only
complete n=256 condition groups. Before resubmission could overwrite those
files, they were frozen under
`outputs/p1_property_program_group_rl_seed7/interim_nodefail_snapshot/` and
evaluated on the paired SFT/Group-RL intersections.

Coverage is 1,408 / 6,000 conditions for 2p-7p and 672 / 1,000 for OOD. The
2p-7p prefix contains only 2p and 3p conditions, so it cannot test the
preregistered 6p/7p claim and must not be treated as the final gate.

| benchmark | budget | SFT raw | Group-RL raw | delta raw | SFT empirical pass@k | Group-RL empirical pass@k | delta pass@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2p-7p paired prefix | 8 | 13.04% | **16.37%** | **+3.33pp** [2.46, 4.22] | 54.40% | **63.99%** | **+9.59pp** [6.68, 12.50] |
| 2p-7p paired prefix | 20 | 12.94% | **16.36%** | **+3.42pp** [2.87, 3.99] | 75.43% | **83.17%** | **+7.74pp** [5.54, 10.01] |
| OOD paired prefix | 8 | 1.43% | **2.12%** | **+0.69pp** [0.20, 1.19] | 10.57% | **15.48%** | **+4.91pp** [1.49, 8.33] |
| OOD paired prefix | 20 | 1.50% | **1.93%** | **+0.43pp** [0.12, 0.74] | 23.07% | **30.95%** | **+7.89pp** [3.57, 12.20] |

The interim verdict is `interim_mixed_or_negative` solely because the missing
6p/7p strata make eight preregistered hard-complexity checks unavailable. All
available overall 2p-7p and OOD point checks at k=8/20 pass, and all three
available primary confidence checks exclude zero. This is evidence against the
"k=256 only" failure mode, but not yet evidence for compositional-complexity
extrapolation.

The one-shot picture is mixed and should not be hidden: Group-RL improves the
2p/3p paired prefix from 12.64% to 14.70%, but decreases OOD one-shot from
2.38% to 1.64%; the OOD advantage appears from k=4 onward. Across the full
n=256 pools, 2p/3p validity rises from 30.02% to 34.93%, whereas OOD validity
falls from 8.90% to 8.34%. Thus the current OOD gain is a low-budget sampling
gain, not a uniformly better first draw or validity result.

The infrastructure reruns are:

- `20300381` — 2p-7p SFT;
- `20300408` — 2p-7p Group-RL;
- `20300462` — OOD Group-RL;
- `20300595` — dependent finalizer.

At the time of this update all four are pending because the requested GPU node
is unavailable. The completed OOD SFT pool from job `20285887` is retained.

## P1/P2 validity and consistency repairs (2026-08-23)

The follow-up jobs completed successfully on Nibi:

- `20331234` --- source-consistency/validity pilot;
- `20333563` --- paired 2p--7p syntax-safe evaluation;
- `20333564` --- paired OOD syntax-safe evaluation;
- `20333565` --- P2 finalizer.

Syntax-safe decoding improves raw validity from 34.9% to 48.2% at `k=1` on
the paired 2p--7p subset and from 12.0% to 29.0% on OOD. Pass@8 improves from
43.8% to 51.6% and from 26.7% to 42.7%, respectively. It remains below the
preregistered +20-point validity promotion gate. The aggregate OOD result also
hides a high-order caveat: OOD 7p Pass@20 decreases from 18.2% to 4.5%.

For executable editing, the protected, source-consistent, and strongly
consistent policies all have 100% raw validity. Their raw first-candidate
strict accuracies are 26.5%, 26.0%, and 23.0%, so consistency ranking does not
pass the raw-edit gate. Under explicitly assisted source-only finalization at
`k=8`, they reach 62.0%, 66.0%, and 64.0%. The 66.0% figure is therefore an
assisted-selection result, not one-shot editing.

The preregistered decision is `stop` for both promotion gates. The de novo
decoder remains a positive ablation worth reporting; the consistency variant
does not warrant further training in its current form.
