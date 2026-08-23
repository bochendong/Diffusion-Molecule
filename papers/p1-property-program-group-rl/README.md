# P1 paper workspace

Working title: **Compositional Property Programs for Direct Molecular
Generation with Group-Relative Reinforcement Learning**.

This is the **ICLR 2027** paper workspace. It deliberately remains independent
of GraphEditDSL and source-conditioned molecular editing. The official ICLR
2027 style files are vendored unchanged under `iclr2027/`.

## Current scope

- single generation seed (`7`); no multi-seed experiment is planned;
- direct autoregressive SMILES generation;
- frozen Qwen2.5-VL-7B-Instruct text-condition features;
- deterministic numerical property-program tokens;
- a 4-layer, 256-dimensional Transformer SMILES decoder;
- SFT with property-count curriculum followed by one epoch of group-relative
  policy optimization;
- raw candidate evaluation at `k=1,4,8,20,32,64,128,256`;
- no molecular library, retrieval materializer, or inference-time property
  reranking in the reported P1 metrics.

## Current drafting stage

Only the abstract and Introduction are being drafted. Method, experiments,
results, limitations, and appendices will be written after this opening is
reviewed and the final P1 tables are mirrored from Nibi.

## Files

- `manuscript.md`: abstract and Introduction working draft;
- `claims-and-evidence.md`: claim ledger and evidence gates;
- `references.bib`: verified starting bibliography.
- `iclr2027/main.tex`: anonymous ICLR abstract/Introduction scaffold;
- `iclr2027/submission-checklist.md`: format, deadline, and scientific gates.

## Result status

The fast 6p/7p `n=20` gate and the full OOD `n=256` pool are available on
Nibi. The full 2p-to-7p `n=256` table remains a result placeholder until the
paired finalizer completes. Numbers marked **preliminary** must be replaced
from the machine-readable P1 final outputs before submission.

Paper-facing generated artifacts are expected at:

```text
SketchMol-Understanding-Condition/outputs/p1_property_program_group_rl_seed7/final/
  p1_report.md
  p1_paper_main_table.csv
  p1_validity_audit.csv
  p1_scaling_summary.csv
  p1_paired_deltas.csv
```
