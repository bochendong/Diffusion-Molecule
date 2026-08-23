# P4: Event-to-SMILES Distillation

P4 transfers the frozen D3 event policy into one source-aware MolProgram
SMILES policy. D3 is used only to create strict train-only editing targets.
At inference the student directly emits SMILES; there is no router, D3 model,
materializer, or property-aware candidate selection.

The student trains only `source_*` modules. The entire source-free de-novo
path remains tensor-for-tensor identical to the frozen base checkpoint. D3
targets are serialized with a source-aligned atom order when that increases
token overlap without changing the canonical molecule.

The single-seed pilot reports two deliberately separate views. The matched
MolEdit Table 1 result uses all 20 raw candidates per input and computes
candidate-level validity and accuracy. Raw@1 and any@8/20 are retained only as
sampling-budget diagnostics. The pilot gate remains diagnostic: raw
`Acc_all(0.65) >= 35%`, any@20 at least 70%, validity at least 95%, and a
checkpoint audit proving that the de-novo path is unchanged.

Run on Nibi:

```bash
bash SketchMol-Understanding-Condition/experiments/p4_event_to_smiles_distillation/submit_p4_event_to_smiles_distillation.sh
```
