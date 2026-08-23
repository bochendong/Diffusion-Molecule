# P4: Event-to-SMILES Distillation

P4 transfers the frozen D3 event policy into one source-aware MolProgram
SMILES policy. D3 is used only to create strict train-only editing targets.
At inference the student directly emits SMILES; there is no router, D3 model,
materializer, or property-aware candidate selection.

The student trains only `source_*` modules. The entire source-free de-novo
path remains tensor-for-tensor identical to the frozen base checkpoint. D3
targets are serialized with a source-aligned atom order when that increases
token overlap without changing the canonical molecule.

The single-seed pilot reports honest raw@1 and any@8/20 Table1 accuracy. The
primary gate is raw `Acc_all(0.65) >= 35%`; `45%` matches the reported MolEditRL
aggregate. Any@20 must reach 70%, validity must stay at least 95%, and the
checkpoint audit must prove the de-novo path is unchanged.

Run on Nibi:

```bash
bash SketchMol-Understanding-Condition/experiments/p4_event_to_smiles_distillation/submit_p4_event_to_smiles_distillation.sh
```
