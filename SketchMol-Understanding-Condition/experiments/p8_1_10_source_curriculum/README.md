# P8.1.10: source-aligned curriculum

This is the full-SMILES unified-policy branch. At inference, one checkpoint, one
decoder, and one softmax emit full-molecule SMILES for both null-source de novo
generation and source-conditioned editing. There is no router, alternate output
family, interpreter, materializer, pointer/copy mechanism, or property reranking.

## Base lineage

P8.1.7 did not train a new checkpoint. It established the condition layout that
keeps the null-source path exactly direct-compatible while placing the task token
and source tokens only on edit inputs. P8.1.10 therefore starts from the
P8.1.4 source-aware checkpoint, whose shared tensors are the unchanged P1
Group-RL policy, and runs it through the P8.1.7 entrypoint. Only source-conditioned
parameters are trainable, so the source-free de novo path is protected by
construction and checked candidate-for-candidate after training.

## Mandatory rounds

- R1: clean source reconstruction, followed by property-edit SFT.
- R2: local span-corrupted source reconstruction, followed by the identical
  property-edit SFT.

The only scientific difference is the R2 stage-1 corruption: one contiguous span
covering approximately 12% (one to four tokens) is deleted from the source input.
Both rounds independently start from the same audited base; R2 does not resume R1.

All curricula are built from the MolEdit Table1 training partition. Canonical
source/target leakage against the P6 Table1 evaluation targets, exact pair overlap,
and ID overlap are fatal. Evaluation uses the fixed P6 hard de novo and Table1
gates, raw generation order, exactly 20 candidates, k=1/8/20, candidate-level
validity/success/identity, and no property-based selection.

```bash
bash experiments/p8_1_10_source_curriculum/submit_queue.sh
```
