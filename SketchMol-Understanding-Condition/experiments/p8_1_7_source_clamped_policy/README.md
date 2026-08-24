# P8.1.7 — Source-Clamped Full-SMILES Policy

P8.1.7 treats source preservation as an internal state constraint rather than
a router or output postprocessor.  It reuses the trained P8.1.4 checkpoint: one
autoregressive decoder, output head, and full-molecule SMILES vocabulary serve
both empty-source generation and source-present editing.  The checkpoint's
learned source encoder produces a residual logit state multiplied by
`source_present`; the source-free branch is therefore an exact identity.

R1 evaluates the checkpoint-native residual scale `1.0`.  R2 always runs and
changes only that scale to `2.0`.  No weights, prompts, decoding parameters,
candidate order, or benchmark rows change.  R2 is queued with `afterany` rather
than an accuracy gate.  The null-source de-novo candidates must be semantically
bit-identical across rounds or the causal comparison fails closed.

This is stricter than a two-family router: the model emits complete SMILES in
both modes through the same softmax.  It is also distinct from a materializer
or property reranker because the clamp acts on decoder logits before sampling
and never observes generated properties.  Raw `k=1,8,20`, candidate validity,
uniqueness, identity copy, strict non-identity success, and source similarity
are reported on the same P6 hard de-novo and MolEdit Table1 subsets.
