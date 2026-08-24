# P8: Validity-Preserving MolProgram

P8 is the preregistered follow-up to P7's de-novo failure. It keeps the frozen
P6 checkpoint and P7 typed grammar, then adds chemical state invariants:

- `<STOP>` is available only for a sanitizable molecular graph;
- duplicate ring closures are masked;
- locally impossible bond orders are masked before sampling.

The fast single-seed gate evaluates the exact same 64 hard 6p/7p conditions at
raw `k=1,8,20`, with no property-aware selection or reranking.
