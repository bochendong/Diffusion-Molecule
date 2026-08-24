# P8.1.3: source-relative macro graph transactions

P6--P8 spend up to 188 decoder tokens assembling one molecule atom by atom.
P8.1.3 tests the contrary claim: both de novo design and editing are one
source-relative region commit.  Empty generation exposes a virtual root;
editing exposes a matched source core.  The same transaction and interpreter
commit a molecular payload in either case.

This first stage is deliberately CPU-only and target-aware: it is an oracle
representation/reachability audit, not a generation score.  It must establish
coverage, exact reconstruction, valid reachability, program compression, and
train-only held-out motif vocabulary support before any checkpoint is trained.

R1 encodes the changed payload as one region.  R2 is mandatory and changes one
factor only: BRICS factorization of that payload.  It directly tests whether
R1's primary expected failure is region-token OOV rather than the common graph
transaction itself.  Both rounds use seed 7 and the exact P6 prepared rows.

```bash
bash SketchMol-Understanding-Condition/experiments/p8_1_3_macro_graph/submit_p8_1_3.sh
```

Training is authorized only if both modes reach at least 85% representation
coverage, 99% exact reconstruction and validity among covered rows, and 40%
held-out reachability from the fit-only motif vocabulary.
