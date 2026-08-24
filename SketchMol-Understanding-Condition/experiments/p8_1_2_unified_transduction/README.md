# P8.1.2 — Empty/Source Unified SELFIES Transduction

This branch takes a strict position: a unified molecular policy must use one
short output language for both de-novo design and editing. It cannot call a
SMILES generator when the source is empty and a graph-edit DSL when it is not.

The sole program is a run-length encoded SELFIES transducer:

```
<TRANSDUCE> (<KEEP_n> | <DELETE_n> | <INSERT> SELFIES* <INSERT_END>)* <STOP>
```

De novo is exactly this interpreter with an empty source sequence. Editing is
the same interpreter with the source molecule encoded as SELFIES. There is one
decoder vocabulary, one interpreter and no mode router or property reranker.

## Fast falsification

R1 uses canonical target serialization. R2 changes only one factor: target
atoms are serialized in source-MCS order before SELFIES encoding. The oracle
reports coverage, exact reconstruction, program-length budget and copy rate by
mode. R1 does not eliminate the branch; its primary failure determines whether
R2 alignment actually shortens the editing program.

If representation coverage and exact reconstruction are near one and both
modes fit the 188-token budget, the next job trains one P6-sized decoder on the
mixed R2 rows, then samples exactly 20 candidates in generation order and
reports raw `k=1,8,20`. No property-ranked selected-1 is permitted.

Run on Nibi:

```bash
bash SketchMol-Understanding-Condition/experiments/p8_1_2_unified_transduction/submit_p8_1_2_oracle.sh
```
