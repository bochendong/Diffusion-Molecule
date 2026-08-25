# P19 frozen expanded unified benchmark

P19 is a test-only, paired expansion of the frozen P17 and P18 unified adapters.
It performs no training and makes no parameter or threshold changes. The 100-row
MolEdit Table1 estimate and 40-row hard de-novo estimate are selected and hashed
before model generation. They remain pilot estimates rather than full benchmarks.

The two adapters receive byte-identical prompt files and matching raw generation
seeds. Candidate budgets 1/4/8 are prefixes of one greedy plus seven sampled
candidates; no target access, static candidate pool, or property reranking is used.
