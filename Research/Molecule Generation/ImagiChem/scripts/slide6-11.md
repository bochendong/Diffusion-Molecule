This page shows the actual translation step: how pixel statistics become a chemical alphabet. 


Each image is flattend and treated as a one-dimensional signal, and then we can compute statistics such as mean, std, contrast, and peak.


将 `mean` 从区间 $[0, 255]$ 线性映射到碳数 $[15, 40]$，再 `clip` 到该范围



Here the mapping starts to become chemical. Mean intensity influences the carbon count, while standard deviation influences the nitrogen count. This is important because both quantities are easy to compute and easy to audit. If a learner inspects the pipeline in Python, they can trace exactly why a brighter or more variable row produces a different atomic count.

Next come the atom rules. Contrast controls the approximate molecular size, and peak frequency contributes to sulfur content. At the same time, halogens are kept under tight limits. That constraint matters because the system is not only generating symbols; it is trying to remain chemically plausible and reasonably drug-like, rather than letting extreme image features create unrealistic compositions.

Once those rules produce atomic counts, the model builds an atomic count vector and then shuffles the resulting atomic sequence into an output form such as a deterministic extended formula string. The shuffle here is still rule-governed, not arbitrary in the stochastic sense. So the final representation preserves reproducibility while giving us a usable symbolic seed for later assembly stages.
So the takeaway from this page is simple: row-level image statistics are converted into atomic counts through explicit, interpretable rules, and those counts become a deterministic chemical string. That gives ImagiChem controllability, debuggability, and a transparent foundation for the next step, where we will look more closely at interactive mapping from visual features to atomic composition.
