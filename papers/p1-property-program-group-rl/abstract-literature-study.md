# Abstract study: ten neighboring papers

This note paraphrases the rhetorical structure of ten closely related papers.
It does not reproduce their abstracts verbatim.

| Paper | Problem and gap used in the abstract | Method and evidence emphasized | Lesson for P1 |
|---|---|---|---|
| Graph DiT (NeurIPS 2024) | Diffusion is promising, but explicit multi-conditional molecular generation is underexplored. | Introduces a condition encoder, graph Transformer denoiser, and graph-dependent noise; closes with broad metric superiority and a domain-expert case study. | State the multi-condition gap before architectural detail; connect the method to one clear practical benefit. |
| SketchMol (Nature Machine Intelligence 2025) | Sequence and graph representations emphasize local structure and may miss global topology. | Introduces image-based diffusion plus molecular-expert RL and stresses task unification across de novo design and editing. | Lead with the representation-level idea and use RL as the refinement mechanism, not as an isolated contribution. |
| STGG+ (2024) | Unconditional validity is insufficient because real design requires arbitrary subsets of desired properties. | Combines random property masking, classifier-free guidance, an auxiliary predictor, and self-criticism; claims ID, OOD, and reward-maximization gains. | Variable property subsets are a first-class problem; describe the mechanisms compactly and organize results by generalization regime. |
| GeLLM3O (ACL 2025) | Existing optimization methods largely stop at one or two properties and generalize poorly to unseen tasks. | Introduces an instruction dataset and instruction-tuned LLM family; highlights five ID and five OOD tasks and zero-shot generalization. | Our strongest framing neighbor: high-order composition and OOD generalization should appear in the opening and result sentences. |
| GeLLM4O-C (EMNLP Findings 2025) | Real optimization must selectively improve some properties while maintaining others, which current instruction methods do not capture. | Introduces property-specific objectives, a dataset, and instruction-tuned models; reports a large relative success-rate gain and zero-shot behavior. | Define what a property program can express and avoid a generic claim of multi-property control. |
| C-MORAL (2026) | LLM molecular optimization struggles with selective and competing constraints. | Uses group-relative RL, property-score alignment, and nonlinear reward aggregation; reports exact ID/OOD success rates and scaffold preservation. | This is the closest RL neighbor. P1 must distinguish zero-source generation, variable-length absolute targets, and budget scaling rather than merely saying Group-RL works. |
| Prompt-MolOpt (Nature Machine Intelligence 2024) | Multi-property optimization is limited by conflicting objectives and scarce multi-property labels. | Uses prompt embeddings to transfer relationships learned from single-property data; reports relative success gains and practical case studies. | Property-count curriculum can be motivated as transfer across compositional complexity, but only if the final ablations support that attribution. |
| MolGen (ICLR 2024) | Molecular language models suffer from invalid outputs, narrow domains, and limited feasible diversity. | Combines large-scale molecular pretraining, prefix tuning, and chemical feedback; emphasizes benchmark breadth rather than one table. | Keep validity as a secondary result and describe feedback as chemically verifiable alignment. |
| MolGPT (JCIM 2022) | SMILES makes Transformer generation natural, but direct conditional control must be demonstrated. | Uses a decoder-only Transformer and reports validity, uniqueness, novelty, multi-property control, scaffold conditioning, and saliency. | Establish direct autoregressive SMILES generation clearly; do not imply that the frozen 7B encoder is the trained molecular generator. |
| PMO / Sample Efficiency Matters (2022) | Molecular optimization results are hard to compare and rarely account for oracle-query budget. | Standardizes 25 algorithms over 23 tasks and shows that many apparent state-of-the-art methods lose under a fixed budget. | This supplies the evaluation logic: candidate budget must be explicit, and a best-of-many selected score must not be presented as one-shot quality. |

## Shared abstract pattern

Across the ten papers, the most common sequence is:

1. establish the important scientific task;
2. identify one concrete failure of existing methods;
3. introduce a named formulation or method immediately;
4. give only the method components needed to explain the claimed improvement;
5. report one main quantitative result, optionally followed by one
   generalization result; and
6. close with the scientific meaning rather than another implementation detail.

Most successful abstracts do not enumerate every metric or caveat. More
specifically, only three of the ten abstracts report an exact comparative
performance number; the others state their empirical conclusion
qualitatively. For P1, the right compression is therefore to describe the
low-budget and OOD findings without exact percentages, retain the one-shot
boundary qualitatively, and move all numerical values to the Introduction and
Results.

## P1 positioning implied by the comparison

P1 should not be presented as another generic conditional SMILES model or as
LLM fine-tuning. Its differentiating question is whether group-relative
post-training changes the probability of satisfying a variable-length property
program at a modest sampling budget. The abstract should therefore lead with
the confounding between compositional complexity and candidate budget, then
present property programs and Group-RL as the response.
