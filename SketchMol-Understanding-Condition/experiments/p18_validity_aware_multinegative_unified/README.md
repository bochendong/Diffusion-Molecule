# P18: validity-aware multi-negative unified continuation

P18 starts from the frozen P16 mixed LoRA, not P17. It keeps the same single
Qwen2.5-VL-7B base, LoRA, tokenizer, prompt, and response schema for direct de-novo
generation and source-conditioned editing. There is no router or mode-specific head.

The frozen objective is chosen-completion CE plus three small train-only margin terms:

1. a much weaker source-copy negative for edit rows;
2. a hard invalid-SMILES corruption for both modes;
3. a valid same-mode target paired with the wrong condition for both modes.

This is multi-negative contrastive SFT with ORPO-like margins, not a true ORPO
implementation. Each negative is forwarded separately to keep 40 GB MIG memory use
bounded. The exact P17 160+160 train rows, ID/OOD development rows, and 20+20 frozen
pilot subsets are reused so P16/P17/P18 comparisons are paired.

The preregistration was frozen before P18 training or evaluation. A failed gate remains
a negative result; the frozen pilot still runs once because its purpose is diagnosis.
