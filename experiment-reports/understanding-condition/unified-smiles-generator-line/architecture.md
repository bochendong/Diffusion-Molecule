# Unified Generator Architecture

This is the proposed architecture for the new experiment line. The inference diagram should stop at final SMILES; official evaluation is downstream and should not be drawn as part of the model.

## Shared condition encoder

```text
Inputs
├─ instruction text / property program
├─ optional source_smiles
└─ optional source_image
```

```text
Frozen HF/VLM condition encoder
├─ if source_image exists:
│  └─ encode source image + instruction
├─ else if source_smiles exists:
│  └─ render source_smiles to a 2D molecule image, then encode image + instruction
└─ else:
   └─ encode instruction/property text only

Outputs
├─ query_tokens: [B, 32, 256]
└─ pooled feature: [B, 3584] in current HF/VLM runs
```

Input-modality ablation keeps the generator fixed and changes only the exported
frozen condition-feature variant:

```text
with_image = variant full
  VLM sees instruction plus source image/render when available

no_image = variant text_only
  VLM sees instruction text only
```

For edit rows, both variants still pass explicit source-SMILES tokens below.
Thus no-image removes visual VLM evidence, not the source molecule itself.

## Explicit condition tokens

```text
Mode token
├─ <DE_NOVO>: source-free property-conditioned generation
└─ <EDIT>: source-conditioned molecular editing

Shape
└─ [B, 1, 256]
```

```text
Property-program tokens
├─ selected properties
├─ target values / tolerances when available
├─ increase / decrease / maintain direction
└─ task property count

Shape
└─ [B, T_prop, 256]
```

```text
Source tokens
├─ only used for <EDIT>
├─ tokenized from source_smiles
├─ max source tokens in current code path: 96
└─ omitted or replaced by a zero/empty source token for <DE_NOVO>

Shape
└─ [B, T_source, 256]
```

## Unified condition memory

```text
De novo memory
└─ concat([query_tokens, mode=<DE_NOVO>, property_tokens])
   shape: [B, 32 + 1 + T_prop, 256]
```

```text
Edit memory
└─ concat([query_tokens, mode=<EDIT>, source_tokens, property_tokens])
   shape: [B, 32 + 1 + T_source + T_prop, 256]
```

The shared VLM tokens preserve the image/text conditioning path; the explicit tokens preserve benchmark-specific properties and the source molecule in a format the SMILES decoder can directly consume.

## Unified SMILES generator

```text
ConditionedSmilesDecoder
├─ type: autoregressive Transformer decoder
├─ condition projection: condition_dim -> d_model
├─ token embedding: SMILES token id -> d_model
├─ d_model: 256
├─ decoder layers: 4
├─ attention heads: 8
├─ FFN hidden size: 1024
├─ max SMILES length: 160
└─ output logits: [B, L, vocab_size]
```

## Inference output

```text
Unified generator
        |
autoregressive decoding
        |
candidate SMILES set
        |
SMILES finalizer
        |
final SMILES
```

The finalizer should be benchmark-aware but model-agnostic:

- de novo: validity + property-constraint ranking.
- edit: validity + source-preservation + property-constraint ranking.
- Table1 assisted rows: any task-success reranking must be declared as assisted/selection, not fair generation.
- MuMO/C-MuMO: candidate CSV must retain top-k candidates for official candidate-level aggregation.

## What this is not

This line is not a claim that current GraphEditDSL is already replaced. It is a new experiment line to test whether one generator can learn both:

- source-free property-conditioned generation.
- source-conditioned instruction-based editing.

GraphEditDSL remains the current strongest assisted edit baseline, especially for MuMO.
