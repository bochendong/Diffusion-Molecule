import torch

from sketchmol_understanding_condition.encoders import (
    HybridConditionEncoder,
    MolecularQueryProjector,
)


def test_molecular_query_projector_shape():
    projector = MolecularQueryProjector(
        mllm_hidden_dim=16,
        context_dim=8,
        num_queries=4,
        hidden_dim=32,
    )
    hidden = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    output = projector(hidden, mask)

    assert output.tokens.shape == (2, 4, 8)
    assert output.attention_mask.shape == (2, 4)
    assert output.attention_mask.dtype == torch.bool


def test_hybrid_condition_encoder_concatenates_property_tokens():
    projector = MolecularQueryProjector(
        mllm_hidden_dim=16,
        context_dim=8,
        num_queries=4,
        hidden_dim=32,
    )
    encoder = HybridConditionEncoder(projector)
    hidden = torch.randn(2, 5, 16)
    property_tokens = torch.randn(2, 3, 8)

    output = encoder(hidden, property_tokens=property_tokens)

    assert output.tokens.shape == (2, 7, 8)
    assert output.attention_mask.shape == (2, 7)
