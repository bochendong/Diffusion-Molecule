import pytest


torch = pytest.importorskip("torch")

from sketchmol_understanding_condition.fusion_image_text_encoder import _supervised_contrastive_loss


def test_supervised_contrastive_loss_rewards_same_label_neighbors():
    close_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        requires_grad=True,
    )
    far_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.95, 0.05],
            [0.0, -1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 2])

    close_loss = _supervised_contrastive_loss(close_embeddings, labels, temperature=0.2)
    far_loss = _supervised_contrastive_loss(far_embeddings, labels, temperature=0.2)

    assert close_loss.item() < far_loss.item()


def test_supervised_contrastive_loss_handles_batches_without_positives():
    embeddings = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])

    loss = _supervised_contrastive_loss(embeddings, labels, temperature=0.2)

    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad
