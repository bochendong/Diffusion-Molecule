"""Optional PyTorch model for the pure-SMILES dual-stream experiment."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in dependency-light envs.
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SmilesDualStreamModel(nn.Module):
        """Small seq2seq + contrastive alignment model.

        The edit stream reconstructs target SMILES from source/corrupted SMILES.
        The alignment stream uses an in-batch InfoNCE loss between encoded input
        and target strings. Fragment-level pools can be added later without
        changing the manifest format.
        """

        def __init__(
            self,
            vocab_size: int,
            *,
            embed_dim: int = 128,
            hidden_dim: int = 256,
            pad_id: int = 0,
            temperature: float = 0.07,
            fragment_chunk_size: int = 8,
        ) -> None:
            super().__init__()
            self.pad_id = int(pad_id)
            self.fragment_chunk_size = int(fragment_chunk_size)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
            self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.target_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.decoder = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim * 2, batch_first=True)
            self.output = nn.Linear(hidden_dim * 2, vocab_size)
            self.projection = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))

        def encode(self, ids: torch.Tensor, encoder: nn.GRU | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            encoder = encoder or self.encoder
            mask = ids.ne(self.pad_id)
            embedded = self.embedding(ids)
            hidden, _ = encoder(embedded)
            pooled = masked_mean(hidden, mask)
            return hidden, pooled

        def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            target_ids: torch.Tensor,
            *,
            reconstruction_loss_weight: float = 1.0,
            alignment_loss_weight: float = 1.0,
            molecule_alignment_weight: float = 1.0,
            token_alignment_weight: float = 0.1,
            fragment_alignment_weight: float = 0.2,
        ) -> dict[str, torch.Tensor]:
            input_hidden, input_pooled = self.encode(input_ids, self.encoder)
            target_hidden, target_pooled = self.encode(target_ids, self.target_encoder)
            input_mask = input_ids.ne(self.pad_id)
            target_mask = target_ids.ne(self.pad_id)

            decoder_emb = self.embedding(decoder_input_ids)
            context = input_pooled.unsqueeze(1).expand(-1, decoder_emb.shape[1], -1)
            decoded, _ = self.decoder(torch.cat([decoder_emb, context], dim=-1))
            logits = self.output(decoded)

            reconstruction_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=self.pad_id,
            )
            # Contrastive alignment losses overflow easily in fp16; keep them in fp32.
            alignment_dtype = torch.float32 if input_pooled.dtype != torch.float32 else input_pooled.dtype
            projected_input = self.projection(input_pooled.to(alignment_dtype))
            projected_target = self.projection(target_pooled.to(alignment_dtype))
            projected_input_hidden = self.projection(input_hidden.to(alignment_dtype))
            projected_target_hidden = self.projection(target_hidden.to(alignment_dtype))
            molecule_alignment_loss = symmetric_infonce(
                F.normalize(projected_input, dim=-1),
                F.normalize(projected_target, dim=-1),
                self.temperature.clamp_min(1e-3).to(alignment_dtype),
            )
            token_alignment_loss = local_set_alignment_loss(
                F.normalize(projected_input_hidden, dim=-1),
                F.normalize(projected_target_hidden, dim=-1),
                input_mask,
                target_mask,
            )
            input_fragments, input_fragment_mask = chunked_pool(input_hidden, input_mask, self.fragment_chunk_size)
            target_fragments, target_fragment_mask = chunked_pool(target_hidden, target_mask, self.fragment_chunk_size)
            fragment_alignment_loss = local_set_alignment_loss(
                F.normalize(self.projection(input_fragments.to(alignment_dtype)), dim=-1),
                F.normalize(self.projection(target_fragments.to(alignment_dtype)), dim=-1),
                input_fragment_mask,
                target_fragment_mask,
            )
            alignment_loss = (
                float(molecule_alignment_weight) * molecule_alignment_loss
                + float(token_alignment_weight) * token_alignment_loss
                + float(fragment_alignment_weight) * fragment_alignment_loss
            )
            loss = float(reconstruction_loss_weight) * reconstruction_loss + float(alignment_loss_weight) * alignment_loss
            return {
                "loss": loss,
                "reconstruction_loss": reconstruction_loss.detach(),
                "alignment_loss": alignment_loss.detach(),
                "molecule_alignment_loss": molecule_alignment_loss.detach(),
                "token_alignment_loss": token_alignment_loss.detach(),
                "fragment_alignment_loss": fragment_alignment_loss.detach(),
                "logits": logits,
            }


    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denom


    def symmetric_infonce(left: torch.Tensor, right: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        logits = left @ right.T / temperature
        labels = torch.arange(left.shape[0], device=left.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


    def local_set_alignment_loss(
        left: torch.Tensor,
        right: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for index in range(left.shape[0]):
            left_items = left[index][left_mask[index]]
            right_items = right[index][right_mask[index]]
            if left_items.numel() == 0 or right_items.numel() == 0:
                continue
            similarity = left_items @ right_items.T
            left_to_right = 1.0 - similarity.max(dim=1).values.mean()
            right_to_left = 1.0 - similarity.max(dim=0).values.mean()
            losses.append(0.5 * (left_to_right + right_to_left))
        if not losses:
            return left.new_tensor(0.0)
        return torch.stack(losses).mean()


    def chunked_pool(hidden: torch.Tensor, mask: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = max(1, int(chunk_size))
        pooled_chunks = []
        masks = []
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(hidden.shape[1], start + chunk_size)
            chunk = hidden[:, start:end]
            chunk_mask = mask[:, start:end]
            weights = chunk_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled_chunks.append((chunk * weights).sum(dim=1) / denom)
            masks.append(chunk_mask.any(dim=1))
        if not pooled_chunks:
            return hidden.new_zeros((hidden.shape[0], 0, hidden.shape[-1])), mask.new_zeros((mask.shape[0], 0))
        return torch.stack(pooled_chunks, dim=1), torch.stack(masks, dim=1)

else:

    class SmilesDualStreamModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for SmilesDualStreamModel. Install torch to train the model.")
