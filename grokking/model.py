from __future__ import annotations

import torch
import torch.nn as nn


class ModularAddTransformer(nn.Module):
    """
    Tiny Transformer classifier for modular addition.

    Input: token ids for (a,b), shape [B, 2], values in [0, p-1]
    Output: logits over classes [0, p-1], shape [B, p]
    """

    def __init__(
        self,
        *,
        p: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if p <= 1:
            raise ValueError(f"p must be > 1, got {p}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}")

        self.p = p
        self.seq_len = 2

        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(self.seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2] long
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Expected x with shape [B,2], got {tuple(x.shape)}")
        h = self.tok_emb(x) + self.pos_emb.unsqueeze(0)  # [B,2,d]
        h = self.encoder(h)  # [B,2,d]
        h = self.ln(h)
        pooled = h.mean(dim=1)  # [B,d]
        return self.head(pooled)  # [B,p]


