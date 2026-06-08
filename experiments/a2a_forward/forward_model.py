"""Forward models (cerebellum analog) for predicting transformer activations.

ForwardModel: per-position MLP. Structurally blind to cross-position effects.
TransformerForwardModel: small transformer. Capacity-bottlenecked, not
position-blind — the residual captures computational novelty, not attention's
existence. Architecture mirrors the cerebellar circuit:
  - Causal attention with compressed projections (pontine relay)
  - MLP expansion (granule cell combinatorial coding)
  - Linear readout (Purkinje cell)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"ForwardModel: {n_params/1e3:.1f}K parameters "
              f"(d_model={d_model}, hidden={hidden})")

    def forward(self, x):
        return self.net(x)


class ForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, mlp_mult: int):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_head * n_head)
        self.k_proj = nn.Linear(d_model, d_head * n_head)
        self.v_proj = nn.Linear(d_model, d_head * n_head)
        self.out_proj = nn.Linear(d_head * n_head, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        mlp_hidden = d_model * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x, causal_mask):
        B, T, C = x.size()

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_head * self.n_head)
        x = x + self.out_proj(y)

        x = x + self.mlp(self.ln2(x))
        return x


class CerebellarGate(nn.Module):
    """Learned gated projection for injecting forward model predictions.

    Zero-initialized so the injection starts at exactly zero.
    The projection learns to transform the forward model's prediction
    into a useful signal for the main model's residual stream (thalamic
    relay analog). Injection magnitude grows from zero as training
    discovers useful structure.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        print(f"CerebellarGate: {sum(p.numel() for p in self.parameters())/1e3:.1f}K "
              f"parameters (d_model={d_model})")

    def forward(self, fwd_pred):
        return self.projection(fwd_pred)

    def injection_norm(self):
        return self.projection.weight.norm().item()


class LossPredictor(nn.Module):
    """Predicts per-token loss from early-layer activations (emotion analog).

    Returns both a low-dimensional embedding (for injection via EmotionGate)
    and a scalar loss prediction (for training via MSE against actual CE).
    When embed_dim=1, the embedding IS the scalar prediction.
    When embed_dim>1, a linear head maps the embedding to a scalar for training,
    but the full embedding is what gets injected.
    """

    def __init__(self, d_model: int, embed_dim: int = 1, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.head = nn.Linear(embed_dim, 1) if embed_dim > 1 else None
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LossPredictor: {n_params/1e3:.1f}K parameters "
              f"(d_model={d_model}, embed_dim={embed_dim}, hidden={hidden})")

    def forward(self, x):
        embed = self.encoder(x)  # (B, T, embed_dim)
        if self.head is not None:
            loss_pred = self.head(embed).squeeze(-1)  # (B, T)
        else:
            loss_pred = embed.squeeze(-1)  # (B, T)
        return embed, loss_pred


class EmotionGate(nn.Module):
    """Learned projection for injecting evaluative signals into the residual stream.

    Zero-initialized like CerebellarGate. Projects from a low-dimensional
    evaluative embedding to full residual stream dimensionality.
    """

    def __init__(self, embed_dim: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(embed_dim, d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        print(f"EmotionGate: {sum(p.numel() for p in self.parameters())/1e3:.1f}K "
              f"parameters (embed_dim={embed_dim}, d_model={d_model})")

    def forward(self, embed):
        return self.projection(embed)

    def injection_norm(self):
        return self.projection.weight.norm().item()


class TransformerForwardModel(nn.Module):
    def __init__(self, d_model: int, d_head: int = 64, n_head: int = 1,
                 n_layer: int = 1, mlp_mult: int = 2, block_size: int = 128,
                 causal: bool = True):
        super().__init__()
        self.d_model = d_model

        if causal:
            mask = torch.tril(torch.ones(block_size, block_size))
        else:
            mask = torch.ones(block_size, block_size)
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

        self.blocks = nn.ModuleList([
            ForwardBlock(d_model, d_head, n_head, mlp_mult)
            for _ in range(n_layer)
        ])

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TransformerForwardModel: {n_params/1e3:.1f}K parameters "
              f"(d_model={d_model}, d_head={d_head}, n_head={n_head}, "
              f"n_layer={n_layer}, mlp_hidden={d_model * mlp_mult}"
              f"{', bidirectional' if not causal else ''})")

    def forward(self, x):
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return x
