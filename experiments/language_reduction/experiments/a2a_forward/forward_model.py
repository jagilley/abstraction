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


class TransformerForwardModel(nn.Module):
    def __init__(self, d_model: int, d_head: int = 64, n_head: int = 1,
                 n_layer: int = 1, mlp_mult: int = 2, block_size: int = 128):
        super().__init__()
        self.d_model = d_model

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

        self.blocks = nn.ModuleList([
            ForwardBlock(d_model, d_head, n_head, mlp_mult)
            for _ in range(n_layer)
        ])

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TransformerForwardModel: {n_params/1e3:.1f}K parameters "
              f"(d_model={d_model}, d_head={d_head}, n_head={n_head}, "
              f"n_layer={n_layer}, mlp_hidden={d_model * mlp_mult})")

    def forward(self, x):
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return x
