"""Minimal GPT-2 model for scaling law experiments.

Stripped-down implementation following Karpathy's nanoGPT pattern.
Supports both absolute and rotary positional encodings to match
Cagnetta et al.'s experimental setup.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_rope=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        if not use_rope:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def _apply_rope(self, x, seq_len):
        d = self.head_dim
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        dim = torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
        freqs = pos[:, None] / (10000.0 ** (dim[None, :] / d))
        cos = freqs.cos()
        sin = freqs.sin()
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self._apply_rope(q, T)
            k = self._apply_rope(k, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if self.use_rope:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal, float("-inf"))
        else:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_rope=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, use_rope)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, use_rope=False, tie_weights=False):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            drop=nn.Dropout(0.0),
            h=nn.ModuleList([
                Block(n_embd, n_head, block_size, use_rope) for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        if not use_rope:
            self.transformer["wpe"] = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.use_rope = use_rope

        self.apply(self._init_weights)

        if tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT: {n_params/1e6:.2f}M parameters"
              f"{' (tied weights)' if tie_weights else ''}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_intermediates=False,
                cerebellar_fn=None, cerebellar_input_block=0,
                cerebellar_inject_block=1):
        B, T = idx.size()
        assert T <= self.block_size

        tok_emb = self.transformer.wte(idx)
        if not self.use_rope:
            pos = torch.arange(0, T, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        intermediates = {}
        if return_intermediates:
            intermediates["post_embed"] = x

        cerebellar_injection = None
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if return_intermediates:
                intermediates[f"post_block{i}"] = x
            if cerebellar_fn is not None and i == cerebellar_input_block:
                cerebellar_injection = cerebellar_fn(x)
            if cerebellar_injection is not None and i == cerebellar_inject_block:
                x = x + cerebellar_injection
                cerebellar_injection = None

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_intermediates:
            return logits, loss, intermediates
        return logits, loss

    def get_ngram_losses(self, idx):
        """Compute per-position losses for n-gram analysis.

        Returns tensor of shape (B, T-1) where entry [b, t] is the
        cross-entropy loss for predicting position t+1 given context 0..t.
        This gives L_n(P, M) when averaged over the dataset and decomposed
        by context length n = t+1.
        """
        B, T = idx.size()
        logits, _ = self.forward(idx)
        log_probs = F.log_softmax(logits, dim=-1)
        # loss at position t: -log p(x_{t+1} | x_{0:t})
        targets = idx[:, 1:]
        log_probs_shifted = log_probs[:, :-1, :]
        per_pos_loss = -log_probs_shifted.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return per_pos_loss
