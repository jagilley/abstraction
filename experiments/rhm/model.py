"""Minimal GPT-2 model for scaling law experiments.

Copied from language_reduction/model.py with minor simplifications.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, causal=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if self.causal:  # bidirectional encoder when causal=False (e.g. masked-span)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, causal=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, causal=causal)
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
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, causal=True):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(0.0),
            h=nn.ModuleList([
                Block(n_embd, n_head, block_size, causal=causal) for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_intermediates=False,
                cerebellar_fn=None, cerebellar_input_block=0,
                cerebellar_inject_block=1,
                cerebellar_mode="add", cerebellar_readd_block=None):
        """cerebellar_mode:
          "add"    -- summation / side-channel: x = x + inj at the inject block.
                      The historical behaviour; every prior result used this path
                      and it is bit-identical to before when the new args default.
          "cancel" -- efference copy / Rao-Ballard: x = x - inj at the inject block,
                      the RESIDUAL propagates, and inj is restored at
                      `cerebellar_readd_block` (defaults to the inject block, i.e. a
                      no-op; callers should set it to the FM's target block).
                      See ideas/efference_copy_cancellation.md.

        Under "cancel", `intermediates[f"post_block{k}"]` at the re-add block is the
        DEVIATION stream (recorded pre-re-add, consistent with every other block);
        the restored read-out is exposed separately as `post_block{k}_eff`. The FM's
        training target must be the `_eff` tensor, or the loop is degenerate (the FM
        would chase the deviation its own forecast creates, driving inj -> 0).
        """
        B, T = idx.size()
        assert T <= self.block_size
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device))
        x = self.transformer.drop(tok_emb + pos_emb)

        intermediates = {}
        if return_intermediates:
            intermediates["post_embed"] = x

        readd_block = (cerebellar_inject_block if cerebellar_readd_block is None
                       else cerebellar_readd_block)
        if cerebellar_mode == "cancel" and cerebellar_fn is not None:
            # Otherwise the forecast is subtracted and silently never restored.
            assert cerebellar_inject_block <= readd_block < len(self.transformer.h), (
                f"cancel needs inject_block ({cerebellar_inject_block}) <= readd_block "
                f"({readd_block}) < n_layer ({len(self.transformer.h)})")
        cerebellar_injection = None
        held_injection = None
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if return_intermediates:
                intermediates[f"post_block{i}"] = x
            if cerebellar_fn is not None and i == cerebellar_input_block:
                cerebellar_injection = cerebellar_fn(x)
            if cerebellar_injection is not None and i == cerebellar_inject_block:
                if cerebellar_mode == "cancel":
                    x = x - cerebellar_injection
                    held_injection = cerebellar_injection
                else:
                    x = x + cerebellar_injection
                cerebellar_injection = None
            if held_injection is not None and i == readd_block:
                x = x + held_injection
                held_injection = None
                if return_intermediates:
                    intermediates[f"post_block{i}_eff"] = x

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_intermediates:
            return logits, loss, intermediates
        return logits, loss
