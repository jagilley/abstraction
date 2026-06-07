"""Vision Transformer for MNIST classification.

Minimal ViT with the same intermediate/cerebellar interface as GPT in model.py.
Bidirectional attention (no causal mask). Patch embedding with [CLS] token.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class ViTBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head)
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


class ViT(nn.Module):
    """Vision Transformer for image classification.

    Same interface as GPT in model.py: supports return_intermediates,
    cerebellar_fn, and perturbation injection.
    """

    def __init__(self, img_size=28, patch_size=4, in_channels=1, n_classes=10,
                 n_layer=4, n_head=4, n_embd=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.n_positions = self.n_patches + 1

        self.patch_proj = nn.Linear(patch_size * patch_size * in_channels, n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Embedding(self.n_positions, n_embd)

        self.blocks = nn.ModuleList([
            ViTBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ViT: {n_params / 1e6:.2f}M parameters "
              f"({n_layer}L {n_head}H {n_embd}D, "
              f"patch={patch_size}, {self.n_patches} patches)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_patches, -1)
        return self.patch_proj(x)

    def forward(self, x, targets=None, return_intermediates=False,
                cerebellar_fn=None, cerebellar_input_block=0,
                cerebellar_inject_block=1, perturbation=None):
        B = x.shape[0]

        patches = self._patchify(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        pos = torch.arange(self.n_positions, device=x.device)
        x = x + self.pos_embed(pos)

        intermediates = {}
        if return_intermediates:
            intermediates["post_embed"] = x

        cerebellar_injection = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_intermediates:
                intermediates[f"post_block{i}"] = x
            if cerebellar_fn is not None and i == cerebellar_input_block:
                cerebellar_injection = cerebellar_fn(x)
            if cerebellar_injection is not None and i == cerebellar_inject_block:
                x = x + cerebellar_injection
                cerebellar_injection = None
            if perturbation is not None and i == perturbation[0]:
                x = x + perturbation[1]

        x = self.ln_f(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        if return_intermediates:
            return logits, loss, intermediates
        return logits, loss
