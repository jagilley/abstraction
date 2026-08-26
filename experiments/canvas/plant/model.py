"""The any-order masked token model over a 16x16 code grid, and the mask sampler.

ONE class serves both roles the node needs: the *plant* (the generator the loop will act
through) and the *grader* (a second reader of the same form, trained on a disjoint split),
exactly as `critic/` used one `rhm.model.GPT` for the reader and the agent's generator. What
differs is the data it sees and how its output is read: the plant is read as a proposal
distribution, the grader as a mean per-token NLL of a proposed completion.

WHY ANY-ORDER (MaskGIT) AND NOT RASTER. The piece is inpainting; the seam is the filled
NEIGHBOURHOOD around a hole, not a prefix. A raster autoregression would make the seam a
prefix and would not let a move fill an arbitrary set of cells.

WHY A STYLE EMBEDDING WITH DROPOUT. The memo's critic conditions on "surround + request";
the style id is the request. Rather than run two models we train ONE with the style token
dropped to `unknown` on `p_style_drop` of examples, so the same weights give both readouts:
conditioned (the loop's regime) and surround-only (how much the neighbourhood alone carries
the style). Cost: one extra embedding row.
"""

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 16
MASK_LADDER = (1, 2, 3, 4, 6, 8, 11, 16)     # rect side; area 1..256 (=whole 16x16 grid)


def amp(dev):
    """bf16 autocast on CUDA. Attention over 256 cells at bs=128 dominates the step, and in
    fp32 an L4 runs it at ~0.25 s/step, which would put the node in the hours. Every logit is
    cast back to fp32 before any log_softmax, so the reported NLLs are fp32 numbers."""
    if str(dev).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class Block(nn.Module):
    def __init__(self, d, heads, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
                                 nn.Dropout(p))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class MaskedGrid(nn.Module):
    """Bidirectional transformer over GRID*GRID code cells. Index K is the MASK token."""

    def __init__(self, K, n_styles, d=256, layers=6, heads=8, grid=GRID, p=0.0):
        super().__init__()
        self.K, self.grid, self.n_styles = K, grid, n_styles
        self.tok = nn.Embedding(K + 1, d)
        self.pos = nn.Embedding(grid * grid, d)
        self.sty = nn.Embedding(n_styles + 1, d)       # index n_styles == "unknown"
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, K)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, codes, style):
        """codes: (B, T) long with MASK==K at hidden cells. style: (B,) long."""
        B, T = codes.shape
        pos = torch.arange(T, device=codes.device)
        x = self.tok(codes) + self.pos(pos)[None] + self.sty(style)[:, None, :]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #

def rect_mask(rng, side, grid=GRID):
    """A contiguous square hole of the given side, uniformly placed. Returns (T,) bool."""
    side = min(side, grid)
    r0 = int(rng.integers(0, grid - side + 1))
    c0 = int(rng.integers(0, grid - side + 1))
    m = np.zeros((grid, grid), bool)
    m[r0:r0 + side, c0:c0 + side] = True
    return m.reshape(-1)


def train_mask(rng, grid=GRID, ladder=MASK_LADDER, p_rect=0.75, p_full=0.5):
    """Training masks. A rectangle is the piece; a partially-revealed rectangle is what the
    model sees mid-decode (so any-order decoding INSIDE a hole is in distribution); a scatter
    mask with a cosine ratio is MaskGIT's own schedule, kept as the tail."""
    if rng.random() < p_rect:
        m = rect_mask(rng, int(rng.choice(ladder)), grid)
        if rng.random() >= p_full:                       # partially revealed rectangle
            idx = np.flatnonzero(m)
            keep = max(1, int(math.ceil(rng.random() * len(idx))))
            sub = rng.permutation(idx)[:keep]
            m = np.zeros(grid * grid, bool); m[sub] = True
        return m
    ratio = math.cos(0.5 * math.pi * rng.random())       # in (0, 1]
    n = max(1, int(round(ratio * grid * grid)))
    m = np.zeros(grid * grid, bool)
    m[rng.permutation(grid * grid)[:n]] = True
    return m


def batch_masks(rng, B, grid=GRID, **kw):
    return np.stack([train_mask(rng, grid, **kw) for _ in range(B)])


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train(model, codes, styles, steps, bs=128, lr=3e-4, wd=0.05, seed=0, dev="cuda",
          p_style_drop=0.1, log_every=0, warm=200, grid=GRID, ckpts=(), on_ckpt=None):
    """codes: (N, T) int64 code grids. styles: (N,) int64 style ids."""
    rng = np.random.default_rng(seed)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * t / steps)))
    C = torch.as_tensor(codes, dtype=torch.long, device=dev)
    S = torch.as_tensor(styles, dtype=torch.long, device=dev)
    N, T = C.shape
    ck = set(int(c) for c in ckpts)
    hist = []
    for t in range(steps):
        idx = torch.as_tensor(rng.integers(0, N, size=bs), device=dev)
        x, s = C[idx].clone(), S[idx].clone()
        m = torch.as_tensor(batch_masks(rng, bs, grid), device=dev)
        y = x.clone()
        x[m] = model.K
        drop = torch.as_tensor(rng.random(bs) < p_style_drop, device=dev)
        s[drop] = model.n_styles
        with amp(dev):
            logits = model(x, s)
        loss = F.cross_entropy(logits[m].float(), y[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log_every and (t % log_every == 0 or t == steps - 1):
            hist.append((t, float(loss)))
            print(f"    step {t:6d}  loss {float(loss):.4f}", flush=True)
        if on_ckpt and (t + 1) in ck:
            model.eval(); on_ckpt(t + 1, float(loss)); model.train()
    model.eval()
    return hist


# --------------------------------------------------------------------------- #
# evaluation: per-(style, mask-size) held-out NLL
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ladder_nll(model, codes, styles, ladder=MASK_LADDER, reps=16, seed=0, dev="cuda",
               bs=256, use_style=True, grid=GRID):
    """Mean per-token NLL of the TRUE codes at a full rect hole of each side. (n_style, n_side)."""
    rng = np.random.default_rng(seed)
    model.eval()
    C = torch.as_tensor(codes, dtype=torch.long, device=dev)
    S = torch.as_tensor(styles, dtype=torch.long, device=dev)
    n_style = int(S.max().item()) + 1
    out = np.zeros((n_style, len(ladder)))
    for j, side in enumerate(ladder):
        tot = np.zeros(n_style); cnt = np.zeros(n_style)
        rows = np.repeat(np.arange(len(codes)), reps)
        for a in range(0, len(rows), bs):
            sel = rows[a:a + bs]
            idx = torch.as_tensor(sel, device=dev)
            x, s = C[idx].clone(), S[idx].clone()
            if not use_style:
                s = torch.full_like(s, model.n_styles)
            m = torch.as_tensor(np.stack([rect_mask(rng, side, grid) for _ in sel]), device=dev)
            y = x.clone()
            x[m] = model.K
            with amp(dev):
                logits = model(x, s)
            lp = F.log_softmax(logits.float(), -1)
            nll = -lp.gather(-1, y[..., None])[..., 0]
            per = (nll * m).sum(1) / m.sum(1)
            pn = per.detach().cpu().numpy()
            sn = S[idx].detach().cpu().numpy()
            np.add.at(tot, sn, pn); np.add.at(cnt, sn, 1)
        out[:, j] = tot / np.maximum(cnt, 1)
    return out
