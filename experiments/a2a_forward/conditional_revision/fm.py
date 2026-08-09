"""The temporal forward model, and the probe / discriminant heads.

`TemporalFM` is a small causal transformer over the reader's residual stream:

    FM( h[<=t] )  ->  Delta_t = h[t+1] - h[t]

which is `rhm/conditional_revision`'s Gate-0 object, one substrate over. The
architecture mirrors `a2a_forward/forward_model.py::TransformerForwardModel`
(compressed q/k/v projections = pontine relay, MLP expansion = granule layer,
linear readout = Purkinje). It is re-stated here rather than imported because
the two experiments live in different Modal apps and images; behaviour is the
same and the `direct` parametrisation is the one RHM's control showed is not
load-bearing (+0.652 state-form vs +0.638 direct).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, d_model, d_head, n_head, mlp_mult):
        super().__init__()
        self.d_head, self.n_head = d_head, n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_head * n_head)
        self.k = nn.Linear(d_model, d_head * n_head)
        self.v = nn.Linear(d_model, d_head * n_head)
        self.o = nn.Linear(d_head * n_head, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hid = int(d_model * mlp_mult)
        self.mlp = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(),
                                 nn.Linear(hid, d_model))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.o(y)
        return x + self.mlp(self.ln2(x))


class TemporalFM(nn.Module):
    """Causal FM over the (standardised) residual stream; predicts the update."""

    def __init__(self, d_model, d_head=64, n_head=8, n_layer=1, mlp_mult=1.0,
                 block_size=512):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList(
            [_Block(d_model, d_head, n_head, mlp_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)
        nn.init.normal_(self.pos, std=0.01)

    def forward(self, h):
        x = h + self.pos[:, :h.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class SlotProbe(nn.Module):
    """h -> per-slot categorical over the 4 values. One hidden layer, matching
    the RHM probe recipe (`gates_ab.py::train_probe`)."""

    def __init__(self, d_model, n_slots, n_val, hidden=1024):
        super().__init__()
        self.n_slots, self.n_val = n_slots, n_val
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_slots * n_val))

    def forward(self, x):
        return self.net(x).view(-1, self.n_slots, self.n_val)


class LinearDiscriminant(nn.Module):
    """Cross-fit linear readout used for the directional-vs-scalar comparison.

    `sculpt_slip` step 1 is the precedent: the slip was identifiable in the FM
    residual's *direction* at 0.70-0.75 AUC while its *norm* read 0.50. Any
    claim that a residual 'carries' something has to be made directionally or
    it is testing the wrong thing.
    """

    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, x):
        return self.w(x).squeeze(-1)


def fit_discriminant(X_tr, y_tr, X_te, steps=600, lr=3e-3, wd=1e-2,
                     batch=1024, seed=0, device="cuda", normalise=True):
    """Balanced-class logistic regression, trained on train stories only.

    `normalise=True` projects onto the unit sphere first, so the readout is
    strictly directional and cannot smuggle in the norm.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xtr = X_tr.float()
    Xte = X_te.float()
    if normalise:
        Xtr = Xtr / Xtr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Xte = Xte / Xte.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    pos = torch.where(y_tr > 0.5)[0]
    neg = torch.where(y_tr <= 0.5)[0]
    if len(pos) < 10 or len(neg) < 10:
        return None
    d = LinearDiscriminant(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(d.parameters(), lr=lr, weight_decay=wd)
    half = batch // 2
    for _ in range(steps):
        ip = pos[torch.randint(0, len(pos), (half,), generator=g)]
        inn = neg[torch.randint(0, len(neg), (half,), generator=g)]
        ix = torch.cat([ip, inn])
        xb = Xtr[ix].to(device)
        yb = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)
        loss = F.binary_cross_entropy_with_logits(d(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    d.eval()
    with torch.no_grad():
        out = []
        for i in range(0, Xte.shape[0], 4096):
            out.append(d(Xte[i:i + 4096].to(device)).cpu())
    return torch.cat(out).numpy()
