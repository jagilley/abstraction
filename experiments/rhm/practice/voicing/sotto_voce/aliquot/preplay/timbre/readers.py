"""[preplay/timbre] THE READER FORMS — the same pooled state, read by richer readouts.

The object `pp1` / `pp3` / `pp5` read is `VoProjBank`'s projection: the plant's per-block hiddens
pooled two ways (over all blocks, over the span) on the configuration with the first block outside
the span masked, standardised, crossed with a root one-hot, and read by a ridge-logistic IRLS
(1737 columns). Every claim those rounds made was made with that linear readout. This module adds
readers of the SAME state, fit on the SAME bank rows (pp1's shared subsample), through the SAME
three trunks, so a column in the re-read differs from pp1's by the reader's form alone:

  ridge      the projection's own form, refit (`preplay.pj_fit`, imported, so arm 0 IS pp1's
             `shaped_refit` / `frozen` / `twin` objects)                       ARM 0, THE GATE
  belief     the same ridge with THE PLANT'S BELIEF appended: the block head's log-softmax over
             the level-1 alphabet (v = 8), from the SAME trunk pass on the SAME masked input,
             pooled the same two ways the hiddens are (16 columns), standardised exactly like
             every state column and crossed with the root like them (1881 columns)
  bonly_lin  the ridge on the 16 belief columns alone (153 columns)            pp1 / pp3 only
  mlp        a 128-unit GELU MLP over the standardised pooled hiddens plus the root one-hot,
             BCE, AdamW, early-stopped and (lr, weight decay)-selected on the validation rows
  bonly      the same MLP on the 16 belief columns plus the root one-hot: the belief alone,
             read nonlinearly (`precision`'s public reader, ported)
  slot       the projection's form fit per SLOT: on the bank rows at the fired cell's own
             (blk0, span) only, all of them (the shared subsample holds too few); built only
             where the slot holds >= SLOT_MIN_TR training rows                  pp1 / pp3 only

THE STANDARDISATION. `precision/express.py` scales every appended column to the state's own
per-dimension RMS through the Gram, because its state columns are raw and a shared ridge penalty
would otherwise crush or favour the appended block. Here `pj_design` already z-scores EVERY state
column to unit sd before the uniform penalty, so the port of the idea is to z-score the belief
columns on the same slice by the same code: each appended column then meets the penalty on exactly
the scale each state column does. No mapping back is needed; `(mu, sd)` travel with the head.

A STRUCTURAL FACT about this plant, stated because it bounds what `belief` can add: the block head
is `feature_head(pooled)`, a linear map of the very per-block hidden that is pooled, so the pooled
LOGITS are already inside the linear span of the pooled hiddens. The log-softmax differs from the
logits by one log-partition per block; pooled, the belief block adds to the span only the two
pooled log-partitions (all blocks, span), each crossed with the root. Any difference between
`belief` and `ridge` is therefore those two scalars or the ridge's shrinkage treating 16 handed-over
directions differently from the same directions inside 192 -- `precision`'s `+logits` question,
answered here by the plant's construction.
"""

import numpy as np

# the MLP recipe: `striatum/task.py`'s 128 hidden units, GELU, one hidden layer; BCE because the
# target is a verdict and the projection is logistic; held-out selection because ~8k rows. The
# learning rate and the weight decay are BOTH a ladder chosen on the validation rows, per fit
# (`mlp_recipe` in `timbre.py` showed a single fixed rate under-trains the frozen / twin readers
# or over-shoots the shaped one; NOTES section 2). The six (lr, wd) models train as one batched
# ensemble on the same minibatch stream, so the ladder costs one fit's kernel launches.
MLP_HIDDEN = 128
MLP_LRS = (1e-3, 3e-3)
MLP_WDS = (0.0, 1e-2, 1e-1)
MLP_STEPS = 8000
MLP_BS = 512
MLP_EVERY = 100
MLP_SEED = 20260923

SLOT_MIN_TR = 1000
SLOT_MIN_VA = 200

# form -> (estimator, feature block). "h" = pooled hiddens, "b" = pooled belief, "hb" = both.
FORMS = {"ridge": ("lin", "h"), "belief": ("lin", "hb"), "bonly_lin": ("lin", "b"),
         "mlp": ("mlp", "h"), "bonly": ("mlp", "b"), "slot": ("lin", "h")}
TRUNKS = ("shaped", "frozen", "twin")


def feats(core, x, blk, spn, s, nb, tdim, device, mask=True, chunk=4096):
    """`preplay.pj_features` op for op (so the hidden block is BIT-identical to it, gate T-1),
    returning also the block head's log-softmax from the SAME `SN.trunk` pass, pooled the same two
    ways. Returns `(Fh (n, 2*tdim), Fb (n, 2*v))`, float32 on the CPU."""
    import torch
    import rhm.practice.native.span.span_net as SN
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.preplay import _col
    n = int(x.shape[0])
    blk = _col(blk, n)
    spn = _col(spn, n)
    unk = (blk < 0)
    b0_all = torch.where(unk, torch.zeros_like(blk), blk)
    sw_all = torch.where(unk, torch.full_like(spn, nb - 1), spn.clamp(min=1)).clamp(max=nb - 1)
    ar = torch.arange(nb)
    oh, ob = [], []
    with torch.no_grad():
        for a in range(0, n, int(chunk)):
            b_ = min(a + int(chunk), n)
            xb = x[a:b_].to(device).long().clone()
            b0 = b0_all[a:b_]
            sw = sw_all[a:b_]
            inside = ((ar[None, :] >= b0[:, None]) & (ar[None, :] < (b0 + sw)[:, None]))
            if mask:
                mb = (~inside).long().argmax(1)
                pos = (mb[:, None] * s + torch.arange(s)[None, :]).to(device)
                xb = xb.scatter(1, pos, torch.full_like(pos, -1))
            pooled, logits = SN.trunk(core, xb)
            wgt = inside.float().to(device)
            wgt = wgt / wgt.sum(1, keepdim=True).clamp(min=1.0)
            oh.append(torch.cat([pooled.mean(1),
                                 (pooled * wgt[:, :, None]).sum(1)], 1).float().cpu())
            lsm = torch.log_softmax(logits.float(), dim=-1)
            ob.append(torch.cat([lsm.mean(1), (lsm * wgt[:, :, None]).sum(1)], 1).float().cpu())
    if not oh:
        return torch.zeros(0, 2 * tdim), torch.zeros(0, 2 * int(core.feature_head.out_features))
    return torch.cat(oh), torch.cat(ob)


def pick(Fh, Fb, cols):
    import torch
    if cols == "h":
        return Fh
    if cols == "b":
        return Fb
    return torch.cat([Fh, Fb], 1)


# --------------------------------------------------------------------------------------- #
# the linear readout on any feature block: `preplay.pj_fit`'s solve, features handed in
# --------------------------------------------------------------------------------------- #

def fit_lin(F, r, y, ntr, v, device):
    """`preplay.pj_fit`'s body with the features handed in rather than computed: `pj_design`
    (standardise on the pooled train+validation slice, root one-hot, root x feature, bias),
    `pj_irls` over the projection's own ridge ladder, the ridge chosen on validation AUC. On the
    hidden block this reproduces `pj_fit` (gate T-2)."""
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    Zall, mu, sd = PP.pj_design(F, r, v)
    pen = PP.pj_pen(int(Zall.shape[1]))
    Zt, Zv = Zall[:ntr].to(device), Zall[ntr:].to(device)
    yt, yv = y[:ntr], y[ntr:].numpy()
    best = None
    for lam in PP.PJ_RIDGES:
        w = PP.pj_irls(Zt, yt, lam, pen)
        au = PP.vo_auc(PP.pj_score(Zv, w, device).numpy(), yv)
        if au is not None and (best is None or au > best[0]):
            best = (au, lam, w)
    if best is None:
        return None
    return {"kind": "lin", "w": best[2], "mu": mu, "sd": sd, "lam": float(best[1]),
            "val_auc": float(best[0]), "nfeat": int(Zall.shape[1])}


def lin_predict(F, r, hd, v, device):
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    Z, _, _ = PP.pj_design(F, r, v, mu=hd["mu"], sd=hd["sd"])
    return PP.pj_score(Z, hd["w"], device)


# --------------------------------------------------------------------------------------- #
# the MLP readout
# --------------------------------------------------------------------------------------- #

def _mlp_design(F, r, v, mu, sd):
    import torch
    F = F.float()
    n = int(F.shape[0])
    oh = torch.zeros(n, v)
    oh.scatter_(1, r.cpu().long().clamp(0, v - 1)[:, None], 1.0)
    return torch.cat([(F - mu) / sd, oh], 1)


def _mk_net(d):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(int(d), MLP_HIDDEN), nn.GELU(), nn.Linear(MLP_HIDDEN, 1))


def _auc_rows(S, yb):
    """Rank AUC of each row of S (K, n) against the boolean labels; ties are not averaged, which
    is immaterial for continuous scores and used only for early stopping / selection."""
    r = S.argsort(1).argsort(1).double() + 1.0
    n1 = float(yb.sum())
    n0 = float(yb.numel()) - n1
    return ((r[:, yb].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)).cpu().numpy()


def fit_mlp(F, r, y, ntr, v, device, seed=MLP_SEED, steps=MLP_STEPS, lrs=MLP_LRS, wds=MLP_WDS):
    """128 GELU units over [standardised features, root one-hot], BCE with logits, AdamW
    (decoupled weight decay, betas 0.9 / 0.999) with a cosine schedule, minibatches of 512 drawn
    from the training rows. HELD-OUT SELECTION: every (lr, wd) pair of the ladder is trained, the
    validation AUC is read every 100 steps and each model's best state kept (early stopping), and
    the pair with the best validation AUC is the reader -- the ridge's lambda ladder's analogue,
    on the same rows by the same criterion. The K models share one initialisation and one
    minibatch stream and are trained as one batched ensemble (a hand-written AdamW over the
    stacked parameters, so each model has its own rate and decay). Nothing touches the global
    torch RNG."""
    import torch
    import torch.nn.functional as TF
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    F = F.float()
    mu = F.mean(0)
    sd = F.std(0).clamp(min=1e-6)
    X = _mlp_design(F, r, v, mu, sd)
    Xt, Xv = X[:ntr].to(device), X[ntr:].to(device)
    yt = y[:ntr].float().to(device)
    yvb = (y[ntr:] > 0.5).to(device)
    d, H = int(X.shape[1]), MLP_HIDDEN
    grid = [(float(a), float(b)) for a in lrs for b in wds]
    K = len(grid)
    lr0 = torch.tensor([q[0] for q in grid], dtype=torch.float32, device=device)
    wdk = torch.tensor([q[1] for q in grid], dtype=torch.float32, device=device)
    base = float(np.clip(float(yt.mean()), 1e-4, 1 - 1e-4))
    g = torch.Generator().manual_seed(int(seed))
    bnd = 1.0 / np.sqrt(d)
    W1 = ((torch.rand(d, H, generator=g) * 2 - 1) * bnd).to(device)
    b1 = ((torch.rand(H, generator=g) * 2 - 1) * bnd).to(device)
    P = [W1[None].repeat(K, 1, 1).contiguous(), b1[None].repeat(K, 1).contiguous(),
         torch.zeros(K, H, device=device),
         torch.full((K,), float(np.log(base / (1 - base))), device=device)]
    for p_ in P:
        p_.requires_grad_(True)
    M = [torch.zeros_like(p_) for p_ in P]
    V = [torch.zeros_like(p_) for p_ in P]
    be1, be2, eps = 0.9, 0.999, 1e-8
    gb = torch.Generator().manual_seed(int(seed) + 1)

    def fwd(Xb):
        h = TF.gelu(torch.einsum("bd,kdh->kbh", Xb, P[0]) + P[1][:, None, :])
        return torch.einsum("kbh,kh->kb", h, P[2]) + P[3][:, None]

    best_auc = np.full(K, -1.0)
    best_step = np.zeros(K, np.int64)
    best_P = [p_.detach().clone() for p_ in P]
    for st in range(int(steps)):
        i = torch.randint(0, int(ntr), (MLP_BS,), generator=gb).to(device)
        o = fwd(Xt[i])
        loss = TF.binary_cross_entropy_with_logits(
            o, yt[i][None, :].expand(K, -1), reduction="none").mean(1).sum()
        grads = torch.autograd.grad(loss, P)
        with torch.no_grad():
            lr_t = lr0 * 0.5 * (1.0 + float(np.cos(np.pi * st / float(steps))))
            c1, c2 = 1.0 - be1 ** (st + 1), 1.0 - be2 ** (st + 1)
            for p_, g_, m_, v_ in zip(P, grads, M, V):
                sh = (K,) + (1,) * (p_.dim() - 1)
                p_.mul_(1.0 - (lr_t * wdk).view(sh))
                m_.mul_(be1).add_(g_, alpha=1 - be1)
                v_.mul_(be2).addcmul_(g_, g_, value=1 - be2)
                p_.sub_(lr_t.view(sh) * (m_ / c1) / ((v_ / c2).sqrt() + eps))
        if (st + 1) % MLP_EVERY == 0 or st + 1 == int(steps):
            with torch.no_grad():
                au = _auc_rows(fwd(Xv), yvb)
                up = au > best_auc
                if up.any():
                    ku = torch.from_numpy(np.nonzero(up)[0]).to(device)
                    for bp, p_ in zip(best_P, P):
                        bp[ku] = p_.detach()[ku]
                    best_auc[up] = au[up]
                    best_step[up] = st + 1
    kb = int(np.argmax(best_auc))
    net = _mk_net(d)
    with torch.no_grad():
        net[0].weight.copy_(best_P[0][kb].T.cpu())
        net[0].bias.copy_(best_P[1][kb].cpu())
        net[2].weight.copy_(best_P[2][kb][None, :].cpu())
        net[2].bias.copy_(best_P[3][kb].reshape(1).cpu())
    net = net.to(device).eval()
    hd = {"kind": "mlp", "net": net, "mu": mu, "sd": sd, "lr": grid[kb][0], "wd": grid[kb][1],
          "step": int(best_step[kb]), "nfeat": d, "lam": -1.0,
          "trace": [{"lr": grid[k][0], "wd": grid[k][1], "val_auc_fast": float(best_auc[k]),
                     "step": int(best_step[k])} for k in range(K)]}
    # the reported validation AUC is the exact tie-averaged one, through the prediction path
    pv = mlp_predict(F[ntr:], r[ntr:], hd, v, device).numpy()
    hd["val_auc"] = float(PP.vo_auc(pv, y[ntr:].numpy()))
    return hd


def mlp_predict(F, r, hd, v, device):
    import torch
    X = _mlp_design(F, r, v, hd["mu"], hd["sd"]).to(device)
    with torch.no_grad():
        return torch.sigmoid(hd["net"](X).squeeze(-1)).float().cpu()


def predict(hd, form, Fh, Fb, r, v, device):
    est, cols = FORMS[form]
    F = pick(Fh, Fb, cols)
    return (lin_predict if est == "lin" else mlp_predict)(F, r, hd, v, device)


def fit_form(form, Fh, Fb, r, y, ntr, v, device):
    est, cols = FORMS[form]
    F = pick(Fh, Fb, cols)
    return (fit_lin if est == "lin" else fit_mlp)(F, r, y, ntr, v, device)


def head_record(hd):
    """The JSON-able part of a head."""
    q = {"kind": hd["kind"], "lam": hd.get("lam"), "val_auc": hd.get("val_auc"),
         "nfeat": hd.get("nfeat")}
    if hd["kind"] == "mlp":
        q.update({"lr": hd.get("lr"), "wd": hd["wd"], "step": hd["step"],
                  "trace": hd["trace"]})
    for k in ("n_tr", "n_va"):
        if k in hd:
            q[k] = hd[k]
    return q


def read_heads(cores, heads, banked, xf, roots_t, blk0, span, s, nb, tdim, v, device):
    """ONE trunk pass per trunk on the fired configurations, every head read off it. `banked` is
    the arm's `(w, mu, sd)` read through the shaped trunk's hidden block, which is
    `preplay.pj_predict` bit for bit (gate T-2). Returns {reader key: float32 per instance}."""
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    out = {}
    for tk, core in cores.items():
        want = [k for k in heads if k.split("@")[1] == tk]
        if not want and not (tk == "shaped" and banked is not None):
            continue
        Fh, Fb = feats(core, xf, blk0, span, s, nb, tdim, device)
        if tk == "shaped" and banked is not None:
            Z, _, _ = PP.pj_design(Fh, roots_t, v, mu=banked[1], sd=banked[2])
            out["banked"] = PP.pj_score(Z, banked[0], device).numpy().astype("float32")
        for k in want:
            out[k] = predict(heads[k], k.split("@")[0], Fh, Fb, roots_t, v,
                             device).numpy().astype("float32")
    return out


def xr_keys(x_np, r_np):
    """One hashable key per (configuration, root) row -- `readgate.keyset`'s idiom."""
    a = np.ascontiguousarray(np.concatenate(
        [np.asarray(x_np).astype(np.int64), np.asarray(r_np).astype(np.int64)[:, None]], 1))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
