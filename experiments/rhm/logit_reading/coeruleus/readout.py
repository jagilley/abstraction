"""Q2: a self-supervised readout of the model's own excess surprise.

A small head on the FROZEN model's residual stream at position t, trained with no labels to
predict

    y_t = sum_{u=t}^{t+H} ( -log q_u(x_{u+1}) - H(q_u) )

the model's own realised surprise over the next H+1 predictions, in excess of the uncertainty
it stated for them. By the `altitude/` Q1 identity this quantity is zero-mean once the model is
trained, so the head is asked "am I about to be surprised beyond my own stated uncertainty",
which is a different question from "is this a hard position": at the model's frontier the
entropy is high and the loss matches it; after a fallback the entropy is low and the loss does
not.

Trained on fresh eps-corrupted windows from the distribution the model was trained on. Never
sees a corruption mask or a violation label.

**The realised target is very noisy.** Its sd is ~2.6 nats over the horizon while the
conditional mean the head can actually predict moves by ~0.05 nats between a quiet position and
a post-corruption one, so an SGD-fit MLP is not a fair test of whether the information is there.
The primary readout here is therefore the exact minimum-MSE LINEAR head, solved in closed form
from a streamed Gram matrix (every block, every ridge penalty, no optimiser), with an MLP beside
it. Both are scored the same way.

Readouts:
  * held-out R^2 / rank correlation against the target
  * does it fire at corrupted positions -- AUC(t = c) vs quiet positions, and the profile over tau
  * does it fire at the `swap65k` same-prefix violations (the parent's Part 2c design), with the
    ORACLE-LABEL probe on the same states as the reference
  * calibrated-hard vs overconfident-wrong: the cell that stated a lot of uncertainty over the
    horizon and was not surprised, against the cell that stated little and was

Floors carried from the parent: a random-init trunk, and the input embedding (`post_embed`).
Contrasts: the same heads trained on the RAW future NLL, and H(q) alone.

Run:
  modal run -m rhm.logit_reading.coeruleus.readout::readout_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_eps01_s42/step064000.pt \
      --n-train 1024 --n-val 512 --n-mlp 512 --steps 300 --max-windows 4000 --tag smoke
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import BLOCKS, auc, fit_probe, _load_stimuli
from rhm.logit_reading.phasic import balance_tokens, balance_ds_sign, paired_win
from rhm.logit_reading.coeruleus.events import (
    corrupted_windows, events_from_mask, quiet_positions, log_softmax, ent)

LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# forward helpers
# ---------------------------------------------------------------------------

def states_at(model, windows, block, device="cuda", bs=256):
    import torch
    T = model.block_size
    out = []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            _, _, inter = model(x, return_intermediates=True)
            out.append(inter[block].float().cpu().numpy())
    return np.concatenate(out)


def states_at_t(model, windows, t, block, device="cuda", bs=256):
    import torch
    T = model.block_size
    out = []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            tt = torch.as_tensor(t[c0:c0 + bs], device=device)
            _, _, inter = model(x, return_intermediates=True)
            out.append(inter[block][torch.arange(len(x), device=device), tt].float().cpu().numpy())
    return np.concatenate(out)


def logq_of(model, windows, device="cuda", bs=256):
    import torch
    T = model.block_size
    out = []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            lg, _ = model(x)
            out.append(torch.log_softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def logq_at(model, windows, t, device="cuda", bs=256):
    import torch
    T = model.block_size
    out = []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            tt = torch.as_tensor(t[c0:c0 + bs], device=device)
            lg, _ = model(x)
            lsm = torch.log_softmax(lg.float(), -1)
            out.append(lsm[torch.arange(len(x), device=device), tt].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def horizon_targets(logq, tok, H):
    """(y_excess, y_nll, y_ent) of shape (n, T-H) for t = 0..T-1-H, plus (nll, Hq)."""
    nll = -np.take_along_axis(logq, tok[..., None], -1)[..., 0]
    Hq = ent(np.exp(logq))

    def csum(a):
        c = np.concatenate([np.zeros((len(a), 1)), np.cumsum(a, 1)], 1)
        return c[:, H + 1:] - c[:, :-(H + 1)]
    return csum(nll - Hq), csum(nll), csum(Hq), nll, Hq


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------

class LinearHead:
    """beta from a closed-form ridge solve; numpy, no optimiser."""

    def __init__(self, beta, block, lam=0.0):
        self.beta, self.block, self.lam = np.asarray(beta, np.float64), block, lam

    def __call__(self, X):
        X = np.asarray(X, np.float64)
        return X @ self.beta[:-1] + self.beta[-1]

    def torch_state(self):
        import torch
        return {"kind": "linear", "beta": torch.as_tensor(self.beta), "block": self.block,
                "lam": self.lam}


class Head:
    """Standardising MLP, kept on the device it trains on."""

    def __init__(self, net, mu, sd, ym, ys, block):
        self.net, self.mu, self.sd, self.ym, self.ys, self.block = net, mu, sd, ym, ys, block

    def __call__(self, X):
        import torch
        out = []
        X = np.asarray(X, np.float32)
        with torch.no_grad():
            for i in range(0, len(X), 262144):
                t = torch.as_tensor(X[i:i + 262144], device=self.mu.device)
                out.append((self.net((t - self.mu) / self.sd)[:, 0] * self.ys
                            + self.ym).cpu().numpy())
        return np.concatenate(out)

    def torch_state(self):
        return {"kind": "mlp",
                "net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "mu": self.mu.detach().cpu(), "sd": self.sd.detach().cpu(),
                "ym": float(self.ym), "ys": float(self.ys), "block": self.block}


def train_head(X, y, block="", hidden=128, steps=3000, bs=8192, lr=3e-4, wd=0.0,
               seed=0, device="cuda"):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    yt = torch.as_tensor(np.asarray(y, np.float32), device=device)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-5
    ym, ys = float(yt.mean()), float(yt.std()) + 1e-8
    d = Xt.shape[1]
    net = (nn.Linear(d, 1) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))).to(device)
    with torch.no_grad():                      # start at the mean predictor
        last = net if hidden == 0 else net[-1]
        last.weight.zero_(); last.bias.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed + 1)
    yn = (yt - ym) / ys
    for _ in range(steps):
        i = torch.randint(0, len(Xt), (min(bs, len(Xt)),), generator=g).to(device)
        loss = ((net((Xt[i] - mu) / sd)[:, 0] - yn[i]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    del Xt, yt, yn
    return Head(net, mu, sd, ym, ys, block)


def load_head(path, device="cuda"):
    import torch
    import torch.nn as nn
    sd = torch.load(path, map_location=device)
    st = sd["state"]
    if st.get("kind", "mlp") == "linear":
        return LinearHead(st["beta"].cpu().numpy(), st["block"], st.get("lam", 0.0)), sd["config"]
    d = st["mu"].shape[1]
    hidden = sd["config"]["hidden"]
    net = (nn.Linear(d, 1) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))).to(device)
    net.load_state_dict({k: v.to(device) for k, v in st["net"].items()})
    net.eval()
    return Head(net, st["mu"].to(device), st["sd"].to(device), st["ym"], st["ys"],
                st["block"]), sd["config"]


# ---------------------------------------------------------------------------
# streamed ridge over every block at once
# ---------------------------------------------------------------------------

def gram_stream(model, windows, targets, blocks, Tm, device="cuda", bs=512):
    """Accumulate [X 1]^T [X 1] and [X 1]^T y for every block, in float64 on the GPU.

    targets: dict name -> (n, Tm) array. Returns (A, C, n_rows) with A[b] (d+1, d+1) and
    C[b][name] (d+1,)."""
    import torch
    T = model.block_size
    d = model.transformer.wte.weight.shape[1]
    A = {b: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device) for b in blocks}
    C = {b: {k: torch.zeros(d + 1, dtype=torch.float64, device=device) for k in targets}
         for b in blocks}
    n_rows = 0
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            _, _, inter = model(x, return_intermediates=True)
            ys = {k: torch.as_tensor(v[c0:c0 + bs].reshape(-1), dtype=torch.float64,
                                     device=device) for k, v in targets.items()}
            for b in blocks:
                X = inter[b][:, :Tm].reshape(-1, d).double()
                X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64, device=device)], 1)
                A[b] += X.T @ X
                for k, yv in ys.items():
                    C[b][k] += X.T @ yv
            n_rows += len(x) * Tm
    return ({b: A[b].cpu().numpy() for b in blocks},
            {b: {k: v.cpu().numpy() for k, v in C[b].items()} for b in blocks}, n_rows)


def ridge_solve(A, c, lam, n_rows):
    d = A.shape[0] - 1
    R = np.eye(d + 1) * (lam * np.trace(A[:d, :d]) / max(d, 1))
    R[-1, -1] = 0.0
    return np.linalg.solve(A + R, c)


def _r2(pred, y):
    y = np.asarray(y, np.float64)
    p = np.asarray(pred, np.float64)
    return float(1.0 - ((y - p) ** 2).mean() / max(y.var(), 1e-12))


def _spearman(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


class IdentityHead:
    block = "scalar"

    def __call__(self, X):
        return np.asarray(X, np.float64)[:, 0]


# ---------------------------------------------------------------------------
# same-prefix pairs (the parent's Part 2c construction, both splits)
# ---------------------------------------------------------------------------

def prefix_pairs(model, S, device="cuda", seed=0, caliper=0.3, t_lo=16, max_windows=0):
    We = S["windows_edit"]
    T = We.shape[1] - 1
    et, tv, ks, pL = S["etype"], S["t_v"], S["k_star"], S["pL_edit"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    if max_windows and len(vw) > max_windows:
        vw = np.sort(np.random.default_rng(seed + 7).choice(vw, max_windows, replace=False))
    t = tv[vw]
    lqp_v = logq_at(model, We[vw], t - 1, device)
    x_v = We[vw, t]
    s_all = -lqp_v
    s_v = s_all[np.arange(len(vw)), x_v]
    legal = pL[vw, t] > 0
    assert not legal[np.arange(len(vw)), x_v].any(), "a violating token is legal under p_L"
    dist = np.where(legal, np.abs(s_all - s_v[:, None]), np.inf)
    x_c = dist.argmin(1)
    has = dist.min(1) <= caliper
    P = np.where(has)[0]
    twin = We[vw[P]].copy()
    twin[np.arange(len(P)), t[P]] = x_c[P]
    lqp_c = logq_at(model, twin, t[P] - 1, device)
    surpr_v = -lqp_v[P][np.arange(len(P)), x_v[P]]
    surpr_c = -lqp_c[np.arange(len(P)), x_c[P]]
    chk = float(np.abs(lqp_c - lqp_v[P]).max())
    assert chk < 1e-3, f"same-prefix twins must share q at t-1 ({chk})"
    rng = np.random.default_rng(seed + 2)
    split = np.random.default_rng(seed + 1).random(len(P)) < 0.6
    keep = np.zeros(len(P), bool)
    for flag in (split, ~split):
        sub = balance_ds_sign(surpr_v - surpr_c, np.where(flag)[0], rng)
        if len(sub):
            keep[balance_tokens(x_v[P], x_c[P], sub, rng)] = True
    return {"n_candidates": int(len(vw)), "n_matched": int(has.sum()),
            "tr": (split & keep), "te": ((~split) & keep),
            "t": t[P], "kstar": ks[vw[P]], "viol_win": We[vw[P]], "twin_win": twin,
            "ds": surpr_v - surpr_c, "prefix_check": chk}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def eval_pred(pred, y, y_ent, corrupt, H, name, block="?", seed=0):
    n, Tm = y.shape
    out = {"name": name, "block": block, "R2": _r2(pred.ravel(), y.ravel()),
           "spearman": _spearman(pred.ravel(), y.ravel())}
    win, c = events_from_mask(corrupt, H, c_lo=8)
    qw, qt = quiet_positions(corrupt, H, t_lo=8, rng=np.random.default_rng(seed))
    m = qt < Tm
    qw, qt = qw[m], qt[m]
    if len(win) >= 50 and len(qw) >= 50:
        out["auc_event_vs_quiet"] = auc(
            np.concatenate([pred[win, c], pred[qw, qt]]),
            np.concatenate([np.ones(len(win), bool), np.zeros(len(qw), bool)]))
        out["mean_quiet"] = float(pred[qw, qt].mean())
        out["sd_quiet"] = float(pred[qw, qt].std())
        out["n_event"], out["n_quiet"] = int(len(win)), int(len(qw))
        out["by_tau"] = {}
        for tau in range(-2, H + 1):
            t_ = c + tau
            ok = (t_ >= 0) & (t_ < Tm)
            if ok.sum() > 20:
                out["by_tau"][tau] = float(pred[win[ok], t_[ok]].mean())
    e, h, p_ = y.ravel(), y_ent.ravel(), pred.ravel()
    qe, qh = np.quantile(e, [0.25, 0.75]), np.quantile(h, [0.25, 0.75])
    hard, wrong = (h >= qh[1]) & (e <= qe[0]), (h <= qh[0]) & (e >= qe[1])
    if hard.sum() > 50 and wrong.sum() > 50:
        out["hard_vs_wrong"] = {
            "n_hard": int(hard.sum()), "n_wrong": int(wrong.sum()),
            "auc": auc(np.concatenate([p_[wrong], p_[hard]]),
                       np.concatenate([np.ones(int(wrong.sum()), bool),
                                       np.zeros(int(hard.sum()), bool)])),
            "mean_hard": float(p_[hard].mean()), "mean_wrong": float(p_[wrong].mean())}
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=24576,
              max_containers=4)
def readout_ckpt(ckpt: str, eps_data: float = 0.01, horizon: int = 8, hidden: int = 128,
                 n_train: int = 65536, n_val: int = 4096, n_mlp: int = 16384,
                 steps: int = 6000, train_seed: int = 90210, val_seed: int = 90211,
                 stim_tag: str = "swap65k", caliper: float = 0.3, max_windows: int = 0,
                 seed: int = 0, save: bool = True, tag: str = ""):
    import resource
    import torch
    from rhm.model import GPT
    volume.reload()
    H = horizon
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    T = cfg["s"] ** cfg["L"]
    Tm = T - H

    def build(n, sd):
        w, _, corrupt, _ = corrupted_windows(rules, rule_w, n, T + 1, sd, eps_data,
                                             noise_seed=sd + 31)
        lq = logq_of(model, w, "cuda")
        y_ex, y_nll, y_ent, nll, Hq = horizon_targets(lq, w[:, 1:], H)
        return dict(w=w, corrupt=corrupt, y_ex=y_ex, y_nll=y_nll, y_ent=y_ent, Hq=Hq)

    tr, va = build(n_train, train_seed), build(n_val, val_seed)
    res = {"ckpt": ckpt, "step": cfg.get("step"), "eps_train": cfg.get("eps_train", 0.0),
           "eps_data": eps_data, "H": H, "n_train": n_train, "n_val": n_val, "hidden": hidden,
           "n_rows_train": int(n_train * Tm),
           "target_mean": float(tr["y_ex"].mean()), "target_sd": float(tr["y_ex"].std()),
           "rawnll_mean": float(tr["y_nll"].mean()), "rawnll_sd": float(tr["y_nll"].std()),
           "corr_target_rawnll": float(np.corrcoef(tr["y_ex"].ravel(),
                                                   tr["y_nll"].ravel())[0, 1])}
    print(f"step {cfg.get('step')}  excess target mean {res['target_mean']:+.4f} "
          f"sd {res['target_sd']:.4f}  corr with raw NLL {res['corr_target_rawnll']:+.3f}  "
          f"rows {res['n_rows_train']:,}", flush=True)

    torch.manual_seed(12345)
    rand = GPT(cfg["v"], T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to("cuda").eval()

    # ---- exact linear heads, every block, every lambda, streamed ----------
    targets = {"excess": tr["y_ex"], "rawnll": tr["y_nll"]}
    res["ridge"] = {}
    heads, preds = {}, {}
    for tname, mdl in (("state", model), ("rand", rand)):
        A, C, nr = gram_stream(mdl, tr["w"], targets, BLOCKS, Tm)
        Sva = {}
        for b in BLOCKS:
            Sva[b] = states_at(mdl, va["w"], b, "cuda")[:, :Tm]
        for target in ("excess", "rawnll"):
            yv = va["y_ex"] if target == "excess" else va["y_nll"]
            best = (None, -1e9)
            for b in BLOCKS:
                for lam in LAMS:
                    beta = ridge_solve(A[b], C[b][target], lam, nr)
                    d = Sva[b].shape[-1]
                    pv = (Sva[b].reshape(-1, d) @ beta[:-1] + beta[-1]).reshape(n_val, Tm)
                    r2 = _r2(pv.ravel(), yv.ravel())
                    res["ridge"][f"{tname}_{target}_{b}_lam{lam}"] = r2
                    if r2 > best[1]:
                        best = ((b, lam, beta, pv), r2)
            (b, lam, beta, pv), r2 = best
            key = f"{tname}_{target}_ridge"
            heads[key] = LinearHead(beta, b, lam)
            preds[key] = pv
            res.setdefault("best_ridge", {})[key] = {"block": b, "lam": lam, "val_R2": r2}
            print(f"  ridge {key:22s} block {b} lam {lam:g}  val R2 {r2:+.6f}", flush=True)
        if tname == "state":
            best_block = heads["state_excess_ridge"].block
            emb = "post_embed"
            for target in ("excess",):
                yv = va["y_ex"]
                bb = (None, -1e9)
                for lam in LAMS:
                    beta = ridge_solve(A[emb], C[emb][target], lam, nr)
                    d = Sva[emb].shape[-1]
                    pv = (Sva[emb].reshape(-1, d) @ beta[:-1] + beta[-1]).reshape(n_val, Tm)
                    r2 = _r2(pv.ravel(), yv.ravel())
                    if r2 > bb[1]:
                        bb = ((lam, beta, pv), r2)
                (lam, beta, pv), r2 = bb
                heads["embed_excess_ridge"] = LinearHead(beta, emb, lam)
                preds["embed_excess_ridge"] = pv
                res["best_ridge"]["embed_excess_ridge"] = {"block": emb, "lam": lam,
                                                           "val_R2": r2}
        del A, C, Sva

    # ---- MLP heads on the best block --------------------------------------
    nm = min(n_mlp, n_train)
    Xm = states_at(model, tr["w"][:nm], best_block, "cuda")[:, :Tm]
    d = Xm.shape[-1]
    Xva = states_at(model, va["w"], best_block, "cuda")[:, :Tm]
    for target in ("excess", "rawnll"):
        yy = (tr["y_ex"] if target == "excess" else tr["y_nll"])[:nm]
        h = train_head(Xm.reshape(-1, d), yy.ravel(), best_block, hidden=hidden, steps=steps,
                       seed=seed)
        key = f"state_{target}_mlp"
        heads[key] = h
        preds[key] = h(Xva.reshape(-1, d)).reshape(n_val, Tm)
    del Xm, Xva

    # ---- scalar references -------------------------------------------------
    preds["Hsum_raw"] = va["y_ent"]
    preds["Hq_raw"] = va["Hq"][:, :Tm]
    heads["Hsum_raw"] = IdentityHead()

    res["heads"] = {}
    for key, pv in preds.items():
        blk = getattr(heads.get(key, IdentityHead()), "block", "scalar")
        res["heads"][key] = eval_pred(pv, va["y_ex"], va["y_ent"], va["corrupt"], H, key,
                                      blk, seed)
        if key.endswith("rawnll_ridge") or key.endswith("rawnll_mlp"):
            res["heads"][key]["R2_on_rawnll_target"] = _r2(pv.ravel(), va["y_nll"].ravel())
        r = res["heads"][key]
        print(f"  head {key:24s} R2 {r['R2']:+.6f}  rho {r['spearman']:+.4f}  "
              f"AUC(evt/quiet) {r.get('auc_event_vs_quiet', float('nan')):.3f}  "
              f"hard-vs-wrong {r.get('hard_vs_wrong', {}).get('auc', float('nan')):.3f}",
              flush=True)

    # ---- the same-prefix violation pairs (never seen by the head) ---------
    S = _load_stimuli(stim_tag)
    pp = prefix_pairs(model, S, "cuda", seed=seed, caliper=caliper, max_windows=max_windows)
    te, tstar = pp["te"], pp["t"]
    res["pairs"] = {"n_candidates": pp["n_candidates"], "n_matched": pp["n_matched"],
                    "n_train": int(pp["tr"].sum()), "n_test": int(te.sum()),
                    "prefix_check": pp["prefix_check"], "readouts": {}}
    y2 = np.concatenate([np.ones(int(te.sum()), bool), np.zeros(int(te.sum()), bool)])

    def pair_row(a, b):
        row = {"paired": paired_win(a[te], b[te]),
               "auc": auc(np.concatenate([a[te], b[te]]), y2),
               "mean_viol": float(a[te].mean()), "mean_twin": float(b[te].mean())}
        row["by_kstar"] = {int(k): {"n": int((te & (pp["kstar"] == k)).sum()),
                                    "paired": paired_win(a[te & (pp["kstar"] == k)],
                                                         b[te & (pp["kstar"] == k)])}
                           for k in range(1, 7) if (te & (pp["kstar"] == k)).sum() >= 30}
        return row

    sv = states_at_t(model, pp["viol_win"], tstar, best_block, "cuda")
    sc = states_at_t(model, pp["twin_win"], tstar, best_block, "cuda")
    for key in ("state_excess_ridge", "state_rawnll_ridge", "state_excess_mlp",
                "state_rawnll_mlp"):
        res["pairs"]["readouts"][key] = pair_row(heads[key](sv), heads[key](sc))
    for key, blk, mdl in (("embed_excess_ridge", "post_embed", model),
                          ("rand_excess_ridge", heads["rand_excess_ridge"].block, rand)):
        a = states_at_t(mdl, pp["viol_win"], tstar, blk, "cuda")
        b = states_at_t(mdl, pp["twin_win"], tstar, blk, "cuda")
        res["pairs"]["readouts"][key] = pair_row(heads[key](a), heads[key](b))
        del a, b
    lqv = logq_at(model, pp["viol_win"], tstar, "cuda")
    lqc = logq_at(model, pp["twin_win"], tstar, "cuda")
    res["pairs"]["readouts"]["H_next"] = pair_row(ent(np.exp(lqv)), ent(np.exp(lqc)))

    Xp = np.concatenate([sv, sc])
    yp = np.concatenate([np.ones(len(tstar), bool), np.zeros(len(tstar), bool)])
    m_tr = np.concatenate([pp["tr"], pp["tr"]])
    m_te = np.concatenate([pp["te"], pp["te"]])
    for nmp, hid in (("oracle_linear", 0), ("oracle_mlp", 256)):
        f = fit_probe(Xp[m_tr], yp[m_tr], hidden=hid, steps=800 if hid else 400)
        s_ = f(Xp)
        res["pairs"]["readouts"][nmp] = pair_row(s_[:len(tstar)], s_[len(tstar):])
        res["pairs"]["readouts"][nmp]["auc_pooled"] = auc(s_[m_te], yp[m_te])
    print(f"  pairs: matched {pp['n_matched']}  train {int(pp['tr'].sum())} "
          f"test {int(te.sum())}  prefix {pp['prefix_check']:.1e}", flush=True)
    for k, r in res["pairs"]["readouts"].items():
        print(f"  pairs {k:24s} paired {r['paired']:.3f}  auc {r['auc']:.3f}", flush=True)

    if save:
        for key in ("state_excess_ridge", "state_excess_mlp"):
            hp = f"{ckpt[:-3]}_coeruleus_head_{key}{('_' + tag) if tag else ''}.pt"
            torch.save({"state": heads[key].torch_state(),
                        "config": {"ckpt": ckpt, "step": cfg.get("step"), "H": H,
                                   "hidden": hidden, "block": heads[key].block,
                                   "eps_data": eps_data, "train_seed": train_seed,
                                   "n_train": n_train, "key": key,
                                   "quiet_mean": res["heads"][key].get("mean_quiet"),
                                   "quiet_sd": res["heads"][key].get("sd_quiet")}}, hp)
            res.setdefault("head_paths", {})[key] = hp
    name = f"{ckpt[:-3]}_coeruleus_readout{('_' + tag) if tag else ''}.json"
    with open(name, "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {name}",
          flush=True)
    return {"ckpt": ckpt, "step": cfg.get("step")}
