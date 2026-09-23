"""[grokking] Rung controls (tag r2): why all-48 and top1 fail in `rung.py` r1.

Control A: all-48 as an orthogonal reparameterization of one-hot (MLP only).
  With DC included, the per-operand feature map is square. Q (97x97) has rows DC/sqrt(p),
  sqrt(2/p) cos(2 pi w a/p), sqrt(2/p) sin(2 pi w a/p) for w = 1..48, and is orthogonal. Two scales:
    q  x = blockdiag(Q, Q) onehot          (per-input norm sqrt 2, same as raw)
    f  x = sqrt(p/2) blockdiag(Q, Q) onehot (r1's per-coordinate cos/sin scale, plus a DC of 1/sqrt 2)
  Init: 'transported' sets W0 = W0_raw M^+ (M the feature map), so the initial function equals raw's
  seed-42 init; 'default' is torch's init under seed 42 (the same numbers as raw's W0, applied to
  the rotated features). Arms A1: {q transported, f transported, q default} under the recipe.
  A2 (run only if A1 fails): raw vs q-transported under SGD + L2 (`--groups A2`, settings in
  `--a2-grid lr:wd:momentum,...`; `--a2-pair 0` runs raw only, to find a setting where raw groks).

Control B: top1 (single frequency 30, 4 input dims).
  B1  bilinear m=4 with the exact one-frequency solution planted (h = [cc, ss, cs, sc],
      W[c] = [cos wc, -cos wc, sin wc, sin wc], U = V = t * selector, W = t * base, amplitude A = t^3),
      then trained under the recipe with no early stop. A in {1, 10, 100, 1e3, 1e4}.
  B2  weight decay {1.0, 0.1, 0.01, 0.0} x {bilinear m=4, MLP} from random init, cap 40k.
  B3  r1's minted-13 bilinear arm re-run (stop at 1.0, which must reproduce e100 = 3760) and the same
      arm continued to 40k, with the readout's amplitude and margin probed along the way.

Probe (logged, never consumed): logits on all p^2 pairs. Per-pair margin = logit[true] - max other
(min / mean over test and train). Diagonal part: gbar(d) = mean over (a, b) of the row-centred logit
at c = a+b-d; alpha_w = (2/p) sum_d gbar(d) cos(2 pi w d/p); gbar margin = gbar(0) - max_{d!=0} gbar(d);
diag_frac = share of row-centred logit energy that is a function of d.

Commands (from experiments/, MODAL_PROFILE=chromatic):
    modal run grokking/rung_controls.py::controls --tag r2smoke --groups A1,B1 --smoke 1
    modal run --detach grokking/rung_controls.py::controls --tag r2 --groups A1,B1,B2,B3
    python3 grokking/reduce_rung_controls.py --tag r2 --fetch
"""

import json
import os
import time

import numpy as np

import grokking.rung as RG
from grokking.shared import (DATA_DIR, HIDDEN, LR, P, SEED, WEIGHT_DECAY, GrokMLP, app,
                                  n_params, onehot, peak_rss_mb, split_pairs, torch, train_net,
                                  volume)

K13 = [10, 11, 18, 24, 30, 33, 36, 38, 39, 44, 45, 46, 47]
TOP1 = 30
EVAL_EVERY = 10
PROBE_EVERY = 500


def q_matrix(p=P):
    a = np.arange(p)
    rows = [np.full(p, 1.0 / np.sqrt(p))]
    rows += [np.sqrt(2.0 / p) * np.cos(2 * np.pi * w * a / p) for w in range(1, (p - 1) // 2 + 1)]
    rows += [np.sqrt(2.0 / p) * np.sin(2 * np.pi * w * a / p) for w in range(1, (p - 1) // 2 + 1)]
    return np.stack(rows)                       # (p, p), orthogonal


def feature_map(scale):
    Q = q_matrix()
    s = 1.0 if scale == "q" else np.sqrt(P / 2.0)
    M = np.zeros((2 * P, 2 * P))
    M[:P, :P] = s * Q
    M[P:, P:] = s * Q
    return M                                     # x_feat = M @ x_onehot


def inputs(alph, a, b):
    """Train/test inputs for an alphabet: 'raw', 'all48dc_q', 'all48dc_f', 'top1', 'minted13'."""
    if alph == "raw":
        return onehot(a, b)
    if alph.startswith("all48dc_"):
        M = feature_map(alph.split("_")[1])
        x = onehot(a, b).double().numpy() @ M.T
        return torch.tensor(x, dtype=torch.float32)
    if alph == "top1":
        return RG.features("top1", a, b, [TOP1])
    if alph == "minted13":
        return RG.features("minted", a, b, K13)
    raise ValueError(alph)


def make_probe(x_all, y_all, is_tr, freqs):
    """Probe on all p^2 pairs (a-major order)."""
    a_all = np.repeat(np.arange(P), P)
    b_all = np.tile(np.arange(P), P)
    s_all = (a_all + b_all) % P
    d = np.arange(P)
    cosm = {w: np.cos(2 * np.pi * w * d / P) for w in range(1, 49)}

    def probe(model):
        with torch.no_grad():
            L = model(x_all).double().numpy()
        n = len(L)
        true = L[np.arange(n), y_all]
        Lm = L.copy()
        Lm[np.arange(n), y_all] = -np.inf
        marg = true - Lm.max(1)
        Lc = L - L.mean(1, keepdims=True)
        gbar = np.array([Lc[np.arange(n), (s_all - dd) % P].mean() for dd in range(P)])
        rec = gbar[(s_all[:, None] - np.arange(P)[None, :]) % P]           # gbar at d = a+b-c
        alpha = {w: float(2.0 / P * (gbar * cosm[w]).sum()) for w in range(1, 49)}
        out = {"margin_te_min": float(marg[~is_tr].min()), "margin_te_mean": float(marg[~is_tr].mean()),
               "margin_tr_min": float(marg[is_tr].min()), "margin_tr_mean": float(marg[is_tr].mean()),
               "gbar_margin": float(gbar[0] - gbar[1:].max()), "gbar0": float(gbar[0]),
               "diag_frac": float((rec ** 2).sum() / ((Lc ** 2).sum() + 1e-300)),
               "alpha": {str(w): round(alpha[w], 6) for w in freqs} if freqs else {},
               "alpha_sum_freqs": float(sum(alpha[w] for w in freqs)) if freqs else None,
               "alpha_top5": sorted(((round(v, 5), w) for w, v in alpha.items()), reverse=True)[:5],
               "wnorm": float(sum((q.detach().double() ** 2).sum().item() for q in model.parameters()) ** 0.5)}
        return out
    return probe


def plant_top1(model, amp):
    """Exact one-frequency solution in the bilinear m=4 top1 net (U, V selectors, W readout)."""
    t = float(amp) ** (1.0 / 3.0)
    om = 2 * np.pi * TOP1 * np.arange(P) / P
    U = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float64)     # [ca, sa, ca, sa]
    V = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], np.float64)     # [cb, sb, sb, cb]
    W = np.stack([np.cos(om), -np.cos(om), np.sin(om), np.sin(om)], 1)   # h = [cc, ss, cs, sc]
    with torch.no_grad():
        model.U.weight.copy_(torch.tensor(t * U, dtype=torch.float32))
        model.V.weight.copy_(torch.tensor(t * V, dtype=torch.float32))
        model.W.weight.copy_(torch.tensor(t * W, dtype=torch.float32))
        model.W.bias.zero_()


@app.function(cpu=8.0, memory=2048, timeout=4 * 3600, volumes={DATA_DIR: volume}, max_containers=4)
def ctrl_arm(tag: str, arm: dict):
    t0 = time.time()
    a_tr, b_tr, a_te, b_te = split_pairs(P, 0.3, SEED)
    alph = arm["alphabet"]
    tr_x, te_x = inputs(alph, a_tr, b_tr), inputs(alph, a_te, b_te)
    tr_y, te_y = torch.as_tensor((a_tr + b_tr) % P), torch.as_tensor((a_te + b_te) % P)
    D = tr_x.shape[1]
    torch.manual_seed(SEED)
    info = {}
    if arm["learner"] == "mlp":
        model = GrokMLP(p=P, hidden_dims=HIDDEN, in_dim=D)
        if arm.get("init") == "transported":
            torch.manual_seed(SEED)
            raw = GrokMLP(p=P, hidden_dims=HIDDEN)
            M = feature_map(alph.split("_")[1])
            W0 = raw.layer0.weight.detach().double().numpy() @ np.linalg.pinv(M)
            sd = {k: v.clone() for k, v in raw.state_dict().items()}
            sd["layer0.weight"] = torch.tensor(W0, dtype=torch.float32)
            model.load_state_dict(sd)
            x_raw_all = onehot(np.repeat(np.arange(P), P), np.tile(np.arange(P), P))
            x_all_ = inputs(alph, np.repeat(np.arange(P), P), np.tile(np.arange(P), P))
            with torch.no_grad():
                info["init_maxabs_vs_raw"] = float((model(x_all_) - raw(x_raw_all)).abs().max())
    else:
        model = RG.Bilinear(D // 2, int(arm["m"]), P)
        if arm.get("plant") is not None:
            plant_top1(model, arm["plant"])
    a_all, b_all = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
    x_all = inputs(alph, a_all, b_all)
    y_all = (a_all + b_all) % P
    tr_set = set(zip(a_tr.tolist(), b_tr.tolist()))
    is_tr = np.array([(int(x), int(y)) in tr_set for x, y in zip(a_all, b_all)])
    freqs = {"top1": [TOP1], "minted13": K13}.get(alph, [])
    probe = make_probe(x_all, y_all, is_tr, freqs)
    p0 = dict(probe(model), epoch=-1)
    with torch.no_grad():
        p0["test_acc"] = float((model(te_x).argmax(-1) == te_y).float().mean())
        p0["train_acc"] = float((model(tr_x).argmax(-1) == tr_y).float().mean())
    res = train_net(model, tr_x, tr_y, te_x, te_y, int(arm["cap"]), lr=arm.get("lr", LR),
                    weight_decay=arm.get("wd", WEIGHT_DECAY), eval_every=EVAL_EVERY,
                    stop_at_test=arm.get("stop", 1.0), log_every=10000, tag=f"[{arm['name']}]",
                    optimizer=arm.get("opt", "adamw"), momentum=arm.get("momentum", 0.0),
                    probe=probe, probe_every=arm.get("probe_every", PROBE_EVERY))
    npar = n_params(model)
    out = {"arm": arm, "in_dim": int(D), "params": npar, "first_hit": res["first_hit"],
           "epochs_run": res["epochs_run"], "final_test": res["curve"][-1][2],
           "final_train": res["curve"][-1][1], "final_loss": res["curve"][-1][3],
           "curve": res["curve"][::10] + [res["curve"][-1]], "probes": [p0] + res["probes"],
           "seconds": res["seconds"], "peak_rss_mb": peak_rss_mb(), **info}
    if arm.get("save_state"):
        d = os.path.join(DATA_DIR, tag)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{arm['name']}_final.npz"), **res["final_state"])
        volume.commit()
    fp = res["probes"][-1] if res["probes"] else p0
    print(f"[{arm['name']}] params={npar} e99={res['first_hit']['0.99']} e100={res['first_hit']['1.0']} "
          f"final_te={out['final_test']:.4f} final_tr={out['final_train']:.4f} "
          f"gbar_margin={fp['gbar_margin']:.4g} margin_te_min={fp['margin_te_min']:.4g} "
          f"{time.time() - t0:.0f}s {info}", flush=True)
    return out


def build(groups, cap=40000, smoke=0):
    arms = []
    if "A1" in groups:
        arms += [{"name": "A1_q_transported", "alphabet": "all48dc_q", "learner": "mlp", "init": "transported"},
                 {"name": "A1_f_transported", "alphabet": "all48dc_f", "learner": "mlp", "init": "transported"},
                 {"name": "A1_q_default", "alphabet": "all48dc_q", "learner": "mlp", "init": "default"}]
    if "B1" in groups:
        for amp in (1, 10, 100, 1000, 10000):
            arms.append({"name": f"B1_plant_A{amp}", "alphabet": "top1", "learner": "bilinear", "m": 4,
                         "plant": amp, "stop": None, "cap": 20000})
    if "B2" in groups:
        for wd in (1.0, 0.1, 0.01, 0.0):
            arms.append({"name": f"B2_bil4_wd{wd}", "alphabet": "top1", "learner": "bilinear", "m": 4, "wd": wd})
            arms.append({"name": f"B2_mlp_wd{wd}", "alphabet": "top1", "learner": "mlp", "wd": wd})
    if "B3" in groups:
        arms += [{"name": "B3_minted13_bil_stop", "alphabet": "minted13", "learner": "bilinear", "m": 52,
                  "save_state": True, "probe_every": 250},
                 {"name": "B3_minted13_bil_40k", "alphabet": "minted13", "learner": "bilinear", "m": 52,
                  "stop": None, "probe_every": 1000}]
    for a in arms:
        a.setdefault("cap", cap)
        if smoke:
            a["cap"] = min(a["cap"], 1000)
    return arms


def build_A2(grid, pair, cap):
    """grid: 'lr:wd:momentum,...'. One raw SGD arm per setting; `pair` adds the q-transported twin."""
    arms = []
    for g in [x for x in grid.split(",") if x]:
        lr, wd, mom = (float(v) for v in g.split(":"))
        sfx = f"lr{lr:g}_wd{wd:g}_m{mom:g}"
        base = {"learner": "mlp", "opt": "sgd", "lr": lr, "wd": wd, "momentum": mom, "cap": cap,
                "probe_every": 5000}
        if pair != 2:                     # pair: 0 raw only, 1 raw + twin, 2 twin only (raw banked)
            arms.append(dict(base, name=f"A2_raw_sgd_{sfx}", alphabet="raw"))
        if pair:
            arms.append(dict(base, name=f"A2_q_transported_sgd_{sfx}", alphabet="all48dc_q", init="transported"))
    return arms


@app.function(cpu=1.0, memory=2048, timeout=12 * 3600, volumes={DATA_DIR: volume})
def controls(tag: str = "r2", groups: str = "A1,B1,B2,B3", smoke: int = 0,
             a2_grid: str = "0.1:0.001:0.9", a2_pair: int = 1, a2_cap: int = 100000, out_name: str = ""):
    t0 = time.time()
    g = groups.split(",")
    arms = build(g, smoke=smoke)
    if "A2" in g:
        arms += build_A2(a2_grid, a2_pair, 1000 if smoke else a2_cap)
    arms = sorted(arms, key=lambda a: -a["cap"] * (3 if a["learner"] == "mlp" else 1))   # longest first
    print(f"{len(arms)} arms: {[a['name'] for a in arms]}", flush=True)
    results = list(ctrl_arm.starmap([(tag, a) for a in arms]))
    d = os.path.join(DATA_DIR, tag)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"controls_{out_name or '_'.join(g)}.json")
    with open(path, "w") as f:
        json.dump({"tag": tag, "groups": g, "results": results, "smoke": smoke}, f)
    volume.commit()
    print(f"controls done in {time.time() - t0:.0f}s -> {path}", flush=True)
    return path
