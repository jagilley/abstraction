"""The endogenous gauge: recover the grid from the reader's own entropy period, then
recompute `at_support` on it.

For every cycle of the recorded practice run and every piece that cycle's miner read:

  1. the piece is placed at slot 1 of a 4-piece stream built from that cycle's OWN solved
     beam tips, and a window of W+1 tokens is cut at a uniformly random offset, so the
     piece sits wholly inside the window at a phase the reader is not told. The window set
     is drawn ONCE from a fixed seed and is identical for every checkpoint and every arm:
     the only thing that varies across the panel is the reader.
  2. the reader's per-position entropy profile is detrended across windows and handed to
     `frontier.common.nested_phase` in template-free `meanprof` mode, bottom-up over
     k = 1..6. `r_hat_6 + 1` is where the detector thinks the piece begins.
  3. `at_support` is recomputed cumulatively over cycles on three grids -- the TRUE piece
     origin, the RECOVERED one, and a SHUFFLED one (a uniform draw over the 64 offsets: a
     wrong grid at the right period) -- at every level 2..6.

Nothing endogenous reads the oracle grid: the true origin enters only as the reference the
recovered one is scored against, and the detector is model-free.

Run (from experiments/):
  modal run -m rhm.practice.perception.evaluate::evaluate --tag pc0 --arm corpus
"""

import glob
import json
import os
import time

import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, volume
from rhm.practice.perception.common import (
    DEPTH, M, REMOTE, RULE_SEED, S, T_SEQ, V, TokenMiner, app, extract,
    meanprof_phase, piece_start_hat, span_leaves, unconstrained_acc)

LEVELS = (2, 3, 4, 5, 6)
K_STREAM = 4                 # pieces per constructed stream; the target sits at slot 1


# --------------------------------------------------------------------------- #
# the windows
# --------------------------------------------------------------------------- #

def build_windows(npz, W, seed=20260917):
    """One window per mined piece, at a uniformly random offset inside a stream of that
    cycle's own solved productions. Returns windows (n, W+1) uint8, phase (n,), cyc (n,),
    and the per-piece true origin in window-token coordinates."""
    rng = np.random.default_rng(seed)
    mine_x, mine_cyc = npz["mine_x"], npz["mine_cyc"]
    sol_x, sol_cyc = npz["solved_x"], npz["solved_cyc"]
    pool = {}
    for c in np.unique(sol_cyc):
        pool[int(c)] = sol_x[sol_cyc == c]
    wins, phases, cycs = [], [], []
    for i in range(mine_x.shape[0]):
        c = int(mine_cyc[i])
        p = pool.get(c)
        if p is None or p.shape[0] == 0:
            p = mine_x[mine_cyc == c]
        fill = p[rng.integers(0, p.shape[0], K_STREAM)]
        stream = fill.copy()
        stream[1] = mine_x[i]
        stream = stream.reshape(-1)
        o = int(rng.integers(1, T_SEQ + 1))          # target piece start = T_SEQ - o in [0, 63]
        wins.append(stream[o:o + W + 1])
        phases.append(o % T_SEQ)
        cycs.append(c)
    return (np.stack(wins).astype(np.uint8), np.asarray(phases, np.int64),
            np.asarray(cycs, np.int64))


def build_windows_mine(npz, W, seed=20260917):
    """The same construction, but every piece in the stream is one the miner READ, so the
    recorded level-1 parse (`mine_pf`, on the true block grid) exists for all of them and a
    FEATURE stream can be carried beside the token stream.

    The detector recovers the block grid at k = 1; whenever it is right, the blocks of the
    recovered grid ARE the true blocks, so the parse at the recovered grid is exactly the
    recorded one, shifted. Where the k = 1 parity is wrong the span straddles blocks and no
    feature representation of it exists; those observations are counted and dropped."""
    rng = np.random.default_rng(seed)
    mine_x, mine_pf, mine_cyc = npz["mine_x"], npz["mine_pf"], npz["mine_cyc"]
    pool = {}
    for c in np.unique(mine_cyc):
        pool[int(c)] = np.where(mine_cyc == c)[0]
    wins, feats, phases, cycs, offs = [], [], [], [], []
    for i in range(mine_x.shape[0]):
        c = int(mine_cyc[i])
        cand = pool[c]
        sel = cand[rng.integers(0, cand.shape[0], K_STREAM)]
        tok = mine_x[sel].copy()
        pf = mine_pf[sel].copy()
        tok[1], pf[1] = mine_x[i], mine_pf[i]
        o = int(rng.integers(1, T_SEQ + 1))
        wins.append(tok.reshape(-1)[o:o + W + 1])
        feats.append(pf.reshape(-1))
        phases.append(o % T_SEQ)
        cycs.append(c)
        offs.append(o)
    return (np.stack(wins).astype(np.uint8), np.stack(feats).astype(np.int64),
            np.asarray(phases, np.int64), np.asarray(cycs, np.int64),
            np.asarray(offs, np.int64))


def feature_at_support_series(feat_stream, offs, cycs, starts, eras_by_cycle, n_cycles,
                              support, levels=(3, 4, 5, 6), s=S):
    """`at_support` in the THERMOSTAT'S OWN coordinates -- level-1 feature tuples -- on
    whatever piece origin `starts` gives. Returns the series and the number of pieces
    dropped for block-parity failure."""
    miners = {ell: TokenMiner(support) for ell in levels}
    out = {ell: [] for ell in levels}
    by_cycle = {}
    for i in range(len(cycs)):
        by_cycle.setdefault(int(cycs[i]), []).append(i)
    n_drop = 0
    for c in range(1, n_cycles + 1):
        era = eras_by_cycle[c]
        for i in by_cycle.get(c, []):
            abs_tok = int(offs[i]) + int(starts[i])
            if abs_tok % s:
                n_drop += 1
                continue
            b_f = abs_tok // s
            for ell in levels:
                lo, hi = span_leaves(era["level"], era["node"], ell)
                miners[ell].observe(feat_stream[i][b_f + lo // s:b_f + hi // s][None, :])
        for ell in levels:
            out[ell].append(miners[ell].at_support())
    return {ell: out[ell] for ell in levels}, n_drop


def corpus_windows(rules, n, W, seed=4242):
    """The altitude venue: held-out windows of the practice grammar at random offsets."""
    from rhm.logit_reading.grammar import generate
    rng = np.random.default_rng(seed)
    n_seq = int(np.ceil((n * (W + 1)) / T_SEQ)) + K_STREAM + 2
    stream = generate(rules, None, n_seq, rng).reshape(-1)
    o = rng.integers(0, stream.shape[0] - W - 1, n)
    wins = stream[o[:, None] + np.arange(W + 1)[None, :]].astype(np.uint8)
    return wins, (o % T_SEQ).astype(np.int64)


# --------------------------------------------------------------------------- #
# the reader's profile
# --------------------------------------------------------------------------- #

def entropy_profile(model, windows, device, batch=256):
    import torch
    H, NLL = [], []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            w = torch.as_tensor(windows[i:i + batch].astype(np.int64), device=device)
            x, y = w[:, :-1], w[:, 1:]
            logits, _ = model(x)
            logq = torch.log_softmax(logits.float(), -1)
            q = logq.exp()
            H.append((-(q * logq).sum(-1)).cpu().numpy())
            NLL.append((-logq.gather(-1, y[..., None])[..., 0]).cpu().numpy())
    return np.concatenate(H), np.concatenate(NLL)


def detector(Hq, phase, L=DEPTH, s=S):
    from rhm.logit_reading.frontier.common import detrend, true_boundary_column
    Xd = detrend(Hq)
    rh = meanprof_phase(Xd, L=L, s=s)
    acc = {k: float((rh[k] == true_boundary_column(phase, s ** k)).mean())
           for k in range(1, L + 1)}
    return rh, {"acc_nested": acc,
                "acc_unconstrained": unconstrained_acc(Xd, phase, L, s),
                "chance": {k: 1.0 / s ** k for k in range(1, L + 1)}}


# --------------------------------------------------------------------------- #
# the gauge
# --------------------------------------------------------------------------- #

def at_support_series(windows, cycs, starts, eras_by_cycle, n_cycles, support, levels=LEVELS):
    """Cumulative `at_support` per level per cycle, on whatever origin `starts` gives."""
    miners = {ell: TokenMiner(support) for ell in levels}
    out = {ell: [] for ell in levels}
    order = np.argsort(cycs, kind="stable")
    by_cycle = {}
    for i in order:
        by_cycle.setdefault(int(cycs[i]), []).append(i)
    for c in range(1, n_cycles + 1):
        idx = by_cycle.get(c, [])
        era = eras_by_cycle[c]
        if idx:
            for ell in levels:
                lo, hi = span_leaves(era["level"], era["node"], ell)
                miners[ell].observe(extract(windows[idx], starts[idx], lo, hi))
        for ell in levels:
            out[ell].append(miners[ell].at_support())
    return {ell: out[ell] for ell in levels}, {ell: miners[ell].n_distinct() for ell in levels}


def grid_quality(phase, starts_hat, cycs, eras_by_cycle, levels=LEVELS, s=S, depth=DEPTH):
    """Per level: how often the recovered origin puts the miner on the RIGHT span, on a
    genuine constituent of the right level at the WRONG node, or off the grid entirely."""
    b_true = (-phase) % (s ** depth)
    out = {}
    for ell in levels:
        P = s ** ell
        exact = (starts_hat == b_true)
        aligned = ((starts_hat - b_true) % P == 0)
        out[ell] = {"exact": float(exact.mean()),
                    "right_level_wrong_node": float((aligned & ~exact).mean()),
                    "off_grid": float((~aligned).mean())}
    # the endo read's positions: the parent span of the era's own cell, per era
    per_era = {}
    for c, era in eras_by_cycle.items():
        m = cycs == c
        if not m.any():
            continue
        e = per_era.setdefault(int(era["era"]), {"n": 0, "exact": 0, "aligned": 0,
                                                 "parent_level": int(era["level"]) + 1})
        P = s ** (int(era["level"]) + 1)
        e["n"] += int(m.sum())
        e["exact"] += int((starts_hat[m] == b_true[m]).sum())
        e["aligned"] += int(((starts_hat[m] - b_true[m]) % P == 0).sum())
    for e in per_era.values():
        e["frac_exact"] = e["exact"] / max(e["n"], 1)
        e["frac_aligned"] = e["aligned"] / max(e["n"], 1)
    return out, per_era


# --------------------------------------------------------------------------- #
# the entrypoint
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=8192)
def evaluate(tag: str = "pc0", arm: str = "corpus", cd_tag: str = "",
             cd_arm: str = "outer_yield", n_corpus: int = 2048, window_seed: int = 20260917,
             shuffle_seed: int = 77, limit_ckpt: int = 0, out_name: str = "",
             stream_from: str = "solved"):
    import resource
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    volume.reload()
    device = "cuda"
    cd_tag = cd_tag or f"perception_{tag}"
    root = f"{DATA_DIR}/{REMOTE}/{tag}"
    res = json.load(open(f"{DATA_DIR}/rhm_practice_conductor/{cd_tag}/{cd_arm}/results.json"))
    support = int(res["config"]["mine_support"])
    n_cycles = len(res["log"]["cycle"])
    eras_by_cycle = {}
    for i, c in enumerate(res["log"]["cycle"]):
        ei = res["log"]["era"][i] - 1
        eras_by_cycle[int(c)] = {"era": res["log"]["era"][i], "level": res["eras"][ei]["level"],
                                 "node": res["eras"][ei]["node"]}
    z = np.load(f"{root}/pieces.npz")
    paths = sorted(glob.glob(f"{root}/reader_{arm}/step*.pt"))
    if limit_ckpt:
        paths = paths[:limit_ckpt]
    assert paths, f"no checkpoints in {root}/reader_{arm}"
    W = int(torch.load(paths[0], map_location="cpu")["config"]["window"])

    feat_stream = offs = None
    if stream_from == "mine":
        wins, feat_stream, phase, cycs, offs = build_windows_mine(z, W, seed=window_seed)
    else:
        wins, phase, cycs = build_windows(z, W, seed=window_seed)
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    cwins, cphase = corpus_windows(rules, n_corpus, W)
    b_true = (-phase) % T_SEQ
    rng = np.random.default_rng(shuffle_seed)
    b_shuf = rng.integers(0, T_SEQ, phase.shape[0])
    print(f"[eval] {wins.shape[0]} practice windows over {n_cycles} cycles, W={W}; "
          f"{cwins.shape[0]} corpus windows; {len(paths)} checkpoints", flush=True)

    # the two grid-independent references
    out = {"tag": tag, "arm": arm, "cd_tag": cd_tag, "cd_arm": cd_arm, "W": int(W),
           "support": support, "n_cycles": n_cycles, "n_windows": int(wins.shape[0]),
           "n_corpus_windows": int(cwins.shape[0]), "window_seed": window_seed,
           "stream_from": stream_from,
           "levels": list(LEVELS), "eras": res["eras"],
           "read_level": res["log"]["panel"] and [p["read_level"] for p in res["log"]["panel"]],
           "era_by_cycle": [res["log"]["era"][i] for i in range(n_cycles)],
           "exact_feature_obs_hist": {k: v for k, v in (res.get("obs_hist") or {}).items()},
           "loop_actions": res.get("loop_actions"), "loop": res.get("loop"),
           "floors": {"yield_L3": res["config"]["tol_yield_l3"],
                      "yield_L4": res["config"]["tol_yield_l4"]},
           "loop_cfg": {k: res["config"][k] for k in ("loop_span", "loop_W", "loop_burn",
                                                      "loop_alpha")},
           "era_caps": res.get("era_caps"), "checkpoints": {}}
    # gate P-1: the conductor's OWN gauge, recomputed here from the recorded level-1
    # feature parse on the true grid, must equal the run's `obs_hist` cycle for cycle.
    fmine = {ell: TokenMiner(support) for ell in LEVELS}
    fser = {ell: [] for ell in LEVELS}
    by_c = {}
    for i, c in enumerate(z["mine_cyc"]):
        by_c.setdefault(int(c), []).append(i)
    for c in range(1, n_cycles + 1):
        idx = by_c.get(c, [])
        era = eras_by_cycle[c]
        for ell in LEVELS:
            if idx:
                lo, hi = span_leaves(era["level"], era["node"], ell)
                fmine[ell].observe(z["mine_pf"][idx][:, lo // S:hi // S])
            fser[ell].append(fmine[ell].at_support())
    out["feature_true_grid"] = {str(k): v for k, v in fser.items()}

    ser_true, nd_true = at_support_series(wins, cycs, b_true, eras_by_cycle, n_cycles, support)
    ser_shuf, nd_shuf = at_support_series(wins, cycs, b_shuf, eras_by_cycle, n_cycles, support)
    out["token_true_grid"] = {str(k): v for k, v in ser_true.items()}
    out["token_true_n_distinct"] = {str(k): v for k, v in nd_true.items()}
    out["token_shuffled_grid"] = {str(k): v for k, v in ser_shuf.items()}
    out["token_shuffled_n_distinct"] = {str(k): v for k, v in nd_shuf.items()}
    qs, _ = grid_quality(phase, b_shuf, cycs, eras_by_cycle)
    out["shuffled_grid_quality"] = {str(k): v for k, v in qs.items()}
    if feat_stream is not None:
        fs, nd_ = feature_at_support_series(feat_stream, offs, cycs, b_true, eras_by_cycle,
                                            n_cycles, support)
        out["feature_true_grid_windowed"] = {str(k): v for k, v in fs.items()}
        out["feature_true_grid_windowed_dropped"] = nd_
        fs, nd_ = feature_at_support_series(feat_stream, offs, cycs, b_shuf, eras_by_cycle,
                                            n_cycles, support)
        out["feature_shuffled_grid"] = {str(k): v for k, v in fs.items()}
        out["feature_shuffled_grid_dropped"] = nd_

    for pth in paths:
        t0 = time.time()
        ck = torch.load(pth, map_location="cpu")
        step = int(ck["config"]["step"])
        model = GPT(V, W, ck["config"]["n_layer"], ck["config"]["n_head"],
                    ck["config"]["n_embd"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        Hq, NLL = entropy_profile(model, wins, device)
        cHq, cNLL = entropy_profile(model, cwins, device)
        rh, det = detector(Hq, phase)
        _, cdet = detector(cHq, cphase)
        b_hat = piece_start_hat(rh[DEPTH])
        ser, nd = at_support_series(wins, cycs, b_hat, eras_by_cycle, n_cycles, support)
        gq, per_era = grid_quality(phase, b_hat, cycs, eras_by_cycle)
        out["checkpoints"][str(step)] = {
            "step": step, "ckpt": os.path.basename(pth),
            "train_loss": ck.get("recent_train_loss"),
            "nll_practice": float(NLL.mean()), "nll_corpus": float(cNLL.mean()),
            "H_practice": float(Hq.mean()), "H_corpus": float(cHq.mean()),
            "detector_practice": det, "detector_corpus": cdet,
            "grid_quality": {str(k): v for k, v in gq.items()},
            "grid_quality_by_era_parent_span": {str(k): v for k, v in per_era.items()},
            "token_endo_grid": {str(k): v for k, v in ser.items()},
            "token_endo_n_distinct": {str(k): v for k, v in nd.items()},
            "eval_s": time.time() - t0}
        if feat_stream is not None:
            fs, ndrop = feature_at_support_series(feat_stream, offs, cycs, b_hat,
                                                  eras_by_cycle, n_cycles, support)
            out["checkpoints"][str(step)]["feature_endo_grid"] = {
                str(k): v for k, v in fs.items()}
            out["checkpoints"][str(step)]["feature_endo_dropped"] = ndrop
        del model
        torch.cuda.empty_cache()
        print(f"  step {step:>6} ({time.time() - t0:.0f}s) nll {NLL.mean():.4f} "
              f"acc_k " + " ".join(f"{det['acc_nested'][k]:.2f}" for k in range(1, 7))
              + f" | corpus " + " ".join(f"{cdet['acc_nested'][k]:.2f}" for k in range(1, 7))
              + f" | exact@L4 {gq[4]['exact']:.2f}", flush=True)

    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dst = f"{root}/{out_name or ('panel_' + arm)}.json"
    with open(dst, "w") as f:
        json.dump(out, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"saved -> {dst}", flush=True)
    return dst
