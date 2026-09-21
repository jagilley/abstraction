"""regime/world.py -- a bursty corruption world, its i.i.d. control, and the exact
regime filter that says how much a token moved the belief about the hidden state.

THE WORLD. The grammar's flat windows with a two-state hidden regime running along the
window, a Markov chain with mean dwell `dwell_c` in CLEAN and `dwell_n` in NOISY,
initialised from its stationary distribution. In regime r every token is independently
replaced by a uniform draw with probability `eps_r` -- exactly `altitude/train_noisy.py`'s
corruption, with the rate switched by the regime instead of fixed. So an illegal token is
evidence the stream is currently noisy, and the noisy regime predicts more corrupted
tokens over the next few positions, which cost a downstream actor at its goal.

  burst  dwell_c = 288, dwell_n = 32 (p_noisy = 0.10), eps_c = 0.002, eps_n = 0.12
  iid    the same MARGINAL corruption rate, eps = 0.10*0.12 + 0.90*0.002 = 0.0138,
         with no regime: a violation carries no information about the future.

Window-level and stream-level regimes are the same distribution (the chain is started
stationary), so the training stream may be corrupted in one pass (`regime_path`) while
the evaluation windows carry their own per-window path (`regime_windows`); a window drawn
at a random offset of a stationary stream and a window with a stationary-initialised path
are identically distributed.

THE FILTER (`regime_filter`). The exact forward filter over the two regimes whose
emission model is the eps-observer predictive at each regime's rate,
`flat_oracle.flat_predictive(..., noise_eps=eps_r)` at the model's own altitude k, read at
the observed token. `basalis/hold.py` computed a one-shot log Bayes factor for a single
glitch; this is its running form. Per position it returns

  b_pre[t]   P(R_t = noisy | x_<t)            the belief before the token arrives
  b_fwd[t]   P(R_{t+1} = noisy | x_<=t)       the FORWARD-LOOKING belief, what predicts cost
  db[t]      b_fwd[t] - b_fwd[t-1]            the probability revision the token caused
  dlogbf[t]  logit(b_fwd[t]) - logit(b_fwd[t-1])   the same revision in evidence currency
  llr[t]     log p^{eps_n}(x_t|x_<t) - log p^{eps_c}(x_t|x_<t)   the token's own evidence

`db` and `dlogbf` come apart exactly where the emotion story's "weighted by expectation"
clause lives: deep inside a stretch the reader already believes is noisy, a further
violation carries the same `llr` and revises `b_fwd` hardly at all.

Run:
  modal run -m rhm.logit_reading.orbitofrontal.regime.world::world_build \
      --world burst --n 256 --tag smoke
  modal run --detach -m rhm.logit_reading.orbitofrontal.regime.world::world_build_sweep
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import tb_key


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree --
    another agent writing its launch log there makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-striatum-regime", image=image)

# the one setting of the world this node runs first
WORLDS = {
    "burst": dict(dwell_c=288.0, dwell_n=32.0, eps_c=0.002, eps_n=0.12),
    "iid":   dict(dwell_c=1.0, dwell_n=0.0, eps_c=0.0138, eps_n=0.0138),
}
OBS_K = 5          # the model's own altitude (kappa* = 4.85-4.90 on this grammar)
READER = "burst"   # the filter is ALWAYS the burst world's reader, so that the same
                   # statistic is defined on the control world, where it predicts nothing


def world_params(world: str):
    p = dict(WORLDS[world])
    if world == "iid":
        p["p_noisy"] = 0.0
        p["eps_mean"] = p["eps_c"]
    else:
        p["p_noisy"] = p["dwell_n"] / (p["dwell_c"] + p["dwell_n"])
        p["eps_mean"] = p["p_noisy"] * p["eps_n"] + (1 - p["p_noisy"]) * p["eps_c"]
    return p


# ---------------------------------------------------------------------------
# the regime process
# ---------------------------------------------------------------------------

def regime_path(n, dwell_c, dwell_n, rng):
    """A stationary two-state path of length `n` (0 = clean, 1 = noisy), built from
    geometric dwell times.  Used on the TRAINING stream, where a per-token loop would be
    wasteful."""
    if dwell_n <= 0:
        return np.zeros(n, np.uint8)
    p_n = dwell_n / (dwell_c + dwell_n)
    start = int(rng.random() < p_n)
    segs, tot = [], 0
    while tot < n:
        k = max(1024, int(2.0 * (n - tot) / (dwell_c + dwell_n)) + 8)
        rc = rng.geometric(1.0 / dwell_c, size=k)
        rn = rng.geometric(1.0 / dwell_n, size=k)
        runs = (np.stack([rc, rn], 1) if start == 0 else np.stack([rn, rc], 1)).reshape(-1)
        segs.append(runs)
        tot += int(runs.sum())
    runs = np.concatenate(segs)
    st = np.empty(len(runs), np.uint8)
    st[0::2] = start
    st[1::2] = 1 - start
    return np.repeat(st, runs)[:n]


def regime_windows(n, T1, dwell_c, dwell_n, rng):
    """(n, T1) independent stationary paths, one per window."""
    r = np.zeros((n, T1), np.uint8)
    if dwell_n <= 0:
        return r
    p_n = dwell_n / (dwell_c + dwell_n)
    cur = (rng.random(n) < p_n).astype(np.uint8)
    r[:, 0] = cur
    pc, pn = 1.0 / dwell_c, 1.0 / dwell_n
    for t in range(1, T1):
        u = rng.random(n)
        sw = np.where(cur == 0, u < pc, u < pn)
        cur = np.where(sw, 1 - cur, cur).astype(np.uint8)
        r[:, t] = cur
    return r


def corrupt(tokens, regime, eps_c, eps_n, v, rng):
    """Uniform-redraw corruption at the regime's rate.  Returns (tokens, fired, changed)."""
    eps = np.where(regime == 1, eps_n, eps_c)
    fired = rng.random(tokens.shape) < eps
    draw = rng.integers(0, v, tokens.shape)
    out = np.where(fired, draw, tokens)
    return out.astype(tokens.dtype), fired, out != tokens


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------

def regime_filter(logL_c, logL_n, dwell_c, dwell_n, t_lo=0):
    """Forward filter over {clean, noisy}.

    logL_* : (n, T) log emission of the observed token at each position under each regime.
    Returns a dict of (n, T) arrays; position 0 uses the stationary prior.
    """
    n, T = logL_c.shape
    if dwell_n <= 0:                       # the control world has one regime
        z = np.zeros((n, T), np.float64)
        return {"b_pre": z, "b_fwd": z.copy(), "db": z.copy(), "dlogbf": z.copy(),
                "logbf": z.copy(), "llr": (logL_n - logL_c).astype(np.float64)}
    p_n = dwell_n / (dwell_c + dwell_n)
    A = np.array([[1 - 1.0 / dwell_c, 1.0 / dwell_c],
                  [1.0 / dwell_n, 1 - 1.0 / dwell_n]])            # A[r, r']
    logA = np.log(A)
    b_pre = np.zeros((n, T)); b_fwd = np.zeros((n, T)); logbf = np.zeros((n, T))
    lp = np.stack([np.full(n, np.log(1 - p_n)), np.full(n, np.log(p_n))], 1)   # prior at t=0
    for t in range(T):
        b_pre[:, t] = np.exp(lp[:, 1] - np.logaddexp(lp[:, 0], lp[:, 1]))
        post = lp + np.stack([logL_c[:, t], logL_n[:, t]], 1)
        post = post - np.logaddexp(post[:, 0], post[:, 1])[:, None]
        logbf[:, t] = post[:, 1] - post[:, 0]
        lp = np.stack([np.logaddexp(post[:, 0] + logA[0, 0], post[:, 1] + logA[1, 0]),
                       np.logaddexp(post[:, 0] + logA[0, 1], post[:, 1] + logA[1, 1])], 1)
        b_fwd[:, t] = np.exp(lp[:, 1] - np.logaddexp(lp[:, 0], lp[:, 1]))
    lo = np.log(np.clip(b_fwd, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - b_fwd, 1e-12, 1))
    db = np.zeros_like(b_fwd); dlogbf = np.zeros_like(b_fwd)
    db[:, 1:] = b_fwd[:, 1:] - b_fwd[:, :-1]
    dlogbf[:, 1:] = lo[:, 1:] - lo[:, :-1]
    db[:, 0] = b_fwd[:, 0] - p_n
    dlogbf[:, 0] = lo[:, 0] - (np.log(p_n) - np.log(1 - p_n))
    return {"b_pre": b_pre, "b_fwd": b_fwd, "db": db, "dlogbf": dlogbf,
            "logbf": logbf, "llr": (logL_n - logL_c).astype(np.float64)}


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=5 * 3600, memory=24576,
              max_containers=4)
def world_build(world: str = "burst", v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                rule_seed: int = 0, alpha: float = 1.0, weight_seed: int = 1,
                n: int = 16384, seed: int = 3131, tag: str = "", chunk6: int = 8,
                chunk5: int = 48, skip_pl: bool = False):
    """One evaluation venue: n windows of the world, their clean counterparts, the level
    answers of the (uncorrupted) generating tree, the exact legal support, and the filter."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    from rhm.logit_reading.flat_oracle import flat_predictive
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    volume.reload()
    t00 = time.time()
    P = world_params(world)
    L, T = depth, s ** depth
    T1 = T + 1
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)

    wins_c, phase, _, fs = windows_with_parse(rules, rule_w, n, T1, seed)
    y = clean_answers(fs, phase, T1, L, s, 3)                       # (6, n, T1) int16
    rng = np.random.default_rng(seed + 7)
    reg = regime_windows(n, T1, P["dwell_c"], P["dwell_n"], rng)
    wins_x, fired, changed = corrupt(wins_c, reg, P["eps_c"], P["eps_n"], v, rng)
    print(f"{world}: n={n} p_noisy(real)={reg.mean():.4f} fired={fired.mean():.5f} "
          f"changed={changed.mean():.5f} (target eps_mean {P['eps_mean']:.5f})", flush=True)

    # the exact legal support under the true DGP (k = L), on the CORRUPTED windows
    t0 = time.time()
    if skip_pl:
        legal = np.ones((n, T1, v), np.uint8)
        ptok_L = np.ones((n, T1), np.float32)
    else:
        pL = flat_predictive(wins_x, rules, L, device="cuda", chunk=chunk6,
                             rule_w=rule_w).numpy()
        pL = np.nan_to_num(pL, nan=0.0)
        legal = (pL > 0).astype(np.uint8)
        ptok_L = np.take_along_axis(pL, wins_x[..., None], -1)[..., 0].astype(np.float32)
        del pL
    print(f"  p_L (k={L}) {time.time() - t0:.0f}s  legal frac at corrupt positions "
          f"{float((ptok_L[changed] > 0).mean()):.3f}", flush=True)

    # the filter's emissions: the eps-observer at each regime's rate, at the model's
    # altitude.  The READER is the burst world's process in BOTH worlds, so `db`, `llr`
    # and `b_fwd` are the same statistic on the control venue -- where, by construction,
    # they predict nothing about the future.
    Q = world_params(READER)
    lp = {}
    for nm, eps in (("c", Q["eps_c"]), ("n", Q["eps_n"])):
        t0 = time.time()
        pr = flat_predictive(wins_x, rules, OBS_K, device="cuda", chunk=chunk5,
                             rule_w=rule_w, noise_eps=eps).numpy()
        lp[nm] = np.log(np.clip(np.take_along_axis(pr, wins_x[..., None], -1)[..., 0],
                                1e-300, None))
        print(f"  observer k={OBS_K} eps={eps} {time.time() - t0:.0f}s", flush=True)
        del pr
    F = regime_filter(lp["c"], lp["n"], Q["dwell_c"], Q["dwell_n"])

    # sanity: does the filter track the truth?
    m_ev = changed & (np.arange(T1)[None, :] >= 8)
    summary = {
        "world": world, "params": P, "n": n, "T1": T1, "obs_k": OBS_K, "seed": seed,
        "p_noisy_realised": float(reg.mean()),
        "fire_rate": float(fired.mean()), "change_rate": float(changed.mean()),
        "change_rate_by_regime": [float(changed[reg == 0].mean()),
                                  float(changed[reg == 1].mean()) if (reg == 1).any() else None],
        "illegal_rate_at_change": float((ptok_L[changed] == 0).mean()),
        "illegal_rate_overall": float((ptok_L[:, 1:] == 0).mean()),
        "events_per_window": float(m_ev.sum() / n),
        "reader": Q,
        "filter_auc_regime": _auc(F["b_fwd"][:, 8:-1].ravel(), reg[:, 9:].ravel() == 1),
        "filter_auc_next_corrupt": _auc(F["b_fwd"][:, 8:-1].ravel(),
                                        changed[:, 9:].ravel()),
        "filter_auc_ahead2": _auc(F["b_fwd"][:, 8:T1 - 13].ravel(),
                                  np.stack([changed[:, 9 + k:T1 - 12 + k]
                                            for k in range(12)], -1).sum(-1).ravel() >= 2),
        "b_fwd_mean_by_regime": [float(F["b_fwd"][:, 8:-1][reg[:, 9:] == 0].mean()),
                                 float(F["b_fwd"][:, 8:-1][reg[:, 9:] == 1].mean())
                                 if (reg[:, 9:] == 1).any() else None],
        "llr_mean_illegal": float(F["llr"][changed & (ptok_L == 0)].mean()),
        "llr_mean_legal_tok": float(F["llr"][(~changed) & (ptok_L > 0)].mean()),
        "db_mean_at_event": float(F["db"][m_ev].mean()),
        "dlogbf_mean_at_event": float(F["dlogbf"][m_ev].mean()),
    }
    print(json.dumps(summary, indent=1, cls=NumpyEncoder), flush=True)

    key = tb_key(v, s, L, m)
    out_dir = f"{DATA_DIR}/{key}/logit_reading"
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{out_dir}/regime_world_{world}" + (f"_{tag}" if tag else "")
    np.savez_compressed(
        f"{stem}.npz", windows_clean=wins_c.astype(np.uint8),
        windows=wins_x.astype(np.uint8), phase=phase.astype(np.int32),
        regime=reg, fired=fired.astype(np.uint8), changed=changed.astype(np.uint8),
        y=y.astype(np.int16), legal=legal, ptok_L=ptok_L,
        logL_c=lp["c"].astype(np.float32), logL_n=lp["n"].astype(np.float32),
        **{k: vv.astype(np.float32) for k, vv in F.items()})
    with open(f"{stem}.json", "w") as f:
        json.dump(summary, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}.npz  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)", flush=True)
    return summary


def _auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(score, np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def world_build_sweep(worlds: str = "burst,iid", n: int = 16384, seed: int = 3131,
                      tag: str = ""):
    volume.reload()
    args = [(w, 16, 2, 6, 4, 0, 1.0, 1, n, seed, tag) for w in worlds.split(",")]
    outs = list(world_build.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str, cls=NumpyEncoder)[:600], flush=True)
    return [str(o)[:400] for o in outs]
