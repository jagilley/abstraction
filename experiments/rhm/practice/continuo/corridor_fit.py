"""corridor_fit — E2 PASS 2, the offline instrument. No GPU, no Modal, no torch.

Reads `corridor.py`'s per-cycle span-head snapshots and asks pass 1's question of the EXECUTOR:
can a sub-saturation forward self-model's residual read which of the learner's macro slots are
COMMANDS (gate open — the head materialises the span in one pass, no DP, no table) and which are
still DATA (gate closed — `macros.apply_any` masks the span, reads per-block infill evidence and
runs the max-sum DP over the committed table)?

WHAT IS DIFFERENT FROM PASS 1, AND WHY IT MATTERS FOR THE CONTROL THAT KILLED IT. Pass 1's
label was `ms_level[j] >= 2` — a slot is a command iff it is a macro — which is very nearly the
same variable as "this token is young", and the +0.052 headline decayed to zero under a 20-cycle
token-age floor with the scalar-norm negative moving in lockstep. Here the label is the PARITY
GATE, which

  * varies WITHIN a level (at `ma_s0/mperf_log` c60, 10 of 16 L2 slots are open and 6 are not),
  * varies WITHIN a slot over time (44 gate events over 153 cycles, re-closures included),
  * and is not the slot's age: a slot minted at the L2 commit may open at c49 or at c71.

So "command" and "new" are separable in this donor by construction, and the age floors (0 / 20 /
50 cycles since MINT, plus a second clock: cycles since the slot first OPENED) are applied from
the first cell rather than discovered afterwards.

THE LADDER OF LABELS, all read off the run's own record:
    open       the gate the executor actually consults (`exact >= span_tau_fire`, 0.50 here)
    open_tau   the donor's own criterion (`exact >= span_tau`, 0.95) — a stricter grade of the
               same state, and the balanced one late in the run where `open` saturates
    level      L2 / L3 / L4 — pass 1's age disambiguator, carried as a three-way read

SUB-SATURATION. Paper 1's law and `two_deltas` finding 3's bound for this substrate: the FM must
stay under ~6% of the PREDICTED SPAN's parameters. The predicted span is `SpanHead` — with
`state_dim` 96, v 8, `max_span` 8, 28 slots and `hidden_mult` 4 that is 133,736 parameters, and
the % is COMPUTED from the snapshot's own shapes here, never asserted. A 100%-rung is swept
deliberately as this substrate's saturation demonstration, flagged, and excluded from every
headline.

THE FM, AND ITS ONE HONEST APPROXIMATION.

    a_i = (the head's own read of the state, the slot identity)
    a_j = the head's block-0 logits over the v level-1 features

`SpanHead.span_state` is `ctx(pooled.mean(1)) + slot(sid) + sum_j in_proj[j](pooled[blk0+j])`.
The snapshot stores `pooled.mean(1)`, the span's mean and the span's first block — the head's
global read exactly, and its span read up to the per-offset weighting. The FM therefore cannot
represent the per-offset part, which means a piece of the residual is input the FM never saw
rather than structure it failed to theorise. That is a cost of keeping one fixed 3x96 input
dimension for every slot at every level; the alternative (the full padded per-offset read, 8x96)
puts a dimensional signature of the slot's LEVEL into the FM's input, which is the exact
confound this pass exists to control. The call is stated, not hidden, and the ridge rung —
convex, unique minimiser, so its residual cannot be instrument noise — is carried at every cell
so no headline depends on the MLP's optimisation.

    self   R      the residual column          (the un-theorised part)
    ctrl   FM     the self-theory's column     (paper 2 section 6's PRED)
    obs    LOGIT  the raw block-0 logit column (paper 2 section 6's AJ; the strongest observer)
    obs    IO     the softmax column           (paper 2's O_io)
    obs    BEH    the EMITTED feature          (strictly public behaviour: what the executor
                                                actually wrote into the sequence)
    ctrl   TRUNK  the same FM class fitted to the TRUNK's own per-block feature logits, which
                  carry no slot conditioning at all — pass 1 design call 1b's value-head
                  control, ported. Anything that reorganises at a gate event on both is the
                  state distribution or the calendar, not the command port.
    neg    NORM   ||R_j|| alone
    neg    SHUF   R with the probe axis shuffled per slot
    null   PERM   slot labels permuted, refit

TWO DECODES, because the label has two axes:
  A (per checkpoint, rows = slots)      pass 1's exact shape, giving a trajectory. Reported only
                                        on checkpoints where both classes have >= `--min-class`
                                        members, and that count is printed.
  B (pooled, rows = (slot, checkpoint))  leave-one-SLOT-out folds, so slot identity cannot leak.
                                        This is where the age floors bite and where the headline
                                        lives, because it uses the within-slot temporal variation
                                        the executor-side bit has and the routing-side bit did
                                        not.

DRAWS. `--seed` redraws the probe split, the projection, the codebook and the FM init together;
`--fm-seed` perturbs the FM init alone. Pass 1's draw variance exceeded its headline, so no cell
here is reported from one draw.
"""

import argparse
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from rhm.practice.continuo.continuo import (  # noqa: E402
    _gelu, _dgelu, _proj, _loo_ridge_bal, _loo_bal_perm, spherical_kmeans, unit_rows,
    eta2, eta2_vs_random, partial_spearman, at_support_series, MIN_LEV)

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
AGE_FLOORS = (0, 20, 50)
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# --------------------------------------------------------------------------- #
# the FM: `continuo.FM` minus the embedding
# --------------------------------------------------------------------------- #

class SpanFM:
    """LayerNorm -> Linear(d_in, H) -> GELU -> Linear(H, d_out), zero-init output.

    Differs from `continuo.FM` in exactly one thing: the conditioning index (there the root
    target, here the SLOT) enters as a plain one-hot concatenated to the state read rather than
    through a learned embedding. The reason is the parameter budget — an embedding (28, 288)
    would be 8,064 parameters, 6% of the predicted head on its own, so the FM could not be
    sub-saturation and conditioned at the same time. A one-hot is also the more faithful read:
    `SpanHead`'s slot embedding IS part of the span being predicted, so handing the FM a learned
    version of it would give it a piece of its own target (pass 1's design call on `a_i`).
    """

    def __init__(self, d_in, H, d_out, seed, dtype=np.float32):
        rng = np.random.default_rng(seed)
        dt = dtype
        self.g = np.ones(d_in, dtype=dt)
        self.b = np.zeros(d_in, dtype=dt)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, H)).astype(dt)
        self.b1 = np.zeros(H, dtype=dt)
        self.W2 = np.zeros((H, d_out), dtype=dt)
        self.b2 = np.zeros(d_out, dtype=dt)
        self.d_in, self.H, self.d_out, self.dtype = d_in, H, d_out, dt

    def n_params(self):
        return 2 * self.d_in + self.d_in * self.H + self.H + self.H * self.d_out + self.d_out

    def params(self):
        return [self.g, self.b, self.W1, self.b1, self.W2, self.b2]

    def forward(self, x, cache=False):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + 1e-5)
        xh = (x - mu) * inv
        h0 = xh * self.g + self.b
        a = h0 @ self.W1 + self.b1
        h = _gelu(a)
        y = h @ self.W2 + self.b2
        if cache:
            self._c = (x, inv, xh, h0, a, h)
        return y

    def backward(self, dy):
        x, inv, xh, h0, a, h = self._c
        gW2 = h.T @ dy
        gb2 = dy.sum(0)
        dh = dy @ self.W2.T
        da = dh * _dgelu(a)
        gW1 = h0.T @ da
        gb1 = da.sum(0)
        dh0 = da @ self.W1.T
        gg = (dh0 * xh).sum(0)
        gb = dh0.sum(0)
        return [gg, gb, gW1, gb1, gW2, gb2]

    def fit(self, x, y, iters, lr, wd=1e-4):
        ps = self.params()
        m = [np.zeros_like(p) for p in ps]
        vv = [np.zeros_like(p) for p in ps]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = x.shape[0]
        loss = np.nan
        for t in range(1, iters + 1):
            pred = self.forward(x, cache=True)
            r = pred - y
            loss = float((r * r).mean())
            dy = (2.0 * r / (n * y.shape[1])).astype(self.dtype)
            gs = self.backward(dy)
            cur = lr * (0.5 * (1 + np.cos(np.pi * t / iters)))
            for i, (p, g) in enumerate(zip(ps, gs)):
                m[i] = b1 * m[i] + (1 - b1) * g
                vv[i] = b2 * vv[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** t)
                vh = vv[i] / (1 - b2 ** t)
                p -= cur * (mh / (np.sqrt(vh) + eps) + wd * p)
        return loss


def _gradcheck():
    """Hard assert: analytic gradients against central differences (float64). There is no
    autograd in this file, so this is the only thing standing between a sign error and a
    result."""
    rng = np.random.default_rng(0)
    f = SpanFM(7, 5, 3, seed=1, dtype=np.float64)
    x = rng.normal(size=(9, 7))
    y = rng.normal(size=(9, 3))
    f.W2 = rng.normal(size=f.W2.shape)          # zero-init would hide W1's gradient
    pred = f.forward(x, cache=True)
    r = pred - y
    dy = 2.0 * r / (x.shape[0] * y.shape[1])
    gs = f.backward(dy)
    eps, worst = 1e-6, 0.0
    for p, g in zip(f.params(), gs):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 7)):
            old = flat[i]
            flat[i] = old + eps
            lp = float((((f.forward(x) - y) ** 2).mean()))
            flat[i] = old - eps
            lm = float((((f.forward(x) - y) ** 2).mean()))
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gflat[i]) / max(1.0, abs(num)))
    assert worst < 2e-5, f"gradcheck failed: worst relative error {worst:.3e}"
    return worst


def head_params(n_slots, v, dim, max_span, hidden_mult=4):
    """`span_net._build_span_head`'s parameter count, from the snapshot's own shapes."""
    h = dim * hidden_mult
    return (n_slots * dim                       # slot embedding
            + max_span * dim                    # step embedding
            + v * max_span * dim                # feat embedding
            + dim * dim + dim                   # ctx
            + max_span * (dim * dim + dim)      # in_proj
            + dim * h + h + h * v + v)          # mlp


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_run(tag, arm):
    root = os.path.join(DATA, tag, arm)
    snapdir = os.path.join(root, "snapshots")
    files = sorted(f for f in os.listdir(snapdir) if f.startswith("span_c"))
    snaps = []
    for f in files:
        d = np.load(os.path.join(snapdir, f), allow_pickle=False)
        snaps.append({k: d[k] for k in d.files})
    res = json.load(open(os.path.join(root, "results.json")))
    return snaps, res


def slot_table(snaps):
    """Per slot: its id, level, span, the cycle it was MINTED (first snapshot it appears in),
    and the cycle it FIRST OPENED. Both clocks are read off the record, never hardcoded."""
    tab = {}
    for sn in snaps:
        c = int(sn["cycle"])
        for a, k in enumerate(sn["keys"]):
            k = str(k)
            t = tab.setdefault(k, {"id": int(sn["sid"][a]), "level": int(sn["level"][a]),
                                   "span": int(sn["span"][a]), "mint": c, "first_open": None,
                                   "first_open_tau": None})
            if sn["open_post"][a] and t["first_open"] is None:
                t["first_open"] = c
            if sn["open_tau"][a] and t["first_open_tau"] is None:
                t["first_open_tau"] = c
    return tab


# --------------------------------------------------------------------------- #
# one checkpoint: the fit and the columns
# --------------------------------------------------------------------------- #

def build_xy(sn, n_slots, target="lg0"):
    """(n_slot, P, d_in) inputs and (n_slot, P, v) targets for one checkpoint."""
    feat = sn["feat"].astype(np.float32)                  # (n, P, 3, dim)
    n, P = feat.shape[0], feat.shape[1]
    st = feat.reshape(n, P, -1)
    oh = np.zeros((n, n_slots), dtype=np.float32)
    oh[np.arange(n), sn["sid"].astype(int)] = 1.0
    X = np.concatenate([st, np.repeat(oh[:, None, :], P, axis=1)], axis=-1)
    Y = sn[target].astype(np.float32)
    return X, Y


def fit_checkpoint(sn, n_slots, tr, te, H, seed, iters, lr, target="lg0"):
    """Fit the FM on the train probe states and return the held-out residual and prediction,
    both (n_slot, P_te, v), in the target's own z-scored units."""
    X, Y = build_xy(sn, n_slots, target=target)
    n, P, d_in = X.shape
    v = Y.shape[2]
    xtr = X[:, tr].reshape(-1, d_in)
    xte = X[:, te].reshape(-1, d_in)
    xm, xs = xtr.mean(0), xtr.std(0) + 1e-6
    xtr = (xtr - xm) / xs
    xte = (xte - xm) / xs
    ytr = Y[:, tr].reshape(-1, v)
    yte = Y[:, te].reshape(-1, v)
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    ytr = (ytr - ym) / ys
    yte = (yte - ym) / ys
    if H <= 0:
        # the ridge rung: convex, unique minimiser, so the residual cannot be instrument noise.
        A = np.concatenate([xtr, np.ones((xtr.shape[0], 1), np.float32)], 1)
        B = np.concatenate([xte, np.ones((xte.shape[0], 1), np.float32)], 1)
        lam = 1.0
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32), A.T @ ytr)
        pte = B @ W
        ptr = A @ W
        npar = int(A.shape[1] * v)
    else:
        fm = SpanFM(d_in, H, v, seed=seed)
        fm.fit(xtr, ytr, iters=iters, lr=lr)
        ptr = fm.forward(xtr)
        pte = fm.forward(xte)
        npar = fm.n_params()
    R = (yte - pte).reshape(n, len(te), v)
    info = {"mse_tr": float(((ptr - ytr) ** 2).mean()),
            "mse_te": float(((pte - yte) ** 2).mean()),
            "n_params": npar,
            "rel": float(np.linalg.norm(yte - pte) / (np.linalg.norm(yte) + 1e-12))}
    return R, pte.reshape(n, len(te), v), yte.reshape(n, len(te), v), info


def columns(sn, R, pred, Yz, te, rng):
    """One (n_slot, d) matrix per cell, all over the SAME held-out probe states."""
    n, P_te, v = R.shape
    raw = sn["lg0"].astype(np.float32)[:, te]
    ex = np.exp(raw - raw.max(-1, keepdims=True))
    io = ex / ex.sum(-1, keepdims=True)
    beh = np.zeros_like(raw)
    em0 = sn["emit"][:, te, 0].astype(int)
    idx = np.clip(em0, 0, v - 1)
    np.put_along_axis(beh, idx[:, :, None], 1.0, axis=2)
    shuf = np.empty_like(R)
    for a in range(n):
        shuf[a] = R[a][rng.permutation(P_te)]
    flat = lambda M: M.reshape(n, -1)
    cols = {"R": flat(R), "FM": flat(pred), "LOGIT": flat(Yz), "IO": flat(io),
            "BEH": flat(beh), "SHUF": flat(shuf)}
    norms = np.linalg.norm(flat(R), axis=1)
    return cols, norms


# --------------------------------------------------------------------------- #
# decode B: pooled over (slot, checkpoint), leave-one-SLOT-out
# --------------------------------------------------------------------------- #

def _bal(pred, y):
    pos, neg = y > 0, y < 0
    tpr = float((pred[pos] > 0).mean()) if pos.any() else 0.0
    tnr = float((pred[neg] < 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def group_loo_bal(X, y, groups, lams=LAMS):
    """Leave-one-GROUP-out ridge, balanced accuracy, one number per lambda. Groups are slots,
    so nothing a classifier could learn about slot identity survives into its own fold."""
    X = np.concatenate([np.asarray(X, np.float64),
                        np.ones((X.shape[0], 1), np.float64)], 1)
    gs = np.unique(groups)
    out = []
    for lam in lams:
        pred = np.zeros(len(y))
        ok = True
        for g in gs:
            m = groups == g
            A, b = X[~m], y[~m]
            G = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
            try:
                W = np.linalg.solve(G, A.T @ b)
            except np.linalg.LinAlgError:
                ok = False
                break
            pred[m] = X[m] @ W
        out.append(_bal(np.sign(pred), y) if ok else np.nan)
    return np.asarray(out)


def perm_null_within_cycle(X, y, groups, cycles, lams, n_perm, seed):
    """The permutation null that respects this decode's structure. Rows are (slot, checkpoint),
    so a free row-permutation would destroy the per-checkpoint base rate as well as the
    slot-label pairing and would be trivially easy to beat. This permutes labels WITHIN each
    checkpoint: how many slots are open at cycle c is preserved exactly, and only WHICH slots
    are open is destroyed — which is the null the claim needs."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        yp = y.copy()
        for c in np.unique(cycles):
            m = cycles == c
            yp[m] = y[m][rng.permutation(int(m.sum()))]
        out.append(np.nanmedian(group_loo_bal(X, yp, groups, lams)))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="co_s0")
    ap.add_argument("--arm", default="mperf_log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--h-primary", type=int, default=16)
    ap.add_argument("--h-sweep", default="0,8,16,384")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fm-seed", type=int, default=None)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-projs", default="8,16,32")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--min-class", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--quickfit", action="store_true")
    args = ap.parse_args()
    if args.quickfit:
        args.iters, args.n_perm, args.n_seeds = 120, 40, 2
    t_start = time.time()
    out = args.out or os.path.join(FIG, f"{args.tag}_{args.arm}")
    os.makedirs(out, exist_ok=True)
    L = []
    def say(s=""):
        print(s, flush=True)
        L.append(s)

    gc = _gradcheck()
    snaps, res = load_run(args.tag, args.arm)
    snaps = snaps[::max(1, args.stride)]
    tab = slot_table(snaps)
    n_slots = max(t["id"] for t in tab.values()) + 1
    cfg = res["config"]
    dim = int(cfg["state_dim"]); v = int(snaps[0]["lg0"].shape[2])
    max_span = int(max(t["span"] for t in tab.values()))
    n_slots_head = int(sum(cfg["s"] ** (cfg["depth"] - e)
                           for e in range(2, cfg["max_macro_level"] + 1)))
    HP = head_params(n_slots_head, v, dim, int(cfg["s"] ** (cfg["max_macro_level"] - 1)),
                     cfg["span_hidden_mult"])
    P = int(snaps[0]["n_probe"])
    rng0 = np.random.default_rng(args.seed + 991)
    perm = rng0.permutation(P)
    tr, te = np.sort(perm[: P // 2]), np.sort(perm[P // 2:])

    say("=" * 78)
    say(f"corridor_fit — E2 pass 2 (executor side)   tag={args.tag} arm={args.arm} "
        f"seed={args.seed}")
    say("=" * 78)
    say("")
    say("§0 GATES AND SCALE")
    say(f"  gradcheck worst relative error      {gc:.2e}   (hard assert < 2e-5)")
    say(f"  checkpoints                         {len(snaps)}  "
        f"c{int(snaps[0]['cycle'])}..c{int(snaps[-1]['cycle'])}  (stride {args.stride})")
    say(f"  slots ever live                     {len(tab)}   head slot table {n_slots_head}")
    say(f"  probe states                        {P}  ({len(tr)} train / {len(te)} held out), "
        f"split by state, fixed across every checkpoint")
    say(f"  SpanHead parameters (predicted span) {HP:,}   dim={dim} v={v} "
        f"max_span={cfg['s'] ** (cfg['max_macro_level'] - 1)} hidden_mult="
        f"{cfg['span_hidden_mult']}")
    d_in = 3 * dim + n_slots
    say(f"  FM input dimension                  {d_in} = 3x{dim} state reads + {n_slots} slot "
        f"one-hot")
    hs = [int(x) for x in args.h_sweep.split(",")]
    for H in sorted(set(hs + [args.h_primary])):
        npar = (d_in + 1) * v if H <= 0 else (2 * d_in + d_in * H + H + H * v + v)
        say(f"    H={H:<4d} FM params {npar:>8,}   {100.0 * npar / HP:6.2f}% of the head"
            f"{'   <-- SATURATION RUNG, excluded from every headline' if npar >= 0.5 * HP else ''}"
            f"{'   <-- PRIMARY' if H == args.h_primary else ''}")
    say("")

    # ---- the label ---------------------------------------------------------------------- #
    say("§1 THE LABEL (the parity gate), read off the run's own record")
    ge = res.get("gate_events") or []
    say(f"  gate events                         {len(ge)}  "
        f"(opens {sum(1 for e in ge if e['open'])}, closes {sum(1 for e in ge if not e['open'])})")
    say(f"  commits                             "
        f"{[(e['level'], e['cycle']) for e in res['events'] if e['kind'] == 'commit']}")
    say("  slot                mint  first_open  first_open@tau  level  span")
    for k in sorted(tab, key=lambda q: (tab[q]["level"], tab[q]["id"])):
        t = tab[k]
        say(f"    {k:<8s}        {t['mint']:>4d}  {str(t['first_open']):>10s}  "
            f"{str(t['first_open_tau']):>14s}  {t['level']:>5d}  {t['span']:>4d}")
    say("")
    say("  cycle  n_live  n_open  n_open@tau   by level (live/open)")
    for sn in snaps[::max(1, len(snaps) // 24)]:
        c = int(sn["cycle"])
        per = {}
        for a in range(len(sn["keys"])):
            lv = int(sn["level"][a])
            q = per.setdefault(lv, [0, 0])
            q[0] += 1
            q[1] += int(sn["open_post"][a])
        say(f"  {c:>5d}  {len(sn['keys']):>6d}  {int(sn['open_post'].sum()):>6d}  "
            f"{int(sn['open_tau'].sum()):>10d}   "
            + "  ".join(f"L{lv}:{q[0]}/{q[1]}" for lv, q in sorted(per.items())))
    say("")

    # ---- the fits ----------------------------------------------------------------------- #
    say("§2 THE FIT, PER CHECKPOINT (primary capacity), AND THE INSTRUMENT GUARDS")
    fm_seed0 = args.seed * 1000 + 17 if args.fm_seed is None else args.fm_seed
    store, ens, cover = {}, {}, {}
    for H in sorted(set(hs + [args.h_primary])):
        npar_H = (d_in + 1) * v if H <= 0 else (2 * d_in + d_in * H + H + H * v + v)
        # A rung at or above ~20% of the head costs ~100x the primary rung's fit time and is
        # carried only as the SATURATION DEMONSTRATION, so it is fitted on a strided subset of
        # checkpoints. Every headline cell is at a sub-6% rung on the full set.
        stride = 1 if npar_H < 0.20 * HP else max(1, len(snaps) // 12)
        cis = list(range(0, len(snaps), stride))
        cover[H] = len(cis)
        # `ens_cos` needs >= 2 independent-seed FMs, but only at the rung a headline is read
        # off: refitting every rung n_seeds times triples the reduction's runtime to certify
        # capacities no cell is quoted from. The primary rung gets the full ensemble; the rest
        # get one fit each, and the `ens_cos` column says so.
        n_s = (args.n_seeds if (H > 0 and H == args.h_primary) else 1) if stride == 1 else 1
        rs = []
        for si in range(n_s):
            per_c = {}
            for ci in cis:
                R, pred, Yz, info = fit_checkpoint(snaps[ci], n_slots, tr, te, H,
                                                   fm_seed0 + 101 * si, args.iters, args.lr)
                per_c[ci] = (R, pred, Yz, info)
            rs.append(per_c)
        store[H] = rs[0]
        if H > 0 and len(rs) > 1:
            cos = []
            for ci in cis:
                mats = [unit_rows(r[ci][0].reshape(r[ci][0].shape[0], -1)) for r in rs]
                pw = [float(np.mean(np.sum(mats[i] * mats[j], axis=1)))
                      for i in range(len(mats)) for j in range(i + 1, len(mats))]
                cos.append(float(np.mean(pw)))
            ens[H] = float(np.mean(cos))
        else:
            ens[H] = 1.0
    say("   H   FM%     mse_tr   mse_te    |r|/|t|   ens_cos   n_ckpt  (means over checkpoints)")
    for H in sorted(store):
        infos = [x[3] for x in store[H].values()]
        npar = infos[0]["n_params"]
        say(f"  {H:>3d}  {100.0 * npar / HP:5.2f}%  "
            f"{np.mean([i['mse_tr'] for i in infos]):8.4f} "
            f"{np.mean([i['mse_te'] for i in infos]):8.4f}  "
            f"{np.mean([i['rel'] for i in infos]):8.4f}  "
            f"{('%7.3f' % ens[H]) if (H == 0 or H == args.h_primary) else '      -'}  "
            f"{cover[H]:>5d}"
            f"{'   [ens_cos junk band < 0.65]' if 0 < ens[H] < 0.65 else ''}"
            f"{'   [SATURATION RUNG]' if npar >= 0.5 * HP else ''}")
    say("")

    # ---- decode ------------------------------------------------------------------------- #
    d_projs = [int(x) for x in args.d_projs.split(",")]
    keys_all = sorted(tab, key=lambda q: tab[q]["id"])
    labels = {"open": "open_post", "open_tau": "open_tau"}

    def pooled_rows(H, label_key, floor, clock="mint", level=None):
        """Rows = (slot, checkpoint) admitted by the age floor. Returns cols, y, groups."""
        per = store[H]
        Xc = {k: [] for k in ("R", "FM", "LOGIT", "IO", "BEH", "SHUF")}
        ys, gs, norms, cys = [], [], [], []
        rng = np.random.default_rng(args.seed + 555)
        for ci in sorted(per):
            sn = snaps[ci]
            c = int(sn["cycle"])
            R, pred, Yz, _ = per[ci]
            cols, nrm = columns(sn, R, pred, Yz, te, rng)
            for a, k in enumerate(sn["keys"]):
                k = str(k)
                t = tab[k]
                if level is not None and t["level"] != level:
                    continue
                if clock == "mint":
                    age = c - t["mint"]
                else:
                    fo = t["first_open"]
                    if fo is None:
                        continue
                    age = c - fo
                if age < floor:
                    continue
                for nm in Xc:
                    Xc[nm].append(cols[nm][a])
                ys.append(1.0 if sn[labels[label_key]][a] else -1.0)
                gs.append(t["id"])
                norms.append(nrm[a])
                cys.append(c)
        if not ys:
            return None
        return ({nm: np.asarray(M, dtype=np.float32) for nm, M in Xc.items()},
                np.asarray(ys), np.asarray(gs), np.asarray(norms), np.asarray(cys))

    def decode_pooled(bundle, d, tag_):
        cols, y, gs, norms, cys = bundle
        rec = {}
        P_ = None
        for nm, M in cols.items():
            M = unit_rows(M.astype(np.float32))
            if P_ is None or P_.shape[0] != M.shape[1]:
                P_ = _proj(M.shape[1], d, args.seed + 13)
            accs = group_loo_bal(M @ P_, y, gs)
            rec[nm] = {"med": float(np.nanmedian(accs)), "min": float(np.nanmin(accs)),
                       "max": float(np.nanmax(accs))}
        X = np.log(np.clip(norms, 1e-12, None))[:, None].astype(np.float32)
        accs = group_loo_bal(X, y, gs)
        rec["NORM"] = {"med": float(np.nanmedian(accs)), "min": float(np.nanmin(accs)),
                       "max": float(np.nanmax(accs))}
        if args.n_perm:
            M = unit_rows(cols["R"].astype(np.float32))
            null = perm_null_within_cycle(M @ _proj(M.shape[1], d, args.seed + 13), y, gs,
                                          cys, LAMS, max(20, args.n_perm // 5),
                                          args.seed + 71)
            rec["R"]["perm_mean"] = float(np.nanmean(null))
            rec["R"]["perm_p95"] = float(np.nanquantile(null, 0.95))
            rec["R"]["perm_p"] = float(np.nanmean(null >= rec["R"]["med"]))
        rec["_n"] = int(len(y))
        rec["_pos"] = int((y > 0).sum())
        rec["_slots"] = int(len(np.unique(gs)))
        return rec

    say("§3 DECODE B — pooled over (slot, checkpoint), leave-one-SLOT-out folds")
    say("   headline = median balanced accuracy over the lambda grid; chance 0.500")
    say("   'adv' = self R minus the best I/O-only observer (LOGIT / IO / BEH)")
    say("")
    dB = {}
    # The grid is run at BOTH the deterministic RIDGE rung and the swept MLP. Pass 1's
    # instrument-regime confound (an FM sitting in the `ens_cos` junk band at matched H on one
    # arm and not the other) was fixed exactly by reading the arms at the ridge, whose residual
    # is a unique minimiser and whose `ens_cos` is 1.000 by construction — and this pass has
    # two arms to compare, so that fix is carried from the start rather than after the fact.
    for H in sorted({0, args.h_primary}):
        say(f"  --- capacity H={H} ({100.0 * store[H][sorted(store[H])[0]][3]['n_params'] / HP:.2f}% "
            f"of the head, ens_cos {ens[H]:.3f})"
            + ("   [the deterministic ridge rung]" if H == 0 else "") + " ---")
        for label_key in ("open", "open_tau"):
            for clock in ("mint", "open"):
                if clock == "open" and label_key == "open":
                    continue    # degenerate: conditioning on having opened fixes the label
                for floor in AGE_FLOORS:
                    b = pooled_rows(H, label_key, floor, clock=clock)
                    if b is None or (b[1] > 0).sum() < args.min_class or \
                            (b[1] < 0).sum() < args.min_class:
                        say(f"  label={label_key:<8s} clock={clock:<5s} floor={floor:<3d}  "
                            f"— skipped (a class has < {args.min_class} rows)")
                        continue
                    for d in d_projs:
                        r = decode_pooled(b, d, f"{label_key}/{clock}/{floor}/d{d}")
                        dB[f"H{H}|{label_key}|{clock}|{floor}|d{d}"] = r
                        best_obs = max(r["LOGIT"]["med"], r["IO"]["med"], r["BEH"]["med"])
                        say(f"  label={label_key:<8s} clock={clock:<5s} floor={floor:<3d} "
                            f"d={d:<3d} n={r['_n']:<5d} pos={r['_pos']:<5d} "
                            f"slots={r['_slots']:<3d} | "
                            f"R {r['R']['med']:.3f}  FM {r['FM']['med']:.3f}  "
                            f"LOGIT {r['LOGIT']['med']:.3f}  IO {r['IO']['med']:.3f}  "
                            f"BEH {r['BEH']['med']:.3f}  SHUF {r['SHUF']['med']:.3f}  "
                            f"NORM {r['NORM']['med']:.3f} | adv "
                            f"{r['R']['med'] - best_obs:+.3f}"
                            f"  perm_p {r['R'].get('perm_p', float('nan')):.3f}")
                say("")

    say("§3b CAPACITY INVARIANCE of the headline cell (label=open, clock=mint, floor=20)")
    for H in sorted(store):
        b = pooled_rows(H, "open", 20)
        if b is None:
            continue
        r = decode_pooled(b, d_projs[len(d_projs) // 2], f"cap{H}")
        best_obs = max(r["LOGIT"]["med"], r["IO"]["med"], r["BEH"]["med"])
        npar = store[H][0][3]["n_params"]
        say(f"  H={H:<4d} ({100.0 * npar / HP:5.2f}% of head, ens_cos {ens[H]:.3f})  "
            f"R {r['R']['med']:.3f}  best-obs {best_obs:.3f}  adv {r['R']['med'] - best_obs:+.3f}"
            f"  NORM {r['NORM']['med']:.3f}")
    say("")

    say("§3c THE THREE-WAY / WITHIN-LEVEL READ (pass 1 design call 5ii's age disambiguator)")
    for lv in sorted({t["level"] for t in tab.values()}):
        for floor in (0, 20):
            b = pooled_rows(args.h_primary, "open", floor, level=lv)
            if b is None or (b[1] > 0).sum() < args.min_class or (b[1] < 0).sum() < args.min_class:
                say(f"  level L{lv} floor={floor:<3d} — skipped (a class has "
                    f"< {args.min_class} rows)")
                continue
            r = decode_pooled(b, d_projs[len(d_projs) // 2], f"L{lv}")
            best_obs = max(r["LOGIT"]["med"], r["IO"]["med"], r["BEH"]["med"])
            say(f"  level L{lv} floor={floor:<3d} n={r['_n']:<5d} pos={r['_pos']:<5d} "
                f"slots={r['_slots']:<3d} | R {r['R']['med']:.3f}  best-obs {best_obs:.3f}  "
                f"adv {r['R']['med'] - best_obs:+.3f}  NORM {r['NORM']['med']:.3f}")
    say("")

    # ---- decode A ----------------------------------------------------------------------- #
    say("§4 DECODE A — per checkpoint, rows = slots (pass 1's exact shape)")
    d = d_projs[len(d_projs) // 2]
    rowsA = []
    rng = np.random.default_rng(args.seed + 777)
    for ci, sn in enumerate(snaps):
        R, pred, Yz, _ = store[args.h_primary][ci]
        y = np.where(sn["open_post"] > 0, 1.0, -1.0)
        if (y > 0).sum() < args.min_class or (y < 0).sum() < args.min_class:
            continue
        cols, nrm = columns(sn, R, pred, Yz, te, rng)
        rec = {}
        for nm, M in cols.items():
            M = unit_rows(M.astype(np.float32))
            X = M @ _proj(M.shape[1], d, args.seed + 13)
            accs, valid, _ = _loo_ridge_bal(X, y, LAMS)
            rec[nm] = float(np.nanmedian(accs)) if valid.any() else np.nan
        X = np.log(np.clip(nrm, 1e-12, None))[:, None].astype(np.float32)
        accs, valid, _ = _loo_ridge_bal(X, y, LAMS)
        rec["NORM"] = float(np.nanmedian(accs)) if valid.any() else np.nan
        rec["cycle"] = int(sn["cycle"])
        rec["n_open"] = int((y > 0).sum())
        rec["n_closed"] = int((y < 0).sum())
        rowsA.append(rec)
    if rowsA:
        say(f"  {len(rowsA)} checkpoints have both classes at >= {args.min_class}")
        for nm in ("R", "FM", "LOGIT", "IO", "BEH", "SHUF", "NORM"):
            xs = np.array([r[nm] for r in rowsA], float)
            say(f"    {nm:<6s} median {np.nanmedian(xs):.3f}   IQR "
                f"[{np.nanquantile(xs, .25):.3f}, {np.nanquantile(xs, .75):.3f}]")
        bo = np.array([max(r["LOGIT"], r["IO"], r["BEH"]) for r in rowsA], float)
        rr = np.array([r["R"] for r in rowsA], float)
        say(f"    advantage (R - best observer): median {np.nanmedian(rr - bo):+.3f}  "
            f"IQR [{np.nanquantile(rr - bo, .25):+.3f}, {np.nanquantile(rr - bo, .75):+.3f}]")
    else:
        say("  no checkpoint had both classes at the minimum — decode A is vacuous here")
    say("")

    # ---- occupancy ---------------------------------------------------------------------- #
    say("§5 OCCUPANCY — one codebook over residual directions, applied unchanged everywhere")
    pool = []
    for ci in range(0, len(snaps), max(1, len(snaps) // 20)):
        R = store[args.h_primary][ci][0]
        pool.append(unit_rows(R.reshape(R.shape[0], -1)))
    pool = np.concatenate(pool, 0)
    cb, _lab, _inert = spherical_kmeans(pool, args.K, seed=args.seed + 31)
    occ, cyc = [], []
    for ci, sn in enumerate(snaps):
        R = store[args.h_primary][ci][0]
        U = unit_rows(R.reshape(R.shape[0], -1))
        a = np.argmax(U @ cb.T, axis=1)
        h = np.bincount(a, minlength=args.K).astype(float)
        occ.append(h / max(h.sum(), 1))
        cyc.append(int(sn["cycle"]))
    occ = np.asarray(occ)
    tv = 0.5 * np.abs(np.diff(occ, axis=0)).sum(1)
    cyc_tv = cyc[1:]
    ev = {f"commit_L{e['level']}": e["cycle"] for e in res["events"] if e["kind"] == "commit"}
    for e in ge:
        ev.setdefault(f"gate_{e['slot']}_{'open' if e['open'] else 'close'}_c{e['cycle']}",
                      e["cycle"])
    say(f"  codebook K={args.K} over {pool.shape[0]} directions; "
        f"TV drift median {np.median(tv):.4f}, p90 {np.quantile(tv, .9):.4f}")
    say("  named event            cycle   TV drift   percentile among all checkpoint steps")
    for name, c in sorted(ev.items(), key=lambda q: q[1]):
        if c not in cyc_tv:
            continue
        i = cyc_tv.index(c)
        say(f"    {name:<26s} {c:>4d}   {tv[i]:.4f}   "
            f"{100.0 * float((tv < tv[i]).mean()):5.1f}%")
    say("")

    # ---- guards ------------------------------------------------------------------------- #
    say("§6 GUARDS")
    say("  hierarchy-eta2 on the residual norm, POOLED over (slot, checkpoint) rows, each")
    say("  conditioner beside a matched-random grouping of the same sizes. The GRAMMAR")
    say("  conditioner is `dp_content` — the level-1 spelling the DP would write for that span,")
    say("  i.e. the deep-hierarchy content this substrate has. `cycle` and `era` are the junk")
    say("  conditioners: a residual that keys only to the calendar is junk (paper 2 section 8).")
    nrm_p, c_dp, c_lev, c_sid, c_cyc, c_era = [], [], [], [], [], []
    for ci in sorted(store[args.h_primary]):
        sn = snaps[ci]
        R = store[args.h_primary][ci][0]
        nrm_p.append(np.linalg.norm(R.reshape(R.shape[0], -1), axis=1))
        c_dp.append(sn["dp"][:, 0, 0].astype(int))
        c_lev.append(sn["level"].astype(int))
        c_sid.append(sn["sid"].astype(int))
        c_cyc.append(np.full(len(sn["keys"]), int(sn["cycle"])))
        c_era.append(np.full(len(sn["keys"]), int(sn["era"])))
    nrm_p = np.concatenate(nrm_p)
    conds = {"dp_content": np.concatenate(c_dp), "slot_id": np.concatenate(c_sid),
             "level": np.concatenate(c_lev), "cycle [junk]": np.concatenate(c_cyc),
             "era [junk]": np.concatenate(c_era)}
    for nm, g in conds.items():
        e_, rnd, dz = eta2_vs_random(nrm_p, g, seed=args.seed + 5)
        say(f"    {nm:<14s} n_groups {len(np.unique(g)):>4d}   eta2 {e_:.3f}   "
            f"random-matched {rnd:.3f}   excess {dz:+.3f}")
    say("  the TRUNK control target (no slot conditioning anywhere in its computation):")
    b_ok = True
    try:
        R2 = {}
        for ci in sorted(store[args.h_primary]):
            R2[ci] = fit_checkpoint(snaps[ci], n_slots, tr, te, args.h_primary,
                                    fm_seed0, args.iters, args.lr, target="bl0")
        keep = store[args.h_primary]
        for fl in (0, 20, 50):
            store[args.h_primary] = R2
            b = pooled_rows(args.h_primary, "open", fl)
            store[args.h_primary] = keep
            if b is None or (b[1] > 0).sum() < args.min_class or \
                    (b[1] < 0).sum() < args.min_class:
                say(f"    floor={fl:<3d} — skipped (a class has < {args.min_class} rows)")
                continue
            r = decode_pooled(b, d_projs[len(d_projs) // 2], "trunk")
            best_obs = max(r["LOGIT"]["med"], r["IO"]["med"], r["BEH"]["med"])
            say(f"    floor={fl:<3d} label=open  n={r['_n']:<5d}  R {r['R']['med']:.3f}  "
                f"best-obs {best_obs:.3f}  adv {r['R']['med'] - best_obs:+.3f}")
    except Exception as exc:      # noqa: BLE001
        b_ok = False
        say(f"    TRUNK control failed: {exc}")
    say("")

    # ---- at_support --------------------------------------------------------------------- #
    say("§7 THE OUTER LOOP'S OWN GAUGE — command-energy share vs at_support")
    cm = []
    for ci, sn in enumerate(snaps):
        R = store[args.h_primary][ci][0]
        e_ = (R.reshape(R.shape[0], -1) ** 2).sum(1)
        op = sn["open_post"] > 0
        cm.append(float(e_[op].sum() / max(e_.sum(), 1e-12)) if op.any() else np.nan)
    cm = np.asarray(cm)
    eras = np.asarray([int(sn["era"]) for sn in snaps])
    cyc_a = np.asarray(cyc, float)
    for lvl in (3, 4):
        ats = np.asarray(at_support_series(res, lvl), float)
        a = np.asarray([ats[c - 1] if 0 < c <= len(ats) else np.nan for c in cyc], float)
        m = np.isfinite(a) & np.isfinite(cm)
        if m.sum() < 8:
            say(f"  at_support@L{lvl}: too few aligned cycles")
            continue
        raw, praw, part, nn_ = partial_spearman(cm[m], a[m], cyc_a[m], eras[m])
        say(f"  cmass_R ~ at_support@L{lvl}   spearman raw {raw:+.3f} (p={praw:.3f})   "
            f"residualised on cycle+era {part:+.3f}   (n={nn_})")
    say("")
    say(f"[done in {time.time() - t_start:.0f}s]  -> {out}")

    with open(os.path.join(out, "reduction.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out, "corridor.json"), "w") as fh:
        json.dump({"args": vars(args), "head_params": HP, "ens_cos": ens,
                   "decodeB": dB, "decodeA": rowsA, "occ": occ.tolist(), "cycles": cyc,
                   "tv": tv.tolist(), "cmass": cm.tolist(),
                   "slots": tab, "trunk_control_ok": b_ok}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
