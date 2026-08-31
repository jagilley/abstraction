"""audiation/fit — E1 PHASE 2, THE OFFLINE FITTER AND DECODE MATRIX. No GPU, no Modal, no
substrate: everything here reads the phase-1 bytes under `figures/<tag>/<arm>/` and the two
head snapshots that bracket each cycle, and re-derives every readout it needs.

The register is `../maestro/fit.py`'s — an offline phase that fits on already-fetched bytes and
states its design calls where they can be disagreed with — and the recipe is
`../../confabulation/temporal/epistemics/temporal_epistemics.py`'s: the matched FM1/FM2 pair
(`SlotFM`), the update parametrisation, stratum-residualised targets with train-rows-only means,
matched readout capacity (closed-form ridge at the linear rung, then MLP-16/64/256), and the
guards (`ens_cos`, a junk-residual eta^2, scalar-norm negative controls, a shuffled null). The
observer ladder's RUNG STRUCTURE is `../../confabulation/temporal/temporal_confabulation.py`'s
(`Observer`, `train_observer`, `run_ladder`) -- capacity-swept third-party predictors holding
increasing PUBLIC information plus a half-data budget control; its sequence SHAPE is not ported,
because the practice datum is a configuration, not a prefix.

THE QUESTION (ROADMAP.md 4.2, E1): is there a per-datum revision signal, is it gated by agency,
and does forming a residual against the forecast do any work the raw update does not.

--------------------------------------------------------------------------------------------
WHY THIS IS ALL NUMPY. `analyze_audiation.py` already re-implements both heads in numpy and
validates that re-implementation against the run's own fp32 readouts; this file imports that
function rather than re-deriving it, so the recompute path is the one phase 1 certified. The
learned objects here -- the FM pair, the probes, the observers -- are all one-hidden-layer MLPs,
so the whole fitter is a few matmuls and an AdamW, and `_gradcheck()` is a hard assert against
finite differences at import of `main`. That keeps phase 2 runnable with no torch and no GPU,
which is the property phase 1 was built to give it.

--------------------------------------------------------------------------------------------
THE DATUM, and why it is this one.

A CHOICE STATE: one surviving beam tip at a step `t < budget` of one instance of one cycle. The
tip is a state the agent stood at and chose from; there are 6,528 of them per routed cycle
(1 + 5 + 6*16 per instance, 64 instances).

  state  s  = (z, root)          exactly what both heads read, and nothing else
  move   u  = the slot of the HIGHEST-SCORING scored child of that tip

`u` needs a definition because a beam takes a SET of moves at a state: the topk is global across
the beam, so a tip has 0..kk kept children. The agent's own ranking at `s` is unambiguous
though -- `c_score` is `value(child, r*)`, the number the topk ranked -- so `u` is the move the
agent's own value head put first at `s`. Whenever exactly one child survives (the modal case)
that child IS the argmax, because the global topk is monotone in the score; when none survives,
`u` is still the move the agent would have taken. `u_kept` and `n_kept` are carried so the
distinction is visible rather than assumed. The alternative datum -- one row per KEPT EDGE --
duplicates the target across a parent's surviving children for no gain; the alternative state --
the child's -- makes `u` retrospective, and the efference copy the arity-2 slot is about is
prospective.

  provenance `u_src` is `c_src` of that argmax child: 0 enumerated / 1 pi-proposed / 2 forced /
  4 EXPLORE-INJECTED. This is the agency label. Explore candidates are drawn uniformly from what
  pi did NOT propose on a private stream, so where `u_src == 4` the chosen move is not a
  deterministic function of the state -- the `g > 0` condition teacher-forced NTP structurally
  lacks.

  grade `ybar` = the fraction of the tip's DESCENDANT terminal survivors that solved, with
  `n_desc` their count. This is not a convenience: `run_arm` pushes `traj[t]` for every surviving
  trajectory with `y = succ` of that trajectory's tip, so the value buffer contains the tip
  exactly `n_desc` times with exactly those labels, and `(ybar, n_desc)` is a sufficient
  statistic for the tip's contribution to the value update. A tip with `n_desc == 0` was pruned
  and never entered the buffer at all; those rows are kept and flagged, never silently dropped.

--------------------------------------------------------------------------------------------
THE PAIR. `SlotFM` ported: one MLP over `[z, onehot(root), onehot(slot)]` predicting the UPDATE

    Delta(s) = [ v_{c+1}(s) - v_c(s) ,  pi_{c+1}(s) - pi_c(s) ]   in R^57

recomputed offline from the two fp32 snapshots at the SAME logged fp16 `z`, so the storage
quantisation is common-mode and cancels. FM2 gets `onehot(u)`; FM1 gets `onehot(MASK)`, MASK
being slot index 56. Same init seed (asserted equal at construction), same optimiser, same step
budget, same batches -- they are stepped in ONE loop -- and ONLY the slot's content differs.

Why Delta and not the state: this is the donor's discipline (a), and it is load-bearing here for
the same reason. A forecaster reading `s` and asked for the post-update readout would spend its
capacity reproducing the pre-update readout, which it can compute exactly.

ONE DEVIATION FROM THE DONOR, stated: the FM's TARGET is per-dimension z-scored on FM-train rows
and the forecast is un-scaled back before any residual is formed. Raw-MSE would weight the 57
output dimensions by their raw variance, and `|dpi|` is ~3x `|dv|`, so a raw-MSE FM systematically
under-fits the value component -- which would hand `r` the value error by construction and
corrupt the r-vs-Delta comparison that is the whole point. This is `_standardise`'s discipline
applied on the target side, for `_standardise`'s reason.

--------------------------------------------------------------------------------------------
WHAT LEAKS, AND WHY THE SPLIT DOES NOT. Two nested splits, because two different objects are
being fit and they have different leaks.

  FM SPLIT, by CYCLE. The FM is trained on one set of cycles and every source in the decode
  matrix is computed on rows of DISJOINT cycles. `Delta` is one shared per-cycle update, so an FM
  that had seen any row of cycle c could partly identify cycle c's update map and its residual
  there would be small for a reason that has nothing to do with the forecast. Holding out whole
  cycles is also the only split under which the forecast could have been materialised BEFORE the
  cycle's grades arrived, which is the timing claim E1 is a measurement of. Primary split is
  INTERLEAVED (every 4th cycle held out) so both sides cover all five eras; `--split forward`
  re-runs the crux cells with the last quarter of the run held out, which is strictly harder and
  says whether the interleaving bought anything.

  PROBE SPLIT, by INSTANCE within the held-out cycles. The probes and the observers are fit and
  scored inside the FM's test cycles, split by instance -- a cycle's 64 instances are independent
  freshly-corrupted problems, and a whole trajectory family stays on one side, so no probe row
  shares a beam with its own training rows. This is what lets the strata be CYCLE x STEP: with a
  cycle-level split every test stratum would be unseen in train and `_stratum_residualise` would
  fall back to the global mean, i.e. would not residualise at all.

  STRATA = CYCLE x STEP. Phase 1 flagged three non-stationarities and the cycle term absorbs all
  three (the |dv| 3x decay across the run, the |dpi| p95 spike on the cycle after a commit, the
  era structure); the step term absorbs the beam's depth, along which the tip population, the
  survival rate and the value's calibration all change systematically. Without them the matrix
  ranks sources by how well they encode the calendar -- the donor observed exactly that, with
  one-hot position beating every content source. Raw targets are reported beside the
  residualised ones so the gap between the two families reads off how much of a decode is
  calendar.

--------------------------------------------------------------------------------------------
CYCLES 1-4 ARE EXCLUDED. The beam enumerated (`routed=False`), pi was never read (the logged
`t_pi` is all-NaN), and there is no explore injection, so the agency contrast does not exist
there and the update regime is different. 5..116 is the analysed range; the exclusion is stated
in the reduction, not silent.

Run (from experiments/):
    python3 rhm/practice/audiation/fit.py --tag au_s0 --arm anchor
    python3 rhm/practice/audiation/fit.py --tag au_s0 --smoke        # ~2 min wiring pass
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_audiation import head_forward, load_heads  # noqa: E402  (the certified recompute)

FIG = os.path.join(HERE, "figures")
MASK_SLOT = None          # set from n_slots at build time: index n_slots is the learned MASK
SRC_NAMES = {0: "enumerated", 1: "proposed", 2: "forced", 4: "explore"}


# ======================================================================
# Readout statistics. Ported from the donor (`temporal_epistemics.py`
# lines 182-260), which is already numpy; docstrings trimmed to the part
# that still applies here.
# ======================================================================

def _r2_np(y, yhat):
    """Out-of-sample R^2 against the test set's own mean. Allowed to go negative."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _stratum_residualise(y, strata, tr_idx):
    """y minus E[y | stratum], stratum means from TRAIN rows only; strata unseen in train fall
    back to the global train mean, so no test row is residualised by its own label."""
    y = np.asarray(y, float)
    st = np.asarray(strata)
    K = int(st.max()) + 1
    tr_mask = np.zeros(y.size, bool)
    tr_mask[tr_idx] = True
    cnt = np.bincount(st[tr_mask], minlength=K).astype(float)
    tot = np.bincount(st[tr_mask], weights=y[tr_mask], minlength=K)
    gm = float(y[tr_mask].mean())
    mu = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), gm)
    return y - mu[st]


def _standardise(X, tr_idx):
    """Per-dimension z-score with statistics from TRAIN rows only. Matched readout capacity is
    meaningless without it: the sources differ in norm by an order of magnitude."""
    mu = X[tr_idx].mean(0, keepdims=True)
    sd = X[tr_idx].std(0, keepdims=True)
    return ((X - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def _ridge(Xtr, ytr, Xte, lams=(1e-4, 1e-2, 1.0), cut=None):
    """Closed-form ridge, lambda on a held-in split of the train rows. The linear rung is
    solved exactly rather than by SGD -- optimiser noise at the small end is exactly the confound
    that would fake a concentration curve.

    `cut` is where the caller has arranged the held-in rows to start. It must respect the SAME
    grouping the train/test split respects (see `_group_split`); a positional 90/10 does not, and
    on this substrate that silently selects too little regularisation.
    """
    n, d = Xtr.shape
    cut = max(1, int(0.9 * n)) if cut is None else max(1, min(cut, n - 1))
    Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]
    if Xb.shape[0] < 10:
        Xa, ya, Xb, yb = Xtr, ytr, Xtr, ytr
    I = np.eye(d)
    ma = ya.mean()
    Ga, ba = Xa.T @ Xa, Xa.T @ (ya - ma)
    mf = ytr.mean()
    Gf, bf = Xtr.T @ Xtr, Xtr.T @ (ytr - mf)
    best, best_lam = None, lams[0]
    for lam in lams:
        w = np.linalg.solve(Ga + lam * Xa.shape[0] * I, ba)
        sc = -float(((yb - ma - Xb @ w) ** 2).mean())
        if best is None or sc > best:
            best, best_lam = sc, lam
    w = np.linalg.solve(Gf + best_lam * n * I, bf)
    return Xte @ w + mf, best_lam


def _ensemble_cos(residuals):
    """`rhm_confabulation._ensemble_cos`, numpy. Input-centred pairwise cosine between residuals
    left by INDEPENDENT-seed FMs. High => the residual is determined by the input (a gap any FM
    misses identically); low => it is the particular FM's idiosyncratic noise."""
    cent = [r - r.mean(0, keepdims=True) for r in residuals]
    sims = []
    for i in range(len(cent)):
        for j in range(i + 1, len(cent)):
            a, b = cent[i], cent[j]
            na = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            sims.append(float(np.mean((a * b).sum(-1) / np.maximum(na, 1e-12))))
    return float(np.mean(sims)) if sims else float("nan")


def _eta2(x, groups):
    """Fraction of a source's total variance explained by a grouping -- the junk-residual check,
    practice-shaped. `x` is (n, d); returns the variance-weighted eta^2 pooled over dimensions.
    Near 1 against `cycle` means the source is a calendar and nothing else."""
    x = np.asarray(x, np.float64)
    g = np.asarray(groups)
    K = int(g.max()) + 1
    cnt = np.bincount(g, minlength=K).astype(np.float64)
    tot = np.zeros((K, x.shape[1]))
    np.add.at(tot, g, x)
    mu = tot / np.maximum(cnt, 1)[:, None]
    gm = x.mean(0, keepdims=True)
    ssb = float((cnt[:, None] * (mu - gm) ** 2).sum())
    sst = float(((x - gm) ** 2).sum())
    return ssb / sst if sst > 0 else float("nan")


# ======================================================================
# The learned objects: one-hidden-layer MLPs, AdamW, in numpy.
# ======================================================================

_SQRT2 = np.sqrt(2.0).astype(np.float32) if hasattr(np.sqrt(2.0), "astype") else np.float32(np.sqrt(2.0))


def _gelu(x):
    """`nn.GELU()`'s default is the EXACT erf form. Imported from scipy in
    `analyze_audiation`; here the tanh approximation would be a silent 1e-3 mismatch against
    nothing, since nothing outside this file consumes these activations -- but the heads use
    erf, so the instruments use erf too."""
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _dgelu(x):
    from scipy.special import erf
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


class MLP:
    """d_in -> hidden -> d_out, GELU. `hidden == 0` is a bare linear map (used only by the
    gradcheck; the linear rung of every real sweep is the exact ridge)."""

    def __init__(self, d_in, hidden, d_out, seed):
        rng = np.random.default_rng(seed)
        if hidden:
            self.W1 = (rng.standard_normal((d_in, hidden)) / np.sqrt(d_in)).astype(np.float32)
            self.b1 = np.zeros(hidden, np.float32)
            self.W2 = (rng.standard_normal((hidden, d_out)) / np.sqrt(hidden)).astype(np.float32)
        else:
            self.W1 = self.b1 = None
            self.W2 = (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)
        self.b2 = np.zeros(d_out, np.float32)
        self.hidden = hidden

    def params(self):
        return [p for p in (self.W1, self.b1, self.W2, self.b2) if p is not None]

    def forward(self, X, cache=False):
        if self.hidden:
            a = X @ self.W1 + self.b1
            h = _gelu(a)
            out = h @ self.W2 + self.b2
            return (out, (X, a, h)) if cache else out
        out = X @ self.W2 + self.b2
        return (out, (X, None, None)) if cache else out

    def backward(self, cx, dout):
        X, a, h = cx
        n = X.shape[0]
        if self.hidden:
            gW2 = h.T @ dout / n
            gb2 = dout.mean(0)
            dh = dout @ self.W2.T
            da = dh * _dgelu(a)
            gW1 = X.T @ da / n
            gb1 = da.mean(0)
            return [gW1, gb1, gW2, gb2]
        return [X.T @ dout / n, dout.mean(0)]


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads, clip=1.0):
        # global-norm clipping at 1.0, the donor's `clip_grad_norm_` -- without it a probe on a
        # heavy-tailed source occasionally takes one step that costs it the fit, and the cell
        # then reads as a capacity fact instead of an optimiser accident.
        if clip:
            gn = np.sqrt(sum(float((g * g).sum()) for g in grads))
            if gn > clip:
                grads = [g * (clip / gn) for g in grads]
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            if p.ndim > 1 and self.wd:
                p -= self.lr * self.wd * p
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


def _group_split(idx, groups, frac, seed):
    """Split `idx` into (fit, held-in) by GROUP, never by row.

    THE BUG THIS EXISTS TO PREVENT, because it is invisible and it inverts the capacity ladder.
    A beam's 16 tips at one step of one instance descend from one corrupted start and differ by a
    few moves, so within an instance the rows are near-duplicates with near-identical targets. A
    positional 90/10 of the train rows therefore puts a copy of almost every held-in row in the
    fit set: validation MSE falls monotonically (0.65 -> 0.36 measured) while out-of-sample R^2
    on rows from OTHER instances collapses (-0.08 -> -0.56 measured). Early stopping on that
    slice does not stop early, it stops at the worst iterate. The train/test split is by
    instance, so every held-in split inside it must be by instance too.
    """
    g = np.asarray(groups)[idx]
    ug = np.unique(g)
    rs = np.random.default_rng(seed).permutation(ug.size)
    n_hold = max(1, int(frac * ug.size))
    hold = np.isin(g, ug[rs[:n_hold]])
    if hold.all() or (~hold).sum() < 64:
        return idx, idx[:0]
    return idx[~hold], idx[hold]


def train_mlp(net, X, Y, tr, *, steps, batch, lr, seed, wd=0.01, groups=None, val_frac=0.1,
              evals=12):
    """MSE on standardised targets, with EARLY STOPPING on a held-in slice of the train rows.

    A fixed step budget across cells of very different size is what a matched-capacity ladder
    needs; what is scored is the best iterate on rows the fit never touched. The held-in slice is
    taken BY GROUP (`_group_split`) — see there for why a row-wise slice makes this function
    actively harmful. The linear rung never enters here; it is a closed-form ridge.
    """
    if groups is None:
        n_val = max(64, int(val_frac * tr.size))
        rs = np.random.default_rng(seed + 77).permutation(tr.size)
        tr_val, tr_fit = tr[rs[:n_val]], tr[rs[n_val:]]
    else:
        tr_fit, tr_val = _group_split(tr, groups, val_frac, seed + 77)
    if tr_fit.size < batch:
        tr_fit, tr_val = tr, tr[:0]
    opt = AdamW(net.params(), lr=lr, wd=wd)
    rng = np.random.default_rng(seed)
    every = max(1, steps // evals)
    best, best_p = np.inf, None
    vX = X[tr_val[:20000]] if tr_val.size else None
    vY = Y[tr_val[:20000]] if tr_val.size else None
    for st in range(steps):
        i = tr_fit[rng.integers(0, tr_fit.size, batch)]
        out, cx = net.forward(X[i], cache=True)
        d = (out - Y[i]) * (2.0 / out.shape[-1])
        opt.step(net.backward(cx, d.astype(np.float32)))
        if vX is not None and ((st + 1) % every == 0 or st == steps - 1):
            v = float(((net.forward(vX) - vY) ** 2).mean())
            if v < best:
                best, best_p = v, [p.copy() for p in net.params()]
    if best_p is not None:
        for p, q in zip(net.params(), best_p):
            p[...] = q
    return net


def predict(net, X, chunk=65536):
    return np.concatenate([net.forward(X[i:i + chunk]) for i in range(0, X.shape[0], chunk)])


def _gradcheck():
    """Hard assert: the hand-written backward matches central differences. Runs in ~0.1 s and
    is the reason this file is allowed to have no autograd."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((17, 6)).astype(np.float32)
    Y = rng.standard_normal((17, 3)).astype(np.float32)
    for hidden in (0, 5):
        net = MLP(6, hidden, 3, seed=1)
        out, cx = net.forward(X, cache=True)
        loss = lambda: float(((net.forward(X) - Y) ** 2).mean())
        g = net.backward(cx, ((out - Y) * (2.0 / 3)).astype(np.float32))
        for p, gp in zip(net.params(), g):
            flat = p.reshape(-1)
            for k in rng.integers(0, flat.size, min(8, flat.size)):
                old = float(flat[k])
                flat[k] = old + 1e-3
                lp = loss()
                flat[k] = old - 1e-3
                lm = loss()
                flat[k] = old
                num = (lp - lm) / 2e-3
                ana = float(gp.reshape(-1)[k])
                assert abs(num - ana) <= 2e-3 * max(1.0, abs(num)), \
                    f"gradcheck hidden={hidden}: fd {num:.6f} vs analytic {ana:.6f}"
    return True


# ======================================================================
# (1) EXTRACTION -- the per-datum table, off the fetched bytes
# ======================================================================

def _shard_for(files, c):
    for f in files:
        b = os.path.basename(f)
        if int(b[5:9]) <= c <= int(b[11:15]):
            return f
    return None


def build_table(root, *, rows_per_cycle, seed, cycles=None, verbose=True):
    """One row per CHOICE STATE. See the module docstring for what a row is and why.

    The beam's record is perfectly regular -- TIPS are contiguous per cycle and sorted by
    (step, inst, tip), CANDIDATES by (step, inst, parent) with `kk` consecutive rows per parent
    -- so every join here is arithmetic on offsets, and the arithmetic is ASSERTED against the
    stored index columns on every cycle rather than trusted.
    """
    audi = json.load(open(os.path.join(root, "audiation.json")))
    cyc_meta = {c["cycle"]: c for c in audi["cycles"]}
    n_slots = int(audi["n_slots"])
    files = sorted(glob.glob(os.path.join(root, "decisions", "*.npz")))
    want = cycles if cycles is not None else [c for c in sorted(cyc_meta) if cyc_meta[c]["routed"]]

    cols = {k: [] for k in ("cycle", "era", "step", "inst", "tip", "root", "u_slot", "u_src",
                            "u_kept", "n_kept", "n_desc", "sum_succ", "v_c", "kk",
                            "u_on_solved")}
    Z, DPI, PIC, XS = [], [], [], []
    DV, DEL = [], []
    cur_path, d = None, None
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for c in want:
        meta = cyc_meta[c]
        bud = int(meta["budget"])
        ms_slots = np.asarray(meta["ms_slots"], np.int64)
        path = _shard_for(files, c)
        if path is None:
            continue
        if path != cur_path:
            d, cur_path = np.load(path), path
        tm = np.nonzero(d["t_cycle"] == c)[0]
        cmk = np.nonzero(d["c_cycle"] == c)[0]
        assert tm.size and (np.diff(tm) == 1).all() and (np.diff(cmk) == 1).all()
        t_off, c_off = tm[0], cmk[0]
        st = d["t_step"][tm]
        ni = int(d["t_inst"][tm].max()) + 1
        w = {int(s): int((st == s).sum()) // ni for s in np.unique(st)}
        # tip row layout, asserted
        toff, base = {}, 0
        for s_ in range(bud + 1):
            toff[s_] = base
            idx = t_off + np.arange(base, base + ni * w[s_])
            assert (d["t_step"][idx] == s_).all()
            assert (d["t_inst"][idx] == np.repeat(np.arange(ni), w[s_])).all()
            assert (d["t_tip"][idx] == np.tile(np.arange(w[s_]), ni)).all()
            base += ni * w[s_]
        cs = d["c_step"][cmk]
        coff, cbase, kk = {}, 0, None
        for s_ in range(bud):
            k_ = int((cs == s_).sum()) // (ni * w[s_])
            kk = k_ if kk is None else kk
            assert k_ == kk, "kk varies within a cycle"
            coff[s_] = cbase
            idx = c_off + np.arange(cbase, cbase + ni * w[s_] * kk)
            assert (d["c_inst"][idx] == np.repeat(np.arange(ni), w[s_] * kk)).all()
            assert (d["c_parent"][idx] == np.tile(np.repeat(np.arange(w[s_]), kk), ni)).all()
            cbase += ni * w[s_] * kk

        # ---- descendants: n_desc and sum_succ, walked back from the graded tips ----
        n_tip = base
        n_desc = np.zeros(n_tip, np.int32)
        sum_succ = np.zeros(n_tip, np.float64)
        idxB = np.arange(toff[bud], toff[bud] + ni * w[bud])
        n_desc[idxB] = 1
        sum_succ[idxB] = (d["t_succ"][t_off + idxB] > 0.5).astype(np.float64)
        for s_ in range(bud, 0, -1):
            idx = np.arange(toff[s_], toff[s_] + ni * w[s_])
            par = d["t_parent"][t_off + idx].astype(np.int64)
            inst = np.repeat(np.arange(ni), w[s_])
            prow = toff[s_ - 1] + inst * w[s_ - 1] + par
            np.add.at(n_desc, prow, n_desc[idx])
            np.add.at(sum_succ, prow, sum_succ[idx])

        # ---- the agent's own move at each choice state: argmax over its scored children ----
        u_mv = np.zeros(n_tip, np.int64) - 1
        u_src = np.zeros(n_tip, np.int8) - 1
        u_kept = np.zeros(n_tip, bool)
        n_kept = np.zeros(n_tip, np.int16)
        for s_ in range(bud):
            m_ = ni * w[s_]
            idx = c_off + np.arange(coff[s_], coff[s_] + m_ * kk)
            sc = d["c_score"][idx].reshape(m_, kk)
            best = sc.argmax(1)
            rows = np.arange(m_)
            trow = toff[s_] + rows
            u_mv[trow] = d["c_mv"][idx].reshape(m_, kk)[rows, best]
            u_src[trow] = d["c_src"][idx].reshape(m_, kk)[rows, best]
            kept = (d["c_child"][idx].reshape(m_, kk) >= 0)
            u_kept[trow] = kept[rows, best]
            n_kept[trow] = kept.sum(1)

        # ---- THE DELIBERATION STATE: the moves the beam materialised, valued, and then did
        #      not keep. `c_score` is the learner's OWN value head applied to a child an
        #      observer cannot score, so this block is private by construction and is the one
        #      source in the matrix that is neither the update nor a public covariate. It has
        #      to be a fixed-width summary because `kk` is 5 on a steady routed cycle and 13 or
        #      21 on a forced one; everything is expressed RELATIVE to the parent's own `v_c`,
        #      which is what makes a score comparable across states.
        DELIB_D = 12
        delib = np.zeros((n_tip, DELIB_D), np.float32)
        for s_ in range(bud):
            m_ = ni * w[s_]
            idx = c_off + np.arange(coff[s_], coff[s_] + m_ * kk)
            sc = d["c_score"][idx].reshape(m_, kk).astype(np.float32)
            sr = d["c_src"][idx].reshape(m_, kk)
            kept = (d["c_child"][idx].reshape(m_, kk) >= 0)
            top = np.sort(sc, 1)[:, ::-1][:, :5]
            NEG = np.float32(-1e30)
            bp = np.where((sr == 1) | (sr == 0) | (sr == 2), sc, NEG).max(1)
            be = np.where(sr == 4, sc, NEG).max(1)
            has_e = be > NEG / 2
            has_p = bp > NEG / 2
            trow = toff[s_] + np.arange(m_)
            delib[trow, 0:5] = top
            delib[trow, 5] = top[:, 0] - top[:, 1]
            delib[trow, 6] = sc.mean(1)
            delib[trow, 7] = sc.std(1)
            delib[trow, 8] = np.log(kk)
            delib[trow, 9] = np.where(has_p, bp, 0.0) - np.where(has_e, be, 0.0)
            delib[trow, 10] = has_e.astype(np.float32)
            delib[trow, 11] = kept.sum(1)

        # ---- did the update's OWN pi supervision at this state name the move the agent
        #      ranked first? `prop_train` regresses pi at `s` onto the move the SOLVED
        #      trajectory took there, so this is the test of whether the arity-2 slot's content
        #      is the same move the update pushed toward. Not assumed -- measured, per row.
        u_on_solved = np.zeros(n_tip, bool)
        for s_ in range(1, bud + 1):
            idx = np.arange(toff[s_], toff[s_] + ni * w[s_])
            solved = sum_succ[idx] > 0
            par = d["t_parent"][t_off + idx].astype(np.int64)
            inst = np.repeat(np.arange(ni), w[s_])
            prow = toff[s_ - 1] + inst * w[s_ - 1] + par
            hit = solved & (d["t_mv"][t_off + idx].astype(np.int64) == u_mv[prow])
            u_on_solved[prow[hit]] = True

        # ---- the revision at every logged state, from the two fp32 snapshots ----
        sa, sb = load_heads(root, c), load_heads(root, c + 1)
        assert sa is not None and sb is not None, f"missing snapshot at cycle {c}"
        zz = d["t_z"][tm].astype(np.float32)
        rr = d["t_root"][tm]
        v_c = head_forward(sa, "value", zz, rr)
        dv = head_forward(sb, "value", zz, rr) - v_c
        pi_c = head_forward(sa, "prop", zz, rr)
        dpi = head_forward(sb, "prop", zz, rr) - pi_c

        # ---- select the choice states and (optionally) subsample within the cycle ----
        sel = np.arange(0, toff[bud])            # every tip at step < budget
        if rows_per_cycle and sel.size > rows_per_cycle:
            sel = np.sort(rng.choice(sel, rows_per_cycle, replace=False))
        step_of = np.concatenate([np.full(ni * w[s_], s_, np.int16) for s_ in range(bud)])
        cols["cycle"].append(np.full(sel.size, c, np.int16))
        cols["era"].append(np.full(sel.size, meta["era"], np.int8))
        cols["step"].append(step_of[sel])
        cols["inst"].append(d["t_inst"][t_off + sel])
        cols["tip"].append(d["t_tip"][t_off + sel])
        cols["root"].append(d["t_root"][t_off + sel])
        cols["u_slot"].append(ms_slots[u_mv[sel]].astype(np.int16))
        cols["u_src"].append(u_src[sel])
        cols["u_kept"].append(u_kept[sel])
        cols["n_kept"].append(n_kept[sel])
        cols["n_desc"].append(n_desc[sel])
        cols["sum_succ"].append(sum_succ[sel].astype(np.float32))
        cols["v_c"].append(v_c[sel].astype(np.float32))
        cols["kk"].append(np.full(sel.size, kk, np.int16))
        cols["u_on_solved"].append(u_on_solved[sel])
        DEL.append(delib[sel])
        Z.append(zz[sel])
        DV.append(dv[sel].astype(np.float32))
        DPI.append(dpi[sel].astype(np.float32))
        PIC.append(pi_c[sel].astype(np.float32))
        XS.append(d["t_x"][t_off + sel])
        if verbose and c % 20 == 0:
            print(f"    ...cycle {c} ({time.time() - t0:.0f}s)", flush=True)

    tab = {k: np.concatenate(v) for k, v in cols.items()}
    tab["z"] = np.concatenate(Z)
    tab["dv"] = np.concatenate(DV)
    tab["dpi"] = np.concatenate(DPI)
    tab["pi_c"] = np.concatenate(PIC)
    tab["x"] = np.concatenate(XS)
    tab["delib"] = np.concatenate(DEL)
    tab["n_slots"] = np.int64(n_slots)
    # per-cycle available-slot mask, as a (n_cycles_present, n_slots) bool table
    ucyc = np.unique(tab["cycle"])
    av = np.zeros((ucyc.size, n_slots), bool)
    for i, c in enumerate(ucyc):
        av[i, np.asarray(cyc_meta[int(c)]["avail_slots"], np.int64)] = True
    tab["avail_cycles"] = ucyc
    tab["avail"] = av
    return tab


# ======================================================================
# (2) TARGETS
# ======================================================================

def make_targets(tab):
    """The two ROADMAP targets plus the wiring diagnostics they have to be read against.

    verr   -- the value head's error against the grade. `ybar - sigmoid(v_c)` at the state, which
              is exactly the per-row BCE gradient `value_steps` applied there (the head is trained
              with `binary_cross_entropy_with_logits` against the trajectory's terminal `succ`).
              PUBLIC: the grade is the arm's own paid-for oracle readout. Defined only where the
              state entered the buffer at all (`n_desc > 0`).
    dprop  -- the change in pi's PROPOSAL at that state: the log-softmax over the cycle's LIVE
              slots, at the move the agent chose, after minus before. The learner's own quantity.
    dv     -- the raw value revision. A WIRING CELL, not a finding: it is a coordinate of the
              source `delta` by construction, and it is here so the tautological corner of the
              matrix is visible instead of being mistaken for a result.
    dprop_max -- the change in pi's own confidence (max live log-prob), move-independent.
    """
    n_slots = int(tab["n_slots"])
    cyc_idx = np.searchsorted(tab["avail_cycles"], tab["cycle"])
    live = tab["avail"][cyc_idx]                                    # (n, n_slots) bool
    neg = -1e30
    pa = np.where(live, tab["pi_c"], neg)
    pb = np.where(live, tab["pi_c"] + tab["dpi"], neg)
    lsa = pa - _logsumexp(pa)
    lsb = pb - _logsumexp(pb)
    rows = np.arange(tab["cycle"].size)
    u = tab["u_slot"].astype(np.int64)
    dprop = (lsb[rows, u] - lsa[rows, u]).astype(np.float64)
    dprop_max = (lsb.max(1) - lsa.max(1)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ybar = tab["sum_succ"] / np.maximum(tab["n_desc"], 1)
    ok = tab["n_desc"] > 0
    verr = np.where(ok, ybar - _sigmoid(tab["v_c"]), np.nan).astype(np.float64)
    return {"verr": (verr, ok), "dprop": (dprop, np.ones_like(ok)),
            "dv": (tab["dv"].astype(np.float64), np.ones_like(ok)),
            "dprop_max": (dprop_max, np.ones_like(ok))}


def _logsumexp(a):
    m = a.max(1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(1, keepdims=True))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ======================================================================
# (3) THE MATCHED PAIR
# ======================================================================

def fm_inputs(tab, slot):
    """[z, onehot(root), onehot(slot)]. `slot` is the u_slot vector for FM2 and MASK for FM1."""
    n = tab["cycle"].size
    ns = int(tab["n_slots"])
    nroot = int(tab["root"].max()) + 1
    Xs = np.zeros((n, ns + 1), np.float32)
    Xs[np.arange(n), np.asarray(slot, np.int64)] = 1.0
    Xr = np.zeros((n, nroot), np.float32)
    Xr[np.arange(n), tab["root"].astype(np.int64)] = 1.0
    return np.concatenate([tab["z"].astype(np.float32), Xr, Xs], 1)


def train_pair(Xz, Xr, u_slot, Y, tr, *, hidden, seed, steps, batch, lr, mask_slot):
    """The donor's Gate C pair, in one function so the match is structural. Same init seed
    (asserted), one loop, one batch stream, one optimiser recipe: ONLY the slot's content
    differs."""
    n, ns1 = Xz.shape[0], mask_slot + 1
    d_in = Xz.shape[1] + Xr.shape[1] + ns1
    fm1, fm2 = MLP(d_in, hidden, Y.shape[1], seed), MLP(d_in, hidden, Y.shape[1], seed)
    for a, b in zip(fm1.params(), fm2.params()):
        assert np.array_equal(a, b), "pair inits diverged"
    o1, o2 = AdamW(fm1.params(), lr=lr), AdamW(fm2.params(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    base = np.concatenate([Xz, Xr], 1)
    for _ in range(steps):
        i = tr[rng.integers(0, tr.size, batch)]
        S2 = np.zeros((batch, ns1), np.float32)
        S2[np.arange(batch), u_slot[i]] = 1.0
        S1 = np.zeros((batch, ns1), np.float32)
        S1[:, mask_slot] = 1.0
        for net, opt, S in ((fm1, o1, S1), (fm2, o2, S2)):
            Xb = np.concatenate([base[i], S], 1)
            out, cx = net.forward(Xb, cache=True)
            d = ((out - Y[i]) * (2.0 / Y.shape[1])).astype(np.float32)
            opt.step(net.backward(cx, d))
    return fm1, fm2


def fm_predict(net, Xz, Xr, slot, mask_slot, chunk=65536):
    n = Xz.shape[0]
    out = []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        S = np.zeros((sl.stop - sl.start, mask_slot + 1), np.float32)
        if np.isscalar(slot):
            S[:, int(slot)] = 1.0
        else:
            S[np.arange(sl.stop - sl.start), np.asarray(slot[sl], np.int64)] = 1.0
        out.append(net.forward(np.concatenate([Xz[sl], Xr[sl], S], 1)))
    return np.concatenate(out)


# ======================================================================
# (4) DECODE
# ======================================================================

def standardise_(X, tr):
    """`_standardise` in place, for the observer inputs (a 579-column one-hot block over 180K
    rows is 420 MB and a second copy is not worth having)."""
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True)
    np.subtract(X, mu, out=X)
    np.divide(X, np.maximum(sd, 1e-6), out=X)
    return X


def decode(X, tgt, tr, te, *, caps, seed, steps, batch, lr, standardise=True, groups=None):
    """All targets for one source, at every readout capacity.

    ONE standardisation, from ALL probe-train rows, shared by every target and by the observer
    twins -- the donor grouped it per validity mask for efficiency; sharing it here is what makes
    the self side and the observer side literally the same readout stack applied to different
    inputs, which is the comparison the ladder is for. Train-only either way, so nothing leaks.
    `standardise=False` is for a caller that has already done it in place (the half-data budget
    control re-uses the FULL-data statistics on purpose: re-standardising on half the rows makes
    a one-hot column that happens to be constant in that half explode, which is a bug about the
    standardiser and not a fact about the budget).
    """
    Xs = _standardise(X, tr) if standardise else X
    out = {}
    for hid in caps:
        cell = {}
        for tname, (y, mask) in tgt.items():
            tr_ = tr[mask[tr]]
            te_ = te[mask[te]]
            if tr_.size < 200 or te_.size < 200:
                continue
            ym, ys = float(y[tr_].mean()), float(y[tr_].std()) + 1e-12
            if hid == 0:
                # lambda is chosen on a GROUP-held-in slice, arranged to sit last
                if groups is None:
                    order, cut = tr_, None
                else:
                    a, b = _group_split(tr_, groups, 0.1, seed + 5000)
                    order, cut = np.concatenate([a, b]), a.size
                yhat, lam = _ridge(Xs[order].astype(np.float64), (y[order] - ym) / ys,
                                   Xs[te_].astype(np.float64), cut=cut)
                yhat = yhat * ys + ym
            else:
                net = MLP(X.shape[1], hid, 1, seed + 5000 + hid)
                Ystd = ((y - ym) / ys).astype(np.float32)[:, None]
                train_mlp(net, Xs, Ystd, tr_, steps=steps, batch=batch, lr=lr,
                          seed=seed + 5001 + hid, groups=groups)
                yhat = predict(net, Xs[te_]).ravel().astype(np.float64) * ys + ym
                lam = None
            cell[tname] = {"r2": _r2_np(y[te_], yhat), "n_test": int(te_.size)}
            if lam is not None:
                cell[tname]["lam"] = lam
        out[f"h{hid}"] = cell
    return out


def obs_inputs(tab, rung, n_slots):
    """PUBLIC information only, in the ladder's three rungs. `x` is the configuration the world
    handed the agent -- the same object `z` is a frozen encoding of -- so the observer has to
    learn its own encoder, which is exactly the reconstruction cost paper 2 measures."""
    n = tab["cycle"].size
    T = tab["x"].shape[1]
    vx = int(tab["x"].max()) + 1
    X1 = np.zeros((n, T * vx), np.float32)
    X1[np.arange(n)[:, None], np.arange(T)[None, :] * vx + tab["x"].astype(np.int64)] = 1.0
    nroot = int(tab["root"].max()) + 1
    Xr = np.zeros((n, nroot), np.float32)
    Xr[np.arange(n), tab["root"].astype(np.int64)] = 1.0
    parts = [X1, Xr]
    if rung >= 1:
        Xa = np.zeros((n, n_slots), np.float32)
        Xa[np.arange(n), tab["u_slot"].astype(np.int64)] = 1.0
        parts.append(Xa)
    if rung >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            ybar = tab["sum_succ"] / np.maximum(tab["n_desc"], 1)
        parts.append(np.stack([ybar, np.log1p(tab["n_desc"]).astype(np.float32),
                               (tab["n_desc"] > 0).astype(np.float32)], 1).astype(np.float32))
    return np.concatenate(parts, 1)


# ======================================================================
# (5) THE RUN
# ======================================================================

def _fmt(v, w=6, p=3):
    return ("{:>%d.%df}" % (w, p)).format(v) if v == v else " " * (w - 3) + "nan"


def _cos(a, b):
    return float(np.mean((a * b).sum(1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="au_s0")
    ap.add_argument("--arm", default="anchor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows-per-cycle", type=int, default=0, help="0 = every choice state")
    ap.add_argument("--fm-rows", type=int, default=300_000, help="FM training rows (subsampled)")
    ap.add_argument("--fm-steps", type=int, default=6000)
    ap.add_argument("--fm-caps", default="64,256,1024")
    ap.add_argument("--probe-caps", default="0,16,64,256")
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--obs-caps", default="0,64,256")
    ap.add_argument("--obs-steps", type=int, default=3000)
    ap.add_argument("--ens", type=int, default=3)
    ap.add_argument("--ladder", action="store_true",
                    help="run ONLY the conditioning-completion ladder (section 7) and write "
                         "ladder.txt; the default path is untouched")
    ap.add_argument("--e1b", action="store_true",
                    help="run ONLY the E1b per-datum refit (section 8) and write reduction.txt "
                         "under figures/<tag>/fit/; the phase-2 and ladder paths are untouched")
    ap.add_argument("--e1b-no-cycle-row", action="store_true",
                    help="skip the whole-cycle comparability row (it rebuilds a phase-2 table)")
    ap.add_argument("--e1b-reencode", action="store_true",
                    help="run ONLY the symmetric re-encode check on finding 8(i) (section 9) "
                         "and write reencode.txt/json under figures/<tag>/fit/; the phase-2, "
                         "ladder and --e1b paths are untouched")
    ap.add_argument("--proj-k", type=int, default=4,
                    help="width of the fixed z-projection used by the rung-4 batch encoding")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.fm_steps, args.probe_steps, args.obs_steps = 400, 300, 300
        args.fm_caps, args.probe_caps, args.obs_caps = "64", "0,64", "0,64"
        args.ens, args.rows_per_cycle, args.fm_rows = 2, 512, 20_000

    root = os.path.join(FIG, args.tag, args.arm)
    # the smoke writes somewhere else: it is a wiring pass at 512 rows/cycle and 400 FM steps,
    # and letting it land on the real reduction would silently replace a result with a check.
    outdir = os.path.join(FIG, args.tag, "fit_smoke" if args.smoke else "fit")
    os.makedirs(outdir, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    t_start = time.time()
    assert _gradcheck(), "analytic gradients disagree with finite differences"

    # ---------------- E1b: a disjoint mode reading the revision tables ----------------
    if args.e1b:
        res = e1b_refit(args, root, outdir, out)
        out("")
        out(f"END (E1b) - {time.time() - t_start:.0f}s total")
        with open(os.path.join(outdir, "reduction.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(outdir, "e1b.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"\nwrote {outdir}/reduction.txt and e1b.json", flush=True)
        return

    # -------- the symmetric re-encode check: a fourth disjoint mode, section 9 --------
    if args.e1b_reencode:
        res = e1b_reencode(args, root, outdir, out)
        out("")
        out(f"END (re-encode) - {time.time() - t_start:.0f}s total")
        with open(os.path.join(outdir, "reencode.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(outdir, "reencode.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"\nwrote {outdir}/reencode.txt and reencode.json", flush=True)
        return

    # ---------------- the table ----------------
    # `v2` is the SCHEMA key, not a run key: bump it whenever `build_table`'s columns change,
    # or a stale cache is loaded silently and the new columns are simply absent.
    os.makedirs(os.path.join(FIG, args.tag, "fit"), exist_ok=True)
    cache = os.path.join(FIG, args.tag, "fit", f"table_v2_r{args.rows_per_cycle}.npz")
    if os.path.isfile(cache) and not args.no_cache and not args.smoke:
        tab = dict(np.load(cache))
        print(f"  loaded cached table {cache}", flush=True)
    else:
        print("  building the per-datum table off the fetched bytes...", flush=True)
        tab = build_table(root, rows_per_cycle=args.rows_per_cycle, seed=args.seed)
        if not args.smoke:
            np.savez(cache, **tab)
    n = tab["cycle"].size
    n_slots = int(tab["n_slots"])
    mask_slot = n_slots                       # the learned MASK id, exactly as SlotFM's `v`
    cycles = np.unique(tab["cycle"])

    out("AUDIATION E1 PHASE 2 - the arity-2 self-model on practice"
        + ("  [--ladder: SECTION 7 ONLY; the decode matrix is in reduction.txt]"
           if args.ladder else ""))
    out(f"  tag={args.tag} arm={args.arm} seed={args.seed}")
    out(f"  cycles {int(cycles.min())}..{int(cycles.max())} ({cycles.size}); "
        f"choice-state rows {n}; slots {n_slots}")
    out("")
    out("=" * 78)
    out("(0) THE DATUM")
    out("=" * 78)
    import collections
    sc = collections.Counter(tab["u_src"].tolist())
    out("  the agent's own move at a choice state = the argmax-scored child. Its provenance:")
    for k in sorted(sc):
        out(f"    {SRC_NAMES.get(k, k):11s} {sc[k]:>9d}  {sc[k] / n:6.2%}")
    vb = tab["n_desc"] > 0
    out(f"  argmax child survived the topk        : {tab['u_kept'].mean():6.2%}")
    out(f"  kept children per choice state (mean) : {tab['n_kept'].mean():.3f}")
    out(f"  entered the VALUE buffer (n_desc>0)   : {vb.mean():6.2%}  "
        f"({int(vb.sum())} rows; mean multiplicity {tab['n_desc'][vb].mean():.1f})")
    out(f"  has a SOLVED descendant               : {(tab['sum_succ'] > 0).mean():6.2%}")
    out(f"  the move pi was then trained toward at this state IS the agent's argmax:")
    out(f"    all rows {tab['u_on_solved'].mean():6.2%} | value-buffer rows "
        f"{tab['u_on_solved'][vb].mean():6.2%}")

    # ---------------- targets ----------------
    tg_all = make_targets(tab)
    ybar = np.where(vb, tab["sum_succ"] / np.maximum(tab["n_desc"], 1), np.nan)
    out(f"  grade ybar on buffer rows: mean {np.nanmean(ybar):.4f}; "
        f"sigmoid(v_c) mean {_sigmoid(tab['v_c'][vb]).mean():.4f}")

    # ---------------- splits ----------------
    #  FM split by CYCLE (see the module docstring); probe split by INSTANCE inside the FM's
    #  test cycles.
    test_cyc = cycles[3::4]
    train_cyc = np.setdiff1d(cycles, test_cyc)
    is_test = np.isin(tab["cycle"], test_cyc)
    dec_rows = np.nonzero(is_test)[0]
    fm_pool = np.nonzero(~is_test)[0]
    rng = np.random.default_rng(args.seed + 7)
    fm_tr = np.sort(rng.choice(fm_pool, min(args.fm_rows, fm_pool.size), replace=False))
    inst = tab["inst"][dec_rows]
    ptr = np.nonzero(inst % 4 != 3)[0]
    pte = np.nonzero(inst % 4 == 3)[0]
    out("")
    out(f"  FM split (by cycle):   {train_cyc.size} train / {test_cyc.size} held-out cycles; "
        f"FM fit on {fm_tr.size} rows")
    out(f"  decode set = the held-out cycles' rows: {dec_rows.size} "
        f"({ptr.size} probe-train / {pte.size} probe-test, split by instance)")
    out(f"  eras present in test cycles: "
        f"{sorted(set(tab['era'][dec_rows].tolist()))}")

    # ---------------- the FM's inputs and target ----------------
    zs_mu = tab["z"][fm_tr].mean(0, keepdims=True)
    zs_sd = np.maximum(tab["z"][fm_tr].std(0, keepdims=True), 1e-6)
    Xz = ((tab["z"] - zs_mu) / zs_sd).astype(np.float32)
    nroot = int(tab["root"].max()) + 1
    Xr = np.zeros((n, nroot), np.float32)
    Xr[np.arange(n), tab["root"].astype(np.int64)] = 1.0
    D = np.concatenate([tab["dv"][:, None], tab["dpi"]], 1).astype(np.float32)
    dmu = D[fm_tr].mean(0, keepdims=True)
    dsd = np.maximum(D[fm_tr].std(0, keepdims=True), 1e-6)
    Ds = ((D - dmu) / dsd).astype(np.float32)          # the FM's target, z-scored on train rows
    del D
    u_slot = tab["u_slot"].astype(np.int64)

    head_params = 0
    sd0 = load_heads(root, int(cycles.min()))
    if sd0 is not None:
        head_params = int(sum(v.size for v in sd0.values()))

    # ---------------- strata and the residualised targets ----------------
    cyc_i = np.searchsorted(cycles, tab["cycle"][dec_rows])
    strata = (cyc_i * 16 + tab["step"][dec_rows]).astype(np.int64)
    #  THE GROUPING every held-in split inside the probe-train rows must respect. One group is
    #  one (cycle, instance) -- one corrupted start and the whole beam that grew out of it. The
    #  train/test split is already by instance; `_group_split` says what goes wrong when the
    #  splits inside it are not.
    grp = (cyc_i.astype(np.int64) * 4096 + tab["inst"][dec_rows].astype(np.int64))
    tgt = {}
    for name in ("verr", "dprop", "dv", "dprop_max"):
        y_all, m_all = tg_all[name]
        y, m = y_all[dec_rows], m_all[dec_rows].astype(bool)
        y = np.where(m, y, 0.0)
        tgt[name] = (y, m)
        tgt[name + "_cs"] = (_stratum_residualise(y, strata, ptr[m[ptr]]), m)
    tgt["dprop_vb"] = (tgt["dprop"][0], tgt["verr"][1])
    tgt["dprop_vb_cs"] = (tgt["dprop_cs"][0], tgt["verr"][1])
    PRIM = ["verr_cs", "dprop_cs", "verr", "dprop"]
    DIAG = ["dv_cs", "dprop_max_cs", "dprop_vb_cs"]
    tgt_prim = {k: tgt[k] for k in PRIM}
    tgt_diag = {k: tgt[k] for k in DIAG}

    out("")
    out("  headroom after residualisation (frac of Var(target) left on probe-test rows):")
    for name in ("verr", "dprop", "dv", "dprop_max"):
        y, m = tgt[name]
        yc = tgt[name + "_cs"][0]
        sel = pte[m[pte]]
        v0 = float(np.var(y[sel]))
        out(f"    {name:10s} raw var {v0:9.5f} -> cycle x step residualised "
            f"{float(np.var(yc[sel])) / max(v0, 1e-12):6.3f}   "
            f"[{np.unique(strata).size} strata, "
            f"{ptr.size / max(np.unique(strata[ptr]).size, 1):.0f} probe-train rows each]")

    # ---------------- the conditioning-completion ladder, and nothing else ----------------
    if args.ladder:
        conditioning_ladder(args, root, outdir, tab, cycles, dec_rows, fm_tr, ptr, pte, grp,
                            cyc_i, Xz, Xr, Ds, u_slot, mask_slot, tgt_prim, PRIM, out)
        out("")
        out(f"END (ladder) - {time.time() - t_start:.0f}s total")
        with open(os.path.join(outdir, "ladder.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {outdir}/ladder.txt and ladder.json", flush=True)
        return

    # ---------------- the matched pair, per instrument capacity ----------------
    out("")
    out("=" * 78)
    out("(1) THE MATCHED PAIR  (FM2 gets the slot of the agent's own move; FM1 a MASK id)")
    out("=" * 78)
    fm_caps = [int(q) for q in args.fm_caps.split(",")]
    pcaps = [int(q) for q in args.probe_caps.split(",")]
    pcaps_red = [q for q in pcaps if q in (0, 64)]
    d_in = Xz.shape[1] + nroot + mask_slot + 1
    by_cap, sources_default = {}, None
    for ci, hid in enumerate(fm_caps):
        t0 = time.time()
        fm1, fm2 = train_pair(Xz, Xr, u_slot, Ds, fm_tr, hidden=hid, seed=args.seed + 4441,
                              steps=args.fm_steps, batch=512, lr=1e-3, mask_slot=mask_slot)
        p2 = fm_predict(fm2, Xz[dec_rows], Xr[dec_rows], u_slot[dec_rows], mask_slot)
        p1 = fm_predict(fm1, Xz[dec_rows], Xr[dec_rows], mask_slot, mask_slot)
        dd = Ds[dec_rows]
        r = dd - p2
        r1 = dd - p1
        gapv = p2 - p1
        nparam = sum(q.size for q in fm2.params())
        cosd = _cos(p2, dd)
        rel = float(np.linalg.norm(r, axis=1).mean() / np.linalg.norm(dd, axis=1).mean())
        # the same two numbers on the FM's OWN training cycles. Without this pair a null in the
        # matrix is ambiguous between `the update is unforecastable` and `the FM did not fit`,
        # and those have opposite readings.
        tr_sub = fm_tr[::7]
        p2tr = fm_predict(fm2, Xz[tr_sub], Xr[tr_sub], u_slot[tr_sub], mask_slot)
        cos_in = _cos(p2tr, Ds[tr_sub])
        rel_in = float(np.linalg.norm(Ds[tr_sub] - p2tr, axis=1).mean()
                       / np.linalg.norm(Ds[tr_sub], axis=1).mean())
        del p2tr
        gnorm = np.linalg.norm(gapv, axis=1)
        # the ensemble guard, at the default capacity only (independent seeds, own data stream)
        enc = float("nan")
        if ci == 0 and args.ens > 1:
            res = [r]
            for j in range(1, args.ens):
                a1, a2 = train_pair(Xz, Xr, u_slot, Ds, fm_tr, hidden=hid,
                                    seed=args.seed + 4441 + 977 * j, steps=args.fm_steps,
                                    batch=512, lr=1e-3, mask_slot=mask_slot)
                res.append(dd - fm_predict(a2, Xz[dec_rows], Xr[dec_rows], u_slot[dec_rows],
                                           mask_slot))
                del a1, a2
            enc = _ensemble_cos(res)
            del res
        out(f"  FM h{hid:<5d} {nparam / 1e3:6.1f}K params "
            f"({100 * nparam / max(head_params, 1):5.1f}% of the two heads' {head_params / 1e3:.0f}K)"
            f"  held-out cos(p2,delta)={cosd:+.3f} |r|/|delta|={rel:.3f}"
            f" | in-sample cos={cos_in:+.3f} |r|/|delta|={rel_in:.3f}"
            f" | |g|/|delta|={float(gnorm.mean() / np.linalg.norm(dd, axis=1).mean()):.3f}"
            f"  ens_cos={enc:.3f}  [{time.time() - t0:.0f}s]")
        by_cap[f"h{hid}"] = {"params": nparam, "pct_of_heads": 100 * nparam / max(head_params, 1),
                             "cos_p2_delta": cosd, "rel_r": rel, "ens_cos": enc,
                             "cos_in_sample": cos_in, "rel_r_in_sample": rel_in,
                             "g_mean": float(gnorm.mean())}
        if ci == 0:
            sources_default = {"r": r, "delta": dd, "gap": gapv, "r1": r1, "p2": p2}
            gnorm0 = gnorm
        else:
            # capacity invariance: the crux sources only, reduced probe ladder
            crux = {"r": r, "delta": dd, "gap": gapv}
            by_cap[f"h{hid}"]["decode"] = {
                s: decode(X, tgt_prim, ptr, pte, caps=pcaps_red, seed=args.seed,
                          steps=args.probe_steps, batch=512, lr=1e-3, groups=grp)
                for s, X in crux.items()}
            del crux
        del fm1, fm2, p1, p2, r, r1, gapv

    # ---------------- (1b) how forecastable is the update AT ALL ----------------
    #  A WITHIN-CYCLE pair: trained on the other instances of the SAME cycles it is scored on.
    #  This is NOT a valid forecast -- it has seen the cycle's own update, so it violates the
    #  timing the whole construction is about -- but it is the upper bound the honest split has
    #  to be read against. Reported, never used as a source.
    wc_tr = np.sort(rng.choice(dec_rows[ptr], min(args.fm_rows, ptr.size), replace=False))
    w1, w2 = train_pair(Xz, Xr, u_slot, Ds, wc_tr, hidden=fm_caps[0], seed=args.seed + 4441,
                        steps=args.fm_steps, batch=512, lr=1e-3, mask_slot=mask_slot)
    wp2 = fm_predict(w2, Xz[dec_rows][pte], Xr[dec_rows][pte], u_slot[dec_rows][pte], mask_slot)
    wp1 = fm_predict(w1, Xz[dec_rows][pte], Xr[dec_rows][pte], mask_slot, mask_slot)
    wdd = Ds[dec_rows][pte]
    within = {"cos_p2_delta": _cos(wp2, wdd),
              "rel_r": float(np.linalg.norm(wdd - wp2, axis=1).mean()
                             / np.linalg.norm(wdd, axis=1).mean()),
              "g_rel": float(np.linalg.norm(wp2 - wp1, axis=1).mean()
                             / np.linalg.norm(wdd, axis=1).mean())}
    out(f"  WITHIN-CYCLE upper bound (an FM that HAS seen the cycle's own update, on other "
        f"instances):")
    out(f"    cos(p2,delta)={within['cos_p2_delta']:+.3f}  |r|/|delta|={within['rel_r']:.3f}  "
        f"|g|/|delta|={within['g_rel']:.3f}   -- not a forecast, the ceiling to read the "
        f"honest split against")
    del w1, w2, wp1, wp2, wdd

    # ---------------- the sources ----------------
    dd = Ds[dec_rows]
    src = dict(sources_default)
    #  TWO REFERENCE ROWS, not self-model sources. `z_state` is what both heads read, so it is
    #  the floor for "was this already in the state before any update"; `readout0` is the
    #  learner's own PRE-update readout at s, which is the other half of `verr` (the grade being
    #  the public half) and therefore the row that says how much of a decode is the revision at
    #  all rather than the standing readout.
    src["z_state"] = Xz[dec_rows]
    src["readout0"] = np.concatenate([tab["v_c"][dec_rows, None], tab["pi_c"][dec_rows]],
                                     1).astype(np.float32)
    #  THE CONTROL THE CRUX ACTUALLY NEEDS. `r = Delta - FM2(s,u)` carries `u` inside it, and
    #  `dprop` is a u-INDEXED coordinate of Delta_pi -- so `r` could beat `delta` for no reason
    #  better than telling the probe which coordinate to read. `delta_u` is the raw update WITH
    #  the efference copy stapled on and no forecast anywhere: if `r` does not beat it, forming
    #  the residual is doing nothing that concatenating the action does not already do. `u` is
    #  public (it is the observer ladder's second rung), so this control costs the self side
    #  nothing it was entitled to.
    slot_oh = np.zeros((dec_rows.size, mask_slot + 1), np.float32)
    slot_oh[np.arange(dec_rows.size), u_slot[dec_rows]] = 1.0
    src["delta_u"] = np.concatenate([dd, slot_oh], 1)
    src["u_only"] = slot_oh
    #  THE CALENDAR-ONLY FORECAST'S RESIDUAL. The update is one batched step per cycle, so most
    #  of what a forecaster could know about Delta(s) is `which cycle is this` -- a quantity the
    #  self-model is structurally denied (it reads the state, and the state does not name the
    #  cycle) and an observer is not. `r_cyc = Delta - E[Delta | cycle]`, cycle means from
    #  PROBE-TRAIN rows only, is what a forecaster that knew nothing but the calendar would
    #  leave behind, and it is the baseline the learned forecast has to beat before `r` can be
    #  said to do work `Delta` does not.
    cmu = np.zeros((cycles.size, dd.shape[1]), np.float64)
    ccnt = np.zeros(cycles.size)
    np.add.at(cmu, cyc_i[ptr], dd[ptr].astype(np.float64))
    np.add.at(ccnt, cyc_i[ptr], 1.0)
    cmu = np.where(ccnt[:, None] > 0, cmu / np.maximum(ccnt, 1)[:, None],
                   dd[ptr].mean(0, keepdims=True))
    src["r_cyc"] = (dd - cmu[cyc_i]).astype(np.float32)
    #  THE DELIBERATION STATE, the one PRIVATE source that is not the update. `c_score` is the
    #  learner's own value head applied to children an observer cannot score, so if anything in
    #  this matrix is going to beat the observer ladder on a target the observer holds no
    #  ingredient for, this row is where it would show.
    src["delib"] = np.concatenate(
        [tab["delib"][dec_rows][:, :8] - tab["v_c"][dec_rows, None] * np.array(
            [1, 1, 1, 1, 1, 0, 1, 0], np.float32),
         tab["delib"][dec_rows][:, 8:]], 1).astype(np.float32)
    era_oh = np.zeros((dec_rows.size, 6), np.float32)
    era_oh[np.arange(dec_rows.size), tab["era"][dec_rows].astype(np.int64)] = 1.0
    step_oh = np.zeros((dec_rows.size, 16), np.float32)
    step_oh[np.arange(dec_rows.size), tab["step"][dec_rows].astype(np.int64)] = 1.0
    src["calendar"] = np.concatenate([era_oh, step_oh,
                                      (cyc_i / cycles.size).astype(np.float32)[:, None]], 1)
    scal = {"|r|": np.linalg.norm(src["r"], axis=1),
            "|delta|": np.linalg.norm(dd, axis=1),
            "g": gnorm0}
    for k, v in scal.items():
        src[k] = v.astype(np.float32)[:, None]
    # the position/stratum-preserving null: r permuted WITHIN cycle x step
    perm = np.arange(dec_rows.size)
    rsh = np.random.default_rng(args.seed + 99)
    for s_ in np.unique(strata):
        idx = np.nonzero(strata == s_)[0]
        perm[idx] = idx[rsh.permutation(idx.size)]
    src["r_shuffled"] = src["r"][perm]

    out("")
    out("=" * 78)
    out("(2) THE DECODE MATRIX   R^2 out of sample, probe-test rows of held-out cycles")
    out("=" * 78)
    out("  content sources at the full readout ladder; scalar norms, the calendar and the")
    out("  shuffled null at the reduced one. `_cs` = residualised on cycle x step.")
    CONTENT = ["r", "delta", "delta_u", "r_cyc", "gap", "r1", "p2", "delib", "z_state",
               "readout0"]
    out("")
    out("  WHAT A FORECAST OF THE UPDATE COULD KNOW. The update is ONE batched step per cycle,")
    out("  so the first question is how much of Delta(s) is `which cycle is this` rather than")
    out("  `which state is this`, and the self-model is structurally denied the former.")
    out(f"    eta^2(Delta ~ cycle) on the decode rows        = {_eta2(dd, cyc_i):.3f}")
    out(f"    |Delta - E[Delta|cycle]| / |Delta|             = "
        f"{float(np.linalg.norm(src['r_cyc'], axis=1).mean() / np.linalg.norm(dd, axis=1).mean()):.3f}"
        f"   (a calendar-only forecaster)")
    out(f"    |r| / |Delta|, the learned arity-2 forecaster  = "
        f"{by_cap[f'h{fm_caps[0]}']['rel_r']:.3f}")
    dec = {}
    for sname, X in src.items():
        caps_here = pcaps if sname in CONTENT else pcaps_red
        t0 = time.time()
        dec[sname] = decode(X, tgt_prim, ptr, pte, caps=caps_here, seed=args.seed,
                            steps=args.probe_steps, batch=512, lr=1e-3, groups=grp)
        best = {t: max(dec[sname][f"h{h}"].get(t, {"r2": float("nan")})["r2"]
                       for h in caps_here) for t in PRIM}
        out(f"    [{sname:11s} d={X.shape[1]:<4d}] " +
            "  ".join(f"{t}={_fmt(best[t])}" for t in PRIM) + f"   [{time.time() - t0:.0f}s]")
    # the diagnostics, default FM capacity, reduced ladder
    dec_diag = {}
    for sname in ("r", "delta", "delta_u", "gap", "delib", "z_state"):
        dec_diag[sname] = decode(src[sname], tgt_diag, ptr, pte, caps=pcaps_red,
                                 seed=args.seed, steps=args.probe_steps, batch=512, lr=1e-3,
                                 groups=grp)
    out("  diagnostics (wiring cells; `dv` is a coordinate of `delta` by construction):")
    for sname, cell in dec_diag.items():
        best = {t: max(cell[f"h{h}"].get(t, {"r2": float("nan")})["r2"] for h in pcaps_red)
                for t in DIAG}
        out(f"    [{sname:11s}] " + "  ".join(f"{t}={_fmt(best[t])}" for t in DIAG))

    out("")
    out("  the ladder in full (rows = source, cols = capacity), per target:")
    for t in PRIM:
        out(f"    --- {t} ---")
        for sname in src:
            caps_here = pcaps if sname in CONTENT else pcaps_red
            row = "  ".join(f"h{h}={_fmt(dec[sname][f'h{h}'].get(t, {'r2': float('nan')})['r2'])}"
                            for h in caps_here)
            out(f"      {sname:12s} {row}")

    # ---------------- agency ----------------
    out("")
    out("=" * 78)
    out("(3) AGENCY  -- the arity gap and the decode, by the provenance of the chosen move")
    out("=" * 78)
    prov = tab["u_src"][dec_rows]
    dn = np.linalg.norm(dd, axis=1)
    agency = {}
    out(f"    {'class':12s} {'n':>8s} {'|g|':>8s} {'|g|/|d|':>8s} {'|r|':>8s} {'|r1|':>8s} "
        f"{'|d|':>8s} {'E[dprop]':>9s} {'E[verr]':>9s}")
    r1n = np.linalg.norm(src["r1"], axis=1)
    rn = np.linalg.norm(src["r"], axis=1)
    for k in sorted(set(prov.tolist())):
        m = prov == k
        mv = m & tgt["verr"][1]
        agency[SRC_NAMES.get(k, str(k))] = {
            "n": int(m.sum()), "g": float(gnorm0[m].mean()),
            "g_rel": float((gnorm0[m] / np.maximum(dn[m], 1e-9)).mean()),
            "r": float(rn[m].mean()), "r1": float(r1n[m].mean()), "d": float(dn[m].mean()),
            "mean_dprop": float(tgt["dprop"][0][m].mean()),
            "mean_verr": float(tgt["verr"][0][mv].mean()) if mv.sum() else float("nan")}
        a = agency[SRC_NAMES.get(k, str(k))]
        out(f"    {SRC_NAMES.get(k, k):12s} {a['n']:>8d} {a['g']:>8.3f} {a['g_rel']:>8.3f} "
            f"{a['r']:>8.3f} {a['r1']:>8.3f} {a['d']:>8.3f} {a['mean_dprop']:>+9.4f} "
            f"{a['mean_verr']:>+9.4f}")
    out("  |r1| - |r| is what the efference copy buys the forecast; g is the same thing as a")
    out("  forecast displacement rather than an error reduction.")

    # decode conditioned on provenance, at MATCHED n
    out("")
    out("  decode conditioned on the provenance of the chosen move (matched n, h0 and h64):")
    prov_dec = {}
    for k in (1, 4):
        if (prov == k).sum() < 2000:
            continue
        keep = np.zeros(dec_rows.size, bool)
        keep[prov == k] = True
        n_match = int(min((prov == 1).sum(), (prov == 4).sum()))
        idx = np.nonzero(keep)[0]
        idx = np.sort(np.random.default_rng(args.seed + 5).choice(idx, n_match, replace=False))
        km = np.zeros(dec_rows.size, bool)
        km[idx] = True
        sub_t = {t: (tgt_prim[t][0], tgt_prim[t][1] & km) for t in PRIM}
        prov_dec[SRC_NAMES[k]] = {
            s: decode(src[s], sub_t, ptr, pte, caps=[0, 64], seed=args.seed,
                      steps=args.probe_steps, batch=512, lr=1e-3, groups=grp)
            for s in ("r", "delta", "delta_u", "gap")}
        for s in ("r", "delta", "delta_u", "gap"):
            cell = prov_dec[SRC_NAMES[k]][s]
            out(f"    [{SRC_NAMES[k]:9s} {s:6s} n={n_match}] " + "  ".join(
                f"{t}=" + "/".join(_fmt(cell[f'h{h}'].get(t, {'r2': float('nan')})['r2'], 6, 3)
                                   for h in (0, 64)) for t in PRIM))

    # ---------------- the observer ladder ----------------
    out("")
    out("=" * 78)
    out("(4) THE OBSERVER LADDER  -- public information only, same targets, same strata,")
    out("    same readout stack. Any 'introspection helps X' claim must beat its twin.")
    out("=" * 78)
    out("  ONE ASYMMETRY, STATED. The self side reads `z`, the FROZEN CONTROLLER's encoding of")
    out("  the same `x` the observer gets raw, and the observer has to rebuild an encoder with")
    out("  the same one-hidden-layer stack the probes use. That is paper 2's reconstruction")
    out("  cost, but part of it is an architecture gap and not a privacy fact; the capacity")
    out("  sweep and the half-data control are what separate `cannot` from `not yet`.")
    ocaps = [int(q) for q in args.obs_caps.split(",")]
    tab_dec = {k: (v[dec_rows] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v)
               for k, v in tab.items()}
    obs = {}
    for rung, label in ((0, "O_x"), (1, "O_x+a"), (2, "O_x+a+grade")):
        t0 = time.time()
        Xo = standardise_(obs_inputs(tab_dec, rung, n_slots), ptr)
        obs[label] = decode(Xo, tgt_prim, ptr, pte, caps=ocaps, seed=args.seed + 200,
                            steps=args.obs_steps, batch=512, lr=1e-3, standardise=False,
                            groups=grp)
        best = {t: max(obs[label][f"h{h}"].get(t, {"r2": float("nan")})["r2"] for h in ocaps)
                for t in PRIM}
        out(f"    [{label:12s} d={Xo.shape[1]:<4d}] " +
            "  ".join(f"{t}={_fmt(best[t])}" for t in PRIM) + f"   [{time.time() - t0:.0f}s]")
        if rung == 2:
            # a BUDGET control, so it must be half the data and the same data-generating
            # range: `ptr[:ptr.size // 2]` would be the first fourteen cycles, which is a
            # different experiment. Half the instances, drawn over all the held-out cycles.
            half, _ = _group_split(ptr, grp, 0.5, args.seed + 303)
            obs[label + "_half"] = decode(Xo, tgt_prim, half, pte, caps=[ocaps[-1]],
                                          seed=args.seed + 201, steps=args.obs_steps,
                                          batch=512, lr=1e-3, standardise=False, groups=grp)
            bh = {t: obs[label + "_half"][f"h{ocaps[-1]}"].get(t, {"r2": float("nan")})["r2"]
                  for t in PRIM}
            out(f"    [{label + '_half':12s} budget control] " +
                "  ".join(f"{t}={_fmt(bh[t])}" for t in PRIM))
        del Xo
    out("")
    out("  self vs observer (best over capacities):")
    out(f"    {'target':14s} {'best self':>12s} {'src':>9s} {'O_x':>8s} {'O_x+a':>8s} "
        f"{'O_x+a+g':>8s} {'advantage':>10s}")
    adv = {}
    for t in PRIM:
        selfb, selfs = -9, None
        for s in ("r", "delta", "delta_u", "gap", "r1", "p2", "delib"):
            b = max(dec[s][f"h{h}"].get(t, {"r2": float("nan")})["r2"]
                    for h in pcaps if f"h{h}" in dec[s])
            if b == b and b > selfb:
                selfb, selfs = b, s
        ob = {}
        for label in ("O_x", "O_x+a", "O_x+a+grade"):
            ob[label] = max(obs[label][f"h{h}"].get(t, {"r2": float("nan")})["r2"]
                            for h in ocaps)
        adv[t] = {"self": selfb, "self_src": selfs, **ob,
                  "advantage": selfb - max(ob.values())}
        out(f"    {t:14s} {_fmt(selfb, 12, 3)} {selfs:>9s} {_fmt(ob['O_x'], 8, 3)} "
            f"{_fmt(ob['O_x+a'], 8, 3)} {_fmt(ob['O_x+a+grade'], 8, 3)} "
            f"{_fmt(adv[t]['advantage'], 10, 3)}")

    # ---------------- guards ----------------
    out("")
    out("=" * 78)
    out("(5) GUARDS")
    out("=" * 78)
    out(f"  gradcheck (analytic vs central differences, both rungs)            PASS")
    out(f"  ens_cos over {args.ens} independent-seed FM2 residuals: "
        f"{by_cap[f'h{fm_caps[0]}']['ens_cos']:.3f}")
    out("    (high => the residual is input-determined and any FM misses it identically;")
    out("     low  => it is the particular FM's own idiosyncratic noise)")
    out("  junk-residual eta^2 -- how much of each source is a calendar:")
    out(f"    {'source':12s} {'cycle':>8s} {'step':>8s} {'era':>8s}")
    eta = {}
    grp_c = np.searchsorted(cycles, tab["cycle"][dec_rows]).astype(np.int64)
    grp_s = tab["step"][dec_rows].astype(np.int64)
    grp_e = tab["era"][dec_rows].astype(np.int64)
    for sname in ("r", "delta", "delta_u", "r_cyc", "gap", "r1", "p2", "delib", "z_state",
                  "readout0"):
        X = src[sname]
        eta[sname] = {"cycle": _eta2(X, grp_c), "step": _eta2(X, grp_s), "era": _eta2(X, grp_e)}
        out(f"    {sname:12s} {eta[sname]['cycle']:>8.3f} {eta[sname]['step']:>8.3f} "
            f"{eta[sname]['era']:>8.3f}")
    out("  capacity invariance of the crux decode across FM instrument capacity:")
    out(f"    {'FM':>8s} {'params':>9s} {'%heads':>7s}  " +
        "  ".join(f"{t:>12s}" for t in PRIM))
    for hid in fm_caps:
        k = f"h{hid}"
        if "decode" in by_cap[k]:
            row = by_cap[k]["decode"]["r"]
        else:
            row = dec["r"]
        vals = [max(row[f"h{h}"].get(t, {"r2": float("nan")})["r2"]
                    for h in pcaps_red if f"h{h}" in row) for t in PRIM]
        out(f"    {k:>8s} {by_cap[k]['params']:>9d} {by_cap[k]['pct_of_heads']:>6.1f}%  " +
            "  ".join(_fmt(v, 12, 3) for v in vals) + "   (source r)")
    out("  scalar-norm negative controls and the shuffled null are rows of the matrix above.")

    # ---------------- (6) the strict-timing split ----------------
    #  The interleaved split gives the FM neighbouring cycles of every test cycle, so it can
    #  interpolate the calendar even though it never sees the test cycle itself. The forward
    #  split cannot: it is trained on the first three quarters of the run and scored on the
    #  last quarter, which is the only arrangement under which the forecast could actually have
    #  been on a wire before the datum arrived. Crux sources only, reduced ladder.
    out("")
    out("=" * 78)
    out("(6) THE STRICT-TIMING SPLIT  -- FM trained on the first 3/4 of the run, scored on the")
    out("    last 1/4. Same targets, same strata, same readout ladder; crux sources only.")
    out("=" * 78)
    fwd = {}
    cut = cycles[int(0.75 * cycles.size)]
    f_test = np.nonzero(tab["cycle"] >= cut)[0]
    f_pool = np.nonzero(tab["cycle"] < cut)[0]
    f_tr = np.sort(rng.choice(f_pool, min(args.fm_rows, f_pool.size), replace=False))
    hid0 = fm_caps[0]
    fm1, fm2 = train_pair(Xz, Xr, u_slot, Ds, f_tr, hidden=hid0, seed=args.seed + 4441,
                          steps=args.fm_steps, batch=512, lr=1e-3, mask_slot=mask_slot)
    fp2 = fm_predict(fm2, Xz[f_test], Xr[f_test], u_slot[f_test], mask_slot)
    fp1 = fm_predict(fm1, Xz[f_test], Xr[f_test], mask_slot, mask_slot)
    fdd = Ds[f_test]
    f_cyc_i = np.searchsorted(cycles, tab["cycle"][f_test])
    f_strata = (f_cyc_i * 16 + tab["step"][f_test]).astype(np.int64)
    f_inst = tab["inst"][f_test]
    f_ptr, f_pte = np.nonzero(f_inst % 4 != 3)[0], np.nonzero(f_inst % 4 == 3)[0]
    f_grp = (f_cyc_i.astype(np.int64) * 4096 + f_inst.astype(np.int64))
    f_tgt = {}
    for name in ("verr", "dprop"):
        y_all, m_all = tg_all[name]
        y, m = np.where(m_all[f_test], y_all[f_test], 0.0), m_all[f_test].astype(bool)
        f_tgt[name] = (y, m)
        f_tgt[name + "_cs"] = (_stratum_residualise(y, f_strata, f_ptr[m[f_ptr]]), m)
    out(f"  cycles >= {int(cut)} held out ({np.unique(tab['cycle'][f_test]).size} cycles, "
        f"{f_test.size} rows; {f_ptr.size} probe-train / {f_pte.size} probe-test)")
    f_slot_oh = np.zeros((f_test.size, mask_slot + 1), np.float32)
    f_slot_oh[np.arange(f_test.size), u_slot[f_test]] = 1.0
    for sname, X in (("r", fdd - fp2), ("delta", fdd),
                     ("delta_u", np.concatenate([fdd, f_slot_oh], 1)),
                     ("gap", fp2 - fp1), ("r1", fdd - fp1)):
        fwd[sname] = decode(X, f_tgt, f_ptr, f_pte, caps=pcaps_red, seed=args.seed,
                            steps=args.probe_steps, batch=512, lr=1e-3, groups=f_grp)
        out(f"    [{sname:8s}] " + "  ".join(
            f"{t}=" + "/".join(_fmt(fwd[sname][f'h{h}'].get(t, {'r2': float('nan')})['r2'], 6, 3)
                               for h in pcaps_red) for t in ("verr_cs", "dprop_cs",
                                                             "verr", "dprop")))
    out(f"  |r|/|delta| under the strict split: "
        f"{float(np.linalg.norm(fdd - fp2, axis=1).mean() / np.linalg.norm(fdd, axis=1).mean()):.3f}"
        f"   |g|/|delta|: "
        f"{float(np.linalg.norm(fp2 - fp1, axis=1).mean() / np.linalg.norm(fdd, axis=1).mean()):.3f}")
    del fm1, fm2, fp1, fp2, fdd

    # ---------------- figures ----------------
    figs = []
    try:
        figs = make_figures(outdir, dec, dec_diag, obs, adv, agency, by_cap, eta, src, tgt_prim,
                            ptr, pte, prov, gnorm0, dn, PRIM, CONTENT, pcaps, pcaps_red, ocaps,
                            fm_caps)
    except Exception as e:                                     # pragma: no cover
        out(f"  [figures skipped: {e}]")
    if figs:
        out("")
        out("  figures: " + ", ".join(os.path.basename(f) for f in figs))

    out("")
    out(f"END - {time.time() - t_start:.0f}s total")
    with open(os.path.join(outdir, "reduction.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    res = {"config": vars(args), "n_rows": int(n), "n_slots": n_slots,
           "test_cycles": test_cyc.tolist(), "by_capacity": by_cap, "decode": dec,
           "decode_diag": dec_diag, "observer": obs, "advantage": adv, "agency": agency,
           "provenance_decode": prov_dec, "eta2": eta, "forward_split": fwd,
           "forward_cut_cycle": int(cut), "within_cycle_bound": within,
           "datum": {"prov": {SRC_NAMES.get(k, k): int(v) for k, v in sc.items()},
                     "argmax_kept": float(tab["u_kept"].mean()),
                     "in_value_buffer": float(vb.mean()),
                     "u_on_solved_vb": float(tab["u_on_solved"][vb].mean())}}
    with open(os.path.join(outdir, "fit.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nwrote {outdir}/reduction.txt and fit.json", flush=True)


def make_figures(outdir, dec, dec_diag, obs, adv, agency, by_cap, eta, src, tgt, ptr, pte,
                 prov, gnorm, dn, PRIM, CONTENT, pcaps, pcaps_red, ocaps, fm_caps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    best = lambda cell, t, caps: max(
        cell[f"h{h}"].get(t, {"r2": float("nan")})["r2"] for h in caps if f"h{h}" in cell)

    # --- 1. the decode matrix ---
    names = list(dec)
    M = np.full((len(names), len(PRIM)), np.nan)
    for i, s in enumerate(names):
        caps_here = pcaps if s in CONTENT else pcaps_red
        for j, t in enumerate(PRIM):
            M[i, j] = best(dec[s], t, caps_here)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(names) + 1.8))
    lim = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(PRIM)), PRIM, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(PRIM)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("decode matrix: best out-of-sample $R^2$ over the readout ladder", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    p = os.path.join(outdir, "decode_matrix.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- 2. capacity curves ---
    fig, axes = plt.subplots(1, len(PRIM), figsize=(3.1 * len(PRIM), 3.0), sharey=False)
    for j, t in enumerate(PRIM):
        ax = axes[j]
        for s in ("r", "delta", "delta_u", "r_cyc", "gap", "r1", "p2", "delib", "z_state",
                  "readout0"):
            ys = [dec[s][f"h{h}"].get(t, {"r2": np.nan})["r2"] for h in pcaps]
            ax.plot(range(len(pcaps)), ys, marker="o", ms=3, label=s)
        for s in ("r_shuffled", "|r|", "|delta|", "g", "calendar"):
            ys = [dec[s][f"h{h}"].get(t, {"r2": np.nan})["r2"] for h in pcaps_red]
            ax.plot([pcaps.index(h) for h in pcaps_red], ys, ls=":", marker="s", ms=3,
                    lw=1, label=s)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(pcaps)), ["ridge"] + [f"h{h}" for h in pcaps[1:]], fontsize=7)
        ax.set_title(t, fontsize=9)
        if j == 0:
            ax.set_ylabel("out-of-sample $R^2$")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.suptitle("readout-capacity ladder, per target", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "capacity_ladder.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- 3. agency ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ks = [k for k in agency]
    axes[0].bar(range(len(ks)), [agency[k]["g_rel"] for k in ks], color="#4C72B0")
    axes[0].set_xticks(range(len(ks)), ks, rotation=20, fontsize=8)
    axes[0].set_ylabel(r"$\|g\|\,/\,\|\Delta\|$")
    axes[0].set_title("arity gap by provenance of the chosen move", fontsize=9)
    for k, c in (("proposed", "#4C72B0"), ("explore", "#DD8452")):
        m = prov == (1 if k == "proposed" else 4)
        if m.sum() > 10:
            axes[1].hist(gnorm[m] / np.maximum(dn[m], 1e-9), bins=60, range=(0, 1.2),
                         histtype="step", density=True, label=k, color=c)
    axes[1].set_xlabel(r"$\|g\|/\|\Delta\|$")
    axes[1].legend(fontsize=8)
    axes[1].set_title("distribution", fontsize=9)
    axes[2].bar(range(len(ks)), [agency[k]["r1"] - agency[k]["r"] for k in ks], color="#55A868")
    axes[2].set_xticks(range(len(ks)), ks, rotation=20, fontsize=8)
    axes[2].set_ylabel(r"$\|r_1\| - \|r\|$")
    axes[2].set_title("what the efference copy buys the forecast", fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "agency.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- 4. observer ladder ---
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = ["O_x", "O_x+a", "O_x+a+grade", "O_x+a+grade_half"]
    w = 0.2
    for i, t in enumerate(PRIM):
        vals = [adv[t]["self"]] + [best(obs[l], t, ocaps if "half" not in l else [ocaps[-1]])
                                   for l in labels]
        ax.bar(np.arange(len(vals)) + i * w, vals, width=w, label=t)
    ax.set_xticks(np.arange(5) + 1.5 * w, ["self (best)"] + labels, rotation=15, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_title("self vs the observer ladder (public information only)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "observer_ladder.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- 5. guards ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ks = list(eta)
    xx = np.arange(len(ks))
    for i, g in enumerate(("cycle", "step", "era")):
        axes[0].bar(xx + i * 0.26, [eta[k][g] for k in ks], width=0.26, label=g)
    axes[0].set_xticks(xx + 0.26, ks, rotation=20, fontsize=8)
    axes[0].set_ylabel(r"$\eta^2$")
    axes[0].set_title("how much of each source is a calendar", fontsize=9)
    axes[0].legend(fontsize=7)
    caps = [k for k in by_cap]
    axes[1].plot([by_cap[k]["params"] for k in caps],
                 [by_cap[k]["rel_r"] for k in caps], marker="o", label=r"$|r|/|\Delta|$")
    axes[1].plot([by_cap[k]["params"] for k in caps],
                 [by_cap[k]["cos_p2_delta"] for k in caps], marker="s",
                 label=r"$\cos(p_2,\Delta)$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("FM parameters")
    axes[1].axvline(by_cap[caps[0]]["params"] / max(by_cap[caps[0]]["pct_of_heads"], 1e-9) * 100,
                    color="k", ls="--", lw=0.8)
    axes[1].set_title("instrument-capacity sweep (dashed = the heads' own size)", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "guards.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths




# ======================================================================
# (7) THE CONDITIONING-COMPLETION LADDER  (`--ladder`)
#
# The decode run leaves one thing undecomposed: the update is 44% a per-cycle common mode on the
# decode rows, a forecaster reading (s, a) has no out-of-cycle skill, and a cycle ONE-HOT would
# cut the residual to 0.731 but cannot generalise to a held-out cycle by construction. So the
# open question is what KIND of thing that common mode is: is it a function of the cycle's BATCH
# CONTENT -- public, prospective, and in principle available to a forecaster on any cycle -- or
# is it optimizer/composition noise that no conditioning on data can reach?
#
# The ladder answers it by completing the conditioning set, one block at a time, in the SAME
# FM2 class, on the SAME cycle-held-out split, with the SAME target scaling and the SAME training
# recipe (fixed steps, no early stopping, same seed / lr / batch), so the only thing that moves
# between rungs is what the forecaster is allowed to read:
#
#   rung 1  (s, a)                       -- the decode run's own FM2. REUSED from fit.json.
#   rung 2  + own-datum supervision      -- this ROW's membership in the update's data
#   rung 3  + cycle batch summary        -- period-c scalars and histograms
#   rung 4  + richer batch content       -- a projected encoding of the actual training sets
#   rung 5  + own pre-update readout     -- NOT in the brief; added because a gradient step's
#                                           effect at s depends on the head's current output at
#                                           s, and that is the most obvious remaining omission.
#
# EVERY BLOCK IS PROSPECTIVE. Blocks 3 and 4 are computed from cycle c's own practice beam, which
# runs in block (a); the updates run in blocks (c)/(c'). So all of it is on the wire before the
# update happens, which is the property the timing claim needs. Nothing derived from the update's
# OUTCOME (vloss, gloss, the post-update readouts) is admitted.
#
# WHAT IS RECONSTRUCTED, AND HOW EXACTLY.
#   * The VALUE buffer. `run_arm` pushes `traj[t]` for t=0..budget of every surviving trajectory
#     with `y = succ` of that trajectory's tip, so cycle c pushes exactly B*W*(budget+1) = 9,216
#     rows, and a tip row enters `n_desc` times with its descendants' labels. Per-row `n_desc` and
#     `sum_succ` are therefore a sufficient statistic for the push, and `buf_cap = 100,000` makes
#     the live buffer the last ~10.85 cycles of pushes.
#   * The pi BUFFER. `prop_pairs` emits, for each SOLVED tip, one (state, root, slot) pair per
#     step, so the push is `8 * n_solved` pairs and the per-slot histogram is `sum_succ` of each
#     tip row at step>=1 charged to `slot(t_mv)` with the PARENT's state. `prop_buf_cap = 60,000`.
#     Both reconstructions are ASSERTED against the run's own `results.json['log']['prop']`
#     (`n_pairs` and the live `n`), which the arm logged independently.
#   * The REPLAY POOL is not logged, and is NOT approximated. It does not need to be: it is built
#     once in `_spiral_shared` (`shared["replay"] = vb`) and never rebound inside `run_arm`, so
#     its CONTENT is constant across cycles and cannot itself carry a per-cycle signature. What
#     it does carry is a gradient that varies with `value_c`, and that is head state, not batch
#     content -- rung 5 is the block that speaks to it. Stated, not closed.
#   * The EXACT SAMPLING INDICES are drawn from the arm's private `rng` / `prng` and are not
#     recoverable. Rungs 3 and 4 are therefore summaries of the POOL the update sampled from,
#     never of the sampled batches themselves. This is a real gap and it is not approximated away.
#
# THE BINDING LIMIT, stated once and loudly: blocks 3 and 4 are CONSTANT WITHIN A CYCLE, so
# however many rows the FM is fit on, the effective sample size for those blocks is the number of
# TRAINING CYCLES -- 84. A rung that fails to transfer is therefore ambiguous between `batch
# content does not explain the common mode` and `84 cycles is not enough to learn that it does`,
# and the h64/h256 pair is the only handle offered on which it is.
# ======================================================================

def _proj_matrix(d, k, seed):
    """A fixed seeded Gaussian projection of `z`, shared by every cycle so that a per-slot or
    per-label mean is comparable across cycles. Fixed, not learned: a learned compression of the
    batch would be a second instrument with its own capacity story."""
    g = np.random.default_rng(seed).standard_normal((d, k)) / np.sqrt(d)
    return g.astype(np.float32)


def push_blocks(root, proj, verbose=True):
    """Per-cycle PUSH statistics for both buffers, from the shards. One pass, one cycle at a
    time; see the section header for why these statistics are sufficient."""
    audi = json.load(open(os.path.join(root, "audiation.json")))
    cyc_meta = {c["cycle"]: c for c in audi["cycles"]}
    n_slots = int(audi["n_slots"])
    k = proj.shape[1]
    files = sorted(glob.glob(os.path.join(root, "decisions", "*.npz")))
    out, cur_path, d = {}, None, None
    t0 = time.time()
    for c in sorted(cyc_meta):
        meta = cyc_meta[c]
        bud = int(meta["budget"])
        ms_slots = np.asarray(meta["ms_slots"], np.int64)
        path = _shard_for(files, c)
        if path is None:
            continue
        if path != cur_path:
            d, cur_path = np.load(path), path
        tm = np.nonzero(d["t_cycle"] == c)[0]
        t_off = tm[0]
        st = d["t_step"][tm]
        ni = int(d["t_inst"][tm].max()) + 1
        w = {int(s): int((st == s).sum()) // ni for s in np.unique(st)}
        toff, base = {}, 0
        for s_ in range(bud + 1):
            toff[s_] = base
            base += ni * w[s_]
        n_tip = base
        n_desc = np.zeros(n_tip, np.int64)
        sum_succ = np.zeros(n_tip, np.int64)
        idxB = np.arange(toff[bud], toff[bud] + ni * w[bud])
        n_desc[idxB] = 1
        sum_succ[idxB] = (d["t_succ"][t_off + idxB] > 0.5).astype(np.int64)
        for s_ in range(bud, 0, -1):
            idx = np.arange(toff[s_], toff[s_] + ni * w[s_])
            par = d["t_parent"][t_off + idx].astype(np.int64)
            inst = np.repeat(np.arange(ni), w[s_])
            prow = toff[s_ - 1] + inst * w[s_ - 1] + par
            np.add.at(n_desc, prow, n_desc[idx])
            np.add.at(sum_succ, prow, sum_succ[idx])
        zp = d["t_z"][tm].astype(np.float32) @ proj                      # (n_tip, k)

        # ---- the value buffer's push: every tip row, `n_desc` times, labelled by descendants
        pos = sum_succ.astype(np.float64)
        neg = (n_desc - sum_succ).astype(np.float64)
        blk = {"v_n": float(n_desc.sum()), "v_pos": float(pos.sum()),
               "v_zpos": (pos[:, None] * zp).sum(0), "v_zneg": (neg[:, None] * zp).sum(0)}

        # ---- the pi buffer's push: one pair per step of every SOLVED trajectory
        hist = np.zeros(n_slots)
        zsum = np.zeros((n_slots, k))
        for s_ in range(1, bud + 1):
            idx = np.arange(toff[s_], toff[s_] + ni * w[s_])
            wgt = sum_succ[idx].astype(np.float64)
            if not wgt.any():
                continue
            slot = ms_slots[d["t_mv"][t_off + idx].astype(np.int64)]
            par = d["t_parent"][t_off + idx].astype(np.int64)
            inst = np.repeat(np.arange(ni), w[s_])
            prow = toff[s_ - 1] + inst * w[s_ - 1] + par                 # the PAIR's own state
            np.add.at(hist, slot, wgt)
            np.add.at(zsum, slot, wgt[:, None] * zp[prow])
        blk.update({"p_n": float(hist.sum()), "p_hist": hist, "p_zsum": zsum,
                    "n_solved": float(sum_succ[idxB].sum()), "n_slots": n_slots,
                    "n_moves": float(meta["n_moves"]), "k_eff": float(meta["k_eff"]),
                    "routed": float(bool(meta["routed"])),
                    "filter_on": float(bool(meta["filter_on"])),
                    "e_practice": float(meta.get("e_practice") or 0.0)})
        out[c] = blk
        if verbose and c % 30 == 0:
            print(f"    ...batch features, cycle {c} ({time.time() - t0:.0f}s)", flush=True)
    return out, n_slots


def buffer_state(blocks, c, cap, key_n):
    """The live buffer at cycle c: the last `cap` rows in ARRIVAL order, as per-cycle weights.

    Pushes inside one cycle are ordered, so a partially-included oldest cycle really drops a
    specific prefix of its own push; this weights that cycle uniformly by the surviving fraction
    instead. The approximation is confined to at most ONE cycle out of ~11 (value) or ~34 (pi)
    and it is named here rather than buried.
    """
    wts, total = {}, 0.0
    for cc in range(c, 0, -1):
        if cc not in blocks:
            continue
        n = blocks[cc][key_n]
        if total + n <= cap:
            wts[cc], total = 1.0, total + n
        else:
            room = cap - total
            if room > 0:
                wts[cc] = room / n
                total = cap
            break
    return wts, total


def batch_features(root, proj, seed, verbose=True):
    """The rung-3 and rung-4 blocks, per cycle, plus the exactness check against the run's log."""
    blocks, n_slots = push_blocks(root, proj, verbose=verbose)
    res = json.load(open(os.path.join(root, "results.json")))
    log = res["log"]
    by_cycle = {int(c): i for i, c in enumerate(log["cycle"])}
    cfg = res["config"]
    vcap, pcap = float(cfg["buf_cap"]), float(cfg["prop_buf_cap"])
    k = proj.shape[1]
    f3, f4, chk = {}, {}, {"n_pairs": [], "pbuf_n": []}
    for c in sorted(blocks):
        b = blocks[c]
        vw, vtot = buffer_state(blocks, c, vcap, "v_n")
        pw, ptot = buffer_state(blocks, c, pcap, "p_n")
        v_pos = sum(w * blocks[cc]["v_pos"] for cc, w in vw.items())
        p_hist = sum(w * blocks[cc]["p_hist"] for cc, w in pw.items())
        p_zsum = sum(w * blocks[cc]["p_zsum"] for cc, w in pw.items())
        v_zpos = sum(w * blocks[cc]["v_zpos"] for cc, w in vw.items())
        v_zneg = sum(w * blocks[cc]["v_zneg"] for cc, w in vw.items())
        v_neg = vtot - v_pos
        i = by_cycle.get(c)
        n_mined = float(log["n_mined"][i]) if i is not None else 0.0
        # ---- rung 3: period-c scalars and the two histograms, all normalised ----
        f3[c] = np.concatenate([
            np.array([b["n_solved"] / 1024.0, n_mined / 8.0, b["e_practice"],
                      b["n_moves"] / 56.0, b["k_eff"] / 32.0, b["routed"], b["filter_on"],
                      v_pos / max(vtot, 1.0), vtot / vcap, len(vw) / 12.0,
                      b["v_n"] / max(vtot, 1.0),
                      ptot / pcap, len(pw) / 40.0, b["p_n"] / max(ptot, 1.0)]),
            p_hist / max(p_hist.sum(), 1.0),                       # the BUFFER's pi targets
            b["p_hist"] / max(b["p_n"], 1.0),                      # THIS CYCLE's pi targets
        ]).astype(np.float32)
        # ---- rung 4: a projected encoding of what is actually in the two training sets ----
        f4[c] = np.concatenate([
            v_zpos / max(v_pos, 1.0), v_zneg / max(v_neg, 1.0),
            (p_zsum / np.maximum(p_hist, 1.0)[:, None]).reshape(-1),
        ]).astype(np.float32)
        if i is not None:
            pr = log["prop"][i]
            if isinstance(pr, dict) and pr.get("n_pairs") is not None:
                chk["n_pairs"].append(abs(b["p_n"] - pr["n_pairs"]))
                chk["pbuf_n"].append(abs(ptot - pr["n"]))
    chk = {k_: (float(np.max(v)) if v else float("nan")) for k_, v in chk.items()}
    return f3, f4, chk, n_slots


def train_fm2(X, Y, tr, *, hidden, seed, steps, batch, lr):
    """FM2 alone, trained EXACTLY as `train_pair` trains its FM2: same init seed, same optimiser,
    same step budget, same batch stream. `train_pair` steps fm1 and fm2 on the same `i` with
    separate optimisers, so dropping fm1 leaves fm2's trajectory bit-identical -- which the
    ladder ASSERTS by reproducing rung 1's own numbers from `fit.json` before it reports a rung.
    """
    net = MLP(X.shape[1], hidden, Y.shape[1], seed)
    opt = AdamW(net.params(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    for _ in range(steps):
        i = tr[rng.integers(0, tr.size, batch)]
        out, cx = net.forward(X[i], cache=True)
        d = ((out - Y[i]) * (2.0 / Y.shape[1])).astype(np.float32)
        opt.step(net.backward(cx, d))
    return net


LADDER_RUNGS = [
    ("1_state_action", "(s, a)  [reused from fit.json]"),
    ("2_own_datum", "+ own-datum supervision flags"),
    ("3_batch_summary", "+ cycle batch summary"),
    ("4_batch_content", "+ richer batch content"),
    ("5_own_readout", "+ own pre-update readout  [beyond the brief]"),
]


def conditioning_ladder(args, root, outdir, tab, cycles, dec_rows, fm_tr, ptr, pte, grp,
                        cyc_i, Xz, Xr, Ds, u_slot, mask_slot, tgt_prim, PRIM, out):
    """Rungs 2-5, trained and scored exactly like rung 1. See the section header."""
    n = tab["cycle"].size
    proj = _proj_matrix(tab["z"].shape[1], args.proj_k, args.seed + 31337)
    cache = os.path.join(outdir, f"batch_v1_k{args.proj_k}.npz")
    if os.path.isfile(cache) and not args.no_cache:
        zc = np.load(cache)
        f3 = {int(c): zc["f3"][i] for i, c in enumerate(zc["cyc"])}
        f4 = {int(c): zc["f4"][i] for i, c in enumerate(zc["cyc"])}
        chk = {"n_pairs": float(zc["chk"][0]), "pbuf_n": float(zc["chk"][1])}
        print(f"  loaded cached batch features {cache}", flush=True)
    else:
        print("  reconstructing the two training buffers off the fetched bytes...", flush=True)
        f3, f4, chk, _ = batch_features(root, proj, args.seed)
        cs = sorted(f3)
        np.savez(cache, cyc=np.asarray(cs),
                 f3=np.stack([f3[c] for c in cs]), f4=np.stack([f4[c] for c in cs]),
                 chk=np.asarray([chk["n_pairs"], chk["pbuf_n"]]))

    out("")
    out("=" * 78)
    out("(7) THE CONDITIONING-COMPLETION LADDER")
    out("=" * 78)
    out("  Is the update's per-cycle common mode a function of public, prospective BATCH")
    out("  CONTENT, or is it optimizer/composition noise? Same FM2 class, same cycle-held-out")
    out("  split, same target scaling, same training recipe; only the conditioning set moves.")
    out("")
    out("  BUFFER RECONSTRUCTION, checked against the arm's own independently logged series:")
    out(f"    max |reconstructed pi pushes  - log['prop']['n_pairs']| = {chk['n_pairs']:.0f}")
    out(f"    max |reconstructed pi buffer  - log['prop']['n']|       = {chk['pbuf_n']:.0f}")
    out("    (the second is >0 only where the cap truncates a cycle mid-push; see `buffer_state`)")
    out("  The REPLAY POOL is bound once in `_spiral_shared` and never rebound in `run_arm`, so")
    out("  its CONTENT is constant across cycles and cannot carry a per-cycle signature; what it")
    out("  contributes that does vary is a gradient through `value_c`, i.e. head state, which is")
    out("  rung 5's block. Stated, not closed.")
    out("  The EXACT SAMPLING INDICES come from the arm's private RNG and are NOT recoverable:")
    out("  rungs 3-4 summarise the POOL the update sampled from, never the sampled batches.")
    out("  BINDING LIMIT: rungs 3-4 are constant within a cycle, so their effective sample size")
    out(f"  is the number of TRAINING CYCLES ({np.setdiff1d(cycles, cycles[3::4]).size}), not the"
        f" number of rows.")

    # ---- the blocks ----
    vb = tab["n_desc"] > 0
    ybar = np.where(vb, tab["sum_succ"] / np.maximum(tab["n_desc"], 1), 0.0)
    row2 = np.stack([vb.astype(np.float32), np.log1p(tab["n_desc"]).astype(np.float32),
                     ybar.astype(np.float32), np.log1p(tab["sum_succ"]).astype(np.float32),
                     tab["u_on_solved"].astype(np.float32),
                     (tab["sum_succ"] > 0).astype(np.float32)], 1)
    cyc_all = tab["cycle"]
    B3 = np.stack([f3[int(c)] for c in sorted(f3)])
    B4 = np.stack([f4[int(c)] for c in sorted(f4)])
    idx3 = {c: i for i, c in enumerate(sorted(f3))}
    take = np.asarray([idx3[int(c)] for c in cyc_all])
    blk3, blk4 = B3[take], B4[take]
    blk5 = np.concatenate([tab["v_c"][:, None], tab["pi_c"]], 1).astype(np.float32)
    slot_oh = np.zeros((n, mask_slot + 1), np.float32)
    slot_oh[np.arange(n), u_slot] = 1.0

    def std_(A):
        return ((A - A[fm_tr].mean(0, keepdims=True))
                / np.maximum(A[fm_tr].std(0, keepdims=True), 1e-6)).astype(np.float32)

    base = np.concatenate([Xz, Xr, slot_oh], 1)          # rung 1, in train_pair's exact order
    parts = {"2_own_datum": std_(row2), "3_batch_summary": std_(blk3),
             "4_batch_content": std_(blk4), "5_own_readout": std_(blk5)}
    inputs, acc = {"1_state_action": base}, [base]
    for name, _ in LADDER_RUNGS[1:]:
        acc = acc + [parts[name]]
        inputs[name] = np.concatenate(acc, 1)

    # ---- rung 1, reused ----
    fj = os.path.join(outdir, "fit.json")
    prev = json.load(open(fj)) if os.path.isfile(fj) else None
    assert prev is not None, ("rung 1 is REUSED, not refitted: run the default path first so "
                              f"{fj} exists")
    r1 = (prev or {}).get("by_capacity", {})
    dd = Ds[dec_rows]
    dn = np.linalg.norm(dd, axis=1).mean()
    tr_sub = fm_tr[::7]
    res = {"buffer_check": chk, "proj_k": args.proj_k, "rungs": {}}
    out("")
    out(f"  {'rung':18s} {'d_in':>5s} {'FM':>6s} {'|r|/|d|':>8s} {'cos':>7s} {'in-cos':>7s}"
        f"   {'  '.join(f'{t:>9s}' for t in PRIM)}")

    def decode_row(rname, X):
        r = decode(X, tgt_prim, ptr, pte, caps=[0, 64], seed=args.seed,
                   steps=args.probe_steps, batch=512, lr=1e-3, groups=grp)
        return {t: max(r[f"h{h}"].get(t, {"r2": float("nan")})["r2"] for h in (0, 64))
                for t in PRIM}, r

    for hid in [int(q) for q in args.fm_caps.split(",")][:2]:
        for name, label in LADDER_RUNGS:
            key = f"{name}@h{hid}"
            if name == "1_state_action":
                cell = r1.get(f"h{hid}")
                if cell is None:
                    continue
                rel, cos_o, cos_i = cell["rel_r"], cell["cos_p2_delta"], cell["cos_in_sample"]
                dec1 = (prev or {}).get("decode", {}).get("r", {})
                best = {t: max(dec1.get(f"h{h}", {}).get(t, {"r2": float("nan")})["r2"]
                               for h in (0, 64) if f"h{h}" in dec1) for t in PRIM} \
                    if dec1 and hid == int(args.fm_caps.split(",")[0]) else \
                    {t: float("nan") for t in PRIM}
                dfull = None
                nd = inputs[name].shape[1]
            else:
                X = inputs[name]
                nd = X.shape[1]
                net = train_fm2(X, Ds, fm_tr, hidden=hid, seed=args.seed + 4441,
                                steps=args.fm_steps, batch=512, lr=1e-3)
                p2 = predict(net, X[dec_rows])
                rr = dd - p2
                rel = float(np.linalg.norm(rr, axis=1).mean() / dn)
                cos_o = _cos(p2, dd)
                p2i = predict(net, X[tr_sub])
                cos_i = _cos(p2i, Ds[tr_sub])
                best, dfull = decode_row(name, rr) if hid == int(
                    args.fm_caps.split(",")[0]) else ({t: float("nan") for t in PRIM}, None)
                del net, p2, p2i
            out(f"  {label[:18]:18s} {nd:>5d} {('h%d' % hid):>6s} {rel:>8.3f} {cos_o:>+7.3f} "
                f"{cos_i:>+7.3f}   " + "  ".join(_fmt(best[t], 9, 3) for t in PRIM))
            res["rungs"][key] = {"d_in": int(nd), "rel_r": rel, "cos": cos_o, "cos_in": cos_i,
                                 "decode_best": best,
                                 "decode_full": (dfull if dfull is not None else {})}
    out("")
    out("  REFERENCE POINTS (both already measured, neither a rung):")
    out("    calendar one-hot          |r_cyc|/|d| = 0.731 -- not generalisable across held-out")
    out("                                                     cycles by construction")
    out("    within-cycle FM (ceiling) |r|/|d|     = 1.057 at cos +0.141")
    out("  and the crux row `delta` decodes, at the same probe settings (from fit.json):")
    dec_d = (prev or {}).get("decode", {}).get("delta", {})
    if dec_d:
        out("    delta                        " + "  ".join(
            _fmt(max(dec_d.get(f"h{h}", {}).get(t, {"r2": float("nan")})["r2"]
                     for h in (0, 64) if f"h{h}" in dec_d), 9, 3) for t in PRIM))
    with open(os.path.join(outdir, "ladder.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    try:
        p = ladder_figure(outdir, res, PRIM, dec_d)
        out(f"  figure: {os.path.basename(p)}")
    except Exception as e:                                          # pragma: no cover
        out(f"  [figure skipped: {e}]")
    return res


def ladder_figure(outdir, res, PRIM, dec_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hid0 = sorted({k.split("@h")[1] for k in res["rungs"]}, key=int)[0]
    names = [n for n, _ in LADDER_RUNGS if f"{n}@h{hid0}" in res["rungs"]]
    rel = [res["rungs"][f"{n}@h{hid0}"]["rel_r"] for n in names]
    cos = [res["rungs"][f"{n}@h{hid0}"]["cos"] for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(range(len(names)), rel, marker="o", color="#4C72B0")
    ax[0].axhline(1.0, color="k", lw=0.7, ls="-", label="no forecast")
    ax[0].axhline(0.731, color="#C44E52", lw=0.9, ls="--", label="calendar one-hot")
    ax[0].axhline(1.057, color="#8172B3", lw=0.9, ls=":", label="within-cycle ceiling")
    ax[0].set_ylabel(r"$\|r\|/\|\Delta\|$")
    ax[0].legend(fontsize=6)
    ax[1].plot(range(len(names)), cos, marker="s", color="#DD8452")
    ax[1].axhline(0.0, color="k", lw=0.7)
    ax[1].axhline(0.141, color="#8172B3", lw=0.9, ls=":", label="within-cycle ceiling")
    ax[1].set_ylabel(r"held-out $\cos(p_2,\Delta)$")
    ax[1].legend(fontsize=6)
    for j, t in enumerate(PRIM):
        ax[2].plot(range(len(names)),
                   [res["rungs"][f"{n}@h{hid0}"]["decode_best"].get(t, np.nan) for n in names],
                   marker="o", ms=3, label=f"r -> {t}")
        if dec_d:
            v = max(dec_d.get(f"h{h}", {}).get(t, {"r2": float("nan")})["r2"]
                    for h in (0, 64) if f"h{h}" in dec_d)
            ax[2].axhline(v, ls="--", lw=0.7, color=f"C{j}")
    ax[2].set_ylabel("out-of-sample $R^2$")
    ax[2].legend(fontsize=5, ncol=2)
    ax[2].set_title("dashed = the same target off raw $\\Delta$", fontsize=8)
    for a in ax:
        a.set_xticks(range(len(names)), [n.split("_", 1)[0] for n in names], fontsize=8)
        a.set_xlabel("conditioning rung")
    fig.suptitle("conditioning-completion ladder (FM$_2$, cycle-held-out)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "ladder.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p



# ======================================================================
# (8) THE E1b REFIT  (`--e1b`)  -- per-datum credit, on `au_s1/perdatum`
#
# WHAT CHANGED UNDER US. On `au_s0` every revision was caused by a BATCH, so the forecastable
# part of the update was `which cycle is this` and the self-model was denied it by construction:
# |r|/|Delta| > 1 at every capacity, and the conditioning-completion ladder could not recover it
# from batch content either. `au_s1/perdatum` spends the cycle's grades ONE TRAJECTORY AT A TIME,
# so each of the 14,848 update events has exactly one attributable cause, and the causes are
# things a forecaster can actually hold: the datum's lineage states and moves, its grade, the
# head state, and the update's position in the pass. This is the same measurement on a substrate
# where the thing being measured is no longer structurally hidden.
#
# THE DATUM: one (update event, own state t) row -- 14,848 x 9 = 133,632. The (s, a) ladder uses
# t in [0, budget) (118,784 rows) because `e_own_a` has `budget` entries against `budget+1`
# states: the action at row t carries state t to state t+1 and the terminal state has none.
# Q2's value-identity block uses all 9, since the value step touched all 9.
#
# SPLITS. FM by CYCLE (every 4th held out), exactly as phase 2. Probes by EVENT inside the
# held-out cycles: an event's 9 state rows share one lineage AND one gradient step, so the event
# is the group -- `_group_split`'s lesson, applied to the unit this dataset actually has.
# STRATA are cycle x update-order bucket (8 buckets of 16 over `e_order` in [0,128)): the head
# drifts within a pass, so order is this substrate's within-cycle calendar. Era is nested in
# cycle.
#
# THREE ZERO-INFLATION AND SCALE TRAPS, from the builder's own §6, honoured here:
#   * `e_did_pi == e_succ` EXACTLY on all 14,848 events (asserted). So `+ grade` hands the
#     forecaster the fact that `dpi` is identically zero on 79% of events. On the joint
#     [dv, dpi] target that is a WIRING gain, not forecasting skill, and it is why the ladder is
#     reported on three target blocks -- `dv1` (never zero), `own57@pi` (rows that stepped pi),
#     and `own57` (phase 2's exact target shape, comparable but zero-inflated) -- rather than one.
#   * The three revision sets hold different numbers of states (own 9, probe 32, reference 32),
#     so every cross-set quantity here is an RMS, never a sum or a max.
#   * The probe set changes at an era boundary and the reference set is redrawn every cycle, so
#     a probe DIRECTION is a within-era object and a reference direction is a within-cycle one.
#     Only magnitudes cross cycles. Targets are chosen accordingly.
#
# NOT IN THE TABLES, and not reconstructed: the pooled MAINTENANCE pass (value replay-only steps,
# pi history-only steps). The builder measured it at a median 4.8% of the cycle-boundary
# revision. Everything here is per-datum revision only; the whole-cycle comparability row is the
# one place the boundary snapshots are used, and there it is the whole thing including
# maintenance.
# ======================================================================

E1B_KEYS = ["e_cycle", "e_order", "e_inst", "e_tip", "e_succ", "e_dres", "e_root", "e_did_pi",
            "e_n_pairs", "e_vloss", "e_ploss", "e_own_a", "e_own_z", "e_own_v0", "e_own_pi0",
            "e_own_dv", "e_own_dpi", "e_prb_dv", "e_ref_dv", "e_own_x"]


def load_revisions(root, keys=E1B_KEYS, extra=()):
    """The revision tables, concatenated in cycle order. `e_prb_dpi` / `e_ref_dpi` are 32x56 per
    event and are loaded only on request -- they are most of the 50 MB."""
    out = {k: [] for k in list(keys) + list(extra)}
    for f in sorted(glob.glob(os.path.join(root, "revisions", "*.npz"))):
        d = np.load(f)
        for k in out:
            out[k].append(d[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def e1b_provenance(root, cache_path):
    """`analyze_audiation.datum_provenance`, cached. Provenance is not in the revision tables;
    it is derived by walking `t_parent` back through the decision tables and reading `c_src` off
    the candidate that became each tip. Not reimplemented here -- imported, so the agency axis is
    the same object the data report used."""
    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        return {(int(a), int(b), int(c)): (int(x), int(y), int(w))
                for a, b, c, x, y, w in z["p"]}
    from analyze_audiation import datum_provenance
    audi = json.load(open(os.path.join(root, "audiation.json")))
    bud = int(audi["cycles"][0]["budget"])
    files = sorted(glob.glob(os.path.join(root, "decisions", "*.npz")))
    prov = datum_provenance(files, bud)
    arr = np.asarray([[k[0], k[1], k[2], v[0], v[1], v[2]] for k, v in prov.items()],
                     dtype=np.int32)
    np.savez(cache_path, p=arr)
    return prov


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def e1b_refit(args, root, outdir, out):
    """The whole E1b analysis. Sections keyed to the four questions, in their order of weight."""
    t0 = time.time()
    rev = load_revisions(root)
    audi = json.load(open(os.path.join(root, "audiation.json")))
    cyc_meta = {c["cycle"]: c for c in audi["cycles"]}
    n_slots = int(audi["n_slots"])
    mask_slot = n_slots
    bud = int(rev["e_own_a"].shape[1])
    n_own = int(rev["e_own_dv"].shape[1])
    ne = rev["e_cycle"].size
    assert n_own == bud + 1
    assert (rev["e_did_pi"] == (rev["e_succ"] > 0)).all(), "did_pi is not exactly succ"

    out("AUDIATION E1b REFIT - per-datum credit, au_s1/perdatum")
    out(f"  tag={args.tag} arm={args.arm} seed={args.seed}")
    out(f"  {ne} update events over {np.unique(rev['e_cycle']).size} cycles; "
        f"own states {n_own}, actions {bud}, slots {n_slots}")
    out(f"  pi stepped on {int(rev['e_did_pi'].sum())} events ({rev['e_did_pi'].mean():.1%}); "
        f"e_did_pi == e_succ exactly (asserted)")
    out("")
    out("  THE DATUM is one (event, own state t) row. The (s,a) ladder uses t in [0,budget)")
    out("  because `e_own_a` has `budget` entries against `budget+1` states; Q2's value block")
    out("  uses all of them. Splits: FM by CYCLE, probes by EVENT inside the held-out cycles.")
    out("  Strata: cycle x update-order bucket (the head drifts within a pass).")

    # ---------------- the state-level table ----------------
    ev = np.repeat(np.arange(ne), n_own)                     # event id per state row
    tix = np.tile(np.arange(n_own), ne)                      # position in the lineage
    Z = rev["e_own_z"].reshape(ne * n_own, -1).astype(np.float32)
    V0 = rev["e_own_v0"].reshape(ne * n_own).astype(np.float32)
    PI0 = rev["e_own_pi0"].reshape(ne * n_own, n_slots).astype(np.float32)
    DV = rev["e_own_dv"].reshape(ne * n_own).astype(np.float32)
    DPI = rev["e_own_dpi"].reshape(ne * n_own, n_slots).astype(np.float32)
    act = np.concatenate([rev["e_own_a"], np.full((ne, 1), mask_slot, np.int16)], 1)
    A = act.reshape(ne * n_own).astype(np.int64)
    cyc = rev["e_cycle"][ev].astype(np.int64)
    order = rev["e_order"][ev].astype(np.int64)
    root_i = rev["e_root"][ev].astype(np.int64)
    succ = rev["e_succ"][ev].astype(np.float64)
    did_pi = rev["e_did_pi"][ev].astype(bool)
    era = np.asarray([cyc_meta[int(c)]["era"] for c in rev["e_cycle"]])[ev].astype(np.int64)

    # ---------------- provenance (the agency axis) ----------------
    prov = e1b_provenance(root, os.path.join(outdir, "provenance.npz"))
    nex = np.asarray([prov.get((int(c), int(i), int(t)), (0, 0, 0))[0]
                      for c, i, t in zip(rev["e_cycle"], rev["e_inst"], rev["e_tip"])])
    hit = sum(1 for c, i, t in zip(rev["e_cycle"], rev["e_inst"], rev["e_tip"])
              if (int(c), int(i), int(t)) in prov)
    out(f"  provenance join: {hit}/{ne} events resolved; n_explore mean {nex.mean():.2f}")

    # ---------------- splits and strata ----------------
    cycles = np.unique(rev["e_cycle"])
    test_cyc = cycles[3::4]
    is_test_e = np.isin(rev["e_cycle"], test_cyc)
    st_test = is_test_e[ev]
    dec = np.nonzero(st_test)[0]
    fm_pool = np.nonzero(~st_test)[0]
    rng = np.random.default_rng(args.seed + 7)
    fm_tr = np.sort(rng.choice(fm_pool, min(args.fm_rows, fm_pool.size), replace=False))
    grp = ev[dec]
    ptr, pte = _group_split(np.arange(dec.size), grp, 0.25, args.seed + 11)
    ci = np.searchsorted(cycles, cyc[dec])
    strata = (ci * 8 + np.minimum(order[dec] // 16, 7)).astype(np.int64)
    out(f"  FM split: {cycles.size - test_cyc.size} train / {test_cyc.size} held-out cycles; "
        f"FM fit on {fm_tr.size} state rows")
    out(f"  decode set {dec.size} state rows ({ptr.size} probe-train / {pte.size} probe-test, "
        f"split by event); {np.unique(strata).size} strata")

    # ---------------- FM inputs, by rung ----------------
    zmu, zsd = Z[fm_tr].mean(0, keepdims=True), np.maximum(Z[fm_tr].std(0, keepdims=True), 1e-6)
    Xz = ((Z - zmu) / zsd).astype(np.float32)
    nroot = int(root_i.max()) + 1
    Xr = np.zeros((ne * n_own, nroot), np.float32)
    Xr[np.arange(ne * n_own), root_i] = 1.0
    Sa = np.zeros((ne * n_own, mask_slot + 1), np.float32)
    Sa[np.arange(ne * n_own), A] = 1.0
    base = np.concatenate([Xz, Xr, Sa], 1)

    def std_(M):
        return ((M - M[fm_tr].mean(0, keepdims=True))
                / np.maximum(M[fm_tr].std(0, keepdims=True), 1e-6)).astype(np.float32)

    grade_blk = std_(np.stack([succ, rev["e_dres"][ev].astype(np.float64)], 1).astype(np.float32))
    lin_z = ((rev["e_own_z"].astype(np.float32) - zmu[None]) / zsd[None]
             ).reshape(ne, -1)[ev]                                   # all 9 z's, standardised
    lin_a = np.zeros((ne, bud * (mask_slot + 1)), np.float32)
    for t in range(bud):
        lin_a[np.arange(ne), t * (mask_slot + 1) + rev["e_own_a"][:, t].astype(np.int64)] = 1.0
    t_oh = np.zeros((ne * n_own, n_own), np.float32)
    t_oh[np.arange(ne * n_own), tix] = 1.0
    lineage_blk = np.concatenate([lin_z, lin_a[ev], t_oh,
                                  std_(nex[ev][:, None].astype(np.float32))], 1)
    readout_blk = std_(np.concatenate([V0[:, None], PI0], 1))
    rungs = {"1_state_action": base,
             "2_plus_grade": np.concatenate([base, grade_blk], 1),
             "3_plus_lineage": np.concatenate([base, grade_blk, lineage_blk], 1),
             "4_plus_readout": np.concatenate([base, grade_blk, lineage_blk, readout_blk], 1)}

    # ---------------- FM targets ----------------
    T57 = np.concatenate([DV[:, None], DPI], 1)
    has_a = tix < bud
    tgt_blocks = {
        "own57  (phase 2's target shape; dpi==0 on 79% of rows)": (T57, has_a),
        "dv1    (the value component alone; never zero)": (DV[:, None], has_a),
        "own57@pi (rows whose event stepped pi)": (T57, has_a & did_pi),
    }

    # ==================================================================
    # Q1 -- IS THE PER-DATUM REVISION FORECASTABLE AT ALL, AND BY WHAT?
    # ==================================================================
    out("")
    out("=" * 78)
    out("(Q1) THE CONDITIONING LADDER  -- forecastability of the per-datum revision")
    out("=" * 78)
    out("  Same FM2 class, same recipe and same cycle-held-out discipline as phase 2, so the")
    out("  `(s,a)` rung is directly comparable to au_s0's. Three target blocks, because")
    out("  `e_did_pi == e_succ`: on the joint target the `+grade` rung is handed the fact that")
    out("  dpi is identically zero on 79% of rows, which is WIRING, not forecasting skill.")
    res = {"rungs": {}, "n_events": int(ne)}
    caps = [int(q) for q in args.fm_caps.split(",")][:2]
    fms = {}
    for tname, (Y, m) in tgt_blocks.items():
        ymu = Y[fm_tr[m[fm_tr]]].mean(0, keepdims=True)
        ysd = np.maximum(Y[fm_tr[m[fm_tr]]].std(0, keepdims=True), 1e-6)
        Ys = ((Y - ymu) / ysd).astype(np.float32)
        tr_ = fm_tr[m[fm_tr]]
        de_ = dec[m[dec]]
        out("")
        out(f"  target {tname}   [{tr_.size} fit rows, {de_.size} held-out rows]")
        out(f"    {'rung':18s} {'d_in':>5s} {'FM':>6s} {'|r|/|d|':>9s} {'cos':>8s} {'in-cos':>8s}")
        for hid in caps:
            for rname, X in rungs.items():
                net = train_fm2(X, Ys, tr_, hidden=hid, seed=args.seed + 4441,
                                steps=args.fm_steps, batch=512, lr=1e-3)
                p = predict(net, X[de_])
                dd_ = Ys[de_]
                rel = float(np.linalg.norm(dd_ - p, axis=1).mean()
                            / np.linalg.norm(dd_, axis=1).mean())
                co = _cos(p, dd_)
                sub = tr_[::7]
                ci_ = _cos(predict(net, X[sub]), Ys[sub])
                out(f"    {rname:18s} {X.shape[1]:>5d} {('h%d' % hid):>6s} {rel:>9.3f} "
                    f"{co:>+8.3f} {ci_:>+8.3f}")
                res["rungs"][f"{tname.split()[0]}|{rname}|h{hid}"] = {
                    "d_in": int(X.shape[1]), "rel_r": rel, "cos": co, "cos_in": ci_}
                if hid == caps[0] and tname.startswith("own57 "):
                    fms[rname] = (net, X)
                del net, p
    out("")
    out("")
    out("  READ Q1's `dv1 / +grade` ROW TOGETHER WITH Q2, NOT BESIDE IT. The grade enters the")
    out("  forecaster only through delta = sigmoid(v0)-y, and Q2 measures how much of dv_t that")
    out("  scalar already explains on its own. The two are the SAME fact seen twice, and adding")
    out("  them would be double-counting.")
    out("  au_s0 (POOLED) at the same rung and capacity, for the contrast:")
    out("    1_state_action     161    h64     1.069   +0.003   +0.134")
    out("    1_state_action     161   h256     1.118   -0.000   +0.266")
    out("    (au_s0's within-cycle ceiling, an FM that had seen the cycle: 1.057 at cos +0.141)")

    # ---------------- the matched pair, for g ----------------
    Ysj = ((T57 - T57[fm_tr[has_a[fm_tr]]].mean(0, keepdims=True))
           / np.maximum(T57[fm_tr[has_a[fm_tr]]].std(0, keepdims=True), 1e-6)).astype(np.float32)
    fm1, fm2 = train_pair(Xz, Xr, A, Ysj, fm_tr[has_a[fm_tr]], hidden=caps[0],
                          seed=args.seed + 4441, steps=args.fm_steps, batch=512, lr=1e-3,
                          mask_slot=mask_slot)
    p2 = fm_predict(fm2, Xz[dec], Xr[dec], A[dec], mask_slot)
    p1 = fm_predict(fm1, Xz[dec], Xr[dec], mask_slot, mask_slot)
    r_sa = Ysj[dec] - p2
    gap = p2 - p1
    gnorm = np.linalg.norm(gap, axis=1)
    res["arity_gap_rel"] = float(gnorm.mean() / np.linalg.norm(Ysj[dec], axis=1).mean())
    out(f"  arity gap |g|/|Delta| at the (s,a) rung: {res['arity_gap_rel']:.3f} "
        f"(au_s0: 0.303 at h64)")

    # ---------------- guards ----------------
    ens = [r_sa]
    for j in range(1, max(2, args.ens)):
        a1, a2 = train_pair(Xz, Xr, A, Ysj, fm_tr[has_a[fm_tr]], hidden=caps[0],
                            seed=args.seed + 4441 + 977 * j, steps=args.fm_steps, batch=512,
                            lr=1e-3, mask_slot=mask_slot)
        ens.append(Ysj[dec] - fm_predict(a2, Xz[dec], Xr[dec], A[dec], mask_slot))
        del a1, a2
    res["ens_cos"] = _ensemble_cos(ens)
    del ens
    out(f"  ens_cos over {max(2, args.ens)} independent-seed FM2 residuals: {res['ens_cos']:.3f}")
    return _e1b_part2(args, root, outdir, out, res, locals())


def _e1b_part2(args, root, outdir, out, res, L):
    """Q2-Q4 and the comparability row. `L` is Q1's frame, passed rather than recomputed."""
    g = lambda k: L[k]
    ne, n_own, bud, n_slots = g("ne"), g("n_own"), g("bud"), g("n_slots")
    rev, dec, ptr, pte, grp, strata = (g("rev"), g("dec"), g("ptr"), g("pte"), g("grp"),
                                       g("strata"))
    ev, tix, V0, DV, DPI, A = g("ev"), g("tix"), g("V0"), g("DV"), g("DPI"), g("A")
    succ, did_pi, nex, era, cyc = g("succ"), g("did_pi"), g("nex"), g("era"), g("cyc")
    has_a, base, rungs, caps = g("has_a"), g("base"), g("rungs"), g("caps")
    r_sa, gap, gnorm, Ysj = g("r_sa"), g("gap"), g("gnorm"), g("Ysj")
    mask_slot = g("mask_slot")

    # ==================================================================
    # Q2 -- THE r-BECOMES-delta IDENTITY, MEASURED
    # ==================================================================
    out("")
    out("=" * 78)
    out("(Q2) THE VALUE STEP: HOW MUCH OF IT IS THE ANALYTIC delta?")
    out("=" * 78)
    out("  BCE's logit gradient is delta_t = sigmoid(v0_t) - y, so if `Delta v` at the datum's")
    out("  OWN states were simply -eta*delta_t, then `a forecaster without the grade leaves a")
    out("  residual that IS the vector delta` would be an IDENTITY about the loss, not a finding")
    out("  about self-models. WHAT THIS CELL CAN MEAN: a high R^2 from delta alone means the")
    out("  value component is wiring and nothing built on it is evidence about forecasting.")
    out("  WHAT IT CANNOT MEAN: it says nothing about the pi component or about spillover, which")
    out("  are where the generalization geometry lives. Adam normalises per-parameter, and one")
    out("  step is shared by all 9 states, so the identity is not guaranteed a priori.")
    dlt = (_sig(V0) - succ)
    a9 = has_a | True                                     # all 9 states: the value step saw them
    tr_, te_ = ptr, pte
    q2 = {}
    q2src = {
        "delta_t = sigmoid(v0)-y  [1]": dlt[dec][:, None].astype(np.float32),
        "(v0, y)                  [2]": np.stack([V0[dec], succ[dec]], 1).astype(np.float32),
        "all 9 deltas of the event [9]": dlt.reshape(ne, n_own)[ev[dec]].astype(np.float32),
        "+ z                       [.]": np.concatenate(
            [dlt[dec][:, None], g("Xz")[dec]], 1).astype(np.float32),
    }
    ydv = {"own dv_t": (DV[dec].astype(np.float64), np.ones(dec.size, bool))}
    out(f"    {'source':30s} {'R^2(dv_t)':>10s}")
    for nm, X in q2src.items():
        d_ = decode(X, ydv, tr_, te_, caps=[0, 64], seed=args.seed, steps=args.probe_steps,
                    batch=512, lr=1e-3, groups=grp)
        b = max(d_[f"h{h}"]["own dv_t"]["r2"] for h in (0, 64))
        q2[nm] = b
        out(f"    {nm:30s} {b:>10.3f}")
    sa = float((np.sign(DV[dec]) == np.sign(-dlt[dec])).mean())
    cc = float(np.corrcoef(DV[dec], -dlt[dec])[0, 1])
    out(f"    raw corr(dv_t, -delta_t) = {cc:+.3f}; sign agreement {sa:.1%}")
    out(f"    |dv| by |delta| decile is NOT monotone (Adam normalises): "
        + " ".join(f"{np.abs(DV[dec][(np.abs(dlt[dec]) >= lo) & (np.abs(dlt[dec]) < hi)]).mean():.2e}"
                   for lo, hi in ((0, .1), (.1, .5), (.5, .9), (.9, 1.01))))
    res["q2_value_identity"] = q2
    res["q2_sign_agree"], res["q2_corr"] = sa, cc

    # ==================================================================
    # Q3 -- DOES THE VECTOR FORM CARRY ANYTHING BEYOND THE SCALAR delta?
    # ==================================================================
    out("")
    out("=" * 78)
    out("(Q3) INCREMENTAL DECODE OVER THE SCALAR delta THAT IS ALREADY FREE ON THE WIRE")
    out("=" * 78)
    out("  Every cell below is R^2 with a COVARIATE BASELINE already in it: the per-state")
    out("  analytic delta = sigmoid(v0)-y, its mean, the grade, and log|Delta_i| -- i.e. every")
    out("  SCALAR that is already free on the wire, INCLUDING the update's own magnitude. The")
    out("  first row IS that baseline; every later row is the same fit with a VECTOR source")
    out("  added, so the gap between them is what the vector form buys.")
    out("  WHAT THESE CELLS CAN MEAN: an increment over the baseline row means the VECTOR update")
    out("  carries spillover content the free SCALARS do not. WHAT THEY CANNOT MEAN: nothing here")
    out("  licenses a claim about privacy (that is Q4's ladder), and the RAW columns are not the")
    out("  evidence -- only the [cs] columns are, because a gain that vanishes under cycle x")
    out("  update-order residualisation was the calendar and not the datum.")
    out("  The earlier `ref rms / own rms` ratio target was DROPPED: `own_rms` sat in both the")
    out("  target and every update-derived source, so those cells scored by tautology. It is now")
    out("  a covariate instead, which is what makes the remaining increments mean anything.")
    prb = rev["e_prb_dv"].astype(np.float64)
    ref = rev["e_ref_dv"].astype(np.float64)
    own_rms = np.sqrt((rev["e_own_dv"].astype(np.float64) ** 2).mean(1))
    # DIRECTION targets, which is where `generalization geometry` actually lives. A raw 32-vector
    # is not comparable across cycles (the reference set is redrawn every cycle) or across eras
    # (the probe changes), but the ANGLE between this event's spillover and the LEAVE-ONE-OUT
    # mean spillover of its own cycle/era is a scalar in [-1,1] and is comparable everywhere.
    # It asks: how idiosyncratic is the direction in which this datum moves the learner
    # elsewhere, relative to what the other data of its own cycle did?
    def _loo_dir_cos(M, key):
        kk = np.asarray(key)
        outv = np.zeros(ne)
        for u in np.unique(kk):
            i_ = np.nonzero(kk == u)[0]
            if i_.size < 3:
                continue
            S = M[i_].sum(0, keepdims=True)
            loo = (S - M[i_]) / max(i_.size - 1, 1)
            num = (M[i_] * loo).sum(1)
            den = np.linalg.norm(M[i_], axis=1) * np.linalg.norm(loo, axis=1)
            outv[i_] = num / np.maximum(den, 1e-12)
        return outv

    era_e = np.asarray([g("cyc_meta")[int(c)]["era"] for c in rev["e_cycle"]])
    e_tgt = {
        "spill_ref_rms (magnitude)": (np.log(np.sqrt((ref ** 2).mean(1)) + 1e-12),
                                      np.ones(ne, bool)),
        "spill_dir_ref (geometry)": (_loo_dir_cos(ref, rev["e_cycle"]), np.ones(ne, bool)),
        "spill_dir_prb (geometry)": (_loo_dir_cos(prb, era_e), np.ones(ne, bool)),
    }
    # event-level rows, split by the same cycles / events
    e_is_test = np.isin(rev["e_cycle"], np.unique(cyc[dec]))
    e_dec = np.nonzero(e_is_test)[0]
    e_ptr, e_pte = _group_split(np.arange(e_dec.size), e_dec, 0.25, args.seed + 11)
    e_str = (np.searchsorted(np.unique(rev["e_cycle"]), rev["e_cycle"][e_dec]) * 8
             + np.minimum(rev["e_order"][e_dec] // 16, 7)).astype(np.int64)
    e_tgt_cs = {}
    for k, (y, m) in e_tgt.items():
        e_tgt_cs[k] = (y[e_dec], m[e_dec])
        e_tgt_cs[k + " [cs]"] = (_stratum_residualise(y[e_dec], e_str, e_ptr), m[e_dec])
    # event-level sources
    flat = lambda M: M.reshape(ne, -1)[e_dec].astype(np.float32)
    r_ev = np.zeros((ne, n_own, Ysj.shape[1]), np.float32)
    r_ev.reshape(ne * n_own, -1)[dec] = r_sa
    g_ev = np.zeros((ne, n_own, Ysj.shape[1]), np.float32)
    g_ev.reshape(ne * n_own, -1)[dec] = gap
    dlt_ev = (_sig(rev["e_own_v0"].astype(np.float64))
              - rev["e_succ"].astype(np.float64)[:, None])
    # THE BASELINE HOLDS EVERY SCALAR THAT IS ALREADY FREE ON THE WIRE: the per-state analytic
    # delta, its mean, the grade, AND log|Delta_i| -- the own-update magnitude. Without that last
    # term the ratio target would have `own_rms` on both sides and every update-derived source
    # would score by tautology; with it, an increment means the VECTOR form carries something the
    # free SCALARS do not, which is the question.
    cov = np.concatenate([dlt_ev[e_dec], dlt_ev[e_dec].mean(1, keepdims=True),
                          rev["e_succ"][e_dec][:, None],
                          np.log(own_rms[e_dec] + 1e-12)[:, None]], 1).astype(np.float32)
    e_src = {
        "baseline (free scalars)": cov,
        "+ Delta_i (raw update, vec)": np.concatenate([cov, flat(rev["e_own_dv"])], 1),
        "+ r        (residual, s,a)": np.concatenate([cov, r_ev.reshape(ne, -1)[e_dec]], 1),
        "+ g        (arity gap)": np.concatenate([cov, g_ev.reshape(ne, -1)[e_dec]], 1),
        "+ own dpi  (the pi step)": np.concatenate(
            [cov, np.sqrt((rev["e_own_dpi"][e_dec].astype(np.float32) ** 2).mean(2))], 1),
    }
    names = list(e_tgt_cs)
    out("")
    out(f"    {'source':30s} " + "  ".join(f"{k.split(' ')[0][:13]:>13s}" for k in names))
    q3 = {}
    for nm, X in e_src.items():
        d_ = decode(X, e_tgt_cs, e_ptr, e_pte, caps=[0, 64], seed=args.seed,
                    steps=args.probe_steps, batch=512, lr=1e-3, groups=e_dec)
        row = {k: max(d_[f"h{h}"].get(k, {"r2": float("nan")})["r2"] for h in (0, 64))
               for k in names}
        q3[nm] = row
        out(f"    {nm:30s} " + "  ".join(_fmt(row[k], 13, 3) for k in names))
    res["q3_spillover"] = q3
    out("  Columns are raw then [cs] (residualised on cycle x update-order). `spill_*_rms` is an")
    out("  RMS over 32 states -- size-invariant, comparable across cycles. The DIRECTION targets")
    out("  are leave-one-out cosines against the cycle's (reference) or era's (probe) own mean")
    out("  spillover, which is how a direction is made comparable when the state set is redrawn.")
    L["q3_dir_ref"] = e_tgt["spill_dir_ref (geometry)"][0]
    L["r_ev"] = r_ev
    return _e1b_part3(args, root, outdir, out, res, L, e_dec, e_ptr, e_pte, e_str, cov, q3)


def _e1b_part3(args, root, outdir, out, res, L, e_dec, e_ptr, e_pte, e_str, cov, q3):
    """Q4 (agency + the observer twin), the nulls, and the whole-cycle comparability row."""
    g = lambda k: L[k]
    ne, n_own, n_slots = g("ne"), g("n_own"), g("n_slots")
    rev, dec, ptr, pte, grp = g("rev"), g("dec"), g("ptr"), g("pte"), g("grp")
    ev, DV, A, succ, nex = g("ev"), g("DV"), g("A"), g("succ"), g("nex")
    strata, mask_slot, caps = g("strata"), g("mask_slot"), g("caps")
    r_sa, gnorm, Ysj = g("r_sa"), g("gnorm"), g("Ysj")

    # ==================================================================
    # Q4 -- AGENCY AND THE OBSERVER TWIN
    # ==================================================================
    out("")
    out("=" * 78)
    out("(Q4) AGENCY, THE OBSERVER TWIN, AND THE NULLS")
    out("=" * 78)
    own_rms = np.sqrt((rev["e_own_dv"].astype(np.float64) ** 2).mean(1))
    ref_rms = np.sqrt((rev["e_ref_dv"].astype(np.float64) ** 2).mean(1))
    out("  arity gap and residual by the datum's explore count (the agency axis):")
    out(f"    {'n_explore':>10s} {'events':>8s} {'|g|/|d|':>9s} {'|r|/|d|':>9s} "
        f"{'dv rms':>10s} {'spill':>8s} {'solved':>8s}")
    ag = {}
    dn_ = np.linalg.norm(Ysj[dec], axis=1)
    for k in range(0, 7):
        m_e = nex == k
        m_s = m_e[ev[dec]]
        if m_e.sum() < 30:
            continue
        ag[k] = {"n": int(m_e.sum()),
                 "g_rel": float(gnorm[m_s].mean() / dn_[m_s].mean()),
                 "r_rel": float(np.linalg.norm(r_sa[m_s], axis=1).mean() / dn_[m_s].mean()),
                 "dv_rms": float(np.median(own_rms[m_e])),
                 "spill": float(np.median(ref_rms[m_e] / (own_rms[m_e] + 1e-12))),
                 "solved": float(rev["e_succ"][m_e].mean())}
        a = ag[k]
        out(f"    {k:>10d} {a['n']:>8d} {a['g_rel']:>9.3f} {a['r_rel']:>9.3f} "
            f"{a['dv_rms']:>10.3e} {a['spill']:>8.3f} {a['solved']:>8.1%}")
    res["agency"] = ag

    # decode of spillover from provenance alone, and the observer twin
    # the observer ladder is scored on the GEOMETRY target, not the magnitude one: magnitude is
    # largely `how big was this step`, which the free scalars already carry.
    e_tgt = {"spill_dir_ref [cs]": (_stratum_residualise(
        g("q3_dir_ref")[e_dec], e_str, e_ptr), np.ones(e_dec.size, bool))}
    vx = int(rev["e_own_x"].max()) + 1
    T = rev["e_own_x"].shape[2]
    x0 = rev["e_own_x"][:, 0, :].astype(np.int64)[e_dec]
    Xo = np.zeros((e_dec.size, T * vx), np.float32)
    Xo[np.arange(e_dec.size)[:, None], np.arange(T)[None, :] * vx + x0] = 1.0
    Aoh = np.zeros((e_dec.size, rev["e_own_a"].shape[1] * (mask_slot + 1)), np.float32)
    for t in range(rev["e_own_a"].shape[1]):
        Aoh[np.arange(e_dec.size),
            t * (mask_slot + 1) + rev["e_own_a"][e_dec, t].astype(np.int64)] = 1.0
    grade_o = rev["e_succ"][e_dec][:, None].astype(np.float32)
    obs_sets = {
        "O_x            (public)": Xo,
        "O_x+a          (public)": np.concatenate([Xo, Aoh], 1),
        "O_x+a+grade    (public)": np.concatenate([Xo, Aoh, grade_o], 1),
        "SELF: free scalars only": cov,
        "SELF: z0+a+grade": np.concatenate(
            [rev["e_own_z"][e_dec, 0].astype(np.float32), Aoh, grade_o], 1),
        "SELF: + Delta_i (vector)": np.concatenate(
            [cov, rev["e_own_dv"][e_dec].astype(np.float32)], 1),
        "SELF: + r (residual)": np.concatenate([cov, g("r_ev").reshape(ne, -1)[e_dec]], 1),
        "n_explore only  (public)": nex[e_dec][:, None].astype(np.float32),
    }
    out("")
    out("  spillover DIRECTION (leave-one-out cos vs the cycle's own mean spillover),")
    out("  residualised on cycle x order -- self vs the observer twin:")
    obs = {}
    for nm, X in obs_sets.items():
        d_ = decode(X, e_tgt, e_ptr, e_pte, caps=[0, 64], seed=args.seed + 200,
                    steps=args.obs_steps, batch=512, lr=1e-3, groups=e_dec)
        obs[nm] = max(d_[f"h{h}"]["spill_dir_ref [cs]"]["r2"] for h in (0, 64))
        out(f"    {nm:26s} d={X.shape[1]:<5d} R^2={_fmt(obs[nm], 7, 3)}")
    half, _ = _group_split(e_ptr, e_dec, 0.5, args.seed + 303)
    # standardised ONCE on the full probe-train rows and reused: re-standardising on half the
    # rows makes a one-hot column that happens to be constant in that half explode, which is a
    # fact about the standardiser and not about the budget.
    Xh = standardise_(np.array(obs_sets["O_x+a+grade    (public)"], copy=True), e_ptr)
    d_ = decode(Xh, e_tgt, half, e_pte, caps=[64], seed=args.seed + 201,
                steps=args.obs_steps, batch=512, lr=1e-3, groups=e_dec, standardise=False)
    out(f"    {'O_x+a+grade half-data':26s}       R^2="
        f"{_fmt(d_['h64']['spill_dir_ref [cs]']['r2'], 7, 3)}   (budget control)")
    res["q4_observer"] = obs

    # ---- nulls ----
    out("")
    out("  NULLS on the same target (all should sit at ~0):")
    perm = np.arange(e_dec.size)
    rs = np.random.default_rng(args.seed + 99)
    for s_ in np.unique(e_str):
        i_ = np.nonzero(e_str == s_)[0]
        perm[i_] = i_[rs.permutation(i_.size)]
    nulls = {"Delta_i shuffled within cycle x order":
             rev["e_own_dv"][e_dec].astype(np.float32)[perm],
             "own dpi rms shuffled likewise":
             np.sqrt((rev["e_own_dpi"][e_dec].astype(np.float32) ** 2).mean(2))[perm]}
    for nm, X in nulls.items():
        d_ = decode(X, e_tgt, e_ptr, e_pte, caps=[0, 64], seed=args.seed,
                    steps=args.probe_steps, batch=512, lr=1e-3, groups=e_dec)
        v = max(d_[f"h{h}"]["spill_dir_ref [cs]"]["r2"] for h in (0, 64))
        out(f"    {nm:44s} R^2={_fmt(v, 7, 3)}")
        res.setdefault("q4_nulls", {})[nm] = v

    # ==================================================================
    # THE COMPARABILITY ROW -- the phase 2 (s,a) FM class on this arm's WHOLE-CYCLE Delta
    # ==================================================================
    out("")
    out("=" * 78)
    out("(C) COMPARABILITY -- the SAME phase-2 FM class on au_s1/perdatum's WHOLE-CYCLE Delta")
    out("=" * 78)
    out("  Same code path as phase 2 (`build_table` + `train_pair`), pointed at this arm's own")
    out("  boundary snapshots, so the pooled-vs-per-datum contrast is also stated at the CYCLE")
    out("  granularity on the same trajectories. This target INCLUDES the maintenance pass.")
    if args.e1b_no_cycle_row:
        out("  [skipped: --e1b-no-cycle-row]")
        return res
    ccache = os.path.join(outdir, "table_v2_r4096.npz")
    if os.path.isfile(ccache) and not args.no_cache:
        tab = dict(np.load(ccache))
    else:
        tab = build_table(root, rows_per_cycle=4096, seed=args.seed)
        np.savez(ccache, **tab)
    n2 = tab["cycle"].size
    cy2 = np.unique(tab["cycle"])
    te2 = cy2[3::4]
    ist2 = np.isin(tab["cycle"], te2)
    d2, p2pool = np.nonzero(ist2)[0], np.nonzero(~ist2)[0]
    rr = np.random.default_rng(args.seed + 7)
    tr2 = np.sort(rr.choice(p2pool, min(args.fm_rows, p2pool.size), replace=False))
    z2 = tab["z"]
    Xz2 = ((z2 - z2[tr2].mean(0, keepdims=True))
           / np.maximum(z2[tr2].std(0, keepdims=True), 1e-6)).astype(np.float32)
    nr2 = int(tab["root"].max()) + 1
    Xr2 = np.zeros((n2, nr2), np.float32)
    Xr2[np.arange(n2), tab["root"].astype(np.int64)] = 1.0
    D2 = np.concatenate([tab["dv"][:, None], tab["dpi"]], 1).astype(np.float32)
    D2 = ((D2 - D2[tr2].mean(0, keepdims=True))
          / np.maximum(D2[tr2].std(0, keepdims=True), 1e-6)).astype(np.float32)
    u2 = tab["u_slot"].astype(np.int64)
    ms2 = int(tab["n_slots"])
    out(f"    {'FM':>6s} {'|r|/|d|':>9s} {'cos':>8s} {'in-cos':>8s} {'|g|/|d|':>9s}")
    for hid in caps:
        a1, a2 = train_pair(Xz2, Xr2, u2, D2, tr2, hidden=hid, seed=args.seed + 4441,
                            steps=args.fm_steps, batch=512, lr=1e-3, mask_slot=ms2)
        q2_ = fm_predict(a2, Xz2[d2], Xr2[d2], u2[d2], ms2)
        q1_ = fm_predict(a1, Xz2[d2], Xr2[d2], ms2, ms2)
        dd2 = D2[d2]
        rel = float(np.linalg.norm(dd2 - q2_, axis=1).mean() / np.linalg.norm(dd2, axis=1).mean())
        sub = tr2[::7]
        row = {"rel_r": rel, "cos": _cos(q2_, dd2),
               "cos_in": _cos(fm_predict(a2, Xz2[sub], Xr2[sub], u2[sub], ms2), D2[sub]),
               "g_rel": float(np.linalg.norm(q2_ - q1_, axis=1).mean()
                              / np.linalg.norm(dd2, axis=1).mean())}
        res.setdefault("cycle_row", {})[f"h{hid}"] = row
        out(f"    {('h%d' % hid):>6s} {rel:>9.3f} {row['cos']:>+8.3f} {row['cos_in']:>+8.3f} "
            f"{row['g_rel']:>9.3f}")
        del a1, a2, q1_, q2_
    out("    au_s0/anchor, the pooled arm, same class:  h64 1.069 +0.003 +0.134 0.303")
    out("                                               h256 1.118 -0.000 +0.266 0.356")
    return res


# ======================================================================
# (9) THE SYMMETRIC RE-ENCODE CHECK  --  `--e1b-reencode`, a fourth disjoint mode
# ======================================================================
#
# WHAT IT IS FOR. Finding 8(i) of the node's README records one unsettled cell: on the
# spillover-DIRECTION target, `O_x` -- an observer holding the raw public configuration as a
# 64x8 one-hot -- reaches R^2 = 0.289 while every self source sits <= 0.062, including
# `SELF: z0+a+grade` at 0.036, a twin holding (nominally) the same configuration in the
# learner's own frozen 96-dim fp16 encoding. The node flagged rather than claimed the cell for
# two stated reasons: the rungs are unstable (adding the action LOWERS the observer, 0.289 ->
# 0.098; the half-data control goes negative), and a one-hot-`x` vs fp16-`z` ENCODING ASYMMETRY
# is unexcluded -- the two sides differ not only in whose information they hold but in the input
# FORMAT they hold it in.
#
# WHAT "SYMMETRIC" HAS TO MEAN. The incumbent pair confounds four axes at once:
#   (1) CONTENT      x0  vs  z0 = enc(x0)
#   (2) BASIS        a sparse indicator code (64 of 512 columns hot, every "token j at
#                    position t" a free LINEAR feature) vs a dense real code (96 coordinates,
#                    every such indicator a function the probe must first learn)
#   (3) WIDTH        512 free features vs 96
#   (4) CONDITIONING `O_x` is state-ONLY; the self twin also carries the 456-dim action block
#                    and the grade -- exactly the two additions that demonstrably LOWER the
#                    observer when it gets them.
# and, in the background, (5) PRECISION: one-hot values are exact in fp16, `z` is stored in it.
#
# So this mode crosses CONTENT x BASIS explicitly, holding conditioning fixed, and then runs the
# conditioning axis as its own contrast on the four principal cells. Both directions of
# re-encoding are built, because only running both can distinguish "the encoder discarded it"
# from "a dense code of any origin is unreadable here":
#
#            |  sparse indicator basis        |  dense 96-d fp16 code
#   ---------+--------------------------------+--------------------------------
#     x      |  one-hot 64x8 = 512  [INCUMBENT]|  x_oh @ G  (random, 96) AND
#            |                                |  x_oh -> PCA-96      <- NEW
#   ---------+--------------------------------+--------------------------------
#     z      |  96 dims x 8 quantile bins     |  z0 as logged, 96    <- NEW
#            |  = 768, 96 hot        <- NEW   |  (never run WITHOUT a+grade before)
#
# The two dense-x cells bracket the compression: the random projection is the format-only
# control (no information selection at all), PCA-96 is the GENEROUS one (the best linear 96-dim
# summary of x, fitted on probe-train rows only). If even PCA-96 collapses, no 96-dim dense code
# of x carries this target readably and `z`'s 0.036 is a fact about format. If PCA-96 holds at
# ~0.29 while `z` stays at ~0.04, the learner's encoder specifically is where it goes.
# Two further z cells separate BASIS from WIDTH, which the grid above still bundles:
#   * `z -> random LINEAR lift to 512`: a width-only change. Ridge is exactly invariant to an
#     injective linear map and an MLP nearly so, so this cell is a NULL -- it must reproduce
#     `z`'s own number, and if it does, width is not the axis and anything the binned cell moves
#     is about the BASIS.
#   * `z -> 512 random ReLU features`: the observer's width AND a fixed nonlinear basis, without
#     discretisation.
# Precision is handled by construction rather than by a cell: {0,1} is exact in fp16, so the
# precision axis cannot favour the incumbent observer, while every dense-x code here is put
# through an fp16 round trip so the x side PAYS `z`'s storage cost.
#
# A CONFOUND THE GRID SURFACED, recorded here because it is prior to the format question.
# `e_own_x[:, 0]` is the datum's ROOT state -- the corrupted problem its trajectory started
# from -- and on this arm `(cycle, e_inst)` <-> `(cycle, x0)` is a BIJECTION over all 14,848
# events. So `O_x` holds instance identity and nothing else about the datum. The `--e1b` probe
# split is by EVENT (`_group_split(..., groups=e_dec)`, i.e. every event its own group), and
# 86.6% of events share their `(cycle, x0)` with at least one other event, so for most test rows
# an EXACT input duplicate sits in the probe-train set. That is bug #1 of the phase-2 log
# (`_group_split`'s docstring) one level up, and it is FORMAT-GATED: a 512-column indicator code
# can express an instance lookup as a linear combination, a 96-dim dense code cannot. Every cell
# is therefore run twice -- under the incumbent EVENT split (which reproduces the published
# numbers) and under an INSTANCE-grouped split on `(cycle, e_inst)` -- and a group-mean
# instance-identity predictor rides along as the leak's own ceiling.
#
# DISCIPLINE, unchanged from the file's: one train-only standardisation per source inside
# `decode`, the same capacity sweep on every cell (ridge + MLP, `--obs-caps`), a shuffled null
# within cycle x update-order, and the half-data budget control -- here on the PRINCIPAL cells
# rather than only on the top rung, because running it only where R^2 is already 0.085 is what
# made the incumbent control uninformative.
#
# Writes `figures/<tag>/fit/reencode.{txt,json}`. `reduction.txt` is not touched.
# ======================================================================


def _loo_dir_cos_ev(M, key):
    """`_e1b_part2`'s spillover-direction target, verbatim, as a module-level function so this
    mode forms exactly the same object rather than an equivalent-looking one."""
    M = np.asarray(M, np.float64)
    kk = np.asarray(key)
    outv = np.zeros(M.shape[0])
    for u in np.unique(kk):
        i_ = np.nonzero(kk == u)[0]
        if i_.size < 3:
            continue
        S = M[i_].sum(0, keepdims=True)
        loo = (S - M[i_]) / max(i_.size - 1, 1)
        num = (M[i_] * loo).sum(1)
        den = np.linalg.norm(M[i_], axis=1) * np.linalg.norm(loo, axis=1)
        outv[i_] = num / np.maximum(den, 1e-12)
    return outv


def _fp16(X):
    """The storage round trip `z` already paid, applied to a re-encoding of `x`."""
    return X.astype(np.float16).astype(np.float32)


def _onehot(idx, k):
    n, T = idx.shape
    O = np.zeros((n, T * k), np.float32)
    O[np.arange(n)[:, None], np.arange(T)[None, :] * k + idx] = 1.0
    return O


def _group_mean_predict(y, groups, tr, te):
    """R^2 of the best possible GROUP-IDENTITY predictor: each test row gets its group's mean
    target over train rows, and the train global mean when its group is unseen. Under the event
    split this is the ceiling on what instance memorisation alone can buy; under the instance
    split every test group is unseen by construction, so it must fall to ~0."""
    y = np.asarray(y, float)
    gtr = np.asarray(groups)[tr]
    gm = float(y[tr].mean())
    tot, cnt = {}, {}
    for gg, yy in zip(gtr.tolist(), y[tr].tolist()):
        tot[gg] = tot.get(gg, 0.0) + yy
        cnt[gg] = cnt.get(gg, 0) + 1
    seen = 0
    yhat = np.empty(te.size)
    for i, gg in enumerate(np.asarray(groups)[te].tolist()):
        if gg in cnt:
            yhat[i] = tot[gg] / cnt[gg]
            seen += 1
        else:
            yhat[i] = gm
    return _r2_np(y[te], yhat), seen / max(te.size, 1)


def _eta2_group(y, groups):
    """Fraction of the variance of `y` that lies BETWEEN groups. Descriptive: how much of the
    target is an instance-level property at all."""
    y = np.asarray(y, float)
    gg = np.asarray(groups)
    ss_b = 0.0
    mu = y.mean()
    for u in np.unique(gg):
        i_ = gg == u
        ss_b += i_.sum() * (y[i_].mean() - mu) ** 2
    ss_t = ((y - mu) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def e1b_reencode(args, root, outdir, out):
    """The symmetric re-encode check on finding 8(i). Reads the revision tables only."""
    rev = load_revisions(root)
    audi = json.load(open(os.path.join(root, "audiation.json")))
    n_slots = int(audi["n_slots"])
    mask_slot = n_slots
    ne = rev["e_cycle"].size
    bud = int(rev["e_own_a"].shape[1])

    out("AUDIATION -- THE SYMMETRIC RE-ENCODE CHECK on finding 8(i)")
    out(f"  tag={args.tag} arm={args.arm} seed={args.seed}")
    out("  A fourth disjoint mode. It re-forms `--e1b`'s Q4 frame exactly (same target, same")
    out("  cycle hold-out, same strata, same `decode` stack) and crosses CONTENT x BASIS on the")
    out("  observer/self pair, so that input FORMAT is excluded as the driver of the")
    out("  0.289-vs-0.036 gap. `reduction.txt` is not touched.")
    out("")

    # ---------------- the frame, re-formed exactly as `_e1b_part2` does ----------------
    cycles = np.unique(rev["e_cycle"])
    test_cyc = cycles[3::4]
    e_dec = np.nonzero(np.isin(rev["e_cycle"], test_cyc))[0]
    e_str = (np.searchsorted(cycles, rev["e_cycle"][e_dec]) * 8
             + np.minimum(rev["e_order"][e_dec] // 16, 7)).astype(np.int64)
    y_all = _loo_dir_cos_ev(rev["e_ref_dv"], rev["e_cycle"])
    N = e_dec.size

    # the two grouping keys. `(cycle, e_inst)` is the instance; `e_dec` (every event its own
    # group) is what `--e1b` used and is reproduced here so the published cells come back.
    inst_key = (rev["e_cycle"].astype(np.int64) * 100000
                + rev["e_inst"].astype(np.int64))[e_dec]

    # ---------------- what x0 actually is ----------------
    x_all = rev["e_own_x"][:, 0, :].astype(np.int64)
    vx = int(rev["e_own_x"].max()) + 1
    T = x_all.shape[1]
    xb = [x_all[i].tobytes() for i in range(ne)]
    ck = rev["e_cycle"].astype(np.int64).tolist()
    pair_x, pair_i = {}, {}
    for c_, b_, i_ in zip(ck, xb, rev["e_inst"].astype(np.int64).tolist()):
        pair_x.setdefault((c_, b_), set()).add(i_)
        pair_i.setdefault((c_, i_), set()).add(b_)
    bij = (max(len(v) for v in pair_x.values()) == 1
           and max(len(v) for v in pair_i.values()) == 1)
    cnt_x = {}
    for c_, b_ in zip(ck, xb):
        cnt_x[(c_, b_)] = cnt_x.get((c_, b_), 0) + 1
    shared = float(np.mean([cnt_x[(c_, b_)] > 1 for c_, b_ in zip(ck, xb)]))
    out("  WHAT `x0` IS (prior to the format question):")
    out(f"    `e_own_x[:,0]` is the datum's ROOT configuration. Over all {ne} events,")
    out(f"    (cycle, e_inst) <-> (cycle, x0) is {'a BIJECTION' if bij else 'NOT a bijection'}")
    out(f"    ({len(pair_x)} distinct pairs), so `O_x` holds INSTANCE IDENTITY and nothing")
    out(f"    else about the datum. {shared:.1%} of events share their (cycle, x0) with at")
    out("    least one other event, and `--e1b`'s probe split is by EVENT -- so for most test")
    out("    rows an exact input duplicate sits in the probe-train set. Hence split B below.")
    out("")

    splits = {}
    for sname, gkey in (("A_event  (the --e1b split)", e_dec.copy()),
                        ("B_instance (cycle,e_inst)", inst_key)):
        ptr, pte = _group_split(np.arange(N), gkey, 0.25, args.seed + 11)
        y_cs = _stratum_residualise(y_all[e_dec], e_str, ptr)
        splits[sname] = {"g": gkey, "ptr": ptr, "pte": pte,
                         "tgt": {"spill_dir_ref [cs]": (y_cs, np.ones(N, bool))}}
        out(f"  split {sname}: {np.unique(gkey).size} groups -> "
            f"{ptr.size} probe-train / {pte.size} probe-test events "
            f"({np.unique(e_str).size} strata)")
    out(f"  decode rows are EVENTS ({N} in the {test_cyc.size} held-out cycles), as in Q4.")
    out("")

    # ---------------- the two contents ----------------
    x0 = x_all[e_dec]
    Xoh = _onehot(x0, vx)                                        # 512, the incumbent O_x
    z0 = rev["e_own_z"][e_dec, 0].astype(np.float32)             # 96, fp16 as logged
    Aoh = np.zeros((N, bud * (mask_slot + 1)), np.float32)
    for t in range(bud):
        Aoh[np.arange(N), t * (mask_slot + 1) + rev["e_own_a"][e_dec, t].astype(np.int64)] = 1.0
    grade_o = rev["e_succ"][e_dec][:, None].astype(np.float32)
    # Q4's two self sources that are NOT a state code, rebuilt exactly (d=12 and d=21 as
    # published). They are per-DATUM rather than per-instance, so they are the one place the
    # instance-grouped split could leave something standing, and the grid is not complete
    # against `reduction.txt`'s list without them.
    own_rms = np.sqrt((rev["e_own_dv"].astype(np.float64) ** 2).mean(1))
    dlt_ev = (_sig(rev["e_own_v0"].astype(np.float64))
              - rev["e_succ"].astype(np.float64)[:, None])
    cov = np.concatenate([dlt_ev[e_dec], dlt_ev[e_dec].mean(1, keepdims=True),
                          rev["e_succ"][e_dec][:, None],
                          np.log(own_rms[e_dec] + 1e-12)[:, None]], 1).astype(np.float32)
    cov_dv = np.concatenate([cov, rev["e_own_dv"][e_dec].astype(np.float32)], 1)

    def _std_fit(M, tr):
        mu, sd = M[tr].mean(0, keepdims=True), np.maximum(M[tr].std(0, keepdims=True), 1e-6)
        return ((M - mu) / sd).astype(np.float32)

    def _format_maps(tr):
        """Every re-encoding, fitted on THIS split's probe-train rows only. All of them are
        unsupervised (a projection, a set of quantile edges, a fixed random matrix), so nothing
        about the target enters; refitting per split is only about not letting a held-out
        instance contribute to the map that encodes it."""
        rs = np.random.default_rng(args.seed + 8100)
        G = (rs.standard_normal((Xoh.shape[1], 96)) / np.sqrt(Xoh.shape[1])).astype(np.float32)
        Xd_rand = _fp16(_std_fit(Xoh @ G, tr))
        Xc = Xoh - Xoh[tr].mean(0, keepdims=True)
        w_, V_ = np.linalg.eigh(((Xc[tr].T @ Xc[tr]) / max(tr.size - 1, 1)).astype(np.float64))
        ordr = np.argsort(-w_)
        Xd_pca = _fp16(_std_fit(Xc @ V_[:, ordr[:96]].astype(np.float32), tr))
        evr = float(w_[ordr[:96]].sum() / max(w_.sum(), 1e-12))

        zs = _std_fit(z0, tr)
        binn = lambda M: _onehot(np.stack(
            [np.searchsorted(np.quantile(M[tr, j], np.arange(1, 8) / 8.0), M[:, j])
             for j in range(M.shape[1])], 1).astype(np.int64), 8)
        Zsparse = binn(zs)                                       # 768, 96 hot
        wz, Vz = np.linalg.eigh(np.cov(zs[tr].T).astype(np.float64))
        Zpca64 = _std_fit((zs @ Vz[:, np.argsort(-wz)[:64]].astype(np.float32)), tr)
        Zsp512 = binn(Zpca64)                                    # 512, 64 hot -- MATCHED to O_x
        Wl = (rs.standard_normal((zs.shape[1], 512)) / np.sqrt(zs.shape[1])).astype(np.float32)
        Wr = (rs.standard_normal((zs.shape[1], 512)) / np.sqrt(zs.shape[1])).astype(np.float32)
        br = rs.standard_normal(512).astype(np.float32)
        return {"Xd_rand": Xd_rand, "Xd_pca": Xd_pca, "evr": evr, "Zsparse": Zsparse,
                "Zsp512": Zsp512, "Zlift": (zs @ Wl).astype(np.float32),
                "Zrelu": np.maximum(zs @ Wr + br, 0.0).astype(np.float32)}

    caps = [int(q) for q in args.obs_caps.split(",")]
    inc = [c for c in caps if c in (0, 64)]
    res = {"n_events": int(ne), "n_decode": int(N), "bijection_inst_x0": bool(bij),
           "frac_events_sharing_x0": shared, "caps": caps, "cells": {}}
    out(f"  the one-hot block of `x` has rank <= {T * (vx - 1)}; every dense re-encoding of it")
    out("  below is put through the same fp16 round trip `z` already pays in storage.")
    out("")
    for sname, S in splits.items():
        M = _format_maps(S["ptr"])
        ag = lambda Q: np.concatenate([Q, Aoh, grade_o], 1)
        grid = [
            ("x", "sparse 64x8 one-hot   [INCUMBENT O_x]", Xoh),
            ("x", "dense96 random proj, fp16", M["Xd_rand"]),
            ("x", "dense96 PCA, fp16", M["Xd_pca"]),
            ("z", "dense96 as logged, fp16  [NEW: no a+grade]", z0),
            ("z", "sparse 64x8 bins of PCA-64  [MATCHED to O_x]", M["Zsp512"]),
            ("z", "sparse 96x8 quantile bins (all of z)", M["Zsparse"]),
            ("z", "dense512 random LINEAR lift (width null)", M["Zlift"]),
            ("z", "sparse512 random ReLU features", M["Zrelu"]),
            ("x", "sparse one-hot + a + grade  [INCUMBENT]", ag(Xoh)),
            ("x", "dense96 PCA + a + grade", ag(M["Xd_pca"])),
            ("z", "dense96 as logged + a + grade [INCUMBENT]", ag(z0)),
            ("z", "sparse 64x8 bins of PCA-64 + a + grade", ag(M["Zsp512"])),
            ("self", "free scalars only          [INCUMBENT]", cov),
            ("self", "free scalars + Delta_i     [INCUMBENT]", cov_dv),
        ]
        Zsparse = M["Zsp512"]
        out("  " + "-" * 74)
        out(f"  SPLIT {sname} -- R^2 on `spill_dir_ref [cs]` (cycle x order residualised)")
        out(f"  (PCA-96 of the one-hot keeps {M['evr']:.1%} of this split's train variance)")
        out("  " + "-" * 74)
        out(f"    {'content':>8s}  {'format':44s} {'d':>5s} "
            + " ".join(f"{('h%d' % c):>7s}" for c in caps)
            + f" {'best[0,64]':>11s}")
        for content, fname, X in grid:
            d_ = decode(np.ascontiguousarray(X), S["tgt"], S["ptr"], S["pte"], caps=caps,
                        seed=args.seed + 200, steps=args.obs_steps, batch=512, lr=1e-3,
                        groups=S["g"])
            per = {f"h{c}": d_[f"h{c}"]["spill_dir_ref [cs]"]["r2"] for c in caps}
            best = max(per[f"h{c}"] for c in inc) if inc else float("nan")
            out(f"    {content:>8s}  {fname:44s} {X.shape[1]:>5d} "
                + " ".join(_fmt(per[f'h{c}'], 7, 3) for c in caps)
                + f" {_fmt(best, 11, 3)}")
            res["cells"].setdefault(sname, {})[f"{content}|{fname}"] = {
                "d": int(X.shape[1]), "per_cap": per, "best_0_64": best}
        # ---- the controls, the same ones the file already uses ----
        out("")
        out("    controls on the same split:")
        # the leak's own ceiling: memorise the INSTANCE, always, under whichever split is live.
        r2g, cov_ = _group_mean_predict(S["tgt"]["spill_dir_ref [cs]"][0], inst_key,
                                        S["ptr"], S["pte"])
        out(f"      {'instance-identity group mean (the leak ceiling)':52s} "
            f"R^2={_fmt(r2g, 7, 3)}  ({cov_:.0%} of test rows in a seen instance)")
        res["cells"][sname]["_group_mean_r2"] = r2g
        res["cells"][sname]["_group_mean_cov"] = cov_
        for nm, X in (("O_x one-hot", Xoh), ("z sparse 64x8 (matched)", Zsparse),
                      ("z dense96 as logged", z0)):
            half, _ = _group_split(S["ptr"], S["g"], 0.5, args.seed + 303)
            Xh = standardise_(np.array(X, np.float32, copy=True), S["ptr"])
            d_ = decode(Xh, S["tgt"], half, S["pte"], caps=[64], seed=args.seed + 201,
                        steps=args.obs_steps, batch=512, lr=1e-3, groups=S["g"],
                        standardise=False)
            v = d_["h64"]["spill_dir_ref [cs]"]["r2"]
            out(f"      {('half-data budget, ' + nm):52s} R^2={_fmt(v, 7, 3)}")
            res["cells"][sname][f"_half_{nm}"] = v
        perm = np.arange(N)
        rp = np.random.default_rng(args.seed + 99)
        for s_ in np.unique(e_str):
            i_ = np.nonzero(e_str == s_)[0]
            perm[i_] = i_[rp.permutation(i_.size)]
        for nm, X in (("O_x one-hot", Xoh), ("z sparse 64x8 (matched)", Zsparse)):
            d_ = decode(np.ascontiguousarray(X[perm]), S["tgt"], S["ptr"], S["pte"],
                        caps=inc or [64], seed=args.seed, steps=args.obs_steps, batch=512,
                        lr=1e-3, groups=S["g"])
            v = max(d_[f"h{c}"]["spill_dir_ref [cs]"]["r2"] for c in (inc or [64]))
            out(f"      {('shuffled null within cycle x order, ' + nm):52s} "
                f"R^2={_fmt(v, 7, 3)}")
            res["cells"][sname][f"_null_{nm}"] = v
        out("")

    eta = _eta2_group(splits["A_event  (the --e1b split)"]["tgt"]["spill_dir_ref [cs]"][0],
                      inst_key)
    res["eta2_target_by_instance"] = eta
    out(f"  eta^2(residualised target ~ (cycle, e_inst)) = {eta:.3f} over {N} events in "
        f"{np.unique(inst_key).size} instances")
    out("  (descriptive: how much of the target is an instance-level property at all. With")
    out("   ~2.3 events per instance a chance value is ~n_groups/n = "
        f"{np.unique(inst_key).size / N:.3f}.)")
    out("")
    out("  THE INCUMBENT CELLS, quoted from figures/au_s1/fit/reduction.txt for the contrast:")
    out("    O_x            (public)    d=512   R^2=  0.289")
    out("    O_x+a          (public)    d=968   R^2=  0.098")
    out("    O_x+a+grade    (public)    d=969   R^2=  0.085")
    out("    SELF: free scalars only    d=12    R^2=  0.062")
    out("    SELF: z0+a+grade           d=553   R^2=  0.036")
    out("    O_x+a+grade half-data              R^2= -0.144")
    return res


if __name__ == "__main__":
    main()
