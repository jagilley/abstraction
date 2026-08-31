"""continuo — E2: consolidation state off the residual. The instruction/data bit as a
self-readable, non-audition-shaped gauge.

NAME. A figured bass is a *command*: the continuo player receives a numeral over a bass note
and realises it from a compiled vocabulary. The same music written out is *data*: every note
spelled. The practice learner holds exactly this pair — a committed macro token that pi
proposes and the executor runs (routed, "can't-decompose"), and a base move that is still
expanded through its primitive spelling. E2 asks whether the learner can read that bit off its
own residual.

ASKED IN. ROADMAP.md 4.2 first shape E2, as reframed by 7.1.3 ("can the learner read, off its
own residual, which of its tokens are commands and which are still data?"), and QUEUE.md
"Track E' -> E2", which carries the design warnings this file answers. Paper 2
(papers/forward_self_models_paper2.md) supplies the object (implementation fact), the criterion
(criterion 2: beat the I/O-only observer twin at matched capacity), the clustering (spherical
k-means over residual directions), the component contrast (section 6) and the junk guards
(section 8). Paper 1's saturation law supplies the sub-saturation discipline.

DONOR (untouched, read-only). ../audiation/ — `au_s0/anchor` is a per-decision log of a
bit-identical replay of `conductor`'s `cd_s0/anchor` (0.000e+00 over all 116 cycles), carrying
117 per-cycle head snapshots, the per-cycle action-set context, and the outer loop's panel.
Nothing in this node writes into audiation, runs a substrate, or touches a GPU.


================================================================================
THE DESIGN, AND WHY EACH CALL WAS MADE (disagree with them here)
================================================================================

1. WHAT THE FM PREDICTS: the pi head's own computation.

   a_i = (z, root) — exactly and only what the head reads. `z` is the FROZEN encoder's state
   (audiation design call 2); `root` is the target index.
   a_j = pi's 56 logits.

   Why pi and not the value head, and not the "trunk" of 4.2: the encoder is frozen (so 4.2's
   trunk is the wrong target, as 7.1.3's design note says), and of the two trainable heads only
   pi has a per-TOKEN output space. The instruction/data bit is a property of the action
   vocabulary; the value head emits a scalar and has no index to carry it. Consolidation is
   pi's event besides: at a commit the action set grows, macro slots go live, pi learns to
   route to them, and pi's mass on a chunk's own spelling collapses (`native`'s
   can't-decompose readout).

   The VALUE head rides along as a CONTROL target in the same FM class. It has no token
   structure, so any "the residual reorganises at consolidation" that shows up equally on the
   value head is a fact about the encoder's state distribution or about the calendar, not about
   the command port.

   THE CORRIDOR, RE-SCOPED. `au_s0` is conductor's anchor and runs `span_mode = False`: there
   is no corridor/span head in this donor at all. Consolidation here is ROUTING-ONLY — pi's
   vocabulary gains macro tokens and pi learns to route to them, while the executor stays the
   exogenous beam DP. That is still exactly the instruction/data object (routed macro = command,
   base move = data). An executor-side consolidation (the corridor running a macro in one pass,
   `native`/`spiral`'s span port) is a SECOND, additive question and is named as the follow-up
   rather than smuggled in here.

2. THE PROBE DISTRIBUTION: the practice beam's own states, fixed across checkpoints.

   NOT `probe_trace.npz`. Those are the era's own METERING set — the performance eval. Using
   them would make the gauge audition-shaped, which is the one thing 7.1.3 says E2 must not be.
   The practice beam's tip states are the learner's own working distribution: the states it
   deliberates over and the states its updates are computed at.

   The probe set is FIXED across all 117 checkpoints, so a change in the residual geometry is
   the head changing and not the state distribution changing. A per-cycle "live" probe rides
   along as a robustness read on the cycles whose shards are fetched.

   Warmup cycles 1-4 are excluded (audiation design call 6: the beam enumerated, pi was never
   read). Splits are BY INSTANCE, never by row (audiation's bug 1: a beam's tips at one step of
   one instance are near-duplicates with near-identical targets).

3. SUB-SATURATION. Paper 1's law: the FM saturates at ~the predicted layers' parameter count,
   and a saturated instrument counterfeits the whole result (paper 2 section 8 — the smoke
   artifact returned advantage +0.118 and steering ratio 2.07 from nothing). The predicted span
   here is pi's trunk+out. The FM is given the SAME SHAPE OF READ as the head
   (LayerNorm -> Linear -> GELU -> Linear) so nothing is bought or lost by architecture
   mismatch, and swept from ~1.7% to 100% of the head's parameters. The 100% rung is included
   deliberately: it is this substrate's saturation demonstration, and the guards have to flag
   it.

4. WHAT "OCCUPANCY REORGANISES" IS MEASURED AGAINST. The events are on the record and are not
   chosen after the fact: L2 commit c18 (action set grows at c19), L3 commit c69 (grows at c70),
   era advances c49/c89/c101/c110, recerts. Occupancy is scored against ONE codebook (spherical
   k-means, K=8, fitted on an era-balanced pool of residual directions from all checkpoints and
   then applied unchanged), so per-checkpoint histograms are comparable across time.

5. THE LABEL. Per token, the ground-truth instruction/data bit is checkable: `ms_level[j] >= 2`
   is a committed macro (command, routed), `ms_level[j] == 1` is a base move (data, enumerated).
   Two things this makes necessary:

   (i) DEGENERATE COLUMNS MUST BE EXCLUDED. pi's output layer is zero-init and slots off the
       live action set receive no gradient, so a slot's logit column is EXACTLY 0 until a few
       cycles after its commit. A decoder handed a constant-zero column reads "command"
       perfectly, and that is wiring, not a finding. Every cell is reported over live
       NON-DEGENERATE slots, with the excluded count printed.
   (ii) AGE IS A CONFOUND, AND IS DISAMBIGUATED RATHER THAN ASSUMED AWAY. A command token is
       also a young token. So the binary read is reported beside a THREE-WAY read
       (base / L2-macro / L3-macro): at late checkpoints L2 slots are ~98 cycles old and L3
       slots ~47, so a decoder reading age separates L2 from L3 while a decoder reading
       command-ness does not.

6. THE OBSERVER TWIN, per paper 2 criterion 2 — mandatory, I/O-only, matched capacity.
   Every side is handed a COLUMN over the same held-out probe states, put through the same
   fixed random projection, the same classifier class, the same leave-one-slot-out folds.
   Matching the SHAPE is what makes this twin fair where audiation's low observer rungs were
   architecture-limited (they had to rebuild the frozen encoder through a probe stack and sat
   at ~0): here neither side encodes anything.

     self   R      the residual column                  (the un-theorised part)
     ctrl   FM     the self-theory's column              (paper 2 section 6's PRED)
     obs    LOGIT  the raw logit column                  (paper 2 section 6's AJ; a strictly
                                                          stronger observer than O_io)
     obs    IO     the softmax column                    (paper 2's O_io: M's full output
                                                          distribution at every position)
     obs    BEH    the top-k selection column            (strictly public behaviour: the moves
                                                          the beam would expand)
     neg    NORM   the scalar ||R_j|| alone              (the scalar-norm negative)
     neg    SHUF   R with the probe axis shuffled per slot
     null   PERM   slot labels permuted, refit

   Statistic: balanced accuracy (the classes are 32:16 and 32:24, so raw accuracy is not
   readable), plus paper 2 section 6's headroom fraction
   frac = (best I/O obs - chance) / (ceiling - chance).

7. GUARDS (component_control / audiation discipline; a saturated instrument counterfeits
   everything downstream, so these disarm it):
     - `ens_cos`: 3 independent-seed FMs per (checkpoint, capacity), mean pairwise cosine of
       their residuals. Paper 2's reference band: 0.78-0.91 real, 0.65 junk.
     - hierarchy-eta2: the practice port of paper 2's structure diagnostic. The residual's
       norm and its class label conditioned on the ROOT TARGET (the grammar's top latent — the
       deep-hierarchy conditioning variable this substrate has) against the junk conditioners
       CYCLE / STEP / ERA. A residual that keys only to the calendar is junk.
     - scalar-norm negatives, a shuffled null, a slot-label permutation null.
     - capacity invariance across the whole sweep.
     - `_gradcheck()` against central differences as a hard assert (there is no autograd here).
     - a head-recompute gate: the numpy pi must reproduce the run's own logged `t_pi` to the
       fp16 storage scale audiation measured (1.2e-3 on a logit scale of 3.4).

8. NO PRE-REGISTERED INTERPRETATIONS. Per repo norms this file computes; it does not say what
   an outcome would mean.


================================================================================
MODES
================================================================================

  --smoke     wiring pass, tiny probe, 3 checkpoints. Minutes. Numbers meaningless.
  (default)   the full pass: primary capacity on every checkpoint, the capacity x seed sweep
              on the event-anchored checkpoint subset, both axes, all guards.
  --quickfit  same shape, fewer iterations / smaller probe — for iterating on the reduction.

Outputs: figures/<tag>/{reduction.txt, continuo.json, *.png}
"""

import argparse
import json
import os
import sys
import time

# MUST precede the numpy import. The image ships OpenBLAS built with MAX_THREADS=64 and
# NO_AFFINITY; on a 4-core box its spin-wait overhead dominates the small matmuls this file is
# made of — measured 8 ms for one (2148,192)@(192,6) product at the default, 0.2 ms at 2
# threads, a 40x difference that decides whether the full pass takes 40 minutes or a day.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from rhm.practice.audiation.analyze_audiation import head_forward  # noqa: E402  (read-only)

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")

# The known consolidation events of `au_s0/anchor`, read off the donor's own record and NOT
# chosen after seeing any residual. (results.json events + audiation.json per-cycle context.)
#   commit L2 at cycle 18; the practice beam's action set grows at 19 (the per-cycle context is
#   recorded in block (a), before the commit lands later in the same cycle).
#   commit L3 at cycle 69; action set grows at 70.
#   era advances at 49, 89, 101, 110 (log['era'] changes).
EVENTS_AU_S0 = {
    "commit_L2": 18, "ms_grow_L2": 19,
    "commit_L3": 69, "ms_grow_L3": 70,
    "era2": 49, "era3": 89, "era4": 101, "era5": 110,
}


def derive_events(res, audi):
    """The consolidation events read off the RUN'S OWN record, never hardcoded.

    `au_s0/anchor` and `au_s1/perdatum` do not share a clock — the per-datum arm commits L2 one
    cycle later and L3 seven cycles EARLIER (`audiation` finding 8(ii)) — so a hardcoded event
    table would silently mis-align the second arm. The au_s0 values are kept only as a
    regression check on this function."""
    ev = {}
    for e in res.get("events", []):
        if e.get("kind") == "commit":
            lv, cy = int(e["level"]), int(e["cycle"])
            if f"commit_L{lv}" not in ev:                 # first commit of that level
                ev[f"commit_L{lv}"] = cy
    # the practice beam's action set grows the cycle AFTER the commit (the per-cycle context is
    # recorded in block (a), before the commit lands later in the same cycle)
    for k in [k for k in ev if k.startswith("commit_L")]:
        grow = ev[k] + 1
        cy = audi["cycles"]
        if grow - 1 < len(cy):
            ev["ms_grow_L" + k.split("_L")[1]] = grow
    era = res["log"]["era"]
    for i in range(1, len(era)):
        if era[i] != era[i - 1]:
            ev[f"era{era[i]}"] = i + 1
    return ev


def derive_sweep(events, last_snap, n=10):
    """Event-anchored checkpoints for the capacity x seed sweep: every commit and the cycle
    after it, plus evenly spaced fillers, plus the terminal snapshot."""
    want = set()
    for k, v in events.items():
        if k.startswith("commit_L") or k.startswith("ms_grow_L"):
            want.add(v)
    want.add(6)
    want.add(last_snap - 1)
    fill = np.linspace(6, last_snap - 1, max(2, n - len(want)) + 2)[1:-1]
    for f in fill:
        want.add(int(round(f)))
    return sorted(x for x in want if 5 <= x <= last_snap)


# minimum slot AGE floors for follow-up (a): a macro slot's age at checkpoint c is
# c - (the cycle its slot first went live). Base slots are live from cycle 1 throughout.
AGE_FLOORS = [0, 3, 5, 10, 20, 30, 50]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_donor(tag, arm):
    """audiation.json (per-cycle action-set context) + results.json (the run's own log)."""
    root = os.path.join(DATA, tag, arm)
    audi = json.load(open(os.path.join(HERE, "..", "audiation", "figures", tag, arm,
                                       "audiation.json")))
    res = json.load(open(os.path.join(HERE, "..", "audiation", "figures", tag, arm,
                                      "results.json")))
    return root, audi, res


def load_heads(root, c):
    p = os.path.join(root, "snapshots", f"heads_c{c:04d}.npz")
    if not os.path.isfile(p):
        return None
    return dict(np.load(p))


def load_shards(root, want_cycles=None):
    """The fetched decision shards, concatenated on the TIPS table only (the candidate table is
    only needed for the behaviour observer, which is derived from pi instead — see design call
    6: the BEH column is the selection the CURRENT head would make, which is what an outside
    observer of checkpoint c sees, and it is computable from pi alone)."""
    d = os.path.join(root, "decisions")
    keep = ["t_x", "t_z", "t_pi", "t_cycle", "t_inst", "t_step", "t_root", "t_succ"]
    out = {k: [] for k in keep}
    files = sorted(f for f in os.listdir(d) if f.startswith("dec_"))
    for f in files:
        z = np.load(os.path.join(d, f))
        m = np.ones(z["t_cycle"].shape[0], dtype=bool)
        if want_cycles is not None:
            m = np.isin(z["t_cycle"], np.asarray(list(want_cycles)))
        if not m.any():
            continue
        for k in keep:
            out[k].append(z[k][m])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}, files


# --------------------------------------------------------------------------- #
# the probe set
# --------------------------------------------------------------------------- #

def build_probe(tips, n_probe, seed, min_cycle=5, n_inst=8):
    """A FIXED probe set of practice-beam tip states, stratified over the shards' cycles.

    Two things this has to get right at once. (a) Stratify by cycle, so no era dominates the
    state manifold the FM is fitted on. (b) Draw MANY INSTANCES per cycle and only then subsample
    rows inside each, because the split that follows is by (cycle, instance) and the number of
    independent groups is what the FM's held-out score actually rests on. An earlier version
    filled each cycle's row quota from whole instances, which — a beam instance contributes
    (budget+1) x width = 144 rows — silently collapsed to ONE instance per cycle, so the
    instance-level split degenerated into a cycle-level one.

    `n_inst` instances per cycle, `n_probe / (n_cycles * n_inst)` rows sampled uniformly within
    each (over the whole (step, tip) grid), instance grouping preserved."""
    rng = np.random.default_rng(seed)
    ok = tips["t_cycle"] >= min_cycle
    cyc = np.unique(tips["t_cycle"][ok])
    per_inst = max(4, int(round(n_probe / (len(cyc) * n_inst))))
    idx = []
    for c in cyc:
        rows = np.flatnonzero(ok & (tips["t_cycle"] == c))
        insts = np.unique(tips["t_inst"][rows])
        rng.shuffle(insts)
        for i in insts[:n_inst]:
            r = rows[tips["t_inst"][rows] == i]
            if len(r) > per_inst:
                r = rng.choice(r, per_inst, replace=False)
            idx.append(r)
    idx = np.sort(np.concatenate(idx))
    return {
        "z": np.asarray(tips["t_z"][idx], dtype=np.float32),
        "root": np.asarray(tips["t_root"][idx], dtype=np.int64),
        "cycle": np.asarray(tips["t_cycle"][idx], dtype=np.int32),
        "inst": np.asarray(tips["t_inst"][idx], dtype=np.int32),
        "step": np.asarray(tips["t_step"][idx], dtype=np.int32),
        "row": idx,
    }


def group_split(probe, frac_train, seed):
    """Split by (cycle, instance) — never by row."""
    rng = np.random.default_rng(seed)
    gid = probe["cycle"].astype(np.int64) * 10000 + probe["inst"].astype(np.int64)
    groups = np.unique(gid)
    rng.shuffle(groups)
    n_tr = int(round(frac_train * len(groups)))
    tr = set(groups[:n_tr].tolist())
    m = np.array([g in tr for g in gid])
    return m, ~m


# --------------------------------------------------------------------------- #
# the FM: same shape of read as the head, in numpy, with AdamW
# --------------------------------------------------------------------------- #

def _gelu(x):
    # nn.GELU()'s default is the EXACT erf form. audiation records that the tanh form is a
    # ~1e-3 error masquerading as a mismatch; the FM is our own module so only consistency
    # matters, but matching the head's nonlinearity keeps the class honest.
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _dgelu(x):
    from scipy.special import erf
    c = 1.0 / np.sqrt(2.0 * np.pi)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0))) + x * c * np.exp(-0.5 * x * x)


class FM:
    """The FM is architecturally a `ProposalHead` with hidden width H in place of 4*state_dim:

        root_embedding(v, d) ; cat([z, emb]) -> LayerNorm(2d) -> Linear(2d, H) -> GELU
                                             -> Linear(H, d_out)

    Same shape of read as the head it predicts (design call 3), so at H = 4*state_dim = 384 the
    FM *is* the head's architecture and the sweep's top rung is exactly 100% of the predicted
    span's parameters — this substrate's saturation demonstration, on purpose.

    a_i is the head's RAW input (z, root index), not (z, root_embedding(root)): the embedding is
    a learned layer of the head, so handing it over would give the FM a piece of the span it is
    supposed to predict, and would also make the FM's input move across checkpoints — which
    would confound every cross-checkpoint comparison this node makes.

    No autograd; `_gradcheck` against central differences is a hard assert at the top of main."""

    def __init__(self, d_state, v, H, d_out, seed, dtype=np.float32):
        rng = np.random.default_rng(seed)
        dt = dtype
        d_in = 2 * d_state
        self.E = rng.normal(0, 1.0, (v, d_state)).astype(dt)
        self.g = np.ones(d_in, dtype=dt)
        self.b = np.zeros(d_in, dtype=dt)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, H)).astype(dt)
        self.b1 = np.zeros(H, dtype=dt)
        # zero-init output layer, exactly as `ProposalHead` does it: with a z-scored target the
        # FM then starts at the target mean rather than at a random large prediction, which is
        # what lets the small rungs converge at all inside the iteration budget.
        self.W2 = np.zeros((H, d_out), dtype=dt)
        self.b2 = np.zeros(d_out, dtype=dt)
        self.d_state, self.v, self.H, self.d_out = d_state, v, H, d_out
        self.dtype = dt

    def n_params(self):
        d_in = 2 * self.d_state
        return (self.v * self.d_state + 2 * d_in + d_in * self.H + self.H
                + self.H * self.d_out + self.d_out)

    def params(self):
        return [self.E, self.g, self.b, self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def onehot(root, v):
        oh = np.zeros((len(root), v), dtype=np.float32)
        oh[np.arange(len(root)), np.asarray(root, dtype=np.int64)] = 1.0
        return oh

    def forward(self, z, oh, cache=False):
        # the embedding lookup is written as a (n, v) @ (v, d) matmul rather than a gather:
        # `np.add.at` on the backward gather is ~50x slower than the equivalent matmul and was
        # 80% of this file's runtime before the change.
        emb = oh @ self.E
        x = np.concatenate([z, emb], axis=-1)
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + 1e-5)
        xh = (x - mu) * inv
        h0 = xh * self.g + self.b
        a = h0 @ self.W1 + self.b1
        h = _gelu(a)
        y = h @ self.W2 + self.b2
        if cache:
            self._c = (oh, x, inv, xh, h0, a, h)
        return y

    def backward(self, dy):
        oh, x, inv, xh, h0, a, h = self._c
        gW2 = h.T @ dy
        gb2 = dy.sum(0)
        dh = dy @ self.W2.T
        da = dh * _dgelu(a)
        gW1 = h0.T @ da
        gb1 = da.sum(0)
        dh0 = da @ self.W1.T
        gg = (dh0 * xh).sum(0)
        gb = dh0.sum(0)
        dxh = dh0 * self.g
        dx = inv * (dxh - dxh.mean(-1, keepdims=True)
                    - xh * (dxh * xh).mean(-1, keepdims=True))
        demb = dx[:, self.d_state:]
        gE = oh.T @ demb
        return [gE, gg, gb, gW1, gb1, gW2, gb2]

    def fit(self, z, oh, y, iters, lr, wd=1e-4):
        ps = self.params()
        m = [np.zeros_like(p) for p in ps]
        v = [np.zeros_like(p) for p in ps]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = z.shape[0]
        loss = np.nan
        for t in range(1, iters + 1):
            pred = self.forward(z, oh, cache=True)
            r = pred - y
            loss = float((r * r).mean())
            dy = (2.0 * r / (n * y.shape[1])).astype(self.dtype)
            gs = self.backward(dy)
            cur = lr * (0.5 * (1 + np.cos(np.pi * t / iters)))
            for i, (p, g) in enumerate(zip(ps, gs)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** t)
                vh = v[i] / (1 - b2 ** t)
                p -= cur * (mh / (np.sqrt(vh) + eps) + wd * p)
        return loss


def _gradcheck():
    """Hard assert: analytic gradients against central differences (float64)."""
    rng = np.random.default_rng(0)
    f = FM(4, 3, 5, 3, seed=1, dtype=np.float64)
    z = rng.normal(size=(11, 4))
    root = rng.integers(0, 3, size=11)
    oh = FM.onehot(root, 3).astype(np.float64)
    y = rng.normal(size=(11, 3))
    pred = f.forward(z, oh, cache=True)
    r = pred - y
    dy = 2.0 * r / (z.shape[0] * y.shape[1])
    gs = f.backward(dy)
    worst = 0.0
    for p, g in zip(f.params(), gs):
        flat = p.reshape(-1)
        gflat = g.reshape(-1)
        for k in rng.choice(len(flat), size=min(6, len(flat)), replace=False):
            o = float(flat[k])
            h = 1e-6
            flat[k] = o + h
            lp = ((f.forward(z, oh) - y) ** 2).mean()
            flat[k] = o - h
            lm = ((f.forward(z, oh) - y) ** 2).mean()
            flat[k] = o
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - gflat[k]) / max(1e-8, abs(num)))
    assert worst < 1e-4, f"gradcheck failed, worst rel err {worst:.3e}"
    return worst


# --------------------------------------------------------------------------- #
# per-checkpoint residual
# --------------------------------------------------------------------------- #

def slot_levels(audi):
    """slot id -> level, from the run's own `ms_slots`/`ms_level` (not from arithmetic on
    `slot_offsets`, so it stays right if the layout ever changes)."""
    lev = {}
    for c in audi["cycles"]:
        for s, l in zip(c["ms_slots"], c["ms_level"]):
            lev[int(s)] = int(l)
    return lev


def slot_live_cycle(audi):
    """slot id -> the first cycle at which it appears in `avail_slots`. Follow-up (a) needs the
    token's AGE, because a command token is also a young token and the pass-1 decode's whole
    advantage sat in the cycles just after a commit."""
    first = {}
    for c in audi["cycles"]:
        for s in c["avail_slots"]:
            first.setdefault(int(s), int(c["cycle"]))
    return first


def live_slots(audi, c):
    """The slots pi's softmax was taken under at cycle c. Snapshot `heads_c` is the head as it
    stood BEFORE cycle c ran, so cycle c's own mask is the right one; the terminal snapshot
    c_last+1 inherits the last cycle's."""
    cy = audi["cycles"]
    i = min(max(c - 1, 0), len(cy) - 1)
    return np.asarray(cy[i]["avail_slots"], dtype=np.int64), cy[i]


def slot_nodes(audi):
    """slot id -> (level, node). The slot vocabulary is one slot per (level, node) of the tree
    (`prop_net.slot_layout`): 32 level-1 nodes, 16 level-2, 8 level-3 at depth 6, s = 2."""
    nd = {}
    for c in audi["cycles"]:
        for s, l, n in zip(c["ms_slots"], c["ms_level"], c["ms_node"]):
            nd[int(s)] = (int(l), int(n))
    return nd


def cant_decompose(pi, live, nd, off1, s=2):
    """`native`'s can't-decompose readout, recomputed offline at the probe states.

    Per state: pick the argmax LIVE MACRO slot (level >= 2), find that (level, node) macro's own
    primitive spelling (`prop_net.decomposition_slots`: level l node n spells out to base slots
    off1 + n*s^(l-1) + [0, s^(l-1)) ), and read pi's softmax mass there against the mass on the
    macro itself. An expert cannot decompose its chunks; a head that still puts mass on the
    spelling has not chunked. Returns (macro_mass, own_spell_mass, own_macro_mass)."""
    ex = np.exp(pi - pi.max(1, keepdims=True))
    p = ex / ex.sum(1, keepdims=True)                    # softmax over LIVE slots only
    lv = np.asarray([nd[int(j)][0] for j in live])
    no = np.asarray([nd[int(j)][1] for j in live])
    mac = np.flatnonzero(lv >= 2)
    if mac.size == 0:
        return 0.0, np.nan, np.nan
    macro_mass = float(p[:, mac].sum(1).mean())
    am = mac[p[:, mac].argmax(1)]
    base_pos = {int(j): i for i, j in enumerate(live) if nd[int(j)][0] == 1}
    spell, own = np.zeros(len(p)), np.zeros(len(p))
    for i, a in enumerate(am):
        l, n = int(lv[a]), int(no[a])
        span = s ** (l - 1)
        cols = [base_pos[off1 + n * span + j] for j in range(span)
                if (off1 + n * span + j) in base_pos]
        spell[i] = p[i, cols].sum() if cols else np.nan
        own[i] = p[i, a]
    return macro_mass, float(np.nanmean(spell)), float(np.mean(own))


def pi_at(heads, probe):
    return head_forward(heads, "prop", probe["z"], probe["root"]).astype(np.float32)


def v_at(heads, probe):
    return head_forward(heads, "value", probe["z"], probe["root"]).astype(np.float32)


def _ridge_fm(z, root, Y, tr, v, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """H = 0: the LINEAR theory of the head, from (z, one-hot root).

    Why it earns a rung. `ens_cos` exists (paper 2 sec. 8) because a nonconvex FM's residual can
    be a property of the instrument rather than of M. A convex FM has a unique minimiser, so its
    residual is a property of the head by construction and `ens_cos` is 1.000 trivially — which
    makes it useless as a guard but makes it the one rung whose residual cannot be instrument
    noise. It is a strictly weaker theory (so it leaves a larger remainder), and it is reported
    as the floor of the theory ladder, not as a substitute for the swept MLP."""
    X = np.concatenate([z, np.eye(v, dtype=np.float32)[root],
                        np.ones((len(z), 1), dtype=np.float32)], axis=1)
    Xt = X[tr].astype(np.float64)
    Yt = Y[tr].astype(np.float64)
    n_in = X.shape[1]
    # inner split of the FM-train rows to pick lambda; no test row is ever touched
    n_h = int(0.8 * Xt.shape[0])
    A = Xt[:n_h].T @ Xt[:n_h]
    B = Xt[:n_h].T @ Yt[:n_h]
    best, bl = None, None
    for lam in lams:
        W = np.linalg.solve(A + lam * np.eye(n_in), B)
        mse = float(((Xt[n_h:] @ W - Yt[n_h:]) ** 2).mean())
        if best is None or mse < best:
            best, bl = mse, lam
    W = np.linalg.solve(Xt.T @ Xt + bl * np.eye(n_in), Xt.T @ Yt)
    return (X.astype(np.float64) @ W).astype(np.float32), n_in * Y.shape[1] + 0, bl


def fit_residual(target, probe, tr, te, H, seed, iters, lr, v=8):
    """Fit the FM on FM-train rows, return the held-out residual and diagnostics.

    Target scaling: per-output z-scoring on FM-train rows, un-scaled before the residual is
    formed (audiation's deviation 4 — raw MSE would weight outputs by raw variance, and here
    the spread across slots is extreme because a freshly-live slot's logit column is still
    near-zero).

    H = 0 selects the deterministic ridge rung (see `_ridge_fm`)."""
    d_out = target.shape[1]
    mu = target[tr].mean(0)
    sd = target[tr].std(0)
    degen = sd < 1e-6
    sd_safe = np.where(degen, 1.0, sd)
    Y = ((target - mu) / sd_safe).astype(np.float32)
    if H == 0:
        pred_s, npar, lam = _ridge_fm(probe["z"], probe["root"], Y, tr, v)
        tr_loss = float(((pred_s[tr] - Y[tr]) ** 2).mean())
        extra = {"ridge_lam": lam}
    else:
        oh = probe.get("_oh")
        if oh is None:
            oh = FM.onehot(probe["root"], v)
            probe["_oh"] = oh
        fm = FM(probe["z"].shape[1], v, H, d_out, seed)
        tr_loss = fm.fit(probe["z"][tr], oh[tr], Y[tr], iters, lr)
        pred_s = fm.forward(probe["z"], oh)
        npar = fm.n_params()
        extra = {}
    te_loss = float(((pred_s[te] - Y[te]) ** 2).mean())
    pred = pred_s * sd_safe + mu
    R = target - pred
    o = {
        "R": R[te], "pred": pred[te], "target": target[te],
        "degen": degen, "n_params": npar,
        "tr_loss": tr_loss, "te_loss": te_loss,
        "r_over_t": float(np.linalg.norm(R[te]) / max(1e-12, np.linalg.norm(target[te] - mu))),
    }
    o.update(extra)
    return o


# --------------------------------------------------------------------------- #
# axis 1 — residual-class occupancy
# --------------------------------------------------------------------------- #

def spherical_kmeans(X, K, seed, iters=60, restarts=4):
    """X: (n, d) unit rows. Returns (centroids (K,d) unit, labels, inertia)."""
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + 1000 * r)
        C = X[rng.choice(X.shape[0], K, replace=False)].copy()
        lab = None
        for _ in range(iters):
            sim = X @ C.T
            lab = sim.argmax(1)
            for k in range(K):
                m = lab == k
                if m.sum() == 0:
                    C[k] = X[rng.integers(0, X.shape[0])]
                else:
                    c = X[m].sum(0)
                    n = np.linalg.norm(c)
                    C[k] = c / n if n > 0 else C[k]
        inertia = float((X @ C.T).max(1).mean())
        if best is None or inertia > best[2]:
            best = (C, lab, inertia)
    return best


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return M / n


def embed56(R, live, n_slots=56):
    """Lift a (n, n_live) residual into full slot space with zeros off the live set, so a
    codebook fitted before a commit and one fitted after live in the same space."""
    out = np.zeros((R.shape[0], n_slots), dtype=np.float32)
    out[:, live] = R
    return out


def eta2(values, groups):
    """Between-group variance fraction."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    gt = values.mean()
    tot = ((values - gt) ** 2).sum()
    if tot <= 0:
        return 0.0
    bet = 0.0
    for g in np.unique(groups):
        m = groups == g
        bet += m.sum() * (values[m].mean() - gt) ** 2
    return float(bet / tot)


def eta2_vs_random(values, groups, seed, n=32):
    """eta^2 beside a matched-random grouping of the same sizes.

    eta^2 is inflated by group count and by small groups, so the number is unreadable without
    its own null: this returns (observed, mean-of-random, observed - mean-of-random)."""
    obs = eta2(values, groups)
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    nulls = [eta2(values, g[rng.permutation(len(g))]) for _ in range(n)]
    return obs, float(np.mean(nulls)), obs - float(np.mean(nulls))


# --------------------------------------------------------------------------- #
# axis 2 — the instruction/data decode, self vs the observer twin
# --------------------------------------------------------------------------- #

def _proj(n_te, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0 / np.sqrt(d), (n_te, d)).astype(np.float32)


MIN_LEV = 0.05   # smallest admissible 1 - h_ii; see _loo_ridge_bal


def _loo_ridge_bal(X, y, lams):
    """Leave-one-out ridge over ROWS (= slots), closed form via the hat matrix, at every lambda.

    THE DEGENERACY THIS GUARDS. There are only 48-56 slots and the projected feature dimension
    can exceed that, so at small lambda the fit interpolates: h_ii -> 1, the LOO correction
    r_i / (1 - h_ii) becomes 0/0, and every cell — self, observer, shuffled null alike — scores
    a spurious 1.000. Any lambda whose minimum leverage margin `1 - h_ii` falls below MIN_LEV is
    therefore marked INVALID (nan) and excluded from the median rather than clipped. The count
    of valid lambdas is reported so a cell that had none is visible.

    Returns (bal_acc per lambda with nan for invalid, valid mask, (U, S))."""
    n, d = X.shape
    Xc = X - X.mean(0, keepdims=True)
    ym = y.mean()
    yc = y - ym
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    accs, valid = [], []
    pos, neg = y > 0, y < 0
    for lam in lams:
        f = S ** 2 / (S ** 2 + lam)
        Huu = (U * f) @ U.T
        denom = 1.0 - (np.diag(Huu) + 1.0 / n)
        if denom.min() < MIN_LEV:
            accs.append(np.nan)
            valid.append(False)
            continue
        loo = y - (y - (Huu @ yc + ym)) / denom
        pred = np.sign(loo)
        tpr = float((pred[pos] > 0).mean()) if pos.any() else 0.0
        tnr = float((pred[neg] < 0).mean()) if neg.any() else 0.0
        accs.append(0.5 * (tpr + tnr))
        valid.append(True)
    return np.asarray(accs), np.asarray(valid), (U, S)


def _loo_bal_perm(U, S, y, lams, n_perm, seed):
    """The same statistic under permuted slot labels — the hat matrix is reused, so this is
    essentially free."""
    rng = np.random.default_rng(seed)
    n = U.shape[0]
    out = np.zeros((n_perm, len(lams)))
    # the hat matrix depends on lambda only, never on y — hoisted out of the permutation loop
    Hs = []
    for lam in lams:
        f = S ** 2 / (S ** 2 + lam)
        Huu = (U * f) @ U.T
        denom = 1.0 - (np.diag(Huu) + 1.0 / n)
        Hs.append(None if denom.min() < MIN_LEV else (Huu, denom))
    for p in range(n_perm):
        yp = y[rng.permutation(n)]
        ycm = yp.mean()
        pos, neg = yp > 0, yp < 0
        for li, hd in enumerate(Hs):
            if hd is None:
                out[p, li] = np.nan
                continue
            Huu, denom = hd
            loo = yp - (yp - (Huu @ (yp - ycm) + ycm)) / denom
            pred = np.sign(loo)
            tpr = float((pred[pos] > 0).mean()) if pos.any() else 0.0
            tnr = float((pred[neg] < 0).mean()) if neg.any() else 0.0
            out[p, li] = 0.5 * (tpr + tnr)
    return out


def build_columns(R, pred, target, live, lev, k_eff, seed):
    """One (n_slots_kept, n_te) matrix per cell, all over the SAME held-out probe states.

    self  R      residual column
    ctrl  FM     the self-theory's column
    obs   LOGIT  raw logit column         (paper 2 sec. 6's AJ; the strongest observer rung)
    obs   IO     softmax column           (paper 2's O_io: M's full output distribution)
    obs   BEH    top-k selection column   (strictly public behaviour)
    neg   SHUF   R, probe axis shuffled independently per slot
    neg   NORM   ||R_j|| alone (1-dim)
    """
    rng = np.random.default_rng(seed)
    ex = np.exp(target - target.max(1, keepdims=True))
    io = ex / ex.sum(1, keepdims=True)
    kk = int(min(max(k_eff, 1), target.shape[1]))
    order = np.argsort(-target, axis=1, kind="stable")
    beh = np.zeros_like(target)
    np.put_along_axis(beh, order[:, :kk], 1.0, axis=1)
    shuf = np.empty_like(R)
    for j in range(R.shape[1]):
        shuf[:, j] = R[rng.permutation(R.shape[0]), j]
    cols = {"R": R.T, "FM": pred.T, "LOGIT": target.T, "IO": io.T, "BEH": beh.T,
            "SHUF": shuf.T}
    norms = np.linalg.norm(R, axis=0)
    return cols, norms


def decode_table(cols, norms, keep, labels, d_projs, lams, n_perm, seed):
    """Every cell through the same projection, the same LOO folds, the same lambda grid.
    Headline = the MEDIAN balanced accuracy over the lambda grid — no selection on the test
    slots anywhere, and capacity invariance is readable off the grid's spread."""
    y = np.where(labels > 0, 1.0, -1.0)
    out = {}
    for d in d_projs:
        P = None
        for name, C in cols.items():
            M = C[keep]
            M = unit_rows(M.astype(np.float32))
            if P is None or P.shape != (M.shape[1], d):
                P = _proj(M.shape[1], d, seed)
            X = M @ P
            accs, valid, (U, S) = _loo_ridge_bal(X, y, lams)
            rec = {"per_lam": accs.tolist(), "n_valid_lam": int(valid.sum()),
                   "med": float(np.nanmedian(accs)) if valid.any() else np.nan,
                   "max": float(np.nanmax(accs)) if valid.any() else np.nan,
                   "min": float(np.nanmin(accs)) if valid.any() else np.nan}
            if name == "R" and n_perm and valid.any():
                null = _loo_bal_perm(U, S, y, lams, n_perm, seed + 7)
                nm = np.nanmedian(null, axis=1)
                rec["perm_mean"] = float(np.nanmean(nm))
                rec["perm_p95"] = float(np.nanquantile(nm, 0.95))
                rec["perm_p"] = float(np.nanmean(nm >= rec["med"]))
            out[f"{name}@d{d}"] = rec
        # the scalar-norm negative: one feature, no projection, same folds
        X = np.log(np.clip(norms[keep], 1e-12, None))[:, None].astype(np.float32)
        accs, valid, _ = _loo_ridge_bal(X, y, lams)
        out[f"NORM@d{d}"] = {"per_lam": accs.tolist(), "n_valid_lam": int(valid.sum()),
                             "med": float(np.nanmedian(accs)) if valid.any() else np.nan,
                             "max": float(np.nanmax(accs)) if valid.any() else np.nan,
                             "min": float(np.nanmin(accs)) if valid.any() else np.nan}
    return out


# --------------------------------------------------------------------------- #
# the outer loop's gauge, and the calendar control
# --------------------------------------------------------------------------- #

def at_support_series(res, level):
    """`panel[c]['at_support'][level]` — the free one-level-up gauge A1/A2 drive on."""
    out = []
    for p in res["log"]["panel"]:
        if p is None or "at_support" not in p:
            out.append(np.nan)
        else:
            out.append(float(p["at_support"].get(str(level), np.nan)))
    return np.asarray(out, dtype=float)


def partial_spearman(a, b, cycle, era):
    """Spearman after residualising BOTH series on cycle and era.

    audiation finding 1 measured eta^2(update ~ cycle) = 0.438 on this very run: the calendar
    confound is real here, and a raw correlation between two monotone-in-time series says
    nothing. Both the raw and the de-drifted number are reported."""
    from scipy.stats import rankdata, spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    a, b, cycle, era = a[m], b[m], cycle[m], era[m]
    raw = spearmanr(a, b)
    ra, rb = rankdata(a), rankdata(b)
    D = [np.ones_like(cycle, dtype=float), cycle.astype(float)]
    for e in np.unique(era)[1:]:
        D.append((era == e).astype(float))
    D = np.stack(D, axis=1)
    Q, _ = np.linalg.qr(D)
    ra_r = ra - Q @ (Q.T @ ra)
    rb_r = rb - Q @ (Q.T @ rb)
    den = np.linalg.norm(ra_r) * np.linalg.norm(rb_r)
    part = float(ra_r @ rb_r / den) if den > 0 else np.nan
    return float(raw.statistic), float(raw.pvalue), part, int(m.sum())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

# event-anchored: pre-L2 / the L2 commit / just after / mid-era-1 / the era-2 advance /
# pre-L3 / the L3 commit / just after / late / the last checkpoint.
SWEEP_CKPTS = [6, 18, 19, 25, 45, 49, 69, 70, 80, 116]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="au_s0")
    ap.add_argument("--arm", default="anchor")
    ap.add_argument("--n-probe", type=int, default=6144)
    ap.add_argument("--frac-train", type=float, default=0.70)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--h-primary", type=int, default=16)
    ap.add_argument("--h-sweep", default="0,6,16,42,128,384")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--kmeans-k", type=int, default=8)
    ap.add_argument("--d-projs", default="16,64,256")
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    # follow-up (c): perturb the FM's init ALONE, leaving the probe draw, the split,
    # the projection and the codebook at --seed, so instrument variance is separated
    # from draw variance instead of confounded with it.
    ap.add_argument("--fm-seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.smoke:
        a.n_probe, a.iters, a.n_seeds, a.n_perm = 512, 60, 2, 40
        a.h_sweep, a.d_projs = "0,6,384", "16,64"

    t0 = time.time()
    outdir = a.out or os.path.join(FIG, a.tag)
    os.makedirs(outdir, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out(f"continuo — E2: consolidation state off the residual   tag={a.tag} arm={a.arm}")
    out("=" * 78)
    out(f"argv: {' '.join(sys.argv[1:])}")

    # ---------------- gates ----------------
    worst = _gradcheck()
    out(f"\n(0) GATES\n  gradcheck (central differences, float64): PASS, worst rel err {worst:.2e}")

    root, audi, res = load_donor(a.tag, a.arm)
    lev = slot_levels(audi)
    nd = slot_nodes(audi)
    slot_born = slot_live_cycle(audi)
    off1 = int(audi["slot_offsets"]["1"])
    EVENTS = derive_events(res, audi)
    n_slots = int(audi["n_slots"])
    n_cyc = int(audi["n_cycles_logged"])
    last_snap = int(audi["final_snapshot_cycle"])
    out(f"  donor: {n_cyc} cycles, {last_snap} head snapshots, n_slots={n_slots}, "
        f"span_mode={res['span_mode']}, prop_k={res['prop_k']}")
    out("  events derived from the run's own record: "
        + ", ".join(f"{k}=c{v}" for k, v in sorted(EVENTS.items(), key=lambda kv: kv[1])))
    if a.tag == "au_s0" and a.arm == "anchor":
        assert EVENTS == EVENTS_AU_S0, f"event derivation regressed: {EVENTS}"
        out("    (regression check against the hardcoded au_s0/anchor table: PASS)")
    # The replay gate belongs to the arm that HAS a replay reference. `au_s1/perdatum` is E1b's
    # TREATMENT arm — it is meant to diverge, and its gate is the in-tag `anchor`'s inverse gate
    # (0.000e+00 against cd_s0/anchor and au_s0/anchor), reported here from that arm's own file
    # rather than fabricated for this one.
    rg = audi.get("replay_gate") or {}
    if rg.get("max_abs_delta") is not None:
        out(f"  replay gate (this arm's own): max_abs_delta={rg['max_abs_delta']:.3e} over "
            f"{rg['n_cycles_equal']}/{rg['n_cycles']} cycles")
    else:
        sib = os.path.join(HERE, "..", "audiation", "figures", a.tag, "anchor", "audiation.json")
        srg = (json.load(open(sib)).get("replay_gate") or {}) if os.path.isfile(sib) else {}
        d_ = srg.get("max_abs_delta")
        out(f"  replay gate: this arm is a TREATMENT arm and has no replay reference; the tag's"
            f" in-tag `anchor` inverse gate reads "
            + (f"{d_:.3e} over {srg.get('n_cycles_equal')}/{srg.get('n_cycles')} cycles"
               if d_ is not None else "unavailable"))

    tips, files = load_shards(root)
    out(f"  shards fetched: {len(files)} -> {tips['t_cycle'].shape[0]} tip rows, cycles "
        f"{tips['t_cycle'].min()}..{tips['t_cycle'].max()}")

    probe = build_probe(tips, a.n_probe, a.seed)
    tr, te = group_split(probe, a.frac_train, a.seed)
    out(f"  probe: {len(probe['z'])} fixed practice-beam states over "
        f"{len(np.unique(probe['cycle']))} cycles, "
        f"{tr.sum()} FM-train / {te.sum()} held-out rows, split by (cycle, instance)")

    # head-recompute gate: the numpy pi must reproduce the run's own logged t_pi
    ucy = np.unique(probe["cycle"])
    gate_c = int(ucy[len(ucy) // 2])
    heads = load_heads(root, gate_c)
    rows = np.flatnonzero(tips["t_cycle"] == gate_c)[:4096]
    pl = head_forward(heads, "prop", np.asarray(tips["t_z"][rows], dtype=np.float32),
                      np.asarray(tips["t_root"][rows], dtype=np.int64))
    logged = np.asarray(tips["t_pi"][rows], dtype=np.float32)
    m = np.isfinite(logged)
    dev = float(np.abs(pl[m] - logged[m]).max())
    scale = float(np.abs(logged[m]).max())
    out(f"  head-recompute gate at c{gate_c}: max|numpy pi - logged t_pi| = {dev:.3e} "
        f"on a logit scale of {scale:.2f}  "
        f"({'PASS' if dev < 5e-3 else 'FAIL'}; audiation measured 1.2e-3 at fp16-z storage)")
    assert dev < 5e-3, "head recompute gate failed"

    # ---------------- the primary pass ----------------
    d_projs = [int(x) for x in a.d_projs.split(",")]
    # spans interpolation to heavy shrinkage; the projected features are built to
    # have unit-ish row norm, so the informative range sits around 0.1-10
    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    ckpts = list(range(5, last_snap + 1))
    if a.smoke:
        ckpts = [19, 45, 70, 116]

    fm_seed = a.seed if a.fm_seed is None else a.fm_seed
    out(f"\n(1) PRIMARY PASS — FM hidden H={a.h_primary}, draw/split/projection seed {a.seed}, "
        f"FM init seed {fm_seed}, {len(ckpts)} checkpoints")
    per_ck = {}
    R_store = {}
    for c in ckpts:
        h = load_heads(root, c)
        if h is None:
            continue
        live, ctx = live_slots(audi, c)
        target = pi_at(h, probe)[:, live]
        f = fit_residual(target, probe, tr, te, a.h_primary, fm_seed, a.iters, a.lr)
        levl = np.asarray([lev[int(s)] for s in live])
        keep = ~f["degen"]
        macro = levl >= 2
        R = f["R"]
        # variance share, centred, for all three objects alike, so "which slots does this
        # object vary over" is the same question of R, of the theory component and of the logit
        Rc = R - R.mean(0)
        e_tot = (Rc ** 2).sum(1)
        e_mac = (Rc[:, macro] ** 2).sum(1) if macro.any() else np.zeros_like(e_tot)
        cmass = float(np.mean(e_mac / np.clip(e_tot, 1e-12, None)))
        T = f["target"]
        t_tot = ((T - T.mean(0)) ** 2).sum(1)
        t_mac = ((T[:, macro] - T[:, macro].mean(0)) ** 2).sum(1) if macro.any() \
            else np.zeros_like(t_tot)
        cmass_logit = float(np.mean(t_mac / np.clip(t_tot, 1e-12, None)))
        P = f["pred"]
        p_tot = ((P - P.mean(0)) ** 2).sum(1)
        p_mac = ((P[:, macro] - P[:, macro].mean(0)) ** 2).sum(1) if macro.any() \
            else np.zeros_like(p_tot)
        cmass_fm = float(np.mean(p_mac / np.clip(p_tot, 1e-12, None)))
        share = float(macro.sum()) / float(len(live))
        mm, sp, om = cant_decompose(f["target"], live, nd, off1)
        per_ck[c] = {
            "macro_mass": mm, "own_spell": sp, "own_macro": om,
            "spell_ratio": (sp / (sp + om)) if np.isfinite(sp) and (sp + om) > 0 else np.nan,
            "n_live": int(len(live)), "n_macro": int(macro.sum()),
            "n_degen": int(f["degen"].sum()), "n_keep": int(keep.sum()),
            "macro_share": share, "cmass_R": cmass, "cmass_LOGIT": cmass_logit,
            "cmass_FM": cmass_fm, "cmass_excess": cmass - share,
            "tr_loss": f["tr_loss"], "te_loss": f["te_loss"], "r_over_t": f["r_over_t"],
            "era": int(ctx["era"]), "n_params_fm": int(f["n_params"]),
        }
        R_store[c] = (R, live, levl, keep, f["pred"], f["target"], int(ctx["prop_k"]))
        if c % 20 == 0 or c == ckpts[-1]:
            out(f"  c{c:3d} live {len(live):2d} degen {f['degen'].sum():2d} "
                f"te_mse {f['te_loss']:.4f} |r|/|t| {f['r_over_t']:.3f} "
                f"cmass {cmass:.3f} (share {share:.3f})")
    out(f"  [{time.time() - t0:.0f}s]")

    # ---------------- axis 1: residual-class occupancy ----------------
    out("\n(2) AXIS 1 — RESIDUAL-CLASS OCCUPANCY")
    rng = np.random.default_rng(a.seed + 11)
    pool = []
    n_pool_per = max(8, int(6000 / max(1, len(R_store))))
    for c, (R, live, levl, keep, _, _, _) in R_store.items():
        F = embed56(R, live, n_slots)
        sel = rng.choice(F.shape[0], min(n_pool_per, F.shape[0]), replace=False)
        pool.append(F[sel])
    pool = unit_rows(np.concatenate(pool, 0))
    C, _, inertia = spherical_kmeans(pool, a.kmeans_k, a.seed)
    out(f"  codebook: spherical k-means K={a.kmeans_k} on {pool.shape[0]} pooled residual "
        f"directions (era-balanced, all checkpoints), mean cos-to-centroid {inertia:.3f}")
    is_macro56 = np.asarray([lev.get(j, 1) >= 2 for j in range(n_slots)])
    cls_cmd = (C[:, is_macro56] ** 2).sum(1)
    out("  class command-mass (fraction of the centroid's energy on macro slots):")
    out("    " + "  ".join(f"k{k}={cls_cmd[k]:.3f}" for k in range(a.kmeans_k)))

    occ, occ_c = {}, []
    for c in sorted(R_store):
        R, live, levl, keep, _, _, _ = R_store[c]
        F = unit_rows(embed56(R, live, n_slots))
        labk = (F @ C.T).argmax(1)
        h = np.bincount(labk, minlength=a.kmeans_k).astype(float)
        occ[c] = h / h.sum()
        occ_c.append(c)
    occ_c = np.asarray(occ_c)
    O = np.stack([occ[c] for c in occ_c])
    drift = np.concatenate([[np.nan], 0.5 * np.abs(np.diff(O, axis=0)).sum(1)])
    occ_cmd = O @ cls_cmd

    order = np.argsort(-np.nan_to_num(drift, nan=-1))
    rank_of = {int(occ_c[i]): int(r) + 1 for r, i in enumerate(order)}
    out(f"\n  occupancy drift TV(o_c, o_c-1): median {np.nanmedian(drift):.4f}, "
        f"p90 {np.nanquantile(drift, 0.90):.4f}, max {np.nanmax(drift):.4f} "
        f"over {np.isfinite(drift).sum()} checkpoints")
    out("  where the named consolidation events sit in that distribution (rank 1 = largest):")
    ev_rows = []
    for name, cyc in EVENTS.items():
        if cyc in rank_of:
            i = int(np.flatnonzero(occ_c == cyc)[0])
            pct = 100.0 * (1.0 - (rank_of[cyc] - 1) / np.isfinite(drift).sum())
            ev_rows.append((name, cyc, float(drift[i]), rank_of[cyc], pct))
            out(f"    {name:12s} c{cyc:3d}  drift {drift[i]:.4f}  rank {rank_of[cyc]:3d}"
                f"/{np.isfinite(drift).sum()}  (top {100 - pct:.0f}%)")

    out("\n  command-energy share of the residual, against the share of live slots that are"
        " commands:")
    out("    cycle  n_live  macro_share  cmass_R  excess_R  cmass_LOGIT  cmass_FM  occ_cmd")
    for c in sorted(per_ck):
        if c % 10 == 0 or c in EVENTS.values():
            p = per_ck[c]
            i = int(np.flatnonzero(occ_c == c)[0])
            out(f"    c{c:4d}  {p['n_live']:4d}   {p['macro_share']:.3f}      "
                f"{p['cmass_R']:.3f}   {p['cmass_excess']:+.3f}      {p['cmass_LOGIT']:.3f}"
                f"       {p['cmass_FM']:.3f}    {occ_cmd[i]:.3f}")

    # ---------------- axis 2: the instruction/data decode ----------------
    out("\n(3) AXIS 2 — THE INSTRUCTION/DATA DECODE, SELF vs THE OBSERVER TWIN")
    out("  balanced accuracy, leave-one-slot-out, median over the lambda grid; chance = 0.500")
    dec = {}
    dec3 = {}
    dec_age = {}
    perm_ck = set(derive_sweep(EVENTS, last_snap))
    for c in sorted(R_store):
        R, live, levl, keep, pred, target, pk = R_store[c]
        cols, norms = build_columns(R, pred, target, live, levl, pk, a.seed + c)
        lab = (levl >= 2).astype(int)
        k = keep & np.isfinite(norms) & (norms > 0)
        if k.sum() < 8 or len(np.unique(lab[k])) < 2:
            continue
        dec[c] = decode_table(cols, norms, k, lab[k], d_projs, lams,
                              a.n_perm if c in perm_ck else 0, a.seed + c)
        # the age disambiguator: L2 vs L3 among macro slots only
        km = k & (levl >= 2)
        lab3 = (levl[km] >= 3).astype(int)
        if km.sum() >= 8 and len(np.unique(lab3)) > 1:
            dec3[c] = decode_table(cols, norms, km, lab3, [64], lams, 0, a.seed + c)
        # follow-up (a): the same decode with a minimum TOKEN AGE floor. Pass 1 put nearly all
        # of the advantage in the cycles just after a commit, where a command token is also a
        # brand-new token; this asks what survives once the young tokens are removed.
        age = np.asarray([c - slot_born.get(int(sl), 1) for sl in live])
        dec_age[c] = {}
        for A in AGE_FLOORS:
            ka = k & (age >= A)
            if ka.sum() >= 8 and len(np.unique(lab[ka])) > 1:
                dec_age[c][A] = decode_table(cols, norms, ka, lab[ka], [64], lams, 0,
                                             a.seed + c)
                dec_age[c][A]["_n"] = int(ka.sum())
                dec_age[c][A]["_n_cmd"] = int((lab[ka] > 0).sum())

    def agg(cell, cks):
        v = [dec[c][cell]["med"] for c in cks if c in dec and cell in dec[c]]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)

    warm = [c for c in sorted(dec) if per_ck[c]["n_degen"] == 0]
    post_l3 = [c for c in warm if c >= EVENTS["ms_grow_L3"] + 10]
    out(f"\n  checkpoint sets: all-with-both-classes n={len(dec)}; "
        f"warm (no degenerate live slot) n={len(warm)}; warm & >=10 cycles past the L3 commit "
        f"n={len(post_l3)}")
    for setname, cks in [("all", sorted(dec)), ("warm", warm), ("warm_postL3", post_l3)]:
        if not cks:
            continue
        out(f"\n  --- {setname} (mean over {len(cks)} checkpoints) ---")
        out(f"    {'cell':8s}" + "".join(f"  d={d:<4d}" for d in d_projs))
        for cell in ["R", "FM", "LOGIT", "IO", "BEH", "SHUF", "NORM"]:
            row = f"    {cell:8s}"
            for d in d_projs:
                m, s, n = agg(f"{cell}@d{d}", cks)
                row += f"  {m:.3f} " if np.isfinite(m) else "   --   "
            out(row)
        pv = [dec[c]["R@d64"].get("perm_p") for c in cks if "perm_p" in dec[c].get("R@d64", {})]
        pm = [dec[c]["R@d64"].get("perm_mean") for c in cks
              if "perm_mean" in dec[c].get("R@d64", {})]
        if pv:
            out(f"    slot-label permutation null on R@d64: mean {np.mean(pm):.3f}, "
                f"mean p = {np.mean(pv):.3f} over {len(pv)} checkpoints")
    if dec3:
        v = [dec3[c]["R@d64"]["med"] for c in dec3]
        vl = [dec3[c]["LOGIT@d64"]["med"] for c in dec3]
        out(f"\n  AGE DISAMBIGUATOR (L2-macro vs L3-macro, macro slots only, d=64): "
            f"R {np.mean(v):.3f}  LOGIT {np.mean(vl):.3f}  over {len(dec3)} checkpoints")
        out("    (a decoder reading token AGE separates these; one reading command-ness does"
            " not)")

    out("\n  the decode across time (d=64), against the consolidation events:")
    out(f"    {'cycle':>6s} {'n_keep':>7s} {'self R':>8s} {'FM':>7s} {'LOGIT':>7s} {'IO':>7s} "
        f"{'BEH':>7s} {'SHUF':>7s} {'adv':>7s}")
    for c in sorted(dec):
        if c % 10 == 0 or c in EVENTS.values():
            g = dec[c]
            bi = max(g['LOGIT@d64']['med'], g['IO@d64']['med'], g['BEH@d64']['med'])
            out(f"    c{c:5d} {per_ck[c]['n_keep']:7d} {g['R@d64']['med']:8.3f} "
                f"{g['FM@d64']['med']:7.3f} {g['LOGIT@d64']['med']:7.3f} "
                f"{g['IO@d64']['med']:7.3f} {g['BEH@d64']['med']:7.3f} "
                f"{g['SHUF@d64']['med']:7.3f} {g['R@d64']['med'] - bi:+7.3f}")

    out("\n  FOLLOW-UP (a) — the decode under a minimum TOKEN-AGE floor (d=64).")
    out("  A command token is also a young token: pass 1's advantage sat in the cycles just")
    out("  after a commit. This removes slots younger than A cycles and asks what survives.")
    out(f"    {'A':>4s} {'n_ckpt':>7s} {'slots':>6s} {'cmd':>4s} {'self R':>8s} {'best I/O':>9s} "
        f"{'adv':>8s} {'NORM':>7s} {'SHUF':>7s}")
    age_rows = []
    for A in AGE_FLOORS:
        cks = [c for c in sorted(dec_age) if A in dec_age[c]]
        if not cks:
            continue
        def m_(n):
            return float(np.mean([dec_age[c][A][n]["med"] for c in cks]))
        bi = float(np.mean([max(dec_age[c][A]["LOGIT@d64"]["med"],
                                dec_age[c][A]["IO@d64"]["med"],
                                dec_age[c][A]["BEH@d64"]["med"]) for c in cks]))
        r = m_("R@d64")
        ns = float(np.mean([dec_age[c][A]["_n"] for c in cks]))
        nc = float(np.mean([dec_age[c][A]["_n_cmd"] for c in cks]))
        age_rows.append({"A": A, "n_ckpt": len(cks), "n_slots": ns, "n_cmd": nc,
                         "self": r, "best_io": bi, "adv": r - bi,
                         "norm": m_("NORM@d64"), "shuf": m_("SHUF@d64")})
        out(f"    {A:4d} {len(cks):7d} {ns:6.1f} {nc:4.1f} {r:8.3f} {bi:9.3f} "
            f"{r - bi:+8.3f} {m_('NORM@d64'):7.3f} {m_('SHUF@d64'):7.3f}")

    # paper 2 section 6's headroom fraction
    def frac_row(cks, d=64):
        selfv, _, _ = agg(f"R@d{d}", cks)
        ios = [agg(f"{o}@d{d}", cks)[0] for o in ("LOGIT", "IO", "BEH")]
        best_io = float(np.nanmax(ios))
        ceil = max(selfv, best_io)
        # frac is only defined when something clears chance; below chance it is not a
        # headroom fraction, it is noise, and is reported as nan rather than as a number.
        fr = (best_io - 0.5) / (ceil - 0.5) if ceil - 0.5 > 0.02 else np.nan
        return selfv, best_io, selfv - best_io, fr
    out("\n  paper 2 criterion 2 / section 6, at d=64:")
    out(f"    {'set':14s} {'self R':>8s} {'best I/O':>9s} {'advantage':>10s} {'frac':>7s}")
    for setname, cks in [("all", sorted(dec)), ("warm", warm), ("warm_postL3", post_l3)]:
        if not cks:
            continue
        s, b, adv, fr = frac_row(cks)
        out(f"    {setname:14s} {s:8.3f} {b:9.3f} {adv:+10.3f} {fr:7.3f}")

    # ---------------- guards ----------------
    out("\n(4) GUARDS")
    hs = [int(x) for x in a.h_sweep.split(",")]
    sw_ck = [c for c in derive_sweep(EVENTS, last_snap) if c in R_store] \
        or sorted(R_store)[::8]
    head_params = None
    guard = {}
    out(f"  capacity x seed sweep on {len(sw_ck)} checkpoints, {a.n_seeds} seeds per cell")
    out(f"    {'H':>5s} {'FM par':>8s} {'% head':>7s} {'ens_cos':>8s} {'|r|/|t|':>8s} "
        f"{'te_mse':>8s} {'R@d64':>7s} {'LOGIT@d64':>10s} {'cmass_R':>8s}")
    for H in hs:
        ec, rt, tm, dr, dl, cm = [], [], [], [], [], []
        for c in sw_ck:
            h = load_heads(root, c)
            live, ctx = live_slots(audi, c)
            target = pi_at(h, probe)[:, live]
            Rs, fs = [], []
            n_s = 1 if H == 0 else a.n_seeds     # the ridge rung has one minimiser
            for s in range(n_s):
                f = fit_residual(target, probe, tr, te, H, a.seed + 101 * s, a.iters, a.lr)
                Rs.append(f["R"]); fs.append(f)
            for i in range(len(Rs)):
                for j in range(i + 1, len(Rs)):
                    x, y = Rs[i].ravel(), Rs[j].ravel()
                    ec.append(float(x @ y / max(1e-12, np.linalg.norm(x) * np.linalg.norm(y))))
            rt.append(fs[0]["r_over_t"]); tm.append(fs[0]["te_loss"])
            levl = np.asarray([lev[int(s)] for s in live])
            macro = levl >= 2
            R = fs[0]["R"]
            e = (R ** 2)
            cm.append(float(np.mean(e[:, macro].sum(1) / np.clip(e.sum(1), 1e-12, None)))
                      if macro.any() else np.nan)
            keep = ~fs[0]["degen"]
            lab = (levl >= 2).astype(int)
            if keep.sum() >= 8 and len(np.unique(lab[keep])) > 1:
                cols, norms = build_columns(R, fs[0]["pred"], fs[0]["target"], live, levl,
                                            int(ctx["prop_k"]), a.seed + c)
                t = decode_table(cols, norms, keep, lab[keep], [64], lams, 0, a.seed + c)
                dr.append(t["R@d64"]["med"]); dl.append(t["LOGIT@d64"]["med"])
        npar = fs[0]["n_params"]
        hpar = FM(96, 8, 384, 56, 0).n_params()
        guard[H] = {"ens_cos": float(np.mean(ec)) if ec else 1.0,
                    "r_over_t": float(np.mean(rt)),
                    "te_mse": float(np.mean(tm)), "R_d64": float(np.mean(dr)) if dr else np.nan,
                    "LOGIT_d64": float(np.mean(dl)) if dl else np.nan,
                    "cmass_R": float(np.nanmean(cm)), "n_params": npar,
                    "pct_head": 100.0 * npar / hpar}
        g = guard[H]
        out(f"    {H:5d} {npar:8d} {g['pct_head']:6.1f}% {g['ens_cos']:8.3f} "
            f"{g['r_over_t']:8.3f} {g['te_mse']:8.4f} {g['R_d64']:7.3f} "
            f"{g['LOGIT_d64']:10.3f} {g['cmass_R']:8.3f}")
    out("  (paper 2's reference band for ens_cos: 0.78-0.91 real, 0.65 = the junk artifact.")
    out("   H=384 is exactly the head's own architecture — the saturation rung, on purpose.)")

    # hierarchy eta^2 — paper 2's structure diagnostic, in this substrate's currency
    out("\n  hierarchy-eta2: the residual's structure, conditioned on the grammar's own latent")
    out("  (the ROOT TARGET — this substrate's deep-hierarchy variable, paper 2's eta2 ~ 0.32)")
    out("  against the junk conditioners. Every cell is reported as observed / matched-random,")
    out("  because eta2 is inflated by group count and small groups and is unreadable alone.")
    out(f"    {'ckpt':>6s} {'|R| ~ root':>18s} {'|R| ~ step':>18s} {'|R| ~ src cycle':>18s} "
        f"{'class ~ root':>18s} {'|R| ~ root | cycle':>18s}")
    e2s = []
    pr, pst, pcy = probe["root"][te], probe["step"][te], probe["cycle"][te]
    for c in sw_ck:
        R, live, levl, keep, _, _, _ = R_store[c]
        nrm = np.linalg.norm(R, axis=1)
        F = unit_rows(embed56(R, live, n_slots))
        labk = (F @ C.T).argmax(1)
        er = eta2_vs_random(nrm, pr, a.seed + c)
        es = eta2_vs_random(nrm, pst, a.seed + c)
        ecy = eta2_vs_random(nrm, pcy, a.seed + c)
        ek = eta2_vs_random(labk.astype(float), pr, a.seed + c)
        # root and source-cycle are confounded on a fixed probe set, so the root read is also
        # taken WITHIN cycle strata: |R| is de-meaned per source cycle first
        nrm_w = nrm.copy()
        for cc in np.unique(pcy):
            mm_ = pcy == cc
            nrm_w[mm_] -= nrm_w[mm_].mean()
        erw = eta2_vs_random(nrm_w, pr, a.seed + c)
        e2s.append((c, er, es, ecy, ek, erw))
    for r in e2s[::max(1, len(e2s) // 8)]:
        out(f"    c{r[0]:5d} " + " ".join(f"{x[0]:8.3f}/{x[1]:<8.3f}" for x in r[1:]))
    e2a = np.asarray([[x[2] for x in r[1:]] for r in e2s])
    out(f"    mean excess over matched-random: root {e2a[:, 0].mean():+.3f}  "
        f"step {e2a[:, 1].mean():+.3f}  src-cycle {e2a[:, 2].mean():+.3f}  "
        f"class~root {e2a[:, 3].mean():+.3f}  root|cycle {e2a[:, 4].mean():+.3f}")

    # the value head as the no-token-structure control target
    out("\n  VALUE-HEAD CONTROL TARGET (same FM class, scalar output, no token index):")
    vd = []
    for c in sw_ck:
        h = load_heads(root, c)
        tgt = v_at(h, probe)[:, None]
        f = fit_residual(tgt, probe, tr, te, a.h_primary, fm_seed, a.iters, a.lr)
        vd.append((c, f["r_over_t"], float(np.linalg.norm(f["R"]))))
    vr = np.asarray([x[1] for x in vd])
    vn = np.asarray([x[2] for x in vd])
    out(f"    |r|/|t| mean {vr.mean():.3f} (sd {vr.std():.3f}); ||R|| across checkpoints "
        f"cv {vn.std() / max(1e-12, vn.mean()):.3f}")
    out("    (a value-head residual that moves at the commits as much as pi's would say the"
        " reorganisation is the state distribution or the calendar, not the command port)")

    # ---------------- at_support ----------------
    out("\n(5) DOES IT TRACK at_support YIELD?")
    cyc_arr = np.asarray(sorted(per_ck))
    era_arr = np.asarray([per_ck[c]["era"] for c in cyc_arr])
    series = {
        "cmass_R": np.asarray([per_ck[c]["cmass_R"] for c in cyc_arr]),
        "cmass_excess": np.asarray([per_ck[c]["cmass_excess"] for c in cyc_arr]),
        "occ_drift": np.asarray([drift[int(np.flatnonzero(occ_c == c)[0])] for c in cyc_arr]),
        "occ_cmd": np.asarray([occ_cmd[int(np.flatnonzero(occ_c == c)[0])] for c in cyc_arr]),
        "decode_R": np.asarray([dec[c]["R@d64"]["med"] if c in dec else np.nan
                                for c in cyc_arr]),
        "decode_adv": np.asarray([
            (dec[c]["R@d64"]["med"] - max(dec[c]["LOGIT@d64"]["med"], dec[c]["IO@d64"]["med"],
                                          dec[c]["BEH@d64"]["med"]))
            if c in dec else np.nan for c in cyc_arr]),
        "r_over_t": np.asarray([per_ck[c]["r_over_t"] for c in cyc_arr]),
    }
    behav = {
        "macro_mass": np.asarray([per_ck[c]["macro_mass"] for c in cyc_arr]),
        "own_spell": np.asarray([per_ck[c]["own_spell"] for c in cyc_arr]),
        "spell_ratio": np.asarray([per_ck[c]["spell_ratio"] for c in cyc_arr]),
    }
    out("  gauge = panel['at_support'][level], the free one-level-up read A1/A2 drive on.")
    out(f"  {'series':14s} {'level':>5s} {'rho_raw':>8s} {'p':>8s} {'rho_partial':>12s} {'n':>5s}")
    corr = {}
    for level in (3, 4):
        gs = at_support_series(res, level)
        g = np.asarray([gs[c - 1] if 1 <= c <= len(gs) else np.nan for c in cyc_arr])
        for name, s in series.items():
            rho, p, part, n = partial_spearman(s, g, cyc_arr.astype(float), era_arr)
            corr[f"{name}@L{level}"] = {"rho": rho, "p": p, "partial": part, "n": n}
            out(f"  {name:14s} {level:5d} {rho:+8.3f} {p:8.3f} {part:+12.3f} {n:5d}")

    out("\n  the consolidation BEHAVIOUR series, recomputed offline at the same probe states")
    out("  (`native`'s can't-decompose readout: pi's mass on the argmax macro's own spelling):")
    out(f"    {'cycle':>6s} {'macro_mass':>11s} {'own_macro':>10s} {'own_spell':>10s} "
        f"{'spell_ratio':>12s}")
    for c in sorted(per_ck):
        if c % 10 == 0 or c in EVENTS.values():
            p = per_ck[c]
            out(f"    c{c:5d} {p['macro_mass']:11.4f} {p['own_macro']:10.4f} "
                f"{p['own_spell']:10.4f} {p['spell_ratio']:12.4f}")
    out(f"\n  residual series against the consolidation behaviour series "
        f"(rho_raw / rho_partial):")
    for bname, bs in behav.items():
        row = f"    {bname:12s}"
        for name, s in series.items():
            rho, p, part, n = partial_spearman(s, bs, cyc_arr.astype(float), era_arr)
            corr[f"{name}~{bname}"] = {"rho": rho, "p": p, "partial": part, "n": n}
            row += f"  {name}={rho:+.2f}/{part:+.2f}"
        out(row)
    out("  rho_partial residualises BOTH series on cycle and era (audiation finding 1 measured")
    out("  eta2(update~cycle) = 0.438 on this very run — the calendar confound is real here).")

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ev = [EVENTS["ms_grow_L2"], EVENTS["ms_grow_L3"]]
        eras = [EVENTS["era2"], EVENTS["era3"], EVENTS["era4"], EVENTS["era5"]]

        fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
        ax[0].stackplot(occ_c, O.T, labels=[f"k{k} (cmd {cls_cmd[k]:.2f})"
                                            for k in range(a.kmeans_k)])
        ax[0].set_ylabel("residual-class occupancy")
        ax[0].legend(fontsize=6, ncol=4, loc="upper left")
        ax[1].plot(occ_c, drift, lw=1.2)
        ax[1].set_ylabel("occupancy drift TV(o_c, o_{c-1})")
        ax[2].plot(cyc_arr, [per_ck[c]["cmass_R"] for c in cyc_arr], label="cmass R")
        ax[2].plot(cyc_arr, [per_ck[c]["cmass_LOGIT"] for c in cyc_arr], label="cmass LOGIT")
        ax[2].plot(cyc_arr, [per_ck[c]["macro_share"] for c in cyc_arr], "k--",
                   label="macro share of live slots")
        ax[2].set_ylabel("command-energy share")
        ax[2].set_xlabel("checkpoint (cycle)")
        ax[2].legend(fontsize=7)
        for x in ev:
            for b in ax:
                b.axvline(x, color="crimson", lw=1.0)
        for x in eras:
            for b in ax:
                b.axvline(x, color="grey", lw=0.7, ls=":")
        fig.suptitle(f"continuo / {a.tag}: residual-class occupancy across a practice run\n"
                     "red = action-set growth at a commit; dotted = era advance")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "occupancy.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        cells = ["R", "FM", "LOGIT", "IO", "BEH", "SHUF", "NORM"]
        w = 0.8 / len(d_projs)
        for i, d in enumerate(d_projs):
            vals = [agg(f"{cc}@d{d}", warm or sorted(dec))[0] for cc in cells]
            ax[0].bar(np.arange(len(cells)) + i * w, vals, width=w, label=f"d={d}")
        ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
        ax[0].set_xticks(np.arange(len(cells)) + 0.4)
        ax[0].set_xticklabels(cells)
        ax[0].set_ylabel("balanced acc (LOO over slots)")
        ax[0].set_title("instruction/data decode: self vs the observer twin")
        ax[0].legend(fontsize=7)
        hh = sorted(guard)
        ax[1].plot([guard[x]["pct_head"] for x in hh], [guard[x]["ens_cos"] for x in hh],
                   "o-", label="ens_cos")
        ax[1].plot([guard[x]["pct_head"] for x in hh], [guard[x]["R_d64"] for x in hh],
                   "s-", label="decode R@d64")
        ax[1].plot([guard[x]["pct_head"] for x in hh], [guard[x]["r_over_t"] for x in hh],
                   "^-", label="|r|/|t|")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("FM params as % of the pi head's")
        ax[1].axhline(0.65, color="crimson", lw=0.8, ls=":")
        ax[1].set_title("the saturation guard")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "decode_guards.png"), dpi=130)
        plt.close(fig)
        out("\n  figures: occupancy.png, decode_guards.png")
    except Exception as e:  # noqa: BLE001
        out(f"\n  FIGURES FAILED: {e}")

    out(f"\nTOTAL {time.time() - t0:.0f}s")
    open(os.path.join(outdir, "reduction.txt"), "w").write("\n".join(lines) + "\n")
    json.dump({
        "argv": sys.argv[1:], "per_ck": per_ck, "guard": {str(k): v for k, v in guard.items()},
        "events": EVENTS, "event_drift": ev_rows,
        "occ_cycles": occ_c.tolist(), "occ": O.tolist(), "drift": drift.tolist(),
        "class_cmd_mass": cls_cmd.tolist(), "occ_cmd": occ_cmd.tolist(),
        "decode": {str(k): v for k, v in dec.items()},
        "decode_age": {str(k): v for k, v in dec3.items()},
        "decode_age_floor": age_rows,
        "corr": corr, "eta2": [[r[0]] + [list(map(float, x)) for x in r[1:]] for r in e2s],
    }, open(os.path.join(outdir, "continuo.json"), "w"), indent=1)
    print(f"\nwrote {outdir}/reduction.txt")


if __name__ == "__main__":
    main()
