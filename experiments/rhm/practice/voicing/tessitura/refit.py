"""[tessitura] THE DIETS — norm §1's between-subjects design, on voicing's trunk held fixed.

`reduce_rows.py`'s [R] scored each banked critic on the other's rows and had to carry a caveat
it could not remove: **each critic carries its own trunk**, so that comparison is two whole
readers and not two readouts of one state, and the two arms differ in seed AND in probe draw at
once. `logit_reading/striatum/norm/` had neither problem — one frozen trunk, one shared set of
held-out rows, eleven diets differing only in the outcome distribution they were fed.

This file recreates that design here. The trunk is `vo_heads.pt`'s, held FIXED; the rows are the
banked buffers; the critic is **re-fitted from scratch** under each diet with the same
architecture, the same initialisation, the same optimizer, the same number of steps and the same
batch size; and every refit's level is read on the IDENTICAL held-out rows — the run's own
bijective hold code (`code < 100 * vo_critic_hold`), which no diet ever trains on.

WHAT THE DIETS ARE. Two families, both pinned the way norm pinned its `(etype, j)` cell
histogram, which here means pinning the SLOT counts (and therefore the level counts, since a
slot has one level) and the SOURCE composition:

  the SOURCE family     `filed`, `probe`, `both` — the axis `voicing` Q3 built and `overtone`
                        measured. The three differ enormously in expected outcome (a filed write
                        solves ~0.34 of the time, a substituted class ~0.07), which is exactly
                        the manipulation norm's `out` family had to construct by selection.
  the OUTCOME family    `out_lo`, `out_mid`, `out_hi` — within each slot AND within each source,
                        three target base rates on a multiplicative ladder (0.6x / 1.0x / 1.4x)
                        around each cell's OWN natural rate, drawn from that cell's own rows. So the
                        three diets hold the slot counts, the level counts and the filed:probe
                        ratio identical and move only E[outcome]. This is norm's "terciles taken
                        inside each cell with equal counts leave the histogram bit-identical and
                        only the outcome moves", one substrate over.

  plus `full` (the whole training pool — the banked diet, and the regression's reference) and a
  per-diet SHUFFLED-OUTCOME floor.

THE RANDOM-INIT TRUNK IS CARRIED, because it is the control that made norm §1 a finding rather
than an observation: on a random feature map the diets could only SHIFT the norm, while on the
trained trunk they also RESCALED it. The same two columns are here.

WHAT THIS STILL IS NOT. The trunk is `voicing`'s trained executor trunk, not a frozen
next-token trunk, and it was shaped by the run that produced these rows — so "fixed" means fixed
across the diets, not independent of them. And the rows are the ring-buffer tail, as everywhere
in this node.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/refit.py
    python3 rhm/practice/voicing/tessitura/refit.py --tag ov_s2 --arm ovt_comp_pr_dis --steps 400
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICING = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(VOICING, "..", "..", "..")))
FIG = os.path.join(VOICING, "figures")

DIETS = ["full", "filed", "probe", "both", "out_lo", "out_mid", "out_hi"]


def auc(x, y):
    """the rank identity with ties averaged — `voicing.py::vo_auc` on numpy alone."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(1, len(x) + 1)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ols(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    x, y = x[m], y[m]
    if x.std() < 1e-12:
        return None
    b = float(np.cov(x, y, bias=True)[0, 1] / (x.std() ** 2))
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": b, "intercept": float(y.mean() - b * x.mean()), "r2": r * r,
            "sd_ratio": float(y.std() / x.std()), "n": int(m.sum())}


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def f(x, nd=3):
    return ("  -  " if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"{x:.{nd}f}")


# ------------------------------------------------------------------ the feature cache --- #

def build_cache(core, z, meta, thr, SN, torch, batch=1024):
    """Per slot: the two things `Critic.ctx_state` reads out of `pooled`, for every row.

    `ctx_state` is `ctx(pooled.mean(1)) + slot(sid) + sum_j in_proj[j](pooled[:, blk0+j, :])`,
    so the mean vector and the slot's own `span` block vectors are EXACTLY sufficient and the
    trunk never has to be run again. The rows keep their buffer order, so the hold code, the
    verdict and the source tag all index the same way.
    """
    out = {}
    for pre in sorted(meta):
        which, slot = pre.split(":", 1)
        mt = meta[pre]
        blk0, span = int(mt["blk0"]), int(mt["span"])
        ob = z[f"{pre}|obs"].astype(np.int64)
        # ONE TRUNK FORWARD PER DISTINCT CONTEXT. The buffers hold the same masked context many
        # times over (one per surviving beam row), so deduplicating is a 3-7x saving and changes
        # nothing: the state is a function of the context alone. `inv` puts every row back.
        key = np.ascontiguousarray(ob).view([("", ob.dtype)] * ob.shape[1]).ravel()
        uq, inv = np.unique(key, return_inverse=True)
        seen = {}
        for i_, q_ in enumerate(inv):
            if q_ not in seen:
                seen[q_] = i_
        first = np.array([seen[q_] for q_ in range(len(uq))], np.int64)
        obs_u = torch.from_numpy(ob[first])
        mus, blks = [], []
        with torch.no_grad():
            for a_ in range(0, obs_u.shape[0], batch):
                pl, _ = SN.trunk(core, obs_u[a_:a_ + batch])
                mus.append(pl.mean(1))
                blks.append(pl[:, blk0:blk0 + span, :])
        out[pre] = {"mu_u": torch.cat(mus), "blk_u": torch.cat(blks),
                    "inv": torch.from_numpy(inv.astype(np.int64)),
                    "w": torch.from_numpy(z[f"{pre}|write"].astype(np.int64)),
                    "y": torch.from_numpy(z[f"{pre}|y"].astype(np.float32)),
                    "code": z[f"{pre}|code"].astype(np.int64),
                    "sid": int(mt["slot_id"]), "span": span, "blk0": blk0,
                    "level": int(mt["level"]), "which": which, "slot": slot,
                    "n_ctx": int(len(uq))}
    return out


def ctx_state(critic, cc, idx, torch):
    """`Critic.ctx_state` recomputed from the cache. Gate T-6 asserts it is the identity."""
    ui = cc["inv"][torch.as_tensor(np.asarray(idx, np.int64))]
    u = critic.ctx(cc["mu_u"][ui]) + critic.slot(
        torch.full((len(idx),), cc["sid"], dtype=torch.long))
    bl = cc["blk_u"][ui]
    for j in range(cc["span"]):
        u = u + critic.in_proj[j](bl[:, j, :])
    return u


def score(critic, cc, idx, torch):
    u = ctx_state(critic, cc, idx, torch)
    e = critic.cand_state(cc["w"][idx], cc["span"])
    return critic.mlp(u + e).squeeze(-1)


# ------------------------------------------------------------------------ the diets --- #

def make_diets(cache, thr, rng):
    """{diet: {slot_prefix: train row indices}} with the slot counts, the level counts and the
    filed:probe ratio PINNED, and only E[outcome] moving."""
    slots = sorted({cache[p]["slot"] for p in cache},
                   key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
    out = {d: {} for d in DIETS}
    for slot in slots:
        pres = [p for p in cache if cache[p]["slot"] == slot]
        tr = {p: np.nonzero(cache[p]["code"] >= thr)[0] for p in pres}
        fp = [p for p in pres if cache[p]["which"] == "filed"]
        pp = [p for p in pres if cache[p]["which"] == "probe"]
        if not fp or not pp:
            continue
        nf, npb = len(tr[fp[0]]), len(tr[pp[0]])
        # THE SOURCE FAMILY, at equal TOTAL rows per slot so a diet cannot win on volume.
        n_src = min(nf, npb)
        out["filed"][fp[0]] = tr[fp[0]][-n_src:]
        out["probe"][pp[0]] = tr[pp[0]][-n_src:]
        out["both"][fp[0]] = tr[fp[0]][-(n_src // 2):]
        out["both"][pp[0]] = tr[pp[0]][-(n_src - n_src // 2):]
        out["full"][fp[0]] = tr[fp[0]]
        out["full"][pp[0]] = tr[pp[0]]
        # THE OUTCOME FAMILY — A BASE-RATE LADDER, and the first construction is WITHDRAWN.
        #
        # WHAT WAS TRIED FIRST, and why it is not here: norm's `out_lo/mid/hi` are within-cell
        # terciles of the WINDOW'S MEAN OUTCOME, a continuous quantity, and all three of its
        # diets kept a workable base rate (0.596 / 0.686 / 0.770). The direct transcription
        # here — group the training rows by context, sort the contexts by their mean verdict,
        # cut three groups of equal row count — produced `out_lo` with **E[outcome] = 0.0000**
        # and `out_hi` with 0.5888. The reason is a property of this substrate rather than a
        # bug: a context's rows are beam siblings that nearly always share one verdict (the
        # twin reduction measures the disagreement rate at 0.09-0.18), so a sort on the context
        # mean is a near-binary sort and the bottom third is simply every all-zero context. A
        # diet with no successes in it is not a low-outcome world; it is a world where nothing
        # ever works, and a critic cannot be identified on it at all. The reading is kept in
        # `DESIGN.md` beside its correction.
        #
        # WHAT IS HERE INSTEAD: three target base rates on a multiplicative ladder around each
        # (slot, source)'s OWN natural rate, `0.6x / 1.0x / 1.4x`, realised by drawing the
        # positives and the negatives in the required proportion from that cell's own training
        # rows. Slot counts, level counts and the filed:probe ratio stay pinned exactly, every
        # diet stays trainable, and the only thing that moves is E[outcome] — which is what
        # norm's family was for. Selecting on the critic's own teacher IS the manipulation, as
        # norm says of its own: "as Xiang's experimenters chose which offers each group saw".
        for p in (fp[0], pp[0]):
            rows = tr[p]
            y = cache[p]["y"].numpy()[rows]
            pos, neg = rows[y > 0.5], rows[y <= 0.5]
            p0 = len(pos) / max(len(rows), 1)
            if len(pos) < 24 or len(neg) < 24:
                for nm in ("out_lo", "out_mid", "out_hi"):
                    out[nm][p] = rows           # too few of one class to ladder; kept whole
                continue
            targets = [max(0.01, min(0.99, p0 * r)) for r in (0.6, 1.0, 1.4)]
            # the largest equal n every target can serve from this cell's own rows
            n_eq = int(min(min(len(pos) / t, len(neg) / (1.0 - t)) for t in targets))
            n_eq = max(24, min(n_eq, len(rows)))
            for nm, t in zip(("out_lo", "out_mid", "out_hi"), targets):
                k = int(round(t * n_eq))
                k = max(1, min(k, len(pos), n_eq - 1))
                take_p = pos[rng.choice(len(pos), size=k, replace=False)]
                take_n = neg[rng.choice(len(neg), size=n_eq - k, replace=False)]
                out[nm][p] = np.sort(np.concatenate([take_p, take_n]))
    return out


# ------------------------------------------------------------------------ the refit --- #

def refit(cache, diet, steps, lr, seed, cfgh, SN, torch, build_critic, batch=64,
          shuffle_y=False, rng=None):
    """One critic, from scratch, on one diet. Same architecture, same init seed, same optimizer,
    same step count and same batch as every other diet — the only thing that differs is which
    rows it is shown."""
    import torch.nn.functional as F
    hm = int(cfgh.get("ov_critic_hidden", -1))
    cr = build_critic(SN.slot_count(int(cfgh["s"]), int(cfgh["depth"]),
                                    int(cfgh["max_macro_level"])),
                      int(cfgh["v"]), int(cfgh["state_dim"]),
                      int(cfgh["s"]) ** (int(cfgh["max_macro_level"]) - 1), seed=int(seed),
                      device=torch.device("cpu"),
                      hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
    cr.train()
    opt = torch.optim.Adam(cr.parameters(), lr=float(lr))
    g = np.random.default_rng(int(seed) + 991)
    ys = {}
    if shuffle_y:
        for p, idx in diet.items():
            yy = cache[p]["y"].numpy()[idx].copy()
            (rng or g).shuffle(yy)
            ys[p] = torch.from_numpy(yy)
    losses = []
    keys = sorted(diet)
    for _ in range(int(steps)):
        terms = []
        for p in keys:
            idx = diet[p]
            if len(idx) < 8:
                continue
            take = g.integers(0, len(idx), size=min(batch, len(idx)))
            sel = idx[take]
            lg = score(cr, cache[p], sel, torch)
            y = (ys[p][take] if shuffle_y else cache[p]["y"][sel])
            terms.append(F.binary_cross_entropy_with_logits(lg, y))
        if not terms:
            break
        loss = sum(terms) / len(terms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    cr.eval()
    return cr, (float(np.mean(losses[-20:])) if losses else None)


def read_hold(cr, cache, thr, torch):
    """Every refit is read on the IDENTICAL held-out rows: `code < thr`, which no diet trains
    on. Returns {slot_prefix: (p (n,), y (n,))}."""
    out = {}
    with torch.no_grad():
        for p, cc in cache.items():
            idx = np.nonzero(cc["code"] < thr)[0]
            if len(idx) < 16:
                continue
            lg = score(cr, cc, idx, torch).numpy().astype(np.float64)
            out[p] = (1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30))), cc["y"].numpy()[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ov_s0b")
    ap.add_argument("--arm", default="ovt_comp_pr_sh")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import rhm.rhm_generative_planner as GP
    import rhm.practice.native.span.span_net as SN
    from rhm.practice.voicing.voicing import build_critic

    root = os.path.join(FIG, a.tag, a.arm)
    blob = torch.load(os.path.join(root, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    cfgh = blob["cfg"]
    v, s, depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
    dim, maxl = int(cfgh["state_dim"]), int(cfgh["max_macro_level"])
    thr = int(round(float(cfgh["vo_critic_hold"]) * 100))
    length = s ** depth
    core = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False)
    core.load_state_dict(blob["core"])
    core.eval()
    torch.manual_seed(20260917)
    rcore = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                  root_conditioned=False)
    rcore.eval()
    z = np.load(os.path.join(root, "vo_rows.npz"))
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))

    lines = []

    def o(t=""):
        lines.append(t)
        print(t, flush=True)

    o("=" * 104)
    o(f"[tessitura] THE DIETS — norm §1 on a fixed trunk: {a.tag}:{a.arm}")
    o("=" * 104)
    for ln in __doc__.strip().split("Usage")[0].strip().splitlines():
        o("  " + ln)
    o("")
    o(f"  steps {a.steps} · lr {a.lr} · batch 64 per slot per step · init seed {a.seed} "
      f"(the SAME for every diet) · hold code < {thr}")
    o("")

    caches = {}
    for nm, cr_core in (("trained", core), ("random", rcore)):
        caches[nm] = build_cache(cr_core, z, meta, thr, SN, torch)
    o(f"  feature cache built for both trunks: {len(caches['trained'])} buffers")

    # ---------------------------------------------------------------- gate T-6 --------- #
    o("")
    o("-" * 104)
    o("[T] THE GATES")
    o("-" * 104)
    o("  T-6  THE CACHE IS THE TRUNK. `Critic.ctx_state` reads `pooled.mean(1)` and the slot's")
    o("       own `span` blocks and nothing else, so caching those two is sufficient and the")
    o("       trunk never runs again. Against the real forward through the BANKED critic that")
    o("       must be an identity, or every refit below is reading a different state.")
    hm = int(cfgh.get("ov_critic_hidden", -1))
    cr_bank = build_critic(SN.slot_count(s, depth, maxl), v, dim, s ** (maxl - 1), seed=0,
                           device=torch.device("cpu"),
                           hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
    cr_bank.load_state_dict(blob["critic"])
    cr_bank.eval()
    worst = 0.0
    for p in list(caches["trained"])[:6]:
        cc = caches["trained"][p]
        idx = np.arange(min(1024, len(cc["y"])))
        with torch.no_grad():
            got = score(cr_bank, cc, idx, torch).numpy()
            obs = torch.from_numpy(z[f"{p}|obs"].astype(np.int64))[idx]
            pl, _ = SN.trunk(core, obs)
            sid = torch.full((len(idx),), cc["sid"], dtype=torch.long)
            want = cr_bank(pl, cc["blk0"], cc["span"], sid, cc["w"][idx]).numpy()
        worst = max(worst, float(np.abs(got - want).max()))
    o(f"       max |delta| over 6 buffers x 1024 rows = {worst:.3e}  "
      f"[{'PASS' if worst < 1e-5 else 'FAIL'}]")
    assert worst < 1e-5, "T-6 failed: the cached state is not the trunk's"
    o("       FALSIFIED: perturbing one cached block vector by 1e-3 must break it —")
    cc = caches["trained"][list(caches["trained"])[0]]
    # the perturbation has to land on the unique entries the FIRST FOUR ROWS actually map to;
    # `np.unique` sorts, so unique slot 0 is the lexicographically smallest context and not
    # row 0's. Perturbing `[:4]` blind is how this falsification first came out vacuous.
    ui4 = cc["inv"][:4].clone()
    keep = cc["blk_u"][ui4].clone()
    cc["blk_u"][ui4] += 1e-3
    with torch.no_grad():
        got = score(cr_bank, cc, np.arange(4), torch).numpy()
        obs = torch.from_numpy(z[f"{list(caches['trained'])[0]}|obs"].astype(np.int64))[:4]
        pl, _ = SN.trunk(core, obs)
        sid = torch.full((4,), cc["sid"], dtype=torch.long)
        want = cr_bank(pl, cc["blk0"], cc["span"], sid, cc["w"][:4]).numpy()
    d_bad = float(np.abs(got - want).max())
    cc["blk_u"][ui4] = keep
    o(f"       perturbed max |delta| = {d_bad:.3e}  "
      f"[{'FAILS AS REQUIRED' if d_bad > 1e-5 else 'VACUOUS'}]")
    assert d_bad > 1e-5, "T-6 falsification vacuous"

    rng = np.random.default_rng(4242)
    diets = make_diets(caches["trained"], thr, rng)

    o("")
    o("  T-7  THE DIETS ARE PINNED. Every diet inside a family must carry the IDENTICAL number")
    o("       of rows per slot and the IDENTICAL filed:probe ratio, so the only thing that")
    o("       moves is E[outcome] — norm's `(etype, j)` cell histogram, one substrate over.")
    fams = {"source": ["filed", "probe", "both"], "outcome": ["out_lo", "out_mid", "out_hi"]}
    for fam, ds in fams.items():
        per = {}
        for d in ds:
            bysl = collections.defaultdict(lambda: [0, 0])
            for p, idx in diets[d].items():
                bysl[caches["trained"][p]["slot"]][0 if p.startswith("filed") else 1] += len(idx)
            per[d] = {k: tuple(vv) for k, vv in bysl.items()}
        ref = per[ds[0]]
        if fam == "source":
            eq = all(sum(per[d][k]) == sum(ref[k]) for d in ds for k in ref)
            o(f"       {fam:8} family: identical TOTAL rows per slot across "
              f"{ds} = {eq}  (the filed:probe ratio IS the manipulation here)")
        else:
            eq = all(per[d][k] == ref[k] for d in ds for k in ref)
            o(f"       {fam:8} family: identical (filed, probe) counts per slot across "
              f"{ds} = {eq}")
        assert eq, f"T-7 failed for the {fam} family"

    o("")
    o("  T-8  THE HELD-OUT ROWS ARE IDENTICAL AND UNTOUCHED. Every refit is read on")
    o("       `code < thr`; no diet may contain one of those rows.")
    leak = sum(int((caches["trained"][p]["code"][idx] < thr).sum())
               for d in DIETS for p, idx in diets[d].items())
    n_hold = sum(int((cc["code"] < thr).sum()) for cc in caches["trained"].values())
    o(f"       held-out rows {n_hold} · training rows that leak into a diet = {leak}  "
      f"[{'PASS' if leak == 0 else 'FAIL'}]")
    assert leak == 0, "T-8 failed: a diet trains on held-out rows"
    o("")

    # ------------------------------------------------------------- the diet table ------ #
    o("-" * 104)
    o("[D] THE DIETS — what each was fed (norm's own table)")
    o("-" * 104)
    o(f"    {'diet':8} {'rows':>9} {'filed':>9} {'probe':>9} {'E[outcome]':>11} "
      f"{'E[y] filed':>11} {'E[y] probe':>11}")
    dstat = {}
    for d in DIETS:
        n = nf = npb = 0
        sy = syf = syp = 0.0
        for p, idx in diets[d].items():
            y = caches["trained"][p]["y"].numpy()[idx]
            n += len(idx)
            sy += y.sum()
            if p.startswith("filed"):
                nf += len(idx)
                syf += y.sum()
            else:
                npb += len(idx)
                syp += y.sum()
        dstat[d] = {"n": n, "e": sy / max(n, 1)}
        o(f"    {d:8} {n:>9} {nf:>9} {npb:>9} {sy / max(n, 1):>11.4f} "
          f"{(syf / nf if nf else float('nan')):>11.4f} "
          f"{(syp / npb if npb else float('nan')):>11.4f}")
    o("")

    # ------------------------------------------------------------------ the refits ----- #
    o("-" * 104)
    o("[N] THE NORM PER DIET, ON THE IDENTICAL HELD-OUT ROWS")
    o("-" * 104)
    o("  `norm` is the refit's mean predicted P(solve) over every held-out row of every slot —")
    o("  the SAME rows for every diet and every trunk. `AUC` is its ranking on those rows and")
    o("  `floor` the same refit trained on a per-diet SHUFFLE of the outcome, which is the only")
    o("  thing that says whether a diet carried any signal at all.")
    o("")
    norms, aucs = {}, {}
    for trunk in ("trained", "random"):
        o(f"  TRUNK = {trunk}")
        o(f"    {'diet':8} {'E[outcome]':>11} {'norm':>8} {'norm-E[out]':>12} "
          f"{'norm/fil':>9} {'norm/prb':>9} {'AUC':>8} {'floor':>8} {'loss':>8}")
        for d in DIETS:
            cr, loss = refit(caches[trunk], diets[d], a.steps, a.lr, a.seed, cfgh, SN, torch,
                             build_critic)
            rd = read_hold(cr, caches[trunk], thr, torch)
            p_all = np.concatenate([rd[p][0] for p in sorted(rd)])
            y_all = np.concatenate([rd[p][1] for p in sorted(rd)])
            # filed and probe held-out rows are DIFFERENT OBJECTS on this node and are never
            # pooled in a reading; the pooled column is kept because the regression in [S] is
            # taken over every held-out row and the two have to add up to it.
            pf_ = np.concatenate([rd[p][0] for p in sorted(rd) if p.startswith("filed")])
            pp_ = np.concatenate([rd[p][0] for p in sorted(rd) if p.startswith("probe")])
            norms[(trunk, d)] = rd
            # AUC per slot, pooled by pair count (pooling raw scores would rank slot identity)
            num = den = 0.0
            for p in sorted(rd):
                aa = auc(rd[p][0], rd[p][1])
                if aa is None:
                    continue
                w = float((rd[p][1] > 0.5).sum()) * float((rd[p][1] <= 0.5).sum())
                num += aa * w
                den += w
            aucs[(trunk, d)] = (num / den) if den else None
            crs, _ = refit(caches[trunk], diets[d], a.steps, a.lr, a.seed, cfgh, SN, torch,
                           build_critic, shuffle_y=True,
                           rng=np.random.default_rng(1234))
            rs = read_hold(crs, caches[trunk], thr, torch)
            numf = denf = 0.0
            for p in sorted(rs):
                aa = auc(rs[p][0], rs[p][1])
                if aa is None:
                    continue
                w = float((rs[p][1] > 0.5).sum()) * float((rs[p][1] <= 0.5).sum())
                numf += aa * w
                denf += w
            o(f"    {d:8} {dstat[d]['e']:>11.4f} {p_all.mean():>8.4f} "
              f"{p_all.mean() - dstat[d]['e']:>+12.4f} {pf_.mean():>9.4f} "
              f"{pp_.mean():>9.4f} {f(aucs[(trunk, d)]):>8} "
              f"{f((numf / denf) if denf else None):>8} {f(loss, 4):>8}")
        o("")

    # --------------------------------------------------- shift versus rescaling -------- #
    o("-" * 104)
    o("[S] SHIFT OR RESCALING — norm §1's regression, on the identical rows")
    o("-" * 104)
    o("  Each diet's per-row norm regressed on `full`'s across the SAME held-out rows. Slope 1")
    o("  with an intercept is a pure LEVEL SHIFT; slope != 1 or sd ratio != 1 is a RESCALING;")
    o("  R^2 < 1 says the map is not exactly affine either. norm found a random-init trunk can")
    o("  only shift and the trained trunk also rescales — this is that comparison.")
    o("")
    for trunk in ("trained", "random"):
        o(f"  TRUNK = {trunk}")
        o(f"    {'diet':8} {'slope':>8} {'intercept':>10} {'R2':>7} {'sd ratio':>9} "
          f"{'mean shift':>11} {'n':>8}")
        ref = norms[(trunk, "full")]
        xs = np.concatenate([ref[p][0] for p in sorted(ref)])
        for d in DIETS:
            rd = norms[(trunk, d)]
            ys = np.concatenate([rd[p][0] for p in sorted(rd)])
            r = ols(xs, ys)
            if r is None:
                continue
            o(f"    {d:8} {r['slope']:>8.3f} {r['intercept']:>+10.4f} {r['r2']:>7.3f} "
              f"{r['sd_ratio']:>9.3f} {ys.mean() - xs.mean():>+11.4f} {r['n']:>8}")
        o("")
    o("  RANK ORDER (norm's headline): Spearman rho of the diet's norm against the diet's own")
    o("  E[outcome], inside each family. +1 is the rank-one result norm reported at every")
    o("  level, checkpoint and anchor.")
    for trunk in ("trained", "random"):
        for fam, ds in fams.items():
            e = [dstat[d]["e"] for d in ds]
            nn = [float(np.concatenate([norms[(trunk, d)][p][0]
                                        for p in sorted(norms[(trunk, d)])]).mean())
                  for d in ds]
            o(f"    {trunk:8} {fam:8} E[outcome] {[round(x, 4) for x in e]} -> "
              f"norm {[round(x, 4) for x in nn]}   rho = {f(spearman(e, nn))}")
    o("")
    o("  NON-VACUITY: the diets must actually produce DIFFERENT readers. Max pairwise")
    o("  |delta norm| on the identical rows, per trunk:")
    for trunk in ("trained", "random"):
        vals = [float(np.concatenate([norms[(trunk, d)][p][0]
                                      for p in sorted(norms[(trunk, d)])]).mean())
                for d in DIETS]
        o(f"    {trunk:8} spread = {max(vals) - min(vals):.4f} over {len(DIETS)} diets")
        assert max(vals) - min(vals) > 1e-3, f"the diets moved nothing on the {trunk} trunk"

    out = a.out or os.path.join(HERE, "results", f"refit_{a.tag}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
