"""woodshed — F1 part 1: the trust-formation-rate instrument, offline.

Reads the per-cycle logs already fetched under each A-lineage node's `figures/<tag>/<arm>/
results.json` and produces post-arrival growth curves for pi's per-level proposal mass:
mass vs cycles-since-arrival, per level, per arm, per tag.

Nothing here trains anything. No Modal, no GPU. It is a pure re-read of bytes that are
already on disk, so every number it prints is a property of runs that already happened.

WHAT THE READOUT IS (semantics, verified against the producing code)
--------------------------------------------------------------------
`log["probe"][k]["pi"]["mean_mass"]` is written by
`native/prop/prop_net.py::decompose_probe` (line 242):

    logits = prop(z, roots).masked_fill(~avail_mask[None, :], -inf)
    p      = softmax(logits, -1)
    mean_mass = p.mean(0)                       # mean over the probe's states

so it is a *normalised distribution over action slots*, averaged over the probe batch.
Slots are laid out level-major by `prop_net.py::slot_layout(depth, s, max_level)`:
offset[l] = sum_{k<l} s**(depth-k), width[l] = s**(depth-l). At depth 6, s 2 that is
L1 0..31, L2 32..47, L3 48..55, L4 56..59 (56 slots with max_level 3, 60 with 4).

Per-level mass at a probe := mean_mass[offset[l] : offset[l]+width[l]].sum().
Per-level argmax share := argmax_hist[same slice].sum() / n.

Two consequences that the reduction reports explicitly rather than hiding:
 (i) unavailable slots are masked to -inf, hence *exactly* 0.0 after the softmax. A level
     with no committed table therefore reads 0.000 by construction, not by measurement.
 (ii) a level's mass is spread over only the (level, node) slots that are actually in the
     live action set, i.e. that have a table entry. Table growth (extension/recert) raises
     the number of open slots. So raw mass conflates "more slots open" with "more mass per
     open slot", and both are tabulated (`n_avail`, `mass/slot`).

The probe cadence is `--probe-every` (8 in every tag read here) plus extra probes fired at
event cycles, so the clock is coarse: ~5-8 cycles per sample.

ARRIVAL
-------
Arrival cycle a(l) is the cycle at which level l's table becomes installed:
  * earned arms: the `commit` event for that level (`events[kind=commit]`);
  * gifted arms (`given_*`): the table is present at cycle 1 (`log["vocab"][0][l]` is not
    None), so a(l) = 1.
tau := probe cycle - a(l).

Usage:
    python3 rhm/practice/woodshed/reduce_trust.py            # text reduction -> stdout+file
    python3 rhm/practice/woodshed/reduce_trust.py --figures  # + PNGs
"""

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")

# (node dir, tag, family label, note). Order = the order the reduction prints them in.
MANIFEST = [
    ("conductor", "cd_s0", "A1", "the thermostat round; 6 arms"),
    ("maestro", "ma_s0", "A2", "the learned-policy round; 5 arms"),
    ("maestro", "ma_s1", "A2", "A2's stream-displaced twins"),
    ("crescendo", "cr3_s0", "A3", "L4 opened; draw A"),
    ("crescendo", "cr3_s1", "A3", "L4 opened; draw B (displaced)"),
    ("audiation", "au_s1", "E1b", "per-datum credit vs pooled anchor"),
    ("census", "cs_s0", "F0", "gate/yoke/extend + the two gifted arms"),
    ("assay", "as_s0", "F0", "the surgery six: arrival crossed with content"),
    ("assay", "as_s1", "F0", "stream-displaced twins of given_c1 / exact"),
    # bonus, different volume path from maestro's ma_s0 despite the tag-name collision
    ("intonation", "ma_s0", "bonus", "delta_perf on the maestro stack (NOT maestro/ma_s0)"),
    ("intonation", "in_s0", "bonus", "delta_perf on the crescendo stack"),
    ("caesura", "ca_s0", "bonus", "delta_silence pacing"),
    ("spiral", "sp_s0", "bonus", "the pre-A1 donor"),
]

LEVELS = (2, 3, 4)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def slot_layout(depth, s, max_level):
    off, cur = {}, 0
    for ell in range(1, max_level + 1):
        off[ell] = cur
        cur += s ** (depth - ell)
    return off, cur


def infer_max_level(n_slots, depth, s):
    for ml in range(1, depth + 1):
        _, n = slot_layout(depth, s, ml)
        if n == n_slots:
            return ml
    raise ValueError(f"no max_level with {n_slots} slots at depth {depth}, s {s}")


def load_arm(path):
    """One arm -> a dict of per-probe series plus arrival cycles. None if unusable."""
    with open(path) as f:
        d = json.load(f)
    lg = d.get("log") or {}
    probes = [q for q in (lg.get("probe") or []) if q and q.get("pi")]
    if not probes:
        return None
    cfg = d.get("config") or {}
    depth, s = int(cfg["depth"]), int(cfg["s"])
    n_slots = len(probes[0]["pi"]["mean_mass"])
    ml = infer_max_level(n_slots, depth, s)
    off, _ = slot_layout(depth, s, ml)

    cyc = np.array([q["cycle"] for q in probes], float)
    era = np.array([q.get("era", 0) for q in probes], float)
    mass, amax, navail = {}, {}, {}
    for ell in range(1, ml + 1):
        lo, hi = off[ell], off[ell] + s ** (depth - ell)
        mass[ell] = np.array([sum(q["pi"]["mean_mass"][lo:hi]) for q in probes], float)
        amax[ell] = np.array(
            [sum(q["pi"]["argmax_hist"][lo:hi]) / max(q["pi"]["n"], 1) for q in probes], float)
        navail[ell] = np.array(
            [sum(1 for x in q["pi"]["mean_mass"][lo:hi] if x > 0.0) for q in probes], float)
    # total open slots at each probe -> the uniform-over-live-action-set reference
    navail_all = np.array([sum(1 for x in q["pi"]["mean_mass"] if x > 0.0) for q in probes],
                          float)
    macro = np.array([q["pi"].get("macro_mass", float("nan")) for q in probes], float)
    top1 = np.array([q["pi"].get("top1", float("nan")) for q in probes], float)

    # arrival
    commits = {}
    for e in (d.get("events") or []):
        if e.get("kind") == "commit" and e.get("level") is not None:
            commits.setdefault(int(e["level"]), []).append(
                (int(e["cycle"]), bool(e.get("provisional", False))))
    arrival, arrival_kind = {}, {}
    voc0 = (lg.get("vocab") or [{}])[0] or {}
    for ell in range(2, ml + 1):
        if commits.get(ell):
            arrival[ell] = commits[ell][0][0]
            arrival_kind[ell] = "provisional" if commits[ell][0][1] else "commit"
        elif voc0.get(str(ell)) is not None:
            arrival[ell] = 1
            arrival_kind[ell] = "gift@c1"

    # table size per cycle (all cycles, not probe cycles)
    vocab = {ell: np.array([(v or {}).get(str(ell)) if v else None
                            for v in (lg.get("vocab") or [])], dtype=object)
             for ell in range(2, ml + 1)}

    # exposure: executions per level per cycle, from the entry-identity histogram when present
    ent = lg.get("entry")
    use = {}
    if ent and any(ent):
        for ell in range(2, ml + 1):
            col = []
            for q in ent:
                h = ((q or {}).get("hist") or {}).get("beam") or {}
                col.append(float(sum(h.get(str(ell), []) or [])))
            use[ell] = np.array(col, float)

    return dict(path=path, arm=d.get("arm"), depth=depth, s=s, max_level=ml,
                probe_cycles=cyc, probe_era=era, mass=mass, amax=amax, navail=navail,
                navail_all=navail_all,
                macro=macro, top1=top1, arrival=arrival, arrival_kind=arrival_kind,
                n_cycles=len(lg.get("cycle") or []), vocab=vocab, use=use,
                cycles=np.array(lg.get("cycle") or [], float),
                era_by_cycle=np.array(lg.get("era") or [], float))


def load_tag(node, tag):
    root = os.path.join(PRACTICE, node, "figures", tag)
    if not os.path.isdir(root):
        return None
    arms = OrderedDict()
    for a in sorted(os.listdir(root)):
        p = os.path.join(root, a, "results.json")
        if os.path.isfile(p):
            r = load_arm(p)
            if r is not None:
                arms[a] = r
    return arms or None


# --------------------------------------------------------------------------- #
# the rate statistics
# --------------------------------------------------------------------------- #

def interp_at(tau, y, t):
    """Linear interpolation of y(tau) at t; nan outside the observed window."""
    if len(tau) < 2 or t < tau[0] or t > tau[-1]:
        return float("nan")
    return float(np.interp(t, tau, y))


T_GRID_MAX = 400.0


def sat_fit(tau, y):
    """Least squares fit of m(tau) = M * (1 - exp(-tau / T)), anchored at m(0) = 0.

    T scanned on a log grid; M closed-form given T. Returns (M, T, r2, d0) with
    d0 = M / T, the initial slope. This is a descriptive summary of the curve's shape,
    not a claim that the process is exponential.

    IDENTIFIABILITY: when the observation window is short relative to the true T the
    curve is still in its linear regime and (M, T) trade off along M/T = const. Only
    d0 = M/T is identified there. The caller marks such cells `T~cens`.
    """
    tau = np.asarray(tau, float)
    y = np.asarray(y, float)
    nan = float("nan")
    if len(tau) < 4 or not np.isfinite(y).all() or y.max() <= 0:
        return nan, nan, nan, nan
    ss_tot = float(((y - y.mean()) ** 2).sum())
    best = (nan, nan, -np.inf)
    for T in np.exp(np.linspace(np.log(0.5), np.log(T_GRID_MAX), 600)):
        b = 1.0 - np.exp(-tau / T)
        den = float((b * b).sum())
        if den <= 0:
            continue
        M = float((b * y).sum() / den)
        ss_res = float(((y - M * b) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
        if r2 > best[2]:
            best = (M, float(T), r2)
    M, T, r2 = best
    return M, T, r2, (M / T if T and np.isfinite(T) else nan)


def rate_stats(rec, ell, min_points=3):
    """Post-arrival growth statistics for one (arm, level)."""
    if ell not in rec["arrival"]:
        return None
    a = rec["arrival"][ell]
    c = rec["probe_cycles"]
    m = rec["mass"][ell]
    sel = c >= a
    if sel.sum() < min_points:
        return None
    tau = c[sel] - a
    y = m[sel]
    ax = rec["amax"][ell][sel]
    nv = rec["navail"][ell][sel]

    # OLS slope over the whole post-arrival window
    A = np.vstack([tau, np.ones_like(tau)]).T
    slope, icept = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (slope * tau + icept)
    r2_lin = 1.0 - float((resid ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)

    # monotonicity
    dif = np.diff(y)
    frac_up = float((dif > 0).mean()) if len(dif) else float("nan")
    rho = float("nan")
    if len(y) >= 3:
        rt, ry = np.argsort(np.argsort(tau)), np.argsort(np.argsort(y))
        rho = float(np.corrcoef(rt, ry)[0, 1])

    m_end = float(y[-1])
    # time to half the arm's own end value, interpolated on the probe grid.
    # If the series already sits at/above half at the first post-arrival probe, the
    # answer is 0 (nothing to wait for); if the series never grows, it is undefined.
    t_half = float("nan")
    if m_end > 0 and y[0] >= 0.5 * m_end:
        t_half = 0.0                     # already at half when the level arrived
    elif m_end > 0:
        for i in range(1, len(y)):
            if y[i] >= 0.5 * m_end:
                y0, y1 = y[i - 1], y[i]
                t_half = (float(tau[i - 1] + (tau[i] - tau[i - 1])
                                * (0.5 * m_end - y0) / (y1 - y0))
                          if y1 != y0 else float(tau[i]))
                break
        else:
            t_half = float("inf")

    M, T, r2_sat, d0 = sat_fit(tau, y)
    # the fit is only interpretable where it fits and where T sits inside the window
    T_ok = bool(np.isfinite(r2_sat) and r2_sat >= 0.5 and np.isfinite(T)
                and T <= float(tau[-1]) and T < 0.9 * T_GRID_MAX)
    if not (np.isfinite(r2_sat) and r2_sat >= 0.3):
        d0 = float("nan")           # the fit failed; M/T is not a rate of anything

    # uniform-over-live-action-set reference: what this level would hold under a flat pi
    nva = rec["navail_all"][sel]
    unif = np.where(nva > 0, nv / np.maximum(nva, 1), np.nan)
    lift = np.where(unif > 0, y / unif, np.nan)

    # the shape of the lift curve: does the level get de-funded below chance first, and
    # how long until pi is merely indifferent to it (lift = 1)?
    lf = np.asarray(lift, float)
    i_min = int(np.nanargmin(lf)) if np.isfinite(lf).any() else 0
    lift_min, tau_lift_min = float(lf[i_min]), float(tau[i_min])
    # The crossing is measured AFTER the trough. An arm whose first post-arrival probe
    # lands on the commit cycle itself reads lift ~1 there (untrained logits at the moment
    # of unmasking); that is the starting point, not a crossing. The recovery time is the
    # first tau at or after the trough at which lift reaches 1 again.
    tau_cross1 = float("nan")
    for i in range(i_min, len(lf)):
        if np.isfinite(lf[i]) and lf[i] >= 1.0:
            if i == i_min:
                tau_cross1 = float(tau[i])
            else:
                y0, y1 = lf[i - 1], lf[i]
                tau_cross1 = (float(tau[i - 1] + (tau[i] - tau[i - 1])
                                    * (1.0 - y0) / (y1 - y0))
                              if y1 != y0 else float(tau[i]))
            break

    # absolute-mass crossings — the complement to `lift`, whose 1.0 threshold needs less
    # absolute mass at higher levels (L2 0.286, L3 0.143, L4 0.067 of the distribution)
    def tau_at_mass(thr):
        for i in range(len(y)):
            if y[i] >= thr:
                if i == 0:
                    return float(tau[0])
                y0, y1 = y[i - 1], y[i]
                return (float(tau[i - 1] + (tau[i] - tau[i - 1]) * (thr - y0) / (y1 - y0))
                        if y1 != y0 else float(tau[i]))
        return float("nan")

    # censoring-safe windowed slopes, from the interpolated curve
    def wslope(W):
        a0, a1 = interp_at(tau, y, 0), interp_at(tau, y, W)
        if not np.isfinite(a0):                      # arrival probe not on the grid
            a0 = float(y[0]) if tau[0] <= 0.5 * W else float("nan")
        return (a1 - a0) / W * 10.0 if np.isfinite(a0) and np.isfinite(a1) else float("nan")

    return dict(level=ell, arrival=a, kind=rec["arrival_kind"][ell],
                n_post=int(sel.sum()), tau_max=float(tau[-1]),
                m_first=float(y[0]), m_end=m_end,
                slope_per10=float(slope) * 10.0, r2_lin=r2_lin,
                d16=wslope(16), d32=wslope(32),
                frac_up=frac_up, rho=rho, t_half=t_half,
                M=M, T=T, r2_sat=r2_sat, d0=d0, T_ok=T_ok,
                amax_end=float(ax[-1]), navail_end=float(nv[-1]),
                unif_first=float(unif[0]), unif_end=float(unif[-1]),
                lift_first=float(lift[0]), lift_end=float(lift[-1]),
                lift_min=lift_min, tau_lift_min=tau_lift_min, tau_cross1=tau_cross1,
                tau_m10=tau_at_mass(0.10), tau_m20=tau_at_mass(0.20),
                mass_per_slot_end=float(m_end / nv[-1]) if nv[-1] > 0 else float("nan"),
                at={t: interp_at(tau, y, t) for t in (0, 8, 16, 24, 32, 40, 56, 72)},
                lift_at={t: interp_at(tau, lift, t) for t in (0, 8, 16, 32, 56)},
                tau=tau, y=y, ax=ax, nv=nv, lift=lift, unif=unif)


# --------------------------------------------------------------------------- #
# exposure vs credit (offline half of the control)
# --------------------------------------------------------------------------- #

def exposure_cols(rec, ell):
    """Cumulative level-l executions and cumulative solved cycles at each probe cycle."""
    if ell not in rec["use"] or not len(rec["cycles"]):
        return None
    u = rec["use"][ell]
    cum = np.cumsum(u)
    idx = np.searchsorted(rec["cycles"], rec["probe_cycles"], side="right") - 1
    idx = np.clip(idx, 0, len(cum) - 1)
    return cum[idx]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def fmt(x, w=8, p=4):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return " " * (w - 1) + "-"
    if isinstance(x, float) and math.isinf(x):
        return f"{'inf':>{w}s}"
    return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{x:>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "reduction.txt"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    data = OrderedDict()
    for node, tag, fam, note in MANIFEST:
        arms = load_tag(node, tag)
        if arms:
            data[(node, tag)] = (fam, note, arms)

    out("=" * 100)
    out("woodshed / F1 part 1 — the trust-formation-rate instrument, read off existing logs")
    out("=" * 100)
    out("pi per-level proposal mass = sum of `pi.mean_mass` over that level's (level,node)")
    out("slots (level-major layout, prop_net.slot_layout). It is a normalised distribution")
    out("over the live action set, averaged over the probe batch. Slots not in the live set")
    out("are masked to -inf and read exactly 0.0, so a pre-arrival level's 0.000 is a")
    out("construction, not a measurement. tau = probe cycle - arrival cycle.")
    out("")

    # ---------------- section 0: inventory ----------------
    out("=" * 100)
    out("(0) INVENTORY — what was read, and each arm's arrival cycles")
    out("=" * 100)
    out(f"  {'node/tag':<22s} {'arm':<18s} {'fam':<5s} {'cyc':>4s} {'prb':>4s} {'maxL':>4s}"
        f"  {'arrival (level: cycle/kind)'}")
    for (node, tag), (fam, note, arms) in data.items():
        for a, r in arms.items():
            arr = " ".join(f"L{l}:c{r['arrival'][l]}/{r['arrival_kind'][l]}"
                           for l in sorted(r["arrival"]))
            out(f"  {node + '/' + tag:<22s} {a:<18s} {fam:<5s} {r['n_cycles']:>4d} "
                f"{len(r['probe_cycles']):>4d} {r['max_level']:>4d}  {arr or '(none)'}")
    out("")
    out("  Note the tag-name collision: `maestro/ma_s0` and `intonation/ma_s0` are different")
    out("  runs on different volume paths. Both are listed above under their node.")

    # ---------------- section 1: per-arm, per-level rate table ----------------
    out("")
    out("=" * 100)
    out("(1) POST-ARRIVAL GROWTH — one row per (arm, level) with an arrival")
    out("=" * 100)
    out("  m0/mend  : mass at the first post-arrival probe / at the last probe")
    out("  u0/uend  : the UNIFORM reference — this level's open slots / all open slots, i.e.")
    out("             the mass a flat pi would put here. m0 is not zero at arrival because")
    out("             newly unmasked slots enter the softmax with untrained logits.")
    out("  lift     : mend / uend. 1.0 = pi is indifferent to the level; >1 = it favours it.")
    out("  d16/d32  : mass gained per 10 cycles over tau in [0,16] and [0,32] — the")
    out("             censoring-safe rate, since every arm is measured on the same window.")
    out("  d/10cyc  : OLS slope of mass on tau over the WHOLE post-arrival window (r2)")
    out("  up       : fraction of consecutive post-arrival probe steps that increase")
    out("  rho      : Spearman rank correlation of mass with tau")
    out("  t1/2     : cycles after arrival to reach half the arm's own final mass")
    out("  T, d0    : saturating fit m = M(1-exp(-tau/T)); d0 = M/T is the initial rate and")
    out("             is identified even when T is not. `*` marks T outside the window or a")
    out("             fit with r2 < 0.5 — read d0/d32 there, not T.")
    out("  nav      : open (level,node) slots at the last probe")
    out("  win      : tau at the last probe (the observation window; censoring matters)")
    recs = OrderedDict()
    for (node, tag), (fam, note, arms) in data.items():
        out("")
        out(f"  --- {node}/{tag}  [{fam}] {note}")
        out(f"  {'arm':<18s} {'L':>2s} {'arr':>4s} {'kind':<11s} {'m0':>7s} {'mend':>7s} "
            f"{'uend':>6s} {'lift':>5s} {'d16':>7s} {'d32':>7s} {'d/10':>7s} {'r2l':>5s} "
            f"{'up':>5s} {'rho':>6s} {'t1/2':>6s} {'T':>7s} {'d0':>7s} {'r2s':>5s} "
            f"{'nav':>4s} {'win':>4s} {'amax':>6s}")
        for a, r in arms.items():
            for ell in LEVELS:
                st = rate_stats(r, ell)
                if st is None:
                    continue
                recs[(node, tag, a, ell)] = st
                tstr = (f"{st['T']:>7.1f}" if st["T_ok"] else
                        (f"{st['T']:>6.1f}*" if np.isfinite(st["T"]) else f"{'-':>7s}"))
                out(f"  {a:<18s} {ell:>2d} {st['arrival']:>4d} {st['kind']:<11s} "
                    f"{st['m_first']:>7.4f} {st['m_end']:>7.4f} {st['unif_end']:>6.3f} "
                    f"{st['lift_end']:>5.2f} {fmt(st['d16'], 7, 4)} {fmt(st['d32'], 7, 4)} "
                    f"{st['slope_per10']:>7.4f} {st['r2_lin']:>5.2f} "
                    f"{st['frac_up']:>5.2f} {st['rho']:>6.2f} "
                    f"{fmt(st['t_half'], 6, 1)} {tstr} {fmt(st['d0'], 7, 4)} "
                    f"{fmt(st['r2_sat'], 5, 2)} {int(st['navail_end']):>4d} "
                    f"{int(st['tau_max']):>4d} {st['amax_end']:>6.3f}")

    # duplicate detection: the arc's replay discipline makes several arms bit-identical
    out("")
    out("  REPLAY GROUPS (bit-identical post-arrival mass series — the arc's fidelity")
    out("  discipline showing up in this instrument; aggregates below de-duplicate these):")
    groups = OrderedDict()
    for k, st in recs.items():
        h = (k[3], tuple(np.round(st["y"], 12)), tuple(np.round(st["tau"], 6)))
        groups.setdefault(h, []).append(k)
    ndup = 0
    for h, ks in groups.items():
        if len(ks) > 1:
            ndup += 1
            out(f"    L{h[0]}: " + " == ".join(f"{n}/{t}/{a}" for n, t, a, _ in ks))
    if not ndup:
        out("    (none)")
    canon = {ks[0] for ks in groups.values()}

    # ---------------- section 2: window-matched mass ----------------
    out("")
    out("=" * 100)
    out("(2) WINDOW-MATCHED MASS — the same clock for every arm: mass at fixed tau")
    out("=" * 100)
    out("  Blank = the arm's post-arrival window does not reach that tau (right-censored).")
    out("  One row per replay group (duplicates collapsed). Sorted by mass at tau=32.")
    taus = (0, 8, 16, 24, 32, 40, 56, 72)
    for ell in LEVELS:
        rows = [(k, v) for k, v in recs.items() if k[3] == ell and k in canon]
        if not rows:
            continue
        out("")
        out(f"  --- level {ell}   (mass; then the same rows as LIFT = mass / uniform share)")
        out(f"  {'node/tag/arm':<44s} {'kind':<11s} "
            + " ".join(f"{'t=' + str(t):>7s}" for t in taus))
        key32 = lambda kv: (-kv[1]["at"][32] if np.isfinite(kv[1]["at"][32])
                            else -kv[1]["m_end"])
        for (node, tag, a, _), st in sorted(rows, key=key32):
            key = f"{node}/{tag}/{a}"
            out(f"  {key:<44s} {st['kind']:<11s} "
                + " ".join(fmt(st["at"][t], 7, 4) for t in taus))
        out(f"  {'-- lift (x uniform) --':<44s} {'':<11s} "
            + " ".join(f"{'t=' + str(t):>7s}" for t in (0, 8, 16, 32, 56)))
        for (node, tag, a, _), st in sorted(rows, key=key32):
            key = f"{node}/{tag}/{a}"
            out(f"  {key:<44s} {st['kind']:<11s} "
                + " ".join(fmt(st["lift_at"][t], 7, 3) for t in (0, 8, 16, 32, 56)))

    # ---------------- section 2b: the calendar clock ----------------
    out("")
    out("=" * 100)
    out("(2b) THE CALENDAR CLOCK — the same table on ABSOLUTE cycle, not tau")
    out("=" * 100)
    out("  The since-arrival clock and the calendar clock are different questions. On the")
    out("  calendar an earned arm's level reads 0.000 before its commit; a gifted arm's")
    out("  reads its whole history. Both views are needed because a gifted arm's tau=0 is")
    out("  cycle 1, when the trunk is untrained, while an earned arm's tau=0 is cycle 18-129,")
    out("  when it is not: early-tau rates across the two families are not like for like.")
    cals = (16, 32, 48, 64, 80, 96, 116, 140, 162)
    for ell in LEVELS:
        rows = []
        for (node, tag), (fam, note, arms) in data.items():
            for a, r in arms.items():
                if ell > r["max_level"]:
                    continue
                c, m = r["probe_cycles"], r["mass"][ell]
                if (node, tag, a, ell) not in canon and (node, tag, a, ell) in recs:
                    continue
                rows.append((f"{node}/{tag}/{a}", r["arrival"].get(ell),
                             [interp_at(c, m, t) for t in cals]))
        if not rows:
            continue
        out("")
        out(f"  --- level {ell}")
        out(f"  {'node/tag/arm':<44s} {'arr':>4s} " + " ".join(f"{'c' + str(t):>7s}"
                                                               for t in cals))
        for name, arr, vals in sorted(rows, key=lambda z: -(z[2][6] if np.isfinite(z[2][6])
                                                            else -1)):
            out(f"  {name:<44s} {str(arr) if arr else '-':>4s} "
                + " ".join(fmt(v, 7, 4) for v in vals))

    # ---------------- section 2c: the shape of the lift curve ----------------
    out("")
    out("=" * 100)
    out("(2c) THE SHAPE — de-funding, then the crossing back to indifference")
    out("=" * 100)
    out("  lift = mass / (this level's open slots / all open slots). A newly unmasked level")
    out("  enters the softmax at lift ~= 1 (untrained logits). The columns:")
    out("    lift0    lift at the first post-arrival probe")
    out("    liftmin  the minimum lift over the post-arrival window, and the tau it occurs at")
    out("    cross1   the first tau at which lift reaches 1.0 (interpolated). This is the")
    out("             censoring-safe 'time to indifference' — pi has stopped de-funding the")
    out("             level and is proposing it at chance.")
    out("    liftend  lift at the last probe")
    for ell in LEVELS:
        rows = [(k, v) for k, v in recs.items() if k[3] == ell and k in canon]
        if not rows:
            continue
        out("")
        out(f"  --- level {ell}")
        out(f"  {'node/tag/arm':<44s} {'kind':<11s} {'lift0':>7s} {'liftmin':>8s} "
            f"{'@tau':>6s} {'cross1':>7s} {'liftend':>8s} {'win':>4s}")
        for (node, tag, a, _), st in sorted(
                rows, key=lambda kv: (kv[1]["tau_cross1"] if np.isfinite(kv[1]["tau_cross1"])
                                      else 1e9)):
            out(f"  {node + '/' + tag + '/' + a:<44s} {st['kind']:<11s} "
                f"{st['lift_first']:>7.3f} {st['lift_min']:>8.3f} "
                f"{st['tau_lift_min']:>6.0f} {fmt(st['tau_cross1'], 7, 1)} "
                f"{st['lift_end']:>8.3f} {int(st['tau_max']):>4d}")
        gi = [st for (_, _, _, _), st in rows if st["kind"] == "gift@c1"]
        ea = [st for (_, _, _, _), st in rows if st["kind"] != "gift@c1"]
        for lab, grp in (("gift@c1", gi), ("earned", ea)):
            if not grp:
                continue
            cr = [s["tau_cross1"] for s in grp if np.isfinite(s["tau_cross1"])]
            lm = [s["lift_min"] for s in grp]
            le = [s["lift_end"] for s in grp]
            out(f"    {lab:<10s} n={len(grp):>2d}  median cross1="
                f"{(f'{np.median(cr):.1f}' if cr else 'never'):>6s} "
                f"({len(cr)}/{len(grp)} ever cross)  median liftmin={np.median(lm):.3f}  "
                f"median liftend={np.median(le):.3f}")

    # ---------------- section 3: per-level ordering ----------------
    out("")
    out("=" * 100)
    out("(3) PER-LEVEL ORDERING — is the clock slower higher up?")
    out("=" * 100)
    out("  Within each arm that has arrivals at >=2 levels: T, t1/2 and slope by level.")
    for (node, tag), (fam, note, arms) in data.items():
        for a, r in arms.items():
            got = [(ell, recs[(node, tag, a, ell)]) for ell in LEVELS
                   if (node, tag, a, ell) in recs]
            if len(got) < 2:
                continue
            out(f"  {node + '/' + tag + '/' + a:<44s} "
                + "  ".join(f"L{e}: d32={fmt(s['d32'], 6, 4)} t1/2={fmt(s['t_half'], 5, 1)} "
                            f"lift={fmt(s['lift_end'], 5, 2)} win={int(s['tau_max'])}"
                            for e, s in got))
    out("")
    out("  Within-arm paired comparison (only arms with BOTH levels; sign of L3 minus L2):")
    for f_, lab in (("d32", "d32 (mass/10cyc over tau<=32)"), ("t_half", "t1/2"),
                    ("m_end", "mass at last probe"), ("lift_end", "lift at last probe")):
        pairs = []
        for (node, tag), (fam, note, arms) in data.items():
            for a in arms:
                k2, k3 = (node, tag, a, 2), (node, tag, a, 3)
                if k2 in canon and k2 in recs and k3 in recs:
                    v2, v3 = recs[k2][f_], recs[k3][f_]
                    if np.isfinite(v2) and np.isfinite(v3):
                        pairs.append((v2, v3))
        if pairs:
            arr = np.array(pairs)
            n_up = int((arr[:, 1] > arr[:, 0]).sum())
            out(f"    {lab:<34s} n={len(pairs):>2d}  L3>L2 in {n_up}/{len(pairs)}  "
                f"median L2={np.median(arr[:, 0]):.4f}  median L3={np.median(arr[:, 1]):.4f}")
    for f_, lab in (("d32", "d32"), ("t_half", "t1/2"), ("m_end", "mass_end"),
                    ("lift_end", "lift_end")):
        pairs = []
        for (node, tag), (fam, note, arms) in data.items():
            for a in arms:
                k3, k4 = (node, tag, a, 3), (node, tag, a, 4)
                if k3 in canon and k3 in recs and k4 in recs:
                    v3, v4 = recs[k3][f_], recs[k4][f_]
                    if np.isfinite(v3) and np.isfinite(v4):
                        pairs.append((v3, v4))
        if pairs:
            arr = np.array(pairs)
            n_up = int((arr[:, 1] > arr[:, 0]).sum())
            out(f"    L4 vs L3: {lab:<24s} n={len(pairs):>2d}  L4>L3 in {n_up}/{len(pairs)}  "
                f"median L3={np.median(arr[:, 0]):.4f}  median L4={np.median(arr[:, 1]):.4f}")
    out("")
    out("  Aggregate over de-duplicated (arm, level) cells, by level and arrival kind:")
    out("  cross1 = cycles from arrival back to chance-level share (after the trough).")
    out("  tau@.10 / tau@.20 = cycles to reach an ABSOLUTE mass of 0.10 / 0.20 — the")
    out("  complement to cross1, whose threshold needs less absolute mass higher up")
    out("  (uniform share is 0.286 at L2, 0.143 at L3, 0.067 at L4).")
    out(f"  {'level':>6s} {'kind':<8s} {'n':>3s} {'med d16':>9s} {'med d32':>9s} "
        f"{'med t1/2':>9s} {'med cross1':>11s} {'med t@.10':>10s} {'med t@.20':>10s} "
        f"{'med mend':>9s} {'med lift':>9s} {'med win':>8s}")
    for ell in LEVELS:
        for kind in ("earned", "gift@c1"):
            ss = [s for k, s in recs.items() if k[3] == ell and k in canon
                  and ((s["kind"] == "gift@c1") == (kind == "gift@c1"))]
            if not ss:
                continue

            def med(f):
                v = [f(s) for s in ss if np.isfinite(f(s))]
                return float(np.median(v)) if v else float("nan")
            out(f"  {ell:>6d} {kind:<8s} {len(ss):>3d} "
                f"{fmt(med(lambda s: s['d16']), 9, 4)} "
                f"{fmt(med(lambda s: s['d32']), 9, 4)} "
                f"{fmt(med(lambda s: s['t_half']), 9, 1)} "
                f"{fmt(med(lambda s: s['tau_cross1']), 11, 1)} "
                f"{fmt(med(lambda s: s['tau_m10']), 10, 1)} "
                f"{fmt(med(lambda s: s['tau_m20']), 10, 1)} "
                f"{fmt(med(lambda s: s['m_end']), 9, 4)} "
                f"{fmt(med(lambda s: s['lift_end']), 9, 2)} "
                f"{fmt(med(lambda s: s['tau_max']), 8, 1)}")

    # ---------------- section 4: arm contrasts ----------------
    out("")
    out("=" * 100)
    out("(4) NAMED CONTRASTS")
    out("=" * 100)

    def cell(node, tag, arm, ell, field):
        st = recs.get((node, tag, arm, ell))
        return st[field] if st else float("nan")

    def pair_block(title, pairs, ell):
        out("")
        out(f"  --- {title}  (level {ell})")
        out(f"  {'arm':<40s} {'arr':>4s} {'m0':>7s} {'mend':>7s} {'d16':>7s} {'d32':>7s} "
            f"{'t1/2':>6s} {'m@16':>7s} {'m@32':>7s} {'m@56':>7s} {'lift@32':>8s} "
            f"{'liftend':>8s} {'amax':>6s} {'win':>4s}")
        for node, tag, arm in pairs:
            st = recs.get((node, tag, arm, ell))
            if not st:
                out(f"  {node + '/' + tag + '/' + arm:<40s}  (no arrival at L{ell})")
                continue
            out(f"  {node + '/' + tag + '/' + arm:<40s} {st['arrival']:>4d} "
                f"{st['m_first']:>7.4f} {st['m_end']:>7.4f} {fmt(st['d16'], 7, 4)} "
                f"{fmt(st['d32'], 7, 4)} {fmt(st['t_half'], 6, 1)} "
                f"{fmt(st['at'][16], 7, 4)} {fmt(st['at'][32], 7, 4)} "
                f"{fmt(st['at'][56], 7, 4)} {fmt(st['lift_at'][32], 8, 3)} "
                f"{fmt(st['lift_end'], 8, 3)} {st['amax_end']:>6.3f} "
                f"{int(st['tau_max']):>4d}")

    pair_block("ARRIVAL: gift at c1 vs perfect content at commit time (assay)",
               [("assay", "as_s0", "given_c1"), ("assay", "as_s0", "exact"),
                ("assay", "as_s0", "complete"), ("assay", "as_s0", "anchor"),
                ("assay", "as_s0", "junk_dose"), ("assay", "as_s0", "strip"),
                ("assay", "as_s1", "given_c1_j"), ("assay", "as_s1", "exact_j"),
                ("census", "cs_s0", "given_route"), ("census", "cs_s0", "given_native"),
                ("census", "cs_s0", "spiral_route")], 3)
    pair_block("ARRIVAL, level 2", [("assay", "as_s0", "given_c1"),
                                    ("assay", "as_s0", "exact"),
                                    ("assay", "as_s0", "anchor"),
                                    ("census", "cs_s0", "given_route"),
                                    ("census", "cs_s0", "spiral_route")], 2)
    pair_block("CREDIT: per-datum vs pooled (audiation au_s1)",
               [("audiation", "au_s1", "anchor"), ("audiation", "au_s1", "perdatum")], 3)
    pair_block("CREDIT: per-datum vs pooled, level 2",
               [("audiation", "au_s1", "anchor"), ("audiation", "au_s1", "perdatum")], 2)
    pair_block("THE NEW RUNG: L4 after commit, both draws (crescendo)",
               [("crescendo", "cr3_s0", "outer_yield_m4"),
                ("crescendo", "cr3_s0", "outer_yield_m4x"),
                ("crescendo", "cr3_s0", "anchor_long"),
                ("crescendo", "cr3_s1", "outer_yield_m4_j"),
                ("intonation", "in_s0", "perf_gain"),
                ("intonation", "ma_s0", "mperf_gain"),
                ("caesura", "ca_s0", "dsil_yield")], 4)
    pair_block("PACING: how the commit was paced (A1/A2/A3 L3)",
               [("conductor", "cd_s0", "anchor"), ("conductor", "cd_s0", "outer_yield"),
                ("conductor", "cd_s0", "yoked_yield"),
                ("maestro", "ma_s0", "learned_yield"),
                ("maestro", "ma_s0", "yoked_learned"),
                ("maestro", "ma_s0", "outer_yield"),
                ("crescendo", "cr3_s0", "outer_yield_m4"),
                ("crescendo", "cr3_s0", "ceiling_m3"),
                ("census", "cs_s0", "census_extend"),
                ("census", "cs_s0", "census_gate")], 3)

    # ---------------- section 5: exposure vs credit ----------------
    out("")
    out("=" * 100)
    out("(5) EXPOSURE vs CLOCK — does mass track cycles or executions?")
    out("=" * 100)
    out("  Cumulative level-l executions at each probe come from `log.entry.hist.beam[l]`")
    out("  (the per-entry execution histogram of the beam/solve path). For each (arm, level)")
    out("  we regress post-arrival mass on tau and on log10(1+cumulative executions) and")
    out("  report both r2, plus the partial correlation of mass with executions given tau.")
    out(f"  {'node/tag/arm':<44s} {'L':>2s} {'r2(tau)':>8s} {'r2(logU)':>9s} "
        f"{'r(m,U|tau)':>11s} {'U_end':>10s}")
    exp_rows = []
    for (node, tag), (fam, note, arms) in data.items():
        for a, r in arms.items():
            for ell in LEVELS:
                st = recs.get((node, tag, a, ell))
                if st is None:
                    continue
                cu = exposure_cols(r, ell)
                if cu is None:
                    continue
                sel = r["probe_cycles"] >= st["arrival"]
                u = cu[sel]
                if len(u) < 4 or u[-1] <= 0:
                    continue
                tau, y = st["tau"], st["y"]
                lu = np.log10(1.0 + u)

                def r2_of(x):
                    A = np.vstack([x, np.ones_like(x)]).T
                    b = np.linalg.lstsq(A, y, rcond=None)[0]
                    res = y - A @ b
                    tot = float(((y - y.mean()) ** 2).sum())
                    return 1.0 - float((res ** 2).sum()) / max(tot, 1e-12)

                def resid(x, z):
                    A = np.vstack([z, np.ones_like(z)]).T
                    b = np.linalg.lstsq(A, x, rcond=None)[0]
                    return x - A @ b

                ry, ru = resid(y, tau), resid(lu, tau)
                pc = (float(np.corrcoef(ry, ru)[0, 1])
                      if ry.std() > 1e-12 and ru.std() > 1e-12 else float("nan"))
                exp_rows.append((node, tag, a, ell, r2_of(tau), r2_of(lu), pc, float(u[-1])))
    for node, tag, a, ell, r2t, r2u, pc, uend in exp_rows:
        out(f"  {node + '/' + tag + '/' + a:<44s} {ell:>2d} {r2t:>8.3f} {r2u:>9.3f} "
            f"{fmt(pc, 11, 3)} {uend:>10.0f}")
    if exp_rows:
        out("")
        arr = np.array([[r[4], r[5], r[6]] for r in exp_rows], float)
        out(f"  medians over {len(exp_rows)} cells: r2(tau)={np.nanmedian(arr[:, 0]):.3f}  "
            f"r2(logU)={np.nanmedian(arr[:, 1]):.3f}  "
            f"partial r(m,U|tau)={np.nanmedian(arr[:, 2]):+.3f}")
        out("  (tau and cumulative executions are near-collinear post-arrival, so the levels")
        out("   regression cannot separate them. The increments below can, a little.)")

    # increments: does a probe interval with MORE executions of level l gain more mass?
    out("")
    out("  INCREMENTS — between consecutive probes, mass gained per cycle vs executions of")
    out("  that level per cycle over the same interval. r_raw is their correlation; r|tau")
    out("  partials out the interval's midpoint tau (mass gain decelerates with tau, and")
    out("  so does everything else, which is the whole confound).")
    out(f"  {'node/tag/arm':<44s} {'L':>2s} {'n':>3s} {'r_raw':>7s} {'r|tau':>7s} "
        f"{'Urate_med':>10s}")
    inc_rows = []
    for (node, tag), (fam, note, arms) in data.items():
        for a, r in arms.items():
            for ell in LEVELS:
                st = recs.get((node, tag, a, ell))
                if st is None or (node, tag, a, ell) not in canon:
                    continue
                if ell not in r["use"] or not len(r["cycles"]):
                    continue
                sel = r["probe_cycles"] >= st["arrival"]
                pc = r["probe_cycles"][sel]
                if len(pc) < 5:
                    continue
                u_per = []
                for i in range(1, len(pc)):
                    m_ = (r["cycles"] > pc[i - 1]) & (r["cycles"] <= pc[i])
                    dc = max(pc[i] - pc[i - 1], 1.0)
                    u_per.append(float(r["use"][ell][m_].sum()) / dc)
                u_per = np.array(u_per, float)
                dm = np.diff(st["y"]) / np.maximum(np.diff(st["tau"]), 1.0)
                mid = 0.5 * (st["tau"][1:] + st["tau"][:-1])
                if u_per.std() < 1e-9 or dm.std() < 1e-12:
                    continue

                def resid(x, z):
                    A = np.vstack([z, np.ones_like(z)]).T
                    b = np.linalg.lstsq(A, x, rcond=None)[0]
                    return x - A @ b
                r_raw = float(np.corrcoef(dm, u_per)[0, 1])
                ry, ru = resid(dm, mid), resid(u_per, mid)
                r_p = (float(np.corrcoef(ry, ru)[0, 1])
                       if ry.std() > 1e-12 and ru.std() > 1e-12 else float("nan"))
                inc_rows.append((node, tag, a, ell, len(dm), r_raw, r_p,
                                 float(np.median(u_per))))
    for node, tag, a, ell, n, rr, rp, um in inc_rows:
        out(f"  {node + '/' + tag + '/' + a:<44s} {ell:>2d} {n:>3d} {rr:>7.3f} "
            f"{fmt(rp, 7, 3)} {um:>10.0f}")
    if inc_rows:
        arr = np.array([[r[5], r[6]] for r in inc_rows], float)
        for ell in LEVELS:
            sub = np.array([[r[5], r[6]] for r in inc_rows if r[3] == ell], float)
            if len(sub):
                out(f"    L{ell}: n={len(sub):>3d}  median r_raw={np.nanmedian(sub[:, 0]):+.3f}"
                    f"  median r|tau={np.nanmedian(sub[:, 1]):+.3f}  "
                    f"positive r|tau in {int(np.nansum(sub[:, 1] > 0))}/{len(sub)}")
        out(f"    ALL: n={len(arr):>3d}  median r_raw={np.nanmedian(arr[:, 0]):+.3f}  "
            f"median r|tau={np.nanmedian(arr[:, 1]):+.3f}  "
            f"positive r|tau in {int(np.nansum(arr[:, 1] > 0))}/{len(arr)}")

    # ---------------- section 6: instrument caveats ----------------
    out("")
    out("=" * 100)
    out("(6) INSTRUMENT — what this readout can and cannot resolve")
    out("=" * 100)
    for i, t in enumerate([
        "Probe cadence is `probe_every` = 8 cycles, plus the first and last cycle of every "
        "era, so any time constant below ~8 cycles is at the sampling limit and T/t1/2 for "
        "such cells is reported but not resolved. There is no per-cycle pi mass series in "
        "`results.json`; the only per-cycle pi artifact in the arc is audiation's "
        "`snapshots/probe_trace.npz` (raw, unmasked logits), which lives on the volume and "
        "is not in the fetched mirror.",
        "m0 is NOT zero. A level's slots are masked out of the softmax until its table is "
        "committed, and they re-enter carrying whatever the untrained logits give — near "
        "the uniform share of the live action set. So the number to read is the rise above "
        "the uniform reference (`lift`), not the raw mass; `m0` differences across arms are "
        "partly the head's arbitrary state at the moment of unmasking.",
        "Right-censoring is severe and uneven: L4 arrivals leave 21-46 cycles of window "
        "while L2 arrivals leave 60-180. `M` (the saturating asymptote) is extrapolation "
        "wherever `win` is short relative to `T`; `m@tau` at matched tau is the "
        "censoring-safe comparison and is why section (2) exists.",
        "Mass is a normalised distribution over the live action set: L1's 32 primitive "
        "slots always compete, and every level's mass is jointly constrained. A rise at "
        "L3 is arithmetically a fall somewhere else.",
        "A level's mass is spread over only the (level,node) slots that hold a table "
        "entry, and extension/recert opens more slots mid-run. `navail` and `m/slot` "
        "separate the two, but the split is only read at the last probe here.",
        "Arrival for gifted arms is cycle 1 by construction, which is also the cycle the "
        "probe series starts, so their m0 is the untrained head's mass, whereas an earned "
        "arm's m0 is measured after 18-129 cycles of unrelated training. The two m0s are "
        "not the same kind of number; m@tau is.",
        "Single seed everywhere. The stream-displaced twins (`ma_s1`, `cr3_s1`, `as_s1`) "
        "bound draw-luck only. Census finding 7's floors apply to error-rate readings, "
        "not directly to mass; no floor for mass has been measured, and the displaced "
        "pairs here are the only handle on one.",
        "The exposure column exists only where `entry` recording was on; census's gifted "
        "arms have no `entry` and are absent from section (5).",
    ]):
        out(f"  {i + 1}. {t}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote {args.out}]")

    if args.figures:
        make_figures(data, recs)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(data, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)

    # fig 1: post-arrival curves, mass (top) and lift (bottom), one column per level
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    cmap = plt.get_cmap("tab20")
    nodes = sorted({k[0] for k in recs})
    col = {n: cmap(i % 20) for i, n in enumerate(nodes)}
    for j, ell in enumerate(LEVELS):
        for row, (field, lab) in enumerate((("y", "pi per-level proposal mass"),
                                            ("lift", "lift = mass / uniform share"))):
            ax = axes[row][j]
            for (node, tag, a, e), st in recs.items():
                if e != ell:
                    continue
                ls = "--" if st["kind"] == "gift@c1" else "-"
                ax.plot(st["tau"], st[field], ls, color=col[node], lw=1.3, alpha=0.85,
                        marker="o", ms=2.5)
            if row == 1:
                ax.axhline(1.0, color="k", lw=1.0, ls=":")
                ax.set_ylim(0, 6)
            ax.set_title(f"level {ell}: {lab.split(' =')[0]} vs cycles since arrival")
            ax.set_xlabel("tau = cycle - arrival")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], color=col[n], label=n) for n in nodes]
    handles += [plt.Line2D([], [], color="k", ls="--", label="gift @ c1"),
                plt.Line2D([], [], color="k", ls="-", label="earned commit"),
                plt.Line2D([], [], color="k", ls=":", label="lift = 1 (chance)")]
    axes[0][-1].legend(handles=handles, fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_growth_by_level.png"), dpi=150)
    plt.close(fig)

    # fig 2: the named contrasts
    panels = [
        ("arrival (assay L3)", 3,
         [("assay", "as_s0", "given_c1"), ("assay", "as_s0", "exact"),
          ("assay", "as_s0", "complete"), ("assay", "as_s0", "anchor"),
          ("assay", "as_s0", "junk_dose")]),
        ("credit (audiation L3)", 3,
         [("audiation", "au_s1", "anchor"), ("audiation", "au_s1", "perdatum")]),
        ("the new rung (L4)", 4,
         [("crescendo", "cr3_s0", "outer_yield_m4"),
          ("crescendo", "cr3_s0", "outer_yield_m4x"),
          ("crescendo", "cr3_s1", "outer_yield_m4_j"),
          ("crescendo", "cr3_s0", "anchor_long"),
          ("caesura", "ca_s0", "dsil_yield")]),
        ("pacing (A1/A2/A3 L3)", 3,
         [("conductor", "cd_s0", "outer_yield"), ("maestro", "ma_s0", "outer_yield"),
          ("maestro", "ma_s0", "learned_yield"),
          ("crescendo", "cr3_s0", "outer_yield_m4"),
          ("census", "cs_s0", "census_extend")]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, (title, ell, keys) in zip(axes, panels):
        for node, tag, a in keys:
            st = recs.get((node, tag, a, ell))
            if not st:
                continue
            ls = "--" if st["kind"] == "gift@c1" else "-"
            ax.plot(st["tau"], st["y"], ls, lw=1.6, marker="o", ms=3,
                    label=f"{tag}/{a} (c{st['arrival']})")
        ax.set_title(f"{title}")
        ax.set_xlabel("tau")
        ax.set_ylabel(f"pi L{ell} mass")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_contrasts.png"), dpi=150)
    plt.close(fig)

    # fig 3: the rate statistics by level, gift vs earned
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, field, lab, logy in (
            (axes[0], "tau_cross1", "cycles from arrival back to chance share", False),
            (axes[1], "t_half", "cycles to half final mass", True),
            (axes[2], "lift_end", "lift at the last probe", False)):
        for kind, mk, c in (("gift@c1", "D", "tab:orange"), ("earned", "o", "tab:blue")):
            xs, ys = [], []
            for (node, tag, a, e), st in recs.items():
                if ((st["kind"] == "gift@c1") != (kind == "gift@c1")):
                    continue
                v = st[field]
                if np.isfinite(v) and (not logy or v > 0):
                    xs.append(e + np.random.uniform(-0.14, 0.14))
                    ys.append(v)
            ax.scatter(xs, ys, s=18, alpha=0.75, marker=mk, color=c, label=kind)
        ax.set_xticks(list(LEVELS))
        ax.set_xlabel("level")
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale("log")
        if field == "lift_end":
            ax.axhline(1.0, color="k", lw=1.0, ls=":")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_rate_by_level.png"), dpi=150)
    plt.close(fig)
    print(f"[wrote 3 figures to {OUT}]")


if __name__ == "__main__":
    main()
