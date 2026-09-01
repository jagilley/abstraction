"""tuning/analyze_wledger — reduce the weight-ledger typer check to figures + JSON + text.

Consumes `figures/<tag>/wledger.json` (produced by `wledger.py` off the banked Gate-1
checkpoints). NO training, no replay, no counterfactual pass, no second reader.

THE INSTRUMENT, stated once.

  read        Per-layer-group relative weight change over a trailing window,
              rel_L(a,b) = ||theta_L(b) - theta_L(a)|| / ||theta_L(a)||, contracted into
              three scale-free log-ratios BETWEEN groups inside one interval:

                  c_emb  = log2( rel_wte               / gmean(rel_h0..h7) )
                  c_gain = log2( rel_ln_f              / gmean(rel_h0..h7) )
                  c_comp = log2( gmean(rel_h1, rel_h2) / gmean(rel_h3..h7) )

              Gate 1G's disjoint-subspace result names the three groups (rotation: wte;
              burst: ln_f; drift: h1-h2). Nothing here needs a counterfactual evaluation,
              a second reader or an oracle -- only quantities a learner already has.

  null        Each contrast is n-dependent (the groups have different quiet scaling
              exponents), so the baseline is a per-arm linear model of the contrast in
              log n, fitted on that arm's OWN quiet intervals, and the read is the
              standardised residual  z = (contrast - baseline(n)) / sigma_quiet.
              Online this is an EWMA of the arm's own per-group update norms -- the same
              self-referential idiom Gate 1's op maps used on surprisal.

              Quiet intervals: gate1g's `quiet_pairs` admission rule verbatim (disjoint
              from every event's guard), MINUS the pre-maturity warm-up (`--warmup`,
              default = the first event onset). That exclusion is not cosmetic: the
              0->2000 interval is the initialisation transient (rel_wte 0.61 against a
              mature 0.04) and it alone carried a +10.3 sigma quiet outlier on c_gain.
              Gate 1's own idiom is in-tag MATURE floors; this is that, on the weight side.
              Quiet intervals are scored leave-one-out.

  floors      (i) sigma_quiet, the unit z is in; (ii) the arm's empirical quiet max|z| and
              p90|z|, which is the honest firing threshold; (iii) an EVENT-MATCHED control
              for the rotation column that no gauge in this node had before: two arms whose
              wall never rotates (`pair_parse`, kind="none", rot_period=0; `rekey_dead`,
              kind="dead", rot_period=0) live in the same world on the same steps through
              the same bursts and drifts, but the rotating token is never in their input.
              Their c_emb at a rotation onset is what "no rotation news" reads.

  columns     Primary: the three contrasts (argmax, and argmax gated at the arm's own quiet
              max|z|). Secondary, both cheap and both requiring one labelled example of the
              type (so: available online only after the first event of a class):
                * profile-template — cosine between the standardised 12-group log-profile
                  and the profile of the first event of each type;
                * directional — cosine between the raw flat dtheta of the event's own group
                  and the same group's dtheta at the first event of that type. This is
                  Gate 1G's burst x burst ln_f alignment (0.957) turned into a typer column.

  CAVEATS, load-bearing.
    * A checkpoint delta over n steps is ||sum_t g_t||, not sum_t ||g_t||: an online
      per-group grad-norm meter reads the latter. The per-group quiet scaling exponents
      reported below measure the gap (0.5 = random walk, 1.0 = coherent).
    * The stride quantises lag. Banked: rotation {onset,+125}; burst {onset,+25},
      {onset,+125}; drift {onset,+125} (wave 1) and {onset,+25} (wave 2's micro-grid).
      Gate 1's op maps fire at +50...+125 steps.
    * Quiet intervals admit absorption tails (gate1g's documented choice), so every event
      excess is a lower bound -- and, symmetrically, the quiet floors are upper bounds.
    * Arms that fired a `skip` have dtheta EXACTLY zero over the skipped window. The ledger
      is silent precisely where the arm declined to learn (finding 7's asymmetry, on the
      weight side). Those events are reported as SKIPPED, never as misses.

Usage:  python3 rhm/practice/tuning/analyze_wledger.py [--tag wl0] [--warmup 5000]
"""

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

LAYERS = ["wte", "wpe"] + [f"h{i}" for i in range(8)] + ["ln_f", "lm_head"]
BLK = [f"h{i}" for i in range(8)]
COMP = ["h1", "h2"]
LATE = ["h3", "h4", "h5", "h6", "h7"]
AXES = ["c_emb", "c_gain", "c_comp"]
AXIS_OF = {"rotation": "c_emb", "burst": "c_gain", "drift": "c_comp"}
KIND_OF_AXIS = {v: k for k, v in AXIS_OF.items()}
KINDS = ["rotation", "burst", "drift"]
GRP_OF = {"rotation": "wte", "burst": "ln_f", "drift": "h1"}
COL = {"rotation": "#c0392b", "burst": "#2980b9", "drift": "#27ae60"}
ROTS = [8000, 10000, 12000, 14000, 16000, 18000]
BURSTS = [5000, 11000, 17000]
DRIFTS = [6000, 13000, 19000]
ONSETS = {"rotation": ROTS, "burst": BURSTS, "drift": DRIFTS}

# --- arm roles (from ARM_SPECS in gate1_lm/gate2_lm + the README's merge ledger) --------
PRIMARY = "g1a:track"
# consumes all three event types, never merges, never fires an op: a pure absorption record
ABSORBERS = ["g1a:track", "g2a:self_rekey"]
# wall never rotates -> a rotation is NOT an event for these; the event-matched null
ROT_BLIND = {"g1a:pair_parse": 'kind="none", rot_period=0',
             "g2a:rekey_dead": 'kind="dead", rot_period=0'}
# past its merge an arm no longer consumes the rotating wall: rotations stop being news
MERGE_AT = {"g1a:self": 10050, "g1a:pair_key": 10050, "g2a:self_v2": 8050,
            "g2c:self_fm": 6025}


def gm(rel, ls):
    return float(np.exp(np.mean([np.log(max(rel[l], 1e-30)) for l in ls])))


def contrasts(rel):
    b = gm(rel, BLK)
    return {"c_emb": float(np.log2(max(rel["wte"], 1e-30) / b)),
            "c_gain": float(np.log2(max(rel["ln_f"], 1e-30) / b)),
            "c_comp": float(np.log2(gm(rel, COMP) / gm(rel, LATE)))}


def dead(r):
    """True if the arm did not move at all over this interval -- it fired a `skip`."""
    return min(r["rel"].values()) <= 0.0


def linfit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ co
    sd = float(res.std(ddof=2)) if len(y) > 2 else float("nan")
    return co, sd, 1.0 - float(res.var()) / max(float(y.var()), 1e-12)


def calibrate(arm, warmup):
    """The per-arm quiet baseline for the three contrasts and the twelve groups."""
    q = [r for r in arm["intervals"]
         if r["quiet"] and r["a"] >= warmup and not dead(r)]
    x = np.log(np.array([r["n"] for r in q], float))
    C = {k: np.array([contrasts(r["rel"])[k] for r in q]) for k in AXES}
    G = {L: np.log(np.array([max(r["rel"][L], 1e-30) for r in q], float))
         for L in LAYERS}
    fit, floor = {}, {}
    for k in AXES:
        co, sd, r2 = linfit(x, C[k])
        fit[k] = {"intercept": float(co[0]), "slope": float(co[1]), "sigma": sd,
                  "r2": float(r2), "n_quiet": len(q)}
        loo = []
        for i in range(len(q)):
            m = np.ones(len(q), bool)
            m[i] = False
            co2, sd2, _ = linfit(x[m], C[k][m])
            loo.append(float((C[k][i] - (co2[0] + co2[1] * x[i])) / max(sd2, 1e-12)))
        worst = int(np.argmax(np.abs(loo)))
        floor[k] = {"max_abs": float(np.max(np.abs(loo))),
                    "p90_abs": float(np.percentile(np.abs(loo), 90)),
                    "worst_interval": f"{q[worst]['a']}_{q[worst]['b']}",
                    "n": len(q), "loo_z": [round(v, 3) for v in loo]}
    gfit = {}
    for L in LAYERS:
        co, sd, r2 = linfit(x, G[L])
        gfit[L] = {"alpha": float(co[1]), "intercept": float(co[0]), "sigma": sd,
                   "r2": float(r2)}
    return fit, floor, gfit, [(r["a"], r["b"], r["n"]) for r in q]


def zof(fit, r):
    c = contrasts(r["rel"])
    ln = np.log(r["n"])
    return {k: float((c[k] - (fit[k]["intercept"] + fit[k]["slope"] * ln))
                     / max(fit[k]["sigma"], 1e-12)) for k in AXES}


def profile(gfit, r):
    """standardised 12-group log-profile of one interval."""
    ln = np.log(r["n"])
    return np.array([(np.log(max(r["rel"][L], 1e-30))
                      - (gfit[L]["intercept"] + gfit[L]["alpha"] * ln))
                     / max(gfit[L]["sigma"], 1e-12) for L in LAYERS])


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def event_rows(arm, fit, gfit, name):
    """Every lag-0 window anchored on an event onset, with all columns."""
    blind = name in ROT_BLIND
    merged = MERGE_AT.get(name)

    def typable(r):
        if r["kind"] != "rotation":
            return True
        if blind:
            return False
        return merged is None or r["onset"] < merged

    ev = [r for r in arm["intervals"] if not r["quiet"] and r["lag"] == 0]
    tmpl = {}
    for r in sorted(ev, key=lambda r: r["onset"]):
        if r["n"] == 125 and not dead(r) and r["kind"] not in tmpl:
            tmpl[r["kind"]] = (profile(gfit, r), (r["a"], r["b"]))
    out = []
    for r in sorted(ev, key=lambda r: (r["onset"], r["n"])):
        ax = AXIS_OF[r["kind"]]
        if dead(r):
            out.append({"kind": r["kind"], "onset": r["onset"], "n": r["n"],
                        "dead": True, "axis": ax, "typable": typable(r)})
            continue
        z = zof(fit, r)
        p = profile(gfit, r)
        pc = {k: cosv(p, t[0]) for k, (t, _) in
              ((k, (v, None)) for k, v in tmpl.items())}
        is_t = any((r["a"], r["b"]) == v[1] for v in tmpl.values())
        out.append({"kind": r["kind"], "onset": r["onset"], "n": r["n"], "dead": False,
                    "axis": ax, "z": z, "z_axis": z[ax],
                    "margin": z[ax] - max(z[k] for k in AXES if k != ax),
                    "pred": max(AXES, key=lambda k: z[k]),
                    "mult_at_axis": float(2 ** (z[ax] * fit[ax]["sigma"])),
                    "profile_cos": pc, "profile_pred": max(pc, key=pc.get),
                    "is_template": bool(is_t), "typable": typable(r)})
    return out, {k: list(v[1]) for k, v in tmpl.items()}


def typer(rows, floor, sel):
    """Detection is scored only on events the arm actually experienced. A rotation is not
    an event for an arm whose wall never rotates (`ROT_BLIND`) or that merged the wall away
    before the onset (`MERGE_AT`); those onsets are scored the other way round, as
    SPECIFICITY -- the read should stay under floor there."""
    live = [e for e in rows if sel(e) and not e["dead"]]
    dead_ = [e for e in rows if sel(e) and e["dead"]]
    thr = {k: floor[k]["max_abs"] for k in AXES}

    def fires(e):
        return max(e["z"][k] - thr[k] for k in AXES) > 0

    real = [e for e in live if e["typable"]]
    non = [e for e in live if not e["typable"]]
    fired = [e for e in real if fires(e)]
    ok = [e for e in fired if e["pred"] == e["axis"]]
    fp = [e for e in non if fires(e)]
    pt = [e for e in real if not e["is_template"]]
    return {"n_events": len(live) + len(dead_), "n_skipped": len(dead_),
            "n_live": len(live), "n_real": len(real), "n_non_events": len(non),
            "argmax_correct": sum(1 for e in real if e["pred"] == e["axis"]),
            "argmax_accuracy": (sum(1 for e in real if e["pred"] == e["axis"]) / len(real)
                                if real else None),
            "min_margin": float(min(e["margin"] for e in real)) if real else None,
            "gated_fired": len(fired), "gated_correct": len(ok),
            "gated_missed_fire": [f"{e['kind']}@{e['onset']}"
                                  for e in real if e not in fired],
            "false_fires": [f"{e['kind']}@{e['onset']}->{e['pred'][2:]}" for e in fp],
            "n_false_fires": len(fp),
            "profile_correct": sum(1 for e in pt if e["profile_pred"] == e["kind"]),
            "profile_n": len(pt),
            "rows": [{"kind": e["kind"], "onset": e["onset"], "n": e["n"],
                      "dead": e["dead"],
                      **({} if e["dead"] else
                         {"z": e["z"], "z_axis": e["z_axis"], "margin": e["margin"],
                          "pred": e["pred"], "mult": e["mult_at_axis"],
                          "typable": e["typable"], "fires": fires(e)})}
                     for e in rows if sel(e)]}


def directional(arm, warmup):
    """cos(dtheta_group, dtheta_group at the FIRST event of that type)."""
    cos, fl = arm["cos"], arm["cos_floor"]

    def key(i, j):
        a = f"{i[0]}_{i[1]}|{j[0]}_{j[1]}"
        b = f"{j[0]}_{j[1]}|{i[0]}_{i[1]}"
        return a if a in cos else (b if b in cos else None)

    iv = {(r["a"], r["b"]): r for r in arm["intervals"]}
    ref = {}
    for kd in KINDS:
        e = ONSETS[kd][0]
        if (e, e + 125) in iv and not dead(iv[(e, e + 125)]):
            ref[kd] = (e, e + 125)
    out = {"reference": {k: list(v) for k, v in ref.items()},
           "floor_random_p99": {L: fl[L]["measured_p99"] for L in fl},
           "same_type": {}, "cross_type": {}, "quiet": {}}
    for kd, rk in ref.items():
        L = GRP_OF[kd]
        same, cross, quiet = [], [], []
        for (a, b), r in sorted(iv.items()):
            if (a, b) == rk or dead(r) or a < warmup:
                continue
            k = key((a, b), rk)
            if not k or L not in cos[k]:
                continue
            c = cos[k][L]
            if r["quiet"]:
                quiet.append({"a": a, "n": r["n"], "cos": c})
            elif r["lag"] == 0 and r["n"] == 125:
                (same if r["kind"] == kd else cross).append(
                    {"a": a, "kind": r["kind"], "cos": c})
        out["same_type"][kd] = same
        out["cross_type"][kd] = cross
        out["quiet"][kd] = quiet
        qa = np.abs([d["cos"] for d in quiet]) if quiet else np.array([])
        out.setdefault("summary", {})[kd] = {
            "group": L, "dim": fl[L]["d"],
            "same_mean": float(np.mean([d["cos"] for d in same])) if same else None,
            "same_min": float(np.min([d["cos"] for d in same])) if same else None,
            "cross_max_abs": float(np.max(np.abs([d["cos"] for d in cross])))
                             if cross else None,
            "quiet_max_abs": float(qa.max()) if qa.size else None,
            "quiet_p90_abs": float(np.percentile(qa, 90)) if qa.size else None,
            "quiet_mean_abs": float(qa.mean()) if qa.size else None,
            "margin_over_quiet_max": (float(np.min([d["cos"] for d in same]) - qa.max())
                                      if same and qa.size else None),
            "random_p99": fl[L]["measured_p99"], "n_same": len(same),
            "n_quiet": len(quiet)}
    return out


# --------------------------------------------------------------------------- #

def reduce_all(D, warmup):
    red = {"warmup": warmup, "arms": {}}
    for name, arm in D["arms"].items():
        fit, floor, gfit, quiet = calibrate(arm, warmup)
        rows, tmpl = event_rows(arm, fit, gfit, name)
        rec = {"role": ("absorber" if name in ABSORBERS else
                        "rotation-blind control" if name in ROT_BLIND else
                        f"merged at {MERGE_AT.get(name)}"),
               "fit": fit, "floor": floor, "group_fit": gfit,
               "quiet_intervals": quiet, "events": rows, "templates": tmpl,
               "directional": directional(arm, warmup),
               "merge_at": MERGE_AT.get(name),
               "n_skipped_intervals": sum(1 for r in arm["intervals"] if dead(r))}
        rec["typer_n125"] = typer(rows, floor, lambda e: e["n"] == 125)

        def fastest(e, rows=rows):
            same = [r for r in rows if r["kind"] == e["kind"] and r["onset"] == e["onset"]]
            return e["n"] == min(r["n"] for r in same)
        rec["typer_fastest"] = typer(rows, floor, fastest)
        red["arms"][name] = rec

    # ---- sign/rank consistency across event types (the finding-1 discipline) ----
    cons = {}
    for name in ABSORBERS + list(ROT_BLIND):
        if name not in red["arms"]:
            continue
        ev = [e for e in red["arms"][name]["events"]
              if e["n"] == 125 and not e["dead"] and e["typable"]]
        tot = ok = 0
        detail = {}
        for ax in AXES:
            kd = KIND_OF_AXIS[ax]
            mine = [e for e in ev if e["kind"] == kd]
            oth = [e for e in ev if e["kind"] != kd]
            p = [a["z"][ax] > b["z"][ax] for a in mine for b in oth]
            detail[ax] = {"n_pairs": len(p), "n_consistent": int(sum(p)),
                          "min_own": float(min(e["z"][ax] for e in mine)) if mine else None,
                          "max_other": float(max(e["z"][ax] for e in oth)) if oth else None}
            tot += len(p)
            ok += int(sum(p))
        cons[name] = {"pairwise_total": tot, "pairwise_consistent": ok,
                      "frac": ok / max(tot, 1), "by_axis": detail}
    red["sign_consistency"] = cons

    # ---- longitudinal: does the weight ledger drain across the run? ----
    lg = {}
    for name in list(red["arms"]):
        ev = red["arms"][name]["events"]
        d = {}
        for kd in KINDS:
            s = {}
            for e in ONSETS[kd]:
                m = [x for x in ev if x["onset"] == e and x["kind"] == kd
                     and x["n"] == 125]
                if m:
                    s[str(e)] = ({"dead": True} if m[0]["dead"] else
                                 {"z": m[0]["z_axis"], "mult": m[0]["mult_at_axis"],
                                  "dead": False})
            d[kd] = s
            live = [v["z"] for v in s.values() if not v["dead"]]
            if len(live) >= 2:
                d[kd + "_first_last"] = {"first": live[0], "last": live[-1],
                                         "ratio": live[-1] / live[0] if live[0] else None}
        lg[name] = d
    red["longitudinal"] = lg

    # ---- the event-matched rotation control ----
    ctrl = {"blind_arms": ROT_BLIND, "rows": []}
    if PRIMARY in red["arms"]:
        t = {e["onset"]: e for e in red["arms"][PRIMARY]["events"]
             if e["n"] == 125 and not e["dead"]}
        for kd in KINDS:
            for e in ONSETS[kd]:
                if e not in t:
                    continue
                row = {"kind": kd, "onset": e, "axis": AXIS_OF[kd],
                       PRIMARY: t[e]["z"][AXIS_OF[kd]]}
                for b in ROT_BLIND:
                    if b not in red["arms"]:
                        continue
                    m = [x for x in red["arms"][b]["events"]
                         if x["onset"] == e and x["n"] == 125 and not x["dead"]]
                    row[b] = m[0]["z"][AXIS_OF[kd]] if m else None
                ctrl["rows"].append(row)
    red["event_matched_control"] = ctrl

    # ---- lag response on the primary arm ----
    red["lag_response"] = {}
    for name in ABSORBERS:
        if name not in red["arms"]:
            continue
        A = D["arms"][name]
        fit = red["arms"][name]["fit"]
        rr = []
        for r in A["intervals"]:
            if r["kind"] is None or dead(r):
                continue
            rr.append({"kind": r["kind"], "onset": r["onset"], "lag": r["lag"],
                       "n": r["n"], "z_axis": zof(fit, r)[AXIS_OF[r["kind"]]],
                       "from_null_set": False})
        # a window starting at onset+125 may be admissible as QUIET under gate1g's rule
        # (its guard has expired) and therefore sit inside the null. Score it
        # leave-one-out and report it as the event's own +125 tail datum -- for a rotation
        # this is the only banked lag>0 window there is.
        qi = red["arms"][name]["quiet_intervals"]
        for kd in KINDS:
            for e in ONSETS[kd]:
                for i, (a, b, n) in enumerate(qi):
                    if a == e + 125:
                        rr.append({"kind": kd, "onset": e, "lag": 125, "n": n,
                                   "z_axis": red["arms"][name]["floor"][
                                       AXIS_OF[kd]]["loo_z"][i],
                                   "from_null_set": True})
        red["lag_response"][name] = sorted(rr, key=lambda d: (d["kind"], d["onset"],
                                                              d["lag"], d["n"]))
    return red


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _floorline(ax, red, arm, axis, horiz):
    f = red["arms"][arm]["floor"][axis]["max_abs"]
    fn = (ax.axhline if horiz else ax.axvline)
    fn(f, color="grey", ls=":", lw=1.2)
    fn(-f, color="grey", ls=":", lw=1.2)


def fig_corners(red, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    arms = [a for a in ABSORBERS + list(ROT_BLIND) if a in red["arms"]]
    mk = {"g1a:track": "o", "g2a:self_rekey": "v",
          "g1a:pair_parse": "^", "g2a:rekey_dead": "s"}
    for ax, (xa, ya) in zip(axes, [("c_emb", "c_gain"), ("c_emb", "c_comp"),
                                   ("c_gain", "c_comp")]):
        for a in arms:
            blind = a in ROT_BLIND
            for e in red["arms"][a]["events"]:
                if e["n"] != 125 or e["dead"]:
                    continue
                if blind and e["kind"] == "rotation":
                    fc, ec = "none", COL[e["kind"]]
                else:
                    fc, ec = COL[e["kind"]], "k"
                ax.scatter(e["z"][xa], e["z"][ya], facecolor=fc, edgecolor=ec,
                           marker=mk.get(a, "o"), s=80 if a == PRIMARY else 44,
                           lw=1.0, alpha=0.95 if a == PRIMARY else 0.65, zorder=3)
        _floorline(ax, red, PRIMARY, xa, False)
        _floorline(ax, red, PRIMARY, ya, True)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(f"{xa}   z vs the arm's own quiet baseline")
        ax.set_ylabel(f"{ya}   z")
        ax.grid(alpha=0.25)
    h = [plt.Line2D([], [], ls="", marker="o", color=COL[k], label=k) for k in KINDS]
    h += [plt.Line2D([], [], ls="", marker=mk[a], color="k", mfc="none", label=a)
          for a in arms]
    h += [plt.Line2D([], [], ls="", marker="o", color="k", mfc="none",
                     label="open = event-matched null\n(rotation on a blind arm)"),
          plt.Line2D([], [], ls=":", color="grey", label="track quiet max|z|")]
    axes[1].legend(handles=h, fontsize=7, loc="lower right")
    fig.suptitle("Weight-ledger typer — the three-corner separation at the matched "
                 "125-step unit (= the op map's own decision window)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_events(red, path):
    arms = [a for a in [PRIMARY, "g2a:self_rekey", "g1a:pair_parse", "g2a:rekey_dead"]
            if a in red["arms"]]
    fig, axes = plt.subplots(len(arms), 1, figsize=(13.5, 3.1 * len(arms)), sharex=False)
    axes = np.atleast_1d(axes)
    order = sorted([(kd, e) for kd in KINDS for e in ONSETS[kd]], key=lambda t: t[1])
    for ax, arm in zip(axes, arms):
        A = red["arms"][arm]
        ev = {(e["kind"], e["onset"]): e for e in A["events"] if e["n"] == 125}
        xs, labs = [], []
        for kd, e in order:
            k = (kd, e)
            if k not in ev:
                continue
            i = len(xs)
            if ev[k]["dead"]:
                ax.text(i, 0.2, "skip", ha="center", fontsize=7, color="#7f8c8d")
            else:
                for j, a_ in enumerate(AXES):
                    ax.bar(i + (j - 1) * 0.26, ev[k]["z"][a_], 0.26,
                           color=COL[KIND_OF_AXIS[a_]],
                           alpha=1.0 if AXIS_OF[kd] == a_ else 0.28)
            xs.append(i)
            labs.append(f"{kd[:3]}\n{e // 1000}k")
        fl = max(A["floor"][a_]["max_abs"] for a_ in AXES)
        ax.axhline(fl, color="k", ls="--", lw=1.1,
                   label=f"quiet max|z| (worst axis) {fl:.2f}")
        ax.axhline(-fl, color="k", ls="--", lw=1.1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylabel("z")
        t = A["typer_n125"]
        ax.set_title(f"{arm}  [{A['role']}]   argmax {t['argmax_correct']}/{t['n_real']}"
                     f"   gated fires {t['gated_fired']}, correct {t['gated_correct']}"
                     + (f"   ({t['n_skipped']} skipped)" if t["n_skipped"] else "")
                     + (f"   [{t['n_non_events']} rotations are NOT events here: "
                        f"{t['n_false_fires']} false fires]" if t["n_non_events"] else ""),
                     fontsize=9.5)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7)
    h = [plt.Line2D([], [], lw=6, color=COL[KIND_OF_AXIS[a_]], label=a_) for a_ in AXES]
    axes[0].legend(handles=h + axes[0].get_legend_handles_labels()[0], fontsize=7,
                   loc="upper left", ncol=2)
    fig.suptitle("Weight-ledger typer — per-event reads on all three axes at the 125-step "
                 "unit (solid = the axis that event type owns)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_longitudinal_and_lag(red, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    for arm, ls in [(PRIMARY, "-"), ("g2a:self_rekey", "--")]:
        L = red["longitudinal"].get(arm, {})
        for kd in KINDS:
            s = L.get(kd, {})
            xs = [int(k) for k in s if not s[k]["dead"]]
            if not xs:
                continue
            ax.plot(xs, [s[str(x)]["z"] for x in xs], marker="o", ls=ls,
                    color=COL[kd], lw=2.2,
                    label=f"{kd} ({arm.split(':')[1]})")
    fl = max(red["arms"][PRIMARY]["floor"][a]["max_abs"] for a in AXES)
    ax.axhline(fl, color="k", ls=":", lw=1.2, label="quiet max|z| floor")
    ax.set_xlabel("event onset step")
    ax.set_ylabel("z on the event's own axis (125-step window)")
    ax.set_title("does the weight ledger drain across the run?", fontsize=10)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for arm, ls in [(PRIMARY, "-"), ("g2a:self_rekey", "--")]:
        T = red["lag_response"].get(arm, [])
        for kd in KINDS:
            by = {}
            for t in T:
                if t["kind"] == kd:
                    by.setdefault(t["lag"], []).append(t["z_axis"])
            xs = sorted(by)
            if len(xs) < 2:
                continue
            ax.plot(xs, [np.mean(by[x]) for x in xs], marker="o", ls=ls,
                    color=COL[kd], lw=2.0, label=f"{kd} ({arm.split(':')[1]})")
    ax.axhline(fl, color="k", ls=":", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag of the window's LEFT edge from onset (steps)")
    ax.set_ylabel("z on the event's own axis")
    ax.set_title("a window read, not a state:\nthe window has to contain the onset\n"
                 "(window lengths differ per point; per-onset table in reduction.txt)",
                 fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    rows = [r for r in red["event_matched_control"]["rows"] if r["kind"] == "rotation"]
    x = np.arange(len(rows))
    w = 0.26
    ax.bar(x - w, [r[PRIMARY] for r in rows], w, color=COL["rotation"],
           label="track (consumes the rotation)")
    for i, (b, lab) in enumerate([("g1a:pair_parse", "pair_parse (blind)"),
                                  ("g2a:rekey_dead", "rekey_dead (blind)")]):
        if b in red["arms"]:
            ax.bar(x + i * w, [r.get(b) or 0.0 for r in rows], w,
                   color=["#bdc3c7", "#7f8c8d"][i], label=lab)
    ax.axhline(red["arms"][PRIMARY]["floor"]["c_emb"]["max_abs"], color="k", ls="--",
               lw=1.1, label="track quiet max|z| (c_emb)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['onset'] // 1000}k" for r in rows])
    ax.set_ylabel("c_emb  z")
    ax.set_xlabel("rotation onset")
    ax.set_title("the event-matched rotation floor", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Weight-ledger typer — longitudinal readability, lag response, and the "
                 "event-matched control", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_directional(red, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
    D = red["arms"][PRIMARY]["directional"]
    for ax, kd in zip(axes, KINDS):
        s = D["summary"].get(kd)
        if s is None:
            continue
        for d in D["quiet"][kd]:
            ax.scatter([d["a"]], [d["cos"]], s=30, color="#95a5a6", zorder=2)
        for d in D["cross_type"][kd]:
            ax.scatter([d["a"]], [d["cos"]], s=44, color=COL[d["kind"]], marker="o",
                       alpha=0.55, zorder=3)
        for d in D["same_type"][kd]:
            ax.scatter([d["a"]], [d["cos"]], s=95, color=COL[kd], marker="D",
                       edgecolor="k", zorder=4)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(s["random_p99"], color="grey", ls=":", lw=1.2,
                   label=f"random p99 {s['random_p99']:.3f}")
        ax.axhline(-s["random_p99"], color="grey", ls=":", lw=1.2)
        if s["quiet_max_abs"] is not None:
            ax.axhline(s["quiet_max_abs"], color="k", ls="--", lw=1.1,
                       label=f"quiet max|cos| {s['quiet_max_abs']:.3f}")
        ax.set_title(f"cos to the first {kd}, group {s['group']}\n"
                     f"same-type mean "
                     + ("n/a" if s["same_mean"] is None else f"{s['same_mean']:+.3f}"),
                     fontsize=9.5)
        ax.set_xlabel("window start step")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cosine to the stored template direction")
    fig.suptitle("Weight-ledger typer — the directional column on `track` "
                 "(diamond = same type, dot = other event, grey = quiet)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_text(red, path):
    o = []
    P = o.append
    P("WEIGHT-LEDGER TYPER CHECK — reduction")
    P("=" * 79)
    P("read   c_emb  = log2(wte  / gmean(h0..h7))          [rotation's subspace, Gate 1G]")
    P("       c_gain = log2(ln_f / gmean(h0..h7))          [burst's subspace]")
    P("       c_comp = log2(gmean(h1,h2)/gmean(h3..h7))    [drift's subspace]")
    P("z = (contrast - the arm's own quiet baseline at that window length) / sigma_quiet.")
    P(f"quiet set = gate1g's admission rule, minus intervals starting before "
      f"step {red['warmup']} (pre-maturity warm-up); scored leave-one-out.")
    P("A rotation is NOT an event for an arm whose wall never rotates or that merged the")
    P("wall away before the onset; those onsets score specificity (should stay silent),")
    P("not detection. A window over which the arm fired a `skip` has dtheta exactly 0.")
    P("")
    P("HEADLINE (g1a:track — consumes all three event types, never merges, never acts)")
    A0 = red["arms"][PRIMARY]
    for tag_ in ["n125", "fastest"]:
        t = A0[f"typer_{tag_}"]
        P(f"  [{tag_:7s}] argmax {t['argmax_correct']}/{t['n_real']}, "
          f"min margin {t['min_margin']:+.2f} sigma;  gated at the arm's own quiet "
          f"max|z|: {t['gated_fired']}/{t['n_real']} fire, {t['gated_correct']} correct"
          + (f"; silent on {', '.join(t['gated_missed_fire'])}"
             if t["gated_missed_fire"] else ""))
    sc = red["sign_consistency"].get(PRIMARY, {})
    if sc:
        P(f"  cross-type rank consistency {sc['pairwise_consistent']}/"
          f"{sc['pairwise_total']} ({100 * sc['frac']:.0f}%)")
    L0 = red["longitudinal"][PRIMARY]
    for kd in KINDS:
        fl = L0.get(kd + "_first_last")
        if fl and fl["ratio"] is not None:
            P(f"  {kd:9s} first->last  {fl['first']:+.2f} -> {fl['last']:+.2f} sigma "
              f"(x{fl['ratio']:.2f})")
    P("")
    P("ALL EIGHT BANKED ARMS at the 125-step unit "
      "(detection on real events | specificity on non-events)")
    P(f"  {'arm':18s}{'role':24s}{'argmax':>9s}{'gated':>9s}{'ok':>5s}"
      f"{'skip':>6s}{'non-ev':>8s}{'falsefire':>11s}")
    tot = dict(real=0, arg=0, fire=0, ok=0, skip=0, non=0, fp=0)
    for name, R in sorted(red["arms"].items()):
        t = R["typer_n125"]
        P(f"  {name:18s}{R['role']:24s}"
          f"{'%d/%d' % (t['argmax_correct'], t['n_real']):>9s}"
          f"{'%d/%d' % (t['gated_fired'], t['n_real']):>9s}"
          f"{t['gated_correct']:>5d}{t['n_skipped']:>6d}{t['n_non_events']:>8d}"
          f"{t['n_false_fires']:>11d}")
        tot["real"] += t["n_real"]
        tot["arg"] += t["argmax_correct"]
        tot["fire"] += t["gated_fired"]
        tot["ok"] += t["gated_correct"]
        tot["skip"] += t["n_skipped"]
        tot["non"] += t["n_non_events"]
        tot["fp"] += t["n_false_fires"]
    P(f"  {'TOTAL':18s}{'':24s}"
      f"{'%d/%d' % (tot['arg'], tot['real']):>9s}"
      f"{'%d/%d' % (tot['fire'], tot['real']):>9s}"
      f"{tot['ok']:>5d}{tot['skip']:>6d}{tot['non']:>8d}{tot['fp']:>11d}")
    P("")
    A = red["arms"][PRIMARY]
    P(f"--- instrument calibration, {PRIMARY} ---")
    for k in AXES:
        f, fl = A["fit"][k], A["floor"][k]
        P(f"  {k:7s} baseline slope {f['slope']:+.3f}/ln n   sigma {f['sigma']:.3f} log2 "
          f"(= x{2 ** f['sigma']:.2f})   n_quiet {f['n_quiet']}")
        P(f"          quiet floor: max|z| {fl['max_abs']:.2f} (at {fl['worst_interval']}), "
          f"p90|z| {fl['p90_abs']:.2f}")
    P("  per-group quiet scaling exponent alpha (coherence audit; 0.5 = random walk, "
      "1.0 = coherent):")
    P("    " + "   ".join(f"{L} {A['group_fit'][L]['alpha']:.2f}"
                          for L in ["wte", "h1", "h2", "h4", "ln_f", "lm_head"]))
    P("")
    for name in [PRIMARY] + [a for a in red["arms"] if a != PRIMARY]:
        R = red["arms"][name]
        P(f"--- {name}   [{R['role']}] ---")
        for tag_ in ["n125", "fastest"]:
            t = R[f"typer_{tag_}"]
            if not t["n_real"]:
                P(f"  typer[{tag_}]  no real events (rotation-blind arm has none at "
                  f"these onsets)")
                continue
            P(f"  typer[{tag_}]  argmax {t['argmax_correct']}/{t['n_real']}"
              f"  min margin {t['min_margin']:+.2f} sigma"
              f"  |  gated: fires {t['gated_fired']}/{t['n_real']}, "
              f"correct {t['gated_correct']}"
              + (f", silent on {', '.join(t['gated_missed_fire'])}"
                 if t["gated_missed_fire"] else "")
              + (f"  |  {t['n_skipped']} SKIPPED" if t["n_skipped"] else "")
              + f"  |  profile-template {t['profile_correct']}/{t['profile_n']}")
            if t["n_non_events"]:
                P(f"                specificity: {t['n_non_events']} non-events "
                  f"(rotations this arm cannot see), "
                  f"{t['n_false_fires']} false fires"
                  + (f": {', '.join(t['false_fires'])}" if t["false_fires"] else ""))
        P(f"  {'event':17s}{'n':>5s}{'c_emb':>9s}{'c_gain':>9s}{'c_comp':>9s}"
          f"{'own':>6s}{'margin':>8s}{'xfloor':>8s}  flags")
        for e in R["events"]:
            tagname = f"{e['kind']}@{e['onset']}"
            if e["dead"]:
                P(f"  {tagname:17s}{e['n']:5d}        SKIPPED (dtheta exactly 0)")
                continue
            fl = ("" if e["typable"] else "NOT-AN-EVENT ")
            fl += ("fires" if max(e["z"][k] - R["floor"][k]["max_abs"]
                                  for k in AXES) > 0 else "silent")
            P(f"  {tagname:17s}{e['n']:5d}"
              + "".join(f"{e['z'][a]:9.2f}" for a in AXES)
              + f"{e['axis'][2:]:>6s}{e['margin']:8.2f}{e['mult_at_axis']:8.2f}  {fl}")
        P("")
    P("--- sign/rank consistency at the 125-step unit (the finding-1 discipline) ---")
    for name, c in red["sign_consistency"].items():
        note = "  [rotation is NOT an event for this arm]" if name in ROT_BLIND else ""
        P(f"  {name}: {c['pairwise_consistent']}/{c['pairwise_total']} "
          f"({100 * c['frac']:.0f}%) cross-type comparisons in the right direction{note}")
        for ax, d in c["by_axis"].items():
            if d["min_own"] is None or d["max_other"] is None:
                P(f"     {ax}: not scored (this arm has no events of that type)")
                continue
            P(f"     {ax}: min(own type) {d['min_own']:+.2f}   "
              f"max(other types) {d['max_other']:+.2f}")
    P("")
    P("--- longitudinal: does the weight ledger drain across the run? ---")
    for name, d in red["longitudinal"].items():
        if name not in ABSORBERS + list(ROT_BLIND):
            continue
        P(f"  {name}")
        for kd in KINDS:
            s = d.get(kd, {})
            if not s:
                continue
            P(f"    {kd:9s} " + "  ".join(
                f"{k}:" + ("skip" if v["dead"] else f"{v['z']:+.2f}")
                for k, v in s.items()))
            fl = d.get(kd + "_first_last")
            if fl and fl["ratio"] is not None:
                P(f"      first {fl['first']:+.2f} -> last {fl['last']:+.2f}  "
                  f"(x{fl['ratio']:.2f})")
    P("")
    P("--- event-matched control (rotation-blind arms) ---")
    for b, why in ROT_BLIND.items():
        P(f"  {b}: {why}")
    P(f"  {'event':17s}{'axis':>7s}{PRIMARY:>16s}"
      + "".join(f"{b:>17s}" for b in ROT_BLIND))
    for r in red["event_matched_control"]["rows"]:
        P(f"  {r['kind'] + '@' + str(r['onset']):17s}{r['axis'][2:]:>7s}"
          f"{r[PRIMARY]:16.2f}"
          + "".join(("%17.2f" % r[b]) if r.get(b) is not None else f"{'--':>17s}"
                    for b in ROT_BLIND))
    P("")
    P("--- directional column (cos of raw dtheta to the first event of that type) ---")
    for name in ABSORBERS:
        if name not in red["arms"]:
            continue
        P(f"  {name}")
        for kd, s in red["arms"][name]["directional"].get("summary", {}).items():
            P(f"    {kd:9s} group {s['group']:7s} d={s['dim']:<7d} same-type mean "
              + ("n/a" if s["same_mean"] is None else
                 f"{s['same_mean']:+.3f} (min {s['same_min']:+.3f}, n={s['n_same']})"))
            P(f"              cross-type max|cos| "
              + ("n/a" if s["cross_max_abs"] is None else f"{s['cross_max_abs']:.3f}")
              + "   quiet |cos| mean/p90/max "
              + ("n/a" if s["quiet_max_abs"] is None else
                 f"{s['quiet_mean_abs']:.3f}/{s['quiet_p90_abs']:.3f}/"
                 f"{s['quiet_max_abs']:.3f}")
              + f"   random p99 {s['random_p99']:.3f}")
            if s["margin_over_quiet_max"] is not None:
                P(f"              margin of the worst same-type read over the worst "
                  f"quiet read: {s['margin_over_quiet_max']:+.3f}")
    P("")
    P("--- lag response: z on the event's own axis for a window [onset+lag, +lag+n] ---")
    P("    (this is what an evaluator reading a TRAILING window sees; the op map decides")
    P("     at +50...+125, where the trailing 125-step window still contains the onset)")
    for name, rr in red["lag_response"].items():
        P(f"  {name}")
        for kd in KINDS:
            for e in ONSETS[kd]:
                rows = [t for t in rr if t["kind"] == kd and t["onset"] == e]
                if len(rows) > 1:
                    P(f"    {kd + '@' + str(e):17s} " + "  ".join(
                        f"lag{t['lag']}+n{t['n']}: {t['z_axis']:+.2f}"
                        + ("*" if t["from_null_set"] else "") for t in rows))
    P("    (* = a window gate1g's rule admits as QUIET, scored leave-one-out: it sits in")
    P("       the null, which is exactly why the null is an upper bound on the floor)")
    P("")
    P("--- arms that acted: the meter's own blind spot ---")
    for name, R in red["arms"].items():
        if R["n_skipped_intervals"]:
            P(f"  {name}: {R['n_skipped_intervals']} intervals with dtheta EXACTLY 0 "
              f"(fired skips)")
    P("")
    P("--- merged arms (rotation stops being news once the wall is merged away) ---")
    for name, R in red["arms"].items():
        if R["merge_at"] is None:
            continue
        s = red["longitudinal"][name]["rotation"]
        P(f"  {name} (merge at {R['merge_at']}): rotation c_emb z  "
          + "  ".join(f"{k}:" + ("skip" if v["dead"] else f"{v['z']:+.2f}")
                      for k, v in s.items()))
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wl0")
    ap.add_argument("--warmup", type=int, default=5000,
                    help="quiet intervals starting before this step are excluded from the "
                         "null (pre-maturity warm-up). Default = the first event onset.")
    a = ap.parse_args()
    fig = os.path.join(HERE, "figures", a.tag)
    D = json.load(open(os.path.join(fig, "wledger.json")))
    red = reduce_all(D, a.warmup)
    red["tag"] = a.tag
    red["null_design"] = {
        "baseline": "per-arm linear model of each contrast in log n, fitted on that arm's "
                    "own quiet intervals (gate1g's quiet_pairs rule) with the pre-maturity "
                    "warm-up excluded; quiet intervals scored leave-one-out.",
        "floors": "sigma_quiet (the z unit), the arm's empirical quiet max|z| and p90|z|, "
                  "and two event-matched rotation-blind arms.",
        "caveats": ["quiet intervals admit absorption tails, so event excesses are lower "
                    "bounds and quiet floors are upper bounds",
                    "a checkpoint delta is ||sum_t g_t||, not sum_t ||g_t||",
                    "banked lags: rotation 125; burst 25 and 125; drift 125 (wave 1) "
                    "and 25 (wave 2)"]}
    with open(os.path.join(fig, "reduction.json"), "w") as fh:
        json.dump(red, fh, indent=2)
    fig_corners(red, os.path.join(fig, "fig1_three_corners.png"))
    fig_events(red, os.path.join(fig, "fig2_per_event_z.png"))
    fig_longitudinal_and_lag(red, os.path.join(fig, "fig3_longitudinal_lag.png"))
    fig_directional(red, os.path.join(fig, "fig4_directional.png"))
    txt = write_text(red, os.path.join(fig, "reduction.txt"))
    print(txt)
    print("\nwrote reduction.json, reduction.txt + 4 figures to", fig)


if __name__ == "__main__":
    main()
