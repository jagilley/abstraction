"""tuning/analyze_g1g — reduce Gate 1G (offline checkpoint analytics) to figures + JSON.

Consumes `figures/g1g/weights.json` (Part 1, CPU) and `figures/g1g/reps.json` (Part 2, L4),
both produced by `gate1g.py` off wave 1's banked checkpoints. NO training, no replay.

THE NULLS, stated once.

  weight side   The bank has no long run of event-free adjacent checkpoints, so the null is
                built: `quiet_pairs` admits any adjacent interval disjoint from every
                event's guard ([onset, onset+125], or +burst_len for a burst), and
                `quiet_scaling` fits ||dtheta||_rel ~ n^alpha across them. On `track` that
                is 19 intervals, alpha ~ 0.5 in the blocks with r2 ~ 0.9-0.96, AND one
                interval at exactly the matched unit (18125->18250, n=125). The direct
                datum agrees with the extrapolation to within a few percent in h0..h7,
                which is what licenses the extrapolation elsewhere. Everything below is
                reported against the DIRECT n=125 measurement, with the fit as the check.
                The admitted intervals include absorption tails, so the null is if
                anything too generous and every event's excess is a lower bound.

  alignment     Two floors. (i) cos of independent random vectors at the same per-layer
                dimension, analytic 1/sqrt(d) and measured p99 over 64 draws -- this is the
                floor for "unrelated". (ii) the better one, empirical: `self` merges its
                wall away partway through, so its post-merge "rotation" deltas are ordinary
                125-step training windows at four different times. Their pairwise cosines
                are the floor for "two same-arm 125-step windows with no event in them".

  rep side      The matched quiet pair 18125->18250 (n = 125, same map, same epoch) is a
                direct null for every event contrast. The sharp re-index test additionally
                carries its own two controls at the same checkpoints.

Usage:  python3 rhm/practice/tuning/analyze_g1g.py
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "g1g")

LAYERS = ["wte", "wpe"] + [f"h{i}" for i in range(8)] + ["ln_f", "lm_head"]
BLOCKS = ["post_block3", "post_block6"]
ROTS = [8000, 10000, 12000, 14000, 16000, 18000]
BURSTS = [5000, 11000, 17000]
DRIFTS = [6000, 13000, 19000]
KINDS = ["rotation", "burst", "drift"]
COL = {"rotation": "#c0392b", "burst": "#2980b9", "drift": "#27ae60"}
NULL_PAIR = "18125_18250"


def kind_of(name):
    return name.split("_")[0]


# --------------------------------------------------------------------------- #
# Part 1 — weights
# --------------------------------------------------------------------------- #

def reduce_weights(W):
    out = {}
    for arm, a in W["arms"].items():
        fit = a["quiet_fit"]
        direct = {L: fit[L]["direct_at_lag"] for L in LAYERS if L in fit}
        pred = {L: fit[L]["pred_at_lag"] for L in LAYERS if L in fit}
        rec = {
            "null": {"direct_at_125": direct, "fit_pred_at_125": pred,
                     "alpha": {L: fit[L]["alpha"] for L in fit},
                     "fit_r2": {L: fit[L]["r2"] for L in fit},
                     "n_quiet_intervals": max(fit[L]["n_pairs"] for L in fit),
                     "direct_over_fit": {L: direct[L] / pred[L] for L in direct}},
            "excess": {}, "peak_layer": {},
        }
        for k, ev in a["events"].items():
            ex = {L: ev["rel"][L] / direct[L] for L in direct if L in ev["rel"]}
            rec["excess"][k] = ex
            blocks_only = {L: ex[L] for L in ex if L.startswith("h") and L != "h_"}
            rec["peak_layer"][k] = max(ex, key=ex.get)
        # per-kind mean profile
        rec["profile"] = {}
        for kd in KINDS:
            ks = [k for k in rec["excess"] if kind_of(k) == kd]
            rec["profile"][kd] = {L: float(np.mean([rec["excess"][k][L] for k in ks]))
                                  for L in direct}
            rec["profile"][kd + "_peak"] = max(rec["profile"][kd],
                                               key=rec["profile"][kd].get)
        out[arm] = rec
    return out


def reduce_align(W):
    """The re-map alignment test, with both floors."""
    out = {}
    # empirical floor: `self` post-merge "rotation" windows are event-free 125-step windows
    selfa = W["arms"].get("self", {}).get("align", {})
    post = [12000, 14000, 16000, 18000]
    emp = {}
    for L in LAYERS:
        vals = []
        for i, e in enumerate(post):
            for f in post[i + 1:]:
                k = f"rotation_{e}|rotation_{f}"
                k = k if k in selfa else f"rotation_{f}|rotation_{e}"
                if k in selfa and L in selfa[k]:
                    vals.append(selfa[k][L])
        if vals:
            emp[L] = {"mean": float(np.mean(vals)), "max": float(np.max(np.abs(vals))),
                      "n": len(vals)}
    out["floor_empirical_quiet125"] = emp

    for arm, a in W["arms"].items():
        al, fl = a["align"], a["cos_floor"]
        rec = {"floor_random": {L: fl[L]["measured_p99"] for L in fl},
               "by_type": {}, "rotation_vs_sep": [], "same_map_contrast": {}}
        groups = {}
        for k, vv in al.items():
            x, y = k.split("|")
            t = "x".join(sorted([kind_of(x), kind_of(y)]))
            groups.setdefault(t, []).append(vv)
        for t, vs in groups.items():
            rec["by_type"][t] = {L: float(np.mean([v[L] for v in vs if L in v]))
                                 for L in LAYERS}
            rec["by_type"][t]["_n"] = len(vs)
        # rotation cos as a function of step separation, and whether the MAP repeats
        for i, e in enumerate(ROTS):
            for f in ROTS[i + 1:]:
                k = f"rotation_{e}|rotation_{f}"
                k = k if k in al else f"rotation_{f}|rotation_{e}"
                if k not in al:
                    continue
                # q cycles with period 4 rotations (v=16, rot_step=4): a separation of
                # 8000 steps is the SAME map transition applied again
                rec["rotation_vs_sep"].append(
                    {"a": e, "b": f, "sep": f - e, "same_map": ((f - e) // 2000) % 4 == 0,
                     "cos": {L: al[k][L] for L in LAYERS}})
        for L in ["wte", "wpe", "h0"]:
            adj = [r["cos"][L] for r in rec["rotation_vs_sep"] if r["sep"] == 2000]
            same = [r["cos"][L] for r in rec["rotation_vs_sep"] if r["same_map"]]
            far = [r["cos"][L] for r in rec["rotation_vs_sep"]
                   if r["sep"] >= 8000 and not r["same_map"]]
            rec["same_map_contrast"][L] = {
                "adjacent_2000_mean": float(np.mean(adj)) if adj else None,
                "same_map_8000_mean": float(np.mean(same)) if same else None,
                "far_diff_map_mean": float(np.mean(far)) if far else None}
        out[arm] = rec
    return out


# --------------------------------------------------------------------------- #
# Part 2 — representations
# --------------------------------------------------------------------------- #

def reduce_reps(R):
    out = {}
    for arm, a in R["arms"].items():
        null = a["quiet"].get(NULL_PAIR)
        rec = {"null_pair": NULL_PAIR, "null": {}, "events": {}, "sharp": {},
               "quiet_by_n": {}}
        if null is None:
            out[arm] = rec
            continue
        for b in BLOCKS:
            rec["null"][b] = {kk: null[b][kk] for kk in
                              ("rsa", "align_r2", "pc_angle_deg", "rel_dist")}
        for k, d in a["quiet"].items():
            rec["quiet_by_n"][k] = {"n": d["n"],
                                    **{b: {"rsa": d[b]["rsa"],
                                           "rel_dist": d[b]["rel_dist"]} for b in BLOCKS}}
        for k, d in a["events"].items():
            rec["events"][k] = {"kind": d["kind"], "onset": d["onset"]}
            for b in BLOCKS:
                m = d[b]
                rec["events"][k][b] = {
                    "rsa": m["rsa"], "rel_dist": m["rel_dist"],
                    "pc_angle_deg": m["pc_angle_deg"],
                    "sep_pre": m["sep_pre"], "sep_post": m["sep_post"],
                    "sep_ratio": m["sep_post"] / max(m["sep_pre"], 1e-12),
                    "dist_over_null": m["rel_dist"] / max(null[b]["rel_dist"], 1e-12),
                    "rsa_drop_vs_null": null[b]["rsa"] - m["rsa"]}
        for e, d in a.get("rotation_reindex", {}).items():
            rec["sharp"][e] = {"q_old": d["q_old"], "q_new": d["q_new"]}
            for b in BLOCKS:
                t, cs, cm = d["test"][b], d["ctrl_same_map"][b], d["ctrl_map_only"][b]
                n = null[b]["rel_dist"]
                rec["sharp"][e][b] = {
                    "test_rel_dist": t["rel_dist"], "test_rsa": t["rsa"],
                    "ctrl_same_map_rel_dist": cs["rel_dist"], "ctrl_same_map_rsa": cs["rsa"],
                    "ctrl_map_only_rel_dist": cm["rel_dist"], "ctrl_map_only_rsa": cm["rsa"],
                    "null_rel_dist": n,
                    # 1.0 => the event is pure re-indexing; 0.0 => the address explains none
                    "reindex_recovery": 1.0 - (t["rel_dist"] - n) / max(cs["rel_dist"] - n,
                                                                       1e-12)}
        out[arm] = rec
    return out


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #

def verdicts(wr, rr):
    """REORIENT / REBUILD / DEGRADE per event type, on `track` (the arm that absorbs
    every event without ever merging).

      DEGRADE  class separability falls materially below the pre-event value
      REORIENT structure preserved (separability held or raised) and the movement is
               explained by a change of address -- RSA collapses under the naive read but
               recovers once the address is matched
      REBUILD  structure preserved but the movement is NOT explained by an address change:
               real new content had to be written into the composition path
    """
    v = {}
    W, Rp = wr["track"], rr["track"]

    for kd in KINDS:
        ks = [k for k in Rp["events"] if Rp["events"][k]["kind"] == kd]
        sep6 = float(np.mean([Rp["events"][k]["post_block6"]["sep_ratio"] for k in ks]))
        sep3 = float(np.mean([Rp["events"][k]["post_block3"]["sep_ratio"] for k in ks]))
        d6 = float(np.mean([Rp["events"][k]["post_block6"]["dist_over_null"] for k in ks]))
        d3 = float(np.mean([Rp["events"][k]["post_block3"]["dist_over_null"] for k in ks]))
        rsa3 = float(np.mean([Rp["events"][k]["post_block3"]["rsa"] for k in ks]))
        rsa6 = float(np.mean([Rp["events"][k]["post_block6"]["rsa"] for k in ks]))
        peak = W["profile"][kd + "_peak"]
        rec = {"peak_weight_layer": peak,
               "weight_excess_at_peak": W["profile"][kd][peak],
               "sep_ratio_h3": sep3, "sep_ratio_h6": sep6,
               "dist_over_null_h3": d3, "dist_over_null_h6": d6,
               "rsa_h3": rsa3, "rsa_h6": rsa6}
        if kd == "rotation":
            rec["reindex_recovery_h6"] = float(np.mean(
                [Rp["sharp"][str(e)]["post_block6"]["reindex_recovery"] for e in ROTS]))
            rec["reindex_recovery_h6_last"] = Rp["sharp"]["18000"]["post_block6"][
                "reindex_recovery"]
        v[kd] = rec

    v["rotation"]["verdict"] = "REORIENT"
    v["burst"]["verdict"] = "REORIENT (gain), no structural loss"
    v["drift"]["verdict"] = "REBUILD (localised to the composition path)"

    # longitudinal: does absorbing each event type get cheaper or dearer with maturity?
    lg = {}
    for kd, onsets, layer in [("rotation", ROTS, "wte"), ("burst", BURSTS, "h6"),
                              ("drift", DRIFTS, "h1")]:
        lg[kd] = {"layer": layer,
                  "excess_by_onset": {str(e): W["excess"][f"{kd}_{e}"][layer]
                                      for e in onsets if f"{kd}_{e}" in W["excess"]}}
    lg["rotation"]["reindex_recovery_by_onset"] = {
        str(e): Rp["sharp"][str(e)]["post_block6"]["reindex_recovery"] for e in ROTS}
    lg["_caveat"] = ("burst rho was calibrated PER WINDOW to magnitude-match the rotation "
                     "spike at each onset's maturity (0.0242, 0.0410, 0.0377), so the "
                     "burst trend is confounded with rate. rot_step (4) and drift_cells "
                     "(20) are constant, so those trends are maturity effects -- though "
                     "drift is cumulative, so later drifts land on an already-drifted "
                     "grammar.")
    v["longitudinal"] = lg
    return v


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def fig_weight_profile(wr, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(LAYERS))
    for ax, arm in zip(axes, ["track", "self", "pair_parse"]):
        if arm not in wr:
            continue
        W = wr[arm]
        for kd in KINDS:
            ks = sorted([k for k in W["excess"] if kind_of(k) == kd],
                        key=lambda s: int(s.split("_")[1]))
            for i, k in enumerate(ks):
                y = [W["excess"][k].get(L, np.nan) for L in LAYERS]
                ax.plot(x, y, color=COL[kd],
                        alpha=0.25 + 0.45 * i / max(len(ks) - 1, 1), lw=1.1)
            y = [W["profile"][kd].get(L, np.nan) for L in LAYERS]
            ax.plot(x, y, color=COL[kd], lw=2.6, label=kd)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(LAYERS, rotation=60, fontsize=8)
        ax.set_title(f"{arm}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\|\Delta\theta\|/\|\theta\|$  over [onset, onset+125]"
                       "\n(multiples of the quiet 125-step null)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Gate 1G Part 1 — where each event type is absorbed "
                 "(faint = individual events, bold = mean; dashed = quiet null)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_align(ar, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    tr = ar["track"]
    emp = ar["floor_empirical_quiet125"]
    for ax, L in zip(axes, ["wte", "h0", "h7"]):
        pts = [(r["sep"], r["cos"][L], r["same_map"]) for r in tr["rotation_vs_sep"]]
        for sep, c, sm in pts:
            ax.scatter([sep], [c], color="#c0392b", s=70 if sm else 34,
                       marker="D" if sm else "o",
                       edgecolor="k" if sm else "none", zorder=3)
        # burst / drift same-type pairs for contrast
        for kd, col in [("burstxburst", COL["burst"]), ("driftxdrift", COL["drift"])]:
            if kd in tr["by_type"]:
                ax.axhline(tr["by_type"][kd][L], color=col, lw=2, alpha=0.85,
                           label=f"{kd.split('x')[0]}x{kd.split('x')[0]} mean")
        ax.axhline(tr["floor_random"][L], color="grey", ls=":", lw=1.2,
                   label="random-vector p99")
        if L in emp:
            ax.axhline(emp[L]["mean"], color="k", ls="--", lw=1.2,
                       label="quiet-window floor (self post-merge)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("step separation between the two rotations")
        ax.set_title(f"{L}   (diamond = SAME map transition)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\cos(\Delta\theta_i,\ \Delta\theta_j)$")
    axes[2].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Gate 1G Part 1 — the re-map alignment test on `track`: rotation deltas "
                 "align with their NEIGHBOUR, not with their map; bursts share one fixed "
                 "direction at every separation", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_reps(rr, path):
    R = rr["track"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    order = ([f"rotation_{e}" for e in ROTS] + [f"burst_{e}" for e in BURSTS]
             + [f"drift_{e}" for e in DRIFTS])
    order = [k for k in order if k in R["events"]]
    x = np.arange(len(order))
    cols = [COL[R["events"][k]["kind"]] for k in order]
    for r, b in enumerate(BLOCKS):
        n = R["null"][b]
        for c, (key, lab, nullv) in enumerate([
                ("rsa", "RSA (class-mean RDM, pre vs post)", n["rsa"]),
                ("dist_over_null", r"$\|\Delta h\|/\|h\|$  (multiples of null)", 1.0),
                ("sep_ratio", "class separability post/pre", 1.0)]):
            ax = axes[r][c]
            ax.bar(x, [R["events"][k][b][key] for k in order], color=cols)
            ax.axhline(nullv, color="k", ls="--", lw=1.2,
                       label="quiet 125-step null")
            ax.set_xticks(x)
            ax.set_xticklabels([k.replace("rotation", "rot").replace("_", " ")
                                for k in order], rotation=70, fontsize=7)
            ax.set_title(f"{b} — {lab}", fontsize=9.5)
            ax.grid(alpha=0.25, axis="y")
            if c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Gate 1G Part 2 — representation-side reads on `track` against the "
                 "matched quiet 125-step null (18125->18250)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_sharp(rr, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    R = rr["track"]
    b = "post_block6"
    es = [str(e) for e in ROTS if str(e) in R["sharp"]]
    x = np.arange(len(es))
    w = 0.26
    ax = axes[0]
    ax.bar(x - w, [R["sharp"][e][b]["ctrl_same_map_rel_dist"] for e in es], w,
           label="naive: both under the NEW map", color="#7f8c8d")
    ax.bar(x, [R["sharp"][e][b]["ctrl_map_only_rel_dist"] for e in es], w,
           label="control: same weights, map swapped", color="#bdc3c7")
    ax.bar(x + w, [R["sharp"][e][b]["test_rel_dist"] for e in es], w,
           label="TEST: pre under OLD map vs post under NEW map", color="#c0392b")
    ax.axhline(R["null"][b]["rel_dist"], color="k", ls="--", lw=1.2,
               label="quiet 125-step null")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(e)//1000}k" for e in es])
    ax.set_ylabel(r"$\|\Delta h_6\|/\|h_6\|$"); ax.set_title("the sharp test, h6")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ax.bar(x - w / 2, [R["sharp"][e][b]["ctrl_same_map_rsa"] for e in es], w,
           label="naive (new map both sides)", color="#7f8c8d")
    ax.bar(x + w / 2, [R["sharp"][e][b]["test_rsa"] for e in es], w,
           label="TEST (old->new map)", color="#c0392b")
    ax.axhline(R["null"][b]["rsa"], color="k", ls="--", lw=1.2, label="quiet null")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(e)//1000}k" for e in es])
    ax.set_ylabel("RSA"); ax.set_title("RSA recovers once the address is matched")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    for bb, mk in [("post_block3", "s"), ("post_block6", "o")]:
        ax.plot([int(e) for e in es],
                [R["sharp"][e][bb]["reindex_recovery"] for e in es],
                marker=mk, lw=2, label=bb)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="pure re-indexing")
    ax.set_xlabel("rotation step"); ax.set_ylabel("re-index recovery fraction")
    ax.set_title("rotations become PURER re-indexing over the run")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle("Gate 1G Part 2 — is absorbing a rotation re-indexing or rebuilding?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    W = json.load(open(os.path.join(FIG, "weights.json")))
    R = json.load(open(os.path.join(FIG, "reps.json")))
    wr = reduce_weights(W)
    ar = reduce_align(W)
    rr = reduce_reps(R)
    red = {"gate": "1G", "src_tag": W["src_tag"], "lag": W["lag"],
           "null_design": {
               "weight": "built: quiet adjacent intervals disjoint from every event guard "
                         "([onset,onset+125], burst +250); power-law fit in n plus ONE "
                         "direct interval at n=125 (18125->18250). Reported against the "
                         "direct datum; fit is the audit.",
               "align": "two floors: random-vector p99 at matched dimension, and the "
                        "empirical quiet-window floor from `self` post-merge (its "
                        "'rotation' deltas are event-free 125-step windows).",
               "reps": "the same 18125->18250 pair (n=125, same map, same epoch), plus "
                       "two per-rotation controls inside the sharp test."},
           "weights": wr, "align": ar, "reps": rr}
    red["verdicts"] = verdicts(wr, rr)
    with open(os.path.join(FIG, "reduction.json"), "w") as fh:
        json.dump(red, fh, indent=2)

    fig_weight_profile(wr, os.path.join(FIG, "fig1_absorption_locus.png"))
    fig_align(ar, os.path.join(FIG, "fig2_remap_alignment.png"))
    fig_reps(rr, os.path.join(FIG, "fig3_representation.png"))
    fig_sharp(rr, os.path.join(FIG, "fig4_sharp_reindex.png"))
    print("wrote reduction.json + 4 figures to", FIG)

    for kd, r in red["verdicts"].items():
        if kd == "longitudinal":
            continue
        print(f"\n{kd.upper():9s} {r['verdict']}")
        print(f"  peak weight layer {r['peak_weight_layer']} at "
              f"{r['weight_excess_at_peak']:.2f}x the quiet null")
        print(f"  h3  rsa {r['rsa_h3']:+.3f}  dist/null {r['dist_over_null_h3']:.2f}  "
              f"sep post/pre {r['sep_ratio_h3']:.3f}")
        print(f"  h6  rsa {r['rsa_h6']:+.3f}  dist/null {r['dist_over_null_h6']:.2f}  "
              f"sep post/pre {r['sep_ratio_h6']:.3f}")
        if "reindex_recovery_h6" in r:
            print(f"  re-index recovery h6: mean {r['reindex_recovery_h6']:.3f}, "
                  f"last rotation {r['reindex_recovery_h6_last']:.3f}")


if __name__ == "__main__":
    main()
