"""Reduce a `tall` run.

    python3 rhm/practice/tall/analyze_tall.py --tag tl_s0 --fetch --figures

Readouts, in the order the round asks for them:
  0. THE OPERATING-REGIME CHECK, which must be read before anything else: where did competence
     actually land between the two references the setup measured -- `stale` (never commit) and
     `floor` (the exact-DP oracle)? Every downstream question presupposes the loop opened that
     range. `tl_s0` did not, so this readout is what licenses or voids the rest.
  1. the discriminator: certificate / vocabulary firing at INTERIOR level 3, per arm
  2. schedule-shape ranking at matched T -- ranks first, fractions second
  3. the mechanism instruments: entry identity, and the fixed-reference (base action set) probe
  4. the deep eras, where the earnable vocabulary is inert by construction
  5. pacer boundary placements against the sized schedules
  6. instrument suite: recert, plant guard, grader cost, mined-vs-random
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from rhm.practice.ear.analyze_ear import replay_unitlp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME, REMOTE = "rhm-scaling-data", "rhm_practice_tall"
ORDER = ["given", "sched_uniform", "sched_bottom", "sched_earnable", "pace_cert", "pace_vocab"]
# the stream-position noise measured on the depth-4 arc (recital, rc_s0 vs rv_s0 drift guards):
# an identical arm spec moved by up to this much on the deep cell, so differences smaller than
# it are not interpretable at n=1.
STREAM_NOISE = 0.034


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for a in ORDER + sorted(os.listdir(root)):
        p = os.path.join(root, a, "results.json")
        if a not in arms and os.path.isfile(p):
            arms[a] = json.load(open(p))
    return setup, arms


def ladder(r, key="all_eras"):
    pr = [p for p in r["log"]["probe"] if key in p]
    return [pr[-1][key][str(j)] for j in range(5)] if pr else None


def report(tag):
    setup, arms = load(tag)
    cfg, refs, eras = setup["config"], setup["refs"], setup["eras"]
    nm = [e["name"] for e in eras]
    n_e = len(eras)
    print(f"\n=== {tag} ===")
    print(f"depth={cfg['depth']} m={cfg['m']} seqlen={cfg['s']**cfg['depth']} "
          f"max_macro_level={cfg['max_macro_level']} budget={cfg['budget']} "
          f"g_budget={cfg['g_budget']} T={cfg['t_budget']:.0f} probe_every={cfg['probe_every']}")
    print("eras: " + " | ".join(
        f"{nm[i]} d0={refs['d0'][i]:.2f} stale={refs['stale'][i]:.3f} floor={refs['floor'][i]:.3f}"
        for i in range(n_e)))

    # ---- 0. OPERATING REGIME -----------------------------------------------------------
    print("\n-- 0. OPERATING-REGIME CHECK: fraction of the available range (stale -> floor) "
          "that competence actually recovered, per era visited --")
    print(f"{'arm':15s} " + " ".join(f"{x:>10s}" for x in nm))
    for a, r in arms.items():
        log = r["log"]
        era = np.asarray(log["era"])
        cells = []
        for k in range(1, n_e + 1):
            idx = np.where(era == k)[0]
            if len(idx) < 6:
                cells.append(f"{'--':>10s}"); continue
            l5 = float(np.mean([log["e"][i] for i in idx[-5:]]))
            st, fl = refs["stale"][k - 1], refs["floor"][k - 1]
            cells.append(f"{(st - l5) / (st - fl):10.3f}")
        print(f"{a:15s} " + " ".join(cells))
    print("   (1.0 == reached the exact-DP oracle; 0.0 == no better than never committing.\n"
          "    Deep-era values are inflated: stale-floor is only "
          f"{refs['stale'][-1] - refs['floor'][-1]:.3f} at {nm[-1]}, so the denominator is "
          "nearly zero and noise dominates.)")

    # ---- 1. the discriminator ------------------------------------------------------------
    kw = dict(alpha=cfg["alpha"], c=cfg["sil_c"], cv=cfg["sil_cv"], W=cfg["sil_W"],
              hold=cfg["sil_hold"], min_cycle=cfg["sil_min_cycle"],
              min_drop=cfg["lp_min_drop"])
    print("\n-- 1. DISCRIMINATOR: detector firing in the two EARNING eras (level 3 is INTERIOR "
          "here; at depth 4 it was top-of-earnable) --")
    print(f"{'arm':15s} {'era':>3s} {'n':>3s} {'cert@':>6s} {'vocab@':>7s} {'task@':>6s} "
          f"{'advanced':>10s} {'commit':>26s}")
    for a, r in arms.items():
        log = r["log"]; era = np.asarray(log["era"])
        for k in (1, 2):
            idx = np.where(era == k)[0]
            if not len(idx):
                continue
            fc, _ = replay_unitlp([log["cert"][i]["A"] for i in idx], **kw)
            fv, _ = replay_unitlp([log["vcert"][i]["A"] for i in idx], **kw)
            ft, _ = replay_unitlp([log["e"][i] for i in idx], **kw)
            adv = [e for e in r["events"] if e["kind"] == "advance" and e["from_era"] == k]
            cm = [e for e in r["events"] if e["kind"] == "commit" and e["era"] == k]
            cs = (f"L{cm[0]['level']}@c{cm[0]['c_in_era']} "
                  f"{'PROV' if cm[0]['provisional'] else 'CERT'} n{cm[0]['n_entries']} "
                  f"r{cm[0]['tab_recall']:.3f}") if cm else "-"
            av = f"c{adv[0]['c_in_era']}({adv[0]['reason'][:4]})" if adv else "-"
            print(f"{a:15s} {k:3d} {len(idx):3d} {str(fc):>6s} {str(fv):>7s} {str(ft):>6s} "
                  f"{av:>10s} {cs:>26s}")

    # ---- 2. the matched-budget exam -------------------------------------------------------
    print("\n-- 2. TERMINAL ALL-ERAS EXAM at matched T (RANKS first; the spread is compared to "
          f"the depth-4 stream-position noise of +/-{STREAM_NOISE:.3f}) --")
    G = {a: ladder(r) for a, r in arms.items()}
    print(f"{'arm':15s} " + " ".join(f"{x:>8s}" for x in nm) + f" {'mean':>8s} {'rank':>5s}")
    rk = sorted(G, key=lambda b: np.mean(G[b]))
    for a in arms:
        print(f"{a:15s} " + " ".join(f"{x:8.4f}" for x in G[a])
              + f" {np.mean(G[a]):8.4f} {rk.index(a) + 1:5d}")
    sp = max(np.mean(G[a]) for a in G) - min(np.mean(G[a]) for a in G)
    print(f"   spread of arm means = {sp:.4f}  "
          f"{'<= stream noise: RANKING NOT INTERPRETABLE' if sp <= STREAM_NOISE else '> stream noise'}")

    # ---- 3. mechanism instruments ---------------------------------------------------------
    print("\n-- 3a. FIXED-REFERENCE POLICY PROBE (base action set only: vocabulary held "
          "constant, so what moves is policy/value alone) --")
    B = {a: ladder(r, "base_ms") for a, r in arms.items()}
    print(f"{'arm':15s} " + " ".join(f"{x:>8s}" for x in nm)
          + f" {'mean':>8s} | {'vocab gain (base - macros)':>26s}")
    for a in arms:
        gain = float(np.mean([B[a][j] - G[a][j] for j in range(n_e)]))
        print(f"{a:15s} " + " ".join(f"{x:8.4f}" for x in B[a])
              + f" {np.mean(B[a]):8.4f} | {gain:26.4f}")

    print("\n-- 3b. ENTRY-IDENTITY: does the mined table CHURN, or only GROW? --")
    print(f"{'arm':15s} {'lvl':>3s} {'final':>5s} {'ever added':>10s} {'ever REMOVED':>12s}")
    for a, r in arms.items():
        for lv in ("2", "3"):
            seqs = [set(map(tuple, c[lv]["keys_at_support"])) for c in r["log"]["miner"]
                    if lv in c and "keys_at_support" in c[lv]]
            if not seqs or not seqs[-1]:
                continue
            add = set(); rem = set()
            for x, y in zip(seqs, seqs[1:]):
                add |= (y - x); rem |= (x - y)
            print(f"{a:15s} {lv:>3s} {len(seqs[-1]):5d} {len(add):10d} {len(rem):12d}")

    # ---- 4. deep eras ---------------------------------------------------------------------
    print("\n-- 4. DEEP ERAS: separation against the range that exists there --")
    print(f"{'era':8s} {'spread':>8s} {'sd':>8s} {'stale-floor':>12s} {'spread/range':>13s}")
    for j in range(n_e):
        v = np.array([G[a][j] for a in arms])
        rng = refs["stale"][j] - refs["floor"][j]
        print(f"{nm[j]:8s} {v.max() - v.min():8.4f} {v.std():8.4f} {rng:12.3f} "
              f"{(v.max() - v.min()) / rng:13.3f}")

    # ---- 5. boundaries --------------------------------------------------------------------
    print("\n-- 5. BOUNDARY PLACEMENTS (cycles / % of T / reason) --")
    print(f"{'arm':15s} " + " ".join(f"{'e' + str(k):>17s}" for k in range(1, n_e + 1)))
    for a, r in arms.items():
        log = r["log"]; era = np.asarray(log["era"]); cells = []
        for k in range(1, n_e + 1):
            idx = np.where(era == k)[0]
            if not len(idx):
                cells.append(f"{'--':>17s}"); continue
            adv = [e for e in r["events"] if e["kind"] == "advance" and e["from_era"] == k]
            t0 = log["t_cum"][idx[0] - 1] if idx[0] else 0.0
            cells.append(f"{len(idx):3d}c/{100 * (log['t_cum'][idx[-1]] - t0) / cfg['t_budget']:4.1f}%"
                         f"{(adv[0]['reason'][:4] if adv else 'end'):>6s}")
        print(f"{a:15s} " + " ".join(cells))

    # ---- 6. instrument suite ---------------------------------------------------------------
    print("\n-- 6. INSTRUMENTS --")
    for a, r in arms.items():
        log = r["log"]
        rec = [e for e in r["events"] if e["kind"] == "recert"]
        sw = sum(int(bool(e["swapped"])) for e in rec)
        tot = 0.0
        for g in log["gcost"]:
            for c in g.values():
                tot += c["ground"] * cfg["d_fb"] + c["mat"] * cfg["c_mat"]
        mv = []
        for lv in ("2", "3"):
            d = [(log["aud"][i][lv].get("cand"), log["aud"][i][lv].get("rand_k"))
                 for i in range(len(log["cycle"])) if lv in log["aud"][i]]
            d = [(x, y) for x, y in d if x is not None and y is not None]
            if len(d) >= 5:
                mv.append(f"L{lv} {np.mean([x - y for x, y in d]):+.3f}")
        pl = [(p["cycle"], p["plant"]["parse_acc"], p["plant"]["infill_acc"])
              for p in log["probe"] if "plant" in p]
        print(f"{a:15s} recert={len(rec):3d} swaps={sw:2d} grader={100 * tot / log['t_cum'][-1]:5.1f}%"
              f"  mined-vs-random: {'  '.join(mv):24s} "
              f"plant {pl[0][1]:.3f}/{pl[0][2]:.3f}->{pl[-1][1]:.3f}/{pl[-1][2]:.3f}")
    return setup, arms, G, B


def figures(tag, setup, arms, G, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(FIG, tag)
    cfg, refs, eras = setup["config"], setup["refs"], setup["eras"]
    nm = [e["name"] for e in eras]
    colors = {a: c for a, c in zip(arms, plt.cm.tab10.colors)}

    # fig1 -- THE headline: competence against the two references it lives between
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for a, r in arms.items():
        log = r["log"]
        axes[0].plot(log["t_cum"], log["e"], color=colors[a], lw=1.2, label=a)
        for ev in r["events"]:
            if ev["kind"] == "advance":
                axes[0].axvline(ev["t_cum"], color=colors[a], ls="--", alpha=0.3)
    era_t = {}
    for a, r in arms.items():
        era = np.asarray(r["log"]["era"])
        for k in range(1, len(eras) + 1):
            idx = np.where(era == k)[0]
            if len(idx):
                era_t.setdefault(k, []).append(
                    (r["log"]["t_cum"][idx[0]], r["log"]["t_cum"][idx[-1]]))
    for k, seg in era_t.items():
        lo = min(s[0] for s in seg); hi = max(s[1] for s in seg)
        axes[0].hlines(refs["stale"][k - 1], lo, hi, color="k", lw=1.4)
        axes[0].hlines(refs["floor"][k - 1], lo, hi, color="tab:green", lw=1.4)
    axes[0].set_xlabel("cumulative priced time"); axes[0].set_ylabel("e on the CURRENT era")
    axes[0].set_title(f"{tag}: competence never left the stale baseline\n"
                      "(black = stale/never-commit, green = exact-DP floor)")
    axes[0].legend(fontsize=7); axes[0].set_ylim(0, 1)

    x = np.arange(len(nm)); w = 0.8 / max(len(arms), 1)
    for i, a in enumerate(arms):
        axes[1].bar(x + i * w, G[a], width=w, color=colors[a], label=a)
    axes[1].plot(x + 0.4 - w / 2, refs["stale"], "k_", ms=26, mew=2, label="stale")
    axes[1].plot(x + 0.4 - w / 2, refs["floor"], "_", color="tab:green", ms=26, mew=2,
                 label="floor")
    axes[1].set_xticks(x + 0.4 - w / 2, nm); axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("e on that era's held-out set")
    axes[1].set_title("terminal all-eras exam at matched T")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_regime.png"), dpi=130)

    # fig2 -- the fixed-reference probe: policy quality with vocabulary held constant
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, a in enumerate(arms):
        axes[0].bar(x + i * w, B[a], width=w, color=colors[a], label=a)
    axes[0].set_xticks(x + 0.4 - w / 2, nm); axes[0].set_ylim(0, 1)
    axes[0].set_title("fixed-reference policy probe (BASE action set only)")
    axes[0].set_ylabel("e"); axes[0].legend(fontsize=6)
    for i, a in enumerate(arms):
        axes[1].bar(x + i * w, [B[a][j] - G[a][j] for j in range(len(nm))], width=w,
                    color=colors[a])
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(x + 0.4 - w / 2, nm)
    axes[1].set_ylabel("base-only error minus with-macros error")
    axes[1].set_title("what the earned vocabulary is worth (positive = macros help)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_policy_vs_vocab.png"), dpi=130)

    # fig3 -- the pacing signals and where each arm advanced
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for a, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], [c["A"] if c["A"] is not None else np.nan
                                    for c in log["cert"]], color=colors[a], lw=1.1, label=a)
        axes[1].plot(log["cycle"], [c["A"] if c["A"] is not None else np.nan
                                    for c in log["vcert"]], color=colors[a], lw=1.1)
        for ev in r["events"]:
            if ev["kind"] == "advance":
                for ax in axes:
                    ax.axvline(ev["cycle"], color=colors[a], ls=":", alpha=0.5)
    axes[0].set_ylabel("unit-LP audition (mfg/pol)"); axes[0].legend(fontsize=6)
    axes[0].set_title(f"{tag}: the two endogenous signals (dotted = that arm's advancement)")
    axes[1].set_ylabel("novel-tuple admission rate"); axes[1].set_xlabel("cycle")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_signals.png"), dpi=130)

    # fig4 -- era occupancy: the shape each arm actually ran
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(arms)
    bot = np.zeros(len(names))
    for k in range(1, len(eras) + 1):
        share = []
        for a in names:
            era = np.asarray(arms[a]["log"]["era"])
            idx = np.where(era == k)[0]
            share.append(len(idx) / len(era) if len(idx) else 0.0)
        ax.barh(names, share, left=bot, label=nm[k - 1])
        bot += np.array(share)
    ax.set_xlabel("fraction of the arm's cycles"); ax.legend(fontsize=7, ncol=5)
    ax.set_title(f"{tag}: the curriculum shape each policy actually ran")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_shapes.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    setup, arms, G, B = report(a.tag)
    if a.figures:
        figures(a.tag, setup, arms, G, B)


if __name__ == "__main__":
    main()
