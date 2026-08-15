"""Reduce a ratchet run.

    python3 rhm/practice/ratchet/analyze_ratchet.py --tag rr_s0 --fetch --figures
    python3 rhm/practice/ratchet/analyze_ratchet.py --tag call_s0 --fetch --cal

Readouts, in the order the round asks for them:
  1. priced cost-to-competence per depth era, per arm  -- the cost-to-depth curve
  2. the EARNED-VS-GIVEN fraction: how much of `given`'s advantage over `never_base` does
     `practice_gated` recover by earning the vocabulary
  3. commit events, table precision/recall against the DGP's own vocabulary, and the
     early-vs-gated poison comparison at matched pricing
  4. unit-LP (shadow-audition) curves per era, with the certificate's firing marked, against
     the true-table floor on the same instances
  5. compounding: the era-2 level-3 descent rate given each arm's level-2 vocabulary
  6. the plant guard (parse / infill accuracy on clean held-out configs)
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_ratchet"
ORDER = ["never_base", "given", "practice_gated", "practice_early", "practice_late"]


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if os.path.isfile(p):
            arms[arm] = json.load(open(p))
    keyed = {a: arms[a] for a in ORDER if a in arms}
    keyed.update({a: r for a, r in arms.items() if a not in keyed})
    return setup, keyed


def interp(t_series, y_series, t_at):
    t, y = np.asarray(t_series, float), np.asarray(y_series, float)
    if t_at <= t[0]:
        return float(y[0])
    if t_at >= t[-1]:
        return float(y[-1])
    return float(np.interp(t_at, t, y))


def era_slices(log):
    era = np.asarray(log["era"], int)
    return {int(k): np.nonzero(era == k)[0] for k in np.unique(era)}


def report(tag, tail=5):
    setup, arms = load(tag)
    eras, refs = setup["eras"], setup["refs"]
    print(f"\n=== {tag} ===")
    print("eras: " + " | ".join(
        f"{e['name']} d0={refs['d0'][i]:.2f} stale_base={refs['stale'][i]:.3f} "
        f"stale_true={refs['stale_true'][i]:.3f} floor={refs['floor'][i]:.3f} "
        f"macro_true={refs['macro_true'][i]}" for i, e in enumerate(eras)))
    print(f"declared budget: base w={refs['width_base']} ({refs['g_base']}g), "
          f"true w={refs['width_true']} ({refs['g_true']}g)")

    # ---- 1. priced cost-to-competence per era -----------------------------------------
    print(f"\n-- per-era competence (mean e over the last {tail} cycles of the era) "
          "and priced time --")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        e = np.asarray(log["e"], float)
        t = np.asarray(log["t_cum"], float)
        per[arm] = {k: {"e": float(e[idx[-tail:]].mean()),
                        "e_last": float(e[idx[-1]]),
                        "t_end": float(t[idx[-1]]),
                        "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                        "nm": int(log["n_moves"][idx[-1]]),
                        "w": int(log["width"][idx[-1]])}
                    for k, idx in sl.items()}
        per[arm]["_t"] = t
        per[arm]["_e"] = e
        per[arm]["_complete"] = r["complete"]
    ks = sorted(era_slices(list(arms.values())[0]["log"]).keys())
    hdr = f"{'arm':16s} " + " ".join(f"{'era' + str(k):>22s}" for k in ks) + f"{'t_cum':>12s}"
    print(hdr)
    for arm, x in per.items():
        cells = " ".join(f"  e={x[k]['e']:.4f} (t={x[k]['t_era']:8.0f}, nm{x[k]['nm']}w{x[k]['w']})"
                         for k in ks)
        print(f"{arm:16s} {cells} {x[ks[-1]]['t_end']:12.0f}"
              + ("" if x["_complete"] else "  [INCOMPLETE]"))

    # ---- 2. the earned-vs-given fraction ------------------------------------------------
    if "never_base" in per and "given" in per:
        print("\n-- EARNED-VS-GIVEN fraction per era:  (e_never - e_arm) / (e_never - e_given) --")
        print(f"{'arm':16s} " + " ".join(f"{'era' + str(k):>10s}" for k in ks))
        for arm in per:
            if arm in ("never_base", "given"):
                continue
            row = []
            for k in ks:
                den = per["never_base"][k]["e"] - per["given"][k]["e"]
                row.append((per["never_base"][k]["e"] - per[arm][k]["e"]) / den
                           if abs(den) > 1e-9 else float("nan"))
            print(f"{arm:16s} " + " ".join(f"{x:10.3f}" for x in row))
        print("   (denominators: " + " ".join(
            f"era{k}={per['never_base'][k]['e'] - per['given'][k]['e']:+.4f}" for k in ks) + ")")

        print("\n-- matched priced time (each arm's e vs never_base's own trajectory at the "
              "same t) --")
        for arm, x in per.items():
            if arm == "never_base":
                continue
            for k in ks:
                nb = interp(per["never_base"]["_t"], per["never_base"]["_e"], x[k]["t_end"])
                print(f"{arm:16s} era{k}  t={x[k]['t_end']:10.0f}  e={x[k]['e_last']:.4f}  "
                      f"never@t={nb:.4f}  margin={nb - x[k]['e_last']:+.4f}")

    # ---- 3. commits, the earned table, and the poison comparison ------------------------
    print("\n-- commit events (an earned vocabulary against the DGP's own) --")
    print(f"{'arm':16s} {'era':>3s} {'lvl':>3s} {'cyc':>4s} {'c_in':>4s} {'entries':>7s} "
          f"{'recall':>7s} {'prec':>6s} {'audition':>8s} {'shadow':>7s} {'e_task':>7s} {'sil':>4s}")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "commit":
                continue
            print(f"{arm:16s} {ev['era']:3d} {ev['level']:3d} {ev['cycle']:4d} "
                  f"{ev['c_in_era']:4d} {ev['n_entries']:7d} {ev['tab_recall']:7.3f} "
                  f"{(ev['tab_precision'] if ev['tab_precision'] is not None else float('nan')):6.3f} "
                  f"{ev['audition']:8.4f} "
                  f"{(ev['shadow'] if ev['shadow'] is not None else float('nan')):7.4f} "
                  f"{ev['e_task']:7.4f} {ev['sil_run']:4d}")

    # the optimism gap: commit audition (on era k) vs realised competence (in era k+1)
    print("\n-- the depth seam: commit audition (auditioned on era k's damage) vs the arm's "
          "realised competence in era k+1, where the macro is actually consumed --")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "commit":
                continue
            nxt = ev["era"] + 1
            if nxt not in per[arm]:
                continue
            print(f"{arm:16s} L{ev['level']} committed era{ev['era']} audition="
                  f"{ev['audition']:.4f}  ->  era{nxt} competence={per[arm][nxt]['e']:.4f}  "
                  f"gap={per[arm][nxt]['e'] / max(ev['audition'], 1e-6):.2f}")

    # ---- 4. the unit-LP curves ----------------------------------------------------------
    print("\n-- unit-LP: the shadow audition of the candidate macro (A), the true-table floor "
          "on the same instances, and the earned table's recall --")
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        fired = {ev["era"]: ev["cycle"] for ev in r["events"] if ev["kind"] == "commit"}
        for k, idx in sl.items():
            lvl = int(log["level"][idx[0]]) + 1
            A = [log["aud"][i].get(str(lvl), {}).get("cand") for i in idx]
            T = [log["aud"][i].get(str(lvl), {}).get("true") for i in idx]
            R = [log["aud"][i].get(str(lvl), {}).get("tab_recall") for i in idx]
            if all(a is None for a in A):
                continue
            fmt = lambda xs: " ".join("  . " if x is None else f"{x:.2f}" for x in xs)
            print(f"{arm:16s} era{k} L{lvl}  A    : {fmt(A)}"
                  + (f"   [commit c{fired[k]}]" if k in fired else ""))
            print(f"{'':16s}      true : {fmt(T)}")
            print(f"{'':16s}      recall:{fmt(R)}")

    # ---- 5. compounding -----------------------------------------------------------------
    print("\n-- compounding: how fast the level-3 audition descends in era 2, per arm's "
          "level-2 vocabulary --")
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        if 2 not in sl:
            continue
        idx = sl[2]
        A = np.array([log["aud"][i].get("3", {}).get("cand") or np.nan for i in idx], float)
        ok = ~np.isnan(A)
        if ok.sum() < 3:
            print(f"{arm:16s} era2: no level-3 candidate")
            continue
        first = int(np.nonzero(ok)[0][0])
        half = A[ok][0] - 0.5 * (A[ok][0] - np.nanmin(A))
        reach = np.nonzero(ok & (A <= half))[0]
        v2 = log["vocab"][idx[0]].get("2")
        print(f"{arm:16s} era2: L2 vocab at era start={v2}  A3 first={A[ok][0]:.3f} "
              f"(c_in_era {first + 1})  min={np.nanmin(A):.3f}  "
              f"half-descent at c_in_era={(reach[0] + 1) if len(reach) else None}  "
              f"final recall={log['aud'][idx[-1]].get('3', {}).get('tab_recall')}")

    # ---- 6. the plant guard --------------------------------------------------------------
    print("\n-- plant guard (clean held-out): parse / infill feature accuracy --")
    for arm, r in arms.items():
        pr = [(p["cycle"], p["plant"]["parse_acc"], p["plant"]["infill_acc"])
              for p in r["log"]["probe"] if "plant" in p]
        if not pr:
            continue
        print(f"{arm:16s} " + " ".join(f"c{c}:{a:.3f}/{b:.3f}" for c, a, b in pr[::max(1, len(pr) // 8)]))
    return setup, arms, per


def cal_report(tag):
    root = os.path.join(FIG, tag)
    d = json.load(open(os.path.join(root, "cal_ladder.json")))
    print(f"\n=== {tag} (the depth ladder) ===")
    print(f"action-set sizes: {d['n_moves']}")
    for name, cell in d["cells"].items():
        print(f"\n-- era {name} (d0={cell['d0']:.2f}, on_grammar={cell['on_grammar']:.3f}) --")
        for t in ("base", "l2", "true"):
            if t in cell:
                print(f"  {t:5s}: " + " | ".join(
                    f"w{w}: e={c['e']:.3f} ({c['g']:.0f}g)" for w, c in
                    sorted(cell[t].items(), key=lambda kv: int(kv[0]))))
        print(f"  one true macro action: " + " ".join(
            f"L{k}={x:.3f}" for k, x in cell["macro_true"].items())
            + f"   DP floor (base) = {cell['dp_floor_base']:.3f}")
        for ell, row in cell.get("coverage", {}).items():
            print(f"  coverage L{ell}: " + " ".join(
                f"{k}:{c['mean']:.3f}" for k, c in sorted(row.items(), key=lambda kv: int(kv[0]))))
        for g, row in sorted(cell["at_budget"].items(), key=lambda kv: int(kv[0])):
            print(f"  @G={int(g):4d}: " + " | ".join(
                f"{t} w{row[t]['w']}({row[t]['g']}g) e={row[t]['e']:.3f}"
                for t in row if row[t]["e"] is not None))
    return d


def figures(tag, setup, arms, per):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    eras = setup["eras"]
    colors = {a: c for a, c in zip(arms, plt.cm.tab10.colors)}
    bounds = np.cumsum([len(era_slices(list(arms.values())[0]["log"])[k])
                        for k in sorted(era_slices(list(arms.values())[0]["log"]))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], log["e"], label=arm, color=colors[arm])
        axes[1].plot(log["t_cum"], log["e"], label=arm, color=colors[arm])
        for ev in r["events"]:
            if ev["kind"] == "commit":
                axes[0].axvline(ev["cycle"], color=colors[arm], ls=":", alpha=0.6)
    for b in bounds[:-1]:
        axes[0].axvline(b + 0.5, color="grey", lw=0.8)
    axes[0].set_xlabel("cycle"); axes[0].set_ylabel("e = 1 - success @ declared budget")
    axes[0].set_title(f"{tag}: competence, eras " + " -> ".join(e["name"] for e in eras))
    axes[1].set_xlabel("cumulative priced time"); axes[1].set_ylabel("e")
    axes[1].set_title("priced cost to competence")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm, r in arms.items():
        log = r["log"]
        for j, lvl in enumerate((2, 3)):
            A = [c.get(str(lvl), {}).get("cand") for c in log["aud"]]
            T = [c.get(str(lvl), {}).get("true") for c in log["aud"]]
            axes[j].plot(log["cycle"], [np.nan if x is None else x for x in A],
                         color=colors[arm], label=arm)
            if arm == "given":
                axes[j].plot(log["cycle"], [np.nan if x is None else x for x in T],
                             color="black", ls="--", lw=1, label="true-table floor")
            for ev in r["events"]:
                if ev["kind"] == "commit" and ev["level"] == lvl:
                    axes[j].axvline(ev["cycle"], color=colors[arm], ls=":", alpha=0.7)
            axes[j].set_title(f"level-{lvl} macro: shadow audition (unit-LP)")
            axes[j].set_xlabel("cycle"); axes[j].set_ylabel("audition e")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_unitlp.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for arm, r in arms.items():
        log = r["log"]
        for j, lvl in enumerate((2, 3)):
            R = [c.get(str(lvl), {}).get("tab_recall") for c in log["aud"]]
            axes[j].plot(log["cycle"], [np.nan if x is None else x for x in R],
                         color=colors[arm], label=arm)
            axes[j].set_title(f"level-{lvl} table recall vs the DGP's own vocabulary")
            axes[j].set_xlabel("cycle"); axes[j].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_vocab.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    if a.cal:
        cal_report(a.tag)
        return
    setup, arms, per = report(a.tag, tail=a.tail)
    if a.figures:
        figures(a.tag, setup, arms, per)


if __name__ == "__main__":
    main()


def replay_unitlp(A, *, alpha, c, cv, W, hold, min_cycle, min_drop):
    """Replay the unit-LP certificate over a recorded shadow-audition series, exactly as
    `run_arm` runs it: silence on the committable content's trajectory PLUS the explicit
    'it descended first' precondition. Offline, so the detector grid costs no GPU."""
    b = ref = emin = None
    hist, dh, run = [], [], 0
    for i, a in enumerate(A, start=1):
        if a is None or (isinstance(a, float) and a != a):
            continue
        if ref is None:
            ref = b = emin = a
        emin = min(emin, a)
        d = b - a
        scale = max(ref - emin, b, 1e-6)
        hist.append(a); dh.append(d)
        we, wd = hist[-W:], dh[-W:]
        quiet = (len(we) >= W and abs(float(np.mean(wd))) < c * scale
                 and float(np.std(we)) < cv * scale)
        run = run + 1 if quiet else 0
        b = b + alpha * (a - b)
        if run >= hold and i >= min_cycle and (ref - emin) >= min_drop:
            return i, a
    return None, None


def unitlp_grid(tags, level=2, era=1):
    import itertools
    for tag in tags.split(","):
        setup, arms = load(tag)
        alpha = setup["config"]["alpha"]
        print(f"\n=== unit-LP detector replay: {tag} (era {era}, level {level}) ===")
        for a, r in arms.items():
            log = r["log"]
            idx = np.nonzero(np.asarray(log["era"], int) == era)[0]
            A = [log["aud"][i].get(str(level), {}).get("cand") for i in idx]
            vals = [x for x in A if x is not None]
            if len(vals) < 6:
                continue
            span = max(vals[:1]) - min(vals)
            print(f"-- {a}: first={vals[0]:.3f} min={min(vals):.3f} span={span:.3f} "
                  f"last5={np.mean(vals[-5:]):.3f}")
            for c, cv, W, hold, md in itertools.product((0.06, 0.10), (0.10, 0.15, 0.20),
                                                        (5, 7), (2, 3), (0.05, 0.10)):
                fc, fe = replay_unitlp(A, alpha=alpha, c=c, cv=cv, W=W, hold=hold,
                                       min_cycle=6, min_drop=md)
                if fc is None:
                    continue
                print(f"   c={c:.2f} cv={cv:.2f} W={W} hold={hold} drop={md:.2f} -> "
                      f"fire c{fc:2d}  A={fe:.3f}  vs min {fe - min(vals):+.3f}  "
                      f"vs last5 {fe - np.mean(vals[-5:]):+.3f}")
