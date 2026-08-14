"""Reduce a crystallize run: priced grade, commit events, the audition gap, the
certificate-vs-floor oracle check, and the compile-op discriminator.

    python3 rhm/practice/crystallize/analyze_crystallize.py --tag cg_s0 --fetch --figures
    python3 rhm/practice/crystallize/analyze_crystallize.py --tag calu_s0 --cal
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_crystallize"


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
    return setup, arms


def interp(t_series, y_series, t_at):
    """Interpolate an arm's own trajectory to a priced time (the etude's matched-time idiom)."""
    t, y = np.asarray(t_series, float), np.asarray(y_series, float)
    if t_at <= t[0]:
        return float(y[0])
    if t_at >= t[-1]:
        return float(y[-1])
    return float(np.interp(t_at, t, y))


def report(tag):
    setup, arms = load(tag)
    ctxs = setup["contexts"]
    refs = setup["refs"]
    names = [c["name"] for c in ctxs]
    print(f"\n=== {tag} ===")
    print(f"contexts: " + " | ".join(
        f"{c['name']} (L{c['level']} n{c['nodes']}): d0={refs['d0'][i]:.2f} "
        f"stale={refs['stale'][i]:.3f} floor={refs['floor'][i]:.3f} wide={refs['wide'][i]:.3f}"
        for i, c in enumerate(ctxs)))
    print(f"groundings/solve: beam={np.round(refs['ground_beam'], 1).tolist()} unit=2.0")

    print("\n-- priced grade (last cycle) --")
    rows = {}
    for arm, r in arms.items():
        log = r["log"]
        e = np.array(log["e"], float)
        rows[arm] = {"t": log["t_cum"][-1], "e": e[-1].mean(), "per_ctx": e[-1],
                     "t_series": log["t_cum"], "e_series": e.mean(1),
                     "cycles": log["cycle"][-1], "complete": r["complete"],
                     "gps": np.array(log["ground_per_solve"], float)[-1]}
    hdr = f"{'arm':14s} {'cycles':>6s} {'t_cum':>12s} {'e_mean':>8s} " + \
          " ".join(f"{'e_' + n:>10s}" for n in names) + f" {'g/solve':>16s}"
    print(hdr)
    for arm, x in rows.items():
        print(f"{arm:14s} {x['cycles']:6d} {x['t']:12.0f} {x['e']:8.4f} " +
              " ".join(f"{v:10.4f}" for v in x['per_ctx']) +
              "  " + "/".join(f"{g:.0f}" for g in x["gps"]) +
              ("" if x["complete"] else "   [INCOMPLETE]"))

    if "never" in rows:
        base = rows["never"]
        print("\n-- matched priced time (never's own trajectory interpolated) --")
        for arm, x in rows.items():
            if arm == "never":
                continue
            n_at = interp(base["t_series"], base["e_series"], x["t"])
            print(f"{arm:14s} t={x['t']:10.0f}  e={x['e']:.4f}  never@same_t={n_at:.4f}  "
                  f"margin={n_at - x['e']:+.4f}   never_best_ever={min(base['e_series']):.4f}")

    print("\n-- compile events --")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "compile":
                continue
            k = ev["ctx"]
            u = ev["unit"]
            desc = u.get("seq") if u["kind"] == "seq" else f"lib[{len(u['by_root'])}]"
            print(f"{arm:14s} ctx={ev['ctx_name']:8s} c{ev['cycle']:<3d} key={ev['key']:6s} "
                  f"e@commit={ev['e']:.3f} floor={refs['floor'][k]:.3f} "
                  f"unit={desc} in_sample={ev['chosen_score']:.3f} "
                  f"audition={ev['audition']:.3f} med={ev['score_med']:.3f} "
                  f"own_ok={ev['score_own_ok']:.3f} best1={ev['score_best_single']:.3f} "
                  f"pool={ev['n_pool']}/{ev['n_distinct']}")
        for ev in r["events"]:
            if ev["kind"] != "anchor":
                continue
            k = ev["ctx"]
            cm = next((c for c in r["events"] if c["kind"] == "compile" and c["ctx"] == k), None)
            g1 = ev["level"] / cm["audition"] if (cm and cm.get("audition")) else None
            g2 = ev["level"] / cm["chosen_score"] if (cm and cm.get("chosen_score")) else None
            print(f"{arm:14s} ctx={k} anchor c{ev['cycle']} level={ev['level']:.4f}" +
                  (f"  gap vs held-out audition={g1:.2f}  vs in-sample={g2:.2f}" if g1 else ""))

    print("\n-- compile-hit discriminator (delta_gate certificate state) --")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "discriminate":
                continue
            print(f"  ctx={ev['ctx_name']} c{ev['cycle']}")
            for key in sorted(ev):
                cell = ev[key]
                if isinstance(cell, dict) and "e" in cell:
                    extra = ""
                    if "on_grammar" in cell:
                        extra = (f"  on_grammar={cell['on_grammar']:.3f} "
                                 f"(realisations {cell['on_grammar_realisations']:.3f})")
                    if "seq" in cell:
                        extra += f"  seq={cell['seq']}"
                    if "audition" in cell:
                        extra += f"  audition={cell['audition']:.3f}"
                    if "ground_per_solve" in cell:
                        extra += f"  g/solve={cell['ground_per_solve']:.0f}"
                    print(f"    {key:18s} e={cell['e']:.4f}  d*={cell['dres']:.3f}{extra}")
    return setup, arms, rows


def cal_report(tag):
    root = os.path.join(FIG, tag)
    d = json.load(open(os.path.join(root, "cal_unit.json")))
    print(f"\n=== {tag} (compilation headroom) ===")
    for name, cell in d["cells"].items():
        print(f"\n-- ctx {name} (d0={cell['d0']:.2f}) --")
        print("  beam: " + " | ".join(
            f"w{w}: e={c['e']:.3f} ({c['ground_per_solve']:.0f}g)"
            for w, c in cell["beam"].items()))
        print(f"  dp_oracle e={cell['dp_oracle']:.3f}      committed unit costs 2g")
        for ln in ("len1", "len2", "len3"):
            if ln not in cell:
                continue
            c = cell[ln]
            print(f"  {ln}: best_in={c['best_in']:.3f} {c['best_in_prog']} | "
                  f"best_out={c['best_out']:.3f} {c['best_out_prog']} | med={c['median']:.3f} | "
                  f"lib_r* in={c['lib_in']:.3f} out={c['lib_out']:.3f}")
        top = sorted(cell["per_move"].items(), key=lambda kv: kv[1])[:5]
        print("  best single moves: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
    return d


def replay_detector(e_series, stale, *, alpha, c, cv, W, hold, min_cycle):
    """Replay the delta-silence detector over a recorded e-series, exactly as `run_arm` does.
    Returns (fire_cycle or None, e at fire)."""
    b = emin = None
    eh, dh, run = [], [], 0
    for i, e in enumerate(e_series, start=1):
        if b is None:
            b = emin = e
        emin = min(emin, e)
        d = b - e
        scale = max(stale - emin, b, 1e-6)
        eh.append(e); dh.append(d)
        we, wd = eh[-W:], dh[-W:]
        quiet = (len(we) >= W and abs(float(np.mean(wd))) < c * scale
                 and float(np.std(we)) < cv * scale)
        run = run + 1 if quiet else 0
        b = b + alpha * (e - b)
        if run >= hold and i >= min_cycle:
            return i, e
    return None, None


def detector_grid(tag, arm=None):
    """CALIBRATION 3: what (c, c_v, W, hold) makes the certificate fire at mastery rather than
    mid-descent? Replayed offline over a recorded descent, so it costs no GPU."""
    setup, arms = load(tag)
    refs = setup["refs"]
    ctxs = setup["contexts"]
    alpha = setup["config"]["alpha"]
    picked = {arm: arms[arm]} if arm else arms
    for a, r in picked.items():
        e = np.array(r["log"]["e"], float)
        print(f"\n=== detector replay: {tag}/{a} ===")
        for k, ctx in enumerate(ctxs):
            series = e[:, k]
            fin = float(series[-5:].mean())
            print(f"-- ctx {ctx['name']}: e {series[0]:.3f} -> {fin:.3f} "
                  f"(min {series.min():.3f}), stale={refs['stale'][k]:.3f}, "
                  f"floor={refs['floor'][k]:.3f}")
            print(f"   {'c':>6s} {'c_v':>6s} {'W':>3s} {'hold':>4s} {'fire@':>6s} "
                  f"{'e@fire':>7s} {'vs final':>9s}")
            for c in (0.03, 0.06, 0.10):
                for cv in (0.05, 0.10, 0.20):
                    for W in (5, 7):
                        for hold in (2, 3):
                            fc, fe = replay_detector(series, refs["stale"][k], alpha=alpha,
                                                     c=c, cv=cv, W=W, hold=hold, min_cycle=4)
                            if fc is None:
                                continue
                            print(f"   {c:6.2f} {cv:6.2f} {W:3d} {hold:4d} {fc:6d} "
                                  f"{fe:7.3f} {fe - fin:+9.3f}")
    return picked


def figures(tag, setup, arms, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    ctxs = setup["contexts"]
    refs = setup["refs"]
    K = len(ctxs)
    colors = {a: c for a, c in zip(arms, plt.cm.tab10.colors)}

    fig, axes = plt.subplots(2, K, figsize=(6 * K, 7), squeeze=False)
    for k in range(K):
        for arm, r in arms.items():
            log = r["log"]
            e = np.array(log["e"], float)[:, k]
            d = np.array(log["delta"], float)[:, k]
            axes[0][k].plot(log["cycle"], e, label=arm, color=colors[arm])
            axes[1][k].plot(log["cycle"], d, label=arm, color=colors[arm])
            for ev in r["events"]:
                if ev["kind"] == "compile" and ev["ctx"] == k:
                    axes[0][k].axvline(ev["cycle"], color=colors[arm], ls=":", alpha=0.7)
        axes[0][k].axhline(refs["stale"][k], color="grey", ls="--", label="stale ref")
        axes[0][k].axhline(refs["floor"][k], color="black", ls="-.", label="DP-oracle floor")
        axes[0][k].set_title(f"{ctxs[k]['name']}: e (1 - success) @ performance conditions")
        axes[1][k].axhline(0, color="black", lw=0.5)
        axes[1][k].set_title(f"{ctxs[k]['name']}: delta = b - e")
        axes[1][k].set_xlabel("cycle")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_metering.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, x in rows.items():
        ax.plot(x["t_series"], x["e_series"], marker="o", ms=3, label=arm, color=colors[arm])
    ax.set_xlabel("cumulative priced time"); ax.set_ylabel("mean e over contexts")
    ax.set_title(f"{tag}: priced grade"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_priced.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, r in arms.items():
        g = np.array(r["log"]["ground_per_solve"], float).mean(1)
        ax.semilogy(r["log"]["cycle"], g, label=arm, color=colors[arm])
    ax.set_xlabel("cycle"); ax.set_ylabel("groundings / solve (mean over contexts)")
    ax.set_title(f"{tag}: performance-time feedback cost"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_ground.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    if a.cal:
        cal_report(a.tag)
        return
    if a.detector:
        detector_grid(a.tag, a.arm)
        return
    setup, arms, rows = report(a.tag)
    if a.figures:
        figures(a.tag, setup, arms, rows)


if __name__ == "__main__":
    main()
