"""Reduce prestissimo's Phase A run (the fast piece + the static level ladder) to a text report
and figures.

Pure python for the reduction (json + math only): the local client has neither numpy nor
matplotlib (`offbook/FILES.md` Gotcha). `--figures` renders remotely on Modal and fetches PNGs.

    python3 mjc/practice/tempo/prestissimo/analyze_ladder.py --tag a0 --fetch
    python3 mjc/practice/tempo/prestissimo/analyze_ladder.py --tag a0 --figures
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# `python3 mjc/practice/tempo/prestissimo/analyze_ladder.py` puts THIS directory on sys.path, not
# `experiments/`, so the package import below would fail; `modal run` does put it there. Adding it
# explicitly makes both entry paths work (solo/analyze_solo.py's line, verbatim in intent).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from mjc.shared import app, volume, DATA_DIR   # noqa: E402

VOL = "mujoco-control-data"


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL,
                    f"/practice_prestissimo/{tag}/ladder.json",
                    os.path.join(dst, "ladder.json"), "--force"], check=True)
    return os.path.join(dst, "ladder.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    dt = d.get("piece", {}).get("dt_ctrl", 0.024)
    P(f"=== prestissimo Phase A — the fast piece and its level ladder, tag {cfg['tag']}, "
      f"seed {cfg['seed']} ({'COMPLETE' if d.get('complete') else 'INCOMPLETE'}, "
      f"{d.get('wall_s', 0):.0f}s wall) ===")
    P("no forward model anywhere; no learned component at all in Phase A; single seed")

    # ---------------------------------------------------------------- gates
    P("\n--- gates ---")
    P(f"P-F0 donor constants: {d['P_F0']['checked']} checked, "
      f"{len(d['P_F0']['mismatches'])} mismatched, pass={d['P_F0']['pass']}")
    b = d.get("P_F1", {})
    if b.get("applicable"):
        P(f"P-F1 the fork on the DONOR square vs acappella b1: {b['n_checks']} checks, "
          f"max|Δ| = {b['max_abs']:.3e}  gains_match={b['gains_match']}  pass={b['pass']}")
        for k, v in sorted(b.items()):
            if k.endswith("_max_abs"):
                P(f"       {k[:-8]:22s} {v:.3e}")
    else:
        P(f"P-F1 NOT DEFINED here ({b.get('note', 'seed != 0')})")
    n = d.get("P_N", {})
    P(f"P-N  nesting: {n.get('n_checked', 0)} level-(l>1) members spelled by their two parents "
      f"({n.get('spell_bad')} bad); weld == halves in sequence max|Δ| = "
      f"{n.get('exec_max_abs', float('nan')):.3e}  pass={n.get('pass')}")
    a = d.get("P_A", {})
    L = int(d["piece"]["n_levels"])
    if a:
        P(f"P-A  the approach removed acappella finding 5's artifact: the deepest committed arm "
          f"(key_L{L}) spreads {a[f'key_L{L}']['spread']:.4f} "
          f"({a[f'key_L{L}']['ratio']:.2f}×) over the Δ ladder  pass={a.get('pass')}")
        for pre in ("key", "aud"):
            for e in range(1, L + 1):
                r = a[f"{pre}_L{e}"]
                P(f"       {pre}_L{e}: {r['ratio']:.2f}×  "
                  f"[{', '.join(f'{x:.4f}' for x in r['vals'])}]")

    # ---------------------------------------------------------------- the calibration
    t = d["P_T"]
    P(f"\n--- P-T: the tempo calibration (R × Δ on the reflex alone; rule fixed before the grid "
      f"was read) ---")
    P(f"{'R':>6s} {'leg':>7s} {'v m/s':>7s} {'band':>7s} " + " ".join(
        f"{'e@' + str(x):>8s}" for x in cfg["cal_deltas"])
      + f" {'range':>7s} {'comp':>5s} {'dem':>5s}")
    for r in t["rows"]:
        P(f"{r['R']:6.3f} {r['mean_leg']:7.4f} {r['speed']:7.2f} {r['band']:7.4f} "
          + " ".join(f"{r[f'rot@{x}']:8.4f}" for x in cfg["cal_deltas"])
          + f" {r['usable_range']:6.1f}× {str(r['competence']):>5s} {str(r['demand']):>5s}")
    P(f"rule: {t['rule']}  ->  R* = {t['R_star']:.3f}"
      + ("  (FORCED by flag)" if t.get("forced") else ""))

    pc = d["piece"]
    P(f"\n--- the piece: {pc['name']} ---")
    P(f"{pc['K_drill']} drilled segments × {pc['H']} steps ({pc['seg_ms']:.0f} ms) + "
      f"{pc['k0']} approach segments; mean leg {pc['mean_leg']:.4f} m; "
      f"mean tip speed {pc['speed']:.2f} m/s; {pc['n_levels']} rungs")
    P(f"legs   {[round(x, 4) for x in pc['legs']]}")
    P(f"turns  {pc['turns']} deg")
    P(f"band   {pc['band']:.4f} = max(½·leg {pc['half_leg']:.4f}, "
      f"ref_play scaled {pc['ref_play_scaled']:.4f})")

    P("\n--- the reflex law, re-fit at every Δ on the CLEAN world (every advantage to the "
      "incumbent) ---")
    P(f"{'Δ':>3s} {'ms':>5s} {'kp':>7s} {'kd':>6s} {'clean e':>9s} {'edges':>10s}")
    for k in sorted(d["reflex_cal"], key=lambda x: int(x)):
        v = d["reflex_cal"][k]
        P(f"{int(k):3d} {int(k) * dt * 1000:5.0f} {v['kp']:7.1f} {v['kd']:6.2f} "
          f"{v['e_piece']:9.4f} {'kp' if v['at_kp_edge'] else '':>5s}"
          f"{'kd' if v['at_kd_edge'] else '':>5s}")

    # ---------------------------------------------------------------- the library
    lb = d["library"]
    P(f"\n--- the ladder, built statically (harvest gains frozen at the Δ=0 fit "
      f"{lb['harvest_gains']}) ---")
    P(f"slots per level: {lb['counts']}   {lb['build_ledger']['n_ground']:.0f} groundings/perf, "
      f"{lb['build_ledger']['t_ground']:.0f} s priced")
    P(f"{'lvl':>4s} {'seam':>5s} {'chosen':>8s} {'median':>8s} {'worst':>8s} {'pairs':>6s} "
      f"{'distinct a/b':>13s}")
    for r in lb["level1"]:
        P(f"{1:4d} {r['drilled']:5d} {r['chosen_score']:8.4f} {r['score_med']:8.4f} "
          f"{r['score_max']:8.4f} {'':>6s} {'':>13s}")
    for r in lb["levels"]:
        if r.get("skipped"):
            P(f"{r['level']:4d} {r['seam']:5d}  SKIPPED: {r['skipped']}")
            continue
        P(f"{r['level']:4d} {r['seam']:5d} {r['chosen_score']:8.4f} {r['score_med']:8.4f} "
          f"{r['score_max']:8.4f} {r['n_pair']:6d} "
          f"{str(r['n_distinct_a']) + '/' + str(r['n_distinct_b']):>13s}")

    # ---------------------------------------------------------------- the Δ sweep
    ladder_by = d.get("ladder_by_mode") or {"arm": d["ladder"]}
    eras_by = d.get("eras_by_mode") or {"arm": d["eras"]}
    modes = list(ladder_by)

    if d.get("handover"):
        P("\n--- DECISION 17 · the HAND-OVER into drilled seam 0 (shared by every arm at a "
          "given cell) ---")
        P(f"{'app':>5s} {'Δ':>3s} {'|x−V0|':>8s} {'|v|':>7s} {'|v−v*|':>8s} {'read Δx':>8s} "
          f"{'read Δv':>8s} {'read@':>7s} {'e_app':>8s}")
        for r in d["handover"]:
            P(f"{r['app']:>5s} {r['delta']:3d} {r['ho_wp']:8.4f} {r['ho_speed']:7.3f} "
              f"{r['ho_v_err']:8.3f} {r['read_pos_err']:8.4f} {r['read_vel_err']:8.3f} "
              f"{r['read_step']:7d} {r['e_app']:8.4f}"
              + ("   CLAMPED AT RESET" if r["clamped"] else ""))
        P(f"(the read at drilled seam 0 is clamped at the traversal start whenever k0·H < Δ; "
          f"here k0·H = {pc['k0'] * pc['H']})")

    if d.get("reflex_is_tape"):
        P("\n--- DECISION 19 · the closed-loop reflex against its OWN open-loop replay "
          "(pre-fixed reading: ratio ≤ 1.10 ⇒ the incumbent is a tape at that Δ) ---")
        for app in modes:
            P(f"[{app}] " + "  ".join(
                f"Δ{r['delta']}={r['reflex_ol_ratio']:.2f}"
                + ("*" if r["reflex_ol_ratio"] <= 1.10 else "")
                for r in ladder_by[app]))
            P("      cl: " + "  ".join(f"{r['reflex']:.4f}" for r in ladder_by[app]))
            P("      ol: " + "  ".join(f"{r['reflex_ol']:.4f}" for r in ladder_by[app]))

    P("\n--- THE Δ SWEEP: piece error per level (drilled segments only), band "
      f"{pc['band']:.4f}; * = inside the band ---")
    for app in modes:
        for pre, lbl in (("key", "frozen posture key, 0 groundings"),
                         ("aud", "plant audition of every member")):
            P(f"\n[app={app} · {pre}]  {lbl}")
            P(f"{'Δ':>3s} {'ms':>5s} {'reflex':>9s} {'refl_ol':>9s} "
              + " ".join(f"{'L' + str(e):>9s}" for e in range(1, L + 1))
              + f" {'all':>9s} {'best':>5s} " + " ".join(
                  f"{'m' + str(e) + str(e + 1):>8s}" for e in range(1, L)))
            for r in ladder_by[app]:
                def mark(x):
                    return f"{x:8.4f}" + ("*" if x <= r["band"] else " ")
                P(f"{r['delta']:3d} {r['ms']:5d} {mark(r['reflex'])} "
                  + mark(r.get("reflex_ol", float("nan"))) + " "
                  + " ".join(mark(r[f"{pre}_L{e}"]) for e in range(1, L + 1))
                  + f" {mark(r[f'{pre}_all'])} {r[f'{pre}_best']:5d} "
                  + " ".join(f"{r[f'{pre}_m{e}{e + 1}']:+8.4f}" for e in range(1, L)))
        P(f"era ladder [app={app}] (smallest Δ at which level l fails the band and l+1 does "
          f"not): {json.dumps(eras_by[app])}")

    P("\n--- feedback consumption per traversal, DRILLED ONLY (the approach's "
      f"{pc['k0'] * pc['H']} fb/perf is shared and subtracted) ---")
    P(f"{'Δ':>3s} " + " ".join(f"{a:>12s}" for a in ("reflex",) + tuple(
        f"key_L{e}" for e in range(1, L + 1))))
    for r in d["ladder"]:
        P(f"{r['delta']:3d} " + f"{'':>0s}" + " ".join(
            [f"{[s for s in d['sweep'] if s['delta'] == r['delta'] and s['arm'] == 'reflex'][0]['ledger_drilled']['n_fb']:12.1f}"]
            + [f"{r[f'key_L{e}_fb']:12.1f}" for e in range(1, L + 1)]))

    P("\n--- pass fractions at the THREE band widths (per-performer mean drilled waypoint "
      f"error; ½·leg = {pc['half_leg']:.4f} is the pre-fixed band, ⅓·leg = {pc['mean_leg'] / 3:.4f}"
      f", ¼·leg = {pc['mean_leg'] / 4:.4f}) ---")
    for app in modes:
        for pre in ("key", "aud"):
            P(f"[app={app} · {pre}]  " + f"{'Δ':>3s} | " + " | ".join(
                w + ": " + " ".join(f"{c:>5s}" for c in
                                    ["rf", "ol"] + [f"L{e}" for e in range(1, L + 1)])
                for w in ("½", "⅓", "¼")))
            for D in cfg["deltas"]:
                rows = [s_ for s_ in d["sweep"]
                        if s_["delta"] == D and s_.get("app", "arm") == app]

                def get(nm, key):
                    m = [x for x in rows if x["arm"] == nm]
                    return m[0][key] if m else float("nan")
                cells = []
                for key in ("in_h", "in_t", "in_q"):
                    cells.append("   " + " ".join(
                        [f"{get('reflex', key):5.2f}", f"{get('reflex_ol', key):5.2f}"]
                        + [f"{get(f'{pre}_L{e}', key):5.2f}" for e in range(1, L + 1)]))
                P(f"      {D:3d} | " + " | ".join(cells))
    P("\n--- the do-nothing floor (presto's arm-neutral reference): arms ABOVE it are diverged, "
      "not degraded ---")
    dnr = [r for r in d["P_T"]["rows"] if abs(r["R"] - float(pc["meta"]["R"])) < 1e-9]
    dn = dnr[0]["do_nothing"] if dnr else float("nan")
    P(f"do-nothing floor at R = {pc['meta']['R']}: {dn:.4f} "
      f"({dn / pc['mean_leg']:.1f} mean legs)")
    for app in modes:
        for D in cfg["deltas"]:
            rows = [s_ for s_ in d["sweep"]
                    if s_["delta"] == D and s_.get("app", "arm") == app]
            over = [x["arm"] for x in rows if x["e_piece"] > dn]
            P(f"      app={app:>4s} Δ={D:2d}  above the floor: "
              f"{', '.join(over) if over else '—'}")

    # ---------------------------------------------------------------- depth selection
    P("\n--- aud_all: which depth does a free selector choose, per Δ (acappella finding 6's "
      "instrument) ---")
    P(f"{'Δ':>3s} {'mean level':>11s} {'e_piece':>9s} " + " ".join(
        f"{'f(L' + str(e) + ')':>7s}" for e in range(1, L + 1)))
    for app in modes:
        P(f"[app={app}]")
        for D in cfg["deltas"]:
            rows = [s_ for s_ in d["sweep"] if s_["delta"] == D and s_["arm"] == "aud_all"
                    and s_.get("app", "arm") == app]
            if not rows:
                continue
            r = rows[0]
            fb = r.get("frac_by_level", {})
            P(f"{D:3d} {r.get('mean_level', float('nan')):11.2f} {r['e_piece']:9.4f} "
              + " ".join(f"{fb.get(str(e), 0.0):7.3f}" for e in range(1, L + 1)))

    # ---------------------------------------------------------------- seam information
    P("\n--- P-S: seam information under the delayed read (per-state oracle over each cell) ---")
    P(f"{'app':>5s} {'Δ':>3s} {'lvl':>4s} {'mean gain':>10s} {'range':>13s} "
      f"{'argmins':>8s} {'mean oracle':>12s}")
    for app in sorted({r.get("app", "arm") for r in d["P_S"]}):
        for D in sorted({r["delta"] for r in d["P_S"]}):
            for e in range(1, L + 1):
                rr = [r for r in d["P_S"] if r.get("app", "arm") == app
                      and r["delta"] == D and r["level"] == e]
                if not rr:
                    continue
                g = [r["gain"] for r in rr]
                P(f"{app:>5s} {D:3d} {e:4d} {sum(g) / len(g):9.3f}× "
                  f"{f'{min(g):.2f}-{max(g):.2f}':>13s} "
                  f"{sum(r['n_distinct_argmin'] for r in rr) / len(rr):5.1f}/{rr[0]['n_slot']:<2d}"
                  f" {sum(r['per_state_oracle'] for r in rr) / len(rr):12.4f}")

    P("\n--- e_app: what the shared approach itself cost, per Δ ---")
    for app in modes:
        P(f"  [{app}] " + "  ".join(
            f"Δ{D}=" + f"{[s_ for s_ in d['sweep'] if s_['delta'] == D and s_['arm'] == 'reflex' and s_.get('app', 'arm') == app][0]['e_app']:.4f}"
            for D in cfg["deltas"]))
    P(f"\npiece: seg/τ = {pc['meta'].get('seg_over_tau', float('nan')):.2f} "
      f"(damping {pc['meta'].get('damping', 2.0)}, terminal speed "
      f"{pc['meta'].get('v_terminal', float('nan')):.2f} m/s); étude's square is 1.63")

    P("\n--- flags, unsmoothed ---")
    for k, v in d.get("flags", {}).items():
        P(f"  [{k}] {v}")
    P(f"\ntimings: P-F1 {d.get('t_pf1', 0):.0f}s · P-T {d.get('t_pt', 0):.0f}s · "
      f"reflex cal {d.get('t_reflex_cal', 0):.0f}s · library {d.get('t_library', 0):.0f}s · "
      f"sweep {d.get('t_sweep', 0):.0f}s · total {d.get('wall_s', 0):.0f}s")


# --------------------------------------------------------------------------- figures (REMOTE)

@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_ladder_figures(tag: str) -> list:
    """Drawn REMOTELY and committed to the volume: the analysis machine in this repo's sessions
    has neither numpy nor matplotlib (`offbook/FILES.md` Gotcha).

    Axes are in the piece's own units. No panel title states a reading; titles name the quantity.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_prestissimo", tag)
    figdir = os.path.join(root, "figs")
    os.makedirs(figdir, exist_ok=True)
    d = json.load(open(os.path.join(root, "ladder.json")))
    cfg, pc = d["config"], d["piece"]
    L = int(pc["n_levels"])
    band, dt = pc["band"], pc["dt_ctrl"]
    made = []
    CL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name), dpi=140)
        plt.close(fig)
        made.append(name)

    # --- fig1: the era ladder ------------------------------------------------------------
    if d.get("ladder"):
        ms = [r["ms"] for r in d["ladder"]]
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
        for j, pre in enumerate(("key", "aud")):
            a = ax[j]
            a.plot(ms, [r["reflex"] for r in d["ladder"]], "k--o", ms=4, lw=1.6,
                   label="reflex (re-fit per Δ)")
            for e in range(1, L + 1):
                a.plot(ms, [r[f"{pre}_L{e}"] for r in d["ladder"]], "-o", ms=4, color=CL[e - 1],
                       label=f"L{e} ({1 << (e - 1)} seg)")
            a.plot(ms, [r[f"{pre}_all"] for r in d["ladder"]], ":s", ms=4, color="#7f7f7f",
                   label="all levels legal")
            a.axhline(band, color="#888", lw=1.0)
            a.text(ms[0], band, "  band", va="bottom", ha="left", fontsize=8, color="#666")
            a.set_xlabel("observation delay Δ (ms)")
            a.set_yscale("log")
            a.set_title(f"{pre}: " + ("frozen posture key, 0 groundings" if pre == "key"
                                      else "plant audition of every member"), fontsize=10)
            a.grid(alpha=0.25)
        ax[0].set_ylabel("piece error, drilled segments (arena units; "
                         f"leg = {pc['mean_leg']:.3f})")
        ax[0].legend(fontsize=8, ncol=2)
        save(fig, "fig1_era_ladder.png")

    # --- fig2: the piece -----------------------------------------------------------------
    wps = np.array(d.get("piece_wps") or [], float)
    if wps.size == 0:
        from mjc.practice.tempo.prestissimo import piece as PP
        wps = np.array(PP.fast_piece(R=float(pc["meta"]["R"]), k_app=int(pc["k0"]),
                                     h_seg=int(pc["H"]), n_levels=L).wps, float)
    k0 = int(pc["k0"])
    fig, a = plt.subplots(figsize=(5.6, 5.6))
    a.plot(wps[:k0 + 1, 0], wps[:k0 + 1, 1], "-o", color="#999", ms=4, lw=1.4,
           label=f"approach ({k0} seg, primitive)")
    a.plot(wps[k0:, 0], wps[k0:, 1], "-o", color="#1f77b4", ms=5, lw=1.8,
           label=f"drilled ({pc['K_drill']} seg × {pc['H']} steps = {pc['seg_ms']:.0f} ms)")
    for i, (x, y) in enumerate(wps[k0:-1]):
        a.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    for rg in pc.get("regions", []):
        a.add_patch(plt.Circle(rg["center"], rg["sigma"], color="#d62728", alpha=0.18))
        a.add_patch(plt.Circle(rg["center"], 2 * rg["sigma"], color="#d62728", alpha=0.08))
    a.plot([wps[0, 0]], [wps[0, 1]], "k*", ms=13, label="reset (start jitter ±0.05)")
    a.add_patch(plt.Circle((wps[0, 0], wps[0, 1]), 0.12, fill=False, ls=":", color="#555"))
    a.set_aspect("equal"); a.grid(alpha=0.25)
    a.set_title(f"{pc['name']}: mean tip speed {pc['speed']:.2f} m/s, turns "
                f"{min(pc['turns']):.0f}–{max(pc['turns']):.0f}°\n"
                f"(dotted circle = the pusher's own radius)", fontsize=9)
    a.legend(fontsize=8, loc="lower right")
    save(fig, "fig2_piece.png")

    # --- fig3: the tempo calibration -----------------------------------------------------
    t = d["P_T"]
    fig, a = plt.subplots(figsize=(6.4, 4.6))
    R = [r["R"] for r in t["rows"]]
    for i, D in enumerate(cfg["cal_deltas"]):
        a.plot(R, [r[f"rot@{D}"] for r in t["rows"]], "-o", ms=4, color=CL[i],
               label=f"reflex @ Δ={D} ({D * dt * 1000:.0f} ms)")
    a.plot(R, [r["band"] for r in t["rows"]], "k--", lw=1.4, label="band = ½·mean leg")
    a.plot(R, [r["do_nothing"] for r in t["rows"]], ":", color="#888", label="do-nothing floor")
    a.axvline(t["R_star"], color="#d62728", lw=1.2)
    a.text(t["R_star"], a.get_ylim()[1], f" R* = {t['R_star']:.2f}", va="top", fontsize=9,
           color="#d62728")
    a.set_xlabel("R (circumradius; the linear-speed knob — turn rate is invariant in R)")
    a.set_ylabel("piece error (arena units)")
    a.set_yscale("log"); a.grid(alpha=0.25); a.legend(fontsize=8)
    a.set_title("P-T: the tempo calibration, incumbent only", fontsize=10)
    save(fig, "fig3_calibration.png")

    # --- fig4: in-band fraction, and the rent ---------------------------------------------
    if d.get("ladder"):
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
        for e in range(1, L + 1):
            ax[0].plot(ms, [r[f"key_L{e}_in"] for r in d["ladder"]], "-o", ms=4,
                       color=CL[e - 1], label=f"L{e}")
        ax[0].set_xlabel("observation delay Δ (ms)")
        ax[0].set_ylabel("fraction of performers inside the band")
        ax[0].set_ylim(-0.03, 1.03); ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8)
        ax[0].set_title("key arms: in-band fraction", fontsize=10)
        for e in range(1, L + 1):
            xs = [r[f"key_L{e}_fb"] for r in d["ladder"]]
            ys = [r[f"key_L{e}"] for r in d["ladder"]]
            ax[1].plot(xs, ys, "o", ms=5, color=CL[e - 1], label=f"L{e}")
        ax[1].plot([r_["ledger_drilled"]["n_fb"] for r_ in d["sweep"] if r_["arm"] == "reflex"],
                   [r_["e_piece"] for r_ in d["sweep"] if r_["arm"] == "reflex"], "k+", ms=8,
                   label="reflex")
        ax[1].axhline(band, color="#888", lw=1.0)
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("feedback events per traversal (drilled only)")
        ax[1].set_ylabel("piece error")
        ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8)
        ax[1].set_title("the rent, every Δ pooled", fontsize=10)
        save(fig, "fig4_band_and_rent.png")

    volume.commit()
    return made


@app.local_entrypoint()
def prestissimo_figs(tag: str = "a0"):
    made = make_ladder_figures.remote(tag)
    print(f"[figs] wrote {made} to /data/practice_prestissimo/{tag}/figs on the volume")


def figures(tag):
    """Dispatch the remote job, then pull the PNGs into `figures/<tag>/`."""
    here = os.path.relpath(os.path.abspath(__file__),
                           os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    subprocess.run(["modal", "run", f"{here}::prestissimo_figs", "--tag", tag], check=True,
                   cwd=root)
    dst = os.path.join(HERE, "figures", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOL,
                    f"/practice_prestissimo/{tag}/figs", dst], check=True)
    nest = os.path.join(dst, "figs")          # `volume get` of a directory nests it; flatten
    if os.path.isdir(nest):
        for f in os.listdir(nest):
            os.replace(os.path.join(nest, f), os.path.join(dst, f))
        os.rmdir(nest)
    print(f"[figures] -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    path = (fetch(args.tag) if args.fetch
            else os.path.join(HERE, "results", args.tag, "ladder.json"))
    if args.figures and not os.path.exists(path):
        path = fetch(args.tag)
    d = json.load(open(path))
    out = []
    report(d, out)
    rp = os.path.join(HERE, "results", args.tag, "reduction.txt")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    open(rp, "w").write("\n".join(out) + "\n")
    print(f"\n[written] {rp}")
    if args.figures:
        figures(args.tag)


if __name__ == "__main__":
    main()
