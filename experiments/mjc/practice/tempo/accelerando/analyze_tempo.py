"""Reduce accelerando's Phase 1 run (the fast body + the tempo ladder) to a text report and
figures.

Pure python for the reduction (json + math only): the local client has neither numpy nor
matplotlib (`offbook/FILES.md` Gotcha, carried through every node of this arc). `--figures`
renders remotely on Modal and pulls the PNGs back.

    python3 mjc/practice/tempo/accelerando/analyze_tempo.py --tag t1 --fetch
    python3 mjc/practice/tempo/accelerando/analyze_tempo.py --tag t1 --figures
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from mjc.shared import app, volume, DATA_DIR   # noqa: E402

VOL = "mujoco-control-data"


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL,
                    f"/practice_accelerando/{tag}/tempo.json",
                    os.path.join(dst, "tempo.json"), "--force"], check=True)
    return os.path.join(dst, "tempo.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    tempos = cfg["tempos"]
    L = int(cfg["n_levels"])
    Dfix = int(cfg["delta_fix"])
    modes = cfg["app_modes"]
    exps = [int(x) for x in cfg["scale_exponents"]]
    srcs = ["rec"] + [f"scl{e}" for e in exps]

    P("=" * 100)
    P(f"accelerando Phase 1 — tag {cfg['tag']}   seed {cfg['seed']}   "
      f"wall {d.get('wall_s', 0):.0f}s   complete={d.get('complete')}")
    P("=" * 100)
    P(f"tempo ladder H = {tempos} (steps/note)   Delta FIXED = {Dfix} "
      f"({1000 * Dfix * 0.024:.0f} ms)   levels {L}   "
      f"listener window {cfg['w_listen']} steps ({1000 * cfg['w_listen'] * 0.024:.0f} ms)")
    P(f"body: mass {cfg['mass']}  damping {cfg['damping']}   sources {srcs}   "
      f"app modes {modes} (arm of record: {modes[-1]})")
    P()

    # ---------------------------------------------------------------- gates
    P("--- GATES " + "-" * 88)
    f0 = d.get("P_F0", {})
    P(f"P-F0  donor constants: {f0.get('checked')} checked, "
      f"{len(f0.get('mismatches', {}))} mismatched   pass={f0.get('pass')}")
    f1 = d.get("P_F1", {})
    if f1.get("applicable"):
        P(f"P-F1  fork on the DONOR square vs acappella/b1: {f1.get('n_checks')} checks, "
          f"max|delta| = {f1.get('max_abs'):.3e}, gains_match={f1.get('gains_match')}   "
          f"pass={f1.get('pass')}")
    else:
        P(f"P-F1  {f1.get('note', 'not applicable')}")
    f2 = d.get("P_F2", {})
    if f2.get("applicable"):
        nn = f2.get("nesting", {})
        P(f"P-F2  fork on prestissimo/a1g's FAST PIECE: {f2.get('n_checks')} checks, "
          f"max|delta| = {f2.get('max_abs'):.3e}; nesting {nn.get('n_checked')} spelled / "
          f"{nn.get('spell_bad')} bad / exec {nn.get('exec_max_abs'):.3e}   pass={f2.get('pass')}")
    else:
        P(f"P-F2  {f2.get('note', 'not applicable')}")
    pt = d.get("P_TAU", {})
    if pt:
        P(f"P-TAU tau measured from a step response: fast "
          f"{1000 * pt['fast']['tau_fit']:.3f} ms vs m/c {1000 * pt['fast']['tau_expected']:.3f} "
          f"ms (rel {pt['fast']['rel_err']:.5f}), v_inf {pt['fast']['v_inf_fit']:.4f} vs "
          f"{pt['fast']['v_inf_expected']:.2f}; donor {1000 * pt['donor']['tau_fit']:.1f} ms vs "
          f"{1000 * pt['donor']['tau_expected']:.1f} ms (rel {pt['donor']['rel_err']:.5f})   "
          f"tol {pt['tol']}   pass={pt['pass']}")
        P("      H   note ms   note/tau   tau<T<Delta")
        for r in pt["tempo_table"]:
            P(f"      {r['H']:2d}   {r['note_ms']:7.0f}   {r['note_over_tau']:8.2f}   "
              f"{r['in_band']}")
    prr = d.get("P_R", {})
    if prr:
        P(f"P-R   resampler identity at s = 1 (H {prr['src_tempo']} -> {prr['src_tempo']}, "
          f"p = 0): max|delta| = {prr['max_abs']:.3e} over "
          f"{prr['rows'][0]['n_members']} members   pass={prr['pass']}")
    pn = d.get("P_N", {})
    if pn:
        P(f"P-N   the nesting op, per tempo   pass={pn.get('pass')}")
        for H in tempos:
            v = pn["by_tempo"].get(str(H))
            if v:
                P(f"      H={H:2d}  {v['n_checked']} members spelled by their parents "
                  f"({v['spell_bad']} bad); weld == halves max|delta| = "
                  f"{v['exec_max_abs']:.3e} (tol {v['exec_tol']:.0e})")
    pa = d.get("P_A", {})
    if pa:
        P(f"P-A   read at drilled seam 0 stale at every Delta>0: {pa.get('read_grows')}; "
          f"never clamped at the reset state: {pa.get('never_clamped')}   pass={pa.get('pass')}")
        for m in modes:
            P(f"      {m:>5s}: " + "  ".join(
                f"H{H} key_L{L} spread {pa[f'{m}_H{H}_key_L{L}']['spread']:.4f} "
                f"({pa[f'{m}_H{H}_key_L{L}']['ratio']:.2f}x)" for H in tempos))
    P()

    # ---------------------------------------------------------------- the piece / the body
    pc = d.get("piece", {})
    P("--- THE PIECE AND THE BODY " + "-" * 72)
    if pc:
        P(f"R* = {d['P_T']['R_star']:.3f}   ({d['P_T']['rule']})")
        P(f"mean leg {pc['mean_leg']:.4f} m   band {pc['band']:.4f} (= max(1/2 leg "
          f"{pc['half_leg']:.4f}, ref_play scaled {pc['ref_play_scaled']:.4f}))   "
          f"turns {pc['turns']}")
        P(f"tau {1000 * pc['tau']:.1f} ms   A = gear/m {pc['accel']:.1f} m/s^2   "
          f"v_terminal {pc['v_terminal']:.2f} m/s   K_drill {pc['K_drill']} + k0 {pc['k0']}")
        P()
        P("      H   note ms   speed m/s   note/tau   u_turn   do-nothing e_piece / e_listen")
        for H in tempos:
            q = d["piece_by_tempo"][str(H)]
            dn = d.get("do_nothing", {}).get(str(H), {})
            P(f"      {H:2d}   {q['seg_ms']:7.0f}   {q['speed']:9.3f}   "
              f"{1.0 / q['tau_over_note']:8.2f}   {q['turn']['u_turn_max']:6.3f}   "
              f"{dn.get('e_piece', float('nan')):.4f} / {dn.get('e_listen', float('nan')):.4f}")
    P()

    # ---------------------------------------------------------------- P-T
    P("--- P-T: the R calibration (guards on the incumbent and the apparatus only) " + "-" * 24)
    rows = d["P_T"]["rows"]
    ct = rows[0].get("cal_tempos", tempos)
    hdr = "      R      leg      band  "
    for H in ct:
        hdr += f"| H{H}: v      u_turn  e_piece  e_listen  dn      gains        "
    P(hdr + "| feas comp demand")
    for r in rows:
        line = f"      {r['R']:.3f}  {r['mean_leg']:.4f}  {r['band']:.4f}  "
        for H in ct:
            q = r[f"H{H}"]
            line += (f"| {q['speed']:6.3f}  {q['u_turn']:6.3f}  {q['rot']:7.4f}  "
                     f"{q['listen']:8.4f}  {q['do_nothing']:6.3f}  "
                     f"{str(q['gains']):12s} ")
        line += f"| {str(r['feasible']):5s} {str(r['competence']):4s} {str(r['demand'])}"
        P(line)
    P(f"      u_turn threshold {d['P_T']['u_turn_max']}   n passing all three: "
      f"{d['P_T']['n_pass_all']}")
    P()

    # ---------------------------------------------------------------- gains and libraries
    P("--- the incumbent's per-tempo, per-Delta re-fit (clean world, every advantage) " + "-" * 21)
    P("      H  " + "  ".join(f"D{D}: kp/kd  e_clean   " for D in cfg["cal_deltas"]))
    for H in tempos:
        g = d["library"][str(H)]
        line = f"      {H:2d}  "
        for D in cfg["cal_deltas"]:
            # gains live in the library record only for D = 0; the sweep rows carry the rest
            line += "                      "
        P(line.rstrip() + f"   harvest gains {g['harvest_gains']}   "
          f"library {g['counts']}   {g['build_ledger']['n_ground']:.0f} groundings/performer")
    edges = d.get("flags", {}).get("gain_grid_edges", {})
    if edges:
        hit = [(H, D) for H in edges for D in edges[H]
               if edges[H][D]["kp"] or edges[H][D]["kd"]]
        P(f"      gain-grid edges hit at (H, Delta): {hit}")
    P()

    # ---------------------------------------------------------------- the main sweep
    def cell(app_m, H, nm):
        for r in d["sweep"]:
            if r["H"] == H and r["arm"] == nm and r["app"] == app_m:
                return r
        return None

    for app_m in modes:
        P(f"--- THE TEMPO SWEEP at Delta = {Dfix}, app = {app_m}"
          f"{'  (ARM OF RECORD)' if app_m == modes[-1] else ''} " + "-" * 30)
        band = pc["band"]
        P(f"    band {band:.4f}; '*' = inside it. e_piece / e_listen, then the 1/2-leg pass "
          f"fraction under each grade.")
        arms = ["reflex", "reflex_ol"]
        for s in srcs:
            pfx = "" if s == "rec" else f"s{s[3:]}_"
            for pre in ("key", "aud"):
                arms += [f"{pre}_{pfx}L{e}" for e in range(1, L + 1)]
        arms += ["key_all", "aud_all"]
        P("      arm            " + "".join(f"|      H={H:<2d}          " for H in tempos))
        for nm in arms:
            line = f"      {nm:<15s}"
            ok = False
            for H in tempos:
                c = cell(app_m, H, nm)
                if c is None:
                    line += "|                    "
                    continue
                ok = True
                st = "*" if c["e_piece"] <= band else " "
                sl = "*" if c["e_listen"] <= band else " "
                line += (f"| {c['e_piece']:.4f}{st}/{c['e_listen']:.4f}{sl} ")
            if ok:
                P(line)
        P("      -- 1/2-leg pass fraction (piece / listen) --")
        for nm in arms:
            line = f"      {nm:<15s}"
            ok = False
            for H in tempos:
                c = cell(app_m, H, nm)
                if c is None:
                    line += "|                    "
                    continue
                ok = True
                line += f"|   {c['in_h']:.2f} / {c['lis_h']:.2f}      "
            if ok:
                P(line)
        P("      -- feedback events per drilled traversal --")
        line = "      " + " " * 15
        for H in tempos:
            c = cell(app_m, H, "reflex")
            c4 = cell(app_m, H, f"aud_L{L}")
            line += f"| rf {c['ledger_drilled']['n_fb']:.0f} / L{L} {c4['ledger_drilled']['n_fb']:.0f}   "
        P(line)
        er = d["eras_by_mode"][app_m]
        P("      -- the era table (H at which level l fails the band and l+1 does not; "
          "H falls = tempo rises) --")
        for grade in ("piece", "listen"):
            got = {k.replace(f"{grade}_", ""): v for k, v in er.items()
                   if k.startswith(grade) and v is not None}
            null = sorted(k.replace(f"{grade}_", "") for k, v in er.items()
                          if k.startswith(grade) and v is None)
            P(f"      {grade:>6s}: {json.dumps(got)}")
            P(f"              null: {null}")
        P("      -- ordering margins on e_piece (L_l - L_(l+1), positive = the deeper rung "
          "wins) --")
        for pre in ("key", "aud"):
            for ell in range(1, L):
                line = f"      {pre} m{ell}{ell + 1}   "
                for H in tempos:
                    row = [r for r in d["ladder_by_mode"][app_m] if r["H"] == H][0]
                    line += f"| {row[f'{pre}_m{ell}{ell + 1}']:+.4f}   "
                P(line)
        P("      -- the incumbent against its own open-loop replay (ol/cl; <=1.10 => a tape) --")
        line = "      ratio        "
        for H in tempos:
            row = [r for r in d["ladder_by_mode"][app_m] if r["H"] == H][0]
            line += f"| {row['reflex_ol_ratio']:8.2f}"
            line += "*  " if row["reflex_ol_ratio"] <= 1.10 else "   "
        P(line)
        P()

    # ---------------------------------------------------------------- hand-over
    P("--- the hand-over into drilled seam 0 " + "-" * 61)
    P("      app     H  |x-V0| (legs)   |v|    v_sched  read err pos / vel   read@  e_app")
    for r in d["handover"]:
        P(f"      {r['app']:>5s} {r['H']:3d}  {r['ho_wp']:.4f} ({r['ho_legs']:.2f})  "
          f"{r['ho_speed']:6.3f}  {r['v_sched']:7.3f}  {r['read_pos_err']:.4f} / "
          f"{r['read_vel_err']:6.3f}   {r['read_step']}/{r['lead_steps']}"
          f"{' CLAMPED' if r['clamped'] else ''}  {r['e_app']:.4f}")
    P()

    # ---------------------------------------------------------------- the Delta instrument
    P("--- the Delta INSTRUMENT, per tempo (Delta is not the era knob here) " + "-" * 31)
    for app_m in modes:
        for H in tempos:
            rr = [r for r in d["instrument"] if r["H"] == H and r["app"] == app_m]
            if not rr:
                continue
            nms = sorted({r["arm"] for r in rr},
                         key=lambda x: ("reflex" not in x, x))
            P(f"      app={app_m} H={H}")
            for nm in nms:
                line = f"        {nm:<12s}"
                for D in cfg["deltas"]:
                    v = [r for r in rr if r["arm"] == nm and r["delta"] == D]
                    line += f"| D{D}: {v[0]['e_piece']:.4f}/{v[0]['e_listen']:.4f} " if v \
                        else "|            "
                P(line)
    P()

    # ---------------------------------------------------------------- P-S
    P("--- P-S: seam information (per-state oracle over the best fixed slot) " + "-" * 30)
    P("      H   Delta  " + "  ".join(f"L{e} gain (argmins)" for e in range(1, L + 1)))
    for H in tempos:
        for D in cfg["ps_deltas"]:
            line = f"      {H:2d}  {D:5d}  "
            for e in range(1, L + 1):
                rr = [r for r in d["P_S"] if r["H"] == H and r["delta"] == D and r["level"] == e]
                if not rr:
                    line += "                 "
                    continue
                g = sum(r["gain"] for r in rr) / len(rr)
                am = sum(r["n_distinct_argmin"] for r in rr) / len(rr)
                line += f"  {g:.3f} ({am:.1f}/{rr[0]['n_slot']})  "
            P(line)
    P()
    P("--- timings " + "-" * 87)
    P("      " + "  ".join(f"{k}={v:.0f}s" for k, v in d.items()
                           if k.startswith("t_") and isinstance(v, (int, float))))
    return out


# --------------------------------------------------------------------------- figures (REMOTE)

@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_tempo_figures(tag: str) -> list:
    """Drawn REMOTELY and committed to the volume: the analysis machine in this repo's sessions
    has neither numpy nor matplotlib. Axes are in the piece's own units; no panel title states a
    reading."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_accelerando", tag)
    figdir = os.path.join(root, "figs")
    os.makedirs(figdir, exist_ok=True)
    d = json.load(open(os.path.join(root, "tempo.json")))
    cfg, pc = d["config"], d["piece"]
    L = int(cfg["n_levels"])
    tempos = cfg["tempos"]
    band = pc["band"]
    ml = pc["mean_leg"]
    modes = cfg["app_modes"]
    rec = modes[-1]
    exps = [int(x) for x in cfg["scale_exponents"]]
    made = []
    CL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    note_ms = [1000.0 * H * 0.024 for H in tempos]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name), dpi=140)
        plt.close(fig)
        made.append(name)

    def cell(app_m, H, nm, key="e_piece"):
        for r in d["sweep"]:
            if r["H"] == H and r["arm"] == nm and r["app"] == app_m:
                return r[key]
        return None

    # --- fig1: the era ladder, both grades ------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True)
    for i, key in enumerate(("e_piece", "e_listen")):
        for j, pre in enumerate(("key", "aud")):
            a = ax[i][j]
            a.plot(note_ms, [cell(rec, H, "reflex", key) for H in tempos], "k--o", ms=5, lw=1.8,
                   label="reflex (re-fit per tempo)")
            for e in range(1, L + 1):
                a.plot(note_ms, [cell(rec, H, f"{pre}_L{e}", key) for H in tempos], "-o", ms=5,
                       color=CL[e - 1], label=f"L{e} ({1 << (e - 1)} notes)")
            a.plot(note_ms, [cell(rec, H, f"{pre}_all", key) for H in tempos], ":s", ms=4,
                   color="#7f7f7f", label="all levels legal")
            a.axhline(band, color="#888", lw=1.0)
            a.text(note_ms[-1], band, " band", va="bottom", ha="left", fontsize=8, color="#666")
            a.set_xscale("log")
            a.set_xticks(note_ms)
            a.set_xticklabels([f"{m:.0f}" for m in note_ms])
            a.set_yscale("log")
            a.set_xlabel("note length (ms)  — faster to the left")
            a.set_ylabel(key + " (m)")
            a.set_title(f"{key}   selection = {pre}   app = {rec}", fontsize=10)
            if i == 0 and j == 0:
                a.legend(fontsize=7.5, ncol=2)
    save(fig, "fig1_era_ladder.png")

    # --- fig2: rec vs the resampled tapes --------------------------------------------------
    fig, ax = plt.subplots(1, L, figsize=(3.6 * L, 4.2), sharey=True)
    for e in range(1, L + 1):
        a = ax[e - 1]
        a.plot(note_ms, [cell(rec, H, f"aud_L{e}") for H in tempos], "-o", ms=5, color=CL[0],
               label="rec (harvested at this tempo)")
        for t, p in enumerate(exps):
            a.plot(note_ms, [cell(rec, H, f"aud_s{p}_L{e}") for H in tempos], "-^", ms=5,
                   color=CL[t + 1], label=f"scl{p} (slow tape, resampled)")
        a.plot(note_ms, [cell(rec, H, "reflex") for H in tempos], "k--", lw=1.4, label="reflex")
        a.axhline(band, color="#888", lw=1.0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xticks(note_ms); a.set_xticklabels([f"{m:.0f}" for m in note_ms])
        a.set_xlabel("note length (ms)")
        a.set_title(f"level {e} ({1 << (e - 1)} notes), audition", fontsize=10)
        if e == 1:
            a.set_ylabel("e_piece (m)")
            a.legend(fontsize=7.5)
    save(fig, "fig2_recording_vs_resampled.png")

    # --- fig3: the Delta instrument --------------------------------------------------------
    ds = cfg["deltas"]
    fig, ax = plt.subplots(len(modes), len(tempos), figsize=(3.1 * len(tempos), 3.6 * len(modes)),
                           sharey=True, squeeze=False)
    for i, app_m in enumerate(modes):
        for j, H in enumerate(tempos):
            a = ax[i][j]
            for nm, c, st in (("reflex", "k", "--o"), (f"key_L1", CL[0], "-o"),
                              (f"key_L{L}", CL[3], "-o"), (f"aud_L1", CL[0], "--^"),
                              (f"aud_L{L}", CL[3], "--^")):
                v = [[r["e_piece"] for r in d["instrument"]
                      if r["H"] == H and r["app"] == app_m and r["delta"] == D and r["arm"] == nm]
                     for D in ds]
                v = [x[0] if x else None for x in v]
                if all(x is None for x in v):
                    continue
                a.plot([D * 24 for D in ds], v, st, ms=4, color=c, lw=1.4, label=nm)
            a.axhline(band, color="#888", lw=1.0)
            a.axvline(cfg["delta_fix"] * 24, color="#c33", lw=1.0, ls=":")
            a.set_yscale("log")
            a.set_title(f"app={app_m}  H={H} ({1000 * H * 0.024:.0f} ms)", fontsize=9)
            a.set_xlabel("Delta (ms)")
            if j == 0:
                a.set_ylabel("e_piece (m)")
                a.legend(fontsize=7)
    save(fig, "fig3_delta_instrument.png")

    # --- fig4: seam information ------------------------------------------------------------
    fig, ax = plt.subplots(1, len(cfg["ps_deltas"]), figsize=(5.4 * len(cfg["ps_deltas"]), 4.2),
                           squeeze=False)
    for i, D in enumerate(cfg["ps_deltas"]):
        a = ax[0][i]
        for e in range(1, L + 1):
            g = []
            for H in tempos:
                rr = [r["gain"] for r in d["P_S"]
                      if r["H"] == H and r["delta"] == D and r["level"] == e]
                g.append(sum(rr) / len(rr) if rr else None)
            a.plot(note_ms, g, "-o", ms=5, color=CL[e - 1], label=f"L{e}")
        a.axhline(1.0, color="#888", lw=1.0)
        a.set_xscale("log"); a.set_xticks(note_ms)
        a.set_xticklabels([f"{m:.0f}" for m in note_ms])
        a.set_xlabel("note length (ms)")
        a.set_ylabel("best fixed slot / per-state oracle")
        a.set_title(f"seam information, Delta = {D}", fontsize=10)
        a.legend(fontsize=8)
    save(fig, "fig4_seam_information.png")

    # --- fig5: the two grades against each other -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for j, app_m in enumerate(modes):
        a = ax[j]
        for e in range(1, L + 1):
            xs = [cell(app_m, H, f"aud_L{e}") for H in tempos]
            ys = [cell(app_m, H, f"aud_L{e}", "e_listen") for H in tempos]
            a.plot(xs, ys, "-o", ms=5, color=CL[e - 1], label=f"aud L{e}")
        xs = [cell(app_m, H, "reflex") for H in tempos]
        ys = [cell(app_m, H, "reflex", "e_listen") for H in tempos]
        a.plot(xs, ys, "k--o", ms=5, label="reflex")
        lim = [min(xs + ys) * 0.7, max(xs + ys) * 1.4]
        a.plot(lim, lim, color="#bbb", lw=1.0)
        a.axhline(band, color="#888", lw=0.8); a.axvline(band, color="#888", lw=0.8)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("e_piece (per waypoint)"); a.set_ylabel("e_listen (listener's clock)")
        a.set_title(f"the two grades, app = {app_m}", fontsize=10)
        a.legend(fontsize=7.5)
    save(fig, "fig5_two_grades.png")

    # --- fig6: the piece, and the body's step response -------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    a = ax[0]
    q = d["piece_by_tempo"][str(tempos[0])]
    a.bar([str(H) for H in tempos], [d["piece_by_tempo"][str(H)]["turn"]["u_turn_max"]
                                     for H in tempos], color=CL[0])
    a.axhline(1.0, color="#c33", lw=1.2)
    a.set_xlabel("H (steps per note)"); a.set_ylabel("peak schedule command |u|")
    a.set_title("what the schedule demands of the actuator", fontsize=10)
    a = ax[1]
    a.bar([str(H) for H in tempos], [d["piece_by_tempo"][str(H)]["speed"] for H in tempos],
          color=CL[1])
    a.axhline(q["v_terminal"], color="#c33", lw=1.2)
    a.text(0, q["v_terminal"], " v_terminal", va="bottom", fontsize=8, color="#c33")
    a.set_xlabel("H (steps per note)"); a.set_ylabel("schedule speed (m/s)")
    a.set_title("the tempo ladder in m/s", fontsize=10)
    save(fig, "fig6_apparatus.png")

    volume.commit()
    return made


@app.local_entrypoint()
def accelerando_figs(tag: str = "t1"):
    made = make_tempo_figures.remote(tag)
    print(f"[figs] wrote {made} to /data/practice_accelerando/{tag}/figs on the volume")


def figures(tag):
    here = os.path.relpath(os.path.abspath(__file__),
                           os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    subprocess.run(["modal", "run", f"{here}::accelerando_figs", "--tag", tag], check=True,
                   cwd=root)
    dst = os.path.join(HERE, "figures", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOL,
                    f"/practice_accelerando/{tag}/figs", dst], check=True)
    nest = os.path.join(dst, "figs")
    if os.path.isdir(nest):
        for f in os.listdir(nest):
            os.replace(os.path.join(nest, f), os.path.join(dst, f))
        os.rmdir(nest)
    print(f"[figures] -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    path = (fetch(args.tag) if args.fetch
            else os.path.join(HERE, "results", args.tag, "tempo.json"))
    if not os.path.exists(path):
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
