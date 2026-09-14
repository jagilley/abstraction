"""Reduce accelerando's Phase 2 run (the program against the recording, across tempo) to a text
report and figures.

Pure python for the reduction (json + math only): the local client has neither numpy nor
matplotlib. `--figures` renders remotely on Modal and pulls the PNGs back.

    python3 mjc/practice/tempo/accelerando/analyze_program.py --tag p1 --fetch
    python3 mjc/practice/tempo/accelerando/analyze_program.py --tag p1 --figures
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
                    f"/practice_accelerando/{tag}/program.json",
                    os.path.join(dst, "program.json"), "--force"], check=True)
    return os.path.join(dst, "program.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    T = cfg["tempos"]
    L = int(cfg["n_levels"])
    Dfix = int(cfg["delta_fix"])
    prac = set(cfg["practiced"])
    prac2 = set(cfg["practiced_short"])
    band = d["piece_by_tempo"][str(T[0])]["band"]
    ml = d["piece_by_tempo"][str(T[0])]["mean_leg"]

    def c(H, nm, k="e_piece", D=None, app_m="fixed"):
        for r in d["sweep"]:
            if (r["H"] == H and r["arm"] == nm and r["app"] == app_m
                    and r["delta"] == (Dfix if D is None else D)):
                return r.get(k)
        return None

    P("=" * 100)
    P(f"accelerando Phase 2 — tag {cfg['tag']}   seed {cfg['seed']}   "
      f"wall {d.get('wall_s', 0):.0f}s   complete={d.get('complete')}   "
      f"device={d.get('device')}")
    P("=" * 100)
    P(f"tempo ladder H = {T}   PRACTICED {sorted(prac)} (prog)  {sorted(prac2)} (prog2)   "
      f"Delta FIXED = {Dfix}   band {band:.4f}   mean leg {ml:.4f}")
    P("Arms that see a tempo they never practiced at: prog, prog2, proggate, aud_s0, aud_s2.")
    P("Arms at an UNPRACTICED tempo that are ORACLES (what practice there would have bought, "
      "and no prog arm can see): key_L*, aud_L*.")
    P()

    P("--- GATES " + "-" * 88)
    f0 = d.get("P_F0", {})
    P(f"P-F0  donor constants: {f0.get('checked')} checked, "
      f"{len(f0.get('mismatches', {}))} mismatched   pass={f0.get('pass')}")
    pg = d.get("P_G", {})
    P(f"P-G   gain table, cell (H={pg['cell'][0]}, Delta={pg['cell'][1]}) re-fit from the full "
      f"grid: {pg['refit']} vs t1's {pg['pinned']}   applicable={pg.get('applicable')}   "
      f"pass={pg.get('pass')}"
      + ("" if pg.get("applicable") else
         "   (not applicable: this run RE-FIT its whole gain table rather than pinning it, so "
         "there is nothing to check against — the agreement shown IS the re-fit reproducing "
         "t1's rule)"))
    p1 = d.get("P_T1", {})
    if p1.get("applicable"):
        P(f"P-T1  Phase 2 reproduces `t1`: {p1['n_checks']} checks, max|delta| = "
          f"{p1['max_abs']:.3e}   pass={p1['pass']}")
        if p1.get("worst"):
            P("      worst: " + "  ".join(
                f"{w['what']} {abs(w['got'] - w['want']):.2e}" for w in p1["worst"][:3]))
    else:
        P(f"P-T1  NOT APPLICABLE — config differs from t1: {sorted(p1.get('cfg_diff', {}))}")
    pn = d.get("P_N", {})
    P(f"P-N   the nesting op, per tempo   pass={pn.get('pass')}   " + "  ".join(
        f"H{H}: {pn['by_tempo'][str(H)]['n_checked']}/{pn['by_tempo'][str(H)]['spell_bad']} bad, "
        f"{pn['by_tempo'][str(H)]['exec_max_abs']:.1e}" for H in T if str(H) in pn["by_tempo"]))
    pr = d.get("P_R", {})
    P(f"P-R   resampler identity at s = 1: max|delta| = {pr['max_abs']:.3e} over "
      f"{pr['n_members']} members   pass={pr['pass']}")
    ph = d.get("P_H", {})
    P(f"P-H   train/held-out disjoint: {ph['n_train']} train / {ph['n_hold']} held-out over "
      f"{ph['n_cells']} (slot, tempo) cells, overlap {ph['overlap']}   pass={ph['pass']}")
    P()

    # ------------------------------------------------ the pool, beside the head's training loss
    P("--- THE PRACTICE POOL AND THE HEAD'S LOSS, side by side " + "-" * 44)
    P("(Phase 1 measured the slot partition degenerate at 4 of 5 tempi; the head trains on that "
      "same pool at the same sigma = 0.0628, unchanged so t1 stays comparable.)")
    P("      H   | pool: cmd_sd  cmd_rms  sd/rms  traj_sd   err_mean  err_sd | head mse   "
      "zero      slot-mean  frac explained")
    for H in T:
        ps = d["pool"][str(H)]
        n = len(ps)
        cs = sum(q["cmd_sd"] for q in ps) / n
        cr = sum(q["cmd_rms"] for q in ps) / n
        ts = sum(q["traj_sd"] for q in ps) / n
        em = sum(q["err_mean"] for q in ps) / n
        es = sum(q["err_sd"] for q in ps) / n
        tr = d["training"].get("prog") or d["training"][sorted(d["training"])[0]]
        mse = tr["by_tempo"].get(str(H), tr["by_tempo"].get(H))
        bl = tr.get("by_tempo_baseline", {}).get(str(H), tr.get("by_tempo_baseline", {}).get(H))
        if mse is None:
            P(f"      {H:2d}  | {cs:9.4f}  {cr:7.4f}  {cs / cr:6.3f}  {ts:8.5f}  {em:8.4f}  "
              f"{es:6.4f} |  (not practiced)")
        else:
            z0 = bl["zero"] if bl else float("nan")
            sm = bl["slot_mean"] if bl else float("nan")
            P(f"      {H:2d}  | {cs:9.4f}  {cr:7.4f}  {cs / cr:6.3f}  {ts:8.5f}  {em:8.4f}  "
              f"{es:6.4f} | {mse:.6f}  {z0:.6f}  {sm:.6f}   {1.0 - mse / z0:.3f}")
    for nm in d["training"]:
        r = d["training"][nm]
        P(f"      {nm}: {r['n_rows']} rows on {r['practiced']}, final mse {r['final']:.6f} "
          f"(zero {r['baseline']['zero']:.6f}, slot-mean {r['baseline']['slot_mean']:.6f}); "
          f"loss series {[round(x, 5) for x in r['series'][::max(1, len(r['series']) // 6)]]}")
    P()

    # ------------------------------------------------------------------------------ parity
    P("--- PARITY, per slot per tempo (execution reproduction on the plant, held-out states) "
      + "-" * 14)
    P(f"    tau = {d['parity']['tau']}; a slot is OPEN when the head's emission, executed, is at "
      f"least as good as the reference's own")
    P("    per-state audition winner on at least tau of that slot's held-out launch states.")
    for nm in sorted({r["head"] for r in d["parity"]["rows"]}):
        P(f"    head = {nm}   (practiced {d['training'][nm]['practiced']})")
        for ref in ("rec", "scl0"):
            P(f"      vs {ref:<5s}" + "".join(f"|   H={H:<3d}      " for H in T))
            for ell in range(1, L + 1):
                line = f"        L{ell} open  "
                for H in T:
                    rr = [r for r in d["parity"]["rows"]
                          if r["head"] == nm and r["ref"] == ref and r["H"] == H
                          and r["level"] == ell]
                    if not rr:
                        line += "|             "
                        continue
                    op = sum(1 for r in rr if r["open"])
                    line += f"| {op:2d}/{len(rr):<2d} f={sum(r['frac'] for r in rr) / len(rr):.2f} "
                P(line)
            line = "        mean gap  "
            for H in T:
                rr = [r for r in d["parity"]["rows"]
                      if r["head"] == nm and r["ref"] == ref and r["H"] == H]
                line += (f"| {sum(r['mean_gap'] for r in rr) / len(rr):+.4f}    " if rr
                         else "|             ")
            P(line)
            line = "        head/tape "
            for H in T:
                rr = [r for r in d["parity"]["rows"]
                      if r["head"] == nm and r["ref"] == ref and r["H"] == H]
                if rr:
                    a = sum(r["mean_e_head"] for r in rr) / len(rr)
                    b = sum(r["mean_e_tape"] for r in rr) / len(rr)
                    line += f"| {a:.4f}/{b:.4f}"
                else:
                    line += "|             "
            P(line)
    P()

    # ------------------------------------------------------------------------------ the sweep
    P(f"--- THE SWEEP at Delta = {Dfix}, app = fixed   (e_piece / e_listen; '*' = in band; "
      f"'P' = practiced) " + "-" * 5)
    arms = ["reflex", "reflex_ol"]
    for e in range(1, L + 1):
        arms += [f"key_L{e}", f"aud_L{e}", f"aud_s0_L{e}", f"aud_s2_L{e}",
                 f"prog_L{e}", f"proga_L{e}", f"proggate_L{e}", f"prog2_L{e}",
                 f"progall_L{e}"]
    P("      arm            " + "".join(
        f"|   H={H:<2d}{'  P' if H in prac else '   '}        " for H in T))
    for nm in arms:
        line = f"      {nm:<15s}"
        ok = False
        for H in T:
            v, w = c(H, nm), c(H, nm, "e_listen")
            if v is None:
                line += "|                    "
                continue
            ok = True
            line += (f"| {v:.4f}{'*' if v <= band else ' '}/{w:.4f}"
                     f"{'*' if w <= band else ' '} ")
        if ok:
            P(line)
    P("      -- pass fractions 1/2 | 1/3 | 1/4 mean-leg, e_piece --")
    for nm in arms:
        line = f"      {nm:<15s}"
        ok = False
        for H in T:
            v = c(H, nm, "in_h")
            if v is None:
                line += "|                    "
                continue
            ok = True
            line += f"| {v:.2f} {c(H, nm, 'in_t'):.2f} {c(H, nm, 'in_q'):.2f}      "
        if ok:
            P(line)
    P("      -- pass fractions 1/2 | 1/3 | 1/4 mean-leg, e_listen --")
    for nm in arms:
        line = f"      {nm:<15s}"
        ok = False
        for H in T:
            v = c(H, nm, "lis_h")
            if v is None:
                line += "|                    "
                continue
            ok = True
            line += f"| {v:.2f} {c(H, nm, 'lis_t'):.2f} {c(H, nm, 'lis_q'):.2f}      "
        if ok:
            P(line)
    P("      -- how much of a prog arm actually came from the head (head / fallback launches) --")
    for e in range(1, L + 1):
        for nm in (f"prog_L{e}", f"proggate_L{e}"):
            line = f"      {nm:<15s}"
            for H in T:
                a, b = c(H, nm, "n_head"), c(H, nm, "n_fallback")
                line += (f"| {a}/{b}".ljust(20) if a is not None else "|" + " " * 19)
            P(line)
    P()

    # -------------------------------------------------- the transfer table (the headline shape)
    P("--- TRANSFER: at each tempo, the program against what a learner without practice there "
      "has " + "-" * 8)
    P("    ratios > 1 mean the program is WORSE. `scl0` is the deployable alternative; `rec` is "
      "the oracle.")
    P("      level | " + "  ".join(f"H={H}{'(P)' if H in prac else '   '}" for H in T))
    for e in range(1, L + 1):
        for tag, ref in (("prog/scl0", f"aud_s0_L{e}"), ("prog/rec ", f"aud_L{e}"),
                         ("gate/scl0", f"aud_s0_L{e}")):
            src = f"proggate_L{e}" if tag.startswith("gate") else f"prog_L{e}"
            line = f"      L{e} {tag} | "
            for H in T:
                a, b = c(H, src), c(H, ref)
                line += (f"{a / b:8.2f}x  " if (a and b) else "         .  ")
            P(line)
    P()

    # ------------------------------------------------------------------ the Delta instrument
    P("--- the Delta INSTRUMENT, read at L1 and L2 ONLY " + "-" * 51)
    P("    (under app=fixed a level-4 arm decides once from a lead-in that does not move with "
      "Delta and is")
    P("    Delta-flat BY CONSTRUCTION — t1 measured key_L4 spreads of 1.00x-1.09x. L1 re-decides "
      "8 times, L2 4.)")
    inames = ["reflex", "reflex_ol", "key_L1", "aud_L1", "aud_s0_L1", "prog_L1",
              "key_L2", "aud_L2", "aud_s0_L2", "prog_L2"]
    for H in T:
        P(f"      H={H}{'  (practiced)' if H in prac else ''}")
        for nm in inames:
            line = f"        {nm:<12s}"
            ok = False
            for D in cfg["deltas"]:
                v = [r for r in d["instrument"]
                     if r["H"] == H and r["arm"] == nm and r["delta"] == D]
                if not v:
                    line += "|            "
                    continue
                ok = True
                line += f"| D{D}: {v[0]['e_piece']:.4f}{'*' if v[0]['e_piece'] <= band else ' '}"
            if ok:
                P(line)
    P()
    P("--- THE METRONOME LADDER: how far ahead of the practiced tempo the gate opens " + "-" * 22)
    P("    (`inc_i` practices tempos[:i+1] and is ASKED at tempos[i+1]. `n_dec` is the number of")
    P("    STALE-READ decisions that level makes per traversal — the quantity the deep-best /")
    P("    shallow-worst ordering would be explained by, if anything explains it.)")
    specs = {a: b for a, b in d.get("head_specs", [])}
    inc = sorted([k for k in specs if k.startswith("inc")],
                 key=lambda k: len(specs[k]))
    if inc:
        P("      head  practiced        asked at  step  | level  n_dec | parity open/checked  "
          "head/tape        | sweep e_piece   aud_L*    band?")
        for nm in inc:
            prac = specs[nm]
            i = len(prac) - 1
            if i + 1 >= len(T):
                continue
            Ha = T[i + 1]
            step = float(prac[-1]) / float(Ha)
            for e in range(1, L + 1):
                rr = [r for r in d["parity"]["rows"] if r["head"] == nm and r["ref"] == "rec"
                      and r["H"] == Ha and r["level"] == e]
                sw, au = c(Ha, f"{nm}_L{e}"), c(Ha, f"aud_L{e}")
                if sw is None:
                    sw = c(Ha, f"prog_L{e}")
                op = f"{sum(1 for r in rr if r['open'])}/{len(rr)}" if rr else "  -  "
                ht = (f"{sum(r['mean_e_head'] for r in rr) / len(rr):.4f}/"
                      f"{sum(r['mean_e_tape'] for r in rr) / len(rr):.4f}" if rr else "")
                P(f"      {nm:<5s} {str(prac):<16s} {Ha:>8d}  {step:.2f}  |  L{e}    "
                  f"{1 << (L - e):>4d} | {op:>13s}  {ht:<16s} | "
                  f"{'' if sw is None else f'{sw:.4f}'}"
                  f"{'*' if (sw is not None and sw <= band) else ' '}   "
                  f"{'' if au is None else f'{au:.4f}'}")
    P()
    P("--- P-E: what an open-loop unit costs per unit of command error " + "-" * 36)
    P("    (the library's OWN best member, perturbed by eps and executed from held-out launch "
      "states)")
    eps = sorted({r["eps"] for r in d.get("P_E", [])})
    for H in T:
        for ell in range(1, L + 1):
            rr = {r["eps"]: r["e_mean"] for r in d.get("P_E", [])
                  if r["H"] == H and r["level"] == ell}
            if rr:
                P(f"      H={H:2d} L{ell}  " + "  ".join(
                    f"eps={e:.2f}: {rr[e]:.4f}{'*' if rr[e] <= band else ' '}"
                    for e in eps if e in rr))
    P()
    P("--- the head's command error: train vs held-out, per tempo " + "-" * 41)
    for nm in d["training"]:
        r = d["training"][nm]
        P(f"      {nm:<8s} practiced {r['practiced']}")
        P(f"        train    " + "  ".join(
            f"H{H}: {r['by_tempo'].get(str(H), r['by_tempo'].get(H, float('nan'))):.6f}"
            for H in T if str(H) in r["by_tempo"] or H in r["by_tempo"]))
        ho = r.get("hold_mse_by_tempo", {})
        P(f"        held-out " + "  ".join(
            f"H{H}: {ho.get(str(H), ho.get(H, float('nan'))):.6f}" for H in T))
    P()
    P("--- timings " + "-" * 87)
    P("      " + "  ".join(f"{k}={v:.0f}s" for k, v in d.items()
                           if k.startswith("t_") and isinstance(v, (int, float))))
    return out


# --------------------------------------------------------------------------- figures (REMOTE)

@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_program_figures(tag: str) -> list:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_accelerando", tag)
    figdir = os.path.join(root, "figs")
    os.makedirs(figdir, exist_ok=True)
    d = json.load(open(os.path.join(root, "program.json")))
    cfg = d["config"]
    T = cfg["tempos"]
    L = int(cfg["n_levels"])
    prac = set(cfg["practiced"])
    band = d["piece_by_tempo"][str(T[0])]["band"]
    ms = [1000.0 * H * 0.024 for H in T]
    made = []
    CL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name), dpi=140)
        plt.close(fig)
        made.append(name)

    def c(H, nm, k="e_piece"):
        for r in d["sweep"]:
            if r["H"] == H and r["arm"] == nm and r["app"] == "fixed" \
                    and r["delta"] == cfg["delta_fix"]:
                return r.get(k)
        return None

    def shade(a):
        for H, m in zip(T, ms):
            if H in prac:
                a.axvspan(m / 1.19, m * 1.19, color="#eef3f8", zorder=0)

    # --- fig1: the program against the recording and the resampled tape, per level ----------
    fig, ax = plt.subplots(1, L, figsize=(3.7 * L, 4.4), sharey=True)
    for e in range(1, L + 1):
        a = ax[e - 1]
        shade(a)
        for nm, lab, col, st in ((f"aud_L{e}", "rec (this tempo — oracle where unpracticed)",
                                  CL[0], "-o"),
                                 (f"aud_s0_L{e}", "scl0 (slow tape, resampled)", CL[2], "-^"),
                                 (f"prog_L{e}", "prog (head, ungated)", CL[3], "-s"),
                                 (f"proggate_L{e}", "prog gated by parity vs scl0", CL[4], "--D"),
                                 (f"prog2_L{e}", "prog2 (two tempi practiced)", CL[5], ":v"),
                                 (f"progall_L{e}", "progall (every tempo — ORACLE)", "#777",
                                  ":x")):
            v = [c(H, nm) for H in T]
            if all(x is None for x in v):
                continue
            a.plot(ms, v, st, ms=5, color=col, lw=1.5, label=lab)
        a.plot(ms, [c(H, "reflex") for H in T], "k--", lw=1.4, label="reflex")
        a.axhline(band, color="#888", lw=1.0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xticks(ms); a.set_xticklabels([f"{m:.0f}" for m in ms])
        a.set_xlabel("note length (ms) — faster to the left")
        a.set_title(f"level {e} ({1 << (e - 1)} notes)", fontsize=10)
        if e == 1:
            a.set_ylabel("e_piece (m)")
            a.legend(fontsize=7)
    save(fig, "fig1_program_vs_recording.png")

    # --- fig2: parity open fractions ---------------------------------------------------------
    heads = sorted({r["head"] for r in d["parity"]["rows"]})
    fig, ax = plt.subplots(len(heads), 2, figsize=(11, 4.0 * len(heads)), squeeze=False)
    for i, nm in enumerate(heads):
        for j, ref in enumerate(("rec", "scl0")):
            a = ax[i][j]
            shade(a)
            for e in range(1, L + 1):
                v = []
                for H in T:
                    rr = [r for r in d["parity"]["rows"] if r["head"] == nm and r["ref"] == ref
                          and r["H"] == H and r["level"] == e]
                    v.append(sum(1 for r in rr if r["open"]) / len(rr) if rr else None)
                a.plot(ms, v, "-o", ms=5, color=CL[e - 1], label=f"L{e}")
            a.axhline(0.5, color="#ccc", lw=0.8)
            a.set_xscale("log"); a.set_xticks(ms)
            a.set_xticklabels([f"{m:.0f}" for m in ms])
            a.set_ylim(-0.05, 1.05)
            a.set_xlabel("note length (ms)")
            a.set_ylabel("fraction of slots open")
            a.set_title(f"parity: {nm} vs {ref}", fontsize=10)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    save(fig, "fig2_parity.png")

    # --- fig3: the pool's spread against the head's loss --------------------------------------
    fig, a = plt.subplots(figsize=(7.2, 4.4))
    cs = [sum(q["cmd_sd"] for q in d["pool"][str(H)]) / len(d["pool"][str(H)]) for H in T]
    cr = [sum(q["cmd_rms"] for q in d["pool"][str(H)]) / len(d["pool"][str(H)]) for H in T]
    shade(a)
    a.plot(ms, [x / y for x, y in zip(cs, cr)], "-o", color=CL[0], label="pool cmd sd / rms")
    a.set_xscale("log"); a.set_xticks(ms); a.set_xticklabels([f"{m:.0f}" for m in ms])
    a.set_xlabel("note length (ms)"); a.set_ylabel("pool rendition spread (relative)")
    a2 = a.twinx()
    tr = d["training"].get("prog") or d["training"][sorted(d["training"])[0]]
    hm = [tr["by_tempo"].get(str(H), tr["by_tempo"].get(H)) for H in T]
    bz = [(tr.get("by_tempo_baseline", {}).get(str(H)) or {}).get("zero") for H in T]
    a2.plot(ms, hm, "-s", color=CL[3], label="head mse (practiced only)")
    a2.plot(ms, bz, ":s", color="#999", label="mse of predicting nothing")
    a2.set_ylabel("head training mse")
    a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    a.set_title("the practice pool's diversity and the head's loss", fontsize=10)
    save(fig, "fig3_pool_and_loss.png")

    # --- fig4: the Delta instrument at L1 and L2 ----------------------------------------------
    ds = cfg["deltas"]
    fig, ax = plt.subplots(2, len(T), figsize=(3.0 * len(T), 7.2), sharey=True, squeeze=False)
    for i, e in enumerate((1, 2)):
        for j, H in enumerate(T):
            a = ax[i][j]
            for nm, col, st in (("reflex", "k", "--o"), (f"aud_L{e}", CL[0], "-o"),
                                (f"aud_s0_L{e}", CL[2], "-^"), (f"prog_L{e}", CL[3], "-s")):
                v = []
                for D in ds:
                    r = [x for x in d["instrument"] if x["H"] == H and x["arm"] == nm
                         and x["delta"] == D]
                    v.append(r[0]["e_piece"] if r else None)
                if all(x is None for x in v):
                    continue
                a.plot([D * 24 for D in ds], v, st, ms=4, color=col, lw=1.3, label=nm)
            a.axhline(band, color="#888", lw=1.0)
            a.axvline(cfg["delta_fix"] * 24, color="#c33", lw=1.0, ls=":")
            a.set_yscale("log")
            a.set_title(f"L{e}  H={H}{' (P)' if H in prac else ''}", fontsize=9)
            a.set_xlabel("Delta (ms)")
            if j == 0:
                a.set_ylabel("e_piece (m)")
                a.legend(fontsize=7)
    save(fig, "fig4_delta_instrument_L1_L2.png")

    # --- fig5: the two grades ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for j, e in enumerate((1, L)):
        a = ax[j]
        for nm, col in ((f"aud_L{e}", CL[0]), (f"aud_s0_L{e}", CL[2]), (f"prog_L{e}", CL[3]),
                        (f"proggate_L{e}", CL[4]), ("reflex", "k")):
            xs = [c(H, nm) for H in T]
            ys = [c(H, nm, "e_listen") for H in T]
            if any(x is None for x in xs):
                continue
            a.plot(xs, ys, "-o", ms=5, color=col, label=nm)
        lim = [band / 20, band * 12]
        a.plot(lim, lim, color="#bbb", lw=1.0)
        a.axhline(band, color="#888", lw=0.8); a.axvline(band, color="#888", lw=0.8)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("e_piece"); a.set_ylabel("e_listen")
        a.set_title(f"the two grades, level {e}", fontsize=10)
        a.legend(fontsize=7.5)
    save(fig, "fig5_two_grades.png")

    volume.commit()
    return made


@app.local_entrypoint()
def accelerando_prog_figs(tag: str = "p1"):
    made = make_program_figures.remote(tag)
    print(f"[figs] wrote {made} to /data/practice_accelerando/{tag}/figs")


def figures(tag):
    here = os.path.relpath(os.path.abspath(__file__),
                           os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    subprocess.run(["modal", "run", f"{here}::accelerando_prog_figs", "--tag", tag], check=True,
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
    ap.add_argument("--tag", default="p1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    path = (fetch(args.tag) if args.fetch
            else os.path.join(HERE, "results", args.tag, "program.json"))
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
