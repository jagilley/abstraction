"""Reduce rubato's Phase 1 run (the kinematic unit through a perfect cerebellum, across tempo)
to a text report and figures.

Pure python for the reduction (json + math only): the local client has neither numpy nor
matplotlib. `--figures` renders remotely on Modal and pulls the PNGs back.

    python3 mjc/practice/tempo/rubato/analyze_kin.py --tag k1 --fetch
    python3 mjc/practice/tempo/rubato/analyze_kin.py --tag k1 --figures
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

# the content sources, in the order every table prints them
SRC_DESC = {
    "rec": "same-tempo recording (ORACLE at an unpracticed tempo)",
    "s0": "slow tape, time-resampled (scl0)",
    "s2": "slow tape, resampled x s^2 (scl2)",
    "kzs": "KINEMATIC, exact (zoh) inverse, from the SLOWEST tempo's paths",
    "kzo": "kinematic, exact (zoh), from THIS tempo's own paths (s = 1) — conversion-cost control",
    "kzsa": "kinematic, exact (zoh), slow paths, ANTI-ALIAS low-pass (1 destination step)",
    "kzsf": "kinematic, exact (zoh), slow paths, NOTE-PERIOD low-pass — decision 20's treatment",
    "kzof": "kinematic, exact (zoh), own paths, note-period low-pass — the filter's own cost",
    "kzoa": "kinematic, exact (zoh), own paths, anti-alias (identity at s = 1)",
    "kfl": "FITTED linear-in-(v,a) executor, slow paths, anti-alias",
    "kfm": "FITTED MLP executor, slow paths, anti-alias",
    "kcs": "kinematic, continuous-time (ct) inverse, from the slowest tempo's paths",
    "kco": "kinematic, continuous-time (ct) inverse, from this tempo's own paths",
}
ORDER = ["rec", "s0", "s2", "kzs", "kzo", "kzoa", "kzsa", "kzsf", "kzof", "kfl", "kfm",
         "kcs", "kco"]


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL,
                    f"/practice_rubato/{tag}/kin.json",
                    os.path.join(dst, "kin.json"), "--force"], check=True)
    return os.path.join(dst, "kin.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    T = list(cfg["tempos"])
    L = int(cfg["n_levels"])
    dt = 0.024
    sw = d.get("sweep", [])
    # `fixed` is the ARM OF RECORD wherever it was run (prestissimo decision 20, accelerando
    # decision 11); `arm` is the continuity row and is reported separately, never as the table.
    appm = "fixed" if "fixed" in cfg["app_modes"] else cfg["app_modes"][0]
    alt = [m for m in cfg["app_modes"] if m != appm]
    present = {r["kind"] for r in sw}
    SRC = [(k, SRC_DESC[k]) for k in ORDER if k in present or k == "rec"]

    def row(H, arm):
        for r in sw:
            if r["H"] == H and r["arm"] == arm and r["app"] == appm:
                return r
        return None

    def band(H):
        return d["piece_by_tempo"][str(H)]["band"]

    P(f"# rubato / {cfg['tag']} — Phase 1: the ceiling of the factored program")
    P()
    P(f"seed {cfg['seed']} · R* {cfg['r_star']} · tempi {T} · Delta {cfg['delta_fix']} "
      f"({1000 * cfg['delta_fix'] * dt:.0f} ms) · app={appm} · "
      f"{d.get('wall_s', 0):.0f}s (library {d.get('t_library', 0):.0f} · "
      f"kin {d.get('t_kin', 0):.0f} · sweep {d.get('t_sweep', 0):.0f})")
    P(f"complete={d.get('complete')}")
    P()

    # ------------------------------------------------------------------ gates
    P("## Gates")
    P()
    P("| gate | what | result |")
    P("|---|---|---|")
    f0 = d.get("P_F0", {})
    P(f"| P-F0 | the 13 donor constants | {f0.get('checked')} checked, "
      f"{len(f0.get('mismatches', {}))} bad — **{'pass' if f0.get('pass') else 'FAIL'}** |")
    t1 = d.get("P_T1", {})
    P(f"| P-T1 | with every new arm off, this fork reproduces `accelerando/t1` | "
      f"{t1.get('n_checks')} checks, max\\|delta\\| = {t1.get('max_abs', -1):.3e}, "
      f"applicable={t1.get('applicable')} — "
      f"**{'pass' if t1.get('pass') else ('n/a' if not t1.get('applicable') else 'FAIL')}** |")
    pg = d.get("P_G", {})
    P(f"| P-G | one pinned gain cell re-fit from the full grid | cell {pg.get('cell')}: "
      f"re-fit {pg.get('refit')} vs pinned {pg.get('pinned')}, applicable={pg.get('applicable')}"
      f" — **{'pass' if pg.get('pass') else ('n/a' if not pg.get('applicable') else 'FAIL')}** |")
    tau = d.get("P_TAU", {})
    P(f"| P-TAU | tau measured from a step response | fast "
      f"{1000 * tau.get('fast', {}).get('tau_fit', 0):.2f} ms vs "
      f"{1000 * tau.get('fast', {}).get('tau_expected', 0):.2f}; donor "
      f"{1000 * tau.get('donor', {}).get('tau_fit', 0):.1f} vs "
      f"{1000 * tau.get('donor', {}).get('tau_expected', 0):.1f} — "
      f"**{'pass' if tau.get('pass') else 'FAIL'}** |")
    pz = d.get("P_ZOH", {})
    P(f"| P-ZOH | the discretisation the oracle inverts is the plant's | alpha measured "
      f"{pz.get('alpha_fit', 0):.8f} vs analytic {pz.get('alpha_analytic', 0):.8f} "
      f"(rel {pz.get('alpha_rel', 0):.1e}) — **{'pass' if pz.get('pass') else 'FAIL'}** |")
    pr = d.get("P_R", {})
    P(f"| P-R | the command resampler is the identity at s = 1 | "
      f"max\\|delta\\| = {pr.get('max_abs', -1):.3e} over {pr.get('n_members')} members — "
      f"**{'pass' if pr.get('pass') else 'FAIL'}** |")
    pn = d.get("P_N", {})
    worst = max((v.get("exec_max_abs", 0) for v in pn.get("by_tempo", {}).values()), default=-1)
    nbad = sum(v.get("spell_bad", 0) for v in pn.get("by_tempo", {}).values())
    nchk = sum(v.get("n_checked", 0) for v in pn.get("by_tempo", {}).values())
    P(f"| P-N | the nesting op is exact at every tempo | {nchk} spelled, {nbad} bad, weld "
      f"max\\|delta\\| = {worst:.2e} — **{'pass' if pn.get('pass') else 'FAIL'}** |")
    pkn = d.get("P_KN", {})
    P(f"| P-KN | the phase resampler splits at the span midpoint | "
      f"max\\|delta\\| = {pkn.get('max_abs', -1):.3e} over {pkn.get('n')} members — "
      f"**{'pass' if pkn.get('pass') else 'FAIL'}** |")
    pk = d.get("P_K", {})
    P(f"| P-K | THE ORACLE IDENTITY, outside the rotation region | out max\\|du\\| = "
      f"{pk.get('out_max', -1):.3e} (tol {pk.get('tol', 0):.0e}), rms "
      f"{pk.get('out_rms', -1):.2e} over {pk.get('n_out')} steps — "
      f"**{'pass' if pk.get('pass') else 'FAIL'}** |")
    P()
    P("**P-K, the residual against the world's own Gaussian gate weight.** The body model is "
      "the world model exactly where the command-rotation region is off; this is the table that "
      "says the residual is the rotation and nothing else.")
    P()
    P("| gate weight w | steps | rms \\|du\\| | max \\|du\\| |")
    P("|---|---|---|---|")
    for b in pk.get("buckets", []):
        P(f"| {b['w_lo']:.0e} - {b['w_hi']:.0e} | {b['n']} | {b['rms']:.2e} | {b['max']:.2e} |")
    P()
    P(f"In-region residual (REPORTED, not gated): rms {pk.get('in_rms', 0):.3e}, max "
      f"{pk.get('in_max', 0):.3e} over {pk.get('n_in')} steps.")
    P()
    P("**The two inverse schemes, algebraically.** Both are `u = c_a * a + c_v * v` and they "
      f"agree exactly on drag (c_v = {pz.get('drag_coeff', 0):.6f} = 1/v_term). They differ "
      f"only on acceleration: zoh {pz.get('accel_coeff_zoh', 0):.6f} against ct "
      f"{pz.get('accel_coeff_ct', 0):.6f}, a ratio of "
      f"**{pz.get('scheme_ratio', 0):.4f}**, because dt_ctrl/tau = "
      f"{pz.get('dt_over_tau', 0):.3f} is not small on this body.")
    P()
    fw = pz.get("clean", {})
    fr = pz.get("rotated", {})
    P(f"**The forward step's one-step residual** (the same model, run forwards): clean world "
      f"\\|dv\\| rms {fw.get('dv_rms', 0):.3e} / max {fw.get('dv_max', 0):.3e}; ROTATED world "
      f"rms {fr.get('dv_rms', 0):.3e} / max {fr.get('dv_max', 0):.3e}, at a velocity scale of "
      f"{fw.get('v_scale', 0):.3f} m/s. The body model does not contain the world's "
      "command-rotation region, by construction.")
    P()

    # ------------------------------------------------------------------ the piece, per tempo
    P("## The piece at each tempo")
    P()
    P("| H | note ms | note/tau | speed m/s | u_turn (schedule) | band | do-nothing | "
      "reflex | reflex_ec | reflex_ec0 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for H in T:
        pc = d["piece_by_tempo"][str(H)]
        dn = d.get("do_nothing", {}).get(str(H), {})
        rr = row(H, "reflex")
        re1 = row(H, "reflex_ec")
        re0 = row(H, "reflex_ec0")
        P(f"| {H} | {pc['seg_ms']:.0f} | {1 / pc['tau_over_note']:.2f} | {pc['speed']:.2f} | "
          f"{pc['turn']['u_turn_max']:.2f} | {pc['band']:.4f} | "
          f"{dn.get('e_piece', float('nan')):.4f} | "
          f"{(rr or {}).get('e_piece', float('nan')):.4f} | "
          f"{(re1 or {}).get('e_piece', float('nan')):.4f} | "
          f"{(re0 or {}).get('e_piece', float('nan')):.4f} |")
    P()
    P("`reflex_ec` / `reflex_ec0` are the REPORTED REFERENCE ROWS and are never headlined: the "
      "incumbent handed the same body model the learner holds, used only to carry its "
      "Delta-stale read forward along the commands it already issued (`accompanist` d3b's "
      "operator). `ec` keeps the Delta-fit gains; `ec0` plays the Delta = 0 gains an un-delayed "
      "read licenses.")
    P()

    # ------------------------------------------------------------------ the continuity row
    if alt:
        P("## The continuity row (`app = arm`)")
        P()
        P("`app = fixed` is the arm of record everywhere above and below. `app = arm` — the "
          "lead-in read at the arm's own Delta, playing the per-Delta re-fit gains — is run at "
          "the named tempi only, because `prestissimo` decision 20 and `accelerando` flag 1 both "
          "measured it to blow the lead-in up on this body (hand-over 3.3-5.3 legs against "
          "0.18-0.25 under `fixed`). It is here so the series stays comparable with `t1`'s, not "
          "as a second sweep.")
        P()
        for m in alt:
            hs = sorted({r["H"] for r in sw if r["app"] == m})
            if not hs:
                continue
            P(f"| H | e_app | " + " | ".join(
                ["reflex"] + [f"aud_{k}_L{L}" for k in ("rec" if False else "kzsa",)]
                + [f"aud_L{L}", f"aud_s0_L{L}", f"aud_kzs_L{L}"]) + " |")
            P("|---|---|" + "---|" * 5)
            for H in hs:
                def g(nm):
                    for r in sw:
                        if r["H"] == H and r["app"] == m and r["arm"] == nm:
                            return r
                    return None
                rr = g("reflex")
                P(f"| {H} | {(rr or {}).get('e_app', float('nan')):.4f} | "
                  + " | ".join(
                      f"{(g(nm) or {}).get('e_piece', float('nan')):.4f}"
                      for nm in ("reflex", f"aud_kzsa_L{L}", f"aud_L{L}", f"aud_s0_L{L}",
                                 f"aud_kzs_L{L}")) + " |")
            P()

    # ------------------------------------------------------------------ the main table
    for grade, key, bkey in (("e_piece (per-waypoint, drilled)", "e_piece", "in_band"),
                             ("e_listen (the listener's clock)", "e_listen", "lis_band")):
        P(f"## The sweep, {grade}")
        P()
        for mode in ("aud", "key"):
            P(f"### `{mode}` — " + ("plant audition at the seam" if mode == "aud"
                                    else "frozen posture key, no audition"))
            P()
            hdr = "| H | band | reflex |" + "".join(
                f" {s}_L{e} |" for s, _ in SRC for e in range(1, L + 1))
            P(hdr)
            P("|---|---|---|" + "---|" * (len(SRC) * L))
            for H in T:
                cells = [f"| {H} | {band(H):.4f} | "
                         f"{(row(H, 'reflex') or {}).get(key, float('nan')):.4f} |"]
                for s, _ in SRC:
                    for e in range(1, L + 1):
                        nm = f"{mode}_L{e}" if s == "rec" else f"{mode}_{s}_L{e}"
                        r = row(H, nm)
                        v = (r or {}).get(key)
                        star = "*" if (r or {}).get(bkey, 0) is not None and v is not None \
                            and v <= band(H) else ""
                        cells.append(f" {v:.4f}{star} |" if v is not None else " — |")
                P("".join(cells))
            P()
        P("`*` = inside the band.")
        P()
        P(f"**Pass fractions at three widths** ({'per-waypoint' if key == 'e_piece' else 'listener'}"
          " grade): fraction of the 32 shared eval geometries inside 1/2, 1/3 and 1/4 of a mean "
          "drilled leg. `aud` mode, deepest and shallowest rung plus the incumbent.")
        P()
        pf = ("in_h", "in_t", "in_q") if key == "e_piece" else ("lis_h", "lis_t", "lis_q")
        P("| H | arm | 1/2 leg | 1/3 leg | 1/4 leg |")
        P("|---|---|---|---|---|")
        for H in T:
            names = ["reflex", "reflex_ec0"]
            names += [f"aud_L{e}" for e in (1, L)]
            names += [f"aud_s0_L{e}" for e in (1, L)]
            for kk in ("kzs", "kzsa", "kzsf", "kzo", "kzof", "kcs"):
                if kk in present:
                    names += [f"aud_{kk}_L{e}" for e in range(1, L + 1)]
            for nm in names:
                r = row(H, nm)
                if not r:
                    continue
                P(f"| {H} | {nm} | " + " | ".join(f"{r.get(k2, float('nan')):.2f}"
                                                  for k2 in pf) + " |")
        P()

    # ------------------------------------------------------------------ the transfer question
    P("## The transfer question, isolated")
    P()
    P("Three numbers per (tempo, level), all `aud`, all in `e_piece`: the ORACLE content "
      "harvested at the asked tempo (`rec` — not deployable), the same-tempo kinematic "
      "conversion (`kzo`, s = 1 — the CONVERSION COST alone), and the kinematic unit carried "
      "from the slowest tempo (`kzs` — the transfer). `kzs / kzo` is what the tempo change "
      "costs once the conversion is paid for; `kzo / rec` is what the conversion costs.")
    P()
    P("| H | L | band | rec | kzo | kzs | scl0 | kzo/rec | kzs/kzo | kzs/scl0 | sat(kzs) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    sat = d.get("saturation", [])
    for H in T:
        for e in range(1, L + 1):
            g = {}
            for s in ("rec", "kzo", "kzs", "s0"):
                nm = f"aud_L{e}" if s == "rec" else f"aud_{s}_L{e}"
                g[s] = (row(H, nm) or {}).get("e_piece")
            ss = [r["sat"] for r in sat
                  if r["arm"] == "kzs" and r["H"] == H and r["level"] == e and not r["poison"]]
            sm = sum(ss) / len(ss) if ss else float("nan")

            def rat(a, b):
                return f"{a / b:.2f}x" if (a and b and b > 0) else "—"
            b_ = band(H)
            P(f"| {H} | {e} | {b_:.4f} | "
              + " | ".join(f"{g[s]:.4f}" + ("*" if g[s] is not None and g[s] <= b_ else "")
                           if g[s] is not None else "—" for s in ("rec", "kzo", "kzs", "s0"))
              + f" | {rat(g['kzo'], g['rec'])} | {rat(g['kzs'], g['kzo'])} | "
                f"{rat(g['kzs'], g['s0'])} | {sm:.3f} |")
    P()

    # ------------------------------------------------------------------ the scheme comparison
    P("## The derivative scheme, measured")
    P()
    P("The exact (`zoh`) inverse against the continuous-time (`ct`) one the brief names, at the "
      "same content and the same source. They differ only in the coefficient on acceleration, "
      f"by {pz.get('scheme_ratio', 0):.4f}x.")
    P()
    P("| H | L | kzo | kco | kco/kzo | kzs | kcs | kcs/kzs |")
    P("|---|---|---|---|---|---|---|---|")
    for H in T:
        for e in range(1, L + 1):
            g = {s: (row(H, f"aud_{s}_L{e}") or {}).get("e_piece")
                 for s in ("kzo", "kco", "kzs", "kcs")}

            def rat(a, b):
                return f"{a / b:.2f}x" if (a and b and b > 0) else "—"
            P(f"| {H} | {e} | "
              + " | ".join(f"{g[s]:.4f}" if g[s] is not None else "—"
                           for s in ("kzo", "kco"))
              + f" | {rat(g['kco'], g['kzo'])} | "
              + " | ".join(f"{g[s]:.4f}" if g[s] is not None else "—"
                           for s in ("kzs", "kcs"))
              + f" | {rat(g['kcs'], g['kzs'])} |")
    P()

    # ------------------------------------------------------------------ saturation
    P("## Saturation: the actuator ceiling")
    P()
    P("Fraction of scalar command components the clip bound when the path was converted, per "
      "(arm, tempo, level), non-poison members only.")
    P()
    P("| arm | " + " | ".join(f"H{H} L{e}" for H in T for e in range(1, L + 1)) + " |")
    P("|---|" + "---|" * (len(T) * L))
    for tag in sorted({r["arm"] for r in sat}):
        cells = []
        for H in T:
            for e in range(1, L + 1):
                ss = [r["sat"] for r in sat if r["arm"] == tag and r["H"] == H
                      and r["level"] == e and not r["poison"]]
                cells.append(f"{sum(ss) / len(ss):.3f}" if ss else "—")
        P(f"| {tag} | " + " | ".join(cells) + " |")
    P()

    # ------------------------------------------------------------------ the era table
    P("## The era table: which content wins at each tempo")
    P()
    P("Best arm at each tempo under each grade, over every committed arm (the incumbent and its "
      "two instruments excluded), and the same over the DEPLOYABLE arms only — those whose "
      "content exists without practising at the asked tempo (`s0`, `s2`, `kzs`, `kcs`).")
    P()
    # DEPLOYABLE = content that exists without practising at the asked tempo. Derived from the
    # arms actually present rather than hard-coded, so an arm added later cannot be silently
    # left out of the table it belongs in — `k1`'s first reduction excluded `kzsa`, the arm of
    # record, exactly that way.
    OWN = {"rec", "kzo", "kzoa", "kzof", "kco"}
    dep = tuple(k for k in present if k not in OWN)
    P("| H | best (any) | e | in band | best DEPLOYABLE | e | in band | reflex | ratio |")
    P("|---|---|---|---|---|---|---|---|---|")
    for H in T:
        cand = [r for r in sw if r["H"] == H and r["app"] == appm
                and r["kind"] not in ("reflex", "reflex_ol", "reflex_ec", "reflex_ec0")]
        depc = [r for r in cand if r["kind"] in dep]
        rr = (row(H, "reflex") or {}).get("e_piece", float("nan"))
        if not cand:
            continue
        b1 = min(cand, key=lambda r: r["e_piece"])
        b2 = min(depc, key=lambda r: r["e_piece"]) if depc else None
        P(f"| {H} | {b1['arm']} | {b1['e_piece']:.4f} | "
          f"{'yes' if b1['e_piece'] <= band(H) else 'no'} | "
          + (f"{b2['arm']} | {b2['e_piece']:.4f} | "
             f"{'yes' if b2['e_piece'] <= band(H) else 'no'} | " if b2 else "— | — | — | ")
          + f"{rr:.4f} | "
          + (f"{rr / b2['e_piece']:.2f}x |" if b2 and b2["e_piece"] > 0 else "— |"))
    P()

    # ------------------------------------------------------------------ parity
    par = d.get("parity")
    if par:
        P("## Parity per slot against the same-tempo recording")
        P()
        P(f"A slot is OPEN at a tempo iff the arm's content, EXECUTED ON THE PLANT from that "
          f"slot's HELD-OUT launch states, is at least as good as the same-tempo recording's own "
          f"audition winner on at least tau = {par.get('tau')} of them. `solo`'s gate with the "
          "tempo added to the index; the reference is the per-state winner, not a fixed member. "
          "The reference side is an instrument and is not charged.")
        P()
        P("| arm | " + " | ".join(f"H{H} L{e}" for H in T for e in range(1, L + 1)) + " |")
        P("|---|" + "---|" * (len(T) * L))
        arms = sorted({r["arm"] for r in par.get("rows", [])})
        for a2 in arms:
            cells = []
            for H in T:
                for e in range(1, L + 1):
                    rs = [r for r in par["rows"] if r["arm"] == a2 and r["H"] == H
                          and r["level"] == e]
                    if not rs:
                        cells.append("—")
                    else:
                        cells.append(f"{sum(1 for r in rs if r['open'])}/{len(rs)}")
            P(f"| {a2} | " + " | ".join(cells) + " |")
        P()
        P("Mean executed error on the held-out states, arm vs the recording it is measured "
          "against (arm / reference):")
        P()
        P("| arm | " + " | ".join(f"H{H} L{e}" for H in T for e in range(1, L + 1)) + " |")
        P("|---|" + "---|" * (len(T) * L))
        for a2 in arms:
            cells = []
            for H in T:
                for e in range(1, L + 1):
                    rs = [r for r in par["rows"] if r["arm"] == a2 and r["H"] == H
                          and r["level"] == e]
                    if not rs:
                        cells.append("—")
                    else:
                        ma = sum(r["mean_e_arm"] for r in rs) / len(rs)
                        mr = sum(r["mean_e_ref"] for r in rs) / len(rs)
                        cells.append(f"{ma:.4f}/{mr:.4f}")
            P(f"| {a2} | " + " | ".join(cells) + " |")
        P()

    # ------------------------------------------------------------------ the learned cerebellum
    ft = d.get("fit")
    if ft:
        P("## Phase 2 — the learned cerebellum")
        P()
        lr = ft.get("lin", {})
        P(f"**The linear family's recovered coefficients**, against the analytic inverse it is "
          f"trying to be. Fitted as a general 2x4 map with a bias, so the isotropy and the "
          f"which-channel-depends-on-what were NOT given to it.")
        P()
        P("| | fitted (x, y) | analytic |")
        P("|---|---|---|")
        P(f"| drag `c_v` | {lr.get('c_v_fit', [0, 0])[0]:.6f}, "
          f"{lr.get('c_v_fit', [0, 0])[1]:.6f} | {lr.get('c_v_true', 0):.6f} |")
        P(f"| accel `c_a` | {lr.get('c_a_fit', [0, 0])[0]:.6f}, "
          f"{lr.get('c_a_fit', [0, 0])[1]:.6f} | {lr.get('c_a_true', 0):.6f} |")
        P(f"| off-diagonal (max abs) | {lr.get('off_diag_max', 0):.2e} | 0 |")
        P(f"| bias (max abs) | {lr.get('bias_max', 0):.2e} | 0 |")
        P()
        lc = ft.get("linc")
        if lc:
            P("**The clean-world discriminator (decision 34)** — the same renditions replayed "
              "with the rotation region OFF, linear family refitted:")
            P()
            P("| fit | c_v | c_a | off-diag max | bias max |")
            P("|---|---|---|---|---|")
            P(f"| rotated world | {lr['c_v_fit'][0]:.6f}, {lr['c_v_fit'][1]:.6f} | "
              f"{lr['c_a_fit'][0]:.6f}, {lr['c_a_fit'][1]:.6f} | {lr['off_diag_max']:.2e} | "
              f"{lr['bias_max']:.2e} |")
            P(f"| **clean world** | {lc['c_v_fit'][0]:.6f}, {lc['c_v_fit'][1]:.6f} | "
              f"{lc['c_a_fit'][0]:.6f}, {lc['c_a_fit'][1]:.6f} | {lc['off_diag_max']:.2e} | "
              f"{lc['bias_max']:.2e} |")
            P()
        hr = d.get("fit_heldout", [])
        if hr:
            P("**Held-out fit at every tempo, in command units**, beside how far outside the "
              "training region the query lands. `reach` is the query's 99.5th-percentile "
              "magnitude over the fit's; `frac out` is the fraction of query points beyond the "
              "fit's 99.5th percentile. A faster tempo scales `|v|` by `s` and `|a|` by `s^2`, "
              "so this is the extrapolation question asked directly.")
            P()
            P("| H | fitted? | lin mse | mlp mse | v reach | a reach | v frac out | a frac out |")
            P("|---|---|---|---|---|---|---|---|")
            for H in T:
                q = {r["family"]: r for r in hr if r["H"] == H}
                if not q:
                    continue
                any_r = list(q.values())[0]
                P(f"| {H} | {'yes' if any_r['fitted'] else 'no'} | "
                  f"{q.get('lin', {}).get('mse', float('nan')):.3e} | "
                  f"{q.get('mlp', {}).get('mse', float('nan')):.3e} | "
                  f"{any_r['v_reach']:.2f}x | {any_r['a_reach']:.2f}x | "
                  f"{any_r['v_frac_out']:.3f} | {any_r['a_frac_out']:.3f} |")
            P()
        P("**How much of the ceiling's in-band set survives.** `kzsa` is the oracle on the same "
          "resampler (decision 22); `kfl` and `kfm` are the two fitted families on exactly that "
          "resampler, so the only difference is where the inverse came from. `aud`, `e_piece`.")
        P()
        P("| H | L | band | kzsa (ceiling) | kfl (linear) | kfm (MLP) | kfl/kzsa | kfm/kzsa |")
        P("|---|---|---|---|---|---|---|---|")
        n_ceil = n_lin = n_mlp = 0
        for H in T:
            for e in range(1, L + 1):
                g = {k2: (row(H, f"aud_{k2}_L{e}") or {}).get("e_piece")
                     for k2 in ("kzsa", "kfl", "kfm")}
                if g["kzsa"] is None:
                    continue
                b_ = band(H)
                n_ceil += int(g["kzsa"] <= b_)
                n_lin += int(g["kfl"] is not None and g["kfl"] <= b_)
                n_mlp += int(g["kfm"] is not None and g["kfm"] <= b_)

                def f2(x):
                    return "—" if x is None else f"{x:.4f}" + ("*" if x <= b_ else "")

                def rt(x):
                    return "—" if (x is None or not g["kzsa"]) else f"{x / g['kzsa']:.2f}x"
                P(f"| {H} | {e} | {b_:.4f} | {f2(g['kzsa'])} | {f2(g['kfl'])} | "
                  f"{f2(g['kfm'])} | {rt(g['kfl'])} | {rt(g['kfm'])} |")
        P()
        P(f"**In-band cells: ceiling {n_ceil}, linear {n_lin}, MLP {n_mlp}** (of "
          f"{len(T) * L}).")
        P()

    # ------------------------------------------------------------------ Phase 3
    cr = d.get("corrected")
    if cr:
        pdg = {r["H"]: r for r in d.get("pd_gains", [])}
        P("## Phase 3 — the corrected executor")
        P()
        P("Feedforward from the kinematic program, corrected online against a forecast rolled "
          "from the Delta-stale read through the commands this executor already issued. `R` is "
          "the read period in control steps; **every read is charged as one feedback event**, "
          "the launch read included.")
        P()
        if pdg and "rho" in list(pdg.values())[0]:
            P("### The PD gains (decision 36: eigenvalues of the exact discrete error system "
              "placed at ρ), per tempo, with edges and the R-invariance verdict")
            P()
            P("| H | ρ\* (R = 1) | kp | kd | e_piece | edge | ρ\* at R = 16 | R-invariant? |")
            P("|---|---|---|---|---|---|---|---|")
            for H in T:
                r = pdg.get(H)
                if not r:
                    continue
                rb = r.get("rho_by_R", {})
                other = [v for k2, v in sorted(rb.items()) if int(k2) != 1]
                inv = bool(len(set(rb.values())) <= 1)
                P(f"| {H} | {r['rho']:g} | {r['kp']:.4g} | {r['kd']:.5g} | {r['e_piece']:.4f} | "
                  f"{'**EDGE**' if r.get('at_edge') else 'no'} | "
                  f"{other[0]:g} | {'yes' if inv else '**no**'} |")
            P()
            P("A negative `kd` is the allowed-and-reported branch of the rule (decision 36), not "
              "a clip. The frozen table is the R = 1 fit.")
            P()
        else:
            P("### The PD gains (decision 32), per tempo, with their edges")
            P()
            P("| H | analytic kp | analytic kd | chosen multiplier | kp | kd | e_piece | "
              "kp edge | kd edge |")
            P("|---|---|---|---|---|---|---|---|---|")
            for H in T:
                r = pdg.get(H)
                if not r:
                    continue
                P(f"| {H} | {r['kp_an']:.4g} | {r['kd_an']:.4g} | ×{r['mkp']:g} / ×{r['mkd']:g} "
                  f"| {r['kp']:.4g} | {r['kd']:.4g} | {r['e_piece']:.4f} | "
                  f"{'**EDGE**' if r['at_kp_edge'] else 'no'} | "
                  f"{'**EDGE**' if r['at_kd_edge'] else 'no'} |")
            P()
        P("### (a) The fewest reads per span at which the corrected unit is in band")
        P()
        P("`x` = never in band at any read count on the ladder. **The ladder is capped at "
          "R = 16 control steps**, so at the slow tempi its coarsest rung is still many reads "
          "per span (16 at H = 32 L4) — those rows are the LADDER's floor, not a measured "
          "minimum. The true one-read points are the open-loop columns.")
        P()
        P("| H | L | exact FM | fitted FM | open `kfl` (1 read) | open `kzsa` (1 read) |")
        P("|---|---|---|---|---|---|")
        for H in T:
            for e in range(1, L + 1):
                cells = []
                for fm in ("exact", "fitted"):
                    rs = [r for r in cr if r["H"] == H and r["ff"] == "kfl"
                          and r["fm"] == fm and r["level"] == e and r["eps"] == 0.0
                          and r["e_piece"] <= band(H)]
                    if rs:
                        b = min(rs, key=lambda r: r["reads_per_span"])
                        cells.append(f"**{b['reads_per_span']}** (R={b['R']}) {b['e_piece']:.4f}")
                    else:
                        cells.append("x")
                kf = row(H, f"key_kfl_L{e}")
                kz = row(H, f"key_kzsa_L{e}")

                def f2(r):
                    if not r:
                        return "—"
                    v = r["e_piece"]
                    return f"{v:.4f}" + ("*" if v <= band(H) else "")
                P(f"| {H} | {e} | {cells[0]} | {cells[1]} | {f2(kf)} | {f2(kz)} |")
        P()
        P("### The full read ladder (exact FM)")
        P()
        P("| H | L | rows as `e_piece` [realised reads per span], coarsest first |")
        P("|---|---|---|")
        for H in T:
            for e in range(1, L + 1):
                q = sorted([r for r in cr if r["H"] == H and r["ff"] == "kfl"
                            and r["fm"] == "exact" and r["level"] == e and r["eps"] == 0.0],
                           key=lambda r: r["reads_per_span"])
                P(f"| {H} | {e} | " + "  ".join(
                    f"{r['e_piece']:.4f}" + ("*" if r["e_piece"] <= band(H) else "")
                    + f" [{r['reads_per_span']}]" for r in q) + " |")
        P()
        P("### (c) `reflex_ec` beside the R = 1 row, and the oracle-feedforward control")
        P()
        P("| H | reflex (fb) | reflex_ec | reflex_ec0 | cor densest L1 (fb) | L4 (fb) | "
          "oracle-ff densest L1 | L4 |")
        P("|---|---|---|---|---|---|---|---|")
        for H in T:
            def cc(ff, e):
                q = [r for r in cr if r["H"] == H and r["ff"] == ff and r["fm"] == "exact"
                     and r["level"] == e and r["eps"] == 0.0]
                return max(q, key=lambda r: r["reads_per_span"]) if q else None
            rf = row(H, "reflex")
            a1, a4 = cc("kfl", 1), cc("kfl", L)
            o1, o4 = cc("kzsa", 1), cc("kzsa", L)
            P(f"| {H} | {(rf or {}).get('e_piece', float('nan')):.4f} "
              f"({(rf or {}).get('ledger_drilled', {}).get('n_fb', 0):.0f}) | "
              f"{(row(H, 'reflex_ec') or {}).get('e_piece', float('nan')):.4f} | "
              f"{(row(H, 'reflex_ec0') or {}).get('e_piece', float('nan')):.4f} | "
              + " | ".join(
                  (f"{q['e_piece']:.4f} ({q['n_fb']:.0f})" if q else "—") for q in (a1, a4))
              + " | " + " | ".join((f"{q['e_piece']:.4f}" if q else "—") for q in (o1, o4))
              + " |")
        P()
    pe = d.get("pe")
    if pe:
        eps = d["config"]["pe_eps"]
        P("### (b) P-E, with and without correction")
        P()
        P(f"`e_piece` across ε = {eps} added to the FEEDFORWARD command only. `open` is the same "
          "executor with no reads past launch, so the two sides differ in exactly the "
          "correction. Beneath, the ε at which the 0.0611 band is spent, linearly interpolated "
          "on the ladder.")
        P()
        KEY = "n_span" if any("n_span" in q for q in pe) else "R"
        P("| H | L | rows by reads per span: `e_piece` across ε |")
        P("|---|---|---|")
        for H in T:
            for e in d["config"]["pe_levels"]:
                ns = sorted({q[KEY] for q in pe if q["H"] == H and q["level"] == e})
                cs = []
                for n in ns:
                    v = [[q for q in pe if q["H"] == H and q["level"] == e and q[KEY] == n
                          and q["eps"] == x][0]["e_piece"] for x in eps]
                    cs.append(f"n={n}: " + " / ".join(f"{x:.4f}" for x in v))
                P(f"| {H} | {e} | " + "  ·  ".join(cs) + " |")
        P()

        def spend(H, e, R):
            v = [[q for q in pe if q["H"] == H and q["level"] == e and q[KEY] == R
                  and q["eps"] == x][0]["e_piece"] for x in eps]
            if v[0] > band(H):
                return "already"
            for i in range(1, len(v)):
                if v[i] > band(H):
                    t = (band(H) - v[i - 1]) / (v[i] - v[i - 1])
                    return f"{eps[i - 1] + t * (eps[i] - eps[i - 1]):.3f}"
            return f">{eps[-1]}"
        P("**ε at which the band is spent** (`n = 1` is launch only, i.e. uncorrected):")
        P()
        P("| H | L | by reads per span |")
        P("|---|---|---|")
        for H in T:
            for e in d["config"]["pe_levels"]:
                ns = sorted({q[KEY] for q in pe if q["H"] == H and q["level"] == e})
                P(f"| {H} | {e} | " + "  ·  ".join(f"n={n}: {spend(H, e, n)}" for n in ns)
                  + " |")
        P()

    # ------------------------------------------------------------------ the stale hand-over
    if any("seam_read" in r for r in sw):
        P("## The stale hand-over per seam (decision 23)")
        P()
        P("A level-l arm re-decides at `K / 2^(l-1)` drilled seams — 8 / 4 / 2 / 1 at L1 / L2 / "
          "L3 / L4 — and every one is a launch from a Delta-old read. `read_dx` / `read_dv` are "
          "the read's own error at that seam; `e_seg` is the arm's realised waypoint error "
          "there. NOTE, against what this table's first draft asserted: the read error is NOT "
          "arm-independent. It is the distance the body travels in Delta steps, so it depends on "
          "where the arm has put the body and how fast it is going, and `klsmoke1` measures it "
          "differing by 1.6x BETWEEN ARMS at the same tempo (H = 2: L1 0.3780, L2 0.3053, L4 "
          "0.2352). Read it as a joint property of the delay, the tempo and the arm.")
        P()
        for nm in ("aud_kzsa_L1", "aud_kzsa_L2", "aud_kzsa_L4"):
            if not any(r["arm"] == nm for r in sw):
                continue
            P(f"### `{nm}`")
            P()
            P("| H | launches | mean read_dx | mean read_dv | e_seg by drilled seam |")
            P("|---|---|---|---|---|")
            for H in T:
                r = row(H, nm)
                if not r or "seam_read" not in r:
                    continue
                sr = r["seam_read"]
                P(f"| {H} | {r.get('n_launch', '—')} | "
                  f"{sum(x['read_dx'] for x in sr) / len(sr):.4f} | "
                  f"{sum(x['read_dv'] for x in sr) / len(sr):.3f} | "
                  + " ".join(f"{x['e_seg']:.3f}" for x in sr) + " |")
            P()

    # ------------------------------------------------------------------ the two diagnostics
    shp = d.get("shape")
    if shp:
        P("## (1) Path representability — the shape distance")
        P()
        P("Mean-over-phase distance between LAUNCH-ALIGNED position curves, in metres. `near` is "
          "each scaled slow path's distance to the NEAREST path the target tempo's own body "
          "produced in that cell. **The primary form is `d_near / band`** — metres against the "
          "competence criterion the whole node is graded on. The ratio to `d_within` (the "
          "destination library's distance to itself) is kept as a column but is NOT the "
          "headline: `d_within` collapses 7x across the ladder because the fast library is "
          "nearly one path (`accelerando`'s degenerate slot partition, P-S 1.000x at four of "
          "five tempi), so that ratio rises largely for its denominator.")
        P()
        lps = sorted({r["lp"] for r in shp}, key=lambda x: (x != "None", x))
        P("| H | low-pass | w (src samples) | " + " | ".join(
            f"L{e}: d_near / band  (x within)" for e in range(1, L + 1)) + " |")
        P("|---|---|---|" + "---|" * L)
        for H in T:
            for lp in lps:
                rs = [r for r in shp if r["H"] == H and r["lp"] == lp]
                if not rs:
                    continue
                cells = []
                for e in range(1, L + 1):
                    q = [r for r in rs if r["level"] == e]
                    if not q:
                        cells.append("—")
                        continue
                    nr = sum(r["d_near"] for r in q) / len(q)
                    wi = sum(r["d_within"] for r in q) / len(q)
                    cells.append(f"**{nr / band(H):.2f}** ({nr:.4f} m)"
                                 + (f"  [{nr / wi:.1f}x within]" if wi else ""))
                P(f"| {H} | {lp} | {rs[0]['lp_w']} | " + " | ".join(cells) + " |")
        P()
        P("Beside it, `kzs`'s own error (`aud`, `e_piece`) at the same cells, for the "
          "does-it-track reading:")
        P()
        P("| H | " + " | ".join(f"L{e}" for e in range(1, L + 1)) + " |")
        P("|---|" + "---|" * L)
        for H in T:
            P(f"| {H} | " + " | ".join(
                f"{(row(H, f'aud_kzs_L{e}') or {}).get('e_piece', float('nan')):.4f}"
                for e in range(1, L + 1)) + " |")
        P()

    lau = d.get("launch")
    if lau:
        P("## (2) Launch mismatch")
        P()
        P("Each stored unit's converted commands executed from TWO launches: `own` = the scaled "
          "path's own first sample, `act` = the actual hand-over states the target tempo "
          "delivers. Mean waypoint error over the unit's own span (at the top rung the span is "
          "the whole drilled figure, so that row is exactly `e_piece`). `dx` is the launch "
          "position gap and `dv*tau` the position error a velocity gap leaves on a body that "
          "forgets velocity in one tau. Band is in the piece table above.")
        P()
        for tag in sorted({r["arm"] for r in lau}):
            P(f"### `{tag}`")
            P()
            P("| H | L | e from own launch | e from actual | dx | dv | dv*tau |")
            P("|---|---|---|---|---|---|---|")
            for H in T:
                for e in range(1, L + 1):
                    q = [r for r in lau if r["arm"] == tag and r["H"] == H and r["level"] == e]
                    if not q:
                        continue

                    def mn(k2):
                        return sum(r[k2] for r in q) / len(q)
                    b_ = band(H)
                    o, a2 = mn("e_from_own_launch"), mn("e_from_actual")
                    P(f"| {H} | {e} | {o:.4f}{'*' if o <= b_ else ''} | "
                      f"{a2:.4f}{'*' if a2 <= b_ else ''} | {mn('dx'):.4f} | {mn('dv'):.4f} | "
                      f"{mn('dv_tau'):.4f} |")
            P()
        P("`*` = inside the band.")
        P()

    # ------------------------------------------------------------------ ledger
    P("## The ledger, drilled only (per performer)")
    P()
    P("| H | arm | n_fb | n_ground | t_piece (s) |")
    P("|---|---|---|---|---|")
    for H in T:
        for nm in ("reflex", "reflex_ec", f"aud_L{L}", f"key_L{L}", f"aud_kzs_L{L}",
                   f"key_kzs_L{L}", "aud_kzs_L1", "key_kzs_L1"):
            r = row(H, nm)
            if not r:
                continue
            ld = r["ledger_drilled"]
            P(f"| {H} | {nm} | {ld['n_fb']:.1f} | {ld['n_ground']:.1f} | {ld['t_piece']:.3f} |")
    P()
    return out


@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_kin_figures(tag: str) -> list:
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = os.path.join(DATA_DIR, "practice_rubato", tag)
    d = _json.load(open(os.path.join(src, "kin.json")))
    dst = os.path.join(src, "figures")
    os.makedirs(dst, exist_ok=True)
    cfg = d["config"]
    T = list(cfg["tempos"])
    L = int(cfg["n_levels"])
    appm = cfg["app_modes"][0]
    sw = d["sweep"]

    def row(H, arm):
        for r in sw:
            if r["H"] == H and r["arm"] == arm and r["app"] == appm:
                return r
        return None

    def ser(arm_fmt, key="e_piece"):
        return [((row(H, arm_fmt.format(H=H)) or {}).get(key)) for H in T]

    bands = [d["piece_by_tempo"][str(H)]["band"] for H in T]
    ms = [1000.0 * H * 0.024 for H in T]
    made = []
    COL = dict(rec="#111111", s0="#4c72b0", s2="#8fb3dd", kzs="#c44e52", kzo="#dd8452",
               kzsa="#7b3294", kzsf="#c994c7", kzof="#e7a3d0",
               kzoa="#dd8452", kfl="#1f9e89", kfm="#b8860b",
               kcs="#55a868", kco="#8fce9e")
    present = {r["kind"] for r in sw}
    PANEL = [(k, lab) for k, lab in
             (("rec", "rec (oracle)"), ("s0", "scl0"), ("s2", "scl2"),
              ("kzs", "kin <- slow"), ("kzsa", "kin <- slow, anti-alias"),
              ("kzsf", "kin <- slow, note LP"), ("kzo", "kin, own tempo"),
              ("kzof", "kin, own tempo, note LP"), ("kzoa", "kin, own tempo, aa"),
              ("kfl", "FITTED linear"), ("kfm", "FITTED mlp"), ("kcs", "kin ct <- slow"))
             if k in present]

    # ---- fig 1: the transfer question, one panel per level -----------------
    fig, axes = plt.subplots(1, L, figsize=(4.0 * L, 3.6), sharey=True)
    axes = [axes] if L == 1 else list(axes)
    for i, e in enumerate(range(1, L + 1)):
        ax = axes[i]
        for s, lab in PANEL:
            nm = "aud_L{e}" if s == "rec" else "aud_" + s + "_L{e}"
            y = [((row(H, nm.format(e=e)) or {}).get("e_piece")) for H in T]
            ax.plot(ms, y, "o-", color=COL[s], label=lab, lw=1.6, ms=4)
        ax.plot(ms, [((row(H, "reflex") or {}).get("e_piece")) for H in T], "s--",
                color="#888888", label="reflex (re-fit)", lw=1.2, ms=4)
        ax.plot(ms, bands, ":", color="#000000", lw=1.0, label="band")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ms); ax.set_xticklabels([f"{m:.0f}" for m in ms])
        ax.set_title(f"level {e}  ({2 ** (e - 1)} notes/unit)")
        ax.set_xlabel("note (ms)")
        ax.grid(alpha=0.25, which="both")
        if i == 0:
            ax.set_ylabel("e_piece (m)")
            ax.legend(fontsize=7)
    fig.suptitle(f"rubato/{cfg['tag']}: the kinematic unit across tempo (audit mode)")
    fig.tight_layout()
    p = os.path.join(dst, "f1_transfer.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- fig 2: conversion cost vs transfer cost ---------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    own = "kzo" if "kzo" in present else ("kzoa" if "kzoa" in present else None)
    for e in range(1, L + 1):
        if own is None:
            break
        a = [((row(H, f"aud_{own}_L{e}") or {}).get("e_piece")) for H in T]
        b = [((row(H, f"aud_L{e}") or {}).get("e_piece")) for H in T]
        c = [((row(H, f"aud_kzs_L{e}") or {}).get("e_piece")) for H in T]
        for tg, ls in (("kzsa", ":"), ("kzsf", "-.")):
            if tg in present:
                q = [((row(H, f"aud_{tg}_L{e}") or {}).get("e_piece")) for H in T]
                ax[1].plot(ms, [x / y if (x and y) else None for x, y in zip(q, a)], ls,
                           color=COL[tg], lw=1.2, alpha=0.7,
                           label=(f"{tg} L{e}" if e == 1 else None))
        ax[0].plot(ms, [x / y if (x and y) else None for x, y in zip(a, b)], "o-",
                   label=f"L{e}", lw=1.6, ms=4)
        ax[1].plot(ms, [x / y if (x and y) else None for x, y in zip(c, a)], "o-",
                   label=f"L{e}", lw=1.6, ms=4)
    for k, t in enumerate((f"conversion cost: {own} / rec", f"transfer cost: kzs / {own}")):
        ax[k].axhline(1.0, color="k", ls=":", lw=1)
        ax[k].set_xscale("log")
        if ax[k].get_lines():
            try:
                ax[k].set_yscale("log")
            except ValueError:
                pass
        ax[k].set_xticks(ms); ax[k].set_xticklabels([f"{m:.0f}" for m in ms])
        ax[k].set_xlabel("note (ms)"); ax[k].set_title(t); ax[k].grid(alpha=0.25, which="both")
        ax[k].legend(fontsize=8)
    ax[0].set_ylabel("ratio")
    fig.suptitle(f"rubato/{cfg['tag']}: what the executor costs, and what the tempo costs")
    fig.tight_layout()
    p = os.path.join(dst, "f2_decompose.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- fig 3: P-K, the residual against the world's own gate weight ------
    pk = d.get("P_K", {})
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bs = pk.get("buckets", [])
    xs = list(range(len(bs)))
    ax.bar([x - 0.2 for x in xs], [max(b["rms"], 1e-12) for b in bs], 0.4, label="rms |du|")
    ax.bar([x + 0.2 for x in xs], [max(b["max"], 1e-12) for b in bs], 0.4, label="max |du|")
    ax.axhline(pk.get("tol", 1e-3), color="r", ls="--", lw=1, label="P-K tolerance")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"w<{b['w_hi']:.0e}\nn={b['n']}" for b in bs], fontsize=7)
    ax.set_xlabel("the world's command-rotation gate weight over the step")
    ax.set_ylabel("|du| against the tape's own command")
    ax.set_title(f"rubato/{cfg['tag']}: P-K — the residual IS the rotation")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y", which="both")
    fig.tight_layout()
    p = os.path.join(dst, "f3_pk.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- fig 4: saturation --------------------------------------------------
    sat = d.get("saturation", [])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for tag2, mk in (("kzs", "o-"), ("kzo", "s--"), ("kcs", "^-"), ("kco", "v--")):
        for e in (1, L):
            y = []
            for H in T:
                ss = [r["sat"] for r in sat if r["arm"] == tag2 and r["H"] == H
                      and r["level"] == e and not r["poison"]]
                y.append(sum(ss) / len(ss) if ss else None)
            ax.plot(ms, y, mk, label=f"{tag2} L{e}", lw=1.4, ms=4)
    ax.set_xscale("log"); ax.set_xticks(ms)
    ax.set_xticklabels([f"{m:.0f}" for m in ms])
    ax.set_xlabel("note (ms)"); ax.set_ylabel("fraction of command components clipped")
    ax.set_title(f"rubato/{cfg['tag']}: the actuator ceiling")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(dst, "f4_saturation.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # ---- fig 5: both grades, deepest and shallowest rung --------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for j, (key, ttl) in enumerate((("e_piece", "e_piece"), ("e_listen", "e_listen"))):
        ax = axes[j]
        for e in (1, L):
            for s, ls in (("rec", "-"), ("s0", "--"), ("kzs", "-")):
                nm = f"aud_L{e}" if s == "rec" else f"aud_{s}_L{e}"
                ax.plot(ms, [((row(H, nm) or {}).get(key)) for H in T], ls, color=COL[s],
                        marker=("o" if e == L else "s"), lw=1.6, ms=4,
                        label=f"{s} L{e}")
        ax.plot(ms, [((row(H, "reflex") or {}).get(key)) for H in T], "s--", color="#888888",
                label="reflex", lw=1.2, ms=4)
        ax.plot(ms, bands, ":", color="k", lw=1.0, label="band")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ms)
        ax.set_xticklabels([f"{m:.0f}" for m in ms])
        ax.set_xlabel("note (ms)"); ax.set_title(ttl); ax.grid(alpha=0.25, which="both")
        if j == 0:
            ax.set_ylabel("error (m)"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"rubato/{cfg['tag']}: both grades, shallowest and deepest rung")
    fig.tight_layout()
    p = os.path.join(dst, "f5_grades.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    volume.commit()
    return made


@app.local_entrypoint()
def rubato_kin_figs(tag: str = "k1"):
    for p in make_kin_figures.remote(tag):
        print(p)


def figures(tag):
    subprocess.run(["modal", "run",
                    os.path.join("mjc", "practice", "rubato", "analyze_kin.py")
                    + "::rubato_kin_figs", "--tag", tag],
                   check=True, cwd=os.path.dirname(os.path.dirname(
                       os.path.dirname(HERE))))
    dst = os.path.join(HERE, "figures", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL,
                    f"/practice_rubato/{tag}/figures", dst, "--force"], check=False)
    # `volume get` on a directory nests it one level; flatten so `figures/<tag>/*.png` is flat
    nested = os.path.join(dst, "figures")
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            os.replace(os.path.join(nested, f), os.path.join(dst, f))
        os.rmdir(nested)
    print(f"[figures] -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="k1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    path = fetch(a.tag) if a.fetch else os.path.join(HERE, "results", a.tag, "kin.json")
    d = json.load(open(path))
    out = []
    report(d, out)
    rp = os.path.join(HERE, "results", a.tag, "report.md")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\n[report] -> {rp}")
    if a.figures:
        figures(a.tag)


if __name__ == "__main__":
    main()
