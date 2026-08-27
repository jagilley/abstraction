"""Reduce a presto Phase A (`tempo.py`) run into the design record.

Prints, in order: the fork-fidelity assertion, the tempo ladder (one row per (damping, R) cell with
the do-nothing floor, both playability anchors, the incumbent across delay, and the open-loop
feasibility precondition at both spans), the composition horizon, the seam-information table, the
planner ladder, and the content calibration.

The DESIGN QUESTION this report exists to answer, stated so the table can be read against it:
is there a cell where (i) the piece is playable open-loop by measured content at Delta = 0 -- the
`ballistic/` 4b precondition -- and (ii) the incumbent reactive controller is already broken at a
human-realistic delay? Neither half is assumed; both are columns.

    python3 mjc/practice/accompanist/presto/analyze_tempo.py --tag t0 --fetch
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(tag):
    import subprocess

    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    src = f"/practice_presto/{tag}/tempo/results.json"
    r = subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data", src,
                        os.path.join(dst, "tempo.json")], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"fetch failed for {src}")
    return os.path.join(dst, "tempo.json")


def f(x, n=4):
    return "  n/a " if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t0")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()

    path = os.path.join(HERE, "results", a.tag, "tempo.json")
    if a.fetch or not os.path.exists(path):
        path = fetch(a.tag)
    R = json.load(open(path))
    cells = R.get("cells", {})
    delays = R.get("ladder", {}).get("delays", [])
    dshort = R.get("config", {}).get("delays_short", [])
    lines = []

    def W(*s):
        t = " ".join(str(x) for x in s)
        print(t)
        lines.append(t)

    W(f"\n=== presto Phase A ({a.tag}) — complete={R.get('complete')} "
      f"wall={R.get('wall_s', 0) / 60:.0f} min ===")

    g = R.get("gF", {})
    if g:
        mx = max(v for k, v in g.items() if isinstance(v, float))
        W(f"\n[G-F] fork fidelity vs offbook on the DONOR piece+plant: max|delta| = {mx:.3e} "
          f"over 7 arrays | fb_equal={g.get('fb_equal')} t_equal={g.get('t_equal')} "
          f"-> PASS={g.get('pass')}")
    g2 = R.get("gF2", {})
    if g2:
        mx2 = max(v for k, v in g2.items() if isinstance(v, float))
        W(f"[G-F2] `obs_predict` (efference copy) is a strict no-op at Delta = 0: "
          f"max|delta| = {mx2:.3e} -> PASS={g2.get('pass')}")

    # ---------------------------------------------------------------- the tempo ladder
    W("\n=== G-T — the tempo ladder (mean piece error, 24 held-out geometries, sigma_perf) ===")
    W("`hold` = zero commands after the approach (the do-nothing floor).  Playability anchors: "
      "`ref_stale` = ballistic-per-segment under the PRE-PRACTICE FM undelayed (offbook's, carried "
      "unchanged); `half_leg` = half the mean drilled leg (task-anchored, so the anchor cannot "
      "collapse below 'the figure is still recognisable').")
    W("Two incumbents per delay: `rct` = offbook's NAIVE delayed observation (act on where you "
      "were), `prd` = the same controller with EFFERENCE COPY through the delay (act on where "
      "your own forward model says you now are). `prd` is the honest strong baseline; `rct` is "
      "the continuity read against offbook Round 4.")
    hdr = (f"{'damp':>5} {'R':>5} {'m/s':>5} {'hold':>7} {'refst':>7} {'hleg':>7} | "
           + " ".join(f"{'rct/prd d' + str(d):>15}" for d in delays) + " | "
           + " ".join(f"{'seg/p d' + str(d):>15}" for d in dshort) + " | "
           + " ".join(f"{'chn d' + str(d):>7}" for d in dshort) + f" | {'v_max':>6}")
    W(hdr); W("-" * len(hdr))
    for k, c in cells.items():
        row = (f"{c['damping']:>5} {c['R']:>5} {c['nominal_speed']:>5.2f} "
               f"{c['hold']['e']:>7.4f} {c['ref_stale']['e']:>7.4f} {c['half_leg']:>7.4f} | "
               + " ".join(f"{c[f'react_d{d}']['e']:>7.4f}/"
                          f"{c.get(f'reactp_d{d}', {}).get('e', float('nan')):<7.4f}"
                          for d in delays) + " | "
               + " ".join(f"{c[f'live_seg_d{d}']['e']:>7.4f}/"
                          f"{c.get(f'live_segp_d{d}', {}).get('e', float('nan')):<7.4f}"
                          for d in dshort) + " | "
               + " ".join(f"{c[f'live_chain_d{d}']['e']:>7.4f}" for d in dshort) + " | "
               + f"{c['react_d0']['v_max']:>6.2f}")
        W(row)

    if cells:
        c0 = cells[list(cells)[0]]
        W("\nfeedback events per traversal (the currency a delay taxes): "
          + ", ".join(f"{n}={c0[n + '_d0']['fb']:.0f}"
                      for n in ("react", "live_seg", "live_chain")))

    # ---------------------------------------------------------------- horizon
    W("\n=== G-H — the composition horizon, re-measured on this plant "
      "(FM rolled open-loop along EXECUTED commands; first step median tip divergence crosses) ===")
    W(f"{'cell':>14} {'h@.05 stale':>12} {'h@.05 prac':>11} {'h@.02 stale':>12} "
      f"{'h@.02 prac':>11} {'1step task':>11} {'1step pool':>11}   segment={{H_seg}}  "
      f"phrase={{H_phrase}}")
    for k, c in cells.items():
        h = c.get("gH")
        if not h:
            continue
        W(f"{k:>14} {h['stale']['h_005']:>12} {h['practised']['h_005']:>11} "
          f"{h['stale']['h_002']:>12} {h['practised']['h_002']:>11} "
          f"{h['one_step_task']:>11.4f} {h['one_step_pool']:>11.4f}   "
          f"segment={c['h_phrase'] // c['n_seg']}  phrase={c['h_phrase']}")

    # ---------------------------------------------------------------- seams
    W("\n=== G-S — seam information (posture spread / matched repeat-noise; >1 means the seam "
      "carries state a key could read) ===")
    for k, c in cells.items():
        s = c.get("gS")
        if not s:
            continue
        W(f"{k:>14}  " + "  ".join(
            f"seam{j}: {v['ratio']:.2f}x (tip {v['tip_spread'] / max(v['tip_noise'], 1e-9):.2f}x, "
            f"|v|={v['speed']:.2f})" for j, v in s.items()))

    # ---------------------------------------------------------------- planner
    W("\n=== G-P — the planner ladder (arm_substrate P3: if error is still falling at the top, "
      "the arm is planner-starved and every span comparison is noise) ===")
    for k, c in cells.items():
        gp = c.get("gP")
        if not gp:
            continue
        W(f"{k:>14}  seg  " + "  ".join(f"{ks}:{v:.4f}" for ks, v in gp["seg"].items())
          + "   | chain " + "  ".join(f"{ks}:{v:.4f}" for ks, v in gp["chain"].items()))
        for lk, v in gp["react_look"].items():
            W(f"{'':>14}  reactive look={lk:>3}  "
              + "  ".join(f"{d}:{e:.4f}" for d, e in v.items()))

    # ---------------------------------------------------------------- content
    W("\n=== G-L — content calibration (legato recipe: tapes harvested at sigma_perf, "
      "cross-state PLANT audition on a held-out score set, keyed via select_library / "
      "single tape via select_fixed) ===")
    for k, c in cells.items():
        gl = c.get("gL")
        if not gl:
            continue
        W(f"\n  {k}  (anchors: ref_stale {c['ref_stale']['e']:.4f}, "
          f"half_leg {c['half_leg']:.4f})")
        for d in dshort:
            W(f"    delay {d:>2} ({d * 20:>3} ms):  "
              + "  ".join(f"{nm}={gl[f'{nm}_d{d}']['e']:.4f}(fb{gl[f'{nm}_d{d}']['fb']:.0f})"
                          for nm in ("seg_tape", "seg_tapep", "seg_fixed", "chain",
                                     "chain_fixed")
                          if f"{nm}_d{d}" in gl))
        au = [(nm, gl[nm]) for nm in gl if nm.startswith("seg") and "_d" not in nm]
        au += [("chain", gl["chain"])] if "chain" in gl and "n_cand" in gl.get("chain", {}) else []
        W("    audition: " + "  ".join(
            f"{nm}[best_fixed {v['best_fixed']:.3f} / oracle {v['per_state_oracle']:.3f} "
            f"= {v['oracle_gain']:.2f}x, {v['n_distinct']}/{len(v['cell_sizes'])} distinct]"
            for nm, v in au))

    # ---------------------------------------------------------------- the design question
    W("\n=== the design question, as a table ===")
    W("A cell is a candidate when measured content clears an anchor at Delta = 0 (the "
      "`ballistic/` 4b feasibility precondition) AND the incumbent is already broken at a "
      "human-realistic delay. Both columns, no verdict.")
    W(f"{'cell':>14} {'best stored @d0':>16} {'anchor(max)':>12} {'stored playable':>16} "
      f"{'rct d0':>8} {'rct d4':>8} {'prd d4':>8} {'naive brk':>10} {'pred brk':>9}")
    for k, c in cells.items():
        gl = c.get("gL")
        if not gl:
            continue
        stored = [gl[f"{nm}_d0"]["e"]
                  for nm in ("seg_tape", "seg_fixed", "chain", "chain_fixed")
                  if f"{nm}_d0" in gl]
        best = min(stored) if stored else None
        anch = max(c["ref_stale"]["e"], c["half_leg"])
        r0 = c["react_d0"]["e"]
        r4 = c.get("react_d4", {}).get("e")
        r4p = c.get("reactp_d4", {}).get("e")
        W(f"{k:>14} {f(best):>16} {anch:>12.4f} "
          f"{str(best is not None and best <= anch):>16} {r0:>8.4f} {f(r4):>8} {f(r4p):>8} "
          f"{str(r4 is not None and r4 > anch):>10} "
          f"{str(r4p is not None and r4p > anch):>9}")

    dst = os.path.join(HERE, "results", a.tag)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "tempo_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {dst}/tempo_report.txt")


if __name__ == "__main__":
    main()
