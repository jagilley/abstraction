"""Reduce a cappella's Phase A gate run to a text report.

Pure python (json + math only): the local client in these sessions has neither numpy nor
matplotlib, so nothing here may import them (`offbook/FILES.md` Gotcha).

    python3 mjc/practice/acappella/analyze_gates.py --tag a0 --fetch
"""

import argparse
import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "mujoco-control-data"


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL, f"/practice_acappella/{tag}/gates.json",
                    os.path.join(dst, "gates.json"), "--force"], check=True)
    return os.path.join(dst, "gates.json")


def f(x, n=4):
    if x is None:
        return "  -  "
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return " nan "
    return f"{x:.{n}f}"


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    P(f"=== a cappella Phase A — tag {cfg['tag']}, seed {cfg['seed']} "
      f"({'COMPLETE' if d.get('complete') else 'INCOMPLETE'}, {d.get('wall_s', 0):.0f}s wall) ===")
    P(f"n_proc={cfg['n_proc']} n_eval={cfg['n_eval']} n_rt={cfg['n_rt']} batch={cfg['batch']} "
      f"n_warm={cfg['n_warm']} n_cand={cfg['n_cand']} n_score={cfg['n_score']} "
      f"n_slot={cfg['n_slot']} cem_iters={cfg['cem_iters']}")

    P("\n--- A-F  fork fidelity ---")
    P(f"A-F1 donor constants: {d['A_F1']['checked']} checked, "
      f"{len(d['A_F1']['mismatches'])} mismatched, pass={d['A_F1']['pass']}")
    for nm in ("noisefree", "noisy"):
        g = d["A_F2"][nm]
        mx = max(v for v in g.values() if isinstance(v, float))
        P(f"A-F2 {nm:10s} max|delta| = {mx:.3e}  fb={g['fb_equal']}  t={g['t_equal']}  "
          f"pass={g['pass']}")

    rc = d["reflex_cal"]["chosen"]
    r = d["reflex"]
    P("\n--- the reflex law (gains calibrated on the CLEAN world, then frozen) ---")
    P(f"kp={rc['kp']} kd={rc['kd']}   clean e_piece={f(rc['e_piece'])}  "
      f"clean seg={[f(x) for x in rc['e_seg']]}")
    P(f"ROTATED world: e_piece={f(r['e_piece'])} seg={[f(x) for x in r['e_seg']]}  "
      f"fb={r['n_fb']:.0f}  ground={r['n_ground']:.0f}  priced={f(r['t_piece'], 1)}s")

    P("\n--- A-B  the budget ladder (performance tempo, arm-neutral, no library exists) ---")
    P(f"{'G':>6s} {'e_piece':>9s} {'per-segment':>36s} {'ground/trav':>12s} {'priced s':>10s} "
      f"{'wall s':>8s}")
    for row in d["A_B"]["ladder"]:
        P(f"{row['G']:6d} {f(row['e_piece']):>9s} "
          f"{str([f(x) for x in row['e_seg']]):>36s} {row['n_ground']:12.0f} "
          f"{row['t_piece']:10.1f} {row.get('wall_s', 0):8.1f}")
    P(f"RULE (pre-fixed): {d['A_B']['rule']} -> G* = {d['A_B']['G_star']} "
      f"(ladder best {f(d['A_B']['e_best'])})")
    em = [r for r in d["A_B"]["ladder"] if r.get("elite_mean_err")]
    if em:
        P("instrument — elite MEAN vs best SAMPLED sequence, realised (etude finding 2 on a real "
          "rollout; per seam):")
        for row in em:
            P(f"  G={row['G']:6d} mean={[f(x) for x in row['elite_mean_err']]}  "
              f"best={[f(x) for x in row['best_sampled_err']]}")

    if d["A_B"].get("tempo"):
        P("\n--- A-B* the tempo dimension (instrument; the budget rule does not read it) ---")
        P(d["A_B"]["tempo_note"])
        P(f"{'R':>4s} {'G':>7s} {'e_piece':>9s} {'ground/trav':>12s} {'fb':>6s} {'priced s':>10s}")
        for row in d["A_B"]["tempo"]:
            P(f"{row['R']:4d} {row['G']:7d} {f(row['e_piece']):>9s} {row['n_ground']:12.0f} "
              f"{row['n_fb']:6.0f} {row['t_piece']:10.1f}")

    a = d["A_I"]
    P("\n--- A-I  gate 2: does the incumbent reach? ---")
    P(f"search @ G*={a['G_star']}: e_piece={f(a['e_search_star'])} at {a['t_search_star']:.0f}s "
      f"priced;  étude `never` band {a['never_band']}  ->  reaches={a['reaches_band']}")
    P(f"reflex law:            e_piece={f(a['e_reflex'])} at {a['t_reflex']:.1f}s priced")
    b = a["best_incumbent_cell"]
    P(f"best cell of the whole incumbent family: e_piece={f(b['e_piece'])} at {b['t_piece']:.0f}s "
      f"({b.get('kind')}, R={b.get('R', 'H')}, G={b['G']})")
    P(f"cells beating the reflex on ERROR: {a['n_cells_beating_reflex_on_error']}; "
      f"cells DOMINATING it (error AND priced time): {a['n_cells_dominating_reflex']}")
    P(f"PRE-FIXED HALT FIRED: {a['halt']}   (gate pass = {a['pass']})")

    if "library" in d:
        P("\n--- the library (nested: candidates for seam k harvested with seams < k committed) ---")
        P(f"sizes {d['library']['sizes']}   build cost "
          f"{d['library']['build_ledger']['raw']['n_ground']:.0f} groundings")
        P(f"{'cell':>10s} {'pool':>6s} {'cand':>5s} {'chosen':>8s} {'pool med':>9s} "
          f"{'curse cf':>9s} {'s0 disp':>8s} {'s0 sprd':>8s} {'s0 speed':>9s}")
        for rec in d["library"]["build"]:
            P(f"{'(1,%d)' % rec['seam']:>10s} {rec['n_pool']:6d} {rec['n_cand']:5d} "
              f"{f(rec['chosen_score']):>8s} {f(rec['score_med']):>9s} "
              f"{f(rec['score_of_best_own']):>9s} {f(rec['s0_disp']):>8s} "
              f"{f(rec['s0_spread']):>8s} {f(rec['s0_speed'], 3):>9s}")
        for rec in d["library"]["chains"]:
            c = rec["control_reflex_sourced"]
            P(f"{'(%d,%d)' % (rec['ns'], rec['seam']):>10s} {rec['n_pool']:6d} {rec['n_cand']:5d} "
              f"{f(rec['chosen_score']):>8s} {f(rec['score_med']):>9s}   "
              f"| reflex-sourced control: chosen={f(c['chosen_score'])} med={f(c['score_med'])}")

    if "A_C" in d:
        P("\n--- A-C  audition calibration on the plant ---")
        P(f"A-C0 seam-time audition exactness: max|chosen - realised| = "
          f"{d['A_C']['exactness_max_abs']:.3e}  exact={d['A_C']['exact']}  "
          f"(the rollout IS the plant, so this op cannot be optimistic)")
        P(f"A-C1 library-construction audition: {'level':>6s} {'cell':>8s} {'chosen':>8s} "
          f"{'realised':>9s} {'gap':>6s}")
        for c in d["A_C"]["calibration"]:
            P(f"{'':36s}{c['level']:>6s} {'(%d,%d)' % (c['ns'], c['seam']):>8s} "
              f"{f(c['chosen']):>8s} {f(c['realised']):>9s} {c['gap']:6.2f}")

    if "A_R" in d:
        P(f"\n--- A-R  the rent (G* = {d['A_R']['G_star']}; top-k order is construction-score "
          f"order, i.e. OPTIMISTIC — no pi in Phase A) ---")
        P(f"{'arm':>16s} {'k':>3s} {'e_piece':>9s} {'ground/trav':>12s} {'fb':>6s} "
          f"{'aud/dec':>8s} {'priced s':>10s} {'prim':>6s} {'chain':>6s}")
        for row in d["A_R"]["ladder"]:
            nm = row["mode"] + ("" if not row["levels"] else "+" + "+".join(row["levels"]))
            P(f"{nm:>16s} {row['k']:3d} {f(row['e_piece']):>9s} {row['n_ground']:12.1f} "
              f"{row['n_fb']:6.1f} {row['n_aud']:8.1f} {row['t_piece']:10.1f} "
              f"{row['frac_prim']:6.2f} {row['frac_chain']:6.2f}")
        P("pricing sensitivity — priced seconds if a grounding costs only its re-grounding event "
          "(RHM's convention: n_ground*d_fb, no trial execution time):")
        for row in d["A_R"]["ladder"]:
            alt = row["steps"] * 0.024 + (row["n_fb"] + row["n_ground"]) * 0.10
            P(f"{row['mode']:>16s} k={row['k']:<3d} full={row['t_piece']:9.1f}s  "
              f"fb-only={alt:9.1f}s")

    if "A_S" in d:
        P("\n--- A-S  seam information ---")
        P(d["A_S"]["note"])
        P(f"{'seam':>5s} {'spread':>8s} {'repeat nz':>10s} {'ratio':>7s} {'pos ratio':>10s} "
          f"{'perf spread':>12s} {'speed':>7s}")
        for s in d["A_S"]["seams"]:
            P(f"{s['seam']:5d} {f(s['spread']):>8s} {f(s['repeat_noise']):>10s} "
              f"{f(s['ratio'], 2):>7s} {f(s['pos_ratio'], 2):>10s} "
              f"{f(s['perf_spread']):>12s} {f(s['speed'], 3):>7s}"
              + ("   [undefined]" if s.get("structurally_undefined") else ""))
        P(f"{'seam':>5s} {'best fixed':>11s} {'per-state':>10s} {'oracle gain':>12s} "
          f"{'distinct argmins':>17s}")
        for s in d["A_S"].get("oracle", []):
            P(f"{s['seam']:5d} {f(s['best_fixed']):>11s} {f(s['per_state']):>10s} "
              f"{s['gain']:11.2f}x {('%d/%d' % (s['n_distinct_argmin'], s['n_slot'])):>17s}")

    if "A_D" in d:
        P("\n--- A-D  cycle cost (what Phase B is sized from) ---")
        P(json.dumps({k: round(v, 1) for k, v in d["A_D"]["wall_by_phase"].items() if v}))
        P("search traversal wall-clock by budget: "
          + json.dumps({k: round(v, 1) for k, v in
                        d["A_D"]["search_traversal_wall_by_G"].items()}))

    if d.get("halted_at"):
        P(f"\n*** HALTED AT {d['halted_at']} — Phase B is NOT licensed by this run. ***")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    p = fetch(a.tag) if a.fetch else os.path.join(HERE, "results", a.tag, "gates.json")
    d = json.load(open(p))
    lines = report(d, [])
    rp = os.path.join(os.path.dirname(p), "gate_report.txt")
    with open(rp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[report] {rp}")


if __name__ == "__main__":
    main()
