"""Reduce a cappella's B1 Δ-ladder run to a text report.

Pure python (json + math only): the local client has neither numpy nor matplotlib
(`offbook/FILES.md` Gotcha).

    python3 mjc/practice/acappella/analyze_delay.py --tag b1 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "mujoco-control-data"
ARMS = ("reflex", "key_seg", "lib_seg", "lib_all", "key_chain", "aud_chain")


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL, f"/practice_acappella/{tag}/delay.json",
                    os.path.join(dst, "delay.json"), "--force"], check=True)
    return os.path.join(dst, "delay.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    P(f"=== a cappella B1 — the Δ ladder, tag {cfg['tag']}, seed {cfg['seed']} "
      f"({'COMPLETE' if d.get('complete') else 'INCOMPLETE'}, {d.get('wall_s', 0):.0f}s wall) ===")
    P(f"Δ ladder {cfg['d_ladder']} control steps = "
      f"{[round(x * 0.024 * 1000) for x in cfg['d_ladder']]} ms;  n_eval={cfg['n_eval']}  "
      f"n_slot={cfg['n_slot']}  no forward model anywhere")

    P("\n--- gates ---")
    P(f"B-F0 donor constants: {d['B_F0']['checked']} checked, "
      f"{len(d['B_F0']['mismatches'])} mismatched, pass={d['B_F0']['pass']}")
    b = d["B_F1"]
    mx = max(v for k, v in b.items() if k.endswith("_max_abs"))
    P(f"B-F1 cross-tag exact control vs a0 (library build + the Δ=0 rows): max|Δ| = {mx:.3e}  "
      f"applicable={b.get('applicable')}  pass={b['pass']}")
    for k, v in sorted(b.items()):
        if k.endswith("_max_abs"):
            P(f"       {k[:-8]:22s} {v:.3e}")

    a = d["anchors"]
    P("\n--- anchors (all computed blind to every arm's number) ---")
    P(f"ref_play = {a['ref_play']}  (étude `never` @ performance tempo, 3-seed mean — the "
      f"pre-fixed guard)")
    P(f"reported beside: half_leg = {a['half_leg']} (vacuous on this piece) · "
      f"do-nothing floor = {a['do_nothing']:.4f} · half of it = {a['half_do_nothing']:.4f}")

    P("\n--- the reflex law, re-fit at every Δ on the CLEAN world (every advantage to the "
      "incumbent) ---")
    P(f"{'Δ':>3s} {'ms':>5s} {'kp':>6s} {'kd':>6s} {'clean e':>9s} {'kp edge':>8s} {'kd edge':>8s}")
    for k in sorted(d["reflex_cal"], key=lambda x: int(x)):
        r = d["reflex_cal"][k]
        P(f"{int(k):3d} {round(int(k) * 0.024 * 1000):5d} {r['kp']:6.1f} {r['kd']:6.2f} "
          f"{r['e_piece']:9.4f} {str(r['at_kp_edge']):>8s} {str(r['at_kd_edge']):>8s}")

    by = {}
    for r in d["sweep"]:
        if not r.get("record_rung"):
            by.setdefault(r["delta"], {})[r["arm"]] = r
    P("\n--- the sweep: piece error by Δ × arm (fb/traversal in brackets) ---")
    P(f"{'Δ':>3s} {'ms':>5s} " + " ".join(f"{a:>16s}" for a in ARMS))
    for D in sorted(by):
        P(f"{D:3d} {round(D * 0.024 * 1000):5d} "
          + " ".join(f"{by[D][a]['e_piece']:9.4f}[{by[D][a]['n_fb']:5.1f}]" for a in ARMS))
    P("\ngroundings/traversal and priced seconds (Δ=0 row; identical across Δ for these arms):")
    for a in ARMS:
        r = by[0][a]
        P(f"  {a:10s} ground={r['n_ground']:7.1f}  fb={r['n_fb']:6.1f}  "
          f"priced={r['t_piece']:8.1f}s")

    rr = [r for r in d["sweep"] if r.get("record_rung")]
    if rr:
        P("\n--- the record rung: the priced search at Δ=0 (A-I retired it as the headline) ---")
        for r in rr:
            P(f"  {r['arm']:12s} e_piece={r['e_piece']:.4f}  ground={r['n_ground']:.0f}  "
              f"priced={r['t_piece']:.0f}s")

    v = d["verdict"]
    P("\n--- THE GATE: the pre-fixed niche criterion ---")
    P(d["criterion"]["niche_rule"])
    P(f"{'Δ':>3s} {'ms':>5s} {'best committed':>15s} {'fb':>6s} {'e_best':>8s} {'e_reflex':>9s} "
      f"{'margin':>9s} {'guard m':>9s} {'NICHE':>6s}")
    for r in v["rows"]:
        P(f"{r['delta']:3d} {r['ms']:5d} {r['best_committed_arm']:>15s} "
          f"{r['best_committed_fb']:6.1f} {r['best_committed_e']:8.4f} {r['e_reflex']:9.4f} "
          f"{r['niche_margin']:9.4f} {r['niche_guard_margin_m']:9.4f} {str(r['niche']):>6s}")
    P("\n--- the donor's 3-term depth ordering (chain ≤ seg ≤ reflex), with margins ---")
    P(f"{'Δ':>3s} {'ms':>5s} {'e_chain':>9s} {'e_seg':>9s} {'e_reflex':>9s} {'ch-seg':>9s} "
      f"{'seg-rfx':>9s} {'guard m':>9s} {'guard %':>8s} {'ord':>5s} {'play':>5s} {'PASS':>5s}")
    for r in v["rows"]:
        P(f"{r['delta']:3d} {r['ms']:5d} {r['e_chain']:9.4f} {r['e_seg']:9.4f} "
          f"{r['e_reflex']:9.4f} {r['margin_chain_seg']:9.4f} {r['margin_seg_reflex']:9.4f} "
          f"{r['guard_margin_m']:9.4f} {100 * r['guard_margin_frac']:7.1f}% "
          f"{str(r['ordered']):>5s} {str(r['playable']):>5s} {str(r['passes_ordering']):>5s}")

    P("\n--- degradation across the ladder (offbook finding 7: ordered by fb consumption) ---")
    P(f"{'arm':>10s} {'fb':>7s} {'e(0)':>9s} {'e(max Δ)':>10s} {'factor':>9s}")
    for nm, x in sorted(v["degradation"].items(), key=lambda kv: -kv[1]["fb"]):
        P(f"{nm:>10s} {x['fb']:7.1f} {x['e0']:9.4f} {x['e_max']:10.4f} {x['factor']:8.2f}x")

    opt = [(r["delta"], r["arm"], r.get("aud_optimism"), r.get("aud_max_abs"))
           for r in d["sweep"] if r.get("aud_optimism") is not None]
    if opt:
        P("\n--- the audition under delay (A-C0 said it is EXACT at Δ=0; this is what Δ costs it) ---")
        P(f"{'Δ':>3s} {'arm':>10s} {'realised/chosen':>16s} {'max|Δ|':>10s}")
        for D, arm, o, m in opt:
            P(f"{D:3d} {arm:>10s} {o:16.3f} {m:10.4f}")

    P("\n" + ("*** Δ*_niche = %s (%s ms) — THE GATE PASSES ***"
              % (v["delta_star_niche"], round((v["delta_star_niche"] or 0) * 0.024 * 1000))
              if v["delta_star_niche"] is not None else
              "*** Δ*_niche = None — THE GATE FAILS; the ladder is not extended (legato F2) ***"))
    P("3-term ordering Δ* = %s" % v["delta_star_ordering"]
      + ("  -> a CHAIN-span niche" if v["delta_star_ordering"] is not None
         else "  -> never; a SEGMENT-span niche if any (offbook finding 7d)"))
    if v["converse_halt_fired"]:
        P("*** CONVERSE HALT FIRED: the reflex law is best at every Δ. No niche; B2 not licensed. ***")
    P("\nSTRUCTURAL NOTE carried with every chain number:")
    P("  " + d["criterion"]["seam0_structural_note"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    p = fetch(a.tag) if a.fetch else os.path.join(HERE, "results", a.tag, "delay.json")
    d = json.load(open(p))
    lines = report(d, [])
    rp = os.path.join(os.path.dirname(p), "delay_report.txt")
    with open(rp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[report] {rp}")


if __name__ == "__main__":
    main()
