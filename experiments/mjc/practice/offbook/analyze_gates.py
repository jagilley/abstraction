"""Fetch the Phase-A gate results off the volume and print the gate report.

READING RULE, inherited from the parent nodes: **ranks and shapes are the currency, not absolute
values.** Single seed. A gate has to establish that an axis EXISTS and sits above its own noise
floor; a calibration has to establish which setting a MEASUREMENT picks. Neither is a claim about
the third decimal place. And per legato F2, a shared knob is chosen by what keeps the measurement
axis alive, not by what makes an arm look good.

The three gates that can legitimately fail, and what a failure means:

  G-K / G-B   the rent does not bind, or a chunk never beats the live plan at any budget. That is a
              fact about this substrate -- the arm's planner is cheap relative to its library in a
              way RHM's beam was not -- and the answer is to report it, run the main node with the
              budget conversion off and the rent as a cost claim, or not to run Phase B at all.
  G-S         the grown pools carry no more seam information than the metering noise. Then there is
              nothing for a key, an audition or a policy to condition on, and `fingering/` G1 does
              not port.
  G-C         seam-time audition is badly calibrated at the chain level. That is `span/` F2 arriving
              as a constraint on the OP rather than on the content, and `aud_horizon` is read off it.

Usage (from experiments/):
    python3 mjc/practice/offbook/analyze_gates.py --tag g0 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_offbook/{tag}", dst], check=False,
                   env={**os.environ,
                        "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "chromatic")})


def load(tag):
    root = os.path.join(HERE, "results", tag)
    for p in (os.path.join(root, tag, "gates", "results.json"),
              os.path.join(root, "gates", "results.json"),
              os.path.join(root, "gates.json")):
        if os.path.isfile(p):
            return json.load(open(p))
    raise SystemExit(f"no gate results under {root}; run with --fetch")


def f(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g0")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    d = load(a.tag)
    cfg = d.get("config", {})
    L = []

    def W(s=""):
        L.append(s)
        print(s)

    W(f"\n=== offbook Phase A gates — tag {a.tag} (seed {cfg.get('seed')}) ===")
    W(f"warm cycles {cfg.get('n_warm')} · library K/cell {cfg.get('lib_k')} · "
      f"n_slot {cfg.get('n_slot')} · k_prop {cfg.get('k_prop')} · "
      f"aud_horizon {cfg.get('aud_horizon')} · complete={d.get('complete')}")

    # ---------------------------------------------------------------- G-F / G-R / G-P
    gF, gR, gP = d.get("gF", {}), d.get("gR", {}), d.get("gP", {})
    W("\n-- G-F fork fidelity (offbook.World ≡ legato.World on the donor path) --")
    W("  " + "  ".join(f"{k}={f(v, 3)}" for k, v in gF.items() if isinstance(v, float)))
    W(f"  fb_equal={f(gF.get('fb_equal'))}  t_equal={f(gF.get('t_equal'))}  "
      f"PASS={f(gF.get('pass'))}")
    W("\n-- G-R rng discipline (minting a head leaves the shared stream untouched) --")
    W(f"  torch_state_unchanged={f(gR.get('torch_state_unchanged'))}  "
      f"prop_out_zero_init={f(gR.get('prop_out_zero') == 0.0)}  PASS={f(gR.get('pass'))}")
    W("\n-- G-P π fidelity: top-k at k=N ≡ full seam audition, bit-for-bit --")
    W("  " + "  ".join(f"{k}={f(v, 3)}" for k, v in gP.items()
                       if isinstance(v, float) and k != "n_cand_mean"))
    W(f"  aud_equal={f(gP.get('aud_equal'))}  PASS={f(gP.get('pass'))}")
    W(f"\n  cost ladder at one seam (per performer-decision):")
    W(f"    {'k':>6} {'materialisations':>18} {'aud rollout-steps':>19} {'cand/perf':>10}")
    W(f"    {'ALL':>6} {gP.get('audit_all_aud', 0):>18} "
      f"{gP.get('audit_all_aud_delib', 0):>19} {f(gP.get('n_cand_mean'), 1):>10}")
    for k in sorted(int(x[1:]) for x in gP if x.startswith("k") and x[1:].isdigit()):
        c = gP[f"k{k}"]
        W(f"    {k:>6} {c['aud']:>18} {c.get('aud_delib', c['delib']):>19} "
          f"{f(c['cand_mean'], 1):>10}")

    # ---------------------------------------------------------------- G-S
    gS = d.get("gS", {})
    W("\n-- G-S pool spread on the GROWN pools (fingering G1's instrument) --")
    W(f"  {'seam':>6} {'posture sd':>11} {'noise':>9} {'ratio':>7} "
      f"{'tip sd':>9} {'noise':>9} {'ratio':>7} {'speed':>7}")
    for k in sorted(gS):
        v = gS[k]
        W(f"  {k:>6} {f(v['posture_spread']):>11} {f(v['posture_noise']):>9} "
          f"{f(v['posture_over_noise'], 2):>7} {f(v['tip_spread']):>9} "
          f"{f(v['tip_noise']):>9} {f(v['tip_over_noise'], 2):>7} {f(v['speed'], 3):>7}")

    # ---------------------------------------------------------------- G-K
    gK = d.get("gK", {})
    W("\n-- G-K(a) the planner ladder: realised piece-segment error vs CEM width --")
    rungs = gK.get("rungs", {})
    W("  " + "  ".join(f"{k}:{f(v)}" for k, v in sorted(rungs.items(), key=lambda x: int(x[0]))))
    W("     (monotone falling = the planner is the axis the budget conversion can spend on;")
    W("      flat = the rent has nothing to buy; RISING = search is exploiting model error,")
    W("      `arm_substrate` P3's sign-inversion regime, and the ladder is unusable as a currency)")
    W("\n-- G-K(b) the library curve: realised error of best-of-K seam-time audition --")
    for cell, row in sorted(gK.get("lib_curve", {}).items()):
        W(f"  ns:seam {cell:>5}  " + "  ".join(
            f"{k}:{f(v)}" for k, v in sorted(row.items(), key=lambda x: int(x[0]))))
    W(f"  library sizes: {json.dumps(gK.get('K_full', {}))}")
    W("     (CAVEAT, legato F5: this gate's chain pool is harvested from REACTIVE warm-up, which is")
    W("      exactly the configuration F1 measured keying at 1.03× and F5 corrected to 1.34× once the")
    W("      pool came from segment-committed traversals. Read the chain rows as a LOWER bound.)")
    W("\n-- G-K(c) THE RENT TABLE: which CEM rung each policy can still afford --")
    for D, row in sorted(gK.get("rent", {}).items(), key=lambda x: int(x[0])):
        parts = [f"audit_all spend={f(row.get('spend_audit_all'), 0)} "
                 f"rung={row.get('rung_audit_all')} e={f(row.get('e_audit_all'))}"]
        for kk in sorted(int(x[1:]) for x in row if x.startswith("k") and x[1:].isdigit()):
            c = row[f"k{kk}"]
            parts.append(f"k={kk} spend={f(c['spend'], 0)} rung={c['rung']} "
                         f"e={f(c['e'])} RENT={f(c['rent'])}")
        W(f"  D={D}\n    " + "\n    ".join(parts))

    # ---------------------------------------------------------------- G-B
    gB = d.get("gB", {})
    W("\n-- G-B does a chunk ever pay? piece-level, restricted action sets, per budget --")
    W(f"  {'D':>8} {'prim_only':>22} {'seg_only':>22} {'chain_only':>22} {'all':>26}")
    for D, cell in sorted(gB.items(), key=lambda x: int(x[0])):
        cols = []
        for nm in ("prim_only", "seg_only", "chain_only", "all"):
            v = cell.get(nm, {})
            if "e_piece" not in v:
                cols.append("priced out")
            else:
                cols.append(f"{f(v['e_piece'])} k{v['k_shoot']} fb{f(v['fb'], 1)}"
                            + (f" ch{f(v['frac_chain'], 2)}" if nm == "all" else ""))
        W(f"  {D:>8} {cols[0]:>22} {cols[1]:>22} {cols[2]:>22} {cols[3]:>26}")
    W("     (the budget D at which `seg_only`/`chain_only` first beats `prim_only` is the meter")
    W("      the main run has to be set at; if there is none, a chunk never pays on this plant and")
    W("      that is the finding — do not sand it down by widening the library)")

    # ---------------------------------------------------------------- G-C
    gC = d.get("gC", {})
    W("\n-- G-C audition calibration, per level (the `aud_horizon` knob is read off this) --")
    W(f"  {'horizon':>8} {'cell':>7} {'rho':>7} {'regret':>9} {'ratio':>7} {'optimism':>9} "
      f"{'oracle':>7} {'distinct':>9}")
    for hz, cell in sorted(gC.items(), key=lambda x: int(x[0])):
        for key, v in sorted(cell.items()):
            W(f"  {hz:>8} {key:>7} {f(v['rho'], 3):>7} {f(v['regret']):>9} "
              f"{f(v['regret_ratio'], 2):>7} {f(v['optimism'], 2):>9} "
              f"{f(v['oracle_gain'], 2):>7} "
              f"{str(v.get('n_distinct_slots')) + '/' + str(v.get('n_slots_present')):>9}")
    W("     (rho = per-state rank agreement model↔plant; regret = how much worse the model's pick")
    W("      is than the plant's, in meters; optimism = true/predicted on MATCHED checkpoints;")
    W("      oracle = best-fixed ÷ per-state-oracle, fingering G1's 4.00× is the reference;")
    W("      distinct = how many slots the per-state oracle actually visits — 1 means the library")
    W("      is a `fixed` unit wearing a library's name)")

    # ---------------------------------------------------------------- G-D
    gD = d.get("gD", {})
    W("\n-- G-D cycle cost (what Phase B is sized from) --")
    W(f"  reactive practice cycle: {f(gD.get('warm_s_per_reactive_cycle'), 1)} s")
    for m in ("audit", "prop"):
        v = gD.get(m)
        if v:
            W(f"  {m:>6}: {f(v['s_per_traversal'], 2)} s/traversal  aud={v['aud']}  "
              f"delib={v['delib']}  plans={v['plans']}  e={f(v['e_piece'])}  "
              f"chain={f(v['frac_chain'], 2)}")
    led = d.get("ledger", {})
    if led:
        W(f"  gate ledger: agent steps {led.get('steps_agent')} · fb {led.get('fb_agent')} · "
          f"delib {led.get('delib_agent')} · aud {led.get('aud_agent')} · "
          f"t_priced {f(led.get('t_priced'), 1)}s")

    W("\n-- verdict --")
    for nm, ok in (("G-F fork fidelity", gF.get("pass")),
                   ("G-R rng discipline", gR.get("pass")),
                   ("G-P π fidelity at k=N", gP.get("pass"))):
        W(f"  {nm:<26} {'PASS' if ok else 'FAIL'}")
    W("  G-K / G-B / G-S / G-C are MEASUREMENTS, not pass/fail — read the tables above and pick")
    W("  (delib_budget, lib_add, aud_horizon, n_cycles) off them, with the reason on the record.")

    outp = os.path.join(HERE, "results", a.tag, "gate_report.txt")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[wrote] {outp}")


if __name__ == "__main__":
    main()
