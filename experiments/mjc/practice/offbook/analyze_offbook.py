"""Reduce an offbook main run: the headline, the routing readouts, the two ports' own instruments,
the consumption-phase ledger, and the end-of-run battery.

READING RULES this node inherits and does not relitigate:
  * Single seed. RANKS, SIGNS and in-tag twin floors are the claims; absolute values are not.
  * Arm-difference seed noise on this substrate is MEASURED (`practice/README.md`): delta-fixed
    adaptation 0.00518 +- 0.00083 and friends. Absolute probe variance is several times larger and is
    the wrong scale for arm comparisons.
  * The delta certificate reads the earliest of three clocks and is a DETECTOR only; commits here
    are scheduled from outside, as everywhere in this port.
  * Adam is invariant to global loss rescaling, so `span_lam` acts only through the relative weight
    of the two loss terms -- never read it as a plasticity budget.
  * If an arm was truncated by the wall clock, everything is reported at the COMMON HORIZON and
    flagged (the etude `eg_s0` / legato `never` precedent).

Usage (from experiments/):
    python3 mjc/practice/offbook/analyze_offbook.py --tag O1 --fetch
    modal run mjc/practice/offbook/analyze_offbook.py::figs --tag O1     # figures, on the volume
    modal volume get --force mujoco-control-data practice_offbook/O1/figs \\
        mjc/practice/offbook/results/O1
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# `python3 mjc/practice/offbook/analyze_offbook.py` puts THIS directory on sys.path, not
# `experiments/`, so the package import below would fail; `modal run` does put it there. Adding it
# explicitly makes the same file work both ways.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from mjc.shared import app, volume, DATA_DIR   # noqa: E402

ORDER = ["never", "key_frozen", "audit_all", "audit_prop_kN", "audit_prop_k", "route_native", "fid"]


# --------------------------------------------------------------------------- loading

def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_offbook/{tag}", dst], check=False,
                   env={**os.environ,
                        "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "chromatic")})


def load(tag):
    root = os.path.join(HERE, "results", tag)
    out = {}
    for base in (os.path.join(root, tag), root):
        if not os.path.isdir(base):
            continue
        for w in sorted(os.listdir(base)):
            p = os.path.join(base, w, "results.json")
            if os.path.isfile(p):
                d = json.load(open(p))
                # gate output has no `arm` key; ignore it, so a tag namespace collision is harmless
                if isinstance(d, dict) and d.get("arm"):
                    out[d["arm"]] = d
    if os.path.isdir(root):
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".json"):
                d = json.load(open(os.path.join(root, fn)))
                if isinstance(d, dict) and d.get("arm"):
                    out.setdefault(d["arm"], d)
    if not out:
        raise SystemExit(f"no arm results under {root}; run with --fetch")
    return {k: out[k] for k in ORDER if k in out} | {k: v for k, v in out.items()
                                                     if k not in ORDER}


def f(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    if isinstance(x, (list, tuple)):
        return "[" + " ".join(f(v, p) for v in x) + "]"
    return str(x)


def at_cycle(d, c):
    """The ladder record at or just before cycle c."""
    best = None
    for r in d.get("ladder", []):
        if r["cycle"] <= c:
            best = r
    return best or {}


# --------------------------------------------------------------------------- the report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="O1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--d-fb", default="0.0,0.1,1.0,3.0")
    ap.add_argument("--d-delib", default="0,1e-6,1e-5")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    D = load(a.tag)
    L = []

    def W(s=""):
        L.append(s)
        print(s)

    cfgs = {k: v.get("config", {}) for k, v in D.items()}
    c0 = next(iter(cfgs.values()))
    last = {k: max((r["cycle"] for r in d.get("ladder", [])), default=0) for k, d in D.items()}
    done = [k for k, d in D.items() if d.get("complete")]
    part = [k for k in D if k not in done]
    # TWO horizons, because the run was cancelled externally mid-flight and the truncation is not
    # symmetric. legato's precedent (and the etude's `eg_s0` before it): reduce at the COMMON
    # horizon whatever spans every arm, and keep the within-committed comparisons at their own full
    # length, since those arms share a schedule and are not affected by another arm dying.
    horizon = min(last.values()) if last else 0                 # spans every arm incl. partials
    hc = min([last[k] for k in done], default=horizon)           # the completed arms' own horizon
    trunc = part
    W(f"\n=== offbook — tag {a.tag} (seed {c0.get('seed')}) ===")
    W(f"arms {list(D)} · n_cycles {c0.get('n_cycles')} (reactive cap {c0.get('n_cycles_react')}) · "
      f"commits seg {c0.get('commit_seg')} chain {c0.get('commit_chain')} · lib_add "
      f"{c0.get('lib_add')} · n_slot {c0.get('n_slot')} · k_prop {c0.get('k_prop')} · "
      f"delib_budget {c0.get('delib_budget')} · aud_horizon {c0.get('aud_horizon')}")
    W(f"COMMON HORIZON c{horizon} (spans every arm)   ·   COMPLETED-ARM HORIZON c{hc}")
    if trunc:
        W(f"  PARTIAL ARMS: " + ", ".join(f"{k} (died c{last[k]}, complete=False)" for k in trunc))
        W("  The app received an EXTERNAL CANCELLATION at 02:12 UTC (clean cancel signal in the")
        W("  modal-client log — not a timeout and not preemption; cause unknown, recorded as such).")
        W("  So: any claim involving a partial arm is read at c%d and flagged; every" % horizon)
        W("  within-committed claim is read at c%d, where all four completed arms share a" % hc)
        W("  schedule and are unaffected by another arm dying.")
    st = next(iter(D.values())).get("setup", {})
    W(f"references (shared): stale {f(st.get('ref_stale'))} · ceiling {f(st.get('ref_ceiling'))} · "
      f"reactive {f(st.get('ref_reactive'))} · usable range {f(st.get('usable_range'))}")

    # ---------------------------------------------------------------- headline
    def headline(hz, arms, title):
        W(f"\n-- headline at c{hz} — {title} --")
        W(f"  {'arm':<15} {'e_perf':>8} {'by segment':>26} {'fb/pc':>7} {'t_priced':>10} "
          f"{'plans':>9} {'delib':>13} {'aud':>11} {'guard':>8}")
        rows = []
        for k in arms:
            r = at_cycle(D[k], hz)
            rows.append((k, r.get("e_perf")))
            W(f"  {k:<15} {f(r.get('e_perf')):>8} {f(r.get('e_perf_by_seg')):>26} "
              f"{f(r.get('n_fb_perf'), 1):>7} {f(r.get('t_cum'), 1):>10} "
              f"{r.get('plans_agent', 0):>9} {r.get('delib_agent', 0):>13} "
              f"{r.get('aud_agent', 0):>11} {f(r.get('guard_fm'), 5):>8}")
        rk = sorted([x for x in rows if x[1] is not None], key=lambda x: x[1])
        W("  rank on e_perf: " + " < ".join(f"{k}({f(v)})" for k, v in rk))

    headline(horizon, list(D), "EVERY arm, incl. the two the cancellation truncated")
    if hc != horizon:
        headline(hc, done, "the four arms that completed; within-committed claims live here")

    # ---------------------------------------------------------------- fidelity twins
    W("\n-- in-tag fidelity twins (the assertion, not a hope) --")
    for twin, ref in (("fid", "audit_all"), ("audit_prop_kN", "audit_all")):
        if twin in D and ref in D:
            mx, n = 0.0, 0
            for r1, r2 in zip(D[twin].get("ladder", []), D[ref].get("ladder", [])):
                for key in ("e_perf", "e_react", "e_ball_seg", "t_cum", "guard_fm"):
                    if r1.get(key) is not None and r2.get(key) is not None:
                        mx = max(mx, abs(float(r1[key]) - float(r2[key]))); n += 1
            ov = min(last.get(twin, 0), last.get(ref, 0))
            W(f"  {twin:<15} vs {ref:<12} max|Δ| over {n} probe values (through c{ov}) "
              f"= {mx:.3e}   {'PASS' if mx == 0.0 else 'FAIL'}")
            if last.get(twin, 0) < last.get(ref, 0):
                W(f"    ({twin} was truncated at c{last[twin]}; the assertion holds over the "
                  f"{n} probe values it reached, not over the full run)")
        elif twin in D or ref in D:
            W(f"  {twin} vs {ref}: one side absent, twin not asserted")

    # ---------------------------------------------------------------- the routing readouts
    W("\n-- what the routing actually chose (final performance configuration) --")
    W(f"  {'arm':<15} {'chain':>7} {'tape':>7} {'head':>7} {'prim':>7} {'cand/dec':>9} "
      f"{'aud/piece':>10} {'poison':>8}")
    for k, d in D.items():
        m = (d.get("summary", {}) or {}).get("final_mix")
        if not m:
            why = ("no library by construction: reactive throughout"
                   if d.get("config", {}).get("arm") == "never" or k == "never"
                   else f"truncated at c{last.get(k, 0)} before the summary block was written")
            W(f"  {k:<15} —  ({why})")
            continue
        W(f"  {k:<15} {f(m['frac_chain'], 3):>7} {f(m['frac_tape'], 3):>7} "
          f"{f(m['frac_head'], 3):>7} {f(m['frac_prim'], 3):>7} {f(m['cand_mean'], 1):>9} "
          f"{m['aud_total']:>10} {f(m.get('poison_launch'), 3):>8}")
    W("     (frac_chain = decisions that committed a unit spanning >1 segment — the readout legato")
    W("      F4's crossover predicts should be non-zero only where the unit exceeds the horizon;")
    W("      frac_head = Port 2 serving; cand/dec = the O(K)→O(k) ladder, measured)")

    # ---------------------------------------------------------------- the rent, and at what K
    W("\n-- THE RENT, and what it would be at a different library size --")
    W("  Gate G-K measured competence FLAT past K≈16 (ns=1) and DEGRADING past K=64 (ns=3), so a")
    W("  library far beyond that plateau would manufacture the routing arm's advantage: rent no")
    W("  competent agent needed to pay. K is therefore set AT the plateau. Note what this table")
    W("  shows: the RATIO is a property of the slot partition (n_slot ÷ k_prop), not of K — the")
    W("  absolute price scales with the library, the cut does not.")
    W(f"  {'arm':<15} {'aud/piece':>10} " + " ".join(f"{'K=' + str(k):>10}" for k in (16, 64, 288)))
    base_K = None
    for k, d in D.items():
        m = (d.get("summary", {}) or {}).get("final_mix")
        lib = (d.get("library") or {}).get("sizes") or {}
        if not m or not lib:
            continue
        Kc = max(int(v) for v in lib.values()) or 1
        base_K = base_K or Kc
        cells = [f"{m['aud_total'] * kp / Kc:>10.0f}" for kp in (16, 64, 288)]
        W(f"  {k:<15} {m['aud_total']:>10} " + " ".join(cells))
    W(f"     (linear in K within a cell, which is exact for this design — every cell is grown by the")
    W(f"      same `lib_add`. Measured K/cell = {base_K}.)")

    # ---------------------------------------------------------------- trust formation
    W("\n-- PORT 1: trust formation (π's per-level proposal mass over cycles) --")
    for k, d in D.items():
        pr = d.get("prop", [])
        if not pr:
            continue
        W(f"  {k}:")
        W(f"    {'cycle':>6} {'seg':>7} {'chain':>7} {'prim':>7} {'poison':>7} "
          f"{'top1':>7} {'H':>7}")
        for p in pr:
            W(f"    {p['cycle']:>6} {f(p['mass_seg'], 3):>7} {f(p['mass_chain'], 3):>7} "
              f"{f(p['mass_prim'], 3):>7} {f(p['mass_poison'], 3):>7} "
              f"{f(p['top1'], 3):>7} {f(p['entropy'], 3):>7}")
    W("     (`census/`+`assay/`: the deep-era value of a vocabulary rides on ARRIVAL — trust, π's")
    W("      per-level proposal mass, formed only by time-in-use. This is that series, on a plant.)")

    W("\n-- PORT 1: the can't-decompose signature (mass on a chain's own segment spelling) --")
    for k, d in D.items():
        pr = d.get("prop", [])
        if not pr:
            continue
        for p in pr[-1:]:
            ch = [c for c in p.get("chains", []) if c.get("n_argmax", 0) > 0]
            if not ch:
                W(f"  {k}: π never argmaxes a chain slot at c{p['cycle']} "
                  f"(all-state mass: chain {f(max([c['p_chain_all'] for c in p.get('chains', [])] or [0]), 4)}"
                  f" vs spelling {f(max([c['p_spell_all'] for c in p.get('chains', [])] or [0]), 4)})")
                continue
            for c in ch:
                W(f"  {k}: slot {c['slot']} (ns={c['ns']}, seam {c['seam']}) argmax on "
                  f"{f(c['frac_argmax'], 3)} of states — p(chain) {f(c['p_chain'], 3)} vs "
                  f"p(spelling) {f(c['p_spell'], 3)}  ratio {f(c.get('ratio'), 3)}")

    # ---------------------------------------------------------------- port 2
    W("\n-- PORT 2: parity trajectory (a slot fires from the head only once it clears τ) --")
    for k, d in D.items():
        pl = d.get("parity", [])
        if not pl:
            continue
        opens = [(p["cycle"], p["open"], p["n"]) for p in pl]
        first = next((c for c, o, _ in opens if o > 0), None)
        W(f"  {k}: slots with a held-out set {opens[-1][2]} · open at end {opens[-1][1]} · "
          f"first open at c{first if first is not None else '—'} · "
          f"τ={c0.get('span_tau')}")
        last = pl[-1].get("frac", {})
        vals = sorted((v for v in last.values() if v is not None), reverse=True)
        W(f"     end parity fractions (top 8): {f([float(x) for x in vals[:8]], 2)}")
        s = d.get("summary", {})
        W(f"     head calls {s.get('head_calls')} vs tape calls {s.get('tape_calls')}")
    W("     (parity is re-checked EVERY cycle; a slot that drops below τ falls back to tape playback")
    W("      verbatim, so 'below parity → keep the tape' cannot introduce drift)")

    # ---------------------------------------------------------------- plant guard
    W("\n-- PLANT GUARD: one-step FM error on the fixed corridor probe, and e_react --")
    W(f"  {'arm':<15} {'guard c0':>9} {'guard end':>10} {'Δ':>9} {'e_react c0':>11} "
      f"{'e_react end':>12} {'@cycle':>7}")
    for k, d in D.items():
        lg = d.get("log", {})
        g = lg.get("guard_fm", [])
        lad = d.get("ladder", [])
        g0 = d.get("setup", {}).get("guard_fm0")
        r0 = lad[0].get("e_react") if lad else None
        hz = last.get(k, horizon)
        rn = at_cycle(d, hz).get("e_react")
        gn = at_cycle(d, hz).get("guard_fm", g[-1] if g else None)
        W(f"  {k:<15} {f(g0, 5):>9} {f(gn, 5):>10} "
          f"{f((gn - g0) if (gn is not None and g0 is not None) else None, 5):>9} "
          f"{f(r0):>11} {f(rn):>12} {('c' + str(hz)):>7}")
    W("     (flat across arms is the pass. Port 2's loss reaches the forward model's own trunk by")
    W("      design — this is where interference would show, and it is the mjc analogue of RHM's")
    W("      parse/infill staying inert under a head on the generator trunk.)")

    # ---------------------------------------------------------------- the consumption phase
    W("\n-- the CONSUMPTION-PHASE ledger (legato's named round-3 design item) --")
    dt = 0.02
    steps_pc = int(c0.get("h_app", 14)) + sum(int(x) for x in c0.get("h_seg", [20, 20, 20]))
    W(f"  one traversal = {steps_pc} control steps at dt={dt}s = {steps_pc * dt:.2f}s of body time")
    W(f"  {'arm':<15} {'e_perf':>8} {'fb/pc':>7} " + " ".join(
        f"{'t@dfb=' + x:>13}" for x in a.d_fb.split(",")))
    for k, d in D.items():
        r = at_cycle(d, last.get(k, horizon))
        fb = r.get("n_fb_perf")
        if fb is None:
            continue
        cells = []
        for x in a.d_fb.split(","):
            cells.append(f"{steps_pc * dt + fb * float(x):>13.3f}")
        W(f"  {k:<15} {f(r.get('e_perf')):>8} {f(fb, 1):>7} " + " ".join(cells))
    W("  steady-state per-traversal deliberation (the axis routing is supposed to buy):")
    W(f"  {'arm':<15} {'plans/pc':>9} {'delib/pc':>12} {'aud/pc':>10}")
    for k, d in D.items():
        lg = d.get("log", {})
        cy = lg.get("cycle", [])
        if len(cy) < 4:
            continue
        i0 = max(0, len(cy) - max(2, len(cy) // 3)) - 1
        n_tr = (cy[-1] - cy[i0]) * 2          # practice + metering traversals per cycle
        dp = (lg["plans_agent"][-1] - lg["plans_agent"][i0]) / max(n_tr, 1)
        dd = (lg["delib_agent"][-1] - lg["delib_agent"][i0]) / max(n_tr, 1)
        da = (lg["aud_agent"][-1] - lg["aud_agent"][i0]) / max(n_tr, 1)
        W(f"  {k:<15} {dp:>9.1f} {dd:>12.0f} {da:>10.1f}")
    W("     (measured over the last third of cycles, i.e. after the final commit — the acquisition")
    W("      phase dominated every mjc ledger so far, which is why `never` always won cumulatively)")

    # ---------------------------------------------------------------- the battery
    W("\n-- END-OF-RUN BATTERY on the final trained state --")
    W(f"  {'arm':<15} {'probe':<12} {'e_perf':>8} {'minability':>11} {'tape':>6} {'head':>6} "
      f"{'prim':>6} {'chain':>6}")
    for k, d in D.items():
        b = d.get("battery", {})
        for nm in ("base", "no_prim", "no_table", "restored"):
            v = b.get(nm)
            if not v:
                continue
            if "error" in v:
                W(f"  {k:<15} {nm:<12} {'— ' + v['error'][:52]}")
                continue
            m = v.get("mix") or {}
            W(f"  {k:<15} {nm:<12} {f(v['e_perf']):>8} {v['minability']:>11} "
              f"{f(m.get('frac_tape'), 2):>6} {f(m.get('frac_head'), 2):>6} "
              f"{f(m.get('frac_prim'), 2):>6} {f(m.get('frac_chain'), 2):>6}")
    miss = [k for k in D if not D[k].get("battery")]
    if miss:
        W(f"  NO BATTERY: {', '.join(miss)} — "
          f"{'never has no library by construction (reactive throughout), so none is defined; ' if 'never' in miss else ''}"
          f"the rest were truncated by the cancellation before the end-of-run block ran.")
    W("     (`no_table` = the address book deleted: π + the span head must serve. Routing-not-pruning")
    W("      predicts the CURRENT level survives while the next level's BUILT entries do not — and")
    W("      the observation stream, `minability`, is counted table-free from the agent's own chosen")
    W("      traversals, so it is the half that should not move. `fid`'s untrained heads under the")
    W("      same forced-open policy are the negative control.)")

    # ---------------------------------------------------------------- the poison twin
    W("\n-- the POISON TWIN (one plausible-but-bad address in the library) --")
    W(f"  {'arm':<15} {'π mass (end)':>13} {'launch frac':>12} "
      f"{'materialised every seam?':>26}")
    for k, d in D.items():
        pr = d.get("prop", [])
        m = (d.get("summary", {}) or {}).get("final_mix") or {}
        mass = pr[-1]["mass_poison"] if pr else None
        W(f"  {k:<15} {f(mass, 4):>13} {f(m.get('poison_launch'), 4):>12} "
          f"{('n/a (no library)' if k == 'never' else ('yes (enumeration has no filter)' if ARM_ENUM(k) else 'no (π gates it)')):>26}")
    W("     (`native/` finding 6 inverted the SPEC's guess: enumeration is FORCED to materialise a")
    W("      bogus address at every tip while a routed policy, trained only on successful")
    W("      traversals, starves it — selection-before-regression is also a filter against a")
    W("      poisoned vocabulary. This is that contrast on a plant.)")

    W("\n-- library at the end --")
    for k, d in D.items():
        lib = d.get("library")
        if lib:
            W(f"  {k}: sizes {json.dumps(lib['sizes'])}")
            W(f"     slot occupancy {json.dumps(lib['slot_counts'])}")
            W(f"     chain spellings {json.dumps(lib['spellings'])[:200]}")

    outp = os.path.join(HERE, "results", a.tag, "report.txt")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[wrote] {outp}")


def ARM_ENUM(name):
    return name in ("audit_all", "fid", "audit_prop_kN", "key_frozen")


# --------------------------------------------------------------------------- figures (remote)

@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_figures(tag: str) -> list:
    """Figures are drawn REMOTELY and committed to the volume, because the analysis machine in this
    repo's sessions has neither numpy nor matplotlib (the `arm_env.py` contract, one level up)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_offbook", tag)
    figs = os.path.join(root, "figs")
    os.makedirs(figs, exist_ok=True)
    D = {}
    for w in sorted(os.listdir(root)):
        p = os.path.join(root, w, "results.json")
        if os.path.isfile(p):
            d = json.load(open(p))
            if d.get("arm"):
                D[d["arm"]] = d
    order = [k for k in ORDER if k in D] + [k for k in D if k not in ORDER]
    cols = dict(zip(order, plt.cm.tab10.colors))
    made = []

    def save(fig, name):
        p = os.path.join(figs, name)
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        made.append(name)

    # fig1 -- the ladder + the plant guard
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in order:
        lad = D[k].get("ladder", [])
        c = [r["cycle"] for r in lad]
        ax[0].plot(c, [r["e_perf"] for r in lad], label=k, color=cols[k])
        ax[1].plot(c, [r.get("e_react") for r in lad], label=k, color=cols[k])
        ax[2].plot(c, [r.get("guard_fm") for r in lad], label=k, color=cols[k])
    for i, t in enumerate(("held-out piece error (own configuration)",
                           "PLANT GUARD: e_react (common control mode)",
                           "PLANT GUARD: one-step FM corridor error")):
        ax[i].set_title(t, fontsize=9); ax[i].set_xlabel("cycle"); ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    save(fig, "fig1_ladder_and_guard.png")

    # fig2 -- trust formation and the level mix
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in order:
        pr = D[k].get("prop", [])
        if pr:
            c = [p["cycle"] for p in pr]
            ax[0].plot(c, [p["mass_seg"] for p in pr], color=cols[k], label=f"{k} seg")
            ax[0].plot(c, [p["mass_chain"] for p in pr], color=cols[k], ls="--",
                       label=f"{k} chain")
            ax[1].plot(c, [p["mass_prim"] for p in pr], color=cols[k], label=f"{k} prim")
            ax[1].plot(c, [p["mass_poison"] for p in pr], color=cols[k], ls=":",
                       label=f"{k} poison")
        lad = D[k].get("ladder", [])
        mx = [(r["cycle"], (r.get("mix") or {}).get("frac_chain")) for r in lad]
        mx = [(a_, b_) for a_, b_ in mx if b_ is not None]
        if mx:
            ax[2].plot([a_ for a_, _ in mx], [b_ for _, b_ in mx], color=cols[k], label=k)
    ax[0].set_title("PORT 1 trust formation: π mass by level", fontsize=9)
    ax[1].set_title("π mass on the live plan and on the poison twin", fontsize=9)
    ax[2].set_title("fraction of decisions that committed a CHAIN", fontsize=9)
    for x in ax:
        x.set_xlabel("cycle"); x.grid(alpha=.3); x.legend(fontsize=6)
    save(fig, "fig2_trust_and_mix.png")

    # fig3 -- the cost ladder: what routing buys, per traversal
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in order:
        lg = D[k].get("log", {})
        if not lg.get("cycle"):
            continue
        c = lg["cycle"]
        ax[0].plot(c, lg.get("aud_agent", []), color=cols[k], label=k)
        ax[1].plot(c, lg.get("delib_agent", []), color=cols[k], label=k)
    ax[0].set_title("cumulative candidate MATERIALISATIONS (the O(K) rent)", fontsize=9)
    ax[1].set_title("cumulative FM rollout-steps (audition + search)", fontsize=9)
    for x in ax:
        x.set_xlabel("cycle"); x.set_yscale("log"); x.grid(alpha=.3); x.legend(fontsize=7)
    save(fig, "fig3_cost_ladder.png")

    # fig4 -- port 2 parity and the battery
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in order:
        pl = D[k].get("parity", [])
        if pl:
            ax[0].plot([p["cycle"] for p in pl], [p["open"] for p in pl], color=cols[k], label=k)
    ax[0].set_title("PORT 2: slots above parity τ (re-checked every cycle)", fontsize=9)
    ax[0].set_xlabel("cycle"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    names = ["base", "no_prim", "no_table", "restored"]
    wdt = 0.8 / max(len(order), 1)
    for i, k in enumerate(order):
        b = D[k].get("battery", {})
        vals = [(b.get(n, {}) or {}).get("e_perf", np.nan) for n in names]
        ax[1].bar(np.arange(len(names)) + i * wdt, vals, wdt, label=k, color=cols[k])
    ax[1].set_xticks(np.arange(len(names)) + 0.4)
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_title("end-of-run battery: piece error", fontsize=9)
    ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=7)
    save(fig, "fig4_ports_and_battery.png")

    volume.commit()
    return made


@app.local_entrypoint()
def figs(tag: str = "O1"):
    made = make_figures.remote(tag)
    print(f"[figs] wrote {made} to /data/practice_offbook/{tag}/figs on the volume")
    print(f"[figs] pull with: modal volume get --force mujoco-control-data "
          f"practice_offbook/{tag}/figs mjc/practice/offbook/results/{tag}")


if __name__ == "__main__":
    main()
