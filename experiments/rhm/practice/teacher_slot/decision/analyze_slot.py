"""Reduce a teacher_slot/decision run.

    python3 rhm/practice/teacher_slot/decision/analyze_slot.py --tag tsdA --fetch --figures

Sections, in the order Rung A asks for them:
  0. SETUP — the world, the ledger (budget, measured probe price), and what each loop reads.
  1. FIDELITY GATE — the hard-coded arms must reproduce their `fourwall/lm` twins
     BIT-FOR-BIT at every shared checkpoint (the donor's 9a discipline, max|delta| = 0.0).
     Nothing below this line is licensed if this fails.
  2. THE DECISION — whether and when each loop first collapses `w`; when its VALUE flips;
     when it commits; whether it restores during the transient (declines / bail-outs).
  3. THE LEDGER — probe spend as a fraction of budget, and the training steps it bought.
  4. TIMING against the exogenous bracket (merge_1000 / merge_8000 / merge_13000).
  5. TERMINAL — pathway wholeness (key-anchor d4 rand and the probe-free twin) and task NLL.
  6. THE LIFETIME TASK INTEGRAL — `fourwall/lm`'s definition, plus a common-grid version
     (arms that bought probes train for fewer steps, so the raw mean is not matched).
  7. THE TRANSIENT — what each currency does in the interval after a commit: the fact the
     whole rung turns on is that one of them improves inside the dip.
  8. THE DECISION TRACE — the scientific object, printed per arm.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_teacher_slot_decision"
DONOR_REMOTE = "rhm_practice_fourwall_lm"
DONOR_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                           "practice", "fourwall", "lm", "figures")
GATE_INSTR = ("nll.true.idx", "nll.true.out", "binding.idx",
              "index.transfer_gap", "index.jsd_mean")
# gate arm -> (donor tag, donor arm). The twins licensed by the donor's own 9a gate.
GATE_TWINS = {"wall": [("fwlm0", "wall")],
              "no_wall": [("fwlm1", "no_wall"), ("fwlm0", "no_wall")],
              "merge_8000": [("fwlm1", "merge_8000")],
              "merge_1000": [("fwlm1", "merge_1000")]}
# fwlm1's stored records, licensed for comparison once the gates pass (same machinery,
# only the hard-coded step differs)
BRACKET = {"merge_1000": ("fwlm1", 1000), "merge_8000": ("fwlm1", 8000),
           "merge_13000": ("fwlm1", 13000), "no_wall": ("fwlm1", 0),
           "wall": ("fwlm0", None), "true_wall": ("fwlm0", None)}


def fetch(tag, remote=REMOTE, dest=None):
    dest = dest or FIG
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{remote}/{tag}", dest],
                   check=True)


def load_dir(root):
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name == "setup.json":
            continue
        r = json.load(open(os.path.join(root, name)))
        arms[r["arm"]] = r
    order = [a for a in setup["arms"] if a in arms]
    order += [a for a in arms if a not in order]      # never drop a record on the disk
    return setup, {a: arms[a] for a in order}


def load_donor(tag):
    """The donor's fetched records, from its own figures/ dir if present else ours."""
    for root in (os.path.join(DONOR_LOCAL, tag), os.path.join(FIG, tag)):
        if os.path.isdir(root) and os.path.exists(os.path.join(root, "setup.json")):
            return load_dir(root)
    return None, {}


def _dig(d, parts):
    if not parts:
        return d
    if not isinstance(d, dict):
        return None
    for k in range(len(parts), 0, -1):
        key = ".".join(parts[:k])
        if key in d:
            r = _dig(d[key], parts[k:])
            if r is not None:
                return r
    return None


def series(rec, path):
    return np.array([np.nan if _dig(r, path.split(".")) is None
                     else float(_dig(r, path.split("."))) for r in rec["log"]])


def steps(rec):
    return np.array([r["step"] for r in rec["log"]])


def at(rec, path, step, tol=None):
    s, y = steps(rec), series(rec, path)
    i = int(np.argmin(np.abs(s - step)))
    if tol is not None and abs(s[i] - step) > tol:
        return np.nan
    return y[i]


def cons_idx(rec):
    """Indexed-span NLL under the arm's CONSUMED condition — the task readout."""
    return np.array([r["nll"][r.get("consumed", "true")]["idx"] for r in rec["log"]])


def path_excess(rec):
    """The probe-free pathway readout: indexed-span excess over exact Bayes under the
    NEUTRAL wall. Lower = the token-derived inference pathway is more whole."""
    return series(rec, "excess.none.idx")


def probe_d4(rec, step, cond="rand", tol=1500):
    cands = [r for r in rec["log"] if "levels" in r and cond in r["levels"]]
    if not cands:
        return None
    r = min(cands, key=lambda z: abs(z["step"] - step))
    if abs(r["step"] - step) > tol:
        return None
    return r["levels"][cond]["key"]["d4"], r["step"]


def probe_cons(rec, step, tol=1500):
    cands = [r for r in rec["log"] if "levels" in r]
    if not cands:
        return None
    r = min(cands, key=lambda z: abs(z["step"] - step))
    if abs(r["step"] - step) > tol:
        return None
    c = r.get("consumed", "true")
    c = c if c in r["levels"] else "true"
    return r["levels"][c]["key"]["d4"], r["step"]


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fetch-donor", action="store_true",
                    help="also pull fwlm0/fwlm1 from the volume (default: use the "
                         "donor's own fetched copies under fourwall/lm/figures/)")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--trace-every", type=int, default=1)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    if a.fetch_donor:
        for t in ("fwlm0", "fwlm1"):
            fetch(t, DONOR_REMOTE)
    setup, arms = load_dir(os.path.join(FIG, a.tag))
    cfg, refs = setup["config"], setup["refs"]
    W = 96
    MS = cfg["max_steps"]
    ck = cfg["ckpt_every"]
    bracket_ref = {t: load_donor(t) for t in ("fwlm0", "fwlm1")}

    # ---------------- 0. setup ---------------- #
    print("=" * W)
    print(f"teacher_slot/decision — tag {a.tag}   v{cfg['v']}_s{cfg['s']}_L{cfg['depth']}"
          f"_m{cfg['m']}  {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D")
    print("=" * W)
    g0 = refs["gate0"]
    print(f"  refs source: {setup.get('refs_source')}")
    print(f"  exact index bracket (indexed span) {g0['index_value_indexed_span']:+.4f} "
          f"nats   Bayes level {g0['bayes_indexed_span']:.4f}")
    print(f"  key-anchor d4 exact ceilings: no-wall {refs['ceil']['key_none']['d4']:.3f} "
          f"/ wall given {refs['ceil']['key_wall']['d4']:.3f}")
    print(f"  budget {cfg['budget_total']} step-equivalents; pathway probe "
          f"{cfg['probe_price']} steps ({cfg['n_policy']} sequences); task read free")
    print(f"  decision cadence {cfg['decide_every']} steps; checkpoint grid {ck}")
    print(f"\n  {'arm':17s} {'kind':6s} {'rot':>6s}  policy")
    for arm, rec in arms.items():
        sc = rec["sched"]
        print(f"  {arm:17s} {sc['kind']:6s} {sc['rot_period']:6d}  {sc['policy']}")

    # ---------------- 1. the fidelity gate ---------------- #
    print("\n" + "=" * W)
    print("1. FIDELITY GATE — the hard-coded arms vs their `fourwall/lm` twins")
    print("=" * W)
    print("   A policy hard-coded to a step function reduces to the donor's own condition")
    print("   rule, probes nothing, and consumes no RNG; refs are LOADED from the donor's")
    print("   setup.json. So these must agree in EVERY PRINTED DIGIT at every shared")
    print(f"   checkpoint, on {len(GATE_INSTR)} instruments. Required: max|delta| = 0.0.")
    gate_ok, gate_rows = True, []
    for arm in arms:
        for dtag, darm in GATE_TWINS.get(arm, []):
            dar = bracket_ref[dtag][1]
            if darm not in dar:
                print(f"   {arm:17s} twin {dtag}/{darm} not available locally")
                continue
            twin = dar[darm]
            tmap = {r["step"]: r for r in twin["log"]}
            worst, n, where = 0.0, 0, None
            for r in arms[arm]["log"]:
                q = tmap.get(r["step"])
                if q is None:
                    continue
                n += 1
                for p in GATE_INSTR:
                    x, y = _dig(r, p.split(".")), _dig(q, p.split("."))
                    if x is None or y is None:
                        continue
                    d = abs(float(x) - float(y))
                    if d > worst:
                        worst, where = d, (r["step"], p)
            ok = worst == 0.0 and n > 0
            gate_ok &= ok
            gate_rows.append((arm, f"{dtag}/{darm}", n, worst))
            print(f"   {arm:17s} vs {dtag}/{darm:12s} {n:4d} shared checkpoints  "
                  f"max|delta| = {worst:.3e}  {'PASS' if ok else 'FAIL ' + str(where)}")
    print(f"\n   GATE: {'PASS — cross-tag references licensed' if gate_ok else 'FAIL'}")

    # ---------------- 2. the decision ---------------- #
    print("\n" + "=" * W)
    print("2. THE DECISION — whether, when, and whether it held")
    print("=" * W)
    print("   first        first interval under COLLAPSE (warm-up sampling included)")
    print("   value-flip   first decision point whose argmax Q is COLLAPSE")
    print("   commit       start of the terminal run of greedily-chosen COLLAPSE")
    print("   decline      a greedy return to KEEP right after any collapse interval")
    print("   bail-out     a greedy return to KEEP right after a CHOSEN collapse interval")
    hdr = (f"   {'arm':17s} {'first':>7s} {'v-flip':>7s} {'commit':>7s} {'holds':>6s} "
           f"{'decl':>5s} {'bail':>5s} {'switch':>7s} {'%coll':>7s} {'final':>9s}")
    print(hdr)
    for arm, rec in arms.items():
        sm = rec.get("summary") or {}
        if not sm:
            print(f"   {arm:17s}  (hard-coded / no trace)")
            continue
        f = lambda k: "-" if sm.get(k) is None else str(sm[k])
        print(f"   {arm:17s} {f('first_collapse_step'):>7s} "
              f"{f('first_greedy_collapse_step'):>7s} {f('commit_step'):>7s} "
              f"{sm.get('n_holds', 0):6d} {sm.get('n_declines', 0):5d} "
              f"{sm.get('n_bailouts', 0):5d} {sm.get('n_switches', 0):7d} "
              f"{100 * sm.get('frac_decisions_collapsed', 0):6.1f}% "
              f"{sm.get('terminal_action', '-'):>9s}")
    print("\n   exploratory contamination (steps spent under COLLAPSE by warm-up or a")
    print("   scheduled trial rather than by choice):")
    for arm, rec in arms.items():
        sm = rec.get("summary") or {}
        if sm:
            print(f"   {arm:17s} {sm.get('exploratory_collapsed_steps', 0):6d} of "
                  f"{sm.get('collapsed_steps', 0):6d} collapsed steps")

    # ---------------- 3. the ledger ---------------- #
    print("\n" + "=" * W)
    print("3. THE LEDGER — evaluation is priced (ear's convention)")
    print("=" * W)
    print(f"   {'arm':17s} {'probes':>7s} {'spend':>7s} {'%budget':>8s} "
          f"{'trained':>8s} {'ckpts':>6s}")
    for arm, rec in arms.items():
        b = rec.get("budget", {})
        term = rec.get("terminal_step", steps(rec)[-1])
        print(f"   {arm:17s} {b.get('n_probes', 0):7d} {b.get('spend', 0):7d} "
              f"{100 * b.get('spend', 0) / max(1, b.get('total', MS)):7.2f}% "
              f"{term:8d} {len(rec['log']):6d}")

    # ---------------- 4. timing vs the bracket ---------------- #
    print("\n" + "=" * W)
    print("4. TIMING against the exogenous bracket")
    print("=" * W)
    print("   fwlm1's supplied merges sit at 1000 / 8000 / 13000; the `wall` schedule's")
    print(f"   first rotation is at {cfg['phase1']}, then every {cfg['rot_period']}.")
    for arm, rec in arms.items():
        sm = rec.get("summary") or {}
        c = sm.get("commit_step")
        if c is None:
            continue
        rel = "before the first rotation" if c < cfg["phase1"] else \
            f"{(c - cfg['phase1']) // max(1, cfg['rot_period']) + 1} rotations in"
        print(f"   {arm:17s} commits at {c:6d}  ({rel})")

    # ---------------- 5. terminal ---------------- #
    print("\n" + "=" * W)
    print("5. TERMINAL — pathway wholeness and task error")
    print("=" * W)
    ceil = refs["ceil"]["key_none"]["d4"]
    print(f"   d4 rand / none: probe recovery of the keyed latent with the wall")
    print(f"   randomised / neutral — the TOKEN-DERIVED pathway. Exact ceiling {ceil:.3f}.")
    print(f"   probe-free twin: indexed-span excess over exact Bayes, neutral wall (lower")
    print("   = more whole). Both are read at each arm's own terminal checkpoint.")
    print(f"\n   {'arm':17s} {'step':>7s} {'d4 rand':>8s} {'d4 none':>8s} "
          f"{'pathfree':>9s} {'task NLL':>9s} {'vs no_wall':>11s}")
    base = arms.get("no_wall") or bracket_ref["fwlm1"][1].get("no_wall")
    rows_term = {}
    for arm, rec in arms.items():
        t = int(steps(rec)[-1])
        dr = probe_d4(rec, t, "rand"); dn = probe_d4(rec, t, "none")
        pf = float(path_excess(rec)[-1]); tk = float(cons_idx(rec)[-1])
        bt = np.nan if base is None else float(at(base, "nll.none.idx", t))
        rows_term[arm] = {"step": t, "d4_rand": None if dr is None else dr[0],
                          "d4_none": None if dn is None else dn[0],
                          "pathfree": pf, "task": tk}
        print(f"   {arm:17s} {t:7d} "
              f"{'    -   ' if dr is None else f'{dr[0]:8.3f}'} "
              f"{'    -   ' if dn is None else f'{dn[0]:8.3f}'} "
              f"{pf:9.4f} {tk:9.4f} {tk - bt:+11.4f}")
    # arms that bought probes trained for fewer steps, so also read everything at the
    # last checkpoint they ALL reached — the matched-step comparison
    common = min(int(steps(r)[-1]) for r in arms.values())
    common = (common // ck) * ck
    print(f"\n   matched-step readout, every arm at step {common} (the last checkpoint")
    print("   every arm reached; probe-buying arms stop earlier than the free ones):")
    print(f"   {'arm':17s} {'step':>7s} {'d4 rand':>8s} {'pathfree':>9s} {'task NLL':>9s}")
    for arm, rec in arms.items():
        dr = probe_d4(rec, common, "rand")
        pf = float(at(rec, "excess.none.idx", common, tol=ck))
        s_, y_ = steps(rec), cons_idx(rec)
        tk = float(y_[int(np.argmin(np.abs(s_ - common)))])
        print(f"   {arm:17s} {common:7d} "
              f"{'    -   ' if dr is None else f'{dr[0]:8.3f}'} {pf:9.4f} {tk:9.4f}")

    for dtag in ("fwlm0", "fwlm1"):
        _, dar = bracket_ref[dtag]
        for darm, rec in dar.items():
            if darm in arms and dtag == "fwlm1":
                continue
            t = int(steps(rec)[-1])
            if darm in ("wall", "wall_fast") and t == MS:
                t = int(steps(rec)[steps(rec) <= MS - ck][-1])
            dr = probe_d4(rec, t, "rand")
            pf = float(at(rec, "excess.none.idx", t))
            tk = float(cons_idx(rec)[np.argmin(np.abs(steps(rec) - t))])
            print(f"   [{dtag}] {darm:11s} {t:7d} "
                  f"{'    -   ' if dr is None else f'{dr[0]:8.3f}'} "
                  f"{'    -   ':>8s} {pf:9.4f} {tk:9.4f}")

    # ---------------- 6. the lifetime integral ---------------- #
    print("\n" + "=" * W)
    print("6. THE LIFETIME TASK INTEGRAL — fwlm1's definition")
    print("=" * W)
    print("   `raw` = mean of indexed-span NLL under the consumed condition over ALL of")
    print("   the arm's checkpoints (fwlm1's definition, reproduced). `grid` restricts to")
    print(f"   the common {ck}-step grid up to the shortest arm's terminal step, so arms")
    print("   that bought probes (and therefore trained less) are compared like for like.")
    common_end = min(int(steps(r)[-1]) for r in arms.values())
    grid = [s for s in range(0, common_end + 1, ck)]
    print(f"   common grid: 0 .. {common_end} step {ck} ({len(grid)} points)")
    print(f"\n   {'arm':17s} {'raw':>9s} {'grid':>9s} {'rank':>5s}")
    lifetimes = {}
    for arm, rec in arms.items():
        raw = float(np.mean(cons_idx(rec)))
        smap = {int(s): y for s, y in zip(steps(rec), cons_idx(rec))}
        gvals = [smap[s] for s in grid if s in smap]
        lifetimes[arm] = (raw, float(np.mean(gvals)) if gvals else np.nan, len(gvals))
    for dtag, darm in (("fwlm0", "true_wall"), ("fwlm1", "merge_1000"),
                       ("fwlm1", "merge_13000")):
        _, dar = bracket_ref[dtag]
        if darm in dar:
            rec = dar[darm]
            raw = float(np.mean(cons_idx(rec)))
            smap = {int(s): y for s, y in zip(steps(rec), cons_idx(rec))}
            gvals = [smap[s] for s in grid if s in smap]
            lifetimes[f"[{dtag}] {darm}"] = (raw, float(np.mean(gvals)) if gvals
                                             else np.nan, len(gvals))
    order = sorted(lifetimes, key=lambda k: lifetimes[k][1])
    for i, arm in enumerate(order):
        raw, gr, ng = lifetimes[arm]
        print(f"   {arm:17s} {raw:9.4f} {gr:9.4f} {i + 1:5d}   (n={ng})")

    # ---------------- 7. the transient ---------------- #
    print("\n" + "=" * W)
    print("7. THE TRANSIENT — the two currencies inside the dip")
    print("=" * W)
    print("   The rung turns on fwlm1's observation that at the instant of the op, task")
    print("   NLL is spiked while the pathway readout is already climbing. Read at each")
    print("   arm's own commit (policy arms) or merge step (hard-coded arms), M+d:")
    offs = (0, 25, 50, 75, 125, 250, 500, 1000, 2000)
    for lab, fn in (("task NLL (consumed)", cons_idx),
                    ("pathway excess (neutral, lower=whole)", path_excess)):
        print(f"\n   {lab}:")
        print(f"   {'arm':17s} {'M':>6s} " + "".join(f"{'M+' + str(d):>9s}" for d in offs))
        for arm, rec in arms.items():
            sm = rec.get("summary") or {}
            M = sm.get("commit_step")
            if M is None:
                M = rec["sched"].get("merge_at")
                if M is None or M > int(steps(rec)[-1]):
                    continue
            s, y = steps(rec), fn(rec)
            row = ""
            for d in offs:
                i = int(np.argmin(np.abs(s - (M + d))))
                row += "     -   " if abs(s[i] - (M + d)) > ck else f"{y[i]:9.4f}"
            print(f"   {arm:17s} {M:6d} {row}")

    # ---------------- 8. the trace ---------------- #
    print("\n" + "=" * W)
    print("8. THE DECISION TRACE — the scientific object")
    print("=" * W)
    for arm, rec in arms.items():
        tr = rec.get("trace") or []
        if not tr or tr[0].get("rule") != "bandit":
            continue
        print(f"\n   --- {arm}  (reads `{rec['sched']['policy']['read']}`, mode "
              f"`{rec['sched']['policy'].get('mode', 'delta')}`) ---")
        print(f"   {'t':>4s} {'step':>6s} {'act':>9s} {'why':>13s} {'read e':>9s} "
              f"{'reward':>9s} {'base':>9s} {'Q keep':>9s} {'Q coll':>9s} {'spend':>6s}")
        for d in tr[::a.trace_every]:
            fmt = lambda x: "     -   " if x is None else f"{x:9.4f}"
            print(f"   {d['t']:4d} {d['step']:6d} {d['action']:>9s} {d['why']:>13s} "
                  f"{fmt(d.get('e'))} {fmt(d.get('reward'))} {fmt(d.get('baseline'))} "
                  f"{fmt(d.get('q_keep'))} {fmt(d.get('q_collapse'))} "
                  f"{d.get('spend', 0):6d}")

    # ---------------- 9. the d5 trajectory (Rung A-1/2) ---------------- #
    print("\n" + "=" * W)
    print("9. NEXT-LEVEL YIELD — the d5 trajectory, key anchor, neutral wall")
    print("=" * W)
    print("   d5 is the latent one level ABOVE the keyed one. fwlm ran full probes only")
    print("   at 8000 and 20000, so the shape between them has never been measured; the")
    print("   science probe grid here samples it for every arm, and the yield policies")
    print("   sample it again at their own trial boundaries (seed-pinned probe init).")
    gridd = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 19000]
    print(f"\n   {'arm':17s}" + "".join(f"{g:>8d}" for g in gridd))
    for lab, rec in list(arms.items()) + [(f"[fwlm1] {k}", v) for k, v in
                                          bracket_ref["fwlm1"][1].items()] + \
            [(f"[fwlm0] {k}", v) for k, v in bracket_ref["fwlm0"][1].items()]:
        row = ""
        for g in gridd:
            pc = probe_d4(rec, g, "none", tol=cfg["ckpt_every"] * 2)
            r5 = None
            cands = [r for r in rec["log"] if "levels" in r
                     and "none" in r["levels"] and abs(r["step"] - g) <= cfg["ckpt_every"] * 2]
            if cands:
                r5 = min(cands, key=lambda z: abs(z["step"] - g))["levels"]["none"]["key"].get("d5")
            row += "     -  " if r5 is None else f"{r5:8.3f}"
        print(f"   {lab:17s}{row}")
    print("\n   the yield policies' OWN reads (d5 / d4 at each trial boundary):")
    for arm, rec in arms.items():
        pts = [(d["step"], d["yield_d5"], d["yield_d4"]) for d in (rec.get("trace") or [])
               if d.get("yield_d5") is not None]
        if not pts:
            continue
        print(f"   {arm}:")
        print("     " + "  ".join(f"s{s}:{d5:.3f}/{d4:.3f}" for s, d5, d4 in pts))

    if a.figures:
        figures(a.tag, setup, arms, refs, cfg, bracket_ref)
    print("\n" + "=" * W)
    print(f"FIDELITY GATE: {'PASS' if gate_ok else 'FAIL'}   "
          f"(rows: {gate_rows})")
    print("=" * W)


# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, refs, cfg, bracket_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    MS, ck = cfg["max_steps"], cfg["ckpt_every"]
    rots = setup["rotations"].get("wall", [])
    names = list(arms)
    cols = {n: c for n, c in zip(names, plt.cm.tab10(np.linspace(0, 0.9, 10)))}

    def cond_spans(rec):
        s = steps(rec)
        merged = np.array([bool(r["merged"]) for r in rec["log"]])
        spans, i = [], 0
        while i < len(s):
            if merged[i]:
                j = i
                while j + 1 < len(s) and merged[j + 1]:
                    j += 1
                spans.append((s[i], s[j] + (s[j + 1] - s[j] if j + 1 < len(s) else ck)))
                i = j + 1
            else:
                i += 1
        return spans

    # --- fig1: the decision raster ---
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(names) + 2.4))
    for k, n in enumerate(names):
        y = len(names) - 1 - k
        ax.add_patch(plt.Rectangle((0, y - 0.32), MS, 0.64, fc="#dfe7ef", ec="none"))
        for lo, hi in cond_spans(arms[n]):
            ax.add_patch(plt.Rectangle((lo, y - 0.32), hi - lo, 0.64,
                                       fc=cols[n], ec="none", alpha=0.95))
        sm = arms[n].get("summary") or {}
        if sm.get("commit_step") is not None:
            ax.plot([sm["commit_step"]], [y], "k*", ms=13, zorder=5)
        for st in (sm.get("bailout_steps") or []):
            ax.plot([st], [y], "kv", ms=6, zorder=5)
    for r in rots:
        ax.axvline(r, color="0.45", lw=0.8, ls=":", zorder=0)
    for b in (1000, 8000, 13000):
        ax.axvline(b, color="crimson", lw=0.9, ls="--", alpha=0.55, zorder=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlim(0, MS); ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("training step")
    ax.set_title("Rung A — the condition each outer loop chose\n"
                 "filled = `w` COLLAPSED to the neutral filler; pale = `w` present.  "
                 "★ commit   ▾ bail-out\n"
                 "dotted = rotations; dashed red = fwlm1's supplied merge steps",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_decisions.png"), dpi=150)
    plt.close(fig)

    # --- fig2: the two currencies ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for n in names:
        rec = arms[n]
        s = steps(rec)
        axes[0].plot(s, cons_idx(rec), color=cols[n], lw=1.2, label=n)
        axes[1].plot(s, path_excess(rec), color=cols[n], lw=1.2, label=n)
    _, d1 = bracket_ref.get("fwlm1", (None, {}))
    for darm, ls in (("merge_8000", "--"), ("no_wall", ":")):
        if darm in d1 and darm not in arms:
            rec = d1[darm]
            axes[0].plot(steps(rec), cons_idx(rec), "0.4", lw=1.0, ls=ls,
                         label=f"[fwlm1] {darm}")
            axes[1].plot(steps(rec), series(rec, "excess.none.idx"), "0.4", lw=1.0,
                         ls=ls, label=f"[fwlm1] {darm}")
    axes[0].axhline(refs["gate0"]["bayes_indexed_span"], color="k", lw=0.8, ls="-.",
                    label="exact Bayes (indexed span)")
    for ax in axes:
        for r in rots:
            ax.axvline(r, color="0.6", lw=0.7, ls=":", zorder=0)
    axes[0].set_ylabel("task NLL, indexed span\n(consumed condition)")
    axes[1].set_ylabel("pathway readout: excess over Bayes\n(neutral wall; lower = whole)")
    axes[1].set_xlabel("training step")
    axes[0].set_title("The two currencies. `outer_task` reads the top panel; "
                      "`outer_path`/`_lp` read the bottom.", fontsize=10)
    axes[0].legend(fontsize=7, ncol=3); axes[1].legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_currencies.png"), dpi=150)
    plt.close(fig)

    # --- fig3: the value estimates ---
    pol = [n for n in names if (arms[n].get("trace") or [{}])[0].get("rule")
           in ("bandit", "paired")]
    if pol:
        fig, axes = plt.subplots(len(pol), 1, figsize=(13, 2.6 * len(pol)), sharex=True,
                                 squeeze=False)
        for k, n in enumerate(pol):
            ax = axes[k][0]
            tr = arms[n]["trace"]
            st = [d["step"] for d in tr]
            nan = lambda k: [np.nan if d.get(k) is None else d[k] for d in tr]
            if tr[0].get("rule") == "paired":
                ax.plot(st, nan("V"), color="#d62728", lw=1.4,
                        label="V  (paired contrast: >0 favours collapse)")
                dd = [(d["step"], d["contrast"]) for d in tr
                      if d.get("contrast") is not None]
                ax.plot([x for x, _ in dd], [y for _, y in dd], "ko", ms=5,
                        label="D (one ABBA trial)")
                tol = tr[0].get("v_tol") or 0.0
                ax.axhspan(-tol, tol, color="0.6", alpha=0.30, lw=0,
                           label=f"dead zone +/-{tol}")
            else:
                ax.plot(st, nan("q_keep"), color="#1f77b4", lw=1.3, label="Q[keep]")
                ax.plot(st, nan("q_collapse"), color="#d62728", lw=1.3,
                        label="Q[collapse]")
            ax.axhline(0, color="0.7", lw=0.7)
            for d in tr:
                if d["action"] == "collapse":
                    ax.axvspan(d["step"], d["step"] + cfg["decide_every"],
                               color="#d62728",
                               alpha=0.22 if d["why"] == "greedy" else 0.09, lw=0)  # noqa
                if d["why"] in ("forced_trial", "trial"):
                    ax.plot([d["step"]], [0], "k|", ms=7)
            for r in rots:
                ax.axvline(r, color="0.6", lw=0.7, ls=":", zorder=0)
            ax.set_ylabel(f"{n}\ncontrast", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
        axes[-1][0].set_xlabel("training step")
        axes[0][0].set_title("The value estimates that produced the decisions. Shaded = "
                             "the interval was run COLLAPSED\n(dark = chosen greedily, "
                             "pale = warm-up or scheduled trial); | = forced trial",
                             fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_values.png"), dpi=150)
        plt.close(fig)

    # --- fig4: terminal wholeness vs lifetime price ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ceil = refs["ceil"]["key_none"]["d4"]
    lab, d4v, pfv, lfv = [], [], [], []
    for n in names:
        rec = arms[n]
        t = int(steps(rec)[-1])
        d = probe_d4(rec, t, "rand")
        lab.append(n); d4v.append(np.nan if d is None else d[0])
        pfv.append(float(path_excess(rec)[-1]))
        lfv.append(float(np.mean(cons_idx(rec))))
    x = np.arange(len(lab))
    axes[0].bar(x, d4v, color=[cols[n] for n in lab])
    axes[0].axhline(ceil, color="k", ls="--", lw=1, label=f"exact ceiling {ceil:.3f}")
    axes[0].set_title("terminal pathway wholeness\nkey-anchor d4, wall randomised")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, pfv, color=[cols[n] for n in lab])
    axes[1].set_title("terminal probe-free twin\nexcess over Bayes, neutral (lower=whole)")
    axes[2].bar(x, lfv, color=[cols[n] for n in lab])
    axes[2].axhline(refs["gate0"]["bayes_indexed_span"], color="k", ls="-.", lw=1)
    axes[2].set_title("lifetime task integral\n(mean consumed indexed-span NLL)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_terminal.png"), dpi=150)
    plt.close(fig)
    # --- fig5: the next-level yield trajectory ---
    fig, ax = plt.subplots(figsize=(13, 5))
    for n in names:
        rec = arms[n]
        pts = [(r["step"], r["levels"]["none"]["key"].get("d5")) for r in rec["log"]
               if "levels" in r and "none" in r["levels"]
               and r["levels"]["none"]["key"].get("d5") is not None]
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], "-o", ms=3,
                    color=cols[n], lw=1.2, label=n)
        tr = [(d["step"], d["yield_d5"]) for d in (rec.get("trace") or [])
              if d.get("yield_d5") is not None]
        if tr:
            ax.plot([x for x, _ in tr], [y for _, y in tr], "s", ms=5, mfc="none",
                    color=cols[n], label=f"{n} (policy reads)")
    for dt, dn, st in (("fwlm1", "no_wall", "--"), ("fwlm0", "wall", ":")):
        d = bracket_ref.get(dt, (None, {}))[1].get(dn)
        if d is not None:
            pts = [(r["step"], r["levels"]["none"]["key"].get("d5")) for r in d["log"]
                   if "levels" in r and "none" in r["levels"]
                   and r["levels"]["none"]["key"].get("d5") is not None]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], st, color="0.35",
                        lw=1.1, label=f"[{dt}] {dn}")
    ax.set_xlabel("training step")
    ax.set_ylabel("d5 (key anchor, neutral wall)")
    ax.set_title("Rung A-1/2 — the next-level yield. d5 is the latent ONE LEVEL ABOVE\n"
                 "the keyed one; squares are the yield policies' own priced reads.",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_yield_d5.png"), dpi=150)
    plt.close(fig)
    print(f"\n[figures] -> {out}/fig1_decisions.png fig2_currencies.png "
          f"fig3_values.png fig4_terminal.png fig5_yield_d5.png")


if __name__ == "__main__":
    main()
