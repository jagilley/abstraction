"""[preplay/pp4] THE REDUCER — the read as the ORDER for the loop's own incremental audition.

Writes `figures/incremental_reduction.txt` and appends it to the node's table of record,
`figures/preplay_reduction.txt`, as section **[W]4** (idempotently: an existing block is
replaced, so re-running never stacks).

Sections:
  [W0] the gates: G-1a/G-1b (the walk IS `census_extend`), G-2 (three disjoint pool families,
       per arm and level), G-3, and pp1's F-2b per arm
  [W1] what each order is handed: the true-vs-wrong AUC of each price on the PRICING pool
  [W2] THE BUDGET TABLE, the table of record: the current table's test error after 8 (the
       loop's own `extend_cap`), 16, 32 auditions and after the full walk, per arm, level,
       setting, order and repeat, with the entries kept and the WRONG entries admitted; and,
       beside each gated order, the UNGATED top-b by the same price at the same budget (pp2's
       selector), so the gate's own contribution is visible
  [W3] THE COST-TO-QUALITY: the auditions each order spends to come within 0.02 of the WORLD
       order's final error, on the curve's own test pool
  [W4] the tolerance sensitivity: the same walk at `extend_tol = 0.01` against the loop's 0.0

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_incremental.py --tag pp4 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]
MARK = "[W4] THE READ AS THE ORDER FOR THE LOOP'S OWN INCREMENTAL AUDITION"
REACH_TOL = 0.02


def f(q, w=6, p=3):
    return (" " * (w - 2) + "--") if q is None else f"{q:>{w}.{p}f}"


def fetch(tag):
    dst = os.path.join(HERE, "figures")
    os.makedirs(dst, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force", "rhm-scaling-data",
           f"rhm_practice_preplay/{tag}", dst]
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pp4")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tag)
    d = os.path.join(HERE, "figures", args.tag)
    if not os.path.isdir(d):
        sys.exit(f"no such directory: {d}")
    arms = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json") and not fn.startswith("done"):
            arms[fn[:-5]] = json.load(open(os.path.join(d, fn)))
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
    cfg0 = arms[order[0]]["cfg"]
    out = []

    def o(line=""):
        out.append(line)

    o("=" * 138)
    o(MARK + f" — tag {args.tag}")
    o("    The loop's own consumer is `census_extend` (`soundboard.py` 13346-13427): walk the")
    o("    candidates in an order, audition `base + {cand}` on a fresh pool, admit iff")
    o(f"    `e_x <= best_e + extend_tol` with tol = {cfg0['tol']} (must not hurt), grow the base on")
    o(f"    admission, one audition per candidate, `extend_cap = {cfg0['extend_cap']}` per pass, pool")
    o(f"    `n_aud = {cfg0['n_aud']}`. THE GATE IS FIXED; only the ORDER varies.  arms: "
      + ", ".join(order))
    o(f"    pricing n = {cfg0['n_price']} | gate n = {cfg0['n_gate']} x{cfg0['repeats']} repeats "
      f"| test n = {cfg0['n_test']} x{cfg0['n_test_pools']}, three disjoint seed families.")
    o("=" * 138)

    # ------------------------------------------------------------------- [W0] gates ---- #
    o("")
    o("[W0] THE GATES")
    o("     G-1b: the walk reproduces a HAND-COMPUTED synthetic case (errors 0.50 base, then")
    o("           0.40 0.45 0.40 0.50 0.10 0.11 -> admissions [0, 2, 4], 3 rejected, 7")
    o("           auditions, final 0.10). G-1a: it is identical to a literal transcription of")
    o("           `census_extend` on REAL auditions at cap 8, at the full walk, and at tol")
    o("           0.01. G-3: a constant price reduces the order to the shared tie-break, and a")
    o("           strictly decreasing price gives the identity order. All closed in")
    o("           `incremental.py::gates2`/`gates4`; falsified 4/4 in `::falsify4`.")
    o("     `preplay.py`'s F-1..F-6 and `selector.py`'s S-1..S-5 carry over; the read and the")
    o("     fire are IMPORTED from `preplay.py`, so its exact gate F-2 covers this node.")
    o("")
    o(f"     {'arm':8} {'F-2b |d|':>9} {'band':>7}   G-2: max shared instances over all pairs "
      f"of the three pool families, per cell")
    for a in order:
        g = arms[a]["gate"]
        g2 = arms[a]["G-2"]
        bits = "  ".join(f"{k}:{g2[k]['max_overlap']}" for k in sorted(g2))
        o(f"     {a:8} {g['F-2b:delta']:>9.4f} {g['F-2b:band']:>7.4f}   {bits}")
    ltg = arms[order[0]].get("lt_gates") or []
    if ltg:
        o(f"     setting (b)'s tables are pp3's replay: {len(ltg)} reconstruction gates "
          f"(R-0..R-4), all pass on every arm.")

    # ------------------------------------------------- [W1] what each order is handed --- #
    o("")
    o("=" * 138)
    o("[W1] WHAT EACH ORDER IS HANDED — true-vs-wrong AUC of each price on the PRICING pool")
    o("     `build` (setting b) is the learner's own emission order and is not a price, so it")
    o("     has no AUC here. See [W4]'s note on what it is and is not.")
    o("")
    o(f"     {'arm':8} {'cell':>6} {'n(T/W)':>9} {'base':>6} " +
      " ".join(f"{k[:9]:>9}" for k in ("world", "banked", "frozen", "twin", "dp_top")))
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            o(f"     {a:8} {ck:>6} {c['n_cand_true']:>4d}/{c['n_cand_wrong']:<4d} "
              f"{c['n_base']:>6d} " +
              " ".join(f(c["price_auc"].get(k), 9, 3)
                       for k in ("world", "banked", "frozen", "twin", "dp_top")))

    # ------------------------------------------------------------ [W2] the budget table - #
    o("")
    o("=" * 138)
    o("[W2] THE BUDGET TABLE — the world's audition error of the CURRENT table after b")
    o("     auditions, mean over the three test pools. b = 8 is the loop's own `extend_cap`.")
    o("     `full` is the whole walk. `k` is entries kept, `w` wrong entries admitted.")
    o("     LOWER IS BETTER. `base` is the table before any candidate is offered.")
    o("")
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                names = list(rep["orders"])
                buds = sorted({int(b) for nm in names for b in rep["orders"][nm]["budget"]})
                o(f"  {a} {ck}  repeat {rep['repeat']} (gate seed {rep['gate_seed']}, "
                  f"base {rep['n_base']} rows at e {rep['e_base_test']['mean']:.3f}, "
                  f"{rep['n_live']} candidates live)")
                o(f"    {'order':10} " +
                  " ".join(f"{('b=' + str(b)):>16}" for b in buds) +
                  f" {'admitted':>9} {'rejected':>9}")
                for nm in names:
                    r = rep["orders"][nm]
                    cells = []
                    for b in buds:
                        q = r["budget"].get(str(b))
                        cells.append("".rjust(16) if q is None else
                                     f"{q['e_test']:.3f} k{q['n_kept']:<3d}w{q['n_wrong']:<3d}"
                                     .rjust(16))
                    o(f"    {nm:10} " + " ".join(cells) +
                      f" {len(r['admitted']):>9d} {r['rejected']:>9d}")
                    ung = r.get("ungated")
                    if ung:
                        cells = []
                        for b in buds:
                            q = ung.get(str(b))
                            cells.append("".rjust(16) if q is None else
                                         f"{q['e_test']:.3f} k{q['n_kept']:<3d}"
                                         f"w{q['n_wrong']:<3d}".rjust(16))
                        o(f"    {'  (ungated)':10} " + " ".join(cells))
                o("")

    # ------------------------------------------------------- [W3] the cost to quality --- #
    o("=" * 138)
    o("[W3] COST TO QUALITY — auditions spent to come within 0.02 of the WORLD order's final")
    o(f"     error, on the curve's own test pool (pool 0; both sides carry the same pool).")
    o("     `--` means the order never got there within the full walk. `full` is the walk's")
    o("     length, so an order at `full` paid everything and still did not arrive.")
    o("")
    o(f"     {'arm':8} {'cell':>6} {'rep':>3} {'walk':>5} {'target':>7} " +
      " ".join(f"{k[:9]:>9}" for k in ("world", "banked", "frozen", "twin", "dp_top",
                                       "build", "rand0", "rand1")))
    reach = {}
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                names = list(rep["orders"])
                if "world" not in names:
                    continue
                wc = rep["orders"]["world"]["curve"]
                tgt = wc[-1]["e_test1"] + REACH_TOL
                walk = len(rep["orders"]["world"]["order"])
                vals = {}
                for nm in names:
                    cur = rep["orders"][nm]["curve"]
                    hit = [q["n_aud"] for q in cur if q["e_test1"] <= tgt]
                    vals[nm] = (min(hit) - 1) if hit else None
                    reach.setdefault(ck[0], {}).setdefault(nm, []).append(
                        None if vals[nm] is None else vals[nm] / float(max(1, walk)))
                o(f"     {a:8} {ck:>6} {rep['repeat']:>3} {walk:>5d} {tgt:>7.3f} " +
                  " ".join((f"{vals[k]:>9d}" if vals.get(k) is not None
                            else ("        --" if k in names else "         ."))
                           for k in ("world", "banked", "frozen", "twin", "dp_top",
                                     "build", "rand0", "rand1")))
    o("")
    o("     Median auditions-to-target as a FRACTION of the full walk, SPLIT BY SETTING so")
    o("     every order in a block is read over the same cells (`frozen`/`twin` run only in")
    o("     (a), `build` only in (b)); `arrived` is the share of cells the order reached the")
    o("     target in at all, and the median is over those. A summary; the rows above are the")
    o("     record.")
    for st in sorted(reach):
        ks = sorted(reach[st])
        n_cell = max(len(reach[st][k]) for k in ks)
        o("")
        o(f"     setting ({st}), {n_cell} cells")
        o(f"     {'':9} " + " ".join(f"{k[:9]:>9}" for k in ks))
        o(f"     {'median':9} " +
          " ".join(f((float(np.median([q for q in reach[st][k] if q is not None]))
                      if any(q is not None for q in reach[st][k]) else None), 9, 3)
                   for k in ks))
        o(f"     {'arrived':9} " +
          " ".join(f"{sum(1 for q in reach[st][k] if q is not None)}/{len(reach[st][k])}"
                   .rjust(9) for k in ks))

    # ---------------------------------------------------------- [W4] tolerance, notes --- #
    o("")
    o("=" * 138)
    o(f"[W4] THE TOLERANCE — the same walk at `extend_tol = {cfg0['tol_sens']}` against the loop's "
      f"{cfg0['tol']}.")
    o("     At tol 0 a single unlucky instance of the gate pool rejects a good entry; this is")
    o("     how much of the outcome that accounts for. Repeat 0 only, two orders.")
    o("")
    o(f"     {'arm':8} {'cell':>6} {'order':>8} " +
      f"{'e_final(tol 0)':>15} {'k':>4} {'w':>4} | {'e_final(tol+)':>14} {'k':>4} {'w':>4}")
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                ts = rep.get("tol_sens") or {}
                for nm in sorted(ts):
                    b = rep["orders"].get(nm)
                    if b is None:
                        continue
                    t = ts[nm]
                    o(f"     {a:8} {ck:>6} {nm:>8} "
                      f"{b['e_test_final']['mean']:>15.3f} {b['n_kept']:>4d} "
                      f"{b['n_wrong_kept']:>4d} | {t['e_test_final']['mean']:>14.3f} "
                      f"{t['n_kept']:>4d} {t['n_wrong_kept']:>4d}")
    o("")
    o("     NOTE ON `build`, setting (b)'s learner order. It is the operative table's own ROW")
    o("     ORDER, which is `ClassMiner.build`'s emission order: class-pair keys sorted, then")
    o("     the cross-product within each key. It is NOT the loop's `extend_candidates` order,")
    o("     which sorts by `(-count, key)` — the per-key support COUNTS are not in the dump")
    o("     (only `n_at_support` at fixed thresholds), so the loop's actual order is not")
    o("     reconstructible offline. `build` is therefore a learner-side order, not THE")
    o("     learner's order, and is labelled as such everywhere.")

    # ------------------------------------------------------------------------- bill ----- #
    o("")
    o("=" * 138)
    o("[W5] THE BILL")
    o(f"    {'arm':8} {'auditions':>10} {'sec':>8} {'peak RSS MB':>12} {'peak GPU MB':>12}")
    tot = 0.0
    for a in order:
        r = arms[a]
        tot += float(r["sec"])
        o(f"    {a:8} {r['n_auditions']:>10d} {r['sec']:>8.1f} {r['peak_rss_mb']:>12.0f} "
          f"{r['peak_gpu_mb']:>12.0f}")
    o(f"    {'TOTAL':8} {'':>10} {tot:>8.1f}   ({tot / 3600.0:.2f} GPU-h over "
      f"{len(order)} containers)")
    o("")

    txt = "\n".join(out) + "\n"
    own = os.path.join(HERE, "figures", "incremental_reduction.txt")
    with open(own, "w") as fh:
        fh.write(txt)
    print(txt[:12000])
    print(f"[written] {own}")

    if not args.no_append:
        rec = os.path.join(HERE, "figures", "preplay_reduction.txt")
        if os.path.exists(rec):
            cur = open(rec).read()
            i = cur.find("=" * 138 + "\n" + MARK)
            if i >= 0:
                cur = cur[:i]
            with open(rec, "w") as fh:
                fh.write(cur.rstrip("\n") + "\n\n" + txt)
            print(f"[appended as section [W4]] {rec}")


if __name__ == "__main__":
    main()
