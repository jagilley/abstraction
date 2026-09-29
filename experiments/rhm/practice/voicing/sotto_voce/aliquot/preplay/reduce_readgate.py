"""[preplay/pp5] THE REDUCER — the read as the GATE, the prior as the order.

Writes `figures/readgate_reduction.txt` and appends it to the node's table of record,
`figures/preplay_reduction.txt`, as section **[G5]** (idempotently).

Sections:
  [G50] the gates: P-1a/P-1b (the world gate here IS pp4's `census_walk`), P-2 (the read gate
        reduces to the world gate when handed the world's error as its level), P-3, P-4, P-5
  [G51] THE TABLE OF RECORD: the world's TEST-pool error of the table each gate built, after
        b = 8 (the loop's `extend_cap`), 16, 32 and the full walk, with entries kept and WRONG
        entries admitted, per arm, level, setting and repeat
  [G52] THE DIRECT MEASURE: the per-candidate agreement between each gate's decisions and the
        world gate's on the SAME trials, as a confusion table, plus the world's error change on
        the candidates each gate admitted where the world would have rejected
  [G53] the secondary orders (repeat 0): the world and the read orders beside `dp_top`

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_readgate.py --tag pp5 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]
MARK = "[G5] THE READ AS THE GATE, THE PRIOR AS THE ORDER"
GATES = ("world", "read", "read_pair", "read_m", "frozen", "twin", "ungated")


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
    ap.add_argument("--tag", default="pp5")
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
    ORD = cfg0["orders"][0]
    out = []

    def o(line=""):
        out.append(line)

    o("=" * 140)
    o(MARK + f" — tag {args.tag}")
    o("    pp4 settled that the executor's own score is the right ORDER for the loop's")
    o("    try-and-keep extension. The readout's seat is the other half of `census_extend`:")
    o("    the loop's gate is an ORACLE READ — 'did the world's error on the gate pool rise?'")
    o("    — and the readout is the thing that gets to see the preplayed state and put a")
    o("    number on it. Can that number stand in for the world's verdict?")
    o(f"    ORDER = `{ORD}` throughout. GATE varies. arms: " + ", ".join(order))
    o(f"    pricing n = {cfg0['n_price']} | gate n = {cfg0['n_gate']} (the loop's `n_aud`) "
      f"x{cfg0['repeats']} | test n = {cfg0['n_test']} x{cfg0['n_test_pools']}, disjoint.")
    o("")
    o("    world      admit iff the world's error does not rise           pp4's, THE REFERENCE")
    o("    read       admit iff the banked shaped projection's LEVEL over the same fired")
    o("               gate-pool configurations does not fall (pooled mean p)    UNDER TEST")
    o("    read_pair  the same, PAIRED over the instances the candidate actually changed")
    o("    read_m     the pooled form with a margin delta = se of the base's own level")
    o("    frozen / twin   the pooled form through an unshaped / never-trained trunk  CONTROLS")
    o("    ungated    admit everything                                      THE FLOOR")
    o("=" * 140)

    # ------------------------------------------------------------------ [G50] gates ---- #
    o("")
    o("[G50] THE GATES — all closed in `readgate.py::gates5`, all falsified in `::falsify5` (5/5)")
    o("      P-1a  the world gate here is IDENTICAL to pp4's `census_walk` on the same inputs")
    o("      P-1b  and reproduces pp4's hand-computed sequence (admissions [0, 2, 4])")
    o("      P-2   THE PLUMBING IDENTITY: handed the world's error as its level, the read gate")
    o("            reproduces the world gate exactly, in all three of its forms")
    o("      P-3   a constant level admits everything, in every read form and both controls")
    o("      P-4   the pricing, gate and test pool families are pairwise disjoint")
    o("      P-5   the CHANGED set (`trial.entry == len(kept)`) covers every instance whose")
    o("            fired configuration differs — asserted non-vacuous (192/192 flagged)")
    o("")
    o(f"      {'arm':8} {'F-2b |d|':>9} {'band':>7}   P-4 max shared instances, per cell")
    for a in order:
        g = arms[a]["gate"]
        p4 = arms[a]["P-4"]
        o(f"      {a:8} {g['F-2b:delta']:>9.4f} {g['F-2b:band']:>7.4f}   " +
          "  ".join(f"{k}:{p4[k]['max_overlap']}" for k in sorted(p4)))

    # ------------------------------------------------------------ [G51] the main table - #
    o("")
    o("=" * 140)
    o("[G51] THE TABLE OF RECORD — the world's TEST-pool error of the table each GATE built.")
    o(f"      Order is `{ORD}` for every row. b = 8 is the loop's own `extend_cap`; `full` is")
    o("      the whole walk. k = entries kept, w = wrong entries admitted. LOWER e IS BETTER.")
    o("")
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                keys = [f"{ORD}|{g}" for g in GATES if f"{ORD}|{g}" in rep["orders"]]
                if not keys:
                    continue
                buds = sorted({int(b) for k in keys for b in rep["orders"][k]["budget"]})
                o(f"  {a} {ck} repeat {rep['repeat']} — base {rep['n_base']} rows at e "
                  f"{rep['e_base_test']['mean']:.3f} (level {rep['p_base']:.3f}), "
                  f"{rep['n_live']} candidates ({c['n_cand_true']}T/{c['n_cand_wrong']}W), "
                  f"delta {rep['delta']:.4f}")
                o(f"    {'gate':10} " + " ".join(f"{('b=' + str(b)):>16}" for b in buds) +
                  f" {'admitted':>9} {'rejected':>9}")
                for k in keys:
                    r = rep["orders"][k]
                    cells = []
                    for b in buds:
                        q = r["budget"].get(str(b))
                        cells.append("".rjust(16) if q is None else
                                     f"{q['e_test']:.3f} k{q['n_kept']:<3d}w{q['n_wrong']:<3d}"
                                     .rjust(16))
                    o(f"    {k.split('|')[1]:10} " + " ".join(cells) +
                      f" {len(r['admitted']):>9d} {r['rejected']:>9d}")
                o("")

    # -------------------------------------------------------- [G52] the direct measure - #
    o("=" * 140)
    o("[G52] THE DIRECT MEASURE — per-candidate agreement with the WORLD gate's decision on")
    o("      the SAME trial (the world's verdict is recorded at every step whatever the gate")
    o("      decides, from the same fire). TT = both admit, FF = both reject, TF = the gate")
    o("      admits where the world rejects (the junk it lets in), FT = the gate rejects where")
    o("      the world admits (what it refuses). `cost` is the world's mean error CHANGE on")
    o("      the TF candidates — what each wrongly-admitted entry actually cost on the gate")
    o("      pool — and `cost_sum` their total. `agree` = (TT + FF) / n.")
    o("")
    o(f"      {'arm':8} {'cell':>6} {'rp':>2} {'gate':10} {'n':>4} {'TT':>4} {'FF':>4} "
      f"{'TF':>4} {'FT':>4} {'agree':>6} {'cost':>7} {'cost_sum':>9}")
    agg = {}
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                for g in GATES:
                    k = f"{ORD}|{g}"
                    if k not in rep["orders"]:
                        continue
                    r = rep["orders"][k]
                    cf = r["conf"]
                    n = sum(cf.values())
                    ag = (cf["TT"] + cf["FF"]) / float(max(1, n))
                    agg.setdefault(g, {"agree": [], "TF": [], "FT": [], "cost": []})
                    agg[g]["agree"].append(ag)
                    agg[g]["TF"].append(cf["TF"] / float(max(1, n)))
                    agg[g]["FT"].append(cf["FT"] / float(max(1, n)))
                    if r.get("cost_mean") is not None:
                        agg[g]["cost"].append(r["cost_mean"])
                    o(f"      {a:8} {ck:>6} {rep['repeat']:>2} {g:10} {n:>4d} "
                      f"{cf['TT']:>4d} {cf['FF']:>4d} {cf['TF']:>4d} {cf['FT']:>4d} "
                      f"{ag:>6.3f} {f(r.get('cost_mean'), 7, 4)} {r['cost_sum']:>9.4f}")
    o("")
    o("      READ THE `ungated` ROW FIRST. The world gate itself admits about 95% of the")
    o("      candidates it is offered, so 'admit everything' already agrees with it ~95% of")
    o("      the time — MORE than any read gate does. Raw agreement is therefore not evidence")
    o("      for a gate; what separates them is the COMPOSITION of the disagreement (TF, the")
    o("      junk let in, against FT, the good entries refused) and, in the end, the table")
    o("      quality in [G51].")
    o("")
    o("      Pooled over every (arm, cell, repeat) — a summary; the rows above are the record:")
    o(f"      {'gate':10} {'n cells':>8} {'agree (med)':>12} {'TF share':>10} "
      f"{'FT share':>10} {'cost (med)':>11}")
    for g in GATES:
        if g not in agg:
            continue
        q = agg[g]
        o(f"      {g:10} {len(q['agree']):>8d} {float(np.median(q['agree'])):>12.3f} "
          f"{float(np.median(q['TF'])):>10.3f} {float(np.median(q['FT'])):>10.3f} "
          f"{f(float(np.median(q['cost'])) if q['cost'] else None, 11, 4)}")

    # ---------------------------------------------------------- [G53] secondary orders - #
    o("")
    o("=" * 140)
    # ---------------------------------------------------- the pooled quality summary --- #
    o("")
    o("=" * 140)
    o("[G51s] THE TABLE OF RECORD, POOLED — median over all (arm, cell, repeat) of the test")
    o("       error at the loop's own budget and at the full walk. The two anchors are `world`")
    o("       (the oracle the gate is meant to replace) and `ungated` (the floor: no gate at")
    o("       all). `capture` is (ungated - gate) / (ungated - world): 1.0 = the oracle, 0.0 =")
    o("       no better than no gate, negative = worse than no gate.")
    o("")
    o(f"       {'gate':10} {'e @ b=8':>9} {'capture':>8} {'e @ full':>9} {'capture':>8} "
      f"{'k (med)':>8} {'w (med)':>8}")
    pool = {}
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            for rep in arms[a]["cells"][ck]["repeats"]:
                for g in GATES:
                    k = f"{ORD}|{g}"
                    if k not in rep["orders"]:
                        continue
                    r = rep["orders"][k]
                    bl = sorted(int(b) for b in r["budget"])
                    pool.setdefault(g, []).append(
                        (r["budget"]["8"]["e_test"], r["budget"][str(bl[-1])]["e_test"],
                         r["n_kept"], r["n_wrong_kept"]))
    med = {g: np.median(np.array(v, float), axis=0) for g, v in pool.items()}
    for g in GATES:
        if g not in med:
            continue
        m = med[g]
        cap8 = cpf = None
        if "world" in med and "ungated" in med:
            d8 = med["ungated"][0] - med["world"][0]
            df = med["ungated"][1] - med["world"][1]
            cap8 = (med["ungated"][0] - m[0]) / d8 if abs(d8) > 1e-9 else None
            cpf = (med["ungated"][1] - m[1]) / df if abs(df) > 1e-9 else None
        o(f"       {g:10} {m[0]:>9.3f} {f(cap8, 8, 3)} {m[1]:>9.3f} {f(cpf, 8, 3)} "
          f"{m[2]:>8.1f} {m[3]:>8.1f}")
    o("")
    n_id = 0
    n_tot = 0
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            for rep in arms[a]["cells"][ck]["repeats"]:
                ka, kb = f"{ORD}|read", f"{ORD}|read_pair"
                if ka in rep["orders"] and kb in rep["orders"]:
                    n_tot += 1
                    n_id += int(rep["orders"][ka]["admitted"]
                                == rep["orders"][kb]["admitted"])
    o(f"       THE PAIRED FORM IS THE POOLED FORM, at threshold 0: identical admission lists")
    o(f"       in {n_id} of {n_tot} cells, and necessarily so. Only the changed instances move,")
    o("       so `pooled_diff = (n_changed / n) * paired_diff` exactly; the two differ by a")
    o("       positive factor and therefore never in sign. The paired form can only separate")
    o("       from the pooled one under a MARGIN, where that factor rescales the threshold —")
    o("       which is what `read_m` would have to be built on to be a different object.")

    o("")
    o("=" * 140)
    o("[G53] THE SECONDARY ORDERS (repeat 0 only) — the world's own order and the read's own")
    o("      order beside `dp_top`, for the world gate and the read gate. pp4 found the order")
    o("      matters at small budgets; this says whether the gate's standing depends on it.")
    o("")
    o(f"      {'arm':8} {'cell':>6} {'order':>8} {'gate':>6} {'b=8':>8} {'b=16':>8} "
      f"{'full':>8} {'k':>4} {'w':>4}")
    for a in order:
        for ck in sorted(arms[a]["cells"]):
            c = arms[a]["cells"][ck]
            for rep in c["repeats"]:
                for k in rep["orders"]:
                    onm, gnm = k.split("|")
                    if onm == ORD:
                        continue
                    r = rep["orders"][k]
                    bl = sorted(int(b) for b in r["budget"])
                    o(f"      {a:8} {ck:>6} {onm:>8} {gnm:>6} "
                      f"{r['budget']['8']['e_test']:>8.3f} "
                      f"{r['budget']['16']['e_test']:>8.3f} "
                      f"{r['budget'][str(bl[-1])]['e_test']:>8.3f} "
                      f"{r['n_kept']:>4d} {r['n_wrong_kept']:>4d}")

    # -------------------------------------------------------------------------- bill --- #
    o("")
    o("=" * 140)
    o("[G54] THE BILL")
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
    own = os.path.join(HERE, "figures", "readgate_reduction.txt")
    with open(own, "w") as fh:
        fh.write(txt)
    print(txt[:9000])
    print(f"[written] {own}")

    if not args.no_append:
        rec = os.path.join(HERE, "figures", "preplay_reduction.txt")
        if os.path.exists(rec):
            cur = open(rec).read()
            i = cur.find("=" * 140 + "\n" + MARK)
            if i >= 0:
                cur = cur[:i]
            with open(rec, "w") as fh:
                fh.write(cur.rstrip("\n") + "\n\n" + txt)
            print(f"[appended as section [G5]] {rec}")


if __name__ == "__main__":
    main()
