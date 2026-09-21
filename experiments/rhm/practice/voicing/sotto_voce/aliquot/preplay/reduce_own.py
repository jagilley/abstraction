"""[preplay/pp3] THE REDUCER — the learner's own rows, priced, and the read as their selector.

Reads the per-arm JSON `own.py` wrote (mirrored under `figures/pp3/`) and produces, per arm and
per level, with the seeds NEVER averaged:

  [G]  the gates: F-2b (the banked read inside the readout's own drift band), F-1 per level, the
       trunk fingerprints, and the step-1 reconstruction gates R-0..R-4 with a note on where
       R-3 was closed (locally -- `entry.json.gz` is not on the volume)
  [T]  THE LEARNER'S OWN OPERATIVE TABLES: rows, how many are TRUE, the distinct-flat precision
       the loop itself logged, the lower table they stand on, and the frozen committed size that
       sets k in step 3
  [H]  the readout's held-out AUC on the arm's OWN experience, so the reads below are read
       beside what they are on their home ground
  [P]  STEP 2, THE TABLE OF RECORD: over the learner's own rows, the AUC with which each read
       separates its TRUE rows from its FALSE ones, beside the world's own price; the rank
       correlation of each read's price against the world's over all rows and within the true
       rows; and each read's mean level on true and on false rows
  [S]  STEP 3: the world's audition error of the top-k rows by each selector on DISJOINT test
       pools, at k = the frozen committed size and at a smaller k, against the whole operative
       table and the true rows alone, with the number of FALSE rows each selector admits
  [C]  the bill

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_own.py --tag pp3 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]
RECORD = ("s0_sv", "s2_sv")
READS = ("banked", "shaped_refit", "frozen", "twin", "dp_top")
SEL = ("world", "banked", "shaped_refit", "frozen", "twin", "dp_top")
LEVELS = ("2", "3", "4")


def f(x, n=3, w=0):
    if x is None:
        return "--".rjust(w) if w else "--"
    t = f"{float(x):.{n}f}"
    return t.rjust(w) if w else t


def fs(x, n=3, w=0):
    """...with an explicit sign, for the margin columns."""
    if x is None:
        return "--".rjust(w) if w else "--"
    t = f"{float(x):+.{n}f}"
    return t.rjust(w) if w else t


def fetch(tag):
    dst = os.path.join(HERE, "figures")
    os.makedirs(dst, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force", "rhm-scaling-data",
           f"rhm_practice_preplay/{tag}", dst]
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def rand_stats(sel, k):
    """The uniform-random selector at matched count: mean and sd over its draws, each of which
    is itself a mean over the test pools."""
    es = []
    d = 0
    while f"random{d}|{k}" in sel:
        q = sel[f"random{d}|{k}"]
        if q.get("e_mean") is not None:
            es.append(q["e_mean"])
        d += 1
    if not es:
        return None, None, 0
    return float(np.mean(es)), float(np.std(es)), len(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pp3")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tag)
    d = os.path.join(HERE, "figures", args.tag)
    if not os.path.isdir(d):
        sys.exit(f"no such directory: {d}")
    arms = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith("_own.json"):
            arms[fn[:-len("_own.json")]] = json.load(open(os.path.join(d, fn)))
    order = [k for k in ARM_ORDER if k in arms] + [k for k in arms if k not in ARM_ORDER]
    if not order:
        sys.exit(f"no per-arm JSON in {d}")
    L = []

    def P(s=""):
        L.append(s)

    W = 118
    P("=" * W)
    P(f"[own] tag {args.tag} — the LEARNER'S OWN mined entries, priced, and the read as their")
    P(f"      selector.   arms: {', '.join(order)}   (arms of record: {', '.join(RECORD)})")
    P("=" * W)
    P()
    P("  The object is the arm's own OPERATIVE table at the last cycle, reconstructed offline")
    P("  from `results.json` (`learner_tables.py`): the rows its own miner built, over its own")
    P("  lower vocabulary, at its own at-support class-pair keys, capped by its own spell_cap.")
    P("  Its FALSE rows are the ones the learner itself produced -- not random pairs of true")
    P("  lower rows, which is what `preplay` (pp1) priced. L5 is NOT reconstructible: its last")
    P("  build has n_keys_built = 0 while the operative table holds 128 rows from an earlier")
    P("  build whose keys and picks are not in the dump. The FROZEN committed tables are not")
    P("  reconstructible either -- only their sizes and truth masks are banked.")
    P()

    # ---------------------------------------------------------------- [G] gates
    P("=" * W)
    P("[G] THE GATES")
    P("    F-1   the recording DP (`fire_rec`) reproduces `MC.apply_any` BIT FOR BIT, on the")
    P("          first fire of every level of every arm -- so the free-signal instrument")
    P("          (`dp_top`) cannot move a fired state. pp1's gate, re-run here.")
    P("    F-2b  the re-implemented read scored against the arm's BANKED hold AUC. It cannot")
    P("          match exactly: the loop trains the plant AFTER the readout's refresh in the")
    P("          same cycle, so `vo_heads.pt`'s core is one plant update newer than the core")
    P("          the logged AUC was measured through (pp1 NOTES defect 1). Band = 2x the")
    P("          readout's own |drift| p90.")
    P("    R-*   the step-1 reconstruction gates, asserted inside every paid container.")
    P("          R-3 (the per-row truth vector against the banked `operative_mask`, row for")
    P("          row) can only be closed LOCALLY: `entry.json.gz` is not on the volume. R-4")
    P("          carries that closure into the containers as a digest of the exact table R-3")
    P("          was closed on, and it is the gate that fires there.")
    P()
    P("    arm     hold AUC banked   re-read     |d|    band   F-1          R-1/R-1b/R-1c/R-2"
      "   R-4 (digest)  R-3 here")
    for k in order:
        a = arms[k]
        g = a["gate"]
        f1 = ",".join(f"L{q['level']}" for q in a["gate_f1"] if q["identical"])
        rg = a["rgates"]
        n_r13 = sum(1 for q in rg if q["gate"].startswith(("R-0", "R-1", "R-2")))
        ok13 = all(q["pass"] for q in rg if q["gate"].startswith(("R-0", "R-1", "R-2")))
        r4 = [q for q in rg if q["gate"].startswith("R-4")]
        P(f"    {k:<8}      {f(g['F-2b:banked'], 4)}    {f(g['F-2b:here'], 4)}  "
          f"{f(g['F-2b:delta'], 4)}  {f(g['F-2b:band'], 4)}   PASS {f1:<8} "
          f"{'PASS' if ok13 else 'FAIL'} ({n_r13}/{n_r13})        "
          f"{'PASS' if all(q['pass'] for q in r4) else 'FAIL'} ({len(r4)}/{len(r4)})     "
          f"{'ran' if a['r3_ran'] else 'not on volume'}")
    P()
    P("    arm         shaped     frozen (overtone)   twin (never trained)   trunk fingerprints")
    for k in order:
        fp = arms[k]["fingerprints"]
        P(f"    {k:<8} {fp['shaped']:>12.3f} {fp['frozen']:>18.3f} {fp['twin']:>22.3f}")
    P()

    # ---------------------------------------------------------------- [T] the tables
    P("=" * W)
    P("[T] THE LEARNER'S OWN OPERATIVE TABLES (reconstructed; L5 out of reach)")
    P("    `rows` is the operative table's size; `TRUE` counts rows whose level-1 spelling is a")
    P("    legal derivation; `prec` is `MC.grade_table`'s distinct-flat precision, the number")
    P("    the loop itself logs. `lower` is the reconstructed level-(l-1) table it stands on.")
    P("    `k_big` is the arm's own FROZEN COMMITTED size at that level -- the k step 3 uses.")
    P("    The pool seed is a function of the arm's SEED only, so the three arms of a seed are")
    P("    priced and auditioned on IDENTICAL pools: within a seed the arms are paired, and the")
    P("    oracle `e:world` at k = 1 is literally the same number across them.")
    P()
    P("    arm      L    cell     rows   TRUE  FALSE    prec   lower  k_big  k_small   "
      "boot succ   pool seed")
    for k in order:
        a = arms[k]
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            st, gr = c["stat"], (c.get("grade") or {})
            P(f"    {k:<8} {ell}  {c['era']:<6} {st['n_rows']:>6} {st['n_true']:>6} "
              f"{st['n_false']:>6}  {f(gr.get('precision'), 4):>6}  {c['n_lower']:>5} "
              f"{c['k_big']:>6} {c['k_small']:>8}   {f(st['boot_success'], 4):>9}   "
              f"{st['pool_seed']}")
    P()

    # ---------------------------------------------------------------- [H] home ground
    P("=" * W)
    P("[H] THE READOUT ON THE ARM'S OWN EXPERIENCE (held-out AUC on the bank's hold split)")
    P("    The three refits share one train/validation subsample -- pp1's own, at the same")
    P("    `FIT_SEED_BASE` -- so they differ only by the trunk, and the frozen and twin columns")
    P("    here are the same objects pp1 read.")
    P()
    P("    arm        banked  shaped refit   frozen     twin   lam(shaped) lam(frozen) lam(twin)")
    for k in order:
        a = arms[k]
        h, ft = a["hold_read"], a["fits"]
        P(f"    {k:<8} {f(h['banked'], 4):>9} {f(h['shaped_refit'], 4):>13} "
          f"{f(h['frozen'], 4):>8} {f(h['twin'], 4):>8} "
          f"{ft['shaped_refit']['lam']:>13.0f} {ft['frozen']['lam']:>11.0f} "
          f"{ft['twin']['lam']:>9.0f}")
    P()

    # ---------------------------------------------------------------- [P] step 2
    P("=" * W)
    P("[P] STEP 2 — THE TABLE OF RECORD. Each operative ROW alone as a one-row table over the")
    P("    learner's own lower table, fired through the arm's SHAPED executor on a fresh pool")
    P("    of the level's era cell (n_score = 256), world-graded, and read on the SAME fired")
    P("    configurations.")
    P()
    P("    sep    = AUC separating the learner's TRUE rows from its FALSE ones (0.5 = none).")
    P("    W(sep) = the world's own price doing the same job -- the instrument, not a read.")
    P("    On a ONE-ROW table the DP's `lse` and `marg` are identically zero by construction,")
    P("    so the executor's free-signal column is `dp_top` (pp1 decision 9).")
    P()
    P("    arm      L   n(T/F)     W(sep)   sep:banked sep:shaped_r  sep:frozen    sep:twin"
      "  sep:dp_top")
    for k in order:
        a = arms[k]
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            s_ = c["stat"]
            P(f"    {k:<8} {ell}  {s_['n_true']:>3}/{s_['n_false']:<5} "
              f"{f(s_['sep_world'], 3):>9}  {f(s_['sep_banked'], 3):>11} "
              f"{f(s_['sep_shaped_refit'], 3):>12} {f(s_['sep_frozen'], 3):>11} "
              f"{f(s_['sep_twin'], 3):>11} {f(s_['sep_dp_top'], 3):>11}")
    P()
    P("    rho = Spearman of the read's price against the WORLD's price, over the learner's own")
    P("          rows. `all` is over every row; `true` is within the TRUE rows alone (where the")
    P("          question is no longer true-vs-false but how well a legal row repairs).")
    P()
    P("    arm      L   rho:banked  rho:shaped_r  rho:frozen    rho:twin  rho:dp_top  |  "
      "rho_true: banked shaped_r  frozen    twin  dp_top")
    for k in order:
        a = arms[k]
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            s_ = c["stat"]
            P(f"    {k:<8} {ell}  {f(s_['rho_banked'], 3):>10} "
              f"{f(s_['rho_shaped_refit'], 3):>13} {f(s_['rho_frozen'], 3):>11} "
              f"{f(s_['rho_twin'], 3):>11} {f(s_['rho_dp_top'], 3):>11}  |  "
              f"{f(s_['rho_true_banked'], 3):>15} {f(s_['rho_true_shaped_refit'], 3):>8} "
              f"{f(s_['rho_true_frozen'], 3):>7} {f(s_['rho_true_twin'], 3):>7} "
              f"{f(s_['rho_true_dp_top'], 3):>7}")
    P()
    P("    The mean LEVEL of each read on the learner's TRUE rows and on its FALSE ones, with")
    P("    the world's beside them. A read can separate (sep > 0.5) while sitting at the same")
    P("    absolute level on both, and the loop's chooser reads the level, not the rank.")
    P()
    P("    arm      L   W:true  W:false |  banked T/F        shaped_r T/F      frozen T/F     "
      "   twin T/F          dp_top T/F")
    for k in order:
        a = arms[k]
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            s_ = c["stat"]
            cells = []
            for nm in READS:
                cells.append(f"{f(s_[f'mean_true_{nm}'], 3)}/{f(s_[f'mean_false_{nm}'], 3)}")
            P(f"    {k:<8} {ell}  {f(s_['mean_true_world'], 3):>6} "
              f"{f(s_['mean_false_world'], 3):>7} |  " +
              "  ".join(q.ljust(16) for q in cells))
    P()

    # ---------------------------------------------------------------- [Q] the margin
    P("=" * W)
    P("[Q] WHAT THE READ ADDS OVER WHAT WAS ALREADY THERE. pp1 carried forward that the")
    P("    executor's own prior already separated its CONSTRUCTED true from wrong entries at")
    P("    0.54-0.68, so the question is the MARGIN of the banked read over the executor's own")
    P("    `dp_top`, over the unshaped frozen trunk and over the never-trained twin -- not")
    P("    whether the read is above chance. Positive = the banked read separates better.")
    P("    `W-banked` is how far the banked read is from the WORLD's own price, the ceiling.")
    P()
    P("    arm      L   sep:banked  -dp_top   -frozen     -twin  |  W(sep)  W-banked")
    for k in order:
        a = arms[k]
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            s_ = c["stat"]
            b = s_["sep_banked"]
            def dm(nm):
                q = s_[f"sep_{nm}"]
                return None if (b is None or q is None) else b - q
            gap = (None if (b is None or s_["sep_world"] is None)
                   else s_["sep_world"] - b)
            P(f"    {k:<8} {ell}  {f(b, 3):>10} {fs(dm('dp_top'), 3, 8)} "
              f"{fs(dm('frozen'), 3, 9)} {fs(dm('twin'), 3, 9)}  |  "
              f"{f(s_['sep_world'], 3):>6} {fs(gap, 3, 9)}")
    P()

    # ---------------------------------------------------------------- [S] step 3
    P("=" * W)
    P("[S] STEP 3 — THE READ AS THE SELECTOR. The top-k rows of the operative table by each")
    P("    selector, as a table, auditioned by the WORLD on DISJOINT test pools (their own seed")
    P("    family), mean over the pools. `world` and `true_only` are ORACLE selectors and are")
    P("    the ceiling, not competitors. `all_rows` is the whole operative table -- what the")
    P("    executor actually ran on -- and `random` is a uniform draw at matched count, mean")
    P("    and sd over its draws.")
    P()
    for k in order:
        a = arms[k]
        n_test = a["cfg"]["n_test"]
        P(f"    --- {k}   ({n_test} test pools, n_score {a['cfg']['n_score']}, "
          f"{a['cfg']['n_rand']} random draws) ---")
        P("    L   k      e:world  e:banked e:shaped_r e:frozen  e:twin  e:dp_top  "
          "e:random(sd)   e:all_rows e:true_only")
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            sel = c["selectors"]
            for k_ in c.get("ks", [c["k_big"], c["k_small"]]):
                rm, rs, nd = rand_stats(sel, k_)
                tag = ("big" if k_ == c["k_big"]
                       else ("one" if k_ == 1 else "sml"))
                deg = "*" if k_ >= c["stat"]["n_rows"] else " "
                row = [f(sel.get(f'{nm}|{k_}', {}).get('e_mean'), 4, 8) for nm in SEL]
                P(f"    {ell} {k_:>4}{tag[0]}{deg}" + " ".join(row) +
                  f"  {f(rm, 4)}({f(rs, 3)})   "
                  f"{f(sel.get('all_rows|-1', {}).get('e_mean'), 4)}     "
                  f"{f(sel.get('true_only|-1', {}).get('e_mean'), 4)}")
        P()
        P("    Row tags: b = k_big (the arm's own FROZEN COMMITTED size at that level),")
        P("    s = k_small (= round(k_big/3)), o = k = 1, the top row alone. * = k equals the")
        P("    whole operative table, so that row is the table, not a choice. The `random`")
        P("    column is the mean and sd over its draws; those draws changed between this tag's")
        P("    two passes because the rng now draws three k's instead of two -- every")
        P("    DETERMINISTIC column is identical across the passes.")
        P()
        P("    HOW DIFFERENTLY THE SELECTORS CHOOSE. `jac` is the Jaccard overlap of each")
        P("    selector's top-k with the WORLD's top-k at k_big. `w_max` is the best SINGLE")
        P("    row's own world price at that level and `rank@sel` is that row's rank under each")
        P("    selector (1 = first). Read these BESIDE the error table above: where the errors")
        P("    coincide while `jac` is far from 1, the audition is not distinguishing the reads")
        P("    however differently they order the table. No mechanism is claimed here.")
        P("    L    w_max  best row  rank: world banked shaped_r frozen   twin dp_top  |  "
          "jac vs world: banked shaped_r frozen   twin dp_top")
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            sel, rws = c["selectors"], c["rows"]
            wv = np.array([r["w_succ"] for r in rws], np.float64)
            ibest = int(np.argmax(wv))
            sc = {"world": wv, "dp_top": np.array([r["dp_top"] for r in rws], np.float64)}
            for nm in ("banked", "shaped_refit", "frozen", "twin"):
                sc[nm] = np.array([r[f"p_{nm}"] for r in rws], np.float64)
            rk = {}
            for nm in SEL:
                o = np.argsort(-sc[nm], kind="mergesort")
                rk[nm] = int(np.nonzero(o == ibest)[0][0]) + 1
            base = set(sel[f"world|{c['k_big']}"]["idx"])
            jac = {}
            for nm in SEL[1:]:
                o2 = set(sel[f"{nm}|{c['k_big']}"]["idx"])
                jac[nm] = len(base & o2) / max(1, len(base | o2))
            P(f"    {ell}   {f(wv.max(), 3):>6}  #{ibest:<7} " +
              " ".join(f"{rk[nm]:>6}" for nm in SEL) + "  |               " +
              " ".join(f"{f(jac[nm], 3):>6}" for nm in SEL[1:]))
        P()
        P("    FALSE rows admitted by each selector (of the level's own false count), and the")
        P("    TRUE rows it keeps:")
        P("    L   k      world     banked   shaped_r   frozen     twin     dp_top   | "
          "false/rows")
        for ell in LEVELS:
            c = a["cells"].get(ell)
            if not c:
                continue
            sel, s_ = c["selectors"], c["stat"]
            for k_ in c.get("ks", [c["k_big"], c["k_small"]]):
                cells = []
                for nm in SEL:
                    q = sel.get(f"{nm}|{k_}", {})
                    cells.append(f"{q.get('n_false', '-')}F/{q.get('n_true', '-')}T")
                P(f"    {ell} {k_:>4}  " + " ".join(q.rjust(9) for q in cells) +
                  f"  | {s_['n_false']}/{s_['n_rows']}")
        P()

    # ---------------------------------------------------------------- [C] the bill
    P("=" * W)
    P("[C] THE BILL")
    P("    arm       auditions      sec  peak RSS MB  peak GPU MB")
    tot = 0.0
    for k in order:
        a = arms[k]
        tot += a["sec"]
        P(f"    {k:<8} {a['n_auditions']:>10} {a['sec']:>8.1f} {a['peak_rss_mb']:>12.0f} "
          f"{a['peak_gpu_mb']:>12.0f}")
    P(f"    TOTAL              {tot:>13.1f}   ({tot / 3600.0:.3f} GPU-h over "
      f"{len(order)} containers)")
    P()

    txt = "\n".join(L) + "\n"
    out = args.out or os.path.join(HERE, "figures", "own_reduction.txt")
    with open(out, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"wrote {out}")

    # per-row TSVs, the per-candidate table of this node
    for k in order:
        a = arms[k]
        p = os.path.join(d, f"{k}_rows.tsv")
        with open(p, "w") as fh:
            cols = (["arm", "level", "row", "child", "truth", "w_succ", "e", "dres"] +
                    [f"p_{nm}" for nm in ("banked", "shaped_refit", "frozen", "twin")] +
                    ["dp_top", "dp_lse", "dp_marg"])
            fh.write("\t".join(cols) + "\n")
            for ell in LEVELS:
                c = a["cells"].get(ell)
                if not c:
                    continue
                for r in c["rows"]:
                    fh.write("\t".join([k, ell, str(r["i"]),
                                        ",".join(str(q) for q in r["child"]),
                                        str(r["truth"]), f"{r['w_succ']:.6f}",
                                        f"{r['e']:.6f}", f"{r['dres']:.6f}"] +
                                       [f"{r['p_' + nm]:.6f}" for nm in
                                        ("banked", "shaped_refit", "frozen", "twin")] +
                                       [f"{r['dp_top']:.6f}", f"{r['dp_lse']:.6f}",
                                        f"{r['dp_marg']:.6f}"]) + "\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
