"""[preplay/select] THE REDUCER for tag pp2 — the read in the selector's seat.

Writes `figures/select_reduction.txt` and appends the same content to the node's table of
record, `figures/preplay_reduction.txt`, as section [S] (idempotently: an existing [S] block is
replaced, so re-running never stacks).

Sections:
  [S0] the gates: S-1's pool disjointness per arm and level, and F-2b per arm
  [S1] the PRICING pool's own true-vs-wrong AUC per selector — pp1's [A:single] re-measured at
       the larger pricing n, so the ranking the selectors are handed is on the record beside
       what they then do with it
  [S]  THE TABLE OF RECORD: per arm, level, form and k, the world's audition error of each
       selector's table on three DISJOINT test pools (mean +/- sd over pools), with the number
       of WRONG entries each selector admitted, beside the random baseline, the world ceiling,
       pp1's matched-size random subset of the true table, and the full true table
  [S2] THE CAPTURE FRACTION: (e_random - e_selector) / (e_random - e_world) — how much of the
       world selector's advantage over a random draw the endogenous read recovers. Undefined
       and printed `--` where the world's own advantage is below the test pools' own noise.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_selector.py --tag pp2 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = ("world", "banked", "shaped_refit", "frozen", "twin", "dp_top", "random")
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]
MARK = "[S] THE READ IN THE SELECTOR'S SEAT"


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
    ap.add_argument("--tag", default="pp2")
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
    out = []

    def o(line=""):
        out.append(line)

    cfg0 = arms[order[0]]["cfg"]
    o("=" * 126)
    o(MARK + f" — tag {args.tag}")
    o("    Does selecting candidate entries by the endogenous read build a better table than")
    o("    selecting at random, and how close to selecting by the world? The abstraction-")
    o("    supervision consumer in one offline step.  arms: " + ", ".join(order))
    o(f"    pricing pool n = {cfg0['n_price']} (one, its own seed family); test pools "
      f"{cfg0['n_test_pools']} x n = {cfg0['n_test']}, DISJOINT seed family, never priced on.")
    o(f"    candidate pool: up to {cfg0['k_pool']} true + {cfg0['k_pool']} wrong entries, "
      f"pp1's construction. Ties broken by one permutation shared by every selector.")
    o("=" * 126)

    # ------------------------------------------------------------------- [S0] gates ---- #
    o("")
    o("[S0] THE GATES")
    o("     S-2 (a constant price reduces exactly to the random draw; the top-k really is the")
    o("     top-k), S-3 (the selection's own true/wrong accounting ties to `MC.grade_table`)")
    o("     and S-4 (selecting by the world's own price beats a random k on the pool it was")
    o("     priced on) are closed in `selector.py::gates2` and falsified in `::falsify2`.")
    o("     `preplay.py`'s F-1..F-6 carry over; the projection's read is IMPORTED from")
    o("     `preplay.py`, so its exact gate F-2 (max|dp| = 0.000e+00 against `VoProjBank`")
    o("     itself, all six arms) covers this node by identity.")
    o("")
    o(f"     {'arm':8} {'F-2b |d|':>9} {'band':>7}   S-1: shared instances, pricing x test "
      f"and test x test, per level")
    for a in order:
        g = arms[a]["gate"]
        s1 = arms[a]["S-1"]
        bits = []
        for lv in sorted(s1, key=lambda z: int(z[1:])):
            bits.append(f"{lv}:{max(s1[lv]['overlap_price_test'] + s1[lv]['overlap_test_test'] + [0])}")
        o(f"     {a:8} {g['F-2b:delta']:>9.4f} {g['F-2b:band']:>7.4f}   " + "  ".join(bits))

    # ------------------------------------------------- [S1] the ranking they are handed - #
    o("")
    o("=" * 126)
    o("[S1] THE RANKING EACH SELECTOR IS HANDED — true-vs-wrong AUC on the PRICING pool")
    o(f"     (n = {cfg0['n_price']}; pp1 measured the same quantity at n = 256 and its")
    o("     [A:single] column is the comparison). `random` is 0.5 by construction.")
    o("")
    o(f"     {'arm':8} {'L':>2} {'n(T/W)':>9} " +
      " ".join(f"{k[:12]:>12}" for k in SEL if k != "random"))
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            c = arms[a]["cells"][lv]
            o(f"     {a:8} {lv:>2} {c['n_cand_true']:>4d}/{c['n_cand_wrong']:<4d} " +
              " ".join(f(c["price_auc"].get(k), 12, 3) for k in SEL if k != "random"))

    # ------------------------------------------------------------- [S] the main table --- #
    for form in ("scratch", "extend"):
        o("")
        o("=" * 126)
        if form == "scratch":
            o("[S:scratch] FROM SCRATCH — the table IS the top-k of the candidate pool.")
            o("    e = the world's audition error of the built table, mean over the three test")
            o("    pools (sd in parentheses). w = wrong entries admitted. k = the arm's")
            o("    committed n_entries and half and a quarter of it.")
            o("    `randT` is pp1's reference: a matched-size random subset of the TRUE table")
            o("    (3 draws) — a selector with no pool to choose from. `true_full` is the whole")
            o("    true table. LOWER e IS BETTER.")
        else:
            o("[S:extend] EXTENSION — a small base (a quarter of the committed size, because")
            o("    pp1 measured that at the committed sizes the DP never picks 92-99% of added")
            o("    entries) plus the top-k of the remaining candidates. Base drawn from the")
            o("    TRUE table, three draws, each row a draw. `base` is the base alone.")
        o("")
        hdr = (f"    {'arm':8} {'L':>2} {'k':>5} {'base':>6} " +
               " ".join(f"{k[:13]:>13}" for k in SEL) +
               (f" {'randT':>13} {'true_full':>9}" if form == "scratch" else ""))
        o(hdr)
        for a in order:
            for lv in sorted(arms[a]["cells"], key=int):
                c = arms[a]["cells"][lv]
                if form == "scratch":
                    for k in c["ks"]:
                        r = c["scratch"][str(k)]
                        cells = []
                        for sel in SEL:
                            q = r[sel]
                            cells.append(f"{q['e_mean']:.3f}({q['e_sd']:.2f})w{q['n_wrong_admitted']:<2d}"
                                         .rjust(13))
                        rt = c["refs"]["rand_true_k"][str(k)]
                        o(f"    {a:8} {lv:>2} {k:>5d} {'-':>6} " + " ".join(cells) +
                          f" {rt['e_mean']:.3f}({rt['e_sd']:.2f}) ".rjust(14) +
                          f"{c['refs']['true_full']['e_mean']:>9.3f}")
                else:
                    for blk in c["extend"]["draws"]:
                        for ke in c["k_exts"]:
                            r = blk["k"][str(ke)]
                            cells = []
                            for sel in SEL:
                                q = r[sel]
                                cells.append(
                                    f"{q['e_mean']:.3f}({q['e_sd']:.2f})w{q['n_wrong_admitted']:<2d}"
                                    .rjust(13))
                            o(f"    {a:8} {lv:>2} {ke:>5d} "
                              f"{blk['base_alone']['e_mean']:>6.3f} " + " ".join(cells))

    # -------------------------------------------------------- [S2] the capture fraction - #
    o("")
    o("=" * 126)
    o("[S2] THE CAPTURE FRACTION — (e_random - e_selector) / (e_random - e_world).")
    o("     1.0 means the selector matches the world's own choice; 0.0 means it is no better")
    o("     than a random draw from the same pool; negative means worse than random. Printed")
    o("     `--` where the WORLD's own advantage over random is smaller than the test pools'")
    o("     own spread (the denominator is then noise and the ratio means nothing).")
    o("")
    o(f"    {'arm':8} {'L':>2} {'form':>8} {'k':>5} {'e_rand':>7} {'e_world':>8} "
      f"{'gap':>6} " + " ".join(f"{k[:12]:>12}" for k in SEL
                                if k not in ("world", "random")))
    caps = {k: [] for k in SEL if k not in ("world", "random")}
    caps_big = {k: [] for k in SEL if k not in ("world", "random")}
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            c = arms[a]["cells"][lv]
            rows = []
            for k in c["ks"]:
                rows.append(("scratch", k, c["scratch"][str(k)], 0.0))
            for blk in c["extend"]["draws"]:
                for ke in c["k_exts"]:
                    rows.append(("extend", ke, blk["k"][str(ke)],
                                 blk["base_alone"]["e_mean"]))
            for form, k, r, _b in rows:
                er, ew = r["random"]["e_mean"], r["world"]["e_mean"]
                noise = max(r["random"]["e_sd"], r["world"]["e_sd"])
                gap = er - ew
                ok = gap > max(noise, 1e-9)
                vals = []
                for sel in SEL:
                    if sel in ("world", "random"):
                        continue
                    q = (er - r[sel]["e_mean"]) / gap if ok else None
                    vals.append(q)
                    if q is not None:
                        caps[sel].append(q)
                        if gap >= 0.10:
                            caps_big[sel].append(q)
                o(f"    {a:8} {lv:>2} {form:>8} {k:>5d} {er:>7.3f} {ew:>8.3f} "
                  f"{gap:>6.3f} " + " ".join(f(q, 12, 3) for q in vals))
    o("")
    o("    Pooled over every (arm, level, form, k) cell where the world's advantage exceeds")
    o("    the test spread — a summary, NOT a seed-averaged claim; the per-cell rows above are")
    o("    the record:")
    o(f"    {'':8} {'n':>4} " + " ".join(f"{k[:12]:>12}" for k in caps))
    o(f"    {'median':8} {len(caps[list(caps)[0]]):>4d} " +
      " ".join(f(float(np.median(caps[k])) if caps[k] else None, 12, 3) for k in caps))
    o(f"    {'>0 share':8} {'':>4} " +
      " ".join(f(float(np.mean([q > 0 for q in caps[k]])) if caps[k] else None, 12, 3)
               for k in caps))
    o("")
    o("    The same, restricted to cells where the world's own advantage is at least 0.10 of")
    o("    audition error — the cells where the question is well posed and the ratio is not")
    o("    dominated by its denominator:")
    o(f"    {'':8} {'n':>4} " + " ".join(f"{k[:12]:>12}" for k in caps_big))
    o(f"    {'median':8} {len(caps_big[list(caps_big)[0]]):>4d} " +
      " ".join(f(float(np.median(caps_big[k])) if caps_big[k] else None, 12, 3)
               for k in caps_big))
    o(f"    {'>0 share':8} {'':>4} " +
      " ".join(f(float(np.mean([q > 0 for q in caps_big[k]])) if caps_big[k] else None, 12, 3)
               for k in caps_big))

    # ------------------------------------------------- [S4] what each selector admits --- #
    o("")
    o("=" * 126)
    o("[S4] WHAT EACH SELECTOR LETS ONTO THE TABLE — wrong entries admitted as a share of k,")
    o("     from-scratch form at k = the arm's committed n_entries. This is the least noisy")
    o("     column in the round: it needs no audition at all, only the selector's ranking and")
    o("     the oracle's own set. `exp` is the share a uniform draw from the pool would admit")
    o("     (n_wrong / n_cand), and `random` is the realised draw.")
    o("")
    o(f"    {'arm':8} {'L':>2} {'k':>5} {'exp':>6} " +
      " ".join(f"{k[:8]:>8}" for k in SEL))
    adm = {k: [] for k in SEL}
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            c = arms[a]["cells"][lv]
            k = c["ks"][0]
            r = c["scratch"][str(k)]
            exp = c["n_cand_wrong"] / float(c["n_cand"])
            vals = []
            for sel in SEL:
                q = r[sel]["n_wrong_admitted"] / float(max(1, r[sel]["n_sel"]))
                vals.append(q)
                adm[sel].append(q / exp if exp > 0 else None)
            o(f"    {a:8} {lv:>2} {k:>5d} {exp:>6.3f} " + " ".join(f(q, 8, 3) for q in vals))
    o("")
    o("    Median ratio to what a uniform draw from the same pool would admit (1.0 = no")
    o("    discrimination, 0.0 = admits no wrong entry at all), over the 24 cells above:")
    o(f"    {'':8} " + " ".join(f"{k[:8]:>8}" for k in SEL))
    o(f"    {'median':8} " +
      " ".join(f(float(np.median([q for q in adm[k] if q is not None])), 8, 3) for k in SEL))

    # ----------------------------------------------------------------------- the bill --- #
    o("")
    o("=" * 126)
    o("[S3] THE BILL")
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
    own = os.path.join(HERE, "figures", "select_reduction.txt")
    with open(own, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"[written] {own}")

    if not args.no_append:
        rec = os.path.join(HERE, "figures", "preplay_reduction.txt")
        if os.path.exists(rec):
            cur = open(rec).read()
            i = cur.find("=" * 126 + "\n" + MARK)
            if i >= 0:
                cur = cur[:i]
            if not cur.endswith("\n"):
                cur += "\n"
            with open(rec, "w") as fh:
                fh.write(cur.rstrip("\n") + "\n\n" + txt)
            print(f"[appended as section [S]] {rec}")


if __name__ == "__main__":
    main()
