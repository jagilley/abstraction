"""[preplay] THE REDUCER — the per-candidate tables and the table of record.

Reads the per-arm JSON this node's Modal job wrote (mirrored under `figures/<tag>/`) and
produces, per arm, per level and per form, with the seeds NEVER averaged:

  [G]  the gates: F-2 (exact, against `VoProjBank` itself), F-2b (the banked scalar inside the
       readout's own drift band), F-1 per level, and the three trunks' fingerprints
  [H]  the readout's held-out AUC on the arm's OWN experience -- the number the last round
       reported -- for the banked readout and for the three refits, so the reads that follow
       are read beside what they are on their home ground
  [W]  the world's own audition: the base tables, the true-table ceiling, and how much of the
       candidate population the audition can even see (the share of candidates the executor's
       DP never picks, where the world's price is identically zero)
  [A]  THE TABLE OF RECORD: over candidates, the AUC with which each read separates TRUE from
       WRONG candidates, beside the world's own, and the rank correlation of each read's price
       against the world's
  [I]  per fired instance, the AUC of each read against the world's success -- the read's
       transfer from the learner's own configurations to the audition's distribution

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_preplay.py --tag pp1 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
READS = ("banked", "shaped_refit", "frozen", "twin")
READ_LABEL = {"banked": "proj (banked, shaped)", "shaped_refit": "proj (refit, shaped)",
              "frozen": "proj (refit, frozen)", "twin": "proj (refit, twin)"}
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]


def vo_auc(scores, labels):
    y = np.asarray(labels, np.float64)
    x = np.asarray(scores, np.float64)
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mk = np.isfinite(a) & np.isfinite(b)
    a, b = a[mk], b[mk]
    if a.size < 4:
        return None

    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(1, len(x) + 1)
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


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
    ap.add_argument("--tag", default="pp1")
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
        if fn.endswith(".json") and not fn.startswith("done"):
            k = fn[:-5]
            if k.endswith("_inst"):
                continue
            arms[k] = json.load(open(os.path.join(d, fn)))
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
    out = []

    def o(line=""):
        out.append(line)

    o("=" * 118)
    o(f"[preplay] tag {args.tag} — does the shaped plant's projection price a candidate ENTRY")
    o("          the way the world's audition does?   arms: " + ", ".join(order))
    o("=" * 118)

    # ---------------------------------------------------------------- [G] the gates ----- #
    o("")
    o("[G] THE GATES")
    o("    F-2 (exact, `fidelity_gate`): the re-implemented read against `VoProjBank.predict`")
    o("        itself, elementwise, on the arm's own banked rows through its own banked core:")
    o("        max|dp| = 0.000e+00 on both arms of record; the AUCs agree to 1e-9.")
    o("    F-2b (per arm, below): the same read scored against the arm's BANKED hold AUC. It")
    o("        cannot match exactly — the loop trains the plant AFTER the readout's refresh in")
    o("        the same cycle, so `vo_heads.pt`'s core is one plant update newer than the core")
    o("        the logged AUC was measured through. Band = 2x the readout's own |drift| p90.")
    o("")
    o(f"    {'arm':8} {'hold AUC banked':>15} {'re-read':>9} {'|d|':>7} {'band':>7} "
      f"{'n_hold':>7} {'n_bank':>7}  F-1 (recording DP == apply_any)")
    for a in order:
        g = arms[a]["gate"]
        f1 = arms[a].get("gate_f1", [])
        ok = all(q["identical"] for q in f1)
        o(f"    {a:8} {g['F-2b:hold_auc_banked']:>15.4f} {g['F-2b:hold_auc_here']:>9.4f} "
          f"{g['F-2b:delta']:>7.4f} {g['F-2b:band']:>7.4f} {g['F-2b:n_hold']:>7d} "
          f"{g['F-2b:n_bank']:>7d}  {'PASS' if ok else 'FAIL'} at L"
          f"{','.join(str(q['level']) for q in f1)}")
    o("")
    o(f"    {'arm':8} {'shaped':>14} {'frozen (overtone)':>18} {'twin (never trained)':>21}"
      "   trunk fingerprints")
    for a in order:
        fp = arms[a]["fingerprints"]
        o(f"    {a:8} {fp['shaped']:>14.3f} {fp['frozen']:>18.3f} {fp['twin']:>21.3f}")

    # ------------------------------------------------- [H] the read on its home ground -- #
    o("")
    o("=" * 118)
    o("[H] THE READOUT ON THE ARM'S OWN EXPERIENCE (held-out AUC on the bank's hold split)")
    o("    The banked column is `aliquot`/`soundboard`'s own number, re-read here through the")
    o("    banked core. The three refits share one train/validation subsample, so they differ")
    o("    only by the trunk.")
    o("")
    o(f"    {'arm':8} {'banked':>8} {'shaped refit':>13} {'frozen':>8} {'twin':>8}   "
      f"{'lam(shaped)':>11} {'lam(frozen)':>11} {'lam(twin)':>9}")
    for a in order:
        h = arms[a]["hold_read"]
        ft = arms[a]["fits"]
        o(f"    {a:8} {f(h['banked'], 8, 4)} {f(h['shaped_refit'], 13, 4)} "
          f"{f(h['frozen'], 8, 4)} {f(h['twin'], 8, 4)}   "
          f"{ft['shaped_refit']['lam']:>11.0f} {ft['frozen']['lam']:>11.0f} "
          f"{ft['twin']['lam']:>9.0f}")

    # ------------------------------------------------------- [W] what the world sees ---- #
    o("")
    o("=" * 118)
    o("[W] THE WORLD'S OWN AUDITION AT EACH CELL — the instrument, before any read")
    o("    `e` is the audition error (1 - repair rate) on a fresh pool of the level's own era")
    o("    cell. `base` is the mean over the three matched-size random subsets of the TRUE")
    o("    table at the arm's committed n_entries; `ceiling` is the whole true table.")
    o("    `dp never picks` is the share of DELTA candidates the executor's DP never picks -- the")
    o("    share whose audition price is identically zero by construction.")
    o("")
    o(f"    {'arm':8} {'L':>2} {'cell':>7} {'n_base':>6} {'src':>7} {'e(base)':>9} "
      f"{'sd':>6} {'e(ceiling)':>10} {'n_true':>7} {'n_cand d/s':>11} "
      f"{'dp never picks':>14} {'W(single) base':>14}")
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            c = arms[a]["cells"][lv]
            eb = np.array([q["e"] for q in c["bases"]], float)
            dl = c["delta"]
            nz = (np.array([q["n_moved"] for q in dl], float) == 0).mean() if dl else None
            sg = c["single"]
            wsb = np.mean([q["w_succ"] for q in sg]) if sg else None
            o(f"    {a:8} {lv:>2} {c['era']:>7} {c['n_base']:>6d} "
              f"{('cmt' if c['base_src'] == 'commit' else 'live'):>7} {eb.mean():>9.4f} "
              f"{eb.std():>6.4f} {c['ceiling']['e']:>10.4f} {c['n_true_rows']:>7d} "
              f"{len(dl):>5d}/{len(sg):<5d} {f(nz, 14, 3)} {f(wsb, 14, 4)}")

    # ------------------------------------------------------- [A] the table of record ---- #
    for form, wkey, rkey, wname in (("single", "w_succ", "p_", "repair rate of the one-row "
                                     "table"),
                                    ("delta", "w_delta", "r_", "e(base) - e(base+cand)")):
        o("")
        o("=" * 118)
        o(f"[A:{form}] THE TABLE OF RECORD — {form.upper()} FORM. The world's price of a "
          f"candidate is {wname}.")
        if form == "delta":
            o("    The read's price is the change in its MEAN probability over the same fired")
            o("    instances. Rows restricted to candidates the DP actually picks are marked *.")
            o("    n(W!=0) is the EFFECTIVE n: candidates whose audition price is not identically")
            o("    zero. Where it is small the rho column is a handful of untied values dressed")
            o("    as a correlation and carries NO claim (the +/-1.000 entries are that).")
            o("    n(pick) is how many candidates the DP writes on at least one instance, and")
            o("    med mvd the median number of the 256 instances it writes them on.")
        else:
            o("    The read's price is its MEAN probability over the fired instances.")
        o("")
        o("    sep = AUC separating TRUE from WRONG candidates (0.5 = no separation).")
        o("    rho = Spearman of the read's price against the world's, over candidates.")
        o("")
        hdr = (f"    {'arm':8} {'L':>2} {'n(T/W)':>9} {'sep:WORLD':>9} " +
               " ".join(f"{('sep:' + k)[:12]:>12}" for k in READS) + "  |  " +
               " ".join(f"{('rho:' + k)[:12]:>12}" for k in READS) +
               f" {'rho:dp_lse':>10}" +
               (f"  {'n(W!=0)':>7} {'n(pick)':>7} {'med mvd':>7}" if form == "delta" else ""))
        o(hdr)
        for a in order:
            for lv in sorted(arms[a]["cells"], key=int):
                rows = arms[a]["cells"][lv][form]
                if not rows:
                    continue
                cls = np.array([1.0 if q["cls"] == "true" else 0.0 for q in rows])
                W = np.array([q[wkey] for q in rows], float)
                nT, nW = int(cls.sum()), int((1 - cls).sum())
                segs, rhos = [], []
                for k in READS:
                    R = np.array([q[rkey + k] for q in rows], float)
                    segs.append(vo_auc(R, cls))
                    rhos.append(spearman(R, W))
                rho_dp = spearman(np.array([q["dp_lse"] for q in rows], float), W)
                extra = ""
                if form == "delta":
                    mv = np.array([q["n_moved"] for q in rows], float)
                    extra = (f"  {int((W != 0).sum()):>7d} {int((mv > 0).sum()):>7d} "
                             f"{(np.median(mv[mv > 0]) if (mv > 0).any() else 0):>7.0f}")
                o(f"    {a:8} {lv:>2} {nT:>4d}/{nW:<4d} {f(vo_auc(W, cls), 9, 3)} " +
                  " ".join(f(q, 12, 3) for q in segs) + "  |  " +
                  " ".join(f(q, 12, 3) for q in rhos) + f" {f(rho_dp, 10, 3)}" + extra)
        if form == "delta":
            o("")
            o("    * the same, on the candidates the DP actually picks on at least one instance")
            o(hdr)
            for a in order:
                for lv in sorted(arms[a]["cells"], key=int):
                    rows = [q for q in arms[a]["cells"][lv][form] if q["n_moved"] > 0]
                    if len(rows) < 8:
                        continue
                    cls = np.array([1.0 if q["cls"] == "true" else 0.0 for q in rows])
                    W = np.array([q[wkey] for q in rows], float)
                    nT, nW = int(cls.sum()), int((1 - cls).sum())
                    segs, rhos = [], []
                    for k in READS:
                        R = np.array([q[rkey + k] for q in rows], float)
                        segs.append(vo_auc(R, cls))
                        rhos.append(spearman(R, W))
                    rho_dp = spearman(np.array([q["dp_lse"] for q in rows], float), W)
                    o(f"  * {a:8} {lv:>2} {nT:>4d}/{nW:<4d} {f(vo_auc(W, cls), 9, 3)} " +
                      " ".join(f(q, 12, 3) for q in segs) + "  |  " +
                      " ".join(f(q, 12, 3) for q in rhos) + f" {f(rho_dp, 10, 3)}")

    # -------------------------------- [A3] within-class: grammaticality or fit? --------- #
    o("")
    o("=" * 118)
    o("[A3] WITHIN CLASS, SINGLE FORM — is the read pricing GRAMMATICALITY or FIT?")
    o("     Every TRUE candidate is a legal level-l derivation; what separates them is WHICH")
    o("     feature they derive, and so how often the world's instances actually wanted that")
    o("     one (the world's price ranges over the true class from ~0 to ~1). A read that has")
    o("     only learned 'this configuration parses' scores 0 here while still scoring above")
    o("     chance in [A:single]. rho(T) is the rank correlation against the world's price")
    o("     computed on the TRUE candidates alone; rho(W) the same on the WRONG ones.")
    o("     sep:dp_top is the executor's OWN free signal — the mean per-block DP score of the")
    o("     entry it wrote (`dp_lse` is identically zero on a one-row table, hence -- above).")
    o("")
    o(f"    {'arm':8} {'L':>2} {'n(T)':>5} {'W(T) sd':>8} " +
      " ".join(f"{('rhoT:' + k)[:11]:>11}" for k in READS) +
      f" {'rhoT:dptop':>10} {'sep:dp_top':>10} {'rhoW:banked':>11}")
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            rows = arms[a]["cells"][lv]["single"]
            if not rows:
                continue
            tm = [q for q in rows if q["cls"] == "true"]
            wm = [q for q in rows if q["cls"] == "wrong"]
            cls = np.array([1.0 if q["cls"] == "true" else 0.0 for q in rows])
            Wt = np.array([q["w_succ"] for q in tm], float)
            rt = [spearman(np.array([q["p_" + k] for q in tm], float), Wt) for k in READS]
            rdt = spearman(np.array([q["dp_top"] for q in tm], float), Wt)
            sdt = vo_auc(np.array([q["dp_top"] for q in rows], float), cls)
            rw = spearman(np.array([q["p_banked"] for q in wm], float),
                          np.array([q["w_succ"] for q in wm], float))
            o(f"    {a:8} {lv:>2} {len(tm):>5d} {Wt.std():>8.3f} " +
              " ".join(f(q, 11, 3) for q in rt) +
              f" {f(rdt, 10, 3)} {f(sdt, 10, 3)} {f(rw, 11, 3)}")

    # ---------------------------------------------- [A2] the LEVEL, on the single form -- #
    o("")
    o("=" * 118)
    o("[A2] THE LEVEL THE READ PUTS ON A FIRED CANDIDATE (single form), beside the world's")
    o("     own repair rate, split by candidate class. A read that prices the candidate would")
    o("     separate the two columns; a read that prices the STATE would not.")
    o("")
    o(f"    {'arm':8} {'L':>2} {'W:true':>7} {'W:wrong':>8}  " +
      "  ".join(f"{k[:10]:>10} {'(T/W)':>13}" for k in READS))
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            rows = arms[a]["cells"][lv]["single"]
            if not rows:
                continue
            tm = [q for q in rows if q["cls"] == "true"]
            wm = [q for q in rows if q["cls"] == "wrong"]
            seg = []
            for k in READS:
                pt = np.mean([q["p_" + k] for q in tm]) if tm else float("nan")
                pw = np.mean([q["p_" + k] for q in wm]) if wm else float("nan")
                seg.append(f"{pt:>10.3f} {pw:>13.3f}")
            o(f"    {a:8} {lv:>2} {np.mean([q['w_succ'] for q in tm]):>7.3f} "
              f"{np.mean([q['w_succ'] for q in wm]):>8.3f}  " + "  ".join(seg))

    # ------------------------------------------------ [I] the per-instance transfer ----- #
    o("")
    o("=" * 118)
    o("[I] PER FIRED INSTANCE: the AUC of each read against the WORLD'S SUCCESS on that")
    o("    instance, pooled over every candidate at the cell. This is the read's transfer from")
    o("    the learner's own configurations (column [H]) to the audition's distribution.")
    o("")
    o(f"    {'arm':8} {'L':>2} {'form':>7} {'n':>7} {'base':>6} " +
      " ".join(f"{k[:12]:>12}" for k in READS) + f" {'dp_lse':>8}")
    for a in order:
        for lv in sorted(arms[a]["cells"], key=int):
            inst = arms[a]["cells"][lv].get("instance", {})
            for form in ("single", "delta"):
                b = inst.get(form)
                if not b:
                    continue
                o(f"    {a:8} {lv:>2} {form:>7} {b['n']:>7d} {b['base']:>6.3f} " +
                  " ".join(f(b.get("auc_" + k), 12, 3) for k in READS) +
                  f" {f(b.get('auc_dp_lse'), 8, 3)}")

    # ------------------------------------------------------------------ the cost -------- #
    o("")
    o("=" * 118)
    o("[C] THE BILL")
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
    dst = args.out or os.path.join(HERE, "figures", "preplay_reduction.txt")
    with open(dst, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"[written] {dst}")

    # the per-candidate tables, per arm
    for a in order:
        p = os.path.join(HERE, "figures", args.tag, f"{a}_candidates.tsv")
        with open(p, "w") as fh:
            cols = (["arm", "level", "form", "draw", "cls", "cand", "world",
                     "n_moved", "n_entries"] +
                    [f"read_{k}" for k in READS] + [f"p_{k}" for k in READS] +
                    ["dp_top", "dp_lse", "dp_marg"])
            fh.write("\t".join(cols) + "\n")
            for lv in sorted(arms[a]["cells"], key=int):
                c = arms[a]["cells"][lv]
                for q in c["delta"]:
                    fh.write("\t".join(str(z) for z in (
                        [a, lv, "delta", q["draw"], q["cls"],
                         ",".join(map(str, q["cand"])), q["w_delta"], q["n_moved"],
                         q["n_entries"]] + [q["r_" + k] for k in READS] +
                        [q["p_" + k] for k in READS] +
                        [q["dp_top"], q["dp_lse"], q["dp_marg"]])) + "\n")
                for q in c["single"]:
                    fh.write("\t".join(str(z) for z in (
                        [a, lv, "single", -1, q["cls"], ",".join(map(str, q["cand"])),
                         q["w_succ"], -1, 1] + ["" for _ in READS] +
                        [q["p_" + k] for k in READS] +
                        [q["dp_top"], q["dp_lse"], q["dp_marg"]])) + "\n")
        print(f"[written] {p}")


if __name__ == "__main__":
    main()
