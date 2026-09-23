"""[sostenuto] THE CROSS-SEED TABLE OF RECORD -> `figures/st_seedtable.txt`.

The two seeds, NEVER AVERAGED: a replication header computed directly off the arm files, then
every section [A] to [J] of each per-tag reduction, seed 0's block and seed 2's block adjacent
under one heading. `reduce_sostenuto.py` must have been run for both tags first.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/sostenuto/mk_seedtable.py
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BANK = os.path.join(os.path.dirname(HERE), "soundboard", "figures")
ORDER = ("st_gw_yk", "st_gr_yk", "st_gn_yk", "st_gy_yk", "st_gd_yk")    # [sostenuto r2]
GATE = {"st_gw_yk": "world", "st_gr_yk": "read", "st_gn_yk": "none",
        "st_gy_yk": "yield", "st_gd_yk": "demand"}
SEEDS = (("st_s1", 0, "sb_s1"), ("st_s2", 2, "sb_s2"))
# [sostenuto r2] round 2's tags carry only the two new arms; round 1's are read from theirs
SEEDS_R2 = (("st2_s1", 0, "sb_s1"), ("st2_s2", 2, "sb_s2"))
R1_OF = {"st2_s1": "st_s1", "st2_s2": "st_s2"}
SEP = "=" * 100


def arm_files(tag):
    out = {}
    for t_ in (tag, R1_OF.get(tag)):                  # [sostenuto r2] the arm's own tag first
        if t_ is None:
            continue
        for a in ORDER:
            p = os.path.join(FIG, t_, a, "results.json")
            if a not in out and os.path.isfile(p):
                with open(p) as fh:
                    out[a] = json.load(fh)
    return out


def bank_file(bt):
    p = os.path.join(BANK, bt, "sb_sv_yk", "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def conf_of(r, deep=False):
    c = {k: 0 for k in ("TT", "TF", "FT", "FF")}
    for ps in ((r.get("adm") or {}).get("passes") or []):
        for lr in ps["levels"]:
            if deep and lr["level"] < 3:
                continue
            for st in lr["steps"]:
                c["TT" if (st["admit_gate"] and st["admit_world"])
                  else "TF" if st["admit_gate"]
                  else "FT" if st["admit_world"] else "FF"] += 1
    return c


def sections(path):
    """[(heading, body)] of a reduction file, split on the `====` rules."""
    lines = open(path).read().splitlines()
    out, head, body, i = [], None, [], 0
    while i < len(lines):
        if lines[i].startswith("====") and i + 2 < len(lines) and lines[i + 2].startswith("===="):
            if head is not None:
                out.append((head, body))
            head, body, i = lines[i + 1], [], i + 3
            continue
        (body if head is not None else out).append(lines[i]) if head is not None else None
        if head is None:
            out.append(("__preamble__", [lines[i]]))
        i += 1
    if head is not None:
        out.append((head, body))
    merged, pre = [], []
    for h, b in out:
        if h == "__preamble__":
            pre += b
        else:
            merged.append((h, b))
    return pre, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--round", type=int, default=1)             # [sostenuto r2]
    a = ap.parse_args()
    global SEEDS
    if a.round == 2:
        SEEDS = tuple(q for q in SEEDS_R2
                      if os.path.isfile(os.path.join(FIG, f"{q[0]}_reduction.txt")))
        assert SEEDS, "run reduce_sostenuto.py --tag st2_s1 --with-tag st_s1 first"
    out_path = a.out or os.path.join(FIG, "st_seedtable.txt" if a.round == 1
                                     else "st2_seedtable.txt")
    L = []
    L.append("[sostenuto] THE ADMISSION SET ON THE MINER'S LIVE BUILD — THE TWO SEEDS, "
             "NEVER AVERAGED")
    L.append("  " + "   ".join(f"{t} (seed {sd})" for t, sd, _ in SEEDS)
             + "   yoked to vo_s3:voi3_dp (seed 0) / vo_s3d2:voi3_dp (seed 2)")
    if a.round == 2:
        L.append("  ROUND 2: st_gy_yk (yield) and st_gd_yk (demand) from st2_s*, round 1's three")
        L.append("  from st_s* (not re-run)")
    L.append("  no-walk reference: sb_s1:sb_sv_yk and sb_s2:sb_sv_yk, banked, not re-run")
    L.append("  Ranks, signs and per-cell counts are the claims. The seeds are never averaged.")

    A = {t: arm_files(t) for t, _, _ in SEEDS}
    B = {t: bank_file(bt) for t, _, bt in SEEDS}

    # ---- (a) the replication header ---------------------------------------------------- #
    L.append("")
    L.append(SEP)
    L.append("(a) WHAT REPLICATED — the three numbers the round turns on, per seed")
    L.append(SEP)
    L.append("")
    L.append("  (i) THE PER-CANDIDATE CONFUSION OVER L3-L5 (where the read gate ever had a")
    L.append("      live decision; L2 is decided before the readout's first fit on both seeds)")
    L.append(f"      {'seed':>4} {'arm':<10} {'n':>4} {'TT':>4} {'TF':>4} {'FT':>4} {'FF':>4}"
             f" {'TF rate':>8} {'FT rate':>8}")
    for t, sd, _ in SEEDS:
        for arm in ORDER:
            if arm not in A[t]:
                continue
            c = conf_of(A[t][arm], deep=True)
            n = sum(c.values())
            if not n:
                continue
            L.append(f"      {sd:>4} {arm:<10} {n:>4} {c['TT']:>4} {c['TF']:>4} {c['FT']:>4}"
                     f" {c['FF']:>4} {c['TF'] / n:>8.3f} {c['FT'] / n:>8.3f}")
    L.append("      pp5 offline, at the same budget: read TF 0.022 FT 0.050 against ungated's")
    L.append("      TF 0.045 FT 0.000.")

    L.append("")
    L.append("  (ii) THE DEEP-ERA TASK ERROR, against each seed's own banked no-walk reference")
    L.append(f"      {'seed':>4} {'arm':<10} " + " ".join(f"{'era' + str(i):>8}"
                                                          for i in range(1, 6)))
    for t, sd, _ in SEEDS:
        base = None
        if B[t] is not None:
            e = np.asarray(B[t]["log"]["e"], float)
            er = np.asarray(B[t]["log"]["era"], int)
            base = [float(e[er == i].mean()) if (er == i).any() else None for i in range(1, 6)]
        for arm in ORDER:
            if arm not in A[t]:
                continue
            e = np.asarray(A[t][arm]["log"]["e"], float)
            er = np.asarray(A[t][arm]["log"]["era"], int)
            row = [float(e[er == i].mean()) if (er == i).any() else None for i in range(1, 6)]
            L.append(f"      {sd:>4} {arm:<10} " + " ".join(
                (f"{row[i] - base[i]:>+8.4f}" if (base and row[i] is not None
                                                  and base[i] is not None) else f"{'—':>8}")
                for i in range(5)))
    L.append("      (deltas against the reference; negative = the walk arm is better)")
    L.append("")
    L.append("      the same, against the NONE arm (the gate net of the op's own footprint)")
    for t, sd, _ in SEEDS:
        if "st_gn_yk" not in A[t]:
            continue
        e0 = np.asarray(A[t]["st_gn_yk"]["log"]["e"], float)
        er0 = np.asarray(A[t]["st_gn_yk"]["log"]["era"], int)
        bn = [float(e0[er0 == i].mean()) if (er0 == i).any() else None for i in range(1, 6)]
        for arm in ORDER:
            if arm not in A[t] or arm == "st_gn_yk":
                continue
            e = np.asarray(A[t][arm]["log"]["e"], float)
            er = np.asarray(A[t][arm]["log"]["era"], int)
            row = [float(e[er == i].mean()) if (er == i).any() else None for i in range(1, 6)]
            L.append(f"      {sd:>4} {arm:<10} " + " ".join(
                (f"{row[i] - bn[i]:>+8.4f}" if (row[i] is not None and bn[i] is not None)
                 else f"{'—':>8}") for i in range(5)))

    L.append("")
    L.append("  (iii) THE COVERAGE TRACE — keys refused, and the demand-blindness signature:")
    L.append("       of the keys the UNGATED arm supported at level l, how many have a half")
    L.append("       the gate arm REFUSED at level l-1")
    L.append(f"      {'seed':>4} {'arm':<10} {'refused L2':>11} {'refused L3':>11}"
             f" {'refused L4+':>12} {'L3 blocked':>11} {'L4 blocked':>11}")
    for t, sd, _ in SEEDS:
        for arm in [q for q in ORDER if q != "st_gn_yk"]:           # [sostenuto r2]
            if arm not in A[t] or "st_gn_yk" not in A[t]:
                continue
            dec = {}
            for ps in ((A[t][arm].get("adm") or {}).get("passes") or []):
                for lr in ps["levels"]:
                    for st in lr["steps"]:
                        dec[(lr["level"], tuple(st["key"]))] = bool(st["admit_gate"])
            nd = {}
            for ps in ((A[t]["st_gn_yk"].get("adm") or {}).get("passes") or []):
                for lr in ps["levels"]:
                    for st in lr["steps"]:
                        nd.setdefault(lr["level"], set()).add(tuple(st["key"]))
            ref = {lv: {k for (l_, k), ok in dec.items() if l_ == lv and not ok}
                   for lv in (2, 3, 4)}
            blocked = {}
            for lv in (3, 4):
                ks = nd.get(lv, set())
                blocked[lv] = (sum(1 for k in ks
                                   if any(h in ref.get(lv - 1, set())
                                          for h in (k[:len(k) // 2], k[len(k) // 2:]))),
                               len(ks))
            L.append(f"      {sd:>4} {arm:<10} {len(ref[2]):>11} {len(ref[3]):>11}"
                     f" {len(ref[4]):>12}"
                     f" {(str(blocked[3][0]) + '/' + str(blocked[3][1])):>11}"
                     f" {(str(blocked[4][0]) + '/' + str(blocked[4][1])):>11}")

    L.append("")
    L.append("  (iv) THE BILL, and the rows served")
    L.append(f"      {'seed':>4} {'arm':<10} {'gate':>6} {'offered':>8} {'refused':>8}"
             f" {'world billed':>13} {'L3 rows adm/live':>17} {'L5 rows adm/live':>17}")
    for t, sd, _ in SEEDS:
        for arm in ORDER:
            if arm not in A[t]:
                continue
            adm = A[t][arm].get("adm") or {}
            last = {}
            for ps in adm.get("passes") or []:
                for lr in ps["levels"]:
                    last[lr["level"]] = lr
            def rr(lv):
                q = last.get(lv)
                return f"{q['n_kept_rows']}/{q['n_live_rows']}" if q else "—"
            L.append(f"      {sd:>4} {arm:<10} {str(adm.get('gate')):>6}"
                     f" {adm.get('n_offered', 0):>8} {adm.get('n_reject', 0):>8}"
                     f" {adm.get('n_world_billed', 0):>13} {rr(3):>17} {rr(5):>17}")

    # ---- (b) every section, both seeds ------------------------------------------------- #
    cuts = {}
    for t, sd, _ in SEEDS:
        p = os.path.join(FIG, f"{t}_reduction.txt")
        assert os.path.isfile(p), f"run reduce_sostenuto.py --tag {t} first"
        _pre, secs = sections(p)
        cuts[t] = secs
    heads = [h for h, _ in cuts[SEEDS[0][0]]]
    for h in heads:
        L.append("")
        L.append(SEP)
        L.append(h)
        L.append(SEP)
        for t, sd, _ in SEEDS:
            body = dict(cuts[t]).get(h)
            L.append("")
            L.append(f"  ---------- SEED {sd}  ({t}) ----------")
            L += (body if body is not None else ["  (section absent for this seed)"])
    txt = "\n".join(L) + "\n"
    with open(out_path, "w") as fh:
        fh.write(txt)
    print(txt[:txt.index("(b)") if "(b)" in txt else 9000])
    print(f"\n  -> {out_path}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
