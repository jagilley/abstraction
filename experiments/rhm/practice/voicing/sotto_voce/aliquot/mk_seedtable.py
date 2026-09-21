"""[aliquot] THE TWO-SEED TABLE OF RECORD — assembled from the banked and the new arm files.

`analyze_aliquot.py` reduces ONE tag at a time, which is right for the per-slot and per-cycle
sections but leaves the cross-seed comparison in two files. This script is the seed table, the
`sotto_voce` node's `figures/so_seedtable.txt` one node on: (a) the chooser, (b) the critic,
(c) the grader against the world per level, (d) the blind region per (level, in-support),
(e) the readout itself, (f) the bill and the yoke, (g) the grader's OWN ranking of the world's
verdict from the per-probe reservoir sample.

Two things it does deliberately and says so in the output. (b)'s aggregate is a recipe of this
script's own, applied to every arm alike, because `sotto_voce`'s seedtable used one that could
not be reproduced from its arm files; only WITHIN-table comparisons are valid there. And (g)
prints the banked mirror's row with a caveat rather than silently comparing it, because
`sotto_voce`'s own sample is front-loaded to cycles 50-54 while these tags' is a whole-run
reservoir.

Nothing here re-runs an arm; every number is read off a `results.json` already on disk. The
banked arms are never rewritten.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/mk_seedtable.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VO = os.path.dirname(os.path.dirname(HERE))          # .../practice/voicing
ROOTS = [f"{VO}/sotto_voce/aliquot/figures", f"{VO}/sotto_voce/figures", f"{VO}/figures"]

def load(tag, arm):
    for r in ROOTS:
        p = os.path.join(r, tag, arm, "results.json")
        if os.path.isfile(p):
            return json.load(open(p))
    return None

def critic_probe_auc(r):
    """median over slots of the median over cycles, for both the FILED and the WORLD target."""
    v = [c for c in r["log"]["vo"] if c]
    slots = sorted({s for c in v for s in (c.get("critic") or {})})
    out = {}
    for key in ("probe_auc", "probe_auc_world", "auc"):
        per = []
        for s in slots:
            seq = [c["critic"][s][key] for c in v
                   if (c.get("critic") or {}).get(s) and c["critic"][s].get(key) is not None]
            if seq:
                per.append(float(np.median(seq)))
        out[key] = (float(np.median(per)) if per else None, len(per))
    return out

def mirror_levels(r):
    mir = r.get("vo_mirror") or {}
    lev = {}
    for k, c in mir.items():
        L = int(k.split("s")[0][1:]); b = int(k.split("s")[1])
        d = lev.setdefault(L, {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "world_solved": 0,
                               "p_sum": 0.0, "byb": {}})
        for q in ("n", "tp", "fp", "fn", "tn", "world_solved"):
            d[q] += int(c[q])
        d["p_sum"] += float(c["p_sum"])
        d["byb"][b] = c
    return lev

def bill(r):
    b = r["log"].get("vo_bill") or []
    g = int(r.get("vo_probe_ground", 0) or 0)
    tot = sum(float(x["cycle_priced"]) for x in b) or 1.0
    return g, float(sum(float(x["probe_priced"]) for x in b)) / tot

def yoke(r):
    a = r.get("loop_actions") or []
    return (sum(1 for x in a if x["kind"] == "commit" and not x.get("cancelled")),
            sum(1 for x in a if x.get("cancelled")))

def om(r):
    return r.get("vo_om_state") or {}

SEEDS = {
    0: dict(tag="al_s1", anchor=("vo_s3", "voi3_dp"),
            floor=("vo_s3b", "voi3b_comp_yk"), ceil=("vo_s3b", "voi3b_comp_pr_yk"),
            mirror=("so_s1", "so_mg_yk"), comm=("so_s1", "so_cg_yk"), hyb=("so_s1", "so_hy_yk")),
    2: dict(tag="al_s2", anchor=("vo_s3d2", "voi3_dp"),
            floor=("vo_s3e", "voi3b_comp_yk"), ceil=("vo_s3e", "voi3b_comp_pr_yk"),
            mirror=("so_s2", "so_mg_yk"), comm=("so_s2", "so_cg_yk"), hyb=("so_s2", "so_hy_yk")),
}
ORDER = [("CEILING world-graded", "ceil"), ("hybrid (banked)", "hyb"),
         ("MIRROR model-graded", "mirror"), ("committee (banked)", "comm"),
         ("PROJECTION live trunk", "pj"), ("PROJECTION random trunk", "rt"),
         ("FLOOR filed diet", "floor"), ("anchor", "anchor")]

A = {}
for sd, spec in SEEDS.items():
    A[sd] = {}
    for nm, key in (("ceil", "ceil"), ("floor", "floor"), ("mirror", "mirror"),
                    ("comm", "comm"), ("hyb", "hyb"), ("anchor", "anchor")):
        A[sd][nm] = load(*spec[key])
    A[sd]["pj"] = load(spec["tag"], "al_pj_yk")
    A[sd]["rt"] = load(spec["tag"], "al_rt_yk")
    for k, v in A[sd].items():
        if v is None:
            print(f"MISSING seed {sd} {k}", file=sys.stderr)

L = []
def o(s=""):
    L.append(s); print(s)

o("=" * 100)
o("[aliquot] THE TWO-SEED TABLE — the verdict read off the world model's own state, seeds 0 and 2")
o("=" * 100)
o("  Every arm is the COMPOSED chooser with the probe channel on, clock-yoked to its seed's banked")
o("  anchor. The FLOOR is the composed chooser on the FILED diet, the CEILING the same chooser on")
o("  WORLD-graded probes, and the MIRROR / committee / hybrid are `sotto_voce`'s three arms. All of")
o("  those are banked and NOT re-run. The two new arms are the PROJECTION — a ridge-logistic")
o("  readout of `SN.trunk`'s pooled hiddens over the substituted configuration, per root — over the")
o("  LIVE trunk and over a NEVER-TRAINED trunk of the same architecture. THE SEEDS ARE NOT AVERAGED.")
o("")
o("  THE LADDERS, which set the denominators (the anchors', replayed by every arm):")
o("    seed 0  L2@c48  L3@c100  L4@c151  L5@c186   -> L4 AND L5 cells")
o("    seed 2  L2@c59  L3@c77   L4@c157  (no L5)   -> L4 cells ONLY")
o("")

# ---------------- (a) the chooser ------------------------------------------------------ #
o("-" * 100)
o("(a) THE CHOOSER — pooled L2/L3 repair accuracy on held-out rows, the cell `voicing` replicated")
o("    on every seed that produced those rungs. Parsed from each tag's own reduction, so the")
o("    denominators and the floor are the reducer's and not re-derived here.")
o("-" * 100)
FIGDIR = f"{VO}/sotto_voce/aliquot/figures"
for sd, spec in SEEDS.items():
    txt = open(os.path.join(FIGDIR, f"{spec['tag']}_reduction.txt")).read()
    i = txt.find("THE POOLED L2/L3 PANEL")
    j = txt.find("=" * 40, i)
    o(f"  seed {sd}")
    for line in txt[i:j].splitlines():
        if line.strip().startswith(("arm ", "vo_", "so_", "al_")):
            o("    " + line.strip())
    o("")
o("")

# ---------------- (b) the critic ------------------------------------------------------- #
o("-" * 100)
o("(b) THE CRITIC — held-out AUC on the PROBE diet, median over slots of the median over cycles.")
o("    `voicing` measured 0.491 for a critic fed only filed writes. `vsWORLD` is the number the")
o("    question is about: can the critic rank what the WORLD would have said, having been taught")
o("    by this grader. THE RECIPE IS MINE AND IS APPLIED TO EVERY ARM ALIKE — `sotto_voce`'s own")
o("    seedtable used an aggregate I could not reproduce exactly (it reports 0.667/0.617 for the")
o("    seed-0 mirror where this recipe gives 0.675/0.641), so compare WITHIN this table only.")
o("-" * 100)
o("  arm                        s0 vsFILED  s0 vsWORLD    s2 vsFILED  s2 vsWORLD    (filed-diet AUC s0/s2)")
CA = {sd: {k: (critic_probe_auc(A[sd][k]) if A[sd][k] else None) for k, _ in
           [(v, 0) for _, v in ORDER]} for sd in (0, 2)}
for label, k in ORDER:
    row = []
    for sd in (0, 2):
        c = CA[sd].get(k)
        if not c:
            row += ["   -  ", "   -  "]
            continue
        f_, w_ = c["probe_auc"][0], c["probe_auc_world"][0]
        row += [f"{f_:.3f}" if f_ is not None else "   -  ",
                f"{w_:.3f}" if w_ is not None else "   -  "]
    fd = []
    for sd in (0, 2):
        c = CA[sd].get(k)
        fd.append(f"{c['auc'][0]:.3f}" if c and c["auc"][0] is not None else "  -  ")
    o(f"  {label:26} {row[0]:>10}  {row[1]:>10}    {row[2]:>10}  {row[3]:>10}    "
      f"{fd[0]} / {fd[1]}")
o("")

# ---------------- (c) the grader against the world, per level -------------------------- #
o("-" * 100)
o("(c) THE GRADER AGAINST THE WORLD, per level, on the probes. `acc` is the thresholded verdict")
o("    against the world's; `world` is the world's own solve rate there. The world's verdict is")
o("    computed for EVERY probe on every arm as an instrument no arm consumes (gate M-1).")
o("-" * 100)
for sd in (0, 2):
    o(f"  seed {sd}")
    o("    arm                        level        n    world    acc      p̄    prec   recall")
    for label, k in (("MIRROR model-graded", "mirror"), ("PROJECTION live trunk", "pj"),
                     ("PROJECTION random trunk", "rt")):
        r = A[sd][k]
        if not r:
            continue
        lev = mirror_levels(r)
        for Lv in sorted(lev):
            c = lev[Lv]
            n = max(1, c["n"])
            pr = c["tp"] / max(1, c["tp"] + c["fp"])
            rc = c["tp"] / max(1, c["tp"] + c["fn"])
            o(f"    {label:26} L{Lv}   {c['n']:>8}  {c['world_solved']/n:.4f}  "
              f"{(c['tp']+c['tn'])/n:.4f}  {c['p_sum']/n:.4f}  {pr:.3f}   {rc:.3f}")
        o("")
o("")

# ---------------- (d) the blind region, per (level, in-support) ------------------------ #
o("-" * 100)
o("(d) THE BLIND REGION — the same table split by whether the SUBSTITUTED class is one the")
o("    learner has actually written at that slot (`sup=1`, at >= `vo_om_freq_share` of the")
o("    writes filed there) or not (`sup=0`). This is where the grader's reach is LOCATED.")
o("-" * 100)
for sd in (0, 2):
    o(f"  seed {sd}")
    o("    arm                        cell        n    world     acc      p̄")
    for label, k in (("MIRROR model-graded", "mirror"), ("PROJECTION live trunk", "pj"),
                     ("PROJECTION random trunk", "rt")):
        r = A[sd][k]
        if not r:
            continue
        mir = r.get("vo_mirror") or {}
        for key in sorted(mir, key=lambda z: (int(z.split("s")[0][1:]), int(z.split("s")[1]))):
            c = mir[key]
            n = max(1, int(c["n"]))
            o(f"    {label:26} {key:6} {int(c['n']):>8}  {c['world_solved']/n:.4f}  "
              f"{(c['tp']+c['tn'])/n:.4f}  {c['p_sum']/n:.4f}")
        o("")
o("")

# ---------------- (e) the readout itself ---------------------------------------------- #
o("-" * 100)
o("(e) THE READOUT — refits, the ridge, the held-out AUC on EXPERIENCE and its drift between")
o("    refits, the secondary variants at the same ridge on the same rows, and the trunk's own")
o("    fingerprint (sum of |parameter|) over the run.")
o("-" * 100)
o("  arm                        seed refits skip degen  ridge  nfeat  holdAUC   drift  |  add    mean  nomask | trunk range")
for sd in (0, 2):
    for label, k in (("PROJECTION live trunk", "pj"), ("PROJECTION random trunk", "rt")):
        r = A[sd][k]
        if not r:
            continue
        st = om(r)
        oms = [c for c in r["log"]["vo_om"] if c]
        au = [c["hold_auc"] for c in oms if c.get("hold_auc") is not None]
        dr = [abs(c["hold_auc_drift"]) for c in oms if c.get("hold_auc_drift") is not None]
        sig = [float(c["trunk_sig"]) for c in oms if c.get("trunk_sig") is not None]

        def lastof(key):
            z = [c[key] for c in oms if c.get(key) is not None]
            return z[-1] if z else None

        def f(x):
            return f"{x:.4f}" if x is not None else "  -   "
        o(f"  {label:26} {sd:>4} {int(st.get('n_refit') or 0):>6} "
          f"{int(st.get('refit_skipped') or 0):>4} {int(st.get('n_degenerate') or 0):>5} "
          f"{str(st.get('lam')):>6} {str(st.get('nfeat')):>6}  {f(au[-1] if au else None)} "
          f"{(np.mean(dr) if dr else float('nan')):.4f}  | {f(lastof('hold_auc_add'))} "
          f"{f(lastof('hold_auc_mean'))} {f(lastof('hold_auc_nomask'))} | "
          f"{(max(sig)-min(sig) if sig else float('nan')):.6f}")
o("")

# ---------------- (f) the bill and the yoke ------------------------------------------- #
o("-" * 100)
o("(f) THE BILL and THE CLOCK YOKE. The projection arms must bill ZERO — the world's verdict on")
o("    every probe is an instrument and nothing else (gates M-1, M-3).")
o("-" * 100)
o("  arm                        seed  probes billed  bill share of priced  commits  cancelled")
for sd in (0, 2):
    for label, k in ORDER:
        r = A[sd][k]
        if not r:
            continue
        g, sh = bill(r)
        nc, canc = yoke(r)
        o(f"  {label:26} {sd:>4}  {g:>13}  {sh*100:>19.4f}%  {nc:>7}  {canc:>9}")
    o("")

# ---------------- (g) the grader's own ranking of the world's verdict ------------------ #
def _auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    ordr = np.argsort(x, kind="mergesort"); rk = np.empty(len(x)); rk[ordr] = np.arange(1, len(x) + 1)
    xs = x[ordr]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            rk[ordr[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((rk[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

o("-" * 100)
o("(g) THE GRADER'S OWN RANKING OF THE WORLD'S VERDICT ON THE PROBES — AUC(p, yw) from the")
o("    per-probe instrument sample, per level and by in-support. This is the RANKING number;")
o("    (c)'s accuracy is a CALIBRATION number at a 0.05 base rate and the two can disagree.")
o("")
o("    ONE CAVEAT AND IT IS DECISIVE FOR THE COMPARISON. `sotto_voce`'s own README records that")
o("    its row-level sample filled FRONT-TO-BACK and holds cycles 50-54 only; reservoir sampling")
o("    shipped for later tags, which these are. So the banked mirror's row below is a reading of a")
o("    FIVE-CYCLE-OLD mirror over ~4 cycles of probes, and the projection's is a uniform sample of")
o("    the WHOLE run. The mirror row is printed for completeness and MUST NOT be compared to the")
o("    projection rows. The mirror-vs-projection comparison runs through (c)'s 2x2, which is")
o("    whole-run on every arm, and through (b)'s downstream critic AUC.")
o("-" * 100)
for sd in (0, 2):
    o(f"  seed {sd}")
    o("    arm                        cycles      n   AUC_all      L2         L3         L4         L5")
    for label, k in (("MIRROR model-graded", "mirror"), ("PROJECTION live trunk", "pj"),
                     ("PROJECTION random trunk", "rt")):
        r = A[sd][k]
        if not r:
            continue
        pr = r.get("vo_probe_rows") or []
        if not pr:
            o(f"    {label:26} (no per-probe sample)")
            continue
        cy = sorted({int(z["c"]) for z in pr})
        pp = np.array([z["p"] for z in pr]); yw = np.array([z["yw"] for z in pr])
        cells = []
        for Lv in (2, 3, 4, 5):
            m = np.array([int(z["L"]) == Lv for z in pr])
            if m.sum() < 16:
                cells.append("     -    ")
                continue
            av = _auc(pp[m], yw[m])
            cells.append(f"{av:.4f}({int(m.sum())})" if av is not None else "     -    ")
        aa = _auc(pp, yw)
        o(f"    {label:26} {cy[0]}-{cy[-1]:<4} {len(pr):>6}   "
          f"{(('%.4f' % aa) if aa is not None else '  -   '):>7}  " + " ".join(cells))
    o("    in-support split, pooled over levels (sup=1: the substituted class is one the learner")
    o("    writes at that slot):")
    for label, k in (("MIRROR model-graded", "mirror"), ("PROJECTION live trunk", "pj"),
                     ("PROJECTION random trunk", "rt")):
        r = A[sd][k]
        pr = (r or {}).get("vo_probe_rows") or []
        if not pr:
            continue
        pp = np.array([z["p"] for z in pr]); yw = np.array([z["yw"] for z in pr])
        sup = np.array([int(z["sup"]) for z in pr])
        z0 = _auc(pp[sup == 0], yw[sup == 0])
        z1 = _auc(pp[sup == 1], yw[sup == 1]) if (sup == 1).sum() >= 16 else None
        o(f"      {label:26} sup0 {('%.4f' % z0) if z0 else '  -   '} (n={int((sup==0).sum())})"
          f"   sup1 {('%.4f' % z1) if z1 else '  -   '} (n={int((sup==1).sum())})")
    o("")
o("")

out = f"{VO}/sotto_voce/aliquot/figures/al_seedtable.txt"
open(out, "w").write("\n".join(L) + "\n")
print(f"\n[wrote] {out}")
