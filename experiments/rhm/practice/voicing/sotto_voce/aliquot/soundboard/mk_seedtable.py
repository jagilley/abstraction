"""[soundboard] THE TWO-SEED TABLE OF RECORD — `aliquot/mk_seedtable.py` forked, its seven
sections kept and applied to the three shaped arms beside the banked frozen-plant projection, its
random twin, the mirror, the committee, the hybrid, the floor and the ceiling; plus three new ones:
(h) the shaping itself, (i) the world-model diagnostics over the run, (j) the yield currency, in
both currencies.

DONOR DOCSTRING FOLLOWS.

[aliquot] THE TWO-SEED TABLE OF RECORD — assembled from the banked and the new arm files.

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
VO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../practice/voicing
ROOTS = [f"{VO}/sotto_voce/aliquot/soundboard/figures", f"{VO}/sotto_voce/aliquot/figures",
         f"{VO}/sotto_voce/figures", f"{VO}/figures"]

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
    0: dict(tag="sb_s1", al="al_s1", anchor=("vo_s3", "voi3_dp"),
            floor=("vo_s3b", "voi3b_comp_yk"), ceil=("vo_s3b", "voi3b_comp_pr_yk"),
            mirror=("so_s1", "so_mg_yk"), comm=("so_s1", "so_cg_yk"), hyb=("so_s1", "so_hy_yk")),
    2: dict(tag="sb_s2", al="al_s2", anchor=("vo_s3d2", "voi3_dp"),
            floor=("vo_s3e", "voi3b_comp_yk"), ceil=("vo_s3e", "voi3b_comp_pr_yk"),
            mirror=("so_s2", "so_mg_yk"), comm=("so_s2", "so_cg_yk"), hyb=("so_s2", "so_hy_yk")),
}
# [soundboard] the three shaped arms sit BETWEEN the frozen-plant projection and its random twin,
# so the table reads top to bottom as "the world paid -> a net of its own -> the plant's own state,
# shaped -> the plant's own state, unshaped -> a never-trained plant's state -> nothing paid".
ORDER = [("CEILING world-graded", "ceil"), ("hybrid (banked)", "hyb"),
         ("MIRROR model-graded", "mirror"), ("committee (banked)", "comm"),
         ("SHAPED on the verdict", "sv"), ("SHAPED on next-level yield", "yd"),
         ("SHAPED verdict, no infill", "so"),
         ("PROJECTION live trunk (banked)", "pj"),
         ("PROJECTION random trunk (banked)", "rt"),
         ("FLOOR filed diet", "floor"), ("anchor", "anchor")]
SHAPED = [("SHAPED on the verdict", "sv"), ("SHAPED on next-level yield", "yd"),
          ("SHAPED verdict, no infill", "so")]
GRADERS = [("MIRROR model-graded", "mirror")] + SHAPED + [
    ("PROJECTION live trunk (banked)", "pj"), ("PROJECTION random trunk (banked)", "rt")]

A = {}
for sd, spec in SEEDS.items():
    A[sd] = {}
    for nm, key in (("ceil", "ceil"), ("floor", "floor"), ("mirror", "mirror"),
                    ("comm", "comm"), ("hyb", "hyb"), ("anchor", "anchor")):
        A[sd][nm] = load(*spec[key])
    A[sd]["pj"] = load(spec["al"], "al_pj_yk")          # banked, `aliquot`'s
    A[sd]["rt"] = load(spec["al"], "al_rt_yk")          # banked, `aliquot`'s
    A[sd]["sv"] = load(spec["tag"], "sb_sv_yk")
    A[sd]["yd"] = load(spec["tag"], "sb_yd_yk")
    A[sd]["so"] = load(spec["tag"], "sb_so_yk")
    for k, v in A[sd].items():
        if v is None:
            print(f"MISSING seed {sd} {k}", file=sys.stderr)

L = []
def o(s=""):
    L.append(s); print(s)

o("=" * 100)
o("[soundboard] THE TWO-SEED TABLE — the outcome error in the plant's own weights, seeds 0 and 2")
o("=" * 100)
o("  Every arm is the COMPOSED chooser with the probe channel on, clock-yoked to its seed's banked")
o("  anchor. The FLOOR is the composed chooser on the FILED diet, the CEILING the same chooser on")
o("  WORLD-graded probes, and the MIRROR / committee / hybrid are `sotto_voce`'s three arms. All of")
o("  those are banked and NOT re-run, and so are `aliquot`'s two PROJECTION arms — a ridge-logistic")
o("  readout of `SN.trunk`'s pooled hiddens over the substituted configuration, per root, over the")
o("  LIVE trunk and over a NEVER-TRAINED one. THE THREE NEW ARMS are that same readout in the same")
o("  seat with ONE thing added: an outcome head over the plant's pooled hiddens whose gradient")
o("  reaches the TRUNK in the same optimizer step as the plant's own infill training — on the")
o("  world's VERDICT, on NEXT-LEVEL YIELD from the learner's own miner, and on the verdict with the")
o("  plant's own infill term OFF. THE SEEDS ARE NOT AVERAGED.")
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
FIGDIR = f"{VO}/sotto_voce/aliquot/soundboard/figures"
for sd, spec in SEEDS.items():
    txt = open(os.path.join(FIGDIR, f"{spec['tag']}_reduction.txt")).read()
    i = txt.find("THE POOLED L2/L3 PANEL")
    j = txt.find("=" * 40, i)
    o(f"  seed {sd}")
    for line in txt[i:j].splitlines():
        if line.strip().startswith(("arm ", "vo_", "so_", "al_", "sb_")):
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
    for label, k in GRADERS:
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
    for label, k in GRADERS:
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
    for label, k in [z for z in GRADERS if z[1] != "mirror"]:
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
    for label, k in GRADERS:
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
    for label, k in GRADERS:
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

# ---------------- (h) the shaping itself ---------------------------------------------- #
def sbst(r):
    return (r or {}).get("sb") or {}

def sbrows(r):
    return [c for c in (((r or {}).get("log") or {}).get("sb") or []) if c]

o("-" * 100)
o("(h) THE SHAPING ITSELF — the outcome head's own within-batch BCE / AUC / base rate, the steps")
o("    it took, and the PLANT'S OWN FINGERPRINT (sum of |parameter|) over the run. The fingerprint")
o("    is the number gate S-1r reads: the outcome error reached the trunk iff this moved. The")
o("    banked `aliquot` arms predate the series and carry none, which is why gate S-4r — the")
o("    shaped plant against an UNSHAPED twin's, cycle by cycle — is read on the PREFLIGHT.")
o("-" * 100)
o("  arm                        seed target infill   steps     rows   BCE first/last  AUC last  base")
for sd in (0, 2):
    for label, k in SHAPED:
        r = A[sd][k]
        if not r:
            continue
        sb = sbst(r)
        shp = [c["shape"] for c in sbrows(r) if c.get("shape")]
        bce = [c["bce"] for c in shp if c.get("bce") is not None]
        auc = [c["auc"] for c in shp if c.get("auc") is not None]
        bas = [c["base"] for c in shp if c.get("base") is not None]
        o(f"  {label:26} {sd:>4} {str(sb.get('target')):6} "
          f"{('on' if sb.get('infill', True) else 'OFF'):6} "
          f"{(shp[-1]['n_step'] if shp else 0):>7} {(shp[-1]['n_rows'] if shp else 0):>8}   "
          f"{(bce[0] if bce else float('nan')):.4f} / {(bce[-1] if bce else float('nan')):.4f}"
          f"    {(auc[-1] if auc else float('nan')):.4f}  {(bas[-1] if bas else float('nan')):.4f}")
o("")
o("  the plant's own fingerprint, sampled over the run (range = max - min):")
for sd in (0, 2):
    for label, k in SHAPED:
        r = A[sd][k]
        sig = [(i, float(c["sig"])) for i, c in enumerate(sbrows(r))
               if c.get("sig") is not None]
        if not sig:
            continue
        cyc = [c for c in (r["log"]["cycle"])]
        step = max(1, len(sig) // 7)
        o(f"    {label:26} {sd:>4} range {max(x for _, x in sig) - min(x for _, x in sig):.6f}   "
          + "  ".join(f"{x:.2f}" for _, x in sig[::step]))
o("")

# ---------------- (i) the world-model diagnostics ------------------------------------- #
o("-" * 100)
o("(i) THE WORLD-MODEL DIAGNOSTICS — what the shaping cost the plant's PREDICTIVE competence, and")
o("    whether its grasp of the nested grammar survived. `infill` is the HELD-OUT masked-infill")
o("    cross-entropy and fill accuracy on FRESH corpus windows at a FIXED window/mask plan drawn")
o("    once per arm; `parse` is the block head's level-1 parse accuracy against the EXACT features,")
o("    read with one block masked (`parse_features`' idiom) and unmasked — an oracle instrument;")
o("    `dpL` is the nested max-sum DP over the TRUE table at that rung, span masked, recovered")
o("    level-1 features against the exact ones (`duplex` §4.3's altitude instrument), at the arm's")
o("    LAST read. `duplex`'s offline reference on a frozen trunk: infill_ce 1.418, parse_mask1")
o("    0.65, dpL2..L5 0.39/0.31/0.30/0.30; a never-trained trunk reads 2.176 / 0.194 /")
o("    0.15/0.12/0.11/0.11. Those are a DIFFERENT plant (the banked checkpoint) and are quoted as")
o("    a scale, never as this table's control — the control is the first cycle of each arm.")
o("-" * 100)
o("  arm                        seed  infill_ce f/l     infill_acc f/l   parse_mask1 f/l"
  "   parse_nomask f/l")
for sd in (0, 2):
    for label, k in SHAPED:
        r = A[sd][k]
        wm = [c["wm"] for c in sbrows(r) if c.get("wm")]
        if not wm:
            continue

        def fl(key, wm=wm):
            z = [w[key] for w in wm if w.get(key) is not None]
            return (z[0], z[-1]) if z else (float("nan"), float("nan"))
        ce, ac = fl("infill_ce"), fl("infill_acc")
        pm, pn = fl("parse_mask1"), fl("parse_nomask")
        o(f"  {label:26} {sd:>4}  {ce[0]:.4f} / {ce[1]:.4f}    {ac[0]:.3f} / {ac[1]:.3f}"
          f"      {pm[0]:.3f} / {pm[1]:.3f}       {pn[0]:.3f} / {pn[1]:.3f}")
o("")
o("  the nested DP parse, FIRST checkpoint and LAST (end of arm):")
o("  arm                        seed  when      L2      L3      L4      L5")
for sd in (0, 2):
    for label, k in SHAPED:
        r = A[sd][k]
        dps = [c["wm"]["dp_parse"] for c in sbrows(r)
               if c.get("wm") and c["wm"].get("dp_parse")]
        fin = (sbst(r).get("wm_final") or {}).get("dp_parse")
        for when, d in ([("first", dps[0])] if dps else []) + ([("final", fin)] if fin else []):
            o(f"  {label:26} {sd:>4}  {when:8}  " + "  ".join(
                f"{(d.get(str(L)) or {}).get('acc', float('nan')):.4f}" if d.get(str(L))
                else "   -   " for L in (2, 3, 4, 5)))
o("")
o("  the held-out infill CE by HOW MANY BLOCKS ARE MASKED, at the arm's last read:")
for sd in (0, 2):
    for label, k in SHAPED:
        wm = [c["wm"] for c in sbrows(A[sd][k]) if c.get("wm")]
        if not wm or not wm[-1].get("infill_by_nmask"):
            continue
        o(f"    {label:26} {sd:>4}  " + "  ".join(
            f"{kk}:{vv['ce']:.4f}" for kk, vv in sorted(wm[-1]["infill_by_nmask"].items())))
o("")

# ---------------- (j) the yield currency, and the projection scored in it -------------- #
o("-" * 100)
o("(j) THE YIELD CURRENCY. The top block is what the `yield` arm's head was TRAINED on: the share")
o("    of a solved configuration's level-(era+1) spans whose key is at support in the learner's own")
o("    miner, zero on an unsolved piece, computed at PUSH time from the miner's counts as they")
o("    stood then. The bottom block is the per-probe instrument, three columns, consumed by")
o("    nothing: `endo` is the reader's parse of the SUBSTITUTED configuration against the learner's")
o("    own miner (the currency the head was shaped on), `true` the same parse against the TRUE")
o("    table, `orc` the EXACT parse against the true table. AUC(p, endo>0) is the grader scored")
o("    against the currency the plant was shaped on; AUC(p, world) is the same grader scored")
o("    against the solve verdict it is FIT to. The two are the point of this section.")
o("-" * 100)
o("  arm                        seed  target levels     n_label   label mean   label > 0")
for sd in (0, 2):
    for label, k in SHAPED:
        r = A[sd][k]
        sb = sbst(r)
        if not sb.get("n_label"):
            continue
        lv = sorted({int(c["tgt_level"]) for c in sbrows(r) if c.get("tgt_level")})
        o(f"  {label:26} {sd:>4}  {str(lv):16} {int(sb['n_label']):>8}   "
          f"{(sb.get('label_mean') if sb.get('label_mean') is not None else float('nan')):.4f}"
          f"       {int(sb.get('label_pos') or 0)}")
o("")
o("  arm                        seed      n   endo   true    orc   AUC(p,endo>0)  AUC(p,world)")
for sd in (0, 2):
    for label, k in [z for z in GRADERS if z[1] != "mirror"]:
        r = A[sd][k]
        pr = [x for x in ((r or {}).get("vo_probe_rows") or []) if "yqe" in x]
        if not pr:
            continue
        p_ = np.asarray([x["p"] for x in pr], float)
        ye = np.asarray([x["yqe"] for x in pr], float)
        yt = np.asarray([x["yqt"] for x in pr], float)
        yo = np.asarray([x["yqo"] for x in pr], float)
        yw = np.asarray([x["yw"] for x in pr], float)
        ae, aw = _auc(p_, (ye > 0).astype(float)), _auc(p_, yw)
        o(f"  {label:26} {sd:>4} {len(pr):>6}  {ye.mean():.4f} {yt.mean():.4f} {yo.mean():.4f}"
          f"   {(ae if ae is not None else float('nan')):>11.4f}   "
          f"{(aw if aw is not None else float('nan')):>11.4f}")
o("")
o("  the same per LEVEL (AUC(p, endo>0); the currency's own level moves with the era, so a")
o("  per-level split is the only honest cut):")
for sd in (0, 2):
    for label, k in [z for z in GRADERS if z[1] != "mirror"]:
        pr = [x for x in ((A[sd][k] or {}).get("vo_probe_rows") or []) if "yqe" in x]
        if not pr:
            continue
        cells = []
        for Lv in (2, 3, 4, 5):
            sel = [x for x in pr if int(x["L"]) == Lv]
            if len(sel) < 20:
                cells.append(f"L{Lv}:   -    ")
                continue
            av = _auc(np.asarray([x["p"] for x in sel], float),
                      np.asarray([(x["yqe"] > 0) for x in sel], float))
            cells.append(f"L{Lv}:{(av if av is not None else float('nan')):.3f}({len(sel)})")
        o(f"    {label:26} {sd:>4}  " + "  ".join(cells))
o("")

# ---------------- (k) the readout's LEVEL and the duplicate share ---------------------- #
o("-" * 100)
o("(k) THE READOUT'S LEVEL, beside its ranking, on the bank's HELD-OUT filed rows — because")
o("    `duplex` found that most of what shaping buys on filed rows is CONTEXT rather than")
o("    candidate, and that is a statement about the LEVEL. And THE CROSS-CYCLE DUPLICATE SHARE,")
o("    the channel `duplex` §2.6 named and could not size: the bank's hold key is a COIN FLIP")
o("    against a per-cycle dedup, so the same (configuration, root) recurring later straddles the")
o("    split carrying the same verdict. The draw is KEPT for comparability with the banked arms;")
o("    what is new is the number. `aliquot`'s banked arms carry no such column.")
o("-" * 100)
o("  arm                        seed      p̄    base   Brier     ECE     AUC | dup share  same y")
for sd in (0, 2):
    for label, k in [z for z in GRADERS if z[1] != "mirror"]:
        r = A[sd][k]
        if not r:
            continue
        oms = [c for c in r["log"]["vo_om"] if c]
        st = om(r)

        def lk(key, oms=oms):
            z = [c[key] for c in oms if c.get(key) is not None]
            return z[-1] if z else None

        def g(x):
            return f"{x:.4f}" if x is not None else "  -   "
        o(f"  {label:26} {sd:>4}  {g(lk('hold_pbar'))} {g(lk('hold_base'))} "
          f"{g(lk('hold_brier'))}  {g(lk('hold_ece'))}  {g(lk('hold_auc'))} | "
          f"{g(st.get('dup_share'))}     {g(st.get('dup_same_y'))}")
o("")

out = f"{VO}/sotto_voce/aliquot/soundboard/figures/sb_seedtable.txt"
open(out, "w").write("\n".join(L) + "\n")
print(f"\n[wrote] {out}")
