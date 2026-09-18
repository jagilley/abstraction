"""[tessitura] THE ZERO-COST PASS — what the banked `results.json` files already answer.

`logit_reading/striatum/norm/` asked three things of an outcome-trained critic on a frozen
trunk: does its pre-event level calibrate to the world it was fed; at the event does it read
cost or structure; and on same-prefix twins how does it price the illegal token. This node asks
the same three of `voicing`'s judge, which is the same kind of object under the opposite
conditions (live learner, online training, priced feedback, a Q over candidates rather than a
state value).

THIS FILE IS THE WORLD SIDE ONLY, and that limit is the point of running it first. What every
arm banked per cycle per slot is `vo_critic_audit`'s record:

    n, auc, base_rate, probe_n, probe_auc, probe_base_rate          (+ `free`, `sh_auc`)

`base_rate` is the mean verdict over the held-out slice of the slot's ring buffer — THE WORLD,
as the critic was fed it, lagged by the buffer. The critic's own LEVEL (its mean predicted
P(solve)) is nowhere in the log: `vo_critic_terms` computes a per-slot `cstat` with the batch's
base rate in it and the call site discards it (`voicing.py` ~line 2248), and the audit reports
only a rank statistic. So the NORM is not in the banked record and no reduction can recover it.
What is here instead:

  [W] the world each arm was fed, per era and per slot — the outcome distribution that a norm
      would have to calibrate TO, across seeds, diets and graders;
  [E] the era ladder as a within-subject shift in expected cost, the thing norm's between-critic
      diets could not give us: the size of each step and how fast the filed stream settles;
  [B] the ring buffer's own lag — the audit's trailing base rate against the cycle's
      instantaneous one, which is the diet's time constant and therefore the floor any
      measurement of the judge's lag has to be read against;
  [A] the judge's ranking quality over time, filed and probe apart, with the free prior beside
      it where `ov_free` was on.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/reduce_logged.py
    python3 rhm/practice/voicing/tessitura/reduce_logged.py --arms ov_s0b:ovt_comp_pr_sh
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICING = os.path.dirname(HERE)

# (tag, arm, root of the tag's figure mirror) for every arm whose run trained a critic.
ROOTS = [os.path.join(VOICING, "figures"),
         os.path.join(VOICING, "sotto_voce", "figures")]

# how each arm was fed, for the tables. `diet` is what entered the critic's loss; `grader` is
# who answered the probe. Read off `FILES.md` and the two child READMEs, asserted against the
# arm's own config below (gate T-1).
DIET = {
    "voi2_critic": ("filed", "-", "replace"),
    "voi2_critic_xp": ("filed", "-", "replace"),
    "voi3_comp": ("filed", "-", "composed"),
    "voi3_comp_pr": ("filed", "-", "composed"),      # defect #8: probes filed, never trained on
    "voi3_rep_pr": ("filed", "-", "replace"),        # ditto
    "voi3b_comp_yk": ("filed", "-", "composed"),
    "voi3b_comp_pr_yk": ("filed+probe", "world", "composed"),
    "voi3b_rep_yk": ("filed", "-", "replace"),
    "voi3b_rep_pr_yk": ("filed+probe", "world", "replace"),
    "ovt_comp_pr_sh": ("filed+probe", "world", "composed"),
    "ovt_comp_pr_lin": ("filed+probe", "world", "composed"),
    "ovt_comp_pr_dis": ("filed+probe/dis", "world", "composed"),
    "so_mg_yk": ("filed+probe", "mirror", "composed"),
    "so_cg_yk": ("filed+probe", "committee", "composed"),
    "so_hy_yk": ("filed+probe", "hybrid", "composed"),
}


def find_arms():
    out = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for tag in sorted(os.listdir(root)):
            td = os.path.join(root, tag)
            if not os.path.isdir(td):
                continue
            for arm in sorted(os.listdir(td)):
                p = os.path.join(td, arm, "results.json")
                if os.path.isfile(p):
                    out.append((tag, arm, p))
    return out


def era_of_cycle(log):
    return np.asarray(log["era"], np.int64)


def load(p):
    d = json.load(open(p))
    return d


def per_cycle_world(v):
    """The cycle's OWN filed and probe solve rates — instantaneous, not the buffer's."""
    nf = np.array([(c or {}).get("n_filed", 0) for c in v], np.float64)
    ns = np.array([(c or {}).get("n_solved", 0) for c in v], np.float64)
    npb = np.array([(c or {}).get("n_probe", 0) for c in v], np.float64)
    nps = np.array([(c or {}).get("n_probe_solved", 0) for c in v], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nf > 0, ns / np.maximum(nf, 1), np.nan), \
            np.where(npb > 0, nps / np.maximum(npb, 1), np.nan), nf, npb


def audit_series(v):
    """Per cycle: the n-weighted mean of the audit's trailing base rates over the slots it read,
    filed and probe apart, plus the same for the AUCs."""
    out = {k: np.full(len(v), np.nan) for k in
           ("br", "pbr", "auc", "pauc", "dp", "pdp", "n", "pn", "nslot")}
    for i, c in enumerate(v):
        cr = (c or {}).get("critic") or {}
        if not cr:
            continue
        acc = collections.defaultdict(float)
        for _, q in cr.items():
            n = float(q.get("n") or 0)
            pn = float(q.get("probe_n") or 0)
            if n:
                acc["n"] += n
                if q.get("base_rate") is not None:
                    acc["br"] += n * float(q["base_rate"])
                if q.get("auc") is not None:
                    acc["auc"] += n * float(q["auc"])
                    acc["auc_n"] += n
                fr = (q.get("free") or {})
                if fr.get("dp_auc") is not None:
                    acc["dp"] += n * float(fr["dp_auc"])
                    acc["dp_n"] += n
            if pn:
                acc["pn"] += pn
                if q.get("probe_base_rate") is not None:
                    acc["pbr"] += pn * float(q["probe_base_rate"])
                if q.get("probe_auc") is not None:
                    acc["pauc"] += pn * float(q["probe_auc"])
                    acc["pauc_n"] += pn
                pfr = ((q.get("probe_x") or {}).get("free") or {})
                if pfr.get("dp_auc") is not None:
                    acc["pdp"] += pn * float(pfr["dp_auc"])
                    acc["pdp_n"] += pn
        out["nslot"][i] = len(cr)
        if acc["n"]:
            out["n"][i] = acc["n"]
            out["br"][i] = acc["br"] / acc["n"]
            if acc.get("auc_n"):
                out["auc"][i] = acc["auc"] / acc["auc_n"]
            if acc.get("dp_n"):
                out["dp"][i] = acc["dp"] / acc["dp_n"]
        if acc["pn"]:
            out["pn"][i] = acc["pn"]
            out["pbr"][i] = acc["pbr"] / acc["pn"]
            if acc.get("pauc_n"):
                out["pauc"][i] = acc["pauc"] / acc["pauc_n"]
            if acc.get("pdp_n"):
                out["pdp"][i] = acc["pdp"] / acc["pdp_n"]
    return out


def ema_lag(inst, trail, taus=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80)):
    """The effective window of the trailing series, by the best-fitting exponential moving
    average of the instantaneous one. Reported as a NUMBER OF CYCLES, with the residual, so a
    bad fit is visible rather than glossed. Not a claim about the critic — the critic's own
    level is not logged; this is the DIET's lag."""
    m = np.isfinite(inst) & np.isfinite(trail)
    if int(m.sum()) < 20:
        return None
    best = None
    for tau in taus:
        a = 1.0 / float(tau)
        e = np.nan
        run = None
        pred = np.full(len(inst), np.nan)
        for i in range(len(inst)):
            if not np.isfinite(inst[i]):
                continue
            run = inst[i] if run is None else (1 - a) * run + a * inst[i]
            pred[i] = run
        mm = m & np.isfinite(pred)
        if int(mm.sum()) < 20:
            continue
        e = float(np.sqrt(np.mean((pred[mm] - trail[mm]) ** 2)))
        if best is None or e < best[1]:
            best = (tau, e, int(mm.sum()))
    return best


def settle(inst, eras, era_id, frac=0.5):
    """After the step into `era_id`, how many cycles until the instantaneous filed rate first
    reaches `frac` of the way from its last-era level to its own new-era asymptote."""
    idx = np.nonzero(eras == era_id)[0]
    if len(idx) < 4:
        return None
    prev = np.nonzero(eras == era_id - 1)[0]
    if len(prev) < 4:
        return None
    a = np.nanmean(inst[prev[-min(10, len(prev)):]])
    b = np.nanmean(inst[idx[-min(10, len(idx)):]])
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-9:
        return (0, a, b)
    tgt = a + frac * (b - a)
    for k, i in enumerate(idx):
        if np.isfinite(inst[i]) and ((inst[i] >= tgt) if b > a else (inst[i] <= tgt)):
            return (k, a, b)
    return (len(idx), a, b)


def slot_table(v, eras, era_id):
    """Per slot, inside one era: n-weighted mean base rate, probe base rate, AUCs."""
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for i, c in enumerate(v):
        if eras[i] != era_id:
            continue
        for k, q in ((c or {}).get("critic") or {}).items():
            n, pn = float(q.get("n") or 0), float(q.get("probe_n") or 0)
            a = acc[k]
            if n:
                a["n"] += n
                a["br"] += n * float(q.get("base_rate") or 0.0)
                if q.get("auc") is not None:
                    a["auc"] += n * float(q["auc"]); a["auc_n"] += n
            if pn:
                a["pn"] += pn
                a["pbr"] += pn * float(q.get("probe_base_rate") or 0.0)
                if q.get("probe_auc") is not None:
                    a["pauc"] += pn * float(q["probe_auc"]); a["pauc_n"] += pn
    out = {}
    for k, a in acc.items():
        out[k] = {
            "n": a["n"], "br": (a["br"] / a["n"]) if a["n"] else None,
            "auc": (a["auc"] / a["auc_n"]) if a.get("auc_n") else None,
            "pn": a["pn"], "pbr": (a["pbr"] / a["pn"]) if a["pn"] else None,
            "pauc": (a["pauc"] / a["pauc_n"]) if a.get("pauc_n") else None}
    return out


def f(x, nd=3):
    return ("-" if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"{x:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="tag:arm,... (default: every arm with a critic)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "logged.txt"))
    a = ap.parse_args()

    want = {x for x in a.arms.split(",") if x}
    lines = []

    def o(s=""):
        lines.append(s)
        print(s)

    o("=" * 104)
    o("[tessitura] THE ZERO-COST PASS — the world the judge was fed, from the banked results.json")
    o("=" * 104)
    o("  The judge's own LEVEL is not in the banked record (see this file's header): the audit")
    o("  logs a rank statistic and the training call site discards the per-slot stat. Every")
    o("  number below is the WORLD — the outcome distribution a norm would have to calibrate to")
    o("  — plus the judge's ranking quality, which is not a level.")
    o("")

    rows = []
    for tag, arm, p in find_arms():
        if want and f"{tag}:{arm}" not in want:
            continue
        d = load(p)
        v = d["log"].get("vo") or []
        if not any((c or {}).get("critic") for c in v):
            continue
        cfg = d["config"]
        eras_cfg = d["eras"]
        eras = era_of_cycle(d["log"])
        inst, pinst, nf, npb = per_cycle_world(v)
        aud = audit_series(v)
        diet, grader, mode = DIET.get(arm, ("?", "?", "?"))
        # gate T-1: the table's diet column against the arm's own config.
        probe_on = bool(cfg.get("vo_probe"))
        assert (diet == "filed") == (not probe_on) or arm in ("voi3_comp_pr", "voi3_rep_pr"), \
            f"T-1: {tag}:{arm} diet column {diet!r} disagrees with vo_probe={probe_on}"
        gm = cfg.get("vo_govern_mode") or "replace"   # the knob postdates `vo_s2`
        assert mode in (gm, "?"), \
            f"T-1: {tag}:{arm} mode column {mode!r} != {gm!r}"
        rows.append(dict(tag=tag, arm=arm, seed=int(cfg.get("seed", -1)), diet=diet,
                         grader=grader, mode=mode, cfg=cfg, eras_cfg=eras_cfg, eras=eras,
                         inst=inst, pinst=pinst, nf=nf, npb=npb, aud=aud, v=v,
                         hidden=cfg.get("ov_critic_hidden")))

    # ---------------------------------------------------------------- [W] the worlds --- #
    o("-" * 104)
    o("[W] THE WORLD EACH ARM WAS FED — the cycle's own filed solve rate, pooled per era")
    o("-" * 104)
    o("    `filed` is the verdict on what the learner wrote (the outcome the critic is trained")
    o("    on); `probe` is the verdict on a substituted class at the same contexts. The era")
    o("    ladder's damage depth is the era index: era e corrupts at level e.")
    o("")
    o(f"    {'tag:arm':34} {'sd':>2} {'diet':14} {'grader':9} "
      + "  ".join(f"e{e:<5}" for e in range(1, 6)))
    for r in rows:
        cells = []
        for e in range(1, 6):
            m = (r["eras"] == e)
            n = r["nf"][m].sum()
            cells.append(f"{(np.nansum(r['inst'][m] * r['nf'][m]) / n):.3f} " if n else "  -   ")
        o(f"    {r['tag']+':'+r['arm']:34} {r['seed']:>2} {r['diet']:14} {r['grader']:9} "
          + " ".join(cells))
    o("")
    o(f"    PROBE solve rate (the substituted class), same layout")
    o(f"    {'tag:arm':34} {'sd':>2} {'diet':14} {'grader':9} "
      + "  ".join(f"e{e:<5}" for e in range(1, 6)))
    for r in rows:
        if not np.isfinite(r["pinst"]).any():
            continue
        cells = []
        for e in range(1, 6):
            m = (r["eras"] == e)
            n = r["npb"][m].sum()
            cells.append(f"{(np.nansum(r['pinst'][m] * r['npb'][m]) / n):.3f} " if n else "  -   ")
        o(f"    {r['tag']+':'+r['arm']:34} {r['seed']:>2} {r['diet']:14} {r['grader']:9} "
          + " ".join(cells))
    o("")

    # ------------------------------------------------------ [E] the within-subject step --- #
    o("-" * 104)
    o("[E] THE ERA LADDER AS A WITHIN-SUBJECT SHIFT IN EXPECTED COST")
    o("-" * 104)
    o("    norm's diets were BETWEEN critics: three critics, three worlds, one shared set of")
    o("    rows. Here one critic meets five worlds in sequence, which is Xiang's actual design.")
    o("    `step` is the change in the filed solve rate from the last 10 cycles of era e-1 to")
    o("    the last 10 of era e; `t50` is the cycles after the boundary at which the")
    o("    instantaneous rate first covers half of that step.")
    o("")
    o(f"    {'tag:arm':34} " + "  ".join(f"{'e'+str(e)+' step/t50':>16}" for e in range(2, 6)))
    for r in rows:
        cells = []
        for e in range(2, 6):
            st = settle(r["inst"], r["eras"], e)
            if st is None:
                cells.append(f"{'-':>16}")
            else:
                k, aa, bb = st
                cells.append(f"{bb-aa:>+9.3f}/{k:<5d}"[:16].rjust(16))
        o(f"    {r['tag']+':'+r['arm']:34} " + "  ".join(cells))
    o("")
    o("    era boundaries (cycle of first appearance) and era lengths, per arm:")
    for r in rows:
        bnd = [int(np.nonzero(r["eras"] == e)[0][0]) if (r["eras"] == e).any() else None
               for e in range(1, 6)]
        ln = [int((r["eras"] == e).sum()) for e in range(1, 6)]
        o(f"      {r['tag']+':'+r['arm']:34} starts {bnd}  lengths {ln}")
    o("")

    # ------------------------------------------------------- [B] the diet's own lag --- #
    o("-" * 104)
    o("[B] THE RING BUFFER'S OWN LAG — what the critic is shown, against what the world just did")
    o("-" * 104)
    o("    The audit's `base_rate` is the mean verdict over the held-out slice of a slot's ring")
    o("    buffer (cap 8192 rows per slot, the last 1024 held-out rows scored). It is therefore")
    o("    a TRAILING estimate of the world. Fitting an exponential moving average of the")
    o("    cycle's own rate to it gives the diet's effective window in cycles. ANY measurement")
    o("    of the judge's own lag has to be read against this floor: a judge that tracked its")
    o("    buffer instantly would still lag the world by this much.")
    o("")
    o("    TWO CAVEATS ON THE PROBE COLUMN, said before it is read. (i) The probe solve rate is")
    o("    nearly flat over a run, so its EMA fit is ill-conditioned and the tau it returns is")
    o("    not interpretable as a window; the filed column is the one to read. (ii) On the")
    o("    `sotto_voce` arms the probe buffer's `base_rate` is the MIRROR's mean predicted")
    o("    PROBABILITY, filed as a soft label (`sotto_voce.py` ~6168: `ym = pm.clone()`), not a")
    o("    verdict rate — a different object, and not comparable across the grader column. The")
    o("    WORLD's verdict on those same probes is the [W] table's probe row, which is computed")
    o("    on every probe as an instrument no arm consumes.")
    o("")
    o(f"    {'tag:arm':34} {'tau_filed':>10} {'rmse':>7} {'n':>5} | "
      f"{'tau_probe':>10} {'rmse':>7} {'n':>5} | {'corr(inst,trail)':>17}")
    for r in rows:
        bf = ema_lag(r["inst"], r["aud"]["br"])
        bp = ema_lag(r["pinst"], r["aud"]["pbr"])
        m = np.isfinite(r["inst"]) & np.isfinite(r["aud"]["br"])
        cc = (float(np.corrcoef(r["inst"][m], r["aud"]["br"][m])[0, 1])
              if int(m.sum()) > 10 else None)
        o(f"    {r['tag']+':'+r['arm']:34} "
          + (f"{bf[0]:>10d} {bf[1]:>7.4f} {bf[2]:>5d} | " if bf else f"{'-':>10} {'-':>7} {'-':>5} | ")
          + (f"{bp[0]:>10d} {bp[1]:>7.4f} {bp[2]:>5d} | " if bp else f"{'-':>10} {'-':>7} {'-':>5} | ")
          + f"{f(cc):>17}")
    o("")

    # ------------------------------------------------------ [A] ranking quality --- #
    o("-" * 104)
    o("[A] THE JUDGE'S RANKING QUALITY OVER TIME — filed and probe apart, per era")
    o("-" * 104)
    o("    n-weighted mean of the per-slot held-out AUCs inside each era. `dp` is the free")
    o("    prior's AUC on the same rows where `ov_free` was on. A rank statistic is NOT a level;")
    o("    this section is here because it is the only thing about the judge the log carries.")
    o("")
    for which, ka, kd in (("FILED", "auc", "dp"), ("PROBE", "pauc", "pdp")):
        o(f"    {which}")
        o(f"      {'tag:arm':34} " + "  ".join(f"{'e'+str(e):>13}" for e in range(1, 6)))
        for r in rows:
            cells = []
            for e in range(1, 6):
                m = (r["eras"] == e) & np.isfinite(r["aud"][ka])
                if not m.any():
                    cells.append(f"{'-':>13}")
                    continue
                w = r["aud"]["n" if which == "FILED" else "pn"][m]
                au = float(np.nansum(r["aud"][ka][m] * w) / np.nansum(w))
                md = (r["aud"][kd][m])
                dp = (float(np.nansum(md * w) / np.nansum(w))
                      if np.isfinite(md).any() else None)
                cells.append((f"{au:.3f}/{dp:.3f}" if dp is not None else f"{au:.3f}/  -  ").rjust(13))
            o(f"      {r['tag']+':'+r['arm']:34} " + "  ".join(cells))
        o("")

    # --------------------------------------------------- [S] per-slot worlds --- #
    o("-" * 104)
    o("[S] THE WORLD PER SLOT — the buffer's trailing base rate by slot, inside each era")
    o("-" * 104)
    o("    A slot is (macro level : node). This is the per-slot version of [W] and is the")
    o("    granularity a per-slot norm would calibrate at.")
    o("")
    for r in rows:
        o(f"    {r['tag']}:{r['arm']}  (seed {r['seed']}, {r['diet']}, {r['grader']})")
        allk = sorted({k for e in range(1, 6) for k in slot_table(r["v"], r["eras"], e)},
                      key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
        tabs = {e: slot_table(r["v"], r["eras"], e) for e in range(1, 6)}
        o("      slot  " + "  ".join(f"{'e'+str(e)+' br/pbr':>15}" for e in range(1, 6)))
        for k in allk:
            cells = []
            for e in range(1, 6):
                q = tabs[e].get(k)
                cells.append((f"{f(q['br'])}/{f(q['pbr'])}" if q else "-").rjust(15))
            o(f"      {k:5} " + "  ".join(cells))
        o("")

    out = a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
