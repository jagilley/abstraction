"""Post-hoc reduction of a dispositional run (CPU, local -- no Modal, no GPU).

    modal volume get rhm-scaling-data \
        /rhm_confabulation/v16_s2_L6_m4_distinct/dispositional/main/main_summary.json .
    python3 -m a2a_forward.fsm_part3.dispositional_grammar.analyze_dispositional \
        --summary main_summary.json --tag main

Writes `figures/<tag>_reduction.txt` (every table) and `figures/<tag>_*.png`.
Reduction only: no interpretation is written here.
"""

import argparse
import json
import os

TARGETS = ["IMPL_PRO", "IMPL_RETRO", "IMPL_NOW", "REORG_PRO", "REORG_RETRO",
           "BEHAV_PRO", "BEHAV_RETRO", "ENT_NOW"]
FAMILY = {"IMPL_PRO": "IMPL", "IMPL_RETRO": "IMPL", "IMPL_NOW": "IMPL",
          "REORG_PRO": "IMPL", "REORG_RETRO": "IMPL",
          "BEHAV_PRO": "BEHAV", "BEHAV_RETRO": "BEHAV", "ENT_NOW": "BEHAV"}
DIRN = {"IMPL_PRO": "prospective", "REORG_PRO": "prospective", "BEHAV_PRO": "prospective",
        "IMPL_RETRO": "retrospective", "REORG_RETRO": "retrospective",
        "BEHAV_RETRO": "retrospective", "IMPL_NOW": "occurrent", "ENT_NOW": "occurrent"}


def fnum(x, w=7, p=3, sign=False):
    try:
        if x is None or x != x:
            return " " * (w - 3) + "nan"
        return f"{x:{'+' if sign else ''}{w}.{p}f}"
    except Exception:
        return " " * (w - 1) + "-"


def reduce_run(S, tag, outdir):
    L = []
    w = L.append
    k = S["targets"]["horizon"]
    sched = S["schedule"]
    w(f"DISPOSITIONAL BATTERY -- reduction   tag={tag}")
    w(f"substrate ntp_aux_cl  v16_s2_L6_m4_distinct  8L/8H/256D")
    w(f"checkpoints ({len(sched)}): {sched}")
    w(f"horizon k = {k} checkpoint indices "
      f"(geometric schedule: a backward window spans the same step ratio as a forward one)")
    w(f"target checkpoints: {S['targets']['tc_steps']}")
    w(f"held-out checkpoint window: {S['targets']['held_steps']}")
    w("")

    # ---------- harness validation ----------
    w("=" * 100)
    w("1. HARNESS VALIDATION")
    w("=" * 100)
    wk = S.get("wake", {})
    fv = wk.get("final_val")
    if fv is None and S.get("control"):
        fv = list(S["control"].values())[0].get("val")   # the coordinator may have been
        # restarted after the wake, in which case the resumed call returns no val
    w(f"  final-checkpoint val loss (this run): {fnum(fv, 8, 4)}"
      f"   published reference for ntp_aux_cl: 2.3829")
    if S.get("control"):
        w(f"  control-battery val (recomputed at the final checkpoint): "
          f"{fnum(list(S['control'].values())[0].get('val'), 8, 4)}")
    w("")
    w("  occurrent battery on THIS model's final checkpoint (paper 2's rows):")
    w(f"    {'target':8s} {'self':>7s} {'bestO_io':>9s} {'advantage':>10s} {'base':>7s}"
      f"   published advantage")
    pub = {"IMPL": "+0.096", "BEHAV": "-0.001", "ENT": "-0.035", "WORLD": "-0.038"}
    for t in ("IMPL", "BEHAV", "ENT", "WORLD"):
        c = S.get("control", {}).get(t)
        if not c:
            continue
        w(f"    {t:8s} {fnum(c['self'])} {fnum(c['best_O_io'], 9)} "
          f"{fnum(c['advantage'], 10, sign=True)} {fnum(c['baseline'])}"
          f"   {pub[t]}")
    for t in ("IMPL",):
        c = S.get("control", {}).get(t)
        if c:
            w(f"    ladder/{t}: " + "  ".join(f"{kk}={vv:.3f}"
                                              for kk, vv in c["observers"].items()))
    w("")

    # ---------- guards ----------
    w("=" * 100)
    w("2. INSTRUMENT GUARDS ALONG THE TRAJECTORY  (a saturated instrument counterfeits")
    w("   the headline: read no IMPL number without these)")
    w("=" * 100)
    for cap in sorted({c for mm in S["instruments"].values() for c in mm["caps"]}):
        w(f"  instrument {cap}  ({S['instruments'][str(sched[0])]['caps'][cap]['fm_params']/1e3:.0f}K params)")
        w(f"    {'step':>7s} {'cos':>7s} {'|r|':>7s} {'ens_cos':>8s} {'d6 eta2':>8s} "
          f"{'d5 eta2':>8s} {'d4 eta2':>8s} {'rule d6':>8s}")
        for st in sched:
            c = S["instruments"][str(st)]["caps"][cap]
            e, ru = c["eta2_reference_last_token"], c["rule_eta2_reference"]
            w(f"    {st:7d} {fnum(c['fwd_cosine'])} {fnum(c['res_norm'])} "
              f"{fnum(c['ens_cos'], 8)} {fnum(e.get('d6'), 8)} {fnum(e.get('d5'), 8)} "
              f"{fnum(e.get('d4'), 8)} {fnum(ru.get('d6'), 8)}")
        w("")
    w("  reference band from the parent battery at the final checkpoint: cosine 0.85-0.90,")
    w("  ens_cos 0.87-0.91, d6 eta2 0.31-0.33 (ntp_aux_cl).  Junk-regime signature: ")
    w("  ens_cos 0.65 with d6 eta2 0.003 (the 300-step smoke artifact).")
    w("")

    # ---------- what the model learns, by level ----------
    w("=" * 100)
    w("3. WHICH LEVELS HAVE DYNAMICS AT ALL  (mean per-position NLL by hierarchy level)")
    w("=" * 100)
    lt = S["targets"].get("level_trajectory", {})
    if lt:
        w(f"    {'step':>7s} " + " ".join(f"{'L'+str(l):>8s}" for l in range(6)))
        for st in sched:
            r = lt.get(str(st), {}).get("mean_nll_by_level", {})
            w(f"    {st:7d} " + " ".join(fnum(r.get(str(l), r.get(l)), 8) for l in range(6)))
        first, last = lt.get(str(sched[0]), {}), lt.get(str(sched[-1]), {})
        if first and last:
            w("")
            w(f"    {'delta':>7s} " + " ".join(
                fnum((last['mean_nll_by_level'].get(str(l), last['mean_nll_by_level'].get(l))
                      - first['mean_nll_by_level'].get(str(l), first['mean_nll_by_level'].get(l))),
                     8, sign=True) for l in range(6)))
    w("")

    # ---------- the autocorrelation gate ----------
    w("=" * 100)
    w("4. THE AUTOCORRELATION GATE  (run BEFORE any head: probe-able does not imply")
    w("   learnable from a given target -- committee_head Phase B, lprog ac1 = +0.007)")
    w("=" * 100)
    for cap, cc in S["targets"]["caps"].items():
        w(f"  instrument {cap}")
        w("    disjoint-window autocorrelation (lag k+1, no shared term) is the statistic")
        w("    the horizon rule reads; lag 1 is shown for the record and is biased")
        w("    negative at k=1 because adjacent windows share an endpoint with opposite sign.")
        w(f"    {'k':>3s} " + " ".join(f"{nm:>26s}" for nm in
                                       ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO")))
        w(f"    {'':>3s} " + " ".join(f"{'disjoint / lag1':>26s}" for _ in range(4)))
        for kk in sorted(cc["gate"], key=lambda z: int(z[1:])):
            row = []
            for nm in ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO"):
                g = cc["gate"][kk][nm]
                row.append(f"{g.get('autocorr_disjoint_within', float('nan')):+.3f} / "
                           f"{g['autocorr_lag1_within']:+.3f}".rjust(26))
            w(f"    {kk[1:]:>3s} " + " ".join(row))
        w("")
        w("    IMPL class persistence P(class_c == class_{c+k}) vs chance:")
        for kk in sorted(cc["class_persistence"], key=lambda z: int(z[1:])):
            p = cc["class_persistence"][kk]
            w(f"      k={kk[1:]}: {p['persistence']:.3f}  (chance {p['chance']:.3f})")
        w(f"    instrument reliability of the target:")
        w(f"      IMPL class agreement (FM seed0 vs seed1, one codebook): "
          f"{fnum(cc['cls_instrument_agreement'])}")
        for nm, x in cc.get("reorg_instrument_reliability", {}).items():
            w(f"      REORG {nm} r(seed0, seed1) = {fnum(x, sign=True)}")
        w("")

    # ---------- the confound ----------
    w("=" * 100)
    w("5. THE LEVEL / CHECKPOINT / POSITION CONFOUND IN THE TARGETS")
    w("   (categorical targets are quantile-binned WITHIN (level, checkpoint), so the")
    w("    bins carry no level or checkpoint information by construction; the pooled-")
    w("    quantile variants are kept below for the record)")
    w("=" * 100)
    for cap, cc in S["targets"]["caps"].items():
        w(f"  instrument {cap}")
        for nm, d in cc["target_confounds"].items():
            if "between_level_var_share" in d:
                w(f"    {nm:20s} var share  level={d['between_level_var_share']:.3f}  "
                  f"position={d['between_position_var_share']:.3f}  "
                  f"checkpoint={d['between_checkpoint_var_share']:.3f}   "
                  f"| within-bin base={d['bin_base_majority']:.3f} "
                  f"(position-conditional {d['bin_base_position']:.3f})  "
                  f"| pooled-bin base={d['binpooled_base_majority']:.3f} "
                  f"(level-conditional {d['binpooled_base_level']:.3f})")
            else:
                w(f"    {nm:20s} base maj={d['base_majority']:.3f}  "
                  f"level-conditional={d['base_level']:.3f}  "
                  f"checkpoint-conditional={d['base_checkpoint']:.3f}  "
                  f"position-conditional={d['base_position']:.3f}")
        w("")

    # ---------- headline ----------
    cells = S["cells"]
    w("=" * 100)
    w("6. HEADLINE -- ADVANTAGE PER TARGET PER DIRECTION")
    w("   advantage = self-report - best capacity-matched O_io (tokens + M_c's logits)")
    w("   held_seq  = held-out sequences at trained checkpoints")
    w("   held_both = held-out sequences at a HELD-OUT checkpoint window")
    w("=" * 100)
    w(f"  {'cell':16s} {'fact':6s} {'time':14s} {'self':>7s} {'confab':>7s} {'bestOio':>8s} "
      f"{'ADV':>8s} {'ADV(ck)':>8s} {'O_hist':>8s} {'O_act':>8s} {'base':>7s} {'posbase':>8s}")
    for t in TARGETS:
        c = cells.get(t)
        if not c:
            continue
        obs = c["observers"]
        bio = max((vv["held_seq"] for kk, vv in obs.items() if kk.startswith("O_io@")),
                  default=float("nan"))
        w(f"  {t:16s} {FAMILY[t]:6s} {DIRN[t]:14s} {fnum(c['self']['held_seq'])} "
          f"{fnum(c.get('confabulator', {}).get('held_seq'))} {fnum(bio, 8)} "
          f"{fnum(c['advantage']['held_seq'], 8, sign=True)} "
          f"{fnum(c['advantage']['held_both'], 8, sign=True)} "
          f"{fnum(obs.get('O_hist@8:256', {}).get('held_seq'), 8)} "
          f"{fnum(obs.get('O_act@8:256', {}).get('held_seq'), 8)} "
          f"{fnum(c['baselines']['majority']['held_seq'])} "
          f"{fnum(c['baselines']['position_majority']['held_seq'], 8)}")
    for t in TARGETS:
        c = cells.get(t + "_h4m0.25")
        if c:
            obs = c["observers"]
            bio = max((vv["held_seq"] for kk, vv in obs.items() if kk.startswith("O_io@")),
                      default=float("nan"))
            w(f"  {t + ' @h4m0.25':16s} {FAMILY[t]:6s} {DIRN[t]:14s} "
              f"{fnum(c['self']['held_seq'])} {fnum(c.get('confabulator', {}).get('held_seq'))} "
              f"{fnum(bio, 8)} {fnum(c['advantage']['held_seq'], 8, sign=True)} "
              f"{fnum(c['advantage']['held_both'], 8, sign=True)} "
              f"{fnum(obs.get('O_hist@8:256', {}).get('held_seq'), 8)} "
              f"{fnum(obs.get('O_act@8:256', {}).get('held_seq'), 8)} "
              f"{fnum(c['baselines']['majority']['held_seq'])} "
              f"{fnum(c['baselines']['position_majority']['held_seq'], 8)}")
    w("")

    # ---------- within level ----------
    w("=" * 100)
    w("7. THE SAME, WITHIN HIERARCHY LEVEL  (age of acquisition ~ level, and level is an")
    w("   input fact any observer reads off the token index)")
    w("=" * 100)
    w(f"  {'cell':16s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6)))
    for t in TARGETS + [t + "_h4m0.25" for t in TARGETS]:
        c = cells.get(t)
        if not c:
            continue
        abl = c.get("advantage_by_level", {})
        w(f"  {t:16s} " + " ".join(
            fnum(abl.get(str(l), abl.get(l)), 9, sign=True) for l in range(6)))
    w("")
    w("  self-report accuracy by level:")
    w(f"  {'cell':16s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6)))
    for t in TARGETS:
        c = cells.get(t)
        if not c:
            continue
        d = c["self"]["held_seq_by_level"]
        w(f"  {t:16s} " + " ".join(fnum(d.get(str(l), d.get(l)), 9) for l in range(6)))
    w("")

    # ---------- ladders ----------
    w("=" * 100)
    w("8. THE OBSERVER LADDER  (flat in capacity = access-limited, not resource-limited)")
    w("=" * 100)
    caps_order = ["1:64", "2:128", "4:192", "8:256"]
    for t in TARGETS + [t + "_h4m0.25" for t in TARGETS]:
        c = cells.get(t)
        if not c:
            continue
        obs = c["observers"]
        w(f"  {t}   (self = {c['self']['held_seq']:.3f})")
        for row in ("O_input", "O_io"):
            vals = [obs.get(f"{row}@{cp}", {}).get("held_seq") for cp in caps_order]
            w(f"    {row:10s} " + " ".join(fnum(x, 9) for x in vals))
        for extra in ("O_io_half@8:256", "O_hist@8:256", "O_act@8:256", "O_acthist@8:256"):
            if extra in obs:
                w(f"    {extra:16s} {fnum(obs[extra]['held_seq'], 9)}   "
                  f"(ckpt-held {fnum(obs[extra]['held_both'], 9)})")
        w("")

    # ---------- continuous ----------
    w("=" * 100)
    w("9. CONTINUOUS READOUT  (Spearman rho against the within-(level,checkpoint)")
    w("   z-scored target; secondary to the categorical battery)")
    w("=" * 100)
    w(f"  {'cell':16s} {'self':>8s} {'O_io':>8s} {'O_hist':>8s} {'O_act':>8s} {'ADV':>8s}")
    for t in TARGETS:
        c = cells.get(t)
        if not c or "cont" not in c:
            continue
        d = c["cont"]
        adv = d["self"]["held_seq"] - d.get("O_io@8:256", {}).get("held_seq", float("nan"))
        w(f"  {t:16s} {fnum(d['self']['held_seq'], 8)} "
          f"{fnum(d.get('O_io@8:256', {}).get('held_seq'), 8)} "
          f"{fnum(d.get('O_hist@8:256', {}).get('held_seq'), 8)} "
          f"{fnum(d.get('O_act@8:256', {}).get('held_seq'), 8)} {fnum(adv, 8, sign=True)}")
    w("")
    txt = "\n".join(L)
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{tag}_reduction.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return txt


def make_figures(S, tag, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    sched = S["schedule"]
    cells = S["cells"]

    # Fig 1 -- guards along the trajectory
    caps = sorted({c for mm in S["instruments"].values() for c in mm["caps"]})
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for cap in caps:
        cos = [S["instruments"][str(st)]["caps"][cap]["fwd_cosine"] for st in sched]
        ens = [S["instruments"][str(st)]["caps"][cap]["ens_cos"] for st in sched]
        e6 = [S["instruments"][str(st)]["caps"][cap]["eta2_reference_last_token"].get("d6")
              for st in sched]
        ax[0].plot(sched, cos, "o-", label=cap)
        ax[1].plot(sched, ens, "o-", label=cap)
        ax[2].plot(sched, e6, "o-", label=cap)
    for a, ttl, hl in ((ax[0], "FM cosine (saturation)", 0.99),
                       (ax[1], "ens_cos (junk discriminator)", 0.65),
                       (ax[2], "d6 hierarchy $\\eta^2$ (structure)", 0.003)):
        a.set_xscale("log"); a.set_xlabel("training step"); a.set_title(ttl)
        a.axhline(hl, color="crimson", ls=":", lw=1)
        a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle("Instrument guards along M's training trajectory", y=1.02)
    fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_guards.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # Fig 2 -- per-level loss trajectory
    lt = S["targets"].get("level_trajectory", {})
    if lt:
        fig, a = plt.subplots(figsize=(5.4, 3.8))
        for l in range(6):
            a.plot(sched, [lt[str(st)]["mean_nll_by_level"].get(str(l),
                           lt[str(st)]["mean_nll_by_level"].get(l)) for st in sched],
                   "o-", label=f"L{l}")
        a.set_xscale("log"); a.set_xlabel("training step"); a.set_ylabel("mean NLL")
        a.set_title("What M learns, and when (by hierarchy level)")
        a.legend(fontsize=8); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_level_trajectory.png", dpi=150)
        plt.close(fig)

    # Fig 3 -- advantage per target per direction
    order = [t for t in TARGETS if t in cells]
    if order:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        x = np.arange(len(order))
        for i, (kk, ttl) in enumerate((("held_seq", "held-out sequences"),
                                       ("held_both", "held-out sequences x held-out checkpoints"))):
            vals = [cells[t]["advantage"][kk] for t in order]
            col = ["#2b6cb0" if FAMILY[t] == "IMPL" else "#b7791f" for t in order]
            ax[i].bar(x, vals, color=col)
            ax[i].axhline(0, color="k", lw=1)
            ax[i].set_xticks(x); ax[i].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
            ax[i].set_ylabel("advantage (self - best $O_{io}$)")
            ax[i].set_title(ttl); ax[i].grid(alpha=.3, axis="y")
        fig.suptitle("First-person advantage on dispositional targets "
                     "(blue = implementation, gold = behaviour)", y=1.02)
        fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_advantage.png", dpi=150,
                                        bbox_inches="tight"); plt.close(fig)

        # Fig 4 -- within level
        fig, a = plt.subplots(figsize=(8, 4))
        wdt = 0.8 / max(1, len(order))
        for j, t in enumerate(order):
            abl = cells[t].get("advantage_by_level", {})
            ls = sorted(int(z) for z in abl)
            a.bar(np.array(ls) + j * wdt - 0.4, [abl[str(l)] if str(l) in abl else abl[l]
                                                 for l in ls], width=wdt, label=t)
        a.axhline(0, color="k", lw=1); a.set_xlabel("hierarchy level")
        a.set_ylabel("advantage"); a.legend(fontsize=7, ncol=2); a.grid(alpha=.3, axis="y")
        a.set_title("Advantage within hierarchy level")
        fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_advantage_by_level.png", dpi=150)
        plt.close(fig)

        # Fig 5 -- observer ladders
        caps_order = ["1:64", "2:128", "4:192", "8:256"]
        xs = [64, 128, 192, 256]
        n = len(order)
        fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3.4), sharey=False)
        axs = np.atleast_1d(axs)
        for j, t in enumerate(order):
            obs = cells[t]["observers"]
            for row, mk in (("O_input", "s--"), ("O_io", "o-")):
                ys = [obs.get(f"{row}@{cp}", {}).get("held_seq", np.nan) for cp in caps_order]
                axs[j].plot(xs, ys, mk, label=row, ms=4)
            axs[j].axhline(cells[t]["self"]["held_seq"], color="crimson", lw=1.4, label="self")
            if "O_act@8:256" in obs:
                axs[j].axhline(obs["O_act@8:256"]["held_seq"], color="green", ls=":", lw=1.2,
                               label="O_act")
            if "O_hist@8:256" in obs:
                axs[j].axhline(obs["O_hist@8:256"]["held_seq"], color="purple", ls="-.", lw=1.2,
                               label="O_hist")
            axs[j].set_title(t, fontsize=8); axs[j].set_xlabel("observer width")
            axs[j].grid(alpha=.3)
            if j == 0:
                axs[j].set_ylabel("accuracy"); axs[j].legend(fontsize=6)
        fig.suptitle("Observer ladders", y=1.03)
        fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_ladders.png", dpi=150,
                                        bbox_inches="tight"); plt.close(fig)

    # Fig 6 -- the autocorrelation gate
    tc = S["targets"]["caps"]
    cap0 = list(tc)[0]
    gate = tc[cap0]["gate"]
    ks = sorted(gate, key=lambda z: int(z[1:]))
    nms = ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO")
    fig, a = plt.subplots(figsize=(6, 3.6))
    for nm in nms:
        a.plot([int(z[1:]) for z in ks],
               [gate[z][nm].get("autocorr_disjoint_within", float("nan")) for z in ks],
               "o-", label=nm)
        a.plot([int(z[1:]) for z in ks],
               [gate[z][nm]["autocorr_lag1_within"] for z in ks], "s:", alpha=.4)
    a.axhline(0, color="k", lw=1)
    a.set_xlabel("horizon k (checkpoint indices)")
    a.set_ylabel("autocorrelation, within (checkpoint, level)\n(solid: disjoint windows; dotted: lag 1)")
    a.set_title(f"The gate: is the target learnable at this resolution? ({cap0})")
    a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/{tag}_gate.png", dpi=150); plt.close(fig)
    print(f"figures -> {outdir}")


# ======================================================================
# Cross-arm / cross-horizon comparison
# ======================================================================

def compare(runs, outdir, name="compare"):
    """runs: list of (label, summary dict). Emits one combined reduction over arms and
    horizons: harness validation, headline advantage, within-level advantage, the gate,
    the guards, and the continuous readout, all side by side."""
    L = []
    w = L.append
    w(f"DISPOSITIONAL BATTERY -- cross-arm reduction   arms: "
      + ", ".join(lb for lb, _ in runs))
    w("")
    w("=" * 110)
    w("A. HARNESS VALIDATION PER ARM  (the occurrent battery at each arm's final checkpoint)")
    w("=" * 110)
    w(f"  {'arm':10s} {'val':>8s} {'ref':>8s} | " +
      " ".join(f"{t:>18s}" for t in ("IMPL", "BEHAV", "ENT", "WORLD")))
    seen = set()
    for lb, S in runs:
        arm = S.get("arm") or lb.split("-")[0]
        if arm in seen or not S.get("control"):
            continue
        seen.add(arm)
        ref = "1.5447" if arm == "ntp_aux" else "2.3829"
        val = list(S["control"].values())[0].get("val")
        cells = []
        for t in ("IMPL", "BEHAV", "ENT", "WORLD"):
            c = S["control"].get(t)
            cells.append((f"{c['self']:.3f}/{c['advantage']:+.3f}" if c else "-").rjust(18))
        w(f"  {arm:10s} {fnum(val, 8, 4)} {ref:>8s} | " + " ".join(cells))
    w("  cells are self / advantage.  published: CL IMPL +0.096 BEHAV -0.001 ENT -0.035")
    w("  WORLD -0.038 at val 2.3829;  OL IMPL ~+0.06 with the three controls at ~0 at val 1.5447")
    w("")

    w("=" * 110)
    w("B. HEADLINE -- ADVANTAGE (self - best capacity-matched O_io) ACROSS ARMS AND HORIZONS")
    w("=" * 110)
    w(f"  {'cell':18s} {'fact':6s} {'time':14s} " +
      " ".join(f"{lb:>18s}" for lb, _ in runs))
    w(f"  {'':18s} {'':6s} {'':14s} " +
      " ".join(f"{'held_seq/held_ck':>18s}" for _ in runs))
    for t in TARGETS + [t + "_h4m0.25" for t in TARGETS]:
        if not any(t in S["cells"] for _, S in runs):
            continue
        base = t.replace("_h4m0.25", "")
        cells = []
        for _, S in runs:
            c = S["cells"].get(t)
            cells.append((f"{c['advantage']['held_seq']:+.3f}/{c['advantage']['held_both']:+.3f}"
                          if c else "-").rjust(18))
        w(f"  {t:18s} {FAMILY[base]:6s} {DIRN[base]:14s} " + " ".join(cells))
    w("")
    w("  self-report accuracy (held_seq) / majority baseline:")
    for t in TARGETS:
        if not any(t in S["cells"] for _, S in runs):
            continue
        cells = []
        for _, S in runs:
            c = S["cells"].get(t)
            cells.append((f"{c['self']['held_seq']:.3f}/{c['baselines']['majority']['held_seq']:.3f}"
                          if c else "-").rjust(18))
        w(f"  {t:18s} {'':6s} {'':14s} " + " ".join(cells))
    w("")

    w("=" * 110)
    w("C. WITHIN HIERARCHY LEVEL -- advantage per level, per arm/horizon")
    w("=" * 110)
    for t in TARGETS:
        if not any(t in S["cells"] for _, S in runs):
            continue
        w(f"  {t}")
        w(f"    {'run':18s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6)))
        for lb, S in runs:
            c = S["cells"].get(t)
            if not c:
                continue
            abl = c.get("advantage_by_level", {})
            w(f"    {lb:18s} " + " ".join(
                fnum(abl.get(str(l), abl.get(l)), 9, sign=True) for l in range(6)))
        w("")

    w("=" * 110)
    w("D. THE GATE PER ARM  (disjoint-window autocorrelation / lag 1; lag 1 is clean at k>=2)")
    w("=" * 110)
    seen = set()
    for lb, S in runs:
        arm = S.get("arm") or lb.split("-")[0]
        if arm in seen:
            continue
        seen.add(arm)
        cap0 = list(S["targets"]["caps"])[0]
        g = S["targets"]["caps"][cap0]["gate"]
        w(f"  arm {arm} (instrument {cap0})")
        w(f"    {'k':>3s} " + " ".join(f"{nm:>26s}" for nm in
                                       ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO")))
        for kk in sorted(g, key=lambda z: int(z[1:])):
            row = []
            for nm in ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO"):
                d = g[kk][nm]
                row.append(f"{d.get('autocorr_disjoint_within', float('nan')):+.3f} / "
                           f"{d['autocorr_lag1_within']:+.3f}".rjust(26))
            w(f"    {kk[1:]:>3s} " + " ".join(row))
        cp = S["targets"]["caps"][cap0].get("class_persistence", {})
        w("    IMPL class persistence / chance: " + "  ".join(
            f"k={z[1:]}: {cp[z]['persistence']:.3f}/{cp[z]['chance']:.3f}"
            for z in sorted(cp, key=lambda q: int(q[1:]))))
        w(f"    instrument reliability: IMPL class agreement "
          f"{S['targets']['caps'][cap0]['cls_instrument_agreement']:.3f}; REORG "
          + " ".join(f"{nm}={x:+.3f}" for nm, x in
                     S["targets"]["caps"][cap0].get("reorg_instrument_reliability", {}).items()))
        w("")

    w("=" * 110)
    w("E. GUARDS PER ARM  (instrument h16m1 along each arm's own trajectory)")
    w("=" * 110)
    seen = set()
    for lb, S in runs:
        arm = S.get("arm") or lb.split("-")[0]
        if arm in seen:
            continue
        seen.add(arm)
        w(f"  arm {arm}")
        w(f"    {'step':>7s} {'cos':>7s} {'|r|':>7s} {'ens_cos':>8s} {'d6 eta2':>8s} "
          f"{'d5 eta2':>8s} {'d4 eta2':>8s}")
        for st in S["schedule"]:
            c = S["instruments"][str(st)]["caps"]["h16m1"]
            e = c["eta2_reference_last_token"]
            w(f"    {st:7d} {fnum(c['fwd_cosine'])} {fnum(c['res_norm'])} "
              f"{fnum(c['ens_cos'], 8)} {fnum(e.get('d6'), 8)} {fnum(e.get('d5'), 8)} "
              f"{fnum(e.get('d4'), 8)}")
        w("")

    w("=" * 110)
    w("F. PER-LEVEL NLL, START AND END  (which levels have dynamics on each arm)")
    w("=" * 110)
    seen = set()
    for lb, S in runs:
        arm = S.get("arm") or lb.split("-")[0]
        if arm in seen:
            continue
        seen.add(arm)
        lt = S["targets"].get("level_trajectory", {})
        if not lt:
            continue
        a, b = lt[str(S["schedule"][0])], lt[str(S["schedule"][-1])]
        g = lambda d, l: d["mean_nll_by_level"].get(str(l), d["mean_nll_by_level"].get(l))
        w(f"  arm {arm}")
        w(f"    {'':10s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6)))
        w(f"    {'first':10s} " + " ".join(fnum(g(a, l), 9) for l in range(6)))
        w(f"    {'final':10s} " + " ".join(fnum(g(b, l), 9) for l in range(6)))
        w(f"    {'delta':10s} " + " ".join(fnum(g(b, l) - g(a, l), 9, sign=True)
                                           for l in range(6)))
        w("")

    w("=" * 110)
    w("G. CONTINUOUS READOUT (Spearman rho; advantage over O_io@8:256)")
    w("=" * 110)
    w(f"  {'cell':18s} " + " ".join(f"{lb:>18s}" for lb, _ in runs))
    for t in TARGETS:
        if not any(t in S["cells"] and "cont" in S["cells"].get(t, {}) for _, S in runs):
            continue
        cells = []
        for _, S in runs:
            c = S["cells"].get(t, {}).get("cont")
            if not c:
                cells.append("-".rjust(18)); continue
            adv = c["self"]["held_seq"] - c.get("O_io@8:256", {}).get("held_seq", float("nan"))
            cells.append(f"{c['self']['held_seq']:.3f}/{adv:+.3f}".rjust(18))
        w(f"  {t:18s} " + " ".join(cells))
    w("")
    w("=" * 110)
    w("H. O_hist -- the history observer, which computes BEHAV_RETRO almost exactly")
    w("=" * 110)
    w(f"  {'cell':18s} " + " ".join(f"{lb:>22s}" for lb, _ in runs))
    w(f"  {'':18s} " + " ".join(f"{'self / O_io / O_hist':>22s}" for _ in runs))
    for t in TARGETS:
        if not any(t in S["cells"] for _, S in runs):
            continue
        cells = []
        for _, S in runs:
            c = S["cells"].get(t)
            if not c:
                cells.append("-".rjust(22)); continue
            bio = max((vv["held_seq"] for kk, vv in c["observers"].items()
                       if kk.startswith("O_io@")), default=float("nan"))
            oh = c["observers"].get("O_hist@8:256", {}).get("held_seq", float("nan"))
            cells.append(f"{c['self']['held_seq']:.3f}/{bio:.3f}/{oh:.3f}".rjust(22))
        w(f"  {t:18s} " + " ".join(cells))
    w("")
    txt = "\n".join(L)
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{name}_reduction.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return txt


def compare_figure(runs, outdir, name="compare"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    order = [t for t in TARGETS if any(t in S["cells"] for _, S in runs)]
    x = np.arange(len(order))
    wdt = 0.8 / max(1, len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
    for i, (kk, ttl) in enumerate((("held_seq", "held-out sequences"),
                                   ("held_both", "held-out sequences x held-out checkpoints"))):
        for j, (lb, S) in enumerate(runs):
            vals = [S["cells"].get(t, {}).get("advantage", {}).get(kk, np.nan) for t in order]
            ax[i].bar(x + j * wdt - 0.4, vals, width=wdt, label=lb)
        ax[i].axhline(0, color="k", lw=1)
        ax[i].set_xticks(x); ax[i].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax[i].set_ylabel("advantage (self - best $O_{io}$)"); ax[i].set_title(ttl)
        ax[i].grid(alpha=.3, axis="y")
        if i == 0:
            ax[i].legend(fontsize=8)
    fig.suptitle("Dispositional advantage across arms and horizons", y=1.02)
    fig.tight_layout(); fig.savefig(f"{outdir}/{name}_advantage.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    for a, t in zip(axs.ravel(), order):
        for lb, S in runs:
            c = S["cells"].get(t)
            if not c:
                continue
            abl = c.get("advantage_by_level", {})
            ls = sorted(int(z) for z in abl)
            a.plot(ls, [abl.get(str(l), abl.get(l)) for l in ls], "o-", ms=4, label=lb)
        a.axhline(0, color="k", lw=1); a.set_title(t, fontsize=9); a.grid(alpha=.3)
        a.set_xlabel("hierarchy level")
    axs.ravel()[0].set_ylabel("advantage"); axs.ravel()[0].legend(fontsize=7)
    fig.suptitle("Advantage within hierarchy level, across arms and horizons", y=1.01)
    fig.tight_layout(); fig.savefig(f"{outdir}/{name}_by_level.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"comparison figures -> {outdir}")


# ======================================================================
# Ceiling breakdown: within-level accuracy of every observer, not just O_io
# ======================================================================

CEIL_ORDER = ["self", "bestO_io", "O_hist@8:256", "O_act@8:256", "O_acthist@8:256",
              "confabulator"]


def ceilings(runs, outdir, name="ceilings"):
    """Within-level accuracy of the self-report and of EVERY observer rung (not only the
    advantage-defining O_io), plus per-level test n and the within-level advantage against
    max(O_io, O_hist). Read entirely from the saved cells -- every rung's
    `held_seq_by_level` / `held_both_by_level` was stored when the cell ran."""
    L = []
    w = L.append
    w("CEILING BREAKDOWN WITHIN LEVEL -- self against every rung, not only O_io")
    w("  advantage columns: ADV_io = self - best O_io (the headline statistic)")
    w("                     ADV_c  = self - max(best O_io, O_hist)   [the requested one]")
    w("")
    for lb, S in runs:
        tm = S["targets"]
        N, n_str = tm["N"], tm["n_seq_train"]
        n_te = N - n_str
        n_tr_ck, n_hd_ck = len(tm["train_target_indices"]), len(tm["held_target_indices"])
        lv = tm["levels"]
        cnt = [sum(1 for z in lv if z == l) for l in range(6)]
        w("=" * 118)
        w(f"{lb}   arm={S.get('arm', '?')}  k={tm['horizon']}  "
          f"train ckpts={n_tr_ck} held ckpts={n_hd_ck}  test sequences={n_te}")
        w(f"  per-level test n (held_seq)  : " +
          "  ".join(f"L{l}={n_tr_ck * n_te * cnt[l]:,}" for l in range(6)))
        w(f"  per-level test n (held_both) : " +
          "  ".join(f"L{l}={n_hd_ck * n_te * cnt[l]:,}" for l in range(6)))
        w("=" * 118)
        for t in TARGETS:
            c = S["cells"].get(t)
            if not c:
                continue
            obs = c["observers"]
            bio_key = max((kk for kk in obs if kk.startswith("O_io@")),
                          key=lambda kk: obs[kk]["held_seq"], default=None)
            rows = [("self", c["self"]),
                    (f"bestO_io ({bio_key.split('@')[1] if bio_key else '-'})",
                     obs.get(bio_key, {}))]
            for kk in ("O_hist@8:256", "O_act@8:256", "O_acthist@8:256"):
                if kk in obs:
                    rows.append((kk, obs[kk]))
            if "confabulator" in c:
                rows.append(("confabulator", c["confabulator"]))
            w(f"  {t}   (pooled: self {c['self']['held_seq']:.3f})")
            w(f"    {'rung':22s} " + " ".join(f"{'L' + str(l):>8s}" for l in range(6))
              + f" {'pooled':>9s}")
            for nm, d in rows:
                bl = d.get("held_seq_by_level", {})
                w(f"    {nm:22s} " + " ".join(
                    fnum(bl.get(str(l), bl.get(l)), 8) for l in range(6))
                  + f" {fnum(d.get('held_seq'), 9)}")
            sb = c["self"]["held_seq_by_level"]
            ib = obs.get(bio_key, {}).get("held_seq_by_level", {})
            hb = obs.get("O_hist@8:256", {}).get("held_seq_by_level", {})
            g = lambda d_, l: d_.get(str(l), d_.get(l))
            w(f"    {'ADV_io':22s} " + " ".join(
                fnum(g(sb, l) - g(ib, l), 8, sign=True) for l in range(6))
              + f" {fnum(c['advantage']['held_seq'], 9, sign=True)}")
            if hb:
                w(f"    {'ADV_c (vs io|hist)':22s} " + " ".join(
                    fnum(g(sb, l) - max(g(ib, l), g(hb, l)), 8, sign=True)
                    for l in range(6))
                  + f" {fnum(c['self']['held_seq'] - max(obs[bio_key]['held_seq'], obs['O_hist@8:256']['held_seq']), 9, sign=True)}")
            sb2 = c["self"].get("held_both_by_level", {})
            ib2 = obs.get(bio_key, {}).get("held_both_by_level", {})
            hb2 = obs.get("O_hist@8:256", {}).get("held_both_by_level", {})
            if sb2 and ib2:
                w(f"    {'ADV_c  held_both':22s} " + " ".join(
                    fnum(g(sb2, l) - max(g(ib2, l), g(hb2, l) if hb2 else -9), 8, sign=True)
                    for l in range(6))
                  + f" {fnum(c['advantage']['held_both'], 9, sign=True)}")
            w("")
        w("")
    txt = "\n".join(L)
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{name}_reduction.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return txt


# ======================================================================
# Follow-ups: the three instrument readings, and the continuous readout by level
# ======================================================================

def followups(FU, base_runs, outdir, name="ol_followups"):
    """FU: the followups summary ({suffix: {excess_meta, cells}}).
    base_runs: {suffix: main-run summary}, for the FIXED reading's cells."""
    L = []
    w = L.append
    w("OL FOLLOW-UPS -- the three reorganization readings, and the continuous readout by level")
    w("")
    w("  raw fresh  1 - cos(r_c, r_{c+-k}) with each checkpoint's OWN fresh FM.")
    w("             Floor = the instrument's own disagreement at a fixed checkpoint.")
    w("  excess     the same, minus the per-position same-checkpoint floor.")
    w("  fixed      1 - cos(r^(c)_c, r^(c)_{c+-k}); one FM on both sides, floor = 0 exactly.")
    w("  All three are quantile-binned WITHIN (level, checkpoint), so the bins carry no")
    w("  level or checkpoint information and the three are directly comparable.")
    w("")
    w("=" * 112)
    w("A. THE TARGET ITSELF -- magnitude of each reading, by level")
    w("=" * 112)
    for sfx, d in FU.items():
        em = d.get("excess_meta", {})
        for nm, t in em.items():
            if not isinstance(t, dict) or "floor_by_level" not in t:
                continue
            g = lambda q, l: t[q].get(str(l), t[q].get(l))
            w(f"  {sfx[1:]}  {nm}")
            w(f"    {'':16s} " + " ".join(f"{'L' + str(l):>8s}" for l in range(6)))
            for q, lab in (("fresh_by_level", "raw fresh"), ("floor_by_level", "floor"),
                           ("excess_by_level", "excess"), ("fixed_by_level", "fixed"),
                           ("floor_share_by_level", "floor share")):
                w(f"    {lab:16s} " + " ".join(fnum(g(q, l), 8) for l in range(6)))
            w(f"    rank corr(fresh, excess) = {t.get('rank_corr_fresh_vs_excess', float('nan')):.3f}"
              f"   (1.000 would mean the floor subtraction changes no ranking)")
            w("")

    w("=" * 112)
    w("B. ADVANTAGE ACROSS THE THREE READINGS  (self - best O_io; ADV_c also nets O_hist)")
    w("=" * 112)
    w(f"  {'row':26s} {'self':>7s} {'bestOio':>8s} {'O_hist':>8s} {'O_act':>8s} "
      f"{'ADV':>8s} {'ADV_c':>8s} {'ADV(ck)':>8s} {'base':>7s}")
    for sfx in FU:
        for nm in ("REORG_PRO", "REORG_RETRO"):
            for variant, src_ in (("fresh", FU[sfx]["cells"].get(f"{nm}_fresh")),
                                  ("excess", FU[sfx]["cells"].get(f"{nm}_excess")),
                                  ("fixed", base_runs.get(sfx, {}).get("cells", {}).get(nm))):
                if not src_ or "observers" not in src_:
                    continue
                obs = src_["observers"]
                bio = max(vv["held_seq"] for kk, vv in obs.items() if kk.startswith("O_io@"))
                oh = obs.get("O_hist@8:256", {}).get("held_seq", float("nan"))
                oa = obs.get("O_act@8:256", {}).get("held_seq", float("nan"))
                w(f"  {sfx[1:] + ' ' + nm + ' ' + variant:26s} {fnum(src_['self']['held_seq'])} "
                  f"{fnum(bio, 8)} {fnum(oh, 8)} {fnum(oa, 8)} "
                  f"{fnum(src_['self']['held_seq'] - bio, 8, sign=True)} "
                  f"{fnum(src_['self']['held_seq'] - max(bio, oh), 8, sign=True)} "
                  f"{fnum(src_['advantage']['held_both'], 8, sign=True)} "
                  f"{fnum(src_['baselines']['majority']['held_seq'])}")
        w("")

    w("=" * 112)
    w("C. THE SAME, WITHIN HIERARCHY LEVEL  (ADV = self - best O_io)")
    w("=" * 112)
    for sfx in FU:
        for nm in ("REORG_PRO", "REORG_RETRO"):
            w(f"  {sfx[1:]}  {nm}")
            w(f"    {'reading':16s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6))
              + f" {'pooled':>9s}")
            for variant, src_ in (("raw fresh", FU[sfx]["cells"].get(f"{nm}_fresh")),
                                  ("excess", FU[sfx]["cells"].get(f"{nm}_excess")),
                                  ("fixed", base_runs.get(sfx, {}).get("cells", {}).get(nm))):
                if not src_ or "advantage_by_level" not in src_:
                    continue
                abl = src_["advantage_by_level"]
                w(f"    {variant:16s} " + " ".join(
                    fnum(abl.get(str(l), abl.get(l)), 9, sign=True) for l in range(6))
                  + f" {fnum(src_['advantage']['held_seq'], 9, sign=True)}")
            for variant, src_ in (("raw fresh", FU[sfx]["cells"].get(f"{nm}_fresh")),
                                  ("excess", FU[sfx]["cells"].get(f"{nm}_excess")),
                                  ("fixed", base_runs.get(sfx, {}).get("cells", {}).get(nm))):
                if not src_ or "observers" not in src_:
                    continue
                obs = src_["observers"]
                bk = max((kk for kk in obs if kk.startswith("O_io@")),
                         key=lambda kk: obs[kk]["held_seq"])
                sb, ib = src_["self"]["held_seq_by_level"], obs[bk]["held_seq_by_level"]
                hb = obs.get("O_hist@8:256", {}).get("held_seq_by_level", {})
                g = lambda d_, l: d_.get(str(l), d_.get(l))
                w(f"    {variant + ' ADV_c':16s} " + " ".join(
                    fnum(g(sb, l) - max(g(ib, l), g(hb, l) if hb else -9), 9, sign=True)
                    for l in range(6)))
            w("")

    w("=" * 112)
    w("D. CONTINUOUS READOUT WITHIN LEVEL  (Spearman rho against the within-(level,")
    w("   checkpoint) z-scored target; rank models, predictions saved)")
    w("=" * 112)
    for sfx in FU:
        for nm in ("REORG_PRO", "REORG_RETRO", "BEHAV_PRO", "BEHAV_RETRO"):
            c = FU[sfx]["cells"].get(f"{nm}_cont")
            if not c or "cont" not in c:
                continue
            w(f"  {sfx[1:]}  {nm}")
            w(f"    {'model':16s} " + " ".join(f"{'L' + str(l):>9s}" for l in range(6))
              + f" {'pooled':>9s}")
            for mk in ("self", "O_io@8:256", "O_hist@8:256", "O_act@8:256"):
                d = c["cont"].get(mk)
                if not d:
                    continue
                bl = d.get("held_seq_by_level", {})
                w(f"    {mk:16s} " + " ".join(
                    fnum(bl.get(str(l), bl.get(l)), 9) for l in range(6))
                  + f" {fnum(d['held_seq'], 9)}")
            abl = c.get("cont_advantage_by_level", {})
            if abl:
                w(f"    {'ADV vs best obs':16s} " + " ".join(
                    fnum(abl.get(str(l), abl.get(l)), 9, sign=True) for l in range(6)))
            w("")
    txt = "\n".join(L)
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{name}_reduction.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return txt


def followups_figure(FU, base_runs, outdir, name="ol_followups"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sfxs = list(FU)
    rows = [(s_, nm) for s_ in sfxs for nm in ("REORG_PRO", "REORG_RETRO")]
    fig, axs = plt.subplots(1, len(rows), figsize=(3.1 * len(rows), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for a, (s_, nm) in zip(axs, rows):
        for variant, src_, mk in (("raw fresh", FU[s_]["cells"].get(f"{nm}_fresh"), "o-"),
                                  ("excess", FU[s_]["cells"].get(f"{nm}_excess"), "s--"),
                                  ("fixed", base_runs.get(s_, {}).get("cells", {}).get(nm), "^-.")):
            if not src_ or "advantage_by_level" not in src_:
                continue
            abl = src_["advantage_by_level"]
            ls = sorted(int(z) for z in abl)
            a.plot(ls, [abl.get(str(l), abl.get(l)) for l in ls], mk, ms=4, label=variant)
        a.axhline(0, color="k", lw=1); a.grid(alpha=.3)
        a.set_title(f"{s_[1:]}  {nm}", fontsize=9); a.set_xlabel("hierarchy level")
    axs[0].set_ylabel("advantage (self - best $O_{io}$)"); axs[0].legend(fontsize=7)
    fig.suptitle("Reorganization advantage under the three instrument readings (OL)", y=1.03)
    fig.tight_layout(); fig.savefig(f"{outdir}/{name}_readings.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"figure -> {outdir}/{name}_readings.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="label=path pairs, e.g. CL-k1=main_summary.json OL-k3=ol_k3_summary.json")
    ap.add_argument("--name", default="compare")
    ap.add_argument("--followups", help="path to the followups summary json")
    ap.add_argument("--base", nargs="*", default=[],
                    help="suffix=path pairs for the FIXED reading's runs, e.g. _k1=ol_k1_summary.json")
    ap.add_argument("--ceilings", action="store_true",
                    help="with --compare: emit the within-level ceiling breakdown instead")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    if a.followups:
        FU = json.load(open(a.followups))
        base = {z.split("=", 1)[0]: json.load(open(z.split("=", 1)[1])) for z in a.base}
        followups(FU, base, a.outdir, a.name)
        if not a.no_figures:
            try:
                followups_figure(FU, base, a.outdir, a.name)
            except Exception as e:
                print(f"[figure skipped: {e}]")
        raise SystemExit(0)
    if a.compare:
        runs = [(z.split("=", 1)[0], json.load(open(z.split("=", 1)[1]))) for z in a.compare]
        if a.ceilings:
            ceilings(runs, a.outdir, a.name)
            raise SystemExit(0)
        compare(runs, a.outdir, a.name)
        if not a.no_figures:
            try:
                compare_figure(runs, a.outdir, a.name)
            except Exception as e:
                print(f"[comparison figures skipped: {e}]")
    else:
        S = json.load(open(a.summary))
        reduce_run(S, a.tag, a.outdir)
        if not a.no_figures:
            try:
                make_figures(S, a.tag, a.outdir)
            except Exception as e:
                print(f"[figures skipped: {e}]")
