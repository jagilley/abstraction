"""Local post-processing for practice/estimability runs (no compute; reads the per-burst
results.json written by `estimability.py`).

Prints, and draws to `figures/<tag>/`:
  fig1_learning      per-class E[e|pos] trajectories per schedule (ungated `fixed` arm)
                     + the acquisition/retention summary vs burst
  fig2_estimability  the 2D surface: b(s) fidelity (RMSE / corr vs the moving truth) for
                     every panel estimator x every schedule -- benchmark timescale and
                     revisit rate as the two halves of one condition
  fig3_by_lag        credit error binned by REVISIT LAG, pooled across schedules
  fig4_reentry       the re-estimation transient: b error vs position-within-burst
  fig5_generalize    the net-vs-EWMA decomposition: change in b at an ABSENT context vs
                     change in the truth there (slope 1 = generalization tracks the
                     drift; slope 0 = frozen, i.e. EWMA-like; negative = interference)
  fig6_consumption   what the schedule did to delta: credit fidelity, sign errors,
                     allocation TV vs an oracle benchmark, and the learning outcome

Usage:
    python3 mjc/practice/estimability/analyze_estimability.py --tag est_s0
    # if the local mirror is missing, pull from the volume first:
    #   modal volume get mujoco-control-data practice_estimability/<tag> \
    #       mjc/practice/estimability/results/<tag>_vol
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EST_ORDER = ["net@0.001", "net@0.003", "net@0.01", "net@0.03",
             "ewma_ctx@0.01", "ewma_ctx@0.05", "ewma_ctx@0.2",
             "ewma_scalar@0.05", "frozen"]
EST_COLOR = {"net": "#1f77b4", "ewma_ctx": "#d62728", "ewma_scalar": "#ff7f0e",
             "frozen": "#7f7f7f"}
EST_STYLE = {0: "-", 1: "--", 2: "-.", 3: ":"}


def load(tag):
    runs = {}
    for pat in (os.path.join(HERE, "results", tag, "burst*.json"),
                os.path.join(HERE, "results", tag, "*", "results.json")):
        for p in sorted(glob.glob(pat)):
            with open(p) as fh:
                R = json.load(fh)
            runs[int(R["config"]["burst"])] = R
    if not runs:
        raise SystemExit(f"no results under results/{tag}/ -- pull from the volume first")
    return dict(sorted(runs.items(), reverse=True))     # massed -> interleaved


def ck_series(ck, key, cls):
    return np.array([d.get(cls, np.nan) for d in ck[key]], float)


def visit_intervals(ctx_of_cycle, j):
    """(leave_cycle, return_cycle) for every absence of context j."""
    v = np.where(np.asarray(ctx_of_cycle) == j)[0]
    out = []
    for a, b in zip(v[:-1], v[1:]):
        if b - a > 1:
            out.append((int(a), int(b)))
    return out


def report(runs, tag, learn_cls):
    bursts = list(runs.keys())
    K = len(runs[bursts[0]]["config"]["contexts"])  # noqa: F841
    CLS = runs[bursts[0]]["classes"]
    print(f"\n######## practice/estimability :: {tag} ########")
    r0 = runs[bursts[0]]
    print(f"contexts: {CLS} | K={K} | n_bpc={r0['config']['n_bpc']} "
          f"batch={r0['config']['batch']} | per-context samples FIXED at "
          f"{r0['config']['n_bpc'] * r0['config']['batch']}")
    print(f"calib: tau={r0['calibration']['tau']:.5f} "
          f"gate AUROC={r0['calibration']['gate_auroc_train']:.3f}")
    print("stale E[e|pos]: " + " ".join(
        f"{k}={v:.4f}" for k, v in r0["calibration"]["stale_field_per_class"].items()))

    # ---------------- learning outcome ----------------
    print("\n== learning outcome (E[e|pos], ungated `fixed` arm; mean over the "
          f"{len(learn_cls)} learnable contexts {learn_cls}) ==")
    print(f"{'burst':>6} {'absence':>8} | {'final mean':>10} {'AUC mean':>9} | " +
          " ".join(f"{c:>9}" for c in CLS))
    for b in bursts:
        R = runs[b]; a = R["arms"]["fixed"]
        fin = a["final_field"]
        auc = {c: float(np.nanmean(ck_series(a["ckpt"], "truth", c))) for c in CLS}
        print(f"{b:>6} {R['schedule']['absence_at_reentry']:>8} | "
              f"{np.mean([fin[c] for c in learn_cls]):>10.4f} "
              f"{np.mean([auc[c] for c in learn_cls]):>9.4f} | " +
              " ".join(f"{fin[c]:>9.4f}" for c in CLS))

    # ---------------- b(s) fidelity: the estimability surface ----------------
    for metric, lab in (("b_rmse", "b RMSE vs the moving truth (lower=better)"),
                        ("b_corr", "corr(b, b*) over the consumed stream"),
                        ("sign_err_material", "material sign-error rate of delta"),
                        ("alloc_tv", "allocation TV vs an oracle benchmark")):
        print(f"\n== {lab} ==")
        print(f"{'estimator':>18} | " + " ".join(f"{'b=' + str(b):>9}" for b in bursts))
        for en in EST_ORDER:
            row = []
            for b in bursts:
                pan = runs[b]["arms"][runs[b]["config"]["panel_arm"]].get("panel", {})
                v = pan.get(en, {}).get(metric)
                row.append("      n/a" if v is None else f"{v:>9.4f}")
            print(f"{en:>18} | " + " ".join(row))

    # ---------------- active arms ----------------
    print("\n== active arms (delta consumed as a gain; fidelity + outcome) ==")
    arms = [a for a in r0["config"]["arms"]]
    hdr = f"{'burst':>6} {'arm':>13} | {'b_corr':>7} {'d_corr':>7} {'signerr':>8} " \
          f"{'allocTV':>8} {'mean_w':>7} | {'final mean':>10}"
    print(hdr)
    for b in bursts:
        R = runs[b]
        for an in arms:
            a = R["arms"][an]; f = a["fidelity"] or {}
            fin = np.mean([a["final_field"][c] for c in learn_cls])
            se = f.get("sign_err_material")
            print(f"{b:>6} {an:>13} | "
                  f"{f.get('b_corr', float('nan')):>7.3f} {f.get('delta_corr', float('nan')):>7.3f} "
                  f"{(float('nan') if se is None else se):>8.3f} "
                  f"{f.get('alloc_tv', float('nan')):>8.3f} {a['mean_w']:>7.3f} | {fin:>10.4f}")

    # ---------------- the within-run revisit-lag dose-response ----------------
    print("\n== within-run: b error at CONSUMPTION, by cycles since the context was last "
          "practiced ==")
    print("(continuation = the lag-1..2 bin; re-entry = the bin at (K-1)*burst; "
          "'first' = never practiced post-drift)")
    for en in ("net@0.003", "ewma_ctx@0.05", "ewma_scalar@0.05"):
        print(f"  -- {en} --")
        print(f"{'burst':>6} {'absence':>8} | " +
              " ".join(f"{h:>22}" for h in ("continuation", "re-entry", "first")))
        for b in bursts:
            R = runs[b]
            bl = R["arms"][R["config"]["panel_arm"]].get("panel", {}).get(en, {}).get("by_lag", {})
            cells = []
            cont = [v for k, v in bl.items() if k != "first" and float(k.split("-")[1]) <= 2]
            reent = [v for k, v in bl.items() if k != "first" and float(k.split("-")[0]) >= 2]
            for grp in (cont, reent, [bl["first"]] if "first" in bl else []):
                if not grp:
                    cells.append(f"{'-':>22}")
                else:
                    ae = np.mean([g["b_abs_err"] for g in grp])
                    dc = np.mean([g["delta_corr"] for g in grp])
                    nn = sum(g["n"] for g in grp)
                    cells.append(f"{ae:.4f}/dc{dc:.2f}/n{nn:<6d}"[:22].rjust(22))
            print(f"{b:>6} {R['schedule']['absence_at_reentry']:>8} | " + " ".join(cells))

    # ---------------- generalization vs interference ----------------
    print("\n== absent-context decomposition: does b MOVE while you are away, and does "
          "it move WITH the truth? ==")
    print("(slope of d_b on d_truth: 1 = generalization tracks the drift; 0 = frozen; "
          "<0 = interference)")
    print(f"{'burst':>6} {'estimator':>18} | {'n':>4} {'slope':>7} {'corr':>7} "
          f"{'sd(d_truth)':>12} {'sd(d_b)':>9}")
    decomp = {}
    for b in bursts:
        R = runs[b]; a = R["arms"][R["config"]["panel_arm"]]
        ck = a["ckpt"]; cyc = np.array(ck["cycle"])
        for en in EST_ORDER:
            key = f"pan_{en}"
            if key not in ck:
                continue
            dt, db = [], []
            for j in range(K):
                for (lo, hi) in visit_intervals(R["schedule"]["ctx_of_cycle"], j):
                    # i0: first checkpoint at/after the last visit; i1: last checkpoint
                    # STRICTLY BEFORE the return, so no post-return update contaminates
                    # d_b (a tabular EWMA must read exactly 0 here, by construction).
                    i0 = np.searchsorted(cyc, lo, "left")
                    i1 = np.searchsorted(cyc, hi, "left") - 1
                    if i1 - i0 < 1:
                        continue
                    t = ck_series(ck, "truth", CLS[j]); bb = ck_series(ck, key, CLS[j])
                    dt.append(t[i1] - t[i0]); db.append(bb[i1] - bb[i0])
            if len(dt) < 4:
                continue
            dt = np.array(dt); db = np.array(db)
            sl = float(np.polyfit(dt, db, 1)[0]) if np.std(dt) > 1e-9 else np.nan
            cr = float(np.corrcoef(dt, db)[0, 1]) if min(np.std(dt), np.std(db)) > 1e-9 else np.nan
            decomp[(b, en)] = dict(n=len(dt), slope=sl, corr=cr,
                                   sd_t=float(np.std(dt)), sd_b=float(np.std(db)),
                                   dt=dt.tolist(), db=db.tolist())
            print(f"{b:>6} {en:>18} | {len(dt):>4} {sl:>7.3f} {cr:>7.3f} "
                  f"{np.std(dt):>12.4f} {np.std(db):>9.4f}")
    if not decomp:
        print("  (no absence longer than the checkpoint spacing in any schedule)")
    return decomp


def figures(runs, tag, learn_cls, decomp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = os.path.join(HERE, "figures", tag)
    os.makedirs(outdir, exist_ok=True)
    bursts = list(runs.keys())
    CLS = runs[bursts[0]]["classes"]
    absence = {b: runs[b]["schedule"]["absence_at_reentry"] for b in bursts}

    def savefig(fig, nm):
        p = os.path.join(outdir, nm); fig.tight_layout(); fig.savefig(p, dpi=130)
        plt.close(fig); print(f"[fig] {p}")

    # ---- fig1: learning trajectories + outcome vs burst ----
    n = len(bursts)
    fig, ax = plt.subplots(2, max(n, 3), figsize=(3.1 * max(n, 3), 6.0), squeeze=False)
    for i, b in enumerate(bursts):
        R = runs[b]; a = R["arms"]["fixed"]; ck = a["ckpt"]
        t = np.array(ck["t"])
        for ci, c in enumerate(CLS):
            ax[0][i].plot(t, ck_series(ck, "truth", c), label=c, lw=1.2)
        ax[0][i].set_title(f"burst={b} (absence {absence[b]})", fontsize=9)
        ax[0][i].set_xlabel("transitions"); ax[0][i].set_ylabel("E[e|pos]")
        if i == 0:
            ax[0][i].legend(fontsize=6)
    for i in range(n, max(n, 3)):
        ax[0][i].axis("off")
    for arm in runs[bursts[0]]["config"]["arms"]:
        y = [np.mean([runs[b]["arms"][arm]["final_field"][c] for c in learn_cls]) for b in bursts]
        ax[1][0].plot([absence[b] for b in bursts], y, "o-", label=arm, lw=1.4)
    ax[1][0].set_xscale("symlog"); ax[1][0].set_xlabel("absence at re-entry (cycles)")
    ax[1][0].set_ylabel("final mean E[e|pos]"); ax[1][0].legend(fontsize=7)
    ax[1][0].set_title("outcome: massed (right) -> interleaved (left)", fontsize=9)
    for ci, c in enumerate(CLS):
        y = [runs[b]["arms"]["fixed"]["final_field"][c] for b in bursts]
        ax[1][1].plot([absence[b] for b in bursts], y, "o-", label=c, lw=1.2)
    ax[1][1].set_xscale("symlog"); ax[1][1].set_xlabel("absence at re-entry")
    ax[1][1].set_ylabel("final E[e|pos]"); ax[1][1].legend(fontsize=6)
    ax[1][1].set_title("per-context retention profile (fixed arm)", fontsize=9)
    for i in range(2, max(n, 3)):
        ax[1][i].axis("off")
    savefig(fig, "fig1_learning.png")

    # ---- fig2: the estimability surface ----
    mets = [("b_rmse", "b RMSE vs moving truth"), ("b_corr", "corr(b, b*)"),
            ("sign_err_material", "material sign-error rate"),
            ("alloc_tv", "allocation TV vs oracle b")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for mi, (m, lab) in enumerate(mets):
        for en in EST_ORDER:
            fam = en.split("@")[0]
            k = [e for e in EST_ORDER if e.split("@")[0] == fam].index(en)
            y = []
            for b in bursts:
                pan = runs[b]["arms"][runs[b]["config"]["panel_arm"]].get("panel", {})
                y.append(pan.get(en, {}).get(m, np.nan))
            y = np.array([np.nan if v is None else v for v in y], float)
            ax[mi].plot([absence[b] for b in bursts], y, "o" + EST_STYLE[k % 4],
                        color=EST_COLOR[fam], label=en, lw=1.4, ms=4)
        ax[mi].set_xscale("symlog"); ax[mi].set_xlabel("absence at re-entry (cycles)")
        ax[mi].set_title(lab, fontsize=9)
        if mi == 0:
            ax[mi].legend(fontsize=6, ncol=2)
    fig.suptitle("estimability surface: benchmark timescale x revisit rate "
                 "(all estimators read the SAME stream off the SAME FM trajectory)",
                 fontsize=10)
    savefig(fig, "fig2_estimability.png")

    # ---- fig3: by revisit lag ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for en in EST_ORDER:
        fam = en.split("@")[0]
        k = [e for e in EST_ORDER if e.split("@")[0] == fam].index(en)
        xs, ye, yd, ys = [], [], [], []
        for b in bursts:
            pan = runs[b]["arms"][runs[b]["config"]["panel_arm"]].get("panel", {})
            for bin_name, d in pan.get(en, {}).get("by_lag", {}).items():
                if bin_name == "first":
                    continue
                lo = float(bin_name.split("-")[0]); hi = float(bin_name.split("-")[1])
                xs.append(0.5 * (lo + min(hi, 1000))); ye.append(d["b_abs_err"])
                yd.append(d["delta_corr"]); ys.append(d["sign_err"])
        if not xs:
            continue
        o = np.argsort(xs)
        xs = np.array(xs)[o]
        for a_, y in ((ax[0], np.array(ye, float)[o]), (ax[1], np.array(yd, float)[o]),
                      (ax[2], np.array([np.nan if v is None else v for v in ys], float)[o])):
            a_.plot(xs, y, "o" + EST_STYLE[k % 4], color=EST_COLOR[fam], label=en,
                    lw=1.2, ms=4, alpha=0.85)
    for a_, lab in zip(ax, ["|b - b*| at consumption", "corr(delta, delta_oracle)",
                            "material sign-error rate"]):
        a_.set_xscale("symlog"); a_.set_xlabel("cycles since context last practiced")
        a_.set_title(lab, fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    fig.suptitle("credit error vs revisit lag (pooled across schedules)", fontsize=10)
    savefig(fig, "fig3_by_lag.png")

    # ---- fig4: re-estimation transient ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for b in bursts:
        pan = runs[b]["arms"][runs[b]["config"]["panel_arm"]].get("panel", {})
        for en, a_ in (("net@0.003", ax[0]), ("ewma_ctx@0.05", ax[1])):
            d = pan.get(en, {}).get("by_pib", {})
            if not d:
                continue
            xs = [float(k.split("-")[0]) for k in d]
            ys = [v["b_abs_err"] for v in d.values()]
            o = np.argsort(xs)
            a_.plot(np.array(xs)[o] + 0.5, np.array(ys, float)[o], "o-",
                    label=f"burst={b}", lw=1.3, ms=4)
    for a_, t in zip(ax, ["net@0.003", "ewma_ctx@0.05"]):
        a_.set_xscale("symlog"); a_.set_xlabel("batches into the burst (0 = re-entry)")
        a_.set_ylabel("|b - b*|"); a_.set_title(t, fontsize=9); a_.legend(fontsize=7)
    fig.suptitle("the re-estimation transient after an absence", fontsize=10)
    savefig(fig, "fig4_reentry.png")

    # ---- fig5: generalization vs interference ----
    keys = [k for k in decomp if k[1] in ("net@0.003", "net@0.01", "ewma_ctx@0.05", "frozen")]
    if keys:
        fams = sorted({k[1] for k in keys})
        fig, ax = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3.6), squeeze=False)
        for i, en in enumerate(fams):
            a_ = ax[0][i]
            for b in bursts:
                if (b, en) not in decomp:
                    continue
                d = decomp[(b, en)]
                a_.scatter(d["dt"], d["db"], s=12, alpha=0.6, label=f"b={b} (m={d['slope']:.2f})")
            lim = np.array(a_.get_xlim())
            a_.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
            a_.axhline(0, color="k", lw=0.6, alpha=0.4)
            a_.set_xlabel("d truth at the absent context"); a_.set_ylabel("d b there")
            a_.set_title(en, fontsize=9); a_.legend(fontsize=6)
        fig.suptitle("while you are away: does b track the drift (slope 1) or drift on "
                     "its own (interference)?", fontsize=10)
        savefig(fig, "fig5_generalize.png")

    # ---- fig6: consumption ----
    arms = [a for a in runs[bursts[0]]["config"]["arms"] if a.startswith(("delta", "oracle"))]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for arm in arms:
        x = [absence[b] for b in bursts]
        for mi, m in enumerate(["b_corr", "delta_corr", "sign_err_material", "alloc_tv"]):
            y = [(runs[b]["arms"][arm]["fidelity"] or {}).get(m, np.nan) for b in bursts]
            ax[mi].plot(x, [np.nan if v is None else v for v in y], "o-", label=arm, lw=1.4)
    for a_, lab in zip(ax, ["corr(b, b*)", "corr(delta, delta_oracle)",
                            "material sign-error rate", "allocation TV vs oracle"]):
        a_.set_xscale("symlog"); a_.set_xlabel("absence at re-entry"); a_.set_title(lab, fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("consumption: what the schedule did to the delta actually applied", fontsize=10)
    savefig(fig, "fig6_consumption.png")


def multiseed(tags, learn_cls):
    """mean +- sd across seeds for the headline tables, and the per-seed ordering."""
    S = {t: load(t) for t in tags}
    bursts = list(S[tags[0]].keys())
    print(f"\n######## multi-seed ({len(tags)} seeds: {', '.join(tags)}) ########")

    def agg(f):
        return {b: np.array([f(S[t][b]) for t in tags], float) for b in bursts}

    print("\n== learning outcome, ungated `fixed` arm (final mean over learnable "
          "contexts; lower=better) ==")
    o = agg(lambda R: np.mean([R["arms"]["fixed"]["final_field"][c] for c in learn_cls]))
    print(f"{'burst':>6} {'absence':>8} | {'mean':>7} {'sd':>7} | per-seed")
    for b in bursts:
        print(f"{b:>6} {S[tags[0]][b]['schedule']['absence_at_reentry']:>8} | "
              f"{o[b].mean():>7.4f} {o[b].std():>7.4f} | " +
              " ".join(f"{v:.4f}" for v in o[b]))
    best = {t: min(bursts, key=lambda b: np.mean([S[t][b]["arms"]["fixed"]["final_field"][c]
                                                  for c in learn_cls])) for t in tags}
    print(f"  argmin burst per seed: {best}")

    for en in ("net@0.001", "net@0.003", "net@0.01", "ewma_ctx@0.05", "ewma_ctx@0.2",
               "ewma_scalar@0.05"):
        for m in ("b_rmse", "sign_err_material"):
            v = agg(lambda R, en=en, m=m: (R["arms"][R["config"]["panel_arm"]]
                                           .get("panel", {}).get(en, {}).get(m, np.nan)))
            print(f"{en:>18} {m:>18} | " +
                  " ".join(f"{v[b].mean():.4f}+-{v[b].std():.4f}" for b in bursts))
    print(f"{'':>18} {'(bursts)':>18} | " + " ".join(f"{'b=' + str(b):>15}" for b in bursts))

    print("\n== active arms: final mean over learnable contexts (mean +- sd) ==")
    arms = S[tags[0]][bursts[0]]["config"]["arms"]
    print(f"{'burst':>6} | " + " ".join(f"{a:>17}" for a in arms))
    for b in bursts:
        row = []
        for a in arms:
            v = np.array([np.mean([S[t][b]["arms"][a]["final_field"][c] for c in learn_cls])
                          for t in tags], float)
            row.append(f"{v.mean():.4f}+-{v.std():.4f}".rjust(17))
        print(f"{b:>6} | " + " ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="est_s0")
    ap.add_argument("--tags", default="", help="comma-separated tags -> multi-seed summary")
    ap.add_argument("--learn-cls", default="A-rot+,B-rot-,C-rot~",
                    help="contexts whose error is reducible (the learning outcome)")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    learn = [c for c in a.learn_cls.split(",") if c]
    if a.tags:
        multiseed([t for t in a.tags.split(",") if t], learn)
        return
    runs = load(a.tag)
    decomp = report(runs, a.tag, learn)
    if not a.no_figs:
        figures(runs, a.tag, learn, decomp)


if __name__ == "__main__":
    main()
