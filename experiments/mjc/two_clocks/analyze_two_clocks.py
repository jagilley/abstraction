"""Aggregate two_clocks runs: the (ii) architecture table from main_s0 and the
(i) window-matching diagonal from sweep_s0, plus the dense fidelity-offset curve
computed from the logged per-sample series (no compute -- post-processing only).

Usage:
    cd experiments/
    python3 mjc/two_clocks/analyze_two_clocks.py                  # default tags
    python3 mjc/two_clocks/analyze_two_clocks.py --tags main_s0 sweep_s0
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures", "analysis")


def fetch(tag: str, refetch: bool = True) -> str:
    dest = os.path.join(RESULTS, tag)
    os.makedirs(RESULTS, exist_ok=True)
    if refetch or not os.path.exists(os.path.join(dest, "results.json")):
        subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                        f"two_clocks/{tag}", RESULTS + "/"], check=True)
    return dest


def load(tag: str):
    d = fetch(tag)
    with open(os.path.join(d, "results.json")) as fh:
        R = json.load(fh)
    npz_path = os.path.join(d, "traces.npz")
    Z = np.load(npz_path, allow_pickle=False) if os.path.exists(npz_path) else None
    return R, Z


def lad_series(R, arm, key):
    return [(r["transitions"], r[key]) for r in R["arms"][arm].get("ladder", []) if key in r]


def recovery_auc(R, arm, key="ballistic_a", t_min=512):
    pts = [y for t, y in lad_series(R, arm, key) if t >= t_min]
    return float(np.mean(pts)) if pts else float("nan")


def probe_auc(R, arm, probe, t_min=256):
    tr = R["arms"][arm]["trace"]
    ys = [y for t, y in zip(tr["t"], tr[f"probe_{probe}"]) if t >= t_min]
    return float(np.mean(ys)) if ys else float("nan")


def over_weight(R, arm, cls):
    bud = R["arms"][arm]["budget"]
    ss = bud["sample_share"].get(cls, 0.0)
    return bud["cum_w_share"].get(cls, 0.0) / ss if ss > 0 else float("nan")


def final_probe(R, arm, probe):
    tr = R["arms"][arm]["trace"]
    return tr[f"probe_{probe}"][-1] if tr[f"probe_{probe}"] else float("nan")


def main_table(R):
    print("\n================ (ii) ARCHITECTURE TABLE ================")
    print(f"stale   ballistic_a={R['stale']['ballistic_a']:.4f}  "
          f"ceiling ballistic_a={R['ceiling']['ballistic_a']:.4f}")
    tri = R["trials"]
    print(f"trials: fam-a clean/drift {tri['fam_a_out_clean']:.3f}/{tri['fam_a_out_drift']:.3f}  "
          f"fam-b {tri['fam_b_out_clean']:.3f}/{tri['fam_b_out_drift']:.3f}")
    hdr = (f"{'arm':22s} {'bal_a AUC':>9s} {'bal_a end':>9s} {'R1 AUC':>8s} {'R1 end':>8s} "
           f"{'R1 ow':>6s} {'R2 ow':>6s} {'noise ow':>8s} {'base end':>8s} {'mean_w':>6s} "
           f"{'fid_f':>6s} {'fid_s':>6s}")
    print(hdr); print("-" * len(hdr))
    for arm in R["arms"]:
        fid = R["arms"][arm].get("fidelity", {})
        ff = max([v for k, v in fid.items() if k.endswith("_vs_fast")], default=float("nan"))
        fs = max([v for k, v in fid.items() if k.endswith("_vs_slow")], default=float("nan"))
        bal = lad_series(R, arm, "ballistic_a")
        bal_end = bal[-1][1] if bal else float("nan")
        print(f"{arm:22s} {recovery_auc(R, arm):9.4f} "
              f"{bal_end:9.4f} "
              f"{probe_auc(R, arm, 'R1-on'):8.4f} {final_probe(R, arm, 'R1-on'):8.4f} "
              f"{over_weight(R, arm, 'R1-on'):6.2f} {over_weight(R, arm, 'R2-off'):6.2f} "
              f"{over_weight(R, arm, 'R3-noise'):8.2f} {final_probe(R, arm, 'base'):8.4f} "
              f"{R['arms'][arm]['budget']['mean_w']:6.3f} {ff:6.3f} {fs:6.3f}")
    # the full/light extra readouts where present
    print("\nextra controls (where graded):")
    for arm in R["arms"]:
        for key in ("ballistic_b", "reactive_a"):
            pts = lad_series(R, arm, key)
            if pts:
                print(f"  {arm:22s} {key:12s} " +
                      " ".join(f"{t}:{y:.4f}" for t, y in pts))


def sweep_table(R):
    import re
    print("\n================ (i) WINDOW-MATCHING TABLE ================")
    rows = []
    for arm in R["arms"]:
        m = re.match(r"fast:t(\d+):d(\d+)", arm)
        if not m:
            continue
        tau, d = int(m.group(1)), int(m.group(2))
        rows.append((d, tau, arm))
    rows.sort()
    hdr = (f"{'arm':16s} {'d_fm':>4s} {'tau':>4s} {'bal_a AUC':>9s} {'R1 AUC':>8s} "
           f"{'R2 AUC':>8s} {'noise ow':>8s} {'base end':>8s} {'fid_f':>6s}")
    print(hdr); print("-" * len(hdr))
    if "fixed" in R["arms"]:
        print(f"{'fixed':16s} {'-':>4s} {'-':>4s} {recovery_auc(R, 'fixed'):9.4f} "
              f"{probe_auc(R, 'fixed', 'R1-on'):8.4f} {probe_auc(R, 'fixed', 'R2-off'):8.4f} "
              f"{over_weight(R, 'fixed', 'R3-noise'):8.2f} {final_probe(R, 'fixed', 'base'):8.4f} "
              f"{'':>6s}")
    for d, tau, arm in rows:
        fid = R["arms"][arm].get("fidelity", {})
        print(f"{arm:16s} {d:4d} {tau:4d} {recovery_auc(R, arm):9.4f} "
              f"{probe_auc(R, arm, 'R1-on'):8.4f} {probe_auc(R, arm, 'R2-off'):8.4f} "
              f"{over_weight(R, arm, 'R3-noise'):8.2f} {final_probe(R, arm, 'base'):8.4f} "
              f"{fid.get('led0_vs_fast', float('nan')):6.3f}")
    return rows


def kernel(tau, sig, A):
    ages = np.arange(A + 1, dtype=np.float64)
    Kk = np.exp(-(ages - tau) ** 2 / (2.0 * sig ** 2))
    return Kk / max(Kk.sum(), 1e-12)


def offset_curve(z_f, d_fm, A, taus):
    """Dense fidelity-vs-window curve from one logged delta series: credit under a
    kernel at tau for a signal delayed d_fm, correlated with the oracle."""
    T = len(z_f)
    out = []
    for tau in taus:
        Kk = kernel(tau, max(0.35 * tau, 1.0), A)
        cred = np.zeros(T)
        for age in range(A + 1):
            if Kk[age] < 1e-6:
                continue
            # signal about t' arrives at t'+d; deposits on t = t'+d-age
            src = np.arange(T)
            tgt = src + d_fm - age
            ok = (tgt >= 0) & (tgt < T)
            cred[tgt[ok]] += Kk[age] * z_f[src[ok]]
        sd = z_f.std()
        out.append(float(np.corrcoef(cred, z_f)[0, 1]) if sd > 1e-9 and cred.std() > 1e-9 else 0.0)
    return np.array(out)


def sweep_figure(R, Z, rows, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(FIGDIR, exist_ok=True)
    ds = sorted(set(d for d, _, _ in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    colors = {d: c for d, c in zip(ds, ["#e8590c", "#1971c2", "#7048e8"])}
    for d in ds:
        sub = [(tau, arm) for dd, tau, arm in rows if dd == d]
        taus = [t for t, _ in sub]
        axes[0].plot(taus, [recovery_auc(R, a) for _, a in sub], "o-", color=colors[d],
                     lw=1.8, label=f"d_fm={d}")
        axes[1].plot(taus, [np.nanmean([probe_auc(R, a, 'R1-on'), probe_auc(R, a, 'R2-off')])
                            for _, a in sub], "o-", color=colors[d], lw=1.8, label=f"d_fm={d}")
        axes[0].axvline(d, color=colors[d], ls=":", lw=1)
        axes[1].axvline(d, color=colors[d], ls=":", lw=1)
    if "fixed" in R["arms"]:
        axes[0].axhline(recovery_auc(R, "fixed"), color="#868e96", ls="--", lw=1.2, label="fixed")
        axes[1].axhline(np.nanmean([probe_auc(R, "fixed", "R1-on"),
                                    probe_auc(R, "fixed", "R2-off")]),
                        color="#868e96", ls="--", lw=1.2, label="fixed")
    for ax, ttl in ((axes[0], "ballistic_a recovery AUC (lower=better)"),
                    (axes[1], "reducible-region probe AUC (lower=better)")):
        ax.set_xscale("log"); ax.set_xlabel("tau_fast (window peak)"); ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7)
    # dense offset curve from the logged series (matched arm's own delta stream)
    A = R["config"]["a_max"]
    if Z is not None:
        for d in ds:
            match_arm = min(((abs(tau - d), arm) for dd, tau, arm in rows if dd == d))[1]
            key = f"{match_arm}__z_f"
            if key not in Z.files:
                continue
            z_f = Z[key].astype(np.float64)
            taus = np.unique(np.round(np.geomspace(1, A / 1.2, 24)).astype(int))
            fc = offset_curve(z_f, d, A, taus)
            axes[2].plot(taus, fc, "-", color=colors[d], lw=1.8, label=f"d_fm={d}")
            axes[2].axvline(d, color=colors[d], ls=":", lw=1)
        axes[2].set_xscale("log"); axes[2].set_xlabel("tau_fast")
        axes[2].set_title("credit fidelity vs window (dense, from logged series)", fontsize=9)
        axes[2].legend(fontsize=7)
    fig.suptitle("Suvrathan matching: does performance peak at tau_fast = d_fm, "
                 "and does the peak TRACK the delay?", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"fig_diagonal_{tag}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["main_s0", "sweep_s0"])
    ap.add_argument("--no-refetch", action="store_true")
    args = ap.parse_args()
    for tag in args.tags:
        dest = os.path.join(RESULTS, tag)
        try:
            R, Z = load(tag) if not args.no_refetch else load(tag)
        except Exception as ex:
            print(f"[skip] {tag}: {ex}")
            continue
        print(f"\n########## {tag} (complete={R.get('complete')}) ##########")
        if any(a.startswith("fast:") for a in R["arms"]) and \
                not any(a.startswith(("two:", "shared:")) for a in R["arms"]):
            rows = sweep_table(R)
            sweep_figure(R, Z, rows, tag)
        else:
            main_table(R)
