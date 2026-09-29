"""[grokking/whittle] Reduce a `whittle.py::whittle_run` run to the table of record, the figure and a summary.

    python3 grokking/whittle/reduce_whittle.py --tag w1 --fetch      # from experiments/

--fetch pulls /data/<tag>/whittle.json off `grokking-mint-data` into results/<tag>/whittle.json (git-ignored;
the volume copy is the artifact, with each arm's final state in /data/<tag>/<arm>_final.npz). Writes:
    figures/whittle_<tag>_table.txt      table of record: setup, reference sizes, per-arm gates / start / final,
                                         the comparison table, admitted-round trajectories, surviving structure
    figures/whittle_<tag>_survivors.png  per arm: held-out vs surviving weights over admitted rounds (top) and
                                         per-layer survivors vs round (bottom)
    results/whittle_<tag>_summary.json   compact per-arm series behind the figure
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = [("original net (one-hot, 3 layers)", 53985), ("form B, unit-pruned to K13, lt1", 45264),
        ("form C, layer 0 filtered to DC + K13, lt1", 36065), ("SVD truncation, rank 36", 29261),
        ("SVD truncation, rank 33 (0.99 held-out)", 26852),
        ("bilinear top1, width 4, wd <= 0.1 (Rung controls B2)", 501),
        ("minted-13 bilinear, width 52 (Rung r1)", 7845),
        ("amplitude-fit closed form over the recovered basis (README update 6)", 20)]
ARM_ORDER = ("committed", "unrestricted", "both_sides", "all48", "committed_wd1", "bilinear_both", "committed_R2000")
LAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3 (dataviz reference palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _get(vol_path, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE="chromatic")
    subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", vol_path, local], check=True, env=env)


def load(tag, fetch):
    path = os.path.join(HERE, "results", tag, "whittle.json")
    if fetch:
        _get(f"{tag}/whittle.json", path)
    with open(path) as f:
        return json.load(f)


def layers_of(r):
    return ["U", "V", "W"] if r["cfg"]["learner"] == "bilinear" else ["layer0", "layer1", "head"]


def fmt(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def admitted(r):
    return [x for x in r["walk"]["rounds"] if x["admit"]]


def series(r):
    """Start state + every admitted round: (round, weights, per-layer weights, heldout, gbar_margin, ...)."""
    Ls = layers_of(r)
    s = r["start"]
    out = [{"round": 0, "weights": s["weights"], **{L: s[f"{L}_w"] for L in Ls},
            **{f"{L}_g": s[f"{L}_g"] for L in Ls}, "heldout": s["heldout"],
            "gbar_margin": s["probe"]["gbar_margin"], "margin_te_min": s["probe"]["margin_te_min"],
            "effective": s["effective"], "units0": s.get("units0", s.get("units")), "units1": s.get("units1")}]
    for x in admitted(r):
        c = x["counts"]
        out.append({"round": x["round"], "layer": x["layer"], "n_prune": x["n_prune"], "weights": c["weights"],
                    **{L: c[f"{L}_w"] for L in Ls}, **{f"{L}_g": c[f"{L}_g"] for L in Ls},
                    "heldout": x["heldout"], "gbar_margin": x["probe"]["gbar_margin"],
                    "margin_te_min": x["probe"]["margin_te_min"], "effective": c["effective"],
                    "units0": c.get("units0", c.get("units")), "units1": c.get("units1"),
                    "first_ok": x["first_ok_epoch"], "post": x["agree_post_prune"]})
    return out


def rejected(r):
    return [{"round": x["round"], "layer": x["layer"], "n_prune": x["n_prune"], "frac": x["frac"],
             "n_disagree": x["n_disagree"], "heldout": x["heldout"], "post": x["agree_post_prune"],
             "attempted": None} for x in r["walk"]["rounds"] if not x["admit"]]


def write_table(d, tag, path):
    R = d["results"]
    arms = [a for a in ARM_ORDER if a in R]
    lines = []
    w = lines.append
    any_r = R[arms[0]]
    w(f"# grokking / whittle -- tag {tag}; seed 42; m1's banked final net (e39999); CPU; one container per arm")
    w(f"# retrain per round: fresh AdamW lr 1e-3, full batch, targets = the original net's train argmax (= the train labels);"
      f" stops once train agreement has held {d.get('settle')} consecutive epochs, at most {d['retrain']} epochs"
      f" (committed_R2000: at most 2000); f0 {d['f0']}, halved on rejection, floor {d['floor']}; max rounds {d['max_rounds']}")
    w(f"# gate: admit iff 0 of 2,822 train pairs disagree with the original net's argmax. Held-out (6,587 pairs) and the"
      f" probe are logged, never consumed.")
    w(f"# minted coordinates: Pm = QR([u0 | Q_j for the kept 13, producer order]); feature scale s = sqrt(97/2) ="
      f" {any_r['s_feat']:.4f} (unit-amplitude waves)")
    km = [m["k_dft"] for m in R["committed"]["meta"][1:]] if "committed" in R else []
    w(f"# kept candidates K (form C, lt1, reproduced by _walk_K): {R[arms[0]]['K']}; their DFT match (oracle, logged"
      f" only), in producer order: {km}")
    w("# probe (rung_controls.make_probe, all p^2 pairs): gbar_margin = gbar(0) - max_{d!=0} gbar(d) of the"
      " diagonal-averaged row-centred logit; margin = logit[true] - max other")
    w("")
    w("## Reference sizes (parameters)")
    for n, v in REFS:
        w(f"  {v:>7,d}  {n}")
    w("")
    w("## Gates at initialization (G1) and recovery")
    w(f"{'arm':>16s} {'reference':>44s} {'max|dlogit|':>11s} {'train agr':>9s} {'dis':>4s} {'held-out':>8s} "
      f"{'params':>7s} {'gram off':>8s} {'QR chg':>8s} {'sym ov min':>10s} | recovery")
    for a in arms:
        r = R[a]
        g = r["init"]
        bi = r["basis_info"]
        rc = r["recover"]
        rtxt = "not needed" if rc is None else (f"{rc['chunks']} x {r['retrain']} epochs, recovered={rc['recovered']}")
        extra = ""
        if "agree_unprojected_ref" in g:
            extra = f" (unprojected ref agrees {g['agree_unprojected_ref']:.4f})"
        if "bil_rot_resid" in bi:
            extra += f" (input rotation resid {bi['bil_rot_resid']:.1e}, orth err {bi['bil_rot_orth_err']:.1e})"
        w(f"{a:>16s} {g['ref'][:44]:>44s} {g['maxabs_vs_ref']:>11.2e} {g['train_agree']:>9.4f} {g['n_disagree']:>4d} "
          f"{g['heldout']:>8.4f} {r['n_params_init']:>7,d} {bi['gram_offdiag_max']:>8.1e} "
          f"{bi['qr_col_change_max']:>8.1e} {bi['sym_overlap_min']:>10.7f} | {rtxt}{extra}")
    w("")
    w("## Final state per arm (weights = surviving weight scalars; groups = symbol-connections in minted layers,"
      " scalars elsewhere; effective = weights + live biases + head bias)")
    hdr = (f"{'arm':>16s} {'wd':>3s} {'start w':>8s} {'final w':>8s} {'L0 w/g':>12s} {'L1 w':>6s} {'head w/g':>11s} "
           f"{'effective':>9s} {'units':>7s} {'held-out':>8s} {'gbar_m':>7s} {'te m min':>8s} {'diag E':>6s} {'dc E':>5s} "
           f"{'n90':>3s} {'rounds a/r':>10s} {'sec':>5s}")
    w(hdr)
    for a in arms:
        r = R[a]
        f, s = r["final"], r["start"]
        Ls = layers_of(r)
        na = len(admitted(r))
        nr = r["walk"]["n_rounds"] - na
        le = f["logit_energy"]
        units = f"{f.get('units0', f.get('units'))}/{f.get('units1', '-')}"
        w(f"{a:>16s} {r['cfg']['wd']:>3g} {s['weights']:>8,d} {f['weights']:>8,d} "
          f"{f[Ls[0] + '_w']:>6,d}/{f[Ls[0] + '_g']:<5,d} {f[Ls[1] + '_w']:>6,d} "
          f"{f[Ls[2] + '_w']:>5,d}/{f[Ls[2] + '_g']:<5,d} {f['effective']:>9,d} {units:>7s} {f['heldout']:>8.4f} "
          f"{f['probe']['gbar_margin']:>7.3f} {f['probe']['margin_te_min']:>8.3f} {le['diag_total']:>6.3f} "
          f"{le['dc_frac']:>5.3f} {le['n90']:>3d} {na:>4d}/{nr:<5d} {r['seconds']:>5.0f}")
    w("  (bilinear_both: L0 = U, L1 = V, head = W; its units column is product units)")
    w("")
    w("## Walk end state and compute profile per arm")
    for a in arms:
        r = R[a]
        st = r["stats"]
        ff = {k: round(v, 5) for k, v in r["walk"]["frac_final"].items()}
        w(f"{a:>16s}: rounds {r['walk']['n_rounds']} (cap hit: {r['walk']['hit_cap']}), final fractions {ff}; "
          f"retrain {st['t_retrain']:.0f}s, probe {st['t_probe']:.0f}s, bookkeeping {st['t_book']:.1f}s, total "
          f"{r['seconds']:.0f}s; folds (constant units -> next bias) L0 {st['n_fold0']} L1 {st['n_fold1']}; "
          f"peak RSS {r['peak_rss_mb']:.0f} MB")
    w("")
    w("## Drift twin: the start state, nothing pruned, retrained with the admitted rounds' epoch counts (fresh AdamW per"
      " chunk). Held-out of the walk vs the twin after the same cumulative retraining")
    w(f"{'arm':>16s} {'chunks':>6s} {'epochs':>7s} | {'twin ho min':>11s} {'median':>7s} {'final':>7s} {'n_dis max':>9s} | "
      f"{'walk ho min':>11s} {'median':>7s} {'final':>7s}")
    for a in arms:
        r = R[a]
        tw = r.get("drift_twin", [])
        if not tw:
            continue
        th = [x["heldout"] for x in tw]
        wh = [x["heldout"] for x in admitted(r)]
        w(f"{a:>16s} {len(tw):>6d} {sum(x['epochs'] for x in tw):>7d} | {min(th):>11.4f} {np.median(th):>7.4f} "
          f"{th[-1]:>7.4f} {max(x['n_disagree'] for x in tw):>9d} | {min(wh):>11.4f} {np.median(wh):>7.4f} {wh[-1]:>7.4f}")
    w("")
    w("## Logit-table Fourier energy at the end (mint.logit_table_energy on all p^2 pairs; DFT, oracle read)")
    for a in arms:
        r = R[a]
        le0 = r["final"]["logit_energy"]
        w(f"{a:>16s}: diagonal {le0['diag_total']:.4f}, DC {le0['dc_frac']:.4f}, n90 {le0['n90']}, n99 {le0['n99']}, "
          f"top frequencies {le0['top_k'][:13]}")
    w("")
    for a in arms:
        r = R[a]
        Ls = layers_of(r)
        w(f"## Trajectory: {a} (start + admitted rounds; rejected rounds listed after)")
        w(f"{'round':>5s} {'layer':>6s} {'-n':>6s} {'weights':>8s} " + " ".join(f"{L:>7s}" for L in Ls) +
          f" {'L0 grp':>7s} {'units':>7s} {'held-out':>8s} {'gbar_m':>7s} {'te m min':>8s} {'post':>6s} {'ok@':>4s}")
        for x in series(r):
            w(f"{x['round']:>5d} {x.get('layer', 'start'):>6s} {x.get('n_prune', 0):>6d} {x['weights']:>8,d} " +
              " ".join(f"{x[L]:>7,d}" for L in Ls) +
              f" {x[Ls[0] + '_g']:>7,d} {str(x['units0']) + '/' + str(x['units1'] if x['units1'] is not None else '-'):>7s}"
              f" {x['heldout']:>8.4f} {x['gbar_margin']:>7.3f} {x['margin_te_min']:>8.3f} "
              f"{fmt(x.get('post'), 3):>6s} {fmt(x.get('first_ok')):>4s}")
        rj = rejected(r)
        w(f"  rejected ({len(rj)}): " + "; ".join(
            f"r{x['round']} {x['layer']} -{x['n_prune']} (f {x['frac']:.4f}) dis {x['n_disagree']} ho {x['heldout']:.4f}"
            for x in rj))
        w("")
    w("## Surviving structure")
    for a in arms:
        r = R[a]
        s = r["structure"]
        meta = r["meta"]
        kd = s["sym_k_dft"]
        w(f"### {a}")
        if r["cfg"]["learner"] == "bilinear":
            w(f"  product units live {s['n_units']}, single-symbol {s['n_single']}; symbols read {s['read_syms']} "
              f"(DFT k {s['read_k_dft']}); symbols written {s['write_syms']}")
            for u in s["units"]:
                w(f"    unit {u['u']:>3d}: U {u['U']} V {u['V']} W {u['W']}  (k: U {[kd[t] for t in u['U']]}, "
                  f"W {[kd[t] for t in u['W']]})")
            w("")
            continue
        w(f"  layer-0 units connected {s['n_units0']}, layer-1 units connected {s['n_units1']}")
        if "n_single" in s:
            w(f"  layer 0 (minted): single-symbol units {s['n_single']} / {s['n_units0']}; same symbols on a and b "
              f"{s['n_same_ab']} / {s['n_units0']}; distinct symbols read across the net {len(set(s['read_syms']) - {0})}"
              f" (+DC: {0 in s['read_syms']}); DFT k of the symbols read (oracle) {s['read_k_dft']}")
        else:
            w(f"  layer 0 (one-hot, read by projection on the full recovered basis): distinct dominant symbols "
              f"{len(s['read_syms'])}; their DFT k (oracle) {s['read_k_dft']}; median concentration on the dominant "
              f"symbol {np.median([u['conc'] for u in s['units0']]) if s['units0'] else float('nan'):.3f}; median "
              f"surviving residues per unit a/b {np.median([u['n_res_a'] for u in s['units0']]) if s['units0'] else 0:.0f}/"
              f"{np.median([u['n_res_b'] for u in s['units0']]) if s['units0'] else 0:.0f}")
        w(f"  layer 1: units whose layer-0 inputs share one symbol {s['n_l1_single_sym']} / {s['n_units1']}; median fan-in "
          f"{s['l1_fan_in_median']}")
        if "write_syms" in s:
            w(f"  head (minted): symbols written {s['write_syms']} (DFT k {[kd[t] for t in s['write_syms']]}); layer-1 "
              f"units that write only symbols they read {s['n_l1_writes_own_read']} / {s['n_units1']}")
        ps = [p for p in s["per_sym"] if p["n_units0"] > 0]
        w("  layer-0 units per dominant symbol: " + ", ".join(
            f"s{p['sym']}(k{p['k_dft']}):{p['n_units0']}" for p in ps))
        pc = [p for p in s["per_sym"] if p.get("n_conn_a", 0) + p.get("n_conn_b", 0) > 0]
        if pc:
            w("  surviving (unit, operand) connections per symbol, a/b: " + ", ".join(
                f"s{p['sym']}(k{p['k_dft']}):{p['n_conn_a']}/{p['n_conn_b']}" for p in pc))
        if "n_single" in s:
            w("  per layer-0 unit (symbols read on a | on b; s0 = DC):")
            for u in s["units0"]:
                w(f"    unit {u['u']:>3d}: a {u['a']} | b {u['b']}  k: a {[kd[t] for t in u['a']]} | b "
                  f"{[kd[t] for t in u['b']]}")
        else:
            w("  per layer-0 unit (dominant symbol, concentration, surviving residues a/b):")
            for u in s["units0"]:
                w(f"    unit {u['u']:>3d}: s{u['dom']} (k {kd[u['dom']]}) conc {u['conc']:.3f} res {u['n_res_a']}/{u['n_res_b']}")
        w("  per layer-1 unit (fan-in, dominant symbols of its inputs" + (", symbols written" if "write_syms" in s else
                                                                          ", classes written") + "):")
        for u in s["units1"]:
            tail = f"writes {u['writes']}" if "writes" in u else f"classes {u['n_classes']}"
            w(f"    unit {u['u']:>3d}: fan-in {u['fan_in']:>3d} in_syms {u['in_syms']} {tail}")
        if a == "all48":
            K13 = sorted(m["k_dft"] for m in R["committed"]["meta"][1:]) if "committed" in R else []
            rk = s["read_k_dft"]
            w(f"  all48 vs the mint's 13 (oracle): symbols read {len(set(rk) - {0})}; in K13 {sorted(set(rk) & set(K13))}; "
              f"outside K13 {sorted(set(rk) - set(K13) - {0})}; K13 not read {sorted(set(K13) - set(rk))}")
        w("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def _logticks(axis, lo, hi):
    """Major ticks at 1-2-5 steps inside [lo, hi], compact labels, no minor labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    cands = [m * 10 ** e for e in range(0, 6) for m in (1, 2, 5)]
    ticks = [t for t in cands if lo / 1.05 <= t <= hi * 1.05] or [lo, hi]
    if len(ticks) > 6:
        ticks = [t for t in ticks if str(t)[0] in "15"]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def write_figure(d, tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = d["results"]
    arms = [a for a in ARM_ORDER if a in R]
    n = len(arms)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.4), squeeze=False)
    for j, a in enumerate(arms):
        r = R[a]
        S = series(r)
        Ls = layers_of(r)
        ax = axes[0, j]
        x = [s["weights"] for s in S]
        y = [s["heldout"] for s in S]
        tw = {t["round"]: t["heldout"] for t in r.get("drift_twin", [])}
        if tw:
            xt = [s_["weights"] for s_ in S[1:] if s_["round"] in tw]
            yt = [tw[s_["round"]] for s_ in S[1:] if s_["round"] in tw]
            ax.plot(xt, yt, "-", color="#a3a29d", lw=1.2, zorder=2, label="drift twin (nothing pruned)")
        ax.plot(x, y, "-", color=LAYER_COLORS[0], lw=2, zorder=3, label="walk (admitted rounds)")
        ax.plot(x, y, "o", color=LAYER_COLORS[0], ms=4, mec="white", mew=1, zorder=4)
        ax.plot([x[0]], [y[0]], "s", color=INK, ms=5, zorder=5)
        ax.plot([x[-1]], [y[-1]], "D", color=INK, ms=5, zorder=5)
        ax.set_xscale("log")
        _logticks(ax.xaxis, min(x), max(x))
        ax.invert_xaxis()
        ax.set_title(f"{a} (wd {r['cfg']['wd']:g})\nend: {x[-1]:,d} weights, held-out {y[-1]:.4f}", fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(labelsize=7.5, colors=INK2)
        for ref, lab in ((29261, "SVD r36"), (501, "501")):
            if min(x) * 0.8 < ref < max(x) * 1.2:
                ax.axvline(ref, color=INK2, lw=0.8, ls=":")
                ax.text(ref, 1.0, lab, fontsize=6.5, color=INK2, rotation=90, va="top", ha="right",
                        transform=ax.get_xaxis_transform())
        lo = min(y + (yt if tw else []))
        ax.set_ylim(min(0.99, lo - 0.002), 1.0008)
        if j == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        if j == 0:
            ax.set_ylabel("held-out accuracy (oracle, logged)", fontsize=8, color=INK2)
        ax.set_xlabel("surviving weights (log, decreasing)", fontsize=8, color=INK2)
        ax2 = axes[1, j]
        rr = [s["round"] for s in S]
        for L, c in zip(Ls, LAYER_COLORS):
            ax2.plot(rr, [s[L] for s in S], "-", color=c, lw=2, label=L)
        rj = rejected(r)
        for x_ in rj:
            ax2.axvline(x_["round"], color="#c9c8c3", lw=0.7, zorder=0)
        ax2.set_yscale("log")
        _logticks(ax2.yaxis, min(min(s_[L] for s_ in S) for L in Ls), max(max(s_[L] for s_ in S) for L in Ls))
        ax2.grid(True, axis="y", color=GRID, lw=0.6)
        ax2.tick_params(labelsize=7.5, colors=INK2)
        ax2.set_xlabel("round (grey ticks: rejected)", fontsize=8, color=INK2)
        if j == 0:
            ax2.set_ylabel("surviving weights per layer", fontsize=8, color=INK2)
        ax2.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"whittle {tag}: prune-and-retrain under the train-agreement gate, one panel column per arm",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary(d, tag, path):
    R = d["results"]
    out = {"tag": tag, "retrain": d["retrain"], "f0": d["f0"], "floor": d["floor"], "arms": {}}
    for a in [a for a in ARM_ORDER if a in R]:
        r = R[a]
        f = r["final"]
        s = r["structure"]
        out["arms"][a] = {
            "cfg": r["cfg"], "n_params_init": r["n_params_init"], "init": {k: r["init"][k] for k in
                                                                           ("maxabs_vs_ref", "train_agree", "heldout")},
            "recover": None if r["recover"] is None else {k: r["recover"][k] for k in ("chunks", "epochs", "recovered")},
            "final": {k: v for k, v in f.items() if k not in ("probe", "logit_energy")},
            "final_probe": {k: v for k, v in f["probe"].items()},
            "final_logit_energy": {k: f["logit_energy"][k] for k in ("diag_total", "dc_frac", "n90", "n99", "top_k")},
            "structure": {k: v for k, v in s.items() if k not in ("units0", "units1", "units", "per_sym")},
            "series": series(r), "rejected": rejected(r), "drift_twin": r.get("drift_twin"),
            "seconds": r["seconds"], "stats": r["stats"]}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--local", default=None, help="read this whittle.json instead (no fetch)")
    args = ap.parse_args()
    if args.local:
        with open(args.local) as f:
            d = json.load(f)
    else:
        d = load(args.tag, args.fetch)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    t = os.path.join(HERE, "figures", f"whittle_{args.tag}_table.txt")
    write_table(d, args.tag, t)
    write_figure(d, args.tag, os.path.join(HERE, "figures", f"whittle_{args.tag}_survivors.png"))
    write_summary(d, args.tag, os.path.join(HERE, "results", f"whittle_{args.tag}_summary.json"))
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
