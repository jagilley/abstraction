"""Reduction and figures for question_model Half 1.

  python3 -m rhm.question_model.analyze_qm --tag dec0 --fetch --figures

`--fetch` pulls `decompose_<tag>.json` off the `rhm-scaling-data` volume (chromatic) into
`figures/<tag>/`. The default report is the one an orchestrator reads: the identity, the
channel sizes, the drift-off fidelity, the loss matrix, the laundering probe against its
exact ceiling, and the specificity matrix.
"""

import argparse
import json
import os
import subprocess
import sys

# Okabe-Ito subset, validated (light surface, categorical): lightness band PASS,
# chroma floor PASS, worst adjacent CVD dE 11.0 (deutan) PASS, normal-vision floor 21.1 PASS.
# The contrast WARN on the two lightest slots is relieved by direct labels + legend text in
# ink rather than in the series colour.
C_DEMAND = "#0072B2"      # blue        -- the demand / question channel
C_STRUCT = "#D55E00"      # vermillion  -- the structure channel
C_ALEA = "#999999"        # grey        -- the irreducible residue
C_MODEL = {"drift": "#0072B2", "incoh": "#E69F00", "uniform": "#009E73",
           "unigram_baseline": "#CC79A7",
           "tap_learned": "#0072B2", "tap_learned_shuffled": "#56B4E9",
           "tap_oracle": "#D55E00", "tap_shuffled": "#E69F00",
           "loss_alea": "#009E73", "loss_alea_blind": "#CC79A7", "plain": "#999999"}
INK, INK2, GRID = "#1b1b1b", "#5a5a5a", "#dcdcdc"

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/v16_s2_L6_m4_distinct/question_model"


def fetch(tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
    for name in (f"decompose_{tag}.json", f"calibration_{tag}.json"):
        dst = os.path.join(outdir, name)
        r = subprocess.run(["modal", "volume", "get", "rhm-scaling-data",
                            f"{REMOTE}/{name}", dst, "--force"],
                           env=env, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  fetched {name}")
    return outdir


def _fmt(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def report(R):
    L = R["config"]["L"]
    names = R["components"]
    print("=" * 78)
    print(f"QUESTION MODEL -- Half 1   spec {R['spec']}")
    print(f"  n_states {R['n_states']}   H(theta) {R['H_theta']:.3f} nats")
    print("=" * 78)

    print("\n-- G-7  THE THREE-WAY IDENTITY (window filter) --")
    print(f"  {'level':6s}{'total':>10s}{'demand':>10s}{'structure':>11s}"
          f"{'aleatoric':>11s}{'sum':>10s}{'err':>10s}")
    for k in sorted(R["three_way"]["per_D"], key=lambda z: int(z[1:])):
        r = R["three_way"]["per_D"][k]
        print(f"  {r['d_name']:6s}{r['total']:10.4f}{r['demand']:10.4f}"
              f"{r['structure']:11.4f}{r['aleatoric']:11.4f}{r['sum']:10.4f}"
              f"{r['abs_err']:10.2e}")
    print(f"  passed: {R['three_way']['passed']}")

    d = R["demand"]
    print("\n-- the demand channel --")
    print(f"  window  {d['Dm_window']:.5f} nats/token   "
          f"({100 * d['Dm_window'] / d['surp_marg_window']:.1f}% of total surprisal "
          f"{d['surp_marg_window']:.4f})")
    print(f"  history {d['Dm_history']:.5f} nats/token   "
          f"ratio history/window {d['Dm_history'] / max(d['Dm_window'], 1e-9):.3f}")
    print(f"  per component (window):  " + "  ".join(
        f"{k} {x:.5f}" for k, x in d["per_component_window"].items()))
    print(f"  per component (history): " + "  ".join(
        f"{k} {x:.5f}" for k, x in d["per_component_history"].items()))
    print(f"  question uncertainty H(theta) {R['H_theta']:.3f} -> "
          f"{d['H_theta_window_end']:.3f} within one block; "
          f"history prior {d['H_theta_history_start']:.3f}")

    off, on = R["drift_off"], R["drift_on"]
    print("\n-- G-8  drift-off control --")
    print(f"  fidelity vs conditional_revision/oracle.py: {off['fidelity_max_delta']:.2e}"
          f"   demand channel max|Dm| {off['Dm_max']:.2e}")
    print(f"  {'level':6s}{'S off':>10s}{'S on':>10s}{'ratio':>8s}"
          f"{'fracS0 off':>12s}{'Hirr off':>10s}{'Hirr on':>10s}")
    for lv in [f"d{L - D}" for D in range(L)]:
        so, sn = off["mean_S_by_D"][lv], on["mean_S_by_D"][lv]
        print(f"  {lv:6s}{so:10.4f}{sn:10.4f}{sn / max(so, 1e-9):8.3f}"
              f"{off['frac_S_zero_by_D'][lv]:12.3f}"
              f"{off['mean_Hirr_by_D'][lv]:10.4f}{on['mean_Hirr_by_D'][lv]:10.4f}")
    print(f"  published static reference (conditional_revision README):"
          f" d6 0.039 d5 0.061 d4 0.105 d3 0.191 d2 0.361 d1 0.699;"
          f" frac zero 0.500/0.498/0.485/0.451/0.383/0.291")

    if "loss_matrix" in R:
        print("\n-- the loss matrix (nats/token, held-out windows) --")
        ms = sorted({m for w in R["loss_matrix"].values() for m in w})
        print(f"  {'world':10s}" + "".join(f"{m:>12s}" for m in ms))
        for w, row in R["loss_matrix"].items():
            print(f"  {w:10s}" + "".join(f"{row.get(m, float('nan')):12.4f}" for m in ms))
        if "realised_question_value" in R:
            g = R["realised_question_value"]
            print(f"  realised value of the question channel "
                  f"(incoh-trained - drift-trained, on drift data): {g:+.5f}")
            print(f"  exact ceiling E[Dm_window] {d['Dm_window']:.5f}  "
                  f"-> {100 * g / max(d['Dm_window'], 1e-9):.1f}% realised")

    if "laundering" in R:
        lm = R["laundering"]
        print(f"\n-- the laundering probe (chance {lm['chance']:.3f}) --")
        cw = R["theta_ceiling"]["window"]["terminal_acc"]
        ch = R["theta_ceiling"]["history"]["terminal_acc"]
        print(f"  exact filter terminal accuracy, window : " + "  ".join(
            f"{k} {x:.3f}" for k, x in cw.items()))
        print(f"  exact filter terminal accuracy, history: " + "  ".join(
            f"{k} {x:.3f}" for k, x in ch.items()))
        for mn, bks in lm["per_model"].items():
            for bk, rr in bks.items():
                print(f"  {mn:18s} {bk:13s} terminal " + "  ".join(
                    f"{k} {x:.3f}" for k, x in rr["terminal_acc"].items()))
        print(f"  shuffled-label control: {lm['shuffled_terminal_acc']}")
        print(f"  probe / exact-ceiling (terminal, best block):")
        for mn, bb in lm["frac_of_ceiling_window"].items():
            print(f"    {mn:18s} " + "  ".join(f"{k} {x:.3f}" for k, x in bb.items()))

    if "partial_r2" in R:
        print("\n-- specificity: partial R^2 (rank) of a model signal on an oracle "
              "channel, given the marginal surprisal --")
        cols = ["Dm"] + [f"Dm_{n}" for n in names] + \
               [f"S_chain_d{k}" for k in (1, 2, 3, 6)] + ["Hirr_d1"]
        cols = [c for c in cols if c in next(iter(R["partial_r2"].values()))]
        rows = ["M_theta"] + [f"M_theta_{n}" for n in names] + \
               [f"M_struct_d{k}" for k in (1, 2, 3, 6)] + \
               ["r_temporal", "delta_norm", "nll", "M_theta_shuffled",
                "M_struct_shuffled"]
        print("  " + "signal".ljust(20) + "".join(c.rjust(13) for c in cols))
        for r in rows:
            if r not in R["partial_r2"]:
                continue
            print("  " + r.ljust(20)
                  + "".join(f"{R['partial_r2'][r][c]['rank']:13.4f}" for c in cols))

    if "matched_families" in R:
        print("\n-- matched contrast at (position x exact marginal surprisal) --")
        for lv, blk in R["matched_families"].items():
            if "demand_vs_syn" not in blk:
                continue
            print(f"  {lv}: n(syn/struct/demand) = "
                  f"{blk['n_syn']}/{blk['n_struct']}/{blk['n_demand']}")
            for cname in ("struct_vs_syn", "demand_vs_syn", "demand_vs_struct"):
                if cname not in blk:
                    continue
                e = blk[cname]
                keys = ["surp_marg_guard", "nll", "M_theta", f"M_struct_{lv}",
                        "r_temporal", "delta_norm", "M_theta_shuffled",
                        "M_struct_shuffled", "Dm_oracle", f"S_oracle_{lv}"]
                print(f"    {cname:18s}" + "  ".join(
                    f"{k.replace('_' + lv, '')} {e[k]:.3f}" for k in keys if k in e))


def figures(R, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    L = R["config"]["L"]
    names = R["components"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": GRID, "grid.linewidth": 0.6,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    # fig1 -- the three channels, per abstraction level
    per = R["three_way"]["per_D"]
    ks = sorted(per, key=lambda z: -int(z[1:]))
    lab = [per[k]["d_name"] for k in ks]
    dem = np.array([per[k]["demand"] for k in ks])
    stc = np.array([per[k]["structure"] for k in ks])
    alc = np.array([per[k]["aleatoric"] for k in ks])
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(ks))
    h = 0.55
    ax.barh(y, dem, h, color=C_DEMAND, label="demand (what is asked)")
    ax.barh(y, stc, h, left=dem + 0.004, color=C_STRUCT,
            label="structure (this sequence's parse)")
    ax.barh(y, alc, h, left=dem + stc + 0.008, color=C_ALEA,
            label="aleatoric residue")
    for i, k in enumerate(ks):
        ax.text(per[k]["total"] + 0.02, y[i], f"{per[k]['total']:.3f}", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y, lab)
    ax.set_xlabel("nats per token")
    ax.set_ylabel("latents kept, z$_{\\leq D}$")
    ax.set_title("A token's news, decomposed exactly (drift on)", loc="left")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_three_channels.png", dpi=170)
    plt.close(fig)

    # fig2 -- the demand channel within a block
    d = R["demand"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    t = np.arange(len(d["Dm_window_by_position"]))
    axes[0].plot(t, d["Dm_window_by_position"], lw=2, color=C_DEMAND,
                 label="window filter (the model's conditioning set)")
    axes[0].plot(t, d["Dm_history_by_position"], lw=2, color=C_STRUCT,
                 label="history filter (unbounded memory)")
    axes[0].set_xlabel("position within the block")
    axes[0].set_ylabel("demand revision (nats)")
    axes[0].set_title("Question news arrives early", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True)
    axes[0].set_axisbelow(True)
    pc = d["per_component_window"]
    ph = d["per_component_history"]
    x = np.arange(len(names))
    axes[1].bar(x - 0.19, [pc[n] for n in names], 0.36, color=C_DEMAND, label="window")
    axes[1].bar(x + 0.19, [ph[n] for n in names], 0.36, color=C_STRUCT, label="history")
    for i, n in enumerate(names):
        axes[1].text(i - 0.19, pc[n] + 0.0004, f"{pc[n]:.4f}", ha="center", fontsize=7,
                     color=INK2)
        axes[1].text(i + 0.19, ph[n] + 0.0004, f"{ph[n]:.4f}", ha="center", fontsize=7,
                     color=INK2)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("demand revision (nats/token)")
    axes[1].set_title("Level-resolved: one draw vs 56 draws per block", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_demand_channel.png", dpi=170)
    plt.close(fig)

    # fig3 -- the laundering probe against its exact ceiling
    if "laundering" in R:
        lm = R["laundering"]
        ceil = np.array(R["theta_ceiling"]["window"]["acc_by_position"])
        fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for c, nm in enumerate(names):
            ax = axes[c]
            ax.plot(ceil[c], lw=2, color=INK, ls="--",
                    label="exact filter (ceiling)" if c == 0 else None)
            # The honest null is the SHUFFLED-LABEL probe, not 1/K: the OU stationary
            # law over the grid is not uniform, so a majority-class head already scores
            # ~0.25 against a nominal chance of 1/7 = 0.143.
            nullv = (lm["shuffled_terminal_acc"] or {}).get(nm, lm["chance"])
            ax.axhline(nullv, color=GRID, lw=1.4, ls=":",
                       label="shuffled-label null" if c == 0 else None)
            for mn, bks in lm["per_model"].items():
                best = max(bks.values(), key=lambda rr: rr["terminal_acc"][nm])
                ax.plot(np.array(best["acc_by_position"])[c], lw=2,
                        color=C_MODEL.get(mn, INK2),
                        label=mn if c == 0 else None)
            ax.set_title(f"theta[{nm}]", loc="left")
            ax.set_xlabel("position within the block")
            ax.grid(True)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("recovery accuracy")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("Is a low-dimensional question estimate recoverable from activations?",
                     x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(f"{outdir}/fig3_laundering.png", dpi=170)
        plt.close(fig)

    # fig4 -- specificity matrix (sequential single hue, magnitude)
    if "partial_r2" in R:
        cols = ["Dm"] + [f"Dm_{n}" for n in names] + \
               [f"S_chain_d{k}" for k in (1, 2, 3, 6)] + ["Hirr_d1"]
        any_row = next(iter(R["partial_r2"].values()))
        cols = [c for c in cols if c in any_row]
        rows = [r for r in (["M_theta"] + [f"M_theta_{n}" for n in names]
                            + [f"M_struct_d{k}" for k in (1, 2, 3, 6)]
                            + ["r_temporal", "delta_norm", "nll",
                               "M_theta_shuffled", "M_struct_shuffled"])
                if r in R["partial_r2"]]
        Mx = np.array([[R["partial_r2"][r][c]["rank"] for c in cols] for r in rows])
        fig, ax = plt.subplots(figsize=(0.95 * len(cols) + 3.0, 0.42 * len(rows) + 2.0))
        im = ax.imshow(np.clip(Mx, 0, None), cmap="Blues", aspect="auto",
                       vmin=0, vmax=max(np.percentile(Mx, 98), 1e-3))
        ax.set_xticks(np.arange(len(cols)), cols, rotation=40, ha="right")
        ax.set_yticks(np.arange(len(rows)), rows)
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f"{Mx[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color=("white" if Mx[i, j] > 0.6 * im.get_clim()[1] else INK))
        ax.set_title("partial R$^2$ (rank), model signal ~ oracle channel | marginal surprisal",
                     loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7, label="partial R$^2$")
        fig.tight_layout()
        fig.savefig(f"{outdir}/fig4_specificity.png", dpi=170)
        plt.close(fig)
    print(f"\nfigures -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dec0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(HERE, "figures", a.tag)
    if a.fetch:
        fetch(a.tag, outdir)
    p = os.path.join(outdir, f"decompose_{a.tag}.json")
    if not os.path.exists(p):
        sys.exit(f"no {p}; run with --fetch")
    with open(p) as f:
        R = json.load(f)
    report(R)
    if a.figures:
        figures(R, outdir)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Half 2
# --------------------------------------------------------------------------- #

def report_h2(R):
    print("=" * 84)
    print(f"QUESTION MODEL -- Half 2   arms {R['arms']}")
    print("=" * 84)
    print(f"\n-- the slow tap's own accuracy (block-entry, before the block is seen) --")
    print(f"  {R['tap_argmax_acc']}")

    print(f"\n-- E1/E3  held-out loss, per level, and gradient economics --")
    e = R["economics"]
    lv = sorted({k for a in e.values() for k in a["nll_by_level"]})
    print(f"  {'arm':18s}{'val nll':>10s}{'grad norm':>11s}"
          + "".join(f"{k:>9s}" for k in lv) + "  gradIrred  nllIrred")
    for a, r in e.items():
        print(f"  {a:18s}{r['val_nll']:10.4f}{r['val_gradnorm']:11.4f}"
              + "".join(f"{r['nll_by_level'].get(k, float('nan')):9.3f}" for k in lv)
              + f"{r['frac_gradnorm_irreducible']:11.4f}"
              f"{r['frac_nll_irreducible']:10.4f}")

    s = R["sk_setup"]
    print(f"\n-- E2  epistemic self-knowledge under demand shift --")
    print(f"  regions by theta_hi: A {s['n_A']} blocks / B {s['n_B']} "
          f"(fit {s['n_B_fit']}, test {s['n_B_test']});  demand distance A->B "
          f"{s['kl_A_to_B']:.3f} nats/block vs within-A {s['kl_within_A']:.3f}")
    for arm, r in R["self_knowledge"].items():
        ie = r["internal_estimate"]
        print(f"\n  [{arm}]  own theta estimate on B_test: root {ie['acc_on_B_test']['root']:.3f}"
              f"  hi {ie['acc_on_B_test']['hi']:.3f}   <- {ie['source']}")
        print(f"    token accuracy: A {r['accuracy_A']:.4f} -> B_test {r['accuracy_B_test']:.4f}")
        keys = sorted(r["scores"])
        print(f"    {'predictor':42s}{'AUC':>8s}{'Brier':>9s}{'ECE':>8s}")
        for k in keys:
            sc = r["scores"][k]
            print(f"    {k:42s}{sc['auc']:8.4f}{sc['brier']:9.4f}{sc['ece']:8.4f}")

    print(f"\n-- E4  does delta_norm's demand-tracking survive richer controls? --")
    e4 = R["delta_norm_control"]
    for k, val in e4.items():
        if isinstance(val, dict) and "delta_norm~Dm" in val:
            print(f"  {k:36s} Dm {val['delta_norm~Dm']:.4f}   Dm_lo {val['delta_norm~Dm_lo']:.4f}")


def main_h2():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h2")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(HERE, "figures", a.tag)
    if a.fetch:
        os.makedirs(outdir, exist_ok=True)
        env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
        for name in (f"h2_{a.tag}_evaluate.json", f"h2_{a.tag}_precompute.json"):
            subprocess.run(["modal", "volume", "get", "rhm-scaling-data",
                            f"{REMOTE}/{name}", os.path.join(outdir, name), "--force"],
                           env=env, capture_output=True, text=True)
    with open(os.path.join(outdir, f"h2_{a.tag}_evaluate.json")) as f:
        report_h2(json.load(f))


def figures_h2(R, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": GRID, "grid.linewidth": 0.6,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    sk = R["self_knowledge"]
    arms = list(sk)
    rows = [("output entropy (incumbent)", "entropy_incumbent", C_ALEA),
            ("activation probe, frozen on A", "probe_activation::frozen_A", C_STRUCT),
            ("activation probe, maintained", "probe_activation::maintained_Bfit", C_STRUCT),
            ("ledger: family only, frozen", "ledger_family::frozen_A", C_DEMAND),
            ("ledger: + demand key, frozen", "ledger_demand_true::frozen_A", C_DEMAND),
            ("ledger: + demand key, maintained", "ledger_demand_true::maintained_Bfit",
             C_DEMAND)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(rows))
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        vals = [sk[a]["scores"][k]["auc"] for _, k, _ in rows]
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, vals, w * 0.92,
               color=C_MODEL.get(a, INK2), label=a)
    ax.axhline(0.5, color=GRID, lw=1.2, ls=":")
    ax.set_xticks(x, [r[0] for r in rows], rotation=18, ha="right")
    ax.set_ylim(0.45, 0.90)
    ax.set_ylabel("AUC, predicting own per-token success on region B")
    ax.set_title("Epistemic self-knowledge after a demand shift", loc="left")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig5_self_knowledge.png", dpi=170)
    plt.close(fig)

    e = R["economics"]
    lv = sorted(next(iter(e.values()))["nll_by_level"])
    base = np.array([e["plain"]["nll_by_level"][k] for k in lv])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for a in ("tap_oracle", "tap_shuffled", "tap_learned", "tap_learned_shuffled",
              "loss_alea", "loss_alea_blind"):
        if a not in e:
            continue
        vv = np.array([e[a]["nll_by_level"][k] for k in lv])
        ax.plot(np.arange(len(lv)), 100 * (vv - base) / base, marker="o", lw=2,
                label=a, color=C_MODEL.get(a, None))
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(np.arange(len(lv)), [f"lvl{i}" for i in range(len(lv))])
    ax.set_xlabel("hierarchy level completed by the arriving token (0 = root, 6 = leaf)")
    ax.set_ylabel("% change in nll vs the plain twin")
    ax.set_title("Where each intervention moves the loss", loc="left")
    ax.grid(True); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig6_level_tradeoff.png", dpi=170)
    plt.close(fig)
    print(f"figures -> {outdir}")
