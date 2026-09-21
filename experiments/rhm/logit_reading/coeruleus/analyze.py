"""Local reduction of the coeruleus JSONs pulled off the volume. No Modal, no GPU.

  python -m rhm.logit_reading.coeruleus.analyze <dir> [--figs rhm/logit_reading/coeruleus/figs]

<dir> holds the JSONs in the same relative layout as the volume, e.g.
  traj_eps01_s42/step064000_prize_d0.01.json
  traj_eps01_s42/step064000_coeruleus_readout.json
  traj_eps01_s42/step064000_coeruleus_gain.json
  coeruleus/cont_c1/<arm>/step024000_{prize_d0.01,calibration,dual_noisy0.01}.json
"""

import argparse
import glob
import json
import os

import numpy as np

VENUES = [
    ("traj_eps01_s42/step064000", "eps-trained 64k"),
    ("traj_eps01_s42/step024000", "eps-trained 24k"),
    ("traj_eps01_s42/step008000", "eps-trained 8k"),
    ("traj_a1_s42/step064000", "clean-trained 64k"),
]


def load(root, rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def sep(n):
    return "|" + "|".join(["---"] * n) + "|"


# ---------------------------------------------------------------------------

def q1(root, out):
    out.append("## Q1. The prize after a corrupted token, and what a one-knob consumer reaches\n")
    for rel, name in VENUES:
        r = load(root, f"{rel}_prize_d0.01.json")
        if r is None:
            continue
        out.append(f"### {name} (kappa*_eps = {r['kappa_star_eps']:.2f}, "
                   f"{r['n_events']} isolated events, H = {r['H']})\n")
        out.append(row("tau", "H(q)", "H(p_L^eps)", "excess vs truth", "excess vs ceiling",
                       "alpha*", "temp recovers", "w*", "floor recovers", "affine recovers"))
        out.append(sep(10))
        for d in r["by_tau"]:
            af = d["affine"]
            out.append(row(d["tau"], f"{d['H_q']:.4f}", f"{d['H_truth']:.4f}",
                           f"{d['excess_truth']:+.4f}", f"{d['excess_ceil']:+.4f}",
                           f"{d['knob_alpha_pooled']:.3f}",
                           f"{d['CE_model'] - d['knob_CE_alpha_pooled']:+.4f}",
                           f"{d['knob_w_pooled']:.4f}",
                           f"{d['CE_model'] - d['knob_CE_w_pooled']:+.4f}",
                           f"{af['CE_te_base'] - af['CE_te_affine']:+.4f}"))
        q = r["quiet"]
        out.append(row("quiet", f"{q['H_q']:.4f}", f"{q['H_truth']:.4f}",
                       f"{q['excess_truth']:+.4f}", f"{q['excess_ceil']:+.4f}",
                       f"{q['knob_alpha_pooled']:.3f}",
                       f"{q['CE_model'] - q['knob_CE_alpha_pooled']:+.4f}",
                       f"{q['knob_w_pooled']:.4f}",
                       f"{q['CE_model'] - q['knob_CE_w_pooled']:+.4f}", "-"))
        ew = r["event_window"]
        rc = ew["recovered"]
        out.append("")
        out.append(f"Prize over tau = 0..{r['H']}: **{ew['prize_nats_per_pred']:.4f} nats per "
                   f"prediction** ({ew['prize_nats_per_event']:.3f} per event); quiet background "
                   f"{q['excess_ceil']:.4f}. Fraction recovered:\n")
        out.append(row("consumer", "temperature", "noise floor"))
        out.append(sep(3))
        for k, lab in (("global", "one knob for the whole stream"),
                       ("per_offset", "one knob per post-event offset"),
                       ("per_event_oracle", "one knob per event (oracle)"),
                       ("per_pred_oracle", "one knob per prediction (oracle)")):
            out.append(row(lab, f"{rc['temp_' + k]['frac']:+.3f}", f"{rc['floor_' + k]['frac']:+.3f}"))
        out.append("")
        d0 = r["by_tau"][1]
        out.append("Mixing the forecast toward observer k at tau = 0 (diagnostic; the model does "
                   "not have these at inference, and mixing toward `p_kappa*` reaches the ceiling "
                   "by construction):\n")
        ks = sorted(d0["mix_obs"], key=int)
        out.append(row("k", *ks))
        out.append(sep(len(ks) + 1))
        out.append(row("optimal w", *[f"{d0['mix_obs'][k]['w']:.3f}" for k in ks]))
        out.append(row("nats recovered", *[f"{d0['CE_model'] - d0['mix_obs'][k]['CE']:+.4f}"
                                           for k in ks]))
        out.append("")


HEAD_ORDER = ["state_excess_ridge", "state_excess_mlp", "state_rawnll_ridge", "state_rawnll_mlp",
              "embed_excess_ridge", "rand_excess_ridge", "rand_rawnll_ridge",
              "Hsum_raw", "Hq_raw"]


def q2(root, out):
    out.append("## Q2. The self-supervised readout of the model's own excess surprise\n")
    for rel, name, tag in (("traj_eps01_s42/step064000", "eps-trained 64k", ""),
                           ("traj_eps01_s42/step008000", "eps-trained 8k", ""),
                           ("traj_a1_s42/step064000", "clean-trained 64k, corrupted stream",
                            "_onnoise")):
        r = load(root, f"{rel}_coeruleus_readout{tag}.json")
        if r is None:
            continue
        out.append(f"### {name}\n")
        out.append(f"Target: sum over tau=0..{r['H']} of (NLL - H(q)); mean "
                   f"{r['target_mean']:+.4f}, sd {r['target_sd']:.3f} nats, correlation with the "
                   f"raw future NLL {r['corr_target_rawnll']:+.3f}. "
                   f"{r['n_rows_train']:,} training positions, none labelled. "
                   f"Best block {r.get('best_ridge', {}).get('state_excess_ridge', {}).get('block')}.\n")
        out.append(row("readout", "val R2", "rank corr", "AUC(event vs quiet)",
                       "hard vs overconfident-wrong", "same-prefix pairs (paired)"))
        out.append(sep(6))
        pr = r.get("pairs", {}).get("readouts", {})
        for k in HEAD_ORDER:
            h = r["heads"].get(k)
            if h is None:
                continue
            out.append(row(k, f"{h['R2']:+.6f}" if abs(h["R2"]) < 1 else "n/a",
                           f"{h['spearman']:+.4f}",
                           f"{h.get('auc_event_vs_quiet', float('nan')):.3f}",
                           f"{h.get('hard_vs_wrong', {}).get('auc', float('nan')):.3f}",
                           f"{pr.get(k, {}).get('paired', float('nan')):.3f}"))
        for k in ("H_next", "oracle_linear", "oracle_mlp"):
            if k in pr:
                out.append(row(k + " (reference)", "-", "-", "-", "-",
                               f"{pr[k]['paired']:.3f}"))
        p = r.get("pairs", {})
        out.append("")
        out.append(f"Pairs: {p.get('n_matched')} matched, {p.get('n_test')} held-out; "
                   f"prefix identity check {p.get('prefix_check', float('nan')):.1e}. "
                   f"`oracle_*` are probes on the SAME states trained with the violation label.\n")
        h = r["heads"].get("state_excess_ridge", {})
        if "by_tau" in h:
            ks = sorted(h["by_tau"], key=lambda x: int(x))
            out.append(row("tau", *ks))
            out.append(sep(len(ks) + 1))
            for k in HEAD_ORDER[:2] + ["rand_excess_ridge"]:
                hh = r["heads"].get(k, {})
                if "by_tau" in hh:
                    out.append(row(k, *[f"{hh['by_tau'][t]:+.3f}" for t in ks]))
            out.append("")


GAIN_ORDER = ["base", "const", "head_ridge", "head_ridge_loglin", "head_mlp", "head_mlp_loglin",
              "oracle_probe", "oracle_offset"]


def q3(root, out):
    out.append("## Q3. The gain loop\n")
    for rel, name, tag in (("traj_eps01_s42/step064000", "eps-trained 64k", ""),
                           ("traj_a1_s42/step064000", "clean-trained 64k, corrupted stream",
                            "_onnoise")):
        r = load(root, f"{rel}_coeruleus_gain{tag}.json")
        if r is None:
            continue
        out.append(f"### {name} (prize {r['prize_nats_per_pred']:.4f} nats/prediction over "
                   f"{r['n_events']} events; {r.get('prize_nats_per_pred_isolated', float('nan')):.4f} "
                   f"over the {r.get('n_events_isolated')} isolated ones)\n")
        out.append(f"Maps are fit on held-out corrupted windows by the model's own realised NLL, "
                   f"at window positions t >= {r['fit'].get('t_lo')} only -- the positions they are "
                   f"applied at. A negative cost is an improvement.\n")
        b = r["arms"]["base"]
        out.append(row("arm", "alpha at tau=0", "prize recovered (vs the eps-observer ceiling)",
                       "d realised NLL, event window", "d realised NLL, quiet",
                       "d realised NLL, whole stream", "H(q) after event"))
        out.append(sep(7))
        for k in GAIN_ORDER:
            a = r["arms"].get(k)
            if a is None:
                continue
            out.append(row(k, f"{a['by_tau']['0']['alpha']:.3f}",
                           f"{a['frac_prize_recovered']:+.3f}",
                           f"{a['event_nll'] - b['event_nll']:+.4f}",
                           f"{a['quiet_nll'] - b['quiet_nll']:+.5f}",
                           f"{a['nll_all'] - b['nll_all']:+.5f}",
                           f"{a['by_tau']['0']['H_q']:.4f}"))
        out.append("")
        out.append("The two loss columns can disagree in sign, and the reason is not noise. "
                   "The post-event population is selected on a latent -- whether the token at t "
                   "was corrupted -- that no observer of the window can see, so the conditional "
                   "law of the realised token there is NOT `p_L^eps`. Against the ceiling a "
                   "window-observer could actually reach, the prize column is the honest one; "
                   "on realised tokens given side information about the latent, the NLL columns "
                   "are.\n")
        out.append(f"Same-prefix pairs ({r['pairs']['n_test']} held-out), paired win rate that "
                   "the forecast after the ILLEGAL token is LESS confident than after its legal "
                   "twin (> 0.5 = the normative direction):\n")
        out.append(row("arm", "H_next paired", "alpha viol / twin"))
        out.append(sep(3))
        for k, v in r["pairs"]["readouts"].items():
            out.append(row(k, f"{v['H_next_paired']:.3f}",
                           f"{v['alpha_viol']:.3f} / {v['alpha_twin']:.3f}"))
        out.append("")
        out.append("Fitted maps (alpha per quantile bin of the gate, fit by the model's own "
                   "realised NLL on held-out windows):\n")
        for k, v in r["fit"].items():
            if k.endswith("_map"):
                out.append(f"- `{k}`: {v['alphas']} (monotone increasing {v['increasing']}, "
                           f"decreasing {v['decreasing']})")
        out.append(f"- `oracle_offset`: {r['fit']['oracle_offset_alpha']} "
                   f"(index = offset since the last corrupted token; "
                   f"{len(r['fit']['oracle_offset_alpha']) - 1} = none within H)")
        out.append(f"- constant: {r['fit']['const_alpha']:.4f}; "
                   f"oracle corruption probe AUC {r['fit']['oracle_probe_auc']:.3f}\n")


def q4(root, out, tag="c1"):
    out.append("## Q4. The plasticity loop\n")
    dirs = sorted(glob.glob(os.path.join(root, f"coeruleus/cont_{tag}/*")))
    if not dirs:
        return
    rows = []
    for d in dirs:
        arm = os.path.basename(d)
        for step in (12000, 16000, 24000):
            cal = load(root, f"coeruleus/cont_{tag}/{arm}/step{step:06d}_calibration.json")
            dual = load(root, f"coeruleus/cont_{tag}/{arm}/step{step:06d}_dual_noisy0.01.json")
            pz = load(root, f"coeruleus/cont_{tag}/{arm}/step{step:06d}_prize_d0.01.json")
            if cal is None and dual is None:
                continue
            rows.append({"arm": arm, "step": step, "cal": cal, "dual": dual, "prize": pz})
    steps = sorted({r["step"] for r in rows})
    for step in steps:
        out.append(f"### step {step} (continued from eps-trained step 8000)\n")
        out.append(row("arm", "kappa* (eps, corrupted)", "KL(p_kappa*^eps||q)",
                       "CE corrupted", "CE clean", "KL(p_L||q) clean", "best_k clean",
                       "event excess vs ceiling", "quiet excess"))
        out.append(sep(9))
        for r in [x for x in rows if x["step"] == step]:
            d, c, p = r["dual"], r["cal"], r["prize"]
            ew = (p or {}).get("event_window", {})
            out.append(row(r["arm"],
                           f"{d['eps']['kappa_star']:.2f}" if d else "-",
                           f"{d['eps']['KL_at_kappa']:.4f}" if d else "-",
                           f"{d['eps']['CE']:.4f}" if d else "-",
                           f"{c['overall']['CE']:.4f}" if c else "-",
                           f"{c['overall']['KL_p_q']:.4f}" if c else "-",
                           f"{int(np.argmin(c['overall']['KL_k_q']))}" if c else "-",
                           f"{ew.get('excess_ceil', float('nan')):.4f}" if p else "-",
                           f"{p['quiet']['excess_ceil']:.4f}" if p else "-"))
        out.append("")
        cs = [x for x in rows if x["step"] == step and x["cal"]]
        if cs:
            L = len(cs[0]["cal"]["overall"]["KL_k_q"])
            out.append("Distance to each clean observer, `KL(p_k||q)` on clean windows:\n")
            out.append(row("arm", *[f"k={k}" for k in range(L)]))
            out.append(sep(L + 1))
            for r in cs:
                out.append(row(r["arm"], *[f"{x:.4f}" for x in r["cal"]["overall"]["KL_k_q"]]))
            out.append("")
            out.append("Per-leaf-level CE on clean windows:\n")
            lv = sorted(cs[0]["cal"]["by_level"], key=int)
            out.append(row("arm", *[f"lev {l}" for l in lv]))
            out.append(sep(len(lv) + 1))
            for r in cs:
                out.append(row(r["arm"], *[f"{r['cal']['by_level'][l]['CE']:.4f}" for l in lv]))
            out.append("")
    logs = {os.path.basename(d): load(root, f"coeruleus/cont_{tag}/{os.path.basename(d)}/train_log.json")
            for d in dirs}
    out.append("Gate diagnostics during training (correlation of the gate with the realised "
               "horizon excess, and the unweighted train loss):\n")
    out.append(row("arm", "gate-target corr (first / last)", "train loss (first / last)"))
    out.append(sep(3))
    for a, lg in logs.items():
        if not lg or not lg["log"]:
            continue
        f_, l_ = lg["log"][0], lg["log"][-1]
        out.append(row(a, f"{f_['gate_target_corr']:+.4f} / {l_['gate_target_corr']:+.4f}",
                       f"{f_['loss']:.4f} / {l_['loss']:.4f}"))
    out.append("")


def figures(root, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for rel, name in VENUES:
        r = load(root, f"{rel}_prize_d0.01.json")
        if r is None:
            continue
        taus = [d["tau"] for d in r["by_tau"]]
        axes[0].plot(taus, [d["excess_ceil"] for d in r["by_tau"]], "o-", label=name)
        axes[1].plot(taus, [d["knob_alpha_pooled"] for d in r["by_tau"]], "o-", label=name)
        axes[2].plot(taus, [(d["CE_model"] - d["knob_CE_alpha_pooled"])
                            / max(d["excess_ceil"], 1e-9) for d in r["by_tau"]], "o-", label=name)
    axes[0].set_yscale("log"); axes[0].set_ylabel("excess over the ceiling (nats)")
    axes[1].axhline(1.0, color="k", lw=0.5); axes[1].set_ylabel("best temperature at this offset")
    axes[2].set_ylabel("fraction of the excess a temperature reaches")
    for a in axes:
        a.set_xlabel("offset tau since the corrupted token"); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle("Q1: the prize after a corrupted token, and the reach of a gain knob")
    fig.tight_layout(); fig.savefig(f"{figdir}/prize.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labs, keys = [], ["state_excess_ridge", "state_excess_mlp", "state_rawnll_mlp",
                      "rand_excess_ridge", "embed_excess_ridge", "Hsum_raw"]
    width = 0.13
    for i, (rel, name, tag) in enumerate(
            (("traj_eps01_s42/step064000", "eps 64k", ""),
             ("traj_eps01_s42/step008000", "eps 8k", ""),
             ("traj_a1_s42/step064000", "clean 64k on noise", "_onnoise"))):
        r = load(root, f"{rel}_coeruleus_readout{tag}.json")
        if r is None:
            continue
        labs.append(name)
        for j, k in enumerate(keys):
            h = r["heads"].get(k, {})
            axes[0].bar(i + (j - len(keys) / 2) * width, h.get("auc_event_vs_quiet", np.nan),
                        width, color=plt.cm.viridis(j / len(keys)),
                        label=k if i == 0 else None)
            pr = r.get("pairs", {}).get("readouts", {}).get(k, {})
            axes[1].bar(i + (j - len(keys) / 2) * width, pr.get("paired", np.nan), width,
                        color=plt.cm.viridis(j / len(keys)))
        pr = r.get("pairs", {}).get("readouts", {}).get("oracle_mlp", {})
        axes[1].plot([i - 0.45, i + 0.45], [pr.get("paired", np.nan)] * 2, "r--", lw=1.5,
                     label="oracle-label probe" if i == 0 else None)
    for a, t in zip(axes, ["AUC(corrupted position vs quiet)", "same-prefix pairs, paired win"]):
        a.axhline(0.5, color="k", lw=0.5); a.set_xticks(range(len(labs)))
        a.set_xticklabels(labs); a.set_title(t); a.legend(fontsize=6); a.grid(alpha=.3)
    fig.suptitle("Q2: what the self-supervised excess readout fires on")
    fig.tight_layout(); fig.savefig(f"{figdir}/readout.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (rel, name, tag) in zip(axes, (("traj_eps01_s42/step064000", "eps-trained 64k", ""),
                                           ("traj_a1_s42/step064000",
                                            "clean-trained 64k on noise", "_onnoise"))):
        r = load(root, f"{rel}_coeruleus_gain{tag}.json")
        if r is None:
            continue
        ks = [k for k in GAIN_ORDER if k in r["arms"]]
        ax.bar(range(len(ks)), [r["arms"][k]["frac_prize_recovered"] for k in ks],
               color=["#888"] + ["#4c72b0"] * (len(ks) - 1))
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=40, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{name}\nprize {r['prize_nats_per_pred']:.3f} nats/prediction", fontsize=9)
        ax.set_ylabel("fraction of the prize recovered"); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Q3: what the closed gain loop recovers")
    fig.tight_layout(); fig.savefig(f"{figdir}/gain.png", dpi=140); plt.close(fig)

    dirs = sorted(glob.glob(os.path.join(root, "coeruleus/cont_c1/*")))
    if dirs:
        arms = [os.path.basename(d) for d in dirs]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
        series = {"CE clean": [], "KL(p_L||q) clean": [], "post-event excess vs ceiling": []}
        for a in arms:
            c = load(root, f"coeruleus/cont_c1/{a}/step024000_calibration.json")
            p = load(root, f"coeruleus/cont_c1/{a}/step024000_prize_d0.01.json")
            series["CE clean"].append(c["overall"]["CE"] if c else np.nan)
            series["KL(p_L||q) clean"].append(c["overall"]["KL_p_q"] if c else np.nan)
            series["post-event excess vs ceiling"].append(
                p["event_window"]["excess_ceil"] if p else np.nan)
        for ax, (k, vv) in zip(axes, series.items()):
            col = ["#c44e52" if a == "uniform" else ("#55a868" if a.startswith("oracle")
                                                     else "#4c72b0") for a in arms]
            ax.bar(range(len(arms)), vv, color=col)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
            ax.set_title(k, fontsize=10); ax.grid(alpha=.3, axis="y")
            lo, hi = np.nanmin(vv), np.nanmax(vv)
            ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.15 * (hi - lo))
        fig.suptitle("Q4: continued training from eps-trained step 8000, 16k gated steps "
                     "(identical weight multisets except `uniform`)")
        fig.tight_layout(); fig.savefig(f"{figdir}/plasticity.png", dpi=140); plt.close(fig)
    print(f"figures -> {figdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--figs", default="")
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = ["# coeruleus -- reduced tables\n"]
    q1(a.root, out)
    q2(a.root, out)
    q3(a.root, out)
    q4(a.root, out, a.tag)
    txt = "\n".join(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
        print(f"tables -> {a.out}")
    else:
        print(txt)
    if a.figs:
        figures(a.root, a.figs)


if __name__ == "__main__":
    main()
