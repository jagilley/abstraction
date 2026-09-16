"""Local aggregation for the `altitude` round. Reads JSON only -- no Modal, no GPU.

    python -m rhm.logit_reading.altitude.analyze <dir> [--figs <dir>]

<dir> holds the files pulled off the volume:
  altitude_identity.json  altitude_frontier.json  altitude_units.json
  altitude_phase_identifiability.json  altitude_persistence_a1.json
  tonic/<step>.json  entdir/<step>.json  dual_<ws>/<step>.json  (per-checkpoint)
"""

import glob
import json
import os
import sys


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _load_steps(d, sub):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, sub, "*.json"))):
        j = json.load(open(p))
        step = j.get("step", (j.get("config") or {}).get("step"))
        out[int(step)] = j
    return out


def table(rows, head):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(head)]
    print("| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)) + " |")
    print("|" + "|".join("-" * (w[i] + 2) for i in range(len(head))) + "|")
    for r in rows:
        print("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    print()


def part1(d):
    idj = _load(d, "altitude_identity.json")
    if not idj:
        return None
    print("\n## Q1a. The identity, and the temperature that zeroes it\n")
    rows = []
    for r in idj:
        i = r["identity"]
        rows.append([r["step"], f"{i['mean_gap']:+.5f}", f"{i['mean_tempgrad']:+.5f}",
                     f"{i['max_abs_diff']:.1e}", f"{r['alpha_star_overall']:.4f}",
                     f"{r['altitude']['fit_quality']['KL_pL_q']:.4f}"])
    table(rows, ["step", "CE-H(q)", "<q-p,z>", "max|diff|", "alpha*", "KL(p_L||q)"])

    print("## Q1b. What rescaling by alpha does (last checkpoint)\n")
    last = idj[-1]
    rows = []
    for a in last["alpha_sweep"]:
        rows.append([f"{a['alpha']:.4f}", f"{a['gap']:+.4f}", f"{a['KL_p_q']:.4f}",
                     f"{a['ece_realised']:.4f}", a["best_k_meanargmin"],
                     " ".join(f"{x:.3f}" for x in a["KL_k_q"]), f"{a['R2_CE_on_Hq']:.3f}"])
    table(rows, ["alpha", "gap", "KL(p_L||q)", "ECE", "best_k", "KL(p_k||q), k=0..6", "R2(CE~Hq)"])

    print("## Q1c. Per-class gap, its own alpha*, and R2 (last checkpoint)\n")
    rows = []
    for nm, c in last["classes"].items():
        rows.append([nm, c["n"], f"{c['mean_gap']:+.4f}", f"{c['sd_gap']:.4f}",
                     f"{c['alpha_star']:.4f}", f"{c['R2_CE_on_Hq']:.3f}",
                     f"{c['var_gap_over_var_CE']:.4f}"])
    table(rows, ["class", "n", "mean gap", "sd gap", "alpha*", "R2(CE~Hq)", "var(gap)/var(CE)"])
    rd = last["R2_decomposition"]
    print(f"R2 decomposition (last ckpt): var(H_q) {rd['var_Hq']:.4f}  var(gap) {rd['var_gap']:.6f}"
          f"  var(CE) {rd['var_CE']:.4f}  cov(H_q,gap) {rd['cov_Hq_gap']:+.5f}\n"
          f"  pooled R2 {rd['R2_CE_on_Hq']:.4f}   within-window-position R2 "
          f"{rd['R2_CE_on_Hq_within_winpos']:.4f}\n")

    print("## Q1d. A continuous altitude\n")
    rows = []
    for r in idj:
        a = r["altitude"]
        rows.append([r["step"], a["discrete"]["best_k"], f"{a['discrete']['KL_best']:.4f}",
                     f"{a['convex']['kappa_star']:.2f}", f"{a['convex']['KL_at_star']:.4f}",
                     f"{a['loglinear']['kappa_star']:.2f}", f"{a['loglinear']['KL_at_star']:.4f}",
                     f"{a['smoothed']['best_k']}+{a['smoothed']['best_eps']}",
                     f"{a['smoothed']['KL_at_best']:.4f}",
                     f"{a['tempered']['best_alpha']:.2f}", f"{a['tempered']['KL_at_best']:.4f}",
                     f"{a['fit_quality']['KL_pL_q']:.4f}",
                     f"{a['convex']['KL_at_star'] / a['fit_quality']['KL_pL_q']:.3f}",
                     f"{a['fit_quality']['KL_pL_pkappa']:.4f}",
                     f"{a['convex']['kappa_pp_mean']:.2f}"])
    table(rows, ["step", "best_k", "KL@k", "kappa*", "KL@kappa", "ll-kappa", "KL@ll",
                 "(k,eps)", "KL@(k,eps)", "temp a", "KL@temp", "KL(p_L||q)",
                 "KL@kappa / KL(p_L||q)", "KL(p_L||p_k*)", "mean per-pos kappa"])
    print("Per-position kappa histogram (bins of 0.5 from 0 to 6), last checkpoint:")
    print("  " + " ".join(str(x) for x in idj[-1]["altitude"]["convex"]["kappa_pp_hist"]) + "\n")
    return idj


def part2(d, idj):
    fr = _load(d, "altitude_frontier.json")
    ph = _load(d, "altitude_phase_identifiability.json")
    if not fr:
        return
    kap = {r["step"]: r["altitude"]["convex"]["kappa_star"] for r in (idj or [])}
    print("## Q2a. Locating the boundary from the logits alone (entropy of q)\n")
    rows = []
    for r in fr:
        p = r["periodicity"]["H_q"]
        rows.append([r["step"], f"{kap.get(r['step'], float('nan')):.2f}"]
                    + [f"{p[str(k)]['acc_template']:.2f}" for k in range(1, 7)]
                    + [f"{p['dominant_k_mean']:.2f}"])
    table(rows, ["step", "kappa*"] + [f"k={k}" for k in range(1, 7)] + ["dominant k (<=4)"])
    if ph:
        print("Ceilings -- observer k's OWN posterior over the phase mod s^k "
              "(window positions 32-64):")
        rows = [[k, ph["by_k"][str(k)]["S"], f"{1.0 / ph['by_k'][str(k)]['S']:.3f}",
                 f"{ph['by_k'][str(k)]['acc_by_pos']['32-65']:.3f}",
                 f"{ph['by_k'][str(k)]['logpost_true_by_pos']['32-65']:+.3f}"]
                for k in range(1, 7)]
        table(rows, ["k", "s^k", "chance", "observer acc", "log post of true phase"])
    print("Argmax-of-offset-profile detector (same rows, for comparison):\n")
    rows = []
    for r in fr:
        p = r["periodicity"]["H_q"]
        rows.append([r["step"]] + [f"{p[str(k)]['acc_argmax']:.2f}" for k in range(1, 7)])
    table(rows, ["step"] + [f"k={k}" for k in range(1, 7)])

    print("## Q2b. Where KL(p_L||q) lives, by leaf level (last checkpoint)\n")
    last = fr[-1]
    rows = []
    for l, b in sorted(last["by_level"].items(), key=lambda kv: int(kv[0])):
        rows.append([l, b["n"], f"{b['frac']:.3f}", f"{b['H_q']:.3f}", f"{b['H_pL']:.3f}",
                     f"{b['KL_pL_q']:.4f}", f"{b['KL_pL_q_share']:.3f}",
                     " ".join(f"{x:.3f}" for x in b["news_k_to_k1"])])
    table(rows, ["leaf level", "n", "frac", "H(q)", "H(p_L)", "KL(p_L||q)", "share of KL",
                 "KL(p_{k+1}||p_k), k=0..5"])


def part2c(d):
    """Q2b over the whole trajectory: the model's residual vs the FITTED observer's."""
    rj = _load(d, "altitude_residual.json")
    if not rj:
        return
    import math
    print("## Q2b-traj. Per leaf level, KL(p_L||p_kappa*) / KL(p_L||q) -- how much of the "
          "model's\nresidual the fitted observer accounts for (1.0 = all of it)\n")
    rows = []
    for r in rj:
        bl = r["by_level"]
        sm = [bl[str(l)]["share_model"] for l in range(7)]
        so = [bl[str(l)]["share_obs"] for l in range(7)]
        l1 = 0.5 * sum(abs(a - b) for a, b in zip(sm, so))
        rows.append([r["step"], f"{r['kappa']:.2f}",
                     f"{r['KL_model_total']:.4f}", f"{r['KL_obs_total']:.4f}",
                     f"{r['KL_obs_total'] / r['KL_model_total']:.3f}"]
                    + [f"{bl[str(l)]['obs_over_model']:.3f}" for l in range(7)]
                    + [f"{l1:.3f}"])
    table(rows, ["step", "kappa*", "KL(p_L||q)", "KL(p_L||p_k*)", "total ratio"]
          + [f"lev {l}" for l in range(7)] + ["L1 dist of shares"])
    print("Absolute per-level KL, model / fitted observer (nats)\n")
    rows = []
    for r in rj:
        bl = r["by_level"]
        rows.append([r["step"], f"{r['kappa']:.2f}"]
                    + [f"{bl[str(l)]['KL_model']:.4f}/{bl[str(l)]['KL_obs']:.4f}"
                       for l in range(7)])
    table(rows, ["step", "kappa*"] + [f"lev {l}" for l in range(7)])


def part3(d, idj):
    uj = _load(d, "altitude_units.json")
    if not uj:
        return
    kap = {r["step"]: r["altitude"]["convex"]["kappa_star"] for r in (idj or [])}
    rows = uj["rows"]
    js = sorted(int(j) for j in rows[0]["by_j"])
    print("## Q3a. Delta for TRUE units (nats): the candidate's advantage over the model\n")
    tb = []
    for r in rows:
        tb.append([r["step"], f"{kap.get(r['step'], float('nan')):.2f}"]
                  + [f"{r['by_j'][str(j)]['delta_model'][0]:+.2f}" for j in js])
    table(tb, ["step", "kappa*"] + [f"j={j}" for j in js])
    print("## Q3b. AUC(true vs chance) with the model as the null\n")
    tb = []
    for r in rows:
        tb.append([r["step"], f"{kap.get(r['step'], float('nan')):.2f}"]
                  + [f"{r['by_j'][str(j)]['auc_delta_model']:.3f}" for j in js])
    table(tb, ["step", "kappa*"] + [f"j={j}" for j in js])
    print("## Q3b2. Per-TOKEN Delta for true units (Delta / 2^j, nats/token)\n")
    tb = []
    for r in rows:
        tb.append([r["step"], f"{kap.get(r['step'], float('nan')):.2f}"]
                  + [f"{r['by_j'][str(j)]['delta_model'][0] / 2 ** j:+.3f}" for j in js])
    table(tb, ["step", "kappa*"] + [f"j={j}" for j in js])
    print("## Q3b3. Where each j's separation peaks over training\n")
    tb = []
    for j in js:
        best = max(rows, key=lambda r: r["by_j"][str(j)]["auc_delta_model"])
        k_at = kap.get(best["step"], float("nan"))
        zero = next((r["step"] for r in rows
                     if r["by_j"][str(j)]["delta_model"][0] <= 0), None)
        tb.append([j, f"{best['by_j'][str(j)]['auc_delta_model']:.3f}", best["step"],
                   f"{k_at:.2f}", f"{k_at - j:+.2f}",
                   f"{max(r['by_j'][str(j)]['auc_delta_obs'][str(j)] for r in rows[:1]):.3f}",
                   zero if zero is not None else "-"])
    table(tb, ["j", "peak AUC(model)", "at step", "kappa* there", "kappa*-j",
               "obs k=j ceiling", "step where Delta_true<=0"])
    print("## Q3c. The two baselines (constant across checkpoints), and the null's own "
          "coarseness\n")
    b0 = rows[0]["by_j"]
    tb = []
    for j in js:
        b = b0[str(j)]
        tb.append([j, b["n_true"], b["n_chance"], f"{uj['meta']['mi'][str(j)]['true']:.3f}",
                   f"{uj['meta']['mi'][str(j)]['chance']:.3f}", f"{b['auc_pmi']:.3f}",
                   f"{b['auc_cand_gain']:.3f}"]
                  + [f"{b['auc_delta_obs'][str(k)]:.3f}" for k in range(7)])
    table(tb, ["j", "n true", "n chance", "MI true", "MI chance", "AUC PMI", "AUC cand gain"]
          + [f"obs k={k}" for k in range(7)])
    print("## Q3d. The model's own cost on B alone (the second-child confound)\n")
    tb = []
    for r in rows:
        tb.append([r["step"]] + [f"{r['by_j'][str(j)]['auc_cost_model_alone']:.3f}" for j in js])
    table(tb, ["step"] + [f"j={j}" for j in js])


def part4(d):
    tn = _load_steps(d, "tonic_a1") or _load_steps(d, "tonic")
    if tn:
        print("## Q4a. Tonic: running surprisal over the 4 tokens before the event, "
              "2b's own pairs\n")
        rows = []
        for st in sorted(tn):
            a = tn[st]["test_all"]
            rows.append([st, tn[st]["n_pairs_test"], f"{a['guard_surprisal']:.3f}",
                         f"{a['run_model_4']:.3f}", f"{a['run_model_15']:.3f}"]
                        + [f"{a[f'run_obs{k}_4']:.3f}" for k in range(7)]
                        + [f"{a['run_excess6_4']:.3f}"])
        table(rows, ["step", "pairs", "guard", "model(4)", "model(15)"]
              + [f"obs{k}(4)" for k in range(7)] + ["model-obs6"])
    ed = _load_steps(d, "entdir_a1") or _load_steps(d, "entdir")
    if ed:
        print("## Q4b. Entropy direction on the same-prefix pairs "
              "(AUC / paired win; >0.5 = less confident after the violation)\n")
        rows = []
        for st in sorted(ed):
            e = ed[st]
            if "model" not in e:
                continue
            r = [st, e["n_scored"], f"{e['model']['H_next_auc']:.3f}/"
                 f"{e['model']['H_next_paired']:.3f}"]
            for k in sorted(int(x) for x in e["observers"]):
                o = e["observers"][str(k)]
                r.append(f"{o['H_next_auc']:.3f}/{o['H_next_paired']:.3f}")
            rows.append(r)
        ks = sorted(int(x) for x in ed[max(ed)]["observers"])
        table(rows, ["step", "pairs", "model"] + [f"obs k={k}+eps" for k in ks])
        last = ed[max(ed)]
        if "by_kstar" in last["model"]:
            print("Paired win rate split by the violation's k* (last checkpoint; "
                  "n>=30 cells only)\n")
            kss = sorted(int(x) for x in last["model"]["by_kstar"])
            rows = [["model"] + [f"{last['model']['by_kstar'][str(kk)]:.3f}" for kk in kss]]
            for k in ks:
                o = last["observers"][str(k)]["by_kstar"]
                rows.append([f"obs {k}+eps"] + [f"{o[str(kk)]:.3f}" if str(kk) in o else "-"
                                                for kk in kss])
            table(rows, ["reader"] + [f"k*={kk}" for kk in kss])
            print("k* histogram of the scored pairs: "
                  + str(last.get("kstar_hist")) + "\n")
    pj = _load(d, "altitude_persistence_a1.json")
    if pj:
        print("## Q4b-ii. Output-level persistence: dH after a k*=5 violation, by tau\n")
        rows = []
        mr = pj["model"]
        last = str(max(int(s) for s in mr))
        for g in ("viol_k3", "viol_k4", "viol_k5", "rare"):
            if g not in mr[last]:
                continue
            mrow = mr[last][g]
            rows.append([f"model (64k) {g}"] + [f"{r['dH_model']:+.3f}" for r in mrow[:6]])
            rows.append([f"  obs L+eps {g}"] + [f"{r['dH_bayes_eps_L']:+.3f}" for r in mrow[:6]])
            for k in sorted(int(x) for x in pj["by_observer"]):
                rr = pj["by_observer"][str(k)].get(g)
                if rr:
                    rows.append([f"  obs {k}+eps {g}"] + [f"{r['dH_bayes_k']:+.3f}" for r in rr[:6]])
        table(rows, ["series"] + [f"tau={t}" for t in range(6)])


def part4b(d):
    """Q4b sharpened: which observer is the post-token forecast closest to?"""
    import glob
    paths = sorted(glob.glob(os.path.join(d, "whither", "*.json")))
    if not paths:
        return
    for pth in paths:
        j = json.load(open(pth))
        nm = "eps-trained" if j.get("eps_train") else "clean-trained"
        print(f"## Q4b-whither. KL(q_next || p_k,next) after the token -- {nm}, "
              f"step {j['step']}\n")
        print(f"matched {j['n_matched']} / {j['n_candidates']} candidates, test {j['n_test']}, "
              f"scored {j['n_scored']}; k* histogram {j['kstar_hist']}; "
              f"H(q_next) viol {j['H_next']['viol']:.3f} vs twin {j['H_next']['twin']:.3f}\n")
        for fam in ("eps", "clean"):
            F = j["families"][fam]
            ks = sorted(int(k) for k in F if k.isdigit())
            rows = [[k, f"{F[str(k)]['KL_viol']:.4f}", f"{F[str(k)]['KL_twin']:.4f}",
                     f"{F[str(k)]['defined_viol']:.2f}", f"{F[str(k)]['defined_twin']:.2f}"]
                    for k in ks]
            print(f"{fam} family (pooled):\n")
            table(rows, ["observer k", "after violation", "after twin",
                         "defined (viol)", "defined (twin)"])
        F = j["families"]["eps"]
        ks = sorted(int(k) for k in F if k.isdigit())
        kstars = sorted({int(x) for k in ks for x in F[str(ks[0])]["by_kstar"]})
        print("eps family, split by the violation's k*:\n")
        rows = []
        for side, lbl in (("viol", "after violation"), ("twin", "after twin")):
            for k in ks:
                rows.append([f"{lbl}, obs k={k}"]
                            + [f"{F[str(k)]['by_kstar'][str(s_)][f'KL_{side}']:.4f}"
                               for s_ in kstars])
            rows.append([f"{lbl}: argmin k"]
                        + [str(min(ks, key=lambda k: F[str(k)]["by_kstar"][str(s_)][f"KL_{side}"]))
                           for s_ in kstars])
        table(rows, ["series"] + [f"k*={s_} (n={F[str(ks[0])]['by_kstar'][str(s_)]['n']})"
                                  for s_ in kstars])


def part5(d):
    for ws in ("clean", "noisy0.01"):
        for traj in ("a1", "eps01"):
            rows_d = _load_steps(d, f"dual_{traj}_{ws}")
            if not rows_d:
                continue
            print(f"## Q5. Dual-ladder fit -- trajectory {traj}, window set {ws}\n")
            rows = []
            for st in sorted(rows_d):
                r = rows_d[st]
                rows.append([st, f"{r['clean']['best_k']}", f"{r['clean']['kappa_star']:.2f}",
                             f"{r['clean']['KL_at_kappa']:.4f}", f"{r['clean']['KL_pL_q']:.4f}",
                             f"{r['eps']['best_k']}", f"{r['eps']['kappa_star']:.2f}",
                             f"{r['eps']['KL_at_kappa']:.4f}", f"{r['eps']['KL_pL_q']:.4f}",
                             r["winner"], f"{r['clean']['gap']:+.4f}",
                             f"{r['ece_realised']:.4f}"])
            table(rows, ["step", "k clean", "kappa clean", "KL clean", "KL(p_L||q)",
                         "k eps", "kappa eps", "KL eps", "KL(p_L^eps||q)", "winner",
                         "gap vs clean", "ECE"])
            print("The gap is family-relative (defined_frac = where that observer is "
                  "defined at all):\n")
            rows = [[st, f"{rows_d[st]['clean']['gap']:+.4f}", f"{rows_d[st]['eps']['gap']:+.4f}",
                     f"{rows_d[st]['clean'].get('defined_frac', 1.0):.3f}",
                     f"{rows_d[st]['eps'].get('defined_frac', 1.0):.3f}",
                     f"{rows_d[st]['H_q']:.4f}"] for st in sorted(rows_d)]
            table(rows, ["step", "gap vs p_L", "gap vs p_L^eps", "clean defined",
                         "eps defined", "H(q)"])


def part2b(d):
    """2c / 2d / 2e side by side for the clean and the noise-trained trajectory."""
    for tag, sub_ph, sub_gr in (("clean a1", "phasic_a1", "gradient_a1"),
                                ("noise-trained eps01", "phasic_eps01", "gradient_eps01")):
        ph, gr = _load_steps(d, sub_ph), _load_steps(d, sub_gr)
        if not ph and not gr:
            continue
        print(f"## Q5b. Same-prefix readouts, trajectory {tag}\n")
        rows = []
        for st in sorted(set(ph) | set(gr)):
            a = (ph.get(st) or {}).get("test_all")
            g = (gr.get(st) or {}).get("test_all")
            rows.append([st,
                         a["n_pairs"] if a else "-",
                         f"{a['post_logit_mlp']['auc']:.3f}" if a else "-",
                         f"{a['state_best_mlp']['auc']:.3f}" if a else "-",
                         f"{a['H_next']['auc']:.3f}/{a['H_next']['paired']:.3f}" if a else "-",
                         g["n_pairs"] if g else "-",
                         f"{g['post_logit_mlp']['auc']:.3f}" if g else "-",
                         f"{g['H_next']['auc']:.3f}/{g['H_next']['paired']:.3f}" if g else "-"])
        table(rows, ["step", "2c pairs", "2c forecast", "2c state", "2e H_next (viol)",
                     "2d pairs", "2d forecast", "2d H_next (legal-rare)"])


def figures(d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    idj = _load(d, "altitude_identity.json")
    fr = _load(d, "altitude_frontier.json")
    uj = _load(d, "altitude_units.json")
    ph = _load(d, "altitude_phase_identifiability.json")
    tn = _load_steps(d, "tonic_a1") or _load_steps(d, "tonic")
    ed = _load_steps(d, "entdir_a1") or _load_steps(d, "entdir")
    xs = lambda ss: [max(s, 100) for s in ss]

    if idj:
        steps = [r["step"] for r in idj]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].step(xs(steps), [r["altitude"]["discrete"]["best_k"] for r in idj], where="post",
                   label="argmin_k KL(p_k||q)", color="0.6")
        ax[0].plot(xs(steps), [r["altitude"]["convex"]["kappa_star"] for r in idj], "o-",
                   label=r"$\kappa^*$ (convex)", color="C0")
        ax[0].plot(xs(steps), [r["altitude"]["convex"]["kappa_pp_mean"] for r in idj], "s--",
                   label=r"mean per-position $\kappa$", color="C1", ms=4)
        ax[0].set_xscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("altitude")
        ax[0].legend(fontsize=8); ax[0].set_title("A continuous altitude")
        ax[1].plot(xs(steps), [r["altitude"]["discrete"]["KL_best"] for r in idj], "o-",
                   label="best discrete rung", color="0.6")
        ax[1].plot(xs(steps), [r["altitude"]["convex"]["KL_at_star"] for r in idj], "o-",
                   label=r"best $\kappa$", color="C0")
        ax[1].plot(xs(steps), [r["altitude"]["smoothed"]["KL_at_best"] for r in idj], "^-",
                   label=r"best $(k,\epsilon)$", color="C2", ms=4)
        ax[1].plot(xs(steps), [r["altitude"]["fit_quality"]["KL_pL_q"] for r in idj], "k--",
                   label="KL(p_L||q)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("step")
        ax[1].set_ylabel("mean KL to q (nats)"); ax[1].legend(fontsize=8)
        ax[1].set_title("Residual of the fit")
        fig.tight_layout(); fig.savefig(f"{outdir}/altitude.png", dpi=140); plt.close(fig)

    if fr:
        steps = [r["step"] for r in fr]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in range(1, 7):
            ax.plot(xs(steps), [r["periodicity"]["H_q"][str(k)]["acc_template"] for r in fr],
                    "o-", ms=3, color=plt.cm.viridis(k / 6), label=f"k={k}")
            if ph:
                ax.axhline(ph["by_k"][str(k)]["acc_by_pos"]["32-65"], color=plt.cm.viridis(k / 6),
                           ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("step")
        ax.set_ylabel("P(boundary offset mod $s^k$ correct)")
        ax.set_title("Boundary located from the model's entropy\n(dotted = observer k's own ceiling)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"{outdir}/frontier.png", dpi=140); plt.close(fig)

    if uj:
        rows = uj["rows"]
        js = sorted(int(j) for j in rows[0]["by_j"])
        steps = [r["step"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for j in js:
            ax[0].plot(xs(steps), [r["by_j"][str(j)]["delta_model"][0] for r in rows], "o-",
                       ms=3, color=plt.cm.plasma(j / 5), label=f"j={j}")
            ax[1].plot(xs(steps), [r["by_j"][str(j)]["auc_delta_model"] for r in rows], "o-",
                       ms=3, color=plt.cm.plasma(j / 5), label=f"j={j}")
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=0.1)
        ax[0].set_xlabel("step"); ax[0].set_ylabel(r"$\Delta$ for true units (nats)")
        ax[0].set_title("Absorption"); ax[0].legend(fontsize=8)
        ax[1].axhline(0.5, color="k", lw=0.8)
        ax[1].set_xscale("log"); ax[1].set_xlabel("step")
        ax[1].set_ylabel("AUC(true vs chance), model as null")
        ax[1].set_title("Separation"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{outdir}/units.png", dpi=140); plt.close(fig)

    if tn or ed:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if tn:
            ss = sorted(tn)
            ax[0].plot(xs(ss), [tn[s]["test_all"]["run_model_4"] for s in ss], "o-",
                       label="model's own", color="C3")
            for k in (3, 4, 5, 6):
                ax[0].plot(xs(ss), [tn[s]["test_all"][f"run_obs{k}_4"] for s in ss], "-",
                           lw=1.2, color=plt.cm.viridis(k / 6), label=f"observer {k}")
            ax[0].axhline(0.5, color="k", lw=0.8)
            ax[0].set_xscale("log"); ax[0].set_xlabel("step")
            ax[0].set_ylabel("AUC, running surprisal (last 4)")
            ax[0].set_title("Q4a: tonic signal, model vs observers"); ax[0].legend(fontsize=7)
        if ed:
            ss = [s for s in sorted(ed) if "model" in ed[s]]
            ax[1].plot(xs(ss), [ed[s]["model"]["H_next_paired"] for s in ss], "o-",
                       color="C3", label="model")
            for k in sorted(int(x) for x in ed[ss[-1]]["observers"]):
                ax[1].plot(xs(ss), [ed[s]["observers"][str(k)]["H_next_paired"] for s in ss],
                           "-", lw=1.2, color=plt.cm.viridis(k / 6), label=f"obs {k}+eps")
            ax[1].axhline(0.5, color="k", lw=0.8)
            ax[1].set_xscale("log"); ax[1].set_xlabel("step")
            ax[1].set_ylabel("paired win rate, H(next forecast)")
            ax[1].set_title("Q4b: entropy direction after a violation"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{outdir}/controls.png", dpi=140); plt.close(fig)
    rj = _load(d, "altitude_residual.json")
    if rj:
        steps = [r["step"] for r in rj]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for l in range(7):
            c = plt.cm.viridis(l / 6)
            ax[0].plot(xs(steps), [r["by_level"][str(l)]["share_model"] for r in rj], "o-",
                       ms=3, color=c, label=f"leaf level {l}")
            ax[1].plot(xs(steps), [r["by_level"][str(l)]["share_obs"] for r in rj], "o-",
                       ms=3, color=c)
        ax[0].set_title(r"model: share of KL$(p_L\|q)$ by leaf level")
        ax[1].set_title(r"fitted observer: share of KL$(p_L\|p_{\kappa^*})$")
        for a in ax:
            a.set_xscale("log"); a.set_xlabel("step"); a.set_ylim(0, 0.55)
        ax[0].set_ylabel("share of total residual"); ax[0].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"{outdir}/residual.png", dpi=140); plt.close(fig)

    wpaths = sorted(glob.glob(os.path.join(d, "whither", "*.json")))
    if wpaths:
        fig, ax = plt.subplots(1, len(wpaths), figsize=(5.5 * len(wpaths), 4), squeeze=False)
        for i, pth in enumerate(wpaths):
            j = json.load(open(pth))
            F = j["families"]["eps"]
            ks = sorted(int(k) for k in F if k.isdigit())
            kstars = sorted({int(x) for x in F[str(ks[0])]["by_kstar"]})
            a = ax[0][i]
            for s_ in kstars:
                n = F[str(ks[0])]["by_kstar"][str(s_)]["n"]
                if n < 40:
                    continue
                c = plt.cm.plasma((s_ - 1) / 5)
                a.plot(ks, [F[str(k)]["by_kstar"][str(s_)]["KL_viol"] for k in ks], "o-",
                       color=c, ms=4, label=f"k*={s_} (n={n}), violation")
                a.plot(ks, [F[str(k)]["by_kstar"][str(s_)]["KL_twin"] for k in ks], "s--",
                       color=c, ms=3, alpha=0.6, label=f"k*={s_}, twin")
                a.axvline(s_ - 1, color=c, lw=0.8, ls=":", alpha=0.5)
            a.set_yscale("log"); a.set_xlabel("observer k (eps family)")
            a.set_ylabel(r"KL$(q_{next}\|p_{k,next})$ (nats)")
            a.set_title(("eps-trained" if j.get("eps_train") else "clean-trained")
                        + f", step {j['step']}")
            a.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{outdir}/whither.png", dpi=140); plt.close(fig)

    # Q5: which observer family, and does the entropy sign move
    duals = {t: _load_steps(d, f"dual_{t}_clean") for t in ("a1", "eps01")}
    if all(duals.values()):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, (t, lbl) in enumerate((("a1", "clean-trained"), ("eps01", "eps-trained"))):
            ss = sorted(duals[t])
            ax[0].plot(xs(ss), [duals[t][s]["clean"]["KL_at_kappa"] for s in ss], "o-",
                       color=f"C{i}", ms=4, label=f"{lbl}: clean family")
            ax[0].plot(xs(ss), [duals[t][s]["eps"]["KL_at_kappa"] for s in ss], "s--",
                       color=f"C{i}", ms=4, label=f"{lbl}: eps family")
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("step")
        ax[0].set_ylabel(r"$\min_\kappa$ KL$(p_\kappa\|q)$ (nats)")
        ax[0].set_title("Which family, on clean windows"); ax[0].legend(fontsize=7)
        for i, (t, lbl) in enumerate((("a1", "clean-trained"), ("eps01", "eps-trained"))):
            e = _load_steps(d, f"entdir_{'a1' if t == 'a1' else 'eps01'}")
            ss = [s for s in sorted(e) if "model" in e[s]]
            if not ss:
                continue
            ax[1].plot(xs(ss), [e[s]["model"]["H_next_paired"] for s in ss], "o-",
                       color=f"C{i}", ms=4, label=f"{lbl} model")
            ax[1].plot(xs(ss), [e[s]["observers"]["5"]["H_next_paired"] for s in ss], ":",
                       color=f"C{i}", lw=1.4, label=f"{lbl} pairs, obs 5+eps")
        ax[1].axhline(0.5, color="k", lw=0.8)
        ax[1].set_xscale("log"); ax[1].set_xlabel("step")
        ax[1].set_ylabel("paired win rate, H(next forecast)")
        ax[1].set_title("Entropy direction after a violation"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{outdir}/noise_trained.png", dpi=140); plt.close(fig)
    print(f"figures -> {outdir}")


def main():
    d = sys.argv[1]
    idj = part1(d)
    part2(d, idj)
    part2c(d)
    part3(d, idj)
    part4(d)
    part4b(d)
    part5(d)
    part2b(d)
    if "--figs" in sys.argv:
        figures(d, sys.argv[sys.argv.index("--figs") + 1])


if __name__ == "__main__":
    main()
