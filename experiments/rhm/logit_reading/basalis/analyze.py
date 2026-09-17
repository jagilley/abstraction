"""Local reduction of the basalis JSONs pulled off the volume. No Modal, no GPU.

  python -m rhm.logit_reading.basalis.analyze <dir> \
      --out rhm/logit_reading/basalis/results/tables.md \
      --figs rhm/logit_reading/basalis/figs

<dir> holds the JSONs in the same relative layout as the volume, e.g.
  traj_eps01_s42/step064000_basalis.json
  traj_a1_s42/step064000_basalis.json
"""

import argparse
import json
import os

import numpy as np

MODELS = [("traj_eps01_s42/step064000_basalis.json", "eps-trained 64k"),
          ("traj_a1_s42/step064000_basalis.json", "clean-trained 64k")]
ARMS = ["edit", "glitch_s", "glitch_m", "quiet"]
ARM_LABEL = {"edit": "edit (swap)", "glitch_s": "glitch, surprisal-matched",
             "glitch_m": "glitch, uniform draw", "quiet": "quiet (false alarm)"}
GLITCHES = ["glitch_s", "glitch_m"]


def load(root, rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def loaded(root):
    return [(nm, r) for rel, nm in MODELS for r in [load(root, rel)] if r is not None]


def row(*c):
    return "| " + " | ".join(str(x) for x in c) + " |"


def sep(n):
    return "|" + "|".join(["---"] * n) + "|"


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# ---------------------------------------------------------------------------

def q0(root, out):
    out.append("## The stimulus populations\n")
    out.append(row("model", "events", "of eligible", "k* hist (1..6)", "j hist (0..4)",
                   "mean t_v - e", "t_v beyond the edited region",
                   "s_event edit / glitch_s / glitch_m / quiet"))
    out.append(sep(8))
    for nm, r in loaded(root):
        s = r["stimuli"]
        se = s.get("s_event_by_arm", {})
        out.append(row(nm, s["n_events"], s["n_eligible"], s["kstar_hist"][1:], s["j_hist"],
                       f"{s['delay_mean']:.1f}", f"{s['tv_beyond_region_frac']:.2f}",
                       " / ".join(f3(se.get(a)) for a in ARMS)))
    out.append("\n`edit` and both glitch arms are the SAME original stream at the SAME position: "
               "the edit reads it with one constituent swapped and regrown legally, the glitch "
               "reads it unedited with the token at `t_v` replaced (uniform draw for `glitch_m`, "
               "the token nearest the edit's violator in model surprisal for `glitch_s`). "
               "`quiet` is a disjoint set of unedited windows flagged at a random position -- "
               "the cost of a false alarm.\n")


def q1(root, out):
    out.append("## Q1. Which imputation\n")
    out.append("`nll(w=1)` is the realised NLL over tau = 0..8 under the pure hold policy; "
               "`w*(tau=0)` the exact minimiser of the realised NLL at the event.\n")
    for nm, r in loaded(root):
        im = r["imputation"]
        out.append(f"### {nm} (prior mass at the event: top-1 "
                   f"{f3(im['top1_mass'].get('edit'))}, top-4 {f3(im['top4_mass'].get('edit'))} "
                   "on the edit arm)\n")
        out.append(row("variant", *[f"{a}: nll(w=1) / w*(0)" for a in ARMS]))
        out.append(sep(1 + len(ARMS)))
        for var, d in im["by_variant"].items():
            out.append(row(var, *[f"{f4(d[a]['nll_hold_pooled'])} / {f3(d[a]['w_star_tau0'])}"
                                  for a in ARMS if a in d]))
        out.append("")


def q2(root, out):
    out.append("## Q2. The prize per world, per offset\n")
    out.append("Realised NLL of the actual continuation. `w*` is the exact per-offset "
               "minimiser fitted on the same rows (an in-hindsight oracle, stated as such); "
               "`frac hold better` is the fraction of predictions on which the held forecast "
               "gives the realised token more mass than the native one.\n")
    for nm, r in loaded(root):
        out.append(f"### {nm}\n")
        for a in ARMS:
            if a not in r["by_world"]:
                continue
            d = r["by_world"][a]
            out.append(f"**{ARM_LABEL[a]}** (n = {d['n']})\n")
            out.append(row("tau", "nll native", "nll hold", "w*", "nll at w*",
                           "frac hold better", "H native", "H hold"))
            out.append(sep(8))
            for x in d["all"]:
                out.append(row(x["tau"], f4(x["nll_native"]), f4(x["nll_hold"]),
                               f3(x["w_star"]), f4(x["nll_at_wstar"]),
                               f3(x["frac_hold_better"]), f3(x["H_native"]), f3(x["H_hold"])))
            out.append("")
        # the size of the edit is the knob
        if "by_j" in r["by_world"].get("edit", {}):
            out.append("**The edit's size `j` (the swapped constituent spans 2^j leaves; "
                       "j = 0 is a single-leaf swap, i.e. a glitch drawn from the grammar's "
                       "own alphabet)**\n")
            out.append(row("arm", "j", "n", "w*(0)", "nll nat (0)", "nll at w* (0)",
                           "w*(1)", "w*(2)", "w*(4)"))
            out.append(sep(9))
            for a in ("edit", "glitch_s"):
                for j, rows in sorted(r["by_world"].get(a, {}).get("by_j", {}).items(),
                                      key=lambda kv: int(kv[0])):
                    g = {x["tau"]: x for x in rows}
                    out.append(row(a, j, g[0]["n"], f3(g[0]["w_star"]),
                                   f4(g[0]["nll_native"]), f4(g[0]["nll_at_wstar"]),
                                   f3(g[1]["w_star"]), f3(g[2]["w_star"]), f3(g[4]["w_star"])))
            out.append("")
            out.append(row("arm", "k*", "n", "w*(0)", "nll nat (0)", "nll at w* (0)", "w*(1)",
                           "w*(2)", "w*(4)"))
            out.append(sep(9))
            for a in ("edit",):
                for k, rows in sorted(r["by_world"][a].get("by_kstar", {}).items(),
                                      key=lambda kv: int(kv[0])):
                    g = {x["tau"]: x for x in rows}
                    out.append(row(a, k, g[0]["n"], f3(g[0]["w_star"]), f4(g[0]["nll_native"]),
                                   f4(g[0]["nll_at_wstar"]), f3(g[1]["w_star"]),
                                   f3(g[2]["w_star"]), f3(g[4]["w_star"])))
            out.append("")
        out.append("**Each fixed policy in each world** (pooled realised NLL over tau = 0..8)\n")
        out.append(row("policy", *[ARM_LABEL[a] for a in ARMS if a in r["cross_world"]["native"]]))
        out.append(sep(1 + len(ARMS)))
        for p, d in r["cross_world"].items():
            out.append(row(p + "  " + str(r["policies"][p][:3]) + "...",
                           *[f4(d[a]["nll_pooled"]) for a in ARMS if a in d]))
        out.append("")


def q2_lookback(root, out):
    out.append("## Q2b. Where the culprit is: imputing `delta` tokens before the flag\n")
    out.append("Top-4 prior imputation at position `t_v - delta`, scored at the same offsets. "
               "`delta_oracle` uses the true distance from the flag back to the start of the "
               "edited constituent (`t_v - e`), which is the culprit only when the edit is one "
               "leaf; for the glitch arms the culprit is always at `delta = 0`, so that row is "
               "a control.\n")
    for nm, r in loaded(root):
        out.append(f"### {nm}\n")
        out.append(row("arm", "delta", "mean delta", "w*(0)", "nll nat (0)", "nll at w* (0)",
                       "gain (0)", "gain (1)", "gain (2)"))
        out.append(sep(9))
        for a, dd in r["lookback"].items():
            for ln, d in dd.items():
                g = {x["tau"]: x for x in d["by_tau"]}
                out.append(row(a, ln, f"{d['delta_mean']:.1f}", f3(g[0]["w_star"]),
                               f4(g[0]["nll_native"]), f4(g[0]["nll_at_wstar"]),
                               f"{g[0]['nll_native'] - g[0]['nll_at_wstar']:+.4f}",
                               f"{g[1]['nll_native'] - g[1]['nll_at_wstar']:+.4f}",
                               f"{g[2]['nll_native'] - g[2]['nll_at_wstar']:+.4f}"))
        out.append("")
        ora = r["lookback"].get("edit", {}).get("delta_oracle", {}).get("by_j")
        if ora:
            out.append("**`delta_oracle` on the edit arm, by the edit's size `j`**\n")
            out.append(row("j", "n", "w*(0)", "nll nat (0)", "nll at w* (0)", "gain (0)"))
            out.append(sep(6))
            for j, rows in sorted(ora.items(), key=lambda kv: int(kv[0])):
                g = rows[0]
                out.append(row(j, g["n"], f3(g["w_star"]), f4(g["nll_native"]),
                               f4(g["nll_at_wstar"]),
                               f"{g['nll_native'] - g['nll_at_wstar']:+.4f}"))
            out.append("")


def q2_natural(root, out):
    out.append("## Q2c. The eps venue, where the exact ladder is on file\n")
    out.append("The same consumer on `coeruleus/`'s isolated corruptions. **Two references, "
               "both honest, and they disagree** (`coeruleus/`'s gotcha, here computed on both "
               "sides). Column A scores against `p_L^eps`, the law for an observer OF THE "
               "WINDOW, which does not know the token was corrupted -- its ceiling is the eps "
               "family at the model's own `kappa*`. Column B scores against `p_L^clean` on the "
               "uncorrupted draw of the same windows, which is the law the realised "
               "continuation is actually drawn from once you are told the token was noise -- "
               "its ceiling is the clean family at `kappa*_clean`, i.e. what the model's own "
               "altitude would have forecast had the glitch never happened.\n")
    out.append(row("model", "events", "kappa*_eps", "kappa*_clean",
                   "A: prize", "A: hold recovers", "A: w*-mix recovers",
                   "B: prize", "B: hold recovers", "B: w*-mix recovers",
                   "nll native", "nll hold"))
    out.append(sep(12))
    for nm, r in loaded(root):
        n_ = r.get("natural")
        if not n_ or "pooled" in n_ and "by_tau" not in n_:
            continue
        p, b = n_["pooled"], n_.get("pooledB", {})
        out.append(row(nm, n_["n_events"], f3(n_["kappa_star_eps"]),
                       f3(n_.get("kappa_star_clean")), f4(p["prize"]),
                       f3(p["frac_prize_hold"]), f3(p["frac_prize_wstar"]),
                       f4(b.get("prize")), f3(b.get("frac_prize_hold")),
                       f3(b.get("frac_prize_wstar")), f4(p["nll_native"]), f4(p["nll_hold"])))
    out.append("")
    for nm, r in loaded(root):
        n_ = r.get("natural")
        if not n_ or "by_tau" not in n_:
            continue
        out.append(f"### {nm} -- per offset\n")
        out.append(row("tau", "nll native", "nll hold", "w* (realised)",
                       "A: excess native", "A: excess hold", "A: w*",
                       "B: excess native", "B: excess hold", "B: w*",
                       "H native", "H hold", "H(p_L^clean)"))
        out.append(sep(13))
        for d in n_["by_tau"]:
            out.append(row(d["tau"], f4(d["nll_native"]), f4(d["nll_hold"]),
                           f3(d["w_star_realised"]), f4(d["excess_ceil_native"]),
                           f4(d["excess_ceil_hold"]), f3(d["w_star_ceil"]),
                           f4(d.get("excess_ceilB_native")), f4(d.get("excess_ceilB_hold")),
                           f3(d.get("w_star_ceilB")), f3(d["H_native"]), f3(d["H_hold"]),
                           f3(d.get("H_truthB"))))
        out.append("")


def q3(root, out):
    out.append("## Q3. The mixed world\n")
    out.append("A trigger fires; it is an edit, a glitch, or nothing, at the stated rates. "
               "The headline is the maximum-entropy choice, 1 : 1 : 1, with the full rate grid "
               "below. Every gated arm's map is fitted per offset on the FIT split's realised "
               "NLL at the same rates and read out on the held-out TEST split. `label` is the "
               "exact world; `shuf_*` is the same gate's values permuted across events -- the "
               "`endogenous_teacher` control that says whether a learned gate did anything the "
               "fitted map's marginal did not.\n")
    for nm, r in loaded(root):
        for gl in GLITCHES:
            m = r["mixed"]["by_glitch"].get(gl)
            if not m:
                continue
            base = m["fixed"]["native"]["nll"]
            out.append(f"### {nm} -- glitch arm `{gl}`, rates 1:1:1 "
                       f"(n_test = {r['gates']['n_test']} per arm)\n")
            out.append(row("policy", "mixed nll", "vs native", "w(tau=0) edit / glitch / quiet"))
            out.append(sep(4))
            for k, d in m["fixed"].items():
                w0 = (f"{d['w'][0]:.3f} (per offset)" if "w" in d else "")
                out.append(row(k, f4(d["nll"]), f"{d['nll'] - base:+.4f}", w0))
            for k, d in m["gated"].items():
                w = d["w_tau0_by_world"]
                out.append(row("gate: " + k, f4(d["nll"]), f"{d['nll'] - base:+.4f}",
                               f"{f3(w['edit'])} / {f3(w[gl])} / {f3(w['quiet'])}"))
            out.append("")
            bf = m["fixed"]["best_fixed_fit"]["nll"]
            lab = m["gated"]["label"]["nll"]
            orc = m["fixed"]["oracle_per_pred"]["nll"]
            out.append(f"*The arbitration prize.* A gate can only earn what knowing the world "
                       f"earns: the exact label is worth {bf - lab:.4f} nats over the best "
                       f"fixed per-offset `w` ({f4(bf)} -> {f4(lab)}), and an oracle choosing "
                       f"`w` per prediction {bf - orc:.4f}. Each learned gate below is scored "
                       f"against that, and against its own shuffle -- the only comparison that "
                       f"says whether the gate's ordering did anything.\n")
            out.append(row("gate", "mixed nll", "vs best fixed", "vs its own shuffle",
                           "share of the label prize"))
            out.append(sep(5))
            for k, d in m["gated"].items():
                if k.startswith("shuf_") or k == "label":
                    continue
                sh = m["gated"].get("shuf_" + k, {}).get("nll")
                out.append(row(k, f4(d["nll"]), f"{d['nll'] - bf:+.4f}",
                               "n/a" if sh is None else f"{d['nll'] - sh:+.4f}",
                               f"{(bf - d['nll']) / (bf - lab):+.2f}" if bf > lab else "n/a"))
            out.append("")
    out.append("### Rate sensitivity (mixed nll; `edit share` is among the two event "
               "classes, `quiet` is the false-alarm rate)\n")
    for nm, r in loaded(root):
        rows = r["mixed"]["grid"]
        gates = sorted({g for x in rows for g in x["gated"]})
        out.append(f"**{nm}**\n")
        out.append(row("glitch arm", "edit", "glitch", "quiet", "native", "hold",
                       "best fixed", "oracle/pred", *[f"gate {g}" for g in gates]))
        out.append(sep(8 + len(gates)))
        for x in rows:
            rr = x["rates"]
            out.append(row(x.get("glitch_arm", ""), f3(rr["edit"]), f3(rr["glitch"]),
                           f3(rr["quiet"]), f4(x["fixed"]["native"]), f4(x["fixed"]["hold"]),
                           f4(x["fixed"]["best_fixed_fit"]), f4(x["fixed"]["oracle_per_pred"]),
                           *[f4(x["gated"].get(g)) for g in gates]))
        out.append("")


def q4(root, out):
    out.append("## Q4. Gates: what separates the worlds, and how fast\n")
    out.append("AUC(edit vs glitch) on the held-out split at each offset after the flag. 0.5 is "
               "no separation; below 0.5 means the gate reads higher on the glitch (the sign is "
               "informative, the distance from 0.5 is the separation). `probe` is a supervised "
               "world-label probe on the residual stream -- the upper bound on how fast the "
               "state separates the worlds -- with `probe_embed` (the token+position embedding) "
               "and `probe_rand` (a random-init trunk) as its floors.\n")
    for nm, r in loaded(root):
        for gl in GLITCHES:
            sp = r["gates"]["separation"].get(gl)
            if not sp:
                continue
            out.append(f"### {nm} -- edit vs `{gl}`\n")
            out.append(row("gate", *[f"tau={t}" for t in range(len(
                next(iter(sp.values()))["auc_edit_vs_glitch_by_tau"]))]))
            out.append(sep(1 + len(next(iter(sp.values()))["auc_edit_vs_glitch_by_tau"])))
            for g, d in sp.items():
                out.append(row(g, *[f3(x) for x in d["auc_edit_vs_glitch_by_tau"]]))
            out.append("")
            out.append(row("gate (vs quiet)", *[f"tau={t}" for t in range(len(
                next(iter(sp.values()))["auc_edit_vs_quiet_by_tau"]))]))
            out.append(sep(1 + len(next(iter(sp.values()))["auc_edit_vs_quiet_by_tau"])))
            for g, d in sp.items():
                out.append(row(g, *[f3(x) for x in d["auc_edit_vs_quiet_by_tau"]]))
            out.append("")
    out.append("### The flagged token is not the signal: the surprisal-matched guard\n")
    out.append("`glitch_m`'s corrupting token is a uniform draw and so is more surprising than "
               "the edit's violator; these rows re-read every gate on a nearest-neighbour "
               "subset matched on the model's own surprisal for the flagged token, which pins "
               "that guard at 0.5.\n")
    for nm, r in loaded(root):
        g = r["gates"].get("matched_guard", {})
        if not g.get("by_gate"):
            continue
        out.append(f"**{nm}** (n = {g['n_matched']} matched pairs, caliper {g['caliper']}, "
                   f"guard AUC on the flagged token's surprisal {f3(g['guard_sevent_auc'])})\n")
        k0 = next(iter(g["by_gate"]))
        out.append(row("gate", *[f"tau={t}" for t in range(len(g["by_gate"][k0]))]))
        out.append(sep(1 + len(g["by_gate"][k0])))
        for k, v in g["by_gate"].items():
            out.append(row(k, *[f3(x) for x in v]))
        out.append("")


def q5(root, out):
    out.append("## Q2d. The parent's 2e entropy sign and the persistence profile "
               "under each policy\n")
    out.append("Same-prefix pairs (the parent's Part 2c construction). The violator and its "
               "legal twin differ only at `t_v`, so the imputed forecast is IDENTICAL between "
               "them and the paired entropy statistic is 0.5 by construction at w = 1 -- "
               "reported as an exact self-check on the whole imputation path.\n")
    for nm, r in loaded(root):
        p = r.get("pairs")
        if not p or not p.get("readouts"):
            continue
        out.append(f"### {nm} (n_test = {p['n_test']} pairs, prefix check "
                   f"{p['prefix_check']:.1e}, hold-identical check "
                   f"{p.get('hold_identical_check', float('nan')):.1e})\n")
        out.append(row("w", "H_next paired (>0.5 = less confident after the violation)",
                       "H_next AUC", "nll viol (tau=1)", "nll twin (tau=1)",
                       "nll paired (tau=1)", "dH viol tau=0..3"))
        out.append(sep(7))
        for w, d in p["readouts"].items():
            out.append(row(w, f3(d["H_next_paired"]), f3(d["H_next_auc"]),
                           f4(d["nll_viol_by_tau"][1]), f4(d["nll_twin_by_tau"][1]),
                           f3(d["nll_paired_tau1"]),
                           " ".join(f"{x:+.3f}" for x in d["dH_viol_by_tau"][:4])))
        out.append("")
    out.append("### Unmatched output-level persistence, every arm, under each policy\n")
    out.append("`dH` is the model's entropy at `t_v + tau` minus its own entropy at `t_v - 1`. "
               "**This is not the parent's 2e `dH`**, which differenced the edited window "
               "against the unedited one at the same offset; here the reference is the same "
               "window one step earlier, so the `quiet` arm is the control that makes the "
               "column readable (a flagged position where nothing happened reads +0.003). "
               "Read it for what each policy does to the model's stated uncertainty, not "
               "against the parent's numbers.\n")
    for nm, r in loaded(root):
        out.append(f"**{nm}**\n")
        out.append(row("arm", "w", *[f"dH(tau={t})" for t in range(9)]))
        out.append(sep(11))
        for a in ARMS:
            for w, d in r["persistence"].get(a, {}).items():
                out.append(row(a, w, *[f"{x:+.3f}" for x in d["dH_by_tau"]]))
        out.append("")


# ---------------------------------------------------------------------------

def figures(root, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    ms = loaded(root)
    col = {"edit": "#c44e52", "glitch_s": "#4c72b0", "glitch_m": "#8fa8cf", "quiet": "#888888"}

    # 1. the two policies per world
    fig, axes = plt.subplots(2, len(ms), figsize=(5.2 * len(ms), 7), squeeze=False)
    for i, (nm, r) in enumerate(ms):
        ax = axes[0][i]
        for a in ARMS:
            d = r["by_world"].get(a)
            if not d:
                continue
            t = [x["tau"] for x in d["all"]]
            ax.plot(t, [x["w_star"] for x in d["all"]], "o-", color=col[a],
                    label=ARM_LABEL[a])
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("offset tau after the flagged token")
        ax.set_ylabel("oracle mixing weight w*")
        ax.set_title(f"{nm}: how much to hold"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        ax = axes[1][i]
        for a in ARMS:
            d = r["by_world"].get(a)
            if not d:
                continue
            t = [x["tau"] for x in d["all"]]
            ax.plot(t, [x["nll_native"] for x in d["all"]], "o-", color=col[a],
                    label=f"{a} native")
            ax.plot(t, [x["nll_hold"] for x in d["all"]], "s--", color=col[a], alpha=.6,
                    label=f"{a} hold")
        ax.set_xlabel("offset tau"); ax.set_ylabel("realised NLL")
        ax.set_title("reinterpret (solid) vs hold (dashed)"); ax.grid(alpha=.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Q2: the same event, two worlds, opposite right answers")
    fig.tight_layout(); fig.savefig(f"{figdir}/worlds.png", dpi=140); plt.close(fig)

    # 2. gate separation
    fig, axes = plt.subplots(1, len(ms), figsize=(5.6 * len(ms), 4.2), squeeze=False)
    for i, (nm, r) in enumerate(ms):
        ax = axes[0][i]
        sp = r["gates"]["separation"].get("glitch_s", {})
        for g, d in sp.items():
            y = np.array(d["auc_edit_vs_glitch_by_tau"])
            ls = "--" if g.startswith("probe_") or g in ("probe_embed", "probe_rand") else "-"
            ax.plot(range(len(y)), y, ls, marker="o", ms=3, label=g)
        ax.axhline(0.5, color="k", lw=.8)
        ax.set_xlabel("offset tau"); ax.set_ylabel("AUC(edit vs glitch)")
        ax.set_title(f"{nm}"); ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Q4: how fast anything readable tells the two worlds apart "
                 "(surprisal-matched glitch)")
    fig.tight_layout(); fig.savefig(f"{figdir}/gates.png", dpi=140); plt.close(fig)

    # 3. the mixed world
    fig, axes = plt.subplots(1, len(ms), figsize=(6.5 * len(ms), 4.6), squeeze=False)
    for i, (nm, r) in enumerate(ms):
        ax = axes[0][i]
        m = r["mixed"]["by_glitch"].get("glitch_s") or r["mixed"]["headline"]
        names = list(m["fixed"]) + ["gate: " + g for g in m["gated"]]
        vals = [m["fixed"][k]["nll"] for k in m["fixed"]] + \
               [m["gated"][g]["nll"] for g in m["gated"]]
        cols = ["#c44e52" if k in ("native", "hold") else
                ("#55a868" if ("oracle" in k or "label" in k) else
                 ("#cccccc" if "shuf" in k else "#4c72b0")) for k in names]
        ax.barh(range(len(names)), vals, color=cols)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo - 0.12 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xlabel("mixed-world realised NLL (lower is better)")
        ax.set_title(nm); ax.grid(alpha=.3, axis="x")
    fig.suptitle("Q3: a trigger fires at 1:1:1 edit / glitch / nothing -- what a gate is worth")
    fig.tight_layout(); fig.savefig(f"{figdir}/mixed.png", dpi=140); plt.close(fig)

    # 4. the lookback bound
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 4), squeeze=False)
    for i, (nm, r) in enumerate(ms):
        ax = axes[0][i]
        for a, dd in r["lookback"].items():
            xs, ys = [], []
            for ln, d in dd.items():
                if not ln.startswith("delta") or ln == "delta_oracle":
                    continue
                g = d["by_tau"][0]
                xs.append(d["delta_mean"])
                ys.append(g["nll_native"] - g["nll_at_wstar"])
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", color=col.get(a, "k"), label=a)
        ax.set_xlabel("delta: tokens before the flag that are imputed")
        ax.set_ylabel("nats a hold buys at tau = 0")
        ax.set_title(nm); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("Q2b: a single-token repair pays only where the damage is a single token")
    fig.tight_layout(); fig.savefig(f"{figdir}/lookback.png", dpi=140); plt.close(fig)
    print(f"figures -> {figdir}")


# ---------------------------------------------------------------------------
# Addendum (2026-09-17): `gates2.py`
# ---------------------------------------------------------------------------

G2 = [("traj_eps01_s42/step064000_basalis_gates2.json", "eps-trained 64k"),
      ("traj_a1_s42/step064000_basalis_gates2.json", "clean-trained 64k")]


def loaded2(root):
    return [(nm, r) for rel, nm in G2 for r in [load(root, rel)] if r is not None]


def _short(g, gl):
    return g.replace(f"_{gl}", "")


def addendum(root, out):
    ms = loaded2(root)
    if not ms:
        return
    out.append("\n---\n")
    out.append("# Addendum, 2026-09-17 --- `gates2.py`\n")
    out.append("The first round found that a supervised world probe separates the two worlds "
               "(0.71 at the flagged token, surprisal-matched, against a 0.52 floor) and yet, "
               "used as a gate, is worse than its own shuffle. Four additions test the mundane "
               "explanations for that before it is read as a fact about the state. Same "
               "populations, same seed, same 60/40 split as the main run, asserted against the "
               "main JSON's per-arm event surprisals; a separate artifact, so nothing above is "
               "recomputed.\n")
    out.append(row("model", "population check (mean s_event vs the main run)", "n fit / test",
                   "events at j >= 2", "t_v > e", "identical-prefix state check (t_v == e)"))
    out.append(sep(6))
    for nm, r in ms:
        pc = r["population_check"]
        out.append(row(nm, "0 (exact)" if pc.get("checked") else "not checked",
                       f"{r['split']['n_fit']} / {r['split']['n_test']}",
                       list(r["subsets"].values())[1],
                       f"{r['preamble']['frac_tv_gt_e']:.2f}",
                       f"{r['preamble']['identical_prefix_state_check']:.1e}"))
    out.append("")

    # ---- A1 ------------------------------------------------------------
    out.append("## A1. Restricting to structural edits (`j >= 2`)\n")
    out.append("`j = 0` and `j = 1` swaps behave as glitches by the main round's own `j`-split, "
               "so provenance is the wrong label for them and a provenance-trained probe is "
               "taught to call glitch-like events \"edit\". Everything is recomputed on `j >= 2` "
               "edits against their paired glitches, beside the full population.\n")
    for nm, r in ms:
        for gl in GLITCHES:
            out.append(f"### {nm} -- glitch arm `{gl}`\n")
            out.append(row("subset", "n test / arm", "native", "hold", "best fixed", "label",
                           "oracle / pred", "arbitration prize (best fixed - label)"))
            out.append(sep(8))
            for sname, blk in r["by_subset"].items():
                m = blk["mixed"].get(gl)
                if not m:
                    continue
                bf, lab = m["fixed"]["best_fixed_fit"]["nll"], m["gated"]["label"]["nll"]
                out.append(row(sname, m["n_test_per_arm"], f4(m["fixed"]["native"]["nll"]),
                               f4(m["fixed"]["hold"]["nll"]), f4(bf), f4(lab),
                               f4(m["fixed"]["oracle_per_pred"]["nll"]), f"{bf - lab:+.4f}"))
            out.append("")
            names = [g for g in r["by_subset"]["all"]["mixed"][gl]["gated"]
                     if not g.startswith("shuf_") and g != "label"]
            out.append(row("gate", *[f"{s}: nll / vs shuffle / share of prize"
                                     for s in r["by_subset"]]))
            out.append(sep(1 + len(r["by_subset"])))
            for g in names:
                cells = []
                for sname, blk in r["by_subset"].items():
                    m = blk["mixed"][gl]
                    if g not in m["gated"]:
                        cells.append("n/a")
                        continue
                    bf, lab = m["fixed"]["best_fixed_fit"]["nll"], m["gated"]["label"]["nll"]
                    sh = m["gated"].get("shuf_" + g, {}).get("nll")
                    val = m["gated"][g]["nll"]
                    vs = "n/a" if sh is None else f"{val - sh:+.4f}"
                    cells.append(f"{f4(val)} / {vs} / {(bf - val) / (bf - lab):+.2f}")
                out.append(row(_short(g, gl), *cells))
            out.append("")

    # ---- A2 ------------------------------------------------------------
    out.append("## A2. What each gate is ordered on: the world, or where holding paid\n")
    out.append("Left: AUC(edit vs glitch), the world. Right: AUC against `b > a` -- did holding "
               "give the realised token more mass at this position -- pooled over the three arms "
               "of the mixed world on the test split. This is the label the fitted map is built "
               "on, so it is the ordering a gate has to have. Both on the held-out split; 0.5 is "
               "chance and the distance from 0.5 is the separation.\n")
    for nm, r in ms:
        for sname, blk in r["by_subset"].items():
            sp = blk["separation"].get(GLITCHES[0])
            if not sp:
                continue
            gl0 = GLITCHES[0]
            others = [o for o in GLITCHES if o != gl0]
            gates = [g for g in sp if isinstance(sp[g], dict)
                     and "auc_holdbetter_by_tau" in sp[g]
                     and not g.startswith(f"probe_{gl0}")
                     and not any(g.endswith("_" + o) for o in others)]
            out.append(f"### {nm} -- `{sname}`, edit vs `{GLITCHES[0]}`\n")
            out.append(row("gate", "world AUC tau=0", "tau=1", "tau=2", "tau=4",
                           "hold-better AUC tau=0", "tau=1", "tau=2", "tau=4"))
            out.append(sep(9))
            for g in gates:
                w_ = sp[g]["auc_edit_vs_glitch_by_tau"]
                h_ = sp[g]["auc_holdbetter_by_tau"]
                out.append(row(_short(g, GLITCHES[0]), f3(w_[0]), f3(w_[1]), f3(w_[2]),
                               f3(w_[4]), f3(h_[0]), f3(h_[1]), f3(h_[2]), f3(h_[4])))
            out.append("")
    out.append("### The payoff-trained probe on its own target\n")
    out.append("A probe on the same states whose label is `b > a` (no world label anywhere), "
               "trained on the fit split pooled over the three arms.\n")
    for nm, r in ms:
        for gl, d in r.get("new_probes", {}).items():
            out.append(row(f"{nm} / {gl}", *[f"tau={t}" for t in
                                             range(len(d["payoff_auc_holdbetter_by_tau"]))]))
            out.append(sep(1 + len(d["payoff_auc_holdbetter_by_tau"])))
            out.append(row("payoff probe, AUC(hold-better)",
                           *[f3(x) for x in d["payoff_auc_holdbetter_by_tau"]]))
            out.append(row("3-class probe, AUC(hold-better)",
                           *[f3(x) for x in d["probe3_auc_holdbetter_by_tau"]]))
            out.append("")

    # ---- A3 ------------------------------------------------------------
    out.append("## A3. The three-class probe, which has seen false alarms\n")
    out.append("Classes {edit, glitch, quiet}; gate = `log p(glitch) - log p(edit)`.\n")
    for nm, r in ms:
        for gl, d in r.get("new_probes", {}).items():
            out.append(row(f"{nm} / {gl}", *[f"tau={t}" for t in
                                             range(len(d["probe3_3way_accuracy_by_tau"]))]))
            out.append(sep(1 + len(d["probe3_3way_accuracy_by_tau"])))
            out.append(row("3-way accuracy (chance 0.33)",
                           *[f3(x) for x in d["probe3_3way_accuracy_by_tau"]]))
            out.append(row("p(quiet) it assigns to a quiet event",
                           *[f3(x) for x in d["probe3_p_quiet_on_quiet_by_tau"]]))
            out.append("")
    out.append(row("model", "glitch arm", "subset", "w it sets at tau=0: edit / glitch / quiet",
                   "mixed nll", "vs its shuffle"))
    out.append(sep(6))
    for nm, r in ms:
        for gl in GLITCHES:
            for sname, blk in r["by_subset"].items():
                m = blk["mixed"].get(gl, {})
                g = f"probe3_{gl}"
                if g not in m.get("gated", {}):
                    continue
                w0 = m["gated"][g]["w_tau0_by_world"]
                sh = m["gated"].get("shuf_" + g, {}).get("nll")
                out.append(row(nm, gl, sname,
                               f"{f3(w0['edit'])} / {f3(w0[gl])} / {f3(w0['quiet'])}",
                               f4(m["gated"][g]["nll"]),
                               "n/a" if sh is None else f"{m['gated'][g]['nll'] - sh:+.4f}"))
    out.append("")

    # ---- A4 ------------------------------------------------------------
    out.append("## A4. The preamble: what the state knows BEFORE the flagged token\n")
    out.append("The world probe read on the state at `t_v - 1`. Where `t_v == e` the edit and "
               "glitch windows share their entire prefix, so those states are identical and the "
               "probe must read exactly 0.500 -- an internal floor, asserted at 0 to machine "
               "precision. `runsurp4` is the parent's 2b scalar (the model's own mean surprisal "
               "over the four predictions already scored when the forecast at `t_v - 1` is "
               "made); `runex4` is the same minus the entropy it stated.\n")
    for nm, r in ms:
        for gl, pb in r["preamble"].get("by_glitch", {}).items():
            blkname = [k for k in pb if k.startswith("post_block")]
            blkname = blkname[0] if blkname else "post_embed"
            d = pb[blkname]
            out.append(f"### {nm} -- edit vs `{gl}`\n")
            out.append(row("readout at t_v - 1", "all test", f"t_v > e (n={d['n_gt']})",
                           f"t_v == e (n={d['n_eq']})", f"matched (n={d['n_matched']})",
                           "matched, t_v > e",
                           "reference: the same probe at tau = 0, matched"))
            out.append(sep(7))
            out.append(row(f"state probe ({blkname})", f3(d["auc_tm1_all"]),
                           f3(d["auc_tm1_tv_gt_e"]), f3(d["auc_tm1_tv_eq_e"]),
                           f3(d["auc_tm1_matched"]),
                           f3(d.get("auc_tm1_matched_tv_gt_e")),
                           f3(d.get("auc_tau0_matched_ref"))))
            if "post_embed" in pb:
                e = pb["post_embed"]
                out.append(row("state probe (post_embed floor)", f3(e["auc_tm1_all"]),
                               f3(e["auc_tm1_tv_gt_e"]), f3(e["auc_tm1_tv_eq_e"]),
                               f3(e["auc_tm1_matched"]),
                               f3(e.get("auc_tm1_matched_tv_gt_e")), "--"))
            for k2 in ("runsurp4", "runex4", "Hq_prev"):
                if k2 in pb:
                    out.append(row(k2, f3(pb[k2]["auc_tm1_all"]),
                                   f3(pb[k2]["auc_tm1_tv_gt_e"]), "n/a (identical)",
                                   f3(pb[k2]["auc_tm1_matched"]), "n/a",
                                   f"edit {f3(pb[k2]['mean_edit'])} vs glitch "
                                   f"{f3(pb[k2]['mean_glitch'])}"))
            out.append("")


def a5(root, out):
    ms = loaded2(root)
    sp0 = ms[0][1]["by_subset"]["all"]["separation"][GLITCHES[0]] if ms else {}
    if not any(isinstance(v, dict) and "spearman_logba_by_tau" in v for v in sp0.values()):
        return
    out.append("## A5. Ordering by the sign of the payoff versus ordering by its size\n")
    out.append("`log(b/a)` is the signed, continuous payoff -- how much more (or less) mass the "
               "held forecast put on the token that actually came. Its rank correlation with a "
               "gate is the direct form of \"is this gate ordered by the stake\". Beside it: the "
               "gate's AUC on the SIGN of that payoff (`b > a`), its world AUC, and what the "
               "gate was finally worth in the mixed world against its own shuffle. All on the "
               "same held-out rows, pooled over the three arms. A bin map is invariant to any "
               "monotone reparametrisation of a gate, so only the ordering can matter -- "
               "`binned R2` is that same step function used to predict the stake instead of "
               "`w`, fit on the fit rows and scored on the test rows, i.e. how much of the "
               "stake a gate recovers through the machinery the gate actually has.\n")
    for nm, r in ms:
        for sname, blk in r["by_subset"].items():
            gl = GLITCHES[0]
            sp = blk["separation"].get(gl, {})
            mx = blk["mixed"].get(gl, {})
            gates = [g for g in sp if isinstance(sp[g], dict)
                     and "spearman_logba_by_tau" in sp[g]
                     and not g.startswith(f"probe_{gl}")
                     and not any(g.endswith("_" + o) for o in GLITCHES if o != gl)]
            if not gates:
                continue
            out.append(f"### {nm} -- `{sname}`, vs `{gl}`\n")
            out.append(row("gate", "rho(gate, log b/a) tau=0", "tau=1", "tau=2", "tau=4",
                           "binned R2 on log(b/a) tau=0", "tau=1",
                           "AUC on sign(b-a), tau=1", "world AUC tau=1",
                           "mixed nll vs its shuffle"))
            out.append(sep(10))
            for g in gates:
                rr = sp[g]["spearman_logba_by_tau"]
                br = sp[g].get("binr2_logba_by_tau", [None] * 9)
                gk = f"probe_{gl}" if g == "probe" else g
                d = mx.get("gated", {}).get(gk, {})
                sh = mx.get("gated", {}).get("shuf_" + gk, {}).get("nll")
                vs = ("n/a" if (not d or sh is None) else f"{d['nll'] - sh:+.4f}")
                out.append(row(_short(g, gl), f3(rr[0]), f3(rr[1]), f3(rr[2]), f3(rr[4]),
                               f4(br[0]), f4(br[1]),
                               f3(sp[g]["auc_holdbetter_by_tau"][1]),
                               f3(sp[g]["auc_edit_vs_glitch_by_tau"][1]), vs))
            out.append("")


def a6(root, out):
    ms = [(nm, r) for nm, r in loaded2(root) if "stake" in r]
    if not ms:
        return
    out.append("## A6. A readout trained on the stake itself\n")
    out.append("The same states, the same fit split, trained to predict `log(b/a)` -- a "
               "label-free target, since it needs only the second forward pass the consumer "
               "already runs. Exact closed-form ridge (lambda chosen on a held-out slice of "
               "the fit rows), an MLP beside it, and the model's own log Bayes factor rescaled "
               "on the same rows so all three are compared as PREDICTORS before they are "
               "compared as gates.\n")
    for nm, r in ms:
        for sname, d in r["stake"].items():
            for gl, dd in d.items():
                if gl != GLITCHES[0]:
                    continue
                out.append(f"### {nm} -- `{sname}`, vs `{gl}` "
                           f"({dd['n_fit_rows']} fit rows, {dd['n_test_rows']} test rows)\n")
                out.append(row("tau", "sd of log(b/a)", "ridge R2", "MLP R2",
                               "logbf rescaled R2", "ridge rho", "MLP rho", "logbf rho",
                               "lambda"))
                out.append(sep(9))
                for x in dd["by_tau"]:
                    out.append(row(x["tau"], f3(x["stake_sd_test"]), f4(x["ridge_R2"]),
                                   f4(x["mlp_R2"]), f4(x["logbf_rescaled_R2"]),
                                   f3(x["ridge_spearman"]), f3(x["mlp_spearman"]),
                                   f3(x["logbf_spearman"]), x["lam"]))
                out.append("")
    out.append("### The same two readouts used as gates\n")
    out.append(row("model", "glitch arm", "subset", "gate", "mixed nll", "vs its shuffle",
                   "share of the label prize", "w at tau=0: edit / glitch / quiet"))
    out.append(sep(8))
    for nm, r in ms:
        for gl in GLITCHES:
            for sname, blk in r["by_subset"].items():
                m = blk["mixed"].get(gl, {})
                bf = m.get("fixed", {}).get("best_fixed_fit", {}).get("nll")
                lab = m.get("gated", {}).get("label", {}).get("nll")
                for g in (f"stake_ridge_{gl}", f"stake_mlp_{gl}"):
                    if g not in m.get("gated", {}):
                        continue
                    val = m["gated"][g]["nll"]
                    sh = m["gated"].get("shuf_" + g, {}).get("nll")
                    w0 = m["gated"][g]["w_tau0_by_world"]
                    out.append(row(nm, gl, sname, _short(g, gl), f4(val),
                                   "n/a" if sh is None else f"{val - sh:+.4f}",
                                   f"{(bf - val) / (bf - lab):+.2f}" if bf and lab else "n/a",
                                   f"{f3(w0['edit'])} / {f3(w0[gl])} / {f3(w0['quiet'])}"))
    out.append("")
    chk = [(nm, r.get("regression_check", {})) for nm, r in ms]
    if any(c.get("max_abs_diff") is not None for _, c in chk):
        out.append("*Regression check.* A5 and A6 only add gates and columns, so every A1-A4 "
                   "scalar must be unchanged from the previous artifact. Max |difference| over "
                   "every numeric leaf shared by the two: " +
                   ", ".join(f"{nm} {c['max_abs_diff']:.1e}"
                             for nm, c in chk if c.get("max_abs_diff") is not None) + ".\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--figs", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = ["# basalis -- reduced tables\n",
           "Generated by `python -m rhm.logit_reading.basalis.analyze <dir>`; every number "
           "below comes from the per-checkpoint `*_basalis.json` on the volume.\n"]
    q0(a.root, out)
    q1(a.root, out)
    q2(a.root, out)
    q2_lookback(a.root, out)
    q2_natural(a.root, out)
    q5(a.root, out)
    q3(a.root, out)
    q4(a.root, out)
    addendum(a.root, out)
    a5(a.root, out)
    a6(a.root, out)
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
