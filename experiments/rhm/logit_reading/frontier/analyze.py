"""Local reduction for the `frontier` node. Reads JSON only -- no Modal, no GPU.

    python3 -m rhm.logit_reading.frontier.analyze --tag w0 --fetch --figs

`--fetch` pulls every JSON this needs off the `rhm-scaling-data` volume into
`results/raw/`; without it, whatever is already there is used. Writes
`results/tables.md` (every number) and `figs/*.png`.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
VOL = "rhm-scaling-data"
BASE = "/v16_s2_L6_m4_distinct/logit_reading"
TRAJ = f"{BASE}/traj_a1_s42"
ARMS = ["fresh", "frozen_200000", "frozen_16384", "frozen_2048"]
LVLS = list(range(7))
OUT = []


def emit(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line)
    print(line)


def table(rows, head):
    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) if rows else len(str(h))
         for i, h in enumerate(head)]
    emit("| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)) + " |")
    emit("|" + "|".join("-" * (w[i] + 2) for i in range(len(head))) + "|")
    for r in rows:
        emit("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    emit("")


def fetch(tag):
    os.makedirs(RAW, exist_ok=True)
    want = [(f"{BASE}/frontier/frontier_panel.json", "frontier_panel.json"),
            (f"{BASE}/frontier/refs_{tag}.json", f"refs_{tag}.json"),
            (f"{BASE}/frontier/{tag}/summary.json", f"{tag}_summary.json"),
            (f"{TRAJ}/altitude_identity.json", "altitude_identity.json"),
            (f"{TRAJ}/altitude_units.json", "altitude_units.json"),
            (f"{TRAJ}/altitude_residual.json", "altitude_residual.json"),
            (f"{TRAJ}/altitude_frontier.json", "altitude_frontier.json"),
            (f"{BASE}/altitude_phase_identifiability.json",
             "altitude_phase_identifiability.json")]
    want += [(f"{BASE}/frontier/{tag}/{a}.json", f"{tag}_{a}.json") for a in ARMS]
    want += [(f"/rhm_practice_reread_lm/lm0/{a}.json", f"lm0_{a}.json") for a in ARMS]
    for src, dst in want:
        p = os.path.join(RAW, dst)
        r = subprocess.run(["modal", "volume", "get", VOL, src, p, "--force"],
                           capture_output=True, text=True)
        print(("  got " if r.returncode == 0 else "  MISS "), dst, file=sys.stderr)


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ik(d):
    """JSON string keys -> int keys."""
    return {int(k): v for k, v in (d or {}).items()}


def f(x, n=3, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


# --------------------------------------------------------------------------- #
# Q1 -- the endogenous panel on a learner that climbs
# --------------------------------------------------------------------------- #

def q1(panel, ident, units, resid, alt_front, phase_id):
    steps = [r["step"] for r in panel]
    kap = {r["step"]: r["altitude"]["convex"]["kappa_star"] for r in (ident or [])}
    ora = [ik(r["by_level_oracle"]) for r in panel]
    endo = {m: [ik(r["by_level_endo"][m]) for r in panel] for m in ("template", "meanprof")}

    emit("\n# frontier — results\n")
    emit("Facts only; no interpretation. Regenerate with\n")
    emit("```bash")
    emit("cd experiments            # MODAL_PROFILE=chromatic")
    emit("# Q1 (CPU, ~40 s): reduces the 13 stepNNNNNN_calibration.npz the parent wrote")
    emit("modal run -m rhm.logit_reading.frontier.panel::panel_sweep \\")
    emit("    --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42")
    emit("# Q2 (GPU): the shared model-independent references, then the four arms, one")
    emit("# L4 container each (~14 min wall clock)")
    emit("modal run -m rhm.logit_reading.frontier.wall::build_refs --tag w0")
    emit("modal run --detach -m rhm.logit_reading.frontier.wall::wall_sweep --tag w0")
    emit("# tables and figures")
    emit("python3 -m rhm.logit_reading.frontier.analyze --tag w0 --fetch --figs")
    emit("```\n")
    emit("Volume (`rhm-scaling-data`, workspace `chromatic`): "
         "`/data/v16_s2_L6_m4_distinct/logit_reading/frontier/` holds "
         "`frontier_panel.json` (Q1), `refs_w0.{npz,json}`, and `w0/` with one "
         "`<arm>.json` per arm, `<arm>_stepNNNNNN.pt` at 13 steps per arm, and "
         "`summary.json`. Figures: `figs/q1_panel.png`, `q1_endogenous.png`, "
         "`q2_wall.png`, `q2_by_level.png`.\n")
    emit("Substrate, checkpoints and eval windows are the parent's: `v16 s2 L6 m4`, "
         "`generate_rules_distinct(seed=0)`, 8L/8H/256D, the 13-checkpoint `traj_a1_s42` "
         "trajectory (Dirichlet(1) synonym weights), 4096 held-out flat windows at "
         "`eval_seed 4242` shared by every checkpoint. Leaf level `l` is "
         "`flat_oracle.leaf_levels` of the target's absolute leaf index: the highest "
         "constituent the token opens, 0 = odd leaf … 6 = a sequence boundary. "
         "Per-position counts per window are 32/16/8/4/2/1/1 for l = 0…6.\n")

    emit("## Q1a. The excess by leaf level, with ORACLE labels\n")
    emit("`CE_l - H(q)_l`, nats. `CE` is the exact expected loss under `p_L` on the eval "
         "windows; `H(q)` the model's stated entropy. altitude Q1: this is the "
         "temperature-direction derivative of the loss, so a converged softmax with a free "
         "logit scale sits at zero in every class it can scale separately.\n")
    table([[st, f(kap.get(st), 2)] + [f(ora[i][l]["excess_oracle"], 4, True) for l in LVLS]
           for i, st in enumerate(steps)],
          ["step", "kappa*"] + [f"l={l}" for l in LVLS])

    emit("## Q1a2. The same, REALISED (`NLL_l - H(q)_l`) — what the learner can compute\n")
    emit("Monte-Carlo standard errors of the realised excess at the last checkpoint: "
         + " ".join(f"l={l}:{ora[-1][l]['se_excess_realised']:.4f}" for l in LVLS) + "\n")
    table([[st] + [f(ora[i][l]["excess_realised"], 4, True) for l in LVLS]
           for i, st in enumerate(steps)], ["step"] + [f"l={l}" for l in LVLS])

    emit("## Q1a3. Excess in units of its own standard error (realised, oracle labels)\n")
    table([[st] + [f(ora[i][l]["excess_realised"] / ora[i][l]["se_excess_realised"], 1, True)
                   for l in LVLS] for i, st in enumerate(steps)],
          ["step"] + [f"l={l}" for l in LVLS])

    emit("## Q1a4. `NLL_l - CE_l` — the realised estimator's offset from the exact one\n")
    emit("Same positions, same tokens: `E[NLL_l] = E[CE_l]` by the tower property, so this "
         "is the sampling fluctuation of the ONE fixed eval window set. It does not shrink "
         "with training (the tokens are fixed), so it is a floor on how well a realised "
         "per-level excess can approximate the exact one. It is the whole difference "
         "between Q1a and Q1a2.\n")
    table([[st] + [f(ora[i][l]["NLL"] - ora[i][l]["CE"], 4, True) for l in LVLS]
           for i, st in enumerate(steps)], ["step"] + [f"l={l}" for l in LVLS])

    emit("## Q1b-GT. The ground truth: `KL(p_L||q)` by leaf level\n")
    table([[st, f(kap.get(st), 2)] + [f(ora[i][l]["KL_pL_q"], 4) for l in LVLS]
           for i, st in enumerate(steps)],
          ["step", "kappa*"] + [f"l={l}" for l in LVLS])

    emit("## Q1b. The loss trend `-Delta CE_l` per checkpoint interval (learning progress)\n")
    emit("On a FIXED window set `CE_l = KL(p_L||q)_l + H(p_L)_l` with `H(p_L)_l` constant "
         "across checkpoints, so `-Delta CE_l` and `-Delta KL_l` are the same number "
         "identically; the max |difference| over all (interval, level) is printed below. "
         "The endogenous estimator of the same quantity is `-Delta NLL_l`.\n")
    dmax = 0.0
    rows_ce, rows_nll, rows_kl = [], [], []
    for i in range(1, len(steps)):
        rce = [ora[i - 1][l]["CE"] - ora[i][l]["CE"] for l in LVLS]
        rkl = [ora[i - 1][l]["KL_pL_q"] - ora[i][l]["KL_pL_q"] for l in LVLS]
        rnl = [ora[i - 1][l]["NLL"] - ora[i][l]["NLL"] for l in LVLS]
        dmax = max(dmax, max(abs(a - b) for a, b in zip(rce, rkl)))
        lab = f"{steps[i-1]}->{steps[i]}"
        rows_ce.append([lab] + [f(x, 4, True) for x in rce]
                       + [f"l={rce.index(max(rce))}"])
        rows_kl.append([lab] + [f(x, 4, True) for x in rkl])
        rows_nll.append([lab] + [f(x, 4, True) for x in rnl]
                        + [f"l={rnl.index(max(rnl))}"])
    emit(f"max |(-Delta CE_l) - (-Delta KL_l)| over all intervals and levels: {dmax:.2e}\n")
    table(rows_ce, ["interval"] + [f"l={l}" for l in LVLS] + ["argmax"])
    emit("### Q1b2. `-Delta NLL_l` — the realised (endogenous) form of the same derivative\n")
    table(rows_nll, ["interval"] + [f"l={l}" for l in LVLS] + ["argmax"])

    emit("### Q1b3. The RELATIVE trend, `-Delta KL_l / KL_l(before)` — learning progress "
         "normalised by how much is left at that level\n")
    rows = []
    for i in range(1, len(steps)):
        r = [(ora[i - 1][l]["KL_pL_q"] - ora[i][l]["KL_pL_q"]) / ora[i - 1][l]["KL_pL_q"]
             for l in LVLS]
        rows.append([f"{steps[i-1]}->{steps[i]}"] + [f(x, 3, True) for x in r]
                    + [f"l={r.index(max(r))}", f(kap.get(steps[i]), 2)])
    table(rows, ["interval"] + [f"l={l}" for l in LVLS] + ["argmax", "kappa* after"])

    emit("## Q1c. The period detector\n")
    emit("Nested bottom-up estimate of the depth-k boundary offset from the model's own "
         "per-position entropy profile: accuracy = P(r_hat_k == true boundary column mod "
         "s^k). `field` = the fraction of positions whose induced level label equals the "
         "oracle's. Ceiling = observer k's own posterior over the phase mod s^k at window "
         "positions 32-64 (`altitude_phase_identifiability`, same Dirichlet grammar).\n")
    ph = (phase_id or {}).get("by_k", {})
    rows = []
    for i, st in enumerate(steps):
        d = panel[i]["detector"]
        rows.append([st, f(kap.get(st), 2)]
                    + [f(ik(d["template"]["acc_nested"])[k], 2) for k in range(1, 7)]
                    + [f(d["template"]["frac_exact_field"], 3),
                       f(d["meanprof"]["frac_exact_field"], 3)])
    rows.append(["chance", ""] + [f(1 / 2 ** k, 3) for k in range(1, 7)] + ["", ""])
    rows.append(["ceiling", ""] + [f(ph[str(k)]["acc_by_pos"]["32-65"], 3)
                                   if str(k) in ph else "-" for k in range(1, 7)] + ["", ""])
    table(rows, ["step", "kappa*"] + [f"k={k}" for k in range(1, 7)]
          + ["field(tmpl)", "field(meanprof)"])

    emit("### Q1c2. The model-free nested detector (`meanprof`: no template, entropy only)\n")
    table([[st] + [f(ik(panel[i]["detector"]["meanprof"]["acc_nested"])[k], 2)
                   for k in range(1, 7)] for i, st in enumerate(steps)],
          ["step"] + [f"k={k}" for k in range(1, 7)])

    emit("### Q1c3. Unconstrained per-k accuracy (altitude Q2 A's statistic, recomputed)\n")
    table([[st] + [f(ik(panel[i]["detector"]["acc_unconstrained_template"])[k], 2)
                   for k in range(1, 7)] for i, st in enumerate(steps)],
          ["step"] + [f"k={k}" for k in range(1, 7)])

    emit("## Q1d. Q3's candidate gain (read from `altitude_units.json`, not recomputed)\n")
    emit("Per-token gain a true unit still buys against the model as the null, "
         "`Delta / 2^j` (altitude Q3b2), beside the AUC(true vs chance) with the model as "
         "the null (Q3b).\n")
    if units:
        rows = []
        for r in units["rows"]:
            bj = r["by_j"]
            rows.append([r["step"], f(kap.get(r["step"]), 2)]
                        + [f(bj[str(j)]["delta_model"][0] / 2 ** j, 3, True)
                           for j in range(1, 6)]
                        + [f(bj[str(j)]["auc_delta_model"], 3) for j in range(1, 6)])
        table(rows, ["step", "kappa*"] + [f"gain j={j}" for j in range(1, 6)]
              + [f"AUC j={j}" for j in range(1, 6)])
    else:
        emit("(altitude_units.json not fetched)\n")

    emit("## Q1e. THE NEW ARM: the panel with ENDOGENOUS level labels\n")
    emit("Each position's level is assigned by the model's own nested entropy-period "
         "detector (`l_hat(j) = leaf_levels((j - r_hat_6) mod 64)`), not by the oracle. "
         "Level `l` is labelled correctly exactly when `r_hat_6` is right mod `2^l`.\n")
    for mode in ("template", "meanprof"):
        emit(f"### Q1e1-{mode}. Excess `CE_lhat - H(q)_lhat`\n")
        table([[st] + [f(endo[mode][i][l]["excess_oracle"], 4, True) if l in endo[mode][i]
                       else "-" for l in LVLS] for i, st in enumerate(steps)],
              ["step"] + [f"lhat={l}" for l in LVLS])
        emit(f"### Q1e2-{mode}. `-Delta CE_lhat` (the trend with endogenous labels)\n")
        rows, agree = [], []
        for i in range(1, len(steps)):
            rr = [endo[mode][i - 1][l]["CE"] - endo[mode][i][l]["CE"] for l in LVLS]
            gt = [ora[i - 1][l]["KL_pL_q"] - ora[i][l]["KL_pL_q"] for l in LVLS]
            agree.append(rr.index(max(rr)) == gt.index(max(gt)))
            rows.append([f"{steps[i-1]}->{steps[i]}"] + [f(x, 4, True) for x in rr]
                        + [f"l={rr.index(max(rr))}", f"l={gt.index(max(gt))}"])
        table(rows, ["interval"] + [f"lhat={l}" for l in LVLS]
              + ["argmax", "oracle argmax -dKL"])
        emit(f"argmax agreement with the oracle's rung: {sum(agree)}/{len(agree)} intervals\n")

    emit("### Q1e3. Confusion P(lhat = c | l = r), template detector, selected checkpoints\n")
    for i, st in enumerate(steps):
        if st not in (0, 2000, 8000, 24000, 64000):
            continue
        emit(f"step {st}:\n")
        C = panel[i]["confusion"]["template"]
        table([[f"l={r}"] + [f(C[r][c], 3) for c in LVLS] for r in LVLS],
              ["true \\ hat"] + [f"{c}" for c in LVLS])

    emit("### Q1e4. Per-level survival: |excess| and trend, oracle vs endogenous labels, "
         "at the last checkpoint and over the last interval\n")
    i = len(steps) - 1
    rows = []
    for l in LVLS:
        e_o = ora[i][l]["excess_oracle"]
        e_t = endo["template"][i].get(l, {}).get("excess_oracle")
        t_o = ora[i - 1][l]["CE"] - ora[i][l]["CE"]
        t_t = (endo["template"][i - 1][l]["CE"] - endo["template"][i][l]["CE"]
               if l in endo["template"][i] and l in endo["template"][i - 1] else None)
        C = panel[i]["confusion"]["template"]
        rows.append([f"l={l}", f(e_o, 4, True), f(e_t, 4, True), f(t_o, 4, True),
                     f(t_t, 4, True), f(C[l][l], 3)])
    table(rows, ["level", "excess(oracle)", "excess(endo)", "-dCE(oracle)", "-dCE(endo)",
                 "P(lhat=l|l)"])
    return steps, ora, endo, kap


# --------------------------------------------------------------------------- #
# Q2 -- the wall from the inside
# --------------------------------------------------------------------------- #

def q2(tag, summary):
    arms = {}
    for a in ARMS:
        j = load(f"{tag}_{a}.json")
        if j:
            arms[a] = j
    if not arms:
        emit("\n(no wall results yet)\n")
        return
    steps = [r["step"] for r in arms[ARMS[0]]["log"]]

    emit("\n# Q2 — the wall from the inside\n")
    emit("`reread/lm`'s arms re-run with the endogenous panel. The training loop, the "
         "model init, the optimiser, the batch sampler and every seed are `lm_reread`'s; "
         "the arms differ only in whether the corpus tensor is redrawn. Venue V = 2048 "
         "held-out fresh windows (the same for every arm and checkpoint) from "
         "`lm_reread`'s own held-out val corpus; venue C = 2048 windows from the arm's own "
         "corpus (for `fresh`, the pool currently loaded). Uniform synonyms, so the "
         "observer ladder differs from Q1's Dirichlet one.\n")

    emit("## Q2-gate. Fidelity against `lm0`\n")
    emit("Every quantity `lm0` logged that this node recomputes, at the checkpoints the "
         "two ladders share.\n")
    g = (summary or {}).get("fidelity_vs_lm0", {})
    table([[a, g[a].get("n_shared_ckpts"), f"{g[a].get('max_abs_d_val_nll', 0):.2e}",
            f"{g[a].get('max_abs_d_train_nll', 0):.2e}"
            if g[a].get("max_abs_d_train_nll") is not None else "-",
            f"{g[a].get('max_abs_d_excess_by_level', 0):.2e}"]
           for a in ARMS if a in g and "error" not in g[a]],
          ["arm", "shared ckpts", "max|d val_nll|", "max|d train_nll|", "max|d excess|"])

    emit("## Q2a. Loss and the memorisation gap\n")
    emit("`val` = `lm_reread.flat_nll` on the held-out corpus, `train` = the same on the "
         "arm's own corpus (both verbatim from `lm_reread`). `NLL_V`/`NLL_C` are the "
         "panel's realised losses on venues V and C.\n")
    for a in ARMS:
        if a not in arms:
            continue
        emit(f"### {a} ({arms[a]['corpus_tokens']:,} distinct tokens)\n")
        rows = []
        for r in arms[a]["log"]:
            V, C = r["venue_V"]["overall"], (r.get("venue_C") or {}).get("overall", {})
            rows.append([r["step"], f"{r['tokens']/1e6:.2f}",
                         f(r["epochs"], 1) if r["epochs"] else "inf",
                         f(r["val_nll"], 4), f(r["train_nll"], 4),
                         f(V["NLL"], 4), f(V["H_q"], 4), f(V["excess_realised"], 4, True),
                         f(C.get("NLL"), 4), f(C.get("H_q"), 4),
                         f(C.get("excess_realised"), 4, True),
                         f(V["NLL"] - C["NLL"], 4, True) if C else "-",
                         f(V["KL_pL_q"], 4)])
        table(rows, ["step", "Mtok", "epochs", "val", "train", "NLL_V", "Hq_V", "exc_V",
                     "NLL_C", "Hq_C", "exc_C", "gap", "KL(pL||q)_V"])

    for key, lab, venue in (("excess_realised", "excess `NLL_l - H(q)_l`", "V"),
                            ("excess_realised", "excess `NLL_l - H(q)_l`", "C"),
                            ("NLL", "realised loss `NLL_l`", "V"),
                            ("KL_pL_q", "oracle residual `KL(p_L||q)_l`", "V")):
        if venue == "C" and key == "KL_pL_q":
            continue
        emit(f"## Q2b-{venue}-{key}. Per leaf level: {lab}, venue {venue}\n")
        for a in ARMS:
            if a not in arms:
                continue
            vk = "venue_V" if venue == "V" else "venue_C"
            rows = []
            for r in arms[a]["log"]:
                if vk not in r:
                    continue
                bl = ik(r[vk]["by_level_oracle"])
                if key not in bl[0]:
                    continue
                rows.append([r["step"]] + [f(bl[l][key], 4, key == "excess_realised")
                                           for l in LVLS])
            if rows:
                emit(f"**{a}**\n")
                table(rows, ["step"] + [f"l={l}" for l in LVLS])

    emit("## Q2c. The memorisation gap per level, `NLL_V,l - NLL_C,l`\n")
    for a in ARMS:
        if a not in arms:
            continue
        rows = []
        for r in arms[a]["log"]:
            if "venue_C" not in r:
                continue
            bV, bC = ik(r["venue_V"]["by_level_oracle"]), ik(r["venue_C"]["by_level_oracle"])
            rows.append([r["step"]] + [f(bV[l]["NLL"] - bC[l]["NLL"], 4, True) for l in LVLS])
        if rows:
            emit(f"**{a}**\n")
            table(rows, ["step"] + [f"l={l}" for l in LVLS])

    emit("## Q2d. The loss trend per level on held-out data\n")
    emit("`-Delta CE_V,l` is exact (the target is an expectation under `p_L`, and on a "
         "fixed window set it equals `-Delta KL(p_L||q)_l` identically). `-Delta NLL_V,l` "
         "is the realised estimator of the same derivative, the form a learner without an "
         "oracle has; their difference is Monte-Carlo noise at 2048 windows.\n")
    for a in ARMS:
        if a not in arms:
            continue
        lg = arms[a]["log"]
        rows = []
        for i in range(1, len(lg)):
            b0 = ik(lg[i - 1]["venue_V"]["by_level_oracle"])
            b1 = ik(lg[i]["venue_V"]["by_level_oracle"])
            dce = [b0[l]["CE"] - b1[l]["CE"] for l in LVLS]
            dnl = [b0[l]["NLL"] - b1[l]["NLL"] for l in LVLS]
            rows.append([f"{lg[i-1]['step']}->{lg[i]['step']}"]
                        + [f(x, 4, True) for x in dce]
                        + [f(x, 4, True) for x in dnl]
                        + [f"l={dce.index(max(dce))}", f"l={dnl.index(max(dnl))}"])
        emit(f"**{a}**\n")
        table(rows, ["interval"] + [f"CE l={l}" for l in LVLS]
              + [f"NLL l={l}" for l in LVLS] + ["argmax CE", "argmax NLL"])
    emit("### Q2d2. How well the realised derivative estimates the exact one\n")
    emit("Over every (interval, level) pair of an arm: RMS of "
         "`(-Delta NLL_V,l) - (-Delta CE_V,l)`, and the sign-agreement rate.\n")
    rows = []
    for a in ARMS:
        if a not in arms:
            continue
        lg = arms[a]["log"]
        errs, agree = [], []
        for i in range(1, len(lg)):
            b0 = ik(lg[i - 1]["venue_V"]["by_level_oracle"])
            b1 = ik(lg[i]["venue_V"]["by_level_oracle"])
            for l in LVLS:
                dce = b0[l]["CE"] - b1[l]["CE"]
                dnl = b0[l]["NLL"] - b1[l]["NLL"]
                errs.append((dnl - dce) ** 2)
                agree.append((dce > 0) == (dnl > 0))
        rows.append([a, len(errs), f((sum(errs) / len(errs)) ** 0.5, 4),
                     f(sum(agree) / len(agree), 3)])
    table(rows, ["arm", "n pairs", "RMS(dNLL - dCE)", "sign agreement"])

    emit("## Q2d3. The same on the arm's OWN corpus, `-Delta NLL_C,l`\n")
    for a in ARMS:
        if a not in arms or "venue_C" not in arms[a]["log"][-1]:
            continue
        lg = [r for r in arms[a]["log"] if "venue_C" in r]
        rows = []
        for i in range(1, len(lg)):
            b0 = ik(lg[i - 1]["venue_C"]["by_level_oracle"])
            b1 = ik(lg[i]["venue_C"]["by_level_oracle"])
            d = [b0[l]["NLL"] - b1[l]["NLL"] for l in LVLS]
            rows.append([f"{lg[i-1]['step']}->{lg[i]['step']}"] + [f(x, 4, True) for x in d])
        emit(f"**{a}**\n")
        table(rows, ["interval"] + [f"l={l}" for l in LVLS])

    emit("## Q2e. The period detector per arm, venue V (nested, template)\n")
    for a in ARMS:
        if a not in arms:
            continue
        rows = []
        for r in arms[a]["log"]:
            d = r["venue_V"]["detector"]
            rows.append([r["step"]] + [f(ik(d["template"]["acc_nested"])[k], 2)
                                       for k in range(1, 7)]
                        + [f(d["template"]["frac_exact_field"], 3)])
        emit(f"**{a}**\n")
        table(rows, ["step"] + [f"k={k}" for k in range(1, 7)] + ["field"])

    emit("## Q2e2. The period detector on the arm's OWN corpus, venue C\n")
    for a in ARMS:
        if a not in arms or "venue_C" not in arms[a]["log"][-1]:
            continue
        rows = []
        for r in arms[a]["log"]:
            if "venue_C" not in r:
                continue
            d = r["venue_C"]["detector"]
            rows.append([r["step"]] + [f(ik(d["template"]["acc_nested"])[k], 2)
                                       for k in range(1, 7)])
        emit(f"**{a}**\n")
        table(rows, ["step"] + [f"k={k}" for k in range(1, 7)])

    emit("## Q2f. When each reading turns, per level\n")
    emit("Crossings are PERSISTENT (the condition holds at that checkpoint and at every "
         "later one) and restricted to steps >= 250; `first turn` is the first "
         "non-persistent crossing, for comparison. Floors are the arc's own: the step-0 "
         "checkpoint. `stall(l)` = the first checkpoint from which the exact held-out "
         "derivative `-Delta CE_V,l` stays <= 0 — the level has gone quiet to a "
         "derivative. `exc>0(l)` = the first checkpoint from which the held-out excess "
         "`NLL_V,l - H(q)_V,l` persistently exceeds its OWN step-0 value (the random-init "
         "floor, +0.09 to +0.12 nats at every level). `V-C>.05(l)` = the same for the "
         "held-out MINUS own-corpus excess, `exc_V,l - exc_C,l` exceeding 0.05 nats, a "
         "quantity a learner computes from its own stream alone (its step-0 value is "
         "0.000 +- 0.002). `gap>0.05(l)` = the same for the per-level "
         "memorisation gap exceeding 0.05 nats (its step-0 value is 0.000 +- 0.002). "
         "`-` = never, within the ladder.\n")

    def first_persistent(steps_, cond, lo=250):
        idx = [i for i, st in enumerate(steps_) if st >= lo]
        for i in idx:
            if all(cond[k] for k in idx if k >= i):
                return steps_[i]
        return "-"

    def first_any(steps_, cond, lo=250):
        for i, st in enumerate(steps_):
            if st >= lo and cond[i]:
                return st
        return "-"

    rows = []
    for a in ARMS:
        if a not in arms:
            continue
        lg = arms[a]["log"]
        sts = [r["step"] for r in lg]
        for l in LVLS:
            bV = [ik(r["venue_V"]["by_level_oracle"])[l] for r in lg]
            exc = [b["excess_realised"] for b in bV]
            excC = [ik(r["venue_C"]["by_level_oracle"])[l]["excess_realised"]
                    if "venue_C" in r else None for r in lg]
            gapc = [(bV[i]["NLL"] - ik(lg[i]["venue_C"]["by_level_oracle"])[l]["NLL"])
                    if "venue_C" in lg[i] else None for i in range(len(lg))]
            dce = [None] + [bV[i - 1]["CE"] - bV[i]["CE"] for i in range(1, len(lg))]
            cond_stall = [d is not None and d <= 0 for d in dce]
            vc = [exc[i] - excC[i] if excC[i] is not None else None for i in range(len(lg))]
            rows.append([a, f"l={l}",
                         first_any(sts, cond_stall), first_persistent(sts, cond_stall),
                         first_persistent(sts, [exc[i] > exc[0] for i in range(len(lg))]),
                         first_persistent(sts, [vc[i] is not None and vc[i] > 0.05
                                                for i in range(len(lg))]),
                         first_persistent(sts, [gapc[i] is not None and gapc[i] > 0.05
                                                for i in range(len(lg))]),
                         f(exc[0], 4, True), f(exc[-1], 4, True),
                         f(vc[-1], 4, True) if vc[-1] is not None else "-",
                         f(gapc[-1], 4, True) if gapc[-1] is not None else "-"])
    table(rows, ["arm", "level", "first turn", "stall(l)", "exc>0(l)", "V-C>.05(l)",
                 "gap>0.05(l)", "exc_V step 0", "final exc_V", "final exc_V-exc_C",
                 "final gap"])

    emit("## Q2f2. When the period detector turns, per k\n")
    emit("`peak` = the checkpoint at which the nested detector's accuracy at period "
         "`s^k` is highest on held-out windows; `final` its value at 20000; `drop` the "
         "fall from peak to final. A level that was absorbed keeps its period; a level "
         "lost to the wall gives it back.\n")
    rows = []
    for a in ARMS:
        if a not in arms:
            continue
        lg = arms[a]["log"]
        for k in range(1, 7):
            acc = [ik(r["venue_V"]["detector"]["template"]["acc_nested"])[k] for r in lg]
            i = acc.index(max(acc))
            rows.append([a, f"k={k}", lg[i]["step"], f(acc[i], 3), f(acc[-1], 3),
                         f(acc[i] - acc[-1], 3, True)])
    table(rows, ["arm", "period", "peak at", "peak acc", "final acc", "drop"])

    emit("## Q2g. The three endogenous readings side by side, at the level each arm "
         "was climbing when it stalled\n")
    emit("Level 2 is the level `-Delta CE_V,l` peaks at in every arm (as in Q1b); the "
         "rows are the whole ladder for that level. `trend` = `-Delta CE_V,2` over the "
         "interval ending at that step (exact); `NLL-trend` its realised estimator; "
         "`excess` = `NLL_V,2 - H(q)_V,2`; `gap` = `NLL_V,2 - NLL_C,2`; `k=3 acc` the "
         "period detector one rung above.\n")
    L2 = 2
    for a in ARMS:
        if a not in arms:
            continue
        lg = arms[a]["log"]
        rows = []
        for i, r in enumerate(lg):
            bV = ik(r["venue_V"]["by_level_oracle"])[L2]
            bC = ik(r["venue_C"]["by_level_oracle"])[L2] if "venue_C" in r else None
            p0 = ik(lg[i - 1]["venue_V"]["by_level_oracle"])[L2] if i else None
            rows.append([r["step"],
                         f(p0["CE"] - bV["CE"], 4, True) if p0 else "-",
                         f(p0["NLL"] - bV["NLL"], 4, True) if p0 else "-",
                         f(bV["excess_realised"], 4, True),
                         f(bV["NLL"] - bC["NLL"], 4, True) if bC else "-",
                         f(ik(r["venue_V"]["detector"]["template"]["acc_nested"])[3], 3),
                         f(bV["KL_pL_q"], 4)])
        emit(f"**{a}**\n")
        table(rows, ["step", "trend(exact)", "NLL-trend", "excess", "gap", "k=3 acc",
                     "KL(pL||q)_2"])


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, panel, phase_id, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(outdir, exist_ok=True)

    steps = np.array([max(r["step"], 100) for r in panel])
    ora = [ik(r["by_level_oracle"]) for r in panel]

    # fig1: Q1 -- the panel with oracle labels
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for l in LVLS:
        c = plt.cm.viridis(l / 6)
        ax[0].plot(steps, [o[l]["excess_oracle"] for o in ora], "o-", ms=3, color=c,
                   label=f"l={l}")
        ax[1].plot(steps, [o[l]["KL_pL_q"] for o in ora], "o-", ms=3, color=c)
        d = [ora[i - 1][l]["CE"] - ora[i][l]["CE"] for i in range(1, len(ora))]
        ax[2].plot(steps[1:], d, "o-", ms=3, color=c)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xscale="log", xlabel="step", ylabel="CE_l - H(q)_l  (nats)",
              title="(a) the excess by leaf level")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(xscale="log", yscale="log", xlabel="step", ylabel="KL(p_L||q)_l",
              title="ground truth: the oracle residual")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xscale="log", xlabel="step", ylabel="-Delta CE_l",
              title="(b) the loss trend (learning progress)")
    ax[2].set_yscale("symlog", linthresh=1e-3)
    fig.tight_layout(); fig.savefig(f"{outdir}/q1_panel.png", dpi=140); plt.close(fig)

    # fig2: Q1 -- the detector and the endogenous panel
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ph = (phase_id or {}).get("by_k", {})
    for k in range(1, 7):
        c = plt.cm.plasma((k - 1) / 5)
        ax[0].plot(steps, [ik(r["detector"]["template"]["acc_nested"])[k] for r in panel],
                   "o-", ms=3, color=c, label=f"k={k}")
        if str(k) in ph:
            ax[0].axhline(ph[str(k)]["acc_by_pos"]["32-65"], color=c, ls=":", lw=0.8)
        ax[0].axhline(1 / 2 ** k, color=c, ls="--", lw=0.6)
    ax[0].set(xscale="log", xlabel="step", ylabel="P(r_hat_k correct)",
              title="(c) nested period detector\n(dotted = observer ceiling, dashed = chance)")
    ax[0].legend(fontsize=7, ncol=2)
    for mode, ls in (("template", "-"), ("meanprof", "--")):
        ax[1].plot(steps, [r["detector"][mode]["frac_exact_field"] for r in panel],
                   "o" + ls, ms=3, label=mode)
    ax[1].set(xscale="log", xlabel="step", ylabel="fraction of positions",
              title="endogenous level field == oracle's")
    ax[1].legend(fontsize=8)
    endo = [ik(r["by_level_endo"]["template"]) for r in panel]
    for l in LVLS:
        c = plt.cm.viridis(l / 6)
        ax[2].plot(steps, [o[l]["excess_oracle"] for o in ora], "-", lw=1.0, color=c)
        ax[2].plot(steps, [e[l]["excess_oracle"] if l in e else np.nan for e in endo],
                   "o--", ms=3, color=c, label=f"l={l}")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xscale="log", xlabel="step", ylabel="excess (nats)",
              title="(e) excess: oracle labels (solid) vs endogenous (dashed)")
    ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/q1_endogenous.png", dpi=140); plt.close(fig)

    # fig3 + fig4: Q2
    arms = {a: load(f"{tag}_{a}.json") for a in ARMS}
    arms = {a: j for a, j in arms.items() if j}
    if not arms:
        return
    cols = {"fresh": "#1b7837", "frozen_200000": "#4393c3", "frozen_16384": "#d6604d",
            "frozen_2048": "#762a83"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, j in arms.items():
        st = np.array([max(r["step"], 100) for r in j["log"]])
        ax[0].plot(st, [r["val_nll"] for r in j["log"]], "o-", ms=3, color=cols[a], label=a)
        ax[0].plot(st, [r["train_nll"] for r in j["log"]], "--", lw=1, color=cols[a])
        ax[1].plot(st, [r["venue_V"]["overall"]["excess_realised"] for r in j["log"]],
                   "o-", ms=3, color=cols[a], label=a)
        ax[1].plot(st, [(r.get("venue_C") or {"overall": {}})["overall"]
                        .get("excess_realised", np.nan) for r in j["log"]],
                   "--", lw=1, color=cols[a])
        ax[2].plot(st, [r["venue_V"]["overall"]["KL_pL_q"] for r in j["log"]],
                   "o-", ms=3, color=cols[a], label=a)
    ax[0].set(xscale="log", xlabel="step", ylabel="NLL",
              title="val (solid) and own-corpus (dashed) loss")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xscale="log", xlabel="step", ylabel="NLL - H(q)",
              title="the excess: venue V (solid), venue C (dashed)")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[2].set(xscale="log", yscale="log", xlabel="step", ylabel="KL(p_L||q) on V",
              title="the oracle residual")
    fig.tight_layout(); fig.savefig(f"{outdir}/q2_wall.png", dpi=140); plt.close(fig)

    n = len(arms)
    fig, ax = plt.subplots(3, n, figsize=(4.4 * n, 11), squeeze=False, sharey="row")
    for ai, (a, j) in enumerate(arms.items()):
        st = np.array([max(r["step"], 100) for r in j["log"]])
        for l in LVLS:
            c = plt.cm.viridis(l / 6)
            bl = [ik(r["venue_V"]["by_level_oracle"])[l] for r in j["log"]]
            ax[0][ai].plot(st, [b["excess_realised"] for b in bl], "o-", ms=2.5, color=c,
                           label=f"l={l}")
            d = [bl[i - 1]["NLL"] - bl[i]["NLL"] for i in range(1, len(bl))]
            ax[1][ai].plot(st[1:], d, "o-", ms=2.5, color=c)
            if "venue_C" in j["log"][-1]:
                bc = [ik(r["venue_C"]["by_level_oracle"])[l] for r in j["log"]
                      if "venue_C" in r]
                sc = np.array([max(r["step"], 100) for r in j["log"] if "venue_C" in r])
                ax[2][ai].plot(sc, [bl[i]["NLL"] - bc[i]["NLL"] for i in range(len(bc))],
                               "o-", ms=2.5, color=c)
        for row, ttl, yl in ((0, "held-out excess by level", "NLL_V,l - H(q)_V,l"),
                             (1, "held-out loss trend by level", "-Delta NLL_V,l"),
                             (2, "memorisation gap by level", "NLL_V,l - NLL_C,l")):
            ax[row][ai].axhline(0, color="k", lw=0.6)
            ax[row][ai].set(xscale="log", xlabel="step",
                            title=f"{a}\n{ttl}" if row == 0 else ttl,
                            ylabel=yl if ai == 0 else "")
            ax[row][ai].set_yscale("symlog", linthresh=1e-3)
        ax[0][ai].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/q2_by_level.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    panel = load("frontier_panel.json")
    if panel:
        q1(panel, load("altitude_identity.json"), load("altitude_units.json"),
           load("altitude_residual.json"), load("altitude_frontier.json"),
           load("altitude_phase_identifiability.json"))
    q2(a.tag, load(f"{a.tag}_summary.json"))
    with open(os.path.join(HERE, "results", "tables.md"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    if a.figs and panel:
        figures(a.tag, panel, load("altitude_phase_identifiability.json"),
                os.path.join(HERE, "figs"))
    print(f"\n-> {os.path.join(HERE, 'results', 'tables.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
