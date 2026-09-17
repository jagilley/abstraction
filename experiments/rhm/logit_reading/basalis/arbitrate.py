"""basalis -- the hold-and-discount consumer, what it is worth in each world, and whether
anything readable can arbitrate between the two policies.

`altitude/` Q4b: after an off-grammar token the model falls back to the eps-noise observer at
`k* - 1` and sharpens. `coeruleus/`: that is Bayes-right for a structural edit (the swapped
subtree regrows legally, so the continuation follows the one-rung-down reading) and Bayes-wrong
for a glitch (where the old structure does continue and a noise-aware observer should hold its
reading and discount the token). Nothing in the model or in either observer family arbitrates.

This round builds the second policy in the model's own terms -- a mixing weight `w` between the
native forecast and the forecast with the flagged token imputed from the model's own prior
(`hold.py`) -- and asks:

  Q1  which imputation, and what the held forecast looks like
  Q2  the per-world, per-offset oracle `w`; what the right policy buys and the wrong one costs,
      in realised NLL and (on the eps venue, where the exact ladder is on file) against the
      observer-family ceiling `coeruleus/` Q1 used; the lookback bound on where the culprit is;
      the parent's 2e entropy sign and the persistence profile under each policy
  Q3  a mixed world at stated edit / glitch / false-alarm rates, where arbitration is
      load-bearing, with the rate grid
  Q4  gates: the banked excess-surprise head, a supervised world probe on the state (the upper
      bound on how fast the state separates the worlds, as a function of tau), the model's own
      entropy, the running excess since the event, the model's own log Bayes factor between the
      two hypotheses, and -- beside every learned gate -- a shuffled gate and the exact label,
      the `endogenous_teacher` control that makes a null attributable.

Every fitted gate -> w map is fit on the model's own realised NLL on a FIT split, with no
oracle in it, and evaluated on a held-out TEST split (`coeruleus/gainloop.py`'s shape).

Run:
  modal run -m rhm.logit_reading.basalis.arbitrate::arbitrate_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_eps01_s42/step064000.pt \
      --n-events 400 --n-ladder 512 --max-pair-windows 3000 --tag smoke
  modal run --detach -m rhm.logit_reading.basalis.arbitrate::arbitrate_sweep
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import auc, fit_probe
from rhm.logit_reading.phasic import paired_win
from rhm.logit_reading.coeruleus.events import (
    load_ladder, convex_obs, fit_kappa_pooled, mix_opt, mix_ce)
from rhm.logit_reading.coeruleus.readout import load_head
from rhm.logit_reading.basalis import worlds
from rhm.logit_reading.basalis.hold import (
    logq_full, gather_offsets, states_offsets, impute_logq, prep_imputed, hold_forecast,
    log_bayes_factor, mix_logq, realised_ab, nll_at_w, w_opt, fit_wmap, ent, xent)

VARIANTS = ("argmax", "top2", "top4", "full", "pf")
GATES = ("head_ridge", "head_mlp", "Hq", "runex", "logbf", "sevent", "probe")
GRID_GATES = ("probe", "logbf", "runex", "head_ridge", "shuf_probe", "label")
ARMS = ("edit", "glitch_m", "glitch_s", "quiet")
GLITCHES = ("glitch_s", "glitch_m")


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

def arm_record(model, win, pos, H, blocks, variant, device="cuda", bs=512):
    """Everything about one event population that the reductions need."""
    n = len(win)
    ar = np.arange(n)
    v = model.transformer.wte.weight.shape[0]
    taus = np.arange(H + 1)
    lq = logq_full(model, win, device, bs)                                   # (n, T, v)
    T = lq.shape[1]
    tok_all = win[:, 1:T + 1]
    nll_all = -np.take_along_axis(lq, tok_all[..., None], -1)[..., 0]
    Hq_all = ent(np.exp(lq))
    cum_ex = np.concatenate([np.zeros((n, 1)), np.cumsum(nll_all - Hq_all, 1)], 1)

    logq_nat = gather_offsets(lq, pos, taus).astype(np.float64)              # (n, H+1, v)
    tok = win[ar[:, None], (pos[:, None] + taus[None, :] + 1)]
    s_event = nll_all[ar, pos - 1]
    H_prev = Hq_all[ar, pos - 1]
    log_prior = lq[ar, pos - 1, :]

    P = prep_imputed(impute_logq(model, win, pos, pos,
                                 np.tile(np.arange(v), (n, 1)), H, device, bs),
                     log_prior, tok)
    rec = {"n": n, "pos": pos, "tok": tok, "s_event": s_event, "H_prev": H_prev,
           "logq_nat": logq_nat.astype(np.float32), "H_nat": ent(np.exp(logq_nat)),
           "nll_nat": -np.log(np.clip(np.take_along_axis(
               np.exp(logq_nat), tok[..., None], -1)[..., 0], 1e-300, None)),
           "top1_mass": float(np.exp(log_prior).max(1).mean()),
           "top4_mass": float(np.sort(np.exp(log_prior), 1)[:, -4:].sum(1).mean()),
           "hold": {}}
    for var in VARIANTS:
        q_h = hold_forecast(P, var)
        a, b = realised_ab(logq_nat, q_h, tok)
        rec["hold"][var] = {"a": a, "b": b, "H": ent(q_h)}
        if var == variant:
            rec["q_hold"] = q_h.astype(np.float32)
    rec["gates"] = {
        "Hq": rec["H_nat"],
        "sevent": np.repeat(s_event[:, None], H + 1, 1),
        "logbf": log_bayes_factor(P, logq_nat, tok, s_event),
        # running excess since the event, causal: the predictions made at pos-1 .. pos+tau-1,
        # all of which have been scored by the time the forecast at pos+tau is made.
        "runex": (cum_ex[ar[:, None], (pos[:, None] + taus[None, :])]
                  - cum_ex[ar, pos - 1][:, None]),
    }
    rec["log_prior"] = log_prior
    rec["states"] = states_offsets(model, win, pos, taus, blocks, device)
    del P, lq, nll_all, Hq_all, cum_ex
    return rec


def _tau_table(rec, variant, sel, H):
    a, b = rec["hold"][variant]["a"], rec["hold"][variant]["b"]
    s = slice(None) if sel is None else sel
    rows = []
    for t in range(H + 1):
        aa, bb = a[s, t], b[s, t]
        if len(aa) < 10:
            continue
        w = w_opt(aa, bb)
        rows.append({"tau": t, "n": int(len(aa)),
                     "nll_native": float(nll_at_w(aa, bb, 0.0).mean()),
                     "nll_hold": float(nll_at_w(aa, bb, 1.0).mean()),
                     "w_star": w, "nll_at_wstar": float(nll_at_w(aa, bb, w).mean()),
                     "frac_hold_better": float((bb > aa).mean()),
                     "H_native": float(rec["H_nat"][s, t].mean()),
                     "H_hold": float(rec["hold"][variant]["H"][s, t].mean())})
    return rows


def _pooled(a, b, w, sw=None):
    x = nll_at_w(a, b, w)
    if sw is None:
        return float(x.mean())
    W = np.broadcast_to(np.asarray(sw, np.float64), x.shape)
    return float((x * W).sum() / W.sum())


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=16384,
              max_containers=4)
def arbitrate_ckpt(ckpt: str, head_tag: str = "", tag: str = "", stim_tag: str = "swap65k",
                   horizon: int = 8, n_events: int = 6000, eps_data: float = 0.01,
                   n_ladder: int = 4096, nat_max_events: int = 0, variant: str = "pf",
                   seed: int = 0, t_lo: int = 16,
                   caliper: float = 0.3, max_pair_windows: int = 20000,
                   deltas: str = "0,1,2,4,8", lookback_k: int = 4, do_pairs: bool = True,
                   do_natural: bool = True, probe_steps: int = 500, probe_hidden: int = 256):
    import resource
    import torch
    from rhm.model import GPT
    volume.reload()
    H = horizon
    rng = np.random.default_rng(seed)
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    v, T = cfg["v"], cfg["s"] ** cfg["L"]
    hp = f"{ckpt[:-3]}_coeruleus_head_state_excess_%s{('_' + head_tag) if head_tag else ''}.pt"
    heads = {f"head_{k}": load_head(hp % k, "cuda")[0] for k in ("ridge", "mlp")}
    blk = heads["head_ridge"].block
    torch.manual_seed(12345)
    rand = GPT(v, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to("cuda").eval()
    res = {"config": {"ckpt": ckpt, "step": cfg.get("step"),
                      "eps_train": cfg.get("eps_train", 0.0), "stim_tag": stim_tag, "H": H,
                      "n_events": n_events, "eps_data": eps_data, "seed": seed, "t_lo": t_lo,
                      "block": blk, "head_tag": head_tag, "variant": variant,
                      "variants": list(VARIANTS), "caliper": caliper}}
    print(f"step {cfg.get('step')} eps_train {cfg.get('eps_train', 0.0)} head block {blk}",
          flush=True)

    # ---------------- the two matched worlds and the false-alarm background ----
    S = worlds.load_swap_stimuli(stim_tag, data_dir=DATA_DIR)
    spec, meta = worlds.build_arms(S, H, n_events, seed=seed, t_lo=t_lo, v=v)
    del S
    n_ev = meta["n_events"]
    res["stimuli"] = {"n_eligible": meta["n_eligible"], "n_events": n_ev,
                      "kstar_hist": np.bincount(meta["kstar"], minlength=7).tolist(),
                      "j_hist": np.bincount(meta["j"], minlength=5).tolist(),
                      "delay_mean": float(meta["delay"].mean()),
                      "tv_beyond_region_frac": float(meta["tv_beyond_region"].mean())}
    print(f"events {n_ev} of {meta['n_eligible']} eligible; k* {res['stimuli']['kstar_hist']} "
          f"j {res['stimuli']['j_hist']}", flush=True)

    blocks = [blk, "post_embed"]
    R = {}

    def run_arm(nm):
        R[nm] = arm_record(model, spec[nm]["win"], spec[nm]["pos"], H, blocks, variant)
        R[nm]["states"]["rand"] = states_offsets(rand, spec[nm]["win"], spec[nm]["pos"],
                                                 np.arange(H + 1), blk)[blk]
        for k, h in heads.items():
            X = R[nm]["states"][h.block]
            R[nm]["gates"][k] = h(X.reshape(-1, X.shape[-1])).reshape(R[nm]["n"], H + 1)
        print(f"  arm {nm:9s} n={R[nm]['n']}  s_event {R[nm]['s_event'].mean():.3f}  "
              f"nll_nat {R[nm]['nll_nat'].mean():.3f}", flush=True)

    for nm in ("edit", "glitch_m", "quiet"):
        run_arm(nm)
    # `glitch_m`'s corrupting token is a uniform draw, so it is more surprising than the edit's
    # violator (which is original material read under an edited prefix) and the flagged token
    # alone would separate the worlds. `glitch_s` fixes that by construction: the same original
    # window, the same position, and the token whose surprisal under the (identical) original
    # prefix is nearest the edit's violator's -- a plausible-but-wrong token rather than noise.
    # The model is causal, so the forecast at pos-1 is the same whatever sits at pos.
    lp_orig = R["glitch_m"]["log_prior"]
    ar_ = np.arange(n_ev)
    tgt = R["edit"]["s_event"]
    dist = np.abs(-lp_orig - tgt[:, None])
    dist[ar_, meta["orig_token"]] = np.inf
    pick = dist.argmin(1)
    Ws = np.array(spec["glitch_m"]["win"])
    Ws[ar_, spec["glitch_m"]["pos"]] = pick
    spec["glitch_s"] = {"win": Ws, "pos": spec["glitch_m"]["pos"].copy(),
                        "src": spec["glitch_m"]["src"]}
    meta["glitch_s_token"] = pick
    run_arm("glitch_s")
    res["stimuli"]["s_event_by_arm"] = {nm: float(R[nm]["s_event"].mean()) for nm in R}
    res["stimuli"]["s_event_match_paired"] = paired_win(R["edit"]["s_event"],
                                                        R["glitch_s"]["s_event"])
    res["stimuli"]["s_event_match_maxdiff"] = float(
        np.abs(R["edit"]["s_event"] - R["glitch_s"]["s_event"]).max())

    # ---------------- Q1: which imputation ---------------------------------
    res["imputation"] = {"top1_mass": {nm: R[nm]["top1_mass"] for nm in R},
                         "top4_mass": {nm: R[nm]["top4_mass"] for nm in R},
                         "by_variant": {}}
    for var in VARIANTS:
        res["imputation"]["by_variant"][var] = {nm: {
            "nll_hold_pooled": float(nll_at_w(R[nm]["hold"][var]["a"],
                                              R[nm]["hold"][var]["b"], 1.0).mean()),
            "nll_hold_tau0": float(nll_at_w(R[nm]["hold"][var]["a"][:, 0],
                                            R[nm]["hold"][var]["b"][:, 0], 1.0).mean()),
            "H_hold_tau0": float(R[nm]["hold"][var]["H"][:, 0].mean()),
            "w_star_tau0": w_opt(R[nm]["hold"][var]["a"][:, 0], R[nm]["hold"][var]["b"][:, 0]),
            "w_star_pooled": w_opt(R[nm]["hold"][var]["a"], R[nm]["hold"][var]["b"])}
            for nm in R}
        print(f"  impute {var:7s} " + "  ".join(
            f"{nm} nll(w=1) {res['imputation']['by_variant'][var][nm]['nll_hold_pooled']:.4f} "
            f"w*0 {res['imputation']['by_variant'][var][nm]['w_star_tau0']:.3f}" for nm in R),
            flush=True)

    # ---------------- Q2: the prize per world, per offset -------------------
    res["by_world"] = {}
    for nm in R:
        d = {"n": R[nm]["n"], "all": _tau_table(R[nm], variant, None, H),
             "s_event_mean": float(R[nm]["s_event"].mean())}
        if nm != "quiet":
            d["by_kstar"] = {int(k): _tau_table(R[nm], variant, meta["kstar"] == k, H)
                             for k in range(1, 7) if (meta["kstar"] == k).sum() >= 100}
            d["by_j"] = {int(jj): _tau_table(R[nm], variant, meta["j"] == jj, H)
                         for jj in range(5) if (meta["j"] == jj).sum() >= 100}
            d["structural"] = _tau_table(R[nm], variant, meta["j"] >= 1, H)
            d["point"] = _tau_table(R[nm], variant, meta["j"] == 0, H)
        res["by_world"][nm] = d
        r0 = d["all"][0]
        print(f"  world {nm:9s} tau0 nll nat {r0['nll_native']:.4f} hold {r0['nll_hold']:.4f} "
              f"w* {r0['w_star']:.3f} -> {r0['nll_at_wstar']:.4f} | H nat {r0['H_native']:.3f} "
              f"hold {r0['H_hold']:.3f}", flush=True)

    pol = {"native": np.zeros(H + 1), "hold": np.ones(H + 1)}
    for nm in R:
        pol[f"wstar_{nm}"] = np.array([r["w_star"] for r in res["by_world"][nm]["all"]])
    res["policies"] = {p: wv.round(4).tolist() for p, wv in pol.items()}
    res["cross_world"] = {p: {nm: {
        "nll_pooled": _pooled(R[nm]["hold"][variant]["a"], R[nm]["hold"][variant]["b"],
                              wv[None, :]),
        "by_tau": [float(nll_at_w(R[nm]["hold"][variant]["a"][:, t],
                                  R[nm]["hold"][variant]["b"][:, t], wv[t]).mean())
                   for t in range(H + 1)]} for nm in R} for p, wv in pol.items()}

    # ---------------- the lookback bound: where is the culprit --------------
    res["lookback"] = {}
    dl = [int(x) for x in deltas.split(",") if x]
    for nm in ("edit",) + GLITCHES:
        sp = spec[nm]
        lq = logq_full(model, sp["win"], "cuda")
        ar = np.arange(len(sp["win"]))
        taus = np.arange(H + 1)
        tok = sp["win"][ar[:, None], (sp["pos"][:, None] + taus[None, :] + 1)]
        lqn = gather_offsets(lq, sp["pos"], taus).astype(np.float64)
        out = {}
        specs = [(f"delta{d_}", np.full(len(ar), d_)) for d_ in dl]
        specs.append(("delta_oracle", np.clip(meta["tv"] - meta["e"], 0, sp["pos"] - 1)))
        for lname, dvec in specs:
            p2 = np.maximum(sp["pos"] - dvec, 1)
            lp = lq[ar, p2 - 1, :]
            cand = np.argsort(lp, 1)[:, -lookback_k:]
            Pk = prep_imputed(impute_logq(model, sp["win"], p2, sp["pos"], cand, H, "cuda"),
                              np.take_along_axis(lp, cand, 1), tok)
            q_h = hold_forecast(Pk, "full")
            a, b = realised_ab(lqn, q_h, tok)
            out[lname] = {"delta_mean": float(dvec.mean()), "by_tau": [
                {"tau": t, "w_star": w_opt(a[:, t], b[:, t]),
                 "nll_native": float(nll_at_w(a[:, t], b[:, t], 0.).mean()),
                 "nll_hold": float(nll_at_w(a[:, t], b[:, t], 1.).mean()),
                 "nll_at_wstar": float(nll_at_w(a[:, t], b[:, t],
                                                w_opt(a[:, t], b[:, t])).mean())}
                for t in range(H + 1)]}
            if lname == "delta_oracle":
                out[lname]["by_j"] = {int(jj): [
                    {"tau": t, "n": int((meta["j"] == jj).sum()),
                     "w_star": w_opt(a[meta["j"] == jj, t], b[meta["j"] == jj, t]),
                     "nll_native": float(nll_at_w(a[meta["j"] == jj, t],
                                                  b[meta["j"] == jj, t], 0.).mean()),
                     "nll_at_wstar": float(nll_at_w(
                         a[meta["j"] == jj, t], b[meta["j"] == jj, t],
                         w_opt(a[meta["j"] == jj, t], b[meta["j"] == jj, t])).mean())}
                    for t in (0, 1, 2)]
                    for jj in range(5) if (meta["j"] == jj).sum() >= 100}
            del Pk, q_h
            print(f"  lookback {nm:9s} {lname:13s} tau0 w* {out[lname]['by_tau'][0]['w_star']:.3f}"
                  f"  gain {out[lname]['by_tau'][0]['nll_native'] - out[lname]['by_tau'][0]['nll_at_wstar']:+.4f}",
                  flush=True)
        res["lookback"][nm] = out
        del lq

    # ---------------- Q4: gates -------------------------------------------
    split = rng.random(n_ev) < 0.6
    ntr, nte = int(split.sum()), int((~split).sum())
    res["gates"] = {"n_fit": ntr, "n_test": nte, "separation": {}}
    yte = np.concatenate([np.ones(nte, bool), np.zeros(nte, bool)])
    for gl in GLITCHES:
        sep = {}
        for label, key, hid in (("probe", blk, probe_hidden), ("probe_lin", blk, 0),
                                ("probe_embed", "post_embed", 0), ("probe_rand", "rand", 0)):
            a_eg, a_eq, sc = [], [], {nm: np.zeros((n_ev, H + 1)) for nm in R}
            for t in range(H + 1):
                Xe, Xg = R["edit"]["states"][key][:, t], R[gl]["states"][key][:, t]
                f = fit_probe(np.concatenate([Xe[split], Xg[split]]),
                              np.concatenate([np.ones(ntr), np.zeros(ntr)]),
                              hidden=hid, steps=probe_steps)
                for nm in R:
                    sc[nm][:, t] = f(R[nm]["states"][key][:, t])
                a_eg.append(auc(np.concatenate([sc["edit"][~split, t], sc[gl][~split, t]]), yte))
                a_eq.append(auc(np.concatenate([sc["edit"][~split, t],
                                                sc["quiet"][~split, t]]), yte))
            sep[label] = {"auc_edit_vs_glitch_by_tau": a_eg, "auc_edit_vs_quiet_by_tau": a_eq}
            if label == "probe":
                for nm in R:
                    R[nm]["gates"][f"probe_{gl}"] = sc[nm]
            print(f"  [{gl}] probe {label:12s} edit-vs-glitch "
                  f"{[round(x, 3) for x in a_eg]}", flush=True)
        for g in GATES:
            if g == "probe":
                continue
            ge, gg, gq = R["edit"]["gates"][g], R[gl]["gates"][g], R["quiet"]["gates"][g]
            sep[g] = {
                "auc_edit_vs_glitch_by_tau": [auc(np.concatenate([ge[~split, t],
                                                                  gg[~split, t]]), yte)
                                              for t in range(H + 1)],
                "auc_edit_vs_quiet_by_tau": [auc(np.concatenate([ge[~split, t],
                                                                 gq[~split, t]]), yte)
                                             for t in range(H + 1)],
                "mean_edit": ge.mean(0).round(4).tolist(),
                "mean_glitch": gg.mean(0).round(4).tolist(),
                "mean_quiet": gq.mean(0).round(4).tolist()}
            print(f"  [{gl}] gate  {g:12s} edit-vs-glitch "
                  f"{[round(x, 3) for x in sep[g]['auc_edit_vs_glitch_by_tau']]}", flush=True)
        res["gates"]["separation"][gl] = sep
    # `glitch_s` is matched on the flagged token by construction; this matches `glitch_m`
    # post hoc, so the two agree or the construction is doing the work.
    res["gates"]["matched_guard"] = surprisal_matched_sep(R, split, H, rng)

    for nm, lab in (("edit", 1.0), ("glitch_m", 0.0), ("glitch_s", 0.0), ("quiet", 0.5)):
        R[nm]["gates"]["label"] = np.full((n_ev, H + 1), lab)
    shufflable = [g for g in GATES if g != "probe"] + [f"probe_{gl}" for gl in GLITCHES]
    perm = rng.permutation(len(ARMS) * n_ev)
    for g in shufflable:
        st = np.concatenate([R[nm]["gates"][g] for nm in ARMS])[perm]
        for i, nm in enumerate(ARMS):
            R[nm]["gates"][f"shuf_{g}"] = st[i * n_ev:(i + 1) * n_ev]

    def gate_names(gl, short=False):
        base = list(GRID_GATES) if short else list(GATES)
        out = []
        for g in base:
            out.append(f"probe_{gl}" if g == "probe" else
                       (f"shuf_probe_{gl}" if g == "shuf_probe" else g))
        if not short:
            out += [f"shuf_{g}" for g in out] + ["label"]
        return out

    # ---------------- Q3: the mixed world ----------------------------------
    A = {nm: R[nm]["hold"][variant]["a"] for nm in R}
    B = {nm: R[nm]["hold"][variant]["b"] for nm in R}

    def mixed_eval(gl, p_edit, p_glitch, p_quiet, gates):
        arms = ("edit", gl, "quiet")
        a_te = np.concatenate([A[nm][~split] for nm in arms])
        b_te = np.concatenate([B[nm][~split] for nm in arms])
        a_tr = np.concatenate([A[nm][split] for nm in arms])
        b_tr = np.concatenate([B[nm][split] for nm in arms])
        wt = {"edit": p_edit, gl: p_glitch, "quiet": p_quiet}
        fit_sw = np.concatenate([np.full(ntr, wt[nm] / max(ntr, 1)) for nm in arms])
        te_sw = np.concatenate([np.full(nte, wt[nm] / max(nte, 1)) for nm in arms])
        out = {"glitch_arm": gl,
               "rates": {"edit": p_edit, "glitch": p_glitch, "quiet": p_quiet},
               "fixed": {}, "gated": {}}
        for p, wv in pol.items():
            out["fixed"][p] = {"nll": _pooled(a_te, b_te, wv[None, :], te_sw[:, None]),
                               "by_tau": [_pooled(a_te[:, t], b_te[:, t], wv[t], te_sw)
                                          for t in range(H + 1)]}
        w_fix = np.array([w_opt(a_tr[:, t], b_tr[:, t], fit_sw) for t in range(H + 1)])
        out["fixed"]["best_fixed_fit"] = {
            "w": w_fix.round(4).tolist(),
            "nll": _pooled(a_te, b_te, w_fix[None, :], te_sw[:, None]),
            "by_tau": [_pooled(a_te[:, t], b_te[:, t], w_fix[t], te_sw) for t in range(H + 1)]}
        w_or = np.where(b_te > a_te, 1.0, 0.0)
        out["fixed"]["oracle_per_pred"] = {
            "nll": _pooled(a_te, b_te, w_or, te_sw[:, None]),
            "by_tau": [_pooled(a_te[:, t], b_te[:, t], w_or[:, t], te_sw)
                       for t in range(H + 1)]}
        for g in gates:
            g_tr = np.concatenate([R[nm]["gates"][g][split] for nm in arms])
            g_te = np.concatenate([R[nm]["gates"][g][~split] for nm in arms])
            maps = [fit_wmap(g_tr[:, t], a_tr[:, t], b_tr[:, t], fit_sw) for t in range(H + 1)]
            wte = np.stack([maps[t](g_te[:, t]) for t in range(H + 1)], 1)
            out["gated"][g] = {
                "nll": _pooled(a_te, b_te, wte, te_sw[:, None]),
                "by_tau": [_pooled(a_te[:, t], b_te[:, t], wte[:, t], te_sw)
                           for t in range(H + 1)],
                "w_tau0_by_world": {nm: float(maps[0](R[nm]["gates"][g][~split, 0]).mean())
                                    for nm in R},
                "w_mean_by_world": {nm: float(np.mean([
                    maps[t](R[nm]["gates"][g][~split, t]).mean() for t in range(H + 1)]))
                    for nm in R},
                "map_tau0": maps[0].summary()}
        return out

    res["mixed"] = {"headline_rates": [1 / 3, 1 / 3, 1 / 3], "by_glitch": {}, "grid": []}
    for gl in GLITCHES:
        hm = mixed_eval(gl, 1 / 3, 1 / 3, 1 / 3, gate_names(gl))
        res["mixed"]["by_glitch"][gl] = hm
        print(f"  mixed [{gl}] (1:1:1)  " + "  ".join(
            f"{k} {vv['nll']:.4f}" for k, vv in hm["fixed"].items()), flush=True)
        for g in hm["gated"]:
            print(f"    gated {g:20s} {hm['gated'][g]['nll']:.4f}  w(tau0) e/g/q "
                  f"{hm['gated'][g]['w_tau0_by_world']['edit']:.3f}/"
                  f"{hm['gated'][g]['w_tau0_by_world'][gl]:.3f}/"
                  f"{hm['gated'][g]['w_tau0_by_world']['quiet']:.3f}", flush=True)
    res["mixed"]["headline"] = res["mixed"]["by_glitch"][GLITCHES[0]]
    for gl in GLITCHES:
        for pe in (0.25, 0.5, 0.75):
            for pq in (0.0, 1 / 3, 0.5):
                r = mixed_eval(gl, pe * (1 - pq), (1 - pe) * (1 - pq), pq, gate_names(gl, True))
                res["mixed"]["grid"].append({
                    "glitch_arm": gl, "rates": r["rates"],
                    "fixed": {k: vv["nll"] for k, vv in r["fixed"].items()},
                    "gated": {k: vv["nll"] for k, vv in r["gated"].items()}})

    # ---------------- persistence under each policy ------------------------
    res["persistence"] = {}
    for nm in R:
        prof = {}
        for wv in (0.0, 0.25, 0.5, 0.75, 1.0):
            Hm = ent(np.exp(mix_logq(R[nm]["logq_nat"], R[nm]["q_hold"], wv)))
            prof[str(wv)] = {
                "H_by_tau": Hm.mean(0).round(4).tolist(),
                "dH_by_tau": (Hm - R[nm]["H_prev"][:, None]).mean(0).round(4).tolist(),
                "nll_by_tau": nll_at_w(R[nm]["hold"][variant]["a"],
                                       R[nm]["hold"][variant]["b"],
                                       wv).mean(0).round(4).tolist()}
        res["persistence"][nm] = prof
    print("  persistence dH(edit) w=0 " + str(res["persistence"]["edit"]["0.0"]["dH_by_tau"][:4])
          + " w=1 " + str(res["persistence"]["edit"]["1.0"]["dH_by_tau"][:4]), flush=True)
    for nm in R:
        del R[nm]["states"]

    # ---------------- the eps venue, where the ceiling is on file -----------
    if do_natural:
        res["natural"] = natural_venue(model, rules, rule_w, eps_data, H, n_ladder, variant,
                                      max_events=nat_max_events, seed=seed)

    # ---------------- the parent's 2e pairs under each policy ---------------
    if do_pairs:
        res["pairs"] = pair_readouts(model, stim_tag, H, caliper, seed, max_pair_windows,
                                     variant)

    name = f"{ckpt[:-3]}_basalis{('_' + tag) if tag else ''}.json"
    with open(name, "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {name}",
          flush=True)
    return {"ckpt": ckpt, "step": cfg.get("step")}


def surprisal_matched_sep(R, split, H, rng, caliper=0.25):
    """The flagged token is the one thing the two worlds are not matched on by construction
    (the edit's violator is original material read under an edited prefix; the glitch's is a
    uniform draw). Nearest-neighbour match on the model's own surprisal for it and re-read
    every gate's separation there, so a gate cannot win on the event token alone."""
    from rhm.logit_reading.violation import nn_match
    se, sg = R["edit"]["s_event"], R["glitch_m"]["s_event"]
    idx = np.where(~split)[0]
    cell = np.zeros(len(idx), np.int64)
    pv, pc = nn_match(idx, cell, se[idx], idx.copy(), cell.copy(), sg[idx], caliper, rng)
    out = {"n_matched": int(len(pv)), "caliper": caliper,
           "guard_sevent_auc": (auc(np.concatenate([se[pv], sg[pc]]),
                                    np.concatenate([np.ones(len(pv), bool),
                                                    np.zeros(len(pc), bool)]))
                                if len(pv) > 30 else float("nan")),
           "by_gate": {}}
    if len(pv) <= 30:
        return out
    y = np.concatenate([np.ones(len(pv), bool), np.zeros(len(pc), bool)])
    for g in R["edit"]["gates"]:
        if g.startswith("shuf_") or g == "label":
            continue
        out["by_gate"][g] = [auc(np.concatenate([R["edit"]["gates"][g][pv, t],
                                                 R["glitch_m"]["gates"][g][pc, t]]), y)
                             for t in range(H + 1)]
    return out


# ---------------------------------------------------------------------------
# the eps venue: the same consumer against `coeruleus/` Q1's ceiling
# ---------------------------------------------------------------------------

def natural_venue(model, rules, rule_w, eps_data, H, n_ladder, variant, L=6, max_events=0,
                  seed=0):
    from rhm.logit_reading.coeruleus.events import assert_cache_matches, events_from_mask
    w, w_clean, corrupt, phase = assert_cache_matches(rules, rule_w, eps_data, n_ladder, 4242)
    pe, wl, _ = load_ladder(eps_data, eps_data)
    assert (wl == w).all()
    T = w.shape[1] - 1
    lq_all = logq_full(model, w, "cuda")
    pe64 = pe.astype(np.float64)
    kap, klk = fit_kappa_pooled(pe64, lq_all, L)
    logceil = np.log(np.clip(convex_obs(pe64, kap, L), 1e-300, None))
    pE = pe64[..., L, :]
    # The second, equally honest reference. `p_L^eps` is the law for an observer of the
    # window, which does not know the token at c was corrupted; our consumer is told that it
    # was, and on that subpopulation the realised token is drawn from the CLEAN stream. The
    # clean ladder on the uncorrupted draw of the same windows is on file, so
    # CE(p_L^clean, .) is the exact, variance-reduced version of the realised-NLL column, and
    # the model's own altitude on the undamaged window is its ceiling. (`coeruleus/`'s
    # "two loss columns, both honest, can disagree in sign" -- here they are both computed.)
    pc, wcl, _ = load_ladder(0.0, 0.0)
    assert (wcl == w_clean).all(), "the clean ladder is not the same draw"
    lq_clean = logq_full(model, w_clean, "cuda")
    pc64 = pc.astype(np.float64)
    kap_c, klk_c = fit_kappa_pooled(pc64, lq_clean, L)
    logceil_c = np.log(np.clip(convex_obs(pc64, kap_c, L), 1e-300, None))
    pC = pc64[..., L, :]
    del lq_clean
    win, c = events_from_mask(corrupt, H, c_lo=16, isolate=True)
    ok = c <= T - 1 - H
    win, c = win[ok], c[ok]
    n_all = len(win)
    if max_events and n_all > max_events:
        sel = np.sort(np.random.default_rng(seed + 5).choice(n_all, max_events, replace=False))
        win, c = win[sel], c[sel]
    out = {"kappa_star_eps": kap, "n_events_all": int(n_all), "KL_at_kappa": klk,
           "kappa_star_clean": kap_c, "KL_at_kappa_clean": klk_c,
           "n_events": int(len(win)), "n_windows": int(n_ladder)}
    if len(win) < 50:
        return out
    ar, taus = np.arange(len(win)), np.arange(H + 1)
    W = w[win]
    lqn = gather_offsets(lq_all[win], c, taus).astype(np.float64)
    tok = W[ar[:, None], c[:, None] + taus[None, :] + 1]
    v = lq_all.shape[-1]
    P = prep_imputed(impute_logq(model, W, c, c, np.tile(np.arange(v), (len(win), 1)), H,
                                 "cuda"), lq_all[win, c - 1, :], tok)
    q_h = hold_forecast(P, variant)
    del P
    a, b = realised_ab(lqn, q_h, tok)
    p_tr = pE[win[:, None], c[:, None] + taus[None, :]]
    lc = logceil[win[:, None], c[:, None] + taus[None, :]]
    p_cl = pC[win[:, None], c[:, None] + taus[None, :]]
    lc_cl = logceil_c[win[:, None], c[:, None] + taus[None, :]]
    rows = []
    for t in range(H + 1):
        wst = w_opt(a[:, t], b[:, t])
        row = {"tau": t, "w_star_realised": wst,
               "nll_native": float(nll_at_w(a[:, t], b[:, t], 0.).mean()),
               "nll_hold": float(nll_at_w(a[:, t], b[:, t], 1.).mean()),
               "nll_at_wstar": float(nll_at_w(a[:, t], b[:, t], wst).mean()),
               "ceil_CE": float(xent(p_tr[:, t], lc[:, t]).mean()),
               "ceilB_CE": float(xent(p_cl[:, t], lc_cl[:, t]).mean()),
               "H_truthB": float(ent(p_cl[:, t]).mean())}
        for wv, nmw in ((0.0, "native"), (1.0, "hold")):
            lqm = mix_logq(lqn[:, t], q_h[:, t], wv)
            row[f"CE_{nmw}"] = float(xent(p_tr[:, t], lqm).mean())
            row[f"excess_ceil_{nmw}"] = row[f"CE_{nmw}"] - row["ceil_CE"]
            row[f"CEB_{nmw}"] = float(xent(p_cl[:, t], lqm).mean())
            row[f"excess_ceilB_{nmw}"] = row[f"CEB_{nmw}"] - row["ceilB_CE"]
            row[f"H_{nmw}"] = float(ent(np.exp(lqm)).mean())
        wb = float(mix_opt(lqn[:, t], q_h[:, t], p_cl[:, t], pooled=True))
        row["w_star_ceilB"] = wb
        row["CEB_at_wstar"] = float(mix_ce(lqn[:, t], q_h[:, t], p_cl[:, t], wb).mean())
        row["excess_ceilB_at_wstar"] = row["CEB_at_wstar"] - row["ceilB_CE"]
        wc = float(mix_opt(lqn[:, t], q_h[:, t], p_tr[:, t], pooled=True))
        row["w_star_ceil"] = wc
        row["CE_at_wstar_ceil"] = float(mix_ce(lqn[:, t], q_h[:, t], p_tr[:, t], wc).mean())
        row["excess_ceil_at_wstar"] = row["CE_at_wstar_ceil"] - row["ceil_CE"]
        rows.append(row)
    out["by_tau"] = rows
    m = lambda k: float(np.mean([r[k] for r in rows]))                       # noqa: E731
    prize = m("CE_native") - m("ceil_CE")
    out["pooled"] = {
        "CE_native": m("CE_native"), "CE_ceil": m("ceil_CE"), "prize": prize,
        "CE_hold": m("CE_hold"), "CE_at_wstar_ceil": m("CE_at_wstar_ceil"),
        "frac_prize_hold": float((m("CE_native") - m("CE_hold")) / max(prize, 1e-9)),
        "frac_prize_wstar": float((m("CE_native") - m("CE_at_wstar_ceil")) / max(prize, 1e-9)),
        "nll_native": m("nll_native"), "nll_hold": m("nll_hold"),
        "nll_at_wstar": m("nll_at_wstar")}
    prizeB = m("CEB_native") - m("ceilB_CE")
    out["pooledB"] = {
        "CE_native": m("CEB_native"), "CE_ceil": m("ceilB_CE"), "H_truth": m("H_truthB"),
        "prize": prizeB, "CE_hold": m("CEB_hold"), "CE_at_wstar": m("CEB_at_wstar"),
        "frac_prize_hold": float((m("CEB_native") - m("CEB_hold")) / max(prizeB, 1e-9)),
        "frac_prize_wstar": float((m("CEB_native") - m("CEB_at_wstar")) / max(prizeB, 1e-9))}
    print(f"  natural venue col B (clean truth): kappa*_clean {kap_c:.2f} prize {prizeB:.4f} "
          f"hold {out['pooledB']['frac_prize_hold']:+.3f} "
          f"w* {out['pooledB']['frac_prize_wstar']:+.3f}", flush=True)
    print(f"  natural venue: kappa* {kap:.2f} events {len(win)} prize {prize:.4f} "
          f"hold {out['pooled']['frac_prize_hold']:+.3f} w* {out['pooled']['frac_prize_wstar']:+.3f}",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# the parent's 2e same-prefix pairs, read under each policy
# ---------------------------------------------------------------------------

def pair_readouts(model, stim_tag, H, caliper, seed, max_windows, variant):
    from rhm.logit_reading.coeruleus.readout import prefix_pairs
    from rhm.logit_reading.violation import _load_stimuli
    S = _load_stimuli(stim_tag)
    pp = prefix_pairs(model, S, "cuda", seed=seed, caliper=caliper, max_windows=max_windows)
    del S
    T = pp["viol_win"].shape[1] - 1
    keep = pp["t"] <= T - 1 - H
    out = {"n_matched": pp["n_matched"], "n_test": int((pp["te"] & keep).sum()),
           "prefix_check": pp["prefix_check"], "caliper": caliper, "readouts": {}}
    if (pp["te"] & keep).sum() < 30:
        return out
    idx = np.where(keep)[0]
    tt, teo, ks = pp["t"][idx], pp["te"][idx], pp["kstar"][idx]
    taus = np.arange(H + 1)
    v = model.transformer.wte.weight.shape[0]
    ar = np.arange(len(idx))
    packs = {}
    for side, Wn in (("viol", pp["viol_win"][idx]), ("twin", pp["twin_win"][idx])):
        lq = logq_full(model, Wn, "cuda")
        tok = Wn[ar[:, None], tt[:, None] + taus[None, :] + 1]
        lqn = gather_offsets(lq, tt, taus).astype(np.float64)
        P = prep_imputed(impute_logq(model, Wn, tt, tt, np.tile(np.arange(v), (len(idx), 1)),
                                     H, "cuda"), lq[ar, tt - 1, :], tok)
        q_h = hold_forecast(P, variant)
        packs[side] = {"lqn": lqn, "q_h": q_h, "ab": realised_ab(lqn, q_h, tok),
                       "H_prev": ent(np.exp(lq[ar, tt - 1, :]))}
        del P, lq
    # the two windows differ only at t_v, so the imputed forecast must be identical: an exact
    # self-check on the whole imputation path, and the reason the paired entropy statistic is
    # 0.5 by construction at w = 1.
    out["hold_identical_check"] = float(np.abs(packs["viol"]["q_h"] - packs["twin"]["q_h"]).max())
    for wv in (0.0, 0.25, 0.5, 0.75, 1.0):
        row, Hs = {}, {}
        for side in ("viol", "twin"):
            Hs[side] = ent(np.exp(mix_logq(packs[side]["lqn"], packs[side]["q_h"], wv)))
            a, b = packs[side]["ab"]
            row[f"nll_{side}_by_tau"] = nll_at_w(a, b, wv)[teo].mean(0).round(4).tolist()
            row[f"dH_{side}_by_tau"] = (Hs[side] - packs[side]["H_prev"][:, None]
                                        )[teo].mean(0).round(4).tolist()
        row["H_next_paired"] = paired_win(Hs["viol"][teo, 0], Hs["twin"][teo, 0])
        row["H_next_auc"] = auc(np.concatenate([Hs["viol"][teo, 0], Hs["twin"][teo, 0]]),
                                np.concatenate([np.ones(int(teo.sum()), bool),
                                                np.zeros(int(teo.sum()), bool)]))
        row["by_kstar"] = {int(k): {"n": int((teo & (ks == k)).sum()),
                                    "H_next_paired": paired_win(Hs["viol"][teo & (ks == k), 0],
                                                                Hs["twin"][teo & (ks == k), 0])}
                           for k in range(1, 7) if (teo & (ks == k)).sum() >= 30}
        row["nll_paired_tau1"] = paired_win(nll_at_w(*packs["viol"]["ab"], wv)[teo, 1],
                                            nll_at_w(*packs["twin"]["ab"], wv)[teo, 1])
        out["readouts"][str(wv)] = row
        print(f"  pairs w={wv:.2f} H_next paired {row['H_next_paired']:.3f}  nll v/t tau1 "
              f"{row['nll_viol_by_tau'][1]:.3f}/{row['nll_twin_by_tau'][1]:.3f}", flush=True)
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def arbitrate_sweep(ckpts: str = "", head_tags: str = "", tags: str = "", n_events: int = 6000,
                    n_ladder: int = 4096, variant: str = "pf"):
    """One container per checkpoint (no extra GPU-hours, just no queueing)."""
    volume.reload()
    d = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    cs = ckpts.split(",") if ckpts else [f"{d}/traj_eps01_s42/step064000.pt",
                                         f"{d}/traj_a1_s42/step064000.pt"]
    hs = head_tags.split(",") if head_tags else ["" if "eps01" in c else "onnoise" for c in cs]
    ts = tags.split(",") if tags else [""] * len(cs)
    return list(arbitrate_ckpt.starmap(
        list(zip(cs, hs, ts)),
        kwargs={"n_events": n_events, "n_ladder": n_ladder, "variant": variant}))
