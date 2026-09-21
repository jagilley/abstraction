"""Addendum to `arbitrate.py` (2026-09-17): four additions on the gate / arbitration side.

The first round found that a supervised world probe on the residual stream separates the two
worlds well (0.71 at the flagged token, surprisal-matched, against a 0.52 floor) and yet, used
as a gate, is WORSE than its own shuffle. Two mundane causes have to be ruled in or out before
that is read as "the state cannot be cashed":

 1. **The label may be wrong.** `j = 0` and `j = 1` swaps behave as glitches by the round's own
    `j`-split (`w*` 0.15-0.75 at the flag), so "edit" is the wrong label for them from the
    policy's point of view and a provenance-trained probe is being taught to call glitch-like
    events "edit". Every separation and mixed-world table is therefore recomputed on `j >= 2`
    edits against their paired glitches, beside the full population.
 2. **The target may be wrong.** A probe trained on WHERE HOLDING PAID (`b > a` at that offset,
    the same realised quantity the fitted map is built on -- no world label anywhere) beside
    the provenance one, with its own shuffle. This is the direct test of "the probe's ordering
    is not the ordering of where holding pays".
 3. **The probe may never have seen a false alarm.** A three-class probe over
    {edit, glitch, quiet}, gate = `log p(glitch) - log p(edit)`, so the quiet cost is not an
    artefact of a two-class probe reading a false alarm as "not an edit". What it assigns to
    quiet, and what `w` the fitted map then sets there, are reported.
 4. **The lead at tau = 0 may be the parent's tonic signal.** The edit arm carries ~6 tokens of
    regrown material before the flag; the glitch arm carries none. The world probe is therefore
    read on the state at `t_v - 1`, BEFORE the flagged token, on the surprisal-matched pairs and
    split by whether the edit actually precedes the flag (`t_v > e`). Where `t_v == e` the two
    windows share their whole prefix, so those states are identical and the probe must read
    exactly 0.5 -- an internal floor, asserted. The parent's 2b scalar (the model's own running
    surprisal over the last four tokens) is read at the same place.

Two more (2026-09-17, second pass), on the same launch shape:

 5. **Is a gate ordered by the size of the payoff, or only by its sign?** Every gate's rank
    correlation with `log(b/a)` -- the signed, continuous stake -- per offset, beside the
    hold-better AUC it already has, on the same held-out rows.
 6. **A stake-regression probe as a gate.** The same states, trained to predict `log(b/a)`
    itself: a label-free target, since it needs only the second forward pass the consumer
    already runs. Exact closed-form ridge (the `coeruleus/` gotcha about noisy targets) with an
    MLP beside it, refit per subset, with its shuffle. Reported first as a PREDICTOR (held-out
    R^2 and Spearman against `log(b/a)`, beside the model's own log Bayes factor rescaled on
    the fit split) and then as a GATE. If a readout trained on the stake itself beats its
    shuffle, the state can be cashed and every earlier probe had the wrong target; if it does
    not, the state carries where holding pays but not how much.

Same populations, same seed, same 60/40 split as `arbitrate.py`, asserted against the main run's
saved per-arm event surprisals before anything is computed. Writes a separate JSON, so the first
round's artifact and tables are untouched. A5-A6 are additive: the run reports the max deviation
of every A1-A4 scalar from the previous artifact (`regression_check`) rather than assuming it.

Run:
  modal run --detach -m rhm.logit_reading.basalis.gates2::gates2_sweep
"""

import json

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt
from rhm.logit_reading.violation import auc, fit_probe, nn_match
from rhm.logit_reading.coeruleus.readout import (
    load_head, ridge_solve, train_head, _r2, _spearman, LAMS)
from rhm.logit_reading.basalis import worlds
from rhm.logit_reading.basalis.hold import (
    logq_full, states_offsets, w_opt, fit_wmap, ent)
from rhm.logit_reading.basalis.arbitrate import arm_record, _pooled

ARMS = ("edit", "glitch_m", "glitch_s", "quiet")
GLITCHES = ("glitch_s", "glitch_m")
SCALAR_GATES = ("head_ridge", "head_mlp", "Hq", "runex", "logbf", "sevent")


def _ignore(path):
    """Mount the Python sources only, and never any `results/` tree or `.log` -- another agent
    writing its launch log inside the package makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-basalis-gates2", image=image)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fit_probe_multi(Xtr, ytr, n_cls=3, hidden=256, steps=500, lr=1e-2, wd=1e-3,
                    device="cuda", seed=0):
    """`violation.fit_probe` with a softmax head. Returns class probabilities."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(np.asarray(Xtr), dtype=torch.float32, device=device)
    ytr = torch.as_tensor(np.asarray(ytr), dtype=torch.long, device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    d = Xtr.shape[1]
    net = (nn.Linear(d, n_cls) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_cls))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xn = (Xtr - mu) / sd
    for _ in range(steps):
        loss = nn.functional.cross_entropy(net(Xn), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict(X):
        with torch.no_grad():
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
            return torch.softmax(net((X - mu) / sd), -1).cpu().numpy()
    return predict


def bin_r2(g_tr, y_tr, g_te, y_te, nbins=14):
    """How much of the stake's variance a gate recovers THROUGH ITS OWN MACHINERY: the same
    quantile step function the fitted `w` map uses, here predicting `log(b/a)` -- fit on the
    fit rows, scored on the test rows. Monotone-invariant, like the map itself, so it is the
    apples-to-apples form of "R^2" for a gate whose relation to the stake is not linear."""
    g_tr, y_tr = np.asarray(g_tr, np.float64), np.asarray(y_tr, np.float64)
    g_te, y_te = np.asarray(g_te, np.float64), np.asarray(y_te, np.float64)
    edges = np.unique(np.quantile(g_tr, np.linspace(0, 1, nbins + 1)[1:-1]))
    b_tr = np.clip(np.digitize(g_tr, edges), 0, len(edges))
    mu = np.array([y_tr[b_tr == i].mean() if (b_tr == i).sum() >= 20 else y_tr.mean()
                   for i in range(len(edges) + 1)])
    pred = mu[np.clip(np.digitize(g_te, edges), 0, len(edges))]
    return float(1.0 - ((y_te - pred) ** 2).mean() / max(y_te.var(), 1e-12))


def _auc2(a, b):
    """AUC that `a` scores above `b`."""
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    return auc(np.concatenate([a, b]),
               np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)]))


STAKE_CLIP = 25.0


def stake(R, nm, variant):
    """log(b/a): how much more (or less) mass the held forecast put on the token that came."""
    a = np.clip(R[nm]["hold"][variant]["a"], 1e-12, None)
    b = np.clip(R[nm]["hold"][variant]["b"], 1e-12, None)
    return np.clip(np.log(b) - np.log(a), -STAKE_CLIP, STAKE_CLIP)


def gate_names(gl, scalars=SCALAR_GATES):
    base = (list(scalars) + [f"payoff_{gl}", f"probe3_{gl}", f"probe_{gl}",
                             f"stake_ridge_{gl}", f"stake_mlp_{gl}"])
    return base + [f"shuf_{g}" for g in base] + ["label"]


def stake_probes(R, gl, sel, split, H, blk, variant, mlp_hidden=256, mlp_steps=1500,
                 inner=0.8, seed=0):
    """A6. Predict the stake itself from the state. Exact ridge from the normal equations
    (lambda chosen on a held-out slice of the FIT rows -- the target is label-free, so no
    oracle enters), an MLP beside it, and the model's own log Bayes factor rescaled on the
    same fit rows so the three are compared as predictors on the same footing."""
    arms3 = ("edit", gl, "quiet")
    itr, ite = np.where(sel & split)[0], np.where(sel & ~split)[0]
    rg = np.random.default_rng(seed + 55)
    out = {"n_fit_rows": int(3 * len(itr)), "n_test_rows": int(3 * len(ite)), "by_tau": []}
    pred = {k: {nm: np.zeros((len(R[nm]["s_event"]), H + 1)) for nm in R}
            for k in ("stake_ridge", "stake_mlp")}
    for t in range(H + 1):
        X = np.concatenate([R[nm]["states"][blk][itr, t] for nm in arms3]).astype(np.float64)
        y = np.concatenate([stake(R, nm, variant)[itr, t] for nm in arms3])
        Xte = np.concatenate([R[nm]["states"][blk][ite, t] for nm in arms3]).astype(np.float64)
        yte = np.concatenate([stake(R, nm, variant)[ite, t] for nm in arms3])
        m = rg.random(len(X)) < inner
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        A0 = Xb[m].T @ Xb[m]
        c0 = Xb[m].T @ y[m]
        best = (None, -1e18)
        for lam in LAMS:
            beta = ridge_solve(A0, c0, lam, int(m.sum()))
            r2 = _r2(Xb[~m] @ beta, y[~m])
            if r2 > best[1]:
                best = (lam, r2)
            del beta
        lam = best[0]
        A = Xb.T @ Xb
        beta = ridge_solve(A, Xb.T @ y, lam, len(X))
        pr = Xte @ beta[:-1] + beta[-1]
        row = {"tau": t, "lam": lam, "ridge_R2": _r2(pr, yte),
               "ridge_spearman": _spearman(pr, yte)}
        h = train_head(X.astype(np.float32), y.astype(np.float32), blk, hidden=mlp_hidden,
                       steps=mlp_steps, seed=seed)
        prm = h(Xte.astype(np.float32))
        row["mlp_R2"] = _r2(prm, yte)
        row["mlp_spearman"] = _spearman(prm, yte)
        # the model's own log Bayes factor, rescaled on the fit rows: same footing
        gtr = np.concatenate([R[nm]["gates"]["logbf"][itr, t] for nm in arms3])
        gte = np.concatenate([R[nm]["gates"]["logbf"][ite, t] for nm in arms3])
        v_ = np.var(gtr)
        sl = (np.cov(gtr, y)[0, 1] / v_) if v_ > 1e-12 else 0.0
        ic = float(y.mean() - sl * gtr.mean())
        row["logbf_rescaled_R2"] = _r2(sl * gte + ic, yte)
        row["logbf_spearman"] = _spearman(gte, yte)
        row["stake_sd_test"] = float(yte.std())
        row["stake_mean_test"] = float(yte.mean())
        for nm in R:
            Xn = R[nm]["states"][blk][:, t].astype(np.float64)
            pred["stake_ridge"][nm][:, t] = Xn @ beta[:-1] + beta[-1]
            pred["stake_mlp"][nm][:, t] = h(R[nm]["states"][blk][:, t].astype(np.float32))
        out["by_tau"].append(row)
        del X, Xb, Xte, A, A0, h
    for k, d in pred.items():
        for nm in R:
            R[nm]["gates"][f"{k}_{gl}"] = d[nm]
    return out


def separation(R, gl, sel, split, H, probe_steps, probe_hidden, blk, variant):
    """Every gate's AUC(edit vs glitch) and AUC(edit vs quiet) by offset, fitted on `sel`'s
    fit split and read on its test split. Returns (table, the probe's scores for every arm)."""
    tr = np.where(sel & split)[0]
    te = np.where(sel & ~split)[0]
    n = len(R["edit"]["s_event"])
    out = {"n_fit": int(len(tr)), "n_test": int(len(te))}
    scores = {nm: np.zeros((n, H + 1)) for nm in R}
    arms3 = ("edit", gl, "quiet")

    def hb(get):
        """AUC of a gate against the label the fitted map actually cares about: did holding
        give the realised token more mass here. Pooled over the three arms of the mixed world,
        on the test split. A gate can separate the worlds perfectly and still be at chance
        here -- that is the whole question."""
        return [auc(np.concatenate([get(nm)[te, t] for nm in arms3]),
                    np.concatenate([(R[nm]["hold"][variant]["b"][te, t]
                                     > R[nm]["hold"][variant]["a"][te, t])
                                    for nm in arms3])) for t in range(H + 1)]

    def br2(get):
        return [bin_r2(np.concatenate([get(nm)[tr, t] for nm in arms3]),
                       np.concatenate([stake(R, nm, variant)[tr, t] for nm in arms3]),
                       np.concatenate([get(nm)[te, t] for nm in arms3]),
                       np.concatenate([stake(R, nm, variant)[te, t] for nm in arms3]))
                for t in range(H + 1)]

    def rho(get):
        """Rank correlation with the SIGNED CONTINUOUS payoff `log(b/a)` -- the stake. A gate
        can be well ordered by the sign of the payoff and badly ordered by its size; the loss
        is in nats, so it is the size that a fitted map can convert into nats."""
        return [_spearman(np.concatenate([get(nm)[te, t] for nm in arms3]),
                          np.concatenate([stake(R, nm, variant)[te, t] for nm in arms3]))
                for t in range(H + 1)]
    for label, key, hid in (("probe", blk, probe_hidden), ("probe_embed", "post_embed", 0),
                            ("probe_rand", "rand", 0)):
        eg, eq = [], []
        psc = {nm: np.zeros((n, H + 1)) for nm in R}
        for t in range(H + 1):
            Xe, Xg = R["edit"]["states"][key][:, t], R[gl]["states"][key][:, t]
            f = fit_probe(np.concatenate([Xe[tr], Xg[tr]]),
                          np.concatenate([np.ones(len(tr)), np.zeros(len(tr))]),
                          hidden=hid, steps=probe_steps)
            for nm in R:
                psc[nm][:, t] = f(R[nm]["states"][key][:, t])
            if label == "probe":
                for nm in R:
                    scores[nm][:, t] = psc[nm][:, t]
            se, sg = psc["edit"][:, t], psc[gl][:, t]
            eg.append(_auc2(se[te], sg[te]))
            eq.append(_auc2(se[te], psc["quiet"][te, t]))
        out[label] = {"auc_edit_vs_glitch_by_tau": eg, "auc_edit_vs_quiet_by_tau": eq,
                      "auc_holdbetter_by_tau": hb(lambda nm: psc[nm]),
                      "spearman_logba_by_tau": rho(lambda nm: psc[nm]),
                      "binr2_logba_by_tau": br2(lambda nm: psc[nm])}
    for g in R["edit"]["gates"]:
        if g.startswith("shuf_") or g == "label" or g.startswith("probe_"):
            continue
        out[g] = {"auc_edit_vs_glitch_by_tau":
                  [_auc2(R["edit"]["gates"][g][te, t], R[gl]["gates"][g][te, t])
                   for t in range(H + 1)],
                  "auc_edit_vs_quiet_by_tau":
                  [_auc2(R["edit"]["gates"][g][te, t], R["quiet"]["gates"][g][te, t])
                   for t in range(H + 1)],
                  "auc_holdbetter_by_tau": hb(lambda nm, g=g: R[nm]["gates"][g]),
                  "spearman_logba_by_tau": rho(lambda nm, g=g: R[nm]["gates"][g]),
                  "binr2_logba_by_tau": br2(lambda nm, g=g: R[nm]["gates"][g])}
    return out, scores


def matched_pairs(R, gl, sel, split, caliper, rng):
    """Held-out events matched on the model's own surprisal for the flagged token."""
    se, sg = R["edit"]["s_event"], R[gl]["s_event"]
    idx = np.where(sel & ~split)[0]
    cell = np.zeros(len(idx), np.int64)
    return nn_match(idx, cell, se[idx], idx.copy(), cell.copy(), sg[idx], caliper, rng)


def matched_sep(R, gl, pv, pc, H, caliper):
    out = {"n_matched": int(len(pv)), "caliper": caliper, "glitch_arm": gl, "by_gate": {}}
    if len(pv) <= 30:
        return out
    out["guard_sevent_auc"] = _auc2(R["edit"]["s_event"][pv], R[gl]["s_event"][pc])
    for g in R["edit"]["gates"]:
        if g.startswith("shuf_") or g == "label":
            continue
        out["by_gate"][g] = [_auc2(R["edit"]["gates"][g][pv, t], R[gl]["gates"][g][pc, t])
                             for t in range(H + 1)]
    return out


def mixed_eval(R, variant, gl, sel, split, rates, gates, H):
    """The mixed world restricted to `sel`, weights `rates` = (edit, glitch, quiet)."""
    arms = ("edit", gl, "quiet")
    wt = dict(zip(arms, rates))
    itr, ite = np.where(sel & split)[0], np.where(sel & ~split)[0]
    A = {nm: R[nm]["hold"][variant]["a"] for nm in arms}
    B = {nm: R[nm]["hold"][variant]["b"] for nm in arms}
    a_tr = np.concatenate([A[nm][itr] for nm in arms])
    b_tr = np.concatenate([B[nm][itr] for nm in arms])
    a_te = np.concatenate([A[nm][ite] for nm in arms])
    b_te = np.concatenate([B[nm][ite] for nm in arms])
    sw_tr = np.concatenate([np.full(len(itr), wt[nm] / max(len(itr), 1)) for nm in arms])
    sw_te = np.concatenate([np.full(len(ite), wt[nm] / max(len(ite), 1)) for nm in arms])
    out = {"glitch_arm": gl, "n_fit_per_arm": int(len(itr)), "n_test_per_arm": int(len(ite)),
           "rates": dict(zip(("edit", "glitch", "quiet"), rates)), "fixed": {}, "gated": {}}
    for nm_, wv in (("native", np.zeros(H + 1)), ("hold", np.ones(H + 1))):
        out["fixed"][nm_] = {"nll": _pooled(a_te, b_te, wv[None, :], sw_te[:, None])}
    w_fix = np.array([w_opt(a_tr[:, t], b_tr[:, t], sw_tr) for t in range(H + 1)])
    out["fixed"]["best_fixed_fit"] = {"w": w_fix.round(4).tolist(),
                                      "nll": _pooled(a_te, b_te, w_fix[None, :], sw_te[:, None])}
    out["fixed"]["oracle_per_pred"] = {
        "nll": _pooled(a_te, b_te, np.where(b_te > a_te, 1.0, 0.0), sw_te[:, None])}
    for g in gates:
        if g not in R["edit"]["gates"]:
            continue
        g_tr = np.concatenate([R[nm]["gates"][g][itr] for nm in arms])
        g_te = np.concatenate([R[nm]["gates"][g][ite] for nm in arms])
        maps = [fit_wmap(g_tr[:, t], a_tr[:, t], b_tr[:, t], sw_tr) for t in range(H + 1)]
        wte = np.stack([maps[t](g_te[:, t]) for t in range(H + 1)], 1)
        out["gated"][g] = {
            "nll": _pooled(a_te, b_te, wte, sw_te[:, None]),
            "w_tau0_by_world": {nm: float(maps[0](R[nm]["gates"][g][ite, 0]).mean())
                                for nm in arms},
            "w_mean_by_world": {nm: float(np.mean([maps[t](R[nm]["gates"][g][ite, t]).mean()
                                                   for t in range(H + 1)])) for nm in arms}}
    return out


def _common_scalars(a, b, path=""):
    """Every numeric leaf present in both nested structures, with its path."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                yield from _common_scalars(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _common_scalars(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if not (np.isnan(a) or np.isnan(b)):
            yield path, float(a), float(b)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=16384,
              max_containers=4)
def gates2_ckpt(ckpt: str, head_tag: str = "", tag: str = "", stim_tag: str = "swap65k",
                horizon: int = 8, n_events: int = 6000, variant: str = "pf", seed: int = 0,
                t_lo: int = 16, probe_steps: int = 500, probe_hidden: int = 256,
                j_min: int = 2, caliper: float = 0.25, main_tag: str = "",
                check_population: bool = True):
    import resource
    import torch
    from rhm.model import GPT
    volume.reload()
    H = horizon
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    v, T = cfg["v"], cfg["s"] ** cfg["L"]
    hp = f"{ckpt[:-3]}_coeruleus_head_state_excess_%s{('_' + head_tag) if head_tag else ''}.pt"
    heads = {f"head_{k}": load_head(hp % k, "cuda")[0] for k in ("ridge", "mlp")}
    blk = heads["head_ridge"].block
    torch.manual_seed(12345)
    rand = GPT(v, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to("cuda").eval()
    res = {"config": {"ckpt": ckpt, "step": cfg.get("step"),
                      "eps_train": cfg.get("eps_train", 0.0), "H": H, "n_events": n_events,
                      "seed": seed, "variant": variant, "j_min": j_min, "block": blk,
                      "caliper": caliper, "probe_hidden": probe_hidden,
                      "addendum_of": f"{ckpt[:-3]}_basalis.json"}}

    # ---- the same populations as the main run, rebuilt and then asserted ----
    S = worlds.load_swap_stimuli(stim_tag, data_dir=DATA_DIR)
    spec, meta = worlds.build_arms(S, H, n_events, seed=seed, t_lo=t_lo, v=v)
    del S
    n_ev = meta["n_events"]
    ar_ = np.arange(n_ev)
    blocks = [blk, "post_embed"]
    R = {}

    def run_arm(nm):
        R[nm] = arm_record(model, spec[nm]["win"], spec[nm]["pos"], H, blocks, variant)
        R[nm]["states"]["rand"] = states_offsets(rand, spec[nm]["win"], spec[nm]["pos"],
                                                 np.arange(H + 1), blk)[blk]
        for k, h in heads.items():
            X = R[nm]["states"][h.block]
            R[nm]["gates"][k] = h(X.reshape(-1, X.shape[-1])).reshape(R[nm]["n"], H + 1)

    for nm in ("edit", "glitch_m", "quiet"):
        run_arm(nm)
    dist = np.abs(-R["glitch_m"]["log_prior"] - R["edit"]["s_event"][:, None])
    dist[ar_, meta["orig_token"]] = np.inf
    Ws = np.array(spec["glitch_m"]["win"])
    Ws[ar_, spec["glitch_m"]["pos"]] = dist.argmin(1)
    spec["glitch_s"] = {"win": Ws, "pos": spec["glitch_m"]["pos"].copy(),
                        "src": spec["glitch_m"]["src"]}
    run_arm("glitch_s")
    main = f"{ckpt[:-3]}_basalis{('_' + main_tag) if main_tag else ''}.json"
    worst = float("nan")
    if check_population:
        with open(main) as f:
            prev = json.load(f)["stimuli"]["s_event_by_arm"]
        worst = max(abs(float(R[nm]["s_event"].mean()) - prev[nm]) for nm in R)
        assert worst < 1e-9, f"populations differ from the main run ({worst})"
    res["population_check"] = {"max_abs_diff_s_event": worst, "against": main,
                               "checked": bool(check_population)}
    rng = np.random.default_rng(seed)
    split = rng.random(n_ev) < 0.6                     # identical to the main run's split
    res["split"] = {"n_fit": int(split.sum()), "n_test": int((~split).sum())}
    print(f"step {cfg.get('step')}: populations match the main run (max diff {worst:.1e}); "
          f"n={n_ev}, fit {int(split.sum())}, test {int((~split).sum())}", flush=True)

    # ---- addition 4's extra reads: the state and the running surprisal at t_v - 1 ----
    pre = {}
    for nm in ARMS:
        st = states_offsets(model, spec[nm]["win"], spec[nm]["pos"], [-1], blocks, "cuda")
        lq = logq_full(model, spec[nm]["win"], "cuda")
        tok = spec[nm]["win"][:, 1:lq.shape[1] + 1]
        nll = -np.take_along_axis(lq, tok[..., None], -1)[..., 0]
        Hq = ent(np.exp(lq))
        p = spec[nm]["pos"]
        # the four predictions fully scored by the time the forecast at pos-1 is made
        cols = np.stack([p - 5 + i for i in range(4)], 1)
        pre[nm] = {"state": {b: st[b][:, 0] for b in blocks},
                   "runsurp4": np.take_along_axis(nll, cols, 1).mean(1),
                   "runex4": np.take_along_axis(nll - Hq, cols, 1).mean(1),
                   "Hq_prev": Hq[ar_, p - 1]}
        del lq, nll, Hq, st
    gt = meta["delay"] > 0
    same = ~gt
    chk = float(np.abs(pre["edit"]["state"][blk][same]
                       - pre["glitch_s"]["state"][blk][same]).max()) if same.any() else 0.0
    assert chk < 1e-4, f"t_v == e events must share their whole prefix ({chk})"
    res["preamble"] = {"n_tv_gt_e": int(gt.sum()), "frac_tv_gt_e": float(gt.mean()),
                       "identical_prefix_state_check": chk, "by_glitch": {}}
    print(f"  preamble: t_v > e on {gt.mean():.2f} of events; identical-prefix state check "
          f"{chk:.1e}", flush=True)

    # ---- additions 2 and 3: the payoff-trained and three-class probes -------
    for gl in GLITCHES:
        arms3 = ("edit", gl, "quiet")
        tr = np.where(split)[0]
        ite = np.where(~split)[0]
        pay = {nm: np.zeros((n_ev, H + 1)) for nm in ARMS}
        p3 = {nm: np.zeros((n_ev, H + 1)) for nm in ARMS}
        a_pay, a_p3, pq, cls_acc = [], [], [], []
        for t in range(H + 1):
            X = np.concatenate([R[nm]["states"][blk][tr, t] for nm in arms3])
            y = np.concatenate([(R[nm]["hold"][variant]["b"][tr, t]
                                 > R[nm]["hold"][variant]["a"][tr, t]).astype(float)
                                for nm in arms3])
            f = fit_probe(X, y, hidden=probe_hidden, steps=probe_steps)
            yc = np.concatenate([np.full(len(tr), i) for i in range(3)])
            f3 = fit_probe_multi(X, yc, 3, hidden=probe_hidden, steps=probe_steps)
            for nm in ARMS:
                pay[nm][:, t] = f(R[nm]["states"][blk][:, t])
                pr = f3(R[nm]["states"][blk][:, t])
                p3[nm][:, t] = (np.log(np.clip(pr[:, 1], 1e-9, None))
                                - np.log(np.clip(pr[:, 0], 1e-9, None)))
                if nm == "quiet":
                    pq.append(float(pr[ite, 2].mean()))
            yy = np.concatenate([(R[nm]["hold"][variant]["b"][ite, t]
                                  > R[nm]["hold"][variant]["a"][ite, t]) for nm in arms3])
            a_pay.append(auc(np.concatenate([pay[nm][ite, t] for nm in arms3]), yy))
            a_p3.append(auc(np.concatenate([p3[nm][ite, t] for nm in arms3]), yy))
            yt3 = np.concatenate([np.full(len(ite), i) for i in range(3)])
            prs = np.concatenate([f3(R[nm]["states"][blk][ite, t]) for nm in arms3])
            cls_acc.append(float((prs.argmax(1) == yt3).mean()))
        for nm in ARMS:
            R[nm]["gates"][f"payoff_{gl}"] = pay[nm]
            R[nm]["gates"][f"probe3_{gl}"] = p3[nm]
        res.setdefault("new_probes", {})[gl] = {
            "payoff_auc_holdbetter_by_tau": a_pay, "probe3_auc_holdbetter_by_tau": a_p3,
            "probe3_3way_accuracy_by_tau": cls_acc, "probe3_p_quiet_on_quiet_by_tau": pq}
        print(f"  [{gl}] payoff probe AUC(hold-better) {[round(x, 3) for x in a_pay]}",
              flush=True)
        print(f"  [{gl}] probe3 3-way acc {[round(x, 3) for x in cls_acc]}  "
              f"p(quiet) on quiet {[round(x, 2) for x in pq]}", flush=True)

    # ---- addition 1: everything, on the full population and on j >= j_min ----
    subsets = {"all": np.ones(n_ev, bool), f"j{j_min}+": meta["j"] >= j_min}
    res["subsets"] = {k: int(m.sum()) for k, m in subsets.items()}
    res["by_subset"] = {}
    res["stake"] = {}
    for sname, sel in subsets.items():
        block = {"n_events": int(sel.sum()), "separation": {}, "matched_guard": {}, "mixed": {},
                 "kstar_hist": np.bincount(meta["kstar"][sel], minlength=7).tolist(),
                 "j_hist": np.bincount(meta["j"][sel], minlength=5).tolist()}
        res["stake"][sname] = {}
        for gl in GLITCHES:
            # A6 first: the stake gates must exist before A5's table is built over every gate
            res["stake"][sname][gl] = stake_probes(R, gl, sel, split, H, blk, variant,
                                                   mlp_hidden=probe_hidden, seed=seed)
            sr = res["stake"][sname][gl]["by_tau"]
            print(f"  [stake {sname}/{gl}] R2 ridge/mlp/logbf-rescaled  " + "  ".join(
                f"t{d['tau']} {d['ridge_R2']:+.3f}/{d['mlp_R2']:+.3f}/"
                f"{d['logbf_rescaled_R2']:+.3f}" for d in sr[:4]), flush=True)
            print(f"  [stake {sname}/{gl}] rho ridge/mlp/logbf  " + "  ".join(
                f"t{d['tau']} {d['ridge_spearman']:+.3f}/{d['mlp_spearman']:+.3f}/"
                f"{d['logbf_spearman']:+.3f}" for d in sr[:4]), flush=True)
            sep, sc = separation(R, gl, sel, split, H, probe_steps, probe_hidden, blk,
                                 variant)
            for nm in ARMS:
                R[nm]["gates"][f"probe_{gl}"] = sc[nm]
            # shuffles, rebuilt after every gate exists, over the whole event set
            perm = np.random.default_rng(seed + 101).permutation(len(ARMS) * n_ev)
            for g in (list(SCALAR_GATES) + [f"payoff_{gl}", f"probe3_{gl}", f"probe_{gl}",
                                            f"stake_ridge_{gl}", f"stake_mlp_{gl}"]):
                st = np.concatenate([R[nm]["gates"][g] for nm in ARMS])[perm]
                for i, nm in enumerate(ARMS):
                    R[nm]["gates"][f"shuf_{g}"] = st[i * n_ev:(i + 1) * n_ev]
            for nm, lab in (("edit", 1.0), ("glitch_m", 0.0), ("glitch_s", 0.0),
                            ("quiet", 0.5)):
                R[nm]["gates"]["label"] = np.full((n_ev, H + 1), lab)
            sep[f"probe_{gl}"] = dict(sep["probe"])
            block["separation"][gl] = sep
            pv, pc = matched_pairs(R, gl, sel, split, caliper,
                                   np.random.default_rng(seed + 3))
            block["matched_guard"][gl] = matched_sep(R, gl, pv, pc, H, caliper)
            block["mixed"][gl] = mixed_eval(R, variant, gl, sel, split, (1 / 3, 1 / 3, 1 / 3),
                                            gate_names(gl), H)
            m = block["mixed"][gl]
            bf, lab_ = m["fixed"]["best_fixed_fit"]["nll"], m["gated"]["label"]["nll"]
            print(f"  [{sname}/{gl}] n_te/arm {m['n_test_per_arm']}  native "
                  f"{m['fixed']['native']['nll']:.4f}  best_fixed {bf:.4f}  label {lab_:.4f}  "
                  f"(prize {bf - lab_:+.4f})", flush=True)
            for g in gate_names(gl):
                if g.startswith("shuf_") or g not in m["gated"]:
                    continue
                sh = m["gated"].get("shuf_" + g, {}).get("nll")
                d = f"{m['gated'][g]['nll'] - sh:+.4f}" if sh is not None else "n/a"
                w0 = m["gated"][g]["w_tau0_by_world"]
                print(f"      gate {g:18s} {m['gated'][g]['nll']:.4f}  vs shuffle {d}  "
                      f"w0 e/g/q {w0['edit']:.3f}/{w0[gl]:.3f}/{w0['quiet']:.3f}", flush=True)
            if sname == "all":
                pb = {}
                ite = np.where(~split)[0]
                for key in blocks:
                    trm = np.where(split)[0]
                    f = fit_probe(np.concatenate([pre["edit"]["state"][key][trm],
                                                  pre[gl]["state"][key][trm]]),
                                  np.concatenate([np.ones(len(trm)), np.zeros(len(trm))]),
                                  hidden=(probe_hidden if key == blk else 0),
                                  steps=probe_steps)
                    se, sg = f(pre["edit"]["state"][key]), f(pre[gl]["state"][key])
                    g_ = ite[gt[ite]]
                    e_ = ite[~gt[ite]]
                    pb[key] = {"auc_tm1_all": _auc2(se[ite], sg[ite]),
                               "auc_tm1_tv_gt_e": _auc2(se[g_], sg[g_]),
                               "auc_tm1_tv_eq_e": _auc2(se[e_], sg[e_]),
                               "n_gt": int(len(g_)), "n_eq": int(len(e_)),
                               "auc_tm1_matched": _auc2(se[pv], sg[pc]),
                               "n_matched": int(len(pv)),
                               "auc_tau0_matched_ref":
                                   block["matched_guard"][gl]["by_gate"]
                                   .get(f"probe_{gl}", [float("nan")])[0]}
                    gm = meta["delay"][pv] > 0
                    if gm.sum() > 30:
                        pb[key]["auc_tm1_matched_tv_gt_e"] = _auc2(se[pv[gm]], sg[pc[gm]])
                for k2 in ("runsurp4", "runex4", "Hq_prev"):
                    g_ = ite[gt[ite]]
                    pb[k2] = {"auc_tm1_all": _auc2(pre["edit"][k2][ite], pre[gl][k2][ite]),
                              "auc_tm1_tv_gt_e": _auc2(pre["edit"][k2][g_], pre[gl][k2][g_]),
                              "auc_tm1_matched": _auc2(pre["edit"][k2][pv], pre[gl][k2][pc]),
                              "mean_edit": float(pre["edit"][k2][ite].mean()),
                              "mean_glitch": float(pre[gl][k2][ite].mean())}
                res["preamble"]["by_glitch"][gl] = pb
                print(f"  [preamble/{gl}] state probe at t_v-1: all {pb[blk]['auc_tm1_all']:.3f}"
                      f"  t_v>e {pb[blk]['auc_tm1_tv_gt_e']:.3f}  t_v==e "
                      f"{pb[blk]['auc_tm1_tv_eq_e']:.3f}  matched "
                      f"{pb[blk]['auc_tm1_matched']:.3f}  (tau=0 matched ref "
                      f"{pb[blk]['auc_tau0_matched_ref']:.3f})  |  running surprisal "
                      f"{pb['runsurp4']['auc_tm1_all']:.3f}", flush=True)
        res["by_subset"][sname] = block

    # A1-A4 must be unchanged: this pass only adds gates and columns. Reported, not assumed.
    prev_path = f"{ckpt[:-3]}_basalis_gates2{('_' + tag) if tag else ''}.json"
    try:
        with open(prev_path) as f:
            old = json.load(f)
        worst_k, worst_v = "", 0.0
        for k1 in ("by_subset", "new_probes", "preamble"):
            for path, a_, b_ in _common_scalars(old.get(k1), res.get(k1), k1):
                d_ = abs(a_ - b_)
                if d_ > worst_v:
                    worst_k, worst_v = path, d_
        res["regression_check"] = {"against": prev_path, "max_abs_diff": worst_v,
                                   "worst_path": worst_k}
        print(f"  regression check vs the previous artifact: max |diff| {worst_v:.2e} "
              f"at {worst_k}", flush=True)
    except FileNotFoundError:
        res["regression_check"] = {"against": prev_path, "max_abs_diff": None}

    name = f"{ckpt[:-3]}_basalis_gates2{('_' + tag) if tag else ''}.json"
    with open(name, "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {name}",
          flush=True)
    return {"ckpt": ckpt, "step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def gates2_sweep(ckpts: str = "", head_tags: str = "", tags: str = "", n_events: int = 6000,
                 j_min: int = 2, probe_steps: int = 500):
    volume.reload()
    d = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    cs = ckpts.split(",") if ckpts else [f"{d}/traj_eps01_s42/step064000.pt",
                                         f"{d}/traj_a1_s42/step064000.pt"]
    hs = head_tags.split(",") if head_tags else ["" if "eps01" in c else "onnoise" for c in cs]
    ts = tags.split(",") if tags else [""] * len(cs)
    return list(gates2_ckpt.starmap(list(zip(cs, hs, ts)),
                                    kwargs={"n_events": n_events, "j_min": j_min,
                                            "probe_steps": probe_steps}))
