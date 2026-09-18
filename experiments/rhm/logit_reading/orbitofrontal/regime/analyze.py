"""regime/analyze.py -- the local reduction: every cell's json + per-event npz ->
`results/tables.md` and `figs/`.

Facts only; the interpretation waits on a discussion.

Run (from `experiments/`, after pulling the artefacts into a local mirror):
  python -m rhm.logit_reading.orbitofrontal.regime.analyze --root <mirror> \
      --out rhm/logit_reading/orbitofrontal/regime/results \
      --figs rhm/logit_reading/orbitofrontal/regime/figs
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.violation import auc
from rhm.logit_reading.striatum.task import match_nn
from rhm.logit_reading.phasic import balance_tokens, balance_ds_sign

LEVELS = [1, 2, 3, 4]
A_LIST = [0, 1, 2, 4, 8]
CAL = 0.30


def _f(x, k=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"


def corr(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pcorr(x, y, z):
    """Partial correlation of x and y given z (one or more columns)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    Z = np.asarray(z, np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 30:
        return None
    Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    b, *_ = np.linalg.lstsq(Zb, np.stack([x, y], 1), rcond=None)
    r = np.stack([x, y], 1) - Zb @ b
    return corr(r[:, 0], r[:, 1])


def slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20 or x.std() < 1e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


def match_pairs(keys, y, x, caliper, rng, random_pair=False):
    """1:1 matching inside exact strata of `keys`.  `random_pair` ignores `x` and pairs at
    random (the junction Q1a recipe, which leaves the model's surprisal in as the
    input-level cue a state readout could use); otherwise nearest-neighbour on `x` within
    `caliper` (the surprisal-matched recipe).  Returns (pos_idx, neg_idx)."""
    ps, ns = [], []
    for kk in np.unique(keys):
        pos = np.where((keys == kk) & y)[0]
        neg = np.where((keys == kk) & ~y)[0]
        if not len(pos) or not len(neg):
            continue
        if random_pair:
            k = min(len(pos), len(neg))
            ps.extend(rng.permutation(pos)[:k])
            ns.extend(rng.permutation(neg)[:k])
            continue
        pos = pos[np.argsort(x[pos])]
        left = list(neg[np.argsort(x[neg])])
        for pp in pos:
            if not left:
                break
            arr = np.asarray(left)
            dd = np.abs(x[arr] - x[pp])
            k = int(np.argmin(dd))
            if dd[k] <= caliper:
                ps.append(pp)
                ns.append(left.pop(k))
    return np.asarray(ps, np.int64), np.asarray(ns, np.int64)


def pair_auc(score, ps, ns):
    """AUC over the matched union, positives = the `ps` rows."""
    idx = np.concatenate([ps, ns])
    y = np.concatenate([np.ones(len(ps), bool), np.zeros(len(ns), bool)])
    return auc(np.asarray(score)[idx], y)


def mse(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1) / max(np.sqrt(len(x)), 1)), len(x)) \
        if len(x) > 1 else (None, None, len(x))


# ---------------------------------------------------------------------------

def load_cells(root, tag=""):
    cells = {}
    pat = os.path.join(root, "traj_regime_*", "step*_regime_*.json")
    for jp in sorted(glob.glob(pat)):
        npz = jp[:-5] + ".npz"
        if not os.path.exists(npz):
            print(f"  (no npz for {jp})")
            continue
        J = json.load(open(jp))
        if tag and not jp.endswith(f"_{tag}.json"):
            continue
        Z = np.load(npz)
        R = {k: Z[k] for k in Z.files}
        cells[(J["world"], int(J["step"]))] = (J, R)
    return cells


def ev_masks(R):
    te = R["split"] == 2
    cm = te & (R["kind"] == 1)
    qm = te & (R["kind"] == 0)
    lp = cm & (R["legal_prefix"] == 1)
    return te, cm, qm, lp


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def tab_world(out, wjson):
    out.append("## 0. The two worlds\n")
    out.append("| field | " + " | ".join(wjson) + " |")
    out.append("|---|" + "---|" * len(wjson))
    keys = ["p_noisy_realised", "fire_rate", "change_rate", "illegal_rate_at_change",
            "events_per_window", "filter_auc_regime", "llr_mean_illegal",
            "llr_mean_legal_tok", "db_mean_at_event", "dlogbf_mean_at_event"]
    for k in keys:
        out.append(f"| {k} | " + " | ".join(_f(wjson[w].get(k), 4) for w in wjson) + " |")
    for k in ("dwell_c", "dwell_n", "eps_c", "eps_n", "p_noisy", "eps_mean"):
        out.append(f"| param {k} | " +
                   " | ".join(_f(wjson[w]["params"].get(k), 4) for w in wjson) + " |")
    out.append("| change rate clean / noisy | " + " | ".join(
        "/".join(_f(x, 4) for x in wjson[w]["change_rate_by_regime"]) for w in wjson) + " |")
    out.append("| b_fwd mean clean / noisy | " + " | ".join(
        "/".join(_f(x, 3) for x in wjson[w]["b_fwd_mean_by_regime"]) for w in wjson) + " |")
    out.append("")


def tab_headline(out, cells, T=64, dmax=12):
    out.append("## A. The contrast at a glance\n")
    out.append("`burst` and `iid` differ only in whether the same average corruption comes "
               "in bursts.  Every column is on held-out windows; `ahead` rows are restricted "
               "to events with a full horizon; the legality row is panel 7b (surprisal "
               "matched, guard 0.50).\n")
    out.append("| reading | burst 0 | burst 8k | burst 24k | burst 64k | iid 0 | iid 8k | "
               "iid 24k | iid 64k |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    order = [("burst", s) for s in (0, 8000, 24000, 64000)] + \
            [("iid", s) for s in (0, 8000, 24000, 64000)]

    def row(name, fn, k=3):
        vals = []
        for key in order:
            if key not in cells:
                vals.append("--")
                continue
            J, R = cells[key]
            try:
                vals.append(_f(fn(J, R), k))
            except Exception:
                vals.append("--")
        out.append(f"| {name} | " + " | ".join(vals) + " |")

    row("regime probe on the state (AUC, every position)",
        lambda J, R: J["probe_pos"]["post_block7"].get("regime_auc"))
    row("the filter's own belief in the state (R^2)",
        lambda J, R: J["probe_pos"]["post_block7"].get("bfwd_r2"))
    row("corr(R[l2, a=2], db) -- the revision against the filter's",
        lambda J, R: corr(R["db"][ev_masks(R)[1]], R["R_l2_a2"][ev_masks(R)[1]]))
    row("  the same, given the model's own surprisal",
        lambda J, R: pcorr(R["R_l2_a2"][ev_masks(R)[1]], R["db"][ev_masks(R)[1]],
                           R["nll"][ev_masks(R)[1]]))
    row("  the shuffled-outcome critic's revision",
        lambda J, R: corr(R["db"][ev_masks(R)[1]], R["Rsh_l2_a2"][ev_masks(R)[1]]))
    row("AUC(-V[l2], corruption ahead >= 1)",
        lambda J, R: auc(-R["V_l2_a0"][ev_masks(R)[0] & (R["t"] + dmax <= T - 1)],
                         R["n_corrupt_ahead"][ev_masks(R)[0] & (R["t"] + dmax <= T - 1)] >= 1))
    row("AUC(-V[l2], the true regime)",
        lambda J, R: auc(-R["V_l2_a0"][ev_masks(R)[0]], R["regime"][ev_masks(R)[0]] == 1))
    row("dV(natural twin) at l = 2, offset 8",
        lambda J, R: mse(R["dVnat_l2_t8"][(R["split"] == 2) & (R["has_nat"] == 1)
                                          & (R["tmax_ok"] == 1)])[0], 4)
    row("the model's entropy change at a corrupted token",
        lambda J, R: J["census"].get("dH_corrupt"), 4)
    out.append("")


def tab_trunks(out, cells):
    out.append("## 1. The trunks and the actor\n")
    out.append("### Actor clean held-out accuracy on `post_block7` (chance 0.0625)\n")
    out.append("| world | step | l=1 | l=2 | l=3 | l=4 | l=5 | l=6 | emb l1 | emb l3 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, _) in sorted(cells.items()):
        a = J["actor_clean_acc"]
        out.append(f"| {w} | {st} | " +
                   " | ".join(_f(a[f"post_block7/l{l}"]) for l in range(1, 7)) +
                   f" | {_f(a['post_embed/l1'])} | {_f(a['post_embed/l3'])} |")
    out.append("")
    out.append("### The critic: held-out R^2 of V[l, d] (`post_block7`), shuffled-outcome "
               "floor and clean-only critic in brackets\n")
    out.append("| world | step | l1 d0 | l2 d0 | l3 d0 | l4 d0 | l2 d4 | l2 d8 | "
               "shuf l2 d0 | clean l2 d0 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, _) in sorted(cells.items()):
        c = J["critic_val_r2"]
        out.append(f"| {w} | {st} | " +
                   " | ".join(_f(c[f"post_block7/l{l}_d0"]) for l in (1, 2, 3, 4)) +
                   f" | {_f(c['post_block7/l2_d4'])} | {_f(c['post_block7/l2_d8'])}"
                   f" | {_f(c['post_block7/sh_l2_d0'])}"
                   f" | {_f(J['clean_critic_val_r2']['l2_d0'])} |")
    out.append("")


def tab_regime_in_state(out, cells):
    out.append("## 2. Is the regime in the state?\n")
    out.append("Ridge probe on the residual stream at every position `t` in [8, 50] of the "
               "held-out windows, split in half for fit and test.  `regime AUC` is for the "
               "TRUE regime; `b_fwd R^2` is for the exact filter's forward belief.\n")
    out.append("| world | step | block7 regime AUC | block7 b_fwd R^2 | block7 b_fwd corr | "
               "embed regime AUC | event-state regime AUC | event-state 'corruption ahead' AUC |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (w, st), (J, _) in sorted(cells.items()):
        p = J["probe_pos"]
        h = J["headline"]
        out.append(f"| {w} | {st} | {_f(p['post_block7'].get('regime_auc'))} | "
                   f"{_f(p['post_block7'].get('bfwd_r2'))} | "
                   f"{_f(p['post_block7'].get('bfwd_corr'))} | "
                   f"{_f(p['post_embed'].get('regime_auc'))} | "
                   f"{_f(h.get('probe_regime_auc_event'))} | "
                   f"{_f(h.get('probe_ahead_auc2'))} |")
    out.append("")


def tab_events(out, cells):
    out.append("## 3. The events\n")
    out.append("| world | step | n corrupt | n quiet | illegal | illegal (legal prefix) | "
               "legal-prefix rate | regime at event | b_fwd_pre corrupt / quiet | "
               "nll corrupt / quiet | dH corrupt / quiet |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, _) in sorted(cells.items()):
        c = J["census"]
        out.append(f"| {w} | {st} | {J['n_corrupt']} | {J['n_quiet']} | "
                   f"{_f(c['illegal_rate'])} | {_f(c.get('illegal_rate_lp'))} | "
                   f"{_f(c['legal_prefix_rate'])} | {_f(c['regime_at_event'])} | "
                   f"{_f(c['b_fwd_pre_mean_corrupt'])} / {_f(c['b_fwd_pre_mean_quiet'])} | "
                   f"{_f(c.get('nll_corrupt'), 2)} / {_f(c.get('nll_quiet'), 2)} | "
                   f"{_f(c.get('dH_corrupt'))} / {_f(c.get('dH_quiet'))} |")
    out.append("")


def tab_revision(out, cells):
    out.append("## 4. Does the critic's revision track the regime revision?\n")
    out.append("Corrupted events on test windows.  `R = V[l,a](s_t) - V[l,a+1](s_{t-1})`; "
               "`db` is the filter's revision of `P(next token's regime = noisy)`, `llr` the "
               "token's own log likelihood ratio, `dlogbf` the revision in log-odds.  "
               "`R|nll` is the partial correlation of `R` with `db` given the model's own "
               "surprisal at the token.  Floors: the shuffled-outcome critic's revision and "
               "the clean-only critic's.\n")
    for a in A_LIST:
        out.append(f"### a = {a}\n")
        out.append("| world | step | l | n | corr(R, db) | corr(R, dlogbf) | corr(R, llr) | "
                   "partial corr(R, db \\| nll) | corr(Rsh, db) | corr(Rclean, db) | "
                   "corr(Remb, db) | corr(nll, db) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (w, st), (J, R) in sorted(cells.items()):
            _, cm, _, _ = ev_masks(R)
            for l in LEVELS:
                k = f"l{l}_a{a}"
                if f"R_{k}" not in R:
                    continue
                out.append(
                    f"| {w} | {st} | {l} | {int(cm.sum())} | "
                    f"{_f(corr(R['db'][cm], R[f'R_{k}'][cm]))} | "
                    f"{_f(corr(R['dlogbf'][cm], R[f'R_{k}'][cm]))} | "
                    f"{_f(corr(R['llr'][cm], R[f'R_{k}'][cm]))} | "
                    f"{_f(pcorr(R[f'R_{k}'][cm], R['db'][cm], R['nll'][cm]))} | "
                    f"{_f(corr(R['db'][cm], R[f'Rsh_{k}'][cm]))} | "
                    f"{_f(corr(R['db'][cm], R[f'Rclean_{k}'][cm]))} | "
                    f"{_f(corr(R['db'][cm], R[f'Remb_{k}'][cm]))} | "
                    f"{_f(corr(R['db'][cm], R['nll'][cm]))} |")
        out.append("")


def tab_expectation(out, cells, llr_lo=3.0):
    out.append("## 5. Weighted by expectation: the revision inside terciles of the "
               "pre-event belief\n")
    out.append(f"Corrupted test events whose token carries strong regime evidence "
               f"(`llr >= {llr_lo}`, i.e. illegal under the filter's own k = 5 observer), "
               "binned by `b_fwd_pre = P(this token's regime = noisy | prefix)`.  Inside "
               "this set the token's OWN evidence `llr` is nearly constant, so what varies "
               "across the bins is what the reader already believed -- and `db`, the "
               "revision that belief allows.  `R` is the critic's revision at l = 2.\n")
    out.append("| world | step | bin | n | b_fwd_pre | db | dlogbf | llr | R (a=0) | "
               "R (a=2) | R (a=8) | V (a=2) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, R) in sorted(cells.items()):
        _, cm, _, _ = ev_masks(R)
        m = cm & (R["llr"] >= llr_lo)
        if m.sum() < 90:
            continue
        b = R["b_fwd_pre"][m]
        idx = np.where(m)[0]
        if b.std() < 1e-9:
            bins = [np.ones(len(idx), bool)]
        else:
            q = np.quantile(b, [1 / 3, 2 / 3])
            bins = [b <= q[0], (b > q[0]) & (b <= q[1]), b > q[1]]
        for bi, bm in enumerate(bins):
            sel = idx[bm]
            if len(sel) < 30:
                continue
            out.append(f"| {w} | {st} | {bi} | {len(sel)} | {_f(R['b_fwd_pre'][sel].mean())} | "
                       f"{_f(R['db'][sel].mean())} | {_f(R['dlogbf'][sel].mean(), 2)} | "
                       f"{_f(R['llr'][sel].mean(), 2)} | "
                       f"{_f(R['R_l2_a0'][sel].mean(), 4)} | {_f(R['R_l2_a2'][sel].mean(), 4)} | "
                       f"{_f(R['R_l2_a8'][sel].mean(), 4)} | {_f(R['V_l2_a2'][sel].mean(), 4)} |")
    out.append("")


def tab_ahead(out, cells, T=64, dmax=12):
    out.append("## 6. The cross-world reading: does the value level see the corruption "
               "that has not happened yet?\n")
    out.append("`ahead` = the number of corrupted tokens in `(t, t+12]`, the critic's own "
               "horizon.  Defined in both worlds; predictable from the past only in `burst`.  "
               "Restricted to events whose whole horizon fits inside the window "
               f"(`t + {dmax} <= {T - 1}`), because otherwise the count is truncated by "
               "position and the state knows the position.\n")
    out.append("| world | step | n | corr(V, ahead) l2 | AUC(-V, ahead>=1) | "
               "AUC(-V, ahead>=2) | AUC(-Vpre, ahead>=2) | oracle probe AUC | "
               "AUC(-V, regime) | AUC(-Vpre, regime) | AUC(-V, regime) quiet events |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, R) in sorted(cells.items()):
        te, cm, qm, _ = ev_masks(R)
        full = R["t"] + dmax <= T - 1
        c, tq = cm & full, te & full
        qq = qm & full
        out.append(f"| {w} | {st} | {int(c.sum())} | "
                   f"{_f(corr(R['n_corrupt_ahead'][c], R['V_l2_a0'][c]))} | "
                   f"{_f(auc(-R['V_l2_a0'][tq], R['n_corrupt_ahead'][tq] >= 1))} | "
                   f"{_f(auc(-R['V_l2_a0'][tq], R['n_corrupt_ahead'][tq] >= 2))} | "
                   f"{_f(auc(-R['Vpre_l2_a0'][tq], R['n_corrupt_ahead'][tq] >= 2))} | "
                   f"{_f(auc(R['probe_ahead'][tq], R['n_corrupt_ahead'][tq] >= 2)) if 'probe_ahead' in R else '--'} | "
                   f"{_f(auc(-R['V_l2_a0'][tq], R['regime'][tq] == 1))} | "
                   f"{_f(auc(-R['Vpre_l2_a0'][tq], R['regime'][tq] == 1))} | "
                   f"{_f(auc(-R['V_l2_a0'][qq], R['regime'][qq] == 1))} |")
    out.append("")


def tab_legality(out, cells):
    out.append("## 7. The junction question: legality at matched immediate consequence\n")
    out.append("Corrupted events with a LEGAL PREFIX (so `p_L` is defined at `t`), test "
               "windows.  Illegal vs legal corrupted token, exact strata on (immediate "
               "damage at this level from the natural twin, window-position bucket of 8).  "
               "**Panel a** pairs at random inside those strata and leaves the model's "
               "surprisal in as the input-level cue a state readout could use (junction's "
               "Q1a recipe); **panel b** adds 1:1 nearest-neighbour matching on that "
               "surprisal within a caliper of 0.3, with the sign balanced inside |ds| bands, "
               "which pins the guard at 0.5 and costs most of the rows.  Every score is "
               "oriented so that a higher value means 'illegal'.\n")
    rng = np.random.default_rng(0)
    for panel, rand in (("a -- strata only, surprisal left in", True),
                        ("b -- surprisal matched and sign-balanced", False)):
        out.append(f"### Panel {panel}\n")
        out.append("| world | step | l | n matched | AUC(-R) | AUC(-V) | AUC(-Rclean) | "
                   "AUC(-Rsh) | oracle legality probe | guard/cue: nll | guard: position | "
                   "**warrant**: E[dmg\\|illegal] - E[dmg\\|legal] at a=2 | at a=8 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (w, st), (J, R) in sorted(cells.items()):
            _, cm, _, lp = ev_masks(R)
            idx = np.where(lp)[0]
            if len(idx) < 100:
                continue
            ill = R["illegal"][idx] == 1
            posb = (R["t"][idx] // 8).astype(np.int64)
            for l in LEVELS:
                dmg = (R[f"onat_l{l}_a0"][idx].astype(np.int16)
                       - R[f"ox_l{l}_a0"][idx].astype(np.int16))
                keys = posb * 10 + (dmg + 1)
                ps, ns = match_pairs(keys, ill, R["nll"][idx], CAL, rng, random_pair=rand)
                if len(ps) >= 20 and not rand:
                    kp = balance_ds_sign(R["nll"][idx][ps] - R["nll"][idx][ns],
                                         np.arange(len(ps)), rng)
                    if len(kp) >= 20:
                        ps, ns = ps[kp], ns[kp]
                if len(ps) < 20:
                    continue
                P, N = idx[ps], idx[ns]
                row = [f"{w}", f"{st}", f"{l}", f"{2 * len(P)}",
                       _f(pair_auc(-R[f"R_l{l}_a0"], P, N)),
                       _f(pair_auc(-R[f"V_l{l}_a0"], P, N)),
                       _f(pair_auc(-R[f"Rclean_l{l}_a0"], P, N)),
                       _f(pair_auc(-R[f"Rsh_l{l}_a0"], P, N)),
                       _f(pair_auc(R["probe_illegal"], P, N)) if "probe_illegal" in R else "--",
                       _f(pair_auc(R["nll"], P, N)),
                       _f(pair_auc(R["t"].astype(np.float64), P, N))]
                for aa in (2, 8):
                    dg = (R[f"onat_l{l}_a{aa}"].astype(np.int16)
                          - R[f"ox_l{l}_a{aa}"].astype(np.int16)).astype(np.float64)
                    ok = (R[f"ok_l{l}_a{aa}"] == 1)
                    d = dg[P][ok[P] & ok[N]] - dg[N][ok[P] & ok[N]]
                    mu, se, nn2 = mse(d)
                    row.append(f"{_f(mu, 4)} +- {_f(se, 4)}" if mu is not None else "--")
                out.append("| " + " | ".join(row) + " |")
        out.append("")


def tab_warrant(out, cells):
    out.append("## 7c. What a violation actually costs, and what the critic says it costs\n")
    out.append("Corrupted events with a legal prefix, test windows, l = 2, no matching -- "
               "the raw contrast the matched panels above are a controlled version of.  "
               "`damage` is `o(natural twin) - o(this window)`: what THIS corruption did at "
               "the goal.  `outcome` is the actor's realised accuracy on the corrupted "
               "window.  `V` is the critic's prediction of that outcome and `R` its revision "
               "at the token.\n")
    out.append("| world | step | a | n illegal / legal | damage illegal | damage legal | "
               "outcome illegal | outcome legal | V illegal | V legal | R illegal | R legal |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (w, st), (J, R) in sorted(cells.items()):
        if st not in (0, max(s for _, s in cells)):
            continue
        _, cm, _, lp = ev_masks(R)
        ill = R["illegal"] == 1
        for a in A_LIST:
            ok = (R[f"ok_l2_a{a}"] == 1) & (R["has_nat"] == 1)
            A_, B_ = lp & ok & ill, lp & ok & ~ill
            if A_.sum() < 50 or B_.sum() < 50:
                continue
            dm = (R[f"onat_l2_a{a}"].astype(np.int16)
                  - R[f"ox_l2_a{a}"].astype(np.int16)).astype(np.float64)
            out.append(
                f"| {w} | {st} | {a} | {int(A_.sum())} / {int(B_.sum())} | "
                f"{_f(dm[A_].mean(), 4)} | {_f(dm[B_].mean(), 4)} | "
                f"{_f(R[f'ox_l2_a{a}'][A_].mean())} | {_f(R[f'ox_l2_a{a}'][B_].mean())} | "
                f"{_f(R[f'V_l2_a{a}'][A_].mean())} | {_f(R[f'V_l2_a{a}'][B_].mean())} | "
                f"{_f(R[f'R_l2_a{a}'][A_].mean(), 4)} | {_f(R[f'R_l2_a{a}'][B_].mean(), 4)} |")
    out.append("")


def tab_twins(out, cells):
    out.append("## 8. The twins\n")
    out.append("`dV = V(the window as it is) - V(the twin)` at offset 0, the only clean "
               "event reading (after `t` the two windows share the same continuation).  "
               "NATURAL twin: this corruption undone, every other token kept -- available at "
               "every corrupted event.  LEGAL twin: the corrupted token replaced by the legal "
               "token of nearest model surprisal, every pair the picker found, token-balanced "
               "and sign-balanced on the surprisal gap (which pins the PAIRED surprisal guard "
               "at 0.5 whatever the caliper); its population is re-selected per checkpoint by "
               "the model's own surprisal, phasic's own caveat.\n")
    out.append("| world | step | twin | l | n | mean \\|ds\\| | dV (mean +- se) | sigma | "
               "dV damage-matched |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(1)
    for (w, st), (J, R) in sorted(cells.items()):
        te = R["split"] == 2
        for nm, hk, tokk in (("natural", "has_nat", "tok_clean"), ("legal", "has_leg", "leg_tok")):
            m = te & (R[hk] == 1)
            idx = np.where(m)[0]
            if len(idx) < 40:
                continue
            keep = balance_tokens(R["tok"], R[tokk], idx, rng)
            if nm == "legal" and len(keep) > 40:
                keep = balance_ds_sign(R["leg_ds"], keep, rng)
            cal_note = ("all pairs the picker found, sign-balanced on |ds|"
                        if nm == "legal" else "")
            if len(keep) < 40:
                keep = idx
            for l in LEVELS:
                k = f"dV{'nat' if nm == 'natural' else 'leg'}_l{l}_t0"
                if k not in R:
                    continue
                mu, se, nn = mse(R[k][keep])
                if mu is None:
                    continue
                dm = ((R[f"onat_l{l}_a0"][keep].astype(np.int16)
                       - R[f"ox_l{l}_a0"][keep].astype(np.int16)) == 0
                      if nm == "natural" else np.ones(len(keep), bool))
                mu2, se2, n2 = mse(R[k][keep][dm])
                ds = (float(np.abs(R["leg_ds"][keep]).mean()) if nm == "legal" else None)
                out.append(f"| {w} | {st} | {nm} | {l} | {nn} | {_f(ds, 3)} | "
                           f"{_f(mu, 4)} +- {_f(se, 4)} | "
                           f"{_f(mu / se if se else None, 1)} | "
                           f"{_f(mu2, 4)} (n={n2}) |")
    out.append("")
    out.append("### Persistence of the natural twin's value gap, l = 2 (token-balanced)\n")
    out.append("| world | step | " + " | ".join(f"t={t}" for t in range(9)) + " |")
    out.append("|---|---|" + "---|" * 9)
    for (w, st), (J, R) in sorted(cells.items()):
        te = R["split"] == 2
        idx = np.where(te & (R["has_nat"] == 1) & (R["tmax_ok"] == 1))[0]
        if len(idx) < 40:
            continue
        keep = balance_tokens(R["tok"], R["tok_clean"], idx, rng)
        if len(keep) < 40:
            keep = idx
        row = [_f(mse(R[f"dVnat_l2_t{t}"][keep])[0], 4) for t in range(9)]
        out.append(f"| {w} | {st} | " + " | ".join(row) + " |")
    out.append("")


def tab_switch(out, cells):
    out.append("## 9. Around a regime switch\n")
    out.append("Held-out windows of the burst world.  `on` = the clean -> noisy switch at "
               "tau = 0, `off` the reverse.  `surprise` is the outcome minus the critic's "
               "own prediction of it one token earlier (the Xiang currency of `norm/`).\n")
    for (w, st), (J, R) in sorted(cells.items()):
        for nm, ent in J.get("trace", {}).items():
            taus = ent["taus"]
            out.append(f"**{w} step {st}, switch `{nm}` (n = {ent['n']})**\n")
            out.append("| quantity | " + " | ".join(str(t) for t in taus) + " |")
            out.append("|---|" + "---|" * len(taus))
            for q in ("V_l2", "surprise_l2", "acc_l2", "V_l3", "surprise_l3"):
                if q in ent:
                    out.append(f"| {q} | " + " | ".join(_f(x, 4) for x in ent[q]) + " |")
            out.append("")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(cells, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figs, exist_ok=True)
    steps = sorted({st for (_, st) in cells})
    worlds = sorted({w for (w, _) in cells})
    top = max(steps)

    # 1. R against the filter's revision, binned, split by the pre-event belief
    fig, axes = plt.subplots(1, len(worlds), figsize=(5.2 * len(worlds), 4), squeeze=False)
    for wi, w in enumerate(worlds):
        ax = axes[0][wi]
        if (w, top) not in cells:
            continue
        _, R = cells[(w, top)]
        _, cm, _, _ = ev_masks(R)
        x, yv = R["db"][cm], R["R_l2_a0"][cm]
        if x.std() < 1e-9:
            ax.text(0.5, 0.5, "no regime in this world", ha="center", transform=ax.transAxes)
        else:
            q = np.quantile(x, np.linspace(0, 1, 11))
            q = np.unique(q)
            b = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
            mx = [x[b == i].mean() for i in range(len(q) - 1) if (b == i).sum() > 5]
            my = [yv[b == i].mean() for i in range(len(q) - 1) if (b == i).sum() > 5]
            se = [yv[b == i].std(ddof=1) / np.sqrt((b == i).sum())
                  for i in range(len(q) - 1) if (b == i).sum() > 5]
            ax.errorbar(mx, my, yerr=se, marker="o", color="#b4413c")
            ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{w}, step {top}")
        ax.set_xlabel("filter revision  db = dP(noisy next)")
        ax.set_ylabel("critic revision R[l=2, a=0]")
    fig.suptitle("The value revision against the regime revision, at a corrupted token")
    fig.tight_layout()
    fig.savefig(f"{figs}/regime_revision.png", dpi=150)
    plt.close(fig)

    # 2. the regime probe through training
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for w in worlds:
        xs = [st for st in steps if (w, st) in cells]
        ys = [cells[(w, st)][0]["probe_pos"]["post_block7"].get("regime_auc") for st in xs]
        ax.plot(xs, [y if y is not None else np.nan for y in ys], marker="o", label=f"{w}: regime")
        ys2 = [cells[(w, st)][0]["headline"].get("probe_ahead_auc2") for st in xs]
        ax.plot(xs, [y if y is not None else np.nan for y in ys2], marker="s", ls="--",
                label=f"{w}: corruption ahead")
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set_xscale("symlog", linthresh=250)
    ax.set_xlabel("step"); ax.set_ylabel("AUC")
    ax.set_title("What the state carries about the hidden state")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{figs}/regime_probe.png", dpi=150)
    plt.close(fig)

    # 3. the switch trace
    have = [(w, st) for (w, st) in cells if cells[(w, st)][0].get("trace")]
    if have:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for (w, st) in have:
            if st != top:
                continue
            for nm, ent in cells[(w, st)][0]["trace"].items():
                axes[0].plot(ent["taus"], ent["V_l2"], marker="o", label=f"{w} {nm}")
                axes[1].plot(ent["taus"], ent["surprise_l2"], marker="o", label=f"{w} {nm}")
        axes[0].set_title("critic level V[l=2]"); axes[1].set_title("outcome surprise")
        for a in axes:
            a.axvline(0, color="0.7", lw=0.8)
            a.set_xlabel("offset from the regime switch")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{figs}/regime_switch.png", dpi=150)
        plt.close(fig)

    # 4. the natural twin's persistence
    fig, ax = plt.subplots(figsize=(5.5, 4))
    rng = np.random.default_rng(2)
    for w in worlds:
        if (w, top) not in cells:
            continue
        _, R = cells[(w, top)]
        te = R["split"] == 2
        idx = np.where(te & (R["has_nat"] == 1) & (R["tmax_ok"] == 1))[0]
        if len(idx) < 40:
            continue
        keep = balance_tokens(R["tok"], R["tok_clean"], idx, rng)
        if len(keep) < 40:
            keep = idx
        ys = [mse(R[f"dVnat_l2_t{t}"][keep])[0] for t in range(9)]
        es = [mse(R[f"dVnat_l2_t{t}"][keep])[1] for t in range(9)]
        ax.errorbar(range(9), ys, yerr=es, marker="o", label=w)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("offset after the corrupted token"); ax.set_ylabel("dV = V(corrupt) - V(twin)")
    ax.set_title("What the corruption did to the value, against its natural twin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{figs}/regime_twin.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="rhm/logit_reading/orbitofrontal/regime/results")
    ap.add_argument("--figs", default="rhm/logit_reading/orbitofrontal/regime/figs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    cells = load_cells(a.root, a.tag)
    print(f"{len(cells)} cells: {sorted(cells)}")
    wjson = {}
    for wp in sorted(glob.glob(os.path.join(a.root, "regime_world_*.json"))):
        j = json.load(open(wp))
        wjson[j["world"]] = j
    out = ["# regime -- tables", "",
           "Facts only; the interpretation waits on a discussion.  Regenerated by",
           "`python -m rhm.logit_reading.orbitofrontal.regime.analyze --root <mirror>`.", "",
           "## Reproduction", "", "```bash", "cd experiments            # MODAL_PROFILE=chromatic",
           "# the whole node in one detached launch: wave 1 builds each world's venue and",
           "# trains its trunk (4 containers), wave 2 runs the readout cell for every",
           "# (world, checkpoint).",
           "modal run --detach -m rhm.logit_reading.orbitofrontal.regime.task::regime_all \\",
           "    --worlds burst,iid --n 65536 --steps 64000 --read-steps 0,8000,24000,64000",
           "# the tables and figures (local, after pulling the artefacts)",
           "bash rhm/logit_reading/orbitofrontal/regime/results/fetch.sh  <mirror>",
           "bash rhm/logit_reading/orbitofrontal/regime/results/reduce.sh <mirror>", "```", "",
           "Measured on an L4 (`chromatic`): the venue 1750 s per world (peak RSS 5.9 GB, "
           "dominated by `p_L` at `k = 6` on 65,536 windows), the trajectory 2370 s per world "
           "(5.6 GB), a readout cell 140-185 s (8.6 GB).  Wall clock for the whole node, four "
           "containers in wave 1 and four at a time in wave 2: about 80 minutes.", ""]
    tab_headline(out, cells)
    if wjson:
        tab_world(out, wjson)
    tab_trunks(out, cells)
    tab_regime_in_state(out, cells)
    tab_events(out, cells)
    tab_revision(out, cells)
    tab_expectation(out, cells)
    tab_ahead(out, cells)
    tab_legality(out, cells)
    tab_warrant(out, cells)
    tab_twins(out, cells)
    tab_switch(out, cells)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {os.path.join(a.out, 'tables.md')}")
    if not a.no_figs:
        figures(cells, a.figs)
        print(f"wrote figures to {a.figs}")


if __name__ == "__main__":
    main()
