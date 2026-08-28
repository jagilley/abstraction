"""maestro/fit — THE OFFLINE PHASE. Fit the learned rule on the logged corpus, measure its dead
zone by the donor's own null method, and replay it counterfactually over every logged arm —
entirely before any GPU, exactly as `../conductor/floors.py` derived A1's floors before A1 ran.

WHY THE FIT IS OFFLINE, STATED PLAINLY. Both of the crank's actions are ABSORBING: a commit
freezes a table into the ratchet and an era advance moves the world's damage cell down, and
`../conductor/FILES.md` records why a reversible trial-commit is unsound here (it leaves a
permanent residue of exactly the treatment it is measuring). A run therefore contains one or two
commit events and five era exits, so WITHIN-RUN EXPLORATION OF COMMIT TIMING IS CLOSE TO
IMPOSSIBLE: a policy that commits early never observes what holding longer would have bought.
The honest form is to fit on the corpus of PREVIOUS TURNS OF THE CRANK and then run frozen —
meta-learning across turns rather than within one, which is what §2.2's outer-loop pillar reads
as anyway.

WHAT THE n ACTUALLY SUPPORTS, AND WHAT IT DOES NOT. `cd_s0` carries four DISTINCT trajectories at
matched substrate (`yoked_yield`/`yoked_endo` are bit-identical replays of their gauge arms and
are excluded rather than double-counted; `cd_ef/anchor` replays `cd_s0/anchor` at 0.000e+00 and
is the floor REFERENCE, not extra data). Between them they commit L2 at c10/16/18/49 and L3 at
c69/92-or-never and leave era 1 at c18/48/56/60. That is SIX commit events and twenty era exits.

  * A POLICY-GRADIENT fit over those action events would be estimating a return differential
    from n=6, which this corpus cannot support, and there is no reversible trial that would
    grow it.
  * What the corpus DOES support densely is the VALUE: every arm logs its gauge readings and
    its next-level yield every cycle, so "what forward reward rate does this gauge configuration
    predict" has ~257 windows behind it, not six.

So the learned object is the VALUE — which gauge mixture tracks the reward's forward rate — and
the ACTION RULE is greedy with respect to it against a MEASURED dead zone, which is the same
device A1 used and needs no counterfactual returns. Nothing here estimates the value of ACTING.
That is the round's honest limit and it is restated in the reduction.

THE FITTED OBJECT (see `policy.LearnedPolicy` for the form it plugs into):

    Vhat(t) = sum_k a[s(t)]_k * V_k(t) / tol_k(t)      the decision statistic, in FLOOR UNITS
    theta[s]                                            its own measured dead zone
    rule:  HOLD while Vhat > theta ; ACT when Vhat <= theta

  a[s]   a unit-norm mixture over the three gauges' floor-normalised paired-interval slopes,
         per level-state bucket s = will_commit. Only the DIRECTION is learned: `Vhat` and
         `theta` are both linear in `a`, so the decision is invariant to its scale.
  V_k    THE DONOR'S ESTIMATOR, by identity — the features are read out of `LearnedPolicy`
         itself, so the fit sees exactly what the live policy will see.

THE REGRESSION, identical for both twins and differing ONLY in the target:

    target(t) = ( R(t + GAP + H) - R(t + GAP) ) / H        the forward reward RATE
      learned_yield : R = at_support one level up (`-panel['yield']`)   NEXT-LEVEL YIELD
      learned_task  : R = -(the arm's own metering error)               WITHIN-LEVEL

    a[s] = normalise( ridge( target ~ [V_ledger, V_yield, V_endo_excess] | s ) )

  GAP = 1 IS NOT A TUNED KNOB, it is the minimum gap for which the trailing window and the
  forward window SHARE NO DATA POINT. Without it the reading at t enters the features and the
  target with OPPOSITE signs, so that one point's noise manufactures a correlation on its own —
  the round-1 postmortem's hazard class (an instrument measuring itself), designed out before
  the GPU rather than diagnosed after it. `gap_sensitivity` in `fit.json` records the fits at
  GAP 0/1/2: the next-level mixture is barely sensitive (its `yield` weight is 0.946/0.944/0.917
  and it beats the thermostat's own predictor at every gap), while the WITHIN-LEVEL mixture's
  commit-bucket direction is not stable across the gap at all (`ledger` -0.843/-0.781/-0.066),
  which is itself a reading on how much signal that reward carries here.

  H = W * SPAN = 4 is the donor's block geometry, so the target is the same kind of object the
  features are. LAMBDA is chosen by LEAVE-ONE-ARM-OUT over a fixed grid, which is the only
  cross-validation this corpus admits (four trajectories = four folds) and is applied
  identically to both twins.

WINDOWS DROPPED. A window containing an era boundary or a commit cycle is a regime change and
not noise — `../conductor/floors.py`'s convention, applied to the trailing window, the gap, and
the forward window alike.

TWO WINDOW SETS, ON PURPOSE. The FIT needs a clean trailing window AND a clean forward window;
the FLOOR needs only a clean trailing window, because `Nhat` is a property of the trailing block
alone. Using the fit's (shorter) set for the floor would throw away a third of the reference
arm's windows for no reason.

Run (no GPU, no Modal, no substrate):
    cd experiments/ && PYTHONPATH=. python3 rhm/practice/maestro/fit.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from rhm.practice.maestro import policy as PO          # noqa: E402

# --------------------------------------------------------------------------- #
# the corpus and the constants, all stated before any number is computed
# --------------------------------------------------------------------------- #
CD = os.path.join(os.path.dirname(HERE), "conductor", "figures")

# The four DISTINCT trajectories in `cd_s0`. The two yoked arms are bit-identical replays of
# their gauge arms (A1's reduction §2: max|delta| 0.000e+00 over 12 series, first divergence
# None) and are excluded so the fit does not double-count them.
CORPUS = [("cd_s0", "anchor"), ("cd_s0", "outer_yield"),
          ("cd_s0", "outer_endo"), ("cd_s0", "outer_ledger")]
# the fixed-condition reference the dead zone is measured on: the SCHEDULE arm, which is
# fixed-condition by construction. This is A1's own floor reference (`cd_ef/anchor` replays it
# cross-tag at 0.000e+00 over all 116 cycles).
REFERENCE = ("cd_s0", "anchor")

GAUGES = ("ledger", "yield", "endo_excess")
SPAN, W, BURN, ALPHA = 1, 4, 4, 0.5          # the donor's block geometry, unchanged
GAP, H = 1, W * SPAN                         # see the module docstring
LAM_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
BUCKETS = ("1", "0")                         # will_commit
MAX_MACRO_LEVEL = 3
ERA_CAPS = (60, 50, 15, 12, 9)

# The measured dead zones that governed A1, reused here as the mixture's NORMALISING UNITS.
# Copied from `../conductor/conductor.py::MEASURED_FLOORS` (`cd_ef/anchor`, full config, null-
# ABBA span 1 W 4) rather than re-derived, because a mixture normalised by a different floor
# than A1's would not contain A1's thermostat as the point `a = e_yield`.
FLOORS = {"ledger": 0.02944712566990095,
          "endo_excess": 0.01091575129919287,
          "yield_by_level": {3: 0.46127129019246205, 4: 0.5166900731510206}}

# A1's own in-tag re-derivation on `cd_s0/anchor`, printed by `analyze_conductor.py` §6, used as
# gate F-1: this file's window logic must reproduce it from the raw series.
A1_INTAG = {"yield": {"n_win": 82, "sd_N": 0.98299, "v_tol": 0.49150},
            "ledger": {"n_win": 82, "sd_N": 0.05889, "v_tol": 0.02945},
            "endo_excess": {"n_win": 82, "sd_N": 0.02183, "v_tol": 0.01092}}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_arm(tag, arm, root=CD):
    """The arm's per-cycle shadow panel, plus the level state the live policy will read.

    `will_commit` is reconstructed the way the substrate computes it at the decision instant:
    block (d2) builds the panel BEFORE block (g) runs the commit, so a commit landing on cycle
    c is NOT yet in `committed` when cycle c's decision is taken -- hence `cc < cycle`, not
    `<=`. An off-by-one here would put the fit and the live policy in different states.
    """
    path = os.path.join(root, tag, arm, "results.json")
    with open(path) as fh:
        d = json.load(fh)
    pan = [dict(p) for p in d["log"]["panel"]]
    ev = d.get("events", [])
    commits = [(int(e["cycle"]), int(e["level"])) for e in ev if e["kind"] == "commit"]
    for p in pan:
        # `cd_ef` predates the excess form; it is exactly parent-minus-cell either way.
        if p.get("endo_excess") is None:
            p["endo_excess"] = (None if (p.get("endo") is None or p.get("endo_cell") is None)
                                else p["endo"] - p["endo_cell"])
        done = {lv for (cc, lv) in commits if cc < p["cycle"]}
        act = int(p["active"])
        p["will_commit"] = int(act <= MAX_MACRO_LEVEL and act not in done)
    return {"tag": tag, "arm": arm, "panel": pan,
            "commit_cycles": sorted(c for c, _ in commits),
            "commits": commits,
            "loop_actions": d.get("loop_actions", []),
            "n_cycles": len(pan)}


def _probe_policy():
    """A `LearnedPolicy` used ONLY as the feature extractor, so the fit's per-gauge V and N are
    produced by the code object the live policy runs. The mixture is a placeholder: nothing
    below reads its `V`, only `per_gauge`."""
    return PO.LearnedPolicy({"1": [1.0, 0.0, 0.0], "0": [1.0, 0.0, 0.0]},
                            {"1": 1.0, "0": 1.0}, FLOORS, gauges=GAUGES,
                            span=SPAN, W=W, burn=BURN, alpha=ALPHA)


def trace_arm(rec):
    """Per-cycle features, in the substrate's own order: reset at every era start and after
    every action the arm actually took (A1's discipline, since each changes the regime the
    gauge is measured in)."""
    pol = _probe_policy()
    rows, prev_era, c_in_era = [], None, 0
    for p in rec["panel"]:
        if p["era"] != prev_era:
            prev_era, c_in_era = p["era"], 0
            pol.reset(f"era{p['era']}")
        c_in_era += 1
        info = pol.step(p["cycle"], p)
        per = info["per_gauge"]
        rows.append({
            "cycle": int(p["cycle"]), "era": int(p["era"]), "c_in_era": c_in_era,
            "V": [per[g]["V"] for g in GAUGES], "N": [per[g]["N"] for g in GAUGES],
            "bucket": info["bucket"], "read_level": int(p["read_level"]),
            "R_yield": (None if p.get("yield") is None else -float(p["yield"])),
            "R_task": (None if p.get("ledger") is None else -float(p["ledger"])),
            "frac_era": c_in_era / ERA_CAPS[p["era"] - 1],
        })
        if int(p["cycle"]) in rec["commit_cycles"]:
            pol.reset(f"commit@{p['cycle']}")
    return rows


def _bad_cycles(rows, rec):
    """Era boundaries and commit cycles: regime changes, not noise."""
    bad, prev = set(), None
    for r in rows:
        if r["era"] != prev:
            bad.add(r["cycle"])
            prev = r["era"]
    bad |= set(rec["commit_cycles"])
    return bad


def fit_windows(rows, rec):
    """The FIT's window set: a clean trailing block, the gap, and a clean forward block."""
    bad = _bad_cycles(rows, rec)
    out = []
    for i, r in enumerate(rows):
        j = i + GAP
        if j + H >= len(rows):
            break
        if any(v is None for v in r["V"]):
            continue
        if set(range(r["cycle"] - W * SPAN, r["cycle"] + GAP + H + 1)) & bad:
            continue
        if any(rows[k][t] is None for k in (j, j + H) for t in ("R_yield", "R_task")):
            continue
        # THE TARGET IS IN THE SAME FLOOR UNITS AS THE FEATURES: the forward reward rate
        # divided by the MEASURED dead zone of the gauge that meters that reward. This is what
        # makes the degenerate point exact -- with the mixture on `yield` alone the regression's
        # coefficient is 1 and the rule is `V_yield <= tol_yield`, A1's thermostat -- and it is
        # what puts the two rewards' statistics on a common, interpretable axis: "the predicted
        # forward rate, in multiples of this currency's own measurement noise".
        ty = FLOORS["yield_by_level"][int(r["read_level"])]
        out.append({"V": np.array(r["V"], dtype=float), "bucket": r["bucket"],
                    "cycle": r["cycle"], "era": r["era"],
                    "y_yield": ((rows[j + H]["R_yield"] - rows[j]["R_yield"]) / H) / ty,
                    "y_task": ((rows[j + H]["R_task"] - rows[j]["R_task"]) / H)
                    / FLOORS["ledger"]})
    return out


def series_skip(rec):
    """Era boundaries and commit cycles as 0-based indices — A1's `_series_skip`, verbatim
    (`../conductor/analyze_conductor.py`), so this file's floor windows ARE A1's floor
    windows. Note the very first cycle is NOT a boundary: A1's loop starts the era test at
    i > 0, and reproducing its window COUNT is gate F-1."""
    pan = rec["panel"]
    commits = set(rec["commit_cycles"])
    skip = set()
    for i, p in enumerate(pan):
        if i > 0 and p["era"] != pan[i - 1]["era"]:
            skip.add(i)
        if (i + 1) in commits:
            skip.add(i)
    return skip


def mixture_series(rec, a):
    """`z(t) = sum_k a_k * g_k(t) / tol_k(t)` — the learned SYNTHETIC READ, in floor units.

    The dead zone is then measured by running the DONOR'S OWN `null_abba` over `z`. `null_abba`
    is linear in its series, so `N_z = sum_k a_k N_k / tol_k` exactly, with the donor's window
    range and the donor's alternating polarity — which is what makes `a = e_yield` return A1's
    own measured number rather than something merely close to it (gate F-1)."""
    z = []
    for p in rec["panel"]:
        acc, ok = 0.0, True
        for k, g in enumerate(GAUGES):
            v = p.get(g)
            if v is None:
                ok = False
                break
            tol = (FLOORS["yield_by_level"][int(p["read_level"])] if g == "yield"
                   else FLOORS[g])
            acc += a[k] * float(v) / tol
        z.append(acc if ok else None)
    return z


def null_by_bucket(rec, a):
    """`null_abba` over the mixture series, with each window tagged by the level-state bucket
    at its DECISION POINT (the newest index of the block, which is where the live policy would
    be standing). Same loop, same skip set, same polarity as the donor's."""
    z = mixture_series(rec, a)
    pan = rec["panel"]
    skip = series_skip(rec)
    n = len(z)
    out = []
    for t0 in range(0, n - W * SPAN):
        idx = [t0 + k * SPAN for k in range(W + 1)]
        if any(j in skip for j in idx):
            continue
        if any(z[j] is None for j in idx):
            continue
        vals = [float(z[j]) for j in idx]
        u = [vals[k] - vals[k + 1] for k in range(W)]
        letters = ("CKKC" if len(out) % 2 == 0 else "KCCK")[:W]
        uc = [u[k] for k in range(W) if letters[k] == "C"]
        uk = [u[k] for k in range(W) if letters[k] == "K"]
        if not (uc and uk):
            continue
        dec = pan[idx[-1]]
        out.append({"N": sum(uc) / len(uc) - sum(uk) / len(uk),
                    "D": sum(u) / len(u),
                    "bucket": "1" if int(dec["will_commit"]) else "0",
                    "cycle": int(dec["cycle"]), "era": int(dec["era"])})
    return out


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _ridge(X, y, lam):
    """Ridge THROUGH THE ORIGIN -- no centring and no intercept, deliberately.

    This is the one modelling choice that decides whether the round is well posed, so it is
    stated rather than buried. A centred fit with an intercept estimates the direction that
    explains the target's VARIANCE, and its prediction is `ybar + small`, whose zero point is
    arbitrary; thresholding it against a noise floor is then meaningless, and measured on this
    corpus it is worse than meaningless -- the shrunk prediction sits at 0.87 tuples/cycle and
    never comes near any floor, so the arm would never act, and the two twins' statistics sit at
    unrelated locations so they are not even being compared on the same axis.

    Fitting through the origin instead makes `c . V` an ESTIMATE OF THE FORWARD REWARD RATE
    ITSELF, so its zero IS the reward's zero and A1's semantics survive intact: act when the
    predicted rate is indistinguishable from no movement. It also keeps the degenerate point
    exact -- regressing the forward yield rate on the trailing yield rate through the origin
    returns a coefficient of 1, which is A1's thermostat."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _loao_mse(X, y, aid, lam):
    """Leave-one-ARM-out: the only cross-validation four trajectories admit. Folds are whole
    trajectories, never shuffled cycles -- windows inside one arm are serially dependent, and a
    random split would leak a neighbour of every held-out point into the training set."""
    errs = []
    for k in np.unique(aid):
        tr, te = aid != k, aid == k
        if tr.sum() < 3 * X.shape[1] or te.sum() < 2:
            continue
        c = _ridge(X[tr], y[tr], lam)
        errs.append(float(((X[te] @ c - y[te]) ** 2).mean()))
    return float(np.mean(errs)) if errs else float("nan")


def fit_direction(X, y, aid, thermostat_col):
    """Ridge through the origin, lambda by LOAO, then normalised: only the DIRECTION is the
    policy (`Vhat` and `theta` are both linear in the mixture, so the scale cancels).

    THE COMPARATOR IS REPORTED BESIDE IT. A1's thermostat is a predictor too -- it estimates the
    forward rate of the currency it reads by that currency's own trailing rate, i.e. the fixed
    unit coefficient on one column, with no parameters and so nothing to cross-validate. Its
    out-of-arm error is printed next to the fit's, which is the pre-GPU form of "what could
    learning buy": a better estimate of the same forward rate, on a held-out trajectory."""
    lam = min(LAM_GRID, key=lambda L: _loao_mse(X, y, aid, L))
    mse = _loao_mse(X, y, aid, lam)
    c = _ridge(X, y, lam)
    nrm = float(np.linalg.norm(c))
    a = (c / nrm) if nrm > 0 else c
    folds = []
    for k in np.unique(aid):
        tr = aid != k
        if tr.sum() < 3 * X.shape[1]:
            continue
        ck = _ridge(X[tr], y[tr], lam)
        nk = float(np.linalg.norm(ck))
        if nk > 0:
            folds.append(float(a @ (ck / nk)))
    ms0 = float((y ** 2).mean())                       # "the rate is zero" -- the rule's null
    therm = float(((X[:, thermostat_col] - y) ** 2).mean())
    return {"a": [float(x) for x in a], "c": [float(x) for x in c], "lam": float(lam),
            "coef_norm": nrm, "loao_mse": mse,
            "r2_vs_zero": (None if ms0 <= 0 else float(1 - mse / ms0)),
            "r2_vs_mean": (None if y.var() <= 0 else float(1 - mse / y.var())),
            "thermostat_mse": therm,
            "beats_thermostat": bool(mse < therm),
            "mse_ratio_vs_thermostat": (None if therm <= 0 else float(mse / therm)),
            "fold_cos": folds, "n": int(len(y)), "target_mean": float(y.mean()),
            "target_sd": float(y.std()),
            "phat_mean": float((X @ c).mean()), "phat_sd": float((X @ c).std()),
            "phat_min": float((X @ c).min())}


def measure_theta(a_by_bucket, ref_rec):
    """THE DEAD ZONE, by the donor's null method applied to the mixture.

    `Nhat = sum_k a_k N_k / tol_k` is the donor's own contrast carried through the mixture: on a
    fixed-condition series its true value is exactly zero under any improvement rate linear in
    time, so its live spread IS the mixture's noise. `theta = sd(Nhat)/2` is A1's `v_tol =
    sd(N)/2` at W=4, and reduces to it exactly when the mixture is one gauge (gate L-2)."""
    out = {}
    for b in BUCKETS:
        ws = null_by_bucket(ref_rec, a_by_bucket[b])
        vals = [w["N"] for w in ws if w["bucket"] == b]
        allv = [w["N"] for w in ws]
        sd = float(np.std(vals)) if len(vals) > 1 else None
        out[b] = {"n_win": len(vals), "sd_Nhat": sd,
                  "mean_Nhat": float(np.mean(vals)) if vals else None,
                  "theta": (None if sd is None else sd / 2.0),
                  "n_win_all_buckets": len(allv),
                  "theta_pooled": float(np.std(allv)) / 2.0 if len(allv) > 1 else None}
    return out


# --------------------------------------------------------------------------- #
# the counterfactual replay (the pre-GPU vetting, A1's shadow-panel pattern)
# --------------------------------------------------------------------------- #
def replay(rec, fitobj):
    """Replay a fitted policy over one logged arm's own series.

    EXACT ONLY UP TO THE FIRST DIVERGENCE. After the replayed policy's first action differs from
    the arm's own, the series it is reading was produced by a trajectory this policy would not
    have taken, so every later firing is a counterfactual and must never be read as a measured
    one -- A1's shadow-panel discipline, restated."""
    pol = PO.build_policy({"kind": "learned", "fit": fitobj, "span": SPAN, "W": W,
                           "burn": BURN, "alpha": ALPHA}, floors=FLOORS)
    fires, prev_era = [], None
    for p in rec["panel"]:
        if p["era"] != prev_era:
            prev_era = p["era"]
            pol.acted("era_start", p["cycle"], why=f"era{p['era']}")
        info = pol.step(p["cycle"], p)
        if pol.quiet:
            kind = "commit" if int(p["will_commit"]) else "advance"
            fires.append({"cycle": int(p["cycle"]), "era": int(p["era"]), "kind": kind,
                          "level": (int(p["active"]) if int(p["will_commit"]) else None),
                          "bucket": info["bucket"], "V": info["V"], "theta": info["v_tol"],
                          "v_mult": info["v_mult"]})
            pol.acted(kind, p["cycle"], why="quiet")
    own = [{"cycle": a["cycle"], "kind": a["kind"], "level": a.get("level")}
           for a in rec.get("loop_actions", []) if not a.get("cancelled")]
    first_div = None
    for i in range(max(len(fires), len(own))):
        f = fires[i] if i < len(fires) else None
        o = own[i] if i < len(own) else None
        if f is None or o is None or f["cycle"] != o["cycle"] or f["kind"] != o["kind"]:
            first_div = (f or o)["cycle"] if (f or o) else None
            break
    return {"fires": fires, "arm_own_actions": own, "first_divergence": first_div}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def gate_f1(ref_rec, verbose=True):
    """F-1: THE FLOOR MACHINERY REPRODUCES A1's OWN MEASURED NUMBERS on A1's own reference arm.

    A new statistic needs a new measured floor, but the MACHINERY that measures it must be the
    one A1 was governed by. Put the mixture on one gauge, undo the floor normalisation, and the
    result must be `analyze_conductor.py` §6's printed in-tag numbers for `cd_s0/anchor` — the
    window COUNT included, since the window set is what the two derivations must share."""
    out, ok = {}, True
    for i, g in enumerate(GAUGES):
        a = [0.0] * len(GAUGES)
        a[i] = 1.0
        ws = null_by_bucket(ref_rec, a)
        # the floor normalisation is a constant within any window that survives the skip set
        # (the yield read level changes only at the era 1->2 boundary, which is skipped), so
        # undoing it is a single multiply.
        tol = (FLOORS["yield_by_level"][3] if g == "yield" else FLOORS[g])
        vals = []
        for w in ws:
            t = tol
            if g == "yield" and w["era"] >= 2:
                t = FLOORS["yield_by_level"][4]
            vals.append(w["N"] * t)
        sd = float(np.std(vals))
        ref = A1_INTAG[g]
        rel = abs(sd - ref["sd_N"]) / ref["sd_N"]
        good = bool(len(vals) == ref["n_win"] and rel < 0.001)
        ok &= good
        out[g] = {"n_win": len(vals), "expected_n_win": ref["n_win"], "sd_N": sd,
                  "A1_sd_N": ref["sd_N"], "rel_diff": rel, "v_tol": sd / 2,
                  "A1_v_tol": ref["v_tol"], "pass": good}
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] F-1 {g:12s} n_win {len(vals)} "
                  f"(A1 {ref['n_win']})  sd(N) {sd:.5f} (A1 {ref['sd_N']:.5f})  "
                  f"rel {rel:.3%}  -> v_tol {sd / 2:.6f} (A1 {ref['v_tol']:.5f})")
    out["ALL"] = bool(ok)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(out_path=None, verbose=True):
    recs = {f"{t}/{a}": load_arm(t, a) for t, a in CORPUS}
    rows = {k: trace_arm(v) for k, v in recs.items()}
    keys = list(recs)

    if verbose:
        print("=" * 78)
        print("maestro/fit — the offline phase (no GPU, no Modal, no substrate)")
        print("=" * 78)
        print(f"corpus: {len(keys)} DISTINCT trajectories "
              f"(yoked arms excluded as bit-identical replays; cd_ef/anchor is the floor "
              f"reference, not extra data)")
        for k in keys:
            r = recs[k]
            la = [(a['kind'], a['cycle']) for a in r['loop_actions'] if not a.get('cancelled')]
            print(f"  {k:24s} {r['n_cycles']:3d} cycles  commits @ "
                  f"{r['commit_cycles']}  loop actions {la}")

    # ---- the design matrix ---------------------------------------------------------- #
    Xs, Ys, Yt, AID, BK, META = [], [], [], [], [], []
    per_arm = {}
    for i, k in enumerate(keys):
        fw = fit_windows(rows[k], recs[k])
        per_arm[k] = len(fw)
        for w in fw:
            Xs.append(w["V"])
            Ys.append(w["y_yield"])
            Yt.append(w["y_task"])
            AID.append(i)
            BK.append(w["bucket"])
            META.append((k, w["cycle"], w["era"]))
    X = np.array(Xs)
    AID = np.array(AID)
    BK = np.array(BK)
    Y = {"yield": np.array(Ys), "task": np.array(Yt)}
    if verbose:
        print(f"\nfit windows: {len(X)} total  " +
              "  ".join(f"{k.split('/')[1]}={n}" for k, n in per_arm.items()))
        print(f"  by bucket: will_commit=1 (a firing INSTALLS A TABLE) n={int((BK=='1').sum())}"
              f" ; will_commit=0 (a firing MOVES THE WORLD) n={int((BK=='0').sum())}")
        print(f"  V in floor units: mean {np.round(X.mean(0), 3)}  sd {np.round(X.std(0), 3)}"
              f"   over {GAUGES}")
        print(f"  geometry: span {SPAN} W {W} burn {BURN} alpha {ALPHA}; "
              f"target = forward rate over H={H} after GAP={GAP}")

    # ---- gate F-1 ------------------------------------------------------------------- #
    if verbose:
        print("\n" + "-" * 78)
        print("GATE F-1 — the window logic reproduces A1's own in-tag floors "
              f"({REFERENCE[0]}/{REFERENCE[1]})")
        print("-" * 78)
    ref_key = f"{REFERENCE[0]}/{REFERENCE[1]}"
    f1 = gate_f1(recs[ref_key], verbose=verbose)

    # ---- the two fits --------------------------------------------------------------- #
    fits = {}
    for reward in ("yield", "task"):
        if verbose:
            print("\n" + "-" * 78)
            print(f"FIT — reward = {reward.upper()}   "
                  + ("target R = at_support one level up (NEXT-LEVEL YIELD)" if reward == "yield"
                     else "target R = -(the arm's own metering error) (WITHIN-LEVEL)"))
            print("-" * 78)
        # the hand-written comparator's own predictor: the currency's trailing rate
        tcol = GAUGES.index("yield" if reward == "yield" else "ledger")
        mixes, diag = {}, {}
        for b in BUCKETS:
            m = BK == b
            d = fit_direction(X[m], Y[reward][m], AID[m], tcol)
            mixes[b] = d["a"]
            diag[b] = d
            if verbose:
                lbl = "installs a table" if b == "1" else "moves the world  "
                print(f"  bucket will_commit={b} ({lbl}) n={d['n']:3d}  lambda={d['lam']:<6g}")
                print(f"    a = " + "  ".join(f"{g} {x:+.4f}" for g, x in zip(GAUGES, d["a"]))
                      + f"   (raw c = {np.round(d['c'], 4)})")
                print(f"    LOAO mse {d['loao_mse']:.4f} vs the thermostat's own predictor "
                      f"({GAUGES[tcol]}) {d['thermostat_mse']:.4f}  -> ratio "
                      f"{d['mse_ratio_vs_thermostat']:.3f}  "
                      f"{'BEATS it' if d['beats_thermostat'] else 'does NOT beat it'}")
                print(f"    R2 vs 'the rate is zero' {d['r2_vs_zero']:+.3f}   "
                      f"R2 vs the mean {d['r2_vs_mean']:+.3f}   "
                      f"fold direction cos {np.round(d['fold_cos'], 3)}")
                print(f"    target mean {d['target_mean']:+.3f} sd {d['target_sd']:.3f} "
                      f"(floor units)   Vhat: mean {d['phat_mean'] / max(d['coef_norm'], 1e-9):+.3f}"
                      f" sd {d['phat_sd'] / max(d['coef_norm'], 1e-9):.3f}")
        theta = measure_theta(mixes, recs[ref_key])
        if verbose:
            for b in BUCKETS:
                t = theta[b]
                print(f"  theta[will_commit={b}] = {t['theta']:.4f} floor units  "
                      f"(sd(Nhat) {t['sd_Nhat']:.4f}, mean(Nhat) {t['mean_Nhat']:+.4f} "
                      f"-- the null construction checking itself -- over {t['n_win']} "
                      f"reference windows)")
        fits[reward] = {
            "reward": reward, "fit_id": f"maestro-fit-{reward}",
            "gauges": list(GAUGES), "buckets": list(BUCKETS),
            "mixes": mixes, "thetas": {b: theta[b]["theta"] for b in BUCKETS},
            "theta_detail": theta, "diagnostics": diag,
            "geometry": {"span": SPAN, "W": W, "burn": BURN, "alpha": ALPHA,
                         "gap": GAP, "horizon": H, "lam_grid": list(LAM_GRID)},
            "floors_used_as_units": FLOORS,
            "corpus": [f"{t}/{a}" for t, a in CORPUS],
            "reference": ref_key,
        }

    # ---- how the two fits relate, and how each relates to the thermostat ------------ #
    e_yield = np.array([0.0, 1.0, 0.0])
    rel = {}
    for b in BUCKETS:
        ay = np.array(fits["yield"]["mixes"][b])
        at = np.array(fits["task"]["mixes"][b])
        rel[b] = {"cos_yield_task": float(ay @ at),
                  "cos_yield_thermostat": float(ay @ e_yield),
                  "cos_task_thermostat": float(at @ e_yield)}
    if verbose:
        print("\n" + "-" * 78)
        print("THE CONTRAST — the two rewards' mixtures, and each against A1's thermostat")
        print("-" * 78)
        print("  A1's thermostat is the point a = e_yield = (0, 1, 0) in this same class.")
        for b in BUCKETS:
            r = rel[b]
            print(f"  will_commit={b}:  cos(yield, task) {r['cos_yield_task']:+.3f}   "
                  f"cos(yield, thermostat) {r['cos_yield_thermostat']:+.3f}   "
                  f"cos(task, thermostat) {r['cos_task_thermostat']:+.3f}")

    # ---- the counterfactual replay -------------------------------------------------- #
    if verbose:
        print("\n" + "-" * 78)
        print("COUNTERFACTUAL REPLAY — each fitted policy over every logged arm's own series")
        print("  (exact only up to first divergence; later firings are counterfactual)")
        print("-" * 78)
    replays = {}
    for reward in ("yield", "task"):
        replays[reward] = {}
        for k in keys:
            rp = replay(recs[k], fits[reward])
            replays[reward][k] = rp
            if verbose:
                fs = ", ".join(f"c{f['cycle']}/{f['kind'][:3]}"
                               + (f"L{f['level']}" if f["level"] else "")
                               + f"(e{f['era']} V/th {f['v_mult']:.2f})" for f in rp["fires"])
                print(f"  learned_{reward:5s} on {k.split('/')[1]:14s}: "
                      f"{fs if fs else '(never fires)'}")
                if k == "cd_s0/outer_yield":
                    own = ", ".join(f"c{a['cycle']}/{a['kind'][:3]}"
                                    + (f"L{a['level']}" if a.get("level") else "")
                                    for a in rp["arm_own_actions"])
                    print(f"      A1's THERMOSTAT actually did: {own}")
                    print(f"      first divergence: c{rp['first_divergence']}")
    for reward in ("yield", "task"):
        fits[reward]["replay"] = {k: replays[reward][k] for k in keys}

    # ---- is the live contrast non-vacuous? ------------------------------------------ #
    same = all(
        [f["cycle"] for f in replays["yield"][k]["fires"]]
        == [f["cycle"] for f in replays["task"][k]["fires"]] for k in keys)
    fires_at_all = any(replays[r][k]["fires"] for r in ("yield", "task") for k in keys)
    verdict = {"twins_identical_on_every_logged_arm": bool(same),
               "at_least_one_policy_acts": bool(fires_at_all),
               "vacuous": bool(same or not fires_at_all)}
    if verbose:
        print("\n" + "-" * 78)
        print("IS THE LIVE CONTRAST NON-VACUOUS?")
        print("-" * 78)
        print(f"  the twins fire at identical cycles on every logged arm : {same}")
        print(f"  at least one fitted policy ever acts                   : {fires_at_all}")
        print(f"  => VACUOUS: {verdict['vacuous']}"
              + ("   (halt and report -- do not launch)" if verdict["vacuous"]
                 else "   (the live contrast is well posed)"))

    # ---- gap sensitivity, recorded as a diagnostic (see the docstring) -------------- #
    gap_sens = {}
    keep = GAP
    try:
        for g in (0, 1, 2):
            globals()["GAP"] = g
            Xg, Yg, Ag, Bg = [], {"yield": [], "task": []}, [], []
            for i, k in enumerate(keys):
                for w in fit_windows(rows[k], recs[k]):
                    Xg.append(w["V"])
                    Yg["yield"].append(w["y_yield"])
                    Yg["task"].append(w["y_task"])
                    Ag.append(i)
                    Bg.append(w["bucket"])
            Xg, Ag, Bg = np.array(Xg), np.array(Ag), np.array(Bg)
            gap_sens[str(g)] = {"n": int(len(Xg))}
            for rw in ("yield", "task"):
                yv = np.array(Yg[rw])
                tc = GAUGES.index("yield" if rw == "yield" else "ledger")
                for b in BUCKETS:
                    m = Bg == b
                    d = fit_direction(Xg[m], yv[m], Ag[m], tc)
                    gap_sens[str(g)][f"{rw}_wc{b}"] = {
                        "a": d["a"], "mse_ratio_vs_thermostat": d["mse_ratio_vs_thermostat"],
                        "r2_vs_zero": d["r2_vs_zero"]}
    finally:
        globals()["GAP"] = keep

    obj = {"fits": fits, "relation": rel, "gate_f1": f1, "verdict": verdict,
           "gap_sensitivity": gap_sens,
           "corpus_windows": per_arm, "n_windows": int(len(X)),
           "n_by_bucket": {b: int((BK == b).sum()) for b in BUCKETS},
           "constants": {"gauges": list(GAUGES), "span": SPAN, "W": W, "burn": BURN,
                         "alpha": ALPHA, "gap": GAP, "horizon": H,
                         "lam_grid": list(LAM_GRID), "era_caps": list(ERA_CAPS),
                         "max_macro_level": MAX_MACRO_LEVEL},
           "floors_used_as_units": FLOORS}
    out_path = out_path or os.path.join(HERE, "fit.json")
    with open(out_path, "w") as fh:
        json.dump(obj, fh, indent=2)
    if verbose:
        print(f"\n[fit] -> {out_path}")
    return obj


if __name__ == "__main__":
    r = main()
    ok = bool(r["gate_f1"]["ALL"]) and not r["verdict"]["vacuous"]
    sys.exit(0 if ok else 1)
