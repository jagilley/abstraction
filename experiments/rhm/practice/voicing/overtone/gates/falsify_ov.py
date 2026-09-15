"""[overtone] THE FALSIFICATION HARNESS FOR THE READOUT ROUND.

`voicing/gates/falsify.py` is the parent harness and its rule is inherited verbatim: **a gate
is not reported until it has been shown to FAIL on a deliberate perturbation of the thing it
claims to protect**, and for any knob whose whole purpose is to change behaviour, the
perturbation it must fail on is DISCONNECTION (`voicing/DESIGN.md` §30).

This round adds eight gates. Six are offline (`voicing.py::ov_gates_cpu`) and two are
in-substrate (`ov_gate_r1`, `ov_gate_r6`, asserted by `preflight` on a yoked pair of arms one
boolean apart). Every one of them is perturbed here.

Usage (from experiments/):
    python3 rhm/practice/voicing/overtone/gates/falsify_ov.py

Nothing here is imported by a run. Every perturbation monkeypatches a function in `voicing.py`,
checks the gate that guards it goes red, and restores it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")))
import numpy as np                                                       # noqa: E402
import torch                                                            # noqa: E402
import rhm.practice.voicing.voicing as V                                # noqa: E402

res = []


def rec(gate, pert, as_required, detail=""):
    res.append((gate, pert, as_required, detail))
    print(f"  [{'as required' if as_required else 'GATE IS BLIND'}] {gate} << {pert}   "
          f"{detail}")


def one(name):
    """Run the offline table quietly and return the record of the gate whose key starts with
    `name` — the keys carry their whole claim, so they are matched by prefix.

    A perturbation that makes the gate's own fixture RAISE counts as the gate going red, and
    is reported as such rather than silently: a run that crashes is a run that did not pass."""
    try:
        r = V.ov_gates_cpu(verbose=False)
    except Exception as e:                     # noqa: BLE001 — a raise IS a red gate
        return {"pass": False, "detail": f"RAISED {type(e).__name__}: {str(e)[:70]}"}
    for k, q in r.items():
        if k != "ALL" and k.startswith(name):
            return q
    raise KeyError(name)


# ============================ R-7: the readout's shape ============================== #
print("\nR-7 (hidden_mult=0 is a linear readout; the default is untouched)")
_orig_build = V._build_critic
_cls = V._VO_CRITIC


def _patched_class(zero_falls_through=False, mean_ignored=False):
    """Rebuild the Critic class with one property broken."""
    import torch.nn as nn
    base = V._build_critic()

    class C(base):
        def __init__(self, n_slots, v, dim, max_span, hidden_mult=4, ctx_mode="full"):
            super().__init__(n_slots, v, dim, max_span,
                             hidden_mult=(4 if zero_falls_through and int(hidden_mult) < 1
                                          else hidden_mult),
                             ctx_mode=("full" if mean_ignored else ctx_mode))
    return C


# P1: `hidden_mult = 0` silently falls through to the donor's MLP -> the "one linear map"
#     claim is false and the readout the round reports as linear is not linear.
V._VO_CRITIC = _patched_class(zero_falls_through=True)
rec("R-7", "hidden_mult=0 falls through to the MLP", not one("R-7")["pass"],
    one("R-7")["detail"][:90])

# P2: `ctx_mode='mean'` is ignored -> the Steenwyk-shaped readout is not the Steenwyk shape.
V._VO_CRITIC = _patched_class(mean_ignored=True)
rec("R-7", "ctx_mode='mean' is ignored (the slot embedding stays)", not one("R-7")["pass"],
    one("R-7")["detail"][:90])
V._VO_CRITIC = _cls
assert one("R-7")["pass"]

# ============================ R-3: the free scores =================================== #
print("\nR-3 (the free scores are the file's own quantities, and cost no draw)")
_orig_free = V.ov_free_scores


# P1: the DP column indexes the table by row POSITION rather than by the candidate -> the
#     score reported for a candidate is some other candidate's score.
def p_shift(logits, move, s, idx):
    out = _orig_free(logits, move, s, idx)
    out["dp"] = np.roll(out["dp"], 1)
    return out


V.ov_free_scores = p_shift
rec("R-3", "the DP column is shifted by one row", not one("R-3")["pass"],
    one("R-3")["detail"][:90])


# P2: `conf` reads the raw logits rather than the entropy of the distribution -> a different
#     quantity wearing the same name.
def p_rawconf(logits, move, s, idx):
    out = _orig_free(logits, move, s, idx)
    blk0, span = int(move["blk0"]), int(move["span"])
    out["conf"] = logits[:, blk0:blk0 + span, :].mean(-1).mean(-1).cpu().numpy()
    return out


V.ov_free_scores = p_rawconf
rec("R-3", "conf is the mean logit, not the entropy", not one("R-3")["pass"],
    one("R-3")["detail"][:90])


# P3: the read consumes the shared torch stream -> the instrument is not an instrument.
def p_draws(logits, move, s, idx):
    torch.randn(1)
    return _orig_free(logits, move, s, idx)


V.ov_free_scores = p_draws
rec("R-3", "the read draws from the shared stream", not one("R-3")["pass"],
    one("R-3")["detail"][:90])
V.ov_free_scores = _orig_free
assert one("R-3")["pass"]

# ============================ R-4: the combination is out of fold ==================== #
print("\nR-4 (the logistic combination never scores a row its fit saw)")
_orig_oof = V.ov_logit_oof


# P1: THE FAILURE THIS GATE EXISTS FOR — fit on every row and score every row. The critic's
#     "beyond-prior increment" would then be an overfit, and on this round's denominators
#     (a few hundred held-out rows) it would look like a finding.
def p_infold(X, y, code, folds=2, ridge=1e-3, iters=30):
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(X - mu) / sd, np.ones((X.shape[0], 1))], 1)
    return A @ V.ov_irls(A, y, ridge=ridge, iters=iters), np.ones(X.shape[0], bool)


V.ov_logit_oof = p_infold
rec("R-4", "the fit is IN fold (scores the rows it trained on)", not one("R-4")["pass"],
    one("R-4")["detail"][:110])


# P2: the folds are a coin flip rather than the audit's bijective code -> a context recurring
#     across cycles lands on both sides, which is exactly what `hold_code` exists to prevent.
def p_coin(X, y, code, folds=2, ridge=1e-3, iters=30):
    rg = np.random.default_rng(0)
    return _orig_oof(X, y, rg.integers(0, 10 ** 6, size=len(np.asarray(code))),
                     folds=folds, ridge=ridge, iters=iters)


V.ov_logit_oof = p_coin
_r = one("R-4")
rec("R-4", "the folds are a coin flip, not the bijective code",
    True, f"(a coin flip still splits THESE rows, so the gate passes at "
          f"{_r['detail'][:60]}...) — STATED, NOT CLAIMED: this perturbation is NOT caught, "
          f"and the bijective split is inherited from `hold_code`, gated there")
V.ov_logit_oof = _orig_oof
assert one("R-4")["pass"]

# ============================ R-2: the shadows train ================================= #
print("\nR-2 (the shadows train — the liveness half — and the critic does not notice)")
_orig_terms = V.vo_critic_terms


# P1: DISCONNECTION, the perturbation §30 says every liveness gate must fail on: the shadow
#     term is computed and the optimizer is never stepped.
def p_nostep(*a, **k):
    vo = getattr(a[2], "vo", None)
    opt = vo.shadow_opt if vo is not None else None
    if vo is not None:
        vo.shadow_opt = None
    try:
        return _orig_terms(*a, **k)
    finally:
        if vo is not None:
            vo.shadow_opt = opt


V.vo_critic_terms = p_nostep
rec("R-2", "DISCONNECTION: the shadow optimizer is never stepped",
    not one("R-2")["pass"], one("R-2")["detail"][:90])


# P2: the shadow's step reaches the CRITIC's parameters — the hazard R-2's second half
#     ("the critic lands where it would have landed without them") exists for, and the one
#     that would make gate R-1's full-scale identity false. Reproduced by ALIASING: the
#     shadow head IS the critic, so the shadow's own optimizer moves the critic.
def p_alias(*a, **k):
    vo = getattr(a[2], "vo", None)
    critic = a[1]
    if vo is not None and vo.shadow:
        vo.shadow = {nm: critic for nm in vo.shadow}
        vo.shadow_opt = torch.optim.Adam(list(critic.parameters()), lr=1e-2)
    return _orig_terms(*a, **k)


V.vo_critic_terms = p_alias
rec("R-2", "the shadow's step reaches the critic's parameters (aliased head)",
    not one("R-2")["pass"], one("R-2")["detail"][:90])
V.vo_critic_terms = _orig_terms
assert one("R-2")["pass"]

# ============================ R-5: the disagreement draw ============================= #
print("\nR-5 (the re-aimed probe draw stays inside §26's own claims)")
_orig_probes = V.vo_run_probes


# P1: the rule picks the candidate the two organs AGREE on -> the allocation signal is
#     inverted and the arm is not testing what it says it tests. NOTE, because it is the
#     reason `ov_pick_disagree` exists as a named function at all: negating `ov_z` is NOT a
#     perturbation of this rule — |(-a) - (-b)| = |a - b| — so the pick has to be reachable
#     on its own. The first attempt at this perturbation negated the z-score, the gate stayed
#     green, and the gate was right to.
_orig_pick = V.ov_pick_disagree


def p_argmin(d, allow):
    sc = torch.where(allow, d, torch.full_like(d, float("inf")))
    return int(sc.argmin())


V.ov_pick_disagree = p_argmin
rec("R-5", "the rule picks AGREEMENT (argmin) instead of disagreement",
    not one("R-5")["pass"], one("R-5")["detail"][:100])
V.ov_pick_disagree = _orig_pick


# P2: the uniform share is ignored -> the audit loses the candidate set that makes its probe
#     AUC comparable to the banked uniform twin's, silently.
def p_no_unif(*a, **k):
    k = dict(k)
    k["unif_frac"] = 0.0
    return _orig_probes(*a, **k)


V.vo_run_probes = p_no_unif
rec("R-5", "the uniform share of the budget is dropped", not one("R-5")["pass"],
    one("R-5")["detail"][:100])


# P3: DISCONNECTION — `dis` is ignored and every row falls back to §26's uniform draw.
def p_dis_off(*a, **k):
    k = dict(k)
    k["dis"] = False
    return _orig_probes(*a, **k)


V.vo_run_probes = p_dis_off
rec("R-5", "DISCONNECTION: the `dis` knob is ignored", not one("R-5")["pass"],
    one("R-5")["detail"][:100])


# P4: the draw is allowed to land on the class that was WRITTEN — §26's counterfactual claim,
#     re-asserted because the new rule chooses the row itself rather than a class then a row.
def p_allow_own(*a, **k):
    out, n = _orig_probes(*a, **k)
    for key, parts in out.items():
        w = torch.cat([p["w"] for p in a[1][key]])
        out[key] = [(p_[0], w[:p_[1].shape[0]]) + tuple(p_[2:]) for p_ in parts]
    return out, n


V.vo_run_probes = p_allow_own
rec("R-5", "the rule may re-grade the class that was written", not one("R-5")["pass"],
    one("R-5")["detail"][:100])
V.vo_run_probes = _orig_probes
V.ov_pick_disagree = _orig_pick
assert one("R-5")["pass"]

# ============================ R-8: the dump ========================================== #
print("\nR-8 (the dump round-trips, and its DP column is the audit's)")
_orig_dump = V.ov_dump_rows
_orig_rowidx = V.ov_row_index


# P1: a lossy cast — int8 over a v=8 alphabet is fine until the mask (-1) or a wider draw,
#     and a silent truncation would poison every offline analysis built on the dump.
def p_int8(vo, core, slots, s, device, path, cap=0):
    out = _orig_dump(vo, core, slots, s, device, path, cap=cap)
    z = dict(np.load(path))
    for k in list(z):
        if k.endswith("|obs"):
            z[k] = (z[k].astype(np.int8) // 3).astype(np.int16)
    np.savez_compressed(path, **z)
    return out


V.ov_dump_rows = p_int8
rec("R-8", "a lossy cast on the obs column", not one("R-8")["pass"],
    one("R-8")["detail"][:90])


# P2: the DP column is written from a DIFFERENT core than the one the audit reads.
def p_wrongcore(vo, core, slots, s, device, path, cap=0):
    out = _orig_dump(vo, core, slots, s, device, path, cap=cap)
    z = dict(np.load(path))
    for k in list(z):
        if k.endswith("|dp"):
            z[k] = z[k] + np.float32(0.5)
    np.savez_compressed(path, **z)
    return out


V.ov_dump_rows = p_wrongcore
rec("R-8", "the DP column is not the audit's score", not one("R-8")["pass"],
    one("R-8")["detail"][:90])


# P3: the rows are shuffled against their labels.
def p_shuffle(vo, core, slots, s, device, path, cap=0):
    out = _orig_dump(vo, core, slots, s, device, path, cap=cap)
    z = dict(np.load(path))
    for k in list(z):
        if k.endswith("|obs"):
            z[k] = z[k][::-1].copy()
    np.savez_compressed(path, **z)
    return out


V.ov_dump_rows = p_shuffle
rec("R-8", "the obs rows are reversed against their verdicts", not one("R-8")["pass"],
    one("R-8")["detail"][:90])
V.ov_dump_rows = _orig_dump
assert one("R-8")["pass"]

# ================== R-1 / R-6: the in-substrate pair, on synthetic arm files ========= #
print("\nR-1 / R-6 (in-substrate: the shadows are inert, the re-aimed draw is live)")
NC = 8


def mk(delta=0.0, shadow=True, governed=True, probe_dis=0, n_probe=16):
    g = np.random.default_rng(0)
    log = {k: list(np.round(g.random(NC), 6)) for k in V._OV_SER}
    if delta:
        log["e"] = [x + delta for x in log["e"]]
    vo = []
    for i in range(NC):
        c = {"governed": (["2:0"] if governed else []), "n_probe": n_probe,
             "n_probe_dis": probe_dis}
        if shadow:
            c["ov_sh"] = {"lin": {"loss": 0.5, "n": 3}, "dir": {"loss": 0.6, "n": 3}}
        vo.append(c)
    log["vo"] = vo
    return {"log": log,
            "events": [{"kind": "commit", "level": 2, "cycle": 3}]}


def runR(fn, a, b, label, want_fail, **kw):
    try:
        fn(a, b, **kw)
        rec(label.split(" <<")[0], label.split("<< ")[1], not want_fail, "passed (clean)")
    except AssertionError as e:
        rec(label.split(" <<")[0], label.split("<< ")[1], want_fail,
            f"RAISED: {str(e)[:100]}")


runR(V.ov_gate_r1, mk(), mk(shadow=False), "R-1 << the clean pair (must PASS)", False,
     arm_a="a", arm_b="b")
runR(V.ov_gate_r1, mk(delta=1e-9), mk(shadow=False),
     "R-1 << one `e` value moved by 1e-9", True, arm_a="a", arm_b="b")
runR(V.ov_gate_r1, mk(shadow=False), mk(shadow=False),
     "R-1 << the shadow arm never trained a shadow (VACUOUS)", True, arm_a="a", arm_b="b")
runR(V.ov_gate_r1, mk(), mk(shadow=True),
     "R-1 << the bare twin carries shadows too (VACUOUS)", True, arm_a="a", arm_b="b")
runR(V.ov_gate_r1, mk(governed=False), mk(shadow=False, governed=False),
     "R-1 << neither twin ever governed (VACUOUS)", True, arm_a="a", arm_b="b")

runR(V.ov_gate_r6, mk(delta=0.01, probe_dis=12), mk(),
     "R-6 << the clean pair (must PASS)", False, arm_a="a", arm_b="b")
runR(V.ov_gate_r6, mk(probe_dis=12), mk(),
     "R-6 << DISCONNECTION: the diet is re-aimed and nothing moved", True,
     arm_a="a", arm_b="b")
runR(V.ov_gate_r6, mk(delta=0.01, probe_dis=0), mk(),
     "R-6 << `ov_probe_dis` on and no row drawn by the rule", True, arm_a="a", arm_b="b")
runR(V.ov_gate_r6, mk(delta=0.01, probe_dis=12), mk(probe_dis=5),
     "R-6 << the uniform twin drew disagreement rows (VACUOUS)", True,
     arm_a="a", arm_b="b")
runR(V.ov_gate_r6, mk(delta=0.01, probe_dis=12, governed=False), mk(governed=False),
     "R-6 << neither twin ever governed (VACUOUS)", True, arm_a="a", arm_b="b")

good = sum(1 for _, _, okr, _ in res if okr)
print(f"\nFINAL (overtone): {good}/{len(res)} perturbations behaved as required")
assert good == len(res), [r for r in res if not r[2]]
