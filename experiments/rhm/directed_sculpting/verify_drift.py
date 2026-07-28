"""Verify the support-fixed rule drift has the properties the directed-sculpting run needs.

D0 back-compatibility  -- uniform mixture weights reproduce the plain generator, and node
                          counts per depth come out at exactly s^l.
D1 support-fixed       -- after heavy drift every sequence is still on-grammar (its true
                          root stays in the root's possible-set), the rule TABLES are
                          bit-identical, and the inverse maps are unchanged. So the DP d*
                          -- a function of the rule support alone -- cannot move. This is
                          the property that makes drift here immune to the catastrophic
                          forgetting that made the wholesale rule-seed swap measure nothing.
D2 exact magnitude     -- the closed-form KL/sequence matches the realised log-likelihood
                          ratio along recorded derivations.
D3 matched levels      -- per-level sigma from `calibrate_sigma` equalises KL/sequence
                          across the level sweep, which a shared sigma does not (a cell at
                          depth l fires s^l times per sequence).
D4 stationary walk     -- the OU walk's weight distribution is stationary, so the
                          shift-generating process does not itself drift.

Run from experiments/:
  modal run rhm/directed_sculpting/verify_drift.py::verify_drift
  python3 -c "from rhm.directed_sculpting.verify_drift import run; run()"   # local, ~1 min
"""

import copy
import json
import os

import modal
import numpy as np

from rhm.rhm_drift import (
    calibrate_sigma, drift_kl, make_drift_state, node_counts, ou_step,
    sample_derivations_weighted, state_weights, trace_loglik, uniform_weights)
from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import possible_sets, sample_derivations
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-verify-drift", image=image)


def run(v=8, s=2, L=5, m=2, rule_seed=0, kappa=0.05, target_kl=0.30, heavy_sigma=0.6,
        n_seq=20000, verbose=True):
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rules_snapshot = copy.deepcopy(rules)
    imaps = build_inverse_maps(rules)
    base = uniform_weights(rules)
    out = {"config": {"v": v, "s": s, "L": L, "m": m, "rule_seed": rule_seed,
                      "kappa": kappa, "target_kl": target_kl,
                      "heavy_sigma": heavy_sigma, "n_seq": n_seq}}

    def say(*a):
        if verbose:
            print(*a)

    # -- D0 ------------------------------------------------------------------
    roots = np.random.default_rng(5).integers(0, v, size=n_seq)
    plain = sample_derivations(rules, roots, s, np.random.default_rng(0))
    weighted = sample_derivations_weighted(rules, roots, s, np.random.default_rng(0), base)
    hp = np.bincount(plain.reshape(-1), minlength=v) / plain.size
    hw = np.bincount(weighted.reshape(-1), minlength=v) / weighted.size
    counts = node_counts(rules, s).sum(1)
    out["d0_backcompat"] = {
        "leaf_marginal_tv_plain_vs_weighted_uniform": float(0.5 * np.abs(hp - hw).sum()),
        "node_counts_per_depth": counts.tolist(),
        "expected_node_counts": [s ** l for l in range(L)],
        "node_counts_exact": bool(np.allclose(counts, [s ** l for l in range(L)])),
    }
    say(f"[D0] leaf-marginal TV(plain, weighted-uniform) = "
        f"{out['d0_backcompat']['leaf_marginal_tv_plain_vs_weighted_uniform']:.5f}")
    say(f"     node counts per depth {counts.round(3).tolist()} "
        f"(exact: {out['d0_backcompat']['node_counts_exact']})")

    # -- D1 ------------------------------------------------------------------
    st = make_drift_state(rules, levels=range(L), seed=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        ou_step(st, kappa, heavy_sigma, rng)
    w = state_weights(rules, st)
    drifted = sample_derivations_weighted(rules, roots, s, np.random.default_rng(8), w)
    ps = possible_sets(rules, drifted, s)[-1][:, 0, :]
    on_grammar = float(ps[np.arange(n_seq), roots].mean())
    out["d1_support_fixed"] = {
        "max_weight_deviation_from_uniform": float(
            np.abs(np.concatenate([x.ravel() for x in w]) - 1.0 / m).max()),
        "on_grammar_rate_after_drift": on_grammar,
        "rule_tables_bit_identical": all(np.array_equal(a, b)
                                         for a, b in zip(rules, rules_snapshot)),
        "inverse_maps_unchanged": all(np.array_equal(a, b) for a, b
                                      in zip(imaps, build_inverse_maps(rules))),
    }
    say(f"[D1] on-grammar rate after heavy drift = {on_grammar:.4f}; "
        f"rule tables identical = {out['d1_support_fixed']['rule_tables_bit_identical']}; "
        f"inverse maps unchanged = {out['d1_support_fixed']['inverse_maps_unchanged']}")

    # -- D2 ------------------------------------------------------------------
    kl_analytic, per_level = drift_kl(rules, s, base, w)
    _lv, trace = sample_derivations_weighted(rules, roots, s, np.random.default_rng(11), w,
                                             return_trace=True)
    empirical = float((trace_loglik(w, trace) - trace_loglik(base, trace)).mean())
    out["d2_exact_magnitude"] = {
        "kl_analytic_nats_per_seq": kl_analytic,
        "kl_empirical_nats_per_seq": empirical,
        "relative_error": abs(empirical - kl_analytic) / kl_analytic,
        "per_level_kl": per_level.tolist(),
        "shared_sigma_kl_spread": float(per_level.max() / per_level.min()),
    }
    say(f"[D2] KL/seq analytic={kl_analytic:.5f} empirical={empirical:.5f} "
        f"(rel.err {out['d2_exact_magnitude']['relative_error']:.2%}); "
        f"per-level spread at shared sigma = "
        f"{out['d2_exact_magnitude']['shared_sigma_kl_spread']:.1f}x")

    # -- D3 ------------------------------------------------------------------
    sigma = calibrate_sigma(rules, s, list(range(L)), kappa, target_kl, seed=0)
    realised = {}
    for ell in range(L):
        per_seed = []
        for sd in (11, 22, 33):
            stx = make_drift_state(rules, [ell], seed=0)
            rx = np.random.default_rng(sd)
            k = []
            for t in range(3000):
                ou_step(stx, kappa, sigma[ell], rx)
                if t >= 500:
                    k.append(drift_kl(rules, s, base, state_weights(rules, stx))[0])
            per_seed.append(float(np.mean(k)))
        realised[ell] = per_seed
    within = [max(realised[l][i] for l in range(L)) / min(realised[l][i] for l in range(L))
              for i in range(3)]
    out["d3_matched_levels"] = {
        "sigma_per_level": {str(k): float(x) for k, x in sigma.items()},
        "realised_kl_per_level_per_seed": {str(k): x for k, x in realised.items()},
        "within_realisation_level_spread": within,
        "sigma_root_over_surface": float(sigma[0] / sigma[L - 1]),
    }
    say(f"[D3] sigma per level " +
        " ".join(f"L{l}={sigma[l]:.4f}" for l in range(L)))
    say(f"     within-realisation cross-level KL spread = "
        f"{['%.2fx' % x for x in within]} (shared sigma: "
        f"{out['d2_exact_magnitude']['shared_sigma_kl_spread']:.1f}x)")

    # -- D4 ------------------------------------------------------------------
    stz = make_drift_state(rules, [L - 1], seed=0)
    rz = np.random.default_rng(4)
    tv = []
    for _ in range(2000):
        ou_step(stz, kappa, heavy_sigma, rz)
        ww = state_weights(rules, stz)[L - 1]
        tv.append(float(0.5 * np.abs(ww - 1.0 / m).sum(-1).mean()))
    windows = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
    out["d4_stationary"] = {
        "tv_from_uniform_by_window": [
            {"from": a, "to": b, "mean": float(np.mean(tv[a:b])),
             "sd": float(np.std(tv[a:b]))} for a, b in windows]}
    say("[D4] TV-from-uniform by window: " + "  ".join(
        f"{a}-{b}: {np.mean(tv[a:b]):.4f}" for a, b in windows))

    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=8192)
def verify_drift(v: int = 8, s: int = 2, depth: int = 5, m: int = 2, kappa: float = 0.05,
                 target_kl: float = 0.30, tag: str = "v1"):
    res = run(v=v, s=s, L=depth, m=m, kappa=kappa, target_kl=target_kl)
    out_dir = f"{DATA_DIR}/directed_sculpting/drift_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json")
    return res
