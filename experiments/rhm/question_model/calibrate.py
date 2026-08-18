"""Offline calibration: set every drift knob by a measurement, in the currency it is read in.

`setlist/FILES.md`'s lesson, transposed: sigma and kappa are DIFFERENT KNOBS and the objective is
not monotone, so sweep, do not solve. What changed is which currency the knob is calibrated in.
`setlist` read nats of demand over a mined vocabulary; here the readout is the *decomposition
itself*, so the knobs are set against:

  demand    E[Dm] nats per token (window filter = the model's conditioning set, and history
            filter = an unbounded-memory learner), and its per-component split;
  structure E[S_D] per level against the DRIFT-OFF value -- the admissibility gate. `setlist`'s
            sigma = 2.5 was inadmissible because the arms stopped separating; the analogue here
            is that a loud demand CONCENTRATES the mixture, which makes the corpus easier and
            shrinks the structure channel the experiment is supposed to decompose against. News
            too loud is weather.
  aleatoric E[H_irr] -- the third channel, which a concentrated mixture also eats into (knowing
            theta makes the synonym choice partly predictable). Worth watching for its own sake:
            it is the quantity Half 2's loss-side tap would target.

Run locally (numpy only, a few minutes):
    cd experiments && python3 -m rhm.question_model.calibrate
"""

import json
import time

import numpy as np

from rhm.question_model import demand_state as DS
from rhm.question_model import oracle_q as OQ
from rhm.question_model import qfilter as QF


def state_weight_arrays(spec, dirs, L):
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
    rp = np.stack([W[g][0] for g in range(G)])
    return W, rw, rp


def cell(rules, spec, n=256, seed=3, do_structure=True, world="drift", chunk=64):
    """One calibration cell. Returns a flat dict of measured quantities."""
    L = len(rules)
    _v, _m, s = rules[0].shape
    dirs = DS.make_directions(rules, spec)
    W, rw_s, rp_s = state_weight_arrays(spec, dirs, L)
    idx, pj = DS.state_grid(spec)
    C = idx.shape[1]
    names = [nm for nm, _ in spec["components"]]

    seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, n, seed, world=world)
    ll = QF.prefix_loglik(rules, seqs, rw_s, rp_s, chunk=chunk)

    dc = QF.demand_channel(ll, pj, state_idx=idx, n_comp=C, K=spec["K"])
    out = {"sigma_s": spec["sigma_s"], "kappa": spec["kappa"], "K": spec["K"],
           "n_states": int(idx.shape[0]), "world": world, "n": n,
           "H_theta_nats": DS.theta_entropy(spec),
           "H_theta_start": float(dc["H_theta"][:, 0].mean()),
           "H_theta_end": float(dc["H_theta"][:, -1].mean()),
           "Dm_window": float(dc["Dm"].mean()),
           "surp_marg": float(dc["surprisal"].mean())}
    for c, nm in enumerate(names):
        out[f"Dm_window_{nm}"] = float(dc["Dm_comp"][c].mean())
    if tidx.min() >= 0:
        out["surp_theta"] = float(-ll[np.arange(n), tidx, :][:, 1:].mean())
        out["demand_identity_err"] = abs(out["surp_marg"] - out["surp_theta"]
                                         - out["Dm_window"])
    # history filter: blocks are in trajectory order, prior carried through the OU kernel
    if idx.shape[0] > 1:
        P = DS.joint_transition(spec)
        hp = QF.history_priors(ll, pj, P)
        dh = QF.demand_channel(ll, hp, state_idx=idx, n_comp=C, K=spec["K"])
        half = n // 2                     # drop the burn-in half
        out["Dm_history"] = float(dh["Dm"][half:].mean())
        out["H_theta_history_start"] = float(dh["H_theta"][half:, 0].mean())
        for c, nm in enumerate(names):
            out[f"Dm_history_{nm}"] = float(dh["Dm_comp"][c][half:].mean())
    else:
        out["Dm_history"] = 0.0
        out["H_theta_history_start"] = 0.0

    if do_structure:
        rw = [np.stack([W[i][1][d] for i in np.maximum(tidx, 0)]) for d in range(L)]
        rp = np.stack([W[i][0] for i in np.maximum(tidx, 0)])
        if tidx.min() < 0:                # incoherent world: the marginal mixture
            mrp, mrw = DS.marginal_weights(spec, dirs)
            rw = [np.repeat(mrw[d][None], n, 0) for d in range(L)]
            rp = np.repeat(mrp[None], n, 0)
        sc = OQ.structure_channel(rules, seqs, lf, rw, rp, chunk=chunk, verbose=False)
        out["H_tot_theta"] = float(sc["H_tot"].mean())
        for D in sc["Ds"]:
            out[f"S_D{D}"] = float(sc["S_joint"][D].mean())
            out[f"Hirr_D{D}"] = float(sc["H_irr"][D].mean())
        out["three_way"] = QF.three_way_check(
            dc["surprisal"], dc["Dm"], sc["S_joint"], sc["H_irr"], sc["Ds"], L)["per_D"]
    return out


def sweep(v=16, s=2, L=6, m=4, rule_seed=0, n=256, seed=3, out_path=None):
    from rhm.rhm_data import generate_rules_distinct
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    cells = []
    plan = ([("sigma", DS.make_spec(K=7, sigma_s=0.0, kappa=0.15, L=L))]
            + [("sigma", DS.make_spec(K=7, sigma_s=sg, kappa=0.15, L=L))
               for sg in (0.4, 0.7, 1.0, 1.4, 2.0)]
            + [("K", DS.make_spec(K=k, sigma_s=1.0, kappa=0.15, L=L)) for k in (5, 9)]
            + [("kappa", DS.make_spec(K=7, sigma_s=1.0, kappa=kp, L=L))
               for kp in (0.05, 0.40, 1.00)])
    for axis, spec in plan:
        t0 = time.time()
        r = cell(rules, spec, n=n, seed=seed)
        r["axis"] = axis
        r["secs"] = time.time() - t0
        cells.append(r)
        print(f"[{axis}] sigma_s={r['sigma_s']:.2f} kappa={r['kappa']:.2f} K={r['K']} "
              f"G={r['n_states']:4d} | Dm_win {r['Dm_window']:.4f} "
              f"(root {r.get('Dm_window_root', 0):.4f} hi {r.get('Dm_window_hi', 0):.4f} "
              f"lo {r.get('Dm_window_lo', 0):.4f}) Dm_hist {r['Dm_history']:.4f} | "
              f"surp_marg {r['surp_marg']:.4f} | S_d1 {r.get('S_D5', 0):.4f} "
              f"S_d3 {r.get('S_D3', 0):.4f} S_d6 {r.get('S_D0', 0):.4f} | "
              f"Hirr_d1 {r.get('Hirr_D5', 0):.4f} | {r['secs']:.0f}s", flush=True)
    # admissibility: structure channel relative to drift-off
    base = cells[0]
    print("\nadmissibility (structure channel relative to drift-off):")
    for r in cells:
        rel = {f"d{L - D}": r[f"S_D{D}"] / max(base[f"S_D{D}"], 1e-12) for D in range(L)}
        print(f"  sigma_s={r['sigma_s']:.2f} K={r['K']} kappa={r['kappa']:.2f}: "
              + "  ".join(f"{k} {x:.3f}" for k, x in rel.items())
              + f"   surp {r['surp_marg'] / base['surp_marg']:.3f}")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(cells, f, indent=2)
    return cells


if __name__ == "__main__":
    sweep(out_path="/tmp/qm_calibration.json")
