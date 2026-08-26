"""PHASE A -- the gates that have to pass (or measurably fail) before any main launch.

Each one is a precondition the design rests on, and each is cheap. The rule this node inherits from
legato F2: a shared knob is chosen by what keeps the MEASUREMENT AXIS alive, not by what makes an arm
look good -- and where the design meets reality and does not fit, that is a finding to bring back,
not something to sand down.

  G-F  FORK FIDELITY. offbook's `World` reproduces legato's `World` bit-for-bit on the donor code
       path (the same traversal, same geometries, same seed). If this fails the fork has drifted and
       nothing downstream is comparable to the parent node.

  G-R  RNG DISCIPLINE. Minting either head leaves the shared torch stream exactly where it found it,
       and the exploration draw has its own generator. This is what licenses the `fid` twin.

  G-P  pi FIDELITY. `audit_prop_k` at k = every legal slot returns the SAME candidate set, the same
       scores and the same launched commands as `audit_all` -- bit-for-bit, whatever pi's weights are
       (`native/prop`'s `prop_kN` idiom).

  G-K  THE RENT GATE -- the one that can legitimately fail. At the chosen library size K and
       per-decision deliberation budget D, does full seam audition MEASURABLY bind? Three curves:
       (a) the planner ladder -- realised piece error of a live plan at each CEM rung, from realised
       seam states; (b) the library curve -- realised error of the best-of-K seam-time audition
       against K; (c) the rent table -- which rung `audit_all` can afford against which rung
       top-k routing can afford, and the error between them. If the rent does not bind at any
       affordable K, that is a fact about this substrate (the arm's planner is cheap relative to its
       library in a way RHM's beam was not) and the node runs with the budget conversion OFF, the
       rent reported as a cost rather than converted into an outcome.

  G-S  POOL SPREAD (fingering G1, re-checked on the GROWN pools). Seam-state spread against the
       metering noise floor, per seam; the per-state oracle gain over the grown library; and whether
       the content slots are distinct or a `fixed` unit wearing a library's name.

  G-C  AUDITION CALIBRATION, per level. Seam-time audition scores with the MODEL, and `span/` F2 is
       a standing fact about this plant: the model's promised far-seam arrival stays ~0.07 m at every
       cycle while the truth walks to ~1 m. So a chain scored on the model's own 60-step imagination
       may be systematically over-rated relative to a segment. Measured as rank correlation and
       ARGMIN REGRET (how much worse the model's pick is than the plant's) at each level, under the
       full-span and horizon-truncated conventions. This is the knob `aud_horizon` is read off.

  G-D  CYCLE COST. Measured wall-clock and priced time per cycle per mode, which is what Phase B's
       cycle budget is sized from -- not a projection. Legato's `never` was killed by the 8 h Modal
       wall clock at c75 of 90; this node runs with `timeout=43200` and a separate reactive cycle
       cap, and G-D is what sets it.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/offbook/gates.py::offbook_gates --quick
    python3 mjc/practice/offbook/launch_detached.py --fn gates --tag g0 --seed 0
    python3 mjc/practice/offbook/analyze_gates.py --tag g0 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.offbook.piece import DEF_WPS, DEF_PATCH_SEG


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_offbook_gates(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler
    from mjc.practice.offbook.world import (World, Ledger, Library, RoutePolicy,
                                            start_postures, elite_for)
    from mjc.practice.offbook import nets as N

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    NS = W.n_seg
    led = Ledger()
    out = {"config": cfg, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_offbook", cfg["tag"], "gates")
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[gates]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}

    # ================================================================= G-F: fork fidelity
    # The donor path, run in both worlds with the same seed. Nothing about the offbook additions can
    # touch it -- but "cannot" is what a gate is for.
    t0 = time.time()
    from mjc.practice.legato.world import World as LegatoWorld
    WL = LegatoWorld(cfg, device)
    ledL = Ledger()
    q_f = start_postures(W.qc, W.Ls, 8, np.random.default_rng(cfg["seed"] + 31),
                         cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])
    fm_f = W.mlp(cfg["seed"] + 40)
    fmL = copy.deepcopy(fm_f)
    S1, U1, S21, _ = W.collect_ou(1200, np.random.default_rng(cfg["seed"] + 11))
    W.set_norm(S1, U1, S21); WL.set_norm(S1, U1, S21)
    oA = W.traverse(fm_f, W.routing("reactive"), q_f, np.random.default_rng(cfg["seed"] + 32),
                    0.06, led, who="instrument", kind="gF")
    oB = WL.traverse(fmL, WL.routing("reactive"), q_f, np.random.default_rng(cfg["seed"] + 32),
                     0.06, ledL, who="instrument", kind="gF")
    gF = {k: float(np.nanmax(np.abs(np.asarray(oA[k], np.float64)
                                    - np.asarray(oB[k], np.float64))))
          for k in ("e_piece", "e_seg", "acts", "acts_raw", "tips", "launch", "final")}
    gF["fb_equal"] = bool(oA["n_fb"] == oB["n_fb"])
    gF["t_equal"] = bool(abs(oA["t_total"] - oB["t_total"]) < 1e-12)
    gF["pass"] = bool(max(v for k, v in gF.items() if isinstance(v, float)) == 0.0
                      and gF["fb_equal"] and gF["t_equal"])
    out["gF"] = gF
    P(f"[G-F] fork fidelity: max|delta| over 7 arrays = "
      f"{max(v for k, v in gF.items() if isinstance(v, float)):.3e}  pass={gF['pass']}")
    save()

    # ================================================================= setup: the real diet
    boot = W.mlp(cfg["seed"] + 40)
    W.train_steps(boot, torch.optim.Adam(boot.parameters(), lr=cfg["fm_lr"]),
                  S1, U1, S21, cfg["fm_steps_boot"], np.random.default_rng(cfg["seed"] + 300))
    gs = make_arm_goal_sampler(W.Ls, cfg["reach_amp"], cfg["reach_lo"], cfg["reach_hi"])
    S1, U1, S21, _ = W.collect_ou(cfg["pool_ou"], np.random.default_rng(cfg["seed"] + 11))
    S2_, U2_, S22, _ = W.collect_reach(cfg["pool_reach"], np.random.default_rng(cfg["seed"] + 12),
                                       boot, gs)
    Sa = np.concatenate([S1, S2_]); Ua = np.concatenate([U1, U2_]); S2a = np.concatenate([S21, S22])
    Se, Ue, S2e, keep = W.exclude_region(Sa, Ua, S2a, w=cfg["exclude_w"])
    W.set_norm(Se, Ue, S2e)
    fm0 = W.mlp(cfg["seed"] + 41)
    W.train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                  Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    fm_app = copy.deepcopy(fm0)
    for p in fm_app.parameters():
        p.requires_grad_(False)

    def geom(m, seed):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    R_REACT = W.routing("reactive")
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    brng = np.random.default_rng(cfg["seed"] + 800)
    buf, traces = [], []
    P(f"[warm] {cfg['n_warm']} reactive practice cycles (batch {cfg['batch']}) ...")
    t_warm = time.time()
    for c in range(1, int(cfg["n_warm"]) + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c)
        pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + c),
                        cfg["sigma_practice"], led, who="agent", kind="practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c), collect=True)
        buf.append(pr["trans"])
        traces.append(dict(acts=pr["acts"].copy(), e_seg=pr["e_seg"].copy(),
                           launches={k: v.copy() for k, v in pr["launches"].items()}))
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])
    warm_s = time.time() - t_warm
    P(f"[warm] done in {warm_s:.0f}s ({warm_s / max(cfg['n_warm'], 1):.1f}s/reactive cycle)")

    # ================================================================= the grown library
    layout = N.SlotLayout(NS, cfg["n_slot"], poison=False)
    library = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
    lrng = np.random.default_rng(cfg["seed"] + 820)
    for k in range(NS):
        for ns in layout.levels(k):
            rows = []
            for h in traces:
                lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k + ns - 1])
                tp = h["acts"][:, lo:hi, :]
                er = np.nanmean(h["e_seg"][:, k:k + ns], 1)
                for b in range(len(tp)):
                    if np.isfinite(tp[b]).all():
                        rows.append((tp[b], h["launches"][k][b], er[b]))
            pick = lrng.permutation(len(rows))[: int(cfg["lib_k"])]
            library.cell(ns, k).add(np.stack([rows[i][0] for i in pick]),
                                    np.stack([rows[i][1] for i in pick]),
                                    np.array([rows[i][2] for i in pick], np.float32),
                                    0, -np.ones((len(pick), ns), int), library.rng)
    out["library"] = dict(sizes=library.sizes(),
                          slot_counts={f"{ns}:{k}": [int(x) for x in c.key_n]
                                       for (ns, k), c in library.cells.items()})
    P(f"[lib] {library.sizes()}")

    # ---- held-out realised seam states, from a fresh reactive traversal at performance tempo
    q_s = geom(cfg["n_score"], cfg["seed"] + 5100)
    os_ = W.traverse(fm, R_REACT, q_s, np.random.default_rng(cfg["seed"] + 4400),
                     cfg["sigma_perf"], led, who="instrument", kind="score_set",
                     approach_plan=app_plan(q_s, cfg["seed"] + 5150))
    seam_S = {k: os_["launches"][k] for k in range(NS)}
    # the metering noise floor: two independent traversals from the same geometries
    os2 = W.traverse(fm, R_REACT, q_s, np.random.default_rng(cfg["seed"] + 4401),
                     cfg["sigma_perf"], led, who="instrument", kind="score_set",
                     approach_plan=app_plan(q_s, cfg["seed"] + 5150))

    # ================================================================= G-S: pool spread
    # The spread that matters here is POSTURAL. `fingering/` G1: the hand arrives within ~3 cm of
    # the waypoint (tip sd 0.031) while the arrival POSTURE spreads 0.29 rad along the self-motion
    # manifold -- boundary information invisible in tip space, which is the whole reason a keyed or
    # routed library has anything to condition on. Both are reported against a MATCHED repeat-noise
    # floor: the same geometries traversed twice with a different motor-noise draw, which is the
    # paired difference's sd (~sqrt(2)x the per-traversal noise, and stated as such).
    gS = {}
    for k in range(NS):
        A, Bs = seam_S[k], os2["launches"][k]
        tip_f = float(np.linalg.norm((W.tip(A) - W.tip(Bs)).std(0)) / np.sqrt(2.0))
        pos_f = float(np.linalg.norm((A[:, :W.n] - Bs[:, :W.n]).std(0)) / np.sqrt(2.0))
        gS[f"seam{k}"] = dict(
            tip_spread=float(np.linalg.norm(W.tip(A).std(0))), tip_noise=tip_f,
            tip_over_noise=float(np.linalg.norm(W.tip(A).std(0)) / max(tip_f, 1e-9)),
            posture_spread=float(np.linalg.norm(A[:, :W.n].std(0))), posture_noise=pos_f,
            posture_over_noise=float(np.linalg.norm(A[:, :W.n].std(0)) / max(pos_f, 1e-9)),
            speed=float(np.linalg.norm(A[:, W.n:], axis=1).mean()))
    out["gS"] = gS
    P("[G-S] posture spread/noise by seam: "
      + "  ".join(f"s{k}={gS[f'seam{k}']['posture_over_noise']:.1f}x" for k in range(NS))
      + "   tip: " + "  ".join(f"s{k}={gS[f'seam{k}']['tip_over_noise']:.1f}x" for k in range(NS)))
    save()

    # ================================================================= realised-error instruments
    def realise(k, ns, S, cmds):
        """Ground truth: replay `cmds[i]` in the PLANT from `S[i]` and score its waypoints. The
        oracle instrument -- free and labelled as such, never available to an agent at a seam."""
        cps = W.checkpoints(k, ns)
        e = np.empty(len(S), np.float32)
        for i in range(len(S)):
            e[i] = W.replay(S[i:i + 1], np.asarray(cmds[i], np.float32), cps).mean()
        return e

    ladder = [int(x) for x in cfg["cem_ladder"]]
    gK = {"ladder": ladder, "rungs": {}, "lib_curve": {}, "rent": {}}
    n_ev = int(cfg["n_eval_gate"])
    for k in [0]:                       # seam 0 carries both levels; the ladder is span-1 anyway
        S = seam_S[k][:n_ev]
        H1 = W.span(k, 1)
        for ks in ladder:
            pf = W.plan_fn(fm, H1, k_shoot=ks, cem_iters=CALP[1]["cem_iters"],
                           cem_elite=elite_for(ks), vel_pen=W.vel_pen_for(k, 1),
                           wp_mask=W.waypoint_mask(int(W.seg_lo[k]), H1))
            cm, _ = pf(S, np.tile(W.goal_schedule(k, 1)[None], (len(S), 1, 1)),
                       np.random.default_rng(cfg["seed"] + 7700 + ks))
            gK["rungs"][str(ks)] = float(realise(k, 1, S, cm).mean())
        P("[G-K] planner ladder (realised, seam 0, span 1): "
          + "  ".join(f"{ks}:{gK['rungs'][str(ks)]:.4f}" for ks in ladder))
        for ns in layout.levels(k):
            c = library.cell(ns, k)
            Ks = [int(x) for x in cfg["k_curve"].split(",") if int(x) <= len(c.tapes)]
            if len(c.tapes) not in Ks:
                Ks.append(int(len(c.tapes)))
            row = {}
            for K in Ks:
                idx = np.random.default_rng(cfg["seed"] + 991).permutation(len(c.tapes))[:K]
                pairs = np.array([[i, j] for i in range(len(S)) for j in idx], int)
                e = W.audition_fm(fm, S, c.tapes, pairs, k, ns, horizon=cfg["aud_horizon"])
                e = e.reshape(len(S), K)
                pick = idx[e.argmin(1)]
                row[str(K)] = float(realise(k, ns, S, c.tapes[pick]).mean())
            gK["lib_curve"][f"{ns}:{k}"] = row
            P(f"[G-K] library curve ns={ns}: " + "  ".join(f"{K}:{v:.4f}" for K, v in row.items()))

    gK["K_full"] = {f"{ns}:{k}": int(len(library.cell(ns, k).tapes))
                    for k in range(NS) for ns in layout.levels(k)}
    out["gK"] = gK
    save()

    # ================================================================= G-C: audition calibration
    gC = {}
    for hz in [int(x) for x in cfg["horizon_grid"].split(",")]:
        cell = {}
        for k in [0]:
            S = seam_S[k][:n_ev]
            for ns in layout.levels(k):
                c = library.cell(ns, k)
                idx = np.random.default_rng(cfg["seed"] + 992).permutation(
                    len(c.tapes))[: int(cfg["n_cal_cand"])]
                pairs = np.array([[i, j] for i in range(len(S)) for j in idx], int)
                pred = W.audition_fm(fm, S, c.tapes, pairs, k, ns, horizon=hz
                                     ).reshape(len(S), len(idx))
                true = np.stack([realise(k, ns, np.repeat(S[i:i + 1], len(idx), 0), c.tapes[idx])
                                 for i in range(len(S))])
                # matched-horizon truth, so `optimism` compares like with like: the model's score
                # is over the checkpoints INSIDE its rollout, and so is this.
                cps_h, Hh = W.aud_checkpoints(k, ns, hz)
                true_h = np.stack([
                    np.array([W.replay(S[i:i + 1], np.asarray(c.tapes[j][:Hh], np.float32),
                                       cps_h).mean() for j in idx])
                    for i in range(len(S))])
                pick_m = pred.argmin(1)
                pick_t = true.argmin(1)
                got = true[np.arange(len(S)), pick_m]
                best = true[np.arange(len(S)), pick_t]
                rho = float(np.mean([np.corrcoef(pred[i], true[i])[0, 1] for i in range(len(S))]))
                slots = c.slot[idx]
                cell[f"{ns}:{k}"] = dict(
                    rho=rho, regret=float((got - best).mean()),
                    regret_ratio=float(got.mean() / max(best.mean(), 1e-9)),
                    e_pred_mean=float(pred.mean()), e_true_mean=float(true.mean()),
                    optimism=float(true_h[np.arange(len(S)), pick_m].mean()
                                   / max(pred[np.arange(len(S)), pick_m].mean(), 1e-9)),
                    e_chosen_true=float(got.mean()), e_oracle_true=float(best.mean()),
                    e_mean_true=float(true.mean()),
                    best_fixed=float(true.mean(0).min()),
                    oracle_gain=float(true.mean(0).min() / max(best.mean(), 1e-9)),
                    n_distinct_slots=int(len(set(int(x) for x in slots[pick_t]))),
                    n_slots_present=int(len(set(int(x) for x in slots))))
        gC[str(hz)] = cell
        P(f"[G-C] horizon={hz}: " + "  ".join(
            f"{key} rho={v['rho']:.2f} regret={v['regret']:.4f} optimism={v['optimism']:.2f}x "
            f"oracle={v['oracle_gain']:.2f}x distinct={v['n_distinct_slots']}"
            for key, v in cell.items()))
    out["gC"] = gC
    save()

    # ================================================================= G-R / G-P: the port gates
    st = torch.get_rng_state().clone()
    pi = N.build_prop(W.SD, NS, layout.n_slots, cfg["seed"] + 2300, device)
    sp = N.build_span(layout.n_slots, W.SD, W.trunk_dim(), W.H_phrase, W.AD,
                      cfg["seed"] + 2500, device)
    gR = {"torch_state_unchanged": bool(torch.equal(st, torch.get_rng_state())),
          "prop_out_zero": float(sp is not None and
                                 max(float(pi.out.weight.abs().max()),
                                     float(pi.out.bias.abs().max())))}
    gR["pass"] = bool(gR["torch_state_unchanged"] and gR["prop_out_zero"] == 0.0)
    out["gR"] = gR
    P(f"[G-R] rng discipline: {gR}")

    # perturb pi so the fidelity gate is not passing on a zero head
    with torch.no_grad():
        for p in pi.parameters():
            p.add_(torch.randn(p.shape, generator=torch.Generator(device="cpu").manual_seed(7),
                               device="cpu").to(p.device) * 0.5)

    def mk_pol(mode, k_prop):
        return RoutePolicy(mode, W, layout, library, k_prop=k_prop, eps=0.0,
                           aud_horizon=cfg["aud_horizon"], delib_budget=0.0,
                           cem_ladder=ladder, cem_iters=CALP[1]["cem_iters"],
                           k_shoot=CALP[1]["k_shoot"], prim=True, seed=cfg["seed"] + 2200,
                           device=device)

    pol_a = mk_pol("audit", layout.n_slots)
    pol_a.set_norm(np.concatenate([seam_S[k] for k in range(NS)]))
    pol_p = mk_pol("prop", layout.n_slots)
    pol_p.pi = pi
    pol_p.norm, pol_p.norm_set = pol_a.norm, True
    Sg = seam_S[0][: int(cfg["n_fid"])]
    dA = pol_a.decide(W, fm, Sg, 0, np.random.default_rng(cfg["seed"] + 3131))
    dB = pol_p.decide(W, fm, Sg, 0, np.random.default_rng(cfg["seed"] + 3131))
    gP = {key: float(np.nanmax(np.abs(np.asarray(dA[key], np.float64)
                                      - np.asarray(dB[key], np.float64))))
          for key in ("cmds", "n_segs", "slot", "src", "score", "n_cand")}
    gP["aud_equal"] = bool(dA["aud"] == dB["aud"])
    gP["n_cand_mean"] = float(np.mean(dA["n_cand"]))
    gP["pass"] = bool(max(v for key, v in gP.items() if isinstance(v, float)
                          and key != "n_cand_mean") == 0.0 and gP["aud_equal"])
    # and the ladder the gate exists for: what does top-k actually cost?
    for kk in [int(x) for x in cfg["k_grid"].split(",")]:
        pol_k = mk_pol("prop", kk)
        pol_k.pi = pi
        pol_k.norm, pol_k.norm_set = pol_a.norm, True
        dk = pol_k.decide(W, fm, Sg, 0, np.random.default_rng(cfg["seed"] + 3131))
        gP[f"k{kk}"] = dict(aud=int(dk["aud"]), delib=int(dk["delib"]),
                            aud_delib=int(dk["aud_delib"]),
                            cand_mean=float(np.mean(dk["n_cand"])))
    gP["audit_all_aud"] = int(dA["aud"])
    gP["audit_all_delib"] = int(dA["delib"])
    gP["audit_all_aud_delib"] = int(dA["aud_delib"])
    out["gP"] = gP

    # ---- THE RENT TABLE, in the currency the budget conversion actually spends.
    # The spends are MEASURED (G-P's own decisions, per performer), not modelled: `delib` is the
    # FM rollout-steps one decision's audition costs, which is the same unit a CEM search is priced
    # in. For each declared per-decision budget D, which CEM rung can enumeration still afford, and
    # which can top-k routing afford -- and what is the realised error between those two rungs on
    # the ladder above. That difference IS the rent, in meters.
    n_dec = max(len(Sg), 1)
    for D in [float(x) for x in cfg["budget_grid"].split(",")]:
        sp_all = float(dA["aud_delib"]) / n_dec
        row = dict(spend_audit_all=sp_all)
        r_all = W.fit_cem(D - sp_all, W.span(0, 1), CALP[1]["cem_iters"], ladder)
        row["rung_audit_all"] = (int(r_all) if r_all else None)
        row["e_audit_all"] = (gK["rungs"].get(str(r_all)) if r_all else None)
        for kk in [int(x) for x in cfg["k_grid"].split(",")]:
            sp = float(gP[f"k{kk}"]["aud_delib"]) / n_dec
            rk = W.fit_cem(D - sp, W.span(0, 1), CALP[1]["cem_iters"], ladder)
            row[f"k{kk}"] = dict(spend=sp, rung=(int(rk) if rk else None),
                                 e=(gK["rungs"].get(str(rk)) if rk else None),
                                 rent=(None if (rk is None or r_all is None) else
                                       float(gK["rungs"][str(r_all)] - gK["rungs"][str(rk)])))
        gK["rent"][str(int(D))] = row
    out["gK"] = gK
    P(f"[G-K] rent table: {json.dumps(gK['rent'])}")
    P(f"[G-P] pi fidelity at k=N: pass={gP['pass']}  (audit_all aud={gP['audit_all_aud']}, "
      f"delib={gP['audit_all_delib']})")
    save()

    # ================================================================= G-B: does a chunk ever pay?
    # THE DECISIVE SUBSTRATE GATE, and the one most likely to say "the mapping breaks here".
    # `fingering/` measured that inside the composition horizon a LIVE plan at launch dominates every
    # frozen op 1.6x; `legato/` F4 that beyond it a measured chain wins 1.83x. So an action set
    # containing a per-segment live plan should, on ERROR alone, be won by the live plan at every
    # seam -- and if it is, the library never fires and every library readout in the main run is
    # dead. What is supposed to make a chunk pay is that deliberation is METERED (the arc's §18
    # meter): a declared per-decision budget D buys a narrower CEM, and at some D the measured tape
    # -- which carries no composition error, because the body produced it -- overtakes the plan the
    # model can still afford to search for. This gate finds that D, or reports that there is none.
    #
    # Four restricted policies at each D, all at the PIECE level so the comparison is the one the
    # main run is graded on: live plan only / segment tapes only / one chain / the whole action set.
    gB = {}
    qb = geom(cfg["batch"], cfg["seed"] + 9800)
    ab = app_plan(qb, cfg["seed"] + 9801)
    chain_levels = {ns for k in range(NS) for ns in layout.levels(k) if ns > 1}
    for D in [0.0] + [float(x) for x in cfg["budget_grid"].split(",")]:
        cellB = {}
        for name, has_prim, lv in (("prim_only", True, set()),
                                   ("seg_only", False, {1}),
                                   ("chain_only", False, chain_levels),
                                   ("all", True, None)):
            pol = mk_pol("audit", layout.n_slots)
            pol.set_norm(np.concatenate([seam_S[k] for k in range(NS)]))
            pol.delib_budget = D
            pol.prim = has_prim
            pol.levels = lv
            try:
                o = W.traverse_route(fm, pol, qb, np.random.default_rng(cfg["seed"] + 9802),
                                     cfg["sigma_perf"], led, who="instrument", kind="gB",
                                     approach_plan=ab)
                cellB[name] = dict(
                    e_piece=float(np.median(o["e_piece"])),
                    e_piece_mean=float(o["e_piece"].mean()),
                    by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
                    fb=float(o["n_fb"]), plans=int(o["n_plan"]), delib=int(o["delib"]),
                    aud_delib=int(o["aud_delib"]), aud=int(o["aud"]),
                    k_shoot=int(np.median([d["k_shoot"] for d in o["decisions"]])),
                    frac_chain=float(np.mean([x > 1 for d in o["decisions"]
                                              for x in d["n_segs"]])))
            except RuntimeError as ex:
                cellB[name] = {"error": str(ex)}
        gB[str(int(D))] = cellB
        P(f"[G-B] D={int(D)}: " + "  ".join(
            f"{nm}={(v.get('e_piece') if 'e_piece' in v else 'ERR')}"
            if 'e_piece' not in v else f"{nm}={v['e_piece']:.4f}(k{v['k_shoot']},fb{v['fb']:.0f})"
            for nm, v in cellB.items()))
    out["gB"] = gB
    save()

    # ================================================================= G-D: cycle cost
    gD = {"warm_s_per_reactive_cycle": warm_s / max(int(cfg["n_warm"]), 1)}
    for mode, kp in (("audit", layout.n_slots), ("prop", int(cfg["k_prop"]))):
        pol = mk_pol(mode, kp)
        pol.set_norm(np.concatenate([seam_S[k] for k in range(NS)]))
        if mode == "prop":
            pol.pi = pi
        qp = geom(cfg["batch"], cfg["seed"] + 9911)
        t1 = time.time()
        o = W.traverse_route(fm, pol, qp, np.random.default_rng(cfg["seed"] + 9912),
                             cfg["sigma_practice"], led, who="instrument", kind="gD",
                             approach_plan=app_plan(qp, cfg["seed"] + 9913))
        gD[mode] = dict(s_per_traversal=time.time() - t1, aud=int(o["aud"]),
                        delib=int(o["delib"]), plans=int(o["n_plan"]),
                        e_piece=float(np.median(o["e_piece"])),
                        frac_chain=float(np.mean([x > 1 for d in o["decisions"]
                                                  for x in d["n_segs"]])))
        P(f"[G-D] {mode}: {gD[mode]['s_per_traversal']:.1f}s/traversal  "
          f"aud={gD[mode]['aud']}  e={gD[mode]['e_piece']:.4f}  "
          f"chain={gD[mode]['frac_chain']:.2f}")
    out["gD"] = gD
    out["ledger"] = led.snapshot()
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P("[done]")
    return out


@app.local_entrypoint()
def offbook_gates(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_warm: int = 20,
    lib_k: int = 288,
    n_slot: int = 8,
    k_prop: int = 2,
    k_grid: str = "1,2,4,8",
    k_curve: str = "1,4,16,64,144,288",
    budget_grid: str = "20480,40960,81920,163840,327680",
    horizon_grid: str = "0,20",
    aud_horizon: int = 0,
    cem_ladder: str = "32,64,128,256,512,1024,2048",
    n_score: int = 64,
    n_eval_gate: int = 24,
    n_cal_cand: int = 32,
    n_fid: int = 8,
    # ---- the world (legato's calibrated l2 cell, verbatim)
    curl_b: float = 14.0,
    push_a: float = 0.0,
    waypoints: str = DEF_WPS,
    h_app: int = 14,
    h_seg: str = "20,20,20",
    patch_seg: int = DEF_PATCH_SEG,
    q_jit: float = 0.15,
    null_jit: float = 0.30,
    start_mode: str = "iso",
    patch_sigma: float = 0.08,
    patch_center: str = "",
    push_sigma: float = 0.0,
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,1.1,0.8",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    timestep: float = 0.002,
    wrap_limit: float = 3.0,
    pool_ou: int = 6000,
    pool_reach: int = 12000,
    exclude_w: float = 0.1,
    ep_len: int = 14,
    n_par: int = 16,
    op_q_range: float = 0.25,
    sigma_u: float = 0.15,
    ou_sigma: float = 0.7,
    ou_theta: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    fm_steps_boot: int = 3000,
    adapt_lr: float = 3e-4,
    n_grad: int = 6,
    replay_frac: float = 0.5,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.0,
    vel_pen_mid: float = 0.0,
    w_waypoint: float = 16.0,
    lookahead_gamma: float = 0.0,
    react_look: int = 20,
    plan_max_elems: int = 40_000_000,
    calp: str = "1:1024:8,2:4096:12,3:4096:12",
    batch: int = 16,
    trace_window: int = 10,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500
        fm_steps = 1000; fm_steps_boot = 500
        n_warm = 3; lib_k = 24; n_slot = 3; batch = 8
        k_grid = "1,2"; k_curve = "1,4,12,24"; n_score = 16; n_eval_gate = 6
        n_cal_cand = 8; n_fid = 4
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,2:256:4,3:256:4"
        cem_ladder = "64,128,256"; budget_grid = "5120,10240"
        tag = tag or "gsmoke"
    tag = tag or "g0"
    cfg = dict(
        tag=tag, seed=seed, n_warm=n_warm, lib_k=lib_k, n_slot=n_slot, k_prop=k_prop,
        k_grid=k_grid, k_curve=k_curve, budget_grid=budget_grid, horizon_grid=horizon_grid,
        aud_horizon=aud_horizon, cem_ladder=[int(x) for x in cem_ladder.split(",") if x],
        n_score=n_score, n_eval_gate=n_eval_gate, n_cal_cand=n_cal_cand, n_fid=n_fid,
        curl_b=curl_b, push_a=push_a,
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints.split(";") if p],
        h_app=h_app, h_seg=[int(v) for v in h_seg.split(",")], patch_seg=patch_seg,
        q_jit=q_jit, null_jit=null_jit, start_mode=start_mode, patch_sigma=patch_sigma,
        patch_center=([float(x) for x in patch_center.split(",")] if patch_center else None),
        push_sigma=(push_sigma or patch_sigma), push_center=None,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit, pool_ou=pool_ou, pool_reach=pool_reach, exclude_w=exclude_w,
        ep_len=ep_len, n_par=n_par, op_q_range=op_q_range, sigma_u=sigma_u,
        ou_sigma=ou_sigma, ou_theta=ou_theta, collect_k_shoot=collect_k_shoot,
        collect_cem_iters=collect_cem_iters, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr,
        fm_batch=fm_batch, fm_steps=fm_steps, fm_steps_boot=fm_steps_boot,
        adapt_lr=adapt_lr, n_grad=n_grad, replay_frac=replay_frac,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, vel_pen_mid=vel_pen_mid,
        w_waypoint=w_waypoint, lookahead_gamma=lookahead_gamma, react_look=react_look,
        plan_max_elems=plan_max_elems,
        calp={p.split(":")[0]: [int(p.split(":")[1]), int(p.split(":")[2])]
              for p in calp.split(",") if p},
        batch=batch, trace_window=trace_window, sigma_practice=sigma_practice,
        sigma_perf=sigma_perf, d_fb=d_fb,
    )
    o = run_offbook_gates.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "gates.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/gates.json")
