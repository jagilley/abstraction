"""S2 -- disentangling S1's behavioural negative: corridor narrowing vs planner starvation.

S1 measured, on `legato`'s piece with nothing committed and `e_react` flat at ~0.0097 throughout:
the ON-CORRIDOR composition horizon GROWS with practice (21 -> 26 at 0.05 m, 11 -> 22 at 0.02 m,
rho ~ +0.77), while the executed span of a single live phrase plan does NOT (one waypoint
delivered, every cycle) and its FAR END gets 5x WORSE (W0 arrival 0.179 -> 0.928 m across the
c22->c26 regime change). Two named suspects, which S1 cannot tell apart:

  (1) CORRIDOR NARROWING. The forward model sharpens on the narrowing on-policy corridor and
      degrades off it, so a fixed-sizing CEM -- which searches wherever the model says it is
      cheap -- exploits that off-corridor error more confidently each cycle.
  (2) PLANNER STARVATION. `legato`'s CAL-P recorded the phrase span as still improving at the top
      of its grid on the stale model. S1 held CAL-P's sizing fixed at every snapshot, so the span
      may simply be search-limited and the degradation an artifact of a fixed, too-small search.

THE SEPARATION, one container, NO RETRAINING -- every model is loaded from S1's `fm_snapshots.pt`
and every geometry, approach plan and reference set is regenerated deterministically from S1's
own config and seed offsets, so this run is a re-measurement of S1's models, not a second run.

  A. THREE DIVERGENCE SOURCES per snapshot, all from the SAME launch states at the SAME motor
     noise, so the only thing that varies is where the commands came from:
       `fixed`   -- the frozen cycle-0 reactive reference commands (S1's on-corridor number)
       `current` -- a reactive traversal under THIS snapshot, from the same launch states
                    (the corridor the model is currently practising)
       `plan`    -- the snapshot's OWN phrase plan, flown open-loop (where the CEM actually goes)
     Suspect (1) predicts the `plan` curve decaying while `fixed`/`current` improve.

  B. THE EXPLOITATION GAP. The CEM optimises under the model, so the model's PREDICTED tip at each
     seam is what it believed it had bought. `gap = true arrival - predicted arrival` is the size
     of the belief that did not survive contact with the plant.

  C. ONE-STEP FM ERROR, on-corridor vs off-corridor, on matched transitions. The narrowing
     hypothesis in its most direct form.

  D. CORRIDOR GEOMETRY per cycle: the spread of the practice traversal's visited states, and the
     nearest-corridor-state distance of the plan's own trajectory. The practice traversal at cycle
     c+1 is reproduced EXACTLY (snapshot c is the model that flew it, and the geometry and noise
     seeds are deterministic), so this is S1's actual diet, not a proxy.

  E. CAL-P PER SNAPSHOT. At eight cycles spanning the regime change, sweep the planner grid and
     record BOTH the true and the predicted seam arrivals at every sizing. Suspect (2) predicts
     the far-end degradation vanishing as the search grows. Suspect (1) predicts the predicted
     arrival IMPROVING while the true one does not -- more search buying more exploitation.

RESUMABLE. `results.json` is rewritten whole after every snapshot and every CAL-P cycle, and the
run skips anything already in it (`--resume`, default true). The function also carries Modal
retries, so a preempted container restarts and picks up where it left off.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/span/s2.py::s2 --quick
    python3 mjc/practice/span/launch_detached.py --fn s2 --tag S2 --src-tag S1
    python3 mjc/practice/span/analyze_s2.py --tag S2 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

THRS = (0.01, 0.02, 0.05, 0.10, 0.20)
# the planner grid for the per-snapshot CAL-P. `legato`'s chosen phrase sizing is (4096, 12); the
# grid brackets it by 4x below and 4x/2x above so "still improving at the top" is answerable.
DEF_GRID = "1024:8,2048:10,4096:12,8192:16,16384:24"
# available snapshots are {0, 5, ..., 80} (S1 `weight_every=5`); these bracket the c22->c26 change
DEF_CALP_CYCLES = "0,5,15,20,25,30,55,80"


# `retries` + the resume block below are the answer to S2's first launch, which lost its
# container to PREEMPTION at c50/17 ("Container terminated due to preemption" in the Modal
# task log; the client saw only `RemoteError: cancelled by user or a failure`). A retry
# re-enters at the top, re-reads the committed `results.json`, and skips everything already
# measured -- so a preemption now costs the reference-set rebuild plus the in-flight
# snapshot, not the run.
@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume},
              retries=modal.Retries(max_retries=3, backoff_coefficient=1.0,
                                    initial_delay=10.0))
def run_s2(c2: dict) -> dict:
    import numpy as np
    import torch

    from mjc.practice.legato.world import World, Ledger, start_postures, elite_for

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = os.path.join(DATA_DIR, "practice_span", c2["src_tag"])
    S1 = json.load(open(os.path.join(src, "results.json")))
    cfg = S1["config"]                                   # S1's config, verbatim
    snap = torch.load(os.path.join(src, "fm_snapshots.pt"), map_location="cpu",
                      weights_only=False)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    W = World(cfg, device)
    W.norm = {k: v.to(device) for k, v in snap["norm"].items()}     # set once in S1, before the
    n, NS, AD, SD = W.n, W.n_seg, W.AD, W.SD                        # loop; shared by every snapshot
    led = Ledger()
    out = {"config": c2, "src_config": cfg, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_span", c2["tag"])
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[s2]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ---- RESUME. The committed `results.json` is rewritten whole after every snapshot and every
    # CAL-P cycle, so anything in it is finished work. Rebuilding the reference set is cheap
    # (~50 s) and re-runs the fidelity check, so it is always redone.
    rows, calp = [], []
    if c2.get("resume", True):
        prev_p = os.path.join(outdir, "results.json")
        if os.path.isfile(prev_p):
            try:
                prev = json.load(open(prev_p))
                rows = [r for r in prev.get("snapshots", []) if "corridor" in r]
                calp = [r for r in prev.get("calp", []) if r.get("grid")]
                print(f"[s2] [resume] found {len(rows)} snapshots "
                      f"{[r['cycle'] for r in rows]} and {len(calp)} CAL-P cycles "
                      f"{[r['cycle'] for r in calp]}", flush=True)
            except Exception as e:                                  # noqa: BLE001
                print(f"[s2] [resume] could not read prior results ({e}); starting fresh",
                      flush=True)
                rows, calp = [], []
    out["snapshots"], out["calp"] = rows, calp
    done_snap = {int(r["cycle"]) for r in rows}
    done_calp = {int(r["cycle"]) for r in calp}

    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}
    GRID = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in c2["grid"].split(",") if p]
    cycles = sorted(int(k) for k in snap["weights"])
    if c2["max_snap"]:
        cycles = cycles[:c2["max_snap"]]
    calp_cycles = [c for c in (int(x) for x in c2["calp_cycles"].split(",") if x) if c in cycles]
    P(f"[setup] device={device} src={c2['src_tag']} snapshots={cycles} calp_at={calp_cycles} "
      f"grid={GRID} calp_chosen={cfg['calp']}")

    def load_fm(cycle):
        net = W.mlp(cfg["seed"] + 41)
        net.load_state_dict({k: v.to(device) for k, v in snap["weights"][cycle].items()})
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    fm0 = load_fm(0)          # S1 saved weights[0] before ANY online update, so this IS `fm0`
    fm_app = fm0

    def geom(m, seed):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    R_REACT = W.routing("reactive")
    SCHED = W.goal_schedule(0, NS)
    SEAM = [int(x) - 1 for x in W.seg_hi]
    WP = np.stack([W.goals[k + 1] for k in range(NS)]).astype(np.float64)

    # ---------------------------------------------------------------- the fixed reference set
    ex1 = W.traverse(fm0, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 7001),
                     cfg["sigma_perf"], led, who="instrument", kind="ref_lap1",
                     approach_plan=plan_ev)
    S0_ref, fin = ex1["launch"], ex1["final"]
    ex2 = W.traverse(fm0, R_REACT, fin[:, :n].astype(np.float64),
                     np.random.default_rng(cfg["seed"] + 7011), cfg["sigma_perf"], led,
                     who="instrument", kind="ref_lap2", qd0=fin[:, n:].astype(np.float64))
    cmd_ref = np.concatenate([ex1["acts"], ex2["acts_app"], ex2["acts"]], axis=1)
    true_ref = np.concatenate([ex1["tips"], ex2["tips_app"], ex2["tips"]], axis=1)
    H_REF = cmd_ref.shape[1]
    fid_launch = float(np.abs(W.tip(S0_ref) - np.asarray(S1["reference"]["launch_tip"])).max())
    P(f"[fidelity] reference set rebuilt: launch tips match S1 to {fid_launch:.2e} m | "
      f"H_REF={H_REF} | lap1 e_piece={float(np.median(ex1['e_piece'])):.4f} (S1 "
      f"{S1['reference']['e_piece_lap1']:.4f})")

    # ---------------------------------------------------------------- helpers
    def true_states(S0, cmds):
        """Ground-truth STATE after each of `cmds`' steps, executed in the real plant from each of
        `S0`. `World.true_tips` with the state kept, so one plant pass yields the tips AND the
        transitions a one-step FM error needs."""
        H = cmds.shape[1]
        out_ = np.empty((len(S0), H, SD), np.float32)
        for i in range(len(S0)):
            W.env.set_state(S0[i, :n].astype(np.float64), S0[i, n:].astype(np.float64))
            for hh in range(H):
                W.env.step(cmds[i, hh], W.fs)
                out_[i, hh] = W.env.get_state()
        return out_

    def tips_of(states):
        B, H, _ = states.shape
        return W.tip(states.reshape(-1, SD)).reshape(B, H, 2)

    def cross(curve, thr):
        idx = np.nonzero(np.asarray(curve) > thr)[0]
        return int(idx[0] + 1) if len(idx) else int(len(curve) + 1)

    def curve_row(d):
        med = np.median(d, 0)
        return dict(med=[float(v) for v in med], h={f"{t:g}": cross(med, t) for t in THRS})

    def divergence(net, S0, cmds, truth):
        pred = W.fm_rollout_tips(net, S0, cmds)
        row = curve_row(np.linalg.norm(pred - truth, axis=2))
        row["pred_tips"] = pred
        return row

    def one_step_err(net, S, U, S2):
        """Median |predicted delta - true delta| over transitions, in state units."""
        return float(np.median(np.linalg.norm(W.fm_delta(net, S, U) - (S2 - S), axis=1)))

    # state normaliser (the first SD columns of the [S,U] normaliser) for corridor geometry
    mx = W.norm["mx"].detach().cpu().numpy()[:SD]
    sx = W.norm["sx"].detach().cpu().numpy()[:SD]

    def nrm(S):
        return (np.asarray(S, np.float64) - mx) / sx

    def phrase_plan(net, seed, k_shoot=None, cem_iters=None):
        """One live phrase plan from W1, flown open-loop with performance motor noise. Identical
        rng path to S1's `executed_span` when the CAL-P sizing is used, so `e_seam` reproduces."""
        rng = np.random.default_rng(seed)
        kw = (CALP[NS] if k_shoot is None else
              dict(k_shoot=int(k_shoot), cem_iters=int(cem_iters),
                   cem_elite=elite_for(int(k_shoot))))
        pf = W.plan_fn(net, W.H_phrase, vel_pen=W.vel_pen_for(0, NS),
                       wp_mask=W.waypoint_mask(0, W.H_phrase), **kw)
        cmds, delib = pf(S0_ref, np.tile(SCHED[None], (len(S0_ref), 1, 1)), rng)
        cmds = np.clip(cmds, -1, 1)
        noisy = np.clip(cmds + cfg["sigma_perf"] * rng.standard_normal(cmds.shape),
                        -1, 1).astype(np.float32)
        st = true_states(S0_ref, noisy)
        tp = tips_of(st)
        e_seam = [float(np.median(np.linalg.norm(tp[:, SEAM[k]] - WP[k][None], axis=1)))
                  for k in range(NS)]
        return dict(cmds=cmds, noisy=noisy, states=st, tips=tp, e_seam=e_seam, delib=int(delib))

    def seam_pair(net, pl):
        """(true, predicted) arrival error at each seam under the plan the CEM actually chose."""
        pred = W.fm_rollout_tips(net, S0_ref, pl["noisy"])
        p = [float(np.median(np.linalg.norm(pred[:, SEAM[k]] - WP[k][None], axis=1)))
             for k in range(NS)]
        return pl["e_seam"], p

    s1_span = {s["cycle"]: s["span"]["e_seam"] for s in S1["snapshots"] if "span" in s}

    # ================================================================= per-snapshot measurement
    t0 = time.time()
    for c in cycles:
        if c in done_snap:
            continue
        t1 = time.time()
        fm = load_fm(c)
        rec = {"cycle": c}

        # --- A1 fixed reference corridor (S1's on-corridor number, re-measured here) ---
        d_fixed = divergence(fm, S0_ref, cmd_ref, true_ref)
        rec["div_fixed"] = {k: v for k, v in d_fixed.items() if k != "pred_tips"}

        # --- A2 the corridor this snapshot is currently practising ---
        # same q_ev, same mastered approach plan, same rng seed -> the SAME launch states; only the
        # reactive commands differ, because the model choosing them has moved.
        exc = W.traverse(fm, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 7001),
                         cfg["sigma_perf"], led, who="instrument", kind="cur_corridor",
                         approach_plan=plan_ev, collect=True)
        rec["launch_match"] = float(np.abs(exc["launch"] - S0_ref).max())
        d_cur = divergence(fm, S0_ref, exc["acts"], exc["tips"])
        rec["div_current"] = {k: v for k, v in d_cur.items() if k != "pred_tips"}
        rec["e_react"] = float(np.median(exc["e_piece"]))

        # --- A3/B off-corridor: the snapshot's own phrase plan, and what it believed ---
        pl = phrase_plan(fm, cfg["seed"] + 7300 + c)
        d_plan = divergence(fm, S0_ref, pl["noisy"], pl["tips"])
        rec["div_plan"] = {k: v for k, v in d_plan.items() if k != "pred_tips"}
        true_s, pred_s = seam_pair(fm, pl)
        rec["e_seam"] = true_s
        rec["e_seam_pred"] = pred_s
        rec["exploit_gap"] = [float(a - b) for a, b in zip(true_s, pred_s)]
        rec["e_piece"] = float(np.mean(true_s))
        rec["fidelity_e_seam"] = (
            [float(a - b) for a, b in zip(true_s, s1_span[c])] if c in s1_span else None)

        # --- C/D the diet at cycle c+1 -- EXACTLY the traversal S1 trained on next ---
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c + 1)
        pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + c + 1),
                        cfg["sigma_practice"], led, who="agent", kind="practice_repro",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c + 1), collect=True)
        Sc, Uc, S2c = pr["trans"]
        rec["fm_err_corridor"] = one_step_err(fm, Sc, Uc, S2c)
        # matched-noise on-corridor control: the perf-tempo reactive traversal collected above
        # flies at the SAME sigma the plan does, so this comparison is not confounded by the
        # practice diet's higher motor noise.
        Sr, Ur, S2r = exc["trans"]
        rec["fm_err_corridor_perf"] = one_step_err(fm, Sr, Ur, S2r)
        # matched transitions off-corridor: the plan's own executed trajectory
        Sp = np.concatenate([S0_ref[:, None, :], pl["states"][:, :-1, :]], 1).reshape(-1, SD)
        Up = pl["noisy"].reshape(-1, AD)
        S2p = pl["states"].reshape(-1, SD)
        rec["fm_err_plan"] = one_step_err(fm, Sp, Up, S2p)
        rec["fm_err_ratio"] = rec["fm_err_plan"] / max(rec["fm_err_corridor"], 1e-12)
        rec["fm_err_ratio_perf"] = rec["fm_err_plan"] / max(rec["fm_err_corridor_perf"], 1e-12)

        # corridor geometry + how far off it the plan goes
        Cn, Pn = nrm(Sc), nrm(Sp)
        ev = np.linalg.eigvalsh(np.cov(Cn.T) + 1e-12 * np.eye(SD))
        d2 = ((Pn[:, None, :] - Cn[None, ::3, :]) ** 2).sum(-1)
        nnd = np.sqrt(d2.min(1))
        rec["corridor"] = dict(
            n=int(len(Cn)), logdet=float(np.log(np.maximum(ev, 1e-12)).sum()),
            trace=float(ev.sum()), tip_area=float(np.sqrt(max(
                np.linalg.det(np.cov(W.tip(Sc).T)), 1e-18))),
            speed=float(np.linalg.norm(Sc[:, n:], axis=1).mean()),
            plan_nn_med=float(np.median(nnd)), plan_nn_p90=float(np.percentile(nnd, 90)),
            plan_nn_by_step=[float(np.median(np.sqrt(d2.reshape(
                len(S0_ref), W.H_phrase, -1).min(2))[:, hh])) for hh in range(W.H_phrase)])

        rows.append(rec)
        rows.sort(key=lambda r: r["cycle"])
        P(f"[c{c:3d}] div h@0.05  fixed {d_fixed['h']['0.05']:3d}  current "
          f"{d_cur['h']['0.05']:3d}  PLAN {d_plan['h']['0.05']:3d} | 1-step err corridor "
          f"{rec['fm_err_corridor_perf']:.4f} plan {rec['fm_err_plan']:.4f} "
          f"({rec['fm_err_ratio_perf']:.2f}x)"
          f" | seam true {[round(v,3) for v in true_s]} pred {[round(v,3) for v in pred_s]} | "
          f"corridor logdet {rec['corridor']['logdet']:.2f} plan-nn "
          f"{rec['corridor']['plan_nn_med']:.3f} | fid "
          f"{[round(v,4) for v in rec['fidelity_e_seam']] if rec['fidelity_e_seam'] else '-'} | "
          f"{time.time() - t1:.1f}s")
        out["snapshots"] = rows
        save()
    P(f"[phase A-D done] {time.time() - t0:.0f}s")

    # ================================================================= E: CAL-P per snapshot
    for c in calp_cycles:
        if c in done_calp:
            continue
        fm = load_fm(c)
        row = {"cycle": c, "grid": []}
        for ks, ci in GRID:
            t1 = time.time()
            pl = phrase_plan(fm, cfg["seed"] + 7300 + c, k_shoot=ks, cem_iters=ci)
            ts, ps = seam_pair(fm, pl)
            row["grid"].append(dict(k_shoot=ks, cem_iters=ci, delib=pl["delib"],
                                    e_seam=ts, e_seam_pred=ps,
                                    gap=[float(a - b) for a, b in zip(ts, ps)],
                                    e_piece=float(np.mean(ts)),
                                    e_piece_pred=float(np.mean(ps)),
                                    wall=float(time.time() - t1)))
            g = row["grid"][-1]
            P(f"[CAL-P c{c:3d}] k={ks:6d} i={ci:2d} -> e_piece {g['e_piece']:.4f} "
              f"(model believed {g['e_piece_pred']:.4f}) seam {[round(v,3) for v in ts]} "
              f"pred {[round(v,3) for v in ps]} | {g['wall']:.1f}s")
        v = [g["e_piece"] for g in row["grid"]]
        vp = [g["e_piece_pred"] for g in row["grid"]]
        row["still_improving"] = bool(v[-1] < v[-2] - 0.01)
        row["last_step_gain"] = float(v[-2] - v[-1])
        row["best_e_piece"] = float(min(v))
        row["best_at"] = GRID[int(np.argmin(v))]
        row["w0_at_top"] = float(row["grid"][-1]["e_seam"][-1])
        row["pred_monotone_down"] = bool(vp[-1] <= vp[0])
        calp.append(row)
        calp.sort(key=lambda r: r["cycle"])
        P(f"[CAL-P c{c:3d}] still improving at top? {row['still_improving']} "
          f"(last gain {row['last_step_gain']:+.4f}) | best {row['best_e_piece']:.4f} at "
          f"{row['best_at']} | W0 at the largest search {row['w0_at_top']:.4f} | model's belief "
          f"falls with search? {row['pred_monotone_down']}")
        out["calp"] = calp
        save()

    out.update(snapshots=rows, calp=calp, ledger=led.snapshot(), complete=True,
               reference=dict(H_ref=H_REF, seam=SEAM, launch_fidelity=fid_launch,
                              e_piece_lap1=float(np.median(ex1["e_piece"]))))
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[done] snapshots={len(rows)} calp_cycles={len(calp)} total={time.time() - t0:.0f}s")
    return out


@app.local_entrypoint()
def s2(quick: bool = False, tag: str = "", src_tag: str = "S1", grid: str = DEF_GRID,
       calp_cycles: str = DEF_CALP_CYCLES, max_snap: int = 0, resume: bool = True):
    if quick:
        grid = "256:4,512:6"
        calp_cycles = "0,5"
        max_snap = 2
        tag = tag or "s2smoke"
    tag = tag or "S2"
    c2 = dict(tag=tag, src_tag=src_tag, grid=grid, calp_cycles=calp_cycles, max_snap=max_snap,
              resume=bool(resume))
    o = run_s2.remote(c2)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/results.json")
