"""accelerando Phase 2 — the program: does a unit survive a tempo it was never practiced at?

WHAT PHASE 1 LEFT. `t1` measured, on a body with tau = 30 ms at a fixed 120 ms delay, that a
recording harvested AT ITS OWN TEMPO beats the per-tempo re-fit reflex by 1.18x / 2.32x / 2.96x /
10.02x as the note shortens from 384 to 48 ms, at one feedback event against 128 to 16; that era_1
exists at H = 2 (the incumbent leaves the 1/2-leg band, every level stays inside); and that the
SAME recording asked for at another tempo is worthless — `scl0` (the slow tape, note-wise
time-resampled) is 10.2x / 11.9x / 17.7x / 35.6x worse than the same-tempo recording at level 4,
in band at no tempo but its own, and the s^2 inertial correction is worse still.

So the library is exactly the thing a tempo ladder destroys, and a ratchet cannot be built on it.

THE QUESTION PHASE 2 ASKS. Replace the tape by a PROGRAM — a phase-conditioned emitter
`u(phi, z, slot, tempo)` trained by self-imitation on the body's own executed traversals at the
tempi practiced so far — and ask whether the program survives what the recording did not:

    at a tempo the learner has NEVER practiced at, does the head's emission beat the best
    zero-practice content it has (the resampled slow tape), and does it reach the recording that
    practice there would have bought?

The head is gated per slot per tempo by PARITY ON THE PLANT — solo's gate, execution reproduction
from held-out launch states — so "the head is trusted here" is a measurement and not a hope. At a
practiced tempo the reference is the same-tempo recording (solo's gate verbatim, and `solo`
finding 6's question: does the head beat the tape it was trained on?). At an unpracticed tempo
there IS no same-tempo recording, so the deployable reference is `scl0` and the oracle reference
is the recording practice would have made; both are run and both are reported.

STANDING CONSTRAINT, verbatim from `acappella/`, `solo/`, `prestissimo/` and Phase 1: **no forward
model anywhere.** The head predicts no state, is never rolled forward, and the plant is still the
only oracle. Phase 2 has exactly one learned component and it emits commands.

THE GATES
  P-F0   the 13 donor constants, `ast`-read out of `etude/etude.py`, never imported.
  P-T1   Phase 2 rebuilds `t1`'s five libraries from the same seeds and reproduces its 75
         construction scores and 50 pinned sweep numbers at max|delta| = 0. Since Phase 2 adds no
         code to `world.py` or `piece.py`, this certifies the shared substrate has not moved, and
         `t1`'s own P-F1 (acappella/b1) and P-F2 (prestissimo/a1g) are inherited through it.
  P-G    the pinned per-(tempo, Delta) gain table is not taken on trust: one cell is re-fit from
         the full 63-cell grid and must reproduce `t1`'s choice and its error exactly.
  P-N    the nesting op is exact, at every tempo (carried).
  P-R    the resampler is the identity at s = 1 (carried).
  P-H    the head's held-out states are never trained on: the train and hold splits of every
         (slot, tempo) are disjoint by `split_code`, asserted rather than assumed.

Run:
    cd experiments/                        # MODAL_PROFILE=chromatic
    modal run mjc/practice/tempo/accelerando/program.py::accelerando_program --quick --tag psmoke
    modal run --detach mjc/practice/tempo/accelerando/program.py::accelerando_program --spawn --tag p1
    python3 mjc/practice/tempo/accelerando/analyze_program.py --tag p1 --fetch --figures
"""

import json
import os
import time

import modal  # noqa: F401  (the app is imported from mjc.shared; kept for parity with siblings)

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.tempo.accelerando import piece as P
from mjc.practice.tempo.accelerando import t1_pins as T1

# Copied from `tempo.py` rather than imported: importing a sibling runner registers ITS
# `@app.function` / `@app.local_entrypoint` into the one shared Modal app, which `offbook/FILES.md`
# records as a gotcha and every node in this arc has avoided. The values are `t1`'s and gate P-G
# re-fits one cell of the grid they define, so a drift here cannot pass silently.
T_KP_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
T_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
APP_MODES = {"arm": None, "fixed": 0}
APP_D0_GAINS = {"arm": False, "fixed": True}

DELTAS_INSTR = (0, 2, 5, 8, 12)

# THE PRACTICED SET. The head is trained on these tempi and asked at all five. Two heads, so the
# extrapolation DISTANCE is a measured axis rather than one cell: `prog` practices the three slow
# rungs and extrapolates one and two rungs past them; `prog2` practices two and extrapolates
# three. Both are minted from independent generators (`build_prog`) and trained on the same
# buffer, so the only difference between them is which tempi were legal targets.
PRACTICED = (32, 16, 8)
PRACTICED_SHORT = (32, 16)


@app.function(cpu=16.0, gpu="L4", memory=65536, timeout=28800,
              volumes={DATA_DIR: volume})
def run_program(cfg: dict) -> dict:
    import ast

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(int(cfg["torch_threads"]))
    # An L4 for the head, 16 CPUs for the plant. The `psmoke` run measured why: at 6 epochs on
    # CPU the head sat at mse 0.0293 against a target mean-square of ~0.04 and `prog_L4` came
    # back 20x worse than the tape it was cloning — the head was UNTRAINED, not incapable, and
    # the fix is hundreds of epochs, which is ~1 ms a step on a GPU and ~50 ms on these cores.
    # The GPU idles through the ~900 s of plant work; that is the cheaper mistake.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mjc.practice.tempo.accelerando.world import (World, Ledger, ReflexDecider, LevelLibrary,
                                                LadderLayout, LadderDecider, build_level1_cell,
                                                build_level_cell, FrozenReflexDecider,
                                                resample_library)
    from mjc.practice.tempo.accelerando.nets import (ProgBuffer, ProgDecider, build_prog, train_prog,
                                               make_prog_fn, parity, ref_errors)

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_accelerando", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "phase": "2",
           "contract": dict(
               no_forward_model=("The head maps (phase, posture, slot, tempo) -> command. It "
                                 "predicts no state, is never rolled forward, and nothing in "
                                 "this node imports, trains or evaluates an f(s,u). The plant is "
                                 "still the only oracle."),
               practiced=("The head is trained ONLY on executed traversals at the practiced "
                          "tempi; every faster rung is extrapolation. Recordings at an "
                          "unpracticed tempo are reported as the ORACLE reference (what practice "
                          "there would have bought) and are never available to any prog arm."),
               parity=("Per slot per tempo, execution reproduction on the plant from HELD-OUT "
                       "launch states split by a deterministic code of the state. Two "
                       "references: `rec` (the same-tempo recording — solo's gate verbatim, an "
                       "oracle at an unpracticed tempo) and `scl0` (the resampled slow tape — "
                       "the deployable one). Continuous fractions logged at every cell."),
               grades=("Both of Phase 1's, unchanged: the per-waypoint drilled error with the "
                       "1/2-leg band, and the listener's-clock error at a fixed 384 ms window. "
                       "Pass fractions at 1/4, 1/3, 1/2 mean leg for both."),
               delta=("Delta is FIXED at 5 steps for every headline number. The Delta sweep is "
                      "read at L1 and L2 ONLY: under `app=fixed` the deep arms make one decision "
                      "from a lead-in that does not move with Delta and are Delta-flat BY "
                      "CONSTRUCTION (t1 measured key_L4 at 1.00x-1.09x spread), so they cannot "
                      "say anything about feedback. L1 and L2 re-decide 8 and 4 times."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "program.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ================================================================= P-F0: the donor constants
    src = None
    for root in ("/root", os.getcwd()):
        p = os.path.join(root, "mjc/practice/etude/etude.py")
        if os.path.exists(p):
            src = p
            break
    if src is None:
        import mjc
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mjc.__file__))),
                           "mjc/practice/etude/etude.py")
    tree = ast.parse(open(src).read())
    mod_const, fn_def = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                mod_const[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
        if isinstance(node, ast.FunctionDef) and node.name == "etude":
            args, defs = node.args.args, node.args.defaults
            for a, dd in zip(args[len(args) - len(defs):], defs):
                try:
                    fn_def[a.arg] = ast.literal_eval(dd)
                except Exception:
                    pass
    want = {"DEF_WAYPOINTS": (mod_const.get("DEF_WAYPOINTS"), P.DEF_WAYPOINTS),
            "DEF_REGIONS": (mod_const.get("DEF_REGIONS"), P.DEF_REGIONS),
            "seg_h": (fn_def.get("seg_h"), P.SEG_H),
            "frame_skip": (fn_def.get("frame_skip"), P.FRAME_SKIP),
            "gear": (fn_def.get("gear"), P.GEAR), "damping": (fn_def.get("damping"), P.DAMPING),
            "arena_half": (fn_def.get("arena_half"), P.ARENA_HALF),
            "start_jit": (fn_def.get("start_jit"), P.START_JIT),
            "v0_std": (fn_def.get("v0_std"), P.V0_STD),
            "explore_sigma": (fn_def.get("explore_sigma"), P.EXPLORE_SIGMA),
            "d_fb": (fn_def.get("d_fb"), P.D_FB),
            "drill_seg": (fn_def.get("drill_seg"), P.DRILL_SEG),
            "box_half": (fn_def.get("box_half"), P.BOX_HALF)}
    bad = {k: list(v) for k, v in want.items() if v[0] != v[1]}
    out["P_F0"] = dict(checked=len(want), mismatches=bad, **{"pass": not bad})
    PR(f"[P-F0] donor constants: {len(want)} checked, {len(bad)} mismatched  pass={not bad}")
    if bad:
        save()
        raise SystemExit("P-F0 failed: the fork has drifted from the donor")

    # =========================================================== the static build (t1's op)
    # `tempo.py`'s `build_ladder`, re-stated here rather than imported: it is nested inside that
    # runner's Modal function. The copy cannot silently drift, because gate P-T1 asserts its
    # output equals `t1`'s at max|delta| = 0 — which is a stronger check than sharing the source.
    def build_ladder(W, pc, kc, seed, harvest_gains, llb, rt):
        K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
        W.kp, W.kd = float(harvest_gains[0]), float(harvest_gains[1])
        lvlib = LevelLibrary(K, kc["n_slot"], L, poison=True)
        b1_rec, bl_rec = [], []
        pool_stats = []

        def prefix_dec(level_map, mode="key"):
            return LadderDecider(W, lvlib, mode=mode, level_map=level_map, fallback=False)

        def harvest(level_map, n_cycles, batch, seed0):
            pools = {k: dict(cmds=[], raw=[], s0=[], err=[], traj=[]) for k in range(W.K)}
            led = Ledger(W.dt_ctrl, W.d_fb)
            for c in range(n_cycles):
                rng = np.random.default_rng(seed0 + c)
                o = W.traverse(prefix_dec(level_map), W.start_states(rng, batch), rng, led,
                               explore=pc.explore_sigma, collect=True, collect_traj=True)
                for k in range(W.K):
                    tr = o["traces"][k]
                    for f in ("cmds", "raw", "s0", "err", "traj"):
                        pools[k][f].append(tr[f])
            for k in pools:
                for f in list(pools[k]):
                    pools[k][f] = np.concatenate(pools[k][f], 0) if len(pools[k][f]) else None
            return pools

        def score_set(level_map, kw, sd):
            led = Ledger(W.dt_ctrl, W.d_fb)
            sp = W.traverse(prefix_dec(level_map), rt, np.random.default_rng(sd), led)
            return sp["seam"][kw][:kc["n_score"]]

        for d in range(K):
            lm = {e: (1,) for e in range(d)}
            pools = harvest(lm, kc["n_warm"], kc["batch"], seed + 20000 + 97 * d)
            S0 = score_set(lm, k0 + d, seed + 4400 + d)
            # THE POOL'S RENDITION SPREAD — logged beside the head's training loss so the two can
            # be read together (Phase 1 measured the slot partition degenerate at 4 of 5 tempi,
            # and the head trains on the same pool). All three are in the units they live in.
            cm = pools[k0 + d]["cmds"]
            tj = pools[k0 + d]["traj"]
            pool_stats.append(dict(
                drilled=int(d), n_rend=int(len(cm)),
                cmd_sd=float(np.mean(cm.std(0))), cmd_rms=float(np.sqrt((cm ** 2).mean())),
                traj_sd=float(np.mean(tj.std(0))),
                s0_sd=[float(x) for x in pools[k0 + d]["s0"].std(0)],
                err_sd=float(pools[k0 + d]["err"].std()),
                err_mean=float(pools[k0 + d]["err"].mean())))
            slots, pois, rec = build_level1_cell(
                W, pools[k0 + d], S0, k0 + d, np.random.default_rng(seed + 77777 + 13 * d),
                llb, kc["m_cand"], kc["n_slot"], kc["m_member"], kc["n_poison"])
            for g in slots:
                lvlib.add_slot(1, d, g, np.mean([m["key"] for m in g], 0))
            if pois:
                lvlib.add_slot(1, d, pois, np.mean([m["key"] for m in pois], 0), poison=True)
            rec.update(level=1, drilled=int(d))
            b1_rec.append(rec)
        for ell in range(2, L + 1):
            sp = LadderLayout.span(ell)
            done = []
            for d in range(0, K, sp):
                if d + sp > K:
                    continue
                lm = {e: (ell,) for e in done}
                for e in range(K):
                    lm.setdefault(e, (ell - 1,))
                S0 = score_set(lm, k0 + d, seed + 6600 + 71 * ell + d)
                slots, pois, rec = build_level_cell(
                    W, lvlib, ell, d, S0,
                    np.random.default_rng(seed + 88888 + 131 * ell + 13 * d), llb,
                    kc["m_weld"], kc["n_slot"], kc["m_member"], kc["n_poison"])
                if slots:
                    for g in slots:
                        lvlib.add_slot(ell, d, g["members"], g["key"], parents=g["parents"])
                    if pois:
                        lvlib.add_slot(ell, d, pois, pois[0]["key"], poison=True)
                    done.append(d)
                bl_rec.append(rec)
        return lvlib, b1_rec, bl_rec, pool_stats

    def check_nesting(W, lvlib, pc, rt):
        K, H, L = W.K_drill, W.H, pc.n_levels
        k0 = W.k0
        pn = dict(spell_bad=0, exec_max_abs=0.0, n_checked=0, exec_tol=1e-6)
        for ell in range(2, L + 1):
            spn = LadderLayout.span(ell)
            for d in range(0, K, spn):
                for sl in lvlib.slots(ell, d, with_poison=False):
                    pa = lvlib.slot_by_id(sl["parents"][0])
                    pb = lvlib.slot_by_id(sl["parents"][1])
                    halfL = (spn // 2) * H
                    for mb in sl["members"]:
                        okA = any(np.array_equal(mb["cmds"][:halfL], q["cmds"])
                                  for q in pa["members"])
                        okB = any(np.array_equal(mb["cmds"][halfL:], q["cmds"])
                                  for q in pb["members"])
                        pn["n_checked"] += 1
                        if not (okA and okB):
                            pn["spell_bad"] += 1
                cell = lvlib.slots(ell, d, with_poison=False)
                if not cell:
                    continue
                mb = cell[0]["members"][0]
                S0 = rt[:min(8, len(rt))]
                h2 = spn // 2
                e_w, _ = W.rollout(S0, np.tile(mb["cmds"][None], (len(S0), 1, 1)), k0 + d, spn,
                                   None)
                e_a, fa = W.rollout(S0, np.tile(mb["cmds"][:h2 * H][None], (len(S0), 1, 1)),
                                    k0 + d, h2, None)
                e_b, _ = W.rollout(fa, np.tile(mb["cmds"][h2 * H:][None], (len(S0), 1, 1)),
                                   k0 + d + h2, h2, None)
                pn["exec_max_abs"] = max(pn["exec_max_abs"], float(
                    np.max(np.abs(np.concatenate([e_a, e_b], 1) - e_w))))
        pn["pass"] = bool(pn["spell_bad"] == 0 and pn["exec_max_abs"] <= pn["exec_tol"]
                          and pn["n_checked"] > 0)
        return pn

    # ================================================= the pieces, the pinned gains, gate P-G
    R_star = float(cfg["r_star"])
    tempos = list(cfg["tempos"])
    L = int(cfg["n_levels"])
    Ws, pcs, evs, rts, libs, pools_by = {}, {}, {}, {}, {}, {}
    # THE GAIN TABLE. Pinned from `t1` for the dyadic tempi it measured; RE-FIT from the full
    # 63-cell grid for any tempo `t1` never ran (the fine-ladder round's 7 / 6 / 5). The rule is
    # the same either way — the incumbent's own argmin on the clean world at that (tempo, Delta) —
    # and gate P-G checks the pinned half against a live re-fit.
    gains, refit_rows = {}, []
    for H in tempos:
        if int(H) in T1.T1_GAINS and not cfg.get("refit_gains"):
            gains[int(H)] = {int(D): tuple(v) for D, v in T1.T1_GAINS[int(H)].items()}
            continue
        pcH = P.accel_piece(float(cfg["r_star"]), H, k_app=cfg["k_app"],
                            n_levels=int(cfg["n_levels"]), damping=cfg["damping"],
                            mass=cfg["mass"])
        Wf = World(cfg, pcH, n_proc=1, rot=False)
        evf = Wf.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        g = {}
        need_D = {0, int(cfg["delta_fix"])}
        if not cfg.get("no_instr"):
            need_D |= set(int(x) for x in cfg["deltas"])
        for D in sorted(need_D):
            grid = []
            for kp in cfg["kp_grid"]:
                for kd in cfg["kd_grid"]:
                    Wf.kp, Wf.kd = float(kp), float(kd)
                    led = Ledger(Wf.dt_ctrl, Wf.d_fb)
                    oc = Wf.traverse(ReflexDecider(), evf, np.random.default_rng(1), led,
                                     obs_delay=D)
                    grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
            bb = min(grid, key=lambda r: r["e_piece"])
            g[int(D)] = (bb["kp"], bb["kd"])
            refit_rows.append(dict(H=int(H), delta=int(D), kp=bb["kp"], kd=bb["kd"],
                                   e_piece=bb["e_piece"],
                                   at_edge=[bool(bb["kp"] in (min(cfg["kp_grid"]),
                                                              max(cfg["kp_grid"]))),
                                            bool(bb["kd"] in (min(cfg["kd_grid"]),
                                                              max(cfg["kd_grid"])))]))
        gains[int(H)] = g
        Wf.close()
    if refit_rows:
        out["gains_refit"] = refit_rows
        PR("[gains] re-fit for tempi t1 never ran: " + "  ".join(
            f"H{r['H']}D{r['delta']}:({r['kp']:g},{r['kd']:g})e{r['e_piece']:.4f}"
            for r in refit_rows))
    t1_ok = all(float(cfg.get(k, -1)) == float(v) for k, v in T1.T1_CFG.items())
    pt1 = dict(rows=[], max_abs=0.0, cfg_match=bool(t1_ok),
               cfg_diff={k: [cfg.get(k), v] for k, v in T1.T1_CFG.items()
                         if float(cfg.get(k, -1)) != float(v)})

    # P-G: the pinned table is not taken on trust. One cell is re-fit from the full 63-cell grid.
    Hg, Dg = T1.T1_GAIN_CHECK
    t0 = time.time()
    _pg_ok = bool(Hg in cfg["tempos"] and not cfg.get("refit_gains"))
    pcg = P.accel_piece(R_star, Hg, k_app=cfg["k_app"], n_levels=L, damping=cfg["damping"],
                        mass=cfg["mass"])
    Wg = World(cfg, pcg, n_proc=1, rot=False)
    evg = Wg.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
    grid = []
    for kp in cfg["kp_grid"]:
        for kd in cfg["kd_grid"]:
            Wg.kp, Wg.kd = float(kp), float(kd)
            led = Ledger(Wg.dt_ctrl, Wg.d_fb)
            oc = Wg.traverse(ReflexDecider(), evg, np.random.default_rng(1), led, obs_delay=Dg)
            grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
    b = min(grid, key=lambda r: r["e_piece"])
    Wg.close()
    pg = dict(cell=[int(Hg), int(Dg)], refit=[b["kp"], b["kd"]], applicable=_pg_ok,
              pinned=([float(x) for x in gains[Hg][Dg]] if Hg in gains else None),
              e_piece=b["e_piece"],
              **{"pass": bool(_pg_ok and Hg in gains
                              and [b["kp"], b["kd"]] == [float(x) for x in gains[Hg][Dg]])})
    out["P_G"] = pg
    out["t_pg"] = time.time() - t0
    PR(f"[P-G] gain table check at (H={Hg}, Delta={Dg}): re-fit {pg['refit']} vs pinned "
       f"{pg['pinned']}  e={b['e_piece']:.6f}  pass={pg['pass']}  [{out['t_pg']:.0f}s]")
    if _pg_ok and not pg["pass"]:
        save()
        raise SystemExit("P-G failed: the pinned gain table is not what this fork re-fits")

    # ================================================= the libraries, and gate P-T1's build half
    t0 = time.time()
    pn_by, pool_by = {}, {}
    for H in tempos:
        pc = P.accel_piece(R_star, H, k_app=cfg["k_app"], n_levels=L, damping=cfg["damping"],
                           mass=cfg["mass"])
        W = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
        ev = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        rt = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])
        Ws[H], pcs[H], evs[H], rts[H] = W, pc, ev, rt
        out.setdefault("piece_by_tempo", {})[str(H)] = pc.describe()
        llb = Ledger(W.dt_ctrl, W.d_fb)
        lvlib, b1r, blr, pstat = build_ladder(W, pc, cfg, cfg["seed"], gains[H][0], llb, rt)
        libs[H] = lvlib
        pool_by[str(H)] = pstat
        out.setdefault("library", {})[str(H)] = dict(
            level1=b1r, levels=blr, counts=lvlib.counts(), build_ledger=llb.snap(),
            harvest_gains=list(gains[H][0]))
        if t1_ok and H in T1.T1_L1:
            for i, r in enumerate(b1r):
                pt1["rows"].append(dict(what=f"L1@{H}[{i}]",
                                        got=r["chosen_score"], want=T1.T1_L1[H][i]))
            got = {(r["level"], r["seam"]): r.get("chosen_score") for r in blr}
            for e, k, v in T1.T1_LN[H]:
                pt1["rows"].append(dict(what=f"L{e}@{H}[{k}]", got=got[(e, k)], want=v))
        pnn = check_nesting(W, lvlib, pc, rt)
        pn_by[str(H)] = pnn
        PR(f"[lib H={H:2d}] {lvlib.counts()}  {llb.ground:.0f} groundings/perf  "
           f"P-N {pnn['n_checked']} spelled / {pnn['spell_bad']} bad / "
           f"{pnn['exec_max_abs']:.3e}   pool: cmd_sd "
           f"{np.mean([q['cmd_sd'] for q in pstat]):.4f} (rms "
           f"{np.mean([q['cmd_rms'] for q in pstat]):.4f})  traj_sd "
           f"{np.mean([q['traj_sd'] for q in pstat]):.5f}  err_sd/mean "
           f"{np.mean([q['err_sd'] for q in pstat]):.4f}/"
           f"{np.mean([q['err_mean'] for q in pstat]):.4f}")
        save()
    out["P_N"] = dict(by_tempo=pn_by, **{"pass": all(v["pass"] for v in pn_by.values())})
    out["pool"] = pool_by
    out["t_library"] = time.time() - t0

    # =================================================== the scaled libraries, and gate P-R
    H_src = tempos[0]
    scaled = {}
    for p_exp in cfg["scale_exponents"]:
        for H in tempos:
            scaled[(p_exp, H)] = resample_library(libs[H_src], H_src, H, p=p_exp)
    s_id = scaled[(0.0, H_src)]
    mx, ncm = 0.0, 0
    for (ell, k) in sorted(libs[H_src].layout.off):
        for a, bq in zip(libs[H_src].slots(ell, k), s_id.slots(ell, k)):
            for ma, mb in zip(a["members"], bq["members"]):
                mx = max(mx, float(np.max(np.abs(ma["cmds"] - mb["cmds"]))))
                ncm += 1
    out["P_R"] = dict(src_tempo=int(H_src), max_abs=mx, n_members=ncm,
                      **{"pass": bool(mx == 0.0)})
    PR(f"[P-R] resampler identity at s = 1: max|delta| = {mx:.3e} over {ncm} members  "
       f"pass={mx == 0.0}")

    # ================================================ the self-imitation capture, at EVERY tempo
    # States at every tempo (the parity gate needs held-out launch states wherever it is asked);
    # TARGETS are only ever trained on at the PRACTICED tempi (`train_prog(tempi=...)`).
    t0 = time.time()
    buf = ProgBuffer(hold_frac=float(cfg["hold_frac"]))
    cap_led = Ledger(P.DT_CTRL, P.D_FB)
    cap_rows = []
    for H in tempos:
        W, pc, rt = Ws[H], pcs[H], rts[H]
        K, k0 = W.K_drill, W.k0
        # THE CAPTURE CONDITION (decision 33). `harvest` is `p1`'s: the launch distribution the
        # LIBRARY was built in — a level-1-keyed traversal at Delta = 0 under the harvest gains.
        # `deploy` is the fix round's: the distribution the units are actually DEPLOYED from,
        # Delta = delta_fix under `app=fixed`, i.e. the sweep's own condition. `p1` measured why
        # this can matter more than anything about tempo: `progall` matched the tape on held-out
        # CAPTURE states at H = 2 (head 0.0082 vs tape 0.0081) and read 0.7823 in the SWEEP. A
        # tape is invariant to the condition it is replayed under; a posture-conditioned head is
        # not, so capturing where the library was built and deploying where the arm plays asks
        # the head to extrapolate in posture on top of everything else.
        Dcap = int(cfg["delta_fix"]) if str(cfg.get("capture_at")) == "deploy" else 0
        W.kp, W.kd = gains[H][Dcap]
        W.kp_app, W.kd_app = gains[H][0]
        led = Ledger(W.dt_ctrl, W.d_fb)
        # the launch distribution the units are actually deployed from: a level-1-keyed traversal
        # at Delta = 0 with the harvest gains, i.e. exactly the configuration the library was
        # built in (prestissimo decision 13's "walk the piece in order", carried).
        okey = W.traverse(LadderDecider(W, libs[H], mode="key", levels=(1,), obs_delay=Dcap),
                          W.start_states(np.random.default_rng(cfg["seed"] + 71000 + H),
                                         cfg["n_capture"]),
                          np.random.default_rng(cfg["seed"] + 72000 + H), led,
                          obs_delay=Dcap, obs_delay_app=0, collect_obs=True)
        for ell in range(1, L + 1):
            spn = LadderLayout.span(ell)
            for d in range(0, K, spn):
                S = okey["seam"][k0 + d]                       # where the body actually is
                # THE HEAD'S INPUT (decision 35). `true` is p1's; `obs` is the Delta-stale read
                # the decider is really handed at deployment, so that training posture and
                # deployment posture are the same object.
                S_in = (okey["obs"][:, (k0 + d) * W.H, :]
                        if str(cfg.get("posture")) == "obs" else S)
                for sl in libs[H].slots(ell, d, with_poison=False):
                    mm = sl["members"]
                    n, m = len(S), len(mm)
                    s0 = np.repeat(S, m, axis=0)
                    cm = np.tile(np.stack([b2["cmds"] for b2 in mm]), (n, 1, 1))
                    errs, _ = W.rollout(s0, cm, k0 + d, spn, cap_led)
                    e = errs.mean(1).reshape(n, m)
                    win = e.argmin(1)
                    C = np.stack([mm[int(j)]["cmds"] for j in win])
                    buf.store(int(sl["slot"]), H, spn, S_in, S, C)
                    cap_rows.append(dict(slot=int(sl["slot"]), H=int(H), level=int(ell),
                                         drilled=int(d), n=int(n),
                                         mean_win_err=float(e.min(1).mean()),
                                         n_distinct_member=int(len(np.unique(win)))))
    tr_sizes, ho_sizes = buf.sizes()
    # P-H: the split is disjoint by construction; asserted rather than assumed.
    ph = dict(n_cells=len(tr_sizes), overlap=0, n_train=int(sum(tr_sizes.values())),
              n_hold=int(sum(ho_sizes.values())))
    for key in set(buf.train) & set(buf.hold):
        a = {tuple(np.round(x, 6)) for x in buf.train[key][1]}
        b2 = {tuple(np.round(x, 6)) for x in buf.hold[key][1]}
        ph["overlap"] += len(a & b2)
    ph["pass"] = bool(ph["overlap"] == 0 and ph["n_hold"] > 0)
    out["P_H"] = ph
    out["capture"] = dict(rows=cap_rows, groundings=cap_led.ground,
                          train_cells=len(tr_sizes), hold_cells=len(ho_sizes))
    out["t_capture"] = time.time() - t0
    PR(f"[capture] {ph['n_train']} train / {ph['n_hold']} held-out launch states over "
       f"{ph['n_cells']} (slot, tempo) cells; {cap_led.ground:.0f} auditions.  "
       f"[P-H] overlap {ph['overlap']}  pass={ph['pass']}  [{out['t_capture']:.0f}s]")
    save()

    # ============================================================== the heads, and their training
    t0 = time.time()
    allS = np.concatenate([buf.train[k][0] for k in buf.train
                           if int(k[1]) in set(cfg["practiced"])], 0)   # the head's INPUTS
    NORM = (allS.mean(0).astype(np.float64), allS.std(0).astype(np.float64) + 1e-6)
    n_slots = libs[tempos[0]].layout.n_slots
    heads, prog_fns, train_rec = {}, {}, {}
    # THREE heads. `prog` and `prog2` differ only in how far they must extrapolate; `progall`
    # practices EVERY tempo and is not a transfer arm at all — it is the head's own ceiling, the
    # control that separates "the head cannot extrapolate to this tempo" from "the head cannot
    # represent this tempo at all". Added after `psmoke4` measured the held-out command mse at
    # 0.0007 inside the practiced set and 0.346 one rung outside it: the failure was already
    # located at the tempo axis, and `progall` is what turns that from an inference into a
    # measurement. It is an ORACLE, exactly like `rec` at an unpracticed tempo, and is labelled
    # as one everywhere.
    if str(cfg.get("head_mode", "fixed")) == "incremental":
        # THE METRONOME LADDER (decision 34). One head per rung: `inc_i` practices `tempos[:i+1]`
        # and is ASKED at `tempos[i+1]` — the pianist's own protocol, in which the next increment
        # is small and the head is refit on its own executed traversals at every step. `p1` asked
        # for a whole octave in one jump and the gate never opened; this asks how far ahead of the
        # practiced tempo it reaches when the step is a quarter-octave.
        head_specs = [(f"inc{i}", cfg["tempos"][:i + 1], 31350 + 7 * i)
                      for i in range(len(cfg["tempos"]) - 1)]
    else:
        head_specs = [("prog", cfg["practiced"], 31339),
                      ("prog2", cfg["practiced_short"], 31341),
                      ("progall", cfg["tempos"], 31343)]
    out["head_specs"] = [[a, list(bq)] for a, bq, _ in head_specs]
    PRIMARY = head_specs[0][0]      # the head the `prog` / `proga` / `proggate` arms play
    for nm, prac, sd in head_specs:
        h = build_prog(n_slots, cfg["seed"] + sd, device, hidden=cfg["head_hidden"],
                       layers=cfg["head_layers"], emb=cfg["head_emb"], n_freq=cfg["head_freq"])
        PR(f"    [{nm}] practiced tempi {list(prac)}")
        rec = train_prog(h, buf, NORM, device, cfg["seed"] + sd + 7, tempi=prac,
                         epochs=cfg["head_epochs"], batch=cfg["head_batch"],
                         lr=cfg["head_lr"], n_freq=cfg["head_freq"], log=PR)
        heads[nm] = h
        prog_fns[nm] = make_prog_fn(h, device, NORM, n_freq=cfg["head_freq"])
        train_rec[nm] = dict(practiced=list(prac), **rec)
    # the HELD-OUT command mse, so "fits its targets" and "generalises to a new launch state"
    # are two numbers rather than one. `psmoke3` needed this: train mse 0.0013 with the head
    # still 15x worse than the tape on held-out states at a PRACTICED tempo.
    from mjc.practice.tempo.accelerando.nets import phase_feats
    for nm in train_rec:
        ho = {}
        for (sid, H), (S_in, S, C, sp) in sorted(buf.hold.items()):
            pf = prog_fns[nm](S_in, int(sid), int(H), int(sp))
            ho.setdefault(int(H), []).append(float(((pf - C) ** 2).mean()))
        train_rec[nm]["hold_mse_by_tempo"] = {k: float(np.mean(v)) for k, v in ho.items()}
    out["training"] = train_rec
    out["device"] = str(device)
    out["norm"] = [[float(x) for x in NORM[0]], [float(x) for x in NORM[1]]]
    out["t_train"] = time.time() - t0
    for nm in train_rec:
        r = train_rec[nm]
        PR(f"[train/{nm}] {r['n_rows']} rows on tempi {r['practiced']}: final mse "
           f"{r['final']:.6f}  vs baselines zero {r['baseline']['zero']:.6f} / slot-mean "
           f"{r['baseline']['slot_mean']:.6f}   by tempo "
           + json.dumps({k: round(v, 6) for k, v in r["by_tempo"].items()}))
    for nm in train_rec:
        PR(f"[train/{nm}] HELD-OUT command mse by tempo: "
           + json.dumps({k: round(v, 6) for k, v in
                         sorted(train_rec[nm]["hold_mse_by_tempo"].items())}))
    PR(f"[train] device={device}  [{out['t_train']:.0f}s]")
    save()

    # ============================== P-E: how exact does an open-loop unit have to BE on this body?
    # `psmoke3` measured the head fitting its training targets to mse 0.0013 — BELOW the
    # exploration noise's own floor of sigma^2 = 0.0040, i.e. memorising — and still executing
    # 15x worse than the tape at a tempo it had practiced. That is not a capacity result and not
    # a transfer result: it says an open-loop unit on a body with tau = 30 ms is exquisitely
    # sensitive to command error, because the span is many tau long and nothing re-grounds.
    # P-E measures the exchange rate directly: take the library's own best member, perturb its
    # commands by epsilon, execute, and read the piece error. It converts the head's command
    # RMSE into the execution error the body will charge for it, per tempo and per level, and it
    # is the instrument that separates "the head is bad" from "no approximation could work here".
    t0 = time.time()
    pe = []
    rngE = np.random.default_rng(cfg["seed"] + 99001)
    for H in tempos:
        W, pc = Ws[H], pcs[H]
        K, k0 = W.K_drill, W.k0
        for ell in range(1, L + 1):
            spn = LadderLayout.span(ell)
            for d in range(0, K, spn):
                S = None
                for (sid, Hh), (_si, Sh, _c, _sp) in sorted(buf.hold.items()):
                    if int(Hh) == int(H) and libs[H].layout.cell_of(int(sid))[:2] == (ell, d):
                        S = Sh[:min(8, len(Sh))]
                        break
                cell = libs[H].slots(ell, d, with_poison=False)
                if S is None or not cell or len(S) == 0:
                    continue
                base = np.asarray(cell[0]["members"][0]["cmds"], np.float32)
                for eps in cfg["eps_ladder"]:
                    cm = np.tile(base[None], (len(S), 1, 1))
                    if eps > 0:
                        cm = np.clip(cm + float(eps) * rngE.standard_normal(cm.shape),
                                     -1, 1).astype(np.float32)
                    e, _ = W.rollout(S, cm, k0 + d, spn, None)
                    pe.append(dict(H=int(H), level=int(ell), drilled=int(d), eps=float(eps),
                                   n=int(len(S)), e_mean=float(e.mean()),
                                   e_med=float(np.median(e.mean(1)))))
                break                      # one cell per (tempo, level) is enough for the curve
    out["P_E"] = pe
    out["t_pe"] = time.time() - t0
    PR("[P-E] execution error of the library's own tape under command noise eps "
       f"(band {pcs[tempos[0]].band():.4f}):")
    for H in tempos:
        for ell in (1, L):
            rr = [r for r in pe if r["H"] == H and r["level"] == ell]
            if rr:
                PR(f"      H={H:2d} L{ell}  " + "  ".join(
                    f"e{r['eps']:.2f}={r['e_mean']:.4f}" for r in rr))
    save()

    # ================================================================ PARITY, per slot per tempo
    t0 = time.time()
    par_rows, opens = [], {}
    # the reference side is the SAME for every head, so it is auditioned once per (tempo, library)
    refs = {}
    for H in tempos:
        for ref_name, ref_lib in (("rec", libs[H]), ("scl0", scaled[(0.0, H)])):
            refs[(H, ref_name)] = ref_errors(Ws[H], buf, ref_lib, H, cfg["parity_min"], np)
    for nm in heads:
        for H in tempos:
            W, pc = Ws[H], pcs[H]
            W.kp, W.kd = gains[H][0]
            for ref_name, ref_lib in (("rec", libs[H]), ("scl0", scaled[(0.0, H)])):
                led = Ledger(W.dt_ctrl, W.d_fb)
                op, rows = parity(W, buf, prog_fns[nm], ref_lib, refs[(H, ref_name)], H,
                                  cfg["parity_tau"], cfg["parity_min"], pc.band(), np, led,
                                  ref_name=ref_name)
                for r in rows:
                    r["head"] = nm
                par_rows += rows
                opens[(nm, H, ref_name)] = op
            PR(f"[parity/{nm}] H={H:2d}  vs rec: {len(opens[(nm, H, 'rec')])} open / "
               f"{len([r for r in par_rows if r['head'] == nm and r['H'] == H and r['ref'] == 'rec'])} checked"
               f"   vs scl0: {len(opens[(nm, H, 'scl0')])} open / "
               f"{len([r for r in par_rows if r['head'] == nm and r['H'] == H and r['ref'] == 'scl0'])} checked")
    out["parity"] = dict(rows=par_rows, tau=cfg["parity_tau"],
                         opens={f"{a}|{b}|{c}": sorted(v) for (a, b, c), v in opens.items()})
    out["t_parity"] = time.time() - t0
    save()

    # ============================================================== THE SWEEP at the fixed Delta
    t0 = time.time()
    frac_ths = dict(q=0.25, t=1.0 / 3.0, h=0.5)

    def run_arm(H, D, nm, kind, mode, ell, appm="fixed"):
        W, pc, ev = Ws[H], pcs[H], evs[H]
        band, ml = pc.band(), pc.mean_leg()
        W.kp_app, W.kd_app = (gains[H][0] if APP_D0_GAINS[appm] else (None, None))
        W.kp, W.kd = gains[H][D]
        led = Ledger(W.dt_ctrl, W.d_fb)
        if kind == "reflex":
            dec = ReflexDecider()
        elif kind == "reflex_ol":
            dec = FrozenReflexDecider(W)
        elif kind in ("rec", "scl0", "scl2"):
            lb = libs[H] if kind == "rec" else scaled[(0.0 if kind == "scl0" else 2.0, H)]
            dec = LadderDecider(W, lb, mode=mode, levels=(ell,), obs_delay=D)
        else:                                  # a prog arm: kind is the head's name (+ "_gate")
            hn = kind.replace("_gate", "")
            op = (sorted(opens[(hn, H, "scl0")]) if kind.endswith("_gate") else None)
            dec = ProgDecider(W, libs[H], prog_fns[hn], H, mode=mode, levels=(ell,),
                              obs_delay=D, open_slots=op,
                              fallback_lib=scaled[(0.0, H)])
        o = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                       obs_delay=D, obs_delay_app=APP_MODES[appm])
        ep = np.asarray(o["ep"], np.float64)
        el = np.asarray(o["ep_listen"], np.float64)
        row = dict(H=int(H), delta=int(D), arm=nm, kind=kind, mode=mode, app=appm,
                   level=(None if ell is None else int(ell)),
                   e_piece=o["e_piece"], e_listen=o["e_listen"], e_app=o["e_app"],
                   in_band=float(np.mean(ep <= band)), lis_band=float(np.mean(el <= band)),
                   **{f"in_{k2}": float(np.mean(ep <= v * ml)) for k2, v in frac_ths.items()},
                   **{f"lis_{k2}": float(np.mean(el <= v * ml)) for k2, v in frac_ths.items()},
                   ledger=led.snap(), ledger_drilled=o["ledger_drilled"])
        if isinstance(dec, ProgDecider):
            row["n_head"] = int(dec.n_head)
            row["n_fallback"] = int(dec.n_fallback)
        return row

    def arm_table(full=True):
        A = [("reflex", "reflex", None, None), ("reflex_ol", "reflex_ol", None, None)]
        levels = range(1, L + 1) if full else (1, 2)
        for e in levels:
            A.append((f"key_L{e}", "rec", "key", e))
            A.append((f"aud_L{e}", "rec", "audit", e))
            A.append((f"aud_s0_L{e}", "scl0", "audit", e))
            if full:
                A.append((f"aud_s2_L{e}", "scl2", "audit", e))
            A.append((f"prog_L{e}", PRIMARY, "key", e))
            if full:
                A.append((f"proga_L{e}", PRIMARY, "audit", e))
                A.append((f"proggate_L{e}", PRIMARY + "_gate", "key", e))
                for hn in sorted(heads):
                    if hn != PRIMARY:
                        A.append((f"{hn}_L{e}", hn, "key", e))
        return A

    sweep = []
    D0 = int(cfg["delta_fix"])
    for appm in cfg["app_modes"]:
        for H in tempos:
            rows = [run_arm(H, D0, nm, kind, mode, ell, appm)
                    for nm, kind, mode, ell in arm_table(True)]
            sweep += rows
            g = {r["arm"]: r["e_piece"] for r in rows}
            PR(f"[sweep/{appm}] H={H:2d}  rf={g['reflex']:.4f}  " + "  ".join(
                f"{a}={g[a]:.4f}" for a in
                (f"key_L{L}", f"aud_L{L}", f"aud_s0_L{L}", f"prog_L{L}", f"proggate_L{L}",
                 f"prog2_L{L}", "prog_L1", "aud_L1", "aud_s0_L1") if a in g)
               + f"   band={pcs[H].band():.4f}")
            save()
    out["sweep"] = sweep

    # gate P-T1's sweep half: `t1`'s pinned rows must come back at max|delta| = 0
    if t1_ok and "fixed" in cfg["app_modes"]:
        for H in tempos:
            if H not in T1.T1_ROWS:
                continue
            for nm in ("reflex", "key_L1", "key_L4", "aud_L1", "aud_L4"):
                r = [x for x in sweep if x["H"] == H and x["app"] == "fixed"
                     and x["arm"] == nm and x["delta"] == D0]
                if not r or nm not in T1.T1_ROWS[H]:
                    continue
                pt1["rows"].append(dict(what=f"{nm}@{H}", got=r[0]["e_piece"],
                                        want=T1.T1_ROWS[H][nm]))
                pt1["rows"].append(dict(what=f"{nm}@{H}:listen", got=r[0]["e_listen"],
                                        want=T1.T1_ROWS[H][nm + "_listen"]))
    pt1["max_abs"] = max([abs(r["got"] - r["want"]) for r in pt1["rows"]], default=-1.0)
    pt1["n_checks"] = len(pt1["rows"])
    pt1["applicable"] = bool(t1_ok)
    pt1["pass"] = bool(pt1["applicable"] and pt1["max_abs"] == 0.0 and pt1["n_checks"] > 0)
    pt1["worst"] = sorted(pt1["rows"], key=lambda r: -abs(r["got"] - r["want"]))[:5]
    out["P_T1"] = pt1
    PR(f"[P-T1] Phase 2 reproduces t1: {pt1['n_checks']} checks, max|delta| = "
       f"{pt1['max_abs']:.3e}  applicable={pt1['applicable']}  pass={pt1['pass']}"
       + ("" if t1_ok else f"  (config differs from t1: {sorted(pt1['cfg_diff'])})"))
    out["t_sweep"] = time.time() - t0
    save()
    if pt1["applicable"] and not pt1["pass"] and not cfg.get("refit_gains"):
        raise SystemExit("P-T1 failed: Phase 2's substrate is not t1's")

    # ============================== the Delta instrument, read at L1 and L2 (decision: see below)
    # Under `app=fixed` a level-4 arm makes ONE decision from a lead-in that does not move with
    # Delta, so it is Delta-flat BY CONSTRUCTION — `t1` measured key_L4 spreads of 1.00x-1.09x.
    # L1 re-decides at all eight drilled seams and L2 at four, so those are the rungs where the
    # Delta axis reads anything at all about feedback.
    t0 = time.time()
    instr = []
    for H in (tempos if not cfg.get("no_instr") else []):
        for D in cfg["deltas"]:
            instr += [run_arm(H, D, nm, kind, mode, ell, "fixed")
                      for nm, kind, mode, ell in arm_table(False)]
        rr = [r for r in instr if r["H"] == H]

        def gg(D, nm):
            v = [x for x in rr if x["delta"] == D and x["arm"] == nm]
            return v[0]["e_piece"] if v else float("nan")
        PR(f"[instr] H={H:2d}  " + "  ".join(
            f"D{D}: rf={gg(D, 'reflex'):.4f} aL1={gg(D, 'aud_L1'):.4f} "
            f"pL1={gg(D, 'prog_L1'):.4f} pL2={gg(D, 'prog_L2'):.4f}"
            for D in cfg["deltas"]))
        save()
    out["instrument"] = instr
    out["t_instr"] = time.time() - t0

    out["flags"] = dict(
        delta_at_L1_L2=("Read the Delta sweep at L1 and L2 only. Under `app=fixed` the deep arms "
                        "decide once from a lead-in that does not move with Delta and are "
                        "Delta-flat by construction (t1: key_L4 spread 1.00x-1.09x); L1 and L2 "
                        "re-decide 8 and 4 times per traversal and are not."),
        pool_diversity=("Phase 1 measured the slot partition degenerate at 4 of 5 tempi (P-S "
                        "1.000x, 1 of 6 distinct argmins) and the head trains on that same pool "
                        "at sigma = 0.0628. sigma is NOT changed here — that would break "
                        "comparability with t1 — and the pool's rendition spread (`cmd_sd`, "
                        "`traj_sd`, `err_sd`, the launch-state sd) is logged per tempo beside "
                        "the head's per-tempo training loss so the two can be read together."),
        parity_reference=("At a practiced tempo the reference is the same-tempo recording — "
                          "solo's gate verbatim, and solo finding 6's question. At an "
                          "UNPRACTICED tempo no same-tempo recording exists for a learner, so "
                          "`scl0` is the deployable reference and `rec` is the oracle one; both "
                          "are run at every tempo and reported side by side. `proggate` is the "
                          "arm that respects the deployable gate and falls back to the scl0 "
                          "tape below parity, so the gate can never introduce drift; `prog` is "
                          "ungated and measures TRANSFER rather than adoption."),
        oracle_rows=("`rec`, `key_L*` and `aud_L*` at an unpracticed tempo are ORACLE rows: they "
                     "are what practice at that tempo would have bought, and no prog arm has "
                     "access to them. They are reported as the ceiling, never as a baseline the "
                     "head is entitled to."))
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    for H in tempos:
        Ws[H].close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


@app.local_entrypoint()
def accelerando_program(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    torch_threads: int = 12,
    r_star: float = T1.T1_R_STAR,
    k_app: int = P.A_K_APP,
    n_levels: int = P.N_LEVELS,
    mass: float = P.FAST_MASS,
    damping: float = P.DAMPING,
    delta_fix: int = P.DELTA_FIX,
    tempos: str = ",".join(str(x) for x in T1.T1_TEMPOS),
    practiced: str = ",".join(str(x) for x in PRACTICED),
    practiced_short: str = ",".join(str(x) for x in PRACTICED_SHORT),
    n_eval: int = 32,
    n_rt: int = 48,
    n_capture: int = 64,
    batch: int = 24,
    n_warm: int = 8,
    n_score: int = 16,
    n_slot: int = 6,
    m_cand: int = 48,
    m_member: int = 4,
    m_weld: int = 6,
    n_poison: int = 3,
    hold_frac: float = 0.25,
    parity_tau: float = 0.75,
    parity_min: int = 4,
    head_hidden: int = 512,
    head_layers: int = 4,
    head_emb: int = 32,
    head_freq: int = 8,
    head_epochs: int = 800,
    head_batch: int = 8192,
    head_lr: float = 2e-3,
    app_modes: str = "fixed",
    capture_at: str = "harvest",
    posture: str = "true",
    head_mode: str = "fixed",
    refit_gains: bool = False,
    no_instr: bool = False,
    fixsmoke: bool = False,
):
    kp, kd = P.gain_grid(T_KP_GRID, T_KD_GRID, mass)
    cfg = dict(tag=tag or "p1", seed=seed, n_proc=n_proc, quick=bool(quick),
               torch_threads=int(torch_threads),
               r_star=float(r_star), k_app=k_app, n_levels=n_levels,
               mass=float(mass), damping=float(damping), delta_fix=int(delta_fix),
               tempos=[int(x) for x in tempos.split(",") if x],
               practiced=[int(x) for x in practiced.split(",") if x],
               practiced_short=[int(x) for x in practiced_short.split(",") if x],
               deltas=[int(x) for x in DELTAS_INSTR],
               scale_exponents=[0.0, 2.0], w_listen=P.W_LISTEN,
               eps_ladder=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
               app_modes=[x for x in app_modes.split(",") if x],
               kp_grid=list(kp), kd_grid=list(kd),
               n_eval=n_eval, n_rt=n_rt, n_capture=n_capture, batch=batch, n_warm=n_warm,
               n_score=n_score, n_slot=n_slot, m_cand=m_cand, m_member=m_member,
               m_weld=m_weld, n_poison=n_poison, hold_frac=hold_frac,
               parity_tau=parity_tau, parity_min=parity_min,
               head_hidden=head_hidden, head_layers=head_layers, head_emb=head_emb,
               head_freq=head_freq, head_epochs=head_epochs, head_batch=head_batch,
               head_lr=head_lr, capture_at=str(capture_at), posture=str(posture),
               head_mode=str(head_mode),
               refit_gains=bool(refit_gains), no_instr=bool(no_instr),
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5)
    if quick:
        cfg.update(tag=tag or "psmoke", n_proc=8, torch_threads=4, n_eval=8, n_rt=8,
                   n_capture=12, batch=6, n_warm=2, n_score=4, n_slot=3, m_cand=8, m_member=2,
                   m_weld=2, n_poison=2, tempos=[16, 8, 2], practiced=[16, 8],
                   practiced_short=[16], deltas=[0, 5], head_epochs=400,
                   head_batch=4096, head_hidden=256, head_layers=3, parity_min=2)
    if fixsmoke:
        # THE FIX ROUND'S SMOKE PRESET: `t1`'s own knobs, three tempi so H = 2 is in it, no Delta
        # instrument, enough epochs for the head to fit. Big enough to read, small enough to show
        # before anything is launched at full scale.
        cfg.update(tag=tag or "fsmoke", tempos=[16, 8, 2], practiced=[16, 8],
                   practiced_short=[16], head_epochs=400, no_instr=True,
                   deltas=[int(delta_fix)])
    if spawn:
        c = run_program.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_accelerando/{cfg['tag']}/program.json")
        return
    r = run_program.remote(cfg)
    print("\n".join(r["log"][-120:]))
