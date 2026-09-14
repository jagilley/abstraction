"""prestissimo Phase A — the fast piece and its static level ladder, model-free.

THE QUESTION. `solo/` gave the motor row address, trust, routing, corridor and adoption at the
SEGMENT rung, model-free, and `acappella/` measured the niche that rung pays in (stored segment
tapes beat a per-Delta re-fit reflex from Delta = 8 = 192 ms). Neither has a LADDER: nothing on
this substrate has ever asked whether a unit made of two committed units buys anything, because
etude's square starts from rest — its chain arms decide at seam 0 with the TRUE state at every
Delta and are exactly delay-invariant (`acappella` finding 5), so no chain number there is
evidence about depth. Phase A builds the pre-declared escalation — a fast piece whose first seam
is not its start — builds every rung of a dyadic ladder statically, and sweeps Delta:

    at which Delta does a level-l unit stop being playable, and does level l+1 stay playable?

That reading is the ERA CALIBRATION Phase B's crank is paced against. It is also the first thing
this piece can say, including the possibility that the deep rungs are not readable at all.

STANDING CONSTRAINT, verbatim from `acappella/` and `solo/`: **no forward model anywhere in this
node, not even as an extra control or reference arm.** Nothing here imports, trains or evaluates
an `f(s,u)`. Phase A has no learned component at all: the reflex law, executed content, and the
plant as the only oracle.

THE GATES (all pre-fixed, all reported whether or not they pass)
  P-F0  the 13 donor constants, read out of `etude/etude.py` with `ast`, never imported.
  P-F1  the FORK runs the DONOR SQUARE and reproduces `acappella/b1`'s library build and its
        Delta = 0 and Delta = 8 rows at max|delta| = 0. The piece is the only variable.
  P-T   the tempo calibration: R x Delta on the reflex alone, design point read off the sweep
        under rules fixed before the grid is read (legato F2).
  P-N   the nesting op is exact: a level-l member's commands ARE the concatenation of two
        committed level-(l-1) members', and executing the weld equals executing the halves in
        sequence at max|delta| = 0 (legato F4's fusion control, made an identity).
  P-A   the approach works: the deepest committed arm is NOT delay-invariant, which is the
        artifact this whole piece exists to remove.

Run:
    cd experiments/                        # MODAL_PROFILE=chromatic
    modal run mjc/practice/tempo/prestissimo/ladder.py::prestissimo_ladder --quick --tag psmoke
    modal run --detach mjc/practice/tempo/prestissimo/ladder.py::prestissimo_ladder --spawn --tag a0
    python3 mjc/practice/tempo/prestissimo/analyze_ladder.py --tag a0 --fetch --figures
"""

import json
import os
import time

import modal  # noqa: F401  (the app is imported from mjc.shared; kept for parity with siblings)

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.tempo.prestissimo import piece as P

# --- cross-tag exact controls against `acappella/b1` (offbook's bit-identity idiom) ---------- #
# Full-precision values read off `acappella/results/b1/delay.json`, carried through `solo/solo.py`
# (which reproduced all fourteen at 0.000e+00 as its gate S-F1). The control is DEFINED only under
# b1's own configuration, which this node pins inside the gate rather than in its own knobs.
A0_BUILD_SEG = [0.0575661185200656, 0.05059425708165344, 0.06815679122042403, 0.09710761575940084]
A0_BUILD_CHAIN = [0.1143509246470914, 0.0749743464935404, 0.07725866898369263]
B1_ROWS = {
    0: {"key_seg": 0.06480563431978226, "lib_seg": 0.039249300956726074,
        "lib_all": 0.03519067168235779, "key_chain": 0.11837589740753174,
        "aud_chain": 0.08519072085618973, "reflex": 0.004462865646928549},
    8: {"key_seg": 0.10340435802936554, "lib_seg": 0.08381044864654541,
        "lib_all": 0.09221908450126648, "key_chain": 0.11837589740753174,
        "aud_chain": 0.08519072085618973, "reflex": 0.16968291997909546},
}
B1_GAINS = {0: (10.0, 2.0), 8: (5.0, 4.0)}
A0_KP, A0_KD = 10.0, 2.0                        # b1's frozen HARVEST gains on the donor square
# b1's own configuration and its own gain grid. Pinned here so the gate is defined at every
# treatment config: the donor rows are a property of the DONOR piece, not of this node's knobs.
DCFG = dict(n_eval=32, n_rt=48, batch=24, n_warm=8, n_cand=24, n_score=16, n_slot=8)
D_KP_GRID = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
D_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# the TREATMENT grid: widened upward, because the fast figure's schedule moves ~1.5x faster over
# a segment 7x shorter and the donor grid's top cell would be an edge by construction. Edges are
# reported at every cell either way (`at_kp_edge` / `at_kd_edge`), as in acappella and solo.
# `a0` measured the re-fit landing on this grid's BOTTOM rung (kp = 2.0) at 8 of 9 delays, so the
# incumbent's floor was clipped rather than measured. Two rungs are added below it (0.5, 1.0) —
# two rather than one so the floor has headroom to be found instead of merely moved. Every cell
# also reports the argmin RESTRICTED to kp >= 2.0, which is `a0`'s own grid, so the two are
# comparable within one tag (decision 19).
T_KP_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
T_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A0_KP_FLOOR = 2.0                       # `a0`'s grid floor, for the restricted-argmin control
# the approach's own read delay, and whether it uses the Delta = 0 GAINS (decision 20).
#   arm   — the approach reads at the arm's Delta, at the per-Delta re-fit gains (the arm of record)
#   zero  — the approach reads at Delta = 0 but keeps the per-Delta re-fit gains. Measured NOT to
#           isolate the read: a controller fitted for a stale read consuming a fresh one
#           under-drives, so the hand-over got WORSE at every Delta >= 3 in all four diagnostic
#           cells while `rd dv` went to ~0 -- a gain effect wearing a read effect's name.
#   fixed — decision 20, the clean variant: the approach reads at Delta = 0 AND plays the
#           Delta = 0 gains, so the lead-in is ONE pre-computed plan (presto decision 3),
#           identical at every rung of the sweep, and the hand-over distribution into drilled
#           seam 0 is the same at every Delta. The sweep then isolates the drilled figure.
APP_MODES = {"arm": None, "zero": 0, "fixed": 0}
APP_D0_GAINS = {"arm": False, "zero": False, "fixed": True}
# acappella b1's Delta ladder, unchanged and NOT re-tuned (legato F2). 0-384 ms at dt = 0.024 s.
DELTAS = (0, 1, 2, 3, 4, 6, 8, 12, 16)


@app.function(cpu=16.0, memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_ladder(cfg: dict) -> dict:
    import ast

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np

    from mjc.practice.tempo.prestissimo.world import (World, Ledger, ReflexDecider, LibraryDecider,
                                                Library, select_tapes, LevelLibrary,
                                                LadderLayout, LadderDecider, build_level1_cell,
                                                build_level_cell, FrozenReflexDecider)

    class ZeroDecider:
        """The do-nothing floor: zero command everywhere. presto decision 8's competence guard
        is `practised error <= half of this`, and it is arm-neutral by construction."""

        def __init__(self, W):
            self.H = int(W.H)

        def __call__(self, k, states, need, rng, led):
            return dict(kind="open", cmds=np.zeros((len(states), self.H, 2), np.float32),
                        span=np.ones(len(states), int), src="zero")

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_prestissimo", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "phase": "A",
           "contract": dict(
               no_forward_model=("Nothing in this node imports, trains or evaluates an f(s,u). "
                                 "Phase A has no learned component at all: the reflex law, "
                                 "executed content, and the plant as the only oracle."),
               levels=("A dyadic ladder, s = 2 as in RHM. A level-l entry at drilled seam k is "
                       "the pair of committed level-(l-1) entries at k and k + 2^(l-2), "
                       "materialised as their fused tape: one feedback event at launch, the "
                       "internal seam's feedback dropped. Entries are ALIGNED (k = 0 mod "
                       "2^(l-1)) so the ladder tiles the piece and the top rung IS the piece."),
               era_knob=("Delta is the era knob, not tempo: a tape stays valid across delays but "
                         "not across tempi, so the vocabulary persists across eras the way T[l] "
                         "persists across RHM's damage eras. The library is therefore built ONCE, "
                         "at Delta = 0 with frozen harvest gains (acappella's discipline), and "
                         "swept across Delta."),
               band=("Pre-fixed and arm-neutral: max(1/2 x mean drilled leg, ref_play x mean_leg "
                     "/ 0.8) with ref_play = 0.1066 (etude's `never`, published blind). Pass "
                     "fractions at 1/4, 1/3 and 1/2 mean-leg are logged for every arm at every "
                     "Delta so the ladder can be re-read at another band without re-running."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "ladder.json"), "w") as fh:
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

    # ================================================ P-F1: the fork on the DONOR square == b1
    if cfg["do_pf1"]:
        t0 = time.time()
        dp = P.donor_piece()
        Wd = World(cfg, dp, n_proc=int(cfg["n_proc"]), rot=True)
        Wdc = World(cfg, dp, n_proc=int(cfg["n_proc"]), rot=False)
        Kd, Hd = Wd.K, Wd.H
        ev_d = Wd.start_states(np.random.default_rng(cfg["seed"] + 6000), DCFG["n_eval"])
        rt_d = Wd.start_states(np.random.default_rng(cfg["seed"] + 5000), DCFG["n_rt"])
        Wd.kp, Wd.kd = A0_KP, A0_KD
        lib = Library(Kd, DCFG["n_slot"], poison=True)
        build, chain_rec, seam_sets, lb = [], [], {}, Ledger(Wd.dt_ctrl, Wd.d_fb)

        class MixedDecider:
            def __init__(self, upto):
                self.upto = int(upto)
                self.key = LibraryDecider(Wd, lib, mode="key", levels=("seg",))
                self.reflex = ReflexDecider()

            def __call__(self, k, states, need, rng, led):
                return (self.key if k < self.upto else self.reflex)(k, states, need, rng, led)

        def dharvest(upto, n_cycles, batch, seed0):
            pools = {k: dict(cmds=[], raw=[], s0=[], err=[], traj=[]) for k in range(Kd)}
            chains = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(Kd)}
            led = Ledger(Wd.dt_ctrl, Wd.d_fb)
            for c in range(n_cycles):
                rng = np.random.default_rng(seed0 + c)
                o = Wd.traverse(MixedDecider(upto), Wd.start_states(rng, batch), rng, led,
                                explore=P.EXPLORE_SIGMA, collect=True, collect_traj=True)
                for k in range(Kd):
                    tr = o["traces"][k]
                    for f in ("cmds", "raw", "s0", "err", "traj"):
                        pools[k][f].append(tr[f])
                    if Kd - k > 1:
                        chains[k]["cmds"].append(o["acts"][:, k * Hd:, :].copy())
                        chains[k]["raw"].append(o["raw"][:, k * Hd:, :].copy())
                        chains[k]["s0"].append(o["seam"][k].copy())
                        chains[k]["err"].append(o["wp_err"][:, k:].mean(1).copy())
            for d in (pools, chains):
                for k in d:
                    for f in list(d[k]):
                        d[k][f] = np.concatenate(d[k][f], 0) if len(d[k][f]) else None
            return pools, chains

        for k in range(Kd):
            pools, _ = dharvest(k, DCFG["n_warm"], DCFG["batch"], cfg["seed"] + 20000 + 97 * k)
            led = Ledger(Wd.dt_ctrl, Wd.d_fb)
            sp = Wd.traverse(MixedDecider(k), rt_d, np.random.default_rng(cfg["seed"] + 4400), led)
            seam_sets[k] = sp["seam"][k]
            slots, rec = select_tapes(Wd, pools[k], sp["seam"][k][:DCFG["n_score"]], k, 1,
                                      np.random.default_rng(cfg["seed"] + 953 + k), lb,
                                      DCFG["n_cand"], DCFG["n_slot"])
            for s in slots:
                lib.add(1, k, s["cmds"], s["key"], s["score"])
            rec.update(seam=k, ns=1, level="seg")
            build.append(rec)
        for k in range(Kd):
            ns = Kd - k
            if ns <= 1:
                continue
            _, chains_spell = dharvest(Kd, DCFG["n_warm"], DCFG["batch"],
                                       cfg["seed"] + 40000 + 97 * k)
            slots, rec = select_tapes(Wd, chains_spell[k], seam_sets[k][:DCFG["n_score"]], k, ns,
                                      np.random.default_rng(cfg["seed"] + 1953 + k), lb,
                                      DCFG["n_cand"], DCFG["n_slot"])
            for s in slots:
                lib.add(ns, k, s["cmds"], s["key"], s["score"])
            rec.update(seam=k, ns=ns, level="chain")
            chain_rec.append(rec)
        dgains = {}
        for D in (0, 8):
            grid = []
            for kp in D_KP_GRID:
                for kd in D_KD_GRID:
                    Wdc.kp, Wdc.kd = float(kp), float(kd)
                    led = Ledger(Wd.dt_ctrl, Wd.d_fb)
                    oc = Wdc.traverse(ReflexDecider(), ev_d, np.random.default_rng(1), led,
                                      obs_delay=D)
                    grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
            dgains[D] = min(grid, key=lambda r: r["e_piece"])
        DONOR_ARMS = {
            "key_seg":   lambda: LibraryDecider(Wd, lib, mode="key", levels=("seg",)),
            "key_chain": lambda: LibraryDecider(Wd, lib, mode="key", levels=("chain",)),
            "lib_seg":   lambda: LibraryDecider(Wd, lib, mode="audit_all", budget=0,
                                                with_prim=False, levels=("seg",)),
            "lib_all":   lambda: LibraryDecider(Wd, lib, mode="audit_all", budget=0,
                                                with_prim=False, levels=("seg", "chain")),
            "aud_chain": lambda: LibraryDecider(Wd, lib, mode="audit_all", budget=0,
                                                with_prim=False, levels=("chain",)),
        }
        pf1 = {"build_seg_max_abs": max(abs(build[i]["chosen_score"] - A0_BUILD_SEG[i])
                                        for i in range(Kd)),
               "build_chain_max_abs": max(abs(chain_rec[i]["chosen_score"] - A0_BUILD_CHAIN[i])
                                          for i in range(len(chain_rec)))}
        donor_rows = {}
        for D in (0, 8):
            for nm, mk in DONOR_ARMS.items():
                led = Ledger(Wd.dt_ctrl, Wd.d_fb)
                o = Wd.traverse(mk(), ev_d, np.random.default_rng(cfg["seed"] + 35), led,
                                obs_delay=D)
                donor_rows[f"{nm}@{D}"] = dict(e_piece=o["e_piece"], **led.snap())
                pf1[f"{nm}@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D][nm])
            Wd.kp, Wd.kd = dgains[D]["kp"], dgains[D]["kd"]
            led = Ledger(Wd.dt_ctrl, Wd.d_fb)
            o = Wd.traverse(ReflexDecider(), ev_d, np.random.default_rng(cfg["seed"] + 35), led,
                            obs_delay=D)
            donor_rows[f"reflex@{D}"] = dict(e_piece=o["e_piece"], **led.snap())
            pf1[f"reflex@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D]["reflex"])
            pf1[f"gains@{D}_match"] = bool((dgains[D]["kp"], dgains[D]["kd"]) == B1_GAINS[D])
        mx = max(v for k, v in pf1.items() if k.endswith("_max_abs"))
        pf1["gains_match"] = all(v for k, v in pf1.items() if k.endswith("_match"))
        # The gate pins b1's OWN config (DCFG / D_KP_GRID / D_KD_GRID) rather than inheriting
        # this node's knobs, so it is DEFINED at every treatment config — including `--quick`,
        # where it is the smoke's strongest single check. Rollouts are deterministic and chunking
        # across the pool is bit-identical, so `n_proc` cannot move it either.
        pf1["applicable"] = bool(int(cfg["seed"]) == 0)
        pf1["max_abs"] = mx
        pf1["pass"] = bool(pf1["applicable"] and mx == 0.0 and pf1["gains_match"])
        pf1["n_checks"] = int(sum(1 for k in pf1 if k.endswith("_max_abs")))
        out["P_F1"] = pf1
        out["donor_rows"] = donor_rows
        out["donor_library"] = dict(sizes=lib.sizes(), build_ledger=lb.snap())
        out["t_pf1"] = time.time() - t0
        PR(f"[P-F1] fork on the DONOR square vs acappella b1: {pf1['n_checks']} checks, "
           f"max|delta| = {mx:.3e}  gains_match={pf1['gains_match']}  "
           f"applicable={pf1['applicable']}  pass={pf1['pass']}  [{out['t_pf1']:.0f}s]")
        Wd.close(); Wdc.close()
        save()
        if pf1["applicable"] and not pf1["pass"]:
            raise SystemExit("P-F1 failed: the fork is not acappella b1's code path")
    else:
        out["P_F1"] = dict(applicable=False, note="skipped by --no-pf1")
        PR("[P-F1] SKIPPED by flag")

    # =============================================== P-T: the tempo calibration (R x Delta)
    # RULES, WRITTEN DOWN BEFORE THE GRID IS READ (legato F2 — a shared calibration knob may not
    # be chosen by one arm's minimum error; both guards below name ONLY the incumbent, which is
    # the thing every committed arm is measured against, and neither can be moved by a committed
    # arm's outcome):
    #   competence guard  e_reflex(D=0, gains re-fit on the CLEAN world, measured on the world
    #                       the arms face) <= band(R) — the incumbent must be INSIDE the
    #                       playability band at zero delay, or there is no era 0 to leave and
    #                       nothing in the node is measurable. Stated in the band's own units
    #                       rather than as a second arbitrary constant, so the calibration and
    #                       the era ladder use ONE quantity. presto's own competence guard
    #                       ("practised error <= half the do-nothing floor") is computed and
    #                       reported beside it at every cell as the independent second read.
    #   demand guard      e_reflex(D=8) > band(R) — at the arc's own published niche delay the
    #                       incumbent must be OUT of the band, or the delay axis has no bite
    #                       (offbook d0's measured failure: playable-for-everyone).
    #   R* = the LARGEST R passing both guards — the fastest tempo feel can still play. That is
    #        the escalation's own design principle stated as a rule ("a piece whose TEMPO is set
    #        against the reflex delay", presto's header) and it names only the incumbent.
    #        REVISED 2026-09-05 on the `pcal` grid, before any committed arm existed and before
    #        any treatment number was computed; the first draft selected `argmax usable range =
    #        e_reflex(16)/e_reflex(0)`, and the grid showed that criterion is maximised by making
    #        the piece EASY (the range is a ratio to the incumbent's own floor, so it grows as
    #        the floor shrinks: 41.4x at R = 0.08 against 4.7x at R = 0.28) — i.e. it selects
    #        against the whole reason this node exists. `usable_range` is still computed and
    #        reported at every cell as an instrument. The full reversal, with the grid it was
    #        made on, is FILES.md decision 8.
    #   fallback          if none passes both, take the LARGEST R passing the competence guard
    #                       and say so in the record.
    t0 = time.time()
    cal, cal_rows = {}, []
    for R in cfg["r_ladder"]:
        pc = P.fast_piece(R=R, k_app=cfg["k_app"], h_seg=cfg["h_seg"],
                          n_levels=cfg["n_levels"], damping=cfg["damping"])
        Wr = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
        Wrc = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=False)
        evr = Wr.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_cal"])
        row = dict(R=float(R), **pc.band_parts(), speed=pc.mean_leg() / (pc.H * pc.dt_ctrl))
        # presto's arm-neutral floor: the zero-command policy from the same start states.
        led = Ledger(Wr.dt_ctrl, Wr.d_fb)
        odn = Wr.traverse(ZeroDecider(Wr), evr, np.random.default_rng(1), led)
        row["do_nothing"] = odn["e_piece"]
        for D in cfg["cal_deltas"]:
            grid = []
            for kp in cfg["kp_grid"]:
                for kd in cfg["kd_grid"]:
                    Wrc.kp, Wrc.kd = float(kp), float(kd)
                    led = Ledger(Wr.dt_ctrl, Wr.d_fb)
                    oc = Wrc.traverse(ReflexDecider(), evr, np.random.default_rng(1), led,
                                      obs_delay=D)
                    grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"],
                                     e_app=oc["e_app"]))
            b = min(grid, key=lambda r: r["e_piece"])
            Wr.kp, Wr.kd = b["kp"], b["kd"]
            led = Ledger(Wr.dt_ctrl, Wr.d_fb)
            orot = Wr.traverse(ReflexDecider(), evr, np.random.default_rng(1), led, obs_delay=D)
            row[f"clean@{D}"] = b["e_piece"]
            row[f"rot@{D}"] = orot["e_piece"]
            row[f"app@{D}"] = orot["e_app"]
            row[f"gains@{D}"] = [b["kp"], b["kd"]]
            row[f"edge@{D}"] = [bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"]))),
                                bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))]
        d0, d8, dmax = cfg["cal_deltas"][0], cfg["cal_deltas"][1], cfg["cal_deltas"][-1]
        row["competence"] = bool(row[f"rot@{d0}"] <= row["band"])
        row["presto_competence"] = bool(row[f"rot@{d0}"] <= 0.5 * row["do_nothing"])
        row["e_over_leg"] = row[f"rot@{d0}"] / row["mean_leg"]
        row["demand"] = bool(row[f"rot@{d8}"] > row["band"])
        row["usable_range"] = row[f"rot@{dmax}"] / max(row[f"rot@{d0}"], 1e-9)
        cal_rows.append(row)
        PR(f"[P-T] R={R:.3f}  leg={row['mean_leg']:.4f}  v={row['speed']:.2f} m/s  "
           f"band={row['band']:.4f}  dn={row['do_nothing']:.4f}  "
           f"e_rot@{d0}={row[f'rot@{d0}']:.4f} ({row['e_over_leg']:.2f} leg, "
           f"gains {row[f'gains@{d0}']}, edge {row[f'edge@{d0}']})  "
           f"@{d8}={row[f'rot@{d8}']:.4f}  @{dmax}={row[f'rot@{dmax}']:.4f}  "
           f"range={row['usable_range']:.1f}x  comp={row['competence']}"
           f"/presto {row['presto_competence']}  demand={row['demand']}")
        Wr.close(); Wrc.close()
    ok = [r for r in cal_rows if r["competence"] and r["demand"]]
    if ok:
        best = max(ok, key=lambda r: r["R"])
        rule = "both guards; R* = the largest (fastest) R the incumbent can still play in-band"
    else:
        comp = [r for r in cal_rows if r["competence"]]
        best = (max(comp, key=lambda r: r["R"]) if comp else min(cal_rows, key=lambda r: r["R"]))
        rule = ("FALLBACK: no R passed both guards; largest R passing competence"
                if comp else "FALLBACK: no R passed the competence guard; SMALLEST R — the piece "
                             "is unplayable by the incumbent at every tempo on the ladder, which "
                             "is a finding and not a knob")
    R_star = float(cfg["r_force"]) if cfg["r_force"] > 0 else float(best["R"])
    cal = dict(rows=cal_rows, rule=rule, R_star=R_star, forced=bool(cfg["r_force"] > 0),
               n_pass_both=len(ok))
    out["P_T"] = cal
    out["t_pt"] = time.time() - t0
    PR(f"[P-T] R* = {R_star:.3f}  ({rule})  [{out['t_pt']:.0f}s]")
    save()
    if cfg["cal_only"]:
        out["complete"] = True
        out["note"] = "calibration probe only (--cal-only): nothing after P-T was run"
        save()
        PR("[cal-only] stopping after P-T by flag")
        return out

    # ============================================================ the piece, and its Delta grid
    pc = P.fast_piece(R=R_star, k_app=cfg["k_app"], h_seg=cfg["h_seg"],
                      n_levels=cfg["n_levels"], damping=cfg["damping"])
    W = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
    Wc = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=False)
    K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
    band = pc.band()
    out["piece"] = pc.describe()
    PR(f"[piece] {pc.name}: {K} drilled x {H} steps ({1000 * H * W.dt_ctrl:.0f} ms) + {k0} "
       f"approach; mean leg {pc.mean_leg():.4f} m, speed {out['piece']['speed']:.2f} m/s, "
       f"turns {out['piece']['turns']}, band {band:.4f}, levels {L}")

    ev = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
    rt = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])

    t0 = time.time()
    gains = {}
    for D in cfg["deltas"]:
        grid = []
        for kp in cfg["kp_grid"]:
            for kd in cfg["kd_grid"]:
                Wc.kp, Wc.kd = float(kp), float(kd)
                led = Ledger(W.dt_ctrl, W.d_fb)
                oc = Wc.traverse(ReflexDecider(), ev, np.random.default_rng(1), led, obs_delay=D)
                grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
        b = min(grid, key=lambda r: r["e_piece"])
        b["at_kp_edge"] = bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"])))
        b["at_kd_edge"] = bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))
        # the a0-comparable control: the same fit restricted to a0's own grid floor
        r2 = [r for r in grid if r["kp"] >= A0_KP_FLOOR]
        b2 = min(r2, key=lambda r: r["e_piece"]) if r2 else dict(kp=None, kd=None, e_piece=None)
        b["restricted"] = dict(kp=b2["kp"], kd=b2["kd"], e_piece=b2["e_piece"],
                               floor=A0_KP_FLOOR)
        gains[D] = b
        PR(f"[cal] Delta={D:2d} ({D * W.dt_ctrl * 1000:3.0f} ms)  reflex kp={b['kp']:>6} "
           f"kd={b['kd']:>5}  clean e={b['e_piece']:.4f}  "
           f"edge(kp/kd)={b['at_kp_edge']}/{b['at_kd_edge']}  "
           f"[kp>={A0_KP_FLOOR} restricted: kp={b['restricted']['kp']} "
           f"kd={b['restricted']['kd']} e={b['restricted']['e_piece']:.4f}]")
    out["reflex_cal"] = {str(k): v for k, v in gains.items()}
    out["t_reflex_cal"] = time.time() - t0
    save()

    # ============================================================ the LADDER, built statically
    # THE LIBRARY IS BUILT ONCE, at Delta = 0, with the HARVEST gains frozen at the Delta = 0
    # fit (acappella's discipline: a0's gains were frozen through b1's whole sweep). A tape is
    # valid across delays; the era ladder therefore sweeps Delta over ONE vocabulary.
    t0 = time.time()
    HKP, HKD = gains[0]["kp"], gains[0]["kd"]
    W.kp, W.kd = HKP, HKD
    lvlib = LevelLibrary(K, cfg["n_slot"], L, poison=True)
    llb = Ledger(W.dt_ctrl, W.d_fb)
    b1_rec, bl_rec = [], []

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

    def score_set(level_map, kw, seed):
        led = Ledger(W.dt_ctrl, W.d_fb)
        sp = W.traverse(prefix_dec(level_map), rt, np.random.default_rng(seed), led)
        return sp["seam"][kw][:cfg["n_score"]]

    # --- LEVEL 1: solo's member cells, at every drilled seam, greedy nested prefix -----------
    for d in range(K):
        lm = {e: (1,) for e in range(d)}
        pools = harvest(lm, cfg["n_warm"], cfg["batch"], cfg["seed"] + 20000 + 97 * d)
        S0 = score_set(lm, k0 + d, cfg["seed"] + 4400 + d)
        slots, pois, rec = build_level1_cell(
            W, pools[k0 + d], S0, k0 + d, np.random.default_rng(cfg["seed"] + 77777 + 13 * d),
            llb, cfg["m_cand"], cfg["n_slot"], cfg["m_member"], cfg["n_poison"])
        for g in slots:
            lvlib.add_slot(1, d, g, np.mean([m["key"] for m in g], 0))
        if pois:
            lvlib.add_slot(1, d, pois, np.mean([m["key"] for m in pois], 0), poison=True)
        rec.update(level=1, drilled=int(d))
        b1_rec.append(rec)
    PR(f"[lib] L1 built: {lvlib.counts()}  {llb.ground:.0f} groundings  {time.time() - t0:.0f}s")

    # --- LEVELS 2..L: the nesting op, walked in order, greedy nested prefix ------------------
    for ell in range(2, L + 1):
        sp = LadderLayout.span(ell)
        done = []
        for d in range(0, K, sp):
            if d + sp > K:
                continue
            lm = {e: (ell,) for e in done}
            for e in range(K):
                lm.setdefault(e, (ell - 1,))
            S0 = score_set(lm, k0 + d, cfg["seed"] + 6600 + 71 * ell + d)
            slots, pois, rec = build_level_cell(
                W, lvlib, ell, d, S0,
                np.random.default_rng(cfg["seed"] + 88888 + 131 * ell + 13 * d), llb,
                cfg["m_weld"], cfg["n_slot"], cfg["m_member"], cfg["n_poison"])
            if slots:
                for g in slots:
                    lvlib.add_slot(ell, d, g["members"], g["key"], parents=g["parents"])
                if pois:
                    lvlib.add_slot(ell, d, pois, pois[0]["key"], poison=True)
                done.append(d)
            bl_rec.append(rec)
        PR(f"[lib] L{ell} built at seams {done}: {lvlib.counts()}  "
           f"{llb.ground:.0f} groundings cum  {time.time() - t0:.0f}s")
    out["library"] = dict(level1=b1_rec, levels=bl_rec, sizes=lvlib.sizes(),
                          counts=lvlib.counts(), build_ledger=llb.snap(),
                          harvest_gains=[HKP, HKD], n_slots=lvlib.layout.n_slots)
    out["t_library"] = time.time() - t0
    save()

    # ======================================================= P-N: the nesting op is EXACT
    pn = dict(rows=[], spell_bad=0, exec_max_abs=0.0, n_checked=0)
    for ell in range(2, L + 1):
        spn = LadderLayout.span(ell)
        for d in range(0, K, spn):
            for sl in lvlib.slots(ell, d, with_poison=False):
                pa = lvlib.slot_by_id(sl["parents"][0])
                pb = lvlib.slot_by_id(sl["parents"][1])
                halfL = (spn // 2) * H
                for mb in sl["members"]:
                    okA = any(np.array_equal(mb["cmds"][:halfL], q["cmds"]) for q in pa["members"])
                    okB = any(np.array_equal(mb["cmds"][halfL:], q["cmds"]) for q in pb["members"])
                    pn["n_checked"] += 1
                    if not (okA and okB):
                        pn["spell_bad"] += 1
    # execution identity: the weld played as one span == the halves played in sequence
    for ell in range(2, L + 1):
        spn = LadderLayout.span(ell)
        for d in range(0, K, spn):
            cell = lvlib.slots(ell, d, with_poison=False)
            if not cell:
                continue
            sl = cell[0]
            mb = sl["members"][0]
            S0 = rt[:min(8, len(rt))]
            nS = len(S0)
            h2 = spn // 2
            e_w, _ = W.rollout(S0, np.tile(mb["cmds"][None], (nS, 1, 1)), k0 + d, spn, None)
            e_a, fa = W.rollout(S0, np.tile(mb["cmds"][:h2 * H][None], (nS, 1, 1)), k0 + d, h2,
                                None)
            e_b, _ = W.rollout(fa, np.tile(mb["cmds"][h2 * H:][None], (nS, 1, 1)), k0 + d + h2,
                               h2, None)
            mxe = float(np.max(np.abs(np.concatenate([e_a, e_b], 1) - e_w)))
            pn["exec_max_abs"] = max(pn["exec_max_abs"], mxe)
            pn["rows"].append(dict(level=int(ell), drilled=int(d), max_abs=mxe))
    # The SPELLING half is exact (bit-equality of command arrays) and is the binding one. The
    # EXECUTION half carries a stated tolerance of 1e-6 and cannot be exact: `World.rollout`
    # casts its start states to float32 (the donor's line, untouched), so playing the halves in
    # sequence round-trips the internal hand-over state through float32 while the weld keeps it
    # in float64 inside MuJoCo. The residual measured below IS that round-trip, not composition
    # error — which is the point: on a deterministic plant, welding measured content is free
    # (`legato` F4 measured -0.004 on the arm; here it is 1e-7). What a level changes is not
    # execution but WHEN the second half was chosen.
    pn["exec_tol"] = 1e-6
    pn["pass"] = bool(pn["spell_bad"] == 0 and pn["exec_max_abs"] <= pn["exec_tol"]
                      and pn["n_checked"] > 0)
    out["P_N"] = pn
    PR(f"[P-N] nesting: {pn['n_checked']} members spelled by their parents "
       f"({pn['spell_bad']} bad); weld == halves in sequence max|delta| = "
       f"{pn['exec_max_abs']:.3e} (tol {pn['exec_tol']:.0e}, the float32 hand-over round-trip)"
       f"  pass={pn['pass']}")
    save()

    # ============================================================== THE SWEEP: the era ladder
    t0 = time.time()
    ARMS = [("reflex", None, None)]
    for ell in range(1, L + 1):
        ARMS.append((f"key_L{ell}", "key", (ell,)))
    for ell in range(1, L + 1):
        ARMS.append((f"aud_L{ell}", "audit", (ell,)))
    ARMS += [("key_all", "key", tuple(range(1, L + 1))),
             ("aud_all", "audit", tuple(range(1, L + 1))),
             ("reflex_ol", "frozen", None)]
    frac_ths = dict(q=0.25, t=1.0 / 3.0, h=0.5)
    sweep, hand = [], []
    for app in cfg["app_modes"]:
        Dapp = APP_MODES[app]
        W.kp_app, W.kd_app = ((gains[0]["kp"], gains[0]["kd"]) if APP_D0_GAINS[app]
                              else (None, None))
        for D in cfg["deltas"]:
            W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
            # ---- the HAND-OVER instrument: what state the drilled figure is launched from,
            # and what the agent READS there. Identical for every arm at a given (mode, Delta)
            # because the approach is shared, so it is measured once.
            led = Ledger(W.dt_ctrl, W.d_fb)
            oh = W.traverse(ReflexDecider(), ev, np.random.default_rng(cfg["seed"] + 35), led,
                            obs_delay=D, obs_delay_app=Dapp, collect_obs=True)
            st0 = oh["seam"][k0]
            rd0 = oh["obs"][:, k0 * H, :]
            vnom = (W.wps[k0 + 1] - W.wps[k0]) / (H * W.dt_ctrl)
            hand.append(dict(
                app=app, delta=int(D),
                ho_wp=float(np.mean(np.linalg.norm(st0[:, :2] - W.wps[k0][None, :], axis=1))),
                ho_speed=float(np.mean(np.linalg.norm(st0[:, 2:], axis=1))),
                ho_v_err=float(np.mean(np.linalg.norm(st0[:, 2:] - vnom[None, :], axis=1))),
                read_pos_err=float(np.mean(np.linalg.norm(rd0[:, :2] - st0[:, :2], axis=1))),
                read_vel_err=float(np.mean(np.linalg.norm(rd0[:, 2:] - st0[:, 2:], axis=1))),
                # the read at drilled seam 0 is clamped at the traversal START whenever the
                # approach is shorter than the delay: k0*H < Delta. At k0*H = 15 and Delta = 16
                # the agent reads the RESET state, which is aliasing, not a niche.
                clamped=bool(k0 * H < D), read_step=int(max(0, k0 * H - D)),
                pos_mean=[float(x) for x in st0[:, :2].mean(0)],
                pos_sd=[float(x) for x in st0[:, :2].std(0)],
                vel_mean=[float(x) for x in st0[:, 2:].mean(0)],
                vel_sd=[float(x) for x in st0[:, 2:].std(0)],
                e_app=oh["e_app"]))
            for nm, mode, lv in ARMS:
                led = Ledger(W.dt_ctrl, W.d_fb)
                dec = (ReflexDecider() if mode is None else
                       FrozenReflexDecider(W) if mode == "frozen" else
                       LadderDecider(W, lvlib, mode=mode, levels=lv, obs_delay=D))
                o = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                               obs_delay=D, obs_delay_app=Dapp)
                ep = np.asarray(o["ep"], np.float64)
                row = dict(delta=int(D), arm=nm, mode=mode, app=app,
                           levels=(None if lv is None else list(lv)),
                           e_piece=o["e_piece"], e_app=o["e_app"],
                           e_seg=[float(x) for x in o["e_seg"]],
                           ep_med=float(np.median(ep)), ep_mean=float(ep.mean()),
                           in_band=float(np.mean(ep <= band)),
                           **{f"in_{kk}": float(np.mean(ep <= v * pc.mean_leg()))
                              for kk, v in frac_ths.items()},
                           ledger=led.snap(), ledger_drilled=o["ledger_drilled"])
                if mode is not None and dec.picks:
                    row["n_aud"] = float(np.mean([p["n_aud"] for p in dec.picks]))
                    row["n_dec"] = len(dec.picks)
                    row["mean_level"] = float(np.mean([p["mean_level"] for p in dec.picks]))
                    lvs = [x for p in dec.picks for x in p["level"]]
                    row["frac_by_level"] = {str(e): float(np.mean(np.asarray(lvs) == e))
                                            for e in range(1, L + 1)}
                    row["n_distinct_slot"] = len({x for p in dec.picks for x in p["slot"]})
                sweep.append(row)
            cur = [r for r in sweep if r["delta"] == D and r["app"] == app]
            ref = [r for r in cur if r["arm"] == "reflex"][0]
            ol = [r for r in cur if r["arm"] == "reflex_ol"][0]
            best = min((r for r in cur if r["arm"] not in ("reflex", "reflex_ol")),
                       key=lambda r: r["e_piece"])
            PR(f"[sweep/{app}] Delta={D:2d}  reflex={ref['e_piece']:.4f} "
               f"(fb {ref['ledger_drilled']['n_fb']:.0f})  ol={ol['e_piece']:.4f} "
               f"(x{ol['e_piece'] / max(ref['e_piece'], 1e-9):.2f})  " + "  ".join(
                   f"{r['arm'].replace('key_L', 'kL').replace('aud_L', 'aL')}={r['e_piece']:.4f}"
                   for r in cur if r["arm"] not in ("reflex", "reflex_ol"))
               + f"   best={best['arm']} band={band:.4f}")
            save()
    out["sweep"] = sweep
    out["handover"] = hand
    out["t_sweep"] = time.time() - t0
    PR("[hand-over] the state the drilled figure is launched from, per approach mode:")
    for r in hand:
        PR(f"      app={r['app']:>4s} D={r['delta']:2d}  |x-V0|={r['ho_wp']:.4f}  "
           f"|v|={r['ho_speed']:.3f}  |v-v_nom|={r['ho_v_err']:.3f}  "
           f"read err pos={r['read_pos_err']:.4f} vel={r['read_vel_err']:.3f}  "
           f"read@step {r['read_step']}/{k0 * H}{'  CLAMPED-AT-RESET' if r['clamped'] else ''}  "
           f"e_app={r['e_app']:.4f}")

    # ------------------------------------- the era ladder + the ordering margins, per Delta
    def cell(app, D, nm):
        for r in sweep:
            if r["delta"] == D and r["arm"] == nm and r["app"] == app:
                return r
        return None

    ladder_by, eras_by = {}, {}
    for app in cfg["app_modes"]:
        eras, ladder_rows = {}, []
        for D in cfg["deltas"]:
            rf = cell(app, D, "reflex")
            ol = cell(app, D, "reflex_ol")
            row = dict(delta=int(D), ms=round(D * W.dt_ctrl * 1000), band=band, app=app,
                       reflex=rf["e_piece"], reflex_ol=ol["e_piece"],
                       reflex_ol_ratio=ol["e_piece"] / max(rf["e_piece"], 1e-9),
                       reflex_fb=rf["ledger_drilled"]["n_fb"],
                       reflex_ol_fb=ol["ledger_drilled"]["n_fb"],
                       **{f"reflex_{k2}": rf[k2] for k2 in ("in_h", "in_t", "in_q")},
                       **{f"reflex_ol_{k2}": ol[k2] for k2 in ("in_h", "in_t", "in_q")})
            for pre in ("key", "aud"):
                for ell in range(1, L + 1):
                    c = cell(app, D, f"{pre}_L{ell}")
                    row[f"{pre}_L{ell}"] = c["e_piece"]
                    row[f"{pre}_L{ell}_in"] = c["in_band"]
                    row[f"{pre}_L{ell}_fb"] = c["ledger_drilled"]["n_fb"]
                for ell in range(1, L):
                    row[f"{pre}_m{ell}{ell + 1}"] = (row[f"{pre}_L{ell}"]
                                                     - row[f"{pre}_L{ell + 1}"])
                row[f"{pre}_m_ref"] = (row["reflex"]
                                       - min(row[f"{pre}_L{e}"] for e in range(1, L + 1)))
                row[f"{pre}_best"] = min(range(1, L + 1), key=lambda e: row[f"{pre}_L{e}"])
            row["key_all"] = cell(app, D, "key_all")["e_piece"]
            row["aud_all"] = cell(app, D, "aud_all")["e_piece"]
            row["aud_all_lvl"] = cell(app, D, "aud_all").get("mean_level")
            ladder_rows.append(row)
        for pre in ("key", "aud"):
            for ell in range(1, L):
                hit = None
                for row in ladder_rows:
                    if row[f"{pre}_L{ell}"] > band and row[f"{pre}_L{ell + 1}"] <= band:
                        hit = int(row["delta"])
                        break
                eras[f"{pre}_era_{ell + 1}"] = hit
            hit = None
            for row in ladder_rows:
                if row["reflex"] > band and row[f"{pre}_L1"] <= band:
                    hit = int(row["delta"])
                    break
            eras[f"{pre}_era_1"] = hit
        ladder_by[app] = ladder_rows
        eras_by[app] = eras
        PR(f"[era/{app}] smallest Delta at which level l fails the band and l+1 does not: "
           + json.dumps(eras))
    out["ladder"] = ladder_by[cfg["app_modes"][0]]      # back-compatible: the default mode
    out["eras"] = eras_by[cfg["app_modes"][0]]
    out["ladder_by_mode"] = ladder_by
    out["eras_by_mode"] = eras_by
    # PRE-FIXED READING (decision 19), stated before the grid: the closed-loop reflex is called
    # A TAPE at a Delta when its open-loop replay is within 10% of it -- the feedback events it
    # is charged for buy less than 10%. The continuous ratio is reported at every cell.
    out["reflex_is_tape"] = {
        app: {str(r["delta"]): dict(ratio=r["reflex_ol_ratio"],
                                    tape=bool(r["reflex_ol_ratio"] <= 1.10))
              for r in ladder_by[app]} for app in cfg["app_modes"]}
    PR("[ol] closed-loop reflex vs its own open-loop replay (ratio ol/cl; <=1.10 => the "
       "incumbent is a tape at that Delta):")
    for app in cfg["app_modes"]:
        PR(f"      app={app:>4s}  " + "  ".join(
            f"D{r['delta']}={r['reflex_ol_ratio']:.2f}"
            + ("*" if r["reflex_ol_ratio"] <= 1.10 else "")
            for r in ladder_by[app]))

    # ============================================ P-A: the approach removed the artifact
    W.kp_app, W.kd_app = None, None
    pa = {}
    for pre in ("key", "aud"):
        for ell in range(1, L + 1):
            v = [cell(cfg["app_modes"][0], D, f"{pre}_L{ell}")["e_piece"] for D in cfg["deltas"]]
            pa[f"{pre}_L{ell}"] = dict(spread=float(max(v) - min(v)),
                                       ratio=float(max(v) / max(min(v), 1e-9)),
                                       vals=[float(x) for x in v])
    deep = pa[f"key_L{L}"]["spread"]
    pa["pass"] = bool(deep > 0.0)
    pa["note"] = ("acappella finding 5: on etude's square the chain arms decide once at seam 0 "
                  "from rest, so their read is the TRUE state at every Delta and they are "
                  "exactly delay-invariant (0.0852 / 0.1184 to four decimals). With k0 approach "
                  "segments the deepest arm's spread over the ladder is this node's measurement "
                  "that the artifact is gone.")
    out["P_A"] = pa
    PR(f"[P-A] deepest committed arm (key_L{L}) spread over the Delta ladder: "
       f"{deep:.4f} ({pa[f'key_L{L}']['ratio']:.2f}x)  pass={pa['pass']}")
    save()

    # =============================================== P-S: seam information, per level per Delta
    ss = []
    for app in cfg["app_modes"]:
      W.kp_app, W.kd_app = ((gains[0]["kp"], gains[0]["kd"]) if APP_D0_GAINS[app]
                            else (None, None))
      for D in cfg["ps_deltas"]:
        W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
        led = Ledger(W.dt_ctrl, W.d_fb)
        dec = LadderDecider(W, lvlib, mode="key", levels=(1,), obs_delay=D)
        okey = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led, obs_delay=D,
                          obs_delay_app=APP_MODES[app], collect_obs=True)
        for ell in range(1, L + 1):
            spn = LadderLayout.span(ell)
            for d in range(0, K, spn):
                S = lvlib.slots(ell, d, with_poison=False)
                if not S:
                    continue
                st = okey["seam"][k0 + d]
                sc = np.full((len(st), len(S)), np.inf)
                for si, sl in enumerate(S):
                    per = np.full(len(st), np.inf)
                    for mb in sl["members"]:
                        e, _ = W.rollout(st, np.tile(mb["cmds"][None], (len(st), 1, 1)),
                                         k0 + d, spn, None)
                        per = np.minimum(per, e.mean(1))
                    sc[:, si] = per
                fixed = float(sc.mean(0).min())
                oracle = float(sc.min(1).mean())
                ss.append(dict(app=app, delta=int(D), level=int(ell), drilled=int(d),
                               best_fixed_slot=fixed, per_state_oracle=oracle,
                               gain=fixed / max(oracle, 1e-9),
                               n_distinct_argmin=int(len(np.unique(sc.argmin(1)))),
                               n_slot=len(S)))
    out["P_S"] = ss
    PR("[P-S] seam information under the DELAYED read (per-state oracle over each level's cell):")
    for app in cfg["app_modes"]:
        for D in cfg["ps_deltas"]:
            for ell in range(1, L + 1):
                rr = [r for r in ss if r["app"] == app and r["delta"] == D and r["level"] == ell]
                if not rr:
                    continue
                PR(f"      app={app:>4s} D={D:2d} L{ell}  mean gain "
                   f"{sum(r['gain'] for r in rr) / len(rr):.3f}x  "
                   f"(range {min(r['gain'] for r in rr):.2f}-{max(r['gain'] for r in rr):.2f})  "
                   f"argmins {sum(r['n_distinct_argmin'] for r in rr) / len(rr):.1f}/"
                   f"{rr[0]['n_slot']}  oracle "
                   f"{sum(r['per_state_oracle'] for r in rr) / len(rr):.4f}")

    out["flags"] = dict(
        gain_grid_edges={str(k): dict(kp=v["at_kp_edge"], kd=v["at_kd_edge"])
                         for k, v in gains.items()},
        library_built_once=("Built at Delta = 0 with the harvest gains frozen at the Delta = 0 "
                            "fit, then swept across Delta (acappella's discipline). The launch "
                            "distribution at the first drilled seam DOES move with Delta, "
                            "because the shared approach is played closed-loop by the reflex at "
                            "the per-Delta re-fit gains; every arm inherits the same shift, and "
                            "`e_app` is reported at every cell so the cost of the lead-in is on "
                            "the record."),
        approach=("The approach uses the SAME per-Delta re-fit gains as the reflex arm — every "
                  "advantage to the incumbent, applied to the shared lead-in, so it is "
                  "arm-neutral. `e_app` per Delta says what the lead-in itself cost."),
        cross_level_scoring=("A decision across levels compares the MEAN over the span's "
                             "waypoints (offbook's cross-level convention, carried from "
                             "acappella so `aud_all` stays comparable). acappella finding 6 says "
                             "this selector is the most delay-fragile thing in the system; "
                             "`aud_all`'s `mean_level` per Delta is that instrument here."),
        no_mining=("Phase A builds every rung STATICALLY, by construction audition. Nothing is "
                   "mined from the learner's own routed successes and there is no at_support "
                   "here: that is Phase B."),
        approach_instrument=("DECISION 17. `app=zero` plays the shared approach at Delta = 0 "
                             "while the drilled figure keeps the arm's Delta. The lead-in is "
                             "already excluded from the comparison (presto decision 3 had it as "
                             "a shared PRE-COMPUTED plan precisely so it was not a variable); "
                             "under `app=arm` every committed arm launches the figure from a "
                             "hand-over the delayed incumbent has already lost, which is "
                             "acappella finding 5 traded for its mirror. Both columns are "
                             "reported at every cell; `app=arm` is the arm of record. The read "
                             "at drilled seam 0 is a SEPARATE matter and is clamped at the "
                             "traversal start whenever k0*H < Delta (here 15 < 16), which the "
                             "hand-over log reports as `clamped` per cell."),
        damping_sweep=("DECISION 18. `damping` is a plant knob. Any tag run at a damping other "
                       "than the donor's 2.0 LEAVES THE DONOR PLANT: no cross-tag control "
                       "against acappella, solo or etude applies to its treatment numbers, and "
                       "none is claimed. P-F1 is unaffected — it builds the DONOR piece, which "
                       "carries damping 2.0 whatever the treatment cell is — and must still "
                       "pass at 0.000e+00 in every tag. The quantity the sweep is about is "
                       "seg/tau = H*dt_ctrl/(m/c): 0.24 at (c=2,H=5), 0.48 at (2,10), 0.60 at "
                       "(5,5), 1.20 at (10,5), against etude's 1.63. Terminal speed is "
                       "gear/c, so a higher damping caps the tempo and the R rule re-selects."),
        open_loop_reflex=("DECISION 19. `reflex_ol` is an INSTRUMENT, not an arm: the re-fit "
                          "reflex law evaluated ONCE at the drilled launch and played open-loop "
                          "to the end. PRE-FIXED READING, stated before the grid: the "
                          "closed-loop incumbent is called A TAPE at a Delta when the ratio "
                          "ol/cl <= 1.10, i.e. the 40 feedback events it is charged for buy "
                          "less than 10%. The continuous ratio is reported at every cell. The "
                          "kp grid also gains two rungs below a0's floor (0.5, 1.0) so the "
                          "re-fit's floor is measured rather than clipped; every cell reports "
                          "the argmin restricted to kp >= 2.0 (a0's own grid) beside it."))
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    W.close(); Wc.close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


@app.local_entrypoint()
def prestissimo_ladder(
    quick: bool = False,
    spawn: bool = False,
    cal_only: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    no_pf1: bool = False,
    r_force: float = 0.0,
    k_app: int = P.K_APP,
    h_seg: int = P.H_SEG,
    n_levels: int = P.N_LEVELS,
    n_eval: int = 32,
    n_cal: int = 16,
    n_rt: int = 48,
    batch: int = 24,
    n_warm: int = 8,
    n_score: int = 16,
    n_slot: int = 6,
    m_cand: int = 48,
    m_member: int = 4,
    m_weld: int = 6,
    n_poison: int = 3,
    damping: float = P.DAMPING,
    app_modes: str = "arm,fixed",
    ps_deltas: str = "0,4,8",
):
    cfg = dict(tag=tag or "a0", seed=seed, n_proc=n_proc, quick=bool(quick),
               do_pf1=not no_pf1, r_force=float(r_force), cal_only=bool(cal_only),
               k_app=k_app, h_seg=h_seg, n_levels=n_levels,
               r_ladder=list(P.R_LADDER), cal_deltas=[0, 8, 16],
               deltas=list(DELTAS), damping=float(damping),
               app_modes=[x for x in app_modes.split(",") if x],
               ps_deltas=[int(x) for x in ps_deltas.split(",") if x],
               kp_grid=list(T_KP_GRID), kd_grid=list(T_KD_GRID),
               n_eval=n_eval, n_cal=n_cal, n_rt=n_rt, batch=batch, n_warm=n_warm,
               n_score=n_score, n_slot=n_slot, m_cand=m_cand, m_member=m_member,
               m_weld=m_weld, n_poison=n_poison,
               # inert here (the fork carries solo's search machinery unused in Phase A)
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5)
    if quick:
        cfg.update(tag=tag or "psmoke", n_proc=8, n_eval=8, n_cal=6, n_rt=8, batch=6, n_warm=2,
                   n_score=4, n_slot=3, m_cand=8, m_member=2, m_weld=2, n_poison=2,
                   r_ladder=[0.20, 0.28], deltas=[0, 4, 8, 16], ps_deltas=[0, 8],
                   kp_grid=[2.0, 10.0, 40.0], kd_grid=[1.0, 4.0])
    if spawn:
        c = run_ladder.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_prestissimo/{cfg['tag']}/ladder.json")
        return
    r = run_ladder.remote(cfg)
    print("\n".join(r["log"][-100:]))
