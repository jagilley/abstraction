"""accelerando Phase 1 — the fast body, the tempo ladder, recordings only.

THE QUESTION. `prestissimo/` put a four-rung execution-span ladder on `solo/`'s pusher and found
the mapping's core prediction holds model-free, but three of its measurements shape this node:
the band `tau < T < Delta` was EMPTY on that plant (tau = 0.5 s against a 0.19 s niche delay), so
Delta had to be inflated to 384 ms to reach it; rungs 1-2 never paid on their own; and seam
information was ~1.0 under delay in every cell. A pianist's Delta does not change — T does. So:
make the band exist physically (a FAST BODY), fix Delta at a human value, and sweep TEMPO as the
era ladder.

    at which TEMPO does a level-l unit stop being playable, and does level l+1 stay playable,
    at a delay that never moves?

Phase 1 is the calibration everything else reads from: whether tempo alone opens a rung-by-rung
ladder on a fast body, and whether seams carry information. It has NO learned component — the
reflex law, executed content, and the plant as the only oracle — and, standing constraint from
`acappella/`, `solo/` and `prestissimo/`, **no forward model anywhere**: nothing here imports,
trains or evaluates an `f(s,u)`.

THE THREE CONTENT SOURCES IN PHASE 1 (the `prog` head is Phase 2)
  `rec`    a recording harvested AT THE CURRENT TEMPO — prestissimo's construction, one library
           per tempo. What a tape is worth when the tempo it was cut at is the tempo asked for.
  `scl0`   the SLOWEST tempo's recording, time-resampled note-wise to the asked tempo. The
           cheapest possible program: "play the tape faster".
  `scl2`   the same, with the amplitude scaled by `(H_src/H_dst)**2` — the dimensional-analysis
           correction for an inertial body. Free, so it is measured rather than argued.

THE TWO GRADES, both pre-fixed and reported at every cell
  `e_piece`   prestissimo's per-waypoint drilled error, with the 1/2-leg band and pass fractions
              at 1/4, 1/3, 1/2 mean leg.
  `e_listen`  THE LISTENER'S CLOCK: executed path and schedule reference both low-passed with the
              same 384 ms (one slow note) boxcar and compared densely, so at fast tempi several
              notes fall inside one grading unit. Same units, same three widths.

THE GATES (all pre-fixed, all reported whether or not they pass)
  P-F0   the 13 donor constants, read out of `etude/etude.py` with `ast`, never imported.
  P-F1   the fork runs the DONOR SQUARE and reproduces `acappella/b1`'s library build and its
         Delta = 0 / Delta = 8 rows at max|delta| = 0.
  P-F2   the fork runs `prestissimo/a1g`'s FAST PIECE on the donor plant and reproduces its
         library-construction scores, its nesting identity and a pinned set of its sweep rows at
         max|delta| = 0. P-F1 does not exercise the fast-piece code path; this does.
  P-TAU  the body's velocity time constant is MEASURED from a step response and matched against
         `m/c` on both the fast body and the donor body (the measurement's own control).
  P-R    the resampler is the identity at `s = 1`: the source tempo's library rebuilt through
         `resample_cmds` is bit-equal to itself.
  P-T    the tempo calibration: R x tempo on the incumbent alone, at the FIXED delay, under
         rules fixed before the grid is read.
  P-N    the nesting op is exact, at every tempo.
  P-A    the approach works: the deepest committed arm is not delay-invariant (read off the
         Delta instrument sweep at the design tempo).
  P-S    seam information per level per tempo — a calibration readout, not an assumption.

Run:
    cd experiments/                        # MODAL_PROFILE=chromatic
    modal run mjc/practice/tempo/accelerando/tempo.py::accelerando_tempo --quick --tag tsmoke
    modal run --detach mjc/practice/tempo/accelerando/tempo.py::accelerando_tempo --spawn --tag t1
    python3 mjc/practice/tempo/accelerando/analyze_tempo.py --tag t1 --fetch --figures
"""

import json
import os
import time

import modal  # noqa: F401  (the app is imported from mjc.shared; kept for parity with siblings)

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.tempo.accelerando import piece as P

# --- cross-tag exact controls against `acappella/b1` (offbook's bit-identity idiom) ---------- #
# Carried VERBATIM from `prestissimo/ladder.py`, which carried them from `solo/solo.py`, which
# carried them from `acappella/results/b1/delay.json`. Full precision.
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
DCFG = dict(n_eval=32, n_rt=48, batch=24, n_warm=8, n_cand=24, n_score=16, n_slot=8)
D_KP_GRID = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
D_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# --- cross-tag exact controls against `prestissimo/a1g` (the FAST-PIECE code path) ----------- #
# Full-precision values read off `prestissimo/results/a1g/ladder.json` (run 2026-09-05, the clean
# approach variant on the DONOR plant, seed 0). P-F1 pins the donor SQUARE and cannot see the
# fast piece's ladder, the approach, the weld or the listener path; P-F2 pins all of them. Like
# P-F1 it carries a1g's OWN configuration, so it is defined at every treatment config.
A1G_CFG = dict(R=0.20, k_app=3, h_seg=5, n_levels=4, damping=2.0, mass=1.0,
               n_eval=32, n_rt=48, batch=24, n_warm=8, n_score=16, n_slot=6,
               m_cand=48, m_member=4, m_weld=6, n_poison=3)
A1G_GAINS = {0: (2.0, 16.0), 8: (0.5, 0.5)}       # a1g's own per-Delta re-fit, pinned
A1G_L1 = [0.08083933638241222, 0.08813551123042412, 0.04508835657363781, 0.10310668234174866,
          0.08768221062887338, 0.10810236797417319, 0.10048824217600387, 0.11578951283467628]
A1G_LN = [(2, 0, 0.07761806553513567), (2, 2, 0.12100779358405139),
          (2, 4, 0.15325333233875985), (2, 6, 0.258167714471215),
          (3, 0, 0.0800554545715215), (3, 4, 0.20127766089935728),
          (4, 0, 0.10558729441641992)]
A1G_ROWS = {                                       # app = "fixed", a1g's arm of record's twin
    0: {"reflex": 0.07258787751197815, "key_L1": 0.09392751008272171,
        "key_L4": 0.10430121421813965, "aud_L1": 0.1649807095527649,
        "aud_L4": 0.0926976278424263, "reflex_ol": 0.9855258464813232},
    8: {"reflex": 0.1534598171710968, "key_L1": 0.17738023400306702,
        "key_L4": 0.10818873345851898, "aud_L1": 0.20288512110710144,
        "aud_L4": 0.11276023089885712, "reflex_ol": 0.4668457508087158},
}
A1G_E_APP = 0.09328839927911758

# the TREATMENT gain grid. prestissimo's extended grid, unchanged: it already spans this body
# from "softer than a note" (kp = 0.5, natural period 0.69 s at A = 167 m/s^2) to bang-bang
# (kp = 160 saturates |u| = 1 at a 6 mm error), so both ends are past the interesting regime.
# Edges are reported at every cell (`at_kp_edge` / `at_kd_edge`), as in every node of this arc.
T_KP_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
T_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

# the approach modes, carried from prestissimo decision 20 with its measured verdict:
#   arm   — the lead-in reads at the arm's Delta and plays the per-Delta re-fit gains.
#   fixed — the lead-in reads at Delta = 0 AND plays the Delta = 0 gains, so it is ONE
#           pre-computed plan (presto decision 3) and the hand-over into drilled seam 0 is
#           numerically identical at every Delta. THE ARM OF RECORD, per the brief.
# (`zero` is not carried: prestissimo measured it to be a gain effect wearing a read effect's
# name — the hand-over got WORSE at every Delta >= 3 in all four diagnostic cells.)
APP_MODES = {"arm": None, "fixed": 0}
APP_D0_GAINS = {"arm": False, "fixed": True}

# the Delta ladder for the INSTRUMENT sweep. Delta is NOT the era knob here; it is the
# decomposition instrument run at each tempo, and `P.DELTA_FIX = 5` (120 ms) is the constant of
# the body every headline number is read at.
DELTAS_INSTR = (0, 2, 5, 8, 12)


@app.function(cpu=16.0, memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_tempo(cfg: dict) -> dict:
    import ast

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np

    from mjc.practice.tempo.accelerando.world import (World, Ledger, ReflexDecider, LibraryDecider,
                                                Library, select_tapes, LevelLibrary,
                                                LadderLayout, LadderDecider, build_level1_cell,
                                                build_level_cell, FrozenReflexDecider,
                                                resample_cmds, resample_library, listener_err)

    class ZeroDecider:
        """The do-nothing floor: zero command everywhere. Arm-neutral by construction, and the
        one number that says whether either grade is degenerate at a tempo."""

        def __init__(self, W):
            self.H = int(W.H)

        def __call__(self, k, states, need, rng, led):
            return dict(kind="open", cmds=np.zeros((len(states), self.H, 2), np.float32),
                        span=np.ones(len(states), int), src="zero")

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_accelerando", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "phase": "1",
           "contract": dict(
               no_forward_model=("Nothing in this node imports, trains or evaluates an f(s,u). "
                                 "Phase 1 has no learned component at all: the reflex law, "
                                 "executed content, the tempo resampler (arithmetic on a stored "
                                 "command tape, reading no state), and the plant as the only "
                                 "oracle."),
               era_knob=("TEMPO is the era knob and Delta is FIXED at a human value "
                         f"({P.DELTA_FIX} steps = {1000 * P.DELTA_FIX * P.DT_CTRL:.0f} ms), "
                         "chosen before any arm was read and held across all three phases. A "
                         "recording is a command sequence in absolute time and is wrong the "
                         "moment the tempo changes; that is what makes tempo the knob a "
                         "ratchet's library has to survive. The Delta sweep survives as the "
                         "instrument that decomposes a result into its feedback and execution "
                         "halves, run at each tempo, never as the era knob."),
               body=("The body is two numbers: tau = m/c and A = gear/m. Mass alone is changed "
                     "(1.0 -> 0.06), so tau = 0.03 s and A = 167 m/s^2 while gear, damping and "
                     "the terminal speed gear/c = 5 m/s stay at the donor's values. tau is "
                     "MEASURED from a step response (P-TAU), not assumed."),
               grades=("TWO, both pre-fixed and reported everywhere: prestissimo's per-waypoint "
                       "drilled error with the 1/2-leg band, and the listener's-clock error "
                       "(both signals low-passed with the same 384 ms boxcar). Pass fractions "
                       "at 1/4, 1/3, 1/2 mean leg for both, for every arm at every cell."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "tempo.json"), "w") as fh:
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
                donor_rows[f"{nm}@{D}"] = dict(e_piece=o["e_piece"], e_listen=o["e_listen"],
                                               **led.snap())
                pf1[f"{nm}@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D][nm])
            Wd.kp, Wd.kd = dgains[D]["kp"], dgains[D]["kd"]
            led = Ledger(Wd.dt_ctrl, Wd.d_fb)
            o = Wd.traverse(ReflexDecider(), ev_d, np.random.default_rng(cfg["seed"] + 35), led,
                            obs_delay=D)
            donor_rows[f"reflex@{D}"] = dict(e_piece=o["e_piece"], e_listen=o["e_listen"],
                                             **led.snap())
            pf1[f"reflex@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D]["reflex"])
            pf1[f"gains@{D}_match"] = bool((dgains[D]["kp"], dgains[D]["kd"]) == B1_GAINS[D])
        mx = max(v for k, v in pf1.items() if k.endswith("_max_abs"))
        pf1["gains_match"] = all(v for k, v in pf1.items() if k.endswith("_match"))
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

    # ===================================================================================== #
    # THE STATIC LADDER BUILD, factored out so gate P-F2 and every tempo run the SAME op.
    # prestissimo/ladder.py's block, verbatim in operation (decisions 3, 11, 12, 13):
    # level-1 cells are solo's member cells at every drilled seam; levels 2..L are welded pairs
    # of committed level-(l-1) slots; both the score set and the candidate pool come from the
    # configuration that will deploy the unit, walked in order; harvesting happens after the
    # full warm-up.
    # ===================================================================================== #
    def build_ladder(W, pc, kc, seed, harvest_gains, llb, rt, quiet=False):
        K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
        W.kp, W.kd = float(harvest_gains[0]), float(harvest_gains[1])
        lvlib = LevelLibrary(K, kc["n_slot"], L, poison=True)
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

        def score_set(level_map, kw, sd):
            led = Ledger(W.dt_ctrl, W.d_fb)
            sp = W.traverse(prefix_dec(level_map), rt, np.random.default_rng(sd), led)
            return sp["seam"][kw][:kc["n_score"]]

        for d in range(K):
            lm = {e: (1,) for e in range(d)}
            pools = harvest(lm, kc["n_warm"], kc["batch"], seed + 20000 + 97 * d)
            S0 = score_set(lm, k0 + d, seed + 4400 + d)
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
            if not quiet:
                PR(f"      [lib] L{ell} at seams {done}: {lvlib.counts()}  "
                   f"{llb.ground:.0f} groundings cum")
        return lvlib, b1_rec, bl_rec

    def check_nesting(W, lvlib, pc, rt):
        """P-N, at one tempo. Spelling is bit-equality and binding; execution carries the stated
        1e-6 tolerance because `World.rollout` casts start states to float32 (the donor's line),
        so playing the halves in sequence round-trips the internal hand-over through float32."""
        K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
        pn = dict(rows=[], spell_bad=0, exec_max_abs=0.0, n_checked=0)
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
        for ell in range(2, L + 1):
            spn = LadderLayout.span(ell)
            for d in range(0, K, spn):
                cell = lvlib.slots(ell, d, with_poison=False)
                if not cell:
                    continue
                mb = cell[0]["members"][0]
                S0 = rt[:min(8, len(rt))]
                nS = len(S0)
                h2 = spn // 2
                e_w, _ = W.rollout(S0, np.tile(mb["cmds"][None], (nS, 1, 1)), k0 + d, spn, None)
                e_a, fa = W.rollout(S0, np.tile(mb["cmds"][:h2 * H][None], (nS, 1, 1)),
                                    k0 + d, h2, None)
                e_b, _ = W.rollout(fa, np.tile(mb["cmds"][h2 * H:][None], (nS, 1, 1)),
                                   k0 + d + h2, h2, None)
                mxe = float(np.max(np.abs(np.concatenate([e_a, e_b], 1) - e_w)))
                pn["exec_max_abs"] = max(pn["exec_max_abs"], mxe)
                pn["rows"].append(dict(level=int(ell), drilled=int(d), max_abs=mxe))
        pn["exec_tol"] = 1e-6
        pn["pass"] = bool(pn["spell_bad"] == 0 and pn["exec_max_abs"] <= pn["exec_tol"]
                          and pn["n_checked"] > 0)
        return pn

    # ============================================== P-F2: the fork on prestissimo's FAST PIECE
    if cfg["do_pf2"]:
        t0 = time.time()
        ap = P.a1g_piece()
        Wa = World(cfg, ap, n_proc=int(cfg["n_proc"]), rot=True)
        kc = {k: A1G_CFG[k] for k in ("n_rt", "batch", "n_warm", "n_score", "n_slot",
                                      "m_cand", "m_member", "m_weld", "n_poison")}
        ev_a = Wa.start_states(np.random.default_rng(cfg["seed"] + 6000), A1G_CFG["n_eval"])
        rt_a = Wa.start_states(np.random.default_rng(cfg["seed"] + 5000), A1G_CFG["n_rt"])
        alb = Ledger(Wa.dt_ctrl, Wa.d_fb)
        alib, ab1, abl = build_ladder(Wa, ap, kc, cfg["seed"], A1G_GAINS[0], alb, rt_a,
                                      quiet=True)
        pf2 = {}
        pf2["build_L1_max_abs"] = max(abs(ab1[i]["chosen_score"] - A1G_L1[i])
                                      for i in range(len(A1G_L1)))
        got = {(r["level"], r["seam"]): r.get("chosen_score") for r in abl}
        pf2["build_Ln_max_abs"] = max(abs(got[(e, k)] - v) for e, k, v in A1G_LN)
        apn = check_nesting(Wa, alib, ap, rt_a)
        pf2["nesting"] = dict(spell_bad=apn["spell_bad"], exec_max_abs=apn["exec_max_abs"],
                              n_checked=apn["n_checked"], **{"pass": apn["pass"]})
        Wa.kp_app, Wa.kd_app = A1G_GAINS[0]                      # app = "fixed"
        a1g_rows = {}
        for D in (0, 8):
            Wa.kp, Wa.kd = A1G_GAINS[D]
            for nm in A1G_ROWS[D]:
                if nm == "reflex":
                    dec = ReflexDecider()
                elif nm == "reflex_ol":
                    dec = FrozenReflexDecider(Wa)
                else:
                    mode = "key" if nm.startswith("key") else "audit"
                    dec = LadderDecider(Wa, alib, mode=mode, levels=(int(nm[-1]),), obs_delay=D)
                led = Ledger(Wa.dt_ctrl, Wa.d_fb)
                o = Wa.traverse(dec, ev_a, np.random.default_rng(cfg["seed"] + 35), led,
                                obs_delay=D, obs_delay_app=0)
                a1g_rows[f"{nm}@{D}"] = dict(e_piece=o["e_piece"], e_listen=o["e_listen"],
                                             e_app=o["e_app"])
                pf2[f"{nm}@{D}_max_abs"] = abs(o["e_piece"] - A1G_ROWS[D][nm])
                if nm == "reflex":
                    pf2[f"e_app@{D}_max_abs"] = abs(o["e_app"] - A1G_E_APP)
        mx2 = max(v for k, v in pf2.items() if k.endswith("_max_abs"))
        pf2["applicable"] = bool(int(cfg["seed"]) == 0)
        pf2["max_abs"] = mx2
        pf2["n_checks"] = int(sum(1 for k in pf2 if k.endswith("_max_abs")))
        pf2["pass"] = bool(pf2["applicable"] and mx2 == 0.0 and apn["pass"])
        out["P_F2"] = pf2
        out["a1g_rows"] = a1g_rows
        out["t_pf2"] = time.time() - t0
        PR(f"[P-F2] fork on prestissimo's FAST PIECE (a1g) : {pf2['n_checks']} checks, "
           f"max|delta| = {mx2:.3e}  nesting {apn['n_checked']} spelled / "
           f"{apn['spell_bad']} bad / exec {apn['exec_max_abs']:.3e}  "
           f"applicable={pf2['applicable']}  pass={pf2['pass']}  [{out['t_pf2']:.0f}s]")
        Wa.close()
        save()
        if pf2["applicable"] and not pf2["pass"]:
            raise SystemExit("P-F2 failed: the fork is not prestissimo a1g's code path")
    else:
        out["P_F2"] = dict(applicable=False, note="skipped by --no-pf2")
        PR("[P-F2] SKIPPED by flag")

    # ================================================================= P-TAU: MEASURE the body
    # A step response at PHYSICS-SUBSTEP resolution on the clean world, fit for tau, on BOTH the
    # fast body and the donor body — the donor is the measurement's own control, since its tau is
    # known to be 0.5 s from `m/c` and every prior node in the arc has run on it.
    t0 = time.time()

    def measure_tau(pc_body):
        from mjc.pusher_env import PusherEnv
        env = PusherEnv(pc_body.dgp(False), with_puck=False)
        tau_x = pc_body.tau
        n = int(max(100, round(10.0 * tau_x / pc_body.timestep)))
        # The pusher is RE-SEATED at the origin before every substep. Nothing in the clean
        # world's dynamics depends on position (`a = (gear/m) u - (c/m) v`), so the velocity
        # sequence is exactly the free step response — and the donor body, which would otherwise
        # cover 20 m in the 10 tau its own fit needs, never reaches a wall.
        v = np.zeros(2)
        vs = []
        for _ in range(n):
            env.set_state(np.zeros(2), v)
            st, _ = env.step(np.array([1.0, 0.0], np.float32), 1)
            v = np.asarray(st[2:], np.float64)
            vs.append(float(v[0]))
        v = np.asarray(vs, np.float64)
        t = (np.arange(n) + 1.0) * pc_body.timestep
        vinf = float(v[-1])
        m = (v / vinf > 0.05) & (v / vinf < 0.90)
        y = np.log(1.0 - v[m] / vinf)
        A = np.stack([t[m], np.ones(m.sum())], 1)
        sl, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        tau_fit = float(-1.0 / sl)
        return dict(tau_fit=tau_fit, tau_expected=float(tau_x),
                    rel_err=float(abs(tau_fit - tau_x) / tau_x),
                    v_inf_fit=vinf, v_inf_expected=float(pc_body.v_terminal),
                    v_rel_err=float(abs(vinf - pc_body.v_terminal) / pc_body.v_terminal),
                    n_substeps=n, n_fit=int(m.sum()))

    ptau = dict(tol=0.02,
                fast=measure_tau(P.accel_piece(P.A_R_LADDER[0], cfg["tempos"][0],
                                               k_app=cfg["k_app"], mass=cfg["mass"],
                                               damping=cfg["damping"])),
                donor=measure_tau(P.donor_piece()))
    ptau["pass"] = bool(ptau["fast"]["rel_err"] <= ptau["tol"]
                        and ptau["donor"]["rel_err"] <= ptau["tol"]
                        and ptau["fast"]["v_rel_err"] <= ptau["tol"])
    ptau["tempo_table"] = []
    for H in cfg["tempos"]:
        pcH = P.accel_piece(P.A_R_LADDER[0], H, k_app=cfg["k_app"], mass=cfg["mass"],
                            damping=cfg["damping"])
        ptau["tempo_table"].append(dict(
            H=int(H), note_ms=1000.0 * H * P.DT_CTRL, tau_over_note=pcH.tau / pcH.note_s(),
            note_over_tau=pcH.note_s() / pcH.tau,
            delta_ms=1000.0 * cfg["delta_fix"] * P.DT_CTRL,
            in_band=bool(pcH.tau < pcH.note_s()
                         and pcH.note_s() < cfg["delta_fix"] * P.DT_CTRL)))
    ptau["band_note"] = ("`tau < T < Delta` is the band prestissimo could not reach: a stored "
                         "unit can execute a note blind only if tau < T, and feel fails only if "
                         "T < Delta. `in_band` per tempo says which rungs of this ladder sit "
                         "inside it at the fixed Delta.")
    out["P_TAU"] = ptau
    out["t_ptau"] = time.time() - t0
    PR(f"[P-TAU] fast body tau_fit={ptau['fast']['tau_fit'] * 1000:.2f} ms vs m/c="
       f"{ptau['fast']['tau_expected'] * 1000:.2f} ms (rel {ptau['fast']['rel_err']:.4f}); "
       f"v_inf {ptau['fast']['v_inf_fit']:.4f} vs {ptau['fast']['v_inf_expected']:.2f}; "
       f"donor tau_fit={ptau['donor']['tau_fit'] * 1000:.1f} ms vs "
       f"{ptau['donor']['tau_expected'] * 1000:.1f} ms (rel "
       f"{ptau['donor']['rel_err']:.4f})  pass={ptau['pass']}")
    for r in ptau["tempo_table"]:
        PR(f"      H={r['H']:2d}  note {r['note_ms']:5.0f} ms = {r['note_over_tau']:5.2f} tau  "
           f"Delta {r['delta_ms']:.0f} ms   tau<T<Delta: {r['in_band']}")
    save()
    if not ptau["pass"]:
        save()
        raise SystemExit("P-TAU failed: the body is not the body it was designed to be")

    # ================================================================= P-T: the R calibration
    # RULES, WRITTEN DOWN BEFORE THE GRID IS READ. All three name ONLY the incumbent or the
    # apparatus, so none can be moved by a committed arm's outcome (legato F2).
    #   feasibility   u_turn_max(H_fast) <= U_TURN_MAX. The SCHEDULE's own peak actuator demand
    #                 at the fastest tempo must leave headroom for correction; above it, the
    #                 tempo axis would be measuring the actuator's ceiling and not the
    #                 consumption of feedback. Algebra on the body's two constants, computed by
    #                 the experimenter (like the do-nothing floor and the band).
    #   competence    e_reflex(H_slow, Delta_fix, gains re-fit on the CLEAN world) <= band(R).
    #                 The incumbent must be inside the playability band at the slowest tempo, or
    #                 there is no era 0 to leave.
    #   demand        e_reflex(H_fast, Delta_fix) > band(R). At the fastest tempo the incumbent
    #                 must be out of the band, or the tempo axis has no bite (offbook d0's
    #                 measured failure: playable-for-everyone).
    #   R* = the LARGEST R passing all three — the biggest figure the incumbent can still play,
    #        which is prestissimo decision 8's rule with the tempo axis substituted for Delta.
    #   fallback: the largest R passing feasibility and competence, and say so in the record.
    t0 = time.time()
    cal_rows = []
    # Only the two GUARD tempi are gridded: every rule below names the slowest or the fastest
    # rung and nothing else, and the incumbent's own number at the intermediate tempi is
    # re-measured by the main run's per-tempo re-fit and reported there. On a five-rung ladder
    # whose slowest note is 768 ms, gridding all five here would double the node's wall clock to
    # produce numbers the run already produces.
    cal_tempos = [cfg["tempos"][0], cfg["tempos"][-1]]
    for R in cfg["r_ladder"]:
        row = dict(R=float(R), cal_tempos=list(cal_tempos))
        for H in cal_tempos:
            pc = P.accel_piece(R, H, k_app=cfg["k_app"], n_levels=cfg["n_levels"],
                               damping=cfg["damping"], mass=cfg["mass"])
            tq = pc.turn_command()
            # n_proc = 1: P-T runs `traverse` only (the incumbent and the do-nothing floor
            # never roll out), so a 16-process pool here would be 96 forks for nothing.
            Wr = World(cfg, pc, n_proc=1, rot=True)
            Wrc = World(cfg, pc, n_proc=1, rot=False)
            evr = Wr.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_cal"])
            row.update(**pc.band_parts())
            led = Ledger(Wr.dt_ctrl, Wr.d_fb)
            odn = Wr.traverse(ZeroDecider(Wr), evr, np.random.default_rng(1), led)
            D = int(cfg["delta_fix"])
            grid = []
            for kp in cfg["kp_grid"]:
                for kd in cfg["kd_grid"]:
                    Wrc.kp, Wrc.kd = float(kp), float(kd)
                    led = Ledger(Wr.dt_ctrl, Wr.d_fb)
                    oc = Wrc.traverse(ReflexDecider(), evr, np.random.default_rng(1), led,
                                      obs_delay=D)
                    grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
            b = min(grid, key=lambda r: r["e_piece"])
            Wr.kp, Wr.kd = b["kp"], b["kd"]
            led = Ledger(Wr.dt_ctrl, Wr.d_fb)
            orot = Wr.traverse(ReflexDecider(), evr, np.random.default_rng(1), led, obs_delay=D)
            row[f"H{H}"] = dict(
                speed=pc.mean_leg() / pc.note_s(), u_turn=tq["u_turn_max"],
                do_nothing=odn["e_piece"], do_nothing_listen=odn["e_listen"],
                clean=b["e_piece"], rot=orot["e_piece"], listen=orot["e_listen"],
                app=orot["e_app"], gains=[b["kp"], b["kd"]],
                edge=[bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"]))),
                      bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))])
            Wr.close(); Wrc.close()
        Hs, Hf = cal_tempos[0], cal_tempos[-1]
        row["feasible"] = bool(row[f"H{Hf}"]["u_turn"] <= P.U_TURN_MAX)
        row["competence"] = bool(row[f"H{Hs}"]["rot"] <= row["band"])
        row["presto_competence"] = bool(row[f"H{Hs}"]["rot"] <= 0.5 * row[f"H{Hs}"]["do_nothing"])
        row["demand"] = bool(row[f"H{Hf}"]["rot"] > row["band"])
        row["usable_range"] = row[f"H{Hf}"]["rot"] / max(row[f"H{Hs}"]["rot"], 1e-9)
        cal_rows.append(row)
        PR(f"[P-T] R={R:.3f} leg={row['mean_leg']:.4f} band={row['band']:.4f}  " + "  ".join(
            f"H{H}: v={row[f'H{H}']['speed']:.2f} u={row[f'H{H}']['u_turn']:.2f} "
            f"e={row[f'H{H}']['rot']:.4f} l={row[f'H{H}']['listen']:.4f} "
            f"dn={row[f'H{H}']['do_nothing']:.3f}" for H in cal_tempos)
           + f"  feas={row['feasible']} comp={row['competence']}"
           f"/presto {row['presto_competence']} demand={row['demand']} "
           f"range={row['usable_range']:.1f}x")
    ok = [r for r in cal_rows if r["feasible"] and r["competence"] and r["demand"]]
    if ok:
        best = max(ok, key=lambda r: r["R"])
        rule = "all three guards; R* = the largest figure the incumbent can still play"
    else:
        fc = [r for r in cal_rows if r["feasible"] and r["competence"]]
        fd = [r for r in cal_rows if r["feasible"] and r["demand"]]
        if fc:
            best = max(fc, key=lambda r: r["R"])
            rule = "FALLBACK: no R passed all three; largest R passing feasibility+competence"
        elif fd:
            # Competence failing at EVERY R is a statement about the delay and the body, not
            # about the figure's size: `e_reflex / band` is only weakly R-dependent and what
            # dependence it has runs the WRONG WAY for shrinking (measured on `tsmoke1`: 4.5 at
            # R = 0.14 against 3.1 at R = 0.20). Shrinking the figure would therefore make the
            # piece slower AND no more playable — prestissimo decision 8's reversal, in this
            # node's own units. The largest feasible figure is kept and era 0 is recorded as
            # absent.
            best = max(fd, key=lambda r: r["R"])
            rule = ("FALLBACK: no R passed competence at the slowest tempo; largest R passing "
                    "feasibility+demand — era 0 does not exist on this ladder, which is a "
                    "finding about the delay and the body, not a knob")
        else:
            best = min(cal_rows, key=lambda r: r["R"])
            rule = ("FALLBACK: no R passed feasibility+demand; SMALLEST R — the piece is "
                    "unplayable at every size on the ladder, which is a finding")
    R_star = float(cfg["r_force"]) if cfg["r_force"] > 0 else float(best["R"])
    out["P_T"] = dict(rows=cal_rows, rule=rule, R_star=R_star, u_turn_max=P.U_TURN_MAX,
                      forced=bool(cfg["r_force"] > 0), n_pass_all=len(ok))
    out["t_pt"] = time.time() - t0
    PR(f"[P-T] R* = {R_star:.3f}  ({rule})  [{out['t_pt']:.0f}s]")
    save()
    if cfg["cal_only"]:
        out["complete"] = True
        out["note"] = "calibration probe only (--cal-only): nothing after P-T was run"
        save()
        PR("[cal-only] stopping after P-T by flag")
        return out

    # ============================================================ per tempo: gains and library
    Ws, pcs, gains, libs, rts, evs = {}, {}, {}, {}, {}, {}
    pn_by, ps_rows = {}, []
    t_lib = 0.0
    for H in cfg["tempos"]:
        t0 = time.time()
        pc = P.accel_piece(R_star, H, k_app=cfg["k_app"], n_levels=cfg["n_levels"],
                           damping=cfg["damping"], mass=cfg["mass"])
        W = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
        Wc = World(cfg, pc, n_proc=1, rot=False)      # the gain grid never rolls out
        ev = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        rt = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])
        Ws[H], pcs[H], evs[H], rts[H] = W, pc, ev, rt
        if H == cfg["tempos"][0]:
            out["piece"] = pc.describe()
            out["piece_by_tempo"] = {}
        out["piece_by_tempo"][str(H)] = pc.describe()
        g = {}
        for D in cfg["cal_deltas"]:
            grid = []
            for kp in cfg["kp_grid"]:
                for kd in cfg["kd_grid"]:
                    Wc.kp, Wc.kd = float(kp), float(kd)
                    led = Ledger(W.dt_ctrl, W.d_fb)
                    oc = Wc.traverse(ReflexDecider(), ev, np.random.default_rng(1), led,
                                     obs_delay=D)
                    grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
            b = min(grid, key=lambda r: r["e_piece"])
            b["at_kp_edge"] = bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"])))
            b["at_kd_edge"] = bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))
            b["A_kp"] = b["kp"] * pc.accel
            b["A_kd"] = b["kd"] * pc.accel
            g[D] = b
        gains[H] = g
        Wc.close()
        # the do-nothing floor at this tempo, under BOTH grades
        led = Ledger(W.dt_ctrl, W.d_fb)
        odn = W.traverse(ZeroDecider(W), ev, np.random.default_rng(1), led)
        out.setdefault("do_nothing", {})[str(H)] = dict(
            e_piece=odn["e_piece"], e_listen=odn["e_listen"],
            in_h=float(np.mean(np.asarray(odn["ep"]) <= 0.5 * pc.mean_leg())),
            lis_h=float(np.mean(np.asarray(odn["ep_listen"]) <= 0.5 * pc.mean_leg())))
        PR(f"[tempo H={H:2d}] note {1000 * H * W.dt_ctrl:.0f} ms  speed "
           f"{pc.mean_leg() / pc.note_s():.2f} m/s  u_turn {pc.turn_command()['u_turn_max']:.2f}"
           f"  do-nothing {odn['e_piece']:.4f} / listen {odn['e_listen']:.4f}   gains " +
           " ".join(f"D{D}:({g[D]['kp']:g},{g[D]['kd']:g})e{g[D]['e_piece']:.4f}"
                    for D in cfg["cal_deltas"]))
        # the library, built ONCE at this tempo at Delta = 0 with the harvest gains frozen at
        # this tempo's Delta = 0 fit (acappella's discipline, prestissimo decision 10).
        llb = Ledger(W.dt_ctrl, W.d_fb)
        lvlib, b1r, blr = build_ladder(W, pc, cfg, cfg["seed"], (g[0]["kp"], g[0]["kd"]),
                                       llb, rt)
        libs[H] = lvlib
        out.setdefault("library", {})[str(H)] = dict(
            level1=b1r, levels=blr, sizes=lvlib.sizes(), counts=lvlib.counts(),
            build_ledger=llb.snap(), harvest_gains=[g[0]["kp"], g[0]["kd"]],
            n_slots=lvlib.layout.n_slots)
        pn = check_nesting(W, lvlib, pc, rt)
        pn_by[str(H)] = {k: v for k, v in pn.items() if k != "rows"}
        PR(f"      [P-N H={H}] {pn['n_checked']} members spelled by their parents "
           f"({pn['spell_bad']} bad); weld == halves max|delta| = {pn['exec_max_abs']:.3e}  "
           f"pass={pn['pass']}")
        t_lib += time.time() - t0
        save()
    out["P_N"] = dict(by_tempo=pn_by,
                      **{"pass": all(v["pass"] for v in pn_by.values())})
    out["t_library"] = t_lib

    # ================================================= the SCALED libraries, and P-R (identity)
    H_src = cfg["tempos"][0]                      # the SLOWEST tempo is the recording's source
    scaled = {}
    pr = dict(src_tempo=int(H_src), rows=[], max_abs=0.0)
    for p_exp in cfg["scale_exponents"]:
        for H in cfg["tempos"]:
            scaled[(p_exp, H)] = resample_library(libs[H_src], H_src, H, p=p_exp)
    for p_exp in cfg["scale_exponents"]:
        s_id = scaled[(p_exp, H_src)]
        mx = 0.0
        n = 0
        for (ell, k) in sorted(libs[H_src].layout.off):
            for a, b in zip(libs[H_src].slots(ell, k), s_id.slots(ell, k)):
                for ma, mb in zip(a["members"], b["members"]):
                    mx = max(mx, float(np.max(np.abs(ma["cmds"] - mb["cmds"]))))
                    n += 1
        pr["rows"].append(dict(p=float(p_exp), n_members=n, max_abs=mx))
        if p_exp == 0.0:
            pr["max_abs"] = mx
    pr["pass"] = bool(pr["max_abs"] == 0.0)
    out["P_R"] = pr
    PR(f"[P-R] resampler identity at s = 1 (H {H_src} -> {H_src}, p = 0): max|delta| = "
       f"{pr['max_abs']:.3e} over {pr['rows'][0]['n_members']} members  pass={pr['pass']}")
    save()

    # =========================================================== THE MAIN SWEEP: the era ladder
    t0 = time.time()
    L = int(cfg["n_levels"])
    frac_ths = dict(q=0.25, t=1.0 / 3.0, h=0.5)

    def arm_list(full=True):
        A = [("reflex", None, None, None), ("reflex_ol", "frozen", None, None)]
        if not full:
            # the Delta INSTRUMENT's arm set: the incumbent, its open-loop replay, and the two
            # ENDS of the ladder (one note against the whole figure) at both modes. That is what
            # decomposes a result into its feedback and execution halves; the full per-level
            # picture is the main sweep's job, at the fixed Delta.
            return A + [("key_L1", "key", (1,), "rec"), (f"key_L{L}", "key", (L,), "rec"),
                        ("aud_L1", "audit", (1,), "rec"), (f"aud_L{L}", "audit", (L,), "rec")]
        for ell in range(1, L + 1):
            A.append((f"key_L{ell}", "key", (ell,), "rec"))
        for ell in range(1, L + 1):
            A.append((f"aud_L{ell}", "audit", (ell,), "rec"))
        if full:
            A += [("key_all", "key", tuple(range(1, L + 1)), "rec"),
                  ("aud_all", "audit", tuple(range(1, L + 1)), "rec")]
            for p_exp in cfg["scale_exponents"]:
                tagp = f"s{int(p_exp)}"
                for ell in range(1, L + 1):
                    A.append((f"key_{tagp}_L{ell}", "key", (ell,), p_exp))
                for ell in range(1, L + 1):
                    A.append((f"aud_{tagp}_L{ell}", "audit", (ell,), p_exp))
        return A

    def run_cell(H, D, appm, arms):
        W, pc, ev, band = Ws[H], pcs[H], evs[H], pcs[H].band()
        ml = pc.mean_leg()
        Dapp = APP_MODES[appm]
        W.kp_app, W.kd_app = ((gains[H][0]["kp"], gains[H][0]["kd"])
                              if APP_D0_GAINS[appm] else (None, None))
        W.kp, W.kd = gains[H][D]["kp"], gains[H][D]["kd"]
        rows = []
        for nm, mode, lv, srcv in arms:
            led = Ledger(W.dt_ctrl, W.d_fb)
            if mode is None:
                dec = ReflexDecider()
            elif mode == "frozen":
                dec = FrozenReflexDecider(W)
            else:
                lb = libs[H] if srcv == "rec" else scaled[(srcv, H)]
                dec = LadderDecider(W, lb, mode=mode, levels=lv, obs_delay=D)
            o = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                           obs_delay=D, obs_delay_app=Dapp)
            ep = np.asarray(o["ep"], np.float64)
            el = np.asarray(o["ep_listen"], np.float64)
            row = dict(H=int(H), delta=int(D), arm=nm, mode=mode, app=appm,
                       source=("rec" if srcv in (None, "rec") else f"scl{int(srcv)}"),
                       levels=(None if lv is None else list(lv)),
                       e_piece=o["e_piece"], e_listen=o["e_listen"], e_app=o["e_app"],
                       e_seg=[float(x) for x in o["e_seg"]],
                       ep_med=float(np.median(ep)), lis_med=float(np.median(el)),
                       in_band=float(np.mean(ep <= band)),
                       lis_band=float(np.mean(el <= band)),
                       **{f"in_{kk}": float(np.mean(ep <= v * ml)) for kk, v in frac_ths.items()},
                       **{f"lis_{kk}": float(np.mean(el <= v * ml))
                          for kk, v in frac_ths.items()},
                       ledger=led.snap(), ledger_drilled=o["ledger_drilled"])
            if mode is not None and dec.picks:
                row["n_aud"] = float(np.mean([p["n_aud"] for p in dec.picks]))
                row["n_dec"] = len(dec.picks)
                row["mean_level"] = float(np.mean([p["mean_level"] for p in dec.picks]))
                row["n_distinct_slot"] = len({x for p in dec.picks for x in p["slot"]})
            rows.append(row)
        return rows

    sweep, hand = [], []
    for appm in cfg["app_modes"]:
        for H in cfg["tempos"]:
            W, pc, ev = Ws[H], pcs[H], evs[H]
            D = int(cfg["delta_fix"])
            Dapp = APP_MODES[appm]
            W.kp_app, W.kd_app = ((gains[H][0]["kp"], gains[H][0]["kd"])
                                  if APP_D0_GAINS[appm] else (None, None))
            W.kp, W.kd = gains[H][D]["kp"], gains[H][D]["kd"]
            led = Ledger(W.dt_ctrl, W.d_fb)
            oh = W.traverse(ReflexDecider(), ev, np.random.default_rng(cfg["seed"] + 35), led,
                            obs_delay=D, obs_delay_app=Dapp, collect_obs=True)
            k0, Hh = W.k0, W.H
            st0 = oh["seam"][k0]
            rd0 = oh["obs"][:, k0 * Hh, :]
            vnom = (W.wps[k0 + 1] - W.wps[k0]) / (Hh * W.dt_ctrl)
            hand.append(dict(
                app=appm, H=int(H), delta=int(D),
                ho_wp=float(np.mean(np.linalg.norm(st0[:, :2] - W.wps[k0][None, :], axis=1))),
                ho_legs=float(np.mean(np.linalg.norm(st0[:, :2] - W.wps[k0][None, :], axis=1))
                              / pc.mean_leg()),
                ho_speed=float(np.mean(np.linalg.norm(st0[:, 2:], axis=1))),
                v_sched=float(np.linalg.norm(vnom)),
                ho_v_err=float(np.mean(np.linalg.norm(st0[:, 2:] - vnom[None, :], axis=1))),
                read_pos_err=float(np.mean(np.linalg.norm(rd0[:, :2] - st0[:, :2], axis=1))),
                read_vel_err=float(np.mean(np.linalg.norm(rd0[:, 2:] - st0[:, 2:], axis=1))),
                clamped=bool(k0 * Hh < D), read_step=int(max(0, k0 * Hh - D)),
                lead_steps=int(k0 * Hh), e_app=oh["e_app"]))
            rows = run_cell(H, D, appm, arm_list(full=True))
            sweep += rows
            ref = [r for r in rows if r["arm"] == "reflex"][0]
            best = min((r for r in rows if r["arm"] not in ("reflex", "reflex_ol")),
                       key=lambda r: r["e_piece"])
            PR(f"[sweep/{appm}] H={H:2d} D={D}  reflex={ref['e_piece']:.4f}"
               f"/{ref['e_listen']:.4f} (fb {ref['ledger_drilled']['n_fb']:.0f})   " + "  ".join(
                   f"{r['arm']}={r['e_piece']:.4f}" for r in rows
                   if r["arm"].startswith(("key_L", "aud_L")))
               + f"   best={best['arm']} {best['e_piece']:.4f}  band={pc.band():.4f}")
            save()
    out["sweep"] = sweep
    out["handover"] = hand
    out["t_sweep"] = time.time() - t0

    PR("[hand-over] the state the drilled figure is launched from, per tempo and mode:")
    for r in hand:
        PR(f"      app={r['app']:>5s} H={r['H']:2d}  |x-V0|={r['ho_wp']:.4f} "
           f"({r['ho_legs']:.2f} legs)  |v|={r['ho_speed']:.3f} vs sched {r['v_sched']:.3f}  "
           f"read err pos={r['read_pos_err']:.4f} vel={r['read_vel_err']:.3f}  "
           f"read@{r['read_step']}/{r['lead_steps']}"
           f"{'  CLAMPED-AT-RESET' if r['clamped'] else ''}  e_app={r['e_app']:.4f}")

    # ------------------------------------------------- the era table, per grade, per app mode
    def cellf(app, H, nm):
        for r in sweep:
            if r["H"] == H and r["arm"] == nm and r["app"] == app and r["delta"] == cfg[
                    "delta_fix"]:
                return r
        return None

    eras_by, ladder_by = {}, {}
    for appm in cfg["app_modes"]:
        rows_t = []
        for H in cfg["tempos"]:
            band = pcs[H].band()
            rf = cellf(appm, H, "reflex")
            ol = cellf(appm, H, "reflex_ol")
            row = dict(H=int(H), note_ms=round(1000.0 * H * P.DT_CTRL), band=band, app=appm,
                       reflex=rf["e_piece"], reflex_listen=rf["e_listen"],
                       reflex_ol=ol["e_piece"], reflex_ol_listen=ol["e_listen"],
                       reflex_ol_ratio=ol["e_piece"] / max(rf["e_piece"], 1e-9),
                       reflex_fb=rf["ledger_drilled"]["n_fb"],
                       **{f"reflex_{k2}": rf[k2] for k2 in ("in_h", "in_t", "in_q",
                                                            "lis_h", "lis_t", "lis_q")})
            for pre in ("key", "aud"):
                for srcn, pfx in [("rec", ""), *[(f"scl{int(p)}", f"s{int(p)}_")
                                                 for p in cfg["scale_exponents"]]]:
                    for ell in range(1, L + 1):
                        c = cellf(appm, H, f"{pre}_{pfx}L{ell}")
                        if c is None:
                            continue
                        row[f"{pre}_{srcn}_L{ell}"] = c["e_piece"]
                        row[f"{pre}_{srcn}_L{ell}_lis"] = c["e_listen"]
                        row[f"{pre}_{srcn}_L{ell}_in"] = c["in_band"]
                        row[f"{pre}_{srcn}_L{ell}_lisin"] = c["lis_band"]
                        row[f"{pre}_{srcn}_L{ell}_fb"] = c["ledger_drilled"]["n_fb"]
                for ell in range(1, L):
                    row[f"{pre}_m{ell}{ell + 1}"] = (row[f"{pre}_rec_L{ell}"]
                                                     - row[f"{pre}_rec_L{ell + 1}"])
                row[f"{pre}_m_ref"] = (row["reflex"]
                                       - min(row[f"{pre}_rec_L{e}"] for e in range(1, L + 1)))
                row[f"{pre}_best"] = min(range(1, L + 1),
                                         key=lambda e: row[f"{pre}_rec_L{e}"])
            row["key_all"] = cellf(appm, H, "key_all")["e_piece"]
            row["aud_all"] = cellf(appm, H, "aud_all")["e_piece"]
            rows_t.append(row)
        eras = {}
        for grade, suf in (("piece", ""), ("listen", "_lis")):
            for pre in ("key", "aud"):
                for srcn in ["rec"] + [f"scl{int(p)}" for p in cfg["scale_exponents"]]:
                    for ell in range(1, L):
                        hit = None
                        for row in rows_t:
                            a = row.get(f"{pre}_{srcn}_L{ell}{suf}")
                            b = row.get(f"{pre}_{srcn}_L{ell + 1}{suf}")
                            if a is None or b is None:
                                continue
                            if a > row["band"] and b <= row["band"]:
                                hit = int(row["H"])
                                break
                        eras[f"{grade}_{pre}_{srcn}_era_{ell + 1}"] = hit
                    hit = None
                    for row in rows_t:
                        rr = row["reflex"] if suf == "" else row["reflex_listen"]
                        a = row.get(f"{pre}_{srcn}_L1{suf}")
                        if a is None:
                            continue
                        if rr > row["band"] and a <= row["band"]:
                            hit = int(row["H"])
                            break
                    eras[f"{grade}_{pre}_{srcn}_era_1"] = hit
            hit = None
            for row in rows_t:
                rr = row["reflex"] if suf == "" else row["reflex_listen"]
                if rr > row["band"]:
                    hit = int(row["H"])
                    break
            eras[f"{grade}_reflex_fails"] = hit
        ladder_by[appm], eras_by[appm] = rows_t, eras
        PR(f"[era/{appm}] the FASTEST-first tempo at which level l fails the band and l+1 does "
           f"not (H, so smaller = faster): "
           + json.dumps({k: v for k, v in eras.items() if v is not None}))
        PR(f"          (null: {sorted(k for k, v in eras.items() if v is None)})")
    out["ladder_by_mode"] = ladder_by
    out["eras_by_mode"] = eras_by
    out["ladder"] = ladder_by[cfg["app_modes"][-1]]        # `fixed` is the arm of record
    out["eras"] = eras_by[cfg["app_modes"][-1]]
    out["reflex_is_tape"] = {
        appm: {str(r["H"]): dict(ratio=r["reflex_ol_ratio"], tape=bool(r["reflex_ol_ratio"]
                                                                      <= 1.10))
               for r in ladder_by[appm]} for appm in cfg["app_modes"]}
    save()

    # ================================================== the Delta INSTRUMENT sweep, per tempo
    t0 = time.time()
    instr, instr_hand = [], []
    for appm in cfg["app_modes"]:
        for H in cfg["tempos"]:
            W, pc, ev = Ws[H], pcs[H], evs[H]
            for D in cfg["deltas"]:
                Dapp = APP_MODES[appm]
                W.kp_app, W.kd_app = ((gains[H][0]["kp"], gains[H][0]["kd"])
                                      if APP_D0_GAINS[appm] else (None, None))
                W.kp, W.kd = gains[H][D]["kp"], gains[H][D]["kd"]
                led = Ledger(W.dt_ctrl, W.d_fb)
                oh = W.traverse(ReflexDecider(), ev, np.random.default_rng(cfg["seed"] + 35),
                                led, obs_delay=D, obs_delay_app=Dapp, collect_obs=True)
                k0, Hh = W.k0, W.H
                st0, rd0 = oh["seam"][k0], oh["obs"][:, k0 * Hh, :]
                instr_hand.append(dict(
                    app=appm, H=int(H), delta=int(D),
                    ho_wp=float(np.mean(np.linalg.norm(st0[:, :2] - W.wps[k0][None, :], axis=1))),
                    ho_legs=float(np.mean(np.linalg.norm(st0[:, :2] - W.wps[k0][None, :],
                                                         axis=1)) / pc.mean_leg()),
                    ho_speed=float(np.mean(np.linalg.norm(st0[:, 2:], axis=1))),
                    read_pos_err=float(np.mean(np.linalg.norm(rd0[:, :2] - st0[:, :2], axis=1))),
                    read_vel_err=float(np.mean(np.linalg.norm(rd0[:, 2:] - st0[:, 2:], axis=1))),
                    clamped=bool(k0 * Hh < D), read_step=int(max(0, k0 * Hh - D)),
                    lead_steps=int(k0 * Hh), e_app=oh["e_app"]))
                instr += run_cell(H, D, appm, arm_list(full=False))
            rr = [r for r in instr if r["H"] == H and r["app"] == appm]

            def g(D, nm):
                return [x for x in rr if x["delta"] == D and x["arm"] == nm][0]["e_piece"]
            PR(f"[instr/{appm}] H={H:2d}  " + "  ".join(
                f"D{D}: rf={g(D, 'reflex'):.4f} kL1={g(D, 'key_L1'):.4f} "
                f"kL{L}={g(D, f'key_L{L}'):.4f} aL{L}={g(D, f'aud_L{L}'):.4f}"
                for D in cfg["deltas"]))
            save()
    out["instrument"] = instr
    out["instrument_handover"] = instr_hand
    out["t_instr"] = time.time() - t0

    # ============================================ P-A: the approach removed acappella finding 5
    # Measured on `app = arm`, as prestissimo's P-A was: under `fixed` the lead-in is one
    # pre-computed plan, so the deepest KEY arm makes one decision from a hand-over that does not
    # move with Delta and is Delta-invariant BY CONSTRUCTION whenever its slot choice is stable —
    # which is a property of the instrument, not the artifact acappella finding 5 named. Both
    # modes are reported at every level.
    pa = {"mode": "arm"}
    for appm in cfg["app_modes"]:
        for H in cfg["tempos"]:
            for pre in ("key", "aud"):
                for ell in (1, L):
                    v = [r["e_piece"] for D in cfg["deltas"] for r in instr
                         if r["H"] == H and r["delta"] == D and r["arm"] == f"{pre}_L{ell}"
                         and r["app"] == appm]
                    pa[f"{appm}_H{H}_{pre}_L{ell}"] = dict(
                        spread=float(max(v) - min(v)),
                        ratio=float(max(v) / max(min(v), 1e-9)),
                        vals=[float(x) for x in v])
    pa["read_grows"] = bool(all(
        r["read_pos_err"] > 0.0 for r in instr_hand if r["delta"] > 0))
    pa["never_clamped"] = bool(not any(r["clamped"] for r in instr_hand))
    pa["pass"] = bool(all(pa[f"arm_H{H}_key_L{L}"]["spread"] > 0.0 for H in cfg["tempos"])
                      and pa["read_grows"] and pa["never_clamped"])
    pa["note"] = ("acappella finding 5: on etude's square the chain arms decide once at seam 0 "
                  "from rest, so their read is the TRUE state at every Delta and they are "
                  "exactly delay-invariant. With k_app approach notes the deepest arm's spread "
                  "over the Delta ladder is this node's measurement that the artifact is gone, "
                  "at every tempo. `k0*H > Delta_fix` at every tempo by construction "
                  f"(k_app = {cfg['k_app']}), so the drilled-seam-0 read is never clamped at "
                  "the reset state — the aliasing prestissimo located at k0*H = 15 < 16.")
    out["P_A"] = pa
    for appm in cfg["app_modes"]:
        PR(f"[P-A/{appm}] deepest committed arm (key_L{L}) spread over the Delta ladder: " +
           "  ".join(f"H{H}={pa[f'{appm}_H{H}_key_L{L}']['spread']:.4f} "
                     f"({pa[f'{appm}_H{H}_key_L{L}']['ratio']:.2f}x)" for H in cfg["tempos"]))
    PR(f"[P-A] read at drilled seam 0 stale at every Delta>0: {pa['read_grows']}; never clamped "
       f"at the reset state: {pa['never_clamped']}  pass={pa['pass']}")
    save()

    # ============================================ P-S: seam information, per level per tempo
    t0 = time.time()
    for H in cfg["tempos"]:
        W, pc, ev = Ws[H], pcs[H], evs[H]
        K, k0 = W.K_drill, W.k0
        for D in cfg["ps_deltas"]:
            W.kp_app, W.kd_app = gains[H][0]["kp"], gains[H][0]["kd"]
            W.kp, W.kd = gains[H][D]["kp"], gains[H][D]["kd"]
            led = Ledger(W.dt_ctrl, W.d_fb)
            dec = LadderDecider(W, libs[H], mode="key", levels=(1,), obs_delay=D)
            okey = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                              obs_delay=D, obs_delay_app=0, collect_obs=True)
            for ell in range(1, L + 1):
                spn = LadderLayout.span(ell)
                for d in range(0, K, spn):
                    S = libs[H].slots(ell, d, with_poison=False)
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
                    ps_rows.append(dict(H=int(H), delta=int(D), level=int(ell), drilled=int(d),
                                        best_fixed_slot=fixed, per_state_oracle=oracle,
                                        gain=fixed / max(oracle, 1e-9),
                                        n_distinct_argmin=int(len(np.unique(sc.argmin(1)))),
                                        n_slot=len(S)))
    out["P_S"] = ps_rows
    out["t_ps"] = time.time() - t0
    PR("[P-S] seam information under the delayed read (per-state oracle over each level's cell):")
    for H in cfg["tempos"]:
        for D in cfg["ps_deltas"]:
            for ell in range(1, L + 1):
                rr = [r for r in ps_rows if r["H"] == H and r["delta"] == D
                      and r["level"] == ell]
                if not rr:
                    continue
                PR(f"      H={H:2d} D={D:2d} L{ell}  mean gain "
                   f"{sum(r['gain'] for r in rr) / len(rr):.3f}x  "
                   f"(range {min(r['gain'] for r in rr):.2f}-{max(r['gain'] for r in rr):.2f})  "
                   f"argmins {sum(r['n_distinct_argmin'] for r in rr) / len(rr):.1f}/"
                   f"{rr[0]['n_slot']}  oracle "
                   f"{sum(r['per_state_oracle'] for r in rr) / len(rr):.4f}")

    out["flags"] = dict(
        gain_grid_edges={str(H): {str(D): dict(kp=gains[H][D]["at_kp_edge"],
                                               kd=gains[H][D]["at_kd_edge"])
                                  for D in cfg["cal_deltas"]} for H in cfg["tempos"]},
        library_per_tempo=("`rec` is a library HARVESTED AT ITS OWN TEMPO, built once at "
                           "Delta = 0 with the harvest gains frozen at that tempo's Delta = 0 "
                           "fit. `scl0` / `scl2` are the SLOWEST tempo's library resampled, and "
                           "carry the slow tempo's keys and construction scores, which are not "
                           "valid at the destination tempo — `key` mode therefore selects on "
                           "stale keys and `audit` mode on the plant, and both are reported."),
        practice_sigma=("The donor's command noise 0.25 is rescaled to 0.0628 so the two bodies "
                        "have the SAME realised velocity spread during practice "
                        "(sd(v) = sigma * v_terminal * sqrt((1-a)/(1+a))); the rule involves the "
                        "two bodies only, no piece and no tempo, and returns exactly 0.25 on the "
                        "donor body, which is what keeps P-F1 and P-F2 exact."),
        two_grades=("Both are reported at every cell and neither is privileged. The per-waypoint "
                    "grade rewards arms that resolve each waypoint and is the one every prior "
                    "number in this arc is in; the listener's-clock grade low-passes both "
                    "signals at 384 ms, so it forgives within-window wiggle and timing slop and "
                    "should favour whole-figure units at fast tempi. Where they disagree, the "
                    "disagreement is the readout."),
        no_mining=("Phase 1 builds every rung STATICALLY, by construction audition. Nothing is "
                   "mined from the learner's own routed successes and there is no at_support "
                   "here: that is Phase 3."),
        delta_is_an_instrument=("Delta is FIXED at the niche value for every headline number. "
                                "The Delta ladder is run per tempo, under BOTH approach modes, "
                                "on the incumbent plus the two ENDS of the level ladder, purely "
                                "as the instrument that decomposes a result into its feedback "
                                "and execution halves."))
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    for H in cfg["tempos"]:
        Ws[H].close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


@app.local_entrypoint()
def accelerando_tempo(
    quick: bool = False,
    spawn: bool = False,
    cal_only: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    no_pf1: bool = False,
    no_pf2: bool = False,
    r_force: float = 0.0,
    k_app: int = P.A_K_APP,
    n_levels: int = P.N_LEVELS,
    mass: float = P.FAST_MASS,
    damping: float = P.DAMPING,
    delta_fix: int = P.DELTA_FIX,
    tempos: str = ",".join(str(x) for x in P.TEMPO_LADDER),
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
    app_modes: str = "arm,fixed",
    ps_deltas: str = "0,5",
):
    tp = [int(x) for x in tempos.split(",") if x]
    dl = sorted({int(delta_fix), *DELTAS_INSTR})
    cfg = dict(tag=tag or "t1", seed=seed, n_proc=n_proc, quick=bool(quick),
               do_pf1=not no_pf1, do_pf2=not no_pf2, r_force=float(r_force),
               cal_only=bool(cal_only),
               k_app=k_app, n_levels=n_levels, mass=float(mass), damping=float(damping),
               delta_fix=int(delta_fix), tempos=tp,
               r_ladder=list(P.A_R_LADDER),
               deltas=[int(x) for x in DELTAS_INSTR],
               cal_deltas=dl,
               scale_exponents=[0.0, 2.0], w_listen=P.W_LISTEN,
               app_modes=[x for x in app_modes.split(",") if x],
               ps_deltas=[int(x) for x in ps_deltas.split(",") if x],
               **dict(zip(("kp_grid", "kd_grid"),
                          [list(x) for x in P.gain_grid(T_KP_GRID, T_KD_GRID, mass)])),
               n_eval=n_eval, n_cal=n_cal, n_rt=n_rt, batch=batch, n_warm=n_warm,
               n_score=n_score, n_slot=n_slot, m_cand=m_cand, m_member=m_member,
               m_weld=m_weld, n_poison=n_poison,
               # inert here (the fork carries solo's search machinery unused in Phase 1)
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5)
    if quick:
        cfg.update(tag=tag or "tsmoke", n_proc=8, n_eval=8, n_cal=6, n_rt=8, batch=6, n_warm=2,
                   n_score=4, n_slot=3, m_cand=8, m_member=2, m_weld=2, n_poison=2,
                   r_ladder=[0.14, 0.20], tempos=[16, 2],
                   deltas=[0, 5], cal_deltas=[0, 5], ps_deltas=[0, 5],
                   **dict(zip(("kp_grid", "kd_grid"),
                              [list(x) for x in P.gain_grid((2.0, 10.0, 40.0),
                                                            (0.25, 1.0, 4.0), mass)])))
    if spawn:
        c = run_tempo.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_accelerando/{cfg['tag']}/tempo.json")
        return
    r = run_tempo.remote(cfg)
    print("\n".join(r["log"][-120:]))
