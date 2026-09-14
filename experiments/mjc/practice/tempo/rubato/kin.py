"""rubato Phase 1 — the CEILING of the factored program: a kinematic unit through a perfect
cerebellum.

THE QUESTION. `accelerando/` measured, on this body at this delay, that a posture-conditioned
head trained on the posture it can see matches or beats the verbatim tape at every practiced
tempo (L4 0.0056 vs tape 0.0069 at 48 ms notes, against a re-fit reflex at 0.0690) and transfers
to NO unpracticed tempo at any step the body can express (0/6 slots at L3-L4 even at 1.14x). The
head was asked to emit FORCES. Forces are not tempo-invariant on any body: the drag part scales
with speed and the inertial part with speed squared, in a ratio set by the body's time constant,
and `accelerando/t1` measured both naive scalings failing in opposite directions. What IS
tempo-invariant is the PATH.

    does a unit stored as a PATH — position and velocity against phase, with the clock outside
    the program — play in band at a tempo it was not cut at, when an executor that knows the
    body converts it to commands?

Phase 1 asks it with a PERFECT cerebellum: the exact inverse dynamics of the free-flight body,
from its true constants. That is an EXPERIMENTER-SIDE ORACLE, the way RHM uses exact DP, and it
is the ceiling of the architecture. If the kinematic unit does not transfer through it, the
factoring is not the answer on this body and that is the finding. Phase 2 replaces the oracle
with a model fitted on slow-tempo data; Phase 3 adds online correction.

WHAT THIS NODE LIFTS, AND THE ONE WAY IT LIFTS IT. `solo`/`prestissimo`/`accelerando` forbade a
forward model anywhere. Here a BODY MODEL (inverse and forward) may sit in the EXECUTOR, on the
LEARNER's side of the meter. It may NOT be handed to the incumbent as a planner — that is the
`accompanist` confound that invalidated `offbook`. The reflex stays model-free and re-fit per
tempo. `reflex_ec` — the incumbent given the same body model through the delay (`accompanist`
d3b's operator) — is ONE REPORTED REFERENCE ROW and is never headlined.

THE CONTENT SOURCES IN PHASE 1, all in every table
  `rec`     a recording harvested AT THE ASKED TEMPO. The oracle content: what practice there
            would have bought. Not deployable at an unpracticed tempo.
  `scl0`    the SLOWEST tempo's recording, time-resampled note-wise. "Play the tape faster."
  `scl2`    the same with amplitude x (H_src/H_dst)^2, the inertial dimensional correction.
  `kzs`     THE HEADLINE: the SLOWEST tempo's units stored as PATHS, converted by the exact ZOH
            inverse at the asked tempo. A program with no tempo in it.
  `kzo`     the same conversion applied to the ASKED tempo's OWN paths (s = 1). THE CONVERSION-
            COST CONTROL: it isolates "inverse dynamics costs X" from "tempo transfer costs Y".
  `kcs`,
  `kco`     the same two under the CONTINUOUS-TIME inverse `u = (m a + c v)/gear`, which on this
            body differs from the exact one by 1.4528x on the acceleration term alone.

THE GATES (all pre-fixed, all reported whether or not they pass)
  P-F0   the 13 donor constants, `ast`-read out of `etude/etude.py`, never imported.
  P-T1   with every new arm off, this fork reproduces `accelerando/t1`'s library-construction
         scores and its pinned sweep rows at max|delta| = 0. The fork-fidelity gate; `t1`'s own
         P-F1 (vs `acappella/b1`) and P-F2 (vs `prestissimo/a1g`) are inherited through it.
  P-G    the pinned per-(tempo, Delta) gain table is not taken on trust: one cell is re-fit.
  P-TAU  tau measured from a step response on both bodies.
  P-ZOH  the DISCRETISATION the oracle inverts is the one the plant runs: alpha measured from
         the plant's own control-step step response against exp(-dt/tau), and the one-step
         forward residual of `fwd_zoh` on the clean and the rotated world.
  P-R    the command resampler is the identity at s = 1 (accelerando's, inherited).
  P-N    the nesting op is exact, at every tempo.
  P-K    THE ORACLE IDENTITY: a member's kinematic unit, converted back at its OWN tempo,
         reproduces its own commands. Gated OUTSIDE the command-rotation region (where the body
         model is the world model); the in-region residual is REPORTED, since the rotation is
         exactly what a body model does not contain and `acappella` chose to leave unmodelled.
  P-KN   the phase resampler commutes with splitting a span at its midpoint, at max|delta| = 0.

Run:
    cd experiments/                        # MODAL_PROFILE=chromatic
    modal run mjc/practice/tempo/rubato/kin.py::rubato_kin --quick --tag ksmoke0
    modal run --detach mjc/practice/tempo/rubato/kin.py::rubato_kin --spawn --tag k1
    python3 mjc/practice/tempo/rubato/analyze_kin.py --tag k1 --fetch --figures
"""

import json
import math
import os
import time

import modal  # noqa: F401  (the app is imported from mjc.shared; kept for parity with siblings)

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.tempo.rubato import piece as P
from mjc.practice.tempo.rubato import t1_pins as T1

# accelerando's treatment gain grid, carried verbatim. It is only ever USED here for the P-G
# re-fit cell and for tempi `t1` never ran; the table itself is pinned.
T_KP_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
T_KD_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
APP_MODES = {"arm": None, "fixed": 0}
APP_D0_GAINS = {"arm": False, "fixed": True}

# THE KINEMATIC ARM SETS. (tag, inverse scheme, source, low-pass). `s` = the SLOWEST tempo's
# paths (the transfer question); `o` = this tempo's OWN paths (the conversion-cost control,
# s = 1). The low-pass is a boxcar on the SOURCE grid: `aa` = one destination control step (the
# anti-alias width the decimation itself requires — an apparatus check on the resampler, not a
# treatment) and `note` = one note (decision 20's treatment).
KIN_SETS = {
    # the Phase 1 ceiling set: two inverse schemes x two sources
    "ceiling": (("kzs", "zoh", "slow", None), ("kzo", "zoh", "own", None),
                ("kcs", "ct", "slow", None), ("kco", "ct", "own", None)),
    # the DIAGNOSTIC set: the exact scheme only, plus the two filtered arms and their control.
    # `ct` is dropped because `ksmoke1` measured neither scheme to dominate (kco/kzo 0.59-2.29x),
    # so it buys nothing here and costs 16 arms.
    "diag": (("kzs", "zoh", "slow", None), ("kzo", "zoh", "own", None),
             ("kzsa", "zoh", "slow", "aa"), ("kzsf", "zoh", "slow", "note"),
             ("kzof", "zoh", "own", "note")),
    # PHASE 2. `kzsa` is the CEILING (decision 22: the anti-aliased oracle is the arm of record),
    # `kzs` is kept as the aliased control so the retraction stays legible, and the two fitted
    # families sit on exactly `kzsa`'s resampler so the only thing that differs between the
    # ceiling and a learned executor is where the inverse came from.
    # PHASE 1's RUN OF RECORD. `kzsa` is the arm of record (decision 22); `kzs` is the aliased
    # control that keeps the retraction legible; `kzo` is the conversion-cost control at s = 1;
    # `kzof` is the note-period filter's own cost control. `scl0` rides along as a source.
    "record": (("kzsa", "zoh", "slow", "aa"), ("kzs", "zoh", "slow", None),
               ("kzo", "zoh", "own", None), ("kzof", "zoh", "own", "note")),
    "learn": (("kzsa", "zoh", "slow", "aa"), ("kzs", "zoh", "slow", None),
              ("kzoa", "zoh", "own", "aa"),
              ("kfl", "fit:lin", "slow", "aa"), ("kfm", "fit:mlp", "slow", "aa")),
    # PHASE 3. The open-loop rows the corrected arms are read against, plus the clean-world fit
    # that discriminates `klsmoke3`'s unresolved observation.
    "corr": (("kzsa", "zoh", "slow", "aa"), ("kzoa", "zoh", "own", "aa"),
             ("kfl", "fit:lin", "slow", "aa"), ("kflc", "fit:linc", "slow", "aa")),
}
KIN_ARMS = KIN_SETS["ceiling"]

# P-K's tolerance on the out-of-region command residual, FIXED BEFORE THE GATE WAS READ.
#   1e-3 of full scale. The reason: `accelerando`'s P-E measured a 5% command error already
#   spending the whole band at the deepest rung, so a residual 50x below the smallest command
#   error that node could see in execution cannot move any number this node reports. The
#   physical floor it is being held against is RK4 at 0.002 s on a 0.03 s time constant
#   (relative local error ~1e-7 per substep) plus one float32 round-trip on the launch state.
PK_TOL = 1e-3
PK_REGION_W = 1e-4      # the env's OWN Gaussian-gate cutoff in `_apply_rot_regions`


@app.function(cpu=16.0, gpu="L4", memory=49152, timeout=28800, volumes={DATA_DIR: volume})
def run_kin(cfg: dict) -> dict:
    import ast

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np

    # The MLP family is the only thing here that wants a GPU, and it wants one for seconds. The
    # L4 idles through the plant work; that is the cheaper mistake (accelerando decision 26c).
    device = "cpu"
    if cfg.get("fit_tempos"):
        try:
            import torch
            torch.set_num_threads(1)
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    from mjc.practice.tempo.rubato.world import (World, Ledger, ReflexDecider, LevelLibrary,
                                           LadderLayout, LadderDecider, build_level1_cell,
                                           build_level_cell, FrozenReflexDecider,
                                           resample_library, resample_path, kin_cmds,
                                           kin_library, measure_paths, body_consts, fwd_zoh,
                                           lp_path, lp_width, shape_distance, launch_rows,
                                           kin_targets, CorrectedDecider)
    from mjc.practice.tempo.rubato import nets as N

    class ZeroDecider:
        """The do-nothing floor: zero command everywhere. accelerando's, verbatim."""

        def __init__(self, W):
            self.H = int(W.H)

        def __call__(self, k, states, need, rng, led):
            return dict(kind="open", cmds=np.zeros((len(states), self.H, 2), np.float32),
                        span=np.ones(len(states), int), src="zero")

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_rubato", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "phase": "1",
           "contract": dict(
               what_is_lifted=(
                   "A BODY MODEL (inverse and forward) sits in the EXECUTOR, on the LEARNER's "
                   "side of the meter, converting a kinematic program to commands. It is NEVER "
                   "handed to the incumbent as a planner (the `accompanist` confound). The "
                   "reflex law is model-free and re-fit per tempo, exactly as in accelerando. "
                   "`reflex_ec` — the incumbent given the same model through the delay — is one "
                   "REPORTED reference row and is never headlined."),
               oracle=("Phase 1's executor is an EXPERIMENTER-SIDE ORACLE: the exact inverse of "
                       "the zero-order-hold discretisation of the free-flight plant, from the "
                       "body's true constants, the way RHM uses exact DP. Nothing is learned "
                       "anywhere in this phase. It gives the CEILING of the architecture with a "
                       "perfect cerebellum; Phase 2 replaces it with a model fitted on the "
                       "learner's own slow-tempo data."),
               kinematic_unit=(
                   "A committed unit's content is the REALISED PATH of its own tape — position "
                   "and velocity at every control step, executed on the plant from the launch "
                   "the tape was cut at (measured content, `legato` F4: the body produced it, so "
                   "it carries no composition error). Expressed against phase phi in [0,1] over "
                   "the span, it has NO TEMPO IN IT: at a target tempo the same curve is sampled "
                   "on the destination step grid and its velocities scale by s = H_src/H_dst."),
               derivative_scheme=(
                   "The stored object is (x, v) as the body realised them, so velocity is "
                   "MEASURED, never differenced from an interpolated position. The only "
                   "numerical operation is linear interpolation in phase. The acceleration the "
                   "executor consumes is the destination grid's own one-step velocity "
                   "difference — the quantity the ZOH inverse is exact for. Two schemes are "
                   "reported at every cell: `zoh` (exact for the discretisation the plant runs; "
                   "the scheme of record) and `ct` (the continuous-time `u = (m a + c v)/gear`), "
                   "which differ ONLY in the coefficient on acceleration and by 1.4528x on this "
                   "body, because dt_ctrl/tau = 0.8 is not small."),
               grades=("accelerando's two, unchanged and both reported everywhere: the "
                       "per-waypoint drilled error with the 1/2-leg band, and the listener's-"
                       "clock error at a fixed 384 ms window. Pass fractions at 1/4, 1/3, 1/2 "
                       "mean leg for both."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "kin.json"), "w") as fh:
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

    # ===================================================================================== #
    # THE STATIC LADDER BUILD — `accelerando/tempo.py`'s block, copied verbatim in operation.
    # Gate P-T1 asserts its output against `t1`'s recorded construction scores, so it cannot
    # silently drift.
    # ===================================================================================== #
    def build_ladder(W, pc, kc, seed, harvest_gains, llb, rt):
        K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
        W.kp, W.kd = float(harvest_gains[0]), float(harvest_gains[1])
        lvlib = LevelLibrary(K, kc["n_slot"], L, poison=True)
        b1_rec, bl_rec = [], []

        def prefix_dec(level_map, mode="key"):
            return LadderDecider(W, lvlib, mode=mode, level_map=level_map, fallback=False)

        train_pool = {}

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
            # PHASE 2's TRAINING SET, taken from the harvest that already happened: the reflex
            # law's own noisy renditions of this segment. Stored as (s0, cmds) only; the
            # velocities come from replaying them on the plant (`nets.triples_from_pool`), which
            # is a charged grounding and bit-identical to the harvest itself.
            pk = pools[k0 + d]
            train_pool[int(d)] = dict(s0=np.asarray(pk["s0"], np.float32),
                                      cmds=np.asarray(pk["cmds"], np.float32))
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
        return lvlib, b1_rec, bl_rec, train_pool

    def check_nesting(W, lvlib, pc, rt):
        """P-N, accelerando's, verbatim."""
        K, H, k0, L = W.K_drill, W.H, W.k0, pc.n_levels
        pn = dict(spell_bad=0, exec_max_abs=0.0, n_checked=0)
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
                pn["exec_max_abs"] = max(pn["exec_max_abs"],
                                         float(np.max(np.abs(np.concatenate([e_a, e_b], 1)
                                                             - e_w))))
        pn["exec_tol"] = 1e-6
        pn["pass"] = bool(pn["spell_bad"] == 0 and pn["exec_max_abs"] <= pn["exec_tol"]
                          and pn["n_checked"] > 0)
        return pn

    # ================================================================= P-TAU: MEASURE the body
    t0 = time.time()

    def measure_tau(pc_body):
        from mjc.pusher_env import PusherEnv
        env = PusherEnv(pc_body.dgp(False), with_puck=False)
        tau_x = pc_body.tau
        n = int(max(100, round(10.0 * tau_x / pc_body.timestep)))
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
        return dict(tau_fit=float(-1.0 / sl), tau_expected=float(tau_x),
                    rel_err=float(abs(-1.0 / sl - tau_x) / tau_x),
                    v_inf_fit=vinf, v_inf_expected=float(pc_body.v_terminal),
                    v_rel_err=float(abs(vinf - pc_body.v_terminal) / pc_body.v_terminal))

    KIN_ARMS = KIN_SETS[str(cfg.get("arm_set", "ceiling"))]
    R_star = float(cfg["r_star"])
    tempos = list(cfg["tempos"])
    L = int(cfg["n_levels"])
    pc0 = P.accel_piece(R_star, tempos[0], k_app=cfg["k_app"], n_levels=L,
                        damping=cfg["damping"], mass=cfg["mass"])
    ptau = dict(tol=0.02, fast=measure_tau(pc0), donor=measure_tau(P.donor_piece()))
    ptau["pass"] = bool(ptau["fast"]["rel_err"] <= ptau["tol"]
                        and ptau["donor"]["rel_err"] <= ptau["tol"]
                        and ptau["fast"]["v_rel_err"] <= ptau["tol"])
    out["P_TAU"] = ptau
    PR(f"[P-TAU] fast tau_fit={ptau['fast']['tau_fit'] * 1000:.2f} ms vs "
       f"{ptau['fast']['tau_expected'] * 1000:.2f}  (rel {ptau['fast']['rel_err']:.2e}); "
       f"donor {ptau['donor']['tau_fit'] * 1000:.1f} vs "
       f"{ptau['donor']['tau_expected'] * 1000:.1f}  pass={ptau['pass']}")
    if not ptau["pass"]:
        save()
        raise SystemExit("P-TAU failed: the body is not the body it was designed to be")

    # ===================================================== P-ZOH: the discretisation the oracle
    # inverts is the one the plant runs. TWO readings, both from the plant itself:
    #   (a) alpha, the CONTROL-STEP pole, fit from a free step response at control-step
    #       resolution, against the analytic exp(-dt_ctrl / tau) the oracle uses;
    #   (b) the one-step residual of `fwd_zoh` against the plant on random (state, command)
    #       pairs, on the CLEAN world (where the body model is the world model) and on the
    #       ROTATED world (where it is not, by construction).
    # The oracle uses the ANALYTIC constants — "the body's true constants", per the brief — so
    # (a) is a measurement of what that choice costs, not a fitting step.
    bc = body_consts(pc0)

    def measure_alpha(pc_body, rot):
        from mjc.pusher_env import PusherEnv
        env = PusherEnv(pc_body.dgp(rot), with_puck=False)
        env.set_state(np.zeros(2), np.zeros(2))
        vs = [0.0]
        for _ in range(40):
            st, _ = env.step(np.array([1.0, 0.0], np.float32), int(pc_body.frame_skip))
            vs.append(float(st[2]))
            env.set_state(np.zeros(2), np.asarray(st[2:], np.float64))   # position-free
        v = np.asarray(vs, np.float64)
        vinf = float(v[-1])
        m = np.abs(v[:-1] - vinf) > 1e-6
        al = float(np.median((v[1:][m] - vinf) / (v[:-1][m] - vinf)))
        return dict(alpha_fit=al, v_inf=vinf, n=int(m.sum()))

    def fwd_residual(pc_body, rot, n=256, seed=11):
        from mjc.pusher_env import PusherEnv
        env = PusherEnv(pc_body.dgp(rot), with_puck=False)
        rng = np.random.default_rng(seed)
        wp = np.asarray(pc_body.wps, np.float64)
        # states drawn ON the figure (so the rotated world's region is actually visited) with
        # velocities at the schedule's own scale
        idx = rng.integers(0, len(wp) - 1, n)
        f = rng.random((n, 1))
        x = wp[idx] * (1 - f) + wp[idx + 1] * f
        vsc = pc_body.mean_leg() / pc_body.note_s()
        v = rng.normal(0, vsc, (n, 2))
        st = np.concatenate([x, v], 1).astype(np.float32)
        u = np.clip(rng.normal(0, 0.5, (n, 2)), -1, 1).astype(np.float32)
        nxt = np.empty((n, 4), np.float64)
        for i in range(n):
            env.set_state(st[i, :2].astype(np.float64), st[i, 2:].astype(np.float64))
            s2, _ = env.step(u[i], int(pc_body.frame_skip))
            nxt[i] = s2
        pred = np.asarray(fwd_zoh(body_consts(pc_body), st, u), np.float64)
        dv = np.linalg.norm(pred[:, 2:] - nxt[:, 2:], axis=1)
        dx = np.linalg.norm(pred[:, :2] - nxt[:, :2], axis=1)
        return dict(dv_max=float(dv.max()), dv_rms=float(np.sqrt((dv ** 2).mean())),
                    dx_max=float(dx.max()), dx_rms=float(np.sqrt((dx ** 2).mean())),
                    v_scale=float(vsc), n=int(n))

    ma = measure_alpha(pc0, rot=False)
    pzoh = dict(alpha_analytic=bc["alpha"], alpha_fit=ma["alpha_fit"],
                alpha_rel=float(abs(ma["alpha_fit"] - bc["alpha"]) / bc["alpha"]),
                v_inf_fit=ma["v_inf"], v_term=bc["v_term"], dt_over_tau=bc["dt"] / bc["tau"],
                accel_coeff_zoh=bc["dt"] / ((1 - bc["alpha"]) * bc["v_term"]),
                accel_coeff_ct=1.0 / bc["accel"], drag_coeff=1.0 / bc["v_term"],
                scheme_ratio=(bc["dt"] / bc["tau"]) / (1 - bc["alpha"]),
                clean=fwd_residual(pc0, rot=False), rotated=fwd_residual(pc0, rot=True),
                tol_alpha=1e-3)
    pzoh["pass"] = bool(pzoh["alpha_rel"] <= pzoh["tol_alpha"])
    out["P_ZOH"] = pzoh
    out["t_ptau"] = time.time() - t0
    PR(f"[P-ZOH] alpha measured {pzoh['alpha_fit']:.8f} vs analytic {pzoh['alpha_analytic']:.8f} "
       f"(rel {pzoh['alpha_rel']:.2e})  dt/tau={pzoh['dt_over_tau']:.3f}  "
       f"zoh/ct accel ratio={pzoh['scheme_ratio']:.4f}  pass={pzoh['pass']}")
    PR(f"        fwd_zoh one-step residual: clean |dv| rms {pzoh['clean']['dv_rms']:.3e} max "
       f"{pzoh['clean']['dv_max']:.3e} | rotated rms {pzoh['rotated']['dv_rms']:.3e} max "
       f"{pzoh['rotated']['dv_max']:.3e}   (v scale {pzoh['clean']['v_scale']:.3f} m/s)")
    save()
    if not pzoh["pass"]:
        save()
        raise SystemExit("P-ZOH failed: the plant is not the discretisation the oracle inverts")

    # ============================================ the pinned gain table, and gate P-G
    gains, refit_rows = {}, []
    for H in tempos:
        if int(H) in T1.T1_GAINS and not cfg.get("refit_gains"):
            gains[int(H)] = {int(D): tuple(v) for D, v in T1.T1_GAINS[int(H)].items()}
            continue
        pcH = P.accel_piece(R_star, H, k_app=cfg["k_app"], n_levels=L,
                            damping=cfg["damping"], mass=cfg["mass"])
        Wf = World(cfg, pcH, n_proc=1, rot=False)
        evf = Wf.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        g = {}
        for D in sorted({0, int(cfg["delta_fix"])}):
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
                                   e_piece=bb["e_piece"]))
        gains[int(H)] = g
        Wf.close()
    if refit_rows:
        out["gains_refit"] = refit_rows
        PR("[gains] re-fit for tempi t1 never ran: " + "  ".join(
            f"H{r['H']}D{r['delta']}:({r['kp']:g},{r['kd']:g})" for r in refit_rows))

    t1_ok = all(float(cfg.get(k, -1)) == float(v) for k, v in T1.T1_CFG.items())
    pt1 = dict(rows=[], max_abs=0.0, cfg_match=bool(t1_ok),
               cfg_diff={k: [cfg.get(k), v] for k, v in T1.T1_CFG.items()
                         if float(cfg.get(k, -1)) != float(v)})

    t0 = time.time()
    Hg, Dg = T1.T1_GAIN_CHECK
    _pg_ok = bool(Hg in tempos and not cfg.get("refit_gains"))
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
    PR(f"[P-G] gain check at (H={Hg}, D={Dg}): re-fit {pg['refit']} vs pinned {pg['pinned']}  "
       f"e={b['e_piece']:.6f}  applicable={pg['applicable']}  pass={pg['pass']}")
    if _pg_ok and not pg["pass"]:
        save()
        raise SystemExit("P-G failed: the pinned gain table is not what this fork re-fits")
    save()

    # ================================== the libraries, their PATHS, and gate P-T1's build half
    t0 = time.time()
    Ws, pcs, evs, rts, libs, paths, pn_by = {}, {}, {}, {}, {}, {}, {}
    train_pools = {}
    for H in tempos:
        pc = P.accel_piece(R_star, H, k_app=cfg["k_app"], n_levels=L, damping=cfg["damping"],
                           mass=cfg["mass"])
        W = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
        ev = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        rt = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])
        Ws[H], pcs[H], evs[H], rts[H] = W, pc, ev, rt
        out.setdefault("piece_by_tempo", {})[str(H)] = pc.describe()
        llb = Ledger(W.dt_ctrl, W.d_fb)
        lvlib, b1r, blr, tpool = build_ladder(W, pc, cfg, cfg["seed"], gains[H][0], llb, rt)
        train_pools[H] = tpool
        libs[H] = lvlib
        # THE PATHS. One grounding per member, executed from its own launch key. Charged and
        # reported: measuring a unit's path is a real trial on the plant, exactly as auditioning
        # a tape is, and the ledger says so.
        plb = Ledger(W.dt_ctrl, W.d_fb)
        paths[H] = measure_paths(W, lvlib, led=plb, charge=True)
        out.setdefault("library", {})[str(H)] = dict(
            level1=b1r, levels=blr, counts=lvlib.counts(), build_ledger=llb.snap(),
            path_ledger=plb.snap(), n_paths=len(paths[H]), harvest_gains=list(gains[H][0]))
        if t1_ok and H in T1.T1_L1:
            for i, r in enumerate(b1r):
                pt1["rows"].append(dict(what=f"L1@{H}[{i}]", got=r["chosen_score"],
                                        want=T1.T1_L1[H][i]))
            got = {(r["level"], r["seam"]): r.get("chosen_score") for r in blr}
            for e, k, v in T1.T1_LN[H]:
                pt1["rows"].append(dict(what=f"L{e}@{H}[{k}]", got=got[(e, k)], want=v))
        pnn = check_nesting(W, lvlib, pc, rt)
        pn_by[str(H)] = pnn
        # the do-nothing floor, both grades
        led = Ledger(W.dt_ctrl, W.d_fb)
        odn = W.traverse(ZeroDecider(W), ev, np.random.default_rng(1), led)
        out.setdefault("do_nothing", {})[str(H)] = dict(e_piece=odn["e_piece"],
                                                        e_listen=odn["e_listen"])
        PR(f"[lib H={H:2d}] {lvlib.counts()}  {len(paths[H])} paths  "
           f"{llb.ground:.0f}+{plb.ground:.0f} groundings  P-N {pnn['n_checked']} spelled/"
           f"{pnn['spell_bad']} bad/{pnn['exec_max_abs']:.2e}  do-nothing {odn['e_piece']:.4f}"
           f"/{odn['e_listen']:.4f}  band {pc.band():.4f}")
        save()
    out["P_N"] = dict(by_tempo=pn_by, **{"pass": all(v["pass"] for v in pn_by.values())})
    out["t_library"] = time.time() - t0

    # ================================================= the SCALED libraries, and P-R (identity)
    H_src = tempos[0]                      # the SLOWEST tempo is the stored program's source
    scaled = {}
    for p_exp in cfg["scale_exponents"]:
        for H in tempos:
            scaled[(p_exp, H)] = resample_library(libs[H_src], H_src, H, p=p_exp)
    s_id = scaled[(0.0, H_src)]
    mx, nchk = 0.0, 0
    for (ell, k) in sorted(libs[H_src].layout.off):
        for a, bsl in zip(libs[H_src].slots(ell, k), s_id.slots(ell, k)):
            for ma, mb in zip(a["members"], bsl["members"]):
                mx = max(mx, float(np.max(np.abs(ma["cmds"] - mb["cmds"]))))
                nchk += 1
    out["P_R"] = dict(src_tempo=int(H_src), n_members=nchk, max_abs=mx,
                      **{"pass": bool(mx == 0.0)})
    PR(f"[P-R] command resampler identity at s=1: max|delta| = {mx:.3e} over {nchk} members  "
       f"pass={mx == 0.0}")

    # ============================================ P-KN: the PHASE resampler splits exactly
    # A level-l unit's phase grid, split at its midpoint, is its two halves' phase grids. The
    # destination sample counts are exact multiples at every rung of a dyadic ladder, so this is
    # an identity and not a tolerance. It is what licenses reading a kinematic level-l unit as
    # two kinematic level-(l-1) units in sequence.
    pkn = dict(rows=[], max_abs=0.0, n=0)
    for H in tempos:
        for (ell, k) in sorted(libs[H_src].layout.off):
            if ell < 2:
                continue
            spn = LadderLayout.span(ell)
            for sl in libs[H_src].slots(ell, k, with_poison=False):
                for mi in range(len(sl["members"])):
                    pth = paths[H_src][(ell, k, sl["j"], mi)]
                    n_dst = spn * int(H)
                    whole = resample_path(pth, n_dst, float(H_src) / float(H))
                    half = pth.shape[0] // 2
                    a_h = resample_path(pth[:half + 1], n_dst // 2, float(H_src) / float(H))
                    b_h = resample_path(pth[half:], n_dst // 2, float(H_src) / float(H))
                    d = max(float(np.max(np.abs(whole[:n_dst // 2 + 1] - a_h))),
                            float(np.max(np.abs(whole[n_dst // 2:] - b_h))))
                    pkn["max_abs"] = max(pkn["max_abs"], d)
                    pkn["n"] += 1
    pkn["pass"] = bool(pkn["max_abs"] == 0.0 and pkn["n"] > 0)
    out["P_KN"] = pkn
    PR(f"[P-KN] phase resampler splits at the span midpoint: max|delta| = {pkn['max_abs']:.3e} "
       f"over {pkn['n']} members  pass={pkn['pass']}")
    save()

    # =============================================== PHASE 2: the LEARNED cerebellum, fitted on
    # the body's own executed data at the SLOW tempi only. Two families, both reported: the
    # minimal one (linear in (v, a) — the true inverse's own functional form, with two free
    # numbers) and a generic MLP. Nothing here is handed to the incumbent; the fitted model is an
    # EXECUTOR on the learner's side of the meter.
    t0 = time.time()
    fits, fit_rep = {}, {}
    if cfg.get("fit_tempos"):
        ft = [int(x) for x in cfg["fit_tempos"]]
        flb = Ledger(Ws[ft[0]].dt_ctrl, Ws[ft[0]].d_fb)
        Vt, At, Ut, own = [], [], [], {}
        n0 = 0
        for H in ft:
            v, a, u = N.triples_from_pool(Ws[H], train_pools[H], Ws[H].dt_ctrl, led=flb,
                                          max_rend=cfg.get("fit_max_rend"),
                                          rng=np.random.default_rng(cfg["seed"] + 7))
            own[int(H)] = (n0, n0 + len(v))          # this tempo's slice of the pooled triples
            n0 += len(v)
            Vt.append(v); At.append(a); Ut.append(u)
        V = np.concatenate(Vt, 0); A = np.concatenate(At, 0); U = np.concatenate(Ut, 0)
        itr, iho = N.split_triples(len(V), cfg["fit_hold"], cfg["seed"])
        pfh = dict(n=int(len(V)), n_train=int(len(itr)), n_hold=int(len(iho)),
                   overlap=int(len(np.intersect1d(itr, iho))))
        pfh["pass"] = bool(pfh["overlap"] == 0 and pfh["n_hold"] > 0)
        out["P_FH"] = pfh
        PR(f"[P-FH] fit split: {pfh['n_train']} train / {pfh['n_hold']} hold, overlap "
           f"{pfh['overlap']}  pass={pfh['pass']}   (fitted on tempi {ft}, "
           f"{flb.ground:.0f} groundings)")
        fits["lin"] = N.fit_linear(V[itr], A[itr], U[itr])
        fit_rep["lin"] = dict(family="linear in (v, a)", n_train=int(len(itr)),
                              **N.linear_report(fits["lin"], bc))
        lr = fit_rep["lin"]
        PR(f"[fit/lin] c_v {lr['c_v_fit'][0]:.6f},{lr['c_v_fit'][1]:.6f} vs true "
           f"{lr['c_v_true']:.6f}   c_a {lr['c_a_fit'][0]:.6f},{lr['c_a_fit'][1]:.6f} vs true "
           f"{lr['c_a_true']:.6f}   |off-diag| max {lr['off_diag_max']:.2e}  |bias| max "
           f"{lr['bias_max']:.2e}")
        # THE FREE DISCRIMINATOR for `klsmoke3`'s unresolved observation (fitted executors
        # beating the exact one at the deep rung): the SAME renditions replayed on the CLEAN
        # world, rotation region off. If the clean-world fit's bias and off-diagonals collapse
        # while its coefficients stay put, the rotated-world fit's bias IS the absorbed rotation;
        # if the bias survives, it is not. One extra `rollout_path` batch per fit tempo.
        if cfg.get("fit_clean"):
            Vc, Ac, Uc = [], [], []
            for H in ft:
                Wc = World(cfg, pcs[H], n_proc=int(cfg["n_proc"]), rot=False)
                v, a, u = N.triples_from_pool(Wc, train_pools[H], Wc.dt_ctrl, led=None,
                                              max_rend=cfg.get("fit_max_rend"),
                                              rng=np.random.default_rng(cfg["seed"] + 7))
                Wc.close()
                Vc.append(v); Ac.append(a); Uc.append(u)
            Vc = np.concatenate(Vc, 0); Ac = np.concatenate(Ac, 0); Uc = np.concatenate(Uc, 0)
            fits["linc"] = N.fit_linear(Vc[itr], Ac[itr], Uc[itr])
            fit_rep["linc"] = dict(family="linear in (v, a), CLEAN world (rotation off)",
                                   n_train=int(len(itr)), **N.linear_report(fits["linc"], bc))
            lc = fit_rep["linc"]
            PR(f"[fit/linc] CLEAN world: c_v {lc['c_v_fit'][0]:.6f},{lc['c_v_fit'][1]:.6f}  "
               f"c_a {lc['c_a_fit'][0]:.6f},{lc['c_a_fit'][1]:.6f}   |off-diag| max "
               f"{lc['off_diag_max']:.2e}  |bias| max {lc['bias_max']:.2e}   (rotated-world fit: "
               f"off {lr['off_diag_max']:.2e}, bias {lr['bias_max']:.2e})")
        fits["mlp"] = N.fit_mlp(V[itr], A[itr], U[itr], seed=cfg["seed"],
                                hidden=cfg["mlp_hidden"], layers=cfg["mlp_layers"],
                                epochs=cfg["mlp_epochs"], device=device, log=PR)
        fit_rep["mlp"] = dict(family="MLP", n_train=int(len(itr)),
                              hidden=cfg["mlp_hidden"], layers=cfg["mlp_layers"],
                              epochs=cfg["mlp_epochs"],
                              hist=fits["mlp"]["hist"])
        # HELD-OUT FIT AT EVERY TEMPO, beside execution: the same two families queried on each
        # tempo's OWN practice triples. This is the extrapolation question asked directly in
        # command units, separate from whatever execution then charges for it.
        rows = []
        for H in tempos:
            v, a, u = N.triples_from_pool(Ws[H], train_pools[H], Ws[H].dt_ctrl, led=None,
                                          max_rend=cfg.get("fit_eval_rend"),
                                          rng=np.random.default_rng(cfg["seed"] + 11))
            if int(H) in ft:
                # At a fitted tempo, score THAT TEMPO'S OWN held-out slice, not the pooled one.
                # With more than one tempo in the fit set the pooled hold-out mixes them and the
                # per-tempo row stops meaning what its column header says.
                lo, hi = own[int(H)]
                sel = iho[(iho >= lo) & (iho < hi)]
                v, a, u = V[sel], A[sel], U[sel]
            cov = N.coverage(V[itr], A[itr], v, a)
            for nm in ("lin", "mlp"):
                r = N.heldout_mse(fits[nm], v, a, u)
                rows.append(dict(family=nm, H=int(H), fitted=bool(int(H) in ft), **r,
                                 v_reach=cov["v"]["reach"], a_reach=cov["a"]["reach"],
                                 v_frac_out=cov["v"]["frac_outside"],
                                 a_frac_out=cov["a"]["frac_outside"]))
            PR(f"[fit/heldout] H={H:2d}{' (fitted)' if int(H) in ft else '        '}  "
               + "  ".join(f"{nm} mse {[r for r in rows if r['family'] == nm and r['H'] == H][0]['mse']:.3e}"
                           for nm in ("lin", "mlp"))
               + f"   reach v {cov['v']['reach']:.2f}x a {cov['a']['reach']:.2f}x   "
                 f"frac outside v {cov['v']['frac_outside']:.3f} a "
                 f"{cov['a']['frac_outside']:.3f}")
        out["fit_heldout"] = rows
        out["fit"] = fit_rep
    out["t_fit"] = time.time() - t0
    save()

    # ============================================ the KINEMATIC libraries, and P-K (the oracle
    # ============================================ the KINEMATIC libraries, and P-K (the oracle
    # identity). Built for every (scheme, source, destination tempo). The `own` source is s = 1
    # and is the CONVERSION-COST CONTROL; the `slow` source is the transfer question.
    t0 = time.time()
    kin, sat = {}, []

    def span_notes_of(ell):
        return LadderLayout.span(int(ell))

    def scheme_of(tagname):
        if isinstance(tagname, str) and tagname.startswith("fit:"):
            fam = tagname.split(":", 1)[1]
            if fam not in fits:
                return None
            return N.inv_fitted(fits[fam], bc["dt"])
        return tagname

    for tag, scheme, srckind, lpk in KIN_ARMS:
        sch = scheme_of(scheme)
        if sch is None:
            PR(f"[kin] arm {tag} skipped: no fitted model for {scheme}")
            continue
        for H in tempos:
            hs = H_src if srckind == "slow" else H
            lb, srows = kin_library(libs[hs], paths[hs], bc, span_notes_of, hs, H,
                                    scheme=sch, lp_kind=lpk)
            kin[(tag, H)] = lb
            for r in srows:
                sat.append(dict(arm=tag, scheme=str(scheme), source=srckind, H=int(H),
                                h_src=int(hs), lp=str(lpk),
                                lp_w=int(lp_width(lpk, hs, H)), **r))
    out["saturation"] = sat
    out["t_kin"] = time.time() - t0
    # THE SATURATION INSTRUMENT, per (arm, tempo, level): the fraction of scalar command
    # components the clip actually bound when the path was converted. `accelerando`'s own tempo
    # calibration found FEASIBILITY binding at 48 ms notes (u_turn 0.87 of the actuator at
    # R* = 0.160), so the ceiling of the factored architecture is an actuator ceiling before it
    # is anything else, and this is the number that says so.
    for tag, scheme, srckind, lpk in KIN_ARMS:
        if (tag, tempos[0]) not in kin:
            continue
        msg = []
        for H in tempos:
            per = {}
            for r in sat:
                if r["arm"] == tag and r["H"] == H and not r["poison"]:
                    per.setdefault(r["level"], []).append(r["sat"])
            msg.append(f"H{H}:" + "/".join(f"{np.mean(per[e]):.3f}" for e in sorted(per)))
        PR(f"[sat/{tag}] (L1/L2/L3/L4 mean clip fraction)  " + "   ".join(msg))
    save()

    # P-K: the ORACLE IDENTITY. A member's kinematic unit converted back at its OWN tempo must
    # reproduce its own commands. Split by whether the step was taken INSIDE the world's
    # command-rotation region — where the BODY model is not the WORLD model, by construction and
    # by `acappella`'s standing choice to leave the hard passage unmodelled rather than wrongly
    # modelled. The gate is the out-of-region half; the in-region residual is REPORTED, and so
    # is the whole residual-vs-gate-weight curve, so that "the residual is the rotation and
    # nothing else" is a table anyone can read rather than a claim.
    #
    # WHY THE WEIGHT IS THE MAX OVER THE STEP'S TWO ENDPOINTS AND NOT ITS START. `ksmoke0`
    # measured the apparatus fact: the env re-evaluates its Gaussian gate at every 2 ms
    # PHYSICS substep, and at the fastest tempo the body covers 0.06 m in one 24 ms control
    # step against a region sigma of 0.0244 m — 2.5 sigma per step. A step that starts 4 sigma
    # outside can therefore finish inside, and classifying it by its start state alone put
    # genuinely in-region steps in the out-of-region half (out_max 4.8e-3 against an rms of
    # 1.8e-4). Taking the max over both endpoints bounds the traversal for any step short
    # enough not to enter and leave between them, which at 2.5 sigma it is. The threshold
    # itself is unchanged and is the env's OWN cutoff.
    W_BUCKETS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
    pk = dict(tol=PK_TOL, region_w=PK_REGION_W, buckets=[],
              out_max=0.0, out_rms=0.0, in_max=0.0, in_rms=0.0, n_out=0, n_in=0)
    so, si, no, ni = 0.0, 0.0, 0, 0
    bsum = [0.0] * (len(W_BUCKETS) - 1)
    bmax = [0.0] * (len(W_BUCKETS) - 1)
    bn = [0] * (len(W_BUCKETS) - 1)
    for H in tempos:
        reg = pcs[H].regions
        for (ell, k) in sorted(libs[H].layout.off):
            for sl in libs[H].slots(ell, k, with_poison=True):
                for mi, m in enumerate(sl["members"]):
                    pth = paths[H][(ell, k, sl["j"], mi)]
                    n_dst = span_notes_of(ell) * int(H)
                    cm, _s, rp = kin_cmds(bc, pth, n_dst, 1.0, scheme="zoh")
                    d = np.abs(np.asarray(cm, np.float64) - np.asarray(m["cmds"], np.float64))
                    w = np.zeros(n_dst)
                    for rg in reg:
                        c = np.asarray(rg["center"], np.float64)
                        g = np.exp(-((rp[:, 0] - c[0]) ** 2 + (rp[:, 1] - c[1]) ** 2)
                                   / (2.0 * float(rg["sigma"]) ** 2))
                        w = np.maximum(w, np.maximum(g[:-1], g[1:]))
                    inr = w >= PK_REGION_W
                    if (~inr).any():
                        dd = d[~inr]
                        pk["out_max"] = max(pk["out_max"], float(dd.max()))
                        so += float((dd ** 2).sum()); no += dd.size
                    if inr.any():
                        dd = d[inr]
                        pk["in_max"] = max(pk["in_max"], float(dd.max()))
                        si += float((dd ** 2).sum()); ni += dd.size
                    for bi in range(len(W_BUCKETS) - 1):
                        msk = (w >= W_BUCKETS[bi]) & (w < W_BUCKETS[bi + 1])
                        if msk.any():
                            dd = d[msk]
                            bsum[bi] += float((dd ** 2).sum()); bn[bi] += dd.size
                            bmax[bi] = max(bmax[bi], float(dd.max()))
    pk["n_out"], pk["n_in"] = int(no), int(ni)
    pk["out_rms"] = float(np.sqrt(so / max(1, no)))
    pk["in_rms"] = float(np.sqrt(si / max(1, ni)))
    pk["buckets"] = [dict(w_lo=W_BUCKETS[i], w_hi=W_BUCKETS[i + 1], n=bn[i], max=bmax[i],
                          rms=float(np.sqrt(bsum[i] / max(1, bn[i]))))
                     for i in range(len(W_BUCKETS) - 1)]
    pk["pass"] = bool(no > 0 and pk["out_max"] <= PK_TOL)
    out["P_K"] = pk
    PR(f"[P-K] oracle identity (zoh, own tempo, s=1): OUT of region max|du| = "
       f"{pk['out_max']:.3e} rms {pk['out_rms']:.3e} over {pk['n_out']} steps (tol {PK_TOL:.0e}) "
       f"pass={pk['pass']}   IN region max {pk['in_max']:.3e} rms {pk['in_rms']:.3e} over "
       f"{pk['n_in']} steps (REPORTED, not gated)")
    PR("      residual by the world's own gate weight: " + "  ".join(
        f"w<{b['w_hi']:.0e}:n{b['n']} rms {b['rms']:.2e} max {b['max']:.2e}"
        for b in pk["buckets"]))
    save()
    if not pk["pass"]:
        save()
        raise SystemExit("P-K failed: the kinematic unit does not invert to its own tape")

    # ======================================================= PARITY, per slot, per tempo
    # `solo`'s gate with the tempo added to the index, and with BOTH sides content rather than a
    # head and a tape: a slot is OPEN at a tempo iff the arm's own members, EXECUTED ON THE PLANT
    # from that slot's HELD-OUT launch states, beat the SAME-TEMPO RECORDING's audition winner on
    # at least `tau` of them. The reference is the per-state winner, not a fixed member, exactly
    # as `nets.ref_errors` computes it.
    #
    # HELD-OUT means held out of CONSTRUCTION, twice over. The states are collected under the
    # DEPLOYMENT condition (Delta = delta_fix, app = fixed, a level-1-keyed traversal) rather
    # than the harvest condition the construction audition ran at, which is `accelerando`
    # decision 33's fix carried forward. That already makes them different states at every
    # drilled seam past the first — but NOT at drilled seam 0, where `app = fixed` plays the
    # same Delta = 0 lead-in at the same harvest gains and the hand-over is numerically
    # identical to `score_set`'s. So the `[n_score:]` slice is kept on top of it, which
    # guarantees disjointness at seam 0 as well. Both guards, not either.
    #
    # The reference side is an INSTRUMENT and is not charged; the arm side is a real self-check
    # and is. Continuous fractions are logged so another tau can be read off the record.
    t0 = time.time()
    D0_PAR = int(cfg["delta_fix"])
    par_rows = []
    hold_by = {}
    if cfg.get("do_parity", True):
        tau_p = float(cfg["parity_tau"])
        min_p = int(cfg["parity_min"])
        for H in tempos:
            W, pc = Ws[H], pcs[H]
            W.kp_app, W.kd_app = gains[H][0]
            W.kp, W.kd = gains[H][D0_PAR]
            W.obs_predict, W.fm_step = False, None
            led = Ledger(W.dt_ctrl, W.d_fb)
            oh = W.traverse(LadderDecider(W, libs[H], mode="key", levels=(1,),
                                          obs_delay=D0_PAR),
                            rts[H], np.random.default_rng(cfg["seed"] + 4400), led,
                            obs_delay=D0_PAR, obs_delay_app=0)
            hold = {d: np.asarray(oh["seam"][W.k0 + d], np.float32)[int(cfg["n_score"]):]
                    for d in range(W.K_drill)}
            hold_by[H] = hold
            n_open = {}
            for (ell, k) in sorted(libs[H].layout.off):
                S = hold[int(k)]
                if len(S) < min_p:
                    continue
                spn = LadderLayout.span(ell)
                kw = int(W.k0) + int(k)
                for sl in libs[H].slots(ell, k, with_poison=False):
                    mm = sl["members"]
                    n, m = len(S), len(mm)
                    s0 = np.repeat(S, m, axis=0)
                    cm = np.tile(np.stack([b["cmds"] for b in mm]), (n, 1, 1))
                    e_ref, _ = W.rollout(s0, cm, kw, spn, None, charge=False)
                    e_ref = e_ref.mean(1).reshape(n, m).min(1)
                    for tag, _sch, _sk, _lp in KIN_ARMS:
                        if (tag, H) not in kin:
                            continue
                        asl = kin[(tag, H)].slot_by_id(int(sl["slot"]))
                        if asl is None:
                            continue
                        am = asl["members"]
                        cma = np.tile(np.stack([b["cmds"] for b in am]), (n, 1, 1))
                        s0a = np.repeat(S, len(am), axis=0)
                        e_a, _ = W.rollout(s0a, cma, kw, spn, led)
                        e_a = e_a.mean(1).reshape(n, len(am)).min(1)
                        fr = float(np.mean(e_a <= e_ref))
                        par_rows.append(dict(
                            arm=tag, H=int(H), level=int(ell), drilled=int(k),
                            slot=int(sl["slot"]), n_hold=int(n), frac=fr,
                            mean_e_arm=float(e_a.mean()), mean_e_ref=float(e_ref.mean()),
                            mean_gap=float((e_a - e_ref).mean()),
                            arm_in_band=float(np.mean(e_a <= pc.band())),
                            ref_in_band=float(np.mean(e_ref <= pc.band())),
                            open=bool(fr >= tau_p)))
                        n_open[tag] = n_open.get(tag, 0) + int(fr >= tau_p)
            tot = {t: len([r for r in par_rows if r["arm"] == t and r["H"] == H])
                   for t, _a, _b, _c in KIN_ARMS}
            PR(f"[parity] H={H:2d} vs the same-tempo recording (tau={tau_p}): " + "  ".join(
                f"{t}: {n_open.get(t, 0)}/{tot.get(t, 0)}" for t, _a, _b, _c in KIN_ARMS))
            save()
    out["parity"] = dict(rows=par_rows, tau=cfg.get("parity_tau"),
                         min_hold=cfg.get("parity_min"), reference="rec (same tempo)")
    out["t_parity"] = time.time() - t0
    save()

    # ===================================== THE TWO DIAGNOSTICS (decisions 19 and 21)
    # Run only when asked (`--diagnostics`), so the ceiling tags stay exactly what they were.
    # Both consume the SAME held-out launch states parity uses, and neither is charged: they are
    # instruments the experimenter reads, not decisions any arm makes.
    t0 = time.time()
    if cfg.get("do_diag") and hold_by:
        shp, lau = [], []
        for H in tempos:
            W = Ws[H]
            for lpk in cfg["diag_lp"]:
                for r in shape_distance(paths[H_src], paths[H], libs[H], span_notes_of,
                                        H_src, H, lp_kind=lpk):
                    shp.append(dict(H=int(H), h_src=int(H_src), lp=str(lpk),
                                    lp_w=int(lp_width(lpk, H_src, H)), **r))
            for tag, _sch, srckind, lpk in KIN_ARMS:
                if srckind != "slow" or (tag, H) not in kin:
                    continue
                for r in launch_rows(W, libs[H], kin[(tag, H)], paths[H_src], span_notes_of,
                                     H_src, H, hold_by[H], bc, lp_kind=lpk, led=None):
                    lau.append(dict(arm=tag, H=int(H), **r))
            for lpk in cfg["diag_lp"]:
                rs = [r for r in shp if r["H"] == H and r["lp"] == str(lpk)]
                PR(f"[shape] H={H:2d} lp={lpk} (w={lp_width(lpk, H_src, H)} src samples)  "
                   + "  ".join(
                       f"L{e}: near {np.mean([r['d_near'] for r in rs if r['level'] == e]):.4f} "
                       f"within {np.mean([r['d_within'] for r in rs if r['level'] == e]):.4f}"
                       for e in range(1, L + 1)
                       if any(r["level"] == e for r in rs)))
            for tag in sorted({r["arm"] for r in lau if r["H"] == H}):
                rs = [r for r in lau if r["H"] == H and r["arm"] == tag]
                PR(f"[launch] H={H:2d} {tag}  " + "  ".join(
                    f"L{e}: own {np.mean([r['e_from_own_launch'] for r in rs if r['level'] == e]):.4f}"
                    f" act {np.mean([r['e_from_actual'] for r in rs if r['level'] == e]):.4f}"
                    f" dx {np.mean([r['dx'] for r in rs if r['level'] == e]):.4f}"
                    f" dv*tau {np.mean([r['dv_tau'] for r in rs if r['level'] == e]):.4f}"
                    for e in range(1, L + 1) if any(r["level"] == e for r in rs)))
            save()
        out["shape"] = shp
        out["launch"] = lau
    out["t_diag"] = time.time() - t0
    save()

    # ================================================ PHASE 3: the CORRECTED executor
    # Feedforward from the kinematic program, corrected online against a forecast rolled from the
    # Delta-stale read through the commands this executor already issued. The READ LADDER
    # (decision 18) is the arm of record's own coordinate: R = 1 reads every step and is the row
    # comparable with `reflex_ec`; R >= span reads only at launch.
    t0 = time.time()
    corr_rows, pe_rows, pd_rows = [], [], []
    if cfg.get("do_corrected"):
        targets, fwds = {}, {}
        for tag, scheme, srckind, lpk in KIN_ARMS:
            if (tag, tempos[0]) not in kin:
                continue
            for H in tempos:
                hs = H_src if srckind == "slow" else H
                targets[(tag, H)] = kin_targets(libs[hs], paths[hs], span_notes_of, hs, H,
                                                lp_kind=lpk)
        # THE TWO FORWARD MODELS. `exact` is `fwd_zoh`, which is `inv_zoh`'s algebraic inverse
        # on the same two constants; `fitted` is derived from the fitted LINEAR inverse by
        # algebraic inversion and introduces no new parameters. In both cases the forward and the
        # inverse ARE ONE BODY MODEL used in two directions, which is what licenses reading the
        # comparison as a comparison of one model's accuracy.
        fwds["exact"] = lambda st, u: fwd_zoh(bc, st, u)
        if "lin" in fits:
            fwds["fitted"] = N.fwd_from_linear(fits["lin"], bc["dt"])
        out["fm_note"] = ("exact: fwd_zoh is inv_zoh's algebraic inverse on the same alpha and "
                          "v_term. fitted: derived from the fitted linear inverse by inverting "
                          "its 2x2 acceleration block; no new parameters. Position is the "
                          "trapezoidal integral of the velocity channel in both, which is "
                          "kinematics and not a second model.")

        # ---- the PD gains: DECISION 36's rule, fixed before its grid is read --------------
        # `nets.pd_deadbeat` places both eigenvalues of the EXACT two-state discrete error system
        # at `rho`; the derivation, and the four reasons it replaces `pd_analytic`, are stated in
        # that function. The search is a 1-D grid over `rho` — one number, the per-control-step
        # decay of the tracking error — so an edge means "the body wanted a faster or a softer
        # correction than the grid spans" rather than naming a corner of a 2-D box.
        #
        # FIT PER (tempo, R), AS ASKED, WITH A PREDICTION WRITTEN DOWN FIRST. The brief poses the
        # loop as carrying an effective delay of the read interval R. That is true when the
        # estimate is HELD between reads; this executor DEAD-RECKONS every step, so with an exact
        # forward model on the clean world the estimate equals the true state at every step
        # whatever R is, and the loop carries no lag. The prediction is therefore that the chosen
        # `rho` is R-INVARIANT under exactly the conditions the fit runs in, and the fit is taken
        # at BOTH ends of the read ladder (R = 1 and the coarsest) so the prediction is measured
        # rather than asserted. If they disagree, the derivation above is wrong and the per-R
        # table is used as-is.
        RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        pdg, pd_check = {}, []
        for H in tempos:
            pc = pcs[H]
            Wc = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=False)
            evc = Wc.start_states(np.random.default_rng(cfg["seed"] + 6000),
                                  cfg["n_eval"])[:int(cfg.get("n_cal", 16))]
            Wc.kp_app, Wc.kd_app = gains[H][0]
            Wc.kp, Wc.kd = gains[H][D0_PAR]
            best_by_R = {}
            for Rfit in (1, int(max(cfg["read_ladder_fit_R"]))):
                grid = []
                for rho in RHO_GRID:
                    pdv = N.pd_deadbeat(bc, rho)
                    dec = CorrectedDecider(Wc, kin[("kzsa", H)], targets[("kzsa", H)],
                                           pdv["kp"], pdv["kd"], Rfit, levels=(1,),
                                           obs_delay=D0_PAR, fm=fwds["exact"], seed=cfg["seed"])
                    led = Ledger(Wc.dt_ctrl, Wc.d_fb)
                    o = Wc.traverse(dec, evc, np.random.default_rng(cfg["seed"] + 35), led,
                                    obs_delay=D0_PAR, obs_delay_app=0)
                    grid.append(dict(rho=rho, kp=pdv["kp"], kd=pdv["kd"],
                                     e_piece=o["e_piece"]))
                b = dict(min(grid, key=lambda r: r["e_piece"]))
                b["at_edge"] = bool(b["rho"] in (min(RHO_GRID), max(RHO_GRID)))
                b["R_fit"] = int(Rfit)
                # a PLAIN copy of the grid: `b` is one of its entries, so storing the list on
                # `b` itself would be a cycle and `json.dump` refuses it.
                b["grid"] = [dict(rho=r["rho"], kp=r["kp"], kd=r["kd"], e_piece=r["e_piece"])
                             for r in grid]
                best_by_R[int(Rfit)] = b
            Wc.close()
            rhos = sorted({b["rho"] for b in best_by_R.values()})
            pd_check.append(dict(H=int(H), rho_by_R={str(k): v["rho"]
                                                     for k, v in best_by_R.items()},
                                 r_invariant=bool(len(rhos) == 1)))
            b = best_by_R[1]
            pdg[H] = b
            pd_rows.append(dict(H=int(H), rho=b["rho"], kp=b["kp"], kd=b["kd"],
                                e_piece=b["e_piece"], at_edge=b["at_edge"],
                                rho_by_R={str(k): v["rho"] for k, v in best_by_R.items()},
                                grid=b["grid"]))
            PR(f"[pd/H={H:2d}] rho* = {b['rho']:g} -> kp={b['kp']:.4g} kd={b['kd']:.4g}  "
               f"e={b['e_piece']:.4f}  edge={b['at_edge']}   rho by R: " +
               " ".join(f"R{k}:{v['rho']:g}" for k, v in sorted(best_by_R.items())) +
               f"  R-invariant={pd_check[-1]['r_invariant']}")
            save()
        out["pd_gains"] = pd_rows
        out["pd_r_check"] = pd_check

        # ---- the read ladder ---------------------------------------------------------------
        def run_corr(H, nm, ff_tag, fm_name, R, ell, eps=0.0):
            W, pc, ev = Ws[H], pcs[H], evs[H]
            band, ml = pc.band(), pc.mean_leg()
            W.kp_app, W.kd_app = gains[H][0]
            W.kp, W.kd = gains[H][D0_PAR]
            W.obs_predict, W.fm_step = False, None
            g = pdg[H]
            dec = CorrectedDecider(W, kin[(ff_tag, H)], targets[(ff_tag, H)], g["kp"], g["kd"],
                                   R, levels=(int(ell),), obs_delay=D0_PAR,
                                   fm=fwds[fm_name], eps=eps, seed=cfg["seed"] + 5)
            led = Ledger(W.dt_ctrl, W.d_fb)
            o = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                           obs_delay=D0_PAR, obs_delay_app=0)
            ep = np.asarray(o["ep"], np.float64)
            el = np.asarray(o["ep_listen"], np.float64)
            spn = 1 << (int(ell) - 1)
            return dict(H=int(H), arm=nm, ff=ff_tag, fm=fm_name, R=int(R), level=int(ell),
                        eps=float(eps), e_piece=o["e_piece"], e_listen=o["e_listen"],
                        in_band=float(np.mean(ep <= band)),
                        lis_band=float(np.mean(el <= band)),
                        **{f"in_{k2}": float(np.mean(ep <= v * ml))
                           for k2, v in dict(q=0.25, t=1.0 / 3.0, h=0.5).items()},
                        **{f"lis_{k2}": float(np.mean(el <= v * ml))
                           for k2, v in dict(q=0.25, t=1.0 / 3.0, h=0.5).items()},
                        span_steps=int(spn * H), reads_per_span=int(np.ceil(spn * H / R)),
                        n_fb=float(o["ledger_drilled"]["n_fb"]),
                        ledger_drilled=o["ledger_drilled"])

        # DECISION 37: THE LADDER IS IN READS PER SPAN, not control steps. `kc1`'s ladder was
        # `R` in steps and capped at 16, so at the slow tempi its coarsest rung still cost 16
        # reads on a 256-step level-4 span and the "fewest reads" question was never asked there
        # at all. Reads-per-span `n` gives the SAME set at every (tempo, level): `n = 1` is
        # launch only — the once-at-launch end of decision 18's fork — and `n = span` is every
        # step. `R = ceil(span / n)`, and `n` is capped at the span so a 2-step span cannot be
        # asked for 16 reads; the realised `reads_per_span` is reported beside every row.
        def R_of(H, ell, n):
            span = (1 << (int(ell) - 1)) * int(H)
            n = int(min(int(n), span))
            return int(max(1, math.ceil(span / n))), n

        for H in tempos:
            for ell in range(1, L + 1):
                seen = set()
                for n in cfg["read_spans"]:
                    R, nn = R_of(H, ell, n)
                    if R in seen:
                        continue
                    seen.add(R)
                    corr_rows.append(run_corr(H, f"cor_lin_ex_n{nn}_L{ell}", "kfl", "exact",
                                              R, ell))
                if "fitted" in fwds:
                    seen = set()
                    for n in cfg["read_spans_fitted"]:
                        R, nn = R_of(H, ell, n)
                        if R in seen:
                            continue
                        seen.add(R)
                        corr_rows.append(run_corr(H, f"cor_lin_fit_n{nn}_L{ell}", "kfl",
                                                  "fitted", R, ell))
                seen = set()
                for n in cfg["read_spans_ctrl"]:
                    R, nn = R_of(H, ell, n)
                    if R in seen:
                        continue
                    seen.add(R)
                    corr_rows.append(run_corr(H, f"cor_orc_ex_n{nn}_L{ell}", "kzsa",
                                              "exact", R, ell))
            g = {r["arm"]: r["e_piece"] for r in corr_rows if r["H"] == H}
            PR(f"[corr] H={H:2d} band={pcs[H].band():.4f}  " + "   ".join(
                f"L{e}: " + "/".join(
                    f"{g.get(f'cor_lin_ex_n{R_of(H, e, n)[1]}_L{e}', float('nan')):.4f}"
                    for n in cfg["read_spans"])
                for e in range(1, L + 1)) + "   (reads/span = " +
               "/".join(str(n) for n in cfg["read_spans"]) + ")")
            save()
        out["corrected"] = corr_rows

        # ---- P-E, with and without correction ----------------------------------------------
        # `accelerando`'s exchange rate, re-asked on the corrected arm: perturb the FEEDFORWARD
        # command by eps and read the band cost per level at each read count. The uncorrected row
        # is the same executor with the PD term off and no reads past launch, so the two sides of
        # the comparison differ in exactly the correction.
        for H in tempos:
            for ell in cfg["pe_levels"]:
                for n in cfg["pe_spans"]:
                    R, nn = R_of(H, ell, n)
                    for eps in cfg["pe_eps"]:
                        r = run_corr(H, f"pe_n{nn}_L{ell}", "kfl", "exact", R, ell, eps=eps)
                        r["corrected"] = bool(nn > 1)
                        r["n_span"] = int(nn)
                        pe_rows.append(r)
            PR(f"[P-E] H={H:2d} " + "   ".join(
                f"L{e} n{R_of(H, e, n)[1]}: " + "/".join(
                    f"{[q for q in pe_rows if q['H'] == H and q['level'] == e and q['n_span'] == R_of(H, e, n)[1] and q['eps'] == ee][0]['e_piece']:.4f}"
                    for ee in cfg["pe_eps"])
                for e in cfg["pe_levels"] for n in cfg["pe_spans"]))
            save()
        out["pe"] = pe_rows
    out["t_corr"] = time.time() - t0
    save()

    # =========================================================== THE MAIN SWEEP: the era ladder
    # =========================================================== THE MAIN SWEEP: the era ladder
    t0 = time.time()
    frac_ths = dict(q=0.25, t=1.0 / 3.0, h=0.5)

    def arm_table():
        A = [("reflex", "reflex", None, None), ("reflex_ol", "reflex_ol", None, None),
             ("reflex_ec", "reflex_ec", None, None), ("reflex_ec0", "reflex_ec0", None, None)]
        for e in range(1, L + 1):
            A.append((f"key_L{e}", "rec", "key", e))
        for e in range(1, L + 1):
            A.append((f"aud_L{e}", "rec", "audit", e))
        A += [("key_all", "rec", "key", "all"), ("aud_all", "rec", "audit", "all")]
        for p_exp in cfg["scale_exponents"]:
            tg = f"s{int(p_exp)}"
            for e in range(1, L + 1):
                A.append((f"key_{tg}_L{e}", tg, "key", e))
            for e in range(1, L + 1):
                A.append((f"aud_{tg}_L{e}", tg, "audit", e))
        for tag, _sch, _sk, _lp in KIN_ARMS:
            # an arm whose library was never built (its fitted inverse was not requested) is
            # skipped rather than emitted and then crashed on
            if (tag, tempos[0]) not in kin:
                continue
            for e in range(1, L + 1):
                A.append((f"key_{tag}_L{e}", tag, "key", e))
            for e in range(1, L + 1):
                A.append((f"aud_{tag}_L{e}", tag, "audit", e))
        return A

    def run_arm(H, D, nm, kind, mode, ell, appm):
        W, pc, ev = Ws[H], pcs[H], evs[H]
        band, ml = pc.band(), pc.mean_leg()
        W.kp_app, W.kd_app = (gains[H][0] if APP_D0_GAINS[appm] else (None, None))
        W.kp, W.kd = gains[H][D]
        W.obs_predict, W.fm_step = False, None
        led = Ledger(W.dt_ctrl, W.d_fb)
        if kind == "reflex":
            dec = ReflexDecider()
        elif kind == "reflex_ol":
            dec = FrozenReflexDecider(W)
        elif kind in ("reflex_ec", "reflex_ec0"):
            # THE REPORTED REFERENCE ROW, never headlined: the incumbent handed the same body
            # model the learner holds, used ONLY to carry its stale read forward along the
            # commands it already issued (`accompanist` d3b's operator). `reflex_ec` keeps the
            # Delta-fit gains — the same controller, a better read; `reflex_ec0` plays the
            # Delta = 0 gains, which is the controller an un-delayed read licenses. Both are
            # reported so the row cannot be read as either strawman.
            dec = ReflexDecider()
            W.obs_predict = True
            W.fm_step = lambda st, u: fwd_zoh(bc, st, u)
            if kind == "reflex_ec0":
                W.kp, W.kd = gains[H][0]
        else:
            lv = tuple(range(1, L + 1)) if ell == "all" else (int(ell),)
            if kind == "rec":
                lb = libs[H]
            elif kind in ("s0", "s2"):
                lb = scaled[(0.0 if kind == "s0" else 2.0, H)]
            else:
                lb = kin[(kind, H)]
            dec = LadderDecider(W, lb, mode=mode, levels=lv, obs_delay=D)
        o = W.traverse(dec, ev, np.random.default_rng(cfg["seed"] + 35), led,
                       obs_delay=D, obs_delay_app=APP_MODES[appm], collect_obs=True)
        W.obs_predict, W.fm_step = False, None
        ep = np.asarray(o["ep"], np.float64)
        el = np.asarray(o["ep_listen"], np.float64)
        row = dict(H=int(H), delta=int(D), arm=nm, kind=kind, mode=mode, app=appm,
                   level=(None if ell is None else (0 if ell == "all" else int(ell))),
                   e_piece=o["e_piece"], e_listen=o["e_listen"], e_app=o["e_app"],
                   in_band=float(np.mean(ep <= band)), lis_band=float(np.mean(el <= band)),
                   **{f"in_{k2}": float(np.mean(ep <= v * ml)) for k2, v in frac_ths.items()},
                   **{f"lis_{k2}": float(np.mean(el <= v * ml)) for k2, v in frac_ths.items()},
                   e_seg=[float(x) for x in o["e_seg"]],
                   ledger=led.snap(), ledger_drilled=o["ledger_drilled"])
        # DECISION 23: the STALE HAND-OVER at every drilled seam. A level-l arm re-decides at
        # K/2^(l-1) seams — 8 at L1, 4 at L2, 2 at L3, 1 at L4 — and every one of those is a
        # launch from a Delta-old read. `accelerando/fsmoke_a2` measured a deep-best /
        # shallow-worst ordering it could not explain; the candidate mechanism is that depth
        # AMORTISES the stale read, and this is the number that tests it: how wrong the read is
        # at each seam, and how the arm's own per-seam error sits beside it.
        ob = o["obs"]
        rd = []
        for kk in range(W.k0, W.K):
            st = o["seam"][kk]
            rr = ob[:, kk * W.H, :]
            rd.append(dict(seam=int(kk - W.k0),
                           read_dx=float(np.mean(np.linalg.norm(rr[:, :2] - st[:, :2], axis=1))),
                           read_dv=float(np.mean(np.linalg.norm(rr[:, 2:] - st[:, 2:], axis=1))),
                           e_seg=float(o["e_seg"][kk])))
        row["seam_read"] = rd
        row["n_launch"] = int(W.K_drill // (1 if ell in (None, "all") else
                                            (1 << (int(ell) - 1))))
        if mode is not None and getattr(dec, "picks", None):
            row["n_dec"] = len(dec.picks)
            row["n_aud"] = float(np.mean([p["n_aud"] for p in dec.picks]))
            row["mean_level"] = float(np.mean([p["mean_level"] for p in dec.picks]))
            row["n_distinct_slot"] = len({x for p in dec.picks for x in p["slot"]})
        return row

    sweep = []
    D0 = int(cfg["delta_fix"])
    for appm in cfg["app_modes"]:
        for H in tempos:
            # `app = arm` is the CONTINUITY ROW, not a second sweep: `prestissimo` decision 20
            # and `accelerando` flag 1 both measured the delayed lead-in to be an artifact on
            # this body (hand-over 3.3-5.3 legs against 0.18-0.25 under `fixed`), so it is run
            # only where it is asked for and never at every tempo.
            if appm == "arm" and cfg.get("arm_tempos") and int(H) not in cfg["arm_tempos"]:
                continue
            rows = [run_arm(H, D0, nm, kind, mode, ell, appm)
                    for nm, kind, mode, ell in arm_table()]
            sweep += rows
            g = {r["arm"]: r["e_piece"] for r in rows}
            PR(f"[sweep/{appm}] H={H:2d} band={pcs[H].band():.4f}  rf={g['reflex']:.4f} "
               f"ec={g['reflex_ec']:.4f}/{g['reflex_ec0']:.4f}  " + "  ".join(
                   f"{a}={g[a]:.4f}" for a in
                   (f"aud_L{L}", f"aud_s0_L{L}", f"aud_kzs_L{L}", f"aud_kzo_L{L}",
                    f"aud_kcs_L{L}", "aud_L1", "aud_s0_L1", "aud_kzs_L1", "aud_kzo_L1")
                   if a in g))
            save()
    out["sweep"] = sweep
    out["t_sweep"] = time.time() - t0

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
    PR(f"[P-T1] rubato reproduces t1: {pt1['n_checks']} checks, max|delta| = "
       f"{pt1['max_abs']:.3e}  applicable={pt1['applicable']}  pass={pt1['pass']}"
       + ("" if t1_ok else f"  (config differs: {sorted(pt1['cfg_diff'])})"))
    save()
    if pt1["applicable"] and not pt1["pass"]:
        raise SystemExit("P-T1 failed: rubato's substrate is not t1's")

    for W in Ws.values():
        W.close()
    out["complete"] = True
    save()
    PR(f"[done] {out['wall_s']:.0f}s  (library {out['t_library']:.0f} · kin {out['t_kin']:.0f} "
       f"· sweep {out['t_sweep']:.0f})")
    save()
    return out


@app.local_entrypoint()
def rubato_kin(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    r_star: float = T1.T1_R_STAR,
    k_app: int = P.A_K_APP,
    n_levels: int = P.N_LEVELS,
    mass: float = P.FAST_MASS,
    damping: float = P.DAMPING,
    delta_fix: int = P.DELTA_FIX,
    tempos: str = ",".join(str(x) for x in T1.T1_TEMPOS),
    refit_gains: bool = False,
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
    app_modes: str = "fixed",
    arm_tempos: str = "",
    parity_tau: float = 0.75,
    parity_min: int = 4,
    no_parity: bool = False,
    arm_set: str = "ceiling",
    fit_tempos: str = "",
    fit_hold: float = 0.2,
    fit_max_rend: int = 24,
    fit_eval_rend: int = 8,
    mlp_hidden: int = 128,
    mlp_layers: int = 3,
    mlp_epochs: int = 400,
    fit_clean: bool = False,
    corrected: bool = False,
    read_spans: str = "1,2,4,8,16",
    read_spans_fitted: str = "1,4,16",
    read_spans_ctrl: str = "1,16",
    read_fit_r: str = "16",
    pe_levels: str = "1,4",
    pe_spans: str = "1,4,16",
    pe_eps: str = "0,0.02,0.05,0.20",
    diagnostics: bool = False,
    diag_lp: str = "none,aa,note",
):
    tp = [int(x) for x in tempos.split(",") if x]
    kp, kd = P.gain_grid(T_KP_GRID, T_KD_GRID, mass)
    # the diagnostic set drops `scl2`: `ksmoke1` measured it worst everywhere (0.31-0.99 against
    # a 0.0611 band at every unpractised cell), and dropping it pays for the three filtered arms
    # so the diagnostic tag costs what the ceiling tag did.
    sexp = [0.0] if str(arm_set) in ("diag", "learn", "record", "corr") else [0.0, 2.0]
    cfg = dict(tag=tag or "k1", seed=seed, n_proc=n_proc, quick=bool(quick),
               r_star=float(r_star), k_app=k_app, n_levels=n_levels,
               mass=float(mass), damping=float(damping), delta_fix=int(delta_fix),
               tempos=tp, refit_gains=bool(refit_gains),
               scale_exponents=sexp, w_listen=P.W_LISTEN,
               app_modes=[x for x in app_modes.split(",") if x],
               arm_tempos=[int(x) for x in arm_tempos.split(",") if x],
               kp_grid=list(kp), kd_grid=list(kd),
               n_eval=n_eval, n_cal=int(n_cal), n_rt=n_rt, batch=batch, n_warm=n_warm, n_score=n_score,
               n_slot=n_slot, m_cand=m_cand, m_member=m_member, m_weld=m_weld,
               n_poison=n_poison,
               parity_tau=float(parity_tau), parity_min=int(parity_min),
               do_parity=not bool(no_parity), arm_set=str(arm_set),
               do_diag=bool(diagnostics),
               fit_tempos=[int(x) for x in fit_tempos.split(",") if x],
               fit_hold=float(fit_hold), fit_max_rend=int(fit_max_rend),
               fit_eval_rend=int(fit_eval_rend), mlp_hidden=int(mlp_hidden),
               fit_clean=bool(fit_clean), do_corrected=bool(corrected),
               read_spans=[int(x) for x in read_spans.split(",") if x],
               read_spans_fitted=[int(x) for x in read_spans_fitted.split(",") if x],
               read_spans_ctrl=[int(x) for x in read_spans_ctrl.split(",") if x],
               read_ladder_fit_R=[int(x) for x in read_fit_r.split(",") if x],
               pe_levels=[int(x) for x in pe_levels.split(",") if x],
               pe_spans=[int(x) for x in pe_spans.split(",") if x],
               pe_eps=[float(x) for x in pe_eps.split(",") if x],
               mlp_layers=int(mlp_layers), mlp_epochs=int(mlp_epochs),
               diag_lp=[(None if x == "none" else x)
                        for x in diag_lp.split(",") if x],
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5)
    if quick:
        cfg.update(tag=tag or "ksmoke", n_proc=8, n_eval=8, n_rt=8, batch=6, n_warm=2,
                   n_score=4, n_slot=3, m_cand=8, m_member=2, m_weld=2, n_poison=2,
                   tempos=[8, 2], refit_gains=True,
                   kp_grid=list(P.gain_grid((2.0, 10.0, 40.0), (0.25, 1.0, 4.0), mass)[0]),
                   kd_grid=list(P.gain_grid((2.0, 10.0, 40.0), (0.25, 1.0, 4.0), mass)[1]))
    if spawn:
        c = run_kin.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_rubato/{cfg['tag']}/kin.json")
        return
    r = run_kin.remote(cfg)
    print("\n".join(r["log"][-140:]))
