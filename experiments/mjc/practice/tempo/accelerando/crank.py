"""accelerando Phase 3 — the crank: mined nested tables under conductor's thermostat, with TEMPO
as the era the loop advances.

WHAT PHASES 1 AND 2 LEFT.
  * `t1`: on a body with tau = 30 ms at a fixed 120 ms delay, a recording harvested AT ITS OWN
    TEMPO beats the per-tempo re-fit reflex by 1.18x / 2.32x / 2.96x / 10.02x as the note shortens
    from 384 to 48 ms, at one feedback event against 128 to 16. **era_1 exists** at H = 2 — the
    incumbent leaves the 1/2-leg band and every level stays inside it — which it never did in
    prestissimo. Seam information is 1.000x at four of five tempi and **1.978x at H = 4**.
  * `p1`: the program does NOT survive a tempo it never practiced. Its parity gate against the
    recording is **0/90 open at H = 4 and 0/54 at H = 2**, against 30/84 at the practiced H = 8;
    the held-out command mse goes 0.000000 (practiced) -> 0.208 / 0.221 (one and two rungs out),
    while `progall`, which practiced there, reads 0.00266 / 0.000018. Where it does open — against
    `scl0`, the only zero-practice content a learner has — the resulting arm is still 0.15-0.24
    against a 0.0611 band, because `scl0` is itself out of band everywhere but its source tempo.

SO THE CRANK RUNS OVER RECORDINGS, AND THE PROGRAM IS AN INSTRUMENT (decision 28). A ratchet needs
content that is valid where it is deployed. At every tempo the recording harvested there is that
content and the program is not, so `rec` is the executor and the head is trained at each era
boundary on the tempi practiced so far and only LOGGED — its parity fraction at the current tempo
and at the next one, i.e. "would this table have survived the advance the loop is about to make?"

THE QUESTION. `conductor` composed a gauge-reading outer loop over RHM's practice learner and
found that the free one-level-up currency (`at_support`) drives the crank at least as well as the
schedule, while the within-level reader (the arm's own error) refuses and starves the level above
it. Phase 3 asks the same question on a motor substrate whose era knob is TEMPO:

    does an outer loop reading its own one-level-up yield drive commit/hold and TEMPO ADVANCE at
    least as well as a schedule — on a ladder whose lower rungs may be scaffolding, where the
    within-level reader has nothing to see?

STANDING CONSTRAINT, verbatim from every node in this arc: **no forward model anywhere.**

THE GATES
  P-F0   the 13 donor constants, `ast`-read out of `etude/etude.py`, never imported.
  P-T1   the level-1 libraries reproduce `t1`'s construction scores at max|delta| = 0 (the same op
         with the same seeds), so the substrate is `t1`'s and `t1`'s own P-F1/P-F2 are inherited.
  P-P    `conductor.policy.policy_gate()` — the donor's 12 offline checks, run in-process.
  P-D    the dead zones are MEASURED by null-ABBA on THIS NODE's own anchor series, never chosen.
  P-Y    every yoked arm is bit-identical to its gauge arm over every logged series.

Run:
    cd experiments/                        # MODAL_PROFILE=chromatic
    modal run mjc/practice/tempo/accelerando/crank.py::accelerando_crank --quick --tag csmoke
    modal run --detach mjc/practice/tempo/accelerando/crank.py::accelerando_crank --spawn --tag c1
    python3 mjc/practice/tempo/accelerando/analyze_crank.py --tag c1 --fetch --figures
"""

import json
import os
import time

import modal  # noqa: F401

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.tempo.accelerando import piece as P
from mjc.practice.tempo.accelerando import t1_pins as T1

APP_D0_GAINS = True          # the lead-in is one pre-computed plan; `app=fixed`, t1's arm of record

# The scheduled crank the gauge arms are measured against, and the clock the dead zones are
# derived on. Fixed before any gauge existed: commit level l+1 after `SCHED_COMMIT` cycles at the
# current level, advance the tempo after `SCHED_ERA` cycles at the current tempo. conductor's
# `anchor` is the same object — a schedule that is a comparator and a replay carrier, not a
# baseline anyone claims is optimal.
SCHED_COMMIT = 6
SCHED_ERA = 10
ERA_CAP_MULT = 1.25          # conductor's: a refusing arm rides its caps to termination


@app.function(cpu=16.0, gpu="L4", memory=65536, timeout=28800, volumes={DATA_DIR: volume})
def run_crank(cfg: dict) -> dict:
    import ast

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np

    from mjc.practice.tempo.accelerando.world import (World, Ledger, ReflexDecider, LevelLibrary,
                                                LadderLayout, LadderDecider, build_level1_cell,
                                                _weld)
    from rhm.practice.conductor.policy import (QuietPolicy, SchedulePolicy, YokePolicy,
                                               null_abba, policy_gate, _sd)

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_accelerando", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "phase": "3",
           "contract": dict(
               no_forward_model=("Nothing in this node imports, trains or evaluates an f(s,u). "
                                 "The optional program instrument emits commands and predicts no "
                                 "state; it never executes."),
               content=("RECORDINGS, harvested at each tempo. `p1` measured the program's parity "
                        "gate at 0/90 and 0/54 open off the practiced tempi against the "
                        "recording, so a program-driven ratchet would be committing content that "
                        "is invalid where it is deployed. The head is trained at each era "
                        "boundary on the tempi practiced so far and LOGGED, never executed."),
               era=("TEMPO is the era the loop advances, and Delta is fixed at 5 steps. The "
                    "loop owns two actions: commit level l+1, and advance the tempo."),
               dead_zones=("Measured by null-ABBA on THIS node's own anchor series (gate P-D), "
                           "never chosen. conductor's rule: a defaulted floor is a chosen one."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "crank.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ================================================================= P-F0
    src = None
    for root in ("/root", os.getcwd()):
        p = os.path.join(root, "mjc/practice/etude/etude.py")
        if os.path.exists(p):
            src = p
            break
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

    # ================================================================= P-P: the donor's policy gate
    t0 = time.time()
    gp = policy_gate(verbose=False)
    ok_pol = bool(gp.get("ALL"))
    out["P_P"] = dict(checks={k: bool(v["pass"]) for k, v in gp.items() if k != "ALL"},
                      n_checks=len([k for k in gp if k != "ALL"]),
                      note="conductor/policy.py's own offline checks, run in-process",
                      **{"pass": ok_pol})
    PR(f"[P-P] conductor's policy gate: {out['P_P']['n_checks']} offline checks, "
       f"pass={ok_pol}  [{time.time() - t0:.1f}s]"
       + ("" if ok_pol else "  FAILED: "
          + str([k for k, v in gp.items() if k != "ALL" and not v["pass"]])))

    # ============================== the level-1 vocabulary, built ONCE per tempo and SHARED
    # The primitive alphabet is a property of the piece and the tempo, not of the arm, so every
    # arm mines above the SAME level-1 tables — which is what makes the arms differ in nothing but
    # what their loop reads (conductor's contract). Levels 2..L are never built statically here:
    # they are MINED.
    t0 = time.time()
    R_star = float(cfg["r_star"])
    tempos = list(cfg["tempos"])
    L = int(cfg["n_levels"])
    Dfix = int(cfg["delta_fix"])
    gains = {int(H): {int(D): tuple(v) for D, v in T1.T1_GAINS[int(H)].items()} for H in tempos}
    Ws, pcs, evs, rts, base = {}, {}, {}, {}, {}
    pt1 = dict(rows=[])
    t1_ok = (int(cfg["seed"]) == 0 and float(cfg["r_star"]) == T1.T1_CFG["r_star"]
             and all(int(cfg[k]) == int(T1.T1_CFG[k]) for k in
                     ("k_app", "n_levels", "n_eval", "n_rt", "batch", "n_warm", "n_score",
                      "n_slot", "m_cand", "m_member", "n_poison"))
             and float(cfg["mass"]) == T1.T1_CFG["mass"])
    for H in tempos:
        pc = P.accel_piece(R_star, H, k_app=cfg["k_app"], n_levels=L, damping=cfg["damping"],
                           mass=cfg["mass"])
        W = World(cfg, pc, n_proc=int(cfg["n_proc"]), rot=True)
        ev = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
        rt = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])
        Ws[H], pcs[H], evs[H], rts[H] = W, pc, ev, rt
        out.setdefault("piece_by_tempo", {})[str(H)] = pc.describe()
        K, k0 = W.K_drill, W.k0
        W.kp, W.kd = gains[H][0]
        lvlib = LevelLibrary(K, cfg["n_slot"], L, poison=True)
        llb = Ledger(W.dt_ctrl, W.d_fb)
        recs = []
        for d in range(K):
            lm = {e: (1,) for e in range(d)}

            def prefix_dec(level_map=lm):
                return LadderDecider(W, lvlib, mode="key", level_map=level_map, fallback=False)
            pools = {k: dict(cmds=[], raw=[], s0=[], err=[], traj=[]) for k in range(W.K)}
            led = Ledger(W.dt_ctrl, W.d_fb)
            for c in range(cfg["n_warm"]):
                rng = np.random.default_rng(cfg["seed"] + 20000 + 97 * d + c)
                o = W.traverse(prefix_dec(), W.start_states(rng, cfg["batch"]), rng, led,
                               explore=pc.explore_sigma, collect=True, collect_traj=True)
                for k in range(W.K):
                    tr = o["traces"][k]
                    for f in ("cmds", "raw", "s0", "err", "traj"):
                        pools[k][f].append(tr[f])
            for k in pools:
                for f in list(pools[k]):
                    pools[k][f] = np.concatenate(pools[k][f], 0) if len(pools[k][f]) else None
            ledS = Ledger(W.dt_ctrl, W.d_fb)
            sp = W.traverse(prefix_dec(), rt, np.random.default_rng(cfg["seed"] + 4400 + d), ledS)
            S0 = sp["seam"][k0 + d][:cfg["n_score"]]
            slots, pois, rec = build_level1_cell(
                W, pools[k0 + d], S0, k0 + d, np.random.default_rng(cfg["seed"] + 77777 + 13 * d),
                llb, cfg["m_cand"], cfg["n_slot"], cfg["m_member"], cfg["n_poison"])
            for g in slots:
                lvlib.add_slot(1, d, g, np.mean([m["key"] for m in g], 0))
            if pois:
                lvlib.add_slot(1, d, pois, np.mean([m["key"] for m in pois], 0), poison=True)
            recs.append(rec["chosen_score"])
        base[H] = lvlib
        if t1_ok and H in T1.T1_L1:
            for i, v in enumerate(recs):
                pt1["rows"].append(dict(what=f"L1@{H}[{i}]", got=v, want=T1.T1_L1[H][i]))
        PR(f"[L1 H={H:2d}] {lvlib.counts()}  {llb.ground:.0f} groundings/perf")
        save()
    pt1["max_abs"] = max([abs(r["got"] - r["want"]) for r in pt1["rows"]], default=-1.0)
    pt1["n_checks"] = len(pt1["rows"])
    pt1["applicable"] = bool(t1_ok)
    pt1["pass"] = bool(t1_ok and pt1["max_abs"] == 0.0 and pt1["n_checks"] > 0)
    out["P_T1"] = pt1
    out["t_l1"] = time.time() - t0
    PR(f"[P-T1] level-1 vocabulary reproduces t1: {pt1['n_checks']} checks, max|delta| = "
       f"{pt1['max_abs']:.3e}  applicable={pt1['applicable']}  pass={pt1['pass']}  "
       f"[{out['t_l1']:.0f}s]")
    save()
    if t1_ok and not pt1["pass"]:
        raise SystemExit("P-T1 failed: Phase 3's substrate is not t1's")

    # =========================================================== the crank, one arm at a time
    def fresh_tables(H):
        """A copy of the tempo's level-1 vocabulary with levels 2..L empty. Mined entries are
        added into it as the loop commits them, so each arm owns its own table history."""
        src_lib = base[H]
        lib = LevelLibrary(src_lib.K, src_lib.n_slot, src_lib.n_levels,
                           poison=src_lib.layout.poison)
        for k in range(src_lib.K):
            for sl in src_lib.slots(1, k, with_poison=True):
                lib.add_slot(1, k, sl["members"], sl["key"], parents=sl["parents"],
                             j=(None if sl["poison"] else sl["j"]), poison=sl["poison"])
        return lib

    def mine(lib, picks_by_seam, solved, ell):
        """THE MINER, `mine_from = chosen`: a level-(ell) candidate at aligned drilled seam d is
        the ORDERED PAIR of slots the agent itself chose at d and at d + span/2, in a traversal
        that was SOLVED. RHM's ratchet mines the beam's own final answer; here the beam's own
        final answer is the pair of units a successful performance actually played.

        Returns {(d, slot_a, slot_b): count}. Nothing is committed here — support is the
        threshold, and `at_support` is the free one-level-up gauge read off this dict.
        """
        span = LadderLayout.span(ell)
        half = span // 2
        cand = {}
        for d in range(0, lib.K, span):
            if d + span > lib.K:
                continue
            pa, pb = picks_by_seam.get(d), picks_by_seam.get(d + half)
            if pa is None or pb is None:
                continue
            for b in range(len(pa)):
                if not solved[b] or pa[b] < 0 or pb[b] < 0:
                    continue
                key = (int(d), int(pa[b]), int(pb[b]))
                cand[key] = cand.get(key, 0) + 1
        return cand

    def n_at_support(counts, sup):
        return int(sum(1 for v in counts.values() if v >= int(sup)))

    def commit_level(W, lib, counts, ell, sup, S0_by_seam, led, rng):
        """Materialise every mined pair at support as a level-`ell` slot: the WELD of its two
        committed parents' members, auditioned on the plant from the seam states the deploying
        configuration produces. `build_level_cell`'s op with the candidate set supplied by the
        MINER instead of by enumeration — which is the whole difference between Phase 1's static
        ladder and a ratchet."""
        span = LadderLayout.span(ell)
        H = W.H
        added, rows = 0, []
        by_seam = {}
        for (d, sa, sb), n in counts.items():
            if n >= int(sup):
                by_seam.setdefault(d, []).append((sa, sb, n))
        for d, prs in sorted(by_seam.items()):
            A = {s["slot"]: s for s in lib.slots(ell - 1, d, with_poison=False)}
            B = {s["slot"]: s for s in lib.slots(ell - 1, d + span // 2, with_poison=False)}
            S0 = S0_by_seam.get(d)
            if S0 is None or len(S0) == 0:
                continue
            cands = []
            for sa, sb, n in sorted(prs, key=lambda x: -x[2]):
                if sa not in A or sb not in B:
                    continue
                a, b = A[sa], B[sb]
                for i in range(min(len(a["members"]), int(cfg["m_member"]))):
                    for j in range(min(len(b["members"]), int(cfg["m_member"]))):
                        cands.append((sa, sb, n, _weld(a["members"][i]["cmds"],
                                                       b["members"][j]["cmds"]),
                                      a["members"][i]["key"]))
            if not cands:
                continue
            cm = np.stack([c[3] for c in cands]).astype(np.float32)
            m = len(S0)
            s0 = np.tile(np.asarray(S0, np.float32)[None], (len(cands), 1, 1)).reshape(-1, 4)
            errs, _ = W.rollout(s0, np.repeat(cm, m, axis=0), W.k0 + d, span, led)
            sc = errs.reshape(len(cands), m, span).mean((1, 2))
            pair_best = {}
            order = np.argsort(sc, kind="stable")
            for t in order:
                pair_best.setdefault((cands[t][0], cands[t][1]), []).append(int(t))
            pairs = sorted(pair_best, key=lambda kk: sc[pair_best[kk][0]])[:int(cfg["n_slot"])]
            for (sa, sb) in pairs:
                idx = pair_best[(sa, sb)][:int(cfg["m_member"])]
                mem = [dict(cmds=cands[t][3], key=cands[t][4], traj=None, score=float(sc[t]),
                            spell=(sa, sb)) for t in idx]
                lib.add_slot(ell, d, mem, mem[0]["key"], parents=(sa, sb))
                added += 1
                rows.append(dict(level=int(ell), drilled=int(d), parents=[int(sa), int(sb)],
                                 support=int([p[2] for p in prs
                                              if p[0] == sa and p[1] == sb][0]),
                                 score=float(sc[idx[0]])))
        return added, rows

    def run_arm(name, policy, prog_log=None):
        """One arm's crank. The loop owns two actions — commit level l+1, advance the tempo — and
        nothing else differs between arms."""
        t_arm = time.time()
        tempo_i, level = 0, 1
        H = tempos[tempo_i]
        libs = {h: fresh_tables(h) for h in tempos}
        counts = {h: {} for h in tempos}
        cycles, actions = [], []
        n_since_commit, n_since_era = 0, 0
        cap_commit = int(round(ERA_CAP_MULT * SCHED_COMMIT))
        cap_era = int(round(ERA_CAP_MULT * SCHED_ERA))
        n_capped = 0
        for cyc in range(int(cfg["n_cycles"])):
            H = tempos[tempo_i]
            W, pc, ev, rt = Ws[H], pcs[H], evs[H], rts[H]
            band = pc.band()
            lib = libs[H]
            lvls = tuple(range(1, level + 1))
            # ---- PRACTICE: a traversal at the current tempo with the tables committed so far.
            W.kp, W.kd = gains[H][0]
            W.kp_app, W.kd_app = gains[H][0]
            led_pr = Ledger(W.dt_ctrl, W.d_fb)
            dec = LadderDecider(W, lib, mode="audit", levels=lvls, obs_delay=0)
            rngc = np.random.default_rng(cfg["seed"] + 500000 + 1013 * cyc)
            o = W.traverse(dec, W.start_states(rngc, cfg["batch"]), rngc, led_pr,
                           explore=pc.explore_sigma, obs_delay=0, obs_delay_app=0)
            ep = np.asarray(o["ep"], np.float64)
            solved = ep <= band
            picks = {}
            for p in dec.picks:
                d = int(p["drilled"])
                arr = np.full(len(ep), -1, int)
                for j, b in enumerate(p["need"]):
                    arr[int(b)] = int(p["slot"][j])
                if d in picks:
                    picks[d] = np.where(arr >= 0, arr, picks[d])
                else:
                    picks[d] = arr
            # a unit spanning several seams leaves the covered seams undecided; fill them from
            # the launching slot so an adjacency at the NEXT level is still countable.
            for d in range(W.K_drill):
                picks.setdefault(d, np.full(len(ep), -1, int))
            # ---- MINE at the level above the committed one
            if level < L:
                c_new = mine(lib, picks, solved, level + 1)
                for k, v in c_new.items():
                    counts[H][k] = counts[H].get(k, 0) + v
            at_sup = n_at_support(counts[H], cfg["mine_support"]) if level < L else 0
            # ---- METER at the fixed delay: the arm's own error (the within-level read)
            W.kp, W.kd = gains[H][Dfix]
            W.kp_app, W.kd_app = gains[H][0]
            led_me = Ledger(W.dt_ctrl, W.d_fb)
            dec_m = LadderDecider(W, lib, mode="audit", levels=lvls, obs_delay=Dfix)
            om = W.traverse(dec_m, ev, np.random.default_rng(cfg["seed"] + 35), led_me,
                            obs_delay=Dfix, obs_delay_app=0)
            epm = np.asarray(om["ep"], np.float64)
            row = dict(cycle=int(cyc), H=int(H), level=int(level), at_support=int(at_sup),
                       n_cand=int(len(counts[H])),
                       solved_frac=float(np.mean(solved)),
                       e_practice=float(o["e_piece"]),
                       e_piece=float(om["e_piece"]), e_listen=float(om["e_listen"]),
                       in_band=float(np.mean(epm <= band)),
                       mean_level=float(np.mean([p["mean_level"] for p in dec_m.picks])),
                       n_fb=float(om["ledger_drilled"]["n_fb"]),
                       counts=dict(lib.counts()))
            # ---- THE READS, in error convention (falling = the agent is earning)
            row["support_mass"] = int(sum(counts[H].values()))
            reads = {"ledger": row["e_piece"], "ledger_level": int(level),
                     "yield": -float(at_sup), "yield_level": int(level + 1)}
            info = policy.step(cyc, reads)
            row["policy"] = {k: info.get(k) for k in ("D", "V", "v_tol", "quiet", "moved",
                                                      "n_since", "in_burn", "v_mult")}
            n_since_commit += 1
            n_since_era += 1
            act = None
            want = bool(info.get("quiet")) if policy.kind != "schedule" else None
            if policy.kind == "schedule":
                want_commit = n_since_commit >= SCHED_COMMIT
                want_era = n_since_era >= SCHED_ERA
            elif policy.kind == "yoke":
                want_commit = policy.commit_now(cyc)
                want_era = policy.advance_now(cyc)
            else:
                want_commit = want_era = bool(want)
            # caps: a refusing arm rides them to termination, as in conductor
            capped = False
            if not want_commit and n_since_commit >= cap_commit and level < L:
                want_commit, capped = True, True
            if not want_era and n_since_era >= cap_era:
                want_era, capped = True, True
            if want_commit and level < L:
                S0_by = {}
                for d in range(0, W.K_drill, LadderLayout.span(level + 1)):
                    S0_by[d] = om["seam"][W.k0 + d][:cfg["n_score"]]
                led_c = Ledger(W.dt_ctrl, W.d_fb)
                added, mrows = commit_level(W, lib, counts[H], level + 1, cfg["mine_support"],
                                            S0_by, led_c, rngc)
                if added:
                    level += 1
                    act = "commit"
                    n_since_commit = 0
                    n_capped += int(capped)
                    actions.append(dict(kind="commit", cycle=int(cyc), level=int(level),
                                        H=int(H), n_added=int(added), capped=bool(capped),
                                        at_support=int(at_sup), rows=mrows))
                    policy.acted("commit", cyc)
            elif want_era and tempo_i + 1 < len(tempos):
                if prog_log is not None:
                    prog_log(name, tempo_i, libs, level, cyc)
                tempo_i += 1
                act = "advance"
                n_since_era = 0
                n_capped += int(capped)
                actions.append(dict(kind="advance", cycle=int(cyc), H=int(tempos[tempo_i]),
                                    level=int(level), capped=bool(capped)))
                policy.acted("advance", cyc)
                # a table is per-tempo: the new tempo starts from its own level-1 vocabulary,
                # and the level the loop has committed carries over as a CLAIM that has to be
                # re-earned by mining there. That is the ratchet's exposure to the era knob.
                level = 1
            row["action"] = act
            cycles.append(row)
        return dict(name=name, cycles=cycles, actions=actions, wall=time.time() - t_arm,
                    final_tempo=int(tempos[tempo_i]), final_level=int(level),
                    n_capped=int(n_capped), tables={str(h): dict(libs[h].counts())
                                                    for h in tempos},
                    at_support_final={str(h): n_at_support(counts[h], cfg["mine_support"])
                                      for h in tempos})

    # ------------------------------------------------- Phase 0: the anchor, and the dead zones
    t0 = time.time()
    arms = {}
    arms["anchor"] = run_arm("anchor", SchedulePolicy())
    out["arms"] = arms
    out["t_anchor"] = time.time() - t0
    a = arms["anchor"]
    PR(f"[anchor] {len(a['cycles'])} cycles, ended H={a['final_tempo']} L={a['final_level']}, "
       f"{len(a['actions'])} actions ({sum(1 for x in a['actions'] if x['kind'] == 'commit')} "
       f"commits, {sum(1 for x in a['actions'] if x['kind'] == 'advance')} advances)  "
       f"[{out['t_anchor']:.0f}s]")
    save()

    # ================================================== P-D: the dead zones, MEASURED, not chosen
    skip = tuple(int(x["cycle"]) for x in a["actions"])
    floors, v_tol = {}, {}
    for key, series in (("ledger", [c["e_piece"] for c in a["cycles"]]),
                        ("yield", [-float(c["at_support"]) for c in a["cycles"]])):
        Ns, Ds = null_abba(series, skip=skip, span=int(cfg["span"]), W=int(cfg["W"]))
        sdN, sdD = _sd(Ns), _sd(Ds)
        # policy.py's own derivation: for iid quarter improvements sd(N) = 2 sd(D), so the dead
        # zone for the decision statistic D is sd(N)/2. Never chosen — measured here.
        tol = (sdN / 2.0) if sdN else 0.0
        floors[key] = dict(n_windows=len(Ns), sd_N=sdN, sd_D=sdD, v_tol=tol,
                           ratio=(sdN / sdD if (sdN and sdD) else None))
        v_tol[key] = float(tol)
    out["P_D"] = dict(floors=floors, skip=list(skip),
                      method=("null-ABBA on the ANCHOR's own series, with the action cycles "
                              "skipped: on a fixed-condition series the donor's contrast has "
                              "true value zero, so its spread is this tag's own noise floor, "
                              "and the dead zone for D is sd(N)/2."))
    out["P_D"]["v_tol"] = v_tol
    degen = sorted(k for k, v in v_tol.items() if not v > 0)
    out["P_D"]["degenerate"] = degen
    out["P_D"]["pass"] = bool(not degen)
    out["P_D"]["note_degenerate"] = (
        "A floor of exactly 0 means the anchor's own series for that read did not move at all "
        "over any admissible window — the gauge has no instrument noise because it has no "
        "signal. conductor's rule is that a defaulted floor is a chosen one, so such a read is "
        "NOT given a substitute: its arm is flagged `floor_degenerate` and its trace is read as "
        "'this currency never armed in this tag', which is conductor's own verdict on its raw "
        "`endo` read (mean(D)/floor -0.14, it never armed anywhere).")
    PR(f"[P-D] measured dead zones (null-ABBA on this node's anchor series, "
       f"{floors['ledger']['n_windows']} windows): "
       + json.dumps({k: round(v, 6) for k, v in v_tol.items()})
       + f"   sd(N)/sd(D) = "
       + json.dumps({k: (None if floors[k]['ratio'] is None else round(floors[k]['ratio'], 3))
                     for k in floors})
       + f"  pass={out['P_D']['pass']}"
       + (f"  DEGENERATE: {degen}" if degen else ""))
    save()

    # ------------------------------------------------------------------ the program instrument
    prog_rows = []

    def prog_log(name, tempo_i, libs, level, cyc):
        prog_rows.append(dict(arm=name, at_cycle=int(cyc), from_H=int(tempos[tempo_i]),
                              to_H=int(tempos[tempo_i + 1]), level=int(level),
                              note=("logged only; p1 measured the head's parity at 0/90 and "
                                    "0/54 open one and two rungs past its practiced set, which "
                                    "is why the crank runs over recordings (decision 28)")))

    # ------------------------------------------------------------------ the driven arms + yokes
    for nm, key in (("outer_yield", "yield"), ("outer_ledger", "ledger")):
        t0 = time.time()
        if key in degen:
            arms[nm] = dict(name=nm, floor_degenerate=True, cycles=[], actions=[],
                            note=out["P_D"]["note_degenerate"])
            PR(f"[{nm}] NOT RUN: its read's measured dead zone is 0 — the gauge never moved "
               f"above its own instrument noise in the anchor series. Reported, not substituted.")
            continue
        pol = QuietPolicy(key, v_tol=v_tol[key], span=int(cfg["span"]), W=int(cfg["W"]),
                          burn=int(cfg["burn"]))
        arms[nm] = run_arm(nm, pol, prog_log=prog_log)
        arms[nm]["policy_trace"] = pol.trace
        r = arms[nm]
        PR(f"[{nm}] {len(r['cycles'])} cycles, ended H={r['final_tempo']} L={r['final_level']}, "
           f"{sum(1 for x in r['actions'] if x['kind'] == 'commit')} commits / "
           f"{sum(1 for x in r['actions'] if x['kind'] == 'advance')} advances, "
           f"{r['n_capped']} capped  [{time.time() - t0:.0f}s]")
        save()
    for nm in ("outer_yield", "outer_ledger"):
        if arms[nm].get("floor_degenerate"):
            continue
        plan = [dict(kind=x["kind"], cycle=x["cycle"]) for x in arms[nm]["actions"]]
        yk = f"yoked_{nm.split('_')[1]}"
        arms[yk] = run_arm(yk, YokePolicy(plan))
        PR(f"[{yk}] replays {nm}'s clock: "
           f"{sum(1 for x in arms[yk]['actions'] if x['kind'] == 'commit')} commits / "
           f"{sum(1 for x in arms[yk]['actions'] if x['kind'] == 'advance')} advances")
        save()

    # ================================================================= P-Y: the yokes are inert
    py = dict(rows=[], max_abs=0.0)
    for nm in ("outer_yield", "outer_ledger"):
        yk = f"yoked_{nm.split('_')[1]}"
        if yk not in arms:
            continue
        for f in ("e_piece", "e_listen", "at_support", "solved_frac", "mean_level"):
            va = [c[f] for c in arms[nm]["cycles"]]
            vb = [c[f] for c in arms[yk]["cycles"]]
            n = min(len(va), len(vb))
            mx = max([abs(va[i] - vb[i]) for i in range(n)], default=0.0)
            py["rows"].append(dict(arm=nm, field=f, n=n, max_abs=mx))
            py["max_abs"] = max(py["max_abs"], mx)
    py["pass"] = bool(py["max_abs"] == 0.0)
    out["P_Y"] = py
    PR(f"[P-Y] yoked arms bit-identical to their gauge arms over "
       f"{len(py['rows'])} series: max|delta| = {py['max_abs']:.3e}  pass={py['pass']}")

    # ============================================ the battery: adoption and can't-decompose at L2
    t0 = time.time()
    bat = []
    for nm in arms:
        r = arms[nm]
        if r.get("floor_degenerate"):
            continue
        H = r["final_tempo"]
        W, pc, ev = Ws[H], pcs[H], evs[H]
        band = pc.band()
        W.kp, W.kd = gains[H][Dfix]
        W.kp_app, W.kd_app = gains[H][0]
        # rebuild this arm's final table at its final tempo from its own commit record, so the
        # battery is asked of the table the crank actually earned
        lib = fresh_tables(H)
        for act in r["actions"]:
            if act["kind"] == "commit" and int(act["H"]) == int(H):
                for mr in act.get("rows", []):
                    ell, d = int(mr["level"]), int(mr["drilled"])
                    sa, sb = mr["parents"]
                    A = {q["slot"]: q for q in lib.slots(ell - 1, d, with_poison=False)}
                    B = {q["slot"]: q for q in
                         lib.slots(ell - 1, d + LadderLayout.span(ell) // 2, with_poison=False)}
                    if sa not in A or sb not in B:
                        continue
                    mem = [dict(cmds=_weld(A[sa]["members"][0]["cmds"],
                                           B[sb]["members"][0]["cmds"]),
                                key=A[sa]["members"][0]["key"], traj=None,
                                score=float(mr["score"]), spell=(sa, sb))]
                    lib.add_slot(ell, d, mem, mem[0]["key"], parents=(sa, sb))
        lvl = max([1] + [e for e in range(1, L + 1)
                         for d in range(0, W.K_drill, LadderLayout.span(e))
                         if lib.slots(e, d, with_poison=False)])
        row = dict(arm=nm, H=int(H), level_committed=int(r["final_level"]),
                   level_in_table=int(lvl), tables=dict(lib.counts()))
        # THE ADDRESS-BOOK BATTERY, per level: what the table is worth when only that rung is
        # legal, at the fixed delay, against the incumbent and the do-nothing floor.
        for e in range(1, lvl + 1):
            if not any(lib.slots(e, d, with_poison=False)
                       for d in range(0, W.K_drill, LadderLayout.span(e))):
                continue
            led = Ledger(W.dt_ctrl, W.d_fb)
            o = W.traverse(LadderDecider(W, lib, mode="audit", levels=(e,), obs_delay=Dfix,
                                         fallback=True),
                           ev, np.random.default_rng(cfg["seed"] + 35), led,
                           obs_delay=Dfix, obs_delay_app=0)
            row[f"L{e}_e"] = o["e_piece"]
            row[f"L{e}_listen"] = o["e_listen"]
            row[f"L{e}_in"] = float(np.mean(np.asarray(o["ep"]) <= band))
            row[f"L{e}_fb"] = o["ledger_drilled"]["n_fb"]
        led = Ledger(W.dt_ctrl, W.d_fb)
        orf = W.traverse(ReflexDecider(), ev, np.random.default_rng(cfg["seed"] + 35), led,
                         obs_delay=Dfix, obs_delay_app=0)
        row["reflex_e"] = orf["e_piece"]
        row["reflex_listen"] = orf["e_listen"]
        row["reflex_fb"] = orf["ledger_drilled"]["n_fb"]
        # CAN'T-DECOMPOSE AT LEVEL 2: a level-2 unit and its own two parents played in sequence
        # execute IDENTICALLY on a deterministic plant (gate P-N, 1e-07). What the level changes
        # is WHEN the second half was chosen — at the first seam, with the hand-over known at
        # construction time, instead of read Delta-old at the second. So the readout is the
        # DECISION difference under delay, and the feedback ledger is the price of it.
        if lvl >= 2 and "L2_e" in row and "L1_e" in row:
            row["decompose_gap"] = row["L1_e"] - row["L2_e"]
            row["decompose_fb"] = row["L1_fb"] - row["L2_fb"]
        bat.append(row)
        PR(f"[battery/{nm:<13s}] H={H:2d} table {row['tables']}  " + "  ".join(
            f"L{e}={row[f'L{e}_e']:.4f}(fb {row[f'L{e}_fb']:.0f})"
            for e in range(1, lvl + 1) if f"L{e}_e" in row)
           + f"   reflex={row['reflex_e']:.4f}(fb {row['reflex_fb']:.0f})  band={band:.4f}")
    out["battery"] = bat
    out["program_instrument"] = prog_rows
    out["t_battery"] = time.time() - t0

    out["flags"] = dict(
        content_is_recordings=("Decision 28. `p1` measured the program's parity gate at 0/90 and "
                               "0/54 open one and two rungs past its practiced set against the "
                               "recording, and the arm that respects its deployable gate still "
                               "0.15-0.24 against a 0.0611 band. A ratchet commits content it "
                               "will deploy; at every tempo that content is the recording."),
        no_learned_router=("pi over slots is NOT learned here, and the reason is measured: `t1`'s "
                           "P-S is 1.000x with 1 of 6 distinct argmins at four of five tempi, and "
                           "`p1`'s capture found n_distinct_member = 1.00 at the same four. A "
                           "learned router would have nothing to select on. The decider is `key`/"
                           "`audit`, as in Phase 1, and H = 4 (P-S 1.978x) is where a router "
                           "would first have something to do."),
        table_per_tempo=("A table is per tempo. On a tempo advance the committed LEVEL carries "
                         "over as a claim that has to be re-earned by mining at the new tempo, "
                         "because `t1` measured a recording to be worth 10-36x less at any tempo "
                         "but its own. That exposure is the point of putting tempo on the era "
                         "axis."))
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
def accelerando_crank(
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
    n_cycles: int = 80,
    mine_support: int = 3,
    span: int = 1,
    w_quarters: int = 4,
    burn: int = 4,
    n_eval: int = 32,
    n_rt: int = 48,
    batch: int = 24,
    n_warm: int = 8,
    n_score: int = 16,
    n_slot: int = 6,
    m_cand: int = 48,
    m_member: int = 4,
    n_poison: int = 3,
):
    cfg = dict(tag=tag or "c1", seed=seed, n_proc=n_proc, quick=bool(quick),
               r_star=float(r_star), k_app=k_app, n_levels=n_levels,
               mass=float(mass), damping=float(damping), delta_fix=int(delta_fix),
               tempos=[int(x) for x in tempos.split(",") if x],
               n_cycles=int(n_cycles), mine_support=int(mine_support),
               span=int(span), W=int(w_quarters), burn=int(burn),
               n_eval=n_eval, n_rt=n_rt, batch=batch, n_warm=n_warm, n_score=n_score,
               n_slot=n_slot, m_cand=m_cand, m_member=m_member, n_poison=n_poison,
               w_listen=P.W_LISTEN,
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5)
    if quick:
        cfg.update(tag=tag or "csmoke", n_proc=8, n_eval=8, n_rt=8, batch=6, n_warm=2,
                   n_score=4, n_slot=3, m_cand=8, m_member=2, n_poison=2,
                   tempos=[16, 8, 2], n_cycles=14, burn=2)
    if spawn:
        c = run_crank.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_accelerando/{cfg['tag']}/crank.json")
        return
    r = run_crank.remote(cfg)
    print("\n".join(r["log"][-120:]))
