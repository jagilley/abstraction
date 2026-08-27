"""a cappella — PHASE B1. The Δ ladder on étude's piece, with no forward model anywhere.

WHY THIS ROUND EXISTS. Phase A (`a0`) retired the grounding economy: gate A-I's pre-fixed halt
fired, because 0 of 15 cells of the priced-rollout incumbent family beat the reflex law on error
(best 0.0607 at 15011 s priced against the reflex's 0.0045 at 16.9 s). **That halt stands and is
not retracted.** What A-I's own rent table showed is that on a plant the arms are separated not by
groundings but by FEEDBACK EVENTS — reflex 136, segment library 4, whole-piece chain 1 — which is
the meter's native form here. Re-site decision: Jasper, 2026-08-27, on reading Phase A.

THE OPERATOR. `obs_delay` Δ: every AGENT-SIDE feedback consumer sees the state from Δ control steps
ago — the reflex law's per-step read, the library key at a launch, the seam-time audition's rollout
start, the search's initial state. Experimenter-side instruments stay TRUE-state.
**Nothing bridges it.** `accompanist/` showed the naive delayed operator is a strawman exactly when
a predictor exists (`d3b`: efference copy recovered 6–11x of the naive penalty); with no forward
model anywhere it is the honest operator. `dt_ctrl = 0.024 s`, so Δ = 4/8/16 is 96/192/384 ms
against a 816 ms segment — human proprioceptive loops run ~100 ms and visual ~200–250 ms, so the
middle of the sweep is where a person lives.

  ONE OPTION DELIBERATELY NOT TAKEN, recorded rather than silently omitted: an arm that already
  pays for plant access could bridge Δ by re-executing its own issued commands on a resettable
  copy — efference copy without a forward model. It is not implemented (the round's constraint is
  that nothing may be added, and it is the treatment `d3b` already showed dissolves the niche).
  Note the asymmetry it implies: for the reflex law and the frozen key the naive operator is a
  NECESSITY (they have no plant access at all); for the auditioning arms it is a CHOICE.

THE ARMS, ordered by feedback consumption — the axis `offbook/` finding 7 says delay sensitivity
is ordered by:

  | arm | fb/traversal | groundings/traversal | role |
  |---|---|---|---|
  | `reflex`    | 136 | 0  | the incumbent. Gains RE-FIT per Δ cell on the clean world. |
  | `key_seg`   | 4   | 0  | frozen segment tapes, keyed on the Δ-old posture |
  | `lib_seg`   | 4   | 32 | segment library, seam-time plant audition  — **criterion `e_seg`** |
  | `lib_all`   | ≤4  | 49 | segment+chain library, seam-time plant audition |
  | `key_chain` | 1   | 0  | frozen whole-piece chain, keyed on the Δ-old posture |
  | `aud_chain` | 1   | 8  | whole-piece chain, seam-time audition — **criterion `e_chain`** |
  | `search`    | 4   | 4G | Δ = 0 ONLY, the record rung — not the headline (A-I retired it) |

THE CRITERION AND THE GUARD ARE PRE-FIXED HERE, BEFORE THE RUN (see `FILES.md` §B1).

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/acappella/delay_gate.py::acappella_delay --quick --tag bsmoke
    modal run --detach mjc/practice/acappella/delay_gate.py::acappella_delay --spawn --tag b1 --seed 0
    python3 mjc/practice/acappella/analyze_delay.py --tag b1 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.acappella import piece as P

D_LADDER = "0,1,2,3,4,6,8,12,16"          # d0's range, in control steps. NOT to be extended.
SEARCH_G = "1024,4096"                    # the record rung, Δ = 0 only

# --- the pre-fixed playability anchor ------------------------------------------------------- #
# étude's `never` piece error at performance tempo, 3 seeds (0.1066 +- 0.0059): the donor node's
# own ballistic-per-segment incumbent under a STALE model, undelayed. That is exactly `offbook/`
# d0's `ref_stale` construction, on this very piece, and it is arm-neutral here by construction —
# no arm in this node produces it, and it was published before this node existed.
REF_PLAY = 0.1066
HALF_LEG = 0.4                            # presto's task-anchored second term (leg 0.8 / 2)

# --- cross-tag exact controls against `a0` (offbook's bit-identity idiom) -------------------- #
A0_BUILD_SEG = [0.0575661185200656, 0.05059425708165344, 0.06815679122042403, 0.09710761575940084]
A0_BUILD_CHAIN = [0.1143509246470914, 0.0749743464935404, 0.07725866898369263]
A0_ROWS = {"key_seg": 0.06480563431978226, "lib_seg": 0.039249300956726074,
           "lib_all": 0.03519067168235779, "reflex_a0gains": 0.004462865646928549}
A0_KP, A0_KD = 10.0, 2.0                  # a0's calibrated gains; the LIBRARY BUILD is frozen here


@app.function(cpu=16.0, memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_acappella_delay(cfg: dict) -> dict:
    import ast
    import numpy as np

    from mjc.practice.acappella.world import (World, Ledger, ReflexDecider, SearchDecider,
                                              TapeDecider, LibraryDecider, Library, select_tapes)

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_acappella", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False,
           "criterion": dict(
               rule=("Delta* = the SMALLEST Delta in the pre-declared ladder at which "
                     "e_chain <= e_seg <= e_reflex on piece error, subject to "
                     "min(e_chain, e_seg, e_reflex) <= ref_play"),
               e_chain_arm="aud_chain", e_seg_arm="lib_seg", e_reflex_arm="reflex",
               niche_rule=("THE GATE. Delta*_niche = the SMALLEST Delta at which the BEST "
                           "committed-content arm (min over key_seg, lib_seg, lib_all, key_chain, "
                           "aud_chain) is STRICTLY better than the reflex law on piece error, "
                           "subject to that arm being <= ref_play. This is the pre-fixed niche "
                           "criterion; the 3-term ordering below is the donor's depth test and is "
                           "reported per Delta, distinguishing a SEGMENT-span niche (offbook "
                           "finding 7d) from a CHAIN-span one."),
               seam0_structural_note=(
                   "aud_chain and key_chain decide ONCE, at seam 0, where the piece starts from "
                   "rest and there is no stale history to read — so obs(Delta) is the true state "
                   "there for every Delta and these arms are EXACTLY delay-invariant. That is a "
                   "property of THIS PIECE (etude's loop starts at the first waypoint; offbook's "
                   "had an approach leg before its first seam), not a property of depth. Any "
                   "reading of chain flatness must say so."),
               ref_play=REF_PLAY,
               ref_play_source=("etude `never` piece error at performance tempo, 3-seed mean "
                                "(0.1066 +- 0.0059) — the donor node's ballistic-per-segment "
                                "incumbent under a stale model, undelayed; offbook d0's "
                                "`ref_stale` construction, arm-neutral here by construction"),
               reported_beside=dict(half_leg=HALF_LEG,
                                    note="presto's task-anchored term; NOT used in the verdict "
                                         "because on this piece (leg 0.8) it is 0.4 and would "
                                         "make the guard vacuous"),
               converse_halt=("if the reflex law is strictly best on piece error at EVERY Delta "
                              "in the ladder, B1 reports no niche and B2 is not licensed, "
                              "regardless of the priced ledger"),
               range_is_fixed=("the ladder is not extended or re-tuned if the gate fails "
                               "(legato F2); the pre-declared escalation is the fast piece"))}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "delay.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    H, K = cfg["seg_H"], P.K_SEG
    W = World(cfg, n_proc=int(cfg["n_proc"]), rot=True)
    Wc = World(cfg, n_proc=int(cfg["n_proc"]), rot=False)
    wps = W.wps
    PR(f"[setup] K={K} H={H} n_proc={cfg['n_proc']} dt_ctrl={W.dt_ctrl:.4f} d_fb={W.d_fb} "
       f"Delta ladder={cfg['d_ladder']} ({[round(d * W.dt_ctrl * 1000) for d in cfg['d_ladder']]} ms)")

    # ================================================================= B-F0: the fork, re-gated
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
    out["B_F0"] = dict(checked=len(want), mismatches=bad, **{"pass": not bad})
    PR(f"[B-F0] donor constants: {len(want)} checked, {len(bad)} mismatched  pass={not bad}")
    if bad:
        save()
        raise SystemExit("B-F0 failed: the fork has drifted from the donor")

    ev_starts = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
    rt_starts = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])

    # ================================================================= the library — a0's, exactly
    # d0's construction: the delay is a DECISION constraint, not a learning-data treatment. The
    # library is therefore built ONCE, undelayed, and the harvest reflex is FROZEN at a0's
    # calibrated gains — which is what lets B-F1 assert bit-identity with a0. (The `reflex` ARM
    # below re-fits per Delta on a WIDER grid; that advantage is given to the incumbent only.)
    t0 = time.time()
    W.kp, W.kd = A0_KP, A0_KD
    lib = Library(K, cfg["n_slot"], poison=True)
    build, chain_rec, seam_sets = [], [], {}
    lb = Ledger(W.dt_ctrl, W.d_fb)

    class MixedDecider:
        def __init__(self, upto):
            self.upto = int(upto)
            self.key = LibraryDecider(W, lib, mode="key", levels=("seg",))
            self.reflex = ReflexDecider()

        def __call__(self, k, states, need, rng, led):
            return (self.key if k < self.upto else self.reflex)(k, states, need, rng, led)

    def harvest(upto, n_cycles, batch, seed0):
        pools = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(K)}
        chains = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(K)}
        led = Ledger(W.dt_ctrl, W.d_fb)
        for c in range(n_cycles):
            rng = np.random.default_rng(seed0 + c)
            o = W.traverse(MixedDecider(upto), W.start_states(rng, batch), rng, led,
                           explore=cfg["explore_sigma"], collect=True)
            for k in range(K):
                tr = o["traces"][k]
                for f in ("cmds", "raw", "s0", "err"):
                    pools[k][f].append(tr[f])
                if K - k > 1:
                    chains[k]["cmds"].append(o["acts"][:, k * H:, :].copy())
                    chains[k]["raw"].append(o["raw"][:, k * H:, :].copy())
                    chains[k]["s0"].append(o["seam"][k].copy())
                    chains[k]["err"].append(o["wp_err"][:, k:].mean(1).copy())
        for d in (pools, chains):
            for k in d:
                for f in list(d[k]):
                    d[k][f] = np.concatenate(d[k][f], 0) if len(d[k][f]) else None
        return pools, chains

    for k in range(K):
        pools, chains = harvest(k, cfg["n_warm"], cfg["batch"], cfg["seed"] + 20000 + 97 * k)
        led = Ledger(W.dt_ctrl, W.d_fb)
        sp = W.traverse(MixedDecider(k), rt_starts, np.random.default_rng(cfg["seed"] + 4400), led)
        S0 = sp["seam"][k][:cfg["n_score"]]
        seam_sets[k] = sp["seam"][k]
        slots, rec = select_tapes(W, pools[k], S0, k, 1,
                                  np.random.default_rng(cfg["seed"] + 953 + k), lb,
                                  cfg["n_cand"], cfg["n_slot"])
        for s in slots:
            lib.add(1, k, s["cmds"], s["key"], s["score"])
        rec.update(seam=k, ns=1, level="seg")
        build.append(rec)
    for k in range(K):
        ns = K - k
        if ns <= 1:
            continue
        _, chains_spell = harvest(K, cfg["n_warm"], cfg["batch"], cfg["seed"] + 40000 + 97 * k)
        S0 = seam_sets[k][:cfg["n_score"]]
        slots, rec = select_tapes(W, chains_spell[k], S0, k, ns,
                                  np.random.default_rng(cfg["seed"] + 1953 + k), lb,
                                  cfg["n_cand"], cfg["n_slot"])
        for s in slots:
            lib.add(ns, k, s["cmds"], s["key"], s["score"])
        rec.update(seam=k, ns=ns, level="chain")
        chain_rec.append(rec)
    out["library"] = dict(build=build, chains=chain_rec, sizes=lib.sizes(),
                          build_ledger=lb.snap())
    PR(f"[lib] sizes {lib.sizes()}  build {lb.ground:.0f} groundings")
    out["t_library"] = time.time() - t0

    # ================================================================= B-F1: cross-tag exact ctrl
    # The control is DEFINED against a0's exact configuration: it asserts that the delayed code
    # path at Delta = 0 reproduces a0's library and a0's rows bit-for-bit. Under any other config
    # (a smoke, a different seed) the library is a different object and the control is inapplicable
    # — reported as such, never silently "passed".
    A0_CFG = dict(seed=0, n_eval=32, n_rt=48, batch=24, n_warm=8, n_cand=24, n_score=16, n_slot=8,
                  seg_H=P.SEG_H, explore_sigma=P.EXPLORE_SIGMA)
    applicable = all(cfg.get(k) == v for k, v in A0_CFG.items())
    dseg = max(abs(build[i]["chosen_score"] - A0_BUILD_SEG[i]) for i in range(K))
    dch = max(abs(chain_rec[i]["chosen_score"] - A0_BUILD_CHAIN[i]) for i in range(len(chain_rec)))
    bf1 = {"build_seg_max_abs": dseg, "build_chain_max_abs": dch}
    ARMS = {
        "reflex":    lambda: ReflexDecider(),
        "key_seg":   lambda: LibraryDecider(W, lib, mode="key", levels=("seg",)),
        "key_chain": lambda: LibraryDecider(W, lib, mode="key", levels=("chain",)),
        "lib_seg":   lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("seg",)),
        "lib_all":   lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("seg", "chain")),
        "aud_chain": lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("chain",)),
    }
    for nm in ("key_seg", "lib_seg", "lib_all"):
        led = Ledger(W.dt_ctrl, W.d_fb)
        o = W.traverse(ARMS[nm](), ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                       obs_delay=0)
        bf1[f"{nm}_max_abs"] = abs(o["e_piece"] - A0_ROWS[nm])
    led = Ledger(W.dt_ctrl, W.d_fb)
    o = W.traverse(ReflexDecider(), ev_starts, np.random.default_rng(cfg["seed"] + 2), led)
    bf1["reflex_a0gains_max_abs"] = abs(o["e_piece"] - A0_ROWS["reflex_a0gains"])
    mx = max(v for k, v in bf1.items() if k.endswith("_max_abs"))
    bf1["applicable"] = bool(applicable)
    bf1["pass"] = bool(applicable and mx == 0.0)
    out["B_F1"] = bf1
    PR(f"[B-F1] cross-tag exact control vs a0: max|delta| = {mx:.3e}  "
       f"applicable={applicable}  pass={bf1['pass']}"
       + ("" if applicable else "  [config differs from a0's; control not defined here]"))
    save()
    if applicable and not bf1["pass"]:
        raise SystemExit("B-F1 failed: the delayed code path is not a0's at Delta = 0")

    # ================================================================= anchors, computed blind
    led = Ledger(W.dt_ctrl, W.d_fb)
    zero = {k: np.zeros((H, 2), np.float32) for k in range(K)}
    o0 = W.traverse(TapeDecider(W, zero), ev_starts, np.random.default_rng(7), led)
    out["anchors"] = dict(ref_play=REF_PLAY, half_leg=HALF_LEG, do_nothing=o0["e_piece"],
                          half_do_nothing=0.5 * o0["e_piece"])
    PR(f"[anchor] ref_play={REF_PLAY} (etude never, 3 seeds) | half_leg={HALF_LEG} | "
       f"do-nothing floor={o0['e_piece']:.4f} | half of it={0.5 * o0['e_piece']:.4f}")

    # ================================================================= the reflex, re-fit per Δ
    # Every advantage to the incumbent: gains re-fit at EVERY Delta on the CLEAN world, criterion
    # pre-fixed exactly as in Phase A (min piece error, held-out geometry), on a grid WIDENED past
    # a0's kd edge (a0 chose kd = 2.0, the grid maximum — flagged in FILES.md).
    t0 = time.time()
    gains = {}
    for D in cfg["d_ladder"]:
        grid = []
        for kp in cfg["kp_grid"]:
            for kd in cfg["kd_grid"]:
                Wc.kp, Wc.kd = float(kp), float(kd)
                led = Ledger(W.dt_ctrl, W.d_fb)
                oc = Wc.traverse(ReflexDecider(), ev_starts, np.random.default_rng(1), led,
                                 obs_delay=D)
                grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
        b = min(grid, key=lambda r: r["e_piece"])
        gains[D] = b
        gains[D]["at_kp_edge"] = bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"])))
        gains[D]["at_kd_edge"] = bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))
        PR(f"[cal] Delta={D:2d}  reflex kp={b['kp']:>5} kd={b['kd']:>5}  clean e={b['e_piece']:.4f}"
           f"  edge(kp/kd)={gains[D]['at_kp_edge']}/{gains[D]['at_kd_edge']}")
        save()
    out["reflex_cal"] = {str(k): v for k, v in gains.items()}
    out["t_reflex_cal"] = time.time() - t0

    # ================================================================= the sweep
    t0 = time.time()
    sweep = []
    for D in cfg["d_ladder"]:
        for nm, mk in ARMS.items():
            if nm == "reflex":
                W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
            led = Ledger(W.dt_ctrl, W.d_fb)
            dec = mk()
            o = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                           obs_delay=D)
            row = dict(delta=int(D), ms=round(D * W.dt_ctrl * 1000), arm=nm,
                       e_piece=o["e_piece"], e_seg=[float(x) for x in o["e_seg"]], **led.snap())
            if nm == "reflex":
                row.update(kp=gains[D]["kp"], kd=gains[D]["kd"])
            if hasattr(dec, "picks") and dec.picks:
                row["frac_chain"] = float(np.mean([p.get("frac_chain", 0.0) for p in dec.picks]))
                row["n_aud"] = float(np.mean([p.get("n_aud", 0.0) for p in dec.picks]))
                # THE AUDITION'S OPTIMISM UNDER DELAY. At Delta = 0 the rollout IS the plant, so
                # chosen == realised (A-C0, 1.6e-07). Under delay the audition starts from a state
                # the body has already left, and this is what that costs — measured, not assumed.
                if "score" in dec.picks[0]:
                    ch, rl = [], []
                    for p in dec.picks:
                        for j, b in enumerate(p["need"]):
                            kk, ns = int(p["seam"]), int(p["ns"][j])
                            ch.append(float(p["score"][j]))
                            rl.append(float(o["wp_err"][b, kk:kk + ns].mean()))
                    ch = np.array(ch); rl = np.array(rl)
                    row["aud_chosen"] = float(ch.mean())
                    row["aud_realised"] = float(rl.mean())
                    row["aud_optimism"] = float(rl.mean() / max(ch.mean(), 1e-9))
                    row["aud_max_abs"] = float(np.max(np.abs(rl - ch)))
            sweep.append(row)
        # how far behind is the agent's read, at the seams?
        led = Ledger(W.dt_ctrl, W.d_fb)
        W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
        r = [x for x in sweep if x["delta"] == D]
        PR(f"[B1] Delta={D:2d} ({r[0]['ms']:3d} ms)  "
           + "  ".join(f"{x['arm']}={x['e_piece']:.4f}(fb{x['n_fb']:.1f})" for x in r))
        save()
    # the record rung: the priced search, Delta = 0 only. A-I retired it as the headline.
    for G in cfg["search_g"]:
        led = Ledger(W.dt_ctrl, W.d_fb)
        o = W.traverse(SearchDecider(W, G), ev_starts,
                       np.random.default_rng(cfg["seed"] + 9000 + G), led, obs_delay=0)
        sweep.append(dict(delta=0, ms=0, arm=f"search_G{G}", e_piece=o["e_piece"],
                          e_seg=[float(x) for x in o["e_seg"]], record_rung=True, **led.snap()))
        PR(f"[B1] record rung: search G={G} at Delta=0 -> e_piece={o['e_piece']:.4f} "
           f"({led.snap()['n_ground']:.0f} groundings, {led.snap()['t_piece']:.0f}s priced)")
        save()
    out["sweep"] = sweep
    out["t_sweep"] = time.time() - t0

    # ================================================================= the verdict
    by = {}
    for r in sweep:
        by.setdefault(r["delta"], {})[r["arm"]] = r
    COMMITTED = ("key_seg", "lib_seg", "lib_all", "key_chain", "aud_chain")
    rows, dstar, dniche = [], None, None
    for D in cfg["d_ladder"]:
        c = by[D]["aud_chain"]["e_piece"]
        sg = by[D]["lib_seg"]["e_piece"]
        x = by[D]["reflex"]["e_piece"]
        best_arm = min(COMMITTED, key=lambda a: by[D][a]["e_piece"])
        best_e = by[D][best_arm]["e_piece"]
        niche = bool(best_e < x and best_e <= REF_PLAY)
        ordered = bool(c <= sg <= x)
        playable = bool(min(c, sg, x) <= REF_PLAY)
        rows.append(dict(delta=int(D), ms=round(D * W.dt_ctrl * 1000),
                         e_chain=c, e_seg=sg, e_reflex=x,
                         best_committed_arm=best_arm, best_committed_e=best_e,
                         best_committed_fb=by[D][best_arm]["n_fb"],
                         niche_margin=x - best_e,
                         niche_guard_margin_m=REF_PLAY - best_e,
                         niche=niche,
                         margin_chain_seg=sg - c, margin_seg_reflex=x - sg,
                         guard_margin_m=REF_PLAY - min(c, sg, x),
                         guard_margin_frac=(REF_PLAY - min(c, sg, x)) / REF_PLAY,
                         chain_to_guard_m=REF_PLAY - c,
                         ordered=ordered, playable=playable,
                         passes_ordering=bool(ordered and playable)))
        if niche and dniche is None:
            dniche = int(D)
        if rows[-1]["passes_ordering"] and dstar is None:
            dstar = int(D)
    reflex_best_every = all(
        by[D]["reflex"]["e_piece"] <= min(v["e_piece"] for k, v in by[D].items()
                                          if not k.startswith("search"))
        for D in cfg["d_ladder"])
    base = {r["arm"]: r["e_piece"] for r in sweep if r["delta"] == 0 and not r.get("record_rung")}
    degr = {nm: {"e0": base[nm],
                 "e_max": by[max(cfg["d_ladder"])][nm]["e_piece"],
                 "factor": by[max(cfg["d_ladder"])][nm]["e_piece"] / max(base[nm], 1e-12),
                 "fb": by[0][nm]["n_fb"]} for nm in ARMS}
    out["verdict"] = dict(rows=rows, delta_star_niche=dniche, delta_star_ordering=dstar,
                          converse_halt_fired=bool(reflex_best_every),
                          degradation=degr)
    PR("\n[B1] THE GATE — the niche criterion, per Delta (best committed arm vs the reflex law):")
    PR(f"{'D':>3s} {'ms':>4s} {'best committed':>15s} {'fb':>6s} {'e_best':>8s} "
       f"{'e_reflex':>9s} {'margin':>9s} {'guard m':>9s} {'NICHE':>6s}")
    for r in rows:
        PR(f"{r['delta']:3d} {r['ms']:4d} {r['best_committed_arm']:>15s} "
           f"{r['best_committed_fb']:6.1f} {r['best_committed_e']:8.4f} {r['e_reflex']:9.4f} "
           f"{r['niche_margin']:9.4f} {r['niche_guard_margin_m']:9.4f} {str(r['niche']):>6s}")
    PR("\n[B1] the donor's 3-term depth ordering (chain <= seg <= reflex), per Delta:")
    PR(f"{'D':>3s} {'ms':>4s} {'e_chain':>9s} {'e_seg':>9s} {'e_reflex':>9s} "
       f"{'ch-seg':>9s} {'seg-rfx':>9s} {'guard m':>9s} {'ord':>5s} {'play':>5s} {'PASS':>5s}")
    for r in rows:
        PR(f"{r['delta']:3d} {r['ms']:4d} {r['e_chain']:9.4f} {r['e_seg']:9.4f} "
           f"{r['e_reflex']:9.4f} {r['margin_chain_seg']:9.4f} {r['margin_seg_reflex']:9.4f} "
           f"{r['guard_margin_m']:9.4f} {str(r['ordered']):>5s} {str(r['playable']):>5s} "
           f"{str(r['passes_ordering']):>5s}")
    PR("\n[B1] degradation across the ladder (offbook finding 7: ordered by fb consumption):")
    for nm, v in sorted(degr.items(), key=lambda kv: -kv[1]["fb"]):
        PR(f"  {nm:10s} fb={v['fb']:6.1f}  e(0)={v['e0']:.4f} -> e({max(cfg['d_ladder'])})="
           f"{v['e_max']:.4f}   {v['factor']:.2f}x")
    if dniche is not None:
        r = [q for q in rows if q["delta"] == dniche][0]
        PR(f"\n[B1] *** Delta*_niche = {dniche} ({r['ms']} ms). THE GATE PASSES: committed "
           f"content ({r['best_committed_arm']}, {r['best_committed_fb']:.1f} fb) is strictly best "
           f"at {r['best_committed_e']:.4f} against the reflex's {r['e_reflex']:.4f}, inside "
           f"playability. B2 is licensed at this Delta, subject to the orchestrator's read. ***")
        PR(f"[B1] depth reading: the 3-term ordering "
           + (f"also passes first at Delta = {dstar} — a CHAIN-span niche."
              if dstar is not None else
              "never passes — this is a SEGMENT-span niche (offbook finding 7d), not a chain one."))
    else:
        PR("\n[B1] *** Delta*_niche = None. THE GATE FAILS. The ladder is NOT extended or "
           "re-tuned (legato F2); the pre-declared escalation is the fast piece, separate round. ***")
    if reflex_best_every:
        PR("[B1] *** CONVERSE HALT also fired: the reflex law is strictly best at every Delta. "
           "No niche, and B2 is not licensed. ***")

    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    W.close(); Wc.close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


@app.local_entrypoint()
def acappella_delay(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    d_ladder: str = D_LADDER,
    search_g: str = SEARCH_G,
    kp_grid: str = "2,5,10,20,40,80",
    kd_grid: str = "0.25,0.5,1,2,4,8",
    n_eval: int = 32,
    n_rt: int = 48,
    batch: int = 24,
    n_warm: int = 8,
    n_cand: int = 24,
    n_score: int = 16,
    n_slot: int = 8,
    cem_iters: int = 4,
):
    cfg = dict(tag=tag or "b1", seed=seed, n_proc=n_proc,
               seg_H=P.SEG_H, frame_skip=P.FRAME_SKIP, d_fb=P.D_FB,
               explore_sigma=P.EXPLORE_SIGMA,
               d_ladder=[int(x) for x in d_ladder.split(",") if x],
               search_g=[int(x) for x in search_g.split(",") if x],
               kp_grid=[float(x) for x in kp_grid.split(",") if x],
               kd_grid=[float(x) for x in kd_grid.split(",") if x],
               cem_iters=cem_iters, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5,
               n_eval=n_eval, n_rt=n_rt, batch=batch, n_warm=n_warm,
               n_cand=n_cand, n_score=n_score, n_slot=n_slot)
    if quick:
        cfg.update(tag=tag or "bsmoke", n_proc=8, d_ladder=[0, 4, 16], search_g=[32],
                   kp_grid=[10.0], kd_grid=[2.0], n_eval=8, n_rt=8, batch=6, n_warm=2,
                   n_cand=4, n_score=4, n_slot=3)
    if spawn:
        c = run_acappella_delay.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_acappella/{cfg['tag']}/delay.json")
        return
    r = run_acappella_delay.remote(cfg)
    print("\n".join(r["log"][-60:]))
