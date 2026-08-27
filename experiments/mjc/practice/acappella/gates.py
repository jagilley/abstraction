"""a cappella — PHASE A. The five gates of `SPEC.md`, before any treatment runs.

  A-F  FORK FIDELITY (SPEC gate 1). Two parts, both exact.
       A-F1 every world/piece/metering constant in `piece.py` equals `etude/etude.py`'s own source,
            read by `ast` (never by import: importing a sibling runner registers its
            `@app.function` / `@app.local_entrypoint` into the one shared Modal app —
            `offbook/FILES.md` Gotcha).
       A-F2/3 the forked `World.traverse` reproduces a VERBATIM transcription of étude's traversal
            inner loop bit-for-bit on a fixed command sequence — states, executed commands,
            per-segment boundary error, feedback count and priced time — with and without motor
            noise (so the RNG consumption order is gated too).

  A-B  THE BUDGET, sized by a rule fixed BEFORE the run (SPEC: "pick by a pre-fixed neutral rule,
       on an arm-neutral reference, before any treatment runs"; `legato/` F2). The reference is the
       priced-rollout planner playing the piece at performance tempo on the held-out geometry with
       NO library in existence, swept over a declared budget ladder. The rule, declared here:

           G* = the SMALLEST budget on the ladder whose reference piece error is within
                `sat_tol` (= 5%) of the BEST piece error anywhere on the ladder.

       Rationale, and why it is neutral: at G* the incumbent has bought essentially everything more
       search can buy it, so no later reading can be dismissed as the incumbent having been starved
       (`ratchet/`'s "the only point where the base arm spends the most", in the continuous-action
       form). The rule mentions only the incumbent's own saturation — it cannot be tuned to make a
       library win, because no library exists when it is evaluated. The ladder around G* is
       reported every time (`ratchet/` §"the declared budget sets the headline").

  A-I  THE INCUMBENT REACHES (SPEC gate 2). At G*, is the priced planner inside étude's `never`
       band (0.10–0.11) or better, and does the reflex law play the piece at all? **Pre-fixed halt:
       if the priced planner is worse than the reflex law at every affordable budget, the round
       stops and says so — the search is then not an incumbent worth beating.**

  A-R  THE RENT IS REAL (SPEC gate 3). A materialisation ladder k = 1 … all over the library, in
       groundings AND priced time AND realised error — so "the library pays rent" is a statement
       about cost and outcome in the same currency.

  A-S  SEAM INFORMATION (SPEC gate 4; `fingering/` G1, `offbook/` G-S). Posture spread ÷ repeat
       noise per seam. Keying is only meaningful where this is > 1. On this plant performance is
       noise-free and deterministic, so the repeat-noise floor is measured against the PRACTICE
       motor noise: the spread of seam states over `n_rep` noisy repeats of ONE start state,
       against the spread over the held-out geometry.

  A-C  AUDITION CALIBRATION ON THE PLANT (SPEC gate 5), both levels.
       A-C0 (structural) seam-time audition is EXACT here: the rollout is the plant, so chosen ==
            realised to machine precision. Asserted, not assumed.
       A-C1 (the real one) the LIBRARY-CONSTRUCTION audition: `chosen_score` on held-out hand-over
            states vs the slot's realised error when consumed at performance. This is where
            étude's winner's curse and seam-state shift live, and they do not go away.

  A-D  CYCLE COST. Measured wall-clock per mode; what Phase B's cycle budget is sized from.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/acappella/gates.py::acappella_gates --quick --tag asmoke
    modal run --detach mjc/practice/acappella/gates.py::acappella_gates --spawn --tag a0 --seed 0
    python3 mjc/practice/acappella/analyze_gates.py --tag a0 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.acappella import piece as P

DONOR = "mjc/practice/etude/etude.py"
G_LADDER = "16,32,64,128,256,512,1024,2048,4096"
R_LADDER = "1,2,4,8,17,34"
REACT_LADDER = "8,32,128"
SAT_TOL = 0.05                     # the pre-fixed saturation tolerance of the budget rule
NEVER_BAND = (0.10, 0.11)          # étude's `never` piece error at performance tempo, 3 seeds


@app.function(cpu=16.0, memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_acappella_gates(cfg: dict) -> dict:
    import ast
    import numpy as np

    from mjc.practice.acappella.world import (World, Ledger, ReflexDecider, SearchDecider,
                                              TapeDecider, LibraryDecider, Library, select_tapes)

    t_start = time.time()
    outdir = os.path.join(DATA_DIR, "practice_acappella", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "gates.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    H = cfg["seg_H"]
    K = P.K_SEG
    n_proc = int(cfg["n_proc"])
    W = World(cfg, n_proc=n_proc, rot=True)
    Wc = World(cfg, n_proc=n_proc, rot=False)          # the CLEAN world: reflex calibration only
    wps = W.wps
    PR(f"[setup] K={K} H={H} n_proc={n_proc} dt_ctrl={W.dt_ctrl:.4f} d_fb={W.d_fb}")

    # ================================================================= A-F1: constants
    t0 = time.time()
    src = None
    for root in ("/root", os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))):
        p = os.path.join(root, DONOR)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        import mjc
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mjc.__file__))), DONOR)
    tree = ast.parse(open(src).read())
    mod_const, fn_def = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                mod_const[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
        if isinstance(node, ast.FunctionDef) and node.name == "etude":
            args = node.args.args
            defs = node.args.defaults
            for a, d in zip(args[len(args) - len(defs):], defs):
                try:
                    fn_def[a.arg] = ast.literal_eval(d)
                except Exception:
                    pass
    want = {
        "DEF_WAYPOINTS": (mod_const.get("DEF_WAYPOINTS"), P.DEF_WAYPOINTS),
        "DEF_REGIONS": (mod_const.get("DEF_REGIONS"), P.DEF_REGIONS),
        "drill_seg": (fn_def.get("drill_seg"), P.DRILL_SEG),
        "seg_h": (fn_def.get("seg_h"), P.SEG_H),
        "frame_skip": (fn_def.get("frame_skip"), P.FRAME_SKIP),
        "arena_half": (fn_def.get("arena_half"), P.ARENA_HALF),
        "gear": (fn_def.get("gear"), P.GEAR),
        "damping": (fn_def.get("damping"), P.DAMPING),
        "box_half": (fn_def.get("box_half"), P.BOX_HALF),
        "start_jit": (fn_def.get("start_jit"), P.START_JIT),
        "v0_std": (fn_def.get("v0_std"), P.V0_STD),
        "explore_sigma": (fn_def.get("explore_sigma"), P.EXPLORE_SIGMA),
        "d_fb": (fn_def.get("d_fb"), P.D_FB),
    }
    bad = {k: v for k, v in want.items() if v[0] != v[1]}
    out["A_F1"] = dict(donor=src, checked=len(want), mismatches={k: list(v) for k, v in bad.items()},
                       **{"pass": not bad})
    PR(f"[A-F1] donor constants: {len(want)} checked, {len(bad)} mismatched  pass={not bad}")
    if bad:
        PR(f"[A-F1] MISMATCH {bad}")
        save()
        raise SystemExit("A-F1 failed: the fork has drifted from the donor")

    # ================================================================= A-F2/3: the donor code path
    def donor_traverse(env, starts, fixed, explore, rng):
        """VERBATIM transcription of `etude.py` §3 `traverse` for a routing of four `fixed` units
        (etude.py lines 376-425). Nothing here may be 'improved'."""
        Bn = starts.shape[0]
        states = starts.copy()
        wp_err, n_fb = [], []
        acts_l, raw_l = [], []
        for ui in range(K):
            hh = H
            goal = np.tile(wps[ui + 1][None, :], (Bn, 1)).astype(np.float32)
            acts = np.empty((Bn, hh, 2), np.float32)
            acts_raw = np.empty((Bn, hh, 2), np.float32)
            plan = np.tile(fixed[ui][None], (Bn, 1, 1)); fb = 1
            for h in range(hh):
                a = plan[:, h, :]
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                acts_raw[:, h, :] = a
                if explore > 0.0:
                    a = np.clip(a + explore * rng.standard_normal(a.shape).astype(np.float32),
                                -1.0, 1.0).astype(np.float32)
                acts[:, h, :] = a
                nxt = np.empty_like(states)
                for b in range(Bn):
                    env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                    s2, _ = env.step(a[b], W.fs); nxt[b] = s2
                states = nxt
            wp_err.append(np.linalg.norm(states[:, :2] - goal, axis=1))
            n_fb.append(fb)
            acts_l.append(acts); raw_l.append(acts_raw)
        total_steps = K * H
        t_piece = total_steps * W.dt_ctrl + float(np.sum(n_fb)) * W.d_fb
        return dict(wp_err=np.stack(wp_err, 1), n_fb=n_fb, t_piece=t_piece,
                    acts=np.concatenate(acts_l, 1), raw=np.concatenate(raw_l, 1),
                    final=np.linalg.norm(states[:, :2] - wps[-1][None, :], axis=1))

    gF = {}
    fixed = np.random.default_rng(cfg["seed"] + 77).uniform(-1, 1, (K, H, 2)).astype(np.float32)
    st = W.start_states(np.random.default_rng(cfg["seed"] + 78), 8)
    for nm, expl in (("noisefree", 0.0), ("noisy", P.EXPLORE_SIGMA)):
        oD = donor_traverse(W.env, st, fixed, expl, np.random.default_rng(cfg["seed"] + 79))
        led = Ledger(W.dt_ctrl, W.d_fb)
        oA = W.traverse(TapeDecider(W, {k: fixed[k] for k in range(K)}), st,
                        np.random.default_rng(cfg["seed"] + 79), led, explore=expl, collect=True)
        d = {k: float(np.max(np.abs(np.asarray(oA[k], np.float64) - np.asarray(oD[k], np.float64))))
             for k in ("wp_err", "acts", "raw", "final")}
        sn = led.snap()
        d["fb_equal"] = bool(abs(sn["n_fb"] - float(np.sum(oD["n_fb"]))) < 1e-12)
        d["t_equal"] = bool(abs(sn["t_exec"] - oD["t_piece"]) < 1e-12)
        d["pass"] = bool(max(v for v in d.values() if isinstance(v, float)) == 0.0
                         and d["fb_equal"] and d["t_equal"])
        gF[nm] = d
        PR(f"[A-F2] fork fidelity ({nm}): max|delta| over 4 arrays = "
           f"{max(v for v in d.values() if isinstance(v, float)):.3e} fb={d['fb_equal']} "
           f"t={d['t_equal']} pass={d['pass']}")
    gF["pass"] = bool(all(gF[n]["pass"] for n in ("noisefree", "noisy")))
    out["A_F2"] = gF
    out["t_A_F"] = time.time() - t0
    save()
    if not gF["pass"]:
        raise SystemExit("A-F2 failed: the forked traversal is not the donor's")

    # ================================================================= setup: the reflex law
    # Gains calibrated ONCE on the CLEAN world (the rotation region absent) — the model-free
    # analogue of étude's `pretrain_mode=exclude`: the hard passage is unmodelled, not mis-modelled.
    # Neutral criterion, fixed before the sweep: piece error (mean of per-segment medians) on the
    # held-out geometry, in the CLEAN world. No treatment exists yet.
    t0 = time.time()
    ev_starts = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
    rt_starts = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])
    grid = []
    for kp in cfg["kp_grid"]:
        for kd in cfg["kd_grid"]:
            Wc.kp, Wc.kd = float(kp), float(kd)
            led = Ledger(W.dt_ctrl, W.d_fb)
            o = Wc.traverse(ReflexDecider(), ev_starts, np.random.default_rng(1), led)
            grid.append(dict(kp=float(kp), kd=float(kd), e_piece=o["e_piece"],
                             e_seg=[float(x) for x in o["e_seg"]]))
    best = min(grid, key=lambda r: r["e_piece"])
    W.kp, W.kd = best["kp"], best["kd"]
    Wc.kp, Wc.kd = best["kp"], best["kd"]
    out["reflex_cal"] = dict(grid=grid, chosen=best,
                             criterion="min piece error on the CLEAN world, held-out geometry")
    PR(f"[cal] reflex gains kp={best['kp']} kd={best['kd']} "
       f"clean e_piece={best['e_piece']:.4f} seg={np.round(best['e_seg'], 4).tolist()}")

    led = Ledger(W.dt_ctrl, W.d_fb)
    o_ref = W.traverse(ReflexDecider(), ev_starts, np.random.default_rng(2), led)
    reflex = dict(e_piece=o_ref["e_piece"], e_seg=[float(x) for x in o_ref["e_seg"]], **led.snap())
    out["reflex"] = reflex
    PR(f"[ref] reflex law in the ROTATED world: e_piece={reflex['e_piece']:.4f} "
       f"seg={np.round(reflex['e_seg'], 4).tolist()} fb={reflex['n_fb']:.0f} "
       f"t={reflex['t_piece']:.2f}s ground={reflex['n_ground']:.0f}")
    out["t_reflex_cal"] = time.time() - t0
    save()

    # ================================================================= A-B: the budget ladder
    t0 = time.time()
    ladder = []
    for G in cfg["g_ladder"]:
        led = Ledger(W.dt_ctrl, W.d_fb)
        dec = SearchDecider(W, G, instrument=True)
        tt = time.time()
        o = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 9000 + G), led)
        row = dict(G=int(G), e_piece=o["e_piece"], e_seg=[float(x) for x in o["e_seg"]],
                   wall_s=time.time() - tt,
                   elite_mean_err=[float(i.get("elite_mean_err", np.nan)) for i in dec.info],
                   best_sampled_err=[float(i.get("best_sampled_err", np.nan)) for i in dec.info],
                   **led.snap())
        ladder.append(row)
        PR(f"[A-B] G={G:5d} e_piece={row['e_piece']:.4f} seg={np.round(row['e_seg'], 4).tolist()} "
           f"ground/perf={row['n_ground']:.0f} t={row['t_piece']:.1f}s wall={row['wall_s']:.1f}s")
        save()
    e_best = min(r["e_piece"] for r in ladder)
    afford = [r for r in ladder if r["e_piece"] <= e_best * (1.0 + SAT_TOL)]
    G_star = int(min(afford, key=lambda r: r["G"])["G"])
    row_star = [r for r in ladder if r["G"] == G_star][0]
    out["A_B"] = dict(ladder=ladder, rule=(f"smallest G within {SAT_TOL:.0%} of the ladder's best "
                                           f"piece error"), sat_tol=SAT_TOL,
                      e_best=e_best, G_star=G_star, chosen=row_star)
    PR(f"[A-B] RULE: smallest G within {SAT_TOL:.0%} of ladder best ({e_best:.4f}) "
       f"-> G* = {G_star}  (e_piece {row_star['e_piece']:.4f}, "
       f"{row_star['n_ground']:.0f} groundings/traversal, {row_star['t_piece']:.1f}s priced)")
    out["t_A_B"] = time.time() - t0
    save()

    # ---- the tempo dimension of the same budget, at MATCHED groundings per traversal ----------
    # "each arm runs the widest search its own action set affords at G" has a second axis on a
    # plant: how OFTEN to decide. R = H plans once per segment (performance tempo, the metering
    # configuration); R < H re-plans the full segment horizon every R steps, paying a fresh
    # feedback event and a fresh budget each time. Held at matched groundings/traversal so this
    # measures allocation, not spend. Instrument: the budget RULE does not read it.
    t0 = time.time()
    tempo = []
    ev_t = ev_starts[:cfg["n_eval_tempo"]]
    for R in cfg["r_ladder"]:
        dps = int(np.ceil(H / R))                       # decisions per segment
        G_R = max(1, int(round(G_star / dps)))
        led = Ledger(W.dt_ctrl, W.d_fb)
        tt = time.time()
        o = W.traverse(SearchDecider(W, G_R, replan_every=R), ev_t,
                       np.random.default_rng(cfg["seed"] + 9500 + R), led)
        tempo.append(dict(R=int(R), G=G_R, decisions_per_seg=dps, e_piece=o["e_piece"],
                          e_seg=[float(x) for x in o["e_seg"]], wall_s=time.time() - tt,
                          **led.snap()))
        PR(f"[A-B*] R={R:3d} G={G_R:5d} e_piece={tempo[-1]['e_piece']:.4f} "
           f"ground/perf={tempo[-1]['n_ground']:.0f} fb={tempo[-1]['n_fb']:.0f} "
           f"t={tempo[-1]['t_piece']:.1f}s wall={tempo[-1]['wall_s']:.1f}s")
        save()
    # ---- and the REACTIVE rung at a real per-decision budget --------------------------------
    # étude's `never` re-planned every 20 ms in imagination at zero price and sat at 0.0089. This
    # is that controller with the imagination replaced by the plant: R = 1, budget G per decision,
    # 136 decisions per traversal. It is the strongest member of the incumbent family and the
    # honest thing to test the reflex law against, so it is measured rather than assumed away.
    for G in cfg["react_ladder"]:
        led = Ledger(W.dt_ctrl, W.d_fb)
        tt = time.time()
        o = W.traverse(SearchDecider(W, G, replan_every=1), ev_t,
                       np.random.default_rng(cfg["seed"] + 9700 + G), led)
        tempo.append(dict(R=1, G=int(G), decisions_per_seg=H, e_piece=o["e_piece"],
                          e_seg=[float(x) for x in o["e_seg"]], wall_s=time.time() - tt,
                          reactive_rung=True, **led.snap()))
        PR(f"[A-B*] R=  1 G={G:5d} e_piece={tempo[-1]['e_piece']:.4f} "
           f"ground/perf={tempo[-1]['n_ground']:.0f} fb={tempo[-1]['n_fb']:.0f} "
           f"t={tempo[-1]['t_piece']:.1f}s wall={tempo[-1]['wall_s']:.1f}s")
        save()
    out["A_B"]["tempo"] = tempo
    out["A_B"]["tempo_note"] = ("first block: matched groundings/traversal to the G* cell; second "
                                "block: R=1 at absolute per-decision budgets (the reactive rung). "
                                f"n_eval={cfg['n_eval_tempo']}; instrument, not read by the "
                                "budget rule")
    out["t_A_B_tempo"] = time.time() - t0

    # ================================================================= A-I: gate 2
    lo, hi = NEVER_BAND
    reach = bool(row_star["e_piece"] <= hi)
    fam = [dict(kind="tempo_H", **r) for r in ladder] + [dict(kind="tempo_R", **r) for r in tempo]
    beats_err = [r for r in fam if r["e_piece"] < reflex["e_piece"]]
    dominates = [r for r in fam if r["e_piece"] < reflex["e_piece"]
                 and r["t_piece"] < reflex["t_piece"]]
    best_cell = min(fam, key=lambda r: r["e_piece"])
    out["A_I"] = dict(
        never_band=list(NEVER_BAND), G_star=G_star,
        e_search_star=row_star["e_piece"], t_search_star=row_star["t_piece"],
        e_reflex=reflex["e_piece"], t_reflex=reflex["t_piece"],
        reaches_band=reach,
        best_incumbent_cell={k: v for k, v in best_cell.items() if k not in ("e_seg", "raw")},
        n_cells_beating_reflex_on_error=len(beats_err),
        n_cells_dominating_reflex=len(dominates),
        beats_reflex_at_any_budget=bool(beats_err),
        halt=bool(not beats_err))
    out["A_I"]["pass"] = bool(reach and beats_err)
    PR(f"[A-I] search@G* {row_star['e_piece']:.4f} (priced {row_star['t_piece']:.0f}s) vs étude "
       f"never band {lo}-{hi} -> reaches={reach}")
    PR(f"[A-I] reflex law {reflex['e_piece']:.4f} (priced {reflex['t_piece']:.1f}s); best search "
       f"cell {best_cell['e_piece']:.4f} at {best_cell['t_piece']:.0f}s "
       f"({best_cell['kind']}, G={best_cell['G']}); cells beating reflex on error "
       f"{len(beats_err)}/{len(fam)}, dominating it {len(dominates)}/{len(fam)}")
    if not beats_err:
        PR("[A-I] *** PRE-FIXED HALT (SPEC gate 2): the priced planner is worse than the reflex "
           "law at every affordable budget AND every tempo. The search is not an incumbent worth "
           "beating. PHASE B IS NOT LICENSED. The remaining Phase A gates below are model-free "
           "instruments that do not presuppose the search, and are completed so the halt is "
           "reported with its context; no treatment arm is run. ***")
        out["halted_at"] = "A-I"
    save()

    # ================================================================= the library (nested)
    # legato/`offbook` d2: candidates for seam k are harvested in the configuration with seams < k
    # already committed — the nesting, not the audition op, is what carries the content.
    t0 = time.time()
    lib = Library(K, cfg["n_slot"], poison=True)
    build = []

    class MixedDecider:
        """Seams < `upto` play their keyed library slot; the rest are reflex. This is 'the
        already-committed configuration' the next seam's candidates are harvested in."""

        def __init__(self, upto):
            self.upto = int(upto)
            self.key = LibraryDecider(W, lib, mode="key", levels=("seg",))
            self.reflex = ReflexDecider()

        def __call__(self, k, states, need, rng, led):
            return (self.key if k < self.upto else self.reflex)(k, states, need, rng, led)

    def harvest(upto, n_cycles, batch, seed0):
        """Practice renditions in the committed configuration. The reflex law makes them, motor
        noise varies them, nothing is planned: zero groundings."""
        pools = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(K)}
        chains = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(K)}
        led = Ledger(W.dt_ctrl, W.d_fb)
        for c in range(n_cycles):
            rng = np.random.default_rng(seed0 + c)
            starts = W.start_states(rng, batch)
            o = W.traverse(MixedDecider(upto), starts, rng, led,
                           explore=cfg["explore_sigma"], collect=True)
            for k in range(K):
                tr = o["traces"][k]
                for f in ("cmds", "raw", "s0", "err"):
                    pools[k][f].append(tr[f])
                ns = K - k
                if ns > 1:
                    chains[k]["cmds"].append(o["acts"][:, k * H:, :].copy())
                    chains[k]["raw"].append(o["raw"][:, k * H:, :].copy())
                    chains[k]["s0"].append(o["seam"][k].copy())
                    chains[k]["err"].append(o["wp_err"][:, k:].mean(1).copy())
        for d in (pools, chains):
            for k in d:
                for f in list(d[k]):
                    d[k][f] = np.concatenate(d[k][f], 0) if len(d[k][f]) else None
        return pools, chains, led

    lb = Ledger(W.dt_ctrl, W.d_fb)
    seam_sets = {}
    for k in range(K):
        pools, chains, hled = harvest(k, cfg["n_warm"], cfg["batch"], cfg["seed"] + 20000 + 97 * k)
        # the score set: hand-over states into seam k from the CURRENT performance configuration
        led = Ledger(W.dt_ctrl, W.d_fb)
        sp = W.traverse(MixedDecider(k), rt_starts, np.random.default_rng(cfg["seed"] + 4400), led)
        S0 = sp["seam"][k][:cfg["n_score"]]
        seam_sets[k] = sp["seam"][k]
        slots, rec = select_tapes(W, pools[k], S0, k, 1, np.random.default_rng(cfg["seed"] + 953 + k),
                                  lb, cfg["n_cand"], cfg["n_slot"])
        for s in slots:
            lib.add(1, k, s["cmds"], s["key"], s["score"])
        rec.update(seam=k, ns=1, level="seg", n_harvest_cycles=cfg["n_warm"],
                   harvest_ledger=hled.snap(),
                   s0_disp=float(np.linalg.norm(S0[:, :2].mean(0) - wps[k])),
                   s0_spread=float(np.linalg.norm(S0[:, :2].std(0))),
                   s0_speed=float(np.linalg.norm(S0[:, 2:], axis=1).mean()))
        build.append(rec)
        PR(f"[lib] seg cell (1,{k}): pool={rec['n_pool']} cand={rec['n_cand']} "
           f"score={rec['chosen_score']:.4f} med={rec['score_med']:.4f} "
           f"curse_counterfactual={rec['score_of_best_own']:.4f} "
           f"s0 disp={rec['s0_disp']:.4f} spread={rec['s0_spread']:.4f} speed={rec['s0_speed']:.3f}")
        save()

    # chain cells, grown over SEGMENT-SLOT SPELLINGS (legato F5) — the lower level's library funds
    # the upper level's addressable variation; the reflex-sourced pool is the declared control.
    chain_rec = []
    for k in range(K):
        ns = K - k
        if ns <= 1:
            continue
        _, chains_spell, _ = harvest(K, cfg["n_warm"], cfg["batch"], cfg["seed"] + 40000 + 97 * k)
        _, chains_ctrl, _ = harvest(k, cfg["n_warm"], cfg["batch"], cfg["seed"] + 60000 + 97 * k)
        S0 = seam_sets[k][:cfg["n_score"]]
        slots, rec = select_tapes(W, chains_spell[k], S0, k, ns,
                                  np.random.default_rng(cfg["seed"] + 1953 + k), lb,
                                  cfg["n_cand"], cfg["n_slot"])
        for s in slots:
            lib.add(ns, k, s["cmds"], s["key"], s["score"])
        _, crec = select_tapes(W, chains_ctrl[k], S0, k, ns,
                               np.random.default_rng(cfg["seed"] + 2953 + k), lb,
                               cfg["n_cand"], cfg["n_slot"])
        rec.update(seam=k, ns=ns, level="chain", source="segment_slot_spellings",
                   control_reflex_sourced=crec)
        chain_rec.append(rec)
        PR(f"[lib] chain cell ({ns},{k}): spellings score={rec['chosen_score']:.4f} "
           f"med={rec['score_med']:.4f} | reflex-sourced control score={crec['chosen_score']:.4f} "
           f"med={crec['score_med']:.4f}")
        save()
    out["library"] = dict(build=build, chains=chain_rec, sizes=lib.sizes(),
                          build_ledger=lb.snap())
    out["t_library"] = time.time() - t0
    PR(f"[lib] sizes {lib.sizes()}  build cost {lb.ground:.0f} groundings, {lb.t:.0f}s priced")
    save()

    # ================================================================= A-C: audition calibration
    t0 = time.time()
    # A-C0: seam-time audition is exact on this plant. Assert it rather than assume it.
    led = Ledger(W.dt_ctrl, W.d_fb)
    dec = LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False, levels=("seg",))
    o = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 31), led)
    chosen = np.array([p["score"] for p in dec.picks], float)          # (K, n_eval)
    realised = o["wp_err"].T                                            # (K, n_eval)
    ac0 = float(np.max(np.abs(chosen - realised)))
    PR(f"[A-C0] seam-time audition exactness: max|chosen - realised| = {ac0:.3e} "
       f"(the rollout IS the plant)")

    # A-C1: the library-construction audition, both levels.
    led = Ledger(W.dt_ctrl, W.d_fb)
    dec_all = LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                             levels=("seg", "chain"))
    o_all = W.traverse(dec_all, ev_starts, np.random.default_rng(cfg["seed"] + 32), led)
    ac1 = []
    for rec in build + chain_rec:
        k, ns = rec["seam"], rec["ns"]
        cell = lib.cell(ns, k)
        led2 = Ledger(W.dt_ctrl, W.d_fb)
        cons = W.traverse(MixedDecider(k), ev_starts, np.random.default_rng(cfg["seed"] + 33), led2)
        S = cons["seam"][k]
        cm = np.tile(np.asarray(cell[0]["cmds"], np.float32)[None], (len(S), 1, 1))
        errs, _ = W.rollout(S, cm, k, ns, led2, charge=False)   # instrument, not charged
        r = float(np.mean(errs.mean(1)))
        ac1.append(dict(seam=k, ns=ns, level=rec["level"], chosen=rec["chosen_score"],
                        realised=r, gap=r / max(rec["chosen_score"], 1e-9)))
        PR(f"[A-C1] {rec['level']:5s} ({ns},{k}): chosen={rec['chosen_score']:.4f} "
           f"realised={r:.4f} gap={ac1[-1]['gap']:.2f}")
    out["A_C"] = dict(exactness_max_abs=ac0, exact=bool(ac0 < 1e-5), calibration=ac1)
    out["t_A_C"] = time.time() - t0
    save()

    # ================================================================= A-R: the rent
    t0 = time.time()
    rent = []
    n_all = max(len(lib.cell(ns, k)) for k in range(K) for ns in lib.levels(k))
    ks = sorted(set([1, 2, 4, cfg["n_slot"]] + [2 * cfg["n_slot"]]))
    ks = [k for k in ks if k <= 2 * cfg["n_slot"]]
    for mode, kk, lv in ([("key", 0, ("seg",))]
                         + [("audit_k", k, ("seg", "chain")) for k in ks]
                         + [("audit_all", 0, ("seg", "chain"))]):
        led = Ledger(W.dt_ctrl, W.d_fb)
        d = LibraryDecider(W, lib, mode=mode, k_prop=kk, budget=G_star,
                           with_prim=(mode != "key"), levels=lv)
        tt = time.time()
        o = W.traverse(d, ev_starts, np.random.default_rng(cfg["seed"] + 34), led)
        row = dict(mode=mode, k=int(kk), levels=list(lv), e_piece=o["e_piece"],
                   topk_order="library insertion order (= construction-score order); no pi in "
                              "Phase A, so this is an OPTIMISTIC top-k",
                   e_seg=[float(x) for x in o["e_seg"]], wall_s=time.time() - tt,
                   n_aud=float(np.mean([p.get("n_aud", 0.0) for p in d.picks])),
                   frac_prim=float(np.mean([p.get("frac_prim", 0.0) for p in d.picks])),
                   frac_chain=float(np.mean([p.get("frac_chain", 0.0) for p in d.picks])),
                   **led.snap())
        rent.append(row)
        PR(f"[A-R] {mode:9s} k={kk:2d}: e_piece={row['e_piece']:.4f} "
           f"ground/perf={row['n_ground']:.1f} t={row['t_piece']:.1f}s "
           f"aud={row['n_aud']:.1f} prim={row['frac_prim']:.2f} chain={row['frac_chain']:.2f}")
        save()
    # the library alone, no primitive at all (what does the vocabulary reach unaided?)
    for lv in (("seg",), ("seg", "chain")):
        led = Ledger(W.dt_ctrl, W.d_fb)
        d = LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False, levels=lv)
        o = W.traverse(d, ev_starts, np.random.default_rng(cfg["seed"] + 35), led)
        rent.append(dict(mode="lib_only", k=0, levels=list(lv), e_piece=o["e_piece"],
                         e_seg=[float(x) for x in o["e_seg"]],
                         n_aud=float(np.mean([p.get("n_aud", 0.0) for p in d.picks])),
                         frac_prim=0.0,
                         frac_chain=float(np.mean([p.get("frac_chain", 0.0) for p in d.picks])),
                         **led.snap()))
        PR(f"[A-R] lib_only {str(lv):18s}: e_piece={rent[-1]['e_piece']:.4f} "
           f"ground/perf={rent[-1]['n_ground']:.1f} t={rent[-1]['t_piece']:.1f}s "
           f"chain={rent[-1]['frac_chain']:.2f}")
    # the two incumbents, on the same table and in the same currency
    rent.append(dict(mode="never_reflex", k=0, levels=[], e_piece=reflex["e_piece"],
                     e_seg=reflex["e_seg"], n_aud=0.0, frac_prim=0.0, frac_chain=0.0,
                     steps=reflex["steps"], n_fb=reflex["n_fb"], n_ground=reflex["n_ground"],
                     ground_steps=reflex["ground_steps"], t_exec=reflex["t_exec"],
                     t_ground=reflex["t_ground"], t_piece=reflex["t_piece"]))
    rent.append(dict(mode="never_search", k=0, levels=[], e_piece=row_star["e_piece"],
                     e_seg=row_star["e_seg"], n_aud=0.0, frac_prim=1.0, frac_chain=0.0,
                     **{f: row_star[f] for f in ("steps", "n_fb", "n_ground", "ground_steps",
                                                 "t_exec", "t_ground", "t_piece")}))
    out["A_R"] = dict(ladder=rent, G_star=G_star)
    out["t_A_R"] = time.time() - t0
    save()

    # ================================================================= A-S: seam information
    # MATCHED conditions on both sides (fingering G1 / offbook G-S): between-start spread and
    # within-start repeat spread are BOTH measured under practice motor noise, so the ratio is
    # "does where you started still show at this seam, above what the noise alone does". The
    # noise-free performance spread is reported alongside, because that is what a key would
    # actually see. Seam 0 is structurally undefined (its state IS the start state, so a repeat of
    # one start has zero spread there) and is reported as such rather than as a huge ratio.
    t0 = time.time()
    led = Ledger(W.dt_ctrl, W.d_fb)
    o_geo = W.traverse(ReflexDecider(), ev_starts, np.random.default_rng(cfg["seed"] + 36), led,
                       explore=cfg["explore_sigma"])
    led = Ledger(W.dt_ctrl, W.d_fb)
    o_perf = W.traverse(ReflexDecider(), ev_starts, np.random.default_rng(cfg["seed"] + 38), led)
    one = np.tile(ev_starts[:1], (cfg["n_rep"], 1))
    led = Ledger(W.dt_ctrl, W.d_fb)
    o_rep = W.traverse(ReflexDecider(), one, np.random.default_rng(cfg["seed"] + 37), led,
                       explore=cfg["explore_sigma"])
    seam_info = []
    for k in range(K):
        A = o_geo["seam"][k]; B = o_rep["seam"][k]; C = o_perf["seam"][k]
        sp = float(np.linalg.norm(A.std(0)))
        nz = float(np.linalg.norm(B.std(0)))
        pos_sp = float(np.linalg.norm(A[:, :2].std(0))); pos_nz = float(np.linalg.norm(B[:, :2].std(0)))
        undef = (k == 0)
        seam_info.append(dict(seam=k, spread=sp, repeat_noise=nz,
                              ratio=(float("nan") if undef else sp / max(nz, 1e-12)),
                              pos_spread=pos_sp, pos_noise=pos_nz,
                              pos_ratio=(float("nan") if undef else pos_sp / max(pos_nz, 1e-12)),
                              perf_spread=float(np.linalg.norm(C.std(0))),
                              perf_pos_spread=float(np.linalg.norm(C[:, :2].std(0))),
                              structurally_undefined=bool(undef),
                              speed=float(np.linalg.norm(C[:, 2:], axis=1).mean())))
        PR(f"[A-S] seam {k}: spread={sp:.4f} repeat_noise={nz:.4f} "
           f"ratio={seam_info[-1]['ratio']:.2f} (pos {pos_sp:.4f}/{pos_nz:.4f} = "
           f"{seam_info[-1]['pos_ratio']:.2f}) perf_spread={seam_info[-1]['perf_spread']:.4f} "
           f"speed={seam_info[-1]['speed']:.3f}"
           + ("   [seam 0: repeat spread is 0 by construction]" if undef else ""))
    # THE PER-STATE ORACLE GAIN — the value of keying, measured (fingering G1: 4.0x; offbook G-S).
    # best single fixed slot for the whole held-out set  vs  the per-state argmin over slots.
    led = Ledger(W.dt_ctrl, W.d_fb)
    d_or = LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False, levels=("seg",))
    W.traverse(d_or, ev_starts, np.random.default_rng(cfg["seed"] + 39), led)
    oracle = []
    for p in d_or.picks:
        M = np.array(p["score_mat"], float)          # (n_eval, n_legal)
        bf = float(M.mean(0).min())                  # the best single fixed slot
        ps = float(M.min(1).mean())                  # the per-state oracle
        oracle.append(dict(seam=int(p["seam"]), best_fixed=bf, per_state=ps,
                           gain=bf / max(ps, 1e-12), n_slot=int(M.shape[1]),
                           n_distinct_argmin=int(len(np.unique(M.argmin(1))))))
        PR(f"[A-S] seam {p['seam']} oracle: best_fixed={bf:.4f} per_state={ps:.4f} "
           f"gain={oracle[-1]['gain']:.2f}x  distinct argmins={oracle[-1]['n_distinct_argmin']}"
           f"/{oracle[-1]['n_slot']}")
    out["A_S"] = dict(seams=seam_info, oracle=oracle,
                      note="between-start and within-start spreads are BOTH under practice motor "
                           "noise (matched); performance is deterministic and noise-free, so its "
                           "spread is reported separately")
    out["t_A_S"] = time.time() - t0
    save()

    # ================================================================= A-D: cycle cost
    out["A_D"] = dict(
        wall_by_phase={k: out.get(k) for k in ("t_A_F", "t_reflex_cal", "t_A_B", "t_library",
                                               "t_A_C", "t_A_R", "t_A_S")},
        search_traversal_wall_by_G={str(r["G"]): r["wall_s"] for r in ladder},
        rent_traversal_wall={f"{r['mode']}_k{r['k']}": r.get("wall_s") for r in rent},
        n_proc=n_proc)
    PR(f"[A-D] wall by phase {json.dumps({k: round(v, 1) for k, v in out['A_D']['wall_by_phase'].items() if v})}")

    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    W.close(); Wc.close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


@app.local_entrypoint()
def acappella_gates(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    g_ladder: str = G_LADDER,
    r_ladder: str = R_LADDER,
    react_ladder: str = REACT_LADDER,
    cem_iters: int = 4,
    cem_init_sigma: float = 0.8,
    cem_elite_frac: float = 0.125,
    vel_pen: float = 0.5,
    kp_grid: str = "2,5,10,20,40,80",
    kd_grid: str = "0.1,0.25,0.5,1,2",
    n_eval: int = 32,
    n_eval_tempo: int = 16,
    n_rt: int = 48,
    n_rep: int = 32,
    batch: int = 24,
    n_warm: int = 8,
    n_cand: int = 24,
    n_score: int = 16,
    n_slot: int = 8,
):
    cfg = dict(
        tag=tag or "a0", seed=seed, n_proc=n_proc,
        seg_H=P.SEG_H, frame_skip=P.FRAME_SKIP, d_fb=P.D_FB,
        explore_sigma=P.EXPLORE_SIGMA,
        g_ladder=[int(x) for x in g_ladder.split(",") if x],
        r_ladder=[int(x) for x in r_ladder.split(",") if x],
        react_ladder=[int(x) for x in react_ladder.split(",") if x],
        cem_iters=cem_iters, cem_init_sigma=cem_init_sigma, cem_elite_frac=cem_elite_frac,
        vel_pen=vel_pen,
        kp_grid=[float(x) for x in kp_grid.split(",") if x],
        kd_grid=[float(x) for x in kd_grid.split(",") if x],
        n_eval=n_eval, n_eval_tempo=n_eval_tempo, n_rt=n_rt, n_rep=n_rep, batch=batch, n_warm=n_warm,
        n_cand=n_cand, n_score=n_score, n_slot=n_slot,
    )
    if quick:
        cfg.update(tag=tag or "asmoke", n_proc=8, g_ladder=[8, 16, 32],
                   r_ladder=[1, 34], react_ladder=[4, 16], n_eval_tempo=4,
                   kp_grid=[5.0, 20.0], kd_grid=[0.25, 1.0],
                   n_eval=6, n_rt=8, n_rep=6, batch=6, n_warm=2,
                   n_cand=4, n_score=4, n_slot=3)
    if spawn:
        # `modal run --detach <file>::<local_entrypoint>` does NOT protect the run: a local
        # entrypoint runs on the CLIENT and blocking it in `.remote()` means the run lives only as
        # long as the launching process. `.spawn()` returns in seconds. (`offbook/FILES.md` Gotcha,
        # which cost two runs.)
        c = run_acappella_gates.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at "
              f"/data/practice_acappella/{cfg['tag']}/gates.json")
        return
    r = run_acappella_gates.remote(cfg)
    print(json.dumps({k: v for k, v in r.items() if k not in ("log", "config")},
                     indent=2, cls=NumpyEncoder)[:8000])
