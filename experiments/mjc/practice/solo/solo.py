"""solo — re-internalization of committed motor macros on the pusher, with NO FORWARD MODEL.

THE QUESTION. `rhm/practice/native/`'s consolidation — a committed vocabulary PROPOSED by the
learner's own routing head, EMITTED by a corridor head on the learner's own executor, the table
outside and deletable, trust formed only by use — on a plant, with nothing imaginary in the loop,
at the delay where committed content is measured to pay (`acappella/b1`: Delta = 8 control steps
= 192 ms, where stored segment tapes at 4 feedback events beat a per-Delta re-fit reflex at 136
by 2.0x).

STANDING CONSTRAINT, verbatim from `acappella/`: **no forward model anywhere in this node, not
even as an extra control or reference arm.** Nothing here imports, trains or evaluates an
`f(s,u)`. The behaviour-cloned trunk predicts no state; it reproduces what the reflex law would
COMMAND from the read it has. If the design ever seems to demand a model, that is a finding to
halt on, not a gap to patch.

THE THREE THINGS THIS NODE ADDS TO `acappella/`
  1. A TRUNK WITHOUT A MODEL — the reflex law behaviour-cloned into a small policy net, gated to
     reproduce its teacher before any port is wired (gate S-T1). It is both the primitive every
     seam may choose and the trunk Port 2's span head reads.
  2. THE TWO PORTS — pi over library slots (Port 1) and a span head on the cloned trunk (Port 2),
     the latter behind a per-slot parity gate measured as execution reproduction ON THE PLANT.
  3. THE ADDRESS-BOOK BATTERY — `no_table` / `no_prim` / `no_table_no_prim` / `restored`, with
     `fid`'s untrained heads as the negative control.

THE ARMS (all at Delta = 8 unless noted)
  | arm             | decision rule                                      | what it isolates |
  |---|---|---|
  | `reflex`        | the reflex law THROUGH THE TRUNK, closed loop       | the reference    |
  | `key_seg`       | frozen posture key, no audition, 0 groundings       | committed content, unrouted |
  | `audit_all`     | audition every member of every slot + the primitive | the enumeration reference |
  | `audit_prop_k`  | PORT 1: pi's top-k slots auditioned                 | routing |
  | `route_native`  | PORTS 1 + 2                                         | + corridor |
  | `fid`           | both ports wired and SHUT                           | must be identical to `audit_all` |
  | `audit_prop_kN` | pi live at k = every legal slot                     | must be identical to `audit_all` |
  | `prop_k_d0`     | `audit_prop_k` at Delta = 0                         | the adoption control: does the learner KEEP playing by feel where feel pays? |

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/solo/solo.py::solo_run --quick --tag ssmoke
    modal run --detach mjc/practice/solo/solo.py::solo_run --spawn --tag s0 --seed 0
    python3 mjc/practice/solo/analyze_solo.py --tag s0 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.solo import piece as P

# --- cross-tag exact controls against `acappella/b1` (offbook's bit-identity idiom) ---------- #
# Full-precision values read off `acappella/results/b1/delay.json`. The control is DEFINED only
# under b1's own configuration; under any other config the library is a different object and the
# control is reported INAPPLICABLE, never silently "passed" (acappella's B-F1 discipline).
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
B1_GAINS = {0: (10.0, 2.0), 8: (5.0, 4.0)}      # the per-Delta re-fit, on b1's widened grid
A0_KP, A0_KD = 10.0, 2.0                        # the HARVEST gains: frozen, as in b1
B1_CFG = dict(seed=0, n_eval=32, n_rt=48, batch=24, n_warm=8, n_cand=24, n_score=16, n_slot=8,
              seg_H=P.SEG_H, explore_sigma=P.EXPLORE_SIGMA)
# --- the two PRE-FIXED absolute competence bands (round 2, `s1q`) --------------------------- #
# Both are published, arm-neutral references on THIS piece at Delta = 8, fixed before the probe and
# never chosen by an outcome.
#   REF_PLAY  — etude's `never` at performance tempo, 3-seed mean; acappella's own playability guard,
#               used blind there and published before any node in this arc existed.
#   BAND_LIB  — the donor 8-tape `lib_seg` at Delta = 8 from acappella b1, re-derived at 0.000e+00
#               by gate S-F1 in `s0`: a competent-EXECUTION reference on the same piece and delay.
REF_PLAY = 0.1066
BAND_LIB = 0.08381044864654541          # == B1_ROWS[8]["lib_seg"]
# Config keys that must agree for a cross-tag control against the reference run to be DEFINED.
# `deltas`, `arms`, `n_cycles`, `tag` and the reference-run knobs are deliberately excluded: the
# arms are independent of one another (every head is minted through `isolated_rng`/`build_span`,
# every trainer carries its own numpy stream, and every per-cycle stream is seeded by the cycle
# index), so running fewer arms for fewer cycles cannot move an arm that both runs share.
S0_KEYS = ("seed", "n_eval", "n_rt", "batch", "n_warm", "n_cand", "n_score", "n_slot", "m_cand",
           "m_member", "n_poison", "batch_pr", "k_prop", "eps_act", "explore_eps", "force_window",
           "prop_hidden", "prop_layers", "prop_lr", "prop_cap", "prop_steps", "prop_batch",
           "span_hidden", "span_layers", "span_lr", "span_steps", "span_batch", "span_cap",
           "span_hold_cap", "span_hold_frac", "parity_tau", "parity_every", "parity_min",
           "probe_every", "seg_H", "explore_sigma", "bc_cycles", "bc_batch", "bc_hidden",
           "bc_layers", "bc_lr", "bc_steps", "bc_batch_sz", "bc_explore", "dperf_ema", "dperf_sil")
KP_GRID = "2,5,10,20,40,80"
KD_GRID = "0.25,0.5,1,2,4,8"


@app.function(cpu=16.0, memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_solo(cfg: dict) -> dict:
    import ast

    # Tiny MLPs against a 16-process MuJoCo pool: torch intra-op threading is pure contention
    # here, and the pool workers are forked from this process. Set before torch initialises.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)

    from mjc.practice.solo.world import (World, Ledger, ReflexDecider, TapeDecider,
                                         LibraryDecider, Library, select_tapes,
                                         MemberLibrary, build_member_cell, SoloDecider,
                                         clone_trunk, make_trunk_fn, make_trunk_read,
                                         trunk_bundle, trunk_inputs, _trunk_forward)
    from mjc.practice.offbook.nets import (build_prop, PropTrainer, prop_probe, build_span,
                                           SpanBuffer)

    t_start = time.time()
    device = "cpu"
    outdir = os.path.join(DATA_DIR, "practice_solo", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    out = {"config": cfg, "complete": False, "device": device,
           "contract": dict(
               no_forward_model=("Nothing in this node imports, trains or evaluates an f(s,u). "
                                 "The trunk is a behaviour clone of the reflex law's COMMAND, "
                                 "not a predictor of state."),
               operating_point=("Delta = 8 control steps = 192 ms — acappella b1's sustained "
                                "niche. Delta = 0 carries one control cell (`prop_k_d0`)."),
               levels=("SlotLayout populates segments and chains and the DONOR library builds "
                       "both (that is what makes gate S-F1 exact), but the TREATMENT library and "
                       "every treatment arm are SEGMENT-SPAN ONLY. acappella finding 5: on this "
                       "piece the chain arms decide once at seam 0 from rest and are exactly "
                       "delay-invariant, so no chain number here would be evidence about depth."),
               seeds="single seed, as everywhere in this arc")}
    lines = []

    def PR(s):
        print(s, flush=True)
        lines.append(s)

    def save():
        out["log"] = lines
        out["wall_s"] = time.time() - t_start
        with open(os.path.join(outdir, "solo.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    H, K = cfg["seg_H"], P.K_SEG
    W = World(cfg, n_proc=int(cfg["n_proc"]), rot=True)
    Wc = World(cfg, n_proc=int(cfg["n_proc"]), rot=False)
    PR(f"[setup] K={K} H={H} n_proc={cfg['n_proc']} dt_ctrl={W.dt_ctrl:.4f} device={device} "
       f"deltas={cfg['deltas']} ({[round(d * W.dt_ctrl * 1000) for d in cfg['deltas']]} ms)")

    # ================================================================= S-F0: the fork, re-gated
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
    out["S_F0"] = dict(checked=len(want), mismatches=bad, **{"pass": not bad})
    PR(f"[S-F0] donor constants: {len(want)} checked, {len(bad)} mismatched  pass={not bad}")
    if bad:
        save()
        raise SystemExit("S-F0 failed: the fork has drifted from the donor")

    ev_starts = W.start_states(np.random.default_rng(cfg["seed"] + 6000), cfg["n_eval"])
    rt_starts = W.start_states(np.random.default_rng(cfg["seed"] + 5000), cfg["n_rt"])

    # ================ the DONOR library — acappella's build, byte-for-byte, harvest gains frozen
    t0 = time.time()
    W.kp, W.kd = A0_KP, A0_KD
    lib = Library(K, cfg["n_slot"], poison=True)
    build, chain_rec, seam_sets, keep_pools = [], [], {}, {}
    lb = Ledger(W.dt_ctrl, W.d_fb)

    class MixedDecider:
        def __init__(self, upto):
            self.upto = int(upto)
            self.key = LibraryDecider(W, lib, mode="key", levels=("seg",))
            self.reflex = ReflexDecider()

        def __call__(self, k, states, need, rng, led):
            return (self.key if k < self.upto else self.reflex)(k, states, need, rng, led)

    def harvest(upto, n_cycles, batch, seed0):
        pools = {k: dict(cmds=[], raw=[], s0=[], err=[], traj=[]) for k in range(K)}
        chains = {k: dict(cmds=[], raw=[], s0=[], err=[]) for k in range(K)}
        led = Ledger(W.dt_ctrl, W.d_fb)
        for c in range(n_cycles):
            rng = np.random.default_rng(seed0 + c)
            o = W.traverse(MixedDecider(upto), W.start_states(rng, batch), rng, led,
                           explore=cfg["explore_sigma"], collect=True, collect_traj=True)
            for k in range(K):
                tr = o["traces"][k]
                for f in ("cmds", "raw", "s0", "err", "traj"):
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
        keep_pools[k] = pools[k]                     # reused by the TREATMENT build below
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
    out["donor_library"] = dict(build=build, chains=chain_rec, sizes=lib.sizes(),
                                build_ledger=lb.snap())
    out["t_donor_library"] = time.time() - t0
    PR(f"[lib] donor library {lib.sizes()}  {lb.ground:.0f} groundings  "
       f"{out['t_donor_library']:.0f}s")

    # ================================================= the reflex, re-fit per Delta (b1's grid)
    t0 = time.time()
    gains = {}
    for D in sorted(set(list(cfg["deltas"]) + [0])):
        grid = []
        for kp in cfg["kp_grid"]:
            for kd in cfg["kd_grid"]:
                Wc.kp, Wc.kd, Wc.trunk_fn = float(kp), float(kd), None
                led = Ledger(W.dt_ctrl, W.d_fb)
                oc = Wc.traverse(ReflexDecider(), ev_starts, np.random.default_rng(1), led,
                                 obs_delay=D)
                grid.append(dict(kp=float(kp), kd=float(kd), e_piece=oc["e_piece"]))
        b = min(grid, key=lambda r: r["e_piece"])
        b["at_kp_edge"] = bool(b["kp"] in (min(cfg["kp_grid"]), max(cfg["kp_grid"])))
        b["at_kd_edge"] = bool(b["kd"] in (min(cfg["kd_grid"]), max(cfg["kd_grid"])))
        gains[D] = b
        PR(f"[cal] Delta={D:2d}  reflex kp={b['kp']:>5} kd={b['kd']:>5}  clean e={b['e_piece']:.4f}"
           f"  edge(kp/kd)={b['at_kp_edge']}/{b['at_kd_edge']}")
    out["reflex_cal"] = {str(k): v for k, v in gains.items()}
    out["t_reflex_cal"] = time.time() - t0

    # ============================================== S-F1: cross-tag exact control vs acappella b1
    applicable = all(cfg.get(k) == v for k, v in B1_CFG.items())
    sf1 = {"build_seg_max_abs": max(abs(build[i]["chosen_score"] - A0_BUILD_SEG[i])
                                    for i in range(K)),
           "build_chain_max_abs": max(abs(chain_rec[i]["chosen_score"] - A0_BUILD_CHAIN[i])
                                      for i in range(len(chain_rec)))}
    DONOR_ARMS = {
        "key_seg":   lambda: LibraryDecider(W, lib, mode="key", levels=("seg",)),
        "key_chain": lambda: LibraryDecider(W, lib, mode="key", levels=("chain",)),
        "lib_seg":   lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("seg",)),
        "lib_all":   lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("seg", "chain")),
        "aud_chain": lambda: LibraryDecider(W, lib, mode="audit_all", budget=0, with_prim=False,
                                            levels=("chain",)),
    }
    donor_rows = {}
    for D in (0, 8):
        if D not in B1_ROWS:
            continue
        W.trunk_fn = None
        for nm, mk in DONOR_ARMS.items():
            led = Ledger(W.dt_ctrl, W.d_fb)
            o = W.traverse(mk(), ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                           obs_delay=D)
            donor_rows[f"{nm}@{D}"] = dict(e_piece=o["e_piece"], **led.snap())
            sf1[f"{nm}@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D][nm])
        W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
        led = Ledger(W.dt_ctrl, W.d_fb)
        o = W.traverse(ReflexDecider(), ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                       obs_delay=D)
        donor_rows[f"reflex@{D}"] = dict(e_piece=o["e_piece"], **led.snap())
        sf1[f"reflex@{D}_max_abs"] = abs(o["e_piece"] - B1_ROWS[D]["reflex"])
        sf1[f"gains@{D}_match"] = bool((gains[D]["kp"], gains[D]["kd"]) == B1_GAINS[D])
    mx = max(v for k, v in sf1.items() if k.endswith("_max_abs"))
    sf1["gains_match"] = all(v for k, v in sf1.items() if k.endswith("_match"))
    sf1["applicable"] = bool(applicable)
    sf1["pass"] = bool(applicable and mx == 0.0 and sf1["gains_match"])
    sf1["max_abs"] = mx
    out["S_F1"] = sf1
    out["donor_rows"] = donor_rows
    PR(f"[S-F1] cross-tag exact control vs acappella b1: max|delta| = {mx:.3e}  "
       f"gains_match={sf1['gains_match']}  applicable={applicable}  pass={sf1['pass']}"
       + ("" if applicable else "  [config differs from b1's; control not defined here]"))
    save()
    if applicable and not sf1["pass"]:
        raise SystemExit("S-F1 failed: the fork is not acappella b1's code path")

    # ================================================================= THE TRUNK, and its gate
    t0 = time.time()
    trunks, trunk_fns, trunk_reads, trunk_nps, bc_rec = {}, {}, {}, {}, {}
    for D in cfg["deltas"]:
        g = (gains[D]["kp"], gains[D]["kd"])
        net, norm, rec = clone_trunk([(W, g), (Wc, g)], cfg, cfg["seed"] + 100 * D, device, K,
                                     deltas=tuple(sorted(set(list(cfg["deltas"]) + [0]))), log=PR)
        trunks[D] = net
        trunk_fns[D] = make_trunk_fn(net, norm, device, K, H)
        trunk_reads[D] = make_trunk_read(net, norm, device, K, H)
        trunk_nps[D] = trunk_bundle(net, norm)
        bc_rec[D] = dict(gains=list(g), **rec)
        PR(f"[trunk] Delta={D} cloned from the reflex law at kp={g[0]} kd={g[1]}: "
           f"{rec['n']} samples, mse {rec['mse_full']:.3e}")
    out["bc"] = bc_rec
    # S-T0: the numpy forward the pool workers use IS the torch forward
    st0 = 0.0
    rng0 = np.random.default_rng(11)
    for D in cfg["deltas"]:
        s = rng0.normal(0, 0.6, (64, 4)).astype(np.float32)
        for k in range(K):
            a = trunk_fns[D](s, k, 0)
            b = _trunk_forward(trunk_nps[D], s.astype(np.float64), k, 1.0 / H, K)
            st0 = max(st0, float(np.max(np.abs(a - b))))
    out["S_T0"] = dict(max_abs=st0, tol=1e-4, **{"pass": bool(st0 < 1e-4)})
    PR(f"[S-T0] numpy trunk forward == torch trunk forward: max|delta| = {st0:.3e}  "
       f"pass={st0 < 1e-4}")

    # S-T1: the clone reproduces the reflex law, both worlds, every Delta. PRE-FIXED TOLERANCE.
    tol_rel, tol_abs = float(cfg["trunk_tol_rel"]), float(cfg["trunk_tol_abs"])
    st1, ok = [], True
    for D in cfg["deltas"]:
        for wn, Wx in (("rot", W), ("clean", Wc)):
            for Dv in sorted(set(list(cfg["deltas"]) + [0])):
                Wx.kp, Wx.kd = gains[D]["kp"], gains[D]["kd"]
                Wx.trunk_fn = None
                led = Ledger(W.dt_ctrl, W.d_fb)
                law = Wx.traverse(ReflexDecider(), ev_starts, np.random.default_rng(3), led,
                                  obs_delay=Dv)
                Wx.trunk_fn = trunk_fns[D]
                led = Ledger(W.dt_ctrl, W.d_fb)
                clo = Wx.traverse(ReflexDecider(), ev_starts, np.random.default_rng(3), led,
                                  obs_delay=Dv)
                Wx.trunk_fn = None
                band = max(tol_rel * law["e_piece"], tol_abs)
                row = dict(trunk_delta=int(D), world=wn, eval_delta=int(Dv),
                           e_law=law["e_piece"], e_clone=clo["e_piece"],
                           diff=clo["e_piece"] - law["e_piece"], band=band,
                           e_seg_law=[float(x) for x in law["e_seg"]],
                           e_seg_clone=[float(x) for x in clo["e_seg"]])
                row["pass"] = bool(abs(row["diff"]) <= band)
                ok = ok and row["pass"]
                st1.append(row)
    out["S_T1"] = dict(rows=st1, tol_rel=tol_rel, tol_abs=tol_abs, **{"pass": bool(ok)})
    PR("[S-T1] the clone vs the law it clones (pre-fixed band: "
       f"|diff| <= max({tol_rel:.2f}*e_law, {tol_abs:.3f})):")
    for r in st1:
        PR(f"       trunk(D={r['trunk_delta']}) {r['world']:>5s} @D={r['eval_delta']:2d}  "
           f"law={r['e_law']:.4f}  clone={r['e_clone']:.4f}  diff={r['diff']:+.4f}  "
           f"band={r['band']:.4f}  {'PASS' if r['pass'] else 'FAIL'}")
    out["t_trunk"] = time.time() - t0
    out["S_T1"]["binding"] = bool(cfg["trunk_gate_binding"])
    save()
    if not ok and not cfg["trunk_gate_binding"]:
        PR("[S-T1] *** the trunk gate FAILED but is NOT BINDING under --quick (the clone is "
           "deliberately undertrained there: bc_steps/bc_cycles are smoke-sized). Continuing so "
           "the rest of the chain is exercised. On a real run this is a hard stop. ***")
    if not ok and cfg["trunk_gate_binding"]:
        PR("[S-T1] *** THE TRUNK GATE FAILED. Stopping before any port is wired, per the SPEC: "
           "a trunk that does not play the law it clones would confound the primitive and every "
           "battery row. No tuning around it. ***")
        save()
        raise SystemExit("S-T1 failed: the cloned trunk does not reproduce the reflex law")

    # ============================== S-T2: the trunk is bit-for-bit the reference run's trunk
    # The probe re-clones rather than reloading (no weights are persisted), so this asserts the
    # re-clone landed exactly where `s0`'s did. Same seed, same cloning config, same gains grid.
    refp = os.path.join(DATA_DIR, "practice_solo", str(cfg.get("ref_tag") or ""), "solo.json")
    refd = None
    if cfg.get("ref_tag") and os.path.exists(refp):
        try:
            refd = json.load(open(refp))
        except Exception as e:
            PR(f"[refd] could not read {refp}: {e}")
    ref_ok = bool(refd) and all(refd["config"].get(k) == cfg.get(k) for k in S0_KEYS)
    ref_bad = ([k for k in S0_KEYS if refd and refd["config"].get(k) != cfg.get(k)] if refd else [])
    st2 = dict(ref_tag=cfg.get("ref_tag"), ref_found=bool(refd), applicable=bool(ref_ok),
               config_mismatches=ref_bad)
    if ref_ok:
        mx = 0.0
        for r in out["S_T1"]["rows"]:
            m = [q for q in refd["S_T1"]["rows"]
                 if q["trunk_delta"] == r["trunk_delta"] and q["world"] == r["world"]
                 and q["eval_delta"] == r["eval_delta"]]
            if m:
                mx = max(mx, abs(r["e_clone"] - m[0]["e_clone"]))
        st2["max_abs"] = mx
        st2["mse_max_abs"] = max(
            abs(out["bc"][D]["mse_full"] - refd["bc"][str(D)]["mse_full"])
            for D in cfg["deltas"] if str(D) in refd["bc"])
        st2["pass"] = bool(mx == 0.0)
    out["S_T2"] = st2
    PR(f"[S-T2] the re-cloned trunk vs `{cfg.get('ref_tag')}`'s: "
       + (f"max|delta| = {st2['max_abs']:.3e}  pass={st2['pass']}" if ref_ok
          else f"NOT DEFINED here (ref_found={bool(refd)}, mismatches={ref_bad[:6]})"))
    save()

    # =============================================== the TREATMENT library: slots with MEMBERS
    t0 = time.time()
    mlib = MemberLibrary(K, cfg["n_slot"], poison=True)
    mb_rec = []
    mlb = Ledger(W.dt_ctrl, W.d_fb)
    for k in range(K):
        S0 = seam_sets[k][:cfg["n_score"]]
        slots, poison, rec = build_member_cell(
            W, keep_pools[k], S0, k, np.random.default_rng(cfg["seed"] + 77777 + 13 * k), mlb,
            cfg["m_cand"], cfg["n_slot"], cfg["m_member"], cfg["n_poison"])
        for g in slots:
            mlib.add_slot(k, g, np.mean([m["key"] for m in g], 0))
        if poison:
            mlib.add_slot(k, poison, np.mean([m["key"] for m in poison], 0), poison=True)
        mb_rec.append(rec)
    out["member_library"] = dict(build=mb_rec, sizes=mlib.sizes(), build_ledger=mlb.snap())
    out["t_member_library"] = time.time() - t0
    PR(f"[lib] member library {mlib.sizes()}  {mlb.ground:.0f} groundings  "
       f"{out['t_member_library']:.0f}s")

    n_slots = mlib.layout.n_slots
    # the seam a slot belongs to, resolved ONCE from the layout: the span trainer and the parity
    # gate both need it and neither should re-derive it.
    cfg["_slot_seam"] = {sid: int(mlib.layout.cell_of(sid)[1]) for sid in range(n_slots)}
    legal_all = mlib.legal_by_seam(with_prim=True)
    legal_noprim = mlib.legal_by_seam(with_prim=False)

    # the seam-state normaliser pi and the probe share: computed ONCE from a reflex traversal, so
    # it is arm-neutral and identical for every arm.
    W.trunk_fn = None
    W.kp, W.kd = gains[cfg["deltas"][0]]["kp"], gains[cfg["deltas"][0]]["kd"]
    led = Ledger(W.dt_ctrl, W.d_fb)
    onorm = W.traverse(ReflexDecider(), rt_starts, np.random.default_rng(cfg["seed"] + 61), led)
    Zn = np.concatenate([onorm["seam"][k] for k in range(K)], 0)
    NORM = (Zn.mean(0).astype(np.float32), (Zn.std(0) + 1e-6).astype(np.float32))

    # ============================================ S-S: seam information at Delta, delayed read
    ss = []
    for D in cfg["deltas"]:
        W.trunk_fn, W.trunk_np = trunk_fns[D], trunk_nps[D]
        led = Ledger(W.dt_ctrl, W.d_fb)
        dec = SoloDecider(W, mlib, mode="key", with_prim=False, obs_delay=D)
        okey = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                          obs_delay=D, collect_obs=True)
        for k in range(K):
            st = okey["seam"][k]
            S = mlib.slots(k)
            sc = np.full((len(st), len(S)), np.inf)
            for si, sl in enumerate(S):
                per = np.full(len(st), np.inf)
                for mb in sl["members"]:
                    e, _ = W.rollout(st, np.tile(mb["cmds"][None], (len(st), 1, 1)), k, 1, None)
                    per = np.minimum(per, e.mean(1))
                sc[:, si] = per
            fixed = float(sc.mean(0).min())
            oracle = float(sc.min(1).mean())
            ss.append(dict(delta=int(D), seam=int(k), best_fixed_slot=fixed,
                           per_state_oracle=oracle, gain=fixed / max(oracle, 1e-9),
                           n_distinct_argmin=int(len(np.unique(sc.argmin(1)))),
                           n_slot=len(S),
                           key_achieved=float(np.mean([
                               sc[i, int(np.argmin([np.linalg.norm(
                                   (okey["obs"][i, k * H, :] - s["key"]) / NORM[1])
                                   for s in S]))] for i in range(len(st))]))))
    out["S_S"] = ss
    PR("[S-S] seam information under the DELAYED read (per-state oracle over the member library):")
    for r in ss:
        PR(f"      D={r['delta']:2d} seam {r['seam']}  best fixed slot {r['best_fixed_slot']:.4f}"
           f"  per-state oracle {r['per_state_oracle']:.4f}  gain {r['gain']:.2f}x"
           f"  distinct argmins {r['n_distinct_argmin']}/{r['n_slot']}"
           f"  key achieves {r['key_achieved']:.4f}")
    save()

    # =============================================================== the arms
    ARM_SPEC = {
        "reflex":        dict(mode="reflex",    delta=8, learn=False, meter_once=True),
        "key_seg":       dict(mode="key",       delta=8, learn=False, meter_once=True,
                              with_prim=False),
        "audit_all":     dict(mode="audit_all", delta=8, learn=False, meter_once=True),
        "audit_prop_k":  dict(mode="prop_k",    delta=8, learn=True),
        "route_native":  dict(mode="native",    delta=8, learn=True, consult_head=True),
        "fid":           dict(mode="audit_all", delta=8, learn=False, mint=True),
        "audit_prop_kN": dict(mode="prop_kN",   delta=8, learn=True),
        "prop_k_d0":     dict(mode="prop_k",    delta=0, learn=True),
        # round 2 (`s1q`): pi's imitation filter as the ONE variable. Same arms, same everything,
        # an absolute competence band in place of s0's per-cycle median.
        "audit_prop_k_ref": dict(mode="prop_k", delta=8, learn=True,
                                 pi_filter="band", pi_band=REF_PLAY),
        "route_native_ref": dict(mode="native", delta=8, learn=True, consult_head=True,
                                 pi_filter="band", pi_band=REF_PLAY),
        "audit_prop_k_lib": dict(mode="prop_k", delta=8, learn=True,
                                 pi_filter="band", pi_band=BAND_LIB),
        "route_native_lib": dict(mode="native", delta=8, learn=True, consult_head=True,
                                 pi_filter="band", pi_band=BAND_LIB),
    }
    arms = {}
    skipped = [nm for nm in cfg["arms"] if int(ARM_SPEC[nm]["delta"]) not in cfg["deltas"]]
    if skipped:
        PR(f"[arms] skipped (their Delta is not in the ladder): {skipped}")
    out["skipped_arms"] = skipped
    for nm in [x for x in cfg["arms"] if x not in skipped]:
        spec = ARM_SPEC[nm]
        D = int(spec["delta"])
        t_arm = time.time()
        W.trunk_fn, W.trunk_np = trunk_fns[D], trunk_nps[D]
        W.kp, W.kd = gains[D]["kp"], gains[D]["kd"]
        with_prim = bool(spec.get("with_prim", True))
        learn = bool(spec["learn"])
        pi_filter = str(spec.get("pi_filter", cfg["pi_filter"]))
        pi_band = float(spec.get("pi_band", cfg["pi_band"]))
        mint = bool(spec.get("mint", False)) or learn
        prop = span = trainer = sbuf = None
        if mint:
            prop = build_prop(4, K, n_slots, cfg["seed"] + 31337, device,
                              hidden=cfg["prop_hidden"], layers=cfg["prop_layers"])
            trainer = PropTrainer(prop, n_slots, cfg["prop_lr"], cfg["prop_cap"],
                                  cfg["seed"] + 31338, device)
            span = build_span(n_slots, 4, int(trunks[D].hidden), H, 2, cfg["seed"] + 31339,
                              device, hidden=cfg["span_hidden"], layers=cfg["span_layers"])
            sbuf = SpanBuffer(cfg["span_cap"], cfg["span_hold_cap"], cfg["span_hold_frac"],
                              cfg["seed"] + 31340)
            span_opt = torch.optim.Adam(span.parameters(), lr=float(cfg["span_lr"]))

        import torch.nn.functional as F
        act_rng, exp_rng = [None], [None]

        def prop_fn(states, k, _p=prop):
            if _p is None:
                return np.zeros((len(states), n_slots), np.float32)
            with torch.no_grad():
                z = torch.tensor((np.asarray(states, np.float32) - NORM[0]) / NORM[1],
                                 device=device)
                oh = F.one_hot(torch.full((len(states),), int(k), dtype=torch.long,
                                          device=device), num_classes=K).float()
                return _p(z, oh).cpu().numpy()

        def span_fn(states, sid, k, _s=span, _tr=trunk_reads[D]):
            with torch.no_grad():
                z = torch.tensor((np.asarray(states, np.float32) - NORM[0]) / NORM[1],
                                 device=device)
                tr = _tr(np.asarray(states, np.float32), k)
                sl = torch.full((len(states),), int(sid), dtype=torch.long, device=device)
                y = _s(z, tr, sl).cpu().numpy()
            return np.clip(y[:, :H, :], -1.0, 1.0).astype(np.float32)

        def make_dec(mode, delay, capture, practice, force_all=False, wp=None, opens=(),
                     forced_open=False):
            return SoloDecider(W, mlib, mode=mode, k_prop=cfg["k_prop"],
                               with_prim=(with_prim if wp is None else wp), obs_delay=delay,
                               prop_fn=prop_fn, span_fn=span_fn, open_slots=opens,
                               force_all=force_all,
                               eps_act=(cfg["eps_act"] if practice else 0.0),
                               explore_eps=(cfg["explore_eps"] if practice else 0.0),
                               act_rng=act_rng[0], exp_rng=exp_rng[0], capture=capture,
                               head_forced_open=forced_open)
        open_slots, parity_rows = set(), []
        cycles, dperf_state, dperf_rows = [], {}, []
        n_cyc = 1 if spec.get("meter_once") else int(cfg["n_cycles"])
        for c in range(n_cyc):
            act_rng[0] = np.random.default_rng(cfg["seed"] + 800000 + c)
            exp_rng[0] = np.random.default_rng(cfg["seed"] + 900000 + c)
            t_c0 = time.time()
            led_pr = Ledger(W.dt_ctrl, W.d_fb)
            # ---------------- practice (motor noise on; exploration upstream of pi's gate)
            if learn:
                rng_pr = np.random.default_rng(cfg["seed"] + 700000 + c)
                dpr = make_dec(spec["mode"], D, capture=False, practice=True,
                               force_all=(c < int(cfg["force_window"])),
                               opens=set(open_slots))
                opr = W.traverse(dpr, W.start_states(rng_pr, cfg["batch_pr"]), rng_pr, led_pr,
                                 explore=cfg["explore_sigma"], obs_delay=D)
                ep = opr["wp_err"].mean(1)
                # THE BODY GRADES, never the audition score. Two filters:
                #   "median" — s0's: imitate the better HALF of this cycle's own traversals. This is
                #              also `offbook/`'s "competence band" (offbook.py:485 is a nanmedian),
                #              so an ABSOLUTE band is new to this lineage.
                #   "band"   — imitate every traversal at or under a fixed, pre-published error.
                # Under delay the body's grade is only weakly a function of the slot chosen, so a
                # relative filter admits posture luck as competence; the absolute band is the
                # one-variable change that tests that. STARVATION is the failure mode to watch, so
                # the pass fraction and the number of target rows are logged every cycle.
                good = (ep <= np.median(ep) if pi_filter == "median" else ep <= pi_band)
                row_pass = float(good.mean())
                Z, KK, Y = [], [], []
                for st, kk, ch, nd in dpr.prop_rows:
                    m = good[np.asarray(nd, int)]
                    if m.any():
                        Z.append(st[m]); KK.append(np.full(int(m.sum()), kk)); Y.append(ch[m])
                n_tgt = int(sum(len(z) for z in Z))
                if Z:
                    trainer.add(np.concatenate(Z), np.concatenate(KK), np.concatenate(Y))
            t_pr = time.time() - t_c0
            # ---------------- metering, at performance tempo (no exploration, no motor noise)
            t_m0 = time.time()
            led = Ledger(W.dt_ctrl, W.d_fb)
            if spec["mode"] == "reflex":
                dec = ReflexDecider()
            else:
                dec = make_dec(spec["mode"], D, capture=bool(mint), practice=False,
                               opens=set(open_slots))
            om = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                            obs_delay=D, collect_obs=True, collect_traj=True)
            row = dict(cycle=int(c), arm=nm, delta=D, e_piece=om["e_piece"],
                       e_seg=[float(x) for x in om["e_seg"]], **led.snap())
            if hasattr(dec, "picks") and dec.picks:
                row["frac_prim"] = float(np.mean([p["frac_prim"] for p in dec.picks]))
                row["n_aud"] = float(np.mean([p["n_aud"] for p in dec.picks]))
                row["n_prop"] = float(np.mean([p["n_prop"] for p in dec.picks]))
                # --- delta_perf, against the committed tape's OWN stored trajectory ----------
                ep = om["wp_err"].mean(1)
                gate = ep <= np.median(ep)
                per_slot = {}
                for p in dec.picks:
                    kk = int(p["seam"])
                    for j, b in enumerate(p["need"]):
                        mi, sid = int(p["member"][j]), int(p["slot"][j])
                        if mi < 0:
                            continue
                        sl = [s for s in mlib.slots(kk) if s["slot"] == sid]
                        if not sl or sl[0]["members"][mi]["traj"] is None:
                            continue
                        ref_traj = sl[0]["members"][mi]["traj"]
                        real = om["traj"][b, kk * H:(kk + 1) * H, :]
                        e = float(np.linalg.norm(real - ref_traj, axis=1).mean())
                        ewp = float(np.linalg.norm(real[-1] - ref_traj[-1]))
                        bmk = dperf_state.get(sid)
                        r = per_slot.setdefault(sid, dict(n=0, nd=0, e=0.0, ewp=0.0, d=0.0,
                                                          sil=0))
                        r["n"] += 1; r["e"] += e; r["ewp"] += ewp
                        if bmk is not None:
                            d = (bmk - e) * (1.0 if gate[b] else 0.0)
                            r["nd"] += 1; r["d"] += d
                            r["sil"] += int(abs(d) <= cfg["dperf_sil"] * max(e, 1e-9))
                        dperf_state[sid] = e if bmk is None else (
                            (1 - cfg["dperf_ema"]) * bmk + cfg["dperf_ema"] * e)
                for sid, r in per_slot.items():
                    dperf_rows.append(dict(cycle=int(c), slot=int(sid), n=r["n"], n_d=r["nd"],
                                           e_step=r["e"] / r["n"], e_wp=r["ewp"] / r["n"],
                                           dperf=(r["d"] / r["nd"]) if r["nd"] else None,
                                           frac_silent=((r["sil"] / r["nd"])
                                                        if r["nd"] else None)))
                # --- Port 2's buffer: EVERY selected slot, on the metering traversal ---------
                if mint and learn:
                    for st, sid, tg in dec.cap_rows:
                        sbuf.store(int(sid), st, tg.reshape(len(tg), -1))
            row["t_practice_ledger"] = led_pr.snap()
            row["t_practice"], row["t_meter"] = t_pr, time.time() - t_m0
            if learn:
                row["pi_filter"] = pi_filter
                row["pi_band"] = (pi_band if pi_filter == "band" else None)
                row["band_pass"] = row_pass          # fraction of the practice batch admitted
                row["n_targets"] = n_tgt             # pi target rows added this cycle
                row["n_graded"] = int(len(ep))
            # ---------------- training
            t_t0 = time.time()
            if learn:
                lg = legal_all if with_prim else legal_noprim
                row["prop_loss"] = trainer.train(cfg["prop_steps"], cfg["prop_batch"], lg,
                                                 NORM, K)
                row["prop_buf"] = trainer.size()
                row["span_loss"] = _train_span(span, span_opt, sbuf, span_fn, trunk_reads[D],
                                               NORM, device, cfg, H, torch, np)
            row["t_train"] = time.time() - t_t0
            # ---------------- the parity gate, measured ON THE PLANT (an instrument in every
            # learning arm; only `route_native` lets it change what gets launched)
            t_pa = time.time()
            if learn and (c % int(cfg["parity_every"]) == 0 or c == n_cyc - 1):
                led_par = Ledger(W.dt_ctrl, W.d_fb)
                new_open, prow = _parity(W, mlib, sbuf, span_fn, cfg, np, led_par)
                prow["ledger"] = led_par.snap()
                prow.update(cycle=int(c), consulted=bool(spec.get("consult_head")))
                parity_rows.append(prow)
                if spec.get("consult_head"):
                    open_slots = new_open
                row["n_open"] = len(new_open)
            row["t_parity"] = time.time() - t_pa
            # ---------------- the trust series
            t_p0 = time.time()
            if mint and (c % int(cfg["probe_every"]) == 0 or c == n_cyc - 1):
                Zp = np.concatenate([om["obs"][:, k * H, :] for k in range(K)], 0)
                Kp = np.concatenate([np.full(len(ev_starts), k) for k in range(K)], 0)
                row["probe"] = prop_probe(prop, Zp, Kp, legal_all, NORM, K, n_slots,
                                          mlib.layout, device)
            row["t_probe"] = time.time() - t_p0
            row["t_cycle"] = time.time() - t_c0
            cycles.append(row)
            if c % max(1, n_cyc // 10) == 0 or c == n_cyc - 1:
                PR(f"[{nm}] c{c:3d}  e={row['e_piece']:.4f}  fb={row['n_fb']:.1f}  "
                   f"g={row['n_ground']:.1f}  frac_prim={row.get('frac_prim', float('nan')):.2f}"
                   f"  n_aud={row.get('n_aud', float('nan')):.1f}"
                   + (f"  open={row.get('n_open', 0)}" if learn else "")
                   + (f"  pass={row.get('band_pass', float('nan')):.2f}"
                      f"/{row.get('n_targets', 0)}rows" if learn else "")
                   + f"  [{row.get('t_cycle', 0.0):.1f}s]")
                save()
        # ---------------- the address-book battery, on the final trained state
        battery = {}
        if mint:
            for bn, (bmode, bwp) in dict(
                    base=(spec["mode"], with_prim),
                    no_prim=(spec["mode"], False),
                    no_table=("no_table", with_prim),
                    no_table_no_prim=("no_table", False),
                    restored=(spec["mode"], with_prim)).items():
                led = Ledger(W.dt_ctrl, W.d_fb)
                dec = make_dec(bmode, D, capture=False, practice=False, wp=bwp,
                               opens=set(open_slots))
                ob = W.traverse(dec, ev_starts, np.random.default_rng(cfg["seed"] + 35), led,
                                obs_delay=D)
                battery[bn] = dict(e_piece=ob["e_piece"], **led.snap(),
                                   frac_prim=float(np.mean([p["frac_prim"]
                                                            for p in dec.picks])))
            battery["restored_max_abs"] = abs(battery["restored"]["e_piece"]
                                              - battery["base"]["e_piece"])
            PR(f"[{nm}] battery: " + "  ".join(
                f"{k}={v['e_piece']:.4f}(prim{v['frac_prim']:.2f})"
                for k, v in battery.items() if isinstance(v, dict)))
        arms[nm] = dict(spec=spec, cycles=cycles, battery=battery, parity=parity_rows,
                        dperf=dperf_rows, open_slots=sorted(int(s) for s in open_slots),
                        wall_s=time.time() - t_arm)
        out["arms"] = arms
        PR(f"[{nm}] done in {time.time() - t_arm:.0f}s  final e={cycles[-1]['e_piece']:.4f}")
        save()

    # ============ S-M: the `median` path is byte-identical to the reference run's
    # The probe's ONE variable is pi's imitation filter. This asserts that the arm which did NOT
    # change reproduces `s0` cycle for cycle — on piece error AND on pi's per-level mass — so any
    # difference in a band arm is the filter and nothing else. Declared NOT DEFINED, never
    # silently passed, if the reference run is absent or any config key differs.
    sm = dict(ref_tag=cfg.get("ref_tag"), applicable=False)
    if ref_ok and "audit_prop_k" in arms and "audit_prop_k" in refd.get("arms", {}):
        a, b = arms["audit_prop_k"]["cycles"], refd["arms"]["audit_prop_k"]["cycles"]
        n = min(len(a), len(b))
        de = max(abs(a[i]["e_piece"] - b[i]["e_piece"]) for i in range(n))
        dm = 0.0
        nprobe = 0
        for i in range(n):
            if a[i].get("probe") and b[i].get("probe"):
                nprobe += 1
                for f in ("mass_seg", "mass_prim", "mass_poison", "top1", "entropy"):
                    dm = max(dm, abs(a[i]["probe"][f] - b[i]["probe"][f]))
        sm.update(applicable=True, n_cycles=int(n), e_piece_max_abs=de, probe_max_abs=dm,
                  n_probes=nprobe, **{"pass": bool(de == 0.0 and dm == 0.0)})
    out["S_M"] = sm
    PR("[S-M] the `median` arm vs the reference run, cycle for cycle: "
       + (f"e_piece max|delta| = {sm['e_piece_max_abs']:.3e} over {sm['n_cycles']} cycles, "
          f"pi-mass max|delta| = {sm['probe_max_abs']:.3e} over {sm['n_probes']} probes  "
          f"pass={sm['pass']}" if sm["applicable"] else "NOT DEFINED here"))

    # =============================================================== gate S-P: the twins
    twins = {}
    if "audit_all" in arms:
        ref_e = arms["audit_all"]["cycles"][-1]["e_piece"]
        for nm in ("fid", "audit_prop_kN"):
            if nm not in arms:
                continue
            d = max(abs(r["e_piece"] - ref_e) for r in arms[nm]["cycles"])
            twins[nm] = dict(max_abs=d, n_cycles=len(arms[nm]["cycles"]),
                             **{"pass": bool(d == 0.0)})
            PR(f"[S-P] {nm} == audit_all over {len(arms[nm]['cycles'])} cycles: "
               f"max|delta| = {d:.3e}  pass={d == 0.0}")
    out["S_P"] = twins

    # ---- standing flags, recorded in-run so no reduction has to re-derive them ------------
    out["flags"] = dict(
        gain_grid_edges={str(k): dict(kp=v["at_kp_edge"], kd=v["at_kd_edge"])
                         for k, v in gains.items()},
        trunk_gate_binding=bool(cfg["trunk_gate_binding"]),
        s_f1_applicable=bool(out["S_F1"]["applicable"]),
        cant_decompose=("NOT REPORTED: at segment span a unit's 'spelling' would be the primitive "
                        "executing the same segment, and there is ONE primitive slot per seam "
                        "shared by every segment slot there — so pi's mass on a unit's own "
                        "spelling collapses to `mass_prim`, which is already the adoption "
                        "readout. There is no distinct per-slot spelling to put mass on, so the "
                        "native/ readout has no segment-span form here."),
        given_library=("NOT BUILT this round: an earned-vs-given contrast needs a differently "
                       "SOURCED library, and the only model-free source on this substrate is the "
                       "priced plant search that acappella A-I retired as an incumbent (best "
                       "cell 0.0607 at 15011 s priced). Recorded, deliberately not taken."),
        primitive_audition=("the primitive is auditioned CLOSED-LOOP on a resettable plant copy "
                            "with the same observation delay applied inside the trial; without "
                            "the internal delay the audition would run the primitive under a "
                            "capability the body does not have"),
        trunk_frozen=("the trunk does NOT adapt: the span loss is detached from it. It is the "
                      "incumbent whose piece error every arm is measured against, and offbook's "
                      "`regress` measured what unconditioned regression does to renditions. The "
                      "plant guard is therefore a CONTROL here (the reflex arm's error is "
                      "constant by construction), not a treatment readout."))
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    W.close(); Wc.close()
    PR(f"[save] {outdir}  total wall {time.time() - t_start:.0f}s")
    return out


def _train_span(span, opt, sbuf, span_fn, trunk_read, NORM, device, cfg, H, torch, np):
    """Port 2's regression: (slot, delayed seam posture, FROZEN trunk read) -> the audition's own
    chosen commands for that slot at that state. The trunk read is DETACHED — see FILES.md."""
    if not sbuf.buf:
        return None
    sids, Z, Y = [], [], []
    for sid, (S, C) in sbuf.buf.items():
        sids.append(np.full(len(S), int(sid), np.int64)); Z.append(S); Y.append(C)
    sids = np.concatenate(sids); Z = np.concatenate(Z); Y = np.concatenate(Y)
    n = len(Z)
    if n < 8:
        return None
    rng = np.random.default_rng(int(cfg["seed"]) + 5150)
    Zt = torch.tensor((Z - NORM[0]) / NORM[1], device=device)
    Yt = torch.tensor(Y.reshape(n, H, 2), device=device)
    St = torch.tensor(sids, device=device)
    # The trunk read depends on the SEAM the slot belongs to; a slot is per-seam, so the seam is
    # recoverable from the slot id through the layout. Precomputed ONCE for the whole buffer: the
    # trunk is FROZEN (see FILES.md), so its read of a fixed state never changes.
    seam_of = np.asarray([cfg["_slot_seam"][int(x)] for x in sids], int)
    TR = torch.zeros((n, int(span.trunk[0].in_features) - 4 - span.slot.embedding_dim),
                     device=device)
    for kk in np.unique(seam_of):
        m = seam_of == kk
        TR[torch.tensor(np.flatnonzero(m), device=device)] = trunk_read(Z[m], int(kk))
    TR = TR.detach()
    span.train()
    last = None
    for _ in range(int(cfg["span_steps"])):
        idx = rng.integers(0, n, size=min(int(cfg["span_batch"]), n))
        it = torch.tensor(idx, device=device)
        y = span(Zt[it], TR[it], St[it])
        loss = ((y[:, :H, :] - Yt[it]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    span.eval()
    return last


def _parity(W, mlib, sbuf, span_fn, cfg, np, led=None):
    """PARITY, measured as execution reproduction ON THE PLANT (a grounding), on held-out seam
    states split by a deterministic code of the state (offbook decision 7 — a coin flip would put
    near-copies of one context on both sides and the gate would read its own training data).

    A slot is OPEN iff the head's single emission, executed, is at least as good as the audition's
    own chosen member, executed, on at least `parity_tau` of that slot's held-out states. Below
    parity the tape plays verbatim, so the gate can never introduce drift. Continuous fractions
    are logged so another tau can be read off the record.
    """
    opens, rows = set(), []
    for sid, (S, C) in sbuf.hold.items():
        if len(S) < int(cfg["parity_min"]):
            continue
        k = int(cfg["_slot_seam"][int(sid)])
        tape = C.reshape(len(C), W.H, 2).astype(np.float32)
        # the tape side is a deterministic function of (slot, state) and is the reference the
        # head must match, so it is an instrument, not a cost the agent pays every cycle; the HEAD
        # side is a real self-check the agent performs and is charged.
        e_tape, _ = W.rollout(S, tape, k, 1, led, charge=False)
        head = span_fn(S, int(sid), k)
        e_head, _ = W.rollout(S, head, k, 1, led)
        et, eh = e_tape.mean(1), e_head.mean(1)
        frac = float(np.mean(eh <= et))
        rows.append(dict(slot=int(sid), seam=k, n_hold=int(len(S)), frac=frac,
                         mean_e_tape=float(et.mean()), mean_e_head=float(eh.mean()),
                         mean_gap=float((eh - et).mean()),
                         open=bool(frac >= float(cfg["parity_tau"]))))
        if frac >= float(cfg["parity_tau"]):
            opens.add(int(sid))
    return opens, dict(rows=rows, n_open=len(opens), n_checked=len(rows),
                       tau=float(cfg["parity_tau"]))


@app.local_entrypoint()
def solo_run(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    n_proc: int = 16,
    deltas: str = "8,0",
    arms: str = "reflex,key_seg,audit_all,fid,audit_prop_kN,audit_prop_k,route_native,prop_k_d0",
    n_cycles: int = 80,
    batch_pr: int = 16,
    n_eval: int = 32,
    n_rt: int = 48,
    batch: int = 24,
    n_warm: int = 8,
    n_cand: int = 24,
    n_score: int = 16,
    n_slot: int = 8,
    m_cand: int = 64,
    m_member: int = 4,
    n_poison: int = 4,
    k_prop: int = 2,
    eps_act: float = 0.10,
    explore_eps: float = 0.15,
    force_window: int = 3,
    pi_filter: str = "median",
    pi_band: float = REF_PLAY,
    ref_tag: str = "s0",
    parity_tau: float = 0.75,
    parity_every: int = 5,
    probe_every: int = 5,
    trunk_tol_rel: float = 0.25,
    trunk_tol_abs: float = 0.010,
    bc_cycles: int = 10,
    bc_steps: int = 20000,
    soft_trunk_gate: bool = False,
):
    cfg = dict(tag=tag or "s0", seed=seed, n_proc=n_proc,
               seg_H=P.SEG_H, frame_skip=P.FRAME_SKIP, d_fb=P.D_FB,
               explore_sigma=P.EXPLORE_SIGMA,
               deltas=[int(x) for x in deltas.split(",") if x],
               arms=[x for x in arms.split(",") if x],
               kp_grid=[float(x) for x in KP_GRID.split(",")],
               kd_grid=[float(x) for x in KD_GRID.split(",")],
               cem_iters=4, cem_init_sigma=0.8, cem_elite_frac=0.125, vel_pen=0.5,
               n_eval=n_eval, n_rt=n_rt, batch=batch, n_warm=n_warm, n_cand=n_cand,
               n_score=n_score, n_slot=n_slot,
               # --- the treatment
               n_cycles=n_cycles, batch_pr=batch_pr,
               m_cand=m_cand, m_member=m_member, n_poison=n_poison,
               k_prop=k_prop, eps_act=eps_act, explore_eps=explore_eps,
               force_window=force_window, pi_filter=pi_filter, pi_band=pi_band,
               ref_tag=ref_tag,
               prop_hidden=128, prop_layers=2, prop_lr=1e-3, prop_cap=40000,
               prop_steps=60, prop_batch=128,
               span_hidden=256, span_layers=2, span_lr=1e-3, span_steps=60, span_batch=128,
               span_cap=4000, span_hold_cap=64, span_hold_frac=0.25,
               parity_tau=parity_tau, parity_every=parity_every, parity_min=4,
               probe_every=probe_every,
               dperf_ema=0.2, dperf_sil=0.05,
               # --- the trunk
               trunk_tol_rel=trunk_tol_rel, trunk_tol_abs=trunk_tol_abs,
               trunk_gate_binding=True,
               bc_cycles=bc_cycles, bc_batch=24, bc_explore=P.EXPLORE_SIGMA,
               bc_hidden=256, bc_layers=2, bc_lr=1e-3, bc_steps=bc_steps, bc_batch_sz=1024)
    if soft_trunk_gate:
        cfg["trunk_gate_binding"] = False
    if quick:
        cfg.update(tag=tag or "ssmoke", n_proc=8, n_cycles=4, batch_pr=6, n_eval=8, n_rt=8,
                   batch=6, n_warm=2, n_cand=4, n_score=4, n_slot=3, m_cand=12, m_member=2,
                   n_poison=2, k_prop=2, parity_every=2, probe_every=2,
                   bc_cycles=1, bc_steps=400, bc_batch=8, prop_steps=10, span_steps=10,
                   trunk_gate_binding=False)
    # NB `_slot_seam` is filled in REMOTELY (in `run_solo`): a Modal local entrypoint runs on a
    # client with no numpy, and `offbook/nets.py` imports it (acappella's Gotcha).
    if spawn:
        c = run_solo.spawn(cfg)
        print(f"[spawn] {c.object_id}; results land at /data/practice_solo/{cfg['tag']}/solo.json")
        return
    r = run_solo.remote(cfg)
    print("\n".join(r["log"][-80:]))
