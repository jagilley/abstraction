"""corridor — E2 PASS 2: the executor-side (span/corridor) consolidation object.

WHAT PASS 1 RE-SCOPED OUT, AND WHY THIS FILE EXISTS. `continuo.py` (pass 1) asked whether a
sub-saturation forward self-model's residual reads CONSOLIDATION STATE — which of the learner's
tokens are commands (routed, can't-decompose) and which are still data (enumerated). It asked it
of pi, the ROUTING head, because its donor `audiation/au_s0` runs `span_mode = False` and has no
corridor head at all (pass 1 design call 1c). The executor-side half — a macro EXECUTED in one
pass by a head, against a macro still spelled out by the max-sum DP over the committed table —
was named as the follow-up. `intonation`'s stack makes it available: `native/span`'s `SpanHead`
is live there and, at `span_tau_fire = 0.50`, FALLIBLE.

THE OBJECT, SAID PRECISELY. On the executor side the instruction/data bit is not a property of a
vocabulary slot's LEVEL, it is a property of its GATE:

  * a slot whose parity gate is OPEN is materialised by `SpanHead.emit` in ONE pass — no DP, no
    table consulted at execution time. It is a COMMAND: routed, can't-decompose.
  * a slot whose gate is CLOSED falls back to `macros.apply_any` — mask the span, one per-block
    infill read, a max-sum DP over T[l] -> ... -> T[1], render. It is DATA: enumerated.

This bit is strictly better-conditioned than pass 1's. It varies WITHIN a level and WITHIN a slot
over time (`ma_s0/mperf_log` logs 44 gate events over 153 cycles, including re-closures), so it
is not collinear with token age the way "L2 macro vs base move" was, and the age control that
killed pass 1's headline is built in from the start rather than bolted on. A second, stricter
grade rides free: `open_tau` — parity >= 0.95, the donor's own criterion — which the run logs
beside `open` at every gate event.

WHAT THIS FILE IS. A SNAPSHOT-ENABLED fork of the `intonation` stack, and nothing else. It adds
no arm, no knob that changes any trajectory, and no line inside `intonation.py` (another agent is
concurrently forking that lineage). The whole fork is:

  1. a monkeypatch of `span_net.parity` — the one function `run_arm` calls exactly once per
     cycle with (generator, head, executor, slots) all in hand — which calls the real parity,
     then writes a per-cycle snapshot, then returns the real parity's answer unchanged;
  2. `IN.REMOTE` retargeted to `rhm_practice_corridor`, so no `intonation` tag is written to;
  3. two entrypoints: `corridor_gf` (the inertness gate) and `corridor_run` (the paid run).

INERTNESS, ASSERTED THREE WAYS (gate X below). The snapshot may not perturb the run:
  X-1  in-hook, every cycle: the torch CPU RNG state and every CUDA RNG state are byte-identical
       across the hook, and the executor's buffer/held-out sizes and the slots' `open` flags are
       unchanged. Cheap, always on, so the main run itself certifies its own instrument.
  X-2  `corridor_gf`: the same arm run with the hook OFF and with the hook ON, in ONE process on
       ONE shared dict, log series compared — beside a DONOR SELF-REPLAY control (the same arm
       twice with the hook off), because without it GPU nondeterminism and a real bug look
       identical (`intonation.fidelity_smoke`'s idiom).
  X-3  the arm used for X-2 is `perf_given`, which holds the true tables and therefore mints
       every slot at c1: at smoke sizes a loop arm never commits, no slot is ever minted, and a
       gate over a hook that never fired would certify nothing (`intonation`'s own reason for
       carrying the `perf_given` family into preflight).

THE PROBE (pass 1 design call 2, carried over). The head's input is a MASKED OBSERVATION — the
sequence with the called slot's span blanked. The probe is therefore a fixed set of underlying
sequences drawn once from `shared["replay"]["x"]` — the plant's own replay diet, the learner's
working distribution — with each slot's own span masked per slot, exactly as `SpanExecutor.apply`
constructs its input. NOT `ex.hold`: that is the parity gate's own metering set, i.e. an eval,
and pass 1's design call 2 is that the gauge must not be audition-shaped. The probe SEQUENCES are
shared across slots (only the mask moves), which is what makes one slot's residual column
comparable with another's — the decode's rows are slots, so per-slot probe sets would leak slot
identity straight into the classifier.

WHAT IS STORED, AND WHY IT IS THE REDUCED READ RATHER THAN `pooled`. The offline instrument needs
(a) what the head reads and (b) what the head emits. (b) is small. (a) is `pooled`, which is
(P, n_blocks=32, dim=96) PER SLOT because the mask moves — 28x that per cycle is ~2 GB over a
run. Stored instead are the three reads the head's own `span_state` is built out of, per slot:

    ctxmean   pooled.mean(over all blocks)        the head's `ctx(...)` argument
    spanmean  pooled[blk0 : blk0+span].mean       the span's own evidence, order-collapsed
    first     pooled[blk0]                        the span's leading block

`span_state` is `ctx(pooled.mean(1)) + slot(sid) + sum_j in_proj[j](pooled[blk0+j])`, so the FM
is handed the head's global read exactly, and the span read up to the head's per-offset
weighting. That approximation is a COST, stated here rather than hidden: it means part of what
the FM cannot predict is input it never saw, not only structure it failed to theorise. It buys a
fixed 3x96 input dimension for every slot at every level — no zero-padding, so the FM's input
carries no dimensional signature of the slot's level, which is the confound this pass is built
around. `dim` is 96 and `max_span` 8, so the alternative (the full per-offset read, padded) is
864 dims and pushes the sub-saturation budget onto the LayerNorm.

THE CONTROL TARGET (pass 1 design call 1b's analogue). `trunk` returns `(pooled, feature_head
(pooled))` — the per-block level-1 feature logits, the DP's own evidence. Those are computed with
NO slot conditioning whatsoever. Stored beside the head's logits, they are the control target:
anything that reorganises at a gate event on BOTH is the state distribution or the calendar, not
the command port.

`dp_features(logits, move, s)` is stored too — the DP's spelling of the same span on the same
row, which is the head's parity target — so the offline instrument has the per-probe-row
execution error `e` (graded Hamming) and the exact-match bit for free, on a fixed distribution,
every cycle. That is `intonation`'s delta_perf numerator measured on a probe rather than on the
beam, and it is what makes "how consolidated is this slot" a continuous readout rather than a
latch.

NOT DONE HERE. No fitting, no decode, no guard, no figure: this file produces bytes. The
instrument is `corridor_fit.py`, offline, CPU, no torch.
"""

import json
import os
import sys
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, volume
from rhm.practice.intonation import intonation as IN
from rhm.practice.native.span import span_net as SN

# `rhm.shared.image`'s package spec, plus one ignore rule and nothing else. THE REASON IS
# OPERATIONAL, not scientific: `add_local_python_source` stats the whole `rhm` tree, and a
# sibling node's live `results/*.log` being appended to by a concurrently-running agent fails
# the build with "modified during build process". The pip pins are `rhm.shared`'s verbatim, so
# the container is the donor's.
from modal.file_pattern_matcher import NON_PYTHON_FILES   # noqa: E402


def _ignore(p):
    return bool(NON_PYTHON_FILES(p)) or "results" in p.parts or p.suffix == ".log"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)

# Its OWN app and its OWN remote root, per the arc's convention: no `intonation` tag is ever
# written to, and `modal run` builds only this node's two functions.
app = modal.App("rhm-practice-corridor", image=image)
REMOTE = "rhm_practice_corridor"

# --------------------------------------------------------------------------- #
# the hook
# --------------------------------------------------------------------------- #

_S = {
    "on": False,
    "n_probe": 64,
    "probe_seed": 20260831,
    "X": None,            # (P, length) long, cpu — frozen once per process
    "outroot": None,
    "manifest": {},
    "n_snap": 0,
    "t_snap": 0.0,
    "rng_checks": 0,
}


def _run_arm_frame():
    """The one frame that has `cyc`, `arm`, `shared` and `cfg` in scope. Read-only."""
    f = sys._getframe()
    while f is not None:
        if f.f_code.co_name == "run_arm" and "cyc" in f.f_locals and "shared" in f.f_locals:
            return f
        f = f.f_back
    raise RuntimeError("corridor: `run_arm` frame not found — the hook's host moved")


def _freeze_probe(shared, dev):
    import torch
    if _S["X"] is not None:
        return _S["X"].to(dev)
    rx = shared["replay"]["x"]
    n = int(rx.shape[0])
    k = int(min(_S["n_probe"], n))
    idx = np.sort(np.random.default_rng(_S["probe_seed"]).choice(n, size=k, replace=False))
    _S["X"] = rx[torch.from_numpy(idx)].detach().cpu().clone().long()
    _S["probe_idx"] = idx.tolist()
    print(f"[corridor] probe frozen: {k} of {n} replay states, seed {_S['probe_seed']}",
          flush=True)
    return _S["X"].to(dev)


def _snapshot(par, core, head, ex, slots, s, v):
    import torch
    fr = _run_arm_frame()
    L = fr.f_locals
    cyc, arm, shared, cfg = L["cyc"], L["arm"], L["shared"], L["cfg"]
    era = L.get("era") or {}
    live = [(k, i) for k, i in slots.items() if i.get("move") is not None]
    if not live:
        return
    dev = next(head.parameters()).device
    X = _freeze_probe(shared, dev)
    P = int(X.shape[0])
    max_span = int(s ** (int(cfg["max_macro_level"]) - 1))

    n = len(live)
    feat = np.zeros((n, P, 3, int(cfg["state_dim"])), dtype=np.float16)
    lg0 = np.zeros((n, P, v), dtype=np.float16)
    lgm = np.zeros((n, P, v), dtype=np.float16)
    bl0 = np.zeros((n, P, v), dtype=np.float16)
    blm = np.zeros((n, P, v), dtype=np.float16)
    emit = np.full((n, P, max_span), 255, dtype=np.uint8)
    dpf = np.full((n, P, max_span), 255, dtype=np.uint8)
    err = np.zeros((n, P), dtype=np.float16)
    exact = np.zeros((n, P), dtype=np.uint8)
    meta = []

    with torch.no_grad():
        for a, (key, info) in enumerate(live):
            move = info["move"]
            blk0, span, sid = int(move["blk0"]), int(move["span"]), int(info["id"])
            pos = SN.span_positions(move, P, s, dev)
            obs = X.clone().scatter_(1, pos, torch.full_like(pos, -1))
            pooled, blogits = SN.trunk(core, obs)
            sid_t = torch.full((P,), sid, dtype=torch.long, device=dev)
            lg, em = head(pooled, blk0, span, sid_t)          # free run, as `emit` does
            tgt = SN.dp_features(blogits, move, s)
            sl = pooled[:, blk0:blk0 + span, :]
            feat[a, :, 0] = pooled.mean(dim=1).float().cpu().numpy()
            feat[a, :, 1] = sl.mean(dim=1).float().cpu().numpy()
            feat[a, :, 2] = pooled[:, blk0, :].float().cpu().numpy()
            lg0[a] = lg[:, 0, :].float().cpu().numpy()
            lgm[a] = lg.mean(dim=1).float().cpu().numpy()
            bs = blogits[:, blk0:blk0 + span, :]
            bl0[a] = bs[:, 0, :].float().cpu().numpy()
            blm[a] = bs.mean(dim=1).float().cpu().numpy()
            emit[a, :, :span] = em.to(torch.uint8).cpu().numpy()
            dpf[a, :, :span] = tgt.to(torch.uint8).cpu().numpy()
            hit = (em == tgt)
            err[a] = (1.0 - hit.float().mean(dim=1)).cpu().numpy()
            exact[a] = hit.all(dim=1).to(torch.uint8).cpu().numpy()
            cell = par.get(key) or {}
            meta.append({
                "key": key, "id": sid, "level": int(info["level"]), "node": int(info["node"]),
                "blk0": blk0, "span": span,
                # `open` is still the PRE-gate value here: `run_arm` updates it from `par`
                # after this call returns. Both are recorded so neither has to be inferred.
                "open_pre": bool(info["open"]),
                "parity_exact": cell.get("exact"), "parity_block": cell.get("block"),
                "parity_n": int(cell.get("n") or 0),
            })

    tau_fire = cfg.get("span_tau_fire")
    tau_fire = cfg["span_tau"] if tau_fire is None else float(tau_fire)
    for mrec in meta:
        pe = mrec["parity_exact"]
        mrec["open_post"] = bool(pe is not None and pe >= tau_fire)
        mrec["open_tau"] = bool(pe is not None and pe >= float(cfg["span_tau"]))

    d = os.path.join(_S["outroot"], arm, "snapshots")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, f"span_c{int(cyc):04d}.npz"),
        feat=feat, lg0=lg0, lgm=lgm, bl0=bl0, blm=blm, emit=emit, dp=dpf, err=err,
        exact=exact,
        keys=np.array([mrec["key"] for mrec in meta]),
        sid=np.array([mrec["id"] for mrec in meta], dtype=np.int32),
        level=np.array([mrec["level"] for mrec in meta], dtype=np.int32),
        node=np.array([mrec["node"] for mrec in meta], dtype=np.int32),
        blk0=np.array([mrec["blk0"] for mrec in meta], dtype=np.int32),
        span=np.array([mrec["span"] for mrec in meta], dtype=np.int32),
        open_pre=np.array([mrec["open_pre"] for mrec in meta], dtype=np.uint8),
        open_post=np.array([mrec["open_post"] for mrec in meta], dtype=np.uint8),
        open_tau=np.array([mrec["open_tau"] for mrec in meta], dtype=np.uint8),
        parity=np.array([np.nan if mrec["parity_exact"] is None else mrec["parity_exact"]
                         for mrec in meta], dtype=np.float32),
        parity_n=np.array([mrec["parity_n"] for mrec in meta], dtype=np.int32),
        cycle=np.int32(cyc), era=np.int32(era.get("level", 0)), n_probe=np.int32(P),
    )
    _S["manifest"].setdefault(arm, []).append(
        {"cycle": int(cyc), "n_slots": n, "slots": meta, "era": era.get("name")})
    _S["n_snap"] += 1


def _install():
    """Wrap `span_net.parity`. `intonation` reaches it as `SN.parity` at call time, so one
    assignment on the module object covers every caller."""
    import torch
    orig = SN.parity
    if getattr(orig, "_corridor", False):
        return
    if not _S["on"]:
        return

    def patched(core, head, ex, slots, s, v, min_hold=256, chunk=4096):
        par = orig(core, head, ex, slots, s, v, min_hold=min_hold, chunk=chunk)
        # ---- X-1: the hook is inert -------------------------------------------------------
        st = torch.get_rng_state()
        cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        before = (sorted((k, int(t.shape[0])) for k, t in ex.buf.items()),
                  sorted((k, int(t.shape[0])) for k, t in ex.hold.items()),
                  sorted((k, bool(i["open"])) for k, i in slots.items()))
        t0 = time.time()
        _snapshot(par, core, head, ex, slots, s, v)
        _S["t_snap"] += time.time() - t0
        assert torch.equal(torch.get_rng_state(), st), \
            "corridor X-1: the snapshot moved the torch CPU RNG state"
        if cst is not None:
            now = torch.cuda.get_rng_state_all()
            assert len(now) == len(cst) and all(torch.equal(a, b) for a, b in zip(now, cst)), \
                "corridor X-1: the snapshot moved a CUDA RNG state"
        after = (sorted((k, int(t.shape[0])) for k, t in ex.buf.items()),
                 sorted((k, int(t.shape[0])) for k, t in ex.hold.items()),
                 sorted((k, bool(i["open"])) for k, i in slots.items()))
        assert before == after, "corridor X-1: the snapshot mutated executor or slot state"
        _S["rng_checks"] += 1
        return par

    patched._corridor = True
    patched._orig = orig
    SN.parity = patched
    print("[corridor] hook installed on span_net.parity", flush=True)


def _uninstall():
    p = SN.parity
    if getattr(p, "_corridor", False):
        SN.parity = p._orig
        print("[corridor] hook removed", flush=True)


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def corridor_gf(tag: str = "co_gf", cycles: int = 6, seed: int = 0, n_probe: int = 32):
    """GATE X-2/X-3 — the snapshot hook is INERT.

    One process, one shared dict, one arm (`perf_given`: true tables, so every slot mints at c1
    and the hook actually fires at smoke sizes). Three runs: hook OFF twice (the donor
    self-replay control, which is what makes a non-zero delta interpretable at all) and hook ON
    once. Reported as max |delta| over the log series, fork vs control.
    """
    import torch
    cfg = IN._d6_cfg(era_cycles=cycles, seed=seed, probe_every=2, probe_widths=(1, 2),
                     controller_steps=800, generator_steps=800, value_steps=800,
                     reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                     n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                     checkpoint_every=10 ** 9, gen_steps=5, sil_min_cycle=2, prop_warmup=1,
                     prop_steps=12, prop_buf_cap=20_000, max_macro_level=4,
                     tm_episodes=512, mine_cap=8,
                     span_tau_fire=0.50, span_min_hold=8)
    ers = IN.parse_eras("1:25,2:12,3:6")
    for e_ in ers:
        e_["cycles"] = cycles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    IN._install_identity_miner(cfg["mine_support"])
    IN._install_entry_recorder()
    shared = IN._spiral_shared(cfg, device, ers)
    refs = IN.measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
              "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    arm = "perf_given"
    _S["on"] = False
    _uninstall()
    a1 = IN.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/off1", device)
    a2 = IN.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/off2", device)
    _S.update(on=True, n_probe=int(n_probe), X=None, manifest={}, n_snap=0, t_snap=0.0,
              rng_checks=0, outroot=f"{outdir}/on")
    _install()
    b = IN.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/on", device)
    _uninstall()
    _S["on"] = False

    # `run_arm`'s RETURN dict carries `log`/`events` but not the per-slot records — those go
    # only into the `results.json` it writes. Read them back rather than reaching into the
    # return value (the first cut of this gate crashed on exactly that).
    def _rec(sub):
        with open(os.path.join(outdir, sub, arm, "results.json")) as fh:
            return json.load(fh)
    r_off1, r_off2, r_on = _rec("off1"), _rec("off2"), _rec("on")

    worst_fork, worst_ctrl, rows = 0.0, 0.0, {}
    for k in SERIES:
        x, y, z = (np.asarray(a1["log"][k], float), np.asarray(b["log"][k], float),
                   np.asarray(a2["log"][k], float))
        n = min(len(x), len(y), len(z))
        df = float(np.max(np.abs(x[:n] - y[:n]))) if n else float("nan")
        dc = float(np.max(np.abs(x[:n] - z[:n]))) if n else float("nan")
        rows[k] = {"fork": df, "control": dc, "n": int(n)}
        worst_fork = max(worst_fork, df)
        worst_ctrl = max(worst_ctrl, dc)
    gates = {
        "n_cycles": [len(a1["log"]["cycle"]), len(a2["log"]["cycle"]), len(b["log"]["cycle"])],
        "gate_events_equal": r_off1["gate_events"] == r_on["gate_events"],
        "gate_events_equal_control": r_off1["gate_events"] == r_off2["gate_events"],
        "slot_events_equal": r_off1["slot_events"] == r_on["slot_events"],
        "events_equal": r_off1["events"] == r_on["events"],
        "perf_cells_equal": r_off1["perf_cells"] == r_on["perf_cells"],
        "n_gate_events": len(r_on["gate_events"]),
        "n_snapshots": _S["n_snap"], "rng_checks": _S["rng_checks"],
        "snap_seconds": round(_S["t_snap"], 1),
        "worst_fork": worst_fork, "worst_control": worst_ctrl, "series": rows,
        "probe": {"n": len(_S.get("probe_idx") or []), "seed": _S["probe_seed"]},
    }
    with open(os.path.join(outdir, "gf.json"), "w") as fh:
        json.dump(gates, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(_S["outroot"], "manifest.json"), "w") as fh:
        json.dump(_S["manifest"], fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[X-2] worst |delta| fork={worst_fork:.3e}  control={worst_ctrl:.3e}", flush=True)
    print(f"[X-3] {_S['n_snap']} snapshots, {gates['n_gate_events']} gate events, "
          f"X-1 checked {_S['rng_checks']}x, snapshot cost {_S['t_snap']:.1f}s of "
          f"{time.time() - started:.0f}s", flush=True)
    assert _S["n_snap"] > 0, "X-3: the hook never fired — the gate would certify nothing"
    assert worst_fork <= worst_ctrl + 1e-12, \
        f"X-2 FAIL: fork delta {worst_fork:.3e} exceeds the self-replay control {worst_ctrl:.3e}"
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}", flush=True)
    return gates


def _raw(fn):
    """`intonation_run`'s underlying Python function, whatever this Modal version calls it."""
    for a in ("local", "get_raw_f", "_get_raw_f"):
        f = getattr(fn, a, None)
        if f is not None:
            return f if a == "local" else f()
    return fn


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def corridor_run(tag: str = "co_s0", arms: str = "mperf_log", n_probe: int = 64,
                 probe_seed: int = 20260831,
                 # the only `ma_s0` flags that are not already `intonation_run` defaults
                 n_pr: int = 24, pr_width: int = 8, endo_price: int = 267, seed: int = 0,
                 eras: str = IN.INTON_LADDER, era_caps: str = IN.INTON_CAPS,
                 span_tau_fire: float = 0.50, max_macro_level: int = 4):
    """The paid run: `intonation_run`'s body verbatim, with the hook installed and the remote
    root moved to `rhm_practice_corridor`.

    Defaults are `ma_s0`'s configuration exactly — the strongly-metered regime (`n_pr` 24,
    `pr_width` 8) at A3's ladder, caps and floors. Every other flag `ma_s0` passed is already
    the entrypoint's own default (checked against `intonation/FILES.md`'s reproduce block), so
    nothing is restated here that could drift from it.
    """
    _S.update(on=True, n_probe=int(n_probe), probe_seed=int(probe_seed), X=None, manifest={},
              n_snap=0, t_snap=0.0, rng_checks=0,
              outroot=f"{DATA_DIR}/{REMOTE}/{tag}")
    os.makedirs(_S["outroot"], exist_ok=True)
    IN.REMOTE = REMOTE                       # so `intonation_run` writes under OUR root
    _install()
    try:
        out = _raw(IN.intonation_run)(
            tag=tag, arms=arms, n_pr=n_pr, pr_width=pr_width, endo_price=endo_price,
            seed=seed, eras=eras, era_caps=era_caps, span_tau_fire=span_tau_fire,
            max_macro_level=max_macro_level)
    finally:
        _uninstall()
        with open(os.path.join(_S["outroot"], "manifest.json"), "w") as fh:
            json.dump({"arms": _S["manifest"], "n_snap": _S["n_snap"],
                       "snap_seconds": round(_S["t_snap"], 1),
                       "rng_checks": _S["rng_checks"],
                       "probe": {"n": len(_S.get("probe_idx") or []),
                                 "seed": _S["probe_seed"],
                                 "idx": _S.get("probe_idx")}},
                      fh, indent=2, cls=NumpyEncoder)
        volume.commit()
    print(f"[corridor] {_S['n_snap']} snapshots written, X-1 checked {_S['rng_checks']}x, "
          f"snapshot cost {_S['t_snap']:.0f}s", flush=True)
    return out
