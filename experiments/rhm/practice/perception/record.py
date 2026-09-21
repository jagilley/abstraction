"""Re-run `conductor`'s `outer_yield` arm with a pure recorder attached, so the solved
configurations the yield thermostat mined from exist on disk.

WHY A RE-RUN. `cd_s0` logs the miner's STATE every cycle (`log["miner"]`, `obs_hist`, and
`keys_at_support`) but never the configurations themselves, and a next-token reader cannot
be shown a count. Nothing in `conductor.py` is modified: two module-level functions are
wrapped by a recorder that copies tensors already in flight and then calls the original.

  `macros.parse_features`  called once per cycle with `mine_src` -- the solved chosen
                           answers the miner reads -- and returns their level-1 feature
                           parse. Both are copied.
  `conductor.finetune_generator`  called exactly once per cycle, immediately after the
                           mining block, with `solved` (every successful beam tip). It is
                           the cycle sentinel AND the source of the learner's own
                           production stream.

Neither wrapper draws RNG, allocates on device, or touches gradients, so the arm is
expected to be BIT-IDENTICAL to `cd_s0/outer_yield`; `check_fidelity` asserts it.

Run (from experiments/):
  modal run --detach -m rhm.practice.perception.record::record_run --tag pc0
"""

import json
import os
import time

import numpy as np

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, volume
from rhm.practice.perception.common import REMOTE, app

# `cd_s0`'s launch literals (conductor/FILES.md Reproduce), minus the arms and the yoke:
# everything else is conductor's own default, and `endo_price` is `cd_s0/setup.json`'s
# measured 258 g/read.
CD_S0 = dict(
    eras="1:25:48,2:12:40,3:6:12,4:3:9,5:1:7", era_caps="60,50,15,12,9",
    endo_price=258, budget=8, g_budget=482, n_corrupt=1, mine_cap=8, gy_level=4,
    entry_rec=True, span_min_hold=128, collect_task_matched=True, tm_episodes=8192,
    recert_every=5, n_aud=192, probe_every=8, n_rt=384, n_score=256,
    sil_cv=0.15, lp_min_drop=0.10, seed=0,
)


class Recorder:
    """Copies what is already in flight. Nothing else."""

    def __init__(self):
        self.pending = []          # parse_features calls since the last cycle boundary
        self.mine_x, self.mine_pf, self.mine_cyc = [], [], []
        self.sol_x, self.sol_cyc = [], []
        self.cycle = 0

    def install(self, CD, MC):
        rec = self
        orig_pf = MC.parse_features
        orig_ft = CD.finetune_generator

        def parse_features(generator, x, *a, **kw):
            out = orig_pf(generator, x, *a, **kw)
            try:
                rec.pending.append((x.detach().cpu().numpy().astype(np.uint8),
                                    out.detach().cpu().numpy().astype(np.uint8)))
            except Exception as e:                      # an instrument may never fail a run
                print(f"[rec] parse_features copy failed: {e}", flush=True)
            return out

        def finetune_generator(generator, opt, new_leaves, *a, **kw):
            rec.cycle += 1
            try:
                sx = np.asarray(new_leaves.detach().cpu().numpy(), np.uint8)
                if sx.size:
                    rec.sol_x.append(sx)
                    rec.sol_cyc.append(np.full(sx.shape[0], rec.cycle, np.int32))
                if rec.pending:
                    x, pf = rec.pending[-1]             # the mining call is the last one
                    rec.mine_x.append(x)
                    rec.mine_pf.append(pf)
                    rec.mine_cyc.append(np.full(x.shape[0], rec.cycle, np.int32))
            except Exception as e:
                print(f"[rec] cycle copy failed: {e}", flush=True)
            rec.pending = []
            return orig_ft(generator, opt, new_leaves, *a, **kw)

        MC.parse_features = parse_features
        CD.finetune_generator = finetune_generator

    def arrays(self):
        cat = lambda xs, d: (np.concatenate(xs) if xs else np.zeros((0, d), np.uint8))
        catc = lambda xs: (np.concatenate(xs) if xs else np.zeros((0,), np.int32))
        return {"mine_x": cat(self.mine_x, 64), "mine_pf": cat(self.mine_pf, 32),
                "mine_cyc": catc(self.mine_cyc), "solved_x": cat(self.sol_x, 64),
                "solved_cyc": catc(self.sol_cyc), "n_cycles": np.int64(self.cycle)}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=5 * 3600, memory=24576)
def record_run(tag: str = "pc0", arm: str = "outer_yield", cd_tag: str = "",
               quick: bool = False, eras: str = "", era_caps: str = ""):
    """`conductor_run` for ONE arm, with the recorder installed. `cd_tag` is where the
    conductor's own artefacts land (default `perception_<tag>`, beside its own tags).
    `quick` is the mechanics smoke and is never a measurement."""
    import resource
    import torch
    from rhm.practice.conductor import conductor as CD
    from rhm.practice.ratchet import macros as MC

    cd_tag = cd_tag or f"perception_{tag}"
    kw = dict(CD_S0)
    if eras:
        kw["eras"] = eras
    if era_caps:
        kw["era_caps"] = era_caps
    if quick:
        kw.update(quick=True, eras=eras or "1:25:4,2:12:4,3:6:4",
                  era_caps=era_caps or "8,8,8", tm_episodes=512, n_aud=48)
    rec = Recorder()
    rec.install(CD, MC)
    t0 = time.time()
    out = CD.conductor_run.local(tag=cd_tag, arms=arm, **kw)
    arrs = rec.arrays()
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(f"{outdir}/pieces.npz", **arrs)
    meta = {"tag": tag, "cd_tag": cd_tag, "arm": arm, "literals": kw, "quick": bool(quick),
            "n_cycles": int(arrs["n_cycles"]),
            "n_mine": int(arrs["mine_x"].shape[0]),
            "n_solved": int(arrs["solved_x"].shape[0]),
            "conductor_outdir": f"{DATA_DIR}/rhm_practice_conductor/{cd_tag}",
            "elapsed_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    with open(f"{outdir}/record_meta.json", "w") as f:
        json.dump(meta, f, indent=1, cls=NumpyEncoder)
    volume.commit()
    print(f"[record] {meta}", flush=True)
    return meta


@app.function(volumes={DATA_DIR: volume}, timeout=1800, memory=4096)
def check_fidelity(tag: str = "pc0", cd_tag: str = "", ref_tag: str = "cd_s0",
                   arm: str = "outer_yield"):
    """The gate: the recorded arm against the banked one, series by series, and the
    recorded mine rows against the run's own `n_mined`."""
    volume.reload()
    cd_tag = cd_tag or f"perception_{tag}"
    root = f"{DATA_DIR}/rhm_practice_conductor"
    a = json.load(open(f"{root}/{cd_tag}/{arm}/results.json"))
    b = json.load(open(f"{root}/{ref_tag}/{arm}/results.json"))
    out = {"tag": cd_tag, "ref": ref_tag, "arm": arm, "series": {}}
    for key in ("e", "gloss", "vloss", "succ", "dres", "t_cum", "n_solved", "n_mined",
                "e_practice"):
        xa, xb = a["log"].get(key) or [], b["log"].get(key) or []
        n = min(len(xa), len(xb))
        d = [abs((xa[i] or 0) - (xb[i] or 0)) for i in range(n)]
        out["series"][key] = {"n_a": len(xa), "n_b": len(xb),
                              "max_abs_delta": (max(d) if d else None)}
    for key in ("3", "4", "5", "6"):
        xa = (a.get("obs_hist") or {}).get(key) or []
        xb = (b.get("obs_hist") or {}).get(key) or []
        n = min(len(xa), len(xb))
        out["series"][f"obs_hist[{key}]"] = {
            "n_a": len(xa), "n_b": len(xb),
            "max_abs_delta": (max(abs(xa[i] - xb[i]) for i in range(n)) if n else None)}
    out["loop_a"] = {k: a["loop"][k] for k in ("commit_cycles", "commit_levels",
                                               "advance_cycles", "chosen_cycles",
                                               "capped_cycles", "n_cycles")}
    out["loop_b"] = {k: b["loop"][k] for k in ("commit_cycles", "commit_levels",
                                               "advance_cycles", "chosen_cycles",
                                               "capped_cycles", "n_cycles")}
    out["actions_equal"] = out["loop_a"] == out["loop_b"]
    z = np.load(f"{DATA_DIR}/{REMOTE}/{tag}/pieces.npz")
    mc = z["mine_cyc"]
    n_rec = np.bincount(mc, minlength=len(a["log"]["cycle"]) + 1)[1:]
    n_mined = np.asarray(a["log"]["n_mined"], np.int64)
    out["mine_rows_match"] = bool(len(n_rec) == len(n_mined) and (n_rec == n_mined).all())
    out["mine_rows_mismatch_cycles"] = [int(i + 1) for i in range(min(len(n_rec), len(n_mined)))
                                        if n_rec[i] != n_mined[i]]
    out["n_mine"] = int(z["mine_x"].shape[0])
    out["n_solved"] = int(z["solved_x"].shape[0])
    dst = f"{DATA_DIR}/{REMOTE}/{tag}/fidelity.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1, cls=NumpyEncoder)
    volume.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out
