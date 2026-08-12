"""aleatoric_flip: build the DIRECT OVER-WRITING damage channel that attenuation-on-success
presupposes, and see whether delta (or ensemble disagreement) can protect against it.

Program: `ideas/performance_error_is_the_bridge.md` S14(b) (Kim, Parvin & Ivry's effect is
*attenuation on success* -- a protection mechanism) and
`ideas/practice_manufactures_its_own_credit.md` S1 component (2), metering.
Direct parent (forked wholesale): `mjc/practice/priced_plasticity/priced_plasticity.py`
-- same two-corridor geometry, same priced cell (fm_hidden=32, n_replay=0, static
normalizer, free spend), same corrected delta, same gain law, same frontier grading.

    WHY. `priced_plasticity` graded delta against the whole uniform-learning-rate Pareto
    frontier and found it sits ON or INSIDE it: delta decomposes into a ~1.6x learning-rate
    increase plus an allocation contributing nothing. But it also found WHY, and the reason
    is about the SUBSTRATE. In every geometry this arc has used, retention tracks plasticity
    spent on the mastered corridor POSITIVELY -- the corridor is eroded by interference from
    updates ELSEWHERE, and updates ON it are repair. A matched-total arm spending 59% less on
    the mastered region ended 8% WORSE on retention. Kim et al.'s attenuation-on-success
    presupposes the opposite: that updating a region is what DAMAGES it. That channel has
    never existed here, so protection has never been expressible, let alone tested.

    THE MANIPULATION (the one changed thing). A region the FM has ALREADY MASTERED
    (deterministic command rotation phi, low error, on the eval path) has ALEATORIC NOISE
    SWITCHED ON at t=0, WITH ITS ROTATION UNCHANGED. `pusher_env.py`'s per-region `noise` is
    a zero-mean stochastic force (`amp * N(0,1)` per substep), so E[s'|s,u] in that region is
    unchanged -- the mastered model remains correct in conditional mean -- while every sample
    becomes a noisy draw around it. Fitting those samples can only move the model off a
    correct answer; declining to fit preserves it. That is a direct over-writing damage
    channel. (Contrast the existing R-noise decoy, kept here unchanged: aleatoric from the
    start, so there was never mastered content to destroy, and off the eval path, so its
    damage never reached behaviour.)

    HOW THE ENV DOES IT (no change to pusher_env.py). A region spec gains a `noise_pre`
    field: the noise amplitude the region carries in the PRE-drift world. The pre-drift world
    is built with `noise_pre`, the post-drift world with `noise`. Defaulting
    `noise_pre = noise` reproduces the parent exactly. `pusher_env._apply_rot_regions`
    already sums a region's rotation and its stochastic force, so a region carrying both is
    supported as-is.

    THE RETENTION READOUT. Once the region is noisy, its FM probe error is dominated by the
    irreducible floor and says nothing about what the model retained. So a THIRD world is
    built -- `clean` = the post-drift world with the flip region's noise put back to
    `noise_pre` -- and the flip region is probed (and the mastered corridor graded) there.
    Because the noise is zero-mean, the clean world's Delta s IS the conditional mean, so the
    clean probe measures exactly the model's deviation from the truth. This is checked, not
    assumed: `--cond-mean-check` measures E[s'|s,u] with noise on against the noise-off value
    at matched (s, v, u), with a clean control region for the numerical floor.

    ARMS. `fixed` at an lr grid traces the uniform frontier (the parent's grading
    instrument). Query arms:
      * `delta`      -- the committed delta, unchanged (tau = pretrain MAD: a binary gate)
      * `delta:tauon`-- tau re-scaled to the ONLINE delta spread (the graded gain the law was
                        supposed to be); the miscalibrated one is kept so the arc stays
                        comparable
      * `raw_err`    -- any-error-modulated control
      * `disag`      -- ensemble disagreement as the gain. `benchmark_vs_cost` concluded
                        allocation belongs to disagreement, and `estimability` found a
                        perfect benchmark makes delta silent; disagreement can separate
                        aleatoric from epistemic BY CONSTRUCTION (members agree on the mean
                        where content is random). This geometry demands exactly that
                        discrimination.
      * `conj`       -- the conjunction: delta certifies convergence, disagreement certifies
                        reducibility. w = f(delta) * (d / d_ref).
      * `:unit`      -- allocation only, spend clamped to 1 (the decomposition arm)
    The disagreement signal is a SHADOW ensemble (K random-prior members, Osband et al. 2018,
    the form `curiosity_control.py` committed to) trained UNIFORMLY on the same stream, in
    the same order, exactly like `estimability`'s passive panel: it is identical across arms
    and is precomputed once per stream, so no arm's allocation can influence it.

    MODES.
      * `--mode calibrate` -- preconditions, FM probes only (cheap). The conditional-mean
        check; the disagreement profile; and a flip-amplitude x learning-rate ladder on the
        `fixed` arm that asks the load-bearing question: does uniform plasticity actually
        damage the mastered region, and does the damage scale with the noise?
      * `--mode main` -- the full comparison with control grading and the uniform frontier.

Run (from experiments/):
    modal run mjc/practice/aleatoric_flip/aleatoric_flip.py::flip --quick
    python3 mjc/practice/aleatoric_flip/launch_detached.py --mode calibrate --tag afcal_s0
    python3 mjc/practice/aleatoric_flip/launch_detached.py --mode main --tag af_s0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


ARM_KINDS = ("fixed", "raw_err", "delta", "disag", "conj", "omask")


# ------------------------------------------------------------------ arm spec parsing
def parse_arm(token, d_lr, d_spend, d_norm, d_replay, d_flip):
    """`kind[@lr][:spend][:norm][:rN][:xM][:fA][:tauon]` -> spec dict.  Name = the token."""
    parts = token.split(":")
    head = parts[0]
    kind, _, lr = head.partition("@")
    spec = dict(name=token, kind=kind.strip(),
                lr=float(lr) if lr else d_lr,
                spend_mode=d_spend, norm_mode=d_norm, n_replay=d_replay,
                w_mult=1.0, flip_amp=d_flip, tau_mode="pretrain", ens_mode="shared",
                disag_law="lin", mask_w=0.0)
    for p in parts[1:]:
        p = p.strip()
        if p in ("raw_adam", "free", "unit"):
            spec["spend_mode"] = p
        elif p in ("ewma", "static"):
            spec["norm_mode"] = p
        elif p in ("tauon", "taupre"):
            spec["tau_mode"] = "online" if p == "tauon" else "pretrain"
        elif p in ("shared", "boot"):
            spec["ens_mode"] = p
        elif p in ("expd", "lind"):
            spec["disag_law"] = "exp" if p == "expd" else "lin"
        elif p.startswith("r") and p[1:].isdigit():
            spec["n_replay"] = int(p[1:])
        elif p.startswith("x"):
            spec["w_mult"] = float(p[1:])
        elif p.startswith("m"):
            spec["mask_w"] = float(p[1:])
        elif p.startswith("f"):
            spec["flip_amp"] = float(p[1:])
        else:
            raise ValueError(f"unparsed arm qualifier {p!r} in {token!r}")
    assert spec["kind"] in ARM_KINDS, spec["kind"]
    return spec


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_flip(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    MODE = cfg["mode"]
    FLIP_J = [j for j, r in enumerate(REG) if r["noise_pre"] != r["noise"]]
    print(f"[setup] device={device} mode={MODE} T={cfg['T']} batch={cfg['batch']}", flush=True)
    print("[setup] regions: " + "; ".join(
        f"{r['name']}@({r['center'][0]:+.2f},{r['center'][1]:+.2f}) phi={r['phi']:+.2f} "
        f"noise={r['noise_pre']:.0f}->{r['noise']:.0f} pre={r['pre']}" for r in REG), flush=True)
    print(f"[setup] flip regions: {[REG[j]['name'] for j in FLIP_J]}", flush=True)
    for s in cfg["arm_specs"]:
        print(f"[setup/arm] {s['name']:24s} kind={s['kind']:8s} lr={s['lr']:.2e} "
              f"spend={s['spend_mode']:9s} norm={s['norm_mode']:6s} replay={s['n_replay']} "
              f"h={s['fm_hidden']}x{s['fm_layers']} flip={s['flip_amp']:.0f} "
              f"tau={s['tau_mode']} ens={s['ens_mode']}/{s['disag_law']} "
              f"wx={s['w_mult']:.2f} mask={s['mask_w']:.2f}", flush=True)

    def make_env(world: str, flip_amp: float):
        """world in {'pre','post','clean'}.

        pre   -- only `pre` regions, each at its PRE-drift noise (`noise_pre`).
        post  -- every region at its POST-drift noise; flip regions at `flip_amp`.
        clean -- every region at its POST-drift noise EXCEPT flip regions, which are put
                 back to `noise_pre`.  Identical to `post` in every other respect, so the
                 difference between the two worlds is exactly the aleatoric flip.
        """
        d = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                 joint_damping=cfg["damping"], pusher_r=0.12,
                 noise_seed=cfg["seed"] + 999)
        regs = []
        for j, r in enumerate(REG):
            if world == "pre" and not r["pre"]:
                continue
            if world == "pre":
                nz = r["noise_pre"]
            elif j in FLIP_J:
                nz = flip_amp if world == "post" else r["noise_pre"]
            else:
                nz = r["noise"]
            regs.append(dict(center=tuple(r["center"]), sigma=r["sigma"],
                             phi=float(r["phi"]), noise=float(nz)))
        if regs:
            d["rot_regions"] = regs
        return PusherEnv(d, with_puck=False)

    env0 = make_env("pre", 0.0)      # pre-drift world: B mastered (deterministic), no A/R

    # ---------------- nets ----------------
    def _mlp(seed, din, dout, h, L):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, dout)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    # ---------------- collection ----------------
    def collect_su(cenv, pos, vel, cmd):
        """Step a FIXED (pos, vel, cmd) set in `cenv` -- so two worlds can be stepped on the
        literally identical query set (the paired clean/noisy probes, the cond-mean check)."""
        n = len(pos)
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        for i in range(n):
            cenv.set_state(pos[i].astype(np.float64), vel[i].astype(np.float64))
            s = cenv.get_state(); s2, _ = cenv.step(cmd[i], fs)
            S[i] = s; U[i] = cmd[i]; S2[i] = s2
        return S, U, S2

    def draw_su(pos, rng):
        vel = rng.normal(0, cfg["v_explore"], (len(pos), 2)).astype(np.float32)
        cmd = rng.uniform(-1, 1, (len(pos), 2)).astype(np.float32)
        return vel, cmd

    def collect_at(cenv, pos, rng):
        vel, cmd = draw_su(pos, rng)
        return collect_su(cenv, pos, vel, cmd)

    def box_pos(n, rng):
        return np.stack([rng.uniform(-cfg["box_x"], cfg["box_x"], n),
                         rng.uniform(-cfg["box_y"], cfg["box_y"], n)], 1).astype(np.float32)

    def region_pos(j, n, rng):
        c = REG[j]["center"]; s = REG[j]["sigma"] * cfg["collect_sigma_frac"]
        return np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1).astype(np.float32)

    def collect_box(cenv, n, rng):
        return collect_at(cenv, box_pos(n, rng), rng)

    def gates(pos):
        w = np.empty((len(pos), K), np.float32)
        for j, r in enumerate(REG):
            d2 = (pos[:, 0] - r["center"][0]) ** 2 + (pos[:, 1] - r["center"][1]) ** 2
            w[:, j] = np.exp(-d2 / (2.0 * r["sigma"] ** 2))
        return w

    def classify(S):
        w = gates(S)
        return np.where(w.max(1) > 0.3, w.argmax(1), K)

    CLS_NAMES = [r["name"] for r in REG] + ["base"]

    # ---------------- pools + normalization (identical to the parent) ----------------
    nS, nU, nS2 = collect_box(env0, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([nS, nU], 1).astype(np.float32); Y = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    if cfg["master_extra_n"] > 0:
        mrng = np.random.default_rng(cfg["seed"] + 3)
        for j, r in enumerate(REG):
            if r["pre"]:
                eS, eU, eS2 = collect_at(env0, region_pos(j, cfg["master_extra_n"], mrng), mrng)
                nS = np.concatenate([nS, eS]); nU = np.concatenate([nU, eU])
                nS2 = np.concatenate([nS2, eS2])
                print(f"[pretrain] +{cfg['master_extra_n']} practice samples in {r['name']}",
                      flush=True)
        Y = (nS2 - nS).astype(np.float32)

    huber = nn.HuberLoss(delta=1.0)
    huber_ps = nn.HuberLoss(delta=1.0, reduction="none")

    def train_steps(net, opt, S, U, S2, steps, brng, bs_cap=512):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (Xt - norm["mx"]) / norm["sx"]; Yn = (Yt - norm["my"]) / norm["sy"]
        bs = min(bs_cap, len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fm2_delta(net, S, U):
        with torch.no_grad():
            Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            return (net((Xt - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def fm1_delta(net, S):
        with torch.no_grad():
            Xt = torch.tensor(S, device=device, dtype=torch.float32)
            return (net((Xt - norm["mx"][:4]) / norm["sx"][:4]) * norm["sy"] + norm["my"]).cpu().numpy()

    def per_sample_err(net, S, U, S2):
        return np.linalg.norm(fm2_delta(net, S, U) - (S2 - S), axis=1).astype(np.float32)

    def bench_pred(bnet, S):
        with torch.no_grad():
            Xt = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
            return bnet(Xt).squeeze(-1).cpu().numpy().astype(np.float32)

    def bench_step(bnet, bopt, S, e):
        Xt = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:4]) / norm["sx"][:4]
        t = torch.tensor(e, device=device, dtype=torch.float32)
        bnet.train(); bopt.zero_grad()
        huber(bnet(Xt).squeeze(-1), t).backward(); bopt.step(); bnet.eval()

    # ===================================================================== #
    # 1) THE CONDITIONAL-MEAN CHECK -- the load-bearing assumption of the design
    # ===================================================================== #
    def cond_mean_check(flip_amp):
        """Is E[s'|s,u] in the flip region unchanged by switching the noise on?

        For a fixed set of (pos, vel, cmd) queries: step the CLEAN world once (the noise-off
        Delta, which is the deterministic truth) and the POST world `reps` times (the noisy
        draws).  Report ||mean_reps(Delta_noisy) - Delta_clean|| against the Monte-Carlo
        standard error sd/sqrt(reps).  A ratio near 1 means the two are indistinguishable at
        this sample size, i.e. the noise is mean-preserving.  A clean control region (`base`)
        gives the numerical floor (both worlds are identical there, so the ratio there is
        pure MC noise plus determinism)."""
        rng = np.random.default_rng(cfg["seed"] + 4242)
        e_post = make_env("post", flip_amp); e_clean = make_env("clean", flip_amp)
        out = {}
        targets = [(REG[j]["name"], region_pos(j, cfg["cm_n"], rng)) for j in FLIP_J]
        bp = []
        while sum(len(p) for p in bp) < cfg["cm_n"]:
            cand = box_pos(512, rng)
            bp.append(cand[gates(cand).max(1) < 0.05])
        targets.append(("base", np.concatenate(bp)[:cfg["cm_n"]]))
        for nm, pos in targets:
            vel, cmd = draw_su(pos, rng)
            S0, _, S2c = collect_su(e_clean, pos, vel, cmd)
            d_clean = (S2c - S0).astype(np.float64)
            acc = np.zeros_like(d_clean, dtype=np.float64)
            acc2 = np.zeros_like(d_clean, dtype=np.float64)
            accA = np.zeros_like(d_clean, dtype=np.float64)
            accB = np.zeros_like(d_clean, dtype=np.float64)
            R = cfg["cm_reps"]
            for rr_ in range(R):
                _, _, S2n = collect_su(e_post, pos, vel, cmd)
                dn = (S2n - S0).astype(np.float64)
                acc += dn; acc2 += dn * dn
                if rr_ % 2 == 0:
                    accA += dn
                else:
                    accB += dn
            mu = acc / R
            var = np.maximum(acc2 / (R - 1) - (R / (R - 1.0)) * mu * mu, 0.0)
            dev = mu - d_clean                                          # per-sample deviation
            bias = np.linalg.norm(dev, axis=1)                          # ||E[d] - d_det||
            sd = np.sqrt(var.sum(1))                                    # per-sample noise sd
            se = np.sqrt(var.sum(1) / R)                                # MC standard error
            # chi2/dof: under "the noise is mean-preserving" each standardized deviation is
            # N(0,1), so this is 1.0 at the null and >1 iff a real bias survives the MC floor.
            # Its own sd over n samples is sqrt(2/(4n)), which is the resolution of the test.
            chi2 = (dev ** 2 / np.maximum(var / R, 1e-24)).sum(1) / 4.0
            # pooled signed deviation: catches a SYSTEMATIC (direction-consistent) bias that
            # the per-sample test would leave inside its own noise floor.
            pooled = np.linalg.norm(dev.mean(0))
            pooled_se = float(np.sqrt((var / R).sum(1).mean() / len(pos)))
            # A null with NO reference to the clean world and the identical noise structure:
            # the distance between two independent half-means of the SAME noisy world. Under
            # "the noise is mean-preserving" E[split] = 2*E[bias], so `bias/(split/2)` is 1.0
            # at the null and >1 iff the clean world sits outside the noisy world's own MC
            # scatter. Free of the var-hat degeneracy that inflates chi2 where the noise
            # gate is near zero.
            half = np.linalg.norm(accA / (R - R // 2) - accB / (R // 2), axis=1)
            split_ratio = float((bias / np.maximum(half / 2.0, 1e-12)).mean())
            out[nm] = dict(n=int(len(pos)), reps=int(R),
                           bias=float(bias.mean()), sd=float(sd.mean()),
                           se=float(se.mean()), det=float(np.linalg.norm(d_clean, axis=1).mean()),
                           bias_over_se=float((bias / np.maximum(se, 1e-12)).mean()),
                           bias_over_sd=float((bias / np.maximum(sd, 1e-12)).mean()),
                           chi2_dof=float(chi2.mean()),
                           chi2_dof_res=float(np.sqrt(2.0 / (4.0 * len(pos)))),
                           pooled=float(pooled), pooled_se=pooled_se,
                           split=float(half.mean()), split_ratio=split_ratio)
            print(f"[cond-mean amp={flip_amp:g} {nm}] |d_det|={out[nm]['det']:.5f} "
                  f"bias={out[nm]['bias']:.5f} noise_sd={out[nm]['sd']:.5f} "
                  f"se={out[nm]['se']:.5f} bias/se={out[nm]['bias_over_se']:.2f} "
                  f"chi2/dof={out[nm]['chi2_dof']:.3f}±{out[nm]['chi2_dof_res']:.3f} "
                  f"pooled={pooled:.5f}±{pooled_se:.5f} "
                  f"bias/(split/2)={split_ratio:.3f}", flush=True)
        return out

    # ===================================================================== #
    # 2) STREAMS + PROBES (one per flip amplitude; the pre-drift stack is shared)
    # ===================================================================== #
    _streams = {}

    def build_stream(flip_amp):
        if flip_amp in _streams:
            return _streams[flip_amp]
        print(f"\n===== building stream flip_amp={flip_amp:g} =====", flush=True)
        e_post = make_env("post", flip_amp); e_clean = make_env("clean", flip_amp)
        sS, sU, sS2 = collect_box(e_post, cfg["T"], np.random.default_rng(cfg["seed"] + 500))
        s_cls = classify(sS)
        frac = {CLS_NAMES[c]: float((s_cls == c).mean()) for c in range(K + 1)}
        print(f"[stream amp={flip_amp:g}] T={cfg['T']} class fractions: " +
              " ".join(f"{k}={v:.3f}" for k, v in frac.items()), flush=True)

        prng = np.random.default_rng(cfg["seed"] + 600)
        probes = {"global": collect_box(e_post, cfg["probe_n"], prng)}
        for j in range(K):
            pos = region_pos(j, cfg["probe_n"], prng)
            vel, cmd = draw_su(pos, prng)
            probes[REG[j]["name"]] = collect_su(e_post, pos, vel, cmd)
            if j in FLIP_J:
                # the RETENTION instrument: the identical query set in the clean world, whose
                # Delta s IS the conditional mean (checked by cond_mean_check).
                probes[REG[j]["name"] + "@clean"] = collect_su(e_clean, pos, vel, cmd)
        bs_pos = []
        while sum(len(p) for p in bs_pos) < cfg["probe_n"]:
            cand = box_pos(512, prng)
            bs_pos.append(cand[gates(cand).max(1) < 0.05])
        bs_pos = np.concatenate(bs_pos)[:cfg["probe_n"]]
        probes["base"] = collect_at(e_post, bs_pos, prng)
        st = dict(amp=flip_amp, sS=sS, sU=sU, sS2=sS2, s_cls=s_cls, frac=frac,
                  probes=probes, env_post=e_post, env_clean=e_clean, disag=None)
        _streams[flip_amp] = st
        return st

    def probe_errs(net, stream):
        return {k: float(per_sample_err(net, *p).mean()) for k, p in stream["probes"].items()}

    # calibration probe pool (clean, pre-drift) for tau
    cS, cU, cS2 = collect_box(env0, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 2))

    # ===================================================================== #
    # 3) THE PRE-DRIFT STACK (pretrained FM2/FM1/b(s)/ensemble, calibration)
    #    -- built on env0 only, so it is IDENTICAL across flip amplitudes.
    # ===================================================================== #
    _stacks = {}

    def build_stack(fm_hidden, fm_layers):
        key = (fm_hidden, fm_layers)
        if key in _stacks:
            return _stacks[key]
        print(f"\n===== building stack h={fm_hidden} L={fm_layers} =====", flush=True)
        fm2_0 = _mlp(cfg["seed"] + 40, 6, 4, fm_hidden, fm_layers)
        opt2 = torch.optim.Adam(fm2_0.parameters(), lr=cfg["fm_lr"])
        train_steps(fm2_0, opt2, nS, nU, nS2, cfg["fm_steps"],
                    np.random.default_rng(cfg["seed"] + 300))

        fm1_0 = _mlp(cfg["seed"] + 41, 4, 4, fm_hidden, fm_layers)
        X1 = torch.tensor(nS, device=device, dtype=torch.float32)
        Y1 = torch.tensor(Y, device=device)
        X1n = (X1 - norm["mx"][:4]) / norm["sx"][:4]; Y1n = (Y1 - norm["my"]) / norm["sy"]
        opt1 = torch.optim.Adam(fm1_0.parameters(), lr=cfg["fm_lr"])
        brng1 = np.random.default_rng(cfg["seed"] + 301); fm1_0.train()
        for _ in range(cfg["fm_steps"]):
            idx = torch.tensor(brng1.integers(0, len(nS), size=512), device=device)
            opt1.zero_grad(); huber(fm1_0(X1n[idx]), Y1n[idx]).backward(); opt1.step()
        fm1_0.eval()

        e_pool = per_sample_err(fm2_0, nS, nU, nS2)
        bench_0 = _mlp(cfg["seed"] + 42, 4, 1, cfg["bench_hidden"], cfg["bench_layers"])
        bopt_0 = torch.optim.Adam(bench_0.parameters(), lr=cfg["bench_lr"])
        brngb = np.random.default_rng(cfg["seed"] + 302)
        for _ in range(cfg["bench_pretrain_steps"]):
            idx = brngb.integers(0, len(nS), size=256)
            bench_step(bench_0, bopt_0, nS[idx], e_pool[idx])

        # tau: MAD of (b - e) on a held-out clean probe (parent protocol)
        e_cal = per_sample_err(fm2_0, cS, cU, cS2)
        b_cal = bench_pred(bench_0, cS)
        resid = b_cal - e_cal
        tau = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-4)

        # CENTERED gate, agency_gate protocol, on the TRAINING split
        p1_pool = fm1_delta(fm1_0, nS)
        g_act = np.linalg.norm(fm2_delta(fm2_0, nS, nU) - p1_pool, axis=1)
        g_pas = np.linalg.norm(fm2_delta(fm2_0, nS, np.zeros_like(nU)) - p1_pool, axis=1)
        m_a, m_p = float(np.median(g_act)), float(np.median(g_pas))
        g0 = 0.5 * (m_a + m_p); theta_g = max((m_a - m_p) / 8.0, 1e-9)
        allv = np.concatenate([g_act, g_pas])
        ranks = allv.argsort().argsort().astype(np.float64) + 1.0
        auroc = float((ranks[:len(g_act)].sum() - len(g_act) * (len(g_act) + 1) / 2.0)
                      / (len(g_act) * len(g_pas) + 1e-12))

        def gate_fn(g):
            return (1.0 / (1.0 + np.exp(-(g - g0) / theta_g))).astype(np.float32)

        # ---- the SHADOW random-prior ensemble (curiosity_control's committed form) ----
        beta = cfg["rpf_beta"]

        def build_member(sd):
            net = _mlp(sd, 6, 4, cfg["ens_hidden"], cfg["ens_layers"])
            prior = _mlp(sd + 99991, 6, 4, cfg["ens_hidden"], cfg["ens_layers"])
            for p in prior.parameters():
                p.requires_grad_(False)
            return {"net": net, "prior": prior}

        def mem_fwd(m, Xn):
            return m["net"](Xn) + beta * m["prior"](Xn) if beta > 0 else m["net"](Xn)

        members = [build_member(cfg["seed"] + 60 + 7 * j) for j in range(cfg["ens_k"])]
        Xp = torch.tensor(np.concatenate([nS, nU], 1), device=device, dtype=torch.float32)
        Yp = torch.tensor(Y, device=device)
        Xpn = (Xp - norm["mx"]) / norm["sx"]; Ypn = (Yp - norm["my"]) / norm["sy"]
        for j, m in enumerate(members):
            o = torch.optim.Adam(m["net"].parameters(), lr=cfg["fm_lr"])
            brng = np.random.default_rng(cfg["seed"] + 400 + j)
            m["net"].train()
            for _ in range(cfg["ens_pretrain_steps"]):
                idx = torch.tensor(brng.integers(0, len(nS), size=512), device=device)
                o.zero_grad(); huber(mem_fwd(m, Xpn[idx]), Ypn[idx]).backward(); o.step()
            m["net"].eval()

        def ens_disag(members_, S, U):
            """RMS spread of the members' *denormalized* Delta predictions -- same units as
            `e` (which is the 4-dim norm of the residual)."""
            with torch.no_grad():
                Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
                Xn = (Xt - norm["mx"]) / norm["sx"]
                pr = torch.stack([mem_fwd(m, Xn) * norm["sy"] + norm["my"] for m in members_], 0)
                return pr.var(0).sum(1).sqrt().cpu().numpy().astype(np.float32)

        d_pool = ens_disag(members, nS, nU)
        d_ref = float(d_pool.mean())
        d_med = float(np.median(d_pool))
        # The `expd` consumption law applies delta's OWN calibration recipe to the other
        # signal: exp((d - median)/MAD), so the two gains have comparable dynamic range and
        # a frontier comparison is not secretly a comparison of gain-law scaling. (The
        # linear law w ∝ d is kept as the default -- it is what the drive literature uses.)
        tau_d = max(float(1.4826 * np.median(np.abs(d_pool - d_med))), 1e-9)

        print(f"[calib h={fm_hidden}] fm_err(clean)={e_cal.mean():.4f} tau={tau:.5f} "
              f"gate AUROC={auroc:.4f} g0={g0:.5f} d_ref={d_ref:.5f} "
              f"d_med={d_med:.5f} tau_d={tau_d:.5f}", flush=True)

        st = dict(fm_hidden=fm_hidden, fm_layers=fm_layers, fm2_0=fm2_0, fm1_0=fm1_0,
                  bench_0=bench_0, tau=tau, gate_fn=gate_fn,
                  e_pool_mean=float(e_pool.mean()), members=members, mem_fwd=mem_fwd,
                  ens_disag=ens_disag, d_ref=d_ref, d_med=d_med, tau_d=tau_d,
                  build_member=build_member,
                  calib=dict(tau=tau, g0=g0, theta_g=theta_g, gate_auroc_train=auroc,
                             pretrain_fm_err_clean=float(e_cal.mean()),
                             e_pool_mean=float(e_pool.mean()), d_ref=d_ref,
                             d_med=d_med, tau_d=tau_d,
                             ens_k=cfg["ens_k"], rpf_beta=cfg["rpf_beta"]))
        _stacks[key] = st
        return st

    # ---- per (stack, stream) items: p1 on the stream, the stale probe, tau_online, disag ----
    _pairs = {}

    def pair(stack, stream):
        key = (stack["fm_hidden"], stack["fm_layers"], stream["amp"])
        if key in _pairs:
            return _pairs[key]
        sS, sU = stream["sS"], stream["sU"]
        p1_stream = fm1_delta(stack["fm1_0"], sS)
        stale_probe = probe_errs(stack["fm2_0"], stream)
        e0 = per_sample_err(stack["fm2_0"], sS, sU, stream["sS2"])
        b0 = bench_pred(stack["bench_0"], sS)
        r0 = b0 - e0
        tau_online = max(float(1.4826 * np.median(np.abs(r0 - np.median(r0)))), 1e-4)
        print(f"[stale h={stack['fm_hidden']} amp={stream['amp']:g}] " +
              " ".join(f"{k}={v:.4f}" for k, v in stale_probe.items()), flush=True)
        print(f"[tau h={stack['fm_hidden']} amp={stream['amp']:g}] pretrain={stack['tau']:.5f} "
              f"online={tau_online:.5f} (ratio {tau_online / stack['tau']:.1f}x)", flush=True)

        # ---- the shadow ensemble walk: causal per-batch disagreement over the stream ----
        # Two variants, both precomputed (they are cheap and neither can be influenced by any
        # arm's allocation -- `estimability`'s passive-panel discipline):
        #   `shared` -- every member steps on the SAME batch. This is the form whose claim is
        #               "members agree on the mean where content is random": identical data
        #               pulls every member to the same conditional-mean fit, so aleatoric
        #               content adds no disagreement.
        #   `boot`   -- each member steps on its own bootstrap resample of the batch
        #               (`curiosity_control.update_member`'s committed form). Members then
        #               chase different noise draws, so aleatoric content DOES add
        #               disagreement -- the discrimination the shared form claims is
        #               precisely what bootstrapping gives up.
        beta = cfg["rpf_beta"]; mem_fwd = stack["mem_fwd"]
        B = cfg["batch"]; n_cyc = cfg["T"] // B
        Xs = torch.tensor(np.concatenate([sS, sU], 1), device=device, dtype=torch.float32)
        Ys = torch.tensor((stream["sS2"] - sS).astype(np.float32), device=device)
        Xsn = (Xs - norm["mx"]) / norm["sx"]; Ysn = (Ys - norm["my"]) / norm["sy"]
        d_streams = {}
        for variant in ("shared", "boot"):
            members = [dict(net=copy.deepcopy(m["net"]), prior=m["prior"])
                       for m in stack["members"]]
            opts = [torch.optim.Adam(m["net"].parameters(), lr=cfg["ens_lr"]) for m in members]
            brngs = [np.random.default_rng(cfg["seed"] + 800 + 11 * j)
                     for j in range(len(members))]
            ds = np.zeros(cfg["T"], np.float32)
            for c in range(n_cyc):
                sl = slice(c * B, (c + 1) * B)
                with torch.no_grad():                  # d BEFORE the update -> causal
                    pr = torch.stack([mem_fwd(m, Xsn[sl]) * norm["sy"] + norm["my"]
                                      for m in members], 0)
                    ds[sl] = pr.var(0).sum(1).sqrt().cpu().numpy()
                for m, o, br in zip(members, opts, brngs):
                    if variant == "boot":
                        bi = torch.tensor(c * B + br.integers(0, B, size=B), device=device)
                        xb, yb = Xsn[bi], Ysn[bi]
                    else:
                        xb, yb = Xsn[sl], Ysn[sl]
                    m["net"].train(); o.zero_grad()
                    huber(mem_fwd(m, xb), yb).backward(); o.step(); m["net"].eval()
            d_streams[variant] = ds
        dp = {}
        for variant, ds in d_streams.items():
            dp[variant] = {}
            for cls_i, nm in enumerate(CLS_NAMES):
                msk = stream["s_cls"] == cls_i
                if msk.any():
                    dd = ds[msk]; q = max(len(dd) // 4, 1)
                    dp[variant][nm] = dict(mean=float(dd.mean()),
                                           q1=float(dd[:q].mean()), q2=float(dd[q:2 * q].mean()),
                                           q3=float(dd[2 * q:3 * q].mean()),
                                           q4=float(dd[3 * q:].mean()))
            print(f"[disag/{variant} h={stack['fm_hidden']} amp={stream['amp']:g}] "
                  f"d_ref={stack['d_ref']:.5f}  " +
                  " ".join(f"{k}={v['mean']:.4f}(q1 {v['q1']:.4f}->q4 {v['q4']:.4f})"
                           for k, v in dp[variant].items()), flush=True)
        rec = dict(p1_stream=p1_stream, stale_probe=stale_probe, tau_online=tau_online,
                   d_streams=d_streams, disag_profile=dp)
        _pairs[key] = rec
        return rec

    # ===================================================================== #
    # 4) CONTROL GRADING (reach families; each family names the world it is graded in)
    # ===================================================================== #
    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]; Bev = cfg["n_eval"]

    def mpc_plan(net, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, H, 2), np.float32); sig = np.full((Bn, H, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, H, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, H, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(H):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                    s = s + d; cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    fam_eval = {}
    for fi, (fam, spec_f) in enumerate(cfg["eval_families"].items()):
        y_c, world = spec_f["y"], spec_f["world"]
        ev_rng = np.random.default_rng(cfg["seed"] + 7 + 13 * fi)
        sgn = ev_rng.choice([-1.0, 1.0], Bev).astype(np.float32)
        starts = np.concatenate([
            np.stack([sgn * cr, np.full(Bev, y_c, np.float32)], 1)
            + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32),
            ev_rng.normal(0, cfg["v0_std"], (Bev, 2)).astype(np.float32)], 1)
        goals = np.stack([-sgn * cr, np.full(Bev, y_c, np.float32)], 1) \
            + ev_rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (Bev, 2)).astype(np.float32)
        fam_eval[fam] = (starts, goals, world)

    def rollout(net, replan_every, starts, goals, cenv):
        states = starts.copy(); plan = None
        rng = np.random.default_rng(cfg["seed"] + 7000 + replan_every)
        for step in range(H):
            if step % replan_every == 0:
                plan = mpc_plan(net, states, goals, rng)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for bb in range(Bev):
                cenv.set_state(states[bb, :2].astype(np.float64), states[bb, 2:].astype(np.float64))
                s2, _ = cenv.step(acts[bb], fs); states[bb] = s2
        return float(np.median(np.linalg.norm(states[:, :2] - goals, axis=1)))

    def grade(net, stream):
        rec = {"fm": probe_errs(net, stream)}
        for fam, (starts, goals, world) in fam_eval.items():
            cenv = stream["env_post"] if world == "post" else stream["env_clean"]
            if "reactive" in cfg["controllers"]:
                rec[f"{fam}/reactive"] = rollout(net, 1, starts, goals, cenv)
            if "ballistic_cem" in cfg["controllers"]:
                rec[f"{fam}/ballistic_cem"] = rollout(net, H, starts, goals, cenv)
        for c in cfg["controllers"]:
            rec[f"agg/{c}"] = float(np.mean([rec[f"{fam}/{c}"] for fam in fam_eval]))
        return rec

    def fmt_grade(rec):
        return "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "fm")

    outdir = os.path.join(DATA_DIR, "aleatoric_flip", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    result = {"config": cfg, "complete": False, "streams": {}, "stacks": {}, "cells": {},
              "arms": {}, "cond_mean": {}}

    def checkpoint():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ===================================================================== #
    # 5) THE ARM RUNNER -- weights factorized into (relative allocation) x (spend)
    # ===================================================================== #
    B = cfg["batch"]; n_cycles = cfg["T"] // B
    milestone_set = set(cfg["milestones"])
    w_clip = cfg["w_clip"]; ew_a = cfg["ewma_alpha"]
    LOGQ = ("w", "e", "b", "delta", "g", "gate", "d")

    def run_arm(spec, stack, stream, pr):
        kind = spec["kind"]; base_lr = spec["lr"]
        tau = pr["tau_online"] if spec["tau_mode"] == "online" else stack["tau"]
        gate_fn = stack["gate_fn"]; p1_stream = pr["p1_stream"]
        d_stream = pr["d_streams"][spec["ens_mode"]]
        d_med = stack["d_med"]; tau_d = stack["tau_d"]; logcap = np.log(cfg["w_raw_cap"])
        d_ref = stack["d_ref"]
        sS, sU, sS2, s_cls = stream["sS"], stream["sU"], stream["sS2"], stream["s_cls"]
        fm = copy.deepcopy(stack["fm2_0"])
        opt = torch.optim.Adam(fm.parameters(), lr=base_lr)
        bnet = copy.deepcopy(stack["bench_0"])
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        rng = np.random.default_rng(cfg["seed"] + 700)      # same replay draws per arm
        # w_norm: parent's initialization; `static` freezes it, `ewma` lets it track.
        if kind in ("delta", "fixed", "omask"):
            w_norm = 1.0
        elif kind == "raw_err":
            w_norm = float(stack["e_pool_mean"])
        elif spec["disag_law"] == "exp":                     # disag/conj, exponential law
            w_norm = 1.0
        else:                                               # disag, conj, linear law
            w_norm = float(d_ref)
        snaps = {}
        trace = {"t": [], **{f"probe_{k}": [] for k in stream["probes"]}}
        blog = {"t": [], "spend": [], "clipfrac": []}
        for nm in CLS_NAMES:
            for q in LOGQ:
                blog[f"{q}_{nm}"] = []
        cum_w = np.zeros(K + 1, np.float64); cum_n = np.zeros(K + 1, np.float64)
        w_sum = 0.0; w_cnt = 0; spend_sum = 0.0; spend_cnt = 0; clip_sum = 0.0

        def f_d(d):
            """The disagreement gain, in the arm's chosen law."""
            if spec["disag_law"] == "exp":
                return np.exp(np.clip((d - d_med) / tau_d, -30.0, logcap))
            return d

        def weights_for(idx):
            nonlocal w_norm
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            pred = fm2_delta(fm, S, U)
            e = np.linalg.norm(pred - (S2 - S), axis=1).astype(np.float32)
            g = np.linalg.norm(pred - p1_stream[idx], axis=1).astype(np.float32)  # live g
            gt = gate_fn(g)
            d = d_stream[idx]
            z = np.zeros_like(e)
            if kind == "fixed":
                w_raw = np.full(len(idx), spec["w_mult"], np.float32); b = z; dlt = z
            elif kind == "omask":
                # ORACLE protection: the plasticity a perfect attenuation-on-success rule
                # would deliver if it knew exactly which region had gone aleatoric. Every
                # sample gets w=1 except those classified into a FLIP region, which get
                # `mask_w` (0 = total protection, 1 = uniform, >1 = deliberate over-writing).
                # It is oracle information by construction and is here to measure the VALUE
                # of protection independently of any signal's ability to detect it -- the
                # same role `estimability`'s `oracle_delta` plays for the benchmark.
                w_raw = np.where(np.isin(s_cls[idx], FLIP_J),
                                 np.float32(spec["mask_w"]), np.float32(1.0))
                b = z; dlt = z
            elif kind == "raw_err":
                w_raw = (e * spec["w_mult"]).astype(np.float32); b = z; dlt = z
            elif kind == "disag":
                w_raw = (spec["w_mult"] * f_d(d)).astype(np.float32); b = z; dlt = z
            else:                                            # delta | conj
                b = bench_pred(bnet, S)
                dlt = ((b - e) * gt).astype(np.float32)
                f_dlt = np.exp(np.clip(-dlt / tau, -30.0, logcap))
                w_raw = (spec["w_mult"] * f_dlt).astype(np.float32)
                if kind == "conj":
                    w_raw = (w_raw * f_d(d)).astype(np.float32)
            if spec["norm_mode"] == "ewma" and kind != "fixed":
                w_norm = (1 - ew_a) * w_norm + ew_a * float(w_raw.mean())
            w_hat = np.clip(w_raw / max(w_norm, 1e-8), 0.0, w_clip).astype(np.float32)
            return w_hat, e, b, dlt, g, gt, d

        def gated_step(idx):
            nonlocal w_sum, w_cnt, spend_sum, spend_cnt, clip_sum
            w_hat, e, b, dlt, g, gt, d = weights_for(idx)
            S, U, S2 = sS[idx], sU[idx], sS2[idx]
            spend = float(w_hat.mean())
            if spec["spend_mode"] == "raw_adam":
                w_use = w_hat; lr_now = base_lr           # the parent's exact pipeline
            elif spec["spend_mode"] == "unit":
                w_use = w_hat / max(spend, 1e-8); lr_now = base_lr
            else:                                          # "free"
                w_use = w_hat / max(spend, 1e-8); lr_now = base_lr * spend
            for gp in opt.param_groups:
                gp["lr"] = lr_now
            Xb = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            Yb = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xb - norm["mx"]) / norm["sx"]; Yn = (Yb - norm["my"]) / norm["sy"]
            wt = torch.tensor(w_use, device=device)
            fm.train(); opt.zero_grad()
            (huber_ps(fm(Xn), Yn).mean(1) * wt).mean().backward()
            opt.step(); fm.eval()
            if kind in ("delta", "conj"):
                bench_step(bnet, bopt, S, e)                # b(s) tracks the own-error field
            w_sum += float(w_hat.sum()); w_cnt += len(idx)
            spend_sum += spend; spend_cnt += 1
            clip_sum += float((w_hat >= w_clip - 1e-6).mean())
            return w_hat, e, b, dlt, g, gt, d, spend

        if cfg["grade_control"] and 0 in milestone_set:
            snaps[0] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
        for c in range(n_cycles):
            inc = np.arange(c * B, (c + 1) * B)
            w_hat, e, b, dlt, g, gt, d, spend = gated_step(inc)
            cls = s_cls[inc]
            blog["t"].append((c + 1) * B)
            blog["spend"].append(spend)
            blog["clipfrac"].append(float((w_hat >= w_clip - 1e-6).mean()))
            for ci, nm in enumerate(CLS_NAMES):
                m = cls == ci
                for q, arr in zip(LOGQ, (w_hat, e, b, dlt, g, gt, d)):
                    blog[f"{q}_{nm}"].append(float(np.nanmean(arr[m])) if m.any() else np.nan)
                cum_w[ci] += float(w_hat[m].sum()); cum_n[ci] += int(m.sum())
            for _ in range(spec["n_replay"]):
                ridx = rng.integers(0, (c + 1) * B, size=cfg["replay_batch"])
                gated_step(ridx)
            t_now = (c + 1) * B
            if c % cfg["probe_every"] == 0 or t_now in milestone_set:
                pe = probe_errs(fm, stream)
                trace["t"].append(t_now)
                for k, v in pe.items():
                    trace[f"probe_{k}"].append(v)
            if t_now in milestone_set:
                if cfg["grade_control"]:
                    snaps[t_now] = {k: v.cpu().clone() for k, v in fm.state_dict().items()}
                print(f"[{spec['name']} m={t_now:5d}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in pe.items()), flush=True)
        budget = {"mean_w": w_sum / max(w_cnt, 1),
                  "mean_spend": spend_sum / max(spend_cnt, 1),
                  "clip_frac": clip_sum / max(spend_cnt, 1),
                  "tau_used": float(tau), "ens_mode": spec["ens_mode"],
                  "disag_law": spec["disag_law"], "mask_w": spec["mask_w"],
                  "eff_lr": base_lr * (spend_sum / max(spend_cnt, 1))
                  if spec["spend_mode"] == "free" else base_lr,
                  "cum_w_share": {CLS_NAMES[ci]: float(cum_w[ci] / max(cum_w.sum(), 1e-9))
                                  for ci in range(K + 1)},
                  "sample_share": {CLS_NAMES[ci]: float(cum_n[ci] / max(cum_n.sum(), 1e-9))
                                   for ci in range(K + 1)}}
        print(f"[{spec['name']}] mean_w={budget['mean_w']:.3f} spend={budget['mean_spend']:.3f} "
              f"eff_lr={budget['eff_lr']:.2e} clip={budget['clip_frac']:.3f}  shares: " +
              " ".join(f"{k}={v:.3f}" for k, v in budget["cum_w_share"].items()), flush=True)
        return snaps, trace, blog, budget

    # ===================================================================== #
    # 6) DRIVER
    # ===================================================================== #
    if cfg["cond_mean_check"]:
        for amp in sorted({s["flip_amp"] for s in cfg["arm_specs"] if s["flip_amp"] > 0}):
            result["cond_mean"][f"{amp:g}"] = cond_mean_check(amp)
        checkpoint()

    for spec in cfg["arm_specs"]:
        stack = build_stack(spec["fm_hidden"], spec["fm_layers"])
        stream = build_stream(spec["flip_amp"])
        pr = pair(stack, stream)
        skey = f"h{spec['fm_hidden']}x{spec['fm_layers']}"
        amk = f"{spec['flip_amp']:g}"
        ckey = f"{skey}|f{amk}"
        if skey not in result["stacks"]:
            result["stacks"][skey] = dict(stack["calib"])
        if amk not in result["streams"]:
            result["streams"][amk] = dict(class_fractions=stream["frac"])
        if ckey not in result["cells"]:
            cell = dict(stale_probe=pr["stale_probe"], tau_online=pr["tau_online"],
                        disag_profile=pr["disag_profile"])
            if cfg["grade_control"]:
                g_stale = grade(stack["fm2_0"], stream); g_stale["transitions"] = 0
                print(f"[grade stale   {ckey}] " + fmt_grade(g_stale), flush=True)
                fm_ceil = _mlp(cfg["seed"] + 43, 6, 4, spec["fm_hidden"], spec["fm_layers"])
                dS, dU, dS2 = collect_box(stream["env_post"], cfg["pool_n"],
                                          np.random.default_rng(cfg["seed"] + 20))
                optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
                train_steps(fm_ceil, optc, dS, dU, dS2, cfg["fm_steps"],
                            np.random.default_rng(cfg["seed"] + 303))
                g_ceil = grade(fm_ceil, stream); g_ceil["transitions"] = -1
                print(f"[grade ceiling {ckey}] " + fmt_grade(g_ceil), flush=True)
                cell["stale"] = g_stale; cell["ceiling"] = g_ceil
            result["cells"][ckey] = cell
            checkpoint()
        print(f"\n===== arm: {spec['name']} (cell {ckey}) =====", flush=True)
        snaps, trace, blog, budget = run_arm(spec, stack, stream, pr)
        result["arms"][spec["name"]] = {"spec": spec, "stack": skey, "cell": ckey,
                                        "trace": trace, "blog": blog, "budget": budget}
        checkpoint()
        if cfg["grade_control"]:
            lad_net = _mlp(0, 6, 4, spec["fm_hidden"], spec["fm_layers"])
            lad = []
            for m in sorted(snaps):
                if m == 0:
                    lad.append(dict(result["cells"][ckey]["stale"])); continue
                lad_net.load_state_dict(snaps[m]); lad_net.to(device).eval()
                rec = grade(lad_net, stream); rec["transitions"] = int(m)
                lad.append(rec)
                print(f"[grade {spec['name']:24s} m={m:5d}] " + fmt_grade(rec), flush=True)
                result["arms"][spec["name"]]["ladder"] = lad
                checkpoint()
            result["arms"][spec["name"]]["ladder"] = lad
            checkpoint()
        del snaps

    result["complete"] = True
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("complete\n")
    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": result}


@app.local_entrypoint()
def flip(
    quick: bool = False,
    mode: str = "main",
    tag: str = "",
    seed: int = 0,
    # ---- arms.  token: kind[@lr][:spend][:norm][:rN][:xM][:fA][:tauon]
    arms: str = "",
    lr_grid: str = "",           # main mode: fixed-arm lr grid (traces the frontier)
    controllers: str = "reactive,ballistic_cem",
    # ---- calibrate mode grid (the damage-channel ladder)
    cal_amps: str = "0,3,10,30",
    cal_lr: str = "3e-4,1e-3,3e-3",
    cal_extra: str = ("delta@3e-4:f0,delta@3e-4:f30,delta@3e-4:f30:tauon,"
                      "disag@3e-4:f0,disag@3e-4:f30,disag@3e-4:f30:boot,"
                      "conj@3e-4:f30,raw_err@3e-4:f30"),
    # regions: cx,cy,phi,noise_post,pre,name[,noise_pre]   (noise_pre defaults to noise_post,
    # which reproduces the parent's spec exactly).  B-mastered is the FLIP region: it carries
    # phi=-1.2 in BOTH worlds and gains its noise only at t=0.
    regions: str = ("0.0,0.30,1.2,0.0,0,A-drift; "
                    "0.0,-0.30,-1.2,30.0,1,B-flip,0.0; "
                    "0.0,0.90,0.0,30.0,0,R-noise"),
    region_sigma: float = 0.18,
    flip_amp: float = 30.0,       # default post-flip amplitude for the flip region(s)
    # eval families: name:y[:world]   world in {post, clean} (default post)
    eval_families: str = "drift:0.30:post; mastered:-0.30:clean; mastered_noisy:-0.30:post",
    # ---- the parent's priced cell (defaults = PRICED)
    fm_hidden: int = 32,
    fm_layers: int = 2,
    n_replay: int = 0,
    norm_mode: str = "static",
    spend_mode: str = "free",
    # ---- the shadow disagreement ensemble
    ens_k: int = 4,
    ens_hidden: int = 32,
    ens_layers: int = 2,
    ens_lr: float = 3e-4,
    ens_pretrain_steps: int = 3000,
    rpf_beta: float = 0.6,
    # ---- the conditional-mean check
    cond_mean_check: bool = True,
    cm_n: int = 400,
    cm_reps: int = 128,
    # online adaptation
    total_t: int = 16384,
    batch: int = 16,
    replay_batch: int = 64,
    adapt_lr: float = 3e-4,
    milestones: str = "0,512,2048,4096,8192,16384",
    probe_every: int = 16,
    # gain law
    w_clip: float = 4.0,
    w_raw_cap: float = 20.0,
    ewma_alpha: float = 0.05,
    # benchmark net b(s)
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 3e-3,
    bench_pretrain_steps: int = 1500,
    # env (the 4c/plasticity_gain regime)
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    corridor_r: float = 0.4,
    box_x: float = 0.9,
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,
    pool_n: int = 9000,
    master_extra_n: int = 3000,
    probe_n: int = 500,
    v_explore: float = 1.2,
    fm_lr: float = 1e-3,
    fm_steps: int = 4000,
    # control eval
    n_eval: int = 40,
    goal_jit: float = 0.06,
    v0_std: float = 0.3,
    plan_h: int = 34,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    ctrl_list = [c for c in controllers.split(",") if c]
    REG = []
    for chunk in [c for c in regions.split(";") if c.strip()]:
        q = [p.strip() for p in chunk.split(",")]
        nz = float(q[3])
        REG.append(dict(center=(float(q[0]), float(q[1])), sigma=region_sigma,
                        phi=float(q[2]), noise=nz, pre=bool(int(q[4])), name=q[5],
                        noise_pre=(float(q[6]) if len(q) > 6 else nz)))
    fams = {}
    for chunk in [c for c in eval_families.split(";") if c.strip()]:
        p = [x.strip() for x in chunk.split(":")]
        fams[p[0]] = dict(y=float(p[1]), world=(p[2] if len(p) > 2 else "post"))
    assert all(f["world"] in ("post", "clean") for f in fams.values())
    ms = [int(x) for x in milestones.split(",") if x]

    if quick:
        total_t = 2048; ms = [0, 512, 2048]; pool_n = 2500; probe_n = 250
        fm_steps = 1200; bench_pretrain_steps = 500; ens_pretrain_steps = 800
        n_eval = 12; k_shoot = 96; probe_every = 8; replay_batch = 32
        master_extra_n = 800; cm_n = 60; cm_reps = 24
        cal_amps = "0,30"; cal_lr = "3e-4,3e-3"
        cal_extra = "delta@3e-4:f30,disag@3e-4:f30"
        lr_grid = lr_grid or "3e-4,3e-3"
        arms = arms or ("fixed@3e-4,fixed@3e-3,delta@3e-4,delta@3e-4:tauon,"
                        "raw_err@3e-4,disag@3e-4,conj@3e-4,fixed@3e-4:f0,delta@3e-4:f0")
        tag = tag or ("smoke_" + mode)
    tag = tag or ("afcal" if mode == "calibrate" else "af")

    grade_control = (mode == "main")
    tokens = []
    if arms:
        tokens = [a for a in arms.split(",") if a]
    elif mode == "calibrate":
        for a in [float(x) for x in cal_amps.split(",") if x]:
            for lr in [float(x) for x in cal_lr.split(",") if x]:
                tokens.append(f"fixed@{lr:g}:f{a:g}")
        tokens += [t for t in cal_extra.split(",") if t]
    else:
        grid = [float(x) for x in (lr_grid or "1e-4,3e-4,1e-3,3e-3,1e-2").split(",") if x]
        tokens = [f"fixed@{lr:g}" for lr in grid]
        tokens += [f"delta@{adapt_lr:g}", f"delta@{adapt_lr:g}:unit",
                   f"delta@{adapt_lr:g}:tauon", f"raw_err@{adapt_lr:g}",
                   f"disag@{adapt_lr:g}", f"disag@{adapt_lr:g}:expd",
                   f"disag@{adapt_lr:g}:expd:unit", f"conj@{adapt_lr:g}:expd"]

    specs = []
    for tk in tokens:
        # `:hN` / `:LN` capacity qualifiers are handled here; the rest in parse_arm
        h, L = fm_hidden, fm_layers
        keep = []
        for p in tk.split(":"):
            if p.startswith("h") and p[1:].isdigit():
                h = int(p[1:])
            elif p.startswith("L") and p[1:].isdigit():
                L = int(p[1:])
            else:
                keep.append(p)
        sp = parse_arm(":".join(keep), adapt_lr, spend_mode, norm_mode, n_replay, flip_amp)
        sp["name"] = tk
        sp["fm_hidden"] = h; sp["fm_layers"] = L
        specs.append(sp)
    # group by (capacity, flip amplitude) so each stack/stream is built once
    specs.sort(key=lambda s: (s["fm_hidden"], s["fm_layers"], s["flip_amp"]))

    assert all(m % batch == 0 for m in ms), "milestones must be multiples of batch"
    assert total_t == max(ms), "total_t should equal the last milestone"
    for s in specs:
        assert not (s["n_replay"] > 0 and s["kind"] in ("disag", "conj")), (
            f"{s['name']}: the precomputed causal disagreement stream has no replay-index "
            "semantics (d is logged per online batch, not per replayed sample)")

    cfg = dict(
        tag=tag, seed=seed, mode=mode, arm_specs=specs, controllers=ctrl_list,
        regions=REG, eval_families=fams, grade_control=grade_control,
        T=total_t, batch=batch, replay_batch=replay_batch,
        adapt_lr=adapt_lr, milestones=ms, probe_every=probe_every,
        w_clip=w_clip, w_raw_cap=w_raw_cap, ewma_alpha=ewma_alpha,
        norm_mode=norm_mode, spend_mode=spend_mode, flip_amp=flip_amp,
        fm_hidden=fm_hidden, fm_layers=fm_layers, n_replay=n_replay,
        ens_k=ens_k, ens_hidden=ens_hidden, ens_layers=ens_layers, ens_lr=ens_lr,
        ens_pretrain_steps=ens_pretrain_steps, rpf_beta=rpf_beta,
        cond_mean_check=cond_mean_check, cm_n=cm_n, cm_reps=cm_reps,
        bench_hidden=bench_hidden, bench_layers=bench_layers,
        bench_lr=bench_lr, bench_pretrain_steps=bench_pretrain_steps,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y,
        collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, master_extra_n=master_extra_n, probe_n=probe_n, v_explore=v_explore,
        fm_lr=fm_lr, fm_steps=fm_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_flip.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, f"{tag}.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}/{tag}.json")
