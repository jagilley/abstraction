"""The compounding re-adaptation loop — the meta-layer signature under perpetual drift.

Program: `ideas/two_timescale_value_loop.md` + `ONLINE_VALUE_LOOP_README.md` §"Next steps —
the experiment the whole thing points at". The first fully-online loop found value is a
SLOW/COMMITTED quantity (fast-online discovery obstructed on both levers) and reframed its
benefit as ADAPTATION SPEED, not converged competence. Under PERPETUAL DRIFT you never
converge, so the pre-convergence regime is permanent — and the payoff to look for is
COMPOUNDING: each new drift cheaper than the last because the invariant core is already paid
for. That compounding-across-drifts is the meta-layer signature the program has never isolated.

    THE EXPERIMENT (Phase 1, this function). A sequence of related Type-2 drifts (the
    `push_rot` actuator-rotation conflict — φ and φ+π opposite command→motion maps, the
    input-coupled conflict that makes POOLING COLLAPSE; NOT noise, which collapses meta to
    multitask). Measure TRANSITIONS-TO-RECOVER per drift for:
      * monolithic-continual  f(s,u)  (the "veridical, models everything" agent) — with
        replay (=pooled, conflict-corrupted) and without (=forgetting). Re-learns each drift.
      * factored / value-carved  f(s,u,z)  (meta_context.py's context latent): a stable
        invariant core f (free-flight+drag, shared across φ) + a thin adaptable z (encodes φ).
        Fast adapt = infer z (a forward pass); slow loop consolidates f across the sequence.
    Two references: OFFLINE-META f (pre-trained on the whole φ distribution → the "already
    paid for" ceiling) and SCRATCH-per-drift (no memory → flat-high floor).

    THE PREDICTION. factored's transitions-to-recover DESCENDS over the sequence (compounding
    as f matures); monolithic's is FLAT/RISING (conflict corrupts pooling — it sees all the
    same data but can't amortize). The gap WIDENS = the meta-layer signature. Non-tautological
    because both agents get identical data; the monolithic just can't compound under conflict.

    TWO TIMESCALES, cleanly split. FAST = amortized inference (z from a few transitions, no
    gradient). SLOW = gradient consolidation of the invariant core E+f over the growing task
    bank. The "value" in Phase 1 is the invariant/adaptive CAPACITY SPLIT itself (the latent
    dim), established slowly — the slow/committed quantity the first run showed value has to be.
    The `--carve-sweep` runs the "nothing more, nothing less" test: sweep latent_dim → an
    optimum where you carve exactly the φ-dependent part (φ is 1-D → expect d≈1-2 best).

CONTROLS (both reuse meta_adapt dials, off the headline):
  * `--family damping` — the FLOOR (no conflict): all arms tie / compound trivially → confirms
    it is the CONFLICT that makes factoring load-bearing, not exposure per se.
  * `--drift-mode revisit` — cycle a fixed φ bank: factored recovers instantly on revisit,
    monolithic-noreplay has forgotten (the crispest memory contrast).

Phase 2 (value-driven carving under a value-IRRELEVANT drift) lives in `run_value_carved_drift`.

Run:
    cd experiments/
    modal run mujoco_control/compounding_drift.py::compounding_drift --quick                 # smoke
    modal run --detach mujoco_control/compounding_drift.py::compounding_drift --tag full_v1  # headline (novel φ, conflict)
    modal run --detach mujoco_control/compounding_drift.py::compounding_drift --tag floor_v1 --family damping
    modal run --detach mujoco_control/compounding_drift.py::compounding_drift --tag revisit_v1 --drift-mode revisit
    modal run --detach mujoco_control/compounding_drift.py::compounding_drift --tag carve_v1 --carve-sweep
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (pusher_env.STATE_LABELS), for Phase 2 (puck-present, 8-dim)
PUSHER_POS = [0, 1]
PUCK_POS = [2, 3]
PUSHER_VEL = [4, 5]
PUCK_VEL = [6, 7]
PUSHER = [0, 1, 4, 5]     # value-relevant: support of V = -||pos-g|| - beta*||vel||
PUCK = [2, 3, 6, 7]       # value-irrelevant: the distractor's own (drifting) motion


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_compounding_drift(cfg: dict) -> dict:
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    VEL_DIMS = [2, 3]                      # puck-free: [px,py,vx,vy]; damping/rotation act on vel
    budgets = cfg["budgets"]              # transitions-to-recover N-grid
    family = cfg["family"]

    # ---- the drift SEQUENCE of task parameters ----------------------------- #
    seq_rng = np.random.default_rng(cfg["seed"] + 11)
    T = cfg["n_drifts"]
    if family == "actuator":
        lo, hi = -cfg["conflict"], cfg["conflict"]
    elif family == "damping":
        lo, hi = np.log10(cfg["damp_min"]), np.log10(cfg["damp_max"])
    else:
        raise ValueError(family)

    def draw_param(rng):
        u = rng.uniform(lo, hi)
        return float(u) if family == "actuator" else float(10.0 ** u)

    if cfg["drift_mode"] == "novel":
        phi_seq = [draw_param(seq_rng) for _ in range(T)]
    elif cfg["drift_mode"] == "revisit":
        bank = [draw_param(seq_rng) for _ in range(cfg["revisit_bank"])]
        phi_seq = [bank[t % len(bank)] for t in range(T)]
    else:
        raise ValueError(cfg["drift_mode"])
    print(f"[seq] family={family} mode={cfg['drift_mode']} T={T}\n[seq] φ_seq="
          + " ".join(f"{p:+.3f}" for p in phi_seq), flush=True)

    def task_dgp(p):
        dd = dict(cfg["dgp_base"])
        if family == "damping":
            dd["joint_damping"] = float(p)
        else:
            dd["joint_damping"] = cfg["base_damping"]
            dd["push_rot"] = float(p)
        return dd

    # ---- reward-free central OU collection (dynamics_shift.collect idiom) --- #
    def collect(env, n_transitions, seed, ou_theta=0.2, ou_sigma=0.5):
        rng = np.random.default_rng(seed)
        ep_len, pr = cfg["collect_ep_len"], cfg["collect_pos_range"]
        S, U, S2, C = [], [], [], []
        for _ in range(int(np.ceil(n_transitions / ep_len))):
            env.set_state(rng.uniform(-pr, pr, size=2), rng.normal(0, cfg["v_explore"], size=2))
            a = np.zeros(2)
            for _ in range(ep_len):
                s = env.get_state()
                a = a - ou_theta * a + ou_sigma * rng.normal(size=2)
                u = np.clip(a, -1.0, 1.0)
                s2, info = env.step(u, fs)
                S.append(s); U.append(u.astype(np.float32)); S2.append(s2)
                C.append(info["any_contact"])
        sl = slice(0, n_transitions)
        return (np.asarray(S, np.float32)[sl], np.asarray(U, np.float32)[sl],
                np.asarray(S2, np.float32)[sl], np.asarray(C, bool)[sl])

    # ---- family normalization (fixed once, shared by ALL arms = controlled) - #
    print("[norm] calibrating family normalization ...", flush=True)
    calib_ps = np.linspace(lo, hi, 4)
    calib_ps = [float(p) if family == "actuator" else float(10.0 ** p) for p in calib_ps]
    cS, cU, cS2 = [], [], []
    for i, p in enumerate(calib_ps):
        s, u, s2, _ = collect(PusherEnv(task_dgp(p), with_puck=False), cfg["norm_N"], cfg["seed"] + 700 + i)
        cS.append(s); cU.append(u); cS2.append(s2)
    cS, cU, cS2 = np.concatenate(cS), np.concatenate(cU), np.concatenate(cS2)
    X = np.concatenate([cS, cU], 1).astype(np.float32); Y = (cS2 - cS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
    mx, sx, my, sy = norm["mx"], norm["sx"], norm["my"], norm["sy"]

    def feats(S, U, S2):
        """normalized (su_n [.,6], dy_n [.,4], ctx_n [.,10]) on device."""
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (Xt - mx) / sx; dy_n = (Yt - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    def r2_vel(predn, S, S2, C):
        pred = (predn.detach() * sy + my).cpu().numpy()
        yt = (S2 - S).astype(np.float32)
        if C is not None:
            m = ~C; pred, yt = pred[m], yt[m]
        pred, yt = pred[:, VEL_DIMS], yt[:, VEL_DIMS]
        ssr = ((yt - pred) ** 2).sum(); sst = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    # ---- architectures ----------------------------------------------------- #
    eh, fh, fl, d = cfg["enc_hidden"], cfg["fm_hidden"], cfg["fm_layers"], cfg["latent_dim"]

    class Encoder(nn.Module):
        def __init__(self, dz):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(10, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, dz))

        def forward(self, ctx):               # ctx: (B,K,10) -> z: (B,dz)
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm(dz):
        lyr = [nn.Linear(6 + dz, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(fh, 4)])).to(device)

    def build_mono():
        lyr = [nn.Linear(6, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(fh, 4)])).to(device)

    lossf = nn.HuberLoss(delta=1.0)

    # ---- family oracle ceiling (rotation doesn't change achievable R^2) ----- #
    def train_plain(S, U, S2, epochs, lr, init=None):
        fm = build_mono()
        if init is not None:
            fm.load_state_dict(init)
        su_n, dy_n, _ = feats(S, U, S2)
        o = torch.optim.Adam(fm.parameters(), lr=lr)
        n = su_n.shape[0]; bs = min(512, n); fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                o.zero_grad(); lossf(fm(su_n[idx]), dy_n[idx]).backward(); o.step()
        fm.eval(); return fm

    print("[oracle] family oracle ceiling ...", flush=True)
    orc_r2s = []
    for i, p in enumerate([calib_ps[1], calib_ps[2]]):
        s, u, s2, c = collect(PusherEnv(task_dgp(p), with_puck=False), cfg["oracle_N"], cfg["seed"] + 800 + i)
        fm_o = train_plain(s[:-1500], u[:-1500], s2[:-1500], cfg["fm_epochs"], cfg["fm_lr"])
        orc_r2s.append(r2_vel(fm_o(feats(s[-1500:], u[-1500:], s2[-1500:])[0]), s[-1500:], s2[-1500:], c[-1500:]))
    oracle = float(np.mean(orc_r2s))
    thr = cfg["adapt_frac"] * oracle
    print(f"[oracle] family vel R^2={oracle:.4f} -> recover threshold={thr:.4f}", flush=True)

    # ===================================================================== #
    # OFFLINE-META reference: pre-train E+f on the whole φ distribution
    # (disjoint from the sequence draws) — the "already paid for" ceiling.
    # ===================================================================== #
    def meta_train(enc, cfm, opt, banks, steps, rng):
        """banks: list of (su_n, dy_n, ctx_n) feature tuples (one per task). One
        amortized-inference meta step = sample tasks, K context + P pred, Huber."""
        ctx_Ns = [b for b in budgets if b > 0]
        P = cfg["pred_P"]; Bt = min(cfg["task_batch"], len(banks))
        enc.train(); cfm.train()
        dz = enc.rho[-1].out_features
        for _ in range(steps):
            K = int(rng.choice(ctx_Ns))
            tids = rng.choice(len(banks), size=Bt, replace=False)
            ctx = torch.empty(Bt, K, 10, device=device)
            psu = torch.empty(Bt, P, 6, device=device)
            pdy = torch.empty(Bt, P, 4, device=device)
            for j, tid in enumerate(tids):
                su_n, dy_n, cn = banks[tid]; n = su_n.shape[0]
                ci = torch.tensor(rng.integers(0, n, size=K), device=device)
                pi = torch.tensor(rng.integers(0, n, size=P), device=device)
                ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
            z = enc(ctx)
            pred = cfm(torch.cat([psu, z.unsqueeze(1).expand(-1, P, -1)], 2))
            loss = lossf(pred, pdy)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
            opt.step()
        enc.eval(); cfm.eval()

    print("\n[offline-meta] pre-training the ceiling reference ...", flush=True)
    om_ps = [draw_param(np.random.default_rng(cfg["seed"] + 999 + i)) for i in range(cfg["n_offline_tasks"])]
    om_banks = []
    for i, p in enumerate(om_ps):
        s, u, s2, _ = collect(PusherEnv(task_dgp(p), with_puck=False), cfg["drift_pool_N"], cfg["seed"] + 3000 + i)
        om_banks.append(feats(s, u, s2))
    om_enc, om_cfm = Encoder(d).to(device), build_cfm(d)
    om_opt = torch.optim.Adam(list(om_enc.parameters()) + list(om_cfm.parameters()), lr=cfg["meta_lr"])
    meta_train(om_enc, om_cfm, om_opt, om_banks, cfg["offline_meta_steps"], np.random.default_rng(cfg["seed"] + 5))

    def infer_z(enc, cn, N):
        if N == 0:
            return torch.zeros(enc.rho[-1].out_features, device=device)
        with torch.no_grad():
            return enc(cn[:N].unsqueeze(0))[0]

    # ===================================================================== #
    # persistent per-arm state (the CONTINUAL agents)
    # ===================================================================== #
    arms = cfg["arms"]
    state = {}
    if "factored" in arms:
        enc, cfm = Encoder(d).to(device), build_cfm(d)
        state["factored"] = dict(enc=enc, cfm=cfm,
                                 opt=torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()),
                                                      lr=cfg["meta_lr"]),
                                 banks=[])
    for mono_arm in ("mono_replay", "mono_noreplay"):
        if mono_arm in arms:
            fm = build_mono()
            state[mono_arm] = dict(fm=fm, opt=torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"]),
                                   replay=[])
    rng_slow = np.random.default_rng(cfg["seed"] + 17)

    # per-arm, per-drift transitions-to-recover + the held-out R^2 curve
    ttr = {a: [] for a in arms}
    curves = {a: [] for a in arms}        # list over drifts of {N: R2}

    def recover_N(r2_by_budget):
        for N in budgets:
            if N > 0 and r2_by_budget[N] >= thr:
                return N
        return None    # never recovered within the budget grid

    # ===================================================================== #
    # THE DRIFT LOOP
    # ===================================================================== #
    for t, phi in enumerate(phi_seq):
        env = PusherEnv(task_dgp(phi), with_puck=False)
        S, U, S2, C = collect(env, cfg["drift_pool_N"], cfg["seed"] + 4000 + t)
        nctx = max(budgets)
        # context/adapt pool (first nctx) + held-out eval (rest)
        cS, cU, cS2, cC = S[:nctx], U[:nctx], S2[:nctx], C[:nctx]
        hS, hU, hS2, hC = S[nctx:], U[nctx:], S2[nctx:], C[nctx:]
        _, _, cn_ctx = feats(cS, cU, cS2)     # context features for z-inference
        hsu, _, _ = feats(hS, hU, hS2)         # held-out inputs
        banks_feat = feats(S, U, S2)           # full-drift features (slow consolidation)

        line = [f"drift {t:2d} φ={phi:+.3f}"]
        for a in arms:
            r2b = {}
            if a == "scratch":
                for N in budgets:
                    if N == 0:
                        r2b[N] = r2_vel(build_mono()(hsu), hS, hS2, hC)
                    else:
                        fm = train_plain(cS[:N], cU[:N], cS2[:N], cfg["adapt_epochs"], cfg["fm_lr"])
                        r2b[N] = r2_vel(fm(hsu), hS, hS2, hC)
            elif a == "offline_meta":
                for N in budgets:
                    z = infer_z(om_enc, cn_ctx, N)
                    with torch.no_grad():
                        r2b[N] = r2_vel(om_cfm(torch.cat([hsu, z.expand(hsu.shape[0], -1)], 1)), hS, hS2, hC)
            elif a == "factored":
                st = state[a]
                for N in budgets:
                    z = infer_z(st["enc"], cn_ctx, N)
                    with torch.no_grad():
                        r2b[N] = r2_vel(st["cfm"](torch.cat([hsu, z.expand(hsu.shape[0], -1)], 1)), hS, hS2, hC)
            else:  # mono_replay / mono_noreplay — fine-tune a COPY of current weights on first N
                st = state[a]
                cur = copy.deepcopy(st["fm"].state_dict())
                for N in budgets:
                    if N == 0:
                        with torch.no_grad():
                            st["fm"].load_state_dict(cur)
                            r2b[N] = r2_vel(st["fm"](hsu), hS, hS2, hC)
                    else:
                        fm_a = train_plain(cS[:N], cU[:N], cS2[:N], cfg["adapt_epochs"], cfg["fm_lr"], init=cur)
                        r2b[N] = r2_vel(fm_a(hsu), hS, hS2, hC)
            curves[a].append(r2b)
            rN = recover_N(r2b)
            ttr[a].append(rN)
            line.append(f"{a}:recover@{rN if rN is not None else '>'+str(max(budgets))}")

        # ---- SLOW consolidation (the invariant-core carving) -------------- #
        if "factored" in arms:
            st = state["factored"]
            st["banks"].append(banks_feat)
            if len(st["banks"]) >= 2:          # needs >=2 tasks to learn conditioning
                meta_train(st["enc"], st["cfm"], st["opt"], st["banks"], cfg["slow_steps"], rng_slow)
        for mono_arm in ("mono_replay", "mono_noreplay"):
            if mono_arm not in arms:
                continue
            st = state[mono_arm]
            if mono_arm == "mono_replay":
                st["replay"].append((S, U, S2))
                pool = st["replay"]
            else:
                pool = [(S, U, S2)]
            fm, opt = st["fm"], st["opt"]; fm.train()
            for _ in range(cfg["slow_steps"]):
                bi = rng_slow.integers(0, len(pool))
                bS, bU, bS2 = pool[bi]
                su_n, dy_n, _ = feats(bS, bU, bS2)
                idx = torch.tensor(rng_slow.integers(0, su_n.shape[0], size=min(512, su_n.shape[0])), device=device)
                opt.zero_grad(); lossf(fm(su_n[idx]), dy_n[idx]).backward(); opt.step()
            fm.eval()
        print("  ".join(line), flush=True)

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    big = max(budgets)
    ttr_num = {a: [(rN if rN is not None else big * cfg["censor_mult"]) for rN in ttr[a]] for a in arms}
    cum = {a: np.cumsum(ttr_num[a]).tolist() for a in arms}
    results = {
        "config": cfg, "family": family, "drift_mode": cfg["drift_mode"],
        "phi_seq": phi_seq, "budgets": budgets, "oracle_vel_r2": oracle, "recover_threshold": thr,
        "ttr": {a: ttr[a] for a in arms},          # raw (None = censored)
        "ttr_numeric": ttr_num,                     # censored -> big*mult for plotting/cumsum
        "cumulative": cum,
        "curves": {a: [{str(k): v for k, v in c.items()} for c in curves[a]] for a in arms},
    }
    print("\n===== COMPOUNDING SUMMARY =====", flush=True)
    for a in arms:
        early = np.mean([x for x in ttr_num[a][:max(1, T // 4)]])
        late = np.mean([x for x in ttr_num[a][-max(1, T // 4):]])
        print(f"  {a:14s} recover@ early={early:6.1f}  late={late:6.1f}  "
              f"Δ={late-early:+6.1f}  cum={cum[a][-1]:.0f}", flush=True)

    figures = _make_compounding_figures(results)
    outdir = os.path.join(DATA_DIR, "compounding_drift", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_compounding_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"factored": "#3d6fd1", "mono_replay": "#d1603d", "mono_noreplay": "#e08a3c",
           "offline_meta": "#2f9e44", "scratch": "#888"}
    LAB = {"factored": "factored f(s,u,z) (value-carved)", "mono_replay": "monolithic + replay (pooled)",
           "mono_noreplay": "monolithic, no replay (forgetting)",
           "offline_meta": "offline-meta ceiling", "scratch": "scratch/drift (floor)"}
    figs = {}
    arms = list(R["ttr_numeric"].keys())
    T = len(R["phi_seq"])
    x = np.arange(T)
    big = max(R["budgets"])

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: transitions-to-recover per drift (the compounding curve) ----
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for a in arms:
        y = R["ttr_numeric"][a]
        ax.plot(x, y, "o-", color=COL.get(a, "#555"), lw=2.0, ms=5, label=LAB.get(a, a))
    ax.axhline(big, color="#bbb", ls=":", lw=1.0)
    ax.text(0.01, big, f"censored (>{big})", color="#999", fontsize=8, va="bottom")
    ax.set_xlabel("drift index (novel φ each drift)")
    ax.set_ylabel("transitions-to-recover  (↓ = cheaper adaptation)")
    ax.set_title(f"Compounding: per-drift adaptation cost over a sequence\n"
                 f"(family={R['family']}, mode={R['drift_mode']})", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    figs["fig1_ttr_per_drift.png"] = _save(fig)

    # ---- fig2: cumulative transitions (sublinear = compounding) ----
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for a in arms:
        ax.plot(x, R["cumulative"][a], "-", color=COL.get(a, "#555"), lw=2.2, label=LAB.get(a, a))
    ax.set_xlabel("drifts survived")
    ax.set_ylabel("cumulative transitions-to-recover")
    ax.set_title("Cumulative adaptation cost: sublinear (compounding) vs linear", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    figs["fig2_cumulative.png"] = _save(fig)

    # ---- fig3: recovery curves early vs late (why the cost drops) ----
    if "factored" in arms:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        xb = [max(N, 1) for N in R["budgets"]]
        for ax, which, ttl in [(axes[0], 0, "early drift"), (axes[1], T - 1, "late drift")]:
            for a in arms:
                c = R["curves"][a][which]
                ax.plot(xb, [c[str(N)] for N in R["budgets"]], "o-", color=COL.get(a, "#555"),
                        lw=1.8, ms=4, label=LAB.get(a, a))
            ax.axhline(R["recover_threshold"], color="#2f9e44", ls="--", lw=1.0)
            ax.set_xscale("log"); ax.set_xlabel("adaptation transitions N")
            ax.set_title(f"{ttl} (t={which})", fontsize=10)
        axes[0].set_ylabel("held-out velocity-dim Δs R²")
        axes[1].legend(fontsize=8, loc="lower right")
        fig.suptitle("Recovery curves: the factored core makes late drifts cheap to fit", fontsize=10.5)
        figs["fig3_recovery_curves.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def compounding_drift(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    family: str = "actuator",
    conflict: float = 1.5708,           # Φ = π/2 (the clean meta≫multitask regime)
    base_damping: float = 1.0,
    damp_min: float = 0.05,
    damp_max: float = 2.0,
    drift_mode: str = "novel",          # novel | revisit
    revisit_bank: int = 3,
    n_drifts: int = 24,
    carve_sweep: bool = False,
    # env / collection
    frame_skip: int = 12,
    gear: float = 10.0,
    arena_half: float = 2.0,
    v_explore: float = 1.2,
    collect_ep_len: int = 20,
    collect_pos_range: float = 1.0,
    drift_pool_n: int = 1500,
    norm_n: int = 600,
    oracle_n: int = 6000,
    # arch
    latent_dim: int = 4,
    enc_hidden: int = 128,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    # slow loop / meta
    meta_lr: float = 1e-3,
    fm_lr: float = 1e-3,
    slow_steps: int = 400,
    offline_meta_steps: int = 4000,
    n_offline_tasks: int = 16,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    # fast adapt / oracle
    adapt_epochs: int = 40,
    fm_epochs: int = 40,
    adapt_frac: float = 0.85,           # recover threshold = frac x oracle (0.999 here)
    censor_mult: float = 1.5,
):
    import os

    budgets = [0, 5, 10, 20, 40, 80, 160, 320]
    arms = ["factored", "mono_replay", "mono_noreplay", "offline_meta", "scratch"]
    if quick:
        n_drifts = 6
        drift_pool_n, norm_n, oracle_n = 800, 400, 2500
        enc_hidden, fm_hidden, fm_layers = 64, 128, 2
        slow_steps, offline_meta_steps, n_offline_tasks = 120, 800, 6
        adapt_epochs = fm_epochs = 20
        budgets = [0, 10, 40, 160]
        tag = tag or "smoke"
    tag = tag or "default"

    base_cfg = dict(
        seed=seed, family=family, conflict=conflict, base_damping=base_damping,
        damp_min=damp_min, damp_max=damp_max, drift_mode=drift_mode, revisit_bank=revisit_bank,
        n_drifts=n_drifts, frame_skip=frame_skip,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0, gear=gear),
        v_explore=v_explore, collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        drift_pool_N=drift_pool_n, norm_N=norm_n, oracle_N=oracle_n,
        latent_dim=latent_dim, enc_hidden=enc_hidden, fm_hidden=fm_hidden, fm_layers=fm_layers,
        meta_lr=meta_lr, fm_lr=fm_lr, slow_steps=slow_steps, offline_meta_steps=offline_meta_steps,
        n_offline_tasks=n_offline_tasks, task_batch=task_batch, pred_P=pred_p, grad_clip=grad_clip,
        adapt_epochs=adapt_epochs, fm_epochs=fm_epochs, adapt_frac=adapt_frac,
        censor_mult=censor_mult, budgets=budgets, arms=arms,
    )

    def _mirror(out, subtag):
        localdir = os.path.join(os.path.dirname(__file__), "figures", "compounding_drift_" + subtag)
        os.makedirs(localdir, exist_ok=True)
        for nm, png in out["figures"].items():
            with open(os.path.join(localdir, nm), "wb") as fh:
                fh.write(png)
        with open(os.path.join(localdir, "results.json"), "w") as fh:
            json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
        print(f"[local] wrote {len(out['figures'])} figures + results.json to {localdir}")

    if carve_sweep:
        # "nothing more, nothing less": sweep latent_dim -> optimum where you carve exactly φ
        dims = [1, 2, 4, 8, 16] if not quick else [1, 4, 16]
        summ = {}
        for dz in dims:
            cfg = dict(base_cfg); cfg["latent_dim"] = dz; cfg["arms"] = ["factored"]
            cfg["tag"] = f"{tag}_d{dz}"
            out = run_compounding_drift.remote(cfg)
            _mirror(out, f"{tag}_d{dz}")
            summ[dz] = out["results"]["ttr_numeric"]["factored"]
        print("\n[carve-sweep] latent_dim -> late-drift mean transitions-to-recover:")
        for dz, ttr in summ.items():
            late = float(sum(ttr[-max(1, base_cfg["n_drifts"] // 4):]) / max(1, base_cfg["n_drifts"] // 4))
            print(f"   d={dz:2d}: {late:.1f}")
    else:
        cfg = dict(base_cfg); cfg["tag"] = tag
        out = run_compounding_drift.remote(cfg)
        _mirror(out, tag)


# ========================================================================= #
# Phase 2 — value-driven carving under a value-IRRELEVANT drift.
#
# Composes Cut #4d/#4e's capacity-competition substrate (puck force field =
# value-irrelevant capacity sink; push_rot phi = value-relevant conflict;
# context-latent f(s,u,z)->Δs(8)) with Phase 1's online drift SEQUENCE. The
# NEW ingredient: the value-irrelevant subspace DRIFTS too (puck_phase θ_t
# rotates each drift), so a VERIDICAL FM (models all 8 dims) must re-learn the
# drifting puck every drift, while a VALUE-CARVED FM (drops the puck via the
# value's support) tracks only φ.
#
#   PREDICTION. The value-carved FM re-adapts the value-relevant pusher-vel in
#   FEWER transitions per drift AND compounds better — because value-carving
#   shrinks the adaptable subspace to the value-relevant varying part (φ only),
#   not everything that varies (φ AND θ). Crucially this DIRECTLY re-tests the
#   first online loop's efferent obstruction: the value-shaping benefit that
#   "washed out at convergence" PERSISTS under perpetual drift (every drift is a
#   fresh pre-convergence sprint — the permanent operating regime).
# ========================================================================= #
@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_value_carved_drift(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import collect_transitions, PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    d = cfg["latent_dim"]
    budgets = cfg["budgets"]
    puck_w = cfg["shaped_puck_weight"]

    # ---- the drift SEQUENCE: (φ push_rot, θ puck-field phase) both novel ---- #
    seq_rng = np.random.default_rng(cfg["seed"] + 11)
    T = cfg["n_drifts"]
    Phi = cfg["conflict"]
    phi_seq = [float(seq_rng.uniform(-Phi, Phi)) for _ in range(T)]
    # θ drifts the value-IRRELEVANT puck field (unless --static-puck: then it's a fixed
    # value-irrelevant sink and only φ drifts — the control isolating the drift channel)
    if cfg["static_puck"]:
        theta_seq = [0.0 for _ in range(T)]
    else:
        theta_seq = [float(seq_rng.uniform(-np.pi, np.pi)) for _ in range(T)]
    print(f"[seq] T={T} static_puck={cfg['static_puck']} field_pusher_amp={cfg['field_pusher_amp']}\n"
          f"[seq] φ=" + " ".join(f"{p:+.2f}" for p in phi_seq)
          + "\n[seq] θ=" + " ".join(f"{p:+.2f}" for p in theta_seq), flush=True)

    def task_dgp(phi, theta):
        pf = dict(amp=cfg["field_amp"], pusher_amp=cfg["field_pusher_amp"],
                  central=cfg["field_central"], puck_phase=theta)
        dd = dict(cfg["dgp_base"]); dd["push_rot"] = float(phi); dd["puck_field"] = pf
        return dd

    def collect(dgp, seed):
        return collect_transitions(dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
                                   frame_skip=fs, seed=seed, sigma=cfg["sigma"],
                                   theta=cfg["ou_theta"], seek_gain=cfg["seek_gain"])

    # ---- family normalization (fixed once, shared by both arms) ------------- #
    print("[norm] calibrating family normalization ...", flush=True)
    calib = [collect(task_dgp(p, t), cfg["seed"] + 700 + i) for i, (p, t) in enumerate(
        zip(np.linspace(-Phi, Phi, 4), np.linspace(-np.pi, np.pi, 4)))]
    allS = np.concatenate([c["S"] for c in calib]); allU = np.concatenate([c["U"] for c in calib])
    allS2 = np.concatenate([c["S2"] for c in calib])
    X = np.concatenate([allS, allU], 1).astype(np.float32); Y = (allS2 - allS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
    mx, sx, my, sy = norm["mx"], norm["sx"], norm["my"], norm["sy"]

    def feats(S, U, S2):
        """normalized (su_n [.,10], dy_n [.,8], ctx_n [.,18]) on device."""
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (Xt - mx) / sx; dy_n = (Yt - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    def r2(pred, yt, dims, mask):
        if mask is not None:
            pred, yt = pred[mask], yt[mask]
        p, t = pred[:, dims], yt[:, dims]
        ssr = ((t - p) ** 2).sum(); sst = ((t - t.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    # ---- architectures (meta_value_shaping idiom) -------------------------- #
    eh, fh, fl = cfg["enc_hidden"], cfg["fm_hidden"], cfg["fm_layers"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(18, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, d))

        def forward(self, ctx):
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm():
        lyr = [nn.Linear(10 + d, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(fh, 8)])).to(device)

    huber = nn.HuberLoss(delta=1.0, reduction="none")

    def value_weight(carved):
        w = torch.ones(8, device=device)
        if carved:
            for i in PUCK:
                w[i] = puck_w                        # drop the value-irrelevant (drifting) puck
        return w

    def meta_train(enc, cfm, opt, banks, w, steps, rng):
        ctx_Ns = [b for b in budgets if b > 0]
        P = cfg["pred_P"]; Bt = min(cfg["task_batch"], len(banks))
        enc.train(); cfm.train()
        for _ in range(steps):
            K = int(rng.choice(ctx_Ns))
            tids = rng.choice(len(banks), size=Bt, replace=False)
            ctx = torch.empty(Bt, K, 18, device=device)
            psu = torch.empty(Bt, P, 10, device=device)
            pdy = torch.empty(Bt, P, 8, device=device)
            for j, tid in enumerate(tids):
                su_n, dy_n, cn = banks[tid]; n = su_n.shape[0]
                ci = torch.tensor(rng.integers(0, n, size=K), device=device)
                pi = torch.tensor(rng.integers(0, n, size=P), device=device)
                ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
            z = enc(ctx)
            pred = cfm(torch.cat([psu, z.unsqueeze(1).expand(-1, P, -1)], 2))
            loss = (huber(pred, pdy) * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
            opt.step()
        enc.eval(); cfm.eval()

    # ---- family oracle pusher-vel ceiling (plain f(s,u), all dims) --------- #
    def train_plain(S, U, S2, epochs):
        lyr = [nn.Linear(10, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        fm = nn.Sequential(*(lyr + [nn.Linear(fh, 8)])).to(device)
        xn = (torch.tensor(np.concatenate([S, U], 1), device=device) - mx) / sx
        yn = (torch.tensor((S2 - S).astype(np.float32), device=device) - my) / sy
        o = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"]); n = xn.shape[0]; bs = min(512, n)
        fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                o.zero_grad(); huber(fm(xn[idx]), yn[idx]).mean().backward(); o.step()
        fm.eval(); return fm

    print("[oracle] family pusher-vel ceiling ...", flush=True)
    oc = collect(task_dgp(phi_seq[0], theta_seq[0]), cfg["seed"] + 850)
    ff_o = ~oc["any_contact"]
    fm_o = train_plain(oc["S"][:-2000], oc["U"][:-2000], oc["S2"][:-2000], cfg["fm_epochs"])
    with torch.no_grad():
        xh = (torch.tensor(np.concatenate([oc["S"][-2000:], oc["U"][-2000:]], 1), device=device) - mx) / sx
        pred_o = (fm_o(xh) * sy + my).cpu().numpy()
    oracle_pv = r2(pred_o, (oc["S2"][-2000:] - oc["S"][-2000:]).astype(np.float32), PUSHER_VEL, ff_o[-2000:])
    thr = cfg["adapt_frac"] * oracle_pv
    print(f"[oracle] pusher-vel R^2={oracle_pv:.4f} -> recover threshold={thr:.4f}", flush=True)

    # ---- CEM-MPC goal-reaching (CONTROL-level re-adaptation; guidance #3: the
    #      load-bearing metric, since FM gaps may not transmit through replanning) - #
    do_ctrl = cfg["control"]
    ctrl_budgets = cfg["control_budgets"]
    if do_ctrl:
        pe = np.random.default_rng(cfg["seed"] + 11)
        B = cfg["n_eval_plan"]; ah = cfg["dgp_base"]["arena_half"]; sr = cfg["start_range"]
        ppos = pe.uniform(-sr, sr, (B, 2)); qpos_puck = pe.uniform(-0.6 * ah, 0.6 * ah, (B, 2))
        pvel = pe.normal(0, cfg["v0_std"], (B, 2)); qvel_puck = np.zeros((B, 2))
        ev_starts = np.concatenate([ppos, qpos_puck, pvel, qvel_puck], 1).astype(np.float32)
        ev_goals = pe.uniform(-cfg["goal_range"], cfg["goal_range"], (B, 2)).astype(np.float32)
        rand_dist = float(np.median(np.linalg.norm(ev_starts[:, PUSHER_POS] - ev_goals, axis=1)))
        Hep, Hp, re_ = cfg["plan_H"], cfg["plan_Hp"], cfg["replan_every"]
        Kp, n_elite, vel_pen = cfg["k_shoot"], cfg["cem_elite"], cfg["vel_pen"]
        ctrl_thr = cfg["control_thr"]
        print(f"[control] eval B={B} random start-dist={rand_dist:.3f} thr={ctrl_thr:.3f}", flush=True)

        def mpc_action_puck(cfm, z, states, goals, rng):
            Bn = states.shape[0]
            mu = np.zeros((Bn, Hp, 2), np.float32); sig = np.full((Bn, Hp, 2), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(goals, device=device).repeat_interleave(Kp, 0)
            s0 = torch.tensor(states, device=device).repeat_interleave(Kp, 0)
            zc = z.unsqueeze(0).expand(Bn * Kp, -1)
            for _ in range(cfg["cem_iters"]):
                e = rng.standard_normal((Bn, Kp, Hp, 2)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kp, Hp, 2), device=device)
                    cost = torch.zeros(Bn * Kp, device=device)
                    for h in range(Hp):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + (cfm(torch.cat([(x - mx) / sx, zc], 1)) * sy + my)
                        cost = cost + (s[:, PUSHER_POS] - g_t).norm(dim=1)
                    cost = cost + vel_pen * s[:, PUSHER_VEL].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, Kp), n_elite, dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)

        def eval_control(cfm, z, env, seed):
            rng = np.random.default_rng(seed)
            states = ev_starts.copy(); plan = None
            for step in range(Hep):
                if step % re_ == 0:
                    plan = mpc_action_puck(cfm, z, states, ev_goals, rng)
                acts = plan[:, step % re_, :]
                for b in range(states.shape[0]):
                    env.set_state(states[b, 0:4], states[b, 4:8])
                    s2, _ = env.step(acts[b], fs); states[b] = s2
            return float(np.median(np.linalg.norm(states[:, PUSHER_POS] - ev_goals, axis=1)))

    # ---- persistent arms --------------------------------------------------- #
    arms = ["value_carved", "veridical"]
    state = {}
    for a in arms:
        enc, cfm = Encoder().to(device), build_cfm()
        state[a] = dict(enc=enc, cfm=cfm, w=value_weight(a == "value_carved"),
                        opt=torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()),
                                             lr=cfg["meta_lr"]),
                        banks=[])
    rng_slow = np.random.default_rng(cfg["seed"] + 17)

    ttr = {a: [] for a in arms}          # FM transitions-to-recover pusher-vel per drift
    pv_curves = {a: [] for a in arms}    # pusher-vel R2 vs N per drift
    puck_r2 = {a: [] for a in arms}      # puck-vel R2 @ max budget (drop-vs-keep story)
    ctrl_ttr = {a: [] for a in arms}     # CONTROL transitions-to-recover per drift
    ctrl_curves = {a: [] for a in arms}  # control goal-dist vs N per drift

    def recover_N(r2b):
        for N in budgets:
            if N > 0 and r2b[N] >= thr:
                return N
        return None

    def recover_N_ctrl(cb):
        for N in ctrl_budgets:
            if N > 0 and cb[N] <= ctrl_thr:
                return N
        return None

    # ===================================================================== #
    # THE DRIFT LOOP
    # ===================================================================== #
    for t, (phi, theta) in enumerate(zip(phi_seq, theta_seq)):
        dt = collect(task_dgp(phi, theta), cfg["seed"] + 4000 + t)
        S, U, S2, ac = dt["S"], dt["U"], dt["S2"], dt["any_contact"]
        Yd = (S2 - S).astype(np.float32)
        puck_moving = np.abs(Yd[:, PUCK_VEL]).max(1) > cfg["puck_move_thresh"]
        nctx = max(budgets)
        cS, cU, cS2 = S[:nctx], U[:nctx], S2[:nctx]
        _, _, cn_ctx = feats(cS, cU, cS2)
        nH = cfg["n_holdout"]
        hS, hU, hS2 = S[-nH:], U[-nH:], S2[-nH:]
        h_ff = ~ac[-nH:]; h_slide = (~ac[-nH:]) & puck_moving[-nH:]
        hsu, _, _ = feats(hS, hU, hS2)
        yt_h = (hS2 - hS).astype(np.float32)
        banks_feat = feats(S, U, S2)
        env_ctrl = PusherEnv(task_dgp(phi, theta), with_puck=True) if do_ctrl else None

        line = [f"drift {t:2d} φ={phi:+.2f} θ={theta:+.2f}"]
        for a in arms:
            st = state[a]
            r2b, puck_at = {}, None
            for N in budgets:
                if N == 0:
                    z = torch.zeros(d, device=device)
                else:
                    with torch.no_grad():
                        z = st["enc"](cn_ctx[:N].unsqueeze(0))[0]
                with torch.no_grad():
                    pred = (st["cfm"](torch.cat([hsu, z.expand(hsu.shape[0], -1)], 1)) * sy + my).cpu().numpy()
                r2b[N] = r2(pred, yt_h, PUSHER_VEL, h_ff) if h_ff.sum() >= 20 else float("nan")
                if N == nctx:
                    puck_at = r2(pred, yt_h, PUCK_VEL, h_slide) if h_slide.sum() >= 20 else float("nan")
            pv_curves[a].append(r2b)
            puck_r2[a].append(puck_at)
            rN = recover_N(r2b)
            ttr[a].append(rN)
            seg = f"{a}:recov@{rN if rN is not None else '>'+str(nctx)} puckR²={puck_at:+.2f}"
            # ---- CONTROL-level re-adaptation (guidance #3) ---- #
            if do_ctrl:
                cb = {}
                for N in ctrl_budgets:
                    if N == 0:
                        z = torch.zeros(d, device=device)
                    else:
                        with torch.no_grad():
                            z = st["enc"](cn_ctx[:N].unsqueeze(0))[0]
                    cb[N] = eval_control(st["cfm"], z, env_ctrl, cfg["seed"] + 300 + t)
                ctrl_curves[a].append(cb)
                cN = recover_N_ctrl(cb)
                ctrl_ttr[a].append(cN)
                seg += f" | ctrl-recov@{cN if cN is not None else '>'+str(max(ctrl_budgets))}"
            line.append(seg)

        # ---- slow consolidation (the value-carving, established over the sequence) ---- #
        for a in arms:
            st = state[a]
            st["banks"].append(banks_feat)
            if len(st["banks"]) >= 2:
                meta_train(st["enc"], st["cfm"], st["opt"], st["banks"], st["w"], cfg["slow_steps"], rng_slow)
        print("  ".join(line), flush=True)

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    big = max(budgets)
    ttr_num = {a: [(rN if rN is not None else big * cfg["censor_mult"]) for rN in ttr[a]] for a in arms}
    cum = {a: np.cumsum(ttr_num[a]).tolist() for a in arms}
    cbig = max(ctrl_budgets) if do_ctrl else 0
    ctrl_num = {a: [(cN if cN is not None else cbig * cfg["censor_mult"]) for cN in ctrl_ttr[a]] for a in arms} \
        if do_ctrl else {}
    ctrl_cum = {a: np.cumsum(ctrl_num[a]).tolist() for a in arms} if do_ctrl else {}
    results = {
        "config": cfg, "phi_seq": phi_seq, "theta_seq": theta_seq, "budgets": budgets,
        "control_budgets": ctrl_budgets if do_ctrl else [], "control": do_ctrl,
        "oracle_pusher_vel": oracle_pv, "recover_threshold": thr,
        "control_thr": cfg["control_thr"] if do_ctrl else None,
        "control_random_dist": rand_dist if do_ctrl else None,
        "ttr": {a: ttr[a] for a in arms}, "ttr_numeric": ttr_num, "cumulative": cum,
        "ctrl_ttr": {a: ctrl_ttr[a] for a in arms}, "ctrl_ttr_numeric": ctrl_num, "ctrl_cumulative": ctrl_cum,
        "puck_r2": {a: puck_r2[a] for a in arms},
        "pv_curves": {a: [{str(k): v for k, v in c.items()} for c in pv_curves[a]] for a in arms},
        "ctrl_curves": {a: [{str(k): v for k, v in c.items()} for c in ctrl_curves[a]] for a in arms}
        if do_ctrl else {},
    }
    print("\n===== VALUE-CARVED DRIFT SUMMARY =====", flush=True)
    for a in arms:
        early = float(np.mean(ttr_num[a][:max(1, T // 4)]))
        late = float(np.mean(ttr_num[a][-max(1, T // 4):]))
        pk = float(np.nanmean(puck_r2[a]))
        msg = (f"  {a:13s} FM-recover@ early={early:6.1f} late={late:6.1f} Δ={late-early:+6.1f} "
               f"cum={cum[a][-1]:.0f}  mean puck-R²={pk:+.2f}")
        if do_ctrl:
            ce = float(np.mean(ctrl_num[a][:max(1, T // 4)])); cl = float(np.mean(ctrl_num[a][-max(1, T // 4):]))
            msg += f"  | CTRL early={ce:5.1f} late={cl:5.1f} cum={ctrl_cum[a][-1]:.0f}"
        print(msg, flush=True)

    figures = _make_value_carved_figures(results)
    outdir = os.path.join(DATA_DIR, "value_carved_drift", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_value_carved_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"value_carved": "#3d6fd1", "veridical": "#d1603d"}
    LAB = {"value_carved": "value-carved (drop drifting puck)", "veridical": "veridical (model everything)"}
    figs = {}
    arms = list(R["ttr_numeric"].keys())
    T = len(R["phi_seq"])
    x = np.arange(T)
    big = max(R["budgets"])

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: transitions-to-recover (value-relevant pusher-vel) per drift ----
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for a in arms:
        ax.plot(x, R["ttr_numeric"][a], "o-", color=COL[a], lw=2.1, ms=5, label=LAB[a])
    ax.axhline(big, color="#bbb", ls=":", lw=1.0)
    ax.text(0.01, big, f"censored (>{big})", color="#999", fontsize=8, va="bottom")
    ax.set_xlabel("drift index (novel φ + drifting puck field θ each drift)")
    ax.set_ylabel("transitions-to-recover pusher-vel  (↓ = cheaper)")
    ax.set_title("Value-carving buys faster re-adaptation of the VALUE-RELEVANT dynamics,\n"
                 "and the benefit PERSISTS under perpetual drift (never washes out)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    figs["fig1_ttr_pusher_vel.png"] = _save(fig)

    # ---- fig2: cumulative transitions ----
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for a in arms:
        ax.plot(x, R["cumulative"][a], "-", color=COL[a], lw=2.3, label=LAB[a])
    ax.set_xlabel("drifts survived"); ax.set_ylabel("cumulative transitions-to-recover")
    ax.set_title("Cumulative adaptation cost across the drift sequence", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    figs["fig2_cumulative.png"] = _save(fig)

    # ---- fig3: the drop story — puck-vel R² per drift (veridical tracks the
    #            drifting puck, wasting capacity; value-carved ignores it) ----
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for a in arms:
        y = [v if v == v else np.nan for v in R["puck_r2"][a]]
        ax.plot(x, y, "o-", color=COL[a], lw=2.0, ms=4, label=LAB[a])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylim(-1.6, 1.03)
    ax.set_xlabel("drift index"); ax.set_ylabel("puck-vel R² (value-IRRELEVANT)")
    ax.set_title("Veridical spends capacity re-learning the drifting puck each drift;\n"
                 "value-carved drops it (R²≪0) — the capacity the carving frees", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    figs["fig3_puck_drop.png"] = _save(fig)

    # ---- fig4: CONTROL-level re-adaptation per drift (the load-bearing metric) ----
    if R.get("control") and R.get("ctrl_ttr_numeric"):
        cbig = max(R["control_budgets"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for a in arms:
            axes[0].plot(x, R["ctrl_ttr_numeric"][a], "o-", color=COL[a], lw=2.1, ms=5, label=LAB[a])
            axes[1].plot(x, R["ctrl_cumulative"][a], "-", color=COL[a], lw=2.3, label=LAB[a])
        axes[0].axhline(cbig, color="#bbb", ls=":", lw=1.0)
        axes[0].set_xlabel("drift index"); axes[0].set_ylabel("CONTROL transitions-to-recover (↓)")
        axes[0].set_title("Control re-adaptation per drift", fontsize=10); axes[0].legend(fontsize=8.5)
        axes[1].set_xlabel("drifts survived"); axes[1].set_ylabel("cumulative control transitions")
        axes[1].set_title("Cumulative control-adaptation cost", fontsize=10)
        fig.suptitle("Does value-carving's faster re-adaptation reach CONTROL?  "
                     "(guidance #3: FM gaps may not transmit through CEM-MPC replanning)", fontsize=10.5)
        figs["fig4_control_ttr.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def value_carved_drift(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    conflict: float = 1.5708,
    n_drifts: int = 16,
    static_puck: bool = False,          # True = puck field fixed (only φ drifts) — the control
    # env / field (4d capacity-competition substrate)
    frame_skip: int = 3,
    arena_half: float = 0.9,
    gear: float = 10.0,
    puck_mass: float = 0.5,
    field_amp: float = 1.3,             # puck field (value-irrelevant sink)
    field_pusher_amp: float = 2.0,      # pusher field ON = capacity competition (4d's condition)
    field_central: float = 0.5,
    # collection
    n_episodes: int = 40,
    ep_len: int = 160,
    sigma: float = 0.7,
    ou_theta: float = 0.15,
    seek_gain: float = 0.9,
    n_holdout: int = 3000,
    puck_move_thresh: float = 0.02,
    shaped_puck_weight: float = 0.0,
    # arch (small = capacity-bound, the 4d sweet spot)
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_hidden: int = 64,
    fm_layers: int = 2,
    # slow / fast
    meta_lr: float = 1e-3,
    fm_lr: float = 1e-3,
    slow_steps: int = 400,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    fm_epochs: int = 40,
    adapt_frac: float = 0.85,
    censor_mult: float = 1.5,
    # ---- CONTROL-level re-adaptation (CEM-MPC goal-reaching; guidance #3) ---- #
    control: bool = True,
    control_thr: float = 0.18,          # goal-dist below which control is "recovered"
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    plan_h: int = 80,
    plan_hp: int = 16,
    replan_every: int = 8,
    k_shoot: int = 160,
    cem_iters: int = 3,
    cem_elite: int = 20,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.3,
    n_eval_plan: int = 20,
):
    import os

    budgets = [0, 5, 10, 20, 40, 80, 160]
    control_budgets = [0, 20, 80, 160]
    if quick:
        n_drifts = 6
        n_episodes, ep_len = 24, 120
        enc_hidden = 64
        slow_steps = 150
        fm_epochs = 20
        n_holdout = 2000
        budgets = [0, 10, 40, 160]
        control_budgets = [0, 40, 160]
        plan_h, k_shoot, n_eval_plan = 60, 96, 12
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, conflict=conflict, n_drifts=n_drifts, static_puck=static_puck,
        frame_skip=frame_skip,
        dgp_base=dict(arena_half=arena_half, gear=gear, puck_mass=puck_mass),
        field_amp=field_amp, field_pusher_amp=field_pusher_amp, field_central=field_central,
        n_episodes=n_episodes, ep_len=ep_len, sigma=sigma, ou_theta=ou_theta, seek_gain=seek_gain,
        n_holdout=n_holdout, puck_move_thresh=puck_move_thresh, shaped_puck_weight=shaped_puck_weight,
        latent_dim=latent_dim, enc_hidden=enc_hidden, fm_hidden=fm_hidden, fm_layers=fm_layers,
        meta_lr=meta_lr, fm_lr=fm_lr, slow_steps=slow_steps, task_batch=task_batch, pred_P=pred_p,
        grad_clip=grad_clip, fm_epochs=fm_epochs, adapt_frac=adapt_frac, censor_mult=censor_mult,
        budgets=budgets, control=control, control_budgets=control_budgets, control_thr=control_thr,
        goal_range=goal_range, start_range=start_range, v0_std=v0_std,
        plan_H=plan_h, plan_Hp=plan_hp, replan_every=replan_every, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        n_eval_plan=n_eval_plan,
    )
    out = run_value_carved_drift.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "value_carved_drift_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
