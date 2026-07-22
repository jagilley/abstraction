"""Cut #4b — Context-latent meta-adaptation: the dissectible outer memory.

Program: `ideas/two_timescale_value_loop.md` (§"The interface shape": the value system
reads a LOW-DIM slice of the FM; z is that slice) and the pivot guardrail (a FACTORED,
DISSECTIBLE architecture so "what worked" is legible). Parent: `META_ADAPT_README.md`
(Reptile-init opened the gap under actuator-rotation conflict, but the outer memory was
implicit in the weights — not probeable).

    THE MOVE. Replace the Reptile weight-init with an EXPLICIT context latent z conditioning
    the forward model: f(s,u,z) -> Δs. A permutation-invariant encoder E infers z from a
    few recent (s,u,Δs) transitions (PEARL / amortized system-ID; Rakelly et al. 2019).
    Few-shot "adaptation" is then a FORWARD PASS (encode N context transitions -> z), no
    gradient steps. Two claims:
      1. The context latent ALSO opens the gap (context z vs context-severed z=0) under
         conflict, and collapses at the smooth floor -> the phenomenon is not Reptile-specific.
      2. z DECODES the task parameter (rotation phi / damping): the meta-learner is doing
         SYSTEM IDENTIFICATION, and z is the low-dim value-relevant direction the two-
         timescale doc predicts. A linear probe z -> phi is the dissectible payoff.

DESIGN. Same substrate/family machinery as `meta_adapt.py` (Cut #3 puck-free reaching;
family="damping" the floor, "actuator" the input-coupled conflict via `pusher_env.py`'s
`push_rot`). ARMS (only the context differs):
  * context : f(s,u, z=E(first N context transitions))  -- the meta arm (amortized).
  * z0      : f(s,u, z=0)                                -- context severed = family-average
              (the pooled control; equals `context` at N=0).
  * oracle  : a plain per-task f(s,u) fit on a large disjoint pool (arch-independent ceiling).
Readout = held-out velocity-dim Δs R^2 vs #context transitions N (median over test tasks)
+ the z->phi LINEAR PROBE (train-task z -> fit; test-task z -> R^2 + scatter).

Run:
    cd experiments/
    modal run mjc/meta_adapt/meta_context.py::meta_context --quick                                   # smoke (actuator pi/2)
    modal run mjc/meta_adapt/meta_context.py::meta_context --tag ctx_p2 --family actuator --conflict 1.5708
    modal run mjc/meta_adapt/meta_context.py::meta_context --tag ctx_floor --family damping          # the floor control
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_meta_context(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    budgets = cfg["budgets"]
    d = cfg["latent_dim"]
    VEL_DIMS = [2, 3]

    # ---- task family (same as meta_adapt.py) ------------------------------- #
    family = cfg["family"]
    n_tr, n_te = cfg["n_train_tasks"], cfg["n_test_tasks"]
    if family == "damping":
        all_p = np.logspace(np.log10(cfg["damp_min"]), np.log10(cfg["damp_max"]), n_tr + n_te)
    elif family == "actuator":
        Phi = cfg["conflict"]
        all_p = np.linspace(-Phi, Phi, n_tr + n_te)
    else:
        raise ValueError(family)
    all_p = np.sort(all_p)
    te_idx = np.unique(np.linspace(1, n_tr + n_te - 2, n_te).round().astype(int))
    tr_idx = np.array([i for i in range(n_tr + n_te) if i not in te_idx])
    train_p = all_p[tr_idx].astype(np.float64)
    test_p = all_p[te_idx].astype(np.float64)
    print(f"[family] {family} | train {np.round(train_p,3)}\n[family] test {np.round(test_p,3)}",
          flush=True)

    def task_dgp(p):
        dd = dict(cfg["dgp_base"])
        if family == "damping":
            dd["joint_damping"] = float(p)
        else:
            dd["joint_damping"] = cfg["base_damping"]; dd["push_rot"] = float(p)
        return dd

    # ---- reward-free central OU collection (Cut #3's `collect`) ------------- #
    def collect(env, n_transitions, seed, ou_theta=0.2, ou_sigma=0.5):
        rng = np.random.default_rng(seed)
        ep_len, pr = cfg["collect_ep_len"], cfg["collect_pos_range"]
        S, U, S2, C = [], [], [], []
        for _ in range(int(np.ceil(n_transitions / ep_len))):
            p0 = rng.uniform(-pr, pr, size=2)
            env.set_state(p0, rng.normal(0, cfg["v_explore"], size=2))
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

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    # ===================================================================== #
    # 1. collect per-task buffers + family normalization
    # ===================================================================== #
    print("\n[1] collecting per-task buffers ...", flush=True)
    train_bufs = [collect(PusherEnv(task_dgp(p), with_puck=False), cfg["train_task_N"],
                          cfg["seed"] + 1000 + i) for i, p in enumerate(train_p)]
    nH, nA = cfg["n_holdout"], cfg["n_adapt_pool"]
    test_bufs, test_envs = [], []
    for i, p in enumerate(test_p):
        env = PusherEnv(task_dgp(p), with_puck=False); test_envs.append(env)
        S, U, S2, C = collect(env, cfg["test_task_N"], cfg["seed"] + 2000 + i)
        test_bufs.append({"adapt": (S[:nA], U[:nA], S2[:nA], C[:nA]),
                          "oracle": (S[nA:-nH], U[nA:-nH], S2[nA:-nH], C[nA:-nH]),
                          "hold": (S[-nH:], U[-nH:], S2[-nH:], C[-nH:])})
    allS = np.concatenate([b[0] for b in train_bufs])
    allU = np.concatenate([b[1] for b in train_bufs])
    allS2 = np.concatenate([b[2] for b in train_bufs])
    norm = make_norm(allS, allU, allS2)
    mx, sx, my, sy = norm["mx"], norm["sx"], norm["my"], norm["sy"]

    def feats(S, U, S2):
        """normalized (su_n [.,6], dy_n [.,4], ctx_n [.,10]) tensors on device."""
        X = torch.tensor(np.concatenate([S, U], 1), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (X - mx) / sx; dy_n = (Y - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    # precompute normalized tensors per train task (context source + prediction source)
    tr_feat = [feats(b[0], b[1], b[2]) for b in train_bufs]

    # ===================================================================== #
    # 2. encoder E (DeepSets, permutation-invariant) + conditional FM f(s,u,z)
    # ===================================================================== #
    eh, fh, fl = cfg["enc_hidden"], cfg["fm_hidden"], cfg["fm_layers"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(10, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, d))

        def forward(self, ctx):                 # ctx: (B, K, 10) -> z: (B, d)
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm():
        lyr = [nn.Linear(6 + d, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(fh, 4)])).to(device)

    enc = Encoder().to(device)
    cfm = build_cfm()

    def cfm_pred_n(su_n, z):
        """su_n: (n,6); z: (d,) or (n,d) -> normalized Δs pred (n,4)."""
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(su_n.shape[0], -1)
        return cfm(torch.cat([su_n, z], 1))

    # ===================================================================== #
    # 3. META-TRAIN E + f jointly (amortized inference over the task distribution)
    # ===================================================================== #
    print(f"\n[2] meta-training encoder + conditional FM ({cfg['meta_steps']} steps) ...", flush=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()), lr=cfg["meta_lr"])
    lossf = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(cfg["seed"] + 5)
    ctx_Ns = [b for b in budgets if b > 0]
    Bt, P = min(cfg["task_batch"], len(tr_feat)), cfg["pred_P"]
    enc.train(); cfm.train()
    for step in range(cfg["meta_steps"]):
        K = int(rng.choice(ctx_Ns))                     # variable context size -> robust across N
        tids = rng.choice(len(tr_feat), size=Bt, replace=False)
        ctx = torch.empty(Bt, K, 10, device=device)
        psu = torch.empty(Bt, P, 6, device=device)
        pdy = torch.empty(Bt, P, 4, device=device)
        for j, tid in enumerate(tids):
            su_n, dy_n, cn = tr_feat[tid]; n = su_n.shape[0]
            ci = torch.tensor(rng.integers(0, n, size=K), device=device)
            pi = torch.tensor(rng.integers(0, n, size=P), device=device)
            ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
        z = enc(ctx)                                    # (Bt, d)
        zc = z.unsqueeze(1).expand(-1, P, -1)           # (Bt, P, d)
        pred = cfm(torch.cat([psu, zc], 2))             # (Bt, P, 4)
        loss = lossf(pred, pdy)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
        opt.step()
        if (step + 1) % max(1, cfg["meta_steps"] // 8) == 0:
            print(f"    [meta] step {step+1}/{cfg['meta_steps']}  huber={loss.item():.4f}", flush=True)
    enc.eval(); cfm.eval()

    # ===================================================================== #
    # 4. per-test-task oracle ceiling (plain f(s,u), arch-independent)
    # ===================================================================== #
    def train_plain_fm(S, U, S2, epochs, lr):
        lyr = [nn.Linear(6, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        fm = nn.Sequential(*(lyr + [nn.Linear(fh, 4)])).to(device)
        su_n, dy_n, _ = feats(S, U, S2)
        o = torch.optim.Adam(fm.parameters(), lr=lr); lf = nn.HuberLoss(delta=1.0)
        n = su_n.shape[0]; bs = min(512, n); fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                o.zero_grad(); lf(fm(su_n[idx]), dy_n[idx]).backward(); o.step()
        fm.eval(); return fm

    def r2_from_predn(predn, S, S2, C, dims):
        pred = (predn.detach() * sy + my).cpu().numpy()
        yt = (S2 - S).astype(np.float32)
        if C is not None:
            m = ~C; pred, yt = pred[m], yt[m]
        if dims is not None:
            pred, yt = pred[:, dims], yt[:, dims]
        ssr = ((yt - pred) ** 2).sum(); sst = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    print("\n[3] per-test-task oracle ceilings (plain f(s,u)) ...", flush=True)
    oracle_r2 = []
    for i, tb in enumerate(test_bufs):
        fm_o = train_plain_fm(*tb["oracle"][:3], cfg["fm_epochs"], cfg["fm_lr"])
        su_n, _, _ = feats(*tb["hold"][:3])
        r2 = r2_from_predn(fm_o(su_n), tb["hold"][0], tb["hold"][2], tb["hold"][3], VEL_DIMS)
        oracle_r2.append(r2)
        print(f"    task {i} param={test_p[i]:.3f}: oracle VEL R^2={r2:.4f}", flush=True)
    oracle_mean = float(np.mean(oracle_r2))

    # ===================================================================== #
    # 5. ADAPTATION SWEEP: context (z=E(first N)) vs z0 (severed) on held-out
    # ===================================================================== #
    print(f"\n[4] context-adaptation sweep (budgets={budgets}) ...", flush=True)
    zero_z = torch.zeros(d, device=device)

    def eval_arm(use_context):
        r2_pt = np.full((len(test_bufs), len(budgets)), np.nan)
        for ti, tb in enumerate(test_bufs):
            actx_su, actx_dy, actx_cn = feats(*tb["adapt"][:3])   # context source (adapt pool)
            hsu, _, _ = feats(*tb["hold"][:3])
            for bi, N in enumerate(budgets):
                if (not use_context) or N == 0:
                    z = zero_z
                else:
                    with torch.no_grad():
                        z = enc(actx_cn[:N].unsqueeze(0))[0]
                with torch.no_grad():
                    predn = cfm_pred_n(hsu, z)
                r2_pt[ti, bi] = r2_from_predn(predn, tb["hold"][0], tb["hold"][2], tb["hold"][3], VEL_DIMS)
        return r2_pt

    ctx_pt = eval_arm(True)
    z0_pt = eval_arm(False)
    with np.errstate(invalid="ignore"):
        ctx_med = np.nanmedian(ctx_pt, 0).tolist()
        z0_med = np.nanmedian(z0_pt, 0).tolist()
        gap_med = np.nanmedian(ctx_pt - z0_pt, 0).tolist()
    thr = oracle_mean * cfg["adapt_frac"]
    ctx_adaptN = next((N for N, v in zip(budgets, ctx_med) if N > 0 and v >= thr), None)
    z0_adaptN = next((N for N, v in zip(budgets, z0_med) if N > 0 and v >= thr), None)
    print("    context VEL R^2 (median): " + "  ".join(f"N{N}={v:.3f}" for N, v in zip(budgets, ctx_med)) +
          f"  -> adapt@{ctx_adaptN}", flush=True)
    print("    z0(severed)  R^2 (median): " + "  ".join(f"N{N}={v:.3f}" for N, v in zip(budgets, z0_med)) +
          f"  -> adapt@{z0_adaptN}", flush=True)
    print("    context - z0 gap: " + "  ".join(f"N{N}={g:+.3f}" for N, g in zip(budgets, gap_med)), flush=True)

    # ===================================================================== #
    # 6. z -> param LINEAR PROBE (system-ID): fit on TRAIN-task z, test on TEST-task z.
    #    Multiple random context draws per task -> robust probe. z inferred at N_probe.
    # ===================================================================== #
    print(f"\n[5] z->param probe (N_probe={cfg['n_probe']}, draws={cfg['probe_draws']}) ...", flush=True)
    Np, ndraw = cfg["n_probe"], cfg["probe_draws"]
    prng = np.random.default_rng(cfg["seed"] + 9)

    def gather_z(feat_list, params):
        Z, Y = [], []
        for tid, (su_n, dy_n, cn) in enumerate(feat_list):
            n = cn.shape[0]
            for _ in range(ndraw):
                idx = torch.tensor(prng.integers(0, n, size=Np), device=device)
                with torch.no_grad():
                    z = enc(cn[idx].unsqueeze(0))[0].cpu().numpy()
                Z.append(z); Y.append(params[tid])
        return np.asarray(Z, np.float64), np.asarray(Y, np.float64)

    te_feat = [feats(*tb["adapt"][:3]) for tb in test_bufs]
    Ztr, Ytr = gather_z(tr_feat, train_p)
    Zte, Yte = gather_z(te_feat, test_p)
    # closed-form ridge linear probe z -> param (bias-augmented)
    lam = cfg["probe_ridge"]
    Xtr = np.concatenate([Ztr, np.ones((len(Ztr), 1))], 1)
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    Xte = np.concatenate([Zte, np.ones((len(Zte), 1))], 1)
    Ypred = Xte @ W
    ss_res = float(((Yte - Ypred) ** 2).sum()); ss_tot = float(((Yte - Yte.mean()) ** 2).sum())
    probe_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    # per-test-task predicted-vs-true (median predicted param per task)
    per_task_pred = []
    for ti in range(len(test_p)):
        sl = slice(ti * ndraw, (ti + 1) * ndraw)
        per_task_pred.append(float(np.median(Ypred[sl])))
    print(f"    z->param linear probe: test R^2={probe_r2:.4f}  "
          f"(chance=0; z DECODES the task param => system-ID)", flush=True)
    for ti in range(len(test_p)):
        print(f"      test task {ti}: true={test_p[ti]:+.3f}  z-decoded={per_task_pred[ti]:+.3f}", flush=True)

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    results = {
        "config": cfg, "family": family, "conflict": cfg["conflict"],
        "train_params": train_p.tolist(), "test_params": test_p.tolist(),
        "budgets": list(budgets), "oracle_r2_per_task": oracle_r2, "oracle_mean": oracle_mean,
        "adapt_threshold": thr,
        "context_r2_median": ctx_med, "z0_r2_median": z0_med, "context_minus_z0_gap": gap_med,
        "context_adaptN": ctx_adaptN, "z0_adaptN": z0_adaptN,
        "probe_r2": probe_r2, "probe_true": test_p.tolist(), "probe_pred": per_task_pred,
        "probe_draws_true": Yte.tolist(), "probe_draws_pred": Ypred.tolist(),
    }
    print(f"\n===== META-CONTEXT SUMMARY (family={family}, conflict={cfg['conflict']:.3f}) =====",
          flush=True)
    print(f"oracle mean VEL R^2={oracle_mean:.4f} | context adapt@{ctx_adaptN} z0 adapt@{z0_adaptN} "
          f"| peak context-z0 gap={max(gap_med):+.3f} | z->param probe R^2={probe_r2:.3f}", flush=True)

    figures = _make_context_figures(results)
    outdir = os.path.join(DATA_DIR, "meta_context", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_context_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figs = {}
    budgets = R["budgets"]; xN = [max(N, 1) for N in budgets]

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: context vs z0 adaptation curve ----
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(xN, R["context_r2_median"], "o-", color="#3d6fd1", lw=2.2,
            label=f"context  z=E(N)  (adapt@{R['context_adaptN']})")
    ax.plot(xN, R["z0_r2_median"], "s--", color="#d1603d", lw=2.0,
            label=f"z0 (context severed = pooled)  (adapt@{R['z0_adaptN']})")
    ax.axhline(R["oracle_mean"], color="#2f9e44", lw=1.3, label=f"oracle ceiling {R['oracle_mean']:.3f}")
    ax.axhline(R["adapt_threshold"], color="#2f9e44", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("context transitions N fed to the encoder  (N=0 = z=0)")
    ax.set_ylabel("held-out velocity-dim Δs R^2")
    ax.set_title(f"Amortized context adaptation (family={R['family']}, Φ={R['conflict']:.2f})\n"
                 "z from N transitions vs context-severed z=0", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    figs["fig1_context_curve.png"] = _save(fig)

    # ---- fig2: context - z0 gap ----
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.axhline(0, color="#444", lw=1.1)
    ax.plot(xN, R["context_minus_z0_gap"], "o-", color="#3d6fd1", lw=2.3)
    ax.set_xscale("log")
    ax.set_xlabel("context transitions N")
    ax.set_ylabel("R^2(context z) − R^2(z=0)")
    ax.set_title("Does the context latent buy adaptation? (gap opens under conflict)", fontsize=10)
    figs["fig2_context_gap.png"] = _save(fig)

    # ---- fig3: z -> param probe (system-ID) scatter ----
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    dt = np.asarray(R["probe_draws_true"]); dp = np.asarray(R["probe_draws_pred"])
    ax.scatter(dt, dp, s=10, alpha=0.25, color="#7a3fb0", label="per context draw")
    ax.scatter(R["probe_true"], R["probe_pred"], s=70, color="#3d6fd1",
               edgecolor="k", zorder=3, label="per test task (median)")
    lo = min(dt.min(), dp.min()); hi = max(dt.max(), dp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel("true task parameter (φ / damping)")
    ax.set_ylabel("z-decoded parameter (linear probe)")
    ax.set_title(f"z decodes the task parameter → system-ID\nheld-out probe R^2 = {R['probe_r2']:.3f}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    figs["fig3_z_probe.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_context(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    frame_skip: int = 12,
    goal_range: float = 0.5,       # (kept for cfg parity; planning not used in this cut)
    gear: float = 10.0,
    arena_half: float = 2.0,
    family: str = "actuator",
    conflict: float = 1.5708,      # actuator half-range (rad); pi/2 = the clean regime
    base_damping: float = 1.0,
    damp_min: float = 0.05,
    damp_max: float = 2.0,
    n_train_tasks: int = 16,
    n_test_tasks: int = 6,
    train_task_n: int = 1500,
    test_task_n: int = 6000,
    n_adapt_pool: int = 512,
    n_holdout: int = 1500,
    v_explore: float = 1.2,
    collect_ep_len: int = 20,
    collect_pos_range: float = 1.0,
    # encoder + conditional FM
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    # meta-training (amortized)
    meta_steps: int = 4000,
    meta_lr: float = 1e-3,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    # oracle (plain FM)
    fm_epochs: int = 40,
    fm_lr: float = 1e-3,
    adapt_frac: float = 0.9,
    # z -> param probe
    n_probe: int = 80,
    probe_draws: int = 20,
    probe_ridge: float = 1e-3,
):
    import os

    budgets = [0, 5, 10, 20, 40, 80, 160, 320]
    if quick:
        n_train_tasks, n_test_tasks = 6, 3
        train_task_n, test_task_n, n_adapt_pool, n_holdout = 600, 1600, 400, 500
        enc_hidden, fm_hidden, fm_layers = 64, 128, 2
        meta_steps = 800
        fm_epochs = 15
        budgets = [0, 20, 80, 320]
        probe_draws = 10
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, frame_skip=frame_skip, goal_range=goal_range,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0, gear=gear),
        family=family, conflict=conflict, base_damping=base_damping,
        damp_min=damp_min, damp_max=damp_max,
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks,
        train_task_N=train_task_n, test_task_N=test_task_n,
        n_adapt_pool=n_adapt_pool, n_holdout=n_holdout,
        v_explore=v_explore, collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        latent_dim=latent_dim, enc_hidden=enc_hidden, fm_hidden=fm_hidden, fm_layers=fm_layers,
        meta_steps=meta_steps, meta_lr=meta_lr, task_batch=task_batch, pred_P=pred_p,
        grad_clip=grad_clip, fm_epochs=fm_epochs, fm_lr=fm_lr, adapt_frac=adapt_frac,
        n_probe=n_probe, probe_draws=probe_draws, probe_ridge=probe_ridge,
        budgets=budgets,
    )
    out = run_meta_context.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_context_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
