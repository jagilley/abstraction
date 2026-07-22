"""Cut #4c — Value-directed identification: value (of information) in the loop.

Program: `ideas/two_timescale_value_loop.md` (§"The interface shape": the two afferent
taps — EXPLORE = value off the residual; here, epistemic value = which action best
disambiguates the task). Parent: `META_ADAPT_README.md` Cut #4b (the explicit context
latent z decodes the task parameter). This is the FIRST cut with VALUE in the loop.

    THE QUESTION. Now that z is an explicit task-identifier, does a VALUE-DIRECTED
    (value-of-information) exploration policy IDENTIFY the task (infer a good z) in FEWER
    transitions than passive OU collection? The value here is EPISTEMIC: pick the command
    whose outcome most DISAGREES across a codebook of task-hypotheses {z_k} = the training
    tasks' latents (which SPAN THE REAL TASK MANIFOLD — the fix for DIRECTED_READAPT's
    disagreement being blind to a confident-prior ensemble). VoI redeemed as a policy over
    an explicit latent.

DESIGN. Train the encoder E + conditional FM f(s,u,z) (Cut #4b). Build a z-CODEBOOK
{z_k = E(large context of train task k)}. At each held-out test task compare TWO ways to
collect N reward-free transitions:
  * PASSIVE  : central OU random commands (Cut #3's collect).
  * DIRECTED : at each step score M candidate commands by disagreement =
               Var_k[ f_normalized(s, u, z_k) ] (mean over out dims), execute argmax (+eps).
Then INFER z by OPTIMIZATION (fit z to the first N collected transitions via the frozen
f — robust to how the context was collected, so the comparison isolates COLLECTION
informativeness) and measure identification quality vs N: held-out FM R^2 (primary) +
z->param decode error (a linear probe, secondary).

NOTE ON SCARCITY (the DIRECTED_READAPT lesson). A GLOBAL rotation is information-rich
(every transition reveals phi), so passive already IDs fast — directed can only win by
picking more informative COMMANDS. `--rot-patch` gates the rotation to a Gaussian patch
(spatial info-scarcity): only in-patch transitions carry phi, so passive under-samples and
directed (which navigates to where the hypotheses disagree = the patch) should win bigger.

Run:
    cd experiments/
    modal run mjc/meta_adapt/meta_active.py::meta_active --quick
    modal run mjc/meta_adapt/meta_active.py::meta_active --tag act_p2 --conflict 1.5708
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_meta_active(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]; budgets = cfg["budgets"]; d = cfg["latent_dim"]
    VEL_DIMS = [2, 3]
    Phi = cfg["conflict"]
    n_tr, n_te = cfg["n_train_tasks"], cfg["n_test_tasks"]
    all_p = np.sort(np.linspace(-Phi, Phi, n_tr + n_te))
    te_idx = np.unique(np.linspace(1, n_tr + n_te - 2, n_te).round().astype(int))
    tr_idx = np.array([i for i in range(n_tr + n_te) if i not in te_idx])
    train_p, test_p = all_p[tr_idx].astype(np.float64), all_p[te_idx].astype(np.float64)
    print(f"[family] actuator rot | train {np.round(train_p,3)}\n[family] test {np.round(test_p,3)}",
          flush=True)

    def task_dgp(p):
        dd = dict(cfg["dgp_base"]); dd["joint_damping"] = cfg["base_damping"]; dd["push_rot"] = float(p)
        if cfg.get("rot_patch"):
            dd["rot_patch"] = cfg["rot_patch"]     # gate the rotation to a Gaussian patch
        return dd

    # ---- central OU collection (passive) ----------------------------------- #
    def collect(env, n_transitions, seed, center=None, pr_override=None):
        rng = np.random.default_rng(seed)
        ep_len = cfg["collect_ep_len"]
        pr = cfg["collect_pos_range"] if pr_override is None else pr_override
        c = np.zeros(2) if center is None else np.asarray(center, np.float64)
        S, U, S2, C = [], [], [], []
        for _ in range(int(np.ceil(n_transitions / ep_len))):
            env.set_state(c + rng.uniform(-pr, pr, 2), rng.normal(0, cfg["v_explore"], 2))
            a = np.zeros(2)
            for _ in range(ep_len):
                s = env.get_state()
                a = a - 0.2 * a + 0.5 * rng.normal(size=2)
                u = np.clip(a, -1, 1)
                s2, info = env.step(u, fs)
                S.append(s); U.append(u.astype(np.float32)); S2.append(s2); C.append(info["any_contact"])
        sl = slice(0, n_transitions)
        return (np.asarray(S, np.float32)[sl], np.asarray(U, np.float32)[sl],
                np.asarray(S2, np.float32)[sl], np.asarray(C, bool)[sl])

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    patch = cfg.get("rot_patch")
    preg = cfg["patch_reg_mult"]

    def in_patch(S):
        if patch is None:
            return np.ones(len(S), bool)
        c = np.asarray(patch["center"], np.float32)
        return np.linalg.norm(np.asarray(S)[:, :2] - c, axis=1) < preg * patch["sigma"]

    print("\n[1] collecting per-task buffers ...", flush=True)
    train_bufs = [collect(PusherEnv(task_dgp(p), with_puck=False), cfg["train_task_N"],
                          cfg["seed"] + 1000 + i) for i, p in enumerate(train_p)]
    nH = cfg["n_holdout"]
    test_bufs, test_envs = [], []
    for i, p in enumerate(test_p):
        env = PusherEnv(task_dgp(p), with_puck=False); test_envs.append(env)
        if patch is not None:
            # BALANCED eval pool: half reset ON the patch, half central -> enough in-patch
            # transitions for a well-estimated IN-PATCH R^2 (the sensitive metric — only
            # in-patch dynamics depend on phi; out-patch R^2 is a z-independent control).
            Si, Ui, S2i, Ci = collect(env, nH, cfg["seed"] + 2000 + i,
                                      center=patch["center"], pr_override=patch["sigma"])
            So, Uo, S2o, Co = collect(env, nH, cfg["seed"] + 2500 + i)
            S = np.concatenate([Si, So]); U = np.concatenate([Ui, Uo])
            S2 = np.concatenate([S2i, S2o]); C = np.concatenate([Ci, Co])
        else:
            S, U, S2, C = collect(env, nH, cfg["seed"] + 2000 + i)
        ff = ~C
        test_bufs.append({"hold": (S, U, S2, C), "sel_all": ff,
                          "sel_in": in_patch(S) & ff, "sel_out": (~in_patch(S)) & ff})
    allS = np.concatenate([b[0] for b in train_bufs]); allU = np.concatenate([b[1] for b in train_bufs])
    allS2 = np.concatenate([b[2] for b in train_bufs])
    norm = make_norm(allS, allU, allS2); mx, sx, my, sy = norm["mx"], norm["sx"], norm["my"], norm["sy"]

    def feats(S, U, S2):
        X = torch.tensor(np.concatenate([S, U], 1), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        su_n = (X - mx) / sx; dy_n = (Y - my) / sy
        return su_n, dy_n, torch.cat([su_n, dy_n], 1)

    tr_feat = [feats(b[0], b[1], b[2]) for b in train_bufs]

    # ---- encoder + conditional FM (Cut #4b) ------------------------------- #
    eh, fh, fl = cfg["enc_hidden"], cfg["fm_hidden"], cfg["fm_layers"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(10, eh), nn.SiLU(), nn.Linear(eh, eh), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(eh, eh), nn.SiLU(), nn.Linear(eh, d))

        def forward(self, ctx):
            return self.rho(self.phi(ctx).mean(dim=1))

    def build_cfm():
        lyr = [nn.Linear(6 + d, fh), nn.SiLU()]
        for _ in range(fl - 1):
            lyr += [nn.Linear(fh, fh), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(fh, 4)])).to(device)

    enc, cfm = Encoder().to(device), build_cfm()
    print(f"\n[2] meta-training encoder + conditional FM ({cfg['meta_steps']} steps) ...", flush=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(cfm.parameters()), lr=cfg["meta_lr"])
    lossf = nn.HuberLoss(delta=1.0); rng = np.random.default_rng(cfg["seed"] + 5)
    ctx_Ns = [b for b in budgets if b > 0]
    Bt, P = min(cfg["task_batch"], len(tr_feat)), cfg["pred_P"]
    enc.train(); cfm.train()
    for step in range(cfg["meta_steps"]):
        K = int(rng.choice(ctx_Ns))
        tids = rng.choice(len(tr_feat), size=Bt, replace=False)
        ctx = torch.empty(Bt, K, 10, device=device); psu = torch.empty(Bt, P, 6, device=device)
        pdy = torch.empty(Bt, P, 4, device=device)
        for j, tid in enumerate(tids):
            su_n, dy_n, cn = tr_feat[tid]; n = su_n.shape[0]
            ci = torch.tensor(rng.integers(0, n, size=K), device=device)
            pi = torch.tensor(rng.integers(0, n, size=P), device=device)
            ctx[j] = cn[ci]; psu[j] = su_n[pi]; pdy[j] = dy_n[pi]
        z = enc(ctx); zc = z.unsqueeze(1).expand(-1, P, -1)
        loss = lossf(cfm(torch.cat([psu, zc], 2)), pdy)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cfm.parameters()), cfg["grad_clip"])
        opt.step()
        if (step + 1) % max(1, cfg["meta_steps"] // 6) == 0:
            print(f"    [meta] step {step+1}/{cfg['meta_steps']}  huber={loss.item():.4f}", flush=True)
    enc.eval(); cfm.eval()

    # ---- z-codebook {z_k} from train tasks (typical large contexts -> in-dist for E) ---- #
    with torch.no_grad():
        Zcode = torch.stack([enc(tr_feat[k][2][:cfg["codebook_ctx"]].unsqueeze(0))[0]
                             for k in range(len(tr_feat))], 0)      # (n_tr, d)
    print(f"[2b] built z-codebook: {Zcode.shape[0]} task latents", flush=True)

    # ---- z inference by OPTIMIZATION (robust to collection distribution) ---- #
    def infer_z(su_n, dy_n):
        if su_n.shape[0] == 0:
            return torch.zeros(d, device=device)
        z = torch.zeros(d, device=device, requires_grad=True)
        o = torch.optim.Adam([z], lr=cfg["infer_lr"]); lf = nn.HuberLoss(delta=1.0)
        for _ in range(cfg["infer_steps"]):
            pred = cfm(torch.cat([su_n, z.unsqueeze(0).expand(su_n.shape[0], -1)], 1))
            loss = lf(pred, dy_n) + cfg["infer_l2"] * (z * z).mean()
            o.zero_grad(); loss.backward(); o.step()
        return z.detach()

    def fm_r2(su_n, z, S, S2, sel, dims=VEL_DIMS):
        with torch.no_grad():
            predn = cfm(torch.cat([su_n, z.unsqueeze(0).expand(su_n.shape[0], -1)], 1))
        pred = (predn * sy + my).cpu().numpy(); yt = (S2 - S).astype(np.float32)
        pred, yt = pred[sel][:, dims], yt[sel][:, dims]
        if len(pred) < 5:
            return float("nan")
        ssr = ((yt - pred) ** 2).sum(); sst = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ssr / (sst + 1e-12))

    # ---- DIRECTED collection ---------------------------------------------- #
    #   mode="voi"       : max codebook-disagreement command (the value-of-information drive)
    #   mode="magnitude" : max |command| (the control — is the win genuine VoI or just big u?)
    def directed_collect(env, n_transitions, seed, mode="voi"):
        rng = np.random.default_rng(seed)
        ep_len, pr, M, eps = cfg["collect_ep_len"], cfg["collect_pos_range"], cfg["dir_cand"], cfg["dir_eps"]
        S, U, S2, C = [], [], [], []
        while len(S) < n_transitions:
            state = np.concatenate([rng.uniform(-pr, pr, 2), rng.normal(0, cfg["v_explore"], 2)]).astype(np.float32)
            for _ in range(ep_len):
                if len(S) >= n_transitions:
                    break
                cand = rng.uniform(-1, 1, (M, 2)).astype(np.float32)
                if mode == "magnitude":
                    score = (cand ** 2).sum(1)                        # (M,) |u|^2
                else:
                    srep = np.repeat(state[None, :], M, 0)
                    X = torch.tensor(np.concatenate([srep, cand], 1), device=device)
                    Xn = (X - mx) / sx                                # (M,6)
                    with torch.no_grad():
                        # predicted normalized Δŝ under each codebook hypothesis -> disagreement
                        preds = torch.stack([cfm(torch.cat([Xn, zk.unsqueeze(0).expand(M, -1)], 1))
                                             for zk in Zcode], 0)      # (n_tr, M, 4)
                        score = preds.var(0).mean(1).cpu().numpy()    # (M,)
                b = int(rng.integers(0, M)) if rng.random() < eps else int(score.argmax())
                u = cand[b]
                env.set_state(state[:2], state[2:]); s2, info = env.step(u, fs)
                S.append(state.copy()); U.append(u.astype(np.float32)); S2.append(s2)
                C.append(bool(info["any_contact"])); state = s2
        return (np.asarray(S, np.float32), np.asarray(U, np.float32),
                np.asarray(S2, np.float32), np.asarray(C, bool))

    # ---- NAVIGATING info-directed MPC: plan to REACH high-disagreement states -- #
    # The myopic drive reads disagreement only at the current state, so it can't
    # navigate to a scarce informative region. This CEM planner maximizes CUMULATIVE
    # codebook-disagreement over a rolled-out trajectory (rolled with the mean-codebook
    # z_ref, valid because position dynamics are task-independent up to the patch), so
    # it plans a PATH to the patch and disambiguates there.
    z_ref = Zcode.mean(0)

    def info_cem(state, rng):
        Hp, Kc, it, el = cfg["info_horizon"], cfg["info_K"], cfg["info_iters"], cfg["info_elite"]
        mu = np.zeros((Hp, 2), np.float32); sig = np.full((Hp, 2), cfg["info_init_sigma"], np.float32)
        s0 = torch.tensor(np.tile(state.astype(np.float32), (Kc, 1)), device=device)
        for _ in range(it):
            e = rng.standard_normal((Kc, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[None] + sig[None] * e, -1, 1)                # (Kc,Hp,2)
            seqs_t = torch.tensor(seqs, device=device)
            with torch.no_grad():
                s = s0.clone(); total = torch.zeros(Kc, device=device)
                for h in range(Hp):
                    Xn = (torch.cat([s, seqs_t[:, h, :]], 1) - mx) / sx    # (Kc,6)
                    preds = torch.stack([cfm(torch.cat([Xn, zk.unsqueeze(0).expand(Kc, -1)], 1))
                                         for zk in Zcode], 0)               # (n_tr,Kc,4)
                    total = total + preds.var(0).mean(1)                    # accumulate disagreement
                    s = s + (cfm(torch.cat([Xn, z_ref.unsqueeze(0).expand(Kc, -1)], 1)) * sy + my)
                idx = torch.topk(total, el).indices.cpu().numpy()
            mu = seqs[idx].mean(0); sig = seqs[idx].std(0) + 1e-3
        return mu.astype(np.float32)                                       # (Hp,2)

    def infompc_collect(env, n_transitions, seed):
        rng = np.random.default_rng(seed)
        ep_len, pr, re = cfg["collect_ep_len"], cfg["collect_pos_range"], cfg["info_replan"]
        S, U, S2, C = [], [], [], []
        while len(S) < n_transitions:
            state = np.concatenate([rng.uniform(-pr, pr, 2), rng.normal(0, cfg["v_explore"], 2)]).astype(np.float32)
            plan = None
            for t in range(ep_len):
                if len(S) >= n_transitions:
                    break
                if t % re == 0:
                    plan = info_cem(state, rng)
                u = plan[t % re]
                env.set_state(state[:2], state[2:]); s2, info = env.step(u, fs)
                S.append(state.copy()); U.append(u.astype(np.float32)); S2.append(s2)
                C.append(bool(info["any_contact"])); state = s2
        return (np.asarray(S, np.float32), np.asarray(U, np.float32),
                np.asarray(S2, np.float32), np.asarray(C, bool))

    # ===================================================================== #
    # 3. per-test-task: collection arms -> identification quality vs N
    #    passive (OU) · directed (VoI = codebook disagreement) · magnitude (max|u| control)
    # ===================================================================== #
    ARMS = ["passive", "magnitude", "directed", "navigate"]

    def collect_arm(env, name, seed):
        if name == "passive":
            return collect(env, Nmax, seed)
        if name == "navigate":
            return infompc_collect(env, Nmax, seed)
        return directed_collect(env, Nmax, seed, mode=("voi" if name == "directed" else "magnitude"))

    Nmax = max(budgets)
    print(f"\n[3] identification: passive vs directed(VoI) vs magnitude (budgets={budgets}) ...", flush=True)
    # r2[metric][arm] : (n_test, n_budget). metric in {all, in, out}; in/out only for patch.
    metrics = ["all", "in", "out"] if patch is not None else ["all"]
    r2 = {m: {a: np.full((len(test_bufs), len(budgets)), np.nan) for a in ARMS} for m in metrics}
    zat = {a: {N: [] for N in budgets} for a in ARMS}
    vis = {a: [] for a in ARMS}                          # patch visitation (mechanism check)
    for ti, tb in enumerate(test_bufs):
        hsu, _, _ = feats(*tb["hold"][:3]); hS, hS2 = tb["hold"][0], tb["hold"][2]
        sel = {"all": tb["sel_all"], "in": tb["sel_in"], "out": tb["sel_out"]}
        for ai, a in enumerate(ARMS):
            aS, aU, aS2, _ = collect_arm(test_envs[ti], a, cfg["seed"] + 5000 + 1000 * ai + ti)
            if patch is not None:
                vis[a].append(float(in_patch(aS).mean()))
            asu, ady, _ = feats(aS, aU, aS2)
            for bi, N in enumerate(budgets):
                z = infer_z(asu[:N], ady[:N])
                for m in metrics:
                    r2[m][a][ti, bi] = fm_r2(hsu, z, hS, hS2, sel[m])
                zat[a][N].append(z.cpu().numpy())
    vis_med = {a: (float(np.median(vis[a])) if vis[a] else None) for a in ARMS}

    head = "in" if patch is not None else "all"          # the sensitive metric for this run
    with np.errstate(invalid="ignore"):
        med = {m: {a: np.nanmedian(r2[m][a], 0).tolist() for a in ARMS} for m in metrics}
    thr = cfg["id_frac"] * max(med[head]["directed"][-1], med[head]["passive"][-1])
    idN = {a: next((N for N, v in zip(budgets, med[head][a]) if N > 0 and v >= thr), None) for a in ARMS}

    # ---- z->phi probe (fit on train-task optimization-z*, robust) ---------- #
    ztr = np.stack([infer_z(*tr_feat[k][:2]).cpu().numpy() for k in range(len(tr_feat))], 0)
    Xtr = np.concatenate([ztr, np.ones((len(ztr), 1))], 1)
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1]), Xtr.T @ train_p)

    def phi_err(zdict):
        errs = []
        for N in budgets:
            if N == 0:
                errs.append(float("nan")); continue
            Z = np.asarray(zdict[N]); pred = np.concatenate([Z, np.ones((len(Z), 1))], 1) @ W
            errs.append(float(np.median(np.abs(pred - test_p))))
        return errs
    err = {a: phi_err(zat[a]) for a in ARMS}

    for a in ARMS:
        print(f"    {a:9s} {head}-R^2: " + "  ".join(f"N{N}={v:.3f}" for N, v in zip(budgets, med[head][a]))
              + f"  -> id@{idN[a]}", flush=True)
    for a in ARMS:
        print(f"    {a:9s} phi|err|: " + "  ".join(f"N{N}={v:.3f}" for N, v in zip(budgets, err[a]) if v == v),
              flush=True)
    if patch is not None:
        print("    patch visitation: " + "  ".join(f"{a}={vis_med[a]:.3f}" for a in ARMS), flush=True)

    results = {
        "config": cfg, "conflict": Phi, "rot_patch": cfg.get("rot_patch"), "head_metric": head,
        "train_params": train_p.tolist(), "test_params": test_p.tolist(), "budgets": list(budgets),
        "r2_median": med, "phi_err": err, "idN": idN, "id_threshold": thr, "arms": ARMS,
        "patch_visitation": vis_med,
    }
    print(f"\n===== META-ACTIVE SUMMARY (conflict={Phi:.3f}, patch={patch is not None}) =====", flush=True)
    print(f"[{head}-R^2] id@: " + "  ".join(f"{a}={idN[a]}" for a in ARMS)
          + f"  | directed-passive peak gap="
          + f"{max(d - p for d, p in zip(med[head]['directed'], med[head]['passive'])):+.3f}", flush=True)

    figures = _make_active_figures(results)
    outdir = os.path.join(DATA_DIR, "meta_active", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_active_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"passive": "#888888", "directed": "#3d6fd1", "magnitude": "#d1603d", "navigate": "#2f9e44"}
    MRK = {"passive": "s--", "directed": "o--", "magnitude": "^--", "navigate": "D-"}
    LBL = {"passive": "passive (OU)", "directed": "myopic VoI", "magnitude": "magnitude (max|u|)",
           "navigate": "navigating VoI (info-MPC)"}
    figs = {}
    budgets = R["budgets"]; xN = [max(N, 1) for N in budgets]; arms = R["arms"]
    head = R["head_metric"]

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: head-metric R^2 (in-patch for patch mode; all-region for global) ----
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for a in arms:
        ax.plot(xN, R["r2_median"][head][a], MRK[a], color=COL[a], lw=2.2,
                label=f"{LBL[a]}  id@{R['idN'][a]}")
    ax.axhline(R["id_threshold"], color="#2f9e44", ls="--", lw=1.0, label="id threshold")
    ax.set_xscale("log")
    ax.set_xlabel("reward-free transitions N")
    ax.set_ylabel(f"{'IN-PATCH' if head == 'in' else 'held-out'} velocity-dim Δs R^2 (inferred z)")
    ax.set_title(f"Value-directed identification (Φ={R['conflict']:.2f}, patch={R['rot_patch'] is not None})\n"
                 f"{'in-patch R^2 = the sensitive metric (only there does φ matter)' if head == 'in' else 'does VoI identify the task in fewer transitions?'}",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    figs["fig1_identification_curve.png"] = _save(fig)

    # ---- fig2: task-parameter decode error ----
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for a in arms:
        e = R["phi_err"][a]
        xs = [x for x, v in zip(xN, e) if v == v]; ys = [v for v in e if v == v]
        ax.plot(xs, ys, MRK[a], color=COL[a], lw=2.2, label=LBL[a])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reward-free transitions N")
    ax.set_ylabel("|φ decoded − φ true|  (lower = better ID)")
    ax.set_title("Task-parameter identification error vs #transitions", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    figs["fig2_phi_error.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def meta_active(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    frame_skip: int = 12,
    gear: float = 10.0,
    arena_half: float = 2.0,
    conflict: float = 1.5708,
    base_damping: float = 1.0,
    rot_patch: bool = False,            # gate the rotation to a Gaussian patch (info-scarcity)
    patch_center_x: float = 0.65,
    patch_center_y: float = 0.0,
    patch_sigma: float = 0.30,
    patch_reg_mult: float = 1.5,
    n_train_tasks: int = 16,
    n_test_tasks: int = 6,
    train_task_n: int = 1500,
    test_task_n: int = 3000,
    n_holdout: int = 1500,
    v_explore: float = 1.2,
    collect_ep_len: int = 20,
    collect_pos_range: float = 1.0,
    latent_dim: int = 8,
    enc_hidden: int = 128,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    meta_steps: int = 4000,
    meta_lr: float = 1e-3,
    task_batch: int = 8,
    pred_p: int = 128,
    grad_clip: float = 1.0,
    codebook_ctx: int = 256,
    # z inference by optimization
    infer_steps: int = 150,
    infer_lr: float = 0.05,
    infer_l2: float = 1e-2,
    # directed policy
    dir_cand: int = 64,
    dir_eps: float = 0.1,
    id_frac: float = 0.9,
    # navigating info-MPC (plans to REACH high-disagreement states)
    info_horizon: int = 12,
    info_k: int = 128,
    info_iters: int = 3,
    info_elite: int = 16,
    info_init_sigma: float = 0.8,
    info_replan: int = 4,
):
    import os

    # global rotation is info-rich -> fine low-N budgets; the patch is scarce (only in-patch
    # transitions carry phi) -> passive under-samples, so budgets extend much higher.
    budgets = [0, 5, 10, 20, 50, 100, 200, 400] if rot_patch else [0, 1, 2, 3, 5, 8, 12, 20, 40]
    if quick:
        n_train_tasks, n_test_tasks = 6, 3
        train_task_n, test_task_n, n_holdout = 600, 1600, 500
        enc_hidden, fm_hidden, fm_layers = 64, 128, 2
        meta_steps = 800
        budgets = [0, 2, 5, 12, 40]
        infer_steps = 100
        info_horizon, info_k = 8, 48
        tag = tag or "smoke"
    tag = tag or "default"

    patch = (dict(center=[patch_center_x, patch_center_y], sigma=patch_sigma) if rot_patch else None)
    cfg = dict(
        tag=tag, seed=seed, frame_skip=frame_skip,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0, gear=gear),
        conflict=conflict, base_damping=base_damping, rot_patch=patch, patch_reg_mult=patch_reg_mult,
        n_train_tasks=n_train_tasks, n_test_tasks=n_test_tasks,
        train_task_N=train_task_n, test_task_N=test_task_n, n_holdout=n_holdout,
        v_explore=v_explore, collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        latent_dim=latent_dim, enc_hidden=enc_hidden, fm_hidden=fm_hidden, fm_layers=fm_layers,
        meta_steps=meta_steps, meta_lr=meta_lr, task_batch=task_batch, pred_P=pred_p,
        grad_clip=grad_clip, codebook_ctx=codebook_ctx,
        infer_steps=infer_steps, infer_lr=infer_lr, infer_l2=infer_l2,
        dir_cand=dir_cand, dir_eps=dir_eps, id_frac=id_frac, budgets=budgets,
        info_horizon=info_horizon, info_K=info_k, info_iters=info_iters, info_elite=info_elite,
        info_init_sigma=info_init_sigma, info_replan=info_replan,
    )
    out = run_meta_active.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "meta_active_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
