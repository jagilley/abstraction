"""Agency gate — the ACT-vs-PLAYBACK playback control (Gadagkar 2016 Fig 4 replication).

Program: `ideas/performance_error_is_the_bridge.md` §8 Experiment 2. The claim:
the bridge signal

    e_t  = ||o_{t+1} - FM2(s_t, a_t)||_V     (value-relevant slice of the arity-2 residual)
    b    = EWMA(e)                            (flexible performance benchmark)
    g_t  = ||FM2(s_t, a_t) - FM1(s_t)||      (arity gap = what my action explains)
    dlt  = (b - e) * gate(g)                  (performance error)

is gated on the EXISTENCE OF AN EFFERENCE COPY. Two conditions with IDENTICAL
sensory sequences:

  ACT      — the agent produced the trajectory; the efference-copy slot carries
             the true command u_t.
  PLAYBACK — the identical trajectory replayed as observation; the agent is
             passive, so the efference-copy slot carries motor silence (u = 0).

One fixed bridge computation; the ONLY difference between conditions is the
content of the action slot. This is the principled "no efference copy":
(i) zero is what the motor channel literally carries when passive; (ii) since
E[u] = 0 and free-flight dynamics are near-linear in u, FM2(s,0) ~ E_u[FM2(s,u)]
~ FM1(s) — the arity-2 machine without an efference copy degrades gracefully to
the arity-1 prediction, so g -> 0 in PLAYBACK is a property of the LEARNED
models, not true by construction. (The idea doc's literal `FM2 == FM1` variant,
where g = 0 definitionally, is computed as a secondary trivial readout.)

Because predictability is matched across conditions by construction, a positive
result cannot be the textbook "prediction error gates learning".

The load-bearing protocol constraint (arity_torque, the RHM generator confound):
commands are i.i.d. (theta=1, seek_gain=0), measured max|corr(u,s)| ~ 0. With a
deterministic policy FM1 recovers u from s and g collapses in ACT for reasons
unrelated to agency.

The Gadagkar readout: sensory DISTORTION EVENTS — one-step kicks to the
OBSERVED feedback o_{t+1} on the value slice (the distorted-auditory-feedback
analog; the physics is untouched, so the two conditions' sensory sequences stay
exactly identical) — at spaced free-flight target steps, 50% distorted.
Event-triggered dlt in ACT vs PLAYBACK, distorted vs undistorted, plus the
ungated (b - e) so the gate's contribution is dissectible, plus a
target-conditioned benchmark b_T (the "flexible performance benchmark") as the
omitted-distortion-activation secondary readout.

Gate calibration (transparent, no test-set tuning): gate(g) = sigmoid((g-g0)/th)
with g0 = midpoint of the passive/active g medians and th = their gap / 8, both
measured on the TRAINING split only. sigma(0) = 0.5, so the doc's raw
sigmoid(g/th) leaks 0.5*(b-e) in PLAYBACK; it is reported alongside as
`dlt_naive` to keep that visible.

Run:
    cd experiments/
    modal run mjc/agency_gate/agency_gate.py::agency_gate --quick
    modal run --detach mjc/agency_gate/agency_gate.py::run_agency_gate  # (use entrypoint below)
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# state-dim groups (pusher_env.STATE_LABELS, puck world)
PUSHER_VEL = [4, 5]   # directly actuated -> the value-relevant slice V
PUCK_VEL = [6, 7]
POS = [0, 1, 2, 3]


def _r2(y_true, y_pred, dims=None):
    yt = y_true if dims is None else y_true[:, dims]
    yp = y_pred if dims is None else y_pred[:, dims]
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _pctl(x, ps=(5, 25, 50, 75, 95)):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    d = {f"p{p:02d}": float(np.percentile(x, p)) for p in ps}
    d.update(mean=float(x.mean()), std=float(x.std()), rms=float(np.sqrt((x ** 2).mean())),
             n=int(len(x)))
    return d


def _auroc(pos, neg):
    """AUROC of pos > neg (rank-based, exact)."""
    import numpy as np
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    # handle ties crudely (fine at these separations)
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg) + 1e-12))


@app.function(cpu=8.0, memory=16384, timeout=5400, volumes={DATA_DIR: volume})
def run_agency_gate(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import collect_transitions, STATE_LABELS

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dgp = cfg["dgp"]
    V = PUSHER_VEL

    # ------------------------------------------------------------------ #
    # 1. training data: i.i.d. commands (u perp s) — the arity_torque idiom
    # ------------------------------------------------------------------ #
    print(f"[collect/train] {cfg['n_episodes']} eps x {cfg['ep_len']} (i.i.d. commands)",
          flush=True)
    data = collect_transitions(
        dgp=dgp, n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
        frame_skip=cfg["frame_skip"], seed=cfg["seed"],
        sigma=cfg["sigma"], theta=1.0, seek_gain=0.0,   # theta=1 => i.i.d.
    )
    S, U, S2, ep = data["S"], data["U"], data["S2"], data["ep"]
    contact = data["any_contact"]
    N = len(S)

    corr_train = float(np.max(np.abs([
        np.corrcoef(U[:, a], S[:, b])[0, 1]
        for a in range(2) for b in range(S.shape[1])])))
    print(f"[collect/train] N={N} contact_frac={contact.mean():.3f} "
          f"max|corr(u,s)|={corr_train:.4f} (want ~0)", flush=True)

    # split + normalize (arity_torque idiom: last 20% of episodes held out)
    n_ep = int(ep.max()) + 1
    test_eps = set(range(n_ep - max(1, int(round(0.2 * n_ep))), n_ep))
    te = np.array([e in test_eps for e in ep])
    tr = ~te

    Xa2 = np.concatenate([S, U], axis=1).astype(np.float32)
    Xa1 = S.astype(np.float32)
    Y = (S2 - S).astype(np.float32)

    def _norm(A):
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        return (A - mu) / sd, mu, sd

    Xa2n, mx2, sx2 = _norm(Xa2)
    Xa1n, mx1, sx1 = _norm(Xa1)
    Yn, my, sy = _norm(Y)

    # ------------------------------------------------------------------ #
    # 2. train FM1(s) and FM2(s,u) on the same transitions
    # ------------------------------------------------------------------ #
    def _train(Xn, label):
        din = Xn.shape[1]
        Xtr = torch.tensor(Xn[tr], device=device)
        Ytr = torch.tensor(Yn[tr], device=device)
        h, layers = cfg["hidden"], cfg["layers"]
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(layers - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, Y.shape[1])]
        net = nn.Sequential(*lyr).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        lossf = nn.HuberLoss(delta=1.0)
        ntr = Xtr.shape[0]
        bs = min(cfg["batch"], ntr)
        for _e in range(cfg["epochs"]):
            perm = torch.randperm(ntr, device=device)
            for i in range(0, ntr, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(net(Xtr[idx]), Ytr[idx]).backward()
                opt.step()
        net.eval()
        print(f"[train] {label} done ({sum(p.numel() for p in net.parameters())} params)",
              flush=True)
        return net

    net2 = _train(Xa2n, "FM2(s,u)")
    net1 = _train(Xa1n, "FM1(s)")

    def _predict(net, Xn_batch):
        with torch.no_grad():
            p = net(torch.tensor(Xn_batch, device=device)).cpu().numpy()
        return p * sy + my    # raw-unit delta prediction

    def fm2_delta(Sb, Ub):
        X = np.concatenate([Sb, Ub], axis=1).astype(np.float32)
        return _predict(net2, (X - mx2) / sx2)

    def fm1_delta(Sb):
        return _predict(net1, (Sb.astype(np.float32) - mx1) / sx1)

    # sanity: the arity gap must reproduce (held-out split, free flight)
    ff_te = ~contact[te]
    p2_te = fm2_delta(S[te], U[te])
    p1_te = fm1_delta(S[te])
    Yte = Y[te]
    sanity = {
        "r2_a2_pusher_vel": _r2(Yte[ff_te], p2_te[ff_te], V),
        "r2_a1_pusher_vel": _r2(Yte[ff_te], p1_te[ff_te], V),
        "r2_a2_all": _r2(Yte[ff_te], p2_te[ff_te]),
        "r2_a1_all": _r2(Yte[ff_te], p1_te[ff_te]),
    }
    print(f"[sanity] free-flight R2(pusher-vel): arity-2={sanity['r2_a2_pusher_vel']:.4f} "
          f"arity-1={sanity['r2_a1_pusher_vel']:.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. gate calibration — TRAINING split only, fully reported
    # ------------------------------------------------------------------ #
    Str, Utr_ = S[tr], U[tr]
    p2_act = fm2_delta(Str, Utr_)
    p2_pas = fm2_delta(Str, np.zeros_like(Utr_))
    p1_tr = fm1_delta(Str)
    g_cal_active = np.linalg.norm(p2_act - p1_tr, axis=1)
    g_cal_passive = np.linalg.norm(p2_pas - p1_tr, axis=1)
    m_a = float(np.median(g_cal_active))
    m_p = float(np.median(g_cal_passive))
    g0 = 0.5 * (m_a + m_p)
    theta = max((m_a - m_p) / 8.0, 1e-9)
    calib_degenerate = bool(m_a <= m_p)
    calib = {
        "g_cal_active": _pctl(g_cal_active), "g_cal_passive": _pctl(g_cal_passive),
        "median_active": m_a, "median_passive": m_p,
        "g0": g0, "theta": theta, "degenerate": calib_degenerate,
        "auroc_active_vs_passive_train": _auroc(g_cal_active, g_cal_passive),
    }
    print(f"[calib] g medians: active={m_a:.5f} passive={m_p:.5f} -> g0={g0:.5f} "
          f"theta={theta:.6f} (degenerate={calib_degenerate})", flush=True)

    def gate(g):
        return 1.0 / (1.0 + np.exp(-(g - g0) / theta))

    def gate_naive(g):          # the doc's literal sigma(g/theta): sigma(0)=0.5 leak
        return 1.0 / (1.0 + np.exp(-g / theta))

    # distortion magnitude: 1x the natural per-step value-slice signal scale
    dv_rms = float(np.sqrt((np.linalg.norm(Y[tr & ~contact][:, V], axis=1) ** 2).mean()))
    d_mag = cfg["d_mag_scale"] * dv_rms
    print(f"[distort] free-flight ||dv_pusher|| rms={dv_rms:.5f} -> d_mag={d_mag:.5f}",
          flush=True)

    # ------------------------------------------------------------------ #
    # 4. eval trajectories (held-out seed, same i.i.d. protocol)
    # ------------------------------------------------------------------ #
    print(f"[collect/eval] {cfg['n_eval_episodes']} eps x {cfg['ep_len']}", flush=True)
    ev = collect_transitions(
        dgp=dgp, n_episodes=cfg["n_eval_episodes"], ep_len=cfg["ep_len"],
        frame_skip=cfg["frame_skip"], seed=cfg["seed"] + 1000,
        sigma=cfg["sigma"], theta=1.0, seek_gain=0.0,
    )
    Se, Ue, S2e, con_e = ev["S"], ev["U"], ev["S2"], ev["any_contact"]
    corr_eval = float(np.max(np.abs([
        np.corrcoef(Ue[:, a], Se[:, b])[0, 1]
        for a in range(2) for b in range(Se.shape[1])])))
    n_ev, T = cfg["n_eval_episodes"], cfg["ep_len"]
    print(f"[collect/eval] contact_frac={con_e.mean():.3f} max|corr(u,s)|={corr_eval:.4f}",
          flush=True)

    # target steps + distortion coin flips (sensory-only; physics untouched)
    rng = np.random.default_rng(cfg["seed"] + 7)
    burn, tev, w = cfg["burn_in"], cfg["target_every"], cfg["event_window"]
    con_ep = con_e.reshape(n_ev, T)
    events = []   # (ep, t, distorted)
    D = np.zeros((n_ev, T, 2), np.float32)   # additive kick on observed pusher-vel
    for e_i in range(n_ev):
        for t in range(burn, T - w - 1, tev):
            if con_ep[e_i, t]:
                continue          # keep the distorted transition itself free-flight
            distorted = bool(rng.random() < cfg["distort_prob"])
            if distorted:
                ang = rng.uniform(0, 2 * np.pi)
                D[e_i, t] = d_mag * np.array([np.cos(ang), np.sin(ang)], np.float32)
            events.append((e_i, t, distorted))
    n_dist = sum(1 for _, _, d in events if d)
    print(f"[events] {len(events)} target steps ({n_dist} distorted, "
          f"{len(events) - n_dist} undistorted)", flush=True)

    # observed feedback o_{t+1} = s_{t+1} + kick (value-slice only, one step)
    O2 = S2e.reshape(n_ev, T, -1).copy()
    O2[:, :, V[0]] += D[:, :, 0]
    O2[:, :, V[1]] += D[:, :, 1]
    O2 = O2.reshape(n_ev * T, -1)

    # ------------------------------------------------------------------ #
    # 5. the bridge, per condition — ONE computation, action slot differs
    # ------------------------------------------------------------------ #
    p1_ev = fm1_delta(Se)

    def run_bridge(a_slot, fm1_only=False):
        """a_slot: (N,2) content of the efference-copy slot. Returns per-step dict."""
        if fm1_only:
            p2 = p1_ev
        else:
            p2 = fm2_delta(Se, a_slot)
        dY = O2 - Se                                        # observed delta (w/ kick)
        e = np.linalg.norm(dY[:, V] - p2[:, V], axis=1)     # value-relevant slice
        g = np.linalg.norm(p2 - p1_ev, axis=1)              # arity gap (full state)
        g_v = np.linalg.norm(p2[:, V] - p1_ev[:, V], axis=1)
        e_ep = e.reshape(n_ev, T)
        b = np.zeros_like(e_ep)
        for e_i in range(n_ev):
            bt = float(np.median(e_ep[e_i, :10]))           # per-episode init
            for t in range(T):
                b[e_i, t] = bt                              # b_t BEFORE seeing e_t
                bt = (1 - cfg["alpha"]) * bt + cfg["alpha"] * e_ep[e_i, t]
        b = b.reshape(-1)
        adv = b - e                                         # ungated performance error
        gt = gate(g)
        dlt = adv * gt
        dlt_naive = adv * gate_naive(g)
        return dict(e=e, g=g, g_v=g_v, b=b, adv=adv, gate=gt,
                    dlt=dlt, dlt_naive=dlt_naive)

    act = run_bridge(Ue)                                     # efference copy present
    pb = run_bridge(np.zeros_like(Ue))                       # motor silence
    pb1 = run_bridge(None, fm1_only=True)                    # doc's FM2==FM1 variant

    # ------------------------------------------------------------------ #
    # 6. stats
    # ------------------------------------------------------------------ #
    ff = ~con_e
    stat_mask = ff.copy().reshape(n_ev, T)
    stat_mask[:, :burn] = False                              # EWMA warmup excluded
    stat_mask = stat_mask.reshape(-1)

    def summarize(br):
        return {k: _pctl(br[k][stat_mask]) for k in
                ("e", "g", "g_v", "b", "adv", "gate", "dlt", "dlt_naive")}

    per_step = {"ACT": summarize(act), "PLAYBACK": summarize(pb),
                "PLAYBACK_fm1only": summarize(pb1)}

    # event-triggered traces (aligned windows, within-episode)
    offs = np.arange(-w, w + 1)

    def event_traces(br, key):
        x = br[key].reshape(n_ev, T)
        out = {}
        for name, want in (("distorted", True), ("undistorted", False)):
            tr_list = [x[e_i, t - w:t + w + 1] for (e_i, t, d) in events if d is want]
            A = np.asarray(tr_list)
            out[name] = dict(mean=A.mean(0), sem=A.std(0) / np.sqrt(len(A)), n=len(A))
        return out

    ev_tr = {c: {k: event_traces(br, k) for k in ("dlt", "adv", "e")}
             for c, br in (("ACT", act), ("PLAYBACK", pb))}

    def at0(c, k, name):
        t = ev_tr[c][k][name]
        return {"mean": float(t["mean"][w]), "sem": float(t["sem"][w]), "n": t["n"]}

    event_stats = {c: {k: {nm: at0(c, k, nm) for nm in ("distorted", "undistorted")}
                       for k in ("dlt", "adv", "e")} for c in ("ACT", "PLAYBACK")}

    # target-conditioned benchmark b_T (the flexible performance benchmark):
    # EWMA over TARGET-STEP e only, chronological across episodes, per condition.
    def target_benchmark(br):
        e_ep = br["e"].reshape(n_ev, T)
        g_ep = br["gate"].reshape(n_ev, T)
        rows = []
        bT = None
        for i, (e_i, t, d) in enumerate(events):
            ev_e = float(e_ep[e_i, t])
            if bT is None:
                bT = ev_e
            rows.append((i, d, (bT - ev_e) * float(g_ep[e_i, t]), bT - ev_e))
            bT = (1 - cfg["alpha_T"]) * bT + cfg["alpha_T"] * ev_e
        rows = rows[cfg["bT_warmup"]:]
        out = {}
        for name, want in (("distorted", True), ("undistorted", False)):
            dl = np.array([r[2] for r in rows if r[1] is want])
            ug = np.array([r[3] for r in rows if r[1] is want])
            out[name] = dict(dlt_T_mean=float(dl.mean()),
                             dlt_T_sem=float(dl.std() / np.sqrt(len(dl))),
                             adv_T_mean=float(ug.mean()),
                             adv_T_sem=float(ug.std() / np.sqrt(len(ug))), n=len(dl))
        return out

    bT_stats = {"ACT": target_benchmark(act), "PLAYBACK": target_benchmark(pb)}

    headline = {
        "auroc_g_act_vs_pb": _auroc(act["g"][stat_mask], pb["g"][stat_mask]),
        "gate_median_ACT": per_step["ACT"]["gate"]["p50"],
        "gate_median_PLAYBACK": per_step["PLAYBACK"]["gate"]["p50"],
        "dlt_rms_ACT": per_step["ACT"]["dlt"]["rms"],
        "dlt_rms_PLAYBACK": per_step["PLAYBACK"]["dlt"]["rms"],
        "event_dlt_distorted_ACT": event_stats["ACT"]["dlt"]["distorted"],
        "event_dlt_distorted_PLAYBACK": event_stats["PLAYBACK"]["dlt"]["distorted"],
        "event_adv_distorted_ACT": event_stats["ACT"]["adv"]["distorted"],
        "event_adv_distorted_PLAYBACK": event_stats["PLAYBACK"]["adv"]["distorted"],
    }
    r_act = headline["event_dlt_distorted_ACT"]["mean"]
    r_pb = headline["event_dlt_distorted_PLAYBACK"]["mean"]
    headline["event_dlt_ratio_ACT_over_PB"] = float(r_act / r_pb) if r_pb != 0 else float("inf")

    print("\n===== AGENCY GATE SUMMARY =====", flush=True)
    print(f"gate median: ACT={headline['gate_median_ACT']:.3f}  "
          f"PLAYBACK={headline['gate_median_PLAYBACK']:.4f}", flush=True)
    print(f"event dlt (distorted): ACT={r_act:.5f}+-{headline['event_dlt_distorted_ACT']['sem']:.5f}  "
          f"PLAYBACK={r_pb:.5f}+-{headline['event_dlt_distorted_PLAYBACK']['sem']:.5f}", flush=True)
    print(f"event (b-e) ungated (distorted): "
          f"ACT={headline['event_adv_distorted_ACT']['mean']:.5f}  "
          f"PLAYBACK={headline['event_adv_distorted_PLAYBACK']['mean']:.5f}", flush=True)
    print(f"dlt rms per-step: ACT={headline['dlt_rms_ACT']:.6f}  "
          f"PLAYBACK={headline['dlt_rms_PLAYBACK']:.6f}", flush=True)

    results = {
        "config": cfg, "n_train_transitions": N,
        "contact_frac_train": float(contact.mean()), "contact_frac_eval": float(con_e.mean()),
        "cmd_state_max_corr_train": corr_train, "cmd_state_max_corr_eval": corr_eval,
        "state_labels": STATE_LABELS, "value_slice_dims": V,
        "fm_sanity": sanity, "calibration": calib,
        "dv_rms": dv_rms, "d_mag": d_mag,
        "n_events": len(events), "n_events_distorted": n_dist,
        "per_step": per_step, "event_stats": event_stats,
        "bT_stats": bT_stats, "headline": headline,
    }

    figures = _make_figures(cfg, act, pb, ev_tr, offs, calib, results,
                            stat_mask, n_ev, T, events, w)

    outdir = os.path.join(DATA_DIR, "agency_gate", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    np.savez_compressed(
        os.path.join(outdir, "eval_traces.npz"),
        events=np.array([(e_i, t, int(d)) for e_i, t, d in events], np.int32),
        stat_mask=stat_mask,
        **{f"act_{k}": v for k, v in act.items()},
        **{f"pb_{k}": v for k, v in pb.items()},
        **{f"pb1_{k}": v for k, v in pb1.items()},
    )
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + traces + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_figures(cfg, act, pb, ev_tr, offs, calib, results, stat_mask, n_ev, T,
                  events, w):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    # fixed categorical assignment (matches arity_torque: warm = arity-2/ACT,
    # cool = arity-1/PLAYBACK); distorted/undistorted carried by linestyle.
    C_ACT, C_PB, C_MARK = "#d1603d", "#3d6fd1", "#444444"
    figs = {}

    def _save(fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def _loghist(ax, x, color, label):
        x = np.asarray(x)
        x = x[x > 0]
        bins = np.logspace(np.log10(max(x.min(), 1e-8)), np.log10(x.max() + 1e-12), 60)
        ax.hist(x, bins=bins, color=color, alpha=0.55, label=label, density=True)
        ax.set_xscale("log")

    # (1) per-step distributions: e, g, gate, dlt  (free-flight, post-burn-in)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    ax = axes[0, 0]
    _loghist(ax, act["e"][stat_mask], C_ACT, "ACT")
    _loghist(ax, pb["e"][stat_mask], C_PB, "PLAYBACK")
    ax.set_title("prediction error e (value slice)")
    ax.set_xlabel("e"); ax.set_ylabel("density"); ax.legend()
    ax = axes[0, 1]
    _loghist(ax, act["g"][stat_mask], C_ACT, "ACT")
    _loghist(ax, pb["g"][stat_mask], C_PB, "PLAYBACK")
    ax.axvline(calib["g0"], color=C_MARK, ls="--", lw=1)
    ax.text(calib["g0"], ax.get_ylim()[1] * 0.9, " g0", color=C_MARK, fontsize=8)
    ax.set_title("arity gap g = ||FM2 - FM1||")
    ax.set_xlabel("g"); ax.legend()
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 60)
    ax.hist(act["gate"][stat_mask], bins=bins, color=C_ACT, alpha=0.55,
            label="ACT", density=True)
    ax.hist(pb["gate"][stat_mask], bins=bins, color=C_PB, alpha=0.55,
            label="PLAYBACK", density=True)
    ax.set_yscale("log")
    ax.set_title("gate sigma((g - g0)/theta)")
    ax.set_xlabel("gate"); ax.set_ylabel("density (log)"); ax.legend()
    ax = axes[1, 1]
    lo = min(np.percentile(act["dlt"][stat_mask], 0.5),
             np.percentile(pb["dlt"][stat_mask], 0.5))
    hi = max(np.percentile(act["dlt"][stat_mask], 99.5),
             np.percentile(pb["dlt"][stat_mask], 99.5))
    bins = np.linspace(lo, hi, 80)
    ax.hist(act["dlt"][stat_mask], bins=bins, color=C_ACT, alpha=0.55,
            label="ACT", density=True)
    ax.hist(pb["dlt"][stat_mask], bins=bins, color=C_PB, alpha=0.55,
            label="PLAYBACK", density=True)
    ax.set_yscale("log")
    ax.axvline(0, color=C_MARK, lw=0.8)
    ax.set_title("bridge signal dlt = (b - e) * gate")
    ax.set_xlabel("dlt"); ax.set_ylabel("density (log)"); ax.legend()
    fig.suptitle("Agency gate, per-step distributions (identical sensory sequences; "
                 "only the efference-copy slot differs)", fontsize=11)
    figs["fig1_distributions.png"] = _save(fig)

    # (2) event-triggered traces: the Gadagkar figure
    # NOTE: no sharex — panel 3 is categorical and would clobber the trace axes.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [("dlt", "gated:  dlt = (b - e) * gate", axes[0]),
              ("adv", "ungated:  b - e  (the sensory channel)", axes[1])]
    for key, title, ax in panels:
        for cond, color in (("ACT", C_ACT), ("PLAYBACK", C_PB)):
            for name, ls in (("distorted", "-"), ("undistorted", "--")):
                t = ev_tr[cond][key][name]
                ax.plot(offs, t["mean"], ls, color=color, lw=2,
                        label=f"{cond} {name} (n={t['n']})")
                ax.fill_between(offs, t["mean"] - t["sem"], t["mean"] + t["sem"],
                                color=color, alpha=0.15, lw=0)
        ax.axvline(0, color=C_MARK, lw=0.8, ls=":")
        ax.axhline(0, color=C_MARK, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("steps from feedback event")
    axes[0].set_ylabel("signal")
    axes[0].legend(fontsize=7, loc="lower left")
    # panel 3: target-conditioned benchmark b_T (bar summary at event step)
    ax = axes[2]
    bT = results["bT_stats"]
    xpos, height, err, colors, labels = [], [], [], [], []
    for i, (cond, color) in enumerate((("ACT", C_ACT), ("PLAYBACK", C_PB))):
        for j, name in enumerate(("distorted", "undistorted")):
            r = bT[cond][name]
            xpos.append(i * 2.6 + j)
            height.append(r["dlt_T_mean"]); err.append(r["dlt_T_sem"])
            colors.append(color)
            labels.append(f"{cond if cond == 'ACT' else 'PB'}\n{name[:6]}.")
    bars = ax.bar(xpos, height, yerr=err, color=colors, width=0.75, capsize=3)
    for j in (1, 3):   # undistorted bars hatched (secondary encoding)
        bars[j].set_hatch("//"); bars[j].set_alpha(0.7)
    ax.axhline(0, color=C_MARK, lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("target-conditioned benchmark:\ndlt_T = (b_T - e) * gate at event",
                 fontsize=10)
    fig.suptitle("Event-triggered response to sensory distortion (Gadagkar Fig 4 analog): "
                 "the ungated channel sees the distortion in BOTH conditions; "
                 "the gate decides transmission", fontsize=11)
    figs["fig2_event_triggered.png"] = _save(fig)

    # (3) calibration transparency: raw g distributions + the gate curve
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    _loghist(ax, act["g"][stat_mask], C_ACT, "eval ACT")
    _loghist(ax, pb["g"][stat_mask], C_PB, "eval PLAYBACK")
    ax.axvline(calib["median_active"], color=C_ACT, ls=":", lw=1.4)
    ax.axvline(calib["median_passive"], color=C_PB, ls=":", lw=1.4)
    ax.axvline(calib["g0"], color=C_MARK, ls="--", lw=1.4)
    ax.text(calib["g0"], ax.get_ylim()[1] * 0.85, " g0 (train calib)",
            color=C_MARK, fontsize=8)
    ax.text(calib["median_active"], ax.get_ylim()[1] * 0.65, " train act med",
            color=C_ACT, fontsize=7)
    ax.text(calib["median_passive"], ax.get_ylim()[1] * 0.65, " train pas med",
            color=C_PB, fontsize=7)
    ax.set_ylabel("density"); ax.legend()
    ax.set_title("raw arity-gap distributions and the calibration "
                 f"(g0={calib['g0']:.4f}, theta={calib['theta']:.5f}, "
                 f"train-split medians)", fontsize=10)
    ax = axes[1]
    gs = np.logspace(np.log10(max(calib['median_passive'] * 0.05, 1e-8)),
                     np.log10(calib['median_active'] * 5), 400)
    ax.plot(gs, 1.0 / (1.0 + np.exp(-(gs - calib["g0"]) / calib["theta"])),
            color=C_MARK, lw=2, label="calibrated sigma((g-g0)/theta)")
    ax.plot(gs, 1.0 / (1.0 + np.exp(-gs / calib["theta"])),
            color=C_MARK, lw=1.2, ls="--", label="naive sigma(g/theta)  [sigma(0)=0.5]")
    ax.set_xscale("log")
    ax.set_xlabel("g"); ax.set_ylabel("gate")
    ax.legend(fontsize=8)
    figs["fig3_calibration.png"] = _save(fig)

    # (4) one episode, raw streams
    e_i = 0
    ts = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(ts, act["e"].reshape(n_ev, T)[e_i], color=C_ACT, lw=1.2, label="e ACT")
    ax.plot(ts, pb["e"].reshape(n_ev, T)[e_i], color=C_PB, lw=1.2, label="e PLAYBACK")
    for (ee, t, d) in events:
        if ee == e_i:
            ax.axvline(t, color=C_MARK, lw=0.7, ls="-" if d else ":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("e (log)")
    ax.legend(fontsize=8)
    ax.set_title("one eval episode (event markers: solid = distorted, dotted = clean)",
                 fontsize=10)
    ax = axes[1]
    ax.plot(ts, act["dlt"].reshape(n_ev, T)[e_i], color=C_ACT, lw=1.2, label="dlt ACT")
    ax.plot(ts, pb["dlt"].reshape(n_ev, T)[e_i], color=C_PB, lw=1.2, label="dlt PLAYBACK")
    for (ee, t, d) in events:
        if ee == e_i:
            ax.axvline(t, color=C_MARK, lw=0.7, ls="-" if d else ":", alpha=0.6)
    ax.axhline(0, color=C_MARK, lw=0.8)
    ax.set_ylabel("dlt"); ax.set_xlabel("step")
    ax.legend(fontsize=8)
    figs["fig4_timeseries.png"] = _save(fig)

    return figs


@app.local_entrypoint()
def agency_gate(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_episodes: int = 400,
    ep_len: int = 250,
    n_eval_episodes: int = 50,
    frame_skip: int = 5,
    sigma: float = 0.7,
    hidden: int = 64,
    layers: int = 2,
    epochs: int = 40,
    batch: int = 512,
    lr: float = 1e-3,
    alpha: float = 0.05,
    alpha_t: float = 0.2,
    bt_warmup: int = 5,
    burn_in: int = 20,
    target_every: int = 25,
    event_window: int = 6,
    distort_prob: float = 0.5,
    d_mag_scale: float = 1.0,
):
    import os

    if quick:
        n_episodes, ep_len, epochs = 60, 150, 12
        n_eval_episodes, hidden = 8, 32
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, n_episodes=n_episodes, ep_len=ep_len,
        n_eval_episodes=n_eval_episodes, frame_skip=frame_skip, sigma=sigma,
        hidden=hidden, layers=layers, epochs=epochs, batch=batch, lr=lr,
        alpha=alpha, alpha_T=alpha_t, bT_warmup=bt_warmup,
        burn_in=burn_in, target_every=target_every, event_window=event_window,
        distort_prob=distort_prob, d_mag_scale=d_mag_scale,
        # the arity_torque world; i.i.d. commands keep u perp s
        dgp=dict(arena_half=0.7, pusher_r=0.13, puck_r=0.13),
    )
    out = run_agency_gate.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
