"""Cut #1 — Contact-residual structure on a MuJoCo planar pusher.

Program: `ideas/physical_control_substrate.md`, first (cheapest) cut. The claim,
ported from the a2a residual-rank story (RHM low-rank, MNIST digit-discriminative,
language diffuse) to physics:

    An arity-2 forward model f(s,u) -> Δs of a contact-rich system should have a
    residual (actual - predicted) that CONCENTRATES AT CONTACT EVENTS — because
    free-flight dynamics are near-linear and easy, while contact is a stiff,
    near-discontinuous map from a coarse state.

No policy, no RL: a scripted OU+seek behavior policy generates (s,u,s')
transitions; MuJoCo hands us ground-truth contact labels. We fit a small MLP FM,
then test residual-at-contact vs residual-in-free-flight along several axes:

  * headline    : mean residual norm, contact vs free (ratio + Cohen's d)
  * NOT a magnitude artifact : the same holds for the RELATIVE residual
                  ||r|| / ||Δs|| (controls for contact targets being bigger)
  * dose-response: residual norm vs contact-force magnitude (Spearman)
  * structure    : which state dims carry the residual at contact (expect the
                   velocity dims of the contacting bodies)
  * rank         : effective rank of the residual (contact vs free) — the
                   physics analog of MNIST's low-rank digit-discriminative residual

Run:
    cd experiments/
    # smoke test (fast, ~1-2 min on Modal)
    modal run mjc/contact_residual/contact_residual.py::contact_residual --quick
    # full run
    modal run --detach mjc/contact_residual/contact_residual.py::contact_residual
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


# --------------------------------------------------------------------------- #
# metric helpers (numpy)                                                       #
# --------------------------------------------------------------------------- #
def _cohen_d(a, b):
    import numpy as np

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def _auc(a, b):
    """Common-language effect size: P(random contact residual > random free one).

    Distribution-free — the honest effect size for heavy-tailed contact residuals
    (Cohen's d is dragged down by the contact tail inflating the pooled std).
    0.5 = no separation, 1.0 = every contact residual exceeds every free one.
    """
    import numpy as np
    from scipy.stats import mannwhitneyu

    if len(a) < 1 or len(b) < 1:
        return float("nan")
    u, _ = mannwhitneyu(a, b, alternative="greater")
    return float(u / (len(a) * len(b)))


def _event_triggered(signal, contact_bool, ep_ids, pre, post, kind="onset"):
    """Event-triggered average of `signal` around free<->contact transitions.

    Aligns t=0 at each event and averages `signal` over a [-pre, +post] window,
    respecting episode boundaries (nan-padded at edges). `kind="onset"` triggers
    on free->contact (impact); `"separation"` on contact->free. Returns
    (offsets, mean, sem, n_events). All inputs are time-ordered within episode.
    """
    import numpy as np

    W = pre + post + 1
    windows = []
    for e in np.unique(ep_ids):
        idx = np.where(ep_ids == e)[0]            # contiguous, ascending, ordered
        c = contact_bool[idx]
        r = signal[idx]
        L = len(idx)
        for k in range(1, L):
            event = (c[k] and not c[k - 1]) if kind == "onset" \
                else (not c[k] and c[k - 1])
            if not event:
                continue
            w = np.full(W, np.nan)
            lo, hi = max(0, k - pre), min(L, k + post + 1)
            w[(lo - k) + pre:(hi - k) + pre] = r[lo:hi]
            windows.append(w)
    windows = np.asarray(windows) if windows else np.empty((0, W))
    n = windows.shape[0]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(windows, axis=0) if n else np.full(W, np.nan)
        cnt = np.sum(~np.isnan(windows), axis=0)
        sem = np.where(cnt > 0, np.nanstd(windows, axis=0) / np.sqrt(np.maximum(cnt, 1)),
                       np.nan)
    return np.arange(-pre, post + 1), mean, sem, n


def _effective_rank(R):
    """Participation ratio of the residual covariance spectrum: (Σλ)² / Σλ².

    R is (N, d), centered internally. eff_rank in [1, d]; low => residual lives
    in a few directions (structured), high => diffuse/full-rank.
    """
    import numpy as np

    if R.shape[0] < R.shape[1] + 1:
        return float("nan"), float("nan")
    Rc = R - R.mean(0, keepdims=True)
    sv = np.linalg.svd(Rc, full_matrices=False, compute_uv=False)
    lam = sv ** 2
    eff = float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))
    top1 = float(lam.max() / (lam.sum() + 1e-12))
    return eff, top1


def _spearman(x, y):
    import numpy as np
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def _r2(y_true, y_pred, mask):
    """Fraction of Δs variance the FM explains WITHIN a regime (raw units).

    Scale-free by construction — this is the honest "contact is genuinely less
    predictable, not just bigger" control (unlike ‖r‖/‖Δs‖, which blows up on the
    many near-static free transitions where ‖Δs‖ -> 0). Free-flight is near-linear
    so R² -> 1; contact is stiff/near-discontinuous so R² drops.
    """
    import numpy as np

    if int(mask.sum()) < 2:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# --------------------------------------------------------------------------- #
# the experiment                                                              #
# --------------------------------------------------------------------------- #
@app.function(cpu=8.0, memory=16384, timeout=3600, volumes={DATA_DIR: volume})
def run_contact_residual(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import collect_transitions, STATE_LABELS

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. collect scripted transitions ---------------------------------- #
    print(f"[collect] {cfg['n_episodes']} eps x {cfg['ep_len']} steps "
          f"(frame_skip={cfg['frame_skip']}) ...", flush=True)
    data = collect_transitions(
        dgp=cfg.get("dgp"), n_episodes=cfg["n_episodes"], ep_len=cfg["ep_len"],
        frame_skip=cfg["frame_skip"], seed=cfg["seed"],
        sigma=cfg["sigma"], theta=cfg["theta"], seek_gain=cfg["seek_gain"],
    )
    S, U, S2 = data["S"], data["U"], data["S2"]
    # Primary regime label = ANY contact (pusher-puck, puck-wall, pusher-wall) vs
    # genuine free flight (ncon==0). All contacts are the stiff/discontinuous
    # regime; labeling only puck-contacts would mislabel wall contacts as "free"
    # and pollute the free-flight baseline. puck_force is kept for dose-response.
    contact = data["any_contact"]
    force = data["force"]
    ep = data["ep"]
    N = len(S)
    contact_frac = float(contact.mean())
    print(f"[collect] N={N} transitions, contact_frac={contact_frac:.3f}", flush=True)

    # ---- 2. train/test split by EPISODE (no leakage) ---------------------- #
    n_ep = int(ep.max()) + 1
    n_test_ep = max(1, int(round(0.2 * n_ep)))
    test_eps = set(range(n_ep - n_test_ep, n_ep))
    is_test = np.array([e in test_eps for e in ep])
    tr, te = ~is_test, is_test

    X = np.concatenate([S, U], axis=1).astype(np.float32)   # (N, 10)
    Y = (S2 - S).astype(np.float32)                          # (N, 8) = Δs

    # normalization from TRAIN only
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
    my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xn = (X - mx) / sx
    Yn = (Y - my) / sy

    Xtr = torch.tensor(Xn[tr], device=device)
    Ytr = torch.tensor(Yn[tr], device=device)
    Xte = torch.tensor(Xn[te], device=device)

    # ---- 3. small arity-2 MLP forward model f(s,u) -> Δŝ ------------------ #
    din, dout, dh, nl = X.shape[1], Y.shape[1], cfg["hidden"], cfg["layers"]
    layers = [nn.Linear(din, dh), nn.SiLU()]
    for _ in range(nl - 1):
        layers += [nn.Linear(dh, dh), nn.SiLU()]
    layers += [nn.Linear(dh, dout)]
    fm = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    # Huber (smooth-L1) loss: identical to MSE for normalized residuals below δ,
    # but linear (bounded-gradient) beyond it. Contact transitions have heavy-tailed
    # Δs; under plain MSE they dominate the loss and starve free-flight learning
    # (higher contact fraction -> worse free fit -> the real contrast collapses).
    # Huber keeps the FM a good predictor of the PREDICTABLE (free) dynamics, so
    # the residual cleanly isolates the surprising (contact) regime. δ=inf -> MSE.
    lossf = (nn.MSELoss() if cfg.get("huber_delta", 1.0) in (None, float("inf"))
             else nn.HuberLoss(delta=cfg.get("huber_delta", 1.0)))

    ntr = Xtr.shape[0]
    bs = min(cfg["batch"], ntr)
    print(f"[train] FM {din}->{dh}x{nl}->{dout}, {cfg['epochs']} epochs on {ntr} samples",
          flush=True)
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(ntr, device=device)
        tot = 0.0
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(fm(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % max(1, cfg["epochs"] // 8) == 0 or epoch == cfg["epochs"] - 1:
            print(f"  epoch {epoch:3d}  train_mse={tot / ntr:.5f}", flush=True)

    # ---- 4. residuals on the TEST set ------------------------------------- #
    fm.eval()
    with torch.no_grad():
        Yn_pred = fm(Xte).cpu().numpy()
    Yn_te = Yn[te]                         # normalized true Δs
    Y_pred_raw = Yn_pred * sy + my         # raw-unit predicted Δs
    res_n = Yn_pred - Yn_te                # normalized residual (per-dim)
    res_raw = Y_pred_raw - Y[te]           # raw-unit residual

    rn = np.linalg.norm(res_n, axis=1)         # normalized residual norm
    rr = np.linalg.norm(res_raw, axis=1)       # raw residual norm
    dy = np.linalg.norm(Y[te], axis=1)         # raw Δs norm
    rel = rr / (dy + 1e-6)                      # relative residual (magnitude control)

    c_te = contact[te]
    f_te = force[te]
    # per-transition FM cosine (direction match), normalized space
    cos = (Yn_pred * Yn_te).sum(1) / (
        np.linalg.norm(Yn_pred, axis=1) * np.linalg.norm(Yn_te, axis=1) + 1e-9)

    def split(v):
        return v[c_te], v[~c_te]

    rn_c, rn_f = split(rn)
    rel_c, rel_f = split(rel)
    cos_c, cos_f = split(cos)

    eff_c, top1_c = _effective_rank(res_n[c_te])
    eff_f, top1_f = _effective_rank(res_n[~c_te])
    eff_all, top1_all = _effective_rank(res_n)

    # per-regime R² (the scale-free predictability control) in raw units
    Yte_raw = Y[te]
    r2_c = _r2(Yte_raw, Y_pred_raw, c_te)
    r2_f = _r2(Yte_raw, Y_pred_raw, ~c_te)
    r2_all = _r2(Yte_raw, Y_pred_raw, np.ones(len(c_te), bool))

    perdim_c = np.abs(res_n[c_te]).mean(0)
    perdim_f = np.abs(res_n[~c_te]).mean(0)

    # --- onset-aligned event-triggered average (the "spike at onset" test) ----
    # Does the residual spike at the free->contact IMPACT and decay, or is it
    # uniformly elevated whenever in contact? We also track P(in contact) around
    # the event: if the residual decays while contact persists, the spike marks
    # the ONSET EVENT (regime transition), not the contact STATE.
    ep_te = ep[te]
    PRE, POST = 8, 20
    off, eta_on, sem_on, n_on = _event_triggered(rn, c_te, ep_te, PRE, POST, "onset")
    _, pcon_on, _, _ = _event_triggered(c_te.astype(float), c_te, ep_te, PRE, POST, "onset")
    _, eta_sep, sem_sep, n_sep = _event_triggered(rn, c_te, ep_te, PRE, POST, "separation")

    def _win(a, lo, hi):  # mean of ETA over an offset range [lo, hi] inclusive
        m = (off >= lo) & (off <= hi)
        return float(np.nanmean(a[m]))

    onset_peak = float(eta_on[off == 0][0])
    onset_pre = _win(eta_on, -PRE, -3)      # free approach, before impact
    onset_sustained = _win(eta_on, 8, POST)  # later, mostly still-in-contact steps

    results = {
        "config": cfg,
        "n_transitions": N,
        "n_test": int(te.sum()),
        "contact_frac": contact_frac,
        "contact_frac_test": float(c_te.mean()),
        "state_labels": STATE_LABELS,
        # headline: residual concentrates at contact
        "resid_norm_contact": float(rn_c.mean()),
        "resid_norm_free": float(rn_f.mean()),
        "resid_norm_ratio": float(rn_c.mean() / (rn_f.mean() + 1e-9)),
        "resid_norm_cohen_d": _cohen_d(rn_c, rn_f),
        "resid_norm_auc": _auc(rn_c, rn_f),
        # NOTE: per-regime R² is kept for reference but is NOT a clean control
        # here — free-flight Δs is so tiny/low-variance it sits below the FM's
        # global error floor, so r2_free can even go negative. The scale-free
        # control is fm_cos (direction), above. See README.
        "r2_contact": r2_c, "r2_free": r2_f, "r2_all": r2_all,
        # relative residual ||r||/||Δs|| (also noisy: blows up as ||Δs||->0)
        "rel_resid_contact": float(rel_c.mean()),
        "rel_resid_free": float(rel_f.mean()),
        "rel_resid_ratio": float(rel_c.mean() / (rel_f.mean() + 1e-9)),
        "rel_resid_cohen_d": _cohen_d(rel_c, rel_f),
        # FM direction fidelity
        "fm_cos_contact": float(cos_c.mean()),
        "fm_cos_free": float(cos_f.mean()),
        # dose-response
        "spearman_force_resid_all": _spearman(f_te, rn),
        "spearman_force_resid_contactonly": _spearman(f_te[c_te], rn_c),
        # rank structure
        "eff_rank_contact": eff_c, "top1_frac_contact": top1_c,
        "eff_rank_free": eff_f, "top1_frac_free": top1_f,
        "eff_rank_all": eff_all, "top1_frac_all": top1_all,
        # per-dim residual
        "perdim_resid_contact": perdim_c.tolist(),
        "perdim_resid_free": perdim_f.tolist(),
        # onset-aligned event-triggered average
        "n_onset_events": n_on, "n_separation_events": n_sep,
        "onset_eta_offsets": off.tolist(),
        "onset_eta_mean": eta_on.tolist(),
        "onset_eta_sem": sem_on.tolist(),
        "onset_pcontact": pcon_on.tolist(),
        "separation_eta_mean": eta_sep.tolist(),
        "onset_peak": onset_peak,
        "onset_pre_baseline": onset_pre,
        "onset_sustained": onset_sustained,
        "onset_peak_over_pre": onset_peak / (onset_pre + 1e-9),
        "onset_peak_over_sustained": onset_peak / (onset_sustained + 1e-9),
    }

    print("\n===== CONTACT-RESIDUAL SUMMARY =====", flush=True)
    print(f"contact fraction (test): {results['contact_frac_test']:.3f}", flush=True)
    print(f"resid norm   contact/free = {results['resid_norm_contact']:.4f} / "
          f"{results['resid_norm_free']:.4f}  "
          f"(ratio {results['resid_norm_ratio']:.2f}x, d={results['resid_norm_cohen_d']:.2f}, "
          f"AUC={results['resid_norm_auc']:.3f})", flush=True)
    print(f"FM cosine    contact/free = {results['fm_cos_contact']:.3f} / "
          f"{results['fm_cos_free']:.3f}  (scale-free control: direction, not magnitude)",
          flush=True)
    print(f"Spearman(force, resid)    = {results['spearman_force_resid_all']:.3f} "
          f"(all) / {results['spearman_force_resid_contactonly']:.3f} (contact only)",
          flush=True)
    print(f"eff-rank     contact/free/all = {eff_c:.2f} / {eff_f:.2f} / {eff_all:.2f} "
          f"(of {dout})", flush=True)
    print(f"onset ETA    pre/peak/sustained = {onset_pre:.3f} / {onset_peak:.3f} / "
          f"{onset_sustained:.3f}  (peak/pre {results['onset_peak_over_pre']:.1f}×, "
          f"peak/sustained {results['onset_peak_over_sustained']:.1f}×; "
          f"{n_on} onsets)", flush=True)

    # ---- 5. figures ------------------------------------------------------- #
    figures = _make_figures(
        res_n=res_n, rn=rn, cos=cos, c_te=c_te, f_te=f_te,
        perdim_c=perdim_c, perdim_f=perdim_f, labels=STATE_LABELS,
        # a sample test trajectory for the time series
        traj=_pick_trajectory(ep, te, contact, force, rn=rn),
        eta=dict(off=off, eta_on=eta_on, sem_on=sem_on, pcon=pcon_on,
                 eta_sep=eta_sep, sem_sep=sem_sep, n_on=n_on, n_sep=n_sep,
                 free_baseline=float(rn[~c_te].mean())),
        results=results,
    )

    # ---- 6. persist to volume --------------------------------------------- #
    outdir = os.path.join(DATA_DIR, "contact_residual", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)

    return {"results": results, "figures": figures}


def _pick_trajectory(ep, te, contact, force, rn):
    """Return per-step arrays for ONE test episode with the most contact, for
    the time-series figure. Reconstructs residual norm along that episode."""
    import numpy as np

    test_ep_ids = np.unique(ep[te])
    # score each test episode by contact count, pick the richest
    best, best_score = test_ep_ids[0], -1
    te_ep = ep[te]
    c_te = contact[te]
    for e in test_ep_ids:
        sc = int(c_te[te_ep == e].sum())
        if sc > best_score:
            best_score, best = sc, e
    mask = te_ep == best
    return dict(
        rn=rn[mask], contact=c_te[mask], force=force[te][mask], ep=int(best),
    )


def _make_figures(res_n, rn, cos, c_te, f_te, perdim_c, perdim_f, labels,
                  traj, eta, results):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_CONTACT, C_FREE = "#d1603d", "#3d6fd1"
    figs = {}

    def _save(fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # (1) residual over a sample trajectory, contact shaded
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = np.arange(len(traj["rn"]))
    ax.plot(t, traj["rn"], color="#222", lw=1.4, label="FM residual ‖r‖")
    ymax = max(traj["rn"].max(), 1e-6)
    ax.fill_between(t, 0, ymax * 1.05, where=traj["contact"], color=C_CONTACT,
                    alpha=0.22, step="mid", label="contact (ground truth)")
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_xlabel("control step (test episode)")
    ax.set_ylabel("residual norm (norm. units)")
    ax.set_title(f"Forward-model residual spikes at contact  "
                 f"(episode {traj['ep']}; ratio "
                 f"{results['resid_norm_ratio']:.1f}× contact/free)")
    ax.legend(loc="upper right", framealpha=0.9)
    figs["fig1_trajectory.png"] = _save(fig)

    # (2) LEFT: residual-norm distribution contact vs free.
    #     RIGHT: FM cosine(Δŝ, Δs) — scale-free control (direction, not magnitude).
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    parts = axes[0].violinplot(
        [np.log10(rn[~c_te] + 1e-6), np.log10(rn[c_te] + 1e-6)], showmedians=True)
    for pc, col in zip(parts["bodies"], [C_FREE, C_CONTACT]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["free", "contact"])
    axes[0].set_ylabel("log10 residual norm")
    axes[0].set_title("residual norm ‖r‖")
    axes[0].text(0.02, 0.98,
                 f"ratio {results['resid_norm_ratio']:.1f}×\n"
                 f"d={results['resid_norm_cohen_d']:.2f}",
                 transform=axes[0].transAxes, va="top", fontsize=9)

    parts = axes[1].violinplot([cos[~c_te], cos[c_te]], showmedians=True)
    for pc, col in zip(parts["bodies"], [C_FREE, C_CONTACT]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["free", "contact"])
    axes[1].set_ylabel("cosine(Δŝ, Δs)")
    axes[1].set_title("FM direction fidelity")
    axes[1].text(0.02, 0.02,
                 f"free {results['fm_cos_free']:.3f}\ncontact {results['fm_cos_contact']:.3f}",
                 transform=axes[1].transAxes, va="bottom", fontsize=9)
    fig.suptitle("Contact residual is larger AND directionally worse "
                 "(scale-free, not just bigger Δs)", fontsize=11)
    figs["fig2_distributions.png"] = _save(fig)

    # (3) per-dim residual, contact vs free
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, perdim_f, w, color=C_FREE, label="free")
    ax.bar(x + w / 2, perdim_c, w, color=C_CONTACT, label="contact")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("mean |residual| (norm. units)")
    ax.set_title("Which state dims carry the contact residual")
    ax.legend()
    figs["fig3_perdim.png"] = _save(fig)

    # (4) dose-response: residual vs contact force (contact-only, log-x, binned).
    # Contact force is extremely heavy-tailed (impacts spike to ~1e5), so raw
    # linear axes are illegible; log-x + binned means shows the monotone trend.
    fig, ax = plt.subplots(figsize=(5.6, 4))
    xf, yr = f_te[c_te], rn[c_te]
    m = xf > 0
    xf, yr = xf[m], yr[m]
    ax.scatter(xf, yr, s=7, color=C_CONTACT, alpha=0.18)
    if len(xf) > 24:
        lx = np.log10(xf)
        edges = np.quantile(lx, np.linspace(0, 1, 9))
        cen, mu, sem = [], [], []
        for i in range(len(edges) - 1):
            hi_incl = i == len(edges) - 2
            sel = (lx >= edges[i]) & (lx <= edges[i + 1] if hi_incl else lx < edges[i + 1])
            if sel.sum() >= 5:
                cen.append(10 ** (0.5 * (edges[i] + edges[i + 1])))
                mu.append(yr[sel].mean())
                sem.append(yr[sel].std() / np.sqrt(sel.sum()))
        ax.errorbar(cen, mu, yerr=sem, color="#111", marker="o", lw=1.8,
                    capsize=3, label="binned mean ± sem")
        ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("peak contact force (MuJoCo ground truth, log)")
    ax.set_ylabel("residual norm")
    ax.set_title(f"Dose-response within contact  "
                 f"(Spearman ρ={results['spearman_force_resid_contactonly']:.2f})")
    figs["fig4_doseresponse.png"] = _save(fig)

    # (5) onset-aligned event-triggered average — the mechanism test.
    off = eta["off"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)

    # LEFT: onset. residual ETA (left axis) + P(in contact) (right axis).
    axL = axes[0]
    axL.axvspan(0, off.max(), color=C_CONTACT, alpha=0.07)
    axL.axhline(eta["free_baseline"], color="#888", ls="--", lw=1,
                label="free-flight baseline")
    axL.errorbar(off, eta["eta_on"], yerr=eta["sem_on"], color="#111", marker="o",
                 ms=3, lw=1.6, capsize=2, label="residual ‖r‖")
    axL.axvline(0, color=C_CONTACT, lw=1.2)
    axL.set_xlabel("control steps since contact onset")
    axL.set_ylabel("residual norm (norm. units)")
    axL.set_title(f"Onset-aligned average  (n={eta['n_on']} impacts)")
    axR = axL.twinx()
    axR.plot(off, eta["pcon"], color=C_FREE, lw=1.6, alpha=0.8)
    axR.set_ylabel("P(in contact)", color=C_FREE)
    axR.tick_params(axis="y", labelcolor=C_FREE)
    axR.set_ylim(-0.03, 1.03)
    axL.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axL.text(0.03, 0.97,
             f"peak/pre {results['onset_peak_over_pre']:.1f}×\n"
             f"peak/sustained {results['onset_peak_over_sustained']:.1f}×",
             transform=axL.transAxes, va="top", fontsize=9)

    # RIGHT: separation (contact->free), same window. Impact >> release.
    axS = axes[1]
    axS.axvspan(off.min(), 0, color=C_CONTACT, alpha=0.07)
    axS.axhline(eta["free_baseline"], color="#888", ls="--", lw=1)
    axS.errorbar(off, eta["eta_sep"], yerr=eta["sem_sep"], color="#111", marker="o",
                 ms=3, lw=1.6, capsize=2)
    axS.axvline(0, color="#888", lw=1.2)
    axS.set_xlabel("control steps since contact release")
    axS.set_title(f"Separation-aligned average  (n={eta['n_sep']} releases)")

    fig.suptitle("The residual spikes at the contact ONSET event, then decays "
                 "while contact persists — it marks the transition, not the state",
                 fontsize=11)
    figs["fig5_onset_eta.png"] = _save(fig)

    return figs


@app.local_entrypoint()
def contact_residual(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    n_episodes: int = 500,
    ep_len: int = 250,
    frame_skip: int = 5,
    sigma: float = 0.5,
    theta: float = 0.15,
    seek_gain: float = 1.0,
    hidden: int = 256,
    layers: int = 3,
    epochs: int = 60,
    batch: int = 512,
    lr: float = 1e-3,
    huber_delta: float = 1.0,
):
    import os

    if quick:
        n_episodes, ep_len, epochs = 80, 200, 30
        tag = tag or "smoke"
    tag = tag or "default"

    # Contact-rich-but-open world for cut #1: enough room for genuine free flight
    # to contrast against, slightly larger bodies so the scripted policy engages
    # the puck. Contact = any contact (incl. walls), so contact_frac lands ~0.2-0.4.
    # These are DGP knobs; later cuts sweep them.
    cfg = dict(
        tag=tag, seed=seed, n_episodes=n_episodes, ep_len=ep_len,
        frame_skip=frame_skip, sigma=sigma, theta=theta, seek_gain=seek_gain,
        hidden=hidden, layers=layers, epochs=epochs, batch=batch, lr=lr,
        huber_delta=huber_delta,
        dgp=dict(arena_half=0.8, pusher_r=0.15, puck_r=0.15, puck_mass=2.0),
    )
    out = run_contact_residual.remote(cfg)

    # write figures + results locally so we can look at them
    localdir = os.path.join(os.path.dirname(__file__), "figures", tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
