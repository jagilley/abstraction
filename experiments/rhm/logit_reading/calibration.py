"""Part 1: how tightly do a model's logits track its own epistemic state?

Reads a frozen RHM base model's next-token distribution q against the exact
phase-marginal Bayes predictive (`flat_oracle.py`) on flat windows -- the distribution
the model was trained on -- and against the coarse-observer family p_k (observer k
knows the bottom k grammar levels exactly and nothing above; p_L is the truth).

Per prediction (window, t), all exact expectations under the oracle, no sampling:
  H_p        H(p_L)                 irreducible uncertainty given the window
  H_q        H(q)                   the model's own ("felt") uncertainty
  CE         -sum p_L log q         the model's true expected loss
  gap        CE - H_q               do the logits predict their own loss? (+ = overconfident)
  KL_k       KL(p_k || q)           is q the posterior of observer k?
  best_k     argmin_k KL_k
  KL_known   KL(p_known || q)       p_known = observer L told the true phase
plus top-1 reliability, both realised (argmax q == token) and expected (p_L(argmax q)),
and the per-instance question: how much of the variation in CE does H_q explain?

Self-checks inside every run: mean realised -log p_L(token) must match mean H_p, and
mean realised NLL must match mean CE, to Monte-Carlo error (tower property -- the
token is drawn from the true phase, p_L integrates the phase posterior).

Run:
  # oracle self-test (local, CPU)
  cd experiments && python -m rhm.logit_reading.flat_oracle
  # smoke (attached)
  modal run -m rhm.logit_reading.calibration::calibration --n-windows 32 --tag smoke
  # the cached conditional_revision base
  modal run --detach -m rhm.logit_reading.calibration::calibration --n-windows 2048 --tag cr_base
"""

import json
import os

import modal
import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-logit-reading", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def _ent(p):
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(-1)


def _xent(p, logq):
    return -(p * logq).sum(-1)


def _r2(y, x):
    y = np.asarray(y, np.float64).ravel()
    x = np.asarray(x, np.float64).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def _reliability(conf, acc, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        sel = b == i
        if sel.sum() == 0:
            continue
        c, a = float(conf[sel].mean()), float(acc[sel].mean())
        ece += sel.mean() * abs(c - a)
        rows.append({"bin_lo": float(edges[i]), "n": int(sel.sum()), "conf": c, "acc": a})
    return float(ece), rows


def summarize(q_logits, preds, p_known, windows, phase, logpi_true, L, s, n_bins=15):
    """All Part-1 statistics from arrays. Reusable across checkpoints.

    q_logits: (n, T, v) model logits predicting windows[:, 1:].
    preds:    (n, T, L+1, v) observer predictives p_k at the same targets.
    p_known:  (n, T, v); logpi_true: (n, T) log posterior of the true phase before target.
    """
    from rhm.logit_reading.flat_oracle import leaf_levels
    n, T, v = q_logits.shape
    ql = q_logits.astype(np.float64)
    logq = ql - np.log(np.exp(ql - ql.max(-1, keepdims=True)).sum(-1, keepdims=True)) \
        - ql.max(-1, keepdims=True)
    q = np.exp(logq)
    p = preds[:, :, L, :]
    tok = windows[:, 1:]
    lev = leaf_levels((phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L), s, L)

    H_p, H_q = _ent(p), _ent(q)
    CE = _xent(p, logq)
    KL = CE - H_p
    gap = CE - H_q
    KL_k = np.stack([_xent(preds[:, :, k, :], logq) - _ent(preds[:, :, k, :])
                     for k in range(L + 1)], -1)                      # (n, T, L+1)
    H_k = np.stack([_ent(preds[:, :, k, :]) for k in range(L + 1)], -1)
    KL_truth_k = np.stack([_xent(p, np.log(np.clip(preds[:, :, k, :], 1e-300, None))) - H_p
                           for k in range(L + 1)], -1)                 # KL(p_L || p_k)
    best_k = KL_k.argmin(-1)
    KL_known = _xent(p_known, logq) - _ent(p_known)
    H_known = _ent(p_known)
    nll = -np.take_along_axis(logq, tok[..., None], -1)[..., 0]
    surpr = -np.log(np.clip(np.take_along_axis(p, tok[..., None], -1)[..., 0], 1e-300, None))
    top = q.argmax(-1)
    conf = q.max(-1)
    acc_real = (top == tok).astype(np.float64)
    acc_exp = np.take_along_axis(p, top[..., None], -1)[..., 0]

    def block(sel):
        d = {
            "n": int(sel.sum()),
            "H_p": float(H_p[sel].mean()), "H_q": float(H_q[sel].mean()),
            "CE": float(CE[sel].mean()), "KL_p_q": float(KL[sel].mean()),
            "gap_CE_minus_Hq": float(gap[sel].mean()),
            "abs_gap": float(np.abs(gap[sel]).mean()),
            "H_known": float(H_known[sel].mean()), "KL_known_q": float(KL_known[sel].mean()),
            "KL_k_q": [float(KL_k[..., k][sel].mean()) for k in range(L + 1)],
            "KL_truth_k": [float(KL_truth_k[..., k][sel].mean()) for k in range(L + 1)],
            "H_k": [float(H_k[..., k][sel].mean()) for k in range(L + 1)],
            "best_k_hist": np.bincount(best_k[sel], minlength=L + 1).tolist(),
            "KL_best_q": float(KL_k.min(-1)[sel].mean()),
            "nll_realised": float(nll[sel].mean()), "surprisal_realised": float(surpr[sel].mean()),
            "top1_acc_realised": float(acc_real[sel].mean()),
            "top1_acc_expected": float(acc_exp[sel].mean()),
            "top1_conf": float(conf[sel].mean()),
            "R2_CE_on_Hq": _r2(CE[sel], H_q[sel]),
            "R2_CE_on_Hp": _r2(CE[sel], H_p[sel]),
            "R2_Hq_on_Hp": _r2(H_q[sel], H_p[sel]),
            "R2_Hq_on_Hbestk": _r2(H_q[sel], np.take_along_axis(H_k, best_k[..., None], -1)[..., 0][sel]),
        }
        return d

    allsel = np.ones_like(H_p, dtype=bool)
    out = {"overall": block(allsel)}
    out["by_level"] = {int(l): block(lev == l) for l in range(L + 1) if (lev == l).any()}
    tb = [(0, 4), (4, 16), (16, 32), (32, T)]
    out["by_window_pos"] = {f"{a}-{b}": block(np.broadcast_to(
        (np.arange(T) >= a) & (np.arange(T) < b), H_p.shape)) for a, b in tb}
    out["phase_logpost_true_mean_by_window_pos"] = {
        f"{a}-{b}": float(logpi_true[:, a:b].mean()) for a, b in tb}
    ece_r, rel_r = _reliability(conf.ravel(), acc_real.ravel(), n_bins)
    ece_e, rel_e = _reliability(conf.ravel(), acc_exp.ravel(), n_bins)
    out["ece_realised"], out["reliability_realised"] = ece_r, rel_r
    out["ece_expected"], out["reliability_expected"] = ece_e, rel_e
    # Monte-Carlo self-checks (tower property)
    se = lambda x: float(x.std() / np.sqrt(x.size))
    out["selfcheck"] = {
        "surprisal_minus_Hp": float(surpr.mean() - H_p.mean()), "se_surprisal": se(surpr),
        "nll_minus_CE": float(nll.mean() - CE.mean()), "se_nll": se(nll),
    }
    return out


def evaluate(model, rules, rule_w, n_windows, eval_seed, chunk, device="cuda"):
    """Model logits + every observer's predictive on fresh flat windows -> (res, arrays)."""
    import time
    import torch
    from rhm.logit_reading.flat_oracle import flat_predictive, sample_flat_windows

    L = len(rules)
    v, _, s = rules[0].shape
    T = s ** L
    windows, phase = sample_flat_windows(rules, n_windows, T + 1, eval_seed, rule_w=rule_w)
    with torch.no_grad():
        X = torch.as_tensor(windows[:, :T], device=device)
        logits = torch.cat([model(X[i:i + 512])[0].double().cpu()
                            for i in range(0, n_windows, 512)]).numpy()

    preds = np.empty((n_windows, T, L + 1, v), dtype=np.float64)
    for k in range(L + 1):
        t0 = time.time()
        if k == L:
            pr, logpi, p_psi = flat_predictive(windows, rules, k, device=device, chunk=chunk,
                                               return_phase=True, rule_w=rule_w)
            ar = np.arange(n_windows)
            p_known = p_psi[ar, phase].numpy()[:, 1:, :]
            logpi_true = logpi[ar, :, phase].numpy()[:, :T]           # after w_<=t
            del p_psi
        else:
            pr = flat_predictive(windows, rules, k, device=device, chunk=max(chunk, 64),
                                 rule_w=rule_w)
        preds[:, :, k, :] = pr.numpy()[:, 1:, :]
        print(f"  observer k={k}  {time.time() - t0:.1f}s", flush=True)

    res = summarize(logits, preds, p_known, windows, phase, logpi_true, L, s)
    arrays = dict(logits=logits.astype(np.float32), preds=preds.astype(np.float32),
                  p_known=p_known.astype(np.float32), windows=windows, phase=phase,
                  logpi_true=logpi_true.astype(np.float32))
    return res, arrays


def print_summary(res):
    o = res["overall"]
    print(f"\nOVERALL  H_p {o['H_p']:.4f}  H_q {o['H_q']:.4f}  CE {o['CE']:.4f}  "
          f"KL {o['KL_p_q']:.4f}  gap {o['gap_CE_minus_Hq']:+.4f}  "
          f"R2(CE~Hq) {o['R2_CE_on_Hq']:.3f}  ECE real/exp {res['ece_realised']:.4f}/"
          f"{res['ece_expected']:.4f}")
    print(f"  selfcheck {res['selfcheck']}")
    print("  KL(p_L||p_k) " + " ".join(f"{x:.3f}" for x in o["KL_truth_k"])
          + "   KL(p_k||q) " + " ".join(f"{x:.3f}" for x in o["KL_k_q"]))
    print(f"\n{'lev':>3} {'n':>6} {'H_p':>6} {'H_q':>6} {'CE':>6} {'gap':>7} {'R2':>5}  "
          f"KL_k_q (k=0..)                                    best_k hist")
    for l, b in res["by_level"].items():
        print(f"{l:>3} {b['n']:>6} {b['H_p']:.3f} {b['H_q']:.3f} {b['CE']:.3f} "
              f"{b['gap_CE_minus_Hq']:+.3f} {b['R2_CE_on_Hq']:.2f}  "
              + " ".join(f"{x:.3f}" for x in b["KL_k_q"]) + f"   {b['best_k_hist']}")
    for wp, b in res["by_window_pos"].items():
        print(f"  window {wp:>6}: KL(p||q) {b['KL_p_q']:.3f}  KL(p_known||q) {b['KL_known_q']:.3f}  "
              f"H_p {b['H_p']:.3f}  H_known {b['H_known']:.3f}  gap {b['gap_CE_minus_Hq']:+.3f}")


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def calibration(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    base_steps: int = 12000, seed: int = 42, base_ckpt: str = "",
    n_windows: int = 2048, eval_seed: int = 4242, chunk: int = 8,
    save_arrays: bool = True, tag: str = "",
):
    """The cached conditional_revision base (uniform synonyms)."""
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    ckpt = base_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
    sd = torch.load(ckpt, map_location=device)
    cfg = sd.get("config", {})
    assert cfg.get("rule_seed", rule_seed) == rule_seed, cfg
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"loaded {ckpt}  config {cfg}", flush=True)

    res, arrays = evaluate(model, rules, None, n_windows, eval_seed, chunk, device)
    res["config"] = {"ckpt": ckpt, "ckpt_config": cfg, "n_windows": n_windows,
                     "eval_seed": eval_seed, "key": key}
    print_summary(res)

    out_dir = f"{DATA_DIR}/{key}/logit_reading"
    os.makedirs(out_dir, exist_ok=True)
    name = f"calibration_{tag or 'run'}"
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    if save_arrays:
        np.savez_compressed(f"{out_dir}/{name}.npz", **arrays)
    volume.commit()
    print(f"saved -> {out_dir}/{name}.json", flush=True)
    return res


def load_trajectory_ckpt(path, device):
    """A `train_trajectory` checkpoint -> (model, rules, rule_w, cfg)."""
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    sd = torch.load(path, map_location=device)
    cfg = sd["config"]
    rules = generate_rules_distinct(cfg["v"], cfg["s"], cfg["L"], cfg["m"], seed=cfg["rule_seed"])
    rule_w = (None if cfg["alpha"] <= 0 else
              synonym_weights(cfg["v"], cfg["L"], cfg["m"], cfg["alpha"], cfg["weight_seed"]))
    model = GPT(cfg["v"], cfg["s"] ** cfg["L"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, rules, rule_w, cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=16384, max_containers=4)
def calibration_ckpt(ckpt: str, n_windows: int = 4096, eval_seed: int = 4242, chunk: int = 8,
                     save_arrays: bool = True):
    """Part 1 on one `train_trajectory` checkpoint; writes calibration.{json,npz} beside it."""
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    print(f"loaded {ckpt}  step {cfg.get('step')}", flush=True)
    res, arrays = evaluate(model, rules, rule_w, n_windows, eval_seed, chunk, "cuda")
    res["config"] = {"ckpt": ckpt, "ckpt_config": cfg, "n_windows": n_windows,
                     "eval_seed": eval_seed}
    print_summary(res)
    stem = ckpt[:-3]
    with open(f"{stem}_calibration.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    if save_arrays:
        np.savez_compressed(f"{stem}_calibration.npz", **arrays)
    volume.commit()
    return {"ckpt": ckpt, "step": cfg.get("step"), "overall": res["overall"]}


@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=4096)
def calibration_sweep(traj_dir: str, n_windows: int = 4096, eval_seed: int = 4242):
    """CPU coordinator: Part 1 on every checkpoint in a trajectory dir (<= 4 GPUs at once)."""
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    print(f"{len(ckpts)} checkpoints in {traj_dir}", flush=True)
    rows = list(calibration_ckpt.map(ckpts, kwargs={"n_windows": n_windows, "eval_seed": eval_seed}))
    rows.sort(key=lambda r: r["step"])
    for r in rows:
        o = r["overall"]
        print(f"step {r['step']:>6}  CE {o['CE']:.4f}  H_p {o['H_p']:.4f}  gap {o['gap_CE_minus_Hq']:+.4f}  "
              f"R2 {o['R2_CE_on_Hq']:.3f}  KL_k_q " + " ".join(f"{x:.3f}" for x in o["KL_k_q"]), flush=True)
    with open(f"{traj_dir}/calibration_sweep.json", "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return rows
