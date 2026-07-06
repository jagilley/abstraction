"""RHM latent-loop 2x2: target (token vs oracle-latent) x loop (open vs closed).

The A-lite diagnostic from the sleep-chunking line. We hold DGP / model / init /
eval fixed and cross two independent knobs:

  target ∈ {token, oracle-latent}   -- does the model get an UNDILUTED high-level
                                        signal (oracle_aux, thread_b) on top of NTP?
  loop   ∈ {open, closed}           -- is the a2a self-knowledge apparatus on
                                        (FM predicts the model's own future acts,
                                        injects them, gate-scaled local loss)?

  ┌──────────┬─────────────────────┬───────────────────────────┐
  │          │ token (no aux)      │ oracle-latent (+aux)      │
  ├──────────┼─────────────────────┼───────────────────────────┤
  │ open     │ ntp   (→ frontier)  │ ntp_aux    (→ BP)         │
  │ closed   │ ntp_cl              │ ntp_aux_cl   ← empty cell │
  └──────────┴─────────────────────┴───────────────────────────┘

The two knobs are kept as SEPARATE additive loss terms so the factorial is clean:

    wake loss = NTP  +  λ_aux·aux_loss(inter, oracle_labels)   [latent target]
                     +  λ_local·local_loss(residual, gate)      [self-knowledge loop]

Headline readout is NOT val loss -- it is what the FM residual and the SK-probe do
across the target axis. Prediction (MNIST-depth-1 / target-dilution framework):
the token-target residual is diffuse/high-rank (nothing legible to encode -> weak,
even inverted SK, per sparse_ratchet); the oracle-latent target makes the model
compute deep structured levels, so the residual goes low-rank / deep-η²-conditioned
and SK strengthens. If SK strengthens ONLY with the latent target, self-knowledge
and latent-targets are COUPLED (the latent target manufactures the structured
residual that self-knowledge feeds on); if SK is flat across the target axis, they
are orthogonal even at the residual level.

Honesty flag: the oracle-latent target uses ground-truth latent VALUES (Level-0
privilege). This is a DIAGNOSTIC -- "given the model can climb, does self-knowledge
interact with the climb" -- not a self-supervised escape. Dropping the privilege
(EMA self-distilled latent target) is Option B, run only if this shows coupling.

DGP: distinct-rule v=16, m=4, s=2, L=6 (occupancy 0.25) -- the thread_b regime with a
real learnable-but-unlearned frontier (M stalls ~d3.5; BP 0.98@d4, 0.80@root). Same
BP/greedy reference lines as RHM_DEEP_COMPOSITION.

Distillation extension (2026-07-06): a `+distill` suffix on any closed condition
(e.g. "ntp_aux_cl@0.0+distill") appends a sleep phase after wake: teacher = the
wake-final model WITH its co-trained FM+gate injection (frozen); student = the same
weights continued WITHOUT injection, trained on α·KL(per-token, student||teacher) +
(1−α)·CE(ground truth) (+ the aux term if the condition has it, so the latent anchor
stays on). This is consolidation WITHOUT residual-compression pressure — the
dissociation from local-loss consolidation (which compresses the residual and, on
token targets, inverts fresh-FM SK). All final measurements (probes/residual/SK/
ensemble) then run on the post-sleep model; the matching no-distill condition (same
seed → identical wake) provides the pre-sleep numbers.

With-injection probes (same date): for every closed condition we additionally
measure val and per-level recovery on INJECTED forward passes at wake-final (and
injected val again post-sleep for distill conditions) — the direct test of the
"injection explores beyond the standalone frontier" claim, previously unmeasured.

Run:
  modal run --detach -m rhm.rhm_latent_loop::latent_loop
  # token-only 2x1 quick check:
  modal run --detach -m rhm.rhm_latent_loop::latent_loop --conditions "ntp,ntp_cl"
  # distillation-consolidation ablation (latent target, m4):
  modal run --detach -m rhm.rhm_latent_loop::latent_loop \
      --conditions "ntp_aux,ntp_aux_cl@1.0,ntp_aux_cl@0.0,ntp_aux_cl@0.0+distill,ntp_aux_cl@1.0+distill" \
      --ensemble-n 4 --tag distill
"""

import glob
import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-latent-loop", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ======================================================================
# Data + probe helpers (from rhm_thread_b / rhm_sparse_ratchet)
# ======================================================================

def _generate_with_traces(rules, n_sequences, seed):
    """Aligned leaf sequences + true latent feature and rule-choice at every node.
    level_features[ell]: (n_seq, s^ell) parent features; [L] = leaves.
    level_rules[ell]:    (n_seq, s^ell) synonym choice used at that node."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features, level_rules = [], []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        nxt = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
    level_features.append(current.copy())  # leaves
    return current, level_features, level_rules


def _probe_acc(X, y, v, device, steps, lr, hidden=0, wd=1e-4, train_frac=0.7):
    """Decodability of label y from X. hidden=0 -> linear; hidden>0 -> 1-layer MLP."""
    import torch
    import torch.nn as nn
    n = X.shape[0]
    split = int(train_frac * n)
    mu, sd = X[:split].mean(0, keepdim=True), X[:split].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr, Xte, yte = Xn[:split], y[:split], Xn[split:], y[split:]
    if hidden:
        clf = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, v)).to(device)
    else:
        clf = nn.Linear(X.shape[1], v).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        te = (clf(Xte).argmax(1) == yte).float().mean().item()
    return te


def _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level, block_names, L, v,
                       device, probe_steps, probe_lr, mlp_hidden, mlp_steps, label):
    """Probe (checkpoint x block x level) with linear + MLP probes; best-over-blocks."""
    import torch

    def last_pos_acts(step):
        sd = torch.load(f"{ckpt_dir}/ckpt_step{step}.pt", map_location=device,
                        weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        acc = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, len(eval_x), 256):
                _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
                for b in block_names:
                    acc[b].append(inter[b][:, -1, :].float())
        return {b: torch.cat(acc[b], 0) for b in block_names}

    out = {}
    for step in saved:
        acts = last_pos_acts(step)
        lin_best, mlp_best = {}, {}
        for ell in range(L):
            lin_best[ell] = max(
                _probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr)
                for b in block_names)
            if mlp_hidden:
                mlp_best[ell] = max(
                    _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, 1e-3,
                               hidden=mlp_hidden, wd=1e-3) for b in block_names)
        out[step] = {"linear_best": lin_best, "mlp_best": mlp_best}
        print(f"  [{label}] step {step:6d} | LIN best:  " +
              "  ".join(f"d{L-ell}:{lin_best[ell]:.3f}" for ell in range(L)))
        if mlp_hidden:
            print(f"  [{label}] step {step:6d} | MLP best:  " +
                  "  ".join(f"d{L-ell}:{mlp_best[ell]:.3f}" for ell in range(L)))
    return out


def _compute_hierarchy_eta2(residuals_np, level_features, level_rules, s, L):
    """Fraction of residual (last-token) variance explained by the true latent
    feature / synonym choice at each hierarchy level."""
    import numpy as np
    n_seq, seq_len, d_model = residuals_np.shape
    res_last = residuals_np[:, -1, :]
    last_mean = res_last.mean(axis=0)
    ss_total = float(np.sum((res_last - last_mean) ** 2))
    if ss_total < 1e-12:
        return {f"level_{ell}": {"feature_eta2": 0.0, "rule_eta2": 0.0}
                for ell in range(L)}

    def eta2_between(labels):
        ss = 0.0
        for val in np.unique(labels):
            mask = labels == val
            gm = res_last[mask].mean(axis=0)
            ss += float(mask.sum()) * float(np.sum((gm - last_mean) ** 2))
        return ss / ss_total

    results = {}
    for ell in range(L):
        anc = (seq_len - 1) // (s ** (L - ell))
        results[f"level_{ell}"] = {
            "feature_eta2": eta2_between(level_features[ell][:, anc]),
            "rule_eta2": eta2_between(level_rules[ell][:, anc]),
        }
    return results


def _residual_stats(model, fm, eval_x, level_features, level_rules,
                    predict_from, predict_to, n_embd, s, L, batch_size, device, label):
    """FM residual: cosine, norm, effective rank, top1 PC, per-level η²."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    model.eval(); fm.eval()
    all_res, all_cos = [], []
    with torch.no_grad():
        for i in range(0, len(eval_x), batch_size):
            batch = eval_x[i:i + batch_size]
            if batch.shape[0] < 2:
                continue
            _, _, vi = model(batch, return_intermediates=True)
            pred = fm(vi[predict_from])
            tgt = vi[predict_to]
            all_res.append((tgt - pred).cpu().numpy())
            all_cos.append(float(F.cosine_similarity(pred, tgt, dim=-1).mean()))
    res_np = np.concatenate(all_res, axis=0)
    mean_cos = float(np.mean(all_cos))
    mean_norm = float(np.sqrt((res_np ** 2).sum(axis=-1)).mean())

    res_flat = res_np.reshape(-1, n_embd)
    n = min(50000, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], n, replace=False)
    sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    Sn = S / S.sum()
    eff_rank = float(np.exp(-np.sum(Sn * np.log(Sn + 1e-30))))
    top1 = float((S[0] ** 2) / (S ** 2).sum())

    nc = res_np.shape[0]
    eta2 = _compute_hierarchy_eta2(res_np, [lf[:nc] for lf in level_features],
                                   [lr[:nc] for lr in level_rules], s=s, L=L)
    stats = {"fwd_cosine": mean_cos, "res_norm": mean_norm,
             "eff_rank": eff_rank, "eff_rank_pct": eff_rank / n_embd * 100,
             "top1_pct": top1 * 100, "hierarchy_eta2": eta2}
    eta_str = " ".join(f"d{L-ell}η²={eta2[f'level_{ell}']['feature_eta2']:.3f}"
                       for ell in range(L))
    print(f"  [{label}] cos={mean_cos:.4f} norm={mean_norm:.2f} "
          f"rank={eff_rank:.1f}/{n_embd} ({eff_rank/n_embd*100:.1f}%) "
          f"top1={top1*100:.1f}% | {eta_str}")
    return stats


def _sk_probes(model, fm, eval_x, predict_from, predict_to, n_embd, n_layer,
               batch_size, device, probe_steps, seed, label):
    """Self-knowledge: R² of the FM residual DIRECTION from block activations."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    model.eval(); fm.eval()
    probe_layers = ["post_block0", f"post_block{n_layer - 1}"]
    acts = {k: [] for k in probe_layers}
    residuals = []
    with torch.no_grad():
        for i in range(0, len(eval_x), batch_size):
            batch = eval_x[i:i + batch_size]
            _, _, vi = model(batch, return_intermediates=True)
            pred = fm(vi[predict_from])
            residuals.append((vi[predict_to] - pred).reshape(-1, n_embd).cpu())
            for k in probe_layers:
                acts[k].append(vi[k].reshape(-1, n_embd).cpu())
    all_res = torch.cat(residuals)
    for k in probe_layers:
        acts[k] = torch.cat(acts[k])

    n_total = all_res.shape[0]
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[:n_train], perm[n_train:]
    r2 = {}
    for lk in probe_layers:
        X = acts[lk]
        probe = nn.Linear(n_embd, n_embd).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr, Ytr = X[tr].to(device), all_res[tr].to(device)
        Xte, Yte = X[te].to(device), all_res[te].to(device)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            si = torch.randint(n_train, (bs,))
            F.mse_loss(probe(Xtr[si]), Ytr[si]).backward()
            opt.step(); opt.zero_grad()
        with torch.no_grad():
            mse = F.mse_loss(probe(Xte), Yte).item()
            var = Yte.var().item()
        r2[lk] = 1.0 - mse / var if var > 0 else 0.0
        del probe, Xtr, Ytr, Xte, Yte
        torch.cuda.empty_cache()
    print(f"  [{label}] SK R²: " + " | ".join(f"{k}={v_:.3f}" for k, v_ in r2.items()))
    return r2


def _ensemble_agreement(model, make_fm, get_ntp_batch, fresh_fm_steps, fwd_lr,
                        eval_x, predict_from, predict_to, n_embd, batch_size, device,
                        n_fm, base_seed, label):
    """Train n_fm fresh FMs (different seeds) on the frozen model and ask whether they
    leave the SAME residual or different ones. pairwise_cos on input-centered last-token
    residuals: high => the residual is determined by the INPUT (FM-invariant, DGP-aligned),
    low => by the FM (idiosyncratic noise). This is the direct test of why latent-target
    self-knowledge generalizes across FMs while token-target self-knowledge doesn't."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    for p in model.parameters():
        p.requires_grad = False
    resid = []
    for k in range(n_fm):
        torch.manual_seed(base_seed + 1000 * (k + 1))
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        gen = torch.Generator().manual_seed(base_seed + 2000 * (k + 1))
        for _ in range(fresh_fm_steps):
            fm.train()
            x, _ = get_ntp_batch(gen)
            with torch.no_grad():
                _, _, vi = model(x, return_intermediates=True)
                src, tgt = vi[predict_from], vi[predict_to]
            F.mse_loss(fm(src), tgt).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        fm.eval()
        rk = []
        with torch.no_grad():
            for i in range(0, len(eval_x), batch_size):
                b = eval_x[i:i + batch_size]
                _, _, vi = model(b, return_intermediates=True)
                rk.append((vi[predict_to] - fm(vi[predict_from]))[:, -1, :].cpu())
        resid.append(torch.cat(rk))
        del fm, opt
        torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad = True
    R = torch.stack(resid)                               # (n_fm, N, d)
    Rc = R - R.mean(dim=1, keepdim=True)                 # center per FM over examples
    Rn = F.normalize(Rc, dim=-1)
    cps = [float((Rn[i] * Rn[j]).sum(-1).mean())
           for i in range(n_fm) for j in range(i + 1, n_fm)]
    pairwise = float(np.mean(cps))
    shared = float(R.mean(0).norm(dim=-1).mean() / R.norm(dim=-1).mean())
    print(f"  [{label}] ensemble n={n_fm}: pairwise_cos={pairwise:.3f} "
          f"shared_norm_frac={shared:.3f}")
    return {"pairwise_cos": pairwise, "shared_norm_frac": shared, "n_fm": n_fm}


# ======================================================================
# Main 2x2
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def latent_loop(
    # DGP
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    # Model
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # FM (self-knowledge loop) -- matched-head, modest capacity; applied identically
    # (fresh) to ALL conditions for measurement, so capacity affects absolute numbers
    # but not the token-vs-latent contrast.
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # Training
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    lam_aux: float = 1.0, lam_local: float = 1.0, ug_hidden: int = 64,
    pool_size: int = 200000, data_seed: int = 7,
    # which of {ntp, ntp_aux, ntp_cl, ntp_aux_cl}; closed conditions accept a
    # "@<lam_local>" override and a "+distill" suffix (post-wake sleep phase)
    conditions: str = "ntp,ntp_aux,ntp_cl,ntp_aux_cl",
    # Distillation sleep phase (a2a_forward/distillation.py recipe, per-token KL)
    distill_steps: int = 5000, distill_lr: float = 1e-4, distill_alpha: float = 0.5,
    # Measurement
    fresh_fm_steps: int = 3000, eval_interval: int = 1000,
    ckpt_steps: str = "0,10000",
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, sk_probe_steps: int = 500,
    ensemble_n: int = 0, tag: str = "",
    bp_line: str = "d1 .998 d2 .998 d3 .995 d4 .989 d5 .955 root .800",
    greedy_line: str = "d1 .887 d2 .845 d3 .751 d4 .708 d5 .476",
    seed: int = 42,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L = depth
    T = s ** L
    key = tb_key(v, s, L, m)
    chance = 1.0 / v
    cib = int(predict_from.replace("post_block", ""))
    cond_list = [c.strip() for c in conditions.split(",")]
    want_ckpts = sorted({min(int(x), n_steps - 1) for x in ckpt_steps.split(",")}
                        | {n_steps - 1})
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    print(f"{'='*72}\nRHM LATENT-LOOP 2x2  {key}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1)={m/v**(s-1):.3f}  chance={chance:.4f}  T={T}")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D  {predict_from}->{predict_to} "
          f"inject@{inject_after_block}")
    print(f"  conditions: {cond_list}   n_steps={n_steps}  λ_aux={lam_aux} λ_local={lam_local}")
    print(f"{'='*72}")

    # --- training pool: aligned sequences + per-position ancestor labels ---
    print(f"Generating pool ({pool_size:,} seqs)...")
    pool_seqs, pool_lf, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))                    # (P,T) cpu
    pool_anc = [torch.from_numpy(pool_lf[ell].astype(np.int64)) for ell in range(L)]
    anc_idx = [torch.arange(T) // (s ** (L - ell)) for ell in range(L)]       # pos->node
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - ell)) == 0])
               for ell in range(L)]                                          # completed pos
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]

    # --- eval set (probing + residual + SK), with full traces for η² ---
    eval_seqs, eval_lf, eval_lr = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(eval_lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    # Flat concatenated corpus (phase diversity) for NTP -> EVERY position is
    # NTP-supervised, fixing the aligned-last-position artifact that depressed the
    # token conditions. Aux still uses aligned labeled sequences (ancestor labels
    # are only defined there), applied on a SEPARATE forward pass.
    corpus = pool_x.reshape(-1)                                   # (P*T,) cpu
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]                     # (B,T) windows
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    def get_aux_batch(gen):
        idx = torch.randint(0, pool_size, (batch_size,), generator=gen)
        x = pool_x[idx].to(device)
        labels = [pool_anc[ell][idx][:, anc_idx[ell]].to(device) for ell in range(L)]
        return x, labels

    def aux_loss(inter, labels, aux_heads):
        head_out = {b: aux_heads[b](inter[b]).view(batch_size, T, L, v) for b in sup_blocks}
        per_level = []
        for ell in range(L):
            se = spanend[ell]
            ces = [F.cross_entropy(head_out[b][:, se, ell, :].reshape(-1, v),
                                   labels[ell][:, se].reshape(-1)) for b in sup_blocks]
            per_level.append(torch.stack(ces).mean())
        return torch.stack(per_level).mean()

    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(nn.Linear(2 * d_model, d_hidden), nn.GELU(),
                                          nn.Linear(d_hidden, d_model))
            self.projection = nn.Linear(d_model, d_model)
            for layer in (self.gate_net[-1], self.projection):
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

        def forward(self, activations, fwd_pred):
            gw = torch.sigmoid(self.gate_net(torch.cat([activations, fwd_pred], dim=-1)))
            return gw * self.projection(fwd_pred), gw

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    # --- shared init (identical starting weights across all conditions) ---
    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    def eval_ntp(model):
        model.eval()
        gen = torch.Generator().manual_seed(eval_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(10):
                x, y = get_ntp_batch(gen)
                _, loss = model(x, y)
                tot += loss.item()
        return tot / 10

    def eval_ntp_injected(model, fm_, ugate_):
        """Standalone eval_ntp's twin, with the co-trained FM+gate injection active."""
        model.eval(); fm_.eval(); ugate_.eval()
        gen = torch.Generator().manual_seed(eval_seed + 5)

        def cb(act):
            inj, _ = ugate_(act, fm_(act))
            return inj

        tot = 0.0
        with torch.no_grad():
            for _ in range(10):
                x, y = get_ntp_batch(gen)
                _, loss = model(x, y, cerebellar_fn=cb, cerebellar_input_block=cib,
                                cerebellar_inject_block=inject_after_block)
                tot += loss.item()
        return tot / 10

    def probe_injected(model, fm_, ugate_, label):
        """Per-level recovery (linear+MLP, best-over-blocks) on INJECTED forward
        passes -- what the injection contributes above the standalone frontier."""
        model.eval(); fm_.eval(); ugate_.eval()

        def cb(act):
            inj, _ = ugate_(act, fm_(act))
            return inj

        acc = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, len(eval_x), 256):
                _, _, inter = model(eval_x[i:i + 256], return_intermediates=True,
                                    cerebellar_fn=cb, cerebellar_input_block=cib,
                                    cerebellar_inject_block=inject_after_block)
                for b in block_names:
                    acc[b].append(inter[b][:, -1, :].float())
        acts = {b: torch.cat(acc[b], 0) for b in block_names}
        lin_best, mlp_best = {}, {}
        for ell in range(L):
            lin_best[ell] = max(
                _probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr)
                for b in block_names)
            if mlp_hidden:
                mlp_best[ell] = max(
                    _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, 1e-3,
                               hidden=mlp_hidden, wd=1e-3) for b in block_names)
        shown = mlp_best if mlp_hidden else lin_best
        print(f"  [{label}] INJ {'MLP' if mlp_hidden else 'LIN'} best:  " +
              "  ".join(f"d{L-ell}:{shown[ell]:.3f}" for ell in range(L)))
        return {"linear_best": lin_best, "mlp_best": mlp_best}

    def train_fresh_fm(model):
        """Fresh FM (matched seed/capacity) on the frozen final model -- the
        apples-to-apples residual/SK measurement instrument for every condition."""
        for p in model.parameters():
            p.requires_grad = False
        torch.manual_seed(seed + 911)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        gen = torch.Generator().manual_seed(seed + 912)
        for _ in range(fresh_fm_steps):
            fm.train()
            x, _ = get_ntp_batch(gen)
            with torch.no_grad():
                _, _, vi = model(x, return_intermediates=True)
                src, tgt = vi[predict_from], vi[predict_to]
            F.mse_loss(fm(src), tgt).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            p.requires_grad = True
        return fm

    # ==================================================================
    # Run each condition
    # ==================================================================
    results = {}
    for cond in cond_list:
        base_cond = cond[:-len("+distill")] if cond.endswith("+distill") else cond
        use_distill = cond.endswith("+distill")
        use_aux = "aux" in base_cond
        use_loop = "cl" in base_cond
        assert not (use_distill and not use_loop), f"{cond}: +distill needs a closed loop"
        cl_lam = float(base_cond.split("@")[1]) if "@" in base_cond else lam_local
        print(f"\n{'='*60}\n  CONDITION: {cond}  (aux={use_aux}, loop={use_loop}, "
              f"λ_local={cl_lam}, distill={use_distill})\n{'='*60}")

        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_state)
        main_params = list(model.parameters())

        aux_heads = None
        if use_aux:
            aux_heads = nn.ModuleDict(
                {b: nn.Linear(n_embd, L * v) for b in sup_blocks}).to(device)
            main_params += list(aux_heads.parameters())
        ugate = fm = opt_fwd = None
        if use_loop:
            ugate = UnifiedGate(n_embd, ug_hidden).to(device)
            main_params += list(ugate.parameters())
            fm = make_fm()
            opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)

        # Namespace ckpts by tag so concurrent runs with overlapping condition
        # names (e.g. a distill-alpha control) can't clobber each other's files.
        ckpt_dir = (f"{DATA_DIR}/rhm_latent_loop/{key}/"
                    f"{(tag + '_') if tag else ''}{cond.replace('@', '_').replace('+', '_')}")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_gen = torch.Generator().manual_seed(seed + 1)
        aux_gen = torch.Generator().manual_seed(seed + 2)
        saved = []

        for step in range(n_steps):
            if step in want_ckpts:
                torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{step}.pt")
                saved.append(step)
            model.train()
            if use_loop:
                fm.train(); ugate.train()
            x, y = get_ntp_batch(train_gen)          # phase-diverse: all positions supervised

            cache = {}
            if use_loop:
                def ug_cb(act, _c=cache):
                    fp = fm(act.detach())
                    _c["pred"] = fp
                    inj, gw = ugate(act.detach(), fp.detach())
                    _c["gw"] = gw
                    return inj
                logits, _, inter = model(
                    x, return_intermediates=True, cerebellar_fn=ug_cb,
                    cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block)
            else:
                logits, _, inter = model(x, return_intermediates=True)

            ntp = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
            loss = ntp
            if use_aux:                               # oracle-latent target on aligned batch
                xa, labels = get_aux_batch(aux_gen)
                _, _, inter_a = model(xa, return_intermediates=True)
                loss = loss + lam_aux * aux_loss(inter_a, labels, aux_heads)
            local = None
            if use_loop:
                fwd_pred = cache["pred"]
                tgt_acts = inter[predict_to]
                r = tgt_acts - fwd_pred.detach()                 # keeps tgt in graph
                local = (cache["gw"].detach().mean() * r ** 2).mean()
                loss = loss + cl_lam * local

            opt_main.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            if use_loop:
                fwd_loss = F.mse_loss(fwd_pred, tgt_acts.detach())
                opt_fwd.zero_grad()
                fwd_loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt_fwd.step()

            if step % eval_interval == 0 or step == n_steps - 1:
                vl = eval_ntp(model)
                extra = f" local={local.item():.5f}" if local is not None else ""
                print(f"    step {step:6d}: ntp={ntp.item():.4f} val={vl:.4f}{extra}")

        torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{n_steps-1}.pt")
        saved = sorted(set(saved + [n_steps - 1]))
        volume.commit()

        # --- with-injection measurements at wake-final (co-trained FM+gate) ---
        injected = None
        if use_loop:
            injected = {"val_injected": eval_ntp_injected(model, fm, ugate),
                        "probes_injected": probe_injected(model, fm, ugate,
                                                          cond + "|inj")}
            print(f"  [{cond}] wake-final val: injected={injected['val_injected']:.4f} "
                  f"standalone={eval_ntp(model):.4f}")

        # --- sleep phase: distill teacher(injection) -> student(standalone) ---
        if use_distill:
            print(f"  [{cond}] SLEEP: {distill_steps} steps, "
                  f"α={distill_alpha} lr={distill_lr} (aux kept: {use_aux})")
            teacher = GPT(v, T, n_layer, n_head, n_embd).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            fm.eval(); ugate.eval()
            for p in list(fm.parameters()) + list(ugate.parameters()):
                p.requires_grad = False

            def teacher_cb(act):
                inj, _ = ugate(act, fm(act))
                return inj

            sleep_params = list(model.parameters())
            if use_aux:
                sleep_params += list(aux_heads.parameters())
            opt_sleep = torch.optim.AdamW(sleep_params, lr=distill_lr,
                                          weight_decay=weight_decay)
            sleep_gen = torch.Generator().manual_seed(seed + 31)
            aux_sleep_gen = torch.Generator().manual_seed(seed + 32)
            for dstep in range(distill_steps):
                model.train()
                x, y = get_ntp_batch(sleep_gen)
                with torch.no_grad():
                    t_logits, _, _ = teacher(
                        x, return_intermediates=True, cerebellar_fn=teacher_cb,
                        cerebellar_input_block=cib,
                        cerebellar_inject_block=inject_after_block)
                s_logits, _, _ = model(x, return_intermediates=True)
                ce = F.cross_entropy(s_logits.reshape(-1, v), y.reshape(-1))
                # per-token KL (reshape first) so alpha genuinely balances KL vs CE
                kl = F.kl_div(F.log_softmax(s_logits.reshape(-1, v), dim=-1),
                              F.softmax(t_logits.reshape(-1, v), dim=-1),
                              reduction="batchmean")
                sloss = distill_alpha * kl + (1 - distill_alpha) * ce
                if use_aux:
                    xa, labels = get_aux_batch(aux_sleep_gen)
                    _, _, inter_a = model(xa, return_intermediates=True)
                    sloss = sloss + lam_aux * aux_loss(inter_a, labels, aux_heads)
                opt_sleep.zero_grad()
                sloss.backward()
                torch.nn.utils.clip_grad_norm_(sleep_params, 1.0)
                opt_sleep.step()
                if dstep % eval_interval == 0 or dstep == distill_steps - 1:
                    print(f"    sleep {dstep:6d}: kl={kl.item():.4f} "
                          f"ce={ce.item():.4f} val={eval_ntp(model):.4f}")
            post_step = n_steps - 1 + distill_steps
            torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{post_step}.pt")
            saved = sorted(set(saved + [post_step]))
            injected["val_injected_postsleep"] = eval_ntp_injected(model, fm, ugate)
            print(f"  [{cond}] post-sleep val: standalone={eval_ntp(model):.4f} "
                  f"injected={injected['val_injected_postsleep']:.4f}")
            del teacher, opt_sleep
            torch.cuda.empty_cache()
            volume.commit()

        # --- measurements (all on a fresh, matched FM for apples-to-apples) ---
        model.eval()
        probes = _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level, block_names,
                                    L, v, device, probe_steps, probe_lr, mlp_hidden,
                                    mlp_steps, cond)
        meas_fm = train_fresh_fm(model)
        res = _residual_stats(model, meas_fm, eval_x, eval_lf, eval_lr, predict_from,
                              predict_to, n_embd, s, L, batch_size, device, cond)
        sk = _sk_probes(model, meas_fm, eval_x, predict_from, predict_to, n_embd,
                        n_layer, batch_size, device, sk_probe_steps, seed, cond)
        # For loop conditions, also read SK off the CO-TRAINED FM (the one the model
        # actually learned to complement), not only the post-hoc fresh FM.
        sk_ct = None
        if use_loop:
            sk_ct = _sk_probes(model, fm, eval_x, predict_from, predict_to, n_embd,
                               n_layer, batch_size, device, sk_probe_steps, seed,
                               cond + "|cotrained")

        ens = None
        if ensemble_n > 0:
            ens = _ensemble_agreement(model, make_fm, get_ntp_batch, fresh_fm_steps,
                                      fwd_lr, eval_x, predict_from, predict_to, n_embd,
                                      batch_size, device, ensemble_n, seed, cond)
        results[cond] = {"probes": probes, "residual_stats": res, "self_knowledge": sk,
                         "self_knowledge_cotrained": sk_ct, "ensemble": ens,
                         "injected": injected, "final_val": eval_ntp(model)}
        del model, meas_fm, opt_main
        if use_loop:
            del fm, ugate, opt_fwd
        torch.cuda.empty_cache()

    # ==================================================================
    # Summary tables
    # ==================================================================
    final = n_steps - 1
    print(f"\n{'='*80}\nSUMMARY  {key}  (chance={chance:.4f})")
    print(f"BP ceiling (this regime): {bp_line}")
    print(f"greedy floor:             {greedy_line}")
    print(f"{'='*80}")

    print(f"\n=== Per-level recovery (MLP best-over-blocks, last ckpt = post-sleep for +distill) ===")
    print(f"{'cond':>24} | " + "  ".join(f"d{L-ell}" for ell in range(L)) + "   val")
    for cond in cond_list:
        pr = results[cond]["probes"][max(results[cond]["probes"])]
        best = pr.get("mlp_best", pr["linear_best"])
        print(f"{cond:>24} | " + "  ".join(f"{best[ell]:.3f}" for ell in range(L)) +
              f"  {results[cond]['final_val']:.3f}")

    print(f"\n=== With-injection (co-trained FM+gate) vs standalone, wake-final ===")
    print(f"{'cond':>24} | {'val_inj':>7} {'val_inj_postsleep':>17} | "
          + "  ".join(f"d{L-ell}" for ell in range(L)) + "  (injected MLP best)")
    for cond in cond_list:
        inj = results[cond].get("injected")
        if not inj:
            continue
        best = inj["probes_injected"].get("mlp_best", inj["probes_injected"]["linear_best"])
        ps = f"{inj['val_injected_postsleep']:17.3f}" if "val_injected_postsleep" in inj \
             else f"{'--':>17}"
        print(f"{cond:>24} | {inj['val_injected']:7.3f} {ps} | "
              + "  ".join(f"{best[ell]:.3f}" for ell in range(L)))

    print(f"\n=== FM residual structure (fresh matched FM, final) ===")
    print(f"{'cond':>24} | {'cos':>6} {'norm':>6} {'rank%':>6} {'top1%':>6} | "
          + "  ".join(f"d{L-ell}η²" for ell in range(L)))
    for cond in cond_list:
        rs = results[cond]["residual_stats"]
        e = rs["hierarchy_eta2"]
        print(f"{cond:>24} | {rs['fwd_cosine']:6.3f} {rs['res_norm']:6.2f} "
              f"{rs['eff_rank_pct']:6.1f} {rs['top1_pct']:6.1f} | "
              + "  ".join(f"{e[f'level_{ell}']['feature_eta2']:.3f}" for ell in range(L)))

    bN = f"post_block{n_layer-1}"
    print(f"\n=== Self-knowledge R² (fresh FM | co-trained FM), final ===")
    print(f"{'cond':>24} | {'block0':>8} {bN:>10} | {'ct_block0':>9} {'ct_'+bN:>12}")
    for cond in cond_list:
        sk = results[cond]["self_knowledge"]
        ct = results[cond].get("self_knowledge_cotrained")
        cts = (f"{ct['post_block0']:9.3f} {ct[bN]:12.3f}") if ct else f"{'--':>9} {'--':>12}"
        print(f"{cond:>24} | {sk['post_block0']:8.3f} {sk[bN]:10.3f} | {cts}")

    # The MNIST/language self-knowledge quantity is the CL-OL gap (positive there).
    # OL baseline is ntp for token-loop conditions, ntp_aux for latent-loop conditions.
    print(f"\n=== Δ SK (CL − OL) per closed condition, {bN} [MNIST/lang: POSITIVE] ===")
    for cond in cond_list:
        if "cl" not in cond:
            continue
        ol = "ntp_aux" if "aux" in cond else "ntp"
        if ol in results:
            d = results[cond]["self_knowledge"][bN] - results[ol]["self_knowledge"][bN]
            print(f"  {cond:>24} (vs {ol}): Δ{bN}={d:+.3f}")

    print(f"\n=== SWEEP SUMMARY: root(d6) recov | val | resNorm | d6η² | SK{bN} | ensemble ===")
    hdr = f"{'cond':>24} | {'root':>5} {'val':>6} {'resNrm':>7} {'d6η²':>6} {'SKb7':>7}"
    if ensemble_n > 0:
        hdr += f" | {'ens_cos':>7} {'shared':>7}"
    print(hdr)
    for cond in cond_list:
        pr = results[cond]["probes"]; st = max(pr, key=lambda k: int(k))
        best = pr[st].get("mlp_best", pr[st]["linear_best"])
        rs = results[cond]["residual_stats"]
        row = (f"{cond:>24} | {best[0]:5.3f} {results[cond]['final_val']:6.3f} "
               f"{rs['res_norm']:7.2f} "
               f"{rs['hierarchy_eta2']['level_0']['feature_eta2']:6.3f} "
               f"{results[cond]['self_knowledge'][bN]:7.3f}")
        e = results[cond].get("ensemble")
        if ensemble_n > 0 and e:
            row += f" | {e['pairwise_cos']:7.3f} {e['shared_norm_frac']:7.3f}"
        print(row)

    save_dir = f"{DATA_DIR}/rhm_latent_loop"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{key}_{n_layer}L{n_head}H{n_embd}D_S{n_steps}{('_'+tag) if tag else ''}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_head=n_head,
                                  n_embd=n_embd, predict_from=predict_from,
                                  predict_to=predict_to, inject_after_block=inject_after_block,
                                  fwd=f"{fwd_n_layer}L{fwd_n_head}H{fwd_d_head}D",
                                  n_steps=n_steps, lam_aux=lam_aux, lam_local=lam_local,
                                  distill_steps=distill_steps, distill_lr=distill_lr,
                                  distill_alpha=distill_alpha, chance=chance),
                   "conditions": results}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return results


@app.local_entrypoint()
def main(conditions: str = "ntp,ntp_aux,ntp_cl,ntp_aux_cl", n_steps: int = 20000):
    latent_loop.remote(conditions=conditions, n_steps=n_steps)
    print("done")
