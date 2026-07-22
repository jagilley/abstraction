"""The Confabulation Test: does a self-report track the implementation or a self-theory?

Design doc: ideas/confabulation_test.md
Substrate:  RHM_LATENT_LOOP_README.md -- this file forks its `ntp_aux_cl` / `ntp_aux`
            wake recipe VERBATIM (same DGP, model, FM, gate, init, losses) and adds a
            report battery on top. `ntp_aux*` is not optional: latent-loop showed the
            token target has *inverted* self-knowledge on RHM (fresh-FM SK -0.67), so a
            report experiment on `ntp_cl` would test the channel on a model with nothing
            to report. Only the latent target gives generalizable SK (+0.78 CL / +0.56 OL).

THE CLAIM
  a_j = FM(a_i) + r  splits self-knowledge into the part a compressed model of M can
  anticipate and its complement. FM(a_i) is, by construction, everything a *self-theory*
  could produce -- so confabulation lives in the range of the self-model, and whatever
  cannot be confabulated lives in the residual r.

  A signal S is introspective for M iff (1) S is a function of M's internal state,
  (2) S is not *cheaply* recoverable from M's input-output map, (3) M's behaviour is
  causally sensitive to S. (2) is the discriminator: calibration/entropy are functions
  of the I/O map (third-person accessible -> self-inference); r is a function of the
  *implementation* (two models with identical I/O behaviour have different r).

WHAT THIS SCRIPT MEASURES
  Report targets, all emitted from post_block{n_layer-1} on standalone forward passes
  (fresh-FM instrument, per latent-loop convention):
    IMPL  -- K-way cluster of the direction of M's own FM residual r_t   [implementation]
    BEHAV -- is M's own next-token argmax correct / entropy quartile     [I/O map]
    WORLD -- ground-truth DGP ancestor label at level `world_level`      [input]

  Test 1  observer ladder   O_input(c) / O_io(c) (tokens + M's own logits) / O_act
                            (ceiling: post_block0 -> target, i.e. FM + a probe), each
                            swept over capacity. The gap M - O_io on IMPL is the
                            introspective advantage; on BEHAV/WORLD it should vanish.
  Test 2  channel ablation  substitute a_j at report time: shuffle_r / shuffle_p
                            (distribution-preserving, position-destroying) and zero_r /
                            zero_p. IMPL should collapse under shuffle_r only.
  Test 3  behaviour-null    steer a_j along residual PCs sorted by their causal effect
          steering          on the logits. Perturbing along behaviour-NULL residual
                            directions should move the IMPL report while leaving
                            behaviour (KL) untouched -- a self-theory cannot do this.
  Test 4  confabulator      a report head trained AND evaluated with access restricted
                            to FM(a_i) -- right target, theory-only access. Its accuracy
                            is the confabulation ceiling; IMPL_full - IMPL_confab is the
                            introspective margin. Without this control a null on Test 1
                            is uninterpretable.

Run:
  # smoke (a few minutes, everything tiny -- checks wiring only, numbers meaningless)
  modal run -m rhm.rhm_confabulation::confabulation_test --smoke

  # headline: the loop arm (CL vs OL), full battery
  modal run --detach -m rhm.rhm_confabulation::confabulation_test \
      --conditions "ntp_aux_cl,ntp_aux" --tag main
"""

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
app = modal.App("rhm-confabulation", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ======================================================================
# Data helper (verbatim from rhm_latent_loop._generate_with_traces)
# ======================================================================

def _generate_with_traces(rules, n_sequences, seed):
    """Aligned leaf sequences + true latent feature and rule-choice at every node.
    level_features[ell]: (n_seq, s^ell) parent features; [L] = leaves."""
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
    level_features.append(current.copy())
    return current, level_features, level_rules


# ======================================================================
# Small utilities
# ======================================================================

def _kmeans_fit(X, K, iters=25, seed=0):
    """Spherical k-means on row-normalized X (torch, GPU). Fit on TRAIN rows only;
    test rows are assigned by `_kmeans_assign` so the label definition never sees them."""
    import torch
    Xn = torch.nn.functional.normalize(X, dim=-1)
    g = torch.Generator(device=Xn.device).manual_seed(seed)
    C = Xn[torch.randperm(Xn.shape[0], generator=g, device=Xn.device)[:K]].clone()
    for _ in range(iters):
        labels = _kmeans_assign(Xn, C)
        for k in range(K):
            sel = labels == k
            if sel.any():
                C[k] = torch.nn.functional.normalize(Xn[sel].mean(0), dim=-1)
    return C


def _kmeans_assign(X, C, chunk=65536):
    """Nearest-centroid (cosine) assignment, chunked so no full N x K matrix is built."""
    import torch
    Xn = torch.nn.functional.normalize(X, dim=-1)
    return torch.cat([(Xn[i:i + chunk] @ C.T).argmax(dim=-1)
                      for i in range(0, Xn.shape[0], chunk)])


def _balance(labels, K):
    """Majority-class fraction -- the trivial baseline any head must beat."""
    import torch
    counts = torch.bincount(labels, minlength=K).float()
    return float(counts.max() / counts.sum())


def _eta2_by_level(res, rep_lf, s, L, T):
    """Fraction of residual variance explained by the ground-truth ancestor feature at
    each hierarchy level. A residual that is a genuine DGP/capacity gap is conditioned on
    the hierarchy; architectural-mismatch NOISE is not. Reported as d{L-ell} (d{L}=root).

    res: (N, T, d) cpu. Returns {d_label: eta2}."""
    import numpy as np
    import torch
    out = {}
    tot = float(((res - res.mean(dim=(0, 1), keepdim=True)) ** 2).sum())
    for ell in range(L):
        lab = torch.from_numpy(
            rep_lf[ell][:, np.arange(T) // (s ** (L - ell))].astype(np.int64))  # (N,T)
        flat_r = res.reshape(-1, res.shape[-1])
        flat_l = lab.reshape(-1)
        gmean = flat_r.mean(0, keepdim=True)
        between = 0.0
        for g in torch.unique(flat_l):
            sel = flat_l == g
            between += float(int(sel.sum()) * ((flat_r[sel].mean(0) - gmean) ** 2).sum())
        out[f"d{L - ell}"] = between / tot if tot > 0 else 0.0
    return out


def _ensemble_cos(residuals):
    """Input-centered pairwise cosine between the residuals left by INDEPENDENT fresh
    FMs (different seeds) on the same frozen model. High => the residual is determined by
    the input (an FM-invariant DGP/capacity gap that any FM misses identically); low =>
    it is determined by the particular FM, i.e. idiosyncratic architectural-mismatch
    noise. This is the discriminator between 'genuine computational gap' and 'junk'.
    (Recipe from rhm_latent_loop._ensemble_agreement.)"""
    import torch
    import torch.nn.functional as F
    cent = [r - r.mean(dim=0, keepdim=True) for r in residuals]   # drop per-FM offset
    sims, n = [], len(cent)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(F.cosine_similarity(cent[i], cent[j], dim=-1).mean()))
    return float(sum(sims) / len(sims)) if sims else float("nan")


def _make_head(d_in, n_cls, hidden, device, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, n_cls)).to(device)


def _fit_head(net, X, y, tr, steps, lr, device, bs=4096):
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        si = tr[torch.randint(len(tr), (min(bs, len(tr)),), generator=g)]
        F.cross_entropy(net(X[si].to(device)), y[si].to(device)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    return net


def _head_acc(net, X, y, te, device, bs=16384):
    import torch
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(te), bs):
            si = te[i:i + bs]
            correct += int((net(X[si].to(device)).argmax(-1).cpu() == y[si]).sum())
    net.train()
    return correct / len(te)


# ======================================================================
# Main experiment
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=86400, memory=32768)
def confabulation_test(
    # --- DGP / model / FM: identical to rhm_latent_loop's m4 regime ---
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # --- wake training (verbatim latent_loop recipe) ---
    conditions: str = "ntp_aux_cl,ntp_aux",
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    lam_aux: float = 1.0, lam_local: float = 1.0, ug_hidden: int = 64,
    pool_size: int = 200000, data_seed: int = 7,
    # --- report battery ---
    n_report_sequences: int = 12000, report_seed: int = 999,
    fresh_fm_steps: int = 3000,
    # Instrument-FM capacity sweep (d_head values, n_head fixed at fwd_n_head so the
    # residual stays a COMPUTATIONAL gap rather than a head-count mismatch -- see
    # RESIDUAL_RANK, where a 1H-FM vs 4H-model mismatch dominated the residual).
    # inst_caps[0] is the default and is where the FM-independent targets are run.
    inst_caps_str: str = "16:1.0,4:0.25,64:2.0,128:4.0", ens_n: int = 3,
    diag_seqs: int = 2000,          # subsample for ens_cos / eta2 (distributional)
    impl_k: int = 8, world_level: int = 3,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    # observer ladder: "n_layer:n_embd" points, last one matches M
    observer_caps: str = "1:64,2:128,4:192,8:256",
    obs_steps: int = 3000, obs_lr: float = 3e-4, obs_bs: int = 64,
    # Test 3
    steer_n_pc: int = 32, steer_eps: float = 1.0, steer_target_kl: float = 0.01,
    eval_interval: int = 1000, seed: int = 42, tag: str = "", resume: bool = True,
    smoke: bool = False,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT, Block
    from rhm.rhm_data import generate_rules_distinct
    from a2a_forward.forward_model import TransformerForwardModel

    if smoke:
        n_steps, n_report_sequences, fresh_fm_steps = 300, 400, 200
        head_steps, obs_steps, pool_size = 200, 200, 20000
        observer_caps, eval_interval = "1:64,2:128", 100
        inst_caps_str, ens_n, steer_n_pc = "16:1.0,4:0.25", 2, 8
        diag_seqs = 100

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    cib = int(predict_from.replace("post_block", ""))
    cond_list = [c.strip() for c in conditions.split(",")]
    caps = [tuple(int(z) for z in c.split(":")) for c in observer_caps.split(",")]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    report_block = f"post_block{n_layer - 1}"
    inst_caps = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in inst_caps_str.split(",")]   # (d_head, mlp_mult) pairs
    default_ci = 0                      # FM-independent targets run at inst_caps[0]
    # Parameter accounting: the ratio that matters for saturation is the FM against the
    # BLOCKS IT PREDICTS, not against the whole model (a2a capacity scaling: the FM
    # saturates once it reaches roughly the parameter count of the predicted layers).
    n_pred_blocks = int(predict_to.replace("post_block", "")) - cib
    n_main_params = sum(p.numel() for p in GPT(v, T, n_layer, n_head, n_embd).parameters())
    n_pred_params = n_pred_blocks * (4 * n_embd * n_embd + 2 * n_embd * 4 * n_embd)
    ckpt_dir = f"{DATA_DIR}/rhm_confabulation/{key}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = lambda c: (f"{ckpt_dir}/{c.replace('@', '_')}"
                           f"_s{n_steps}_seed{seed}.pt")

    print(f"{'='*74}\nCONFABULATION TEST  {key}  {n_layer}L/{n_head}H/{n_embd}D"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  FM {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D  {predict_from}->{predict_to}"
          f"  inject@{inject_after_block}   report from {report_block}")
    print(f"  conditions={cond_list}  K={impl_k}  world_level=d{L-world_level}"
          f"  observers={caps}")
    print(f"  instrument-FM d_head sweep={inst_caps} (ens_n={ens_n})   "
          f"predicted blocks={n_pred_blocks} ({n_pred_params/1e6:.2f}M params)\n{'='*74}")

    # ---------------- training pool (verbatim latent_loop) ----------------
    pool_seqs, pool_lf, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))
    pool_anc = [torch.from_numpy(pool_lf[e].astype(np.int64)) for e in range(L)]
    anc_idx = [torch.arange(T) // (s ** (L - e)) for e in range(L)]
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - e)) == 0])
               for e in range(L)]
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]
    corpus = pool_x.reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    def get_aux_batch(gen):
        idx = torch.randint(0, pool_size, (batch_size,), generator=gen)
        x = pool_x[idx].to(device)
        labels = [pool_anc[e][idx][:, anc_idx[e]].to(device) for e in range(L)]
        return x, labels

    def aux_loss(inter, labels, aux_heads):
        head_out = {b: aux_heads[b](inter[b]).view(batch_size, T, L, v) for b in sup_blocks}
        per_level = []
        for e in range(L):
            se = spanend[e]
            ces = [F.cross_entropy(head_out[b][:, se, e, :].reshape(-1, v),
                                   labels[e][:, se].reshape(-1)) for b in sup_blocks]
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

    def make_fm(d_head=None, mlp_mult=None):
        return TransformerForwardModel(
            d_model=n_embd, d_head=(fwd_d_head if d_head is None else d_head),
            n_head=fwd_n_head, n_layer=fwd_n_layer,
            mlp_mult=(fwd_mlp_mult if mlp_mult is None else mlp_mult),
            block_size=T).to(device)

    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    def eval_ntp(model):
        model.eval()
        gen = torch.Generator().manual_seed(report_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(10):
                x, y = get_ntp_batch(gen)
                tot += model(x, y)[1].item()
        return tot / 10

    # ---------------- report/eval set (aligned -> WORLD labels defined) ----------------
    rep_seqs, rep_lf, _ = _generate_with_traces(rules, n_report_sequences, report_seed)
    rep_x = torch.from_numpy(rep_seqs.astype(np.int64)).to(device)
    # ancestor label of every position at `world_level`
    world_np = rep_lf[world_level][:, (np.arange(T) // (s ** (L - world_level)))]
    world_all = torch.from_numpy(world_np.astype(np.int64))                  # (N,T) cpu

    # ==================================================================
    # Observer: causal stack over tokens, optionally + a continuous stream
    # ==================================================================
    class Observer(nn.Module):
        """Third-party predictor of a report target. `extra_dim>0` adds a continuous
        per-position stream (M's logits for O_io, M's post_block0 for O_act) through a
        linear embedding, so the observer sees exactly what that third party sees."""

        def __init__(self, n_cls, o_layer, o_embd, extra_dim=0, tokens=True):
            super().__init__()
            self.tokens = tokens
            if tokens:
                self.wte = nn.Embedding(v, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            self.proj = nn.Linear(extra_dim, o_embd) if extra_dim else None
            self.h = nn.ModuleList([Block(o_embd, min(o_layer * 2, 8), T, causal=True)
                                    for _ in range(o_layer)])
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_cls)

        def forward(self, tok, extra=None):
            B, t = (tok.shape if self.tokens else extra.shape[:2])
            x = self.wpe(torch.arange(t, device=self.wpe.weight.device))[None].expand(B, t, -1)
            if self.tokens:
                x = x + self.wte(tok)
            if self.proj is not None:
                x = x + self.proj(extra)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    results = {}
    for cond in cond_list:
        use_aux, use_loop = "aux" in cond, "cl" in cond
        print(f"\n{'='*66}\n  CONDITION: {cond}  (aux={use_aux}, loop={use_loop})\n{'='*66}")

        # ============ Phase 1: wake training (verbatim latent_loop) ============
        # Resume: the wake run is the expensive half and is fully determined by
        # (cond, n_steps, seed), so a saved checkpoint is reused verbatim. This exists
        # because a crash in the measurement phase once cost a full 20K-step run.
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_state)
        resumed = resume and os.path.exists(ckpt_path(cond))
        if resumed:
            model.load_state_dict(torch.load(ckpt_path(cond), map_location=device))
            print(f"  RESUMED wake model from {ckpt_path(cond)} -- skipping training")
        main_params = list(model.parameters())
        aux_heads = None
        if use_aux:
            aux_heads = nn.ModuleDict(
                {b: nn.Linear(n_embd, L * v) for b in sup_blocks}).to(device)
            main_params += list(aux_heads.parameters())
        ugate = fm_ct = opt_fwd = None
        if use_loop:
            ugate = UnifiedGate(n_embd, ug_hidden).to(device)
            main_params += list(ugate.parameters())
            fm_ct = make_fm()
            opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=fwd_lr, weight_decay=0.01)
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)

        train_gen = torch.Generator().manual_seed(seed + 1)
        aux_gen = torch.Generator().manual_seed(seed + 2)
        for step in range(0 if resumed else n_steps):
            model.train()
            if use_loop:
                fm_ct.train(); ugate.train()
            x, y = get_ntp_batch(train_gen)
            cache = {}
            if use_loop:
                def ug_cb(act, _c=cache):
                    fp = fm_ct(act.detach())
                    _c["pred"] = fp
                    inj, gw = ugate(act.detach(), fp.detach())
                    _c["gw"] = gw
                    return inj
                logits, _, inter = model(
                    x, return_intermediates=True, cerebellar_fn=ug_cb,
                    cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block)
            else:
                logits, _, inter = model(x, return_intermediates=True)
            loss = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
            ntp = loss
            if use_aux:
                xa, labels = get_aux_batch(aux_gen)
                _, _, inter_a = model(xa, return_intermediates=True)
                loss = loss + lam_aux * aux_loss(inter_a, labels, aux_heads)
            if use_loop:
                fwd_pred, tgt_acts = cache["pred"], inter[predict_to]
                r = tgt_acts - fwd_pred.detach()
                loss = loss + lam_local * (cache["gw"].detach().mean() * r ** 2).mean()
            opt_main.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()
            if use_loop:
                opt_fwd.zero_grad()
                F.mse_loss(fwd_pred, tgt_acts.detach()).backward()
                torch.nn.utils.clip_grad_norm_(fm_ct.parameters(), 1.0)
                opt_fwd.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                print(f"    step {step:6d}: ntp={ntp.item():.4f} val={eval_ntp(model):.4f}")

        if not resumed:
            torch.save(model.state_dict(), ckpt_path(cond))
            volume.commit()
            print(f"  wake done, checkpoint saved -> {ckpt_path(cond)}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # ============ Phase 2: cache the FM-INDEPENDENT state once ============
        c_a0, c_a6, c_logit, c_tok = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(rep_x), 128):
                xb = rep_x[i:i + 128]
                lg, _, vi = model(xb, return_intermediates=True)
                c_a0.append(vi[predict_from].cpu()); c_a6.append(vi[predict_to].cpu())
                c_logit.append(lg.cpu()); c_tok.append(xb.cpu())
        a0 = torch.cat(c_a0); a6 = torch.cat(c_a6)
        logit = torch.cat(c_logit); tok = torch.cat(c_tok)
        del c_a0, c_a6, c_logit, c_tok

        # ONE sequence-level split, shared by heads / observers / steering, so a head
        # never trains on positions from a sequence it is tested on.
        N = tok.shape[0]
        n_str = int(0.8 * N)
        s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
        pos_of = lambda si: (si[:, None] * (T - 1)
                             + torch.arange(T - 1)[None, :]).reshape(-1)
        tr, te = pos_of(s_tr), pos_of(s_te)
        flat = lambda z: z[:, :T - 1].reshape(N * (T - 1), -1).contiguous()
        LG = flat(logit)

        # ---- FM-independent report targets (these never move with FM capacity) ----
        y_behav = (logit[:, :T - 1].argmax(-1).reshape(-1)
                   == tok[:, 1:].reshape(-1)).long()
        lp = F.log_softmax(LG, -1)
        ent = -(lp.exp() * lp).sum(-1)
        y_ent = torch.bucketize(ent, torch.quantile(ent[tr], torch.tensor([.25, .5, .75])))
        y_world = world_all[:, :T - 1].reshape(-1)
        fixed_targets = {"BEHAV": (y_behav, 2), "ENT": (y_ent, 4), "WORLD": (y_world, v)}

        last_block, ln_f = model.transformer.h[n_layer - 1], model.transformer.ln_f

        def make_report_input(a6_sub_seq):
            out = []
            with torch.no_grad():
                for i in range(0, a6_sub_seq.shape[0], 128):
                    out.append(last_block(a6_sub_seq[i:i + 128].to(device)).cpu())
            return flat(torch.cat(out))

        X_full = make_report_input(a6)                       # FM-independent
        sh = torch.randperm(N * T, generator=torch.Generator().manual_seed(seed + 13))
        rs = lambda z: z.reshape(N * T, n_embd)[sh].reshape(N, T, n_embd)

        # ---- observer machinery ----
        def train_observer(yt, ncls, o_layer, o_embd, extra_src, use_tokens,
                           data_frac=1.0):
            torch.manual_seed(seed + 31)
            o = Observer(ncls, o_layer, o_embd,
                         extra_dim=(0 if extra_src is None else extra_src.shape[-1]),
                         tokens=use_tokens).to(device)
            opt = torch.optim.AdamW(o.parameters(), lr=obs_lr, weight_decay=0.01)
            g = torch.Generator().manual_seed(seed + 32)
            pool = s_tr[:max(1, int(data_frac * len(s_tr)))]
            for _ in range(obs_steps):
                si = pool[torch.randint(len(pool), (obs_bs,), generator=g)]
                xb = tok[si, :T - 1].to(device)
                eb = None if extra_src is None else extra_src[si, :T - 1].to(device)
                F.cross_entropy(o(xb, eb).reshape(-1, ncls),
                                yt[si].reshape(-1).to(device)).backward()
                torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            o.eval()
            corr = tot_ = 0
            with torch.no_grad():
                for i in range(0, len(s_te), 64):
                    si = s_te[i:i + 64]
                    xb = tok[si, :T - 1].to(device)
                    eb = None if extra_src is None else extra_src[si, :T - 1].to(device)
                    p_ = o(xb, eb).argmax(-1).cpu()
                    corr += int((p_ == yt[si]).sum()); tot_ += p_.numel()
            del o
            torch.cuda.empty_cache()
            return corr / tot_

        def run_ladder(yv, ncls, self_acc, label):
            """Test 1: capacity-swept third-person observers + the data-budget control."""
            yt = yv.view(N, T - 1)
            d = {}
            for (o_layer, o_embd) in caps:
                cap = f"{o_layer}L{o_embd}D"
                d[f"O_input@{cap}"] = train_observer(yt, ncls, o_layer, o_embd, None, True)
                d[f"O_io@{cap}"] = train_observer(yt, ncls, o_layer, o_embd, logit, True)
            ol, oe = caps[-1]
            # ceiling: sees M's own post_block0, i.e. this observer IS FM + a probe
            d[f"O_act@{ol}L{oe}D"] = train_observer(yt, ncls, ol, oe, a0, False)
            # data-budget control: capacity-limited or just starved? If halving the data
            # barely moves it, the shortfall is about access, not about our budget.
            d[f"O_io@{ol}L{oe}D_half_data"] = train_observer(
                yt, ncls, ol, oe, logit, True, data_frac=0.5)
            bio = max(v_ for k_, v_ in d.items()
                      if k_.startswith("O_io") and "half" not in k_)
            print(f"  [observ/{label:12s}] self={self_acc:.3f}  best_O_io={bio:.3f}  "
                  f"advantage={self_acc - bio:+.3f}  |  "
                  + " ".join(f"{k_}={v_:.3f}" for k_, v_ in d.items()))
            return d

        def report_row(yv, ncls, Xv, label):
            """Tests 2 + 4: full-access head, its ablation evals, and the confabulator."""
            h = _fit_head(_make_head(n_embd, ncls, head_hidden, device, seed),
                          Xv["full"], yv, tr, head_steps, head_lr, device)
            row = {"full": _head_acc(h, Xv["full"], yv, te, device)}
            for vn in ("shuffle_r", "shuffle_p", "zero_r", "zero_p"):
                row[f"eval_{vn}"] = _head_acc(h, Xv[vn], yv, te, device)
            hc = _fit_head(_make_head(n_embd, ncls, head_hidden, device, seed),
                           Xv["zero_r"], yv, tr, head_steps, head_lr, device)
            row["confab"] = _head_acc(hc, Xv["zero_r"], yv, te, device)
            row["baseline"] = _balance(yv[te], ncls)
            print(f"  [report/{label:12s}] full={row['full']:.3f} "
                  f"confab={row['confab']:.3f} | eval: shuf_r={row['eval_shuffle_r']:.3f} "
                  f"shuf_p={row['eval_shuffle_p']:.3f} zero_r={row['eval_zero_r']:.3f} "
                  f"| base={row['baseline']:.3f}")
            return row, h

        def train_fresh_fm(fm_seed, d_head, mlp_mult):
            torch.manual_seed(fm_seed)
            fm = make_fm(d_head, mlp_mult)
            opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(fresh_fm_steps):
                fm.train()
                xb, _ = get_ntp_batch(gen)
                with torch.no_grad():
                    _, _, vi = model(xb, return_intermediates=True)
                F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            fm.eval()
            for p in fm.parameters():
                p.requires_grad = False
            return fm

        def fm_predict(fmx, src, n_seq=None):
            lim = len(src) if n_seq is None else min(n_seq, len(src))
            out = []
            with torch.no_grad():
                for i in range(0, lim, 128):
                    out.append(fmx(src[i:min(i + 128, lim)].to(device)).cpu())
            return torch.cat(out)

        # ============ Phase 3: sweep the INSTRUMENT FM capacity ============
        # The residual is only meaningful if the FM is small enough that what it misses is
        # a genuine COMPUTATIONAL gap. On a toy DGP an over-capacity FM saturates (cosine
        # -> 0.999) and `r` degenerates into architectural-mismatch noise -- which would
        # FAKE this experiment's headline: M can report noise (the report head sits
        # downstream of a_j), while no third party can predict noise from tokens+logits.
        # IMPL would then show a large "introspective advantage" built entirely from junk.
        # So every capacity point carries two discriminators, and no IMPL number may be
        # read without them:
        #   ens_cos -- do INDEPENDENT fresh FMs leave the SAME residual? high =>
        #              input-determined (a real gap any FM misses identically);
        #              low => FM-idiosyncratic noise.
        #   eta2    -- is the residual conditioned on the ground-truth DGP hierarchy?
        #              A genuine capacity gap is; noise is not.
        # (rhm_latent_loop Exp 4 established ens_cos; REGIME_TRANSITION is the cautionary
        #  case where cosine 0.994 made the whole measurement untestable.)
        by_cap, fixed_done = {}, False
        for (dh, mm) in inst_caps:
            fms = [train_fresh_fm(seed + 911 + 37 * j, dh, mm)
                   for j in range(max(1, ens_n))]
            fm_params = sum(p.numel() for p in fms[0].parameters())
            pred = fm_predict(fms[0], a0)
            resid = a6 - pred
            fwd_cos = float(F.cosine_similarity(pred, a6, dim=-1).mean())
            res_norm = float(resid.norm(dim=-1).mean())

            # diagnostics on a subsample (cheap; these are distributional quantities)
            nd = min(diag_seqs, N)
            ens = (_ensemble_cos([(a6[:nd] - fm_predict(f_, a0, nd)).reshape(-1, n_embd)
                                  for f_ in fms]) if len(fms) > 1 else float("nan"))
            eta2 = _eta2_by_level(resid[:nd], [lf[:nd] for lf in rep_lf], s, L, T)
            cap_tag = f"h{dh}m{mm:g}"
            print(f"\n  --- instrument FM {cap_tag}: {fm_params/1e3:.0f}K params "
                  f"({100*fm_params/n_main_params:.1f}% of M, "
                  f"{100*fm_params/n_pred_params:.1f}% of the {n_pred_blocks} predicted "
                  f"blocks) ---")
            print(f"      cosine={fwd_cos:.4f}  |r|={res_norm:.3f}  ens_cos={ens:.3f}  "
                  + " ".join(f"{k_}η²={v_:.3f}" for k_, v_ in eta2.items()))

            R, P = flat(resid), flat(pred)
            cents = _kmeans_fit(R[tr].to(device), impl_k, seed=seed)   # TRAIN rows only
            y_impl = _kmeans_assign(R.to(device), cents).cpu()
            X_variants = {
                "full":      X_full,
                "shuffle_r": make_report_input(pred + rs(resid)),
                "shuffle_p": make_report_input(rs(pred) + resid),
                "zero_r":    make_report_input(pred),
                "zero_p":    make_report_input(resid),
            }

            rep_impl, h_impl = report_row(y_impl, impl_k, X_variants, f"IMPL/{cap_tag}")
            obs_impl = run_ladder(y_impl, impl_k, rep_impl["full"], f"IMPL/{cap_tag}")

            # The FM-independent targets are run once, at the default capacity, since
            # neither their labels nor X_full depend on the instrument. (Their ablation
            # columns do, so they are reported alongside that capacity.)
            fixed_rep, fixed_obs = {}, {}
            if not fixed_done:          # inst_caps[0] == the default instrument
                for tn, (yv_, nc_) in fixed_targets.items():
                    fixed_rep[tn], h_ = report_row(yv_, nc_, X_variants, tn)
                    fixed_obs[tn] = run_ladder(yv_, nc_, fixed_rep[tn]["full"], tn)
                    if tn == "BEHAV":
                        h_behav = h_
                fixed_done = True

            # ---- Test 3: matched-KL steering, residual span vs self-theory span ----
            # Rescale every direction to the SAME behavioural effect (target KL on the
            # logits). Matched KL means matched behaviour, so a difference in how far the
            # IMPL report moves cannot be a behavioural artifact -- and BEHAV report-flip
            # becomes a built-in control that should come out equal across families.
            # Prediction: residual-span steering moves the IMPL report more than
            # prediction-span steering. A theory-driven report cannot show that asymmetry.
            steer = {"target_kl": steer_target_kl, "families": {}}
            idx = torch.randperm(len(tr),
                                 generator=torch.Generator().manual_seed(seed + 17))
            sel_ = tr[idx[:min(50000, len(tr))]]

            def top_pcs(M_):
                sub = M_[sel_].to(device)
                return torch.linalg.svd(sub - sub.mean(0),
                                        full_matrices=False)[2][:steer_n_pc]

            steer_seqs = s_te[:min(512, len(s_te))]
            base_a6 = a6[steer_seqs].to(device)

            def run(a6_batch):
                with torch.no_grad():
                    z = last_block(a6_batch)
                    return (F.log_softmax(model.lm_head(ln_f(z))[:, :T - 1], -1),
                            h_impl(z[:, :T - 1]).argmax(-1),
                            h_behav(z[:, :T - 1]).argmax(-1))

            base_lp, base_impl, base_behav = run(base_a6)

            def probe_dir(u, eps):
                lp_, ri_, rb_ = run(base_a6 + eps * u.view(1, 1, -1))
                return (float(F.kl_div(lp_.reshape(-1, v), base_lp.reshape(-1, v),
                                       log_target=True, reduction="batchmean")),
                        float((ri_ != base_impl).float().mean()),
                        float((rb_ != base_behav).float().mean()))

            for fam, pcs in (("residual", top_pcs(R)), ("prediction", top_pcs(P))):
                per = []
                for k in range(steer_n_pc):
                    u = pcs[k]
                    kl0, _, _ = probe_dir(u, steer_eps)
                    if kl0 <= 1e-9:
                        continue
                    # KL is locally quadratic in eps, so this lands close in one shot;
                    # one refinement tightens it. Cap eps to stay near-local.
                    e_k = min(steer_eps * (steer_target_kl / kl0) ** 0.5,
                              20 * steer_eps)
                    kl1, _, _ = probe_dir(u, e_k)
                    if kl1 > 1e-9:
                        e_k = min(e_k * (steer_target_kl / kl1) ** 0.5,
                                  20 * steer_eps)
                    kl_f, fi, fb = probe_dir(u, e_k)
                    per.append({"pc": k, "eps": e_k, "kl": kl_f,
                                "impl_flip": fi, "behav_flip": fb})
                ag = (lambda kk: float(np.mean([p_[kk] for p_ in per]))
                      if per else float("nan"))
                steer["families"][fam] = {
                    "per_pc": per, "kl": ag("kl"),
                    "impl_flip": ag("impl_flip"), "behav_flip": ag("behav_flip")}
                print(f"  [steer/{fam:10s}] @matched KL={ag('kl'):.4f}  "
                      f"IMPL-flip={ag('impl_flip'):.3f}  "
                      f"BEHAV-flip={ag('behav_flip'):.3f}  (eps={ag('eps'):.3f})")
            rs_ = steer["families"].get("residual", {})
            ps_ = steer["families"].get("prediction", {})
            steer["impl_flip_ratio"] = (
                rs_.get("impl_flip", 0) / ps_["impl_flip"]
                if ps_.get("impl_flip", 0) > 0 else float("inf"))
            print(f"  [steer] residual/prediction IMPL-flip ratio at matched "
                  f"behaviour = {steer['impl_flip_ratio']:.2f}x")

            by_cap[cap_tag] = {
                "fm_params": fm_params,
                "pct_of_model": 100 * fm_params / n_main_params,
                "pct_of_predicted_blocks": 100 * fm_params / n_pred_params,
                "fwd_cosine": fwd_cos, "res_norm": res_norm,
                "ens_cos": ens, "hierarchy_eta2": eta2,
                "impl_report": rep_impl, "impl_observers": obs_impl,
                "steering": steer,
                "fixed_report": fixed_rep, "fixed_observers": fixed_obs,
            }
            del R, P, resid, pred, X_variants, fms
            torch.cuda.empty_cache()

        results[cond] = {"val": eval_ntp(model), "by_capacity": by_cap}
        volume.commit()
        del model, a0, a6, logit, X_full
        torch.cuda.empty_cache()

    out_dir = f"{DATA_DIR}/rhm_confabulation/{key}"
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{out_dir}/{(tag + '_') if tag else ''}results.json"
    with open(fn, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---------------- summary ----------------
    bio_of = lambda d: max(v_ for k_, v_ in d.items()
                           if k_.startswith("O_io") and "half" not in k_)
    print(f"\n{'='*74}\nSUMMARY")
    print("  introspective advantage = self-report - best capacity-matched O_io")
    print("  margin = self-report - confabulator (the honest quantity: a chunk of the")
    print("           residual cluster is guessable from FM(a_i) alone)")
    print("  READ IMPL ONLY AGAINST ens_cos / eta2. A saturated FM (cosine -> 1) leaves")
    print("  a junk residual that M can report and no observer can predict -- which")
    print("  fakes exactly the signature this experiment is looking for.")
    for cond, rr in results.items():
        print(f"\n  {cond}   val={rr['val']:.4f}")
        print(f"    {'instrument':12s} {'%pred':>6s} {'cos':>6s} {'ens':>6s} {'d6η²':>6s} "
              f"{'self':>6s} {'confab':>7s} {'margin':>7s} {'bestOio':>8s} {'advant':>7s} "
              f"{'shufR':>7s} {'base':>6s} {'steer':>6s}")
        for cap_tag, c in rr["by_capacity"].items():
            row, bio = c["impl_report"], bio_of(c["impl_observers"])
            sr = c["steering"].get("impl_flip_ratio", float("nan"))
            print(f"    IMPL {cap_tag:7s} {c['pct_of_predicted_blocks']:6.1f} "
                  f"{c['fwd_cosine']:6.3f} {c['ens_cos']:6.3f} "
                  f"{c['hierarchy_eta2'].get('d6', float('nan')):6.3f} "
                  f"{row['full']:6.3f} {row['confab']:7.3f} "
                  f"{row['full'] - row['confab']:+7.3f} {bio:8.3f} "
                  f"{row['full'] - bio:+7.3f} {row['eval_shuffle_r']:7.3f} "
                  f"{row['baseline']:6.3f} {sr:6.2f}")
        for cap_tag, c in rr["by_capacity"].items():
            for tname, row in c.get("fixed_report", {}).items():
                bio = bio_of(c["fixed_observers"][tname])
                print(f"    {tname:12s} {'':6s} {'':6s} {'':6s} {'':6s} "
                      f"{row['full']:6.3f} {row['confab']:7.3f} "
                      f"{row['full'] - row['confab']:+7.3f} {bio:8.3f} "
                      f"{row['full'] - bio:+7.3f} {row['eval_shuffle_r']:7.3f} "
                      f"{row['baseline']:6.3f}")
    print(f"\n  saved -> {fn}\n{'='*74}")
    return results


@app.local_entrypoint()
def main(conditions: str = "ntp_aux_cl,ntp_aux", n_steps: int = 20000,
         tag: str = "", smoke: bool = False):
    confabulation_test.remote(conditions=conditions, n_steps=n_steps,
                              tag=tag, smoke=smoke)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def eta2_recheck(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    fwd_n_layer: int = 1, fwd_n_head: int = 8,
    inst_caps_str: str = "16:1.0,4:0.25,64:2.0,128:4.0",
    conditions: str = "ntp_aux_cl,ntp_aux",
    n_steps: int = 20000, seed: int = 42,
    fresh_fm_steps: int = 3000, fwd_lr: float = 1e-3,
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    pool_size: int = 200000, data_seed: int = 7, batch_size: int = 64,
):
    """Recompute residual hierarchy eta2 with the REFERENCE estimator.

    The main run reports eta2 pooled over all T positions; rhm_latent_loop's
    `_compute_hierarchy_eta2` uses the LAST TOKEN ONLY (`residuals_np[:, -1, :]`, line
    186) -- the root-completing position, where deep-level structure is maximal. Pooling
    over all 64 positions dilutes it heavily, so the two numbers are not comparable and
    the main run's 0.024-vs-reference-0.322 gap may be entirely an estimator artifact.

    This loads the saved wake checkpoints (no retraining) and reports BOTH estimators
    side by side, importing the reference function itself so the comparison is exact.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _compute_hierarchy_eta2      # the reference estimator
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    ckpt_dir = f"{DATA_DIR}/rhm_confabulation/{key}/ckpt"
    caps = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in inst_caps_str.split(",")]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # NTP batches must match the main run's training distribution (flat corpus windows)
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        return corpus[ix[:, None] + arangeT[None, :]].to(device)

    eval_seqs, eval_lf, eval_lr = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    out = {}
    for cond in [c.strip() for c in conditions.split(",")]:
        path = f"{ckpt_dir}/{cond}_s{n_steps}_seed{seed}.pt"
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"\n{'='*72}\n  {cond}   (loaded {path})\n{'='*72}")

        out[cond] = {}
        for (dh, mm) in caps:
            torch.manual_seed(seed + 911)
            fm = TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                         n_layer=fwd_n_layer, mlp_mult=mm,
                                         block_size=T).to(device)
            opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(seed + 912)
            for _ in range(fresh_fm_steps):
                fm.train()
                with torch.no_grad():
                    _, _, vi = model(get_ntp_batch(gen), return_intermediates=True)
                F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            fm.eval()

            res, cos = [], []
            with torch.no_grad():
                for i in range(0, len(eval_x), 128):
                    _, _, vi = model(eval_x[i:i + 128], return_intermediates=True)
                    pr, tg = fm(vi[predict_from]), vi[predict_to]
                    res.append((tg - pr).cpu().numpy())
                    cos.append(float(F.cosine_similarity(pr, tg, dim=-1).mean()))
            res_np = np.concatenate(res, 0)

            ref = _compute_hierarchy_eta2(res_np, eval_lf, eval_lr, s=s, L=L)   # last token
            pooled = _eta2_by_level(torch.from_numpy(res_np), eval_lf, s, L, T)  # all pos
            tag = f"h{dh}m{mm:g}"
            out[cond][tag] = {"fwd_cosine": float(np.mean(cos)),
                              "reference_last_token": ref, "pooled_all_positions": pooled}
            print(f"  {tag:10s} cos={np.mean(cos):.4f}")
            print("     reference (last token) feature η²: " +
                  " ".join(f"d{L-e}={ref[f'level_{e}']['feature_eta2']:.3f}"
                           for e in range(L)))
            print("     reference (last token) rule    η²: " +
                  " ".join(f"d{L-e}={ref[f'level_{e}']['rule_eta2']:.3f}"
                           for e in range(L)))
            print("     pooled    (all positions)      η²: " +
                  " ".join(f"{k_}={v_:.3f}" for k_, v_ in pooled.items()))
            del fm
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    fn = f"{DATA_DIR}/rhm_confabulation/{key}/eta2_recheck.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n  saved -> {fn}")
    print("  Reference line (RHM_LATENT_LOOP Exp 1, fresh FM, d_head16 matched-head):")
    print("    ntp_aux    d6 η²=0.295  d5 η²=0.381")
    print("    ntp_aux_cl d6 η²=0.322  d5 η²=0.421")
    return out
