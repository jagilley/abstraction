"""Authorship: can a transformer tell which parts of its context it wrote itself?

Parent:   ../../confabulation/confabulation.py  (the battery this forks; read its module docstring first)
Paper:    papers/forward_self_models_paper2.md  (S2 criterion 2 is the discriminator)
Design:   DESIGN.md  (every deviation from the parent is argued there)

THE QUESTION
  M generates a continuation at temperature 1, the continuation is spliced back into a
  context, and M re-reads the whole thing in an ordinary teacher-forced forward pass.
  Per position: did M write this token?

THE FRAMEWORK'S PREDICTION, AND WHY IT IS SHARP
  M re-reading its own output runs *exactly* the same computation as reading anyone
  else's: the re-read state is a deterministic function of the token sequence and
  nothing else. There is no implementation trace of having written it. So "mine" is,
  formally, an INPUT-DETERMINED fact -- the same category as the paper's WORLD control,
  not the IMPL category. Criterion 1 (the report is a function of M's state) holds
  vacuously; criterion 2 (not cheaply recoverable from M's I/O map) is the whole
  question, and the construction says it should FAIL: authorship should reduce to "is
  this token likely under me", which an observer holding M's own output distribution
  can compute directly.

  If that holds, the self-recognition reported for LLMs (Panickssery et al. 2024) is a
  public fact -- self-INFERENCE, not introspection. If instead the self-report beats the
  I/O observer, the O_act ceiling and the capacity ladder say whether that is access or
  capacity. Either outcome is the result.

WHAT IS MEASURED (all on the same held-out sequences, same splits, same positions)
  target MINE  -- per position, 1 if M sampled this token, 0 if it came from the corpus
  target IMPL / IMPL_COS -- the parent's implementation target, run on these very same
                 sequences and the very same scored positions, as the positive control
                 that the battery is alive on this model and this data

  The contestants (advantage = best self - best I/O observer):
    SELF_mlp   per-position MLP on M's post_block3 stream  [the parent's report head]
    SELF_seq   same input, but a sequence model at observer capacity -- MINE is a
               span-level fact and the observers are sequence models, so the parent's
               per-position head alone would be an unfair self-report here
    O_input    tokens only
    O_lik      M's per-token likelihood features ONLY (no tokens): the explicit
               "is this likely under me" baseline the framework says everything
               reduces to
    O_io       tokens + top-k output summary + 4 full-distribution scalars + the
               likelihood features
    O_io_nolik the parent's O_io exactly (no likelihood features), to show how much of
               the observer's score the likelihood channel carries
    O_act      tokens + M's own post_block0 (the activation-access ceiling)
    LIK_pos    an MLP on the 6 likelihood features at that position alone -- the
               no-aggregation floor of the likelihood route

DELIBERATE DEVIATIONS FROM THE PARENT (argued in DESIGN.md)
  - O_io is handed the per-token likelihood features. The parent's O_io gets M's output
    distribution at position t but would have to JOIN it across positions with the
    identity of token t+1 to compute log p(x_t) -- an attention lookup it was never
    trained for. log p(x_t | x_<t) is a function of M's I/O map and the input, i.e.
    public by construction, so withholding it would manufacture a self-advantage out of
    an input-format artifact. Handing it over is the parent's own principle (give the
    third party every advantage we can afford). O_io_nolik keeps the parent's version
    alongside so the difference is visible.
  - The self-report gets a sequence-model variant (SELF_seq), for the mirror-image
    reason: the parent's report target is per-position and this one is not.
  - One arm (ol) and one instrument capacity. The arm contrast and the instrument sweep
    are the parent's questions, not this one's; the instrument here exists only to carry
    the IMPL positive control, and it carries its ens_cos / residual-structure guards.

Run (the `chromatic` workspace holds the FineWeb-Edu shards at /data/tokens):

  modal run a2a_forward/fsm_part3/authorship/authorship.py::authorship_test --smoke
  modal run --detach a2a_forward/fsm_part3/authorship/authorship.py::authorship_test \
      --tag main
"""

import json
import os

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


# ======================================================================
# Report-set construction
# ======================================================================

def _build_plans(N, T, prefix_len, span_lens, mode, rng):
    """Which positions M writes.

    mode='mix'   -- after the corpus prefix, spans alternate class, with the starting
                    class and every span length drawn per sequence from a SHARED
                    distribution, so neither absolute position, nor span length, nor a
                    span boundary predicts the class. ~5-6 spans per sequence at T=128.
    mode='whole' -- one boundary: the whole post-prefix region is M's, or the whole of it
                    is the document's true continuation, 50/50 across sequences. No
                    splice anywhere, which is the control for the mix set's one
                    unavoidable asymmetry (a corpus span that follows an M span does not
                    continue it).
    """
    import numpy as np
    plan = np.zeros((N, T), dtype=bool)
    if mode == "whole":
        plan[0::2, prefix_len:] = True
        return plan
    for n in range(N):
        t, cls = prefix_len, int(rng.integers(0, 2))
        while t < T:
            L = int(rng.choice(span_lens))
            if cls:
                plan[n, t:min(t + L, T)] = True
            t += L
            cls = 1 - cls
    return plan


def _sample_into(model, base_tok, plan, prefix_len, temperature, device, seed, bs=256):
    """Teacher-force the corpus document, but at every position flagged in `plan`
    replace the token with one M samples from its own distribution given everything
    already in the sequence (corpus and self-written alike).

    This is the honest version of the question at temperature 1: the token M writes is
    drawn from M's own next-token distribution, so it is not identifiable by being
    'more likely than the corpus' in any trivial per-token sense -- only to the extent
    that M's distribution differs from the text distribution.
    """
    import torch
    N, T = base_tok.shape
    out = base_tok.clone()
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for i in range(0, N, bs):
            x = out[i:i + bs].to(device)
            pl = torch.from_numpy(plan[i:i + bs]).to(device)
            for t in range(prefix_len, T):
                if not bool(pl[:, t].any()):
                    continue
                logits, _ = model(x[:, :t])
                probs = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
                draw = torch.multinomial(probs, 1, generator=g).squeeze(1)
                x[:, t] = torch.where(pl[:, t], draw, x[:, t])
            out[i:i + bs] = x.cpu()
    return out


def _span_offset(plan):
    """Position's index within its own (class-homogeneous) run. 0 at every class change.
    The x-axis of the accuracy-vs-evidence curve: how much of the span the contestant has
    already seen."""
    import numpy as np
    N, T = plan.shape
    off = np.zeros((N, T), dtype=np.int64)
    run = np.zeros(N, dtype=np.int64)
    for t in range(1, T):
        run = np.where(plan[:, t] == plan[:, t - 1], run + 1, 0)
        off[:, t] = run
    return off


def _auc(scores, labels):
    """Rank AUC with tie handling. scores/labels: 1-D numpy."""
    import numpy as np
    from scipy.stats import rankdata
    pos = labels > 0.5
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


# ======================================================================
# Main experiment
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=16384)
def authorship_test(
    # --- data / model: the parent's a2a_loop_train defaults ---
    n_tokens: int = 10_000_000, block_size: int = 128,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block2",
    report_block: str = "", inject_after_block: int = 1,
    fwd_n_layer: int = 2, fwd_d_head: int = 64, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
    cond: str = "ol",
    n_steps: int = 10_000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    # --- report sets ---
    n_report_sequences: int = 3000, report_seed: int = 999,
    prefix_len: int = 24, span_lens_str: str = "8,12,16,24,32",
    temperature: float = 1.0, temperature_lo: float = 0.7,
    gen_bs: int = 256,
    # --- battery ---
    inst_cap_str: str = "16:0.5", ens_n: int = 3, fresh_fm_steps: int = 3000,
    impl_k: int = 8,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    observer_caps: str = "1:64,2:128,4:256",
    obs_steps: int = 2000, obs_lr: float = 3e-4, obs_bs: int = 32,
    obs_topk: int = 64, observer_causal: bool = False,
    skip_impl: bool = False,
    eval_interval: int = 500, seed: int = 42, tag: str = "",
    refresh_wake: bool = False, smoke: bool = False,
):
    import glob
    import resource
    import time
    import numpy as np
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from a2a_forward.model import GPT, Block
    from a2a_forward.forward_model import TransformerForwardModel
    from a2a_forward.confabulation.confabulation import (
        _kmeans_fit, _kmeans_assign, _cluster_quality, _residual_structure,
        _ensemble_cos, _make_head, _fit_head, _head_score, _balance,
        _syntactic_labels)

    t_start = time.time()
    if smoke:
        n_steps, n_report_sequences, fresh_fm_steps = 300, 300, 150
        head_steps, obs_steps, n_tokens = 150, 60, 2_000_000
        observer_caps, eval_interval, ens_n = "1:64,2:128", 100, 2
        gen_bs = 150

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = block_size
    cib = int(predict_from.replace("post_block", "").replace("post_embed", "-1"))
    j_idx = int(predict_to.replace("post_block", ""))
    report_block = report_block or f"post_block{n_layer - 1}"
    r_idx = int(report_block.replace("post_block", ""))
    caps = [tuple(int(z) for z in c.split(":")) for c in observer_caps.split(",")]
    top_cap = caps[-1]
    inst_dh, inst_mm = int(inst_cap_str.split(":")[0]), float(inst_cap_str.split(":")[1])
    span_lens = [int(s) for s in span_lens_str.split(",")]
    mid_blocks = list(range(j_idx + 1, r_idx + 1))
    n_pred_blocks = j_idx - cib
    n_pred_params = n_pred_blocks * 12 * n_embd * n_embd
    N = n_report_sequences

    # ---------------- data (verbatim from the parent) ----------------
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]
    all_tokens, total = [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "shard_*.npy"))):
        toks = np.load(path)
        all_tokens.append(toks)
        total += len(toks)
        if total >= n_tokens:
            break
    data = torch.from_numpy(np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    enc = tiktoken.get_encoding("gpt2")

    print(f"{'='*78}\nAUTHORSHIP  ({cond})  {n_layer}L/{n_head}H/{n_embd}D  "
          f"P={n_tokens:,}  T={T}{'  [SMOKE]' if smoke else ''}")
    print(f"  report from {report_block}  prefix={prefix_len}  spans={span_lens}  "
          f"N={N}  temps={temperature}/{temperature_lo}")
    print(f"  observers={caps}  {'non-causal' if not observer_causal else 'causal'}  "
          f"instrument={inst_cap_str} ens_n={ens_n}\n{'='*78}")

    def get_batch(split_data, gen):
        ix = torch.randint(len(split_data) - T - 1, (batch_size,), generator=gen)
        x = torch.stack([split_data[i:i + T] for i in ix])
        y = torch.stack([split_data[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    # ============ Phase 0: the wake model ============
    # Same recipe, same seed, same data order and the same checkpoint key as the parent's
    # `ol` arm, so a checkpoint trained by the parent (or by a sibling running it) is
    # loaded rather than retrained. For `ol` the co-trained FM has no gradient path into
    # M (its inputs are .detach()ed and opt_main holds only M's parameters), so M is a
    # deterministic function of (seed, arch, data, steps) and any `ol` checkpoint at this
    # config holds the identical M.
    torch.manual_seed(seed)
    model = GPT(vocab_size, T, n_layer, n_head, n_embd).to(device)
    fm_ct = TransformerForwardModel(d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                                    n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                                    block_size=T).to(device)
    ck_key = (f"{cond}_L{n_layer}H{n_head}D{n_embd}_P{n_tokens}_T{T}_s{n_steps}"
              f"_{predict_from}to{predict_to}_inj{inject_after_block}"
              f"_fm{fwd_n_layer}x{fwd_d_head}x{fwd_mlp_mult:g}_seed{seed}")
    ck_dir = f"{DATA_DIR}/a2a_forward/confabulation/wake_ckpt"
    ck_path = f"{ck_dir}/{ck_key}.pt"
    try:
        volume.reload()
    except Exception as e:
        print(f"  volume.reload() failed ({e}) -- continuing")
    loaded = False
    if not refresh_wake and os.path.exists(ck_path):
        try:
            ck = torch.load(ck_path, map_location=device, weights_only=True)
            model.load_state_dict(ck["model"]); fm_ct.load_state_dict(ck["fm"])
            loaded = True
            print(f"  loaded wake checkpoint <- {ck_path}")
        except Exception as e:
            print(f"  wake checkpoint unreadable ({e}) -- retraining")
    if not loaded:
        print(f"  no wake checkpoint at {ck_path} -- training {n_steps} steps")
        opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=fwd_lr, weight_decay=0.01)
        train_gen = torch.Generator().manual_seed(seed + 1)
        for step in range(n_steps):
            model.train(); fm_ct.train()
            x, y = get_batch(train_data, train_gen)
            _, lm_loss, inter = model(x, y, return_intermediates=True)
            fwd_loss = F.mse_loss(fm_ct(inter[predict_from].detach()),
                                  inter[predict_to].detach())
            opt_main.zero_grad(); lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_main.step()
            opt_fwd.zero_grad(); fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm_ct.parameters(), 1.0); opt_fwd.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                g = torch.Generator().manual_seed(report_seed + 5)
                with torch.no_grad():
                    vx, vy = get_batch(val_data, g)
                    vl = float(model(vx, vy)[1])
                print(f"    step {step:6d}: lm={float(lm_loss):.4f} val={vl:.4f} "
                      f"fwd_mse={float(fwd_loss):.5f}")
        os.makedirs(ck_dir, exist_ok=True)
        tmp = f"{ck_path}.tmp{os.getpid()}"
        torch.save({"model": model.state_dict(), "fm": fm_ct.state_dict()}, tmp)
        os.replace(tmp, ck_path)                # atomic: a sibling may be writing it too
        volume.commit()
        print(f"  saved wake checkpoint -> {ck_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def eval_val():
        g = torch.Generator().manual_seed(report_seed + 5)
        with torch.no_grad():
            return float(np.mean([float(model(*get_batch(val_data, g))[1])
                                  for _ in range(20)]))
    val_base = eval_val()
    print(f"  val (standalone) = {val_base:.4f}")

    # ============ Phase 1: build the report sets ============
    rng = np.random.default_rng(report_seed)
    starts = rng.integers(0, len(val_data) - T - 1, size=N)
    base_tok = torch.stack([val_data[i:i + T] for i in starts])          # (N,T) cpu

    set_specs = [(f"mix_t{temperature:g}", "mix", temperature, "full"),
                 (f"whole_t{temperature:g}", "whole", temperature, "light"),
                 (f"mix_t{temperature_lo:g}", "mix", temperature_lo, "light")]

    report_sets = {}
    for k_, (sname, smode, stemp, _l) in enumerate(set_specs):
        plan = _build_plans(N, T, prefix_len, span_lens, smode,
                            np.random.default_rng(report_seed + 101 * k_))
        t0 = time.time()
        tok = _sample_into(model, base_tok, plan, prefix_len, stemp, device,
                           seed + 7 + k_, bs=gen_bs)
        print(f"  built '{sname}': mode={smode} temp={stemp}  mine-fraction="
              f"{plan[:, prefix_len:T - 1].mean():.3f}  ({time.time() - t0:.0f}s)")
        report_sets[sname] = {"tok": tok, "plan": torch.from_numpy(plan),
                              "offset": torch.from_numpy(_span_offset(plan))}
    del base_tok

    # ---- one sequence-level split, shared by every contestant ----
    n_str = int(0.8 * N)
    s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
    pos_of = lambda si: (si[:, None] * (T - 1) + torch.arange(T - 1)[None, :]).reshape(-1)
    flat = lambda z: z[:, :T - 1].reshape(N * (T - 1), -1).contiguous()
    # Scored region: everything after the corpus prefix. The prefix is always corpus and
    # always at the front, so scoring it would hand every contestant a free positional
    # cue; it is context, not evidence.
    scored = torch.zeros(N, T, dtype=torch.bool)
    scored[:, prefix_len:] = True
    scored_seq = scored[:, :T - 1]                                   # (N,T-1)
    scored_flat = scored_seq.reshape(-1)
    tr_all, te_all = pos_of(s_tr), pos_of(s_te)
    tr = tr_all[scored_flat[tr_all]]
    te = te_all[scored_flat[te_all]]
    print(f"  scored positions: train={len(tr):,}  test={len(te):,}  "
          f"(of {N * (T - 1):,})")

    # ============ the observer / self-report sequence model ============
    class Observer(nn.Module):
        """One architecture, five input regimes. `mode` selects which channels are on;
        depth, width, attention pattern, optimizer and data budget are held fixed, so
        the ladder measures access rather than architecture.

          input     tokens
          lik       M's per-token likelihood features, and nothing else
          io        tokens + top-k output summary + full-dist scalars + likelihood feats
          io_nolik  the parent's O_io exactly
          act       tokens + an activation stream of M's (post_block0 => O_act,
                    post_block3 => SELF_seq, the report site)
        """

        def __init__(self, n_out, o_layer, o_embd, mode, act_dim=0, n_lik=6):
            super().__init__()
            self.mode = mode
            self.use_tok = mode != "lik"
            if self.use_tok or mode in ("io", "io_nolik"):
                self.wte = nn.Embedding(vocab_size, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            if mode in ("io", "io_nolik"):
                self.pshape = nn.Linear(obs_topk, o_embd)
                self.sproj = nn.Linear(4, o_embd)
            if mode in ("io", "lik"):
                self.lproj = nn.Linear(n_lik, o_embd)
            if mode == "act":
                self.aproj = nn.Linear(act_dim, o_embd)
            self.h = nn.ModuleList([
                Block(o_embd, min(max(o_layer, 1) * 2, 8), T) for _ in range(o_layer)])
            if not observer_causal:
                for blk in self.h:
                    blk.attn.bias.fill_(1.0)       # bidirectional: generous to the third
            self.ln_f = nn.LayerNorm(o_embd)       # party, as the parent does
            self.head = nn.Linear(o_embd, n_out)

        def forward(self, tok, topk_ids=None, topk_p=None, stats=None, lik=None,
                    act=None):
            B, t = tok.shape
            x = self.wpe(torch.arange(t, device=tok.device))[None].expand(B, t, -1)
            if self.use_tok:
                x = x + self.wte(tok)
            if self.mode in ("io", "io_nolik"):
                x = x + (self.wte(topk_ids) * topk_p.unsqueeze(-1)).sum(-2) \
                      + self.pshape(topk_p) + self.sproj(stats)
            if self.mode in ("io", "lik"):
                x = x + self.lproj(lik)
            if self.mode == "act":
                x = x + self.aproj(act)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    out_dir = f"{DATA_DIR}/a2a_forward/confabulation/authorship/{tag or 'untagged'}"
    os.makedirs(out_dir, exist_ok=True)
    results = {"config": {}, "val_standalone": val_base, "sets": {}}
    saved = {}

    # ============ per-set battery ============
    for sname, smode, stemp, ladder in set_specs:
        rep_tok = report_sets[sname]["tok"]
        plan = report_sets[sname]["plan"]
        print(f"\n{'='*70}\n  SET {sname}   (mode={smode} temp={stemp} ladder={ladder})"
              f"\n{'='*70}")
        t_set = time.time()

        # ---- cache the M-side of this report set ----
        c_a0, c_aj, c_ar, c_ids, c_ps, c_st, c_lik, c_mass = [], [], [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, N, 32):
                xb = rep_tok[i:i + 32].to(device)
                lg, _, vi = model(xb, return_intermediates=True)
                lp_full = F.log_softmax(lg, dim=-1)
                p = lp_full.exp()
                tp, ti = p.topk(obs_topk, dim=-1)
                c_mass.append(float(tp.sum(-1).mean()) * xb.shape[0])
                top2 = lg.topk(2, dim=-1).values
                ent_all = -(p * lp_full).sum(-1)                        # (B,T)
                c_st.append(torch.stack([
                    ent_all, tp[..., 0], top2[..., 0] - top2[..., 1],
                    (1.0 - tp.sum(-1)).clamp_min(0),
                ], dim=-1).cpu())
                # Per-token likelihood features, ALIGNED TO THE TOKEN AT THAT POSITION:
                # the quantities at position t describe x_t under M given x_<t, so they
                # come from the distribution emitted at t-1. Position 0 has none, and is
                # inside the (unscored) prefix in every set.
                lg_prev, lp_prev = lg[:, :-1], lp_full[:, :-1]
                tgt = xb[:, 1:]
                logp = lp_prev.gather(-1, tgt[..., None]).squeeze(-1)
                rank = (lg_prev > lg_prev.gather(-1, tgt[..., None])).sum(-1).float()
                e_prev, mx_prev = ent_all[:, :-1], tp[:, :-1, 0]
                feats = torch.stack([
                    logp,                                    # log p_M(x_t | x_<t)
                    torch.log1p(rank),                       # log-rank of x_t
                    e_prev,                                  # M's uncertainty at t-1
                    mx_prev,                                 # M's confidence at t-1
                    -logp - e_prev,                          # surprisal minus entropy
                    logp - mx_prev.clamp_min(1e-9).log(),    # log ratio to the mode
                ], dim=-1)
                c_lik.append(torch.cat([torch.zeros_like(feats[:, :1]), feats],
                                       dim=1).cpu())
                c_a0.append(vi[predict_from].cpu()); c_aj.append(vi[predict_to].cpu())
                c_ar.append(vi[report_block].cpu())
                c_ids.append(ti.cpu()); c_ps.append(tp.cpu())
        a0 = torch.cat(c_a0); aj = torch.cat(c_aj); a_rep = torch.cat(c_ar)
        topk_ids = torch.cat(c_ids); topk_p = torch.cat(c_ps); io_stats = torch.cat(c_st)
        lik_raw = torch.cat(c_lik)
        topk_mass = sum(c_mass) / N
        del c_a0, c_aj, c_ar, c_ids, c_ps, c_st, c_lik
        torch.cuda.empty_cache()

        # standardize the likelihood features on TRAIN positions only
        lf = lik_raw[:, :T - 1].reshape(-1, lik_raw.shape[-1]).contiguous()
        mu, sd = lf[tr].mean(0), lf[tr].std(0).clamp_min(1e-6)
        lik = (lik_raw - mu) / sd
        lik[:, 0] = 0.0
        del lik_raw

        y_mine_seq = plan[:, :T - 1].long()                             # (N,T-1)
        y_mine = y_mine_seq.reshape(-1)
        y_mine_masked = y_mine_seq.clone()
        y_mine_masked[~scored_seq] = -100
        bal = _balance(y_mine[te], 2)
        lab_te = y_mine[te].numpy()

        # descriptive: the public signal, with no training at all
        lp_te = lf[te, 0].numpy()
        auc_surp = _auc(lp_te, lab_te)
        print(f"  cached: top{obs_topk} mass={topk_mass:.4f}  chance={bal:.3f}  "
              f"mine-frac(test)={lab_te.mean():.3f}")
        print(f"  descriptive: mean log p_M  mine={lp_te[lab_te > .5].mean():+.3f} "
              f"corpus={lp_te[lab_te < .5].mean():+.3f}   "
              f"AUC(raw surprisal, untrained)={auc_surp:.3f}")

        # ---- report-head inputs: the real blocks between a_j and the report site ----
        mids = [model.transformer.h[i] for i in mid_blocks]

        def make_report_input(aj_seq):
            o = []
            with torch.no_grad():
                for i in range(0, aj_seq.shape[0], 64):
                    z = aj_seq[i:i + 64].to(device)
                    for blk in mids:
                        z = blk(z)
                    o.append(z.cpu())
            return flat(torch.cat(o))

        X_full = make_report_input(aj)
        recon = float((X_full - flat(a_rep)).abs().max())
        assert recon < 1e-3, f"report-input reconstruction mismatch ({recon:.2e})"

        # ---------------- contestants ----------------
        def mlp_row(yv, nout, kind, Xv, tname, want_probs=False):
            """Per-position MLP report head: full access, the four ablations, and the
            fair confabulator (trained AND evaluated on theory-only access)."""
            h = _fit_head(_make_head(n_embd, nout, head_hidden, device, seed),
                          Xv["full"], yv, tr, head_steps, head_lr, device, kind)
            row = {"full": _head_score(h, Xv["full"], yv, te, device, kind)}
            for v in ("shuffle_r", "shuffle_p", "zero_r", "zero_p"):
                if v in Xv:
                    row[f"eval_{v}"] = _head_score(h, Xv[v], yv, te, device, kind)
            if "zero_r" in Xv:
                hc = _fit_head(_make_head(n_embd, nout, head_hidden, device, seed),
                               Xv["zero_r"], yv, tr, head_steps, head_lr, device, kind)
                row["confab"] = _head_score(hc, Xv["zero_r"], yv, te, device, kind)
            probs = None
            if want_probs:
                with torch.no_grad():
                    o = torch.cat([h(Xv["full"][te[i:i + 16384]].to(device)).cpu()
                                   for i in range(0, len(te), 16384)])
                probs = torch.softmax(o.float(), -1)[:, 1].numpy()
                row["auc"] = _auc(probs, yv[te].numpy())
            abl = " ".join(f"{k[5:]}={row[k]:.3f}" for k in row if k.startswith("eval_"))
            print(f"  [SELF_mlp/{tname:8s}] full={row['full']:.3f}"
                  + (f" confab={row['confab']:.3f}" if "confab" in row else "")
                  + (f" auc={row['auc']:.3f}" if "auc" in row else "")
                  + (f"  | eval: {abl}" if abl else ""))
            return row, probs

        def train_seqnet(yseq, nout, kind, o_layer, o_embd, mode, act_src=None,
                         data_frac=1.0, want_probs=False):
            torch.manual_seed(seed + 31)
            o = Observer(nout, o_layer, o_embd, mode,
                         act_dim=n_embd, n_lik=lik.shape[-1]).to(device)
            opt = torch.optim.AdamW(o.parameters(), lr=obs_lr, weight_decay=0.01)
            g = torch.Generator().manual_seed(seed + 32)
            pool = s_tr[:max(1, int(data_frac * len(s_tr)))]

            def fwd(si):
                xb = rep_tok[si, :T - 1].to(device)
                kw = {}
                if mode in ("io", "io_nolik"):
                    kw.update(topk_ids=topk_ids[si, :T - 1].to(device),
                              topk_p=topk_p[si, :T - 1].to(device),
                              stats=io_stats[si, :T - 1].to(device))
                if mode in ("io", "lik"):
                    kw["lik"] = lik[si, :T - 1].to(device)
                if mode == "act":
                    kw["act"] = act_src[si, :T - 1].to(device)
                return o(xb, **kw)

            for _ in range(obs_steps):
                si = pool[torch.randint(len(pool), (obs_bs,), generator=g)]
                out = fwd(si)
                t_ = yseq[si].to(device)
                if kind == "cls":
                    loss = F.cross_entropy(out.reshape(-1, nout), t_.reshape(-1),
                                           ignore_index=-100)
                else:                       # cosine, restricted to the scored positions
                    m = scored_seq[si].to(device)
                    loss = ((1.0 - F.cosine_similarity(out, t_, dim=-1)) * m).sum() \
                        / m.sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
                opt.step(); opt.zero_grad()

            o.eval()
            acc = tot = 0.0
            pr = []
            with torch.no_grad():
                for i in range(0, len(s_te), 32):
                    si = s_te[i:i + 32]
                    out = fwd(si).cpu()
                    t_ = yseq[si]
                    if kind == "cls":
                        m = t_ != -100
                        acc += float((out.argmax(-1)[m] == t_[m]).sum())
                        tot += int(m.sum())
                        if want_probs:
                            pr.append(torch.softmax(out.float(), -1)[..., 1][m])
                    else:
                        m = scored_seq[si]
                        acc += float((F.cosine_similarity(out, t_, dim=-1) * m).sum())
                        tot += int(m.sum())
            probs = torch.cat(pr).numpy() if (want_probs and pr) else None
            del o
            torch.cuda.empty_cache()
            return acc / tot, probs

        def run_ladder(yseq, nout, kind, tname, ladder_mode, modes, act_rep,
                       collect=None):
            """Test 1: the capacity-swept third-person ladder, plus SELF_seq (the report
            site read by an equally powerful sequence model), the O_act ceiling, the
            parent's likelihood-free O_io, and the half-data budget control."""
            rows, pb_out = {}, {}
            ol_, oe_ = top_cap
            cap_list = caps if ladder_mode == "full" else [top_cap]
            for (o_layer, o_embd) in cap_list:
                cp = f"{o_layer}L{o_embd}D"
                want_top = (o_layer, o_embd) == top_cap and collect is not None
                for m_ in modes:
                    sc, pb = train_seqnet(yseq, nout, kind, o_layer, o_embd, m_,
                                          want_probs=want_top)
                    rows[f"O_{m_}@{cp}"] = sc
                    if want_top:
                        pb_out[f"O_{m_}"] = pb
                sc, pb = train_seqnet(yseq, nout, kind, o_layer, o_embd, "act",
                                      act_src=act_rep, want_probs=want_top)
                rows[f"SELF_seq@{cp}"] = sc
                if want_top:
                    pb_out["SELF_seq"] = pb
            rows[f"O_act@{ol_}L{oe_}D"] = train_seqnet(
                yseq, nout, kind, ol_, oe_, "act", act_src=a0)[0]
            rows[f"O_io_nolik@{ol_}L{oe_}D"] = train_seqnet(
                yseq, nout, kind, ol_, oe_, "io_nolik")[0]
            rows[f"O_io@{ol_}L{oe_}D_half_data"] = train_seqnet(
                yseq, nout, kind, ol_, oe_, "io", data_frac=0.5)[0]
            best_io = max(v for k, v in rows.items()
                          if k.startswith("O_io@") and "half" not in k)
            print(f"  [ladder/{tname:8s}] best_O_io={best_io:.3f}  | "
                  + " ".join(f"{k}={v:.3f}" for k, v in rows.items()))
            if collect is not None:
                collect.update(pb_out)
            return rows

        set_res = {"mode": smode, "temp": stemp, "topk_mass": topk_mass,
                   "chance": bal, "mine_frac_test": float(lab_te.mean()),
                   "auc_raw_surprisal": auc_surp,
                   "logp_mine": float(lp_te[lab_te > .5].mean()),
                   "logp_corpus": float(lp_te[lab_te < .5].mean()),
                   "report": {}, "observers": {}, "baselines": {"MINE": bal}}
        preds = {}

        # ---------------- the instrument, only on the full set ----------------
        Xv = {"full": X_full}
        run_impl = (ladder == "full") and not skip_impl
        if run_impl:
            def train_fresh_fm(fm_seed, d_head, mlp_mult):
                torch.manual_seed(fm_seed)
                fm = TransformerForwardModel(d_model=n_embd, d_head=d_head,
                                             n_head=fwd_n_head, n_layer=fwd_n_layer,
                                             mlp_mult=mlp_mult, block_size=T).to(device)
                opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
                g = torch.Generator().manual_seed(fm_seed + 1)
                for _ in range(fresh_fm_steps):
                    fm.train()
                    xb, yb = get_batch(train_data, g)
                    with torch.no_grad():
                        _, _, vi = model(xb, yb, return_intermediates=True)
                    F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
                    torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                fm.eval()
                for p_ in fm.parameters():
                    p_.requires_grad = False
                return fm

            def fm_predict(fmx, src, n_seq=None):
                limit = len(src) if n_seq is None else min(n_seq, len(src))
                with torch.no_grad():
                    return torch.cat([fmx(src[i:min(i + 64, limit)].to(device)).cpu()
                                      for i in range(0, limit, 64)])

            fms = [train_fresh_fm(seed + 911 + 37 * jj, inst_dh, inst_mm)
                   for jj in range(max(1, ens_n))]
            fm_params = sum(p.numel() for p in fms[0].parameters())
            pred = fm_predict(fms[0], a0)
            resid = aj - pred
            fwd_cos = float(F.cosine_similarity(pred, aj, dim=-1).mean())
            nd = min(1000, N)
            ens = (_ensemble_cos([(aj[:nd] - fm_predict(f_, a0, nd)).reshape(-1, n_embd)
                                  for f_ in fms]) if len(fms) > 1 else float("nan"))
            y_world_full = _syntactic_labels(rep_tok, vocab_size, enc)
            rstruct = _residual_structure(resid[:nd].reshape(-1, n_embd),
                                          y_world_full[:nd].reshape(-1))
            del fms
            torch.cuda.empty_cache()
            _CATS = {1: "sent_start", 2: "after_punct", 3: "after_opener",
                     4: "before_closer"}
            print(f"\n  --- instrument FM {inst_cap_str}: {fm_params/1e3:.0f}K "
                  f"({100*fm_params/n_pred_params:.1f}% of {n_pred_blocks} predicted "
                  f"blocks) ---")
            print(f"      cosine={fwd_cos:.4f}  "
                  f"|r|={float(resid.norm(dim=-1).mean()):.3f}  ens_cos={ens:.3f}")
            print(f"      eta2_norm={rstruct['eta2_norm']:.4f} "
                  f"eta2_dir={rstruct['eta2_dir']:.4f} "
                  f"eta2_vec={rstruct['eta2_vec']:.4f}  |r| Cohen's d: "
                  + " ".join(f"{v}={rstruct['cohens_d'][k]:+.2f}"
                             for k, v in _CATS.items()))

            sh = torch.randperm(N * T,
                                generator=torch.Generator().manual_seed(seed + 13))
            rsf = lambda z: z.reshape(N * T, n_embd)[sh].reshape(N, T, n_embd)
            Xv = {"full": X_full,
                  "shuffle_r": make_report_input(pred + rsf(resid)),
                  "shuffle_p": make_report_input(rsf(pred) + resid),
                  "zero_r": make_report_input(pred),
                  "zero_p": make_report_input(resid)}
            set_res["instrument"] = {
                "cap": inst_cap_str, "fm_params": fm_params,
                "pct_of_predicted": 100 * fm_params / n_pred_params,
                "fwd_cosine": fwd_cos, "ens_cos": ens,
                "res_norm": float(resid.norm(dim=-1).mean()),
                "residual_structure": rstruct}

        # ---------------- MINE ----------------
        row, p_mlp = mlp_row(y_mine, 2, "cls", Xv, "MINE", want_probs=True)
        set_res["report"]["MINE"] = row
        preds["SELF_mlp"] = p_mlp
        set_res["observers"]["MINE"] = run_ladder(
            y_mine_masked, 2, "cls", "MINE", ladder, ["input", "lik", "io"], a_rep,
            collect=preds)

        # LIK_pos: the no-aggregation floor of the likelihood route
        Xlik = lik[:, :T - 1].reshape(-1, lik.shape[-1]).contiguous()
        hl = _fit_head(_make_head(Xlik.shape[-1], 2, head_hidden, device, seed),
                       Xlik, y_mine, tr, head_steps, head_lr, device, "cls")
        lik_pos = _head_score(hl, Xlik, y_mine, te, device, "cls")
        with torch.no_grad():
            o_ = torch.cat([hl(Xlik[te[i:i + 16384]].to(device)).cpu()
                            for i in range(0, len(te), 16384)])
        preds["LIK_pos"] = torch.softmax(o_.float(), -1)[:, 1].numpy()
        set_res["observers"]["MINE"]["LIK_pos"] = lik_pos
        print(f"  [LIK_pos /MINE    ] {lik_pos:.3f}  "
              f"auc={_auc(preds['LIK_pos'], lab_te):.3f}")

        set_res["auc"] = {k: _auc(v, lab_te) for k, v in preds.items() if v is not None}
        set_res["auc"]["raw_surprisal"] = auc_surp
        print("  AUC: " + "  ".join(f"{k}={v:.3f}" for k, v in set_res["auc"].items()))
        saved[sname] = {k: v for k, v in preds.items() if v is not None}
        saved[sname]["_label"] = lab_te.astype("float32")
        saved[sname]["_logp"] = lp_te.astype("float32")
        saved[sname]["_offset"] = report_sets[sname]["offset"][:, :T - 1] \
            .reshape(-1)[te].numpy()
        saved[sname]["_pos"] = (te % (T - 1)).numpy()
        saved[sname]["_seq"] = (te // (T - 1)).numpy()

        # ---------------- IMPL positive control, same positions ----------------
        if run_impl:
            R = flat(resid)
            cents = _kmeans_fit(R[tr].to(device), impl_k, seed=seed)
            y_impl = _kmeans_assign(R.to(device), cents).cpu()
            cw, cb, cs = _cluster_quality(R.to(device), y_impl.to(device), cents)
            y_impl_cos = F.normalize(R, dim=-1)
            set_res["baselines"]["IMPL"] = _balance(y_impl[te], impl_k)
            set_res["baselines"]["IMPL_COS"] = 0.0
            set_res["cluster_quality"] = {"within": cw, "between": cb, "separation": cs}
            print(f"      residual cluster quality: within={cw:.3f} between={cb:.3f} "
                  f"separation={cs:.3f}  chance={set_res['baselines']['IMPL']:.3f}")

            yi_seq = y_impl.view(N, T - 1).clone()
            yi_seq[~scored_seq] = -100
            set_res["report"]["IMPL"] = mlp_row(y_impl, impl_k, "cls", Xv, "IMPL")[0]
            set_res["observers"]["IMPL"] = run_ladder(
                yi_seq, impl_k, "cls", "IMPL", "full", ["input", "io"], a_rep)
            set_res["report"]["IMPL_COS"] = mlp_row(y_impl_cos, n_embd, "cos", Xv,
                                                    "IMPL_COS")[0]
            set_res["observers"]["IMPL_COS"] = run_ladder(
                y_impl_cos.view(N, T - 1, n_embd), n_embd, "cos", "IMPL_COS", "full",
                ["input", "io"], a_rep)
            del R, y_impl, y_impl_cos, cents, pred, resid

        results["sets"][sname] = set_res
        print(f"  set '{sname}' done in {(time.time() - t_set)/60:.1f} min  "
              f"[elapsed {(time.time() - t_start)/60:.1f} min]")

        # incremental save, so a crash late in the job does not cost the earlier sets
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        np.savez_compressed(f"{out_dir}/predictions.npz",
                            **{f"{s}|{k}": v for s, d in saved.items()
                               for k, v in d.items()})
        volume.commit()

        report_sets[sname]["tok"] = None
        del a0, aj, a_rep, topk_ids, topk_p, io_stats, lik, lf, X_full, Xv, rep_tok
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    results["config"] = {
        "cond": cond, "n_tokens": n_tokens, "block_size": T, "n_layer": n_layer,
        "n_head": n_head, "n_embd": n_embd, "predict_from": predict_from,
        "predict_to": predict_to, "report_block": report_block, "n_steps": n_steps,
        "n_report_sequences": N, "prefix_len": prefix_len, "span_lens": span_lens,
        "temperature": temperature, "temperature_lo": temperature_lo,
        "inst_cap": inst_cap_str, "ens_n": ens_n, "impl_k": impl_k,
        "observer_caps": observer_caps, "obs_steps": obs_steps, "obs_topk": obs_topk,
        "observer_causal": observer_causal, "seed": seed, "smoke": smoke,
        "n_pred_params": n_pred_params, "wake_ckpt": ck_path}
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'='*78}\nSUMMARY   advantage = best self-report - best I/O observer")
    for sname, sr in results["sets"].items():
        print(f"\n  {sname}  (mode={sr['mode']} temp={sr['temp']} chance={sr['chance']:.3f}"
              f"  AUC raw surprisal={sr['auc_raw_surprisal']:.3f})")
        for tn in ("MINE", "IMPL", "IMPL_COS"):
            if tn not in sr["report"]:
                continue
            obs = sr["observers"][tn]
            g = lambda pre: max([v for k, v in obs.items()
                                 if k.startswith(pre) and "half" not in k]
                                or [float("nan")])
            sm, ss = sr["report"][tn]["full"], g("SELF_seq")
            best_self = max(sm, ss) if ss == ss else sm
            # criterion 2 is about the best third party we could build, so the advantage
            # is taken against the strongest observer rung of any kind. O_act is NOT a
            # third party -- it is handed M's own state -- so it is quoted separately.
            best_obs = max([v for k, v in obs.items()
                            if (k.startswith(("O_input", "O_lik", "O_io", "LIK_pos"))
                                and "half" not in k)] or [float("nan")])
            print(f"    {tn:9s} SELF_mlp={sm:.3f} SELF_seq={ss:.3f} | "
                  f"O_input={g('O_input'):.3f} O_lik={g('O_lik'):.3f} "
                  f"O_io={g('O_io@'):.3f} O_io_nolik={g('O_io_nolik'):.3f} "
                  f"O_act={g('O_act'):.3f} | adv(vs O_io)={best_self - g('O_io@'):+.3f}"
                  f"  adv(vs best obs)={best_self - best_obs:+.3f}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\n  peak RSS = {peak:.2f} GB   wall = {(time.time()-t_start)/60:.1f} min")
    print(f"  saved -> {out_dir}/results.json  (+ predictions.npz)\n{'='*78}")
    return results
