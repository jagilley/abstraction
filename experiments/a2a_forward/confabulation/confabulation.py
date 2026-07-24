"""The Confabulation Test on language: does a self-report track the implementation
or a self-theory?

Design doc: ideas/confabulation_test.md  (pre-registration; read it first)
Sibling:    experiments/rhm/rhm_confabulation.py  (the RHM instantiation this ports,
            including its instrument-FM capacity sweep -- see THE JUNK-RESIDUAL TRAP)
Substrate:  a2a_forward/stages.py::a2a_loop_train -- this file forks that wake recipe
            (same GPT, TransformerForwardModel, CerebellarGate, lrs, FineWeb-Edu tokens)
            and adds the report battery on top, in the Run-6 controlled-retrain shape
            (shared init / seed / data order; only the loop differs between arms).

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

WHY LANGUAGE, GIVEN RHM EXISTS
  The design doc chose RHM and explicitly deferred language on cost. Three things make
  the language arm worth its price, and they are the reasons to read the two together:
    - No aux-target crutch. RHM only has generalizable self-knowledge under the *latent*
      target (`ntp_aux_cl`); the plain token target is inverted (fresh-FM SK -0.67), so
      the RHM battery had to be run on a non-canonical arm. On language the canonical
      closed loop already has the self-knowledge (Run 6: residual-probe R2 0.42 CL vs
      0.26 OL, directional not magnitude-based), so the battery runs on the *shipped*
      recipe with no aux head anywhere.
    - The hard case for IMPL. Language's residual is diffuse and full-rank (eff. rank
      199.8/256, top-1 PC 2.4%) where MNIST's is low-rank and RHM's is a deep DGP gap.
      If a residual report is legible *here* it is not riding on a few fat PCs. This is
      also the main risk (see IMPL_COS below).
    - A real I/O map to be third-person about. O_io on RHM predicts a 16-way
      distribution; here it predicts a 50257-way one that carries genuine linguistic
      information, which is exactly the observer that has to fail for criterion (2).

THE JUNK-RESIDUAL TRAP (why every IMPL number is swept over instrument capacity)
  The residual only means anything if the FM is small enough that what it misses is a
  genuine COMPUTATIONAL gap. An over-capacity FM saturates (cosine -> 0.999) and `r`
  degenerates into architectural-mismatch noise -- which would FAKE this experiment's
  headline: M can report noise (the report head sits downstream of a_j), while no third
  party can predict noise from tokens+logits. IMPL would then show a large
  "introspective advantage" built entirely from junk.

  This is a live risk here, not a hypothetical. The a2a capacity scaling result is that
  an FM saturates once it reaches roughly the parameter count of the layers it predicts,
  and the *default* a2a forward model (2L/d_head 64/mlp 2, 660K) is already ~42% of the
  two blocks it predicts. So the default instrument sits at the saturation point.

  Every capacity point therefore carries two discriminators, and no IMPL number may be
  read without them:
    ens_cos -- do INDEPENDENT fresh FMs (different seeds) leave the SAME residual? high
               => input-determined, a real gap any FM misses identically; low =>
               FM-idiosyncratic noise.
    eta2    -- is the residual conditioned on the syntactic structure of the position?
               A genuine computational gap is; noise is not. Reported as `eta2_norm`
               (variance of the residual MAGNITUDE explained by category) alongside
               per-category Cohen's d on |r|, which is directly comparable to the
               published table (before_closer +0.84, sentence_start -0.85, after_punct
               -0.62, after_opener +0.62). NB: eta2 on the raw residual VECTOR is the
               wrong yardstick and runs ~1e-3 even when the magnitude effect is large --
               it is a location statistic and the published effect is a scale one. It is
               still printed (`eta2_vec`) so the discrepancy stays visible.

WHAT THIS SCRIPT MEASURES
  Report targets, all emitted from `report_block` on standalone forward passes (fresh-FM
  instrument -- see the report-mode note below):
    IMPL      -- K-way cluster of the direction of M's own FM residual r_t  [implementation]
    IMPL_COS  -- the residual *direction itself*, scored by cosine (continuous)
    BEHAV     -- is M's own next-token argmax correct                       [I/O map]
    ENT       -- M's own output-entropy quartile                            [I/O map]
    WORLD     -- syntactic category of the position, from tokens alone      [input]

  IMPL_COS is a language-specific addition, and it is load-bearing rather than cosmetic.
  K-way clustering of a diffuse full-rank residual can be weakly separated, and a null on
  IMPL would then be a fact about k-means rather than about introspection. IMPL_COS puts
  no cluster structure in the way: the head regresses the unit residual direction and is
  scored by test cosine. Read the two together -- IMPL for the categorical battery
  (ablations, steering flips), IMPL_COS for whether the signal is there at all. The
  cluster-quality diagnostic printed alongside says which reading to trust.

  Test 1  observer ladder   O_input(c) / O_io(c) (tokens + a faithful summary of M's own
                            output distribution) / O_act (ceiling: post_block0 -> target,
                            i.e. FM + a probe), each swept over capacity, plus a
                            half-data control on the strongest observer.
  Test 2  channel ablation  substitute a_j at report time: shuffle_r / shuffle_p
                            (distribution-preserving, position-destroying) and zero_r /
                            zero_p. IMPL should collapse under shuffle_r only.
  Test 3  matched-KL        steer a_j along PCs of the RESIDUAL span vs PCs of the
          steering          FM-PREDICTION span, every direction rescaled to the same KL
                            on the logits. At matched behaviour a theory-driven report
                            cannot move more for one family than the other.
  Test 4  confabulator      a report head trained AND evaluated with access restricted to
                            FM(a_i) -- right target, theory-only access. Its accuracy is
                            the confabulation ceiling; full - confab is the introspective
                            margin. Without this control a null on Test 1 is uninterpretable.

DELIBERATE DEVIATIONS FROM THE RHM PORT (all flagged, all defaulted to the safer choice)
  - `predict_to=post_block2`, report from `post_block3`. The canonical language gap is
    post_block0->post_block3, but with 4 layers that puts the report site *at* the FM
    target, so the head would read a_j linearly and Tests 2/3 would degenerate into
    input perturbations of a linear readout. Predicting to block2 leaves one real block
    between target and report, exactly as RHM had (predict_to=block6, report=block7), so
    the report is *computed* by M. Pass `--predict-to post_block3` to run the canonical
    gap and accept the degenerate readout.
  - Observers are **non-causal by default** (`--observer-causal` to restore RHM's causal
    stack). A bidirectional observer over the whole sequence is strictly more generous to
    the third party, and the whole argument rests on the third party being given every
    advantage we can afford.
  - O_io sees a top-k summary of M's output distribution, not all 50257 logits (which
    cannot be cached at this scale): the top-k ids and probabilities, PLUS four scalars
    computed from the full distribution (entropy, max prob, top1-top2 logit margin, tail
    mass outside the top-k). The scalars are not optional -- the retained top-k mass is
    measured and printed, and can be well under 1.0, so without them the observer could
    not even compute M's entropy and the ENT control would show a fake advantage.
  - The full observer ladder runs at the default instrument capacity; the other capacity
    points run only the strongest O_io plus the O_act ceiling, which are the two numbers
    that gate the headline. `--full-ladder-every-cap` restores the RHM behaviour at ~4x
    the observer cost.
  - Report-time passes are **standalone (no injection) with a fresh FM**, per the RHM/
    latent-loop convention, so both arms are read by the same instrument and the CL arm's
    residual is not the one it was literally trained against. This is the conservative
    choice and it is *unkind to CL*, which is known to be injection-dependent (~0.11
    nats). `--report-native` additionally re-reads the CL arm in its native injected mode.

Run (NOTE: the FineWeb-Edu token shards live on the `jagilley` workspace volume, not
`chromatic` -- run `modal profile activate jagilley` first):

  # smoke (a few minutes, everything tiny -- checks wiring only, numbers meaningless)
  modal run a2a_forward/confabulation/confabulation.py::confabulation_test --smoke

  # headline: both arms, full battery
  modal run --detach a2a_forward/confabulation/confabulation.py::confabulation_test \
      --conditions "cl,ol" --tag main
"""

import json
import os

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


# ======================================================================
# Small utilities (shared shape with rhm_confabulation)
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


def _cluster_quality(X, labels, C, chunk=65536):
    """How much cluster structure does the residual direction actually have?

    Returns (within, between, separation). `within` is the mean cosine of a row to its
    own centroid, `between` the mean cosine to the others. On a diffuse full-rank
    residual these come out close together and the K-way IMPL target is weak *as a
    target* -- which is a fact about k-means, not about introspection. Read IMPL_COS
    instead when separation is small.
    """
    import torch
    Xn = torch.nn.functional.normalize(X, dim=-1)
    K = C.shape[0]
    w_sum, b_sum, n = 0.0, 0.0, 0
    for i in range(0, Xn.shape[0], chunk):
        sims = Xn[i:i + chunk] @ C.T                       # (chunk, K)
        lab = labels[i:i + chunk]
        own = sims.gather(1, lab[:, None]).squeeze(1)
        w_sum += float(own.sum())
        b_sum += float((sims.sum(1) - own).sum()) / max(K - 1, 1)
        n += sims.shape[0]
    within, between = w_sum / n, b_sum / n
    return within, between, within - between


def _residual_structure(res, lab, n_cls=5):
    """Is the residual conditioned on the *type of computation* at this position?

    The language stand-in for rhm_confabulation's hierarchy-eta2, and the second junk-
    residual discriminator: a genuine computational gap is conditioned on the input,
    architectural-mismatch NOISE is not.

    Three statistics, because they are not interchangeable and only the last is
    comparable to the published table:

      eta2_vec  -- variance of the raw residual VECTOR explained by the label. This is a
                   *location* effect (do the group-mean vectors differ). It is the
                   obvious thing to compute and it is the WRONG yardstick here: the
                   published language signature is a difference in residual MAGNITUDE by
                   category, and a category whose residuals share the grand mean
                   direction but are uniformly larger moves the group mean barely at all.
                   On a diffuse full-rank residual this runs ~1e-3 even when the
                   magnitude effect is large. Kept only so the discrepancy is visible.
      eta2_norm -- variance of the residual NORM explained by the label: the *scale*
                   effect, i.e. the thing the published Cohen's d is about.
      eta2_dir  -- variance of the UNIT residual direction explained by the label: is
                   the direction category-conditioned, independent of magnitude.
      cohens_d  -- per category, |r| in-category vs out-of-category, pooled sd. Directly
                   comparable to the published table (before_closer +0.84, sentence_start
                   -0.85, after_punct -0.62, after_opener +0.62). Sign and rough ordering
                   agreeing is real evidence the residual is the published object; all
                   four near zero alongside a low ens_cos is the junk signature.

    res: (n, d) cpu float. lab: (n,) cpu long."""
    import torch
    import torch.nn.functional as F
    r = res.float()
    nrm = r.norm(dim=-1)
    unit = F.normalize(r, dim=-1)

    def _eta2(x):
        gmean = x.mean(0, keepdim=True)
        tot = float(((x - gmean) ** 2).sum())
        if tot <= 0:
            return 0.0
        between = 0.0
        for g in torch.unique(lab):
            sel = lab == g
            between += float(int(sel.sum()) * ((x[sel].mean(0) - gmean) ** 2).sum())
        return between / tot

    ds = {}
    for g in range(n_cls):
        sel = lab == g
        n_in = int(sel.sum())
        if n_in < 2 or n_in > len(lab) - 2:
            ds[g] = float("nan")
            continue
        a, b = nrm[sel], nrm[~sel]
        sd = (((n_in - 1) * a.var() + (len(lab) - n_in - 1) * b.var())
              / (len(lab) - 2)).sqrt()
        ds[g] = float((a.mean() - b.mean()) / sd) if float(sd) > 0 else 0.0
    return {"eta2_vec": _eta2(r), "eta2_norm": _eta2(nrm[:, None]),
            "eta2_dir": _eta2(unit), "cohens_d": ds}


def _ensemble_cos(residuals):
    """Input-centered pairwise cosine between the residuals left by INDEPENDENT fresh
    FMs (different seeds) on the same frozen model. High => the residual is determined by
    the input (an FM-invariant computational gap that any FM misses identically); low =>
    it is determined by the particular FM, i.e. idiosyncratic architectural-mismatch
    noise. This is the discriminator between 'genuine computational gap' and 'junk', and
    it gates the whole battery: reporting on junk is not introspection about anything.
    (Recipe from rhm_latent_loop._ensemble_agreement.)"""
    import torch.nn.functional as F
    cent = [r - r.mean(dim=0, keepdim=True) for r in residuals]   # drop per-FM offset
    sims, n = [], len(cent)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(F.cosine_similarity(cent[i], cent[j], dim=-1).mean()))
    return float(sum(sims) / len(sims)) if sims else float("nan")


def _make_head(d_in, n_out, hidden, device, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, n_out)).to(device)


def _fit_head(net, X, y, tr, steps, lr, device, kind="cls", bs=4096):
    """`kind='cls'` -> cross-entropy on integer labels; `kind='cos'` -> cosine loss
    against a unit target direction (the IMPL_COS regression variant)."""
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        si = tr[torch.randint(len(tr), (min(bs, len(tr)),), generator=g)]
        out = net(X[si].to(device))
        if kind == "cls":
            loss = F.cross_entropy(out, y[si].to(device))
        else:
            loss = (1.0 - F.cosine_similarity(out, y[si].to(device), dim=-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    return net


def _head_score(net, X, y, te, device, kind="cls", bs=16384):
    """Test accuracy (kind='cls') or mean test cosine (kind='cos')."""
    import torch
    import torch.nn.functional as F
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for i in range(0, len(te), bs):
            si = te[i:i + bs]
            out = net(X[si].to(device))
            if kind == "cls":
                acc += float((out.argmax(-1).cpu() == y[si]).sum())
            else:
                acc += float(F.cosine_similarity(out, y[si].to(device), dim=-1).sum())
    net.train()
    return acc / len(te)


def _balance(labels, K):
    """Majority-class fraction -- the trivial baseline any head must beat."""
    import torch
    counts = torch.bincount(labels, minlength=K).float()
    return float(counts.max() / counts.sum())


def _syntactic_labels(tok, vocab_size, enc):
    """5-way syntactic category per position, a deterministic function of the INPUT
    tokens -- the language stand-in for RHM's ground-truth DGP ancestor label.

    Categories (priority order): sentence_start > after_punct > after_opener >
    before_closer > mid_sentence. Definitions are lifted from
    behavioral_residual.classify_tokens_syntactic (the same taxonomy the residual
    conditioning results are reported in), but vectorized via a per-vocab character
    table instead of per-position decoding.

    Note `before_closer` looks at token t+1, so it is only available to a non-causal
    observer -- which ours is by default. With --observer-causal that class is partly
    unreachable from the prefix and WORLD stops being a clean 'nails it' control.
    """
    import numpy as np
    import torch
    PUNCT, OPENER, CLOSER = set(".,;:!?"), set("([{\"'`"), set(")]}")

    is_punct_end = np.zeros(vocab_size, dtype=bool)
    is_sent_end = np.zeros(vocab_size, dtype=bool)
    is_opener_end = np.zeros(vocab_size, dtype=bool)
    starts_closer = np.zeros(vocab_size, dtype=bool)
    for i in range(vocab_size):
        try:
            txt = enc.decode([i])
        except Exception:
            continue
        st = txt.strip()
        if st:
            if st[-1] in PUNCT:
                is_punct_end[i] = True
                if st[-1] in ".!?" or "\n" in txt:
                    is_sent_end[i] = True
            if st[-1] in OPENER:
                is_opener_end[i] = True
            if st[0] in CLOSER:
                starts_closer[i] = True

    x = tok.numpy()
    N, T = x.shape
    prev = np.concatenate([np.zeros((N, 1), dtype=x.dtype), x[:, :-1]], axis=1)
    nxt = np.concatenate([x[:, 1:], np.zeros((N, 1), dtype=x.dtype)], axis=1)
    lab = np.zeros((N, T), dtype=np.int64)                      # 0 = mid_sentence
    lab[starts_closer[nxt]] = 4                                 # before_closer
    lab[is_opener_end[prev]] = 3                                # after_opener
    lab[is_punct_end[prev]] = 2                                 # after_punct
    lab[is_sent_end[prev]] = 1                                  # sentence_start
    lab[:, 0] = 0                                               # no prefix at t=0
    return torch.from_numpy(lab)


# ======================================================================
# Main experiment
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=65536)
def confabulation_test(
    # --- data / model / FM: the a2a_loop_train defaults ---
    n_tokens: int = 10_000_000, block_size: int = 128,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block2",
    report_block: str = "", inject_after_block: int = 1,
    fwd_n_layer: int = 2, fwd_d_head: int = 64, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
    # --- wake training (forked from a2a_loop_train, Run-6 controlled-retrain shape) ---
    conditions: str = "cl,ol",
    n_steps: int = 10_000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    # --- report battery ---
    n_report_sequences: int = 3000, report_seed: int = 999,
    fresh_fm_steps: int = 3000, report_native: bool = False,
    # Instrument-FM capacity sweep, "(d_head:mlp_mult)" pairs at fwd_n_layer/fwd_n_head.
    # inst_caps[0] is the default and is where the FM-independent targets and the full
    # observer ladder are run. See THE JUNK-RESIDUAL TRAP in the module docstring: the
    # a2a default FM (64:2.0) is already ~42% of the blocks it predicts, i.e. at the
    # saturation point, so the sweep has to reach well below it.
    inst_caps_str: str = "16:0.5,4:0.25,32:1.0,64:2.0", ens_n: int = 3,
    impl_k: int = 8,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    # observer ladder: "n_layer:n_embd" points, last one matches M
    observer_caps: str = "1:64,2:128,4:256",
    obs_steps: int = 2000, obs_lr: float = 3e-4, obs_bs: int = 32,
    obs_topk: int = 64, observer_causal: bool = False,
    full_ladder_every_cap: bool = False,
    # Test 3
    steer_n_pc: int = 32, steer_eps: float = 1.0, steer_target_kl: float = 0.01,
    steer_n_seq: int = 128, steer_chunk: int = 16,
    eval_interval: int = 500, seed: int = 42, tag: str = "",
    refresh_wake: bool = False, smoke: bool = False,
):
    import glob
    import numpy as np
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from a2a_forward.model import GPT, Block
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    if smoke:
        n_steps, n_report_sequences, fresh_fm_steps = 300, 250, 200  # 250 % 64 != 0: exercises the fm_predict chunk-clamp path
        head_steps, obs_steps, n_tokens = 200, 150, 2_000_000
        observer_caps, eval_interval, steer_n_pc = "1:64,2:128", 100, 8
        steer_n_seq, steer_chunk = 32, 8
        inst_caps_str, ens_n = "16:0.5,4:0.25", 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = block_size
    cib = int(predict_from.replace("post_block", "").replace("post_embed", "-1"))
    j_idx = int(predict_to.replace("post_block", ""))
    report_block = report_block or f"post_block{n_layer - 1}"
    r_idx = int(report_block.replace("post_block", ""))
    cond_list = [c.strip() for c in conditions.split(",")]
    caps = [tuple(int(z) for z in c.split(":")) for c in observer_caps.split(",")]
    inst_caps = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in inst_caps_str.split(",")]      # (d_head, mlp_mult) pairs

    # blocks strictly between the FM target and the report site: the real computation
    # the report has to be routed through. Empty => degenerate linear readout (see the
    # deviation note in the module docstring).
    mid_blocks = list(range(j_idx + 1, r_idx + 1))
    # Parameter accounting: what matters for saturation is the FM against the BLOCKS IT
    # PREDICTS, not against the whole model (a2a capacity scaling -- the FM saturates
    # once it reaches roughly the parameter count of the predicted layers). On language
    # the whole-model ratio is badly misleading because ~26M of M's 28.9M params are the
    # tied-free embedding and lm_head, which the FM never has to model.
    n_pred_blocks = j_idx - cib
    n_pred_params = n_pred_blocks * 12 * n_embd * n_embd

    # ---------------- data (verbatim a2a_loop_train) ----------------
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        toks = np.load(path)
        all_tokens.append(toks)
        total += len(toks)
        if total >= n_tokens:
            break
    data = torch.from_numpy(np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    enc = tiktoken.get_encoding("gpt2")

    print(f"{'='*78}\nCONFABULATION TEST (language)  {n_layer}L/{n_head}H/{n_embd}D"
          f"  P={n_tokens:,}  T={T}{'  [SMOKE]' if smoke else ''}")
    print(f"  {predict_from}->{predict_to}  inject@{inject_after_block}  "
          f"report from {report_block}  (blocks between: {mid_blocks or 'NONE -- degenerate'})")
    print(f"  conditions={cond_list}  K={impl_k}  observers={caps}"
          f"  {'non-causal' if not observer_causal else 'causal'}  vocab={vocab_size:,}")
    print(f"  instrument-FM sweep (d_head:mlp_mult)={inst_caps}  ens_n={ens_n}   "
          f"predicted blocks={n_pred_blocks} ({n_pred_params/1e6:.2f}M params)\n{'='*78}")

    def get_batch(split_data, gen):
        ix = torch.randint(len(split_data) - T - 1, (batch_size,), generator=gen)
        x = torch.stack([split_data[i:i + T] for i in ix])
        y = torch.stack([split_data[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    def make_fm(d_head=None, mlp_mult=None):
        return TransformerForwardModel(
            d_model=n_embd, d_head=d_head or fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult if mlp_mult is None else mlp_mult,
            block_size=T).to(device)

    # shared init: Run-6 controlled-retrain -- identical lr/seed/init/data order, only
    # the loop differs, so any arm gap cannot be an lr or seed artifact.
    torch.manual_seed(seed)
    _init = GPT(vocab_size, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: v.cpu().clone() for k, v in _init.state_dict().items()}
    del _init
    torch.cuda.empty_cache()

    # ---------------- report/eval set: held-out val sequences ----------------
    rng = np.random.default_rng(report_seed)
    starts = rng.integers(0, len(val_data) - T - 1, size=n_report_sequences)
    rep_tok = torch.stack([val_data[i:i + T] for i in starts])                 # (N,T) cpu
    y_world_full = _syntactic_labels(rep_tok, vocab_size, enc)                 # (N,T) cpu
    N = rep_tok.shape[0]

    # ==================================================================
    # Observer: a third party that sees M's inputs, and optionally a faithful
    # per-position summary of M's outputs (O_io) or M's early activations (O_act).
    # ==================================================================
    class Observer(nn.Module):
        def __init__(self, n_out, o_layer, o_embd, mode, act_dim=0):
            super().__init__()
            self.mode = mode
            self.wte = nn.Embedding(vocab_size, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            if mode == "io":
                # M's output distribution, as a third party would summarize it: a
                # probability-weighted embedding of the top-k tokens (WHICH tokens M
                # favours), the sorted top-k probability vector (the SHAPE of the
                # distribution), and four scalars computed from the FULL 50257-way
                # distribution -- entropy, max prob, top1-top2 log margin, and the tail
                # mass outside the top-k. The scalars matter: at this scale the top-k
                # can hold well under half the mass, and without them the observer would
                # be unable to compute even the entropy, which would hand us a fake
                # advantage on the ENT control. With them, ENT is *given* to the
                # observer, which is exactly what an I/O-map fact should be.
                self.pshape = nn.Linear(obs_topk, o_embd)
                self.sproj = nn.Linear(4, o_embd)
            if mode == "act":
                self.aproj = nn.Linear(act_dim, o_embd)
            self.h = nn.ModuleList([
                Block(o_embd, min(max(o_layer, 1) * 2, 8), T) for _ in range(o_layer)])
            if not observer_causal:
                for blk in self.h:                      # bidirectional: strictly more
                    blk.attn.bias.fill_(1.0)            # generous to the third party
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_out)

        def forward(self, tok, topk_ids=None, topk_p=None, stats=None, act=None):
            B, t = tok.shape
            x = self.wpe(torch.arange(t, device=tok.device))[None].expand(B, t, -1)
            x = x + self.wte(tok)
            if self.mode == "io":
                x = x + (self.wte(topk_ids) * topk_p.unsqueeze(-1)).sum(-2) \
                      + self.pshape(topk_p) + self.sproj(stats)
            if self.mode == "act":
                x = x + self.aproj(act)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    results = {}
    for cond in cond_list:
        use_loop = cond == "cl"
        print(f"\n{'='*70}\n  CONDITION: {cond}  (loop={use_loop})\n{'='*70}")

        # ============ Phase 1: wake training (forked from a2a_loop_train) ============
        # Checkpointed: the battery downstream is long and has more moving parts than the
        # wake phase, so a battery-side bug should not cost the training again. Keyed on
        # everything the wake phase depends on; --refresh-wake forces a retrain.
        torch.manual_seed(seed)
        model = GPT(vocab_size, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_state)
        fm_ct = make_fm()
        gate = CerebellarGate(n_embd).to(device) if use_loop else None
        main_params = list(model.parameters()) + (list(gate.parameters()) if use_loop else [])
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)
        opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=fwd_lr, weight_decay=0.01)

        ck_key = (f"{cond}_L{n_layer}H{n_head}D{n_embd}_P{n_tokens}_T{T}_s{n_steps}"
                  f"_{predict_from}to{predict_to}_inj{inject_after_block}"
                  f"_fm{fwd_n_layer}x{fwd_d_head}x{fwd_mlp_mult:g}_seed{seed}")
        ck_dir = f"{DATA_DIR}/a2a_forward/confabulation/wake_ckpt"
        os.makedirs(ck_dir, exist_ok=True)
        ck_path = f"{ck_dir}/{ck_key}.pt"
        if not refresh_wake and os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=device, weights_only=True)
            model.load_state_dict(ck["model"]); fm_ct.load_state_dict(ck["fm"])
            if use_loop:
                gate.load_state_dict(ck["gate"])
            print(f"  loaded wake checkpoint -> {ck_path}")
            n_steps_run = 0
        else:
            n_steps_run = n_steps

        train_gen = torch.Generator().manual_seed(seed + 1)
        for step in range(n_steps_run):
            model.train(); fm_ct.train()
            if use_loop:
                gate.train()
            x, y = get_batch(train_data, train_gen)
            cache = {}
            if use_loop:
                def cb(act, _c=cache):
                    fp = fm_ct(act.detach())
                    _c["pred"] = fp
                    return gate(fp.detach())
                _, lm_loss, inter = model(
                    x, y, return_intermediates=True, cerebellar_fn=cb,
                    cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block)
            else:
                _, lm_loss, inter = model(x, y, return_intermediates=True)
                cache["pred"] = fm_ct(inter[predict_from].detach())
            fwd_loss = F.mse_loss(cache["pred"], inter[predict_to].detach())

            opt_main.zero_grad()
            lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm_ct.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                g = torch.Generator().manual_seed(report_seed + 5)
                with torch.no_grad():
                    vx, vy = get_batch(val_data, g)
                    vl = float(model(vx, vy)[1])
                print(f"    step {step:6d}: lm={float(lm_loss):.4f} val={vl:.4f} "
                      f"fwd_mse={float(fwd_loss):.5f}"
                      + (f" gate={gate.injection_norm():.4f}" if use_loop else ""))

        if n_steps_run:
            torch.save({"model": model.state_dict(), "fm": fm_ct.state_dict(),
                        **({"gate": gate.state_dict()} if use_loop else {})}, ck_path)
            volume.commit()
            print(f"  saved wake checkpoint -> {ck_path}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        def eval_val(injected=False):
            g = torch.Generator().manual_seed(report_seed + 5)
            tot = 0.0
            with torch.no_grad():
                for _ in range(20):
                    vx, vy = get_batch(val_data, g)
                    if injected:
                        tot += float(model(vx, vy, cerebellar_fn=lambda a: gate(fm_ct(a)),
                                           cerebellar_input_block=cib,
                                           cerebellar_inject_block=inject_after_block)[1])
                    else:
                        tot += float(model(vx, vy)[1])
            return tot / 20

        val_base = eval_val(False)
        val_inj = eval_val(True) if use_loop else float("nan")
        print(f"  val (standalone)={val_base:.4f}"
              + (f"  val (injected)={val_inj:.4f}  dependency={val_base - val_inj:+.4f}"
                 if use_loop else ""))

        # ============ Phase 2: cache the M-side of the report set (once) ============
        # a0 / a_j / a_report / top-k outputs depend only on M, not on the instrument FM,
        # so they are computed once and reused across the whole capacity sweep.
        native = report_native and use_loop
        cb_native = (lambda a: gate(fm_ct(a))) if native else None
        c_a0, c_aj, c_ar, c_ids, c_ps, c_st = [], [], [], [], [], []
        c_mass, ents, corrs = [], [], []
        with torch.no_grad():
            for i in range(0, N, 64):
                xb = rep_tok[i:i + 64].to(device)
                lg, _, vi = model(xb, return_intermediates=True, cerebellar_fn=cb_native,
                                  cerebellar_input_block=cib,
                                  cerebellar_inject_block=inject_after_block)
                lp_full = F.log_softmax(lg, dim=-1)
                p = lp_full.exp()
                tp, ti = p.topk(obs_topk, dim=-1)
                c_mass.append(float(tp.sum(-1).mean()) * xb.shape[0])
                # scalars from the FULL distribution -- these are what let the observer
                # see past the top-k truncation (entropy, confidence, margin, tail mass)
                top2 = lg.topk(2, dim=-1).values
                c_st.append(torch.stack([
                    -(p * lp_full).sum(-1),                      # entropy
                    tp[..., 0],                                  # max prob
                    top2[..., 0] - top2[..., 1],                 # top1-top2 logit margin
                    (1.0 - tp.sum(-1)).clamp_min(0),             # tail mass outside top-k
                ], dim=-1).cpu())
                c_a0.append(vi[predict_from].cpu()); c_aj.append(vi[predict_to].cpu())
                c_ar.append(vi[report_block].cpu())
                c_ids.append(ti.cpu()); c_ps.append(tp.cpu())
                lg_s = lg[:, :T - 1]
                lp = lp_full[:, :T - 1]
                ents.append(-(lp.exp() * lp).sum(-1).cpu())
                corrs.append(lg_s.argmax(-1).cpu() == rep_tok[i:i + 64][:, 1:])
        a0 = torch.cat(c_a0); aj = torch.cat(c_aj); a_rep = torch.cat(c_ar)
        topk_ids = torch.cat(c_ids); topk_p = torch.cat(c_ps); io_stats = torch.cat(c_st)
        ent = torch.cat(ents).reshape(-1); corr = torch.cat(corrs).reshape(-1)
        topk_mass = sum(c_mass) / N
        del c_a0, c_aj, c_ar, c_ids, c_ps, ents, corrs
        torch.cuda.empty_cache()
        print(f"  cached report set: N={N} seqs  top{obs_topk} mass={topk_mass:.4f}"
              + ("  [NATIVE injected pass]" if native else ""))

        # ---- ONE sequence-level split, shared by heads / observers / steering, so a
        # ---- head never trains on positions from a sequence it is tested on.
        n_str = int(0.8 * N)
        s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
        pos_of = lambda si: (si[:, None] * (T - 1) + torch.arange(T - 1)[None, :]).reshape(-1)
        tr, te = pos_of(s_tr), pos_of(s_te)

        # positions 0..T-2 only, so BEHAV (next-token) is defined everywhere
        flat = lambda z: z[:, :T - 1].reshape(N * (T - 1), -1).contiguous()

        # ---- FM-independent report targets (these never move with FM capacity) ----
        y_behav = corr.long()
        y_ent = torch.bucketize(ent.float(),
                                torch.quantile(ent[tr].float(), torch.tensor([.25, .5, .75])))
        y_world = y_world_full[:, :T - 1].reshape(-1)
        fixed_targets = {"BEHAV": (y_behav, 2, "cls"), "ENT": (y_ent, 4, "cls"),
                         "WORLD": (y_world, 5, "cls")}
        base = {k: _balance(yv[te], nc) for k, (yv, nc, _) in fixed_targets.items()}
        print("  chance baselines (test): "
              + "  ".join(f"{k}={b:.3f}" for k, b in base.items()))

        # ==== report-head inputs: the REAL blocks between a_j and the report site, run
        # ==== over full sequences so their attention is intact, on a (possibly
        # ==== substituted) a_j. This is what makes the report *computed* rather than
        # ==== linearly read off.
        mids = [model.transformer.h[i] for i in mid_blocks]
        ln_f = model.transformer.ln_f

        def make_report_input(aj_sub_seq):
            out = []
            with torch.no_grad():
                for i in range(0, aj_sub_seq.shape[0], 64):
                    z = aj_sub_seq[i:i + 64].to(device)
                    for blk in mids:
                        z = blk(z)
                    out.append(z.cpu())
            return flat(torch.cat(out))

        X_full = make_report_input(aj)
        # Wiring check, and a real one: `intermediates[post_block{i}]` is recorded BEFORE
        # the injection is added, so re-running blocks j+1..r on a_j only reproduces the
        # true report activation if no injection lands in between. This assert is what
        # catches an inject_after_block that has been moved past predict_to.
        recon = float((X_full - flat(a_rep)).abs().max())
        assert recon < 1e-3, (
            f"report-input reconstruction mismatch ({recon:.2e}): an injection lands "
            f"between {predict_to} and {report_block}. Keep inject_after_block "
            f"(={inject_after_block}) <= {j_idx}.")

        # shuffle across ALL (sequence, position) slots, then fold back to sequences:
        # distribution-preserving, position-destroying.
        sh = torch.randperm(N * T, generator=torch.Generator().manual_seed(seed + 13))
        rs = lambda z: z.reshape(N * T, n_embd)[sh].reshape(N, T, n_embd)

        # ---------------- reusable test closures ----------------
        def report_row(yv, nout, kind, X_variants, tname):
            """Tests 2 + 4 for one target: a head trained with full access and evaluated
            under each ablation, plus the fair confabulator (trained AND evaluated with
            theory-only access)."""
            h_full = _fit_head(_make_head(n_embd, nout, head_hidden, device, seed),
                               X_variants["full"], yv, tr, head_steps, head_lr, device, kind)
            row = {"full": _head_score(h_full, X_variants["full"], yv, te, device, kind)}
            for vname in ("shuffle_r", "shuffle_p", "zero_r", "zero_p"):
                row[f"eval_{vname}"] = _head_score(h_full, X_variants[vname], yv, te,
                                                   device, kind)
            h_conf = _fit_head(_make_head(n_embd, nout, head_hidden, device, seed),
                               X_variants["zero_r"], yv, tr, head_steps, head_lr, device, kind)
            row["confab"] = _head_score(h_conf, X_variants["zero_r"], yv, te, device, kind)
            row["baseline"] = base.get(tname.split("/")[0], 0.0)
            print(f"  [report/{tname:14s}] full={row['full']:.3f} "
                  f"confab={row['confab']:.3f} margin={row['full'] - row['confab']:+.3f} "
                  f"| eval: shuf_r={row['eval_shuffle_r']:.3f} "
                  f"shuf_p={row['eval_shuffle_p']:.3f} zero_r={row['eval_zero_r']:.3f}")
            return row, h_full

        def train_observer(yseq, nout, kind, o_layer, o_embd, mode, data_frac=1.0):
            torch.manual_seed(seed + 31)
            o = Observer(nout, o_layer, o_embd, mode, act_dim=n_embd).to(device)
            opt = torch.optim.AdamW(o.parameters(), lr=obs_lr, weight_decay=0.01)
            g = torch.Generator().manual_seed(seed + 32)
            pool = s_tr[:max(1, int(data_frac * len(s_tr)))]

            def fwd(si):
                xb = rep_tok[si, :T - 1].to(device)
                kw = {}
                if mode == "io":
                    kw = {"topk_ids": topk_ids[si, :T - 1].to(device),
                          "topk_p": topk_p[si, :T - 1].to(device),
                          "stats": io_stats[si, :T - 1].to(device)}
                elif mode == "act":
                    kw = {"act": a0[si, :T - 1].to(device)}
                return o(xb, **kw)

            for _ in range(obs_steps):
                si = pool[torch.randint(len(pool), (obs_bs,), generator=g)]
                out = fwd(si)
                t_ = yseq[si].to(device)
                loss = (F.cross_entropy(out.reshape(-1, nout), t_.reshape(-1))
                        if kind == "cls"
                        else (1.0 - F.cosine_similarity(out, t_, dim=-1)).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
                opt.step(); opt.zero_grad()

            o.eval()
            acc = tot = 0.0
            with torch.no_grad():
                for i in range(0, len(s_te), 32):
                    si = s_te[i:i + 32]
                    out = fwd(si).cpu()
                    t_ = yseq[si]
                    if kind == "cls":
                        acc += float((out.argmax(-1) == t_).sum()); tot += t_.numel()
                    else:
                        acc += float(F.cosine_similarity(out, t_, dim=-1).sum())
                        tot += t_.shape[0] * t_.shape[1]
            del o
            torch.cuda.empty_cache()
            return acc / tot

        def run_ladder(yv, nout, kind, self_score, tname, light=False):
            """Test 1: capacity-swept third-person observers + the data-budget control.
            `light` keeps only the strongest O_io and the O_act ceiling -- the two
            numbers that gate the headline -- for the non-default instrument capacities."""
            yseq = (yv.view(N, T - 1) if kind == "cls" else yv.view(N, T - 1, n_embd))
            out = {}
            ol, oe = caps[-1]
            for (o_layer, o_embd) in ([caps[-1]] if light else caps):
                cap = f"{o_layer}L{o_embd}D"
                if not light:
                    out[f"O_input@{cap}"] = train_observer(
                        yseq, nout, kind, o_layer, o_embd, "input")
                out[f"O_io@{cap}"] = train_observer(
                    yseq, nout, kind, o_layer, o_embd, "io")
            # ceiling: sees M's own post_block0 -> this is FM + a probe, should nail IMPL
            out[f"O_act@{ol}L{oe}D"] = train_observer(
                yseq, nout, kind, ol, oe, "act")
            if not light:
                # data-budget control: capacity-limited or just starved? If halving the
                # data barely moves it, the shortfall on IMPL is about capacity/access --
                # not about us having starved the third party.
                out[f"O_io@{ol}L{oe}D_half_data"] = train_observer(
                    yseq, nout, kind, ol, oe, "io", data_frac=0.5)
            best_io = max(v_ for k_, v_ in out.items()
                          if k_.startswith("O_io") and "half" not in k_)
            print(f"  [observ/{tname:14s}] self={self_score:.3f}  best_O_io={best_io:.3f}"
                  f"  advantage={self_score - best_io:+.3f}  | "
                  + " ".join(f"{k_}={v_:.3f}" for k_, v_ in out.items()))
            return out

        def train_fresh_fm(fm_seed, d_head, mlp_mult):
            torch.manual_seed(fm_seed)
            fm = make_fm(d_head, mlp_mult)
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
            for p in fm.parameters():
                p.requires_grad = False
            return fm

        def fm_predict(fmx, src, n_seq=None):
            # NB: the slice must be clamped to `lim`, not just the range step -- with
            # n_seq=1000 and chunk 64 an unclamped src[i:i+64] returns 1024 rows and the
            # caller's `aj[:nd] - pred` then fails to broadcast.
            lim = len(src) if n_seq is None else min(n_seq, len(src))
            out = []
            with torch.no_grad():
                for i in range(0, lim, 64):
                    out.append(fmx(src[i:min(i + 64, lim)].to(device)).cpu())
            return torch.cat(out)

        # ============ Phase 3: sweep the INSTRUMENT FM capacity ============
        # See THE JUNK-RESIDUAL TRAP in the module docstring. No IMPL number may be read
        # without its ens_cos and eta2 on the same row.
        by_cap, fixed_done = {}, False
        for (dh, mm) in inst_caps:
            cap_tag = f"h{dh}m{mm:g}"
            fms = [train_fresh_fm(seed + 911 + 37 * jj, dh, mm)
                   for jj in range(max(1, ens_n))]
            fm_params = sum(p.numel() for p in fms[0].parameters())
            pred = fm_predict(fms[0], a0)
            resid = aj - pred
            fwd_cos = float(F.cosine_similarity(pred, aj, dim=-1).mean())
            res_norm = float(resid.norm(dim=-1).mean())

            # diagnostics on a subsample (cheap; these are distributional quantities)
            nd = min(1000, N)
            ens = (_ensemble_cos([(aj[:nd] - fm_predict(f_, a0, nd)).reshape(-1, n_embd)
                                  for f_ in fms]) if len(fms) > 1 else float("nan"))
            rstruct = _residual_structure(resid[:nd].reshape(-1, n_embd),
                                          y_world_full[:nd].reshape(-1))
            eta2 = rstruct["eta2_norm"]      # the scale effect: the published yardstick
            del fms
            torch.cuda.empty_cache()

            print(f"\n  --- instrument FM {cap_tag}: {fm_params/1e3:.0f}K params "
                  f"({100*fm_params/n_pred_params:.1f}% of the {n_pred_blocks} predicted "
                  f"blocks) ---")
            _CATS = {0: "mid", 1: "sent_start", 2: "after_punct",
                     3: "after_opener", 4: "before_closer"}
            print(f"      cosine={fwd_cos:.4f}  |r|={res_norm:.3f}  ens_cos={ens:.3f}")
            print(f"      residual structure: eta2_norm={rstruct['eta2_norm']:.4f} "
                  f"eta2_dir={rstruct['eta2_dir']:.4f} eta2_vec={rstruct['eta2_vec']:.4f}"
                  f"   |r| Cohen's d vs published: "
                  + " ".join(f"{_CATS[g]}={d_:+.2f}"
                             for g, d_ in rstruct["cohens_d"].items() if g != 0))

            R, P = flat(resid), flat(pred)
            cents = _kmeans_fit(R[tr].to(device), impl_k, seed=seed)   # TRAIN rows only
            y_impl = _kmeans_assign(R.to(device), cents).cpu()
            cq_w, cq_b, cq_sep = _cluster_quality(R.to(device), y_impl.to(device), cents)
            y_impl_cos = F.normalize(R, dim=-1)
            base["IMPL"] = _balance(y_impl[te], impl_k)
            base["IMPL_COS"] = 0.0
            print(f"      residual cluster quality: within={cq_w:.3f} between={cq_b:.3f} "
                  f"separation={cq_sep:.3f}  chance(IMPL)={base['IMPL']:.3f}"
                  + ("   <-- weak; trust IMPL_COS over IMPL" if cq_sep < 0.1 else ""))

            X_variants = {
                "full":      X_full,
                "shuffle_r": make_report_input(pred + rs(resid)),
                "shuffle_p": make_report_input(rs(pred) + resid),
                "zero_r":    make_report_input(pred),
                "zero_p":    make_report_input(resid),
            }

            light = not (full_ladder_every_cap or not fixed_done)
            impl_rep, h_impl = report_row(y_impl, impl_k, "cls", X_variants,
                                          f"IMPL/{cap_tag}")
            impl_obs = run_ladder(y_impl, impl_k, "cls", impl_rep["full"],
                                  f"IMPL/{cap_tag}", light)
            ic_rep, _ = report_row(y_impl_cos, n_embd, "cos", X_variants,
                                   f"IMPL_COS/{cap_tag}")
            ic_obs = run_ladder(y_impl_cos, n_embd, "cos", ic_rep["full"],
                                f"IMPL_COS/{cap_tag}", light)

            # The FM-independent targets are run once, at the default capacity, since
            # neither their labels nor X_full depend on the instrument. (Their ablation
            # columns do, so they are reported alongside that capacity.)
            fixed_rep, fixed_obs = {}, {}
            h_behav = None
            if not fixed_done:              # inst_caps[0] == the default instrument
                for tn, (yv_, nc_, kd_) in fixed_targets.items():
                    fixed_rep[tn], h_ = report_row(yv_, nc_, kd_, X_variants, tn)
                    fixed_obs[tn] = run_ladder(yv_, nc_, kd_, fixed_rep[tn]["full"], tn)
                    if tn == "BEHAV":
                        h_behav = h_
                        h_behav_keep = h_
                fixed_done = True
            else:
                h_behav = h_behav_keep

            # ---- Test 3: matched-KL steering, residual span vs self-theory span ----
            # Rescale every direction to the SAME behavioural effect (target KL on the
            # logits). Matched KL means matched behaviour, so a difference in how far the
            # IMPL report moves cannot be a behavioural artifact -- and BEHAV report-flip
            # becomes a built-in control that should come out equal across families.
            # Prediction: residual-span steering moves the IMPL report more than
            # prediction-span steering. A theory-driven report cannot show that asymmetry.
            idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 17))
            sel_ = tr[idx[:min(50000, len(tr))]]

            def top_pcs(M_):
                sub = M_[sel_].to(device)
                return torch.linalg.svd(sub - sub.mean(0), full_matrices=False)[2][:steer_n_pc]

            # Memory note: a full-sequence log-prob tensor here is
            # n_seq x (T-1) x 50257 floats -- at 256 sequences that is 6.6 GB, and the
            # baseline has to stay resident alongside the perturbed one, which OOMs a
            # 22 GB card. So steering runs in sequence-chunks and accumulates scalars;
            # the baseline log-probs are cached on the CPU in fp16 (the KL here is ~1e-2,
            # far above fp16 resolution) and only one chunk is ever on the GPU.
            steer_seqs = s_te[:min(steer_n_seq, len(s_te))]
            base_aj = aj[steer_seqs]                                    # kept on CPU

            def _fwd(aj_chunk):
                z = aj_chunk.to(device)
                for blk in mids:
                    z = blk(z)
                z = z[:, :T - 1]
                return (F.log_softmax(model.lm_head(ln_f(z)), -1),
                        h_impl(z).argmax(-1), h_behav(z).argmax(-1))

            base_lp_c, base_impl_c, base_behav_c = [], [], []
            with torch.no_grad():
                for i in range(0, len(base_aj), steer_chunk):
                    lp_, ri_, rb_ = _fwd(base_aj[i:i + steer_chunk])
                    base_lp_c.append(lp_.half().cpu())
                    base_impl_c.append(ri_); base_behav_c.append(rb_)

            def probe_dir(u, eps):
                kl_s = im_s = bh_s = 0.0
                n_tok = n_pos = 0
                with torch.no_grad():
                    for ci, i in enumerate(range(0, len(base_aj), steer_chunk)):
                        chunk = base_aj[i:i + steer_chunk].to(device) + eps * u.view(1, 1, -1)
                        lp_, ri_, rb_ = _fwd(chunk)
                        b_lp = base_lp_c[ci].to(device).float()
                        kl_s += float(F.kl_div(lp_.reshape(-1, vocab_size),
                                               b_lp.reshape(-1, vocab_size),
                                               log_target=True, reduction="sum"))
                        n_tok += b_lp.shape[0] * b_lp.shape[1]
                        im_s += float((ri_ != base_impl_c[ci]).float().sum())
                        bh_s += float((rb_ != base_behav_c[ci]).float().sum())
                        n_pos += ri_.numel()
                        del lp_, b_lp, chunk
                return kl_s / n_tok, im_s / n_pos, bh_s / n_pos

            steer = {"target_kl": steer_target_kl, "families": {}}
            for fam, M_ in (("residual", R), ("prediction", P)):
                pcs = top_pcs(M_)
                per = []
                for k in range(min(steer_n_pc, pcs.shape[0])):
                    u = pcs[k]
                    kl0, _, _ = probe_dir(u, steer_eps)
                    if kl0 <= 1e-9:
                        continue
                    # KL is locally quadratic in eps, so this lands close in one shot;
                    # one refinement pass tightens it. Cap eps so we stay near-local.
                    eps_k = min(steer_eps * (steer_target_kl / kl0) ** 0.5, 20 * steer_eps)
                    kl1, _, _ = probe_dir(u, eps_k)
                    if kl1 > 1e-9:
                        eps_k = min(eps_k * (steer_target_kl / kl1) ** 0.5, 20 * steer_eps)
                    kl_f, fi, fb = probe_dir(u, eps_k)
                    per.append({"pc": k, "eps": eps_k, "kl": kl_f,
                                "impl_flip": fi, "behav_flip": fb})
                agg = lambda kk: float(np.mean([p_[kk] for p_ in per])) if per else float("nan")
                steer["families"][fam] = {"per_pc": per, "kl": agg("kl"), "eps": agg("eps"),
                                          "impl_flip": agg("impl_flip"),
                                          "behav_flip": agg("behav_flip")}
                print(f"  [steer/{fam:10s}] @matched KL={agg('kl'):.4f}  "
                      f"IMPL report-flip={agg('impl_flip'):.3f}  "
                      f"BEHAV report-flip={agg('behav_flip'):.3f}  (eps={agg('eps'):.3f})")
            rs_, ps_ = steer["families"]["residual"], steer["families"]["prediction"]
            steer["impl_flip_ratio"] = (rs_["impl_flip"] / ps_["impl_flip"]
                                        if ps_["impl_flip"] > 0 else float("inf"))
            print(f"  [steer] residual/prediction IMPL-flip ratio at matched behaviour = "
                  f"{steer['impl_flip_ratio']:.2f}x")
            del base_lp_c, base_impl_c, base_behav_c, base_aj
            torch.cuda.empty_cache()

            by_cap[cap_tag] = {
                "d_head": dh, "mlp_mult": mm, "fm_params": fm_params,
                "pct_of_predicted": 100 * fm_params / n_pred_params,
                "fwd_cosine": fwd_cos, "res_norm": res_norm,
                "ens_cos": ens, "syntactic_eta2": eta2, "residual_structure": rstruct,
                "cluster_quality": {"within": cq_w, "between": cq_b, "separation": cq_sep},
                "report": {"IMPL": impl_rep, "IMPL_COS": ic_rep, **fixed_rep},
                "observers": {"IMPL": impl_obs, "IMPL_COS": ic_obs, **fixed_obs},
                "baselines": dict(base), "steering": steer, "light_ladder": light,
            }
            del pred, resid, R, P, X_variants, y_impl, y_impl_cos
            torch.cuda.empty_cache()

        results[cond] = {
            "val_standalone": val_base, "val_injected": val_inj,
            "topk_mass": topk_mass, "report_native": native, "by_capacity": by_cap,
        }
        del model, fm_ct, a0, aj, a_rep, topk_ids, topk_p, io_stats, X_full
        torch.cuda.empty_cache()

    cfg = {"n_tokens": n_tokens, "block_size": T, "n_layer": n_layer, "n_head": n_head,
           "n_embd": n_embd, "predict_from": predict_from, "predict_to": predict_to,
           "report_block": report_block, "inject_after_block": inject_after_block,
           "n_steps": n_steps, "impl_k": impl_k, "observer_caps": observer_caps,
           "observer_causal": observer_causal, "obs_topk": obs_topk, "seed": seed,
           "inst_caps": inst_caps_str, "ens_n": ens_n, "n_pred_params": n_pred_params}
    out_dir = f"{DATA_DIR}/a2a_forward/confabulation"
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{out_dir}/{(tag + '_') if tag else ''}results.json"
    with open(fn, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---------------- summary ----------------
    print(f"\n{'='*78}\nSUMMARY")
    print("  introspective advantage = self-report - best capacity-matched O_io")
    print("  READ IMPL/IMPL_COS ONLY AGAINST ens_cos AND eta2. A saturated FM (cosine")
    print("  -> 1) leaves a junk residual that M can report and no observer can predict,")
    print("  which fakes the headline. Low ens_cos + ~0 eta2 = that row is not evidence.")
    for cond, rr in results.items():
        print(f"\n  {cond}  (val={rr['val_standalone']:.4f}"
              + (f"  injected={rr['val_injected']:.4f}" if rr["val_injected"] == rr["val_injected"]
                 else "") + ")")
        print(f"    {'instrument':11s} {'%pred':>6s} {'cos':>6s} {'ens':>6s} {'eta2':>6s} "
              f"{'sep':>6s} | {'IMPL':>6s} {'advant':>7s} {'margin':>7s} | "
              f"{'I_COS':>6s} {'advant':>7s} {'margin':>7s} | {'steer':>6s}")
        for cap_tag, c in rr["by_capacity"].items():
            def adv(tn):
                bio = max(v_ for k_, v_ in c["observers"][tn].items()
                          if k_.startswith("O_io") and "half" not in k_)
                return c["report"][tn]["full"] - bio
            print(f"    {cap_tag:11s} {c['pct_of_predicted']:6.1f} {c['fwd_cosine']:6.3f} "
                  f"{c['ens_cos']:6.3f} {c['syntactic_eta2']:6.4f} "
                  f"{c['cluster_quality']['separation']:6.3f} | "
                  f"{c['report']['IMPL']['full']:6.3f} {adv('IMPL'):+7.3f} "
                  f"{c['report']['IMPL']['full'] - c['report']['IMPL']['confab']:+7.3f} | "
                  f"{c['report']['IMPL_COS']['full']:6.3f} {adv('IMPL_COS'):+7.3f} "
                  f"{c['report']['IMPL_COS']['full'] - c['report']['IMPL_COS']['confab']:+7.3f} | "
                  f"{c['steering']['impl_flip_ratio']:5.2f}x")
        d0 = next(iter(rr["by_capacity"].values()))
        print(f"    control targets @ default instrument (advantage should be ~0):")
        for tn in ("BEHAV", "ENT", "WORLD"):
            if tn not in d0["report"]:
                continue
            bio = max(v_ for k_, v_ in d0["observers"][tn].items()
                      if k_.startswith("O_io") and "half" not in k_)
            r_ = d0["report"][tn]
            print(f"      {tn:6s} self={r_['full']:.3f} confab={r_['confab']:.3f} "
                  f"bestO_io={bio:.3f} advantage={r_['full'] - bio:+.3f} "
                  f"shufR={r_['eval_shuffle_r']:.3f} base={r_['baseline']:.3f}")
    print(f"\n  saved -> {fn}\n{'='*78}")
    return results


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def residual_diagnostics(
    n_tokens: int = 10_000_000, block_size: int = 128,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block2",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2, fwd_d_head: int = 64, fwd_mlp_mult: float = 2.0,
    fwd_n_head: int = 1, fwd_lr: float = 1e-3,
    conditions: str = "cl,ol", n_steps: int = 10_000,
    inst_caps_str: str = "16:0.5,4:0.25,32:1.0,64:2.0", ens_n: int = 3,
    n_report_sequences: int = 3000, report_seed: int = 999,
    fresh_fm_steps: int = 3000, seed: int = 42, tag: str = "",
):
    """Cheap standalone re-read of the two junk-residual discriminators from saved wake
    checkpoints — no report heads, no observer ladder, no steering (~10 min vs ~4 h).

    Exists because the discriminators gate the interpretation of every IMPL number but
    cost almost nothing to compute, so a correction to them should not require re-running
    the whole battery. Requires `confabulation_test` to have populated `wake_ckpt/` for
    the same config.
    """
    import glob
    import numpy as np
    import tiktoken
    import torch
    import torch.nn.functional as F
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = block_size
    inst_caps = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in inst_caps_str.split(",")]
    _CATS = {0: "mid", 1: "sent_start", 2: "after_punct",
             3: "after_opener", 4: "before_closer"}
    PUBLISHED = {1: -0.85, 2: -0.62, 3: +0.62, 4: +0.84}

    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]
    all_tokens, total = [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "shard_*.npy"))):
        t_ = np.load(path); all_tokens.append(t_); total += len(t_)
        if total >= n_tokens:
            break
    data = torch.from_numpy(np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    enc = tiktoken.get_encoding("gpt2")

    rng = np.random.default_rng(report_seed)
    starts = rng.integers(0, len(val_data) - T - 1, size=n_report_sequences)
    rep_tok = torch.stack([val_data[i:i + T] for i in starts])
    y_world = _syntactic_labels(rep_tok, vocab_size, enc)
    N = rep_tok.shape[0]

    out = {}
    for cond in [c.strip() for c in conditions.split(",")]:
        ck_key = (f"{cond}_L{n_layer}H{n_head}D{n_embd}_P{n_tokens}_T{T}_s{n_steps}"
                  f"_{predict_from}to{predict_to}_inj{inject_after_block}"
                  f"_fm{fwd_n_layer}x{fwd_d_head}x{fwd_mlp_mult:g}_seed{seed}")
        ck_path = f"{DATA_DIR}/a2a_forward/confabulation/wake_ckpt/{ck_key}.pt"
        if not os.path.exists(ck_path):
            print(f"  [{cond}] no wake checkpoint at {ck_path} -- skipping")
            continue
        model = GPT(vocab_size, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ck_path, map_location=device,
                                         weights_only=True)["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        c_a0, c_aj = [], []
        with torch.no_grad():
            for i in range(0, N, 64):
                _, _, vi = model(rep_tok[i:i + 64].to(device), return_intermediates=True)
                c_a0.append(vi[predict_from].cpu()); c_aj.append(vi[predict_to].cpu())
        a0, aj = torch.cat(c_a0), torch.cat(c_aj)

        def get_batch(gen):
            ix = torch.randint(len(train_data) - T - 1, (64,), generator=gen)
            return torch.stack([train_data[i:i + T] for i in ix]).to(device)

        def fresh_fm(fm_seed, dh, mm):
            torch.manual_seed(fm_seed)
            fm = TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                         n_layer=fwd_n_layer, mlp_mult=mm,
                                         block_size=T).to(device)
            opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
            g = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(fresh_fm_steps):
                with torch.no_grad():
                    _, _, vi = model(get_batch(g), return_intermediates=True)
                F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            fm.eval()
            return fm

        def predict(fm, n_seq=None):
            lim = len(a0) if n_seq is None else min(n_seq, len(a0))
            with torch.no_grad():
                return torch.cat([fm(a0[i:min(i + 64, lim)].to(device)).cpu()
                                  for i in range(0, lim, 64)])

        print(f"\n{'='*78}\n  {cond}\n{'='*78}")
        out[cond] = {}
        for (dh, mm) in inst_caps:
            fms = [fresh_fm(seed + 911 + 37 * j, dh, mm) for j in range(max(1, ens_n))]
            n_p = sum(p.numel() for p in fms[0].parameters())
            resid = aj - predict(fms[0])
            fwd_cos = float(F.cosine_similarity(aj - resid, aj, dim=-1).mean())
            nd = min(1000, N)
            ens = (_ensemble_cos([(aj[:nd] - predict(f_, nd)).reshape(-1, n_embd)
                                  for f_ in fms]) if len(fms) > 1 else float("nan"))
            rs_ = _residual_structure(resid[:nd].reshape(-1, n_embd),
                                      y_world[:nd].reshape(-1))
            tag_ = f"h{dh}m{mm:g}"
            out[cond][tag_] = {"fm_params": n_p, "fwd_cosine": fwd_cos, "ens_cos": ens,
                               **rs_}
            print(f"  {tag_:10s} {n_p/1e3:5.0f}K  cos={fwd_cos:.4f}  ens_cos={ens:.3f}  "
                  f"eta2_norm={rs_['eta2_norm']:.4f} eta2_dir={rs_['eta2_dir']:.4f} "
                  f"eta2_vec={rs_['eta2_vec']:.4f}")
            print(f"             |r| Cohen's d: "
                  + "  ".join(f"{_CATS[g]}={d_:+.2f}(pub {PUBLISHED[g]:+.2f})"
                              for g, d_ in rs_["cohens_d"].items() if g in PUBLISHED))
            del fms, resid
            torch.cuda.empty_cache()
        del model, a0, aj
        torch.cuda.empty_cache()

    fn = (f"{DATA_DIR}/a2a_forward/confabulation/"
          f"{(tag + '_') if tag else ''}residual_diagnostics.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n  saved -> {fn}")
    return out


@app.local_entrypoint()
def main(conditions: str = "cl,ol", n_steps: int = 10_000,
         tag: str = "", smoke: bool = False):
    confabulation_test.remote(conditions=conditions, n_steps=n_steps,
                              tag=tag, smoke=smoke)
