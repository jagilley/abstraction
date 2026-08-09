"""Does the local-loss simplicity collapse survive an EXOGENOUS conditioning gap?

SPEC: rhm/conditional_revision/local_loss/SPEC.md  (idea: ideas/revision_not_surprisal.md §8)

Every local loss in this repo -- lambda*||sg(FM(h_src)) - h_tgt||^2 added to NTP -- makes the
model LEGIBLE without making it SELF-KNOWING (MNIST_LOCAL_LOSS: FM cos 0.997 at SK R^2 -0.03;
GATED_RATCHET: the gate opens; RHM_FM_REGULARIZER: "self-knowledge is not required for
functional simplification"; RHM_LATENT_LOOP F4: fresh meta destroyed monotonically in lambda).
Idea-doc §8 claims this is not four nulls but one PROOF: h_src and h_tgt are both deterministic
functions of the same input through the same weights, so the only free move is "become simpler."

Swap the target one POSITION instead of six BLOCKS and the proof stops applying:

    depth    (incumbent):  FM( post_embed[t]   ) -> post_block6[t]      gap = 0 tokens
    temporal (this cut) :  FM( post_block6[<=t]) -> post_block6[t+1]    gap = exactly 1 token

`post_block6[t+1]` depends on x_{t+1}, which a causal FM over post_block6[<=t] cannot hold.
Parent Gate 0 verified the resulting residual has a real aleatoric component on RHM
(corr(res,nll) -0.337 -> +0.652, with a protocol-matched depth_frozen control at -0.328).

  >>> With the proof gone, does the collapse happen anyway?  <<<

Two-sided and the negative branch is the valuable one: if temporal REPRODUCES the depth
signature, the conditioning gap is not the operative variable and §8's reading is wrong.

§8 also names what should REPLACE the simplicity collapse if the gap is operative:
INPUT-INVARIANCE (make h6[t+1] insensitive to x_{t+1}; data2vec's known mode). On RHM the
discardable part of x_{t+1} is exactly SYNONYM IDENTITY -- which token realizes a latent is
irrelevant downstream except through the latent. So a third outcome is live: synonym-invariance
rises while structure-sensitivity holds = the aleatoric null appearing as a representational
fact. `_twin_leaves` / `_swap_sensitivity` below is the instrument for that, and the
per-level RULE probe is its decodability twin.

DESIGN -- what is controlled and how
  * Substrate m2 (v16 s2 L6 m2 distinct, rule_seed=0 seq_seed=1), 8L/8H/256D, wd=0.1: the
    substrate the depth signature was MEASURED on (RHM_FM_REGULARIZER). Plain NTP groks the
    whole hierarchy here (d6=0.94) so belief depth is not binding.
  * The depth arm is NOT retrained. `train_temporal_reg` is `rhm_fm_regularizer.train_fm_reg`
    with the target swapped and NOTHING else changed -- same seeds, same optimizers, same
    lambda warmup(2000)+ramp(3000), same wd, same batch fn, same FM architecture. Because the
    FM config is identical the global torch RNG stream is identical, so at every step the
    temporal arm sees the SAME data batches as the incumbent depth arm.
  * MATCH_STEP = 80000 is the step at which checkpoints exist for depth lambda in
    {0.1, 1.0, 3.0} AND for the lambda=0 anchor (wd_sweep wd=0.1). Every arm is read there.
  * Every readout is computed by the SAME code for both arms, with BOTH forward models
    (fresh, seed 911 != training seed) trained on every checkpoint -- so "does the temporal
    loss buy depth legibility, and vice versa" is a 2x2, not two separate scales.

READOUTS (per checkpoint)
  knowledge gate (per-level feature probe)   -- the denominator; rank/eta^2 only mean
                                                anything at matched d6
  per-level RULE probe                       -- synonym-identity decodability (NEW)
  FM-free act rank / top1 / mean dim var     -- leakage-proof complexity + collapse check
  fresh-FM cos, res/tgt norm ratio, res rank -- legibility
  SK / meta-object probes, both alignments   -- self-knowledge (pre = from the forecasting
                                                state at t, post = from the state at t+1)
  ensemble agreement (n=3, both arms)        -- is the residual input-determined or FM-noise
  swap sensitivity (synonym vs structure)    -- the input-invariance / aleatoric-null split

Run:
  cd experiments
  # smoke (attached, ~8 min)
  modal run -m rhm.conditional_revision.local_loss.temporal_local_loss::smoke
  # train the temporal arms (detached, ~1.5 h each on an L4, 3 in parallel)
  modal run --detach -m rhm.conditional_revision.local_loss.temporal_local_loss::tll_sweep
  # analyze both arms at the matched step
  modal run --detach -m rhm.conditional_revision.local_loss.temporal_local_loss::analyze_ll
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_fm_legibility import (
    _tb_key, _compute_effective_rank, _ensure_corpus_distinct, _generate_with_traces,
)
from rhm.rhm_norm_trajectory import _weight_norms
from rhm.rhm_wd_sweep import _probe_knowledge, _wd_tag, _sweep_dir

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-temporal-local-loss", image=image)

MODEL_TAG = "8L8H256D"
WD_FIXED = 0.1
# identical to rhm_fm_regularizer.REG_FM (8H/16d/mlp2 ~791K, "capB") so the RNG stream and
# the FM capacity match the incumbent arm exactly.
REG_FM = {"nh": 8, "dh": 16, "mm": 2.0, "nl": 2}
TGT_BLOCK = "post_block6"      # the target block, SHARED by both arms
DEPTH_SRC = "post_embed"       # incumbent source
MATCH_STEP = 80000             # the step every arm is read at (see module docstring)
LAM_DEFAULT = "0.1,1.0,3.0"


def _lam_tag(lam):
    return str(lam).replace(".", "p").replace("-", "m")


def _tll_dir(key, lam):
    """Temporal-local-loss training dir. Deliberately a different prefix from
    rhm_fm_regularizer's `fmreg_` so the incumbent depth runs can never be clobbered."""
    return f"{DATA_DIR}/{key}/tll_{MODEL_TAG}_wd{_lam_tag(WD_FIXED)}_lam{_lam_tag(lam)}"


def _fmreg_dir(key, lam):
    """The INCUMBENT depth arm (rhm_fm_regularizer._reg_dir) -- read-only here."""
    return f"{DATA_DIR}/{key}/fmreg_{MODEL_TAG}_wd{_lam_tag(WD_FIXED)}_lam{_lam_tag(lam)}"


# ==========================================================================
# Arm definition: the ONE thing that differs between the two conditions
# ==========================================================================

def _arm_src(arm):
    return TGT_BLOCK if arm == "temporal" else DEPTH_SRC


def _pred_tgt(fm, inter, arm):
    """(pred, tgt) both indexed by TARGET POSITION p = 1..T-1, so the two arms are
    shape- and position-matched and index -1 is always "target = last leaf".

      temporal:  pred[:, j] = FM(post_block6[<=j])[j]      tgt[:, j] = post_block6[j+1]
      depth   :  pred[:, j] = FM(post_embed)[j+1]          tgt[:, j] = post_block6[j+1]
    """
    tgt = inter[TGT_BLOCK]
    out = fm(inter[_arm_src(arm)])
    if arm == "temporal":
        return out[:, :-1, :], tgt[:, 1:, :]
    return out[:, 1:, :], tgt[:, 1:, :]


def _reg_loss(fm, inter, arm):
    """The TRAINING term. For depth this is bit-identical to rhm_fm_regularizer
    (all positions); for temporal it is over source positions 0..T-2."""
    import torch.nn.functional as F
    tgt = inter[TGT_BLOCK]
    if arm == "temporal":
        return F.mse_loss(fm(inter[TGT_BLOCK])[:, :-1, :], tgt[:, 1:, :])
    return F.mse_loss(fm(inter[DEPTH_SRC]), tgt)


# ==========================================================================
# Phase 1 -- training (fork of rhm_fm_regularizer.train_fm_reg; target swapped only)
# ==========================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def train_temporal_reg(
    lam: float, arm: str = "temporal", wd: float = WD_FIXED, dir_suffix: str = "",
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    n_steps: int = MATCH_STEP, batch_size: int = 64, lr: float = 3e-4, fm_lr: float = 1e-3,
    seed: int = 42, fm_seed: int = 42, reg_warmup: int = 2000, reg_ramp: int = 3000,
    eval_interval: int = 250, commit_interval: int = 2500,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    L = depth
    block_size = s ** L
    device = "cuda"
    key = _ensure_corpus_distinct(v, s, L, m, n_tokens)
    volume.reload()

    data = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    save_dir = (_tll_dir(key, lam) if arm == "temporal"
                else _fmreg_dir(key, lam) + "_replica") + dir_suffix
    os.makedirs(save_dir, exist_ok=True)
    fracs = [0.0, 0.0033, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    ckpt_steps = sorted(set(round(f * n_steps / eval_interval) * eval_interval for f in fracs))

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    torch.manual_seed(fm_seed)
    fm = TransformerForwardModel(d_model=n_embd, d_head=REG_FM["dh"], n_head=REG_FM["nh"],
                                 n_layer=REG_FM["nl"], mlp_mult=REG_FM["mm"],
                                 block_size=block_size).to(device)
    opt_fm = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)

    def lam_at(step):
        if step < reg_warmup:
            return 0.0
        if step < reg_warmup + reg_ramp:
            return lam * (step - reg_warmup) / reg_ramp
        return lam

    latest_path = os.path.join(save_dir, "latest.pt")
    start_step = 0
    history = {"val_loss": [], "weight_norm": [], "reg_loss": [], "fm_cosine": [],
               "tgt_var": []}
    saved = []
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        fm.load_state_dict(ck["fm"]); opt_fm.load_state_dict(ck["opt_fm"])
        start_step = ck["step"] + 1
        history = ck["history"]; saved = ck["saved"]
        print(f"=== RESUME {arm} lam={lam} ({key}) from step {start_step} ===")
    else:
        print(f"=== START {arm} lam={lam} wd={wd} ({key}) n_steps={n_steps} "
              f"src={_arm_src(arm)} -> {TGT_BLOCK}{'[t+1]' if arm == 'temporal' else '[t]'} ===")

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def commit(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "fm": fm.state_dict(), "opt_fm": opt_fm.state_dict(), "step": step,
                    "history": history, "saved": saved}, latest_path)
        with open(os.path.join(save_dir, "training_info.json"), "w") as f:
            json.dump({"key": key, "arm": arm, "lam": lam, "wd": wd, "model_tag": MODEL_TAG,
                       "n_steps": n_steps, "reg_src": _arm_src(arm), "reg_tgt": TGT_BLOCK,
                       "last_step": step, "checkpoints": saved, "history": history},
                      f, indent=2, cls=NumpyEncoder)
        volume.commit()

    for step in range(start_step, n_steps + 1):
        lam_t = lam_at(step)
        last_reg = 0.0
        if step > 0:
            model.train(); fm.train()
            x, y = get_batch(train_data)
            logits, ntp_loss, inter = model(x, y, return_intermediates=True)

            # --- Main update: NTP + lambda * FM-predictability (grad through MAIN only) ---
            for p in fm.parameters():
                p.requires_grad_(False)
            reg = _reg_loss(fm, inter, arm)
            loss_main = ntp_loss + lam_t * reg
            opt.zero_grad(set_to_none=True)
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for p in fm.parameters():
                p.requires_grad_(True)
            last_reg = float(reg.detach())

            # --- FM update: track the (updated) model; detached so no grad to main ---
            det = {k_: t_.detach() for k_, t_ in inter.items()}
            fm_loss = _reg_loss(fm, det, arm)
            opt_fm.zero_grad(set_to_none=True)
            fm_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fm.step()

        if step % eval_interval == 0:
            model.eval(); fm.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(*get_batch(val_data))[1]) for _ in range(10)]))
                xb, yb = get_batch(val_data)
                _, _, inter = model(xb, yb, return_intermediates=True)
                pred, tgt = _pred_tgt(fm, inter, arm)
                fmcos = float(F.cosine_similarity(pred, tgt, dim=-1).mean())
                tvar = float(tgt.var())
            wnorm = float(sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()) ** 0.5)
            history["val_loss"].append((step, vl))
            history["weight_norm"].append((step, wnorm))
            history["reg_loss"].append((step, last_reg))
            history["fm_cosine"].append((step, fmcos))
            history["tgt_var"].append((step, tvar))
            if step in ckpt_steps and not any(sv["step"] == step for sv in saved):
                p = os.path.join(save_dir, f"ckpt_step{step}.pt")
                torch.save(model.state_dict(), p)
                saved.append({"step": step, "val_loss": vl, "weight_norm": wnorm,
                              "fm_cosine": fmcos, "reg_mse": last_reg, "tgt_var": tvar,
                              "lam_t": lam_t, "path": p})
                print(f"  step {step:6d}: val={vl:.4f} ||W||={wnorm:.1f} regMSE={last_reg:.4f} "
                      f"tgtVar={tvar:.4f} fmcos={fmcos:.4f} lam_t={lam_t:.3f} [CKPT]")
        if step > 0 and step % commit_interval == 0:
            commit(step)

    commit(n_steps)
    print(f"=== DONE {arm} lam={lam}: {len(saved)} checkpoints, "
          f"final val={history['val_loss'][-1][1]:.4f} ===")
    return {"key": key, "arm": arm, "lam": lam, "checkpoints": saved}


# ==========================================================================
# The synonym / structure counterfactual instrument
# ==========================================================================

def _realize(rules, root, rc, s):
    """Rebuild leaves top-down from the root feature and a full set of rule choices."""
    import numpy as np
    cur = root
    for ell in range(len(rules)):
        n_nodes = cur.shape[1]
        nxt = np.empty((cur.shape[0], n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], rc[ell][:, j]]
        cur = nxt
    return cur


def _twin_leaves(rules, n_seq, seed, s, L, v, m):
    """Matched counterfactual twins over the LAST level-(L-2) constituent (leaf
    positions T-s^2 .. T-1), read at the last position.

      base      sampled RHM sequence
      synonym   the span's two level-(L-1) nodes get DIFFERENT rule choices.
                Every latent feature at every level is UNCHANGED -- only which
                production realizes the constituent changed.  (B == 0 by construction.)
      structure synonym's *identical* leaf-rule perturbation PLUS a different rule
                choice at the span's level-(L-2) node, which changes the two
                level-(L-1) FEATURES.  So structure = synonym + exactly one extra
                intervention, and the pair differs only in "did the latent move".
      random    the span replaced by uniform tokens (off-DGP scale reference)
      other     an independent sequence (global spread normalizer)

    Both twins stay perfectly in-distribution, both perturb the same s^2 leaf
    positions, and everything outside the span (i.e. the whole causal prefix up to
    T-s^2) is bit-identical to base.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    root = rng.integers(0, v, size=(n_seq, 1))
    rc = []
    n_nodes = 1
    for ell in range(L):
        rc.append(rng.integers(0, m, size=(n_seq, n_nodes)))
        n_nodes *= s
    n_leafnodes = s ** (L - 1)
    span_leafnodes = list(range(n_leafnodes - s, n_leafnodes))   # the last s leaf-level nodes
    span_parent = s ** (L - 2) - 1                                # the last level-(L-2) node

    def bump(a):
        return (a + rng.integers(1, m, size=a.shape)) % m

    rc_syn = [x.copy() for x in rc]
    for nd in span_leafnodes:
        rc_syn[L - 1][:, nd] = bump(rc[L - 1][:, nd])
    rc_str = [x.copy() for x in rc_syn]
    rc_str[L - 2][:, span_parent] = bump(rc[L - 2][:, span_parent])

    base = _realize(rules, root, rc, s)
    syn = _realize(rules, root, rc_syn, s)
    stru = _realize(rules, root, rc_str, s)
    span = s * s
    rand = base.copy()
    rand[:, -span:] = rng.integers(0, v, size=(n_seq, span))
    other_root = rng.integers(0, v, size=(n_seq, 1))
    other_rc = []
    n_nodes = 1
    for ell in range(L):
        other_rc.append(rng.integers(0, m, size=(n_seq, n_nodes)))
        n_nodes *= s
    other = _realize(rules, other_root, other_rc, s)

    # sanity: outside the span nothing moved
    assert (base[:, :-span] == syn[:, :-span]).all()
    assert (base[:, :-span] == stru[:, :-span]).all()
    return {"base": base, "synonym": syn, "structure": stru, "random": rand, "other": other,
            "span": span}


def _swap_sensitivity(model, rules, n_seq, seed, s, L, v, m, device, batch_size=256,
                      block=TGT_BLOCK):
    """How far does h6[last] move under a synonym swap vs a structure swap?

    invariance collapse (§8's predicted degeneracy) => BOTH d_syn and d_str fall
    aleatoric null as a representational fact    => d_syn falls, d_str holds
    no collapse                                  => both hold
    """
    import numpy as np
    import torch

    tw = _twin_leaves(rules, n_seq, seed, s, L, v, m)
    span = tw["span"]

    def last_acts(leaves):
        x = torch.from_numpy(leaves.astype(np.int64)).to(device)
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                _, _, inter = model(x[i:i + batch_size], return_intermediates=True)
                out.append(inter[block][:, -1, :].float().cpu())
        return torch.cat(out)

    H = {k: last_acts(tw[k]) for k in ["base", "synonym", "structure", "random", "other"]}
    hb = H["base"]
    scale = float(hb.norm(dim=-1).mean())

    def dist(k):
        return (H[k] - hb).norm(dim=-1)

    ham = {k: torch.from_numpy((tw[k][:, -span:] != tw["base"][:, -span:]).sum(1))
           for k in ["synonym", "structure", "random"]}
    out = {"n_seq": int(n_seq), "span": int(span), "mean_h_norm": scale}
    for k in ["synonym", "structure", "random", "other"]:
        d = dist(k)
        out[k] = {"d": float(d.mean()), "d_sd": float(d.std())}
        if k in ham:
            out[k]["hamming"] = float(ham[k].float().mean())
    d_oth = out["other"]["d"]
    out["d_syn_rel"] = out["synonym"]["d"] / (d_oth + 1e-12)
    out["d_str_rel"] = out["structure"]["d"] / (d_oth + 1e-12)
    out["d_rand_rel"] = out["random"]["d"] / (d_oth + 1e-12)
    out["syn_over_str"] = out["synonym"]["d"] / (out["structure"]["d"] + 1e-12)
    out["struct_excess_rel"] = (out["structure"]["d"] - out["synonym"]["d"]) / (d_oth + 1e-12)

    # Hamming-matched cells: the two twins perturb the same positions but not always the
    # same NUMBER of tokens, so report the contrast at fixed surface-change size too.
    by_h = {}
    for h in range(1, span + 1):
        ms, mt = ham["synonym"] == h, ham["structure"] == h
        if int(ms.sum()) >= 30 and int(mt.sum()) >= 30:
            by_h[f"h{h}"] = {"n_syn": int(ms.sum()), "n_str": int(mt.sum()),
                             "d_syn": float(dist("synonym")[ms].mean()),
                             "d_str": float(dist("structure")[mt].mean())}
    out["by_hamming"] = by_h
    print(f"    swap: |h|={scale:.2f}  d_syn={out['synonym']['d']:.3f} "
          f"d_str={out['structure']['d']:.3f} d_rand={out['random']['d']:.3f} "
          f"d_other={d_oth:.3f} | rel syn={out['d_syn_rel']:.3f} str={out['d_str_rel']:.3f} "
          f"syn/str={out['syn_over_str']:.3f}")
    return out


# ==========================================================================
# Probes
# ==========================================================================

def _act_eta2(mat, level_features, level_rules, s, L, block_size):
    """FM-free VARIANCE ALLOCATION of the last-token post_block6 activation: what
    fraction of its variance is explained by the latent FEATURE at each level (structure)
    versus by the RULE CHOICE at each level (which of the m productions realized it --
    synonym identity, irrelevant downstream except through the latent).

    This is the graded, probe-free, scale-free view of the outcome that matters most
    here: a representation that discards synonym identity while keeping structure is a
    REPRESENTATIONAL fact, so unlike a collapse equilibrium it would survive into a system
    where an outer loop keeps supplying fresh compressible material. eta^2 is a variance
    FRACTION, so it is immune to the scale gauge; and it is not capacity-limited the way
    the rule probe is (which saturates at 1.000 in the unpressured model and therefore
    cannot show graded movement until it breaks).
    """
    import numpy as np
    gm = mat.mean(0)
    sst = float(((mat - gm) ** 2).sum())
    nC = mat.shape[0]

    def eta2(lab):
        ss = 0.0
        for u in np.unique(lab):
            msk = lab == u
            ss += float(msk.sum()) * float(((mat[msk].mean(0) - gm) ** 2).sum())
        return ss / max(sst, 1e-30)

    out = {}
    for ell in range(L):
        anc = (block_size - 1) // (s ** (L - ell))
        out[f"d{L - ell}"] = {"feature": eta2(level_features[ell][:nC, anc]),
                              "rule": eta2(level_rules[ell][:nC, anc])}
    return out


def _rank_variants(mat, n_embd):
    """Separate the FM-free rank readout into SCALE and SHAPE.

    `_compute_effective_rank` normalizes singular values by their SUM, so it is exactly
    invariant to a UNIFORM rescale h -> c*h (verified: delta = 0.0 at c = 1e-3). A rank
    move therefore already rules out uniform shrink -- but ANISOTROPIC shrink is still a
    scale effect and reproduces the whole observed move on its own (a pure per-dim gain
    profile, factor structure untouched, moves this estimator -12 points). Two controls:

      rank            covariance spectrum          (the incumbent readout)
      rank_unitnorm   rows rescaled to unit L2     removes per-EXAMPLE magnitude
      rank_corr       per-DIM standardization      the CORRELATION spectrum

    If `rank` falls while `rank_corr` does not, the drop is per-dimension gain change --
    "be small, unevenly" -- rather than a reduction in the number of independent factors.
    Same estimator and same subsample seed as `_compute_effective_rank`, so the three are
    directly comparable. NOTE this does not touch RHM_FM_REGULARIZER's published claim,
    which is a statement about `rank` at matched knowledge and stands as recorded.
    """
    import numpy as np
    r, t1 = _compute_effective_rank(mat, n_embd)
    nrm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    r_u, _ = _compute_effective_rank(mat / nrm, n_embd)
    sd = mat.std(0, keepdims=True)
    r_c, _ = _compute_effective_rank((mat - mat.mean(0, keepdims=True)) / (sd + 1e-12), n_embd)
    dv = mat.var(0)
    return {"effective_rank_pct": float(r / n_embd * 100), "top1_pc": float(t1),
            "rank_unitnorm_pct": float(r_u / n_embd * 100),
            "rank_corr_pct": float(r_c / n_embd * 100),
            "mean_dim_var": float(dv.mean()),
            "dimvar_p90_over_p10": float(np.percentile(dv, 90) / (np.percentile(dv, 10) + 1e-30)),
            "mean_norm": float(np.linalg.norm(mat, axis=1).mean()),
            "norm_cv": float(np.linalg.norm(mat, axis=1).std()
                             / (np.linalg.norm(mat, axis=1).mean() + 1e-12))}


def _probe_labels(acts, labels, n_classes, n_embd, device, probe_steps=400, split=0.7,
                  lr=0.05):
    """Best-over-blocks last-token linear decodability of an arbitrary categorical label.
    Same estimator as rhm_wd_sweep._probe_knowledge, generalized off feature ids so it can
    read the SYNONYM CHOICE (rule index) as well as the latent feature."""
    import torch
    import torch.nn.functional as F
    N = labels.shape[0]
    g = torch.Generator(device=device).manual_seed(0)
    perm = torch.randperm(N, generator=g, device=device)
    tr, te = perm[:int(split * N)], perm[int(split * N):]
    ytr, yte = labels[tr], labels[te]
    best_acc, best_bl = 0.0, -1
    for bl, X in acts.items():
        mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        W = torch.zeros(n_embd, n_classes, device=device, requires_grad=True)
        b = torch.zeros(n_classes, device=device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr)
        for _ in range(probe_steps):
            opt.zero_grad()
            F.cross_entropy(Xtr @ W + b, ytr).backward()
            opt.step()
        with torch.no_grad():
            acc = float(((Xte @ W + b).argmax(1) == yte).float().mean())
        if acc > best_acc:
            best_acc, best_bl = acc, bl
    return {"acc": best_acc, "block": best_bl}


def _probe_rules(model, eval_x, level_rules, s, L, m, n_embd, n_layer, device,
                 probe_steps=400):
    """Per-level decodability of the SYNONYM CHOICE from the last-token residual stream.

    d1 (ell = L-1) is the pure object: which of the m productions realized the final
    constituent. It is irrelevant to everything downstream except through the latent, so
    it is exactly the part of x_{t+1} an input-invariance collapse should discard first."""
    import numpy as np
    import torch
    model.eval()
    seq_len = s ** L
    acts = {bl: [] for bl in range(n_layer)}
    with torch.no_grad():
        for i in range(0, eval_x.shape[0], 256):
            _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
            for bl in range(n_layer):
                acts[bl].append(inter[f"post_block{bl}"][:, -1, :])
    acts = {bl: torch.cat(acts[bl], 0) for bl in range(n_layer)}
    out = {}
    for ell in range(L):
        last_anc = (seq_len - 1) // (s ** (L - ell))
        lab = torch.from_numpy(level_rules[ell][:, last_anc].astype(np.int64)).to(device)
        out[f"d{L - ell}"] = _probe_labels(acts, lab, m, n_embd, device, probe_steps)
    del acts
    torch.cuda.empty_cache()
    return out


def _sk_mo_probes(model, fm, arm, eval_x, n_embd, n_layer, batch_size, device,
                  probe_steps, seed):
    """Self-knowledge + the object/meta split (rhm_latent_loop._sk_probes and
    _meta_object_probes merged -- they share one estimator: linear, no input
    standardization, MSE, global-variance R^2).

    Probe targets at target position p:  prediction (object), residual (composite),
    ortho_residual (pure meta, residual with the prediction direction removed), target.
    Two ALIGNMENTS, both computed for both arms so the arms are never read differently:
      post : from the residual stream at p           (the depth arm's native convention)
      pre  : from the residual stream at p-1         (the forecasting state -- for the
             temporal arm this is *anticipatory* self-knowledge, the honest analogue)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    model.eval(); fm.eval()
    layers = ["post_block0", f"post_block{n_layer - 1}"]
    acc = {("pre", k): [] for k in layers}
    acc.update({("post", k): [] for k in layers})
    preds, resids, orthos, targets = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(eval_x), batch_size):
            b = eval_x[i:i + batch_size]
            _, _, vi = model(b, return_intermediates=True)
            pred, tgt = _pred_tgt(fm, vi, arm)
            r = tgt - pred
            pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            rp = r - (r * pn).sum(-1, keepdim=True) * pn
            preds.append(pred.reshape(-1, n_embd).cpu())
            resids.append(r.reshape(-1, n_embd).cpu())
            orthos.append(rp.reshape(-1, n_embd).cpu())
            targets.append(tgt.reshape(-1, n_embd).cpu())
            for k in layers:                       # target positions are 1..T-1
                acc[("post", k)].append(vi[k][:, 1:, :].reshape(-1, n_embd).cpu())
                acc[("pre", k)].append(vi[k][:, :-1, :].reshape(-1, n_embd).cpu())
    tg = {"prediction": torch.cat(preds), "residual": torch.cat(resids),
          "ortho_residual": torch.cat(orthos), "target": torch.cat(targets)}
    acc = {k: torch.cat(v_) for k, v_ in acc.items()}
    n_rows = tg["residual"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_rows, generator=g)
    n_tr = int(0.8 * n_rows)
    tr, te = perm[:n_tr], perm[n_tr:]

    def probe(X, Y):
        Xtr, Ytr = X[tr].to(device), Y[tr].to(device)
        Xte, Yte = X[te].to(device), Y[te].to(device)
        p = nn.Linear(n_embd, n_embd).to(device)
        opt = torch.optim.Adam(p.parameters(), lr=1e-3)
        bs = min(4096, Xtr.shape[0])
        for _ in range(probe_steps):
            idx = torch.randint(Xtr.shape[0], (bs,), device=device)
            F.mse_loss(p(Xtr[idx]), Ytr[idx]).backward()
            opt.step(); opt.zero_grad()
        with torch.no_grad():
            pr = p(Xte)
            r2 = 1.0 - F.mse_loss(pr, Yte).item() / max(Yte.var().item(), 1e-12)
            cos = float(F.cosine_similarity(pr, Yte, dim=-1).mean())
        del Xtr, Ytr, Xte, Yte, p, opt
        torch.cuda.empty_cache()
        return {"r2": r2, "cosine": cos}

    stats = {f"{t}_var": float(Y.var()) for t, Y in tg.items()}
    stats["ortho_over_res"] = stats["ortho_residual_var"] / (stats["residual_var"] + 1e-12)
    stats["res_over_tgt"] = stats["residual_var"] / (stats["target_var"] + 1e-12)
    out = {"target_stats": stats}
    for align in ["pre", "post"]:
        for k in layers:
            out[f"{align}|{k}"] = {t: probe(acc[(align, k)], tg[t]) for t in tg}
    bN = f"post_block{n_layer - 1}"
    for align in ["pre", "post"]:
        a = out[f"{align}|{bN}"]
        print(f"    SK[{align}|{bN}] OBJ={a['prediction']['r2']:+.3f} "
              f"META={a['ortho_residual']['r2']:+.3f} composite={a['residual']['r2']:+.3f} "
              f"(res/tgt var {stats['res_over_tgt']:.4f})")
    del tg, acc
    torch.cuda.empty_cache()
    return out


def _ensemble(model, make_fm, get_batch, arm, fm_steps, fm_lr, eval_x, n_embd,
              batch_size, device, n_fm, base_seed):
    """Do independently-seeded fresh FMs leave the SAME residual (input-determined,
    DGP-aligned) or different ones (FM-idiosyncratic)? rhm_latent_loop._ensemble_agreement,
    generalized over the arm."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    for p in model.parameters():
        p.requires_grad_(False)
    res = []
    for k in range(n_fm):
        torch.manual_seed(base_seed + 1000 * (k + 1))
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)
        for _ in range(fm_steps):
            fm.train()
            x, y = get_batch()
            with torch.no_grad():
                _, _, vi = model(x, y, return_intermediates=True)
            pred, tgt = _pred_tgt(fm, vi, arm)
            F.mse_loss(pred, tgt.detach()).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        fm.eval()
        rk = []
        with torch.no_grad():
            for i in range(0, len(eval_x), batch_size):
                b = eval_x[i:i + batch_size]
                _, _, vi = model(b, return_intermediates=True)
                pred, tgt = _pred_tgt(fm, vi, arm)
                rk.append((tgt - pred)[:, -1, :].cpu())
        res.append(torch.cat(rk))
        del fm, opt
        torch.cuda.empty_cache()
    R = torch.stack(res)
    Rc = R - R.mean(dim=1, keepdim=True)
    Rn = F.normalize(Rc, dim=-1)
    cps = [float((Rn[i] * Rn[j]).sum(-1).mean()) for i in range(n_fm) for j in range(i + 1, n_fm)]
    return {"pairwise_cos": float(np.mean(cps)), "n_fm": n_fm,
            "shared_norm_frac": float(R.mean(0).norm(dim=-1).mean() / R.norm(dim=-1).mean())}


# ==========================================================================
# Phase 2 -- analysis of ONE checkpoint (identical code for every arm)
# ==========================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def analyze_ll_ckpt(
    label: str, key: str, checkpoint_path: str, step: int,
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    fm_train_steps: int = 6000, fm_lr: float = 1e-3, fm_seed: int = 911,
    n_eval_sequences: int = 8000, n_sk_sequences: int = 1200, n_swap: int = 3000,
    batch_size: int = 64, sk_probe_steps: int = 400, ensemble_n: int = 3,
    ensemble_steps: int = 3000, out_tag: str = "ll1",
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    L = depth
    block_size = s ** L
    device = "cuda"
    volume.reload()

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    corpus = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data, val_data = corpus[:split], corpus[split:]

    def mk_batch(d):
        def f():
            ix = torch.randint(len(d) - block_size - 1, (batch_size,))
            x = torch.stack([d[i:i + block_size] for i in ix])
            y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
            return x.to(device), y.to(device)
        return f
    get_train, get_val = mk_batch(train_data), mk_batch(val_data)

    with torch.no_grad():
        main_val_loss = float(np.mean([float(model(*get_val())[1]) for _ in range(20)]))

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    sk_x = eval_x[:n_sk_sequences]

    print(f"\n=== {label} step {step}   val={main_val_loss:.4f} ===")

    # ---- knowledge gate (the denominator) + synonym-identity probe ----
    know = _probe_knowledge(model, eval_x, level_features, s, L, n_embd, v, n_layer, device)
    rule = _probe_rules(model, eval_x, level_rules, s, L, m, n_embd, n_layer, device)
    print("    feature: " + " ".join(f"d{L-e}:{know[f'd{L-e}']['acc']:.3f}" for e in range(L)))
    print(f"    rule   : " + " ".join(f"d{L-e}:{rule[f'd{L-e}']['acc']:.3f}" for e in range(L))
          + f"   (chance {1.0/m:.3f})")

    # ---- FM-free complexity / collapse readout on post_block6 ----
    # Also the FM-FREE, SCALE-FREE input-invariance index. §8's predicted degeneracy is
    # "make h6[t+1] insensitive to x_{t+1}", whose signature is that the state stops
    # MOVING per token. delta_over_tgt = var(h6[t+1] - h6[t]) / var(h6[t+1]) measures
    # exactly that with no forward model in the loop and no dependence on scale, so it
    # cannot be gamed by the shrink escape hatch and cannot be gamed by the FM.
    acts, dv, tv = [], [], []
    with torch.no_grad():
        for i in range(0, len(eval_x), 256):
            _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
            h = inter[TGT_BLOCK]
            acts.append(h[:, -1, :].cpu().numpy())
            dv.append(float((h[:, 1:, :] - h[:, :-1, :]).var()))
            tv.append(float(h[:, 1:, :].var()))
    mat = np.concatenate(acts, 0)
    act_rank = _rank_variants(mat, n_embd)
    act_rank["delta_var"] = float(np.mean(dv))
    act_rank["tgt_var_allpos"] = float(np.mean(tv))
    act_rank["delta_over_tgt"] = act_rank["delta_var"] / max(act_rank["tgt_var_allpos"], 1e-30)
    act_eta2 = _act_eta2(mat, level_features, level_rules, s, L, block_size)
    print(f"    input-invariance index delta/tgt={act_rank['delta_over_tgt']:.4f} "
          f"(delta_var={act_rank['delta_var']:.5f} tgt_var={act_rank['tgt_var_allpos']:.5f})")
    print("    act eta2 feature: "
          + " ".join(f"d{L-e}:{act_eta2[f'd{L-e}']['feature']:.3f}" for e in range(L)))
    print("    act eta2 rule   : "
          + " ".join(f"d{L-e}:{act_eta2[f'd{L-e}']['rule']:.3f}" for e in range(L)))
    print(f"    FM-free act rank={act_rank['effective_rank_pct']:.1f}% "
          f"(unitnorm {act_rank['rank_unitnorm_pct']:.1f}% corr {act_rank['rank_corr_pct']:.1f}%) "
          f"top1={act_rank['top1_pc']*100:.1f}% |h|={act_rank['mean_norm']:.2f} "
          f"dimvar={act_rank['mean_dim_var']:.4f} p90/p10={act_rank['dimvar_p90_over_p10']:.1f}")

    # ---- the counterfactual instrument ----
    swap = _swap_sensitivity(model, rules, n_swap, 1234, s, L, v, m, device)

    # ---- both fresh FMs on this checkpoint (leakage-proof: seed != training seed) ----
    def make_fm():
        return TransformerForwardModel(d_model=n_embd, d_head=REG_FM["dh"],
                                       n_head=REG_FM["nh"], n_layer=REG_FM["nl"],
                                       mlp_mult=REG_FM["mm"], block_size=block_size).to(device)

    arms = {}
    for arm in ["temporal", "depth"]:
        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)
        for _ in range(fm_train_steps):
            fm.train()
            x, y = get_train()
            with torch.no_grad():
                _, _, vi = model(x, y, return_intermediates=True)
            pred, tgt = _pred_tgt(fm, vi, arm)
            F.mse_loss(pred, tgt.detach()).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        fm.eval()

        cosl, mses, tvars, rlast, tlast, sub = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(eval_x), 256):
                _, _, vi = model(eval_x[i:i + 256], return_intermediates=True)
                pred, tgt = _pred_tgt(fm, vi, arm)
                r = tgt - pred
                cosl.append(float(F.cosine_similarity(pred, tgt, dim=-1).mean()))
                mses.append(float((r ** 2).mean()))
                tvars.append(float(tgt.var()))
                rlast.append(r[:, -1, :].cpu().numpy())
                tlast.append(tgt[:, -1, :].cpu().numpy())
                sub.append(r.reshape(-1, n_embd)[::37].cpu().numpy())
        r_last = np.concatenate(rlast, 0)
        t_last = np.concatenate(tlast, 0)
        r_all = np.concatenate(sub, 0)
        rr, rtop1 = _compute_effective_rank(r_all, n_embd)
        # per-level eta^2 of the last-position residual against the true latents,
        # explicitly indexed on leaf position T-1 (no reliance on tensor length).
        gm = r_last.mean(0)
        sst = float(((r_last - gm) ** 2).sum())

        def eta2(lab):
            ss = 0.0
            for u in np.unique(lab):
                msk = lab == u
                ss += float(msk.sum()) * float(((r_last[msk].mean(0) - gm) ** 2).sum())
            return ss / max(sst, 1e-12)
        nC = r_last.shape[0]
        et = {}
        for ell in range(L):
            anc = (block_size - 1) // (s ** (L - ell))
            et[f"d{L - ell}"] = {"feature": eta2(level_features[ell][:nC, anc]),
                                 "rule": eta2(level_rules[ell][:nC, anc])}
        sk = _sk_mo_probes(model, fm, arm, sk_x, n_embd, n_layer, 64, device,
                           sk_probe_steps, 7)
        ens = _ensemble(model, make_fm, get_train, arm, ensemble_steps, fm_lr,
                        eval_x[:2000], n_embd, 256, device, ensemble_n, fm_seed + 5)
        # SCALE-FREE predictability, and skill against the TRIVIAL forecaster.
        # mse_over_tgtvar is skill vs the mean predictor and is comparable across arms
        # (both targets are post_block6, same marginal variance). For the temporal arm the
        # mean predictor is not the relevant baseline -- ACTIVE_VISION scored 0.94 by
        # echoing carried state -- so mse_over_deltavar scores it against the IDENTITY
        # forecaster instead. Both are invariant to the shrink escape hatch.
        arms[arm] = {
            "fm_cosine": float(np.mean(cosl)), "mse": float(np.mean(mses)),
            "tgt_var": float(np.mean(tvars)),
            "mse_over_tgtvar": float(np.mean(mses)) / max(float(np.mean(tvars)), 1e-12),
            "mse_over_deltavar": float(np.mean(mses)) / max(act_rank["delta_var"], 1e-30),
            "res_norm_last": float(np.linalg.norm(r_last, axis=1).mean()),
            "tgt_norm_last": float(np.linalg.norm(t_last, axis=1).mean()),
            "res_rank_pct": float(rr / n_embd * 100), "res_top1_pc": float(rtop1),
            "eta2": et, "sk": sk, "ensemble": ens,
        }
        a = arms[arm]
        print(f"    [{arm} FM] cos={a['fm_cosine']:.4f} mse/var={a['mse_over_tgtvar']:.4f} "
              f"resRank={a['res_rank_pct']:.1f}% ensCos={ens['pairwise_cos']:.3f} "
              f"| eta2 d1f={et['d1']['feature']:.3f} d1r={et['d1']['rule']:.3f} "
              f"d4f={et['d4']['feature']:.3f}")
        del fm, opt
        torch.cuda.empty_cache()

    result = {"label": label, "step": step, "main_val_loss": main_val_loss,
              "weight": _weight_norms(model), "knowledge": know, "rule_probe": rule,
              "act_rank": act_rank, "act_eta2": act_eta2, "swap": swap, "arms": arms}
    parts = f"{DATA_DIR}/rhm_temporal_local_loss/parts_{out_tag}"
    os.makedirs(parts, exist_ok=True)
    with open(f"{parts}/{label}_step{step}.json", "w") as fp:
        json.dump(result, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


# ==========================================================================
# Orchestrators
# ==========================================================================

@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def tll_sweep(lams: str = LAM_DEFAULT, n_steps: int = MATCH_STEP, arm: str = "temporal"):
    """Train the temporal arm at each lambda (parallel, each resumable)."""
    lam_list = [float(x) for x in lams.split(",")]
    print(f"=== temporal-local-loss sweep (m2) === lams={lam_list} wd={WD_FIXED} "
          f"n_steps={n_steps}  {_arm_src(arm)} -> {TGT_BLOCK}[t+1]")
    handles = [(lam, train_temporal_reg.spawn(lam=lam, arm=arm, n_steps=n_steps))
               for lam in lam_list]
    for lam, h in handles:
        try:
            r = h.get()
            print(f"  lam={lam}: {len(r['checkpoints'])} checkpoints")
        except Exception as e:
            print(f"  lam={lam}: FAILED -- {e}")
    return {"lams": lam_list, "n_steps": n_steps}


def _label_dir(key, label):
    """label -> training dir. 'lam0_base' | '{depth,temporal}_lam<tag>'."""
    if label == "lam0_base":
        return _sweep_dir(key, WD_FIXED)
    kind, lamtag = label.split("_lam")
    lam = float(lamtag.replace("p", ".").replace("m", "-"))
    return _fmreg_dir(key, lam) if kind == "depth" else _tll_dir(key, lam)


def _sources(key, lams, step, extra=""):
    """(label, checkpoint_path, step) for every arm.

    The primary block is read at the MATCHED step: same budget, same batches, same
    protocol on both sides. `extra` adds off-matched-step reads as 'label@step' --
    used for the incumbent depth ENDPOINTS (150k/300k), because the depth collapse is
    only ~half done at 80k, so a matched-budget read alone would understate it. Those
    endpoints anchor the matched-ACHIEVED-COMPRESSION comparison instead: matched lambda
    is not matched pressure (the temporal residual's irreducible floor supplies extra
    input-invariance pressure at equal lambda), so the arms are also compared at equal
    FM-free activation rank, which is protocol-free.

    depth_lam* and lam0_base are INCUMBENT rhm_fm_regularizer / rhm_wd_sweep checkpoints --
    nothing is retrained on that side.
    """
    out = [("lam0_base", os.path.join(_sweep_dir(key, WD_FIXED), f"ckpt_step{step}.pt"), step)]
    for lam in lams:
        out.append((f"depth_lam{_lam_tag(lam)}",
                    os.path.join(_fmreg_dir(key, lam), f"ckpt_step{step}.pt"), step))
    for lam in lams:
        out.append((f"temporal_lam{_lam_tag(lam)}",
                    os.path.join(_tll_dir(key, lam), f"ckpt_step{step}.pt"), step))
    for e in [x.strip() for x in extra.split(",") if x.strip()]:
        lb, st = e.split("@")
        st = int(st)
        out.append((f"{lb}@{st}", os.path.join(_label_dir(key, lb), f"ckpt_step{st}.pt"), st))
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def analyze_ll(lams: str = LAM_DEFAULT, step: int = MATCH_STEP, v: int = 16, m: int = 2,
               s: int = 2, depth: int = 6, out_tag: str = "ll1", extra: str = "",
               fm_train_steps: int = 6000, n_swap: int = 3000, force: bool = False):
    lam_list = [float(x) for x in lams.split(",")]
    key = _tb_key(v, s, depth, m)
    volume.reload()
    parts = f"{DATA_DIR}/rhm_temporal_local_loss/parts_{out_tag}"
    os.makedirs(parts, exist_ok=True)

    src = _sources(key, lam_list, step, extra)
    pending, have = [], []
    for label, path, st in src:
        part = f"{parts}/{label}_step{st}.json"
        if os.path.exists(part) and not force:
            have.append(label); continue
        if not os.path.exists(path):
            print(f"  {label}: MISSING checkpoint {path}"); continue
        pending.append((label, analyze_ll_ckpt.spawn(
            label=label, key=key, checkpoint_path=path, step=st, v=v, m=m, s=s,
            depth=depth, out_tag=out_tag, fm_train_steps=fm_train_steps, n_swap=n_swap)))
    print(f"=== analyze_ll: {len(pending)} jobs ({len(have)} cached); "
          f"matched step {step}, extras=[{extra}] ===")
    for label, h in pending:
        try:
            h.get()
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    volume.reload()
    rows = []
    for label, _, st in src:
        p = f"{parts}/{label}_step{st}.json"
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    _report(rows)
    out = {"key": key, "step": step, "lams": lam_list, "rows": rows}
    with open(f"{DATA_DIR}/rhm_temporal_local_loss/summary_{out_tag}.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


def _report(rows):
    if not rows:
        return
    print("\n" + "=" * 118)
    print("SIGNATURE TABLE  (all arms at the matched step; lam0_base is the wd=0.1 anchor)")
    print("=" * 118)
    print(f"{'arm':>22} {'val':>6} {'d6':>5} {'d1':>5} {'rule_d1':>8} {'actRk%':>7} "
          f"{'top1%':>6} {'dimVar':>7} | {'Tcos':>6} {'Tmse/v':>7} {'TskPre':>7} "
          f"{'TskPost':>8} {'Tens':>6} | {'Dcos':>6} {'Dmse/v':>7} {'Dsk':>7}")
    for r in rows:
        t, d = r["arms"]["temporal"], r["arms"]["depth"]
        b7 = "post_block7"
        print(f"{r['label']:>22} {r['main_val_loss']:>6.3f} "
              f"{r['knowledge']['d6']['acc']:>5.2f} {r['knowledge']['d1']['acc']:>5.2f} "
              f"{r['rule_probe']['d1']['acc']:>8.3f} "
              f"{r['act_rank']['effective_rank_pct']:>7.1f} "
              f"{r['act_rank']['top1_pc']*100:>6.1f} {r['act_rank']['mean_dim_var']:>7.3f} | "
              f"{t['fm_cosine']:>6.3f} {t['mse_over_tgtvar']:>7.4f} "
              f"{t['sk'][f'pre|{b7}']['residual']['r2']:>7.3f} "
              f"{t['sk'][f'post|{b7}']['residual']['r2']:>8.3f} "
              f"{t['ensemble']['pairwise_cos']:>6.3f} | "
              f"{d['fm_cosine']:>6.3f} {d['mse_over_tgtvar']:>7.4f} "
              f"{d['sk'][f'post|{b7}']['residual']['r2']:>7.3f}")
    print("\n" + "=" * 118)
    print("INPUT-INVARIANCE / ALEATORIC-NULL TABLE  (h6[last] displacement under matched twins)")
    print("=" * 118)
    print(f"{'arm':>22} {'|h|':>7} {'d_syn':>7} {'d_str':>7} {'d_rand':>7} {'d_other':>8} "
          f"{'syn/oth':>8} {'str/oth':>8} {'syn/str':>8} {'ham_syn':>8} {'ham_str':>8}")
    for r in rows:
        w = r["swap"]
        print(f"{r['label']:>22} {w['mean_h_norm']:>7.2f} {w['synonym']['d']:>7.3f} "
              f"{w['structure']['d']:>7.3f} {w['random']['d']:>7.3f} {w['other']['d']:>8.3f} "
              f"{w['d_syn_rel']:>8.3f} {w['d_str_rel']:>8.3f} {w['syn_over_str']:>8.3f} "
              f"{w['synonym']['hamming']:>8.2f} {w['structure']['hamming']:>8.2f}")
    print("\n" + "=" * 118)
    print("META/OBJECT (temporal FM, post_block7) + per-level rule probe")
    print("=" * 118)
    print(f"{'arm':>22} {'OBJpre':>7} {'METApre':>8} {'OBJpost':>8} {'METApost':>9} "
          f"{'ortho/res':>10} | {'rule d1':>8} {'d2':>6} {'d3':>6} {'d4':>6} "
          f"{'feat d1':>8} {'d4':>6} {'d6':>6}")
    for r in rows:
        t = r["arms"]["temporal"]
        b7 = "post_block7"
        st = t["sk"]["target_stats"]
        print(f"{r['label']:>22} {t['sk'][f'pre|{b7}']['prediction']['r2']:>7.3f} "
              f"{t['sk'][f'pre|{b7}']['ortho_residual']['r2']:>8.3f} "
              f"{t['sk'][f'post|{b7}']['prediction']['r2']:>8.3f} "
              f"{t['sk'][f'post|{b7}']['ortho_residual']['r2']:>9.3f} "
              f"{st['ortho_over_res']:>10.3f} | "
              f"{r['rule_probe']['d1']['acc']:>8.3f} {r['rule_probe']['d2']['acc']:>6.3f} "
              f"{r['rule_probe']['d3']['acc']:>6.3f} {r['rule_probe']['d4']['acc']:>6.3f} "
              f"{r['knowledge']['d1']['acc']:>8.3f} {r['knowledge']['d4']['acc']:>6.3f} "
              f"{r['knowledge']['d6']['acc']:>6.3f}")

    # GAUGE vs CONTENT. This is an INSTRUMENT finding, not a pathology to engineer around.
    # Uniform scale in a residual stream is near-gauge: every consumer of post_block6 is
    # preceded by a LayerNorm (block7.ln_1, ln_f before lm_head), _probe_knowledge
    # standardizes its inputs, and eta^2 is a variance fraction -- so the scale is
    # discarded by essentially every downstream reader, which is exactly why d6 holds at
    # 0.94-0.95 through a ~300x variance drop. But ||FM(src) - tgt||^2 is NOT scale-free
    # and falls quadratically in it. Consequence: the raw local-loss term spends most of
    # its dynamic range on a gauge and is a bad readout of anything. Report on mse/var and
    # cos instead. corrRk vs actRk says whether the rank move is a per-dim gain profile or
    # a real reduction in independent factors.
    print("\n" + "=" * 118)
    print("GAUGE vs CONTENT  (the raw term is ~90% gauge; mse/var and cos are the "
          "scale-free readouts)")
    print("=" * 118)
    print(f"{'arm':>22} {'|h|':>7} {'dimVar':>9} {'p90/p10':>8} {'actRk%':>7} {'uniRk%':>7} "
          f"{'corrRk%':>8} {'top1%':>6} | {'Tmse/var':>9} {'Dmse/var':>9} {'d6':>5}")
    for r in rows:
        a, t, d = r["act_rank"], r["arms"]["temporal"], r["arms"]["depth"]
        print(f"{r['label']:>22} {a['mean_norm']:>7.2f} {a['mean_dim_var']:>9.5f} "
              f"{a['dimvar_p90_over_p10']:>8.1f} {a['effective_rank_pct']:>7.1f} "
              f"{a['rank_unitnorm_pct']:>7.1f} {a['rank_corr_pct']:>8.1f} "
              f"{a['top1_pc']*100:>6.1f} | {t['mse_over_tgtvar']:>9.4f} "
              f"{d['mse_over_tgtvar']:>9.4f} {r['knowledge']['d6']['acc']:>5.2f}")

    # The SCALE-FREE cross-arm axis. If both arms take the shrink escape hatch, the raw
    # term is ~90% shared-degeneracy noise and a conditioning effect could only show here.
    # But scale-free predictability is ALSO gameable in the temporal arm -- by making
    # h6[t+1] insensitive to x_{t+1}, which raises cos without engaging the conditioning.
    # So it must be read jointly with the two FM-free invariance readouts on the right:
    # delta/tgt (does the state still move per token) and the counterfactual twins.
    print("\n" + "=" * 118)
    print("SCALE-FREE PREDICTABILITY vs INPUT-INVARIANCE  (read together: cos up WITH "
          "delta/tgt or syn/str down = degeneracy, not conditioning)")
    print("=" * 118)
    print(f"{'arm':>22} {'Tcos':>7} {'Tmse/var':>9} {'Tmse/dvar':>10} {'Dcos':>7} "
          f"{'Dmse/var':>9} | {'delta/tgt':>10} {'synRel':>7} {'strRel':>7} {'syn/str':>8} "
          f"{'ruleD1':>7}")
    for r in rows:
        a, t, d, w = r["act_rank"], r["arms"]["temporal"], r["arms"]["depth"], r["swap"]
        print(f"{r['label']:>22} {t['fm_cosine']:>7.4f} {t['mse_over_tgtvar']:>9.4f} "
              f"{t['mse_over_deltavar']:>10.4f} {d['fm_cosine']:>7.4f} "
              f"{d['mse_over_tgtvar']:>9.4f} | {a['delta_over_tgt']:>10.4f} "
              f"{w['d_syn_rel']:>7.3f} {w['d_str_rel']:>7.3f} {w['syn_over_str']:>8.3f} "
              f"{r['rule_probe']['d1']['acc']:>7.3f}")

    # OUTCOME 3 -- synonym identity discarded, structure retained. The branch worth the
    # most care: a representational fact persists into a system where an outer loop keeps
    # supplying fresh compressible material, whereas a collapse equilibrium is a property
    # of this closed setup and would not. Three independent views of the same question,
    # all scale-free:
    #   allocation  act eta^2 rule vs feature -- what share of h6's variance is spent on
    #               which production realized the constituent vs on the latent itself.
    #               Graded, probe-free, and not capacity-limited (rule_d1 saturates at
    #               1.000 unpressured, so it cannot move until it breaks).
    #   decodability rule probe per level
    #   causal      matched counterfactual twins (d_syn vs d_str)
    # The depth arm is the null reference: it discards nothing (rule_d1 = 1.000 at every
    # lambda and step, syn/str pinned 0.59-0.67 including lambda=0).
    print("\n" + "=" * 118)
    print("OUTCOME 3 -- SYNONYM vs STRUCTURE  (allocation / decodability / causal; "
          "all three should move together if real)")
    print("=" * 118)
    print(f"{'arm':>22} | {'eta2 rule d1':>12} {'d2':>6} {'d3':>6} | {'eta2 feat d1':>12} "
          f"{'d2':>6} {'d4':>6} {'d6':>6} | {'rule/feat d1':>12} | {'ruleP d1':>9} "
          f"{'synRel':>7} {'strRel':>7} {'syn/str':>8}")
    for r in rows:
        e, w = r.get("act_eta2"), r["swap"]
        if e is None:
            continue
        rf = e["d1"]["rule"] / max(e["d1"]["feature"], 1e-12)
        print(f"{r['label']:>22} | {e['d1']['rule']:>12.4f} {e['d2']['rule']:>6.4f} "
              f"{e['d3']['rule']:>6.4f} | {e['d1']['feature']:>12.4f} "
              f"{e['d2']['feature']:>6.4f} {e['d4']['feature']:>6.4f} "
              f"{e['d6']['feature']:>6.4f} | {rf:>12.4f} | "
              f"{r['rule_probe']['d1']['acc']:>9.3f} {w['d_syn_rel']:>7.3f} "
              f"{w['d_str_rel']:>7.3f} {w['syn_over_str']:>8.3f}")

    # Matched-ACHIEVED-COMPRESSION view. Matched lambda is not matched pressure: the
    # temporal residual's irreducible floor supplies input-invariance pressure the depth
    # residual has none of, so at equal lambda the temporal arm exerts strictly more.
    # Sorting by FM-free rank makes the arms comparable at equal achieved compression,
    # which is also immune to the depth arm being only ~half-collapsed at 80k.
    print("\n" + "=" * 118)
    print("MATCHED-COMPRESSION VIEW  (sorted by FM-free act rank; compare arms at EQUAL rank)")
    print("=" * 118)
    b7 = "post_block7"
    print(f"{'arm':>22} {'actRk%':>7} {'d6':>5} {'ruleD1':>7} {'synRel':>7} {'strRel':>7} "
          f"{'syn/str':>8} {'TskPre':>7} {'TskPost':>8} {'Tmeta':>7} {'Tens':>6} "
          f"{'Dsk':>7} {'Dmeta':>7}")
    for r in sorted(rows, key=lambda z: -z["act_rank"]["effective_rank_pct"]):
        t, d, w = r["arms"]["temporal"], r["arms"]["depth"], r["swap"]
        print(f"{r['label']:>22} {r['act_rank']['effective_rank_pct']:>7.1f} "
              f"{r['knowledge']['d6']['acc']:>5.2f} {r['rule_probe']['d1']['acc']:>7.3f} "
              f"{w['d_syn_rel']:>7.3f} {w['d_str_rel']:>7.3f} {w['syn_over_str']:>8.3f} "
              f"{t['sk'][f'pre|{b7}']['residual']['r2']:>7.3f} "
              f"{t['sk'][f'post|{b7}']['residual']['r2']:>8.3f} "
              f"{t['sk'][f'post|{b7}']['ortho_residual']['r2']:>7.3f} "
              f"{t['ensemble']['pairwise_cos']:>6.3f} "
              f"{d['sk'][f'post|{b7}']['residual']['r2']:>7.3f} "
              f"{d['sk'][f'post|{b7}']['ortho_residual']['r2']:>7.3f}")


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def rank_traj(arm: str = "depth", lams: str = LAM_DEFAULT, v: int = 16, m: int = 2,
              s: int = 2, depth: int = 6, n_layer: int = 8, n_head: int = 8,
              n_embd: int = 256, min_step: int = 20000, n_eval_sequences: int = 6000,
              n_swap: int = 3000, probe_steps: int = 300, include_base: bool = True,
              out_tag: str = "traj1"):
    """FM-free compression TRAJECTORY -- no forward model is trained, so this is cheap.

    Two things it settles that a single matched step cannot:

    1. HOW FAR ALONG the incumbent depth collapse is at MATCH_STEP. RHM_FM_REGULARIZER
       published the signature at 150k/300k and noted FM-free rank was still trending
       down at 300k. If the depth arm at 80k is materially milder than published,
       "temporal collapses less" would be an early read, not a difference.
    2. What RANK RANGE each arm's lambda sweep spans. Matched lambda is NOT matched
       pressure: the temporal residual has an irreducible floor whose gradient is itself
       the input-invariance pressure, so at equal lambda the temporal arm exerts
       simplify-pressure PLUS discard-pressure. The honest comparison is at matched
       ACHIEVED compression (the idiom RHM_FM_REGULARIZER already uses for the knowledge
       gate), which needs the two sweeps to span overlapping rank. This says whether
       they do, and if not which side needs extra lambda points.
    """
    import numpy as np
    import torch
    from rhm.model import GPT

    L = depth
    block_size = s ** L
    device = "cuda"
    key = _tb_key(v, s, L, m)
    lam_list = [float(x) for x in lams.split(",")]
    volume.reload()
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    sources = []
    if include_base:
        sources.append(("lam0_base", _sweep_dir(key, WD_FIXED)))
    for lam in lam_list:
        d = _fmreg_dir(key, lam) if arm == "depth" else _tll_dir(key, lam)
        sources.append((f"{arm}_lam{_lam_tag(lam)}", d))

    out = {"key": key, "arm": arm, "rows": {}}
    for label, d in sources:
        info_path = os.path.join(d, "training_info.json")
        if not os.path.exists(info_path):
            print(f"  {label}: no training_info at {d}"); continue
        info = json.load(open(info_path))
        rows = []
        seen = set()
        for c in sorted(info["checkpoints"], key=lambda z: z["step"]):
            st = c["step"]
            if st < min_step or st in seen or not os.path.exists(c["path"]):
                continue
            seen.add(st)
            model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
            model.load_state_dict(torch.load(c["path"], map_location=device,
                                             weights_only=True))
            model.eval()
            acts = []
            with torch.no_grad():
                for i in range(0, len(eval_x), 256):
                    _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
                    acts.append(inter[TGT_BLOCK][:, -1, :].cpu().numpy())
            mat = np.concatenate(acts, 0)
            rv = _rank_variants(mat, n_embd)
            know = _probe_knowledge(model, eval_x, level_features, s, L, n_embd, v,
                                    n_layer, device, probe_steps=probe_steps)
            rule = _probe_rules(model, eval_x, level_rules, s, L, m, n_embd, n_layer,
                                device, probe_steps=probe_steps)
            sw = _swap_sensitivity(model, rules, n_swap, 1234, s, L, v, m, device)
            rows.append({"step": st, "act_rank_pct": rv["effective_rank_pct"],
                         "top1_pc": rv["top1_pc"], "mean_dim_var": rv["mean_dim_var"],
                         "rank_unitnorm_pct": rv["rank_unitnorm_pct"],
                         "rank_corr_pct": rv["rank_corr_pct"],
                         "mean_norm": rv["mean_norm"], "norm_cv": rv["norm_cv"],
                         "dimvar_p90_over_p10": rv["dimvar_p90_over_p10"],
                         "d6": know["d6"]["acc"], "d4": know["d4"]["acc"],
                         "d1": know["d1"]["acc"], "rule_d1": rule["d1"]["acc"],
                         "rule_d3": rule["d3"]["acc"],
                         "d_syn_rel": sw["d_syn_rel"], "d_str_rel": sw["d_str_rel"],
                         "syn_over_str": sw["syn_over_str"],
                         "d_syn": sw["synonym"]["d"], "d_str": sw["structure"]["d"],
                         "d_other": sw["other"]["d"]})
            print(f"  [{label}] step {st:>6}: actRk={rows[-1]['act_rank_pct']:5.1f}% "
                  f"d6={rows[-1]['d6']:.2f} ruleD1={rows[-1]['rule_d1']:.3f} "
                  f"synRel={sw['d_syn_rel']:.3f} strRel={sw['d_str_rel']:.3f}")
            del model
            torch.cuda.empty_cache()
        out["rows"][label] = rows

    print("\n" + "=" * 104)
    print(f"FM-FREE COMPRESSION TRAJECTORY ({arm} arm)")
    print("=" * 104)
    for label, rows in out["rows"].items():
        print(f"\n-- {label} --")
        print(f"{'step':>7} {'actRk%':>7} {'uniRk%':>7} {'corrRk%':>8} {'top1%':>6} "
              f"{'|h|':>7} {'dimVar':>8} {'p90/p10':>8} {'d6':>5} {'ruleD1':>7} "
              f"{'synRel':>7} {'strRel':>7} {'syn/str':>8}")
        for r in rows:
            print(f"{r['step']:>7} {r['act_rank_pct']:>7.1f} "
                  f"{r['rank_unitnorm_pct']:>7.1f} {r['rank_corr_pct']:>8.1f} "
                  f"{r['top1_pc']*100:>6.1f} {r['mean_norm']:>7.2f} "
                  f"{r['mean_dim_var']:>8.4f} {r['dimvar_p90_over_p10']:>8.1f} "
                  f"{r['d6']:>5.2f} {r['rule_d1']:>7.3f} {r['d_syn_rel']:>7.3f} "
                  f"{r['d_str_rel']:>7.3f} {r['syn_over_str']:>8.3f}")
    save = f"{DATA_DIR}/rhm_temporal_local_loss"
    os.makedirs(save, exist_ok=True)
    with open(f"{save}/rank_traj_{arm}_{out_tag}.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def smoke(v: int = 16, m: int = 2, s: int = 2, depth: int = 6):
    """Cheap end-to-end check: twin construction invariants, a few hundred training
    steps on both arms, and the full analysis battery at tiny sizes."""
    import numpy as np
    import torch
    from rhm.model import GPT

    L, T = depth, s ** depth
    device = "cuda"
    key = _ensure_corpus_distinct(v, s, L, m, 2_000_000)
    volume.reload()
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]

    tw = _twin_leaves(rules, 500, 0, s, L, v, m)
    span = tw["span"]
    print(f"twins ok: span={span}  "
          f"syn_ham={(tw['synonym'][:, -span:] != tw['base'][:, -span:]).sum(1).mean():.2f} "
          f"str_ham={(tw['structure'][:, -span:] != tw['base'][:, -span:]).sum(1).mean():.2f} "
          f"prefix identical={bool((tw['base'][:, :-span] == tw['structure'][:, :-span]).all())}")
    # every twin must be a REACHABLE sequence of the DGP: check the last constituent is a
    # legal realization of *some* feature at every level above it.
    for nm in ["base", "synonym", "structure"]:
        leaves = tw[nm]
        ok = 0
        pairs = {tuple(rules[L - 1][f, j]) for f in range(v) for j in range(m)}
        for row in leaves[:200]:
            ok += int(tuple(row[-2:]) in pairs and tuple(row[-4:-2]) in pairs)
        print(f"  {nm}: {ok}/200 final leaf-pairs are legal productions")

    model = GPT(v, T, 8, 8, 256).to(device)
    torch.save(model.state_dict(), "/tmp/smoke_ckpt.pt")
    print("\n-- training smoke (200 steps each arm) --")
    for arm in ["temporal", "depth"]:
        train_temporal_reg.local(lam=1.0, arm=arm, dir_suffix="_smoke", n_steps=200,
                                 n_tokens=2_000_000, eval_interval=100,
                                 commit_interval=200, reg_warmup=20, reg_ramp=30)
    print("\n-- analysis smoke --")
    analyze_ll_ckpt.local(label="smoke", key=key, checkpoint_path="/tmp/smoke_ckpt.pt",
                          step=0, v=v, m=m, s=s, depth=depth, n_tokens=2_000_000,
                          fm_train_steps=200, n_eval_sequences=1000, n_sk_sequences=200,
                          n_swap=400, ensemble_steps=100, ensemble_n=2,
                          sk_probe_steps=50, out_tag="smoke")
    return "ok"
