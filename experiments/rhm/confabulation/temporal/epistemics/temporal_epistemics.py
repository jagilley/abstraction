"""Temporal FM epistemics: does the residual CARRY revision, and does the FM CONCENTRATE it?

Design doc: ../../../../../ideas/temporal_confabulation_test.md, section
            "The next thing to run: temporal FM epistemics, on this harness"
            (pre-registered there before this file was written).
Parent:     ../README.md  -- the temporal confabulation test, whose harness, guards,
            instrument sweep and oracle this file keeps and whose report channel,
            steering and CL arm it drops.
Lineage:    ../../../conditional_revision/  -- the temporal FM idiom (Gate 0), the exact
            BP oracle, Gate B's atom-aware matching, SPEC.md's unrun Gate C, and
            aleatoric_fraction's exact A/E split. All imported, none reimplemented.

WHY THIS EXISTS
  The parent settled the PRIVACY question and answered it negatively-in-an-interesting-way:
  the temporal residual is epistemically charged, and the charge is PUBLIC -- self-report
  and a capacity-matched observer track the oracle's belief revision B_t equally well
  (partial R^2 0.007 each), so their difference carries 0.0000 of it. Privilege attaches
  to the manner of updating, never to what public evidence taught you.

  That makes publicity a FEATURE for the question this program actually cares about:

      does the temporal FM's signal carry genuine epistemic content,
      and does the FM do any epistemic WORK in producing it?

  No privileged access is needed to measure a public channel. And this harness reads the
  temporal channel at 0.899 against a validated 0.905 ceiling, where rule_family's Gate 1
  belief probe died at 11-17% of its ceiling. So: repoint the battery from privacy to
  composition.

THE OBJECT -- a source x target decode matrix, at matched readout capacity

      h6[t+1]  =  h6[t]  +  p-  +  r          p- = FM_T(h6[<=t]),  r = Delta - p-
                            ^ forecast  ^ residual

  SOURCES (all d = n_embd, all at the same report positions, all standardised)
    r          the residual              Delta - FM_T(h<=t)      [instrument-swept]
    delta      the raw update            h6[t+1] - h6[t]
    h_next     the raw state             h6[t+1]
    p_minus    the forecast alone        FM_T(h<=t)              [instrument-swept]
    rev_pair   the explicit two-forecast revision p+ - p-, from the matched FM1/FM2 pair
               SPEC.md Gate C specifies: identical architecture, capacity, data,
               optimiser, seed and step budget; FM2 receives x_{t+1}'s embedding in a
               dedicated slot, FM1 a learned MASK token in the same slot. ONLY THE SLOT'S
               CONTENTS DIFFER. This is the arity_torque idiom.        [instrument-swept]
    r_mart     BEYOND THE PRE-REGISTRATION, and the reason the crux is readable:
               the EXACT innovation h6[t+1] - E_{a ~ p_M(.|x<=t)}[h6[t+1] | x_{t+1} = a].
               This is the eps1 -> 0 limit of r -- the residual a forecaster with zero
               compression error would leave, computed by v-way counterfactual
               substitution rather than learned. It makes "is the FM an SNR device"
               a question with a known ceiling instead of an open-ended comparison, and
               it is the object whose martingale null is EXACT (see below).
    pos        one-hot position -- the floor. Everything on the RHM is position-coupled;
               a source that does not beat this has decoded the calendar, not the state.
    r_shuffled r permuted within position -- the content-destroying, position-preserving
               null the parent used. Must read ~0.
    |.|        the scalar norms of r / delta / rev_pair. The design doc's STANDING
               ENTROPY NEGATIVE CONTROL: on the language sibling residual magnitude is
               output entropy at R^2 0.901 and revision at 0.0001.

  TARGETS (exact, model-free, from the oracle)
    B_pxs[D]   THE PRIMARY. Graded belief revision B_t with POSITION x EXACT SURPRISAL
               partialled out nonparametrically -- B minus its mean within an
               (atom-aware surprisal bin x position) stratum, stratum means fit on train
               rows only. Both controls are mandatory and for different reasons.
               Surprisal, because it is the incumbent explanation of everything on this
               axis. Position, because on the RHM constituent boundaries sit at fixed
               offsets, so a target residualised on surprisal alone is still largely a
               CALENDAR -- and the decode matrix would then rank sources by how well
               they encode position, which is not the question. (Measured, in the wiring
               smoke: one-hot position out-decoded every content source on a
               surprisal-only B.) Under position x surprisal the `pos` floor goes to ~0
               by construction and what survives is sequence-specific.
               Training the probe on the residualised target rather than on raw B is
               equally load-bearing: R^2(B ~ bp) is 0.07-0.22, so a SMALL probe trained
               on raw B spends its capacity on the surprisal part -- and the whole crux
               is a statement about small probes.
    B_bp[D]    the surprisal-only residualisation, kept as the secondary. The gap between
               B_bp and B_pxs is itself a readout: how much of a source's apparent
               revision decode is the calendar.
    AE_pxs[D]  aleatoric_fraction's per-position A/(A+E) of the state update -- exact by
               the law of total variance over the v possible arriving tokens with BP
               weights -- residualised the same way. `AE_raw` keeps one level unresidualised
               so the absolute scale stays visible. Positions with a single legal token
               are dropped (both variances are 0 there), as in the parent module.
    bp_sur     exact BP surprisal, and H_tot, the exact predictive entropy. Raw, because
               position structure is the point here. The controls the magnitude sources
               are supposed to load on.
    syn/dis    Gate B's constructed contrast, scored on the B probes' own output:
               synonym positions (B == 0 exactly, ~40% by DGP construction, still ~1 nat
               of surprisal) vs disambiguating positions (B above the median of the
               positive part), under position x exact-surprisal strata.

  READOUT CAPACITY is the new sweep axis and the one the crux is defined on:
  linear (exact ridge, lambda picked on a held-in split) / MLP-16 / MLP-64 / MLP-256.

THE PRE-REGISTERED CRUX:  r  vs  delta,  at matched readout capacity.
  The FM's epistemic claim as an organ is CONCENTRATION: subtracting the forecast should
  cancel the predictable carried-state part and make B-content more accessible from r
  than from delta -- i.e. a higher decode at SMALL probe capacity, converging as the
  probe grows. If r dominates delta and is FM-capacity-invariant, the temporal FM is an
  SNR device for revision and downstream uses should be fed the residual. If r ~ delta
  everywhere, the FM is epistemically inert as a signal-former, and its remaining
  defensible role -- materialising p- at the right TIME for in-loop consumption -- is a
  claim about control topology, to be tested in sparse-feedback settings and not by more
  measurement.

GATE C, folded in for free (SPEC.md, pre-registered since 2026-08-07, never run)
  1. Capacity invariance / the eps-control. The B-decode from r must be FLAT across the
     four-point instrument sweep. "If it moves with capacity we are measuring eps2 - eps1,
     not revision." Here it is checked directly AND read off exactly, because r_mart
     supplies the eps1 = 0 arm: r = r_mart + (p-_mart - p-_learned) is an identity, so
     ||p-_learned - p-_mart|| IS eps1, measured rather than bounded.
  2. The martingale calibration. Idea doc section 3: beliefs are a martingale, so the
     Bayes-optimal p- is the identity in belief space. In activation space the exact
     analogue is p-_mart = E_{a ~ p_M}[h6[t+1] | a], under which E[r_mart | x<=t] = 0
     EXACTLY when x_{t+1} is drawn from p_M. So:
         self-sampled continuations -> drift ratio 1.0 by construction (the null is
                                       exact, not empirical: it validates the estimator)
         corpus text                -> drift = sum_a (p*(a) - p_M(a)) h[t+1|a], the
                                       model's calibration error projected onto the
                                       state-update map. A miscalibration readout.
     The learned r gets the same 2x2: it is trained by MSE on corpus, so corpus drift
     ~1 by construction and self-sampled drift is the informative cell.
  3. The compression term, reported not hidden: ||p-_learned - p-_mart|| / ||p-_mart||
     and cos(p-_learned, p-_mart) per instrument capacity.

GUARDS -- all inherited from the parent and mandatory
  * The junk-residual trap: every r-derived number is swept over instrument-FM capacity
    and carries `ens_cos` (independent fresh FMs leaving the SAME residual) and hierarchy
    eta^2. Axis-specific reading, unchanged from the parent: on the temporal arm a HIGH
    ens_cos is EXPECTED and is not by itself evidence of a computational gap, because the
    aleatoric component is input-determined by construction.
  * Matched heads (fwd_n_head fixed at M's 8).
  * Permute inputs, never labels (r_shuffled).
  * Standardise every source. Without it "concentration" is a conditioning artifact:
    delta and h_next have different norms than r, and a fixed-step optimiser reads that
    as capacity. Ridge at the linear rung is closed-form for the same reason.
  * The position floor. On the RHM everything is position-coupled.

WHAT THIS DELIBERATELY DOES NOT MEASURE
  Privacy (settled by the parent), use (every readout is a trained probe), and the closed
  loop. Nothing here is a claim about the signal being consumed; it is a claim about what
  is in it and about whether subtracting the forecast helps get it out.

Run:
  # wiring smoke (minutes; numbers meaningless by construction)
  modal run -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics --smoke

  # the pre-registered primary (resumes the frozen M the parent and the depth battery share)
  modal run --detach -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics \
      --conditions "ntp_aux" --tag ol

  # the Gates A/B substrate, where the charge was originally measured
  modal run --detach -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics \
      --conditions "cr_base" --tag crbase
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
app = modal.App("rhm-temporal-epistemics", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ======================================================================
# Readout statistics. Everything oracle-side is imported from gates_ab.
# ======================================================================

def _r2_np(y, yhat):
    """Out-of-sample R^2 against the test set's own mean. Can go negative, and is
    allowed to -- a probe that fails should be visibly worse than the mean, not
    silently clipped."""
    import numpy as np
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _stratum_residualise(y, strata, tr_idx):
    """y minus E[y | stratum], stratum means estimated on TRAIN rows only.

    This is the nonparametric partialling-out that defines B_pxs (strata =
    position x exact surprisal) and B_bp (strata = exact surprisal alone).
    A linear control is not enough: bp_surprisal is heavily atomic (-log p piles up on
    0, ln2, ln3, ln4 ...) and its relationship to B is a step function across those
    atoms, which a linear fit leaves most of. `_strata`'s atom-aware binning is the same
    estimator Gate B needed for exactly this reason.

    Strata unseen in train fall back to the global train mean, so no test row is
    residualised by information from its own label.
    """
    import numpy as np
    y = np.asarray(y, float)
    st = np.asarray(strata)
    K = int(st.max()) + 1
    tr_mask = np.zeros(y.size, bool)
    tr_mask[tr_idx] = True
    cnt = np.bincount(st[tr_mask], minlength=K).astype(float)
    tot = np.bincount(st[tr_mask], weights=y[tr_mask], minlength=K)
    gm = float(y[tr_mask].mean())
    mu = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), gm)
    return y - mu[st]


def _standardise(X, tr_idx):
    """Per-dimension z-score with statistics from TRAIN rows only. Matched readout
    capacity is meaningless without it: the sources differ in norm by an order of
    magnitude (||h_next|| >> ||delta|| ~ ||r||) and a fixed-step optimiser converts
    that into an apparent capacity difference."""
    import numpy as np
    mu = X[tr_idx].mean(0, keepdims=True)
    sd = X[tr_idx].std(0, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-6)


def _ridge(Xtr, ytr, Xte, lams=(1e-4, 1e-2, 1.0)):
    """Closed-form ridge with lambda chosen on a held-in 90/10 split of the train rows.

    The linear rung is the cell the concentration claim lives on (`higher decode at
    SMALL probe capacity`), so it is solved exactly rather than by SGD -- optimiser
    noise at the small end is precisely the confound that would fake a concentration
    curve. lambda scales with n so the grid means the same thing at every row count.

    Both Gram matrices are formed once and every lambda is a 256x256 solve, so the cost
    is two matmuls per cell rather than two per lambda.
    """
    import numpy as np
    n, d = Xtr.shape
    cut = max(1, int(0.9 * n))
    Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]
    if Xb.shape[0] < 10:
        Xa, ya, Xb, yb = Xtr, ytr, Xtr, ytr
    I = np.eye(d)
    Ga, ba, ma = Xa.T @ Xa, None, ya.mean()
    ba = Xa.T @ (ya - ma)
    Gf, mf = Xtr.T @ Xtr, ytr.mean()
    bf = Xtr.T @ (ytr - mf)
    best, best_lam = None, lams[0]
    for lam in lams:
        w = np.linalg.solve(Ga + lam * Xa.shape[0] * I, ba)
        sc = -float(((yb - ma - Xb @ w) ** 2).mean())
        if best is None or sc > best:
            best, best_lam = sc, lam
    w = np.linalg.solve(Gf + best_lam * n * I, bf)
    return Xte @ w + mf


# ======================================================================
# Main experiment
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=86400, memory=65536, cpu=8.0)
def temporal_epistemics(
    # --- DGP / model: identical to the parent's m4 regime, so the frozen M is shared ---
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    shallow_block: str = "post_block0", deep_block: str = "post_block6",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # --- wake training (verbatim from the parent; in practice this RESUMES) ---
    conditions: str = "ntp_aux",
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    lam_aux: float = 1.0, lam_local: float = 1.0, ug_hidden: int = 64,
    pool_size: int = 200000, data_seed: int = 7,
    cr_base_steps: int = 12000,
    # --- report set / oracle: the SAME rows the parent measured ---
    n_report_sequences: int = 12000, report_seed: int = 999,
    n_oracle: int = 2000, oracle_ds: str = "2,3,4", oracle_chunk: int = 200,
    n_bp_bins: int = 20, probe_train_frac: float = 0.7,
    # --- instruments ---
    fresh_fm_steps: int = 3000,
    inst_caps_str: str = "16:1.0,4:0.25,64:2.0,128:4.0", ens_n: int = 3,
    diag_seqs: int = 2000,
    # --- readouts ---
    probe_caps: str = "0,16,64,256", probe_caps_reduced: str = "0,64",
    probe_steps: int = 3000, probe_lr: float = 1e-3, probe_bs: int = 4096,
    # --- the martingale arm ---
    cf_chunk: int = 16, n_mart: int = 512, mart_prefix: int = 8,
    eval_interval: int = 1000, seed: int = 42, tag: str = "", resume: bool = True,
    smoke: bool = False,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from a2a_forward.forward_model import TransformerForwardModel
    # Imported, never reimplemented -- "same harness" is meant literally.
    from rhm.confabulation.rhm_confabulation import _generate_with_traces, _ensemble_cos
    from rhm.confabulation.temporal.temporal_confabulation import (
        _eta2_at_positions, _cross)
    from rhm.conditional_revision.gates_ab import (
        _strata, _stratified_auc, _partial_r2, _r2)
    from rhm.conditional_revision import oracle as ORC

    if smoke:
        n_steps, fresh_fm_steps = 300, 200
        n_report_sequences, pool_size = 1000, 20000
        n_oracle, oracle_ds, oracle_chunk = 40, "3,4", 20
        inst_caps_str, ens_n = "16:1.0,4:0.25", 2
        probe_caps, probe_caps_reduced = "0,16", "0"
        probe_steps, probe_bs, eval_interval = 150, 1024, 100
        n_mart, cf_chunk, diag_seqs = 32, 8, 100

    device = "cuda"
    L, T = depth, s ** depth
    NP = T - 2                       # report positions p = 1 .. T-2, signal index t = p-1
    key = tb_key(v, s, L, m)
    cib = int(shallow_block.replace("post_block", ""))
    cond_list = [c.strip() for c in conditions.split(",")]
    Ds = [int(z) for z in oracle_ds.split(",") if z.strip() != ""]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    inst_caps = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in inst_caps_str.split(",")]
    pcaps = [int(z) for z in probe_caps.split(",")]
    pcaps_red = [int(z) for z in probe_caps_reduced.split(",")]
    n_main_params = sum(p.numel() for p in GPT(v, T, n_layer, n_head, n_embd).parameters())
    # the parent's (and the depth battery's) checkpoint dir, on purpose
    ckpt_dir = f"{DATA_DIR}/rhm_confabulation/{key}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = lambda c: f"{ckpt_dir}/{c.replace('@', '_')}_s{n_steps}_seed{seed}.pt"
    orc_dir = f"{DATA_DIR}/rhm_confabulation/{key}/temporal"        # shared with the parent
    out_dir = f"{orc_dir}/epistemics"
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*78}\nTEMPORAL FM EPISTEMICS  {key}  {n_layer}L/{n_head}H/{n_embd}D"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  the question: does the temporal residual CARRY revision, and does the FM")
    print(f"                CONCENTRATE it? (privacy is settled -- see ../README.md)")
    print(f"  crux: r vs delta, B_pxs decode at matched readout capacity {pcaps}")
    print(f"  report positions p in [1, {T-2}]  ({NP} of {T})   conditions={cond_list}")
    print(f"  instrument sweep={inst_caps} (ens_n={ens_n})   oracle D={Ds}\n{'='*78}",
          flush=True)

    # ---------------- training pool (verbatim from the parent) ----------------
    pool_seqs, pool_lf, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))
    pool_anc = [torch.from_numpy(pool_lf[e].astype(np.int64)) for e in range(L)]
    anc_idx = [torch.arange(T) // (s ** (L - e)) for e in range(L)]
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - e)) == 0])
               for e in range(L)]
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]
    corpus = pool_x.reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)
    del pool_seqs, pool_lf

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

    class SlotFM(nn.Module):
        """SPEC.md Gate C's matched pair, in one class so the match is structural.

        The forecaster reads h6[<=t] plus a per-position token slot. FM2 gets the token
        that ARRIVES at t+1 in the slot at t; FM1 gets a learned MASK id in the same
        slot. Identical architecture, identical parameter count, identical init seed,
        identical data order, identical optimiser -- only the slot's contents differ.

        Causality is preserved for the forecast at t: the slot at t' <= t holds
        x_{t'+1}, so position t sees exactly {h_0..h_t, x_1..x_{t+1}}. That is
        `h<=t plus the arriving token` and nothing more.
        """

        def __init__(self, d_head, mlp_mult):
            super().__init__()
            self.core = make_fm(d_head, mlp_mult)
            self.slot = nn.Embedding(v + 1, n_embd)     # index v == MASK
            nn.init.normal_(self.slot.weight, std=0.02)

        def forward(self, h, tok):
            return self.core(h + self.slot(tok))

    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    # ---------------- report set: the parent's exact rows ----------------
    # Generated identically (same rules, same report_seed, same count) and sliced
    # identically, so the cached oracle .npz on the volume is bit-compatible and the two
    # experiments measure the same positions of the same sequences.
    rep_seqs, rep_lf, _ = _generate_with_traces(rules, n_report_sequences, report_seed)
    n_str = int(0.8 * n_report_sequences)
    n_orc = min(n_oracle, n_report_sequences - n_str)
    epi_seqs = rep_seqs[n_str:n_str + n_orc]
    epi_lf = [lf[n_str:n_str + n_orc] for lf in rep_lf]
    epi_x = torch.from_numpy(epi_seqs.astype(np.int64)).to(device)
    del rep_seqs, rep_lf

    n_mart = min(n_mart, n_orc)
    n_ptr = int(probe_train_frac * n_orc)
    rows = lambda a, b: (np.arange(a, b)[:, None] * NP + np.arange(NP)[None, :]).reshape(-1)
    ptr, pte = rows(0, n_ptr), rows(n_ptr, n_orc)
    pos_col = np.tile(np.arange(1, T - 1), n_orc)             # absolute report position
    print(f"\n  probe rows: {ptr.size} train / {pte.size} test "
          f"({n_ptr}/{n_orc - n_ptr} sequences x {NP} positions)", flush=True)

    # ================== the exact oracle (model-independent; cached) ==================
    # Two caches. The first is the parent's, byte-for-byte -- if the parent has run in
    # this regime the BP pass is skipped entirely. The second holds the two next-token
    # posteriors that weight the A/E split; `revision_and_entropy` already computes and
    # discards them, so they cost nothing but storage.
    ostem = f"rs{rule_seed}_seed{report_seed}_n{n_orc}_D{'-'.join(str(d) for d in Ds)}"
    orc_path = f"{orc_dir}/oracle_{ostem}.npz"
    post_path = f"{out_dir}/oracle_posteriors_{ostem}.npz"
    if os.path.exists(orc_path) and os.path.exists(post_path):
        z, zp = np.load(orc_path), np.load(post_path)
        orc = {"B_joint": {D: z[f"B_joint_{D}"] for D in Ds},
               "surprisal": z["surprisal"], "H_tot": z["H_tot"],
               "self_check_passed": bool(z["self_check_passed"])}
        leaf_post = zp["leaf_post"]
        irr_post = {D: zp[f"irr_post_{D}"] for D in Ds}
        print(f"  ORACLE: loaded cached {orc_path} + posteriors", flush=True)
    else:
        print(f"\n  ORACLE: exact BP on {n_orc} sequences, D={Ds} (+ leaf posteriors)",
              flush=True)
        raw = ORC.revision_and_entropy(rules, epi_seqs, epi_lf, Ds=Ds, chunk=oracle_chunk,
                                       verbose=True, return_leaf_posteriors=True)
        chk = ORC.self_check(raw)
        print(f"    instrument self-check: {'PASSED' if chk['passed'] else 'FAILED'}",
              flush=True)
        for D in Ds:
            r_ = chk["per_D"][f"D{D}"]
            print(f"      D{D} ({r_['d_name']}): E[B] {r_['mean_B_joint']:.6f}  "
                  f"H_tot-H_irr {r_['mean_H_tot_minus_H_irr']:.6f}  "
                  f"rel {r_['rel_err']:.2e}")
        orc = {"B_joint": {D: raw["B_joint"][D] for D in Ds},
               "surprisal": raw["surprisal"], "H_tot": raw["H_tot"],
               "self_check_passed": bool(chk["passed"])}
        leaf_post, irr_post = raw["leaf_post"], raw["irr_post"]
        if not os.path.exists(orc_path):
            np.savez_compressed(
                orc_path, surprisal=orc["surprisal"], H_tot=orc["H_tot"],
                self_check_passed=np.array(orc["self_check_passed"]),
                **{f"B_joint_{D}": orc["B_joint"][D] for D in Ds},
                **{f"B_chain_{D}": raw["B_chain"][D] for D in Ds})
        np.savez_compressed(post_path, leaf_post=leaf_post,
                            **{f"irr_post_{D}": irr_post[D] for D in Ds})
        volume.commit()
        print(f"    cached -> {orc_path} + {post_path}", flush=True)
        del raw

    # oracle signal axis is t = 0..T-2; our positions are t = 0..T-3 == the first NP cols
    cut = lambda z: z[:, :NP].reshape(-1)
    bp_sur = cut(orc["surprisal"])
    H_tot = cut(orc["H_tot"])
    B_of = {D: cut(orc["B_joint"][D]) for D in Ds}
    P_leaf = leaf_post[:, :NP]                                  # (n_orc, NP, v)
    Q_irr = {D: irr_post[D][:, :NP] for D in Ds}
    del leaf_post, irr_post
    bp_bins = _strata(bp_sur, n_bp_bins)
    strata_pxbp = _cross(pos_col, bp_bins)
    print(f"  oracle rows = {bp_sur.size}   mean B: "
          + "  ".join(f"d{L-D}={B_of[D].mean():.4f}" for D in Ds)
          + "   frac B==0: "
          + "  ".join(f"d{L-D}={(B_of[D] < 1e-9).mean():.3f}" for D in Ds)
          + f"   R2(B~bp): "
          + "  ".join(f"d{L-D}={_r2(B_of[D], bp_sur):.3f}" for D in Ds), flush=True)

    results = {"config": {
        "conditions": cond_list, "n_steps": n_steps, "seed": seed,
        "n_oracle": n_orc, "oracle_Ds": Ds, "report_positions": [1, T - 2],
        "probe_train_frac": probe_train_frac, "probe_caps": pcaps,
        "probe_caps_reduced": pcaps_red, "inst_caps": inst_caps_str, "ens_n": ens_n,
        "n_mart": n_mart, "mart_prefix": mart_prefix,
        "oracle_self_check_passed": orc["self_check_passed"], "smoke": smoke}}

    # ==================================================================
    for cond in cond_list:
        use_cr = (cond == "cr_base")
        use_aux, use_loop = ("aux" in cond), ("cl" in cond)
        print(f"\n{'='*70}\n  CONDITION: {cond}  "
              f"(aux={use_aux}, loop={use_loop}, cr_base={use_cr})\n{'='*70}", flush=True)

        # ============ Phase 1: the frozen M (shared with the parent) ============
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        if use_cr:
            cr_path = (f"{DATA_DIR}/{key}/conditional_revision/"
                       f"base_{n_layer}L{n_head}H{n_embd}D_steps{cr_base_steps}_"
                       f"seed{seed}.pt")
            if not os.path.exists(cr_path):
                raise FileNotFoundError(f"no conditional_revision base at {cr_path}")
            model.load_state_dict(torch.load(cr_path, map_location=device)["model"])
            print(f"  LOADED conditional_revision base <- {cr_path} (no training)")
            resumed = True
        else:
            model.load_state_dict(init_state)
            resumed = resume and os.path.exists(ckpt_path(cond))
            if resumed:
                model.load_state_dict(torch.load(ckpt_path(cond), map_location=device))
                print(f"  RESUMED wake model <- {ckpt_path(cond)}")

        def eval_ntp(mdl):
            mdl.eval()
            gen = torch.Generator().manual_seed(report_seed + 5)
            tot = 0.0
            with torch.no_grad():
                for _ in range(10):
                    x_, y_ = get_ntp_batch(gen)
                    tot += mdl(x_, y_)[1].item()
            return tot / 10

        if not resumed:
            # Verbatim from the parent, so a cold start lands on the same frozen M the
            # published rows were measured on.
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
                opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=fwd_lr,
                                            weight_decay=0.01)
            opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)
            train_gen = torch.Generator().manual_seed(seed + 1)
            aux_gen = torch.Generator().manual_seed(seed + 2)
            for step in range(n_steps):
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
                        cerebellar_input_block=cib,
                        cerebellar_inject_block=inject_after_block)
                else:
                    logits, _, inter = model(x, return_intermediates=True)
                loss = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
                ntp = loss
                if use_aux:
                    xa, labels = get_aux_batch(aux_gen)
                    _, _, inter_a = model(xa, return_intermediates=True)
                    loss = loss + lam_aux * aux_loss(inter_a, labels, aux_heads)
                if use_loop:
                    fwd_pred, tgt_acts = cache["pred"], inter[deep_block]
                    r_ = tgt_acts - fwd_pred.detach()
                    loss = loss + lam_local * (cache["gw"].detach().mean() * r_ ** 2).mean()
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
                    print(f"    step {step:6d}: ntp={ntp.item():.4f} "
                          f"val={eval_ntp(model):.4f}", flush=True)
            torch.save(model.state_dict(), ckpt_path(cond))
            volume.commit()
            print(f"  wake done, checkpoint saved -> {ckpt_path(cond)}", flush=True)

        model.eval()
        for p_ in model.parameters():
            p_.requires_grad = False
        val = eval_ntp(model)
        print(f"  val = {val:.4f}   (parent reference: ntp_aux 1.5447, cr_base 1.5675)",
              flush=True)

        # ============ Phase 2: frozen state on the oracle rows ============
        c_h6, c_lg = [], []
        with torch.no_grad():
            for i in range(0, len(epi_x), 128):
                lg, _, vi = model(epi_x[i:i + 128], return_intermediates=True)
                c_h6.append(vi[deep_block].cpu()); c_lg.append(lg.cpu())
        h6 = torch.cat(c_h6)                                   # (n_orc, T, d)
        logit = torch.cat(c_lg)
        del c_h6, c_lg
        delta = h6[:, 1:] - h6[:, :-1]                          # (n_orc, T-1, d)
        flat_t = lambda z: z[:, :NP].reshape(n_orc * NP, -1).contiguous()

        # ---- the counterfactual pass: h6[t+1] for every arriving token ----
        # This one pass supplies three of the design doc's items at once: the A/E
        # target, the exact martingale forecast (the eps1 = 0 source), and the drift
        # readout. Substituting x_p only corrupts positions > p, and we read position p
        # alone, so cropping the prefix at p is exact -- asserted against the plain
        # forward pass, the same identity check aleatoric_fraction keeps.
        def counterfactual_pass(seq_t, lgt, h6_ref, want_ae):
            """-> (mart_next (n,NP,d), ae {D: (n,NP)}, ae_valid (n,NP), ident_err)"""
            n_ = seq_t.shape[0]
            mart = torch.zeros(n_, NP, n_embd)
            ae = {D: np.zeros((n_, NP)) for D in Ds} if want_ae else None
            valid = np.zeros((n_, NP), bool) if want_ae else None
            ident = 0.0
            arv = torch.arange(v, device=device)
            for c0 in range(0, n_, cf_chunk):
                c1 = min(n_, c0 + cf_chunk)
                Bc = c1 - c0
                arB = torch.arange(Bc, device=device)
                bx = seq_t[c0:c1].to(device)
                pm = F.softmax(lgt[c0:c1, :NP].to(device), dim=-1)      # (B,NP,v) model
                if want_ae:
                    wp = torch.from_numpy(P_leaf[c0:c1]).float().to(device)
                    wq = torch.stack([torch.from_numpy(Q_irr[D][c0:c1]).float().to(device)
                                      for D in Ds], 1)                  # (B,nD,NP,v)
                for t_ in range(NP):
                    p = t_ + 1
                    xt = bx[:, :p + 1].unsqueeze(1).repeat(1, v, 1)
                    xt[:, :, p] = arv[None, :]
                    with torch.no_grad():
                        _, _, it_ = model(xt.reshape(Bc * v, p + 1),
                                          return_intermediates=True)
                        H = it_[deep_block][:, -1, :].view(Bc, v, n_embd).float()
                    ident = max(ident, float(
                        (H[arB, bx[:, p]] - h6_ref[c0:c1, p].to(device)).abs().max()))
                    mart[c0:c1, t_] = torch.einsum('ba,bac->bc', pm[:, t_], H).cpu()
                    if want_ae:
                        w = wp[:, t_]                                    # (B,v)
                        mu_p = torch.einsum('ba,bac->bc', w, H)
                        tot = (w * ((H - mu_p[:, None]) ** 2).sum(-1)).sum(-1)
                        WQ = wq[:, :, t_]                                # (B,nD,v)
                        mu_q = torch.einsum('bda,bac->bdc', WQ, H)
                        alv = (WQ * ((H[:, None] - mu_q[:, :, None]) ** 2).sum(-1)).sum(-1)
                        ok = tot > 1e-8
                        valid[c0:c1, t_] = ok.cpu().numpy()
                        for j, D in enumerate(Ds):
                            ae[D][c0:c1, t_] = (alv[:, j] / tot.clamp(min=1e-8)).cpu().numpy()
                if want_ae and (c1 % (cf_chunk * 20) == 0 or c1 == n_):
                    print(f"      counterfactual: {c1}/{n_}  (substitution-vs-plain "
                          f"max err {ident:.2e})", flush=True)
            return mart, ae, valid, ident

        print(f"\n  --- counterfactual pass: h6[p] for all {v} arriving tokens ---",
              flush=True)
        mart_next, AE, AE_valid, ident_err = counterfactual_pass(
            epi_x.cpu(), logit, h6, want_ae=True)
        assert ident_err < 5e-3, f"substitution does not reproduce the forward pass: {ident_err}"
        r_mart = h6[:, 1:1 + NP] - mart_next                    # the exact innovation
        p_mart = mart_next - h6[:, :NP]                         # p-_mart, update form
        for D in Ds:
            sel = AE_valid
            print(f"      A/(A+E) d{L-D}: mean {AE[D][sel].mean():.4f} "
                  f"(aleatoric_fraction published d{L-D} in 0.66-0.98 over D)", flush=True)
        print(f"      valid cells {AE_valid.mean():.3f} "
              f"(the rest have a single legal token; both variances are 0)", flush=True)

        # ---- self-sampled continuations, for the martingale null ----
        print(f"\n  --- self-sampled continuations ({n_mart} seqs, prefix {mart_prefix}) "
              f"---", flush=True)
        gsamp = torch.Generator(device=device).manual_seed(seed + 77)
        xs = epi_x[:n_mart, :mart_prefix].clone()
        with torch.no_grad():
            for p in range(mart_prefix, T):
                lg = model(xs)[0]
                nxt = torch.multinomial(F.softmax(lg[:, -1], -1), 1, generator=gsamp)
                xs = torch.cat([xs, nxt], 1)
            c_hs, c_ls = [], []
            for i in range(0, n_mart, 128):
                lg, _, vi = model(xs[i:i + 128], return_intermediates=True)
                c_hs.append(vi[deep_block].cpu()); c_ls.append(lg.cpu())
        h6_s, logit_s = torch.cat(c_hs), torch.cat(c_ls)
        delta_s = h6_s[:, 1:] - h6_s[:, :-1]
        mart_s, _, _, ident_s = counterfactual_pass(xs.cpu(), logit_s, h6_s, want_ae=False)
        r_mart_s = h6_s[:, 1:1 + NP] - mart_s
        assert ident_s < 5e-3

        def drift(r, t_from):
            """||mean_i r|| / sqrt(E||r||^2 / n), pooled over positions t >= t_from.

            Under the martingale null (E[r | prefix] = 0, rows independent) the numerator
            is the SE of the mean, so this reads 1.0. Above 1 is systematic drift. The
            statistic is dimensionless, so the corpus and self-sampled cells and the
            exact and learned residuals are all on one scale."""
            import numpy as np
            rr = r[:, t_from:].double()
            n_ = rr.shape[0]
            num = (rr.mean(0) ** 2).sum(-1)                     # (NP-t_from,)
            den = (rr ** 2).sum(-1).mean(0) / n_
            return float(torch.sqrt(num / den.clamp(min=1e-30)).mean())

        mart_cal = {
            "r_mart_selfsampled": drift(r_mart_s, mart_prefix),
            "r_mart_corpus": drift(r_mart[:n_mart], mart_prefix),
            "ident_err_corpus": ident_err, "ident_err_selfsampled": ident_s}
        print(f"      drift ratio (1.0 == martingale holds): exact r_mart  "
              f"self-sampled {mart_cal['r_mart_selfsampled']:.2f}  "
              f"corpus {mart_cal['r_mart_corpus']:.2f}", flush=True)

        # ============ Phase 3: the fixed (FM-independent) sources ============
        gsh = torch.Generator().manual_seed(seed + 13)
        sh_rows = torch.randperm(n_orc * NP, generator=gsh)
        # position-preserving shuffle: permute sequences within each column
        perm_pos = np.concatenate(
            [np.random.default_rng(seed + 991 + j).permutation(n_orc) * NP + j
             for j in range(NP)])
        restore = np.concatenate([np.arange(n_orc) * NP + j for j in range(NP)])
        pos_shuffle = np.empty(n_orc * NP, dtype=np.int64)
        pos_shuffle[restore] = perm_pos

        onehot_pos = np.zeros((n_orc * NP, NP), dtype=np.float32)
        onehot_pos[np.arange(n_orc * NP), np.tile(np.arange(NP), n_orc)] = 1.0

        fixed_sources = {
            "delta": flat_t(delta).numpy(),
            "h_next": flat_t(h6[:, 1:]).numpy(),
            "r_mart": flat_t(r_mart).numpy(),
            "p_mart": flat_t(p_mart).numpy(),
            "pos": onehot_pos,
        }

        # ============ targets ============
        # The primary form of every content target is residualised on
        # POSITION x EXACT SURPRISAL -- the parent's own control, and not optional.
        # On the RHM everything is position-coupled (constituent boundaries are at fixed
        # offsets), so a target residualised on surprisal ALONE is still largely a
        # calendar, and the decode matrix would rank sources by how well they encode
        # position. The smoke shows exactly that: one-hot `pos` out-decodes every content
        # source on a surprisal-only B. Under position x surprisal the floor goes to ~0
        # by construction and what is left is sequence-specific revision content.
        # `B_bp_*` keeps the surprisal-only form as the secondary, so the gap between the
        # two families is itself the readout of how much of the decode is calendar.
        masks = {"all": np.ones(bp_sur.size, bool), "ae": AE_valid.reshape(-1)}
        tgt = {}
        for D in Ds:
            Bd, dn = B_of[D], f"d{L-D}"
            tgt[f"B_pxs_{dn}"] = (_stratum_residualise(Bd, strata_pxbp, ptr), "all")
            tgt[f"B_bp_{dn}"] = (_stratum_residualise(Bd, bp_bins, ptr), "all")
            tgt[f"AE_pxs_{dn}"] = (_stratum_residualise(AE[D].reshape(-1), strata_pxbp,
                                                        ptr[masks["ae"][ptr]]), "ae")
        tgt[f"AE_raw_d{L-Ds[-1]}"] = (AE[Ds[-1]].reshape(-1), "ae")
        tgt["bp_sur"] = (bp_sur, "all")
        tgt["H_tot"] = (H_tot, "all")

        # How much of B is LEFT after each residualisation, on the test rows. Without
        # this a null on B_pxs is ambiguous between "no source carries it" and "there
        # was nothing left to carry" -- the two readings have opposite implications for
        # the crux, so the headroom is reported next to every decode.
        headroom = {}
        for D in Ds:
            dn = f"d{L-D}"
            v0 = float(np.var(B_of[D][pte]))
            headroom[dn] = {
                "var_frac_left_B_pxs": float(np.var(tgt[f"B_pxs_{dn}"][0][pte]) / v0),
                "var_frac_left_B_bp": float(np.var(tgt[f"B_bp_{dn}"][0][pte]) / v0),
                "n_strata_pxbp": int(np.unique(strata_pxbp).size),
                "rows_per_stratum": float(ptr.size / np.unique(strata_pxbp[ptr]).size)}
        print("  headroom after residualisation (frac of Var(B) left on test rows): "
              + "  ".join(f"{dn}: pxs={h['var_frac_left_B_pxs']:.3f} "
                          f"bp={h['var_frac_left_B_bp']:.3f}"
                          for dn, h in headroom.items())
              + f"   [{headroom[list(headroom)[0]]['n_strata_pxbp']} strata, "
                f"{headroom[list(headroom)[0]]['rows_per_stratum']:.0f} train rows each]",
              flush=True)
        # Gate B's constructed contrast, defined per level on the TEST rows
        syn_dis = {}
        for D in Ds:
            Bd = B_of[D]
            syn = Bd < 1e-9
            posB = Bd[~syn]
            if posB.size == 0 or syn.sum() == 0:
                continue
            dis = Bd > np.median(posB)
            syn_dis[f"d{L-D}"] = (syn | dis, dis)

        # ---- the probe ----
        def decode_cell(X, sname, hidden, store):
            """All targets for one (source, readout capacity).

            Grouped by validity mask so the standardisation, the GPU residency and (at
            the linear rung) the Gram matrices are paid once per mask rather than once
            per target.
            """
            cell = {}
            for mkey, mval in masks.items():
                tr_ = ptr[mval[ptr]]
                te_ = pte[mval[pte]]
                if tr_.size < 50 or te_.size < 50:
                    continue
                mine = [t_ for t_, (_, mk) in tgt.items() if mk == mkey]
                if not mine:
                    continue
                Xs = _standardise(X, tr_)
                if hidden == 0:
                    Xtr64, Xte64 = Xs[tr_].astype(np.float64), Xs[te_].astype(np.float64)
                else:
                    # 124K x 256 float32 == 127MB; resident on the GPU it removes a
                    # per-step host->device copy that otherwise dominates the fit
                    Xg = torch.from_numpy(Xs).float().to(device)
                    tr_t = torch.from_numpy(tr_).to(device)
                    te_t = torch.from_numpy(te_).to(device)
                for tname in mine:
                    y = tgt[tname][0]
                    ym, ys = float(y[tr_].mean()), float(y[tr_].std()) + 1e-12
                    if hidden == 0:
                        yhat = _ridge(Xtr64, (y[tr_] - ym) / ys, Xte64) * ys + ym
                    else:
                        torch.manual_seed(seed + 5000 + hidden)
                        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                                            nn.Linear(hidden, 1)).to(device)
                        opt = torch.optim.AdamW(net.parameters(), lr=probe_lr,
                                                weight_decay=1e-4)
                        yg = torch.from_numpy(
                            ((y - ym) / ys).astype(np.float32)).to(device)
                        g = torch.Generator(device=device).manual_seed(seed + 5001 + hidden)
                        for _ in range(probe_steps):
                            si = tr_t[torch.randint(len(tr_t), (min(probe_bs, len(tr_t)),),
                                                    generator=g, device=device)]
                            F.mse_loss(net(Xg[si]).squeeze(-1), yg[si]).backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                            opt.step(); opt.zero_grad()
                        net.eval()
                        with torch.no_grad():
                            yhat = torch.cat(
                                [net(Xg[te_t[i:i + 32768]]).squeeze(-1)
                                 for i in range(0, len(te_), 32768)]).cpu().numpy()
                        yhat = yhat.astype(np.float64) * ys + ym
                        del net, yg
                    rec = {"r2": _r2_np(y[te_], yhat), "n_test": int(te_.size)}
                    if tname.startswith("B_"):
                        # the same probe output, scored as Gate B's constructed contrast
                        # under position x exact-surprisal strata
                        dn = tname.split("_")[-1]
                        if dn in syn_dis:
                            keep_all, dis_all = syn_dis[dn]
                            k = keep_all[te_]
                            nd_ = dis_all[te_][k].sum()
                            if k.sum() > 50 and 0 < nd_ < k.sum():
                                rec["auc_syn_dis_pos_x_bp"] = _stratified_auc(
                                    yhat[k], dis_all[te_][k], strata_pxbp[te_][k])[0]
                        rec["partial_r2_given_bp"] = _partial_r2(y[te_], yhat,
                                                                 bp_sur[te_])
                    cell[tname] = rec
                if hidden != 0:
                    del Xg, tr_t, te_t
                    torch.cuda.empty_cache()
            store[f"h{hidden}"] = cell
            shw = lambda pre: "  ".join(f"{t_[len(pre):]}={cell[t_]['r2']:+.3f}"
                                        for t_ in cell if t_.startswith(pre))
            print(f"    [{sname:12s} h{hidden:<4d}] B_pxs: {shw('B_pxs_')}  | "
                  f"B_bp: {shw('B_bp_')}  | AE_pxs: {shw('AE_pxs_')}  | "
                  f"bp={cell.get('bp_sur', {}).get('r2', float('nan')):+.3f}", flush=True)
            return cell

        # ============ Phase 4: instrument sweep ============
        def train_temporal_fm(fm_seed, d_head, mlp_mult):
            """One Gate-0 temporal FM, trained exactly as the parent trains its
            instruments: own init, own data stream, update parametrisation, frozen M.
            Independent seeds across the ensemble is what makes `ens_cos` mean
            `input-determined` rather than `same-batches`, so the twins are NOT trained
            in a shared loop even though it would be cheaper."""
            torch.manual_seed(fm_seed)
            f = make_fm(d_head, mlp_mult)
            o = torch.optim.AdamW(f.parameters(), lr=fwd_lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(fresh_fm_steps):
                f.train()
                xb, _ = get_ntp_batch(gen)
                with torch.no_grad():
                    _, _, vi = model(xb, return_intermediates=True)
                    hh = vi[deep_block]
                    dlt = hh[:, 1:, :] - hh[:, :-1, :]
                # update parametrisation (conditional_revision Gate 0's primary): the FM
                # is a residual stream, so reading `fm(h)` biases the forecast toward h
                # itself -- a bad init when the target is a small update.
                F.mse_loss((f(hh) - hh)[:, :-1, :], dlt).backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
                o.step(); o.zero_grad()
            f.eval()
            for p_ in f.parameters():
                p_.requires_grad = False
            return f

        def train_pair(fm_seed, d_head, mlp_mult):
            """SPEC.md Gate C's matched pair. Same init seed, same optimiser, same step
            budget and -- because they are stepped inside ONE loop on ONE batch -- the
            same data in the same order. The only difference in the entire procedure is
            what sits in the token slot."""
            # `.to(device)` consumes no RNG, so the two land on bit-identical inits
            torch.manual_seed(fm_seed)
            fm1 = SlotFM(d_head, mlp_mult).to(device)
            torch.manual_seed(fm_seed)
            fm2 = SlotFM(d_head, mlp_mult).to(device)
            assert torch.equal(fm1.slot.weight, fm2.slot.weight), "pair inits diverged"
            o1 = torch.optim.AdamW(fm1.parameters(), lr=fwd_lr, weight_decay=0.01)
            o2 = torch.optim.AdamW(fm2.parameters(), lr=fwd_lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(fresh_fm_steps):
                fm1.train(); fm2.train()
                xb, _ = get_ntp_batch(gen)
                with torch.no_grad():
                    _, _, vi = model(xb, return_intermediates=True)
                    hh = vi[deep_block]
                    dlt = hh[:, 1:, :] - hh[:, :-1, :]
                mask_tok = torch.full_like(xb, v)
                arr_tok = torch.cat([xb[:, 1:], torch.full_like(xb[:, :1], v)], 1)
                F.mse_loss((fm1(hh, mask_tok) - hh)[:, :-1, :], dlt).backward()
                torch.nn.utils.clip_grad_norm_(fm1.parameters(), 1.0)
                o1.step(); o1.zero_grad()
                F.mse_loss((fm2(hh, arr_tok) - hh)[:, :-1, :], dlt).backward()
                torch.nn.utils.clip_grad_norm_(fm2.parameters(), 1.0)
                o2.step(); o2.zero_grad()
            for f in (fm1, fm2):
                f.eval()
                for p_ in f.parameters():
                    p_.requires_grad = False
            return fm1, fm2

        def fm_update(fmx, src, n_seq=None, tok=None):
            """The forecast UPDATE fm(h) - h, batched. `tok` feeds the slot pair."""
            lim = len(src) if n_seq is None else min(n_seq, len(src))
            out = []
            with torch.no_grad():
                for i in range(0, lim, 128):
                    sb = src[i:min(i + 128, lim)].to(device)
                    o_ = (fmx(sb) if tok is None
                          else fmx(sb, tok[i:min(i + 128, lim)].to(device)))
                    out.append((o_ - sb).cpu())
            return torch.cat(out)

        mask_all = torch.full((n_orc, T), v, dtype=torch.long)
        arr_all = torch.cat([epi_x.cpu()[:, 1:], torch.full((n_orc, 1), v)], 1)
        mask_s = torch.full((n_mart, T), v, dtype=torch.long)

        by_cap = {}
        for ci, (dh, mm) in enumerate(inst_caps):
            is_default = (ci == 0)
            cap_tag = f"h{dh}m{mm:g}"
            # same seed base and recipe as the parent's instruments; the residual is not
            # bit-identical to the parent's (a different draw from the init stream), and
            # `ens_cos` is exactly the statement that this does not matter.
            ens = [train_temporal_fm(seed + 911 + 37 * j, dh, mm)
                   for j in range(max(1, ens_n))]
            fm1, fm2 = train_pair(seed + 4441, dh, mm)
            fm_t = ens[0]
            fm_params = sum(p.numel() for p in fm_t.parameters())
            print(f"\n  --- instrument FM {cap_tag}: {fm_params/1e3:.0f}K params "
                  f"({100*fm_params/n_main_params:.1f}% of M) ---", flush=True)

            p_minus = fm_update(fm_t, h6)[:, :NP]                    # (n,NP,d)
            r_t = delta[:, :NP] - p_minus
            pm1 = fm_update(fm1, h6, tok=mask_all)[:, :NP]
            pm2 = fm_update(fm2, h6, tok=arr_all)[:, :NP]
            rev_pair = pm2 - pm1

            # ---- guards ----
            cos_t = float(F.cosine_similarity(p_minus, delta[:, :NP], dim=-1).mean())
            nd = min(diag_seqs, n_orc)
            ens_t = (_ensemble_cos([(delta[:nd, :NP]
                                     - fm_update(f, h6, nd)[:, :NP]).reshape(-1, n_embd)
                                    for f in ens]) if len(ens) > 1 else float("nan"))
            eta_t = _eta2_at_positions(r_t[:nd].contiguous(),
                                       [lf[:nd] for lf in epi_lf], s, L,
                                       np.arange(1, 1 + NP))
            # ---- Gate C.3: the compression term, measured not bounded ----
            eps1 = p_minus - p_mart
            comp = {"cos_pminus_pmart": float(
                        F.cosine_similarity(p_minus, p_mart, dim=-1).mean()),
                    "rel_eps1": float(eps1.norm(dim=-1).mean()
                                      / p_mart.norm(dim=-1).mean()),
                    "norm_r": float(r_t.norm(dim=-1).mean()),
                    "norm_r_mart": float(r_mart.norm(dim=-1).mean()),
                    "cos_r_rmart": float(
                        F.cosine_similarity(r_t, r_mart, dim=-1).mean())}
            # the learned residual's own martingale cell (corpus is ~1 by training)
            p_minus_s = fm_update(fm_t, h6_s)[:, :NP]
            mart_cal_cap = {
                "r_learned_corpus": drift(r_t[:n_mart], mart_prefix),
                "r_learned_selfsampled": drift(
                    delta_s[:, :NP] - p_minus_s, mart_prefix)}
            print(f"      cos(p-,delta)={cos_t:.4f}  ens_cos={ens_t:.3f}  "
                  + " ".join(f"{k_}eta2={u_:.3f}" for k_, u_ in eta_t.items()))
            print(f"      Gate C.3  cos(p-_learned, p-_mart)={comp['cos_pminus_pmart']:.3f}"
                  f"  rel eps1={comp['rel_eps1']:.3f}  |r|/|r_mart|="
                  f"{comp['norm_r']/max(comp['norm_r_mart'],1e-9):.3f}"
                  f"  cos(r,r_mart)={comp['cos_r_rmart']:.3f}")
            print(f"      Gate C.2  drift, learned r: corpus "
                  f"{mart_cal_cap['r_learned_corpus']:.2f}  self-sampled "
                  f"{mart_cal_cap['r_learned_selfsampled']:.2f}", flush=True)

            sources = {"r": flat_t(r_t).numpy(),
                       "p_minus": flat_t(p_minus).numpy(),
                       "rev_pair": flat_t(rev_pair).numpy()}
            if is_default:
                sources.update(fixed_sources)
                sources["r_shuffled"] = sources["r"][pos_shuffle]
                for nm, arr in (("|r|", flat_t(r_t)), ("|delta|", flat_t(delta[:, :NP])),
                                ("|rev_pair|", flat_t(rev_pair)),
                                ("|r_mart|", flat_t(r_mart))):
                    sources[nm] = arr.norm(dim=-1, keepdim=True).numpy()

            dec = {}
            for sname, X in sources.items():
                caps_here = (pcaps if (is_default and not sname.startswith("|"))
                             else pcaps_red)
                dec[sname] = {}
                for hid in caps_here:
                    decode_cell(X, sname, hid, dec[sname])

            by_cap[cap_tag] = {
                "fm_params": fm_params, "pct_of_model": 100 * fm_params / n_main_params,
                "fwd_cosine": cos_t, "ens_cos": ens_t, "hierarchy_eta2": eta_t,
                "compression": comp, "martingale": mart_cal_cap, "decode": dec}
            del ens, fm1, fm2, p_minus, r_t, pm1, pm2, rev_pair, sources, p_minus_s
            torch.cuda.empty_cache()

        results[cond] = {"val": val, "martingale_exact": mart_cal,
                         "ae_mean": {f"d{L-D}": float(AE[D][AE_valid].mean()) for D in Ds},
                         "ae_valid_frac": float(AE_valid.mean()),
                         "by_capacity": by_cap}
        with open(f"{out_dir}/{(tag + '_') if tag else ''}results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        volume.commit()
        del model, h6, logit, delta, mart_next, r_mart, p_mart, h6_s, logit_s
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    fn = f"{out_dir}/{(tag + '_') if tag else ''}results.json"
    with open(fn, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n{'='*78}\nSUMMARY")
    print("  THE CRUX: r vs delta, B_pxs decode at matched readout capacity.")
    print("    r >> delta at small capacity, converging as the probe grows")
    print("        => the FM CONCENTRATES revision; feed downstream uses the residual.")
    print("    r ~ delta everywhere")
    print("        => the FM is epistemically inert as a signal-former; its remaining")
    print("           defensible role is TIMING, which is a control-topology claim.")
    print("  r_mart is the eps1 = 0 ceiling; pos and r_shuffled are the floors.")
    print("  READ NOTHING WITHOUT ens_cos / eta2 -- and note that on this axis a HIGH")
    print("  ens_cos is expected (the aleatoric part is input-determined by construction).")
    for cond, rr in results.items():
        if cond == "config":
            continue
        print(f"\n  {cond}   val={rr['val']:.4f}   "
              f"exact-martingale drift: self-sampled "
              f"{rr['martingale_exact']['r_mart_selfsampled']:.2f} / corpus "
              f"{rr['martingale_exact']['r_mart_corpus']:.2f}")
        for cap_tag, c in rr["by_capacity"].items():
            print(f"\n    --- {cap_tag}  ({c['fm_params']/1e3:.0f}K, "
                  f"{c['pct_of_model']:.1f}% of M)  cos={c['fwd_cosine']:.3f} "
                  f"ens_cos={c['ens_cos']:.3f}  rel_eps1="
                  f"{c['compression']['rel_eps1']:.3f} ---")
            names = [t for t in rr["by_capacity"][cap_tag]["decode"]]
            hdr = sorted({h for n_ in names for h in c["decode"][n_]},
                         key=lambda z: int(z[1:]))
            tset = [k for k in (list(c["decode"][names[0]].values())[0])]
            for t_ in tset:
                print(f"      {t_:14s} " + "  ".join(f"{h:>8s}" for h in hdr))
                for n_ in names:
                    cells = []
                    for h in hdr:
                        rec = c["decode"][n_].get(h, {}).get(t_)
                        cells.append("       ." if rec is None else f"{rec['r2']:>8.3f}")
                    print(f"        {n_:12s} " + "  ".join(cells))
    print(f"\n  saved -> {fn}\n{'='*78}")
    return results
