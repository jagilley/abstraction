"""The Temporal Confabulation Test: is the composite's privileged access epistemically CHARGED?

Design doc: ../../../../ideas/temporal_confabulation_test.md  (pre-registered before this run)
Parent:     ../README.md          -- the depth battery, whose machinery this forks verbatim
Lineage:    ../../conditional_revision/README.md  -- the temporal FM idiom, the exact oracle,
            and the atom-aware matching discipline (Gate B) this file imports rather than
            reimplements.

THE QUESTION
  The depth battery established that the composite (M + FM) has PRIVILEGED ACCESS to the
  FM residual: reports about `r` beat every capacity-matched third-party observer by
  +0.09..+0.11 while reports about behaviour/entropy/world do not (the observer wins
  there). But the *content* of the depth residual is computational -- "I lacked capacity"
  -- because FM(a_i) and a_j are deterministic functions of the SAME information set. So
  it is privileged access to an epistemically INERT quantity.

  Shift the FM's target by one token and the conditioning gap holds the world's next
  input. The same decomposition now carries what the token just taught the model:

      depth    :  a6[t]     =  FM_d(a0[<=t])[t]                     +  r_depth[t]
      temporal :  h6[t+1]   =  h6[t] + FM_T(h6[<=t])[t]             +  r_temp[t]
                  ^actual       ^ the self-theory's forecast of the    ^ what the token
                   next state     next state, available BEFORE x_{t+1}   added

  Conjecture: depth FM -> privileged access to "how I compute", charged with nothing;
  temporal FM -> privileged access to "what I just learned", private precisely because
  its I/O-visible shadow (surprisal) is exactly what it is not.

WHY THE OBSERVER LADDER IS THE SURPRISAL CONTROL, FOR FREE
  `O_io` receives M's tokens AND M's full output distribution at every position, so at
  report position p it can compute the surprisal of the arriving token x_p EXACTLY
  (it holds logits[p-1] and x_p). Therefore any first-person advantage on a
  temporal-residual report is, by construction, the component of revision that surprisal
  cannot account for. Gate B needed atom-aware strata to pin surprisal; here the ladder
  pins it architecturally. Both are run anyway -- the explicit matching is what ties the
  advantage to the ORACLE's revision B_t.

WHAT THIS SCRIPT MEASURES
  Report position p in [1, T-2] for EVERY target and EVERY observer, so the two arms and
  all controls are position-matched to the row. The temporal residual r_temp[p-1] is only
  physically present once x_p has arrived, which is why the report is emitted at p and
  not at p-1 -- "the difference is on a wire inside the system, at the right time".

  Targets
    TEMP-IMPL  8-way spherical-k-means cluster of the DIRECTION of r_temp   [implementation]
    TEMP-MAG   quartile of ||r_temp||   -- the PRE-REGISTERED NEGATIVE CONTROL. On the
               language sibling, residual magnitude is output entropy at R^2 0.901 and
               revision at 0.0001; it should show no advantage beyond O_io.
    IMPL       the depth arm, same machinery, same positions -- the direct
               inert-vs-charged comparison
    DEPTH-MAG  its magnitude twin, so the magnitude control is symmetric
    BEHAV / ENT / WORLD   the battery's standing controls

  Test 1  observer ladder   O_input / O_io / O_act(post_block0) / O_h6(post_block6),
                            capacity-swept, plus the half-data budget control.
                            O_h6 is a REAL ceiling for the temporal arm and only for it:
                            h6[<=p] contains h6[p] and everything FM_T conditions on, so
                            that observer is FM_T + a subtraction + a probe. The depth
                            battery had no validated ceiling (its O_act came in BELOW
                            O_io); this arm supplies one.
  Test 2  channel ablation  shuffle_r / shuffle_p / zero_r / zero_p, substituted into the
                            state the report head reads. Weak, inherited, corroborative.
  Test 3  matched-KL steer  residual span vs self-theory span at equal behavioural effect,
                            with BEHAV-flip as the built-in control.
  Test 4  confabulator      a head trained AND evaluated on the self-theory's state only.
                            NOTE, stated up front: the temporal confabulator differs in
                            KIND from the depth one. The depth confabulator sees the same
                            time-slice input filtered through the self-theory; the
                            temporal confabulator cannot see x_p at all, because the
                            conditioning gap IS that token. Its margin is therefore not
                            comparable across arms -- only the ADVANTAGE is.
  Test 5  THE CHARGE VALIDATION -- new, and the point of the experiment.
                            The exact BP oracle gives per-position belief revision B_t on
                            the same sequences. Ask not only "is access privileged" but
                            "is the privileged part the revision part": regress the
                            per-position advantage adv = 1[self correct] - 1[best O_io
                            correct] on oracle B with surprisal matched, and run Gate B's
                            synonym-vs-disambiguating family contrast on adv under
                            position x exact-surprisal strata. The depth arm is the
                            built-in null: privileged but inert => no B structure beyond
                            surprisal.

GUARDS (all inherited from the depth battery and mandatory)
  * the junk-residual trap: an over-capacity FM leaves architectural-mismatch noise that
    M can report and no observer can predict -- the target signature, produced by nothing.
    Every IMPL number is swept over instrument-FM capacity and carries `ens_cos`
    (independent fresh FMs leaving the SAME residual => input-determined) and hierarchy
    eta^2. Read no IMPL number without them.
    Axis-specific reading: on the temporal arm a HIGH ens_cos is expected and is not by
    itself evidence of a computational gap, because the aleatoric component is
    input-determined by construction (every FM misses x_{t+1} identically). ens_cos here
    rules OUT idiosyncratic noise; it does not rule IN a capacity gap.
  * matched heads (n_head fixed at M's 8) -- RESIDUAL_RANK found a head-count mismatch
    dominates the residual.
  * the observer data-budget control at half data.
  * permute inputs, never labels (the shuffle_* variants).
  * raw `self` accuracies are NOT comparable across capacity rows (each has its own
    k-means clustering and its own baseline). Only margins and advantages are.

Run:
  # wiring smoke (minutes; numbers meaningless -- deliberately in the artifact regime)
  modal run -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test --smoke

  # the pre-registered primary: open loop first
  modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
      --conditions "ntp_aux" --tag ol

  # follow-up arms
  modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
      --conditions "ntp_aux_cl" --tag cl
  # the conditional_revision substrate itself (plain NTP, no aux head, no loop) -- the
  # exact frozen base Gates A/B ran on. Loads the cached checkpoint, trains nothing.
  modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
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
app = modal.App("rhm-temporal-confabulation", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ======================================================================
# Charge-validation statistics (everything else is imported, not rewritten)
# ======================================================================

def _dense(strata):
    """Relabel arbitrary stratum ids to 0..K-1 so bincount can be used."""
    import numpy as np
    return np.unique(strata, return_inverse=True)[1]


def _cross(a, b):
    """Exact pair encoding of two stratum labellings. Arithmetic like `a * K + b`
    silently aliases across `a` whenever `b` has more atoms than K -- a documented
    gates_ab gotcha."""
    import numpy as np
    return np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)[1]


def _eta2_at_positions(res, rep_lf, s, L, positions):
    """Fraction of residual variance explained by the ground-truth ancestor feature at
    each hierarchy level, for a residual indexed by an explicit position map.

    This is `rhm_confabulation._eta2_by_level` with the position map exposed, and it
    reduces to that reference exactly at `positions = arange(T)` (asserted in the
    smoke). The temporal residual r_t materialises at position t+1 -- the report reads
    it there and so must this diagnostic -- so it is scored against the ancestors of
    positions 1..T-1, not 0..T-2.
    """
    import numpy as np
    import torch
    out = {}
    tot = float(((res - res.mean(dim=(0, 1), keepdim=True)) ** 2).sum())
    for ell in range(L):
        lab = torch.from_numpy(
            rep_lf[ell][:, positions // (s ** (L - ell))].astype(np.int64))
        flat_r = res.reshape(-1, res.shape[-1])
        flat_l = lab.reshape(-1)
        gmean = flat_r.mean(0, keepdim=True)
        between = 0.0
        for g in torch.unique(flat_l):
            sel = flat_l == g
            between += float(int(sel.sum()) * ((flat_r[sel].mean(0) - gmean) ** 2).sum())
        out[f"d{L - ell}"] = between / tot if tot > 0 else 0.0
    return out


def _strat_meandiff(y, pos_mask, strata):
    """Stratified mean difference  mean(y | positive) - mean(y | negative), pooled
    across strata with the same n1*n0/(n1+n0) weights an inverse-variance pooling
    would use. Vectorised with bincount so a permutation null is affordable.

    `y` here is the per-position ADVANTAGE (in {-1,0,1}), which is mostly ties -- a
    rank statistic on it is low-powered, so the mean difference is the primary and the
    stratified AUC is reported alongside.
    """
    import numpy as np
    st = _dense(strata)
    K = st.max() + 1
    p = pos_mask.astype(np.float64)
    n1 = np.bincount(st, weights=p, minlength=K)
    n0 = np.bincount(st, minlength=K) - n1
    s1 = np.bincount(st, weights=y * p, minlength=K)
    s0 = np.bincount(st, weights=y * (1.0 - p), minlength=K)
    ok = (n1 > 0) & (n0 > 0)
    d = np.zeros(K)
    d[ok] = s1[ok] / n1[ok] - s0[ok] / n0[ok]
    w = np.zeros(K)
    w[ok] = n1[ok] * n0[ok] / (n1[ok] + n0[ok])
    tot = w.sum()
    return float((d * w).sum() / tot) if tot > 0 else float("nan")


def _perm_within(pos_mask, strata, rng):
    """Permute family membership WITHIN each stratum -- the null that keeps position
    and surprisal fixed and destroys only which family a row belongs to."""
    import numpy as np
    st = _dense(strata)
    order = np.lexsort((rng.random(st.size), st))     # rows grouped by stratum, shuffled
    canon = np.lexsort((np.arange(st.size), st))      # rows grouped by stratum, in order
    out = np.empty_like(pos_mask)
    out[canon] = pos_mask[order]
    return out


def _meandiff_z(y, pos_mask, strata, n_perm, seed):
    """Stratified mean difference with a within-stratum permutation null."""
    import numpy as np
    obs = _strat_meandiff(y, pos_mask, strata)
    rng = np.random.default_rng(seed)
    null = np.array([_strat_meandiff(y, _perm_within(pos_mask, strata, rng), strata)
                     for _ in range(n_perm)])
    sd = float(null.std(ddof=1))
    return {"meandiff": obs, "null_mean": float(null.mean()), "null_sd": sd,
            "z": float((obs - null.mean()) / sd) if sd > 1e-12 else float("nan"),
            "n_perm": n_perm}


# ======================================================================
# Main experiment
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=86400, memory=49152, cpu=8.0)
def temporal_confabulation_test(
    # --- DGP / model: identical to the depth battery's m4 regime ---
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    shallow_block: str = "post_block0", deep_block: str = "post_block6",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # --- wake training (verbatim latent_loop recipe, shared checkpoint dir with
    #     the depth battery so the two experiments run on the SAME frozen M) ---
    conditions: str = "ntp_aux",
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    lam_aux: float = 1.0, lam_local: float = 1.0, ug_hidden: int = 64,
    pool_size: int = 200000, data_seed: int = 7,
    cr_base_steps: int = 12000,          # the cached conditional_revision base
    # --- report battery ---
    n_report_sequences: int = 12000, report_seed: int = 999,
    fresh_fm_steps: int = 3000,
    inst_caps_str: str = "16:1.0,4:0.25,64:2.0,128:4.0", ens_n: int = 3,
    diag_seqs: int = 2000,          # subsample for ens_cos / eta2 (distributional)
    impl_k: int = 8, mag_k: int = 4, world_level: int = 3,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    observer_caps: str = "1:64,2:128,4:192,8:256",
    observer_caps_reduced: str = "4:192,8:256",
    obs_steps: int = 3000, obs_lr: float = 3e-4, obs_bs: int = 64,
    # --- Test 3 ---
    steer_n_pc: int = 32, steer_eps: float = 1.0, steer_target_kl: float = 0.01,
    # --- Test 5: the exact oracle ---
    n_oracle: int = 2000, oracle_ds: str = "2,3,4", oracle_chunk: int = 200,
    n_perm: int = 200, n_bp_bins: int = 20,
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
    # Imported, never reimplemented: the depth battery's own helpers (so "matched
    # machinery" is literal) and Gate B's atom-aware matching (so the charge
    # validation uses the same estimator that produced the 0.690).
    from rhm.confabulation.rhm_confabulation import (
        _generate_with_traces, _kmeans_fit, _kmeans_assign, _balance,
        _eta2_by_level, _ensemble_cos, _make_head, _fit_head, _head_acc)
    from rhm.conditional_revision.gates_ab import (
        _strata, _stratified_auc, _auc, _partial_r2, _partial_r2_rank, _r2)
    from rhm.conditional_revision import oracle as ORC

    if smoke:
        n_steps, n_report_sequences, fresh_fm_steps = 300, 400, 200
        head_steps, obs_steps, pool_size = 200, 200, 20000
        observer_caps, observer_caps_reduced, eval_interval = "1:64,2:128", "2:128", 100
        inst_caps_str, ens_n, steer_n_pc = "16:1.0,4:0.25", 2, 8
        diag_seqs = 100                     # < n_report_sequences: exercises truncation
        n_oracle, oracle_ds, n_perm = 60, "3,4", 20

    device = "cuda"
    L, T = depth, s ** depth
    NP = T - 2                       # report positions p = 1 .. T-2 (see module docstring)
    key = tb_key(v, s, L, m)
    cib = int(shallow_block.replace("post_block", ""))
    cond_list = [c.strip() for c in conditions.split(",")]
    caps = [tuple(int(z) for z in c.split(":")) for c in observer_caps.split(",")]
    caps_red = [tuple(int(z) for z in c.split(":")) for c in observer_caps_reduced.split(",")]
    Ds = [int(z) for z in oracle_ds.split(",") if z.strip() != ""]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    report_block = f"post_block{n_layer - 1}"
    inst_caps = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in inst_caps_str.split(",")]
    n_pred_blocks = int(deep_block.replace("post_block", "")) - cib
    n_main_params = sum(p.numel() for p in GPT(v, T, n_layer, n_head, n_embd).parameters())
    n_pred_params = n_pred_blocks * (4 * n_embd * n_embd + 2 * n_embd * 4 * n_embd)
    # The depth battery's checkpoint dir, on purpose: if the wake model is already
    # there, both experiments measure the SAME frozen M and the depth arm here is a
    # controlled re-run of the published one rather than a different model.
    ckpt_dir = f"{DATA_DIR}/rhm_confabulation/{key}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = lambda c: (f"{ckpt_dir}/{c.replace('@', '_')}"
                           f"_s{n_steps}_seed{seed}.pt")
    out_dir = f"{DATA_DIR}/rhm_confabulation/{key}/temporal"
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*78}\nTEMPORAL CONFABULATION TEST  {key}  {n_layer}L/{n_head}H/{n_embd}D"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  depth    FM: {shallow_block}[<=t] -> {deep_block}[t]")
    print(f"  temporal FM: {deep_block}[<=t] -> Delta_t = {deep_block}[t+1] - {deep_block}[t]"
          f"   (update parametrisation: pred = fm(h6) - h6)")
    print(f"  report positions p in [1, {T-2}]  ({NP} of {T})   report block {report_block}")
    print(f"  conditions={cond_list}  K={impl_k}  world=d{L-world_level}  observers={caps}")
    print(f"  instrument sweep={inst_caps} (ens_n={ens_n})   oracle D={Ds} on "
          f"{n_oracle} test sequences\n{'='*78}", flush=True)

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

    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    # ---------------- report/eval set (aligned -> WORLD labels + oracle defined) ------
    rep_seqs, rep_lf, _ = _generate_with_traces(rules, n_report_sequences, report_seed)
    rep_x = torch.from_numpy(rep_seqs.astype(np.int64)).to(device)
    world_np = rep_lf[world_level][:, (np.arange(T) // (s ** (L - world_level)))]
    world_all = torch.from_numpy(world_np.astype(np.int64))                  # (N,T) cpu

    N = n_report_sequences
    n_str = int(0.8 * N)
    s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
    n_orc = min(n_oracle, len(s_te))
    # index maps. flat_(): (N,T,·) tensors aligned to absolute positions -> rows for
    # p in [1, T-2]. flat_t(): (N,T-1,·) tensors aligned to positions 1..T-1 (i.e.
    # signal index j = p-1) -> the SAME rows. Every target, observer and oracle column
    # in this file is indexed by these two and nothing else.
    flat_ = lambda z: z[:, 1:T - 1].reshape(N * NP, -1).contiguous()
    flat_t = lambda z: z[:, :NP].reshape(N * NP, -1).contiguous()
    pos_of = lambda si: (si[:, None] * NP + torch.arange(NP)[None, :]).reshape(-1)
    tr, te = pos_of(s_tr), pos_of(s_te)

    # ================== the exact oracle (model-independent; cached) ==================
    # B_t is a property of (rules, sequence) alone, so it is computed once for all
    # conditions and capacities, on the first n_orc TEST sequences.
    orc_path = (f"{out_dir}/oracle_rs{rule_seed}_seed{report_seed}_n{n_orc}"
                f"_D{'-'.join(str(d) for d in Ds)}.npz")
    if os.path.exists(orc_path):
        z = np.load(orc_path)
        orc = {"B_joint": {D: z[f"B_joint_{D}"] for D in Ds},
               "B_chain": {D: z[f"B_chain_{D}"] for D in Ds},
               "surprisal": z["surprisal"], "H_tot": z["H_tot"],
               "self_check_passed": bool(z["self_check_passed"])}
        print(f"\nORACLE: loaded cached {orc_path}", flush=True)
    else:
        print(f"\nORACLE: exact BP on {n_orc} test sequences, D={Ds} "
              f"(junction-tree cliques; the RHM latent graph is a hypertree)", flush=True)
        o_seqs = rep_seqs[n_str:n_str + n_orc]
        o_lf = [lf[n_str:n_str + n_orc] for lf in rep_lf]
        raw = ORC.revision_and_entropy(rules, o_seqs, o_lf, Ds=Ds,
                                       chunk=oracle_chunk, verbose=True)
        chk = ORC.self_check(raw)
        print(f"  instrument self-check: {'PASSED' if chk['passed'] else 'FAILED'}"
              f"   (E[B_D] == H(x|x_<=t) - H(x|z_<=D,x_<=t))", flush=True)
        for D in Ds:
            r_ = chk["per_D"][f"D{D}"]
            print(f"    D{D} ({r_['d_name']}): E[B] {r_['mean_B_joint']:.6f}  "
                  f"H_tot-H_irr {r_['mean_H_tot_minus_H_irr']:.6f}  "
                  f"|diff| {r_['abs_err']:.2e}  rel {r_['rel_err']:.2e}")
        orc = {"B_joint": {D: raw["B_joint"][D] for D in Ds},
               "B_chain": {D: raw["B_chain"][D] for D in Ds},
               "surprisal": raw["surprisal"], "H_tot": raw["H_tot"],
               "self_check_passed": bool(chk["passed"])}
        np.savez_compressed(
            orc_path, surprisal=orc["surprisal"], H_tot=orc["H_tot"],
            self_check_passed=np.array(orc["self_check_passed"]),
            **{f"B_joint_{D}": orc["B_joint"][D] for D in Ds},
            **{f"B_chain_{D}": orc["B_chain"][D] for D in Ds})
        volume.commit()
        print(f"  cached -> {orc_path}", flush=True)
        del raw
    # oracle signal axis is t = 0..T-2 ("x_{t+1} arrives"); our report position is
    # p = t+1 in [1, T-2], so t = 0..T-3 == the first NP columns.
    bp_sur = orc["surprisal"][:, :NP].reshape(-1)
    B_of = {D: orc["B_joint"][D][:, :NP].reshape(-1) for D in Ds}
    Bch_of = {D: orc["B_chain"][D][:, :NP].reshape(-1) for D in Ds}
    pos_orc = np.tile(np.arange(1, T - 1), n_orc)
    print(f"  oracle rows = {bp_sur.size}   mean B_joint: "
          + "  ".join(f"d{L-D}={B_of[D].mean():.4f}" for D in Ds)
          + "   frac B==0: "
          + "  ".join(f"d{L-D}={(B_of[D] < 1e-9).mean():.3f}" for D in Ds), flush=True)

    # ==================================================================
    # Observer: causal stack over tokens, optionally + a continuous stream
    # ==================================================================
    class Observer(nn.Module):
        """Third-party predictor of a report target. `extra_dim>0` adds a continuous
        per-position stream (M's logits for O_io, post_block0 for O_act, post_block6
        for O_h6). Reads positions 0..T-2 and is scored on 1..T-2, so at report
        position p it holds tokens x_{<=p} AND M's logits at p-1 -- i.e. it can compute
        the arriving token's surprisal exactly. That is the architectural surprisal
        control."""

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

        def forward(self, tok_, extra=None):
            B_, t_ = (tok_.shape if self.tokens else extra.shape[:2])
            x = self.wpe(torch.arange(t_, device=self.wpe.weight.device))[None].expand(B_, t_, -1)
            if self.tokens:
                x = x + self.wte(tok_)
            if self.proj is not None:
                x = x + self.proj(extra)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))[:, 1:]      # score positions 1..T-2

    results = {"config": {
        "conditions": cond_list, "n_steps": n_steps, "seed": seed,
        "n_report_sequences": N, "report_positions": [1, T - 2],
        "inst_caps": inst_caps_str, "ens_n": ens_n, "impl_k": impl_k,
        "observer_caps": observer_caps, "observer_caps_reduced": observer_caps_reduced,
        "n_oracle": n_orc, "oracle_Ds": Ds, "n_perm": n_perm,
        "oracle_self_check_passed": orc["self_check_passed"], "smoke": smoke}}

    for cond in cond_list:
        use_cr = (cond == "cr_base")
        use_aux, use_loop = ("aux" in cond), ("cl" in cond)
        print(f"\n{'='*70}\n  CONDITION: {cond}  "
              f"(aux={use_aux}, loop={use_loop}, cr_base={use_cr})\n{'='*70}", flush=True)

        # ============ Phase 1: the frozen M ============
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        if use_cr:
            # the exact frozen base Gates A/B / aleatoric_fraction / tracking ran on
            cr_path = (f"{DATA_DIR}/{key}/conditional_revision/"
                       f"base_{n_layer}L{n_head}H{n_embd}D_steps{cr_base_steps}_"
                       f"seed{seed}.pt")
            if not os.path.exists(cr_path):
                raise FileNotFoundError(
                    f"no conditional_revision base at {cr_path}; run "
                    f"rhm.conditional_revision.conditional_revision::gate0 first.")
            model.load_state_dict(torch.load(cr_path, map_location=device)["model"])
            print(f"  LOADED conditional_revision base <- {cr_path} (no training)")
            resumed = True
        else:
            model.load_state_dict(init_state)
            resumed = resume and os.path.exists(ckpt_path(cond))
            if resumed:
                model.load_state_dict(torch.load(ckpt_path(cond), map_location=device))
                print(f"  RESUMED wake model <- {ckpt_path(cond)} (shared with the "
                      f"depth battery)")

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
            # save the INSTANT training ends -- a measurement crash once cost a full run
            torch.save(model.state_dict(), ckpt_path(cond))
            volume.commit()
            print(f"  wake done, checkpoint saved -> {ckpt_path(cond)}", flush=True)

        model.eval()
        for p_ in model.parameters():
            p_.requires_grad = False
        val = eval_ntp(model)
        print(f"  val = {val:.4f}   (depth-battery reference: ntp_aux 1.5447, "
              f"ntp_aux_cl 2.3829; conditional_revision base ~1.53)", flush=True)

        # ============ Phase 2: cache the FM-independent state once ============
        c_a0, c_a6, c_logit, c_tok = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(rep_x), 128):
                xb = rep_x[i:i + 128]
                lg, _, vi = model(xb, return_intermediates=True)
                c_a0.append(vi[shallow_block].cpu()); c_a6.append(vi[deep_block].cpu())
                c_logit.append(lg.cpu()); c_tok.append(xb.cpu())
        a0 = torch.cat(c_a0); a6 = torch.cat(c_a6)
        logit = torch.cat(c_logit); tok = torch.cat(c_tok)
        del c_a0, c_a6, c_logit, c_tok

        # ---- FM-independent report targets, on positions 1..T-2 ----
        # BEHAV/ENT read M's output AT p (predicting x_{p+1}); WORLD reads p's ancestor.
        y_behav = (logit[:, 1:T - 1].argmax(-1) == tok[:, 2:T]).long().reshape(-1)
        lp_ = F.log_softmax(flat_(logit), -1)
        ent = -(lp_.exp() * lp_).sum(-1)
        y_ent = torch.bucketize(ent, torch.quantile(ent[tr], torch.tensor([.25, .5, .75])))
        y_world = flat_(world_all.unsqueeze(-1)).reshape(-1)
        fixed_targets = {"BEHAV": (y_behav, 2), "ENT": (y_ent, 4), "WORLD": (y_world, v)}
        del lp_, ent

        # M's own nll of the ARRIVING token x_p -- the matching variable whose
        # I/O-visibility is the whole point of criterion (2).
        nll_all = F.cross_entropy(
            logit[:, :T - 1].reshape(-1, v), tok[:, 1:].reshape(-1),
            reduction="none").reshape(N, T - 1)
        nll_rep = flat_t(nll_all.unsqueeze(-1)).reshape(-1).numpy()      # (N*NP,)

        last_block, ln_f = model.transformer.h[n_layer - 1], model.transformer.ln_f

        def make_report_input(a6_seq):
            """Push a (N,T,d) `a6`-like sequence through M's own last block and flatten
            to the report positions. Attention is intact, so a substituted variant is
            read exactly as M would read it."""
            out = []
            with torch.no_grad():
                for i in range(0, a6_seq.shape[0], 128):
                    out.append(last_block(a6_seq[i:i + 128].to(device)).cpu())
            return flat_(torch.cat(out))

        X_full = make_report_input(a6)                       # FM-independent
        gsh = torch.Generator().manual_seed(seed + 13)
        sh_T = torch.randperm(N * T, generator=gsh)
        sh_t = torch.randperm(N * (T - 1), generator=gsh)
        rs_T = lambda z: z.reshape(N * T, n_embd)[sh_T].reshape(N, T, n_embd)
        rs_t = lambda z: z.reshape(N * (T - 1), n_embd)[sh_t].reshape(N, T - 1, n_embd)
        as_state = lambda z: torch.cat([a6[:, :1], z], dim=1)  # (N,T-1,d)->(N,T,d) @ p>=1

        # ---- observers ----
        def train_observer(yt, ncls, o_layer, o_embd, extra_src, use_tokens,
                           data_frac=1.0):
            """Returns (test accuracy, per-position predictions on s_te)."""
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
            hits, corr, tot_ = [], 0, 0
            with torch.no_grad():
                for i in range(0, len(s_te), 64):
                    si = s_te[i:i + 64]
                    xb = tok[si, :T - 1].to(device)
                    eb = None if extra_src is None else extra_src[si, :T - 1].to(device)
                    p_ = o(xb, eb).argmax(-1).cpu()
                    hits.append(p_ == yt[si])
                    corr += int((p_ == yt[si]).sum()); tot_ += p_.numel()
            del o
            torch.cuda.empty_cache()
            return corr / tot_, torch.cat(hits)               # (n_test, NP) correctness

        def run_ladder(yv, ncls, self_acc, label, full=True, deep_ceiling=False):
            """Test 1. `full=False` runs the reduced ladder at off-default instrument
            capacities -- an explicit, logged economy, not a silent cap."""
            yt = yv.view(N, NP)
            d, hits = {}, {}
            for (o_layer, o_embd) in (caps if full else caps_red):
                cp = f"{o_layer}L{o_embd}D"
                d[f"O_input@{cp}"], _ = train_observer(yt, ncls, o_layer, o_embd, None, True)
                d[f"O_io@{cp}"], hits[f"O_io@{cp}"] = train_observer(
                    yt, ncls, o_layer, o_embd, logit, True)
            ol, oe = caps[-1]
            d[f"O_act@{ol}L{oe}D"], _ = train_observer(yt, ncls, ol, oe, a0, False)
            if deep_ceiling:
                # h6[<=p] contains h6[p] AND everything FM_T conditions on -> a genuine
                # ceiling for the temporal arm (and only for it: the depth residual
                # needs a0, which h6 does not cheaply supply).
                d[f"O_h6@{ol}L{oe}D"], _ = train_observer(yt, ncls, ol, oe, a6, False)
            d[f"O_io@{ol}L{oe}D_half_data"], _ = train_observer(
                yt, ncls, ol, oe, logit, True, data_frac=0.5)
            io_keys = [k_ for k_ in d if k_.startswith("O_io") and "half" not in k_]
            bkey = max(io_keys, key=lambda k_: d[k_])
            # how much of the target is recoverable from the TOKENS ALONE. On the
            # temporal axis this is the number to watch: r_temp depends on x_p, which
            # every observer sees, so a high O_input says the residual is largely
            # token-determined and the interesting quantity is what sits above it.
            d["best_O_input"] = max(u_ for k_, u_ in d.items() if k_.startswith("O_input"))
            print(f"  [observ/{label:16s}] self={self_acc:.3f}  best_O_io={d[bkey]:.3f}"
                  f"  advantage={self_acc - d[bkey]:+.3f}  {'(reduced ladder)' if not full else ''}"
                  f"\n      " + "  ".join(f"{k_}={u_:.3f}" for k_, u_ in d.items()),
                  flush=True)
            return d, d[bkey], hits[bkey]

        def report_row(yv, ncls, Xv, label):
            """Tests 2 + 4, plus the per-position correctness the charge validation needs."""
            h = _fit_head(_make_head(n_embd, ncls, head_hidden, device, seed),
                          Xv["full"], yv, tr, head_steps, head_lr, device)
            row = {"full": _head_acc(h, Xv["full"], yv, te, device)}
            for vn in ("shuffle_r", "shuffle_p", "zero_r", "zero_p"):
                row[f"eval_{vn}"] = _head_acc(h, Xv[vn], yv, te, device)
            hc = _fit_head(_make_head(n_embd, ncls, head_hidden, device, seed),
                           Xv["zero_r"], yv, tr, head_steps, head_lr, device)
            row["confab"] = _head_acc(hc, Xv["zero_r"], yv, te, device)
            row["baseline"] = _balance(yv[te], ncls)
            del hc
            # per-position correctness on the test split, in s_te x position order
            with torch.no_grad():
                pr = torch.cat([h(Xv["full"][te[i:i + 16384]].to(device)).argmax(-1).cpu()
                                for i in range(0, len(te), 16384)])
            corr = (pr == yv[te]).view(len(s_te), NP)
            print(f"  [report/{label:16s}] full={row['full']:.3f} "
                  f"confab={row['confab']:.3f} | eval: shuf_r={row['eval_shuffle_r']:.3f} "
                  f"shuf_p={row['eval_shuffle_p']:.3f} zero_r={row['eval_zero_r']:.3f} "
                  f"| base={row['baseline']:.3f}", flush=True)
            return row, h, corr

        # ============ Test 5: the charge validation ============
        # Rows of the oracle, in the same (sequence-major, position-minor) order the
        # heads and observers report in. s_te[:n_orc] are exactly the sequences the
        # oracle was run on.
        orc_rows = pos_of(torch.arange(n_str, n_str + n_orc)).numpy()
        nll_o = nll_rep[orc_rows]
        bp_bins = _strata(bp_sur, n_bp_bins)
        strata_pxbp = _cross(pos_orc, bp_bins)          # position x exact surprisal
        strata_pxnll = _cross(pos_orc, _strata(nll_o, n_bp_bins))
        rng_pos = np.random.default_rng(seed + 4242)
        # content-destroying, position-preserving null: permute rows WITHIN position,
        # so anything that merely varies with position survives and everything
        # sequence-specific dies. Gate B's `M_shuffled` guard, one object over.
        shuf_within_pos = np.concatenate(
            [rng_pos.permutation(np.arange(n_orc) * NP + j) for j in range(NP)])
        restore = np.concatenate([np.arange(n_orc) * NP + j for j in range(NP)])
        pos_shuffle = np.empty(n_orc * NP, dtype=np.int64)
        pos_shuffle[restore] = shuf_within_pos

        def charge_block(corr_self, corr_obs, label):
            """Is the privileged part the revision part?

            adv[p] = 1[self-report correct] - 1[best O_io correct], per position. Two
            readouts, both with surprisal pinned:
              (a) partial R^2 of adv on the exact oracle B, given exact surprisal;
              (b) Gate B's constructed contrast -- synonym positions (B == 0 exactly,
                  ~49% of positions by DGP construction, still carrying ~1 nat of
                  surprisal) vs disambiguating positions (B above the median of the
                  positive part) -- scored under position x exact-surprisal strata.
            Guards that must read ~0.5 / ~0: exact surprisal held against itself, and
            adv permuted within position.
            """
            sc = corr_self[:n_orc].reshape(-1).numpy().astype(np.float64)
            oc = corr_obs[:n_orc].reshape(-1).numpy().astype(np.float64)
            adv = sc - oc
            adv_ps = adv[pos_shuffle]
            out = {}
            for D in Ds:
                B, Bch = B_of[D], Bch_of[D]
                syn = B < 1e-9
                pos_B = B[~syn]
                if pos_B.size == 0 or syn.sum() == 0:
                    continue
                dis = B > np.median(pos_B)
                keep = syn | dis
                row = {
                    "n_syn": int(syn.sum()), "n_dis": int(dis.sum()),
                    "mean_B": float(B.mean()),
                    # (a) partial R^2, exact surprisal as the control variable
                    "partial_r2_adv_from_B_given_bp": _partial_r2(adv, B, bp_sur),
                    "rank_partial_r2_adv_from_B_given_bp": _partial_r2_rank(adv, B, bp_sur),
                    "partial_r2_adv_from_bp_given_B": _partial_r2(adv, bp_sur, B),
                    "partial_r2_adv_from_Bchain_given_bp": _partial_r2(adv, Bch, bp_sur),
                    "partial_r2_adv_from_B_given_nll": _partial_r2(adv, B, nll_o),
                    "shuffled_partial_r2": _partial_r2(adv_ps, B, bp_sur),
                    # which side of the advantage moves
                    "partial_r2_self_from_B_given_bp": _partial_r2(sc, B, bp_sur),
                    "partial_r2_obs_from_B_given_bp": _partial_r2(oc, B, bp_sur),
                    "r2_B_from_bp": _r2(B, bp_sur),
                    # (b) the constructed contrast
                    "auc_adv_raw": _auc(adv[keep], dis[keep]),
                    "auc_adv_pos_x_bp": _stratified_auc(adv[keep], dis[keep],
                                                        strata_pxbp[keep])[0],
                    "auc_adv_pos_x_nll": _stratified_auc(adv[keep], dis[keep],
                                                         strata_pxnll[keep])[0],
                    "auc_self_pos_x_bp": _stratified_auc(sc[keep], dis[keep],
                                                         strata_pxbp[keep])[0],
                    "auc_obs_pos_x_bp": _stratified_auc(oc[keep], dis[keep],
                                                        strata_pxbp[keep])[0],
                    # guards
                    "guard_auc_bp": _stratified_auc(bp_sur[keep], dis[keep],
                                                    strata_pxbp[keep])[0],
                    "guard_auc_adv_posshuffled": _stratified_auc(
                        adv_ps[keep], dis[keep], strata_pxbp[keep])[0],
                    "meandiff_adv": _meandiff_z(adv[keep], dis[keep],
                                                strata_pxbp[keep], n_perm, seed + D),
                    "meandiff_adv_posshuffled": _meandiff_z(
                        adv_ps[keep], dis[keep], strata_pxbp[keep], n_perm, seed + D),
                }
                out[f"d{L - D}"] = row
                g = row["guard_auc_bp"]
                if abs(g - 0.5) > 0.02:
                    print(f"    WARNING [{label} d{L-D}]: surprisal guard reads "
                          f"{g:.4f}, not ~0.5 -- strata leak, column inflated")
            print(f"  [charge/{label:16s}] "
                  + "  ".join(
                      f"d{L-D}: pR2(adv~B|bp)={out[f'd{L-D}']['partial_r2_adv_from_B_given_bp']:+.4f}"
                      f" (shuf {out[f'd{L-D}']['shuffled_partial_r2']:+.4f})"
                      f" AUC={out[f'd{L-D}']['auc_adv_pos_x_bp']:.3f}"
                      f" z={out[f'd{L-D}']['meandiff_adv']['z']:+.1f}"
                      for D in Ds if f"d{L-D}" in out), flush=True)
            return out

        def train_fms(fm_seed, d_head, mlp_mult):
            """One frozen forward pass trains BOTH arms' instruments, so the depth /
            temporal comparison cannot be a data-order artifact."""
            torch.manual_seed(fm_seed)
            fm_d, fm_t = make_fm(d_head, mlp_mult), make_fm(d_head, mlp_mult)
            od = torch.optim.AdamW(fm_d.parameters(), lr=fwd_lr, weight_decay=0.01)
            ot = torch.optim.AdamW(fm_t.parameters(), lr=fwd_lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(fresh_fm_steps):
                fm_d.train(); fm_t.train()
                xb, _ = get_ntp_batch(gen)
                with torch.no_grad():
                    _, _, vi = model(xb, return_intermediates=True)
                    h0, h6 = vi[shallow_block], vi[deep_block]
                    dlt = h6[:, 1:, :] - h6[:, :-1, :]
                F.mse_loss(fm_d(h0), h6).backward()
                torch.nn.utils.clip_grad_norm_(fm_d.parameters(), 1.0)
                od.step(); od.zero_grad()
                # update parametrisation (conditional_revision Gate 0's primary): the
                # FM is a residual stream, so reading `fm(h6)` biases the forecast
                # toward h6 itself -- a bad init when the target is a small update.
                F.mse_loss((fm_t(h6) - h6)[:, :-1, :], dlt).backward()
                torch.nn.utils.clip_grad_norm_(fm_t.parameters(), 1.0)
                ot.step(); ot.zero_grad()
            for f_ in (fm_d, fm_t):
                f_.eval()
                for p_ in f_.parameters():
                    p_.requires_grad = False
            return fm_d, fm_t

        def fm_apply(fmx, src, n_seq=None, update=False):
            lim = len(src) if n_seq is None else min(n_seq, len(src))
            out = []
            with torch.no_grad():
                for i in range(0, lim, 128):
                    sb = src[i:min(i + 128, lim)].to(device)
                    o_ = fmx(sb)
                    out.append(((o_ - sb) if update else o_).cpu())
            return torch.cat(out)

        # ============ Phase 3: sweep the instrument FM capacity ============
        by_cap = {}
        fixed_done = False
        for ci, (dh, mm) in enumerate(inst_caps):
            is_default = (ci == 0)
            pairs = [train_fms(seed + 911 + 37 * j, dh, mm) for j in range(max(1, ens_n))]
            fm_d0, fm_t0 = pairs[0]
            fm_params = sum(p.numel() for p in fm_d0.parameters())
            cap_tag = f"h{dh}m{mm:g}"
            print(f"\n  --- instrument FM {cap_tag}: {fm_params/1e3:.0f}K params "
                  f"({100*fm_params/n_main_params:.1f}% of M, "
                  f"{100*fm_params/n_pred_params:.1f}% of the {n_pred_blocks} depth-"
                  f"predicted blocks) ---", flush=True)

            # ---- the two decompositions, on the same frozen state ----
            pred_d = fm_apply(fm_d0, a0)                        # (N,T,d)  FM_d(a0)
            res_d = a6 - pred_d                                 # (N,T,d)  r_depth
            delta = a6[:, 1:] - a6[:, :-1]                      # (N,T-1,d)
            pred_t = fm_apply(fm_t0, a6, update=True)[:, :-1]   # (N,T-1,d) FM_T's update
            res_t = delta - pred_t                              # (N,T-1,d) r_temporal
            state_t = a6[:, :-1] + pred_t                       # the self-theory's next state
            # the decomposition is exact: forecast state + residual == the actual state
            assert torch.allclose(state_t[:64] + res_t[:64], a6[:64, 1:], atol=1e-3)

            cos_d = float(F.cosine_similarity(pred_d, a6, dim=-1).mean())
            cos_t = float(F.cosine_similarity(pred_t, delta, dim=-1).mean())
            nd = min(diag_seqs, N)
            if len(pairs) > 1:
                ens_d = _ensemble_cos([(a6[:nd] - fm_apply(p_[0], a0, nd)).reshape(-1, n_embd)
                                       for p_ in pairs])
                ens_t = _ensemble_cos(
                    [(delta[:nd] - fm_apply(p_[1], a6, nd, update=True)[:, :-1]
                      ).reshape(-1, n_embd) for p_ in pairs])
            else:
                ens_d = ens_t = float("nan")
            lf_nd = [lf[:nd] for lf in rep_lf]
            eta_d = _eta2_by_level(res_d[:nd], lf_nd, s, L, T)
            # r_temporal materialises at position t+1, so it is scored against the
            # ancestors of positions 1..T-1 (see `_eta2_at_positions`).
            eta_t = _eta2_at_positions(res_t[:nd], lf_nd, s, L, np.arange(1, T))
            if smoke and is_default:          # the estimator reduces to the reference
                chk_ = _eta2_at_positions(res_d[:nd], lf_nd, s, L, np.arange(T))
                assert max(abs(chk_[k_] - eta_d[k_]) for k_ in eta_d) < 1e-9, \
                    "_eta2_at_positions does not reduce to _eta2_by_level"
            print(f"      depth   : cos={cos_d:.4f} |r|={float(res_d.norm(dim=-1).mean()):.3f}"
                  f"  ens_cos={ens_d:.3f}  "
                  + " ".join(f"{k_}eta2={u_:.3f}" for k_, u_ in eta_d.items()))
            print(f"      temporal: cos={cos_t:.4f} |r|={float(res_t.norm(dim=-1).mean()):.3f}"
                  f"  ens_cos={ens_t:.3f}  "
                  + " ".join(f"{k_}eta2={u_:.3f}" for k_, u_ in eta_t.items()), flush=True)
            del pairs
            torch.cuda.empty_cache()

            # ---- labels (k-means fit on TRAIN rows only) ----
            R_t, R_d = flat_t(res_t), flat_(res_d)
            y_temp = _kmeans_assign(
                R_t.to(device), _kmeans_fit(R_t[tr].to(device), impl_k, seed=seed)).cpu()
            y_dep = _kmeans_assign(
                R_d.to(device), _kmeans_fit(R_d[tr].to(device), impl_k, seed=seed)).cpu()
            qs = torch.tensor([(i + 1) / mag_k for i in range(mag_k - 1)])
            mag_t_v, mag_d_v = R_t.norm(dim=-1), R_d.norm(dim=-1)
            y_tmag = torch.bucketize(mag_t_v, torch.quantile(mag_t_v[tr], qs))
            y_dmag = torch.bucketize(mag_d_v, torch.quantile(mag_d_v[tr], qs))
            del mag_t_v, mag_d_v

            cap_out = {"fm_params": fm_params,
                       "pct_of_model": 100 * fm_params / n_main_params,
                       "pct_of_depth_predicted_blocks": 100 * fm_params / n_pred_params,
                       "depth": {"fwd_cosine": cos_d, "ens_cos": ens_d,
                                 "res_norm": float(res_d.norm(dim=-1).mean()),
                                 "hierarchy_eta2": eta_d},
                       "temporal": {"fwd_cosine": cos_t, "ens_cos": ens_t,
                                    "res_norm": float(res_t.norm(dim=-1).mean()),
                                    "hierarchy_eta2": eta_t},
                       "targets": {}}

            # ---- per-arm battery ----
            for arm in ("temporal", "depth"):
                if arm == "temporal":
                    Xv = {"full": X_full,
                          "zero_r": make_report_input(as_state(state_t)),
                          "shuffle_r": make_report_input(as_state(state_t + rs_t(res_t))),
                          "shuffle_p": make_report_input(as_state(rs_t(state_t) + res_t)),
                          "zero_p": make_report_input(as_state(res_t))}
                    # Two self-theory spans on this axis, because the literal fork of
                    # the depth arm has a scale mismatch the depth arm does not:
                    #   `prediction`        = span of the forecast STATE h6[t]+FM_T -- the
                    #                         literal analogue of the depth arm's FM(a_i),
                    #                         but state-scale against an update-scale r;
                    #   `prediction_update` = span of the forecast UPDATE FM_T alone --
                    #                         same scale as r_temp, so the contrast is not
                    #                         "big directions vs small directions".
                    # Matched KL already equalises behavioural effect; this equalises the
                    # object being compared. Report both ratios.
                    span_r, span_p = R_t, flat_t(state_t)
                    span_pu = flat_t(pred_t)
                    tgts = [("TEMP-IMPL", y_temp, impl_k, True)]
                    if is_default:
                        tgts.append(("TEMP-MAG", y_tmag, mag_k, False))
                else:
                    Xv = {"full": X_full,
                          "zero_r": make_report_input(pred_d),
                          "shuffle_r": make_report_input(pred_d + rs_T(res_d)),
                          "shuffle_p": make_report_input(rs_T(pred_d) + res_d),
                          "zero_p": make_report_input(res_d)}
                    span_r, span_p, span_pu = R_d, flat_(pred_d), None
                    tgts = [("IMPL", y_dep, impl_k, False)]
                    if is_default:
                        tgts.append(("DEPTH-MAG", y_dmag, mag_k, False))

                arm_heads = {}
                for tname, yv, ncls, ceiling in tgts:
                    lab = f"{tname}/{cap_tag}"
                    row, h_, self_hits = report_row(yv, ncls, Xv, lab)
                    obs, best_io, obs_hits = run_ladder(
                        yv, ncls, row["full"], lab, full=is_default,
                        deep_ceiling=(ceiling and is_default))
                    cap_out["targets"][tname] = {
                        "report": row, "observers": obs, "best_O_io": best_io,
                        "advantage": row["full"] - best_io,
                        "margin": row["full"] - row["confab"]}
                    arm_heads[tname] = (h_, self_hits, obs_hits)

                # ---- BEHAV / ENT / WORLD, once, with the temporal arm's ablations ----
                # (their `full` column and observers are arm-independent; only the
                # ablation/confab columns come from the temporal decomposition)
                if arm == "temporal" and is_default and not fixed_done:
                    for tn, (yv_, nc_) in fixed_targets.items():
                        row, h_, self_hits = report_row(yv_, nc_, Xv, tn)
                        obs, best_io, obs_hits = run_ladder(yv_, nc_, row["full"], tn)
                        cap_out["targets"][tn] = {
                            "report": row, "observers": obs, "best_O_io": best_io,
                            "advantage": row["full"] - best_io,
                            "margin": row["full"] - row["confab"]}
                        arm_heads[tn] = (h_, self_hits, obs_hits)
                        if tn == "BEHAV":
                            h_behav = h_
                    fixed_done = True

                # ---- Test 3: matched-KL steering ----
                # Perturb the state the report head reads along PCs of the residual span
                # vs the self-theory's span, rescaling each direction to the SAME KL on
                # M's logits. Matched KL == matched behaviour, so BEHAV-flip is a control
                # that must come out equal across families.
                prim = tgts[0][0]
                h_impl = arm_heads[prim][0]
                steer = {"target_kl": steer_target_kl, "families": {}}
                idx = torch.randperm(len(tr),
                                     generator=torch.Generator().manual_seed(seed + 17))
                sel_ = tr[idx[:min(50000, len(tr))]]

                def top_pcs(M_):
                    sub = M_[sel_].to(device)
                    return torch.linalg.svd(sub - sub.mean(0), full_matrices=False)[2][:steer_n_pc]

                steer_seqs = s_te[:min(512, len(s_te))]
                base_a6 = a6[steer_seqs].to(device)

                def run_state(a6_batch):
                    with torch.no_grad():
                        z = last_block(a6_batch)
                        return (F.log_softmax(model.lm_head(ln_f(z))[:, 1:T - 1], -1),
                                h_impl(z[:, 1:T - 1]).argmax(-1),
                                h_behav(z[:, 1:T - 1]).argmax(-1))

                base_lp, base_impl, base_behav = run_state(base_a6)

                def probe_dir(u, eps):
                    lp2, ri_, rb_ = run_state(base_a6 + eps * u.view(1, 1, -1))
                    return (float(F.kl_div(lp2.reshape(-1, v), base_lp.reshape(-1, v),
                                           log_target=True, reduction="batchmean")),
                            float((ri_ != base_impl).float().mean()),
                            float((rb_ != base_behav).float().mean()))

                fams = [("residual", span_r), ("prediction", span_p)]
                if span_pu is not None:
                    fams.append(("prediction_update", span_pu))
                for fam, span in fams:
                    pcs = top_pcs(span)
                    per = []
                    for k in range(steer_n_pc):
                        u = pcs[k]
                        kl0, _, _ = probe_dir(u, steer_eps)
                        if kl0 <= 1e-9:
                            continue
                        e_k = min(steer_eps * (steer_target_kl / kl0) ** 0.5, 20 * steer_eps)
                        kl1, _, _ = probe_dir(u, e_k)
                        if kl1 > 1e-9:
                            e_k = min(e_k * (steer_target_kl / kl1) ** 0.5, 20 * steer_eps)
                        kl_f, fi, fb = probe_dir(u, e_k)
                        per.append({"pc": k, "eps": e_k, "kl": kl_f,
                                    "impl_flip": fi, "behav_flip": fb})
                    ag = (lambda kk: float(np.mean([p_[kk] for p_ in per]))
                          if per else float("nan"))
                    steer["families"][fam] = {
                        "per_pc": per, "kl": ag("kl"), "impl_flip": ag("impl_flip"),
                        "behav_flip": ag("behav_flip")}
                    print(f"  [steer/{arm[:4]}/{fam:10s}] @KL={ag('kl'):.4f}  "
                          f"{prim}-flip={ag('impl_flip'):.3f}  "
                          f"BEHAV-flip={ag('behav_flip'):.3f}", flush=True)
                rr_ = steer["families"]["residual"]
                ratio = lambda k_: (rr_["impl_flip"] / steer["families"][k_]["impl_flip"]
                                    if steer["families"][k_]["impl_flip"] > 0
                                    else float("inf"))
                steer["impl_flip_ratio"] = ratio("prediction")
                msg = f"residual/prediction = {steer['impl_flip_ratio']:.2f}x"
                if "prediction_update" in steer["families"]:
                    steer["impl_flip_ratio_update"] = ratio("prediction_update")
                    msg += (f"   residual/prediction_update (scale-matched) = "
                            f"{steer['impl_flip_ratio_update']:.2f}x")
                cap_out["targets"][prim]["steering"] = steer
                print(f"  [steer/{arm}] flip ratio at matched behaviour: {msg}", flush=True)

                # ============ Test 5: THE CHARGE VALIDATION ============
                for tname, _, _, _ in tgts:
                    cap_out["targets"][tname]["charge"] = charge_block(
                        arm_heads[tname][1], arm_heads[tname][2], f"{tname}/{cap_tag}")
                if arm == "temporal" and is_default:
                    for tn in ("BEHAV", "ENT", "WORLD"):
                        if tn in arm_heads:
                            cap_out["targets"][tn]["charge"] = charge_block(
                                arm_heads[tn][1], arm_heads[tn][2], tn)

                del Xv, arm_heads
                torch.cuda.empty_cache()

            by_cap[cap_tag] = cap_out
            del pred_d, res_d, pred_t, res_t, state_t, delta, R_t, R_d
            torch.cuda.empty_cache()

        results[cond] = {"val": val, "by_capacity": by_cap}
        with open(f"{out_dir}/{(tag + '_') if tag else ''}results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        volume.commit()
        del model, a0, a6, logit, X_full, nll_all
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    fn = f"{out_dir}/{(tag + '_') if tag else ''}results.json"
    with open(fn, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n{'='*78}\nSUMMARY")
    print("  advantage = self-report - best capacity-matched O_io  (the criterion-(2)")
    print("              quantity; on the temporal axis O_io can compute the arriving")
    print("              token's surprisal exactly, so the advantage is beyond-surprisal")
    print("              BY CONSTRUCTION)")
    print("  margin    = self-report - confabulator. NOT comparable across arms: the")
    print("              temporal confabulator cannot see the arriving token at all.")
    print("  READ NO IMPL NUMBER WITHOUT ens_cos / eta2 -- a saturated FM leaves junk")
    print("  that M can report and no observer can predict, faking the headline.")
    for cond, rr in results.items():
        if cond == "config":
            continue
        print(f"\n  {cond}   val={rr['val']:.4f}")
        print(f"    {'target':12s} {'inst':9s} {'cos':>6s} {'ens':>6s} {'self':>6s} "
              f"{'confab':>7s} {'margin':>7s} {'Oinput':>7s} {'bestOio':>8s} "
              f"{'advant':>7s} {'base':>6s} {'steer':>6s}")
        for cap_tag, c in rr["by_capacity"].items():
            for tname, tt in c["targets"].items():
                arm = ("temporal" if tname.startswith("TEMP") else
                       "depth" if tname in ("IMPL", "DEPTH-MAG") else None)
                cs = f"{c[arm]['fwd_cosine']:6.3f}" if arm else " " * 6
                es = f"{c[arm]['ens_cos']:6.3f}" if arm else " " * 6
                row = tt["report"]
                st = tt.get("steering", {}).get("impl_flip_ratio", float("nan"))
                print(f"    {tname:12s} {cap_tag:9s} {cs} {es} {row['full']:6.3f} "
                      f"{row['confab']:7.3f} {tt['margin']:+7.3f} "
                      f"{tt['observers']['best_O_input']:7.3f} {tt['best_O_io']:8.3f} "
                      f"{tt['advantage']:+7.3f} {row['baseline']:6.3f} {st:6.2f}")
        print(f"\n    CHARGE VALIDATION -- is the privileged part the revision part?")
        print(f"    {'target':12s} {'inst':9s} {'lvl':>4s} {'pR2(adv~B|bp)':>14s} "
              f"{'shuf':>7s} {'pR2(adv~bp|B)':>14s} {'AUC_adv':>8s} {'guard':>7s} "
              f"{'meandiff':>9s} {'z':>7s}")
        for cap_tag, c in rr["by_capacity"].items():
            for tname, tt in c["targets"].items():
                for dn, cr in tt.get("charge", {}).items():
                    print(f"    {tname:12s} {cap_tag:9s} {dn:>4s} "
                          f"{cr['partial_r2_adv_from_B_given_bp']:>14.4f} "
                          f"{cr['shuffled_partial_r2']:>7.4f} "
                          f"{cr['partial_r2_adv_from_bp_given_B']:>14.4f} "
                          f"{cr['auc_adv_pos_x_bp']:>8.4f} {cr['guard_auc_bp']:>7.4f} "
                          f"{cr['meandiff_adv']['meandiff']:>+9.4f} "
                          f"{cr['meandiff_adv']['z']:>+7.2f}")
    print(f"\n  saved -> {fn}\n{'='*78}")
    return results
