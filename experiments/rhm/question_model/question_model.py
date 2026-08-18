"""question_model, Half 1: the three-way decomposition of a token's news, and whether a plain
NTP learner keeps a question model.

    E[ -log P(x_{t+1}|x_{<=t}) ]  =  E[Dm]  +  E[S_D|theta]  +  E[H(x|theta,z_{<=D},x_{<=t})]
      total surprisal                DEMAND    STRUCTURE        ALEATORIC

`revision_not_surprisal` SS4 splits a token's surprisal two ways (structure-news + irreducible
residue). This adds the missing term: news about the generative mixture itself -- what is being
ASKED -- which is the currency `practice/typed_gaps` measured the evaluative re-selection organ
to consume and the dense learner to ignore. See SPEC.md.

Entrypoints (all on the `chromatic` workspace, volume `rhm-scaling-data`):

  selfcheck   -- the instrument gates: weighted BP reduces to conditional_revision/oracle.py at
                 machine precision, brute force with non-uniform weights, the O(T L) filter
                 against BP, drift-off exactness, and the identities.
  calibrate   -- the sigma_s x kappa x K sweep. Sets the world's knobs by measurement.
  train_world -- base (8L/8H/256D) + temporal FM for one world. Caches to the volume.
  decompose   -- the main run: the exact decomposition, the laundering probe, and the relation
                 between the model's own registered news and the three oracle channels.

  cd experiments   # MODAL_PROFILE=chromatic
  modal run -m rhm.question_model.question_model::selfcheck
  modal run --detach -m rhm.question_model.question_model::calibrate --tag cal0
  modal run --detach -m rhm.question_model.question_model::train_world --world drift --tag w0
  modal run --detach -m rhm.question_model.question_model::decompose --tag dec0
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
app = modal.App("rhm-question-model", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def qm_dir(v, s, L, m):
    return f"{DATA_DIR}/{tb_key(v, s, L, m)}/question_model"


def _spec_from_args(q_k, sigma_s, kappa, half, dir_seed, L):
    from rhm.question_model import demand_state as DS
    return DS.make_spec(K=q_k, sigma_s=sigma_s, kappa=kappa, half=half,
                        dir_seed=dir_seed, L=L)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def selfcheck():
    from rhm.question_model import oracle_q as OQ
    from rhm.question_model import qfilter as QF
    print("=" * 74)
    print("question_model instrument gates")
    print("=" * 74)
    print("\nG-1..G-3  weighted BP")
    OQ._test_reduces_to_incumbent()
    OQ._test_brute_force_weighted()
    OQ._test_conditional_identity()
    print("\nG-4..G-6  the exact question filter")
    QF._test_loglik_matches_bp()
    QF._test_drift_off()
    QF._test_demand_identity()
    print("\nALL GATES PASSED", flush=True)
    return {"passed": True}


@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=32768, cpu=8.0)
def calibrate(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
              n: int = 512, seed: int = 3, tag: str = "cal0"):
    from rhm.question_model.calibrate import sweep
    d = qm_dir(v, s, depth, m)
    os.makedirs(d, exist_ok=True)
    p = f"{d}/calibration_{tag}.json"
    cells = sweep(v=v, s=s, L=depth, m=m, rule_seed=rule_seed, n=n, seed=seed, out_path=p)
    volume.commit()
    print(f"\nwrote {p}", flush=True)
    return cells


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #

def _build_pool(rules, spec, dirs, world, pool_size, data_seed, verbose=True):
    """The training pool: `pool_size` aligned blocks in trajectory order, concatenated by
    the caller. Traces are dropped (only the decomposition needs them)."""
    import numpy as np
    from rhm.question_model import demand_state as DS
    t = __import__("time").time()
    seqs, _lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, pool_size, data_seed,
                                            world=world)
    if verbose:
        print(f"  pool[{world}]: {pool_size:,} blocks in {__import__('time').time() - t:.0f}s"
              f"   distinct theta states visited: {len(np.unique(tidx))}", flush=True)
    return seqs, tidx


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=65536)
def train_world(
    world: str = "drift",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    deep_block: str = "post_block6",
    q_k: int = 7, sigma_s: float = 1.0, kappa: float = 0.15, half: float = 2.0,
    dir_seed: int = 0,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    base_steps: int = 12000, fm_steps: int = 12000,
    batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    weight_decay: float = 0.01,
    pool_size: int = 200000, data_seed: int = 7,
    log_interval: int = 1000, seed: int = 42, tag: str = "w0",
):
    """One base model + its temporal FM, trained on one world's corpus.

    Protocol is `conditional_revision.gate0`'s verbatim -- same architecture, same steps,
    same optimiser, same flat-window sampling, same `pred = fm(h6) - h6` readout -- so the
    cached uniform-world base is a legitimate third arm rather than an approximate one.
    """
    import numpy as np
    import time
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    spec = _spec_from_args(q_k, sigma_s, kappa, half, dir_seed, L)
    from rhm.question_model import demand_state as DS
    dirs = DS.make_directions(rules, spec)

    d = qm_dir(v, s, depth, m)
    os.makedirs(d, exist_ok=True)
    wtag = f"{world}_K{spec['K']}_sg{sigma_s:g}_kp{kappa:g}_ds{dir_seed}"
    ckpt_path = f"{d}/base_{wtag}_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt"

    print(f"{'=' * 78}\nQUESTION MODEL -- train_world  {wtag}")
    print(f"  {n_layer}L/{n_head}H/{n_embd}D   T={T}   base_steps={base_steps}  "
          f"fm_steps={fm_steps}")
    print(f"  spec: {spec}")
    print(f"  n_states={DS.n_states(spec)}  H(theta)={DS.theta_entropy(spec):.3f} nats")
    print(f"{'=' * 78}", flush=True)

    pool_seqs, pool_tidx = _build_pool(rules, spec, dirs, world, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)
    del pool_seqs

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    if os.path.exists(ckpt_path):
        print(f"--- loading cached {ckpt_path} ---", flush=True)
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model"])
    else:
        print(f"--- BASE: {base_steps} steps plain NTP on the `{world}` corpus ---",
              flush=True)
        t0 = time.time()
        for step in range(base_steps):
            model.train()
            x, y = get_ntp_batch(gen)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % log_interval == 0 or step == base_steps - 1:
                print(f"  base {step:6d}  ntp {loss.item():.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    fm = make_fm()
    opt_fm = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=weight_decay)
    print(f"\n--- temporal FM: {fm_steps} steps against the FROZEN base ---", flush=True)
    gen_fm = torch.Generator().manual_seed(seed + 1000)
    for step in range(fm_steps):
        fm.train()
        x, y = get_ntp_batch(gen_fm)
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
            h = inter[deep_block]
            delta = h[:, 1:, :] - h[:, :-1, :]
        l = F.mse_loss((fm(h) - h)[:, :-1, :], delta)
        opt_fm.zero_grad(); l.backward(); opt_fm.step()
        if step % log_interval == 0 or step == fm_steps - 1:
            print(f"  fm {step:6d}  mse {l.item():.5f}", flush=True)

    g = torch.Generator().manual_seed(9999)
    tot = 0.0
    with torch.no_grad():
        for _ in range(40):
            x, y = get_ntp_batch(g)
            _, ls = model(x, y)
            tot += ls.item()
    val = tot / 40
    print(f"\n{wtag}  own-world val {val:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "fm_temporal": fm.state_dict(),
                "config": {"v": v, "s": s, "L": L, "m": m, "world": world,
                           "spec": spec, "n_layer": n_layer, "n_head": n_head,
                           "n_embd": n_embd, "base_steps": base_steps,
                           "fm_steps": fm_steps, "seed": seed, "rule_seed": rule_seed,
                           "data_seed": data_seed, "pool_size": pool_size,
                           "own_world_val": val}}, ckpt_path)
    volume.commit()
    print(f"  saved -> {ckpt_path}", flush=True)
    return {"world": world, "wtag": wtag, "val": val, "ckpt": ckpt_path}


# --------------------------------------------------------------------------- #
# the main run
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=131072, cpu=8.0)
def decompose(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    deep_block: str = "post_block6",
    probe_blocks: str = "post_block4,post_block6,post_block7",
    q_k: int = 7, sigma_s: float = 1.0, kappa: float = 0.15, half: float = 2.0,
    dir_seed: int = 0,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    base_steps: int = 12000, fm_steps: int = 12000,
    worlds: str = "drift,incoh",
    uniform_ckpt: str = "",
    n_probe_train: int = 4000, n_calib: int = 500, n_test: int = 2000,
    q_probe_steps: int = 3000, q_probe_lr: float = 1e-3, q_probe_hidden: int = 512,
    probe_steps: int = 8000, probe_lr: float = 1e-3, probe_batch: int = 512,
    probe_hidden: int = 1024,
    val_pool: int = 20000, val_seed: int = 123,
    eval_seed: int = 999, oracle_chunk: int = 250, filter_chunk: int = 50,
    n_bins: int = 40, batch_size: int = 64,
    seed: int = 42, tag: str = "dec0",
):
    """The exact three-way decomposition, the laundering probe, and the model-side relation.

    Everything is open-loop against frozen models, as in `conditional_revision` -- this is
    measurement, and Half 2 is where anything consumes it.
    """
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.conditional_revision import oracle as ORC
    from rhm.conditional_revision.gates_ab import (
        _auc, _overlap, _partial_r2, _partial_r2_rank, _r2, _strata, _stratified_auc)
    from rhm.question_model import demand_state as DS
    from rhm.question_model import oracle_q as OQ
    from rhm.question_model import qfilter as QF

    device = "cuda"
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    spec = _spec_from_args(q_k, sigma_s, kappa, half, dir_seed, L)
    dirs = DS.make_directions(rules, spec)
    comp_names = [nm for nm, _ in spec["components"]]
    C = len(comp_names)
    idx_states, pi_stat = DS.state_grid(spec)
    P_trans = DS.joint_transition(spec)
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw_states = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
    rp_states = np.stack([W[g][0] for g in range(G)])
    pblocks = [b.strip() for b in probe_blocks.split(",") if b.strip()]
    wlist = [w.strip() for w in worlds.split(",") if w.strip()]

    n_nodes = (s ** L - 1) // (s - 1)
    node_off = {d: (s ** d - 1) // (s - 1) for d in range(L)}
    anc_idx = np.stack([np.array([node_off[d] + ((t + 1) // (s ** (L - d)))
                                 for t in range(T - 1)]) for d in range(L)])
    anc_mask = np.zeros((T, n_nodes), dtype=bool)
    for t in range(T):
        for u in (t, min(t + 1, T - 1)):
            for d in range(L):
                anc_mask[t, node_off[d] + (u // (s ** (L - d)))] = True
    anc_mask_t = torch.from_numpy(anc_mask)

    print(f"{'=' * 78}\nQUESTION MODEL -- DECOMPOSE  tag={tag}")
    print(f"  spec {spec}")
    print(f"  n_states={G}  H(theta)={DS.theta_entropy(spec):.3f} nats  components={comp_names}")
    print(f"  worlds={wlist}  probe blocks={pblocks}")
    print(f"{'=' * 78}", flush=True)
    R = {"spec": spec, "components": comp_names, "n_states": G,
         "H_theta": DS.theta_entropy(spec), "config": {
             "v": v, "s": s, "L": L, "m": m, "T": T, "rule_seed": rule_seed,
             "n_probe_train": n_probe_train, "n_calib": n_calib, "n_test": n_test,
             "base_steps": base_steps, "fm_steps": fm_steps, "seed": seed}}

    # ---------------- aligned eval trajectory ----------------
    n_align = n_probe_train + n_calib + n_test
    print(f"\n--- eval corpus: {n_align} aligned blocks under `drift` ---", flush=True)
    al_seqs, al_lf, al_lr, al_tidx = DS.sample_corpus(rules, spec, dirs, n_align,
                                                      eval_seed, world="drift")
    al_x = torch.from_numpy(al_seqs.astype(np.int64))
    node_y = np.concatenate([al_lf[d] for d in range(L)], axis=1)
    node_y_t = torch.from_numpy(node_y.astype(np.int64))
    theta_comp = idx_states[al_tidx]                     # (n_align, C) component indices
    sl_tr = slice(0, n_probe_train)
    sl_ca = slice(n_probe_train, n_probe_train + n_calib)
    sl_te = slice(n_probe_train + n_calib, n_align)
    te0 = n_probe_train + n_calib

    # ---------------- the oracle: three channels ----------------
    t0 = time.time()
    print(f"\n--- oracle A: structure + aleatoric at the TRUE theta (test set) ---",
          flush=True)
    rw_te = [np.stack([W[i][1][d] for i in al_tidx[sl_te]]) for d in range(L)]
    rp_te = np.stack([W[i][0] for i in al_tidx[sl_te]])
    sc = OQ.structure_channel(rules, al_seqs[sl_te], [al_lf[d][sl_te] for d in range(L + 1)],
                              rw_te, rp_te, chunk=oracle_chunk, verbose=True,
                              anc_mask=anc_mask)
    ccheck = OQ.conditional_self_check(sc)
    print(f"  conditional identity  E[S_D|theta] vs E[H_tot - H_irr]:")
    for k, r in ccheck["per_D"].items():
        print(f"    {k} ({r['d_name']}): {r['mean_S_joint']:.5f} vs "
              f"{r['mean_H_tot_minus_H_irr']:.5f}  err {r['abs_err']:.2e}  "
              f"{'ok' if r['passed'] else 'FAIL'}")
    if not ccheck["passed"]:
        raise RuntimeError("conditional identity failed -- the weighted BP is wrong")
    R["conditional_self_check"] = ccheck
    R["bayes_acc_chain"] = sc["bayes_acc_chain"]
    print(f"  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n--- oracle B: the exact question filter over all {n_align} blocks ---",
          flush=True)
    t0 = time.time()
    ll = QF.prefix_loglik(rules, al_seqs, rw_states, rp_states, chunk=filter_chunk,
                          verbose=True)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)
    dc_win = QF.demand_channel(ll, pi_stat, state_idx=idx_states, n_comp=C, K=spec["K"])
    hp = QF.history_priors(ll, pi_stat, P_trans)
    dc_hist = QF.demand_channel(ll, hp, state_idx=idx_states, n_comp=C, K=spec["K"])
    surp_theta_all = -ll[np.arange(n_align), al_tidx, :][:, 1:]
    del ll

    # the three-way identity, on the test set
    tw = QF.three_way_check(dc_win["surprisal"][sl_te], dc_win["Dm"][sl_te],
                            sc["S_joint"], sc["H_irr"], sc["Ds"], L)
    print(f"\n  THREE-WAY IDENTITY (window filter), test set:")
    for k, r in tw["per_D"].items():
        print(f"    {k} ({r['d_name']}): total {r['total']:.5f} = demand {r['demand']:.5f}"
              f" + structure {r['structure']:.5f} + aleatoric {r['aleatoric']:.5f}"
              f"  (sum {r['sum']:.5f}, err {r['abs_err']:.2e})  "
              f"{'ok' if r['passed'] else 'FAIL'}")
    R["three_way"] = tw
    R["demand"] = {
        "Dm_window": float(dc_win["Dm"][sl_te].mean()),
        "Dm_history": float(dc_hist["Dm"][sl_te].mean()),
        "surp_marg_window": float(dc_win["surprisal"][sl_te].mean()),
        "surp_marg_history": float(dc_hist["surprisal"][sl_te].mean()),
        "surp_theta": float(surp_theta_all[sl_te].mean()),
        "H_theta_window_start": float(dc_win["H_theta"][sl_te, 0].mean()),
        "H_theta_window_end": float(dc_win["H_theta"][sl_te, -1].mean()),
        "H_theta_history_start": float(dc_hist["H_theta"][sl_te, 0].mean()),
        "H_theta_history_end": float(dc_hist["H_theta"][sl_te, -1].mean()),
        "per_component_window": {comp_names[c]: float(dc_win["Dm_comp"][c][sl_te].mean())
                                 for c in range(C)},
        "per_component_history": {comp_names[c]: float(dc_hist["Dm_comp"][c][sl_te].mean())
                                  for c in range(C)},
        "Dm_window_by_position": dc_win["Dm"][sl_te].mean(0).tolist(),
        "Dm_history_by_position": dc_hist["Dm"][sl_te].mean(0).tolist(),
    }
    # The exact Bayes ladder on drift data. `surp_incoh` is the cross-entropy of the
    # THETA-FREE per-node process the `incoh` world is generated from -- i.e. the Bayes
    # ceiling of the `incoh`-trained model on drift data. Without it the loss gap between
    # the two trained models has no denominator, and reading it against E[Dm] overstates
    # the prize: an `incoh` model pays both for having no question model AND for its
    # per-node marginals not being the drift process's.
    mrp, mrw = DS.marginal_weights(spec, dirs)
    ll_inc = QF.prefix_loglik(rules, al_seqs[sl_te],
                              [mrw[dd][None] for dd in range(L)], mrp[None],
                              chunk=filter_chunk)
    surp_incoh = float(-ll_inc[:, 0, 1:].mean())
    del ll_inc
    R["demand"]["surp_incoh"] = surp_incoh
    R["demand"]["bayes_ladder"] = {
        "knows_theta": R["demand"]["surp_theta"],
        "infers_theta_in_context": R["demand"]["surp_marg_window"],
        "infers_theta_with_history": R["demand"]["surp_marg_history"],
        "no_theta_at_all": surp_incoh,
        "in_context_channel": R["demand"]["surp_marg_window"] - R["demand"]["surp_theta"],
        "coherence_gap": surp_incoh - R["demand"]["surp_marg_window"],
    }
    print(f"\n  exact Bayes ladder on drift data (nats/token): knows theta "
          f"{R['demand']['surp_theta']:.4f} < history {R['demand']['surp_marg_history']:.4f}"
          f" < window {R['demand']['surp_marg_window']:.4f} < no theta {surp_incoh:.4f}")
    print(f"\n  demand channel: window {R['demand']['Dm_window']:.5f} nats/token, "
          f"history {R['demand']['Dm_history']:.5f}   "
          f"(per component window {R['demand']['per_component_window']})")
    print(f"  question uncertainty H(theta): prior {R['H_theta']:.3f} -> "
          f"window-end {R['demand']['H_theta_window_end']:.3f}, "
          f"history-start {R['demand']['H_theta_history_start']:.3f}", flush=True)

    # ---------------- drift-off control ----------------
    print(f"\n--- control: drift OFF (sigma_s = 0) on {n_test} blocks ---", flush=True)
    off_seqs, off_lf, _olr = _generate_with_traces(rules, n_test, eval_seed + 555)
    rw_u, rp_u = OQ.uniform_w(rules, n_test)
    sc_off = OQ.structure_channel(rules, off_seqs, off_lf, rw_u, rp_u,
                                  chunk=oracle_chunk, verbose=False)
    ref_off = ORC.revision_and_entropy(rules, off_seqs, off_lf, chunk=oracle_chunk,
                                       verbose=False)
    fid = max(float(np.abs(sc_off["S_joint"][D] - ref_off["B_joint"][D]).max())
              for D in sc_off["Ds"])
    ll_off = QF.prefix_loglik(rules, off_seqs,
                              [np.full((1, v, m), 1.0 / m) for _ in range(L)],
                              np.full((1, v), 1.0 / v), chunk=filter_chunk)
    dc_off = QF.demand_channel(ll_off, np.ones(1))
    print(f"  fidelity vs conditional_revision/oracle.py: max|delta| = {fid:.2e}")
    print(f"  drift-off demand channel: max|Dm| = {float(np.abs(dc_off['Dm']).max()):.2e}")
    R["drift_off"] = {
        "fidelity_max_delta": fid,
        "Dm_max": float(np.abs(dc_off["Dm"]).max()),
        "mean_S_by_D": {f"d{L - D}": float(sc_off["S_joint"][D].mean())
                        for D in sc_off["Ds"]},
        "frac_S_zero_by_D": {f"d{L - D}": float((sc_off["S_joint"][D] == 0).mean())
                             for D in sc_off["Ds"]},
        "mean_Hirr_by_D": {f"d{L - D}": float(sc_off["H_irr"][D].mean())
                           for D in sc_off["Ds"]},
        "mean_surprisal": float(sc_off["surprisal"].mean()),
    }
    R["drift_on"] = {
        "mean_S_by_D": {f"d{L - D}": float(sc["S_joint"][D].mean()) for D in sc["Ds"]},
        "frac_S_zero_by_D": {f"d{L - D}": float((sc["S_joint"][D] == 0).mean())
                             for D in sc["Ds"]},
        "mean_Hirr_by_D": {f"d{L - D}": float(sc["H_irr"][D].mean()) for D in sc["Ds"]},
        "mean_surprisal_theta": float(sc["surprisal"].mean()),
    }
    print(f"  drift-off  mean S:    {R['drift_off']['mean_S_by_D']}")
    print(f"  drift-off  frac S=0:  {R['drift_off']['frac_S_zero_by_D']}")
    print(f"  (published static reference: d6 0.039 d5 0.061 d4 0.105 d3 0.191 "
          f"d2 0.361 d1 0.699; frac zero 0.500 ... 0.291)")
    print(f"  drift-on   mean S:    {R['drift_on']['mean_S_by_D']}", flush=True)
    if fid > 1e-11:
        raise RuntimeError(f"drift-off fidelity broken ({fid})")
    del ll_off, dc_off, sc_off, ref_off

    # ---- control: a STATIC (kappa -> 0) question state ----------------------
    # The sanity identity SS4 has no analogue for: with theta fixed, an
    # unbounded-memory learner identifies it and its demand channel decays to zero,
    # while a T-token-window learner's does not. This is what separates "the drift
    # keeps the channel alive" (true of the ideal learner) from "the finite context
    # window keeps it alive" (true of the transformer).
    print(f"\n--- control: STATIC question state ({n_test} blocks, one theta) ---",
          flush=True)
    st_seqs, _slf, _slr, st_tidx = DS.sample_corpus(rules, spec, dirs, n_test,
                                                    eval_seed + 777, world="static")
    ll_st = QF.prefix_loglik(rules, st_seqs, rw_states, rp_states, chunk=filter_chunk)
    dc_st_win = QF.demand_channel(ll_st, pi_stat)
    hp_st = QF.history_priors(ll_st, pi_stat, P_trans)
    dc_st_hist = QF.demand_channel(ll_st, hp_st)
    q = max(n_test // 10, 1)
    dec_w = [float(dc_st_win["Dm"][i:i + q].mean()) for i in range(0, n_test, q)]
    dec_h = [float(dc_st_hist["Dm"][i:i + q].mean()) for i in range(0, n_test, q)]
    R["static_control"] = {
        "Dm_window_by_decile": dec_w, "Dm_history_by_decile": dec_h,
        "Dm_window_last": float(np.mean(dec_w[-2:])),
        "Dm_history_last": float(np.mean(dec_h[-2:])),
        "H_theta_history_last": float(dc_st_hist["H_theta"][-n_test // 5:, 0].mean()),
    }
    print(f"  window  Dm by decile of corpus position: "
          + " ".join(f"{x:.4f}" for x in dec_w))
    print(f"  history Dm by decile of corpus position: "
          + " ".join(f"{x:.4f}" for x in dec_h))
    print(f"  -> history channel decays {dec_h[0]:.4f} -> {R['static_control']['Dm_history_last']:.5f}"
          f"; window channel flat at {R['static_control']['Dm_window_last']:.4f}", flush=True)
    del ll_st, dc_st_win, dc_st_hist

    volume.commit()
    with open(f"{qm_dir(v, s, depth, m)}/decompose_{tag}_partial.json", "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---------------- models ----------------
    d = qm_dir(v, s, depth, m)
    key = tb_key(v, s, depth, m)

    def wtag_of(world):
        return f"{world}_K{spec['K']}_sg{sigma_s:g}_kp{kappa:g}_ds{dir_seed}"

    def base_path(world):
        return (f"{d}/base_{wtag_of(world)}_{n_layer}L{n_head}H{n_embd}D_"
                f"steps{base_steps}_seed{seed}.pt")

    def make_model():
        mm = GPT(v, T, n_layer, n_head, n_embd).to(device)
        mm.eval()
        for p in mm.parameters():
            p.requires_grad_(False)
        return mm

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    torch.manual_seed(seed)
    models, fms = {}, {}
    for wd in wlist:
        p = base_path(wd)
        if not os.path.exists(p):
            raise FileNotFoundError(f"no base for world `{wd}` at {p}. Run train_world.")
        sd = torch.load(p, map_location=device)
        mm = make_model()
        mm.load_state_dict(sd["model"])
        models[wd] = mm
        f = make_fm()
        f.load_state_dict(sd["fm_temporal"])
        f.eval()
        fms[wd] = f
        print(f"  loaded {wd}: own-world val {sd['config'].get('own_world_val')}", flush=True)
    ucp = uniform_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                           f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
    if os.path.exists(ucp):
        sd = torch.load(ucp, map_location=device)
        mm = make_model()
        mm.load_state_dict(sd["model"])
        models["uniform"] = mm
        print(f"  loaded uniform (conditional_revision's cached base) from {ucp}", flush=True)
    else:
        print(f"  WARNING no cached uniform base at {ucp}; skipping that arm", flush=True)

    # ---------------- the loss matrix: what is the question channel worth? ----------
    print(f"\n--- loss matrix: every model on every world's held-out windows ---",
          flush=True)
    arangeT = torch.arange(T)

    world_off = {"drift": 0, "incoh": 1, "uniform": 2, "static": 3}
    val_pools = {}

    def val_on(world_name, mm, n_batches=120):
        if world_name not in val_pools:
            pseqs, _ = _build_pool(rules, spec, dirs, world_name, val_pool,
                                   val_seed + 101 * world_off[world_name], verbose=False)
            val_pools[world_name] = torch.from_numpy(
                pseqs.astype(np.int64)).reshape(-1)
        cp = val_pools[world_name]
        nc = cp.shape[0]
        g = torch.Generator().manual_seed(4242)
        tot = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                ix = torch.randint(0, nc - T - 1, (batch_size,), generator=g)
                ii = ix[:, None] + arangeT[None, :]
                x, y = cp[ii].to(device), cp[ii + 1].to(device)
                _, ls = mm(x, y)
                tot += ls.item()
        return tot / n_batches

    loss_matrix = {}
    for wname in ["drift", "incoh", "uniform"]:
        loss_matrix[wname] = {}
        for mname, mm in models.items():
            loss_matrix[wname][mname] = val_on(wname, mm)
        print(f"  world `{wname}`: " + "  ".join(
            f"{k} {x:.4f}" for k, x in loss_matrix[wname].items()), flush=True)
    R["loss_matrix"] = loss_matrix
    if "drift" in models and "incoh" in models:
        gap = loss_matrix["drift"]["incoh"] - loss_matrix["drift"]["drift"]
        R["realised_question_value"] = gap
        print(f"\n  REALISED value of the question channel (incoh-trained minus "
              f"drift-trained, on drift data): {gap:+.5f} nats/token")
        print(f"  exact ceiling E[Dm_window] = {R['demand']['Dm_window']:.5f} nats/token "
              f"({100 * gap / max(R['demand']['Dm_window'], 1e-9):.1f}% realised)", flush=True)

    # ---------------- the laundering probe ----------------
    print(f"\n--- laundering probe: is a low-dimensional theta-estimate recoverable? ---",
          flush=True)
    Kc = spec["K"]
    onehots = [np.eye(Kc)[idx_states[:, c]] for c in range(C)]
    marg_win = [dc_win["post_full"] @ onehots[c] for c in range(C)]     # (n, T+1, K)
    marg_hist = [dc_hist["post_full"] @ onehots[c] for c in range(C)]
    y_theta = torch.from_numpy(theta_comp.astype(np.int64))             # (n_align, C)

    # exact ceilings at the model's conditioning set (prefix length p+1 at position p)
    ceil = {}
    for cnd, mg in (("window", marg_win), ("history", marg_hist)):
        acc = np.zeros((C, T)); ce = np.zeros((C, T))
        for c in range(C):
            mm_ = mg[c][sl_te, 1:T + 1, :]                              # (n_te, T, K)
            tt = theta_comp[sl_te, c]
            acc[c] = (mm_.argmax(-1) == tt[:, None]).mean(0)
            ce[c] = -np.log(np.clip(mm_[np.arange(mm_.shape[0]), :, tt], 1e-30, None)).mean(0)
        ceil[cnd] = {"acc": acc, "ce": ce}
        print(f"  exact filter ({cnd}) terminal accuracy: " + "  ".join(
            f"{comp_names[c]} {acc[c, -1]:.3f}" for c in range(C))
            + f"   chance {1.0 / Kc:.3f}", flush=True)
    R["theta_ceiling"] = {cnd: {"acc_by_position": ceil[cnd]["acc"].tolist(),
                                "ce_by_position": ceil[cnd]["ce"].tolist(),
                                "terminal_acc": {comp_names[c]: float(ceil[cnd]["acc"][c, -1])
                                                 for c in range(C)}}
                          for cnd in ceil}

    @torch.no_grad()
    def collect_block(mm, xs, block):
        out = []
        for i in range(0, xs.shape[0], 256):
            xa = xs[i:i + 256].to(device)
            _, _, inter = mm(xa, return_intermediates=True)
            out.append(inter[block].float().cpu())
        return torch.cat(out)

    class QProbe(nn.Module):
        def __init__(self, d_in, hidden, C_, K_):
            super().__init__()
            self.C, self.K = C_, K_
            self.net = (nn.Linear(d_in, C_ * K_) if hidden == 0 else
                        nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                      nn.Linear(hidden, C_ * K_)))

        def forward(self, x):
            return self.net(x).view(*x.shape[:-1], self.C, self.K)

    def fit_qprobe(Xtr, Ytr, Xca, Yca, Xte, Yte, d_in, label, hidden=None,
                   steps=None, shuffle=False):
        """One head over all positions; returns per-position accuracy/CE and the
        temperature-calibrated posterior on the test set."""
        hidden = q_probe_hidden if hidden is None else hidden
        steps = q_probe_steps if steps is None else steps
        pr = QProbe(d_in, hidden, C, Kc).to(device)
        opt = torch.optim.AdamW(pr.parameters(), lr=q_probe_lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed + 11)
        Xf = Xtr.reshape(-1, d_in)
        Yf = Ytr.unsqueeze(1).expand(-1, T, -1).reshape(-1, C)
        if shuffle:
            perm = torch.randperm(Ytr.shape[0], generator=g)
            Yf = Ytr[perm].unsqueeze(1).expand(-1, T, -1).reshape(-1, C)
        mu, sdv = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True) + 1e-6
        for st in range(steps):
            ix = torch.randint(0, Xf.shape[0], (probe_batch,), generator=g)
            xb = ((Xf[ix] - mu) / sdv).to(device)
            yb = Yf[ix].to(device)
            lg = pr(xb)
            loss = sum(F.cross_entropy(lg[:, c], yb[:, c]) for c in range(C)) / C
            opt.zero_grad(); loss.backward(); opt.step()
            if st % max(steps // 3, 1) == 0:
                print(f"    qprobe[{label}] {st:5d}  ce {loss.item():.4f}", flush=True)
        pr.eval()

        @torch.no_grad()
        def logits_of(X):
            outs = []
            for i in range(0, X.shape[0], 128):
                xb = ((X[i:i + 128] - mu) / sdv).to(device)
                outs.append(pr(xb).cpu())
            return torch.cat(outs)                       # (n, T, C, K)

        lg_ca, lg_te = logits_of(Xca), logits_of(Xte)
        # one temperature per component, fitted on the calib split
        temps = []
        for c in range(C):
            lt = torch.zeros(1, requires_grad=True)
            lgc = lg_ca[:, :, c].reshape(-1, Kc)
            ybc = Yca[:, c].unsqueeze(1).expand(-1, T).reshape(-1)
            o = torch.optim.LBFGS([lt], lr=0.1, max_iter=40)

            def cl():
                o.zero_grad()
                l_ = F.cross_entropy(lgc / lt.exp(), ybc)
                l_.backward()
                return l_
            o.step(cl)
            temps.append(float(lt.exp().item()))
        with torch.no_grad():
            q = torch.stack([F.softmax(lg_te[:, :, c] / temps[c], dim=-1)
                             for c in range(C)], dim=2).numpy()   # (n_te, T, C, K)
        acc = np.zeros((C, T)); ce = np.zeros((C, T))
        yt = Yte.numpy()
        for c in range(C):
            acc[c] = (q[:, :, c].argmax(-1) == yt[:, c][:, None]).mean(0)
            ce[c] = -np.log(np.clip(q[np.arange(q.shape[0]), :, c, yt[:, c]], 1e-30,
                                    None)).mean(0)
        return {"acc": acc, "ce": ce, "temps": temps, "q": q}

    probe_res = {}
    for mname, mm in models.items():
        probe_res[mname] = {}
        for block in pblocks:
            A = collect_block(mm, al_x, block)
            r = fit_qprobe(A[sl_tr], y_theta[sl_tr], A[sl_ca], y_theta[sl_ca],
                           A[sl_te], y_theta[sl_te], n_embd, f"{mname}/{block}")
            probe_res[mname][block] = r
            print(f"  {mname:8s} {block:13s} terminal acc: " + "  ".join(
                f"{comp_names[c]} {r['acc'][c, -1]:.3f}" for c in range(C))
                + "   |  mean-over-positions acc: " + "  ".join(
                f"{comp_names[c]} {r['acc'][c].mean():.3f}" for c in range(C)), flush=True)
            del A

    # explicit-estimator baseline: the prefix unigram histogram (what counting alone buys)
    cnt = np.cumsum(np.eye(v, dtype=np.float32)[al_seqs], axis=1)
    hist_feat = torch.from_numpy(np.concatenate(
        [cnt / np.arange(1, T + 1)[None, :, None],
         np.log1p(np.arange(1, T + 1))[None, :, None].repeat(n_align, 0)],
        axis=2).astype(np.float32))
    r_hist = fit_qprobe(hist_feat[sl_tr], y_theta[sl_tr], hist_feat[sl_ca], y_theta[sl_ca],
                        hist_feat[sl_te], y_theta[sl_te], v + 1, "unigram")
    probe_res["unigram_baseline"] = {"features": r_hist}
    print(f"  {'unigram':8s} {'histogram':13s} terminal acc: " + "  ".join(
        f"{comp_names[c]} {r_hist['acc'][c, -1]:.3f}" for c in range(C)), flush=True)
    # shuffled-label control on the drift model's best block
    r_shuf = None
    if "drift" in models:
        A = collect_block(models["drift"], al_x, deep_block)
        r_shuf = fit_qprobe(A[sl_tr], y_theta[sl_tr], A[sl_ca], y_theta[sl_ca],
                            A[sl_te], y_theta[sl_te], n_embd, "drift/shuffled",
                            shuffle=True)
        print(f"  {'drift':8s} {'shuffled':13s} terminal acc: " + "  ".join(
            f"{comp_names[c]} {r_shuf['acc'][c, -1]:.3f}" for c in range(C)), flush=True)
        del A

    R["laundering"] = {
        "chance": 1.0 / Kc,
        "per_model": {mn: {bk: {"terminal_acc": {comp_names[c]: float(rr["acc"][c, -1])
                                                 for c in range(C)},
                                "mean_acc": {comp_names[c]: float(rr["acc"][c].mean())
                                             for c in range(C)},
                                "acc_by_position": rr["acc"].tolist(),
                                "ce_by_position": rr["ce"].tolist(),
                                "temps": rr["temps"]}
                           for bk, rr in bks.items()}
                      for mn, bks in probe_res.items()},
        "shuffled_terminal_acc": (None if r_shuf is None else
                                  {comp_names[c]: float(r_shuf["acc"][c, -1])
                                   for c in range(C)}),
        "frac_of_ceiling_window": {},
    }
    for mn, bks in probe_res.items():
        best = {}
        for c in range(C):
            best[comp_names[c]] = max(
                float(rr["acc"][c, -1]) / max(ceil["window"]["acc"][c, -1], 1e-9)
                for rr in bks.values())
        R["laundering"]["frac_of_ceiling_window"][mn] = best
    print(f"\n  probe / exact-filter ceiling (terminal, best block):")
    for mn, bb in R["laundering"]["frac_of_ceiling_window"].items():
        print(f"    {mn:18s} " + "  ".join(f"{k} {x:.3f}" for k, x in bb.items()),
              flush=True)

    volume.commit()
    with open(f"{d}/decompose_{tag}_partial.json", "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---------------- the model's own registered news ----------------
    print(f"\n--- the model's own revision: structure (chain probe) and question "
          f"(theta probe) ---", flush=True)
    prim = "drift" if "drift" in models else wlist[0]
    mm = models[prim]
    fm = fms[prim]

    @torch.no_grad()
    def fm_signals(xs):
        nlls, rel_t, dn = [], [], []
        for i in range(0, xs.shape[0], 256):
            xa = xs[i:i + 256].to(device)
            ya = torch.cat([xa[:, 1:], xa[:, :1]], dim=1)
            logits, _, inter = mm(xa, ya, return_intermediates=True)
            nll = F.cross_entropy(logits.reshape(-1, v), ya.reshape(-1),
                                  reduction="none").reshape(xa.shape[0], T)
            h = inter[deep_block]
            delta = h[:, 1:, :] - h[:, :-1, :]
            r = (delta - (fm(h) - h)[:, :-1, :]).norm(dim=-1)
            nlls.append(nll[:, :T - 1].cpu())
            rel_t.append((r / (delta.norm(dim=-1) + 1e-6)).cpu())
            dn.append(delta.norm(dim=-1).cpu())
        return {"nll": torch.cat(nlls).numpy(),
                "r_temporal": torch.cat(rel_t).numpy(),
                "delta_norm": torch.cat(dn).numpy()}

    sig = fm_signals(al_x[sl_te])
    print(f"  corr(r_temporal, nll) = "
          f"{np.corrcoef(sig['r_temporal'].ravel(), sig['nll'].ravel())[0, 1]:+.4f}   "
          f"(conditional_revision gate0 aligned: +0.6527)", flush=True)

    class Probe(nn.Module):
        def __init__(self, d_in, hidden, n_out):
            super().__init__()
            self.net = (nn.Linear(d_in, n_out) if hidden == 0 else
                        nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                      nn.Linear(hidden, n_out)))

        def forward(self, x):
            return self.net(x)

    A_all = collect_block(mm, al_x, deep_block)
    Xtr = A_all[sl_tr].reshape(-1, n_embd)
    Ytr = node_y_t[sl_tr].unsqueeze(1).expand(-1, T, -1).reshape(-1, n_nodes)
    chain = Probe(n_embd, probe_hidden, n_nodes * v).to(device)
    opt = torch.optim.AdamW(chain.parameters(), lr=probe_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 7)
    mu, sdv = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    for st in range(probe_steps):
        ix = torch.randint(0, Xtr.shape[0], (probe_batch,), generator=g)
        xb = ((Xtr[ix] - mu) / sdv).to(device)
        yb = Ytr[ix].to(device)
        out = chain(xb).view(-1, n_nodes, v)
        ce = F.cross_entropy(out.reshape(-1, v), yb.reshape(-1),
                             reduction="none").view(-1, n_nodes)
        mk = anc_mask_t[(ix % T)].to(device)
        loss = (ce * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(probe_steps // 4, 1) == 0:
            print(f"    chain probe {st:5d}  ce {loss.item():.4f}", flush=True)
    chain.eval()

    @torch.no_grad()
    def chain_logits(X, chunk=64):
        outs = []
        for i in range(0, X.shape[0], chunk):
            xb = ((X[i:i + chunk] - mu) / sdv).to(device)
            outs.append(chain(xb).view(xb.shape[0], T, n_nodes, v).cpu())
        return torch.cat(outs)

    lg_ca = chain_logits(A_all[sl_ca])
    lt = torch.zeros(1, requires_grad=True)
    keepm = anc_mask_t.unsqueeze(0).expand(lg_ca.shape[0], -1, -1).reshape(-1)
    lgf = lg_ca.reshape(-1, v)[keepm]
    ybf = node_y_t[sl_ca].unsqueeze(1).expand(-1, T, -1).reshape(-1)[keepm]
    if lgf.shape[0] > 500_000:
        gg = torch.Generator().manual_seed(seed + 13)
        sub = torch.randperm(lgf.shape[0], generator=gg)[:500_000]
        lgf, ybf = lgf[sub], ybf[sub]
    o = torch.optim.LBFGS([lt], lr=0.1, max_iter=50)

    def _cl():
        o.zero_grad()
        l_ = F.cross_entropy(lgf / lt.exp(), ybf)
        l_.backward()
        return l_
    o.step(_cl)
    Tcal = float(lt.exp().item())
    print(f"  chain probe temperature {Tcal:.3f}", flush=True)

    lg_te = chain_logits(A_all[sl_te])
    q_chain = F.softmax(lg_te / Tcal, dim=-1).numpy()          # (n_te, T, n_nodes, v)
    del lg_te, lg_ca, A_all, Xtr, Ytr

    def kl_step(q):
        a = q[:, 1:], q[:, :-1]
        lr_ = np.log(np.clip(a[0], 1e-30, None)) - np.log(np.clip(a[1], 1e-30, None))
        return np.where(a[0] > 0, a[0] * lr_, 0.0).sum(-1)      # (n, T-1, n_nodes)

    kl_nodes = kl_step(q_chain)
    M_lvl = np.stack([np.take_along_axis(kl_nodes, anc_idx[dd][None, :, None],
                                        axis=2)[:, :, 0] for dd in range(L)])
    M_struct = {D: M_lvl[:D + 1].sum(0) for D in sc["Ds"]}
    del q_chain, kl_nodes

    q_theta = probe_res[prim][deep_block]["q"]                  # (n_te, T, C, K)
    qa, qb = q_theta[:, 1:], q_theta[:, :-1]
    lr_ = np.log(np.clip(qa, 1e-30, None)) - np.log(np.clip(qb, 1e-30, None))
    M_theta_comp = np.where(qa > 0, qa * lr_, 0.0).sum(-1)      # (n_te, T-1, C)
    M_theta = M_theta_comp.sum(-1)

    # ---------------- the relation: which channel does the model register? ----
    flat = lambda a: np.asarray(a).ravel()
    surp_marg = flat(dc_win["surprisal"][sl_te])
    surp_th = flat(surp_theta_all[sl_te])
    Dm = flat(dc_win["Dm"][sl_te])
    Dm_h = flat(dc_hist["Dm"][sl_te])
    nllm = flat(sig["nll"])
    pos = np.tile(np.arange(T - 1), (n_test, 1)).ravel()
    oracle_sig = {"Dm": Dm, "Dm_history": Dm_h,
                  **{f"Dm_{comp_names[c]}": flat(dc_win["Dm_comp"][c][sl_te])
                     for c in range(C)},
                  **{f"S_chain_d{L - D}": flat(sc["S_chain"][D]) for D in sc["Ds"]},
                  **{f"S_joint_d{L - D}": flat(sc["S_joint"][D]) for D in sc["Ds"]},
                  **{f"Hirr_d{L - D}": flat(sc["H_irr"][D]) for D in sc["Ds"]}}
    model_sig = {"M_theta": flat(M_theta),
                 **{f"M_theta_{comp_names[c]}": flat(M_theta_comp[:, :, c])
                    for c in range(C)},
                 **{f"M_struct_d{L - D}": flat(M_struct[D]) for D in sc["Ds"]},
                 "r_temporal": flat(sig["r_temporal"]),
                 "delta_norm": flat(sig["delta_norm"]),
                 "nll": nllm}
    rng_sh = np.random.default_rng(seed + 5)
    perm = rng_sh.permutation(n_test)
    model_sig["M_theta_shuffled"] = flat(M_theta[perm])
    model_sig["M_struct_shuffled"] = flat(M_struct[max(sc["Ds"]) - 1][perm])

    print(f"\n--- partial R^2 (model signal ~ oracle channel | marginal surprisal) ---",
          flush=True)
    prel = {}
    for msn, ms in model_sig.items():
        prel[msn] = {}
        for osn, os_ in oracle_sig.items():
            prel[msn][osn] = {"lin": _partial_r2(ms, os_, surp_marg),
                              "rank": _partial_r2_rank(ms, os_, surp_marg)}
        prel[msn]["_surp_given_best"] = _partial_r2(ms, surp_marg, Dm)
    hdr = ["Dm", "Dm_root", "Dm_hi", "Dm_lo", f"S_chain_d2", f"S_chain_d3",
           f"S_chain_d1", "Hirr_d1"]
    hdr = [h for h in hdr if h in oracle_sig]
    print("  " + "signal".ljust(22) + "".join(h.rjust(12) for h in hdr))
    for msn in ["M_theta", "M_theta_root", "M_theta_hi", "M_theta_lo",
                "M_struct_d1", "M_struct_d2", "M_struct_d3", "M_struct_d6",
                "r_temporal", "delta_norm", "nll", "M_theta_shuffled",
                "M_struct_shuffled"]:
        if msn not in prel:
            continue
        print("  " + msn.ljust(22)
              + "".join(f"{prel[msn][h]['rank']:12.4f}" for h in hdr), flush=True)
    R["partial_r2"] = prel

    # ---------------- Gate-B-style matched families ----------------
    print(f"\n--- matched contrast: three families at matched (position x exact "
          f"marginal surprisal) ---", flush=True)
    fam_res = {}
    # Strata are built WITHIN each position with that position's own quantile edges, and
    # offset so no index aliases across positions (gates_ab's `pos * K + bin` gotcha).
    # Under drift the marginal surprisal is continuous rather than atomic, so global
    # quantile edges leave uneven occupancy per (position, bin) cell and the matching
    # variable leaks -- measured at guard 0.73 with global bins on the smoke run.
    strata = np.zeros(pos.size, dtype=np.int64)
    _off = 0
    for _t in range(T - 1):
        _sel = pos == _t
        _st = _strata(surp_marg[_sel], n_bins)
        strata[_sel] = _st + _off
        _off += int(_st.max()) + 1
    # WITHIN-POSITION quantile rank of the demand channel. Dm is strongly front-loaded
    # inside a block (the question is resolved early), so a global quantile would make
    # the demand family a set of early positions and the position-matched contrast would
    # have almost no comparable pairs. Ranking inside each position removes that by
    # construction, and position is a matching variable anyway.
    Dm2 = Dm.reshape(n_test, T - 1)
    dm_rank = np.argsort(np.argsort(Dm2, axis=0), axis=0) / max(n_test - 1, 1)
    dm_rank = dm_rank.ravel()
    for D in sc["Ds"]:
        Sc = flat(sc["S_chain"][D])
        posp = Sc[Sc > 0]
        if posp.size == 0:
            continue
        med = np.median(posp)
        # syn / struct share a demand band and differ only in structure; syn / demand
        # share S == 0 exactly and differ only in demand. So each contrast varies one
        # channel with the other held, and the oracle guards (Dm in struct_vs_syn,
        # S in demand_vs_syn) must both read ~0.5.
        syn = (Sc == 0) & (dm_rank <= 0.25)
        stru = (Sc > med) & (dm_rank <= 0.25)
        dem = (dm_rank >= 0.75) & (Sc == 0)
        block = {"n_syn": int(syn.sum()), "n_struct": int(stru.sum()),
                 "n_demand": int(dem.sum())}
        for cname, pmask in (("struct_vs_syn", stru), ("demand_vs_syn", dem),
                             ("demand_vs_struct", dem)):
            neg = syn if cname != "demand_vs_struct" else stru
            sel = pmask | neg
            if pmask.sum() < 50 or neg.sum() < 50:
                continue
            entry = {}
            for nm, sg in list(model_sig.items()) + [("surp_marg_guard", surp_marg),
                                                     ("Dm_oracle", Dm),
                                                     (f"S_oracle_d{L - D}", Sc)]:
                a, _p = _stratified_auc(sg[sel], pmask[sel], strata[sel])
                entry[nm] = a
            block[cname] = entry
        fam_res[f"d{L - D}"] = block
        if "demand_vs_syn" in block and "struct_vs_syn" in block:
            print(f"  d{L - D}  n(syn/struct/demand) = {block['n_syn']}/"
                  f"{block['n_struct']}/{block['n_demand']}")
            for cname in ("struct_vs_syn", "demand_vs_syn"):
                e = block[cname]
                print(f"    {cname:18s} guard {e['surp_marg_guard']:.3f}  "
                      f"M_theta {e['M_theta']:.3f}  M_struct {e.get(f'M_struct_d{L - D}', float('nan')):.3f}  "
                      f"r_temporal {e['r_temporal']:.3f}  nll {e['nll']:.3f}  "
                      f"| oracle Dm {e['Dm_oracle']:.3f} S {e[f'S_oracle_d{L - D}']:.3f}",
                      flush=True)
    R["matched_families"] = fam_res

    d_out = f"{d}/decompose_{tag}.json"
    with open(d_out, "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {d_out}", flush=True)
    return {"tag": tag, "ok": True}
