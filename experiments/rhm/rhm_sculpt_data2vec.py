"""RHM Sculpting — Stage 1: validate a NON-PRIVILEGED (data2vec) deep belief IN ISOLATION.

Stage 0 (rhm_sculpt_deepbelief.py) established the CEILING: a privileged oracle-ancestor
aux deepens the controller's belief (mid-level probe d2/d3 0.71/0.61 -> 0.94/0.96), which
makes the block FM a better one-step ranker (delta_cos +0.05, top1 +0.08) and helps the
LATENT beam more than the token beam at every width (gap closes 40-60%). So there IS planning
headroom from a deeper belief -- but the oracle aux "cheats" by using ground-truth latents.

This file builds the honest version: a data2vec-style EMA self-distilled aux (no labels), and
VALIDATES IT IN ISOLATION before any planning -- exactly the "be sure the component works on
its own" discipline. Because this is RHM we can check the belief against GROUND TRUTH (the
per-level ancestor probe), which no natural-data domain allows. This doubles as RHM_LATENT_LOOP
next-step #2 ("drop the privilege": replace oracle_aux with an EMA own-lifted-latent target).

SINGLE CONTROLLED VARIABLE = the aux target. All three beliefs are trained with the SAME
root-CE (so each has a functional head for the Stage-2 planner) + the SAME masking schedule,
differing ONLY in the aux:
  - parser   : root CE only                                  (floor)
  - oracle   : root CE + lam * ground-truth ancestor CE      (privileged ceiling)
  - data2vec : root CE + lam * EMA-teacher latent regression (non-privileged candidate)

data2vec aux: a block-masked view -> student predicts the EMA teacher's (full-view, layer-
normed) per-position representation at the masked positions. Predicting a masked block's
representation from context requires climbing the parse hierarchy, so -- if it does not
COLLAPSE (data2vec/DINO's characteristic failure; DEEP_COMPOSITION Exp 5b's early contact) --
it should recruit deep structure without labels.

Readouts per belief: (1) belief-depth probe d1..dL (does data2vec land near oracle at the
mid-levels?); (2) anti-collapse diagnostics (participation ratio of the block latents, mean
per-dim variance) -- collapse would show as PR->1 and a probe near chance (1/v).

"Working in isolation" = data2vec mid-level probes >> parser and approaching oracle, with no
collapse. Only then is it worth wiring into the latent beam (Stage 2).

Run:
  modal run rhm/rhm_sculpt_data2vec.py::data2vec_isolation --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_data2vec.py::data2vec_isolation --m 2  # L=4
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_active_query import _count_parameters, _reveal_blocks
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_latent_loop import _generate_with_traces
from rhm.rhm_sculpt_latent import _block_state_chunked, _build_rich_controller
from rhm.rhm_sculpt_deepbelief import (
    _belief_depth_probe,
    _block_ancestor_index,
    _participation_ratio,
    _subtree_mask,
    _train_data2vec_controller,
    _train_deep_controller,
    _train_mlm_controller,
)


app = modal.App("rhm-sculpt-data2vec", image=image)


def _anticollapse_diag(controller, eval_leaves, device):
    """Participation ratio + mean per-dim variance of the per-block latent (what the FM rolls)."""
    import torch
    with torch.no_grad():
        zb = _block_state_chunked(controller, eval_leaves.to(device))  # (N, n_blocks, D)
    flat = zb.reshape(-1, zb.shape[-1])
    return {"participation_ratio": _participation_ratio(flat),
            "pr_frac": _participation_ratio(flat) / zb.shape[-1],
            "mean_dim_var": float(flat.var(dim=0).mean())}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def data2vec_isolation(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_probe: int = 3_000, state_dim: int = 96,
    controller_steps: int = 12_000, batch_size: int = 256, lam_aux: float = 1.0,
    lam_d2v: float = 1.0, mask_ratio: float = 0.5, mask_mode: str = "subtree",
    ema_tau_start: float = 0.996, ema_tau_end: float = 0.9999,
    beliefs: str = "parser,oracle,data2vec", quick: bool = False,
):
    """Stage-1 isolation: does a non-privileged data2vec aux recruit the deep belief structure
    (probed against ground truth) that the oracle aux gets, without collapsing?"""
    import time
    import torch

    if s != 2:
        raise ValueError("Assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    L = depth
    cond_list = [b for b in beliefs.split(",") if b]
    if quick:
        controller_steps = 1_500
        n_train_episodes, n_probe = 20_000, 1_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"data2vec-isolation: v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, "
          f"lam_aux={lam_aux}, lam_d2v={lam_d2v}, mask_ratio={mask_ratio}, "
          f"conditions={cond_list}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    anc_block_idx = _block_ancestor_index(n_blocks, s, L)

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    gen_leaves, gen_lf, _ = _generate_with_traces(rules, n_train_episodes, train_seed)
    train_leaves = torch.from_numpy(gen_leaves.astype(np.int64))
    train_roots = torch.from_numpy(gen_lf[0][:, 0].astype(np.int64))
    train_anc = [torch.from_numpy(gen_lf[ell][:, anc_block_idx[ell]].astype(np.int64))
                 for ell in range(L)]

    probe_leaves_np, probe_lf, _ = _generate_with_traces(rules, n_probe, train_seed + 7)
    probe_leaves = torch.from_numpy(probe_leaves_np.astype(np.int64))

    RichController = _build_rich_controller()
    results = {}
    for belief in cond_list:
        print(f"\n{'#'*72}\n# BELIEF: {belief}\n{'#'*72}")
        torch.manual_seed(train_seed)          # identical init across conditions
        controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
        if belief == "parser":
            _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                                   n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                                   lr=3e-4, device=device, p_full=0.5)
        elif belief == "oracle":
            _train_deep_controller(controller, train_leaves, train_roots, train_anc,
                                   state_dim=state_dim, batch_size=batch_size, n_blocks=n_blocks,
                                   block_size=s, L=L, v=v, n_steps=controller_steps, lr=3e-4,
                                   lam_aux=lam_aux, device=device, p_full=0.5)
        elif belief == "data2vec":
            _train_data2vec_controller(controller, train_leaves, train_roots, state_dim=state_dim,
                                       batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                                       n_steps=controller_steps, lr=3e-4, lam_d2v=lam_d2v,
                                       mask_ratio=mask_ratio, ema_tau_start=ema_tau_start,
                                       ema_tau_end=ema_tau_end, device=device, p_full=0.5,
                                       s=s, L=L, mask_mode=mask_mode)
        elif belief == "mlm":
            _train_mlm_controller(controller, train_leaves, train_roots, v=v, state_dim=state_dim,
                                  batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                                  n_steps=controller_steps, lr=3e-4, lam_mlm=lam_d2v, device=device,
                                  p_full=0.5, s=s, L=L, mask_mode=mask_mode)
        else:
            raise ValueError(belief)
        controller.eval()
        for p in controller.parameters():
            p.requires_grad_(False)

        depth_acc = _belief_depth_probe(controller, probe_leaves, probe_lf, anc_block_idx,
                                        L=L, v=v, device=device)
        diag = _anticollapse_diag(controller, probe_leaves, device)
        results[belief] = {"depth_probe": {f"d{ell+1}": depth_acc[ell] for ell in range(L)},
                           "anticollapse": diag}
        print(f"  [{belief}] depth " + " ".join(f"d{ell+1}={depth_acc[ell]:.3f}" for ell in range(L))
              + f"  | PR={diag['participation_ratio']:.1f}/{state_dim} "
              f"(frac {diag['pr_frac']:.2f}) mean_var={diag['mean_dim_var']:.3f}")

    # side-by-side --------------------------------------------------------------
    print(f"\n{'='*72}\n=== STAGE-1 ISOLATION SUMMARY (m={m}) — data2vec vs oracle vs parser ===\n{'='*72}")
    print(f"  chance = 1/v = {1.0/v:.3f}")
    for belief in cond_list:
        r = results[belief]
        print(f"  [{belief:9s}] " + " ".join(f"{k}={v_:.3f}" for k, v_ in r["depth_probe"].items())
              + f"  | PR {r['anticollapse']['participation_ratio']:.1f} "
              f"var {r['anticollapse']['mean_dim_var']:.3f}")
    if {"parser", "oracle"} <= set(results):
        pa, ora = results["parser"], results["oracle"]
        for cand in ("data2vec", "mlm"):
            if cand not in results:
                continue
            cr = results[cand]
            print(f"\n  mid-level recruitment (fraction of the oracle−parser gap that {cand} closes):")
            for ell in range(1, L):  # d2..dL
                k = f"d{ell+1}"
                gap = ora["depth_probe"][k] - pa["depth_probe"][k]
                closed = (cr["depth_probe"][k] - pa["depth_probe"][k]) / gap if abs(gap) > 1e-6 else float("nan")
                print(f"    {k}: parser {pa['depth_probe'][k]:.3f} -> {cand} {cr['depth_probe'][k]:.3f} "
                      f"-> oracle {ora['depth_probe'][k]:.3f}  ({closed*100:+.0f}% of gap)")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "state_dim": state_dim,
                   "lam_aux": lam_aux, "lam_d2v": lam_d2v, "mask_ratio": mask_ratio,
                   "ema_tau": [ema_tau_start, ema_tau_end], "conditions": cond_list,
                   "controller_steps": controller_steps},
        "chance": 1.0 / v, "results": results, "elapsed_seconds": time.time() - started,
    }
    tag = f"v{v}_s{s}_L{depth}_m{m}_mask-{mask_mode}_{'-'.join(cond_list)}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_data2vec/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    data2vec_isolation.remote(m=m, quick=quick)
