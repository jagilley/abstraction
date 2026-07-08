"""Ensemble-agreement trajectory: does the FM residual become FM-INVARIANT over training?

Complexodynamics follow-up (RHM_FM_REGULARIZER + RHM_LATENT_LOOP synthesis). The saved
per-checkpoint trajectories show the sophistication proxies rise and then either fall
(under FM-reg pressure) or arrest (WD only), while deep-level residual eta^2 persists in
every arm. The proposed reading: the persistent deep component is the DGP's own
complextropy floor relative to the FM bound -- Soph(DGP | bound) -- while the transient
component is trajectory scaffolding. Prediction: the residual's FM-invariance
(fresh-FM-ensemble pairwise cosine, the discriminator from RHM_LATENT_LOOP Exp 4) should
RISE over training toward ceiling as the transient (FM-idiosyncratic) part is absorbed
and the residual converges onto the invariant DGP-aligned core -- and rise further/faster
under FM-reg pressure than in the WD-only control.

Measurement, per (arm, checkpoint): train n_fm fresh FMs (different seeds; capB config on
the E->b6 gap, identical to the analyze_reg eval FM) on the frozen checkpoint, then
  - pairwise_cos: input-centered last-token residual cosine across FM pairs
    (high => residual determined by the INPUT: FM-invariant / DGP-aligned;
     low  => determined by the FM: idiosyncratic capacity noise)
  - shared_norm_frac: ||mean residual|| / mean ||residual||
  - per-level feature eta^2 of the ensemble-MEAN residual (is the shared component
    deep-DGP-structured?) + of FM #1's residual (continuity with analyze_reg parts)
  - effective rank / top1-PC of the mean last-token residual (concentration of the core)

Arms: "fmreg:<lam>" (rhm_fm_regularizer checkpoints, e.g. fmreg:1.0 to 300k) and
"wd:<wd>" (rhm_wd_sweep checkpoints, e.g. wd:0.1 to 150k -- the lambda=0 control).
Same m2 substrate (v16/m2, 8L/8H/256D, distinct-rule DGP, rule_seed=0) everywhere.

Resumable: one part JSON per (arm, step), skipped if present.

Run:
    cd experiments
    # smoke test (1 ckpt, tiny FMs):
    modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory \
        --arms "fmreg:1.0" --steps "300000" --n-fm 2 --fm-train-steps 300 \
        --n-eval-sequences 800 --tag smoke
    # full run:
    modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_fm_legibility import (
    _tb_key, _generate_with_traces, _compute_hierarchy_eta2, _compute_effective_rank,
)
from rhm.rhm_fm_regularizer import _reg_dir, REG_SRC, REG_TGT, REG_FM
from rhm.rhm_wd_sweep import _sweep_dir

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-ensemble-trajectory", image=image)

OUT_DIR = f"{DATA_DIR}/rhm_ensemble_trajectory"

# Default checkpoint steps per arm kind (log-spaced subsets of each harness's saved grid).
DEFAULT_STEPS = {
    "fmreg": "0,500,4000,12000,30000,80000,160000,300000",
    "wd": "0,500,4000,10000,25000,60000,100000,150000",
}


def _arm_dir(key, arm):
    kind, val = arm.split(":")
    if kind == "fmreg":
        return _reg_dir(key, float(val))
    if kind == "wd":
        return _sweep_dir(key, float(val))
    raise ValueError(f"unknown arm kind: {arm}")


def _arm_tag(arm):
    return arm.replace(":", "_").replace(".", "p")


@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=32768)
def ensemble_ckpt(
    arm: str, checkpoint_path: str, checkpoint_step: int, tag: str = "",
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    n_fm: int = 4, fm_train_steps: int = 8000, fm_lr: float = 1e-3,
    n_eval_sequences: int = 4000, batch_size: int = 64, base_seed: int = 911,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    L = depth
    block_size = s ** L
    device = "cuda"
    key = _tb_key(v, s, L, m)
    volume.reload()

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    corpus = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    train_data = corpus[: int(0.9 * len(corpus))]

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=REG_FM["dh"], n_head=REG_FM["nh"],
            n_layer=REG_FM["nl"], mlp_mult=REG_FM["mm"], block_size=block_size,
        ).to(device)

    resid_last = []       # per-FM (N, d) last-token residuals, float32
    resid_full_sum = None  # running sum of (N, T, d) residuals for the ensemble mean
    rf_first = None        # FM #1's full residual (reference eta^2)
    fm_stats = []
    for k in range(n_fm):
        torch.manual_seed(base_seed + 1000 * (k + 1))
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)
        gen = torch.Generator().manual_seed(base_seed + 2000 * (k + 1))
        for _ in range(fm_train_steps):
            fm.train()
            ix = torch.randint(len(train_data) - block_size - 1, (batch_size,), generator=gen)
            x = torch.stack([train_data[i:i + block_size] for i in ix]).to(device)
            with torch.no_grad():
                _, _, inter = model(x, return_intermediates=True)
                src, tgt = inter[REG_SRC], inter[REG_TGT]
            loss = F.mse_loss(fm(src), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        fm.eval()
        rlast, rfull, coss = [], [], []
        with torch.no_grad():
            for i in range(0, len(eval_x), batch_size):
                b = eval_x[i:i + batch_size]
                _, _, inter = model(b, return_intermediates=True)
                pred, tgt = fm(inter[REG_SRC]), inter[REG_TGT]
                r = tgt - pred
                rlast.append(r[:, -1, :].float().cpu())
                rfull.append(r.half().cpu())
                coss.append(float(F.cosine_similarity(pred, tgt, dim=-1).mean()))
        rl, rf = torch.cat(rlast), torch.cat(rfull)
        resid_last.append(rl)
        resid_full_sum = rf.float() if resid_full_sum is None else resid_full_sum + rf.float()
        if k == 0:
            rf_first = rf
        fm_stats.append({"fm_seed": base_seed + 1000 * (k + 1),
                         "fm_cosine": float(np.mean(coss)),
                         "res_norm_last": float(rl.norm(dim=-1).mean())})
        print(f"  [{arm} step {checkpoint_step}] FM {k + 1}/{n_fm}: "
              f"cos={fm_stats[-1]['fm_cosine']:.4f} resNorm={fm_stats[-1]['res_norm_last']:.2f}")
        del fm, opt
        torch.cuda.empty_cache()

    # Ensemble agreement on input-centered last-token residuals (RHM_LATENT_LOOP Exp 4).
    R = torch.stack(resid_last)                    # (n_fm, N, d)
    Rc = R - R.mean(dim=1, keepdim=True)           # center per FM over examples
    Rn = F.normalize(Rc, dim=-1)
    cps = [float((Rn[i] * Rn[j]).sum(-1).mean())
           for i in range(n_fm) for j in range(i + 1, n_fm)]
    pairwise = float(np.mean(cps))
    shared = float(R.mean(0).norm(dim=-1).mean() / R.norm(dim=-1).mean())

    # Structure of the shared (ensemble-mean) residual: is the invariant core DGP-aligned?
    mean_full = (resid_full_sum / n_fm).numpy().astype(np.float32)
    nC = mean_full.shape[0]
    lf = [x[:nC] for x in level_features]
    lr_ = [x[:nC] for x in level_rules]
    eta2_mean = _compute_hierarchy_eta2(mean_full, lf, lr_, s=s, L=L)
    eta2_fm0 = _compute_hierarchy_eta2(rf_first.float().numpy(), lf, lr_, s=s, L=L)
    eff_rank, top1 = _compute_effective_rank(R.mean(0).numpy(), n_embd)

    result = {
        "arm": arm, "step": checkpoint_step, "key": key, "gap": f"{REG_SRC}->{REG_TGT}",
        "fm_config": REG_FM, "n_fm": n_fm, "fm_train_steps": fm_train_steps,
        "pairwise_cos": pairwise, "pairwise_cos_all": cps, "shared_norm_frac": shared,
        "fm_stats": fm_stats,
        "mean_res_effective_rank_pct": float(eff_rank / n_embd * 100),
        "mean_res_top1_pc": float(top1),
        "eta2_mean_residual": eta2_mean, "eta2_fm0_residual": eta2_fm0,
    }
    parts_dir = f"{OUT_DIR}/parts"
    os.makedirs(parts_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    with open(f"{parts_dir}/{_arm_tag(arm)}_step{checkpoint_step}{suffix}.json", "w") as fp:
        json.dump(result, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"=== {arm} step {checkpoint_step}: pairwise_cos={pairwise:.3f} "
          f"shared={shared:.3f} meanResRank={eff_rank / n_embd * 100:.1f}%")
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=4096)
def ensemble_trajectory(
    arms: str = "fmreg:1.0,wd:0.1", steps: str = "", tag: str = "",
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_fm: int = 4, fm_train_steps: int = 8000, n_eval_sequences: int = 4000,
):
    """Spawn ensemble_ckpt for each (arm, checkpoint step); aggregate + print trajectory."""
    key = _tb_key(v, s, depth, m)
    volume.reload()
    parts_dir = f"{OUT_DIR}/parts"
    os.makedirs(parts_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    pending, planned = [], {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        info_path = os.path.join(_arm_dir(key, arm), "training_info.json")
        if not os.path.exists(info_path):
            print(f"  {arm}: no training_info.json at {info_path}, skipping arm")
            continue
        info = json.load(open(info_path))
        avail = {c["step"]: c["path"] for c in info["checkpoints"]}
        want = [int(x) for x in (steps or DEFAULT_STEPS[arm.split(":")[0]]).split(",")]
        missing = [st for st in want if st not in avail]
        if missing:
            print(f"  {arm}: requested steps not in saved grid, skipping: {missing}")
        planned[arm] = [st for st in want if st in avail]
        for st in planned[arm]:
            part = f"{parts_dir}/{_arm_tag(arm)}_step{st}{suffix}.json"
            if os.path.exists(part):
                continue
            h = ensemble_ckpt.spawn(
                arm=arm, checkpoint_path=avail[st], checkpoint_step=st, tag=tag,
                v=v, m=m, s=s, depth=depth, n_fm=n_fm,
                fm_train_steps=fm_train_steps, n_eval_sequences=n_eval_sequences)
            pending.append((arm, st, h))
    print(f"=== ensemble_trajectory: {len(pending)} (arm, ckpt) jobs to run ===")
    for arm, st, h in pending:
        try:
            h.get()
        except Exception as e:
            print(f"  {arm} step {st}: FAILED -- {e}")

    volume.reload()
    L = depth
    out = {"key": key, "arms": arms, "gap": f"{REG_SRC}->{REG_TGT}", "results": {}}
    for arm, want in planned.items():
        rows = []
        for st in want:
            part = f"{parts_dir}/{_arm_tag(arm)}_step{st}{suffix}.json"
            if os.path.exists(part):
                rows.append(json.load(open(part)))
        rows.sort(key=lambda r: r["step"])
        out["results"][arm] = rows
        print(f"\n=== {arm} (fresh-FM ensemble n={n_fm}, gap {REG_SRC}->{REG_TGT}) ===")
        print(f"{'step':>7} {'ensCos':>6} {'shared':>6} {'fmCos':>6} {'resNorm':>7} "
              f"{'mRk%':>5} {'mTop1%':>6} | mean-residual eta2_last d4/d5/d6")
        for r in rows:
            e = r["eta2_mean_residual"]
            d4, d5, d6 = (e["level_2"]["feature_eta2_last"], e["level_1"]["feature_eta2_last"],
                          e["level_0"]["feature_eta2_last"])
            fmcos = float(sum(fs["fm_cosine"] for fs in r["fm_stats"]) / len(r["fm_stats"]))
            rnorm = float(sum(fs["res_norm_last"] for fs in r["fm_stats"]) / len(r["fm_stats"]))
            print(f"{r['step']:>7} {r['pairwise_cos']:>6.3f} {r['shared_norm_frac']:>6.3f} "
                  f"{fmcos:>6.3f} {rnorm:>7.2f} {r['mean_res_effective_rank_pct']:>5.1f} "
                  f"{r['mean_res_top1_pc'] * 100:>6.1f} | {d4:.3f} / {d5:.3f} / {d6:.3f}")

    with open(f"{OUT_DIR}/summary{suffix}.json", "w") as fp:
        json.dump(out, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
