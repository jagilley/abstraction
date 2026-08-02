"""The decomposition instrument applied along the TRAINING trajectory.

The third axis for this sub-experiment: `rhm_decomposition_audit.py` measures the
decomposition statically (setting x capacity x gap), `rhm_decomposition_ratchet.py`
measures it over wake-sleep cycles, and this measures it over the base model's own
training checkpoints -- the axis every complexodynamics claim rests on and the one
the new instrument has never been run on.

WHY A RE-CUT. The rise-and-fall arc in RHM_COMPLEXODYNAMICS_README was measured with
FM-free activation effective rank + naive residual rank + top1-PC. The 2026-07-27
audit (README.md, this dir) showed the single-number rank instrument conflates the
frontier's SHAPE with its SIZE and inflates toward d_model wherever the FM saturates.
So the load-bearing trajectory numbers were produced by an instrument this
sub-experiment has since retired. This re-runs the same arms, same substrate, same
gap, same FM distribution, and computes the retired and the trusted metrics on the
IDENTICAL (A, P) at every checkpoint -- so any disagreement is attributable to the
instrument and not to run-to-run variation. That paired comparison is the control the
original trajectory work could not have.

WHAT IS MEASURED, per (arm, checkpoint, capacity):
  - full_decomposition(A, P): naive triple (R_act/R_comp/R_res, subadditivity),
    geometry (absorption spectrum, alignment index, and the two variance spectra
    that beta is fit from), repaired triple (frontier_mass, R_res_participation).
  - beta = slope of log res_var(i) vs log act_var(i) across A's principal
    directions, refit post-hoc by analyze_summaries.shadow_law from the saved
    spectra (so it is never recomputed differently here than elsewhere).
  - per-level feature_eta2_last of the SAME residual (survived the audit;
    head-robust; the DGP-alignment readout) + the retired eff-rank/top1-PC.

THE TWO GUARDS, both trajectory-specific:

1. SATURATION. beta collapses to 0.11-0.17 and naive rank inflates to 94-96% of
   d_model wherever the FM drives relative residual to ~0 (README section 3). At
   step 0 the base model computes almost nothing and a fresh FM hits cosine ~0.996,
   i.e. the trajectory STARTS inside the noise regime the audit identified. That is
   an honest reading there (nothing to model is why it saturates, not a broken
   instrument) but it means early-checkpoint numbers must never be compared to late
   ones without the saturation columns in view. `relative_residual` and `mean_cosine`
   are therefore printed on every row, and the FM config is held FIXED across the
   whole trajectory -- a moving instrument would make the arc uninterpretable.

2. CAPACITY INVARIANCE, AT EVERY POINT. The audit's central validity claim is that
   beta is a property of the model's computation (invariant to a 4x FM-capacity
   range, +/-0.01) while the frontier's level is not. Running two capacities at
   every checkpoint turns that into a per-checkpoint validity check: where the
   capacity spread in beta blows past +/-0.01, the reading is not trustworthy, and
   we expect exactly that in the saturated early regime. The spread is a diagnostic
   column, not a nuisance.

Arms are the complexodynamics arms: "fmreg:<lam>" (structured compression pressure)
and "wd:<wd>" (the lambda=0 control). Same m2 substrate, same E->b6 gap, and the
capB FM config is `REG_FM` verbatim, so capB rows are directly comparable to the
existing rhm_ensemble_trajectory / rhm_fm_regularizer parts.

FM training mirrors rhm_ensemble_trajectory.ensemble_ckpt exactly (8k steps, AdamW
lr 1e-3 wd 0.01, grad clip 1.0, no LR schedule, random corpus batches) rather than
the audit's cosine-scheduled loop, so the FMs are drawn from the same distribution
as the ones that produced the numbers being re-cut.

Resumable: one part JSON per (arm, step, capacity), skipped if present.

Run:
    cd experiments
    # smoke (1 ckpt x 1 capacity, tiny FM; parts suffixed _smoke):
    modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::smoke
    # full re-cut (both arms, both capacities):
    modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::decomposition_trajectory
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
app = modal.App("rhm-decomposition-trajectory", image=image)

OUT_DIR = f"{DATA_DIR}/rhm_decomposition_trajectory"

# Checkpoint grids: the same log-spaced subsets rhm_ensemble_trajectory used, so the
# re-cut lands on the exact steps the complexodynamics tables report. The wd control
# additionally takes 125000 -- its grid's last checkpoint, which the complexodynamics
# run stopped short of and its caveats flag as a wanted top-up. Free here.
DEFAULT_STEPS = {
    "fmreg": "0,500,4000,12000,30000,80000,160000,300000",
    "wd": "0,500,4000,10000,25000,60000,100000,125000",
}

# capB is REG_FM verbatim (the eval FM of record for this substrate -- continuity
# with the existing parts).
#
# The invariance legs are capB / capM / capQ2: a ~4x capacity range at FIXED DEPTH
# (nl=2), varying only (d_head, mlp_mult). That is the audit's own axis -- it swept
# 25/50/75/100% of a block at n_layer=1 throughout -- and it is the only sweep whose
# spread can be read against the audit's +/-0.01 tolerance.
#
# capQ (nl=1) is retained deliberately as the DEPTH leg, not a capacity leg. The
# first pass of this re-cut used it as the low-capacity arm, which confounded
# capacity with depth; keeping it as its own labelled axis turns that mistake into
# the scope question worth answering -- is beta invariant to how much FM you spend,
# but not to what shape you spend it in? Never pool capQ with the capB/capM/capQ2
# spread.
CAPACITIES = [
    ("capB", dict(dh=REG_FM["dh"], mm=REG_FM["mm"], nl=REG_FM["nl"], nh=REG_FM["nh"])),
    ("capM", dict(dh=8, mm=1.0, nl=2, nh=REG_FM["nh"])),
    ("capQ2", dict(dh=4, mm=0.5, nl=2, nh=REG_FM["nh"])),
    ("capQ", dict(dh=8, mm=1.0, nl=1, nh=REG_FM["nh"])),
]

# The legs that form the capacity-invariance test (fixed depth, varying capacity).
INVARIANCE_CAPS = ["capB", "capM", "capQ2"]


def _arm_dir(key, arm):
    kind, val = arm.split(":")
    if kind == "fmreg":
        return _reg_dir(key, float(val))
    if kind == "wd":
        return _sweep_dir(key, float(val))
    raise ValueError(f"unknown arm kind: {arm}")


def _arm_tag(arm):
    return arm.replace(":", "_").replace(".", "p")


def _gap_tag(src, tgt):
    """Filename fragment for a non-default gap.

    The default gap (REG_SRC->REG_TGT) returns "" so the parts computed before this
    module was gap-parametrized keep their original names and are not recomputed.
    """
    if (src, tgt) == (REG_SRC, REG_TGT):
        return ""
    short = f"{src}to{tgt}".replace("post_block", "b").replace("post_embed", "E")
    return f"_{short}"


def _cap_cfg(cap_label):
    for lbl, cfg in CAPACITIES:
        if lbl == cap_label:
            return cfg
    raise ValueError(f"unknown capacity label: {cap_label}")


@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=32768)
def decompose_ckpt(
    arm: str, checkpoint_path: str, checkpoint_step: int, cap_label: str = "capB",
    tag: str = "",
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    fm_train_steps: int = 8000, fm_lr: float = 1e-3,
    n_eval_sequences: int = 2000, batch_size: int = 64, seed: int = 911,
    src: str = REG_SRC, tgt: str = REG_TGT,
):
    """Train one fresh FM on a frozen checkpoint, then decompose its (A, P).

    src/tgt select the prediction gap. The default is the trajectory gap of record
    (post_embed->post_block6, 7 blocks). A narrower gap raises the FM's capacity
    RELATIVE to the computation it must model, which is the variable the audit's
    capacity-invariance claim was established at and this module's wide-gap run
    falls outside of.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition
    from rhm.residual_decomposition.analyze_summaries import shadow_law

    L = depth
    block_size = s ** L
    device = "cuda"
    key = _tb_key(v, s, L, m)
    cfg = _cap_cfg(cap_label)
    volume.reload()

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())

    corpus = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    train_data = corpus[: int(0.9 * len(corpus))]

    # Held-out eval set sampled fresh from the DGP (seed 999), with the latent traces
    # eta^2 needs. Same generator the legibility/ensemble harnesses use.
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    torch.manual_seed(seed)
    fm = TransformerForwardModel(
        d_model=n_embd, d_head=cfg["dh"], n_head=cfg["nh"],
        n_layer=cfg["nl"], mlp_mult=cfg["mm"], block_size=block_size,
    ).to(device)
    fm_params = sum(p.numel() for p in fm.parameters())
    print(f"=== {arm} step {checkpoint_step} | {cap_label} FM ({fm_params:,}p, "
          f"{fm_params / block_params:.1%} of block) | {src}->{tgt} ===")

    opt = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 2000)
    for step in range(fm_train_steps):
        fm.train()
        ix = torch.randint(len(train_data) - block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([train_data[i:i + block_size] for i in ix]).to(device)
        with torch.no_grad():
            _, _, inter = model(x, return_intermediates=True)
            src_act, tgt_act = inter[src], inter[tgt]
        loss = F.mse_loss(fm(src_act), tgt_act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"  step {step:6d}: mse={float(loss):.6f}")

    # --- collect (A, P) on the held-out trace set ---
    fm.eval()
    A_list, P_list, res_list, coss = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(eval_x), batch_size):
            b = eval_x[i:i + batch_size]
            _, _, inter = model(b, return_intermediates=True)
            pred, tgt_act = fm(inter[src]), inter[tgt]
            A_list.append(tgt_act.float().cpu())
            P_list.append(pred.float().cpu())
            res_list.append((tgt_act - pred).float().cpu())
            coss.append(float(F.cosine_similarity(pred, tgt_act, dim=-1).mean()))

    A_shaped = torch.cat(A_list).numpy()          # (N, T, d)
    P_shaped = torch.cat(P_list).numpy()
    res_shaped = torch.cat(res_list).numpy()
    A = A_shaped.reshape(-1, n_embd)
    P = P_shaped.reshape(-1, n_embd)

    # --- the trusted instrument (naive triple comes along on the same (A, P)) ---
    decomp = full_decomposition(A, P, seed=seed)
    beta, beta_r2 = shadow_law(decomp["geometry"])

    # --- the survivors from the original trajectory work, same residual ---
    nC = res_shaped.shape[0]
    eta2 = _compute_hierarchy_eta2(res_shaped,
                                   [x[:nC] for x in level_features],
                                   [x[:nC] for x in level_rules], s=s, L=L)
    eff_rank_last, top1_last = _compute_effective_rank(res_shaped[:, -1, :], n_embd)

    b, n, g, r = decomp["basic"], decomp["naive"], decomp["geometry"], decomp["repaired"]
    print(f"  rel_residual={b['relative_residual']:.4f}  cosine={b['mean_cosine']:.4f}")
    print(f"  TRUSTED:  beta={beta:.3f} (R2={beta_r2:.3f})  "
          f"R_res_participation={r['R_res_participation']:.1f}  "
          f"frontier_mass={r['frontier_mass']:.4f}")
    print(f"  RETIRED:  naive R_act={n['R_act']:.1f}  R_res={n['R_res']:.1f} "
          f"({n['R_res'] / n_embd * 100:.1f}% of d)  top1_last={top1_last * 100:.1f}%")
    print(f"  eta2_last d4/d5/d6 = {eta2['level_2']['feature_eta2_last']:.3f} / "
          f"{eta2['level_1']['feature_eta2_last']:.3f} / "
          f"{eta2['level_0']['feature_eta2_last']:.3f}")

    result = {
        "arm": arm, "step": checkpoint_step, "key": key,
        "gap": f"{src}->{tgt}",
        "capacity_label": cap_label,
        "fm": {"n_params": fm_params, "block_params": block_params,
               "block_fraction": fm_params / block_params, **cfg},
        "fm_train_steps": fm_train_steps, "n_eval_sequences": n_eval_sequences,
        "fm_cosine_eval": float(sum(coss) / len(coss)),
        "beta": beta, "beta_r2": beta_r2,
        "decomposition": decomp,
        "eta2_residual": eta2,
        "res_effective_rank_last_pct": float(eff_rank_last / n_embd * 100),
        "res_top1_pc_last": float(top1_last),
    }
    parts_dir = f"{OUT_DIR}/parts"
    os.makedirs(parts_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    with open(f"{parts_dir}/{_arm_tag(arm)}_step{checkpoint_step}_{cap_label}"
              f"{_gap_tag(src, tgt)}{suffix}.json", "w") as fp:
        json.dump(result, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=8192)
def decomposition_trajectory(
    arms: str = "fmreg:1.0,wd:0.1", steps: str = "", caps: str = "capB,capM,capQ2,capQ",
    tag: str = "",
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    fm_train_steps: int = 8000, n_eval_sequences: int = 2000,
    src: str = REG_SRC, tgt: str = REG_TGT, max_dop: int = 8,
):
    """Spawn decompose_ckpt per (arm, step, capacity); aggregate + print the arc.

    max_dop caps how many checkpoint jobs are in flight at once. The workspace has a
    concurrent-GPU cap, and spawning the whole grid at once exceeds it -- so jobs are
    launched in waves of max_dop and each wave is drained before the next is spawned.
    Because every job commits its own part before returning, an interrupted run loses
    at most the in-flight wave; relaunching skips everything already on the volume.
    """
    key = _tb_key(v, s, depth, m)
    volume.reload()
    parts_dir = f"{OUT_DIR}/parts"
    os.makedirs(parts_dir, exist_ok=True)
    suffix = f"{_gap_tag(src, tgt)}" + (f"_{tag}" if tag else "")
    cap_labels = [c.strip() for c in caps.split(",") if c.strip()]

    jobs, planned = [], {}
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
            for cap in cap_labels:
                part = f"{parts_dir}/{_arm_tag(arm)}_step{st}_{cap}{suffix}.json"
                if os.path.exists(part):
                    continue
                jobs.append((arm, st, cap, dict(
                    arm=arm, checkpoint_path=avail[st], checkpoint_step=st,
                    cap_label=cap, tag=tag, v=v, m=m, s=s, depth=depth,
                    fm_train_steps=fm_train_steps, n_eval_sequences=n_eval_sequences,
                    src=src, tgt=tgt)))

    n_waves = (len(jobs) + max_dop - 1) // max_dop if jobs else 0
    print(f"=== decomposition_trajectory: {len(jobs)} (arm, ckpt, cap) jobs, "
          f"max_dop={max_dop} -> {n_waves} wave(s) ===")
    for w in range(n_waves):
        wave = jobs[w * max_dop:(w + 1) * max_dop]
        print(f"--- wave {w + 1}/{n_waves}: spawning {len(wave)} jobs "
              f"({', '.join(f'{a}@{s}:{c}' for a, s, c, _ in wave)}) ---")
        handles = [(a, s, c, decompose_ckpt.spawn(**kw)) for a, s, c, kw in wave]
        for a, s, c, h in handles:
            try:
                h.get()
            except Exception as e:
                print(f"  {a} step {s} {c}: FAILED -- {e}")
        volume.reload()

    volume.reload()
    out = {"key": key, "arms": arms, "caps": cap_labels,
           "gap": f"{src}->{tgt}", "results": {}}
    for arm, want in planned.items():
        rows = []
        for st in want:
            byc = {}
            for cap in cap_labels:
                part = f"{parts_dir}/{_arm_tag(arm)}_step{st}_{cap}{suffix}.json"
                if os.path.exists(part):
                    byc[cap] = json.load(open(part))
            if byc:
                rows.append({"step": st, "by_cap": byc})
        out["results"][arm] = rows

        print(f"\n=== {arm} | gap {src}->{tgt} | trusted vs retired ===")
        print(f"{'step':>7} {'cap':>5} {'relRes':>7} {'cos':>6} | {'beta':>6} {'R2':>5} "
              f"{'R_res_part':>10} {'frontMass':>9} | {'naiveRres':>9} {'%d':>5} "
              f"{'top1%':>6} | {'d4eta2':>7} {'d6eta2':>7}")
        for row in rows:
            for cap, r in row["by_cap"].items():
                d = r["decomposition"]
                b, n, rp = d["basic"], d["naive"], d["repaired"]
                e = r["eta2_residual"]
                print(f"{row['step']:>7} {cap:>5} {b['relative_residual']:>7.4f} "
                      f"{b['mean_cosine']:>6.3f} | {r['beta']:>6.3f} {r['beta_r2']:>5.3f} "
                      f"{rp['R_res_participation']:>10.1f} {rp['frontier_mass']:>9.4f} | "
                      f"{n['R_res']:>9.1f} {n['R_res'] / n['d_model'] * 100:>5.1f} "
                      f"{r['res_top1_pc_last'] * 100:>6.1f} | "
                      f"{e['level_2']['feature_eta2_last']:>7.3f} "
                      f"{e['level_0']['feature_eta2_last']:>7.3f}")

        # The per-checkpoint validity check. Only the fixed-depth legs count: pooling
        # the nl=1 depth leg in here is what made the first pass uninterpretable.
        inv = [c for c in INVARIANCE_CAPS if c in cap_labels]
        if len(inv) > 1:
            print(f"\n  capacity-invariance of beta over {'/'.join(inv)} "
                  f"(fixed depth nl=2, ~4x capacity) -- audit tolerance +/-0.01:")
            for row in rows:
                bs = [row["by_cap"][c]["beta"] for c in inv if c in row["by_cap"]]
                rr = [row["by_cap"][c]["decomposition"]["basic"]["relative_residual"]
                      for c in inv if c in row["by_cap"]]
                if len(bs) > 1:
                    spread = max(bs) - min(bs)
                    flag = "  <-- OUT OF TOLERANCE" if spread > 0.01 else ""
                    print(f"    step {row['step']:>7}: beta = "
                          f"{'/'.join(f'{x:.3f}' for x in bs)}  spread={spread:.3f}  "
                          f"(relRes {'/'.join(f'{x:.4f}' for x in rr)}){flag}")
        if "capQ" in cap_labels and "capM" in cap_labels:
            print(f"\n  DEPTH leg (capM nl=2 vs capQ nl=1, matched dh=8/mm=1.0) -- "
                  f"is beta a function of FM shape rather than FM size?")
            for row in rows:
                if "capM" in row["by_cap"] and "capQ" in row["by_cap"]:
                    bm, bq = row["by_cap"]["capM"]["beta"], row["by_cap"]["capQ"]["beta"]
                    print(f"    step {row['step']:>7}: beta nl2={bm:.3f} nl1={bq:.3f}  "
                          f"delta={bm - bq:+.3f}")

    with open(f"{OUT_DIR}/summary{suffix}.json", "w") as fp:
        json.dump(out, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=8192)
def smoke():
    """Fast end-to-end path check: one arm, one late checkpoint, one capacity."""
    return decomposition_trajectory.local(
        arms="fmreg:1.0", steps="300000", caps="capB", tag="smoke",
        fm_train_steps=300, n_eval_sequences=400)
