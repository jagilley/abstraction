"""FM-as-regularizer on the m2 substrate: does structured FM-predictability pressure
compress the *functional* circuit where weight decay cannot?

Motivation (RHM_FRONTIER_AND_LEGIBILITY_README, Exp 3 + next-steps #3):
  The WD sweep established the bar. On the m2 substrate (plain NTP groks the whole
  hierarchy to the root, d6 = BP = 0.93), generic L2 pressure compresses *weight norm*
  3.2x at zero knowledge cost but barely touches *functional* residual rank (89->77%,
  <=5% at matched root knowledge). The deep RHM inverse has no algebraic handle for
  ||W||^2-minimization to collapse. BUT: a 187K-param FM captures 91-97% of the DGP
  composition function (PER_LEVEL_LOSS) -- a constructive witness that a LOW-functional-
  complexity approximation of the gap computation EXISTS. So the rank ~= 77% WD floor is
  an L2-pressure floor, not an information floor.

  This experiment tests the payoff: co-train an FM alongside the main model and add a
  structured regularization term -- lambda * MSE(actual_target, FM_pred), gradient
  through the MAIN MODEL only (open-loop; NO injection, NO distillation -> NO
  self-knowledge apparatus). The FM defines "the DGP-aligned function"; the term presses
  the main model's computation onto the FM-predictable manifold. Question: does this beat
  the WD floor -- residual rank < 77% at d6 ~= BP -- where generic L2 provably cannot?

  This isolates MECHANISM (a) "structured regularization pressure" from MECHANISM (b)
  "self-knowledge" (the closed-loop injection/distillation that compounds on MNIST). If
  the open-loop term already compresses functional rank below the WD floor at preserved
  root knowledge, self-knowledge is NOT load-bearing for functional simplification.

Design:
  - Substrate: m2 (v16/m2), 8L/8H/256D, IDENTICAL distinct-rule DGP, 150k steps. So
    each lambda run is a drop-in delta over the WD sweep's wd=0.1 row (= the lambda=0
    baseline at matched WD); no separate control run needed.
  - Regularizing FM (co-trained, own optimizer): matched-head 8H/16d/mlp2 (~791K, capB),
    src=post_embed -> tgt=post_block6 -- the gap covering the whole bottom-up hierarchy
    build (blocks 0-5), staying out of output-prep blocks 6-7. capB sits in the
    meaningful cosine band (~0.92 at convergence in the legibility run).
  - lambda_local sweep {0.03, 0.1, 0.3} centered on the prior-used 0.1 (fm_wt_ll).
    Aux losses are sensitive; 3 points bracket an order of magnitude. wd fixed at 0.1
    (the README's new recommended default for this substrate).
  - lambda warmup: FM tracks the model for `reg_warmup` steps at lambda=0 (cold-start
    guard -- regularizing toward a random FM would push toward garbage), then linearly
    ramp to target over `reg_ramp`.

Guards / anti-circularity (no extra training runs; just extra read-outs):
  - KNOWLEDGE GATE (the headline denominator): per-level last-token BP probe accuracy at
    every checkpoint. rank only counts at matched d6 ~= 0.93. A collapse that destroys
    deep knowledge (the WD=1.0 failure mode) reads as "never-learned", not compression.
  - The eval FM is trained with a DIFFERENT seed than the regularizing FM, AND we measure
    rank on a NON-regularized gap (b0->b4) too. If compression only shows on the exact
    regularized gap it's leakage; if it shows on b0->b4 the function genuinely simplified.
  - FM-FREE readout: effective rank of post_block6 activations themselves (no FM in the
    loop) -- the honest, leakage-proof complexity metric. Argue from convergence across
    all three.

Preemption-robust exactly like rhm_wd_sweep (resumable latest.pt every commit_interval;
per-(lambda,ckpt) analysis parts, skipped if present).

Run:
    cd experiments && modal run --detach -m rhm.rhm_fm_regularizer::fm_reg_sweep
    cd experiments && modal run --detach -m rhm.rhm_fm_regularizer::analyze_reg
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_fm_legibility import (
    _tb_key, _block_idx, _generate_with_traces, _compute_hierarchy_eta2,
    _compute_effective_rank, _ensure_corpus_distinct,
)
from rhm.rhm_norm_trajectory import _weight_norms
from rhm.rhm_wd_sweep import _probe_knowledge, _activation_norms

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-fm-regularizer", image=image)

MODEL_TAG = "8L8H256D"
LAM_DEFAULT = "0.03,0.1,0.3"
WD_FIXED = 0.1  # README's new default for this substrate; lambda=0 baseline = wd_sweep wd=0.1

# Regularizing FM (co-trained during training): the gap covering the bottom-up build.
REG_SRC, REG_TGT = "post_embed", "post_block6"
REG_FM = {"nh": 8, "dh": 16, "mm": 2.0, "nl": 2}  # 8H/16d/mlp2 ~791K (capB), in-band cosine

# Analysis FM grid: the regularized gap at two capacities (pick in-band cosine) + one
# NON-regularized gap (b0->b4) to test whether compression generalizes beyond the exact
# regularized target (leakage check).
REG_ANALYSIS_GRID = [
    {"src": "post_embed",  "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 1.0, "label": "E->b6 8H capA (reg gap)"},
    {"src": "post_embed",  "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 2.0, "label": "E->b6 8H capB (reg gap)"},
    {"src": "post_block0", "tgt": "post_block4", "nh": 8, "dh": 16, "mm": 1.0, "label": "b0->b4 8H capA (UNreg gap)"},
]


def _lam_tag(lam):
    return str(lam).replace(".", "p").replace("-", "m")


def _reg_dir(key, lam):
    return f"{DATA_DIR}/{key}/fmreg_{MODEL_TAG}_wd{_lam_tag(WD_FIXED)}_lam{_lam_tag(lam)}"


# --------------------------------------------------------------------------
# Phase 1: resumable training of the main model with a co-trained FM regularizer
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def train_fm_reg(
    lam: float, wd: float = WD_FIXED, v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    n_steps: int = 150000, batch_size: int = 64, lr: float = 3e-4, fm_lr: float = 1e-3,
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

    save_dir = _reg_dir(key, lam)
    os.makedirs(save_dir, exist_ok=True)
    fracs = [0.0, 0.0017, 0.0033, 0.0067, 0.013, 0.027, 0.04, 0.067, 0.10,
             0.167, 0.267, 0.40, 0.533, 0.667, 0.833, 1.0]
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
    history = {"val_loss": [], "weight_norm": [], "reg_loss": [], "fm_cosine": []}
    saved = []
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        fm.load_state_dict(ck["fm"]); opt_fm.load_state_dict(ck["opt_fm"])
        start_step = ck["step"] + 1
        history = ck["history"]; saved = ck["saved"]
        print(f"=== RESUME lam={lam} ({key}) from step {start_step} (ckpts: {len(saved)}) ===")
    else:
        print(f"=== START lam={lam} wd={wd} ({key}) fresh  n_steps={n_steps}  reg_gap={REG_SRC}->{REG_TGT} ===")

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
            json.dump({"key": key, "lam": lam, "wd": wd, "model_tag": MODEL_TAG,
                       "n_steps": n_steps, "reg_src": REG_SRC, "reg_tgt": REG_TGT,
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
            src_act, tgt_act = inter[REG_SRC], inter[REG_TGT]

            # --- Main update: NTP + lambda * FM-predictability (grad through MAIN only) ---
            for p in fm.parameters():
                p.requires_grad_(False)
            reg = F.mse_loss(fm(src_act), tgt_act)
            loss_main = ntp_loss + lam_t * reg
            opt.zero_grad(set_to_none=True)
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for p in fm.parameters():
                p.requires_grad_(True)
            last_reg = float(reg.detach())

            # --- FM update: track the (updated) model; detach so no grad to main ---
            fm_loss = F.mse_loss(fm(src_act.detach()), tgt_act.detach())
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
                pred = fm(inter[REG_SRC])
                fmcos = float(F.cosine_similarity(pred, inter[REG_TGT], dim=-1).mean())
            wnorm = float(sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()) ** 0.5)
            history["val_loss"].append((step, vl))
            history["weight_norm"].append((step, wnorm))
            history["reg_loss"].append((step, last_reg))
            history["fm_cosine"].append((step, fmcos))
            if step in ckpt_steps and not any(sv["step"] == step for sv in saved):
                p = os.path.join(save_dir, f"ckpt_step{step}.pt")
                torch.save(model.state_dict(), p)
                saved.append({"step": step, "val_loss": vl, "weight_norm": wnorm,
                              "fm_cosine": fmcos, "lam_t": lam_t, "path": p})
                print(f"  step {step:6d}: val={vl:.4f} ||W||={wnorm:.1f} regMSE={last_reg:.3f} "
                      f"fmcos={fmcos:.4f} lam_t={lam_t:.3f} [CKPT]")
        if step > 0 and step % commit_interval == 0:
            commit(step)

    commit(n_steps)
    print(f"=== DONE lam={lam}: {len(saved)} checkpoints, final val={history['val_loss'][-1][1]:.4f} ===")
    return {"key": key, "lam": lam, "checkpoints": saved, "history": history}


# --------------------------------------------------------------------------
# Phase 2: analyze one (lambda, checkpoint) -- knowledge gate + FM legibility +
#          FM-free activation rank (the leakage-proof complexity readout)
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=32768)
def analyze_reg_ckpt(
    lam: float, key: str, checkpoint_path: str, checkpoint_step: int,
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    fm_train_steps: int = 8000, fm_lr: float = 1e-3, n_eval_sequences: int = 8000,
    batch_size: int = 64, fm_seed: int = 911, n_act_batches: int = 8,
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
    main_block_params = sum(p.numel() for p in model.transformer.h[0].parameters())

    corpus = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data, val_data = corpus[:split], corpus[split:]

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    with torch.no_grad():
        main_val_loss = float(np.mean([float(model(*get_batch(val_data))[1]) for _ in range(20)]))

    torch.manual_seed(0)
    fixed_batches = [get_batch(val_data)[0] for _ in range(n_act_batches)]
    weight = _weight_norms(model)
    act = _activation_norms(model, fixed_batches, n_layer)

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    # knowledge gate (the headline denominator)
    probe = _probe_knowledge(model, eval_x, level_features, s, L, n_embd, v, n_layer, device)
    probe_str = " ".join(f"{lvl}:{probe[lvl]['acc']:.2f}" for lvl in [f"d{L - e}" for e in range(L)])

    # FM-FREE complexity readout: effective rank of post_block6 last-token activations
    # (no FM in the loop -> leakage-proof).
    act_list = []
    with torch.no_grad():
        for i in range(0, len(eval_x), batch_size):
            bx = eval_x[i:i + batch_size]
            if bx.shape[0] < 2:
                continue
            _, _, inter = model(bx, return_intermediates=True)
            act_list.append(inter[REG_TGT][:, -1, :].cpu().numpy())
    act_mat = np.concatenate(act_list, axis=0)
    act_eff_rank, act_top1 = _compute_effective_rank(act_mat, n_embd)
    act_rank = {"target": REG_TGT, "effective_rank_pct": float(act_eff_rank / n_embd * 100),
                "top1_pc": float(act_top1)}

    print(f"=== analyze lam={lam} {key} step {checkpoint_step}  val={main_val_loss:.4f}")
    print(f"    knowledge(best-block acc, chance={1.0/v:.3f}): {probe_str}")
    print(f"    FM-FREE act rank({REG_TGT})={act_rank['effective_rank_pct']:.1f}% top1={act_top1*100:.1f}%")

    cfg_results = []
    for cfg in REG_ANALYSIS_GRID:
        src, tgt = cfg["src"], cfg["tgt"]
        torch.manual_seed(fm_seed)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=cfg["dh"], n_head=cfg["nh"],
            n_layer=cfg.get("nl", 2), mlp_mult=cfg["mm"], block_size=block_size,
        ).to(device)
        fm_params = sum(p.numel() for p in fm.parameters())
        gap_blocks = _block_idx(tgt) - _block_idx(src)
        cap_pct = 100.0 * fm_params / (gap_blocks * main_block_params)
        opt = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)

        for st in range(fm_train_steps):
            fm.train()
            x, y = get_batch(train_data)
            with torch.no_grad():
                _, _, inter = model(x, y, return_intermediates=True)
                source, target = inter[src], inter[tgt]
            loss = F.mse_loss(fm(source), target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0); opt.step()

        fm.eval()
        res_list, cos_list, norm_list = [], [], []
        with torch.no_grad():
            for i in range(0, len(eval_x), batch_size):
                bx = eval_x[i:i + batch_size]
                if bx.shape[0] < 2:
                    continue
                _, _, inter = model(bx, return_intermediates=True)
                source, target = inter[src], inter[tgt]
                pred = fm(source)
                res_list.append((target - pred).cpu().numpy())
                cos_list.append(float(F.cosine_similarity(pred, target, dim=-1).mean()))
                norm_list.append(float((target - pred).norm(dim=-1).mean()))
        residuals = np.concatenate(res_list, axis=0)
        nC = residuals.shape[0]
        eff_rank, top1 = _compute_effective_rank(residuals.reshape(-1, n_embd), n_embd)
        eta2 = _compute_hierarchy_eta2(residuals, [lf[:nC] for lf in level_features],
                                       [lr[:nC] for lr in level_rules], s=s, L=L)
        cos = float(np.mean(cos_list))
        cfg_results.append({
            "label": cfg["label"], "src": src, "tgt": tgt,
            "nh": cfg["nh"], "dh": cfg["dh"], "mm": cfg["mm"],
            "fm_params": fm_params, "gap_blocks": gap_blocks, "capacity_pct": cap_pct,
            "fm_cosine": cos, "residual_norm": float(np.mean(norm_list)),
            "effective_rank_pct": float(eff_rank / n_embd * 100), "top1_pc": float(top1),
            "hierarchy_eta2": eta2,
        })
        feat_last = " ".join(f"d{L - e}:{eta2[f'level_{e}']['feature_eta2_last']:.3f}" for e in range(L))
        print(f"    [{cfg['label']}] cap={cap_pct:.0f}% cos={cos:.4f} "
              f"rank={eff_rank / n_embd * 100:.0f}% top1={top1*100:.0f}%  {feat_last}")

    result = {"lam": lam, "step": checkpoint_step, "main_val_loss": main_val_loss,
              "weight": weight, "act": act, "act_rank": act_rank,
              "knowledge": probe, "configs": cfg_results}
    parts_dir = f"{DATA_DIR}/rhm_fm_regularizer/parts"
    os.makedirs(parts_dir, exist_ok=True)
    with open(f"{parts_dir}/lam{_lam_tag(lam)}_step{checkpoint_step}.json", "w") as fp:
        json.dump(result, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def fm_reg_sweep(n_steps: int = 150000, lams: str = LAM_DEFAULT):
    """Phase 1: train the m2 substrate at each lambda (parallel, each resumable)."""
    lam_list = [float(x) for x in lams.split(",")]
    print(f"=== FM-regularizer sweep (m2 substrate) === lams={lam_list}  wd={WD_FIXED}  "
          f"n_steps={n_steps}  reg_gap={REG_SRC}->{REG_TGT}")
    handles = [(lam, train_fm_reg.spawn(lam=lam, n_steps=n_steps)) for lam in lam_list]
    for lam, h in handles:
        try:
            r = h.get()
            print(f"  lam={lam}: trained {len(r['checkpoints'])} checkpoints")
        except Exception as e:
            print(f"  lam={lam}: FAILED -- {e}")
    return {"lams": lam_list, "n_steps": n_steps}


@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def analyze_reg(lams: str = LAM_DEFAULT, v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
                fm_train_steps: int = 8000):
    """Phase 2: per (lambda, checkpoint) compute knowledge gate + FM legibility +
    FM-free activation rank. Resumable: skips parts that exist; aggregates from parts."""
    lam_list = [float(x) for x in lams.split(",")]
    key = _tb_key(v, s, depth, m)
    L = depth
    volume.reload()
    parts_dir = f"{DATA_DIR}/rhm_fm_regularizer/parts"
    os.makedirs(parts_dir, exist_ok=True)

    pending = []
    for lam in lam_list:
        info_path = os.path.join(_reg_dir(key, lam), "training_info.json")
        if not os.path.exists(info_path):
            print(f"  lam={lam}: no training_info yet, skipping")
            continue
        info = json.load(open(info_path))
        for c in info["checkpoints"]:
            part = f"{parts_dir}/lam{_lam_tag(lam)}_step{c['step']}.json"
            if os.path.exists(part):
                continue
            h = analyze_reg_ckpt.spawn(
                lam=lam, key=key, checkpoint_path=c["path"], checkpoint_step=c["step"],
                v=v, m=m, s=s, depth=depth, fm_train_steps=fm_train_steps)
            pending.append((lam, c["step"], h))
    print(f"=== analyze_reg: {len(pending)} (lam,checkpoint) jobs to run ===")
    for lam, step, h in pending:
        try:
            h.get()
        except Exception as e:
            print(f"  lam={lam} step {step}: FAILED -- {e}")

    out = {"key": key, "lams": lam_list, "wd": WD_FIXED, "reg_gap": f"{REG_SRC}->{REG_TGT}",
           "results": {}}
    for lam in lam_list:
        info_path = os.path.join(_reg_dir(key, lam), "training_info.json")
        if not os.path.exists(info_path):
            continue
        info = json.load(open(info_path))
        rows = []
        for c in info["checkpoints"]:
            part = f"{parts_dir}/lam{_lam_tag(lam)}_step{c['step']}.json"
            if os.path.exists(part):
                rows.append(json.load(open(part)))
        rows.sort(key=lambda r: r["step"])
        out["results"][f"lam{_lam_tag(lam)}"] = rows
        print(f"\n=== lam={lam} (wd={WD_FIXED}) ===")
        print(f"{'step':>6} {'val':>6} {'Wnorm':>7} {'regCos':>6} {'regRk%':>6} "
              f"{'unrgRk%':>7} {'actRk%':>6} {'d6acc':>6} {'d4η²':>6} {'d6η²':>6}")
        for r in rows:
            reg = [c for c in r["configs"] if "reg gap" in c["label"]]
            unreg = next((c for c in r["configs"] if "UNreg" in c["label"]), None)
            cr = max(reg, key=lambda c: c["fm_cosine"] if 0.90 <= c["fm_cosine"] <= 0.99 else -1)
            e = cr["hierarchy_eta2"]
            print(f"{r['step']:>6} {r['main_val_loss']:>6.3f} {r['weight']['total']['norm']:>7.1f} "
                  f"{cr['fm_cosine']:>6.3f} {cr['effective_rank_pct']:>6.1f} "
                  f"{(unreg['effective_rank_pct'] if unreg else float('nan')):>7.1f} "
                  f"{r['act_rank']['effective_rank_pct']:>6.1f} "
                  f"{r['knowledge']['d6']['acc']:>6.2f} "
                  f"{e['level_2']['feature_eta2_last']:>6.3f} {e['level_0']['feature_eta2_last']:>6.3f}")

    save_dir = f"{DATA_DIR}/rhm_fm_regularizer"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
