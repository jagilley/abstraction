"""Weight-decay sweep on the m2 substrate: does stronger WD compress the circuit?

Motivation (RHM_FRONTIER_AND_LEGIBILITY_README, "Knowledge-grok vs circuit-grok" +
"Norm trajectories"): m2 reaches the information ceiling but shows no circuit collapse
(rank flat ~84%, top1 ~5%). The norm-trajectory probe showed WD=0.01 is effectively
ZERO pressure: weight norm GROWS 2x (82.9->164) and is still climbing at 50k, identical
for m2/m4 -> no functional-complexity signal, pure CE margin-inflation. So "no
circuit-grok" was never actually tested.

This sweep applies real WD pressure (0.01 baseline, 0.1, 0.3, 1.0), trains LONG (150k),
and asks the decisive question: as WD forces weight norm DOWN, does effective rank track
it down (compressible -> circuit-grok: rank falls, cosine rises, deep eta2 resolves) or
plateau at ~84% (irreducible -> RHM's deep inverse is a structureless random-table
lookup; the "algebraic-compressibility-not-description-length" thesis)?

Guards (from the README): gate every circuit-metric read on "knowledge still at BP"
(per-level last-token probe accuracy, best-over-blocks) so compressed != never-learned;
watch that high WD doesn't drop effective capacity below the depth m2 needs for the root.

Metrics per (wd, checkpoint):
  - weight norm (total + per-group) and per-layer activation norm  [rhm_norm_trajectory]
  - FM cosine / residual effective-rank / top1-PC / per-level feature_eta2  [legibility FMs]
  - per-level knowledge probe accuracy (the BP gate)  [thread_b-style, here]

PREEMPTION ROBUSTNESS (Modal A10G preempts with some regularity):
  - train_wd maintains a resumable `latest.pt` (model + optimizer + step + history) on the
    volume, committed every `commit_interval` steps. A preempted call is auto-restarted by
    Modal "with the same input"; on restart it finds latest.pt and RESUMES from it instead
    of step 0. Milestone `ckpt_step*.pt` (model-only) are written for analysis.
  - analyze_wd_ckpt commits each (wd, checkpoint) result to .../parts/ before returning and
    is SKIPPED if its part already exists -> the analysis fan-out is fully resumable and the
    orchestrator never holds results only in memory.

Run:
    cd experiments && modal run --detach -m rhm.rhm_wd_sweep::wd_sweep
    # analysis only (during / after training, from saved checkpoints):
    cd experiments && modal run --detach -m rhm.rhm_wd_sweep::analyze_wd
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
# Pure helpers reused verbatim from the legibility experiment (identical DGP/metrics).
from rhm.rhm_fm_legibility import (
    _tb_key, _block_idx, _generate_with_traces, _compute_hierarchy_eta2,
    _compute_effective_rank, _ensure_corpus_distinct,
)
from rhm.rhm_norm_trajectory import _weight_norms

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-wd-sweep", image=image)

MODEL_TAG = "8L8H256D"
WD_DEFAULT = "0.01,0.1,0.3,1.0"

# Analysis FM grid: the two primary E->b6 8H (matched-head) configs that bracket the
# meaningful cosine band [0.90,0.99]; pick the in-band one per checkpoint in analysis.
# (Head-invariance + b4/b0 gaps already settled in the legibility run; omitted here.)
WD_FM_GRID = [
    {"src": "post_embed", "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 1.0, "label": "E->b6 8H capA"},
    {"src": "post_embed", "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 2.0, "label": "E->b6 8H capB"},
]


def _wd_tag(wd):
    return str(wd).replace(".", "p").replace("-", "m")


def _sweep_dir(key, wd):
    return f"{DATA_DIR}/{key}/wd_sweep_{MODEL_TAG}_wd{_wd_tag(wd)}"


# --------------------------------------------------------------------------
# Knowledge probe (BP gate): per-level last-token linear probe, best-over-blocks
# --------------------------------------------------------------------------

def _probe_knowledge(model, eval_x, level_features, s, L, n_embd, v, n_layer, device,
                     probe_steps=400, split=0.7, lr=0.05):
    """For each hierarchy level (d{L-ell}), train a last-token linear classifier on
    each block's activation -> true latent feature id, report best-over-blocks held-out
    accuracy. Confirms whether knowledge is preserved at BP under stronger WD."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    seq_len = s ** L
    bs = 256
    acts = {bl: [] for bl in range(n_layer)}
    with torch.no_grad():
        for i in range(0, eval_x.shape[0], bs):
            _, _, inter = model(eval_x[i:i + bs], return_intermediates=True)
            for bl in range(n_layer):
                acts[bl].append(inter[f"post_block{bl}"][:, -1, :])
    acts = {bl: torch.cat(acts[bl], 0) for bl in range(n_layer)}

    N = eval_x.shape[0]
    g = torch.Generator(device=device).manual_seed(0)
    perm = torch.randperm(N, generator=g, device=device)
    tr, te = perm[:int(split * N)], perm[int(split * N):]

    out = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        last_anc = (seq_len - 1) // s_power
        lab = torch.from_numpy(level_features[ell][:, last_anc].astype(np.int64)).to(device)
        ytr, yte = lab[tr], lab[te]
        best_acc, best_bl = 0.0, -1
        for bl in range(n_layer):
            X = acts[bl]
            mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            W = torch.zeros(n_embd, v, device=device, requires_grad=True)
            b = torch.zeros(v, device=device, requires_grad=True)
            opt = torch.optim.Adam([W, b], lr=lr)
            for _ in range(probe_steps):
                opt.zero_grad()
                F.cross_entropy(Xtr @ W + b, ytr).backward()
                opt.step()
            with torch.no_grad():
                acc = float(((Xte @ W + b).argmax(1) == yte).float().mean())
            if acc > best_acc:
                best_acc, best_bl = acc, bl
        out[f"d{L - ell}"] = {"acc": best_acc, "best_block": best_bl, "chance": 1.0 / v}
    return out


def _activation_norms(model, fixed_batches, n_layer):
    import numpy as np
    import torch
    keys = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    rms = {k: [] for k in keys}
    last = {k: [] for k in keys}
    with torch.no_grad():
        for xb in fixed_batches:
            _, _, inter = model(xb, return_intermediates=True)
            for k in keys:
                a = inter[k]
                rms[k].append(float(a.pow(2).mean(-1).sqrt().mean()))
                last[k].append(float(a[:, -1, :].pow(2).sum(-1).sqrt().mean()))
    return {k: {"rms": float(np.mean(rms[k])), "last_l2": float(np.mean(last[k]))} for k in keys}


# --------------------------------------------------------------------------
# Phase 1: robust resumable training at one WD value
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=43200, memory=32768)
def train_wd(
    wd: float, v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    n_steps: int = 150000, batch_size: int = 64, lr: float = 3e-4, seed: int = 42,
    eval_interval: int = 250, commit_interval: int = 2500,
):
    import numpy as np
    import torch
    from rhm.model import GPT

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

    save_dir = _sweep_dir(key, wd)
    os.makedirs(save_dir, exist_ok=True)
    # dense early (low levels learn fast) + coverage through saturation, scaled to n_steps
    fracs = [0.0, 0.0017, 0.0033, 0.0067, 0.013, 0.027, 0.04, 0.067, 0.10,
             0.167, 0.267, 0.40, 0.533, 0.667, 0.833, 1.0]
    ckpt_steps = sorted(set(round(f * n_steps / eval_interval) * eval_interval for f in fracs))

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    latest_path = os.path.join(save_dir, "latest.pt")
    start_step = 0
    history = {"val_loss": [], "weight_norm": []}
    saved = []
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        history = ck["history"]
        saved = ck["saved"]
        print(f"=== RESUME wd={wd} ({key}) from step {start_step} (ckpts so far: {len(saved)}) ===")
    else:
        print(f"=== START wd={wd} ({key}) fresh  n_steps={n_steps}  ckpts={ckpt_steps} ===")

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def commit(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "history": history, "saved": saved}, latest_path)
        with open(os.path.join(save_dir, "training_info.json"), "w") as f:
            json.dump({"key": key, "wd": wd, "model_tag": MODEL_TAG, "n_steps": n_steps,
                       "last_step": step, "checkpoints": saved, "history": history},
                      f, indent=2, cls=NumpyEncoder)
        volume.commit()

    for step in range(start_step, n_steps + 1):
        if step > 0:
            model.train()
            x, y = get_batch(train_data)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(*get_batch(val_data))[1]) for _ in range(10)]))
            wnorm = float(sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()) ** 0.5)
            history["val_loss"].append((step, vl))
            history["weight_norm"].append((step, wnorm))
            if step in ckpt_steps and not any(sv["step"] == step for sv in saved):
                p = os.path.join(save_dir, f"ckpt_step{step}.pt")
                torch.save(model.state_dict(), p)
                saved.append({"step": step, "val_loss": vl, "weight_norm": wnorm, "path": p})
                print(f"  step {step:6d}: val={vl:.4f} ||W||={wnorm:.1f} [CKPT]")
        if step > 0 and step % commit_interval == 0:
            commit(step)

    commit(n_steps)
    print(f"=== DONE wd={wd}: {len(saved)} checkpoints, final val={history['val_loss'][-1][1]:.4f} ===")
    return {"key": key, "wd": wd, "checkpoints": saved, "history": history}


# --------------------------------------------------------------------------
# Phase 2: analyze one (wd, checkpoint) -- norms + FM legibility + knowledge gate
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=32768)
def analyze_wd_ckpt(
    wd: float, key: str, checkpoint_path: str, checkpoint_step: int,
    v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, n_tokens: int = 20_000_000,
    fm_train_steps: int = 8000, fm_lr: float = 1e-3, n_eval_sequences: int = 8000,
    batch_size: int = 64, fm_seed: int = 137, n_act_batches: int = 8,
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

    # weight + activation norms
    torch.manual_seed(0)
    fixed_batches = [get_batch(val_data)[0] for _ in range(n_act_batches)]
    weight = _weight_norms(model)
    act = _activation_norms(model, fixed_batches, n_layer)

    # eval set with latent traces (same generator as legibility/thread_b)
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    # knowledge gate
    probe = _probe_knowledge(model, eval_x, level_features, s, L, n_embd, v, n_layer, device)
    probe_str = " ".join(f"{lvl}:{probe[lvl]['acc']:.2f}" for lvl in
                         [f"d{L - e}" for e in range(L)])
    print(f"=== analyze wd={wd} {key} step {checkpoint_step}  val={main_val_loss:.4f}")
    print(f"    knowledge(best-block acc, chance={1.0/v:.3f}): {probe_str}")

    cfg_results = []
    for cfg in WD_FM_GRID:
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
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()

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

    result = {"wd": wd, "step": checkpoint_step, "main_val_loss": main_val_loss,
              "weight": weight, "act": act, "knowledge": probe, "configs": cfg_results}
    parts_dir = f"{DATA_DIR}/rhm_wd_sweep/parts"
    os.makedirs(parts_dir, exist_ok=True)
    with open(f"{parts_dir}/wd{_wd_tag(wd)}_step{checkpoint_step}.json", "w") as fp:
        json.dump(result, fp, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def wd_sweep(n_steps: int = 150000, wds: str = WD_DEFAULT):
    """Phase 1: train the m2 substrate at each WD (parallel, each resumable)."""
    wd_list = [float(x) for x in wds.split(",")]
    print(f"=== WD sweep (m2 substrate) === wds={wd_list}  n_steps={n_steps}")
    handles = [(wd, train_wd.spawn(wd=wd, n_steps=n_steps)) for wd in wd_list]
    results = {}
    for wd, h in handles:
        try:
            results[wd] = h.get()
            print(f"  wd={wd}: trained {len(results[wd]['checkpoints'])} checkpoints")
        except Exception as e:
            print(f"  wd={wd}: FAILED -- {e}")
    return {"wds": wd_list, "n_steps": n_steps}


@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=16384)
def analyze_wd(wds: str = WD_DEFAULT, v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
               fm_train_steps: int = 8000):
    """Phase 2: per (wd, checkpoint) compute norms + FM legibility + knowledge gate.
    Resumable: skips any (wd, step) whose part already exists; aggregates from parts."""
    wd_list = [float(x) for x in wds.split(",")]
    key = _tb_key(v, s, depth, m)
    L = depth
    volume.reload()
    parts_dir = f"{DATA_DIR}/rhm_wd_sweep/parts"
    os.makedirs(parts_dir, exist_ok=True)

    pending = []
    for wd in wd_list:
        info_path = os.path.join(_sweep_dir(key, wd), "training_info.json")
        if not os.path.exists(info_path):
            print(f"  wd={wd}: no training_info yet, skipping")
            continue
        info = json.load(open(info_path))
        for c in info["checkpoints"]:
            part = f"{parts_dir}/wd{_wd_tag(wd)}_step{c['step']}.json"
            if os.path.exists(part):
                continue
            h = analyze_wd_ckpt.spawn(
                wd=wd, key=key, checkpoint_path=c["path"], checkpoint_step=c["step"],
                v=v, m=m, s=s, depth=depth, fm_train_steps=fm_train_steps)
            pending.append((wd, c["step"], h))
    print(f"=== analyze_wd: {len(pending)} (wd,checkpoint) jobs to run ===")
    for wd, step, h in pending:
        try:
            h.get()
        except Exception as e:
            print(f"  wd={wd} step {step}: FAILED -- {e}")

    # aggregate from parts and print compact per-WD trajectory
    out = {"key": key, "wds": wd_list, "results": {}}
    for wd in wd_list:
        info_path = os.path.join(_sweep_dir(key, wd), "training_info.json")
        if not os.path.exists(info_path):
            continue
        info = json.load(open(info_path))
        rows = []
        for c in info["checkpoints"]:
            part = f"{parts_dir}/wd{_wd_tag(wd)}_step{c['step']}.json"
            if os.path.exists(part):
                rows.append(json.load(open(part)))
        rows.sort(key=lambda r: r["step"])
        out["results"][f"wd{_wd_tag(wd)}"] = rows
        print(f"\n=== wd={wd} ===")
        print(f"{'step':>6} {'val':>6} {'Wnorm':>7} {'cos':>6} {'rank%':>6} {'top1%':>5} "
              f"{'d6acc':>6} {'d4η²':>6} {'d6η²':>6}")
        for r in rows:
            cr = max(r["configs"], key=lambda c: c["fm_cosine"] if 0.90 <= c["fm_cosine"] <= 0.99 else -1)
            e = cr["hierarchy_eta2"]
            print(f"{r['step']:>6} {r['main_val_loss']:>6.3f} {r['weight']['total']['norm']:>7.1f} "
                  f"{cr['fm_cosine']:>6.3f} {cr['effective_rank_pct']:>6.1f} {cr['top1_pc']*100:>5.1f} "
                  f"{r['knowledge']['d6']['acc']:>6.2f} "
                  f"{e['level_2']['feature_eta2_last']:>6.3f} {e['level_0']['feature_eta2_last']:>6.3f}")

    save_dir = f"{DATA_DIR}/rhm_wd_sweep"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
