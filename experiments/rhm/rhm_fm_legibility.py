"""FM residual-legibility over training: does the forward model's residual become
hierarchy-legible while the base model is *actively learning* each level, then go
illegible again once that level is learned (regular/FM-predictable)?

Hypothesis (the "traveling bump"): for each hierarchy level, the residual's
hierarchy-conditioning (η² of residual on the true latent feature identity) RISES
while the level is being learned irregularly and FALLS once it's grokked. The
decisive contrast is v16/m2 (plain NTP groks the whole tree to the root — occupancy
sweep: d6≈0.93=BP) vs v16/m4 (NTP stalls ~d4): m2 should show the full bottom-up
bump sequence resolving to the root; m4's low-level bumps should resolve but its
deep (d5/root) bumps should NEVER appear (never computed). This is the temporal,
RHM version of the A2A "amplifier, not source" test: FM-residual legibility should
track what the base model has actually extracted.

Design (grounded in prior FM-on-RHM work; subagent synthesis + verified vs code):
  - Substrate: 8L/8H/256D GPT (~6.34M; per-block 12d²+13d = 789,760), distinct-rule
    DGP (rule_seed=0, seq_seed=1) — IDENTICAL DGP to rhm_thread_b / the occupancy sweep.
  - Two gap SOURCES (user's call: run both): post_embed (whole bottom-up build inside
    the gap → low-level bumps visible as a positive control) and post_block0 (deep-
    focused; d1–d3 muted because already in the FM's input). Two TARGETS: post_block4
    (hierarchy "built by ~block4" under NTP) and post_block6 (margin to capture m2's
    deeper root computation). post_block6 stays out of the final output-prep blocks.
  - Matched-HEAD FM (n_head=8) primary — prior work (RESIDUAL_RANK) shows a head-count
    MISMATCH (1H-on-8H) *concentrates* the residual (rank 77–86% vs matched 90–96%),
    which would masquerade as hierarchy legibility under η². Two capacities (A≈14%,
    B≈20–25% of modeled-block params) bracket the meaningful cosine band [0.90,0.99];
    pick the in-band one per condition in analysis. Plus a fixed-param 1H twin of B
    (1H/128d/mlp2 = 8H/16d/mlp2 = 791K) to prove the η² conclusion is head-invariant —
    the control the regime_trajectory precedent (1H only) never ran.
  - Metric: per-level feature_eta2_last (last token = max causal context, matches how
    thread_b probes), with rank / top1-PC / cosine alongside. GATE η² on cosine∈[0.90,
    0.99] in analysis; treat earliest high-cosine checkpoints as the overshoot baseline;
    read each level's η² against its probe accuracy + BP ceiling (so m4's absent root
    bump = "never computed", not "FM captured it").

Adapts rhm_regime_trajectory.py (the direct precedent: FM residual structure over
training). Pure helpers copied verbatim per the repo's duplication-over-coupling norm.

Run:
    modal run --detach -m rhm.rhm_fm_legibility::run_legibility
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
app = modal.App("rhm-fm-legibility", image=image)


def _tb_key(v, s, L, m):
    """Match rhm_thread_b's key so we reuse the SAME distinct-rule corpus/rules."""
    return f"{setting_key(v, s, L, m)}_distinct"


def _block_idx(name):
    """post_embed -> -1 (before block 0); post_blockN -> N. So gap_blocks =
    idx(to) - idx(from): post_embed->post_block4 = 5 blocks; post_block0->post_block4 = 4."""
    return -1 if name == "post_embed" else int(name.replace("post_block", ""))


# --------------------------------------------------------------------------
# Pure numpy helpers (copied from rhm_regime_trajectory.py; CPU)
# --------------------------------------------------------------------------

def _generate_with_traces(rules, n_sequences, seed=999):
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features = []
    level_rules = []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        next_level = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            next_level[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = next_level
    level_features.append(current.copy())  # leaves
    return current, level_features, level_rules


def _eta2_between(data, labels, grand_mean):
    import numpy as np
    ss = 0.0
    for val in np.unique(labels):
        mask = labels == val
        group_mean = data[mask].mean(axis=0)
        ss += float(mask.sum()) * float(np.sum((group_mean - grand_mean) ** 2))
    return ss


def _compute_hierarchy_eta2(residuals_np, level_features, level_rules, s, L):
    import numpy as np
    n_seq, seq_len, d_model = residuals_np.shape
    res_flat = residuals_np.reshape(-1, d_model)
    grand_mean = res_flat.mean(axis=0)
    ss_total = float(np.sum((res_flat - grand_mean) ** 2))
    res_last = residuals_np[:, -1, :]
    last_mean = res_last.mean(axis=0)
    ss_total_last = float(np.sum((res_last - last_mean) ** 2))
    if ss_total < 1e-12:
        return {f"level_{ell}": {"rule_eta2": 0.0, "feature_eta2": 0.0,
                                 "rule_eta2_last": 0.0, "feature_eta2_last": 0.0,
                                 "tokens_per_subtree": s ** (L - ell)} for ell in range(L)}
    results = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        ancestor_idx_all = np.arange(seq_len) // s_power
        rule_labels_all = level_rules[ell][:, ancestor_idx_all].reshape(-1)
        feat_labels_all = level_features[ell][:, ancestor_idx_all].reshape(-1)
        last_ancestor = (seq_len - 1) // s_power
        rule_labels_last = level_rules[ell][:, last_ancestor]
        feat_labels_last = level_features[ell][:, last_ancestor]
        results[f"level_{ell}"] = {
            "rule_eta2": _eta2_between(res_flat, rule_labels_all, grand_mean) / ss_total,
            "feature_eta2": _eta2_between(res_flat, feat_labels_all, grand_mean) / ss_total,
            "rule_eta2_last": (_eta2_between(res_last, rule_labels_last, last_mean) / ss_total_last
                               if ss_total_last > 1e-12 else 0.0),
            "feature_eta2_last": (_eta2_between(res_last, feat_labels_last, last_mean) / ss_total_last
                                  if ss_total_last > 1e-12 else 0.0),
            "tokens_per_subtree": int(s_power),
        }
    return results


def _compute_effective_rank(res_flat, n_embd, max_samples=50000):
    import numpy as np
    n = min(max_samples, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], n, replace=False)
    sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))
    top1_var = float((S[0] ** 2) / (S ** 2).sum())
    return eff_rank, top1_var


def _ensure_corpus_distinct(v, s, L, m, n_tokens, rule_seed=0, seq_seed=1):
    """Generate (or reuse) the DISTINCT-rule corpus + rules, matching rhm_thread_b
    exactly (same key, same seeds) so the main model trains on the identical DGP."""
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct, generate_sequences_batched
    key = _tb_key(v, s, L, m)
    data_dir = f"{DATA_DIR}/{key}"
    os.makedirs(data_dir, exist_ok=True)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)  # deterministic
    corpus_path = f"{data_dir}/corpus.npy"
    need = (not os.path.exists(corpus_path)) or len(np.load(corpus_path, mmap_mode="r")) < n_tokens
    if need:
        seq_len = s ** L
        n_seq = (n_tokens + seq_len - 1) // seq_len
        print(f"Generating distinct corpus {key}: {n_tokens:,} tokens")
        seqs = generate_sequences_batched(rules, n_seq, seed=seq_seed)
        np.save(corpus_path, seqs.reshape(-1)[:n_tokens])
        for ell, r in enumerate(rules):
            np.save(f"{data_dir}/rules_L{ell}.npy", r)
        volume.commit()
    return key


# --------------------------------------------------------------------------
# Phase 1: train main model with DENSE checkpoints (parametrized n_steps)
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=14400, memory=32768)
def train_main(
    v: int, m: int, s: int = 2, depth: int = 6, n_tokens: int = 20_000_000,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    n_steps: int = 50000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, seed: int = 42, eval_interval: int = 250,
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

    # dense early (low levels learn fast) + coverage through saturation
    fracs = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0]
    ckpt_steps = sorted(set(round(f * n_steps / eval_interval) * eval_interval for f in fracs))
    model_tag = f"{n_layer}L{n_head}H{n_embd}D"
    save_dir = f"{DATA_DIR}/{key}/fm_legibility_{model_tag}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== train_main {key} {model_tag}  n_steps={n_steps}  ckpts={ckpt_steps}")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    history, saved = {"train_loss": [], "val_loss": []}, []
    for step in range(n_steps + 1):
        if step > 0:
            model.train()
            x, y = get_batch(train_data)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(*get_batch(val_data))[1]) for _ in range(10)]))
            history["val_loss"].append((step, vl))
            if step in ckpt_steps:
                p = os.path.join(save_dir, f"ckpt_step{step}.pt")
                torch.save(model.state_dict(), p)
                saved.append({"step": step, "val_loss": vl, "path": p})
                print(f"  step {step:6d}: val={vl:.4f} [CKPT]")
    with open(os.path.join(save_dir, "training_info.json"), "w") as f:
        json.dump({"key": key, "model_tag": model_tag, "n_steps": n_steps,
                   "checkpoints": saved, "history": history}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return {"key": key, "model_tag": model_tag, "checkpoints": saved, "history": history}


# --------------------------------------------------------------------------
# Phase 2: analyze ONE checkpoint across the full FM grid (gap x capacity x heads)
# --------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=32768)
def analyze_checkpoint(
    v: int, m: int, grid: list, checkpoint_path: str, checkpoint_step: int,
    s: int = 2, depth: int = 6, n_tokens: int = 20_000_000,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    fm_train_steps: int = 8000, fm_lr: float = 1e-3,
    n_eval_sequences: int = 8000, batch_size: int = 64, fm_seed: int = 137,
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

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    print(f"=== analyze {key} step {checkpoint_step}  val={main_val_loss:.4f}  ({len(grid)} FMs) ===")

    cfg_results = []
    for cfg in grid:
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
        feat_last = " ".join(f"d{L-e}:{eta2[f'level_{e}']['feature_eta2_last']:.3f}" for e in range(L))
        print(f"  [{cfg['label']}] cap={cap_pct:.0f}% cos={cos:.4f} rank={eff_rank/n_embd*100:.0f}%  {feat_last}")

    result = {"step": checkpoint_step, "main_val_loss": main_val_loss, "configs": cfg_results}
    # Incremental persistence: each checkpoint's result is committed on its own, so a
    # preemption of the orchestrator can never lose collected analysis (resumable).
    parts_dir = f"{DATA_DIR}/rhm_fm_legibility/parts"
    os.makedirs(parts_dir, exist_ok=True)
    with open(f"{parts_dir}/v{v}m{m}_step{checkpoint_step}.json", "w") as fpart:
        json.dump(result, fpart, indent=2, cls=NumpyEncoder)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

# (src, tgt, n_head, d_head, mlp_mult, label). d=256 -> per-block 789,760.
#  A = 8H/16d/mlp1 ~529K (~14-17% of modeled blocks); B = 8H/16d/mlp2 ~791K (~20-25%);
#  headctrl = 1H/128d/mlp2 = 791K (param-twin of B, attention split into 1 head).
GRID = [
    {"src": "post_embed",  "tgt": "post_block4", "nh": 8, "dh": 16, "mm": 1.0, "label": "E->b4 8H16d m1"},
    {"src": "post_block0", "tgt": "post_block4", "nh": 8, "dh": 16, "mm": 1.0, "label": "b0->b4 8H16d m1"},
    {"src": "post_embed",  "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 1.0, "label": "E->b6 8H16d m1"},
    {"src": "post_block0", "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 1.0, "label": "b0->b6 8H16d m1"},
    {"src": "post_embed",  "tgt": "post_block6", "nh": 8, "dh": 16, "mm": 2.0, "label": "E->b6 8H16d m2 capB"},
    {"src": "post_embed",  "tgt": "post_block6", "nh": 1, "dh": 128, "mm": 2.0, "label": "E->b6 1H128d m2 headctrl"},
]
DGPS = [(16, 2), (16, 4)]


@app.function(volumes={DATA_DIR: volume}, timeout=36000, memory=16384)
def run_legibility(n_steps: int = 50000, fm_train_steps: int = 8000):
    print(f"=== FM residual-legibility over training ===")
    print(f"  DGPs (distinct): {DGPS}   FM grid: {len(GRID)} configs   n_steps={n_steps}")

    # Phase 1: train both main models in parallel
    train_handles = {(v, m): train_main.spawn(v=v, m=m, n_steps=n_steps) for v, m in DGPS}
    trainings = {vm: h.get() for vm, h in train_handles.items()}

    # Phase 2: per DGP, per checkpoint, run the full FM grid
    out = {"config": {"dgps": DGPS, "grid": GRID, "n_steps": n_steps,
                      "fm_train_steps": fm_train_steps}, "results": {}}
    for (v, m) in DGPS:
        tr = trainings[(v, m)]
        handles = [(c["step"], analyze_checkpoint.spawn(
            v=v, m=m, grid=GRID, checkpoint_path=c["path"], checkpoint_step=c["step"],
            fm_train_steps=fm_train_steps))
            for c in tr["checkpoints"]]
        ckpt_results = []
        for step, h in handles:
            try:
                ckpt_results.append(h.get())
            except Exception as e:
                print(f"  v{v}m{m} step {step}: FAILED -- {e}")
        ckpt_results.sort(key=lambda r: r["step"])
        out["results"][f"v{v}m{m}"] = {"training": tr["history"], "checkpoints": ckpt_results}

        # compact per-config feature_eta2_last trajectory table
        L = 6
        for cfg in GRID:
            print(f"\n{'='*120}\nv{v}m{m}  [{cfg['label']}]  (feature_eta2_last by depth; d6=root)\n{'='*120}")
            print(f"{'step':>6} {'val':>6} {'cos':>6} {'rank%':>6} {'top1%':>5} | " +
                  " ".join(f"d{6-e}" for e in range(L)))
            for r in ckpt_results:
                cr = next((c for c in r["configs"] if c["label"] == cfg["label"]), None)
                if cr is None:
                    continue
                eta = cr["hierarchy_eta2"]
                cells = " ".join(f"{eta[f'level_{e}']['feature_eta2_last']:.2f}" for e in range(L))
                flag = "" if 0.90 <= cr["fm_cosine"] <= 0.99 else " (cos OOB)"
                print(f"{r['step']:>6} {r['main_val_loss']:>6.3f} {cr['fm_cosine']:>6.3f} "
                      f"{cr['effective_rank_pct']:>6.1f} {cr['top1_pc']*100:>5.1f} | {cells}{flag}")

    save_dir = f"{DATA_DIR}/rhm_fm_legibility"
    os.makedirs(save_dir, exist_ok=True)
    fpath = f"{save_dir}/fm_legibility_8L8H256D_S{n_steps}.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {fpath}")
    return out


def _print_dgp_summary(v, m, ckpt_results, L=6):
    for cfg in GRID:
        print(f"\n{'='*120}\nv{v}m{m}  [{cfg['label']}]  (feature_eta2_last by depth; d6=root)\n{'='*120}")
        print(f"{'step':>6} {'val':>6} {'cos':>6} {'rank%':>6} {'top1%':>5} | " +
              " ".join(f"d{6-e}" for e in range(L)))
        for r in ckpt_results:
            cr = next((c for c in r["configs"] if c["label"] == cfg["label"]), None)
            if cr is None:
                continue
            eta = cr["hierarchy_eta2"]
            cells = " ".join(f"{eta[f'level_{e}']['feature_eta2_last']:.2f}" for e in range(L))
            flag = "" if 0.90 <= cr["fm_cosine"] <= 0.99 else " (cos OOB)"
            print(f"{r['step']:>6} {r['main_val_loss']:>6.3f} {cr['fm_cosine']:>6.3f} "
                  f"{cr['effective_rank_pct']:>6.1f} {cr['top1_pc']*100:>5.1f} | {cells}{flag}")


@app.function(volumes={DATA_DIR: volume}, timeout=36000, memory=16384)
def analyze_only(n_steps: int = 50000, fm_train_steps: int = 8000, model_tag: str = "8L8H256D"):
    """Phase-2-only re-run on T4 from the already-saved main-model checkpoints.

    Robust to preemption: analyze_checkpoint commits each result to /rhm_fm_legibility
    /parts before returning, so this orchestrator (a) SKIPS checkpoints whose part
    already exists (resumable across its own restarts), and (b) aggregates the final
    JSON by reading the parts off the volume, never from orchestrator memory. T4 is
    far more abundant than A10G, so the burst of analyze jobs is unlikely to be
    preempted in the first place. No main-model retraining (checkpoints are reused).
    Writes the same final JSON as run_legibility, so the volume waiter is unchanged.
    """
    volume.reload()
    parts_dir = f"{DATA_DIR}/rhm_fm_legibility/parts"
    os.makedirs(parts_dir, exist_ok=True)
    print(f"=== analyze_only (T4) === DGPs {DGPS}  grid {len(GRID)}  fm_steps={fm_train_steps}")

    pending = []
    for (v, m) in DGPS:
        key = _tb_key(v, 2, 6, m)
        info = json.load(open(f"{DATA_DIR}/{key}/fm_legibility_{model_tag}/training_info.json"))
        for c in info["checkpoints"]:
            part = f"{parts_dir}/v{v}m{m}_step{c['step']}.json"
            if os.path.exists(part):
                print(f"  skip v{v}m{m} step {c['step']} (part exists)")
                continue
            h = analyze_checkpoint.spawn(
                v=v, m=m, grid=GRID, checkpoint_path=c["path"],
                checkpoint_step=c["step"], fm_train_steps=fm_train_steps)
            pending.append((v, m, c["step"], h))
    print(f"  spawned {len(pending)} analyze jobs on T4")
    for v, m, step, h in pending:
        try:
            h.get()
        except Exception as e:
            print(f"  v{v}m{m} step {step}: FAILED -- {e}")

    volume.reload()
    out = {"config": {"dgps": DGPS, "grid": GRID, "n_steps": n_steps,
                      "fm_train_steps": fm_train_steps, "mode": "analyze_only_T4"},
           "results": {}}
    for (v, m) in DGPS:
        key = _tb_key(v, 2, 6, m)
        info = json.load(open(f"{DATA_DIR}/{key}/fm_legibility_{model_tag}/training_info.json"))
        ckpt_results = []
        for c in info["checkpoints"]:
            part = f"{parts_dir}/v{v}m{m}_step{c['step']}.json"
            if os.path.exists(part):
                ckpt_results.append(json.load(open(part)))
        ckpt_results.sort(key=lambda r: r["step"])
        out["results"][f"v{v}m{m}"] = {"training": info["history"], "checkpoints": ckpt_results}
        _print_dgp_summary(v, m, ckpt_results)

    fpath = f"{DATA_DIR}/rhm_fm_legibility/fm_legibility_{model_tag}_S{n_steps}.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {fpath}")
    return out
