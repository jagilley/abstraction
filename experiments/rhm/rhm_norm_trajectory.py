"""Weight-norm and per-layer activation-norm trajectories from the EXISTING
rhm_fm_legibility main-model checkpoints (no retraining).

Motivation (see RHM_FRONTIER_AND_LEGIBILITY_README "Knowledge-grok vs circuit-grok"):
the m2 model reaches the information ceiling (knowledge-grok) but shows no
circuit-complexity collapse (rank flat ~84%, top1 ~5%, cosine falling). The
legibility writeup INFERRED a ~17x post_block6 activation-norm drift from the
cosine identity but never measured it directly. This job measures, per checkpoint:

  - weight norm: total ||W||_2 and per-group (embeddings, per-block attn/mlp/ln,
    head), plus per-param RMS (= norm/sqrt(numel), what weight decay drives down).
  - activation norm: per-token RMS of each post_embed/post_block{i}, mean over a
    fixed val batch, plus the last-token norm (matches thread_b / legibility probes).

Contrast: m2 (groks to root, WD 0.01) vs m4 (stalls ~d4). Decisive question the
norms answer: is the residual-norm growth in the legibility run weight-norm growth,
activation-norm drift, or both? And does either dissociate from the (flat) rank?

Run:
    cd experiments && modal run --detach -m rhm.rhm_norm_trajectory::norm_trajectory
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.7.0")
    .add_local_python_source("rhm")
)
app = modal.App("rhm-norm-trajectory", image=image)


def _tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def _weight_norms(model):
    """L2 norm and RMS per param group, bucketed by name."""
    import torch
    groups = {}  # name -> (sumsq, numel)

    def add(bucket, p):
        ss, n = groups.get(bucket, (0.0, 0))
        groups[bucket] = (ss + float(p.detach().double().pow(2).sum()), n + p.numel())

    for name, p in model.named_parameters():
        if name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
            add("embed", p)
        elif name.startswith("transformer.h."):
            i = int(name.split(".")[2])
            if ".attn." in name:
                add(f"block{i}.attn", p)
            elif ".mlp." in name:
                add(f"block{i}.mlp", p)
            else:
                add(f"block{i}.ln", p)
            add(f"block{i}.all", p)
        elif name.startswith("lm_head"):
            add("head", p)
        elif name.startswith("transformer.ln_f"):
            add("ln_f", p)
        add("total", p)

    out = {}
    for k, (ss, n) in groups.items():
        out[k] = {"norm": ss ** 0.5, "rms": (ss / n) ** 0.5, "numel": n}
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=3600, memory=16384)
def norm_trajectory_one(
    v: int, m: int, s: int = 2, depth: int = 6,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    model_tag: str = "8L8H256D", n_act_batches: int = 8, batch_size: int = 64,
    n_tokens: int = 20_000_000,
):
    import numpy as np
    import torch
    from rhm.model import GPT

    L = depth
    block_size = s ** L
    device = "cuda"
    key = _tb_key(v, s, L, m)
    volume.reload()

    info = json.load(open(f"{DATA_DIR}/{key}/fm_legibility_{model_tag}/training_info.json"))
    ckpts = sorted(info["checkpoints"], key=lambda c: c["step"])

    corpus = torch.from_numpy(np.load(f"{DATA_DIR}/{key}/corpus.npy")[:n_tokens].astype(np.int64))
    val_data = corpus[int(0.9 * len(corpus)):]

    def get_batch():
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,))
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        return x.to(device)

    torch.manual_seed(0)
    fixed_batches = [get_batch() for _ in range(n_act_batches)]
    inter_keys = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    rows = []
    for c in ckpts:
        sd = torch.load(c["path"], map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        wn = _weight_norms(model)

        # activation norms: per-token RMS (||x||/sqrt(d)) and last-token L2, averaged
        acc_rms = {k: [] for k in inter_keys}
        acc_last = {k: [] for k in inter_keys}
        with torch.no_grad():
            for xb in fixed_batches:
                _, _, inter = model(xb, return_intermediates=True)
                for k in inter_keys:
                    a = inter[k]  # (B, T, d)
                    acc_rms[k].append(float(a.pow(2).mean(-1).sqrt().mean()))
                    acc_last[k].append(float(a[:, -1, :].pow(2).sum(-1).sqrt().mean()))
        act = {k: {"rms": float(np.mean(acc_rms[k])),
                   "last_l2": float(np.mean(acc_last[k]))} for k in inter_keys}

        rows.append({"step": c["step"], "val_loss": c.get("val_loss"),
                     "weight": wn, "act": act})
        print(f"  v{v}m{m} step {c['step']:6d}: "
              f"||W||={wn['total']['norm']:.1f} rms={wn['total']['rms']:.4f} "
              f"act_last[pb{n_layer-2}]={act[f'post_block{n_layer-2}']['last_l2']:.2f} "
              f"act_last[pb{n_layer-1}]={act[f'post_block{n_layer-1}']['last_l2']:.2f}")

    save_dir = f"{DATA_DIR}/rhm_norm_trajectory"
    os.makedirs(save_dir, exist_ok=True)
    out = {"v": v, "m": m, "model_tag": model_tag, "rows": rows}
    with open(f"{save_dir}/v{v}m{m}_{model_tag}.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3600)
def norm_trajectory():
    dgps = [(16, 2), (16, 4)]
    handles = [(v, m, norm_trajectory_one.spawn(v=v, m=m)) for v, m in dgps]
    results = {}
    for v, m, h in handles:
        results[f"v{v}m{m}"] = h.get()

    # compact print: weight RMS + post_block6 last-token L2 over training
    for v, m in dgps:
        rows = results[f"v{v}m{m}"]["rows"]
        print(f"\n=== v{v}m{m} ===")
        print(f"{'step':>6} {'val':>6} {'Wnorm':>8} {'Wrms':>7} "
              f"{'pb0':>7} {'pb4':>7} {'pb6':>7} {'pb7':>7}")
        for r in rows:
            a = r["act"]
            print(f"{r['step']:>6} {r['val_loss']:>6.3f} "
                  f"{r['weight']['total']['norm']:>8.1f} {r['weight']['total']['rms']:>7.4f} "
                  f"{a['post_block0']['last_l2']:>7.2f} {a['post_block4']['last_l2']:>7.2f} "
                  f"{a['post_block6']['last_l2']:>7.2f} {a['post_block7']['last_l2']:>7.2f}")
    save_dir = f"{DATA_DIR}/rhm_norm_trajectory"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return results
