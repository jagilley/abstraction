"""Activation-rank analysis of the RL dimensionality ablation checkpoints.

Motivation: a reaction to the rl_dimensionality result claimed that
"capability frying and representation frying are the same geometric process:
RL concentrates mass on narrow trajectories, which collapses the effective rank
of the rest of the activation space." This script measures that claim on the
saved checkpoints (pretrained / vanilla REINFORCE / KL-anchored / expert
iteration / continued-NTP control, x5 m, x2 verifiers).

Two rank quantities are deliberately kept apart:

  data-PR    -- participation ratio of post_block{i} activations on a FIXED set
                of ground-truth RHM sequences (the pretraining distribution).
                This is representation change proper.
  rollout-PR -- the same statistic on the checkpoint's OWN temperature-1
                rollouts. Shrinks tautologically when the policy sharpens
                (the input distribution narrows); reported as the control.

Per checkpoint, per block (post_block0..5 and post_lnf, the readout's input):
  spectra on fixed data (all positions / suffix positions / last position) and
      on own rollouts (suffix positions): PR = (sum l)^2 / sum l^2, entropy
      rank exp(H(l/sum l)), top-1 variance fraction, trace
  "rest of the space": PR of fixed-data activations in the orthogonal
      complement of the checkpoint's top-k rollout directions, paired with the
      pretrained model's PR in the SAME complement
  hierarchy decomposition at the last position: least-squares projection of
      activations onto one-hot ancestor features + rule choices at every level
      (variance fraction explained ~ sum of eta^2), PR of the explained part and
      of the residual -> distinguishes "hierarchy variance collapsed onto one
      junk direction" from "hierarchy variance leaked into many junk directions"
  mechanism: mean activation norm, mean pairwise cosine across sequences at the
      same position (context collapse), logit margin and output entropy on the
      fixed data (sharpening), val NTP loss (cross-check against results.json)
  weight-space: PR of the singular values of dW = W_cond - W_pretrained per
      matrix, and |dW|_F / |W_pre|_F (is the update low-rank?)

Reproduction (from experiments/):
  modal run -m rhm.rl_dimensionality.rank_analysis::analyze --m 2 --smoke
  modal run --detach -m rhm.rl_dimensionality.rank_analysis::analyze_all
Outputs: rl_dimensionality/rank_analysis/rank_m{m}.json on the volume.
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)

app = modal.App("rhm-rl-dim-rank", image=image)

RUN_TAGS = ["", "kl01", "ei01", "ei02"]   # dirs scanned for *.pt (missing ok)
TOP_K_REST = 8


# ----------------------------------------------------------------------
# Spectrum helpers (numpy, float64)
# ----------------------------------------------------------------------

def _spectrum_stats(cov):
    import numpy as np
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0.0, None)
    tot = float(lam.sum())
    if tot <= 0:
        return {"pr": 0.0, "erank": 0.0, "top1_frac": 0.0, "trace": 0.0}
    p = lam / tot
    pr = float(tot ** 2 / np.sum(lam ** 2))
    ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return {"pr": pr, "erank": float(np.exp(ent)),
            "top1_frac": float(lam.max() / tot), "trace": tot}


def _top_eigvecs(cov, k):
    import numpy as np
    w, V = np.linalg.eigh(cov)
    return V[:, ::-1][:, :k]


def _complement_stats(cov, U):
    import numpy as np
    P = np.eye(cov.shape[0]) - U @ U.T
    return _spectrum_stats(P @ cov @ P)


class _Acc:
    """Running mean / second moment / unit-vector sum / norm sum for one
    (block, position-set)."""

    def __init__(self, d, device):
        import torch
        self.n = 0
        self.s = torch.zeros(d, dtype=torch.float64, device=device)
        self.ss = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.u = torch.zeros(d, dtype=torch.float64, device=device)
        self.norm = 0.0

    def add(self, x):  # x: (N, d)
        import torch
        x = x.to(torch.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += x.T @ x
        nrm = x.norm(dim=1)
        self.norm += float(nrm.sum())
        self.u += (x / nrm.clamp_min(1e-12)[:, None]).sum(0)

    def cov(self):
        mu = self.s / self.n
        return (self.ss / self.n - mu[:, None] * mu[None, :]).cpu().numpy()

    def stats(self):
        st = _spectrum_stats(self.cov())
        # mean pairwise cosine between distinct vectors: (|sum u|^2 - n)/(n(n-1))
        usq = float((self.u ** 2).sum())
        st["mean_cos"] = (usq - self.n) / (self.n * (self.n - 1))
        st["mean_norm"] = self.norm / self.n
        return st


# ----------------------------------------------------------------------
# Per-checkpoint measurement
# ----------------------------------------------------------------------

def _measure(model, eval_t, level_features, level_rules, val_t, gen_prefix_t,
             rules_t, prefix_len, suffix_len, s, L, v, batch_size, device,
             pretrained_ref=None, n_layer=6, temperature=1.0):
    """Returns a dict of readouts for one checkpoint.

    pretrained_ref: the pretrained model's measurement dict (with the raw
    covariance matrices attached under "_cov"), used for the paired
    rest-of-space comparison. None when measuring the pretrained model itself.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.rl_dimensionality.rl_dim_ablation import (
        _generate_suffix, parse_valid_fractions_torch, _compute_hierarchy_eta2,
        _suffix_position_levels)

    model.eval()
    keys = [f"post_block{i}" for i in range(n_layer)] + ["post_lnf"]
    d = model.transformer.wte.weight.shape[1]
    seq_len = eval_t.shape[1]
    psets = {"all": slice(0, seq_len), "suffix": slice(prefix_len - 1, seq_len),
             "last": slice(seq_len - 1, seq_len)}
    acc = {k: {ps: _Acc(d, device) for ps in psets} for k in keys}
    last_acts = {k: [] for k in keys}
    ent_sum, margin_sum, cnt = 0.0, 0.0, 0
    suffix_levels = np.array(_suffix_position_levels(prefix_len, seq_len, s))
    ent_lvl = {int(l): [0.0, 0] for l in np.unique(suffix_levels)}

    def run_batches(seqs):
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                b = seqs[i:i + batch_size]
                logits, _, inter = model(b, return_intermediates=True)
                inter["post_lnf"] = model.transformer.ln_f(
                    inter[f"post_block{n_layer - 1}"])
                yield b, logits, inter

    # ---- fixed ground-truth data ----
    for b, logits, inter in run_batches(eval_t):
        for k in keys:
            x = inter[k]
            for ps, sl in psets.items():
                acc[k][ps].add(x[:, sl, :].reshape(-1, d))
            last_acts[k].append(x[:, -1, :].float().cpu().numpy())
        # output-side: entropy + margin at suffix-predicting positions
        lg = logits[:, prefix_len - 1:-1, :].float()     # predicts tokens prefix_len..seq_len-1
        lp = F.log_softmax(lg, dim=-1)
        ent = -(lp.exp() * lp).sum(-1)                   # (B, suffix_len)
        top2 = lg.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        ent_sum += float(ent.sum()); margin_sum += float(margin.sum())
        cnt += ent.numel()
        for l in ent_lvl:
            mask = torch.from_numpy(suffix_levels == l).to(device)
            ent_lvl[l][0] += float(ent[:, mask].sum()); ent_lvl[l][1] += int(mask.sum()) * ent.shape[0]

    out = {"data": {}, "hier": {}, "_cov": {}}
    for k in keys:
        out["data"][k] = {ps: acc[k][ps].stats() for ps in psets}
        out["_cov"][k] = acc[k]["suffix"].cov()
    out["output"] = {
        "entropy_suffix_nats": ent_sum / cnt,
        "logit_margin_suffix": margin_sum / cnt,
        "entropy_per_level": {f"L{l}": a / b for l, (a, b) in ent_lvl.items()},
    }

    # ---- hierarchy decomposition at the last position ----
    n_last = len(eval_t)
    last_anc = [((seq_len - 1) // (s ** (L - ell))) for ell in range(L)]
    Z_cols = []
    for ell in range(L):
        f = level_features[ell][:n_last, last_anc[ell]]
        Z_cols.append(np.eye(v)[f])
        r = level_rules[ell][:n_last, last_anc[ell]]
        Z_cols.append(np.eye(int(r.max()) + 1)[r])
    Z = np.concatenate(Z_cols, axis=1).astype(np.float64)
    for k in keys:
        Y = np.concatenate(last_acts[k], axis=0).astype(np.float64)
        Yc = Y - Y.mean(0)
        beta, *_ = np.linalg.lstsq(Z, Yc, rcond=None)
        Yh = Z @ beta
        R = Yc - Yh
        tot = float((Yc ** 2).sum())
        acts3 = Y[:, None, :]
        eta = _compute_hierarchy_eta2(acts3, [lf[:n_last] for lf in level_features],
                                      [lr[:n_last] for lr in level_rules], s, L)
        out["hier"][k] = {
            "explained_frac": float((Yh ** 2).sum() / tot) if tot > 0 else 0.0,
            "explained": _spectrum_stats(Yh.T @ Yh / n_last),
            "residual": _spectrum_stats(R.T @ R / n_last),
            "feature_eta2_per_level": [eta[f"level_{ell}"]["feature_eta2"] for ell in range(L)],
            "rule_eta2_per_level": [eta[f"level_{ell}"]["rule_eta2"] for ell in range(L)],
        }

    # ---- own rollouts (temperature-1) on the fixed prefixes ----
    roll_acc = {k: _Acc(d, device) for k in keys}
    gens, ems, pfs = [], [], []
    with torch.no_grad():
        for i in range(0, len(gen_prefix_t), batch_size):
            full = gen_prefix_t[i:i + batch_size]
            prefix = full[:, :prefix_len]
            gen = _generate_suffix(model, prefix, suffix_len, temperature)
            seq = torch.cat([prefix, gen], dim=1)
            ems.append((gen == full[:, prefix_len:]).float().mean().item() * gen.shape[0])
            pfs.append(parse_valid_fractions_torch(seq, rules_t).cpu())
            _, _, inter = model(seq, return_intermediates=True)
            inter["post_lnf"] = model.transformer.ln_f(inter[f"post_block{n_layer - 1}"])
            for k in keys:
                roll_acc[k].add(inter[k][:, prefix_len - 1:, :].reshape(-1, d))
            gens.append(gen.cpu())
    gens = torch.cat(gens)
    pf = torch.cat(pfs)
    out["rollout"] = {k: roll_acc[k].stats() for k in keys}
    out["rollout_metrics"] = {
        "exact": float(sum(ems) / len(gen_prefix_t)),
        "parse_overall": float(pf.mean()),
        "parse_root": float(pf[:, -1].mean()),
        "n_unique_suffixes": int(len(set(map(tuple, gens.tolist())))),
    }

    # ---- rest of the space: complement of this checkpoint's top-k rollout dirs
    out["rest"] = {}
    for k in keys:
        U = _top_eigvecs(roll_acc[k].cov(), TOP_K_REST)
        Cd = out["_cov"][k]
        rec = {"top_k": TOP_K_REST,
               "rollout_topk_data_frac": float(np.trace(U.T @ Cd @ U) / max(np.trace(Cd), 1e-30)),
               "rest": _complement_stats(Cd, U)}
        if pretrained_ref is not None:
            rec["pretrained_rest_same_subspace"] = _complement_stats(pretrained_ref["_cov"][k], U)
            rec["pretrained_rollout_topk_data_frac"] = float(
                np.trace(U.T @ pretrained_ref["_cov"][k] @ U) / max(np.trace(pretrained_ref["_cov"][k]), 1e-30))
        out["rest"][k] = rec

    # ---- val NTP loss (cross-check against results.json) ----
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_t), batch_size):
            b = val_t[i:i + batch_size]
            _, loss = model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
            tot += float(loss) * b.shape[0]; n += b.shape[0]
    out["val_loss"] = tot / n
    return out


def _weight_delta_stats(state, ref_state):
    """PR of singular values of dW per 2-D matrix + relative Frobenius norm."""
    import numpy as np
    rec = {}
    for name, W in state.items():
        if W.ndim != 2 or name not in ref_state:
            continue
        dW = (W - ref_state[name]).double().numpy()
        sv = np.linalg.svd(dW, compute_uv=False)
        if sv.sum() <= 0:
            continue
        rec[name] = {
            "pr": float(sv.sum() ** 2 / (sv ** 2).sum()),
            "pr_sq": float((sv ** 2).sum() ** 2 / (sv ** 4).sum()),
            "rel_fro": float(np.linalg.norm(dW) / max(np.linalg.norm(ref_state[name].double().numpy()), 1e-30)),
            "shape": list(dW.shape),
        }
    return rec


# ----------------------------------------------------------------------
# Modal entrypoints
# ----------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def analyze(m: int = 2, v: int = 8, s: int = 2, depth: int = 6, rule_seed: int = 0,
            n_layer: int = 6, n_head: int = 6, n_embd: int = 192, seed: int = 42,
            n_eval_seqs: int = 4000, n_val_seqs: int = 2000, n_gen_eval: int = 1000,
            batch_size: int = 250, smoke: bool = False, run_tags: str = ""):
    import glob
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_invertible, generate_sequences_batched
    from rhm.rl_dimensionality.rl_dim_ablation import _generate_with_traces

    if smoke:
        n_eval_seqs, n_val_seqs, n_gen_eval = 500, 300, 200
    device = "cuda"
    L = depth
    seq_len = s ** L
    prefix_len = seq_len // 2
    suffix_len = seq_len - prefix_len
    key = f"v{v}_s{s}_L{L}_m{m}"
    base = f"{DATA_DIR}/rl_dimensionality/{key}_both_seed{seed}"
    tags = RUN_TAGS if not run_tags else run_tags.split(",")
    out_dir = f"{DATA_DIR}/rl_dimensionality/rank_analysis"
    os.makedirs(out_dir, exist_ok=True)

    # same data as the experiment: eval traces seed 12345, val seed+2, gen seed+4
    rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
    for ell, r in enumerate(rules):
        saved = np.load(os.path.join(base, f"rules_L{ell}.npy"))
        assert (saved == r).all(), "rules mismatch vs saved run"
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_seqs, seed=12345)
    val_seqs = generate_sequences_batched(rules, n_val_seqs, seed=seed + 2)
    gen_eval_seqs = generate_sequences_batched(rules, n_gen_eval, seed=seed + 4)
    eval_t = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    val_t = torch.from_numpy(val_seqs.astype(np.int64)).to(device)
    gen_t = torch.from_numpy(gen_eval_seqs.astype(np.int64)).to(device)
    rules_t = [torch.from_numpy(r).long().to(device) for r in rules]

    # checkpoints: init (reconstructed), pretrained, every *_final.pt / *_r{N}.pt
    ckpts = {}
    for tag in tags:
        d = base + (f"_{tag}" if tag else "")
        for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
            stem = os.path.basename(p)[:-3]
            if stem == "pretrained" and tag:
                continue
            label = stem if not tag else f"{stem}@{tag}"
            ckpts[label] = p
    print(f"[{key}] {len(ckpts)} checkpoints: {list(ckpts)}")

    def make_gpt():
        return GPT(v, seq_len, n_layer, n_head, n_embd).to(device)

    torch.manual_seed(seed)
    init_model = make_gpt()
    init_state = {k: p.cpu().clone() for k, p in init_model.state_dict().items()}
    pre_state = torch.load(ckpts["pretrained"], map_location="cpu")

    results = {"config": {"m": m, "v": v, "s": s, "L": L, "n_eval_seqs": n_eval_seqs,
                          "n_gen_eval": n_gen_eval, "top_k_rest": TOP_K_REST,
                          "checkpoints": ckpts},
               "checkpoints": {}}
    common = dict(eval_t=eval_t, level_features=level_features, level_rules=level_rules,
                  val_t=val_t, gen_prefix_t=gen_t, rules_t=rules_t, prefix_len=prefix_len,
                  suffix_len=suffix_len, s=s, L=L, v=v, batch_size=batch_size,
                  device=device, n_layer=n_layer)

    torch.manual_seed(seed + 7)
    model = make_gpt(); model.load_state_dict(pre_state)
    pre = _measure(model, pretrained_ref=None, **common)
    results["checkpoints"]["pretrained"] = {k_: v_ for k_, v_ in pre.items() if k_ != "_cov"}
    print(f"  pretrained: val={pre['val_loss']:.4f} H={pre['output']['entropy_suffix_nats']:.3f} "
          f"PR(b5,suffix)={pre['data']['post_block5']['suffix']['pr']:.1f}")

    order = ["init"] + [k_ for k_ in ckpts if k_ != "pretrained"]
    for label in order:
        state = init_state if label == "init" else torch.load(ckpts[label], map_location="cpu")
        torch.manual_seed(seed + 7)
        model = make_gpt(); model.load_state_dict(state)
        rec = _measure(model, pretrained_ref=pre, **common)
        rec = {k_: v_ for k_, v_ in rec.items() if k_ != "_cov"}
        rec["weight_delta_vs_pretrained"] = _weight_delta_stats(state, pre_state)
        results["checkpoints"][label] = rec
        print(f"  {label:>28s}: val={rec['val_loss']:.4f} H={rec['output']['entropy_suffix_nats']:.3f} "
              f"PR(b5,suffix) data={rec['data']['post_block5']['suffix']['pr']:.1f} "
              f"roll={rec['rollout']['post_block5']['pr']:.1f} "
              f"rest={rec['rest']['post_block5']['rest']['pr']:.1f} "
              f"(pre same-subspace {rec['rest']['post_block5']['pretrained_rest_same_subspace']['pr']:.1f}) "
              f"cos={rec['data']['post_block5']['suffix']['mean_cos']:.3f} "
              f"uniq={rec['rollout_metrics']['n_unique_suffixes']}")
        del model; torch.cuda.empty_cache()

    fn = os.path.join(out_dir, f"rank_m{m}{'_smoke' if smoke else ''}.json")
    with open(fn, "w") as f:
        json.dump(results, f, indent=1, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved {fn}")
    return fn


@app.function(volumes={DATA_DIR: volume}, timeout=7200)
def analyze_all(ms: str = "1,2,3,4,6", run_tags: str = ""):
    outs = list(analyze.starmap([(int(m),) for m in ms.split(",")],
                                kwargs={"run_tags": run_tags}))
    print(outs)
    return outs
