"""question_model, Half 2: does a learner that carries a question model buy the epistemic half?

Six arms on ONE drift corpus, identical protocol to Half 1's `train_world` (8L/8H/256D, 12k
steps, same optimiser, same flat-window sampling), differing only in what each learner is given:

| arm | what it gets | rung |
|---|---|---|
| `plain` | nothing -- the twin | -- |
| `tap_oracle` | the exact history filter's block-entry posterior over `root` + `hi` | oracle |
| `tap_shuffled` | the same vectors, block-permuted: same input, same capacity, no information | control |
| `tap_learned` | a learned leaky integrator over the previous 32 blocks' token histograms | endogenous |
| `loss_alea` | per-position gradient weight from the exact aleatoric label GIVEN theta | oracle |
| `loss_alea_blind` | the same weighting from the DRIFT-BLIND aleatoric label | control |

Why the tap carries `root` + `hi` and not `lo`: Half 1 measured `lo` at 0.868 against an exact
ceiling of 0.943, so the window already extracts it and supplying it would blur attribution.
`root` (0.239) and `hi` (0.281) sit at the shuffled-label null of ~0.25 with exact ceilings of
0.343 and 0.584. The tap's niche is exactly what the window launders.

Why the loss arm's label is theta-aware: Half 1 measured H_irr falling ~20% once theta is known
(d1 0.684 -> 0.553), so a drift-blind aleatoric label down-weights exactly the positions carrying
question-news. `loss_alea_blind` is that mistake, run deliberately as the control.

  cd experiments   # MODAL_PROFILE=chromatic
  modal run -m rhm.question_model.half2::selfcheck
  modal run --detach -m rhm.question_model.half2::precompute --tag h2
  modal run --detach -m rhm.question_model.half2::train_arm --arm tap_oracle --tag h2
  modal run --detach -m rhm.question_model.half2::evaluate --tag h2
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
app = modal.App("rhm-question-model-h2", image=image)

ARMS = ("plain", "tap_oracle", "tap_oracle_full", "tap_oracle_lo", "tap_shuffled",
        "tap_learned", "tap_learned_shuffled", "loss_alea", "loss_alea_blind")

# The tap payload's component slices inside `tap_all` (K columns each, root|hi|lo).
TAP_SLICE = {"tap_oracle": (0, 2), "tap_shuffled": (0, 2),
             "tap_oracle_full": (0, 3), "tap_oracle_lo": (2, 3)}


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def qm_dir(v, s, L, m):
    return f"{DATA_DIR}/{tb_key(v, s, L, m)}/question_model"


def _spec(q_k, sigma_s, kappa, half, dir_seed, L):
    from rhm.question_model import demand_state as DS
    return DS.make_spec(K=q_k, sigma_s=sigma_s, kappa=kappa, half=half,
                        dir_seed=dir_seed, L=L)


def _pool_paths(d, tag):
    return {k: f"{d}/h2_{tag}_{k}.npy" for k in
            ("seqs", "tidx", "tap", "tap_all", "hist", "alea", "alea_blind", "parent")}


# --------------------------------------------------------------------------- #
# precompute: the shared corpus and every exact label the arms need
# --------------------------------------------------------------------------- #

def _bll_shard(args):
    import numpy as np
    from rhm.question_model.taps import block_loglik
    rules, seqs, rw, rp, chunk = args
    return block_loglik(rules, seqs, rw, rp, chunk=chunk)


@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=65536, cpu=16.0)
def precompute(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
               q_k: int = 7, sigma_s: float = 1.0, kappa: float = 0.15, half: float = 2.0,
               dir_seed: int = 0, pool_size: int = 200000, data_seed: int = 7,
               n_workers: int = 12, tag: str = "h2"):
    """One drift corpus + every exact per-block / per-token label, cached on the volume.

    Every arm trains on the SAME token stream -- the arms differ only in what is supplied
    alongside it, which is the `arity_torque` idiom ("identical data, only the input differs").
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import taps as TP

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    spec = _spec(q_k, sigma_s, kappa, half, dir_seed, L)
    dirs = DS.make_directions(rules, spec)
    d = qm_dir(v, s, depth, m)
    os.makedirs(d, exist_ok=True)
    P = _pool_paths(d, tag)

    print(f"precompute: pool {pool_size:,} blocks, spec {spec}", flush=True)
    t0 = time.time()
    seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, pool_size, data_seed,
                                           world="drift")
    print(f"  corpus in {time.time() - t0:.0f}s", flush=True)

    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw_s = [np.stack([W[g][1][dd] for g in range(G)]) for dd in range(L)]
    rp_s = np.stack([W[g][0] for g in range(G)])

    t0 = time.time()
    bounds = np.linspace(0, pool_size, n_workers + 1).astype(int)
    args = [(rules, seqs[a:b], rw_s, rp_s, 128) for a, b in zip(bounds[:-1], bounds[1:])]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_bll_shard, args))
    bll = np.concatenate(parts, axis=0)
    print(f"  block_loglik ({n_workers} workers) in {time.time() - t0:.0f}s", flush=True)

    idx_states, pi = DS.state_grid(spec)
    Pt = DS.joint_transition(spec)
    names = [nm for nm, _ in spec["components"]]
    cidx = [names.index(c) for c in ("root", "hi")]
    ohs = [np.eye(spec["K"])[idx_states[:, c]] for c in cidx]
    margs = TP.slow_block_posteriors(bll, pi, Pt, ohs, entry=True)
    tap = np.concatenate(margs, axis=1).astype(np.float32)
    acc = [float((margs[k].argmax(1) == idx_states[tidx][:, c]).mean())
           for k, c in enumerate(cidx)]
    print(f"  slow block-entry posterior: argmax accuracy root {acc[0]:.3f}  hi {acc[1]:.3f}"
          f"   (Half-1 window-filter terminal ceilings 0.343 / 0.584)", flush=True)

    rw_true = np.stack([W[i][1][L - 1] for i in tidx])
    alea = TP.aleatoric_label_tokens(rules, seqs, lf[L - 1], rw_true).astype(np.float32)
    alea_b = TP.aleatoric_label_tokens(rules, seqs, lf[L - 1], None).astype(np.float32)
    hist = (np.eye(v, dtype=np.float32)[seqs].sum(1) / T).astype(np.float32)
    print(f"  aleatoric label: theta-aware mean {alea.mean():.4f}  blind {alea_b.mean():.4f}"
          f"  corr {np.corrcoef(alea.ravel(), alea_b.ravel())[0, 1]:+.3f}", flush=True)

    np.save(P["seqs"], seqs.astype(np.int16))
    np.save(P["tidx"], tidx.astype(np.int32))
    np.save(P["tap"], tap)
    np.save(P["hist"], hist)
    np.save(P["alea"], alea)
    np.save(P["alea_blind"], alea_b)
    np.save(P["parent"], lf[L - 1].astype(np.int16))
    volume.commit()
    out = {"pool_size": pool_size, "spec": spec, "slow_argmax_acc": dict(zip(("root", "hi"), acc)),
           "alea_mean": float(alea.mean()), "alea_blind_mean": float(alea_b.mean())}
    with open(f"{d}/h2_{tag}_precompute.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  wrote {P['seqs']} and friends", flush=True)
    return out



@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=65536, cpu=16.0)
def precompute_tap_all(v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                       rule_seed: int = 0, q_k: int = 7, sigma_s: float = 1.0,
                       kappa: float = 0.15, half: float = 2.0, dir_seed: int = 0,
                       pool_size: int = 200000, data_seed: int = 7,
                       n_workers: int = 12, tag: str = "h2"):
    """The block-entry prior over ALL THREE components (root|hi|lo), 3K columns.

    Half 2's first pass restricted the tap to root+hi because Half 1 measured `lo` at 92% of
    its WITHIN-BLOCK ceiling. That conflated two redundancies: decomposing the demand channel
    into its cross-context part (`Dm_window - Dm_history`) gives root +0.00151, hi +0.00482 and
    **lo +0.00746** of a 0.01426-nat joint prize -- so excluding `lo` handed the oracle rung a
    ceiling 2.3x smaller than the endogenous rung's. This regenerates the same corpus
    deterministically and adds the missing columns so the ladder reads on equal footing.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import taps as TP

    L = depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    spec = _spec(q_k, sigma_s, kappa, half, dir_seed, L)
    dirs = DS.make_directions(rules, spec)
    d = qm_dir(v, s, depth, m)
    P = _pool_paths(d, tag)
    seqs, _lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, pool_size, data_seed,
                                            world="drift")
    ref = np.load(P["seqs"]).astype(np.int64)
    assert np.array_equal(ref, seqs), "corpus regeneration is not deterministic"
    print(f"  corpus regenerated bit-identically to {P['seqs']}", flush=True)
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw_s = [np.stack([W[g][1][dd] for g in range(G)]) for dd in range(L)]
    rp_s = np.stack([W[g][0] for g in range(G)])
    t0 = time.time()
    b = np.linspace(0, pool_size, n_workers + 1).astype(int)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_bll_shard, [(rules, seqs[a:c], rw_s, rp_s, 128)
                                         for a, c in zip(b[:-1], b[1:])]))
    bll = np.concatenate(parts, axis=0)
    print(f"  block_loglik in {time.time() - t0:.0f}s", flush=True)
    idx_states, pi = DS.state_grid(spec)
    Pt = DS.joint_transition(spec)
    names = [nm for nm, _ in spec["components"]]
    ohs = [np.eye(spec["K"])[idx_states[:, names.index(c)]] for c in ("root", "hi", "lo")]
    margs = TP.slow_block_posteriors(bll, pi, Pt, ohs, entry=True)
    tap_all = np.concatenate(margs, axis=1).astype(np.float32)
    acc = {c: float((margs[k].argmax(1) == idx_states[tidx][:, names.index(c)]).mean())
           for k, c in enumerate(("root", "hi", "lo"))}
    print(f"  block-entry argmax accuracy: {acc}", flush=True)
    old = np.load(P["tap"])
    err = float(np.abs(tap_all[:, :old.shape[1]] - old).max())
    print(f"  agreement with the cached root+hi tap: max|delta| = {err:.2e}", flush=True)
    assert err < 1e-6
    np.save(P["tap_all"], tap_all)
    volume.commit()
    return {"argmax_acc": acc, "cols": int(tap_all.shape[1])}


# --------------------------------------------------------------------------- #
# the two learned pieces (shared by train_arm and evaluate)
# --------------------------------------------------------------------------- #

def build_tap_classes():
    import torch
    import torch.nn as nn

    class Tap(nn.Module):
        """Injects a per-(batch, position) conditioning vector into the token embedding via a
        forward hook, so `rhm/model.py` is untouched. Zero-initialised, so the arm is exactly
        the plain twin at step 0 and any difference is what the tap carries."""

        def __init__(self, gpt, d_in, d_out):
            super().__init__()
            self.proj = nn.Linear(d_in, d_out)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.cur = None
            gpt.transformer.wte.register_forward_hook(self._hook)

        def _hook(self, _mod, _inp, out):
            return out if self.cur is None else out + self.proj(self.cur)

    class SlowEstimator(nn.Module):
        """A learned leaky integrator over previous blocks' token histograms -- the endogenous
        rung. Its memory is `mem_blocks` blocks (2048 tokens), deliberately longer than the
        model's context: the claim under test is that this is the object a windowed learner
        structurally cannot build in-context. No theta label anywhere; the LM loss is its only
        teacher."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.enc = nn.Linear(d_in, d_out)
            self.rate = nn.Parameter(torch.zeros(d_out))

        def forward(self, summ):                      # (B, J, d_in) -> (B, J, d_out)
            a = torch.sigmoid(self.rate)
            u = torch.tanh(self.enc(summ))
            h = torch.zeros(summ.shape[0], u.shape[-1], device=summ.device)
            outs = []
            for j in range(summ.shape[1]):
                h = (1 - a) * h + a * u[:, j]
                outs.append(h)
            return torch.stack(outs, 1)

    return Tap, SlowEstimator


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=65536)
def train_arm(
    arm: str = "plain",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    q_k: int = 7, sigma_s: float = 1.0, kappa: float = 0.15, half: float = 2.0,
    dir_seed: int = 0,
    base_steps: int = 12000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, alea_gamma: float = 0.0, alea_ratio: float = 4.0,
    mem_blocks: int = 32, d_state: int = 32,
    log_interval: int = 1000, curve_interval: int = 500,
    stream_seed: int = -1,
    seed: int = 42, tag: str = "h2",
):
    """One arm. Identical to Half 1's `train_world` except for the arm's own ingredient.

    `stream_seed` moves ONLY the window sampler -- same data, same initialisation, same
    everything else, a different stream position. Duplicating an arm across two stream seeds
    is `recital`'s convention for measuring a noise floor, and it is what gives the tap
    family's +-0.007 spread a measured denominator instead of a bracketing control. It is a
    noise calibration, not a seed replication of the experiment.
    """
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT

    device = "cuda"
    L, T = depth, s ** depth
    d = qm_dir(v, s, depth, m)
    P = _pool_paths(d, tag)
    if not os.path.exists(P["seqs"]):
        raise FileNotFoundError(f"no precomputed pool at {P['seqs']}; run precompute first")
    seqs = np.load(P["seqs"]).astype(np.int64)
    n_blocks = seqs.shape[0]
    corpus = torch.from_numpy(seqs.reshape(-1))
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)
    print(f"{'=' * 78}\nHALF 2 -- arm `{arm}`   pool {n_blocks:,} blocks / {n_corpus:,} tokens")

    stream_seed = seed if stream_seed < 0 else stream_seed
    label = arm if stream_seed == seed else f"{arm}_str{stream_seed}"
    tap_arr = est_hist = alea_arr = None
    d_tap = 0
    if arm in TAP_SLICE:
        src = P["tap"] if arm in ("tap_oracle", "tap_shuffled") else P["tap_all"]
        tap_arr = np.load(src)
        if arm in ("tap_oracle_full", "tap_oracle_lo"):
            k0, k1 = TAP_SLICE[arm]
            tap_arr = tap_arr[:, k0 * q_k:k1 * q_k]
        if arm == "tap_shuffled":
            rr = np.random.default_rng(seed + 31)
            tap_arr = tap_arr[rr.permutation(n_blocks)]
        d_tap = tap_arr.shape[1]
        tap_t = torch.from_numpy(tap_arr)
    elif arm in ("tap_learned", "tap_learned_shuffled"):
        h_arr = np.load(P["hist"])
        if arm == "tap_learned_shuffled":
            # capacity-matched null for the endogenous rung: the estimator sees histograms
            # from RANDOM blocks, so it has the same parameters and the same input
            # distribution and no information about the local question.
            h_arr = h_arr[np.random.default_rng(seed + 77).permutation(n_blocks)]
        est_hist = torch.from_numpy(h_arr)
        d_tap = d_state
    elif arm in ("loss_alea", "loss_alea_blind"):
        alea_arr = np.load(P["alea"] if arm == "loss_alea" else P["alea_blind"])
        # gamma set BY MEASUREMENT: the weight ratio between the top and bottom aleatoric
        # decile is pinned at `alea_ratio`, so the two loss arms differ only in the LABEL
        # and not in how hard either of them pushes.
        q = np.quantile(alea_arr, [0.1, 0.9])
        if alea_gamma <= 0:
            alea_gamma = float(np.log(alea_ratio) / max(q[1] - q[0], 1e-6))
        wq = np.exp(-alea_gamma * alea_arr)
        print(f"  aleatoric label deciles {q[0]:.4f}/{q[1]:.4f} -> gamma {alea_gamma:.4f} "
              f"(top/bottom weight ratio {alea_ratio:g}); weight mean {wq.mean():.4f}")
        alea_t = torch.from_numpy((wq / wq.mean()).astype(np.float32))
    print(f"  d_tap={d_tap}  base_steps={base_steps}  seed={seed}  "
          f"stream_seed={stream_seed}  label={label}", flush=True)

    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)

    Tap, SlowEstimator = build_tap_classes()

    tap_mod = Tap(model, d_tap, n_embd).to(device) if d_tap else None
    est = (SlowEstimator(v, d_state).to(device)
           if arm in ("tap_learned", "tap_learned_shuffled") else None)
    params = list(model.parameters())
    if tap_mod is not None:
        params += list(tap_mod.parameters())
    if est is not None:
        params += list(est.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(stream_seed)
    pos_level = np.array([min(e for e in range(L + 1) if (p + 1) % (s ** (L - e)) == 0)
                          for p in range(T)], dtype=np.int64)
    lvl_t = torch.from_numpy(pos_level[:T - 1]).to(device)

    def batch(g):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=g)
        idx = ix[:, None] + arangeT[None, :]
        x, y = corpus[idx].to(device), corpus[idx + 1].to(device)
        blk = (idx // T)
        cur = None
        if arm in TAP_SLICE:
            cur = tap_t[blk].to(device)
        elif arm in ("tap_learned", "tap_learned_shuffled"):
            b0 = (ix // T)
            js = torch.clamp(b0[:, None] + torch.arange(-mem_blocks, 1)[None, :], min=0)
            st = est(est_hist[js].to(device))                     # (B, mem+1, d_state)
            two = torch.stack([st[:, -2], st[:, -1]], 1)          # entry for b0, for b0+1
            sel = (blk - b0[:, None]).clamp(0, 1).to(device)
            cur = torch.gather(two, 1, sel[..., None].expand(-1, -1, two.shape[-1]))
        w = None
        if alea_arr is not None:
            w = alea_t[blk, (idx + 1) % T].to(device)
        return x, y, cur, w

    print(f"--- training {base_steps} steps ---", flush=True)
    t0 = time.time()
    curves = []
    for step in range(base_steps):
        model.train()
        x, y, cur, w = batch(gen)
        if tap_mod is not None:
            tap_mod.cur = cur
        logits, _ = model(x, y)
        nll = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                              reduction="none").reshape(x.shape[0], T)
        loss = nll.mean() if w is None else (nll * w).mean()
        if tap_mod is not None:
            tap_mod.cur = None
        opt.zero_grad(); loss.backward(); opt.step()
        if step % curve_interval == 0 or step == base_steps - 1:
            with torch.no_grad():
                per = nll[:, :T - 1].detach()
                row = {"step": step, "loss": float(loss.item()),
                       "nll": float(per.mean().item())}
                for e in range(L + 1):
                    msk = lvl_t == e
                    if msk.any():
                        row[f"nll_lvl{e}"] = float(per[:, msk].mean().item())
                curves.append(row)
        if step % log_interval == 0 or step == base_steps - 1:
            print(f"  {label} {step:6d}  loss {loss.item():.4f}  nll {nll.mean().item():.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    ck = (f"{d}/h2_{tag}_{label}_{n_layer}L{n_head}H{n_embd}D_"
          f"steps{base_steps}_seed{seed}.pt")
    sd = {"model": model.state_dict(), "arm": arm, "label": label, "d_tap": d_tap,
          "stream_seed": stream_seed, "curves": curves,
          "config": {"v": v, "s": s, "L": L, "m": m, "base_steps": base_steps,
                     "seed": seed, "alea_gamma": alea_gamma, "mem_blocks": mem_blocks,
                     "d_state": d_state, "pool_size": n_blocks}}
    if tap_mod is not None:
        sd["tap"] = tap_mod.state_dict()
    if est is not None:
        sd["est"] = est.state_dict()
    torch.save(sd, ck)
    volume.commit()
    print(f"  saved -> {ck}", flush=True)
    return {"arm": label, "ckpt": ck, "final_nll": curves[-1]["nll"]}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=131072, cpu=12.0)
def evaluate(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256, deep_block: str = "post_block6",
    q_k: int = 7, sigma_s: float = 1.0, kappa: float = 0.15, half: float = 2.0,
    dir_seed: int = 0, base_steps: int = 12000, batch_size: int = 64,
    mem_blocks: int = 32, d_state: int = 32,
    arms: str = ("plain,tap_oracle,tap_oracle_full,tap_oracle_lo,tap_shuffled,"
                 "tap_learned,tap_learned_shuffled,loss_alea,loss_alea_blind,"
                 "plain_str43,tap_learned_str43"),
    n_eval: int = 9000, n_filter: int = 1200, n_val_batches: int = 200,
    sk_probe_steps: int = 2500, sk_probe_hidden: int = 256, sk_lr: float = 1e-3,
    n_workers: int = 12, eval_seed: int = 4321, seed: int = 42, tag: str = "h2",
):
    """Half-2 readouts, in the SPEC's priority order:
      1. epistemic self-knowledge under demand shift (the frozen/maintained ledger bracket,
         with the activation probe alongside as the truth-organ arm);
      2. gradient economics against the exact per-position aleatoric labels;
      3. the internal-estimate contrast (tap vs the plain twin's implicit estimate);
      4. the `delta_norm` control the coordinator asked for.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.conditional_revision.gates_ab import _auc, _partial_r2, _partial_r2_rank, _r2
    from rhm.question_model import demand_state as DS
    from rhm.question_model import qfilter as QF
    from rhm.question_model import taps as TP

    device = "cuda"
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    spec = _spec(q_k, sigma_s, kappa, half, dir_seed, L)
    dirs = DS.make_directions(rules, spec)
    d = qm_dir(v, s, depth, m)
    alist = [a.strip() for a in arms.split(",") if a.strip()]
    R = {"spec": spec, "arms": alist, "config": {"n_eval": n_eval, "base_steps": base_steps,
                                                 "seed": seed, "eval_seed": eval_seed}}
    idx_states, pi = DS.state_grid(spec)
    Pt = DS.joint_transition(spec)
    names = [nm for nm, _ in spec["components"]]
    ci_root, ci_hi = names.index("root"), names.index("hi")
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw_s = [np.stack([W[g][1][dd] for g in range(G)]) for dd in range(L)]
    rp_s = np.stack([W[g][0] for g in range(G)])
    grid = DS.grid(spec)

    print(f"{'=' * 78}\nHALF 2 -- EVALUATE  tag={tag}  arms={alist}\n{'=' * 78}", flush=True)

    # ---------------- eval corpus + every exact label ----------------
    t0 = time.time()
    ev_seqs, ev_lf, _lr, ev_tidx = DS.sample_corpus(rules, spec, dirs, n_eval, eval_seed,
                                                    world="drift")
    bounds = np.linspace(0, n_eval, n_workers + 1).astype(int)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_bll_shard,
                            [(rules, ev_seqs[a:b], rw_s, rp_s, 128)
                             for a, b in zip(bounds[:-1], bounds[1:])]))
    bll = np.concatenate(parts, axis=0)
    ci_lo = names.index("lo")
    ohs = [np.eye(spec["K"])[idx_states[:, c]] for c in (ci_root, ci_hi, ci_lo)]
    margs = TP.slow_block_posteriors(bll, pi, Pt, ohs, entry=True)
    ev_tap_all = np.concatenate(margs, axis=1).astype(np.float32)
    ev_tap = ev_tap_all[:, :2 * spec["K"]]
    ev_theta = idx_states[ev_tidx]                                   # (n_eval, C)
    tap_acc = {"root": float((margs[0].argmax(1) == ev_theta[:, ci_root]).mean()),
               "hi": float((margs[1].argmax(1) == ev_theta[:, ci_hi]).mean()),
               "lo": float((margs[2].argmax(1) == ev_theta[:, ci_lo]).mean())}
    ev_hist = (np.eye(v, dtype=np.float32)[ev_seqs].sum(1) / T).astype(np.float32)
    rw_true = np.stack([W[i][1][L - 1] for i in ev_tidx])
    ev_alea = TP.aleatoric_label_tokens(rules, ev_seqs, ev_lf[L - 1], rw_true)
    ev_alea_b = TP.aleatoric_label_tokens(rules, ev_seqs, ev_lf[L - 1], None)
    print(f"  eval corpus + labels in {time.time() - t0:.0f}s; slow-tap argmax accuracy "
          f"{tap_acc}", flush=True)
    R["tap_argmax_acc"] = tap_acc

    pos_level = np.array([min(e for e in range(L + 1) if (p + 1) % (s ** (L - e)) == 0)
                          for p in range(T)], dtype=np.int64)
    ev_x = torch.from_numpy(ev_seqs.astype(np.int64))
    tap_t = torch.from_numpy(ev_tap)
    tap_all_t = torch.from_numpy(ev_tap_all)
    hist_t = torch.from_numpy(ev_hist)
    corpus = ev_x.reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    # ---------------- load arms ----------------
    Tap, SlowEstimator = build_tap_classes()
    base_of = {a: (a.split("_str")[0] if "_str" in a else a) for a in alist}
    models, taps_, ests, curves = {}, {}, {}, {}
    for arm in alist:
        ck = (f"{d}/h2_{tag}_{arm}_{n_layer}L{n_head}H{n_embd}D_"
              f"steps{base_steps}_seed{seed}.pt")
        if not os.path.exists(ck):
            print(f"  WARNING missing {ck}; skipping arm {arm}", flush=True)
            continue
        sd = torch.load(ck, map_location=device)
        torch.manual_seed(seed)
        mm = GPT(v, T, n_layer, n_head, n_embd).to(device)
        dt = sd.get("d_tap", 0)
        tp = Tap(mm, dt, n_embd).to(device) if dt else None
        mm.load_state_dict(sd["model"])
        if tp is not None:
            tp.load_state_dict(sd["tap"])
            tp.eval()
        es = None
        if "est" in sd:
            es = SlowEstimator(v, d_state).to(device)
            es.load_state_dict(sd["est"])
            es.eval()
        mm.eval()
        for p_ in mm.parameters():
            p_.requires_grad_(False)
        models[arm], taps_[arm], ests[arm] = mm, tp, es
        curves[arm] = sd.get("curves", [])
    alist = [a for a in alist if a in models]
    R["training_curves"] = curves

    # `tap_shuffled` is evaluated on shuffled vectors too: it was trained to treat this input
    # as noise, and handing it a suddenly-informative one at eval would test a different thing.
    _perm = np.random.default_rng(seed + 91).permutation(n_eval)
    _K = spec["K"]
    tap_by_arm = {"tap_oracle": tap_t, "tap_shuffled": tap_t[torch.from_numpy(_perm)],
                  "tap_oracle_full": tap_all_t,
                  "tap_oracle_lo": tap_all_t[:, 2 * _K:3 * _K].contiguous()}
    _hp = torch.from_numpy(np.random.default_rng(seed + 77).permutation(n_eval))
    hist_by_arm = {"tap_learned": hist_t, "tap_learned_shuffled": hist_t[_hp]}

    def tap_for(arm, blk_idx, dev):
        """(B, T, d_tap) conditioning payload for the given per-token block indices."""
        if taps_[arm] is None:
            return None
        b = base_of[arm]
        if b in tap_by_arm:
            return tap_by_arm[b][blk_idx].to(dev)
        b0 = blk_idx[:, 0]
        js = torch.clamp(b0[:, None] + torch.arange(-mem_blocks, 1)[None, :], min=0)
        st = ests[arm](hist_by_arm[base_of[arm]][js].to(dev))
        two = torch.stack([st[:, -2], st[:, -1]], 1)
        sel = (blk_idx - b0[:, None]).clamp(0, 1).to(dev)
        return torch.gather(two, 1, sel[..., None].expand(-1, -1, two.shape[-1]))

    # ---------------- E1 / E3: val loss, per level, and gradient economics ----
    print(f"\n--- E1/E3: held-out loss and gradient economics ---", flush=True)
    g = torch.Generator().manual_seed(777)
    win_ix = torch.randint(0, n_corpus - T - 1, (n_val_batches * batch_size,), generator=g)
    alea_flat = torch.from_numpy(ev_alea.reshape(-1).astype(np.float32))
    alea_b_flat = torch.from_numpy(ev_alea_b.reshape(-1).astype(np.float32))
    # The aleatoric label is ATOMIC -- a large atom sits at ln m (a position whose synonym is
    # entirely free) and another at 0 -- so a strict `> q90` group is empty and reads 0.0000.
    # Groups are therefore defined by the atoms themselves.
    q_hi, q_lo = np.quantile(ev_alea, 0.75), np.quantile(ev_alea, 0.25)
    hi_alea = (alea_flat >= q_hi).float()
    lo_alea = (alea_flat <= q_lo).float()
    frac_hi, frac_lo = float(hi_alea.mean()), float(lo_alea.mean())
    print(f"  aleatoric groups: irreducible (label >= {q_hi:.4f}) {frac_hi:.3f} of positions;"
          f" determined (<= {q_lo:.4f}) {frac_lo:.3f}", flush=True)
    econ = {}
    for arm in alist:
        tot = tot_g = 0.0
        lvl_n = np.zeros(L + 1); lvl_s = np.zeros(L + 1)
        gh = gl = nh = nl = 0.0
        with torch.no_grad():
            for b in range(n_val_batches):
                ix = win_ix[b * batch_size:(b + 1) * batch_size]
                idx = ix[:, None] + arangeT[None, :]
                x, y = corpus[idx].to(device), corpus[idx + 1].to(device)
                blk = idx // T
                if taps_[arm] is not None:
                    taps_[arm].cur = tap_for(arm, blk, device)
                logits, _ = models[arm](x, y)
                if taps_[arm] is not None:
                    taps_[arm].cur = None
                nll = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                      reduction="none").reshape(x.shape[0], T)
                pr = F.softmax(logits, -1)
                gn = (pr - F.one_hot(y, v).float()).norm(dim=-1)     # exact d loss / d logits
                tot += float(nll.mean()); tot_g += float(gn.mean())
                lv = pos_level[np.array((idx + 1).reshape(-1) % T)]
                nf = nll.reshape(-1).cpu().numpy()
                for e in range(L + 1):
                    msk = lv == e
                    if msk.any():
                        lvl_s[e] += nf[msk].sum(); lvl_n[e] += msk.sum()
                fi = (idx + 1).reshape(-1)
                ah, al = hi_alea[fi].to(device), lo_alea[fi].to(device)
                gf, nfl = gn.reshape(-1), nll.reshape(-1)
                gh += float((gf * ah).sum()); gl += float(gf.sum())
                nh += float((nfl * ah).sum()); nl += float(nfl.sum())
        econ[arm] = {
            "val_nll": tot / n_val_batches,
            "val_gradnorm": tot_g / n_val_batches,
            "nll_by_level": {f"lvl{e}": float(lvl_s[e] / max(lvl_n[e], 1))
                             for e in range(L + 1)},
            "frac_gradnorm_irreducible": gh / max(gl, 1e-9),
            "frac_nll_irreducible": nh / max(nl, 1e-9),
            "frac_positions_irreducible": frac_hi,
        }
        print(f"  {arm:17s} val nll {econ[arm]['val_nll']:.4f}   grad-norm "
              f"{econ[arm]['val_gradnorm']:.4f}   irreducible share of grad "
              f"{econ[arm]['frac_gradnorm_irreducible']:.4f} / of nll "
              f"{econ[arm]['frac_nll_irreducible']:.4f}"
              f"  (irreducible positions are {frac_hi:.3f} of the total)", flush=True)
    R["economics"] = econ

    # ---------------- E2: epistemic self-knowledge under demand shift --------
    print(f"\n--- E2: epistemic self-knowledge under demand shift ---", flush=True)

    @torch.no_grad()
    def aligned_run(arm, chunk=128):
        """Per-token correctness / entropy / nll on ALIGNED blocks, plus the deep state."""
        cor, ent, nl, acts = [], [], [], []
        for i in range(0, n_eval, chunk):
            xa = ev_x[i:i + chunk].to(device)
            ya = torch.cat([xa[:, 1:], xa[:, :1]], 1)
            blk = torch.arange(i, min(i + chunk, n_eval))[:, None].expand(-1, T)
            if taps_[arm] is not None:
                taps_[arm].cur = tap_for(arm, blk, device)
            logits, _, inter = models[arm](xa, ya, return_intermediates=True)
            if taps_[arm] is not None:
                taps_[arm].cur = None
            pr = F.softmax(logits, -1)
            cor.append((logits.argmax(-1)[:, :T - 1] == ya[:, :T - 1]).cpu().numpy())
            ent.append((-(pr * torch.log(pr.clamp_min(1e-30))).sum(-1))[:, :T - 1]
                       .cpu().numpy())
            nl.append(F.cross_entropy(logits.reshape(-1, v), ya.reshape(-1),
                                      reduction="none").reshape(xa.shape[0], T)[:, :T - 1]
                      .cpu().numpy())
            acts.append(inter[deep_block][:, :T - 1].float().cpu())
        return (np.concatenate(cor), np.concatenate(ent), np.concatenate(nl),
                torch.cat(acts))

    hi_idx, root_idx = ev_theta[:, ci_hi], ev_theta[:, ci_root]
    A = np.flatnonzero(hi_idx <= 2)
    B = np.flatnonzero(hi_idx >= 4)
    nBfit = min(800, len(B) // 4)
    B_fit, B_test = B[:nBfit], B[nBfit:]
    A_budget = A[:nBfit]
    rr = np.random.default_rng(seed + 5)
    kl_ab = float(np.mean([DS.block_kl(rules, spec, dirs, grid[ev_theta[b]],
                                       grid[ev_theta[a]])
                           for a, b in zip(rr.choice(A, 40), rr.choice(B, 40))]))
    kl_aa = float(np.mean([DS.block_kl(rules, spec, dirs, grid[ev_theta[b]],
                                       grid[ev_theta[a]])
                           for a, b in zip(rr.choice(A, 40), rr.choice(A, 40))]))
    print(f"  regions by theta_hi: A(<=2) {len(A)} blocks, B(>=4) {len(B)}"
          f"  (fit {len(B_fit)} / test {len(B_test)})")
    print(f"  demand distance A->B {kl_ab:.3f} nats/block vs within-A {kl_aa:.3f}", flush=True)
    R["sk_setup"] = {"n_A": len(A), "n_B": len(B), "n_B_fit": len(B_fit),
                     "n_B_test": len(B_test), "kl_A_to_B": kl_ab, "kl_within_A": kl_aa}

    fam_all = np.tile(pos_level[:T - 1], (n_eval, 1))

    def brier(p, y):
        return float(np.mean((p - y) ** 2))

    def ece(p, y, nb=10):
        e = np.linspace(0, 1, nb + 1)
        b = np.clip(np.digitize(p, e[1:-1]), 0, nb - 1)
        tot = 0.0
        for k in range(nb):
            sel = b == k
            if sel.sum():
                tot += sel.mean() * abs(p[sel].mean() - y[sel].mean())
        return float(tot)

    def score_set(p, y):
        return {"auc": _auc(p, y.astype(bool)), "brier": brier(p, y.astype(float)),
                "ece": ece(p, y.astype(float)), "mean_p": float(p.mean()),
                "base_rate": float(y.mean())}

    def fit_ledger(cor, fam, bucket, alpha=20.0):
        glob = float(cor.mean())
        fam_p = {}
        for f in np.unique(fam):
            sel = fam == f
            fam_p[int(f)] = float((cor[sel].sum() + alpha * glob) / (sel.sum() + alpha))
        cell = {}
        if bucket is not None:
            key = fam.astype(np.int64) * 10000 + bucket.astype(np.int64)
            for k in np.unique(key):
                sel = key == k
                f = int(k // 10000)
                cell[int(k)] = float((cor[sel].sum() + alpha * fam_p[f])
                                     / (sel.sum() + alpha))
        return {"glob": glob, "fam": fam_p, "cell": cell}

    def apply_ledger(led, fam, bucket):
        out = np.array([led["fam"].get(int(f), led["glob"]) for f in fam])
        if bucket is not None and led["cell"]:
            key = fam.astype(np.int64) * 10000 + bucket.astype(np.int64)
            hit = np.array([led["cell"].get(int(k), np.nan) for k in key])
            out = np.where(np.isnan(hit), out, hit)
        return out

    sk = {}
    for arm in [a for a in alist if base_of[a] in ("plain", "tap_oracle",
                                                   "tap_oracle_full", "tap_learned")]:
        cor, ent, nl, acts = aligned_run(arm)
        # --- the arm's OWN estimate of the slow question state (internal-estimate contrast)
        if base_of[arm] in ("tap_oracle", "tap_oracle_full"):
            est_root = ev_tap[:, :spec["K"]].argmax(1)
            est_hi = ev_tap[:, spec["K"]:2 * spec["K"]].argmax(1)
            est_src = "the supplied filter posterior (exact)"
        else:
            if base_of[arm] in ("tap_learned", "tap_learned_shuffled"):
                js = torch.clamp(torch.arange(n_eval)[:, None]
                                 + torch.arange(-mem_blocks, 0)[None, :], min=0)
                with torch.no_grad():
                    feat = ests[arm](hist_by_arm[base_of[arm]][js].to(device))[:, -1].cpu()
                est_src = "a linear readout of the endogenous estimator state"
            else:
                feat = acts[:, -1]
                est_src = "a probe on the model's own deep state at the block's last position"
            d_in = feat.shape[1]
            head = nn.Linear(d_in, 2 * spec["K"]).to(device)
            o = torch.optim.AdamW(head.parameters(), lr=sk_lr, weight_decay=1e-4)
            At = torch.from_numpy(A)
            mu_, sd_ = feat[At].mean(0, keepdim=True), feat[At].std(0, keepdim=True) + 1e-6
            yr = torch.from_numpy(root_idx.astype(np.int64))
            yh = torch.from_numpy(hi_idx.astype(np.int64))
            gg = torch.Generator().manual_seed(seed + 3)
            A_tr, A_ho = A[:int(0.8 * len(A))], A[int(0.8 * len(A)):]
            for st in range(sk_probe_steps):
                jj = torch.from_numpy(rr.choice(A_tr, 256))
                xb = ((feat[jj] - mu_) / sd_).to(device)
                lg = head(xb)
                lo = (F.cross_entropy(lg[:, :spec["K"]], yr[jj].to(device))
                      + F.cross_entropy(lg[:, spec["K"]:], yh[jj].to(device)))
                o.zero_grad(); lo.backward(); o.step()
            with torch.no_grad():
                lg = head(((feat - mu_) / sd_).to(device)).cpu().numpy()
            est_root = lg[:, :spec["K"]].argmax(1)
            est_hi = lg[:, spec["K"]:].argmax(1)
        A_ho = A[int(0.8 * len(A)):]
        est_acc = {"root": float((est_root[B_test] == root_idx[B_test]).mean()),
                   "hi": float((est_hi[B_test] == hi_idx[B_test]).mean()),
                   "root_in_region_A": float((est_root[A_ho] == root_idx[A_ho]).mean()),
                   "hi_in_region_A": float((est_hi[A_ho] == hi_idx[A_ho]).mean())}
        bucket_est = (est_root * spec["K"] + est_hi)
        bucket_true = (root_idx * spec["K"] + hi_idx)

        res = {"internal_estimate": {"source": est_src, "acc_on_B_test": est_acc},
               "accuracy_A": float(cor[A].mean()), "accuracy_B_test": float(cor[B_test].mean())}
        # how much of the model's competence is demand-dependent at all
        acc_blk = cor.mean(1)
        by_hi = {int(k): float(acc_blk[hi_idx == k].mean()) for k in np.unique(hi_idx)}
        by_root = {int(k): float(acc_blk[root_idx == k].mean()) for k in np.unique(root_idx)}
        fam_flat, cor_flat = fam_all.ravel(), cor.ravel()
        bt = np.repeat(bucket_true, T - 1)
        cell_fam = np.array([float(cor_flat[fam_flat == f].mean())
                             for f in np.unique(fam_flat)])
        ss_fam = float(np.var([cor_flat[fam_flat == f].mean()
                               for f in np.unique(fam_flat)]))
        key = fam_flat.astype(np.int64) * 10000 + bt
        cellm = np.array([cor_flat[key == k].mean() for k in np.unique(key)])
        res["demand_dependence"] = {
            "accuracy_by_theta_hi": by_hi, "accuracy_by_theta_root": by_root,
            "spread_by_theta_hi": float(max(by_hi.values()) - min(by_hi.values())),
            "var_across_families": ss_fam,
            "var_across_family_x_demand_cells": float(np.var(cellm)),
            "family_accuracy_profile": cell_fam.tolist()}

        def blk_flat(a, blocks):
            return a[blocks].reshape(-1)
        yB = blk_flat(cor, B_test).astype(np.int64)
        famB = blk_flat(fam_all, B_test)
        preds = {"entropy_incumbent": -blk_flat(ent, B_test)}
        for lbl, fit_blocks in (("frozen_A", A), ("frozen_A_budget", A_budget),
                                ("maintained_Bfit", B_fit)):
            cy, cf = blk_flat(cor, fit_blocks), blk_flat(fam_all, fit_blocks)
            cbt = np.repeat(bucket_true[fit_blocks], T - 1)
            cbe = np.repeat(bucket_est[fit_blocks], T - 1)
            preds[f"ledger_family::{lbl}"] = apply_ledger(
                fit_ledger(cy, cf, None), famB, None)
            preds[f"ledger_demand_true::{lbl}"] = apply_ledger(
                fit_ledger(cy, cf, cbt), famB, np.repeat(bucket_true[B_test], T - 1))
            preds[f"ledger_demand_est::{lbl}"] = apply_ledger(
                fit_ledger(cy, cf, cbe), famB, np.repeat(bucket_est[B_test], T - 1))
            # the truth-organ arm: an activation probe on the same data budget
            Xf = acts[torch.from_numpy(fit_blocks)].reshape(-1, n_embd)
            yf = torch.from_numpy(cy.astype(np.float32))
            mu2, sd2 = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True) + 1e-6
            pnet = nn.Sequential(nn.Linear(n_embd, sk_probe_hidden), nn.GELU(),
                                 nn.Linear(sk_probe_hidden, 1)).to(device)
            po = torch.optim.AdamW(pnet.parameters(), lr=sk_lr, weight_decay=1e-4)
            for st in range(sk_probe_steps):
                jj = torch.randint(0, Xf.shape[0], (512,))
                lo = F.binary_cross_entropy_with_logits(
                    pnet(((Xf[jj] - mu2) / sd2).to(device))[:, 0], yf[jj].to(device))
                po.zero_grad(); lo.backward(); po.step()
            with torch.no_grad():
                Xt = acts[torch.from_numpy(B_test)].reshape(-1, n_embd)
                out = []
                for i in range(0, Xt.shape[0], 8192):
                    out.append(torch.sigmoid(
                        pnet(((Xt[i:i + 8192] - mu2) / sd2).to(device))[:, 0]).cpu())
                preds[f"probe_activation::{lbl}"] = torch.cat(out).numpy()
        res["scores"] = {k: score_set(p, yB) for k, p in preds.items()}
        sk[arm] = res
        dd = res["demand_dependence"]
        print(f"  [{arm}] internal theta estimate  in-region A: root "
              f"{est_acc['root_in_region_A']:.3f} hi {est_acc['hi_in_region_A']:.3f}   "
              f"out-of-region B: root {est_acc['root']:.3f} hi {est_acc['hi']:.3f}   "
              f"<- {est_src}")
        print(f"    competence spread across theta_hi buckets {dd['spread_by_theta_hi']:.4f};"
              f" var across families {dd['var_across_families']:.5f}, across family x demand"
              f" cells {dd['var_across_family_x_demand_cells']:.5f}")
        print(f"    acc A {res['accuracy_A']:.4f} -> B_test {res['accuracy_B_test']:.4f}")
        for k in sorted(res["scores"]):
            sc_ = res["scores"][k]
            print(f"    {k:38s} AUC {sc_['auc']:.4f}  Brier {sc_['brier']:.4f}  "
                  f"ECE {sc_['ece']:.4f}")
        del acts
    R["self_knowledge"] = sk

    # ---------------- E4: the delta_norm control ----------------------------
    # Half 1 measured partial R^2(delta_norm ~ Dm | marginal surprisal) = 0.088 (0.093 on
    # Dm_lo), higher than any probe-based readout, which is against this repo's
    # four-times-replicated directional-not-scalar pattern. Two candidate readings: the demand
    # currency really is scalar-coded in the residual stream, or the scalar is riding
    # position/token-frequency structure that a single surprisal covariate does not remove.
    # This block adds those covariates and re-reads.
    print(f"\n--- E4: does delta_norm's demand-tracking survive richer controls? ---",
          flush=True)
    nf = min(n_filter, n_eval)
    llf = QF.prefix_loglik(rules, ev_seqs[:nf], rw_s, rp_s, chunk=50)
    dcw = QF.demand_channel(llf, pi, state_idx=idx_states, n_comp=len(names), K=spec["K"])
    Dm = dcw["Dm"].ravel()
    Dm_lo = dcw["Dm_comp"][names.index("lo")].ravel()
    surp = dcw["surprisal"].ravel()
    del llf
    ctrl_arm = "plain" if "plain" in models else alist[0]

    @torch.no_grad()
    def delta_norms(arm, chunk=128):
        out = []
        for i in range(0, nf, chunk):
            j = min(i + chunk, nf)
            xa = ev_x[i:j].to(device)
            blk = torch.arange(i, j)[:, None].expand(-1, T)
            if taps_[arm] is not None:
                taps_[arm].cur = tap_for(arm, blk, device)
            _, _, inter = models[arm](xa, return_intermediates=True)
            if taps_[arm] is not None:
                taps_[arm].cur = None
            h = inter[deep_block]
            out.append((h[:, 1:] - h[:, :-1]).norm(dim=-1).cpu().numpy())
        return np.concatenate(out)

    dn = delta_norms(ctrl_arm).ravel()
    pos = np.tile(np.arange(T - 1), (nf, 1)).ravel()
    unig = np.bincount(ev_seqs.ravel(), minlength=v) / ev_seqs.size
    tokfreq = np.log(unig[ev_seqs[:nf, 1:]].ravel())
    alea_f = ev_alea[:nf, 1:].ravel()

    def resid_multi(y, X):
        Xd = np.column_stack([np.ones_like(y)] + list(X))
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        return y - Xd @ beta

    def partial_multi(y, x, X):
        return _r2(resid_multi(y, X), resid_multi(x, X))

    lvlpos = pos_level[1:][pos]                       # level completed by x_{t+1}
    lvl_oh = [np.asarray(lvlpos == e, float) for e in range(1, L + 1)]
    ctrl_sets = {
        "surprisal only (Half-1 reading)": [surp],
        "+ token log-frequency": [surp, tokfreq],
        "+ hierarchy level (dummies)": [surp, tokfreq] + lvl_oh,
        "+ exact aleatoric label": [surp, tokfreq] + lvl_oh + [alea_f],
    }
    e4 = {"arm": ctrl_arm, "n_blocks": nf}
    for lbl, X in ctrl_sets.items():
        e4[lbl] = {"delta_norm~Dm": partial_multi(dn, Dm, X),
                   "delta_norm~Dm_lo": partial_multi(dn, Dm_lo, X)}
        print(f"  controls: {lbl:34s} delta_norm~Dm {e4[lbl]['delta_norm~Dm']:.4f}   "
              f"~Dm_lo {e4[lbl]['delta_norm~Dm_lo']:.4f}")
    # the strictest version: WITHIN each position, so position is removed exactly
    wp, wp_lo, wn = 0.0, 0.0, 0
    for t in range(T - 1):
        sel = pos == t
        if sel.sum() > 50:
            wp += _r2(resid_multi(dn[sel], [surp[sel], tokfreq[sel], alea_f[sel]]),
                      resid_multi(Dm[sel], [surp[sel], tokfreq[sel], alea_f[sel]]))
            wp_lo += _r2(resid_multi(dn[sel], [surp[sel], tokfreq[sel], alea_f[sel]]),
                         resid_multi(Dm_lo[sel], [surp[sel], tokfreq[sel], alea_f[sel]]))
            wn += 1
    e4["within_position_mean"] = {"delta_norm~Dm": wp / max(wn, 1),
                                  "delta_norm~Dm_lo": wp_lo / max(wn, 1)}
    print(f"  within-position mean (surprisal + freq + aleatoric partialled): "
          f"Dm {wp / max(wn, 1):.4f}   Dm_lo {wp_lo / max(wn, 1):.4f}", flush=True)
    R["delta_norm_control"] = e4

    nf_pairs = [(a, f"{a}_str43") for a in ("plain", "tap_learned")
                if a in econ and f"{a}_str43" in econ]
    if nf_pairs:
        R["noise_floor"] = {}
        print(f"\n--- the paired noise floor (same data and init, different stream "
              f"position) ---", flush=True)
        for a, b in nf_pairs:
            dd_ = econ[b]["val_nll"] - econ[a]["val_nll"]
            lvd = {k: econ[b]["nll_by_level"][k] - econ[a]["nll_by_level"][k]
                   for k in econ[a]["nll_by_level"]}
            R["noise_floor"][a] = {"delta_val_nll": dd_, "delta_by_level": lvd}
            print(f"  {a} vs {b}: delta val nll {dd_:+.5f}   per level "
                  + " ".join(f"{k}{x:+.4f}" for k, x in lvd.items()), flush=True)
    out_p = f"{d}/h2_{tag}_evaluate.json"
    with open(out_p, "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_p}", flush=True)
    return {"tag": tag, "ok": True}


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def selfcheck():
    from rhm.question_model import taps as TP
    print("=" * 74)
    print("Half-2 instrument gates")
    print("=" * 74)
    TP._test_block_loglik()
    TP._test_aleatoric_label()
    print("\nALL GATES PASSED", flush=True)
    return {"passed": True}
