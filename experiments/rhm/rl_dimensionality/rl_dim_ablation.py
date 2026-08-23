"""RL dimensionality ablation on the RHM.

Operationalizes: "RL gets dumber relative to the relevant dimensions of the
problem as the semantic dimensionality of the domain grows" — and the sharper
representational question: does RL create capabilities in the basis pretraining
built, or only rearrange (and degrade) it?

The dimensionality knob is m (synonymic multiplicity), swept at fixed
(v=8, s=2, L=6). m moves the semantic sample-complexity quantity m^L while
holding sequence length, horizon, architecture, and compute per rollout exactly
fixed (unlike L or s, which change s^L). With invertible rules, m=1 is the
"game" endpoint (8 possible sequences, deterministic surface) and m=v^(s-1)=8
is the structureless endpoint (exactly uniform over all sequences), so the
sweep runs m in {1,2,3,4,6}.

Per m, three conditions (all sharing init weights, data streams, and RL
gradient-step budget):
  scratch_rl    — random init -> REINFORCE on the task reward
  pretrain_rl   — NTP pretrain (to plateau) -> pure REINFORCE (no NTP anchor)
  pretrain_only — NTP pretrain (to plateau) -> continued NTP, step-matched

Task: given the first 32 tokens of a 64-token RHM sequence, generate the
remaining 32 autoregressively. Two reward types:
  exact — fraction of suffix tokens matching the sampled ground-truth suffix
          (the ratchet-line reward; verifier demands one canonical surface form)
  parse — mean over hierarchy levels of the fraction of valid nodes when
          bottom-up parsing prefix+generation (synonymy-invariant; verifier
          accepts any grammatical continuation, graded in composition depth)

"Relative to the relevant dimensions" is made exact with sum-product BP on the
known parse tree (rhm_bayes_entropy machinery):
  - NTP Bayes floor per level (aligned sequences)
  - exact-match Bayes ceiling: one BP pass with the prefix observed gives all
    suffix marginals; greedy ceiling = mean_i max_a P(x_i=a|prefix), and the
    sampled-policy ceiling = mean_i sum_a P^2
  - prefix-blind floor: best prefix-ignoring constant predictor (unconditional
    position marginals), plus the uniform floor 1/v
  - parse floors by Monte Carlo (uniform-random suffix; blind-argmax suffix)
Headline metric: fraction of headroom recovered, (achieved - floor)/(ceiling - floor).

Representational readouts at every checkpoint: per-layer feature eta^2 against
ground-truth hierarchy levels, per-level NTP loss vs the Bayes floor (excess
loss = frying), and per-level generation accuracy.

Reproduction (from experiments/):
  # self-tests (CPU, ~1 min)
  modal run -m rhm.rl_dimensionality.rl_dim_ablation::self_test
  # smoke (GPU, ~5 min)
  modal run -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m 2 --smoke
  # full run for one m (detached)
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m 2
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

app = modal.App("rhm-rl-dim", image=image)


# ======================================================================
# Hierarchy helpers (copied from rhm.ratchet.rhm_rl_ratchet, which is an
# abandoned line we don't want a live import against)
# ======================================================================

def _position_levels(seq_len, s):
    """s-adic valuation for each prediction position 1..seq_len-1."""
    import numpy as np
    levels = np.zeros(seq_len - 1, dtype=np.int64)
    for t in range(seq_len - 1):
        p = t + 1
        level = 0
        while p % s == 0:
            p //= s
            level += 1
        levels[t] = level
    return levels


def _suffix_position_levels(prefix_len, seq_len, s):
    """Hierarchy level for each suffix position."""
    levels = []
    for i in range(prefix_len, seq_len):
        p = i
        level = 0
        while p > 0 and p % s == 0:
            p //= s
            level += 1
        levels.append(level)
    return levels


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
    sequences = current
    level_features.append(sequences.copy())
    return sequences, level_features, level_rules


def _compute_hierarchy_eta2(acts_np, level_features, level_rules, s, L):
    """Feature/rule eta^2 from activations at the last sequence position."""
    import numpy as np
    n_seq, seq_len, d_model = acts_np.shape
    last = acts_np[:, -1, :]
    last_mean = last.mean(axis=0)
    ss_total = float(np.sum((last - last_mean) ** 2))
    if ss_total < 1e-12:
        return {f"level_{ell}": {"feature_eta2": 0.0, "rule_eta2": 0.0}
                for ell in range(L)}

    def eta2_between(data, labels, grand_mean):
        ss = 0.0
        for val in np.unique(labels):
            mask = labels == val
            group_mean = data[mask].mean(axis=0)
            ss += float(mask.sum()) * float(
                np.sum((group_mean - grand_mean) ** 2))
        return ss

    results = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        last_ancestor = (seq_len - 1) // s_power
        results[f"level_{ell}"] = {
            "feature_eta2": eta2_between(
                last, level_features[ell][:, last_ancestor], last_mean
            ) / ss_total,
            "rule_eta2": eta2_between(
                last, level_rules[ell][:, last_ancestor], last_mean
            ) / ss_total,
        }
    return results


# ======================================================================
# Exact reference values via BP (numpy, CPU)
# ======================================================================

def suffix_marginals(rules, seqs, prefix_len):
    """Exact P(x_i = a | x_{<prefix_len}) for every suffix position.

    One upward+downward BP pass per batch with the prefix observed and the
    suffix free. Returns (B, suffix_len, v) normalized posteriors.
    """
    import numpy as np
    from rhm.rhm_bayes_entropy import _upward, _downward
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    B = seqs.shape[0]

    up_leaf = np.ones((B, seq_len, v))
    obs = seqs[:, :prefix_len]
    oh = np.zeros((B, prefix_len, v))
    np.put_along_axis(oh, obs[:, :, None], 1.0, axis=2)
    up_leaf[:, :prefix_len, :] = oh

    up = _upward(up_leaf, rules, B, s, L, v)
    down = _downward(up, rules, B, s, L, v)
    post = up[L][:, prefix_len:, :] * down[L][:, prefix_len:, :]
    post = post / np.clip(post.sum(axis=2, keepdims=True), 1e-30, None)
    return post


def blind_marginals(rules):
    """Unconditional P(x_i = a) per leaf position: BP with nothing observed."""
    import numpy as np
    from rhm.rhm_bayes_entropy import _upward, _downward
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    up_leaf = np.ones((1, seq_len, v))
    up = _upward(up_leaf, rules, 1, s, L, v)
    down = _downward(up, rules, 1, s, L, v)
    post = up[L][0] * down[L][0]
    post = post / np.clip(post.sum(axis=1, keepdims=True), 1e-30, None)
    return post  # (seq_len, v)


def exact_match_references(rules, seqs, prefix_len):
    """Bayes ceiling and floors for the exact-match reward, per level and overall.

    greedy ceiling   = E[ mean_i max_a P(x_i=a | prefix) ]   (best any policy can do)
    sampled ceiling  = E[ mean_i sum_a P(x_i=a | prefix)^2 ] (posterior-sampling policy)
    blind floor      = mean_i max_a P(x_i=a)                  (best prefix-ignoring policy)
    uniform floor    = 1/v
    """
    import numpy as np
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    suffix_len = seq_len - prefix_len
    levels = np.array(_suffix_position_levels(prefix_len, seq_len, s))

    post = suffix_marginals(rules, seqs, prefix_len)      # (B, S, v)
    greedy_pos = post.max(axis=2)                         # (B, S)
    sampled_pos = (post ** 2).sum(axis=2)                 # (B, S)

    blind = blind_marginals(rules)[prefix_len:]           # (S, v)
    blind_pos = blind.max(axis=1)                         # (S,)
    blind_argmax = blind.argmax(axis=1)                   # (S,) for the MC parse floor

    def per_level(pos_vals):
        out = {}
        for lvl in range(int(levels.max()) + 1):
            mask = levels == lvl
            if mask.sum():
                out[f"L{lvl}"] = float(pos_vals[..., mask].mean())
        return out

    return {
        "greedy_ceiling": {"overall": float(greedy_pos.mean()), **per_level(greedy_pos)},
        "sampled_ceiling": {"overall": float(sampled_pos.mean()), **per_level(sampled_pos)},
        "blind_floor": {"overall": float(blind_pos.mean()), **per_level(blind_pos)},
        "uniform_floor": 1.0 / v,
    }, blind_argmax


# ======================================================================
# Parse-based reward (torch, GPU) — graded grammaticality
# ======================================================================

def parse_valid_fractions_torch(seqs, rules_t):
    """Per-level fraction of valid nodes in the bottom-up possible-set parse.

    seqs: (B, s^L) long tensor. rules_t: list of L (v, m, s) long tensors
    (level 0 = top, matching rhm_data conventions). Returns (B, L) float:
    column k = fraction of nodes with a non-empty possible-set after fold k+1
    (k=0 is the first composition above the leaves, k=L-1 is the root).
    Invalid nodes propagate empty sets upward, exactly as in
    rhm_data.possible_set_parse.
    """
    import torch
    import torch.nn.functional as F
    B = seqs.shape[0]
    v, m, s = rules_t[0].shape
    L = len(rules_t)
    cur = F.one_hot(seqs, v).bool()  # (B, n, v)
    fracs = []
    for ell in range(L - 1, -1, -1):
        n = cur.shape[1]
        n2 = n // s
        children = cur.view(B, n2, s, v)
        ok = torch.ones(B, n2, v, m, dtype=torch.bool, device=seqs.device)
        for i in range(s):
            idx = rules_t[ell][:, :, i].reshape(-1)          # (v*m,)
            ok &= children[:, :, i, :][:, :, idx].view(B, n2, v, m)
        cur = ok.any(dim=3)                                   # (B, n2, v)
        fracs.append(cur.any(dim=2).float().mean(dim=1))      # (B,)
    return torch.stack(fracs, dim=1)  # (B, L)


# ======================================================================
# Generation + evaluation
# ======================================================================

def _generate_suffix(model, prefix, suffix_len, temperature=1.0, greedy=False):
    import torch
    import torch.nn.functional as F
    x = prefix
    outs = []
    for _ in range(suffix_len):
        logits, _ = model(x)
        nl = logits[:, -1, :]
        if greedy:
            nt = nl.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(nl / temperature, dim=-1)
            nt = torch.multinomial(probs, num_samples=1)
        outs.append(nt)
        x = torch.cat([x, nt], dim=1)
    return torch.cat(outs, dim=1)


# ======================================================================
# Main experiment: one (m, reward_type, seed)
# ======================================================================

@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def run_setting(
    # DGP
    m: int = 2, v: int = 8, s: int = 2, depth: int = 6,
    rule_seed: int = 0,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # Reward arms to train RL conditions on, run sequentially in this job
    reward_type: str = "both",  # "exact" | "parse" | "both"
    # Pretraining (to plateau)
    pretrain_min_steps: int = 3000,
    pretrain_max_steps: int = 30000,
    pretrain_eval_every: int = 250,
    patience: int = 8,
    min_delta: float = 0.003,
    # RL / finetune
    rl_steps: int = 6000,
    ckpt_interval: int = 1500,
    kl_coef: float = 0.0,
    only_kl: bool = False,          # run ONLY the pretrain_rl_kl_* conditions
    pretrained_path: str = "",      # volume-relative pretrained.pt to load
                                    # instead of pretraining (e.g. from the
                                    # base run's save dir)
    # Expert iteration (class-limit arm): iterated best-of-N + SFT on winners
    only_ei: bool = False,          # run ONLY the ei_* conditions
    ei_rounds: int = 6,
    ei_n: int = 16,                 # rollouts per prompt
    ei_prompts_per_round: int = 4000,   # 6*4000*16 = 384K rollouts = RL budget
    ei_sft_epochs: int = 3,
    temperature: float = 1.0,
    # Training
    lr: float = 3e-4,
    batch_size: int = 64,
    seed: int = 42,
    # Data / eval sizes
    n_train_seqs: int = 100000,
    n_val_seqs: int = 2000,
    n_eval_seqs: int = 5000,
    n_gen_eval: int = 1000,
    n_bp_ntp: int = 400,
    # Misc
    smoke: bool = False,
    run_tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_invertible, generate_sequences_batched
    from rhm.rhm_bayes_entropy import summarize

    if smoke:
        pretrain_min_steps, pretrain_max_steps = 100, 300
        pretrain_eval_every, patience = 50, 3
        rl_steps, ckpt_interval = 60, 30
        n_train_seqs, n_val_seqs = 20000, 500
        n_eval_seqs, n_gen_eval, n_bp_ntp = 512, 128, 50
        ei_rounds, ei_n, ei_prompts_per_round, ei_sft_epochs = 2, 4, 256, 1

    device = "cuda"
    L = depth
    seq_len = s ** L
    prefix_len = seq_len // 2
    suffix_len = seq_len - prefix_len
    key = f"v{v}_s{s}_L{L}_m{m}"
    tag = f"_{run_tag}" if run_tag else ""
    save_dir = f"{DATA_DIR}/rl_dimensionality/{key}_{reward_type}_seed{seed}{tag}"
    os.makedirs(save_dir, exist_ok=True)

    suffix_levels = _suffix_position_levels(prefix_len, seq_len, s)
    max_suffix_level = max(suffix_levels)

    print(f"RL DIMENSIONALITY ABLATION  ({key}, reward={reward_type}, seed={seed})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D | prefix={prefix_len} suffix={suffix_len}")
    print(f"  Pretrain: plateau (min={pretrain_min_steps}, max={pretrain_max_steps}, "
          f"patience={patience}x{pretrain_eval_every} steps, min_delta={min_delta})")
    print(f"  RL: {rl_steps} steps, ckpt every {ckpt_interval}, kl_coef={kl_coef}")

    # ------------------------------------------------------------------
    # Data (invertible rules: exact parses, unique roots)
    # ------------------------------------------------------------------
    rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
    for ell, r in enumerate(rules):
        np.save(os.path.join(save_dir, f"rules_L{ell}.npy"), r)

    train_seqs = generate_sequences_batched(rules, n_train_seqs, seed=seed + 1)
    val_seqs = generate_sequences_batched(rules, n_val_seqs, seed=seed + 2)
    rl_seqs = generate_sequences_batched(rules, n_train_seqs, seed=seed + 3)
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_seqs, seed=12345)
    gen_eval_seqs = generate_sequences_batched(rules, n_gen_eval, seed=seed + 4)
    print(f"  Data: train={n_train_seqs} val={n_val_seqs} eval={n_eval_seqs} "
          f"gen_eval={n_gen_eval} sequences of {seq_len} tokens")

    # ------------------------------------------------------------------
    # Exact reference values (CPU BP)
    # ------------------------------------------------------------------
    print("\nComputing exact reference values (BP)...")
    ntp_floor = summarize(rules, val_seqs[:n_bp_ntp])
    print(f"  NTP Bayes floor: {ntp_floor['entropy_rate_pos1plus_nats']:.4f} nats/token "
          f"(positions 1+); uniform = {np.log(v):.4f}")

    em_refs, blind_argmax_suffix = exact_match_references(
        rules, gen_eval_seqs, prefix_len)
    print(f"  Exact-match: greedy ceiling={em_refs['greedy_ceiling']['overall']:.4f} "
          f"sampled ceiling={em_refs['sampled_ceiling']['overall']:.4f} "
          f"blind floor={em_refs['blind_floor']['overall']:.4f} uniform={1/v:.4f}")

    train_t = torch.from_numpy(train_seqs.astype(np.int64))
    val_t = torch.from_numpy(val_seqs.astype(np.int64)).to(device)
    rl_t = torch.from_numpy(rl_seqs.astype(np.int64))
    gen_eval_t = torch.from_numpy(gen_eval_seqs.astype(np.int64)).to(device)
    eval_t = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    rules_t = [torch.from_numpy(r).long().to(device) for r in rules]
    suffix_levels_t = torch.tensor(suffix_levels)

    # Parse floors (MC on the gen-eval prefixes)
    g = torch.Generator(device=device).manual_seed(seed + 10)
    rand_suffix = torch.randint(v, (n_gen_eval, suffix_len), generator=g,
                                device=device)
    pf_rand = parse_valid_fractions_torch(
        torch.cat([gen_eval_t[:, :prefix_len], rand_suffix], dim=1), rules_t)
    blind_suffix = torch.from_numpy(
        np.tile(blind_argmax_suffix, (n_gen_eval, 1)).astype(np.int64)).to(device)
    pf_blind = parse_valid_fractions_torch(
        torch.cat([gen_eval_t[:, :prefix_len], blind_suffix], dim=1), rules_t)
    parse_refs = {
        "ceiling": 1.0,
        "random_floor": {"overall": float(pf_rand.mean()),
                         "per_level": pf_rand.mean(dim=0).tolist()},
        "blind_floor": {"overall": float(pf_blind.mean()),
                        "per_level": pf_blind.mean(dim=0).tolist()},
    }
    print(f"  Parse: random floor={parse_refs['random_floor']['overall']:.4f} "
          f"blind floor={parse_refs['blind_floor']['overall']:.4f} ceiling=1.0")

    # ------------------------------------------------------------------
    # Batches / model factory
    # ------------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    rl_gen = torch.Generator().manual_seed(seed + 100)
    ei_gen = torch.Generator().manual_seed(seed + 200)

    def ntp_batch():
        idx = torch.randint(n_train_seqs, (batch_size,), generator=train_gen)
        seqs = train_t[idx].to(device)
        # contiguous: model.py's loss uses targets.view(-1)
        return seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous()

    def rl_batch():
        idx = torch.randint(n_train_seqs, (batch_size,), generator=rl_gen)
        seqs = rl_t[idx].to(device)
        return seqs[:, :prefix_len], seqs[:, prefix_len:]

    def make_gpt():
        return GPT(v, seq_len, n_layer, n_head, n_embd).to(device)

    torch.manual_seed(seed)
    _init = make_gpt()
    init_state = {k: p.cpu().clone() for k, p in _init.state_dict().items()}
    del _init
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    rewards = ["exact", "parse"] if reward_type == "both" else [reward_type]

    def compute_reward(prefix, generated, true_suffix, rw):
        if rw == "exact":
            return (generated == true_suffix).float().mean(dim=1)
        elif rw == "parse":
            full = torch.cat([prefix, generated], dim=1)
            return parse_valid_fractions_torch(full, rules_t).mean(dim=1)
        raise ValueError(rw)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval_val_loss(model, n_batches=None):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(val_t), batch_size):
                batch = val_t[i:i + batch_size]
                _, loss = model(batch[:, :-1].contiguous(),
                                batch[:, 1:].contiguous())
                losses.append(loss.item())
                if n_batches and len(losses) >= n_batches:
                    break
        return float(np.mean(losses))

    def eval_per_level_ntp(model):
        """Per-level CE on aligned val sequences (targets = positions 1+)."""
        model.eval()
        levels = _position_levels(seq_len, s)
        per_pos = np.zeros(seq_len - 1)
        n = 0
        with torch.no_grad():
            for i in range(0, len(val_t), batch_size):
                batch = val_t[i:i + batch_size]
                logits, _ = model(batch[:, :-1].contiguous())
                lp = F.log_softmax(logits, dim=-1)
                tlp = lp.gather(2, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
                per_pos += (-tlp).sum(dim=0).cpu().numpy()
                n += batch.shape[0]
        per_pos /= n
        out = {}
        for lvl in range(int(levels.max()) + 1):
            mask = levels == lvl
            if mask.sum():
                out[f"L{lvl}"] = float(per_pos[mask].mean())
        out["overall"] = float(per_pos.mean())
        return out

    def eval_generation(model, label):
        """Greedy + sampled generation: exact-match and parse metrics."""
        model.eval()
        out = {}
        with torch.no_grad():
            for mode in ["greedy", "sampled"]:
                em_correct, pfs = [], []
                for i in range(0, n_gen_eval, batch_size):
                    batch = gen_eval_t[i:i + batch_size]
                    prefix = batch[:, :prefix_len]
                    true_suffix = batch[:, prefix_len:]
                    gen = _generate_suffix(model, prefix, suffix_len,
                                           temperature, greedy=(mode == "greedy"))
                    em_correct.append((gen == true_suffix).float().cpu())
                    pfs.append(parse_valid_fractions_torch(
                        torch.cat([prefix, gen], dim=1), rules_t).cpu())
                em = torch.cat(em_correct)
                pf = torch.cat(pfs)
                res = {"exact_overall": float(em.mean()),
                       "parse_overall": float(pf.mean()),
                       "parse_per_level": pf.mean(dim=0).tolist()}
                for lvl in range(max_suffix_level + 1):
                    mask = suffix_levels_t == lvl
                    if mask.sum():
                        res[f"exact_L{lvl}"] = float(em[:, mask].mean())
                # headroom recovered (blind floor -> ceiling)
                ceil = em_refs["greedy_ceiling" if mode == "greedy"
                               else "sampled_ceiling"]["overall"]
                floor = em_refs["blind_floor"]["overall"]
                res["exact_headroom"] = (
                    (res["exact_overall"] - floor) / (ceil - floor)
                    if ceil - floor > 1e-9 else float("nan"))
                pfloor = parse_refs["blind_floor"]["overall"]
                res["parse_headroom"] = (
                    (res["parse_overall"] - pfloor) / (1.0 - pfloor)
                    if 1.0 - pfloor > 1e-9 else float("nan"))
                out[mode] = res
        print(f"  [{label}] greedy: exact={out['greedy']['exact_overall']:.4f} "
              f"(headroom {out['greedy']['exact_headroom']:.3f}) "
              f"parse={out['greedy']['parse_overall']:.4f} | "
              f"sampled: exact={out['sampled']['exact_overall']:.4f} "
              f"parse={out['sampled']['parse_overall']:.4f}")
        return out

    def eval_per_layer_eta2(model):
        model.eval()
        layer_keys = [f"post_block{i}" for i in range(n_layer)]
        layer_acts = {k: [] for k in layer_keys}
        with torch.no_grad():
            for i in range(0, len(eval_t), batch_size):
                batch = eval_t[i:i + batch_size]
                if batch.shape[0] < 2:
                    continue
                _, _, inter = model(batch, return_intermediates=True)
                for k in layer_keys:
                    layer_acts[k].append(inter[k].cpu().numpy())
        results = {}
        for k in layer_keys:
            acts = np.concatenate(layer_acts[k], axis=0)
            nc = acts.shape[0]
            results[k] = _compute_hierarchy_eta2(
                acts, [lf[:nc] for lf in level_features],
                [lr[:nc] for lr in level_rules], s=s, L=L)
        return results

    def full_eval(model, label):
        return {
            "val_loss": eval_val_loss(model),
            "per_level_ntp": eval_per_level_ntp(model),
            "generation": eval_generation(model, label),
            "per_layer_eta2": eval_per_layer_eta2(model),
        }

    if only_kl and kl_coef <= 0:
        raise ValueError("only_kl requires kl_coef > 0")

    # ------------------------------------------------------------------
    # Phase 1: pretrain to plateau (or load a saved checkpoint)
    # ------------------------------------------------------------------
    if pretrained_path:
        print(f"\n{'='*60}\n  PRETRAIN: loading {pretrained_path}\n{'='*60}")
        best_state = torch.load(f"{DATA_DIR}/{pretrained_path}",
                                map_location="cpu")
        model = make_gpt()
        model.load_state_dict(best_state)
        best_val = eval_val_loss(model)
        pretrain_curve = []
        pretrain_steps_used = 0
        bayes_floor = ntp_floor["entropy_rate_pos1plus_nats"]
        uniform = float(np.log(v))
        captured = ((uniform - best_val) / (uniform - bayes_floor)
                    if uniform - bayes_floor > 1e-9 else float("nan"))
        print(f"  Loaded: val={best_val:.4f}, Bayes floor={bayes_floor:.4f}, "
              f"captured={captured:.3f}")
        print("\n  Pretrain checkpoint eval:")
        pretrain_eval = full_eval(model, "pretrained(loaded)")
        pretrained_state = best_state
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n{'='*60}\n  PRETRAIN (to plateau)\n{'='*60}")
        model = make_gpt()
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        best_val = float("inf")
        best_state = None
        evals_since_best = 0
        pretrain_curve = []
        step = 0
        while step < pretrain_max_steps:
            model.train()
            x, y = ntp_batch()
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % pretrain_eval_every == 0:
                vl = eval_val_loss(model)
                pretrain_curve.append({"step": step, "val_loss": vl})
                if vl < best_val - min_delta:
                    best_val = vl
                    best_state = {k: p.cpu().clone()
                                  for k, p in model.state_dict().items()}
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                if step % (pretrain_eval_every * 4) == 0:
                    print(f"  step {step:6d}: val={vl:.4f} "
                          f"(best={best_val:.4f}, stale={evals_since_best})")
                if step >= pretrain_min_steps and evals_since_best >= patience:
                    print(f"  plateau at step {step} (best val={best_val:.4f})")
                    break
        if best_state is None:
            best_state = {k: p.cpu().clone()
                          for k, p in model.state_dict().items()}
        pretrain_steps_used = step
        model.load_state_dict(best_state)

        bayes_floor = ntp_floor["entropy_rate_pos1plus_nats"]
        uniform = float(np.log(v))
        captured = ((uniform - best_val) / (uniform - bayes_floor)
                    if uniform - bayes_floor > 1e-9 else float("nan"))
        print(f"  Pretrained: {pretrain_steps_used} steps, val={best_val:.4f}, "
              f"Bayes floor={bayes_floor:.4f}, "
              f"captured excess entropy={captured:.3f}")

        print("\n  Pretrain checkpoint eval:")
        pretrain_eval = full_eval(model, "pretrained")
        pretrained_state = best_state
        torch.save(pretrained_state, os.path.join(save_dir, "pretrained.pt"))
        del model, opt
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: conditions
    # ------------------------------------------------------------------
    def run_rl(start_state, cond, rw, use_kl=False):
        model = make_gpt()
        model.load_state_dict(start_state)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        ref_model = None
        if use_kl:
            ref_model = make_gpt()
            ref_model.load_state_dict(start_state)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        baseline = None
        trajectory = []
        checkpoints = {}
        for st in range(rl_steps):
            model.eval()
            prefix, true_suffix = rl_batch()
            with torch.no_grad():
                generated = _generate_suffix(model, prefix, suffix_len,
                                             temperature)
            reward = compute_reward(prefix, generated, true_suffix, rw)
            if baseline is None:
                baseline = reward.mean().item()
            else:
                baseline = 0.95 * baseline + 0.05 * reward.mean().item()
            advantage = (reward - baseline).detach()

            model.train()
            scored = torch.cat([prefix, generated], dim=1)
            logits, _ = model(scored)
            sl = logits[:, prefix_len - 1:prefix_len - 1 + suffix_len, :]
            logp = F.log_softmax(sl, dim=-1)
            gen_logp = logp.gather(2, generated.unsqueeze(-1)).squeeze(-1)
            loss = -(advantage.unsqueeze(1) * gen_logp).mean()

            if use_kl:
                with torch.no_grad():
                    ref_logits, _ = ref_model(scored)
                    ref_sl = ref_logits[:, prefix_len - 1:
                                        prefix_len - 1 + suffix_len, :]
                # KL(pi_theta || pi_ref) = sum pi_theta * (log pi_theta - log pi_ref)
                kl = (logp.exp() * (logp - F.log_softmax(ref_sl, dim=-1))
                      ).sum(-1).mean()
                loss = loss + kl_coef * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (st + 1) % 50 == 0:
                trajectory.append({"step": st + 1,
                                   "reward": float(reward.mean()),
                                   "baseline": float(baseline)})
            if (st + 1) % 250 == 0:
                vq = eval_val_loss(model, n_batches=8)
                trajectory[-1]["val_loss_quick"] = vq
                print(f"    [{cond}] step {st+1:5d}: reward={reward.mean():.4f} "
                      f"(ema {baseline:.4f}) val~{vq:.4f}")
            if (st + 1) % ckpt_interval == 0:
                print(f"  [{cond}] CHECKPOINT step {st+1}")
                checkpoints[st + 1] = full_eval(model, f"{cond} s{st+1}")

        torch.save({k: p.cpu() for k, p in model.state_dict().items()},
                   os.path.join(save_dir, f"{cond}_final.pt"))
        del model, opt, ref_model
        torch.cuda.empty_cache()
        return {"trajectory": trajectory, "checkpoints": checkpoints}

    def run_ntp(start_state, cond):
        model = make_gpt()
        model.load_state_dict(start_state)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        trajectory = []
        checkpoints = {}
        for st in range(rl_steps):
            model.train()
            x, y = ntp_batch()
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (st + 1) % 250 == 0:
                model.eval()
                prefix, true_suffix = rl_batch()
                with torch.no_grad():
                    gen = _generate_suffix(model, prefix, suffix_len, temperature)
                r_ex = compute_reward(prefix, gen, true_suffix, "exact")
                r_pa = compute_reward(prefix, gen, true_suffix, "parse")
                vq = eval_val_loss(model, n_batches=8)
                trajectory.append({"step": st + 1,
                                   "reward_exact": float(r_ex.mean()),
                                   "reward_parse": float(r_pa.mean()),
                                   "val_loss_quick": vq})
                if (st + 1) % 1000 == 0:
                    print(f"    [{cond}] step {st+1:5d}: "
                          f"r_exact={r_ex.mean():.4f} r_parse={r_pa.mean():.4f} "
                          f"val~{vq:.4f}")
            if (st + 1) % ckpt_interval == 0:
                print(f"  [{cond}] CHECKPOINT step {st+1}")
                checkpoints[st + 1] = full_eval(model, f"{cond} s{st+1}")
        torch.save({k: p.cpu() for k, p in model.state_dict().items()},
                   os.path.join(save_dir, f"{cond}_final.pt"))
        del model, opt
        torch.cuda.empty_cache()
        return {"trajectory": trajectory, "checkpoints": checkpoints}

    def run_ei(start_state, cond, rw):
        """Expert iteration: best-of-N selection + SFT on winners.

        The noiseless, perfectly-credit-assigned limit of the sample-reweighting
        class — every policy-gradient method is a noisy approximation of
        "increase probability of your own high-reward samples". Rollout budget
        (ei_rounds * ei_prompts_per_round * ei_n) is matched to the RL
        conditions' rl_steps * batch_size.
        """
        model = make_gpt()
        model.load_state_dict(start_state)
        prompts_per_pass = max(1, 1024 // ei_n)
        trajectory = []
        checkpoints = {}
        for rnd in range(ei_rounds):
            # --- sample N per prompt, keep the reward-argmax ---
            model.eval()
            win_seqs, win_rewards = [], []
            sample_sum, sample_cnt = 0.0, 0
            remaining = ei_prompts_per_round
            while remaining > 0:
                k = min(prompts_per_pass, remaining)
                remaining -= k
                idx = torch.randint(n_train_seqs, (k,), generator=ei_gen)
                seqs = rl_t[idx].to(device)
                prefix = seqs[:, :prefix_len]
                true_suffix = seqs[:, prefix_len:]
                rep_prefix = prefix.repeat_interleave(ei_n, dim=0)
                rep_true = true_suffix.repeat_interleave(ei_n, dim=0)
                with torch.no_grad():
                    gen = _generate_suffix(model, rep_prefix, suffix_len,
                                           temperature)
                r = compute_reward(rep_prefix, gen, rep_true, rw).view(k, ei_n)
                sample_sum += float(r.sum())
                sample_cnt += k * ei_n
                best = r.argmax(dim=1)
                gen_k = gen.view(k, ei_n, suffix_len)
                winners = gen_k[torch.arange(k, device=device), best]
                win_seqs.append(torch.cat([prefix, winners], dim=1).cpu())
                win_rewards.append(r[torch.arange(k, device=device), best].cpu())
            win_data = torch.cat(win_seqs)     # (P, seq_len)
            win_r = torch.cat(win_rewards)
            win_pf = parse_valid_fractions_torch(
                win_data.to(device), rules_t).mean(dim=0).cpu()

            # --- SFT on winners, loss on suffix positions only ---
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=0.01)
            n_win = win_data.shape[0]
            sft_steps = 0
            for _ in range(ei_sft_epochs):
                perm = torch.randperm(n_win, generator=ei_gen)
                for i in range(0, n_win - batch_size + 1, batch_size):
                    bseq = win_data[perm[i:i + batch_size]].to(device)
                    x = bseq[:, :-1].contiguous()
                    y = bseq[:, 1:].contiguous()
                    logits, _ = model(x)
                    ce = F.cross_entropy(
                        logits.reshape(-1, v), y.reshape(-1),
                        reduction="none").view(-1, seq_len - 1)
                    loss = ce[:, prefix_len - 1:].mean()  # suffix targets only
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sft_steps += 1
            del opt

            sel = {"sample_reward_mean": sample_sum / max(1, sample_cnt),
                   "winner_reward_mean": float(win_r.mean()),
                   "winner_parse_per_level": win_pf.tolist(),
                   "n_winners": int(n_win), "sft_steps": sft_steps}
            trajectory.append({"round": rnd + 1, **sel})
            print(f"    [{cond}] round {rnd+1}: sample_r="
                  f"{sel['sample_reward_mean']:.4f} winner_r="
                  f"{sel['winner_reward_mean']:.4f} sft_steps={sft_steps}")
            print(f"  [{cond}] CHECKPOINT round {rnd+1}")
            ckpt = full_eval(model, f"{cond} r{rnd+1}")
            ckpt["selection"] = sel
            checkpoints[rnd + 1] = ckpt

        torch.save({k_: p.cpu() for k_, p in model.state_dict().items()},
                   os.path.join(save_dir, f"{cond}_final.pt"))
        del model
        torch.cuda.empty_cache()
        return {"trajectory": trajectory, "checkpoints": checkpoints}

    conditions = {}
    if only_ei:
        for rw in rewards:
            cond = f"ei_{rw}"
            print(f"\n{'='*60}\n  CONDITION: {cond} "
                  f"(N={ei_n}, {ei_rounds} rounds)\n{'='*60}")
            conditions[cond] = run_ei(pretrained_state, cond, rw)
    for rw in (rewards if not only_ei else []):
        if not only_kl:
            cond = f"scratch_rl_{rw}"
            print(f"\n{'='*60}\n  CONDITION: {cond}\n{'='*60}")
            conditions[cond] = run_rl(init_state, cond, rw)
            cond = f"pretrain_rl_{rw}"
            print(f"\n{'='*60}\n  CONDITION: {cond}\n{'='*60}")
            conditions[cond] = run_rl(pretrained_state, cond, rw)
        if kl_coef > 0:
            cond = f"pretrain_rl_kl_{rw}"
            print(f"\n{'='*60}\n  CONDITION: {cond} (coef={kl_coef})\n{'='*60}")
            conditions[cond] = run_rl(pretrained_state, cond, rw, use_kl=True)
    if not only_kl and not only_ei:
        print(f"\n{'='*60}\n  CONDITION: pretrain_only\n{'='*60}")
        conditions["pretrain_only"] = run_ntp(pretrained_state, "pretrain_only")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "prefix_len": prefix_len, "suffix_len": suffix_len,
            "rule_kind": "invertible", "rule_seed": rule_seed,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "reward_type": reward_type, "temperature": temperature,
            "kl_coef": kl_coef, "only_kl": only_kl,
            "pretrained_path": pretrained_path,
            "only_ei": only_ei, "ei_rounds": ei_rounds, "ei_n": ei_n,
            "ei_prompts_per_round": ei_prompts_per_round,
            "ei_sft_epochs": ei_sft_epochs,
            "lr": lr, "batch_size": batch_size,
            "seed": seed, "rl_steps": rl_steps,
            "ckpt_interval": ckpt_interval,
            "pretrain": {"min": pretrain_min_steps, "max": pretrain_max_steps,
                         "eval_every": pretrain_eval_every,
                         "patience": patience, "min_delta": min_delta},
            "n_train_seqs": n_train_seqs, "n_gen_eval": n_gen_eval,
            "smoke": smoke,
        },
        "references": {
            "ntp_bayes": ntp_floor,
            "exact_match": em_refs,
            "parse": parse_refs,
            "uniform_nats": float(np.log(v)),
        },
        "pretrain": {
            "steps_used": pretrain_steps_used,
            "best_val": best_val,
            "captured_excess_entropy": captured,
            "curve": pretrain_curve,
            "eval": pretrain_eval,
        },
        "conditions": conditions,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # Compact summary
    print(f"\n{'='*70}\nSUMMARY {key} reward={reward_type}\n{'='*70}")
    print(f"  Bayes: NTP floor={bayes_floor:.4f} | "
          f"EM ceiling(greedy)={em_refs['greedy_ceiling']['overall']:.4f} "
          f"blind floor={em_refs['blind_floor']['overall']:.4f}")
    print(f"  Pretrain: {pretrain_steps_used} steps, val={best_val:.4f}, "
          f"captured={captured:.3f}, "
          f"greedy exact={pretrain_eval['generation']['greedy']['exact_overall']:.4f} "
          f"(headroom {pretrain_eval['generation']['greedy']['exact_headroom']:.3f})")
    for cond, data in conditions.items():
        if not data["checkpoints"]:
            continue
        last = data["checkpoints"][max(data["checkpoints"])]
        g_ = last["generation"]["greedy"]
        print(f"  {cond:>14s}: val={last['val_loss']:.4f} "
              f"exact={g_['exact_overall']:.4f} "
              f"(headroom {g_['exact_headroom']:.3f}) "
              f"parse={g_['parse_overall']:.4f}")
    print(f"\n  Saved to {save_dir}/results.json")
    return {"save_dir": save_dir, "pretrain_val": best_val,
            "pretrain_steps": pretrain_steps_used}


# ======================================================================
# Self-tests (CPU): torch parse vs possible_set_parse, BP ceiling vs
# brute-force enumeration
# ======================================================================

@app.function(timeout=1200, memory=8192)
def self_test():
    import numpy as np
    import torch
    from rhm.rhm_data import (generate_rules_invertible,
                              generate_sequences_batched, possible_set_parse)
    from rhm.rhm_bayes_entropy import _brute_force_conditionals

    # --- 1. torch parse fractions vs possible_set_parse ---------------
    print("Test 1: parse_valid_fractions_torch vs possible_set_parse")
    for (v, s, L, m) in [(8, 2, 6, 2), (8, 2, 6, 4), (8, 2, 6, 6), (8, 2, 6, 1)]:
        rules = generate_rules_invertible(v, s, L, m, seed=0)
        rules_t = [torch.from_numpy(r).long() for r in rules]
        rng = np.random.default_rng(0)
        on = generate_sequences_batched(rules, 200, seed=1)
        rand = rng.integers(0, v, size=(200, s ** L))
        corrupt = on.copy()
        corrupt[:, -7:] = rng.integers(0, v, size=(200, 7))
        rand_frac = None
        for name, seqs in [("on", on), ("rand", rand), ("corrupt", corrupt)]:
            ref = possible_set_parse(seqs, rules)
            fr = parse_valid_fractions_torch(
                torch.from_numpy(seqs.astype(np.int64)), rules_t).numpy()
            valid_t = np.isclose(fr, 1.0).all(axis=1)
            assert (valid_t == ref["valid"]).all(), f"{name}: valid mismatch"
            # parse_level = consecutive full-valid levels from the bottom
            pl_t = np.zeros(len(seqs), dtype=np.int64)
            for k in range(L):
                pl_t += ((pl_t == k) & np.isclose(fr[:, k], 1.0))
            assert (pl_t == ref["parse_level"]).all(), f"{name}: parse_level mismatch"
            if name == "on":
                assert np.isclose(fr, 1.0).all(), "on-grammar not fully valid"
            if name == "rand":
                rand_frac = fr.mean()
        print(f"  v{v}/s{s}/L{L}/m{m}: OK "
              f"(rand mean frac={rand_frac:.3f})")

    # --- 2. BP suffix marginals vs brute force -------------------------
    print("Test 2: suffix_marginals vs brute-force enumeration")
    for (v, s, L, m) in [(3, 2, 2, 2), (2, 2, 3, 2), (4, 2, 2, 3)]:
        rules = generate_rules_invertible(v, s, L, m, seed=1)
        H_bf, arr, probs = _brute_force_conditionals(rules)
        seq_len = s ** L
        p_len = seq_len // 2
        test_seqs = arr[np.random.default_rng(0).choice(
            len(arr), size=min(20, len(arr)), replace=False)]
        post = suffix_marginals(rules, test_seqs, p_len)
        max_err = 0.0
        for bi, seq in enumerate(test_seqs):
            mask = (arr[:, :p_len] == seq[:p_len]).all(axis=1)
            pm = probs[mask]
            sub = arr[mask]
            for j in range(p_len, seq_len):
                marg = np.zeros(v)
                for a in range(v):
                    marg[a] = pm[sub[:, j] == a].sum()
                marg /= marg.sum()
                max_err = max(max_err, np.abs(
                    marg - post[bi, j - p_len]).max())
        assert max_err < 1e-8, f"BP marginals wrong ({max_err})"
        print(f"  v{v}/s{s}/L{L}/m{m}: max err {max_err:.2e} OK")

    # --- 3. reference ordering sanity on the real setting --------------
    print("Test 3: reference ordering (v=8, s=2, L=6)")
    for m in [1, 2, 4, 6]:
        rules = generate_rules_invertible(8, 2, 6, m, seed=0)
        seqs = generate_sequences_batched(rules, 100, seed=2)
        refs, _ = exact_match_references(rules, seqs, 32)
        gc = refs["greedy_ceiling"]["overall"]
        sc = refs["sampled_ceiling"]["overall"]
        bf = refs["blind_floor"]["overall"]
        # guaranteed orderings only: greedy >= sampled, greedy >= blind,
        # blind >= uniform (sampled vs blind is NOT ordered in general)
        assert gc >= sc - 1e-9 and gc >= bf - 1e-9 and bf >= 1 / 8 - 1e-9, \
            f"ordering violated: {gc} {sc} {bf}"
        print(f"  m={m}: greedy={gc:.4f} sampled={sc:.4f} blind={bf:.4f} "
              f"uniform=0.125")
        # NB: even at m=1 the greedy ceiling is < 1.0 — distinct root rules can
        # share the same LEFT child, so the prefix doesn't always determine the
        # root. The BP ceiling quantifies this exactly; do not assume 1.0.

    print("\nALL SELF-TESTS PASSED")
    return True
