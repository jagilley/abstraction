"""Why do the two seeds' drift directions get more similar as m grows?

Follow-on to `idiolect_drift.py`. That sweep found that two expert-iteration
runs from the identical pretrained checkpoint drift comparably far but in
different directions — and that the angle between the two drifts falls from
~75-80 deg at m=2,3 to 45 deg at m=6. This module tests one candidate cause,
inference-only on the existing checkpoints:

  The parse verifier is invariant to synonym choice on *grammatical* output,
  but the model's attempts are not all grammatical. If the (shared) pretrained
  checkpoint executes some synonyms of a feature more reliably than others,
  best-of-N selection on parse validity favours those synonyms, and SFT on the
  winners moves both seeds the same way. That shared push is a property of the
  starting checkpoint (and possibly of the grammar), not a convention the two
  runs discover.

The measurement replays one EI selection step at a checkpoint without
training: sample k suffixes per prompt, pick the parse-reward argmax exactly
as `run_ei` does, and compare the winners' synonym-choice histogram with the
samples'. That difference is the one-step selection differential. Stored per
checkpoint (pretrained, and every EI-parse round of both seeds) so the
aggregator can ask (i) whether the final shared drift aligns with the
selection differential at the pretrained checkpoint and (ii) whether each
seed's round-to-round change aligns with its own selection differential.

Also stored per (level, feature, rule): the mean parse reward of sequences
using that synonym, and the probability that the node's parent parses given
the node does — the "competence" reading of the same thing.

Reproduction (from experiments/):
  modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::selection_test
  modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::competence_test
  modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::measure_controls
  modal volume get rhm-scaling-data rl_dimensionality/idiolect/{selection_differential,competence,controls}.json \
      rhm/rl_dimensionality/idiolect/direction/
  cd rhm/rl_dimensionality/idiolect/direction && python3 aggregate_direction.py
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rl_dimensionality.idiolect.idiolect_drift import (
    image, build_rule_inverse, recover_rule_usage, free_node_slice, _run_dir)

app = modal.App("rhm-idiolect-direction", image=image)

RUN_ROOT = f"{DATA_DIR}/rl_dimensionality"

M_VALUES = [2, 3, 4, 6]


def _checkpoints(m):
    """(label, path) for pretrained + every EI-parse round of both seeds."""
    out = [("pretrained", os.path.join(_run_dir(m, ""), "pretrained.pt"))]
    for r in range(1, 7):
        out.append((f"s42_r{r}", os.path.join(_run_dir(m, "ei02", 42), f"ei_parse_r{r}.pt")))
    for r in range(1, 7):
        out.append((f"s43_r{r}", os.path.join(_run_dir(m, "ei_s43", 43), f"ei_parse_r{r}.pt")))
    return out


def selection_stats(full, reward, k, rules, prefix_len, tables):
    """One replayed EI selection step.

    full: (P*k, seq_len) int array of prompt-grouped generations, reward:
    (P*k,) parse reward. Returns per-level dicts with the sample and winner
    joint histograms plus per-(feature, rule) reward and upward-validity.
    """
    import numpy as np
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    n_prompts = full.shape[0] // k
    usage = recover_rule_usage(full, rules, tables)
    r2 = reward.reshape(n_prompts, k)
    best = r2.argmax(axis=1)                       # first among ties, as run_ei
    win_idx = np.arange(n_prompts) * k + best
    is_win = np.zeros(full.shape[0], dtype=bool)
    is_win[win_idx] = True

    per_level = {}
    for ell in range(L):
        first_free, n_nodes = free_node_slice(ell, prefix_len, seq_len, s)
        if first_free >= n_nodes:
            continue
        feat = usage[ell]["feat"][:, first_free:]
        rule = usage[ell]["rule"][:, first_free:]
        ok = usage[ell]["valid"][:, first_free:]
        # parent validity of each free node (parent index j//s at level ell-1)
        j = np.arange(first_free, n_nodes)
        par_ok = usage[ell - 1]["valid"][:, j // s] if ell > 0 else np.ones_like(ok)
        rew = np.broadcast_to(reward[:, None], ok.shape)
        win = np.broadcast_to(is_win[:, None], ok.shape)

        joint_s = np.zeros((v, m)); joint_w = np.zeros((v, m))
        rew_sum = np.zeros((v, m)); up_sum = np.zeros((v, m))
        f, r = feat[ok], rule[ok]
        np.add.at(joint_s, (f, r), 1.0)
        np.add.at(joint_w, (f[win[ok]], r[win[ok]]), 1.0)
        np.add.at(rew_sum, (f, r), rew[ok])
        np.add.at(up_sum, (f, r), par_ok[ok].astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            per_level[f"L{ell}"] = {
                "n_nodes": int(ok.size), "n_valid": int(ok.sum()),
                "n_valid_winners": int((ok & win).sum()),
                "valid_frac": float(ok.mean()),
                "sample_hist": (joint_s / max(joint_s.sum(), 1e-30)).tolist(),
                "winner_hist": (joint_w / max(joint_w.sum(), 1e-30)).tolist(),
                "count": joint_s.tolist(),
                "reward_by_rule": np.where(joint_s > 0, rew_sum / joint_s, np.nan).tolist(),
                "up_valid_by_rule": np.where(joint_s > 0, up_sum / joint_s, np.nan).tolist(),
            }
    return per_level


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4",
              timeout=7200, memory=32768)
def selection_test(n_prompts: int = 1500, k: int = 16, eval_seed: int = 987654,
                   batch: int = 4000, out_tag: str = "", only_m: int = 0):
    """Replay one best-of-k parse selection step at every checkpoint (no training)."""
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_invertible, generate_sequences_batched
    from rhm.rl_dimensionality.rl_dim_ablation import (
        _generate_suffix, parse_valid_fractions_torch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, rule_seed = 8, 2, 6, 0
    seq_len, prefix_len = s ** L, (s ** L) // 2
    suffix_len = seq_len - prefix_len
    results = {"config": {"n_prompts": n_prompts, "k": k, "eval_seed": eval_seed,
                          "v": v, "s": s, "L": L, "rule_seed": rule_seed,
                          "prefix_len": prefix_len, "temperature": 1.0},
               "by_m": {}}
    m_values = [only_m] if only_m else M_VALUES
    for m in m_values:
        print(f"\n{'='*66}\n  m = {m}\n{'='*66}")
        rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
        tables = build_rule_inverse(rules)
        rules_t = [torch.from_numpy(r).to(device) for r in rules]
        # same held-out prefixes as idiolect_drift.analyze (first n_prompts of them)
        true_seqs = generate_sequences_batched(rules, n_prompts, seed=eval_seed)
        prefixes = torch.from_numpy(np.asarray(true_seqs[:, :prefix_len])).long().to(device)
        rep = prefixes.repeat_interleave(k, dim=0)          # prompt-grouped
        gen = torch.Generator().manual_seed(eval_seed + 7)

        m_out = {"log2_m": float(np.log2(m)), "checkpoints": {}}
        for label, path in _checkpoints(m):
            if not os.path.exists(path):
                print(f"  [skip] {label} (missing {path})")
                continue
            model = GPT(v, seq_len, 6, 6, 192).to(device)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            fulls, rews = [], []
            with torch.no_grad():
                for lo in range(0, rep.shape[0], batch):
                    pref = rep[lo:lo + batch]
                    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=gen)))
                    suf = _generate_suffix(model, pref, suffix_len, temperature=1.0)
                    full = torch.cat([pref, suf], dim=1)
                    fulls.append(full.cpu())
                    rews.append(parse_valid_fractions_torch(full, rules_t).mean(dim=1).cpu())
            full = torch.cat(fulls).numpy()
            reward = torch.cat(rews).numpy()
            st = selection_stats(full, reward, k, rules, prefix_len, tables)
            r2 = reward.reshape(n_prompts, k)
            st["sample_reward_mean"] = float(reward.mean())
            st["winner_reward_mean"] = float(r2.max(axis=1).mean())
            m_out["checkpoints"][label] = st
            print(f"  {label:12s} sample_r={st['sample_reward_mean']:.4f} "
                  f"winner_r={st['winner_reward_mean']:.4f} "
                  f"L5 valid={st['L5']['valid_frac']:.3f}")
            del model
            torch.cuda.empty_cache()
        results["by_m"][str(m)] = m_out

    out_dir = f"{DATA_DIR}/rl_dimensionality/idiolect"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{out_tag}" if out_tag else ""
    out_path = os.path.join(out_dir, f"selection_differential{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"\nWrote {out_path}")
    return {"path": out_path}


def loss_by_rule(model, x_all, usage, rules, prefix_len, batch, device):
    """Teacher-forced per-token CE on ground-truth sequences, averaged per
    (level, feature, rule) over free nodes: whole subtree, right child only,
    and the first leaf of the right child. Returns (per_level dict, mean CE)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    ce = []
    with torch.no_grad():
        for lo in range(0, x_all.shape[0], batch):
            xb = x_all[lo:lo + batch].to(device)
            logits, _ = model(xb[:, :-1].contiguous())
            c = F.cross_entropy(logits.reshape(-1, v), xb[:, 1:].reshape(-1),
                                reduction="none").view(xb.shape[0], seq_len - 1)
            ce.append(c.cpu().numpy())
    ce = np.concatenate(ce)            # (n, seq_len-1); column t predicts leaf t+1
    per_level = {}
    for ell in range(L):
        first_free, n_nodes = free_node_slice(ell, prefix_len, seq_len, s)
        if first_free >= n_nodes:
            continue
        span = seq_len // n_nodes
        acc = {k: np.zeros((v, m)) for k in ("subtree", "right", "right_first")}
        cnt = np.zeros((v, m))
        for j in range(first_free, n_nodes):
            f = usage[ell]["feat"][:, j]; r = usage[ell]["rule"][:, j]
            lo_leaf = j * span
            np.add.at(cnt, (f, r), 1.0)
            np.add.at(acc["subtree"], (f, r), ce[:, lo_leaf - 1: lo_leaf + span - 1].mean(axis=1))
            np.add.at(acc["right"], (f, r), ce[:, lo_leaf + span // 2 - 1: lo_leaf + span - 1].mean(axis=1))
            np.add.at(acc["right_first"], (f, r), ce[:, lo_leaf + span // 2 - 1])
        per_level[f"L{ell}"] = {
            "count": cnt.tolist(),
            **{k: (a / np.clip(cnt, 1, None)).tolist() for k, a in acc.items()}}
    return per_level, float(ce.mean())


# ======================================================================
# Competence by synonym: teacher-forced loss per (level, feature, rule)
# ======================================================================

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4",
              timeout=3600, memory=32768)
def competence_test(n_seqs: int = 20000, eval_seed: int = 24681357,
                    batch: int = 1000, out_tag: str = ""):
    """How well does each checkpoint *know* each synonym?

    Teacher-forced on fresh DGP sequences, so every internal node's (feature,
    rule) is exact ground truth. For every free node (whole subtree in the
    suffix) we record the model's mean per-token cross-entropy over the node's
    leaves, over its right child's leaves only, and on the single first leaf of
    its right child (the position where the model has to commit to the
    rule's second element, given the first). Averaged per (level, feature,
    rule). Low loss = the synonym is well-learned.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_invertible, generate_sequences_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, rule_seed = 8, 2, 6, 0
    seq_len, prefix_len = s ** L, (s ** L) // 2
    results = {"config": {"n_seqs": n_seqs, "eval_seed": eval_seed, "v": v, "s": s,
                          "L": L, "rule_seed": rule_seed, "prefix_len": prefix_len},
               "by_m": {}}
    for m in M_VALUES:
        rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
        tables = build_rule_inverse(rules)
        seqs = np.asarray(generate_sequences_batched(rules, n_seqs, seed=eval_seed))
        usage = recover_rule_usage(seqs, rules, tables)
        assert all(u["valid"].all() for u in usage)
        x_all = torch.from_numpy(seqs).long()
        m_out = {"checkpoints": {}}
        for label, path in _checkpoints(m):
            if label.startswith("s4") and not label.endswith("r6"):
                continue                       # pretrained + the two finals
            if not os.path.exists(path):
                print(f"  [skip] {label}"); continue
            model = GPT(v, seq_len, 6, 6, 192).to(device)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            per_level, mean_ce = loss_by_rule(model, x_all, usage, rules, prefix_len, batch, device)
            m_out["checkpoints"][label] = per_level
            print(f"  m={m} {label:12s} mean CE={mean_ce:.4f}")
            del model
            torch.cuda.empty_cache()
        results["by_m"][str(m)] = m_out

    out_dir = f"{DATA_DIR}/rl_dimensionality/idiolect"
    tag = f"_{out_tag}" if out_tag else ""
    out_path = os.path.join(out_dir, f"competence{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"\nWrote {out_path}")
    return {"path": out_path}


# ======================================================================
# Controls: selection-free EI (random reward) and EI from a fresh pretraining seed
# ======================================================================

def _control_checkpoints(m):
    base = _run_dir(m, "")
    out = [("pt42", os.path.join(base, "pretrained.pt")),
           ("s42_r6", os.path.join(_run_dir(m, "ei02", 42), "ei_parse_r6.pt")),
           ("s43_r6", os.path.join(_run_dir(m, "ei_s43", 43), "ei_parse_r6.pt"))]
    rand_dir = f"{RUN_ROOT}/v8_s2_L6_m{m}_random_seed42_ei_rand"
    out += [(f"rand_r{r}", os.path.join(rand_dir, f"ei_random_r{r}.pt")) for r in range(1, 7)]
    pt44_dir = f"{RUN_ROOT}/v8_s2_L6_m{m}_parse_seed44_pt44"
    out.append(("pt44", os.path.join(pt44_dir, "pretrained.pt")))
    out += [(f"pt44_r{r}", os.path.join(pt44_dir, f"ei_parse_r{r}.pt")) for r in range(1, 7)]
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4",
              timeout=7200, memory=32768)
def measure_controls(n_prompts: int = 1500, k: int = 16, eval_seed: int = 987654,
                     n_loss_seqs: int = 20000, loss_seed: int = 24681357,
                     batch: int = 4000, out_tag: str = ""):
    """Synonym-choice histograms (same protocol as selection_test) and
    teacher-forced loss-by-rule for the control arms, at m in {2, 6}."""
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_invertible, generate_sequences_batched
    from rhm.rl_dimensionality.rl_dim_ablation import (
        _generate_suffix, parse_valid_fractions_torch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, rule_seed = 8, 2, 6, 0
    seq_len, prefix_len = s ** L, (s ** L) // 2
    suffix_len = seq_len - prefix_len
    results = {"config": {"n_prompts": n_prompts, "k": k, "eval_seed": eval_seed,
                          "n_loss_seqs": n_loss_seqs, "loss_seed": loss_seed,
                          "prefix_len": prefix_len, "temperature": 1.0},
               "by_m": {}}
    for m in (2, 6):
        rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
        tables = build_rule_inverse(rules)
        rules_t = [torch.from_numpy(r).to(device) for r in rules]
        true_seqs = generate_sequences_batched(rules, n_prompts, seed=eval_seed)
        prefixes = torch.from_numpy(np.asarray(true_seqs[:, :prefix_len])).long().to(device)
        rep = prefixes.repeat_interleave(k, dim=0)
        gen = torch.Generator().manual_seed(eval_seed + 7)
        loss_seqs = np.asarray(generate_sequences_batched(rules, n_loss_seqs, seed=loss_seed))
        loss_usage = recover_rule_usage(loss_seqs, rules, tables)
        x_loss = torch.from_numpy(loss_seqs).long()

        m_out = {"checkpoints": {}}
        for label, path in _control_checkpoints(m):
            if not os.path.exists(path):
                print(f"  [skip] {label} (missing {path})"); continue
            model = GPT(v, seq_len, 6, 6, 192).to(device)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            fulls, rews = [], []
            with torch.no_grad():
                for lo in range(0, rep.shape[0], batch):
                    pref = rep[lo:lo + batch]
                    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=gen)))
                    suf = _generate_suffix(model, pref, suffix_len, temperature=1.0)
                    full = torch.cat([pref, suf], dim=1)
                    fulls.append(full.cpu())
                    rews.append(parse_valid_fractions_torch(full, rules_t).mean(dim=1).cpu())
            full = torch.cat(fulls).numpy(); reward = torch.cat(rews).numpy()
            st = selection_stats(full, reward, k, rules, prefix_len, tables)
            st["sample_reward_mean"] = float(reward.mean())
            st["loss"], st["mean_ce"] = loss_by_rule(model, x_loss, loss_usage, rules,
                                                     prefix_len, 1000, device)
            m_out["checkpoints"][label] = st
            print(f"  m={m} {label:10s} parse_r={st['sample_reward_mean']:.4f} "
                  f"L1 valid={st['L1']['valid_frac']:.3f} CE={st['mean_ce']:.4f}")
            del model
            torch.cuda.empty_cache()
        results["by_m"][str(m)] = m_out

    out_dir = f"{DATA_DIR}/rl_dimensionality/idiolect"
    tag = f"_{out_tag}" if out_tag else ""
    out_path = os.path.join(out_dir, f"controls{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"\nWrote {out_path}")
    return {"path": out_path}
