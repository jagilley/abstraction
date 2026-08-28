"""Idiolect drift: does reward optimization develop private synonym preferences?

Follow-on analysis to `rl_dim_ablation.py`, run entirely on ITS saved
checkpoints (no retraining). The question is not whether RL creates structure
(it does not — see ../rl_dimensionality/README.md) but what it does to the one
coordinate the parse verifier is *exactly invariant to*: which of the m
synonymous rules realizes a given latent feature.

Motivation. The parse reward accepts any grammatical continuation, so rule
choice is a free coordinate under it; the exact reward scores against one
sampled ground-truth suffix, so rule choice is pinned. Pretraining calibrates
rule choice to the DGP (uniform over m). If reward optimization drifts along
the free coordinate and then sharpens, the model ends up with a confident
private dialect: grammatical, but no longer speaking the corpus's language.

Because `generate_rules_invertible` makes every legal s-tuple map to exactly
one (feature, rule) pair, the rule used at every internal node of a generated
sequence is EXACTLY recoverable — this is ground truth, not an estimate.

Metrics, per level, over nodes whose whole subtree lies inside the
model-generated suffix (so the choice was the model's, not the prefix's):
  valid_frac   fraction of those nodes that parse at all (grammaticality)
  H_marg       entropy of the marginal rule distribution, bits (DGP: log2 m)
  H_cond       H(rule | feature), bits — the synonym-choice entropy
  kl_cond      log2(m) - H_cond, i.e. KL(model || DGP) on synonym choice
Every metric is also computed on true DGP samples at matched node counts, to
give the finite-sample floor (entropy estimates are downward biased, so kl is
upward biased; the DGP row is the noise floor that must be subtracted by eye).

Reproduction (from experiments/):
  # self-tests (CPU, ~1 min) — recovery vs traces, validity, metric calibration
  modal run -m rhm.rl_dimensionality.idiolect.idiolect_drift::self_test
  # the sweep (GPU, ~70 min) — every checkpoint, inference only
  modal run --detach -m rhm.rl_dimensionality.idiolect.idiolect_drift::analyze \
      --n-eval 4000 --out-tag seed43
  # self-tests without Modal (needs numpy only)
  python3 -m rhm.rl_dimensionality.idiolect.idiolect_drift

The seed-43 privacy control is trained by rl_dim_ablation itself:
  for M in 1 2 3 4 6; do
    modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M \
      --only-ei --run-tag ei_s43 --seed 43 --save-round-ckpts \
      --pretrained-path rl_dimensionality/v8_s2_L6_m${M}_both_seed42/pretrained.pt
  done
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

app = modal.App("rhm-idiolect", image=image)


# ======================================================================
# Exact rule recovery (numpy, CPU) — the core primitive
# ======================================================================

def build_rule_inverse(rules):
    """Per-level tuple-code -> (feature, rule) tables. Requires invertible rules.

    rules[ell] has shape (v, m, s) and expands a level-`ell` feature into its
    level-`ell+1` children (rules[0] = root -> its children, rules[L-1] ->
    leaves), matching rhm_data conventions. Returns a list of L (inv_feat,
    inv_rule) int arrays of length v**s, holding -1 for off-grammar codes.
    """
    import numpy as np
    v, m, s = rules[0].shape
    powers = v ** np.arange(s)
    tables = []
    for layer in rules:
        inv_feat = np.full(v ** s, -1, dtype=np.int64)
        inv_rule = np.full(v ** s, -1, dtype=np.int64)
        codes = (layer * powers).sum(axis=2)           # (v, m)
        if len(np.unique(codes)) != codes.size:
            raise ValueError("rules are not collision-free; recovery is ambiguous")
        feats = np.repeat(np.arange(v), m)
        rls = np.tile(np.arange(m), v)
        inv_feat[codes.reshape(-1)] = feats
        inv_rule[codes.reshape(-1)] = rls
        tables.append((inv_feat, inv_rule))
    return tables


def recover_rule_usage(seqs, rules, tables=None):
    """Bottom-up exact recovery of (feature, rule) at every internal node.

    seqs: (B, s^L) int array of leaf tokens. Returns a list of L dicts (index
    ell = the level whose features/rules are recovered by the fold that
    consumes level ell+1), each with arrays of shape (B, s^ell):
      feat  recovered feature, -1 where the node does not parse
      rule  recovered rule index, -1 where the node does not parse
      valid bool, True iff this node's entire subtree is on-grammar
    Invalidity propagates upward exactly as in rhm_data.possible_set_parse.
    """
    import numpy as np
    v, m, s = rules[0].shape
    L = len(rules)
    if tables is None:
        tables = build_rule_inverse(rules)
    powers = v ** np.arange(s)

    seqs = np.asarray(seqs, dtype=np.int64)
    B = seqs.shape[0]
    cur = seqs
    valid = np.ones_like(cur, dtype=bool)
    out = [None] * L
    for ell in range(L - 1, -1, -1):
        n2 = cur.shape[1] // s
        child = cur.reshape(B, n2, s)
        child_valid = valid.reshape(B, n2, s).all(axis=2)
        code = (child * powers).sum(axis=2)            # (B, n2)
        inv_feat, inv_rule = tables[ell]
        feat = inv_feat[code]
        rule = inv_rule[code]
        node_valid = child_valid & (feat >= 0)
        out[ell] = {
            "feat": np.where(node_valid, feat, -1),
            "rule": np.where(node_valid, rule, -1),
            "valid": node_valid,
        }
        # invalid nodes carry an arbitrary in-range feature; `valid` masks them
        cur = np.where(node_valid, feat, 0)
        valid = node_valid
    return out


# ======================================================================
# Idiolect statistics
# ======================================================================

def _entropy_bits(counts, axis=-1):
    import numpy as np
    tot = counts.sum(axis=axis, keepdims=True)
    p = counts / np.clip(tot, 1e-30, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=axis)


def free_node_slice(ell, prefix_len, seq_len, s):
    """Indices of level-`ell` nodes whose whole subtree lies in the suffix.

    Node j at level ell covers leaves [j*span, (j+1)*span) with
    span = s^(L-ell); it is free iff j*span >= prefix_len.
    """
    n_nodes = s ** ell
    span = seq_len // n_nodes
    first_free = -(-prefix_len // span)     # ceil division
    return first_free, n_nodes


def idiolect_stats(seqs, rules, prefix_len, tables=None):
    """Per-level synonym-choice statistics over model-chosen nodes.

    Returns {"L{ell}": {...}} for every level with at least one free node,
    plus "overall" = node-weighted means of the per-level values.
    """
    import numpy as np
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    usage = recover_rule_usage(seqs, rules, tables)

    per_level = {}
    for ell in range(L):
        first_free, n_nodes = free_node_slice(ell, prefix_len, seq_len, s)
        if first_free >= n_nodes:
            continue
        feat = usage[ell]["feat"][:, first_free:]
        rule = usage[ell]["rule"][:, first_free:]
        ok = usage[ell]["valid"][:, first_free:]
        n_total = int(ok.size)
        n_valid = int(ok.sum())
        entry = {
            "n_nodes": n_total,
            "n_valid": n_valid,
            "valid_frac": n_valid / max(n_total, 1),
        }
        if n_valid == 0 or m == 1:
            entry.update({"H_marg": 0.0, "H_cond": 0.0,
                          "kl_marg": 0.0, "kl_cond": 0.0,
                          "rule_hist": [0.0] * m,
                          "joint_hist": [[0.0] * m for _ in range(v)]})
            per_level[f"L{ell}"] = entry
            continue
        f = feat[ok]
        r = rule[ok]
        joint = np.zeros((v, m), dtype=np.float64)
        np.add.at(joint, (f, r), 1.0)
        marg = joint.sum(axis=0)
        H_marg = float(_entropy_bits(marg))
        w = joint.sum(axis=1)
        H_cond = float((w * _entropy_bits(joint, axis=1)).sum() / max(w.sum(), 1e-30))
        entry.update({
            "H_marg": H_marg,
            "H_cond": H_cond,
            "kl_marg": float(np.log2(m) - H_marg),
            "kl_cond": float(np.log2(m) - H_cond),
            "rule_hist": (marg / marg.sum()).tolist(),
            "joint_hist": (joint / max(joint.sum(), 1e-30)).tolist(),
        })
        per_level[f"L{ell}"] = entry

    keys = ["valid_frac", "H_marg", "H_cond", "kl_marg", "kl_cond"]
    tot = sum(e["n_nodes"] for e in per_level.values()) or 1
    overall = {k: sum(e[k] * e["n_nodes"] for e in per_level.values()) / tot
               for k in keys}
    # validity-weighted for the entropy terms (valid_frac already node-weighted)
    totv = sum(e["n_valid"] for e in per_level.values()) or 1
    for k in ["H_marg", "H_cond", "kl_marg", "kl_cond"]:
        overall[k] = sum(e[k] * e["n_valid"] for e in per_level.values()) / totv
    per_level["overall"] = overall
    return per_level


def js_divergence_bits(p, q):
    """Jensen-Shannon divergence in bits between two (v, m) joint histograms.

    Compares the two models' *conditional* rule distributions, weighted by the
    average feature frequency, so it measures disagreement about synonym choice
    rather than disagreement about which features get produced.
    """
    import numpy as np
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    wp, wq = p.sum(axis=1), q.sum(axis=1)
    w = 0.5 * (wp + wq)
    cp = p / np.clip(wp[:, None], 1e-30, None)
    cq = q / np.clip(wq[:, None], 1e-30, None)
    mid = 0.5 * (cp + cq)

    def _kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(np.clip(a, 1e-30, None)
                                            / np.clip(b, 1e-30, None)), 0.0)
        return t.sum(axis=1)

    return float((w * (0.5 * _kl(cp, mid) + 0.5 * _kl(cq, mid))).sum())


# ======================================================================
# The sweep: idiolect drift across m x mechanism x verifier
# ======================================================================

RUN_ROOT = f"{DATA_DIR}/rl_dimensionality"
M_VALUES = [1, 2, 3, 4, 6]

# (label, run_tag, checkpoint filename, training seed) — run_tag "" is the
# base sweep dir. Seed 43 is the privacy control: an independent EI run from
# the IDENTICAL pretrained checkpoint, so any disagreement with seed 42 is
# arbitrary drift direction rather than a shared better convention.
CONDITIONS = [
    ("pretrained",        "",       "pretrained.pt",              42),
    ("pretrain_only",     "",       "pretrain_only_final.pt",     42),
    ("scratch_rl_exact",  "",       "scratch_rl_exact_final.pt",  42),
    ("scratch_rl_parse",  "",       "scratch_rl_parse_final.pt",  42),
    ("reinforce_exact",   "",       "pretrain_rl_exact_final.pt", 42),
    ("reinforce_parse",   "",       "pretrain_rl_parse_final.pt", 42),
    ("kl_exact",          "kl01",   "pretrain_rl_kl_exact_final.pt", 42),
    ("kl_parse",          "kl01",   "pretrain_rl_kl_parse_final.pt", 42),
    ("ei_exact",          "ei01",   "ei_exact_final.pt",          42),
    ("ei_parse",          "ei01",   "ei_parse_final.pt",          42),
    # same-seed EI re-run: the run-to-run noise floor for the drift metric
    ("ei_exact_rerun",    "ei02",   "ei_exact_final.pt",          42),
    ("ei_parse_rerun",    "ei02",   "ei_parse_final.pt",          42),
    # independent seed from the same basis: the privacy control
    ("ei_exact_seed43",   "ei_s43", "ei_exact_final.pt",          43),
    ("ei_parse_seed43",   "ei_s43", "ei_parse_final.pt",          43),
]

# per-round trajectories (ei02 and the seed-43 arm saved round checkpoints)
ROUND_CONDITIONS = []
for _arm in ("parse", "exact"):
    ROUND_CONDITIONS += [(f"ei_{_arm}_r{r}", "ei02", f"ei_{_arm}_r{r}.pt", 42)
                         for r in range(1, 7)]
    ROUND_CONDITIONS += [(f"ei_{_arm}_seed43_r{r}", "ei_s43", f"ei_{_arm}_r{r}.pt", 43)
                         for r in range(1, 7)]


def _run_dir(m, tag, seed=42, v=8, s=2, L=6):
    suffix = f"_{tag}" if tag else ""
    return f"{RUN_ROOT}/v{v}_s{s}_L{L}_m{m}_both_seed{seed}{suffix}"


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4",
              timeout=10800, memory=32768)
def analyze(n_eval: int = 4000, eval_seed: int = 987654, batch: int = 500,
            include_rounds: bool = True, out_tag: str = "",
            only_m: int = 0):
    """Load every rl_dim_ablation checkpoint and measure synonym-choice drift.

    Inference only — no training, no gradient steps. For each checkpoint we
    sample `n_eval` suffixes from held-out prefixes (and also take the greedy
    continuation), recover the exact rule used at every model-chosen node, and
    report the synonym-choice statistics against the DGP's uniform-over-m.
    """
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
    results = {"config": {"n_eval": n_eval, "eval_seed": eval_seed, "v": v,
                          "s": s, "L": L, "rule_seed": rule_seed,
                          "prefix_len": prefix_len, "temperature": 1.0},
               "by_m": {}}

    conds = CONDITIONS + (ROUND_CONDITIONS if include_rounds else [])
    m_values = [only_m] if only_m else M_VALUES

    for m in m_values:
        print(f"\n{'='*66}\n  m = {m}   (max possible drift = log2(m) = "
              f"{np.log2(m):.4f} bits)\n{'='*66}")
        rules = generate_rules_invertible(v, s, L, m, seed=rule_seed)
        # cross-check against the rules the training run actually saved
        base = _run_dir(m, "")
        for ell in range(L):
            fp = os.path.join(base, f"rules_L{ell}.npy")
            if os.path.exists(fp):
                assert np.array_equal(np.load(fp), rules[ell]), \
                    f"rule mismatch at m={m} level={ell}"
        tables = build_rule_inverse(rules)
        rules_t = [torch.from_numpy(r).to(device) for r in rules]

        true_seqs = generate_sequences_batched(rules, n_eval, seed=eval_seed)
        prefixes = torch.from_numpy(np.asarray(true_seqs[:, :prefix_len])).long().to(device)

        m_out = {"log2_m": float(np.log2(m)), "conditions": {}}
        # the DGP row: the finite-sample floor for every metric below
        dgp = idiolect_stats(np.asarray(true_seqs), rules, prefix_len, tables)
        m_out["conditions"]["dgp"] = {"sampled": dgp}
        print(f"  {'dgp (floor)':20s} sampled: kl_cond={dgp['overall']['kl_cond']:.4f} bits"
              f"  H_cond={dgp['overall']['H_cond']:.4f}  valid=1.0000")

        for label, tag, fname, cseed in conds:
            path = os.path.join(_run_dir(m, tag, cseed), fname)
            if not os.path.exists(path):
                print(f"  [skip] {label:20s} (missing {path})")
                continue
            state = torch.load(path, map_location="cpu")
            model = GPT(v, seq_len, 6, 6, 192).to(device)
            model.load_state_dict(state)
            model.eval()

            entry = {}
            with torch.no_grad():
                for mode, greedy in (("sampled", False), ("greedy", True)):
                    gens = []
                    for lo in range(0, n_eval, batch):
                        pref = prefixes[lo:lo + batch]
                        suf = _generate_suffix(model, pref, suffix_len,
                                               temperature=1.0, greedy=greedy)
                        gens.append(torch.cat([pref, suf], dim=1))
                    full = torch.cat(gens, dim=0)
                    pv = parse_valid_fractions_torch(full, rules_t).mean(dim=0)
                    st = idiolect_stats(full.cpu().numpy(), rules, prefix_len, tables)
                    st["parse_valid_by_level"] = pv.tolist()
                    st["root_valid"] = float(pv[-1].item())
                    entry[mode] = st
            m_out["conditions"][label] = entry
            o = entry["sampled"]["overall"]
            print(f"  {label:20s} sampled: kl_cond={o['kl_cond']:.4f} bits  "
                  f"H_cond={o['H_cond']:.4f}  valid={o['valid_frac']:.4f}  "
                  f"root={entry['sampled']['root_valid']:.4f}")
            del model, state
            torch.cuda.empty_cache()

        # pairwise disagreement about synonym choice (level-weighted JS)
        js = {}
        labels = list(m_out["conditions"].keys())
        for a in labels:
            for b in labels:
                if a >= b:
                    continue
                sa = m_out["conditions"][a].get("sampled")
                sb = m_out["conditions"][b].get("sampled")
                if sa is None or sb is None:
                    continue
                tot, acc = 0.0, 0.0
                for k in sa:
                    if not k.startswith("L") or k not in sb:
                        continue
                    w = sa[k]["n_valid"] + sb[k]["n_valid"]
                    acc += w * js_divergence_bits(sa[k]["joint_hist"], sb[k]["joint_hist"])
                    tot += w
                js[f"{a}|{b}"] = acc / max(tot, 1e-30)
        m_out["js_sampled"] = js
        results["by_m"][str(m)] = m_out

    out_dir = f"{RUN_ROOT}/idiolect"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{out_tag}" if out_tag else ""
    out_path = os.path.join(out_dir, f"idiolect_results{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"\nWrote {out_path}")
    return {"path": out_path, "m_values": m_values}


# ======================================================================
# Self-tests (CPU): recovery vs ground-truth traces, and metric calibration
# ======================================================================

def _run_self_tests():
    import numpy as np
    from rhm.rhm_data import generate_rules_invertible, possible_set_parse
    from rhm.rl_dimensionality.rl_dim_ablation import _generate_with_traces

    v, s, L = 8, 2, 6
    ok = True

    print("Test 1: recover_rule_usage vs ground-truth generation traces")
    for m in (1, 2, 3, 4, 6):
        rules = generate_rules_invertible(v, s, L, m, seed=0)
        seqs, level_features, level_rules = _generate_with_traces(rules, 500, seed=7)
        usage = recover_rule_usage(seqs, rules)
        for ell in range(L):
            f_ok = np.array_equal(usage[ell]["feat"], level_features[ell])
            r_ok = np.array_equal(usage[ell]["rule"], level_rules[ell])
            v_ok = bool(usage[ell]["valid"].all())
            if not (f_ok and r_ok and v_ok):
                ok = False
                print(f"  FAIL m={m} level={ell} feat={f_ok} rule={r_ok} valid={v_ok}")
        print(f"  m={m}: all {L} levels recover features and rules exactly")

    print("\nTest 2: validity agrees with possible_set_parse on corrupted sequences")
    for m in (2, 4):
        rules = generate_rules_invertible(v, s, L, m, seed=0)
        seqs, _, _ = _generate_with_traces(rules, 400, seed=11)
        rng = np.random.default_rng(3)
        corrupt = seqs.copy()
        idx = rng.integers(0, s ** L, size=200)
        corrupt[:200, idx[:200]] = rng.integers(0, v, size=200)
        ref = possible_set_parse(corrupt, rules)
        mine = recover_rule_usage(corrupt, rules)[0]["valid"][:, 0]
        agree = bool(np.array_equal(ref["valid"], mine))
        ok &= agree
        print(f"  m={m}: root validity matches possible_set_parse: {agree} "
              f"({ref['valid'].mean():.3f} valid)")

    print("\nTest 3: metric calibration (DGP ~ 0 bits; rule-0-forced ~ log2 m)")
    for m in (2, 4, 6):
        rules = generate_rules_invertible(v, s, L, m, seed=0)
        seqs, _, _ = _generate_with_traces(rules, 4000, seed=13)
        st = idiolect_stats(seqs, rules, prefix_len=32)
        # a maximally idiosyncratic speaker: always rule 0
        forced = _generate_forced(rules, 4000, rule_choice=0, seed=13)
        st_f = idiolect_stats(forced, rules, prefix_len=32)
        good = st["overall"]["kl_cond"] < 0.05 and abs(
            st_f["overall"]["kl_cond"] - np.log2(m)) < 1e-6
        ok &= good
        print(f"  m={m}: DGP kl_cond={st['overall']['kl_cond']:.4f} bits "
              f"(floor), forced kl_cond={st_f['overall']['kl_cond']:.4f} "
              f"(max={np.log2(m):.4f})  -> {'ok' if good else 'FAIL'}")

    print("\nTest 4: JS divergence is 0 for identical, log2(2)=1 for disjoint")
    p = [[0.5, 0.0], [0.5, 0.0]]
    q = [[0.0, 0.5], [0.0, 0.5]]
    d_same = js_divergence_bits(p, p)
    d_diff = js_divergence_bits(p, q)
    good = abs(d_same) < 1e-12 and abs(d_diff - 1.0) < 1e-12
    ok &= good
    print(f"  identical={d_same:.6f}, disjoint={d_diff:.6f} -> "
          f"{'ok' if good else 'FAIL'}")

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return ok


def _generate_forced(rules, n, rule_choice=0, seed=0):
    """DGP sampling with the rule index pinned — a maximally idiosyncratic speaker."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    current = rng.integers(0, v, size=(n, 1))
    for ell in range(L):
        n_nodes = current.shape[1]
        nxt = np.empty((n, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rule_choice]
        current = nxt
    return current


@app.function(image=image, timeout=1800)
def self_test():
    return _run_self_tests()


if __name__ == "__main__":
    _run_self_tests()
