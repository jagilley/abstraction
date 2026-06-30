"""Thread A: is the local-context synonymy signal cheap and depth-independent?

Decisive, training-free test of the central claim from the meta-learning theory
conversation ("Meta-learning for hierarchical structure discovery in RHM"):

  Synonyms (s-tuples sharing a parent feature) can be recovered from a *local*
  distributional signal -- the distribution over the sibling patch -- which is
  mediated by O(1) rule levels (the grandparent), hence its sample complexity is
  ~depth-independent. This is the cheap route that end-to-end NTP leaves on the
  table. The *supervised* signal (distribution over the root/class) is diluted
  over all L levels and is the expensive (~m^L) route.

We oracle-lift the hierarchy (use ground-truth feature values at every internal
node), which isolates the SIGNAL question from the LEARNING question. For each
bottom-up depth d we ask: how many patch-occurrences are needed to recover the
v-way parent (synonym) grouping, using local context vs the root/class signal?

  Prediction (theory): local recovery curve is ~flat across depth d; supervised
  recovery is strongly depth-dependent (cheap near the root, hopeless near the
  leaves) and overall far more data-hungry.

RHM structure (see rhm_data.py): rules[ell] has shape (v, m, s); it maps a
feature at tree-level ell to an s-tuple at tree-level ell+1. nodes[ell] holds the
feature values at tree-level ell, with nodes[0]=root (1 node) and nodes[L]=leaves
(s^L tokens). Rules are shared across positions, so synonymy is position-
independent and we pool statistics across all patch positions.

Pure numpy/CPU. Runs locally; no Modal, no GPU.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from rhm_data import generate_rules  # same DGP as all prior RHM experiments


def generate_tree(rules, n_sequences, seed=0, batch_size=20000):
    """Generate sequences AND record every internal node value.

    Returns nodes: list of L+1 arrays, nodes[ell] of shape (n_sequences, s^ell).
    nodes[0] is the root (class), nodes[L] is the leaf/token sequence.
    """
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    nodes = [np.empty((n_sequences, s ** ell), dtype=np.int64) for ell in range(L + 1)]

    for start in range(0, n_sequences, batch_size):
        end = min(start + batch_size, n_sequences)
        bs = end - start
        current = rng.integers(0, v, size=(bs, 1))
        nodes[0][start:end] = current
        for ell in range(L):  # rules[ell]: level ell -> level ell+1
            nf = current.shape[1]
            nxt = np.empty((bs, nf * s), dtype=np.int64)
            rule_choices = rng.integers(0, m, size=(bs, nf))
            for j in range(nf):
                nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rule_choices[:, j]]
            current = nxt
            nodes[ell + 1][start:end] = current
    return nodes


def encode_tuple(cols, v):
    """Encode an (N, s) array of symbols in [0,v) as a base-v integer code."""
    s = cols.shape[1]
    code = np.zeros(cols.shape[0], dtype=np.int64)
    for i in range(s):
        code += cols[:, i] * (v ** i)
    return code


def collect_occurrences(nodes, d, v, s, L):
    """Collect all patch-occurrences at bottom-up depth d.

    Representation level r = L-d+1; parent level pr = L-d; grandparent gp = L-d-1.
    Requires gp >= 0, i.e. d <= L-1 (the root level d=L has no sibling context --
    it is definitionally the class-supervised case).

    Returns dict of flat arrays (one entry per (sequence, patch) occurrence):
      tcode  : tuple code at this patch          in [0, v^s)
      parent : ground-truth parent feature       in [0, v)   <- scoring target
      sib    : sibling-patch tuple code           in [0, v^s)  <- local signal
      root   : root/class feature                 in [0, v)    <- supervised signal
    """
    assert s == 2, "sibling logic implemented for s=2 (our setting)"
    r = L - d + 1
    pr = L - d
    rep = nodes[r]                 # (n_seq, s^r)
    par = nodes[pr]                # (n_seq, s^(r-1))  one parent per patch
    root_col = nodes[0][:, 0]      # (n_seq,)
    n_seq = rep.shape[0]
    n_patch = s ** (r - 1)

    tcode_list, parent_list, sib_list, root_list = [], [], [], []
    for p in range(n_patch):
        tup = rep[:, s * p:s * p + s]
        sib_p = p ^ 1              # s=2: the co-child under the shared grandparent
        sib = rep[:, s * sib_p:s * sib_p + s]
        tcode_list.append(encode_tuple(tup, v))
        sib_list.append(encode_tuple(sib, v))
        parent_list.append(par[:, p])
        root_list.append(root_col)

    return {
        "tcode": np.concatenate(tcode_list),
        "parent": np.concatenate(parent_list),
        "sib": np.concatenate(sib_list),
        "root": np.concatenate(root_list),
        "n_patch_per_seq": n_patch,
    }


def cluster_recovery(tcode, context, parent, v, n_ctx, k):
    """Cluster tuple types by P(context | tuple) and score parent recovery.

    tcode   : (O,) tuple codes in [0, v^s)
    context : (O,) context codes in [0, n_ctx)  (sibling code, or root)
    parent  : (O,) ground-truth parent in [0, v)  (scoring target)
    k       : number of clusters to fit (<= v)

    Returns occurrence-weighted accuracy of the best cluster->parent matching.
    Chance ~ 1/v.
    """
    O = len(tcode)
    uniq, inv = np.unique(tcode, return_inverse=True)
    n_types = len(uniq)
    k = min(k, n_types)
    if k < 2:
        return float("nan")

    # P(context | tuple): rows = tuple types, cols = context values
    joint = np.zeros((n_types, n_ctx), dtype=np.float64)
    np.add.at(joint, (inv, context), 1.0)
    counts = joint.sum(axis=1)                       # occurrences per tuple type
    cond = joint / counts[:, None]                   # row-normalized

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    type_clusters = km.fit_predict(cond, sample_weight=counts)

    # occurrence-level confusion between assigned cluster and true parent
    occ_cluster = type_clusters[inv]
    conf = np.zeros((k, v), dtype=np.float64)
    np.add.at(conf, (occ_cluster, parent), 1.0)

    row, col = linear_sum_assignment(-conf)          # maximize matched mass
    matched = conf[row, col].sum()
    return matched / O


def collect_from_rep(rep, parent_arr, v, s):
    """Collect (tuple, sibling, parent) occurrences from a representation array.

    rep        : (n_seq, length) symbols in [0, v)
    parent_arr : (n_seq, length // s) ground-truth parent feature in [0, v)
    Works for both a true-feature representation (oracle path) and a predicted-
    lifted representation (compounded path), since both use a size-v alphabet.
    """
    assert s == 2, "sibling logic implemented for s=2"
    n_patch = rep.shape[1] // s
    tcode_l, sib_l, par_l = [], [], []
    for p in range(n_patch):
        tup = rep[:, s * p:s * p + s]
        sib = rep[:, s * (p ^ 1):s * (p ^ 1) + s]
        tcode_l.append(encode_tuple(tup, v))
        sib_l.append(encode_tuple(sib, v))
        par_l.append(parent_arr[:, p])
    return (np.concatenate(tcode_l), np.concatenate(sib_l), np.concatenate(par_l))


def fit_and_score(tcode, context, parent, v, n_ctx, k, max_code):
    """Cluster tuple types by P(context|tuple); return (accuracy, code->cluster map).

    Uses ALL given occurrences (no subsampling) -- the compounding test asks
    whether the operator composes in the data-rich limit, isolating error
    cascade from sample complexity (which Thread A already settled).
    """
    uniq, inv = np.unique(tcode, return_inverse=True)
    k_eff = min(k, len(uniq))
    joint = np.zeros((len(uniq), n_ctx), dtype=np.float64)
    np.add.at(joint, (inv, context), 1.0)
    counts = joint.sum(axis=1)
    cond = joint / counts[:, None]
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
    type_clusters = km.fit_predict(cond, sample_weight=counts)

    occ_cluster = type_clusters[inv]
    conf = np.zeros((k_eff, v), dtype=np.float64)
    np.add.at(conf, (occ_cluster, parent), 1.0)
    row, col = linear_sum_assignment(-conf)
    acc = conf[row, col].sum() / len(tcode)

    code2cluster = np.full(max_code, -1, dtype=np.int64)
    code2cluster[uniq] = type_clusters
    return acc, code2cluster, len(np.unique(type_clusters))


def lift(rep, code2cluster, v, s):
    """Replace each s-patch by its predicted cluster label -> shorter sequence."""
    n_seq, length = rep.shape
    n_patch = length // s
    patches = rep.reshape(n_seq, n_patch, s)
    codes = np.zeros((n_seq, n_patch), dtype=np.int64)
    for i in range(s):
        codes += patches[:, :, i] * (v ** i)
    return code2cluster[codes]


def run_compounding(v=8, s=2, L=6, m=4, rule_seed=0, seq_seed=1, n_sequences=80000,
                    verbose=True, rules=None):
    """Thread A.5: iterated, NON-oracle cluster-and-lift.

    Lift each level using PREDICTED clusters (not ground truth), recurse, and
    measure per-level parent recovery on the predicted-lifted representation vs
    the oracle (clean-input) recovery. The gap = the cost of error compounding.
    Does the cluster-and-lift operator actually compose, or do errors cascade?

    Pass `rules` to use an externally-generated rule set (e.g. the distinct-rule
    DGP used for Thread B); otherwise the default generate_rules set is used.
    """
    if rules is None:
        rules = generate_rules(v, s, L, m, seed=rule_seed)
    else:
        v, m, s = rules[0].shape
    nodes = generate_tree(rules, n_sequences, seed=seq_seed)
    v_s = v ** s
    chance = 1.0 / v

    rec = {"depth": [], "oracle": [], "compounded": [], "eff_vocab": []}
    R_pred = nodes[L].copy()  # bottom representation = leaves; gets lifted in place
    for d in range(1, L):     # d = 1 .. L-1
        parent = nodes[L - d]                      # ground-truth parent at this level
        # oracle path: cluster the TRUE representation at this level (clean input)
        t_o, sib_o, par_o = collect_from_rep(nodes[L - d + 1], parent, v, s)
        acc_o, _, _ = fit_and_score(t_o, sib_o, par_o, v, v_s, v, v_s)
        # compounded path: cluster the PREDICTED-lifted representation
        t_c, sib_c, par_c = collect_from_rep(R_pred, parent, v, s)
        acc_c, c2c, eff = fit_and_score(t_c, sib_c, par_c, v, v_s, v, v_s)

        rec["depth"].append(d); rec["oracle"].append(float(acc_o))
        rec["compounded"].append(float(acc_c)); rec["eff_vocab"].append(int(eff))
        R_pred = lift(R_pred, c2c, v, s)           # propagate predicted labels up

    if verbose:
        print(f"\nRHM: v={v} s={s} L={L} m={m}  n_seq={n_sequences}  "
              f"rule_seed={rule_seed}  chance={chance:.3f}")
        print("Iterated cluster-and-lift (predicted lifts propagate upward):")
        print(f"{'depth':>6} | {'oracle(clean)':>14} {'compounded(pred)':>17} "
              f"{'gap':>7} {'eff_vocab':>10}")
        for i, d in enumerate(rec["depth"]):
            o, c, e = rec["oracle"][i], rec["compounded"][i], rec["eff_vocab"][i]
            print(f"{d:>6} | {o:>14.3f} {c:>17.3f} {o - c:>7.3f} {e:>10}/{v}")
    return rec


def run_bp_ceiling(v=8, s=2, L=6, m=4, rule_seed=0, seq_seed=1, n_sequences=10000,
                   verbose=True, rules=None):
    """Optimal-inference ceiling: exact tree belief propagation with known rules.

    The RHM is a tree, so sum-product is exact. Given the observed leaves and the
    true rules, we compute the posterior marginal over every internal node and
    report MAP recovery of the true feature per level. This is the best ANY
    inference could do (having localized the rules) -- the information ceiling.

    Compare to the greedy-hard cascade (run_compounding): if BP is flat-high while
    the cascade decays, the cascade's failure is pure inference suboptimality and
    the information is fully present; a learned model's job is to approach BP.

    Returns {level: (map_acc, post_on_true)} for levels 0..L-1 (level ell maps to
    bottom-up depth d = L - ell).
    """
    assert s == 2, "BP implemented for s=2"
    if rules is None:
        rules = generate_rules(v, s, L, m, seed=rule_seed)
    else:
        v, m, s = rules[0].shape
    nodes = generate_tree(rules, n_sequences, seed=seq_seed)
    N = n_sequences

    def norm(x):
        d = x.sum(axis=2, keepdims=True)
        return x / np.where(d > 0, d, 1.0)

    # ---- upward pass: lam[ell][n,j,f] = P(subtree leaves | node (ell,j) = f) ----
    lam = [None] * (L + 1)
    lam[L] = np.eye(v, dtype=np.float64)[nodes[L]]            # leaves: one-hot
    for ell in range(L - 1, -1, -1):
        child = lam[ell + 1]
        left, right = child[:, 0::2, :], child[:, 1::2, :]   # (N, s^ell, v)
        rL, rR = rules[ell][:, :, 0], rules[ell][:, :, 1]    # (v, m)
        msg = (left[:, :, rL] * right[:, :, rR]).sum(axis=3) / m
        lam[ell] = norm(msg)

    # ---- downward pass + per-level marginals ----
    rec = {}
    pi = np.full((N, 1, v), 1.0 / v)                          # root prior
    root_marg = norm(pi * lam[0])
    rec[0] = ((root_marg.argmax(2) == nodes[0]).mean(),
              root_marg[np.arange(N)[:, None], np.zeros((N, 1), int), nodes[0]].mean())

    for ell in range(0, L - 1):                               # produce children at ell+1
        rL, rR = rules[ell][:, :, 0], rules[ell][:, :, 1]
        left, right = lam[ell + 1][:, 0::2, :], lam[ell + 1][:, 1::2, :]
        S_left = np.zeros((v, m, v)); S_right = np.zeros((v, m, v))
        fi, ri = np.arange(v)[:, None], np.arange(m)[None, :]
        S_left[fi, ri, rL] = 1.0                              # scatter by left child value
        S_right[fi, ri, rR] = 1.0
        # downward message to each child = pi(parent) x other-child upward, through factor
        w_left = pi[:, :, :, None] * right[:, :, rR] / m
        w_right = pi[:, :, :, None] * left[:, :, rL] / m
        m_left = norm(np.einsum('njfr,frc->njc', w_left, S_left))
        m_right = norm(np.einsum('njfr,frc->njc', w_right, S_right))
        pin = np.empty((N, s ** (ell + 1), v))
        pin[:, 0::2, :], pin[:, 1::2, :] = m_left, m_right
        pi = pin
        marg = norm(pin * lam[ell + 1])
        map_acc = (marg.argmax(2) == nodes[ell + 1]).mean()
        post_true = np.take_along_axis(marg, nodes[ell + 1][:, :, None], axis=2).mean()
        rec[ell + 1] = (float(map_acc), float(post_true))

    if verbose:
        print(f"\nRHM: v={v} s={s} L={L} m={m}  n_seq={N}  rule_seed={rule_seed}  "
              f"chance={1.0/v:.3f}")
        print("Optimal tree BP (known rules) -- per-level recovery of true feature:")
        print(f"{'level':>5} {'depth d':>8} | {'MAP acc':>8} {'post(true)':>11}")
        for ell in sorted(rec):
            ma, pt = rec[ell]
            print(f"{ell:>5} {L-ell:>8} | {ma:>8.3f} {pt:>11.3f}")
    return rec


def run(v=8, s=2, L=6, m=4, rule_seed=0, seq_seed=1, n_sequences=80000,
        o_grid=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000),
        n_repeat=3, out_dir=None):
    rules = generate_rules(v, s, L, m, seed=rule_seed)
    print(f"RHM: v={v} s={s} L={L} m={m}  seq_len={s**L}  "
          f"n_sequences={n_sequences}  (rule_seed={rule_seed})")
    nodes = generate_tree(rules, n_sequences, seed=seq_seed)

    v_s = v ** s
    chance = 1.0 / v
    rng = np.random.default_rng(123)
    results = {"config": dict(v=v, s=s, L=L, m=m, rule_seed=rule_seed,
                              seq_seed=seq_seed, n_sequences=n_sequences,
                              chance=chance, o_grid=list(o_grid)),
               "depths": {}}

    for d in range(1, L):  # d = 1 .. L-1 ; root level d=L has no sibling context
        occ = collect_occurrences(nodes, d, v, s, L)
        tcode, parent, sib, root = occ["tcode"], occ["parent"], occ["sib"], occ["root"]
        n_avail = len(tcode)
        n_legal = len(np.unique(tcode))
        per_seq = occ["n_patch_per_seq"]
        print(f"\n=== depth d={d}  (rep level {L-d+1} -> parent level {L-d}) ===")
        print(f"    {n_avail:,} occurrences  ({per_seq}/seq)  | "
              f"{n_legal} distinct tuples (<= m*v = {m*v})  | chance={chance:.3f}")
        print(f"    {'O':>8} {'seqs':>9} | {'local(sib)':>12} {'supervised(root)':>17}")

        rec = {"O": [], "seqs_equiv": [], "local": [], "supervised": []}
        for O in o_grid:
            if O > n_avail:
                continue
            loc_acc, sup_acc = [], []
            for _ in range(n_repeat):
                idx = rng.choice(n_avail, size=O, replace=False)
                loc_acc.append(cluster_recovery(tcode[idx], sib[idx], parent[idx],
                                                v, v_s, k=v))
                sup_acc.append(cluster_recovery(tcode[idx], root[idx], parent[idx],
                                                v, v, k=v))
            lo, su = float(np.mean(loc_acc)), float(np.mean(sup_acc))
            seqs_equiv = O / per_seq
            rec["O"].append(O); rec["seqs_equiv"].append(seqs_equiv)
            rec["local"].append(lo); rec["supervised"].append(su)
            print(f"    {O:>8} {seqs_equiv:>9.1f} | {lo:>12.3f} {su:>17.3f}")
        results["depths"][d] = rec

    # Summary: occurrences to reach 80% parent recovery (interpolated), per depth.
    print("\n" + "=" * 64)
    print("SUMMARY: patch-occurrences to reach 0.80 parent-recovery")
    print(f"{'depth':>6} | {'local(sib)':>12} {'supervised(root)':>17}")
    target = 0.80

    def crossing(O_list, acc_list):
        for O, a in zip(O_list, acc_list):
            if a >= target:
                return O
        return None

    for d in range(1, L):
        rec = results["depths"][d]
        lo = crossing(rec["O"], rec["local"])
        su = crossing(rec["O"], rec["supervised"])
        lo_s = f"{lo:,}" if lo else ">max"
        su_s = f"{su:,}" if su else ">max"
        print(f"{d:>6} | {lo_s:>12} {su_s:>17}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "local_signal_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        _plot(results, os.path.join(out_dir, "local_signal.png"))
        print(f"\nSaved results + plot to {out_dir}")
    return results


def _plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted(results["depths"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(depths)))
    chance = results["config"]["chance"]
    for ax, key, title in [(axes[0], "local", "local sibling-context signal"),
                           (axes[1], "supervised", "supervised root/class signal")]:
        for c, d in zip(cmap, depths):
            rec = results["depths"][d]
            ax.plot(rec["O"], rec[key], "-o", color=c, label=f"d={d}")
        ax.axhline(chance, ls="--", color="gray", lw=1, label="chance")
        ax.set_xscale("log")
        ax.set_xlabel("patch-occurrences used")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("parent (synonym) recovery accuracy")
    cfg = results["config"]
    fig.suptitle(f"RHM local vs supervised synonymy signal  "
                 f"(v={cfg['v']} s={cfg['s']} L={cfg['L']} m={cfg['m']})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "compound", "bp"], default="signal",
                    help="signal = Thread A (oracle, sample complexity); "
                         "compound = Thread A.5 (iterated non-oracle lift); "
                         "bp = optimal tree belief-propagation ceiling")
    ap.add_argument("--v", type=int, default=8)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n-sequences", type=int, default=80000)
    ap.add_argument("--rule-seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    if args.mode == "signal":
        run(v=args.v, s=args.s, L=args.L, m=args.m, n_sequences=args.n_sequences,
            rule_seed=args.rule_seed, out_dir=args.out_dir)
    elif args.mode == "compound":
        run_compounding(v=args.v, s=args.s, L=args.L, m=args.m,
                        n_sequences=args.n_sequences, rule_seed=args.rule_seed)
    else:
        run_bp_ceiling(v=args.v, s=args.s, L=args.L, m=args.m,
                       n_sequences=args.n_sequences, rule_seed=args.rule_seed)
