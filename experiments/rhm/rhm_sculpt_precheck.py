"""RHM Sculpting — the perfect-simulator pre-check for a genuine control task.

Motivation (see ACTIVE_RHM_README Phase 4). Active-query RHM is a *sensing* POMDP:
actions reveal a static hidden root, so acting = greedy value-of-information = the
forecast (act ≈ plan), and internalizing a forward model has no headroom. To turn
RHM into a *control* domain (act ≠ plan) we change what an action does: instead of
revealing hidden blocks, the agent EDITS observed leaves toward a target root r*.
Editing an observed token is deterministic and forecastable; the world-model it
must learn is the grammar's bottom-up parse. But this only creates a real
world-model requirement if the task has a genuine **lookahead prize** — if a
coordinated multi-edit plan beats a myopic reflex. Hierarchical coupling (a valid
subtree needs its leaves to be a legal production, and a parent feature couples its
whole subtree) should supply that.

This script validates the design with pure combinatorics, no learning, mirroring
the ceiling-probe methodology: prove the phenomenon is in the *task* before
building the *learning*.

  - PLANNER (perfect simulator): exact DP over the known grammar giving d* = the
    minimum number of token edits to turn the current leaves into ANY valid
    r*-derivation. Succeeds within a budget iff d* ≤ budget. This is full-horizon
    optimal planning.
  - REFLEX (myopic editor): greedy 1-ply hill-climbing on a *local* parse-
    consistency heuristic (how much of the tree parses to anything), with no
    multi-step coordination — the best a model-free reflex could do.

If the reflex solves ~all within-budget-solvable instances → no lookahead prize →
still act ≈ plan → increase coupling and re-check. If the reflex gets stuck at
local optima (everything locally valid, wrong root) while the planner solves →
genuine act ≠ plan → the learned/internalized version is worth building.

Run from experiments/:
  modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck
  modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --depths "3,4,5" --n-instances 2000
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-sculpt-precheck", image=image)

LARGE = 10 ** 6


# --------------------------------------------------------------------------- #
# Perfect simulator: exact grammar operations (all batched over B instances)
# --------------------------------------------------------------------------- #

def sample_derivations(rules, roots, s, rng):
    """Given roots (B,), sample one valid derivation each → leaves (B, s^L)."""
    L = len(rules)
    _v, m, _s = rules[0].shape
    current = roots[:, None]
    for ell in range(L):
        width = current.shape[1]
        choices = rng.integers(0, m, size=(current.shape[0], width))
        nxt = np.empty((current.shape[0], width * s), dtype=np.int64)
        for j in range(width):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], choices[:, j]]
        current = nxt
    return current


def nearest_derivation_cost(rules, leaves, target_roots, s):
    """Exact DP: min token edits to turn each leaf sequence into a valid
    derivation of its target root. leaves (B,T), target_roots (B,) → d* (B,)."""
    L = len(rules)
    v, m, _s = rules[0].shape
    B = leaves.shape[0]
    n_bottom = leaves.shape[1] // s
    blocks = leaves.reshape(B, n_bottom, s)
    bottom = rules[L - 1]  # (v, m, s)
    # per bottom node, per feature f: min over rules r of hamming(block, tuple_{f,r})
    diff = (bottom[None, None] != blocks[:, :, None, None, :]).sum(-1)  # (B,n_bottom,v,m)
    cost = diff.min(-1)  # (B, n_bottom, v)
    for ell in range(L - 2, -1, -1):
        n_parents = cost.shape[1] // s
        children = cost.reshape(B, n_parents, s, v)
        layer = rules[ell]  # (v, m, s)
        total = np.zeros((B, n_parents, v, m), dtype=np.int64)
        for i in range(s):
            idx = layer[:, :, i]  # (v, m)
            total += children[:, :, i, :][:, :, idx]  # (B,n_parents,v,m)
        cost = total.min(-1)  # (B, n_parents, v)
    return cost[np.arange(B), 0, target_roots]


def possible_sets(rules, leaves, s):
    """Bottom-up exact possibility: for each node, which features it can derive.
    leaves (B,T) → list over levels of bool (B, n_nodes, v); levels[-1] is root."""
    L = len(rules)
    v, m, _s = rules[0].shape
    B = leaves.shape[0]
    n_bottom = leaves.shape[1] // s
    blocks = leaves.reshape(B, n_bottom, s)
    bottom = rules[L - 1]  # (v, m, s)
    P = (bottom[None, None] == blocks[:, :, None, None, :]).all(-1).any(-1)  # (B,n_bottom,v)
    levels = [P]
    current = P
    for ell in range(L - 2, -1, -1):
        n_parents = current.shape[1] // s
        children = current.reshape(B, n_parents, s, v)
        layer = rules[ell]
        ok = np.ones((B, n_parents, v, m), dtype=bool)
        for i in range(s):
            idx = layer[:, :, i]  # (v, m)
            ok &= children[:, :, i, :][:, :, idx]  # (B,n_parents,v,m)
        current = ok.any(-1)  # (B, n_parents, v)
        levels.append(current)
    return levels


def parse_success_and_heuristic(rules, leaves, target_roots, s):
    """Returns (success (B,) bool: root parses to r*, heuristic (B,) int: number of
    tree nodes that parse to any feature — the myopic reflex's local score)."""
    B = leaves.shape[0]
    levels = possible_sets(rules, leaves, s)
    root_P = levels[-1][:, 0, :]  # (B, v)
    success = root_P[np.arange(B), target_roots]
    heuristic = np.zeros(B, dtype=np.int64)
    for lvl in levels:
        heuristic += lvl.any(-1).sum(1)  # count nodes with non-empty possible-set
    return success, heuristic


# --------------------------------------------------------------------------- #
# Reflex editor: myopic 1-ply hill-climbing on the local parse-consistency score
# --------------------------------------------------------------------------- #

def greedy_edit_solve(rules, start_leaves, target_root, s, budget):
    """Myopic editor for one instance. Each step: evaluate every single-token edit,
    take the one that most raises parse-consistency (or that solves). Stop on
    success, budget exhaustion, or a local optimum (no strict improvement).
    Returns (solved: bool, stuck: bool, edits_used: int)."""
    T = start_leaves.shape[0]
    v = rules[0].shape[0]
    leaves = start_leaves.copy()
    target = np.array([target_root])

    success, current_h = parse_success_and_heuristic(rules, leaves[None], target, s)
    if success[0]:
        return True, False, 0
    current_h = int(current_h[0])

    positions = np.repeat(np.arange(T), v)
    values = np.tile(np.arange(v), T)
    for step in range(1, budget + 1):
        candidates = np.tile(leaves, (T * v, 1))
        candidates[np.arange(T * v), positions] = values
        succ, hs = parse_success_and_heuristic(rules, candidates, np.full(T * v, target_root), s)
        if succ.any():
            return True, False, step
        best = int(hs.argmax())
        if hs[best] <= current_h:
            return False, True, step - 1  # local optimum: stuck with budget to spare
        leaves = candidates[best]
        current_h = int(hs[best])
    return False, False, budget  # ran out of budget (not a local optimum)


def greedy_topdown_cost(rules, start, roots, s):
    """A STRONGER, r*-aware reflex: commit to an r*-derivation top-down with myopic
    (1-level-lookahead) rule choices — at each node pick the rule whose s child
    targets are most already-derivable (in the child possible-sets) — then pay the
    true edit cost of that choice. d_greedy ≥ d* exactly by the amount the optimal
    solution needs non-myopic (coordinated) rule selection. Returns d_greedy (B,)."""
    import sys

    L = len(rules)
    v, m, _s = rules[0].shape
    B = start.shape[0]
    n_bottom = start.shape[1] // s
    blocks = start.reshape(B, n_bottom, s)
    bottom = rules[L - 1]
    bottom_cost = (bottom[None, None] != blocks[:, :, None, None, :]).sum(-1).min(-1)  # (B,n_bottom,v)
    levels = possible_sets(rules, start, s)  # bottom..root

    def P(t):  # possible-sets at tree level t (0=root .. L-1=bottom): (B, s^t, v)
        return levels[(L - 1) - t]

    sys.setrecursionlimit(10000)

    def assign(b, level, node, feat):
        if level == L - 1:
            return int(bottom_cost[b, node, feat])
        layer = rules[level]  # (v, m, s)
        best_rule, best_shallow = 0, None
        for r in range(m):
            targets = layer[feat, r]
            shallow = sum(0 if P(level + 1)[b, node * s + i, targets[i]] else 1 for i in range(s))
            if best_shallow is None or shallow < best_shallow:
                best_shallow, best_rule = shallow, r
        targets = layer[feat, best_rule]
        return sum(assign(b, level + 1, node * s + i, int(targets[i])) for i in range(s))

    return np.array([assign(b, 0, 0, int(roots[b])) for b in range(B)], dtype=np.int64)


# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def sculpt_precheck(
    v: int = 8,
    s: int = 2,
    depths: str = "3,4,5",
    m: int = 4,
    corrupt_blocks: str = "1,2,3",
    n_instances: int = 1500,
    rule_seed: int = 0,
    data_seed: int = 1,
    quick: bool = False,
):
    """Greedy-reflex vs perfect-simulator-planner on RHM Sculpting."""
    import time

    Ls = [int(x) for x in depths.split(",")]
    corruptions = [int(x) for x in corrupt_blocks.split(",")]
    if quick:
        Ls, corruptions, n_instances = [3, 4], [1, 2], 200

    started = time.time()
    results = {"config": {"v": v, "s": s, "m": m, "depths": Ls,
                          "corrupt_blocks": corruptions, "n_instances": n_instances,
                          "rule_seed": rule_seed, "data_seed": data_seed}, "by_depth": {}}

    for L in Ls:
        seq_len = s ** L
        n_blocks = seq_len // s
        rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
        rng = np.random.default_rng(data_seed + L)
        print(f"\n=== L={L} (seq_len={seq_len}, {n_blocks} blocks) ===")

        # sanity: an uncorrupted derivation parses to its root at d*=0
        chk_roots = rng.integers(0, v, size=64)
        chk_leaves = sample_derivations(rules, chk_roots, s, rng)
        chk_d = nearest_derivation_cost(rules, chk_leaves, chk_roots, s)
        chk_succ, _ = parse_success_and_heuristic(rules, chk_leaves, chk_roots, s)
        assert chk_d.max() == 0, f"uncorrupted d* should be 0, got max {chk_d.max()}"
        assert chk_succ.all(), "uncorrupted sequences must parse to their root"
        print(f"  sanity ok: uncorrupted d*=0, all parse")

        rows = []
        for nc in corruptions:
            if nc >= n_blocks:
                continue
            roots = rng.integers(0, v, size=n_instances)
            clean = sample_derivations(rules, roots, s, rng)
            start = clean.copy()
            for b in range(n_instances):
                blocks = rng.choice(n_blocks, size=nc, replace=False)
                for blk in blocks:
                    start[b, blk * s:(blk + 1) * s] = rng.integers(0, v, size=s)

            budget = nc * s  # generous: enough tokens to fully rewrite corrupted blocks
            dstar = nearest_derivation_cost(rules, start, roots, s)
            frac_solvable = float((dstar <= budget).mean())
            solv_mask = dstar <= budget

            solved = np.zeros(n_instances, dtype=bool)
            stuck = np.zeros(n_instances, dtype=bool)
            for b in range(n_instances):
                sv, st, _ = greedy_edit_solve(rules, start[b], int(roots[b]), s, budget)
                solved[b] = sv
                stuck[b] = st
            greedy_success = float(solved.mean())
            greedy_stuck = float(stuck.mean())
            prize = float((solv_mask & ~solved).mean()) if solv_mask.any() else 0.0

            # stronger r*-aware reflex
            d_topdown = greedy_topdown_cost(rules, start, roots, s)
            topdown_success = float((d_topdown <= budget).mean())
            topdown_prize = float((solv_mask & (d_topdown > budget)).mean()) if solv_mask.any() else 0.0

            row = {"corrupt_blocks": nc, "budget_tokens": budget,
                   "dstar_mean": float(dstar.mean()), "dstar_max": int(dstar.max()),
                   "frac_solvable_planner": frac_solvable,
                   "greedy_parse_success": greedy_success, "greedy_parse_stuck": greedy_stuck,
                   "greedy_parse_prize": prize,
                   "greedy_topdown_success": topdown_success, "greedy_topdown_prize": topdown_prize,
                   "d_topdown_mean": float(d_topdown.mean())}
            rows.append(row)
            print(f"  corrupt={nc} budget={budget}: d*_mean={row['dstar_mean']:.2f} "
                  f"| planner=1.000 | parse-reflex solves={greedy_success:.3f} (stuck {greedy_stuck:.3f}) "
                  f"| r*-reflex solves={topdown_success:.3f} → prize parse={prize:.3f} / r*={topdown_prize:.3f}")
        results["by_depth"][str(L)] = rows

    results["elapsed_seconds"] = time.time() - started

    output_dir = f"{DATA_DIR}/rhm_sculpt_precheck"
    os.makedirs(output_dir, exist_ok=True)
    tag = f"v{v}_s{s}_m{m}_rs{rule_seed}"
    with open(f"{output_dir}/{tag}.json", "w") as handle:
        json.dump(results, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    sculpt_precheck.remote()
