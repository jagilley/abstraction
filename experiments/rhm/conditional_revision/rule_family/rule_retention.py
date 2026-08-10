"""Porting `../synonym_retention/`'s instrument onto the rule-family substrate, to
supply the positive control arm that cut proves cannot exist on fixed rules.

`synonym_retention`'s "An impossibility worth recording" says: on fixed-rules RHM, any
perturbation that changes the future predictive distribution must move a latent whose
descendants extend past the read position, which breaks prefix-identity. So a closed
constituent is very nearly exactly NTP-redundant, there is no NTP-required-at-distance
content, and their retention curve is pinned at chance below and a Bayes ceiling above
with **no "content the model must keep" reference arm**. That is a stated limitation of
their result.

The rule family lifts it. Perturb the rule choice at a node of the DIFFERING level (the
`top` arm at d2, since the leaf-emitting table is shared by design and carries no rule
information): the constituent still closes, prefix-identity outside its span still holds,
but the realisation it emitted is *evidence about which rule set is active*, and the rule
set governs every later sequence in the window. So the perturbation changes the predictive
distribution arbitrarily far ahead -- across sequence boundaries, where the rule posterior
is the ONLY surviving channel.

Two parts, deliberately separable:

  PART 1 (this file, runnable locally on CPU, NO MODEL). The exact next-token TV between
    base and perturbed prefixes as a function of read distance `w`, family vs floor. This
    is the structural claim and it is a property of the DGP alone: the floor arm must
    decay to ~0 (their impossibility) while the family arm stays non-zero across
    boundaries. If that separation does not appear, the substrate does not do what this
    whole sub-experiment assumes and nothing downstream matters.

  PART 2 (Modal, cached checkpoints). Does the MODEL retain it -- decode the perturbed
    rule choice from h[e+w], normalised between chance and the exact Bayes ceiling, family
    vs floor, against their lambda=0 baseline (rule retention +0.858 at w=0, +0.079 at
    w=8, chance by w=16).

The perturbation is FROZEN across the w sweep, per their gotcha: the same bytes change at
every w, so perturbation size cannot confound distance.

    cd experiments
    python3 -m rhm.conditional_revision.rule_family.rule_retention          # part 1
"""

import argparse
import json
import os

import modal
import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

from rhm.conditional_revision.rule_family.family import make_family
from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS
from rhm.conditional_revision.rule_family.oracle_mixture import (
    family_predictive, mixture_profiles,
)
from rhm.conditional_revision.synonym_retention.synonym_retention import _realize


def _traces(rules, n, seed):
    """Root features and per-level rule choices for n sequences, plus the leaves."""
    rng = np.random.default_rng(seed)
    v, m, s = rules[0].shape
    L = len(rules)
    root = rng.integers(0, v, size=(n, 1))
    rc = []
    cur = root
    for d in range(L):
        rc.append(rng.integers(0, m, size=(n, cur.shape[1])))
        nxt = np.empty((n, cur.shape[1] * s), dtype=np.int64)
        for j in range(cur.shape[1]):
            nxt[:, j * s:(j + 1) * s] = rules[d][cur[:, j], rc[d][:, j]]
        cur = nxt
    return root, rc, cur


def _mixture_predictive(family, wins, rule_ids, chunk=2048):
    """P(x_g | x_{<g}) under the exact mixture over rule sets. (n, G, v)."""
    R = len(family)
    n, K, T = wins.shape
    v = family[0][0].shape[0]
    post = family_predictive(family, wins.reshape(-1, T), chunk=chunk)
    post = post.reshape(R, n, K, T, v)
    prof = mixture_profiles(post, wins, rule_ids)
    w_pre = prof["w_pre"]                                   # (n, G, R)
    p = post.reshape(R, n, K * T, v).transpose(1, 2, 0, 3)  # (n, G, R, v)
    return (w_pre[:, :, :, None] * p).sum(axis=2)


def part1(design="d2_R64_nF2", v=16, s=2, L=6, m=4, family_seed=0,
          K=4, n_windows=128, seed=7, d_perturb=None, j_node=0,
          w_grid=None, verbose=True):
    """DGP-only: exact next-token TV vs read distance, family vs floor."""
    R, dl, nF, mode = DESIGNS[design]
    d = dl[0] if d_perturb is None else d_perturb
    T = s ** L
    G = K * T
    span = s ** (L - d)
    close = (j_node + 1) * span - 1            # leaf where the constituent closes
    if w_grid is None:
        w_grid = [0, 1, 2, 4, 8, 16, 32, 48, 60,
                  T - close, T - close + 4, T - close + 32,
                  2 * T - close, 2 * T - close + 32, 3 * T - close - 2]
        w_grid = sorted({w for w in w_grid if 0 <= close + w <= G - 2})

    fam, meta = make_family(v, s, L, m, R=R, differ_levels=dl,
                            n_differ_features=nF, seed=family_seed, mode=mode)
    flo, _ = make_family(v, s, L, m, R=R, differ_levels=[],
                         n_differ_features=nF, seed=family_seed, mode=mode)
    out = {"design": design, "differ_level": d, "node": j_node,
           "span": span, "close_leaf": close, "w_grid": w_grid,
           "K": K, "n_windows": n_windows, "arms": {}}
    if verbose:
        print(f"design={design}  perturb rules[{d}] node {j_node} "
              f"(span {span} leaves, closes at leaf {close})")
        print(f"window = {K} x {T} = {G} tokens; boundaries at "
              f"{[k * T for k in range(1, K)]}")

    for arm, family in (("family", fam), ("floor", flo)):
        rng = np.random.default_rng(seed)
        rule_ids = rng.integers(0, len(family), size=n_windows)
        # sequence 0 of each window carries the perturbation; the rest are fillers
        base = np.empty((n_windows, K, T), dtype=np.int64)
        pert = np.empty((n_windows, K, T), dtype=np.int64)
        for r in range(len(family)):
            sel = np.where(rule_ids == r)[0]
            if not sel.size:
                continue
            root, rc, leaves = _traces(family[r], sel.size, seed + 100 * r + 1)
            # frozen `top` perturbation: bump ONLY the level-d rule choice at node j
            rc2 = [x.copy() for x in rc]
            bump = rng.integers(1, m, size=sel.size)
            rc2[d][:, j_node] = (rc[d][:, j_node] + bump) % m
            leaves2 = _realize(family[r], root, rc2, s)
            assert (leaves[:, :j_node * span] == leaves2[:, :j_node * span]).all()
            assert (leaves[:, close + 1:] == leaves2[:, close + 1:]).all()
            base[sel, 0], pert[sel, 0] = leaves, leaves2
            for k in range(1, K):
                _, _, fl = _traces(family[r], sel.size, seed + 100 * r + 10 * k + 2)
                base[sel, k] = fl
                pert[sel, k] = fl                       # fillers identical across arms

        pb = _mixture_predictive(family, base, rule_ids)
        pp = _mixture_predictive(family, pert, rule_ids)
        tv = 0.5 * np.abs(pb - pp).sum(-1)                          # (n, G)
        row = {}
        for w in w_grid:
            g = close + w
            row[str(w)] = {"tv_mean": float(tv[:, g].mean()),
                           "tv_frac_zero": float((tv[:, g] < 1e-12).mean()),
                           "crosses_boundary": bool(g >= T)}
        out["arms"][arm] = row
        if verbose:
            print(f"\n  {arm}: exact next-token TV at read distance w after close")
            print(f"    {'w':>5}{'g':>6}{'bnd':>5}{'TV':>10}{'frac TV=0':>11}")
            for w in w_grid:
                rr = row[str(w)]
                print(f"    {w:>5}{close + w:>6}"
                      f"{('yes' if rr['crosses_boundary'] else ''):>5}"
                      f"{rr['tv_mean']:>10.5f}{rr['tv_frac_zero']:>11.3f}")

    if verbose and "family" in out["arms"] and "floor" in out["arms"]:
        print(f"\n  {'w':>5}{'family TV':>12}{'floor TV':>11}{'ratio':>10}")
        for w in w_grid:
            a = out["arms"]["family"][str(w)]["tv_mean"]
            b = out["arms"]["floor"][str(w)]["tv_mean"]
            print(f"  {w:>5}{a:>12.5f}{b:>11.5f}"
                  f"{(a / b if b > 1e-12 else float('inf')):>10.1f}")
        print("\n  synonym_retention's impossibility predicts the FLOOR column decays to "
              "~0.\n  A persistent FAMILY column across a boundary is the "
              "NTP-required-at-distance\n  positive control that cut says cannot exist on "
              "fixed rules.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="d2_R64_nF2")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = part1(design=a.design, K=a.K, n_windows=a.n_windows, j_node=a.node)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# PART 2 -- the model-side retention curve
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-rule-retention", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=49152)
def part2(
    design: str = "d2_R64_nF2",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 4, n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    base_steps: int = 12000, seed: int = 42,
    n_seq: int = 6000, j_node: int = 0, eval_seed: int = 4242,
    n_tv_sub: int = 192,
    probe_blocks: str = "post_block6,post_block2",
    probe_steps: int = 500, arms: str = "family,floor", tag: str = "",
):
    """Does the MODEL retain the perturbed rule choice, and for how far?

    Mirrors `../synonym_retention/`'s readout exactly -- decode the constituent's rule
    choice from h[e+w], normalised as (acc - chance)/(bayes - chance) so a falling
    accuracy cannot be confused with a rising task difficulty -- with two changes forced
    by this substrate:

      * the perturbed constituent is at the DIFFERING level, so unlike fixed rules its
        realisation is NTP-required at distance (part1: floor TV is exactly 0 in 100% of
        cells past w=32 while family TV persists across three sequence boundaries);
      * only ~8% of constituents belong to a differing feature, so the readout is
        reported BOTH pooled and restricted to `live` cells (those whose exact next-token
        TV at the read position is non-zero). Pooled dilutes ~12x and is the wrong
        headline.

    Their lambda=0 fixed-rules baseline is the reference to beat: rule retention +0.858
    (w=0), +0.325 (w=4), +0.079 (w=8), chance by w=16.
    """
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.conditional_revision.synonym_retention.synonym_retention import _probe

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = f"{setting_key(v, s, L, m)}_distinct"
    R, dl, nF, mode = DESIGNS[design]
    d = dl[0]
    span = s ** (L - d)
    close = (j_node + 1) * span - 1
    blks = probe_blocks.split(",")
    w_grid = sorted({w for w in [0, 1, 2, 4, 8, 12, 16, 24, 32, 48,
                                 T - close, T - close + 16, 2 * T - close]
                     if 0 <= close + w <= G - 2})

    fam_full, fam_meta = make_family(v, s, L, m, R=R, differ_levels=dl,
                                    n_differ_features=nF, seed=family_seed, mode=mode)
    fams = {"family": fam_full,
            "floor": make_family(v, s, L, m, R=R, differ_levels=[],
                                 n_differ_features=nF, seed=family_seed,
                                 mode=mode)[0]}
    diff_feats = set(fam_meta["differ_features"][d])
    print(f"differing features at rules[{d}]: {sorted(diff_feats)} of {v}", flush=True)
    out = {"config": {"design": design, "d": d, "j_node": j_node, "span": span,
                      "close": close, "w_grid": w_grid, "k_seqs": k_seqs,
                      "n_seq": n_seq, "tag": tag},
           "lam0_reference": {"w0": 0.858, "w4": 0.325, "w8": 0.079, "w16": 0.011},
           "arms": {}}

    for arm in arms.split(","):
        ck = (f"{DATA_DIR}/{key}/rule_family/{design}_K8_{arm}_"
              f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
        if not os.path.exists(ck):
            print(f"  MISSING {ck} -- skipping {arm}", flush=True)
            continue
        model = GPT(v, 8 * T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        model.eval()
        for pm in model.parameters():
            pm.requires_grad_(False)

        family = fams[arm]
        rng = np.random.default_rng(eval_seed)
        rule_ids = rng.integers(0, len(family), size=n_seq)
        base = np.empty((n_seq, k_seqs, T), dtype=np.int64)
        pert = np.empty((n_seq, k_seqs, T), dtype=np.int64)
        y_rule = np.zeros(n_seq, dtype=np.int64)
        for r in range(len(family)):
            sel = np.where(rule_ids == r)[0]
            if not sel.size:
                continue
            need, tries, acc = sel.size, 0, []
            while sum(a[0].shape[0] for a in acc) < need and tries < 60:
                rt, rcx, lvx = _traces(family[r], need * 4,
                                       eval_seed + 100 * r + 7919 * tries + 1)
                # feature at level-d node j_node: rebuild the level-d layer
                cur = rt
                for dd in range(d):
                    nx = np.empty((cur.shape[0], cur.shape[1] * s), dtype=np.int64)
                    for jj in range(cur.shape[1]):
                        nx[:, jj * s:(jj + 1) * s] = (
                            family[r][dd][cur[:, jj], rcx[dd][:, jj]])
                    cur = nx
                keep = (np.isin(cur[:, j_node], list(diff_feats)) if diff_feats
                        else np.ones(cur.shape[0], bool))
                if keep.any():
                    acc.append((rt[keep], [x[keep] for x in rcx], lvx[keep]))
                tries += 1
            root = np.concatenate([a[0] for a in acc])[:need]
            rc = [np.concatenate([a[1][q] for a in acc])[:need] for q in range(L)]
            lv = np.concatenate([a[2] for a in acc])[:need]
            rc2 = [x.copy() for x in rc]
            rc2[d][:, j_node] = (rc[d][:, j_node]
                                 + rng.integers(1, m, size=sel.size)) % m
            base[sel, 0], pert[sel, 0] = lv, _realize(family[r], root, rc2, s)
            y_rule[sel] = rc2[d][:, j_node]
            for k in range(1, k_seqs):
                _, _, fl = _traces(family[r], sel.size, eval_seed + 100 * r + 10 * k + 2)
                base[sel, k] = fl; pert[sel, k] = fl

        # liveness is now structural (differing-feature cells by construction); the
        # exact mixture TV is computed on a SUBSAMPLE to verify that, not to filter.
        nsub = min(n_tv_sub, n_seq)
        pb = _mixture_predictive(family, base[:nsub], rule_ids[:nsub])
        pp = _mixture_predictive(family, pert[:nsub], rule_ids[:nsub])
        tv = 0.5 * np.abs(pb - pp).sum(-1)                       # (nsub, G)
        del pb, pp

        X = torch.from_numpy(pert.reshape(n_seq, G)).to(device)
        gsel = torch.tensor([close + w for w in w_grid], device=device)
        acts = {b: torch.empty(n_seq, len(w_grid), n_embd) for b in blks}
        with torch.no_grad():
            for i in range(0, n_seq, 64):
                _, _, inter = model(X[i:i + 64, :-1].contiguous(),
                                    return_intermediates=True)
                for b in blks:
                    acts[b][i:i + 64] = inter[b][:, gsel, :].float().cpu()

        rows = {}
        for wi, w in enumerate(w_grid):
            g = close + w
            cell = {"frac_live_subsample": float((tv[:, g] > 1e-12).mean()),
                    "tv_mean_subsample": float(tv[:, g].mean())}
            for scope, msk in (("live", np.ones(n_seq, bool)),):
                if msk.sum() < 400:
                    continue
                best, bsh = 0.0, 0.5
                for b in blks:
                    a, sh, _ = _probe(acts[b][msk][:, wi, :].numpy(),
                                      y_rule[msk], m, device,
                                      steps=probe_steps, seed=seed)
                    if a > best:
                        best, bsh = a, sh
                cell[scope] = {"acc": best, "shuffled": bsh,
                               "retention_vs_chance": (best - 1.0 / m) / (1.0 - 1.0 / m)}
            rows[str(w)] = cell
            lv_ = cell.get("live", {})
            print(f"  {arm} w={w:>4} g={g:>4} fracTVnz={cell['frac_live_subsample']:.3f}"
                  f"  acc {lv_.get('acc', float('nan')):.4f}"
                  f"  shuf {lv_.get('shuffled', float('nan')):.4f}"
                  f"  retention {lv_.get('retention_vs_chance', float('nan')):+.4f}",
                  flush=True)
        out["arms"][arm] = rows
        del model, acts
        torch.cuda.empty_cache()

    dd = f"{DATA_DIR}/{key}/rule_family"
    os.makedirs(dd, exist_ok=True)
    with open(f"{dd}/rule_retention_p2_{design}{'_' + tag if tag else ''}.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


# ---------------------------------------------------------------------------
# PART 2b -- the same curve with a label that is IDENTIFIABLE, and a Bayes ceiling
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. `part2` (tag `p2c`) decodes `y_rule = rc2[d][j_node]`, the STORAGE
# INDEX of the rule used at the perturbed node. On the floor arm all R rule sets are
# bit-identical, so that index is a bijection with the emitted s-tuple and the probe
# reads 1.0000 at w=0. On the family arm `make_family(mode="pool")` assigns each
# differing feature its m tuples in a random permutation order per rule set, so the
# index carries NO information about the observable: measured over the 64 rule sets,
# P(storage index | emitted tuple) is uniform (0.125-0.359, mean 0.25 = chance). The
# exact Bayes ceiling for the family arm's label is therefore ~chance, and its measured
# accuracies (0.247-0.280) sit exactly on it.
#
# So `p2c`'s family-vs-floor contrast is not a retention measurement: the two arms are
# being asked questions with ceilings of ~0.25 and 1.00. This is NOTES section 10's own
# gotcha -- "rule sets differing only in the storage ORDER of a feature's m rules are
# distributionally identical" -- re-entering through the PROBE TARGET after
# `make_family` had canonicalised it out of the DGP.
#
# THE FIX, two parts, both of which `../synonym_retention/` already prescribes:
#   (1) the label is the EMITTED TUPLE's index in the shared pool -- a rule-set
#       independent object, and the actual "which realisation was used" bit. It is also
#       the thing that is NTP-required at distance, since it (with the parent feature)
#       is what discriminates rule sets.
#   (2) an EXACT BAYES CEILING, so retention is (acc - chance)/(bayes - chance) and a
#       falling accuracy cannot be confused with a rising task difficulty. Their gotcha:
#       "pin both ends of the probe."
#
# The ceiling is exact and cheap: sequences are i.i.d. given r, so a prefix reaching
# into later sequences informs the perturbed node ONLY through the rule posterior,
#     P(tuple | x_<g) = sum_r w_r(g) * P(tuple | x_seq0 prefix, r),
# with w from `mixture_profiles` and P(tuple | ., r) read off the level-d junction-tree
# clique (parent feature x rule index) and pushed through r's own (f, c) -> tuple map.

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=49152)
def part2b(
    design: str = "d2_R64_nF2",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 4, n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    base_steps: int = 36000, ckpt_at: int = 36000, ckpt_phase: str = "",
    seed: int = 42,
    n_seq: int = 6000, j_node: int = 0, eval_seed: int = 4242,
    n_bayes_sub: int = 384,
    probe_blocks: str = "post_block6,post_block2",
    probe_steps: int = 600, arms: str = "family,floor", tag: str = "",
):
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.conditional_revision import oracle as ORC
    from rhm.conditional_revision.synonym_retention.synonym_retention import _probe

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = f"{setting_key(v, s, L, m)}_distinct"
    R, dl, nF, mode = DESIGNS[design]
    d = dl[0]
    span = s ** (L - d)
    close = (j_node + 1) * span - 1
    blks = probe_blocks.split(",")
    w_grid = sorted({w for w in [0, 1, 2, 4, 8, 12, 16, 24, 32, 48,
                                 T - close, T - close + 16, 2 * T - close]
                     if 0 <= close + w <= G - 2})

    fam_full, fam_meta = make_family(v, s, L, m, R=R, differ_levels=dl,
                                     n_differ_features=nF, seed=family_seed, mode=mode)
    fams = {"family": fam_full,
            "floor": make_family(v, s, L, m, R=R, differ_levels=[],
                                 n_differ_features=nF, seed=family_seed, mode=mode)[0]}
    diff_feats = sorted(fam_meta["differ_features"][d])
    print(f"differing features at rules[{d}]: {diff_feats} of {v}", flush=True)

    out = {"config": {"design": design, "d": d, "j_node": j_node, "span": span,
                      "close": close, "w_grid": w_grid, "k_seqs": k_seqs,
                      "n_seq": n_seq, "ckpt_at": ckpt_at,
                      "ckpt_phase": ckpt_phase or "aligned", "tag": tag,
                      "label": "emitted-tuple index in the shared pool"},
           "lam0_reference": {"w0": 0.858, "w4": 0.325, "w8": 0.079, "w16": 0.011},
           "supersedes": "part2 / tag p2c -- storage-index label, ceiling ~chance "
                         "on the family arm",
           "arms": {}}

    for arm in [a for a in arms.split(",") if a]:
        ph = f"{ckpt_phase}_" if ckpt_phase else ""
        ck = (f"{DATA_DIR}/{key}/rule_family/{design}_K8_{ph}{arm}_"
              f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}"
              f"{f'_at{ckpt_at}' if ckpt_at else ''}.pt")
        if not os.path.exists(ck):
            print(f"  MISSING {ck} -- skipping {arm}", flush=True)
            continue
        family = fams[arm]

        # ---- the shared, rule-set-independent class list ----
        tuples = sorted({tuple(int(x) for x in family[r][d][f, c])
                         for r in range(len(family))
                         for f in diff_feats for c in range(m)})
        cls = {t: i for i, t in enumerate(tuples)}
        C = len(tuples)
        # per-rule-set (f, c) -> class map, -1 for tuples outside the class list
        cmap = np.full((len(family), v, m), -1, dtype=np.int64)
        for r in range(len(family)):
            for f in range(v):
                for c in range(m):
                    cmap[r, f, c] = cls.get(tuple(int(x) for x in family[r][d][f, c]), -1)
        print(f"\n  {arm}: {C} distinct emitted tuples at the differing features "
              f"(chance {1 / C:.4f}); old storage-index chance was {1 / m:.4f}",
              flush=True)

        model = GPT(v, 8 * T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        model.eval()
        for pm in model.parameters():
            pm.requires_grad_(False)

        rng = np.random.default_rng(eval_seed)
        rule_ids = rng.integers(0, len(family), size=n_seq)
        pert = np.empty((n_seq, k_seqs, T), dtype=np.int64)
        y_tup = np.zeros(n_seq, dtype=np.int64)
        y_feat = np.zeros(n_seq, dtype=np.int64)
        y_idx = np.zeros(n_seq, dtype=np.int64)          # the OLD label, for continuity
        for r in range(len(family)):
            sel = np.where(rule_ids == r)[0]
            if not sel.size:
                continue
            need, tries, acc = sel.size, 0, []
            while sum(a[0].shape[0] for a in acc) < need and tries < 60:
                rt, rcx, lvx = _traces(family[r], need * 4,
                                       eval_seed + 100 * r + 7919 * tries + 1)
                cur = rt
                for dd in range(d):
                    nx = np.empty((cur.shape[0], cur.shape[1] * s), dtype=np.int64)
                    for jj in range(cur.shape[1]):
                        nx[:, jj * s:(jj + 1) * s] = (
                            family[r][dd][cur[:, jj], rcx[dd][:, jj]])
                    cur = nx
                keep = np.isin(cur[:, j_node], diff_feats)
                if keep.any():
                    acc.append((rt[keep], [x[keep] for x in rcx], lvx[keep],
                                cur[keep, j_node]))
                tries += 1
            root = np.concatenate([a[0] for a in acc])[:need]
            rc = [np.concatenate([a[1][q] for a in acc])[:need] for q in range(L)]
            feat = np.concatenate([a[3] for a in acc])[:need]
            rc2 = [x.copy() for x in rc]
            rc2[d][:, j_node] = (rc[d][:, j_node]
                                 + rng.integers(1, m, size=sel.size)) % m
            pert[sel, 0] = _realize(family[r], root, rc2, s)
            y_idx[sel] = rc2[d][:, j_node]
            y_feat[sel] = feat
            y_tup[sel] = cmap[r, feat, rc2[d][:, j_node]]
            for k in range(1, k_seqs):
                _, _, fl = _traces(family[r], sel.size, eval_seed + 100 * r + 10 * k + 2)
                pert[sel, k] = fl
        assert (y_tup >= 0).all(), "a selected cell emitted a tuple outside the pool"

        # ---- exact Bayes ceiling on a subsample ----
        nb = min(n_bayes_sub, n_seq)
        seq0 = pert[:nb, 0]
        w_of_g = _mixture_weights(family, pert[:nb], chunk=512)      # (nb, G, R)
        pc_r = {}                                                    # plen -> (R, nb, C)
        plens = sorted({min(T, close + w + 1) for w in w_grid})
        for plen in plens:
            arr = np.zeros((len(family), nb, C))
            for r in range(len(family)):
                # `prefix_beliefs` builds cliques for levels range(Dmax), so reaching
                # level d needs Dmax = d + 1.
                _, cliques, _ = ORC.prefix_beliefs(family[r], seq0, plen, d + 1)
                cl = cliques[d][:, j_node, :, :]                     # (nb, v, m)
                for f in range(v):
                    for c in range(m):
                        if cmap[r, f, c] >= 0:
                            arr[r, :, cmap[r, f, c]] += cl[:, f, c]
            pc_r[plen] = arr
            print(f"    bayes: prefix len {plen} done", flush=True)

        # ---- model states ----
        X = torch.from_numpy(pert.reshape(n_seq, G)).to(device)
        gsel = torch.tensor([close + w for w in w_grid], device=device)
        acts = {b: torch.empty(n_seq, len(w_grid), n_embd) for b in blks}
        with torch.no_grad():
            for i in range(0, n_seq, 64):
                _, _, inter = model(X[i:i + 64, :-1].contiguous(),
                                    return_intermediates=True)
                for b in blks:
                    acts[b][i:i + 64] = inter[b][:, gsel, :].float().cpu()

        rows = {}
        for wi, w in enumerate(w_grid):
            g = close + w
            plen = min(T, close + w + 1)
            pmix = np.einsum("nr,rnc->nc", w_of_g[:, g, :], pc_r[plen])
            tot = pmix.sum(axis=1, keepdims=True)
            pmix = pmix / np.clip(tot, 1e-300, None)          # renormalise onto the C classes
            bayes = float(pmix.max(axis=1).mean())
            cell = {"bayes_tuple": bayes, "mass_in_pool": float(tot.mean())}
            for nm, yy, ncls in (("tuple", y_tup, C), ("feat", y_feat, v),
                                 ("storage_idx", y_idx, m)):
                best, bsh = 0.0, 1.0 / ncls
                for b in blks:
                    a, sh, _ = _probe(acts[b][:, wi, :].numpy(), yy, ncls, device,
                                      steps=probe_steps, seed=seed)
                    if a > best:
                        best, bsh = a, sh
                cell[nm] = {"acc": best, "shuffled": bsh,
                            "retention_vs_chance": (best - 1.0 / ncls) / (1 - 1.0 / ncls)}
            ch = 1.0 / C
            cell["tuple"]["retention_vs_bayes"] = (
                (cell["tuple"]["acc"] - ch) / max(bayes - ch, 1e-9))
            rows[str(w)] = cell
            print(f"  {arm} w={w:>4} g={g:>4}  bayes {bayes:.4f}  "
                  f"tupleAcc {cell['tuple']['acc']:.4f} "
                  f"(shuf {cell['tuple']['shuffled']:.4f})  "
                  f"RET {cell['tuple']['retention_vs_bayes']:+.4f}   "
                  f"featAcc {cell['feat']['acc']:.4f}  "
                  f"oldIdxAcc {cell['storage_idx']['acc']:.4f}", flush=True)
        out["arms"][arm] = {"ckpt": os.path.basename(ck), "n_classes": C,
                            "chance": 1.0 / C, "rows": rows}
        del model, acts
        torch.cuda.empty_cache()

    dd = f"{DATA_DIR}/{key}/rule_family"
    os.makedirs(dd, exist_ok=True)
    fn = f"{dd}/rule_retention_p2b_{design}{'_' + tag if tag else ''}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {fn}", flush=True)
    return out


def _mixture_weights(family, wins, chunk=512):
    """P(r | x_{<g}) for every global position g. (n, G, R)."""
    n, K, T = wins.shape
    post = family_predictive(family, wins.reshape(-1, T), chunk=chunk)
    post = post.reshape(len(family), n, K, T, family[0][0].shape[0])
    return mixture_profiles(post, wins, np.zeros(n, dtype=np.int64))["w_pre"]
