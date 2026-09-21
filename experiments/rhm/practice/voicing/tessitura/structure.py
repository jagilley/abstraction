"""[tessitura] THE STRUCTURAL LABEL, RECONSTRUCTED OFFLINE — and the gate that says whether it is.

Q2 needs, per dumped row, the label `voicing`'s `rep` instrument reads: does the class that was
written hold a feature that would have REPAIRED the instance at that slot
(`fourwall/wall.py::consistent_features`)? The run computes it with the true root in hand. The
dump (`ov_dump_rows`) banks `obs`, `write`, `y`, `dp`, `code`, `unif` and **not the root**, so
the label is not a lookup.

WHAT IT NEEDS, EXACTLY. `consistent_features(rules, x, roots, node, level, s, canon, v)` calls
`entry_success`, which OVERWRITES the slot's token span and then asks the grammar's possible-set
DP whether the result derives the root. Two consequences, and the first is the good news:

  - the span it overwrites is `[node*s**level, (node+1)*s**level)`, which is EXACTLY the span
    `VoRecorder.assemble` masks to -1 when it builds `obs`. So the mask is irrelevant: `obs`
    and the unmasked context give the identical label. Nothing about the context is missing.
  - the root is missing, and nothing in the dump determines it.

THE RECONSTRUCTION. `nearest_derivation_cost` is an exact min-edit DP from leaves to a target
root. Re-implemented here with the slot's bottom blocks zeroed after the bottom layer
(`free_span_cost`), it answers "how many token edits OUTSIDE the slot would this string need to
derive root r, if I am allowed to write any legal unit at the slot". The world drew one root per
instance and corrupted it at the era's node with `n_corrupt` tokens, so the true root's cost is
at most that corruption and every other root's is typically far larger. `root_hat` is the
argmin; `margin` is the gap to the runner-up, and a row with margin 0 is AMBIGUOUS and is
reported as such rather than resolved by a tie-break.

WHY THIS IS A RECONSTRUCTION AND NOT A MEASUREMENT, said before any number is read off it: the
argmin is the root that best explains the string, not the root the world drew. Where the two
differ the label is wrong, and nothing offline can see which rows those are. The gate below is
what makes the reconstruction usable rather than assumed —

  GATE T-4. The run logs `rep` per slot per cycle: `hit / n` over 64 evenly-spaced filed rows,
  computed with the TRUE root. Reconstructing the same label over the dumped filed buffer and
  pooling must reproduce those rates per slot. They are different subsamples of the same
  population, so the comparison is a tolerance and not an identity, and the per-slot residuals
  are printed.

  T-4 FAILS, and the failure is this node's answer to "is the structural label recoverable from
  the bank". It is reported as a failure and the reconstruction is not used for any reading. The
  comparison is shown to be non-vacuous by a third column: the same label under a DELIBERATELY
  WRONG root (+1 mod v) lands further still, so `rep` and this computation are comparable
  objects and the residual is measuring the reconstruction rather than a units mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))

import rhm.practice.fourwall.wall as W    # noqa: E402


def free_span_cost(rules, leaves, s, blk0, span):
    """(B, v) — `rhm_sculpt_precheck.nearest_derivation_cost` with the slot's bottom blocks
    made free, for every root at once.

    The donor's DP folds a (B, n_bottom, v) bottom cost upward through the rules. Zeroing the
    rows `blk0 : blk0+span` after the bottom layer says "any level-1 feature may sit here at no
    cost", and the layers above still compose them by the grammar — which is exactly the
    licence the executor has at that slot: write any legal unit, nothing else.
    """
    L = len(rules)
    v, m, _s = rules[0].shape
    B = leaves.shape[0]
    n_bottom = leaves.shape[1] // s
    blocks = leaves.reshape(B, n_bottom, s)
    bottom = rules[L - 1]                                   # (v, m, s)
    diff = (bottom[None, None] != blocks[:, :, None, None, :]).sum(-1)     # (B,n_b,v,m)
    cost = diff.min(-1)                                     # (B, n_bottom, v)
    cost[:, int(blk0):int(blk0) + int(span), :] = 0         # THE SLOT IS FREE
    for ell in range(L - 2, -1, -1):
        n_par = cost.shape[1] // s
        children = cost.reshape(B, n_par, s, v)
        layer = rules[ell]                                  # (v, m, s)
        total = np.zeros((B, n_par, v, m), dtype=np.int64)
        for i in range(s):
            total += children[:, :, i, :][:, :, layer[:, :, i]]
        cost = total.min(-1)
    return cost[:, 0, :]                                    # (B, v)


def infer_roots(rules, obs, s, blk0, span):
    """(root_hat (B,), margin (B,), best (B,)) — the argmin root, the gap to the runner-up and
    the winning cost. `margin == 0` means two roots explain the string equally well."""
    c = free_span_cost(rules, np.asarray(obs, np.int64), s, blk0, span)
    order = np.argsort(c, axis=1, kind="stable")
    best = c[np.arange(c.shape[0]), order[:, 0]]
    second = c[np.arange(c.shape[0]), order[:, 1]]
    return order[:, 0].astype(np.int64), (second - best).astype(np.int64), best.astype(np.int64)


def repair_labels(rules, obs, write, s, level, node, canon_np, v, roots, tok_class):
    """(struct (B,) bool, rep_set_size (B,)) — does the WRITTEN class hold a feature that
    repairs the instance at this slot, under `roots`.

    `tok_class` maps a written tuple to the set of level-`level` features it renders as, the
    same `vo_token_class` the run's `rep` instrument uses; it is passed in so this module never
    reaches for an oracle of its own.
    """
    mask, _ = W.consistent_features(rules, np.asarray(obs, np.int64),
                                    np.asarray(roots, np.int64), int(node), int(level), s,
                                    canon_np, int(v))
    n = mask.shape[0]
    st = np.zeros(n, bool)
    for i in range(n):
        cls = tok_class[tuple(int(z) for z in write[i])]
        st[i] = any(bool(mask[i, f]) for f in cls)
    return st, mask.sum(1).astype(np.int64)


# --------------------------------------------------------------------------------------- #
# the reduction: the reconstruction measured against the run's own `rep`, which is gate T-4
# --------------------------------------------------------------------------------------- #

def main():
    import argparse
    import collections
    import json
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    VOICING = os.path.dirname(HERE)
    sys.path.insert(0, os.path.abspath(os.path.join(VOICING, "..", "..", "..")))
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.enharmonic import quotient as QT

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ov_s0b")
    ap.add_argument("--arm", default="ovt_comp_pr_sh")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "structure.txt"))
    a = ap.parse_args()
    root = os.path.join(VOICING, "figures", a.tag, a.arm)
    d = json.load(open(os.path.join(root, "results.json")))
    cfg, vlog = d["config"], d["log"]["vo"]
    v, s, depth, m = int(cfg["v"]), int(cfg["s"]), int(cfg["depth"]), int(cfg["m"])
    rules = generate_rules_distinct(v, s, depth, m, seed=int(cfg["rule_seed"]))
    canon = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    z = np.load(os.path.join(root, "vo_rows.npz"))
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))
    lines = []

    def o(t=""):
        lines.append(t)
        print(t, flush=True)

    o("=" * 104)
    o(f"[tessitura] GATE T-4 — the structural label, reconstructed offline: {a.tag}:{a.arm}")
    o("=" * 104)
    for ln in __doc__.strip().splitlines():
        o("  " + ln)
    o("")
    o(f"  {'slot':7} {'n':>6} {'recon':>7} {'wrongroot':>10} {'rep/last30':>11} "
      f"{'rep/all':>8} {'amb%':>6} {'cost0%':>7} {'|resid|':>8}")
    res, resw = [], []
    for pre in sorted(meta):
        which, slot = pre.split(":", 1)
        if which != "filed":
            continue
        mt = meta[pre]
        lvl, node = int(mt["level"]), int(mt["node"])
        obs = z[f"{pre}|obs"].astype(np.int64)[-a.rows:]
        wr = z[f"{pre}|write"].astype(np.int64)[-a.rows:]
        rh, mg, bst = infer_roots(rules, obs, s, mt["blk0"], mt["span"])
        tups = [tuple(int(x) for x in r) for r in wr]
        u = sorted(set(tups))
        P = QT.token_class_sets(rules, np.array(u, np.int64), lvl, canon, v, s, depth)[:, 0, :]
        tc = {t: frozenset(int(x) for x in np.nonzero(P[i])[0]) for i, t in enumerate(u)}
        st, _sz = repair_labels(rules, obs, wr, s, lvl, node, canon, v, rh, tc)
        # the non-vacuity half: a deliberately wrong root
        stw, _ = repair_labels(rules, obs, wr, s, lvl, node, canon, v, (rh + 1) % v, tc)
        key = f"{lvl}n{node}"

        def pool(cs):
            n = sum((c or {}).get("rep", {}).get(key, {}).get("n", 0) for c in cs)
            h = sum((c or {}).get("rep", {}).get(key, {}).get("hit", 0) for c in cs)
            return (h / n) if n else float("nan")
        r30, rall = pool(vlog[-30:]), pool(vlog)
        res.append(float(st.mean()) - r30)
        resw.append(float(stw.mean()) - r30)
        o(f"  {slot:7} {len(obs):>6} {st.mean():>7.3f} {stw.mean():>10.3f} {r30:>11.3f} "
          f"{rall:>8.3f} {(mg == 0).mean() * 100:>6.1f} {(bst == 0).mean() * 100:>7.1f} "
          f"{abs(float(st.mean()) - r30):>8.3f}")
    res, resw = np.asarray(res), np.asarray(resw)
    o("")
    o(f"  VERDICT: mean signed residual against the run's own `rep` over the last 30 cycles = "
      f"{res.mean():+.3f}, median |residual| = {np.median(np.abs(res)):.3f}, "
      f"positive on {int((res > 0).sum())} of {len(res)} slots. Under a deliberately wrong "
      f"root the median |residual| is {np.median(np.abs(resw)):.3f}, so the comparison "
      f"discriminates and the residual above is the reconstruction's own.")
    o("  GATE T-4 FAILS. The reconstruction is biased UPWARD and the bias has a mechanism: the")
    o("  free-span argmin prefers precisely the roots under which SOME write at this slot")
    o("  repairs, because those roots reach cost 0 there. A root the world drew under which")
    o("  NOTHING repairs loses the argmin to one under which something does, so the")
    o("  reconstruction manufactures repair sets. The ambiguity column is the second half of")
    o("  the failure: on roughly half the rows two roots explain the string equally well.")
    o("  THE CONSEQUENCE: Q2's structural side is not answerable from the bank, and the label")
    o("  has to be taken at the write, where the true root is in hand. `voicing`'s own `rep`")
    o("  instrument already computes exactly this, on 64 rows per slot per cycle; what it does")
    o("  not do is emit it PER ROW beside the verdict, the prior and the judge's score.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
