"""Gate -1: pick the rule family from the ORACLE, before any GPU time.

Everything here is exact, model-free and CPU-only. The question it answers is whether a
candidate family's rule identification is GRADUAL -- spread over enough context that a
model could show a decaying in-context trajectory -- or whether it saturates in a
handful of tokens, in which case there is nothing for Gate 2's forward model to watch.

Three things get measured per candidate design:

1. CONCENTRATION / BUDGET. The whole slow component is bounded: summed over a window,
   E[rule revision] = I(r ; x_window) <= ln R. So a family cannot supply more than ln R
   nats of rule-revision no matter how it is built, and the design question is only
   how that budget is SPREAD. Reported as the crossing positions of E[P(r*|prefix)],
   the per-sequence share of the budget, and the per-token ICL gap early vs late.

2. SHORTCUT CLOSURE. RHM_META_LEARNING's diagnosis is that dense NTP is L0-dominated
   and the compositional signal is a thin residual, so a family whose rule set can be
   named from bigram counts teaches nothing about composition. `shortcut_posterior`
   runs an explicit n-gram learner (aligned s^k blocks, treated independently) against
   the exact posterior. If the two curves track, the shortcut is open.

3. WHERE THE EVIDENCE LIVES. Per within-sequence position -- hence per hierarchy level
   -- how much of the rule information arrives there. A design whose evidence sits at
   levels the model does not represent (root recovery on this substrate is 0.088) is
   identifiable in principle and invisible in practice.

The `full` pass adds the parse term, the exact identity self-check, and the readout this
whole regime exists to produce: **the fraction of positions whose total revision is
exactly zero, as a function of context depth**. Under one fixed rule set that fraction
is ~49% and static, which is what lets an NTP-optimal model bake the aleatoric/epistemic
split into weights. If it starts near zero and climbs toward the single-rule value as
the rule posterior concentrates, reducibility is state-dependent within the context and
the null cannot live in weights.

    cd experiments
    python3 -m rhm.conditional_revision.rule_family.gate_minus1 --sweep
    python3 -m rhm.conditional_revision.rule_family.gate_minus1 --full --design <name>
"""

import argparse
import json
import os
import time

import numpy as np

from rhm.conditional_revision.rule_family.family import (
    make_family, generate_windows, pairwise_block_divergence, shortcut_posterior,
)
from rhm.conditional_revision.rule_family.oracle_mixture import (
    family_predictive, mixture_profiles, parse_revision_mixture, self_check_mixture,
)

# ---------------------------------------------------------------------------
# candidate designs. `dl` indexes rules[d]: rules[d] expands a level-d feature, and a
# level-D latent is the repo's `d{L-D}`. So dl=[4] is "how a d2 feature expands",
# dl=[5] is the leaf-emitting table (deliberately the leaky control).
# ---------------------------------------------------------------------------

DESIGNS = {
    #  name                      R    dl      nF   mode
    "floor":                    (8,  [],      8,  "pool"),
    "d2_R8_nF8":                (8,  [4],     8,  "pool"),
    "d3_R8_nF8":                (8,  [3],     8,  "pool"),
    "d2d3_R8_nF8":              (8,  [3, 4],  8,  "pool"),
    "d1_R8_nF8_LEAKY":          (8,  [5],     8,  "pool"),
    "d2_R8_nF8_resample":       (8,  [4],     8,  "resample"),
    "d2_R8_nF4":                (8,  [4],     4,  "pool"),
    "d2_R8_nF2":                (8,  [4],     2,  "pool"),
    "d4_R8_nF8":                (8,  [2],     8,  "pool"),
    "d5_R8_nF8":                (8,  [1],     8,  "pool"),
    "d2_R32_nF8":               (32, [4],     8,  "pool"),
    "d2_R32_nF4":               (32, [4],     4,  "pool"),
    "d2_R4_nF8":                (4,  [4],     8,  "pool"),
    "d2_R32_nF2":               (32, [4],     2,  "pool"),
    "d2_R128_nF4":              (128, [4],    4,  "pool"),
    "d2_R64_nF2":               (64, [4],     2,  "pool"),
    "d2d3_R128_nF4":            (128, [3, 4], 4,  "pool"),
}


def build(name, v=16, s=2, L=6, m=4, seed=0):
    R, dl, nF, mode = DESIGNS[name]
    return make_family(v, s, L, m, R=R, differ_levels=dl, n_differ_features=nF,
                       seed=seed, mode=mode)


# ---------------------------------------------------------------------------
# cheap pass -- rule posterior only, no parse machinery
# ---------------------------------------------------------------------------

def cheap_profile(family, K, n_windows, seed=1, chunk=2048, verbose=False):
    R = len(family)
    L = len(family[0])
    v, m, s = family[0][0].shape
    T = s ** L
    seqs, rule_ids, lf = generate_windows(family, n_windows, K, seed=seed)
    post = family_predictive(family, seqs.reshape(-1, T), chunk=chunk, verbose=verbose)
    post = post.reshape(R, n_windows, K, T, v)
    prof = mixture_profiles(post, seqs, rule_ids)
    return prof, seqs, rule_ids, lf, post


def summarise(prof, family, K):
    """Design-level summary numbers from a cheap profile."""
    R, T = prof["R"], prof["T"]
    G = K * T
    wt = prof["w_true"].mean(0)                     # (G,)
    gap = prof["icl_gap"].mean(0)
    rr = prof["rule_rev"].mean(0)
    Hr = prof["H_rule"].mean(0)

    total_info = float(rr.sum())
    cum = np.cumsum(rr)
    def cross(arr, thr):
        idx = np.where(arr >= thr)[0]
        return int(idx[0]) if idx.size else -1
    frac_by_seq = [float(rr[k * T:(k + 1) * T].sum() / max(total_info, 1e-12))
                   for k in range(K)]
    return {
        "R": R, "K": K, "T": T,
        "ln_R": float(np.log(R)),
        "total_rule_info_nats": total_info,
        "budget_used_frac": total_info / float(np.log(R)) if R > 1 else 0.0,
        "H_rule_start": float(Hr[0]), "H_rule_end": float(Hr[-1]),
        "w_true_end": float(wt[-1]),
        "pos_w_true_50": cross(wt, 0.5),
        "pos_w_true_90": cross(wt, 0.9),
        "pos_w_true_99": cross(wt, 0.99),
        "pos_half_info": cross(cum, 0.5 * total_info) if total_info > 0 else -1,
        "pos_90pct_info": cross(cum, 0.9 * total_info) if total_info > 0 else -1,
        "info_frac_by_sequence": frac_by_seq,
        "icl_gap_seq0": float(gap[:T].mean()),
        "icl_gap_last_seq": float(gap[-T:].mean()),
        "icl_gap_first16": float(gap[:16].mean()),
        "icl_gap_last16": float(gap[-16:].mean()),
        "mean_nll_mix": float(prof["nll_mix"].mean()),
        "mean_nll_true": float(prof["nll_true"].mean()),
        "profile_w_true": wt.tolist(),
        "profile_icl_gap": gap.tolist(),
        "profile_rule_rev": rr.tolist(),
        "profile_H_rule": Hr.tolist(),
    }


def _top_level(T, L, s):
    """position -> highest hierarchy node completed there. Gate 0's convention exactly
    (level 0 = root, level L = leaf-adjacent)."""
    return np.array([min(l for l in range(L + 1) if (p + 1) % (s ** (L - l)) == 0)
                     for p in range(T)])


def evidence_by_level(prof, L, s):
    """Mean REALISED rule revision by the hierarchy level a position completes.

    Confounded on purpose-of-record: within a sequence, position is also context depth
    (you cannot observe position 15 without having seen 0..14), and rule revision decays
    with depth, so a level whose positions sit early reads high for that reason alone.
    `evidence_rate_by_level` is the decoupled companion.
    """
    T = prof["T"]
    rr = prof["rule_rev"].reshape(prof["rule_rev"].shape[0], prof["K"], T).mean(axis=(0, 1))
    top = _top_level(T, L, s)
    return {f"level{l}": float(rr[top == l].mean()) for l in range(L + 1)
            if (top == l).any()}


def evidence_rate_by_level(post, seqs, L, s):
    """Posterior-INDEPENDENT per-position evidence rate, by hierarchy level.

    Mean over rule-set pairs of the symmetrised KL between the two rule sets' exact
    predictive distributions at that position. It does not reference the rule posterior
    at all, so it measures how distinguishable the rule sets are AT a position rather
    than how much belief happened to move there -- which is what makes it comparable
    across designs and across context depth.
    """
    R, n, K, T, v = post.shape
    top = _top_level(T, L, s)
    acc = np.zeros(T)
    npair = 0
    for a in range(R):
        for b in range(a + 1, R):
            pa, pb = post[a], post[b]
            la, lb = np.log(np.clip(pa, 1e-300, None)), np.log(np.clip(pb, 1e-300, None))
            d = 0.5 * ((pa - pb) * (la - lb)).sum(-1)      # (n, K, T)
            acc += d.mean(axis=(0, 1))
            npair += 1
    acc /= max(npair, 1)
    return ({f"level{l}": float(acc[top == l].mean()) for l in range(L + 1)
             if (top == l).any()}, acc)


def shortcut_audit(family, seqs, rule_ids, prof, ks=(0, 1, 2)):
    """Can a cheap n-gram learner name the rule set as fast as full BP can?

    Two complementary readouts per block order k (span s^k tokens):

      pairwise block divergence -- mean symmetrised KL between rule sets on the exact
        aligned-block distribution. EXACTLY zero is a sufficient condition for the
        shortcut to be closed at that order: the statistic is identically distributed
        across the family and carries no information at all. Nonzero is not by itself
        damning, which is what the second readout is for.

      naive-Bayes accuracy -- an explicit learner that sees only which aligned block
        appeared where and multiplies the per-block likelihoods as if blocks were
        independent. They are not (blocks inside one sequence share ancestors), so this
        learner is OVERCONFIDENT and its posterior mass is not calibrated; its ARGMAX
        accuracy is still meaningful, and it is the right comparison because an
        overconfident shortcut that still identifies more slowly than exact BP is
        decisively weak.

    Both are compared at matched token budgets against the exact posterior's accuracy.
    """
    n, K, T = seqs.shape
    s = family[0][0].shape[2]
    exact_acc = (prof["w_pre"].argmax(axis=2) == rule_ids[:, None]).mean(axis=0)
    out = {"exact_acc_by_token": exact_acc.tolist()}
    for k in ks:
        span = s ** k
        div = pairwise_block_divergence(family, k)
        finite = np.isfinite(div)
        lp = shortcut_posterior(family, seqs, k)             # (n, n_blocks+1, R)
        acc = (lp.argmax(axis=2) == rule_ids[:, None]).mean(axis=0)
        # block b of the shortcut learner has consumed b*span tokens
        tok = np.arange(lp.shape[1]) * span
        exact_at = np.array([exact_acc[min(t, len(exact_acc) - 1)] for t in tok])
        out[f"k{k}"] = {
            "span_tokens": int(span),
            "mean_pairwise_sym_KL_per_block": float(np.mean(div[finite])) if finite.any()
                                              else float("inf"),
            "max_pairwise_sym_KL_per_block": float(np.max(div[finite])) if finite.any()
                                             else float("inf"),
            "n_blocks_infinite_KL": int((~finite).sum()),
            "shortcut_acc_final": float(acc[-1]),
            "exact_acc_final": float(exact_acc[-1]),
            "shortcut_acc_by_token": acc.tolist(),
            "exact_acc_at_same_tokens": exact_at.tolist(),
            "tokens_consumed": tok.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# full pass -- parse revision, identity, and the de-nulling profile
# ---------------------------------------------------------------------------

def full_profile(family, K, n_windows, seed=2, Ds=None, chunk=256, verbose=True):
    R = len(family)
    L = len(family[0])
    v, m, s = family[0][0].shape
    T = s ** L
    if Ds is None:
        Ds = list(range(L))
    t0 = time.time()
    prof, seqs, rule_ids, lf, _post = cheap_profile(family, K, n_windows, seed=seed,
                                                    verbose=False)
    if verbose:
        print(f"  cheap pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    parse = parse_revision_mixture(family, seqs, rule_ids, lf, prof, Ds=Ds,
                                   chunk=chunk, verbose=verbose)
    if verbose:
        print(f"  parse pass {time.time() - t0:.1f}s", flush=True)
    chk = self_check_mixture(prof, parse)

    out = {"self_check": chk, "per_D": {}}
    ZERO = 1e-12
    for D in Ds:
        bt = parse["B_total"][D]                     # (n, K, T)
        btrue = parse["B_true"][D]
        pr = parse["parse_rev"][D]
        rr = parse["rule_rev"]
        out["per_D"][f"D{D}"] = {
            "d_name": f"d{L - D}",
            "mean_B_total": float(bt.mean()),
            "mean_rule_rev": float(rr.mean()),
            "mean_parse_rev": float(pr.mean()),
            "mean_B_true_single_rule": float(btrue.mean()),
            "rule_share": float(rr.mean() / max(bt.mean(), 1e-12)),
            # the de-nulling readout, by context depth (sequence index in the window)
            "frac_B_total_zero_by_seq": [float((bt[:, k] <= ZERO).mean())
                                         for k in range(K)],
            "frac_B_true_zero_by_seq": [float((btrue[:, k] <= ZERO).mean())
                                        for k in range(K)],
            "mean_B_total_by_seq": [float(bt[:, k].mean()) for k in range(K)],
            "mean_rule_rev_by_seq": [float(rr[:, k].mean()) for k in range(K)],
            "mean_parse_rev_by_seq": [float(pr[:, k].mean()) for k in range(K)],
        }
    return out, prof, parse


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--shortcut", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--design", default="d2_R8_nF8")
    ap.add_argument("--designs", default="")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=120)
    ap.add_argument("--n-windows-full", type=int, default=48)
    ap.add_argument("--v", type=int, default=16)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {"config": vars(args), "sweep": {}, "full": {}}

    if args.sweep:
        names = ([d for d in args.designs.split(",") if d] or list(DESIGNS))
        print("=" * 96)
        print(f"GATE -1 SWEEP   v{args.v} s{args.s} L{args.L} m{args.m}   "
              f"K={args.K} sequences/window ({args.K * args.s ** args.L} tokens)   "
              f"n_windows={args.n_windows}")
        print("=" * 96)
        hdr = (f"  {'design':<22}{'R':>3}{'lnR':>7}{'I(r;x)':>8}{'used':>7}"
               f"{'p50':>6}{'p90':>6}{'p99':>7}{'50%I':>6}{'90%I':>6}"
               f"{'gap0':>8}{'gapT':>8}{'nllmix':>8}")
        print(hdr)
        for name in names:
            fam, meta = build(name, args.v, args.s, args.L, args.m, args.seed)
            t0 = time.time()
            prof, seqs, rule_ids, lf, post = cheap_profile(
                fam, args.K, args.n_windows, seed=args.seed + 1)
            summ = summarise(prof, fam, args.K)
            summ["meta"] = meta
            summ["evidence_by_level"] = evidence_by_level(prof, args.L, args.s)
            rate, rate_pos = evidence_rate_by_level(post, seqs, args.L, args.s)
            summ["evidence_rate_by_level"] = rate
            summ["evidence_rate_by_position"] = rate_pos.tolist()
            if args.shortcut:
                summ["shortcut"] = shortcut_audit(fam, seqs, rule_ids, prof)
            summ["wall_s"] = time.time() - t0
            results["sweep"][name] = summ
            print(f"  {name:<22}{summ['R']:>3}{summ['ln_R']:>7.2f}"
                  f"{summ['total_rule_info_nats']:>8.3f}"
                  f"{100 * summ['budget_used_frac']:>6.0f}%"
                  f"{summ['pos_w_true_50']:>6}{summ['pos_w_true_90']:>6}"
                  f"{summ['pos_w_true_99']:>7}"
                  f"{summ['pos_half_info']:>6}{summ['pos_90pct_info']:>6}"
                  f"{summ['icl_gap_first16']:>8.4f}{summ['icl_gap_last16']:>8.4f}"
                  f"{summ['mean_nll_mix']:>8.4f}", flush=True)
        print("\n  p50/p90/p99 = first global position where E[P(r*|prefix)] crosses "
              "0.5/0.9/0.99 (-1 = never)")
        print("  50%I/90%I   = position by which that fraction of the total rule "
              "information has arrived")
        print("  gap0/gapT   = mean per-token ICL gap (nll_mix - nll_true) over the "
              "first/last 16 positions")

        if args.shortcut:
            print("\n  SHORTCUT AUDIT -- can an n-gram learner name the rule set?")
            print(f"  {'design':<22}{'order':>7}{'span':>6}{'blockKL':>10}{'shortcut acc':>14}{'exact acc':>11}")
            for name in names:
                sc = results["sweep"][name].get("shortcut")
                if not sc:
                    continue
                for kk in (0, 1, 2):
                    row = sc[f"k{kk}"]
                    print(f"  {name if kk == 0 else '':<22}{kk:>7}"
                          f"{row['span_tokens']:>6}"
                          f"{row['mean_pairwise_sym_KL_per_block']:>10.5f}"
                          f"{row['shortcut_acc_final']:>14.3f}"
                          f"{row['exact_acc_final']:>11.3f}")

        print("\n  where the evidence lives (mean rule revision by the hierarchy level "
              "a position completes):")
        lv = sorted({k for s_ in results["sweep"].values()
                     for k in s_["evidence_by_level"]})
        print(f"  {'design':<22}" + "".join(f"{k:>10}" for k in lv))
        for name in names:
            e = results["sweep"][name]["evidence_by_level"]
            print(f"  {name:<22}" + "".join(f"{e.get(k, float('nan')):>10.4f}"
                                            for k in lv))

    if args.full:
        name = args.design
        fam, meta = build(name, args.v, args.s, args.L, args.m, args.seed)
        print("=" * 96)
        print(f"GATE -1 FULL   design={name}   {meta}")
        print("=" * 96, flush=True)
        out, prof, parse = full_profile(fam, args.K, args.n_windows_full,
                                        seed=args.seed + 2)
        out["meta"] = meta
        out["summary"] = summarise(prof, fam, args.K)
        results["full"][name] = out

        chk = out["self_check"]["per_term"]
        print(f"\n  identity self-check (MC, n={args.n_windows_full} windows):")
        r = chk["rule_only"]
        print(f"    rule-only  E[rule_rev] {r['lhs_mean_rule_rev']:.6f}  vs  "
              f"E[H_mix - H_true] {r['rhs_mean_H_tot_mix_minus_H_tot_true']:.6f}   "
              f"abs_err {r['abs_err']:.2e}  {'ok' if r['passed'] else 'FAIL'}")
        for k, row in chk.items():
            if k == "rule_only":
                continue
            print(f"    {k:<10} E[B_total]  {row['mean_B_total']:.5f}  vs  "
                  f"E[H_mix - H_irr] {row['mean_H_tot_mix_minus_H_irr']:.5f}   "
                  f"abs_err {row['abs_err']:.2e}  {'ok' if row['passed'] else 'FAIL'}")

        print(f"\n  revision decomposition per abstraction level:")
        print(f"    {'':<6}{'B_total':>10}{'rule':>9}{'parse':>9}{'rule%':>8}"
              f"{'B(1 rule)':>11}")
        for k, row in out["per_D"].items():
            print(f"    {row['d_name']:<6}{row['mean_B_total']:>10.4f}"
                  f"{row['mean_rule_rev']:>9.4f}{row['mean_parse_rev']:>9.4f}"
                  f"{100 * row['rule_share']:>7.1f}%"
                  f"{row['mean_B_true_single_rule']:>11.4f}")

        print(f"\n  DE-NULLING: fraction of positions with EXACTLY zero total revision, "
              f"by context depth (sequence index in window)")
        print(f"    {'':<6}" + "".join(f"{'seq' + str(k):>9}" for k in range(args.K))
              + f"{'1-rule':>9}")
        for k, row in out["per_D"].items():
            print(f"    {row['d_name']:<6}"
                  + "".join(f"{x:>9.3f}" for x in row["frac_B_total_zero_by_seq"])
                  + f"{np.mean(row['frac_B_true_zero_by_seq']):>9.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.out}")
    return results


if __name__ == "__main__":
    main()
