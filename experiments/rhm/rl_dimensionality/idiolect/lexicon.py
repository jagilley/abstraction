"""The drift, read out as a dictionary: which literal word each run picked.

`idiolect_drift.py` reports the drift as bits and as Jensen-Shannon distances
between two models' feature-conditioned synonym distributions. Those are
aggregates over every feature and level. This script un-aggregates them.

The key observation is that at the BOTTOM level of the hierarchy (`L5`, whose
rules are `rules[L-1]` and expand a feature straight into `s` leaf tokens) a
`(feature, rule)` pair is a literal token string. So the `m` synonyms of a
bottom-level feature are `m` words for the same thing, and the committed
`joint_hist` — the exact (feature, rule) counts over model-chosen nodes — is
literally how often each model says each word. No new sampling, no new
training: this is a re-cut of the generations behind the committed JSON.

Reads `idiolect_results_seed43.json` and regenerates the rules locally
(`generate_rules_invertible(v, s, L, m, seed=rule_seed)`, matching the config
block in the JSON), so it needs numpy only — no Modal, no checkpoints.

Reproduction (from experiments/):
  python3 -m rhm.rl_dimensionality.idiolect.lexicon           # the slide's entry
  python3 -m rhm.rl_dimensionality.idiolect.lexicon --m 6     # any m in {2,3,4,6}

Caveats worth carrying with the numbers:
  - The error bar to quote is the `ei_parse_rerun` row (a same-seed retrain,
    independently sampled), not a binomial SE: the 16 free bottom-level nodes
    of one generation share a prefix, so they are not independent draws.
  - `ei_parse_r{1..6}` are the ei02 (same-seed rerun) round checkpoints, so a
    round trajectory ends at the rerun's final value, not ei01's. The two
    differ by ~0.2 points, which is itself a useful noise reading.
"""

import argparse
import json
import os

import numpy as np

from rhm.rhm_data import generate_rules_invertible

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "idiolect_results_seed43.json")

# the arms worth putting beside each other: convention, then the two seeds
ARMS = [
    ("dgp", "corpus"),
    ("pretrained", "pretrained"),
    ("pretrain_only", "+ continued NTP"),
    ("kl_parse", "+ KL anchor"),
    ("ei_parse", "EI seed 42"),
    ("ei_parse_rerun", "same-seed rerun"),
    ("ei_parse_seed43", "EI seed 43"),
]


def conditional(results, m, label, level, mode="sampled"):
    """P(rule | feature) as a (v, m) array, from the committed joint counts."""
    entry = results["by_m"][str(m)]["conditions"][label][mode][f"L{level}"]
    joint = np.asarray(entry["joint_hist"], dtype=np.float64)
    return joint / np.clip(joint.sum(axis=1, keepdims=True), 1e-30, None)


def words(rules, feature, alphabet="abcdefgh"):
    """The literal token strings the m rules of a bottom-level feature emit."""
    bottom = rules[-1]                                   # (v, m, s) -> leaves
    return ["".join(alphabet[t] for t in bottom[feature, r])
            for r in range(bottom.shape[1])]


def most_divergent_feature(results, m, level):
    """The feature whose synonym distribution the two seeds disagree on most."""
    d = (conditional(results, m, "ei_parse", level)
         - conditional(results, m, "ei_parse_seed43", level))
    return int(np.argmax(np.abs(d).max(axis=1))), float(np.abs(d).max())


def entry_table(results, rules, m, feature, level):
    """One dictionary entry: every arm's rate for each of the feature's words."""
    ws = words(rules, feature)
    rows = [(name, [conditional(results, m, lab, level)[feature, r] * 100
                    for r in range(len(ws))])
            for lab, name in ARMS
            if lab in results["by_m"][str(m)]["conditions"]]
    return ws, rows


def trajectory(results, m, feature, rule, level, seed_tag=""):
    """A word's share round by round, starting from the shared pretrained value."""
    suffix = f"_{seed_tag}" if seed_tag else ""
    vals = [conditional(results, m, "pretrained", level)[feature, rule] * 100]
    for r in range(1, 7):
        lab = f"ei_parse{suffix}_r{r}"
        if lab not in results["by_m"][str(m)]["conditions"]:
            break
        vals.append(conditional(results, m, lab, level)[feature, rule] * 100)
    return vals


def vocabulary_spread(results, m, level):
    """How many of the level's v*m words each arm moved, vs the noise floor."""
    pre = conditional(results, m, "pretrained", level)
    out = {}
    for lab, name in [("ei_parse", "seed 42 vs pretrained"),
                      ("ei_parse_seed43", "seed 43 vs pretrained")]:
        d = np.abs(conditional(results, m, lab, level) - pre).ravel() * 100
        out[name] = (int((d > 5).sum()), d.size, float(d.max()))
    d = np.abs(conditional(results, m, "ei_parse", level)
               - conditional(results, m, "ei_parse_rerun", level)).ravel() * 100
    out["seed 42 vs its own rerun"] = (int((d > 5).sum()), d.size, float(d.max()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--feature", type=int, default=-1,
                    help="default: the feature the two seeds disagree on most")
    args = ap.parse_args()

    results = json.load(open(args.results))
    cfg = results["config"]
    v, s, L = cfg["v"], cfg["s"], cfg["L"]
    level = L - 1                                        # the bottom level
    rules = generate_rules_invertible(v, s, L, args.m, seed=cfg["rule_seed"])

    feature, gap = most_divergent_feature(results, args.m, level)
    if args.feature >= 0:
        feature = args.feature
    ws, rows = entry_table(results, rules, args.m, feature, level)

    print(f"m = {args.m}, level L{level} (rules expand straight to tokens), "
          f"feature {feature}")
    print(f"the corpus says each of its {args.m} words {100 / args.m:.1f}% of "
          f"the time, by construction\n")
    head = " ".join(f"{w:>7}" for w in ws)
    print(f"{'':>18} {head}")
    for name, vals in rows:
        print(f"{name:>18} " + " ".join(f"{x:>7.1f}" for x in vals))

    top = int(np.argmax([conditional(results, args.m, 'ei_parse', level)[feature, r]
                         for r in range(len(ws))]))
    print(f"\nseed 42's word is '{ws[top]}', round by round "
          f"(round 0 = the shared checkpoint):")
    print("  seed 42:", " ".join(f"{x:5.1f}" for x in
                                 trajectory(results, args.m, feature, top, level)))
    print("  seed 43:", " ".join(f"{x:5.1f}" for x in
                                 trajectory(results, args.m, feature, top, level, "seed43")))

    print(f"\nacross the whole bottom-level vocabulary ({v * args.m} words):")
    for name, (n_moved, n_tot, mx) in vocabulary_spread(results, args.m, level).items():
        print(f"  {name:>26}: {n_moved:>2}/{n_tot} moved >5 points   (max {mx:.1f})")


if __name__ == "__main__":
    main()
