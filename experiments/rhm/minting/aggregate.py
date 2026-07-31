"""Pool the minting arms across replicate seeds and report the §7 contrasts with error bars.

The single-seed q8_m3 run puts the headline effect (verifier - random at d5) at +0.028,
which is only ~2x Exp 1's stated probe noise of +-0.013, so it cannot be read as real
without replication. Each replicate shares that seed's corpus, warm checkpoint and eval
sets across all its arms, so every contrast is PAIRED within a seed; what varies between
replicates is initialisation and the training / acceptance / probe RNG, which is the noise
the contrasts are actually exposed to.

Usage (from experiments/):
  PYTHONPATH=. python3 rhm/minting/aggregate.py <dir_seedA> <dir_seedB> ...
"""

import json
import pathlib
import sys

LEVELS = ["d4", "d5", "d6"]      # d1-d3 saturate in every regime we can mint in
BASE = "narrow_static"


def _load(run_dir):
    arms = {}
    for p in pathlib.Path(run_dir).glob("*.json"):
        if p.name == "all.json":
            continue
        r = json.loads(p.read_text())
        arms[f"{r['breadth']}_{r['acceptance']}"] = r
    return arms


def _mean_sem(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, (var / n) ** 0.5


def main(dirs):
    runs = [_load(d) for d in dirs]
    runs = [r for r in runs if BASE in r]
    if not runs:
        raise SystemExit("no run dir contained narrow_static")
    names = [k for k in runs[0] if all(k in r for r in runs)]
    n = len(runs)
    print(f"{n} seed(s); arms present in all: {len(names)}")
    print(f"levels: {LEVELS}   paired within seed against {BASE}\n")

    def dget(r, arm, lvl):
        return r[arm]["recovery"]["full"]["mlp_best"][lvl]

    print("=" * 92)
    print(f"PAIRED CONTRAST vs {BASE}  (mean +- sem over seeds)")
    print(f"{'arm':<22}" + "".join(f"{l:>20}" for l in LEVELS))
    order = ["narrow_real_data", "narrow_verifier", "narrow_peer_verifier",
             "narrow_peer", "narrow_mirror", "narrow_random"]
    for arm in [a for a in order if a in names] + [
            a for a in names if a not in order and a != BASE]:
        cells = []
        for lvl in LEVELS:
            mu, se = _mean_sem([dget(r, arm, lvl) - dget(r, BASE, lvl) for r in runs])
            cells.append(f"{mu:>+9.4f} +-{se:<8.4f}")
        print(f"{arm:<22}" + "".join(f"{c:>20}" for c in cells))

    print("\n" + "=" * 92)
    print("THE LOAD-BEARING CONTRASTS  (does the acceptance signal's TYPE matter)")
    pairs = [("narrow_verifier", "narrow_random", "legality filter vs no filter"),
             ("narrow_verifier", "narrow_mirror", "non-mirror vs mirror"),
             ("narrow_peer_verifier", "narrow_verifier", "+location on top of legality"),
             ("narrow_peer", "narrow_random", "location filter vs no filter"),
             ("narrow_real_data", "narrow_verifier", "how far short of real data")]
    for a, b, why in pairs:
        if a not in names or b not in names:
            continue
        print(f"  {a} - {b}   ({why})")
        for lvl in LEVELS:
            deltas = [dget(r, a, lvl) - dget(r, b, lvl) for r in runs]
            mu, se = _mean_sem(deltas)
            sign = sum(1 for d in deltas if d > 0)
            t = (mu / se) if se and se == se and se > 0 else float("nan")
            print(f"      {lvl}: {mu:>+8.4f} +- {se:<8.4f}  t={t:>6.2f}  "
                  f"{sign}/{len(deltas)} seeds positive  {[round(d, 4) for d in deltas]}")

    print("\n" + "=" * 92)
    print("RECOVERY FRACTION: what share of real data's gain each grader buys back")
    for lvl in LEVELS:
        ceil = [dget(r, "narrow_real_data", lvl) - dget(r, BASE, lvl) for r in runs]
        cm, _ = _mean_sem(ceil)
        row = []
        for arm in ("narrow_verifier", "narrow_peer_verifier", "narrow_peer",
                    "narrow_mirror", "narrow_random"):
            if arm not in names:
                continue
            gm, _ = _mean_sem([dget(r, arm, lvl) - dget(r, BASE, lvl) for r in runs])
            row.append(f"{arm.replace('narrow_', '')}={gm / cm * 100:>6.1f}%"
                       if abs(cm) > 1e-9 else f"{arm}=n/a")
        print(f"  {lvl}  ceiling {cm:>+7.4f}   " + "  ".join(row))

    print("\n" + "=" * 92)
    print("POOL-LEVEL SEPARATION (final round, mean over seeds) -- the mechanism check")
    print(f"{'arm':<22}{'violation':>12}{'support_exp':>13}{'pool':>10}")
    for arm in names:
        if arm == BASE:
            continue
        last = [r[arm]["rounds"][-1].get("accepted") for r in runs]
        last = [x for x in last if x]
        if not last:
            continue
        vio, _ = _mean_sem([1 - x["valid_frac"] for x in last])
        sup, _ = _mean_sem([x["root_only_outside_S"] for x in last])
        pool, _ = _mean_sem([r[arm]["final_pool_size"] for r in runs])
        print(f"{arm:<22}{vio:>12.3f}{sup:>13.3f}{pool:>10.0f}")
    print("=" * 92)


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
