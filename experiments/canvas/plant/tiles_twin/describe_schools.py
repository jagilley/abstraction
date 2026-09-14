"""`tiles.describe()` for all 36 schools: the DGP's own depth margins, oracle-side.

These are the numbers the cost-to-depth curve is read AGAINST. `describe` measures, per nested
level, how often a seam break is undoable by ONE tile (`repair1` -- small means depth is not
affordable to the primitive action) and how often a greedy raster refill completes at all
(`greedy_ok` -- 1 minus the dead-end rate), plus how much the seam leaves open at level 2
(`completions`, `eff_completions`). `coverage` is the fraction of edge patterns the tileset
realises; 1.0 would mean every break is single-tile repairable, i.e. no depth at all.

Local only, no torch, never on an agent-consumed path.

    cd experiments && python -m canvas.plant.tiles_twin.describe_schools
"""

import json
import os

from canvas import tiles as T
from canvas.plant.tiles_twin.corpus import schools

HERE = os.path.dirname(os.path.abspath(__file__))


def main(n_samples=12, H=8):
    out = {}
    for sc in schools():
        d = T.describe(sc, H=H, n_samples=n_samples, seed=0)
        out[sc.name] = d
        print(f"{sc.name:12s} cov {d['coverage']:.2f} restarts {d['restarts_per_sample']:5.1f} "
              + " ".join(f"L{k}:rep{d[f'L{k}']['repair1']:.2f}/greedy{d[f'L{k}']['greedy_ok']:.2f}"
                         for k in (1, 2, 3))
              + f" trueT2 {d['true_T2']}", flush=True)
    p = os.path.join(HERE, "results", "describe.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
