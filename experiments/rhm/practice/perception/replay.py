"""Q3 -- the offline replay: `conductor`'s own thermostat, driven by the ENDOGENOUS series
instead of the exact one, yoked cycle for cycle to the realised run.

`conductor/floors.py::replay_rule` is the idiom and its scope note is inherited verbatim:
an action changes everything downstream, so a replay cannot predict what a live loop would
have done. What it answers is the yoked question -- given the same run, the same era
timeline and the same measured dead zones, at which cycles would the two reads have
licensed the same actions, and where do they part.

The rule replayed is `run_arm`'s, stated in `conductor/FILES.md` and reproduced here from
its two blocks: (g) a quiet reading COMMITS the active level if there is one left to
commit, (g4) and ADVANCES otherwise; a commit re-arms the loop so one quiet reading can
never fire both; the policy is reset at every era start and after every action.

Deviations from the live loop, stated: the era timeline is the realised one, so a replay
that would have advanced early does not actually shorten its era -- it is recorded as
having left, and takes no further action until the next realised era boundary. Gate R-1
is that replaying the EXACT series reproduces the realised `loop_actions` exactly.
"""

import sys

sys.path.insert(0, __file__.rsplit("/rhm/", 1)[0])

from rhm.practice.conductor.policy import ADVANCE, COMMIT, QuietPolicy


def replay(series, era_by_cycle, read_level_by_cycle, active_by_cycle, floors,
           max_macro_level=3, span=1, W=4, burn=4, alpha=0.5):
    """`series[cycle_index]` is the read in ERROR convention (`-at_support`)."""
    p = QuietPolicy("yield", v_tol_by_level=floors, span=span, W=W, burn=burn, alpha=alpha)
    committed, actions, left = set(), [], False
    prev_era = None
    for i, e in enumerate(era_by_cycle):
        cyc = i + 1
        if e != prev_era:
            p.acted("era_start", cyc, why=f"era{e}")
            prev_era, left = e, False
        val = series[i]
        if val is None:
            continue
        rl = read_level_by_cycle[i]
        p.step(cyc, {"yield": float(val), "yield_level": int(rl)})
        if left or not p.quiet:
            continue
        active = int(active_by_cycle[i])
        if active <= max_macro_level and active not in committed:
            committed.add(active)
            actions.append({"cycle": cyc, "era": e, "kind": COMMIT, "level": active,
                            "why": "quiet", "V": p.V})
            p.acted(COMMIT, cyc)
        else:
            left = True
            actions.append({"cycle": cyc, "era": e, "kind": ADVANCE, "level": None,
                            "why": "quiet", "V": p.V})
            p.acted(ADVANCE, cyc)
    return actions, p


def compare(a, b):
    """Two action traces, matched by (kind, level) in order of occurrence."""
    def keyed(acts):
        out, seen = {}, {}
        for x in acts:
            k = (x["kind"], x["level"])
            seen[k] = seen.get(k, 0) + 1
            out[(k, seen[k])] = x["cycle"]
        return out
    ka, kb = keyed(a), keyed(b)
    rows = []
    for k in sorted(set(ka) | set(kb), key=lambda q: (str(q[0][0]), str(q[0][1]), q[1])):
        ca, cb = ka.get(k), kb.get(k)
        rows.append({"kind": k[0][0], "level": k[0][1], "nth": k[1],
                     "exact_cycle": ca, "endo_cycle": cb,
                     "delta": (None if (ca is None or cb is None) else cb - ca)})
    ca = sorted(x["cycle"] for x in a)
    cb = sorted(x["cycle"] for x in b)
    first_div = None
    for i in range(max(len(ca), len(cb))):
        if i >= len(ca) or i >= len(cb) or ca[i] != cb[i]:
            first_div = (ca[i] if i < len(ca) else None, cb[i] if i < len(cb) else None)
            break
    return {"rows": rows, "n_exact": len(a), "n_endo": len(b),
            "first_divergence": first_div,
            "identical": [x["cycle"] for x in a] == [x["cycle"] for x in b]
                         and [x["kind"] for x in a] == [x["kind"] for x in b]}


def in_series_floors(pol):
    """The dead zone the arc's own null-ABBA method would have measured ON THIS SERIES.

    `QuietPolicy` computes the donor's contrast `N` at every block and never drives it; on
    a fixed-condition series its true value is zero, so its spread is the in-series noise
    floor and `v_tol = sd(N)/2` at W = 4 (`conductor/FILES.md`, the variance relation gate
    P-2). Measured per read level, exactly as `floors.py` measures the exact series'."""
    import math
    by = {}
    for d in pol.trace:
        if d.get("N") is None:
            continue
        by.setdefault(int(d["read_level"]), []).append(float(d["N"]))
    out = {}
    for lv, xs in by.items():
        if len(xs) < 2:
            continue
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
        out[lv] = {"n_blocks": len(xs), "mean_N": mu, "sd_N": sd, "v_tol": sd / 2}
    return out
