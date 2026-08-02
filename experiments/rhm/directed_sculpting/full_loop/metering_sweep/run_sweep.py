"""Launch the metering sweep: one client, one detached app, N parallel `ladder` containers.

Follows `rhm/CLAUDE.md`'s `dgp_sweep` pattern (spawn the primitive, join the handles) rather
than N `modal run --detach` clients, which this node's own gotchas warn evict each other.
The grid comes from `sweep_plan.py` so the launcher and the documented table cannot drift.

Every point differs from every other in ONE thing (design note (c)):
  E1-A  collect_budget + mon_price, at fixed total spend and fixed monitoring quantity
  E1-B  collect_budget only, at price 1.0
  E2    monitoring quantity only, at fixed budget and price

Run from experiments/:
  modal run --detach rhm/directed_sculpting/full_loop/metering_sweep/run_sweep.py::sweep
  modal run --detach rhm/directed_sculpting/full_loop/metering_sweep/run_sweep.py::sweep --which e1a
"""

from rhm.directed_sculpting.full_loop.ladder import app, ladder
from rhm.directed_sculpting.full_loop.metering_sweep.sweep_plan import (
    ARMS, PUB, SEEDS, e1a_points, e1b_points, e1c_points, e2_points, e3_points,
    e3b_points, e4_points, e5_points, e6_points, e7_points, e8_points, FULL_ARMS)

BASE = dict(rounds=12, n_eval=1024, policies=ARMS)


def grid(which="all"):
    """(tag, kwargs) for every run, in launch order."""
    out = []
    if which in ("all", "e1a"):
        for label, r, b, price in e1a_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=price)))
    if which in ("all", "e1b"):
        for label, r, b, price in e1b_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=price)))
    if which in ("all", "e2"):
        for label, r, cfg, m in e2_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=PUB["collect_budget"],
                                 mon_price=1.0, mon_n=cfg["mon_n"], floor_n=cfg["floor_n"],
                                 forecast_n=cfg["forecast_n"])))
    # ---- round 2 ------------------------------------------------------------------------
    if which in ("all2", "e1c"):
        for label, r, b, price in e1c_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=price)))
    if which in ("all2", "e3"):
        for label, kl in e3_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=PUB["collect_budget"],
                                 mon_price=1.0, drift_kl=kl)))
    if which in ("all2", "e3"):
        for label, kl, b in e3b_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=1.0, drift_kl=kl)))
    if which in ("all2", "e4"):
        for label, b, ep, _cut in e4_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=1.0,
                                 fm_epochs=ep)))
    if which in ("all3", "e6"):
        for label, b, ep, _st in e6_points() + e7_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=b, mon_price=1.0,
                                 fm_epochs=ep)))
    if which in ("all3", "e8"):
        for label, ep in e8_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, collect_budget=PUB["collect_budget"],
                                 mon_price=1.0, fm_epochs=ep, policies=FULL_ARMS)))
    if which in ("all2", "e5"):
        for label, t, _bills in e5_points():
            for s in SEEDS:
                out.append((f"met_{label}_s{s}",
                            dict(BASE, seed=s, mon_price=1.0,
                                 charge_own_monitoring=True, total_budget=t)))
    return out


@app.function(timeout=86400, memory=8192)
def sweep(which: str = "all"):
    g = grid(which)
    print(f"metering sweep: spawning {len(g)} ladder runs ({which})")
    for tag, kw in g:
        print(f"  {tag:22s} B={kw.get('collect_budget', 0):>6d} price={kw['mon_price']:.4f} "
              f"mon_n={kw.get('mon_n', PUB['mon_n'])} epochs={kw.get('fm_epochs', 8)} "
              f"kl={kw.get('drift_kl', 0.30)} T={kw.get('total_budget', 0)} seed={kw['seed']}")
    handles = [(tag, ladder.spawn(tag=tag, **kw)) for tag, kw in g]

    done, failed = [], []
    for tag, h in handles:
        try:
            m = h.get()
            pol = m["results"]
            prize = pol["uniform"]["tree_err_mean"] - pol["oracle"]["tree_err_mean"]
            floor = pol["oracle_dup"]["tree_err_mean"] - pol["oracle"]["tree_err_mean"]
            print(f"OK   {tag:22s} ratio={pol['oracle']['meter_ratio']:.4f} "
                  f"uni={pol['uniform']['tree_err_mean']:.4f} "
                  f"ora={pol['oracle']['tree_err_mean']:.4f} "
                  f"vis={pol['visits_only']['tree_err_mean']:.4f} "
                  f"prize={prize:+.4f} floor={floor:+.4f} ({m['elapsed_seconds']:.0f}s)")
            done.append(tag)
        except Exception as e:                                   # noqa: BLE001
            print(f"FAIL {tag:22s} {type(e).__name__}: {e}")
            failed.append(tag)
    print(f"\n{len(done)}/{len(handles)} complete; failed: {failed}")
    return {"done": done, "failed": failed}
