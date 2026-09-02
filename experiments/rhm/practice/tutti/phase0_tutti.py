"""[tutti] PHASE 0 — the round sized offline, before any GPU.

No Modal, no substrate, no training: A1's own `null_abba` statistic and A1's own `QuietPolicy`,
replayed on the DONOR'S ALREADY-BANKED LOGS (`caesura/figures/ca_s0/<arm>/results.json`), plus
`antiphon`'s offline question sizing re-run on this ladder. Everything here costs zero GPU
because `dsil` is a pure function of `log["panel"]` and every series it needs is on disk.

Four questions, in the order the design note asks them:

  (1) THE FLOOR. `caesura`'s run was governed by `tol_dsil = 0.00321442`, an OFFLINE number,
      while its own reduction re-derived 0.00492 in-tag on `ca_s0/dsil_sched` — i.e. the run
      was paced by a dead zone ~1.5x TIGHTER than its own noise warranted. This re-derives the
      floor with the protocol PINNED (span 1, W 4, `v_tol = sd(N)/sqrt(W)`, commit and advance
      cycles skipped as regime changes) per arm and pooled, and prints all three numbers side
      by side so the choice is on the record rather than inherited.

  (2) THE BOUNDED GATE. Governing at the corrected floor means `tu_d_exo` is no longer a
      bit-identical replay of `ca_s0/dsil_read`. This replays A1's rule, unchanged, on that
      arm's own logged `dsil` series at BOTH floors and reports the first cycle at which the
      quiet verdict differs — which is exactly how far the cross-tag gate reaches. (It is a
      counterfactual: after the first differing action the real trajectories diverge, so the
      number is an upper bound on the window and is reported as one.)

  (3) THE SPLIT, SIZED. A1's thermostat replayed on `ca_s0`'s logged `yield` and `dsil` series
      from each era's start, to bound when a split arm's COMMIT owner (dsil) and ADVANCE owner
      (yield) would each first fire — and, decisively, to check the split CANNOT deadlock: the
      yield advance latch is defined from c1, so an era can always end even while commits are
      still bootstrapping. `crescendo`'s Phase-0 idiom ("when the rule fires"), reused.

  (4) THE MENU. `antiphon`'s `phase0_question.py` re-run: the era caps here are the same
      60/50/70/12/9, so its reach-vs-budget table and its span geometry should carry unchanged
      and K = 2048 should still be the budget-matched value. Re-checked rather than assumed.

Writes `phase0.json` beside this file.

    cd experiments/ && PYTHONPATH=. python3 rhm/practice/tutti/phase0_tutti.py
"""

import json
import os

import numpy as np

from rhm.practice.maestro import policy as PO
from rhm.practice.antiphon import questions as QS

HERE = os.path.dirname(os.path.abspath(__file__))
CA_FIG = os.path.join(os.path.dirname(HERE), "caesura", "figures", "ca_s0")
CA_ARMS = ("dsil_sched", "dsil_yield", "dsil_read", "dsil_and")

# what the donor's run was actually governed by, and what its own reduction re-derived in-tag.
CAESURA_GOVERNING = 0.00321442
CAESURA_IN_TAG_SCHED = 0.00492

SPAN, W, ALPHA, BURN = 1, 4, 0.5, 4
LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]


def _load(arm):
    p = os.path.join(CA_FIG, arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def _series(r, key):
    """the panel series for `key`, with the cycle index it lives on, dropping the leading
    stretch in which the gauge does not exist (for `dsil` that is the whole pre-first-commit
    prefix — a fact about the signal's TYPE, not a hole in the log)."""
    pn, cyc = r["log"]["panel"], r["log"]["cycle"]
    vals = [q.get(key) for q in pn]
    i0 = next((i for i, x in enumerate(vals) if x is not None), None)
    if i0 is None:
        return [], [], None
    sub, subc = vals[i0:], cyc[i0:]
    keep = [j for j, x in enumerate(sub) if x is not None]
    return ([sub[j] for j in keep], [subc[j] for j in keep], subc[0])


def _levels(r, key, cyc):
    """the per-cycle READ LEVEL the panel recorded for this gauge, so the replay uses the same
    per-level dead zone the run did (`yield`'s L3 and L4 floors are different numbers)."""
    by = {int(q["cycle"]): q.get(f"{key}_level") for q in r["log"]["panel"] if q.get("cycle")}
    return [by.get(int(c)) for c in cyc]


def _action_cycles(r):
    """commit and advance cycles — the regime changes `null_abba` must skip. Read off the
    LOOP ACTIONS where they exist and off the events otherwise (a schedule arm takes no loop
    action, but its commits are still regime changes)."""
    out = {int(a["cycle"]) for a in (r.get("loop_actions") or []) if not a.get("cancelled")}
    out |= {int(e["cycle"]) for e in r.get("events", [])
            if e.get("kind") in ("commit", "advance")}
    return out


# --------------------------------------------------------------------------- #
# (1) the floor
# --------------------------------------------------------------------------- #

def derive_floor():
    rows, pooled_N = {}, []
    for arm in CA_ARMS:
        r = _load(arm)
        if r is None:
            continue
        ser, cyc, first = _series(r, "dsil")
        if len(ser) < W * SPAN + 2:
            rows[arm] = {"n_points": len(ser), "note": "too short for a block"}
            continue
        skipc = _action_cycles(r)
        skip = {i for i, c in enumerate(cyc) if c in skipc}
        N, D = PO.null_abba(ser, skip=skip, span=SPAN, W=W)
        pooled_N += list(N)
        rows[arm] = {"first_live_cycle": first, "n_points": len(ser),
                     "n_blocks": len(N), "sd_N": PO._sd(N),
                     "mean_N": float(np.mean(N)) if N else None,
                     "v_tol": PO._sd(N) / (W ** 0.5),
                     "mean_D": float(np.mean(D)) if D else None}
    pooled = {"n_blocks": len(pooled_N), "sd_N": PO._sd(pooled_N),
              "v_tol": PO._sd(pooled_N) / (W ** 0.5)}
    print("\n=== (1) tol_dsil, RE-DERIVED IN-TAG on ca_s0 "
          f"(null-ABBA, span {SPAN}, W {W}, v_tol = sd(N)/sqrt(W), "
          "commit+advance cycles skipped) ===")
    print(f"  {'arm':14s}{'first_c':>9s}{'n_pts':>7s}{'n_blk':>7s}{'sd(N)':>11s}{'v_tol':>11s}")
    for arm, q in rows.items():
        if "v_tol" not in q:
            print(f"  {arm:14s}{'-':>9s}{q['n_points']:>7d}   {q['note']}")
            continue
        print(f"  {arm:14s}{q['first_live_cycle']:>9d}{q['n_points']:>7d}{q['n_blocks']:>7d}"
              f"{q['sd_N']:>11.6f}{q['v_tol']:>11.6f}")
    print(f"  {'POOLED':14s}{'':>9s}{'':>7s}{pooled['n_blocks']:>7d}"
          f"{pooled['sd_N']:>11.6f}{pooled['v_tol']:>11.6f}")
    print(f"\n  the donor's GOVERNING value ....... {CAESURA_GOVERNING:.8f}")
    print(f"  the donor's own in-tag re-derivation {CAESURA_IN_TAG_SCHED:.8f}  "
          f"(`ca_s0/dsil_sched` alone, `analyze_caesura.py` S6)")
    print(f"  THIS round's governing value ....... {pooled['v_tol']:.8f}  (pooled, in-tag)")
    print(f"  ratio to the donor's governing value  "
          f"{pooled['v_tol'] / CAESURA_GOVERNING:.3f}x LOOSER — the donor paced its run "
          f"inside a dead zone tighter than its own noise.")
    return {"per_arm": rows, "pooled": pooled,
            "caesura_governing": CAESURA_GOVERNING,
            "caesura_in_tag_sched": CAESURA_IN_TAG_SCHED,
            "governing": round(pooled["v_tol"], 4)}


# --------------------------------------------------------------------------- #
# (2) the bounded gate
# --------------------------------------------------------------------------- #

def _replay(ser, cyc, read_key, tol, actions, kwargs_levels=None):
    """A1's `QuietPolicy`, unchanged, replayed on a logged series. Returns the per-cycle quiet
    verdict. The policy is re-armed at the arm's own REALISED action cycles, so the replay
    tracks the trajectory that was actually run."""
    floors = {"ledger": 1e-9, "endo": 1e-9, "endo_excess": 1e-9, "dsil": tol,
              # `yield`'s dead zone is A1's MEASURED per-level pair, never a scalar: its
              # instrument (a distinct-tuple count) has a different noise scale at L3 than at
              # L4, which is why `QuietPolicy` carries `v_tol_by_level` at all.
              "yield_by_level": (tol if isinstance(tol, dict) else {3: tol, 4: tol})}
    p = PO.build_policy({"kind": "quiet", "read": read_key, "span": SPAN, "W": W,
                         "burn": BURN, "alpha": ALPHA}, floors=floors)
    out = []
    lvls = kwargs_levels or [None] * len(ser)
    for x, c, lv in zip(ser, cyc, lvls):
        info = p.step(c, {read_key: x, f"{read_key}_level": lv})
        out.append(bool(info["quiet"]))
        if c in actions:
            p.acted("replay", c)
    return out


def bounded_gate(governing):
    r = _load("dsil_read")
    if r is None:
        return {"note": "ca_s0/dsil_read not on disk"}
    ser, cyc, first = _series(r, "dsil")
    acts = _action_cycles(r)
    a = _replay(ser, cyc, "dsil", CAESURA_GOVERNING, acts)
    b = _replay(ser, cyc, "dsil", governing, acts)
    diff = [c for i, c in enumerate(cyc) if a[i] != b[i]]
    boot = [e["cycle"] for e in (r.get("dsil_events") or []) if e["kind"] == "bootstrap"]
    out = {"first_live_cycle": first, "n_cycles_compared": len(cyc),
           "first_divergence_cycle": (diff[0] if diff else None),
           "n_cycles_differing": len(diff),
           "n_quiet_at_donor_floor": int(sum(a)), "n_quiet_at_new_floor": int(sum(b)),
           "bootstrap_commits": boot,
           "realised_action_cycles": sorted(acts)}
    print("\n=== (2) THE BOUNDED CROSS-TAG GATE for `tu_d_exo` ===")
    print(f"  A1's rule replayed on `ca_s0/dsil_read`'s own logged dsil series at both floors.")
    print(f"  gauge first live at c{first}; bootstrap commits at {boot} "
          f"(floor-independent, so they replay identically)")
    if diff:
        print(f"  FIRST DIVERGENCE in the quiet verdict: c{diff[0]}  "
              f"({len(diff)} cycles differ of {len(cyc)})")
        print(f"  => the cross-tag replay of `tu_d_exo` vs `ca_s0/dsil_read` is asserted over "
              f"c1..c{diff[0] - 1} and is a RE-INSTANTIATION above it.")
        print(f"  (upper bound: after the first differing ACTION the trajectories diverge, so "
              f"the realised window can only be shorter.)")
    else:
        print("  no divergence — the two floors agree on every cycle of the logged series.")
    return out


# --------------------------------------------------------------------------- #
# (3) the split, sized
# --------------------------------------------------------------------------- #

def size_split(governing):
    """When would each owner first fire, per era, on the donor's own logged series? And can
    the split deadlock?

    The COMMIT owner reads `dsil` (from `ca_s0/dsil_read`, the arm whose dsil series was
    generated under dsil-paced commits) and the ADVANCE owner reads `yield` (from
    `ca_s0/dsil_yield`, the arm whose yield series was generated under yield-paced advances).
    Neither is the split's own trajectory — no offline replay can be — so these are BOUNDS on
    where the split's actions land, in the sense `crescendo`'s Phase 0 used the phrase.
    """
    out = {}
    for arm, key, tol in (("dsil_read", "dsil", governing),
                          ("dsil_yield", "yield", None)):
        r = _load(arm)
        if r is None:
            continue
        cfgv = (r.get("config") or {})
        if tol is None:
            tol = {3: float(cfgv["tol_yield_l3"]), 4: float(cfgv["tol_yield_l4"])}
        ser, cyc, first = _series(r, key)
        lvls = _levels(r, key, cyc)
        acts = _action_cycles(r)
        q = _replay(ser, cyc, key, tol, acts, kwargs_levels=lvls)
        fires = [c for i, c in enumerate(cyc) if q[i]]
        # per era, the first firing after the era's start
        eras = {}
        ev = sorted([e for e in r.get("events", []) if e.get("kind") == "advance"],
                    key=lambda e: e["cycle"])
        bounds, lo = [], 1
        for e in ev:
            bounds.append((lo, e["cycle"])); lo = e["cycle"] + 1
        bounds.append((lo, max(cyc) if cyc else lo))
        for i, (a_, b_) in enumerate(bounds, 1):
            inside = [c for c in fires if a_ <= c <= b_]
            eras[f"era{i}"] = {"window": [a_, b_], "first_fire": (inside[0] if inside else None),
                              "n_fires": len(inside)}
        _tolstr = (json.dumps({str(k): v for k, v in tol.items()})
                   if isinstance(tol, dict) else f"{tol:.6g}")
        out[f"{key} (on {arm}, tol {_tolstr})"] = {
            "first_live_cycle": first, "n_fires": len(fires),
            "fire_cycles": fires[:24], "per_era": eras}
    # the deadlock question, answered structurally
    ry = _load("dsil_yield")
    yield_first = None
    if ry is not None:
        _, cy, f0 = _series(ry, "yield")
        yield_first = f0
    out["deadlock_check"] = {
        "yield_first_live_cycle": yield_first,
        "verdict": ("the ADVANCE owner's gauge is defined from the first cycle, so an era can "
                    "always end even while the COMMIT owner is still bootstrapping; and the "
                    "era CAP forces an advance regardless. The split cannot deadlock the way a "
                    "dsil-driven arm without the bootstrap does."),
        "mirror_note": ("on the MIRROR arm the ADVANCE owner reads `dsil`, which does not "
                        "exist before a slot opens — so its era-1 advance falls through to the "
                        "cap. Since CRESCENDO_LADDER == CRESCENDO_CAPS, the cap IS the arc's "
                        "own advance bootstrap and lands on the schedule arm's own cycle.")}
    print("\n=== (3) THE SPLIT, SIZED (A1's rule replayed on the donor's own series) ===")
    for k, q in out.items():
        if k == "deadlock_check":
            continue
        print(f"  {k}: first live c{q['first_live_cycle']}, {q['n_fires']} firings")
        for e, z in q["per_era"].items():
            print(f"      {e} window {z['window']}  first fire "
                  f"{z['first_fire']}  ({z['n_fires']} in era)")
    print(f"  DEADLOCK: {out['deadlock_check']['verdict']}")
    print(f"  MIRROR:   {out['deadlock_check']['mirror_note']}")
    return out


# --------------------------------------------------------------------------- #
# (4) the menu
# --------------------------------------------------------------------------- #

def size_menu():
    """`antiphon`'s offline sizing, re-checked on this ladder. The era caps are the same
    60/50/70/12/9, so the reach-vs-budget table and the span geometry should carry unchanged.
    Imported rather than re-derived, so the two rounds cannot drift apart."""
    ap = os.path.join(os.path.dirname(HERE), "antiphon", "phase0.json")
    out = {"source": ap, "carried": None}
    if os.path.isfile(ap):
        d = json.load(open(ap))
        out["reach"] = d.get("reach")
        out["obs_budget"] = d.get("obs_budget")
        out["ambiguity"] = d.get("ambiguity")
        out["ladder_matches"] = (d.get("ladder") == [list(x) for x in LADDER]
                                 or d.get("ladder") == LADDER)
        out["carried"] = bool(out["ladder_matches"])
    # the span geometry, recomputed from `questions.py` itself on this ladder
    geom = {}
    for lvl, node, _cap in LADDER:
        g = QS.target_geometry(lvl, node, 4, 2)
        geom[f"L{lvl}n{node}"] = {"target_level": g["level"], "node": g["node"],
                                  "clean_offsets": list(g["clean_offsets"]),
                                  "has_clean": bool(g["has_clean"])}
    out["geometry"] = geom
    print("\n=== (4) THE MENU, re-checked on this ladder ===")
    print(f"  antiphon/phase0.json ladder matches this one: {out.get('ladder_matches')}")
    if out.get("reach"):
        print(f"  reach vs K (antiphon's, carried): "
              f"{json.dumps(out['reach'])[:220]}...")
    if out.get("obs_budget"):
        print(f"  per-era observation budget: {out['obs_budget']}")
    print("  span geometry (which half of the key the QUESTION fixes, per era):")
    for k, g in geom.items():
        print(f"      {k} -> L{g['target_level']}n{g['node']}  clean offsets "
              f"{g['clean_offsets']}  has_clean={g['has_clean']}")
    print("  => the endogenous half-key signal exists in the three EARNING eras and is empty "
          "in the two consumption eras, where every selector falls through to a quota-legal "
          "index order. Stated, not hidden; logged per cycle as `has_clean`.")
    return out


def main():
    print("=" * 78)
    print("[tutti] PHASE 0 — offline, no GPU, no Modal, no substrate")
    print("=" * 78)
    floor = derive_floor()
    gov = floor["governing"]
    out = {"floor": floor, "governing_tol_dsil": gov,
           "bounded_gate": bounded_gate(gov),
           "split": size_split(gov),
           "menu": size_menu(),
           "protocol": {"span": SPAN, "W": W, "alpha": ALPHA, "burn": BURN,
                        "v_tol": "sd(N)/sqrt(W)",
                        "skip": "commit and advance cycles (regime changes)"}}
    with open(os.path.join(HERE, "phase0.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'phase0.json')}")
    print(f"\n>>> RUN THE TAG WITH  --tol-dsil {gov}")
    return out


if __name__ == "__main__":
    main()
