"""conductor/floors — THE MEASURED DEAD ZONES, derived offline from logs already on disk.

`teacher_slot/endo_yield/SPEC.md` §4's method, on this substrate: on a FIXED-CONDITION series
the true ABBA contrast is exactly zero, so every fabricated CKKC block is a draw from the
instrument's own noise. `policy.py`'s P-1 gate checks the cancellation and P-2 the variance
relation that converts the contrast's spread into the decision statistic's dead zone
(`sd(D) = sd(N)/sqrt(W)`, i.e. `sd(N)/2` at W=4).

Sources are the arc's own completed runs, fetched under `../assay/figures/` and
`../census/figures/`: no GPU, no Modal, nothing new run. Windows that contain an era boundary or
a commit cycle are dropped — a regime change is not noise.

The two gauges whose series exist on disk are done here, before any GPU is spent:

  ledger  the arm's own metering error on the era's own cell (`log["e"]`)
  yield   distinct level-(l) tuples at support in the arm's own chosen trajectories, per level:
          level 3 from `gauge_hist["3"]`, level 4 from `log["gy"][...]["n_at_support"]`

The third, `endo`, needs forward passes through a live plant, so it has no offline series at
all. Its floor is measured on the smoke run's own SCHEDULE arm — a fixed-condition series by
construction — exactly as `endo_yield` measured `xspan`'s in-tag on `no_wall`, and the main
run's reduction re-derives it in-tag on this run's own `anchor`.

Run from experiments/:
    python3 rhm/practice/conductor/floors.py            # prints the table, writes floors.json
"""

import json
import os

from rhm.practice.conductor.policy import null_abba, _sd

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)

# Fixed-condition series from the arc's completed runs. Every one is a routing-only depth-6 arm
# on the census ladder, which is this round's substrate exactly.
SOURCES = [
    ("as_s0/anchor", "assay/figures/as_s0/anchor/results.json"),
    ("as_s0/complete", "assay/figures/as_s0/complete/results.json"),
    ("as_s0/junk_dose", "assay/figures/as_s0/junk_dose/results.json"),
    ("as_s0/strip", "assay/figures/as_s0/strip/results.json"),
    ("cs_s0/spiral_route", "census/figures/cs_s0/spiral_route/results.json"),
    ("cs_s0/census_extend", "census/figures/cs_s0/census_extend/results.json"),
    ("as_s1/exact_j", "assay/figures/as_s1/exact_j/results.json"),
]


def series_of(d, support="3"):
    """The three panel series in ERROR convention (lower is better). `at_support` counts RISE
    with more distinct tuples observed, so the yield series is negated — the same convention the
    donor states for every readout it passes to a policy."""
    log = d["log"]
    gh = d.get("gauge_hist") or {}
    out = {"ledger": [float(x) for x in log["e"]]}
    if gh.get("3"):
        out["yield_L3"] = [-float(x) for x in gh["3"]]
    if log.get("gy") and log["gy"][0]:
        out["yield_L4"] = [-float(g["n_at_support"][support]) for g in log["gy"]]
    return out


def skip_of(d):
    """Era boundaries and commit cycles, as 0-based indices into the per-cycle series."""
    log = d["log"]
    era = log["era"]
    commits = {int(e["cycle"]) for e in d.get("events", []) if e.get("kind") == "commit"}
    skip = set()
    for i in range(len(era)):
        if i > 0 and era[i] != era[i - 1]:
            skip.add(i)
        if (i + 1) in commits:
            skip.add(i)
    return skip


def derive(span=1, W=4, sources=None, verbose=True):
    pooled = {}
    per_run = {}
    used = []
    for name, rel in (sources or SOURCES):
        p = os.path.join(PRACTICE, rel)
        if not os.path.isfile(p):
            if verbose:
                print(f"  (missing, skipped) {rel}")
            continue
        d = json.load(open(p))
        skip = skip_of(d)
        per_run[name] = {}
        for k, ser in series_of(d).items():
            N, D = null_abba(ser, skip=skip, span=span, W=W)
            if len(N) < 8:
                continue
            pooled.setdefault(k, {"N": [], "D": []})
            pooled[k]["N"].extend(N)
            pooled[k]["D"].extend(D)
            per_run[name][k] = {"n_windows": len(N), "sd_N": _sd(N),
                                "mean_N": sum(N) / len(N), "mean_D": sum(D) / len(D)}
        used.append(name)
    out = {"span": span, "W": W, "sources": used, "per_run": per_run, "pooled": {}}
    for k, v in pooled.items():
        N, D = v["N"], v["D"]
        tol = _sd(N) / (W ** 0.5)
        out["pooled"][k] = {
            "n_windows": len(N), "sd_N": _sd(N), "mean_N": sum(N) / len(N),
            "mean_D": sum(D) / len(D), "sd_D": _sd(D),
            "v_tol": tol, "signal_over_floor": (sum(D) / len(D)) / tol if tol else None,
            "frac_D_above_tol": sum(1 for x in D if x > tol) / len(D)}
    return out


def replay_rule(floors, path="assay/figures/as_s0/anchor/results.json", burn=4, span=1, W=4):
    """PHASE 0, `census/phase0_replay.py`'s idiom: run the RULE over the donor's own gauge
    series, per era, before any GPU. It cannot predict what the loop will do — an action changes
    everything downstream — but it answers the two questions that size the round: does the rule
    ARM at all (does the gauge ever clear its own floor), and does it then QUIET inside an era's
    cap, or hold past it.

    ONE SCOPE NOTE, load-bearing: `as_s0` carries no observation panel, so its level-3 at-support
    series is identically zero through era 1 by construction. This replay is therefore silent
    about era 1 for the yield read — which is exactly the gap the panel exists to close — and the
    level-4 series is shown beside it as the proxy that the instrument behaves at all."""
    from rhm.practice.conductor.policy import QuietPolicy
    d = json.load(open(os.path.join(PRACTICE, path)))
    ser = series_of(d)
    era = d["log"]["era"]
    bounds = {}
    for i, q in enumerate(era):
        bounds.setdefault(q, [i, i])[1] = i
    out = {}
    for key, tol in (("ledger", floors["ledger"]),
                     ("yield_L3", floors["yield_by_level"][3]),
                     ("yield_L4", floors["yield_by_level"][4])):
        if key not in ser:
            continue
        rows = []
        for ee in sorted(bounds):
            lo, hi = bounds[ee]
            p = QuietPolicy(key, v_tol=tol, burn=burn, span=span, W=W)
            first, armed, nq = None, None, 0
            for i in range(lo, hi + 1):
                p.step(i + 1, {key: ser[key][i]})
                if p.moved and armed is None:
                    armed = i + 1
                if p.quiet:
                    nq += 1
                    first = first if first is not None else i + 1
            rows.append({"era": ee, "cycles": hi - lo + 1, "armed_at": armed,
                         "first_quiet_at": first, "n_quiet": nq, "V_end": p.V})
        out[key] = {"v_tol": tol, "rows": rows}
    return out


def from_tag(tag="cd_ef", arm="anchor", span=1, W=4):
    """THE GOVERNING FLOORS, re-derived at FULL CONFIG on a schedule arm of `tag`.

    A schedule arm takes no actions, so its panel is a fixed-condition series by construction —
    what the null-ABBA method wants — and unlike the offline sources it carries the ACTUAL
    instruments this round drives (the observation panel included, so the level-3 at-support
    series is live in era 1 where `as_s0`'s is identically zero). The yield floor is derived
    PER READ LEVEL, on the stretch of cycles where the rule actually reads that level, because
    a distinct-tuple count has a different noise scale at level 3 than at level 4.

    The offline derivation stays in `floors.json` beside this one as the independent pre-GPU
    cross-check: on `ledger`, the one series both can measure, they agree to 0.09%."""
    p = os.path.join(HERE, "figures", tag, arm, "results.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    panel, lg = d["log"]["panel"], d["log"]
    commits = {int(e["cycle"]) for e in d["events"] if e.get("kind") == "commit"}
    skip = {i for i in range(len(lg["era"]))
            if (i > 0 and lg["era"][i] != lg["era"][i - 1]) or (i + 1) in commits}
    out = {"tag": tag, "arm": arm, "n_cycles": len(panel), "span": span, "W": W, "per_read": {}}

    def _one(series, sk):
        N, D = null_abba(series, skip=sk, span=span, W=W)
        tol = _sd(N) / (W ** 0.5)
        return {"n_windows": len(N), "sd_N": _sd(N), "mean_N": sum(N) / len(N),
                "v_tol": tol, "mean_D": sum(D) / len(D),
                "signal_over_floor": (sum(D) / len(D)) / tol,
                "frac_D_above_tol": sum(1 for x in D if x > tol) / len(D)}

    for key in ("ledger", "endo", "endo_cell", "endo_excess"):
        ser = [(q["endo"] - q["endo_cell"]) if key == "endo_excess" else q.get(key)
               for q in panel]
        if any(x is None for x in ser):
            continue
        out["per_read"][key] = _one(ser, skip)
    for lvl in (3, 4):
        idx = [i for i, q in enumerate(panel) if q.get("yield_level") == lvl]
        if not idx:
            continue
        lo, hi = min(idx), max(idx)
        ser = [panel[i]["yield"] for i in range(lo, hi + 1)]
        sk = {i - lo for i in skip if lo <= i <= hi}
        out["per_read"][f"yield_L{lvl}"] = {**_one(ser, sk),
                                            "cycles": [lo + 1, hi + 1]}
    return out


def main():
    print("conductor/floors — the null-ABBA dead zones, from logs already on disk\n")
    sweeps = {}
    for span in (1, 2, 3):
        r = derive(span=span, verbose=(span == 1))
        sweeps[span] = r
        print(f"\n-- span={span}  (block = {4 * span} decision points, window "
              f"{4 * span + 1} reads)")
        print(f"   {'series':10s} {'n_win':>6s} {'sd(N)':>9s} {'mean(N)':>9s} "
              f"{'v_tol':>9s} {'mean(D)':>9s} {'D/floor':>8s} {'frac>tol':>9s}")
        for k, v in sorted(r["pooled"].items()):
            print(f"   {k:10s} {v['n_windows']:6d} {v['sd_N']:9.5f} {v['mean_N']:+9.5f} "
                  f"{v['v_tol']:9.5f} {v['mean_D']:+9.5f} "
                  f"{(v['signal_over_floor'] or 0):8.2f} {v['frac_D_above_tol']:9.2f}")

    # THE SPAN CHOICE, made here and not in the runner: a longer horizon raises every gauge's
    # signal-over-floor (the improvement per span grows while the contrast's spread grows more
    # slowly) but the window it needs is `4*span+1` reads, and the consumption eras of this
    # ladder are 12/9/7 cycles long. span=1 is the only horizon at which the loop has decision
    # points in EVERY era; the sweep above is the record of what that costs.
    chosen = sweeps[1]
    floors = {
        "ledger": chosen["pooled"]["ledger"]["v_tol"],
        "yield_by_level": {3: chosen["pooled"]["yield_L3"]["v_tol"],
                           4: chosen["pooled"]["yield_L4"]["v_tol"]},
        "endo": None,        # measured in-tag on the schedule arm; see the module docstring
    }
    doc = {
        "method": "null-ABBA (endo_yield/SPEC.md §4): on a fixed-condition series the true "
                  "CKKC contrast is zero, so its spread is the instrument's own noise; "
                  "v_tol = sd(N)/sqrt(W) is that noise on the statistic the rule thresholds "
                  "(policy_gate P-1, P-2).",
        "span": 1, "W": 4, "sources": chosen["sources"],
        "windows_pooled": {k: v["n_windows"] for k, v in chosen["pooled"].items()},
        "floors": floors,
        "span_sweep": {str(s): {k: {"v_tol": v["v_tol"],
                                    "signal_over_floor": v["signal_over_floor"]}
                                for k, v in r["pooled"].items()}
                       for s, r in sweeps.items()},
        "endo_note": "no offline series exists (it needs forward passes through a live plant); "
                     "measured on the smoke tag's schedule arm and re-derived in-tag in the "
                     "main run's reduction.",
    }
    doc["phase0_replay"] = replay_rule(floors)
    print("\nPHASE 0 — the rule replayed over the donor's own series, per era "
          "(armed = the gauge cleared its own floor; quiet = it then fell back inside it):")
    for k, v in doc["phase0_replay"].items():
        print(f"   {k} (tol {v['v_tol']:.4f}): " + "  ".join(
            f"e{r['era']}[{r['cycles']}c armed@{r['armed_at']} quiet@{r['first_quiet_at']}]"
            for r in v["rows"]))

    # THE GOVERNING SET. If a full-config schedule arm exists on disk, its in-tag derivation
    # governs the run and the offline one above is demoted to the cross-check — a floor must be
    # measured on the series the rule actually thresholds, at the configuration it will run at.
    doc["floors_offline"] = dict(floors)
    it = from_tag()
    if it:
        doc["in_tag"] = it
        pr = it["per_read"]
        doc["floors"] = floors = {
            "ledger": pr["ledger"]["v_tol"],
            "yield_by_level": {3: pr["yield_L3"]["v_tol"], 4: pr["yield_L4"]["v_tol"]},
            "endo": pr["endo_excess"]["v_tol"],
        }
        doc["governing"] = f"{it['tag']}/{it['arm']} (full config, {it['n_cycles']} cycles)"
        doc["endo_note"] = ("the DRIVEN endo read is the EXCESS form (parent span minus cell "
                            "span). The raw parent-span NLL degrades over the run on this "
                            "substrate and never clears its own floor at any horizon "
                            "(signal/floor -0.14/-0.34/-0.53/-0.63/-0.81 at spans 1/2/3/4/6), "
                            "because the donor's additive constant cancels in its TWO-CONDITION "
                            "block and an absorbing action has no second condition — so the "
                            "cancellation is built into the read instead.")
        print(f"\nGOVERNING (in-tag, {doc['governing']}):")
        for k in ("ledger", "endo", "endo_cell", "endo_excess", "yield_L3", "yield_L4"):
            if k in pr:
                v = pr[k]
                print(f"   {k:12s} v_tol={v['v_tol']:.6f}  sd(N)={v['sd_N']:.5f}  "
                      f"mean(D)={v['mean_D']:+.5f}  D/floor={v['signal_over_floor']:+.2f}  "
                      f"frac>tol={v['frac_D_above_tol']:.2f}")
        o = doc["floors_offline"]
        print(f"   cross-check on `ledger`: in-tag {floors['ledger']:.6f} vs offline "
              f"{o['ledger']:.6f}  ({abs(floors['ledger'] - o['ledger']) / o['ledger'] * 100:.2f}% apart)")

    p = os.path.join(HERE, "floors.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"\nCHOSEN (span=1, W=4):")
    print(f"   ledger        v_tol = {floors['ledger']:.6f}   (error / cycle)")
    print(f"   yield @ L3    v_tol = {floors['yield_by_level'][3]:.6f}   (tuples / cycle)")
    print(f"   yield @ L4    v_tol = {floors['yield_by_level'][4]:.6f}   (tuples / cycle)")
    print(f"   endo          v_tol = measured in-tag (no offline series)")
    print(f"\n-> {p}")
    return doc


if __name__ == "__main__":
    main()
