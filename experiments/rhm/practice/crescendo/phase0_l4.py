"""PHASE 0 — size the L4 question OFFLINE, before any GPU.

A3 asks whether the earnable range extends: `max_macro_level` goes 3 -> 4, and the outer loop
gets the chance to earn and cash a level the whole arc has treated as unearnable. The single
fact that decides whether that is answerable at an affordable budget is the **r-squared wall**:

    T4 ⊆ T3 × T3        (`macros.Miner.build`: an entry exists only if BOTH of its s halves
                         are already rows of the operative lower table)

so an L4 commit over a frozen L3 of recall r can hold at most ~r**2 of the level. A1/A2 froze
L3 at recall 0.071-0.125. If the buildable set is empty at those recalls the treatment arm has
nothing to commit and the round is a structural wall, not an experiment — and that is knowable
from logs already on disk.

WHY THIS IS COMPUTABLE OFFLINE, EXACTLY. `_install_identity_miner` extends `Miner.state()`
with `keys_at_support`, logged every cycle: `log["miner"][c][l]` carries levels 2..maxl and
`log["gy"][c]` carries the level-4 G-Y miner. `Miner.build` is pure set arithmetic over those
keys and the lower table's `flat`, so this file re-executes `build` on the logged key streams
and reproduces every table the runs actually held. GATE B-1 asserts exactly that: the
reconstructed committed tables must reproduce the runs' own logged `n_entries`, `recall` and
`precision` for every commit event in the corpus. Nothing here is a model of the substrate; it
is the substrate's own `build`, replayed.

THE ONE ASYMMETRY, STATED. The G-Y miner observes in EVERY era; the committable `miners[4]`
that `max_macro_level=4` creates is gated to `era_level + 1` and so observes only from era 3.
So the G-Y key stream is an UPPER bound on what a real L4 miner holds. This file therefore
reports both: the G-Y-cumulative bound and the era-3-onward arrival bound, and the era caps
are sized against the pessimistic one.

Outputs `phase0.json`. No GPU, no Modal, no substrate.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/crescendo/phase0_l4.py
"""

import json
import os

import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.practice.ratchet import macros as MC

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)

# the offline corpus: the fetched copies of A1's and A2's main runs
CORPUS = [
    ("cd_s0", os.path.join(PRACTICE, "conductor", "figures", "cd_s0")),
    ("ma_s0", os.path.join(PRACTICE, "maestro", "figures", "ma_s0")),
]

V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
SUPPORT = 3                      # cfg["mine_support"], the support keys_at_support is logged at


# --------------------------------------------------------------------------- #
# the substrate's own build, on logged key streams
# --------------------------------------------------------------------------- #

def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def buildable(keys, lower_flats, s=S):
    """`Miner.build`'s ratchet test, as a set operation: a key survives iff every one of its
    `s` halves is a row of the lower table. Returns the surviving keys."""
    if not keys:
        return []
    half = len(keys[0]) // s
    out = []
    for k in keys:
        k = tuple(int(x) for x in k)
        if all(k[i * half:(i + 1) * half] in lower_flats for i in range(s)):
            out.append(k)
    return out


def load_arm(d):
    with open(os.path.join(d, "results.json")) as f:
        return json.load(f)


def key_stream(res, level):
    """{cycle -> sorted list of keys at support} for one level, from the logged miner state.
    Level 4 lives in `log["gy"]` (the G-Y instrument); 2..maxl live in `log["miner"]`."""
    out = {}
    cycles = res["log"]["cycle"]
    src = res["log"]["gy"] if level == 4 else res["log"]["miner"]
    for i, c in enumerate(cycles):
        st = src[i] if level == 4 else (src[i] or {}).get(str(level))
        if not st:
            out[c] = []
            continue
        out[c] = [tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])]
    return out


# --------------------------------------------------------------------------- #
def analyse_arm(tag, arm, res, truth):
    """Everything the round needs to know about one logged trajectory."""
    cycles = res["log"]["cycle"]
    eras = res["log"]["era"]
    k2, k3, k4 = key_stream(res, 2), key_stream(res, 3), key_stream(res, 4)
    t2f, t3f, t4f = flats_of(truth[2]), flats_of(truth[3]), flats_of(truth[4])

    commits = [e for e in res["events"] if e["kind"] == "commit"]
    cc = {int(e["level"]): int(e["cycle"]) for e in commits}

    # ---- reconstruct the committed tables (GATE B-1 checks these against the run's own log)
    rec_gate = []
    frozen = {}
    for lv in (2, 3):
        if lv not in cc:
            continue
        c = cc[lv]
        keys = (k2 if lv == 2 else k3)[c]
        lower = t2f if False else (flats_of(MC.base_table(V)) if lv == 2 else frozen.get(3 - 1))
        lower = flats_of(MC.base_table(V)) if lv == 2 else frozen[2]
        got = buildable(keys, lower)
        frozen[lv] = set(got)
        ev = [e for e in commits if int(e["level"]) == lv][0]
        truth_l = t2f if lv == 2 else t3f
        n_corr = len(frozen[lv] & truth_l)
        rec_gate.append({
            "level": lv, "cycle": c,
            "n_recon": len(frozen[lv]), "n_logged": int(ev["n_entries"]),
            "recall_recon": n_corr / len(truth_l),
            "recall_logged": ev.get("tab_recall"),
            "prec_recon": (n_corr / len(frozen[lv])) if frozen[lv] else None,
            "prec_logged": ev.get("tab_precision"),
        })

    # ---- the live (uncommitted) L3 table at every cycle, over the frozen L2
    #      `operative(2)` is the frozen L2 once committed, so the live L3 is built over it too.
    def live_l3(c):
        low = frozen.get(2)
        if low is None:
            low = set(buildable(k2[c], flats_of(MC.base_table(V))))
        return set(buildable(k3[c], low))

    # ---- the L4 question, per cycle, under four lower tables
    traj = []
    for i, c in enumerate(cycles):
        l3_live = live_l3(c)
        l3_frozen = frozen.get(3, l3_live)
        keys4 = k4[c]
        b_fr = buildable(keys4, l3_frozen)
        b_lv = buildable(keys4, l3_live)
        b_tr = buildable(keys4, t3f)
        traj.append({
            "cycle": c, "era": eras[i],
            "n_l4_at_support": len(keys4),
            "n_l3_live": len(l3_live), "n_l3_frozen": len(l3_frozen),
            "recall_l3_live": len(l3_live & t3f) / len(t3f),
            "recall_l3_frozen": len(l3_frozen & t3f) / len(t3f),
            "buildable_l4_frozen": len(b_fr),
            "buildable_l4_frozen_true": len(set(b_fr) & t4f),
            "buildable_l4_live": len(b_lv),
            "buildable_l4_live_true": len(set(b_lv) & t4f),
            "buildable_l4_trueL3": len(b_tr),
            "buildable_l4_trueL3_true": len(set(b_tr) & t4f),
        })

    # ---- when does each L4 key FIRST reach support? (the era-3-onward arrival bound)
    first_at = {}
    for c in cycles:
        for k in k4[c]:
            if k not in first_at:
                first_at[k] = c
    era_start = {}
    for i, c in enumerate(cycles):
        era_start.setdefault(eras[i], c)

    arrival = []
    for e in sorted(era_start):
        ks = [k for k, c in first_at.items() if eras[cycles.index(c)] == e]
        arrival.append({"era": e, "first_cycle": era_start[e], "n_new_l4_keys": len(ks)})

    return {"tag": tag, "arm": arm,
            "n_cycles": len(cycles), "commits": cc,
            "recon_gate": rec_gate, "traj": traj, "arrival": arrival,
            "first_at": {",".join(map(str, k)): v for k, v in sorted(first_at.items())}}


# --------------------------------------------------------------------------- #
def replay_frontier_rule():
    """WHEN would A1's thermostat fire at the frontier? — the measurement that sizes era 3.

    `../conductor/floors.py::replay_rule`'s pattern: run the LIVE rule, unchanged, over each
    donor arm's own logged L4 at-support series, starting from era-3 start (where the live loop
    resets its latch). This does not predict what the loop will do — an action changes
    everything downstream — but it answers the two questions that size the round: does the
    gauge quiet inside era 3 at all, and does it quiet so EARLY that the cold `miners[4]` would
    still be empty (an empty-table cancellation, which costs the loop a re-arm).
    """
    from rhm.practice.maestro.policy import QuietPolicy
    from rhm.practice.maestro.maestro import MEASURED_FLOORS as MF

    tol = {3: MF["yield_by_level"][3], 4: MF["yield_by_level"][4]}
    print("\n[4] A1's THERMOSTAT replayed on the logged L4 at-support series, from era-3 start")
    print(f"    measured floors: yield@L3 {tol[3]:.6f}, yield@L4 {tol[4]:.6f}")
    print(f"    {'tag/arm':<26} {'era3':>5} {'n':>4}  {'first fire':>10}  {'firings (c_in_era3)'}")
    out = []
    for tag, root in CORPUS:
        if not os.path.isdir(root):
            continue
        for arm in sorted(os.listdir(root)):
            d = os.path.join(root, arm)
            if not os.path.isfile(os.path.join(d, "results.json")):
                continue
            res = load_arm(d)
            pn = [p for p in (res["log"].get("panel") or []) if p["era"] >= 3]
            if not pn:
                continue
            pol = QuietPolicy("yield", v_tol_by_level=tol, span=1, W=4, burn=4, alpha=0.5)
            pol.acted("era_start", 0, why="era3")
            c0, fires = pn[0]["cycle"], []
            for p in pn:
                info = pol.step(p["cycle"], p)
                if info.get("quiet"):
                    fires.append(p["cycle"] - c0 + 1)
                    pol.acted("commit", p["cycle"])
            print(f"    {tag + '/' + arm:<26} c{c0:<4} {len(pn):>4}  "
                  f"{(str(fires[0]) if fires else '-'):>10}  {fires}")
            out.append({"tag": tag, "arm": arm, "era3_start": c0, "n_cycles": len(pn),
                        "fires_c_in_era3": fires})
    print("    Read: the commit lands ~c_in_era3 18 and the next firing (the era advance)")
    print("    ~9-21 cycles later, so era 3's cap needs ~2x that for the cap not to bind.")
    return out


def main():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    sizes = {str(l): {"rows": int(truth[l]["child"].shape[0]),
                      "distinct_flats": len(flats_of(truth[l]))} for l in (2, 3, 4, 5)}
    print("=" * 78)
    print("PHASE 0 — the L4 question, sized offline")
    print("=" * 78)
    print("\n[0] the DGP's own levels (distinct flat tuples = the recall denominator)")
    for l in (2, 3, 4, 5):
        print(f"    L{l}: {sizes[str(l)]['rows']:>7d} rows   "
              f"{sizes[str(l)]['distinct_flats']:>7d} distinct")

    out = {"sizes": sizes, "support": SUPPORT, "arms": []}
    for tag, root in CORPUS:
        if not os.path.isdir(root):
            print(f"    [skip] {root} not found")
            continue
        for arm in sorted(os.listdir(root)):
            d = os.path.join(root, arm)
            if not os.path.isfile(os.path.join(d, "results.json")):
                continue
            res = load_arm(d)
            out["arms"].append(analyse_arm(tag, arm, res, truth))

    # ---- GATE B-1: the reconstruction is the substrate's own build ------------------------
    print("\n[B-1] reconstruction gate — replayed `Miner.build` vs the run's own commit log")
    ok, n = True, 0
    for a in out["arms"]:
        for g in a["recon_gate"]:
            n += 1
            same = (g["n_recon"] == g["n_logged"]
                    and abs((g["recall_recon"] or 0) - (g["recall_logged"] or 0)) < 1e-9
                    and abs((g["prec_recon"] or 0) - (g["prec_logged"] or 0)) < 1e-9)
            ok = ok and same
            if not same:
                print(f"    MISMATCH {a['tag']}/{a['arm']} L{g['level']}: "
                      f"n {g['n_recon']} vs {g['n_logged']}, "
                      f"rec {g['recall_recon']:.4f} vs {g['recall_logged']}, "
                      f"prec {g['prec_recon']} vs {g['prec_logged']}")
    print(f"    {'PASS' if ok else 'FAIL'} — {n} commit events reproduced entry-for-entry")
    out["gate_b1"] = {"pass": bool(ok), "n_events": n}

    # ---- the headline table ---------------------------------------------------------------
    print("\n[1] END OF RUN, per arm: what an L4 commit could have held")
    print(f"    {'tag/arm':<26} {'cyc':>4} {'L3fz':>5} {'r(L3)':>6} {'L4@sup':>7} "
          f"{'bld|fz':>7} {'true':>5} {'bld|live':>9} {'true':>5} {'bld|T3true':>11} {'true':>5}")
    for a in out["arms"]:
        t = a["traj"][-1]
        print(f"    {a['tag']+'/'+a['arm']:<26} {t['cycle']:>4} {t['n_l3_frozen']:>5} "
              f"{t['recall_l3_frozen']:>6.3f} {t['n_l4_at_support']:>7} "
              f"{t['buildable_l4_frozen']:>7} {t['buildable_l4_frozen_true']:>5} "
              f"{t['buildable_l4_live']:>9} {t['buildable_l4_live_true']:>5} "
              f"{t['buildable_l4_trueL3']:>11} {t['buildable_l4_trueL3_true']:>5}")

    # ---- the era-3 arrival bound -----------------------------------------------------------
    print("\n[2] L4 keys FIRST reaching support, by era "
          "(the committable miners[4] only observes from era 3)")
    for a in out["arms"]:
        row = " ".join(f"e{x['era']}:{x['n_new_l4_keys']}" for x in a["arrival"])
        print(f"    {a['tag']+'/'+a['arm']:<26} {row}")

    # ---- the trajectory inside the deep eras ----------------------------------------------
    print("\n[3] buildable-L4 trajectory over the deep eras (frozen L3 | live L3)")
    for a in out["arms"]:
        if a["arm"].startswith("yoked"):
            continue
        pts = [t for t in a["traj"] if t["era"] >= 3]
        if not pts:
            continue
        step = max(1, len(pts) // 8)
        row = " ".join(f"c{t['cycle']}e{t['era']}:{t['buildable_l4_frozen']}/"
                       f"{t['buildable_l4_live']}" for t in pts[::step])
        print(f"    {a['tag']+'/'+a['arm']:<26} {row}")

    try:
        out["frontier_rule_replay"] = replay_frontier_rule()
    except Exception as exc:                       # needs the policy module, not the substrate
        print(f"    [skip] frontier rule replay unavailable: {exc}")

    with open(os.path.join(HERE, "phase0.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0.json')}")
    return out


if __name__ == "__main__":
    main()
