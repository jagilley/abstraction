"""Reduction for PHASE 2 of `native`: the composed ports, the ablation battery, the poison twin.

Reuses `ratchet/analyze_ratchet.py` for the standard readouts and `../prop/analyze_prop.py`'s
Port-1 machinery (the ledger, the can't-decompose signature) by retargeting their module-level
volume prefix and figure root — nothing in `ratchet/`, `prop/` or `span/` is modified.

What this adds:

  (0) GATES. Twinning (bit-identical to the untreated twin until the EARLIEST port switches on;
      the native arms have two), and the composed fidelity arm `given_fid` — both ports wired
      and both nailed shut — which must reproduce `given` for the whole run.
  (1) THE ARM x ERA LEDGER for the composed arms, printed against phase 1's single-port numbers
      so composition is readable against each port alone.
  (2) THE ABLATION BATTERY: current-level de per condition per arm (a_full / b_table / b_span /
      c_prims), with each condition's own per-solve cost, and the parity of the slots that were
      forced to serve under `b_table`.
  (3) NEXT-LEVEL CURRENCY under ablation, split into its STRUCTURAL component (entries
      `Miner.build` can assemble over the surviving lower table — 0 under `b_table` by
      construction) and its INFORMATIVE one (distinct level-1 tuples at `mine_support` in the
      chosen trajectories, gated by no table at all).
  (4) THE POISON CONTRAST: `practice_early_prop_k4` vs `practice_early` vs the frozen-table
      baseline, on era-2/3 competence, |T3| and the level-3 audition.
  (5) SPAN diagnostics (parity, open cycles, head share of macro executions) and the PLANT GUARD.

Usage (from experiments/):
    python3 rhm/practice/native/full/analyze_full.py --tag nf_s0 --fetch --figures
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(NATIVE), "ratchet"))
sys.path.insert(0, os.path.join(NATIVE, "prop"))
import analyze_ratchet as AR                                          # noqa: E402
import analyze_prop as AP                                             # noqa: E402

AR.FIG = FIG
AR.REMOTE = "rhm_practice_native_full"
ORDER = ["never_base", "given", "given_fid", "given_prop_k4", "span_true", "given_native",
         "practice_late", "practice_late_prop_k4", "span_mined", "practice_late_native",
         "practice_early", "practice_early_prop_k4"]
AR.ORDER = ORDER
AP.FIG = FIG
AP.TWIN = {"given_fid": "given", "given_prop_k4": "given", "span_true": "given",
           "given_native": "given",
           "practice_late_prop_k4": "practice_late", "span_mined": "practice_late",
           "practice_late_native": "practice_late",
           "practice_early_prop_k4": "practice_early"}

COND = ["a_full", "b_table", "b_span", "c_prims"]
COND_LABEL = {"a_full": "(a) full system", "b_table": "(b) table ablated, head serves all",
              "b_span": "(b') table + span reach gone", "c_prims": "(c) primitives ablated"}

# phase-1 / port-2 reference numbers, for reading composition against each port alone.
PHASE1 = {   # np_s0, per-era e over each era's last 5 cycles
    "given": [0.1937, 0.2281, 0.3458], "given_prop_k4": [0.0839, 0.1047, 0.1557],
    "practice_late": [0.2708, 0.2594, 0.3448],
    "practice_late_prop_k4": [0.2609, 0.1458, 0.1464],
    "never_base": [0.3510, 0.5323, 0.6687],
}


# --------------------------------------------------------------------------- #
# (0) gates
# --------------------------------------------------------------------------- #

def gates(arms, setup):
    warm = int(setup["config"].get("prop_warmup", 4))
    print("\n== (0a) TWINNING GATE — bit-identical to the untreated twin until the EARLIEST "
          "port switches on ==")
    print("   (prop filter switches on at c%d; a span head with slots from cycle 1 switches on "
          "at c1, so a\n    `given`-vocabulary span arm has no pre-window by construction and "
          "its row is expected to be blank)" % (warm + 1))
    print(f"{'arm vs twin':46s} {'first port-on':>13s} {'max|de| pre':>12s} "
          f"{'first diverge':>14s} {'max|de| all':>12s}")
    out = {}
    for arm, tw in AP.TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        r = arms[arm]
        a = np.asarray(r["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        # earliest switch-on: the prop filter, or the cycle the span head's slots first exist
        on = []
        if r.get("prop_k") is not None:
            on.append(warm + 1)
        if r.get("span_mode") and float(r["config"].get("span_lam", 1.0)):
            commits = [ev["cycle"] for ev in r["events"] if ev["kind"] == "commit"]
            on.append(1 if r["config"].get("_") is None and _has_slots_c1(r) else
                      (min(commits) if commits else n + 1))
        first_on = min(on) if on else n + 1
        pre = np.abs(a[:first_on - 1] - b[:first_on - 1])
        d = np.nonzero(np.abs(a - b) > 0)[0]
        out[arm] = {"first_on": int(first_on),
                    "pre": float(pre.max()) if pre.size else float("nan"),
                    "first": int(d[0]) + 1 if len(d) else None,
                    "all": float(np.abs(a - b).max())}
        print(f"{arm + ' - ' + tw:46s} {('c' + str(first_on)):>13s} "
              f"{out[arm]['pre']:12.6f} "
              f"{('c' + str(out[arm]['first'])) if out[arm]['first'] else '(never)':>14s} "
              f"{out[arm]['all']:12.6f}")

    print("\n== (0b) COMPOSED FIDELITY GATE — both ports wired, both nailed shut ==")
    for arm in ("given_fid",):
        if arm not in out:
            continue
        ok = out[arm]["all"] == 0.0
        cfg = arms[arm]["config"]
        print(f"  {arm} vs given (k = n_moves, span_lam={cfg.get('span_lam')}, "
              f"span_tau={cfg.get('span_tau')}): max|de| over the whole run = "
              f"{out[arm]['all']:.8f}  ->  {'PASS' if ok else 'FAIL'}")
    return out


def _has_slots_c1(r):
    sp = r["log"].get("span") or []
    return bool(sp) and bool(sp[0].get("open"))


# --------------------------------------------------------------------------- #
# (1) the composed ledger
# --------------------------------------------------------------------------- #

def ledger(arms, setup, tail=5):
    print(f"\n== (1) THE LEDGER — e (last {tail} cycles of the era), the priced beam's own "
          "cost per solve ==")
    print(f"{'arm':26s} {'era':>3s} {'e':>7s} {'t_era':>10s} {'g/solve':>8s} "
          f"{'mat/solve':>9s} {'k_eff':>6s} {'w':>5s} {'nm':>4s} {'head share':>10s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = AR.era_slices(log)
        t = np.asarray(log["t_cum"], float)
        e = np.asarray(log["e"], float)
        pr = log.get("prop") or []
        bl = log.get("blocks") or []
        out[arm] = {}
        for k, idx in sl.items():
            sel = idx[-tail:]
            mt = [pr[i]["meter"] for i in sel if i < len(pr) and pr[i].get("meter")]
            f = lambda key, d=float("nan"): float(np.mean([c[key] for c in mt])) if mt else d
            hb = [bl[i] for i in sel if i < len(bl)]
            mac = sum(c.get("mat_dp", 0) + c.get("mat_head", 0) for c in hb)
            hd = sum(c.get("mat_head", 0) for c in hb)
            row = {"e": float(e[sel].mean()),
                   "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                   "g_solve": f("g", float(np.asarray(log["g_per_solve"], float)[sel].mean())),
                   "mat_solve": f("mat"), "k_eff": f("k_eff"),
                   "w": f("w", float(np.asarray(log["width"], int)[sel].mean())),
                   "n_moves": int(log["n_moves"][idx[-1]]),
                   "head_share": (hd / mac) if mac else float("nan")}
            out[arm][k] = row
            print(f"{arm:26s} {k:3d} {row['e']:7.4f} {row['t_era']:10.0f} "
                  f"{row['g_solve']:8.1f} {row['mat_solve']:9.1f} {row['k_eff']:6.1f} "
                  f"{row['w']:5.1f} {row['n_moves']:4d} {row['head_share']:10.3f}")

    print("\n-- composition against each port alone (phase-1 `np_s0` / port-2 `sp_s0` numbers "
          "are from THEIR tags; the in-tag enum twin is the anchor) --")
    print(f"{'arm':26s} " + " ".join(f"{'era' + str(k):>26s}" for k in (1, 2, 3)))
    for arm in ORDER:
        if arm not in out:
            continue
        cells = []
        for k in (1, 2, 3):
            if k not in out[arm]:
                cells.append(f"{'-':>26s}"); continue
            mine = out[arm][k]["e"]
            ref = PHASE1.get(arm, [None] * 3)[k - 1]
            cells.append(f"  {mine:.4f}" + (f" (np_s0 {ref:.4f}, {mine - ref:+.4f})"
                                            if ref is not None else " " * 20))
        print(f"{arm:26s} " + " ".join(cells))
    return out


# --------------------------------------------------------------------------- #
# (2)/(3) the ablation battery
# --------------------------------------------------------------------------- #

def ablation(arms, setup):
    print("\n== (2) ABLATION BATTERY — end-of-run probe on the final trained state ==")
    print("   b_table: committed macros stay callable, but nothing may consult the table; the "
          "span head\n   serves EVERY macro call regardless of the parity gate, so a parity "
          "shortfall shows as error.\n")
    out = {}
    for arm, r in arms.items():
        ab = r.get("ablation")
        if not ab:
            continue
        out[arm] = {}
        base = ab.get("a_full")
        print(f"-- {arm} --")
        print(f"   {'condition':34s} {'nm':>3s} {'era':>3s} {'e':>7s} {'de vs (a)':>10s} "
              f"{'g/solve':>8s} {'blk_dp':>7s} {'blk_head':>8s}")
        for c in COND:
            cell = ab.get(c)
            if cell is None:
                print(f"   {COND_LABEL[c]:34s}  (no moves in this condition)")
                continue
            if "skipped" in cell:
                print(f"   {COND_LABEL[c]:34s}  {cell['skipped']}")
                continue
            out[arm][c] = {}
            for j in sorted(cell["eras"], key=int):
                row = cell["eras"][j]
                bref = base["eras"][j]["e"] if base and "eras" in base else float("nan")
                out[arm][c][j] = {"e": row["e"], "de": row["e"] - bref,
                                  "g": row["g_solve"], "mine": row["mine"]}
                print(f"   {COND_LABEL[c] if j == '0' else '':34s} {cell['n_moves']:3d} "
                      f"{int(j) + 1:3d} {row['e']:7.4f} {row['e'] - bref:+10.4f} "
                      f"{row['g_solve']:8.1f} {row['blocks'].get('blk_dp', 0):7.1f} "
                      f"{row['blocks'].get('blk_head', 0):8.1f}")
        par = ab.get("_parity_at_end") or {}
        opn = ab.get("_open_at_end") or {}
        if par:
            print("   parity at end (slot: exact-match vs the DP, * = gate open in the run): "
                  + "  ".join(f"{k}:{'-' if vv is None else f'{vv:.3f}'}"
                              f"{'*' if opn.get(k) else ''}" for k, vv in sorted(par.items())))
        print()
    return out


def currency(arms):
    print("\n== (3) NEXT-LEVEL CURRENCY UNDER ABLATION — structural vs informative ==")
    print("   built      = entries Miner.build can assemble over the surviving lower table.")
    print("   at_support = distinct level-1 tuples seen >= mine_support times in the chosen")
    print("                trajectories. No table gates this; it is the informative half.\n")
    print(f"{'arm':26s} {'condition':12s} {'era':>3s} {'solved':>7s} "
          f"{'T2@sup':>7s} {'T2built':>8s} {'T3@sup':>7s} {'T3built':>8s} {'T3 distinct':>12s}")
    out = {}
    for arm, r in arms.items():
        ab = r.get("ablation")
        if not ab:
            continue
        out[arm] = {}
        for c in COND:
            cell = ab.get(c)
            if not cell or "skipped" in cell:
                continue
            out[arm][c] = {}
            for j in sorted(cell["eras"], key=int):
                row = cell["eras"][j]
                mn = row["mine"]
                m2, m3 = mn.get("2", {}), mn.get("3", {})
                out[arm][c][j] = {"T2_sup": m2.get("at_support"), "T2_built": m2.get("built"),
                                  "T3_sup": m3.get("at_support"), "T3_built": m3.get("built"),
                                  "n_solved": row["n_solved"]}
                print(f"{arm:26s} {c:12s} {int(j) + 1:3d} {row['n_solved']:7d} "
                      f"{m2.get('at_support', 0):7d} {m2.get('built', 0):8d} "
                      f"{m3.get('at_support', 0):7d} {m3.get('built', 0):8d} "
                      f"{m3.get('n_distinct', 0):12d}")
    print("\n-- the informative half, table-ablated minus full (era 2, level 3) --")
    for arm in out:
        a = out[arm].get("a_full", {}).get("1")
        b = out[arm].get("b_table", {}).get("1")
        if a and b:
            print(f"{arm:26s} T3@sup {a['T3_sup']} -> {b['T3_sup']} "
                  f"({b['T3_sup'] - a['T3_sup']:+d})   T3built {a['T3_built']} -> "
                  f"{b['T3_built']} (structural, 0 by construction under b_table)")
    return out


# --------------------------------------------------------------------------- #
# (4) the poison contrast
# --------------------------------------------------------------------------- #

def poison(arms, per, tail=5):
    print("\n== (4) THE POISON CONTRAST — a bad level-2 table, frozen vs natively ROUTED ==")
    rows = ["practice_early", "practice_early_prop_k4"]
    have = [a for a in rows if a in arms]
    if not have:
        print("  (poison arms not in this tag)")
        return {}
    print(f"{'arm':26s} {'era':>3s} {'e':>7s} {'|T3| at sup':>12s} {'T3 built':>9s} "
          f"{'A(L3)':>7s} {'A_true':>7s} {'recall':>7s}")
    out = {}
    for arm in have:
        log = arms[arm]["log"]
        sl = AR.era_slices(log)
        out[arm] = {}
        for k, idx in sl.items():
            lvl = int(log["level"][idx[0]]) + 1
            cells = [log["aud"][i].get(str(lvl), {}) for i in idx[-tail:]]
            f = lambda key: (float(np.mean([c[key] for c in cells if c.get(key) is not None]))
                             if any(c.get(key) is not None for c in cells) else float("nan"))
            mi = [log["miner"][i].get(str(min(lvl, 3)), {}) for i in idx[-tail:]]
            sup = str(arms[arm]["config"]["mine_support"])
            at = float(np.mean([c.get("n_at_support", {}).get(sup, 0) for c in mi])) if mi else 0
            row = {"e": float(np.asarray(log["e"], float)[idx[-tail:]].mean()),
                   "level": lvl, "at_support": at, "built": f("n_entries"),
                   "A": f("cand"), "A_true": f("true"), "recall": f("tab_recall")}
            out[arm][k] = row
            print(f"{arm:26s} {k:3d} {row['e']:7.4f} {row['at_support']:12.1f} "
                  f"{row['built']:9.1f} {row['A']:7.4f} {row['A_true']:7.4f} "
                  f"{row['recall']:7.3f}")
    if len(have) == 2:
        print("\n-- natively routed minus frozen (the SPEC's poison-twin question) --")
        a, b = out[have[1]], out[have[0]]
        for k in sorted(a):
            print(f"   era{k}  de={a[k]['e'] - b[k]['e']:+.4f}  "
                  f"d|T{a[k]['level']}|@sup={a[k]['at_support'] - b[k]['at_support']:+.1f}  "
                  f"dbuilt={a[k]['built'] - b[k]['built']:+.1f}  "
                  f"dA={a[k]['A'] - b[k]['A']:+.4f}")
    print("\n-- commit events on the poison arms --")
    for arm in have:
        for ev in arms[arm]["events"]:
            if ev["kind"] == "commit":
                print(f"   {arm:26s} era{ev['era']} L{ev['level']} c{ev['cycle']} "
                      f"entries={ev['n_entries']} recall={ev['tab_recall']:.3f} "
                      f"prec={ev['tab_precision']} audition={ev['audition']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# (5) span diagnostics + plant guard
# --------------------------------------------------------------------------- #

def span_diag(arms):
    print("\n== (5) SPAN DIAGNOSTICS — parity, gate openness, head share of macro executions ==")
    for arm, r in arms.items():
        if not r.get("span_mode"):
            continue
        sp = r["log"]["span"]
        bl = r["log"]["blocks"]
        keys = sorted({k for c in sp for k in (c.get("parity") or {})})
        n_cycles = len(sp)
        print(f"\n{arm}  (span_lam={r['config'].get('span_lam')}, "
              f"tau={r['config'].get('span_tau')})")
        for k in keys:
            series = [c["parity"].get(k) for c in sp if c.get("parity")]
            vals = [x for x in series if x is not None]
            openc = sum(1 for c in sp if (c.get("open") or {}).get(k))
            first = next((i + 1 for i, c in enumerate(sp) if (c.get("open") or {}).get(k)), None)
            if not vals:
                print(f"   slot {k}: no held-out data")
                continue
            print(f"   slot {k}: parity end={vals[-1]:.3f} max={max(vals):.3f} "
                  f"open {openc}/{n_cycles} cycles, first open "
                  f"{('c' + str(first)) if first else '(never)'}")
        mac = sum(c.get("mat_dp", 0) + c.get("mat_head", 0) for c in bl)
        hd = sum(c.get("mat_head", 0) for c in bl)
        blk_dp = sum(c.get("blk_dp", 0) for c in bl)
        blk_hd = sum(c.get("blk_head", 0) for c in bl)
        print(f"   head served {hd}/{mac} macro executions "
              f"({(hd / mac if mac else 0):.3f}); block-level counterfactual "
              f"blk_dp={blk_dp} blk_head={blk_hd}")
        en = [(p["cycle"], p["plant"]["e_nospan"]) for p in r["log"]["probe"]
              if "e_nospan" in p.get("plant", {})]
        if en:
            e = np.asarray(r["log"]["e"], float)
            d = [x - e[c - 1] for c, x in en]
            print(f"   span-bypassed competence twin: {len(en)} probes, "
                  f"mean e_nospan - e = {np.mean(d):+.4f} (max |d| {np.max(np.abs(d)):.4f})")


def plant_table(arms, tail=3):
    print("\n== PLANT GUARD — parse / infill on clean held-out configurations ==")
    print(f"{'arm':26s} {'parse c1':>9s} {'parse end':>9s} {'d':>8s} "
          f"{'infill c1':>10s} {'infill end':>10s} {'d':>8s} {'read':>7s}")
    rows = {}
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        if not pr:
            continue
        cell = {}
        for k in ("parse_acc", "infill_acc", "read_acc"):
            vals = [p["plant"][k] for p in pr if k in p["plant"]]
            if vals:
                cell[k] = {"first": float(vals[0]), "last": float(np.mean(vals[-tail:])),
                           "series": [float(x) for x in vals],
                           "cycles": [p["cycle"] for p in pr][-len(vals):]}
        rows[arm] = cell
        pa, ia = cell["parse_acc"], cell["infill_acc"]
        print(f"{arm:26s} {pa['first']:9.4f} {pa['last']:9.4f} {pa['last']-pa['first']:+8.4f} "
              f"{ia['first']:10.4f} {ia['last']:10.4f} {ia['last']-ia['first']:+8.4f} "
              f"{cell.get('read_acc', {}).get('last', float('nan')):7.4f}")
    print("\n-- treated minus twin (end-of-run means; handle/'s dparse was -0.094/-0.044) --")
    for arm, tw in AP.TWIN.items():
        if arm not in rows or tw not in rows:
            continue
        a, b = rows[arm], rows[tw]
        print(f"{arm + ' - ' + tw:46s} dparse={a['parse_acc']['last']-b['parse_acc']['last']:+.4f}"
              f"  dinfill={a['infill_acc']['last']-b['infill_acc']['last']:+.4f}")
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

COLOUR = {"given": "tab:green", "given_fid": "black", "given_prop_k4": "tab:orange",
          "span_true": "tab:olive", "given_native": "tab:red",
          "practice_late": "tab:blue", "practice_late_prop_k4": "tab:cyan",
          "span_mined": "steelblue", "practice_late_native": "tab:purple",
          "practice_early": "0.55", "practice_early_prop_k4": "firebrick",
          "never_base": "0.75"}


def figures(tag, setup, arms, per, ab, plant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(FIG, tag)
    os.makedirs(root, exist_ok=True)
    ec = setup["config"]["era_cycles"]
    bounds = [ec * i for i in range(1, len(setup["eras"]))]
    style = {a: ("-" if a in AP.TWIN else "--") for a in arms}

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for arm, r in arms.items():
        ax[0].plot(r["log"]["cycle"], r["log"]["e"], style[arm], lw=1.3,
                   color=COLOUR.get(arm), label=arm)
        ax[1].plot(r["log"]["t_cum"], r["log"]["e"], style[arm], lw=1.3,
                   color=COLOUR.get(arm), label=arm)
    for b_ in bounds:
        ax[0].axvline(b_, color="0.85", lw=0.8, zorder=0)
    ax[0].set_xlabel("cycle"); ax[1].set_xlabel("priced time")
    for a_ in ax:
        a_.set_ylabel("e (1 - success)"); a_.legend(fontsize=6)
    ax[0].set_title("competence per cycle"); ax[1].set_title("against priced time")
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig1_full_competence.png"), dpi=150)
    plt.close(fig)

    # fig2 — the ablation battery, era 3
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    names = [a for a in ORDER if a in ab]
    width = 0.2
    for i, c in enumerate(COND):
        ys = []
        for a in names:
            cell = ab[a].get(c, {}).get("2")
            ys.append(cell["e"] if cell else np.nan)
        ax[0].bar(np.arange(len(names)) + i * width, ys, width, label=COND_LABEL[c])
    ax[0].set_xticks(np.arange(len(names)) + 1.5 * width)
    ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("e, era-3 metering set"); ax[0].legend(fontsize=6)
    ax[0].set_title("ablation battery: competence at the current level")
    for i, c in enumerate(("a_full", "b_table")):
        ys = []
        for a in names:
            cell = ab[a].get(c, {}).get("1")
            ys.append(cell["mine"].get("3", {}).get("at_support", np.nan) if cell else np.nan)
        ax[1].bar(np.arange(len(names)) + i * 0.35, ys, 0.35, label=COND_LABEL[c])
    ax[1].set_xticks(np.arange(len(names)) + 0.17)
    ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("|T3| at mine_support (era 2)"); ax[1].legend(fontsize=6)
    ax[1].set_title("next-level currency, informative half")
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig2_full_ablation.png"), dpi=150)
    plt.close(fig)

    # fig3 — span parity + plant guard
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for arm, r in arms.items():
        if not r.get("span_mode"):
            continue
        sp = r["log"]["span"]
        keys = sorted({k for c in sp for k in (c.get("parity") or {})})
        for k in keys:
            xs, ys = [], []
            for i, c in enumerate(sp):
                pv = (c.get("parity") or {}).get(k)
                if pv is not None:
                    xs.append(r["log"]["cycle"][i]); ys.append(pv)
            if xs:
                ax[0].plot(xs, ys, lw=1.0, alpha=0.8,
                           color=COLOUR.get(arm), label=f"{arm} {k}")
    ax[0].axhline(setup["config"].get("span_tau", 0.95), color="k", ls=":", lw=0.9)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("held-out exact-match parity vs the DP")
    ax[0].set_title("span parity per macro (dotted = tau)"); ax[0].legend(fontsize=5)
    for arm, cell in plant.items():
        if "parse_acc" not in cell:
            continue
        ax[1].plot(cell["parse_acc"]["cycles"], cell["parse_acc"]["series"], lw=1.2,
                   color=COLOUR.get(arm), label=arm)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("parse_acc, clean held-out")
    ax[1].set_title("plant guard"); ax[1].legend(fontsize=6)
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig3_full_span_plant.png"), dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {root}/fig1_full_competence.png, fig2_full_ablation.png, "
          f"fig3_full_span_plant.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nf_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        AR.fetch(a.tag)
    setup, arms = AR.load(a.tag)
    g = gates(arms, setup)
    per = ledger(arms, setup, tail=a.tail)
    ab = ablation(arms, setup)
    cur = currency(arms)
    po = poison(arms, per, tail=a.tail)
    span_diag(arms)
    plant = plant_table(arms)
    try:
        dec = AP.decompose(arms)
    except Exception as exc:                                    # noqa: BLE001
        dec = {}
        print(f"\n(can't-decompose signature unavailable: {exc})")
    os.makedirs(os.path.join(FIG, a.tag), exist_ok=True)
    if a.figures:
        figures(a.tag, setup, arms, per, ab, plant)
    with open(os.path.join(FIG, a.tag, "summary.json"), "w") as fh:
        json.dump({"tag": a.tag, "gates": g, "ledger": per, "ablation": ab,
                   "currency": cur, "poison": po, "decompose": dec}, fh, indent=2)
    print(f"\nsummary -> {os.path.join(FIG, a.tag, 'summary.json')}")


if __name__ == "__main__":
    main()
