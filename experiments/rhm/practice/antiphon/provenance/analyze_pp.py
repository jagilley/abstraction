#!/usr/bin/env python3
"""[antiphon P"] Gates and reduction for `pp_s1` — the level-restricted provenance pair.

CPU-only. Reads `figures/pp_s1/`, `figures/pp_s0/`, `figures/ap_s0/` and the `assay` mirrors.

GATES (section 0). Nothing below is readable unless every one passes.
  (a) TWIN GATE, corrected first-CONSUMED-cycle form. Rehearsal FIRES at c71 but the gate
      may pass nothing for several cycles; the arm is then legitimately identical to
      `ap_s0`'s `exact` until the first cycle rehearsal is actually CONSUMED. Assert:
      bit-identical over 13 series through (first consumed - 1), first divergence exactly
      at the first consumed cycle.
  (b) FIRST-CONSUMED-CYCLE PAIR-COUNT MATCH. `n_pairs_used` equal on that cycle only — the
      `woodshed` lesson in the form `pp_s0` sharpened. After it the arms are different
      agents and their counts legitimately diverge; that IS the treatment.
  (c) IN-TAG ONE-BIT. On the first consumed cycle the two arms must be the same agent:
      identical `n_solved`, `n_eligible` and `n_traj_used`.
  (d) SUBSTRATE IDENTITY vs `ap_s0` (and therefore `wd_s1`): eras/refs/true_tables/config
      identical except the two new keys.

    python3 rhm/practice/antiphon/provenance/analyze_pp.py [--figures]
"""

import argparse
import json
import os
import sys

import numpy as np

np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(HERE, "figures")
ASSAY = os.path.join(PRACTICE, "assay", "figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PRACTICE, "woodshed"))
import reduce_trust as RT                                            # noqa: E402

GF = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
      "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
PAIR = ("exact_rehL_s", "exact_rehL_m")
NEW_CFG = {"ap_rec", "reh_gate"}
L = []


def out(s=""):
    L.append(s)
    print(s)


def fmt(x, w=9, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return " " * (w - 1) + "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):>{w}d}"
    return f"{float(x):>{w}.{p}f}"


def load(tagdir, arm):
    p = os.path.join(tagdir, arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def recovered(lg, refs, n_era):
    st, fl = refs["stale"], refs["floor"]
    r = []
    for j in range(n_era):
        idx = era_idx(lg, j)
        if not idx:
            r.append(np.nan)
            continue
        e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
        r.append((st[j] - e) / (st[j] - fl[j]))
    return r


def first_consumed(d):
    for i, q in enumerate(d["log"]["reh"] or []):
        if q and q.get("n_pairs_used"):
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    S1 = os.path.join(FIG, "pp_s1")
    A = {a: load(S1, a) for a in PAIR}
    ref_exact = load(os.path.join(FIG, "ap_s0"), "exact")
    su1 = json.load(open(os.path.join(S1, "setup.json")))
    su0 = json.load(open(os.path.join(FIG, "ap_s0", "setup.json")))

    out("=" * 100)
    out("[antiphon P\"] pp_s1 — the level-restricted provenance pair: GATES")
    out("=" * 100)
    bad = []

    fc = {a: first_consumed(A[a]) for a in PAIR}
    out("")
    out("  (a) TWIN GATE vs ap_s0/exact, corrected first-CONSUMED-cycle form")
    for a in PAIR:
        reh = A[a]["log"]["reh"]
        fire = next((i + 1 for i, q in enumerate(reh) if q), None)
        c = fc[a]
        worst = max(float(np.abs(np.asarray(ref_exact["log"][k][:c - 1], float)
                                 - np.asarray(A[a]["log"][k][:c - 1], float)).max())
                    for k in GF)
        div = next((i + 1 for i in range(116)
                    if any(float(ref_exact["log"][k][i]) != float(A[a]["log"][k][i])
                           for k in GF)), None)
        ok = worst == 0.0 and div == c
        out(f"      {a:<14} fires c{fire}  first CONSUMED c{c}   "
            f"c1-c{c - 1} max|delta| = {worst:.3e}   first divergence c{div}   "
            f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"twin {a}")

    out("")
    out("  (b) FIRST-CONSUMED-CYCLE PAIR-COUNT MATCH  (c) IN-TAG ONE-BIT (same agent there)")
    c0 = fc[PAIR[0]]
    same_c = fc[PAIR[0]] == fc[PAIR[1]]
    out(f"      first consumed cycle equal across the pair = {same_c}"
        f"   -> {'PASS' if same_c else 'FAIL'}")
    if not same_c:
        bad.append("first-consumed cycle differs")
    else:
        r0, r1 = A[PAIR[0]]["log"]["reh"][c0 - 1], A[PAIR[1]]["log"]["reh"][c0 - 1]
        for key, lab in (("n_pairs_used", "pair count"),):
            v0, v1 = r0[key], r1[key]
            ok = v0 == v1
            out(f"      {lab:<22} {v0} vs {v1}   -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                bad.append(lab)
        for key in ("n_solved", "n_eligible", "n_traj_used"):
            v0, v1 = r0["prov"][key], r1["prov"][key]
            ok = v0 == v1
            out(f"      prov.{key:<17} {v0} vs {v1}   -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                bad.append(key)

    out("")
    out("  (d) SUBSTRATE IDENTITY vs ap_s0")
    for k in ("eras", "refs", "plant", "true_tables", "junk_pool"):
        same = json.dumps(su1.get(k), sort_keys=True) == json.dumps(su0.get(k), sort_keys=True)
        out(f"      setup['{k}']  identical = {same}   -> {'PASS' if same else 'FAIL'}")
        if not same:
            bad.append(f"setup[{k}]")
    diff = {k: (su0["config"].get(k), su1["config"].get(k))
            for k in set(su1["config"]) | set(su0["config"])
            if su1["config"].get(k) != su0["config"].get(k) and k not in NEW_CFG}
    out(f"      config drift (excluding {sorted(NEW_CFG)}): {diff or 'none'}"
        f"   -> {'PASS' if not diff else 'FAIL'}")
    if diff:
        bad.append("config drift")

    out("")
    out("=" * 100)
    out(f"  GATE VERDICT: {'ALL PASS' if not bad else 'FAIL — ' + '; '.join(bad)}")
    out("=" * 100)
    if bad:
        open(os.path.join(FIG, "pp_s1_reduction.txt"), "w").write("\n".join(L) + "\n")
        return 1

    # ---------------------------------------------------------------- (1) dose
    out("")
    out("-" * 100)
    out("(1) THE DELIVERED DOSE — realized vs projected, as an apparatus record")
    out("-" * 100)
    out("")
    out(f"{'arm':<14} {'cycles':>7} {'solved':>8} {'eligible':>9} {'no_L3':>7} "
        f"{'passes':>7} {'pairs':>8} {'dose':>7} {'zero-cyc':>9}")
    dose = {}
    for a in PAIR:
        R = [q for q in A[a]["log"]["reh"] if q and q.get("prov")]
        sol = sum(q["prov"]["n_solved"] for q in R)
        el = sum(q["prov"]["n_eligible"] for q in R)
        nl = sum(q["prov"]["n_no_atlevel"] for q in R)
        us = sum(q["n_pairs_used"] for q in R)
        tu = sum(q["prov"]["n_traj_used"] for q in R)
        z = sum(1 for q in R if not q["n_pairs_used"])
        dose[a] = dict(sol=sol, el=el, nl=nl, used=us, traj=tu, zero=z, n=len(R))
        out(f"{a:<14} {len(R):>7d} {sol:>8d} {el:>9d} {nl:>7d} {tu:>7d} {us:>8d} "
            f"{us / (sol * 8):>7.4f} {z:>9d}")
    out("")
    out("  `dose` is pairs delivered / the pairs plain `credit` would have delivered")
    out("  (= n_solved x budget). References: pp_s0 strict gate 0.0491 (2,872 pairs);")
    out("  wd_s0's recorded near-null 1,728 pairs; wd_s1's full credit 56,840.")
    out("")
    d0 = dose[PAIR[0]]
    out(f"  REALIZED {d0['used'] / (d0['sol'] * 8):.4f} vs PROJECTED 0.125 — the projection")
    out("  came from a Poisson(1.528) x independent-prop(0.2976) model fitted to pp_s0's")
    out("  aggregates, which reproduced pp_s0's own logged level-gate rate to 3%. It")
    out("  over-predicts here because the conjunction it assumes is not independent:")
    el_r = d0["el"] / d0["sol"]
    out(f"    measured eligible share (>=1 L3 macro)  {el_r:.4f}"
        f"   (Poisson predicted {1 - np.exp(-1.528):.4f})")
    out(f"    measured pass | eligible                {d0['traj'] / d0['el']:.4f}"
        f"   (independence predicted {0.1249 / 0.783:.4f})")
    out("  So the eligible share is close to prediction and the CONDITIONAL pass rate is")
    out("  the term that misses: within a trajectory, the sources of its L3 applications")
    out("  are POSITIVELY CORRELATED — a trajectory that explores into one L3 macro tends")
    out("  to explore into the others — so 'every L3 application was pi's own ask' is")
    out("  rarer than an independent model predicts. Recorded as an apparatus fact; it is")
    out("  the same class of error as the `wd_s0` sizing miss, caught this time before the")
    out("  round rather than after.")

    # ---------------------------------------------------------------- (2) trust
    out("")
    out("-" * 100)
    out("(2) TRUST — pi's per-level proposal mass and argmax share, against the handles")
    out("-" * 100)
    out("")
    tags = [("pp_s1", S1, PAIR),
            ("ap_s0", os.path.join(FIG, "ap_s0"),
             ("exact_reh", "exact_exp", "exact", "given_c1")),
            ("pp_s0", os.path.join(FIG, "pp_s0"), ("exact_reh_s",))]
    T = {}
    out(f"{'tag':<7} {'arm':<14} {'L2 mass':>9} {'L2 amax':>9} {'L3 mass':>9} {'L3 amax':>9}")
    for tg, root, arms in tags:
        for a in arms:
            p = os.path.join(root, a, "results.json")
            if not os.path.isfile(p):
                continue
            r = RT.load_arm(p)
            T[(tg, a)] = {(ell, k): float(r[k][ell][-1]) for ell in (2, 3)
                          for k in ("mass", "amax") if ell in r[k]}
            g = T[(tg, a)]
            out(f"{tg:<7} {a:<14} {fmt(g.get((2, 'mass')), 9)} {fmt(g.get((2, 'amax')), 9)} "
                f"{fmt(g.get((3, 'mass')), 9)} {fmt(g.get((3, 'amax')), 9)}")

    def spread(cells, ell, k):
        v = [T[c][(ell, k)] for c in cells if c in T and (ell, k) in T[c]]
        return (max(v) - min(v)) if len(v) >= 2 else np.nan

    out("")
    out("  THE ONE-BIT CONTRAST AND ITS HANDLES (L3 = rehearsed, L2 = never rehearsed)")
    out("")
    out(f"  {'':<44}{'L3 mass':>11}{'L3 amax':>11}{'L2 mass':>11}{'L2 amax':>11}")
    rows = [
        ("pp_s1 pair effect |selfL - matchL|",
         [(("pp_s1", PAIR[0]), ("pp_s1", PAIR[1]))]),
    ]
    e = {}
    for ell in (2, 3):
        for k in ("mass", "amax"):
            e[(ell, k)] = abs(T[("pp_s1", PAIR[0])][(ell, k)]
                              - T[("pp_s1", PAIR[1])][(ell, k)])
    out("  " + f"{'pp_s1 pair effect |selfL - matchL|':<44}"
        + "".join(fmt(e[(ell, k)], 11) for ell, k in
                 ((3, "mass"), (3, "amax"), (2, "mass"), (2, "amax"))))
    h_cells = [("pp_s1", PAIR[0]), ("pp_s1", PAIR[1]), ("ap_s0", "exact")]
    h = {k: spread(h_cells, 2, k) for k in ("mass", "amax")}
    out("  " + f"{'unrehearsed-L2 handle (n=3, incl ap_s0/exact)':<44}"
        + fmt(np.nan, 11) + fmt(np.nan, 11)
        + fmt(h["mass"], 11) + fmt(h["amax"], 11))
    out("  " + f"{'  -> L3 effect / L2 handle':<44}"
        + fmt(e[(3, "mass")] / h["mass"] if h["mass"] else np.nan, 11, 2)
        + fmt(e[(3, "amax")] / h["amax"] if h["amax"] else np.nan, 11, 2))
    out("")
    out("  For reference, the same statistic on ap_s0's credit-vs-exposure pair:")
    ec = {k: abs(T[("ap_s0", "exact_reh")][(3, k)] - T[("ap_s0", "exact_exp")][(3, k)])
          for k in ("mass", "amax")}
    out("  " + f"{'ap_s0 |exact_reh - exact_exp| @L3':<44}"
        + fmt(ec["mass"], 11) + fmt(ec["amax"], 11))

    # ---------------------------------------------------------------- (3) money
    out("")
    out("-" * 100)
    out("(3) VALUE — recovered fraction per era, and the anchor->gift bracket")
    out("-" * 100)
    refs, n_era = su1["refs"], len(su1["eras"])
    anchor = load(os.path.join(ASSAY, "as_s0"), "anchor")
    rs = json.load(open(os.path.join(ASSAY, "as_s0", "setup.json")))
    same_refs = bool(np.allclose(rs["refs"]["stale"], refs["stale"])
                     and np.allclose(rs["refs"]["floor"], refs["floor"]))
    out(f"\n  `anchor` imported from as_s0 (unchanged replay); era references identical "
        f"= {same_refs}")
    M = {}
    for lab, d in ([(a, A[a]) for a in PAIR]
                   + [("ap_s0/" + a, load(os.path.join(FIG, "ap_s0"), a))
                      for a in ("exact_reh", "exact_exp", "exact", "given_c1")]
                   + [("pp_s0/exact_reh_s", load(os.path.join(FIG, "pp_s0"),
                                                 "exact_reh_s"))]
                   + ([("as_s0/anchor", anchor)] if same_refs else [])):
        if d is None:
            continue
        M[lab] = recovered(d["log"], refs, n_era)
    out("")
    out(f"  {'arm':<20}" + "".join(f"{'era' + str(j + 1):>10}" for j in range(n_era)))
    for lab, r in M.items():
        out(f"  {lab:<20}" + "".join(fmt(r[j], 10, 3) for j in range(n_era)))
    if "as_s0/anchor" in M and "ap_s0/given_c1" in M:
        out("")
        out("  ERAS 4-5 BRACKET: as_s0/anchor -> ap_s0/given_c1, closure fraction")
        out(f"  {'arm':<20}" + "".join(f"{'era' + str(j + 1):>10}" for j in (3, 4)))
        B = {}
        for lab, r in M.items():
            cl = []
            for j in (3, 4):
                b0, b1 = M["as_s0/anchor"][j], M["ap_s0/given_c1"][j]
                cl.append((r[j] - b0) / (b1 - b0) if (b1 - b0) else np.nan)
            B[lab] = cl
            out(f"  {lab:<20}" + "".join(fmt(c, 10, 3) for c in cl))
        out("")
        out("  pp_s1 pair difference (selfL - matchL), eras 4/5:  "
            + "  ".join(f"{B[PAIR[0]][i] - B[PAIR[1]][i]:+.3f}" for i in (0, 1)))
        out("  ap_s0 credit-vs-exposure (reh - exp), eras 4/5:    "
            + "  ".join(f"{B['ap_s0/exact_reh'][i] - B['ap_s0/exact_exp'][i]:+.3f}"
                        for i in (0, 1)))
        out("")
        out("  The measured depth-6 stream floor (as_s1, census/assay finding 7) moves")
        out("  eras-3-5 recovered fractions by up to 0.087 (exact family) — so a recovered-")
        out("  fraction difference below ~0.09 is not separable from stream position alone.")

    open(os.path.join(FIG, "pp_s1_reduction.txt"), "w").write("\n".join(L) + "\n")
    print("\nwrote", os.path.join(FIG, "pp_s1_reduction.txt"))
    if args.figures:
        figures(A, T, M, dose)
    return 0


def figures(A, T, M, dose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {PAIR[0]: "#c44e52", PAIR[1]: "#4c72b0"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a in PAIR:
        R = [(i + 1, q) for i, q in enumerate(A[a]["log"]["reh"]) if q and q.get("prov")]
        ax[0].plot([c for c, _ in R], [q["n_pairs_used"] for _, q in R], lw=1.6,
                   color=col[a], label=a)
    ax[0].set_yscale("symlog"); ax[0].set_xlabel("cycle")
    ax[0].set_ylabel("pairs into pi's buffer")
    ax[0].set_title("delivered dose — matched by construction", fontsize=10)
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=7)
    for a in PAIR:
        r = RT.load_arm(os.path.join(FIG, "pp_s1", a, "results.json"))
        ax[1].plot(r["probe_cycles"], r["mass"][3], lw=1.7, color=col[a], label=a)
    rr = RT.load_arm(os.path.join(FIG, "ap_s0", "exact_reh", "results.json"))
    ax[1].plot(rr["probe_cycles"], rr["mass"][3], lw=1.2, color="0.5", ls="--",
               label="ap_s0/exact_reh (full credit)")
    ax[1].axvline(70, color="0.8", lw=.8); ax[1].set_xlabel("cycle")
    ax[1].set_ylabel("pi's L3 proposal mass")
    ax[1].set_title("trust at the rehearsed level", fontsize=10)
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=7)
    labs = [k for k in M if k.startswith(("exact_rehL", "ap_s0/", "as_s0/"))]
    x = np.arange(len(labs))
    ax[2].bar(x - .18, [M[k][3] for k in labs], .36, label="era 4")
    ax[2].bar(x + .18, [M[k][4] for k in labs], .36, label="era 5")
    ax[2].set_xticks(x); ax[2].set_xticklabels([k.replace("ap_s0/", "").replace("as_s0/", "")
                                                for k in labs], rotation=35, ha="right",
                                               fontsize=7)
    ax[2].set_ylabel("recovered fraction"); ax[2].grid(alpha=.25, axis="y")
    ax[2].set_title("deep-era value", fontsize=10); ax[2].legend(fontsize=7)
    fig.suptitle("[antiphon P\"] pp_s1 — level-restricted provenance gate vs its "
                 "count-matched provenance-blind control", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pp2_level_pair.png"), dpi=130)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "pp2_level_pair.png"))


if __name__ == "__main__":
    sys.exit(main())
