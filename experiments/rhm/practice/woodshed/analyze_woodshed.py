"""woodshed — the reduction for the rehearsal tag (F1 part 2).

Deliberately NOT a fork of `../assay/analyze_assay.py`: the trust machinery this round needs
already exists in `reduce_trust.py` (part 1), so this file adds only what is new — the gates
that license the tag, the rehearsal ledger, and the money readout — and imports the rest.

Sections
  (0) GATES        the three no-rehearsal arms must replay `as_s0`'s arms of the same name
                   bit-for-bit over all 116 cycles (cross-tag, full length), and the
                   rehearsal pair must be bit-identical to `exact` up to the cycle rehearsal
                   first fires (in-tag twin gate). Without both, nothing else is readable.
  (1) REHEARSAL LEDGER   what rehearsal actually did: episodes, solve rate, pairs offered,
                   pairs used, what fraction of the used supervision points at a macro slot
                   and at the rehearsed level's slots, and the grounding cost.
  (2) TRUST        the part-1 instrument re-run on this tag: mass, lift, and the crossing
                   times, per arm per level, on the same clock as every A-lineage node.
  (3) MONEY        recovered fraction per era and the eras-4-5 bracket, `assay`'s readout
                   verbatim in form so the numbers are comparable to `as_s0`'s.

Usage (from experiments/):
    python3 rhm/practice/woodshed/analyze_woodshed.py --tag wd_s0 --fetch [--figures]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
ASSAY_FIG = os.path.join(PRACTICE, "assay", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_woodshed"

sys.path.insert(0, HERE)
import reduce_trust as RT                                            # noqa: E402

SHARED_PREFIX = 116
GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
# arms that must be byte-identical to the donor tag's arm of the same name
REPLAY = ("anchor", "exact", "given_c1")


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def _load(root, name):
    p = os.path.join(root, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def _fmt(v, w=8, p=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(w)
    return (f"{v:.{p}f}" if isinstance(v, float) else str(v)).rjust(w)


def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def _series_delta(old, new, n=SHARED_PREFIX):
    worst, per = 0.0, {}
    for k in GF_SERIES:
        x = np.asarray(old[k][:n], float)
        y = np.asarray(new[k][:n], float)
        k_ = min(len(x), len(y))
        d = float(np.abs(x[:k_] - y[:k_]).max()) if k_ else float("nan")
        per[k] = d
        worst = max(worst, d)
    return worst, per


# --------------------------------------------------------------------------- #

def gates(A, out):
    bad = []
    out("\n" + "=" * 90)
    out("(0) GATES — what licenses reading this tag at all")
    out("=" * 90)
    out("  (a) CROSS-TAG REPLAY, full length. `woodshed.py` adds exactly one block, gated on")
    out("      `reh_level`. Every arm that does not set it must reproduce `as_s0`'s arm of")
    out("      the same name bit-for-bit over all 116 cycles. That IS the knobs-off gate.")
    for a in REPLAY:
        ref = _load(os.path.join(ASSAY_FIG, "as_s0", a), "results.json")
        if a not in A:
            out(f"      {a:<12s} not in this tag — skipped")
            continue
        if ref is None:
            out(f"      {a:<12s} as_s0 reference not fetched locally — SKIPPED (not a pass)")
            continue
        worst, per = _series_delta(ref["log"], A[a]["log"])
        n_eq = len(ref["log"]["cycle"]) == len(A[a]["log"]["cycle"])
        ok = worst == 0.0 and n_eq
        out(f"      {a:<12s} max|delta| over {len(GF_SERIES)} series = {worst:.3e}   "
            f"cycle counts equal = {n_eq}   -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"cross-tag replay failed for {a}")
            out("        per-series: "
                + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))

    out("")
    out("  (b) IN-TAG TWIN GATE. `exact_reh` / `exact_exp` are `exact` plus rehearsal, and")
    out("      rehearsal cannot fire before the rehearsed level arrives. Up to that cycle")
    out("      all three must be identical; at it they may diverge and not before.")
    ref = A.get("exact")
    for a in ("exact_reh", "exact_exp"):
        if a not in A or ref is None:
            continue
        reh = A[a]["log"].get("reh") or []
        first = next((i + 1 for i, q in enumerate(reh) if q), None)
        n = min(len(ref["log"]["cycle"]), len(A[a]["log"]["cycle"]))
        pre = (first - 1) if first else n
        worst, per = _series_delta(ref["log"], A[a]["log"], n=pre)
        # first cycle at which any series differs
        div = None
        for i in range(n):
            if any(float(ref["log"][k][i]) != float(A[a]["log"][k][i]) for k in GF_SERIES):
                div = i + 1
                break
        ok = worst == 0.0
        out(f"      {a:<12s} first rehearsal cycle = {first}   pre-window c1-c{pre}: "
            f"max|delta| = {worst:.3e} -> {'PASS' if ok else 'FAIL'}   "
            f"first divergence from `exact` = c{div}")
        if not ok:
            bad.append(f"twin gate failed for {a}")

    out("")
    out("  (c) THE ONE-BIT PAIR. `exact_reh` and `exact_exp` run the same episodes and take")
    out("      the same number of gradient steps on the same-size pair sets; they differ in")
    out("      whether the success filter picks which trajectories pi imitates.")
    r, e = A.get("exact_reh"), A.get("exact_exp")
    if r and e:
        rr = [q for q in (r["log"].get("reh") or []) if q]
        ee = [q for q in (e["log"].get("reh") or []) if q]
        n = min(len(rr), len(ee))
        # The match is only ASSERTABLE on the FIRST rehearsal cycle. Up to that cycle the two
        # arms are the same agent, so the same episodes produce the same solved set and the
        # `expose` draw is the same size by construction. From the next cycle on they are
        # DIFFERENT agents -- that is the point of the treatment -- so their rehearsal solve
        # rates, and hence the matched pair count, legitimately diverge. Asserting equality
        # across all cycles (as `wd_s0`'s first reduction did) tests that the treatment had
        # no effect, which is the opposite of what the gate is for.
        first_ok = bool(n and rr[0]["n_pairs_used"] == ee[0]["n_pairs_used"]
                        and rr[0]["n_pairs_all"] == ee[0]["n_pairs_all"])
        out(f"      rehearsal cycles: {len(rr)} vs {len(ee)}")
        out(f"      first rehearsal cycle pair-count match (the assertable one): "
            f"used {rr[0]['n_pairs_used'] if n else '-'} vs "
            f"{ee[0]['n_pairs_used'] if n else '-'}  -> {'PASS' if first_ok else 'FAIL'}")
        later = sum(1 for i in range(1, n)
                    if rr[i]["n_pairs_used"] != ee[i]["n_pairs_used"])
        out(f"      later cycles differing in pairs-used: {later}/{max(n - 1, 0)} "
            f"(expected once the arms diverge; NOT a gate)")
        out(f"      NOTE: `n_pairs_all` is structurally constant (train_on='tips' keeps every "
            f"surviving trajectory), so matching it is vacuous and is not a gate.")
        if not first_ok or len(rr) != len(ee):
            bad.append("the rehearsal pair is not pair-count matched at the first "
                       "rehearsal cycle")

    out("")
    out("  (d) THE UNREHEARSED LEVEL is this round's only in-tag noise handle. Rehearsal")
    out("      targets L3 only, so whatever spread the three `exact` arms show at L2 is what")
    out("      trajectory divergence alone produces in this instrument. Quantified at the end")
    out("      of section (2). It is not a measured floor (single seed, n=3) but it is the")
    out("      only same-tag reference the round contains.")
    return bad


def rehearsal_ledger(A, out):
    out("\n" + "=" * 90)
    out("(1) REHEARSAL LEDGER — what rehearsal actually did")
    out("=" * 90)
    out("  succ        terminal possible-set success rate of the rehearsal episodes")
    out("  offered     pairs the rehearsal beam produced (all surviving trajectories)")
    out("  used        pairs actually appended to pi's buffer this cycle")
    out("  f_macro     fraction of the USED supervision whose target is a macro slot (L>=2)")
    out("  f_level     fraction whose target is a slot of the REHEARSED level")
    out("  g           rehearsal groundings (extra practice compute, never priced into the")
    out("              arm's declared per-solve budget; recorded so the price is on record)")
    for a, r in A.items():
        reh = [q for q in (r["log"].get("reh") or []) if q]
        if not reh:
            continue
        out(f"\n  --- {a}  (level {reh[0]['level']}, mode {reh[0]['mode']}, "
            f"{reh[0]['n_ep']} episodes/cycle, {len(reh)} rehearsal cycles)")
        out(f"  {'cyc':>5s} {'succ':>7s} {'offered':>8s} {'used':>7s} {'f_macro':>8s} "
            f"{'f_level':>8s} {'pbuf':>8s} {'g':>10s}")
        cycles = [i + 1 for i, q in enumerate(r["log"].get("reh") or []) if q]
        step = max(1, len(reh) // 12)
        for i in range(0, len(reh), step):
            q = reh[i]
            out(f"  {cycles[i]:>5d} {q['succ']:>7.3f} {q['n_pairs_all']:>8d} "
                f"{q['n_pairs_used']:>7d} {q['frac_macro']:>8.3f} "
                f"{q['frac_at_level']:>8.3f} {q['pbuf_n']:>8d} {q['g']:>10d}")
        tot_g = sum(q["g"] for q in reh)
        tot_u = sum(q["n_pairs_used"] for q in reh)
        out(f"  {'TOTAL':>5s} {np.mean([q['succ'] for q in reh]):>7.3f} "
            f"{sum(q['n_pairs_all'] for q in reh):>8d} {tot_u:>7d} "
            f"{np.mean([q['frac_macro'] for q in reh]):>8.3f} "
            f"{np.mean([q['frac_at_level'] for q in reh]):>8.3f} "
            f"{'':>8s} {tot_g:>10d}")


def money(A, setup, order, out):
    refs, n_era = setup["refs"], len(setup["eras"])
    st, fl = refs["stale"], refs["floor"]
    out("\n" + "=" * 90)
    out("(3) MONEY — recovered fraction per era (`assay`'s readout, same form)")
    out("=" * 90)
    out(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
    rows = {}
    for a in order:
        if a not in A:
            continue
        lg = A[a]["log"]
        row, rec = "", []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                rec.append(float("nan"))
                row += "-".rjust(17)
                continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            r = (st[j] - e) / (st[j] - fl[j])
            rec.append(r)
            row += f"{e:.3f} /{r:+.3f}".rjust(17)
        rows[a] = rec
        out(f"  {a:16s}{row}")
    out(f"  {'stale':16s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    out(f"  {'floor':16s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))
    # The bracket's floor is `anchor`. If this tag did not run it (it is an unchanged
    # replay), take it from `as_s0` -- but ONLY after asserting the era references match,
    # since recovered fraction is defined against this tag's own stale/floor.
    lo, hi = "anchor", "given_c1"
    if lo not in rows:
        ref = _load(os.path.join(ASSAY_FIG, "as_s0", "anchor"), "results.json")
        rs = _load(os.path.join(ASSAY_FIG, "as_s0"), "setup.json")
        same_refs = bool(rs and np.allclose(rs["refs"]["stale"], st)
                         and np.allclose(rs["refs"]["floor"], fl))
        out(f"\n  `anchor` not run in this tag (unchanged replay). Importing it from as_s0; "
            f"era references identical = {same_refs}.")
        if ref and same_refs:
            lg = ref["log"]
            rec = []
            for j in range(n_era):
                idx = _era_idx(lg, j)
                e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
                rec.append((st[j] - e) / (st[j] - fl[j]))
            rows[lo] = rec
            out(f"  {'anchor (as_s0)':16s}"
                + "".join(f"{rec[j]:+.3f}".rjust(17) for j in range(n_era)))
        else:
            out("  -> NOT imported; the bracket is skipped.")
    if lo in rows and hi in rows:
        out(f"\n  ERAS 4-5 BRACKET: {lo} -> {hi}")
        for j in (3, 4):
            if j >= n_era:
                continue
            b0, b1 = rows[lo][j], rows[hi][j]
            span = b1 - b0
            out(f"    era {j + 1}: bracket {b0:+.3f} .. {b1:+.3f} (span {span:+.3f})")
            for a in order:
                if a in (lo, hi) or a not in rows:
                    continue
                frac = (rows[a][j] - b0) / span if span else float("nan")
                out(f"      {a:<14s} {rows[a][j]:+.3f}   closes {frac:+.3f} of the bracket")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wd_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tag)
    root = os.path.join(FIG, args.tag)
    setup = _load(root, "setup.json")
    summary = _load(root, "summary.json")
    A = {}
    for a in sorted(os.listdir(root)):
        r = _load(os.path.join(root, a), "results.json")
        if r is not None:
            A[a] = r
    order = (summary or {}).get("order") or sorted(A)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 90)
    out(f"woodshed / {args.tag} — does rehearsal compress the trust clock gift could not skip?")
    out("=" * 90)
    out(f"  arms: {', '.join(order)}")

    bad = gates(A, out)
    rehearsal_ledger(A, out)

    # (2) trust — part 1's instrument, on this tag
    out("\n" + "=" * 90)
    out("(2) TRUST — the part-1 instrument on this tag (see reduce_trust.py for semantics)")
    out("=" * 90)
    recs = {}
    for a, r in A.items():
        rec = RT.load_arm(os.path.join(root, a, "results.json"))
        if rec is None:
            continue
        for ell in RT.LEVELS:
            st = RT.rate_stats(rec, ell)
            if st is not None:
                recs[(a, ell)] = st
    out(f"  {'arm':<14s} {'L':>2s} {'arr':>4s} {'kind':<11s} {'m0':>7s} {'mend':>7s} "
        f"{'lift0':>6s} {'liftmin':>8s} {'cross1':>7s} {'liftend':>8s} {'d16':>7s} "
        f"{'d32':>7s} {'t1/2':>6s} {'m@16':>7s} {'m@32':>7s} {'amax':>6s} {'win':>4s}")
    for (a, ell), st in sorted(recs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out(f"  {a:<14s} {ell:>2d} {st['arrival']:>4d} {st['kind']:<11s} "
            f"{st['m_first']:>7.4f} {st['m_end']:>7.4f} {st['lift_first']:>6.3f} "
            f"{st['lift_min']:>8.3f} {_fmt(st['tau_cross1'], 7, 1)} {st['lift_end']:>8.3f} "
            f"{_fmt(st['d16'], 7, 4)} {_fmt(st['d32'], 7, 4)} {_fmt(st['t_half'], 6, 1)} "
            f"{_fmt(st['at'][16], 7, 4)} {_fmt(st['at'][32], 7, 4)} {st['amax_end']:>6.3f} "
            f"{int(st['tau_max']):>4d}")
    out("")
    out("  THE CELL. `exact` is the clock as it runs with no rehearsal; `exact_reh` is the")
    out("  same clock with credited rehearsal; `exact_exp` with matched uncredited exposure;")
    out("  `given_c1` is the clock already paid. Read cross1, lift@32 and liftend across the")
    out("  four at the same tau.")
    for ell in (3, 2):
        row = [(a, recs.get((a, ell))) for a in ("anchor", "exact", "exact_reh",
                                                 "exact_exp", "given_c1")]
        if not any(v for _, v in row):
            continue
        out(f"    level {ell}:")
        for a, st in row:
            if st is None:
                continue
            out(f"      {a:<12s} arr c{st['arrival']:<4d} cross1={_fmt(st['tau_cross1'], 6, 1)}"
                f"  lift@16={_fmt(st['lift_at'][16], 6, 3)}"
                f"  lift@32={_fmt(st['lift_at'][32], 6, 3)}"
                f"  liftend={st['lift_end']:6.3f}  massend={st['m_end']:6.4f}"
                f"  argmax={st['amax_end']:5.3f}")

    # the L2-vs-L3 spread: divergence noise against the rehearsed-level movement
    out("")
    out("  SPREAD OF THE THREE `exact` ARMS (exact / exact_reh / exact_exp). L2 is NOT")
    out("  rehearsed, so its spread is divergence alone; L3 is the rehearsed level.")
    out(f"    {'level':>6s} {'stat':<10s} {'exact':>8s} {'exact_reh':>10s} {'exact_exp':>10s} "
        f"{'range':>8s} {'range/median':>13s}")
    for ell in (2, 3):
        for stat, lab in (("lift_end", "liftend"), ("m_end", "massend"),
                          ("amax_end", "argmax")):
            vs = []
            for a in ("exact", "exact_reh", "exact_exp"):
                st = recs.get((a, ell))
                vs.append(st[stat] if st else float("nan"))
            arr = np.asarray(vs, float)
            if not np.isfinite(arr).all():
                continue
            rng_ = float(arr.max() - arr.min())
            med = float(np.median(arr))
            out(f"    {ell:>6d} {lab:<10s} {arr[0]:>8.4f} {arr[1]:>10.4f} {arr[2]:>10.4f} "
                f"{rng_:>8.4f} {rng_ / med if med else float('nan'):>13.3f}")

    if setup:
        money(A, setup, order, out)

    out("\n" + "=" * 90)
    out("GATE SUMMARY")
    out("=" * 90)
    out("  " + ("ALL PASS" if not bad else "FAILURES: " + "; ".join(bad)))

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "reduction.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote {os.path.join(root, 'reduction.txt')}]")

    if args.figures:
        figures(root, recs, A)


def figures(root, recs, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"anchor": "0.45", "exact": "tab:blue", "exact_reh": "tab:red",
              "exact_exp": "tab:green", "given_c1": "tab:orange"}
    for ax, (field, lab, hline) in zip(axes, (("y", "pi L3 proposal mass", None),
                                              ("lift", "lift = mass / uniform share", 1.0),
                                              ("ax", "L3 argmax share", None))):
        for a, st in [(a, recs.get((a, 3))) for a in colors]:
            if st is None:
                continue
            ls = "--" if st["kind"] == "gift@c1" else "-"
            ax.plot(st["tau"], st[field], ls, lw=1.8, marker="o", ms=3,
                    color=colors[a], label=f"{a} (c{st['arrival']})")
        if hline:
            ax.axhline(hline, color="k", lw=1.0, ls=":")
        ax.set_xlabel("tau = cycle - arrival")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "wd_trust_L3.png"), dpi=150)
    plt.close(fig)
    print(f"[wrote {os.path.join(root, 'wd_trust_L3.png')}]")


if __name__ == "__main__":
    main()
