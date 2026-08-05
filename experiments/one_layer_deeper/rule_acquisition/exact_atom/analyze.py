"""Aggregate `exact_atom` results into the three cuts' tables.

Everything is reported as an *error rate* `eps` rather than an accuracy, because the
question this node exists to answer is how far `eps` is from the exactness Hard's ladder
gates on. `certifiable_T(eps) = ln2 / (768 * eps)` is the deepest rung a one-step error
rate could carry under `ballistic_depth` §9's test-time re-projection, and it is printed
next to every number so an accuracy is never quoted without its consequence.

  python3 one_layer_deeper/rule_acquisition/exact_atom/analyze.py \\
      --tags divq8,divq64,divqfull,divdensen,mul_a,mul_b,sq_a,sq_b,sq_c
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
HARD_EXAMPLES = 768
LADDER = [1, 2, 4, 8, 16, 32, 64]


def certifiable_T(eps: float) -> float:
    return float("inf") if eps <= 0 else math.log(2.0) / (HARD_EXAMPLES * eps)


def eps_for_rung(T: int) -> float:
    return math.log(2.0) / (HARD_EXAMPLES * T)


def load(tags: list[str], seed: int = 0) -> dict:
    arms = {}
    for tag in tags:
        p = RES / tag / f"results_seed{seed}.json"
        if not p.exists():
            print(f"[missing] {p}")
            continue
        blob = json.loads(p.read_text())
        for name, res in blob["arms"].items():
            arms[name] = {**res, "_tag": tag, "_spec": blob["arm_specs"].get(name, {})}
    return arms


def _fit_tail(log_rows: list[dict], key: str, frac: float = 0.5):
    """Power-law slope of `eps` against `step` over the last `frac` of training.

    A slope near 0 says the error rate has stopped falling — the arm is at an asymptote and
    more budget will not buy exactness. A steep negative slope says it is still closing, and
    the extrapolated step count to reach a given rung is meaningful (as an optimistic bound).
    """
    pts = [
        (r["step"], r[key])
        for r in log_rows
        if key in r and r[key] > 0 and r["step"] > 0
    ]
    pts = pts[int(len(pts) * (1 - frac)) :]
    if len(pts) < 3:
        return None, None
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(e) for _, e in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return slope, intercept


def table(arms: dict, names: list[str], pools: list[str], title: str):
    names = [n for n in names if n in arms]
    if not names:
        return
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))
    hdr = f"{'arm':<26}{'pool':<16}{'n':>9}{'n_err':>9}{'eps':>11}{'acc':>9}{'floor':>8}{'cert T':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        a = arms[name]
        for pool in pools:
            v = a["final"].get(pool)
            if v is None:
                continue
            fl = a["floor_no_reduction"].get(pool, float("nan"))
            print(
                f"{name:<26}{pool:<16}{v['n']:>9}{v['n_err']:>9}{v['eps']:>11.3e}"
                f"{v['acc']:>9.5f}{fl:>8.4f}{certifiable_T(v['eps']):>9.3g}"
            )


def trajectory(arms: dict, names: list[str], pool: str):
    """Is `eps` still closing with budget, or has it flattened?"""
    names = [n for n in names if n in arms]
    if not names:
        return
    print(f"\n=== eps vs budget on `{pool}` " + "=" * 38)
    print(f"{'arm':<26}{'eps@25%':>11}{'eps@50%':>11}{'eps@final':>11}{'tail slope':>12}{'steps->T=1':>13}{'steps->T=64':>13}")
    print("-" * 97)
    for name in names:
        log = arms[name]["train_log"]
        key = f"{pool}.eps"
        rows = [r for r in log if key in r]
        if not rows:
            continue
        q1, q2, fin = rows[len(rows) // 4][key], rows[len(rows) // 2][key], rows[-1][key]
        slope, icept = _fit_tail(rows, key)
        cells = [f"{q1:>11.3e}", f"{q2:>11.3e}", f"{fin:>11.3e}"]
        if slope is None:
            cells += [f"{'--':>12}", f"{'--':>13}", f"{'--':>13}"]
        else:
            cells.append(f"{slope:>12.2f}")
            for T in (1, 64):
                # log eps = icept + slope*log(step)  ->  step = exp((log eps - icept)/slope)
                lg = (math.log(eps_for_rung(T)) - icept) / slope if slope < -1e-3 else 1e9
                cells.append(f"{'never':>13}" if lg > 100 else f"{math.exp(lg):>13.2e}")
        print(f"{name:<26}" + "".join(cells))
    print("  tail slope ~ 0 => asymptote, more budget buys nothing. Extrapolations assume the")
    print("  power law continues and are therefore optimistic bounds, not predictions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="divq8,divq64,divqfull,divdensen,mul_a,mul_b,sq_a,sq_b,sq_c")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = load([t.strip() for t in args.tags.split(",") if t.strip()], args.seed)
    if not arms:
        print("no results found")
        return

    print("\nThe ladder, for reference — eps a rung needs at 768 examples under k=1 re-projection:")
    print("  " + "  ".join(f"T={T}:{eps_for_rung(T):.1e}" for T in LADDER))

    DIV = ["divq8", "divq64", "divqfull", "divqfull_densen", "divq64_densen", "divq64_wide"]
    MUL = ["mul3", "mul4", "mul5", "mul4_sparse", "mul5_sparse"]
    SQ = ["sqpad", "sqpad_div", "sqpad_seam", "sqpad_div_seam", "sqpad_div_seam_densen",
          "sqpad_seamcos", "sqpad_div_seamcos", "sqpad_div_seamcos_w10"]

    table(arms, DIV, ["seen_y", "heldout_y", "heldout_n"],
          "Cut A - the reduce: does eps close with budget?")
    trajectory(arms, DIV, "heldout_y")

    table(arms, MUL, ["seen_x", "heldout_x"],
          "Cut C - the multiply: algorithm or table?")
    for name in [n for n in MUL if n in arms]:
        sp = arms[name]["_spec"]
        w, tf = sp.get("n_digits", 3), sp.get("train_frac", 0.5)
        print(f"    {name:<14} space=10^{w}={10**w:<8} train inputs~{int(tf * 10**w):<7} "
              f"held-out~{int((1 - tf) * 10**w)}")
    trajectory(arms, MUL, "heldout_x")

    table(arms, SQ, ["seen_x", "heldout_x", "heldout_n", "div_heldout_y"],
          "Cut B - the composed atom: {div aux} x {seam closure}")
    print("\n  2x2 on held-out x (floor is ~0.039; the parent's `sq` reads 0.019):")
    grid = {
        ("-", "-"): "sqpad", ("div", "-"): "sqpad_div",
        ("-", "seam"): "sqpad_seam", ("div", "seam"): "sqpad_div_seam",
    }
    for (d, s), nm in grid.items():
        if nm in arms:
            v = arms[nm]["final"].get("heldout_x")
            dv = arms[nm]["final"].get("div_heldout_y")
            extra = f"   (reduce alone: {1-dv['eps']:.4f})" if dv else ""
            print(f"    div={d:<4} seam={s:<5} -> {v['acc']:.4f}{extra}")
    if "sqpad_div_seam_densen" in arms:
        a = arms["sqpad_div_seam_densen"]["final"]
        print(f"    + dense N          -> {a['heldout_x']['acc']:.4f}  "
              f"held-out N: {a['heldout_n']['acc']:.4f}")

    # ---- Cut D: the memorisation boundary ---------------------------------------------
    MEMB = ["memb_d128", "mul5", "memb_d512", "memb_d512L8",
            "mul5_sparse", "memb_n5k", "memb_n20k"]
    memb = [n for n in MEMB if n in arms]
    if memb:
        print("\n=== Cut D - the memorisation boundary " + "=" * 36)
        print("  Cut C's mechanism as a falsifiable prediction: generalisation should track")
        print("  whether the training set FITS, not how big the model is. seen eps > 0 means")
        print("  memorisation failed. If the claim holds, capacity buys *worse* transfer.")
        hdr = (f"{'arm':<14}{'params':>11}{'train n':>9}{'seen eps':>11}{'memorised?':>12}"
               f"{'held-out eps':>14}{'held-out acc':>14}{'slope':>8}")
        print(hdr); print("-" * len(hdr))
        for name in memb:
            a, sp = arms[name], arms[name]["_spec"]
            w, tf = sp.get("n_digits", 5), sp.get("train_frac", 0.5)
            se, he = a["final"]["seen_x"]["eps"], a["final"]["heldout_x"]["eps"]
            slope, _ = _fit_tail(a["train_log"], "heldout_x.eps")
            print(f"{name:<14}{a['n_params']:>11,}{int(tf * 10**w):>9}{se:>11.3e}"
                  f"{('YES' if se == 0 else 'no'):>12}{he:>14.3e}"
                  f"{a['final']['heldout_x']['acc']:>14.4f}"
                  f"{(f'{slope:.2f}' if slope is not None else '--'):>8}")

    # ---- Cut E: the composed atom at 4 digits ------------------------------------------
    E4 = ["div4_q64", "div4_qfull", "sq4_plain", "sq4_div_seamcos",
          "sqpad_div_seamcos_gx"]
    if any(n in arms for n in E4):
        table(arms, ["div4_q64", "div4_qfull"], ["seen_y", "heldout_y", "heldout_n"],
              "Cut E - is the reduce still learnable at 4 digits?")
        table(arms, ["sq4_plain", "sq4_div_seamcos", "sqpad_div_seamcos_gx"],
              ["seen_x", "heldout_x", "heldout_n", "div_heldout_y"],
              "Cut E - the composed atom, clean (global) x split")
        print("\n  Read against the per-modulus-split 3-digit arms: `sqpad_div_seamcos` reads")
        print("  0.0035 held-out x, and `sqpad_div_seamcos_gx` is the same arm with the only")
        print("  split that is honest for an N-independent multiply.")


if __name__ == "__main__":
    main()
