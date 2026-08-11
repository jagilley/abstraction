"""Signal-to-noise analysis of the Cut-3 outer-loop teacher (post-processing only, no compute).

Reproduces every number in README.md from `results.json` files already on disk under
`../figures/`. Three experiments:

  A. SNR diagnosis of the SPARSE (Cut-3) teacher      -> ../figures/online_value_loop_{teacher,selftune15}_s{0,1,2}
  B. DENSE teacher under drift, 3 seeds               -> ../figures/online_value_loop_dense_s{0,1,2}
  C. DENSE teacher, stationary DGP (Type-1), 1 seed   -> ../figures/online_value_loop_type1_s0

Run: python3 mjc/drift_value_loop/teacher_snr/teacher_snr_analysis.py     (from experiments/)
"""
import json, os, numpy as np

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
FIVE = ["b0.0", "b0.3", "b0.5", "b0.7", "b1.0"]
THREE = ["b0.3", "b0.5", "b0.7"]


def load(tag):
    p = os.path.join(FIG, f"online_value_loop_{tag}", "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def tail(recs, key="corridor_err"):
    """Post-transient (2nd half) series of a per-epoch record key."""
    v = [r[key] for r in recs]
    return np.array(v[len(v) // 2:])


def fixed_arms(d):
    return [k for k in d["results"] if k.startswith("b")]


# ---------------------------------------------------------------- A. sparse SNR
def exp_a():
    print("=" * 78)
    print("A. SPARSE (Cut-3) teacher: is the b-landscape resolvable from one epoch?")
    print("=" * 78)
    means, sds = {a: [] for a in FIVE}, []
    for s in range(3):
        d = load(f"teacher_s{s}")
        if not d:
            print("  missing teacher_s%d" % s); return
        for a in FIVE:
            e = tail(d["results"][a]["records"])
            means[a].append(e.mean()); sds.append(e.std(ddof=1))
    tm = np.array([np.mean(means[a]) for a in FIVE])
    depth, noise = tm.max() - tm.min(), float(np.mean(sds))
    for a, m in zip(FIVE, tm):
        print(f"   {a}: corridor_err(2nd half) = {m:.5f}")
    print(f"\n   bowl depth (5-pt) = {depth:.5f}   argmin = {FIVE[int(tm.argmin())]}")
    print(f"   per-epoch nuisance sd at FIXED b = {noise:.5f}   (all variation here is nuisance)")
    print(f"   >>> SNR = {depth / noise:.3f}")

    # shared nuisance across arms -> would a PAIRED (common-random-number) contrast help?
    cs = []
    for s in range(3):
        d = load(f"teacher_s{s}")
        E = {a: np.array([r["corridor_err"] for r in d["results"][a]["records"]]) for a in FIVE}
        for i in range(len(FIVE)):
            for j in range(i + 1, len(FIVE)):
                cs.append(np.corrcoef(E[FIVE[i]], E[FIVE[j]])[0, 1])
    rho = float(np.mean(cs))
    print(f"\n   epoch-wise corr between b-arms (shared nuisance) rho = {rho:+.3f}")
    print(f"   paired-difference sd = sd*sqrt(2(1-rho)) -> paired SNR = "
          f"{depth / (noise * np.sqrt(2 * (1 - rho))):.2f}  (vs {depth / noise:.2f} unpaired)")

    # what the outer loop actually extracts
    E, A = [], []
    for s in range(3):
        d = load(f"selftune15_s{s}")
        if not d: continue
        oh = d["results"]["online_front"]["outer_hist"]; w = d["config"]["outer_warmup"]
        u = [h for h in oh if h["epoch"] >= w and h["eps"] != 0.0]
        E += [h["eps"] for h in u]; A += [h["adv"] for h in u]
    E, A = np.array(E), np.array(A)
    c = np.corrcoef(E, A)[0, 1]
    g = A * E
    print(f"\n   gradient samples n={len(E)}   corr(eps,adv) = {c:+.3f}   r^2 = {c ** 2:.4f}")
    print(f"   adv*eps: mean={g.mean():+.5f} sd={g.std(ddof=1):.5f}  -> n for 2-sigma sign "
          f"certainty ~ {(2 * g.std(ddof=1) / abs(g.mean())) ** 2:.0f}")
    print(f"   final b (target 0.5, init 0.15): "
          f"{[round(load(f'selftune15_s{s}')['results']['online_front']['outer_hist'][-1]['b_theta'], 3) for s in range(3)]}")


# ------------------------------------------------- B/C. dense teacher + variance decomposition
def decompose(tag, arms=THREE):
    """Return (sd_ratio, within_var, between_var, landscape) for one run."""
    d = load(tag)
    if not d: return None
    sd, W, B, land = [], [], [], {}
    for a in [x for x in fixed_arms(d) if x in arms]:
        recs = d["results"][a]["records"]; h = len(recs) // 2
        sp, dn = tail(recs, "corridor_err"), tail(recs, "corridor_err_dense")
        sd.append(dn.std(ddof=1) / sp.std(ddof=1))
        ds = [np.array(r["dense_samples"]) for r in recs[h:] if len(r["dense_samples"]) > 1]
        W.append(np.mean([x.var(ddof=1) for x in ds]))     # within-epoch  (averageable)
        B.append(np.var([x.mean() for x in ds], ddof=1))   # between-epoch (NOT averageable)
        land[a] = float(sp.mean())
    oh = d["results"]["online_front"]["outer_hist"]; w = d["config"]["outer_warmup"]
    u = [h for h in oh if h["epoch"] >= w and h["eps"] != 0.0]
    return dict(regime=d.get("regime"), sd=float(np.mean(sd)), W=float(np.mean(W)),
                B=float(np.mean(B)), land=land,
                corr=float(np.corrcoef([h["eps"] for h in u], [h["adv"] for h in u])[0, 1]),
                traj=[h["b_theta"] for h in oh])


def exp_bc():
    print("\n" + "=" * 78)
    print("B/C. DENSE teacher (epoch mean of M+1 samples) — drift vs stationary")
    print("=" * 78)
    print(f'{"run":<11}{"regime":<12}{"sd ratio":>9}{"within sd":>11}{"between sd":>12}{"B/W":>7}{"corr":>8}')
    print("-" * 70)
    R = {}
    for tag, lbl in [("dense_s0", "drift s0"), ("dense_s1", "drift s1"),
                     ("dense_s2", "drift s2"), ("type1_s0", "static s0")]:
        r = decompose(tag)
        if not r: print(f"{lbl:<11} MISSING"); continue
        R[tag] = r
        print(f'{lbl:<11}{str(r["regime"]):<12}{r["sd"]:9.3f}{np.sqrt(r["W"]):11.5f}'
              f'{np.sqrt(r["B"]):12.5f}{r["B"]/r["W"]:7.2f}{r["corr"]:+8.3f}')

    dr = [R[t] for t in ("dense_s0", "dense_s1", "dense_s2") if t in R]
    if not (dr and "type1_s0" in R): return
    dW, dB = np.mean([x["W"] for x in dr]), np.mean([x["B"] for x in dr])
    st = R["type1_s0"]
    print(f"\n   B/W variance ratio:  drift {dB/dW:.2f}  ->  static {st['B']/st['W']:.2f}"
          "   (unchanged => state noise is intrinsic to the learning FM)")
    print(f"   ceiling on dense at M=inf, sqrt(B/(B+W)):  drift {np.sqrt(dB/(dB+dW)):.3f}"
          f"  ->  static {np.sqrt(st['B']/(st['B']+st['W'])):.3f}")

    print("\n   matched 3-point landscape (b0.3/b0.5/b0.7):")
    for lbl, land, W, B in [("drift ", {a: np.mean([x["land"][a] for x in dr]) for a in THREE}, dW, dB),
                            ("static", st["land"], st["W"], st["B"])]:
        v = np.array([land[a] for a in THREE]); tot = np.sqrt(B + W)
        print(f"     {lbl}: spread={v.max()-v.min():.5f}  per-epoch sd={tot:.5f}  "
              f"SNR={(v.max()-v.min())/tot:.2f}  argmin={THREE[int(v.argmin())]}")

    print(f"\n   static b trajectory (init 0.15, target 0.5): "
          + " ".join(f"{x:.2f}" for x in st["traj"]))
    # local gradient at the operating point vs the perturbation the outer loop makes
    slope = (st["land"]["b0.3"] - st["land"]["b0.5"]) / 0.2      # per unit b, 0.3->0.5 segment
    b0, sig = 0.15, 0.7
    db = b0 * (1 - b0) * sig                                     # sigmoid' * theta-space sigma
    tot = np.sqrt(st["B"] + st["W"])
    print(f"   local check @ b=0.15: slope~{slope:.4f}/unit-b, perturbation db~{db:.3f} "
          f"-> signal {slope*db:.5f} vs noise {tot:.5f}  => local SNR {slope*db/tot:.2f}")


if __name__ == "__main__":
    exp_a(); exp_bc()
