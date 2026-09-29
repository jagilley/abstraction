"""[grokking/basis] Reduce a `basis.py::basis_run` run to the table of record and figure.

    python3 grokking/basis/reduce_basis.py --tag b1 --fetch      # from experiments/

--fetch pulls /data/<tag>/basis.json off `grokking-mint-data` into results/<tag>/basis.json (git-ignored; the
volume copy is the artifact). m1's record is read from ../results/<mint_tag>/mint.json (fetched if missing).
Writes:
    figures/basis_<tag>_table.txt     designed / instrument checks, per-snapshot read and walk tables (each
                                      rule) with kept K matched to DFT frequencies vs m1's K, final-net
                                      per-subspace table, phase check, per-frequency recovery, falsifiers
    figures/basis_<tag>_read.png      DFT overlap and cross-g agreement over epochs, kept k vs m1, and the
                                      per-frequency recovery heatmap
    results/basis_<tag>_summary.json  compact per-snapshot series behind the figure
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
FORMS = ("A", "ALS", "B", "C")
RULES = ("le0", "lt1", "m3")
P = 97
DETAIL_EPOCHS = (0, 5000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
                 25000, 30000, 39999)


def _get(vol_path, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE="chromatic")
    subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", vol_path, local],
                   check=True, env=env)


def load(tag, fetch):
    path = os.path.join(HERE, "results", tag, "basis.json")
    if fetch:
        _get(f"{tag}/basis.json", path)
    with open(path) as f:
        d = json.load(f)
    mpath = os.path.join(PARENT, "results", d["mint_tag"], "mint.json")
    if not os.path.exists(mpath):
        _get(f"{d['mint_tag']}/mint.json", mpath)
    with open(mpath) as f:
        m1 = json.load(f)["records"]
    return d, m1


def by_run(d, run):
    R = [r for r in d["records"].values() if r["run"] == run]
    return sorted(R, key=lambda r: (r["epoch"], r["label"]))


def m1_rec(m1, r):
    key = f"{r['run']}:{r['label']}"
    return m1.get(key)


def kcmp(r, m1r, form, rule):
    """(k_rec, |K_match|, k_m1, |K_match & K_m1|, equal, mean dft_ov over kept candidates)."""
    W = r["walks"][form][rule]
    Km = set(W["K_match"])
    K1 = set(m1r["walks"][form][rule]["K"]) if m1r else set()
    ov = np.asarray(r["read"]["dft_ov"])
    mo = float(ov[W["K"]].mean()) if W["K"] else float("nan")
    return W["k"], len(Km), len(K1), len(Km & K1), Km == K1, mo


def _keq(r, m1, form, rule):
    """1.0 if the matched K equals m1's K, None (skipped) where m1's K is empty."""
    mr = m1_rec(m1, r)
    if not mr["walks"][form][rule]["K"]:
        return None
    return float(kcmp(r, mr, form, rule)[4])


def ops_summary(r):
    o = list(r["ops"].values())
    return (sum(x["bijection"] for x in o), sum(1 for x in o if x["n_cycles"] == 1),
            float(np.mean([x["conf"] for x in o])), float(np.mean([x["shift_acc_hard"] for x in o])),
            float(np.mean([x["shift_mass_soft"] for x in o])), float(np.mean([x["n_fixed"] for x in o])))


def read_row(r):
    rd = r["read"]
    bij, cyc, conf, sacc, smass, nfix = ops_summary(r)
    lm = np.median(rd["lam_mod"]) if rd["lam_mod"] else float("nan")
    return (f"{r['e_hat']:>3} {bij:>2}/{len(r['ops'])} {cyc:>2} {conf:>5.3f} | {rd['m']:>3} {rd['n_real']:>3} "
            f"{lm:>5.3f} | {rd['xg_mean']:>6.4f} {rd['xg_min']:>6.4f} {rd['xg_frac99']:>5.2f} "
            f"{rd['comm_mean']:>6.4f} || {rd['dft_ov_mean']:>6.4f} {rd['dft_ov_min']:>6.4f} "
            f"{rd['dft_rec_min']:>6.4f} {rd['n_kmatch_distinct']:>3} {'y' if rd['one_to_one'] else 'n':>2} "
            f"{rd['phase_err_max']:>8.1e} {rd['phase_err_median']:>8.1e} {rd['dc_ov']:>5.3f} {sacc:>5.3f}")


READ_HDR = (f"{'e^':>3} {'bij':>4} {'1c':>2} {'conf':>5} | {'m':>3} {'nre':>3} {'|lam|':>5} | "
            f"{'xg_mn':>6} {'xg_mi':>6} {'x>.99':>5} {'comm':>6} || {'dft_mn':>6} {'dft_mi':>6} "
            f"{'rec_mi':>6} {'nk':>3} {'11':>2} {'ph_max':>8} {'ph_med':>8} {'dc':>5} {'shift':>5}")


def table(d, m1, tag):
    L = []
    w = L.append
    true, shuf, init = by_run(d, "true"), by_run(d, "shuffled"), by_run(d, "init")
    gens = d["gens"]
    w(f"# grokking / basis -- tag {tag}   (endogenous basis mint on the banked {d['mint_tag']} snapshots; "
      f"p={d['p']}, seed 42, no retraining)")
    w(f"# generators GENS = {gens} (drawn once, RandomState(42), symbols as names only); g_ref = {d['g_ref']}")
    w("# M_g[c,a] = softmax(net(a,g))[c]; candidates = real invariant subspaces of M_g_ref (2-D per complex pair, "
      "1-D per other real eigenvalue); recovered DC = eigenvector at eigenvalue 1.")
    w("# ENDOGENOUS (no labels): e^ = net's own identity symbol (argmax_s mean_a softmax(net(a,s))[a]); bij = # of g "
      "whose hard map a->argmax net(a,g) is a bijection; 1c = # single p-cycles; conf = mean max prob;")
    w("#   m = # candidates, nre = # real eigenvalues (incl. DC), |lam| = median eigenvalue modulus; xg = per-candidate "
      "best projector overlap with the other g's candidates, min over g (mean / min / frac > 0.99 over candidates);")
    w("#   comm = mean ||[M_g, M_h]||_F / ||M_g M_h||_F.")
    w("# ORACLE (logged, never consumed): dft = per-candidate best projector overlap with a true DFT pair (mean / min); "
      "rec_mi = min over the 48 DFT pairs of the best candidate overlap; nk = # distinct matched k; 11 = one-to-one;")
    w("#   ph = |phase of Q^T M_g Q - true character angle 2 pi k g / p (folded)| over 2-D candidates and all 6 g "
      "(max / median); dc = overlap of recovered DC with ones; shift = mean hard-map accuracy vs a -> a+g.")
    w("# Forms over the RECOVERED basis: A = sum_K Re[psi(a)psi(b)conj psi(c)], psi phase-fixed at e^; ALS = same with "
      "joint-LS amplitudes; B = units whose dominant candidate is not in K zeroed; C = layer-0 rows projected onto "
      "span(DC + K).")
    w("# DEGENERACY (carried over from m1 unchanged): with exact characters form A has argmax a+b at EVERY nonempty K, "
      "so A / ALS gates can only decide empty vs nonempty. The non-degenerate outputs here are the certificate (xg), "
      "the oracle (dft, ph) and the B / C K comparison.")
    w("")
    # ---------------------------------------------------------------- designed + instrument checks
    w("## Designed checks (pure operators; steps 2-4 only)")
    w(f"{'check':>22} {'m':>3} {'nre':>3} {'xg_mean':>8} {'xg_min':>8} {'dft_mean':>8} {'dft_min':>8} "
      f"{'rec_min':>8} {'1:1':>4} {'ph_max':>8} {'comm':>8} {'|lam|min':>8}  extra")
    for k, v in d["gates"]["designed"].items():
        extra = {kk: vv for kk, vv in v.items() if kk in ("n_cycles", "relabelled_dft_ov_min")}
        w(f"{k:>22} {v['m']:>3} {v['n_real']:>3} {v['xg_mean']:>8.5f} {v['xg_min']:>8.5f} {v['dft_ov_mean']:>8.5f} "
          f"{v['dft_ov_min']:>8.5f} {v['dft_rec_min']:>8.5f} {'y' if v['one_to_one'] else 'n':>4} "
          f"{v['phase_err_max']:>8.1e} {v['comm_mean']:>8.1e} {v['lam_mod_min']:>8.4f}  {extra}")
    w("  D1 = the exact shift a->a+g (a 97-cycle in the symbols' own labels); D2 = an independent random permutation "
      "per g; D2b = an independent random 97-cycle per g; D3 = sigma o shift_g o sigma^-1 (a relabelled group action, "
      "common sigma); D4 = 0.7 shift_g + 0.3 circulant(softmax(random even kernel)).")
    w("## Instrument checks (banked nets)")
    for k, v in d["gates"]["instrument"].items():
        w(f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk != "K_equal"))
    w("  I-1 = the exact DFT basis injected into this pipeline vs m1's recorded K / held-out (all 12 form x rule gates); "
      "I-2 = recovered basis, form B at K=all bit-exact, form C at K=all max |dlogit|; I-3 = ALS closed form vs "
      "explicit lstsq with per-row intercepts (300-pair subsample).")
    allr = list(d["records"].values())
    w(f"  in-run: form B at K=all bit-exact {sum(r['B_all_bitexact'] for r in allr)}/{len(allr)}; form C at K=all "
      f"max |dlogit| {max(r['C_all_maxabs'] for r in allr):.2e}, min argmax agreement "
      f"{min(r['C_all_argmax_agree'] for r in allr):.4f}")
    w("")
    # ---------------------------------------------------------------- per-snapshot read table
    w("## Per-snapshot READ table (true-label run): operator, certificate (endogenous) || oracle")
    w(f"{'epoch':>6} {'net_tr':>6} {'net_te':>6} | " + READ_HDR)
    for r in true:
        w(f"{r['epoch']:>6} {r['net_train_acc_fit']:>6.3f} {r['net_test_acc']:>6.3f} | " + read_row(r))
    w("")
    # ---------------------------------------------------------------- per-snapshot walk tables
    for rn in RULES:
        w(f"## Per-snapshot WALK table, rule {rn} (true-label run). k = # recovered candidates kept; "
          f"B/C: rec k | #matched k | m1 k | #common | = ; ov = mean DFT overlap of the kept candidates")
        w(f"{'epoch':>6} {'net_te':>6} | {'kA':>3} {'m1':>3} {'heldA':>6} | {'kALS':>4} {'m1':>3} {'hALS':>6} | "
          f"{'kB':>3} {'#m':>3} {'m1':>3} {'cmn':>3} {'=':>1} {'ov':>5} {'units':>5} {'heldB':>6} {'m1':>6} | "
          f"{'kC':>3} {'#m':>3} {'m1':>3} {'cmn':>3} {'=':>1} {'ov':>5} {'heldC':>6} {'m1':>6}")
        for r in true:
            W = r["walks"]
            mr = m1_rec(m1, r)
            kb = kcmp(r, mr, "B", rn)
            kc = kcmp(r, mr, "C", rn)
            w(f"{r['epoch']:>6} {r['net_test_acc']:>6.3f} | "
              f"{W['A'][rn]['k']:>3} {mr['walks']['A'][rn]['k']:>3} {W['A'][rn]['heldout']:>6.3f} | "
              f"{W['ALS'][rn]['k']:>4} {mr['walks']['ALS'][rn]['k']:>3} {W['ALS'][rn]['heldout']:>6.3f} | "
              f"{kb[0]:>3} {kb[1]:>3} {kb[2]:>3} {kb[3]:>3} {'y' if kb[4] else '.':>1} {kb[5]:>5.3f} "
              f"{W['B'][rn]['units']:>5} {W['B'][rn]['heldout']:>6.3f} {mr['walks']['B'][rn]['heldout']:>6.3f} | "
              f"{kc[0]:>3} {kc[1]:>3} {kc[2]:>3} {kc[3]:>3} {'y' if kc[4] else '.':>1} {kc[5]:>5.3f} "
              f"{W['C'][rn]['heldout']:>6.3f} {mr['walks']['C'][rn]['heldout']:>6.3f}")
        w("")
    # ---------------------------------------------------------------- random-order controls
    w("## Random-order controls (5 permutations of the candidates) vs producer order, B and C, at snapshots with controls")
    w(f"{'epoch':>6} {'rule':>4} | B: k_prod k_rand[min..max] held_prod held_rand_mean | C: same")
    for r in true:
        if not r["walks"]["B"]["lt1"]["rand"]:
            continue
        for rn in RULES:
            parts = []
            for f in ("B", "C"):
                W = r["walks"][f][rn]
                ks = [x["k"] for x in W["rand"]]
                hs = [x["heldout"] for x in W["rand"]]
                parts.append(f"{W['k']:>3} [{min(ks):>2}..{max(ks):>2}] {W['heldout']:.3f} {np.mean(hs):.3f}")
            w(f"{r['epoch']:>6} {rn:>4} | " + " | ".join(parts))
    w("")
    # ---------------------------------------------------------------- final net
    fin = true[-1]
    mf = m1_rec(m1, fin)
    rd = fin["read"]
    w(f"## Final net (epoch {fin['epoch']}): net train {fin['net_train_acc_fit']:.4f}, test {fin['net_test_acc']:.4f}, "
      f"e^ = {fin['e_hat']}, m = {rd['m']} candidates ({rd['n_pairs']} pairs, {rd['n_real']} real incl. DC)")
    w("Operators (per g): " + "; ".join(
        f"g={g}: bij={o['bijection']} cycles={o['n_cycles']} conf={o['conf']:.5f} shift_acc={o['shift_acc_hard']:.3f} "
        f"shift_mass={o['shift_mass_soft']:.5f}" for g, o in fin["ops"].items()))
    sc = np.asarray(fin["score"])
    km = np.asarray(rd["k_match"])
    m1sc = np.asarray(mf["score"])
    rel = np.abs(sc - m1sc[km - 1]) / m1sc[km - 1]
    w(f"producer score at matched k vs m1's DFT score: max rel diff {rel.max():.2e}; producer order equal to m1's "
      f"(as matched k): {[int(km[j]) for j in fin['prod_order']] == mf['prod_order']}; unit dominants equal "
      f"(as matched k): {bool(np.array_equal(km[np.asarray(fin['dom'])], np.asarray(mf['dom'])))}; "
      f"median unit concentration {fin['unit_conc_median']:.3f}")
    KB = set(fin["walks"]["B"]["lt1"]["K"])
    th = np.asarray(rd["theta"])
    pe = np.asarray(rd["phase_err"])
    w("Per recovered subspace, producer order. p*theta/2pi = eigenvalue phase of M_g_ref in units of 2pi/p; "
      "expect = fold(k*g_ref mod p); ph_err = max over the 6 g of |restricted phase - true|; * = kept by B/lt1")
    w(f"{'rank':>4} {'j':>3} {'dim':>3} {'k':>3} {'dft_ov':>9} {'xg':>9} {'|lam|':>8} {'p.th/2pi':>9} {'expect':>6} "
      f"{'ph_err':>8} {'score':>6} {'sup':>3}")
    tot = sc.sum()
    gi = 0
    for rank, j in enumerate(fin["prod_order"]):
        k = int(km[j])
        kg = (k * gens[gi]) % P
        exp = min(kg, P - kg)
        w(f"{rank + 1:>4} {j:>3} {rd['cand_dim'][j]:>3} {k:>3} {rd['dft_ov'][j]:>9.6f} {rd['xg'][j]:>9.6f} "
          f"{rd['lam_mod'][j]:>8.5f} {P * th[j, 0] / (2 * np.pi):>9.4f} {exp:>6} {pe[j].max():>8.1e} "
          f"{sc[j] / tot:>6.3f} {fin['support'][j]:>3}{' *' if j in KB else ''}")
    w("")
    w("KEPT K per gate at the final net (recovered basis, as matched DFT k) vs m1:")
    for f in FORMS:
        for rn in RULES:
            W = fin["walks"][f][rn]
            M = mf["walks"][f][rn]
            ks = [x["k"] for x in W["rand"]]
            w(f"  {f:>3} {rn:>4}: k={W['k']:>2} held={W['heldout']:.4f} (m1 k={M['k']:>2} held={M['heldout']:.4f}) "
              f"units={W['units']:>3} equal={set(W['K_match']) == set(M['K'])} rand k=[{','.join(map(str, ks))}] "
              f"K={W['K_match'] if len(W['K_match']) < 48 else 'all 48'}")
    for rn in ("lt1", "m3"):
        w(f"CROSS table, rule {rn}: held-out of each form (columns) at each gate's kept K (rows), recovered basis")
        w(f"  {'gate':>5} {'k':>3} | " + " ".join(f"{f:>6}" for f in FORMS))
        for g in FORMS:
            c = fin["cross"][rn][g]
            w(f"  {g:>5} {fin['walks'][g][rn]['k']:>3} | " + " ".join(f"{c[f]:>6.3f}" for f in FORMS))
    w("")
    # ---------------------------------------------------------------- phase check over the run
    w("## Eigenvalue-phase check (oracle): restricted phase of every 2-D candidate at every g vs the true character "
      "angle at its matched k")
    w(f"{'epoch':>6} {'net_te':>6} {'dft_min':>7} | " + " ".join(f"{'g=' + str(g):>9}" for g in gens) +
      " | max over g (and the fraction of 2-D candidates with err < 1e-3)")
    for r in true:
        if r["epoch"] not in DETAIL_EPOCHS and r["epoch"] != -1:
            continue
        rd = r["read"]
        pe = np.asarray(rd["phase_err"])
        dims = np.asarray(rd["cand_dim"])
        pe2 = pe[dims == 2] if (dims == 2).any() else np.full((1, len(gens)), np.nan)
        w(f"{r['epoch']:>6} {r['net_test_acc']:>6.3f} {rd['dft_ov_min']:>7.4f} | " +
          " ".join(f"{pe2[:, i].max():>9.1e}" for i in range(len(gens))) +
          f" | {pe2.max():.1e} ({(pe2.max(1) < 1e-3).mean():.2f})")
    w("")
    # ---------------------------------------------------------------- per-frequency recovery
    det = [r for r in true if r["epoch"] in DETAIL_EPOCHS]
    w("## Per-DFT-frequency recovery (oracle): best projector overlap of any recovered candidate with pair k; "
      "* = in m1's final K_B (lt1)")
    w(f"{'k':>4} | " + " ".join(f"{r['epoch']:>6}" for r in det))
    K1 = set(mf["walks"]["B"]["lt1"]["K"])
    for k in range(1, 49):
        w(f"{k:>3}{'*' if k in K1 else ' '} | " + " ".join(f"{r['read']['dft_rec'][k - 1]:>6.3f}" for r in det))
    w(f"{'n>.99':>4} | " + " ".join(f"{int((np.asarray(r['read']['dft_rec']) > 0.99).sum()):>6}" for r in det))
    w("")
    w("## Per-candidate DFT overlap quantiles at the detail epochs (min / q10 / median / q90 / max; # > 0.99 of m)")
    for r in det:
        o = np.asarray(r["read"]["dft_ov"])
        q = np.quantile(o, [0, 0.1, 0.5, 0.9, 1.0])
        w(f"{r['epoch']:>6} net_te {r['net_test_acc']:.3f}: " + " / ".join(f"{x:.3f}" for x in q) +
          f";  {int((o > 0.99).sum())} of {len(o)}")
    w("")
    # ---------------------------------------------------------------- falsifiers
    w("## Falsifiers: shuffled-label net (train labels permuted) and untrained inits; held = TRUE-label held-out")
    w(f"{'label':>8} {'net_tr':>6} {'net_te':>6} | " + READ_HDR)
    for r in shuf + init:
        lab = r["label"] if r["run"] != "init" else r["label"].replace("init_", "rinit_")
        w(f"{lab:>8} {r['net_train_acc_fit']:>6.3f} {r['net_test_acc']:>6.3f} | " + read_row(r))
    w("")
    w(f"{'label':>8} | " + " | ".join(f"{f}: k le0/lt1/m3 held(lt1) held(m3)" for f in FORMS) + " | m1 kB, kC (lt1)")
    for r in shuf + init:
        parts = []
        for f in FORMS:
            W = r["walks"][f]
            parts.append(f"{W['le0']['k']:>2}/{W['lt1']['k']:>2}/{W['m3']['k']:>2} "
                         f"{W['lt1']['heldout']:.3f} {W['m3']['heldout']:.3f}")
        mr = m1_rec(m1, r)
        mk = f"{mr['walks']['B']['lt1']['k']:>2}, {mr['walks']['C']['lt1']['k']:>2}" if mr else "  -"
        lab = r["label"] if r["run"] != "init" else r["label"].replace("init_", "rinit_")
        w(f"{lab:>8} | " + " | ".join(parts) + f" | {mk}")
    w("  (rinit_s* = untrained nets, seeds 0..3; the seed-42 init is the 'init' row of each run)")
    w("")
    # ---------------------------------------------------------------- first epochs
    w("## First snapshot epoch at which each series crosses a threshold (true-label run)")

    def first(key_fn, thr):
        for r in true:
            v = key_fn(r)
            if v is not None and v >= thr - 1e-12:
                return r["epoch"]
        return None

    def last_below(key_fn, thr):
        """first epoch after which the series stays >= thr to the end"""
        ep = None
        for r in true:
            v = key_fn(r)
            if v is None or v < thr - 1e-12:
                ep = None
            elif ep is None:
                ep = r["epoch"]
        return ep
    series = {
        "net test acc": lambda r: r["net_test_acc"],
        "all 6 hard maps bijective": lambda r: float(ops_summary(r)[0] == len(r["ops"])),
        "xg_min (certificate)": lambda r: r["read"]["xg_min"],
        "xg_mean (certificate)": lambda r: r["read"]["xg_mean"],
        "dft_ov_min (oracle)": lambda r: r["read"]["dft_ov_min"],
        "dft_ov_mean (oracle)": lambda r: r["read"]["dft_ov_mean"],
        "K_B(lt1) matched == m1": lambda r: _keq(r, m1, "B", "lt1"),
        "K_C(lt1) matched == m1": lambda r: _keq(r, m1, "C", "lt1"),
        "K_B(m3) matched == m1": lambda r: _keq(r, m1, "B", "m3"),
        "K_C(m3) matched == m1": lambda r: _keq(r, m1, "C", "m3"),
        "A(lt1) k == 1": lambda r: float(r["walks"]["A"]["lt1"]["k"] == 1),
    }
    w(f"{'series':>28} " + " ".join(f"{'>=' + str(t):>7}" for t in (0.5, 0.9, 0.99, 1.0)) + "   stays >=0.99 from")
    for name, fn in series.items():
        w(f"{name:>28} " + " ".join(f"{str(first(fn, t)):>7}" for t in (0.5, 0.9, 0.99, 1.0)) +
          f"   {last_below(fn, 0.99)}")
    return "\n".join(L)


def figures(d, m1, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e0"})
    C_NET, C_DFT, C_XG, C_BIJ, C_B, C_C = "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#1baf7a", "#e87ba4"
    true = [r for r in by_run(d, "true") if r["epoch"] >= 0]
    ep = np.array([r["epoch"] for r in true])
    S = {"epoch": ep.tolist(), "net_test": [r["net_test_acc"] for r in true]}
    for key in ("dft_ov_mean", "dft_ov_min", "xg_mean", "xg_min", "dft_rec_min", "phase_err_max", "m", "n_real"):
        S[key] = [r["read"][key] for r in true]
    S["n_bij"] = [ops_summary(r)[0] for r in true]
    for f in FORMS:
        for rn in RULES:
            S[f"k_{f}_{rn}"] = [r["walks"][f][rn]["k"] for r in true]
            S[f"kmatch_{f}_{rn}"] = [r["walks"][f][rn]["K_match"] for r in true]
            S[f"held_{f}_{rn}"] = [r["walks"][f][rn]["heldout"] for r in true]
            S[f"m1_k_{f}_{rn}"] = [m1_rec(m1, r)["walks"][f][rn]["k"] for r in true]
            S[f"m1_held_{f}_{rn}"] = [m1_rec(m1, r)["walks"][f][rn]["heldout"] for r in true]
    S["dft_rec"] = [np.round(r["read"]["dft_rec"], 4).tolist() for r in true]
    S["dft_ov"] = [np.round(r["read"]["dft_ov"], 4).tolist() for r in true]
    S["xg"] = [np.round(r["read"]["xg"], 4).tolist() for r in true]

    fig = plt.figure(figsize=(19, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], height_ratios=[3, 2], hspace=0.28, wspace=0.2)
    for col, (x0, x1, ttl) in enumerate(((None, None, "full run"), (14000, 22000, "zoom: epochs 14k-22k"))):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(ep, S["net_test"], color=C_NET, lw=2.2, label="net held-out acc (logged)", zorder=5)
        ax.plot(ep, S["dft_ov_mean"], color=C_DFT, lw=1.8, label="DFT overlap, mean over candidates (oracle)")
        ax.plot(ep, S["dft_ov_min"], color=C_DFT, lw=1.2, ls=(0, (3, 2)), label="DFT overlap, min (oracle)")
        ax.plot(ep, S["xg_mean"], color=C_XG, lw=1.8, label="cross-g agreement, mean (endogenous)")
        ax.plot(ep, S["xg_min"], color=C_XG, lw=1.2, ls=(0, (3, 2)), label="cross-g agreement, min (endogenous)")
        ax.step(ep, np.asarray(S["n_bij"]) / len(d["gens"]), where="post", color=C_BIJ, lw=1.2,
                label=f"fraction of the {len(d['gens'])} hard maps that are bijections")
        ax.set_ylim(-0.03, 1.03)
        if x0 is not None:
            ax.set_xlim(x0, x1)
        ax.grid(True, axis="y", lw=0.6)
        ax.set_title(f"basis {tag}: operator read from the net ({ttl})", fontsize=10)
        if col == 0:
            ax.set_ylabel("overlap / accuracy")
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        axk = fig.add_subplot(gs[1, col], sharex=ax)
        for f, c in (("B", C_B), ("C", C_C)):
            axk.plot(ep, S[f"k_{f}_lt1"], color=c, lw=1.8, label=f"{f}/lt1 recovered basis: k kept")
            axk.plot(ep, [len(x) for x in S[f"kmatch_{f}_lt1"]], color=c, lw=1.0, ls=(0, (1, 1.5)),
                     label=f"{f}/lt1 recovered: # distinct matched DFT k")
            axk.plot(ep, S[f"m1_k_{f}_lt1"], color=c, lw=1.2, ls=(0, (4, 2)), alpha=0.7,
                     label=f"{f}/lt1 m1 (DFT basis)")
        axk.plot(ep, S["k_A_lt1"], color=C_DFT, lw=1.2, label="A/lt1 recovered basis: k kept (m1: 1)")
        axk.set_xlabel("epoch")
        axk.grid(True, axis="y", lw=0.6)
        if col == 0:
            axk.set_ylabel("kept k")
        else:
            axk.legend(loc="upper right", frameon=False, fontsize=7)
    axh = fig.add_subplot(gs[:, 2])
    R = np.asarray(S["dft_rec"]).T
    im = axh.imshow(R, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1,
                    extent=[ep[0], ep[-1], 0.5, 48.5], interpolation="nearest")
    K1 = sorted(m1_rec(m1, true[-1])["walks"]["B"]["lt1"]["K"])
    axh.set_yticks(K1)
    axh.set_yticklabels([str(k) for k in K1], fontsize=6)
    axh.set_ylabel("DFT frequency k (ticks = m1's final K_B, lt1)")
    axh.set_xlabel("epoch")
    axh.set_title("per-frequency recovery (oracle): best overlap of any\nrecovered candidate with DFT pair k",
                  fontsize=10)
    fig.colorbar(im, ax=axh, fraction=0.05, pad=0.02)
    fig.savefig(os.path.join(HERE, "figures", f"basis_{tag}_read.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(HERE, "results", f"basis_{tag}_summary.json"), "w") as f:
        json.dump(S, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    d, m1 = load(a.tag, a.fetch)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    txt = table(d, m1, a.tag)
    with open(os.path.join(HERE, "figures", f"basis_{a.tag}_table.txt"), "w") as f:
        f.write(txt + "\n")
    figures(d, m1, a.tag)
    print(txt)


if __name__ == "__main__":
    main()
