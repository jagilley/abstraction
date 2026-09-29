"""[preplay/select] THE READ AS THE SELECTOR — does choosing candidate entries by the
endogenous read build a better table than choosing at random, and how close to choosing by the
world? Offline, on banked soundboard state, one step, no loop.

`pp1` (see `NOTES.md`) asked whether the shaped plant's projection RANKS candidate entries as
the world's audition does, and found that it does, weakly but on every arm and every level:
true-vs-wrong AUC 0.61-0.75 against the world's 0.89-0.97, with an unshaped trunk at 0.48-0.61,
and a positive rank correlation WITHIN the true class, so the read is not only reading
grammaticality. That is a statement about a ranking. THIS node asks the consumer's question:
put that ranking in the seat that picks what goes into the table, and audit the table the world
gets. It is the abstraction-supervision consumer in one offline step.

THE SHAPE.

  the candidate pool   `pp1`'s: sampled TRUE entries of the level plus WRONG entries (child-row
                       pairs whose level-1 expansion is off the true table), so a selector has
                       to prefer true over wrong AND, within true, prefer the entries the
                       cell's instances actually want
  the PRICING pool     one fresh pool of the level's era cell, on its own seed family. Every
                       candidate is fired alone as a one-row table and priced there, once, by
                       every selector at once (the world's repair rate, each readout's mean
                       probability, the executor's own DP score)
  the TEST pools       THREE further pools, on a DISJOINT seed family, on which every selected
                       table is auditioned. The world selector therefore never tests on the
                       pool it priced on, and every selector meets the same sampling noise.
                       Gate S-1 asserts the disjointness row by row.
  from scratch         the top-k of the pool by each selector, at k = the arm's committed
                       `n_entries` and at 1/2 and 1/4 of it, against pp1's `rand_k` reference
                       (a matched-size random subset of the TRUE table) and the full true table
  extension            a SMALL base (a quarter of the committed size, because pp1 measured that
                       at the committed sizes the executor's DP never picks 92-99% of added
                       entries) extended by the top-k of the remaining candidates, against the
                       base alone and against the base plus a random k

THE SELECTORS, all ranking ONE pool by ONE price, ties broken by one shared permutation so the
tie-break cannot favour a selector:

    world          the world's repair rate on the pricing pool        THE CEILING
    banked         the arm's banked shaped projection, mean p         THE OBJECT UNDER TEST
    shaped_refit   the same readout refit here through the same core
    frozen         the same readout through `overtone`'s unshaped core   control
    twin           the same readout through a never-trained trunk       floor
    dp_top         the executor's own per-block DP score of the entry   THE FREE PRIOR
    random         a uniform draw at matched count                      THE BASELINE

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/selector.py::gates2
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/selector.py::falsify2
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/selector.py::sweep2 --smoke 1 \\
        --out-tag pp2_smoke --arms s0_sv
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/selector.py::sweep2 \\
        --out-tag pp2
"""

import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay-select", image=image)

REMOTE = "rhm_practice_preplay"
BANK_SB = "rhm_practice_soundboard"
BANK_OV = "rhm_practice_voicing"

LEVELS = (2, 3, 4, 5)
SELECTORS = ("world", "banked", "shaped_refit", "frozen", "twin", "dp_top", "random")
READS = ("banked", "shaped_refit", "frozen", "twin")

# The pools. `n_price` is DELIBERATELY larger than the loop's own audition size: the selector's
# input is a RANKING, and pp1 measured the within-true spread of the world's price at sd
# 0.12-0.22 against a binomial standard error of 0.025 at n = 256. Doubling the pricing pool
# halves that error for a few container-seconds, and the question "is the read a good selector"
# should not be answered against a needlessly noisy ceiling. The TEST pools are kept at the
# loop's own `n_score = 256`, three of them, so the reported spread is the loop's own.
N_PRICE = 512
N_TEST = 256
N_TEST_POOLS = 3
K_POOL = 128                    # candidates per class (true / wrong) in the pool
K_FRACS = (1.0, 0.5, 0.25)      # k as a fraction of the arm's committed n_entries
BASE_FRAC = 0.25                # the extension form's small base, as a fraction of committed
N_BASE_DRAWS = 3

# Seed families, all disjoint from pp1's (5_100_000) and from the run's own audition pools
# (`cfg.seed + 820_000 + ...` for the merge ledger, `+ 900_000 + ...` for commit-then-extend).
PRICE_SEED_BASE = 5_300_000
TEST_SEED_BASE = 5_400_000
DRAW_SEED_BASE = 6_400_000
FIT_SEED_BASE = 7_300_000       # kept at pp1's, so the refit readouts are the same objects


def grade_flat(child_rows, lower_flat, true_flat, s):
    """`MC.grade_table`'s own set arithmetic with the TRUTH SIDE PRECOMPUTED.

    The donor rebuilds both sides on every call, and at L5 the truth side is 205824 tuples of
    32 ints: the pp2 smoke spent 60 of its 76 seconds inside `MC.grade_table` on the
    `true_full` reference alone (NOTES defect 6). Gate S-5 asserts this against the donor on
    real tables at every level.
    """
    arr = np.asarray(child_rows, np.int64).reshape(-1, int(s))
    L = {tuple(int(q) for j in r_ for q in lower_flat[j]) for r_ in arr}
    inter = len(L & true_flat)
    return {"n_learned": len(L), "n_true": len(true_flat), "n_correct": inter,
            "precision": (inter / len(L)) if L else None,
            "recall": (inter / len(true_flat)) if true_flat else 0.0}


def rank_topk(price, tie, k):
    """The top-k by `price`, descending, with `tie` (one shared permutation) as the only
    tie-break. A CONSTANT price therefore reduces exactly to `tie`'s own order, which is what
    the `random` selector is and what gate S-2 asserts."""
    price = np.asarray(price, np.float64)
    order = np.lexsort((np.asarray(tie, np.int64), -price))
    return order[:int(k)]


# --------------------------------------------------------------------------------------- #
# the per-arm job
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=8192)
def select_arm(arm_key, out_tag, smoke=False, levels="", n_price=0, n_test=0, k_pool=0,
               test_pools=0, base_draws=0):
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    # THE READ IS IMPORTED, NOT COPIED, so `preplay.py`'s gate F-2 -- the elementwise check of
    # these exact functions against `VoProjBank` itself, closed at max|dp| = 0.000e+00 on all
    # six arms -- is this node's fidelity gate by identity and not by re-argument.
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    spec = PP.ARMS[arm_key]
    root_sb = os.path.join(DATA_DIR, BANK_SB, spec["tag"], spec["arm"])
    ovs = PP.OVERTONE[spec["ov"]]
    root_ov = os.path.join(DATA_DIR, BANK_OV, ovs["tag"], ovs["arm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want_levels = tuple(int(z) for z in levels.split(",") if z) or LEVELS
    NP_ = int(n_price) or N_PRICE
    NT = int(n_test) or N_TEST
    KP = int(k_pool) or K_POOL
    NTP = int(test_pools) or N_TEST_POOLS
    NBD = int(base_draws) or N_BASE_DRAWS
    if smoke:
        NP_, NT, KP, NTP, NBD = 128, 64, 10, 2, 2
    log("=" * 100)
    log(f"[select] arm {arm_key}: {spec['tag']}/{spec['arm']}   overtone {ovs['tag']}/"
        f"{ovs['arm']}   device {device}   smoke={bool(smoke)}")
    log(f"  levels {want_levels}  n_price {NP_}  n_test {NT} x{NTP}  k_pool {KP}  "
        f"base_draws {NBD}")
    log("=" * 100)

    # ---- the banked state (pp1's loader, same asserts) --------------------------------- #
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m, maxl = int(cfgr["m"]), int(cfgr["max_macro_level"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
    tdim = dim
    assert str(cfgh.get("vo_om_mode")) == "proj" and str(cfgh.get("vo_pj_trunk")) == "live"
    assert bool(cfgh.get("vo_pj_mask")) and str(cfgh.get("vo_pj_root")) == "inter"
    rules = generate_rules_distinct(v, s, depth, m, seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)

    def mk_core():
        return GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                     root_conditioned=False).to(device)

    shaped = mk_core()
    shaped.load_state_dict(blob["core"])
    shaped.eval()
    ovblob = torch.load(os.path.join(root_ov, "vo_heads.pt"), map_location="cpu",
                        weights_only=True)
    frozen = mk_core()
    frozen.load_state_dict(ovblob["core"])
    frozen.eval()
    st_ = torch.get_rng_state()
    torch.manual_seed(PP.TWIN_SEED)
    twin = mk_core()
    torch.set_rng_state(st_)
    twin.eval()

    def fingerprint(c):
        with torch.no_grad():
            return float(sum(float(p_.detach().double().abs().sum()) for p_ in c.parameters()))
    fps = {k_: fingerprint(c_) for k_, c_ in
           (("shaped", shaped), ("frozen", frozen), ("twin", twin))}
    assert abs(fps["shaped"] - fps["frozen"]) > 1e-6, "shaped and frozen cores are identical"
    pj = blob["proj"]
    assert pj is not None and bool(pj["mask"]) and str(pj["root"]) == "inter"
    w_b, mu_b, sd_b = pj["w"], pj["mu"], pj["sd"]

    # ---- F-2b, pp1's banked-scalar check, re-run here --------------------------------- #
    z = np.load(os.path.join(root_sb, "vo_bank.npz"))
    bank = {"x": torch.from_numpy(z["x"].astype(np.int64)),
            "r": torch.from_numpy(z["r"].astype(np.int64)),
            "y": torch.from_numpy(z["y"].astype(np.float32)),
            "blk": torch.from_numpy(z["blk"].astype(np.int64)),
            "spn": torch.from_numpy(z["spn"].astype(np.int64)),
            "h": z["h"].astype(np.float64), "u0": z["u0"].astype(np.float64)}
    hold_idx = np.nonzero(bank["h"] < PP.PJ_HOLD)[0][-PP.PJ_REFRESH_CAP:]
    hi = torch.from_numpy(hold_idx)
    p_hold = PP.pj_predict(shaped, bank["x"][hi], bank["r"][hi], bank["blk"][hi],
                           bank["spn"][hi], w_b, mu_b, sd_b, s, nb, tdim, v, device)
    auc_here = PP.vo_auc(p_hold.numpy(), bank["y"][hi].numpy())
    oms = [q for q in res_j["log"]["vo_om"]
           if isinstance(q, dict) and q.get("hold_auc") is not None]
    auc_banked = float(oms[-1]["hold_auc"])
    drifts = np.array([abs(q["hold_auc_drift"]) for q in oms
                       if q.get("hold_auc_drift") is not None][-60:], dtype=float)
    band = float(max(0.02, 2.0 * np.quantile(drifts, 0.9))) if drifts.size else 0.05
    gate = {"F-2b:hold_auc_here": auc_here, "F-2b:hold_auc_banked": auc_banked,
            "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band}
    log(f"  [F-2b] banked {auc_banked:.6f}  re-read {auc_here:.6f}  "
        f"|d| {abs(auc_here - auc_banked):.4f}  band {band:.4f}")
    assert abs(auc_here - auc_banked) < band, "F-2b FAILED"

    # ---- the three refit readouts, pp1's protocol and pp1's seed ---------------------- #
    trainable = (bank["h"] >= PP.PJ_HOLD)
    tr_all = np.nonzero(trainable & (bank["u0"] < PP.PJ_BOOT))[0]
    va_all = np.nonzero(trainable & (bank["u0"] >= PP.PJ_BOOT))[0]
    frng = np.random.default_rng(FIT_SEED_BASE + int(cfgr["seed"]))
    idx_tr = (np.sort(frng.choice(tr_all, size=PP.PJ_FIT_CAP, replace=False))
              if tr_all.size > PP.PJ_FIT_CAP else tr_all)
    vcap = max(512, PP.PJ_FIT_CAP // 4)
    idx_va = (np.sort(frng.choice(va_all, size=vcap, replace=False))
              if va_all.size > vcap else va_all)
    fits = {}
    for nm, core_ in (("shaped_refit", shaped), ("frozen", frozen), ("twin", twin)):
        fits[nm] = PP.pj_fit(core_, bank, idx_tr, idx_va, s, nb, tdim, v, device,
                             log=log, name=nm)

    def read_all(xf, roots_t, blk0, span):
        out = {}
        for nm in READS:
            if nm == "banked":
                core_, w_, mu_, sd_ = shaped, w_b, mu_b, sd_b
            else:
                core_ = {"shaped_refit": shaped, "frozen": frozen, "twin": twin}[nm]
                f_ = fits[nm]
                w_, mu_, sd_ = f_["w"], f_["mu"], f_["sd"]
            out[nm] = PP.pj_predict(core_, xf, roots_t, blk0, span, w_, mu_, sd_,
                                    s, nb, tdim, v, device).numpy().astype("float32")
        return out

    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {int(e_["level"]): int(e_["n_entries"]) for e_ in res_j["events"]
               if e_.get("kind") == "commit"}
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)
    log(f"  eras {[eras[k_]['name'] for k_ in sorted(eras)]}   commits {commits}")

    out = {"arm": arm_key, "spec": spec, "gate": gate, "fingerprints": fps,
           "fits": {k_: {"lam": q["lam"], "val_auc": q["val_auc"]} for k_, q in fits.items()},
           "cells": {},
           "cfg": {"n_price": NP_, "n_test": NT, "n_test_pools": NTP, "k_pool": KP,
                   "k_fracs": list(K_FRACS), "base_frac": BASE_FRAC, "base_draws": NBD,
                   "levels": list(want_levels), "seed": int(cfgr["seed"]),
                   "rule_seed": int(cfgr["rule_seed"]), "twin_seed": PP.TWIN_SEED,
                   "selectors": list(SELECTORS)}}
    n_aud = 0
    s1 = {}

    for ell in want_levels:
        if ell not in eras:
            continue
        era = eras[ell]
        node = int(era["node"])
        span = s ** (ell - 1)
        blk0 = node * span
        truth_l = truth[ell]
        lower = truth_l["lower"]
        n_true_rows = int(truth_l["child"].shape[0])
        true_flat = {tuple(int(q) for q in r_) for r_ in truth_l["flat"]}
        n_committed = commits.get(ell)
        base_src = "commit"
        if n_committed is None:
            n_committed = int(aud_last.get(str(ell), {}).get("n_entries") or 0)
            base_src = "live_table_last_cycle"
        n_committed = max(2, min(int(n_committed), n_true_rows))
        t_l0 = time.time()
        log("")
        log(f"  --- L{ell} at {era['name']} (node {node}, blk0 {blk0}, span {span})  "
            f"n_true_rows {n_true_rows}  committed {n_committed} ({base_src})")

        # ---- the pools: one to price on, NTP disjoint ones to test on ----------------- #
        pseed = int(cfgr["seed"]) + PRICE_SEED_BASE + 1000 * ell
        pr_roots, pr_x = context_instances(
            rules, {"name": era["name"], "level": ell, "nodes": [node]},
            NP_, s, depth, v, m, seed=pseed)
        tests = []
        for t_ in range(NTP):
            tseed = int(cfgr["seed"]) + TEST_SEED_BASE + 1000 * ell + 17 * t_
            tr_, tx_ = context_instances(
                rules, {"name": era["name"], "level": ell, "nodes": [node]},
                NT, s, depth, v, m, seed=tseed)
            tests.append({"seed": tseed, "roots": tr_, "x_np": tx_,
                          "x": torch.from_numpy(tx_).to(device),
                          "roots_t": torch.from_numpy(tr_).to(device).long()})

        # ---- GATE S-1: the pricing pool and every test pool are DISJOINT --------------- #
        def keyset(roots_, x_):
            a = np.ascontiguousarray(np.concatenate(
                [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
            return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
                       .tolist())
        kp_ = keyset(pr_roots, pr_x)
        ks_ = [keyset(q["roots"], q["x_np"]) for q in tests]
        ov_pt = [len(kp_ & q) for q in ks_]
        ov_tt = [len(ks_[i] & ks_[j]) for i in range(NTP) for j in range(i + 1, NTP)]
        s1[f"L{ell}"] = {"price_seed": pseed, "test_seeds": [q["seed"] for q in tests],
                         "n_price": len(kp_), "n_test": [len(q) for q in ks_],
                         "overlap_price_test": ov_pt, "overlap_test_test": ov_tt}
        assert max(ov_pt + ([0] if not ov_tt else ov_tt)) == 0, (
            f"S-1 FAILED at L{ell}: the pricing and test pools share instances "
            f"{ov_pt} {ov_tt}")
        log(f"      pools: price seed {pseed} n {NP_}; test seeds "
            f"{[q['seed'] for q in tests]} n {NT} each; S-1 overlap {ov_pt} / {ov_tt}")

        # ---- the candidate pool (pp1's construction, its own draw stream) -------------- #
        rng = np.random.default_rng(DRAW_SEED_BASE + int(cfgr["seed"]) * 97 + ell)
        k_true = min(KP, n_true_rows)
        idx_true = np.sort(rng.permutation(n_true_rows)[:k_true])
        n_lower = int(lower["flat"].shape[0])
        wrong = PP.wrong_rows(n_lower, true_flat, lower["flat"], s, KP, rng)
        cands = ([{"cls": "true", "child": [int(q) for q in truth_l["child"][i_]],
                   "src": int(i_)} for i_ in idx_true] +
                 [{"cls": "wrong", "child": [int(q) for q in r_], "src": -1} for r_ in wrong])
        ncand = len(cands)
        tie = rng.permutation(ncand)
        log(f"      pool: {int(k_true)} true + {len(wrong)} wrong = {ncand} candidates "
            f"(n_lower {n_lower})")

        # ---- PRICE every candidate on the pricing pool, once -------------------------- #
        px = torch.from_numpy(pr_x).to(device)
        proots_t = torch.from_numpy(pr_roots).to(device).long()
        price = {k_: np.zeros(ncand) for k_ in SELECTORS}
        price["random"][:] = 0.0
        for ci, cd in enumerate(cands):
            tb = MC.make_table(ell, np.asarray([cd["child"]], np.int64), lower, s)
            mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
            xf, dp = PP.fire_rec(shaped, px, mv, canon, s)
            if ci == 0:
                xa = MC.apply_any(shaped, px, mv, None, canon, depth, v, m, s)
                assert bool((xa == xf).all()), "F-1 FAILED in the pricing pass"
            su, _ = grade(xf.cpu().numpy(), pr_roots, rules, s)
            rd = read_all(xf, proots_t, blk0, span)
            n_aud += 1
            price["world"][ci] = float(su.mean())
            price["dp_top"][ci] = float(dp["top"].mean())
            for nm in READS:
                price[nm][ci] = float(rd[nm].mean())
        cls_arr = np.array([1.0 if q["cls"] == "true" else 0.0 for q in cands])
        log("      priced. true-vs-wrong AUC on the PRICING pool: " +
            "  ".join(f"{k_}={0.0 if PP.vo_auc(price[k_], cls_arr) is None else PP.vo_auc(price[k_], cls_arr):.3f}"
                      for k_ in SELECTORS if k_ != "random"))

        # ---- the test-time audition ---------------------------------------------------- #
        cache = {}

        def audit(child_rows, t_):
            nonlocal n_aud
            arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
            key = (t_, int(arr.shape[0]),
                   hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
            if key in cache:
                return cache[key]
            tb = MC.make_table(ell, arr, lower, s)
            mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
            xf, _ = PP.fire_rec(shaped, tests[t_]["x"], mv, canon, s)
            su, dres = grade(xf.cpu().numpy(), tests[t_]["roots"], rules, s)
            n_aud += 1
            rec = {"e": float(1.0 - su.mean()), "dres": float(dres.mean()),
                   "n_entries": int(arr.shape[0])}
            cache[key] = rec
            return rec

        def grade_cached(child_rows, is_true_full=False):
            """`MC.grade_table`'s own set arithmetic against a truth set built ONCE per level.

            The donor rebuilds BOTH sides on every call, and at L5 the truth side is 205824
            tuples of 32 ints: the smoke spent 60 of its 76 seconds inside `grade_table` on the
            `true_full` reference alone (NOTES defect 6). The arithmetic is identical --
            `MC.grade_table` is `{flats(learned)}` against `{flats(truth)}` -- and gate S-5
            asserts this against the donor on real tables.
            """
            if is_true_full:
                return {"precision": 1.0, "recall": 1.0, "n_correct": len(true_flat)}
            return grade_flat(child_rows, lower["flat"], true_flat, s)

        def score_table(child_rows, sel_idx=None, base_n=0, is_true_full=False):
            """Audition on every test pool and grade the table against the truth."""
            es = [audit(child_rows, t_)["e"] for t_ in range(NTP)]
            n_rows = int(np.asarray(child_rows, np.int64).reshape(-1, s).shape[0])
            g = grade_cached(child_rows, is_true_full=is_true_full)
            d = {"e_mean": float(np.mean(es)), "e_sd": float(np.std(es)),
                 "e": [float(q) for q in es], "n_entries": n_rows,
                 "precision": g["precision"], "recall": g["recall"],
                 "n_correct": g["n_correct"]}
            if sel_idx is not None:
                d["n_sel"] = int(len(sel_idx))
                d["n_wrong_admitted"] = int(sum(1 for i_ in sel_idx
                                                if cands[i_]["cls"] == "wrong"))
                d["n_true_admitted"] = int(len(sel_idx)) - d["n_wrong_admitted"]
                d["base_n"] = int(base_n)
            return d

        ks = sorted({max(2, min(int(round(fr * n_committed)), ncand)) for fr in K_FRACS},
                    reverse=True)
        cell = {"era": era["name"], "node": node, "blk0": blk0, "span": span,
                "n_true_rows": n_true_rows, "n_committed": n_committed,
                "base_src": base_src, "n_cand": ncand, "n_cand_true": int(k_true),
                "n_cand_wrong": len(wrong), "ks": ks, "price_seed": pseed,
                "test_seeds": [q["seed"] for q in tests],
                "price_auc": {k_: PP.vo_auc(price[k_], cls_arr) for k_ in SELECTORS},
                "scratch": {}, "extend": {}, "refs": {}}

        # ---- references --------------------------------------------------------------- #
        cell["refs"]["true_full"] = score_table(truth_l["child"], is_true_full=True)
        cell["refs"]["rand_true_k"] = {}
        for k in ks:
            draws = []
            for d_ in range(NBD):
                keep = np.sort(rng.permutation(n_true_rows)[:min(k, n_true_rows)])
                draws.append(score_table(truth_l["child"][keep]))
            cell["refs"]["rand_true_k"][str(k)] = {
                "draws": draws,
                "e_mean": float(np.mean([q["e_mean"] for q in draws])),
                "e_sd": float(np.std([q["e_mean"] for q in draws]))}

        # ---- FORM 1: from scratch, the top-k of the pool -------------------------------- #
        for k in ks:
            cell["scratch"][str(k)] = {}
            for sel in SELECTORS:
                idx = rank_topk(price[sel], tie, k)
                rows = [cands[i_]["child"] for i_ in idx]
                cell["scratch"][str(k)][sel] = score_table(rows, sel_idx=idx)
            r = cell["scratch"][str(k)]
            log(f"      scratch k={k:<4d} " + "  ".join(
                f"{sel}:{r[sel]['e_mean']:.3f}/w{r[sel]['n_wrong_admitted']}"
                for sel in SELECTORS) +
                f"   [rand_true_k {cell['refs']['rand_true_k'][str(k)]['e_mean']:.3f}]")

        # ---- FORM 2: extension of a small base ------------------------------------------ #
        n_base = max(2, int(round(BASE_FRAC * n_committed)))
        n_base = min(n_base, n_true_rows)
        k_exts = sorted({max(1, n_committed - n_base),
                         max(1, (n_committed - n_base) // 2)}, reverse=True)
        cell["n_base"] = n_base
        cell["k_exts"] = k_exts
        for d_ in range(NBD):
            keep = np.sort(rng.permutation(n_true_rows)[:n_base])
            base_child = [[int(q) for q in r_] for r_ in truth_l["child"][keep]]
            base_flats = {tuple(int(q) for j in r_ for q in lower["flat"][j])
                          for r_ in truth_l["child"][keep]}
            avail = [i_ for i_, cd in enumerate(cands)
                     if tuple(int(q) for j in cd["child"]
                              for q in lower["flat"][j]) not in base_flats]
            blk_ = {"draw": d_, "n_base": n_base, "n_avail": len(avail),
                    "base_alone": score_table(base_child, sel_idx=[], base_n=n_base),
                    "k": {}}
            av = np.array(avail)
            for ke in k_exts:
                ke_ = min(ke, len(avail))
                blk_["k"][str(ke)] = {}
                for sel in SELECTORS:
                    sub = rank_topk(price[sel][av], tie[av], ke_)
                    idx = av[sub]
                    rows = base_child + [cands[i_]["child"] for i_ in idx]
                    blk_["k"][str(ke)][sel] = score_table(rows, sel_idx=idx, base_n=n_base)
            cell["extend"].setdefault("draws", []).append(blk_)
            b = blk_["base_alone"]
            kk = str(k_exts[0])
            log(f"      extend d{d_} base {n_base} (e {b['e_mean']:.3f}) + top-{k_exts[0]}: "
                + "  ".join(f"{sel}:{blk_['k'][kk][sel]['e_mean']:.3f}"
                            f"/w{blk_['k'][kk][sel]['n_wrong_admitted']}"
                            for sel in SELECTORS))

        cell["sec"] = float(time.time() - t_l0)
        out["cells"][str(ell)] = cell
        log(f"      L{ell} done in {cell['sec']:.1f}s   auditions so far {n_aud}")

    out["S-1"] = s1
    out["n_auditions"] = n_aud
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s   auditions {n_aud}   "
        f"peak RSS {out['peak_rss_mb']:.0f} MB   peak GPU {out['peak_gpu_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"arm": arm_key, "sec": out["sec"], "n_aud": n_aud,
            "rss_mb": out["peak_rss_mb"], "gate": gate}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=12600, memory=2048)
def sweep2(out_tag="pp2", smoke=False, arms="", levels="", n_price=0, n_test=0, k_pool=0,
           test_pools=0, base_draws=0):
    """CPU coordinator: the arms across separate containers, as pp1's."""
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    want = [x for x in arms.split(",") if x] or [k for k, q in PP.ARMS.items() if q["record"]]
    args = [(k, out_tag, bool(smoke), levels, int(n_price), int(n_test), int(k_pool),
             int(test_pools), int(base_draws)) for k in want]
    outs = list(select_arm.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"arms": want, "smoke": bool(smoke), "levels": levels,
                             "n_price": int(n_price), "n_test": int(n_test),
                             "k_pool": int(k_pool), "test_pools": int(test_pools),
                             "base_draws": int(base_draws)}) + "\n")
    volume.commit()
    return outs


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def gates2():
    """THIS NODE'S OWN GATES. `preplay.py`'s F-1..F-6 carry over unchanged and are re-run by
    `preplay.py::gates` / `::fidelity_gate`; the read itself is IMPORTED from `preplay`, so F-2
    covers this node by identity. What is new here is the selection machinery."""
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP

    res = {}
    t0 = time.time()

    def tick(msg):
        print(f"[gates2 +{time.time() - t0:6.1f}s] {msg}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tick(f"start, device {device}")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    truth = MC.true_tables(rules, depth, s, v, m, 4)
    tick("rules and true tables built")

    # ---- S-1: the pricing pool and the test pools share no instance ------------------- #
    def keyset(roots_, x_):
        a = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist())
    s1 = {}
    for ell, node in ((2, 12), (3, 6), (4, 3), (5, 1)):
        ctx = {"name": f"L{ell}n{node}", "level": ell, "nodes": [node]}
        pr, px = context_instances(rules, ctx, 256, s, depth, v, m,
                                   seed=PRICE_SEED_BASE + 1000 * ell)
        ts = [context_instances(rules, ctx, 128, s, depth, v, m,
                                seed=TEST_SEED_BASE + 1000 * ell + 17 * t_)
              for t_ in range(3)]
        kp = keyset(pr, px)
        kt = [keyset(a_, b_) for a_, b_ in ts]
        s1[f"L{ell}"] = {"price_test": [len(kp & q) for q in kt],
                         "test_test": [len(kt[0] & kt[1]), len(kt[0] & kt[2]),
                                       len(kt[1] & kt[2])],
                         "n_price_distinct": len(kp)}
        assert max(s1[f"L{ell}"]["price_test"] + s1[f"L{ell}"]["test_test"]) == 0, "S-1 FAILED"
        tick(f"S-1 L{ell} closed")
    res["S-1:pools_disjoint"] = s1

    # ---- S-2: a CONSTANT price reduces exactly to the random selector ----------------- #
    rg = np.random.default_rng(4)
    n = 200
    tie = rg.permutation(n)
    a = rank_topk(np.zeros(n), tie, 37)
    b = rank_topk(np.full(n, 3.14159), tie, 37)
    c = rank_topk(tie.astype(float) * 0.0, tie, 37)
    # `lexsort` returns INDICES ordered by the tie value, i.e. `argsort(tie)[:k]` -- a uniform
    # subset because `tie` is a uniform permutation. The first version of this gate compared
    # against `tie[:k]`, which is the permutation's own first k VALUES and a different object;
    # the gate tripped on the real machinery and the assertion was the thing that was wrong
    # (NOTES defect 4).
    same = bool((a == b).all() and (a == c).all()
                and (a == np.argsort(tie, kind="mergesort")[:37]).all())
    res["S-2:constant_price_is_random"] = same
    assert same, "S-2 FAILED"
    # ... and the top-k really is the top-k: its mean price beats the complement's
    pr = rg.normal(size=n)
    top = rank_topk(pr, tie, 40)
    rest = np.setdiff1d(np.arange(n), top)
    res["S-2:topk_beats_rest"] = {"top": float(pr[top].mean()), "rest": float(pr[rest].mean()),
                                  "min_top_ge_max_rest": bool(pr[top].min() >= pr[rest].max())}
    assert pr[top].min() >= pr[rest].max(), "S-2 FAILED: the top-k is not the top-k"

    # ---- S-3: the selection's own accounting ties to `MC.grade_table` ------------------ #
    s3 = {}
    for ell in (2, 3, 4):
        tl = truth[ell]
        tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
        wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 20,
                           np.random.default_rng(7 + ell))
        nt = min(20, tl["child"].shape[0])
        cands = ([{"cls": "true", "child": [int(q) for q in r_]} for r_ in tl["child"][:nt]] +
                 [{"cls": "wrong", "child": [int(q) for q in r_]} for r_ in wr])
        rgg = np.random.default_rng(13)
        sel = rgg.permutation(len(cands))[:18]
        rows = [cands[i_]["child"] for i_ in sel]
        tb = MC.make_table(ell, np.asarray(rows, np.int64), tl["lower"], s)
        g = MC.grade_table(tb, tl)
        n_true_sel = int(sum(1 for i_ in sel if cands[i_]["cls"] == "true"))
        # `grade_table` counts DISTINCT FLATS, not rows, and the true table carries synonymous
        # rows (L2: 16 child rows, 14 distinct level-1 expansions). The quantity tied to
        # `n_correct` is therefore the number of distinct flats among the selected TRUE
        # candidates; the row count is tied separately. The first version equated `n_correct`
        # with the candidate count and tripped at L2 on the real tables (NOTES defect 5).
        flat_of = {i_: tuple(int(q) for j in cands[i_]["child"]
                             for q in tl["lower"]["flat"][j]) for i_ in sel}
        n_true_flat_sel = len({flat_of[i_] for i_ in sel if cands[i_]["cls"] == "true"})
        s3[f"L{ell}"] = {"n_true_selected": n_true_sel,
                         "n_true_flats_selected": n_true_flat_sel,
                         "grade_n_correct": g["n_correct"],
                         "n_rows": int(tb["child"].shape[0]), "n_sel": int(len(sel))}
        assert g["n_correct"] == n_true_flat_sel, (
            f"S-3 FAILED at L{ell}: the selection holds {n_true_flat_sel} distinct true "
            f"flats, `grade_table` says {g['n_correct']}")
        assert int(tb["child"].shape[0]) == len(sel), "S-3 FAILED: the table is not k rows"
    res["S-3:selection_accounting"] = s3
    tick("S-2 and S-3 closed")

    # ---- S-5: `grade_flat` is `MC.grade_table`, on real tables at every level --------- #
    s5 = {}
    for ell in (2, 3, 4):
        tl = truth[ell]
        tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
        rgg = np.random.default_rng(31 + ell)
        wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 17,
                           rgg)
        rows = ([[int(q) for q in r_] for r_ in
                 tl["child"][np.sort(rgg.permutation(tl["child"].shape[0])[:23])]] +
                [list(r_) for r_ in wr])
        a = MC.grade_table(MC.make_table(ell, np.asarray(rows, np.int64), tl["lower"], s), tl)
        b = grade_flat(rows, tl["lower"]["flat"], tf, s)
        same = all(a[k_] == b[k_] for k_ in ("n_learned", "n_true", "n_correct",
                                             "precision", "recall"))
        s5[f"L{ell}"] = {"donor": a, "here": b, "identical": bool(same)}
        assert same, f"S-5 FAILED at L{ell}: {a} against {b}"
    res["S-5:grade_flat_is_grade_table"] = s5
    tick("S-5 closed")

    # ---- S-4: a table built from the top-k by the WORLD's own price on a pool really is
    #           better on that pool than a random k (the instrument is not inverted) ------ #
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    tl = truth[ell]
    ctx = {"name": f"L{ell}n{node}", "level": ell, "nodes": [node]}
    pr, px = context_instances(rules, ctx, 256, s, depth, v, m, seed=PRICE_SEED_BASE + 77)
    pxt = torch.from_numpy(px).to(device)
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 48,
                       np.random.default_rng(21))
    cands = ([[int(q) for q in r_] for r_ in tl["child"]] + [list(r_) for r_ in wr])
    wp = []
    for cd in cands:
        tb = MC.make_table(ell, np.asarray([cd], np.int64), tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, _ = PP.fire_rec(core, pxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), pr, rules, s)
        wp.append(float(su.mean()))
    wp = np.array(wp)
    tie = np.random.default_rng(2).permutation(len(cands))

    def e_of(idx):
        tb = MC.make_table(ell, np.asarray([cands[i_] for i_ in idx], np.int64),
                           tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, _ = PP.fire_rec(core, pxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), pr, rules, s)
        return float(1.0 - su.mean())
    e_top = e_of(rank_topk(wp, tie, 24))
    e_rnd = float(np.mean([e_of(np.random.default_rng(100 + i).permutation(len(cands))[:24])
                           for i in range(3)]))
    tick(f"S-4 priced {len(cands)} candidates")
    res["S-4:world_selector_beats_random"] = {"e_top": e_top, "e_random": e_rnd}
    assert e_top < e_rnd, "S-4 FAILED: selecting by the world's own price did not help"

    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def falsify2():
    """THE FALSIFICATION HARNESS for this node's gates. `preplay.py::falsify` covers F-1..F-6
    (7/7); these are S-1..S-4."""
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP

    res, n_ok = {}, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    truth = MC.true_tables(rules, depth, s, v, m, 4)

    def keyset(roots_, x_):
        a = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist())

    # ---- S-1 broken: the test pool is drawn on the PRICING pool's own seed ------------- #
    ctx = {"name": "L3n6", "level": 3, "nodes": [6]}
    pr, px = context_instances(rules, ctx, 256, s, depth, v, m, seed=PRICE_SEED_BASE + 3000)
    tr, tx = context_instances(rules, ctx, 256, s, depth, v, m, seed=PRICE_SEED_BASE + 3000)
    ov = len(keyset(pr, px) & keyset(tr, tx))
    res["S-1"] = {"broke": "the test pool is drawn on the pricing pool's own seed",
                  "overlap": ov, "gate_condition_holds": ov == 0}
    assert ov > 0, "S-1 did not trip on a shared seed"
    n_ok += 1

    # ---- S-2 broken: the tie-break is re-drawn per selector instead of shared ---------- #
    rg = np.random.default_rng(4)
    n = 200
    a = rank_topk(np.zeros(n), rg.permutation(n), 37)
    b = rank_topk(np.zeros(n), rg.permutation(n), 37)
    res["S-2"] = {"broke": "each selector draws its own tie-break permutation",
                  "identical": bool((a == b).all()), "gate_condition_holds": bool((a == b).all())}
    assert not bool((a == b).all()), "S-2 did not trip on an unshared tie-break"
    # ... and an INVERTED ranking is not the top-k
    pr_ = rg.normal(size=n)
    tie = rg.permutation(n)
    bad = np.lexsort((tie, pr_))[:40]                      # ascending: the WORST k
    rest = np.setdiff1d(np.arange(n), bad)
    res["S-2b"] = {"broke": "the ranking is ascending, so the bottom-k is taken",
                   "gate_condition_holds": bool(pr_[bad].min() >= pr_[rest].max())}
    assert not (pr_[bad].min() >= pr_[rest].max()), "S-2 did not trip on an inverted ranking"
    n_ok += 2

    # ---- S-3 broken: a WRONG candidate is counted as true ----------------------------- #
    tl = truth[3]
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 20,
                       np.random.default_rng(10))
    rows = [[int(q) for q in r_] for r_ in tl["child"][:10]] + [list(r_) for r_ in wr[:8]]
    tb = MC.make_table(3, np.asarray(rows, np.int64), tl["lower"], s)
    g = MC.grade_table(tb, tl)
    claimed = 18                     # THE BREAK: every selected row claimed as true
    res["S-3"] = {"broke": "the selection counts every admitted row as a true entry",
                  "claimed": claimed, "grade_n_correct": g["n_correct"],
                  "gate_condition_holds": g["n_correct"] == claimed}
    assert g["n_correct"] != claimed, "S-3 did not trip on a miscounted selection"
    n_ok += 1

    # ---- S-4 broken: select by the world's price ASCENDING ---------------------------- #
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    pr2, px2 = context_instances(rules, ctx, 256, s, depth, v, m, seed=PRICE_SEED_BASE + 77)
    pxt = torch.from_numpy(px2).to(device)
    wrn = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 48,
                        np.random.default_rng(21))
    cands = ([[int(q) for q in r_] for r_ in tl["child"]] + [list(r_) for r_ in wrn])
    wp = []
    for cd in cands:
        tb_ = MC.make_table(ell, np.asarray([cd], np.int64), tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb_), device)
        xf, _ = PP.fire_rec(core, pxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), pr2, rules, s)
        wp.append(float(su.mean()))
    wp = np.array(wp)
    tie2 = np.random.default_rng(2).permutation(len(cands))

    def e_of(idx):
        tb_ = MC.make_table(ell, np.asarray([cands[i_] for i_ in idx], np.int64),
                            tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb_), device)
        xf, _ = PP.fire_rec(core, pxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), pr2, rules, s)
        return float(1.0 - su.mean())
    e_bot = e_of(np.lexsort((tie2, wp))[:24])
    e_rnd = float(np.mean([e_of(np.random.default_rng(100 + i).permutation(len(cands))[:24])
                           for i in range(3)]))
    res["S-4"] = {"broke": "the world selector takes the LOWEST-priced candidates",
                  "e_bottom": e_bot, "e_random": e_rnd,
                  "gate_condition_holds": bool(e_bot < e_rnd)}
    assert e_bot >= e_rnd, "S-4 did not trip on an inverted world selector"
    n_ok += 1

    # ---- S-5 broken: the truth side is a SUBSET of the true flats --------------------- #
    tf_full = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    tf_part = set(list(tf_full)[: len(tf_full) // 2])
    rows5 = [[int(q) for q in r_] for r_ in tl["child"][:23]]
    a5 = MC.grade_table(MC.make_table(3, np.asarray(rows5, np.int64), tl["lower"], s), tl)
    b5 = grade_flat(rows5, tl["lower"]["flat"], tf_part, s)
    res["S-5"] = {"broke": "the precomputed truth set holds half the true flats",
                  "donor_n_correct": a5["n_correct"], "here_n_correct": b5["n_correct"],
                  "gate_condition_holds": a5["n_correct"] == b5["n_correct"]}
    assert a5["n_correct"] != b5["n_correct"], "S-5 did not trip on a truncated truth set"
    n_ok += 1

    res["n_gates_falsified"] = n_ok
    print(json.dumps(res, indent=2, default=str))
    return res
