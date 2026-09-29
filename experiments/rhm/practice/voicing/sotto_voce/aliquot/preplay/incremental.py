"""[preplay/pp4] THE READ AS THE **ORDER** FOR THE LOOP'S OWN INCREMENTAL AUDITION.

pp2 put the read in the selector's seat and found that a table's audition is set by which
entries win the executor's argmax, so picking the top-k by any individual price is the wrong
consumer above L2: at L3 the read's table beat the world's own top-k (the world's greedy top-k
is redundant), and at L4 neither helped. The loop's ACTUAL consumer is not a top-k at all. It is
`census_extend` (`soundboard.py` 13346-13427):

    xr, xx = context_instances(rules, era_ctx(era), cfg["n_aud"] = 192, ..., seed=...)
    _, best_e = _aud(base_child)                       # ONE audition of the base
    kept = list(base_child)
    for cand in cands[:cfg["extend_cap"] = 8]:         # the loop's own budget
        _, e_x = _aud(kept + [cand.child])             # ONE audition per candidate
        if e_x <= best_e + cfg["extend_tol"] = 0.0:    # MUST NOT HURT
            kept, best_e = trial, e_x                  # the base GROWS on admission
        else:
            rejected += 1

and the loop walks the candidates in the miner's own support order. **The question: with that
gate fixed, does walking in the READ's order reach a given table quality with fewer auditions
than a random order, and how close to the world's own order?**

THE GATE IS THE LOOP'S, VERBATIM, AND IS ASSERTED AGAINST A LITERAL TRANSCRIPTION OF IT (G-1a)
AND AGAINST A HAND-COMPUTED SYNTHETIC CASE (G-1b). What varies between arms of this node is only
the ORDER in which the candidates are offered.

TWO SETTINGS.

  (a) pp2's extension setting. A quarter-size random base drawn from the TRUE table, candidates
      the remaining true entries plus wrong entries, walked IN FULL so the loop's cap of 8 is
      one point on a budget axis rather than the whole experiment.
  (b) THE LEARNER'S OWN ROWS, from pp3's replayed operative tables (`learner_tables.py`), whose
      false rows are the ones its own miner produced. Base EMPTY -- build the table from nothing
      by incremental audition -- so the walk is the whole story. Levels 2..4 (L5's operative
      table is not reconstructible; see `learner_tables.py`'s docstring).

THREE DISJOINT POOL FAMILIES, gate G-2:
    PRICING   n = 512, pp2's family (5_300_000) -- where every candidate's price comes from
    GATE      n = 192, the loop's own `n_aud`, THIS node's family (5_500_000). One gate pool
              per (level, repeat), shared by every order in that repeat, so the orders differ by
              nothing but the order. Nothing is admitted on any other pool.
    TEST      n = 256 x3, pp2's family (5_400_000) -- where the current table's error is
              REPORTED and on which nothing is ever admitted.

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::gates4
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::falsify4
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::sweep4 \\
        --out-tag pp4_smoke --arms s0_sv --smoke 1
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::sweep4 \\
        --out-tag pp4
"""

import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay-incremental", image=image)

REMOTE = "rhm_practice_preplay"
BANK_SB = "rhm_practice_soundboard"
BANK_OV = "rhm_practice_voicing"

# The loop's own literals (`soundboard.py` config on these arms).
N_AUD = 192                     # cfg["n_aud"] -- the gate pool
EXTEND_CAP = 8                  # cfg["extend_cap"] -- the loop's budget, one point on our axis
EXTEND_TOL = 0.0                # cfg["extend_tol"] -- must not hurt
TOL_SENS = 0.01                 # the sensitivity pass (~2 instances of 192)

N_PRICE = 512                   # pp2's pricing pool
N_TEST = 256                    # pp2's test pools
N_TEST_POOLS = 3
K_POOL_A = 24                   # setting (a): candidates per class
K_POOL_B = 96                   # setting (b): cap on the operative rows walked at a level
N_REPEATS = 2                   # fresh gate pools
N_RAND_ORDERS = 2
BUDGETS = (8, 16, 32)           # plus the full walk, appended per cell
LEVELS_A = (2, 3, 4, 5)
LEVELS_B = (2, 3, 4)            # `learner_tables.LEVELS_OK`

PRICE_SEED_BASE = 5_300_000     # pp2's
TEST_SEED_BASE = 5_400_000      # pp2's
GATE_SEED_BASE = 5_500_000      # THIS node's, disjoint from both
DRAW_SEED_BASE = 6_500_000
FIT_SEED_BASE = 7_300_000       # pp1's, so the frozen/twin controls are the same objects

ORDERS_A = ("world", "banked", "frozen", "twin", "dp_top", "rand0", "rand1")
ORDERS_B = ("world", "banked", "dp_top", "build", "rand0", "rand1")
READS = ("banked", "shaped_refit", "frozen", "twin")


def order_by(price, tie):
    """The full walk order: descending price, the ONE shared tie-break permutation as the only
    secondary key. A constant price therefore reduces exactly to the tie order, which is what a
    `rand*` order is and what gate G-3 asserts."""
    return np.lexsort((np.asarray(tie, np.int64), -np.asarray(price, np.float64)))


def census_walk(aud_fn, base_child, cand_child, order_idx, tol=EXTEND_TOL, budget=None,
                on_change=None):
    """`census_extend`'s admission loop (`soundboard.py` 13396-13414), with the candidates
    offered in `order_idx` instead of the miner's own order and with no cap unless `budget`
    says so. `aud_fn(child_rows) -> e` is ONE audition on the gate pool; it is called exactly
    once for the base and once per candidate offered, which is what the loop pays.

    `on_change(n_aud_so_far, kept)` is called after the base audition and after every ADMISSION
    (and only then: a rejection leaves the table untouched, so its test error is the previous
    one carried forward). Returns the record.
    """
    n_aud = 1
    best_e = aud_fn(base_child)
    kept = [list(map(int, r_)) for r_ in base_child]
    rec = {"e_base_gate": float(best_e), "admitted": [], "rejected": 0,
           "steps": [], "n_auditions": n_aud}
    if on_change is not None:
        on_change(n_aud, kept)
    walk = order_idx if budget is None else order_idx[:int(budget)]
    for step, ci in enumerate(walk):
        e_x = aud_fn(kept + [list(map(int, cand_child[ci]))])
        n_aud += 1
        admit = bool(e_x <= best_e + tol)
        if admit:
            kept = kept + [list(map(int, cand_child[ci]))]
            best_e = e_x
            rec["admitted"].append(int(ci))
        else:
            rec["rejected"] += 1
        rec["steps"].append({"step": int(step), "cand": int(ci), "e_gate": float(e_x),
                             "admit": admit, "n_aud": int(n_aud),
                             "n_kept": int(len(kept))})
        if admit and on_change is not None:
            on_change(n_aud, kept)
    rec["n_auditions"] = n_aud
    rec["e_gate_final"] = float(best_e)
    rec["n_kept"] = int(len(kept))
    rec["kept"] = kept
    return rec


# --------------------------------------------------------------------------------------- #
# the per-arm job
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=8192)
def incremental_arm(arm_key, out_tag, smoke=False, levels="", settings="ab", n_price=0,
                    n_gate=0, n_test=0, k_pool=0, repeats=0):
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import learner_tables as LT

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
    want_levels = tuple(int(z) for z in levels.split(",") if z)
    NP_ = int(n_price) or N_PRICE
    NG = int(n_gate) or N_AUD
    NT = int(n_test) or N_TEST
    KP = int(k_pool) or K_POOL_A
    NR = int(repeats) or N_REPEATS
    NTP = N_TEST_POOLS
    kpb = K_POOL_B
    if smoke:
        NP_, NG, NT, KP, NR, NTP, kpb = 128, 64, 64, 6, 1, 2, 12
    log("=" * 100)
    log(f"[pp4] arm {arm_key}: {spec['tag']}/{spec['arm']}   device {device}  "
        f"smoke={bool(smoke)}  settings={settings}")
    log(f"  n_price {NP_}  n_gate {NG} (loop's n_aud {N_AUD})  n_test {NT} x{NTP}  "
        f"k_pool_a {KP}/class  k_pool_b {kpb}  repeats {NR}  tol {EXTEND_TOL}")
    log("=" * 100)

    # ---- the banked state (pp1/pp2's loader) ------------------------------------------ #
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m, maxl = int(cfgr["m"]), int(cfgr["max_macro_level"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
    assert int(cfgr["n_aud"]) == N_AUD and int(cfgr["extend_cap"]) == EXTEND_CAP
    assert float(cfgr["extend_tol"]) == EXTEND_TOL
    assert str(cfgh.get("vo_om_mode")) == "proj" and str(cfgh.get("vo_pj_trunk")) == "live"
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
    pj = blob["proj"]
    w_b, mu_b, sd_b = pj["w"], pj["mu"], pj["sd"]

    z = np.load(os.path.join(root_sb, "vo_bank.npz"))
    bank = {"x": torch.from_numpy(z["x"].astype(np.int64)),
            "r": torch.from_numpy(z["r"].astype(np.int64)),
            "y": torch.from_numpy(z["y"].astype(np.float32)),
            "blk": torch.from_numpy(z["blk"].astype(np.int64)),
            "spn": torch.from_numpy(z["spn"].astype(np.int64)),
            "h": z["h"].astype(np.float64), "u0": z["u0"].astype(np.float64)}
    hold_idx = np.nonzero(bank["h"] < PP.PJ_HOLD)[0][-PP.PJ_REFRESH_CAP:]
    hi = torch.from_numpy(hold_idx)
    auc_here = PP.vo_auc(
        PP.pj_predict(shaped, bank["x"][hi], bank["r"][hi], bank["blk"][hi], bank["spn"][hi],
                      w_b, mu_b, sd_b, s, nb, dim, v, device).numpy(),
        bank["y"][hi].numpy())
    oms = [q for q in res_j["log"]["vo_om"]
           if isinstance(q, dict) and q.get("hold_auc") is not None]
    auc_banked = float(oms[-1]["hold_auc"])
    drifts = np.array([abs(q["hold_auc_drift"]) for q in oms
                       if q.get("hold_auc_drift") is not None][-60:], dtype=float)
    band = float(max(0.02, 2.0 * np.quantile(drifts, 0.9))) if drifts.size else 0.05
    gate = {"F-2b:hold_auc_here": auc_here, "F-2b:hold_auc_banked": auc_banked,
            "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band}
    assert abs(auc_here - auc_banked) < band, "F-2b FAILED"
    log(f"  [F-2b] banked {auc_banked:.4f}  re-read {auc_here:.4f}  band {band:.4f}")

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
        fits[nm] = PP.pj_fit(core_, bank, idx_tr, idx_va, s, nb, dim, v, device)

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
                                    s, nb, dim, v, device).numpy().astype("float32")
        return out

    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {int(e_["level"]): int(e_["n_entries"]) for e_ in res_j["events"]
               if e_.get("kind") == "commit"}
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)

    # ---- setting (b)'s tables: pp3's replay, gated ------------------------------------ #
    lt_tables, lt_diags, lt_gates = None, None, []
    if "b" in settings:
        import gzip
        ep = os.path.join(root_sb, "entry.json.gz")
        ent_last = (json.load(gzip.open(ep, "rt"))[-1] if os.path.exists(ep)
                    else {"operative_mask": {}})
        lvb = tuple(q for q in LT.LEVELS_OK if (not want_levels or q in want_levels))
        lt_tables, lt_diags, lt_gates = LT.reconstruct(
            res_j, ent_last, truth=truth, levels=lvb, log=(lambda *a: None),
            arm_key=arm_key)
        bad = [g["gate"] for g in lt_gates if not g["pass"]]
        assert not bad, f"pp3's reconstruction did not gate here: {bad}"
        # `reconstruct` returns the whole chain INCLUDING level 1 (`MC.base_table`), which has
        # no diag and is not a level this node walks; the levels of record are the diags'.
        lvl_b = sorted(q for q in lt_diags if q in LEVELS_B)
        sizes_b = {f"L{e}": int(lt_tables[e]["child"].shape[0]) for e in lvl_b}
        ntrue_b = {f"L{e}": int(sum(lt_diags[e].get("truth_row") or [])) for e in lvl_b}
        log(f"  setting (b): operative tables reconstructed and gated "
            f"({len(lt_gates)} gates, all pass); sizes {sizes_b}  true rows {ntrue_b}")

    out = {"arm": arm_key, "spec": spec, "gate": gate, "settings": settings,
           "cfg": {"n_price": NP_, "n_gate": NG, "n_test": NT, "n_test_pools": NTP,
                   "k_pool_a": KP, "k_pool_b": kpb, "repeats": NR, "tol": EXTEND_TOL,
                   "tol_sens": TOL_SENS, "budgets": list(BUDGETS),
                   "extend_cap": EXTEND_CAP, "n_aud": N_AUD,
                   "seed": int(cfgr["seed"]), "rule_seed": int(cfgr["rule_seed"])},
           "lt_gates": [{k_: g[k_] for k_ in ("gate", "pass", "what")} for g in lt_gates],
           "cells": {}}
    n_aud_total = 0
    g2 = {}

    def keyset(roots_, x_):
        a = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist())

    # ===================================================================================== #
    for setting in [q for q in ("a", "b") if q in settings]:
        lvls = LEVELS_A if setting == "a" else LEVELS_B
        if want_levels:
            lvls = tuple(q for q in lvls if q in want_levels)
        for ell in lvls:
            if ell not in eras:
                continue
            if setting == "b" and (lt_diags is None or ell not in lt_diags):
                continue
            era = eras[ell]
            node = int(era["node"])
            span = s ** (ell - 1)
            blk0 = node * span
            t_l0 = time.time()
            log("")
            log(f"  === setting ({setting}) L{ell} at {era['name']} "
                f"(node {node}, blk0 {blk0}, span {span})")

            if setting == "a":
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
                rng = np.random.default_rng(DRAW_SEED_BASE + int(cfgr["seed"]) * 97 + ell)
                k_true = min(KP, n_true_rows)
                idx_true = np.sort(rng.permutation(n_true_rows)[:k_true])
                wrong = PP.wrong_rows(int(lower["flat"].shape[0]), true_flat,
                                      lower["flat"], s, KP, rng)
                cands = ([{"cls": "true", "child": [int(q) for q in truth_l["child"][i_]]}
                          for i_ in idx_true] +
                         [{"cls": "wrong", "child": [int(q) for q in r_]} for r_ in wrong])
                n_base = max(2, min(int(round(0.25 * n_committed)), n_true_rows))
                orders_want = ORDERS_A
            else:
                tl = lt_tables[ell]
                lower = tl["lower"]
                truth_l = truth[ell]
                true_flat = {tuple(int(q) for q in r_) for r_ in truth_l["flat"]}
                trow = lt_diags[ell].get("truth_row")
                n_rows = int(tl["child"].shape[0])
                rng = np.random.default_rng(DRAW_SEED_BASE + 500_000
                                            + int(cfgr["seed"]) * 97 + ell)
                take = np.arange(n_rows)
                if n_rows > kpb:
                    # a random subsample, BUILD ORDER PRESERVED among those taken, so the
                    # `build` order is still the learner's own emission order restricted
                    take = np.sort(rng.permutation(n_rows)[:kpb])
                cands = [{"cls": ("true" if (trow and trow[int(i_)]) else "wrong"),
                          "child": [int(q) for q in tl["child"][i_]], "row": int(i_)}
                         for i_ in take]
                n_base = 0
                base_src = "empty"
                n_committed = n_rows
                orders_want = ORDERS_B
            ncand = len(cands)
            cand_child = [c["child"] for c in cands]
            cls_arr = np.array([1.0 if c["cls"] == "true" else 0.0 for c in cands])
            tie = rng.permutation(ncand)
            log(f"      candidates {ncand} "
                f"({int(cls_arr.sum())} true / {int((1 - cls_arr).sum())} wrong), "
                f"base {n_base} ({base_src})")

            # ---- the three pool families ---------------------------------------------- #
            off = 0 if setting == "a" else 300_000
            pseed = int(cfgr["seed"]) + PRICE_SEED_BASE + off + 1000 * ell
            pr_roots, pr_x = context_instances(
                rules, {"name": era["name"], "level": ell, "nodes": [node]},
                NP_, s, depth, v, m, seed=pseed)
            tests = []
            for t_ in range(NTP):
                ts = int(cfgr["seed"]) + TEST_SEED_BASE + off + 1000 * ell + 17 * t_
                tr_, tx_ = context_instances(
                    rules, {"name": era["name"], "level": ell, "nodes": [node]},
                    NT, s, depth, v, m, seed=ts)
                tests.append({"seed": ts, "roots": tr_,
                              "x": torch.from_numpy(tx_).to(device), "x_np": tx_})
            gates_ = []
            for rp in range(NR):
                gs = int(cfgr["seed"]) + GATE_SEED_BASE + off + 1000 * ell + 29 * rp
                gr_, gx_ = context_instances(
                    rules, {"name": era["name"], "level": ell, "nodes": [node]},
                    NG, s, depth, v, m, seed=gs)
                gates_.append({"seed": gs, "roots": gr_,
                               "x": torch.from_numpy(gx_).to(device), "x_np": gx_})
            # ---- GATE G-2: the three families are pairwise disjoint -------------------- #
            kp_ = keyset(pr_roots, pr_x)
            kt_ = [keyset(q["roots"], q["x_np"]) for q in tests]
            kg_ = [keyset(q["roots"], q["x_np"]) for q in gates_]
            ov = ([len(kp_ & q) for q in kt_] + [len(kp_ & q) for q in kg_] +
                  [len(a_ & b_) for a_ in kt_ for b_ in kg_] +
                  [len(kt_[i] & kt_[j]) for i in range(NTP) for j in range(i + 1, NTP)] +
                  [len(kg_[i] & kg_[j]) for i in range(NR) for j in range(i + 1, NR)])
            g2[f"{setting}L{ell}"] = {"price_seed": pseed,
                                      "test_seeds": [q["seed"] for q in tests],
                                      "gate_seeds": [q["seed"] for q in gates_],
                                      "max_overlap": int(max(ov)) if ov else 0}
            assert max(ov) == 0 if ov else True, (
                f"G-2 FAILED at ({setting}) L{ell}: pool families overlap {ov}")
            log(f"      pools: price {pseed} n{NP_} | gate {[q['seed'] for q in gates_]} "
                f"n{NG} | test {[q['seed'] for q in tests]} n{NT}  G-2 max overlap "
                f"{max(ov) if ov else 0}")

            # ---- price every candidate on the PRICING pool ----------------------------- #
            px = torch.from_numpy(pr_x).to(device)
            proots_t = torch.from_numpy(pr_roots).to(device).long()
            price = {k_: np.zeros(ncand) for k_ in
                     ("world", "dp_top", "build") + tuple(READS)}
            price["build"] = -np.arange(ncand, dtype=float)      # the emission order itself
            for ci, cd in enumerate(cands):
                tb = MC.make_table(ell, np.asarray([cd["child"]], np.int64), lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, px, mv, canon, s)
                su, _ = grade(xf.cpu().numpy(), pr_roots, rules, s)
                rd = read_all(xf, proots_t, blk0, span)
                n_aud_total += 1
                price["world"][ci] = float(su.mean())
                price["dp_top"][ci] = float(dp["top"].mean())
                for nm in READS:
                    price[nm][ci] = float(rd[nm].mean())
            log("      priced. true-vs-wrong AUC: " + "  ".join(
                f"{k_}={0.0 if PP.vo_auc(price[k_], cls_arr) is None else PP.vo_auc(price[k_], cls_arr):.3f}"
                for k_ in ("world", "banked", "frozen", "twin", "dp_top")))

            # ---- the auditions --------------------------------------------------------- #
            gcache, tcache = {}, {}

            def _fire_e(child_rows, pool):
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                if arr.shape[0] == 0:
                    # `census`'s own empty-table case: no move is applied, the pool is graded
                    # as it stands (`soundboard.py` 11172-11176, `keep_case == "absent"`).
                    su, _ = grade(pool["x_np"], pool["roots"], rules, s)
                    return float(1.0 - su.mean())
                tb = MC.make_table(ell, arr, lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, _ = PP.fire_rec(shaped, pool["x"], mv, canon, s)
                su, _ = grade(xf.cpu().numpy(), pool["roots"], rules, s)
                return float(1.0 - su.mean())

            def aud_gate(child_rows, rp):
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = ("g", rp, int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in gcache:
                    return gcache[key]
                e = _fire_e(arr, gates_[rp])
                n_aud_total += 1
                gcache[key] = e
                return e

            def test_e(child_rows):
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = (int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in tcache:
                    return tcache[key]
                es = [_fire_e(arr, tests[t_]) for t_ in range(NTP)]
                n_aud_total += NTP
                d = {"mean": float(np.mean(es)), "sd": float(np.std(es)),
                     "e": [float(q) for q in es]}
                tcache[key] = d
                return d

            def test_e1(child_rows):
                """THE CURVE'S test error, on TEST POOL 0 ONLY. The curve is evaluated after
                every admission and the gate admits most candidates at `tol = 0` (a candidate
                that does not strictly hurt is kept), so a three-pool curve would triple the
                round's cost for a series that is read as a shape. The BUDGET points and the
                FINAL are evaluated on all three pools (`test_e`); the "auditions to come
                within 0.02" statistic is computed on this one-pool series against the world
                order's own one-pool final, so both sides of that comparison carry the same
                pool and the same noise."""
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = ("1", int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in tcache:
                    return tcache[key]
                e = _fire_e(arr, tests[0])
                n_aud_total += 1
                tcache[key] = e
                return e

            def n_wrong(rows):
                st_ = {tuple(map(int, r_)) for r_ in rows}
                return int(sum(1 for c in cands
                               if c["cls"] == "wrong" and tuple(c["child"]) in st_))

            cell = {"setting": setting, "era": era["name"], "node": node, "span": span,
                    "n_cand": ncand, "n_cand_true": int(cls_arr.sum()),
                    "n_cand_wrong": int((1 - cls_arr).sum()),
                    "n_base": n_base, "base_src": base_src, "n_committed": n_committed,
                    "price_auc": {k_: PP.vo_auc(price[k_], cls_arr)
                                  for k_ in ("world", "banked", "frozen", "twin", "dp_top")},
                    "pools": g2[f"{setting}L{ell}"], "repeats": []}

            for rp in range(NR):
                if setting == "a":
                    keep = np.sort(np.random.default_rng(
                        DRAW_SEED_BASE + 7 * rp + 13 * ell
                        + int(cfgr["seed"])).permutation(n_true_rows)[:n_base])
                    base_child = [[int(q) for q in r_] for r_ in truth_l["child"][keep]]
                    base_flats = {tuple(int(q) for j in r_ for q in lower["flat"][j])
                                  for r_ in truth_l["child"][keep]}
                    live = [i_ for i_ in range(ncand)
                            if tuple(int(q) for j in cands[i_]["child"]
                                     for q in lower["flat"][j]) not in base_flats]
                else:
                    base_child = []
                    live = list(range(ncand))
                rep = {"repeat": rp, "gate_seed": gates_[rp]["seed"],
                       "n_base": len(base_child), "n_live": len(live),
                       "e_base_test": test_e(base_child), "orders": {}}
                lv = np.array(live)
                for nm in orders_want:
                    if nm.startswith("rand"):
                        rr = np.random.default_rng(DRAW_SEED_BASE + 991 * ell
                                                   + 37 * int(nm[4:]) + 3 * rp
                                                   + int(cfgr["seed"]))
                        pr_ = np.zeros(ncand)
                        tie_r = rr.permutation(ncand)
                        oi = order_by(pr_[lv], tie_r[lv])
                    else:
                        oi = order_by(price[nm][lv], tie[lv])
                    oi = lv[oi]
                    curve = []

                    def _rec(na, kept, _c=curve):
                        _c.append({"n_aud": int(na), "n_kept": int(len(kept)),
                                   "e_test1": test_e1(kept), "n_wrong": n_wrong(kept)})
                    _pos = {int(c_): i_ for i_, c_ in enumerate(oi)}
                    r = census_walk(lambda cr, _rp=rp: aud_gate(cr, _rp), base_child,
                                    cand_child, oi, tol=EXTEND_TOL, on_change=_rec)
                    r["curve"] = curve
                    r["order"] = [int(q) for q in oi]
                    r["n_wrong_kept"] = n_wrong(r["kept"])
                    r["e_test_final"] = test_e(r["kept"])
                    # the budget points, from the curve (a rejection leaves the table alone)
                    bud = {}
                    for b in list(BUDGETS) + [len(oi)]:
                        na = 1 + int(b)
                        pt = [q for q in curve if q["n_aud"] <= na]
                        last = pt[-1] if pt else curve[0]
                        rows_b = base_child + [cand_child[i_] for i_ in r["admitted"]
                                               if _pos.get(int(i_), 10 ** 9) < int(b)]
                        t3 = test_e(rows_b)
                        bud[str(b)] = {"e_test": t3["mean"], "e_test_sd": t3["sd"],
                                       "e_test1": last["e_test1"], "n_kept": len(rows_b),
                                       "n_wrong": n_wrong(rows_b), "n_aud": na}
                    r["budget"] = bud
                    # THE UNGATED COMPARISON: pp2's selector, same read, same budget
                    if not nm.startswith("rand") and nm != "build":
                        ung = {}
                        for b in list(BUDGETS) + [len(oi)]:
                            rows = base_child + [cand_child[i_] for i_ in oi[:int(b)]]
                            t = test_e(rows)
                            ung[str(b)] = {"e_test": t["mean"], "n_kept": len(rows),
                                           "n_wrong": n_wrong(rows)}
                        r["ungated"] = ung
                    r.pop("kept", None)
                    r.pop("steps", None)
                    rep["orders"][nm] = r
                # ---- the tolerance sensitivity, one repeat, two orders ----------------- #
                if rp == 0:
                    rep["tol_sens"] = {}
                    for nm in ("banked", "rand0"):
                        if nm not in orders_want:
                            continue
                        if nm.startswith("rand"):
                            rr = np.random.default_rng(DRAW_SEED_BASE + 991 * ell
                                                       + 37 * int(nm[4:]) + 3 * rp
                                                       + int(cfgr["seed"]))
                            oi = lv[order_by(np.zeros(ncand)[lv], rr.permutation(ncand)[lv])]
                        else:
                            oi = lv[order_by(price[nm][lv], tie[lv])]
                        r2 = census_walk(lambda cr, _rp=rp: aud_gate(cr, _rp), base_child,
                                         cand_child, oi, tol=TOL_SENS)
                        r2["n_wrong_kept"] = n_wrong(r2["kept"])
                        r2["e_test_final"] = test_e(r2["kept"])
                        r2.pop("kept", None)
                        r2.pop("steps", None)
                        rep["tol_sens"][nm] = r2
                cell["repeats"].append(rep)
                wf = {nm: rep["orders"][nm]["e_test_final"]["mean"] for nm in orders_want}
                log(f"      rp{rp} base_e_test {rep['e_base_test']['mean']:.3f}  final: " +
                    "  ".join(f"{nm}:{wf[nm]:.3f}/k{rep['orders'][nm]['n_kept']}"
                              f"/w{rep['orders'][nm]['n_wrong_kept']}" for nm in orders_want))
                log(f"           at budget 8:  " + "  ".join(
                    f"{nm}:{rep['orders'][nm]['budget']['8']['e_test']:.3f}"
                    for nm in orders_want))
            cell["sec"] = float(time.time() - t_l0)
            out["cells"][f"{setting}L{ell}"] = cell
            log(f"      cell done in {cell['sec']:.1f}s   auditions so far {n_aud_total}")

    out["G-2"] = g2
    out["n_auditions"] = n_aud_total
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s   auditions {n_aud_total}   "
        f"peak RSS {out['peak_rss_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"arm": arm_key, "sec": out["sec"], "n_aud": n_aud_total,
            "rss_mb": out["peak_rss_mb"], "gate": gate}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=12600, memory=2048)
def sweep4(out_tag="pp4", smoke=False, arms="", levels="", settings="ab", n_price=0,
           n_gate=0, n_test=0, k_pool=0, repeats=0):
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    want = [x for x in arms.split(",") if x] or [k for k, q in PP.ARMS.items() if q["record"]]
    args = [(k, out_tag, bool(smoke), levels, settings, int(n_price), int(n_gate),
             int(n_test), int(k_pool), int(repeats)) for k in want]
    outs = list(incremental_arm.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"arms": want, "smoke": bool(smoke), "levels": levels,
                             "settings": settings}) + "\n")
    volume.commit()
    return outs


# --------------------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------------------- #

def _census_literal(aud_fn, base_child, cands_child, tol, cap):
    """`soundboard.py::census_extend` 13396-13414, TRANSCRIBED, with its own `extend_cap` and
    its own `_aud`. The reference `census_walk` is asserted against (G-1a)."""
    _, best_e = None, aud_fn(base_child)
    kept = list(base_child)
    admitted, rejected, n = [], 0, 1
    for i, cand in enumerate(cands_child[:cap]):
        trial = kept + [list(map(int, cand))]
        e_x = aud_fn(trial)
        n += 1
        if e_x <= best_e + tol:
            kept, best_e = trial, e_x
            admitted.append(i)
        else:
            rejected += 1
    return {"admitted": admitted, "rejected": rejected, "n": n, "best_e": best_e,
            "n_kept": len(kept)}


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def gates4():
    """This node's gates. `preplay.py`'s F-1..F-6 and `selector.py`'s S-1..S-5 carry over; the
    read and the fire are IMPORTED, so their exact gates cover this node by identity."""
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
        print(f"[gates4 +{time.time() - t0:6.1f}s] {msg}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    truth = MC.true_tables(rules, depth, s, v, m, 4)
    tick("built")

    # ---- G-1b: a SYNTHETIC case with hand-computed admissions ------------------------- #
    # `_aud` returns a fixed sequence. base 0.50; then 0.40 (admit, best->0.40),
    # 0.45 (reject), 0.40 (admit, tie at tol 0 -> `<=` admits), 0.50 (reject),
    # 0.10 (admit, best->0.10), 0.11 (reject). Expected admissions [0, 2, 4], rejected 3.
    seq = [0.50, 0.40, 0.45, 0.40, 0.50, 0.10, 0.11]
    box = {"i": 0}

    def stub(_rows):
        e = seq[box["i"]]
        box["i"] += 1
        return e
    r = census_walk(stub, [[0, 0]], [[1, 1]] * 6, list(range(6)), tol=0.0)
    res["G-1b:synthetic"] = {"admitted": r["admitted"], "rejected": r["rejected"],
                             "n_auditions": r["n_auditions"],
                             "e_gate_final": r["e_gate_final"]}
    assert r["admitted"] == [0, 2, 4], f"G-1b FAILED: {r['admitted']} != [0, 2, 4]"
    assert r["rejected"] == 3 and r["n_auditions"] == 7, "G-1b FAILED: counts"
    assert abs(r["e_gate_final"] - 0.10) < 1e-12, "G-1b FAILED: best_e"
    tick("G-1b closed")

    # ---- G-1a: against a literal transcription of `census_extend`, on REAL auditions --- #
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    tl = truth[ell]
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    rg = np.random.default_rng(5)
    gr, gx = context_instances(rules, {"name": "L3n6", "level": 3, "nodes": [6]},
                               N_AUD, s, depth, v, m, seed=GATE_SEED_BASE + 11)
    gxt = torch.from_numpy(gx).to(device)

    def aud(child_rows):
        arr = np.asarray(child_rows, np.int64).reshape(-1, s)
        tb = MC.make_table(ell, arr, tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, _ = PP.fire_rec(core, gxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), gr, rules, s)
        return float(1.0 - su.mean())
    base = [[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 12, rg)
    pool = ([[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
            + [list(q) for q in wr])
    for cap, tol in ((EXTEND_CAP, 0.0), (len(pool), 0.0), (len(pool), TOL_SENS)):
        a = _census_literal(aud, base, pool, tol, cap)
        b = census_walk(aud, base, pool, list(range(len(pool))), tol=tol, budget=cap)
        same = (a["admitted"] == b["admitted"] and a["rejected"] == b["rejected"]
                and a["n"] == b["n_auditions"] and abs(a["best_e"] - b["e_gate_final"]) < 1e-12
                and a["n_kept"] == b["n_kept"])
        res[f"G-1a:cap{cap}_tol{tol}"] = {"literal": a["admitted"], "here": b["admitted"],
                                          "identical": bool(same)}
        assert same, f"G-1a FAILED at cap {cap} tol {tol}: {a} against {b}"
    tick("G-1a closed")

    # ---- G-2: the three pool families are pairwise disjoint ---------------------------- #
    def keyset(roots_, x_):
        a = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist())
    g2 = {}
    for ell_, node_ in ((2, 12), (3, 6), (4, 3), (5, 1)):
        ctx = {"name": f"L{ell_}n{node_}", "level": ell_, "nodes": [node_]}
        P = keyset(*context_instances(rules, ctx, 256, s, depth, v, m,
                                      seed=PRICE_SEED_BASE + 1000 * ell_))
        T = [keyset(*context_instances(rules, ctx, 128, s, depth, v, m,
                                       seed=TEST_SEED_BASE + 1000 * ell_ + 17 * t))
             for t in range(3)]
        G = [keyset(*context_instances(rules, ctx, N_AUD, s, depth, v, m,
                                       seed=GATE_SEED_BASE + 1000 * ell_ + 29 * rp))
             for rp in range(2)]
        ov = ([len(P & q) for q in T] + [len(P & q) for q in G] +
              [len(a_ & b_) for a_ in T for b_ in G] +
              [len(T[0] & T[1]), len(T[0] & T[2]), len(T[1] & T[2]), len(G[0] & G[1])])
        g2[f"L{ell_}"] = {"max_overlap": int(max(ov)), "n_gate_distinct": len(G[0])}
        assert max(ov) == 0, f"G-2 FAILED at L{ell_}: {ov}"
    res["G-2:three_families_disjoint"] = g2
    tick("G-2 closed")

    # ---- G-3: a constant price reduces the order to the shared tie-break --------------- #
    n = 150
    tie = np.random.default_rng(3).permutation(n)
    a = order_by(np.zeros(n), tie)
    b = order_by(np.full(n, 2.5), tie)
    c = np.argsort(tie, kind="mergesort")
    res["G-3:constant_price_is_random_order"] = bool((a == b).all() and (a == c).all())
    assert (a == b).all() and (a == c).all(), "G-3 FAILED"
    # ... and a strictly decreasing price yields the identity order
    d = order_by(-np.arange(n, dtype=float), tie)
    res["G-3:build_price_is_identity"] = bool((d == np.arange(n)).all())
    assert (d == np.arange(n)).all(), "G-3 FAILED on the build order"
    tick("G-3 closed")

    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def falsify4():
    """Every gate above, shown to FAIL on a deliberately broken input."""
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

    # ---- G-1b broken: the base does not grow on admission (`best_e` left at the base) -- #
    seq = [0.50, 0.40, 0.45, 0.40, 0.50, 0.10, 0.11]
    box = {"i": 0}

    def stub(_r):
        e = seq[box["i"]]
        box["i"] += 1
        return e

    def walk_broken(aud_fn, base, cands_, order, tol):
        e0 = aud_fn(base)
        adm = []
        for i in order:
            if aud_fn(base + [cands_[i]]) <= e0 + tol:     # THE BREAK: `best_e` never updates
                adm.append(int(i))
        return adm
    bad = walk_broken(stub, [[0, 0]], [[1, 1]] * 6, list(range(6)), 0.0)
    res["G-1b"] = {"broke": "the base never grows, so `best_e` stays at the base's error",
                   "admitted": bad, "want": [0, 2, 4],
                   "gate_condition_holds": bad == [0, 2, 4]}
    assert bad != [0, 2, 4], "G-1b did not trip on a non-growing base"
    n_ok += 1

    # ---- G-1a broken: `<` instead of `<=`, so a tie is rejected ------------------------ #
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    tl = truth[ell]
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    rg = np.random.default_rng(5)
    gr, gx = context_instances(rules, {"name": "L3n6", "level": 3, "nodes": [6]},
                               N_AUD, s, depth, v, m, seed=GATE_SEED_BASE + 11)
    gxt = torch.from_numpy(gx).to(device)

    def aud(child_rows):
        arr = np.asarray(child_rows, np.int64).reshape(-1, s)
        tb = MC.make_table(ell, arr, tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, _ = PP.fire_rec(core, gxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), gr, rules, s)
        return float(1.0 - su.mean())
    base = [[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 12, rg)
    pool = ([[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
            + [list(q) for q in wr])

    def literal_strict(aud_fn, base_, cands_, tol, cap):
        best = aud_fn(base_)
        kept, adm = list(base_), []
        for i, c in enumerate(cands_[:cap]):
            e = aud_fn(kept + [list(map(int, c))])
            if e < best + tol:                                  # THE BREAK: `<` not `<=`
                kept, best = kept + [list(map(int, c))], e
                adm.append(i)
        return adm
    a = literal_strict(aud, base, pool, 0.0, len(pool))
    b = census_walk(aud, base, pool, list(range(len(pool))), tol=0.0)["admitted"]
    res["G-1a"] = {"broke": "the admission test is `<` instead of the loop's `<=`",
                   "strict": a, "here": b, "gate_condition_holds": a == b}
    assert a != b, "G-1a did not trip on a strict inequality"
    n_ok += 1

    # ---- G-2 broken: the gate pool is drawn on the test family's seed ------------------ #
    def keyset(roots_, x_):
        a_ = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a_.view(np.dtype((np.void, a_.dtype.itemsize * a_.shape[1]))).ravel()
                   .tolist())
    ctx = {"name": "L3n6", "level": 3, "nodes": [6]}
    T = keyset(*context_instances(rules, ctx, N_AUD, s, depth, v, m,
                                  seed=TEST_SEED_BASE + 3000))
    G = keyset(*context_instances(rules, ctx, N_AUD, s, depth, v, m,
                                  seed=TEST_SEED_BASE + 3000))
    res["G-2"] = {"broke": "the gate pool is drawn on the test family's seed",
                  "overlap": len(T & G), "gate_condition_holds": len(T & G) == 0}
    assert len(T & G) > 0, "G-2 did not trip on a shared seed"
    n_ok += 1

    # ---- G-3 broken: each order draws its own tie-break -------------------------------- #
    rg2 = np.random.default_rng(3)
    n = 150
    a3 = order_by(np.zeros(n), rg2.permutation(n))
    b3 = order_by(np.zeros(n), rg2.permutation(n))
    res["G-3"] = {"broke": "each order draws its own tie-break permutation",
                  "identical": bool((a3 == b3).all()),
                  "gate_condition_holds": bool((a3 == b3).all())}
    assert not (a3 == b3).all(), "G-3 did not trip on an unshared tie-break"
    n_ok += 1

    res["n_gates_falsified"] = n_ok
    print(json.dumps(res, indent=2, default=str))
    return res
