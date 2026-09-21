"""[preplay/pp5] THE READ AS THE **GATE**, THE PRIOR AS THE ORDER — the fully endogenous
consumer, and the seat where the read replaces the oracle.

pp4 settled that the executor's own score (`dp_top`) is the right ORDER for the loop's
try-and-keep extension and that the value readout is not a scheduler: as an order it beat
random but lost to the free prior in both settings and to a never-trained trunk in one. The
readout's seat is the other half of `census_extend`. The loop's gate is an ORACLE READ -- "did
the world's error on the gate pool rise?" -- and the readout is the thing that gets to see the
preplayed state and put a number on it. **Can that number stand in for the world's verdict?**

    for cand in cands:                                  ORDER = `dp_top` throughout (pp4's answer)
        fire `kept + {cand}` on the gate pool           ONE fire, every gate reads it
        admit iff <GATE>                                GATE = what varies here
        if admit: kept, cur = trial

THE GATES.

    world       admit iff the world's error does not rise      pp4's, THE REFERENCE
    read        admit iff the banked shaped projection's LEVEL over the same fired gate-pool
                configurations does not fall (mean p with against without)   UNDER TEST
    read_pair   the same, PAIRED over the instances the candidate actually changed. Most
                candidates change few instances, so the difference of pooled means is a few
                instances' worth diluted by the rest; the paired form reads only where the
                candidate did anything. "Changed" is exact and free: adding a row to a table
                can only move an instance's argmax TO the new row, so the changed set is
                `trial.entry == len(kept)`.
    read_m      the pooled form with a MARGIN: admit iff the level does not fall by more than
                delta, where delta is the standard error of the base's own level on the gate
                pool (`std(p_base) / sqrt(n_gate)`) -- the read's own noise scale, measured per
                cell and repeat, not a number chosen here.
    frozen      the pooled form through `overtone`'s unshaped core        CONTROL
    twin        the pooled form through a never-trained trunk             FLOOR
    ungated     admit everything                                          THE FLOOR OF FLOORS

THE DIRECT MEASURE. Divergent gates visit divergent states, so "the same candidate" has to be
pinned: at EVERY step the world's decision ON THAT SAME TRIAL is recorded beside the gate's
own, from the same fire. The confusion counts are therefore over the states each gate actually
visited, and the world's error change on the candidates a gate admitted where the world would
have rejected is what those admissions cost.

Pools: pp4's three families, disjoint (gate P-4). Settings (a) and (b): pp4's.

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::gates5
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::falsify5
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::sweep5 \\
        --out-tag pp5_smoke --arms s0_sv --smoke 1
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::sweep5 \\
        --out-tag pp5
"""

import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay-readgate", image=image)

REMOTE = "rhm_practice_preplay"
BANK_SB = "rhm_practice_soundboard"
BANK_OV = "rhm_practice_voicing"

GATES = ("world", "read", "read_pair", "read_m", "frozen", "twin", "ungated")
GATE_READ = {"read": "banked", "read_pair": "banked", "read_m": "banked",
             "frozen": "frozen", "twin": "twin"}
ORDERS = ("dp_top",)                    # pp4's answer; the secondary orders are below
ORDERS_2 = ("world", "banked")          # repeat 0 only, gates `world` and `read` only
GATES_2 = ("world", "read")
BUDGETS = (8, 16, 32)


def decide(gate, cur, tri, n_kept, tol, delta):
    """The admission rule. `cur` and `tri` carry the world's error `e`, the per-instance level
    arrays `p[nm]` and the DP's per-instance winning entry index. Returns (admit, why)."""
    if gate == "ungated":
        return True, 0.0
    if gate == "world":
        return bool(tri["e"] <= cur["e"] + tol), float(cur["e"] - tri["e"])
    nm = GATE_READ[gate]
    if gate == "read_pair":
        if n_kept == 0:
            ch = np.ones(tri["p"][nm].shape[0], bool)
        else:
            ch = (tri["entry"] == int(n_kept))
        if not ch.any():
            # the candidate changed nothing, so it cannot have hurt. The WORLD gate admits
            # here too (`e_x == best_e`, and its test is `<=`), so this is the world's own
            # behaviour on an invisible candidate and not a choice made here.
            return True, 0.0
        d = float((tri["p"][nm][ch] - cur["p"][nm][ch]).mean())
        return bool(d >= -0.0), d
    d = float(tri["p"][nm].mean() - cur["p"][nm].mean())
    return bool(d >= -(delta if gate == "read_m" else 0.0)), d


def gated_walk(stats_fn, base_child, cand_child, order_idx, gate, tol, delta,
               budget=None, on_change=None):
    """`census_extend`'s loop with the ADMISSION TEST swapped. Identical to
    `incremental.census_walk` when `gate == "world"` (gate P-1). One audition for the base and
    one per candidate offered, which is what the loop pays; the world's decision on the same
    trial is recorded at every step whatever the gate decides."""
    cur = stats_fn(base_child)
    n_aud = 1
    kept = [list(map(int, r_)) for r_ in base_child]
    rec = {"gate": gate, "e_base_gate": float(cur["e"]), "admitted": [], "rejected": 0,
           "conf": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "cost_rows": [], "steps": []}
    if on_change is not None:
        on_change(n_aud, kept)
    walk = order_idx if budget is None else order_idx[:int(budget)]
    for ci in walk:
        tri = stats_fn(kept + [list(map(int, cand_child[ci]))])
        n_aud += 1
        a_g, why = decide(gate, cur, tri, len(kept), tol, delta)
        a_w = bool(tri["e"] <= cur["e"] + tol)
        rec["conf"]["TT" if (a_g and a_w) else "TF" if (a_g and not a_w)
                    else "FT" if (not a_g and a_w) else "FF"] += 1
        if a_g and not a_w:
            rec["cost_rows"].append(float(tri["e"] - cur["e"]))
        if a_g:
            kept = kept + [list(map(int, cand_child[ci]))]
            rec["admitted"].append(int(ci))
            cur = tri
        else:
            rec["rejected"] += 1
        rec["steps"].append({"cand": int(ci), "admit_gate": a_g, "admit_world": a_w,
                             "e_gate": float(tri["e"]), "why": float(why),
                             "n_aud": int(n_aud)})
        if a_g and on_change is not None:
            on_change(n_aud, kept)
    rec["n_auditions"] = n_aud
    rec["e_gate_final"] = float(cur["e"])
    rec["n_kept"] = int(len(kept))
    rec["kept"] = kept
    return rec


# --------------------------------------------------------------------------------------- #
# the per-arm job
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=8192)
def readgate_arm(arm_key, out_tag, smoke=False, levels="", settings="ab", n_price=0,
                 n_gate=0, n_test=0, k_pool=0, repeats=0):
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import incremental as IN
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
    NP_ = int(n_price) or IN.N_PRICE
    NG = int(n_gate) or IN.N_AUD
    NT = int(n_test) or IN.N_TEST
    KP = int(k_pool) or IN.K_POOL_A
    NR = int(repeats) or IN.N_REPEATS
    NTP, kpb = IN.N_TEST_POOLS, IN.K_POOL_B
    TOL = IN.EXTEND_TOL
    if smoke:
        NP_, NG, NT, KP, NR, NTP, kpb = 128, 64, 64, 6, 1, 2, 12
    log("=" * 100)
    log(f"[pp5] arm {arm_key}: {spec['tag']}/{spec['arm']}  device {device}  "
        f"smoke={bool(smoke)}  settings={settings}")
    log(f"  order {ORDERS[0]} (pp4's answer)   gates {GATES}   tol {TOL}")
    log(f"  n_price {NP_}  n_gate {NG}  n_test {NT}x{NTP}  k_pool_a {KP}  repeats {NR}")
    log("=" * 100)

    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m, maxl = int(cfgr["m"]), int(cfgr["max_macro_level"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
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
    gate_f = {"F-2b:hold_auc_here": auc_here, "F-2b:hold_auc_banked": auc_banked,
              "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band}
    assert abs(auc_here - auc_banked) < band, "F-2b FAILED"
    log(f"  [F-2b] banked {auc_banked:.4f}  re-read {auc_here:.4f}  band {band:.4f}")

    trainable = (bank["h"] >= PP.PJ_HOLD)
    tr_all = np.nonzero(trainable & (bank["u0"] < PP.PJ_BOOT))[0]
    va_all = np.nonzero(trainable & (bank["u0"] >= PP.PJ_BOOT))[0]
    frng = np.random.default_rng(IN.FIT_SEED_BASE + int(cfgr["seed"]))
    idx_tr = (np.sort(frng.choice(tr_all, size=PP.PJ_FIT_CAP, replace=False))
              if tr_all.size > PP.PJ_FIT_CAP else tr_all)
    vcap = max(512, PP.PJ_FIT_CAP // 4)
    idx_va = (np.sort(frng.choice(va_all, size=vcap, replace=False))
              if va_all.size > vcap else va_all)
    fits = {}
    for nm, core_ in (("frozen", frozen), ("twin", twin)):
        fits[nm] = PP.pj_fit(core_, bank, idx_tr, idx_va, s, nb, dim, v, device)

    RNAMES = ("banked", "frozen", "twin")

    def read_all(xf, roots_t, blk0, span):
        out = {}
        for nm in RNAMES:
            if nm == "banked":
                core_, w_, mu_, sd_ = shaped, w_b, mu_b, sd_b
            else:
                core_, f_ = ({"frozen": frozen, "twin": twin}[nm], fits[nm])
                w_, mu_, sd_ = f_["w"], f_["mu"], f_["sd"]
            out[nm] = PP.pj_predict(core_, xf, roots_t, blk0, span, w_, mu_, sd_,
                                    s, nb, dim, v, device).numpy().astype("float32")
        return out

    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {int(e_["level"]): int(e_["n_entries"]) for e_ in res_j["events"]
               if e_.get("kind") == "commit"}
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)

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
        lvl_b = sorted(q for q in lt_diags if q in IN.LEVELS_B)
        log(f"  setting (b): operative tables gated; sizes "
            + str({f"L{e}": int(lt_tables[e]['child'].shape[0]) for e in lvl_b}))

    out = {"arm": arm_key, "spec": spec, "gate": gate_f, "settings": settings,
           "cfg": {"n_price": NP_, "n_gate": NG, "n_test": NT, "n_test_pools": NTP,
                   "k_pool_a": KP, "k_pool_b": kpb, "repeats": NR, "tol": TOL,
                   "budgets": list(BUDGETS), "gates": list(GATES),
                   "orders": list(ORDERS), "orders_2": list(ORDERS_2),
                   "extend_cap": IN.EXTEND_CAP, "n_aud": IN.N_AUD,
                   "seed": int(cfgr["seed"]), "rule_seed": int(cfgr["rule_seed"])},
           "lt_gates": [{k_: g[k_] for k_ in ("gate", "pass", "what")} for g in lt_gates],
           "cells": {}}
    n_aud_total = 0
    p4 = {}

    def keyset(roots_, x_):
        a = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel().tolist())

    for setting in [q for q in ("a", "b") if q in settings]:
        lvls = IN.LEVELS_A if setting == "a" else IN.LEVELS_B
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
            log(f"  === setting ({setting}) L{ell} at {era['name']}")

            if setting == "a":
                truth_l = truth[ell]
                lower = truth_l["lower"]
                n_true_rows = int(truth_l["child"].shape[0])
                true_flat = {tuple(int(q) for q in r_) for r_ in truth_l["flat"]}
                n_committed = commits.get(ell)
                if n_committed is None:
                    n_committed = int(aud_last.get(str(ell), {}).get("n_entries") or 0)
                n_committed = max(2, min(int(n_committed), n_true_rows))
                rng = np.random.default_rng(IN.DRAW_SEED_BASE + int(cfgr["seed"]) * 97 + ell)
                idx_true = np.sort(rng.permutation(n_true_rows)[:min(KP, n_true_rows)])
                wrong = PP.wrong_rows(int(lower["flat"].shape[0]), true_flat,
                                      lower["flat"], s, KP, rng)
                cands = ([{"cls": "true", "child": [int(q) for q in truth_l["child"][i_]]}
                          for i_ in idx_true] +
                         [{"cls": "wrong", "child": [int(q) for q in r_]} for r_ in wrong])
                n_base = max(2, min(int(round(0.25 * n_committed)), n_true_rows))
                base_src = "quarter of committed, from the true table"
            else:
                tl = lt_tables[ell]
                lower = tl["lower"]
                truth_l = truth[ell]
                trow = lt_diags[ell].get("truth_row")
                n_rows = int(tl["child"].shape[0])
                rng = np.random.default_rng(IN.DRAW_SEED_BASE + 500_000
                                            + int(cfgr["seed"]) * 97 + ell)
                take = (np.arange(n_rows) if n_rows <= kpb
                        else np.sort(rng.permutation(n_rows)[:kpb]))
                cands = [{"cls": ("true" if (trow and trow[int(i_)]) else "wrong"),
                          "child": [int(q) for q in tl["child"][i_]]} for i_ in take]
                n_base, base_src = 0, "empty"
            ncand = len(cands)
            cand_child = [c["child"] for c in cands]
            cls_arr = np.array([1.0 if c["cls"] == "true" else 0.0 for c in cands])
            tie = rng.permutation(ncand)

            off = 0 if setting == "a" else 300_000
            pseed = int(cfgr["seed"]) + IN.PRICE_SEED_BASE + off + 1000 * ell
            pr_roots, pr_x = context_instances(
                rules, {"name": era["name"], "level": ell, "nodes": [node]},
                NP_, s, depth, v, m, seed=pseed)
            tests = []
            for t_ in range(NTP):
                ts = int(cfgr["seed"]) + IN.TEST_SEED_BASE + off + 1000 * ell + 17 * t_
                tr_, tx_ = context_instances(
                    rules, {"name": era["name"], "level": ell, "nodes": [node]},
                    NT, s, depth, v, m, seed=ts)
                tests.append({"seed": ts, "roots": tr_,
                              "x": torch.from_numpy(tx_).to(device), "x_np": tx_})
            gpools = []
            for rp in range(NR):
                gs = int(cfgr["seed"]) + IN.GATE_SEED_BASE + off + 1000 * ell + 29 * rp
                gr_, gx_ = context_instances(
                    rules, {"name": era["name"], "level": ell, "nodes": [node]},
                    NG, s, depth, v, m, seed=gs)
                gpools.append({"seed": gs, "roots": gr_,
                               "x": torch.from_numpy(gx_).to(device), "x_np": gx_,
                               "roots_t": torch.from_numpy(gr_).to(device).long()})
            kp_ = keyset(pr_roots, pr_x)
            kt_ = [keyset(q["roots"], q["x_np"]) for q in tests]
            kg_ = [keyset(q["roots"], q["x_np"]) for q in gpools]
            ov = ([len(kp_ & q) for q in kt_] + [len(kp_ & q) for q in kg_] +
                  [len(a_ & b_) for a_ in kt_ for b_ in kg_] +
                  [len(kt_[i] & kt_[j]) for i in range(NTP) for j in range(i + 1, NTP)] +
                  [len(kg_[i] & kg_[j]) for i in range(NR) for j in range(i + 1, NR)])
            p4[f"{setting}L{ell}"] = {"max_overlap": int(max(ov)) if ov else 0}
            assert (max(ov) == 0) if ov else True, f"P-4 FAILED at ({setting})L{ell}: {ov}"

            # ---- the order: pp4's answer, priced on the PRICING pool ------------------- #
            px = torch.from_numpy(pr_x).to(device)
            proots_t = torch.from_numpy(pr_roots).to(device).long()
            price = {k_: np.zeros(ncand) for k_ in ("world", "dp_top") + RNAMES}
            for ci, cd in enumerate(cands):
                tb = MC.make_table(ell, np.asarray([cd["child"]], np.int64), lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, px, mv, canon, s)
                su, _ = grade(xf.cpu().numpy(), pr_roots, rules, s)
                rd = read_all(xf, proots_t, blk0, span)
                n_aud_total += 1
                price["world"][ci] = float(su.mean())
                price["dp_top"][ci] = float(dp["top"].mean())
                for nm in RNAMES:
                    price[nm][ci] = float(rd[nm].mean())
            log(f"      {ncand} candidates ({int(cls_arr.sum())}T/"
                f"{int((1 - cls_arr).sum())}W), base {n_base} ({base_src}); order AUCs " +
                "  ".join(f"{k_}={0.0 if PP.vo_auc(price[k_], cls_arr) is None else PP.vo_auc(price[k_], cls_arr):.3f}"
                          for k_ in ("world", "dp_top", "banked")))

            scache, tcache = {}, {}

            def _fire(child_rows, pool, want_reads):
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                if arr.shape[0] == 0:
                    su, _ = grade(pool["x_np"], pool["roots"], rules, s)
                    d = {"e": float(1.0 - su.mean()),
                         "entry": np.full(pool["x"].shape[0], -1, np.int32)}
                    if want_reads:
                        d["p"] = read_all(pool["x"], pool["roots_t"], blk0, span)
                    return d
                tb = MC.make_table(ell, arr, lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, pool["x"], mv, canon, s)
                su, _ = grade(xf.cpu().numpy(), pool["roots"], rules, s)
                d = {"e": float(1.0 - su.mean()), "entry": dp["entry"]}
                if want_reads:
                    d["p"] = read_all(xf, pool["roots_t"], blk0, span)
                return d

            def stats(child_rows, rp):
                """ONE fire of the gate pool, read by every gate. The world's error and the
                three readouts' per-instance levels come from the SAME fired configurations,
                which is what makes the per-candidate agreement a comparison and not two
                experiments."""
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = (rp, int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in scache:
                    return scache[key]
                d = _fire(arr, gpools[rp], True)
                n_aud_total += 1
                scache[key] = d
                return d

            def test_e(child_rows):
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = (int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in tcache:
                    return tcache[key]
                es = [_fire(arr, tests[t_], False)["e"] for t_ in range(NTP)]
                n_aud_total += NTP
                d = {"mean": float(np.mean(es)), "sd": float(np.std(es))}
                tcache[key] = d
                return d

            def n_wrong(rows):
                st_ = {tuple(map(int, r_)) for r_ in rows}
                return int(sum(1 for c in cands
                               if c["cls"] == "wrong" and tuple(c["child"]) in st_))

            cell = {"setting": setting, "era": era["name"], "n_cand": ncand,
                    "n_cand_true": int(cls_arr.sum()),
                    "n_cand_wrong": int((1 - cls_arr).sum()),
                    "n_base": n_base, "base_src": base_src,
                    "order_auc": {k_: PP.vo_auc(price[k_], cls_arr)
                                  for k_ in ("world", "dp_top") + RNAMES},
                    "max_overlap": p4[f"{setting}L{ell}"]["max_overlap"], "repeats": []}

            for rp in range(NR):
                if setting == "a":
                    keep = np.sort(np.random.default_rng(
                        IN.DRAW_SEED_BASE + 7 * rp + 13 * ell
                        + int(cfgr["seed"])).permutation(n_true_rows)[:n_base])
                    base_child = [[int(q) for q in r_] for r_ in truth_l["child"][keep]]
                    base_flats = {tuple(int(q) for j in r_ for q in lower["flat"][j])
                                  for r_ in truth_l["child"][keep]}
                    live = [i_ for i_ in range(ncand)
                            if tuple(int(q) for j in cands[i_]["child"]
                                     for q in lower["flat"][j]) not in base_flats]
                else:
                    base_child, live = [], list(range(ncand))
                lv = np.array(live)
                # THE MARGIN, from the read's OWN noise on this gate pool: the standard error
                # of the base table's level. Measured, not chosen.
                b0 = stats(base_child, rp)
                delta = float(b0["p"]["banked"].std() / np.sqrt(max(1, NG)))
                rep = {"repeat": rp, "gate_seed": gpools[rp]["seed"],
                       "n_base": len(base_child), "n_live": len(live),
                       "delta": delta, "e_base_test": test_e(base_child),
                       "p_base": float(b0["p"]["banked"].mean()), "orders": {}}
                todo = [(ORDERS[0], g_) for g_ in GATES]
                if rp == 0:
                    todo += [(o_, g_) for o_ in ORDERS_2 for g_ in GATES_2]
                for onm, gnm in todo:
                    oi = lv[IN.order_by(price[onm][lv], tie[lv])]
                    curve = []

                    def _rec(na, kept, _c=curve):
                        _c.append({"n_aud": int(na), "n_kept": int(len(kept)),
                                   "n_wrong": n_wrong(kept)})
                        _c[-1]["rows"] = [list(map(int, r_)) for r_ in kept]
                    r = gated_walk(lambda cr, _rp=rp: stats(cr, _rp), base_child, cand_child,
                                   oi, gnm, TOL, delta, on_change=_rec)
                    bud = {}
                    _pos = {int(c_): i_ for i_, c_ in enumerate(oi)}
                    for b in list(BUDGETS) + [len(oi)]:
                        rows_b = base_child + [cand_child[i_] for i_ in r["admitted"]
                                               if _pos.get(int(i_), 10 ** 9) < int(b)]
                        t3 = test_e(rows_b)
                        bud[str(b)] = {"e_test": t3["mean"], "e_test_sd": t3["sd"],
                                       "n_kept": len(rows_b), "n_wrong": n_wrong(rows_b)}
                    r["budget"] = bud
                    r["n_wrong_kept"] = n_wrong(r["kept"])
                    r["e_test_final"] = test_e(r["kept"])
                    r["cost_mean"] = (float(np.mean(r["cost_rows"]))
                                      if r["cost_rows"] else None)
                    r["cost_sum"] = float(np.sum(r["cost_rows"])) if r["cost_rows"] else 0.0
                    r.pop("kept", None)
                    r.pop("steps", None)
                    r.pop("cost_rows", None)
                    for q in curve:
                        q.pop("rows", None)
                    r["curve"] = curve
                    rep["orders"][f"{onm}|{gnm}"] = r
                cell["repeats"].append(rep)
                k0 = [f"{ORDERS[0]}|{g_}" for g_ in GATES]
                log(f"      rp{rp} base_e {rep['e_base_test']['mean']:.3f} "
                    f"p_base {rep['p_base']:.3f} delta {delta:.4f}   b=8: " +
                    "  ".join(f"{g_}:{rep['orders'][f'{ORDERS[0]}|{g_}']['budget']['8']['e_test']:.3f}"
                              for g_ in GATES))
                log(f"           final: " + "  ".join(
                    f"{g_}:{rep['orders'][f'{ORDERS[0]}|{g_}']['e_test_final']['mean']:.3f}"
                    f"/k{rep['orders'][f'{ORDERS[0]}|{g_}']['n_kept']}"
                    f"/w{rep['orders'][f'{ORDERS[0]}|{g_}']['n_wrong_kept']}"
                    for g_ in GATES))
            cell["sec"] = float(time.time() - t_l0)
            out["cells"][f"{setting}L{ell}"] = cell
            log(f"      cell done in {cell['sec']:.1f}s  auditions so far {n_aud_total}")

    out["P-4"] = p4
    out["n_auditions"] = n_aud_total
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s  auditions {n_aud_total}  "
        f"peak RSS {out['peak_rss_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"arm": arm_key, "sec": out["sec"], "n_aud": n_aud_total,
            "rss_mb": out["peak_rss_mb"], "gate": gate_f}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=12600, memory=2048)
def sweep5(out_tag="pp5", smoke=False, arms="", levels="", settings="ab", n_price=0,
           n_gate=0, n_test=0, k_pool=0, repeats=0):
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    want = [x for x in arms.split(",") if x] or [k for k, q in PP.ARMS.items() if q["record"]]
    args = [(k, out_tag, bool(smoke), levels, settings, int(n_price), int(n_gate),
             int(n_test), int(k_pool), int(repeats)) for k in want]
    outs = list(readgate_arm.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"arms": want, "smoke": bool(smoke), "settings": settings}) + "\n")
    volume.commit()
    return outs


# --------------------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def gates5():
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import incremental as IN

    res = {}
    t0 = time.time()

    def tick(msg):
        print(f"[gates5 +{time.time() - t0:6.1f}s] {msg}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    truth = MC.true_tables(rules, depth, s, v, m, 4)
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    tl = truth[ell]
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    rg = np.random.default_rng(5)
    gr, gx = context_instances(rules, {"name": "L3n6", "level": 3, "nodes": [6]},
                               IN.N_AUD, s, depth, v, m, seed=IN.GATE_SEED_BASE + 11)
    gxt = torch.from_numpy(gx).to(device)
    base = [[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 12, rg)
    pool = ([[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:14])]]
            + [list(q) for q in wr])

    def raw(child_rows):
        arr = np.asarray(child_rows, np.int64).reshape(-1, s)
        tb = MC.make_table(ell, arr, tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, dp = PP.fire_rec(core, gxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), gr, rules, s)
        return float(1.0 - su.mean()), dp["entry"]

    # ---- P-1: the world gate here IS `incremental.census_walk` ------------------------- #
    def stats_world(child_rows):
        e, ent = raw(child_rows)
        # the level is the NEGATED world error, broadcast: the plumbing identity of P-2
        return {"e": e, "entry": ent,
                "p": {k_: np.full(len(ent), -e, np.float32) for k_ in ("banked", "frozen",
                                                                       "twin")}}
    a = IN.census_walk(lambda cr: raw(cr)[0], base, pool, list(range(len(pool))), tol=0.0)
    b = gated_walk(stats_world, base, pool, list(range(len(pool))), "world", 0.0, 0.0)
    same = (a["admitted"] == b["admitted"] and a["rejected"] == b["rejected"]
            and a["n_auditions"] == b["n_auditions"]
            and abs(a["e_gate_final"] - b["e_gate_final"]) < 1e-12)
    res["P-1a:world_gate_is_census_walk"] = {"pp4": a["admitted"], "here": b["admitted"],
                                             "identical": bool(same),
                                             "n_offered": len(pool)}
    assert same, f"P-1a FAILED: {a['admitted']} against {b['admitted']}"
    # P-1b: the SAME hand-computed case pp4's G-1b uses, so the update rule ("`best_e` moves
    # only on admission") is pinned by a designed sequence and not only by a pool that happens
    # to admit everything. That is what P-1a cannot do: on the real pool above the base is
    # already the minimum, so several wrong update rules coincide with the right one
    # (NOTES defect 9).
    seq = [0.50, 0.40, 0.45, 0.40, 0.50, 0.10, 0.11]
    box = {"i": 0}

    def stub(_rows):
        e = seq[box["i"]]
        box["i"] += 1
        return {"e": e, "entry": np.zeros(4, np.int32),
                "p": {k_: np.full(4, -e, np.float32)
                      for k_ in ("banked", "frozen", "twin")}}
    c1 = gated_walk(stub, [[0, 0]], [[1, 1]] * 6, list(range(6)), "world", 0.0, 0.0)
    res["P-1b:synthetic"] = {"admitted": c1["admitted"], "rejected": c1["rejected"],
                             "n_auditions": c1["n_auditions"]}
    assert c1["admitted"] == [0, 2, 4] and c1["rejected"] == 3, f"P-1b FAILED: {c1}"
    tick("P-1 closed")

    # ---- P-2: THE PLUMBING IDENTITY. Handed the world's error as its level, the READ gate
    #           must reproduce the world gate exactly, in all three of its forms. ---------- #
    p2 = {}
    for gnm in ("read", "read_pair", "read_m"):
        c = gated_walk(stats_world, base, pool, list(range(len(pool))), gnm, 0.0, 0.0)
        ok = (c["admitted"] == a["admitted"] and c["rejected"] == a["rejected"])
        p2[gnm] = {"admitted": c["admitted"], "identical": bool(ok)}
        assert ok, f"P-2 FAILED for {gnm}: {c['admitted']} against {a['admitted']}"
    res["P-2:read_gate_reduces_to_world"] = p2
    tick("P-2 closed")

    # ---- P-3: a CONSTANT level admits everything -------------------------------------- #
    def stats_const(child_rows):
        e, ent = raw(child_rows)
        return {"e": e, "entry": ent,
                "p": {k_: np.full(len(ent), 0.37, np.float32)
                      for k_ in ("banked", "frozen", "twin")}}
    p3 = {}
    for gnm in ("read", "read_pair", "read_m", "frozen", "twin"):
        c = gated_walk(stats_const, base, pool, list(range(len(pool))), gnm, 0.0, 0.0)
        p3[gnm] = {"n_admitted": len(c["admitted"]), "n_offered": len(pool)}
        assert len(c["admitted"]) == len(pool), f"P-3 FAILED for {gnm}"
    ung = gated_walk(stats_const, base, pool, list(range(len(pool))), "ungated", 0.0, 0.0)
    p3["ungated"] = {"n_admitted": len(ung["admitted"]), "n_offered": len(pool)}
    assert len(ung["admitted"]) == len(pool), "P-3 FAILED for ungated"
    res["P-3:constant_level_admits_everything"] = p3
    tick("P-3 closed")

    # ---- P-5: the CHANGED set is exactly "the DP moved to the new row" ----------------- #
    # Adding a row to a table can only move an instance's argmax TO that row, so
    # `trial.entry == len(kept)` must equal the rows whose FIRED CONFIGURATION differs.
    # THE BASE HERE IS ONE ROW, and the candidates are the six the DP scores highest on this
    # pool. With the 12-row base used above the DP never picked the added row on any instance
    # (`n_differ = 0` six times over), which made the gate vacuous -- it cannot fail if nothing
    # ever changes. NOTES defect 10. The non-vacuity is now asserted.
    kept = [base[0]]
    dpt = []
    for c_ in pool:
        mv_ = MC.to_device(MC.make_macro(ell, node, s, MC.make_table(
            ell, np.asarray([c_], np.int64), tl["lower"], s)), device)
        _, d_ = PP.fire_rec(core, gxt, mv_, canon, s)
        dpt.append(float(d_["top"].mean()))
    cand6 = [pool[i_] for i_ in np.argsort(-np.asarray(dpt))[:6]]
    p5 = []
    for c_ in cand6:
        arr0 = np.asarray(kept, np.int64).reshape(-1, s)
        arr1 = np.asarray(kept + [c_], np.int64).reshape(-1, s)
        mv0 = MC.to_device(MC.make_macro(ell, node, s,
                                         MC.make_table(ell, arr0, tl["lower"], s)), device)
        mv1 = MC.to_device(MC.make_macro(ell, node, s,
                                         MC.make_table(ell, arr1, tl["lower"], s)), device)
        x0f, d0 = PP.fire_rec(core, gxt, mv0, canon, s)
        x1f, d1 = PP.fire_rec(core, gxt, mv1, canon, s)
        differ = (x0f != x1f).any(1).cpu().numpy()
        flagged = (d1["entry"] == arr0.shape[0])
        # a synonymous new row can leave the CONFIGURATION identical while winning the argmax,
        # so the flagged set is a superset; what must hold is that nothing differs unflagged.
        p5.append({"n_differ": int(differ.sum()), "n_flagged": int(flagged.sum()),
                   "unflagged_but_differ": int((differ & ~flagged).sum())})
        assert int((differ & ~flagged).sum()) == 0, "P-5 FAILED: a row changed unflagged"
    res["P-5:changed_set_is_exact"] = p5
    assert sum(q["n_flagged"] for q in p5) > 0, (
        "P-5 is VACUOUS: the DP never picked the added row on any instance, so the gate "
        "cannot fail")
    tick("P-5 closed")

    # ---- P-4: pool disjointness, carried over from pp4 --------------------------------- #
    def keyset(roots_, x_):
        a_ = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a_.view(np.dtype((np.void, a_.dtype.itemsize * a_.shape[1]))).ravel()
                   .tolist())
    p4 = {}
    for ell_, node_ in ((2, 12), (3, 6), (4, 3), (5, 1)):
        ctx = {"name": f"L{ell_}n{node_}", "level": ell_, "nodes": [node_]}
        P = keyset(*context_instances(rules, ctx, 256, s, depth, v, m,
                                      seed=IN.PRICE_SEED_BASE + 1000 * ell_))
        T = [keyset(*context_instances(rules, ctx, 128, s, depth, v, m,
                                       seed=IN.TEST_SEED_BASE + 1000 * ell_ + 17 * t))
             for t in range(3)]
        G = [keyset(*context_instances(rules, ctx, IN.N_AUD, s, depth, v, m,
                                       seed=IN.GATE_SEED_BASE + 1000 * ell_ + 29 * rp))
             for rp in range(2)]
        ov = ([len(P & q) for q in T] + [len(P & q) for q in G] +
              [len(a_ & b_) for a_ in T for b_ in G] +
              [len(T[0] & T[1]), len(T[0] & T[2]), len(T[1] & T[2]), len(G[0] & G[1])])
        p4[f"L{ell_}"] = int(max(ov))
        assert max(ov) == 0, f"P-4 FAILED at L{ell_}"
    res["P-4:pools_disjoint"] = p4
    tick("P-4 closed")

    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def falsify5():
    """Every gate above, shown to FAIL on a deliberately broken input."""
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import incremental as IN

    res, n_ok = {}, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    truth = MC.true_tables(rules, depth, s, v, m, 4)
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    ell, node = 3, 6
    tl = truth[ell]
    tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
    rg = np.random.default_rng(5)
    gr, gx = context_instances(rules, {"name": "L3n6", "level": 3, "nodes": [6]},
                               IN.N_AUD, s, depth, v, m, seed=IN.GATE_SEED_BASE + 11)
    gxt = torch.from_numpy(gx).to(device)
    base = [[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:12])]]
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 12, rg)
    pool = ([[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:14])]]
            + [list(q) for q in wr])

    def raw(child_rows):
        arr = np.asarray(child_rows, np.int64).reshape(-1, s)
        tb = MC.make_table(ell, arr, tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, dp = PP.fire_rec(core, gxt, mv, canon, s)
        su, _ = grade(xf.cpu().numpy(), gr, rules, s)
        return float(1.0 - su.mean()), dp["entry"]

    def stats_world(child_rows):
        e, ent = raw(child_rows)
        return {"e": e, "entry": ent,
                "p": {k_: np.full(len(ent), -e, np.float32) for k_ in ("banked", "frozen",
                                                                       "twin")}}
    ref = IN.census_walk(lambda cr: raw(cr)[0], base, pool, list(range(len(pool))), tol=0.0)

    # ---- P-1/P-2 broken: the level is the world error WITHOUT the sign flip ------------ #
    def stats_signflip(child_rows):
        e, ent = raw(child_rows)
        return {"e": e, "entry": ent,
                "p": {k_: np.full(len(ent), +e, np.float32) for k_ in ("banked", "frozen",
                                                                       "twin")}}
    bad = gated_walk(stats_signflip, base, pool, list(range(len(pool))), "read", 0.0, 0.0)
    res["P-2"] = {"broke": "the level is the world's error with the sign NOT flipped, so "
                           "'the level did not fall' means 'the error did not fall'",
                  "admitted": bad["admitted"], "want": ref["admitted"],
                  "gate_condition_holds": bad["admitted"] == ref["admitted"]}
    assert bad["admitted"] != ref["admitted"], "P-2 did not trip on a sign flip"
    n_ok += 1

    # ---- P-1 broken: `best_e` pinned at the base, ON THE DESIGNED SEQUENCE -------------- #
    # The break has to be shown on the designed sequence and not on the real pool: there the
    # base is already the minimum, so a pinned `best_e` and a tracking `best_e` both reproduce
    # the correct admissions and neither shows the gate failing (NOTES defect 9). On the
    # sequence 0.50 | 0.40 0.45 0.40 0.50 0.10 0.11 the right rule admits [0, 2, 4]; a pinned
    # `best_e` admits everything.
    seq = [0.50, 0.40, 0.45, 0.40, 0.50, 0.10, 0.11]

    def walk_pinned(tol):
        e0 = seq[0]
        adm = []
        for i in range(6):
            if seq[i + 1] <= e0 + tol:               # THE BREAK: `best_e` never updates
                adm.append(i)
        return adm
    bp = walk_pinned(0.0)
    res["P-1"] = {"broke": "`best_e` is pinned at the base and never grows, on the designed "
                           "sequence",
                  "admitted": bp, "want": [0, 2, 4],
                  "gate_condition_holds": bp == [0, 2, 4]}
    assert bp != [0, 2, 4], "P-1 did not trip on a pinned best_e"
    n_ok += 1

    # ---- P-3 broken: a constant level with a STRICT `>` test rejects everything --------- #
    def decide_strict(cur, tri, nm="banked"):
        return bool(tri["p"][nm].mean() > cur["p"][nm].mean())

    def stats_const(child_rows):
        e, ent = raw(child_rows)
        return {"e": e, "entry": ent,
                "p": {k_: np.full(len(ent), 0.37, np.float32)
                      for k_ in ("banked", "frozen", "twin")}}
    cur = stats_const(base)
    n_adm = 0
    for c_ in pool:
        if decide_strict(cur, stats_const(base + [list(map(int, c_))])):
            n_adm += 1
    res["P-3"] = {"broke": "the admission test is a strict `>` on the level, so a constant "
                           "level admits nothing",
                  "n_admitted": n_adm, "n_offered": len(pool),
                  "gate_condition_holds": n_adm == len(pool)}
    assert n_adm != len(pool), "P-3 did not trip on a strict level test"
    n_ok += 1

    # ---- P-5 broken: the changed set taken as "the DP's entry index changed" ----------- #
    # Adding a row RENUMBERS nothing, so `d1.entry != d0.entry` is the same set — but taking
    # the changed set as `d1.entry == 0` (a fixed row) is not, and must disagree.
    kept5 = [base[0]]                # a ONE-row base, so the DP actually picks the new row
    dpt = []
    for c_ in pool:
        mv_ = MC.to_device(MC.make_macro(ell, node, s, MC.make_table(
            ell, np.asarray([c_], np.int64), tl["lower"], s)), device)
        _, d_ = PP.fire_rec(core, gxt, mv_, canon, s)
        dpt.append(float(d_["top"].mean()))
    cand6 = [pool[i_] for i_ in np.argsort(-np.asarray(dpt))[:6]]
    arr0 = np.asarray(kept5, np.int64).reshape(-1, s)
    mv0 = MC.to_device(MC.make_macro(ell, node, s,
                                     MC.make_table(ell, arr0, tl["lower"], s)), device)
    x0f, d0 = PP.fire_rec(core, gxt, mv0, canon, s)
    bad5 = []
    for c_ in cand6:
        arr1 = np.asarray(kept5 + [c_], np.int64).reshape(-1, s)
        mv1 = MC.to_device(MC.make_macro(ell, node, s,
                                         MC.make_table(ell, arr1, tl["lower"], s)), device)
        x1f, d1 = PP.fire_rec(core, gxt, mv1, canon, s)
        differ = (x0f != x1f).any(1).cpu().numpy()
        wrongflag = (d1["entry"] == 0)               # THE BREAK: row 0, not the new row
        bad5.append(int((differ & ~wrongflag).sum()))
    res["P-5"] = {"broke": "the changed set is taken as `entry == 0` instead of `entry == "
                           "len(kept)`",
                  "unflagged_but_differ": bad5,
                  "gate_condition_holds": all(q == 0 for q in bad5)}
    assert any(q > 0 for q in bad5), "P-5 did not trip on the wrong flag"
    n_ok += 1

    # ---- P-4 broken: the gate pool on the test family's seed --------------------------- #
    def keyset(roots_, x_):
        a_ = np.ascontiguousarray(np.concatenate(
            [x_.astype(np.int64), roots_.astype(np.int64)[:, None]], 1))
        return set(a_.view(np.dtype((np.void, a_.dtype.itemsize * a_.shape[1]))).ravel()
                   .tolist())
    ctx = {"name": "L3n6", "level": 3, "nodes": [6]}
    T = keyset(*context_instances(rules, ctx, IN.N_AUD, s, depth, v, m,
                                  seed=IN.TEST_SEED_BASE + 3000))
    G = keyset(*context_instances(rules, ctx, IN.N_AUD, s, depth, v, m,
                                  seed=IN.TEST_SEED_BASE + 3000))
    res["P-4"] = {"broke": "the gate pool is drawn on the test family's seed",
                  "overlap": len(T & G), "gate_condition_holds": len(T & G) == 0}
    assert len(T & G) > 0, "P-4 did not trip"
    n_ok += 1

    res["n_gates_falsified"] = n_ok
    print(json.dumps(res, indent=2, default=str))
    return res
