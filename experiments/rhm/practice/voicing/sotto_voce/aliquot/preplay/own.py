"""[preplay/pp3] THE LEARNER'S OWN MINED ENTRIES, PRICED — and the read as the selector over them.

`preplay` (pp1) fired CONSTRUCTED candidate entries through the shaped plant's own executor and
found the banked projection separates true from wrong entries at 0.61-0.75 while an unshaped
trunk and a never-trained twin read at chance. Its candidates were built by this node's author,
not by the learner: true entries came from the grammar's own table over the grammar's own lower
table, wrong entries were random pairs of TRUE lower rows.

THIS NODE PRICES THE LEARNER'S OWN OPERATIVE TABLE, whose false rows are the ones its own miner
produced -- rows over its own (partly wrong) lower vocabulary, at the class-pair keys its own
counts put at support, capped by its own `spell_cap`. That is the realistic object: it is what
the executor actually runs on, and its errors are correlated with the learner's state in a way
random pairs of true rows are not.

  step 1  `learner_tables.py` reconstructs the operative tables from `results.json` alone,
          gated (R-0..R-3) and falsified. L2 / L3 / L4 only -- L5's last build has
          `n_keys_built = 0` while its operative table holds 128 rows from an earlier build
          whose keys and picks are not in the dump.
  step 2  each operative ROW alone as a one-row table over the learner's own reconstructed
          lower table, fired through the arm's shaped executor on a fresh pool of the level's
          era cell at `n_score = 256`, world-graded, and read on the SAME fired configurations
          by the arm's banked projection, by the same readout refit through the shaped core,
          through overtone's frozen core and through a never-trained twin, plus the executor's
          own `dp_top`.
  step 3  the top-k rows by each selector as a table, auditioned by the world on a DISJOINT
          test pool, at k = the frozen committed table's size and at a smaller k, against the
          whole operative table and the true rows alone as references.

Two facts carried in from pp1 and not re-litigated here:
  * `vo_heads.pt`'s `core` and its `proj` are ONE PLANT UPDATE out of step (pp1 NOTES defect 1);
    gate F-2b's band is the readout's own per-refit drift, not a tolerance chosen here.
  * the executor's own prior (`dp_top`) already separated pp1's true from wrong CONSTRUCTED
    entries at 0.54-0.68, so the question the tables have to answer is what the read adds OVER
    the executor's prior, not whether it is above chance.

Seed families, disjoint from pp1's (`5_100_000` pools, `6_200_000` draws, `7_300_000` refits)
and from pp2's: the `8_400_000` family, split as
    price pool   cfg.seed + 8_400_000 + 1000*l
    test pools   cfg.seed + 8_440_000 + 1000*l + 137*d
    random draws cfg.seed * 97 + 8_470_000 + l
The REFIT subsample deliberately reuses pp1's `FIT_SEED_BASE`, so the frozen and twin controls
here are the same objects pp1 read and the two nodes' control columns are comparable.

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::own_gates
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::sweep_own \
        --out-tag pp3_smoke --arms s0_sv --smoke 1
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::sweep_own \
        --out-tag pp3
"""

import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay-own", image=image)

REMOTE = "rhm_practice_preplay"

# the free seed family, split three ways (see the module docstring)
OWN_POOL_BASE = 8_400_000
OWN_TEST_BASE = 8_440_000
OWN_RAND_BASE = 8_470_000

N_SCORE = 256                  # the loop's own audition size, pp1's too
N_TEST_POOLS = 4               # disjoint test-pool draws for step 3
N_RAND_DRAWS = 5               # uniform-random selectors at matched count
LEVELS = (2, 3, 4)             # L5 is out of reach; `learner_tables.py` says why

READS = ("banked", "shaped_refit", "frozen", "twin")
# the selectors of step 3. `world` and `true_only` are ORACLE selectors and are labelled as
# such in the reduction; they are the ceiling the endogenous ones are measured against.
SELECTORS = ("banked", "shaped_refit", "frozen", "twin", "dp_top", "world")


def topk(score, k, n):
    """The k highest-scoring row indices, ties broken by row order (stable), as the loop's own
    `rand_k`/argsort idiom does."""
    k = int(min(max(1, k), n))
    order = np.argsort(-np.asarray(score, np.float64), kind="mergesort")
    return np.sort(order[:k])


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=6144)
def own_arm(arm_key, out_tag, smoke=False, levels="", n_score=0, n_test=0, n_rand=0):
    import gzip
    import hashlib
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import learner_tables as LT
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    spec = PP.ARMS[arm_key]
    root_sb = os.path.join(DATA_DIR, PP.BANK_SB, spec["tag"], spec["arm"])
    ovs = PP.OVERTONE[spec["ov"]]
    root_ov = os.path.join(DATA_DIR, PP.BANK_OV, ovs["tag"], ovs["arm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want_levels = tuple(int(z) for z in levels.split(",") if z) or LEVELS
    NS = int(n_score) or N_SCORE
    NT = int(n_test) or N_TEST_POOLS
    NR = int(n_rand) or N_RAND_DRAWS
    if smoke:
        NS, NT, NR = 48, 2, 2
    log("=" * 104)
    log(f"[own] arm {arm_key}: {spec['tag']}/{spec['arm']}   overtone {ovs['tag']}/{ovs['arm']}"
        f"   device {device}   smoke={bool(smoke)}")
    log(f"  levels {want_levels}  n_score {NS}  test pools {NT}  random draws {NR}")
    log("=" * 104)

    # ---- the banked state (pp1's loader, unchanged in kind) ----------------------------- #
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m = int(cfgr["m"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
    tdim = dim
    assert str(cfgh.get("vo_om_mode")) == "proj" and str(cfgh.get("vo_pj_trunk")) == "live"
    assert bool(cfgh.get("vo_pj_mask")) and str(cfgh.get("vo_pj_root")) == "inter"
    log(f"  cfg v{v} s{s} L{depth} m{m} rule_seed {cfgr['rule_seed']} seed {cfgr['seed']} "
        f"dim {dim}  n_blocks {nb}  spell_cap {res_j['quotient']['spell_cap']}  "
        f"mine_support {cfgr['mine_support']}")

    rules = generate_rules_distinct(v, s, depth, m, seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)

    def mk_core():
        return GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                     root_conditioned=False).to(device)

    shaped = mk_core(); shaped.load_state_dict(blob["core"]); shaped.eval()
    ovblob = torch.load(os.path.join(root_ov, "vo_heads.pt"), map_location="cpu",
                        weights_only=True)
    frozen = mk_core(); frozen.load_state_dict(ovblob["core"]); frozen.eval()
    st_ = torch.get_rng_state(); torch.manual_seed(PP.TWIN_SEED)
    twin = mk_core(); torch.set_rng_state(st_); twin.eval()

    def fingerprint(c):
        with torch.no_grad():
            return float(sum(float(p_.detach().double().abs().sum()) for p_ in c.parameters()))
    fps = {k_: fingerprint(c_) for k_, c_ in
           (("shaped", shaped), ("frozen", frozen), ("twin", twin))}
    assert abs(fps["shaped"] - fps["frozen"]) > 1e-6, "shaped and frozen cores are identical"
    log(f"  trunk fingerprints: " + "  ".join(f"{k_} {q:.3f}" for k_, q in fps.items()))

    pj = blob["proj"]
    assert pj is not None and bool(pj["mask"]) and str(pj["root"]) == "inter"
    w_b, mu_b, sd_b = pj["w"], pj["mu"], pj["sd"]

    # ---- GATE F-2b, pp1's, verbatim in form -------------------------------------------- #
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
    log(f"  [F-2b] banked hold AUC {auc_banked:.6f}  re-read {auc_here:.6f}  "
        f"|d| {abs(auc_here - auc_banked):.4f}  band {band:.4f}  n {hold_idx.size}")
    assert abs(auc_here - auc_banked) < band, "F-2b FAILED"

    # ---- the two control readouts, on pp1's OWN refit slice ----------------------------- #
    trainable = (bank["h"] >= PP.PJ_HOLD)
    tr_all = np.nonzero(trainable & (bank["u0"] < PP.PJ_BOOT))[0]
    va_all = np.nonzero(trainable & (bank["u0"] >= PP.PJ_BOOT))[0]
    frng = np.random.default_rng(PP.FIT_SEED_BASE + int(cfgr["seed"]))
    idx_tr = (np.sort(frng.choice(tr_all, size=PP.PJ_FIT_CAP, replace=False))
              if tr_all.size > PP.PJ_FIT_CAP else tr_all)
    vcap = max(512, PP.PJ_FIT_CAP // 4)
    idx_va = (np.sort(frng.choice(va_all, size=vcap, replace=False))
              if va_all.size > vcap else va_all)
    fits = {}
    for nm, core_ in (("shaped_refit", shaped), ("frozen", frozen), ("twin", twin)):
        fits[nm] = PP.pj_fit(core_, bank, idx_tr, idx_va, s, nb, tdim, v, device,
                             log=log, name=nm)
    hold_read = {"banked": auc_here}
    for nm, core_ in (("shaped_refit", shaped), ("frozen", frozen), ("twin", twin)):
        f_ = fits[nm]
        p_ = PP.pj_predict(core_, bank["x"][hi], bank["r"][hi], bank["blk"][hi],
                           bank["spn"][hi], f_["w"], f_["mu"], f_["sd"],
                           s, nb, tdim, v, device)
        hold_read[nm] = PP.vo_auc(p_.numpy(), bank["y"][hi].numpy())
    log("  held-out AUC on the arm's own experience: " +
        "  ".join(f"{k_}={0.0 if q is None else q:.4f}" for k_, q in hold_read.items()))
    del bank, z

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

    # ---- STEP 1: the learner's own operative tables, reconstructed and gated ------------ #
    ent_path = os.path.join(root_sb, "entry.json.gz")
    ent_last = None
    if os.path.exists(ent_path):
        ent_last = json.load(gzip.open(ent_path, "rt"))[-1]
    else:
        log("  !! entry.json.gz absent on the volume: gate R-3 (the per-row truth vector "
            "against the banked operative_mask) CANNOT RUN on this arm")
    truth = MC.true_tables(rules, depth, s, v, m, max(want_levels))
    tables, diags, rgates = LT.reconstruct(
        res_j, ent_last if ent_last is not None else {"operative_mask": {}},
        truth=truth, levels=tuple(q for q in LT.LEVELS_OK if q in want_levels),
        arm_key=arm_key)
    log("  --- step 1: the operative tables, reconstructed ---")
    for g in rgates:
        if g.get("quiet") and g["pass"]:
            continue
        got, want = g["got"], g["want"]
        if isinstance(got, list) and len(got) > 6:
            got, want = f"<{len(got)}>", f"<{len(want)}>"
        log(f"    [{'PASS' if g['pass'] else 'FAIL'}] {g['gate']:<18} {g['what']:<50} "
            f"got {got} want {want}")
    bad = [g["gate"] for g in rgates if not g["pass"]]
    assert not bad, f"the reconstruction did not gate: {bad}"

    commits = {}
    for e_ in res_j["events"]:
        if e_.get("kind") == "commit":
            commits[int(e_["level"])] = int(e_["n_entries"])
    eras = {int(e["level"]): e for e in res_j["eras"]}
    log(f"  commits (the FROZEN table sizes, k of record) {commits}   "
        f"eras {[eras[k_]['name'] for k_ in sorted(eras)]}")

    out = {"arm": arm_key, "spec": spec, "hold_read": hold_read, "fingerprints": fps,
           "fits": {k_: {"lam": q["lam"], "val_auc": q["val_auc"]} for k_, q in fits.items()},
           "gate": {"F-2b:here": auc_here, "F-2b:banked": auc_banked,
                    "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band},
           "rgates": rgates, "commits": commits, "r3_ran": ent_last is not None,
           "cfg": {"n_score": NS, "n_test": NT, "n_rand": NR, "seed": int(cfgr["seed"]),
                   "levels": list(want_levels)},
           "cells": {}}
    gate_f1 = []
    n_aud = 0

    for ell in want_levels:
        if ell not in eras or ell not in tables:
            log(f"  L{ell}: no era or no reconstructed table, skipped")
            continue
        era, node = eras[ell], int(eras[ell]["node"])
        span = s ** (ell - 1)
        blk0 = node * span
        tbl = tables[ell]
        lower = tables[ell - 1]
        child = np.asarray(tbl["child"], np.int64)
        n_rows = int(child.shape[0])
        tvec = np.asarray(diags[ell]["truth_row"], np.int64)
        t_l0 = time.time()
        log("")
        log(f"  --- L{ell} at {era['name']} (node {node}, blk0 {blk0}, span {span}): "
            f"{n_rows} operative rows, {int(tvec.sum())} TRUE / {n_rows - int(tvec.sum())} "
            f"FALSE; lower table {int(lower['child'].shape[0])} rows ---")

        pool_seed = int(cfgr["seed"]) + OWN_POOL_BASE + 1000 * ell
        roots_np, x_np = context_instances(
            rules, {"name": era["name"], "level": ell, "nodes": [node]},
            NS, s, depth, v, m, seed=pool_seed)
        x0 = torch.from_numpy(x_np).to(device)
        roots_t = torch.from_numpy(roots_np).to(device).long()
        succ0, _ = grade(x_np, roots_np, rules, s)
        log(f"      price pool seed {pool_seed}  n {NS}  bootstrap success {succ0.mean():.4f}")

        f1_done = [False]

        def fire(rows_child, x_, roots_np_, want_reads, tag):
            """One fire of a table through the SHAPED executor + one world grade (+ the reads).
            Gate F-1 (pp1's) on the first fire of the level: the recording DP must reproduce
            `MC.apply_any` bit for bit, so the free-signal instrument cannot move a fired
            state."""
            nonlocal n_aud
            arr = np.ascontiguousarray(np.asarray(rows_child, np.int64).reshape(-1, s))
            tb = MC.make_table(ell, arr, lower, s)
            mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
            xf, dp = PP.fire_rec(shaped, x_, mv, canon, s)
            if not f1_done[0]:
                f1_done[0] = True
                xa = MC.apply_any(shaped, x_, mv, None, canon, depth, v, m, s)
                same = bool((xa == xf).all())
                gate_f1.append({"level": ell, "tag": tag, "identical": same,
                                "n_entries": int(arr.shape[0])})
                assert same, "F-1 FAILED: the recording DP does not reproduce `MC.apply_any`"
            succ, dres = grade(xf.cpu().numpy(), roots_np_, rules, s)
            n_aud += 1
            rec = {"succ": succ.astype("float32"), "e": float(1.0 - succ.mean()),
                   "dres": float(dres.mean()), "dp": dp}
            if want_reads:
                rec["reads"] = read_all(xf, roots_t, blk0, span)
            return rec

        # --- STEP 2: every operative row alone, priced ------------------------------------ #
        rows = []
        for i in range(n_rows):
            rec = fire(child[i], x0, roots_np, True, f"row{i}")
            r = {"i": i, "child": [int(q) for q in child[i]], "truth": int(tvec[i]),
                 "w_succ": float(rec["succ"].mean()), "e": rec["e"], "dres": rec["dres"]}
            for nm in READS:
                r[f"p_{nm}"] = float(rec["reads"][nm].mean())
            for k_ in ("top", "lse", "marg"):
                r[f"dp_{k_}"] = float(rec["dp"][k_].mean())
            rows.append(r)

        y = tvec.astype(np.float64)
        w = np.array([r["w_succ"] for r in rows], np.float64)
        scores = {"world": w, "dp_top": np.array([r["dp_top"] for r in rows], np.float64)}
        for nm in READS:
            scores[nm] = np.array([r[f"p_{nm}"] for r in rows], np.float64)
        stat = {"n_rows": n_rows, "n_true": int(tvec.sum()),
                "n_false": int(n_rows - tvec.sum()),
                "boot_success": float(succ0.mean()), "pool_seed": pool_seed}
        tmask = (tvec > 0)
        for nm, sc in scores.items():
            stat[f"sep_{nm}"] = PP.vo_auc(sc, y)
            stat[f"rho_{nm}"] = (None if nm == "world" else PP.spearman(sc, w))
            stat[f"rho_true_{nm}"] = (None if nm == "world" or tmask.sum() < 4
                                      else PP.spearman(sc[tmask], w[tmask]))
            stat[f"mean_true_{nm}"] = (float(sc[tmask].mean()) if tmask.any() else None)
            stat[f"mean_false_{nm}"] = (float(sc[~tmask].mean()) if (~tmask).any() else None)
        log(f"      [step 2] sep TRUE-vs-FALSE: " +
            "  ".join(f"{nm}={0.0 if stat[f'sep_{nm}'] is None else stat[f'sep_{nm}']:.3f}"
                      for nm in ("world", "banked", "shaped_refit", "frozen", "twin",
                                 "dp_top")))
        log(f"      [step 2] rho vs world (all rows): " +
            "  ".join(f"{nm}={0.0 if stat[f'rho_{nm}'] is None else stat[f'rho_{nm}']:.3f}"
                      for nm in ("banked", "shaped_refit", "frozen", "twin", "dp_top")))

        # --- STEP 3: the read as the SELECTOR, on a disjoint test pool -------------------- #
        k_big = int(min(commits.get(ell, max(1, n_rows // 2)), n_rows))
        k_small = int(max(2, round(k_big / 3.0)))
        # k = 1 is the limit of "a smaller k" and it is here because the first pass raised the
        # question: the whole operative table's audition error looked WORSE than its own best
        # single row's, across two different pools. A top-1 selection on the SAME test pools
        # settles that, and it is also the hardest selection problem a read can be asked.
        ks = sorted({k_big, k_small, 1}, reverse=True)
        rng_sel = np.random.default_rng(int(cfgr["seed"]) * 97 + OWN_RAND_BASE + ell)
        picks = {}
        for nm in SELECTORS:
            for k_ in ks:
                picks[(nm, k_)] = topk(scores[nm], k_, n_rows)
        for d in range(NR):
            for k_ in ks:
                picks[(f"random{d}", k_)] = np.sort(rng_sel.permutation(n_rows)[:k_])
        refs = {"all_rows": np.arange(n_rows),
                "true_only": np.nonzero(tmask)[0]}
        sel_stat = {}
        for key, ix in list(picks.items()) + [((nm, -1), ix) for nm, ix in refs.items()]:
            nm, k_ = key
            sel_stat[f"{nm}|{k_}"] = {"selector": nm, "k": int(k_), "n": int(ix.size),
                                      "n_true": int(tvec[ix].sum()),
                                      "n_false": int(ix.size - tvec[ix].sum()),
                                      "e": [], "idx": [int(q) for q in ix]}
        log(f"      [step 3] k {ks} (k_big = the frozen committed size); "
            f"{NT} disjoint test pools, {NR} random draws")
        for d in range(NT):
            ts = int(cfgr["seed"]) + OWN_TEST_BASE + 1000 * ell + 137 * d
            r_np, xt_np = context_instances(
                rules, {"name": era["name"], "level": ell, "nodes": [node]},
                NS, s, depth, v, m, seed=ts)
            xt = torch.from_numpy(xt_np).to(device)
            cache = {}
            for key, ix in list(picks.items()) + [((nm, -1), ix) for nm, ix in refs.items()]:
                nm, k_ = key
                if ix.size == 0:
                    sel_stat[f"{nm}|{k_}"]["e"].append(None)
                    continue
                h = hashlib.blake2b(np.ascontiguousarray(ix).tobytes(),
                                    digest_size=16).hexdigest()
                if h not in cache:
                    cache[h] = fire(child[ix], xt, r_np, False, f"sel:{nm}")["e"]
                sel_stat[f"{nm}|{k_}"]["e"].append(cache[h])
            del xt
        for key in sorted(sel_stat):
            q = sel_stat[key]
            es = [z_ for z_ in q["e"] if z_ is not None]
            q["e_mean"] = float(np.mean(es)) if es else None
            q["e_sd"] = float(np.std(es)) if es else None
        def _e(nm, k_):
            q = sel_stat.get(f"{nm}|{k_}", {})
            return q.get("e_mean")
        log("      [step 3] world's audition error, k_big:  " +
            "  ".join(f"{nm}={0.0 if _e(nm, k_big) is None else _e(nm, k_big):.4f}"
                      for nm in ("world", "banked", "shaped_refit", "frozen", "twin",
                                 "dp_top")) +
            f"   random={0.0 if _e('random0', k_big) is None else _e('random0', k_big):.4f}"
            f"   all_rows={0.0 if _e('all_rows', -1) is None else _e('all_rows', -1):.4f}"
            f"   true_only={0.0 if _e('true_only', -1) is None else _e('true_only', -1):.4f}")
        for k_ in ks[1:]:
            log(f"      [step 3] world's audition error, k={k_:<5}  " +
                "  ".join(f"{nm}={0.0 if _e(nm, k_) is None else _e(nm, k_):.4f}"
                          for nm in ("world", "banked", "shaped_refit", "frozen", "twin",
                                     "dp_top")) +
                f"   random={0.0 if _e('random0', k_) is None else _e('random0', k_):.4f}")
        log("      [step 3] false rows admitted at k_big:    " +
            "  ".join(f"{nm}={sel_stat[f'{nm}|{k_big}']['n_false']}"
                      for nm in ("world", "banked", "shaped_refit", "frozen", "twin",
                                 "dp_top")) +
            f"   (of {int(n_rows - tvec.sum())} false / {n_rows} rows)")

        out["cells"][str(ell)] = {
            "era": era["name"], "node": node, "blk0": blk0, "span": span,
            "n_lower": int(lower["child"].shape[0]), "k_big": k_big, "k_small": k_small,
            "ks": [int(q) for q in ks],
            "stat": stat, "rows": rows, "selectors": sel_stat,
            "grade": diags[ell].get("grade"),
            "sec": float(time.time() - t_l0)}
        log(f"      L{ell} done in {time.time() - t_l0:.1f}s   auditions so far {n_aud}")
        del x0, roots_t

    out["gate_f1"] = gate_f1
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
    with open(os.path.join(d, f"{arm_key}_own.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}_own.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"arm": arm_key, "sec": out["sec"], "n_aud": n_aud,
            "rss_mb": out["peak_rss_mb"], "r3_ran": out["r3_ran"]}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=10800, memory=2048)
def sweep_own(out_tag="pp3", smoke=False, arms="", levels="", n_score=0, n_test=0, n_rand=0):
    """CPU coordinator: the arms across separate containers. The arms are independent, so the
    fan-out adds no GPU-hours -- it only stops them queueing behind one another."""
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    want = [x for x in arms.split(",") if x] or list(PP.ARMS)
    args = [(k, out_tag, bool(smoke), levels, int(n_score), int(n_test), int(n_rand))
            for k in want]
    outs = list(own_arm.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done_own.txt"), "w") as fh:
        fh.write(json.dumps({"arms": want, "smoke": bool(smoke), "levels": levels,
                             "n_score": int(n_score), "n_test": int(n_test),
                             "n_rand": int(n_rand)}) + "\n")
    volume.commit()
    return outs


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800, memory=2048)
def own_gates(arms=""):
    """The reconstruction gates (R-0..R-3) and their falsification, on the VOLUME's own copies
    of `results.json` and `entry.json.gz` -- so what the paid run will price is what is gated
    here, and not a local mirror of it. CPU only: nothing in step 1 touches the plant."""
    import gzip
    import rhm.practice.ratchet.macros as MC
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import learner_tables as LT

    volume.reload()
    want = [x for x in arms.split(",") if x] or list(PP.ARMS)
    report = {}
    n_fail = n_miss = 0
    for key in want:
        spec = PP.ARMS[key]
        root = os.path.join(DATA_DIR, PP.BANK_SB, spec["tag"], spec["arm"])
        res_j = json.load(open(os.path.join(root, "results.json")))
        ep = os.path.join(root, "entry.json.gz")
        have_ent = os.path.exists(ep)
        ent_last = (json.load(gzip.open(ep, "rt"))[-1] if have_ent
                    else {"operative_mask": {}})
        cfg = res_j["config"]
        rules = generate_rules_distinct(int(cfg["v"]), int(cfg["s"]), int(cfg["depth"]),
                                        int(cfg["m"]), seed=int(cfg["rule_seed"]))
        truth = MC.true_tables(rules, int(cfg["depth"]), int(cfg["s"]), int(cfg["v"]),
                               int(cfg["m"]), max(LT.LEVELS_OK))
        tables, diags, gs = LT.reconstruct(res_j, ent_last, truth=truth, arm_key=key)
        print("=" * 100)
        print(f"{key}  volume copy  entry.json.gz {'present' if have_ent else 'ABSENT'}")
        for g in gs:
            if g.get("quiet") and g["pass"]:
                continue
            got, wnt = g["got"], g["want"]
            if isinstance(got, list) and len(got) > 6:
                got, wnt = f"<{len(got)}>", f"<{len(wnt)}>"
            print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {g['gate']:<18} "
                  f"{g['what']:<50} got {got} want {wnt}")
            n_fail += int(not g["pass"])
        cases = LT.falsify(res_j, ent_last, truth, arm_key=key)
        n_miss += sum(int(not c["caught"]) for c in cases)
        report[key] = {"have_entry": have_ent, "gates": gs, "falsify": cases,
                       "sizes": {str(e): diags[e]["n_entries"] for e in diags},
                       "true_rows": {str(e): int(sum(diags[e]["truth_row"]))
                                     for e in diags if diags[e].get("truth_row")}}
    print("=" * 100)
    print(f"gate failures {n_fail}   falsification misses {n_miss}")
    assert n_fail == 0 and n_miss == 0, "the reconstruction does not gate on the volume"
    return report
