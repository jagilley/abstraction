"""[preplay/timbre] pp1, pp3 and pp5 RE-READ ON RICHER READER FORMS -- offline, banked state,
nothing paid in the loop.

`preplay`'s rounds read one object: the shaped plant's pooled state through a ridge-logistic
readout (`VoProjBank`'s form). pp1 found the projection's mean LEVEL over a fired candidate's
configurations separates true from wrong entries with shaped above frozen above twin on 8 of 8
cells (frozen and twin at chance); pp5 found the same level, as a keep/refuse gate on the
preplayed state, captures 0.27 / 0.57 of the oracle's advantage while the frozen and twin levels
end WORSE than no gate. Both were measured with the linear readout only. On the other substrate
(`logit_reading/striatum/norm/precision` sections 8 and 10) every claim about a reader's LEVEL held
on richer readers and every claim about its RESPONSE, a difference of levels, depended on the
form. This node re-reads the same fires with the reader forms of `readers.py` -- the ridge of
record (arm 0), the plant's belief appended, an MLP on the state, an MLP on the belief alone, and
(pp1 / pp3 only) the belief-only ridge and a per-slot ridge -- each refit on pp1's own shared bank
subsample through the shaped, frozen and twin trunks.

THE PATTERN (`precision/rereads/`): fork the banked machinery with a reader selector, gate arm 0
against the banked columns IN-CONTAINER before anything else is read, run the banked reductions
unchanged on each reader's columns.

  rr1_arm   pp1 (`preplay.preplay_arm`: single and delta forms, same pools, same draws, same
            candidates) and pp3 step 2 (`own.own_arm`: the learner's own operative rows, priced
            one at a time), every reader on the SAME fires. Gate T-5: arm 0's columns
            (`banked`, `ridge@shaped/frozen/twin`) against `/rhm_practice_preplay/pp1/<arm>.json`
            and `pp3/<arm>_own.json`, per candidate, per row.
  rr5_arm   pp5 (`readgate.readgate_arm`: `dp_top` as the order, the gate varied), the banked
            gates under their own names plus every reader's level as a gate (pooled, paired,
            margin through the shaped trunk; pooled through frozen and twin). Gate T-6: the
            banked gates' admissions, confusion counts and test errors against
            `/rhm_practice_preplay/pp5/<arm>.json`, per cell and repeat, EXACTLY.

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/timbre.py::gates_t
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/timbre.py::falsify_t
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/timbre.py::sweep \\
        --out-tag tb_smoke --arms1 s0_sv --arms5 s0_sv --smoke 1
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/timbre.py::sweep \\
        --out-tag tb1
"""

import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay-timbre", image=image)

REMOTE = "rhm_practice_preplay"
FORMS_1 = ("ridge", "belief", "bonly_lin", "mlp", "bonly")        # + `slot`, per level
FORMS_5 = ("ridge", "belief", "mlp", "bonly")
ARMS_1 = "s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd"
ARMS_5 = "s0_sv,s2_sv"
# arm 0: pp1's column name -> this node's reader key
ARM0 = {"banked": "banked", "shaped_refit": "ridge@shaped", "frozen": "ridge@frozen",
        "twin": "ridge@twin"}
TOL_P = 1e-5            # per-candidate mean level, arm 0 against the banked column
TOL_E = 1e-12           # world errors: the same fire, so exact


# --------------------------------------------------------------------------------------- #
# the shared loader: banked state, the three trunks, F-2b, pp1's refit slice, every reader
# --------------------------------------------------------------------------------------- #

def _load(arm_key, device, log):
    import torch
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP

    spec = PP.ARMS[arm_key]
    root_sb = os.path.join(DATA_DIR, PP.BANK_SB, spec["tag"], spec["arm"])
    ovs = PP.OVERTONE[spec["ov"]]
    root_ov = os.path.join(DATA_DIR, PP.BANK_OV, ovs["tag"], ovs["arm"])
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m, maxl = int(cfgr["m"]), int(cfgr["max_macro_level"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
    assert str(cfgh.get("vo_om_mode")) == "proj", cfgh.get("vo_om_mode")
    assert str(cfgh.get("vo_pj_trunk")) == "live", cfgh.get("vo_pj_trunk")
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
                           bank["spn"][hi], w_b, mu_b, sd_b, s, nb, dim, v, device)
    auc_here = PP.vo_auc(p_hold.numpy(), bank["y"][hi].numpy())
    oms = [q for q in res_j["log"]["vo_om"]
           if isinstance(q, dict) and q.get("hold_auc") is not None]
    auc_banked = float(oms[-1]["hold_auc"])
    drifts = np.array([abs(q["hold_auc_drift"]) for q in oms
                       if q.get("hold_auc_drift") is not None][-60:], dtype=float)
    band = float(max(0.02, 2.0 * np.quantile(drifts, 0.9))) if drifts.size else 0.05
    gate = {"F-2b:hold_auc_here": auc_here, "F-2b:hold_auc_banked": auc_banked,
            "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band,
            "F-2b:drift_p90": (float(np.quantile(drifts, 0.9)) if drifts.size else None),
            "F-2b:n_hold": int(hold_idx.size), "F-2b:n_bank": int(bank["x"].shape[0])}
    log(f"  [F-2b] banked hold AUC {auc_banked:.6f}  re-read {auc_here:.6f}  "
        f"|d| {abs(auc_here - auc_banked):.4f}  band {band:.4f}")
    assert abs(auc_here - auc_banked) < band, "F-2b FAILED"

    # pp1's refit slice, verbatim (FIT_SEED_BASE is shared by pp1..pp5)
    trainable = (bank["h"] >= PP.PJ_HOLD)
    tr_all = np.nonzero(trainable & (bank["u0"] < PP.PJ_BOOT))[0]
    va_all = np.nonzero(trainable & (bank["u0"] >= PP.PJ_BOOT))[0]
    frng = np.random.default_rng(PP.FIT_SEED_BASE + int(cfgr["seed"]))
    idx_tr = (np.sort(frng.choice(tr_all, size=PP.PJ_FIT_CAP, replace=False))
              if tr_all.size > PP.PJ_FIT_CAP else tr_all)
    vcap = max(512, PP.PJ_FIT_CAP // 4)
    idx_va = (np.sort(frng.choice(va_all, size=vcap, replace=False))
              if va_all.size > vcap else va_all)
    # the three index sets the readers touch are disjoint by construction (h / u0 splits);
    # asserted, so a reader can never be scored on a row it was selected on
    assert not (set(idx_tr.tolist()) & set(idx_va.tolist()))
    assert not ((set(idx_tr.tolist()) | set(idx_va.tolist())) & set(hold_idx.tolist()))
    return {"spec": spec, "ovs": ovs, "root_sb": root_sb, "res_j": res_j, "cfgr": cfgr,
            "v": v, "s": s, "depth": depth, "m": m, "maxl": maxl, "dim": dim,
            "length": length, "nb": nb, "rules": rules, "canon": canon,
            "cores": {"shaped": shaped, "frozen": frozen, "twin": twin}, "fps": fps,
            "banked": (w_b, mu_b, sd_b), "bank": bank, "hold_idx": hold_idx,
            "gate": gate, "idx_tr": idx_tr, "idx_va": idx_va, "auc_banked_here": auc_here}


def _fit_all(L, forms, device, log):
    """Every reader form through every trunk on pp1's shared subsample. `ridge` is
    `preplay.pj_fit` itself (the pp1 objects); the rest share one feature pass per trunk."""
    import torch
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD
    bank, s, nb, dim, v = L["bank"], L["s"], L["nb"], L["dim"], L["v"]
    idx_tr, idx_va = L["idx_tr"], L["idx_va"]
    idx = np.concatenate([idx_tr, idx_va])
    ii = torch.from_numpy(idx)
    y = bank["y"][ii]
    r = bank["r"][ii]
    ntr = int(idx_tr.size)
    hi = torch.from_numpy(L["hold_idx"])
    yh = bank["y"][hi].numpy()
    heads, recs, hold = {}, {}, {"banked": L["auc_banked_here"]}
    t0 = time.time()
    for tk in RD.TRUNKS:
        core = L["cores"][tk]
        if "ridge" in forms:
            f_ = PP.pj_fit(core, bank, idx_tr, idx_va, s, nb, dim, v, device)
            f_["kind"] = "lin"
            f_["nfeat"] = int(f_["w"].numel())
            heads[f"ridge@{tk}"] = f_
        rest = [f for f in forms if f not in ("ridge", "slot")]
        if rest:
            Fh, Fb = RD.feats(core, bank["x"][ii], bank["blk"][ii], bank["spn"][ii],
                              s, nb, dim, device)
            for f in rest:
                heads[f"{f}@{tk}"] = RD.fit_form(f, Fh, Fb, r, y, ntr, v, device)
        Hh, Hb = RD.feats(core, bank["x"][hi], bank["blk"][hi], bank["spn"][hi],
                          s, nb, dim, device)
        for k in [k for k in heads if k.endswith("@" + tk)]:
            p_ = RD.predict(heads[k], k.split("@")[0], Hh, Hb, bank["r"][hi], v, device)
            hold[k] = PP.vo_auc(p_.numpy(), yh)
    for k, hd in heads.items():
        recs[k] = RD.head_record(hd)
    log(f"  readers fit in {time.time() - t0:.1f}s:")
    for k in sorted(heads):
        q = recs[k]
        extra = (f"lam {q['lam']:>6.0f}" if q["kind"] == "lin"
                 else f"lr {q['lr']:g} wd {q['wd']:<5g} step {q['step']:>4d}")
        log(f"    {k:<18} nfeat {q['nfeat']:>5d}  {extra}  val {q['val_auc']:.4f}  "
            f"hold {0.0 if hold.get(k) is None else hold[k]:.4f}")
    return heads, recs, hold


def _fit_slot(L, blk0, span, device, log):
    """The per-slot form at the fired cell: the projection's design fit on ALL the bank's
    trainable rows at this (blk0, span), the u0 split as the train / validation split. None where
    the slot is too thin to fit (SLOT_MIN_TR / SLOT_MIN_VA)."""
    import torch
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD
    bank, s, nb, dim, v = L["bank"], L["s"], L["nb"], L["dim"], L["v"]
    at = (bank["blk"].numpy() == int(blk0)) & (bank["spn"].numpy() == int(span))
    trainable = bank["h"] >= PP.PJ_HOLD
    tr = np.nonzero(at & trainable & (bank["u0"] < PP.PJ_BOOT))[0]
    va = np.nonzero(at & trainable & (bank["u0"] >= PP.PJ_BOOT))[0]
    ho = np.nonzero(at & (bank["h"] < PP.PJ_HOLD))[0]
    info = {"n_tr": int(tr.size), "n_va": int(va.size), "n_hold": int(ho.size)}
    if tr.size < RD.SLOT_MIN_TR or va.size < RD.SLOT_MIN_VA:
        log(f"    slot ({blk0},{span}): {tr.size} train / {va.size} val rows -- too thin, "
            f"no per-slot reader at this cell")
        return {}, {}, info
    idx = np.concatenate([tr, va])
    ii = torch.from_numpy(idx)
    heads, hold = {}, {}
    for tk in RD.TRUNKS:
        Fh, Fb = RD.feats(L["cores"][tk], bank["x"][ii], bank["blk"][ii], bank["spn"][ii],
                          s, nb, dim, device)
        hd = RD.fit_lin(Fh, bank["r"][ii], bank["y"][ii], int(tr.size), v, device)
        if hd is None:
            continue
        hd["n_tr"], hd["n_va"] = int(tr.size), int(va.size)
        heads[f"slot@{tk}"] = hd
        if ho.size >= 32:
            hh = torch.from_numpy(ho)
            Hh, Hb = RD.feats(L["cores"][tk], bank["x"][hh], bank["blk"][hh], bank["spn"][hh],
                              s, nb, dim, device)
            hold[f"slot@{tk}"] = PP.vo_auc(RD.lin_predict(Hh, bank["r"][hh], hd, v,
                                                          device).numpy(),
                                           bank["y"][hh].numpy())
    log(f"    slot ({blk0},{span}): {tr.size} train / {va.size} val / {ho.size} hold rows; " +
        "  ".join(f"{k} lam {heads[k]['lam']:.0f} val {heads[k]['val_auc']:.3f} hold "
                  f"{0.0 if hold.get(k) is None else hold[k]:.3f}" for k in sorted(heads)))
    return heads, hold, info


def _cmp(tag, got, want, tol, bad, worst):
    d = abs(float(got) - float(want))
    worst[tag] = max(worst.get(tag, 0.0), d)
    if not d <= tol:
        bad.append(f"{tag}: {got!r} != {want!r} (|d| {d:.3e} > {tol:.0e})")


def cmp_pp1_cell(cell, rc):
    """GATE T-5 on one pp1 cell: arm 0's columns against the banked ones, candidate by
    candidate, in both forms, plus the per-instance transfer AUCs. Returns (bad, worst)."""
    bad, worst = [], {}
    for form in ("single", "delta"):
        mine, theirs = cell[form], rc[form]
        if len(mine) != len(theirs):
            bad.append(f"{form}: {len(mine)} rows against {len(theirs)}")
            continue
        for a_, b_ in zip(mine, theirs):
            if a_["cls"] != b_["cls"] or list(a_["cand"]) != list(b_["cand"]):
                bad.append(f"{form}: candidate {a_['cand']} against {b_['cand']}")
                break
            wk = "w_succ" if form == "single" else "w_delta"
            _cmp(f"{form}:world", a_[wk], b_[wk], TOL_E, bad, worst)
            _cmp(f"{form}:dp_top", a_["dp_top"], b_["dp_top"], 1e-6, bad, worst)
            for nm, key in ARM0.items():
                _cmp(f"{form}:p_{nm}", a_[f"p_{key}"], b_[f"p_{nm}"], TOL_P, bad, worst)
                if form == "delta":
                    _cmp(f"{form}:r_{nm}", a_[f"r_{key}"], b_[f"r_{nm}"], TOL_P, bad, worst)
    for nm, key in ARM0.items():
        for form in ("single", "delta"):
            if form in cell.get("instance", {}) and form in rc.get("instance", {}):
                _cmp(f"inst:{form}:{nm}", cell["instance"][form][f"auc_{key}"],
                     rc["instance"][form][f"auc_{nm}"], 1e-4, bad, worst)
    return bad, worst


def cmp_pp3_rows(rows, ref_rows):
    """GATE T-5 on one pp3 level: the learner's rows, the world's price, arm 0's columns."""
    bad, worst = [], {}
    if len(rows) != len(ref_rows):
        return [f"{len(rows)} rows against {len(ref_rows)}"], worst
    for a_, b_ in zip(rows, ref_rows):
        if list(a_["child"]) != list(b_["child"]) or a_["truth"] != b_["truth"]:
            bad.append(f"row {a_['i']}: {a_['child']} against {b_['child']}")
            break
        _cmp("world", a_["w_succ"], b_["w_succ"], TOL_E, bad, worst)
        for nm, key in ARM0.items():
            _cmp(f"p_{nm}", a_[f"p_{key}"], b_[f"p_{nm}"], TOL_P, bad, worst)
    return bad, worst


def cmp_pp5_rep(rep, rr):
    """GATE T-6 on one pp5 (cell, repeat): every banked order|gate walk, exactly -- the
    admission list, the confusion counts, the audition count, the gate-pool and test errors."""
    bad, worst = [], {}
    for fld in ("n_base", "n_live"):
        if rep[fld] != rr[fld]:
            bad.append(f"{fld}: {rep[fld]} against {rr[fld]}")
    _cmp("delta", rep["delta"], rr["delta"], 1e-7, bad, worst)
    _cmp("p_base", rep["p_base"], rr["p_base"], TOL_P, bad, worst)
    _cmp("e_base_test", rep["e_base_test"]["mean"], rr["e_base_test"]["mean"], TOL_E, bad,
         worst)
    for k in rr["orders"]:
        if k not in rep["orders"]:
            bad.append(f"{k}: not walked")
            continue
        a_, b_ = rep["orders"][k], rr["orders"][k]
        for fld in ("admitted", "rejected", "conf", "n_auditions", "n_kept", "n_wrong_kept"):
            if a_[fld] != b_[fld]:
                bad.append(f"{k}:{fld}: {a_[fld]} against {b_[fld]}")
        _cmp(f"{k}:e_gate_final", a_["e_gate_final"], b_["e_gate_final"], TOL_E, bad, worst)
        _cmp(f"{k}:e_test_final", a_["e_test_final"]["mean"], b_["e_test_final"]["mean"],
             TOL_E, bad, worst)
        for bb in b_["budget"]:
            _cmp(f"{k}:e_test@{bb}", a_["budget"][bb]["e_test"], b_["budget"][bb]["e_test"],
                 TOL_E, bad, worst)
    return bad, worst


def as_mine_pp1(cell):
    """A banked pp1 cell rewritten in this node's column names (arm 0 only), so the comparator
    can be run on it: identity must pass, a perturbation or another arm must fail."""
    import copy as _c
    c = _c.deepcopy(cell)
    for form in ("single", "delta"):
        for q in c[form]:
            for nm, key in ARM0.items():
                q[f"p_{key}"] = q[f"p_{nm}"]
                if form == "delta":
                    q[f"r_{key}"] = q[f"r_{nm}"]
    for form, b in c.get("instance", {}).items():
        for nm, key in ARM0.items():
            b[f"auc_{key}"] = b[f"auc_{nm}"]
    return c


# --------------------------------------------------------------------------------------- #
# rr1: pp1 and pp3 step 2, every reader on the same fires
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=8192)
def rr1_arm(arm_key, out_tag, smoke=False, levels="", forms="", own=True, gate=True):
    import gzip
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import own as OW
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import learner_tables as LT
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want_levels = tuple(int(z) for z in levels.split(",") if z) or PP.LEVELS
    forms_ = tuple(f for f in forms.split(",") if f) or FORMS_1
    assert "ridge" in forms_, "arm 0 (the ridge) is the gate and cannot be dropped"
    NS, KD, KS, ND = PP.N_SCORE, PP.K_DELTA, PP.K_SINGLE, PP.N_BASE_DRAWS
    if smoke:
        NS, KD, KS, ND = 64, 4, 6, 2
    do_gate = bool(gate) and not smoke
    log("=" * 100)
    log(f"[timbre rr1] arm {arm_key}  device {device}  smoke={bool(smoke)}  gate={do_gate}")
    log(f"  levels {want_levels}  forms {forms_} (+slot)  n_score {NS}  k_delta {KD}  "
        f"k_single {KS}  draws {ND}")
    log("=" * 100)
    L = _load(arm_key, device, log)
    v, s, depth, m, maxl, nb, dim = (L["v"], L["s"], L["depth"], L["m"], L["maxl"], L["nb"],
                                     L["dim"])
    rules, canon, cfgr, res_j = L["rules"], L["canon"], L["cfgr"], L["res_j"]
    shaped = L["cores"]["shaped"]
    heads, head_recs, hold_read = _fit_all(L, forms_, device, log)

    ref1 = ref3 = None
    if do_gate:
        ref1 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp1", f"{arm_key}.json")))
        p3 = os.path.join(DATA_DIR, REMOTE, "pp3", f"{arm_key}_own.json")
        ref3 = json.load(open(p3)) if os.path.exists(p3) else None
        # the refits ARE pp1's objects: the selected ridge and validation AUC must agree
        bad, worst = [], {}
        for nm, key in (("shaped_refit", "ridge@shaped"), ("frozen", "ridge@frozen"),
                        ("twin", "ridge@twin")):
            _cmp(f"fit:{nm}:lam", head_recs[key]["lam"], ref1["fits"][nm]["lam"], 0.0,
                 bad, worst)
            _cmp(f"fit:{nm}:val_auc", head_recs[key]["val_auc"], ref1["fits"][nm]["val_auc"],
                 1e-9, bad, worst)
            _cmp(f"hold:{nm}", hold_read[key], ref1["hold_read"][nm], 1e-6, bad, worst)
        _cmp("hold:banked", hold_read["banked"], ref1["hold_read"]["banked"], 1e-9, bad, worst)
        log(f"  [T-5 fits] {'PASS' if not bad else 'FAIL'}  worst " +
            "  ".join(f"{k} {q:.1e}" for k, q in worst.items()))
        assert not bad, "T-5 FAILED on the refits: " + "; ".join(bad)

    # the leakage count: fired / pool configurations equal to a row any reader was fit on
    bank = L["bank"]
    fit_rows = np.concatenate([L["idx_tr"], L["idx_va"]])
    bank_all_keys = set(RD.xr_keys(bank["x"].numpy(), bank["r"].numpy()).tolist())
    bank_fit_keys = set(RD.xr_keys(bank["x"].numpy()[fit_rows],
                                   bank["r"].numpy()[fit_rows]).tolist())

    def leak(x_np, r_np, acc):
        ks = RD.xr_keys(x_np, r_np).tolist()
        acc["n"] += len(ks)
        acc["fit"] += sum(1 for k_ in ks if k_ in bank_fit_keys)
        acc["bank"] += sum(1 for k_ in ks if k_ in bank_all_keys)

    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {}
    for e_ in res_j["events"]:
        if e_.get("kind") == "commit":
            commits[int(e_["level"])] = int(e_["n_entries"])
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)

    out = {"arm": arm_key, "spec": L["spec"], "gate": L["gate"], "hold_read": hold_read,
           "heads": head_recs, "fingerprints": L["fps"], "n_bank": int(bank["x"].shape[0]),
           "cells": {}, "own": {},
           "cfg": {"n_score": NS, "k_delta": KD, "k_single": KS, "draws": ND,
                   "levels": list(want_levels), "seed": int(cfgr["seed"]),
                   "rule_seed": int(cfgr["rule_seed"]), "twin_seed": PP.TWIN_SEED,
                   "forms": list(forms_), "smoke": bool(smoke), "gate": do_gate}}
    gate_rec = {"pp1": {}, "pp3": {}}
    n_aud_total = 0
    gate_f1, gate_f1_done = [], set()
    slot_cache = {}

    def slot_heads(blk0_, span_, log_):
        if (blk0_, span_) not in slot_cache:
            slot_cache[(blk0_, span_)] = _fit_slot(L, blk0_, span_, device, log_)
        return slot_cache[(blk0_, span_)]

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
        n_base = commits.get(ell)
        base_src = "commit"
        if n_base is None:
            n_base = int(aud_last.get(str(ell), {}).get("n_entries") or 0)
            base_src = "live_table_last_cycle"
        n_base = max(1, min(int(n_base), n_true_rows))
        t_l0 = time.time()
        log("")
        log(f"  --- L{ell} at cell {era['name']} (node {node}, blk0 {blk0}, span {span}) "
            f"base {n_base} ({base_src})")
        sh, sh_hold, sh_info = slot_heads(blk0, span, log)
        heads_l = dict(heads)
        heads_l.update(sh)
        keys_l = ["banked"] + sorted(heads_l)

        pool_seed = int(cfgr["seed"]) + PP.POOL_SEED_BASE + 1000 * ell
        roots_np, x_np = context_instances(
            rules, {"name": era["name"], "level": ell, "nodes": [node]},
            NS, s, depth, v, m, seed=pool_seed)
        x0 = torch.from_numpy(x_np).to(device)
        roots_t = torch.from_numpy(roots_np).to(device).long()
        succ0, _ = grade(x_np, roots_np, rules, s)
        lk = {"n": 0, "fit": 0, "bank": 0}
        leak(x_np, roots_np, lk)
        rng = np.random.default_rng(PP.DRAW_SEED_BASE + int(cfgr["seed"]) * 97 + ell)
        bases = []
        for d in range(ND):
            bases.append(np.sort(rng.permutation(n_true_rows)[:n_base]))
        k_true_single = min(KS, n_true_rows)
        cand_true_single = np.sort(rng.permutation(n_true_rows)[:k_true_single]).tolist()
        n_lower = int(lower["flat"].shape[0])
        cand_wrong = PP.wrong_rows(n_lower, true_flat, lower["flat"], s, max(KS, KD), rng)
        cache = {}

        def audition(child_rows, tag):
            nonlocal n_aud_total
            arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
            key = (int(arr.shape[0]),
                   hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
            if key in cache:
                return cache[key]
            tb = MC.make_table(ell, arr, lower, s)
            mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
            xf, dp = PP.fire_rec(shaped, x0, mv, canon, s)
            if ell not in gate_f1_done:
                gate_f1_done.add(ell)
                xa = MC.apply_any(shaped, x0, mv, None, canon, depth, v, m, s)
                same = bool((xa == xf).all())
                gate_f1.append({"level": ell, "tag": tag, "identical": same,
                                "n_entries": int(tb["child"].shape[0])})
                assert same, "F-1 FAILED"
            xf_np = xf.cpu().numpy()
            succ, dres = grade(xf_np, roots_np, rules, s)
            leak(xf_np, roots_np, lk)
            rd = RD.read_heads(L["cores"], heads_l, L["banked"], xf, roots_t, blk0, span,
                               s, nb, dim, v, device)
            n_aud_total += 1
            rec = {"e": float(1.0 - succ.mean()), "succ": succ.astype("float32"),
                   "dres": float(dres.mean()), "reads": rd, "dp": dp,
                   "n_entries": int(tb["child"].shape[0])}
            cache[key] = rec
            return rec

        def summarize(rec):
            d = {"e": rec["e"], "dres": rec["dres"], "succ": float(rec["succ"].mean()),
                 "n_entries": rec["n_entries"]}
            for nm in keys_l:
                d[f"p_{nm}"] = float(rec["reads"][nm].mean())
            for k_ in ("top", "lse", "marg"):
                d[f"dp_{k_}"] = float(rec["dp"][k_].mean())
            return d

        cell = {"era": era["name"], "node": node, "blk0": blk0, "span": span,
                "n_true_rows": n_true_rows, "n_true_flat": len(true_flat),
                "n_base": n_base, "base_src": base_src, "pool_seed": pool_seed,
                "boot_success": float(succ0.mean()), "n_lower": n_lower, "keys": keys_l,
                "slot": {"info": sh_info, "hold": sh_hold,
                         "heads": {k: RD.head_record(q) for k, q in sh.items()}},
                "delta": [], "single": []}
        base_recs = [audition(truth_l["child"][k_], f"base{d}") for d, k_ in enumerate(bases)]
        cell["bases"] = [summarize(q) for q in base_recs]
        cell["ceiling"] = summarize(audition(truth_l["child"], "true_full"))
        inst = {"delta": [], "single": []}

        for d, keep in enumerate(bases):
            in_base = set(int(q) for q in keep)
            base_child = [list(map(int, r_)) for r_ in truth_l["child"][keep]]
            br = base_recs[d]
            avail_true = [i_ for i_ in range(n_true_rows) if i_ not in in_base]
            take_true = (np.sort(rng.permutation(len(avail_true))[:KD]).tolist()
                         if len(avail_true) > KD else list(range(len(avail_true))))
            pick_true = [avail_true[i_] for i_ in take_true]
            pick_wrong = cand_wrong[:KD]
            for cls, rows in (("true", [list(map(int, truth_l["child"][i_]))
                                        for i_ in pick_true]),
                              ("wrong", [list(r_) for r_ in pick_wrong])):
                for ci, cr in enumerate(rows):
                    rec = audition(base_child + [cr], f"d{d}:{cls}{ci}")
                    row = {"draw": d, "cls": cls, "cand": cr,
                           "w_delta": float(br["e"] - rec["e"]),
                           "e_with": rec["e"], "e_base": br["e"],
                           "succ_with": float(rec["succ"].mean()),
                           "n_entries": rec["n_entries"]}
                    for nm in keys_l:
                        row[f"r_{nm}"] = float(rec["reads"][nm].mean()
                                               - br["reads"][nm].mean())
                        row[f"p_{nm}"] = float(rec["reads"][nm].mean())
                    for k_ in ("top", "lse", "marg"):
                        row[f"dp_{k_}"] = float(rec["dp"][k_].mean() - br["dp"][k_].mean())
                    row["n_moved"] = int((rec["dp"]["entry"] == rec["n_entries"] - 1).sum())
                    cell["delta"].append(row)
                    inst["delta"].append((rec["succ"], rec["reads"], rec["dp"]["lse"]))

        for cls, rows in (("true", [list(map(int, truth_l["child"][i_]))
                                    for i_ in cand_true_single]),
                          ("wrong", [list(r_) for r_ in cand_wrong[:KS]])):
            for ci, cr in enumerate(rows):
                rec = audition([cr], f"s:{cls}{ci}")
                row = {"cls": cls, "cand": cr, "w_succ": float(rec["succ"].mean()),
                       "e": rec["e"], "dres": rec["dres"]}
                for nm in keys_l:
                    row[f"p_{nm}"] = float(rec["reads"][nm].mean())
                for k_ in ("top", "lse", "marg"):
                    row[f"dp_{k_}"] = float(rec["dp"][k_].mean())
                cell["single"].append(row)
                inst["single"].append((rec["succ"], rec["reads"], rec["dp"]["lse"]))

        cell["instance"] = {}
        for form in ("delta", "single"):
            if not inst[form]:
                continue
            ys = np.concatenate([q[0] for q in inst[form]])
            blk_ = {"n": int(ys.size), "base": float(ys.mean())}
            for nm in keys_l:
                ps = np.concatenate([q[1][nm] for q in inst[form]])
                blk_[f"auc_{nm}"] = PP.vo_auc(ps, ys)
                blk_[f"pbar_{nm}"] = float(ps.mean())
            blk_["auc_dp_lse"] = PP.vo_auc(np.concatenate([q[2] for q in inst[form]]), ys)
            cell["instance"][form] = blk_
        cell["leak"] = lk
        cell["sec"] = float(time.time() - t_l0)
        out["cells"][str(ell)] = cell

        # ---- GATE T-5 (pp1): arm 0 against the banked columns, candidate by candidate ---- #
        if do_gate:
            rc = ref1["cells"].get(str(ell))
            assert rc is not None, f"T-5: banked pp1 has no cell L{ell}"
            bad, worst = cmp_pp1_cell(cell, rc)
            gate_rec["pp1"][str(ell)] = {"pass": not bad, "worst": worst, "bad": bad[:8],
                                         "n_single": len(cell["single"]),
                                         "n_delta": len(cell["delta"])}
            log(f"      [T-5 pp1 L{ell}] {'PASS' if not bad else 'FAIL'}  "
                f"{len(cell['single'])} single + {len(cell['delta'])} delta rows; worst |d|: " +
                "  ".join(f"{k} {q:.1e}" for k, q in sorted(worst.items())
                          if k.startswith(("single:p", "delta:r", "single:world"))))
            assert not bad, f"T-5 FAILED at pp1 L{ell}: " + "; ".join(bad[:8])
        log(f"      L{ell} done in {cell['sec']:.1f}s  auditions so far {n_aud_total}  "
            f"leak {lk}")
        sg = cell["single"]
        if sg:
            cls = np.array([1.0 if q["cls"] == "true" else 0.0 for q in sg])
            log("      single-form sep: world=" +
                f"{PP.vo_auc(np.array([q['w_succ'] for q in sg]), cls):.3f}  " +
                "  ".join(f"{k}={PP.vo_auc(np.array([q['p_' + k] for q in sg]), cls):.3f}"
                          for k in keys_l))
        del x0, roots_t

    # ---- pp3 step 2: the learner's own operative rows, priced one at a time -------------- #
    own_levels = tuple(q for q in OW.LEVELS if q in want_levels)
    if own and own_levels:
        ep = os.path.join(L["root_sb"], "entry.json.gz")
        ent_last = json.load(gzip.open(ep, "rt"))[-1] if os.path.exists(ep) else None
        tables, diags, rgates = LT.reconstruct(
            res_j, ent_last if ent_last is not None else {"operative_mask": {}},
            truth=truth, levels=tuple(q for q in LT.LEVELS_OK if q in own_levels),
            log=(lambda *a: None), arm_key=arm_key)
        bad_r = [g["gate"] for g in rgates if not g["pass"]]
        assert not bad_r, f"the reconstruction did not gate: {bad_r}"
        out["own_rgates"] = [{k_: g[k_] for k_ in ("gate", "pass", "what")} for g in rgates]
        NS3 = 48 if smoke else OW.N_SCORE
        for ell in own_levels:
            if ell not in eras or ell not in tables:
                continue
            era, node = eras[ell], int(eras[ell]["node"])
            span = s ** (ell - 1)
            blk0 = node * span
            child = np.asarray(tables[ell]["child"], np.int64)
            lower = tables[ell - 1]
            n_rows = int(child.shape[0])
            tvec = np.asarray(diags[ell]["truth_row"], np.int64)
            t_l0 = time.time()
            sh, sh_hold, sh_info = slot_heads(blk0, span, (lambda *a: None))
            heads_l = dict(heads)
            heads_l.update(sh)
            keys_l = ["banked"] + sorted(heads_l)
            pool_seed = int(cfgr["seed"]) + OW.OWN_POOL_BASE + 1000 * ell
            roots_np, x_np = context_instances(
                rules, {"name": era["name"], "level": ell, "nodes": [node]},
                NS3, s, depth, v, m, seed=pool_seed)
            x0 = torch.from_numpy(x_np).to(device)
            roots_t = torch.from_numpy(roots_np).to(device).long()
            lk = {"n": 0, "fit": 0, "bank": 0}
            leak(x_np, roots_np, lk)
            rows = []
            for i in range(n_rows):
                tb = MC.make_table(ell, np.ascontiguousarray(child[i].reshape(-1, s)), lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, x0, mv, canon, s)
                xf_np = xf.cpu().numpy()
                succ, dres = grade(xf_np, roots_np, rules, s)
                leak(xf_np, roots_np, lk)
                rd = RD.read_heads(L["cores"], heads_l, L["banked"], xf, roots_t, blk0, span,
                                   s, nb, dim, v, device)
                n_aud_total += 1
                r = {"i": i, "child": [int(q) for q in child[i]], "truth": int(tvec[i]),
                     "w_succ": float(succ.mean()), "e": float(1.0 - succ.mean()),
                     "dres": float(dres.mean())}
                for nm in keys_l:
                    r[f"p_{nm}"] = float(rd[nm].mean())
                for k_ in ("top", "lse", "marg"):
                    r[f"dp_{k_}"] = float(dp[k_].mean())
                rows.append(r)
            ocell = {"era": era["name"], "node": node, "blk0": blk0, "span": span,
                     "n_rows": n_rows, "n_true": int(tvec.sum()), "keys": keys_l,
                     "pool_seed": pool_seed, "rows": rows, "leak": lk,
                     "slot": {"info": sh_info, "hold": sh_hold},
                     "sec": float(time.time() - t_l0)}
            out["own"][str(ell)] = ocell
            if do_gate and ref3 is not None:
                rc = ref3["cells"].get(str(ell))
                assert rc is not None, "T-5: banked pp3 has no such level"
                bad, worst = cmp_pp3_rows(rows, rc["rows"])
                gate_rec["pp3"][str(ell)] = {"pass": not bad, "worst": worst, "bad": bad[:8],
                                             "n_rows": n_rows}
                log(f"      [T-5 pp3 L{ell}] {'PASS' if not bad else 'FAIL'}  {n_rows} rows; "
                    "worst |d|: " + "  ".join(f"{k} {q:.1e}" for k, q in sorted(worst.items())))
                assert not bad, f"T-5 FAILED at pp3 L{ell}: " + "; ".join(bad[:8])
            y3 = tvec.astype(np.float64)
            log(f"      own L{ell}: {n_rows} rows ({int(tvec.sum())} true) in "
                f"{ocell['sec']:.1f}s  leak {lk}  sep: world="
                f"{PP.vo_auc(np.array([q['w_succ'] for q in rows]), y3):.3f}  " +
                "  ".join(f"{k}={PP.vo_auc(np.array([q['p_' + k] for q in rows]), y3):.3f}"
                          for k in keys_l))
            del x0, roots_t

    out["gate_t5"] = gate_rec
    out["gate_f1"] = gate_f1
    out["n_auditions"] = n_aud_total
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s  auditions {n_aud_total}  peak RSS "
        f"{out['peak_rss_mb']:.0f} MB  peak GPU {out['peak_gpu_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag, "rr1")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"job": "rr1", "arm": arm_key, "sec": out["sec"], "n_aud": n_aud_total,
            "rss_mb": out["peak_rss_mb"],
            "gate": {k: all(q["pass"] for q in v_.values()) for k, v_ in gate_rec.items()}}


# --------------------------------------------------------------------------------------- #
# rr5: pp5's walk, every reader's level as a gate
# --------------------------------------------------------------------------------------- #

def decide_g(gate, cur, tri, n_kept, tol, deltas):
    """`readgate.decide` for pp5's own gate names (delegated, so those rows are pp5's code), and
    the same three rules -- pooled, paired over the changed set, pooled with a margin -- for any
    reader key: `<key>`, `<key>~pair`, `<key>~m`."""
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import readgate as RG
    if gate in ("world", "ungated") or gate in RG.GATE_READ:
        return RG.decide(gate, cur, tri, n_kept, tol, deltas["banked"])
    key, _, mode = gate.partition("~")
    if mode == "pair":
        ch = (np.ones(tri["p"][key].shape[0], bool) if n_kept == 0
              else (tri["entry"] == int(n_kept)))
        if not ch.any():
            return True, 0.0
        d = float((tri["p"][key][ch] - cur["p"][key][ch]).mean())
        return bool(d >= -0.0), d
    assert mode in ("", "m"), gate
    d = float(tri["p"][key].mean() - cur["p"][key].mean())
    return bool(d >= -(deltas[key] if mode == "m" else 0.0)), d


def gated_walk_g(stats_fn, base_child, cand_child, order_idx, gate, tol, deltas,
                 budget=None, on_change=None):
    """`readgate.gated_walk` (single-row candidates) with `decide_g` in place of `decide` and a
    margin per reader. Gate T-4 asserts it is `gated_walk` on pp5's own names."""
    cur = stats_fn(base_child)
    n_aud = 1
    kept = [list(map(int, r_)) for r_ in base_child]
    rec = {"gate": gate, "e_base_gate": float(cur["e"]), "admitted": [], "rejected": 0,
           "conf": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "cost_rows": [], "steps": []}
    if on_change is not None:
        on_change(n_aud, kept)
    walk = order_idx if budget is None else order_idx[:int(budget)]
    for ci in walk:
        add = [list(map(int, cand_child[ci]))]
        tri = stats_fn(kept + add)
        n_aud += 1
        a_g, why = decide_g(gate, cur, tri, len(kept), tol, deltas)
        a_w = bool(tri["e"] <= cur["e"] + tol)
        rec["conf"]["TT" if (a_g and a_w) else "TF" if (a_g and not a_w)
                    else "FT" if (not a_g and a_w) else "FF"] += 1
        if a_g and not a_w:
            rec["cost_rows"].append(float(tri["e"] - cur["e"]))
        if a_g:
            kept = kept + add
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


def gates_for(forms):
    """pp5's seven gates under their own names, then every other reader's: through the shaped
    trunk pooled / paired / margin, through frozen and twin pooled. `ridge@frozen` and
    `ridge@twin` ARE pp5's `frozen` / `twin` and are not walked twice."""
    g = ["world", "read", "read_pair", "read_m", "frozen", "twin", "ungated"]
    for f in forms:
        g += [f"{f}@shaped", f"{f}@shaped~pair", f"{f}@shaped~m"]
        if f != "ridge":
            g += [f"{f}@frozen", f"{f}@twin"]
    return g


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=8192)
def rr5_arm(arm_key, out_tag, smoke=False, levels="", settings="ab", forms="", gate=True):
    import resource
    import torch
    import rhm.practice.ratchet.macros as MC
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import incremental as IN
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import learner_tables as LT
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import readgate as RG
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want_levels = tuple(int(z) for z in levels.split(",") if z)
    forms_ = tuple(f for f in forms.split(",") if f) or FORMS_5
    assert "ridge" in forms_
    NP_, NG, NT, KP, NR = IN.N_PRICE, IN.N_AUD, IN.N_TEST, IN.K_POOL_A, IN.N_REPEATS
    NTP, kpb = IN.N_TEST_POOLS, IN.K_POOL_B
    TOL = IN.EXTEND_TOL
    if smoke:
        NP_, NG, NT, KP, NR, NTP, kpb = 128, 64, 64, 6, 1, 2, 12
    do_gate = bool(gate) and not smoke
    GATES_ALL = gates_for(forms_)
    log("=" * 100)
    log(f"[timbre rr5] arm {arm_key}  settings {settings}  device {device}  "
        f"smoke={bool(smoke)}  gate={do_gate}")
    log(f"  forms {forms_}  {len(GATES_ALL)} gates  order {RG.ORDERS[0]}  tol {TOL}")
    log(f"  n_price {NP_}  n_gate {NG}  n_test {NT}x{NTP}  k_pool_a {KP}  repeats {NR}")
    log("=" * 100)
    L = _load(arm_key, device, log)
    v, s, depth, m, maxl, nb, dim = (L["v"], L["s"], L["depth"], L["m"], L["maxl"], L["nb"],
                                     L["dim"])
    rules, canon, cfgr, res_j = L["rules"], L["canon"], L["cfgr"], L["res_j"]
    shaped = L["cores"]["shaped"]
    heads, head_recs, hold_read = _fit_all(L, forms_, device, log)
    RKEYS = ["banked"] + sorted(heads)
    ref5 = None
    if do_gate:
        ref5 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp5", f"{arm_key}.json")))

    bank = L["bank"]
    fit_rows = np.concatenate([L["idx_tr"], L["idx_va"]])
    bank_all_keys = set(RD.xr_keys(bank["x"].numpy(), bank["r"].numpy()).tolist())
    bank_fit_keys = set(RD.xr_keys(bank["x"].numpy()[fit_rows],
                                   bank["r"].numpy()[fit_rows]).tolist())

    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {int(e_["level"]): int(e_["n_entries"]) for e_ in res_j["events"]
               if e_.get("kind") == "commit"}
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)
    lt_tables, lt_diags, lt_gates = None, None, []
    if "b" in settings:
        import gzip
        ep = os.path.join(L["root_sb"], "entry.json.gz")
        ent_last = (json.load(gzip.open(ep, "rt"))[-1] if os.path.exists(ep)
                    else {"operative_mask": {}})
        lvb = tuple(q for q in LT.LEVELS_OK if (not want_levels or q in want_levels))
        lt_tables, lt_diags, lt_gates = LT.reconstruct(
            res_j, ent_last, truth=truth, levels=lvb, log=(lambda *a: None), arm_key=arm_key)
        bad = [g["gate"] for g in lt_gates if not g["pass"]]
        assert not bad, f"pp3's reconstruction did not gate here: {bad}"

    out = {"arm": arm_key, "spec": L["spec"], "gate": L["gate"], "settings": settings,
           "hold_read": hold_read, "heads": head_recs, "rkeys": RKEYS,
           "cfg": {"n_price": NP_, "n_gate": NG, "n_test": NT, "n_test_pools": NTP,
                   "k_pool_a": KP, "k_pool_b": kpb, "repeats": NR, "tol": TOL,
                   "budgets": list(RG.BUDGETS), "gates": GATES_ALL,
                   "orders": list(RG.ORDERS), "orders_2": list(RG.ORDERS_2),
                   "extend_cap": IN.EXTEND_CAP, "n_aud": IN.N_AUD, "forms": list(forms_),
                   "seed": int(cfgr["seed"]), "rule_seed": int(cfgr["rule_seed"]),
                   "smoke": bool(smoke), "gate": do_gate},
           "lt_gates": [{k_: g[k_] for k_ in ("gate", "pass", "what")} for g in lt_gates],
           "cells": {}}
    gate_rec = {}
    n_aud_total = 0
    p4 = {}
    keyset = (lambda roots_, x_: set(RD.xr_keys(x_, roots_).tolist()))

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
            lk = {"n": 0, "fit": 0, "bank": 0,
                  "pools_fit": int(sum(len(q & bank_fit_keys) for q in [kp_] + kt_ + kg_))}

            px = torch.from_numpy(pr_x).to(device)
            proots_t = torch.from_numpy(pr_roots).to(device).long()
            price = {k_: np.zeros(ncand) for k_ in ["world", "dp_top"] + RKEYS}
            for ci, cd in enumerate(cands):
                tb = MC.make_table(ell, np.asarray([cd["child"]], np.int64), lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, px, mv, canon, s)
                su, _ = grade(xf.cpu().numpy(), pr_roots, rules, s)
                rd = RD.read_heads(L["cores"], heads, L["banked"], xf, proots_t, blk0, span,
                                   s, nb, dim, v, device)
                n_aud_total += 1
                price["world"][ci] = float(su.mean())
                price["dp_top"][ci] = float(dp["top"].mean())
                for nm in RKEYS:
                    price[nm][ci] = float(rd[nm].mean())
            # pp5's own order names: `banked`, and the frozen / twin ridge as `frozen` / `twin`
            price["frozen"] = price["ridge@frozen"]
            price["twin"] = price["ridge@twin"]

            scache, tcache = {}, {}

            def _fire(child_rows, pool, want_reads):
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                if arr.shape[0] == 0:
                    su, _ = grade(pool["x_np"], pool["roots"], rules, s)
                    d = {"e": float(1.0 - su.mean()),
                         "entry": np.full(pool["x"].shape[0], -1, np.int32)}
                    if want_reads:
                        d["p"] = RD.read_heads(L["cores"], heads, L["banked"], pool["x"],
                                               pool["roots_t"], blk0, span, s, nb, dim, v,
                                               device)
                    return d
                tb = MC.make_table(ell, arr, lower, s)
                mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
                xf, dp = PP.fire_rec(shaped, pool["x"], mv, canon, s)
                xf_np = xf.cpu().numpy()
                su, _ = grade(xf_np, pool["roots"], rules, s)
                d = {"e": float(1.0 - su.mean()), "entry": dp["entry"]}
                if want_reads:
                    ks_ = RD.xr_keys(xf_np, pool["roots"]).tolist()
                    lk["n"] += len(ks_)
                    lk["fit"] += sum(1 for k_ in ks_ if k_ in bank_fit_keys)
                    lk["bank"] += sum(1 for k_ in ks_ if k_ in bank_all_keys)
                    d["p"] = RD.read_heads(L["cores"], heads, L["banked"], xf,
                                           pool["roots_t"], blk0, span, s, nb, dim, v, device)
                return d

            def stats(child_rows, rp):
                nonlocal n_aud_total
                arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
                key = (rp, int(arr.shape[0]),
                       hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
                if key in scache:
                    return scache[key]
                d = _fire(arr, gpools[rp], True)
                # pp5's own reader names on the same arrays, so `readgate.decide` reads them
                d["p"]["frozen"] = d["p"]["ridge@frozen"]
                d["p"]["twin"] = d["p"]["ridge@twin"]
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
                                  for k_ in ["world", "dp_top", "frozen", "twin"] + RKEYS},
                    "order_rho": {k_: PP.spearman(price[k_], price["world"])
                                  for k_ in ["dp_top"] + RKEYS},
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
                b0 = stats(base_child, rp)
                deltas = {k_: float(b0["p"][k_].std() / np.sqrt(max(1, NG)))
                          for k_ in b0["p"]}
                rep = {"repeat": rp, "gate_seed": gpools[rp]["seed"],
                       "n_base": len(base_child), "n_live": len(live),
                       "delta": deltas["banked"], "deltas": deltas,
                       "e_base_test": test_e(base_child),
                       "p_base": float(b0["p"]["banked"].mean()),
                       "p_bases": {k_: float(q.mean()) for k_, q in b0["p"].items()},
                       "orders": {}}
                todo = [(RG.ORDERS[0], g_) for g_ in GATES_ALL]
                if rp == 0:
                    todo += [(o_, g_) for o_ in RG.ORDERS_2 for g_ in RG.GATES_2]
                for onm, gnm in todo:
                    oi = lv[IN.order_by(price[onm][lv], tie[lv])]
                    curve = []

                    def _rec(na, kept, _c=curve):
                        _c.append({"n_aud": int(na), "n_kept": int(len(kept)),
                                   "n_wrong": n_wrong(kept)})
                    r = gated_walk_g(lambda cr, _rp=rp: stats(cr, _rp), base_child, cand_child,
                                     oi, gnm, TOL, deltas, on_change=_rec)
                    bud = {}
                    _pos = {int(c_): i_ for i_, c_ in enumerate(oi)}
                    for b in list(RG.BUDGETS) + [len(oi)]:
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
                    r["curve"] = curve
                    rep["orders"][f"{onm}|{gnm}"] = r
                cell["repeats"].append(rep)

                # ---- GATE T-6: pp5's banked gates, exactly ---------------------------- #
                if do_gate:
                    ck = f"{setting}L{ell}"
                    rr = ref5["cells"][ck]["repeats"][rp]
                    bad, worst = cmp_pp5_rep(rep, rr)
                    gate_rec[f"{ck}|{rp}"] = {"pass": not bad, "bad": bad[:8],
                                              "n_orders": len(rr["orders"]),
                                              "worst": max(worst.values()) if worst else 0.0}
                    log(f"      [T-6 {ck} rp{rp}] {'PASS' if not bad else 'FAIL'}  "
                        f"{len(rr['orders'])} banked order|gate walks identical; worst |d| "
                        f"{gate_rec[f'{ck}|{rp}']['worst']:.1e}")
                    assert not bad, f"T-6 FAILED at {ck} rp{rp}: " + "; ".join(bad[:8])
                log(f"      rp{rp} b=8 / full: " + "  ".join(
                    f"{g_}:{rep['orders'][f'{RG.ORDERS[0]}|{g_}']['budget']['8']['e_test']:.3f}"
                    f"/{rep['orders'][f'{RG.ORDERS[0]}|{g_}']['e_test_final']['mean']:.3f}"
                    for g_ in GATES_ALL if "~pair" not in g_))
            cell["leak"] = lk
            cell["sec"] = float(time.time() - t_l0)
            out["cells"][f"{setting}L{ell}"] = cell
            log(f"      cell done in {cell['sec']:.1f}s  auditions so far {n_aud_total}  "
                f"leak {lk}")

    out["P-4"] = p4
    out["gate_t6"] = gate_rec
    out["n_auditions"] = n_aud_total
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s  auditions {n_aud_total}  "
        f"peak RSS {out['peak_rss_mb']:.0f} MB  peak GPU {out['peak_gpu_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag, "rr5")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}_{settings}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}_{settings}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"job": "rr5", "arm": arm_key, "settings": settings, "sec": out["sec"],
            "n_aud": n_aud_total, "rss_mb": out["peak_rss_mb"],
            "gate": all(q["pass"] for q in gate_rec.values()) if gate_rec else None}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=14400, memory=2048)
def sweep(out_tag="tb1", arms1=ARMS_1, arms5=ARMS_5, smoke=False, levels1="", levels5="",
          settings="ab", forms1="", forms5="", own=True, split_settings=False):
    """CPU coordinator. rr1 per arm and rr5 per arm (both settings in one container unless
    `split_settings`, which buys wall-clock at the price of a second ~2-minute reader fit) in
    separate containers: every job is an independent arm of the banked design, so the fan-out
    adds no GPU-hours beyond each container's own reader fits."""
    h = []
    for a in [x for x in arms1.split(",") if x]:
        h.append(("rr1", a, rr1_arm.spawn(a, out_tag, bool(smoke), levels1, forms1, bool(own))))
    sets = list(settings) if split_settings else [settings]
    for a in [x for x in arms5.split(",") if x]:
        for st in sets:
            h.append(("rr5", f"{a}_{st}", rr5_arm.spawn(a, out_tag, bool(smoke), levels5, st,
                                                        forms5)))
    outs = []
    for job, a, hd in h:
        try:
            outs.append(hd.get())
        except Exception as e:                                     # noqa: BLE001
            outs.append({"job": job, "arm": a, "error": repr(e)[:2000]})
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"outs": outs, "smoke": bool(smoke), "levels1": levels1,
                             "levels5": levels5, "settings": settings}, default=str) + "\n")
    volume.commit()
    return outs


# --------------------------------------------------------------------------------------- #
# the gates, and the harness that shows each one failing
# --------------------------------------------------------------------------------------- #

def _gate_fixture(arm_key, device):
    """One arm's banked shaped core and the frozen / twin trunks, 2048 held-out bank rows."""
    import torch
    L = _load(arm_key, device, lambda *a: None)
    bank = L["bank"]
    hi = torch.from_numpy(L["hold_idx"])
    rows = {k: bank[k][hi] for k in ("x", "r", "y", "blk", "spn")}
    return L, rows


def _walk_fixture(device):
    """`readgate.gates5`'s synthetic L3 pool, a random-init core, 26 candidates; the level is a
    deterministic function of the FIRED configuration, so it moves with the table."""
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import incremental as IN
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
    base = [[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:4])]]
    wr = PP.wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 12, rg)
    pool = ([[int(q) for q in r_] for r_ in tl["child"][np.sort(rg.permutation(64)[:14])]]
            + [list(q) for q in wr])
    cache = {}

    def stats(child_rows, sign=-1.0):
        arr = np.asarray(child_rows, np.int64).reshape(-1, s)
        key = (arr.tobytes(), sign)
        if key in cache:
            return cache[key]
        tb = MC.make_table(ell, arr, tl["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        xf, dp = PP.fire_rec(core, gxt, mv, canon, s)
        xn = xf.cpu().numpy()
        su, _ = grade(xn, gr, rules, s)
        e = float(1.0 - su.mean())
        lvl = ((xn * np.arange(1, xn.shape[1] + 1)).sum(1) % 97).astype(np.float32) / 97.0
        p = {"banked": lvl, "frozen": np.sqrt(lvl), "twin": 1.0 - lvl,
             "X@shaped": lvl, "Y@frozen": np.sqrt(lvl), "Z@twin": 1.0 - lvl,
             "W@shaped": np.full(len(lvl), sign * e, np.float32)}
        d = {"e": e, "entry": dp["entry"], "p": p}
        cache[key] = d
        return d
    return stats, base, pool


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192)
def gates_t(arm_key="s0_sv"):
    """T-1 features, T-2 heads, T-3 the MLP fitter, T-4 the walk, T-7 the leakage counter, and
    T-5 / T-6's comparators passing on the banked JSON read as its own re-run (the paid runs
    close T-5 / T-6 for real, in-container)."""
    import torch
    import torch.nn.functional as TF_
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import readgate as RG
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD
    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    t0 = time.time()
    L, rows = _gate_fixture(arm_key, device)
    s, nb, dim, v = L["s"], L["nb"], L["dim"], L["v"]

    # ---- T-1: the hidden block IS pj_features, bit for bit; the belief block IS the plant's
    #           own `block_logits` on the same masked input, log-softmaxed and pooled --------- #
    t1 = {}
    for tk in ("shaped", "frozen", "twin"):
        core = L["cores"][tk]
        Fh, Fb = RD.feats(core, rows["x"], rows["blk"], rows["spn"], s, nb, dim, device)
        Fref = PP.pj_features(core, rows["x"], rows["blk"], rows["spn"], s, nb, dim, device)
        # the belief, independently: the generator's own `block_logits` (not `SN.trunk`)
        n = int(rows["x"].shape[0])
        unk = rows["blk"] < 0
        b0 = torch.where(unk, torch.zeros_like(rows["blk"]), rows["blk"])
        sw = torch.where(unk, torch.full_like(rows["spn"], nb - 1),
                         rows["spn"].clamp(min=1)).clamp(max=nb - 1)
        ar = torch.arange(nb)
        inside = (ar[None, :] >= b0[:, None]) & (ar[None, :] < (b0 + sw)[:, None])
        mb = (~inside).long().argmax(1)
        pos = (mb[:, None] * s + torch.arange(s)[None, :])
        xm = rows["x"].clone().scatter(1, pos, torch.full_like(pos, -1))
        with torch.no_grad():
            lg = core.block_logits(xm.to(device)).float()
        lsm = torch.log_softmax(lg, -1).cpu()
        wg = inside.float() / inside.float().sum(1, keepdim=True).clamp(min=1.0)
        Bref = torch.cat([lsm.mean(1), (lsm * wg[:, :, None]).sum(1)], 1)
        t1[tk] = {"max_dFh": float((Fh - Fref).abs().max()),
                  "max_dFb": float((Fb - Bref).abs().max()), "n": n,
                  "nfeat_h": int(Fh.shape[1]), "nfeat_b": int(Fb.shape[1])}
        assert t1[tk]["max_dFh"] == 0.0, f"T-1 FAILED ({tk}): hidden block != pj_features"
        assert t1[tk]["max_dFb"] < 1e-5, f"T-1 FAILED ({tk}): belief block != block_logits"
        assert int(Fb.shape[1]) == 2 * v
    res["T-1:features"] = t1
    print(f"[+{time.time() - t0:5.1f}s] T-1 {t1}", flush=True)

    # ---- T-2: the banked head through `read_heads` IS pj_predict, bit for bit; `fit_lin`
    #           on the hidden block IS pj_fit on pp1's slice --------------------------------- #
    xfake = rows["x"][:256]
    rfake = rows["r"][:256]
    blk0, span = 24, 4                       # L3n6's slot, a real era cell
    out = RD.read_heads({"shaped": L["cores"]["shaped"]}, {}, L["banked"], xfake, rfake, blk0,
                        span, s, nb, dim, v, device)
    pref = PP.pj_predict(L["cores"]["shaped"], xfake, rfake, blk0, span, *L["banked"], s, nb,
                         dim, v, device).numpy()
    import torch as T_
    bank = L["bank"]
    idx = np.concatenate([L["idx_tr"], L["idx_va"]])
    ii = T_.from_numpy(idx)
    fref = PP.pj_fit(L["cores"]["shaped"], bank, L["idx_tr"], L["idx_va"], s, nb, dim, v,
                     device)
    Fh, _ = RD.feats(L["cores"]["shaped"], bank["x"][ii], bank["blk"][ii], bank["spn"][ii],
                     s, nb, dim, device)
    fmine = RD.fit_lin(Fh, bank["r"][ii], bank["y"][ii], int(L["idx_tr"].size), v, device)
    t2 = {"max_dp_banked": float(np.abs(out["banked"] - pref).max()),
          "lam": [fmine["lam"], fref["lam"]], "val_auc": [fmine["val_auc"], fref["val_auc"]],
          "max_dw": float((fmine["w"] - fref["w"]).abs().max())}
    res["T-2:heads"] = t2
    assert t2["max_dp_banked"] == 0.0, "T-2 FAILED: the banked head != pj_predict"
    assert fmine["lam"] == fref["lam"] and abs(fmine["val_auc"] - fref["val_auc"]) < 1e-9
    assert t2["max_dw"] < 1e-6, "T-2 FAILED: fit_lin != pj_fit"
    print(f"[+{time.time() - t0:5.1f}s] T-2 {t2}", flush=True)

    # ---- T-3: the MLP fitter can fit: a target that is a nonlinear function of two columns
    #           of the real features is recovered on held-out rows -------------------------- #
    Fh_ = Fh[:, :4].clone()
    ysyn = ((Fh_[:, 0] - Fh_[:, 0].median()) * (Fh_[:, 1] - Fh_[:, 1].median()) > 0).float()
    ntr = int(L["idx_tr"].size)
    hd = RD.fit_mlp(Fh, bank["r"][ii], ysyn, ntr, v, device, steps=1500)
    lin = RD.fit_lin(Fh, bank["r"][ii], ysyn, ntr, v, device)
    res["T-3:mlp_fits_xor"] = {"mlp_val_auc": hd["val_auc"], "ridge_val_auc": lin["val_auc"],
                              "trace": hd["trace"]}
    assert hd["val_auc"] > 0.9, f"T-3 FAILED: the MLP does not fit an XOR ({hd['val_auc']})"
    print(f"[+{time.time() - t0:5.1f}s] T-3 mlp {hd['val_auc']:.4f} ridge "
          f"{lin['val_auc']:.4f}", flush=True)

    # ---- T-4: the walk IS readgate.gated_walk on pp5's names; a reader key under the generic
    #           rules IS the pp5 rule on the same array; handed -e it IS the world gate ----- #
    stats, base, pool = _walk_fixture(device)
    order = list(range(len(pool)))
    t4 = {}
    deltas = {"banked": 0.004, "X@shaped": 0.004}
    for g_ in ("world", "read", "read_pair", "read_m", "frozen", "twin", "ungated"):
        a = RG.gated_walk(stats, base, pool, order, g_, 0.0, 0.004)
        b = gated_walk_g(stats, base, pool, order, g_, 0.0, deltas)
        t4[g_] = {"n_admitted": len(a["admitted"]), "identical": a["admitted"] == b["admitted"]
                  and a["conf"] == b["conf"]}
        assert t4[g_]["identical"], f"T-4 FAILED on {g_}"
    for g_, ref in (("X@shaped", "read"), ("X@shaped~pair", "read_pair"),
                    ("X@shaped~m", "read_m"), ("Y@frozen", "frozen"), ("Z@twin", "twin")):
        a = RG.gated_walk(stats, base, pool, order, ref, 0.0, 0.004)
        b = gated_walk_g(stats, base, pool, order, g_, 0.0, deltas)
        t4[g_] = {"n_admitted": len(b["admitted"]), "identical": a["admitted"] == b["admitted"]}
        assert t4[g_]["identical"], f"T-4 FAILED on the generic {g_}"
    w = gated_walk_g(stats, base, pool, order, "world", 0.0, deltas)
    for g_ in ("W@shaped", "W@shaped~pair", "W@shaped~m"):
        b = gated_walk_g(stats, base, pool, order, g_, 0.0, {"W@shaped": 0.0})
        t4[g_ + "=world"] = b["admitted"] == w["admitted"]
        assert b["admitted"] == w["admitted"], f"T-4 FAILED: {g_} handed -e is not the world"
    # non-vacuity: the gates under test do not all admit the same set
    adm = {g_: tuple(gated_walk_g(stats, base, pool, order, g_, 0.0, deltas)["admitted"])
           for g_ in ("world", "read", "frozen", "twin", "ungated")}
    t4["distinct_admission_sets"] = len(set(adm.values()))
    assert len(set(adm.values())) >= 3, "T-4 is VACUOUS: the gates admit the same sets"
    res["T-4:walk"] = t4
    print(f"[+{time.time() - t0:5.1f}s] T-4 {t4}", flush=True)

    # ---- T-5 / T-6 comparators: the banked JSON read as its own re-run passes ------------ #
    r1 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp1", f"{arm_key}.json")))
    r3 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp3", f"{arm_key}_own.json")))
    r5 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp5", f"{arm_key}.json")))
    t5 = {}
    for lv, c in r1["cells"].items():
        bad, _ = cmp_pp1_cell(as_mine_pp1(c), c)
        t5[f"pp1:L{lv}"] = not bad
    for lv, c in r3["cells"].items():
        rows3 = [dict(q, **{f"p_{k}": q[f"p_{n}"] for n, k in ARM0.items()})
                 for q in c["rows"]]
        bad, _ = cmp_pp3_rows(rows3, c["rows"])
        t5[f"pp3:L{lv}"] = not bad
    for ck, c in r5["cells"].items():
        for rp in c["repeats"]:
            bad, _ = cmp_pp5_rep(rp, rp)
            t5[f"pp5:{ck}:{rp['repeat']}"] = not bad
    res["T-5/T-6:comparators_identity"] = t5
    assert all(t5.values()), f"T-5/6 comparator fails on identity: {t5}"

    # ---- T-7: the leakage counter counts ------------------------------------------------- #
    fit_rows = np.concatenate([L["idx_tr"], L["idx_va"]])
    kfit = set(RD.xr_keys(bank["x"].numpy()[fit_rows], bank["r"].numpy()[fit_rows]).tolist())
    held = RD.xr_keys(rows["x"].numpy(), rows["r"].numpy()).tolist()
    res["T-7:hold_rows_in_fit_keys"] = int(sum(1 for k in held if k in kfit))
    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192)
def falsify_t(arm_key="s0_sv"):
    """Every gate above fed a deliberately broken input, and shown to trip."""
    import copy as _c
    import torch
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import preplay as PP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay import readgate as RG
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD
    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, n_ok = {}, 0
    L, rows = _gate_fixture(arm_key, device)
    s, nb, dim, v = L["s"], L["nb"], L["dim"], L["v"]
    core = L["cores"]["shaped"]

    # T-1 broken: the features read with the mask OFF
    Fh_bad, Fb_bad = RD.feats(core, rows["x"], rows["blk"], rows["spn"], s, nb, dim, device,
                              mask=False)
    Fref = PP.pj_features(core, rows["x"], rows["blk"], rows["spn"], s, nb, dim, device)
    Fh_ok, Fb_ok = RD.feats(core, rows["x"], rows["blk"], rows["spn"], s, nb, dim, device)
    d1 = float((Fh_bad - Fref).abs().max())
    d1b = float((Fb_bad - Fb_ok).abs().max())
    res["T-1"] = {"broke": "the mask is off, so the hidden and belief blocks read a block the "
                           "readout never sees", "max_dFh": d1, "max_dFb": d1b}
    assert d1 > 0.0 and d1b > 1e-5, "T-1 did not trip on an unmasked read"
    n_ok += 1

    # T-2 broken: the banked head read through the FROZEN trunk in the shaped slot
    xfake, rfake = rows["x"][:256], rows["r"][:256]
    out = RD.read_heads({"shaped": L["cores"]["frozen"]}, {}, L["banked"], xfake, rfake, 24, 4,
                        s, nb, dim, v, device)
    pref = PP.pj_predict(core, xfake, rfake, 24, 4, *L["banked"], s, nb, dim, v,
                         device).numpy()
    d2 = float(np.abs(out["banked"] - pref).max())
    res["T-2"] = {"broke": "the banked head read through the frozen trunk", "max_dp": d2}
    assert d2 > 0.0, "T-2 did not trip on the wrong trunk"
    n_ok += 1

    # T-3 broken: the XOR target with its labels permuted -- nothing to fit
    bank = L["bank"]
    idx = np.concatenate([L["idx_tr"], L["idx_va"]])
    ii = torch.from_numpy(idx)
    Fh, _ = RD.feats(core, bank["x"][ii], bank["blk"][ii], bank["spn"][ii], s, nb, dim, device)
    ysyn = ((Fh[:, 0] - Fh[:, 0].median()) * (Fh[:, 1] - Fh[:, 1].median()) > 0).float()
    yperm = ysyn[torch.from_numpy(np.random.default_rng(3).permutation(len(ysyn)))]
    hd = RD.fit_mlp(Fh, bank["r"][ii], yperm, int(L["idx_tr"].size), v, device, steps=1500)
    res["T-3"] = {"broke": "the target's labels permuted", "val_auc": hd["val_auc"]}
    assert hd["val_auc"] <= 0.9, "T-3 did not trip on permuted labels"
    n_ok += 1

    # T-4 broken: (i) a generic reader handed +e (the sign not flipped) is not the world gate;
    #             (ii) on the designed sequence a walk whose `cur` never moves admits wrongly
    stats, base, pool = _walk_fixture(device)
    order = list(range(len(pool)))
    w = gated_walk_g(stats, base, pool, order, "world", 0.0, {"banked": 0.0})
    b = gated_walk_g(lambda cr: stats(cr, sign=+1.0), base, pool, order, "W@shaped", 0.0,
                     {"W@shaped": 0.0})
    seq = [0.50, 0.40, 0.45, 0.40, 0.50, 0.10, 0.11]
    box = {"i": 0}

    def stub(_rows):
        e = seq[box["i"]]
        box["i"] += 1
        return {"e": e, "entry": np.zeros(4, np.int32),
                "p": {"Q@shaped": np.full(4, -e, np.float32)}}
    good = gated_walk_g(stub, [[0, 0]], [[1, 1]] * 6, list(range(6)), "Q@shaped", 0.0,
                        {"Q@shaped": 0.0})
    box["i"] = 0
    e0 = seq[0]
    pinned = [i for i in range(6) if -seq[i + 1] >= -e0]          # `cur` pinned at the base
    res["T-4"] = {"broke": "(i) the level is +e; (ii) `cur` pinned at the base, on the "
                           "designed sequence",
                  "sign": {"world": w["admitted"], "plus_e": b["admitted"]},
                  "designed": {"right": good["admitted"], "pinned": pinned}}
    assert b["admitted"] != w["admitted"], "T-4 did not trip on a sign flip"
    assert good["admitted"] == [0, 2, 4] and pinned != [0, 2, 4], "T-4 designed sequence"
    n_ok += 1

    # T-5 broken: the pp1 / pp3 comparators on (i) one level nudged by 2e-5, (ii) the OTHER seed
    other = {"s0_sv": "s2_sv", "s2_sv": "s0_sv"}.get(arm_key, "s2_sv")
    r1 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp1", f"{arm_key}.json")))
    r1o = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp1", f"{other}.json")))
    c = as_mine_pp1(r1["cells"]["3"])
    c["single"][5]["p_ridge@frozen"] += 2e-5
    bad_a, _ = cmp_pp1_cell(c, r1["cells"]["3"])
    bad_b, _ = cmp_pp1_cell(as_mine_pp1(r1o["cells"]["3"]), r1["cells"]["3"])
    r3 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp3", f"{arm_key}_own.json")))
    rows3 = [dict(q, **{f"p_{k}": q[f"p_{n}"] for n, k in ARM0.items()})
             for q in r3["cells"]["3"]["rows"]]
    rows3[7]["p_banked"] += 2e-5
    bad_c, _ = cmp_pp3_rows(rows3, r3["cells"]["3"]["rows"])
    res["T-5"] = {"broke": "(i) one frozen level nudged by 2e-5; (ii) the other seed's cell; "
                           "(iii) one pp3 banked level nudged by 2e-5",
                  "n_bad": [len(bad_a), len(bad_b), len(bad_c)]}
    assert bad_a and bad_b and bad_c, "T-5 did not trip"
    n_ok += 1

    # T-6 broken: one admission list reordered, one test error nudged
    r5 = json.load(open(os.path.join(DATA_DIR, REMOTE, "pp5", f"{arm_key}.json")))
    rp = r5["cells"]["aL2"]["repeats"][0]
    mine = _c.deepcopy(rp)
    mine["orders"]["dp_top|twin"]["admitted"] = mine["orders"]["dp_top|twin"]["admitted"][::-1]
    bad_d, _ = cmp_pp5_rep(mine, rp)
    mine = _c.deepcopy(rp)
    mine["orders"]["dp_top|read"]["budget"]["8"]["e_test"] += 1e-9
    bad_e, _ = cmp_pp5_rep(mine, rp)
    res["T-6"] = {"broke": "(i) the twin gate's admissions reversed; (ii) one test error "
                           "nudged by 1e-9", "n_bad": [len(bad_d), len(bad_e)]}
    assert bad_d and bad_e, "T-6 did not trip"
    n_ok += 1

    # T-7 broken: five of the bank's fit rows planted into a pool
    fit_rows = np.concatenate([L["idx_tr"], L["idx_va"]])
    kfit = set(RD.xr_keys(bank["x"].numpy()[fit_rows], bank["r"].numpy()[fit_rows]).tolist())
    xp = np.concatenate([rows["x"].numpy()[:59], bank["x"].numpy()[fit_rows[:5]]])
    rp_ = np.concatenate([rows["r"].numpy()[:59], bank["r"].numpy()[fit_rows[:5]]])
    n_hit = int(sum(1 for k in RD.xr_keys(xp, rp_).tolist() if k in kfit))
    res["T-7"] = {"broke": "five fit rows planted in a pool of 64", "counted": n_hit}
    assert n_hit >= 5, "T-7 did not count planted rows"
    n_ok += 1

    res["n_gates_falsified"] = n_ok
    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192)
def mlp_recipe(arm_key="s0_sv", steps="3000,8000"):
    """The MLP recipe, chosen on the VALIDATION rows only (the hold rows are never looked at
    here): the batched (lr x wd) ladder's per-model best validation AUC and stopping step, at each
    schedule length, for both MLP forms through all three trunks, beside the ridge. Recorded in
    NOTES section 2; the recipe is then fixed in `readers.py` for every arm. (The first probe of
    record ran the earlier serial fitter at three fixed (steps, lr) points; NOTES has its table.)"""
    import torch
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.timbre import readers as RD
    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = _load(arm_key, device, print)
    bank, s, nb, dim, v = L["bank"], L["s"], L["nb"], L["dim"], L["v"]
    idx = np.concatenate([L["idx_tr"], L["idx_va"]])
    ii = torch.from_numpy(idx)
    ntr = int(L["idx_tr"].size)
    out = {}
    for tk in RD.TRUNKS:
        Fh, Fb = RD.feats(L["cores"][tk], bank["x"][ii], bank["blk"][ii], bank["spn"][ii],
                          s, nb, dim, device)
        lin = RD.fit_lin(Fh, bank["r"][ii], bank["y"][ii], ntr, v, device)
        out[f"ridge@{tk}"] = lin["val_auc"]
        for st in steps.split(","):
            for f, F in (("mlp", Fh), ("bonly", Fb)):
                t0 = time.time()
                hd = RD.fit_mlp(F, bank["r"][ii], bank["y"][ii], ntr, v, device, steps=int(st))
                out[f"{f}@{tk}|{st}"] = {"val": hd["val_auc"], "lr": hd["lr"], "wd": hd["wd"],
                                         "step": hd["step"], "trace": hd["trace"],
                                         "sec": time.time() - t0}
                print(f"{f}@{tk} steps {st}: val {hd['val_auc']:.4f} (lr {hd['lr']:g} wd "
                      f"{hd['wd']:g} step {hd['step']})  ridge {lin['val_auc']:.4f}  "
                      f"{time.time() - t0:.1f}s  ladder " +
                      " ".join(f"{q['lr']:g}/{q['wd']:g}:{q['val_auc_fast']:.3f}@{q['step']}"
                               for q in hd["trace"]), flush=True)
    print(json.dumps(out, indent=1, default=str))
    return out
