"""[within] WITHIN A FIXED CONTEXT, OVER ITS CANDIDATE SET, DOES THE READER RANK AS THE WORLD
DOES? — the scoring half, on the banked soundboard dumps.  CPU, no loop, nothing paid.

Every AUC in `aliquot`, `duplex` and `soundboard` is POOLED over rows.  A pooled AUC mixes two
contrasts that a chooser does not: "is this context solvable at all" and "which of this
context's candidates is the one to write".  `duplex` §3 showed the first is most of what shaping
wrote (the PRE-write read rose from 0.64 to 0.72 with the candidate not in the input at all) and
`overtone` showed an additive readout cannot rank candidates within a context.  Neither
isolated the cell the chooser needs.  This node does, on rows that already exist.

WHAT THIS FILE ADDS TO THE DUMP.  The dump already carries, per row, the grader's filed
probability (`|y`, on a probe row the projection's own `p`), the executor's max-sum prior
through the FINAL core divided by span (`|dp`), the hold code and, on probe rows, the world's
verdict on the substitution (`|yw`).  It does not carry the critic's score, and the critic is
the organ the chooser composes with the prior.  So this file rebuilds the plant and the critic
from `vo_heads.pt` and writes, per row:

    |crit   `vo_critic_scores`' quantity for the row's OWN candidate: the critic's logit on
            `SN.trunk(core, obs)` — the chooser's exact information set, final core.
    |pw     HELD-OUT ROWS ONLY: a `duplex`-protocol ridge-logistic readout of the same final
            trunk over `obs (+) candidate` (the masked post-write read, `trunk_features`'
            `postm`), fit by IRLS on the buffer's own TRAINING rows against the WORLD's verdict.
            This is the control the projection column needs: the projection's `p` was computed
            over `fin (+) candidate` and at FILE TIME, so a flat projection column alone cannot
            separate "the readout does not rank within a context" from "the readout is reading
            a different input at a different moment".
    |pf     THE DIET CONTROL, probe buffers, held-out rows only: the SAME estimator, trunk,
            input and moment as `pw`, fit instead on the matching FILED buffer's training rows.
            The run's own projection is fit on the bank, which is `src = 0` — the learner's own
            experienced configurations — and a deterministic argmax chooser writes ONE
            candidate per context there, so that diet carries almost no within-context
            contrast.  `pf` minus `pw` is what the diet alone is worth.

NOTHING IS FORKED.  `trunk_features`, `fit_probe_t`, `auc_t` and `fingerprint` are imported from
`duplex`; `build_critic`, `vo_compose` and `vo_auc` from `voicing`; `trunk`/`slot_count` from
`span_net`; the generator from `rhm_generative_planner`.  The numpy estimators the reducer uses
are imported FROM the reducer and gated against the torch/numpy originals here (gates Z and A).

Run from experiments/ (MODAL_PROFILE=chromatic).  `wi1` is the tag of record; `wi0` is the same
pass without the `|pf` column and is superseded.

    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::falsify
    modal run --detach \
        rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::score \
        --out-tag wi1 --arms all
"""

import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-within", image=image)

REMOTE = "rhm_practice_within"

# (bank, tag, arm).  The six paid soundboard arms; the two frozen-plant overtone dumps beside
# them are the comparator the brief names, read with the same code and the same caveat.
SB = [("rhm_practice_soundboard", "sb_s1", "sb_sv_yk"),
      ("rhm_practice_soundboard", "sb_s1", "sb_yd_yk"),
      ("rhm_practice_soundboard", "sb_s1", "sb_so_yk"),
      ("rhm_practice_soundboard", "sb_s2", "sb_sv_yk"),
      ("rhm_practice_soundboard", "sb_s2", "sb_yd_yk"),
      ("rhm_practice_soundboard", "sb_s2", "sb_so_yk")]
OV = [("rhm_practice_voicing", "ov_s0b", "ovt_comp_pr_sh"),
      ("rhm_practice_voicing", "ov_s2", "ovt_comp_pr_dis")]

BATCH = 4096
GATE_BATCH = 512          # the second batch size gate B compares against

# THE NEVER-TRAINED TWIN.  `build_rand_trunk`'s IN-LOOP seed, which is `vo_pj_rand_seed` in
# every one of these arms' configs, minted the way `preplay` mints it: AFTER the trained core is
# loaded, so the draw is the same object `aliquot`'s random-twin arm read.  Stated rather than
# inherited; `duplex`/`overtone`'s offline twin is a different seed (20260915) and a different
# object.
TWIN_SEED = 20260918

# The global readout's fit.  `vo_pj_boot = 0.8` is the run's own train-internal split and
# `PJ_RIDGES` its own ridge ladder (imported from `preplay`); `GLOB_N` is the fit+validation
# sample, set ABOVE the bank's own 30000 rows' per-refit cap (`vo_pj_fit_cap = 8192`) so the
# global fits are not starved relative to the per-slot ones.
GLOB_N = 24576
GLOB_BOOT = 0.8
GLOB_SEED = 20260921
LVL_IX = {2: 0, 3: 1, 4: 2, 5: 3}


# --------------------------------------------------------------------------------------- #
# the gates that need torch
# --------------------------------------------------------------------------------------- #

def gate_compose(log):
    """GATE Z — the reducer's numpy standardisation IS `voicing::vo_compose`.

    The reducer has to z-score inside a group and there is no torch where it runs, so the
    formula is written twice.  This asserts the two agree to 1e-12 on random candidate sets,
    including the size-1 guard and a degenerate set whose std is zero (where the donor's
    `clamp_min(1e-6)` is the whole behaviour).
    """
    import torch
    from rhm.practice.voicing.voicing import vo_compose
    from rhm.practice.voicing.sotto_voce.aliquot.soundboard.within.reduce_within import (
        np_compose)
    rng = np.random.default_rng(20260921)
    worst = 0.0
    for n in (1, 2, 3, 8, 31):
        for trial in range(20):
            dp = rng.normal(size=n) * (0.0 if trial == 3 else 1.0) + (7.0 if trial == 3 else 0)
            cr = rng.normal(size=n) * 3.0 - 2.0
            span = int(rng.integers(1, 17))
            w = float(rng.uniform(0.0, 2.0))
            # `vo_compose` divides by the span; the dump's |dp is ALREADY divided, so the
            # torch side is fed dp * span to make the two sides the same object.
            a = vo_compose(torch.as_tensor(dp * span)[None, :],
                           torch.as_tensor(cr)[None, :], span, w).numpy()[0]
            b = np_compose(dp, cr, w)
            # A set of size 1 hits the donor's guard, which returns the RAW dp_sc (undivided);
            # the reducer returns the already-divided column.  Both are the prior's own order
            # on one element and there is nothing to rank, so the gate compares them on the
            # same scale rather than pretending the guard is not there.
            if n < 2:
                a = a / span
            worst = max(worst, float(np.max(np.abs(a - b))))
    log(f"  [gate Z] np_compose vs vo_compose: max |delta| {worst:.3e}")
    return {"name": "Z", "max_delta": worst, "pass": bool(worst < 1e-12)}


def gate_auc(log):
    """GATE A — the reducer's numpy AUC IS `voicing::vo_auc` (and `duplex::auc_t`), ties and
    all.  Run on a tie-heavy draw, because the degenerate filed side is nothing but ties."""
    import torch
    from rhm.practice.voicing.voicing import vo_auc
    from rhm.practice.voicing.sotto_voce.aliquot.duplex.duplex import auc_t
    from rhm.practice.voicing.sotto_voce.aliquot.soundboard.within.reduce_within import np_auc
    rng = np.random.default_rng(777)
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(8, 400))
        x = np.round(rng.normal(size=n) * rng.choice([0.2, 1.0, 5.0]), 1)
        y = (rng.uniform(size=n) < 0.35).astype(np.float64)
        if y.max() == y.min():
            continue
        a, b, c = vo_auc(x, y), np_auc(x, y), auc_t(torch.as_tensor(x), torch.as_tensor(y))
        worst = max(worst, abs(a - b), abs(a - c))
    log(f"  [gate A] np_auc vs vo_auc/auc_t: max |delta| {worst:.3e}")
    return {"name": "A", "max_delta": worst, "pass": bool(worst < 1e-12)}


# --------------------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, timeout=7200, memory=8192, cpu=8.0)
def score_arm(bank, tag, arm, out_tag="wi2", smoke=False):
    """Rebuild the plant and the critic from the arm's own `vo_heads.pt`, then write the two
    score columns the dump does not carry.  CPU: the plant is 2 layers at width 96 over length
    64 and the whole arm is ~430k rows, which is minutes."""
    import resource
    import torch
    import rhm.practice.native.span.span_net as SN
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.voicing.voicing import build_critic
    from rhm.practice.voicing.sotto_voce.aliquot.duplex.duplex import (
        trunk_features, fit_probe_t, fingerprint, render_write)
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.preplay import (
        pj_fit, pj_predict, pj_features, pj_design, pj_irls, pj_pen, pj_score,
        vo_auc as pj_auc, PJ_RIDGES)

    t00 = time.time()
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    volume.reload()
    root = os.path.join(DATA_DIR, bank, tag, arm)
    device = torch.device("cpu")
    log("=" * 100)
    log(f"[within] {tag}/{arm}   bank {bank}   device {device}   smoke={bool(smoke)}")
    log("=" * 100)

    gates = [gate_compose(log), gate_auc(log)]

    blob = torch.load(os.path.join(root, "vo_heads.pt"), map_location="cpu", weights_only=True)
    cfgh = blob["cfg"]
    res_j = json.load(open(os.path.join(root, "results.json")))
    cfgr = res_j["config"]
    v, s, depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
    dim, maxl = int(cfgh["state_dim"]), int(cfgh["max_macro_level"])
    length = s ** depth
    n_blocks = length // s
    hold = float(cfgh.get("vo_critic_hold", cfgr["vo_critic_hold"]))
    thr = int(round(hold * 100))
    w_compose = float(cfgh.get("vo_w", cfgr.get("vo_w", 1.0)))
    log(f"  cfg v{v} s{s} L{depth} dim {dim} maxl {maxl}  hold thr {thr}  "
        f"vo_w {w_compose}  span_hidden_mult {cfgh['span_hidden_mult']}")

    rules = generate_rules_distinct(v, s, depth, int(cfgr["m"]), seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)

    core = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.load_state_dict(blob["core"])
    core.eval()
    fp = fingerprint(core)
    # GATE C — the loaded plant IS the plant the run ended with.  `log["sb"][-1]["sig"]` is
    # `soundboard`'s per-cycle fingerprint (DESIGN §4); the overtone dumps predate it and the
    # gate is reported absent there rather than silently passed.
    sig = None
    try:
        sig = float(res_j["log"]["sb"][-1]["sig"])
    except Exception:
        sig = None
    gc = {"name": "C", "fingerprint": fp, "logged_sig": sig,
          "delta": (None if sig is None else abs(fp - sig)),
          "pass": (None if sig is None else bool(abs(fp - sig) < 1e-6))}
    gates.append(gc)
    log(f"  [gate C] plant fingerprint {fp:.6f}  logged {sig}  "
        f"delta {'n/a' if sig is None else f'{abs(fp - sig):.3e}'}")

    # THE NEVER-TRAINED TWIN, minted AFTER the trained core is loaded (preplay's order; the
    # order matters for the draw).  `obs (+) candidate` puts the candidate's own level-1
    # features in the input, so a ridge readout over ANY feature map may rank candidates by
    # identity alone; this trunk is the floor that says whether the plant's training matters at
    # this cell at all.
    torch.manual_seed(TWIN_SEED)
    twin = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    twin.eval()
    fp_twin = fingerprint(twin)
    log(f"  twin (never trained, manual_seed({TWIN_SEED})) fingerprint {fp_twin:.6f}")

    hm = int(cfgh.get("ov_critic_hidden", -1))
    critic = build_critic(SN.slot_count(s, depth, maxl), v, dim, s ** (maxl - 1), seed=0,
                          device=device,
                          hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
    # strict by default: a shape or key mismatch raises rather than scoring a wrong critic.
    critic.load_state_dict(blob["critic"])
    critic.eval()
    log(f"  critic loaded strict; params {sum(p.numel() for p in critic.parameters())}")

    z = np.load(os.path.join(root, "vo_rows.npz"))
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))
    keys = sorted(meta, key=lambda k: (k.split(":")[0], int(k.split(":")[1]),
                                       int(k.split(":")[2])))
    if smoke:
        keys = [k for k in keys if k.split(":")[1] in ("2",) and k.split(":")[2] in ("0",)]

    cols, per_key, glob = {}, [], {}
    filed_cache = {}          # "{level}:{node}" -> (postm, label, hold) of the FILED buffer
    gate_b = {"name": "B", "max_delta_crit": 0.0, "max_delta_feat": 0.0, "n_checked": 0}
    for k in keys:
        mt = meta[k]
        which = k.split(":", 1)[0]
        blk0, span, sid = int(mt["blk0"]), int(mt["span"]), int(mt["slot_id"])
        obs = torch.from_numpy(z[f"{k}|obs"].astype(np.int64))
        wr = torch.from_numpy(z[f"{k}|write"].astype(np.int64))
        n = int(obs.shape[0])
        lab_np = (z[f"{k}|yw"].astype(np.float64) if (which == "probe" and f"{k}|yw" in z.files)
                  else (z[f"{k}|y"] > 0.5).astype(np.float64))
        feats = trunk_features(core, obs, wr, canon, blk0, span, s, n_blocks, device,
                               batch=BATCH, critic=critic, slot_id=sid)
        crit = feats["mlp"].double().cpu().numpy()
        cols[f"{k}|crit"] = crit.astype(np.float32)

        # GATE B — the batching is not the number.  One buffer per arm is re-scored at a
        # different batch size; `trunk_features` batches the trunk pass, so a difference here
        # would mean the pooled read depends on who it was batched with.
        if gate_b["n_checked"] == 0 and n >= GATE_BATCH * 2:
            f2 = trunk_features(core, obs[:GATE_BATCH * 2], wr[:GATE_BATCH * 2], canon, blk0,
                                span, s, n_blocks, device, batch=GATE_BATCH, critic=critic,
                                slot_id=sid)
            gate_b["max_delta_crit"] = float(
                (f2["mlp"].double() - feats["mlp"][:GATE_BATCH * 2].double()).abs().max())
            gate_b["max_delta_feat"] = float(
                (f2["postm"].double() - feats["postm"][:GATE_BATCH * 2].double()).abs().max())
            gate_b["n_checked"] = 1

        # the duplex-protocol readout, fit HERE on this buffer's own training rows
        code = z[f"{k}|code"].astype(np.int64)
        ho = torch.from_numpy(code < thr)
        tr = ~ho
        y_t = torch.from_numpy(lab_np)
        pw = np.full(n, np.nan, np.float64)
        sc = fit_probe_t(feats["postm"], y_t, tr, ho)
        if sc is not None:
            pw[ho.numpy()] = sc.cpu().numpy()
        cols[f"{k}|pw"] = pw.astype(np.float32)

        # THE DIET CONTROL.  `pw` above is fit on COUNTERFACTUAL rows (probe substitutions,
        # world verdicts).  The run's own projection is fit on the bank, which is `src = 0`
        # only — the learner's own experienced configurations, where the chooser is a
        # deterministic argmax and one context therefore carries one write.  `pf` is the same
        # estimator, the same trunk, the same input and the same moment, fit on the FILED
        # buffer's training rows instead and scored on the PROBE buffer's held-out rows, so the
        # only thing that differs between the `pf` and `pw` columns is the DIET.
        # `fit_probe_t` fits on `tr` and scores `ho` of one matrix, so the two buffers are
        # concatenated and the masks select across them; the helper is used verbatim.
        slot = k.split(":", 1)[1]
        if which == "filed":
            filed_cache[slot] = (feats["postm"], y_t, ho)
        pf = np.full(n, np.nan, np.float64)
        pf_fit = False
        if which == "probe" and slot in filed_cache:
            Ff, yf, hf = filed_cache[slot]
            F_cat = torch.cat([Ff, feats["postm"]])
            y_cat = torch.cat([yf, y_t])
            tr_cat = torch.cat([~hf, torch.zeros(n, dtype=torch.bool)])
            ho_cat = torch.cat([torch.zeros(Ff.shape[0], dtype=torch.bool), ho])
            s2_ = fit_probe_t(F_cat, y_cat, tr_cat, ho_cat)
            if s2_ is not None:
                pf[ho.numpy()] = s2_.cpu().numpy()
                pf_fit = True
        cols[f"{k}|pf"] = pf.astype(np.float32)

        # |pr -- the same per-slot fit and score as `pw`, over the NEVER-TRAINED twin.
        fr = trunk_features(twin, obs, wr, canon, blk0, span, s, n_blocks, device, batch=BATCH)
        pr = np.full(n, np.nan, np.float64)
        sr_ = fit_probe_t(fr["postm"], y_t, tr, ho)
        if sr_ is not None:
            pr[ho.numpy()] = sr_.cpu().numpy()
        cols[f"{k}|pr"] = pr.astype(np.float32)

        # the global pass needs the rendered configuration, not the features, so it can use
        # `VoProjBank`'s OWN masking idiom (which is not `mask_outside`'s -- see NOTES).
        with torch.no_grad():
            x2 = render_write(obs.to(device), wr.to(device), canon, blk0, span, s).cpu()
        glob[k] = {"x": x2.numpy().astype(np.int16), "y": lab_np, "hold": ho.numpy(),
                   "blk0": blk0, "span": span, "level": int(mt["level"]), "which": which}

        per_key.append({"key": k, "n": n, "which": which, "level": int(mt["level"]),
                        "n_hold": int(ho.sum()), "base": float(lab_np.mean()),
                        "pw_fit": bool(sc is not None), "pf_fit": bool(pf_fit),
                        "pr_fit": bool(sr_ is not None)})

    # ------------------------------------------------------------------------------- #
    # THE PROJECTION'S OWN FORM, fit on each diet, scored on the held-out probe rows
    # ------------------------------------------------------------------------------- #
    # `pw`/`pf` are `duplex`'s per-slot ridge over `trunk_features`' pooling.  The object in the
    # grader's seat is not that: it is ONE GLOBAL readout with `VoProjBank`'s design -- the
    # masked read (its own mask idiom), all-block AND span pooling, a group one-hot and its
    # INTERACTION with the standardised features, IRLS with the ridge chosen on a
    # train-internal split.  `preplay`'s re-implementation of that form is gated elementwise at
    # 0.000e+00 against `VoProjBank` itself (`preplay.py::fidelity_gate`) and is imported here
    # rather than written again.
    #
    # ONE FORCED DEVIATION, and it is not small: `VoProjBank`'s grouping variable is the ROOT
    # (the goal), which makes it one linear readout PER GOAL at 1737 columns.  THE DUMP CARRIES
    # NO ROOT -- the recorder banks it only into `vo_bank.npz`, whose rows are the trajectory's
    # FINAL configurations and cannot be joined to a probe row.  So the grouping variable here
    # is the slot's LEVEL (4 groups, 965 columns).  The form, the pooling, the interaction, the
    # estimator and the ridge ladder are the projection's; the grouping variable is not, and
    # every number in the `gw`/`gf` columns carries that.
    glob_stat = {}
    if glob:
        keys_g = sorted(glob)
        cat = {c: np.concatenate([glob[k][c] for k in keys_g]) for c in ("y", "hold")}
        cat["x"] = np.concatenate([glob[k]["x"] for k in keys_g])
        cat["blk"] = np.concatenate([np.full(glob[k]["x"].shape[0], glob[k]["blk0"], np.int64)
                                     for k in keys_g])
        cat["spn"] = np.concatenate([np.full(glob[k]["x"].shape[0], glob[k]["span"], np.int64)
                                     for k in keys_g])
        cat["r"] = np.concatenate([np.full(glob[k]["x"].shape[0],
                                           LVL_IX[glob[k]["level"]], np.int64)
                                   for k in keys_g])
        cat["probe"] = np.concatenate([np.full(glob[k]["x"].shape[0],
                                               glob[k]["which"] == "probe")
                                       for k in keys_g])
        bank_t = {"x": torch.from_numpy(cat["x"].astype(np.int64)),
                  "y": torch.from_numpy(cat["y"].astype(np.float64)),
                  "blk": torch.from_numpy(cat["blk"]), "spn": torch.from_numpy(cat["spn"]),
                  "r": torch.from_numpy(cat["r"])}
        # the rows every global fit is SCORED on: the held-out probe rows, all of them
        sc_idx = np.nonzero(cat["hold"] & cat["probe"])[0]
        grng = np.random.default_rng(GLOB_SEED)
        for nm_, want_probe in (("gw", True), ("gf", False)):
            pool = np.nonzero((~cat["hold"]) & (cat["probe"] == want_probe))[0]
            if pool.size < 512 or sc_idx.size < 16:
                glob_stat[nm_] = {"n_pool": int(pool.size), "fit": False}
                continue
            take = grng.permutation(pool.size)[:min(GLOB_N, pool.size)]
            sel = pool[np.sort(take)]
            ntr = int(round(GLOB_BOOT * sel.size))
            fit = pj_fit(core, bank_t, sel[:ntr], sel[ntr:], s, n_blocks, dim, len(LVL_IX),
                         device, log=log, name=f"{nm_}@{arm}")
            if fit is None:
                glob_stat[nm_] = {"n_pool": int(pool.size), "fit": False}
                continue
            pp = pj_predict(core, bank_t["x"][torch.from_numpy(sc_idx)],
                            bank_t["r"][torch.from_numpy(sc_idx)],
                            bank_t["blk"][torch.from_numpy(sc_idx)],
                            bank_t["spn"][torch.from_numpy(sc_idx)],
                            fit["w"], fit["mu"], fit["sd"], s, n_blocks, dim, len(LVL_IX),
                            device).numpy()
            full = np.full(cat["y"].shape[0], np.nan)
            full[sc_idx] = pp
            off = 0
            for k in keys_g:
                nk = glob[k]["x"].shape[0]
                cols[f"{k}|{nm_}"] = full[off:off + nk].astype(np.float32)
                off += nk
            glob_stat[nm_] = {"n_pool": int(pool.size), "n_fit": int(ntr),
                              "n_val": int(sel.size - ntr), "n_scored": int(sc_idx.size),
                              "lam": fit["lam"], "val_auc": fit["val_auc"],
                              "hold_auc_probe": pj_auc(pp, cat["y"][sc_idx]), "fit": True}
            log(f"  [{nm_}] lam {fit['lam']:.0f}  val {fit['val_auc']:.4f}  "
                f"pooled held-out probe AUC {glob_stat[nm_]['hold_auc_probe']}")

    # GATE P -- the projection's form, VERBATIM (root grouping, v = 8, 1737 columns), refit on
    # the FINAL trunk over the arm's own bank, against the readout's own banked held-out AUC.
    # Not an equality: the run refit 141-151 times against a MOVING trunk and this is one refit
    # on the trunk it stopped at, so a band is the most the check can be.  It is what says the
    # `pj_*` machinery is being driven correctly; the `gw`/`gf` columns above are the same
    # machinery with the grouping variable forced.
    gate_p = {"name": "P", "ran": False}
    bank_p = os.path.join(root, "vo_bank.npz")
    if os.path.isfile(bank_p):
        zb = np.load(bank_p)
        bt = {"x": torch.from_numpy(zb["x"].astype(np.int64)),
              "y": torch.from_numpy(zb["y"].astype(np.float64)),
              "blk": torch.from_numpy(zb["blk"].astype(np.int64)),
              "spn": torch.from_numpy(zb["spn"].astype(np.int64)),
              "r": torch.from_numpy(zb["r"].astype(np.int64))}
        hb = zb["h"] < float(cfgh.get("vo_critic_hold", 0.1))
        trb = np.nonzero(~hb)[0]
        hob = np.nonzero(hb)[0]
        rb = np.random.default_rng(GLOB_SEED + 1)
        # the run's OWN per-refit sizes, read off its last `vo_om` record: fit_n 8192,
        # val_n 2048 (`vo_pj_fit_cap = 8192` at `vo_pj_boot = 0.8`), and the held-out AUC is
        # read on `VoOutcomeBank.refresh`'s last 2048 held-out rows, not on all of them.
        trb = trb[np.sort(rb.permutation(trb.size)[:10240])]
        hob = hob[-2048:]
        ntr = int(round(GLOB_BOOT * trb.size))
        fitp = pj_fit(core, bt, trb[:ntr], trb[ntr:], s, n_blocks, dim, v, device,
                      log=log, name=f"P@{arm}")
        if fitp is not None:
            pph = pj_predict(core, bt["x"][torch.from_numpy(hob)],
                             bt["r"][torch.from_numpy(hob)],
                             bt["blk"][torch.from_numpy(hob)],
                             bt["spn"][torch.from_numpy(hob)],
                             fitp["w"], fitp["mu"], fitp["sd"], s, n_blocks, dim, v,
                             device).numpy()
            here = pj_auc(pph, zb["y"][hob].astype(np.float64))
            banked = None
            try:
                banked = float(res_j["log"]["vo_om"][-1]["hold_auc"])
            except Exception:
                for e_ in reversed(res_j["log"].get("vo_om", []) or []):
                    if isinstance(e_, dict) and e_.get("hold_auc") is not None:
                        banked = float(e_["hold_auc"])
                        break
            gate_p = {"name": "P", "ran": True, "hold_auc_here": here,
                      "hold_auc_banked": banked, "lam": fitp["lam"], "n_hold": int(hob.size),
                      "n_fit": int(ntr), "cols": int(2 * dim + v + 2 * dim * v + 1),
                      "pass": (None if banked is None else bool(abs(here - banked) < 0.08))}
            log(f"  [gate P] projection form VERBATIM (v={v}, "
                f"{gate_p['cols']} cols): refit hold AUC {here:.4f} against banked "
                f"{banked}  lam {fitp['lam']:.0f}")

    out_dir = os.path.join(DATA_DIR, REMOTE, out_tag, tag, arm)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "within_scores.npz"), **cols)
    gates.append(gate_b)
    gates.append(gate_p)
    rec = {"bank": bank, "tag": tag, "arm": arm, "gates": gates, "per_key": per_key,
           "glob": glob_stat, "twin_seed": TWIN_SEED, "twin_fingerprint": fp_twin,
           "hold_thr": thr, "vo_w": w_compose, "n_keys": len(per_key),
           "sec": time.time() - t00,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    with open(os.path.join(out_dir, "within_gates.json"), "w") as fh:
        json.dump(rec, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(out_dir, "within.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    log(f"  [gate B] batch {BATCH} vs {GATE_BATCH}: crit {gate_b['max_delta_crit']:.3e}  "
        f"feat {gate_b['max_delta_feat']:.3e}")
    log(f"  done {rec['sec']:.1f}s   peak RSS {rec['peak_rss_mb']:.0f} MB   "
        f"{len(per_key)} keys, {sum(p['n'] for p in per_key)} rows")
    return {"arm": f"{tag}/{arm}", "sec": rec["sec"], "rss_mb": rec["peak_rss_mb"],
            "gates": {g["name"]: g.get("pass") for g in gates},
            "n_rows": sum(p["n"] for p in per_key)}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=7200, memory=1024)
def score(out_tag="wi2", arms="sb", smoke=False):
    """CPU coordinator: the arms across containers.  They are independent reads of independent
    banked files, so the fan-out adds no compute — it only stops eight sequential passes from
    costing eight times the longest one."""
    want = SB if arms == "sb" else OV if arms == "ov" else SB + OV
    outs = list(score_arm.starmap([(b, t, a, out_tag, bool(smoke)) for b, t, a in want]))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"arms": arms, "smoke": bool(smoke),
                             "n": len(want)}) + "\n")
    volume.commit()
    return outs


# --------------------------------------------------------------------------------------- #
# the falsification harness
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800, memory=4096, cpu=2.0)
def falsify(bank="rhm_practice_soundboard", tag="sb_s1", arm="sb_sv_yk"):
    """A gate is not reported until it has been SHOWN TO FAIL.  Each check below breaks exactly
    one thing the corresponding gate watches and asserts the gate trips; every one of them is a
    mistake that was available while this node was built, not a synthetic perturbation.

        Z1  ddof=0 instead of torch's unbiased `.std()`            -> Z must trip
        Z2  the weight w put on the prior instead of the judge     -> Z must trip
        Z3  no clamp_min(1e-6) on a degenerate candidate set       -> Z must trip
        A1  ties ranked by position instead of mid-ranked          -> A must trip
        C1  one plant parameter moved by 1e-3                      -> C must trip
        B1  the UNMASKED post-write read substituted for the       -> B must trip
            masked one (the read of record is vo_pj_mask = True)
        H1  one hold code flipped inside a group                   -> H must trip
        D1  a second candidate injected into a filed group         -> the degeneracy count
            must fall (the filed-side claim is a measurement, and
            this shows the counter can move)
        R1  the twin at another seed / the trained core                -> the fingerprints must
            differ, or `pr` is not the floor it is labelled
        P1  the projection form's root column shuffled                 -> REPORTED, not
            required: the per-goal block turns out to be very nearly
            inert, which is a fact about the object (see the comment)
    """
    import torch
    import rhm.practice.native.span.span_net as SN
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.voicing.voicing import vo_compose, vo_auc
    from rhm.practice.voicing.sotto_voce.aliquot.duplex.duplex import (
        trunk_features, fingerprint)
    from rhm.practice.voicing.sotto_voce.aliquot.soundboard.within.reduce_within import (
        np_auc, np_compose, row_facts, load_arm)

    out = []

    def check(nm, broke, why, require=True):
        out.append({"name": nm, "tripped": bool(broke), "why": why,
                    "required": bool(require)})
        tag_ = ("TRIPPED (good)" if broke else
                ("DID NOT TRIP (BAD)" if require else "DID NOT TRIP (reported, not required)"))
        print(f"  [{nm}] {tag_}  {why}", flush=True)

    rng = np.random.default_rng(31337)
    dp, cr, span, w = rng.normal(size=9), rng.normal(size=9), 8, 0.4
    ref = vo_compose(torch.as_tensor(dp * span)[None, :], torch.as_tensor(cr)[None, :],
                     span, w).numpy()[0]
    z0 = lambda a: (a - a.mean()) / max(a.std(ddof=0), 1e-6)            # noqa: E731
    np_z_ref = lambda a: (a - a.mean()) / max(a.std(ddof=1), 1e-6)      # noqa: E731
    check("Z1", float(np.abs(ref - (z0(dp) + w * z0(cr))).max()) > 1e-12, "ddof=0")
    # Z2's FIRST FORM WAS NOT A FALSIFICATION, and the reason is a fact about the composition
    # worth keeping: `vo_compose` divides the prior by the span and THEN standardises over the
    # candidate set, and the standardisation annihilates any positive scale -- so feeding the
    # dump's already-divided |dp without compensating changes nothing.  The span division is
    # inert inside `vo_compose`; it matters only where the standardisation is skipped, which is
    # the size-1 guard (and for a raw dp column pooled ACROSS levels, which is why the dump
    # divides).  The falsification that does bite is the weight on the wrong organ.
    check("Z2", float(np.abs(ref - (np_z_ref(dp) * w + np_z_ref(cr))).max()) > 1e-12,
          "w applied to the prior instead of the judge (w != 1)")
    dg = np.full(9, 3.0)
    refd = vo_compose(torch.as_tensor(dg * span)[None, :], torch.as_tensor(cr)[None, :],
                      span, w).numpy()[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = (dg - dg.mean()) / (dg.std(ddof=1))
    check("Z3", not np.isfinite(bad).all() or
          float(np.abs(refd - (bad + w * z0(cr))).max()) > 1e-12, "no clamp_min on std 0")

    x = np.round(rng.normal(size=300) * 0.2, 1)
    y = (rng.uniform(size=300) < 0.4).astype(np.float64)
    o = np.argsort(x, kind="stable")
    r = np.empty(300)
    r[o] = np.arange(1, 301)
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    tied = float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    check("A1", abs(vo_auc(x, y) - tied) > 1e-12, "ties ranked by position")
    check("A0", abs(vo_auc(x, y) - np_auc(x, y)) < 1e-12, "the estimator itself still agrees")

    root = os.path.join(DATA_DIR, bank, tag, arm)
    blob = torch.load(os.path.join(root, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    cfgh = blob["cfg"]
    cfgr = json.load(open(os.path.join(root, "results.json")))["config"]
    v, s, depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
    dim, length = int(cfgh["state_dim"]), s ** int(cfgh["depth"])
    core = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False)
    core.load_state_dict(blob["core"])
    core.eval()
    fp0 = fingerprint(core)
    sig = float(json.load(open(os.path.join(root, "results.json")))["log"]["sb"][-1]["sig"])
    with torch.no_grad():
        next(iter(core.parameters())).view(-1)[0] += 1e-3
    check("C1", abs(fingerprint(core) - sig) > 1e-6 and abs(fp0 - sig) < 1e-6,
          "one parameter moved by 1e-3")
    with torch.no_grad():
        next(iter(core.parameters())).view(-1)[0] -= 1e-3

    rules = generate_rules_distinct(v, s, depth, int(cfgr["m"]), seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]), dtype=torch.long)
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))
    z = np.load(os.path.join(root, "vo_rows.npz"))
    k = sorted(kk for kk in meta if kk.startswith("probe"))[0]
    mt = meta[k]
    obs = torch.from_numpy(z[f"{k}|obs"][:1024].astype(np.int64))
    wr = torch.from_numpy(z[f"{k}|write"][:1024].astype(np.int64))
    f = trunk_features(core, obs, wr, canon, int(mt["blk0"]), int(mt["span"]), s,
                       length // s, torch.device("cpu"), batch=1024)
    check("B1", float((f["postm"].double() - f["post"].double()).abs().max()) > 1e-6,
          "unmasked read substituted for the masked read of record")

    thr = int(round(100 * float(cfgr["vo_critic_hold"])))
    bufs = load_arm(os.path.join(DATA_DIR, bank), "", tag, arm, thr)
    base = row_facts(bufs)
    kf = sorted(kk for kk in bufs if kk.startswith("filed"))[0]
    kp = sorted(kk for kk in bufs if kk.startswith("probe"))[0]
    b2 = {kk: dict(vv) for kk, vv in bufs.items()}
    b2[kp] = dict(b2[kp])
    b2[kp]["code"] = b2[kp]["code"].copy()
    # find a group of size >= 2 in the probe buffer and flip one member's code
    seen = {}
    hit = None
    for i in range(b2[kp]["obs"].shape[0]):
        t = b2[kp]["obs"][i].tobytes()
        if t in seen:
            hit = i
            break
        seen[t] = i
    b2[kp]["code"][hit] = (b2[kp]["code"][hit] + 1) % 100
    check("H1", row_facts(b2)["marginal"]["probe"]["code_viol"] > 0
          and base["marginal"]["probe"]["code_viol"] == 0, "one hold code flipped")

    b3 = {kk: dict(vv) for kk, vv in bufs.items()}
    b3[kf] = dict(b3[kf])
    b3[kf]["write"] = b3[kf]["write"].copy()
    seen, hit = {}, None
    for i in range(b3[kf]["obs"].shape[0]):
        t = b3[kf]["obs"][i].tobytes()
        if t in seen:
            hit = i
            break
        seen[t] = i
    b3[kf]["write"][hit] = (b3[kf]["write"][hit] + 1) % int(cfgh["v"])
    check("D1", row_facts(b3)["marginal"]["filed"]["onecand"]
          < base["marginal"]["filed"]["onecand"],
          "a second candidate injected into one filed group")

    # R -- the never-trained twin is a fresh draw and its SEED is load-bearing.  A twin minted
    # before the trained core is loaded, or at another seed, is a different object; `duplex` and
    # `overtone` use 20260915 and `aliquot`'s in-loop arm 20260918, and the two are not the same
    # floor.  Both halves must hold or the `pr` column is not the object it is labelled.
    torch.manual_seed(20260918)
    tw_a = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False)
    torch.manual_seed(20260915)
    tw_b = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False)
    fa, fb = fingerprint(tw_a), fingerprint(tw_b)
    check("R1", abs(fa - fp0) > 1.0 and abs(fa - fb) > 1.0,
          f"twin {fa:.3f} vs trained {fp0:.3f} vs seed-20260915 twin {fb:.3f}")

    # P -- the projection form's per-GOAL block is load-bearing: shuffling the root column
    # should cost the refit's held-out AUC on the bank.  Reported with the number, because a
    # band-based check is only as good as what it is shown to move.
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.preplay import pj_fit, pj_predict
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.preplay import vo_auc as pj_auc
    bp = os.path.join(root, "vo_bank.npz")
    if os.path.isfile(bp):
        zb = np.load(bp)
        mk = lambda rr: {"x": torch.from_numpy(zb["x"].astype(np.int64)),            # noqa
                         "y": torch.from_numpy(zb["y"].astype(np.float64)),
                         "blk": torch.from_numpy(zb["blk"].astype(np.int64)),
                         "spn": torch.from_numpy(zb["spn"].astype(np.int64)),
                         "r": torch.from_numpy(rr.astype(np.int64))}
        hb = zb["h"] < 0.1
        trb = np.nonzero(~hb)[0]
        hob = np.nonzero(hb)[0][-2048:]
        rb = np.random.default_rng(20260922)
        trb = trb[np.sort(rb.permutation(trb.size)[:10240])]
        ntr = int(round(0.8 * trb.size))
        aucs = {}
        for nm_, rr in (("true", zb["r"]), ("shuffled", rb.permutation(zb["r"]))):
            bt = mk(rr)
            ft = pj_fit(core, bt, trb[:ntr], trb[ntr:], s, length // s, dim, v,
                        torch.device("cpu"), name=nm_)
            pp = pj_predict(core, bt["x"][torch.from_numpy(hob)],
                            bt["r"][torch.from_numpy(hob)],
                            bt["blk"][torch.from_numpy(hob)],
                            bt["spn"][torch.from_numpy(hob)],
                            ft["w"], ft["mu"], ft["sd"], s, length // s, dim, v,
                            torch.device("cpu")).numpy()
            aucs[nm_] = pj_auc(pp, zb["y"][hob].astype(np.float64))
        # P1 IS REPORTED, NOT REQUIRED, and the reason is a fact about the object rather than
        # a weak harness: shuffling the root column costs the projection's held-out AUC on the
        # bank about 0.004, so ITS PER-GOAL BLOCK IS VERY NEARLY INERT there.  That is the same
        # order as the run's own banked gap between the interaction design and the additive one
        # (`hold_auc` 0.8187 against `hold_auc_add` 0.8068).  It matters for this node because
        # `gw`/`gf` are forced to group by the LEVEL rather than the root: if the root block
        # were carrying the projection's read, that substitution would be fatal, and it is not.
        check("P1", aucs["true"] - aucs["shuffled"] > 0.01,
              f"root column shuffled: hold AUC {aucs['true']:.4f} -> {aucs['shuffled']:.4f} "
              f"(delta {aucs['true'] - aucs['shuffled']:+.4f})", require=False)

    req = [r_ for r_ in out if r_["required"]]
    n_bad = sum(1 for r_ in req if not r_["tripped"])
    print(f"\n  falsification: {len(req) - n_bad}/{len(req)} required checks tripped; "
          f"{len(out) - len(req)} reported and not required", flush=True)
    assert n_bad == 0, [r_ for r_ in req if not r_["tripped"]]
    return out
