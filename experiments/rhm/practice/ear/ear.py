"""ear — earning an ear at the next level: re-instantiating the EVALUATION LAYER.

Round 3 of the practice arc on RHM. Round 2 (`../ratchet/`) earned a level-indexed macro
vocabulary and recovered 68-98% of what being handed the DGP's own vocabulary buys -- but lost
era 3, because its unit-LP certificate refused to commit a level-3 macro it could not grade.
Its own diagnosis: THE GRADER HAS NOT CLIMBED WITH THE VOCABULARY. A certificate cannot be
better than the audition it reads.

WHAT WAS ACTUALLY WRONG WITH THE AUDITION (measured from `rr_s0`'s own logs, before this round
was built). The ratchet auditioned a candidate by applying it ONCE to held-out instances of the
CURRENT era's damage. Two coordinates were off, and the round only named one:

  * ctx -- the unit is consumed on era-(k+1) damage, and was auditioned on era-k damage;
  * ev  -- the unit is consumed INSIDE the agent's beam at the declared grounding budget, and
           was auditioned as an isolated action.

On this substrate the second dominates. A macro's one-action audition masks its own span and
predicts from the surrounding context, which is clean in EVERY era, so the true level-3 macro's
one-action score is 0.495 / 0.497 / 0.513 across the three eras -- essentially
context-invariant -- while the same vocabulary inside the beam takes era-3 error from 0.642 to
0.347. The ratchet's level-3 series had a total span of 0.11 against 0.04-0.09 of noise, which
is why `lp_min_drop` was never met and the certificate refused.

THREE MECHANISMS, each pricing its own consumption (`grader.py` builds the first two):

 1. SELF-MANUFACTURED AUDITION CONTEXTS. The agent damages ITS OWN clean derivations at level
    k+1 -- overwriting each level-k child of the enclosing node with a different entry of its
    own level-k table, read by its own reader, rendered with the same `canon` it already uses
    -- and auditions there. Practice manufacturing its own tests: component (1) of the
    definition applied to the grader instead of to the task. RHM makes it oracle-checkable, and
    the check runs as an instrument every time a set is built (`grader.context_stats`).

 2. PROVISIONAL COMMITMENT + PRICED LIVE RECERT. Commit cheaply at the era boundary (no
    pre-commit audition paid at all), then grade the unit LIVE, with the policy evaluator, in
    the real level-(k+1) contexts where it is consumed, and swap the committed table for the
    live mined one when the live one wins by a margin. The ratchet measured the frozen-vs-live
    counterfactual; here it is acted on.

 3. A `taught` ARM. The teacher hands over an already-climbed grader -- a correctly-levelled,
    consumption-matched audition set -- while the student still earns every bit of CONTENT
    itself. Teaching as transfer of the grader, not of the content (ratchet interpretation
    (e)), and the upper bound mechanism 1 is trying to reach.

ARMS differ only in HOW THE VOCABULARY IS GRADED (and, for `given`, whether it is earned):

    never_base      base level-1 moves forever                       (anchor)
    given           the DGP's own tables from cycle 1                (ceiling)
    practice_gated  ratchet's grader verbatim: (era_k, action)       (the control to beat)
    practice_late   ratchet's schedule commit                        (schedule control)
    practice_self   (mfg, policy) + unit-LP certificate              (mechanism 1)
    taught          (real, policy) + unit-LP certificate             (mechanism 3)
    practice_prov   no pre-commit grader; provisional + live recert  (mechanism 2)
    practice_climb  (mfg, policy) certificate, provisional fallback, + live recert  (1 + 2)

Arms 1-3 run FIRST, with the ratchet's code path untouched and a dedicated RNG stream for
everything new, so `rr_s0`'s `never_base`, `given` and `practice_gated` reproduce cycle for
cycle -- the fork's fidelity gate.

Every arm logs the whole 3x2 grader grid every cycle as an INSTRUMENT (unpriced): the one-action
and policy auditions on era-k, manufactured and real contexts, plus the no-candidate policy
reference on each. Only the cell an arm's certificate actually reads is priced. That is what
makes "what would grader X have done here" an offline replay rather than another run, and what
makes the grader-cost accounting real.

Run from experiments/:
  modal run rhm/practice/ear/ear.py::selfcheck_remote
  modal run rhm/practice/ear/ear.py::ear --quick --tag smoke0
  python3 rhm/practice/ear/launch_detached.py --fn ear --tag er_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import grade, oracle_rollout
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    audition_macro, beam_moves, build_ms, build_shared, context_instances, era_ctx,
    finetune_generator, fit_width, measure_refs, parse_arms, parse_eras, plant_probe,
    priced, push, selfcheck as ratchet_selfcheck, value_steps)
from rhm.practice.ear import grader as GR


app = modal.App("rhm-practice-ear", image=image)

REMOTE = "rhm_practice_ear"


# --------------------------------------------------------------------------- #
# arms: the vocabulary is earned the same way everywhere; only the GRADER differs
# --------------------------------------------------------------------------- #
#   grader  -- which cell of the (ctx, ev) grid the unit-LP certificate reads
#   pay_era -- whether the arm pays the ratchet's per-cycle era-k audition. True exactly for
#              the arms whose code path IS the ratchet's, so their priced time stays comparable
#              to `rr_s0` cycle for cycle
#   recert  -- whether a committed table is re-graded live, in consumption, and can be swapped
ARMS = {
    "never_base":     {"vocab": "base",   "commit": None,         "grader": None,
                       "pay_era": True,  "recert": False},
    "given":          {"vocab": "true",   "commit": None,         "grader": None,
                       "pay_era": True,  "recert": False},
    "practice_gated": {"vocab": "earned", "commit": "delta",      "grader": "era",
                       "pay_era": True,  "recert": False},
    "practice_late":  {"vocab": "earned", "commit": "late",       "grader": "era",
                       "pay_era": True,  "recert": False},
    "practice_self":  {"vocab": "earned", "commit": "delta",      "grader": "mfg",
                       "pay_era": False, "recert": False},
    "taught":         {"vocab": "earned", "commit": "delta",      "grader": "real",
                       "pay_era": False, "recert": False},
    "practice_prov":  {"vocab": "earned", "commit": "prov",       "grader": None,
                       "pay_era": False, "recert": True},
    "practice_climb": {"vocab": "earned", "commit": "delta_prov", "grader": "mfg",
                       "pay_era": False, "recert": True},
}


def node_at(era, level, s):
    """The level-`level` node containing this era's damage cell. On a NESTED ladder this is
    exactly the next era's damage cell (gate G-N), which is what lets the agent audition at
    the consumption node without being told the future."""
    span = s ** (level - 1)
    return (era["node"] * s ** (era["level"] - 1)) // span


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]
    canon_np, bottom_np = shared["canon_np"], shared["bottom_np"]
    inv_bottom = shared["inverse_maps"][-1]

    # ---- per-arm plant + selector (the ratchet's, verbatim) ------------------------------
    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(),
                             lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    # a DEDICATED stream for the grader machinery, so adding an evaluation layer cannot
    # perturb the ratchet's own draws and arms 1-3 stay comparable to `rr_s0` cycle for cycle
    arng = np.random.default_rng(cfg["seed"] + 4242)

    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
    ms = build_ms(base_ms, committed, s, depth, device)
    p_width = cfg["pr_width"]

    def operative(ell):
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    def pol(x, r_np, mset):
        return GR.policy_e(controller, generator, value, x, r_np, mset, rules, rules_t, canon,
                           depth, v, m, s, budget=cfg["budget"], g_budget=cfg["g_budget"],
                           device=device, beam_moves=beam_moves, fit_width=fit_width)

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    # the agent's own clean derivations, banked -- the raw material a self-manufactured test is
    # made of. It cannot grade itself at level k+1 until it has practised enough to have a test.
    bank_x = torch.zeros(0, shared["length"], dtype=torch.long)
    bank_r = torch.zeros(0, dtype=torch.long)
    t_cum = 0.0
    events = []
    log = {"cycle": [], "era": [], "level": [], "t_cum": [], "e": [], "succ": [], "dres": [],
           "n_moves": [], "width": [], "g_per_solve": [], "e_practice": [], "vloss": [],
           "gloss": [], "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [],
           "probe": [], "vocab": [], "clim": [], "gcost": [], "bank": [], "mfg": []}

    # ---- fixed held-out sets, per era ----------------------------------------------------
    meter, shadow, realset = {}, {}, {}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        meter[i] = (r_np, torch.from_numpy(x_np))
        r2, x2 = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   seed=cfg["seed"] + 6100 + 23 * i)
        shadow[i] = (r2, torch.from_numpy(x2).to(device))
        # THE TEACHER'S SET: real damage at the level being EARNED this era, at the node where
        # the unit will be consumed. On a nested ladder that is exactly the next era's own
        # damage cell (gate G-N) -- but it is defined from this era alone, so the teacher's
        # grader exists even for the last era on the ladder. Handed to `taught` as its audition
        # contexts; carried by every arm as the reference the self-manufactured distribution is
        # graded against (instrument, never acted on).
        act_i = era["level"] + 1
        if act_i <= cfg["max_macro_level"]:
            nd = node_at(era, act_i, s)
            rr, xr = context_instances(
                rules, {"name": f"L{act_i}n{nd}", "level": act_i, "nodes": [nd]},
                cfg["n_aud"], s, depth, v, m, seed=cfg["seed"] + 800_000 + 31 * i)
            realset[i] = (rr, torch.from_numpy(xr).to(device))

    cyc = 0
    for era_i, era in enumerate(eras):
        active = era["level"] + 1                     # the macro level being EARNED this era
        cert = {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0,
                "fired": None}
        mfg = None                                    # the self-manufactured audition set
        era_start, era_end = cyc + 1, cyc + cfg["era_cycles"]
        node_next = node_at(era, active, s) if active <= maxl else None
        print(f"\n----- arm={arm} ERA {era_i + 1}: damage {era['name']} "
              f"(earning level {active}, consumption node {node_next}) "
              f"c{era_start}..{era_end} -----", flush=True)

        for c_in_era in range(1, cfg["era_cycles"] + 1):
            cyc += 1
            counts = {"mat": 0, "ground": 0}
            gcost = {}                     # every grader cell's price, logged whether paid or not

            # --- (a) practice ------------------------------------------------------------
            r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 100_000 + 1000 * cyc)
            out = beam_moves(controller, generator, value, torch.from_numpy(x_np),
                             torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                             budget=cfg["budget"], beam_width=p_width, device=device,
                             collect=True)
            for key in counts:
                counts[key] += out["counts"][key]
            tips = out["tips_x"]
            B, W, T = tips.shape
            tips_flat = tips.reshape(B * W, T)
            succ, _ = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_practice = 1.0 - float(ps.mean())

            # --- (b) mine (the ratchet's, verbatim) ---------------------------------------
            mine_src = (out["x"][torch.from_numpy(ps > 0.5).to(device)]
                        if cfg["mine_from"] == "chosen" else solved)
            if cfg["mine_cap"] and mine_src.shape[0] > cfg["mine_cap"]:
                sel = torch.from_numpy(rng.permutation(mine_src.shape[0])[:cfg["mine_cap"]])
                mine_src = mine_src[sel.to(device)]
            if mine_src.shape[0]:
                pf = MC.parse_features(shared["reader"], mine_src, s=s).cpu().numpy()
                for ell in range(2, min(maxl, era["level"] + 1) + 1):
                    span = s ** (ell - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    miners[ell].observe(pf[:, node * span:(node + 1) * span])

            # --- (b2) BANK the agent's own clean derivations (its solved final answers) ----
            keep = torch.from_numpy(ps > 0.5)
            if bool(keep.any()):
                bank_x = torch.cat([bank_x, out["x"][keep.to(device)].cpu()])[-cfg["bank_cap"]:]
                bank_r = torch.cat([bank_r, torch.from_numpy(r_np)[keep]])[-cfg["bank_cap"]:]

            # --- (c) the plant learns, then the selector ----------------------------------
            gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                       shared["bottom_map"], v=v, s=s,
                                       n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                       batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                       device=device, rng=grng)
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)

            # --- (d) metering under PERFORMANCE conditions --------------------------------
            mr_np, mx = meter[era_i]
            width = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
            b = beam_moves(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                           rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                           beam_width=width, device=device)
            for key in counts:
                counts[key] += b["counts"][key]
            msucc, mdres = grade(b["x"].cpu().numpy(), mr_np, rules, s)
            e = 1.0 - float(msucc.mean())
            gps = b["counts"]["ground"] / mx.shape[0]

            # --- (e) the RATCHET'S shadow audition, unchanged: era-k contexts, one action.
            #         For `pay_era` arms this is exactly `rr_s0`'s code path, priced the same
            #         way; for the climbed-grader arms it becomes an unpriced instrument.
            sr_np, sx = shadow[era_i]
            aud = {}
            for ell in range(2, maxl + 1):
                span = s ** (ell - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                cell = {}
                tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                cell["n_entries"] = int(tbl["child"].shape[0])
                if cell["n_entries"]:
                    mv = MC.to_device(MC.make_macro(ell, node, s, tbl), device)
                    cell["cand"] = audition_macro(generator, sx, sr_np, mv, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    cell.update({f"tab_{k}": val for k, val in
                                 MC.grade_table(tbl, shared["truth"][ell]).items()})
                    if ell == active and committed.get(ell) is None:
                        gcost["era"] = {"ground": sx.shape[0], "mat": sx.shape[0]}
                        if spec["pay_era"]:
                            counts["ground"] += sx.shape[0]
                            counts["mat"] += sx.shape[0]
                else:
                    cell["cand"] = None
                mvt = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
                cell["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                              depth, v, m, s, rules)["e"]
                if cell["n_entries"]:
                    full = shared["truth"][ell]
                    n_all = full["child"].shape[0]
                    k = min(cell["n_entries"], n_all)
                    es = []
                    for _ in range(3):
                        kp = np.sort(rng.permutation(n_all)[:k])
                        sub = MC.make_table(ell, full["child"][kp], full["lower"], s)
                        mvr = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                        es.append(audition_macro(generator, sx, sr_np, mvr, rules_t, canon,
                                                 depth, v, m, s, rules)["e"])
                    cell["rand_k"] = float(np.mean(es))
                    cell["rand_k_sd"] = float(np.std(es))
                if committed[ell] is not None and spec["vocab"] == "earned":
                    # the counterfactual recert readout: what the frozen unit scores against
                    # what the live vocabulary would. Kept running after commitment for EVERY
                    # earned arm -- low-level grading retires per certified unit, not globally.
                    mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                    cell["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    live = miners[ell].build(MC.base_table(v) if ell == 2
                                             else miners[ell - 1].build(MC.base_table(v),
                                                                        cfg["mine_support"]),
                                             cfg["mine_support"])
                    if live["child"].shape[0]:
                        mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                        cell["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t, canon,
                                                      depth, v, m, s, rules)["e"]
                        cell["live_entries"] = int(live["child"].shape[0])
                aud[str(ell)] = cell

            # --- (f) THE CLIMBED GRADER: the (ctx x ev) grid at the active level -----------
            clim = {}
            mfg_ev = None
            if active <= maxl:
                cand = miners[active].build(operative(active - 1), cfg["mine_support"])
                has_cand = cand["child"].shape[0] > 0
                ms_cand = (build_ms(base_ms, {**committed, active: cand}, s, depth, device)
                           if has_cand else None)
                mv_cand = (MC.to_device(MC.make_macro(active, node_next, s, cand), device)
                           if has_cand else None)
                mv_true = MC.to_device(
                    MC.make_macro(active, node_next, s, shared["truth"][active]), device)

                # (f1) build the SELF-MANUFACTURED set once per era, as soon as the agent has
                #      banked enough of its own clean derivations to make a test out of. The
                #      set is then FIXED for the era, exactly as the teacher's set is -- so the
                #      only difference between `practice_self` and `taught` is who made it.
                # `mfg_min_bank`, not `n_aud`, is the readiness bar. `cald_s0` measured why:
                # waiting for a full bank meant the level-2 test did not exist until c5, by
                # which time the descent it is supposed to certify had already happened -- the
                # self-manufactured series had a span of 0.04-0.06 against the real set's 0.36.
                # A test the agent builds out of one cycle's worth of its own solved answers is
                # smaller and noisier, and it is there for the descent. `n` is logged.
                if mfg is None and bank_x.shape[0] >= cfg["mfg_min_bank"]:
                    # `mfg_source` picks which of its own tables the agent damages with:
                    # "child" composes two level-(active-1) entries (always available, drawn
                    # from a different table than the one under audition); "node" overwrites
                    # with one level-`active` entry (a legal chunk, but only once it has
                    # catalogued some). `cal_mfg` decides.
                    lower = (operative(active) if cfg["mfg_source"] == "node"
                             else operative(active - 1))
                    if int(lower["child"].shape[0]) >= 2:
                        n_use = min(cfg["n_aud"], int(bank_x.shape[0]))
                        pick = arng.permutation(bank_x.shape[0])[:n_use]
                        seeds = bank_x[pick].numpy()
                        sroots = bank_r[pick].numpy()
                        rfeat = MC.parse_features(
                            shared["reader"], torch.from_numpy(seeds).to(device), s=s
                        ).cpu().numpy()
                        dmg, chg = GR.manufacture_damage(
                            seeds, rfeat, lower, active, node_next, s, canon_np, arng,
                            bottom_np=bottom_np, render=cfg["mfg_render"])
                        if dmg is not None:
                            mfg = {"x": torch.from_numpy(dmg).to(device), "r": sroots,
                                   "built_cycle": cyc, "n": int(dmg.shape[0]),
                                   "lower_entries": int(lower["child"].shape[0]),
                                   "child_changed": chg, "build_mat": int(dmg.shape[0])}
                            mfg_ev = {"arm": arm, "era": era_i + 1, "cycle": cyc,
                                      "level": active, "node": node_next, "n": mfg["n"],
                                      "bank": int(bank_x.shape[0]),
                                      "lower_entries": mfg["lower_entries"],
                                      "child_changed": chg,
                                      "stats": GR.context_stats(dmg, sroots, seeds, rules,
                                                                inv_bottom, v, s, active,
                                                                node_next)}
                            print(f"  [mfg] arm={arm} era{era_i+1} c{cyc}: built {mfg['n']} "
                                  f"level-{active} contexts from its own derivations "
                                  f"(lower table {mfg['lower_entries']} entries) "
                                  f"{json.dumps(mfg_ev['stats'], cls=NumpyEncoder)}",
                                  flush=True)

                sets = {"era": (sx[:cfg["n_aud"]], sr_np[:cfg["n_aud"]]),
                        "real": ((realset[era_i][1], realset[era_i][0])
                                 if era_i in realset else None),
                        "mfg": ((mfg["x"], mfg["r"]) if mfg is not None else None)}
                for key in ("era", "real", "mfg"):
                    ctx = sets.get(key)
                    if ctx is None:
                        continue
                    xk, rk = ctx
                    c = {"n": int(xk.shape[0])}
                    c["act_true"] = audition_macro(generator, xk, rk, mv_true, rules_t, canon,
                                                   depth, v, m, s, rules)["e"]
                    pb = pol(xk, rk, ms)
                    c["pol_base"] = pb["e"]
                    c["pol_w"] = pb["w"]
                    cost = {"ground": pb["counts"]["ground"], "mat": pb["counts"]["mat"]}
                    if has_cand:
                        c["act"] = audition_macro(generator, xk, rk, mv_cand, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                        pc = pol(xk, rk, ms_cand)
                        c["pol"] = pc["e"]
                        c["pol_g"] = pc["g"]
                        c["margin"] = c["pol_base"] - c["pol"]
                        cost["ground"] += pc["counts"]["ground"]
                        cost["mat"] += pc["counts"]["mat"]
                    if key == "mfg" and mfg is not None and mfg["built_cycle"] == cyc:
                        cost["mat"] += mfg["build_mat"]   # manufacturing is on the meter too
                    gcost[key] = cost
                    clim[key] = c
                # the arm PAYS for exactly the cell its certificate reads
                gr_ = spec["grader"]
                if gr_ in ("mfg", "real") and gr_ in gcost and committed.get(active) is None:
                    counts["ground"] += gcost[gr_]["ground"]
                    counts["mat"] += gcost[gr_]["mat"]

            # --- (g) the unit-LP certificate, on whichever series this arm's grader gives ---
            gr_ = spec["grader"]
            if gr_ == "era":
                A = aud.get(str(active), {}).get("cand")
            elif gr_ in ("mfg", "real"):
                A = clim.get(gr_, {}).get("pol")
            else:
                A = None
            fire_cert = False
            if A is not None and active <= maxl and committed.get(active) is None:
                if cert["ref"] is None:
                    cert["ref"] = A; cert["b"] = A; cert["emin"] = A
                cert["emin"] = min(cert["emin"], A)
                d = cert["b"] - A
                scale = max(cert["ref"] - cert["emin"], cert["b"], 1e-6)
                cert["hist"].append(A); cert["dhist"].append(d)
                we, wd = cert["hist"][-cfg["sil_W"]:], cert["dhist"][-cfg["sil_W"]:]
                quiet = (len(we) >= cfg["sil_W"]
                         and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                         and float(np.std(we)) < cfg["sil_cv"] * scale)
                cert["run"] = cert["run"] + 1 if quiet else 0
                cert["b"] = cert["b"] + cfg["alpha"] * (A - cert["b"])
                dropped = (cert["ref"] - cert["emin"]) >= cfg["lp_min_drop"]
                fire_cert = (cert["run"] >= cfg["sil_hold"] and dropped
                             and c_in_era >= cfg["sil_min_cycle"])

            # --- (h) the commit rule -------------------------------------------------------
            do_commit, prov = False, False
            if spec["commit"] and active <= maxl and committed.get(active) is None:
                at_boundary = c_in_era >= cfg["era_cycles"] - cfg["prov_offset"]
                if spec["commit"] == "delta":
                    do_commit = fire_cert
                elif spec["commit"] == "late":
                    do_commit = c_in_era >= cfg["era_cycles"] - cfg["late_offset"]
                elif spec["commit"] == "prov":
                    do_commit, prov = at_boundary, True
                elif spec["commit"] == "delta_prov":
                    do_commit = bool(fire_cert or at_boundary)
                    prov = not fire_cert
            if do_commit:
                tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
                if tbl["child"].shape[0] == 0:
                    do_commit = False
            if do_commit:
                held = None
                if not prov:
                    # a fresh held-out audition, PRICED -- the ratchet's confirmation step. A
                    # PROVISIONAL commit deliberately skips it: you pay to grade the unit where
                    # it is consumed instead of paying to grade it before you have consumed it.
                    cr_np, cx_np = context_instances(rules, era_ctx(era), cfg["n_score"], s,
                                                     depth, v, m,
                                                     seed=cfg["seed"] + 700_000 + 1000 * cyc)
                    cx = torch.from_numpy(cx_np).to(device)
                    mv = MC.to_device(MC.make_macro(active, node_next, s, tbl), device)
                    held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                          v, m, s, rules)["e"]
                    counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
                committed[active] = tbl
                ms = build_ms(base_ms, committed, s, depth, device)
                cert["fired"] = cyc
                ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                          c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg),
                          provisional=bool(prov), audition=held, shadow=A, e_task=e,
                          aud_real_pol=clim.get("real", {}).get("pol"),
                          aud_mfg_pol=clim.get("mfg", {}).get("pol"),
                          aud_era_act=aud.get(str(active), {}).get("cand"),
                          n_entries=int(tbl["child"].shape[0]),
                          n_moves_after=len(ms), sil_run=int(cert["run"]),
                          **{f"tab_{k}": val for k, val in
                             MC.grade_table(tbl, shared["truth"][active]).items()})
                events.append(ev)
                print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} "
                      f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                      f"recall={ev['tab_recall']:.3f} prec={ev['tab_precision']} "
                      f"audition={held} shadow={A} moves={len(ms)}", flush=True)

            # --- (i) THE LIVE RECERT: grade the committed unit where it is CONSUMED ---------
            #         This era's damage cell IS the consumption distribution for the macro level
            #         committed one era ago, so no oracle context is needed -- only feedback the
            #         agent is already buying, and it is priced.
            rc_level = era["level"]
            if (spec["recert"] and rc_level >= 2 and committed.get(rc_level) is not None
                    and c_in_era % cfg["recert_every"] == 0):
                live = miners[rc_level].build(operative(rc_level - 1), cfg["mine_support"])
                xr = shadow[era_i][1][:cfg["n_aud"]]
                rr_ = shadow[era_i][0][:cfg["n_aud"]]
                pf_ = pol(xr, rr_, ms)
                counts["ground"] += pf_["counts"]["ground"]
                counts["mat"] += pf_["counts"]["mat"]
                rc_cost = {"ground": pf_["counts"]["ground"], "mat": pf_["counts"]["mat"]}
                rec = {"kind": "recert", "arm": arm, "era": era_i + 1, "level": rc_level,
                       "cycle": cyc, "c_in_era": c_in_era, "e_frozen": pf_["e"],
                       "n_frozen": int(committed[rc_level]["child"].shape[0]),
                       "n_live": int(live["child"].shape[0]), "swapped": False}
                if live["child"].shape[0]:
                    ms_live = build_ms(base_ms, {**committed, rc_level: live}, s, depth, device)
                    pl_ = pol(xr, rr_, ms_live)
                    counts["ground"] += pl_["counts"]["ground"]
                    counts["mat"] += pl_["counts"]["mat"]
                    rc_cost["ground"] += pl_["counts"]["ground"]
                    rc_cost["mat"] += pl_["counts"]["mat"]
                    rec["e_live"] = pl_["e"]
                    # how many of the policy's answers the swap would actually change: an
                    # identical `e` from two different tables is only informative if the
                    # answers differ, so the discordance is logged rather than assumed
                    rec["n_diff"] = int((pf_["x"] != pl_["x"]).any(1).sum().item())
                    rec.update({f"live_{k}": val for k, val in
                                MC.grade_table(live, shared["truth"][rc_level]).items()})
                    if pl_["e"] <= pf_["e"] - cfg["recert_margin"]:
                        committed[rc_level] = live
                        ms = build_ms(base_ms, committed, s, depth, device)
                        rec["swapped"] = True
                        print(f"[recert] arm={arm} era{era_i+1} L{rc_level} c{cyc} SWAP "
                              f"{rec['n_frozen']}->{rec['n_live']} entries  "
                              f"e {pf_['e']:.4f} -> {pl_['e']:.4f}", flush=True)
                gcost["recert"] = rc_cost
                events.append(rec)

            # --- (j) probes: the width ladder, all eras' metering sets, the plant guard ------
            if c_in_era == 1 or cyc % cfg["probe_every"] == 0 or c_in_era == cfg["era_cycles"]:
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {}}
                for w in cfg["probe_widths"]:
                    bb = beam_moves(controller, generator, value, mx, torch.from_numpy(mr_np),
                                    ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                                    beam_width=w, device=device)
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["ladder"][str(w)] = {"e": 1.0 - float(sc.mean()),
                                               "g": bb["counts"]["ground"] / mx.shape[0]}
                for j in range(len(eras)):
                    jr, jx = meter[j]
                    wj = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                    bb = beam_moves(controller, generator, value, jx, torch.from_numpy(jr), ms,
                                    rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                                    beam_width=wj, device=device)
                    sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                    probe["all_eras"][str(j)] = 1.0 - float(sc.mean())
                probe["plant"] = plant_probe(generator, shared, cfg, device)
                log["probe"].append(probe)

            t_cum += priced(counts, cfg)
            log["cycle"].append(cyc); log["era"].append(era_i + 1)
            log["level"].append(era["level"])
            log["t_cum"].append(t_cum); log["e"].append(e)
            log["succ"].append(float(msucc.mean()))
            log["dres"].append(float(mdres.mean())); log["n_moves"].append(len(ms))
            log["width"].append(width); log["g_per_solve"].append(gps)
            log["e_practice"].append(e_practice); log["vloss"].append(vloss)
            log["gloss"].append(gloss); log["n_solved"].append(int(solved.shape[0]))
            log["n_mined"].append(int(mine_src.shape[0]))
            log["miner"].append({str(ell): miners[ell].state() for ell in range(2, maxl + 1)})
            log["aud"].append(aud)
            log["clim"].append(clim)
            log["gcost"].append(gcost)
            log["bank"].append(int(bank_x.shape[0]))
            if mfg_ev is not None:
                log["mfg"].append(mfg_ev)
            log["cert"].append({"A": A, "b": cert["b"], "run": int(cert["run"]),
                                "ref": cert["ref"], "emin": cert["emin"]})
            log["vocab"].append({str(ell): (None if committed[ell] is None
                                            else int(committed[ell]["child"].shape[0]))
                                 for ell in range(2, maxl + 1)})
            a_act = aud.get(str(active), {})
            fmt = lambda x: "  .  " if x is None else f"{x:.3f}"
            print(f"[c{cyc:3d}] arm={arm:15s} era{era_i+1} t={t_cum:10.0f} e={e:.4f} "
                  f"w={width} nm={len(ms)} A={fmt(A)} "
                  f"era_act={fmt(a_act.get('cand'))} "
                  f"mfg_pol={fmt(clim.get('mfg', {}).get('pol'))} "
                  f"real_pol={fmt(clim.get('real', {}).get('pol'))} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"bank={bank_x.shape[0]} sil={cert['run']}", flush=True)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, arm, cfg, eras, refs, log, events, complete=False)

    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True)
    return {"log": log, "events": events}


def write_results(outdir, arm, cfg, eras, refs, log, events, complete):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "refs": refs,
                   "log": log, "events": events, "complete": complete},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        reader_steps=6_000, plant_holdout=0, holdout_seed=11,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=3e-5, n_corrupt=3, explore_eps=0.3,
        budget=4, pr_width=16, g_budget=58, max_macro_level=3,
        n_pr=64, n_rt=384, n_score=512, n_probe_clean=512,
        era_cycles=30, n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000,
        gen_lr=1e-4, gen_steps=20, mine_support=3, mine_from="chosen", mine_cap=8,
        alpha=0.2, sil_c=0.06, sil_cv=0.15, sil_W=5, sil_hold=2, sil_min_cycle=6,
        lp_min_drop=0.10, early_offset=1, late_offset=3,
        # --- the evaluation layer --------------------------------------------------------
        n_aud=192, mfg_min_bank=32, bank_cap=1024, mfg_render="canon",
        mfg_source="child",
        recert_every=5, recert_margin=0.05, prov_offset=0,
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_widths=(1, 4, 16), probe_every=4,
    )
    cfg.update(kw)
    return cfg


def _shared_plus(cfg, device):
    """`build_shared`, plus the numpy renderings self-manufacturing needs. Nothing about the
    substrate changes -- this is the ratchet's setup verbatim, which is what makes arms 1-3
    comparable to `rr_s0` cycle for cycle."""
    shared = build_shared(cfg, device)
    rules, depth = shared["rules"], cfg["depth"]
    shared["canon_np"] = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    shared["bottom_np"] = np.ascontiguousarray(rules[depth - 1])
    return shared


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def ear(
    tag: str = "smoke",
    arms: str = ("never_base,given,practice_gated,practice_late,practice_self,taught,"
                 "practice_prov,practice_climb"),
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 30, budget: int = 4,
    pr_width: int = 16, g_budget: int = 58, max_macro_level: int = 3,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 512, n_aud: int = 192,
    mfg_min_bank: int = 32, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 8, bank_cap: int = 1024,
    mfg_render: str = "canon", mfg_source: str = "child", recert_every: int = 5,
    recert_margin: float = 0.05,
    prov_offset: int = 0, value_lr_online: float = 3e-5, sil_c: float = 0.06,
    sil_cv: float = 0.15, sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6,
    lp_min_drop: float = 0.10, late_offset: int = 3, probe_every: int = 4,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, era_cycles=era_cycles, budget=budget, pr_width=pr_width,
               g_budget=g_budget, max_macro_level=max_macro_level, n_pr=n_pr, n_rt=n_rt,
               n_score=n_score, n_aud=n_aud, mfg_min_bank=mfg_min_bank,
               n_grad=n_grad, gen_lr=gen_lr,
               gen_steps=gen_steps, mine_support=mine_support, mine_from=mine_from,
               mine_cap=mine_cap, bank_cap=bank_cap, mfg_render=mfg_render,
               mfg_source=mfg_source,
               recert_every=recert_every, recert_margin=recert_margin,
               prov_offset=prov_offset, value_lr_online=value_lr_online, sil_c=sil_c,
               sil_cv=sil_cv, sil_W=sil_win, sil_hold=sil_hold, sil_min_cycle=sil_min_cycle,
               lp_min_drop=lp_min_drop, late_offset=late_offset, probe_every=probe_every,
               plant_holdout=plant_holdout, holdout_seed=holdout_seed, d_fb=d_fb, c_mat=c_mat)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=6, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, late_offset=1, gen_steps=5,
                   bank_cap=256, recert_every=2, mfg_min_bank=16)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"ear tag={tag} arms={arms} eras={ers} device={device}")

    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_mfg(tag: str = "calm_s0", eras: str = "1:6,2:3,3:1", seed: int = 0, budget: int = 4,
            n_aud: int = 256, g_budget: int = 58, max_macro_level: int = 3,
            quick: bool = False):
    """CALIBRATION 1 -- is a self-manufactured audition context a faithful stand-in for the real
    one, and does the POLICY evaluator have the dynamic range the action evaluator lacks?

    Before any loop is built, on the setup plant, per era k with a successor:

      * build context sets at level k+1, node = the enclosing node (= era k+1's own cell):
        `real` (the oracle's `corrupt_hier` -- the reference), `mfg_canon` / `mfg_rand` (the
        agent's manufacture from the TRUE level-k table, so table quality is factored out, at
        the two renderings), and `mfg_solved` (the same, seeded from configurations the agent
        actually solved rather than from clean DGP samples -- the seed-source question);
      * plus `era_k`, the ratchet's own audition contexts, as the mis-levelled control;
      * grade every set with BOTH evaluators (one macro action; the beam at the declared budget
        with the macro at every node of its level) across a coverage ladder of true-table
        subsets, so the answer is a CURVE rather than a point.

    What it decides: `mfg_render`, the seed source, `n_aud`, and whether the round is admissible
    at all. If the manufactured curve does not track the real curve, the mechanism is not a
    stand-in, and that is a halt-and-report rather than a knob to tune."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_aud=n_aud, g_budget=g_budget,
               max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000, n_aud=96)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = _shared_plus(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen, base_ms = shared["generator0"], shared["base_ms"]
    inv = shared["inverse_maps"][-1]
    N = cfg["n_aud"]
    out = {"config": cfg, "eras": ers, "cells": {},
           "gates": {"recert": GR.gate_recert(shared["truth"], v, s)}}

    def pol(x, r_np, mset):
        return GR.policy_e(shared["controller"], gen, shared["value0"], x, r_np, mset, rules,
                           rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                           g_budget=cfg["g_budget"], device=device, beam_moves=beam_moves,
                           fit_width=fit_width)

    for i, era in enumerate(ers[:-1]):
        level = era["level"] + 1
        node = node_at(era, level, s)
        nxt = ers[i + 1]
        lower = shared["truth"][level - 1] if level - 1 >= 2 else MC.base_table(v)
        rng = np.random.default_rng(cfg["seed"] + 2024 + i)
        sets = {}

        rr, xr, cr = context_instances(rules, era_ctx(nxt), N, s, depth, v, m,
                                       seed=cfg["seed"] + 800_000 + 31 * i, with_clean=True)
        sets["real"] = (rr, xr, cr)
        er_, ex_ = context_instances(rules, era_ctx(era), N, s, depth, v, m,
                                     seed=cfg["seed"] + 6100 + 23 * i)
        sets["era_k"] = (er_, ex_, None)

        roots_c, clean_c = _sample_pool(rules, N, s, cfg["seed"] + 991 + i)
        feats_c = MC.parse_features(shared["reader"], torch.from_numpy(clean_c).to(device),
                                    s=s).cpu().numpy()
        for render in ("canon", "rand"):
            d_, _ = GR.manufacture_damage(clean_c, feats_c, lower, level, node, s,
                                          shared["canon_np"], rng,
                                          bottom_np=shared["bottom_np"], render=render)
            sets[f"mfg_{render}"] = (roots_c, d_, clean_c)
        # the OTHER damage source: one whole-node entry of the level-`level` table, so the
        # damage is a legal chunk of a wrong feature -- which is what `corrupt_hier` produces.
        # The gate measured the difference (manufactured level-2 nodes derive nothing 66% of
        # the time, real ones 0%); this cell prices whether that difference reaches the
        # audition.
        d_, _ = GR.manufacture_damage(clean_c, feats_c, shared["truth"][level], level, node, s,
                                      shared["canon_np"], rng,
                                      bottom_np=shared["bottom_np"], render="canon")
        sets["mfg_node"] = (roots_c, d_, clean_c)

        pr_, px_ = context_instances(rules, era_ctx(era), 4 * N, s, depth, v, m,
                                     seed=cfg["seed"] + 314_159 + i)
        xo, _ = oracle_rollout(gen, torch.from_numpy(px_).to(device), pr_, base_ms, rules,
                               rules_t, canon, depth, v, m, s, cfg["budget"])
        so, _ = grade(xo.cpu().numpy(), pr_, rules, s)
        sv = xo[torch.from_numpy(so > 0.5).to(device)].cpu().numpy()[:N]
        svr = pr_[so > 0.5][:N]
        if sv.shape[0] >= 16:
            feats_s = MC.parse_features(shared["reader"], torch.from_numpy(sv).to(device),
                                        s=s).cpu().numpy()
            d_, _ = GR.manufacture_damage(sv, feats_s, lower, level, node, s,
                                          shared["canon_np"], rng,
                                          bottom_np=shared["bottom_np"], render="canon")
            sets["mfg_solved"] = (svr, d_, sv)

        cell = {"level": level, "node": node, "next_era": nxt["name"], "sets": {}}
        full = shared["truth"][level]
        n_all = full["child"].shape[0]
        ks = sorted({1, 2, 4, 8, max(1, n_all // 2), n_all})
        subs = {}
        krng = np.random.default_rng(cfg["seed"] + 77)
        for k in ks:
            keep = np.sort(krng.permutation(n_all)[:k])
            subs[k] = MC.make_table(level, full["child"][keep], full["lower"], s)
        for name in ("real", "era_k", "mfg_canon", "mfg_rand", "mfg_node", "mfg_solved"):
            if name not in sets:
                continue
            r_np, x_np, c_np = sets[name]
            x0 = torch.from_numpy(np.asarray(x_np)).to(device)
            row = {"n": int(x0.shape[0]),
                   "stats": GR.context_stats(np.asarray(x_np), r_np, c_np, rules, inv, v, s,
                                             level, node),
                   "act": {}, "pol": {}}
            row["pol_base"] = pol(x0, r_np, base_ms)["e"]
            for k, tbl in subs.items():
                mv = MC.to_device(MC.make_macro(level, node, s, tbl), device)
                row["act"][str(k)] = audition_macro(gen, x0, r_np, mv, rules_t, canon, depth,
                                                    v, m, s, rules)["e"]
                msk = build_ms(base_ms, {level: tbl}, s, depth, device)
                row["pol"][str(k)] = pol(x0, r_np, msk)["e"]
            cell["sets"][name] = row
            print(f"  [{era['name']} -> L{level}n{node}] {name:11s} "
                  f"{json.dumps(row['stats'], cls=NumpyEncoder)}\n"
                  f"      act: " + " ".join(f"{k}:{x:.3f}" for k, x in row["act"].items())
                  + f"\n      pol: base={row['pol_base']:.3f} "
                  + " ".join(f"{k}:{x:.3f}" for k, x in row["pol"].items()), flush=True)
        out["cells"][era["name"]] = cell

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_mfg.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_mfg.json")
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3, eras="1:6,2:3,3:1"):
    """The ratchet's gates (C-M the macro is bit-identical to the true level move, C-R the
    ratchet bites, G-D the nested damage is on-grammar) carried forward unchanged -- the fork's
    substrate check -- plus the two this round needs:

    G-N  the ladder really is nested: the level-(k+1) node enclosing era k's damage cell IS
         era k+1's damage cell.
    G-S  a self-manufactured audition context is on-grammar 1.000, confined to that node's own
         token span, and correctly levelled, reported beside the real damage's signature.
    G-R  a live recert undoes the ratchet's cap: the level-3 table rebuilt over a truncated
         committed level-2 table is small, and restoring the level-2 table restores it."""
    out = {"ratchet": ratchet_selfcheck(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed,
                                        n=n, max_level=max_level)}
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    bottom_np = np.ascontiguousarray(rules[depth - 1])
    ers = parse_eras(eras)
    out["GS_manufacture"] = GR.gate_manufacture(rules, inv, truth, canon_np, bottom_np,
                                                depth, v, m, s, ers, n=n, seed=rule_seed)
    out["GR_recert"] = GR.gate_recert(truth, v, s)
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    ear.remote(quick=quick)
