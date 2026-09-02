"""PHASE 0 — size THE TRAP offline, before any GPU.

`antiphon` measured the question port's ORDERING payoff (findings 1-2) and reported the
CORRUPTION payoff as small (finding 4): `q_novel`, the guard-less novelty selector, still mined
39 true L4 keys against `q_exo`'s 22. The SPEC's own reading of that: "on a constrained grammar
nearly every answer is worth something". This file asks what a worthless answer could BE on this
substrate, and sizes a trap that installs some — the `ostinato`/`phase0_question.py` discipline
(size the premise against the banked logs and the DGP's own arithmetic before a GPU is spent).

THE STRUCTURAL FACTS THAT CONSTRAIN ANY TRAP HERE (each is checked below, not assumed):

  (F1) THE ACTION SPACE IS WORLD-TYPED. `crystallize.units.apply_move` materialises a span by a
       max-sum DP over `rules_t` and renders it through `canon = rules[depth-1][:,0,:]`, so every
       move the agent can make writes a world-legal constituent. `macros.apply_any` does the same
       through an earned table. The agent therefore cannot produce a FOREIGN-grammar answer.
  (F2) MINING IS SUCCESS-FILTERED BY THE WORLD'S EXACT GRADER. `run_arm` block (b) mines
       `out["x"][ps > 0.5]` where `ps` is `units.grade(..., rules, s)`. An instance whose answer
       does not grade as a legal derivation of its root contributes NOTHING to the table.
  (F3) THEREFORE a foreign-grammar channel cannot mint keys here: its answers are never solved
       under the world's grader, so it degenerates into "burn priced budget, bank nothing".
       Grading it under a FOREIGN grader would put foreign-legal answers into the value buffer
       and the generator's fine-tune set — moving the plant, not the question distribution, which
       is the SPEC's one hard norm. Recorded as REJECTED, with these reasons.
  (F4) WHAT IS LEFT IS NON-CONVERTIBILITY. Every true level-L flat's two halves are true
       level-(L-1) flats (proved in [2] below, exactly, on the DGP's own tables). So a key whose
       half is NOT a true row is junk with certainty, and `Miner.build`'s ratchet — both halves
       must be rows of the operative lower table — refuses it forever. And the half the QUESTION
       fixes is the one the damage cell does not cover, which reaches the miner as the question
       set it (FILES.md section 1; checked against the logs in [7]).

  ==> THE TRAP ATOM: a menu candidate whose CLEAN HALF parses to a junk key. Its answer is
      worthless with certainty, at matched difficulty, drawn from the world's own cell, with no
      foreign machinery and no substrate surgery. The trap knob is the fraction `f` of the menu
      built from such candidates, presented spread uniformly over the junk-half pool so that they
      look maximally like news.

WHAT THIS FILE MEASURES.
  [0] geometry — where the trap atom exists at all (which eras have a clean half, and of what
      level), and the native distractor rate the world already serves.
  [1] the junk-half pools per era: reachable true vs junk halves at the CLEAN node, their
      frequencies, and how much of the menu they can carry.
  [2] the worthlessness proof: junk half => junk key, on the true tables, at every level.
  [3] d*-ORTHOGONALITY: the trap must not move difficulty. d* histograms of the distractor pool
      vs the native pool, per era, with the total-variation distance and a per-stratum quota
      feasibility check at each f.
  [4] construction cost: rejection rate, bank size per era, wall-clock per menu.
  [5] SELECTOR SIMULATION — the real `questions.py` selectors on real menus, with a mining model
      calibrated to the banked `an_s0` logs: what fraction of each arm's 64 are distractors, and
      what each banks, at f in {native, 0.7, 0.8, 0.9}. Includes the three candidate credit rules
      (novel / close / build) and a prospective ratchet guard.
  [6] closure arithmetic: is an UNLEARNABLE channel constructible here at matched difficulty?
      (pool size vs observation budget / support).
  [7] the banked logs: the clean-half delivery check (does the trap atom actually reach the
      miner), the incumbent credit rule's discrimination, and how leaky an endogenous ratchet
      guard would be (junk rows in the arms' own operative lower tables).
  [8] the two rejected channels, sized: the foreign grammar (F3) and the unpinned unlearnable
      arm's difficulty deviation.

No GPU, no Modal, no substrate. Run from experiments/:
    PYTHONPATH=. python3 rhm/practice/antiphon/trap/phase0_trap.py
"""

import collections
import json
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.crystallize.units import corrupt_hier
from rhm.practice.antiphon import questions as QS

HERE = os.path.dirname(os.path.abspath(__file__))
ANTIPHON = os.path.dirname(HERE)

V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
SUPPORT = 3
MINE_CAP = 8
N_PR = 64
K_MENU = 2048
MAXL = 4

# `crescendo`/`antiphon`'s ladder: era -> (damage level, damage node, cycles)
LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]

AN_S0 = os.path.join(ANTIPHON, "figures", "an_s0")
AN_ARMS = ["q_exo", "q_bisect", "q_endo", "q_novel", "q_comp", "q_comp_free"]

N_POOL = 400_000          # derivations sampled for the half-key marginals
N_DSTAR = 24_000          # instances per era for the d* orthogonality check
F_GRID = (0.15, 0.30, 0.70, 0.80, 0.90)


def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def tv(a, b):
    """Total-variation distance between two d* histograms given as Counters."""
    keys = set(a) | set(b)
    na, nb = max(sum(a.values()), 1), max(sum(b.values()), 1)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


# --------------------------------------------------------------------------- #
# [0]/[1]  geometry and the junk-half pools
# --------------------------------------------------------------------------- #

def era_geometry(rules, truth, ib, pf, n):
    """Per era: the mined span, the clean part, the LEVEL of the clean part, and the reachable
    true/junk half pools at the clean node under the DGP's own draw."""
    out = []
    for lvl, node, cycles in LADDER:
        g = QS.target_geometry(lvl, node, MAXL, S)
        row = {"era_level": lvl, "era_node": node, "cycles": cycles,
               "target_level": g["level"], "span": g.get("span"), "blk0": g.get("blk0"),
               "clean_offsets": g.get("clean_offsets", []),
               "has_clean": bool(g.get("has_clean")),
               "obs_budget": cycles * MINE_CAP}
        if not row["has_clean"]:
            row.update({"clean_level": None, "trap_exists": False,
                        "why": "the damage cell swallows the mined span; no clean half"})
            out.append(row)
            continue
        blocks = [g["blk0"] + o for o in g["clean_offsets"]]
        clean_level = lvl                      # the clean part IS a level-`lvl` constituent
        row["clean_blocks"] = blocks
        row["clean_level"] = int(clean_level)
        if clean_level < 2:
            row.update({"trap_exists": False,
                        "why": "the clean part is a single block = a level-1 feature; every "
                               "level-1 feature is a valid row, so no junk half exists"})
            out.append(row)
            continue
        keys = [tuple(int(x) for x in r) for r in pf[:, blocks]]
        cnt = collections.Counter(keys)
        tl = flats_of(truth[clean_level])
        p_true = {k: c / n for k, c in cnt.items() if k in tl}
        p_junk = {k: c / n for k, c in cnt.items() if k not in tl}
        jf = np.array(sorted(p_junk.values())) if p_junk else np.zeros(1)
        row.update({
            "trap_exists": True,
            "n_true_halves_reachable": len(p_true), "n_true_halves_total": len(tl),
            "n_junk_halves_reachable": len(p_junk),
            "half_space_size": V ** len(blocks),
            "native_distractor_rate": float(sum(p_junk.values())),
            "junk_half_freq": {"min": float(jf[0]), "med": float(np.median(jf)),
                               "max": float(jf[-1])},
            "junk_pool": sorted(p_junk, key=lambda k: -p_junk[k]),
            "true_pool": sorted(p_true, key=lambda k: -p_true[k]),
            "p_junk": {str(k): v for k, v in p_junk.items()},
            "p_true": {str(k): v for k, v in p_true.items()},
        })
        out.append(row)
    return out


def worthlessness_proof(truth):
    """[2] Every true level-L flat's halves are true level-(L-1) flats — so a junk half is a junk
    key, with certainty, and the ratchet refuses it forever. Exact, on the DGP's own tables."""
    rows = []
    for L in (3, 4, 5):
        fl, lo = flats_of(truth[L]), flats_of(truth[L - 1])
        half = len(next(iter(fl))) // S
        bad = sum(1 for k in fl if not (k[:half] in lo and k[half:] in lo))
        rows.append({"level": L, "n_flats": len(fl), "half_width": half,
                     "halves_not_true": int(bad), "ok": bad == 0})
    return rows


# --------------------------------------------------------------------------- #
# [3]  d*-orthogonality and quota feasibility
# --------------------------------------------------------------------------- #

def dstar_orthogonality(rules, truth, ib, n=N_DSTAR, seed=7):
    """The trap must not move difficulty. Draw native instances of each era's cell exactly as
    `context_instances` does, split them by whether the CLEAN half is junk, and compare the two
    d* histograms. Also: can a menu at fraction f still fill the native head's quota from the
    distractor pool, stratum by stratum?"""
    out = []
    for lvl, node, cycles in LADDER:
        g = QS.target_geometry(lvl, node, MAXL, S)
        if not g.get("has_clean") or lvl < 2:
            continue
        blocks = [g["blk0"] + o for o in g["clean_offsets"]]
        tl = flats_of(truth[lvl])
        rng = np.random.default_rng(seed + lvl)
        t0 = time.time()
        # `context_instances`'s own draw: uniform roots, DGP derivation, corrupt at the cell,
        # keep only the instances that actually broke (require_broken=True).
        r = rng.integers(0, V, size=n)
        lv = sample_derivations(rules, r, S, rng)
        xd = corrupt_hier(lv, rules, DEPTH, V, M, S, lvl, [node], rng)
        d = nearest_derivation_cost(rules, xd, r, S)
        keep = d > 0
        secs = time.time() - t0
        pf = MC.exact_features(lv[keep], ib, V, S)
        dk = d[keep]
        junk = np.array([tuple(int(x) for x in row) not in tl for row in pf[:, blocks]])
        hn = collections.Counter(int(x) for x in dk[~junk])
        hj = collections.Counter(int(x) for x in dk[junk])
        # quota feasibility: with the native head's quota (a 64-draw of the native mix) and a
        # menu that is fraction f distractors, is every stratum still fillable from the menu?
        feas = {}
        for f in F_GRID:
            n_d = int(round(f * K_MENU))
            n_n = K_MENU - n_d
            # expected count per stratum in the enriched menu
            av = {t: n_d * hj.get(t, 0) / max(sum(hj.values()), 1)
                     + n_n * hn.get(t, 0) / max(sum(hn.values()), 1)
                  for t in set(hn) | set(hj)}
            need = {t: N_PR * (hj.get(t, 0) * f / max(sum(hj.values()), 1)
                               + hn.get(t, 0) * (1 - f) / max(sum(hn.values()), 1))
                    for t in av}
            feas[str(f)] = {"min_avail_over_need": float(min(
                (av[t] / max(need[t], 1e-9)) for t in av if need[t] > 0.5))}
        out.append({
            "era_level": lvl, "era_node": node, "n_drawn": int(n), "n_broken": int(keep.sum()),
            "seconds_for_n": secs,
            "native_d_hist": {str(k): int(v) for k, v in sorted(hn.items())},
            "distractor_d_hist": {str(k): int(v) for k, v in sorted(hj.items())},
            "d_mean_native": float(dk[~junk].mean()), "d_mean_distractor": float(dk[junk].mean()),
            "tv_distance": float(tv(hn, hj)),
            "distractor_frac_native": float(junk.mean()),
            "quota_feasibility": feas,
        })
    return out


# --------------------------------------------------------------------------- #
# [5]  the selector simulation, with an EMPIRICALLY CALIBRATED delivery model
# --------------------------------------------------------------------------- #

def delivery_model(truth, era_rows, arm="q_exo"):
    """The delivered key, learned from the banked logs rather than assumed.

    [7a] establishes the trap atom's precondition exactly: every at-support key's CLEAN half is a
    legal parse of the clean node (32/32, 31/31, 57/57 across arms), i.e. the question's half
    reaches the miner untouched. What varies is the REPAIRED half, and the logs say it comes from
    a narrow repertoire (63-70 distinct values over an era, the top one carrying 42-57% of the
    mass). So the model is: delivered key = (repaired half drawn from the null arm's own
    empirical distribution, conditioned on whether the clean half is a true row) ++ (the clean
    half, exactly as posed).

    The null arm's distribution is used for EVERY arm, so no guarded arm is handed a
    repaired-half advantage it would have to earn on the substrate.
    """
    out = {}
    p = os.path.join(AN_S0, arm, "results.json")
    if not os.path.isfile(p):
        return out
    with open(p) as fh:
        res = json.load(fh)
    mc = res.get("miner_counts") or {}
    for r in era_rows:
        if not r.get("trap_exists"):
            continue
        tgt = str(r["target_level"])
        if tgt not in mc:
            continue
        half = (S ** (r["target_level"] - 1)) // S
        tl = flats_of(truth[r["clean_level"]])
        pool = {tuple(k) for k in r["junk_pool"]} | {tuple(k) for k in r["true_pool"]}
        by = {True: collections.Counter(), False: collections.Counter()}
        for k, c in mc[tgt]:
            k = tuple(int(z) for z in k)
            ch = k[half:]
            if ch not in pool:                 # a different era's mining node; not this stream
                continue
            by[ch in tl][k[:half]] += c
        out[r["era_level"]] = {}
        for cls, cnt in by.items():
            items = cnt.most_common()
            out[r["era_level"]][bool(cls)] = (
                [h for h, _ in items],
                np.array([c for _, c in items], float) / max(sum(cnt.values()), 1))
    return out


def _sharpen(deliv_era, tau):
    """One free parameter, fitted per era against the null arm's own banked stream.

    `miner_counts` is a FINAL count, so the repaired-half distribution read off it pools the
    era that earns the level with the consumption eras that keep mining the same node — which
    adds rare values the earning era never produced. `tau` sharpens the empirical distribution
    back; it is fitted, reported, and used unchanged across every f and every selector."""
    out = {}
    for cls, (hs, p) in deliv_era.items():
        q = np.power(np.asarray(p, float), float(tau))
        out[cls] = (hs, q / q.sum())
    return out


def fit_tau(era_row, dstar_row, truth, deliv_era, lower_op, target, taus=None):
    """Fit `tau` so the model reproduces the null arm's banked (n_distinct, at_support) at the
    native f. Reports the residual so the fit's quality travels with the table."""
    taus = taus or [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 4.0]
    f = round(era_row["native_distractor_rate"], 3)
    best = None
    for t in taus:
        s = _simulate(era_row, dstar_row, truth, _sharpen(deliv_era, t), lower_op, f, "exo")
        err = (abs(s["n_distinct"] - target["n_distinct"]) / max(target["n_distinct"], 1)
               + abs(s["at_support"] - target["at_support"]) / max(target["at_support"], 1))
        if best is None or err < best[1]:
            best = (t, err, s)
    return {"tau": best[0], "err": best[1], "model": best[2], "target": target}


def _lower_op(era_row, truth, banked, arm="q_exo", rng=None):
    """The arm's OWN operative lower table at this era, modelled from the banked commit record
    (`tab_n_learned` / `tab_n_correct` at the level the question's clean half lives on). This is
    what `pose_questions` hands the oracle as `lower_flat` and what a prospective ratchet guard
    would read — and it is only as precise as the arm made it."""
    rng = rng or np.random.default_rng(0)
    lvl = era_row["clean_level"]
    rec = banked.get("commits", {}).get((arm, lvl))
    n_learned, n_true = (rec["n_learned"], rec["n_correct"]) if rec else (8, 3)
    tp, jp = era_row["true_pool"], era_row["junk_pool"]
    pt = np.array([era_row["p_true"][str(k)] for k in tp]); pt = pt / pt.sum()
    pj = np.array([era_row["p_junk"][str(k)] for k in jp]); pj = pj / pj.sum()
    rows = set()
    for i in rng.choice(len(tp), size=min(n_true, len(tp)), replace=False, p=pt):
        rows.add(tuple(tp[i]))
    nj = min(max(n_learned - n_true, 0), len(jp))
    for i in rng.choice(len(jp), size=nj, replace=False, p=pj):
        rows.add(tuple(jp[i]))
    return rows, {"n_learned": n_learned, "n_true": n_true, "precision": n_true / max(n_learned, 1)}


def _menu(era_row, dstar_row, f, rng, spread="uniform"):
    """One cycle's menu: K candidates, a fraction f of whose CLEAN HALVES are junk, presented
    (by default) UNIFORMLY over the reachable junk-half pool so the trap looks maximally like
    news. Returns (clean halves, is_distractor, d*)."""
    n_d = int(round(f * K_MENU))
    jp, tp = era_row["junk_pool"], era_row["true_pool"]
    pj = np.array([era_row["p_junk"][str(k)] for k in jp]); pj = pj / pj.sum()
    pt = np.array([era_row["p_true"][str(k)] for k in tp]); pt = pt / pt.sum()
    di = (rng.integers(0, len(jp), size=n_d) if spread == "uniform"
          else rng.choice(len(jp), size=n_d, p=pj))
    ni = rng.choice(len(tp), size=K_MENU - n_d, p=pt)
    halves = [tuple(jp[i]) for i in di] + [tuple(tp[i]) for i in ni]
    is_d = np.array([True] * n_d + [False] * (K_MENU - n_d))
    hn = {int(k): v for k, v in dstar_row["native_d_hist"].items()}
    hj = {int(k): v for k, v in dstar_row["distractor_d_hist"].items()}

    def draw(h, m):
        ks = np.array(sorted(h)); p = np.array([h[k] for k in ks], float); p /= p.sum()
        return rng.choice(ks, size=m, p=p)
    d = np.concatenate([draw(hj, n_d), draw(hn, K_MENU - n_d)])
    perm = rng.permutation(K_MENU)
    return [halves[i] for i in perm], is_d[perm], d[perm]


def _simulate(era_row, dstar_row, truth, deliv, lower_op, f, mode, seed=0, spread="uniform",
              deep=None):
    """One era of mining, offline, driven by the REAL selectors in `questions.py`."""
    rng = np.random.default_rng(1000 + seed)
    lvl, tgt = era_row["era_level"], era_row["target_level"]
    tl_clean = flats_of(truth[era_row["clean_level"]])
    truth_tgt = flats_of(truth[tgt])
    counts = collections.Counter()
    banked_keys = set()
    ledger = QS.DeliveryLedger()
    took_d = mined_d = 0
    n_obs = 0
    for cyc in range(era_row["cycles"]):
        halves, is_d, d = _menu(era_row, dstar_row, f, rng, spread=spread)
        quota = QS.quota_of(d, N_PR)
        cov = collections.Counter()
        for k in banked_keys:
            cov[k[len(k) // 2:]] += 1
        if mode == "exo":
            sel = np.arange(N_PR)
        elif mode in ("ratchet", "bisect"):
            ref = lower_op if mode == "ratchet" else tl_clean
            floor = 0.05 if mode == "ratchet" else 0.0   # the oracle never prefers junk
            prio = np.array([(1.0 if h in ref else floor) / (1.0 + cov.get(h, 0))
                             for h in halves])
            sel = QS._round_robin(prio, halves, d, quota, N_PR)
        elif mode in ("ratchet_l2", "trust"):
            # THE GUARD AT THE RIGHT DEPTH: score the clean half by its own two level-2
            # constituents against the arm's own committed L2 table — hard membership
            # (`ratchet_l2`) or weighted by the beam's own per-entry use record (`trust`).
            sc = deep["score"]
            prio = np.array([(sc.get(h, 0.0) + 1e-4) / (1.0 + cov.get(h, 0)) for h in halves])
            sel = QS._round_robin(prio, halves, d, quota, N_PR)
        else:
            agent = {"halves": halves, "covered": dict(cov), "value": rng.random(K_MENU),
                     "ledger": ledger}
            sel = QS.select(mode, N_PR, d=d, quota=quota, agent=agent)
        took_d += int(is_d[sel].sum())
        mi = rng.permutation(len(sel))[:MINE_CAP]      # the mining subsample, class-blind
        posed, credits = [], []
        for i in np.asarray(sel)[mi]:
            h = halves[i]
            reps, p = deliv[bool(h in tl_clean)]
            other = tuple(reps[rng.choice(len(reps), p=p)])
            key = other + h
            credits.append(1 if counts[key] < SUPPORT else 0)
            counts[key] += 1
            posed.append(h)
            n_obs += 1
            if counts[key] >= SUPPORT:
                banked_keys.add(key)
            if is_d[i]:
                mined_d += 1
        ledger.update(posed, credits)
    true_at = sum(1 for k in banked_keys if k in truth_tgt)
    wj = [w for h, w in ledger.w.items() if h not in tl_clean]
    wt = [w for h, w in ledger.w.items() if h in tl_clean]
    return {"ledger_w_junk": float(np.mean(wj)) if wj else None,
            "ledger_w_true": float(np.mean(wt)) if wt else None,
            "era_level": lvl, "target_level": tgt, "mode": mode, "f": float(f), "spread": spread,
            "distractor_take_rate": took_d / (era_row["cycles"] * N_PR),
            "distractor_mined_rate": mined_d / (era_row["cycles"] * MINE_CAP),
            "n_obs": n_obs, "n_distinct": len(counts), "at_support": len(banked_keys),
            "true_at_support": true_at, "junk_at_support": len(banked_keys) - true_at,
            "recall": true_at / len(truth_tgt)}


# --------------------------------------------------------------------------- #
# [9]  the USE RECORD as a guard — does the beam's own per-entry selection history
#      separate a true row from a junk row where the table's precision does not?
# --------------------------------------------------------------------------- #

def use_record_read(truth, arms=None):
    """`log["entry"]["hist"]["beam"][level]` is the priced beam's per-entry selection count, and
    `log["entry"]["true_mask"][level]` marks which of those entries are true rows. Both are
    already logged in every `an_s0` arm (the `assay` instrument `antiphon` section 8 reads pi's
    beam share off), so this costs nothing.

    `census` found the executor's max-sum DP is a junk filter — a 6.2%-true table gives 71-76%
    true executions. If that holds here, the arm's OWN use record discriminates better than its
    own table's precision, and it is the only endogenous signal that does.
    """
    out = []
    for arm in (arms or AN_ARMS):
        p = os.path.join(AN_S0, arm, "results.json")
        if not os.path.isfile(p):
            continue
        with open(p) as fh:
            res = json.load(fh)
        log = res["log"]
        row = {"arm": arm, "levels": {}}
        for lvl in ("2", "3"):
            tot, mask = None, None
            for e in log["entry"]:
                if not e:
                    continue
                h = (e.get("hist") or {}).get("beam") or {}
                if lvl not in h:
                    continue
                v = np.asarray(h[lvl], float)
                if tot is None or len(tot) != len(v):
                    tot = np.zeros(len(v))
                tot += v
                mask = np.asarray(e["true_mask"][lvl])
            if tot is None or mask is None or tot.sum() == 0:
                continue
            tp = float(mask.mean())
            up = float((tot * mask).sum() / tot.sum())
            t, j = tot[mask == 1], tot[mask == 0]
            auc = (float(np.mean([(a > b) + 0.5 * (a == b) for a in t for b in j]))
                   if len(t) and len(j) else None)
            row["levels"][lvl] = {"n_rows": int(len(mask)), "n_true": int(mask.sum()),
                                  "table_precision": tp, "use_weighted_precision": up,
                                  "lift": up / max(tp, 1e-9), "auc_rowwise": auc,
                                  "use": [float(x) for x in tot],
                                  "true_mask": [int(x) for x in mask]}
        out.append(row)
    return out


def committed_l2(arm, truth):
    """Reconstruct the arm's OWN committed level-2 table exactly. `Miner.build` keeps the
    at-support keys, in SORTED key order, whose halves are rows of the lower table — and at
    level 2 the lower table is `base_table(v)`, which contains every level-1 feature, so the
    committed rows are exactly the sorted at-support L2 keys at the commit cycle. Validated
    against the commit event's own `tab_n_learned` / `tab_n_correct`."""
    p = os.path.join(AN_S0, arm, "results.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    log = res["log"]
    ev = next((e for e in res["events"]
               if e["kind"] == "commit" and int(e["level"]) == 2), None)
    if ev is None:
        return None
    i = log["cycle"].index(ev["cycle"])
    ks = sorted(tuple(int(z) for z in k)
                for k in (log["miner"][i]["2"] or {}).get("keys_at_support", []))
    tl = flats_of(truth[2])
    n_true = sum(1 for k in ks if k in tl)
    swaps = sum(1 for e in res["events"] if e["kind"] == "recert" and e.get("swapped"))
    return {"rows": ks, "n_learned": len(ks), "n_true": n_true,
            "logged_n_learned": int(ev["tab_n_learned"]),
            "logged_n_correct": int(ev["tab_n_correct"]),
            "reconstruction_ok": bool(len(ks) == int(ev["tab_n_learned"])
                                      and n_true == int(ev["tab_n_correct"])),
            "recert_swaps": int(swaps), "commit_cycle": int(ev["cycle"])}


def deep_guard_read(era_row, truth, use_rows, arm="q_exo"):
    """THE GUARD AT THE RIGHT DEPTH. A clean half at era 3 is a level-3 key; its own two halves
    are level-2 keys, and THAT is the level where the arm has both a table and a use record.

    Three forms, plus the ceiling:
      unguarded  — breadth-first novelty; junk share = the pool ratio.
      ratchet-L2 — hard membership: both level-2 halves are rows of the arm's own committed L2
                   table (the r-squared ratchet, one level below the clean half).
      trust-L2   — the same, weighted by the product of the two rows' BEAM USE shares.
      ceiling    — the same hard test against the TRUE level-2 table.
    """
    tl2, tl3 = flats_of(truth[2]), flats_of(truth[3])
    junk = [tuple(k) for k in era_row["junk_pool"]]
    true = [tuple(k) for k in era_row["true_pool"]]
    tab = committed_l2(arm, truth)
    if tab is None:
        return None
    idx = {k: i for i, k in enumerate(tab["rows"])}
    ur = next((r for r in use_rows if r["arm"] == arm), None)
    w = None
    if ur and "2" in ur["levels"]:
        u = np.asarray(ur["levels"]["2"]["use"], float)
        if len(u) == len(tab["rows"]):
            w = u / max(u.sum(), 1e-9)
    adm = lambda k: (k[:2] in idx and k[2:] in idx)
    aj, at = [k for k in junk if adm(k)], [k for k in true if adm(k)]
    sc = (lambda k: float(w[idx[k[:2]]] * w[idx[k[2:]]])) if w is not None else (lambda k: 1.0)
    sj, st = sum(sc(k) for k in aj), sum(sc(k) for k in at)
    cj = [k for k in junk if k[:2] in tl2 and k[2:] in tl2]
    return {
        "arm": arm, "committed_L2": {k: v for k, v in tab.items() if k != "rows"},
        "pool_junk": len(junk), "pool_true": len(true),
        "unguarded_junk_share": len(junk) / (len(junk) + len(true)),
        "ratchet_l2_admit_junk": len(aj), "ratchet_l2_admit_true": len(at),
        "ratchet_l2_junk_share": len(aj) / max(len(aj) + len(at), 1),
        "trust_l2_junk_share": sj / max(sj + st, 1e-12),
        "ceiling_admit_junk": len(cj),
        "ceiling_junk_share": len(cj) / max(len(cj) + len(true), 1),
        "_admit": {"junk": aj, "true": at}, "_score": {k: sc(k) for k in aj + at},
    }


# --------------------------------------------------------------------------- #
# [7]  the banked logs
# --------------------------------------------------------------------------- #

def banked_read(era_rows, truth):
    """Everything this design needs from `an_s0`, read offline.

    (a) THE TRAP ATOM'S PRECONDITION. If the question's clean half reaches the miner untouched,
        then every at-support key's clean half must be a legal parse of the CLEAN node — a set of
        140 tuples at era 3, out of 8**4 = 4096 possible. That is a sharp test, and it also gives
        `delivered_true <= P(clean half is a true row)` with near-equality on the arm that
        selects nothing.
    (b) the incumbent credit rule's discrimination (`credit_rate` per era).
    (c) the arms' own tables: at-support composition per level, and the COMMITTED table's
        precision, which is the ceiling on any endogenous ratchet guard.
    """
    out = {"arms": [], "available": os.path.isdir(AN_S0), "commits": {}}
    if not out["available"]:
        return out
    by_era = {r["era_level"]: r for r in era_rows}
    for arm in AN_ARMS:
        p = os.path.join(AN_S0, arm, "results.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            res = json.load(f)
        log = res["log"]
        rows = [x for x in log["q"] if x]
        per = {}
        for x in rows:
            per.setdefault(x["target_level"], []).append(x)
        arm_row = {"arm": arm, "levels": {}}
        for tl, g in sorted(per.items()):
            era = tl - 1
            er = by_era.get(era)
            pred = (1.0 - er["native_distractor_rate"]) if (er and er.get("trap_exists")) else None

            def mean(k):
                vv = [x[k] for x in g if x.get(k) is not None]
                return float(np.mean(vv)) if vv else None
            arm_row["levels"][str(tl)] = {
                "n_cycles": len(g), "credit_rate": mean("credit_rate"),
                "dose_hit": mean("dose_hit"), "delivered_true": mean("delivered_true"),
                "designed_true_mined": mean("designed_true_mined"),
                "n_distinct_halves": mean("n_distinct_halves"),
                "clean_half_true_rate_menu": pred,
                "delivered_true_le_clean_true": (
                    None if pred is None else bool(mean("delivered_true") <= pred + 0.02)),
            }
        # (a) end-of-era mining state, and the clean-half containment test
        arm_row["end_of_era"] = {}
        for er in era_rows:
            if not er.get("trap_exists"):
                continue
            idx = [i for i, e in enumerate(log["era"]) if e == er["era_level"]]
            if not idx:
                continue
            st = (log["miner"][idx[-1]] or {}).get(str(er["target_level"])) or {}
            if not st:
                continue
            ks = [tuple(int(z) for z in k) for k in (st.get("keys_at_support") or [])]
            half = (S ** (er["target_level"] - 1)) // S
            pool = {tuple(k) for k in er["junk_pool"]} | {tuple(k) for k in er["true_pool"]}
            tl_clean = flats_of(truth[er["clean_level"]])
            tgt = flats_of(truth[er["target_level"]])
            clean = [k[half:] for k in ks]
            rep = [k[:half] for k in ks]
            arm_row["end_of_era"][f"era{er['era_level']}"] = {
                "cycle": int(log["cycle"][idx[-1]]),
                "n_obs": int(st.get("n_obs", 0)), "n_distinct": int(st.get("n_distinct", 0)),
                "at_support": len(ks),
                "true_at_support": sum(1 for k in ks if k in tgt),
                "junk_at_support": sum(1 for k in ks if k not in tgt),
                "clean_halves_distinct": len(set(clean)),
                "clean_halves_in_pool": sum(1 for h in set(clean) if h in pool),
                "clean_halves_true": sum(1 for h in set(clean) if h in tl_clean),
                "clean_half_containment": (
                    all(h in pool for h in set(clean)) if clean else None),
                "repaired_halves_distinct": len(set(rep)),
                "repaired_halves_true": sum(1 for h in set(rep) if h in tl_clean),
            }
        mc = res.get("miner_counts") or {}
        tabs = {}
        for lv, items in mc.items():
            ks = {tuple(int(z) for z in k) for k, c in items if c >= SUPPORT}
            tl_ = flats_of(truth[int(lv)])
            tabs[lv] = {"n_at_support": len(ks), "n_true": len(ks & tl_),
                        "n_junk": len(ks - tl_),
                        "junk_keys": sorted(ks - tl_)}
        arm_row["miner_at_support"] = {k: {kk: vv for kk, vv in v.items() if kk != "junk_keys"}
                                       for k, v in tabs.items()}
        arm_row["_junk_at_sup"] = {k: v["junk_keys"] for k, v in tabs.items()}
        cg = [x for x in log["committed_grade"] if x]
        arm_row["committed_grade_final"] = cg[-1] if cg else None
        for e in res["events"]:
            if e["kind"] == "commit" and e.get("tab_n_learned") is not None:
                out["commits"][(arm, int(e["level"]))] = {
                    "cycle": int(e["cycle"]), "n_learned": int(e["tab_n_learned"]),
                    "n_correct": int(e["tab_n_correct"]),
                    "precision": float(e["tab_precision"])}
        out["arms"].append(arm_row)
    return out


def guard_leak(banked, era_rows, truth):
    """The ceiling on an endogenous RATCHET guard. A prospective guard prefers a candidate whose
    clean half is a row of the arm's OWN operative lower table — the committed table at that
    level. It therefore inherits that table's precision exactly: a junk row the arm committed is
    a junk half the guard will actively prefer. Sized two ways: the committed table's own
    precision, and the overlap between the pool the guard's junk rows are drawn from (the level
    mined one era earlier, at a different node) and the pool the trap draws from."""
    out = {"eras": {}}
    for er in era_rows:
        if not er.get("trap_exists"):
            continue
        lvl = er["clean_level"]
        trap_pool = {tuple(k) for k in er["junk_pool"]}
        rows = []
        for a in banked.get("arms", []):
            rec = banked["commits"].get((a["arm"], lvl))
            junk_sup = {tuple(k) for k in a.get("_junk_at_sup", {}).get(str(lvl), [])}
            ov = junk_sup & trap_pool
            rows.append({
                "arm": a["arm"],
                "committed_n_learned": rec["n_learned"] if rec else None,
                "committed_n_true": rec["n_correct"] if rec else None,
                "committed_precision": rec["precision"] if rec else None,
                "junk_at_support": len(junk_sup),
                "junk_at_support_in_trap_pool": len(ov),
                "pool_overlap_frac": (len(ov) / len(junk_sup)) if junk_sup else None,
            })
        out["eras"][f"era{er['era_level']}"] = {
            "clean_level": lvl, "trap_pool_size": len(trap_pool),
            "true_pool_size": len(er["true_pool"]), "arms": rows}
    return out


# --------------------------------------------------------------------------- #
def main():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    ib = build_inverse_maps(rules)[-1]

    print("=" * 88)
    print("PHASE 0 — THE TRAP, sized offline (antiphon/trap)")
    print("=" * 88)

    print("\n[2] WORTHLESSNESS PROOF — halves of a true level-L flat are true level-(L-1) flats")
    proof = worthlessness_proof(truth)
    for r in proof:
        print(f"     L{r['level']}: {r['n_flats']:>7d} true flats, half width {r['half_width']}"
              f"  ->  halves that are NOT true rows: {r['halves_not_true']}   "
              f"{'OK' if r['ok'] else 'FAIL'}")
    print("     => a junk half is a junk key with CERTAINTY, and `Miner.build`'s ratchet (both")
    print("        halves must be rows of the operative lower table) refuses it forever.")

    print(f"\n[0]/[1] GEOMETRY and the JUNK-HALF POOLS  (n = {N_POOL:,} DGP draws)")
    rng = np.random.default_rng(4242)
    roots = rng.integers(0, V, size=N_POOL)
    lv = sample_derivations(rules, roots, S, rng)
    pf = MC.exact_features(lv, ib, V, S)
    eras = era_geometry(rules, truth, ib, pf, N_POOL)
    print(f"     {'era':<8} {'tgt':>3} {'clean blocks':<16} {'clean lvl':>9} {'true halves':>12} "
          f"{'junk halves':>12} {'native f':>9} {'budget':>7}")
    for r in eras:
        cb = ",".join(str(b) for b in r.get("clean_blocks", [])) or "-"
        if r.get("trap_exists"):
            print(f"     L{r['era_level']}n{r['era_node']:<5} {r['target_level']:>3} {cb:<16} "
                  f"{r['clean_level']:>9} "
                  f"{r['n_true_halves_reachable']:>5}/{r['n_true_halves_total']:<6} "
                  f"{r['n_junk_halves_reachable']:>12} "
                  f"{r['native_distractor_rate']:>9.3f} {r['obs_budget']:>7}")
        else:
            print(f"     L{r['era_level']}n{r['era_node']:<5} "
                  f"{(r['target_level'] or 0):>3} {cb:<16} {'-':>9} {'-':>12} {'-':>12} "
                  f"{'-':>9} {r['obs_budget']:>7}   [{r['why']}]")
    live = [r for r in eras if r.get("trap_exists")]
    print(f"     => the trap atom exists in {len(live)} of {len(eras)} eras: "
          + ", ".join(f"L{r['era_level']}n{r['era_node']}" for r in live))
    print("\n[1b] THE BREADTH-FIRST TAKE SHARE — exact, no mining model needed.")
    print("     `_round_robin` takes one candidate per DISTINCT key in priority order before it")
    print("     takes a second, and every uncovered half ties at novelty 1.0. So a novelty")
    print("     selector's distractor share is set by the POOL RATIO, not by the menu's mass:")
    print(f"       {'era':<8} {'junk halves':>12} {'true halves':>12} {'pool share':>11} "
          f"{'over-take vs the null at native f':>34}")
    for r in live:
        j, t = r["n_junk_halves_reachable"], r["n_true_halves_reachable"]
        share = j / (j + t)
        r["pool_share"] = float(share)
        print(f"       L{r['era_level']}n{r['era_node']:<5} {j:>12} {t:>12} {share:>11.3f} "
              f"{share / r['native_distractor_rate']:>34.2f}x")
    print("     => a breadth-first novelty judge OVER-takes the trap while f < the pool share")
    print("        and UNDER-takes it above; the nerdsnipe it can produce is bounded by that")
    print("        ratio, and the ratio is a property of the grammar's ambiguity at the node.")
    print("     The knob's useful direction is therefore BOTH ways. Predicted over-take")
    print("     (novelty judge's junk intake / the null's) across the menu's worthless rate f:")
    print(f"       {'era':<8} " + " ".join(f"{('f=' + str(f)):>9}" for f in
                                           (0.15, 0.30, 0.546, 0.70, 0.90)))
    for r in live:
        sh = r["pool_share"]
        print(f"       L{r['era_level']}n{r['era_node']:<5} " + " ".join(
            f"{sh / f:>9.2f}" for f in (0.15, 0.30, 0.546, 0.70, 0.90)))
    print("     (a ratio above 1 is a judge doing WORSE than not selecting at all — the")
    print("      nerdsnipe; below 1 is breadth capping its own junk intake.)")

    print("\n[6] CLOSURE ARITHMETIC — is an UNLEARNABLE channel constructible at matched d*?")
    print(f"     {'era':<8} {'junk-half pool':>15} {'budget/support':>15} {'closes?':>9}")
    for r in live:
        pool = r["n_junk_halves_reachable"]
        cap = r["obs_budget"] // SUPPORT
        r["closure"] = {"pool": pool, "budget_over_support": cap, "closes": pool <= cap}
        print(f"     L{r['era_level']}n{r['era_node']:<5} {pool:>15} {cap:>15} "
              f"{str(pool <= cap):>9}")
    print("     A channel is 'unlearnable' only if its keys never recur, i.e. if its pool is")
    print("     larger than the era's observation budget divided by mine_support. On this")
    print("     grammar the reachable junk-half pool is SMALLER, so every junk half closes —")
    print("     the noisy-TV guard ('does disagreement close when I collect there') has no work")
    print("     to do at matched difficulty. The unlearnable pole needs the difficulty pin off.")

    print(f"\n[3] d*-ORTHOGONALITY (n = {N_DSTAR:,} per era)")
    dstar = dstar_orthogonality(rules, truth, ib)
    print(f"     {'era':<8} {'native d*':>10} {'distractor d*':>14} {'TV':>7} "
          f"{'native f':>9} {'min avail/need @f=0.9':>22} {'draw s':>7}")
    for r in dstar:
        print(f"     L{r['era_level']}n{r['era_node']:<5} {r['d_mean_native']:>10.3f} "
              f"{r['d_mean_distractor']:>14.3f} {r['tv_distance']:>7.3f} "
              f"{r['distractor_frac_native']:>9.3f} "
              f"{r['quota_feasibility']['0.9']['min_avail_over_need']:>22.1f} "
              f"{r['seconds_for_n']:>7.2f}")
    print("     TV near 0 => the trap does not move difficulty: the d* quota (the native head's")
    print("     own histogram) stays fillable and a distractor cannot redefine 'difficulty'.")

    print("\n[4] CONSTRUCTION COST — the per-era distractor bank")
    cost = []
    for r, dr in zip(live, [d for d in dstar if any(
            d["era_level"] == x["era_level"] for x in live)]):
        need = int(max(F_GRID) * K_MENU) * r["cycles"]
        rate = dr["distractor_frac_native"]
        draws = int(need / max(rate, 1e-6))
        secs = draws / (dr["n_drawn"] / dr["seconds_for_n"])
        cost.append({"era_level": r["era_level"], "slots_needed_at_f0.9": need,
                     "hit_rate": rate, "native_draws": draws, "seconds": secs,
                     "bank_MB": draws * (S ** (DEPTH - 1)) * S / 1e6})
        print(f"     L{r['era_level']}n{r['era_node']}: {need:>8,} distractor slots at f=0.9, "
              f"hit rate {rate:.3f} -> {draws:>9,} native draws ~ {secs:6.1f} s "
              f"({draws * 64 / 1e6:.0f} MB as int8)")
    print("     Drawn ONCE per era at era start (a bank keyed by clean half, so the trap can be")
    print("     presented uniform over the junk-half pool); ~0 marginal cost per cycle against")
    print("     an ~10.5 s cycle.")

    print("\n[7] THE BANKED LOGS (an_s0) — read BEFORE the simulation, which they calibrate")
    banked = banked_read(eras, truth)
    leak = {}
    if banked["available"]:
        print("     (a) THE TRAP ATOM'S PRECONDITION — does the question's clean half reach the")
        print("         miner untouched? Every at-support key's clean half must be a legal parse")
        print("         of the CLEAN node (a set of 140 tuples of 4096 possible, at era 3).")
        print(f"       {'arm':<12} {'era':>4} {'@sup':>5} {'true':>5} {'clean halves':>13} "
              f"{'in pool':>8} {'contained':>10} {'repaired distinct':>18}")
        for a_ in banked["arms"]:
            for ek, x in sorted(a_["end_of_era"].items()):
                print(f"       {a_['arm']:<12} {ek[-1]:>4} {x['at_support']:>5} "
                      f"{x['true_at_support']:>5} {x['clean_halves_distinct']:>13} "
                      f"{x['clean_halves_in_pool']:>8} {str(x['clean_half_containment']):>10} "
                      f"{x['repaired_halves_distinct']:>18}")
        print("     (b) delivered_true vs the menu's clean-half true rate, and the INCUMBENT")
        print("         credit rule (`the delivered key is not yet at support`)")
        print("         (the bound binds only on the arm that SELECTS NOTHING; a selecting arm")
        print("          lifts its own clean-true rate above the menu's, which is the point)")
        print(f"       {'arm':<12} {'lvl':>3} {'delivered_true':>15} {'menu clean-true':>16} "
              f"{'<=?':>5} {'credit_rate':>12} {'dose_hit':>9}")
        for a_ in banked["arms"]:
            for tl, x in sorted(a_["levels"].items()):
                ct = x["clean_half_true_rate_menu"]
                print(f"       {a_['arm']:<12} {tl:>3} {x['delivered_true']:>15.3f} "
                      f"{(f'{ct:.3f}' if ct is not None else '-'):>16} "
                      f"{str(x['delivered_true_le_clean_true']):>5} "
                      f"{x['credit_rate']:>12.3f} {x['dose_hit']:>9.3f}")
        leak = guard_leak(banked, eras, truth)
        print("\n     (c) THE CEILING ON AN ENDOGENOUS RATCHET GUARD — a prospective guard that")
        print("         prefers candidates whose clean half is a row of the arm's OWN operative")
        print("         lower table inherits that table's precision exactly.")
        for ek, e in sorted(leak["eras"].items()):
            print(f"       {ek} (clean level L{e['clean_level']}; trap pool "
                  f"{e['trap_pool_size']} junk halves vs {e['true_pool_size']} true)")
            print(f"         {'arm':<12} {'committed t/n':>14} {'precision':>10} "
                  f"{'junk@sup':>9} {'in trap pool':>13} {'overlap':>8}")
            for r in e["arms"]:
                pr = r["committed_precision"]
                tab = f"{r['committed_n_true']}/{r['committed_n_learned']}"
                ovf = ("-" if not r["pool_overlap_frac"] else f"{r['pool_overlap_frac']:.3f}")
                print(f"         {r['arm']:<12} {tab:>14} "
                      f"{(f'{pr:.3f}' if pr is not None else '-'):>10} "
                      f"{r['junk_at_support']:>9} {r['junk_at_support_in_trap_pool']:>13} "
                      f"{ovf:>8}")
    else:
        banked = {"available": False, "arms": [], "commits": {}}
        print("     an_s0 not fetched under figures/; the simulation falls back to defaults.")

    print("\n[9] THE USE RECORD AS A GUARD — `log['entry']['hist']['beam']` vs `true_mask`")
    use_rows = use_record_read(truth) if banked["available"] else []
    print(f"       {'arm':<12} {'lvl':>3} {'rows':>5} {'true':>5} {'table prec':>11} "
          f"{'use-wt prec':>12} {'lift':>6} {'row-wise AUC':>13}")
    for u in use_rows:
        for lvl, x in sorted(u["levels"].items()):
            au = "-" if x["auc_rowwise"] is None else f"{x['auc_rowwise']:.3f}"
            print(f"       {u['arm']:<12} {lvl:>3} {x['n_rows']:>5} {x['n_true']:>5} "
                  f"{x['table_precision']:>11.3f} {x['use_weighted_precision']:>12.3f} "
                  f"{x['lift']:>6.2f} {au:>13}")
    print("     The lift is at L3, where the table is imprecise (0.25-0.50): the beam's own")
    print("     selection mass sits on TRUE rows at 0.73-0.93. But the row-wise AUC is thin —")
    print("     the mass concentrates on ONE dominant true row rather than rank-ordering the")
    print("     rest — so the record works as a WEIGHT, not as a membership test.")

    print("\n[9b] THE GUARD AT THE RIGHT DEPTH. A clean half at era 3 is a level-3 key; its own")
    print("     two halves are level-2 keys, and that is the level where the arm has both a")
    print("     committed table and a use record. Junk share of the pool a novelty judge draws")
    print("     from, under each guard (lower is better; the pool ratio is the unguarded value):")
    e3 = next((r for r in live if r["era_level"] == 3), None)
    deeps = {}
    if e3 is not None and banked["available"]:
        print(f"       {'arm':<12} {'own L2 table':>13} {'recon ok':>9} {'unguarded':>10} "
              f"{'ratchet-L2':>11} {'trust-L2':>9} {'ceiling':>8}")
        for arm in AN_ARMS:
            g = deep_guard_read(e3, truth, use_rows, arm)
            if g is None:
                continue
            deeps[arm] = g
            c = g["committed_L2"]
            print(f"       {arm:<12} {(str(c['n_true']) + '/' + str(c['n_learned'])):>13} "
                  f"{str(c['reconstruction_ok']):>9} {g['unguarded_junk_share']:>10.3f} "
                  f"{g['ratchet_l2_junk_share']:>11.3f} {g['trust_l2_junk_share']:>9.3f} "
                  f"{g['ceiling_junk_share']:>8.3f}")
        print("     Only 8 of the 96 junk clean halves have BOTH level-2 halves in the true L2")
        print("     table (0.083 by count, 0.140 by mass) — so a perfect L2 table would refuse")
        print("     91.7% of the trap. The arms' own tables carry 3-6 junk L2 rows, and hard")
        print("     membership routes the trap straight through them; USE-WEIGHTING the same")
        print("     rows recovers about a third of the gap, in every arm.")

    print("\n[5] SELECTOR SIMULATION — the real `questions.py` selectors on real menus, with the")
    print("     delivery model taken from the null arm's own banked stream ([7a]).")
    deliv = delivery_model(truth, eras)
    sims, fits = [], []
    for r in live:
        if r["era_level"] not in deliv:
            continue
        dr = next(d for d in dstar if d["era_level"] == r["era_level"])
        lo, lo_meta = _lower_op(r, truth, banked)
        f_native = round(r["native_distractor_rate"], 3)
        tgt_stats = None
        for a_ in banked.get("arms", []):
            if a_["arm"] == "q_exo":
                tgt_stats = a_["end_of_era"].get(f"era{r['era_level']}")
        fit = (fit_tau(r, dr, truth, deliv[r["era_level"]], lo, tgt_stats)
               if tgt_stats else {"tau": 1.0, "err": None, "target": None})
        dl = _sharpen(deliv[r["era_level"]], fit["tau"])
        fits.append({"era_level": r["era_level"], **{k: v for k, v in fit.items()
                                                     if k != "model"}})
        res = "-" if fit.get("err") is None else f"{fit['err']:.3f}"
        print(f"\n     era L{r['era_level']}n{r['era_node']} -> L{r['target_level']}  "
              f"(native f = {f_native:.3f}; true halves {r['n_true_halves_reachable']}, "
              f"junk halves {r['n_junk_halves_reachable']}; modelled operative L"
              f"{r['clean_level']} table {lo_meta['n_true']}/{lo_meta['n_learned']} true; "
              f"delivery tau = {fit['tau']}, fit residual {res})")
        print(f"       {'f':>6} {'selector':<9} {'take':>6} {'mined':>6} {'distinct':>9} "
              f"{'@sup':>5} {'true@sup':>9} {'junk@sup':>9} {'recall':>7}")
        for f in (f_native,) + F_GRID:
            modes = ["exo", "novel", "endo", "ratchet", "comp", "bisect"]
            dg = deeps.get("q_exo") if r["era_level"] == 3 else None
            if dg is not None:
                modes += ["ratchet_l2", "trust"]
            for mode in modes:
                s = _simulate(r, dr, truth, dl, lo, f, mode,
                              deep=(None if dg is None else
                                    {"score": ({k: 1.0 for k in dg["_admit"]["junk"]
                                                + dg["_admit"]["true"]}
                                               if mode == "ratchet_l2" else dg["_score"])}))
                s["modelled_lower"] = lo_meta
                sims.append(s)
                print(f"       {f:>6.3f} {mode:<11} {s['distractor_take_rate']:>6.3f} "
                      f"{s['distractor_mined_rate']:>6.3f} {s['n_distinct']:>9} "
                      f"{s['at_support']:>5} {s['true_at_support']:>9} "
                      f"{s['junk_at_support']:>9} {s['recall']:>7.4f}")
        eo = None
        for a_ in banked.get("arms", []):
            if a_["arm"] == "q_exo":
                eo = a_["end_of_era"].get(f"era{r['era_level']}")
        if eo:
            print(f"       VALIDATION vs banked q_exo end-of-era: distinct "
                  f"{eo['n_distinct']} | @sup {eo['at_support']} | true {eo['true_at_support']} "
                  f"| junk {eo['junk_at_support']}   (model at f={f_native}: see the exo row)")

    print("\n[5c] DOES THE INCUMBENT DELIVERY LEDGER DISCRIMINATE? Its credit rule is `the")
    print("     delivered key was not yet at support`, so a channel with an inexhaustible key")
    print("     supply keeps full weight. Mean ledger weight, junk halves vs true halves,")
    print("     on the `endo` arm:")
    print(f"       {'era':<8} {'f':>6} {'w(junk half)':>13} {'w(true half)':>13} {'ratio':>7}")
    for s in sims:
        if s["mode"] != "endo" or s["ledger_w_junk"] is None or s["ledger_w_true"] is None:
            continue
        print(f"       era{s['era_level']:<5} {s['f']:>6.3f} {s['ledger_w_junk']:>13.3f} "
              f"{s['ledger_w_true']:>13.3f} "
              f"{s['ledger_w_junk'] / max(s['ledger_w_true'], 1e-9):>7.2f}")

    PREM_MODES = ("novel", "endo", "ratchet", "ratchet_l2", "trust", "comp", "bisect")
    print("\n[5b] THE SELECTION PREMIUM vs THE WORLD'S WORTHLESS FRACTION")
    print("     true keys at support at the target level, minus the null arm's, per f.")
    print("     (First-order model: the delivery model, the plant and the committed table are")
    print("      held fixed, so this reads the MENU's arithmetic only.)")
    for r in live:
        rows = [s for s in sims if s["era_level"] == r["era_level"]]
        fs = sorted({s["f"] for s in rows})
        print(f"\n     era L{r['era_level']}n{r['era_node']} -> L{r['target_level']}")
        print(f"       {'f':>6} {'exo true@sup':>13} " + " ".join(
            f"{('D ' + m):>12}" for m in PREM_MODES))
        for f in fs:
            d = {s["mode"]: s for s in rows if s["f"] == f}
            base = d["exo"]["true_at_support"]
            print(f"       {f:>6.3f} {base:>13} " + " ".join(
                f"{(d[m]['true_at_support'] - base):>+12d}" if m in d else f"{'-':>12}"
                for m in PREM_MODES))
    print("\n     The RATCHET guard (prefer a clean half that is a row of MY OWN operative lower")
    print("     table) is rejected here, offline: [7c] measures that table at precision 0.25-0.50")
    print("     at L3, so the guard prefers the arm's own committed JUNK rows 50-75% of the time")
    print("     and its distractor take is indistinguishable from the unguarded judge's. A guard")
    print("     that stands on your own table inherits your table's errors.")

    print("\n[8] THE REJECTED CHANNELS")
    print("     (i) A FOREIGN GRAMMAR (a second RHM at the same shape, independent rules).")
    print("         `crystallize.units.apply_move` materialises every span through a max-sum DP")
    print("         over THIS world's `rules_t` and renders it through `canon`, so the agent")
    print("         cannot produce a foreign-legal answer; and `run_arm` block (b) mines only")
    print("         `out['x'][ps > 0.5]`, i.e. answers the WORLD's exact grader passed. A")
    print("         foreign instance is therefore never mined and the channel degenerates into")
    print("         'burn priced budget, bank nothing'. Grading it under a foreign grader would")
    print("         put foreign-legal answers into the value buffer and the generator's")
    print("         fine-tune set — moving the plant, not the question distribution.")
    print("     (ii) AN UNLEARNABLE CHANNEL AT MATCHED DIFFICULTY. d* small => near a legal")
    print("         derivation => solved => mined; and [6] shows the reachable junk-half pool is")
    print("         smaller than budget/support, so whatever is posed recurs and closes. An")
    print("         unlearnable arm therefore has to give up the difficulty pin, exactly as")
    print("         `q_comp_free` gives it up to be the honest comfort pole.")

    out = {"params": {"V": V, "S": S, "DEPTH": DEPTH, "M": M, "rule_seed": RULE_SEED,
                      "support": SUPPORT, "mine_cap": MINE_CAP, "n_pr": N_PR,
                      "k_menu": K_MENU, "max_macro_level": MAXL, "ladder": LADDER,
                      "n_pool": N_POOL, "n_dstar": N_DSTAR, "f_grid": list(F_GRID)},
           "worthlessness_proof": proof,
           "eras": [{k: v for k, v in r.items() if k not in ("p_junk", "p_true",
                                                             "junk_pool", "true_pool")}
                    for r in eras],
           "junk_pools": {f"L{r['era_level']}n{r['era_node']}": {
               "junk_halves": [list(k) for k in r["junk_pool"]],
               "true_halves": [list(k) for k in r["true_pool"]]}
               for r in live},
           "use_record": [{"arm": u["arm"],
                           "levels": {k: {kk: vv for kk, vv in v.items()
                                          if kk not in ("use", "true_mask")}
                                      for k, v in u["levels"].items()}} for u in use_rows],
           "deep_guard": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                          for k, v in deeps.items()},
           "dstar": dstar, "construction_cost": cost, "simulation": sims,
           "delivery_fit": fits,
           "banked": {"available": banked["available"],
                      "commits": {f"{k[0]}/L{k[1]}": v
                                  for k, v in banked.get("commits", {}).items()},
                      "arms": [{k: v for k, v in a.items() if not k.startswith("_")}
                               for a in banked.get("arms", [])]},
           "guard_leak": leak}
    with open(os.path.join(HERE, "phase0_trap.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0_trap.json')}")
    return out


if __name__ == "__main__":
    main()
