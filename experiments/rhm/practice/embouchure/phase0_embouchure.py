"""phase0_embouchure — Q0 for `embouchure`, offline, CPU, from the BANKED `inflection` logs.

The question the node asks is where the renderer's training signal comes from: the world's
surface (the perceptual route, `inflection`'s `fit_rule`, called `surface` here) or the
learner's own metered attempts (the production route). Q0 settles four things before any GPU
is bought, entirely out of `../inflection/figures/<tag>/<arm>/results.json`:

  (i)   `own_verdict` — the builder's `fit_rule_verdict` ceiling (DESIGN.md §8) — fitted
        OFFLINE from `log["spell_rows"]`, cycle by cycle, with `inflection`'s own head,
        optimiser and report, beside `surface`'s banked `log["render"]` trajectory.
  (ii)  the VOLUME ratio: own written labelled blocks per cycle against harvested surface
        blocks per cycle, by register.
  (iii) the READ-BACK check: on the learner's own written blocks, does the frozen reader ever
        return a feature other than the one intended?
  (iv)  the identifiability ceiling AT OWN-WRITE VOLUME (the sizing lane's `SIZING.md` §2
        method, generalised from "all practised columns observed" to "the cells the stream has
        actually delivered by cycle c").

Facts only. No GPU, no Modal, no substrate. Interpretation is the orchestrator's.

Run (from experiments/):
    PYTHONPATH=. python3 rhm/practice/embouchure/phase0_embouchure.py
"""

import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(os.path.dirname(HERE), "inflection", "figures")

# The banked tags this lane reads, and which of their arms carry `log["spell_rows"]`.
# `spell_rows` is written only where the arm HAS a rendering organ (`rhead` or `spell_store`),
# so `canon`/`given_rule` arms carry none — a fact, and the first thing Q0 measures.
TAGS = {
    "if_q1c_yk": ["canon_s", "given_rule_s", "leaf_s", "fit_rule_s", "fit_index_s"],
    "if_q3_e": ["m_given_rule", "m_fit_rule", "m_leaf"],
    "if_q3_e_sp": ["m_fit_rule_sp", "m_leaf_sp"],
    "if_q3_f2": ["leaf_s", "canon_s", "given_rule_s"],
}

N_PR = 64          # instances per practice cycle (`cfg["n_pr"]`)
N_BLK = 32         # bottom blocks per instance (s**depth / s = 2**6 / 2)
FIT_STEPS, FIT_BATCH, FIT_LR, FIT_BUF_CAP = 64, 256, 3e-2, 200_000


def load(tag, arm):
    with open(os.path.join(BANK, tag, arm, "results.json")) as fh:
        return json.load(fh)


def setup(tag):
    with open(os.path.join(BANK, tag, "setup.json")) as fh:
        return json.load(fh)


def rows_of(res):
    """`log["spell_rows"]` as an (N, 4) int array per cycle: (feature, register,
    k_written, k_rule). The reader's parse supplies the feature."""
    return [np.asarray(c or [], np.int64).reshape(-1, 4)
            for c in (res["log"].get("spell_rows") or [])]


# --------------------------------------------------------------------------- #
# (iii) THE READ-BACK CHECK
# --------------------------------------------------------------------------- #

def sec_readback():
    """Does the frozen reader ever return a feature other than the one intended, on the
    learner's OWN written blocks?

    Two halves, and they meet.

    ANALYTIC. `observed_synonyms(x, f, k_of, v, s)` marks a block `good` iff its code is one of
    the m tuples of the feature the reader returned. On a draw with no bottom-tuple collision
    every legal code has exactly ONE owner, so `good` is true iff the reader returned that
    code's true owner. A misspelling writes the OTHER synonym OF THE SAME FEATURE, which is a
    legal code owned by that same feature — so an exact reader returns the intended feature
    whether or not the spelling is the rule's, and a misspelling has no consequence through the
    reader. This half is decided by counting collisions in the bottom map.

    MEASURED. `log["spell_rows"]` keeps only the `good` blocks of the learner's own answer
    (`out["x"]`, at the solved instances), capped at 512 by a uniform subsample; `render["n"]`
    is the same filter on the world's observation. Both denominators are known
    (n_solved x 32), so the drop rate IS the reader's block error on those configurations, and
    the surviving rows' misspelling rate against `e_sp_practice` (the same quantity computed
    through the EXACT inverse map) says whether misspelled blocks are dropped preferentially.
    """
    from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
    out = {"collisions": {}, "drop": [], "misspell_agreement": []}

    for rs in (0, 6):
        rules = generate_rules_distinct(8, 2, 6, 2, seed=rs)
        bot = rules[-1]                                     # (v, m, s)
        codes = {}
        for f in range(8):
            for k in range(2):
                c = int(bot[f, k, 0] + 8 * bot[f, k, 1])
                codes.setdefault(c, []).append((f, k))
        coll = {c: o for c, o in codes.items() if len(o) > 1}
        inv = build_inverse_maps(rules)[-1] if False else None
        out["collisions"][f"rule_seed_{rs}"] = {
            "n_legal_codes": len(codes), "n_collisions": len(coll),
            "colliding": {str(c): [[int(f), int(k)] for f, k in o] for c, o in coll.items()},
            "readback_identity_for_exact_reader": len(coll) == 0}
        del inv

    for tag, arms in TAGS.items():
        for arm in arms:
            res = load(tag, arm)
            log = res["log"]
            rws = rows_of(res)
            if not any(r.shape[0] for r in rws):
                out["drop"].append({"tag": tag, "arm": arm, "spell_rows": "absent",
                                    "why": "arm has no rendering organ; the channel is not "
                                           "logged for canon / given_rule"})
                continue
            ep = np.asarray(log["e_practice"], float)
            n_sol = np.rint((1.0 - ep) * N_PR).astype(int)
            n_rows = np.array([r.shape[0] for r in rws])
            expect = np.minimum(n_sol * N_BLK, 512)
            unc = n_sol * N_BLK <= 512                       # cycles where the cap did not bind
            # the harvest side, where it exists: `render["n"]` is cumulative and uncapped
            hn = [r.get("n") for r in log.get("render", []) if isinstance(r, dict)]
            harvest_exact = None
            if len(hn) == len(ep) and all(x is not None for x in hn):
                dn = np.diff(np.concatenate([[0], np.asarray(hn, float)]))
                harvest_exact = float(np.mean(dn == n_sol * N_BLK))
            ok = unc & (n_sol > 0)
            drop = (1.0 - n_rows[ok] / (n_sol[ok] * N_BLK)) if ok.any() else None
            A = np.concatenate([r for r in rws if r.shape[0]])
            miss_surv = float((A[:, 2] != A[:, 3]).mean())
            # A dropped block is one `agood` rejected: the reader returned a feature that does
            # not own the written code. Misspelled blocks that SURVIVED are `miss_surv` of the
            # survivors; the exact map says `p` of all blocks are misspelled, so the dropped
            # set contains `p - miss_surv * (1 - d)` misspelled blocks — the enrichment below
            # is how much more likely a misspelling is to be misread than a correct spelling.
            d = float(np.mean(drop)) if drop is not None else None
            p = float(np.mean(np.asarray(log["e_sp_practice"], float)))
            enrich = None
            if d is not None and p > 1e-6:
                m_drop = max(p - miss_surv * (1.0 - d), 0.0)
                fail_miss = m_drop / p
                fail_ok = (d - m_drop) / max(1.0 - p, 1e-9)
                enrich = {"drop_frac": d, "misspell_rate_exact_map": p,
                          "misspell_frac_of_survivors": miss_surv,
                          "readback_fail_given_misspelled": fail_miss,
                          "readback_fail_given_on_rule": fail_ok,
                          "enrichment": (fail_miss / fail_ok) if fail_ok > 1e-9 else None}
            # cells the learner is KNOWN to have written at: a misspelled block can only be an
            # own write (the world spells every block it wrote on-rule).
            M = A[A[:, 2] != A[:, 3]]
            row = {"tag": tag, "arm": arm,
                   "n_cycles": int(len(rws)),
                   "n_uncapped_cycles": int(ok.sum()),
                   "rows_equal_expected_frac": float(np.mean(n_rows == expect)),
                   "readback_acc_uncapped": (float(1.0 - d) if d is not None else None),
                   "harvest_rows_equal_nsol_x_32_frac": harvest_exact,
                   "n_misspelled_rows_surviving": int(M.shape[0]),
                   "own_write_cells_lower_bound": int(len(set(map(tuple, M[:, :2].tolist())))),
                   "misspell_drop": enrich}
            out["drop"].append(row)

            # misspelling rate, reader-parsed rows vs the exact-map `e_sp_practice`
            esp = np.asarray(log["e_sp_practice"], float)
            era = np.asarray(log["era"], int)
            for e in sorted(set(era.tolist())):
                sel = era == e
                B = np.concatenate([rws[i] for i in np.flatnonzero(sel)])
                out["misspell_agreement"].append(
                    {"tag": tag, "arm": arm, "era": int(e),
                     "reader_rows_misspell": float((B[:, 2] != B[:, 3]).mean()),
                     "exact_map_e_sp_practice": float(esp[sel].mean())})
    return out


# --------------------------------------------------------------------------- #
# (ii) THE VOLUME RATIO
# --------------------------------------------------------------------------- #

def sec_volume():
    """Own written labelled blocks per cycle against harvested surface blocks per cycle.

    SURFACE is exact and logged: `render["n"]` grows by `n_solved x 32` every cycle (the whole
    bottom row of every solved observation).

    OWN WRITES are not logged as a count anywhere, and are NOT the renderer's `n_written`
    tally, which counts every block the BEAM rendered including rollouts that were never
    graded. The blocks with a meter-visible label are the ones standing in the graded
    configuration `out["x"]`. Their number is recovered from two logged numbers that share
    that denominator:

        e_sp_practice = (own written blocks that are misspelled) / (32 blocks x 64 instances)
        spell["spell_err"] = (misspelled renders) / (all renders)

    so mean own-written blocks per instance = 32 * e_sp_practice / spell_err. The world spells
    every block it wrote on-rule, so only an own write can be wrong: the numerator is exact.
    The denominator is the renderer's error over the beam's whole render stream rather than
    over the accepted answer alone, which is the estimator's one assumption; it is
    cross-checked across arms and eras below. Arms whose renderer stops misspelling (`fit_*`
    after era 1, `given_rule` always) carry 0/0 and are excluded.
    """
    out = {"per_arm": [], "by_register": []}
    for tag, arms in TAGS.items():
        for arm in arms:
            res = load(tag, arm)
            log = res["log"]
            era = np.asarray(log["era"], int)
            ep = np.asarray(log["e_practice"], float)
            n_sol = np.rint((1.0 - ep) * N_PR).astype(int)
            esp = np.asarray(log["e_sp_practice"], float)
            sp = np.asarray([r["spell_err"] for r in log["spell"]], float)
            for e in sorted(set(era.tolist())):
                sel = era == e
                surf = float((n_sol[sel] * N_BLK).mean())
                num = float(esp[sel].mean())
                den = float(sp[sel].mean())
                blocks_per_inst = (N_BLK * num / den) if den > 1e-6 else None
                own = (blocks_per_inst * N_PR) if blocks_per_inst is not None else None
                out["per_arm"].append(
                    {"tag": tag, "arm": arm, "era": int(e), "n_cycles": int(sel.sum()),
                     "n_solved_per_cycle": float(n_sol[sel].mean()),
                     "surface_blocks_per_cycle": surf,
                     "e_sp_practice": num, "spell_err_written": den,
                     "own_written_blocks_per_instance": blocks_per_inst,
                     "own_written_blocks_per_cycle": own,
                     "surface_over_own": (surf / own) if own else None})
            rws = rows_of(res)
            if any(r.shape[0] for r in rws):
                A = np.concatenate(rws)
                out["by_register"].append(
                    {"tag": tag, "arm": arm,
                     "spell_rows_by_register": np.bincount(A[:, 1], minlength=8).tolist(),
                     "spell_rows_by_feature": np.bincount(A[:, 0], minlength=8).tolist(),
                     "registers_visited": sorted(set(A[:, 1].tolist()))})
    return out


# --------------------------------------------------------------------------- #
# (i) `own_verdict` FITTED OFFLINE FROM THE BANKED ROWS
# --------------------------------------------------------------------------- #

def sec_verdict_fit(tag, arm, volume_frac=None, label="own_verdict",
                    misspelled_only=False, fit_seed_offset=0):
    """`inflection`'s own head, optimiser, step count and report, on the banked rows.

    The row is (feature-as-read, register, k_written, k_rule); the LABEL is `k_rule`, i.e. the
    per-block verdict the meter would return on that write — DESIGN.md §8's `fit_rule_verdict`.
    Buffer discipline is `rule_head_fit`'s: append, keep the last `fit_buf_cap`, fit every
    cycle. `volume_frac` subsamples each cycle's rows to emulate the production route's own
    volume rather than the whole answer's.

    NOT bit-comparable with the banked `surface` trajectory: the run's minibatch stream shares
    `frng` with the spell-row subsample, which cannot be replayed from the log. Same class,
    same hyper-parameters, same report; a trajectory, not a replay.
    """
    import torch
    from rhm.practice.inflection import inflection as I

    res = load(tag, arm)
    log = res["log"]
    st = setup(tag)
    rule = I.make_rule(st["rule"]["name"], st["rule"]["v"], st["rule"]["m"],
                       st["rule"]["n_ctx"])
    assert (np.asarray(rule.K) == np.asarray(st["rule"]["K"])).all(), "rule replay mismatch"
    practiced = st["practiced"]
    dev = torch.device("cpu")
    head = I.build_rule_head(st["rule"]["v"], st["rule"]["n_ctx"], st["rule"]["m"],
                             ctx_rep=("onehot" if "index" in arm else "scalar"),
                             seed=int(res["config"]["seed"]) + 8_675_309 + fit_seed_offset,
                             kind="linear").to(dev)
    head.eval()
    opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
    buf = {"f": torch.zeros(0, dtype=torch.long), "c": torch.zeros(0, dtype=torch.long),
           "k": torch.zeros(0, dtype=torch.long)}
    rng = np.random.default_rng(int(res["config"]["seed"]) + 4_242_424
                            + 1_000 * int(fit_seed_offset))
    traj, seen = [], {f: set() for f in range(st["rule"]["v"])}
    for c, R in enumerate(rows_of(res)):
        if misspelled_only and R.shape[0]:
            R = R[R[:, 2] != R[:, 3]]
        if volume_frac is not None and R.shape[0]:
            n = max(1, int(round(R.shape[0] * volume_frac)))
            R = R[rng.permutation(R.shape[0])[:n]]
        if R.shape[0]:
            buf["f"] = torch.cat([buf["f"], torch.from_numpy(R[:, 0])])[-FIT_BUF_CAP:]
            buf["c"] = torch.cat([buf["c"], torch.from_numpy(R[:, 1])])[-FIT_BUF_CAP:]
            buf["k"] = torch.cat([buf["k"], torch.from_numpy(R[:, 3])])[-FIT_BUF_CAP:]
            for f, r in zip(R[:, 0].tolist(), R[:, 1].tolist()):
                seen[f].add(r)
        info = I.rule_head_fit(head, opt, buf, n_steps=FIT_STEPS, batch=FIT_BATCH,
                               device=dev, rng=rng)
        info.update(I.rule_head_report(head, rule, dev, practiced))
        info["cycle"] = int(log["cycle"][c])
        info["era"] = int(log["era"][c])
        info["n_cells_seen"] = int(sum(len(s) for s in seen.values()))
        traj.append(info)
    return {"tag": tag, "arm": arm, "label": label, "volume_frac": volume_frac,
            "misspelled_only": misspelled_only,
            "traj": [{k: t[k] for k in ("cycle", "era", "n", "loss", "acc_practiced",
                                        "acc_heldout", "acc_all", "det_theta", "monotone",
                                        "theta_hat", "n_cells_seen")} for t in traj]}


def sec_seed_spread(tag, arm, own_frac, seeds=(0, 1, 2, 3)):
    """How much of a cycles-to-convergence difference is the OFFLINE HARNESS's own noise?

    The run's fit stream (`frng`) is shared with the spell-row subsample and cannot be
    replayed from the log, so "own_verdict converged at cycle X, surface at cycle Y" is not a
    safe comparison across harnesses. This measures the spread WITHIN the harness: the first
    cycle at which `acc_heldout` reaches its endpoint and stays there, over four fit seeds, at
    full logged volume and at own-write volume.
    """
    import torch
    out = []
    for vf, name in ((None, "full logged rows"), (own_frac, "own-write volume")):
        for sd in seeds:
            fr = sec_verdict_fit(tag, arm, vf, name, fit_seed_offset=sd)
            acc = [t["acc_heldout"] for t in fr["traj"]]
            end = acc[-1]
            first = next((i + 1 for i in range(len(acc))
                          if all(a >= end - 1e-9 for a in acc[i:])), None)
            out.append({"volume": name, "seed_offset": sd, "acc_heldout_end": end,
                        "first_cycle_at_endpoint": first,
                        "theta_hat_end": fr["traj"][-1]["theta_hat"]})
    return out


def banked_surface(tag, arm):
    """`surface`'s own trajectory, straight out of the banked `log["render"]`."""
    res = load(tag, arm)
    log = res["log"]
    return {"tag": tag, "arm": arm, "label": "surface (banked)",
            "traj": [{"cycle": int(log["cycle"][i]), "era": int(log["era"][i]),
                      "n": r.get("n"), "loss": r.get("loss"),
                      "acc_practiced": r.get("acc_practiced"),
                      "acc_heldout": r.get("acc_heldout"), "acc_all": r.get("acc_all"),
                      "det_theta": r.get("det_theta"), "monotone": r.get("monotone"),
                      "theta_hat": r.get("theta_hat")}
                     for i, r in enumerate(log["render"]) if isinstance(r, dict)]}


# --------------------------------------------------------------------------- #
# (iv) IDENTIFIABILITY AT OWN-WRITE VOLUME
# --------------------------------------------------------------------------- #

def rows_monotone(R):
    return [tuple(1 if c >= t else 0 for c in range(R)) for t in range(R + 1)]


def rows_table(R):
    return [tuple(b) for b in itertools.product((0, 1), repeat=R)]


def ladder_observed(K, admissible, obs, held):
    """`SIZING.md` §2's `ladder_rowsep`, generalised from a practised COLUMN SET to the cells
    the stream has actually delivered.

    `obs[f]` is the set of registers at which feature f has been seen. A class member (a row)
    is consistent iff it agrees with K on exactly those cells; `det` is the fraction of
    features whose value at `held` is then pinned, `acc` the expected accuracy of a predictor
    drawing uniformly from the rows still consistent. With `obs[f]` = the practised set for
    every f this reduces to `ladder_rowsep` exactly.
    """
    v = K.shape[0]
    det = acc = 0.0
    for f in range(v):
        o = sorted(obs.get(f, ()))
        cons = [r for r in admissible if all(r[c] == int(K[f, c]) for c in o)]
        vals = {r[held] for r in cons}
        det += (len(vals) == 1)
        acc += sum(1 for r in cons if r[held] == int(K[f, held])) / len(cons)
    return det / v, acc / v


def sec_identifiability(tag, arm, own_frac):
    """The §2 ladder as a function of CYCLE, at each channel's own volume.

    Both channels draw their (feature, register) cells from the same instances, so the only
    thing that moves is how many rows per cycle arrive: the surface takes all 32 blocks of
    every solved instance, the production route only the blocks it wrote. `own_frac` is
    §(ii)'s measured ratio.
    """
    res = load(tag, arm)
    st = setup(tag)
    K = np.asarray(st["rule"]["K"], np.int64)
    R = K.shape[1]
    prac, held = st["practiced"], [c for c in range(R) if c not in st["practiced"]]
    mono, tab = rows_monotone(R), rows_table(R)
    rng = np.random.default_rng(20260910)

    def run(frac):
        obs, out = {f: set() for f in range(K.shape[0])}, []
        first_full = None
        for c, A in enumerate(rows_of(res)):
            if frac is not None and A.shape[0]:
                n = max(1, int(round(A.shape[0] * frac)))
                A = A[rng.permutation(A.shape[0])[:n]]
            for f, r in zip(A[:, 0].tolist(), A[:, 1].tolist()):
                obs[f].add(r)
            n_cells = sum(len(s) for s in obs.values())
            if first_full is None and n_cells == K.shape[0] * len(prac):
                first_full = c + 1
            dm = np.mean([ladder_observed(K, mono, obs, h) for h in held], axis=0)
            dt = np.mean([ladder_observed(K, tab, obs, h) for h in held], axis=0)
            out.append({"cycle": c + 1, "n_cells_seen": int(n_cells),
                        "scalar_det": float(dm[0]), "scalar_acc": float(dm[1]),
                        "onehot_det": float(dt[0]), "onehot_acc": float(dt[1])})
        return first_full, out

    ff_s, tr_s = run(None)
    ff_o, tr_o = run(own_frac)
    full = {f: set(prac) for f in range(K.shape[0])}
    dm = np.mean([ladder_observed(K, mono, full, h) for h in held], axis=0)
    dt = np.mean([ladder_observed(K, tab, full, h) for h in held], axis=0)
    return {"tag": tag, "arm": arm, "own_frac": own_frac,
            "practiced": prac, "heldout": held,
            "ceiling_at_C3_this_subset": {"additive_scalar_det": float(dm[0]),
                                          "additive_scalar_acc": float(dm[1]),
                                          "table_1hot_det": float(dt[0]),
                                          "table_1hot_acc": float(dt[1])},
            "first_cycle_all_24_cells_surface": ff_s,
            "first_cycle_all_24_cells_own": ff_o,
            "surface": tr_s[:12] + tr_s[::10], "own": tr_o[:12] + tr_o[::10]}


# --------------------------------------------------------------------------- #

def main():
    out = {}
    print("=" * 78)
    print("Q0 (iii) — THE READ-BACK CHECK")
    print("=" * 78)
    rb = sec_readback()
    out["readback"] = rb
    for k, c in rb["collisions"].items():
        print(f"  {k}: {c['n_legal_codes']} legal codes, {c['n_collisions']} collisions "
              f"-> exact-reader read-back is the identity: "
              f"{c['readback_identity_for_exact_reader']}")
        for code, own in c["colliding"].items():
            print(f"      code {code}: {own}")
    print()
    print(f"  {'tag':11s} {'arm':15s} {'cyc':>4s} {'uncap':>5s} {'rows==min(nsol*32,512)':>23s} "
          f"{'readback_acc':>12s} {'harvest==nsol*32':>16s}")
    for r in rb["drop"]:
        if r.get("spell_rows") == "absent":
            print(f"  {r['tag']:11s} {r['arm']:15s}    —  no spell_rows ({r['why'][:44]})")
            continue
        ra = r["readback_acc_uncapped"]
        he = r["harvest_rows_equal_nsol_x_32_frac"]
        print(f"  {r['tag']:11s} {r['arm']:15s} {r['n_cycles']:4d} {r['n_uncapped_cycles']:5d} "
              f"{r['rows_equal_expected_frac']:23.4f} "
              f"{('%.6f' % ra) if ra is not None else '—':>12s} "
              f"{('%.4f' % he) if he is not None else '—':>16s}")
    print()
    print("  misspelled own writes: do they read back to the intended feature?")
    print(f"  {'tag':11s} {'arm':15s} {'miss rows':>9s} {'cells':>5s} {'drop':>7s} "
          f"{'P(fail|misspelled)':>18s} {'P(fail|on-rule)':>15s} {'enrich':>7s}")
    for r in rb["drop"]:
        if r.get("spell_rows") == "absent":
            continue
        e = r.get("misspell_drop")
        print(f"  {r['tag']:11s} {r['arm']:15s} {r['n_misspelled_rows_surviving']:9d} "
              f"{r['own_write_cells_lower_bound']:5d} "
              + (f"{e['drop_frac']:7.4f} {e['readback_fail_given_misspelled']:18.4f} "
                 f"{e['readback_fail_given_on_rule']:15.4f} "
                 + (f"{e['enrichment']:7.2f}" if e['enrichment'] else "      —")
                 if e else "      —                  —               —       —"))
    print()
    print(f"  {'tag':11s} {'arm':15s} {'era':>3s} {'reader-rows misspell':>21s} "
          f"{'exact-map e_sp_prac':>20s}")
    for r in rb["misspell_agreement"]:
        print(f"  {r['tag']:11s} {r['arm']:15s} {r['era']:3d} "
              f"{r['reader_rows_misspell']:21.5f} {r['exact_map_e_sp_practice']:20.5f}")

    print()
    print("=" * 78)
    print("Q0 (ii) — VOLUME: own written blocks vs harvested surface blocks, per cycle")
    print("=" * 78)
    vol = sec_volume()
    out["volume"] = vol
    print(f"  {'tag':11s} {'arm':15s} {'era':>3s} {'n_sol':>6s} {'surface/cyc':>11s} "
          f"{'e_sp_prac':>10s} {'wr_err':>7s} {'own blk/inst':>12s} {'own/cyc':>8s} "
          f"{'surf/own':>8s}")
    for r in vol["per_arm"]:
        bp = r["own_written_blocks_per_instance"]
        ow = r["own_written_blocks_per_cycle"]
        so = r["surface_over_own"]
        print(f"  {r['tag']:11s} {r['arm']:15s} {r['era']:3d} {r['n_solved_per_cycle']:6.1f} "
              f"{r['surface_blocks_per_cycle']:11.0f} {r['e_sp_practice']:10.5f} "
              f"{r['spell_err_written']:7.4f} "
              f"{('%.2f' % bp) if bp else '—':>12s} {('%.0f' % ow) if ow else '—':>8s} "
              f"{('%.2f' % so) if so else '—':>8s}")
    print()
    for r in vol["by_register"]:
        print(f"  {r['tag']:11s} {r['arm']:15s} registers visited {r['registers_visited']}  "
              f"rows by register {r['spell_rows_by_register']}")
    out["own_frac_estimate"] = None

    # the estimator, pooled over the arms whose renderer misspells at a measurable rate
    est = [r["own_written_blocks_per_instance"] / N_BLK for r in vol["per_arm"]
           if r["own_written_blocks_per_instance"] and r["spell_err_written"] > 0.05]
    own_frac = float(np.mean(est)) if est else None
    out["own_frac_estimate"] = {"mean": own_frac, "n": len(est),
                                "min": float(np.min(est)) if est else None,
                                "max": float(np.max(est)) if est else None}
    print(f"\n  own-write fraction of the graded configuration's blocks: "
          f"mean {own_frac:.4f} over {len(est)} (arm, era) cells "
          f"[{np.min(est):.4f}, {np.max(est):.4f}]")

    print()
    print("=" * 78)
    print("Q0 (i) — `own_verdict` fitted offline from `spell_rows`, beside banked `surface`")
    print("=" * 78)
    fits = []
    for tag, arm in (("if_q1c_yk", "fit_rule_s"), ("if_q1c_yk", "leaf_s")):
        fits.append(sec_verdict_fit(tag, arm, None, "own_verdict (full logged rows)"))
        fits.append(sec_verdict_fit(tag, arm, own_frac,
                                    "own_verdict (subsampled to own-write volume)"))
        fits.append(sec_verdict_fit(tag, arm, None,
                                    "own_verdict (misspelled rows only = certainly own writes)",
                                    misspelled_only=True))
    fits.append(banked_surface("if_q1c_yk", "fit_rule_s"))
    fits.append(banked_surface("if_q1c_yk", "fit_index_s"))
    out["fits"] = fits
    marks = [1, 2, 5, 10, 20, 40, 60, 85, 100, 120, 140]
    for fr in fits:
        print(f"\n  {fr['tag']}/{fr['arm']} — {fr['label']}"
              + (f"  (volume_frac {fr['volume_frac']:.4f})"
                 if fr.get("volume_frac") else ""))
        print(f"    {'cyc':>4s} {'n_buf':>7s} {'acc_prac':>8s} {'acc_held':>8s} "
              f"{'det_th':>6s} {'mono':>5s}  theta_hat")
        for t in fr["traj"]:
            if t["cycle"] in marks:
                print(f"    {t['cycle']:4d} {t['n'] or 0:7d} {t['acc_practiced']:8.4f} "
                      f"{t['acc_heldout']:8.4f} {t['det_theta']:6.3f} "
                      f"{str(t['monotone']):>5s}  {t['theta_hat']}")
        last = fr["traj"][-1]
        print(f"    END  {last['n'] or 0:7d} {last['acc_practiced']:8.4f} "
              f"{last['acc_heldout']:8.4f} {last['det_theta']:6.3f}  {last['theta_hat']}")

    print()
    print("  harness noise: first cycle at the endpoint, four fit seeds "
          "(`if_q1c_yk/fit_rule_s` rows)")
    sp = sec_seed_spread("if_q1c_yk", "fit_rule_s", own_frac)
    out["seed_spread"] = sp
    for r in sp:
        print(f"    {r['volume']:20s} seed+{r['seed_offset']}  acc_held_end "
              f"{r['acc_heldout_end']:.4f}  first cycle at endpoint "
              f"{r['first_cycle_at_endpoint']}  theta_hat {r['theta_hat_end']}")

    print()
    print("=" * 78)
    print("Q0 (iv) — identifiability at own-write volume (SIZING.md §2's method)")
    print("=" * 78)
    ident = sec_identifiability("if_q1c_yk", "fit_rule_s", own_frac)
    out["identifiability"] = ident
    ce = ident["ceiling_at_C3_this_subset"]
    print(f"  practised {ident['practiced']}  held out {ident['heldout']}")
    print(f"  ceiling with ALL 24 practised cells observed (this subset, this theta draw):")
    print(f"     additive_scalar  det {ce['additive_scalar_det']:.4f}  "
          f"acc {ce['additive_scalar_acc']:.4f}")
    print(f"     table / one-hot  det {ce['table_1hot_det']:.4f}  "
          f"acc {ce['table_1hot_acc']:.4f}")
    print(f"  first cycle all 24 practised cells observed: "
          f"surface {ident['first_cycle_all_24_cells_surface']}, "
          f"own-write volume {ident['first_cycle_all_24_cells_own']}")
    print(f"    {'cyc':>4s} {'cells(surf)':>11s} {'det':>6s} {'acc':>6s} | "
          f"{'cells(own)':>10s} {'det':>6s} {'acc':>6s}")
    for a, b in zip(ident["surface"][:14], ident["own"][:14]):
        print(f"    {a['cycle']:4d} {a['n_cells_seen']:11d} {a['scalar_det']:6.3f} "
              f"{a['scalar_acc']:6.3f} | {b['n_cells_seen']:10d} {b['scalar_det']:6.3f} "
              f"{b['scalar_acc']:6.3f}")

    p = os.path.join(HERE, "phase0.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[phase0] wrote {p}")


if __name__ == "__main__":
    main()
