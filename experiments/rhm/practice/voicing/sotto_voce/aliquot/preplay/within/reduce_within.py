"""[within] THE REDUCER — numpy only, no torch, no modal.

Every AUC in `aliquot` / `duplex` / `soundboard` is POOLED over rows: it mixes the easy-context
/ hard-context contrast into the same number as the candidate-within-a-context contrast, and
`duplex` showed the first of those is most of what shaping wrote.  This file computes the other
one: the AUC restricted to pairs INSIDE one group, where a group is `(slot key, obs bytes)` —
the same masked context at the same slot.

The scores this reads are the dump's own columns plus the two this node's Modal side adds
(`within.py`): the critic's logit on `SN.trunk(core, obs)` and, on held-out rows only, a
`duplex`-protocol linear readout fit on the buffer's own training rows.

Module-level helpers (`np_auc`, `np_z`, `np_compose`) are imported BY `within.py` on the Modal
side so they can be gated against `voicing::vo_auc` and `voicing::vo_compose` in torch; they are
deliberately written here, where there is no torch, so the gate is a real comparison of two
implementations rather than a tautology.

Usage (from experiments/), after `within.py::score` has written the score files and they have
been fetched:

    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/within/reduce_within.py \
        --root <scratch>/sbdump --scores <scratch>/wiscores --out .../within/figures --arms all
"""

import argparse
import collections
import json
import os

import numpy as np

# The six paid soundboard arms, and the two frozen-plant overtone dumps beside them.
SB_ARMS = [("sb_s1", "sb_sv_yk"), ("sb_s1", "sb_yd_yk"), ("sb_s1", "sb_so_yk"),
           ("sb_s2", "sb_sv_yk"), ("sb_s2", "sb_yd_yk"), ("sb_s2", "sb_so_yk")]
OV_ARMS = [("ov_s0b", "ovt_comp_pr_sh"), ("ov_s2", "ovt_comp_pr_dis")]
SEED_OF = {"sb_s1": 0, "sb_s2": 2, "ov_s0b": 0, "ov_s2": 2}

READERS = ("proj", "dp", "crit", "comp", "pw", "pr", "pf", "gw", "gf")
RD_COL = {r: r for r in READERS}
RD_COL["proj"] = "y"
# the run's own four readers, and the fitted controls this node adds (held-out rows only)
RUN_RD = ("proj", "dp", "crit", "comp")
FIT_RD = ("pw", "pr", "pf", "gw", "gf")


# --------------------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------------------- #

def np_auc(x, y):
    """`voicing::vo_auc`'s rank identity with ties averaged, in numpy. None when degenerate."""
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = float((y > 0.5).sum())
    n0 = float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    o = np.argsort(x, kind="stable")
    xs = x[o]
    r = np.empty(x.shape[0], np.float64)
    _, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt).astype(np.float64)
    start = csum - cnt.astype(np.float64)
    r[o] = ((start + csum + 1.0) / 2.0)[inv]
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def np_z(a):
    """`voicing::vo_compose`'s standardisation of ONE candidate set, in numpy: mean removed,
    UNBIASED std (torch's `.std()` default, ddof=1), clamped at 1e-6 exactly as the donor does.
    Gate Z asserts this against `vo_compose` itself on random sets."""
    a = np.asarray(a, np.float64)
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return (a - a.mean()) / max(sd, 1e-6)


def np_compose(dp, cr, w):
    """`vo_compose(dp_sc, cr_sc, span, w)` over ONE candidate set, with `dp` ALREADY divided by
    the span (the dump's `|dp` column is `g / span`, `sb_dump_rows`), so the donor's own
    `/ span` must NOT be applied a second time.  A set of size < 2 returns the prior's order,
    which is the donor's guard."""
    dp = np.asarray(dp, np.float64)
    if dp.size < 2:
        return dp
    return np_z(dp) + float(w) * np_z(cr)


def wc_auc(score, lab, groups):
    """THE WITHIN-CONTEXT AUC: the concordance of the pairs that lie INSIDE one group.

    Returns a dict.  The headline `wc` is PAIR-weighted — every (positive, negative) pair
    inside a group counts once, so a group with many candidates and a balanced split weighs
    more than a 2-row group.  `wc_groupmean` weights every group alike, and `max_share` is the
    share of all pairs contributed by the single largest group, because a thin cell with one
    enormous group is a different object from a thin cell with many small ones.
    Ties count a half, which is `vo_auc`'s own convention and the reason the degenerate filed
    side reads exactly 0.500 rather than being undefined.
    """
    score = np.asarray(score)
    lab = np.asarray(lab)
    conc = 0.0
    tot = 0
    per, sizes = [], []
    for g in groups:
        sc, y = score[g], lab[g]
        ok = np.isfinite(sc)
        sc, y = sc[ok], y[ok]
        pos, neg = sc[y > 0.5], sc[y <= 0.5]
        if pos.size == 0 or neg.size == 0:
            continue
        d = pos[:, None] - neg[None, :]
        c = float((d > 0).sum()) + 0.5 * float((d == 0).sum())
        np_ = pos.size * neg.size
        conc += c
        tot += np_
        per.append(c / np_)
        sizes.append(np_)
    if tot == 0:
        return {"wc": None, "pairs": 0, "groups_used": 0, "wc_groupmean": None,
                "max_share": None}
    return {"wc": conc / tot, "pairs": int(tot), "groups_used": len(per),
            "wc_groupmean": float(np.mean(per)), "max_share": float(max(sizes) / tot)}


# --------------------------------------------------------------------------------------- #
# the rows
# --------------------------------------------------------------------------------------- #

def load_arm(root, scores_root, tag, arm, hold_thr):
    """One arm's buffers as a dict of per-key records, with the score columns joined on.

    `hold_thr` is `100 * vo_critic_hold`; a row is held out iff `code < hold_thr`, and the code
    is a bijective function of `obs` (`VoRecorder.hold_code`, gate H below), so a GROUP never
    straddles the split — and the critic's own loss reads the training side only
    (`vo_critic_terms`), so a held-out cell is held out from the critic too.

    `label_is_y` marks the buffers where the world's verdict IS the filed column: every filed
    buffer (the filed `|y` is the trajectory's own verdict) and the probe buffers of a
    WORLD-graded arm (no `|yw` column, because there is nothing to keep apart).  The `proj`
    reader is the filed column, so on those buffers it is the label and is not reported.
    """
    d = os.path.join(root, tag, arm)
    z = np.load(os.path.join(d, "vo_rows.npz"))
    meta = json.load(open(os.path.join(d, "vo_rows_meta.json")))
    sp = os.path.join(scores_root, tag, arm, "within_scores.npz")
    sc = np.load(sp) if os.path.isfile(sp) else None
    have = set(sc.files) if sc is not None else set()
    out = {}
    for k, mt in sorted(meta.items()):
        which = k.split(":", 1)[0]
        obs = z[f"{k}|obs"]
        has_yw = (which == "probe" and f"{k}|yw" in z.files)
        lab = (z[f"{k}|yw"].astype(np.float64) if has_yw
               else (z[f"{k}|y"] > 0.5).astype(np.float64))
        rec = {"which": which, "level": int(mt["level"]), "node": int(mt["node"]),
               "span": int(mt["span"]), "blk0": int(mt["blk0"]),
               "slot_id": int(mt["slot_id"]), "n": int(mt["n"]),
               "n_buffer": int(mt["n_buffer"]), "label_is_y": (not has_yw),
               "obs": obs, "write": z[f"{k}|write"],
               "y": z[f"{k}|y"].astype(np.float64), "lab": lab,
               "dp": z[f"{k}|dp"].astype(np.float64),
               "code": z[f"{k}|code"].astype(np.int64),
               "unif": (z[f"{k}|unif"] > 0.5 if f"{k}|unif" in z.files
                        else np.ones(obs.shape[0], bool))}
        rec["hold"] = rec["code"] < hold_thr
        for col, src in (("crit", f"{k}|crit"), ("pw", f"{k}|pw"), ("pr", f"{k}|pr"),
                         ("pf", f"{k}|pf"), ("gw", f"{k}|gw"), ("gf", f"{k}|gf")):
            rec[col] = (sc[src].astype(np.float64) if src in have
                        else np.full(obs.shape[0], np.nan))
        out[k] = rec
    return out


def group_index(obs):
    """The groups of one buffer: index arrays, one per distinct `obs` row, insertion-ordered."""
    g = collections.defaultdict(list)
    for i in range(obs.shape[0]):
        g[obs[i].tobytes()].append(i)
    return [np.asarray(v, np.int64) for v in g.values()]


def bank_facts(root, tag, arm, s_branch=2):
    """THE PROJECTION'S OWN FITTING DIET, read off `vo_bank.npz`.

    The run's readout is fit on this buffer and nothing else.  The question this node asks is
    whether a reader ranks the candidates of ONE context, so the fact that decides what the
    readout could ever have learned is how often the bank holds TWO DIFFERENT CANDIDATES AT THE
    SAME CONTEXT with different verdicts.  The bank stores the post-write configuration `x`, the
    root `r` and the slot (`blk`, `spn`), so the context key is `x` with the slot's span masked
    out, beside the root and the slot.  Returns None where the arm banked no buffer (the
    overtone comparators predate it).
    """
    p_ = os.path.join(root, tag, arm, "vo_bank.npz")
    if not os.path.isfile(p_):
        return None
    z = np.load(p_)
    x, r, blk, spn, y = z["x"], z["r"], z["blk"], z["spn"], z["y"]
    g = collections.defaultdict(list)
    for i in range(x.shape[0]):
        b0, sp = int(blk[i]), int(spn[i])
        xx = x[i].copy()
        xx[b0 * s_branch:(b0 + sp) * s_branch] = -1
        g[(int(r[i]), b0, sp, xx.tobytes())].append(i)
    g2 = both = 0
    for v in g.values():
        if len(v) < 2:
            continue
        g2 += 1
        yy = y[np.asarray(v)]
        if yy.max() > 0.5 and yy.min() <= 0.5:
            both += 1
    src = collections.Counter(z["src"].tolist())
    return {"n": int(x.shape[0]), "src": {str(k): int(v) for k, v in sorted(src.items())},
            "contexts": len(g), "g2": g2, "both": both, "y_mean": float(y.mean()),
            "hold_share": float((z["h"] < 0.1).mean())}


# --------------------------------------------------------------------------------------- #
# the gates and the structure facts that live on the rows
# --------------------------------------------------------------------------------------- #

def row_facts(bufs, unif_only=False):
    """Everything the rows say about whether the question can be asked, before any AUC.

    Per (diet, level): the group-size distribution, how many groups carry both world classes,
    and the pair count.  Per diet: the hold-code gate, the one-candidate-per-group count (the
    filed side's degeneracy), the repeated-(obs, candidate) cells and how many of them
    DISAGREE on the world's verdict — the measure of how far `obs` falls short of pinning the
    configuration the grader actually scored, since it scored `fin (+) candidate` and `fin` is
    not in the dump.
    """
    per = collections.defaultdict(lambda: {"rows": 0, "sizes": [], "both": 0, "pairs": 0,
                                           "hold_both": 0})
    marg = collections.defaultdict(lambda: {"rows": 0, "code_viol": 0, "g2": 0, "onecand": 0,
                                            "dupcell": 0, "discord": 0, "dp_nonfinite": 0,
                                            "const_reader": 0})
    for k, b in bufs.items():
        wch, lv = b["which"], b["level"]
        keep = b["unif"] if unif_only else np.ones(b["obs"].shape[0], bool)
        obs, wr, lab = b["obs"][keep], b["write"][keep], b["lab"][keep]
        code, dp, crit = b["code"][keep], b["dp"][keep], b["crit"][keep]
        hold = b["hold"][keep]
        m, p = marg[wch], per[(wch, lv)]
        m["rows"] += obs.shape[0]
        p["rows"] += obs.shape[0]
        m["dp_nonfinite"] += int((~np.isfinite(dp)).sum())
        seen = {}
        for i in range(obs.shape[0]):
            t = obs[i].tobytes()
            if t in seen:
                m["code_viol"] += int(seen[t] != code[i])
            else:
                seen[t] = code[i]
        for idxs in group_index(obs):
            p["sizes"].append(int(idxs.size))
            if idxs.size < 2:
                continue
            m["g2"] += 1
            cells = collections.defaultdict(list)
            for i in idxs:
                cells[wr[i].tobytes()].append(i)
            m["onecand"] += int(len(cells) == 1)
            for ii in cells.values():
                if len(ii) >= 2:
                    m["dupcell"] += len(ii)
                    yy = lab[np.asarray(ii)]
                    if yy.min() != yy.max():
                        m["discord"] += len(ii)
            dpg, crg = dp[idxs], crit[idxs]
            c1 = (np.nanmax(dpg) - np.nanmin(dpg) <= 1e-9) if np.isfinite(dpg).any() else True
            c2 = (np.nanmax(crg) - np.nanmin(crg) <= 1e-9) if np.isfinite(crg).any() else True
            m["const_reader"] += int(bool(c1 and c2))
            y = lab[idxs]
            if y.max() > 0.5 and y.min() <= 0.5:
                p["both"] += 1
                p["pairs"] += int((y > 0.5).sum() * (y <= 0.5).sum())
                p["hold_both"] += int(bool(hold[idxs[0]]))
    out_per = {}
    for (wch, lv), d in per.items():
        sz = np.asarray(d["sizes"]) if d["sizes"] else np.asarray([0])
        out_per[f"{wch}|{lv}"] = {
            "rows": d["rows"], "groups": int(sz.size),
            "g2": int((sz >= 2).sum()), "both": d["both"], "hold_both": d["hold_both"],
            "pairs": d["pairs"], "mean": float(sz.mean()),
            "p50": float(np.percentile(sz, 50)), "p90": float(np.percentile(sz, 90)),
            "max": int(sz.max())}
    return {"per_level": out_per, "marginal": dict(marg)}


# --------------------------------------------------------------------------------------- #
# the reduction
# --------------------------------------------------------------------------------------- #

def reduce_arm(bufs, w, min_group=2, unif_only=False):
    """Per (diet, level, split), the within-context and pooled AUC of every reader.

    The analysis set is the rows of groups with >= `min_group` rows AND both world classes,
    restricted to rows whose prior `dp` is finite (the `n_on_table` shortfall).  Restricting
    FIRST and grouping after would change the groups, so the filter is applied to the rows and
    the group is re-tested for both classes afterwards.
    """
    cells = collections.defaultdict(list)
    label_is_y = {}
    for k, b in bufs.items():
        wch, lv = b["which"], b["level"]
        label_is_y[wch] = label_is_y.get(wch, False) or b["label_is_y"]
        keep = np.isfinite(b["dp"]) & (b["unif"] if unif_only else True)
        for idxs in group_index(b["obs"]):
            idxs = idxs[keep[idxs]]
            if idxs.size < min_group:
                continue
            y = b["lab"][idxs]
            if y.max() <= 0.5 or y.min() > 0.5:
                continue
            split = "hold" if bool(b["hold"][idxs[0]]) else "train"
            rec = {"y": b["y"][idxs], "dp": b["dp"][idxs], "crit": b["crit"][idxs],
                   "comp": np_compose(b["dp"][idxs], b["crit"][idxs], w),
                   "pw": b["pw"][idxs], "pr": b["pr"][idxs], "pf": b["pf"][idxs],
                   "gw": b["gw"][idxs], "gf": b["gf"][idxs], "lab": y}
            cells[(wch, lv, split)].append(rec)
            cells[(wch, "all", split)].append(rec)
    out = {}
    for key, gs in cells.items():
        cat = {c: np.concatenate([g[c] for g in gs]) for c in
               ("y", "dp", "crit", "comp", "pw", "pr", "pf", "gw", "gf", "lab")}
        off, groups = 0, []
        for g in gs:
            m = g["lab"].shape[0]
            groups.append(np.arange(off, off + m))
            off += m
        row = {"n_groups": len(gs), "n_rows": int(cat["lab"].shape[0]),
               "base": float(cat["lab"].mean()),
               "label_is_y": bool(label_is_y.get(key[0], False))}
        for rd in READERS:
            sc = cat[RD_COL[rd]]
            if rd == "proj" and row["label_is_y"]:
                row[rd] = None          # the filed column IS the label here; not a reader
                continue
            if not np.isfinite(sc).any():
                row[rd] = None
                continue
            d = wc_auc(sc, cat["lab"], groups)
            d["pooled"] = np_auc(sc, cat["lab"])
            d["n_finite"] = int(np.isfinite(sc).sum())
            row[rd] = d
        out[f"{key[0]}|{key[1]}|{key[2]}"] = row
    return out


# --------------------------------------------------------------------------------------- #
# the table of record
# --------------------------------------------------------------------------------------- #

def fmt(v, n=3):
    return "  --  " if v is None else f"{v:.{n}f}"


HEAD = """\
======================================================================================================================
WITHIN-CONTEXT CANDIDATE DISCRIMINATION -- the six soundboard arms, plus overtone's two
frozen-plant dumps, on banked rows.  CPU, no loop, nothing paid.
======================================================================================================================

THE QUESTION.  Every AUC in aliquot / duplex / soundboard is POOLED over rows, so it mixes "is
this context solvable at all" with "which of this context's candidates is the one to write".
Only the second is the cell the chooser occupies.  A GROUP here is (slot key, obs bytes): the
same masked pre-write context at the same slot.  WC is the AUC restricted to (positive,
negative) pairs INSIDE one group, pair-weighted.  P beside it is the ORDINARY POOLED AUC ON THE
SAME ROWS, so the gap is visible.  Only groups with >= 2 rows, both world classes and a finite
prior enter; the hold code is a bijection of obs, so a group never straddles the audit split and
the critic's loss never saw a `hold` row.

THE READERS, and what each one saw:
  proj  the grader's filed probability -- the projection's own p.  Computed AT FILE TIME by the
        readout of that cycle, over `fin (+) candidate` (the trajectory's FINAL configuration
        with the candidate substituted).  `fin` is not in the dump, so this column cannot be
        recomputed and is not on the same input as the three below.  On a WORLD-graded arm
        (both overtone dumps) and on every FILED buffer the filed column IS the world's
        verdict, so `proj` would be the label; it is left blank there.
  dp    the executor's max-sum DP score of the candidate through the FINAL core, already
        divided by the span (sb_dump_rows).  The chooser's prior.
  crit  the critic's logit on SN.trunk(core, obs) for the row's own candidate, FINAL core and
        FINAL critic.  The chooser's judge.  NB the two lineages trained it on different
        targets: on the soundboard arms the probe rows carry the PROJECTION's probability, on
        the overtone arms the WORLD's verdict.
  comp  z(dp) + w*z(crit), standardised WITHIN the group exactly as vo_compose does.  THE
        CHOOSER.  z-scoring is monotone inside a group, so dp's and crit's WC columns are
        unchanged by it and only this one is new.
  pw    an ADDED CONTROL, not part of the run: a duplex-protocol ridge-logistic readout of the
        FINAL trunk over `obs (+) candidate` (trunk_features' masked post-write read), fit by
        IRLS on THIS buffer's own TRAINING rows against the world's verdict, scored on the
        held-out ones.  It is fit PER SLOT, so its pooled column carries 30-60 different
        intercepts and is NOT comparable to the global readers' pooled column; only its WC
        column is.
  pr    THE FLOOR: `pw` again, over a NEVER-TRAINED twin of the same architecture
        (manual_seed(20260918), the in-loop `vo_pj_rand_seed`, minted after the trained core is
        loaded).  `obs (+) candidate` puts the candidate's own level-1 features in the input, so
        a ridge over ANY feature map can rank candidates by their identity alone; this column is
        what says whether the plant's TRAINING matters at this cell.
  pf    THE DIET CONTROL: the same estimator, trunk, input and moment as `pw`, fit instead on
        the matching FILED buffer's training rows.  The run's own projection is fit on the bank
        and the bank is `src = 0` only -- the learner's own experienced configurations, where a
        deterministic argmax chooser writes ONE candidate per context.  `pw` minus `pf` is what
        the DIET alone is worth, with everything else held.
  gw/gf THE SAME DIET CONTRAST AT THE PROJECTION'S OWN FORM: ONE GLOBAL readout with
        `VoProjBank`'s design -- its own masked read, all-block AND span pooling, a group
        one-hot and its INTERACTION with the standardised features, IRLS with the ridge chosen
        on a train-internal split (`preplay`'s re-implementation, gated elementwise at
        0.000e+00 against `VoProjBank` itself) -- fit on the probe diet (`gw`) and on the filed
        diet (`gf`), both scored on the SAME held-out probe rows.
        ONE FORCED DEVIATION: `VoProjBank` groups by the ROOT, 1737 columns, one readout per
        GOAL.  THE DUMP CARRIES NO ROOT (the recorder banks it only into `vo_bank.npz`, whose
        rows are final configurations and cannot be joined to a probe row), so the grouping
        variable here is the slot's LEVEL, 4 groups, 965 columns.  Every gw/gf number carries
        that substitution.

  `pw`, `pr`, `pf`, `gw` and `gf` exist only on HELD-OUT rows, by construction.  `pw`, `pr` and
  `pf` are fit PER SLOT and their pooled columns carry 28-60 intercepts, so only their WC
  columns compare with the global readers'; `gw` and `gf` are single global fits and their
  pooled columns do compare.
"""


def render(res, facts, meta):
    L = [HEAD]

    L.append("-" * 118)
    L.append("[A] GROUP STRUCTURE -- what the rows can be asked, per arm, diet and level")
    L.append("-" * 118)
    L.append(f"{'arm':<26}{'diet':<7}{'L':<4}{'rows':>8}{'groups':>8}{'g>=2':>8}"
             f"{'both':>7}{'both/hold':>10}{'pairs':>8}{'mean':>7}{'p50':>6}{'p90':>6}"
             f"{'max':>6}")
    for nm in meta["order"]:
        for wch in ("probe", "filed"):
            for lv in ("2", "3", "4", "5"):
                d = facts[nm]["per_level"].get(f"{wch}|{lv}")
                if d is None:
                    continue
                L.append(f"{nm:<26}{wch:<7}{lv:<4}{d['rows']:>8}{d['groups']:>8}"
                         f"{d['g2']:>8}{d['both']:>7}{d['hold_both']:>10}{d['pairs']:>8}"
                         f"{d['mean']:>7.2f}{d['p50']:>6.0f}{d['p90']:>6.0f}{d['max']:>6}")
        L.append("")
    L.append("`both` is groups of size >= 2 carrying BOTH world classes -- the groups that can")
    L.append("be asked anything at all; `both/hold` is how many of those are on the held-out")
    L.append("side.  `pairs` counts (positive, negative) pairs inside those groups.")
    L.append("")

    L.append("-" * 118)
    L.append("[A2] WHAT A GROUP IS NOT -- the two ways `(slot, obs)` falls short of "
             "`one context, its candidate set`")
    L.append("-" * 118)
    L.append(f"{'arm':<26}{'diet':<7}{'rows':>8}{'g>=2':>8}{'1cand/grp':>11}"
             f"{'all-tied grp':>13}{'dupcell rows':>14}{'discordant':>12}"
             f"{'dp nonfin':>11}{'codeviol':>10}")
    for nm in meta["order"]:
        for wch in ("probe", "filed"):
            m = facts[nm]["marginal"].get(wch)
            if m is None:
                continue
            L.append(f"{nm:<26}{wch:<7}{m['rows']:>8}{m['g2']:>8}{m['onecand']:>11}"
                     f"{m['const_reader']:>13}{m['dupcell']:>14}{m['discord']:>12}"
                     f"{m['dp_nonfinite']:>11}{m['code_viol']:>10}")
    L.append("")
    L.append("`1cand/grp`   groups of size >= 2 in which EVERY row carries the same candidate.")
    L.append("              On the filed diet this is essentially every group: the chooser is")
    L.append("              argmax(compose) with vo_eps = 0, so one context gets one write and")
    L.append("              the filed diet holds NO within-context candidate contrast at all.")
    L.append("`all-tied grp` groups on which dp AND crit are constant across the group, i.e. on")
    L.append("              which no reader built from (obs, candidate) can rank anything.")
    L.append("`dupcell rows` rows sharing an (obs, candidate) cell with another row.")
    L.append("`discordant`  of those, rows whose cell does NOT agree on the world's verdict.")
    L.append("              The grader scored `fin (+) candidate` and `fin` is not in the dump,")
    L.append("              so a group is `same masked context at this slot`, not `same")
    L.append("              trajectory`; this column is how much that costs, measured.")
    L.append("`codeviol`    rows whose hold code disagrees with another row's at the same obs.")
    L.append("              Zero everywhere: the split cannot cut a group.")
    L.append("")

    for wch in ("probe", "filed"):
        L.append("-" * 118)
        L.append(f"[B] WITHIN-CONTEXT AUC (WC) beside the POOLED AUC on the same rows (P) "
                 f"-- {wch} rows")
        L.append("-" * 118)
        L.append(f"{'arm':<26}{'L':<4}{'split':<7}{'grp':>6}{'rows':>7}{'pairs':>8}"
                 f"{'base':>7}"
                 f"{'projWC':>9}{'/P':>8}{'dpWC':>9}{'/P':>8}{'critWC':>9}{'/P':>8}"
                 f"{'compWC':>9}{'/P':>8}")
        for nm in meta["order"]:
            r = res[nm]
            any_row = False
            for lv in ("2", "3", "4", "5", "all"):
                for split in ("hold", "train"):
                    row = r.get(f"{wch}|{lv}|{split}")
                    if row is None:
                        continue
                    any_row = True
                    pairs = (row["dp"] or {}).get("pairs", 0)
                    line = (f"{nm:<26}{lv:<4}{split:<7}{row['n_groups']:>6}"
                            f"{row['n_rows']:>7}{pairs:>8}{row['base']:>7.3f}")
                    for rd in RUN_RD:
                        d = row.get(rd)
                        line += (f"{'':>9}{'':>8}" if d is None
                                 else f"{fmt(d['wc']):>9}{fmt(d['pooled']):>8}")
                    L.append(line)
            if any_row:
                L.append("")
    L.append("A BLANK `projWC` cell is not a missing number: the filed column IS the world's")
    L.append("verdict on those rows (every filed buffer; both overtone probe buffers), so the")
    L.append("reader would be the label.  The five fitted controls are in [D].")
    L.append("")

    L.append("-" * 118)
    L.append("[D] THE FITTED CONTROLS -- held-out probe rows only, the cells [B] leaves out")
    L.append("-" * 118)
    L.append(f"{'arm':<26}{'L':<4}{'grp':>6}{'pairs':>8}"
             + "".join(f"{rd + 'WC':>8}{'/P':>7}" for rd in FIT_RD))
    for nm in meta["order"]:
        r = res[nm]
        any_row = False
        for lv in ("2", "3", "4", "5", "all"):
            row = r.get(f"probe|{lv}|hold")
            if row is None:
                continue
            any_row = True
            line = (f"{nm:<26}{lv:<4}{row['n_groups']:>6}"
                    f"{(row['dp'] or {}).get('pairs', 0):>8}")
            for rd in FIT_RD:
                d = row.get(rd)
                line += (f"{'':>8}{'':>7}" if d is None
                         else f"{fmt(d['wc']):>8}{fmt(d['pooled']):>7}")
            L.append(line)
        if any_row:
            L.append("")
    L.append("`pw` probe diet / `pf` filed diet, duplex's per-slot form; `pr` the never-trained")
    L.append("twin on the probe diet; `gw` probe diet / `gf` filed diet at the projection's own")
    L.append("global form (level-grouped, see the legend).  The diet contrasts are pw-vs-pf and")
    L.append("gw-vs-gf; the trunk-training contrast is pw-vs-pr.")
    L.append("")

    L.append("-" * 118)
    L.append("[C] THE WEIGHTING -- all levels, held-out.  WCp is pair-weighted (the headline);")
    L.append("    WCg weights every group alike; maxsh is the share of all pairs contributed")
    L.append("    by the single biggest group.")
    L.append("-" * 118)
    for blk, rds in (("the run's own readers", RUN_RD), ("the fitted controls", FIT_RD)):
        L.append(f"  -- {blk} --")
        L.append(f"{'arm':<26}{'diet':<7}{'grp':>6}{'pairs':>8}{'maxsh':>7}"
                 + "".join(f"{rd + ' WCp':>10}{'WCg':>7}" for rd in rds))
        for nm in meta["order"]:
            for wch in ("probe", "filed"):
                row = res[nm].get(f"{wch}|all|hold")
                if row is None:
                    continue
                ref = row.get("dp") or {}
                line = (f"{nm:<26}{wch:<7}{row['n_groups']:>6}{ref.get('pairs', 0):>8}"
                        f"{(ref.get('max_share') or 0):>7.3f}")
                for rd in rds:
                    d = row.get(rd)
                    line += (f"{'':>10}{'':>7}" if d is None
                             else f"{fmt(d['wc']):>10}{fmt(d['wc_groupmean']):>7}")
                L.append(line)
        L.append("")

    L.append("-" * 118)
    L.append("[E] THE PROJECTION'S OWN FITTING DIET -- vo_bank.npz, the only rows the run's")
    L.append("    readout was ever fit on.  A context here is (root, slot, configuration with")
    L.append("    the slot's span masked).")
    L.append("-" * 118)
    L.append(f"{'arm':<26}{'rows':>8}{'src':>10}{'contexts':>10}{'ctx>=2':>9}"
             f"{'ctx>=2+both':>13}{'y mean':>9}{'hold':>7}")
    for nm in meta["order"]:
        b = meta["bank"].get(nm)
        if b is None:
            continue
        L.append(f"{nm:<26}{b['n']:>8}"
                 f"{'/'.join(f'{k}:{v}' for k, v in b['src'].items()):>10}"
                 f"{b['contexts']:>10}{b['g2']:>9}{b['both']:>13}"
                 f"{b['y_mean']:>9.3f}{b['hold_share']:>7.3f}")
    L.append("")
    L.append("`src` is the bank's own source code: 0 is the FILED diet -- the learner's own")
    L.append("experienced final configurations.  Every soundboard arm's bank is src 0 only.")
    L.append("The overtone comparators banked no readout buffer and are absent by construction.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding <tag>/<arm>/vo_rows.npz")
    ap.add_argument("--scores", required=True, help="dir holding <tag>/<arm>/within_scores.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="sb", choices=("sb", "ov", "all"))
    ap.add_argument("--unif-arms", default="ov_s2",
                    help="comma-separated tags to ALSO reduce on the uniform probe draw only "
                         "(ov_s2's probe is 3/4 disagreement-drawn, so its candidate set "
                         "inside a context is not a uniform sample of the book)")
    ap.add_argument("--outfile", default="within_reduction.txt")
    a = ap.parse_args()
    arms = (SB_ARMS if a.arms == "sb" else OV_ARMS if a.arms == "ov" else SB_ARMS + OV_ARMS)
    unif = {x for x in a.unif_arms.split(",") if x}
    res, facts, order, wmap, banks = {}, {}, [], {}, {}
    for tag, arm in arms:
        d = os.path.join(a.root, tag, arm)
        if not os.path.isfile(os.path.join(d, "vo_rows.npz")):
            print(f"  [skip] {tag}/{arm}: no rows")
            continue
        cfg = json.load(open(os.path.join(d, "results.json")))["config"]
        hold_thr = int(round(100 * float(cfg["vo_critic_hold"])))
        w = float(cfg.get("vo_w", 1.0))
        bufs = load_arm(a.root, a.scores, tag, arm, hold_thr)
        for lbl, uo in ((f"s{SEED_OF[tag]}/{arm}", False),
                        *([(f"s{SEED_OF[tag]}/{arm} [unif]", True)] if tag in unif else [])):
            order.append(lbl)
            wmap[lbl] = w
            facts[lbl] = row_facts(bufs, unif_only=uo)
            banks[lbl] = bank_facts(a.root, tag, arm)
            res[lbl] = reduce_arm(bufs, w, unif_only=uo)
            print(f"  [done] {lbl}  hold_thr {hold_thr}  w {w}")
    meta = {"order": order, "w": wmap, "bank": banks}
    os.makedirs(a.out, exist_ok=True)
    txt = render(res, facts, meta)
    with open(os.path.join(a.out, a.outfile), "w") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(a.out, a.outfile.replace(".txt", ".json")), "w") as fh:
        json.dump({"res": res, "facts": facts, "w": wmap, "order": order,
                   "bank": banks}, fh,
                  separators=(",", ":"), default=str)
    print(txt)


if __name__ == "__main__":
    main()
