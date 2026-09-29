"""[scordatura] The reductions this node needs, local and CPU-only. Facts only; the orchestrator
interprets.

  --dose       the pre-check (DESIGN §2): the junk floor's own record against `st_gn_yk`'s at the
               same cycles — the junk rows mined, keys at support both ways (the miner's and its
               solve-fed shadow's), junk-only keys and their true-table mask, the walk's offer cap,
               and the fire survey (every at-support key fired single-entry at its own level's
               cell, world-graded and read). Writes `figures/<tag>_dose.txt`.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/reduce_scordatura.py --dose --tag sc_p1
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RB_FIG = os.path.join(os.path.dirname(HERE), "rubato", "figures")
REF_OF_SEED = {0: ("rb_s1", "st_gn_yk"), 2: ("rb_s2", "st_gn_yk")}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def _q(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "—"
    return "/".join(f"{xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]:g}" for q in qs)


def _sgn(x, eps=0.0):
    if x is None:
        return "na"
    return "+" if x > eps else ("-" if x < -eps else "0")


def dose_section(r, ref, L, every=5):
    """The junk floor `r` against the ungated reference `ref`, cycle by cycle."""
    log, rlog = r["log"], ref["log"]
    mj = log.get("mj") or []
    sup = int(r["config"]["mine_support"])
    n = len(mj)
    L.append("[D1] THE JUNK MINED, per cycle (instances posed 64; `take` = the donor's solved rows, "
             "`junk` = unsolved rows appended)")
    by_era = defaultdict(list)
    for j, q in enumerate(mj):
        by_era[log["era"][j]].append(q)
    L.append(f"    {'era':>3} {'cycles':>6} {'solved/cycle q0/25/50/75/100':>30} "
             f"{'take':>14} {'junk':>22} {'junk:take':>9}")
    for e in sorted(by_era):
        qs = by_era[e]
        jt = sum(q["n_junk"] for q in qs) / max(1, sum(q["n_take"] for q in qs))
        L.append(f"    {e:>3} {len(qs):>6} {_q([q['n_sol'] for q in qs]):>30} "
                 f"{_q([q['n_take'] for q in qs]):>14} {_q([q['n_junk'] for q in qs]):>22} "
                 f"{jt:>9.2f}")
    tot_j, tot_t = sum(q["n_junk"] for q in mj), sum(q["n_take"] for q in mj)
    L.append(f"    total over c1..c{n}: {tot_t} solved rows, {tot_j} junk rows "
             f"({tot_j / max(1, tot_t):.2f} junk per solved row)")
    ok = Counter()
    for q in mj:
        for k, v in (q.get("ok") or {}).items():
            ok[(k, bool(v))] += 1
    L.append("    in-run checks: " + ", ".join(
        f"{k} {ok[(k, True)]}/{ok[(k, True)] + ok[(k, False)]}" for k in ("J1", "J2", "J3", "J4")))
    # the rows the class map dropped (a half with no legal derivation)
    last = mj[-1] if mj else {}
    L.append("    junk rows seen / dropped by the class map (no legal class for a half), cumulative "
             f"at c{n}: " + "  ".join(
                 f"L{lv} {v.get('junk_rows')}/{v.get('junk_dropped')}"
                 for lv, v in sorted((last.get("lv") or {}).items(), key=lambda kv: int(kv[0]))
                 if v.get("junk_rows")) + "  | panel " + "  ".join(
                 f"L{lv} {v.get('junk_rows')}/{v.get('junk_dropped')}"
                 for lv, v in sorted((last.get("panel") or {}).items(), key=lambda kv: int(kv[0]))))

    L.append("")
    L.append("[D2] KEYS AT SUPPORT, per cycle: the committable miner (what the walk offers) and the "
             "panel (what defines a key's consumers)")
    L.append("    here = this arm's miner; shadow = the same arm's solve-fed shadow (the "
             "solve-filtered rule on this arm's own history); st_gn_yk = the reference arm's miner at "
             "the same cycle; jo = junk-only (at support here, not in the shadow); T/F = the "
             "true-table mask (tf 1 / tf 0)")
    rm = rlog.get("miner") or []
    robs = ref.get("obs_hist") or {}
    cols = [("lv", "2"), ("panel", "3"), ("panel", "4"), ("panel", "5"), ("lv", "5")]
    hdr = f"    {'c':>4} " + " ".join(f"{('L' + lv + (' pan' if src == 'panel' else ' min')):>30}"
                                   for src, lv in cols)
    L.append(hdr)
    L.append(f"    {'':>4} " + " ".join(f"{'here(T/F) shadow jo(T/F) st_gn_yk':>30}" for _ in cols))
    for j in range(n):
        c = mj[j]["c"]
        if c % every and c != n and c != 1:
            continue
        cells = []
        for src, lv in cols:
            v = (mj[j].get(src) or {}).get(lv) or {}
            if src == "lv":
                rv = ((rm[j] if j < len(rm) else {}) or {}).get(lv) or {}
                rv = (rv.get("n_at_support") or {}).get(str(sup))
            else:
                rv = (robs.get(lv) or [None] * (j + 1))[j] if j < len(robs.get(lv) or []) else None
            cells.append(f"{v.get('at', '—')}({v.get('at_t', '·')}/{v.get('at_f', '·')}) "
                         f"{v.get('at_sv', '—')} {v.get('jo', '—')}({v.get('jo_t', '·')}/"
                         f"{v.get('jo_f', '·')}) {rv if rv is not None else '—'}")
        L.append(f"    {c:>4} " + " ".join(f"{x:>30}" for x in cells))

    L.append("")
    L.append("[D3] THE WALK (admission, `none`): every pass, per level — offered (cap 8), pending "
             "(at support, undecided, buildable), whether the cap bound, and the junk-only keys "
             "among them; the admitted table's size and row precision; `st_gn_yk`'s pass at the "
             "same cycle beside it")
    rpass = {p["cycle"]: p for p in ((ref.get("adm") or {}).get("passes") or [])}
    L.append(f"    {'c':>4} {'L':>2} {'offer':>5} {'pend':>5} {'cap':>4} {'offer_jo':>8} "
             f"{'pend_jo':>7} {'keys adm':>8} {'prec':>6} | {'ref offer':>9} {'ref pend':>8} "
             f"{'ref keys':>8} {'ref prec':>8}")
    n_bind = 0
    for p in (r.get("adm") or {}).get("passes") or []:
        for lv in p["levels"]:
            m = lv.get("mj") or {}
            rl = next((x for x in (rpass.get(p["cycle"]) or {}).get("levels", [])
                       if x["level"] == lv["level"]), {})
            n_bind += int(bool(m.get("cap_binds")))
            pr = lv.get("tab_precision")
            rpr = rl.get("tab_precision")
            L.append(f"    {p['cycle']:>4} {lv['level']:>2} {lv['n_offered']:>5} "
                     f"{lv['n_pending_keys']:>5} {('yes' if m.get('cap_binds') else 'no'):>4} "
                     f"{m.get('n_offered_jo', '—'):>8} {m.get('n_pending_jo', '—'):>7} "
                     f"{lv['n_admitted_keys']:>8} {('—' if pr is None else f'{pr:.3f}'):>6} | "
                     f"{rl.get('n_offered', '—'):>9} {rl.get('n_pending_keys', '—'):>8} "
                     f"{rl.get('n_admitted_keys', '—'):>8} "
                     f"{('—' if rpr is None else f'{rpr:.3f}'):>8}")
    L.append(f"    passes at which the offer cap of 8 bound: {n_bind}")
    # the admitted keys by junk-only status, at the stop
    adm_now = (mj[-1].get("lv") or {}) if mj else {}
    L.append("    admitted keys at the stop, per walked level: "
             + "  ".join(f"L{lv} {v.get('adm')} (junk-only now {v.get('adm_jo')}, T {v.get('adm_t')}"
                         f" / F {v.get('adm_f')})"
                         for lv, v in sorted(adm_now.items(), key=lambda kv: int(kv[0]))
                         if v.get("adm") is not None))

    # the walk's own fires: base (the admitted table so far) + the candidate's rows, on the gate
    # pool at the era's cell — the only fire on the record where a key CAN hurt (the single-entry
    # pool below starts at success 0, so a single-entry gain cannot be negative there)
    L.append("")
    L.append("[D3b] THE WALK'S OWN FIRES, base-plus-one on the gate pool (192 of the era's cell): "
             "world delta = e_base - e_trial (> 0 helps, < 0 HARMS), per offered key by class")
    L.append("    jo = junk-only at its offer, sv = at support in the shadow too; T/F = true mask")
    agg = defaultdict(list)
    for p in (r.get("adm") or {}).get("passes") or []:
        for lv in p["levels"]:
            for st in lv["steps"]:
                cls, tr = kcls(st.get("mj"))
                agg[(lv["level"], cls, tr)].append(st["e_cur"] - st["e_trial"])
    L.append(f"    {'L':>2} {'class':>5} {'true':>4} {'n':>3} {'world -/0/+':>12} "
             f"{'delta q0/25/50/75/100':>40}")
    for (lvl, cls, tr), ds in sorted(agg.items()):
        sw = Counter(_sgn(x) for x in ds)
        L.append(f"    {lvl:>2} {cls:>5} {tr:>4} {len(ds):>3} {sw['-']:>4}/{sw['0']}/{sw['+']:<5} "
                 f"{_q([round(x, 4) for x in ds]):>40}")
    rref = defaultdict(list)
    for p in ((ref.get("adm") or {}).get("passes") or []):
        if p["cycle"] > n:
            continue
        for lv in p["levels"]:
            for st in lv["steps"]:
                rref[lv["level"]].append(st["e_cur"] - st["e_trial"])
    for lvl, ds in sorted(rref.items()):
        sw = Counter(_sgn(x) for x in ds)
        L.append(f"    st_gn_yk through c{n}: L{lvl} {len(ds)} offered, world -/0/+ "
                 f"{sw['-']}/{sw['0']}/{sw['+']}, delta {_q([round(x, 4) for x in ds])}")

    sv = (r.get("sc") or {}).get("survey")
    L.append("")
    L.append("[D4] THE FIRE SURVEY at the stop: every at-support key fired SINGLE-ENTRY at its own "
             "level's cell (pp1's fire), world gain = success fired - success unfired on the same "
             "pool, read gain = the shaped projection's level likewise")
    if not sv or sv.get("error"):
        L.append(f"    (no survey: {None if not sv else sv.get('error')})")
        return
    L.append(f"    c{sv['cycle']} era {sv['era']}, cap {sv['cap']} per class per source and level, "
             f"{sv.get('t_s')} s")
    for lv in sv["levels"]:
        L.append(f"    L{lv['level']} node {lv['node']}: base success {lv.get('base_succ')}, "
                 f"base read {None if lv.get('base_p') is None else round(lv['base_p'], 4)}; "
                 f"at support: miner {lv.get('n_at_miner')} (junk-only {lv.get('n_jo_miner')}), "
                 f"panel {lv.get('n_at_panel')} (junk-only {lv.get('n_jo_panel')})")
        groups = defaultdict(list)
        for k in lv["keys"]:
            cls, tr = kcls(k)
            groups[(k["src"], cls, tr)].append(k)
        L.append(f"      {'src':>5} {'class':>5} {'true':>4} {'n':>3} {'unbuild':>7} "
                 f"{'world -/0/+':>12} {'world gain q0/25/50/75/100':>34} {'read -/0/+':>11} "
                 f"{'served':>6}")
        for (src, cls, tr), ks in sorted(groups.items()):
            fired = [k for k in ks if k.get("buildable")]
            gw = [k["g_world"] for k in fired]
            gr = [k.get("g_read") for k in fired]
            sw = Counter(_sgn(x) for x in gw)
            sr = Counter(_sgn(x) for x in gr if x is not None)
            L.append(f"      {src:>5} {cls:>5} {tr:>4} {len(ks):>3} "
                     f"{sum(1 for k in ks if not k.get('buildable')):>7} "
                     f"{sw['-']:>4}/{sw['0']}/{sw['+']:<5} "
                     f"{_q([round(x, 4) for x in gw]):>34} "
                     f"{sr['-']:>3}/{sr['0']}/{sr['+']:<5} "
                     f"{sum(1 for k in ks if k.get('served')):>6}")


# =========================================================================== #
# --grade / --seedtable: the floor, the ceiling and the claim (DESIGN §4.3–§4.6)
# =========================================================================== #

SEEDS = {
    0: {"floor": ("sc_s1", "sc_gn_yk"), "pre": ("sc_p1", "sc_gn_yk"),
        "graded": (("sc_g1", "sc_gw_yk"), ("sc_g1", "sc_grd_yk")),
        "ref": ("rb_s1", "st_gn_yk"),
        "rb": (("rb_g1", "rb_gw_yk"), ("rb_g1d", "rb_grd_yk"), ("rb_g1", "rb_gr_yk"))},
    2: {"floor": ("sc_s2", "sc_gn_yk"), "pre": ("sc_p2", "sc_gn_yk"),
        "graded": (("sc_g2w", "sc_gw_yk"), ("sc_g2r", "sc_grd_yk")),
        "ref": ("rb_s2", "st_gn_yk"),
        "rb": (("rb_g2", "rb_gw_yk"), ("rb_g2", "rb_gr_yk"))},
}
ERAS = ((1, 1, 60), (2, 61, 110), (3, 111, 180), (4, 181, 192), (5, 193, 201))


def load_seed(seed):
    S = SEEDS[seed]
    arms = {}
    tf, af = S["floor"]
    arms[af] = load(os.path.join(FIG, tf, af, "results.json"))
    for tg, a in S["graded"]:
        p = os.path.join(FIG, tg, a, "results.json")
        if os.path.isfile(p):
            arms[a] = load(p)
    rt, ra = S["ref"]
    rb = {ra: load(os.path.join(RB_FIG, rt, ra, "results.json"))}
    for t, a in S["rb"]:
        p = os.path.join(RB_FIG, t, a, "results.json")
        if os.path.isfile(p):
            rb[a] = load(p)
    return arms, rb


def _nt(x):
    """JSON-normalised, every wall-clock field (`t_s`) dropped (DESIGN §5.2 of rubato)."""
    def strip(y):
        if isinstance(y, dict):
            return {k: strip(v) for k, v in y.items() if k != "t_s"}
        if isinstance(y, list):
            return [strip(v) for v in y]
        return y
    return json.dumps(strip(x), sort_keys=True)


def graded_passes(adm):
    """Every graded key-pass: (cycle, level, key tuple, record), in pass order."""
    return [(p["cycle"], lv["level"], tuple(k["key"]), k) for p in adm.get("gr_passes") or []
            for lv in p["levels"] for k in lv["keys"] if k.get("state") == "graded"]


def kcls(m):
    """A key's class from its junk columns: jo / sv, and T / F."""
    m = m or {}
    c = "jo" if m.get("jo") else ("sv" if m.get("jo") is False else "?")
    tf = m.get("tf")
    # M = MIXED: 0 < tf < 1, a class pair whose spellings are partly true (after a merge coarsens
    # a half's class, its spellings no longer share one possible set); ? = no mask
    t = "?" if tf is None else ("T" if tf == 1 else ("F" if tf == 0 else "M"))
    return c, t


def walked_keys(r):
    out = defaultdict(set)
    for p in (r.get("adm") or {}).get("passes") or []:
        for lv in p["levels"]:
            for st in lv["steps"]:
                out[lv["level"]].add(tuple(st["key"]))
    return out


def halves(k):
    return (k[:len(k) // 2], k[len(k) // 2:])


def era_err(r):
    e, er = r["log"]["e"], r["log"]["era"]
    out = {}
    for ei, lo, hi in ERAS:
        idx = [i for i, x in enumerate(er) if x == ei]
        out[ei] = (sum(e[i] for i in idx) / len(idx)) if idx else None
    return out


def last_pass_rows(r):
    """The walk's last pass row per level (keys admitted, rows kept / live, row precision)."""
    out = {}
    for p in (r.get("adm") or {}).get("passes") or []:
        for lv in p["levels"]:
            out[lv["level"]] = (p["cycle"], lv)
    return out


def section_identity(arms, L):
    L.append("[G1] PROVENANCE AND IDENTITY")
    for a, r in arms.items():
        rb = r.get("rb") or {}
        segs, cur = [], rb
        while cur:
            segs.append(cur)
            cur = cur.get("prior")
        segs.reverse()
        desc = []
        for g in segs:
            rf = g.get("resumed_from") or {}
            h = g.get("host") or {}
            desc.append(f"from c{rf.get('cycle', 0)} of {rf.get('arm', '(fresh)')}"
                        f"{' R-2 ' + str(rf.get('r2')) if rf else ''}"
                        f"{' (auto)' if g.get('auto_resumed') else ''} avx512={h.get('np_avx512f')}")
        idn = [(q.get("cycle"), q.get("identity")) for g in segs for q in (g.get("saves") or [])]
        ok = [c for c, x in idn if x and x.get("equal")]
        bad = [c for c, x in idn if x and "equal" in x and not x["equal"]]
        sc = r.get("sc") or {}
        ck = sc.get("checks") or {}
        adm = r.get("adm") or {}
        L.append(f"  {a}: {len(r['log']['cycle'])} cycles; segments: " + "; ".join(desc))
        L.append(f"      J1-J4 {[ck.get(j) for j in ('J1', 'J2', 'J3', 'J4')]} red {ck.get('red')}; "
                 f"junk rows {sc.get('n_junk')} beside {sc.get('n_sv')} solved rows; G-I equal at "
                 f"{ok}, differ at {bad}; first revocation c{adm.get('first_revoke')}; "
                 f"G-W {(adm.get('gr_checks') or {}).get('G-W', {}).get('equal') if adm.get('grade') else '—'}; "
                 f"G-S red {(adm.get('gr_checks') or {}).get('G-S_red') if adm.get('grade') else '—'}")


def section_first_div(arms, floor_name, L):
    """G-I off the log: the first cycle any per-cycle series of a graded arm differs from the
    floor's, against the arm's first revocation (the saves' hash check stops there)."""
    F = arms[floor_name]
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        rf = ((r.get("rb") or {}).get("resumed_from") or {})
        cur = r.get("rb") or {}
        while cur.get("prior"):
            cur = cur["prior"]
        c0 = int((cur.get("resumed_from") or {}).get("cycle") or 0)
        skip = {"t_cum", "prop", "vo_bill"} if adm["grade"] == "world" else set()
        n_c = len(r["log"]["cycle"])
        fd = None
        for i, c in enumerate(r["log"]["cycle"]):
            if c <= c0 or i >= len(F["log"]["cycle"]):
                continue
            for k, v in r["log"].items():
                if k in skip or k not in F["log"] or not isinstance(v, list) or len(v) != n_c \
                        or i >= len(F["log"][k]):
                    continue
                if _nt(v[i]) != _nt(F["log"][k][i]):
                    fd = (int(c), k)
                    break
            if fd:
                break
        fr = adm.get("first_revoke")
        L.append(f"  {a}: restored after c{c0}; first cycle a log series differs from the floor: "
                 f"{fd}; first revocation c{fr} "
                 f"({'OK' if fd is None or fr is None or fd[0] >= fr else 'BEFORE the first revocation'})")


def section_errors(arms, rb, L):
    L.append("")
    L.append("[G2] TASK ERROR BY ERA (mean over the era's cycles), and each arm minus st_gn_yk and "
             "minus the junk floor")
    ref = era_err(rb["st_gn_yk"])
    flo = era_err(arms["sc_gn_yk"])
    allarms = [("st_gn_yk", rb["st_gn_yk"])] + [(a, r) for a, r in rb.items() if a != "st_gn_yk"] \
        + list(arms.items())
    L.append(f"    {'arm':>10} " + " ".join(f"{'era ' + str(e):>8}" for e, _, _ in ERAS)
             + "   | minus st_gn_yk, eras 1..5                 | minus the junk floor")
    for a, r in allarms:
        ee = era_err(r)
        d0 = " ".join(f"{(ee[e] - ref[e]) if ee[e] is not None else float('nan'):+.3f}"
                      for e, _, _ in ERAS)
        d1 = " ".join(f"{(ee[e] - flo[e]) if ee[e] is not None else float('nan'):+.3f}"
                      for e, _, _ in ERAS)
        L.append(f"    {a:>10} " + " ".join(f"{ee[e] if ee[e] is not None else float('nan'):>8.4f}"
                                           for e, _, _ in ERAS) + f"   | {d0} | {d1}")


def section_coverage(arms, rb, L):
    L.append("")
    L.append("[G3] COVERAGE. The walk's keys (every key ever offered = admitted under `none`), the "
             "last pass per level (keys admitted | rows kept / live | row precision), and, on a "
             "graded arm, the floor's walked keys one level up with a half held revoked at the end")
    F = arms["sc_gn_yk"]
    fw = walked_keys(F)
    allarms = [("st_gn_yk", rb["st_gn_yk"])] + [(a, r) for a, r in rb.items() if a != "st_gn_yk"] \
        + list(arms.items())
    for a, r in allarms:
        w = walked_keys(r)
        lp = last_pass_rows(r)
        cells = []
        for lv in (2, 3, 4, 5):
            if lv in lp:
                c, x = lp[lv]
                pr = x.get("tab_precision")
                cells.append(f"L{lv} walked {len(w.get(lv, ()))}, c{c}: {x.get('n_admitted_keys')} | "
                             f"{x.get('n_kept_rows')}/{x.get('n_live_rows')} | "
                             f"{'—' if pr is None else f'{pr:.2f}'}")
        L.append(f"  {a:>10}: " + "; ".join(cells))
        adm = r.get("adm") or {}
        if adm.get("grade"):
            rv = {(int(lv), tuple(q["key"])) for lv, qs in (adm.get("revoked") or {}).items()
                  for q in qs}
            ref_w = fw if a.startswith("sc_") else walked_keys(rb["st_gn_yk"])
            nm = "the floor's" if a.startswith("sc_") else "st_gn_yk's"
            for up in (3, 4, 5):
                ks = ref_w.get(up, set())
                if ks:
                    blk = sum(1 for K in ks if any((up - 1, h) in rv for h in halves(K)))
                    L.append(f"             of {nm} {len(ks)} walked L{up} keys, a half held revoked "
                             f"at the end: {blk}")
            L.append(f"             revoked at the end: " + " ".join(
                f"L{lv} {len(qs)}" for lv, qs in sorted((adm.get("revoked") or {}).items())))


def section_served(arms, L, every=20):
    L.append("")
    L.append("[G4] THE SERVED TABLE OVER CYCLES (walked levels): keys admitted and not revoked "
             "(junk-only now; true / false by the mask), from the per-cycle junk record")
    for a, r in arms.items():
        mj = r["log"].get("mj") or []
        L.append(f"  {a}:")
        for q in mj:
            c = q["c"]
            if c % every and c != len(mj):
                continue
            cells = []
            for lv in ("2", "3", "4", "5"):
                v = (q.get("lv") or {}).get(lv) or {}
                if v.get("adm") is not None:
                    cells.append(f"L{lv} {v['adm']} (jo {v.get('adm_jo')}; T {v.get('adm_t')} / "
                                 f"F {v.get('adm_f')})")
            L.append(f"    c{c:>3}: " + ("; ".join(cells) or "no walked level"))


def section_revocations(arms, L):
    L.append("")
    L.append("[G5] REVOCATIONS BY KIND AND BY JUNK-ONLY STATUS at the revoking pass (jo = at support "
             "in the miner, not in its shadow; sv = at support in the shadow too; T/F the true mask); "
             "re-offers; what each arm holds revoked at the end")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        cnt = Counter()
        for c, lv, k, q in graded_passes(adm):
            if q.get("keep"):
                continue
            cc, tt = kcls(q.get("mj"))
            cnt[(lv, q.get("why"), cc, tt)] += 1
        L.append(f"  {a}: {adm.get('n_revoke')} revocations (no consumer {adm.get('n_revoke_nocons')}, "
                 f"worthless {adm.get('n_revoke_worthless')}), re-offers {adm.get('n_reoffer')}, "
                 f"held passes {adm.get('n_held', '—')}")
        for (lv, why, cc, tt), n in sorted(cnt.items()):
            L.append(f"    L{lv} {why:<12} {cc} {tt}: {n}")
        endc = Counter()
        for lv, qs in (adm.get("revoked") or {}).items():
            for q in qs:
                cc, tt = kcls(q.get("mj"))
                endc[(int(lv), q.get("why"), cc, tt)] += 1
        L.append("    held revoked at the end: " + ", ".join(
            f"L{lv} {why} {cc}{tt} {n}" for (lv, why, cc, tt), n in sorted(endc.items())))


def section_decisions(arms, L):
    L.append("")
    L.append("[G6] THE TWO CURRENCIES ON THE SAME FIRES. Every PRICED key-pass (verdict kept or "
             "worthless): this arm's verdict against the other currency's decision at margin 0 on "
             "the same fires (KK/KR/RK/RR = this verdict / other: keep K, revoke R), by level and by "
             "the key's class; then the gains' signs (world, read) by class")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        cf = defaultdict(Counter)
        gs = defaultdict(lambda: [Counter(), Counter()])
        whys = Counter()
        for c, lv, k, q in graded_passes(adm):
            whys[(lv, q.get("why") + ("+held" if q.get("held") else ""))] += 1
            if q.get("why") not in ("kept", "worthless"):
                continue
            cc, tt = kcls(q.get("mj"))
            v = ("K" if q["why"] == "kept" else "R") + ("K" if q.get("keep_other") else "R")
            cf[(lv, cc, tt)][v] += 1
            gs[(lv, cc, tt)][0][_sgn(q.get("g_world"))] += 1
            gs[(lv, cc, tt)][1][_sgn(q.get("g_read"))] += 1
        L.append(f"  {a} ({adm['grade']}): key-passes by verdict {dict(sorted(whys.items()))}")
        L.append(f"    {'L':>2} {'cls':>3} {'T':>2} {'KK':>4} {'KR':>4} {'RK':>4} {'RR':>4}   "
                 f"world gain -/0/+   read gain -/0/+")
        for key in sorted(cf):
            x = cf[key]
            gw, gr = gs[key]
            L.append(f"    {key[0]:>2} {key[1]:>3} {key[2]:>2} {x['KK']:>4} {x['KR']:>4} {x['RK']:>4} "
                     f"{x['RR']:>4}   {gw['-']:>4}/{gw['0']}/{gw['+']:<6} {gr['-']:>4}/{gr['0']}/"
                     f"{gr['+']:<6}")


def section_diet(arms, rb, L, every=4):
    L.append("")
    L.append("[G7] THE DIET AT THE SEAM. The read arm's readout buffer, FIT rows by span, at its grade "
             "passes (every few), beside rubato's diet arm at the same cycle where it exists; the "
             "first priced pass per level; and the bank over cycles on every arm (rows, positive "
             "base rate of the held-out rows, held-out AUC)")
    R = arms.get("sc_grd_yk")
    rd = rb.get("rb_grd_yk")
    rdp = {p["cycle"]: p for p in ((rd or {}).get("adm") or {}).get("gr_passes") or []}
    if R is not None:
        P = (R.get("adm") or {}).get("gr_passes") or []
        fp = {}
        for p in P:
            for lv in p["levels"]:
                for k in lv["keys"]:
                    if k.get("why") in ("kept", "worthless"):
                        fp.setdefault(lv["level"], p["cycle"])
        L.append(f"  sc_grd_yk first priced pass per level: {dict(sorted(fp.items()))}; "
                 f"rb_grd_yk's: " + str({lv: c for lv, c in sorted(
                     {lv["level"]: p["cycle"] for p in reversed(list(rdp.values()))
                      for lv in p["levels"] for k in lv["keys"]
                      if k.get("why") in ("kept", "worthless")}.items())} if rd else "—"))
        for j, p in enumerate(P):
            if j % every and j != len(P) - 1:
                continue
            sf = {int(k): v for k, v in (p.get("span_rows_fit") or {}).items()}
            q = rdp.get(p["cycle"])
            sq = {int(k): v for k, v in ((q or {}).get("span_rows_fit") or {}).items()}
            dk = " ".join(f"L{lv['level']}:{'on' if lv.get('diet_ok') else 'off'}"
                          for lv in p["levels"] if "diet_ok" in lv)
            L.append(f"    c{p['cycle']:>3}: " + " ".join(f"s{sp}={sf.get(sp, 0):>5}"
                                                       for sp in (2, 4, 8, 16))
                     + f"  {dk:<22} | rb_grd_yk: " + (" ".join(f"s{sp}={sq.get(sp, 0):>5}"
                                                            for sp in (2, 4, 8, 16)) if q else "—"))
    allarms = [("st_gn_yk", rb["st_gn_yk"])] + list(arms.items())
    L.append("  the bank (readout buffer) at cycle: rows | held-out positive rate | held-out AUC")
    for a, r in allarms:
        vo = r["log"].get("vo_om") or []
        cells = []
        for c in (60, 100, 140, 180, 201):
            if c - 1 < len(vo):
                x = vo[c - 1] or {}
                br, au = x.get("hold_base"), x.get("hold_auc")
                cells.append(f"c{c} {x.get('n_rows')}|{'—' if br is None else f'{br:.3f}'}|"
                             f"{'—' if au is None else f'{au:.3f}'}")
        L.append(f"    {a:>10}: " + "  ".join(cells))


def section_persist(arms, L):
    L.append("")
    L.append("[G8] PERSISTENCE, RE-READ OFF THE RECORD. Every HELD pass (worthless, run 1 of 2): the "
             "world's decision on the same fires, and what the key's NEXT priced pass said; and the "
             "revocations k = 1 and k = 3 would have made on the recorded passes (a key revoked "
             "earlier than it was is not graded again, so this reads the record, not a replay)")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("gr_persist"):
            continue
        by_key = defaultdict(list)
        for c, lv, k, q in graded_passes(adm):
            by_key[(lv, k)].append((c, q))
        nxt, wk, k3 = Counter(), Counter(), 0
        for key, qs in by_key.items():
            pr = [(c, q) for c, q in qs if q.get("why") in ("kept", "worthless")]
            for j, (c, q) in enumerate(pr):
                if q.get("held"):
                    wk["world keeps" if q.get("keep_other") else "world revokes"] += 1
                    after = pr[j + 1][1] if j + 1 < len(pr) else None
                    nxt["no later priced pass" if after is None else
                        ("next kept (the run broke)" if after.get("why") == "kept"
                         else "next worthless (revoked at run 2)")] += 1
                if q.get("run", 0) >= 3 and q.get("why") == "worthless":
                    k3 += 1
        n_rev_w = sum(1 for c, lv, k, q in graded_passes(adm)
                      if q.get("why") == "worthless" and not q.get("keep"))
        n_held = sum(1 for c, lv, k, q in graded_passes(adm) if q.get("held"))
        L.append(f"  {a}: {n_held} held passes; {n_rev_w} worthless revocations at run 2")
        L.append(f"    on the held passes the world {dict(wk)}; the key's next priced pass: {dict(nxt)}")
        L.append(f"    k = 1 on the recorded passes would have revoked at every one of the {n_held} "
                 f"held passes (and the {n_rev_w} run-2 revocations would then not have been reached "
                 f"as such); k = 3: of the {n_rev_w} run-2 revocations, those whose next pass exists "
                 f"cannot be read (the arm revoked them); runs >= 3 on the record: {k3}")


def section_bill(arms, rb, L):
    L.append("")
    L.append("[G9] THE BILL. Fires (one unfired base pool per level per pass included) and the world "
             "queries billed to the grade")
    for a, r in list(arms.items()) + [(a, r) for a, r in rb.items() if a != "st_gn_yk"]:
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        P = adm.get("gr_passes") or []
        n_f = sum(int(p.get("n_fires") or 0) for p in P)
        c0 = min((p["cycle"] for p in P), default=None)
        n_cyc = (201 - c0 + 1) if c0 else 0
        bill = int(adm.get("grade_world_billed") or 0)
        L.append(f"  {a:>10}: passes {len(P)} from c{c0}; fires {n_f} ({n_f / max(1, n_cyc):.1f} a "
                 f"cycle); world queries billed to the grade {bill:,} ({bill / max(1, n_cyc):,.0f} a "
                 f"cycle)")


def section_survey(r, L, name):
    sv = (r.get("sc") or {}).get("survey")
    L.append(f"  {name}: " + ("no survey" if not sv or sv.get("error") else
                              f"c{sv['cycle']} era {sv['era']}"))
    if not sv or sv.get("error"):
        return
    for lv in sv["levels"]:
        groups = defaultdict(list)
        for k in lv["keys"]:
            groups[(k["src"],) + kcls(k)].append(k)
        L.append(f"    L{lv['level']} node {lv['node']}: base success {lv.get('base_succ')}, base "
                 f"read {None if lv.get('base_p') is None else round(lv['base_p'], 4)}; at support "
                 f"miner {lv.get('n_at_miner')} (jo {lv.get('n_jo_miner')}), panel "
                 f"{lv.get('n_at_panel')} (jo {lv.get('n_jo_panel')})")
        for (src, cls, tr), ks in sorted(groups.items()):
            fired = [k for k in ks if k.get("buildable")]
            sw = Counter(_sgn(k["g_world"]) for k in fired)
            sr = Counter(_sgn(k.get("g_read")) for k in fired)
            L.append(f"      {src:>5} {cls:>2} {tr} n {len(ks):>3} (unbuildable "
                     f"{len(ks) - len(fired)}, served {sum(1 for k in ks if k.get('served'))}): "
                     f"world -/0/+ {sw['-']}/{sw['0']}/{sw['+']} gain "
                     f"{_q([round(k['g_world'], 4) for k in fired])}; read -/0/+ "
                     f"{sr['-']}/{sr['0']}/{sr['+']}")


def grade_reduce(seed):
    arms, rb = load_seed(seed)
    L = [f"[scordatura] THE GRADED ARMS ON THE JUNK FLOOR, seed {seed}. Facts only.",
         f"  junk arms: {list(arms)}; rubato's, on the solve-fed substrate: {list(rb)}", ""]
    section_identity(arms, L)
    section_first_div(arms, "sc_gn_yk", L)
    section_errors(arms, rb, L)
    section_coverage(arms, rb, L)
    section_served(arms, L)
    section_revocations(arms, L)
    section_decisions(arms, L)
    section_diet(arms, rb, L)
    section_persist(arms, L)
    section_bill(arms, rb, L)
    L.append("")
    L.append("[G10] THE FIRE SURVEYS on the floor (single-entry at the key's own cell)")
    tp, ap_ = SEEDS[seed]["pre"]
    pp = os.path.join(FIG, tp, ap_, "results.json")
    if os.path.isfile(pp):
        section_survey(load(pp), L, f"{tp} (the stop at c40)")
    section_survey(arms["sc_gn_yk"], L, f"{SEEDS[seed]['floor'][0]} (the arm's end)")
    return L


def seedtable(seeds=(0, 2)):
    """The two seeds' headline rows side by side (never averaged)."""
    L = ["[scordatura] THE TWO-SEED TABLE. Seeds 0 and 2 side by side, never averaged. Facts only.", ""]
    rows = defaultdict(dict)
    for sd in seeds:
        try:
            arms, rb = load_seed(sd)
        except FileNotFoundError as e:
            L.append(f"  seed {sd}: not all arm files present ({e})")
            continue
        ref = era_err(rb["st_gn_yk"])
        flo = era_err(arms["sc_gn_yk"])
        for a, r in list(arms.items()) + [(a, r) for a, r in rb.items() if a != "st_gn_yk"]:
            ee = era_err(r)
            rows[a][sd] = {
                "err - st_gn_yk": " ".join(f"{ee[e] - ref[e]:+.3f}" for e, _, _ in ERAS),
                "err - floor": (" ".join(f"{ee[e] - flo[e]:+.3f}" for e, _, _ in ERAS)
                                if a.startswith("sc_") else "—")}
            adm = r.get("adm") or {}
            if adm.get("grade"):
                rows[a][sd].update({
                    "first revocation": adm.get("first_revoke"),
                    "revocations (no consumer, worthless)": f"{adm.get('n_revoke')} "
                    f"({adm.get('n_revoke_nocons')}, {adm.get('n_revoke_worthless')})",
                    "re-offers": adm.get("n_reoffer"), "held": adm.get("n_held", "—"),
                    "world billed": f"{int(adm.get('grade_world_billed') or 0):,}"})
            mj = r["log"].get("mj") or []
            if mj:
                lv_ = mj[-1].get("lv") or {}
                rows[a][sd]["served at c201 per level: keys (junk-only; T/F)"] = " ".join(
                    f"L{l}:{lv_[l]['adm']}({lv_[l].get('adm_jo')};{lv_[l].get('adm_t')}/"
                    f"{lv_[l].get('adm_f')})" for l in ("2", "3", "4", "5")
                    if (lv_.get(l) or {}).get("adm") is not None)
            if adm.get("grade"):
                gp = graded_passes(adm)
                rv = [(lv, q) for c, lv, k, q in gp if not q.get("keep")]
                rows[a][sd]["revoked: junk-only / false (tf 0) / mixed of all"] = (
                    f"{sum(1 for lv, q in rv if (q.get('mj') or {}).get('jo'))} / "
                    f"{sum(1 for lv, q in rv if (q.get('mj') or {}).get('tf') == 0)} / "
                    f"{sum(1 for lv, q in rv if (q.get('mj') or {}).get('tf') not in (0, 1, None))}"
                    f" of {len(rv)}")
                # the verdicts on the same fires, L2, where the world's gain is exactly 0
                # (the world revokes) and where it is positive (the world keeps)
                pr = [q for c, lv, k, q in gp if lv == 2 and q.get("why") in ("kept", "worthless")]
                z = [q for q in pr if not q.get("keep_other")] if adm["grade"] == "read" else \
                    [q for q in pr if q.get("why") == "worthless"]
                p_ = [q for q in pr if q.get("keep_other")] if adm["grade"] == "read" else \
                    [q for q in pr if q.get("why") == "kept"]
                if adm["grade"] == "read":
                    rows[a][sd]["L2 priced passes: read revokes where world revokes / keeps"] = (
                        f"{sum(1 for q in z if q['why'] == 'worthless')}/{len(z)} ; "
                        f"{sum(1 for q in p_ if q['why'] == 'worthless')}/{len(p_)}")
                else:
                    rows[a][sd]["L2 priced passes: world revokes / keeps; read would revoke of each"] = (
                        f"{len(z)} ({sum(1 for q in z if not q.get('keep_other'))}) / "
                        f"{len(p_)} ({sum(1 for q in p_ if not q.get('keep_other'))})")
                if adm.get("gr_persist"):
                    held = [q for c, lv, k, q in gp if q.get("held")]
                    rows[a][sd]["held passes: world keeps / revokes"] = (
                        f"{len(held)}: {sum(1 for q in held if q.get('keep_other'))} / "
                        f"{sum(1 for q in held if not q.get('keep_other'))}")
                P_ = adm.get("gr_passes") or []
                c0 = min((p["cycle"] for p in P_), default=None)
                n_f = sum(int(p.get("n_fires") or 0) for p in P_)
                rows[a][sd]["fires a cycle / world queries a cycle"] = (
                    f"{n_f / max(1, 202 - (c0 or 201)):.1f} / "
                    f"{int(adm.get('grade_world_billed') or 0) / max(1, 202 - (c0 or 201)):,.0f}")
            lp = last_pass_rows(r)
            rows[a][sd]["last pass L2/L3/L4/L5 keys | row precision"] = " ".join(
                f"{lp[lv][1].get('n_admitted_keys') if lv in lp else '—'}" for lv in (2, 3, 4, 5)) \
                + " | " + " ".join(
                (f"{lp[lv][1].get('tab_precision'):.2f}" if lv in lp and
                 lp[lv][1].get("tab_precision") is not None else "—") for lv in (2, 3, 4, 5))
    for a in rows:
        L.append(f"  {a}")
        ks = list(dict.fromkeys(k for sd in rows[a] for k in rows[a][sd]))
        for k in ks:
            L.append(f"    {k:<42} " + " || ".join(f"s{sd}: {rows[a][sd].get(k, '—')}"
                                                    for sd in seeds if sd in rows[a]))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--seedtable", action="store_true")
    ap.add_argument("--tag", default="sc_p1")
    ap.add_argument("--arm", default="sc_gn_yk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dose:
        r = load(os.path.join(FIG, a.tag, a.arm, "results.json"))
        rt, ra = REF_OF_SEED[a.seed]
        ref = load(os.path.join(RB_FIG, rt, ra, "results.json"))
        L = [f"[scordatura] THE DOSE READ: {a.tag}/{a.arm} (seed {a.seed}) against "
             f"{rt}/{ra} at the same cycles. Facts only.", ""]
        dose_section(r, ref, L)
        out = os.path.join(FIG, f"{a.tag}_dose.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"\n-> {out}")
    if a.grade:
        L = grade_reduce(a.seed)
        out = os.path.join(FIG, f"sc_grade_s{a.seed}.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"\n-> {out}")
    if a.seedtable:
        L = seedtable()
        out = os.path.join(FIG, "sc_seedtable.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
