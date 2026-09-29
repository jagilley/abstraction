"""[rubato] The reductions this node needs, local and CPU-only.

  --bank-check   a re-run arm (fetched and compacted by `fetch_compact.py`) against its BANKED
                 mirror in `../sostenuto/figures/<bank-tag>/<arm>/`: every key of the arm file,
                 every log series cycle by cycle, every event, the entry record, and the config
                 (which may differ only by the `rb_*` knobs). Wall-clock fields (`t_s`) are the
                 only thing dropped, and the two runs' volume roots are folded to one token.
                 Writes `figures/<tag>_bank_check.txt`.
  --r1           gate R-1's record (`<tag>/r1_gate.json` on the volume, or a local copy) as a
                 table. Writes `figures/<tag>_r1.txt`.
  --grade        phase 2 (DESIGN §10): sostenuto's own reduction [A]..[K] with the two graded
                 arms as two more columns beside its five gates (`reduce_sostenuto.reduce_tag`,
                 imported, its arm table extended), then [L] THE GRADE: revocations by reason and
                 level, re-offers, the served set over time, the per-key confusion of the read's
                 decision against the world's on the SAME fires, every graded key's gain in both
                 currencies (the dial), and the in-loop identity checks. Facts only.
                 Writes `figures/<tag>_reduction.txt`; `--seedtable` sets the two seeds' [L] side
                 by side in `figures/rb_grade_seedtable.txt`.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/reduce_rubato.py --bank-check --tag rb_s1 --bank-tag st_s1
    python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/reduce_rubato.py --r1 --tag rb_r1
"""

import argparse
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BANK = os.path.join(os.path.dirname(HERE), "sostenuto", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_rubato"
ROOTS = ("/data/rhm_practice_rubato/", "/data/rhm_practice_sostenuto/")
SERIES = ("e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
          "vloss", "gloss", "n_solved", "n_mined", "m_per_solve")


def norm(obj, tags=()):
    txt = json.dumps(obj)
    for r in ROOTS:
        txt = txt.replace(r, "<ROOT>/")
    for t in sorted(tags, key=len, reverse=True):
        txt = txt.replace(t, "<TAG>")
    x = json.loads(txt)

    def strip(y):
        if isinstance(y, dict):
            return {k: strip(v) for k, v in y.items() if k != "t_s"}
        if isinstance(y, list):
            return [strip(v) for v in y]
        return y
    return strip(x)


def jdiff(a, b, path="", out=None, cap=200):
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        for k in list(dict.fromkeys(list(a) + list(b))):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'bank' if k not in a else 're-run'}")
            else:
                jdiff(a[k], b[k], f"{path}/{k}", out, cap)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        for j, (x, y) in enumerate(zip(a, b)):
            jdiff(x, y, f"{path}[{j}]", out, cap)
            if len(out) >= cap:
                break
    elif a != b:
        out.append(f"{path}: {str(a)[:50]} vs {str(b)[:50]}")
    return out


def load_arm(d):
    with open(os.path.join(d, "results.json")) as f:
        r = json.load(f)
    ent = None
    p = os.path.join(d, "entry.json.gz")
    if os.path.isfile(p):
        with gzip.open(p, "rt") as f:
            ent = json.load(f)
    return r, ent


def bank_check(tag, bank_tag, arm, tags=()):
    new_d = os.path.join(FIG, tag, arm)
    bank_d = os.path.join(BANK, bank_tag, arm)
    rn, en = load_arm(new_d)
    rb, eb = load_arm(bank_d)
    lines = [f"# BANK CHECK  {tag}/{arm}  (rubato, saved states ON)  against  "
             f"sostenuto/figures/{bank_tag}/{arm}  (the banked mirror)", ""]
    rec = rn.get("rb") or {}
    saves = rec.get("saves") or []
    lines.append(f"saved states written: {len(saves)}  at cycles {[s_['cycle'] for s_ in saves]}")
    if saves:
        mb = [s_["bytes"] / 1e6 for s_ in saves]
        ts = [s_["t_s"] for s_ in saves]
        lines.append(f"  size MB: first {mb[0]:.1f}  last {mb[-1]:.1f}  total {sum(mb):.0f}; "
                     f"seconds per save: median {sorted(ts)[len(ts) // 2]:.1f}  max {max(ts):.1f}  "
                     f"total {sum(ts):.0f}")
        lines.append("  per save (cycle, era, era_last, MB, raw storage MB, packed MB, pickle MB, s):")
        for s_ in saves:
            lines.append(f"    c{s_['cycle']:3d} era {s_['era']} {'E' if s_['era_last'] else ' '} "
                         f"{s_['bytes'] / 1e6:7.1f} {s_['storage_raw'] / 1e6:7.1f} "
                         f"{s_['storage_packed'] / 1e6:7.1f} {s_['pickle_bytes'] / 1e6:6.1f} "
                         f"{s_['t_s']:5.1f}  {s_['path']}")
    lines.append("")
    # config: only the rb knobs may differ
    cn = {k: v for k, v in rn["config"].items()}
    cb = rb["config"]
    cdiff = sorted(k for k in set(cn) | set(cb) if cn.get(k) != cb.get(k))
    lines.append(f"config keys differing: {cdiff}")
    extra = [k for k in cdiff if not k.startswith("rb_")]
    lines.append(f"  outside rb_*: {extra}  -> {'OK' if not extra else 'DIFFERS'}")
    # the arm file
    keys = sorted(set(rn) | set(rb))
    top = []
    for k in keys:
        if k in ("config", "rb"):
            continue
        if k not in rn or k not in rb:
            top.append((k, "only in " + ("bank" if k not in rn else "re-run")))
            continue
        d = jdiff(norm(rn[k], tags), norm(rb[k], tags), f"/{k}", cap=20)
        top.append((k, "equal" if not d else f"{len(d)}+ paths differ: {d[:4]}"))
    lines.append("")
    lines.append("arm file, key by key (t_s dropped, volume roots folded):")
    for k, v in top:
        lines.append(f"  {k:24s} {v}")
    # the log, series by series
    ln, lb = rn["log"], rb["log"]
    lines.append("")
    lines.append(f"log: {len(ln.get('cycle', []))} cycles re-run, {len(lb.get('cycle', []))} banked")
    bad_series = []
    for k in sorted(set(ln) | set(lb)):
        a, b = norm(ln.get(k), tags), norm(lb.get(k), tags)
        if a != b:
            first = None
            if isinstance(a, list) and isinstance(b, list):
                for j, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        first = j + 1
                        break
            bad_series.append((k, first))
    lines.append(f"  series differing: {bad_series if bad_series else 'none'}")
    mx = {}
    for k in SERIES:
        a = [x for x in (ln.get(k) or [])]
        b = [x for x in (lb.get(k) or [])]
        L = min(len(a), len(b))
        dd = [abs(float(x) - float(y)) for x, y in zip(a[:L], b[:L])
              if x is not None and y is not None]
        mx[k] = max(dd) if dd else 0.0
    lines.append(f"  max|re-run - bank| per fidelity series: {mx}")
    # the entry record
    if en is not None and eb is not None:
        de = jdiff(norm(en), norm(eb), "/entry", cap=10)
        lines.append(f"entry record ({len(en)} vs {len(eb)} cycles): "
                     f"{'equal' if not de else de[:4]}")
    else:
        lines.append(f"entry record: re-run {'present' if en is not None else 'absent'}, "
                     f"bank {'present' if eb is not None else 'absent'}")
    # KEYS ONLY IN THE RE-RUN, attributed. The banked `st_s1` arms ran on sostenuto's ROUND-1
    # binary; round 2 then added its gates' bookkeeping to every admission arm's config and `_adm`
    # (sostenuto DESIGN §9.4). A key present only in the re-run carries no value to compare; a
    # key present in both must be equal.
    R2_CFG = {"adm_delta", "adm_demand_floor", "adm_demand_min"}
    R2_ADM = {"r2", "delta", "thr", "floor", "n_silent", "n_nxt_none", "nxt_src",
              "repro_strong_own", "exports", "strong_own"}
    # and rubato's own phase-2 run-level knobs (DESIGN §10), carried with the grade OFF by a
    # re-run made on the phase-2 binary (`rb_s2`): the arm's own `adm_grade` must be absent
    PH2_CFG = {"adm_grade_obs", "adm_grade_margin", "adm_grade_cons", "adm_grade_falsify"}
    only_new, only_ph2 = [], []
    for k in list(extra):
        if k in R2_CFG and k not in cb:
            only_new.append(f"config/{k}")
        if k in PH2_CFG and k not in cb and not cn.get("adm_grade"):
            only_ph2.append(f"config/{k}={cn.get(k)!r}")
    extra = [k for k in extra if not ((k in R2_CFG or (k in PH2_CFG and not cn.get("adm_grade")))
                                      and k not in cb)]
    shared_ok = True
    for k, v in top:
        if v == "equal":
            continue
        d = jdiff(norm(rn[k], tags), norm(rb[k], tags), f"/{k}", cap=500)
        rest = [x for x in d if not (x.endswith("only in re-run")
                                     and x.split(":")[0].split("/")[-1] in R2_ADM)]
        only_new.extend(x.split(":")[0] for x in d if x not in rest)
        if rest:
            shared_ok = False
            lines.append(f"  {k}: {len(rest)} shared paths differ: {rest[:6]}")
    ok = (not extra and shared_ok and not bad_series
          and (en is None or eb is None or not jdiff(norm(en), norm(eb), cap=1)))
    lines.append("")
    lines.append(f"keys only in the re-run (sostenuto round 2's bookkeeping, absent from the "
                 f"round-1 binary the bank ran on): {sorted(only_new)}")
    if only_ph2:
        lines.append(f"keys only in the re-run (rubato phase 2's run-level grade knobs, the grade "
                     f"OFF: the arm's `adm_grade` is {cn.get('adm_grade')!r}): {sorted(only_ph2)}")
    lines.append(f"VERDICT: {'BIT-IDENTICAL on every quantity both files carry' if ok else 'DIFFERS'} "
                 f"(outside the rb_* knobs, the rb record and wall-clock fields)")
    out = os.path.join(FIG, f"{tag}_bank_check.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")
    return ok


def fetch_file(remote_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, remote_path, local_path],
                   check=True)


def r1_table(tag, local=None):
    p = local or os.path.join(FIG, tag, "r1_gate.json")
    if not os.path.isfile(p):
        fetch_file(f"{REMOTE}/{tag}/r1_gate.json", p)
    g = json.load(open(p))
    L = [f"# GATE R-1  {tag}  arm {g['arm']}  (preflight substrate)", "",
         f"plan: {g['n_plan_cycles']} cycles, era boundaries (advances) at {g['advances']}",
         f"saved at c{g['c_era']} (an era boundary) and c{g['c_mid']} (mid-era); every run "
         f"continued to c{g['stop']}", ""]
    L.append(f"containers (MODAL_TASK_ID): {g['tasks']}")
    L.append(f"errors: {g['errors'] or 'none'}")
    L.append("")
    L.append(f"{'run':4s} {'digest=':8s} {'state':>6s} {'file':>6s} {'first div':>9s} "
             f"{'max|de|':>9s}  resumed from / R-2")
    for k, v in g["vs_A"].items():
        rf = v.get("resumed_from") or {}
        L.append(f"{k:4s} {str(v['digest_equal']):8s} {v['n_state_paths_differing']:6d} "
                 f"{v['n_file_paths_differing']:6d} {str(v['first_divergent_cycle']):>9s} "
                 f"{v['series_max_abs'].get('e', float('nan')):9.3g}  "
                 + (f"c{rf.get('cycle')} (era_last={rf.get('era_last')}) R-2 {rf.get('r2')} "
                    f"falsify={rf.get('falsify')}" if rf else "uninterrupted"))
        if v["state_paths"]:
            L.append(f"       state paths: {v['state_paths'][:6]}")
        if v["file_paths"]:
            L.append(f"       file paths:  {v['file_paths'][:4]}")
    L.append("")
    L.append(f"donor st_pf_gn (sostenuto.run_arm) vs A2 on A2's substrate: {g.get('donor')}")
    L.append("")
    L.append(f"saves in A (cycle, bytes, s): {g.get('saves_A')}")
    L.append("")
    L.append("VERDICT: " + json.dumps(g["verdict"]))
    out = os.path.join(FIG, f"{tag}_r1.txt")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {out}")


GRADE_ARMS = ("rb_gw_yk", "rb_gr_yk")
# [rubato r3] the corrected read arm (DESIGN §11), and the tags whose graded arms read beside a
# tag's own: a corrected arm is reduced beside the arms it corrects
DIET_ARMS = ("rb_grd_yk",)
# the ungated re-run each graded tag was restored from (its consumer survey feeds M-6)
REF_OF = {"rb_g1": ("rb_s1",), "rb_g1d": ("rb_s1",), "rb_g2": ("rb_s2",)}
GRADE_WITH = {"rb_g1d": ("rb_g1",)}
GRADE_EXTRA = {"rb_g1": ("rb_g1d",)}
SOS = os.path.join(os.path.dirname(HERE), "sostenuto")


def _q(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "—"
    return " ".join(f"{xs[min(len(xs) - 1, int(q * len(xs)))]:+.4f}" for q in qs) + f" (n {len(xs)})"


def prior_from_log(log_path, arm):
    """A container segment's record, read off its launch log, for a segment whose saves predate
    `rb_hist` (DESIGN §5.5): the restore it started from, its saves' digests and its G-I results,
    up to the preemption that ended it. Facts the log printed, nothing inferred."""
    import ast
    import re
    lines = open(log_path, errors="replace").read().splitlines()
    cut = next((i for i, x in enumerate(lines) if "terminated due to preemption" in x), len(lines))
    lines = lines[:cut]
    rf, saves, idn = None, {}, {}
    for i, x in enumerate(lines):
        if f"arm={arm} RESUMES after" in x:
            for y in reversed(lines[max(0, i - 6):i]):
                m = re.search(r"RESTORED (\S+): c(\d+) .* R-2 (\w+) \(digest ([0-9a-f]+)\)", y)
                if m:
                    pth = m.group(1)
                    rf = {"path": pth, "cycle": int(m.group(2)), "r2": m.group(3),
                          "digest": m.group(4), "arm": pth.split("/ck/")[0].split("/")[-1],
                          "source": os.path.basename(log_path)}
                    break
        m = re.search(rf"\[rubato\] arm={arm} c(\d+) G-I vs \S+: (\{{.*\}})\s*$", x)
        if m:
            idn.setdefault(int(m.group(1)), ast.literal_eval(m.group(2)))
        m = re.search(rf"\[rubato\] arm={arm} c(\d+) SAVED (\S+) .*digest=([0-9a-f]+)", x)
        if m:
            saves.setdefault(int(m.group(1)), {"path": m.group(2), "digest": m.group(3)})
    return {"resumed_from": rf, "host": None, "prior": None, "from_log": log_path,
            "saves": [{"cycle": c, "identity": idn.get(c), **saves[c]} for c in sorted(saves)]}


# the launch logs of container segments whose saves predate `rb_hist` (DESIGN §5.5), per tag
SEG_LOGS = {"rb_g1": (os.path.join(HERE, "results", "launch_rb_g1_seg1.log"),)}


def rb_splice(arms, seg_logs=()):
    """Where an arm's chain of segments stops at a resume from its OWN save (the saving segment
    predates `rb_hist`), splice in that segment's record read off its launch log — only a log
    whose last save's digest IS the digest the arm resumed from."""
    for a, r in arms.items():
        tail = r.get("rb") or {}
        while tail.get("prior"):
            tail = tail["prior"]
        rf_ = tail.get("resumed_from") or {}
        if rf_.get("arm") != a:
            continue
        for lp in seg_logs:
            pr = prior_from_log(lp, a)
            if pr["resumed_from"] and pr["saves"] and pr["saves"][-1]["digest"] == rf_.get("digest"):
                tail["prior"] = pr
                break


def rb_segments(rb):
    """A run's container segments, oldest first: each {resumed_from, saves, host, auto_resumed}.
    A resumed run carries the run it continued in `prior` (the saving run's own record, written
    into every save's metadata, DESIGN §5.5); a save written before that carries none, and the
    chain then stops there (`broken`)."""
    segs, cur = [], rb or {}
    while cur:
        segs.append({k: cur.get(k) for k in ("resumed_from", "saves", "host", "auto_resumed",
                                             "from_log")})
        cur = cur.get("prior")
    segs.reverse()
    return segs


def grade_section(arms, ref, L, seg_logs=()):
    """[L] THE GRADE, facts only. `arms` {name: arm file}; `ref` the ungated arm file.
    `seg_logs`: launch logs of earlier container segments whose saves predate `rb_hist`; an arm
    whose chain is broken there gets that segment's record from the log (DESIGN §5.5)."""
    rb_splice(arms, seg_logs)
    L.append("")
    L.append("=" * 100)
    L.append("[L] THE GRADE — admit-then-grade (DESIGN §10)")
    L.append("=" * 100)
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        rb = r.get("rb") or {}
        segs = rb_segments(rb)
        # the FORK: the first segment's restore (from the ungated arm); later segments are this
        # arm resuming its own saves (a preemption, DESIGN §5.5)
        rf = (segs[0].get("resumed_from") if segs else None) or {}
        broken = bool(rf and rf.get("arm") == a)
        L.append("")
        L.append(f"--- {a}: grade={adm['grade']} window={adm.get('gr_window')} obs "
                 f"margin={adm.get('gr_margin')} cap={adm.get('gr_cap')}  "
                 f"restored from c{rf.get('cycle')} of {rf.get('arm')} (R-2 {rf.get('r2')}, "
                 f"schema added {len(rf.get('schema_added') or [])}, overlay {rf.get('overlay')})")
        L.append(f"  first revocation c{adm.get('first_revoke')}; revoked {adm.get('n_revoke')} "
                 f"(no consumer {adm.get('n_revoke_nocons')}, worthless "
                 f"{adm.get('n_revoke_worthless')}); re-offered {adm.get('n_reoffer')}; kept "
                 f"{adm.get('n_keep')}; graded {adm.get('n_graded')}; young {adm.get('n_young')}; "
                 f"silent-level key-passes {adm.get('gr_n_silent')}; fires {adm.get('n_fires')}; "
                 f"world billed {adm.get('grade_world_billed')}; merged-away revoked "
                 f"{adm.get('n_rekey_revoked')}; commits cancelled "
                 f"{adm.get('n_commit_cancelled_empty')}")
        L.append(f"  container segments: " + "; ".join(
            f"from c{(g_.get('resumed_from') or {}).get('cycle')} of "
            f"{(g_.get('resumed_from') or {}).get('arm')}"
            f"{' (auto)' if g_.get('auto_resumed') else ''} on "
            f"{((g_.get('host') or {}).get('cpu') or '?')[:40]}"
            f" avx512={(g_.get('host') or {}).get('np_avx512f')}" for g_ in segs)
            + ("  [the first segment's record is not carried: its saves predate DESIGN §5.5]"
               if broken else "")
            + "".join(f"  [segment from c{(g_.get('resumed_from') or {}).get('cycle')} read off "
                      f"{os.path.basename(g_['from_log'])}]" for g_ in segs if g_.get("from_log")))
        idn = [(q.get("cycle"), q.get("identity")) for g_ in segs for q in (g_.get("saves") or [])]
        ok = [c for c, x in idn if x and x.get("equal")]
        bad = [(c, x.get("diffs")) for c, x in idn if x and "equal" in x and not x["equal"]]
        L.append(f"  G-I identity with the ungated arm by hash, before the first revocation: "
                 f"{len(ok)} saves equal {ok}; {len(bad)} differ {bad[:3]}")
        # G-I off the arm files: every per-cycle log series, cycle by cycle, against the ungated
        # arm, from the restore to the first revocation (the bill and wall clock left out on world)
        if ref is not None:
            c0 = 0 if broken else int(rf.get("cycle") or 0)
            fr = adm.get("first_revoke")
            skip = {"t_cum", "prop", "vo_bill"} if adm["grade"] == "world" else set()
            first_div = None
            cyc = r["log"]["cycle"]
            n_c = len(cyc)
            # per-cycle series are compared by index; a series of cycle-stamped records (`probe`)
            # by its records' own cycles
            stamped = {k for k, v in r["log"].items() if isinstance(v, list) and len(v) != n_c
                       and v and isinstance(v[0], dict) and "cycle" in v[0]}
            by_c = {k: ({q["cycle"]: q for q in r["log"][k]},
                        {q["cycle"]: q for q in ref["log"].get(k) or []}) for k in stamped}
            for i, c in enumerate(cyc):
                if c <= c0 or i >= len(ref["log"]["cycle"]):
                    continue
                for k in r["log"]:
                    if k in skip or k not in ref["log"]:
                        continue
                    if k in stamped:
                        x_, y_ = by_c[k][0].get(c), by_c[k][1].get(c)
                        if x_ is None and y_ is None:
                            continue
                    elif isinstance(r["log"][k], list) and len(r["log"][k]) == n_c \
                            and i < len(ref["log"][k]):
                        x_, y_ = r["log"][k][i], ref["log"][k][i]
                    else:
                        continue
                    if norm(x_) != norm(y_):
                        first_div = (int(c), k)
                        break
                if first_div:
                    break
            L.append(f"  G-I off the log: restored after c{c0}; first cycle any log series differs "
                     f"from the ungated arm: {first_div}; first revocation c{fr} "
                     f"({'OK' if (first_div is None or fr is None or first_div[0] >= fr) else 'BEFORE the first revocation'})")
            # the two label-carrying record lists, which G-I cannot compare by hash against a
            # digest-v1 reference (DESIGN §5.4): record by record, the arm's label dropped (and
            # on world the bill), every record before the first revocation's cycle
            drop = {"arm", "t_s"} | ({"t_cum"} if adm["grade"] == "world" else set())

            def _upto(evs):
                out_, c_ = [], 0
                for ev in evs or []:
                    c_ = ev.get("cycle", c_) if isinstance(ev, dict) else c_
                    if fr is not None and c_ is not None and c_ >= fr:
                        break
                    out_.append({k: v for k, v in ev.items() if k not in drop}
                                if isinstance(ev, dict) else ev)
                return out_
            for lst in ("events", "merge_events"):
                a_, b_ = _upto(r.get(lst)), _upto(ref.get(lst))
                n_ = min(len(a_), len(b_))
                fd = next((j for j in range(n_) if norm(a_[j]) != norm(b_[j])), None)
                L.append(f"  G-I off the records, {lst} before c{fr}: {len(a_)} | ungated "
                         f"{len(b_)}; first differing record "
                         f"{'none' if fd is None else (fd, a_[fd].get('kind'), a_[fd].get('cycle'))}"
                         f"{'' if len(a_) == len(b_) else '  (LENGTHS DIFFER)'}")
        _fp = {}
        for p_ in adm.get("gr_passes") or []:
            for lv_ in p_["levels"]:
                for k_ in lv_["keys"]:
                    if k_.get("why") in ("kept", "worthless"):
                        _fp.setdefault(lv_["level"], p_["cycle"])
        L.append(f"  first PRICED key-pass (kept or worthless) per level: "
                 f"{ {lv: f'c{c}' for lv, c in sorted(_fp.items())} }; diet "
                 f"{adm.get('gr_diet') or 'off'} (silent for the diet {adm.get('n_no_diet', 0)} "
                 f"key-passes)")
        L.append(f"  G-W {(adm.get('gr_checks') or {}).get('G-W')}; "
                 f"G-S red {(adm.get('gr_checks') or {}).get('G-S_red')}")
        # per level: revocations by reason, re-offers, and the served set's size over time
        P = adm.get("gr_passes") or []
        by = {}
        for p in P:
            for lv in p["levels"]:
                d = by.setdefault(lv["level"], {"graded": 0, "rev_nocons": 0, "rev_worth": 0,
                                                "kept": 0, "unpriced": 0, "no_reader": 0,
                                                "no_diet": 0, "silent_passes": 0, "young": 0})
                if lv.get("silent"):
                    d["silent_passes"] += 1
                for k in lv["keys"]:
                    st = k.get("state")
                    if st == "young":
                        d["young"] += 1
                    if st != "graded":
                        continue
                    d["graded"] += 1
                    w = k.get("why")
                    d["rev_nocons" if w == "no_consumer" else "rev_worth" if w == "worthless"
                      else "kept" if w == "kept" else "unpriced" if w == "unpriced"
                      else "no_reader" if w == "no_reader" else "no_diet" if w == "no_diet"
                      else "kept"] += 1
        L.append(f"  {'level':>5} {'graded':>7} {'kept':>6} {'rev:none':>9} {'rev:worth':>10} "
                 f"{'unpriced':>9} {'noreader':>9} {'nodiet':>7} {'young':>6} {'silent':>7}"
                 f"   (key-passes)")
        for lv in sorted(by):
            d = by[lv]
            L.append(f"  {lv:>5} {d['graded']:>7} {d['kept']:>6} {d['rev_nocons']:>9} "
                     f"{d['rev_worth']:>10} {d['unpriced']:>9} {d['no_reader']:>9} "
                     f"{d['no_diet']:>7} {d['young']:>6} {d['silent_passes']:>7}")
        rel = [(p["cycle"], x) for p in P for x in p.get("released", [])]
        rvk = [(p["cycle"], x) for p in P for x in p.get("revoked", [])]
        L.append(f"  revocations (cycle, level): {[(c, x[0]) for c, x in rvk][:40]}")
        L.append(f"  re-offers   (cycle, level): {[(c, x[0]) for c, x in rel][:40]}")
        # the confusion of this arm's decision against the other currency's, on the same fires
        cf = {}
        gw, gr = [], []
        for p in P:
            for lv in p["levels"]:
                for k in lv["keys"]:
                    if k.get("state") != "graded" or k.get("why") in ("no_consumer",):
                        continue
                    if k.get("g_world") is not None:
                        gw.append(k["g_world"])
                    if k.get("g_read") is not None:
                        gr.append(k["g_read"])
                    if k.get("why") in ("unpriced", "no_reader"):
                        continue
                    key = ("K" if k["keep"] else "R") + ("K" if k.get("keep_other") else "R")
                    cf.setdefault(lv["level"], {}).setdefault(key, 0)
                    cf[lv["level"]][key] += 1
        L.append(f"  this arm's decision vs the other currency's on the same fires "
                 f"(KK/KR/RK/RR = this/other keep K or revoke R), per level: {cf}")
        L.append(f"  gain quantiles (10/25/50/75/90%), world: {_q(gw)}")
        L.append(f"  gain quantiles (10/25/50/75/90%), read:  {_q(gr)}")
        # the served set over time, beside the ungated arm's
        L.append("  served keys per level at every pass (this arm | ungated):")
        un = {}
        if ref is not None:
            for p in (ref.get("adm") or {}).get("passes") or []:
                for q in p["levels"]:
                    un[(p["cycle"], q["level"])] = q.get("n_admitted_keys")
        for p in (adm.get("passes") or [])[::2]:
            row = []
            for q in p["levels"]:
                row.append(f"L{q['level']} {q.get('n_admitted_keys')}|"
                           f"{un.get((p['cycle'], q['level']), '—')}")
            L.append(f"    c{p['cycle']:>3} " + "  ".join(row))
        # the task error in the deep eras against the ungated arm
        if ref is not None:
            e, e0, er = r["log"]["e"], ref["log"]["e"], r["log"]["era"]
            for ei in sorted(set(er)):
                idx = [i for i, x in enumerate(er) if x == ei and i < len(e0)]
                if idx:
                    L.append(f"  era {ei}: mean task error {sum(e[i] for i in idx) / len(idx):.4f}"
                             f"  ungated {sum(e0[i] for i in idx) / len(idx):.4f}  "
                             f"({len(idx)} cycles)")


MARGINS = (-0.20, -0.15, -0.10, -0.075, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0, 0.01,
           0.02, 0.05, 0.10, 0.20, 0.30)


def _kp(adm):
    """Every graded key-pass: (cycle, level, key tuple, record), in pass order."""
    return [(p["cycle"], lv["level"], tuple(k["key"]), k) for p in adm.get("gr_passes") or []
            for lv in p["levels"] for k in lv["keys"] if k.get("state") == "graded"]


def _walked(ref):
    nd = {}
    for ps in ((ref or {}).get("adm") or {}).get("passes") or []:
        for lr in ps["levels"]:
            for st in lr["steps"]:
                nd.setdefault(lr["level"], set()).add(tuple(st["key"]))
    return nd


def _halves(k):
    return (k[:len(k) // 2], k[len(k) // 2:])


def grade_forensics(tag, arms, ref, L, diet=None, true_tables=None, revsurv=None,
                    ref_survey=None):
    """[M] FOUR READINGS OFF THE GRADED ARMS' OWN RECORDS, zero GPU, facts only (the
    coordinator's four questions after seed 0): the dial swept over the recorded gains, the
    persistence of each worthless revocation, the readout's diet at the seam, and the
    re-offers; then the bill's scale."""
    L.append("")
    L.append("=" * 100)
    L.append("[M] THE GRADE'S RECORD, RE-READ: the dial, persistence, the diet at the seam, "
             "re-offers, the bill")
    L.append("=" * 100)
    walked = _walked(ref)
    other_of = {"read": "world", "world": "read"}
    # ---- M-1 the dial ---------------------------------------------------------------- #
    L.append("")
    L.append("M-1 THE DIAL FROM DATA. Keep iff the best consumer's gain > margin, swept over "
             "each arm's RECORDED")
    L.append("    per-fire gains in its own currency; the other currency's decision at margin 0 "
             "on the SAME fires is")
    L.append("    the split (KK/KR/RK/RR = this arm at the margin / the other at 0, keep K or "
             "revoke R). Key-passes")
    L.append("    that are margin-free are held as recorded: `no_consumer` revokes; `unpriced`, "
             "`no_reader`, `no_cell`")
    L.append("    keep. COUNTERFACTUAL ON THE RECORDED PASSES ONLY: a key the arm actually "
             "revoked was not graded")
    L.append("    again until re-offered, so a margin can only turn a recorded decision around, "
             "not replay the run.")
    L.append("    'L3 both' / 'L4 both' = of st_gn_yk's walked L3 (L4) keys, how many have NEITHER "
             "half revoked on any")
    L.append("    recorded pass at this margin (re-offers not simulated). Margin 0 reproduces "
             "the arm's own record.")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        mode = adm.get("grade")
        if not mode:
            continue
        kp = _kp(adm)
        L.append("")
        L.append(f"  {a} ({mode} gains; split by the {other_of[mode]}'s decision at 0), "
                 f"{len(kp)} graded key-passes:")
        L.append(f"    {'margin':>7} {'KK':>5} {'KR':>5} {'RK':>5} {'RR':>5} {'revoked':>8} "
                 f"{'keys rev':>9} {'L2 keys':>8} {'L3 keys':>8} {'L3 both':>8} {'L4 both':>8}")
        for m in MARGINS:
            cf = {"KK": 0, "KR": 0, "RK": 0, "RR": 0}
            rev = set()
            n_rev = 0
            for c, lv, k, q in kp:
                w = q.get("why")
                if w == "no_consumer":
                    keep = False
                elif w in ("kept", "worthless"):
                    keep = (q.get("g") if q.get("g") is not None else 0.0) > m
                else:
                    keep = True
                if not keep:
                    n_rev += 1
                    rev.add((lv, k))
                if w in ("kept", "worthless"):
                    ko = bool(q.get("keep_other"))
                    cf[("K" if keep else "R") + ("K" if ko else "R")] += 1
            both = {}
            for lv in (3, 4):
                ks = walked.get(lv, set())
                both[lv] = (f"{sum(1 for k in ks if not any((lv - 1, h) in rev for h in _halves(k)))}"
                            f"/{len(ks)}")
            L.append(f"    {m:>+7.3f} {cf['KK']:>5} {cf['KR']:>5} {cf['RK']:>5} {cf['RR']:>5} "
                     f"{n_rev:>8} {len(rev):>9} "
                     f"{sum(1 for x in rev if x[0] == 2):>8} {sum(1 for x in rev if x[0] == 3):>8} "
                     f"{both[3]:>8} {both[4]:>8}")
    # ---- M-2 persistence --------------------------------------------------------------- #
    L.append("")
    L.append("M-2 PERSISTENCE. Every 'worthless' revocation: the key's graded passes before it "
             "(gain in this arm's")
    L.append("    currency | the world's), and the run of CONSECUTIVE non-positive passes "
             "ending at the revocation")
    L.append("    (inclusive), counted over the key's PRICED passes (`kept`/`worthless`; a "
             "`no_reader`/`unpriced` pass")
    L.append("    neither extends nor breaks it). Under 'revoke only after k consecutive "
             "non-positive passes' the")
    L.append("    revocation still fires at that pass iff the run >= k. (The key's later passes "
             "under such a rule are")
    L.append("    not in the record: the arm had revoked it.)")
    true2 = None
    if true_tables:
        true2 = {tuple(x) for x in (true_tables.get("2") or {}).get("flat") or []}
    for a, r in arms.items():
        adm = r.get("adm") or {}
        mode = adm.get("grade")
        if not mode:
            continue
        kp = _kp(adm)
        hist = {}
        rows = []
        for c, lv, k, q in kp:
            h = hist.setdefault((lv, k), [])
            if q.get("why") == "worthless":
                run = 1
                for q0 in reversed(h):
                    if q0.get("why") not in ("kept", "worthless"):
                        continue
                    if (q0.get("g") if q0.get("g") is not None else 0.0) <= 0:
                        run += 1
                    else:
                        break
                prior = [q0 for q0 in h if q0.get("why") in ("kept", "worthless")]
                rows.append((c, lv, k, q, len(h), prior, run))
            h.append(dict(q, cycle=c))
        L.append("")
        L.append(f"  {a}: {len(rows)} worthless revocations")
        L.append(f"    {'cycle':>5} {'lvl':>3} {'key':<10} {'true':>5} {'n graded before':>15} "
                 f"{'run':>4}  gain now ({mode}|world)   earlier priced gains ({mode}|world), oldest first")
        for c, lv, k, q, nb, prior, run in rows:
            tr = ("—" if true2 is None or lv != 2 else ("T" if k in true2 else "junk"))
            earlier = " ".join(f"c{q0['cycle']}:{(q0.get('g') or 0):+.3f}|{(q0.get('g_world') if q0.get('g_world') is not None else float('nan')):+.2f}"
                               for q0 in prior)
            L.append(f"    {c:>5} {lv:>3} {str(list(k)):<10} {tr:>5} {nb:>15} {run:>4}  "
                     f"{(q.get('g') or 0):+.4f}|{(q.get('g_world') if q.get('g_world') is not None else float('nan')):+.3f}   {earlier}")
        for kk in (2, 3, 4):
            n_still = sum(1 for x in rows if x[6] >= kk)
            L.append(f"    k={kk}: {n_still} of {len(rows)} revocations still fire at their pass; "
                     f"{len(rows) - n_still} do not (of those, the {other_of[mode]} kept "
                     f"{sum(1 for x in rows if x[6] < kk and x[3].get('keep_other'))} on the "
                     f"same fires)")
    # ---- M-3 the diet at the seam ------------------------------------------------------ #
    L.append("")
    L.append("M-3 THE DIET AT THE SEAM. The readout's row buffer by SPAN at the saves bracketing "
             "each revocation")
    L.append("    (L2 rows span 2, L3 rows span 4, L4 span 8, L5 span 16; the fit draws up to "
             "8192 of the trainable")
    L.append("    rows at random). An L2 key is graded on its L3 consumers' fires, which the "
             "readout reads at span 4.")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        mode = adm.get("grade")
        if not mode:
            continue
        d = sorted(((diet or {}).get(a) or []), key=lambda x: x["cycle"])
        L.append("")
        L.append(f"  {a}: the buffer at every save (cycle: total | span2 span4 span8 span16 | "
                 f"span-4 positives)")
        if not d:
            L.append("    (no diet record for this arm)")
        for x in d:
            bs = {int(k_): v_ for k_, v_ in x["by_span"].items()}
            ps = {int(k_): v_ for k_, v_ in (x.get("pos_by_span") or {}).items()}
            L.append(f"    c{x['cycle']:>3}: {x['n']:>6} | " + " ".join(f"{bs.get(sp, 0):>6}"
                                                                   for sp in (2, 4, 8, 16))
                     + f" | {ps.get(4, 0):>5}")

        # the buffer at every PASS, where the arm logged it (the diet arm, DESIGN §11)
        pr = [p_ for p_ in adm.get("gr_passes") or [] if p_.get("span_rows_fit") is not None]
        if pr:
            L.append(f"  {a}: the buffer's FIT rows (non-hold) by span at every pass, and whether "
                     f"each level's consumers were priced (diet {adm.get('gr_diet')}):")
            for p_ in pr:
                sf = {int(k_): v_ for k_, v_ in p_["span_rows_fit"].items()}
                dk = " ".join(f"L{lv_['level']}:{'on' if lv_.get('diet_ok') else 'off'}"
                              for lv_ in p_["levels"] if "diet_ok" in lv_)
                L.append(f"    c{p_['cycle']:>3}: " + " ".join(f"s{sp}={sf.get(sp, 0)}"
                                                         for sp in (2, 4, 8, 16)) + f"   {dk}")

        def at(c, side, sp=4):
            xs = [x for x in d if (x["cycle"] <= c if side == "lo" else x["cycle"] >= c)]
            if not xs:
                return None
            x = xs[-1] if side == "lo" else xs[0]
            return x["cycle"], {int(k_): v_ for k_, v_ in x["by_span"].items()}.get(sp, 0)
        ws = [(c, lv, k, q) for c, lv, k, q in _kp(adm) if q.get("why") == "worthless"]
        if d and ws:
            oc = other_of[mode]
            L.append(f"  {a}: rows at the CONSUMERS' span (L2 key: span 4; L3 key: span 8) in "
                     f"the buffer at the saves bracketing each worthless revocation (the {oc}'s "
                     f"decision on the same fires):")
            thin, below = [0, 0], [0, 0]
            for c, lv, k, q in ws:
                sp_ = 2 ** lv
                lo, hi = at(c, "lo", sp_), at(c, "hi", sp_)
                og = q.get("g_world") if mode == "read" else q.get("g_read")
                L.append(f"    c{c:>3} L{lv} {str(list(k)):<10} span-{sp_} rows: "
                         f"{('c' + str(lo[0]) + ' ' + str(lo[1])) if lo else '—'} .. "
                         f"{('c' + str(hi[0]) + ' ' + str(hi[1])) if hi else '—'}   "
                         f"{oc} {'KEEP' if q.get('keep_other') else 'revoke'} "
                         f"({oc} gain {og if og is not None else float('nan'):+.3f})")
                if (hi or lo) and (hi or lo)[1] == 0:
                    thin[0] += 1
                    thin[1] += int(bool(q.get("keep_other")))
                if (hi or lo) and (hi or lo)[1] < 512:
                    below[0] += 1
                    below[1] += int(bool(q.get("keep_other")))
            L.append(f"    at a pass whose LATER bracketing save held FEWER THAN 512 rows (all "
                     f"rows, hold included) at the consumers' span: {below[0]} of {len(ws)} "
                     f"({below[1]} of them kept by the {oc})")
            L.append(f"    at a pass whose LATER bracketing save still held NO row at the "
                     f"consumers' span: {thin[0]} of {len(ws)} worthless revocations ({thin[1]} of "
                     f"them kept by the {oc}); with rows there: {len(ws) - thin[0]} "
                     f"({sum(1 for c, lv, k, q in ws if q.get('keep_other')) - thin[1]} kept by "
                     f"the {oc})")
        if d:
            # the revocation RATE on either side of the first span-4 row: every priced key-pass
            # at L2, by whether the save at or after its pass held a span-4 row
            def span4_at(c):
                x = at(c, "hi") or at(c, "lo")
                return x[1] if x else None
            side = {"none": [0, 0, 0], "some": [0, 0, 0]}
            for c, lv, k, q in _kp(adm):
                if lv != 2 or q.get("why") not in ("kept", "worthless"):
                    continue
                sd = "none" if not span4_at(c) else "some"
                side[sd][0] += 1
                side[sd][1] += int(q.get("why") == "worthless")
                side[sd][2] += int(q.get("why") == "worthless" and bool(q.get("keep_other")))
            L.append(f"    L2 priced key-passes with NO span-4 row in the buffer: {side['none'][0]}, "
                     f"revoked {side['none'][1]} ({side['none'][1] / max(1, side['none'][0]):.3f}), "
                     f"of which the {oc} kept {side['none'][2]}; WITH span-4 rows: "
                     f"{side['some'][0]}, revoked {side['some'][1]} "
                     f"({side['some'][1] / max(1, side['some'][0]):.3f}), of which the {oc} kept "
                     f"{side['some'][2]}")
    L.append("")
    L.append("  the cycle histogram of the worthless revocations, by era (era: cycles):")
    eras = [(1, 1, 60), (2, 61, 110), (3, 111, 180), (4, 181, 192), (5, 193, 201)]
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        ws = [c for c, lv, k, q in _kp(adm) if q.get("why") == "worthless"]
        wk = [c for c, lv, k, q in _kp(adm) if q.get("why") == "worthless" and q.get("keep_other")]
        L.append(f"    {a}: " + "  ".join(
            f"era{e} (c{lo}-{hi}): {sum(1 for c in ws if lo <= c <= hi)}"
            f" [other currency keeps {sum(1 for c in wk if lo <= c <= hi)}]"
            for e, lo, hi in eras))
    # ---- M-4 re-offers ----------------------------------------------------------------- #
    L.append("")
    L.append("M-4 RE-OFFERS. Every revocation, whether and when the key came back, and — for a "
             "key that never came")
    L.append("    back — its state at every later save (`rb_consumer_survey --which revoked` over "
             "the arm's own saves):")
    L.append("    a re-offer is CHECKED only where the key is in the live build that pass, and "
             "fires only on a consumer")
    L.append("    at support that was NOT among its consumers at the revocation.")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        mode = adm.get("grade")
        if not mode:
            continue
        oc = other_of[mode]
        P = adm.get("gr_passes") or []
        recs = {}
        for c, lv, k, q in _kp(adm):
            if not q.get("keep"):
                recs.setdefault((lv, k), []).append((c, q))
        rel = {}
        for p in P:
            for x in p.get("released") or []:
                rel.setdefault((int(x[0]), tuple(x[1])), []).append(
                    (p["cycle"], [tuple(y) for y in x[2]]))
        surv = {}
        for row in ((revsurv or {}).get(a) or []):
            surv.setdefault((int(row["level"]), tuple(row["key"])), []).append(row)
        n_rel = sum(len(v) for v in rel.values())
        rel_keep = 0
        L.append("")
        L.append(f"  {a}: {sum(len(v) for v in recs.values())} revocations of {len(recs)} keys; "
                 f"{n_rel} re-offers  (why: worth = worthless, no_co = no consumer; "
                 f"{oc} keeps? Y/N on the same fires)")
        wrong_never = []
        for (lv, k), rv in sorted(recs.items(), key=lambda t: t[1][0][0]):
            rs = rel.get((lv, k), [])
            for c, q in rv:
                if any(cr > c for cr, _ in rs) and q.get("keep_other"):
                    rel_keep += 1
            rv_s = ", ".join("c%d (%s, %s)" % (c, str(q.get("why"))[:5],
                                                "Y" if q.get("keep_other") else "N")
                             for c, q in rv)
            rs_s = ", ".join("c%d (+%d new)" % (cr, len(nw)) for cr, nw in rs) or "never"
            L.append(f"    L{lv} {str(list(k)):<26} revoked {rv_s:<44} re-offered {rs_s}")
            c_last, q_last = rv[-1]
            if not any(cr > c_last for cr, _ in rs):
                cons0 = {tuple(x[0]) for x in (q_last.get("consumers") or [])}
                later = [x for x in surv.get((lv, k), []) if x["cycle"] > c_last]
                if later:
                    nb = sum(1 for x in later if x.get("buildable"))
                    new_c = sorted({tuple(y) for x in later if x.get("buildable")
                                    for y in (x.get("cons") or [])} - cons0)
                    ncs = [x.get("n_consumers") for x in later if x.get("buildable")]
                    L.append(f"        still revoked at the end: {len(cons0)} consumers at "
                             f"c{c_last}; at the {len(later)} later saves buildable at {nb}, "
                             f"consumers there {ncs}; consumers never seen at the revocation: "
                             f"{len(new_c)} {[list(y) for y in new_c][:4]}")
                else:
                    L.append(f"        still revoked at the end: {len(cons0)} consumers at "
                             f"c{c_last}; no later save surveyed")
                if q_last.get("why") == "worthless" and q_last.get("keep_other"):
                    wrong_never.append((lv, k, c_last))
        wrong = [(lv, k, c) for (lv, k), rv in recs.items() for c, q in rv
                 if q.get("why") == "worthless" and q.get("keep_other")]
        back = [(lv, k, c) for lv, k, c in wrong
                if any(cr > c for cr, _ in rel.get((lv, k), []))]
        L.append(f"    re-offers that released a revocation the {oc} would have KEPT: "
                 f"{rel_keep} of {n_rel}")
        L.append(f"    worthless revocations the {oc} would have kept: {len(wrong)} (of "
                 f"{len({(lv, k) for lv, k, c in wrong})} keys); later re-offered: {len(back)}; "
                 f"never: {len(wrong) - len(back)} — keys still revoked at the end, their "
                 f"last revocation one the {oc} would have kept: "
                 f"{[(lv, list(k), c) for lv, k, c in wrong_never]}")
    # ---- M-6 the no-consumer revocations against the ungated arm ------------------------ #
    L.append("")
    L.append("M-6 THE NO-CONSUMER REVOCATIONS, AGAINST THE UNGATED ARM. Every revocation for want "
             "of a consumer (the")
    L.append("    window elapsed with nothing at support one level up): which of st_gn_yk's walked "
             "keys one level up")
    L.append("    have the revoked key as a half, and the revoked key's own consumers IN st_gn_yk "
             "at every save")
    L.append("    (`rb_consumer_survey` over the ungated arm's saves, `<ref tag>_consumer_survey.json`"
             "): the first save")
    L.append("    at which st_gn_yk held a consumer for it, and the key's age then in the panel's "
             "observations (the")
    L.append("    window is 160). 'none by c201' = no consumer at any save.")
    rs = {}
    for row in (ref_survey or []):
        rs.setdefault((int(row["level"]), tuple(row["key"])), []).append(row)
    for a, r in arms.items():
        adm = r.get("adm") or {}
        mode = adm.get("grade")
        if not mode:
            continue
        nc = [(c, lv, k, q) for c, lv, k, q in _kp(adm) if q.get("why") == "no_consumer"]
        summ, hit = {}, {}
        L.append("")
        L.append(f"  {a}: {len(nc)} no-consumer revocations")
        for c, lv, k, q in nc:
            ups = sorted(K for K in walked.get(lv + 1, set()) if k in _halves(K))
            hist = sorted(rs.get((lv, k), []), key=lambda x: x["cycle"])
            first = next((x for x in hist if int(x.get("n_consumers") or 0) > 0), None)
            if not hist:
                ug = "not admitted in st_gn_yk at any surveyed save"
            elif first is None:
                ug = (f"no consumer at any of st_gn_yk's {len(hist)} saves "
                      f"(c{hist[0]['cycle']}–c{hist[-1]['cycle']})")
            else:
                w_ = int(adm.get("gr_window") or 160)
                ag_ = first.get("age_obs")
                ug = (f"first consumer in st_gn_yk at the c{first['cycle']} save "
                      f"({first.get('n_consumers')} consumers, key age {ag_} obs"
                      f"{' — past the ' + str(w_) + '-obs window' if ag_ is not None and ag_ > w_ else ''}; "
                      f"{'AFTER' if first['cycle'] > c else 'by'} this revocation at c{c})")
            back = any(tuple(x[1]) == k and int(x[0]) == lv and p_["cycle"] > c
                       for p_ in adm.get("gr_passes") or [] for x in p_.get("released") or [])
            L.append(f"    c{c:>3} L{lv} {str(list(k)):<26} age {q.get('age_obs')} obs; "
                     f"half of {len(ups)} of st_gn_yk's walked L{lv + 1} keys "
                     f"{[list(K) for K in ups][:4]}{' …' if len(ups) > 4 else ''}; {ug}; "
                     f"re-offered later here: {'yes' if back else 'no'}")
            cat = ("not admitted in st_gn_yk" if not hist else "never consumed in st_gn_yk"
                   if first is None else "consumed in st_gn_yk by the revocation"
                   if first["cycle"] <= c else "first consumed in st_gn_yk after it")
            summ.setdefault(cat, []).append((c, lv, k))
            if ups:
                hit.setdefault("keys", []).append((lv, k))
                hit.setdefault("ups", set()).update((lv + 1, K) for K in ups)
        L.append(f"    summary: {len(nc)} no-consumer revocations; " + "; ".join(
            f"{cat_}: {len(v_)}" for cat_, v_ in sorted(summ.items())))
        L.append(f"    of them halves of st_gn_yk's walked keys one level up: "
                 f"{len(hit.get('keys', []))} {[(lv_, list(k_)) for lv_, k_ in hit.get('keys', [])]}, "
                 f"covering {len(hit.get('ups', set()))} of its keys")
        # the END state: st_gn_yk's walked keys with a half this arm held revoked at the end, by
        # the reason of that revocation
        endr = {}
        for lv_, qs_ in (adm.get("revoked") or {}).items():
            for q_ in qs_:
                endr[(int(lv_), tuple(q_["key"]))] = q_.get("why")
        for lv_up in (3, 4, 5):
            ks_ = walked.get(lv_up, set())
            if not ks_:
                continue
            by_ = {}
            for K in ks_:
                ws_ = {endr[(lv_up - 1, h)] for h in _halves(K) if (lv_up - 1, h) in endr}
                if ws_:
                    by_[" + ".join(sorted(ws_))] = by_.get(" + ".join(sorted(ws_)), 0) + 1
            L.append(f"    st_gn_yk's walked L{lv_up} keys with a half held revoked at the END, by "
                     f"that revocation's reason: {by_ or 'none'} (of {len(ks_)})")
    # ---- the bill ------------------------------------------------------------------------ #
    L.append("")
    L.append("M-5 THE BILL'S SCALE (per arm): fires include one unfired base pool per level per "
             "pass; every fire is")
    L.append("    one pool of `n_aud` instances; the world arm bills each fire's pool.")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        if not adm.get("grade"):
            continue
        P = adm.get("gr_passes") or []
        n_f = sum(int(p.get("n_fires") or 0) for p in P)
        n_g = sum(1 for _ in _kp(adm))
        n_pr = sum(1 for c, lv, k, q in _kp(adm) if q.get("why") in ("kept", "worthless"))
        c0 = min((p["cycle"] for p in P), default=None)
        n_cyc = (201 - c0 + 1) if c0 else 0
        bill = int(adm.get("grade_world_billed") or 0)
        walk = int(adm.get("n_world_billed") or 0)
        L.append(f"  {a}: passes {len(P)} (from c{c0}); fires {n_f} (= {n_f / max(1, len(P)):.1f} "
                 f"a pass, {n_f / max(1, n_cyc):.2f} a cycle over c{c0}-c201); graded key-passes "
                 f"{n_g} ({n_f / max(1, n_g):.2f} fires each), priced {n_pr}; world queries billed "
                 f"to the grade {bill:,} ({bill / max(1, n_f):.0f} a fire, "
                 f"{bill / max(1, n_cyc):,.0f} a cycle); the walk's own world bill {walk:,}")


def _sos_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reduce_sostenuto", os.path.join(SOS, "reduce_sostenuto.py"))
    RS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RS)
    return RS


def grade_reduce(tag, sos_tags, bank_tag, bank_arm="sb_sv_yk", figs=False, arms_=GRADE_ARMS,
                 none_arm="st_gn_yk", seg_logs=()):
    """sostenuto's [A]..[K] with the graded arms added as columns, then [L] and [M]. The
    graded arms are read from the tag and from its companions (`GRADE_WITH`, `GRADE_EXTRA`),
    so a corrected arm reads beside the arms it corrects."""
    arms_tag = {}
    RS = _sos_module()
    arms_ = tuple(arms_) + tuple(a for a in DIET_ARMS if a not in arms_)
    RS.ORDER = tuple(RS.ORDER) + tuple(arms_)
    RS.GATE_OF.update({a: ("grade:read+diet" if a in DIET_ARMS else
                           "grade:" + ("world" if "gw" in a else "read")) for a in arms_})
    RS.COL.update({"rb_gw_yk": "#56B4E9", "rb_gr_yk": "#000000", "rb_grd_yk": "#CC79A7"})
    RS.LAB.update({a: ("graded, read, diet" if a in DIET_ARMS else
                       "graded, world" if "gw" in a else "graded, read") for a in arms_})
    sos_fig = RS.FIG
    RS.FIG = FIG
    base_load = RS.load

    def load(t_, arm, root=None):
        r_ = base_load(t_, arm, root=root)
        if r_ is None and root is None:
            r_ = base_load(t_, arm, root=os.path.join(sos_fig, t_))
        return r_
    RS.load = load
    out = os.path.join(FIG, f"{tag}_reduction.txt")
    comp = GRADE_WITH.get(tag, ()) + GRADE_EXTRA.get(tag, ())
    RS.reduce_tag(tag, bank_tag, bank_arm, out, mk_figs=figs,
                  with_tags=tuple(comp) + tuple(sos_tags))
    arms = {}
    for t_ in (tag,) + tuple(comp):
        for a in arms_:
            if a not in arms:
                r_ = load(t_, a)
                if r_ is not None:
                    arms[a] = r_
                    arms_tag[a] = t_
    rb_splice(arms, tuple(seg_logs) + tuple(lp for t_ in (tag,) + tuple(comp)
                                           for lp in SEG_LOGS.get(t_, ())))
    ref = None
    for t_ in sos_tags:
        ref = ref or load(t_, none_arm)
    L = []
    grade_section(arms, ref, L)
    diet, revsurv, tt = {}, {}, None
    for t_ in (tag,) + tuple(comp):
        for nm_, dst in (("diet", diet), ("revoked_survey", revsurv)):
            fp_ = os.path.join(FIG, f"{t_}_{nm_}.json")
            if os.path.isfile(fp_):
                with open(fp_) as fh:
                    for a, v in json.load(fh).items():
                        if arms_tag.get(a) == t_:
                            dst[a] = v
        sp = os.path.join(FIG, t_, "setup.json.gz")
        if tt is None and os.path.isfile(sp):
            with gzip.open(sp, "rt") as fh:
                tt = json.load(fh).get("true_tables")
    ref_survey = None
    for t_ in REF_OF.get(tag, ()):
        fp_ = os.path.join(FIG, f"{t_}_consumer_survey.json")
        if os.path.isfile(fp_):
            with open(fp_) as fh:
                ref_survey = json.load(fh)
            break
    grade_forensics(tag, arms, ref, L, diet=diet, true_tables=tt, revsurv=revsurv,
                    ref_survey=ref_survey)
    with open(out, "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {out}")


# the two seeds' graded tags, each beside sostenuto's own tags of the same seed (round 2 first,
# then round 1: sostenuto's `mk_seedtable` order) and its banked no-walk reference
GRADE_SEEDS = (("rb_g1", 0, "sb_s1"), ("rb_g2", 2, "sb_s2"))
SOS_OF = {"rb_g1": ("st2_s1", "st_s1"), "rb_g2": ("st2_s2", "st_s2")}


def seedtable(out=None, seeds=GRADE_SEEDS):
    """THE TWO-SEED TABLE OF RECORD with the graded arms beside sostenuto's five gates:
    sostenuto's own `mk_seedtable` (imported, its arm list extended, its arm files read from
    this node for the graded arms and from `sostenuto/figures/` for the five), then (v) the
    grade per seed. `--grade` must have been run for each tag first (its reduction carries the
    sections [A]..[L] that part (b) sets side by side). Facts only; the seeds never averaged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk_seedtable",
                                                  os.path.join(SOS, "mk_seedtable.py"))
    MS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MS)
    sos_order = tuple(MS.ORDER)
    MS.ORDER = sos_order + tuple(GRADE_ARMS) + tuple(DIET_ARMS)
    MS.GATE.update({a: "grade:" + ("world" if "gw" in a else "read") for a in GRADE_ARMS})
    MS.GATE.update({a: "read+diet" for a in DIET_ARMS})
    sos_fig = MS.FIG

    def arm_files(tag):
        out_ = {}
        for t_ in (tag,) + tuple(GRADE_EXTRA.get(tag, ())):
            for a in tuple(GRADE_ARMS) + tuple(DIET_ARMS):
                p_ = os.path.join(FIG, t_, a, "results.json")
                if a not in out_ and os.path.isfile(p_):
                    with open(p_) as fh:
                        out_[a] = json.load(fh)
        rb_splice(out_, tuple(lp for t_ in (tag,) + tuple(GRADE_EXTRA.get(tag, ()))
                              for lp in SEG_LOGS.get(t_, ())))
        for t_ in SOS_OF.get(tag, ()):
            for a in sos_order:
                p_ = os.path.join(sos_fig, t_, a, "results.json")
                if a not in out_ and os.path.isfile(p_):
                    with open(p_) as fh:
                        out_[a] = json.load(fh)
        return out_
    MS.arm_files = arm_files
    MS.FIG = FIG
    MS.SEEDS = tuple(q for q in seeds if os.path.isfile(os.path.join(FIG, f"{q[0]}_reduction.txt")))
    assert MS.SEEDS, "run reduce_rubato.py --grade --tag rb_g1 first"
    out = out or os.path.join(FIG, "rb_grade_seedtable.txt")
    argv0 = sys.argv
    sys.argv = [argv0[0], "--out", out]
    try:
        MS.main()
    finally:
        sys.argv = argv0
    txt = open(out).read().splitlines()
    txt[0] = ("[rubato] ADMIT-THEN-GRADE BESIDE SOSTENUTO'S FIVE GATES — THE TWO SEEDS, "
              "NEVER AVERAGED")
    txt.insert(2, "  graded arms rb_gr_yk (read), rb_gw_yk (world) and, seed 0, rb_grd_yk "
               "(read with the per-span diet, from rb_g1d's c50 restore) from "
               + " / ".join(t for t, _, _ in MS.SEEDS) + ", restored from each seed's "
               "st_gn_yk saved state; sostenuto's five from " + " / ".join(
                   "+".join(SOS_OF[t]) for t, _, _ in MS.SEEDS) + " (not re-run)")
    # (v) the grade, per seed: inserted before part (b)
    V = ["", MS.SEP, "(v) THE GRADE — revocations, re-offers, and the level above, per seed", MS.SEP,
         ""]
    V.append(f"      {'seed':>4} {'arm':<9} {'from':>5} {'1st rev':>7} {'revoked':>7} "
             f"{'none':>5} {'worth':>5} {'re-off':>6} {'end L2/L3/L4':>13} {'fires':>6} "
             f"{'billed':>7} {'L3 blocked':>10} {'L4 blocked':>10}")
    for t, sd, _ in MS.SEEDS:
        A_ = arm_files(t)
        nd = {}
        for ps in ((A_.get("st_gn_yk") or {}).get("adm") or {}).get("passes") or []:
            for lr in ps["levels"]:
                for st in lr["steps"]:
                    nd.setdefault(lr["level"], set()).add(tuple(st["key"]))
        for a in tuple(GRADE_ARMS) + tuple(DIET_ARMS):
            if a not in A_:
                continue
            adm = A_[a].get("adm") or {}
            _sg = rb_segments(A_[a].get("rb") or {})
            rf = ((_sg[0].get("resumed_from") if _sg else None) or {})
            end = {int(lv): {tuple(q["key"]) for q in qs}
                   for lv, qs in (adm.get("revoked") or {}).items()}
            ever = {}
            for p_ in adm.get("gr_passes") or []:
                for lv, k in p_.get("revoked") or []:
                    ever.setdefault(int(lv), set()).add(tuple(k))
            blk = {}
            for lv in (3, 4):
                ks = nd.get(lv, set())
                blk[lv] = (f"{sum(1 for k in ks if any(h in end.get(lv - 1, set()) for h in (k[:len(k) // 2], k[len(k) // 2:])))}"
                           f"/{len(ks)}")
            V.append(f"      {sd:>4} {a:<9} {('c' + str(rf.get('cycle'))):>5} "
                     f"{('c' + str(adm.get('first_revoke'))) if adm.get('first_revoke') is not None else '—':>7} "
                     f"{adm.get('n_revoke', 0):>7} {adm.get('n_revoke_nocons', 0):>5} "
                     f"{adm.get('n_revoke_worthless', 0):>5} {adm.get('n_reoffer', 0):>6} "
                     f"{'/'.join(str(len(end.get(lv, ()))) for lv in (2, 3, 4)):>13} "
                     f"{sum(int(p_.get('n_fires') or 0) for p_ in adm.get('gr_passes') or []):>6} "
                     f"{adm.get('grade_world_billed', 0):>7} {blk[3]:>10} {blk[4]:>10}")
            V.append(f"      {'':>4} {'':<9} ever revoked per level "
                     f"{ {lv: len(v) for lv, v in sorted(ever.items())} }")
    V.append("      revoked: none = no consumer at support in the window, worth = consumers "
             "fired and none above the margin;")
    V.append("      end L2/L3/L4 = keys revoked at the run's end; L3/L4 blocked = of the keys "
             "the ungated arm (st_gn_yk)")
    V.append("      walked at level l, how many have a half this arm held REVOKED at level l-1 "
             "at the run's end")
    cut = next((i for i, x in enumerate(txt) if x.startswith("[A]")), len(txt)) - 1
    txt = txt[:cut] + V + txt[cut:]
    with open(out, "w") as fh:
        fh.write("\n".join(txt) + "\n")
    print("\n".join(V))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-check", action="store_true")
    ap.add_argument("--r1", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--seedtable", action="store_true")
    ap.add_argument("--seg-log", action="append", default=[],
                    help="launch log of an earlier container segment (DESIGN §5.5)")
    ap.add_argument("--sos-tag", action="append", default=[])
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--tag", default="rb_s1")
    ap.add_argument("--bank-tag", default="st_s1")
    ap.add_argument("--arm", default="st_gn_yk")
    ap.add_argument("--local", default=None)
    a = ap.parse_args()
    if a.bank_check:
        ok = bank_check(a.tag, a.bank_tag, a.arm,
                        tags=(f"{a.tag}__{a.arm}", f"{a.bank_tag}__{a.arm}", a.tag, a.bank_tag))
        sys.exit(0 if ok else 1)
    if a.r1:
        r1_table(a.tag, a.local)
    if a.grade:
        grade_reduce(a.tag, a.sos_tag or ["st_s1", "st2_s1"], a.bank_tag, figs=a.figs,
                     seg_logs=tuple(a.seg_log) or SEG_LOGS.get(a.tag, ()))
    if a.seedtable:
        seedtable()


if __name__ == "__main__":
    main()
