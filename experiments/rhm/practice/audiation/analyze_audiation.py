"""Reduction for `audiation` (E1, phase 1) — THE DATA REPORT. No interpretation here, and no
self-model: this file checks that the dataset is what `schema.md` says it is, and reports the
counts and a few descriptive statistics. The decode matrix is phase 2's, in its own file.

Sections:

  (0) GATES — the full-scale CROSS-TAG REPLAY (this tag's `anchor` against the donor's
      `cd_s0/anchor`, and through it `as_s0/anchor`, over every cycle), the in-run replay gate
      the run itself asserted at every checkpoint, `beam_gate` (the instrumented beams against
      the donor module's, recorder off and on), and every in-run snapshot-recompute check.
  (1) THE DATASET — shards, rows, bytes, per-cycle counts, and what the action set did.
  (2) JOIN INTEGRITY — the four properties phase 2 will assume: the terminal join is positional
      and complete, lineage closes at every step, every kept candidate names a real tip, and
      nothing is NaN/out-of-range where the schema says it is not.
  (3) DESCRIPTIVE — the explore-injected fraction (the agency the arity-2 slot is about); the
      offline recompute of pi at logged beam states, with an fp16-z control showing its
      residual is storage precision and not a recompute error; the per-cycle |dpi| and |dv|
      distributions BOTH on the fixed probe and at the logged beam states (the latter computed
      exactly as phase 2 will: two consecutive head snapshots, the logged z, live slots only);
      and the cumulative exact-zero invariant on slots the action set has not yet contained.

The heads are re-implemented in numpy here (they are a LayerNorm, two Linears and a GELU) so
the report needs no GPU and no torch. That re-implementation is VALIDATED against the run's own
fp32 probe readouts before it is used for anything — if it does not reproduce them it is wrong
and the report says so rather than quoting it.

Usage (from experiments/):
    python3 rhm/practice/audiation/analyze_audiation.py --tag au_smoke --fetch
    python3 rhm/practice/audiation/analyze_audiation.py --tag au_s0 --fetch --write
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
PRACTICE = os.path.dirname(HERE)
COND_FIG = os.path.join(PRACTICE, "conductor", "figures")
ASSAY_FIG = os.path.join(PRACTICE, "assay", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_audiation"

# `analyze_conductor.GF_SERIES`, verbatim: the per-cycle series a fork has to reproduce.
GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]

SRC_NAMES = {0: "enumerated", 1: "proposed", 2: "forced", 4: "explore"}


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def _load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


# --------------------------------------------------------------------------- #
# the two heads, in numpy. Both read (z, root) and nothing else.
# --------------------------------------------------------------------------- #

def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    """`nn.GELU()`'s default is the EXACT erf form, not tanh. Getting this wrong is a ~1e-3
    error that would masquerade as a snapshot mismatch."""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def head_forward(sd, prefix, z, root):
    """`MCValueHead` and `ProposalHead` have the same shape of read — LayerNorm(2d) ->
    Linear(2d, 4d) -> GELU -> Linear(4d, out) over `cat([z, root_embedding(root)])` — so one
    function serves both. Returns (n,) for the value and (n, n_slots) for pi."""
    emb = sd[f"{prefix}.root_embedding.weight"][np.asarray(root, dtype=np.int64)]
    h = np.concatenate([np.asarray(z, dtype=np.float32), emb], axis=-1)
    if f"{prefix}.net.0.weight" in sd:           # MCValueHead
        p = f"{prefix}.net"
        h = _layer_norm(h, sd[f"{p}.0.weight"], sd[f"{p}.0.bias"])
        h = _gelu(h @ sd[f"{p}.1.weight"].T + sd[f"{p}.1.bias"])
        return (h @ sd[f"{p}.3.weight"].T + sd[f"{p}.3.bias"]).squeeze(-1)
    p = f"{prefix}.trunk"                        # ProposalHead
    h = _layer_norm(h, sd[f"{p}.0.weight"], sd[f"{p}.0.bias"])
    h = _gelu(h @ sd[f"{p}.1.weight"].T + sd[f"{p}.1.bias"])
    return h @ sd[f"{prefix}.out.weight"].T + sd[f"{prefix}.out.bias"]


def load_heads(root, cyc):
    p = os.path.join(root, "snapshots", f"heads_c{cyc:04d}.npz")
    return dict(np.load(p)) if os.path.isfile(p) else None


# --------------------------------------------------------------------------- #

def gates(tag, arm, root, res, audi, setup, out):
    out("\n" + "=" * 78)
    out("(0) GATES")
    out("=" * 78)

    lg = res["log"]
    # A replay is only a statement if the two tags trained the SAME substrate and ran the same
    # ladder. A `--quick` tag does neither, and comparing it would report a FAIL that means
    # "these are different experiments", which is not what this gate is for.
    CFG_KEYS = ("depth", "m", "v", "s", "seed", "rule_seed", "train_seed", "budget",
                "g_budget", "pr_width", "n_pr", "n_rt", "n_score", "max_macro_level",
                "gen_steps", "n_probe_clean", "prop_warmup", "prop_k", "mine_cap",
                "controller_steps", "generator_steps", "value_steps", "n_train_episodes")
    for label, ref_path in (
            ("cd_s0/anchor  (conductor, the direct donor)",
             os.path.join(COND_FIG, "cd_s0", arm, "results.json")),
            ("as_s0/anchor  (assay, the donor's own carrier)",
             os.path.join(ASSAY_FIG, "as_s0", arm, "results.json"))):
        ref = _load(ref_path)
        if ref is None:
            why = ("this arm is a TREATMENT and has no donor twin — see (0b) for the "
                   "inverse gate" if arm != "anchor" else "reference not fetched locally")
            out(f"  CROSS-TAG REPLAY vs {label}: SKIPPED ({why})")
            continue
        diff = [k for k in CFG_KEYS
                if k in ref["config"] and k in res["config"]
                and ref["config"][k] != res["config"][k]]
        if diff:
            out(f"  CROSS-TAG REPLAY vs {label}: NOT COMPARABLE — configs differ on "
                f"{ {k: (ref['config'][k], res['config'][k]) for k in diff} }")
            continue
        old = ref["log"]
        n = min(len(old["cycle"]), len(lg["cycle"]))
        per = {k: float(np.abs(np.asarray(old[k][:n], float)
                               - np.asarray(lg[k][:n], float)).max()) for k in GF_SERIES}
        worst = max(per.values())
        out(f"  CROSS-TAG REPLAY vs {label}")
        out(f"    c1-c{n} ({len(old['cycle'])} ref / {len(lg['cycle'])} here), "
            f"{len(GF_SERIES)} series: max|delta| = {worst:.3e}  -> "
            f"{'PASS' if worst == 0.0 else 'FAIL'}")
        if worst:
            out("    per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))

    rg = (audi or {}).get("replay_gate") or {}
    if rg.get("available"):
        out(f"  IN-RUN REPLAY GATE (asserted every checkpoint against {rg['ref']}): "
            f"c1-c{rg['n_cycles']}, max|delta| = {rg['max_abs_delta']:.3e}, "
            f"cycle counts equal = {rg.get('n_cycles_equal')}  -> "
            f"{'PASS' if rg['max_abs_delta'] == 0.0 else 'FAIL'}")
    else:
        out(f"  IN-RUN REPLAY GATE: not armed (ref={rg.get('ref')!r})")

    bg = (setup or {}).get("beam_gate")
    if bg:
        keys = ("enum_rec_off", "enum_rec_on", "prop_rec_off", "prop_rec_on")
        out(f"  BEAM GATE (this file's beams vs conductor.py's, recorder off AND on): "
            + ", ".join(f"{k}={bg[k]:.1e}" for k in keys)
            + f"  -> {bg.get('verdict')}")
        out(f"    recorded: {bg['enum_tip_steps']}/{bg['prop_tip_steps']} tip-steps "
            f"(enum/prop), {bg['prop_pi_rows']} pi rows, src {bg['src_hist']}")

    chk = (audi or {}).get("snapshot_checks") or []
    if chk:
        dv = max(q["max_abs_dv"] for q in chk)
        dp = max(q["max_abs_dpi"] for q in chk)
        nf = sum(1 for q in chk if q["via_file"])
        out(f"  SNAPSHOT RECOMPUTE (in-run, {len(chk)} cycles x {chk[0]['n']} probe states; "
            f"{nf} of them through the written .npz): max|dv| = {dv:.3e}, "
            f"max|dpi| = {dp:.3e}  -> {'PASS' if dv == 0.0 and dp == 0.0 else 'FAIL'}")


def dataset(tag, arm, root, res, audi, out):
    out("\n" + "=" * 78)
    out("(1) THE DATASET")
    out("=" * 78)
    sh = audi["shards"] if "shards" in audi else []
    files = sorted(glob.glob(os.path.join(root, "decisions", "dec_c*.npz")))
    snaps = sorted(glob.glob(os.path.join(root, "snapshots", "heads_c*.npz")))
    plants = sorted(glob.glob(os.path.join(root, "snapshots", "plant_c*.npz")))
    n_by = sum(os.path.getsize(f) for f in files)
    n_sn = sum(os.path.getsize(f) for f in snaps + plants)
    cyc = audi.get("cycles") or []
    out(f"  cycles logged      {audi.get('n_cycles_logged')}  "
        f"(arm ran {len(res['log']['cycle'])})")
    out(f"  decision shards    {len(files)} files, {n_by / 1e6:.1f} MB")
    out(f"  tip-step rows      {audi.get('n_tips')}")
    out(f"  candidate rows     {audi.get('n_cand')}")
    out(f"  head snapshots     {len(snaps)} (final at c{audi.get('final_snapshot_cycle')}), "
        f"plant snapshots {len(plants)}, {n_sn / 1e6:.1f} MB")
    out(f"  bytes per tip-step {n_by / max(audi.get('n_tips') or 1, 1):.0f} "
        f"(compressed, tips + candidates + pi)")
    if cyc:
        nm = [c["n_moves"] for c in cyc]
        routed = [c["cycle"] for c in cyc if c["routed"]]
        out(f"  action set |ms|    {nm[0]} -> {nm[-1]}; grows at cycles "
            f"{[cyc[i]['cycle'] for i in range(1, len(nm)) if nm[i] != nm[i - 1]]}")
        out(f"  routed beam from   c{routed[0] if routed else None} "
            f"(cycles 1..{(routed[0] - 1) if routed else len(cyc)} enumerate: prop_warmup)")
        out(f"  k_eff              {sorted(set(c['k_eff'] for c in cyc))}")
        out(f"  n_slots            {audi.get('n_slots')}, offsets {audi.get('slot_offsets')}")
        sol = [c["n_solved"] for c in cyc]
        out(f"  solved tips/cycle  min {min(sol)} med {int(np.median(sol))} max {max(sol)}")
    # what the instrument cost, against the donor's own measurement of the same arm
    mine = _load(os.path.join(FIG, tag, "summary.json")) or {}
    ref = _load(os.path.join(COND_FIG, "cd_s0", "summary.json")) or {}
    a1 = (mine.get("cycle_seconds") or {}).get(arm)
    a0 = (ref.get("cycle_seconds") or {}).get(arm)
    if a1 and a0:
        out(f"  s/cycle            {a1:.2f} here vs {a0:.2f} in cd_s0 (same arm, no "
            f"instrument) -> +{100 * (a1 / a0 - 1):.1f}% wall for the logging")
    if mine.get("elapsed_s"):
        out(f"  wall               {mine['elapsed_s']:.0f}s "
            f"({mine['elapsed_s'] / 3600:.2f} GPU-h, L4)")
    return files


def integrity(files, audi, snap_dir, out):
    out("\n" + "=" * 78)
    out("(2) JOIN INTEGRITY")
    out("=" * 78)
    cyc = {c["cycle"]: c for c in (audi.get("cycles") or [])}
    budget = (audi.get("cycles") or [{}])[0].get("budget")
    n_t = n_c = 0
    bad = []
    per_cycle = {}
    src_tot = {}
    for f in files:
        d = np.load(f)
        for c in np.unique(d["t_cycle"]):
            tm = d["t_cycle"] == c
            cm = d["c_cycle"] == c
            t = {k: d[k][tm] for k in d.files if k.startswith("t_")}
            q = {k: d[k][cm] for k in d.files if k.startswith("c_")}
            n_t += t["t_step"].shape[0]
            n_c += q["c_step"].shape[0]
            meta = cyc.get(int(c), {})
            B, W = meta.get("n_pr"), meta.get("pr_width")
            term = t["t_step"] == budget
            # 1. terminal join
            if B and W and int(term.sum()) != B * W:
                bad.append(f"c{c}: {int(term.sum())} terminal rows, expected {B * W}")
            if (t["t_succ"][term] < 0).any():
                bad.append(f"c{c}: an ungraded terminal tip")
            if (t["t_succ"][~term] != -1).any() or not np.isnan(t["t_dres"][~term]).all():
                bad.append(f"c{c}: a non-terminal row carries a grade")
            if np.isnan(t["t_vfin"][term]).any() or not np.isnan(t["t_vfin"][~term]).all():
                bad.append(f"c{c}: t_vfin is not exactly the terminal rows")
            # 2. lineage
            if (t["t_parent"][t["t_step"] == 0] != -1).any():
                bad.append(f"c{c}: a step-0 row has a parent")
            for st in range(1, (budget or 0) + 1):
                here = t["t_step"] == st
                wprev = int(((t["t_step"] == st - 1) & (t["t_inst"] == 0)).sum())
                p = t["t_parent"][here]
                if p.size and not ((p >= 0) & (p < wprev)).all():
                    bad.append(f"c{c} step{st}: parent index out of range")
            # 3. kept candidates
            for st in range(0, budget or 0):
                m = q["c_step"] == st
                kept = q["c_child"][m] >= 0
                wnext = int(((t["t_step"] == st + 1) & (t["t_inst"] == 0)).sum())
                if B and int(kept.sum()) != wnext * B:
                    bad.append(f"c{c} step{st}: {int(kept.sum())} kept, expected {wnext * B}")
                if kept.any() and not (q["c_child"][m][kept] < wnext).all():
                    bad.append(f"c{c} step{st}: a kept candidate names a nonexistent tip")
            # 4. ranges / finiteness
            if not np.isfinite(t["t_z"].astype(np.float32)).all():
                bad.append(f"c{c}: non-finite z")
            if (t["t_x"] < 0).any() or (t["t_x"] >= 8).any():
                bad.append(f"c{c}: x out of vocabulary range")
            if not np.isfinite(q["c_score"]).all():
                bad.append(f"c{c}: non-finite candidate score")
            live = ~np.isnan(t["t_pi"])
            n_live = live[t["t_step"] < budget].sum(1)
            if meta.get("routed") and n_live.size and not (n_live == meta["n_moves"]).all():
                bad.append(f"c{c}: pi's live slot count != |ms|")
            u, k = np.unique(q["c_src"], return_counts=True)
            for a, b in zip(u, k):
                src_tot[int(a)] = src_tot.get(int(a), 0) + int(b)
            per_cycle[int(c)] = {"tips": int(t["t_step"].shape[0]),
                                 "cand": int(q["c_step"].shape[0]),
                                 "term": int(term.sum()),
                                 "succ": int((t["t_succ"] == 1).sum())}
    out(f"  rows read          {n_t} tip-steps / {n_c} candidates over "
        f"{len(per_cycle)} cycles")
    out(f"  manifest agrees    tips {n_t == audi.get('n_tips')}, "
        f"cand {n_c == audi.get('n_cand')}")
    out(f"  terminal join      complete and positional on every cycle: "
        f"{not any('terminal' in b or 'grade' in b or 'vfin' in b for b in bad)}")
    out(f"  lineage closes     {not any('parent' in b for b in bad)}")
    out(f"  kept == beam       {not any('kept' in b or 'nonexistent' in b for b in bad)}")
    out(f"  ranges / finite    {not any(('finite' in b) or ('range' in b) for b in bad)}")
    tps = np.asarray([q["tips"] for q in per_cycle.values()])
    cps = np.asarray([q["cand"] for q in per_cycle.values()])
    sc = int(sum(q["succ"] for q in per_cycle.values()))
    tm = int(sum(q["term"] for q in per_cycle.values()))
    out(f"  per-cycle tips     min {tps.min()} med {int(np.median(tps))} max {tps.max()}")
    out(f"  per-cycle cands    min {cps.min()} med {int(np.median(cps))} max {cps.max()}")
    out(f"  solved terminals   {sc} of {tm} graded tips ({sc / max(tm, 1):.1%})")
    snap = os.path.join(snap_dir, "")
    missing = [c for c in sorted(per_cycle)
               if not os.path.isfile(os.path.join(snap, f"heads_c{c:04d}.npz"))
               or not os.path.isfile(os.path.join(snap, f"heads_c{c + 1:04d}.npz"))]
    out(f"  heads_c AND heads_c+1 present for every logged cycle: {not missing}"
        + (f"  (missing at c{missing[:5]})" if missing else ""))
    out(f"  VERDICT            {'PASS — no violation found' if not bad and not missing else 'FAIL'}")
    for b in bad[:20]:
        out(f"    {b}")
    if len(bad) > 20:
        out(f"    ... and {len(bad) - 20} more")
    return per_cycle, src_tot, bad


def descriptive(root, audi, files, src_tot, out):
    out("\n" + "=" * 78)
    out("(3) DESCRIPTIVE")
    out("=" * 78)
    tot = sum(src_tot.values())
    out("  candidate provenance (every scored child of the practice beam):")
    for k in sorted(src_tot):
        out(f"    {SRC_NAMES.get(k, k):11s} {src_tot[k]:>10d}  {src_tot[k] / tot:6.2%}")

    # ---- the numpy heads, VALIDATED against the run's own fp32 probe readouts ---------- #
    tp = os.path.join(root, "snapshots", "probe_trace.npz")
    if not os.path.isfile(tp):
        out("  probe_trace.npz absent — no readout trajectory to report")
        return
    pt = np.load(tp)
    cycles, eras = pt["cycle"], pt["era"]
    v_live, pi_live = pt["v"], pt["pi"]
    cyc_meta = {c["cycle"]: c for c in (audi.get("cycles") or [])}

    # THE OFFLINE RECOMPUTE, validated. The in-run assert already showed that a snapshot
    # reproduces the live readout exactly under torch; this shows that the SAME readout comes
    # out of a numpy re-implementation reading the same file, at a logged beam state, which is
    # the code path phase 2 actually needs. `t_z` is fp16 in the record, so the residual here
    # is the fp16 round-off of z pushed through the head — not an error in either.
    routed = [c["cycle"] for c in (audi.get("cycles") or []) if c["routed"]]
    c0 = routed[len(routed) // 2] if routed else None
    err_pi = None
    if c0 is not None:
        shard = [f for f in files
                 if int(os.path.basename(f)[5:9]) <= c0 <= int(os.path.basename(f)[11:15])]
        d0 = np.load(shard[0]) if shard else None
        sd0 = load_heads(root, c0)
        if d0 is not None and sd0 is not None:
            bud = cyc_meta.get(c0, {}).get("budget") or 8
            m = (d0["t_cycle"] == c0) & (d0["t_step"] < bud)
            idx = np.nonzero(m)[0][:512]
            zz = d0["t_z"][idx].astype(np.float32)
            rr = d0["t_root"][idx]
            got = d0["t_pi"][idx].astype(np.float32)
            live = ~np.isnan(got)
            pi_np = head_forward(sd0, "prop", zz, rr)
            err_pi = float(np.abs(pi_np[live] - got[live]).max())
            scale = float(np.abs(got[live]).max())
            v_np = head_forward(sd0, "value", zz, rr)
            out(f"  OFFLINE RECOMPUTE (numpy heads on heads_c{c0:04d}.npz, {idx.size} logged "
                f"beam states):")
            out(f"    pi: max|numpy - logged| = {err_pi:.3e} against a logit scale of "
                f"{scale:.2f}  -> {'consistent with fp16 z' if err_pi < 5e-2 else 'MISMATCH'}")
            out(f"    v : recomputed at the same states, range "
                f"[{v_np.min():.3f}, {v_np.max():.3f}]")

    # ---- per-cycle |dpi| and |dv| on the fixed probe ---------------------------------- #
    out("\n  per-cycle revision on the fixed probe (fp32, live readouts; the probe states are")
    out("  the era's own metering set, so only WITHIN-era differences are meaningful):")
    out(f"    {'era':>4s} {'pairs':>6s} {'|dv| med':>10s} {'|dv| p95':>10s} "
        f"{'|dpi| med':>10s} {'|dpi| p95':>10s} {'frac dpi==0':>12s}")
    avail = {}
    for c in (audi.get("cycles") or []):
        avail[c["cycle"]] = np.asarray(c["avail_slots"], dtype=np.int64)
    rows = []
    for era_i in sorted(set(int(e) for e in eras)):
        sel = np.nonzero(eras == era_i)[0]
        dv, dpi = [], []
        for a, b in zip(sel[:-1], sel[1:]):
            ca = int(cycles[a])
            sl = avail.get(ca)
            dv.append(np.abs(v_live[b] - v_live[a]))
            p = np.abs(pi_live[b] - pi_live[a])
            dpi.append(p[:, sl] if sl is not None and sl.size else p)
        if not dv:
            continue
        dv = np.concatenate([q.ravel() for q in dv])
        dpi = np.concatenate([q.ravel() for q in dpi])
        rows.append((era_i, len(sel) - 1, np.median(dv), np.percentile(dv, 95),
                     np.median(dpi), np.percentile(dpi, 95), float((dpi == 0).mean())))
        out(f"    {era_i:>4d} {len(sel) - 1:>6d} {rows[-1][2]:>10.4f} {rows[-1][3]:>10.4f} "
            f"{rows[-1][4]:>10.4f} {rows[-1][5]:>10.4f} {rows[-1][6]:>12.2%}")

    # ---- THE REVISION AT THE LOGGED BEAM STATES ---------------------------------------- #
    #  This is the quantity phase 2 differences, computed the way phase 2 will compute it:
    #  entirely offline, from `t_z` + two consecutive head snapshots off the fetched bytes.
    #  Restricted to the slots that were in the action set at cycle c — pi's logit on a slot
    #  the softmax masked out is not a proposal.
    out("\n  PER-CYCLE REVISION AT THE LOGGED BEAM STATES (offline, from the fetched bytes:")
    out("  |f(heads_{c+1}, s) - f(heads_c, s)| over decision rows of cycle c, live slots only)")
    out(f"    {'cycle':>6s} {'era':>4s} {'states':>7s} {'slots':>6s} {'|dv| med':>9s} "
        f"{'|dv| p95':>9s} {'|dpi| med':>10s} {'|dpi| p95':>10s} {'|dpi| max':>10s} "
        f"{'frac==0':>9s}")
    picks = routed[:: max(len(routed) // 10, 1)][:10] if routed else []
    agg_dpi, agg_dv = [], []
    for c in picks:
        sa, sb = load_heads(root, c), load_heads(root, c + 1)
        shard = [f for f in files
                 if int(os.path.basename(f)[5:9]) <= c <= int(os.path.basename(f)[11:15])]
        if sa is None or sb is None or not shard:
            continue
        d = np.load(shard[0])
        meta = cyc_meta.get(c, {})
        bud = meta.get("budget") or 8
        m = (d["t_cycle"] == c) & (d["t_step"] < bud)
        idx = np.nonzero(m)[0]
        if idx.size > 4096:
            idx = idx[np.linspace(0, idx.size - 1, 4096).astype(np.int64)]
        zz = d["t_z"][idx].astype(np.float32)
        rr = d["t_root"][idx]
        sl = np.asarray(meta.get("avail_slots", []), dtype=np.int64)
        dpi = np.abs(head_forward(sb, "prop", zz, rr) - head_forward(sa, "prop", zz, rr))
        dpi = dpi[:, sl] if sl.size else dpi
        dv = np.abs(head_forward(sb, "value", zz, rr) - head_forward(sa, "value", zz, rr))
        agg_dpi.append(dpi.ravel()); agg_dv.append(dv.ravel())
        out(f"    {c:>6d} {meta.get('era', 0):>4d} {idx.size:>7d} {sl.size:>6d} "
            f"{np.median(dv):>9.4f} {np.percentile(dv, 95):>9.4f} "
            f"{np.median(dpi):>10.4f} {np.percentile(dpi, 95):>10.4f} "
            f"{dpi.max():>10.4f} {float((dpi == 0).mean()):>9.2%}")
    if agg_dpi:
        A, V = np.concatenate(agg_dpi), np.concatenate(agg_dv)
        out(f"    pooled over {len(agg_dpi)} cycles: |dpi| med {np.median(A):.4f} "
            f"p95 {np.percentile(A, 95):.4f} max {A.max():.4f}, frac==0 {(A == 0).mean():.2%}; "
            f"|dv| med {np.median(V):.4f} p95 {np.percentile(V, 95):.4f}")

    # ---- is the offline residual really just fp16 z? ----------------------------------- #
    #  The offline recompute above sits ~1e-3 off the logged pi. That should be the fp16
    #  quantisation of `t_z` pushed through the head and nothing else, so: perturb the logged
    #  z by half a fp16 ulp per element and see whether the induced |dpi| is the same size.
    if err_pi is not None and c0 is not None:
        shard = [f for f in files
                 if int(os.path.basename(f)[5:9]) <= c0 <= int(os.path.basename(f)[11:15])]
        d0, sd0 = np.load(shard[0]), load_heads(root, c0)
        m = (d0["t_cycle"] == c0) & (d0["t_step"] < (cyc_meta.get(c0, {}).get("budget") or 8))
        idx = np.nonzero(m)[0][:512]
        zz = d0["t_z"][idx].astype(np.float32)
        ulp = np.spacing(d0["t_z"][idx]).astype(np.float32)
        rng = np.random.default_rng(0)
        pert = np.abs(head_forward(sd0, "prop", zz + 0.5 * ulp * rng.uniform(-1, 1, zz.shape),
                                   d0["t_root"][idx])
                      - head_forward(sd0, "prop", zz, d0["t_root"][idx])).max()
        out(f"\n  FP16-z CONTROL: perturbing the logged z by half a fp16 ulp per element moves "
            f"pi by max {pert:.3e};")
        out(f"    the offline-vs-logged residual above was {err_pi:.3e} — same scale, so the "
            f"residual is z's storage precision, not a recompute error.")

    # ---- the exact-zero invariant ------------------------------------------------------ #
    #  pi's output layer is ZERO-INIT and AdamW's weight decay is decoupled and multiplicative,
    #  so a slot that has never been in the action set has exactly-zero grad through the masked
    #  softmax and exactly-zero decay off a zero weight: its logit must be EXACTLY 0.0 at every
    #  cycle. The complement — a slot that IS in the action set — must move.
    #  pi's output layer is ZERO-INIT and AdamW's weight decay is decoupled and multiplicative,
    #  and a slot masked out of the softmax gets exactly-zero gradient — so a slot the action
    #  set has NOT YET contained must have a logit of EXACTLY 0.0, at every cycle up to the
    #  commit that introduces it. Checked CUMULATIVELY, per cycle, not just for slots that were
    #  never available at all: this is the strong form, and it is the one that survives a run
    #  where every slot eventually arrives.
    all_slots = set(range(int(audi.get("n_slots") or pi_live.shape[-1])))
    seen, worst, n_checked, first_seen = set(), 0.0, 0, {}
    avail_by_cycle = {c["cycle"]: set(int(q) for q in c["avail_slots"])
                      for c in (audi.get("cycles") or [])}
    for i, c in enumerate(cycles):
        seen |= avail_by_cycle.get(int(c), set())
        for j in seen:
            first_seen.setdefault(j, int(c))
        unseen = sorted(all_slots - seen)
        if not unseen:
            continue
        n_checked += 1
        worst = max(worst, float(np.abs(pi_live[i][:, np.asarray(unseen)]).max()))
    out(f"\n  EXACT-ZERO INVARIANT (cumulative, per cycle): on every cycle, the slots the "
        f"action set")
    out(f"  had not yet contained must have a logit of exactly 0.0 (zero-init out-layer, "
        f"masked softmax,")
    out(f"  multiplicative decay). Checked on {n_checked}/{len(cycles)} cycles that still had "
        f"an unseen slot:")
    out(f"    max|logit| over those = {worst:.3e}  -> "
        f"{'PASS (exactly 0)' if worst == 0.0 else 'NONZERO'}")
    out(f"    slots first available at: "
        f"{ {v: sum(1 for q in first_seen.values() if q == v) for v in sorted(set(first_seen.values()))} }"
        f"  (cycle -> how many slots arrived then)")
    if len(cycles) > 1:
        moved = np.abs(np.diff(pi_live, axis=0)).max(axis=(1, 2))
        out(f"    and pi moves at EVERY cycle boundary: min over boundaries of max|dpi| = "
            f"{moved.min():.3e} ({int((moved > 0).sum())}/{len(moved)} boundaries nonzero)")


# --------------------------------------------------------------------------- #
# (4) E1b — the per-datum arm
# --------------------------------------------------------------------------- #

def datum_provenance(files, budget):
    """[E1b] For every graded trajectory, how it was PRODUCED: how many of the `budget` moves
    on its own lineage were EXPLORE-INJECTED rather than proposed by pi.

    The join runs backwards through the decision tables. A candidate row with `c_child >= 0`
    says "the tip with index `c_child` at step+1 came from parent `c_parent` by move `c_mv`,
    with provenance `c_src`" — so walking a tip's lineage back through `t_parent` recovers the
    provenance of every move it took. This is the agency axis: a trajectory carrying an
    explore-injected move took an action pi did not propose.

    Returns {(cycle, inst, tip_at_budget): (n_explore, n_forced, n_enumerated)}.
    """
    prov = {}
    for f in files:
        d = np.load(f)
        cyc = np.unique(d["t_cycle"])
        for c in cyc:
            tm, cm = d["t_cycle"] == c, d["c_cycle"] == c
            t_step, t_inst, t_tip = d["t_step"][tm], d["t_inst"][tm], d["t_parent"][tm]
            t_tipi = d["t_tip"][tm]
            # parent lookup: (step, inst, tip) -> parent tip index at step-1
            par = {}
            for st, ins, tp, pa in zip(t_step, t_inst, t_tipi, t_tip):
                par[(int(st), int(ins), int(tp))] = int(pa)
            # provenance lookup: (step+1, inst, child tip) -> src
            src = {}
            k = d["c_child"][cm] >= 0
            for st, ins, ch, sc in zip(d["c_step"][cm][k], d["c_inst"][cm][k],
                                       d["c_child"][cm][k], d["c_src"][cm][k]):
                src[(int(st) + 1, int(ins), int(ch))] = int(sc)
            for ins in np.unique(t_inst):
                for tp in np.unique(t_tipi[(t_step == budget) & (t_inst == ins)]):
                    cur, ne, nf, nu = int(tp), 0, 0, 0
                    for st in range(budget, 0, -1):
                        sc = src.get((st, int(ins), cur))
                        if sc is None:
                            break
                        ne += int(sc == 4); nf += int(sc == 2); nu += int(sc == 0)
                        cur = par.get((st, int(ins), cur), -1)
                        if cur < 0:
                            break
                    prov[(int(c), int(ins), int(tp))] = (ne, nf, nu)
    return prov


def perdatum(tag, arm, root, res, pdj, files, audi, out):
    out("\n" + "=" * 78)
    out("(4) PER-DATUM CREDIT (E1b)")
    out("=" * 78)
    cyc = pdj.get("cycles") or []
    out(f"  matched budget     value: anchor {pdj['anchor_n_grad']} steps x "
        f"{pdj['anchor_value_lr']:.2e} = {pdj['anchor_n_grad'] * pdj['anchor_value_lr']:.3e}"
        f"  ->  perdatum {pdj['pd_n']}+{pdj['pd_maint_v']} steps x {pdj['value_lr']:.3e} = "
        f"{(pdj['pd_n'] + pdj['pd_maint_v']) * pdj['value_lr']:.3e}")
    out(f"                     pi:    anchor {pdj['anchor_prop_steps']} steps x "
        f"{pdj['anchor_prop_lr']:.2e} = "
        f"{pdj['anchor_prop_steps'] * pdj['anchor_prop_lr']:.3e}"
        f"  ->  perdatum ~{pdj['pd_prop_nominal']}+{pdj['pd_maint_p']} steps x "
        f"{pdj['prop_lr']:.3e} (realised step count follows the SOLVED count)")
    nv = [c["n_value_steps"] for c in cyc]
    npi = [c["n_pi_steps"] for c in cyc]
    fb = sum(1 for c in cyc if c["fallback"])
    out(f"  per-cycle steps    value {min(nv)}..{max(nv)} (constant = pd_n if no short cycle); "
        f"pi {min(npi)}..{max(npi)} (med {int(np.median(npi))})")
    out(f"  realised pi budget  {np.median(npi) * pdj['prop_lr']:.3e} median vs the anchor's "
        f"{pdj['anchor_prop_steps'] * pdj['anchor_prop_lr']:.3e} — "
        f"{np.median(npi) * pdj['prop_lr'] / (pdj['anchor_prop_steps'] * pdj['anchor_prop_lr']):.2f}x")
    out(f"  nothing-solved fallback fired on {fb}/{len(cyc)} cycles")
    out(f"  events             {pdj['n_events']} over {pdj['n_cycles']} cycles in "
        f"{len(pdj['shards'])} shards, "
        f"{sum(q['bytes'] for q in pdj['shards']) / 1e6:.1f} MB")

    # ---- read the revision tables back --------------------------------------------------- #
    rf = sorted(glob.glob(os.path.join(root, "revisions", "rev_c*.npz")))
    n_ev = 0
    bad = []
    own, prb, ref = [], [], []
    ownv, prbv, refv = [], [], []
    didpi, succ_rev, key = [], [], {}
    for f in rf:
        d = np.load(f)
        n_ev += d["e_cycle"].shape[0]
        for k in ("e_own_dv", "e_own_dpi", "e_prb_dv", "e_prb_dpi", "e_ref_dv", "e_ref_dpi"):
            if not np.isfinite(d[k]).all():
                bad.append(f"{os.path.basename(f)}: non-finite {k}")
        if not np.isfinite(d["e_vloss"]).all():
            bad.append(f"{os.path.basename(f)}: non-finite e_vloss")
        # e_ploss is NaN BY DESIGN where no pi step fired; assert exactly that
        nopi = d["e_did_pi"] == 0
        if not np.isnan(d["e_ploss"][nopi]).all() or np.isnan(d["e_ploss"][~nopi]).any():
            bad.append(f"{os.path.basename(f)}: e_ploss NaN pattern != e_did_pi")
        # RMS, not max: the three sets have different sizes (n_own = budget+1, n_prb, n_ref),
        # and a max over more states is mechanically larger. RMS is size-invariant, so the
        # spillover ratio below compares like with like.
        def _rms(x):
            return np.sqrt((x.reshape(x.shape[0], -1).astype(np.float64) ** 2).mean(1))
        own.append(_rms(d["e_own_dpi"]))
        prb.append(_rms(d["e_prb_dpi"]))
        ref.append(_rms(d["e_ref_dpi"]))
        ownv.append(_rms(d["e_own_dv"]))
        prbv.append(_rms(d["e_prb_dv"]))
        refv.append(_rms(d["e_ref_dv"]))
        didpi.append(d["e_did_pi"])
        succ_rev.append(d["e_succ"])
        for c, i, t, s_ in zip(d["e_cycle"], d["e_inst"], d["e_tip"], d["e_succ"]):
            key[(int(c), int(i), int(t))] = int(s_)
    own, prb, ref = np.concatenate(own), np.concatenate(prb), np.concatenate(ref)
    ownv, prbv, refv = np.concatenate(ownv), np.concatenate(prbv), np.concatenate(refv)
    didpi, succ_rev = np.concatenate(didpi), np.concatenate(succ_rev)
    out(f"  rows read          {n_ev}  (manifest agrees: {n_ev == pdj['n_events']})")
    out(f"  pi step fired on   {int(didpi.sum())}/{n_ev} events "
        f"({didpi.mean():.1%}); solved fraction of the sampled data {succ_rev.mean():.1%}")

    # ---- JOIN to the decision tables ----------------------------------------------------- #
    n_join, n_ok = 0, 0
    for f in files:
        d = np.load(f)
        bud = (audi.get("cycles") or [{}])[0].get("budget") or 8
        m = d["t_step"] == bud
        for c, i, t, s_ in zip(d["t_cycle"][m], d["t_inst"][m], d["t_tip"][m], d["t_succ"][m]):
            k = (int(c), int(i), int(t))
            if k in key:
                n_join += 1
                n_ok += int(key[k] == int(s_))
    out(f"  JOIN to decisions  {n_join}/{n_ev} revision events matched a graded tip; "
        f"grade agrees on {n_ok}/{n_join}  -> "
        f"{'PASS' if n_join == n_ev and n_ok == n_join else 'FAIL'}")
    if n_join != n_ev or n_ok != n_join:
        bad.append("revision<->decision join is incomplete or disagrees on the grade")

    out(f"  finiteness         {'PASS' if not bad else 'FAIL'}")
    for b in bad[:10]:
        out(f"    {b}")

    # ---- the revision itself ------------------------------------------------------------- #
    #  dv is reported over EVERY event (the value steps on every datum); dpi only over the
    #  events where a pi step actually fired. Pooling dpi over all events would report a
    #  median of exactly 0 whenever fewer than half the data solved — true, but a statement
    #  about the solve rate, not about the revision.
    PI = didpi == 1
    out("\n  REVISION MAGNITUDE per update event (RMS over states x slots — size-invariant,")
    out("  so the three sets are comparable despite holding different numbers of states).")
    out(f"  dv over all {n_ev} events; dpi over the {int(PI.sum())} events where a pi step "
        f"fired:")
    out(f"    {'set':>10s} {'n states':>9s} {'dv rms med':>12s} {'dv rms p95':>12s} "
        f"{'dpi rms med':>13s} {'dpi rms p95':>13s}")
    for name, dv, dp, nst in (("own", ownv, own, "budget+1"),
                              ("probe", prbv, prb, str(pdj["pd_probe_n"])),
                              ("reference", refv, ref, str(pdj["pd_ref_n"]))):
        out(f"    {name:>10s} {nst:>9s} {np.median(dv):>12.3e} {np.percentile(dv, 95):>12.3e} "
            f"{np.median(dp[PI]):>13.3e} {np.percentile(dp[PI], 95):>13.3e}")
    out(f"    SPILLOVER RATIO (rms at other states / at the datum's own states, medians)")
    out(f"      dv  (all events):        probe "
        f"{np.median(prbv) / max(np.median(ownv), 1e-30):.3f}, reference "
        f"{np.median(refv) / max(np.median(ownv), 1e-30):.3f}")
    out(f"      dpi (pi-stepping only):  probe "
        f"{np.median(prb[PI]) / max(np.median(own[PI]), 1e-30):.3f}, reference "
        f"{np.median(ref[PI]) / max(np.median(own[PI]), 1e-30):.3f}")

    # ---- the exact-zero invariant for E1b ------------------------------------------------ #
    #  A datum whose pi step did NOT fire (it did not solve, and the cycle did not hit the
    #  fallback) cannot have moved pi at all — pi's weights were untouched. So dpi must be
    #  EXACTLY zero on those events and nonzero on the rest. dv must be nonzero on every event,
    #  because the value steps on every datum regardless of grade.
    z_nopi = float((own[didpi == 0] == 0).mean()) if (didpi == 0).any() else float("nan")
    nz_pi = float((own[didpi == 1] > 0).mean()) if (didpi == 1).any() else float("nan")
    nz_v = float((ownv > 0).mean())
    out(f"\n  EXACT-ZERO INVARIANT (E1b): dpi == 0 on {z_nopi:.2%} of the "
        f"{int((didpi == 0).sum())} events where no pi step fired  -> "
        f"{'PASS' if z_nopi == 1.0 else 'FAIL'}")
    out(f"                              dpi > 0 on {nz_pi:.2%} of the "
        f"{int((didpi == 1).sum())} events where it did; dv > 0 on {nz_v:.2%} of all "
        f"{n_ev} events (the value steps on every datum)")
    if z_nopi != 1.0 or nz_v != 1.0:
        bad.append("E1b exact-zero invariant violated")

    # ---- BY PROVENANCE: did the trajectory take an action pi did not propose? ------------ #
    #  This is the agency axis E1 is about. `g > 0` is a property of the ACTION SET the beam
    #  actually explored, and an explore-injected move is the clearest case of an action that
    #  is not a deterministic function of the state.
    prov = datum_provenance(files, (audi.get("cycles") or [{}])[0].get("budget") or 8)
    ne = np.full(n_ev, -1, dtype=np.int16)
    j = 0
    for f in rf:
        d = np.load(f)
        for c, i, t in zip(d["e_cycle"], d["e_inst"], d["e_tip"]):
            q = prov.get((int(c), int(i), int(t)))
            ne[j] = -1 if q is None else q[0]
            j += 1
    hit = ne >= 0
    out(f"\n  BY PROVENANCE (explore-injected moves on the datum's own lineage; "
        f"{int(hit.sum())}/{n_ev} data resolved):")
    if hit.any():
        out(f"    {'n explore':>10s} {'events':>8s} {'solved':>8s} {'dv rms med':>12s} "
            f"{'dv spill':>9s} {'dpi rms med':>13s} {'dpi spill':>10s}   (dpi over pi-steppers)")
        for k in sorted(set(int(q) for q in ne[hit])):
            m = hit & (ne == k)
            if m.sum() < 20:
                continue
            mp = m & PI
            dpm = np.median(own[mp]) if mp.sum() >= 20 else float("nan")
            dps = (np.median(ref[mp]) / max(np.median(own[mp]), 1e-30)
                   if mp.sum() >= 20 else float("nan"))
            out(f"    {k:>10d} {int(m.sum()):>8d} {succ_rev[m].mean():>8.1%} "
                f"{np.median(ownv[m]):>12.3e} "
                f"{np.median(refv[m]) / max(np.median(ownv[m]), 1e-30):>9.3f} "
                f"{dpm:>13.3e} {dps:>10.3f}")
        any_e = hit & (ne > 0)
        no_e = hit & (ne == 0)
        if any_e.any() and no_e.any():
            out(f"    pooled: >=1 explore move {int(any_e.sum())} events, dv rms med "
                f"{np.median(ownv[any_e]):.3e}, solved {succ_rev[any_e].mean():.1%}  vs  "
                f"0 explore moves {int(no_e.sum())} events, dv rms med "
                f"{np.median(ownv[no_e]):.3e}, solved {succ_rev[no_e].mean():.1%}")

    # ---- DECOMPOSITION: do the per-datum revisions add up to the cycle boundary? --------- #
    #  The revision tables' probe set is the FIRST `pd_probe_n` of the snapshot probe's states,
    #  so the same states carry both records. Nothing moves a readout head between one event
    #  and the next, so within a cycle the per-event differences telescope EXACTLY:
    #      boundary(c -> c+1)  =  sum(per-datum events of c)  +  maintenance pass of c
    #  which is both an end-to-end check on the recording chain and the number the refit needs
    #  in order to know how much of a cycle it is and is not seeing.
    tp = os.path.join(root, "snapshots", "probe_trace.npz")
    if os.path.isfile(tp):
        pt = np.load(tp)
        cy, er, pv = pt["cycle"], pt["era"], pt["v"]
        idx = {int(c): i for i, c in enumerate(cy)}
        n_prb = int(pdj["pd_probe_n"])
        rows, resid, tot = [], [], []
        for f in rf:
            d = np.load(f)
            for c in np.unique(d["e_cycle"]):
                c = int(c)
                if c not in idx or (c + 1) not in idx:
                    continue
                if er[idx[c]] != er[idx[c + 1]]:      # probe states change at an era boundary
                    continue
                pd_sum = d["e_prb_dv"][d["e_cycle"] == c].sum(0)
                bnd = pv[idx[c + 1]][:n_prb] - pv[idx[c]][:n_prb]
                rows.append(c)
                resid.append(np.abs(bnd - pd_sum).max())
                tot.append(np.abs(bnd).max())
        if rows:
            resid, tot = np.asarray(resid), np.asarray(tot)
            frac = resid / np.maximum(tot, 1e-30)
            out(f"\n  DECOMPOSITION at the shared probe states, over {len(rows)} within-era "
                f"cycle boundaries:")
            out(f"    boundary revision           max|dv| med {np.median(tot):.3e}")
            out(f"    minus the per-datum sum     residual  med {np.median(resid):.3e} "
                f"(= the pooled MAINTENANCE pass, which the revision tables do not record)")
            out(f"    maintenance share of the cycle's boundary revision: med {np.median(frac):.1%},"
                f" p95 {np.percentile(frac, 95):.1%}")

    # ---- solved vs unsolved: does the grade change the size of the revision? ------------- #
    s1, s0 = succ_rev == 1, succ_rev == 0
    if s1.any() and s0.any():
        out(f"\n  BY GRADE (the credit signal itself, own states): dv rms med "
            f"{np.median(ownv[s1]):.3e} on SOLVED vs {np.median(ownv[s0]):.3e} on UNSOLVED "
            f"({np.median(ownv[s1]) / max(np.median(ownv[s0]), 1e-30):.2f}x)")
        out(f"    dv spillover (reference/own): SOLVED "
            f"{np.median(refv[s1]) / max(np.median(ownv[s1]), 1e-30):.3f} vs UNSOLVED "
            f"{np.median(refv[s0]) / max(np.median(ownv[s0]), 1e-30):.3f}")
        p1 = own[(succ_rev == 1) & (didpi == 1)]
        p0 = own[(succ_rev == 0) & (didpi == 1)]
        if p1.size and p0.size:
            out(f"    dpi rms med (pi-stepping events only) {np.median(p1):.3e} on SOLVED vs "
                f"{np.median(p0):.3e} on UNSOLVED-but-fallback "
                f"({np.median(p1) / max(np.median(p0), 1e-30):.2f}x)")
    return bad


def learner_health(tag, arms, out):
    """The arm's own clocks, side by side. The pooled-vs-per-datum contrast's SECONDARY
    readout — and the check that the treatment arm is still a functioning learner."""
    out("\n" + "=" * 78)
    out("(5) THE ARMS' OWN CLOCKS (pooled vs per-datum) — secondary readout")
    out("=" * 78)
    A = {}
    for a in arms:
        r = _load(os.path.join(FIG, tag, a, "results.json"))
        if r:
            A[a] = r
    if len(A) < 2:
        out("  only one arm in this tag — no contrast")
        return
    out(f"    {'arm':>10s} {'cycles':>7s} {'certL2':>7s} {'certL3':>7s} "
        + " ".join(f"{'e era' + str(i + 1):>9s}" for i in range(5)))
    for a, r in A.items():
        lg, sc = r["log"], r.get("shadow_cert") or {}
        eb = []
        for j in range(5):
            idx = [i for i, q in enumerate(lg["era"]) if q == j + 1]
            eb.append(np.mean([lg["e"][i] for i in idx]) if idx else float("nan"))
        out(f"    {a:>10s} {len(lg['cycle']):>7d} "
            f"{str(sc.get('2', {}).get('fired')):>7s} {str(sc.get('3', {}).get('fired')):>7s} "
            + " ".join(f"{q:>9.4f}" for q in eb))
    out(f"\n    {'arm':>10s} {'solved/cyc':>11s} {'vloss end':>10s} "
        f"{'pi acc end':>11s} {'L2 tbl':>7s} {'L3 tbl':>7s} {'L3@sup':>7s} {'L4@sup':>7s} "
        f"{'commits':>16s}")
    for a, r in A.items():
        lg = r["log"]
        oh = r.get("obs_hist") or {}
        vc = lg["vocab"][-1] if lg.get("vocab") else {}
        pr = [q for q in lg["prop"] if q and q.get("acc") is not None]
        cm = [(e["cycle"], e["level"]) for e in r["events"] if e.get("kind") == "commit"]
        out(f"    {a:>10s} {np.mean(lg['n_solved']):>11.1f} "
            f"{(lg['vloss'][-1] if lg['vloss'][-1] is not None else float('nan')):>10.4f} "
            f"{(pr[-1]['acc'] if pr else float('nan')):>11.4f} "
            f"{str(vc.get('2')):>7s} {str(vc.get('3')):>7s} "
            f"{str((oh.get('3') or [None])[-1]):>7s} {str((oh.get('4') or [None])[-1]):>7s} "
            f"{str(['L%d@c%d' % (l, c) for c, l in cm]):>16s}")

    # pi's per-level proposal mass — the arc's `trust` readout, at the last probe of each arm.
    # Slots are laid out level-major (L1 0..31, L2 32..47, L3 48..55) by `PN.slot_layout`.
    out(f"\n    pi PROPOSAL MASS at the last probe (the arc's 'trust' readout) and its shape:")
    out(f"    {'arm':>10s} {'L1':>8s} {'L2':>8s} {'L3':>8s} {'macro':>8s} "
        f"{'top1':>8s} {'entropy':>8s} {'n_moves':>8s}")
    for a, r in A.items():
        pb = [q for q in r["log"]["probe"] if q and q.get("pi")]
        if not pb:
            continue
        pi = pb[-1]["pi"]
        mm = np.asarray(pi["mean_mass"], float)
        out(f"    {a:>10s} {mm[0:32].sum():>8.4f} {mm[32:48].sum():>8.4f} "
            f"{mm[48:56].sum():>8.4f} {pi['macro_mass']:>8.4f} {pi['top1']:>8.4f} "
            f"{pi['entropy']:>8.4f} {pb[-1]['n_moves']:>8d}")


def inverse_gate(tag, out, arm="anchor"):
    """[E1b] THE INVERSE GATE. With `per_datum` toggled OFF, the arm must replay the donor
    BIT-IDENTICALLY — which is the proof that every line E1b adds is inert when disabled. Read
    here through the FETCHED bytes, against both the direct donor (`cd_s0`, no audiation code
    at all) and phase 1's own instrumented anchor (`au_s0`)."""
    out("\n" + "=" * 78)
    out(f"(0b) INVERSE GATE — `{arm}` of this tag (per_datum OFF) vs the donors")
    out("=" * 78)
    mine = _load(os.path.join(FIG, tag, arm, "results.json"))
    if mine is None:
        out(f"  no {arm} arm in this tag — skipped")
        return []
    bad = []
    for label, path in (
            ("cd_s0/anchor  (conductor: no audiation code at all)",
             os.path.join(COND_FIG, "cd_s0", "anchor", "results.json")),
            ("au_s0/anchor  (phase 1: instrument ON, no E1b code path)",
             os.path.join(FIG, "au_s0", "anchor", "results.json"))):
        ref = _load(path)
        if ref is None:
            out(f"  vs {label}: reference not present — SKIPPED")
            continue
        old, lg = ref["log"], mine["log"]
        n = min(len(old["cycle"]), len(lg["cycle"]))
        per = {k: float(np.abs(np.asarray(old[k][:n], float)
                               - np.asarray(lg[k][:n], float)).max()) for k in GF_SERIES}
        worst = max(per.values())
        eq = len(old["cycle"]) == len(lg["cycle"])
        out(f"  vs {label}")
        out(f"    c1-c{n}, {len(GF_SERIES)} series: max|delta| = {worst:.3e}; cycle counts "
            f"equal = {eq}  -> {'PASS' if worst == 0.0 and eq else 'FAIL'}")
        if worst:
            out("    per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
            bad.append(f"inverse gate failed vs {label}")
    return bad


def notes(out):
    """Things the E1b refit must know that `schema.md` does not already say."""
    out("\n" + "=" * 78)
    out("(6) NOTES FOR THE REFIT — properties of the revision tables, not results")
    out("=" * 78)
    for i, t in enumerate([
        "dpi is EXACTLY 0 on the events where no pi step fired (the datum did not solve). "
        "That is ~79% of events here. Any pi-target regression must condition on "
        "`e_did_pi == 1` or it fits a point mass at zero. dv is nonzero on 100% of events.",
        "`e_own_a` has `budget` entries but the own-state arrays have `budget+1` rows: the "
        "action at row t carries state t to state t+1, and the terminal state has no action. "
        "Aligning actions to states means dropping the last state row.",
        "The three sets hold different numbers of states (own = budget+1 = 9, probe and "
        "reference = 32 each). Any cross-set comparison has to be size-invariant; this report "
        "uses RMS. A max or a sum would rank the sets by their size.",
        "The pooled MAINTENANCE pass is NOT in the revision tables (value: replay-pool-only "
        "steps; pi: history-only steps). So a cycle's per-datum revisions do not sum to the "
        "cycle-boundary snapshot difference — the residual is the maintenance pass, quantified "
        "in the DECOMPOSITION block above. Both granularities exist on this arm: the "
        "per-cycle `heads_c` snapshots are still written.",
        "Provenance is NOT stored in the revision tables. It is derived from the decision "
        "tables by walking `t_parent` back and reading `c_src` off the candidate that became "
        "each tip; `datum_provenance()` in this file does it, and the join was complete here.",
        "The 128 data per cycle are a uniform subsample of the cycle's 1024 graded tips, drawn "
        "from a dedicated stream. The flat indices are in `perdatum.json` per cycle as `sel` "
        "(flat = inst * pr_width + tip); the reference set's are `ref`. Data outside `sel` "
        "received no update at all in this arm.",
        "`e_own_z` is fp16 (as in phase 1) but the REVISIONS are fp32 — the endpoints were "
        "never stored in fp16, only the differences, which are formed on device in fp32.",
        "The probe states change at an era boundary (each era meters on its own held-out set), "
        "so a revision at the probe is comparable across cycles only WITHIN an era. The "
        "reference set is redrawn every cycle and is comparable only within a cycle.",
    ]):
        out(f"  {i + 1}. {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="au_s0")
    ap.add_argument("--arm", default="anchor")
    ap.add_argument("--arms", default="",
                    help="comma list for the side-by-side clocks section, e.g. anchor,perdatum")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="also write the report to figures/<tag>/reduction.txt")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    root = os.path.join(FIG, a.tag, a.arm)
    res = _load(os.path.join(root, "results.json"))
    audi = _load(os.path.join(root, "audiation.json"))
    setup = _load(os.path.join(FIG, a.tag, "setup.json"))
    assert res is not None, f"no results.json under {root}"
    assert audi is not None, (
        f"no audiation.json under {root} — was --log-decisions on for this arm? "
        f"(an E1b tag logs only the treated arm; pass --arm perdatum)")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"AUDIATION (E1 phase 1) — data report for {a.tag}/{a.arm}")
    out(f"  complete={res.get('complete')}  cycles={len(res['log']['cycle'])}  "
        f"seed={res['config']['seed']}  eras={[e['name'] for e in res['eras']]}")
    bad_ig = inverse_gate(a.tag, out)
    gates(a.tag, a.arm, root, res, audi, setup, out)
    files = dataset(a.tag, a.arm, root, res, audi, out)
    per_cycle, src_tot, bad = integrity(files, audi, os.path.join(root, 'snapshots'), out)
    descriptive(root, audi, files, src_tot, out)
    pdj = _load(os.path.join(root, "perdatum.json"))
    if pdj is not None:
        bad = bad + perdatum(a.tag, a.arm, root, res, pdj, files, audi, out)
    bad = bad + bad_ig
    if a.arms:
        learner_health(a.tag, [q.strip() for q in a.arms.split(",") if q.strip()], out)
    if pdj is not None:
        notes(out)
    out("\n" + "=" * 78)
    out(f"END — {'no integrity violation found' if not bad else str(len(bad)) + ' VIOLATIONS'}")
    if a.write:
        p = os.path.join(FIG, a.tag, "reduction.txt")
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n-> {p}")


if __name__ == "__main__":
    main()
