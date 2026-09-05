"""Reduce a `solo` run to a text report.

Pure python (json + math only): the local client has neither numpy nor matplotlib
(`offbook/FILES.md` Gotcha).

    python3 mjc/practice/solo/analyze_solo.py --tag s0 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# `python3 mjc/practice/solo/analyze_solo.py` puts THIS directory on sys.path, not `experiments/`,
# so the package import below would fail; `modal run` does put it there. Adding it explicitly makes
# the same file work both ways (`offbook/analyze_offbook.py`'s idiom).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from mjc.shared import app, volume, DATA_DIR   # noqa: E402

VOL = "mujoco-control-data"
# The config keys that must agree for a cross-tag control to be DEFINED (solo.py's S0_KEYS).
S0_KEYS = ("seed", "n_eval", "n_rt", "batch", "n_warm", "n_cand", "n_score", "n_slot", "m_cand",
           "m_member", "n_poison", "batch_pr", "k_prop", "eps_act", "explore_eps", "force_window",
           "prop_hidden", "prop_layers", "prop_lr", "prop_cap", "prop_steps", "prop_batch",
           "span_hidden", "span_layers", "span_lr", "span_steps", "span_batch", "span_cap",
           "span_hold_cap", "span_hold_frac", "parity_tau", "parity_every", "parity_min",
           "probe_every", "seg_H", "explore_sigma", "bc_cycles", "bc_batch", "bc_hidden",
           "bc_layers", "bc_lr", "bc_steps", "bc_batch_sz", "bc_explore", "dperf_ema", "dperf_sil")
ORDER = ("reflex", "key_seg", "audit_all", "fid", "audit_prop_kN", "audit_prop_k",
         "route_native", "prop_k_d0",
         "audit_prop_k_ref", "route_native_ref", "audit_prop_k_lib", "route_native_lib")


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL, f"/practice_solo/{tag}/solo.json",
                    os.path.join(dst, "solo.json"), "--force"], check=True)
    return os.path.join(dst, "solo.json")


def recover_sm(d):
    """Recompute gate S-M from the two SAVED records.

    `s1q`'s runner crashed after the arm loops (a numpy trajectory shadowed the reference-run dict,
    so S-M's `ref.get(...)` hit an ndarray) and never wrote `S_M`, `S_P`, `flags` or `done.txt`.
    Every arm, every cycle row and every battery had already been committed by the per-arm `save()`,
    and S-M is a pure comparison of numbers both records already contain — so it is recovered here
    rather than by re-running. The arithmetic is the runner's, verbatim; the row is marked
    `post_hoc` so the record never claims the runner computed it.
    """
    rt = d["config"].get("ref_tag")
    rp = os.path.join(HERE, "results", str(rt or ""), "solo.json")
    if not rt or not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    ok = all(r["config"].get(k) == d["config"].get(k) for k in S0_KEYS)
    bad = [k for k in S0_KEYS if r["config"].get(k) != d["config"].get(k)]
    sm = dict(ref_tag=rt, applicable=bool(ok), post_hoc=True, config_mismatches=bad)
    if ok and "audit_prop_k" in d["arms"] and "audit_prop_k" in r.get("arms", {}):
        a, b = d["arms"]["audit_prop_k"]["cycles"], r["arms"]["audit_prop_k"]["cycles"]
        n = min(len(a), len(b))
        de = max(abs(a[i]["e_piece"] - b[i]["e_piece"]) for i in range(n))
        dm, np_ = 0.0, 0
        for i in range(n):
            if a[i].get("probe") and b[i].get("probe"):
                np_ += 1
                for f in ("mass_seg", "mass_prim", "mass_poison", "top1", "entropy"):
                    dm = max(dm, abs(a[i]["probe"][f] - b[i]["probe"][f]))
        sm.update(n_cycles=int(n), e_piece_max_abs=de, probe_max_abs=dm, n_probes=np_,
                  **{"pass": bool(de == 0.0 and dm == 0.0)})
    return sm


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    if not d.get("complete"):
        P("*** INCOMPLETE RUN: the runner did not reach its final save. What is present is listed "
          "below; anything absent is named as absent, never inferred. ***")
    P(f"=== solo — re-internalization with no forward model, tag {cfg['tag']}, "
      f"seed {cfg['seed']} ({'COMPLETE' if d.get('complete') else 'INCOMPLETE'}, "
      f"{d.get('wall_s', 0):.0f}s wall) ===")
    P(f"Delta ladder {cfg['deltas']} control steps = "
      f"{[round(x * 0.024 * 1000) for x in cfg['deltas']]} ms | n_cycles={cfg['n_cycles']} "
      f"n_eval={cfg['n_eval']} batch_pr={cfg['batch_pr']} | n_slot={cfg['n_slot']} x "
      f"m_member={cfg['m_member']} (+{cfg['n_poison']} poison) | k_prop={cfg['k_prop']} "
      f"eps_act={cfg['eps_act']} explore_eps={cfg['explore_eps']} fw={cfg['force_window']} | "
      f"parity tau={cfg['parity_tau']}")
    P("NO FORWARD MODEL ANYWHERE. Segment span only; chain cells exist in the donor library "
      "(for gate S-F1) and are unused by every treatment arm.")

    # ------------------------------------------------------------------ gates
    P("\n--- gates ---")
    P(f"S-F0 donor constants: {d['S_F0']['checked']} checked, "
      f"{len(d['S_F0']['mismatches'])} mismatched, pass={d['S_F0']['pass']}")
    b = d["S_F1"]
    P(f"S-F1 cross-tag exact control vs acappella b1 (library build + the Delta=0 AND Delta=8 "
      f"rows): max|delta| = {b['max_abs']:.3e}  gains_match={b.get('gains_match')}  "
      f"applicable={b.get('applicable')}  pass={b['pass']}")
    if not b.get("applicable"):
        P("     [config differs from b1's; the control is NOT DEFINED here and is not 'passed']")
    for k in sorted(b):
        if k.endswith("_max_abs"):
            P(f"       {k[:-8]:22s} {b[k]:.3e}")
    t0 = d.get("S_T0", {})
    P(f"S-T0 numpy trunk forward == torch trunk forward: max|delta| = "
      f"{t0.get('max_abs', float('nan')):.3e}  pass={t0.get('pass')}")
    t1 = d.get("S_T1", {})
    P(f"S-T1 THE TRUNK GATE — the clone vs the reflex law it clones "
      f"(band |diff| <= max({t1.get('tol_rel')}*e_law, {t1.get('tol_abs')})), "
      f"binding={t1.get('binding')}, pass={t1.get('pass')}")
    P(f"     {'trunk':>7s} {'world':>6s} {'evalD':>6s} {'e_law':>9s} {'e_clone':>9s} "
      f"{'diff':>9s} {'band':>8s}  ok")
    for r in t1.get("rows", []):
        P(f"     D={r['trunk_delta']:<5d} {r['world']:>6s} {r['eval_delta']:6d} "
          f"{r['e_law']:9.4f} {r['e_clone']:9.4f} {r['diff']:+9.4f} {r['band']:8.4f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}")
    t2 = d.get("S_T2")
    if t2:
        P(f"S-T2 the re-cloned trunk vs `{t2.get('ref_tag')}`'s: "
          + (f"max|delta| = {t2['max_abs']:.3e}  pass={t2['pass']}" if t2.get("applicable")
             else f"NOT DEFINED (ref_found={t2.get('ref_found')}, "
                  f"mismatches={t2.get('config_mismatches', [])[:6]})"))
    sm = d.get("S_M") or recover_sm(d)
    if sm:
        if sm.get("post_hoc"):
            P("     [S-M recovered POST HOC from the two saved records — the runner crashed after "
              "the arm loops, before writing it; the arithmetic is the runner's, verbatim]")
        P(f"S-M  the `median` arm vs `{sm.get('ref_tag')}`, cycle for cycle: "
          + (f"e_piece max|delta| = {sm['e_piece_max_abs']:.3e} over {sm['n_cycles']} cycles, "
             f"pi-mass max|delta| = {sm['probe_max_abs']:.3e} over {sm['n_probes']} probes  "
             f"pass={sm['pass']}" if sm.get("applicable") else "NOT DEFINED"))
    for nm, v in d.get("S_P", {}).items():
        P(f"S-P  {nm} == audit_all over {v['n_cycles']} metering cycles: "
          f"max|delta| = {v['max_abs']:.3e}  pass={v['pass']}")

    # ------------------------------------------------------------------ the trunk + libraries
    P("\n--- the trunk (behaviour-cloned reflex; no f(s,u) in it) ---")
    for k, v in sorted(d.get("bc", {}).items()):
        P(f"  Delta={k}: gains kp={v['gains'][0]} kd={v['gains'][1]}  {v['n']} samples  "
          f"mse={v['mse_full']:.3e}")
    P("\n--- libraries ---")
    dl = d["donor_library"]
    P(f"  donor (acappella's build, verbatim): sizes {dl['sizes']}  "
      f"{dl['build_ledger']['n_ground']:.0f} groundings")
    ml = d["member_library"]
    P(f"  treatment (slots with MEMBERS): sizes {ml['sizes']}  "
      f"{ml['build_ledger']['n_ground']:.0f} groundings")
    for r in ml["build"]:
        P(f"    seam {r['seam']}: {r['n_cand']} candidates scored on {r['n_score']} held-out "
          f"states; best {r['chosen_score']:.4f} median {r['score_med']:.4f} "
          f"worst {r['score_max']:.4f}; kept the best {r.get('n_pick')} "
          f"(worst kept {r.get('kept_score_max', float('nan')):.4f}); "
          f"cluster sizes {r['cluster_sizes']}, {r.get('n_repaired', 0)} members repaired; "
          f"poison scores {[round(x, 4) for x in r['poison_scores']]}")

    # ------------------------------------------------------------------ seam information
    P("\n--- S-S: seam information under the DELAYED read (member library, per-state oracle) ---")
    P(f"  {'D':>3s} {'seam':>4s} {'best fixed':>11s} {'oracle':>9s} {'gain':>7s} "
      f"{'distinct':>9s} {'key gets':>9s}")
    for r in d.get("S_S", []):
        P(f"  {r['delta']:3d} {r['seam']:4d} {r['best_fixed_slot']:11.4f} "
          f"{r['per_state_oracle']:9.4f} {r['gain']:6.2f}x "
          f"{r['n_distinct_argmin']:4d}/{r['n_slot']:<4d} {r['key_achieved']:9.4f}")

    # ------------------------------------------------------------------ the arm table
    arms = d.get("arms", {})
    names = [n for n in ORDER if n in arms] + [n for n in arms if n not in ORDER]
    P("\n--- the arms, at the last metered cycle (performance tempo) ---")
    P(f"  {'arm':>14s} {'D':>2s} {'cyc':>4s} {'e_piece':>8s} {'fb':>6s} {'ground':>7s} "
      f"{'priced s':>9s} {'fb-only':>8s} {'prim':>5s} {'n_aud':>6s} {'n_prop':>6s}")
    P("  (the BATTERY below is one training round further on than these rows: it runs on the "
      "final trained state, after the last cycle's training.)")
    for nm in names:
        c = arms[nm]["cycles"][-1]
        fbonly = c["steps"] * 0.024 + c["n_fb"] * 0.1
        P(f"  {nm:>14s} {c['delta']:2d} {len(arms[nm]['cycles']):4d} {c['e_piece']:8.4f} "
          f"{c['n_fb']:6.1f} {c['n_ground']:7.1f} {c['t_piece']:9.1f} {fbonly:8.1f} "
          f"{c.get('frac_prim', float('nan')):5.2f} {c.get('n_aud', 0.0):6.1f} "
          f"{c.get('n_prop', 0.0):6.1f}")
    P("  (fb = feedback events per traversal; the reflex law's own reference is 136 fb, "
      "0 groundings)")

    # ------------------------------------------------------------------ trajectories over cycles
    P("\n--- e_piece and frac_prim over cycles (every arm that runs a cycle loop) ---")
    for nm in names:
        cy = arms[nm]["cycles"]
        if len(cy) < 2:
            continue
        step = max(1, len(cy) // 10)
        idx = list(range(0, len(cy), step))
        if idx[-1] != len(cy) - 1:
            idx.append(len(cy) - 1)
        P(f"  {nm}:")
        P("    c      " + " ".join(f"{cy[i]['cycle']:>7d}" for i in idx))
        P("    e      " + " ".join(f"{cy[i]['e_piece']:7.4f}" for i in idx))
        P("    prim   " + " ".join(f"{cy[i].get('frac_prim', float('nan')):7.2f}" for i in idx))
        P("    n_aud  " + " ".join(f"{cy[i].get('n_aud', 0.0):7.1f}" for i in idx))

    # ------------------------------------------------------------------ the pi filter
    band_arms = [n for n in names if arms[n]["cycles"][-1].get("pi_filter")]
    if band_arms:
        P("\n--- pi's imitation filter: what the body admitted as competent, per cycle ---")
        P("  (starvation is the failure mode: `pass` = fraction of the practice batch at or under "
          "the filter, `rows` = pi target rows added that cycle)")
        for nm in band_arms:
            cy = arms[nm]["cycles"]
            f0 = cy[-1]["pi_filter"]
            bd = cy[-1].get("pi_band")
            step = max(1, len(cy) // 10)
            idx = list(range(0, len(cy), step))
            if idx[-1] != len(cy) - 1:
                idx.append(len(cy) - 1)
            tot = sum(c.get("n_targets", 0) for c in cy)
            zero = sum(1 for c in cy if c.get("n_targets", 0) == 0)
            P(f"  {nm}  filter={f0}" + (f" band={bd:.4f}" if bd else "")
              + f"  | mean pass {sum(c.get('band_pass', 0) for c in cy) / len(cy):.3f}"
                f"  total target rows {tot}  cycles admitting NOTHING: {zero}/{len(cy)}")
            P("    c      " + " ".join(f"{cy[i]['cycle']:>7d}" for i in idx))
            P("    pass   " + " ".join(f"{cy[i].get('band_pass', float('nan')):7.2f}" for i in idx))
            P("    rows   " + " ".join(f"{cy[i].get('n_targets', 0):7d}" for i in idx))

    # ------------------------------------------------------------------ trust
    P("\n--- trust: pi's proposal mass, by level, over cycles (the probe ladder) ---")
    for nm in names:
        pr = [(c["cycle"], c["probe"]) for c in arms[nm]["cycles"] if c.get("probe")]
        if not pr:
            continue
        P(f"  {nm}:")
        P("    c        " + " ".join(f"{c:>7d}" for c, _ in pr))
        for f in ("mass_seg", "mass_prim", "mass_poison", "top1", "entropy"):
            P(f"    {f:9s}" + " ".join(f"{p[f]:7.3f}" for _, p in pr))
        P("    argmax_prim " + " ".join(f"{p['argmax_prim']:7.3f}" for _, p in pr))

    # ------------------------------------------------------------------ parity
    P("\n--- Port 2 parity, measured as execution reproduction ON THE PLANT ---")
    for nm in names:
        pr = arms[nm].get("parity") or []
        if not pr:
            continue
        last = pr[-1]
        P(f"  {nm}: {last['n_open']}/{last['n_checked']} slots open at tau={last['tau']} "
          f"(cycle {last['cycle']}); open-slot ids {arms[nm]['open_slots']}")
        P(f"    {'slot':>5s} {'seam':>4s} {'n_hold':>6s} {'frac':>6s} {'e_tape':>8s} "
          f"{'e_head':>8s} {'gap':>8s}  open")
        for r in sorted(last["rows"], key=lambda x: x["slot"]):
            P(f"    {r['slot']:5d} {r['seam']:4d} {r['n_hold']:6d} {r['frac']:6.2f} "
              f"{r['mean_e_tape']:8.4f} {r['mean_e_head']:8.4f} {r['mean_gap']:+8.4f}  "
              f"{r['open']}")
        P("    parity over cycles: " + " ".join(
            f"c{p['cycle']}:{p['n_open']}/{p['n_checked']}" for p in pr))

    # ------------------------------------------------------------------ the battery
    P("\n--- the address-book battery, on the final state of every arm with heads ---")
    P(f"  {'arm':>14s} {'row':>17s} {'e_piece':>8s} {'fb':>6s} {'ground':>7s} {'prim':>5s}")
    for nm in names:
        bt = arms[nm].get("battery") or {}
        for k in ("base", "no_prim", "no_table", "no_table_no_prim", "restored"):
            if k not in bt:
                continue
            v = bt[k]
            P(f"  {nm:>14s} {k:>17s} {v['e_piece']:8.4f} {v['n_fb']:6.1f} "
              f"{v['n_ground']:7.1f} {v['frac_prim']:5.2f}")
        if "restored_max_abs" in bt:
            P(f"  {'':>14s} {'restored==base':>17s} {bt['restored_max_abs']:.3e}")
    P("  ('fid' carries UNTRAINED heads and is the negative control for these rows.)")

    # ------------------------------------------------------------------ delta_perf
    P("\n--- delta_perf instruments (reference = the committed tape's OWN stored trajectory) ---")
    P("  consumed by nothing this round; logged so the certifying role can be read against "
      "model-free candidates on the same run.")
    for nm in names:
        dp = arms[nm].get("dperf") or []
        if not dp:
            continue
        per = {}
        for r in dp:
            q = per.setdefault(r["slot"], dict(n=0, nd=0, e=0.0, ewp=0.0, d=0.0, sil=0.0))
            q["n"] += 1; q["e"] += r["e_step"]; q["ewp"] += r["e_wp"]
            if r.get("dperf") is not None:            # a slot's FIRST launch has no benchmark yet
                q["nd"] += 1; q["d"] += r["dperf"]; q["sil"] += r["frac_silent"]
        P(f"  {nm}:  {'slot':>5s} {'cycles':>7s} {'e_step':>8s} {'e_wp':>8s} "
          f"{'dperf':>9s} {'silent':>7s}")
        for sid, q in sorted(per.items()):
            nd = max(q["nd"], 1)
            P(f"  {'':>{len(nm)}s}  {sid:5d} {q['n']:7d} {q['e'] / q['n']:8.4f} "
              f"{q['ewp'] / q['n']:8.4f} {q['d'] / nd:+9.4f} {q['sil'] / nd:7.2f}")

    P("\n--- flags on the record ---")
    for k, v in (d.get("flags") or {}).items():
        P(f"  {k}: {v}")
    if d.get("skipped_arms"):
        P(f"  skipped arms (Delta not in the ladder): {d['skipped_arms']}")


# --------------------------------------------------------------------------- figures (REMOTE)

@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_solo_figures(tag: str) -> list:
    """Drawn REMOTELY and committed to the volume: the analysis machine in this repo's sessions has
    neither numpy nor matplotlib (`offbook/FILES.md` Gotcha). `--figures` pulls them down.

    Axes are in the piece's own units — piece error is the mean over segments of the median
    boundary error at each waypoint, in arena units, on a piece whose segment leg is 0.8. No title
    on any panel states a reading; titles name the quantity only.
    """
    import numpy as np                                            # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_solo", tag)
    figdir = os.path.join(root, "figs")
    os.makedirs(figdir, exist_ok=True)
    d = json.load(open(os.path.join(root, "solo.json")))
    arms, dr = d["arms"], d["donor_rows"]
    made = []
    YL = "piece error (arena units; segment leg = 0.8)"
    C = {"audit_prop_k": "#1f77b4", "route_native": "#d62728", "audit_prop_kN": "#2ca02c",
         "fid": "#7f7f7f", "prop_k_d0": "#9467bd"}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name), dpi=150)
        plt.close(fig)
        made.append(name)

    def series(nm, key="e_piece"):
        cy = arms[nm]["cycles"]
        return [c["cycle"] for c in cy], [c[key] for c in cy]

    # ---- fig 1: piece error over cycles ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                           gridspec_kw={"width_ratios": [1.6, 1.0]})
    a = ax[0]
    for nm, ls in (("audit_prop_k", "-"), ("route_native", "-"),
                   ("audit_prop_kN", "--"), ("fid", ":")):
        if nm in arms and len(arms[nm]["cycles"]) > 1:
            x, y = series(nm)
            a.plot(x, y, ls, color=C[nm], lw=1.8, label=nm)
    refs = [("reflex through the trunk", arms["reflex"]["cycles"][-1]["e_piece"], "#333333", "-"),
            ("reflex law (PD)", dr["reflex@8"]["e_piece"], "#333333", "--"),
            ("key_seg", arms["key_seg"]["cycles"][-1]["e_piece"], "#ff7f0e", "-."),
            ("audit_all", arms["audit_all"]["cycles"][-1]["e_piece"], "#2ca02c", "-."),
            ("donor lib_seg (acappella b1)", dr["lib_seg@8"]["e_piece"], "#8c564b", ":")]
    for lab, v, col, ls in refs:
        a.axhline(v, color=col, ls=ls, lw=1.2, alpha=0.8)
        a.text(0.995, v, f" {lab} {v:.4f}", transform=a.get_yaxis_transform(),
               ha="right", va="bottom", fontsize=7.5, color=col)
    a.set_xlabel("practice cycle"); a.set_ylabel(YL)
    a.set_title("piece error over cycles, $\\Delta$ = 8 (192 ms)", fontsize=10)
    a.legend(fontsize=8, loc="lower left", ncol=2)
    a.grid(alpha=0.25)

    b = ax[1]
    if "prop_k_d0" in arms:
        x, y = series("prop_k_d0")
        b.plot(x, y, "-", color=C["prop_k_d0"], lw=1.8, label="prop_k_d0")
    for lab, v, col, ls in (("reflex law (PD)", dr["reflex@0"]["e_piece"], "#333333", "--"),
                            ("key_seg", dr["key_seg@0"]["e_piece"], "#ff7f0e", "-."),
                            ("donor lib_seg", dr["lib_seg@0"]["e_piece"], "#8c564b", ":")):
        b.axhline(v, color=col, ls=ls, lw=1.2, alpha=0.8)
        b.text(0.995, v, f" {lab} {v:.4f}", transform=b.get_yaxis_transform(),
               ha="right", va="bottom", fontsize=7.5, color=col)
    b.set_xlabel("practice cycle"); b.set_ylabel(YL)
    b.set_title("piece error over cycles, $\\Delta$ = 0", fontsize=10)
    b.legend(fontsize=8, loc="center right"); b.grid(alpha=0.25)
    save(fig, "fig1_piece_error_over_cycles.png")

    # ---- fig 2: pi's proposal mass by level --------------------------------------------
    def probes(nm):
        return [(c["cycle"], c["probe"]) for c in arms[nm]["cycles"] if c.get("probe")]

    fid_p = probes("fid")[0][1] if "fid" in arms and probes("fid") else None
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for nm in ("audit_prop_k", "route_native", "prop_k_d0"):
        if nm not in arms:
            continue
        pr = probes(nm)
        cyc = [c for c, _ in pr]
        ax[0].plot(cyc, [p["mass_seg"] for _, p in pr], "-", color=C[nm], lw=1.8,
                   label=f"{nm} · segment slots")
        ax[0].plot(cyc, [p["mass_prim"] for _, p in pr], "--", color=C[nm], lw=1.4,
                   label=f"{nm} · primitive")
        ax[1].semilogy(cyc, [max(p["mass_poison"], 1e-5) for _, p in pr], "-", color=C[nm],
                       lw=1.8, label=nm)
    if fid_p is not None:
        ax[0].axhline(fid_p["mass_seg"], color="#7f7f7f", ls=":", lw=1.3)
        ax[0].text(0.99, fid_p["mass_seg"], f" fid (untrained $\\pi$) segment {fid_p['mass_seg']:.3f}",
                   transform=ax[0].get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5,
                   color="#7f7f7f")
        ax[0].axhline(fid_p["mass_prim"], color="#7f7f7f", ls=":", lw=1.3)
        ax[0].text(0.99, fid_p["mass_prim"], f" fid primitive {fid_p['mass_prim']:.3f}",
                   transform=ax[0].get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5,
                   color="#7f7f7f")
        ax[1].axhline(fid_p["mass_poison"], color="#7f7f7f", ls=":", lw=1.3)
        ax[1].text(0.99, fid_p["mass_poison"], f" fid (untrained $\\pi$) {fid_p['mass_poison']:.3f}",
                   transform=ax[1].get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5,
                   color="#7f7f7f")
    ax[0].set_xlabel("practice cycle"); ax[0].set_ylabel("$\\pi$ proposal mass")
    ax[0].set_title("$\\pi$ proposal mass: segment slots and the primitive", fontsize=10)
    ax[0].legend(fontsize=7.5, loc="center right"); ax[0].grid(alpha=0.25)
    ax[1].set_xlabel("practice cycle")
    ax[1].set_ylabel("$\\pi$ proposal mass on the poison slot\n(floored at $10^{-5}$ for the log axis)")
    ax[1].set_title("$\\pi$ proposal mass on the poison twin (log)", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.25, which="both")
    save(fig, "fig2_trust_by_level.png")

    # ---- fig 3: the address-book battery ------------------------------------------------
    ROWS = ("base", "no_prim", "no_table", "no_table_no_prim")
    PREF = ("fid", "audit_prop_kN", "audit_prop_k", "route_native", "prop_k_d0",
            "audit_prop_k_ref", "route_native_ref", "audit_prop_k_lib", "route_native_lib")
    names = ([n for n in PREF if n in arms and arms[n].get("battery")]
             + [n for n in arms if n not in PREF and arms[n].get("battery")])
    fig, a = plt.subplots(figsize=(11, 4.8))
    w = 0.2
    cols = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")
    for j, r in enumerate(ROWS):
        xs = [i + (j - 1.5) * w for i in range(len(names))]
        ys = [arms[n]["battery"][r]["e_piece"] for n in names]
        bars = a.bar(xs, ys, w, color=cols[j], label=r)
        for x, y in zip(xs, ys):
            a.text(x, y * 1.05, f"{y:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    for lab, v, col in (("reflex through the trunk",
                         arms["reflex"]["cycles"][-1]["e_piece"], "#333333"),
                        ("audit_all", arms["audit_all"]["cycles"][-1]["e_piece"], "#2ca02c")):
        a.axhline(v, color=col, ls="--", lw=1.2, alpha=0.85)
        a.text(0.995, v, f" {lab} {v:.4f}", transform=a.get_yaxis_transform(), ha="right",
               va="bottom", fontsize=7.5, color=col)
    a.set_yscale("log")
    lo = min(arms[n]["battery"][r]["e_piece"] for n in names for r in ROWS)
    hi = max(arms[n]["battery"][r]["e_piece"] for n in names for r in ROWS)
    a.set_ylim(lo * 0.55, hi * 4.0)                # headroom for the rotated value labels
    a.set_xticks(range(len(names)))
    a.set_xticklabels([n + ("\n($\\Delta$ = 0)" if n == "prop_k_d0" else
                            "\n(untrained heads)" if n == "fid" else "") for n in names],
                      fontsize=8.5)
    a.set_ylabel(YL + ", log")
    a.set_title("the address-book battery, on each arm's final trained state", fontsize=10)
    a.legend(fontsize=8, ncol=4, loc="upper left"); a.grid(alpha=0.25, axis="y", which="both")
    save(fig, "fig3_address_book_battery.png")


    # ---- fig 4: pi's imitation filter, against the reference run ------------------------
    band = [n for n in arms if arms[n]["cycles"][-1].get("pi_filter") == "band"]
    if band:
        ref = None
        rt = d["config"].get("ref_tag")
        rp = os.path.join(DATA_DIR, "practice_solo", str(rt or ""), "solo.json")
        if rt and os.path.exists(rp):
            ref = json.load(open(rp))
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        PAIR = (("audit_prop_k", ["audit_prop_k_ref", "audit_prop_k_lib"]),
                ("route_native", ["route_native_ref", "route_native_lib"]))
        BC = {"_ref": "#e377c2", "_lib": "#17becf"}
        for j, (base, bands) in enumerate(PAIR):
            a = ax[j]
            n = max(len(arms[b]["cycles"]) for b in bands if b in arms)
            if ref and base in ref.get("arms", {}):
                cy = ref["arms"][base]["cycles"][:n]
                a.plot([c["cycle"] for c in cy], [c["e_piece"] for c in cy], "-",
                       color="#7f7f7f", lw=2.2, alpha=0.9, label=f"{base} ({rt}, median filter)")
            if base in arms:
                cy = arms[base]["cycles"]
                a.plot([c["cycle"] for c in cy], [c["e_piece"] for c in cy], "--",
                       color="#000000", lw=1.2, label=f"{base} (this run, median filter)")
            for b in bands:
                if b not in arms:
                    continue
                cy = arms[b]["cycles"]
                bd = cy[-1].get("pi_band")
                a.plot([c["cycle"] for c in cy], [c["e_piece"] for c in cy], "-",
                       color=BC[b[-4:]], lw=1.8, label=f"{b} (band {bd:.4f})")
            for lab, v, col in (("reflex through the trunk",
                                 arms["reflex"]["cycles"][-1]["e_piece"], "#333333"),
                                ("audit_all", arms["audit_all"]["cycles"][-1]["e_piece"],
                                 "#2ca02c"),
                                ("band: ref_play", 0.1066, BC["_ref"]),
                                ("band: donor lib_seg", dr["lib_seg@8"]["e_piece"], BC["_lib"])):
                a.axhline(v, color=col, ls=":", lw=1.2, alpha=0.85)
                a.text(0.995, v, f" {lab} {v:.4f}", transform=a.get_yaxis_transform(),
                       ha="right", va="bottom", fontsize=7, color=col)
            a.set_xlabel("practice cycle")
            a.set_title(f"{base}: piece error over cycles, $\\Delta$ = 8", fontsize=10)
            a.legend(fontsize=7.5, loc="lower left"); a.grid(alpha=0.25)
        ax[0].set_ylabel(YL)
        save(fig, "fig4_pi_filter_probe.png")

    volume.commit()
    return made


@app.local_entrypoint()
def solo_figs(tag: str = "s0"):
    made = make_solo_figures.remote(tag)
    print(f"[figs] wrote {made} to /data/practice_solo/{tag}/figs on the volume")


def draw_figures(tag):
    """Dispatch the remote job, then pull the PNGs into `figures/<tag>/`."""
    here = os.path.relpath(os.path.abspath(__file__),
                           os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    subprocess.run(["modal", "run", f"{here}::solo_figs", "--tag", tag], check=True, cwd=root)
    dst = os.path.join(HERE, "figures", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOL,
                    f"/practice_solo/{tag}/figs", dst], check=True)
    nest = os.path.join(dst, "figs")          # `volume get` of a directory nests it; flatten
    if os.path.isdir(nest):
        for f in os.listdir(nest):
            os.replace(os.path.join(nest, f), os.path.join(dst, f))
        os.rmdir(nest)
    print(f"[figs] pulled to {dst}: {sorted(os.listdir(dst))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="draw the three figures REMOTELY (no matplotlib locally) and pull them "
                         "into figures/<tag>/")
    a = ap.parse_args()
    path = fetch(a.tag) if a.fetch else os.path.join(HERE, "results", a.tag, "solo.json")
    d = json.load(open(path))
    out = []
    report(d, out)
    dst = os.path.join(HERE, "results", a.tag, "solo_report.txt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\n[written] {dst}")
    if a.figures:
        draw_figures(a.tag)


if __name__ == "__main__":
    main()
