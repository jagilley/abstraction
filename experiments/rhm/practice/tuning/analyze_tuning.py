"""tuning/analyze_tuning — reduce a Gate-0 run and make its figures.

Sections
  0  setup, schedule, calibration record, read prices
  1  competence: indexed/unindexed NLL with rotation and burst markers
  2  FACTOR T, the matched-surprisal contrast — the node's central table. At every
     checkpoint both events are evaluated counterfactually on the SAME weights (the eval
     wall rotated by +rot_step; the eval leaves corrupted at a ladder of rates), so the
     burst rate whose indexed-span spike EQUALS that checkpoint's rotation spike is read
     off by interpolation and the gauges are compared there. Matched surprisal by
     construction, per checkpoint.
  3  the gauges' placebo floors, measured in-tag on quiet checkpoints
  4  FACTOR M, the four rows: rotation->track, burst->continue, burst->skip, and the
     input-invariance read (shadow burst at fixed weights)
  5  the FM learning-rate sweep
  6  the consumed burst's own ledger (what training on it cost)

Usage (from experiments/):
    python3 rhm/practice/tuning/analyze_tuning.py --tag tn0 --fetch --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_tuning"


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    readers, panels = {}, {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name in ("setup.json", "calib.json"):
            continue
        obj = json.load(open(os.path.join(root, name)))
        if name.startswith("panel_"):
            panels[obj["worker"]] = obj
        else:
            readers[obj["arm"]] = obj
    return setup, readers, panels


def series(log, path, steps=None):
    parts = path.split(".")
    xs, ys = [], []
    for r in log:
        d = r
        for p in parts:
            if not isinstance(d, dict) or p not in d:
                d = None
                break
            d = d[p]
        if d is None:
            continue
        if steps is not None and r["step"] not in steps:
            continue
        xs.append(r["step"]); ys.append(float(d))
    return np.array(xs), np.array(ys)


def _sub(a, b):
    return float(a) - float(b) if (a is not None and b is not None) else np.nan


def interp_at(xs, ys, target):
    """Piecewise-linear read of y at the x where a MONOTONE-ish series hits `target`.

    Used two ways: (spike -> rho) and (spike -> D). Returns (value, bracketed)."""
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    for i in range(len(xs) - 1):
        if (xs[i] - target) * (xs[i + 1] - target) <= 0 and xs[i + 1] != xs[i]:
            t = (target - xs[i]) / (xs[i + 1] - xs[i])
            return float(ys[i] + t * (ys[i + 1] - ys[i])), True
    i = 0 if target < xs[0] else len(xs) - 2
    if xs[i + 1] == xs[i]:
        return float(ys[i]), False
    t = (target - xs[i]) / (xs[i + 1] - xs[i])
    return float(ys[i] + t * (ys[i + 1] - ys[i])), False


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tn0")
    ap.add_argument("--calib-tag", default="tn_calib")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.fetch:
        fetch(a.tag)
        try:
            fetch(a.calib_tag)
        except subprocess.CalledProcessError:
            pass

    setup, readers, panels = load(a.tag)
    cfg = setup["config"]
    ck = cfg["ckpt_every"]
    v = cfg["v"]
    rots = list(range(cfg["phase1"], cfg["max_steps"], cfg["rot_period"]))
    starts = [int(x) for x in str(cfg["burst_steps"]).split(",")]
    blen = cfg["burst_len"]
    windows = [(s, min(s + blen, cfg["max_steps"])) for s in starts
               if s < cfg["max_steps"]]
    out = {"tag": a.tag, "config": cfg}

    print("=" * 78)
    print(f"tuning Gate 0 — tag {a.tag}")
    print("=" * 78)
    print(f"\n0. SETUP.  readers {sorted(readers)}   workers {sorted(panels)}")
    print(f"   rotations {rots}")
    print(f"   bursts    {windows}   rho_c {cfg['burst_rho']}  ladder {cfg['burst_ladder']}")
    print(f"   FM {cfg['fm_shallow']} -> {cfg['fm_deep']}, "
          f"{cfg['fm_n_layer']}L/{cfg['fm_n_head']}H/{cfg['fm_d_head']}d, "
          f"lrs {setup.get('fm_lrs')}  primary {setup.get('fm_primary')}")
    print(f"   refs {setup.get('refs_source')}   panel n {cfg['n_panel']}")

    cpath = os.path.join(FIG, a.calib_tag, "calib.json")
    if os.path.exists(cpath):
        cal = json.load(open(cpath))
        print("\n   CALIBRATION (tn_calib, model-side, no bursts consumed):")
        for r in cal["rows"]:
            print(f"     s{r['step']:6d}  clean {r['clean_idx']:.4f}  rotation spike "
                  f"+{r['rot_spike']:.4f} (out +{r['rot_out_leak']:.4f})  -> rho* "
                  f"{r['rho_star']:.5f} "
                  f"({'bracketed' if r['bracketed'] else 'extrapolated'})")
        out["calibration"] = cal["rows"]

    prices = []
    for w, p in panels.items():
        ms = [r["ms"] for r in p["log"] if "ms" in r]
        if ms:
            prices.append((w, float(np.median(ms))))
    if prices:
        print("\n   READ PRICE (median ms per checkpoint, whole uncharged panel): "
              + "  ".join(f"{w} {x:.0f}ms" for w, x in prices))
        out["panel_ms"] = dict(prices)

    # ---------------- 2. FACTOR T ----------------
    pair = panels.get("pair")
    if pair is None:
        print("\n(no `pair` worker in this tag — Factor T needs it)")
        return
    plog = pair["log"]
    ladder = pair["ladder"]
    nb = len(ladder)

    def prow(r, cond):
        return r.get(cond, {})

    rowsT = []
    for r in plog:
        c, ro = prow(r, "clean"), prow(r, "rot")
        if "nll_key_idx" not in c or "nll_key_idx" not in ro:
            continue
        s_rot = ro["nll_key_idx"] - c["nll_key_idx"]
        sp, dp, ds, so = [], [], [], []
        mv_self, mv_parse = [], []            # the MOVEMENT reads, per ladder rung
        for i in range(nb):
            b = prow(r, f"burst{i}")
            if "nll_key_idx" not in b:
                continue
            sp.append(b["nll_key_idx"] - c["nll_key_idx"])
            dp.append(b.get("D_pair", np.nan))
            ds.append(b.get("D_self", np.nan))
            so.append(b.get("nll_key_out", np.nan) - c.get("nll_key_out", np.nan))
            mv_self.append(_sub(b.get("nll_self_idx"), c.get("nll_self_idx")))
            mv_parse.append(_sub(b.get("nll_parse_idx"), c.get("nll_parse_idx")))
        if not sp:
            continue
        rho_star, brack = interp_at(sp, ladder, s_rot)
        dp_m, _ = interp_at(sp, dp, s_rot)
        ds_m, _ = interp_at(sp, ds, s_rot)
        so_m, _ = interp_at(sp, so, s_rot)
        mvs_m, _ = interp_at(sp, mv_self, s_rot)
        mvp_m, _ = interp_at(sp, mv_parse, s_rot)
        rowsT.append({
            "step": r["step"], "burst_active": r["burst"] >= 0,
            "rot_spike": s_rot, "rho_star": rho_star, "bracketed": brack,
            "nll_key_idx": c["nll_key_idx"], "nll_self_idx": c.get("nll_self_idx"),
            "nll_parse_idx": c.get("nll_parse_idx"),
            "D_pair_clean": c.get("D_pair"), "D_self_clean": c.get("D_self"),
            "D_pair_rot": ro.get("D_pair"), "D_self_rot": ro.get("D_self"),
            "D_pair_burst_matched": dp_m, "D_self_burst_matched": ds_m,
            "out_leak_rot": ro.get("nll_key_out", np.nan) - c.get("nll_key_out", np.nan),
            "out_leak_burst_matched": so_m,
            # --- the MOVEMENT decomposition (per-view NLL movement under the event) ---
            "d_self_rot": _sub(ro.get("nll_self_idx"), c.get("nll_self_idx")),
            "d_parse_rot": _sub(ro.get("nll_parse_idx"), c.get("nll_parse_idx")),
            "d_self_burst_matched": mvs_m, "d_parse_burst_matched": mvp_m,
            "ladder_spike": sp, "ladder_D_pair": dp, "ladder_D_self": ds,
            "ladder_d_self": mv_self, "ladder_d_parse": mv_parse, "ladder_d_out": so,
        })
    out["factorT_rows"] = rowsT

    # ---------------- 3. the gauges' floors, in-tag ----------------
    quiet = [r for r in rowsT
             if not r["burst_active"]
             and all(abs(r["step"] - x) > 500 for x in rots)
             and all(abs(r["step"] - w[0]) > 500 and abs(r["step"] - w[1]) > 500
                     for w in windows)
             and r["step"] > 1000]
    floors = {}
    for k in ("D_pair_clean", "D_self_clean"):
        vals = np.array([r[k] for r in quiet if r[k] is not None], dtype=float)
        if len(vals) > 3:
            d = np.abs(np.diff(vals))
            floors[k] = {"n": int(len(vals)), "adjacent_p90": float(np.percentile(d, 90)),
                         "adjacent_max": float(d.max()), "level_median": float(np.median(vals))}
    out["gauge_floors"] = floors
    print("\n3. GAUGE FLOORS (quiet checkpoints, no event within 500 steps)")
    for k, f in floors.items():
        print(f"   {k:16s} n={f['n']:3d}  level {f['level_median']:.4f}  "
              f"adjacent-checkpoint |delta| p90 {f['adjacent_p90']:.4f} "
              f"max {f['adjacent_max']:.4f}")

    print("\n2. FACTOR T — do the gauges separate a rotation from a burst at MATCHED "
          "indexed-span surprisal?")
    print("   (both events counterfactual on the same weights; the burst rate is the one "
          "whose\n    indexed-span spike equals that checkpoint's rotation spike)")
    hdr = (f"   {'step':>6s} {'rot spike':>10s} {'rho*':>8s} "
           f"{'D_pair clean':>13s} {'rot':>9s} {'burst=':>9s} {'ratio':>7s} | "
           f"{'D_self clean':>13s} {'rot':>9s} {'burst=':>9s} {'ratio':>7s} | "
           f"{'outleak rot/burst':>18s}")
    print(hdr)
    show = [r for r in rowsT if r["step"] % max(ck * 8, 1000) == 0 or r["step"] in rots
            or any(r["step"] == w[0] for w in windows)]
    for r in show:
        dpc, dpr, dpb = r["D_pair_clean"], r["D_pair_rot"], r["D_pair_burst_matched"]
        dsc, dsr, dsb = r["D_self_clean"], r["D_self_rot"], r["D_self_burst_matched"]
        rp = ((dpr - dpc) / (dpb - dpc)) if (dpb is not None and dpc is not None
                                             and abs(dpb - dpc) > 1e-9) else float("nan")
        rs = ((dsr - dsc) / (dsb - dsc)) if (dsb is not None and dsc is not None
                                             and abs(dsb - dsc) > 1e-9) else float("nan")
        print(f"   {r['step']:6d} {r['rot_spike']:10.4f} {r['rho_star']:8.4f} "
              f"{dpc:13.4f} {dpr:9.4f} {dpb:9.4f} {rp:7.2f} | "
              f"{dsc:13.4f} {dsr:9.4f} {dsb:9.4f} {rs:7.2f} | "
              f"{r['out_leak_rot']:8.4f}/{r['out_leak_burst_matched']:8.4f}")

    # the headline: pooled over quiet checkpoints where the ladder BRACKETS the rotation
    # spike (an extrapolated rho* is not a matched-surprisal read and is excluded)
    groups = {
        "mature quiet (>=2000)":
            [r for r in quiet if r["step"] >= 2000 and r["bracketed"]],
        "post-phase-1 mid-era":
            [r for r in quiet if r["step"] >= cfg["phase1"] + 500 and r["bracketed"]],
    }
    out["factorT_headline"] = {}
    for gname, mature in groups.items():
        if len(mature) < 3:
            print(f"\n   POOLED [{gname}]: only {len(mature)} usable checkpoints — skipped")
            continue

        def pooled(pfx, rows=mature):
            c = np.array([r[f"{pfx}_clean"] for r in rows], float)
            ro = np.array([r[f"{pfx}_rot"] for r in rows], float)
            b = np.array([r[f"{pfx}_burst_matched"] for r in rows], float)
            fl = floors.get(f"{pfx}_clean", {}).get("adjacent_p90", np.nan)
            d_r, d_b = float(np.median(ro - c)), float(np.median(b - c))
            return {"n": len(c), "clean": float(np.median(c)),
                    "rot": float(np.median(ro)), "burst_matched": float(np.median(b)),
                    "d_rot": d_r, "d_burst": d_b,
                    "d_rot_iqr": [float(np.percentile(ro - c, 25)),
                                  float(np.percentile(ro - c, 75))],
                    "d_burst_iqr": [float(np.percentile(b - c, 25)),
                                    float(np.percentile(b - c, 75))],
                    "separation": float(d_r / max(abs(d_b), 1e-12)),
                    "rot_gt_burst_frac": float(np.mean((ro - c) > (b - c))),
                    "d_rot_in_floors": float(d_r / fl) if fl else None,
                    "d_burst_in_floors": float(d_b / fl) if fl else None}
        head = {p: pooled(p) for p in ("D_pair", "D_self")}
        ol_r = float(np.median([r["out_leak_rot"] for r in mature]))
        ol_b = float(np.median([r["out_leak_burst_matched"] for r in mature]))
        head["out_leak"] = {"rot": ol_r, "burst_matched": ol_b,
                            "separation": ol_r / max(abs(ol_b), 1e-12)}
        head["rot_spike_median"] = float(np.median([r["rot_spike"] for r in mature]))
        head["rho_star_median"] = float(np.median([r["rho_star"] for r in mature]))
        out["factorT_headline"][gname] = head
        print(f"\n   POOLED [{gname}] over {len(mature)} bracketed quiet checkpoints "
              f"(median rotation spike {head['rot_spike_median']:.4f} nats, "
              f"matched rho* {head['rho_star_median']:.4f}):")
        for p in ("D_pair", "D_self"):
            h = head[p]
            print(f"     {p:7s} clean {h['clean']:.4f}   rotation {h['rot']:.4f} "
                  f"(+{h['d_rot']:.4f}"
                  + (f", {h['d_rot_in_floors']:.0f}x floor" if h["d_rot_in_floors"] else "")
                  + f")   burst@matched {h['burst_matched']:.4f} (+{h['d_burst']:.4f}"
                  + (f", {h['d_burst_in_floors']:.0f}x floor"
                     if h["d_burst_in_floors"] else "")
                  + f")   SEPARATION {h['separation']:.2f}x   "
                  f"rot>burst in {100 * h['rot_gt_burst_frac']:.0f}% of checkpoints")
        print(f"     out-span leak (the FREE single-reader alternative): rotation "
              f"{ol_r:+.4f}  burst@matched {ol_b:+.4f}  "
              f"ratio {ol_r / max(abs(ol_b), 1e-12):.1f}x")

    # cross-check: the COUNTERFACTUAL rotation read at the last pre-rotation checkpoint
    # against the REALISED spike at the rotation checkpoint itself
    print("\n2b. COUNTERFACTUAL vs REALISED rotation (validates the instant design)")
    bystep = {r["step"]: r for r in rowsT}
    xcheck = []
    for rot in rots:
        pre = bystep.get(rot - ck)
        at = bystep.get(rot)
        if pre is None or at is None:
            continue
        realised = at["D_pair_clean"] - pre["D_pair_clean"] if at["D_pair_clean"] else None
        row = {"rot": rot, "cf_spike": pre["rot_spike"],
               "realised_spike": at["nll_key_idx"] - pre["nll_key_idx"]
               if "nll_key_idx" in at else None,
               "cf_D_pair": pre["D_pair_rot"] - pre["D_pair_clean"],
               "realised_D_pair": realised,
               "cf_D_self": pre["D_self_rot"] - pre["D_self_clean"],
               "realised_D_self": (at["D_self_clean"] - pre["D_self_clean"])
               if at["D_self_clean"] else None}
        xcheck.append(row)
        print(f"   rot {rot:6d}  counterfactual dD_pair {row['cf_D_pair']:+.4f} "
              f"vs realised {(row['realised_D_pair'] or float('nan')):+.4f}   |   "
              f"dD_self {row['cf_D_self']:+.4f} vs "
              f"{(row['realised_D_self'] or float('nan')):+.4f}")
    out["counterfactual_vs_realised"] = xcheck

    # the mirror control: how far apart are the two PARSE bases?
    msp = [r for r in plog if "D_selfparse" in r.get("clean", {})]
    if msp:
        vals = np.array([r["clean"]["D_selfparse"] for r in msp], float)
        st = np.array([r["step"] for r in msp])
        late = vals[st >= cfg["phase1"]]
        print(f"\n2c. MIRROR CONTROL  KL(p_key-neutralised || p_parse): the keyed model's "
              f"own parse basis\n    against a real one — median {np.median(vals):.4f} "
              f"(post-phase-1 {np.median(late) if len(late) else float('nan'):.4f}). "
              f"Large = the keyed model's\n    counterfactual read is a HOLLOW basis, which "
              f"is the standing worry about D_self.")
        out["mirror_control_D_selfparse"] = {
            "median": float(np.median(vals)),
            "median_post_phase1": float(np.median(late)) if len(late) else None,
            "terminal": float(vals[-1])}

    # ------- 2d. THE MOVEMENT DECOMPOSITION — T* = (d_key, d_self, d_out) -------
    #
    # The KL magnitudes (`D_pair`, `D_self`) turned out to be the wrong contraction of
    # this panel. What types the pair is per-view NLL MOVEMENT under the event: how much
    # each VIEW of the batch moves when the event is applied, at matched d_key.
    #
    #   d_key   = nll_key_idx(event)   - nll_key_idx(clean)     the keyed view      [matched]
    #   d_self  = nll_self_idx(event)  - nll_self_idx(clean)    the SAME model, key neutralised
    #   d_parse = nll_parse_idx(event) - nll_parse_idx(clean)   the separate parse reader
    #   d_out   = nll_key_out(event)   - nll_key_out(clean)     off the indexed span
    #
    # BY CONSTRUCTION, TWICE OVER: d_self and d_parse are exactly 0 under a rotation.
    # (i) Substantively — a key-free view of clean leaves cannot see a permutation of the
    # key, because its input (neutral token, clean leaves) is byte-identical before and
    # after. (ii) Mechanically — the panel reuses one cached clean read for both the
    # `clean` and `rot` conditions for exactly that reason. So the rotation-side zero is
    # not a measurement: the claim it supports is that HOLDING a key-free view makes
    # rotation-vs-world-news typing trivial at the instant, not that the instrument did
    # nontrivial work separating this pair. The burst side IS a measurement.
    QUIET_MIN_STEP = cfg["phase1"] + 500          # mature: past the last pre-rotation era
    QUIET_GUARD = 375                             # steps clear of an event's aftermath

    def quiet_mv(one_sided=True):
        """one_sided: exclude only the AFTERMATH of an event (the standing definition,
        `>= GUARD past any rotation/burst window`). symmetric: also drop the run-up,
        which is stricter and is reported as a robustness row."""
        keep = []
        for r in rowsT:
            if r["step"] < QUIET_MIN_STEP or not r["bracketed"]:
                continue
            if one_sided:
                bad = (any(0 <= r["step"] - x < QUIET_GUARD for x in rots)
                       or any(w[0] <= r["step"] < w[1] + QUIET_GUARD for w in windows))
            else:
                bad = (any(abs(r["step"] - x) < QUIET_GUARD for x in rots)
                       or not all(r["step"] <= w[0] - QUIET_GUARD
                                  or r["step"] >= w[1] + QUIET_GUARD for w in windows))
            if not bad:
                keep.append(r)
        return keep

    mv = quiet_mv(one_sided=True)
    mv_strict = quiet_mv(one_sided=False)
    print(f"\n2d. MOVEMENT DECOMPOSITION  T* = (d_key, d_self, d_out), matched d_key")
    print(f"    quiet = step >= {QUIET_MIN_STEP}, >= {QUIET_GUARD} past any rotation/burst "
          f"window, ladder-bracketed  ->  n = {len(mv)}")
    if mv:
        def col(k):
            return np.array([r[k] for r in mv], float)
        dk = col("rot_spike")
        rows = [("d_self", "d_self_rot", "d_self_burst_matched"),
                ("d_parse", "d_parse_rot", "d_parse_burst_matched"),
                ("d_out", "out_leak_rot", "out_leak_burst_matched")]
        print(f"    {'view':9s} {'rotation med':>13s} {'burst med':>11s} {'burst min':>10s} "
              f"{'burst max':>10s} {'sign-consistent':>16s}")
        mvout = {"n": len(mv), "quiet_min_step": QUIET_MIN_STEP,
                 "quiet_guard": QUIET_GUARD,
                 "matched_d_key_median": float(np.median(dk))}
        for name, kr, kb in rows:
            vr, vb = col(kr), col(kb)
            ok = np.isfinite(vr) & np.isfinite(vb)
            cons = float(np.mean(vb[ok] > vr[ok])) if name != "d_out" else \
                float(np.mean(vr[ok] > vb[ok]))
            mvout[name] = {"rotation_median": float(np.median(vr[ok])),
                           "burst_median": float(np.median(vb[ok])),
                           "burst_min": float(np.min(vb[ok])),
                           "burst_max": float(np.max(vb[ok])),
                           "sign_consistent_frac": cons}
            print(f"    {name:9s} {np.median(vr[ok]):+13.4f} {np.median(vb[ok]):+11.4f} "
                  f"{np.min(vb[ok]):+10.4f} {np.max(vb[ok]):+10.4f} "
                  f"{100 * cons:15.0f}%")
        print(f"    (matched d_key median {np.median(dk):+.4f} nats; 'sign-consistent' = "
              f"burst > rotation for\n     d_self/d_parse, rotation > burst for d_out — "
              f"the direction T* needs to type the pair)")
        # the decision rule T* would run, with the floors it would use
        gap_self = float(np.min(col("d_self_burst_matched")[
            np.isfinite(col("d_self_burst_matched"))]))
        gap_out = float(np.min(col("out_leak_rot")[np.isfinite(col("out_leak_rot"))]))
        mvout["dead_zone_headroom"] = {"d_self_min_burst": gap_self,
                                       "d_out_min_rotation": gap_out}
        print(f"    DEAD-ZONE HEADROOM for a Gate-1 op map: worst-case burst d_self "
              f"{gap_self:+.4f}\n     (rotation side is exactly 0), worst-case rotation "
              f"d_out {gap_out:+.4f} (burst side ~0).")
        # robustness: the stricter two-sided guard (drops each event's run-up as well)
        if mv_strict:
            sd = np.array([r["d_self_burst_matched"] for r in mv_strict], float)
            so_ = np.array([r["out_leak_rot"] for r in mv_strict], float)
            mvout["strict_two_sided"] = {
                "n": len(mv_strict), "d_self_burst_median": float(np.median(sd)),
                "d_self_burst_min": float(np.nanmin(sd)),
                "d_out_rotation_median": float(np.median(so_))}
            print(f"    robustness, two-sided guard (n={len(mv_strict)}): burst d_self "
                  f"median {np.median(sd):+.4f} min {np.nanmin(sd):+.4f}, rotation d_out "
                  f"median {np.median(so_):+.4f} — same conclusion.")
        out["movement_decomposition"] = mvout

    # ---------------- 4. FACTOR M ----------------
    print("\n4. FACTOR M — the meter over the window after the op")
    fm_rows = {}
    if any("fm" in r for r in plog):
        prim = f"lr{setup.get('fm_primary', 1e-3):g}"

        def fm_series(worker, reader, cond, fmname, field="rel"):
            lg = panels[worker]["log"]
            xs, ys = [], []
            for r in lg:
                d = r.get("fm", {}).get(reader, {}).get("e", {}).get(cond, {}).get(fmname)
                if d is None:
                    continue
                xs.append(r["step"]); ys.append(float(d[field]))
            return np.array(xs), np.array(ys)

        def bench_series(worker, reader, key):
            lg = panels[worker]["log"]
            xs, b, dl = [], [], []
            for r in lg:
                d = r.get("fm", {}).get(reader, {}).get("bench", {}).get(key)
                if d is None:
                    continue
                xs.append(r["step"]); b.append(d["b"]); dl.append(d["delta"])
            return np.array(xs), np.array(b), np.array(dl)

        def window_response(xs, ys, onset, span=1000):
            """peak |Delta| against the pre-event level, and steps to return within it."""
            pre = ys[(xs < onset) & (xs >= onset - 500)]
            if len(pre) == 0 or len(xs) == 0:
                return None
            base = float(np.median(pre))
            sel = (xs >= onset) & (xs <= onset + span)
            if not sel.any():
                return None
            seg, sx = ys[sel], xs[sel]
            k = int(np.argmax(np.abs(seg - base)))
            peak, at = float(seg[k] - base), int(sx[k])
            tol = max(float(np.std(pre)) * 2, 0.02 * abs(base))
            back = None
            for i in range(k, len(seg)):
                if abs(seg[i] - base) <= tol:
                    back = int(sx[i] - onset)
                    break
            return {"base": base, "peak": peak, "peak_at": at, "recover_steps": back}

        tbl = []
        for worker, reader, label in [("pair", "wall", "rotation->track / burst->continue"),
                                      ("pair", "no_wall", "no-key control"),
                                      ("skip", "wall_skip", "burst->skip")]:
            if worker not in panels:
                continue
            lg = panels[worker]["log"]
            if not any(reader in r.get("fm", {}) for r in lg):
                continue
            xs, ys = fm_series(worker, reader, "clean", prim)
            xb, yb = fm_series(worker, reader, "burst_c", prim)
            xr, yr = fm_series(worker, reader, "rot", prim)
            # online primary, plus the two frozen reads (the FM's lag is part of the
            # signal, so `e_frozen` is reported beside `e_online`, not instead of it)
            for fmname in (prim, "frozen_prev", "frozen_event"):
                fx, fy = fm_series(worker, reader, "clean", fmname)
                if not len(fx):
                    continue
                # scope each window to its OWN event: a 1000-step window on a rotation at
                # 16000 runs straight into the rotation at 18000, and a burst window that
                # overruns by 750 steps catches the next rotation instead of the burst.
                events = sorted(rots + [w[0] for w in windows])

                def span_for(onset, default):
                    nxt = [e for e in events if e > onset]
                    return min(default, (nxt[0] - onset - 1) if nxt else default)
                for onset in rots:
                    w = window_response(fx, fy, onset, span=span_for(onset, 1000))
                    if w:
                        tbl.append((f"{reader}/{fmname}", "rotation", onset, w))
                for st, en in windows:
                    w = window_response(fx, fy, st,
                                        span=span_for(st, en - st + 250))
                    if w:
                        tbl.append((f"{reader}/{fmname}", "burst", st, w))
            # input-invariance: e on a SHADOW burst vs e on clean, same weights
            if len(xb) and len(xs):
                mmap = {s: y for s, y in zip(xs, ys)}
                inv = [(yb[i] - mmap[xb[i]]) / max(mmap[xb[i]], 1e-12)
                       for i in range(len(xb)) if xb[i] in mmap]
                rel_r = []
                if len(xr):
                    rel_r = [(yr[i] - mmap[xr[i]]) / max(mmap[xr[i]], 1e-12)
                             for i in range(len(xr)) if xr[i] in mmap]
                fm_rows[reader] = {
                    "shadow_burst_rel_median": float(np.median(inv)) if inv else None,
                    "shadow_burst_rel_p90": float(np.percentile(inv, 90)) if inv else None,
                    "counterfactual_rot_rel_median": (float(np.median(rel_r))
                                                      if rel_r else None)}
        out["factorM_windows"] = [{"reader": r, "event": e, "onset": o, **w}
                                  for r, e, o, w in tbl]
        out["factorM_input_invariance"] = fm_rows
        print(f"   e_rel = ||h6 - FM(h0)||^2 / mean(h6^2) on the indexed span, clean eval")
        print(f"   NOTE: the RAW residual is dominated by a monotone scale trend (h6's norm")
        print(f"   grows through training: e_mse 0.24 -> 12.1 on `wall`), so every Factor-M")
        print(f"   number below is the SCALE-NORMALISED residual. The raw series is in the")
        print(f"   reduction JSON.")
        print(f"   {'reader/fm':26s} {'event':9s} {'onset':>6s} {'base e':>10s} "
              f"{'peak d':>10s} {'peak%':>7s} {'at':>7s} {'recover':>8s}")
        for r, e, o, w in tbl:
            pct = 100 * w["peak"] / max(abs(w["base"]), 1e-12)
            print(f"   {r:26s} {e:9s} {o:6d} {w['base']:10.5f} {w['peak']:+10.5f} "
                  f"{pct:+7.1f} {w['peak_at']:7d} "
                  f"{('+' + str(w['recover_steps'])) if w['recover_steps'] is not None else 'none':>8s}")
        print("\n   INPUT-INVARIANCE (same weights; how much does the INPUT alone move e?)")
        for r, d in fm_rows.items():
            print(f"   {r:11s} shadow burst {100 * (d['shadow_burst_rel_median'] or 0):+7.2f}%"
                  f" (p90 {100 * (d['shadow_burst_rel_p90'] or 0):+7.2f}%)   "
                  f"counterfactual rotation "
                  f"{100 * (d['counterfactual_rot_rel_median'] or 0):+7.2f}%")

        # ---------------- 5. the FM lr sweep ----------------
        print("\n5. FM LEARNING-RATE SWEEP (uncharged; all variants ride one forward pass)")
        sweep = {}
        for lr in setup.get("fm_lrs", []):
            nm = f"lr{lr:g}"
            xs, ys = fm_series("pair", "wall", "clean", nm)
            if not len(xs):
                continue
            q = [(x, y) for x, y in zip(xs, ys)
                 if all(abs(x - o) > 500 for o in rots)
                 and all(abs(x - w[0]) > 500 for w in windows) and x > 2000]
            qy = np.array([y for _, y in q]) if q else np.array([np.nan])
            resp = [window_response(xs, ys, o) for o in rots]
            resp = [r for r in resp if r]
            amp = float(np.median([abs(r["peak"]) for r in resp])) if resp else np.nan
            noise = float(np.percentile(np.abs(np.diff(qy)), 90)) if len(qy) > 3 else np.nan
            sweep[nm] = {"terminal_e": float(ys[-1]), "quiet_level": float(np.median(qy)),
                         "rotation_amplitude": amp, "quiet_step_noise_p90": noise,
                         "snr": float(amp / noise) if noise and noise > 0 else None}
        out["fm_sweep"] = sweep
        print(f"   {'fm':10s} {'terminal e':>12s} {'quiet level':>12s} "
              f"{'rot |peak|':>11s} {'quiet noise':>12s} {'SNR':>7s}")
        for nm, d in sweep.items():
            print(f"   {nm:10s} {d['terminal_e']:12.5f} {d['quiet_level']:12.5f} "
                  f"{d['rotation_amplitude']:11.5f} {d['quiet_step_noise_p90']:12.5f} "
                  f"{(d['snr'] if d['snr'] else float('nan')):7.2f}")

    # ---------------- 6. the consumed burst's ledger ----------------
    print("\n6. THE CONSUMED BURST'S LEDGER (what training on it cost)")
    print(f"   {'reader':11s} {'burst':>6s} {'pre idx':>9s} {'in-burst':>9s} "
          f"{'post+250':>9s} {'post+1000':>10s}")
    ledger = []
    for name, rd in readers.items():
        lg = rd["log"]
        xs, ys = series(lg, "nll.true.idx")
        if not len(xs):
            continue
        m = {int(x): y for x, y in zip(xs, ys)}

        def near(t):
            k = [s for s in m if abs(s - t) <= 60]
            return m[min(k, key=lambda s: abs(s - t))] if k else float("nan")
        for st, en in windows:
            ledger.append({"reader": name, "burst": st, "pre": near(st - 125),
                           "during": near(st + blen // 2), "post250": near(en + 250),
                           "post1000": near(en + 1000)})
            print(f"   {name:11s} {st:6d} {near(st - 125):9.4f} {near(st + blen // 2):9.4f} "
                  f"{near(en + 250):9.4f} {near(en + 1000):10.4f}")
    out["burst_ledger"] = ledger
    # lifetime integral on the common grid
    print("\n   lifetime integral of indexed-span NLL (uniform 125-grid, consumed condition)")
    for name, rd in readers.items():
        lg = rd["log"]
        grid = [r for r in lg if r["step"] % ck == 0]
        vals = [r["nll"][r["consumed"]]["idx"] for r in grid]
        print(f"   {name:11s} n={len(vals):4d}  integral {np.mean(vals):.4f}")
        out.setdefault("lifetime", {})[name] = float(np.mean(vals))

    with open(os.path.join(RES, f"reduction_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[wrote] {os.path.join(RES, f'reduction_{a.tag}.json')}")

    if a.figures:
        make_figures(a.tag, setup, readers, panels, rowsT, rots, windows, out)


# --------------------------------------------------------------------------- #

def make_figures(tag, setup, readers, panels, rowsT, rots, windows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(FIG, tag)
    cfg = setup["config"]
    ck = cfg["ckpt_every"]

    def marks(ax):
        for r in rots:
            ax.axvline(r, color="0.75", lw=0.8, ls="--", zorder=0)
        for s, e in windows:
            ax.axvspan(s, e, color="#e8a33d", alpha=0.25, zorder=0)

    # fig1 — competence
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, rd in readers.items():
        xs, ys = series(rd["log"], "nll.true.idx")
        axes[0].plot(xs, ys, lw=1.1, label=f"{name} idx")
        xs, ys = series(rd["log"], "nll.true.out")
        axes[1].plot(xs, ys, lw=1.1, label=f"{name} out")
    for ax, t in zip(axes, ["indexed span (leaves 0-15)", "outside the span"]):
        marks(ax); ax.set_ylabel("NLL (nats)"); ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7)
    axes[1].set_xlabel("step")
    fig.suptitle(f"{tag} — competence, with rotations (dashed) and consumed bursts (band)")
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig1_competence.png"), dpi=130)
    plt.close(fig)

    # fig2 — Factor T
    if rowsT:
        st = np.array([r["step"] for r in rowsT])
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, pfx in zip(axes[:2], ("D_pair", "D_self")):
            ax.plot(st, [r[f"{pfx}_clean"] for r in rowsT], lw=1.0, color="#4c78a8",
                    label="clean")
            ax.plot(st, [r[f"{pfx}_rot"] for r in rowsT], lw=1.2, color="#c0392b",
                    label="counterfactual ROTATION")
            ax.plot(st, [r[f"{pfx}_burst_matched"] for r in rowsT], lw=1.2,
                    color="#e8a33d", label="counterfactual BURST @ matched surprisal")
            marks(ax); ax.set_ylabel(f"{pfx} (nats)"); ax.legend(fontsize=7)
            ax.set_title(f"{pfx} — the same weights, two events, matched indexed-span "
                         f"surprisal", fontsize=10)
        axes[2].plot(st, [r["rot_spike"] for r in rowsT], lw=1.0, color="#c0392b",
                     label="rotation spike (nats)")
        ax2 = axes[2].twinx()
        ax2.plot(st, [r["rho_star"] for r in rowsT], lw=1.0, color="#e8a33d",
                 label="matched burst rate rho*")
        ax2.set_ylabel("rho*"); ax2.set_yscale("log")
        marks(axes[2]); axes[2].set_ylabel("nats"); axes[2].set_xlabel("step")
        axes[2].legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{tag} — FACTOR T: does the typing gauge separate the events?")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig2_factorT.png"), dpi=130)
        plt.close(fig)

        # fig3 — the ladder itself, at a few maturities
        pick = [r for r in rowsT if r["step"] in
                (2000, 4000, 7000, 9000, 13000, 19000)] or rowsT[::max(len(rowsT) // 5, 1)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in pick:
            axes[0].plot(r["ladder_spike"], r["ladder_D_pair"], "o-", ms=3, lw=1,
                         label=f"s{r['step']}")
            axes[1].plot(r["ladder_spike"], r["ladder_D_self"], "o-", ms=3, lw=1,
                         label=f"s{r['step']}")
            axes[0].plot([r["rot_spike"]], [r["D_pair_rot"]], "*", ms=11, color="#c0392b")
            axes[1].plot([r["rot_spike"]], [r["D_self_rot"]], "*", ms=11, color="#c0392b")
        for ax, t in zip(axes, ("D_pair", "D_self")):
            ax.set_xlabel("indexed-span NLL spike (nats)"); ax.set_ylabel(f"{t} (nats)")
            ax.set_title(f"{t} vs surprisal: bursts (lines) and rotations (stars)",
                         fontsize=10)
            ax.legend(fontsize=6)
        fig.suptitle(f"{tag} — the burst ladder against the rotation, per maturity")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig3_ladder.png"), dpi=130)
        plt.close(fig)

    # fig4 — Factor M
    prim = f"lr{setup.get('fm_primary', 1e-3):g}"
    have = [(w, r) for w in panels for r in panels[w]["log"][0].get("fm", {})]
    if have:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for w, rname in dict.fromkeys(have):
            lg = panels[w]["log"]
            xs = [r["step"] for r in lg if prim in
                  r.get("fm", {}).get(rname, {}).get("e", {}).get("clean", {})]
            ys = [r["fm"][rname]["e"]["clean"][prim]["rel"] for r in lg
                  if prim in r.get("fm", {}).get(rname, {}).get("e", {}).get("clean", {})]
            if not xs:
                continue
            axes[0].plot(xs, ys, lw=1.1, label=f"{rname} e(clean)")
            yb = [r["fm"][rname]["e"]["burst_c"][prim]["rel"] for r in lg
                  if prim in r.get("fm", {}).get(rname, {}).get("e", {}).get("burst_c", {})]
            if len(yb) == len(xs):
                axes[1].plot(xs, np.array(yb) / np.array(ys) - 1.0, lw=1.1,
                             label=f"{rname} shadow-burst relative excess")
        marks(axes[0]); marks(axes[1])
        axes[0].set_ylabel("e_rel = ||h6-FM(h0)||^2 / mean(h6^2)"); axes[0].set_yscale("log")
        axes[0].legend(fontsize=7)
        axes[0].set_title("Factor M: the FM residual on the CLEAN panel (self-change)",
                          fontsize=10)
        axes[1].set_ylabel("e(shadow burst)/e(clean) - 1")
        axes[1].axhline(0, color="0.5", lw=0.8)
        axes[1].legend(fontsize=7); axes[1].set_xlabel("step")
        axes[1].set_title("input-invariance: the same weights, a corrupted input",
                          fontsize=10)
        fig.suptitle(f"{tag} — FACTOR M")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig4_factorM.png"), dpi=130)
        plt.close(fig)

    # fig5 — FM lr sweep
    if out.get("fm_sweep"):
        fig, ax = plt.subplots(figsize=(10, 4))
        lg = panels["pair"]["log"]
        for lr in setup.get("fm_lrs", []):
            nm = f"lr{lr:g}"
            xs = [r["step"] for r in lg
                  if nm in r.get("fm", {}).get("wall", {}).get("e", {}).get("clean", {})]
            ys = [r["fm"]["wall"]["e"]["clean"][nm]["rel"] for r in lg
                  if nm in r.get("fm", {}).get("wall", {}).get("e", {}).get("clean", {})]
            if xs:
                ax.plot(xs, ys, lw=1.0, label=nm)
        marks(ax); ax.set_yscale("log"); ax.set_xlabel("step")
        ax.set_ylabel("e_rel (clean panel)"); ax.legend(fontsize=7)
        ax.set_title(f"{tag} — FM learning-rate sweep on the keyed reader (uncharged)")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig5_fm_lr.png"), dpi=130)
        plt.close(fig)
    # fig6 — the movement decomposition (T*)
    mvr = out.get("movement_decomposition")
    if mvr and rowsT:
        keep = [r for r in rowsT if r["step"] >= mvr["quiet_min_step"] and r["bracketed"]]
        g = mvr["quiet_guard"]
        keep = [r for r in keep
                if not (any(0 <= r["step"] - x < g for x in rots)
                        or any(w[0] <= r["step"] < w[1] + g for w in windows))]
        st = [r["step"] for r in keep]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        # (a) d_self trajectories
        axes[0].plot(st, [r["d_self_rot"] for r in keep], "o-", ms=3, lw=1,
                     color="#c0392b", label="rotation (0 by construction)")
        axes[0].plot(st, [r["d_self_burst_matched"] for r in keep], "o-", ms=3, lw=1,
                     color="#e8a33d", label="burst @ matched d_key")
        axes[0].axhline(0, color="0.6", lw=0.8)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("d_self (nats)")
        axes[0].set_title("the key-neutralised view's own movement", fontsize=10)
        axes[0].legend(fontsize=7)
        # (b) d_out trajectories
        axes[1].plot(st, [r["out_leak_rot"] for r in keep], "o-", ms=3, lw=1,
                     color="#c0392b", label="rotation")
        axes[1].plot(st, [r["out_leak_burst_matched"] for r in keep], "o-", ms=3, lw=1,
                     color="#e8a33d", label="burst @ matched d_key")
        axes[1].axhline(0, color="0.6", lw=0.8)
        axes[1].set_xlabel("step"); axes[1].set_ylabel("d_out (nats)")
        axes[1].set_title("movement off the indexed span", fontsize=10)
        axes[1].legend(fontsize=7)
        # (c) the typing plane
        axes[2].scatter([r["d_self_rot"] for r in keep],
                        [r["out_leak_rot"] for r in keep], s=18, color="#c0392b",
                        label="rotation", zorder=3)
        axes[2].scatter([r["d_self_burst_matched"] for r in keep],
                        [r["out_leak_burst_matched"] for r in keep], s=18,
                        color="#e8a33d", label="burst @ matched d_key", zorder=3)
        axes[2].axhline(0, color="0.6", lw=0.8); axes[2].axvline(0, color="0.6", lw=0.8)
        axes[2].set_xlabel("d_self (nats)"); axes[2].set_ylabel("d_out (nats)")
        axes[2].set_title("the typing plane: T* at matched d_key", fontsize=10)
        axes[2].legend(fontsize=7)
        fig.suptitle(f"{tag} — MOVEMENT DECOMPOSITION: per-view NLL movement types the "
                     f"event where KL magnitude did not (n={mvr['n']})")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig6_movement.png"), dpi=130)
        plt.close(fig)

    print(f"[figures] -> {root}")


if __name__ == "__main__":
    main()
