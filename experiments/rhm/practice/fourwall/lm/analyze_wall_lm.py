"""Reduce a fourwall/lm run.

    python3 rhm/practice/fourwall/lm/analyze_wall_lm.py --tag fwlm0 --fetch --figures

Sections, in the order the question asks for them:
  0. GATE 0 (exact, model-free) — what a right index is worth here, per span, and the BP
     probe ceilings under both conditions. Also the keyed latent's demand distribution.
  1. THE NOISE FLOOR — `dead_wall` vs `dead_wall_b` (identical config, different model /
     sampler seed) on every headline instrument, plus a PLACEBO rotation response taken at
     non-rotation checkpoint pairs. Nothing below these is read.
  2. ADMISSIBILITY — the control arms must not move across a rotation, and the wall must
     not move the UNINDEXED span (the within-sequence control).
  3. PHASE 1 — does the free key get taken? Binding in nats against the exact bracket
     (`index capture`), and when it forms.
  4. THE DEBT — per-rotation response of task error, binding, and stale attachment.
  5. THE INDEX — forced transfer, the reader's own implied routing table, and the MDL /
     cardinality readout: merge vs re-key vs stale, read out rather than assigned.
  6. STEP OR SMEAR — per-latent re-addressing across each rotation. The §9 discriminator.
  7. TYPED NEXT-LEVEL READOUT — per-level probe recovery at both anchors, wall present vs
     wall randomised, against the exact BP ceilings; paired with the probe-free
     excess-over-Bayes twin, and the agreement between them gated before either is read.
  8. THE IDENTITY RETURN — the one rotation that puts the phase-1 map back in force.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_fourwall_lm"
LEVELS = ["d1", "d2", "d3", "d4", "d5", "d6"]           # d1 shallowest, d6 root


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name == "setup.json":
            continue
        r = json.load(open(os.path.join(root, name)))
        arms[r["arm"]] = r
    order = [a for a in setup["arms"] if a in arms]
    return setup, {a: arms[a] for a in order}


def _dig(d, parts):
    """Walk a dotted path, greedily -- some leaf keys contain dots themselves
    (`card.tau0.001`), so try the longest join that resolves first."""
    if not parts:
        return d
    if not isinstance(d, dict):
        return None
    for k in range(len(parts), 0, -1):
        key = ".".join(parts[:k])
        if key in d:
            r = _dig(d[key], parts[k:])
            if r is not None:
                return r
    return None


def series(rec, path):
    """`path` is a dotted key path into a checkpoint record; missing -> nan."""
    out = []
    for r in rec["log"]:
        cur = _dig(r, path.split("."))
        out.append(np.nan if cur is None else float(cur))
    return np.array(out)


def steps(rec):
    return np.array([r["step"] for r in rec["log"]])


def at(rec, path, step):
    s, y = steps(rec), series(rec, path)
    i = np.argmin(np.abs(s - step))
    return y[i]


def window(rec, path, lo, hi):
    s, y = steps(rec), series(rec, path)
    msk = (s >= lo) & (s <= hi) & ~np.isnan(y)
    return y[msk]


def event_response(rec, path, ev, back=1, fwd=1):
    """(mean of the `fwd` checkpoints after `ev`) - (mean of the `back` before) — `../`'s
    rotation-response instrument, on the LM clock."""
    s, y = steps(rec), series(rec, path)
    pre = [y[i] for i in range(len(s)) if s[i] < ev][-back:]
    post = [y[i] for i in range(len(s)) if s[i] >= ev][:fwd]
    if not pre or not post:
        return np.nan
    return float(np.nanmean(post) - np.nanmean(pre))


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref-tag", default=None,
                    help="a round-1 tag whose never-merge arms serve as references "
                         "(licensed only if the bit-identity check below passes)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
        if a.ref_tag:
            fetch(a.ref_tag)
    setup, arms = load(a.tag)
    cfg, refs = setup["config"], setup["refs"]
    rots = setup["rotations"]
    # the schedule lives per-arm; `wall` is the treatment and defines the run-level clock
    sc = setup["scheds"]
    ref_arm = "wall" if "wall" in sc else next(
        (k for k, s_ in sc.items() if s_["rot_period"] > 0), list(sc)[0])
    cfg.setdefault("phase1", sc[ref_arm]["phase1"])
    cfg.setdefault("rot_period", max(1, sc[ref_arm]["rot_period"]))
    ev_ref = rots.get(ref_arm, [])

    def era_of(t, arm):
        s_ = sc[arm]
        if s_["rot_period"] <= 0 or t < s_["phase1"]:
            return 0
        return (t - s_["phase1"]) // s_["rot_period"] + 1

    def term_of(arm):
        """The last checkpoint that is genuinely INSIDE its era. The final step of the
        run lands exactly on an (unrun) era boundary for the rotating arms, so its
        record is a 0-gradient-step post-rotation snapshot, not a terminal state."""
        s_ = steps(arms[arm])
        ok = [int(t) for t in s_ if t > 0
              and era_of(int(t), arm) == era_of(max(0, int(t) - cfg["ckpt_every"]), arm)]
        return max(ok) if ok else int(s_[-1])

    term = {arm: term_of(arm) for arm in arms}
    bracket = refs["gate0"]["index_value_indexed_span"]
    W = 78

    print("=" * W)
    print(f"fourwall/lm  tag={a.tag}   arms: {list(arms)}")
    print(f"  {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D  "
          f"v{cfg['v']}/s{cfg['s']}/L{cfg['depth']}/m{cfg['m']}  key = level "
          f"{cfg['key_level']} node {cfg['key_node']}  span leaves "
          f"[{refs['key_lo']},{refs['key_hi']})")
    print(f"  {cfg['max_steps']} steps = "
          f"{cfg['max_steps'] * cfg['batch_size'] * refs['T']:,} tokens; phase 1 = "
          f"{cfg['phase1']}; rotations {ev_ref}")

    # ---------------- 0. Gate 0 ----------------
    print("\n" + "=" * W + "\n0. GATE 0 — the exact oracle bracket (model-free)\n" + "=" * W)
    for k, val in refs["gate0"].items():
        if isinstance(val, (int, float)):
            print(f"  {k:30s} {val:+.4f}")
    print(f"  probe ceilings, KEY anchor  (prefix = the indexed span)")
    print(f"    no wall {refs['ceil']['key_none']}")
    print(f"    wall    {refs['ceil']['key_wall']}")
    print(f"  probe ceilings, LAST anchor (prefix = the whole sequence)")
    print(f"    no wall {refs['ceil']['last_none']}")

    # ---------------- 1. the noise floor ----------------
    print("\n" + "=" * W + "\n1. NOISE FLOOR\n" + "=" * W)
    HEAD = {"nll idx": "nll.true.idx", "nll out": "nll.true.out",
            "binding idx": "binding.idx", "stale idx": "binding.stale_idx",
            "transfer gap": "index.transfer_gap", "jsd": "index.jsd_mean",
            "map match cur": "index.map_match_cur"}
    floor = {}
    if "dead_wall" in arms and "dead_wall_b" in arms:
        print("  seed pair (dead_wall vs dead_wall_b, identical config, seeds 42/43),")
        print("  |delta| over the last quarter of the run:")
        lo = int(0.75 * cfg["max_steps"])
        for lab, p in HEAD.items():
            x = window(arms["dead_wall"], p, lo, cfg["max_steps"])
            y = window(arms["dead_wall_b"], p, lo, cfg["max_steps"])
            n = min(len(x), len(y))
            d = float(np.abs(x[:n] - y[:n]).mean()) if n else np.nan
            floor[lab] = d
            print(f"    {lab:16s} {d:.5f}   (levels {np.nanmean(x):.4f} / "
                  f"{np.nanmean(y):.4f})")
    print("\n  placebo rotation response (same instrument, at NON-rotation checkpoints):")
    placebo = {}
    for lab, p in HEAD.items():
        vals = []
        for arm, rec in arms.items():
            s = steps(rec)
            evs = set(rots.get(arm, []))
            cand = [int(t) for t in s if t > cfg["max_steps"] * 0.4
                    and not any(abs(t - e) <= 2 * cfg["ckpt_every"] for e in evs)]
            vals += [abs(event_response(rec, p, e)) for e in cand[::3]]
        vals = [x for x in vals if np.isfinite(x)]
        placebo[lab] = float(np.nanpercentile(vals, 90)) if vals else np.nan
        print(f"    {lab:16s} p90 |delta| {placebo[lab]:.5f}  (n={len(vals)})")

    # ---------------- 2. admissibility ----------------
    print("\n" + "=" * W + "\n2. ADMISSIBILITY\n" + "=" * W)
    print("  (a) control arms must not respond to a rotation. Response of `nll idx` at the")
    print("      `wall` arm's rotation steps, every arm:")
    ev_list = ev_ref
    for arm, rec in arms.items():
        r = [event_response(rec, "nll.true.idx", e) for e in ev_list]
        print(f"    {arm:14s} " + " ".join(f"{x:+.4f}" for x in r)
              + f"   |mean| {np.nanmean(np.abs(r)):.4f}")
    print("\n  (b) the wall must not touch the UNINDEXED span (within-sequence control).")
    print("      binding, indexed vs outside, at the end of phase 1:")
    for arm, rec in arms.items():
        bi = at(rec, "binding.idx", cfg["phase1"] - cfg["ckpt_every"])
        bo = at(rec, "binding.out", cfg["phase1"] - cfg["ckpt_every"])
        gi = at(rec, "index.transfer_gap", cfg["phase1"] - cfg["ckpt_every"])
        go = at(rec, "index.out_span_transfer_gap", cfg["phase1"] - cfg["ckpt_every"])
        print(f"    {arm:14s} binding {bi:+.4f} / {bo:+.4f}   "
              f"transfer gap {gi:+.4f} / {go:+.4f}")

    # ---------------- 3. phase 1: is the free key taken? ----------------
    print("\n" + "=" * W + "\n3. PHASE 1 — is the free key taken, and when?\n" + "=" * W)
    print(f"  exact bracket on the indexed span: {bracket:.4f} nats")
    print(f"  {'arm':14s} {'nll idx':>9s} {'nll out':>9s} {'binding':>9s} "
          f"{'capture':>8s} {'misleading':>11s} {'gap':>8s}")
    p1end = cfg["phase1"] - cfg["ckpt_every"]
    for arm, rec in arms.items():
        b = at(rec, "binding.idx", p1end)
        print(f"  {arm:14s} {at(rec, 'nll.true.idx', p1end):9.4f} "
              f"{at(rec, 'nll.true.out', p1end):9.4f} {b:+9.4f} "
              f"{b / bracket:8.3f} {at(rec, 'binding.misleading_idx', p1end):+11.4f} "
              f"{at(rec, 'index.transfer_gap', p1end):+8.4f}")
    print("\n  ORACLE CAPTURE — the arm's own indexed-span NLL against `no_wall`'s, as a")
    print("  fraction of the exact bracket. (`binding` above compares a right key against a")
    print("  WRONG one, which for a committed reader is a bigger number than the bracket.)")
    if "no_wall" in arms:
        for g in (2000, 4000, 6000, p1end):
            base = at(arms["no_wall"], "nll.true.idx", g)
            row = "  ".join(f"{arm}={(base - at(rec, 'nll.true.idx', g)) / bracket:+.3f}"
                            for arm, rec in arms.items() if arm != "no_wall")
            print(f"    step {g:6d}  {row}")
    print("\n  binding trajectory through phase 1 (nll rand - nll true, / exact bracket):")
    grid = [x for x in (250, 1000, 2000, 4000, 6000, p1end) if x <= cfg["max_steps"]]
    print(f"  {'arm':14s}" + "".join(f"{g:>9d}" for g in grid))
    for arm, rec in arms.items():
        print(f"  {arm:14s}"
              + "".join(f"{at(rec, 'binding.idx', g) / bracket:9.3f}" for g in grid))

    # ---------------- 4. the debt ----------------
    print("\n" + "=" * W + "\n4. THE DEBT — per-rotation response\n" + "=" * W)
    for lab, p in (("nll idx", "nll.true.idx"), ("binding idx", "binding.idx"),
                   ("stale idx", "binding.stale_idx"),
                   ("map match cur", "index.map_match_cur")):
        print(f"\n  {lab} (post - pre, one column per rotation "
              f"{ev_ref}):")
        for arm, rec in arms.items():
            r = [event_response(rec, p, e) for e in ev_list]
            print(f"    {arm:14s} " + " ".join(f"{x:+.4f}" for x in r)
                  + f"   |mean| {np.nanmean(np.abs(r)):.4f}"
                  + (f"   [floor {floor.get(lab, np.nan):.4f}]" if lab in floor else ""))

    print("\n  RECOVERY after each rotation — steps until `binding idx` is back to 90% of")
    print("  its immediately-pre-rotation level, and until the implied map is fully re-keyed:")
    for arm in ("wall", "wall_fast"):
        if arm not in arms:
            continue
        rec = arms[arm]
        s = steps(rec); b = series(rec, "binding.idx"); mm = series(rec, "index.map_match_cur")
        for e in rots.get(arm, [])[:8]:
            pre = [b[i] for i in range(len(s)) if s[i] < e]
            if not pre:
                continue
            targ = 0.9 * pre[-1]
            post = [(int(s[i]), b[i], mm[i]) for i in range(len(s))
                    if e <= s[i] <= e + (cfg["rot_period"] if arm == "wall" else 1000)]
            t90 = next((t - e for t, y, _ in post if y >= targ), None)
            tmap = next((t - e for t, _, y in post if y >= 0.99), None)
            print(f"    {arm:10s} rot {e:6d}  pre {pre[-1]:+.4f}  "
                  f"t(90% binding) {str(t90):>6s}  t(map fully re-keyed) {str(tmap):>6s}")

    # ---------------- 5. the index state ----------------
    print("\n" + "=" * W + "\n5. THE INDEX — merge vs re-key vs stale, read out\n" + "=" * W)
    marks = sorted({0, cfg["phase1"] - cfg["ckpt_every"]}
                   | {e + cfg["ckpt_every"] for e in ev_list} | {cfg["max_steps"]})
    marks = [x for x in marks if x <= cfg["max_steps"]]
    marks = [x for x in marks if x < cfg["max_steps"]]
    print(f"  (the final step {cfg['max_steps']} lands on an unrun era boundary for the")
    print("   rotating arms, so the terminal column is each arm's last IN-ERA checkpoint:")
    print("   " + "  ".join(f"{a}={term[a]}" for a in arms) + ")")
    for lab, p in (("transfer gap", "index.transfer_gap"),
                   ("map match cur", "index.map_match_cur"),
                   ("map match old", "index.map_match_old"),
                   ("map match ph1", "index.map_match_phase1"),
                   ("jsd (behavioural spread)", "index.jsd_mean"),
                   ("n classes tau=0.0002", "index.card.tau0.0002"),
                   ("n classes tau=0.001", "index.card.tau0.001"),
                   ("n classes tau=0.005", "index.card.tau0.005"),
                   ("nll idx", "nll.true.idx"),
                   ("E: right wall (current map)", "index.e_cur"),
                   ("E: right wall (PREVIOUS map)", "index.e_old"),
                   ("E: wrong wall (mean)", "index.e_off"),
                   ("E: best available wall", "index.e_best")):
        print(f"\n  {lab}:")
        print(f"    {'arm':14s}" + "".join(f"{g:>8d}" for g in marks) + f"{'term':>9s}")
        for arm, rec in arms.items():
            print(f"    {arm:14s}" + "".join(f"{at(rec, p, g):8.3f}" for g in marks)
                  + f"{at(rec, p, term[arm]):9.3f}")

    # ---------------- 6. step or smear ----------------
    print("\n" + "=" * W + "\n6. STEP OR SMEAR — per-latent re-addressing\n" + "=" * W)
    print("  For each rotation: the number of live latents whose implied wall matches the")
    print("  NEW map, at +0/+50/+100/+250/+500/+1000/+2000 steps. A single discrete index")
    print("  event moves them together; a smear moves them one at a time.")
    for arm in ("wall", "true_wall", "wall_fast"):
        if arm not in arms:
            continue
        rec = arms[arm]
        s = steps(rec)
        print(f"\n  {arm}:")
        print("      rot     +0  +50 +100 +250 +500 +750+1000+1500 (live latents on the NEW map)")
        for e in rots.get(arm, [])[:8]:
            row = []
            for d in (0, 50, 100, 250, 500, 750, 1000, 1500):
                i = np.argmin(np.abs(s - (e + d)))
                if (abs(s[i] - (e + d)) > cfg["ckpt_every"] // 2 + 1
                        or era_of(int(s[i]), arm) != era_of(e, arm)):
                    row.append("  - ")
                    continue
                mv = rec["log"][i]["index"]["match_cur_per_latent"]
                row.append(f"{int(np.sum(mv)):3d} ")
            print(f"    step {e:6d}  " + "".join(row))

    # ---------------- 7. typed next-level readout ----------------
    print("\n" + "=" * W + "\n7. NEXT-LEVEL EXTRACTION — probe / probe-free pair\n" + "=" * W)
    pk = [i for i, r in enumerate(arms[list(arms)[0]]["log"]) if "levels" in r]
    pk_steps = [arms[list(arms)[0]]["log"][i]["step"] for i in pk]
    print(f"  probe checkpoints: {pk_steps}")
    def probe_at(rec, step):
        cands = [r for r in rec["log"] if "levels" in r]
        return min(cands, key=lambda r: abs(r["step"] - step)) if cands else None

    # in-era probe checkpoints: late phase 1, mature era 1, mature era 4 (identity return)
    PSTEPS = [(6000, "late phase 1"), (9000, "era 1, +1000 steps"),
              (14250, "era 4 = identity return, +250 steps"),
              (cfg["max_steps"], "final step (0 steps into a fresh rotation)")]
    for anchor, ceilkey in (("key", "key_none"), ("last", "last_none")):
        print(f"\n  --- anchor '{anchor}' (exact ceilings, no wall: "
              + " ".join(f"{d}={refs['ceil'][ceilkey][d]:.3f}" for d in LEVELS) + ")")
        for st, lab in PSTEPS:
            print(f"   @ step {st} ({lab})")
            for cond in ("true", "rand"):
                print(f"     wall {cond}:")
                for arm, rec in arms.items():
                    pr = probe_at(rec, st)
                    if pr is None:
                        continue
                    r = pr["levels"][cond][anchor]
                    print(f"      {arm:14s} "
                          + " ".join(f"{d}={r[d]:.3f}" for d in LEVELS))
    print("\n  THE PAIR GATE — probe binding (d4 true - d4 rand at the key anchor) against")
    print("  probe-free binding (nll rand - nll true on the indexed span), at each arm's")
    print("  last IN-ERA probe checkpoint:")
    pb, fb = [], []
    for arm, rec in arms.items():
        cands = [r for r in rec["log"] if "levels" in r
                 and era_of(r["step"], arm) == era_of(max(0, r["step"] - 250), arm)]
        if not cands:
            continue
        pr = cands[-1]
        d = pr["levels"]["true"]["key"]["d4"] - pr["levels"]["rand"]["key"]["d4"]
        n = at(rec, "binding.idx", pr["step"])
        pb.append(d); fb.append(n)
        print(f"    {arm:14s} @s{pr['step']:6d}  probe {d:+.3f}   "
              f"probe-free {n:+.4f} nats")
    if len(pb) > 2:
        rk = np.corrcoef(np.argsort(np.argsort(pb)), np.argsort(np.argsort(fb)))[0, 1]
        print(f"    rank correlation of the two instruments: {rk:+.3f}")
    print("\n  probe-free excess over exact Bayes, by arrival level, at each arm's last")
    print("  IN-ERA checkpoint (level 0 = the token closes the root):")
    for cond, key in (("wall true", "excess_by_level"),
                      ("wall randomised", "excess_by_level_rand")):
        print(f"   {cond}:")
        for arm, rec in arms.items():
            r = min(rec["log"], key=lambda z: abs(z["step"] - term[arm]))
            e = r[key]
            print(f"    {arm:14s} " + " ".join(f"L{k}:{e[k]:+.3f}"
                                               for k in sorted(e, key=int)))

    # ---------------- 8. the identity return ----------------
    print("\n" + "=" * W + "\n8. THE IDENTITY RETURN\n" + "=" * W)
    ident = [e for e in ev_list
             if (((e - cfg["phase1"]) // cfg["rot_period"] + 1) * cfg["rot_step"]) % cfg["v"] == 0]
    print(f"  the map returns to the phase-1 identity at steps {ident}")
    for e in ident:
        print(f"\n  rotation at {e}:")
        for arm, rec in arms.items():
            print(f"    {arm:14s} nll idx {event_response(rec, 'nll.true.idx', e):+.4f}  "
                  f"binding {event_response(rec, 'binding.idx', e):+.4f}  "
                  f"map->ph1 {at(rec, 'index.map_match_phase1', e + cfg['ckpt_every']):.3f}")
    print("\n  for comparison, the mean |response| of nll idx over NON-identity rotations:")
    for arm, rec in arms.items():
        r = [event_response(rec, "nll.true.idx", e) for e in ev_list if e not in ident]
        print(f"    {arm:14s} {np.nanmean(np.abs(r)):.4f}")

    # ---------------- 9. the merge op (round 2) ----------------
    mg = {arm: sc[arm].get("merge_at", cfg["max_steps"] + 1) for arm in arms}
    if any(m <= cfg["max_steps"] for m in mg.values()):
        ref_setup, ref_arms = (load(a.ref_tag) if a.ref_tag else (None, {}))
        merge_report(setup, arms, refs, cfg, mg, term, era_of, ref_arms, W)

    if a.figures:
        figures(a.tag, setup, arms, refs, cfg, rots, ref_arm)
        if any(m <= cfg["max_steps"] for m in mg.values()):
            merge_figures(a.tag, arms, refs, cfg, mg,
                          load(a.ref_tag)[1] if a.ref_tag else {})
    print("\n" + "=" * W)


# --------------------------------------------------------------------------- #
# round 2 — the merge op
# --------------------------------------------------------------------------- #

def merge_report(setup, arms, refs, cfg, mg, term, era_of, ref_arms, W):
    bracket = refs["gate0"]["index_value_indexed_span"]
    ceil_d4 = refs["ceil"]["key_none"]["d4"]
    MS = cfg["max_steps"]
    merge_arms = [k for k, m in mg.items() if m <= MS]
    base = arms.get("no_wall") or ref_arms.get("no_wall")

    print("\n" + "=" * W + "\n9. THE MERGE OP — supplied exogenously, priced\n" + "=" * W)
    print("  The op replaces the wall token with the neutral filler from `merge_at` on.")
    print("  It touches no parameter; after the op an arm's batches are token-for-token")
    print(f"  identical to `no_wall`'s. merge steps: "
          + "  ".join(f"{k}={mg[k]}" for k in merge_arms))

    # --- 9a. bit-identity: the licensing condition for reading post-merge divergence ---
    print("\n  9a. BIT-IDENTITY against the round-1 twin (pre-merge prefix).")
    if ref_arms:
        for arm in merge_arms + [k for k in ("no_wall",) if k in arms and k in ref_arms]:
            twin = ("no_wall" if arms[arm]["sched"]["kind"] == "none" else
                    "dead_wall" if arms[arm]["sched"]["kind"] == "dead" else
                    "wall_fast" if arms[arm]["sched"]["rot_period"] == 250 else "wall")
            if twin not in ref_arms:
                print(f"    {arm:16s} twin `{twin}` absent from the reference tag")
                continue
            m = mg[arm]
            worst, n = 0.0, 0
            for r in arms[arm]["log"]:
                if r["step"] >= m:
                    break
                q = [z for z in ref_arms[twin]["log"] if z["step"] == r["step"]]
                if not q:
                    continue
                n += 1
                for p in ("nll.true.idx", "nll.true.out", "binding.idx",
                          "index.transfer_gap", "index.jsd_mean"):
                    x = _dig(r, p.split(".")); y = _dig(q[0], p.split("."))
                    if x is not None and y is not None:
                        worst = max(worst, abs(float(x) - float(y)))
            print(f"    {arm:16s} vs {twin:10s} max|delta| over {n:3d} shared "
                  f"pre-merge checkpoints: {worst:.3e}")
    else:
        print("    (no --ref-tag given; cross-tag references are not licensed)")

    # --- 9b. admissibility of the op ---
    print("\n  9b. ADMISSIBILITY of the op — it must delete an INDEX, not shock the model.")
    print("      (i) the token-swap control `merge_dead_8000` deletes zero index content;")
    print("          whatever transient it shows is the price of the swap itself.")
    print("      (ii) the op must not move the UNINDEXED span beyond the floor.")
    if base is not None:
        print(f"    {'arm':16s} {'M':>6s} "
              + "".join(f"{'M+' + str(d):>9s}" for d in (0, 125, 250, 500, 1000, 2000)))
        for lab, path in (("idx vs no_wall", "nll.none.idx"),
                          ("out vs no_wall", "nll.none.out")):
            print(f"   {lab}:")
            for arm in merge_arms:
                m = mg[arm]
                row = "".join(
                    f"{at(arms[arm], path, m + d) - at(base, path, m + d):+9.4f}"
                    for d in (0, 125, 250, 500, 1000, 2000))
                print(f"    {arm:16s} {m:6d} {row}")

    # --- 9c. attribution: does the token-derived pathway climb? ---
    print("\n  9c. ATTRIBUTION — the key-anchor d4 probe under the arm's CONSUMED condition")
    print(f"      (post-merge = `none`, pre-merge = `true`), exact ceiling {ceil_d4:.3f}.")
    print("      The probe-free twin is the indexed-span excess over exact Bayes.")
    grid = sorted({0, 1000, 2000, 4000, 6000, 8000, 9000, 10000, 12000, 13000,
                   14000, 15000, 17000, MS})
    grid = [g for g in grid if g <= MS]

    def probe_cons(rec, step):
        cands = [r for r in rec["log"] if "levels" in r]
        if not cands:
            return None
        r = min(cands, key=lambda z: abs(z["step"] - step))
        if abs(r["step"] - step) > 750:
            return None
        cond = r.get("consumed", "true")
        cond = cond if cond in r["levels"] else "true"
        return r["levels"][cond]["key"]["d4"], r["step"]

    print(f"    {'arm':16s}" + "".join(f"{g:>8d}" for g in grid))
    for arm in list(arms):
        row = ""
        for g in grid:
            pc = probe_cons(arms[arm], g)
            row += "     -  " if pc is None else f"{pc[0]:8.3f}"
        print(f"    {arm:16s}{row}")
    for arm, rec in ref_arms.items():
        if arm in ("wall", "true_wall", "wall_fast", "dead_wall"):
            row = ""
            for g in grid:
                pc = probe_cons(rec, g)
                row += "     -  " if pc is None else f"{pc[0]:8.3f}"
            print(f"    [r1] {arm:11s}{row}")
    print("\n    probe-free twin — indexed-span excess over exact Bayes, consumed condition:")
    print(f"    {'arm':16s}" + "".join(f"{g:>8d}" for g in grid))
    for arm, rec in list(arms.items()) + [(f"[r1] {k}", v) for k, v in ref_arms.items()
                                          if k in ("wall", "true_wall", "wall_fast")]:
        row = ""
        for g in grid:
            r = min(rec["log"], key=lambda z: abs(z["step"] - g))
            cond = r.get("consumed", "true")
            row += f"{r['excess'][cond]['idx']:8.3f}"
        print(f"    {arm:16s}{row}")

    # --- 9d. the lifetime trade ---
    print("\n  9d. THE LIFETIME TRADE — indexed-span NLL under the consumed condition.")
    print("      `terminal` = last in-era checkpoint; `lifetime` = mean over all")
    print("      checkpoints; `post-op` = mean from the arm's merge step onward.")
    print(f"    {'arm':16s} {'terminal':>9s} {'vs no_wall':>11s} {'lifetime':>9s} "
          f"{'post-op':>9s} {'d4 final':>9s}")
    def term_for(rec):
        """A reference arm carries its own schedule, so its terminal checkpoint must be
        computed from THAT, not from this tag's clock."""
        s_ = rec["sched"]
        def e_(t):
            if s_["rot_period"] <= 0 or t < s_["phase1"]:
                return 0
            return (t - s_["phase1"]) // s_["rot_period"] + 1
        ok = [r["step"] for r in rec["log"] if r["step"] > 0
              and e_(r["step"]) == e_(max(0, r["step"] - cfg["ckpt_every"]))]
        return max(ok) if ok else rec["log"][-1]["step"]

    for arm, rec in list(arms.items()) + [(f"[r1] {k}", v) for k, v in ref_arms.items()
                                          if k in ("wall", "true_wall", "wall_fast",
                                                   "dead_wall")]:
        key = arm.replace("[r1] ", "")
        tstep = term.get(key) if key in term else term_for(rec)
        cons_at = lambda r: r["nll"][r.get("consumed", "true")]["idx"]
        rr = min(rec["log"], key=lambda z: abs(z["step"] - tstep))
        life = float(np.mean([cons_at(r) for r in rec["log"]]))
        m = mg.get(key, MS + 1)
        post = [cons_at(r) for r in rec["log"] if r["step"] >= min(m, MS)]
        pc = probe_cons(rec, MS)
        bt = (cons_at(min(base["log"], key=lambda z: abs(z["step"] - tstep)))
              if base is not None else np.nan)
        print(f"    {arm:16s} {cons_at(rr):9.4f} {cons_at(rr) - bt:+11.4f} "
              f"{life:9.4f} {np.mean(post) if post else np.nan:9.4f} "
              + ("     -  " if pc is None else f"{pc[0]:9.3f}"))

    # --- 9e. what survives of the abandoned index ---
    print("\n  9e. THE ABANDONED INDEX — residual wall circuitry after the op.")
    print("      `vs_none` = NLL(neutral) - NLL(right wall): what a wall would still buy.")
    print("      `gap` = forced-transfer gap; `jsd` = wall-conditional behavioural spread.")
    for lab, path in (("vs_none idx", "binding.vs_none_idx"),
                      ("transfer gap", "index.transfer_gap"),
                      ("jsd", "index.jsd_mean")):
        print(f"   {lab}:")
        print(f"    {'arm':16s} {'M':>6s} "
              + "".join(f"{'M+' + str(d):>9s}" for d in (0, 250, 1000, 4000, 8000)))
        for arm in merge_arms:
            m = mg[arm]
            row = "".join(f"{at(arms[arm], path, min(m + d, MS)):9.4f}"
                          for d in (0, 250, 1000, 4000, 8000))
            print(f"    {arm:16s} {m:6d} {row}")


def merge_figures(tag, arms, refs, cfg, mg, ref_arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    ceil_d4 = refs["ceil"]["key_none"]["d4"]

    def cons_series(rec):
        s = np.array([r["step"] for r in rec["log"]])
        y = np.array([r["nll"][r.get("consumed", "true")]["idx"] for r in rec["log"]])
        return s, y

    def d4_series(rec):
        pl = [r for r in rec["log"] if "levels" in r]
        s = [r["step"] for r in pl]
        y = [r["levels"][r.get("consumed", "true")
                         if r.get("consumed", "true") in r["levels"] else "true"]["key"]["d4"]
             for r in pl]
        return s, y

    allr = list(arms.items()) + [(f"r1:{k}", v) for k, v in ref_arms.items()
                                 if k in ("wall", "true_wall", "wall_fast")]
    f, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (arm, rec) in enumerate(allr):
        ls = "--" if arm.startswith("r1:") else "-"
        s, y = cons_series(rec); axs[0].plot(s, y, ls, color=cols[i % 10], lw=1.2, label=arm)
        s, y = d4_series(rec); axs[1].plot(s, y, ls, color=cols[i % 10], lw=1.2, label=arm)
    for k, m in mg.items():
        if m <= cfg["max_steps"]:
            for ax in axs:
                ax.axvline(m, color="k", ls=":", lw=0.7, alpha=0.5)
    axs[0].set_title("indexed-span NLL, consumed condition")
    axs[0].set_ylabel("NLL (nats)")
    axs[1].axhline(ceil_d4, color="gray", ls="-.", lw=1)
    axs[1].set_title("key-anchor d4, consumed condition (exact ceiling dashed)")
    for ax in axs:
        ax.set_xlabel("step"); ax.legend(fontsize=6)
    f.tight_layout(); f.savefig(os.path.join(out, "fig6_merge.png"), dpi=140)
    print(f"\n[figures] -> {os.path.join(out, 'fig6_merge.png')}")


# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, refs, cfg, rots, ref_arm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    ev = rots.get("wall", rots.get(ref_arm, []))
    bn = np.array(refs["bayes_none"]); bk = np.array(refs["bayes_key"])
    lo, hi = refs["key_lo"], refs["key_hi"]
    bracket = refs["gate0"]["index_value_indexed_span"]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    def marks(ax):
        ax.axvline(cfg["phase1"], color="k", ls="--", lw=0.8, alpha=0.6)
        for e in ev:
            ax.axvline(e, color="r", ls=":", lw=0.7, alpha=0.5)

    # fig1 — competence
    f, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, (arm, rec) in enumerate(arms.items()):
        axs[0].plot(steps(rec), series(rec, "nll.true.idx"), color=cols[i], label=arm, lw=1.2)
        axs[1].plot(steps(rec), series(rec, "nll.true.out"), color=cols[i], label=arm, lw=1.2)
    axs[0].axhline(bn[lo:hi].mean(), color="gray", ls="-.", lw=1, label="Bayes, no key")
    axs[0].axhline(bk[lo:hi].mean(), color="green", ls="-.", lw=1, label="Bayes, key given")
    axs[1].axhline(bn[hi:].mean(), color="gray", ls="-.", lw=1, label="Bayes")
    for ax, t in zip(axs, ("indexed span (leaves %d-%d)" % (lo, hi - 1), "unindexed span")):
        marks(ax); ax.set_title(t); ax.set_xlabel("step"); ax.set_ylabel("NLL (nats)")
        ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(os.path.join(out, "fig1_competence.png"), dpi=140)

    # fig2 — binding and the stale address
    f, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, (arm, rec) in enumerate(arms.items()):
        axs[0].plot(steps(rec), series(rec, "binding.idx") / bracket, color=cols[i],
                    label=arm, lw=1.2)
        axs[1].plot(steps(rec), series(rec, "binding.stale_idx"), color=cols[i],
                    label=arm, lw=1.2)
    axs[0].axhline(1.0, color="green", ls="-.", lw=1, label="exact bracket")
    axs[0].axhline(0.0, color="gray", lw=0.8)
    axs[1].axhline(0.0, color="gray", lw=0.8)
    axs[0].set_title("index capture = binding / exact bracket (indexed span)")
    axs[1].set_title("stale attachment: NLL(current map) - NLL(previous map)")
    for ax in axs:
        marks(ax); ax.set_xlabel("step"); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(os.path.join(out, "fig2_binding.png"), dpi=140)

    # fig3 — the index state
    f, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for i, (arm, rec) in enumerate(arms.items()):
        axs[0].plot(steps(rec), series(rec, "index.transfer_gap"), color=cols[i],
                    label=arm, lw=1.2)
        axs[1].plot(steps(rec), series(rec, "index.map_match_cur"), color=cols[i],
                    label=arm, lw=1.2)
        axs[2].plot(steps(rec), series(rec, "index.jsd_mean"), color=cols[i],
                    label=arm, lw=1.2)
    axs[0].set_title("forced transfer gap (wrong wall - right wall), nats")
    axs[1].set_title("implied routing table matches the CURRENT map")
    axs[2].set_title("wall-conditional behavioural spread (the MDL readout)")
    for ax in axs:
        marks(ax); ax.set_xlabel("step"); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(os.path.join(out, "fig3_index.png"), dpi=140)

    # fig4 — per-level extraction, both anchors, wall true vs randomised
    f, axs = plt.subplots(2, 2, figsize=(13, 8))
    for r_i, anchor in enumerate(("key", "last")):
        for c_i, cond in enumerate(("true", "rand")):
            ax = axs[r_i][c_i]
            for i, (arm, rec) in enumerate(arms.items()):
                pl = [r for r in rec["log"] if "levels" in r]
                if not pl:
                    continue
                xs = [r["step"] for r in pl]
                for j, d in enumerate(("d4", "d5", "d6")):
                    ax.plot(xs, [r["levels"][cond][anchor][d] for r in pl], color=cols[i],
                            ls=("-", "--", ":")[j], lw=1.1,
                            label=f"{arm} {d}" if r_i + c_i == 0 else None)
            ck = refs["ceil"]["key_none" if anchor == "key" else "last_none"]
            for j, d in enumerate(("d4", "d5", "d6")):
                ax.axhline(ck[d], color="gray", ls=("-", "--", ":")[j], lw=0.7, alpha=0.6)
            marks(ax); ax.set_title(f"anchor={anchor}  wall={cond}")
            ax.set_xlabel("step"); ax.set_ylabel("probe recovery")
    axs[0][0].legend(fontsize=5, ncol=2)
    f.tight_layout(); f.savefig(os.path.join(out, "fig4_levels.png"), dpi=140)

    # fig5 — per-latent re-addressing: discrete or smeared
    show = ([x for x in ("wall", "wall_fast", "true_wall", "dead_wall") if x in arms]
            or list(arms)[:4])
    f, axs = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.2), squeeze=False)
    for k, arm in enumerate(show):
        rec = arms[arm]
        M = np.array([r["index"]["match_cur_per_latent"] for r in rec["log"]]).T
        axs[0][k].imshow(M, aspect="auto", cmap="magma", interpolation="nearest",
                         extent=[steps(rec)[0], steps(rec)[-1], M.shape[0], 0])
        for e in ev:
            axs[0][k].axvline(e, color="c", ls=":", lw=0.8)
        axs[0][k].axvline(cfg["phase1"], color="w", ls="--", lw=0.8)
        axs[0][k].set_title(f"{arm}: latent -> right wall?")
        axs[0][k].set_xlabel("step"); axs[0][k].set_ylabel("keyed latent")
    f.tight_layout(); f.savefig(os.path.join(out, "fig5_perwall.png"), dpi=140)
    print(f"\n[figures] -> {out}")


if __name__ == "__main__":
    main()
