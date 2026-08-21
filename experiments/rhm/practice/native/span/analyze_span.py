"""Reduction for PORT 2 (`native/span`) — the corridor as a native primitive.

Reuses `ratchet/analyze_ratchet.py` verbatim for every ratchet-standard readout (per-era
competence and priced time, the earned-vs-given fraction, matched-priced-time margins, commit
events, the depth seam, unit-LP trajectories, compounding, the plant guard) by retargeting its
module-level volume prefix and figure root — nothing in `ratchet/` is modified.

On top of it, the readouts this port commits to:

  (0) THE GATES — twin bit-identity (max|de| before the port switches on) and the DP+infill
      parity trajectory per macro, with the cycle each slot's gate first opened.
  (1) THE ARM x ERA TABLE — e (last-5-cycle mean), priced time in the EXISTING ledger, and
      groundings / materialisations per solve, plus the explicitly-labelled counterfactual
      block-level ledger (a level-l DP execution charged its `span` blocks of infill, a
      span-head execution charged 1). The headline ledger is unchanged and comparable to
      `enum`; the counterfactual is printed beside it, never folded in.
  (2) THE PLANT-GUARD CONTRAST vs `teacher_slot/handle/` — clean-config `parse_acc` /
      `infill_acc`, treated minus in-tag twin, against handle/'s measured -0.094/-0.044/+0.008
      (dose-ordered) with infill unmoved. Here the span head is an OUTPUT path only, so
      `block_logits` is untouched and these numbers are already the weights-alone read.
  (3) THE L3 SHADOW AUDITION — handle/'s L3 audition was +0.024/+0.095/+0.047 worse under the
      handle in all three twin pairs; the same contrast, in-tag.
  (4) THE SPAN-HEAD-BYPASSED COMPETENCE TWIN (`e_nospan`) — what FIRING costs or buys,
      separable from what the head's gradient did to the trunk.

Usage (from experiments/):
    python3 rhm/practice/native/span/analyze_span.py --tag sp_s0 --fetch --figures
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RATCHET = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ratchet")
sys.path.insert(0, RATCHET)
import analyze_ratchet as AR                                          # noqa: E402

AR.FIG = FIG
AR.REMOTE = "rhm_practice_native_span"
AR.ORDER = ["never_base", "given", "span_true", "practice_late", "span_mined"]

TWIN = {"span_true": "given", "span_mined": "practice_late"}

# handle/ (teacher_slot, hr_s0, 2026-08-20): the measured negative this port is built against.
HANDLE = {
    "parse_d": [-0.094, -0.044, +0.008],        # dose-ordered by slot-embedding engagement
    "infill_d": "unmoved everywhere",
    "l3_aud_d": [+0.024, +0.095, +0.047],       # worse under the handle in all three pairs
    "floor": (0.01, 0.03),                      # stream-noise floor measured in-tag there
}


# --------------------------------------------------------------------------- #
# (0) the gates
# --------------------------------------------------------------------------- #

def twin_divergence(arms):
    """The comparability gate. A treated arm must track its twin EXACTLY until the port
    switches on. `span_mined` mints its head at its COMMIT cycle, so it has a real pre-switch
    window (the `handle/` standard, max|de| = 0.0 through it). `span_true` holds the true table
    from cycle 1, so its head exists and its loss touches the trunk from the first plant step —
    the window is empty by construction, exactly as `given_handle`'s was."""
    print("\n== (0a) TWIN BIT-IDENTITY GATE ==")
    out = {}
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        a = np.asarray(arms[arm]["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ev = [e["cycle"] for e in arms[arm]["events"] if e["kind"] == "commit"]
        sw = min(ev) if ev else 0                       # head minted at the commit cycle
        pre = np.abs(a[:sw] - b[:sw]) if sw else np.array([0.0])
        d = np.nonzero(np.abs(a - b) > 0)[0]
        out[arm] = {"switch_on": sw, "max_abs_pre": float(pre.max()) if pre.size else 0.0,
                    "first_divergent": int(d[0]) + 1 if len(d) else None,
                    "max_abs_overall": float(np.abs(a - b).max())}
        print(f"{arm:14s} vs {tw:16s} head minted c{sw or '-'}  "
              f"max|de| before switch-on = {out[arm]['max_abs_pre']:.6f}  "
              f"first divergent cycle = c{out[arm]['first_divergent']}  "
              f"max|de| overall = {out[arm]['max_abs_overall']:.4f}")
    print("\n-- in-tag noise floor: |de| between the two untreated arms' own era means "
          "is not defined (different arms); the floor here is the pre-switch-on window "
          "itself, which is exact.")
    return out


def parity_table(arms):
    """The parity gate, per macro slot: when it first opened, how it moved, and what fraction
    of the arm's macro executions the head ended up serving."""
    print("\n== (0b) DP+INFILL PARITY, PER MACRO (gate: held-out exact-match >= span_tau) ==")
    out = {}
    for arm, r in arms.items():
        if not r.get("span_mode"):
            continue
        cfg = r["config"]
        tau, mh = cfg["span_tau"], cfg["span_min_hold"]
        print(f"\n{arm}  (tau={tau}, min held-out={mh}, span_lam={cfg['span_lam']}, "
              f"span_lr={cfg['span_lr']}, span_batch={cfg['span_batch']})")
        for ev in r.get("slot_events", []):
            print(f"  minted L{ev['level']}: {ev['n_nodes']} nodes, "
                  f"{ev['n_entries']} entries, span {ev['span']} blocks")
        cyc = r["log"]["cycle"]
        sp = r["log"]["span"]
        keys = sorted({k for c in sp for k in c["parity"]},
                      key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
        cell = {}
        for k in keys:
            ser = [(c, x["parity"].get(k)) for c, x in zip(cyc, sp)]
            vals = [(c, y) for c, y in ser if y is not None]
            opens = [c for c, x in zip(cyc, sp) if x["open"].get(k)]
            first_open = min(opens) if opens else None
            tail = [y for _, y in vals[-5:]]
            cell[k] = {"first_measured": vals[0][0] if vals else None,
                       "first_open": first_open,
                       "n_cycles_open": len(opens),
                       "parity_end": float(np.mean(tail)) if tail else None,
                       "parity_max": max((y for _, y in vals), default=None),
                       "series": [(int(c), float(y)) for c, y in vals]}
            step = max(1, len(vals) // 10)
            print(f"  slot {k}: first measured c{cell[k]['first_measured']}  "
                  f"first OPEN c{first_open}  open on {len(opens)}/{len(cyc)} cycles  "
                  f"parity(end)={cell[k]['parity_end']}")
            if vals:
                print(f"           " + " ".join(f"c{c}:{y:.3f}" for c, y in vals[::step]))
        # what fraction of macro executions the head actually served
        blk = r["log"]["blocks"]
        dp = sum(b["mat_dp"] for b in blk)
        hd = sum(b["mat_head"] for b in blk)
        print(f"  macro executions (priced beams only): DP {dp}, head {hd}  "
              f"-> head served {hd / max(dp + hd, 1):.3f}")
        out[arm] = {"slots": cell, "mat_dp": dp, "mat_head": hd}
    return out


# --------------------------------------------------------------------------- #
# (1) the arm x era table, with the counterfactual block ledger said out loud
# --------------------------------------------------------------------------- #

def arm_era_table(arms, tail=5):
    print("\n== (1) ARM x ERA: e (last-5 mean), priced time (EXISTING ledger), "
          "groundings & materialisations per solve ==")
    print("   ledger: t = n_ground*d_fb + n_mat*c_mat, one materialisation per move "
          "application per row, whatever the span.")
    print(f"{'arm':14s} {'era':>3s} {'e':>7s} {'t_era':>10s} {'g/solve':>8s} {'m/solve':>8s} "
          f"{'mat_dp':>8s} {'mat_head':>9s} {'blk(cf)':>9s} {'cf/ledger':>10s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = AR.era_slices(log)
        e = np.asarray(log["e"], float)
        t = np.asarray(log["t_cum"], float)
        g = np.asarray(log["g_per_solve"], float)
        mps = np.asarray(log.get("m_per_solve") or [np.nan] * len(e), float)
        blk = log.get("blocks") or [{}] * len(e)
        out[arm] = {}
        for k, idx in sl.items():
            b = [blk[i] for i in idx]
            led = sum(x.get("mat_base", 0) + x.get("mat_dp", 0) + x.get("mat_head", 0)
                      for x in b)
            cf = sum(x.get("blk_base", 0) + x.get("blk_dp", 0) + x.get("blk_head", 0)
                     for x in b)
            row = {"e": float(e[idx[-tail:]].mean()),
                   "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                   "t_end": float(t[idx[-1]]),
                   "g_per_solve": float(g[idx[-tail:]].mean()),
                   "m_per_solve": float(np.nanmean(mps[idx[-tail:]])),
                   "mat_dp": sum(x.get("mat_dp", 0) for x in b),
                   "mat_head": sum(x.get("mat_head", 0) for x in b),
                   "ledger_mat": led, "cf_blocks": cf,
                   "cf_ratio": (cf / led) if led else float("nan")}
            out[arm][k] = row
            print(f"{arm:14s} {k:3d} {row['e']:7.4f} {row['t_era']:10.0f} "
                  f"{row['g_per_solve']:8.1f} {row['m_per_solve']:8.1f} "
                  f"{row['mat_dp']:8d} {row['mat_head']:9d} {row['cf_blocks']:9d} "
                  f"{row['cf_ratio']:10.3f}")
    print("\n   cf = the COUNTERFACTUAL block-level ledger, printed and never folded in: a "
          "level-l DP execution\n   charged its span = s**(l-1) blocks of infill, a base move "
          "1 block, a span-head execution 1.\n   cf/ledger < 1 for a treated arm is the "
          "SPEC's 'one-pass materialisation against s blocks' reading;\n   in the ledger the "
          "run is actually priced by, both cost the same and the port buys nothing on t.")
    return out


# --------------------------------------------------------------------------- #
# (2) the plant guard, against handle/
# --------------------------------------------------------------------------- #

def plant_contrast(arms, tail=3):
    print("\n== (2) PLANT GUARD on clean held-out configurations — the handle/ contrast ==")
    print("   (the span head is an OUTPUT path only, so `block_logits` is untouched and these "
          "are already\n    the weights-alone numbers handle/ had to reconstruct with its "
          "slots-bypassed column.)")
    print(f"{'arm':14s} {'parse c1':>9s} {'parse end':>10s} {'d':>8s} "
          f"{'infill c1':>10s} {'infill end':>11s} {'d':>8s} {'read end':>9s}")
    rows = {}
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        if not pr:
            continue
        cell = {}
        for k in ("parse_acc", "infill_acc", "read_acc", "e_nospan"):
            vals = [(p["cycle"], p["plant"][k]) for p in pr if k in p["plant"]]
            if not vals:
                continue
            cell[k] = {"first": float(vals[0][1]),
                       "last": float(np.mean([y for _, y in vals[-tail:]])),
                       "series": [(int(c), float(y)) for c, y in vals]}
        rows[arm] = cell
        pa, ia = cell["parse_acc"], cell["infill_acc"]
        print(f"{arm:14s} {pa['first']:9.4f} {pa['last']:10.4f} "
              f"{pa['last'] - pa['first']:+8.4f} {ia['first']:10.4f} {ia['last']:11.4f} "
              f"{ia['last'] - ia['first']:+8.4f} {cell['read_acc']['last']:9.4f}")

    print(f"\n-- treated minus in-tag twin (end-of-run means over the last {tail} probes) --")
    print(f"{'pair':34s} {'d parse':>9s} {'d infill':>9s}"
          f"    handle/: dparse {HANDLE['parse_d']}, infill {HANDLE['infill_d']}")
    out = {}
    for arm, tw in TWIN.items():
        if arm not in rows or tw not in rows:
            continue
        dp = rows[arm]["parse_acc"]["last"] - rows[tw]["parse_acc"]["last"]
        di = rows[arm]["infill_acc"]["last"] - rows[tw]["infill_acc"]["last"]
        out[arm] = {"d_parse": dp, "d_infill": di}
        print(f"{arm + ' - ' + tw:34s} {dp:+9.4f} {di:+9.4f}")
    print(f"   (handle/'s in-tag stream-noise floor was {HANDLE['floor'][0]}-"
          f"{HANDLE['floor'][1]}; its parse deltas ran 3-9x that floor.)")

    print("\n-- (4) the span-head-bypassed competence twin (e with the head switched off) --")
    for arm in TWIN:
        if arm not in rows or "e_nospan" not in rows[arm]:
            continue
        ser = rows[arm]["e_nospan"]["series"]
        step = max(1, len(ser) // 10)
        print(f"{arm:14s} e_nospan: " + " ".join(f"c{c}:{y:.3f}" for c, y in ser[::step]))
        log = arms[arm]["log"]
        cyc = {c: i for i, c in enumerate(log["cycle"])}
        pairs = [(c, log["e"][cyc[c]], y) for c, y in ser if c in cyc]
        if pairs:
            d = float(np.mean([a - b for _, a, b in pairs]))
            print(f"{'':14s} mean e(fire) - e(no fire) over probes where the head was "
                  f"firing: {d:+.4f}  (n={len(pairs)})")
    return rows, out


# --------------------------------------------------------------------------- #
# (3) the L3 shadow audition
# --------------------------------------------------------------------------- #

def audition_contrast(arms, tail=5):
    """handle/'s sharpest null-breaker: the level-3 shadow audition got WORSE under the handle
    in all three twin pairs (+0.024/+0.095/+0.047). Same contrast here, per era, on the
    candidate audition A and on the true-table audition A_true (a pure plant readout)."""
    print("\n== (3) SHADOW AUDITION, treated minus twin (handle/: L3 "
          f"{HANDLE['l3_aud_d']} = worse) ==")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        per[arm] = {}
        for k, idx in AR.era_slices(log).items():
            lvl = int(log["level"][idx[0]]) + 1
            cells = [log["aud"][i].get(str(lvl), {}) for i in idx[-tail:]]
            f = lambda key: [c[key] for c in cells if c.get(key) is not None]
            A, T = f("cand"), f("true")
            per[arm][k] = {"level": lvl,
                           "A": float(np.mean(A)) if A else float("nan"),
                           "A_true": float(np.mean(T)) if T else float("nan"),
                           "n_entries": float(np.mean([c.get("n_entries") or 0
                                                       for c in cells]))}
    print(f"{'pair':34s} {'era':>3s} {'lvl':>3s} {'dA':>8s} {'dA_true':>9s} {'d|T|':>7s}")
    out = {}
    for arm, tw in TWIN.items():
        if arm not in per or tw not in per:
            continue
        out[arm] = {}
        for k in sorted(per[arm]):
            a, b = per[arm][k], per[tw][k]
            out[arm][k] = {"level": a["level"], "dA": a["A"] - b["A"],
                           "dA_true": a["A_true"] - b["A_true"]}
            print(f"{arm + ' - ' + tw:34s} {k:3d} {a['level']:3d} "
                  f"{a['A'] - b['A']:+8.4f} {a['A_true'] - b['A_true']:+9.4f} "
                  f"{a['n_entries'] - b['n_entries']:+7.1f}")
    print("\n   dA_true is a PURE PLANT readout (the DGP's own table applied by the DP on fixed "
          "instances):\n   it moves only if the trunk moved. dA additionally carries the "
          "arm's own mined table.")
    return per, out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, per, plant, parity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(FIG, tag)
    os.makedirs(root, exist_ok=True)
    colour = {"never_base": "0.55", "given": "tab:green", "span_true": "darkgreen",
              "practice_late": "tab:blue", "span_mined": "tab:red"}
    style = {a: ("--" if a in TWIN else "-") for a in arms}
    ec = setup["config"]["era_cycles"]
    bounds = [ec * i for i in range(1, len(setup["eras"]))]

    # fig1 — competence and the twin contrast
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for arm, r in arms.items():
        ax[0].plot(r["log"]["cycle"], r["log"]["e"], style[arm], lw=1.4,
                   color=colour.get(arm), label=arm)
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        a = np.asarray(arms[arm]["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        ax[1].plot(arms[arm]["log"]["cycle"][:n], (a - b)[:n], lw=1.4,
                   color=colour.get(arm), label=f"{arm} - {tw}")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] == "commit":
                ax[0].axvline(ev["cycle"], color=colour.get(arm, "k"), lw=0.7, alpha=0.4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("e (1 - success), metering set")
    ax[0].set_title("competence per cycle (commits marked)")
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("e(span) - e(twin)")
    ax[1].set_title("the port contrast (0 before the head is minted = matched pair)")
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
        a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig1_span_competence.png"), dpi=150)
    plt.close(fig)

    # fig2 — the parity gate and what the head served
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for arm, cell in parity.items():
        for k, c in cell["slots"].items():
            if not c["series"]:
                continue
            xs, ys = zip(*c["series"])
            lvl = int(k.split(":")[0])
            ax[0].plot(xs, ys, "-" if lvl == 2 else "--", lw=1.2,
                       color=colour.get(arm), alpha=0.9 if lvl == 2 else 0.55,
                       label=f"{arm} {k}")
    tau = next((r["config"]["span_tau"] for r in arms.values() if r.get("span_mode")), 0.95)
    ax[0].axhline(tau, color="k", lw=0.9, ls=":")
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("held-out exact-match vs the DP")
    ax[0].set_title(f"parity gate per macro (dotted = tau={tau})"); ax[0].legend(fontsize=5)
    for arm, r in arms.items():
        if not r.get("span_mode"):
            continue
        blk = r["log"]["blocks"]
        tot = [x["mat_dp"] + x["mat_head"] for x in blk]
        frac = [x["mat_head"] / t if t else 0.0 for x, t in zip(blk, tot)]
        ax[1].plot(r["log"]["cycle"], frac, "-", lw=1.4, color=colour.get(arm), label=arm)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("fraction of macro executions served by the head")
    ax[1].set_title("firing"); ax[1].legend(fontsize=7)
    for arm, cell in plant.items():
        for key, ls in (("parse_acc", "-"), ("infill_acc", ":")):
            if key not in cell:
                continue
            xs, ys = zip(*cell[key]["series"])
            ax[2].plot(xs, ys, ls, lw=1.3, color=colour.get(arm),
                       label=f"{arm} {key}")
    ax[2].set_xlabel("cycle"); ax[2].set_ylabel("accuracy on clean held-out configs")
    ax[2].set_title("plant guard (solid parse, dotted infill)"); ax[2].legend(fontsize=5)
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig2_span_gate.png"), dpi=150)
    plt.close(fig)

    # fig3 — the shadow audition (the handle/ contrast) and the bypassed twin
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for arm, r in arms.items():
        log = r["log"]
        xs, ys = [], []
        for i, cyc in enumerate(log["cycle"]):
            lvl = int(log["level"][i]) + 1
            c = log["aud"][i].get(str(lvl), {})
            if c.get("true") is not None:
                xs.append(cyc); ys.append(c["true"])
        if xs:
            ax[0].plot(xs, ys, style[arm], lw=1.3, color=colour.get(arm), label=arm)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("A_true (the DGP's table, one action)")
    ax[0].set_title("the pure plant readout: true-table audition")
    for arm in TWIN:
        cell = plant.get(arm, {})
        if "e_nospan" not in cell:
            continue
        xs, ys = zip(*cell["e_nospan"]["series"])
        ax[1].plot(xs, ys, ":", lw=1.5, color=colour.get(arm), label=f"{arm} head OFF")
        log = arms[arm]["log"]
        ax[1].plot(log["cycle"], log["e"], "-", lw=1.2, color=colour.get(arm),
                   label=f"{arm} head ON")
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("e (metering set)")
    ax[1].set_title("the span-head-bypassed competence twin")
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
        a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig3_span_audition.png"), dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {root}/fig1_span_competence.png, fig2_span_gate.png, "
          f"fig3_span_audition.png")


def flap_table(arms):
    """The gate is re-checked every cycle rather than latched, so a slot whose parity sits near
    tau oscillates. Quantified here: transitions, how much of the parity series sits inside a
    +/-0.02 band around tau, and whether the oscillation shows up in competence — the per-cycle
    e-vs-twin delta split by whether the head fired that cycle."""
    print("\n== (0c) GATE FLAPPING — the cost of re-checking rather than latching ==")
    out = {}
    for arm, r in arms.items():
        if not r.get("span_mode"):
            continue
        tau = r["config"]["span_tau"]
        cyc, sp = r["log"]["cycle"], r["log"]["span"]
        keys = sorted({k for c in sp for k in c["parity"]},
                      key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
        print(f"\n{arm}  (tau={tau})")
        print(f"{'slot':6s} {'open':>7s} {'trans':>6s} {'|par-tau|<=.02':>15s} "
              f"{'par mean(meas)':>15s} {'par sd':>8s}")
        cell = {}
        for k in keys:
            op = [bool(c["open"].get(k)) for c in sp]
            vals = [c["parity"][k] for c in sp if c["parity"].get(k) is not None]
            trans = sum(1 for a, b in zip(op, op[1:]) if a != b)
            band = (sum(1 for y in vals if abs(y - tau) <= 0.02) / len(vals)) if vals else 0.0
            cell[k] = {"n_open": sum(op), "n_cycles": len(op), "transitions": trans,
                       "frac_open": sum(op) / len(op), "frac_in_band": band,
                       "parity_mean": float(np.mean(vals)) if vals else None,
                       "parity_sd": float(np.std(vals)) if vals else None}
            print(f"{k:6s} {sum(op):3d}/{len(op):3d} {trans:6d} {band:15.3f} "
                  f"{cell[k]['parity_mean'] or float('nan'):15.4f} "
                  f"{cell[k]['parity_sd'] or float('nan'):8.4f}")
        tw = TWIN.get(arm)
        if tw in arms:
            a = np.asarray(r["log"]["e"], float)
            b = np.asarray(arms[tw]["log"]["e"], float)
            n = min(len(a), len(b))
            fired = np.asarray([x["mat_head"] > 0 for x in r["log"]["blocks"]][:n], bool)
            d = (a - b)[:n]
            sw = min([e["cycle"] for e in r["events"] if e["kind"] == "commit"] or [0])
            post = np.zeros(n, bool)
            post[sw:] = True
            f, nf = post & fired, post & ~fired
            print(f"  e - e(twin), cycles after switch-on:  head FIRED "
                  f"{d[f].mean():+.4f} (n={int(f.sum())})   head IDLE "
                  f"{d[nf].mean() if nf.any() else float('nan'):+.4f} (n={int(nf.sum())})")
            cell["_d_fired"] = float(d[f].mean()) if f.any() else None
            cell["_d_idle"] = float(d[nf].mean()) if nf.any() else None
        out[arm] = cell
    return out


def fixed_level_audition(arms, level=3, tail=5):
    """handle/'s contrast was pinned to the LEVEL-3 audition, not to 'the era's active level'.
    Same here: the level-`level` candidate audition A and the true-table audition A_true, per
    era, treated minus twin."""
    print(f"\n== (3b) LEVEL-{level} AUDITION, per era, treated minus twin ==")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        per[arm] = {}
        for k, idx in AR.era_slices(log).items():
            cells = [log["aud"][i].get(str(level), {}) for i in idx[-tail:]]
            f = lambda key: [c[key] for c in cells if c.get(key) is not None]
            A, T = f("cand"), f("true")
            per[arm][k] = {"A": float(np.mean(A)) if A else float("nan"),
                           "A_true": float(np.mean(T)) if T else float("nan")}
    print(f"{'pair':34s} " + " ".join(f"{'era' + str(k):>20s}" for k in (1, 2, 3)))
    out = {}
    for arm, tw in TWIN.items():
        if arm not in per or tw not in per:
            continue
        cells, cellsT, row = [], [], {}
        for k in (1, 2, 3):
            if k not in per[arm]:
                cells.append(f"{'-':>20s}"); cellsT.append(f"{'-':>20s}"); continue
            dA = per[arm][k]["A"] - per[tw][k]["A"]
            dT = per[arm][k]["A_true"] - per[tw][k]["A_true"]
            row[k] = {"dA": dA, "dA_true": dT}
            cells.append(f"  {per[arm][k]['A']:.4f} ({dA:+.4f})")
            cellsT.append(f"  {per[arm][k]['A_true']:.4f} ({dT:+.4f})")
        out[arm] = row
        print(f"{arm + ' - ' + tw + ' [A]':34s} " + " ".join(cells))
        print(f"{'  [A_true, pure plant]':34s} " + " ".join(cellsT))
    print(f"   handle/'s L3 audition under the handle: {HANDLE['l3_aud_d']} "
          f"(positive = worse) in all three twin pairs.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        AR.fetch(a.tag)
    setup, arms, per = AR.report(a.tag, tail=a.tail)
    gates = twin_divergence(arms)
    parity = parity_table(arms)
    flap = flap_table(arms)
    ledger = arm_era_table(arms, tail=a.tail)
    plant, pcon = plant_contrast(arms)
    aud, acon = audition_contrast(arms, tail=a.tail)
    l3 = fixed_level_audition(arms, level=3, tail=a.tail)
    l2 = fixed_level_audition(arms, level=2, tail=a.tail)
    if a.figures:
        figures(a.tag, setup, arms, per, plant, parity)
    root = os.path.join(FIG, a.tag)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "summary.json"), "w") as fh:
        json.dump({"tag": a.tag, "gates": gates, "ledger": ledger,
                   "plant_contrast": pcon, "audition_contrast": acon,
                   "flap": flap, "audition_L3": l3, "audition_L2": l2,
                   "parity": {k: {"slots": {s: {kk: vv for kk, vv in c.items()
                                                if kk != "series"}
                                            for s, c in v["slots"].items()},
                                  "mat_dp": v["mat_dp"], "mat_head": v["mat_head"]}
                              for k, v in parity.items()},
                   "per_era": {arm: {str(k): v for k, v in cell.items()
                                     if not str(k).startswith("_")}
                               for arm, cell in per.items()}}, fh, indent=2)
    print(f"\nsummary -> {os.path.join(root, 'summary.json')}")


if __name__ == "__main__":
    main()
