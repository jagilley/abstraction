"""Reduction for the `handle` probe.

Reuses `ratchet/analyze_ratchet.py` verbatim for every ratchet-standard readout (per-era
competence and priced time, the earned-vs-given fraction, matched-priced-time margins, commit
events, the depth seam, unit-LP trajectories, compounding, the plant guard) by retargeting its
module-level volume prefix and figure root — nothing in `ratchet/` is modified.

On top of it, the three readouts the spec commits this node to:

  (1) CLEAN-CONFIG PARSE/INFILL — does the plant stop being inert? Reported twice per handle
      arm: conditioned (`parse_acc`, what the agent actually operates with) and slot-bypassed
      (`parse_acc_nc`, the weights alone). The bypassed number is the honest one, because a
      symbol supplied on a fully visible span determines that span's level-1 features exactly.
  (2) NEXT-LEVEL YIELD — the level-k+1 currency as the ratchet/merge instruments define it:
      |T_{k+1}| minability at the end of era k, the level-k+1 candidate audition A, and the
      table's recall/precision against the DGP's own vocabulary.
  (3) EARNED-VS-GIVEN FRACTION — (e_never - e_arm)/(e_never - e_given) per era.

Plus the handle's own diagnostics (did the earned symbol get learned at all?) and the twin
divergence check (a handle arm must be bit-identical to its no-handle twin until the commit).

Usage (from experiments/):
    python3 rhm/practice/teacher_slot/handle/analyze_handle.py --tag hr_s0 --fetch --figures
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
AR.REMOTE = "rhm_practice_teacher_slot_handle"
AR.ORDER = ["never_base", "given", "given_handle", "practice_gated", "practice_early",
            "practice_late", "practice_late_handle", "practice_late_handle_level"]

TWIN = {"given_handle": "given", "practice_late_handle": "practice_late",
        "practice_late_handle_level": "practice_late"}


# --------------------------------------------------------------------------- #
# (1) the plant
# --------------------------------------------------------------------------- #

def plant_table(arms, tail=3):
    """Clean-config parse/infill at the START and END of the run, per arm, with the
    slot-bypassed twin where it exists."""
    print("\n== (1) CLEAN-CONFIG PLANT READOUTS — is the plant still inert? ==")
    print(f"{'arm':28s} {'parse c1':>9s} {'parse end':>9s} {'d':>8s} "
          f"{'infill c1':>10s} {'infill end':>10s} {'d':>8s}")
    rows = {}
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        if not pr:
            continue
        cell = {}
        for k in ("parse_acc", "infill_acc", "parse_acc_nc", "infill_acc_nc"):
            vals = [p["plant"][k] for p in pr if k in p["plant"]]
            if not vals:
                continue
            cell[k] = {"first": float(vals[0]), "last": float(np.mean(vals[-tail:])),
                       "min": float(np.min(vals)), "max": float(np.max(vals)),
                       "series": [float(x) for x in vals],
                       "cycles": [p["cycle"] for p in pr][-len(vals):]}
        rows[arm] = cell
        pa, ia = cell["parse_acc"], cell["infill_acc"]
        print(f"{arm:28s} {pa['first']:9.4f} {pa['last']:9.4f} {pa['last']-pa['first']:+8.4f} "
              f"{ia['first']:10.4f} {ia['last']:10.4f} {ia['last']-ia['first']:+8.4f}")
        if "parse_acc_nc" in cell:
            pn, inn = cell["parse_acc_nc"], cell["infill_acc_nc"]
            print(f"{'  (slots bypassed)':28s} {pn['first']:9.4f} {pn['last']:9.4f} "
                  f"{pn['last']-pn['first']:+8.4f} {inn['first']:10.4f} {inn['last']:10.4f} "
                  f"{inn['last']-inn['first']:+8.4f}")

    print("\n-- handle vs its no-handle twin (end-of-run means over the last "
          f"{tail} probes) --")
    print(f"{'arm vs twin':46s} {'parse d':>9s} {'parse_nc d':>11s} {'infill d':>9s} "
          f"{'infill_nc d':>12s}")
    for arm, tw in TWIN.items():
        if arm not in rows or tw not in rows:
            continue
        a, b = rows[arm], rows[tw]
        print(f"{arm + ' - ' + tw:46s} "
              f"{a['parse_acc']['last'] - b['parse_acc']['last']:+9.4f} "
              f"{a.get('parse_acc_nc', a['parse_acc'])['last'] - b['parse_acc']['last']:+11.4f} "
              f"{a['infill_acc']['last'] - b['infill_acc']['last']:+9.4f} "
              f"{a.get('infill_acc_nc', a['infill_acc'])['last'] - b['infill_acc']['last']:+12.4f}")
    return rows


def handle_diag(arms):
    """Did the earned symbol get learned at all? If macro_acc never leaves its majority-class
    baseline the handle never engaged, and every null downstream is uninterpretable."""
    print("\n== the handle's own diagnostics — did the minted symbol get learned? ==")
    for arm, r in arms.items():
        if not r.get("handle_mode"):
            continue
        print(f"\n{arm}  (mode={r['handle_mode']})")
        for ev in r.get("slot_events", []):
            print(f"  slot L{ev['level']}: {ev['n_entries']} entries -> {ev['n_sym']} symbols "
                  f"(span {ev['span']} blocks, {ev['n_flat_unique']} unique rows, "
                  f"{ev['n_params']} params)")
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        for key in ("2", "3"):
            k = f"macro_acc_{key}"
            vals = [(p["cycle"], p["plant"][k], p["plant"][f"macro_base_{key}"],
                     p["plant"].get(f"slot_emb_norm_{key}", float("nan")))
                    for p in pr if k in p["plant"]]
            if not vals:
                continue
            step = max(1, len(vals) // 8)
            print(f"  L{key} macro_acc (vs majority-class base) on the masked span: "
                  + " ".join(f"c{c}:{a:.2f}/{b:.2f}" for c, a, b, _ in vals[::step]))
            print(f"  L{key} slot embedding norm: "
                  + " ".join(f"c{c}:{n:.3f}" for c, _, _, n in vals[::step]))
        ml = [(cy, h["macc"]) for cy, h in zip(r["log"]["cycle"], r["log"]["handle"])
              if h.get("macc")]
        if ml:
            step = max(1, len(ml) // 8)
            for key in ("2", "3"):
                sel = [(c, d[key]) for c, d in ml if key in d]
                if sel:
                    print(f"  L{key} train-time macro acc: "
                          + " ".join(f"c{c}:{a:.2f}" for c, a in sel[::step]))


def twin_divergence(arms):
    """The comparability gate: a handle arm must track its twin EXACTLY until the cycle its
    slot is minted, and only then diverge."""
    print("\n== twin divergence gate (must be 0.0 before the commit cycle) ==")
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        a = np.asarray(arms[arm]["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        commits = [ev["cycle"] for ev in arms[arm]["events"] if ev["kind"] == "commit"]
        first = min(commits) if commits else 0
        pre = np.abs(a[:first] - b[:first]) if first else np.array([0.0])
        d = np.nonzero(np.abs(a - b) > 0)[0]
        print(f"{arm:30s} vs {tw:18s} first commit c{first or '-'}  "
              f"max|d| pre-commit={float(pre.max()) if pre.size else 0.0:.6f}  "
              f"first divergent cycle=c{int(d[0]) + 1 if len(d) else None}  "
              f"max|d| overall={float(np.abs(a - b).max()):.4f}")


# --------------------------------------------------------------------------- #
# (2) next-level yield
# --------------------------------------------------------------------------- #

def yield_table(arms, tail=5):
    """The level-k+1 currency at the end of each era: how many entries the arm can MINE at the
    next level (|T_{k+1}|, merge round 1's minability), what that candidate scores (A), and how
    it grades against the DGP's own vocabulary."""
    print("\n== (2) NEXT-LEVEL YIELD — the level-k+1 currency at the end of each era ==")
    print(f"{'arm':28s} {'era':>3s} {'lvl':>3s} {'|T|':>5s} {'A':>7s} {'A_true':>7s} "
          f"{'recall':>7s} {'prec':>6s} {'rand_k':>7s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = AR.era_slices(log)
        out[arm] = {}
        for k, idx in sl.items():
            lvl = int(log["level"][idx[0]]) + 1
            cells = [log["aud"][i].get(str(lvl), {}) for i in idx[-tail:]]
            ne = [c.get("n_entries") for c in cells if c.get("n_entries") is not None]
            A = [c.get("cand") for c in cells if c.get("cand") is not None]
            T = [c.get("true") for c in cells if c.get("true") is not None]
            rc = [c.get("tab_recall") for c in cells if c.get("tab_recall") is not None]
            pc = [c.get("tab_precision") for c in cells if c.get("tab_precision") is not None]
            rk = [c.get("rand_k") for c in cells if c.get("rand_k") is not None]
            f = lambda xs: float(np.mean(xs)) if xs else float("nan")
            row = {"level": lvl, "n_entries": f(ne), "A": f(A), "A_true": f(T),
                   "recall": f(rc), "precision": f(pc), "rand_k": f(rk)}
            out[arm][k] = row
            print(f"{arm:28s} {k:3d} {lvl:3d} {row['n_entries']:5.1f} {row['A']:7.4f} "
                  f"{row['A_true']:7.4f} {row['recall']:7.3f} {row['precision']:6.3f} "
                  f"{row['rand_k']:7.4f}")
    print("\n-- handle minus its no-handle twin --")
    for arm, tw in TWIN.items():
        if arm not in out or tw not in out:
            continue
        for k in sorted(out[arm]):
            a, b = out[arm][k], out[tw][k]
            print(f"{arm + ' - ' + tw:46s} era{k} L{a['level']}  "
                  f"d|T|={a['n_entries'] - b['n_entries']:+6.1f}  "
                  f"dA={a['A'] - b['A']:+.4f}  drecall={a['recall'] - b['recall']:+.3f}  "
                  f"dprec={a['precision'] - b['precision']:+.3f}")
    return out


# --------------------------------------------------------------------------- #
# (3) earned-vs-given, against the stored ratchet run
# --------------------------------------------------------------------------- #

RR_S0 = {   # rr_s0 (ratchet, 2026-08-14): per-era e over each era's last 5 cycles
    "never_base": [0.349, 0.530, 0.642], "given": [0.201, 0.240, 0.341],
    "practice_gated": [0.223, 0.288, 0.438], "practice_early": [0.404, 0.564, 0.664],
    "practice_late": [0.275, 0.260, 0.347],
}


def against_rr_s0(per):
    """The no-handle arms here differ from `rr_s0` ONLY in the per-arm torch RNG stream (see
    handle.py's header), so this is a replicate of a node that has a single seed — and the
    spread is the noise floor any handle effect has to clear."""
    print("\n== no-handle arms vs ratchet `rr_s0` (same config, different torch stream) ==")
    print(f"{'arm':20s} " + " ".join(f"{'era' + str(k):>22s}" for k in (1, 2, 3)))
    for arm, ref in RR_S0.items():
        if arm not in per:
            continue
        cells = []
        for i, k in enumerate((1, 2, 3)):
            if k not in per[arm]:
                cells.append(f"{'-':>22s}")
                continue
            mine = per[arm][k]["e"]
            cells.append(f"  {mine:.4f} vs {ref[i]:.3f} ({mine - ref[i]:+.4f})")
        print(f"{arm:20s} " + " ".join(cells))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, per, plant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(FIG, tag)
    os.makedirs(root, exist_ok=True)
    colour = {"never_base": "0.55", "given": "tab:green", "given_handle": "darkgreen",
              "practice_late": "tab:blue", "practice_late_handle": "tab:red",
              "practice_late_handle_level": "tab:orange"}
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
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.8", lw=0.8, zorder=0)
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] == "commit":
                ax[0].axvline(ev["cycle"], color=colour.get(arm, "k"), lw=0.7, alpha=0.4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("e (1 - success), metering set")
    ax[0].set_title("competence per cycle (commits marked)")
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("e(handle) - e(twin)")
    ax[1].set_title("the handle contrast (0 before the commit = matched pair)")
    for a_ in ax:
        a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig1_handle_competence.png"), dpi=150)
    plt.close(fig)

    # fig2 — the plant
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, key in enumerate(("parse_acc", "infill_acc")):
        for arm, cell in plant.items():
            if key not in cell:
                continue
            ax[j].plot(cell[key]["cycles"], cell[key]["series"], "-", lw=1.4,
                       color=colour.get(arm), label=arm)
            nk = key + "_nc"
            if nk in cell:
                ax[j].plot(cell[nk]["cycles"], cell[nk]["series"], ":", lw=1.4,
                           color=colour.get(arm), label=f"{arm} (slots off)")
        ax[j].set_xlabel("cycle"); ax[j].set_ylabel(key)
        ax[j].set_title(f"plant guard: {key} on clean held-out configs")
        ax[j].legend(fontsize=6)
    for arm, r in arms.items():
        if not r.get("handle_mode"):
            continue
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        for key in ("2", "3"):
            vals = [(p["cycle"], p["plant"][f"macro_acc_{key}"]) for p in pr
                    if f"macro_acc_{key}" in p["plant"]]
            base = [(p["cycle"], p["plant"][f"macro_base_{key}"]) for p in pr
                    if f"macro_base_{key}" in p["plant"]]
            if not vals:
                continue
            ax[2].plot(*zip(*vals), "-", lw=1.4, color=colour.get(arm),
                       alpha=1.0 if key == "2" else 0.55, label=f"{arm} L{key}")
            ax[2].plot(*zip(*base), ":", lw=0.9, color=colour.get(arm), alpha=0.5)
    ax[2].set_xlabel("cycle"); ax[2].set_ylabel("macro-symbol accuracy (masked span)")
    ax[2].set_title("did the minted symbol get learned?\n(dotted = majority-class base)")
    ax[2].legend(fontsize=6)
    for a_ in ax:
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig2_handle_plant.png"), dpi=150)
    plt.close(fig)

    # fig3 — next-level yield
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for arm, r in arms.items():
        log = r["log"]
        for j, key in enumerate(("n_entries", "cand")):
            ys, xs = [], []
            for i, cyc in enumerate(log["cycle"]):
                lvl = int(log["level"][i]) + 1
                cell = log["aud"][i].get(str(lvl), {})
                if cell.get(key) is not None:
                    xs.append(cyc); ys.append(cell[key])
            if xs:
                ax[j].plot(xs, ys, style[arm], lw=1.3, color=colour.get(arm), label=arm)
    ax[0].set_ylabel("|T_{k+1}| minable entries"); ax[0].set_title("next-level yield: minability")
    ax[1].set_ylabel("A (level-k+1 candidate audition)")
    ax[1].set_title("next-level yield: what the candidate scores")
    for a_ in ax:
        a_.set_xlabel("cycle"); a_.legend(fontsize=7)
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig3_handle_yield.png"), dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {root}/fig1_handle_competence.png, fig2_handle_plant.png, "
          f"fig3_handle_yield.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hr_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        AR.fetch(a.tag)
    setup, arms, per = AR.report(a.tag, tail=a.tail)
    twin_divergence(arms)
    plant = plant_table(arms)
    handle_diag(arms)
    yield_table(arms, tail=a.tail)
    against_rr_s0(per)
    if a.figures:
        figures(a.tag, setup, arms, per, plant)
    with open(os.path.join(FIG, a.tag, "summary.json"), "w") as fh:
        json.dump({"tag": a.tag,
                   "per_era": {arm: {str(k): v for k, v in cell.items()
                                     if not str(k).startswith("_")}
                               for arm, cell in per.items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
