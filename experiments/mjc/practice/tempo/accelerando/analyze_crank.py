"""Reduce accelerando's Phase 3 run (the crank under conductor's thermostat) to a text report and
figures.

WHAT THIS RUN IS, AND IS NOT — stated at the top of every reduction it writes. A table here is
per tempo (decision 31) and the program is never executed (decision 28), so **a tempo advance is
a change of piece, not a level crossing**: the committed level resets and has to be re-earned by
mining at the new tempo. `c1` therefore measures whether one thermostat paces commit/hold
*inside* a tempo at least as well as a schedule — it does not measure whether a ratchet survives
the era knob, because there is nothing crossing the era boundary to survive.

Pure python for the reduction; `--figures` renders remotely on Modal.

    python3 mjc/practice/tempo/accelerando/analyze_crank.py --tag c1 --fetch
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from mjc.shared import app, volume, DATA_DIR   # noqa: E402

VOL = "mujoco-control-data"
SCOPE = ("SCOPE: a table is per tempo and the program is never executed, so a tempo advance is a "
         "CHANGE OF PIECE, not a level crossing. This run measures within-tempo pacing under one "
         "thermostat; it is not a test of a tempo ratchet.")


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOL, f"/practice_accelerando/{tag}/crank.json",
                    os.path.join(dst, "crank.json"), "--force"], check=True)
    return os.path.join(dst, "crank.json")


def report(d, out):
    def P(s=""):
        print(s)
        out.append(s)

    cfg = d["config"]
    T = cfg["tempos"]
    L = int(cfg["n_levels"])
    arms = d["arms"]
    band = d["piece_by_tempo"][str(T[0])]["band"]

    P("=" * 100)
    P(f"accelerando Phase 3 — tag {cfg['tag']}   seed {cfg['seed']}   "
      f"wall {d.get('wall_s', 0):.0f}s   complete={d.get('complete')}")
    P("=" * 100)
    P(SCOPE)
    P(f"tempi {T}   levels {L}   Delta fixed {cfg['delta_fix']}   cycles {cfg['n_cycles']}   "
      f"mine_support {cfg['mine_support']}   band {band:.4f}")
    P()

    P("--- GATES " + "-" * 88)
    P(f"P-F0  donor constants: {d['P_F0']['checked']} checked, "
      f"{len(d['P_F0']['mismatches'])} mismatched   pass={d['P_F0']['pass']}")
    p1 = d.get("P_T1", {})
    P(f"P-T1  level-1 vocabulary reproduces t1: {p1.get('n_checks')} checks, max|delta| = "
      f"{p1.get('max_abs'):.3e}   applicable={p1.get('applicable')}   pass={p1.get('pass')}")
    pp = d.get("P_P", {})
    P(f"P-P   conductor's own offline policy gate: {pp.get('n_checks')} checks   "
      f"pass={pp.get('pass')}")
    pd = d.get("P_D", {})
    P(f"P-D   dead zones MEASURED by null-ABBA on this node's anchor series: "
      + json.dumps({k: round(v, 6) for k, v in pd.get("v_tol", {}).items()})
      + f"   pass={pd.get('pass')}")
    for k, v in pd.get("floors", {}).items():
        P(f"        {k:<7s} windows {v['n_windows']}  sd(N) {v['sd_N']}  sd(D) {v['sd_D']}  "
          f"sd(N)/sd(D) {v['ratio']}  v_tol {v['v_tol']}")
    if pd.get("degenerate"):
        P(f"      DEGENERATE reads (floor exactly 0, NOT substituted): {pd['degenerate']}")
        P(f"      {pd.get('note_degenerate', '')}")
    py = d.get("P_Y", {})
    P(f"P-Y   yoked arms bit-identical to their gauge arms over {len(py.get('rows', []))} "
      f"series: max|delta| = {py.get('max_abs'):.3e}   pass={py.get('pass')}")
    P()

    P("--- THE ACTION TRACES " + "-" * 77)
    P("      arm            | cycles | commits | advances | capped | ended     | tables at end")
    for nm, r in arms.items():
        if r.get("floor_degenerate"):
            P(f"      {nm:<15s}| NOT RUN — its read's measured dead zone is 0 "
              f"(the gauge never moved above its own noise in the anchor series)")
            continue
        nc = sum(1 for x in r["actions"] if x["kind"] == "commit")
        na = sum(1 for x in r["actions"] if x["kind"] == "advance")
        P(f"      {nm:<15s}|  {len(r['cycles']):4d}  |   {nc:3d}   |   {na:4d}   |  {r['n_capped']:3d}   "
          f"| H={r['final_tempo']:<3d} L={r['final_level']} | "
          + " ".join(f"H{h}:{v.get('L2', 0)}/{v.get('L3', 0)}/{v.get('L4', 0)}"
                     for h, v in sorted(r["tables"].items(), key=lambda kv: -int(kv[0]))))
    P("      (tables column: mined slots at L2/L3/L4 per tempo; L1 is the shared vocabulary)")
    P()
    P("      action cycles, in order:")
    for nm, r in arms.items():
        if r.get("floor_degenerate"):
            continue
        P(f"        {nm:<15s} " + "  ".join(
            f"c{x['cycle']}:{x['kind'][:3]}"
            + (f"->H{x['H']}" if x["kind"] == "advance" else f"L{x['level']}(+{x['n_added']})")
            + ("*" if x.get("capped") else "") for x in r["actions"]))
    P("      (* = the action was forced by the cap, not by the rule)")
    P()

    P("--- THE THREE CLOCKS: what each arm was doing, per cycle " + "-" * 42)
    for nm, r in arms.items():
        if r.get("floor_degenerate"):
            continue
        cy = r["cycles"]
        P(f"      {nm}")
        for fld, lab in (("H", "tempo   "), ("level", "level   "), ("at_support", "at_supp "),
                         ("e_piece", "e_piece "), ("solved_frac", "solved  ")):
            v = [c[fld] for c in cy]
            step = max(1, len(v) // 20)
            P(f"        {lab}" + " ".join(
                (f"{x:6.0f}" if fld in ("H", "level", "at_support") else f"{x:6.3f}")
                for x in v[::step]))
    P()

    P("--- THE ADDRESS-BOOK BATTERY, at each arm's final tempo " + "-" * 43)
    P("      arm            | H  | table        | per-level e_piece (feedback events) | reflex")
    for row in d.get("battery", []):
        lv = [f"L{e}={row[f'L{e}_e']:.4f}({row[f'L{e}_fb']:.0f})"
              for e in range(1, L + 1) if f"L{e}_e" in row]
        P(f"      {row['arm']:<15s}| {row['H']:2d} | "
          f"{str(row['tables']):<12s} | " + "  ".join(lv)
          + f" | {row['reflex_e']:.4f}({row['reflex_fb']:.0f})")
        if "decompose_gap" in row:
            P(f"        can't-decompose at L2: L1 - L2 = {row['decompose_gap']:+.4f} at a "
              f"feedback saving of {row['decompose_fb']:.0f} events "
              f"(execution is identical by P-N; what the level changes is WHEN the second half "
              f"was chosen)")
    P()
    P("--- the program instrument (logged, never executed) " + "-" * 47)
    for r in d.get("program_instrument", [])[:12]:
        P(f"      {r['arm']:<15s} advance at c{r['at_cycle']}: H{r['from_H']} -> H{r['to_H']} "
          f"with level {r['level']} committed")
    P(f"      {d.get('flags', {}).get('content_is_recordings', '')}")
    P()
    P("--- timings " + "-" * 87)
    P("      " + "  ".join(f"{k}={v:.0f}s" for k, v in d.items()
                           if k.startswith("t_") and isinstance(v, (int, float))))
    return out


@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def make_crank_figures(tag: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(DATA_DIR, "practice_accelerando", tag)
    figdir = os.path.join(root, "figs")
    os.makedirs(figdir, exist_ok=True)
    d = json.load(open(os.path.join(root, "crank.json")))
    arms = {k: v for k, v in d["arms"].items() if not v.get("floor_degenerate")}
    made = []
    CL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name), dpi=140)
        plt.close(fig)
        made.append(name)

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for i, (fld, lab) in enumerate((("H", "tempo H (steps/note)"), ("level", "committed level"),
                                    ("at_support", "at_support (one level up)"),
                                    ("e_piece", "metering error (m)"))):
        for j, (nm, r) in enumerate(sorted(arms.items())):
            cy = r["cycles"]
            ax[i].plot([c["cycle"] for c in cy], [c[fld] for c in cy], "-",
                       color=CL[j % len(CL)], lw=1.4, label=nm if i == 0 else None)
            for x in r["actions"]:
                ax[i].axvline(x["cycle"], color=CL[j % len(CL)], lw=0.4, alpha=0.25)
        ax[i].set_ylabel(lab)
        if fld == "e_piece":
            ax[i].axhline(d["piece_by_tempo"][str(d["config"]["tempos"][0])]["band"],
                          color="#888", lw=1.0)
            ax[i].set_yscale("log")
        if i == 0:
            ax[i].legend(fontsize=8, ncol=3)
    ax[-1].set_xlabel("cycle")
    save(fig, "fig1_clocks.png")

    volume.commit()
    return made


@app.local_entrypoint()
def accelerando_crank_figs(tag: str = "c1"):
    print(f"[figs] {make_crank_figures.remote(tag)}")


def figures(tag):
    here = os.path.relpath(os.path.abspath(__file__),
                           os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    subprocess.run(["modal", "run", f"{here}::accelerando_crank_figs", "--tag", tag], check=True,
                   cwd=root)
    dst = os.path.join(HERE, "figures", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOL,
                    f"/practice_accelerando/{tag}/figs", dst], check=True)
    nest = os.path.join(dst, "figs")
    if os.path.isdir(nest):
        for f in os.listdir(nest):
            os.replace(os.path.join(nest, f), os.path.join(dst, f))
        os.rmdir(nest)
    print(f"[figures] -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    path = (fetch(args.tag) if args.fetch
            else os.path.join(HERE, "results", args.tag, "crank.json"))
    if not os.path.exists(path):
        path = fetch(args.tag)
    d = json.load(open(path))
    out = []
    report(d, out)
    rp = os.path.join(HERE, "results", args.tag, "reduction.txt")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    open(rp, "w").write("\n".join(out) + "\n")
    print(f"\n[written] {rp}")
    if args.figures:
        figures(args.tag)


if __name__ == "__main__":
    main()
