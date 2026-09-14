"""Fetch both conditions of the tiles twin and reduce them to figures + a numeric digest.

    cd experiments      # MODAL_PROFILE=chromatic, conda glp
    python3 canvas/plant/tiles_twin/analyze.py --fetch --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
VOL = "canvas-data"
CONDS = ("aligned", "misaligned")
COL = {"aligned": "#c0392b", "misaligned": "#2471a3", "lib0": "#7f8c8d"}
LEVELS = (1, 2, 3)


def fetch():
    os.makedirs(RES, exist_ok=True)
    want = []
    for c in CONDS:
        want += [(f"plant/twq_{c}/quant_stats.json", f"quant_twq_{c}.json"),
                 (f"plant/tw_{c}/run.json", f"run_tw_{c}.json"),
                 (f"plant/tw_{c}/oracle.json", f"oracle_tw_{c}.json")]
    for remote, local in want:
        dst = os.path.join(RES, local)
        if os.path.exists(dst):
            os.remove(dst)
        r = subprocess.run(["modal", "volume", "get", VOL, remote, dst],
                           capture_output=True, text=True,
                           env=dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic")))
        print(("ok   " if r.returncode == 0 else "MISS ") + remote)


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def digest():
    D = {c: load(f"run_tw_{c}.json") for c in CONDS}
    Q = {c: load(f"quant_twq_{c}.json") for c in CONDS}
    O = {c: load(f"oracle_tw_{c}.json") for c in CONDS}
    lib = load(os.path.join("..", "..", "results", "run_pl0.json")) or \
        (json.load(open(os.path.join(HERE, "..", "results", "run_pl0.json")))
         if os.path.exists(os.path.join(HERE, "..", "results", "run_pl0.json")) else None)
    libq = json.load(open(os.path.join(HERE, "..", "results", "quant_q0.json"))) \
        if os.path.exists(os.path.join(HERE, "..", "results", "quant_q0.json")) else None

    print("\n=== 1. quantizer floor (NMSE; 1.0 = the K=1 per-style floor) ===")
    print(f"{'K':>6}" + "".join(f"{c:>14}" for c in CONDS) + f"{'lib0':>14}")
    ks = sorted(Q[CONDS[0]]["per_k"], key=int) if Q[CONDS[0]] else []
    for k in ks:
        row = "".join(f"{Q[c]['per_k'][k]['nmse']:14.4f}" if Q[c] else f"{'-':>14}" for c in CONDS)
        row += f"{libq['per_k'][k]['nmse']:14.4f}" if libq and k in libq["per_k"] else f"{'-':>14}"
        print(f"{k:>6}" + row)

    print("\n=== 2. mask-size ladder, final held-out NLL (mean over 36 styles) ===")
    lad = D[CONDS[0]]["wave"]["ladder"] if D[CONDS[0]] else []
    print(f"{'cond':>12}" + "".join(f"{s*s:>8}" for s in lad))
    for c in CONDS:
        if D[c]:
            m = np.array(D[c]["plant_final"]["test_style"]).mean(0)
            print(f"{c:>12}" + "".join(f"{v:8.3f}" for v in m))
    if lib:
        print(f"{'lib0':>12}" + "".join(f"{v:8.3f}" for v in np.array(lib["plant_final"]["test_style"]).mean(0)))
    print("style id worth (nostyle - style):")
    for c in CONDS:
        if D[c]:
            a = np.array(D[c]["plant_final"]["val_nostyle"]).mean(0)
            b = np.array(D[c]["plant_final"]["val_style"]).mean(0)
            print(f"{c:>12}" + "".join(f"{v:8.3f}" for v in a - b))

    print("\n=== 3. recurrence ===")
    print(f"{'cond':>12}{'T2 d/obs':>10}{'T2 mass@s':>11}{'T3 pure':>9}{'|T2@s|':>8}")
    for c in CONDS:
        if not D[c]:
            continue
        r = D[c]["recurrence"]
        f = lambda k1, k2: float(np.mean([v[k1][k2] for v in r.values()]))
        print(f"{c:>12}{f('T2','distinct_per_obs'):10.3f}{f('T2','mass_at_support'):11.3f}"
              f"{f('T3','frac_obs_all_at_support'):9.3f}{f('T2','n_at_support'):8.0f}")
    if lib:
        r = lib["recurrence"]
        f = lambda k1, k2: float(np.mean([v[k1][k2] for v in r.values()]))
        print(f"{'lib0':>12}{f('T2','distinct_per_obs'):10.3f}{f('T2','mass_at_support'):11.3f}"
              f"{f('T3','frac_obs_all_at_support'):9.3f}{f('T2','n_at_support'):8.0f}")
    print("\nagainst the tile catalogue (oracle):")
    print(f"{'cond':>12}{'block->tile':>13}{'tile->block':>13}{'mass@s':>9}{'motif b->t':>12}{'n_mined_mot':>13}{'n_true_mot':>12}")
    for c in CONDS:
        if not O[c] or not O[c].get("tile_recovery"):
            print(f"{c:>12}{'(aligned only)':>13}")
            continue
        t = O[c]["tile_recovery"]
        f = lambda kk: float(np.nanmean([v[kk] for v in t.values()]))
        print(f"{c:>12}{f('block_to_tile_purity'):13.3f}{f('tile_to_block_purity'):13.3f}"
              f"{f('mass_at_support'):9.3f}{f('motif_block_to_tile_purity'):12.3f}"
              f"{f('n_mined_motifs_at_support'):13.0f}{f('n_true_motifs'):12.0f}")

    print("\n=== 4. cost-to-depth (exact code recovery) ===")
    dsc = load("describe.json")
    sides = sorted({v["side"] for v in D[CONDS[0]]["cost_to_depth"].values()}) if D[CONDS[0]] else []
    for nm in ("argmax1", "iter8", "beam8x4"):
        for c in CONDS:
            if D[c]:
                print(f"{c:>12} {nm:9s}" + "".join(f"{D[c]['cost_to_depth'][f'{s}_{nm}']['code_acc']:8.3f}" for s in sides))
    print(f"{'':>12} {'area':9s}" + "".join(f"{s*s:8d}" for s in sides))
    if dsc:
        print("DGP margins (oracle): repair1 " +
              str({k: round(float(np.mean([v[f'L{k}']['repair1'] for v in dsc.values()])), 3) for k in LEVELS})
              + "  greedy_ok " +
              str({k: round(float(np.mean([v[f'L{k}']['greedy_ok'] for v in dsc.values()])), 3) for k in LEVELS}))

    print("\n=== 5. grader against KNOWN validity (seam=invalid, offstyle=valid) ===")
    for c in CONDS:
        if not O[c] or "damage_grading" not in O[c]:
            continue
        for g, rows in O[c]["damage_grading"].items():
            print(f"-- {c} / {g} --")
            print(f"{'level':>7}" + "".join(f"{k:>26}" for k in ("clean", "seam", "offstyle", "plant_fill")))
            for lv in LEVELS:
                cells = []
                for kind in ("clean", "seam", "offstyle", "plant_fill"):
                    r = rows.get(f"{lv}_{kind}")
                    cells.append(f"{r['pass_one_sided']:.2f}/{r['pass_two_sided']:.2f} n={r['n']:<4}"
                                 if r else "-")
                print(f"{lv:>7}" + "".join(f"{x:>26}" for x in cells))
            print("        (one-sided / two-sided pass rate)")
    print("\n=== 6. the plant's fills, validated against the grammar ===")
    for c in CONDS:
        if O[c] and O[c].get("fill_validity_mean"):
            v = O[c]["fill_validity_mean"]
            print(f"{c:>12}  valid: fill {v['valid_fill']:.3f} | ceiling (clean roundtrip) "
                  f"{v['valid_clean_roundtrip']:.3f} | true {v['valid_true']:.3f}")
            print(f"{'':>12}  viol:  fill {v['viol_fill']:.2f}  | ceiling "
                  f"{v['viol_clean_roundtrip']:.2f} | true {v['viol_true']:.2f}")
            print(f"{'':>12}  tile classifier vs truth {v['classify_acc_vs_truth']:.3f} "
                  f"(in the hole {v['classify_acc_in_hole']:.3f})")
    return D, Q, O, lib, dsc


def figures(D, Q, O, lib, dsc):
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)

    # --- recurrence: the treatment ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    styles = D["aligned"]["styles"]
    xa = [D["aligned"]["recurrence"][s]["T2"]["mass_at_support"] for s in styles]
    xm = [D["misaligned"]["recurrence"][s]["T2"]["mass_at_support"] for s in styles]
    ax[0].scatter(xm, xa, s=26, color=COL["aligned"])
    ax[0].plot([0, 1], [0, 1], "k:", lw=.8)
    if lib:
        ml = float(np.mean([v["T2"]["mass_at_support"] for v in lib["recurrence"].values()]))
        ax[0].axhline(ml, color=COL["lib0"], ls="--", lw=1, label=f"lib0 mean {ml:.3f}")
        ax[0].legend(fontsize=7)
    ax[0].set_xlabel("misaligned: T[2] mass at support"); ax[0].set_ylabel("aligned")
    ax[0].set_title("does aligning the code grid make tuples recur?"); ax[0].grid(alpha=.3)
    for c in CONDS:
        v = sorted([D[c]["recurrence"][s]["T3"]["frac_obs_all_at_support"] for s in styles])
        ax[1].plot(np.linspace(0, 1, len(v)), v, "-o", ms=3, color=COL[c], label=c)
    if lib:
        v = sorted([x["T3"]["frac_obs_all_at_support"] for x in lib["recurrence"].values()])
        ax[1].plot(np.linspace(0, 1, len(v)), v, "-", color=COL["lib0"], label="lib0")
    ax[1].set_xlabel("style quantile"); ax[1].set_ylabel("T[3] occurrences with ALL halves at support")
    ax[1].set_title("does the ratchet constraint survive one level up?")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    tr = O["aligned"].get("tile_recovery") if O["aligned"] else None
    if tr:
        b = [tr[s]["block_to_tile_purity"] for s in tr]
        t = [tr[s]["tile_to_block_purity"] for s in tr]
        ax[2].scatter(t, b, s=26, color=COL["aligned"])
        ax[2].set_xlabel("tile -> block purity"); ax[2].set_ylabel("block -> tile purity")
        ax[2].set_title("aligned: did the codebook recover the tile?"); ax[2].grid(alpha=.3)
        ax[2].set_xlim(0, 1.02); ax[2].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "twin_recurrence.png"), dpi=140); plt.close(fig)

    # --- wave + cost-to-depth ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    lad = D["aligned"]["wave"]["ladder"]; area = [s * s for s in lad]
    for c in CONDS:
        ax[0].plot(area, np.array(D[c]["plant_final"]["test_style"]).mean(0), "-o", color=COL[c], label=c)
    if lib:
        ax[0].plot(area, np.array(lib["plant_final"]["test_style"]).mean(0), "-o", color=COL["lib0"], label="lib0")
    ax[0].set_xscale("log"); ax[0].set_xlabel("mask area (cells)"); ax[0].set_ylabel("held-out NLL / token")
    ax[0].set_title("the depth ladder"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for c in CONDS:
        st = np.array(D[c]["wave"]["steps"]); nl = np.array(D[c]["wave"]["nll"]).mean(1)
        for j, s in enumerate(lad):
            ax[1].plot(st, nl[:, j], "-", lw=1, color=COL[c], alpha=0.3 + 0.7 * j / len(lad))
        ax[1].plot([], [], color=COL[c], label=c)
    ax[1].set_xscale("log"); ax[1].set_xlabel("training step"); ax[1].set_ylabel("held-out NLL / token")
    ax[1].set_title("over training (light = small holes)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    for c in CONDS:
        for nm, ls in (("argmax1", "-"), ("beam8x4", "--")):
            ax[2].plot(area, [D[c]["cost_to_depth"][f"{s}_{nm}"]["code_acc"] for s in lad],
                       ls, marker="o", ms=3, color=COL[c], label=f"{c} {nm}")
    if lib:
        for nm, ls in (("argmax1", "-"), ("beam8x4", "--")):
            ax[2].plot(area, [lib["cost_to_depth"][f"{s}_{nm}"]["code_acc"] for s in lad],
                       ls, marker="o", ms=3, color=COL["lib0"], label=f"lib0 {nm}")
    ax[2].set_xscale("log"); ax[2].set_xlabel("mask area (cells)"); ax[2].set_ylabel("exact code recovery")
    ax[2].set_title("cost to depth: 1 forward vs 32"); ax[2].legend(fontsize=6); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "twin_wave_cost.png"), dpi=140); plt.close(fig)

    # --- grader against known validity ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    kinds = ("clean", "seam", "offstyle", "plant_fill")
    w = 0.11
    for a_, rule in zip(ax, ("pass_one_sided", "pass_two_sided")):
        for i, c in enumerate(CONDS):
            if not O[c] or "damage_grading" not in O[c]:
                continue
            rows = O[c]["damage_grading"].get("gA24", {})
            for j, kind in enumerate(kinds):
                vals = [rows.get(f"{lv}_{kind}", {}).get(rule, np.nan) for lv in LEVELS]
                a_.bar(np.arange(len(LEVELS)) + (i * len(kinds) + j) * w, vals, width=w,
                       color=COL[c], alpha=min(1.0, 0.32 + 0.22 * j), label=f"{c} {kind}")
        a_.axhline(0.90, color="k", ls=":", lw=.8); a_.axhline(0.10, color="k", ls=":", lw=.8)
        a_.set_xticks(np.arange(len(LEVELS)) + 3.5 * w)
        a_.set_xticklabels([f"L{lv} ({2**lv}x{2**lv} codes)" for lv in LEVELS])
        a_.set_ylabel("pass rate"); a_.set_title(rule.replace("_", " "))
        a_.legend(fontsize=5.5, ncol=2); a_.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "twin_grader.png"), dpi=140); plt.close(fig)
    print("figures ->", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    D, Q, O, lib, dsc = digest()
    if a.figures and all(D[c] for c in CONDS):
        figures(D, Q, O, lib, dsc)


if __name__ == "__main__":
    main()
