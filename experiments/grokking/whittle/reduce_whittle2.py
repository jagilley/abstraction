"""[grokking/whittle] Reduce a `whittle2.py::whittle2_run` run (tag w2) to the table of record, a figure and a summary.

    python3 grokking/whittle/reduce_whittle2.py --tag w2 --fetch      # from experiments/

--fetch pulls /data/<tag>/whittle2.json and /data/whittle_gates/gates2.json off `grokking-mint-data` into
results/<tag>/ (git-ignored). Writes figures/whittle_<tag>_table.txt, figures/whittle_<tag>_walk.png,
results/whittle_<tag>_summary.json.
"""

import argparse
import json
import os
import subprocess

import numpy as np

from reduce_whittle import _logticks

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ("bil_channel", "bil_plant1_s4", "bil_plant1_s9", "bil_plant1_s13", "bil_from_start", "mlp_channel",
         "mlp_channel_sgd", "mlp_channel_lr1e-2", "bil_channel_lr1e-2")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _get(vol_path, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE="chromatic")
    subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", vol_path, local], check=True, env=env)


def load(tag, fetch):
    path = os.path.join(HERE, "results", tag, "whittle2.json")
    gpath = os.path.join(HERE, "results", tag, "gates2.json")
    if fetch:
        _get(f"{tag}/whittle2.json", path)
        _get("whittle_gates/gates2.json", gpath)
    with open(path) as f:
        d = json.load(f)
    g = None
    if os.path.exists(gpath):
        with open(gpath) as f:
            g = json.load(f)
    return d, g


def kd(r, syms):
    return [r["meta"][t]["k_dft"] for t in syms]


def adm(r):
    return [x for x in r["walk"]["rounds"] if x["admit"]]


def series(r):
    s = r["start_walk"]
    out = [{"round": 0, "weights": s["weights"], "total_g": s["total_g"], "total_s": s["total_s"],
            "n_ch": len(s["channels"]), "heldout": s["heldout"], "move": "start"}]
    for x in adm(r):
        c = x["counts"]
        out.append({"round": x["round"], "weights": c["weights"], "total_g": c["total_g"], "total_s": c["total_s"],
                    "n_ch": len(c["channels"]), "heldout": x["heldout"], "move": x.get("move", "group"),
                    "layer": x["layer"]})
    return out


def write_table(d, g, tag, path):
    R = d["results"]
    arms = [a for a in ORDER if a in R]
    L = []
    w = L.append
    w(f"# grokking / whittle -- tag {tag}; seed 42; wd 0 everywhere; CPU; one container per arm; {d['seconds']:.0f} s wall")
    w(f"# channel move and plant retrain: cap {d['c_retrain']} epochs, stop once train agreement has held {d['c_settle']}"
      f" epochs; group moves (weights and biases): cap {d['g_retrain']}, settle {d['g_settle']} (w1's)")
    w("# gate: 0 of 2,822 train pairs disagree with the original net's train argmax; held-out (6,587) logged only")
    w("# counts: scalars (weights / biases / total) and groups (symbol connections in minted layers, scalars elsewhere;"
      " bias groups per unit, per symbol in minted heads). Scalar zeros inside a 2-vector depend on the arbitrary"
      " rotation within the subspace; the group count is the claim.")
    w("# k = DFT match of a minted symbol (oracle, logged only)")
    w("")
    if g is not None:
        w("## Gates (whittle2_gates)")
        for k in ("G5_bilinear_both", "G5_both_sides"):
            x = g[k]
            w(f"  {k}: weights {x['weights']} (w1 {x['w1_weights']}), disagreements {x['n_disagree']}, held-out "
              f"{x['heldout']:.4f} (w1 {x['w1_heldout']:.4f}), channels {x['channels']}, pass {x['pass']}")
        for t, x in g["G6a_bilinear_channel"].items():
            w(f"  G6a bilinear remove s{t}: units {x['units_before']} -> {x['units_after']} (removed {x['units_t']}),"
              f" max |logit - (before - channel terms)| {x['maxabs_vs_expected']:.1e}")
        for t, x in g["G6b_mlp_channel"].items():
            w(f"  G6b MLP remove s{t}: max |logit - hand-zeroed| {x['maxabs_vs_zeroed']:.1e}; units L0/L1 "
              f"{x['units_before']} -> {x['units_after']}; weights -> {x['weights_after']}")
        x = g["G6c_revival"]
        w(f"  G6c fold into a pruned head bias: max |dlogit| {x['maxabs']:.1e}, bias entries revived {x['revived']}")
        for t, x in g["G7_plant_start"].items():
            w(f"  G7 plant s{t} (k {x['k_dft']}): units {x['units']}, weights {x['weights']}, biases {x['bias_s']}, "
              f"pass {x['pass']}; train agreement at the plant {x['eval']['train_agree']:.4f}")
        for k, h in g["G8_long_retrain"].items():
            w(f"  G8 zero-prune 20k retrain {k}: " + ", ".join(
                f"{x['epochs']}:{x['n_disagree']}/{x['heldout']:.4f}/{x['gbar_margin']:.1f}" for x in h[::2] + [h[-1]])
              + "   (epochs: disagreements / held-out / gbar margin)")
        x = g["G9_walk2"]
        w(f"  G9 walk2 designed case: channels left {x['channels_left']}, pruned {x['pruned']}, tries "
          f"{x['channel_tries']}, pass {x['pass']}")
        w("")
    w("## Final state per arm")
    w(f"{'arm':>19s} | {'weights':>7s} {'w grp':>5s} {'bias s':>6s} {'b grp':>5s} | {'total s':>7s} {'total g':>7s} | "
      f"{'channels (k)':>28s} {'units':>7s} | {'held-out':>8s} {'gbar_m':>7s} {'m_tr min':>8s} {'m_te min':>8s} | "
      f"{'ch a/r':>6s} {'gr a/r':>7s} {'sec':>5s}")
    for a in arms:
        r = R[a]
        f = r["final"]
        rs = r["walk"]["rounds"]
        ca = sum(1 for x in rs if x.get("move") == "channel" and x["admit"])
        cr = sum(1 for x in rs if x.get("move") == "channel" and not x["admit"])
        ga = sum(1 for x in rs if x.get("move") != "channel" and x["admit"])
        gr = sum(1 for x in rs if x.get("move") != "channel" and not x["admit"])
        ch = f"{f['channels']} ({','.join(str(k) for k in kd(r, f['channels']))})"
        units = f"{f.get('units0', f.get('units'))}/{f.get('units1', '-')}"
        pr = f["probe"]
        w(f"{a:>19s} | {f['weights']:>7d} {f['weight_groups']:>5d} {f['bias_s']:>6d} {f['bias_g']:>5d} | "
          f"{f['total_s']:>7d} {f['total_g']:>7d} | {ch:>28s} {units:>7s} | {f['heldout']:>8.4f} "
          f"{pr['gbar_margin']:>7.3f} {pr['margin_tr_min']:>8.3f} {pr['margin_te_min']:>8.3f} | {ca:>2d}/{cr:<3d} "
          f"{ga:>3d}/{gr:<3d} {r['seconds']:>5.0f}")
    w("  (bilinear units = product units; MLP units = layer-0 / layer-1)")
    w("")
    w("## Start of each walk (after the plant, if any)")
    for a in arms:
        r = R[a]
        s = r["start_walk"]
        w(f"{a:>19s}: weights {s['weights']} (groups {s['weight_groups']}), biases {s['bias_s']} (groups {s['bias_g']}),"
          f" channels {s['channels']} (k {kd(r, s['channels'])}), train dis {s['n_disagree']}, held-out {s['heldout']:.4f}")
    w("")
    plants = [a for a in arms if R[a].get("plant")]
    if plants:
        w("## Plants: only one channel's 3 units, every bias zero, one long retrain")
        for a in plants:
            p = R[a]["plant"]
            c0 = p["counts_at_plant"]
            pa = p["probe_after"]
            k = str(p["k_dft"])
            w(f"{a:>19s}: s{p['sym']} (k {p['k_dft']}): at the plant {c0['units']} units / {c0['weights']} weights / "
              f"{c0['bias_s']} biases, train dis {p['eval_at_plant']['n_disagree']} (agreement "
              f"{p['eval_at_plant']['train_agree']:.4f}); retrain {p['retrain']['epochs']} epochs (first 100% at "
              f"{p['retrain']['first_ok_epoch']}); after: dis {p['eval_after']['n_disagree']}, held-out "
              f"{p['eval_after']['heldout']:.4f}, admitted {p['admit']}; amplitude alpha_k {pa['alpha'].get(k, float('nan')):.3f},"
              f" gbar margin {pa['gbar_margin']:.4f}, margin min tr/te {pa['margin_tr_min']:.4f}/{pa['margin_te_min']:.4f}")
        w("")
    w("## Channel moves (every attempt; amplitude = diagonal cosine coefficient alpha_k of the logit table, an oracle-DFT"
      " read; margins from the probe)")
    for a in arms:
        r = R[a]
        chs = [x for x in r["walk"]["rounds"] if x.get("move") == "channel"]
        if not chs:
            continue
        w(f"### {a}")
        w(f"{'round':>5s} {'sym':>4s} {'k':>3s} {'ch norm':>8s} {'admit':>5s} {'epochs':>6s} {'ok@':>6s} {'dis':>5s} "
          f"{'held-out':>8s} | {'gbar_m before/after':>19s} {'m_tr min before/after':>22s} | amplitudes of the channels"
          f" before -> after")
        for x in chs:
            b = x["before"]["probe"]
            aft = x.get("after", {}).get("probe")
            ks_b = kd(r, x["channels_before"])
            amp_b = ",".join(f"{b['alpha'].get(str(k), float('nan')):.2f}" for k in ks_b)
            if aft:
                rem = [t for t in x["channels_before"] if t != x["sym"]]
                amp_a = ",".join(f"{aft['alpha'].get(str(k), float('nan')):.2f}" for k in kd(r, rem))
                gm = f"{b['gbar_margin']:.3f}/{aft['gbar_margin']:.3f}"
                mt = f"{b['margin_tr_min']:.3f}/{aft['margin_tr_min']:.3f}"
            else:
                amp_a, gm, mt = "-", f"{b['gbar_margin']:.3f}/-", f"{b['margin_tr_min']:.3f}/-"
            w(f"{x['round']:>5d} {x['sym']:>4d} {r['meta'][x['sym']]['k_dft']:>3d} {x['channel_norm']:>8.2f} "
              f"{'yes' if x['admit'] else 'no':>5s} {x['epochs']:>6d} {str(x['first_ok_epoch']):>6s} {x['n_disagree']:>5d} "
              f"{x['heldout']:>8.4f} | {gm:>19s} {mt:>22s} | k {ks_b}: {amp_b} -> {amp_a}")
        w("")
    w("## One-channel bilinear finals, written out (U, V on the symbol's 2 input coordinates, W on its 2 head rows;"
      " T = sum_i U_i x V_i x W_i; fit T ~ A Re(e^{i psi} z_a z_b conj z_c))")
    for a in arms:
        r = R[a]
        ex = r.get("explicit")
        if not ex:
            continue
        w(f"### {a}: channels {r['final']['channels']}; DC head row alive {ex['dc_row_alive']}; head-bias entries alive "
          f"{ex['bias_alive']} (norm {ex['bias_norm']:.3g})")
        m = ex["margins"]
        w(f"  margins on all {m['n_pairs']} pairs: min {m['min']:.4f} (train {m['min_train']:.4f}, held-out "
          f"{m['min_heldout']:.4f}), q01/q50/q99 {[round(v, 4) for v in m['q01_q50_q99']]}, max {m['max']:.4f}, "
          f"non-positive {m['n_nonpositive']}")
        for c in ex["channels"]:
            w(f"  symbol s{c['sym']} (candidate {c['cand']}; eigenvalue phase at g_ref {c['lam_phase']:.6f} = "
              f"2 pi x {c['lam_phase_p_over_2pi']:.5f} / 97; DFT k {c['k_dft']} [oracle]); units {c['units']}")
            for i, u in enumerate(c["units"]):
                w(f"    unit {u:>3d}: U {c['U'][i]}  V {c['V'][i]}  W {c['W'][i]}  W_DC {c['W_dc'][i] if i < len(c['W_dc']) else '-'}")
            w(f"    T = {c['T']}")
            w(f"    fit: A = {c['A']:.4f}, psi = {c['psi']:.5f}; relative residual {c['rel_resid']:.4f}; frame phase at "
              f"e_hat {c['frame_phase_at_e_hat']:.5f} (psi + frame phase = {c['psi'] + c['frame_phase_at_e_hat']:.5f});"
              f" predicted margin A (1 - cos 2pi/97) = {c['A'] * (1 - np.cos(2 * np.pi / 97)):.4f}; coordinate radius "
              f"{[round(v, 5) for v in c['y_radius_min_max']]}")
        w("")
    w("## MLP finals: surviving structure")
    for a in arms:
        r = R[a]
        s = r["structure"]
        if r["cfg"]["base"] != "both_sides":
            continue
        kds = s["sym_k_dft"]
        w(f"### {a}: L0 units {s['n_units0']}, L1 units {s['n_units1']}; single-symbol L0 {s.get('n_single')} / "
          f"{s['n_units0']}; symbols read {s['read_syms']} (k {s['read_k_dft']}); written {s.get('write_syms')} "
          f"(k {[kds[t] for t in s.get('write_syms', [])]})")
        for u in s["units0"]:
            w(f"    L0 unit {u['u']:>3d}: a {u['a']} | b {u['b']}   k: a {[kds[t] for t in u['a']]} | b {[kds[t] for t in u['b']]}")
        for u in s["units1"]:
            w(f"    L1 unit {u['u']:>3d}: fan-in {u['fan_in']} in_syms {u['in_syms']} writes {u.get('writes')}")
        w("")
    w("## Drift twins (the walk's start, nothing pruned, the admitted rounds' epoch counts replayed with the arm's optimizer)")
    for a in arms:
        tw = R[a].get("drift_twin", [])
        if not tw:
            w(f"{a:>19s}: no admitted rounds")
            continue
        h = [x["heldout"] for x in tw]
        w(f"{a:>19s}: {len(tw)} chunks, {sum(x['epochs'] for x in tw)} epochs; held-out min {min(h):.4f} median "
          f"{np.median(h):.4f} final {h[-1]:.4f}; max train disagreements {max(x['n_disagree'] for x in tw)}")
    w("")
    w("## Logit-table energy at the end (DFT read, oracle): diagonal / DC / n90 / top k")
    for a in arms:
        le = R[a]["final"]["logit_energy"]
        w(f"{a:>19s}: {le['diag_total']:.4f} / {le['dc_frac']:.4f} / {le['n90']} / {le['top_k'][:6]}")
    w("")
    for a in arms:
        r = R[a]
        w(f"## Admitted rounds: {a}")
        for x in series(r):
            w(f"  r{x['round']:>3d} {x['move']:>7s} {x.get('layer', ''):>9s}  weights {x['weights']:>5d}  total groups "
              f"{x['total_g']:>5d}  total scalars {x['total_s']:>5d}  channels {x['n_ch']:>2d}  held-out {x['heldout']:.4f}")
        w("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def write_figure(d, tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = d["results"]
    arms = [a for a in ORDER if a in R]
    n = len(arms)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.8), squeeze=False)
    for j, a in enumerate(arms):
        r = R[a]
        S = series(r)
        x = [s["round"] for s in S]
        ax = axes[0, j]
        ax.plot(x, [s["total_g"] for s in S], "-o", color=C1, lw=2, ms=3.5, mec="white", mew=0.6, label="total groups")
        ax.plot(x, [s["total_s"] for s in S], "-o", color=C2, lw=1.4, ms=3, mec="white", mew=0.6, label="total scalars")
        for s in S:
            if s["move"] == "channel":
                ax.axvline(s["round"], color=C3, lw=1.2, zorder=0)
        ax.set_yscale("log")
        vals = [s["total_g"] for s in S] + [s["total_s"] for s in S]
        _logticks(ax.yaxis, min(vals), max(vals))
        ax.set_ylim(min(vals) / 1.3, max(vals) * 1.3)
        if len(x) == 1:
            ax.set_xlim(-1, 1)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        f = r["final"]
        ax.set_title(f"{a}\nend: {f['total_g']} groups / {f['total_s']} scalars, {len(f['channels'])} ch", fontsize=8.5,
                     color=INK)
        ax.tick_params(labelsize=7, colors=INK2)
        if j == 0:
            ax.set_ylabel("surviving parameters (log)", fontsize=8, color=INK2)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax2 = axes[1, j]
        ax2.plot(x, [s["heldout"] for s in S], "-o", color=C1, lw=1.6, ms=3, mec="white", mew=0.6, label="walk")
        tw = {t["round"]: t["heldout"] for t in r.get("drift_twin", [])}
        if tw:
            xs = [s["round"] for s in S[1:] if s["round"] in tw]
            ax2.plot(xs, [tw[t] for t in xs], "-", color="#a3a29d", lw=1.1, label="drift twin")
        ax2.set_ylim(min(0.98, min(s["heldout"] for s in S) - 0.002), 1.001)
        if len(x) == 1:
            ax2.set_xlim(-1, 1)
        ax2.grid(True, color=GRID, lw=0.6)
        ax2.tick_params(labelsize=7, colors=INK2)
        ax2.set_xlabel("round (green: channel removals)", fontsize=7.5, color=INK2)
        if j == 0:
            ax2.set_ylabel("held-out (oracle, logged)", fontsize=8, color=INK2)
            ax2.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle(f"whittle {tag}: channel move + group walk (weights and biases), admitted rounds per arm",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary(d, tag, path):
    R = d["results"]
    out = {"tag": tag, "arms": {}}
    for a in [a for a in ORDER if a in R]:
        r = R[a]
        out["arms"][a] = {"cfg": r["cfg"], "start_walk": {k: v for k, v in r["start_walk"].items()},
                          "final": {k: v for k, v in r["final"].items() if k not in ("logit_energy",)},
                          "plant": r.get("plant"), "explicit": r.get("explicit"),
                          "channel_moves": [{k: v for k, v in x.items() if k not in ("counts", "probe")}
                                            for x in r["walk"]["rounds"] if x.get("move") == "channel"],
                          "series": series(r), "drift_twin": r.get("drift_twin"), "seconds": r["seconds"],
                          "stats": r["stats"]}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="w2")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    d, g = load(args.tag, args.fetch)
    t = os.path.join(HERE, "figures", f"whittle_{args.tag}_table.txt")
    write_table(d, g, args.tag, t)
    write_figure(d, args.tag, os.path.join(HERE, "figures", f"whittle_{args.tag}_walk.png"))
    write_summary(d, args.tag, os.path.join(HERE, "results", f"whittle_{args.tag}_summary.json"))
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
