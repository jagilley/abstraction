"""Reduce a round-0 gate sweep into the gate report.

Fetches `/data/practice_fingering/<tag>/<world>/results.json` off the `mujoco-control-data` volume
(`--fetch`) and prints one table per gate across worlds, plus the calibration record the main run
is configured from. The verdict column is the number, not a pass/fail stamp: `typed_gaps`'s rule is
that a gate that fails is a finding, so the thresholds are stated and the reader adjudicates.

Usage (from experiments/):
    python3 mjc/practice/fingering/analyze_gates.py --tag g0 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_fingering/{tag}", dst],
                   check=False, env={**os.environ, "MODAL_PROFILE":
                                     os.environ.get("MODAL_PROFILE", "chromatic")})
    return dst


def load(tag, fetched):
    """Accept either the volume layout (<tag>/<world>/results.json) or the local-entrypoint
    mirror (gates_<world>.json), whichever survived."""
    root = os.path.join(HERE, "results", tag)
    out = {}
    for base in (os.path.join(root, tag), root):
        if not os.path.isdir(base):
            continue
        for w in sorted(os.listdir(base)):
            p = os.path.join(base, w, "results.json")
            if os.path.isfile(p):
                out[w] = json.load(open(p))
    for f in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if f.startswith("gates_") and f.endswith(".json"):
            d = json.load(open(os.path.join(root, f)))
            out.setdefault(d.get("world", f[6:-5]), d)
    return out


def row(vals, w=13):
    return "".join(str(v).ljust(w) for v in vals)


def fmt(x, p=4):
    return "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    R = load(a.tag, a.fetch)
    if not R:
        raise SystemExit(f"no results found for tag {a.tag}")
    ws = list(R)
    print(f"\n=== fingering round-0 gate report: tag {a.tag} ({len(ws)} worlds) ===\n")
    for w in ws:
        c = R[w]["config"]
        print(f"  {w:10s} curl_b={c['curl_b']:<6} push_a={c['push_a']:<6} "
              f"complete={R[w].get('complete')}  H_drill={c['h_drill']} "
              f"sigma_perf={c['sigma_perf']} q_jit={c['q_jit']}")

    def table(title, rows):
        print(f"\n--- {title}")
        print(row([""] + ws))
        for label, f in rows:
            print(row([label] + [fmt(f(R[w])) for w in ws]))

    table("G3  usable range (stale -> corridor-matched ceiling, drilled @ tempo)", [
        ("stale", lambda r: r["g3"]["stale_tempo"]),
        ("ceiling", lambda r: r["g3"]["ceiling_tempo"]),
        ("reactive", lambda r: r["g3"]["reactive_stale"]),
        ("range", lambda r: r["g3"]["usable_range"]),
        ("noise sd", lambda r: r["g3"]["metering_noise_sd"]),
        ("range/noise", lambda r: r["g3"]["range_over_noise"]),
        ("ball/react", lambda r: r["g3"]["ballistic_over_reactive"]),
    ])
    table("G4  composition horizon (FM rollout tip err > 0.05 m) vs H_drill", [
        ("H_drill", lambda r: r["g4"]["h_drill"]),
        ("horizon stale", lambda r: r["g4"]["horizon_stale_005"]),
        ("horizon ceil", lambda r: r["g4"]["horizon_ceil_005"]),
        ("h>0.02 stale", lambda r: r["g4"]["horizon_stale_002"]),
        ("err@H stale", lambda r: r["g4"]["err_at_H_stale"]),
        ("err@H ceil", lambda r: r["g4"]["err_at_H_ceil"]),
    ])
    table("G1  boundary information", [
        ("ho tip sd", lambda r: r["g1"]["handover_tip_sd"]),
        ("ho tip disp", lambda r: r["g1"]["handover_tip_disp"]),
        ("null sd", lambda r: r["g1"]["null_coord_sd"]),
        ("spread/noise", lambda r: r["g1"]["spread_over_noise"]),
        ("fixed err", lambda r: r["g1"]["fixed_unit_err_mean"]),
        ("fixed cv", lambda r: r["g1"]["fixed_unit_err_cv"]),
        ("R2 on ho", lambda r: r["g1"]["r2_err_on_handover"]),
        ("decile best", lambda r: r["g1"]["decile_best"]),
        ("decile worst", lambda r: r["g1"]["decile_worst"]),
        ("oracle gain", lambda r: r["g1"]["oracle_gain"]),
        ("drift sd", lambda r: r["g1"]["post_commit_sd"]),
        ("drift", lambda r: r["g1"]["post_commit_drift"]),
    ])
    table("G2  multimodality (mean of successful renditions vs its contributors)", [
        ("n successful", lambda r: r["g2"]["n_successful"]),
        ("contrib med", lambda r: r["g2"]["contributor_med"]),
        ("contrib best", lambda r: r["g2"]["contributor_best"]),
        ("MEAN replay", lambda r: r["g2"]["mean_of_successful"]),
        ("mean/med", lambda r: r["g2"]["ratio_mean_over_med"]),
        ("mean/best", lambda r: r["g2"]["ratio_mean_over_best"]),
        ("k2 sep cmds", lambda r: r["g2"]["modes_commands"]["k2"]["separation"]),
        ("k2 sep path", lambda r: r["g2"]["modes_tip_path"]["k2"]["separation"]),
        ("k2 sizes", lambda r: str(r["g2"]["modes_commands"]["k2"]["sizes"])),
        ("within-mean", lambda r: str([round(v, 3) for v in
                                       r["g2"]["modes_commands"]["k2"]["within_cluster_mean_replay"]])),
    ])
    table("audition (the same matrix `fixed` and `library` both read)", [
        ("n_cand", lambda r: r["audition"]["n_cand"]),
        ("n_score", lambda r: r["audition"]["n_score"]),
        ("score med", lambda r: r["audition"]["score_med"]),
        ("best fixed", lambda r: r["audition"]["best_fixed"]),
        ("state oracle", lambda r: r["audition"]["per_state_oracle"]),
        ("oracle gain", lambda r: r["audition"]["oracle_gain"]),
        ("fixed HO", lambda r: r["audition"]["fixed_heldout"]),
        ("library HO", lambda r: r["audition"]["library_heldout"]),
        ("lib/fixed HO", lambda r: r["audition"]["lib_over_fixed_heldout"]),
        ("curse in-samp", lambda r: r["audition"]["curse"]["in_sample"]),
        ("own-err pick", lambda r: r["audition"]["curse"]["own_err_pick"]),
        ("lib cells", lambda r: str(r["audition"]["library"]["cell_sizes"])),
        ("lib distinct", lambda r: r["audition"]["library"]["n_distinct"]),
    ])
    table("FM corridor-slice probe (the spatial-matching readout)", [
        ("stale region", lambda r: r["fm_probe"]["stale_corridor_region"]),
        ("stale clean", lambda r: r["fm_probe"]["stale_corridor_clean"]),
        ("ceil region", lambda r: r["fm_probe"]["ceil_corridor_region"]),
        ("ceil clean", lambda r: r["fm_probe"]["ceil_corridor_clean"]),
        ("n in-region", lambda r: r["fm_probe"]["n_corridor_in_region"]),
    ])
    table("collection diagnostics (COLLECTION_REALISM tier C)", [
        ("corr ou", lambda r: r["diet"]["corr_ou"]["max_abs_corr"]),
        ("corr reach", lambda r: r["diet"]["corr_reach"]["max_abs_corr"]),
        ("corr corridor", lambda r: r["diet"]["corr_corridor"]["max_abs_corr"]),
        ("exclude keep", lambda r: r["diet"]["exclude_keep_frac"]),
        ("speed p95", lambda r: r["diet"]["diagnostics"].get("speed_p95")),
        ("steps agent", lambda r: r["ledger"]["steps_agent"]),
        ("steps instr", lambda r: r["ledger"]["steps_instrument"]),
        ("t_priced", lambda r: r["ledger"]["t_priced"]),
    ])

    print("\n--- CAL-P  planner sizing (drilled ballistic median; gap = stale - ceiling)")
    keys = sorted({k for w in ws for k in R[w].get("cal_planner", {})})
    print(row(["size"] + [f"{w}" for w in ws], 22))
    for k in keys:
        cells = []
        for w in ws:
            c = R[w].get("cal_planner", {}).get(k)
            cells.append("-" if not c else
                         f"{c['stale']['median']:.4f}/{c['ceiling']['median']:.4f} "
                         f"({c['gap']:+.4f})")
        print(row([k] + cells, 22))

    print("\n--- CAL-D  fraction of the stale->ceiling range closed, by practice cycle")
    for w in ws:
        f = R[w].get("cal_descent", {})
        print(f"  {w:10s} n_grad={f.get('n_grad')} lr={f.get('adapt_lr')}  "
              f"{[round(v, 3) for v in f.get('frac_of_range_closed', [])]}")
        print(f"  {'':10s} e_by_cycle {[round(v, 4) for v in f.get('e_by_cycle', [])]}")

    print("\n--- verdict block")
    vs = sorted({k for w in ws for k in R[w].get("verdict", {})})
    print(row([""] + ws, 16))
    for k in vs:
        print(row([k] + [fmt(R[w].get("verdict", {}).get(k)) for w in ws], 16))
    print()


if __name__ == "__main__":
    main()
