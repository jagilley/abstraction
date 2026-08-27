"""Reduce a canvas-ratchet run: fetch off the volume, print the tables, draw the figures.

    cd experiments        # conda glp; MODAL_PROFILE=chromatic
    python3 canvas/practice/ratchet/analyze.py --tag cr_s0 --fetch --figures

Readouts, in the order the node asks for them:
  1. per-era e_k at the declared budget, with the grader's CLEAN FALSE-REJECT floor beside it
     (an era's e can never go below its floor, so every number is read in-tag)
  2. PRICED TIME in forward passes, and the realised cost-to-depth curve
  3. the EARNED-VS-GIVEN fraction (e_never - e_arm) / (e_never - e_given), per era
  4. commit events: when, on what table, its precision/recall against the given vocabulary
  5. the unit-LP (shadow-audition) trajectories with the certificate's firing marked, against
     the given-table floor and the matched-size random control on the same instances
  6. the oracle columns (fill validity vs the clean-round-trip ceiling) and the taste gauge
  7. the plant guard, the width ladder, and the at-support yield of the miner
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
VOLUME = "canvas-data"
REMOTE = "canvas_practice_ratchet"
ORDER = ["never_base", "given", "practice_gated", "practice_early", "practice_late",
         "practice_prov"]
COL = {"never_base": "#7f8c8d", "given": "#2471a3", "practice_gated": "#c0392b",
       "practice_early": "#8e44ad", "practice_late": "#e67e22", "practice_prov": "#16a085"}


def fetch(tag, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", dest],
                   check=True,
                   env=dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE",
                                                                     "chromatic")))


def load(tag, dest):
    root = os.path.join(dest, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for a in sorted(os.listdir(root)):
        p = os.path.join(root, a, "results.json")
        if os.path.isfile(p):
            arms[a] = json.load(open(p))
    keyed = {a: arms[a] for a in ORDER if a in arms}
    keyed.update({a: r for a, r in arms.items() if a not in keyed})
    return setup, keyed


def era_slices(log, n_eras):
    era = np.array(log["era"])
    return {k: np.flatnonzero(era == k) for k in range(1, n_eras + 1)}


def tail(v, idx, n=5):
    return float(np.mean(np.asarray(v, float)[idx[-n:]])) if len(idx) else float("nan")


# --------------------------------------------------------------------------- #

def reduce_run(setup, arms, n_tail=5):
    eras = setup["eras"]
    R = {"tag_config": {k: setup["config"][k] for k in
                        ("seed", "era_cycles", "budget", "w_max", "lam", "gamma", "n_alt",
                         "mine_support", "mine_cap", "given_support", "given_support3",
                         "tileset", "n_pr", "n_rt", "n_score", "gen_steps")},
         "preflight": setup.get("preflight"), "given": setup.get("given_stats"),
         "given_sizes": setup.get("given_sizes"),
         "n_support_exemplars": setup.get("n_support_exemplars"),
         "arms": {}, "eras": eras}
    for a, r in arms.items():
        log = r["log"]
        sl = era_slices(log, len(eras))
        row = {"complete": r["complete"], "n_cycles": len(log["cycle"]),
               "e": {}, "t": {}, "cost": {}, "width": {}, "steps": {}, "n_moves": {},
               "clean_fr": {}, "typ_pass": {}, "typ_nll": {}, "e_practice": {},
               "n_unsup": {}, "incomplete": {}, "vocab_end": log["vocab"][-1],
               "events": r["events"]}
        for k in eras:
            i = sl[k]
            if not len(i):
                continue
            row["e"][str(k)] = tail(log["e"], i, n_tail)
            row["e_practice"][str(k)] = tail(log["e_practice"], i, n_tail)
            row["t"][str(k)] = float(np.asarray(log["t_cum"], float)[i[-1]]
                                     - (np.asarray(log["t_cum"], float)[i[0] - 1]
                                        if i[0] > 0 else 0.0))
            row["cost"][str(k)] = tail(log["cost"], i, n_tail)
            row["width"][str(k)] = int(log["width"][i[-1]])
            row["steps"][str(k)] = int(log["steps"][i[-1]])
            row["n_moves"][str(k)] = int(log["n_moves"][i[-1]])
            row["clean_fr"][str(k)] = float(log["clean_fr"][i[-1]])
            row["typ_pass"][str(k)] = float(np.mean([log["typ"][j]["pass"] for j in i[-n_tail:]]))
            row["typ_nll"][str(k)] = float(np.mean([log["typ"][j]["nll"] for j in i[-n_tail:]]))
            row["n_unsup"][str(k)] = tail(log["n_unsup"], i, n_tail)
            row["incomplete"][str(k)] = tail(log["incomplete"], i, n_tail)
        row["t_cum"] = float(log["t_cum"][-1])
        row["miner_end"] = log["miner"][-1]
        R["arms"][a] = row
    # earned-vs-given, against the in-tag floor
    if "never_base" in R["arms"] and "given" in R["arms"]:
        nb, gv = R["arms"]["never_base"]["e"], R["arms"]["given"]["e"]
        R["denominator"] = {k: nb[k] - gv[k] for k in nb if k in gv}
        for a, row in R["arms"].items():
            row["earned_vs_given"] = {
                k: ((nb[k] - row["e"][k]) / d if abs(d) > 1e-9 else None)
                for k, d in R["denominator"].items() if k in row["e"]}
            row["depth_slope"] = (row["e"][str(eras[-1])] - row["e"][str(eras[0])]
                                  if str(eras[-1]) in row["e"] else None)
    return R


def audition_series(arms, level):
    out = {}
    for a, r in arms.items():
        log = r["log"]
        s = {"cycle": log["cycle"], "era": log["era"],
             "cand": [c.get(str(level), {}).get("cand") for c in log["aud"]],
             "true": [c.get(str(level), {}).get("true") for c in log["aud"]],
             "rand": [c.get(str(level), {}).get("rand_k") for c in log["aud"]],
             "held": [c.get(str(level), {}).get("held") for c in log["aud"]],
             "live": [c.get(str(level), {}).get("live") for c in log["aud"]],
             "n": [c.get(str(level), {}).get("n_entries") for c in log["aud"]],
             "rec": [c.get(str(level), {}).get("tab_recall") for c in log["aud"]],
             "prec": [c.get(str(level), {}).get("tab_precision") for c in log["aud"]],
             "fired": [e["cycle"] for e in r["events"] if e["level"] == level]}
        out[a] = s
    return out


def probe_table(arms):
    out = {}
    for a, r in arms.items():
        pr = r["log"]["probe"]
        if not pr:
            continue
        last = {}
        for p in pr:
            last[p["era"]] = p
        out[a] = {str(k): {"ladder": p["ladder"], "all_eras": p["all_eras"],
                           "oracle": p.get("oracle"), "plant": p.get("plant")}
                  for k, p in last.items()}
    return out


# --------------------------------------------------------------------------- #

def report(R, auds, probes):
    C = R["tag_config"]
    print(f"\n=== canvas ratchet | tileset {C['tileset']} | seed {C['seed']} | "
          f"budget {C['budget']} forwards/solve, w_max {C['w_max']}, lam {C['lam']} ===")
    print(f"support: {R['n_support_exemplars']} exemplars; given |T2| "
          f"{R['given_sizes']['2']}, |T3| {R['given_sizes']['3']} "
          f"(block->tile purity {R['given']['t2_block_to_tile_purity']:.3f}, "
          f"{R['given']['t2_tiles_covered']} catalogue tiles covered)")
    print(f"clean false-reject floor (support grader, held-out truth at the era's hole): "
          + " ".join(f"L{k}: e>={1 - v:.3f}" for k, v in
                     sorted(R["preflight"].items())) if R.get("preflight") else "")

    print("\n-- e_k at the declared budget (mean of each era's last 5 cycles), "
          "priced time in FORWARD PASSES --")
    hdr = f"{'arm':16s}" + "".join(f"{'era ' + str(k):>22s}" for k in R["eras"]) \
        + f"{'t_cum':>10s}{'slope':>8s}"
    print(hdr)
    for a, row in R["arms"].items():
        cells = ""
        for k in R["eras"]:
            e = row["e"].get(str(k))
            t = row["t"].get(str(k))
            cells += f"{e:10.4f} @{t:10.0f}" if e is not None else f"{'-':>22s}"
        sl = row.get("depth_slope")
        print(f"{a:16s}{cells}{row['t_cum']:10.0f}"
              + (f"{sl:+8.3f}" if sl is not None else f"{'-':>8s}"))
    print(f"{'clean floor':16s}"
          + "".join(f"{R['arms'][list(R['arms'])[0]]['clean_fr'][str(k)]:10.4f}{'':>12s}"
                    for k in R["eras"]))

    if "denominator" in R:
        print("\n-- earned-vs-given (e_never - e_arm) / (e_never - e_given) --")
        print(f"{'arm':16s}" + "".join(f"{'era ' + str(k):>10s}" for k in R["eras"]))
        for a, row in R["arms"].items():
            if a in ("never_base", "given"):
                continue
            print(f"{a:16s}" + "".join(
                (f"{row['earned_vs_given'].get(str(k)):10.3f}"
                 if row["earned_vs_given"].get(str(k)) is not None else f"{'-':>10s}")
                for k in R["eras"]))
        print(f"{'denominator':16s}"
              + "".join(f"{R['denominator'][str(k)]:10.4f}" for k in R["eras"]))

    print("\n-- cost to depth: realised forwards/solve, beam width, steps, |moves| --")
    print(f"{'arm':16s}" + "".join(f"{'L' + str(k):>26s}" for k in R["eras"]))
    for a, row in R["arms"].items():
        print(f"{a:16s}" + "".join(
            f"{row['cost'].get(str(k), float('nan')):8.1f}f w{row['width'].get(str(k), 0):<2d}"
            f" s{row['steps'].get(str(k), 0):<3d} m{row['n_moves'].get(str(k), 0):<5d}"
            for k in R["eras"]))

    print("\n-- commits --")
    print(f"{'arm':16s}{'era':>4s}{'lvl':>4s}{'cyc':>5s}{'prov':>6s}{'entries':>8s}"
          f"{'recall':>8s}{'prec':>7s}{'audition':>10s}{'shadow':>9s}")
    for a, row in R["arms"].items():
        for e in row["events"]:
            f = lambda v: (f"{v:.4f}" if isinstance(v, float) else "-")
            print(f"{a:16s}{e['era']:4d}{e['level']:4d}{e['cycle']:5d}"
                  f"{str(e.get('provisional')):>6s}{e['n_entries']:8d}"
                  f"{e['tab_recall']:8.3f}{f(e['tab_precision']):>7s}"
                  f"{f(e['audition']):>10s}{f(e['shadow']):>9s}")

    for lv, aud in sorted(auds.items()):
        print(f"\n-- the unit-LP audition at level {lv} (era means; cand / given / rand_k) --")
        print(f"{'arm':16s}{'era':>4s}{'cand':>9s}{'given':>9s}{'rand_k':>9s}"
              f"{'entries':>9s}{'recall':>8s}{'prec':>7s}")
        for a, s in aud.items():
            era = np.array(s["era"])
            for k in sorted(set(era.tolist())):
                i = np.flatnonzero(era == k)
                g = lambda key: [s[key][j] for j in i if s[key][j] is not None]
                c, t, rk = g("cand"), g("true"), g("rand")
                n, rc, pc = g("n"), g("rec"), g("prec")
                if not (c or t):
                    continue
                fm = lambda v: f"{np.mean(v):9.4f}" if len(v) else f"{'-':>9s}"
                print(f"{a:16s}{k:4d}{fm(c)}{fm(t)}{fm(rk)}"
                      f"{np.mean(n) if n else 0:9.1f}"
                      f"{np.mean(rc) if rc else float('nan'):8.3f}"
                      f"{np.mean(pc) if pc else float('nan'):7.3f}")

    print("\n-- instruments: taste gauge, oracle validity, plant guard --")
    print(f"{'arm':16s}{'era':>4s}{'typ_pass':>9s}{'typ_nll':>9s}{'oracle_valid':>13s}"
          f"{'rt_ceiling':>11s}{'valid|rt':>9s}{'plantNLL(2/4/8)':>22s}")
    for a, row in R["arms"].items():
        for k in R["eras"]:
            p = probes.get(a, {}).get(str(k), {})
            o = p.get("oracle") or {}
            pl = (p.get("plant") or {}).get("ladder_nll") or []
            f = lambda v: (f"{v:.3f}" if isinstance(v, float) else "-")
            print(f"{a:16s}{k:4d}{row['typ_pass'].get(str(k), float('nan')):9.3f}"
                  f"{row['typ_nll'].get(str(k), float('nan')):9.3f}"
                  f"{f(o.get('valid_fill')):>13s}{f(o.get('valid_roundtrip_ceiling')):>11s}"
                  f"{f(o.get('valid_fill_on_clean_rt')):>9s}"
                  f"{' '.join(f'{v:.2f}' for v in pl):>22s}")

    print("\n-- width ladder (instrument, unpriced): e at forced width --")
    for a, p in probes.items():
        for k in R["eras"]:
            lad = p.get(str(k), {}).get("ladder", {})
            if lad:
                print(f"{a:16s} L{k}  " + "  ".join(
                    f"w{w}: e={v['e']:.4f} @{v['cost']:.0f}f" for w, v in sorted(
                        lad.items(), key=lambda x: int(x[0]))))

    print("\n-- miner: distinct tuples and how many are at support --")
    for a, row in R["arms"].items():
        m = row["miner_end"]
        print(f"{a:16s} " + "  ".join(
            f"L{l}: {m[l]['n_distinct']} distinct / "
            f"{m[l]['n_at_support']['3']} at support 3 (obs {m[l]['n_obs']})"
            for l in sorted(m)))


# --------------------------------------------------------------------------- #

def figures(R, auds, probes, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    eras = R["eras"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, row in R["arms"].items():
        y = [row["e"].get(str(k)) for k in eras]
        ax[0].plot(eras, y, "-o", color=COL.get(a, "k"), ms=5, label=a)
        ax[1].plot([row["t"].get(str(k)) for k in eras], y, "-o", color=COL.get(a, "k"), ms=5)
        ax[2].plot(eras, [row["cost"].get(str(k)) for k in eras], "-o",
                   color=COL.get(a, "k"), ms=5)
    fl = [1 - R["preflight"][f"clean_pass_L{k}"] for k in eras] if R.get("preflight") else None
    if fl:
        ax[0].plot(eras, fl, "k:", lw=1.4, label="clean false-reject floor")
    ax[0].set_xlabel("era (hole = $2^k\\times2^k$ codes)"); ax[0].set_ylabel("e (1 - support pass)")
    ax[0].set_title("cost to depth: error by era at the declared budget", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].set_xticks(eras)
    ax[1].set_xlabel("priced time (forward passes, per era)"); ax[1].set_ylabel("e")
    ax[1].set_title("error against priced time", fontsize=10)
    ax[2].set_xlabel("era"); ax[2].set_ylabel("forwards per solve")
    ax[2].set_title("realised cost per solve", fontsize=10); ax[2].set_xticks(eras)
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.suptitle(f"canvas ratchet {tag}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, "cost_to_depth.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for j, lv in enumerate((2, 3)):
        for a, s in auds[lv].items():
            c = np.array([np.nan if v is None else v for v in s["cand"]], float)
            t = np.array([np.nan if v is None else v for v in s["true"]], float)
            ax[j].plot(s["cycle"], c, "-", color=COL.get(a, "k"), lw=1.3, label=a)
            ax[j].plot(s["cycle"], t, ":", color=COL.get(a, "k"), lw=0.9)
            for f in s["fired"]:
                ax[j].axvline(f, color=COL.get(a, "k"), ls="--", lw=0.8, alpha=.6)
        ax[j].set_title(f"shadow audition, level {lv} (solid = earned, dotted = given)",
                        fontsize=10)
        ax[j].set_xlabel("cycle"); ax[j].set_ylabel("audition e"); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "audition.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, row in R["arms"].items():
        if a in ("never_base", "given"):
            continue
        y = [row.get("earned_vs_given", {}).get(str(k)) for k in eras]
        ax[0].plot(eras, y, "-o", color=COL.get(a, "k"), ms=5, label=a)
    ax[0].axhline(1.0, color="k", ls=":", lw=.8); ax[0].axhline(0.0, color="k", ls="-", lw=.6)
    ax[0].set_xticks(eras); ax[0].set_xlabel("era"); ax[0].set_ylabel("fraction of `given`")
    ax[0].set_title("earned vs given", fontsize=10); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for a, row in R["arms"].items():
        ax[1].plot(eras, [row["typ_pass"].get(str(k)) for k in eras], "-o",
                   color=COL.get(a, "k"), ms=5, label=a)
    ax[1].set_xticks(eras); ax[1].set_xlabel("era")
    ax[1].set_ylabel("typicality pass rate (q = 0.90)")
    ax[1].set_title("the taste gauge (logged, decides nothing)", fontsize=10)
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "earned_vs_given.png"), dpi=140)
    plt.close(fig)
    print(f"\nfigures -> {out}/{{cost_to_depth,audition,earned_vs_given}}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cr_s0")
    ap.add_argument("--dest", default=RES)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag, a.dest)
    setup, arms = load(a.tag, a.dest)
    R = reduce_run(setup, arms, a.tail)
    aud = audition_series(arms, 2)
    aud3 = audition_series(arms, 3)
    probes = probe_table(arms)
    auds = {2: aud, 3: aud3}
    report(R, auds, probes)
    os.makedirs(RES, exist_ok=True)
    json.dump(R, open(os.path.join(RES, f"reduced_{a.tag}.json"), "w"), indent=1)
    print(f"\nreduced -> {os.path.join(RES, f'reduced_{a.tag}.json')}")
    if a.figures:
        figures(R, auds, probes, a.tag)


if __name__ == "__main__":
    main()
