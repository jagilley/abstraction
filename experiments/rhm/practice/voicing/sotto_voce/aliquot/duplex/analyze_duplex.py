"""[duplex] The reduction. Facts only; no interpretation — that waits on discussion.

Reads the per-seed JSONs `duplex.py::probe_seed` wrote to the volume (fetched with `--fetch`)
and writes `figures/du_reduction.txt`:

  (G1)  THE REPRODUCTION GATE. The frozen trunk's `post`/`pre`/`pmean` and the random trunk's
        `post` against `overtone/figures/ov_*_dump.txt`'s own `post_slot`/`pre_slot`/`post_mean`/
        `rand_slot` cells, slot by slot, and the `mlp`/`dp` columns too. If this section is not
        clean, nothing below it is a re-read of overtone's protocol.
  (G0)  the estimator gate: the torch IRLS against `voicing::ov_irls` on real rows.
  (a)   THE PROBE, per trunk and diet: median over slots of the held-out AUC.
  (b)   the same, per level.
  (c)   THE WORLD-MODEL DIAGNOSTICS, per trunk, before and after.
  (d)   THE TRANSFER SPLIT: fit at L2/L3, scored at L2/L3 and at L4/L5, beside the L4/L5 fit.
  (e)   the shaping's own trajectory, the trunk fingerprints, and the cost.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/duplex/analyze_duplex.py --tag du0 --fetch
"""

import argparse
import json
import os
import re
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OVFIG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "overtone", "figures"))
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_duplex"

READS = ("pre", "pmean", "post", "postm")
DIETS = ("filed", "probe", "probe/u", "probe/d")
# overtone's column names for the same objects
OV = {"pre": "pre_slot", "pmean": "post_mean", "post": "post_slot", "mlp": "mlp", "dp": "dp"}


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def parse_ov_dump(path, arm):
    """The banked overtone S1b table, per (diet, slot) -> {column: value or None}."""
    if not os.path.isfile(path):
        return {}
    txt = open(path).read()
    i = txt.find("THE POST-WRITE LINEAR PROBE")
    if i < 0:
        return {}
    txt = txt[i:]
    j = txt.find(f"{arm}:")
    if j < 0:
        return {}
    txt = txt[j:]
    k = txt.find("MEDIAN over slots")
    if k > 0:
        txt = txt[:k]
    rx = re.compile(r"^\s+(filed|probe|probe/u|probe/d)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
                    r"([\d.]+)\s*\|\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\|\s*(\S+)\s+(\S+)\s*$")
    out = {}
    for ln in txt.splitlines():
        mm = rx.match(ln)
        if not mm:
            continue
        g = mm.groups()

        def f(x):
            return None if x == "-" else float(x)
        out[(g[0], g[1])] = {"n_tr": int(g[2]), "n_hold": int(g[3]), "base": float(g[4]),
                             "pre_slot": f(g[5]), "rand_slot": f(g[6]),
                             "post_mean": f(g[7]), "post_slot": f(g[8]),
                             "mlp": f(g[9]), "dp": f(g[10])}
    return out


def med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def fmt(x, w=6, p=3):
    return (f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-")


def collect(res):
    """per trunk -> per diet -> per read -> list of (slot, level, value)."""
    out = {}
    for nm, t in res["trunks"].items():
        d = {}
        for rec in t["per_buf"]:
            for lbl, key in (("", "auc"), ("/u", "auc_u"), ("/d", "auc_d")):
                dd = rec.get(key)
                if dd is None:
                    continue
                diet = rec["which"] + lbl
                for rd, val in dd.items():
                    d.setdefault(diet, {}).setdefault(rd, []).append(
                        (rec["slot"], rec["level"], val))
        out[nm] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="du0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--vs", default="", help="other tags to put beside this one in section (f)")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    root = os.path.join(FIG, a.tag)
    seeds = {}
    for k in ("s0", "s2"):
        p = os.path.join(root, f"{k}.json")
        if os.path.isfile(p):
            seeds[k] = json.load(open(p))
    lines = []

    def o(s=""):
        lines.append(s)
        print(s)

    o("=" * 100)
    o(f"[duplex] {a.tag} — THE OUTCOME ERROR IN THE TRUNK'S WEIGHTS, OFFLINE ON BANKED "
      "CHECKPOINTS")
    o("=" * 100)
    o("  TAG STATUS. The `dedup` flag in (G2) is DESIGN 2.5's exact-duplicate filter. It is")
    o("  OFF in `du0`, which is therefore the tag that reproduces overtone and the tag")
    o("  comparable to the loop's own audit, and ON in `du1`, whose FILED column is the one")
    o("  with no exact train/test duplicate in it. `du1`'s (G1) section is EXPECTED to differ")
    o("  from the banked dump; the size of that difference is what the channel was worth.")
    o("")
    o("  Facts only. The protocol is `overtone/analyze_dump.py::post_write_probe`'s: a linear")
    o("  probe on the trunk's pooled hiddens over the POST-WRITE configuration, fit on the")
    o("  critic's TRAINING rows (`hold_code >= 100*vo_critic_hold`) and scored on the rows the")
    o("  critic was AUDITED on. Gate G1 asserts that the two trunks overtone already read —")
    o("  the banked one and a never-trained one at overtone's own init seed — come back cell")
    o("  for cell; every other trunk is that same protocol after a shaping step.")
    o("")
    o("  READS.  pre   [pooled(obs).mean ; pooled(obs)[span].mean]    — the write NOT in the")
    o("                input, so whatever it reaches is context difficulty")
    o("          pmean  pooled(obs+write).mean                        (overtone's `post_mean`)")
    o("          post  [pooled(obs+write).mean ; span mean]           THE READ OF RECORD")
    o("                                                               (overtone's `post_slot`)")
    o("          postm  the same with one block outside the span masked (`macros.parse_features`'")
    o("                `mask_block` idiom; the plant has never seen a fully unmasked input)")
    o("          mlp    the run's own critic, recomputed from `vo_heads.pt`, through THIS trunk")
    o("          dp     the free prior, from the dump; identical on every trunk by construction")
    o("")
    o("  TRUNKS. frozen = the banked trunk. rand = never trained, overtone's `manual_seed"
      "(20260915)`.")
    o("          both@N / out@N / infill@N = N steps of (outcome BCE + infill CE) / (outcome BCE")
    o("          alone) / (infill CE alone) at lr 1e-4, the loop's own `gen_lr`. The outcome")
    o("          rows are the critic's own TRAINING rows (filed and probe both, per slot); the")
    o("          held-out rows enter nothing. `both` and `out` see identical outcome rows;")
    o("          `both` and `infill` see identical corpus windows and identical masks.")
    o("")

    # ---------------- G0 / G1 --------------------------------------------------------- #
    o("-" * 100)
    o("(G0) THE ESTIMATOR GATE — the torch IRLS against `voicing::ov_irls` on real rows.")
    o("-" * 100)
    for k, r in seeds.items():
        g = r["gate"]
        o(f"  seed {k}  {g['key']}:  max |Δw| {g['max_abs_dw']:.3e}   max |Δscore| "
          f"{g['max_abs_dscore']:.3e}   AUC torch {g['auc_torch']:.9f}  numpy "
          f"{g['auc_numpy']:.9f}  (Δ {g['d_auc']:.2e})")
    o("")
    o("-" * 100)
    o("(G2) THE EXACT-DUPLICATE CHANNEL (DESIGN §2.5). A FILED row's post-write configuration IS")
    o("     its trajectory's final configuration, while `hold_code` hashes the PRE-write context,")
    o("     so one trajectory that wrote at several slots puts the SAME post-write input with the")
    o("     SAME verdict on both sides of the audit split. Measured here, on every run. It is")
    o("     inherited from the arc's own split (the in-loop critic's filed audit has it too); it")
    o("     cannot inflate `frozen` or `rand`, which never see a verdict.")
    o("-" * 100)
    for k, r in seeds.items():
        ovs = r.get("overlap")
        if not ovs:
            o(f"  seed {k}: dedup={r.get('dedup', False)}   (this run predates the instrument; "
              "the numbers are in the companion tag)")
            continue
        o(f"  seed {k}   dedup={r.get('dedup')}   filter drops "
          f"{r.get('n_dedup_drop')} training rows")
        for which, d in ovs.items():
            o(f"      {which:6}: {d['n_in_train']:6d} / {d['n_hold']:6d} held-out rows "
              f"({d['share']:.3%}) have their exact post-write configuration in TRAINING")
    o("")
    o("-" * 100)
    o("(G1) THE REPRODUCTION GATE — the frozen and random trunks against the banked overtone")
    o("     dump, cell by cell. `n` counts the cells both tables report; Δ is the max absolute")
    o("     difference over them.")
    o("-" * 100)
    for k, r in seeds.items():
        ov = parse_ov_dump(os.path.join(OVFIG, f"{r['tag']}_dump.txt"), r["arm"])
        if not ov:
            o(f"  seed {k}: no banked dump at {r['tag']}_dump.txt — gate NOT RUN")
            continue
        rows = {("frozen", rd): [] for rd in ("pre", "pmean", "post", "mlp", "dp")}
        rows[("rand", "post")] = []
        for nm in ("frozen", "rand"):
            for rec in r["trunks"][nm]["per_buf"]:
                for lbl, key in (("", "auc"), ("/u", "auc_u"), ("/d", "auc_d")):
                    dd = rec.get(key)
                    if dd is None:
                        continue
                    ref = ov.get((rec["which"] + lbl, rec["slot"]))
                    if ref is None:
                        continue
                    for rd in ("pre", "pmean", "post", "mlp", "dp"):
                        if nm == "rand" and rd != "post":
                            continue
                        mine = dd.get(rd)
                        theirs = ref["rand_slot"] if (nm == "rand") else ref[OV[rd]]
                        if mine is None or theirs is None:
                            continue
                        rows[(nm, rd)].append(abs(mine - theirs))
        o(f"  seed {k}  ({r['tag']}/{r['arm']})")
        allmax = 0.0
        for (nm, rd), ds in rows.items():
            col = "rand_slot" if nm == "rand" else OV[rd]
            if not ds:
                o(f"    {nm:7} {rd:6} vs {col:10}: no comparable cells")
                continue
            allmax = max(allmax, max(ds))
            o(f"    {nm:7} {rd:6} vs {col:10}: n {len(ds):3d}   max |Δ| {max(ds):.4f}   "
              f"mean |Δ| {np.mean(ds):.4f}")
        o(f"    ==> G1 max |Δ| over every compared cell: {allmax:.4f}  "
          f"({'PASS at 0.001' if allmax <= 1e-3 else 'see above'})")
    o("")
    o("-" * 100)
    o("(G1b) THE SAME GATE AT THE MEDIAN — this node's `frozen`/`rand` medians against the")
    o("      `MEDIAN over slots` lines overtone itself printed. This is the comparison that")
    o("      cannot be moved by one ill-conditioned cell.")
    o("-" * 100)
    for k, r in seeds.items():
        path = os.path.join(OVFIG, f"{r['tag']}_dump.txt")
        if not os.path.isfile(path):
            continue
        txt = open(path).read()
        txt = txt[txt.find("THE POST-WRITE LINEAR PROBE"):]
        txt = txt[txt.find(f"{r['arm']}:"):]
        txt = txt[txt.find("MEDIAN over slots"):]
        ref = {}
        for ln in txt.splitlines():
            t = ln.split()
            if not t or t[0] not in DIETS:
                continue
            ref[t[0]] = ({t[i]: float(t[i + 1]) for i in range(1, len(t) - 1, 2)
                          if t[i] in ("pre_slot", "rand_slot", "post_mean", "post_slot",
                                      "mlp", "dp")},
                         t[-1].strip("()") if "slots" in ln else "")
        mm = 0.0
        o(f"  seed {k}  ({r['tag']}/{r['arm']})")
        o("      diet     |   pre  (overtone)  |  pmean (overtone) |  post  (overtone) | "
          " mlp  (overtone) |  dp   (overtone) | rand  (rand_slot)")
        for diet, (d, ns) in ref.items():
            mine = r["trunks"]["frozen"]["median"].get(diet, {})
            mr = r["trunks"]["rand"]["median"].get(diet, {})
            cells = [("pre", "pre_slot", mine.get("pre")), ("pmean", "post_mean",
                                                            mine.get("pmean")),
                     ("post", "post_slot", mine.get("post")), ("mlp", "mlp", mine.get("mlp")),
                     ("dp", "dp", mine.get("dp")), ("rand", "rand_slot", mr.get("post"))]
            row = []
            for _, col, val in cells:
                th = d.get(col)
                if val is None or th is None:
                    row.append("   -      -   ")
                    continue
                mm = max(mm, abs(val - th))
                row.append(f"{val:6.3f} ({th:5.3f})")
            o(f"      {diet:8} | " + " | ".join(row))
        o(f"      ==> G1b max |Δ| over every median cell: {mm:.4f}")
    o("")

    # ---------------- (a) the probe -------------------------------------------------- #
    o("-" * 100)
    o("(a) THE PROBE — median over slots of the held-out AUC, per trunk and diet. `n` is the")
    o("    number of slots contributing. `dp` is the free prior and is trunk-independent.")
    o("-" * 100)
    for k, r in seeds.items():
        c = collect(r)
        order = [x for x in ("frozen", "rand") if x in c] + \
                [x for x in c if x not in ("frozen", "rand")]
        o(f"  seed {k}   base rates: " + "  ".join(
            f"{d} {np.mean([rec['base'] for rec in r['trunks']['frozen']['per_buf'] if rec['which'] == d.split('/')[0]]):.3f}"
            for d in ("filed", "probe")))
        for diet in DIETS:
            if not any(diet in c[nm] for nm in c):
                continue
            o(f"    diet {diet}")
            o("      trunk          n |    pre   pmean    post   postm |    mlp      dp")
            for nm in order:
                d = c[nm].get(diet)
                if not d:
                    continue
                # `n` counts the slots whose AUC is DEFINED on the read of record, not the
                # slots present: on `probe/d` at seed 2 the disagreement-drawn held-out rows
                # solve so rarely that most slots have one class only and no AUC at all.
                n = len([v for _, _, v in d.get("post", []) if v is not None])
                o(f"      {nm:12} {n:3d} | " + " ".join(
                    fmt(med([v for _, _, v in d.get(rd, [])]), 7) for rd in READS)
                  + " | " + " ".join(
                    fmt(med([v for _, _, v in d.get(rd, [])]), 7) for rd in ("mlp", "dp")))
        o("")

    # ---------------- (b) per level -------------------------------------------------- #
    o("-" * 100)
    o("(b) THE READ OF RECORD (`post`) PER LEVEL — median over the slots of that rung.")
    o("    seed 0's ladder reaches L5; seed 2's reaches L4.")
    o("-" * 100)
    for k, r in seeds.items():
        c = collect(r)
        order = [x for x in ("frozen", "rand") if x in c] + \
                [x for x in c if x not in ("frozen", "rand")]
        lv = sorted({l for nm in c for d in c[nm].values() for _, l, _ in d.get("post", [])})
        for diet in ("filed", "probe"):
            o(f"  seed {k}  diet {diet}")
            o("      trunk        " + "  ".join(f"   L{l}   " for l in lv) +
              "   |    mlp per level")
            for nm in order:
                d = c[nm].get(diet)
                if not d:
                    continue
                cells = []
                for l in lv:
                    cells.append(med([v for _, ll, v in d.get("post", []) if ll == l]))
                ms = [med([v for _, ll, v in d.get("mlp", []) if ll == l]) for l in lv]
                o(f"      {nm:12} " + "  ".join(fmt(x, 8) for x in cells) + "   | " +
                  " ".join(fmt(x, 7) for x in ms))
            o("")

    # ---------------- (c) the world model -------------------------------------------- #
    o("-" * 100)
    o("(c) THE WORLD-MODEL DIAGNOSTICS, per trunk. `infill` is the HELD-OUT masked-infill")
    o("    cross-entropy and fill accuracy on FRESH corpus windows, at a FIXED window/mask set")
    o("    identical on every trunk. `parse` is the block head's level-1 parse accuracy against")
    o("    the exact features on VISIBLE blocks — an oracle instrument — read with one block")
    o("    masked (`mask1`, the `parse_features` idiom) and fully unmasked. `dpL` is the nested")
    o("    max-sum DP over the TRUE table at that rung, span masked, recovered level-1 features")
    o("    against the exact ones (2 nodes per rung).")
    o("-" * 100)
    for k, r in seeds.items():
        order = [x for x in ("frozen", "rand") if x in r["trunks"]] + \
                [x for x in r["trunks"] if x not in ("frozen", "rand")]
        lv = sorted({int(l) for nm in r["trunks"]
                     for l in r["trunks"][nm]["wm"]["dp_parse"]})
        o(f"  seed {k}")
        o("      trunk        infill_ce  infill_acc | parse_mask1 parse_nomask | " +
          " ".join(f"  dpL{l} " for l in lv))
        for nm in order:
            w = r["trunks"][nm]["wm"]
            o(f"      {nm:12} {w['infill_ce']:9.4f}  {w['infill_acc']:10.3f} | "
              f"{w['parse_acc_mask1']:11.3f} {w['parse_acc_nomask']:12.3f} | " +
              " ".join(fmt(w["dp_parse"].get(str(l), w["dp_parse"].get(l, {})).get("acc"), 7)
                       for l in lv))
        o("      held-out infill CE by number of masked blocks:")
        bk = sorted({b for nm in r["trunks"] for b in r["trunks"][nm]["wm"]["infill_by_nmask"]})
        o("      trunk        " + "  ".join(f"{b:>8}" for b in bk))
        for nm in order:
            d = r["trunks"][nm]["wm"]["infill_by_nmask"]
            o(f"      {nm:12} " + "  ".join(
                (f"{d[b]['ce']:8.4f}" if b in d else "       -") for b in bk))
        o("")

    # ---------------- (d) the transfer split ----------------------------------------- #
    o("-" * 100)
    o("(d) THE TRANSFER SPLIT — ONE readout over the read of record, pooled across slots, fit")
    o("    on the training rows of the L2/L3 slots and scored on the held-out rows of L2/L3 and")
    o("    of L4/L5, beside the same readout fit on L4/L5's own training rows. The two fits use")
    o("    the same number of rows (`n_fit`), so the comparison is not a sample-size one.")
    o("-" * 100)
    for k, r in seeds.items():
        order = [x for x in ("frozen", "rand") if x in r["trunks"]] + \
                [x for x in r["trunks"] if x not in ("frozen", "rand")]
        for diet in ("filed", "probe"):
            o(f"  seed {k}  diet {diet}")
            o("      trunk        n_fit |  L23->L23   L23->L45   L45->L45 |  n_hold L23 / L45"
              "   base L23 / L45")
            for nm in order:
                x = r["trunks"][nm]["xfer"].get(diet)
                if not x:
                    continue
                o(f"      {nm:12} {x['n_fit']:5d} | "
                  f"{fmt(x['fit23_on23']['auc'], 9)}  {fmt(x['fit23_on45']['auc'], 9)}  "
                  f"{fmt(x['fit45_on45']['auc'], 9)} |  "
                  f"{x['fit23_on23']['n']:6d} / {x['fit23_on45']['n']:6d}   "
                  f"{x['fit23_on23']['base']:.3f} / {x['fit23_on45']['base']:.3f}")
            o("")

    # ---------------- (e) the shaping ------------------------------------------------- #
    o("-" * 100)
    o("(e) THE SHAPING ITSELF — the outcome head's own within-batch BCE and AUC and the infill")
    o("    CE at the first and last logged step, the trunk's fingerprint (sum |parameter|,")
    o("    float64) after, and the cost. The frozen trunk's fingerprint is the baseline.")
    o("-" * 100)
    for k, r in seeds.items():
        o(f"  seed {k}   lr {r['shape']['lr']}   steps {r['shape']['steps']}   "
          f"slot_batch {r['shape']['slot_batch']}   infill_batch {r['shape']['infill_batch']}")
        if r["shape"].get("n_shape_buffers") is not None:
            o(f"      shaping diet: {r['shape']['n_shape_buffers']} buffers"
              + (f", levels {r['shape']['shape_levels']} ONLY (DESIGN §5a)"
                 if r["shape"].get("shape_levels") else ", every slot")
              + f"   dedup={r.get('dedup')}")
        o(f"      frozen fingerprint {r['trunks']['frozen']['fingerprint']:.6f}   "
          f"rand {r['trunks']['rand']['fingerprint']:.6f}")
        o("      arm          fingerprint    Δ vs frozen | out_bce first->last  out_auc "
          "first->last | infill_ce first->last | sec")
        for nm, h in r["shape"]["hist"].items():
            hs = h["hist"]
            if not hs:
                continue
            f0, f1 = hs[0], hs[-1]
            fp = r["trunks"][nm]["fingerprint"]
            d = fp - r["trunks"]["frozen"]["fingerprint"]

            def g(d0, d1, kk):
                if d0.get(kk) is None or (isinstance(d0[kk], float) and
                                          not np.isfinite(d0[kk])):
                    return "    -   ->    -   "
                return f"{d0[kk]:7.4f} -> {d1[kk]:7.4f}"
            o(f"      {nm:12} {fp:12.6f} {d:+11.6f} | {g(f0, f1, 'out_bce')}  "
              f"{g(f0, f1, 'out_auc')} | {g(f0, f1, 'infill_ce')} | {h['sec']:5.1f}")
        o(f"      job {r['sec']:.0f}s   peak RSS {r['peak_rss_mb']:.0f} MB   "
          f"peak GPU {r['peak_gpu_mb']:.0f} MB")
        o("")

    # ---------------- (f) cross-tag --------------------------------------------------- #
    others = [x for x in a.vs.split(",") if x]
    if others:
        tags = {a.tag: seeds}
        for t in others:
            dd = {}
            for k in ("s0", "s2"):
                pth = os.path.join(FIG, t, f"{k}.json")
                if os.path.isfile(pth):
                    dd[k] = json.load(open(pth))
            if dd:
                tags[t] = dd
        o("-" * 100)
        o("(f) CROSS-TAG. Same rows, same protocol, same trunks; the tags differ only in what")
        o("    the SHAPING saw. `du0` every slot, no filter; `du1` every slot, DESIGN §2.5's")
        o("    exact-duplicate filter; `du2` the filter plus the outcome loss heard at L2/L3")
        o("    ONLY (§5a). `frozen` and `rand` never see a verdict, so their movement between")
        o("    tags is the filter's effect on the PROBE's own fit and nothing else.")
        o("-" * 100)
        o("  THE READ OF RECORD (`post`), median over slots")
        for k in ("s0", "s2"):
            if not any(k in v for v in tags.values()):
                continue
            diets = ["filed", "probe"] + (["probe/u"] if any(
                "probe/u" in v[k]["trunks"]["frozen"]["median"]
                for v in tags.values() if k in v) else [])
            for diet in diets:
                o(f"    seed {k}  diet {diet}")
                names = []
                for v in tags.values():
                    if k in v:
                        names += [n for n in v[k]["trunks"] if n not in names]
                o("      trunk        " + "  ".join(f"{t:>9}" for t in tags))
                for nm in names:
                    cells = []
                    for t, v in tags.items():
                        m = (v.get(k, {}).get("trunks", {}).get(nm, {}) or {}).get(
                            "median", {}).get(diet, {})
                        cells.append(fmt(m.get("post"), 9))
                    o(f"      {nm:12} " + "  ".join(cells))
                o("")
        o("  THE TRANSFER SPLIT, fit L2/L3 -> scored L2/L3 | L4/L5, beside the L4/L5-own fit")
        for k in ("s0", "s2"):
            for diet in ("filed", "probe"):
                if not any(k in v for v in tags.values()):
                    continue
                o(f"    seed {k}  diet {diet}")
                o("      trunk        " + "  ".join(
                    f"{t}: 23->23  23->45  45->45" for t in tags))
                names = []
                for v in tags.values():
                    if k in v:
                        names += [n for n in v[k]["trunks"] if n not in names]
                for nm in names:
                    cells = []
                    for t, v in tags.items():
                        x = (v.get(k, {}).get("trunks", {}).get(nm, {}) or {}).get(
                            "xfer", {}).get(diet)
                        cells.append("  ".join(fmt(x[c]["auc"], 7) for c in
                                               ("fit23_on23", "fit23_on45", "fit45_on45"))
                                     if x else "    -        -        -   ")
                    o(f"      {nm:12} " + " | ".join(cells))
                o("")
        o("  CROSS-TAG IDENTITY GATE — `infill@N` is keyed on the step count alone, so the same")
        o("  arm must be the SAME TRUNK under every tag (fingerprint, sum |parameter|):")
        for k in ("s0", "s2"):
            for nm in ("infill@400", "infill@1600", "frozen", "rand"):
                fps = [(t, v[k]["trunks"][nm]["fingerprint"]) for t, v in tags.items()
                       if k in v and nm in v[k]["trunks"]]
                if len(fps) < 2:
                    continue
                sp = max(f for _, f in fps) - min(f for _, f in fps)
                o(f"      seed {k}  {nm:12} " + "  ".join(f"{t} {f:.6f}" for t, f in fps)
                  + f"   spread {sp:.2e}")
        o("")

    out = a.out or os.path.join(FIG, f"{a.tag}_reduction.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
