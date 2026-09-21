"""Local reduction: fetch the panels off the volume, replay the rule, write
`results/tables.md` (facts only) and `figs/*.png`.

Run (from experiments/):
  python3 -m rhm.practice.perception.analyze --tag pc0 --fetch --figs
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
REMOTE = "rhm_practice_perception"
VOL = "rhm-scaling-data"
LEVELS = (2, 3, 4, 5, 6)


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

def fetch(tag, cd_tag=None):
    cd_tag = cd_tag or f"perception_{tag}"
    os.makedirs(RAW, exist_ok=True)
    jobs = [(f"/{REMOTE}/{tag}/panel_corpus.json", "panel_corpus.json"),
            (f"/{REMOTE}/{tag}/panel_own.json", "panel_own.json"),
            (f"/{REMOTE}/{tag}/panel_corpus_feat.json", "panel_corpus_feat.json"),
            (f"/{REMOTE}/{tag}/panel_own_feat.json", "panel_own_feat.json"),
            (f"/{REMOTE}/{tag}/fidelity.json", "fidelity.json"),
            (f"/{REMOTE}/{tag}/record_meta.json", "record_meta.json"),
            (f"/{REMOTE}/{tag}/sweep.json", "sweep.json"),
            (f"/{REMOTE}/{tag}/reader_corpus/train_log.json", "train_corpus.json"),
            (f"/{REMOTE}/{tag}/reader_own/train_log.json", "train_own.json"),
            (f"/rhm_practice_conductor/{cd_tag}/setup.json", "cd_setup.json")]
    for src, dst in jobs:
        d = os.path.join(RAW, dst)
        if os.path.exists(d):
            os.remove(d)
        r = subprocess.run(["modal", "volume", "get", VOL, src, d],
                           capture_output=True, text=True,
                           env={**os.environ, "MODAL_PROFILE": "chromatic"})
        print(("  ok   " if r.returncode == 0 else "  MISS ") + src)


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def rank(x):
    x = np.asarray(x, float)
    o = x.argsort(kind="stable")
    r = np.empty_like(o, float)
    r[o] = np.arange(len(x), dtype=float)
    # average ties
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return corr(rank(a), rank(b))


def summarise_series(exact, cand):
    e, c = np.asarray(exact, float), np.asarray(cand, float)
    n = min(len(e), len(c))
    e, c = e[:n], c[:n]
    return {"pearson": corr(e, c), "spearman": spearman(e, c),
            "mean_exact": float(e.mean()), "mean_cand": float(c.mean()),
            "final_exact": float(e[-1]), "final_cand": float(c[-1]),
            "ratio_final": (float(c[-1] / e[-1]) if e[-1] else None),
            "mean_abs_err": float(np.abs(c - e).mean())}


# --------------------------------------------------------------------------- #
# the Q3 replay, on every candidate series
# --------------------------------------------------------------------------- #

def run_replays(panel, cd_results=None):
    from rhm.practice.perception.replay import compare, replay
    era = panel["era_by_cycle"]
    rl = panel["read_level"]
    n = panel["n_cycles"]
    active = [min(int(x), 6) for x in _active_from(panel)]
    floors = {3: panel["floors"]["yield_L3"], 4: panel["floors"]["yield_L4"]}
    cfg = panel["loop_cfg"]

    def series_for(block):
        return [-float(block[str(rl[i])][i]) for i in range(n)]

    out = {}
    cands = {"A_feature_true": panel["exact_feature_obs_hist"]}
    if "feature_endo_grid" in next(iter(panel["checkpoints"].values())):
        cands["Bf_feature_true_windowed"] = panel["feature_true_grid_windowed"]
        cands["Df_feature_shuffled"] = panel["feature_shuffled_grid"]
        for k, ck in panel["checkpoints"].items():
            cands[f"Cf_feature_endo@{k}"] = ck["feature_endo_grid"]
    else:
        cands["B_token_true"] = panel["token_true_grid"]
        cands["D_token_shuffled"] = panel["token_shuffled_grid"]
        for k, ck in panel["checkpoints"].items():
            cands[f"C_token_endo@{k}"] = ck["token_endo_grid"]
    from rhm.practice.perception.replay import in_series_floors
    traces, own_floors, own_traces = {}, {}, {}
    for name, block in cands.items():
        ser = series_for(block)
        acts, pol = replay(ser, era, rl, active, floors, span=cfg["loop_span"],
                           W=cfg["loop_W"], burn=cfg["loop_burn"], alpha=cfg["loop_alpha"])
        traces[name] = acts
        f2 = in_series_floors(pol)
        own_floors[name] = f2
        if f2:
            fl2 = {lv: max(v["v_tol"], 1e-9) for lv, v in f2.items()}
            for lv in (3, 4):
                fl2.setdefault(lv, floors[lv])
            a2, _ = replay(ser, era, rl, active, fl2, span=cfg["loop_span"],
                           W=cfg["loop_W"], burn=cfg["loop_burn"], alpha=cfg["loop_alpha"])
            own_traces[name] = a2
    ref = traces["A_feature_true"]
    for name, acts in traces.items():
        out[name] = {"actions": [(a["kind"], a["cycle"], a["level"]) for a in acts],
                     "vs_A": compare(ref, acts),
                     "in_series_floors": own_floors.get(name),
                     "actions_own_floor": [(a["kind"], a["cycle"], a["level"])
                                           for a in own_traces.get(name, [])],
                     "vs_A_own_floor": (compare(ref, own_traces[name])
                                        if name in own_traces else None)}
    out["_realised_chosen"] = [(a["kind"], a["cycle"], a.get("level"))
                               for a in (panel.get("loop_actions") or [])
                               if a.get("why") == "quiet" and not a.get("cancelled")]
    out["_realised_all"] = [(a["kind"], a["cycle"], a.get("level"), a.get("why"))
                            for a in (panel.get("loop_actions") or [])]
    out["_R1"] = (out["A_feature_true"]["actions"]
                  == [tuple(x) for x in out["_realised_chosen"]])
    return out, traces


def _active_from(panel):
    """`active` = the level being earned = era.level + 1, per cycle."""
    eras = panel["eras"]
    return [eras[e - 1]["level"] + 1 for e in panel["era_by_cycle"]]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def panels_floor(arm):
    return "0.461271, L4 0.516690"


def fmt(x, n=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def tables(tag, panels, extras, out_path):
    L = []
    W = L.append
    pc = panels.get("corpus")
    W(f"# perception — tables (tag `{tag}`)\n")
    W("Facts only; no interpretation. Figures: `figs/{f1_gauge, f2_reader_climb, "
      "f3_grid_quality, f4_replay}.png`.\n")
    W("## T0 — what the series are, and how to reproduce every number\n")
    W("The practice arc's yield thermostat reads `at_support`: distinct level-(active+1) "
      "tuples seen at least `mine_support` times over the learner's own solved "
      "trajectories. RHM hands that miner a GRID -- the solved piece is a whole aligned "
      "sequence, so `(era.level, era.node)` names which span of it to read. Here the grid "
      "is instead recovered from a next-token reader's own entropy period "
      "(`logit_reading/frontier/common.py::nested_phase`, template-free `meanprof` mode) "
      "on windows in which each mined piece sits at an offset the reader is not told.\n")
    W("| series | keying | grid | what it is |\n|---|---|---|---|")
    W("| **A** | level-1 features | true | the run's own `obs_hist` -- what the thermostat "
      "actually read |")
    W("| **Bf** | level-1 features | true | A recomputed on this window set (identity "
      "check) |")
    W("| **Cf** | level-1 features | recovered | **the endogenous gauge** |")
    W("| **Df** | level-1 features | shuffled offset | a wrong grid at the right period |")
    W("| **B / C / D** | leaf tokens | true / recovered / shuffled | the same three in leaf-"
      "token keying, on streams padded with the cycle's other solved beam tips |")
    W("\n```bash\ncd experiments            # MODAL_PROFILE=chromatic\n"
      "# the whole node, two waves across containers (~1.8 GPU-h, ~58 min wall)\n"
      "modal run --detach -m rhm.practice.perception.sweep::sweep --tag %s\n"
      "# the feature-keyed panels (mined-only streams, the thermostat's own coordinates)\n"
      "modal run -m rhm.practice.perception.evaluate::evaluate --tag %s --arm corpus \\\n"
      "    --stream-from mine --out-name panel_corpus_feat\n"
      "modal run -m rhm.practice.perception.evaluate::evaluate --tag %s --arm own \\\n"
      "    --stream-from mine --out-name panel_own_feat\n"
      "# tables and figures (local, CPU)\n"
      "python3 -m rhm.practice.perception.analyze --tag %s --fetch --figs\n```\n"
      % (tag, tag, tag, tag))

    # --- T1 provenance
    W("\n## T1 — what was run\n")
    rm = extras.get("record_meta") or {}
    W("| item | value |\n|---|---|")
    W(f"| practice arm re-run | `{rm.get('cd_tag')}` / `{rm.get('arm')}` |")
    W(f"| conductor literals | `cd_s0`'s, per conductor/FILES.md §Reproduce |")
    W(f"| cycles recorded | {rm.get('n_cycles')} |")
    W(f"| mined pieces recorded | {rm.get('n_mine')} |")
    W(f"| solved beam tips recorded | {rm.get('n_solved')} |")
    W(f"| practice re-run wall clock | {fmt(rm.get('elapsed_s'), 0)} s |")
    W(f"| practice re-run peak RSS | {fmt(rm.get('peak_rss_gb'), 2)} GB |")
    if pc:
        W(f"| reader window W | {pc['W']} tokens ({pc['W'] // 64} periods) |")
        W(f"| mine_support | {pc['support']} |")
        W(f"| evaluation windows (practice venue) | {pc['n_windows']} |")
        W(f"| evaluation windows (corpus venue) | {pc['n_corpus_windows']} |")
        W(f"| yield dead zones | L3 {pc['floors']['yield_L3']:.6f}, "
          f"L4 {pc['floors']['yield_L4']:.6f} |")
    for arm in ("corpus", "own"):
        tl = extras.get(f"train_{arm}")
        if tl:
            c = tl["config"]
            W(f"| reader `{arm}` | {c['steps']} steps, batch {c['batch_size']}, "
              f"stream {c.get('stream_meta', {}).get('source')}, "
              f"re-read x{c.get('reread_factor', 0):.1f} |")

    # --- T2 gates
    W("\n## T2 — gates\n")
    fd = extras.get("fidelity")
    W("| gate | what | result |\n|---|---|---|")
    if fd:
        worst = max((v["max_abs_delta"] or 0) for v in fd["series"].values())
        W(f"| F-1 | the recorded arm vs banked `cd_s0/{fd['arm']}`, "
          f"{len(fd['series'])} series | max abs delta {worst:.3e} |")
        W(f"| F-2 | realised loop actions equal | {fd['actions_equal']} |")
        W(f"| F-3 | recorded mine rows == run's own `n_mined`, every cycle | "
          f"{fd['mine_rows_match']} |")
    if pc:
        a = pc["exact_feature_obs_hist"]
        b = pc["feature_true_grid"]
        d = {k: max(abs(a[k][i] - b[k][i]) for i in range(min(len(a[k]), len(b[k]))))
             for k in b if k in a}
        W(f"| P-1 | `at_support` recomputed from the recorded level-1 parse on the true "
          f"grid vs the run's own `obs_hist` | max abs delta {max(d.values())} "
          f"(per level {d}) |")
    rep = extras.get("replays_corpus")
    if rep:
        W(f"| R-1 | the rule replayed on the exact series reproduces the realised "
          f"chosen `loop_actions` | {rep['_R1']} |")

    # --- T3/T4 reader
    for arm in ("corpus", "own", "corpus_feat", "own_feat"):
        p = panels.get(arm)
        if not p:
            continue
        W(f"\n## T3.{arm} — the reader `{arm}`: loss and altitude by checkpoint\n")
        W("| step | train loss | NLL corpus | NLL practice | H corpus | H practice |\n"
          "|---|---|---|---|---|---|")
        for k in sorted(p["checkpoints"], key=int):
            c = p["checkpoints"][k]
            W(f"| {k} | {fmt(c['train_loss'], 4)} | {fmt(c['nll_corpus'], 4)} | "
              f"{fmt(c['nll_practice'], 4)} | {fmt(c['H_corpus'], 4)} | "
              f"{fmt(c['H_practice'], 4)} |")

        for venue, key in (("corpus (held-out grammar windows)", "detector_corpus"),
                           ("practice productions (the mined pieces at random offsets)",
                            "detector_practice")):
            W(f"\n## T4.{arm}.{key[9:]} — nested period accuracy, {venue}\n")
            W("| step | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |\n|---|---|---|---|---|---|---|")
            for k in sorted(p["checkpoints"], key=int):
                a = p["checkpoints"][k][key]["acc_nested"]
                W(f"| {k} | " + " | ".join(fmt(a[str(j)], 3) for j in range(1, 7)) + " |")
            W("| chance | " + " | ".join(fmt(1.0 / 2 ** j, 3) for j in range(1, 7)) + " |")

        W(f"\n## T5.{arm} — the recovered grid, per level "
          f"(exact / right level wrong node / off grid)\n")
        W("| step | " + " | ".join(f"L{l}" for l in LEVELS) + " |\n|---|"
          + "---|" * len(LEVELS))
        for k in sorted(p["checkpoints"], key=int):
            gq = p["checkpoints"][k]["grid_quality"]
            W(f"| {k} | " + " | ".join(
                f"{gq[str(l)]['exact']:.3f} / {gq[str(l)]['right_level_wrong_node']:.3f}"
                f" / {gq[str(l)]['off_grid']:.3f}" for l in LEVELS) + " |")
        sq = p["shuffled_grid_quality"]
        W("| shuffled-offset floor | " + " | ".join(
            f"{sq[str(l)]['exact']:.3f} / {sq[str(l)]['right_level_wrong_node']:.3f}"
            f" / {sq[str(l)]['off_grid']:.3f}" for l in LEVELS) + " |")

        W(f"\n## T6.{arm} — the endo read's positions: the parent span of each era's own "
          f"cell\n")
        W("| step | " + " | ".join(
            f"era {e}" for e in sorted(
                p['checkpoints'][sorted(p['checkpoints'], key=int)[0]]
                ['grid_quality_by_era_parent_span'], key=int)) + " |")
        eras_k = sorted(p["checkpoints"][sorted(p["checkpoints"], key=int)[0]]
                        ["grid_quality_by_era_parent_span"], key=int)
        W("|---|" + "---|" * len(eras_k))
        for k in sorted(p["checkpoints"], key=int):
            g = p["checkpoints"][k]["grid_quality_by_era_parent_span"]
            W(f"| {k} | " + " | ".join(
                f"{g[e]['frac_exact']:.3f} / {g[e]['frac_aligned']:.3f}" for e in eras_k)
              + " |")
        W("\n(`exact` / `aligned`: the recovered span is the era's own parent span / is a "
          "genuine constituent of the parent's level at some node. Parent level per era: "
          + ", ".join(f"era {e}: L{g[e]['parent_level']}" for e in eras_k) + ".)")

        # --- the gauge itself
        W(f"\n## T7.{arm} — `at_support` series: how the endogenous gauge tracks the exact "
          f"one\n")
        W("Series A = the thermostat's own (level-1 feature tuples, true grid). "
          "B = leaf-token tuples, true grid. C = leaf-token tuples, recovered grid. "
          "D = leaf-token tuples, shuffled-offset grid.\n")
        W("| level | series | pearson r vs A | spearman rho vs A | mean | final | "
          "pearson r vs B | spearman rho vs B |\n|---|---|---|---|---|---|---|---|")
        A = p["exact_feature_obs_hist"]
        feat = "feature_endo_grid" in next(iter(p["checkpoints"].values()))
        if feat:
            W("\n**Feature keying** (the thermostat's own coordinates; the stream is built "
              "from mined pieces only so the recorded level-1 parse covers it). "
              "Bf = feature tuples, true grid, this window set. Cf = feature tuples, "
              "recovered grid. Df = feature tuples, shuffled-offset grid.\n")
            W("| level | series | pearson r vs A | spearman rho vs A | mean | final |\n"
              "|---|---|---|---|---|---|")
            rowsf = [("Bf feature-true (windowed)", p["feature_true_grid_windowed"])]
            for k in sorted(p["checkpoints"], key=int):
                rowsf.append((f"Cf endo@{k}", p["checkpoints"][k]["feature_endo_grid"]))
            rowsf.append(("Df shuffled", p["feature_shuffled_grid"]))
            for l in (3, 4, 5, 6):
                sl = str(l)
                for name, blk in rowsf:
                    if blk is None or sl not in blk:
                        continue
                    v = summarise_series(A[sl], blk[sl])
                    W(f"| {l} | {name} | {fmt(v['pearson'])} | {fmt(v['spearman'])} | "
                      f"{fmt(v['mean_cand'], 1)} | {fmt(v['final_cand'], 1)} |")
                W(f"| {l} | A feature-true (reference) | 1.000 | 1.000 | "
                  f"{np.mean(A[sl]):.1f} | {A[sl][-1]:.0f} |")
            W(f"\n(pieces dropped for block-parity failure: true grid "
              f"{p.get('feature_true_grid_windowed_dropped')}, shuffled "
              f"{p.get('feature_shuffled_grid_dropped')}, endo per checkpoint "
              + ", ".join(f"{k}:{p['checkpoints'][k]['feature_endo_dropped']}"
                          for k in sorted(p["checkpoints"], key=int)) + ")")
        B = p["token_true_grid"]
        for l in LEVELS:
            sl = str(l)
            rows = [("B token-true", B.get(sl))]
            for k in sorted(p["checkpoints"], key=int):
                rows.append((f"C endo@{k}", p["checkpoints"][k]["token_endo_grid"].get(sl)))
            rows.append(("D shuffled", p["token_shuffled_grid"].get(sl)))
            for name, ser in rows:
                if ser is None:
                    continue
                va = summarise_series(A[sl], ser) if sl in A else {}
                vb = summarise_series(B[sl], ser)
                W(f"| {l} | {name} | {fmt(va.get('pearson'))} | {fmt(va.get('spearman'))} "
                  f"| {fmt(vb['mean_cand'], 1)} | {fmt(vb['final_cand'], 1)} | "
                  f"{fmt(vb['pearson'])} | {fmt(vb['spearman'])} |")
            if sl in A:
                va = summarise_series(A[sl], B[sl])
                W(f"| {l} | A feature-true (reference) | 1.000 | 1.000 | "
                  f"{fmt(va['mean_exact'], 1)} | {fmt(va['final_exact'], 1)} | "
                  f"{fmt(va['pearson'])} | {fmt(va['spearman'])} |")

        W(f"\n## T8.{arm} — the read the thermostat actually consumes "
          f"(`at_support` at `read_level`), per era\n")
        rl = p["read_level"]
        era = p["era_by_cycle"]
        eras = sorted(set(era))
        kk = "feature" if feat else "token"
        blocks = ([p["exact_feature_obs_hist"],
                   p["feature_true_grid_windowed"] if feat else p["token_true_grid"]]
                  + [p["checkpoints"][k][f"{kk}_endo_grid"]
                     for k in sorted(p["checkpoints"], key=int)]
                  + [p["feature_shuffled_grid"] if feat else p["token_shuffled_grid"]])
        W(f"| era (read level) | A feature-true | {'Bf' if feat else 'B'} true grid | "
          + " | ".join(f"{'Cf' if feat else 'C'} endo@{k}"
                       for k in sorted(p["checkpoints"], key=int))
          + f" | {'Df' if feat else 'D'} shuffled |\n|---|---|"
          + "---|" * (len(p["checkpoints"]) + 2))
        for e in eras:
            idx = [i for i, q in enumerate(era) if q == e]
            lv = sorted({rl[i] for i in idx})
            cells = []
            for block in blocks:
                vals = [block[str(rl[i])][i] for i in idx if str(rl[i]) in block]
                cells.append(f"{np.mean(vals):.1f} → {vals[-1]:.0f}" if vals else "-")
            W(f"| {e} (L{'/'.join(map(str, lv))}) | " + " | ".join(cells) + " |")

    # --- T9/T10 the replay
    for arm in ("corpus", "own", "corpus_feat", "own_feat"):
        rep = extras.get(f"replays_{arm}")
        if not rep:
            continue
        W(f"\n## T9.{arm} — Q3: the rule replayed on each series, yoked to the realised run\n")
        W("| series | actions (kind, cycle, level) |\n|---|---|")
        W(f"| realised `loop_actions`, all | {rep['_realised_all']} |")
        W(f"| realised, chosen only (what a rule can reproduce) | "
          f"{rep['_realised_chosen']} |")
        for name in sorted(k for k in rep if not k.startswith('_')):
            W(f"| {name} | {rep[name]['actions']} |")
        W(f"\n## T9b.{arm} — the dead zone the same null-ABBA method measures ON each "
          f"series, and the rule replayed at it\n")
        W("| series | sd(N) L3 | v_tol L3 | sd(N) L4 | v_tol L4 | actions at its own "
          "dead zone |\n|---|---|---|---|---|---|")
        for name in sorted(k for k in rep if not k.startswith('_')):
            f2 = rep[name].get("in_series_floors") or {}
            g = lambda lv, q: fmt((f2.get(str(lv)) or f2.get(lv) or {}).get(q), 4)
            W(f"| {name} | {g(3, 'sd_N')} | {g(3, 'v_tol')} | {g(4, 'sd_N')} | "
              f"{g(4, 'v_tol')} | {rep[name].get('actions_own_floor')} |")
        W(f"\n(the governing dead zones, measured on the exact series by `cd_ef`: "
          f"L3 {panels_floor(arm)}.)")

        W(f"\n## T10.{arm} — divergence from the exact series, action by action\n")
        W("| series | kind | level | nth | exact cycle | this cycle | delta (cycles) |\n"
          "|---|---|---|---|---|---|---|")
        for name in sorted(k for k in rep if not k.startswith('_')):
            if name == "A_feature_true":
                continue
            for r in rep[name]["vs_A"]["rows"]:
                W(f"| {name} | {r['kind']} | {fmt(r['level'])} | {r['nth']} | "
                  f"{fmt(r['exact_cycle'])} | {fmt(r['endo_cycle'])} | {fmt(r['delta'])} |")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out_path} ({len(L)} lines)")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, panels, extras, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    p = panels.get("corpus_feat") or panels.get("corpus")
    if not p:
        return
    steps = sorted(p["checkpoints"], key=int)
    feat = "feature_endo_grid" in p["checkpoints"][steps[0]]
    ekey = "feature_endo_grid" if feat else "token_endo_grid"
    tkey = "feature_true_grid_windowed" if feat else "token_true_grid"
    skey = "feature_shuffled_grid" if feat else "token_shuffled_grid"
    acts = [q for q in (p.get("loop_actions") or []) if q.get("why") == "quiet"]
    c = np.arange(1, p["n_cycles"] + 1)

    # F1 — the gauge, exact vs endogenous vs floors, at the two read levels
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for j, l in enumerate((3, 4)):
        sl, a = str(l), ax[j]
        a.plot(c, p["exact_feature_obs_hist"][sl], "k-", lw=2.4,
               label="A  exact, true grid (what the thermostat read)")
        for i, k in enumerate(steps):
            a.plot(c, p["checkpoints"][k][ekey][sl], lw=1.3,
                   color=plt.cm.viridis(i / max(len(steps) - 1, 1)),
                   label=f"endogenous grid @ reader step {k}")
        a.plot(c, p[skey][sl], color="crimson", lw=1.4, ls=":",
               label="shuffled-offset grid (wrong grid, right period)")
        for q in acts:
            a.axvline(q["cycle"], color="0.82", lw=1.0, zorder=0)
        a.set_title(f"at_support, level {l}"
                    + ("   (era 1's read)" if l == 3 else "   (eras 2-5's read)"))
        a.set_xlabel("practice cycle")
        if j == 0:
            a.set_ylabel("distinct tuples at support")
            a.legend(fontsize=6.6, loc="upper left")
    fig.suptitle("The learner's next-level minability, counted on a grid its own reader "
                 "recovered (grey lines = the realised loop's chosen actions)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/f1_gauge.png", dpi=140); plt.close(fig)

    # F2 — the reader's climb, both diets, both venues
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    panes = [("corpus_feat", "detector_corpus", "grammar reader — grammar venue"),
             ("corpus_feat", "detector_practice", "grammar reader — own-production venue"),
             ("own_feat", "detector_practice", "own-production reader — own-production venue"),
             ("own_feat", "detector_corpus", "own-production reader — grammar venue")]
    for j, (arm, key, ttl) in enumerate(panes):
        q = panels.get(arm)
        if not q:
            continue
        st = sorted(q["checkpoints"], key=int)
        for k in range(1, 7):
            ax[j].plot([int(x) for x in st],
                       [q["checkpoints"][x][key]["acc_nested"][str(k)] for x in st],
                       "o-", ms=3.4, color=plt.cm.plasma((k - 1) / 5), label=f"k={k}")
            ax[j].axhline(1.0 / 2 ** k, color=plt.cm.plasma((k - 1) / 5), lw=0.5, ls=":")
        ax[j].set_xscale("symlog", linthresh=250); ax[j].set_ylim(0, 1.03)
        ax[j].set_xlim(0, max(int(x) for x in st) * 1.15)
        ax[j].set_title(ttl, fontsize=9); ax[j].set_xlabel("reader step")
        if j == 0:
            ax[j].set_ylabel("nested period accuracy (dotted = chance)")
            ax[j].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/f2_reader_climb.png", dpi=140); plt.close(fig)

    # F3 — what the recovered grid is good for, level by level
    fig, ax = plt.subplots(1, len(LEVELS), figsize=(3.2 * len(LEVELS), 3.9), sharey=True)
    for j, l in enumerate(LEVELS):
        st = [int(x) for x in steps]
        gq = [p["checkpoints"][x]["grid_quality"][str(l)] for x in steps]
        ex = [g["exact"] for g in gq]
        al = [g["exact"] + g["right_level_wrong_node"] for g in gq]
        ax[j].fill_between(st, 0, ex, color="#2a6f97", alpha=.9, label="the era's own span")
        ax[j].fill_between(st, ex, al, color="#a9d6e5", alpha=.9,
                           label="a real constituent, wrong node")
        ax[j].fill_between(st, al, 1, color="#ececec", label="off the grid")
        ax[j].axhline(p["shuffled_grid_quality"][str(l)]["exact"] +
                      p["shuffled_grid_quality"][str(l)]["right_level_wrong_node"],
                      color="crimson", lw=1.1, ls=":", label="shuffled-offset floor")
        ax[j].set_xscale("symlog", linthresh=250); ax[j].set_ylim(0, 1)
        ax[j].set_xlim(0, max(st) * 1.05)
        ax[j].set_title(f"level {l}", fontsize=10); ax[j].set_xlabel("reader step")
        if j == 0:
            ax[j].set_ylabel("fraction of mined pieces"); ax[j].legend(fontsize=6.4)
    fig.tight_layout(); fig.savefig(f"{outdir}/f3_grid_quality.png", dpi=140); plt.close(fig)

    # F4 — Q3, the replayed decisions on a timeline
    rows = []
    for arm, lab in (("corpus_feat", "grammar reader"), ("own_feat", "own-prod reader")):
        rep = extras.get(f"replays_{arm}")
        if not rep:
            continue
        for name in sorted((k for k in rep if k.startswith("Cf_") or k.startswith("C_")),
                           key=lambda q: int(q.split("@")[1])):
            rows.append((f"{lab} @{name.split('@')[1]}", rep[name]["actions"]))
    ref = extras.get("replays_corpus_feat", {}).get("A_feature_true", {}).get("actions", [])
    shuf = extras.get("replays_corpus_feat", {}).get("Df_feature_shuffled",
                                                     {}).get("actions", [])
    rows = [("exact (A)", ref)] + rows + [("shuffled-offset grid", shuf)]
    fig, a = plt.subplots(figsize=(12, 0.42 * len(rows) + 2.2))
    style = {"commit": ("o", "#1b4965"), "advance": ("s", "#e07a5f")}
    for i, (lab, acts) in enumerate(rows):
        y = len(rows) - i
        a.axhline(y, color="0.93", lw=0.8, zorder=0)
        for kind, cyc, lv in acts:
            m, col = style[kind]
            a.plot(cyc, y, m, ms=7, color=col, mec="white", mew=0.7)
            if kind == "commit":
                a.annotate(f"L{lv}", (cyc, y), textcoords="offset points",
                           xytext=(0, 7), ha="center", fontsize=6)
    for q in acts if False else [x[1] for x in ref]:
        a.axvline(q, color="0.85", lw=0.9, ls="--", zorder=0)
    a.set_yticks(range(1, len(rows) + 1))
    a.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    a.set_xlabel("practice cycle"); a.set_xlim(0, p["n_cycles"] + 2)
    a.plot([], [], "o", color="#1b4965", label="commit"); 
    a.plot([], [], "s", color="#e07a5f", label="era advance")
    a.legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(left=0.22)
    a.set_title("Q3 — the same thermostat, the same dead zones, driven by each series "
                "(dashed = the exact series' decisions)", fontsize=9)
    fig.savefig(f"{outdir}/f4_replay.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote figures -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pc0")
    ap.add_argument("--cd-tag", default="")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag, a.cd_tag or None)
    panels = {arm: load(f"panel_{arm}.json")
              for arm in ("corpus", "own", "corpus_feat", "own_feat")}
    panels = {k: v for k, v in panels.items() if v}
    extras = {"record_meta": load("record_meta.json"), "fidelity": load("fidelity.json"),
              "train_corpus": load("train_corpus.json"), "train_own": load("train_own.json"),
              "cd_setup": load("cd_setup.json")}
    for arm, p in panels.items():
        rep, _ = run_replays(p)
        extras[f"replays_{arm}"] = rep
    tables(a.tag, panels, extras, os.path.join(HERE, "results", "tables.md"))
    with open(os.path.join(RAW, "replays.json"), "w") as f:
        json.dump({k: v for k, v in extras.items() if k.startswith("replays")}, f, indent=1)
    if a.figs:
        figures(a.tag, panels, extras, os.path.join(HERE, "figs"))


if __name__ == "__main__":
    main()
