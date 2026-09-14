"""Reduce a dispositional run's results.json to the tables and figures.

Run locally (CPU) after fetching the results:

    modal volume get language-reduction-data \
        /a2a_forward/confabulation/dispositional/main/results.json .
    python3 -m a2a_forward.fsm_part3.dispositional_language.reduce_dispositional \
        --results results.json --tag main

Writes `figures/<tag>_reduction.txt` plus the plots. Reduction only -- no
interpretation; the README is written after the numbers are discussed.
"""

import argparse
import json
import os


FAM_ORDER = {"IMPL": 0, "BEHAV": 1}
DIR_ORDER = {"prosp": 0, "retro": 1}


def _fmt(v, w=7, p=3, sign=False):
    if v is None or v != v:
        return " " * (w - 3) + "n/a"
    return f"{v:+{w}.{p}f}" if sign else f"{v:{w}.{p}f}"


def primary_rows(res, horizons, retro_w, cap):
    """The four headline targets in a fixed order."""
    k0 = horizons[0]
    return [
        (f"IMPL_PROSP_k{k0}_{cap}", "IMPL", "prospective"),
        (f"BEHAV_PROSP_k{k0}", "BEHAV", "prospective"),
        (f"IMPL_RETRO_w{retro_w}_{cap}", "IMPL", "retrospective"),
        (f"BEHAV_RETRO_w{retro_w}", "BEHAV", "retrospective"),
    ]


def best_public(row, idx):
    """Best score over EVERY third-party observer that sees only what an outsider sees:
    the single-snapshot I/O observers (paper 2's `O_io`) and the history observer
    (`O_hist`, which also sees M's own output summary at earlier checkpoints). O_hist is
    itself an I/O-map observer -- it reads M's outputs, just over time -- so for a
    DISPOSITIONAL target it is a legitimate comparator, which it would not have been for
    paper 2's occurrent one. O_act is excluded: it is handed M's activations."""
    vals = [v[idx] for k, v in row["observers"].items()
            if (k.startswith("O_io") or k.startswith("O_hist")) and "half" not in k]
    return max(vals) if vals else float("nan")


def frac_public(row, idx):
    """component_control's `frac`, with the self-report as the ceiling: the fraction of
    the self's above-chance score that the best third party reaches. 1.0 = fully public,
    0 = nothing of the fact is reachable from outside. The right statistic when the self
    is far from saturation, which it is on every dispositional target here."""
    ch = row["chance_heldck"] if idx == 1 else row["chance_seq"]
    self_ = row["self_lnf_heldck"] if idx == 1 else row["self_lnf_seq"]
    bp = best_public(row, idx)
    return (bp - ch) / (self_ - ch) if (self_ - ch) > 1e-6 else float("nan")


def obs_get(row, prefix, idx):
    """Best score over observers whose key starts with `prefix`; idx 0 = held-out
    sequences at train checkpoints, 1 = held-out sequences at held-out checkpoints."""
    vals = [v[idx] for k, v in row["observers"].items()
            if k.startswith(prefix) and "half" not in k]
    return max(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args()

    R = json.load(open(a.results))
    cfg, T = R["config"], R["targets"]
    guards, ac, occ = R["guards"], R["autocorr"], R["occurrent"]
    steps = cfg["steps_list"]
    hold = cfg["hold_ckpts"]
    horizons = [int(z) for z in cfg["horizons_str"].split(",")]
    retro_w = cfg["retro_w"]
    cap_tags = [f"h{c.split(':')[0]}m{float(c.split(':')[1]):g}"
                for c in cfg["inst_caps_str"].split(",")]
    cap0 = cap_tags[0]
    L = []
    P = L.append

    P("=" * 96)
    P(f"DISPOSITIONAL SELF-KNOWLEDGE -- reduction  [tag={a.tag}]")
    P("=" * 96)
    P(f"substrate     : {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D GPT, OL arm, "
      f"{cfg['n_steps']} steps on {cfg['n_tokens']:,} FineWeb-Edu tokens")
    P(f"gap / report  : {cfg['predict_from']} -> {cfg['predict_to']}, report from "
      f"{cfg['report_block'] or 'post_block' + str(cfg['n_layer'] - 1)}")
    P(f"checkpoints   : {len(steps)} log-spaced, {steps}")
    P(f"held-out block: {hold}  (steps {[steps[c] for c in hold]})")
    P(f"report set    : N={cfg['n_report_sequences']} held-out sequences, same at every "
      f"checkpoint; 80/20 sequence split")
    P(f"instruments   : {cap_tags}, ens_n={cfg['ens_n']}, fresh_fm_steps="
      f"{cfg['fresh_fm_steps']}")
    P(f"heads/obs     : {cfg['head_steps']}/{cfg['obs_steps']} steps (matched exposure), "
      f"observers {cfg['observer_caps']}, hist_lags={cfg['hist_lags']}")
    P(f"wall / peak   : {cfg.get('wall_s', 0) / 60:.0f} min, "
      f"{cfg.get('peak_rss_gb', float('nan')):.1f} GB RSS")
    P("")

    # ---------------- 1. guards ----------------
    P("-" * 96)
    P("1. JUNK-RESIDUAL GUARDS, per checkpoint")
    P("   A saturated instrument counterfeits the whole result: M can report noise and no")
    P("   observer can predict noise. ens_cos = do independent fresh FMs miss the SAME")
    P("   thing (high = a real computational gap). eta2_norm = is |r| conditioned on the")
    P("   syntactic type of the position. floor = the per-position instrument no-change")
    P("   level, i.e. 1 - cos between two fresh FMs at the SAME checkpoint.")
    P("-" * 96)
    for ct in cap_tags:
        P(f"  instrument {ct}  ({guards[ct]['0']['pct_of_predicted']:.1f}% of the "
          f"predicted blocks, {guards[ct]['0']['fm_params'] / 1e3:.0f}K params)")
        P(f"    {'ck':>3s} {'step':>6s} {'val':>7s} {'FM cos':>8s} {'|r|':>7s} "
          f"{'ens_cos':>8s} {'eta2_n':>8s} {'eta2_dir':>9s} {'floor':>7s}")
        for ci in range(len(steps)):
            g = guards[ct][str(ci)]
            mark = " <-- held out" if ci in hold else ""
            P(f"    {ci:3d} {g['step']:6d} {R['val_loss_traj'][ci]:7.3f} "
              f"{g['fwd_cosine']:8.4f} {g['res_norm']:7.3f} {g['ens_cos']:8.3f} "
              f"{g['eta2_norm']:8.4f} {g['eta2_dir']:9.5f} {g['floor_mean']:7.4f}{mark}")
        P("")
    P("  per-category Cohen's d on |r| at the final checkpoint, default instrument")
    P("  (paper 2 published: sentence_start -0.85, after_punct -0.62, after_opener +0.62,")
    P("   before_closer +0.84):")
    cats = {"1": "sentence_start", "2": "after_punct", "3": "after_opener",
            "4": "before_closer"}
    gd = guards[cap0][str(len(steps) - 1)]["cohens_d"]
    P("    " + "  ".join(f"{cats[k]}={gd[k]:+.2f}" for k in cats if k in gd))
    P("")

    # ---------------- 2. autocorrelation gate ----------------
    P("-" * 96)
    P("2. AUTOCORRELATION GATE  (committee_head Phase B: probe-able != learnable from a")
    P("   given target). r1 = lag-1 Pearson across consecutive checkpoints over positions;")
    P("   rho1 its rank version; btw = fraction of the target's variance that is a")
    P("   per-position constant (high = the head can win by learning a static fact).")
    P("-" * 96)
    P(f"  {'target':38s} {'r1':>7s} {'rho1':>7s} {'btw/tot':>8s} {'n_ck':>5s}")
    for nm, v in ac.items():
        P(f"  {nm:38s} {v['r1']:+7.3f} {v['rho1']:+7.3f} {v['between_frac']:8.3f} "
          f"{v['n_ckpt']:5d}")
    P("")

    # ---------------- 3. headline ----------------
    P("-" * 96)
    P("3. THE HEADLINE: advantage per target per direction")
    P("   All columns are held-out SEQUENCES at the HELD-OUT CHECKPOINT BLOCK (the strict")
    P("   evaluation) unless marked (seq), which is held-out sequences at the training")
    P("   checkpoints. Two advantage statistics are given, because the dispositional")
    P("   setting admits a third-party observer paper 2's occurrent setting did not:")
    P("   O_hist sees M's own output summary at checkpoints c, c-1, ..., c-%d."
      % cfg["hist_lags"])
    P("   O_act (handed M's early activations) stays a ceiling and is never an advantage")
    P("   comparator.")
    P("-" * 96)
    hdr = (f"  {'target':30s} {'fam':>5s} {'direction':>13s} {'chance':>7s} "
           f"{'self':>7s} {'O_io':>7s} {'O_hist':>7s} {'O_act':>7s} | "
           f"{'ADVio':>7s} {'ADVpub':>7s} {'fracpub':>8s} | {'ADVio(seq)':>11s}")

    def hrow(nm, fam, dr, chance_zero=False):
        r = T[nm]
        ch = 0.0 if chance_zero else r["chance_heldck"]
        bp = best_public(r, 1)
        return (f"  {nm:30s} {fam:>5s} {dr:>13s} {ch:7.3f} "
                f"{r['self_lnf_heldck']:7.3f} {r['best_O_io_heldck']:7.3f} "
                f"{obs_get(r, 'O_hist', 1):7.3f} {obs_get(r, 'O_act', 1):7.3f} | "
                f"{r['adv_heldck']:+7.3f} {r['self_lnf_heldck'] - bp:+7.3f} "
                f"{frac_public(r, 1):8.3f} | {r['adv_seq']:+11.3f}")

    P("   ADVio  = self - best single-snapshot O_io  (paper 2's statistic)")
    P("   ADVpub = self - best THIRD-PARTY observer, i.e. max(O_io, O_hist). O_hist reads")
    P("            only M's outputs, just over several snapshots, so for a dispositional")
    P("            target it is a legitimate comparator; for paper 2's occurrent target it")
    P("            would not have been. This is the conservative statistic here.")
    P("   fracpub= (best third party - chance) / (self - chance): the fraction of the")
    P("            self's above-chance score reachable from outside. 1.0 = fully public.")
    P("            component_control's `frac`, with the self as ceiling (the self is far")
    P("            from saturation on every dispositional target, so raw advantage alone")
    P("            confounds privacy with how well the self can report the target at all).")
    P("   Columns are held-out sequences @ HELD-OUT CHECKPOINTS unless marked (seq).")
    P("-" * 96)
    P("  CATEGORICAL (within-checkpoint quartile class, 4-way accuracy)")
    P(hdr)
    for nm, fam, dr in primary_rows(R, horizons, retro_w, cap0):
        if nm in T:
            P(hrow(nm, fam, dr))
    P("")
    P("  CONTINUOUS (rank-loss head, Spearman rho within checkpoint; chance = 0)")
    P(hdr)
    for nm, fam, dr in primary_rows(R, horizons, retro_w, cap0):
        if nm + "_RANK" in T:
            P(hrow(nm + "_RANK", fam, dr, chance_zero=True))
    P("")
    P("  the 2x2, as ADVpub (left) and fracpub (right):")
    for lab, suf in (("categorical", ""), ("continuous ", "_RANK")):
        cells = {}
        for nm, fam, dr in primary_rows(R, horizons, retro_w, cap0):
            if nm + suf in T:
                r = T[nm + suf]
                cells[(fam, dr)] = (r["self_lnf_heldck"] - best_public(r, 1),
                                    frac_public(r, 1))
        P(f"    {lab}      {'IMPL':>18s} {'BEHAV':>18s}")
        for dr in ("prospective", "retrospective"):
            row = f"    {dr:15s}"
            for fam in ("IMPL", "BEHAV"):
                c = cells.get((fam, dr))
                row += (f" {c[0]:+9.3f} / {c[1]:5.2f}" if c else f" {'n/a':>18s}")
            P(row)
        P("")

    # ---------------- 4. the observer ladder on the primary rows ----------------
    P("-" * 96)
    P("4. THE OBSERVER LADDER (strict column). A flat ladder says the I/O map has been")
    P("   exhausted; a climbing one says the fact was within reach all along.")
    P("-" * 96)
    keys = []
    for nm, _, _ in primary_rows(R, horizons, retro_w, cap0):
        if nm in T:
            keys = list(T[nm]["observers"].keys())
            break
    P(f"  {'target':30s} " + " ".join(f"{k.replace('@', ' '):>22s}" for k in keys))
    for nm, _, _ in primary_rows(R, horizons, retro_w, cap0):
        if nm not in T:
            continue
        o = T[nm]["observers"]
        P(f"  {nm:30s} " + " ".join(
            f"{(o[k][1] if k in o else float('nan')):22.3f}" for k in keys))
    P("")
    P("  self-report head variants (strict column):")
    P(f"  {'target':30s} {'MLP/ln_f':>10s} {'MLP/raw':>10s} {'linear':>10s}")
    for nm, _, _ in primary_rows(R, horizons, retro_w, cap0):
        if nm not in T:
            continue
        r = T[nm]
        P(f"  {nm:30s} {_fmt(r.get('self_lnf_heldck'), 10)} "
          f"{_fmt(r.get('self_raw_heldck'), 10)} {_fmt(r.get('self_linear_heldck'), 10)}")
    P("")

    # ---------------- 5. instrument-variation readings ----------------
    P("-" * 96)
    P("5. THE THREE READINGS OF 'MY COMPUTATION REORGANIZED'  (strict column)")
    P("   raw    = 1 - cos(r_c, r_c+k), two different fresh instruments")
    P("   excess = raw minus the per-position instrument no-change floor at c")
    P("   fixed  = the instrument trained at c, applied at c+k (one instrument, both ends)")
    P("-" * 96)
    P(f"  {'reading':34s} {'chance':>7s} {'self':>7s} {'O_io':>7s} {'O_hist':>7s} "
      f"{'ADVio':>7s} {'ADVpub':>7s} {'fracpub':>8s} {'r1':>7s}")
    k0 = horizons[0]
    for ct in cap_tags:
        for label, nm in (("raw", f"IMPL_PROSP_k{k0}_{ct}"),
                          ("excess over floor", f"IMPL_PROSPEXC_k{k0}_{ct}"),
                          ("fixed instrument", f"IMPL_PROSPFIX_k{k0}_{ct}")):
            if nm not in T:
                continue
            r = T[nm]
            bp = best_public(r, 1)
            P(f"  {ct + ' / ' + label:34s} {r['chance_heldck']:7.3f} "
              f"{r['self_lnf_heldck']:7.3f} {r['best_O_io_heldck']:7.3f} "
              f"{obs_get(r, 'O_hist', 1):7.3f} {r['adv_heldck']:+7.3f} "
              f"{r['self_lnf_heldck'] - bp:+7.3f} {frac_public(r, 1):8.3f} "
              f"{ac.get(nm, {}).get('r1', float('nan')):+7.3f}")
    P("")

    # ---------------- 6. every laddered target ----------------
    P("-" * 96)
    P("6. EVERY LADDERED TARGET (strict column), sorted family then direction")
    P("-" * 96)
    P(f"  {'target':38s} {'fam':>5s} {'dir':>6s} {'chance':>7s} {'self':>7s} "
      f"{'O_io':>7s} {'O_hist':>7s} {'O_act':>7s} {'ADVio':>7s} {'ADVpub':>7s} "
      f"{'fracpub':>8s} {'r1':>7s}")
    rows = sorted(T.items(), key=lambda kv: (FAM_ORDER.get(kv[1]["family"], 9),
                                             DIR_ORDER.get(kv[1]["direction"], 9), kv[0]))
    for nm, r in rows:
        bp = best_public(r, 1)
        P(f"  {nm:38s} {r['family']:>5s} {r['direction']:>6s} {r['chance_heldck']:7.3f} "
          f"{r['self_lnf_heldck']:7.3f} {r['best_O_io_heldck']:7.3f} "
          f"{obs_get(r, 'O_hist', 1):7.3f} {obs_get(r, 'O_act', 1):7.3f} "
          f"{r['adv_heldck']:+7.3f} {r['self_lnf_heldck'] - bp:+7.3f} "
          f"{frac_public(r, 1):8.3f} "
          f"{ac.get(nm.replace('_RANK', ''), {}).get('r1', float('nan')):+7.3f}")
    P("")

    # ---------------- 7. occurrent controls ----------------
    P("-" * 96)
    P("7. PAPER 2'S OCCURRENT BATTERY AT THE FINAL CHECKPOINT")
    P("   The dissociation shown alive on THIS model, so the dispositional rows are read")
    P("   against a known-good occurrent baseline rather than a different training run.")
    P("   Published (paper 2, language): IMPL +0.272, IMPL_COS +0.318, BEHAV -0.067,")
    P("   ENT -0.287, WORLD -0.009.")
    P("-" * 96)
    P(f"  {'target':10s} {'fact about':>15s} {'chance':>7s} {'self':>7s} "
      f"{'bestOio':>8s} {'ADV':>8s} {'O_act':>8s} {'fracOio':>8s}")
    about = {"IMPL": "implementation", "IMPL_COS": "implementation", "BEHAV": "I/O map",
             "ENT": "I/O map", "WORLD": "input"}
    for tn in ("IMPL", "IMPL_COS", "BEHAV", "ENT", "WORLD"):
        if tn not in occ:
            continue
        r = occ[tn]
        oa = max([v for k, v in r["observers"].items() if k.startswith("O_act")]
                 or [float("nan")])
        ceil = max(r["self"], oa if oa == oa else r["self"])
        fr = ((r["best_O_io"] - r["chance"]) / (ceil - r["chance"])
              if ceil - r["chance"] > 1e-6 else float("nan"))
        P(f"  {tn:10s} {about.get(tn, ''):>15s} {r['chance']:7.3f} {r['self']:7.3f} "
          f"{r['best_O_io']:8.3f} {r['advantage']:+8.3f} {oa:8.3f} {fr:8.3f}")
    if "IMPL_COS" in occ and "cluster_quality" in occ["IMPL_COS"]:
        cq = occ["IMPL_COS"]["cluster_quality"]
        P(f"  residual k-means separation = {cq['separation']:.3f} "
          f"(within {cq['within']:.3f}, between {cq['between']:.3f})")
    P("")
    P("=" * 96)

    os.makedirs(a.outdir, exist_ok=True)
    txt = "\n".join(L)
    with open(os.path.join(a.outdir, f"{a.tag}_reduction.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    make_figures(R, a.outdir, a.tag, cap_tags, horizons, retro_w)
    print(f"\nwrote {a.outdir}/{a.tag}_reduction.txt and figures")


def make_figures(R, outdir, tag, cap_tags, horizons, retro_w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cfg, T, guards, ac = R["config"], R["targets"], R["guards"], R["autocorr"]
    steps = cfg["steps_list"]
    hold = cfg["hold_ckpts"]
    cap0 = cap_tags[0]
    ci = np.arange(len(steps))

    # --- fig 1: the trajectory and its guards ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(steps, R["val_loss_traj"], "o-", color="#333")
    ax[0].set_xscale("log"); ax[0].set_xlabel("wake step"); ax[0].set_ylabel("val loss")
    ax[0].set_title("M's trajectory")
    for c in hold:
        ax[0].axvline(steps[c], color="#c33", alpha=0.25)
    for ct, col in zip(cap_tags, ["#1f77b4", "#ff7f0e"]):
        g = [guards[ct][str(i)] for i in ci]
        ax[1].plot(steps, [x["ens_cos"] for x in g], "o-", color=col, label=f"{ct} ens_cos")
        ax[1].plot(steps, [x["fwd_cosine"] for x in g], "s--", color=col, alpha=0.5,
                   label=f"{ct} FM cosine")
    ax[1].axhline(0.78, color="#999", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("wake step")
    ax[1].set_title("instrument guards\n(dotted = paper 2's lowest published ens_cos)")
    ax[1].legend(fontsize=7)
    for ct, col in zip(cap_tags, ["#1f77b4", "#ff7f0e"]):
        g = [guards[ct][str(i)] for i in ci]
        ax[2].plot(steps, [x["floor_mean"] for x in g], "o-", color=col,
                   label=f"{ct} instrument floor")
        ax[2].plot(steps, [x["res_norm"] / max(y["res_norm"] for y in g) for x in g],
                   "^:", color=col, alpha=0.5, label=f"{ct} |r| (normalised)")
    ax[2].set_xscale("log"); ax[2].set_xlabel("wake step")
    ax[2].set_title("no-change floor and residual size")
    ax[2].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{tag}_trajectory_guards.png", dpi=140)
    plt.close(fig)

    # --- fig 2: the 2x2 advantage ---
    k0 = horizons[0]
    grid = [(f"IMPL_PROSP_k{k0}_{cap0}", "IMPL", "prospective"),
            (f"BEHAV_PROSP_k{k0}", "BEHAV", "prospective"),
            (f"IMPL_RETRO_w{retro_w}_{cap0}", "IMPL", "retrospective"),
            (f"BEHAV_RETRO_w{retro_w}", "BEHAV", "retrospective")]
    labels, selfv, obsv, actv, histv, chance = [], [], [], [], [], []
    for nm, fam, dr in grid:
        if nm not in T:
            continue
        r = T[nm]
        labels.append(f"{fam}\n{dr}")
        selfv.append(r["self_lnf_heldck"]); obsv.append(r["best_O_io_heldck"])
        actv.append(obs_get(r, "O_act", 1)); histv.append(obs_get(r, "O_hist", 1))
        chance.append(r["chance_heldck"])
    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].bar(x - 1.5 * w, selfv, w, label="self-report", color="#2b6cb0")
    ax[0].bar(x - 0.5 * w, obsv, w, label="best $O_{io}$", color="#a0aec0")
    ax[0].bar(x + 0.5 * w, actv, w, label="$O_{act}$ (ceiling)", color="#68d391")
    ax[0].bar(x + 1.5 * w, histv, w, label="$O_{hist}$ (ceiling)", color="#f6ad55")
    for i, c in enumerate(chance):
        ax[0].hlines(c, i - 2 * w, i + 2 * w, color="#c33", ls="--", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("accuracy (held-out seqs @ held-out checkpoints)")
    ax[0].set_title("the 2x2: implementation vs behaviour, "
                    "prospective vs retrospective\n(red dashes = chance)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2, axis="y")

    advio = [s - o for s, o in zip(selfv, obsv)]
    advpub = [s - max(o, h) for s, o, h in zip(selfv, obsv, histv)]
    cols = ["#2b6cb0" if l.startswith("IMPL") else "#a0aec0" for l in labels]
    ax[1].bar(x - 0.2, advio, 0.38, color=cols, label="vs best $O_{io}$")
    ax[1].bar(x + 0.2, advpub, 0.38, color=cols, hatch="//", edgecolor="w",
              label="vs best third party (incl. $O_{hist}$)")
    ax[1].axhline(0, color="#333", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("advantage")
    ax[1].set_title("first-person advantage by target family and time direction")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{tag}_advantage_2x2.png", dpi=140)
    plt.close(fig)

    # --- fig 3: gate + readings ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    nms = [n for n in ac]
    r1 = [ac[n]["r1"] for n in nms]
    btw = [ac[n]["between_frac"] for n in nms]
    cols = ["#2b6cb0" if n.startswith("IMPL") else "#a0aec0" for n in nms]
    y = np.arange(len(nms))
    ax[0].barh(y, r1, color=cols)
    ax[0].set_yticks(y); ax[0].set_yticklabels(nms, fontsize=6)
    ax[0].axvline(0, color="#333", lw=1)
    ax[0].set_xlabel("lag-1 autocorrelation across checkpoints")
    ax[0].set_title("the gate: is the target learnable at this resolution?\n"
                    "(committee_head's failing target read +0.007)")
    ax[0].grid(alpha=0.2, axis="x")
    ax[1].barh(y, btw, color=cols)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("between/total: fraction that is a per-position constant")
    ax[1].set_title("how much of the target is a static fact")
    ax[1].grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{tag}_autocorrelation_gate.png", dpi=140)
    plt.close(fig)

    # --- fig 4: the three instrument readings + occurrent battery ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    lbl, val = [], []
    for ct in cap_tags:
        for tagl, nm in (("raw", f"IMPL_PROSP_k{k0}_{ct}"),
                         ("excess", f"IMPL_PROSPEXC_k{k0}_{ct}"),
                         ("fixed", f"IMPL_PROSPFIX_k{k0}_{ct}")):
            if nm in T:
                lbl.append(f"{ct}\n{tagl}"); val.append(T[nm]["adv_heldck"])
    ax[0].bar(np.arange(len(lbl)), val, 0.55, color="#2b6cb0")
    ax[0].axhline(0, color="#333", lw=1)
    ax[0].set_xticks(np.arange(len(lbl))); ax[0].set_xticklabels(lbl, fontsize=7)
    ax[0].set_ylabel("advantage")
    ax[0].set_title("IMPL-prospective: three readings of 'my computation reorganized'")
    ax[0].grid(alpha=0.2, axis="y")

    occ = R["occurrent"]
    tns = [t for t in ("IMPL", "IMPL_COS", "BEHAV", "ENT", "WORLD") if t in occ]
    ov = [occ[t]["advantage"] for t in tns]
    ax[1].bar(np.arange(len(tns)), ov, 0.55,
              color=["#2b6cb0" if t.startswith("IMPL") else "#a0aec0" for t in tns])
    ax[1].axhline(0, color="#333", lw=1)
    ax[1].set_xticks(np.arange(len(tns))); ax[1].set_xticklabels(tns)
    ax[1].set_ylabel("advantage")
    ax[1].set_title("paper 2's occurrent battery on this model (final checkpoint)")
    ax[1].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{tag}_readings_and_occurrent.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
