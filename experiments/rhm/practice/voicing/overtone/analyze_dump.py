"""[overtone] The offline reduction of the banked rows — what `vo_rows.npz` is for.

The audit reads the critic's held-out slice once per cycle and reports an AUC. The dump is the
same rows on disk, so a question about the DIET can be asked afterwards on a CPU instead of by
paying for another 1.3-1.9 GPU-h arm. This script is the first such question and doubles as an
independent check of the audit's own `dp_auc` column: the prior's ranking of the verdict,
recomputed here from the stored DP score, over the WHOLE final buffer rather than its 10%
held-out slice.

What it reads, per slot and per buffer (`filed` / `probe`):
    obs, write, y, dp (the DP's per-block score of that candidate through the FINAL core),
    code (the audit's bijective hold code), unif (the probe row's draw tag, S3 arms only)

Usage (from experiments/):
    python3 rhm/practice/voicing/overtone/analyze_dump.py --tag ov_s0
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")


def auc(x, y):
    """the rank identity with ties averaged — `voicing.py::vo_auc`, on numpy alone."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None, int(len(x))
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(1, len(x) + 1)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), int(len(x))


POST_DOC = """
[overtone S1b] THE POST-WRITE LINEAR PROBE — S1 as run tests ADDITIVITY, not linearity.

`lin` and `dir` are `Linear(dim, 1)` over `u_ctx + e_cand`. A linear map of a sum is the sum of
the maps, so within one context their ranking over candidates is `w . e_cand` plus a per-row
offset: THE SAME CANDIDATE ORDER IN EVERY CONTEXT. Their across-row AUC can be respectable —
they read context difficulty and a candidate's base rate — while carrying no "this candidate
HERE" at all. That is the shape of `ovt_comp_pr_lin`'s result: `moved` 0.000 at both L5 cells,
where the MLP moves 0.077, and yet an AUC within 0.03 of it on filed rows.

Steenwyk's probe is linear on the state AFTER the answer is in the context. The analogue here is
a linear probe on the trunk's pooled hiddens of the POST-WRITE configuration — the write
rendered into the masked context, the same rendered object the probe channel grades. The trunk
mixes context and candidate before the probe is linear in anything, so this form CAN carry
"this candidate here".

Four readouts on one set of rows, per slot, filed and probe apart:

  pre_slot    Linear over [pooled(obs).mean ; pooled(obs)[slot blocks].mean] — CONTROL ONE.
              The write is not in it at all, so whatever this reaches is context difficulty.
  rand_slot   `post_slot`'s features through a RANDOMLY INITIALISED trunk of the same
              architecture — CONTROL TWO, and the one that matters most. A random transformer
              is still a nonlinear feature map of the token sequence, so a linear probe on its
              hiddens of the post-write configuration is not reading anything the TRAINED
              trunk learned. This control was not planned: it appeared because the section's
              dry run was done against a synthetic `vo_heads.pt` whose core was random, and
              the probe still reached 0.754 on probe rows. That number is the FLOOR, and
              `post_slot` has to be read against it rather than against 0.5.
  post_mean   Linear over pooled(obs+write).mean
  post_slot   Linear over [pooled(obs+write).mean ; pooled(obs+write)[slot blocks].mean]
  mlp         the run's own critic, recomputed from `vo_heads.pt` on the same rows
  dp          the prior: the DP's per-block score of that candidate, from the dump

WHERE THE PROBE BUFFER IS MIXED (an S3 arm: `ov_probe_unif_frac` of its budget is uniform and
the rest is drawn by disagreement), the probe rows are reported three ways — `probe` over all of
them, `probe/u` over the uniform-drawn held-out rows and `probe/d` over the disagreement-drawn
ones. ONE probe is fitted per slot, on that arm's own training rows, and scored on each subset.
So `probe/u` is comparable across arms and seeds in its EVALUATION set; the training
distributions still differ, and no split can fix that.

THE PROTOCOL IS THE AUDIT'S. The probes are fit on the rows the critic TRAINED on
(`hold_code >= 100*vo_critic_hold`) and scored on the rows the critic was AUDITED on
(`hold_code <` it), so every AUC in the table is on the same held-out rows.

TWO THINGS THIS IS NOT, said before the numbers:
  - for a PROBE row the verdict was earned by substituting into the trajectory's FINAL
    configuration, while the object reconstructed here is the substituted write in the masked
    context at the slot. The critic only ever sees the latter, so the comparison to the critic
    is exact; the comparison to "what the grader saw" is not, and the dump does not carry it.
  - the probes here are fit on ~7k rows of a 96-d (or 192-d) feature and scored on ~800; the
    critic was trained online across the whole run. An offline probe has the easier job.
"""


def _fit_probe(F, y, hold, ridge=1e-2, iters=40):
    """Fit `ov_irls` on the critic's own TRAINING rows, score the audit's held-out rows."""
    from rhm.practice.voicing.voicing import ov_irls
    tr = ~hold
    if int(tr.sum()) < 64 or int(hold.sum()) < 16:
        return None
    yt = y[tr]
    if yt.min() == yt.max():
        return None
    mu, sd = F[tr].mean(0), F[tr].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(F[tr] - mu) / sd, np.ones((int(tr.sum()), 1))], 1)
    Bm = np.concatenate([(F[hold] - mu) / sd, np.ones((int(hold.sum()), 1))], 1)
    w = ov_irls(A, yt, ridge=ridge, iters=iters)
    return Bm @ w


def post_write_probe(tag, arm, o):
    """The section itself. Needs `vo_heads.pt` beside `vo_rows.npz`."""
    import torch
    import rhm.rhm_generative_planner as GP
    import rhm.practice.native.span.span_net as SN
    from rhm.practice.voicing.voicing import build_critic, vo_critic_scores
    from rhm.rhm_data import generate_rules_distinct

    root = os.path.join(FIG, tag, arm)
    hp = os.path.join(root, "vo_heads.pt")
    if not os.path.isfile(hp):
        o(f"    {arm}: no vo_heads.pt — the heads were banked from `ov_s0b` on. Skipped.")
        return
    blob = torch.load(hp, map_location="cpu", weights_only=True)
    cfgh = blob["cfg"]
    cfgr = json.load(open(os.path.join(root, "results.json")))["config"]
    v, s, depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
    dim, maxl = int(cfgh["state_dim"]), int(cfgh["max_macro_level"])
    length = s ** depth
    thr = int(round(float(cfgh["vo_critic_hold"]) * 100))
    core = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False)
    core.load_state_dict(blob["core"])
    core.eval()
    # CONTROL TWO: the same architecture, never trained. Its init seed is fixed and stated so
    # the floor is reproducible; it is the only random draw this reduction makes.
    torch.manual_seed(20260915)
    rcore = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                  root_conditioned=False)
    rcore.eval()
    hm = int(cfgh.get("ov_critic_hidden", -1))
    critic = build_critic(SN.slot_count(s, depth, maxl), v, dim, s ** (maxl - 1), seed=0,
                          device=torch.device("cpu"),
                          hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
    critic.load_state_dict(blob["critic"])
    critic.eval()
    rules = generate_rules_distinct(v, s, depth, int(cfgr["m"]),
                                    seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]), dtype=torch.long)
    z = np.load(os.path.join(root, "vo_rows.npz"))
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))
    o(f"    {arm}:")
    o("      diet      slot   n_tr  n_hold   base | pre_slot rand_slot post_mean post_slot |"
      "   mlp     dp")
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for pre in sorted(meta, key=lambda k: (k.split(":")[0],
                                           int(k.split(":")[1]), int(k.split(":")[2]))):
        which, slot = pre.split(":", 1)
        mt = meta[pre]
        blk0, span, sid_i = int(mt["blk0"]), int(mt["span"]), int(mt["slot_id"])
        obs = torch.from_numpy(z[f"{pre}|obs"].astype(np.int64))
        wr = torch.from_numpy(z[f"{pre}|write"].astype(np.int64))
        y = z[f"{pre}|y"].astype(np.float64)
        code = z[f"{pre}|code"].astype(np.int64)
        dp = z[f"{pre}|dp"].astype(np.float64)
        n = int(obs.shape[0])
        hold = code < thr
        if n < 128 or int(hold.sum()) < 16 or y.min() == y.max():
            continue
        blocks = torch.arange(blk0, blk0 + span)
        pos = (blocks[:, None] * s + torch.arange(s)[None, :]).reshape(-1)
        pm, ps, qm, qs, cl, rm, rs = [], [], [], [], [], [], []
        with torch.no_grad():
            for a_ in range(0, n, 512):
                b_ = min(a_ + 512, n)
                ob = obs[a_:b_]
                wb = wr[a_:b_]
                x2 = ob.clone().scatter_(
                    1, pos[None, :].expand(b_ - a_, -1),
                    canon[wb].reshape(b_ - a_, -1))
                p0, _ = SN.trunk(core, ob)
                p1, _ = SN.trunk(core, x2)
                p2, _ = SN.trunk(rcore, x2)
                sid = torch.full((b_ - a_,), sid_i, dtype=torch.long)
                cl.append(critic(p0, blk0, span, sid, wb).numpy())
                pm.append(p0.mean(1).numpy())
                ps.append(p0[:, blk0:blk0 + span, :].mean(1).numpy())
                qm.append(p1.mean(1).numpy())
                qs.append(p1[:, blk0:blk0 + span, :].mean(1).numpy())
                rm.append(p2.mean(1).numpy())
                rs.append(p2[:, blk0:blk0 + span, :].mean(1).numpy())
        pm, ps = np.concatenate(pm), np.concatenate(ps)
        qm, qs = np.concatenate(qm), np.concatenate(qs)
        rm, rs = np.concatenate(rm), np.concatenate(rs)
        cl = np.concatenate(cl).astype(np.float64)
        feats = {"pre_slot": np.concatenate([pm, ps], 1),
                 "rand_slot": np.concatenate([rm, rs], 1),
                 "post_mean": qm,
                 "post_slot": np.concatenate([qm, qs], 1)}
        # [overtone] THE DRAW SPLIT, so a seed whose probe buffer is mostly disagreement-drawn
        # is still comparable to one that is not. ONE probe is fitted, on the arm's own
        # training rows, and then SCORED on each subpopulation of the held-out rows. The
        # evaluation sets are matched across seeds; the TRAINING distributions are not, and
        # that limitation is the one the header cannot remove.
        uk = f"{pre}|unif"
        uh = None
        if which == "probe" and uk in z:
            um = z[uk] > 0.5
            if um.any() and (~um).any():
                uh = um[hold]
        scored = {}
        for nm, F in feats.items():
            scored[nm] = _fit_probe(F, y, hold)
        scored["mlp"] = cl[hold]
        scored["dp"] = dp[hold]

        def _emit(label, sub):
            r = {}
            for nm, sc in scored.items():
                r[nm] = (auc(sc[sub], y[hold][sub])[0] if sc is not None else None)
            for nm, val in r.items():
                if val is not None:
                    pool[label][nm].append(val)

            def f(x):
                return f"{x:.3f}" if x is not None else "  -  "
            o(f"      {label:8} {slot:6} {int((~hold).sum()):>5} {int(sub.sum()):>6} "
              f"{y[hold][sub].mean():>7.3f} |  {f(r['pre_slot'])}    {f(r['rand_slot'])}"
              f"     {f(r['post_mean'])}     {f(r['post_slot'])}  |  "
              f"{f(r['mlp'])}  {f(r['dp'])}")

        allsub = np.ones(int(hold.sum()), bool)
        _emit(which, allsub)
        if uh is not None:
            if int(uh.sum()) >= 16:
                _emit("probe/u", uh)
            if int((~uh).sum()) >= 16:
                _emit("probe/d", ~uh)
    o("      MEDIAN over slots:")
    for which, d in pool.items():
        o(f"        {which:6} " + "  ".join(
            f"{nm} {np.median(d[nm]):.3f}" for nm in
            ("pre_slot", "rand_slot", "post_mean", "post_slot", "mlp", "dp") if d.get(nm))
          + f"   ({len(d.get('mlp', []))} slots)")
    o("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ov_s0")
    ap.add_argument("--arms", default="")
    ap.add_argument("--post", action="store_true",
                    help="also run the post-write linear probe (needs vo_heads.pt and torch)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = os.path.join(FIG, a.tag)
    arms = [x for x in a.arms.split(",") if x] or sorted(
        d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "vo_rows.npz")))
    lines = []

    def o(s=""):
        lines.append(s)
        print(s)

    o("=" * 100)
    o(f"[overtone] {a.tag} — the banked rows, offline")
    o("=" * 100)
    o("    THE PRIOR'S OWN RANKING OF THE VERDICT, recomputed here from the stored DP score")
    o("    through the FINAL core, over the WHOLE buffer. Two things make it not the same")
    o("    number as the audit's `dp_auc` and both are stated rather than glossed:")
    o("      - the audit reads the 10% held-out slice once per CYCLE and the reduction takes a")
    o("        median over those reads; this is ONE read over every row the buffer still holds")
    o("        at the end, scored through the core as it finished;")
    o("      - a row whose candidate the final operative table no longer holds has no DP score")
    o("        (`n` below counts only the rows that do).")
    o("    Agreement between the two is therefore a consistency check on the machinery, not an")
    o("    identity, and the denominators are printed so the comparison is honest.")
    o("")
    for arm in arms:
        z = np.load(os.path.join(root, arm, "vo_rows.npz"))
        meta = json.load(open(os.path.join(root, arm, "vo_rows_meta.json")))
        o(f"    {arm}:")
        o("      diet    slot     rows   on-table   base rate   AUC(dp vs verdict)")
        pool = collections.defaultdict(lambda: [[], []])
        for pre in sorted(meta, key=lambda k: (k.split(":")[0],
                                               int(k.split(":")[1]), int(k.split(":")[2]))):
            which, slot = pre.split(":", 1)
            y = z[f"{pre}|y"].astype(np.float64)
            dp = z[f"{pre}|dp"].astype(np.float64)
            au, n = auc(dp, y)
            pool[which][0].append(dp)
            pool[which][1].append(y)
            o(f"      {which:6} {slot:6} {len(y):>8}  {meta[pre]['n_on_table']:>8}   "
              f"{y.mean():>9.3f}   " + (f"{au:.3f}  (n={n})" if au is not None else "  -  "))
        o("      POOLED over every slot:")
        for which, (dps, ys) in pool.items():
            au, n = auc(np.concatenate(dps), np.concatenate(ys))
            yy = np.concatenate(ys)
            o(f"        {which:6} rows {len(yy):>8}  base {yy.mean():.3f}  "
              + (f"AUC(dp) {au:.3f} on {n} scored rows" if au is not None else "AUC -"))
        # S3 only: the draw tag
        for pre in sorted(meta):
            k = f"{pre}|unif"
            if k in z and (z[k] < 0.5).any():
                break
        else:
            o("")
            continue
        du, dd = [[], []], [[], []]
        for pre in sorted(meta):
            k = f"{pre}|unif"
            if k not in z:
                continue
            u = z[k] > 0.5
            y = z[f"{pre}|y"].astype(np.float64)
            dp = z[f"{pre}|dp"].astype(np.float64)
            du[0].append(dp[u]); du[1].append(y[u])
            dd[0].append(dp[~u]); dd[1].append(y[~u])
        for nm, (dps, ys) in (("uniform", du), ("disagree", dd)):
            yy = np.concatenate(ys)
            au, n = auc(np.concatenate(dps), yy)
            o(f"        probe/{nm:9} rows {len(yy):>8}  base {yy.mean():.3f}  "
              + (f"AUC(dp) {au:.3f}" if au is not None else "AUC -"))
        o("")
    if a.post:
        o("=" * 100)
        o("[S1b] THE POST-WRITE LINEAR PROBE")
        o("=" * 100)
        for ln in POST_DOC.strip().splitlines():
            o("    " + ln)
        o("")
        for arm in arms:
            post_write_probe(a.tag, arm, o)
    out = a.out or os.path.join(HERE, "figures", f"{a.tag}_dump.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
