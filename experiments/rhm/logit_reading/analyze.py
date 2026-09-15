"""Aggregate Part 1 + Part 2 across a trajectory's checkpoints (local, reads JSON).

  modal volume get rhm-scaling-data v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 <dir>
  cd experiments && python -m rhm.logit_reading.analyze <dir>/traj_a1_s42 --stim-tag a1
"""

import argparse
import glob
import json
import os


def _f(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'nan':>{w}}"


def part1(traj):
    rows = []
    for fn in sorted(glob.glob(f"{traj}/step*_calibration.json")):
        r = json.load(open(fn))
        rows.append((r["config"]["ckpt_config"]["step"], r))
    print("\nPART 1 -- calibration against the exact phase-marginal oracle")
    print(f"{'step':>6} {'CE':>6} {'H_p':>6} {'KL':>6} {'gap':>7} {'|gap|':>6} {'R2':>5} {'ECE':>6}  "
          f"KL(p_k||q) k=0..6                          argmin  frac best_k=0..6")
    for step, r in rows:
        o = r["overall"]
        kl = o["KL_k_q"]
        hist = o["best_k_hist"]
        tot = sum(hist)
        print(f"{step:>6} {_f(o['CE'])} {_f(o['H_p'])} {_f(o['KL_p_q'])} {o['gap_CE_minus_Hq']:+7.3f} "
              f"{_f(o['abs_gap'])} {_f(o['R2_CE_on_Hq'], 5, 2)} {_f(r['ece_realised'], 6, 4)}  "
              + " ".join(_f(x, 5) for x in kl) + f"   k={kl.index(min(kl))}   "
              + " ".join(f"{h / tot:.2f}" for h in hist))
    if rows:
        print("  observer ladder KL(p_L||p_k): " + " ".join(_f(x, 5) for x in rows[-1][1]["overall"]["KL_truth_k"]))
    return rows


def part2(traj, tag):
    rows = []
    for fn in sorted(glob.glob(f"{traj}/step*_violation_{tag}.json")):
        r = json.load(open(fn))
        rows.append((r["config"]["step"], r))
    if not rows:
        return rows
    ks = [k for k in range(1, 7)]
    print("\nPART 2a -- unmatched detection AUC by k_star (violation surprisal percentile among "
          "legal tokens, same token level x window-position bucket)")
    print(f"{'step':>6} " + " ".join(f"{'k*=' + str(k):>6}" for k in ks))
    for step, r in rows:
        d = r["detection_by_kstar"]
        print(f"{step:>6} " + " ".join(_f(d.get(str(k), {}).get("detect_auc_stratified", float('nan')))
                                      for k in ks))
    print("\nPART 2b -- matched (token x level x position bucket x surprisal bin), all k_star pooled")
    cols = ["guard_surprisal", "state_post_embed", "R1_s_minus_H", "H_next_minus_H_prev",
            "logit_linear", "logit_mlp"]
    print(f"{'step':>6} {'n':>6} " + " ".join(f"{c[:10]:>10}" for c in cols) + f" {'best_state':>10} {'block':>12} {'st-logit':>8}")
    for step, r in rows:
        m = r["matched"]["all"]
        bb = r["best_state_block"]
        print(f"{step:>6} {m['n']:>6} " + " ".join(_f(m[c], 10) for c in cols)
              + f" {_f(m[bb], 10)} {bb[6:]:>12} {r['state_minus_logit_mlp']:+8.3f}")
    print("\nPART 2c -- matched by k_star: logit_mlp / best-state / embed-guard AUC")
    print(f"{'step':>6} " + " ".join(f"{'k*=' + str(k):>20}" for k in ks))
    for step, r in rows:
        bb = r["best_state_block"]
        cells = []
        for k in ks:
            m = r["matched"].get(f"k{k}")
            cells.append(f"{_f(m['logit_mlp'], 5, 2)}/{_f(m[bb], 5, 2)}/{_f(m['state_post_embed'], 5, 2)}"
                         if m else f"{'-':>17}")
        print(f"{step:>6} " + " ".join(f"{c:>20}" for c in cells))
    print("\nPART 2c' -- matched vs legal-rare only")
    for step, r in rows:
        m = r["matched"]["vs_rare_only"]
        bb = r["best_state_block"]
        print(f"{step:>6} n {m['n']:>5}  guard {_f(m['guard_surprisal'])}  embed {_f(m['state_post_embed'])}  "
              f"R1 {_f(m['R1_s_minus_H'])}  logit_mlp {_f(m['logit_mlp'])}  state {_f(m[bb])}")
    step, r = rows[-1]
    print(f"\nPART 3 -- state-probe transfer across k_star, step {step} ({r['best_state_block']})")
    tr = r["transfer_best_block"]
    if tr:
        heads = sorted({b for a in tr.values() for b in a})
        print("train\\test " + " ".join(f"{h:>6}" for h in heads))
        for a in sorted(tr, key=int):
            print(f"{a:>10} " + " ".join(_f(tr[a].get(h, float('nan'))) for h in heads))
    print(f"\nPART 4 -- persistence after the event, step {step}: dH_model / dH_bayes_eps / excess KL")
    for g, pr in r["persistence"].items():
        print(f"  {g:>9} n={pr[0]['n']:5d}  " + "  ".join(
            f"t{x['tau']}:{x['dH_model']:+.2f}/{x['dH_bayes_eps']:+.2f}/{x['excess_KL_model_from_bayes_eps']:+.2f}"
            for x in pr if x["tau"] in (0, 1, 2, 4, 8, 12)))
    return rows


def part2_v2(traj, tag):
    rows = []
    for fn in sorted(glob.glob(f"{traj}/step*_violation2_{tag}.json")):
        r = json.load(open(fn))
        rows.append((r["config"]["step"], r))
    if not rows:
        return rows
    ks = range(1, 7)
    print("\nPART 2a (v2) -- unmatched detection AUC by k_star (surprisal percentile among legal "
          "tokens at the same token level and exact window index)")
    print(f"{'step':>6} " + " ".join(f"{'k*=' + str(k):>6}" for k in ks) + "   matched fraction k*=1..6")
    for step, r in rows:
        d, c = r["detection_by_kstar"], r["match_coverage_by_kstar"]
        print(f"{step:>6} " + " ".join(_f(d.get(str(k), {}).get("detect_auc", float("nan"))) for k in ks)
              + "   " + " ".join(f"{c.get(str(k), {}).get('matched_frac', float('nan')):.2f}" for k in ks))
    cols = ["guard_surprisal", "state_post_embed", "R1_s_minus_H", "run_surprisal_4", "run_surprisal_mlp",
            "prev_logit_only_mlp", "pre_logit_mlp", "post_logit_mlp", "logit_mlp",
            "state_tm1_best", "state_tm1_and_t_best"]
    short = ["g_surp", "g_embed", "R1", "run_s4", "run_mlp", "q_prev", "q_prev+x", "q_next", "logits",
             "h[t-1]", "h[t-1,t]"]
    for which in ("matched_vs_legal", "matched_vs_rare"):
        print(f"\nPART 2b (v2) -- {which}: exact (token, level, window index) + surprisal NN "
              f"(caliper), pooled over k_star")
        print(f"{'step':>6} {'pairs':>5} {'|ds|':>5} " + " ".join(f"{s:>8}" for s in short)
              + f" {'h[t]':>6} {'block':>6}")
        for step, r in rows:
            m = r[which]
            if "all" not in m:
                print(f"{step:>6} {m['n_pairs_test']:>5}  (too few pairs)")
                continue
            a, bb = dict(m["all"]), m["best_block"]
            a["state_tm1_best"] = m["phasic_tonic"]["state_tm1"]
            a["state_tm1_and_t_best"] = m["phasic_tonic"]["state_tm1_and_t"]
            print(f"{step:>6} {m['n_pairs_test']:>5} {m['mean_abs_ds_test']:.3f} "
                  + " ".join(_f(a.get(c, float('nan')), 8) for c in cols)
                  + f" {_f(a['state_' + bb])} {bb[10:]:>6}")
    print("\nPART 2c (v2) -- vs legal, by k_star: pairs | R1 | run_s4 | q_prev | q_prev+x | h[t-1] | h[t-1,t]")
    for step, r in rows:
        m = r["matched_vs_legal"]
        cells = []
        for k in ks:
            d = m.get("by_kstar", {}).get(str(k))
            cells.append("-" if not d else
                         f"k{k}:{d['n_pairs']}|{d['R1_s_minus_H']:.2f}|{d['run_surprisal_4']:.2f}|"
                         f"{d['prev_logit_only_mlp']:.2f}|{d['pre_logit_mlp']:.2f}|"
                         f"{d['state_tm1_best']:.2f}|{d['state_tm1_and_t_best']:.2f}")
        print(f"{step:>6}  " + "  ".join(c for c in cells if c != "-"))
    print("\nPART 3 (v2) -- best-block state probe at offsets from the event (vs legal): t-1, t, t+1, t+2, t+4, t+8")
    for step, r in rows:
        m = r["matched_vs_legal"]
        if "all" not in m:
            continue
        off = m["state_offsets_best_block"]
        seq = [off.get("-1", {}).get("auc", float("nan")), m["all"]["state_" + m["best_block"]]] + \
              [off.get(str(o), {}).get("auc", float("nan")) for o in (1, 2, 4, 8)]
        print(f"{step:>6} " + " ".join(_f(x) for x in seq))
    step, r = rows[-1]
    print(f"\nPART 3b (v2) -- transfer across k_star at step {step}")
    for a, row in sorted(r["transfer_best_block"].items()):
        print(f"  train k*={a}: " + "  ".join(f"test {b}: {x:.3f}" for b, x in sorted(row.items())))
    print(f"\nPART 4 (v2) -- output persistence at step {step}: dH_model / dH_bayes_eps (tau 0,1,2,4,8,12)")
    for g, pr in r["persistence"].items():
        print(f"  {g:>8} n={pr[0]['n']:5d}  " + "  ".join(
            f"t{x['tau']}:{x['dH_model']:+.2f}/{x['dH_bayes_eps']:+.2f}" for x in pr if x["tau"] in (0, 1, 2, 4, 8, 12)))
    return rows


def phasic(traj, tag, caliper):
    rows = []
    for fn in sorted(glob.glob(f"{traj}/step*_phasic_{tag}_cal{caliper}.json")):
        r = json.load(open(fn))
        rows.append((r["config"]["step"], r))
    if not rows:
        return rows
    print(f"\nPHASIC -- same prefix, legal twin token at matched model surprisal (caliper {caliper}), "
          f"token identity balanced. AUC (paired win rate) on test pairs")
    cols = [("guard_surprisal", "g_surp"), ("state_post_embed", "g_token"), ("state_tm1_guard", "g_h[t-1]"),
            ("pre_logit_mlp", "q_prev+x"), ("H_next", "H_next"), ("KL_update", "KL_upd"),
            ("post_logit_mlp", "q_next"), ("state_best_mlp", "h[t] mlp")]
    print(f"{'step':>6} {'pairs':>6} {'bal':>5} {'test':>5} " + " ".join(f"{c[1]:>13}" for c in cols)
          + f" {'best lin h[t]':>13}")
    for step, r in rows:
        if "test_all" not in r:
            print(f"{step:>6} {r['n_pairs']:>6}  (too few)")
            continue
        a = r["test_all"]
        bb = "state_" + r["best_block"]
        print(f"{step:>6} {r['n_pairs']:>6} {r['n_pairs_token_balanced']:>5} {a['n_pairs']:>5} "
              + " ".join(f"{a[c]['auc']:.3f} ({a[c]['paired']:.2f})" for c, _ in cols)
              + f" {a[bb]['auc']:.3f} ({a[bb]['paired']:.2f})")
    print("  by k_star (test pairs | q_next AUC | h[t] mlp AUC | H_next AUC):")
    for step, r in rows:
        byk = r.get("test_by_kstar", {})
        if byk:
            print(f"  {step:>6} " + "  ".join(
                f"k{k}: {d['n_pairs']}|{d['post_logit_mlp']['auc']:.2f}|{d['state_best_mlp']['auc']:.2f}|"
                f"{d['H_next']['auc']:.2f}" for k, d in sorted(byk.items(), key=lambda kv: int(kv[0]))))
    return rows


def gradient(traj, tag):
    rows = []
    for fn in sorted(glob.glob(f"{traj}/step*_gradient_{tag}.json")):
        r = json.load(open(fn))
        rows.append((r["config"]["step"], r))
    if not rows:
        return rows
    print("\nGRADED CONTROL -- two LEGAL tokens after the same prefix, matched on the model's "
          "surprisal (sign-balanced), token identity balanced; positive = the one at least 2x less "
          "likely under the true grammar. AUC (paired)")
    cols = [("guard_surprisal", "g_surp"), ("state_post_embed", "g_token"), ("H_next", "H_next"),
            ("pre_logit_mlp", "q_prev+x"), ("post_logit_mlp", "q_next"), ("state_best_mlp", "h[t] mlp")]
    print(f"{'step':>6} {'pairs':>6} {'test':>5} " + " ".join(f"{c[1]:>13}" for c in cols))
    for step, r in rows:
        a = r.get("test_all")
        if not a:
            print(f"{step:>6} {r['n_pairs']:>6}  (too few)")
            continue
        print(f"{step:>6} {r['n_pairs']:>6} {a['n_pairs']:>5} "
              + " ".join(f"{a[c]['auc']:.3f} ({a[c]['paired']:.2f})" for c, _ in cols))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--stim-tag", default="a1")
    ap.add_argument("--phasic-tag", default="swap65k")
    ap.add_argument("--caliper", default="0.3")
    a = ap.parse_args()
    part1(a.traj)
    part2(a.traj, a.stim_tag)
    part2_v2(a.traj, a.stim_tag)
    phasic(a.traj, a.phasic_tag, a.caliper)
    gradient(a.traj, a.phasic_tag)
