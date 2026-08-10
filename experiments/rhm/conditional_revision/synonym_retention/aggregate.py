"""Read `synonym_retention` result JSONs and print the tables. Local, CPU, no Modal.

  # fetch, then:
  cd experiments
  python3 -m rhm.conditional_revision.synonym_retention.aggregate base <path-to-ret1.json>
  python3 -m rhm.conditional_revision.synonym_retention.aggregate arms <parts_arms1-dir> \
      [<parts_ll1-dir>]

`arms` joins each arm's retention against the FM-free activation rank published by
`local_loss/`'s `analyze_ll` (`parts_ll1/*.json` -> `act_rank.effective_rank_pct`), because
matched lambda is not matched pressure -- that folder's gotchas explain why the comparison
has to be made at matched achieved compression.
"""

import glob
import json
import os
import sys

import numpy as np

PB = "post_block6"
W_SHOW = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def _pool(rows, key, **sel):
    v = [r[key] for r in rows
         if key in r and all(r.get(k) == z for k, z in sel.items())
         and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(v)) if v else float("nan")


def _n(rows, **sel):
    return len([r for r in rows if all(r.get(k) == z for k, z in sel.items())])


def base_report(path):
    d = json.load(open(path))
    P = [r for r in d["probe_rows"] if r["block"] == PB]
    Tw = [r for r in d["twin_rows"] if r["block"] == PB]
    c = d["config"]
    print(f"\n{'=' * 100}")
    print(f"SYNONYM RETENTION vs DISTANCE  --  {c['label']}  "
          f"(m={c['m']}, n_seq={c['n_seq']}, n_twin={c['n_twin']}, block={PB})")
    print(f"  chance: rule 1/{c['m']} = {1 / c['m']:.3f}   feature 1/{c['v']} = "
          f"{1 / c['v']:.3f}   probe n_test = {P[0]['n_test'] if P else '?'}")
    print(f"{'=' * 100}")

    for d_lvl in sorted({r["level_d"] for r in P}):
        print(f"\n--- level d={d_lvl}  (constituent span {2 ** (6 - d_lvl)} leaves) ---")
        print(f"{'w':>3} {'k':>2} | {'ruleAcc':>7} {'shuf':>6} {'bayes':>6} {'RET':>7} | "
              f"{'featAcc':>7} {'shuf':>6} {'bayes':>6} {'RET':>7} | "
              f"{'relLeaf':>7} {'relTop':>7} {'relStr':>7} {'relRand':>7} | "
              f"{'syn/str':>7} {'TVleaf':>7} {'tv0%':>5} {'relLf|0':>7}")
        for w in W_SHOW:
            k = _n(P, level_d=d_lvl, w=w)
            if not k:
                continue
            g = lambda key, rows=P: _pool(rows, key, level_d=d_lvl, w=w)
            print(f"{w:>3} {k:>2} | {g('acc_rule'):>7.3f} {g('shuf_rule'):>6.3f} "
                  f"{g('bayes_rule'):>6.3f} {g('ret_rule'):>+7.3f} | "
                  f"{g('acc_feat'):>7.3f} {g('shuf_feat'):>6.3f} {g('bayes_feat'):>6.3f} "
                  f"{g('ret_feat'):>+7.3f} | "
                  f"{g('rel_leaf', Tw):>7.4f} {g('rel_top', Tw):>7.4f} "
                  f"{g('rel_str', Tw):>7.4f} {g('rel_rand', Tw):>7.4f} | "
                  f"{g('syn_over_str', Tw):>7.3f} {g('tv_leaf', Tw):>7.4f} "
                  f"{100 * g('frac_tvzero_leaf', Tw):>5.1f} "
                  f"{g('rel_leaf_tv0', Tw):>7.4f}")

    inc = [r for r in Tw if r["w"] == 0 and r["level_d"] == 4 and r["node_j"] == 15]
    if inc:
        r = inc[0]
        print(f"\nINCUMBENT CELL (d=4, j=15, w=0; NTP-undefined: it is the last position) "
              f"syn/str={r['syn_over_str']:.3f}  relLeaf={r['rel_leaf']:.4f}  "
              f"relStr={r['rel_str']:.4f}")
    print("\nHalf-life of the leaf-synonym probe (first w where ret_rule < 0.1 of its w=0):")
    for d_lvl in sorted({r["level_d"] for r in P}):
        r0 = _pool(P, "ret_rule", level_d=d_lvl, w=0)
        hit = next((w for w in W_SHOW
                    if _n(P, level_d=d_lvl, w=w)
                    and _pool(P, "ret_rule", level_d=d_lvl, w=w) < 0.1 * r0), None)
        print(f"  d{d_lvl}: ret_rule(w=0) = {r0:+.3f}  ->  crosses at w = {hit}")


def arms_report(parts_dir, ll_dir=None):
    rank = {}
    if ll_dir:
        for p in glob.glob(os.path.join(ll_dir, "*.json")):
            z = json.load(open(p))
            rank[z["label"]] = z["act_rank"]["effective_rank_pct"]
    rows = []
    for p in sorted(glob.glob(os.path.join(parts_dir, "*.json"))):
        d = json.load(open(p))
        lb = d["config"]["label"]
        P = [r for r in d["probe_rows"] if r["block"] == PB]
        Tw = [r for r in d["twin_rows"] if r["block"] == PB]
        row = {"label": lb, "actRk": rank.get(lb, float("nan"))}
        for d_lvl in (4, 5):
            for w in (0, 1, 2, 4, 8):
                row[f"rule_d{d_lvl}_w{w}"] = _pool(P, "ret_rule", level_d=d_lvl, w=w)
                row[f"feat_d{d_lvl}_w{w}"] = _pool(P, "ret_feat", level_d=d_lvl, w=w)
                row[f"rel_d{d_lvl}_w{w}"] = _pool(Tw, "rel_leaf", level_d=d_lvl, w=w)
        inc = [r for r in Tw if r["w"] == 0 and r["level_d"] == 4 and r["node_j"] == 15]
        row["syn_over_str_w0"] = inc[0]["syn_over_str"] if inc else float("nan")
        # area under the retention curve: one scale-free number per arm
        for d_lvl in (4, 5):
            ws = sorted({r["w"] for r in P if r["level_d"] == d_lvl and r["w"] > 0})
            vals = [_pool(P, "ret_rule", level_d=d_lvl, w=w) for w in ws]
            row[f"auc_rule_d{d_lvl}"] = float(np.nanmean(vals)) if vals else float("nan")
            vals = [_pool(P, "ret_feat", level_d=d_lvl, w=w) for w in ws]
            row[f"auc_feat_d{d_lvl}"] = float(np.nanmean(vals)) if vals else float("nan")
        rows.append(row)

    rows.sort(key=lambda z: (-z["actRk"] if not np.isnan(z["actRk"]) else 0))
    print(f"\n{'=' * 118}")
    print("LOCAL-LOSS ARM COMPARISON -- synonym retention, sorted by achieved compression")
    print("  ret = (acc - chance)/(bayes - chance) on the leaf-synonym (rule) label; "
          "aucRule = mean over w>0")
    print(f"{'=' * 118}")
    print(f"{'arm':>22} {'actRk%':>7} | {'ruleW0':>7} {'ruleW1':>7} {'ruleW2':>7} "
          f"{'ruleW4':>7} {'ruleW8':>7} {'aucRule':>8} | {'featW0':>7} {'featW4':>7} "
          f"{'aucFeat':>8} | {'syn/str':>7}")
    for r in rows:
        print(f"{r['label']:>22} {r['actRk']:>7.1f} | "
              f"{r['rule_d4_w0']:>+7.3f} {r['rule_d4_w1']:>+7.3f} {r['rule_d4_w2']:>+7.3f} "
              f"{r['rule_d4_w4']:>+7.3f} {r['rule_d4_w8']:>+7.3f} {r['auc_rule_d4']:>+8.3f} | "
              f"{r['feat_d4_w0']:>+7.3f} {r['feat_d4_w4']:>+7.3f} {r['auc_feat_d4']:>+8.3f} | "
              f"{r['syn_over_str_w0']:>7.3f}")
    print("\n(d=4 = the incumbent's constituent level, span 4 leaves. "
          "Compare at matched actRk%, not matched lambda.)")
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "base":
        base_report(sys.argv[2])
    else:
        arms_report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
