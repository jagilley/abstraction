"""Aggregate the RL dimensionality ablation across m (stdlib only).

Usage:
    # pull results locally first:
    for M in 1 2 3 4 6; do
      modal volume get rhm-scaling-data \
        rl_dimensionality/v8_s2_L6_m${M}_both_seed42/results.json m${M}.json --force
    done
    python3 aggregate_results.py [dir-with-m{M}.json]
"""
import json
import os
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
MS = [1, 2, 3, 4, 6]
D = {m: json.load(open(os.path.join(BASE, f"m{m}.json"))) for m in MS}

def refs(m):
    return D[m]["references"]

def cond(m, c):
    ck = D[m]["conditions"][c]["checkpoints"]
    return ck[max(ck, key=int)]

def gen(m, c, mode="greedy"):
    return cond(m, c)["generation"][mode]

def pre_gen(m, mode="greedy"):
    return D[m]["pretrain"]["eval"]["generation"][mode]

print("=" * 100)
print("REFERENCES (exact-match ceilings/floors; parse floors; NTP Bayes floor)")
print("=" * 100)
print(f"{'m':>2} {'EM greedy':>10} {'EM sampled':>10} {'EM blind':>9} "
      f"{'P rand':>7} {'P blind':>8} {'NTP floor':>10} {'uniform':>8}")
for m in MS:
    r = refs(m)
    print(f"{m:>2} {r['exact_match']['greedy_ceiling']['overall']:>10.4f} "
          f"{r['exact_match']['sampled_ceiling']['overall']:>10.4f} "
          f"{r['exact_match']['blind_floor']['overall']:>9.4f} "
          f"{r['parse']['random_floor']['overall']:>7.4f} "
          f"{r['parse']['blind_floor']['overall']:>8.4f} "
          f"{r['ntp_bayes']['entropy_rate_pos1plus_nats']:>10.4f} "
          f"{r['uniform_nats']:>8.4f}")

print()
print("=" * 100)
print("PRETRAIN (to plateau)")
print("=" * 100)
print(f"{'m':>2} {'steps':>6} {'val':>8} {'excess':>8} {'captured':>9} "
      f"{'EM greedy':>10} {'EM sampled':>11} {'parse(g)':>9} {'parse(s)':>9}")
for m in MS:
    p = D[m]["pretrain"]
    ex = p["best_val"] - refs(m)["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    g, s_ = pre_gen(m), pre_gen(m, "sampled")
    print(f"{m:>2} {p['steps_used']:>6} {p['best_val']:>8.4f} {ex:>8.4f} "
          f"{p['captured_excess_entropy']:>9.3f} {g['exact_overall']:>10.4f} "
          f"{s_['exact_overall']:>11.4f} {g['parse_overall']:>9.4f} "
          f"{s_['parse_overall']:>9.4f}")

print()
print("=" * 100)
print("PARSE ARM — greedy parse validity (floor=blind, ceiling=1.0) + headroom recovered")
print("=" * 100)
print(f"{'m':>2} | {'pretrained':>10} | {'scratch_rl':>10} {'hr':>6} | "
      f"{'pretrain_rl':>11} {'hr':>6} | {'pretrain_only':>13} {'hr':>6}")
for m in MS:
    fl = refs(m)["parse"]["blind_floor"]["overall"]
    def hr(x):
        return (x - fl) / (1 - fl) if 1 - fl > 1e-9 else float("nan")
    pg = pre_gen(m)["parse_overall"]
    sc = gen(m, "scratch_rl_parse")["parse_overall"]
    pr = gen(m, "pretrain_rl_parse")["parse_overall"]
    po = gen(m, "pretrain_only")["parse_overall"]
    print(f"{m:>2} | {pg:>10.4f} | {sc:>10.4f} {hr(sc):>6.3f} | "
          f"{pr:>11.4f} {hr(pr):>6.3f} | {po:>13.4f} {hr(po):>6.3f}")

print()
print("PARSE ARM — sampled-mode parse validity (policy as trained/sampled)")
print(f"{'m':>2} | {'pretrained':>10} | {'scratch_rl':>10} | {'pretrain_rl':>11} | "
      f"{'pretrain_only':>13}")
for m in MS:
    print(f"{m:>2} | {pre_gen(m,'sampled')['parse_overall']:>10.4f} | "
          f"{gen(m,'scratch_rl_parse','sampled')['parse_overall']:>10.4f} | "
          f"{gen(m,'pretrain_rl_parse','sampled')['parse_overall']:>11.4f} | "
          f"{gen(m,'pretrain_only','sampled')['parse_overall']:>13.4f}")

print()
print("PARSE ARM — per-level valid fractions (sampled), pretrain_rl_parse vs scratch_rl_parse")
print(f"{'m':>2} | {'cond':>16} | " + " ".join(f"{'lv'+str(k+1):>6}" for k in range(6)))
for m in MS:
    for c in ["scratch_rl_parse", "pretrain_rl_parse", "pretrain_only"]:
        pl = gen(m, c, "sampled")["parse_per_level"]
        print(f"{m:>2} | {c:>16} | " + " ".join(f"{x:>6.3f}" for x in pl))

print()
print("=" * 100)
print("EXACT ARM — exact-match acc (greedy / sampled) vs ceilings")
print("=" * 100)
print(f"{'m':>2} | {'ceil g/s':>15} | {'pretrained g/s':>15} | "
      f"{'scratch g/s':>15} | {'pre_rl g/s':>15} | {'pre_only g/s':>15}")
for m in MS:
    r = refs(m)["exact_match"]
    def gs(c=None):
        if c is None:
            return f"{pre_gen(m)['exact_overall']:.3f}/{pre_gen(m,'sampled')['exact_overall']:.3f}"
        return (f"{gen(m,c)['exact_overall']:.3f}/"
                f"{gen(m,c,'sampled')['exact_overall']:.3f}")
    print(f"{m:>2} | {r['greedy_ceiling']['overall']:.3f}/"
          f"{r['sampled_ceiling']['overall']:.3f}   | {gs():>15} | "
          f"{gs('scratch_rl_exact'):>15} | {gs('pretrain_rl_exact'):>15} | "
          f"{gs('pretrain_only'):>15}")

print()
print("=" * 100)
print("FRYING — NTP val loss (excess over Bayes floor) after 6000 finetune steps")
print("=" * 100)
print(f"{'m':>2} | {'pretrain':>9} | {'pre_rl_exact':>12} | {'pre_rl_parse':>12} | "
      f"{'pre_only':>9} | {'scr_exact':>10} | {'scr_parse':>10}")
for m in MS:
    fl = refs(m)["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    row = [D[m]["pretrain"]["best_val"] - fl]
    for c in ["pretrain_rl_exact", "pretrain_rl_parse", "pretrain_only",
              "scratch_rl_exact", "scratch_rl_parse"]:
        row.append(cond(m, c)["val_loss"] - fl)
    print(f"{m:>2} | {row[0]:>9.4f} | {row[1]:>12.4f} | {row[2]:>12.4f} | "
          f"{row[3]:>9.4f} | {row[4]:>10.4f} | {row[5]:>10.4f}")

print()
print("=" * 100)
print("ETA2 — top-level (L5) feature eta2 progressivity: block0 -> block5 (ratio = progressive computation)")
print("=" * 100)
print(f"{'m':>2} | {'cond':>16} | {'blk0':>7} {'blk5':>7} {'ratio':>7}")
for m in MS:
    rows = [("pretrained", D[m]["pretrain"]["eval"]["per_layer_eta2"])]
    for c in ["pretrain_rl_exact", "pretrain_rl_parse", "pretrain_only",
              "scratch_rl_exact", "scratch_rl_parse"]:
        rows.append((c, cond(m, c)["per_layer_eta2"]))
    for name, e in rows:
        b0 = e["post_block0"]["level_5"]["feature_eta2"]
        b5 = e["post_block5"]["level_5"]["feature_eta2"]
        ratio = b0 / b5 if b5 > 1e-9 else float("inf")
        print(f"{m:>2} | {name:>16} | {b0:>7.3f} {b5:>7.3f} {ratio:>7.2f}")

print()
print("=" * 100)
print("REWARD TRAJECTORIES — trained reward at steps 250 / 1500 / 6000 (RL conds)")
print("=" * 100)
for m in MS:
    for c in ["scratch_rl_exact", "pretrain_rl_exact",
              "scratch_rl_parse", "pretrain_rl_parse"]:
        tr = D[m]["conditions"][c]["trajectory"]
        by = {t["step"]: t for t in tr}
        pts = []
        for st in [250, 1500, 6000]:
            t = by.get(st, {})
            r = t.get("reward", t.get("reward_exact"))
            pts.append(f"{st}:{r:.3f}" if r is not None else f"{st}:--")
        print(f"m={m} {c:>18}: " + "  ".join(pts))
