"""Aggregate the activation-rank analysis across m (stdlib only).

Usage:
    # pull the analysis outputs (and results.json for the Bayes floors) locally:
    mkdir -p rank && for M in 1 2 3 4 6; do
      modal volume get rhm-scaling-data rl_dimensionality/rank_analysis/rank_m${M}.json rank/rank_m${M}.json --force
      modal volume get rhm-scaling-data rl_dimensionality/v8_s2_L6_m${M}_both_seed42/results.json rank/m${M}.json --force
    done
    python3 aggregate_rank.py rank [block]

Prints, per m x condition: NTP excess, suffix output entropy (sharpening),
data-PR / rollout-PR / rest-of-space PR at the chosen block (default
post_block5), context-collapse cosine, number of unique rollouts, and the
hierarchy decomposition at the last position; then the sharpening-vs-rank
scatter as a table, and the weight-update rank.
"""
import json
import math
import os
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "rank"
BLOCK = sys.argv[2] if len(sys.argv) > 2 else "post_block5"
MS = [m for m in [1, 2, 3, 4, 6] if os.path.exists(os.path.join(BASE, f"rank_m{m}.json"))]
R = {m: json.load(open(os.path.join(BASE, f"rank_m{m}.json"))) for m in MS}
FLOOR = {}
for m in MS:
    p = os.path.join(BASE, f"m{m}.json")
    if os.path.exists(p):
        FLOOR[m] = json.load(open(p))["references"]["ntp_bayes"]["entropy_rate_pos1plus_nats"]

ORDER = [
    ("init", "init"),
    ("pretrained", "pretrained"),
    ("pretrain_only_final", "cont. NTP"),
    ("pretrain_rl_exact_final", "REINFORCE exact"),
    ("pretrain_rl_parse_final", "REINFORCE parse"),
    ("scratch_rl_exact_final", "scratch RL exact"),
    ("scratch_rl_parse_final", "scratch RL parse"),
    ("pretrain_rl_kl_exact_final@kl01", "+KL exact"),
    ("pretrain_rl_kl_parse_final@kl01", "+KL parse"),
    ("ei_exact_final@ei01", "EI exact"),
    ("ei_parse_final@ei01", "EI parse"),
]


def ck(m, label):
    return R[m]["checkpoints"].get(label)


def f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


print(f"# Activation-rank analysis (block = {BLOCK}; data spectra on suffix positions)\n")
for m in MS:
    pre = ck(m, "pretrained")
    floor = FLOOR.get(m)
    print(f"## m={m}  (NTP Bayes floor {f(floor, 4)}; pretrained data-PR {f(pre['data'][BLOCK]['suffix']['pr'], 1)}, "
          f"rollout-PR {f(pre['rollout'][BLOCK]['pr'], 1)}, H={f(pre['output']['entropy_suffix_nats'])})\n")
    print("| condition | NTP excess | H(suffix) | ΔH | data-PR | Δ vs pre | rollout-PR | rest-PR (cond / pre same-subspace) | top-8 roll dirs: data var frac | mean cos | uniq/1000 | hier frac | hier PR | resid PR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, name in ORDER:
        c = ck(m, label)
        if c is None:
            continue
        d = c["data"][BLOCK]["suffix"]
        rest = c["rest"][BLOCK]
        pre_rest = rest.get("pretrained_rest_same_subspace", {}).get("pr")
        h = c["hier"][BLOCK]
        excess = c["val_loss"] - floor if floor is not None else None
        dH = c["output"]["entropy_suffix_nats"] - pre["output"]["entropy_suffix_nats"]
        dpr = d["pr"] - pre["data"][BLOCK]["suffix"]["pr"]
        print(f"| {name} | {f(excess)} | {f(c['output']['entropy_suffix_nats'])} | {f(dH)} | "
              f"{f(d['pr'], 1)} | {f(dpr, 1)} | {f(c['rollout'][BLOCK]['pr'], 1)} | "
              f"{f(rest['rest']['pr'], 1)} / {f(pre_rest, 1)} | {f(rest['rollout_topk_data_frac'])} | "
              f"{f(d['mean_cos'])} | {c['rollout_metrics']['n_unique_suffixes']} | "
              f"{f(h['explained_frac'])} | {f(h['explained']['pr'], 1)} | {f(h['residual']['pr'], 1)} |")
    print()

# ---- per-block PR profile for the key conditions ------------------------
print("## Data-PR per block (suffix positions), pretrained → condition\n")
blocks = [f"post_block{i}" for i in range(6)] + ["post_lnf"]
print("| m | condition | " + " | ".join(b.replace("post_", "") for b in blocks) + " |")
print("|---|---|" + "---|" * len(blocks))
for m in MS:
    for label, name in ORDER:
        c = ck(m, label)
        if c is None or label in ("init",):
            continue
        print(f"| {m} | {name} | " + " | ".join(f(c['data'][b]['suffix']['pr'], 1) for b in blocks) + " |")
print()

# ---- sharpening vs rank scatter -----------------------------------------
print("## Sharpening vs representation rank (all cells)\n")
print("ΔH = suffix output entropy minus pretrained (nats; negative = sharpened). "
      "log-ratio = ln(data-PR / pretrained data-PR) at the block. "
      "rest log-ratio = ln(rest-PR_cond / rest-PR_pre) in the SAME complement subspace.\n")
print("| m | condition | ΔH | NTP excess | data-PR log-ratio | rest log-ratio | lnf data-PR log-ratio | mean cos |")
print("|---|---|---|---|---|---|---|---|")
for m in MS:
    pre = ck(m, "pretrained")
    floor = FLOOR.get(m)
    for label, name in ORDER:
        if label in ("init", "pretrained"):
            continue
        c = ck(m, label)
        if c is None:
            continue
        dH = c["output"]["entropy_suffix_nats"] - pre["output"]["entropy_suffix_nats"]
        lr = math.log(c["data"][BLOCK]["suffix"]["pr"] / pre["data"][BLOCK]["suffix"]["pr"])
        rest = c["rest"][BLOCK]
        rlr = math.log(rest["rest"]["pr"] / rest["pretrained_rest_same_subspace"]["pr"])
        llr = math.log(c["data"]["post_lnf"]["suffix"]["pr"] / pre["data"]["post_lnf"]["suffix"]["pr"])
        excess = c["val_loss"] - floor if floor is not None else None
        print(f"| {m} | {name} | {f(dH)} | {f(excess)} | {f(lr)} | {f(rlr)} | {f(llr)} | {f(c['data'][BLOCK]['suffix']['mean_cos'])} |")
print()

# ---- weight-update rank ---------------------------------------------------
print("## Weight update ΔW = W_cond − W_pretrained (mean over 2-D matrices)\n")
print("| m | condition | mean PR(sv) | mean PR(sv²) | mean rel-Fro | lm_head PR(sv²) | wte PR(sv²) |")
print("|---|---|---|---|---|---|---|")
for m in MS:
    for label, name in ORDER:
        if label in ("init", "pretrained"):
            continue
        c = ck(m, label)
        if c is None:
            continue
        w = c["weight_delta_vs_pretrained"]
        if not w:
            continue
        mats = list(w.values())
        print(f"| {m} | {name} | {f(sum(x['pr'] for x in mats)/len(mats), 1)} | "
              f"{f(sum(x['pr_sq'] for x in mats)/len(mats), 1)} | {f(sum(x['rel_fro'] for x in mats)/len(mats))} | "
              f"{f(w.get('lm_head.weight', {}).get('pr_sq'), 1)} | {f(w.get('transformer.wte.weight', {}).get('pr_sq'), 1)} |")
print()

# ---- EI rounds (if ei02 round checkpoints were analyzed) -------------------
have_rounds = any(k.startswith("ei_") and "_r" in k for m in MS for k in R[m]["checkpoints"])
if have_rounds:
    print("## EI round-by-round (ei02 replicate)\n")
    print("| m | verifier | round | val | H(suffix) | data-PR | rest-PR (cond/pre) | mean cos | hier frac | resid PR | root valid (rollout) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MS:
        for rw in ("exact", "parse"):
            for rnd in range(1, 7):
                c = ck(m, f"ei_{rw}_r{rnd}@ei02")
                if c is None:
                    continue
                rest = c["rest"][BLOCK]
                print(f"| {m} | {rw} | {rnd} | {f(c['val_loss'])} | {f(c['output']['entropy_suffix_nats'])} | "
                      f"{f(c['data'][BLOCK]['suffix']['pr'], 1)} | {f(rest['rest']['pr'], 1)}/{f(rest['pretrained_rest_same_subspace']['pr'], 1)} | "
                      f"{f(c['data'][BLOCK]['suffix']['mean_cos'])} | {f(c['hier'][BLOCK]['explained_frac'])} | "
                      f"{f(c['hier'][BLOCK]['residual']['pr'], 1)} | {f(c['rollout_metrics']['parse_root'])} |")
    print()
