"""Three-way comparison: expert iteration (ei01) vs KL-anchored RL (kl01) vs base."""
import json

MS = [1, 2, 3, 4, 6]
B = {m: json.load(open(f"m{m}.json")) for m in MS}
K = {m: json.load(open(f"kl_m{m}.json")) for m in MS}
E = {m: json.load(open(f"ei_m{m}.json")) for m in MS}

def cond(d, m, c):
    ck = d[m]["conditions"][c]["checkpoints"]
    return ck[max(ck, key=int)]

def pre(m, mode, key):
    return B[m]["pretrain"]["eval"]["generation"][mode][key]

print("=" * 108)
print("PARSE VERIFIER — sampled parse validity (policy distribution), root validity, NTP excess")
print("=" * 108)
print(f"{'m':>2} | {'pretrained':>10} | {'RL(unanch)':>10} | {'RL+KL':>7} | {'EI':>7} || "
      f"{'root: pre':>9} | {'RL+KL':>7} | {'EI':>7} || {'excess: KL':>10} | {'EI':>7}")
for m in MS:
    fl = B[m]["references"]["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    bu = cond(B, m, "pretrain_rl_parse")
    kl = cond(K, m, "pretrain_rl_kl_parse")
    ei = cond(E, m, "ei_parse")
    pr_root = pre(m, "sampled", "parse_per_level")[-1]
    kl_root = kl["generation"]["sampled"]["parse_per_level"][-1]
    ei_root = ei["generation"]["sampled"]["parse_per_level"][-1]
    print(f"{m:>2} | {pre(m,'sampled','parse_overall'):>10.3f} | "
          f"{bu['generation']['sampled']['parse_overall']:>10.3f} | "
          f"{kl['generation']['sampled']['parse_overall']:>7.3f} | "
          f"{ei['generation']['sampled']['parse_overall']:>7.3f} || "
          f"{pr_root:>9.3f} | {kl_root:>7.3f} | {ei_root:>7.3f} || "
          f"{kl['val_loss']-fl:>10.3f} | {ei['val_loss']-fl:>7.3f}")

print()
print("EI_parse round-by-round: sample reward (policy quality) and winner reward (selection)")
for m in MS:
    tr = E[m]["conditions"]["ei_parse"]["trajectory"]
    print(f"  m={m} sample: " + " ".join(f"{t['sample_reward_mean']:.3f}" for t in tr)
          + " | winner: " + " ".join(f"{t['winner_reward_mean']:.3f}" for t in tr))

print()
print("EI_parse per-round SAMPLED eval parse (does the distribution improve?)")
for m in MS:
    ck = E[m]["conditions"]["ei_parse"]["checkpoints"]
    vals = [ck[r]["generation"]["sampled"]["parse_overall"] for r in sorted(ck, key=int)]
    print(f"  m={m}: " + " ".join(f"{x:.3f}" for x in vals))

print()
print("=" * 108)
print("EXACT VERIFIER — greedy / sampled vs ceilings")
print("=" * 108)
print(f"{'m':>2} | {'ceil g/s':>13} | {'pretrained':>13} | {'RL+KL':>13} | {'EI':>13} | "
      f"{'EI val excess':>13}")
for m in MS:
    r = B[m]["references"]["exact_match"]
    fl = B[m]["references"]["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    kl = cond(K, m, "pretrain_rl_kl_exact")
    ei = cond(E, m, "ei_exact")
    def gs(src):
        return (f"{src['generation']['greedy']['exact_overall']:.3f}/"
                f"{src['generation']['sampled']['exact_overall']:.3f}")
    print(f"{m:>2} | {r['greedy_ceiling']['overall']:.3f}/{r['sampled_ceiling']['overall']:.3f}"
          f"   | {pre(m,'greedy','exact_overall'):.3f}/{pre(m,'sampled','exact_overall'):.3f}"
          f"   | {gs(kl):>13} | {gs(ei):>13} | {ei['val_loss']-fl:>13.3f}")

print()
print("EI_exact round-by-round sample reward:")
for m in MS:
    tr = E[m]["conditions"]["ei_exact"]["trajectory"]
    print(f"  m={m}: " + " ".join(f"{t['sample_reward_mean']:.3f}" for t in tr))

print()
print("=" * 108)
print("DEEP STRUCTURE — per-level NTP loss minus Bayes floor (levels 3,4,5), pretrained vs EI_parse")
print("=" * 108)
for m in MS:
    bay = {int(k): d["bayes_nats"]
           for k, d in B[m]["references"]["ntp_bayes"]["per_level"].items()}
    p_lv = B[m]["pretrain"]["eval"]["per_level_ntp"]
    e_lv = cond(E, m, "ei_parse")["per_level_ntp"]
    row_p = " ".join(f"L{l}:{p_lv[f'L{l}']-bay[l]:+.3f}" for l in [3, 4, 5])
    row_e = " ".join(f"L{l}:{e_lv[f'L{l}']-bay[l]:+.3f}" for l in [3, 4, 5])
    print(f"  m={m} pretrained: {row_p}")
    print(f"  m={m} ei_parse  : {row_e}")

print()
print("ETA2 L5 blk0->blk5:  pretrained | ei_exact | ei_parse")
for m in MS:
    def e2(d, c=None):
        e = (d[m]["pretrain"]["eval"]["per_layer_eta2"] if c is None
             else cond(d, m, c)["per_layer_eta2"])
        return (f"{e['post_block0']['level_5']['feature_eta2']:.3f}->"
                f"{e['post_block5']['level_5']['feature_eta2']:.3f}")
    print(f"  m={m}: {e2(B):>13} | {e2(E,'ei_exact'):>13} | {e2(E,'ei_parse'):>13}")

print()
print("WINNER DATA QUALITY (what SFT trains on): ei_parse round-1 vs round-6 winner root validity")
for m in MS:
    tr = E[m]["conditions"]["ei_parse"]["trajectory"]
    r1 = tr[0]["winner_parse_per_level"][-1]
    rT = tr[-1]["winner_parse_per_level"][-1]
    print(f"  m={m}: round1={r1:.3f} round{len(tr)}={rT:.3f}")
