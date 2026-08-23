"""Compare KL-anchored (kl01) RL conditions vs the base sweep's unanchored ones."""
import json

MS = [1, 2, 3, 4, 6]
B = {m: json.load(open(f"m{m}.json")) for m in MS}
K = {m: json.load(open(f"kl_m{m}.json")) for m in MS}

def cond(d, m, c):
    ck = d[m]["conditions"][c]["checkpoints"]
    return ck[max(ck, key=int)]

def traj(d, m, c):
    tr = d[m]["conditions"][c]["trajectory"]
    by = {t["step"]: t for t in tr}
    out = []
    for st in [250, 1500, 6000]:
        t = by.get(st, {})
        r = t.get("reward", t.get("reward_exact"))
        out.append(f"{r:.3f}" if r is not None else "--")
    return "/".join(out)

print("SANITY: loaded-pretrain val vs base pretrain val")
for m in MS:
    print(f"  m={m}: base={B[m]['pretrain']['best_val']:.4f} "
          f"kl-run loaded={K[m]['pretrain']['best_val']:.4f}")

print()
print("=" * 110)
print("EXACT ARM: pretrain_rl (unanchored, base) vs pretrain_rl_kl (kl=0.1)")
print("=" * 110)
print(f"{'m':>2} | {'reward 250/1500/6000':^24} | {'val excess':>21} | "
      f"{'greedy exact':>17} | {'sampled exact':>17}")
print(f"{'':>2} | {'unanch':^11} {'KL':^11} | {'unanch':>9} {'KL':>9} | "
      f"{'unanch':>7} {'KL':>7} | {'unanch':>7} {'KL':>7}")
for m in MS:
    fl = B[m]["references"]["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    bu = cond(B, m, "pretrain_rl_exact")
    kl = cond(K, m, "pretrain_rl_kl_exact")
    print(f"{m:>2} | {traj(B,m,'pretrain_rl_exact'):^11} "
          f"{traj(K,m,'pretrain_rl_kl_exact'):^11} | "
          f"{bu['val_loss']-fl:>9.3f} {kl['val_loss']-fl:>9.3f} | "
          f"{bu['generation']['greedy']['exact_overall']:>7.3f} "
          f"{kl['generation']['greedy']['exact_overall']:>7.3f} | "
          f"{bu['generation']['sampled']['exact_overall']:>7.3f} "
          f"{kl['generation']['sampled']['exact_overall']:>7.3f}")
print("  (pretrained greedy exact / greedy ceiling per m: "
      + ", ".join(f"m={m}: "
                  f"{B[m]['pretrain']['eval']['generation']['greedy']['exact_overall']:.3f}"
                  f"/{B[m]['references']['exact_match']['greedy_ceiling']['overall']:.3f}"
                  for m in MS) + ")")

print()
print("=" * 110)
print("PARSE ARM: pretrain_rl (unanchored) vs pretrain_rl_kl (kl=0.1)")
print("=" * 110)
print(f"{'m':>2} | {'reward 250/1500/6000':^24} | {'val excess':>21} | "
      f"{'sampled parse':>17} | {'greedy parse':>17} | {'pretrained(s)':>13}")
for m in MS:
    fl = B[m]["references"]["ntp_bayes"]["entropy_rate_pos1plus_nats"]
    bu = cond(B, m, "pretrain_rl_parse")
    kl = cond(K, m, "pretrain_rl_kl_parse")
    ps = B[m]["pretrain"]["eval"]["generation"]["sampled"]["parse_overall"]
    print(f"{m:>2} | {traj(B,m,'pretrain_rl_parse'):^11} "
          f"{traj(K,m,'pretrain_rl_kl_parse'):^11} | "
          f"{bu['val_loss']-fl:>9.3f} {kl['val_loss']-fl:>9.3f} | "
          f"{bu['generation']['sampled']['parse_overall']:>7.3f} "
          f"{kl['generation']['sampled']['parse_overall']:>7.3f} | "
          f"{bu['generation']['greedy']['parse_overall']:>7.3f} "
          f"{kl['generation']['greedy']['parse_overall']:>7.3f} | {ps:>13.3f}")

print()
print("KL ARM per-level sampled parse fractions (lv1..lv6=root)")
for m in MS:
    pl = cond(K, m, "pretrain_rl_kl_parse")["generation"]["sampled"]["parse_per_level"]
    pp = B[m]["pretrain"]["eval"]["generation"]["sampled"]["parse_per_level"]
    print(f"  m={m} kl_parse : " + " ".join(f"{x:.3f}" for x in pl))
    print(f"  m={m} pretrain : " + " ".join(f"{x:.3f}" for x in pp))

print()
print("ETA2 L5 (blk0 -> blk5) — does KL-anchored RL preserve the basis?")
print(f"{'m':>2} | {'pretrained':>15} | {'unanch exact':>15} | {'KL exact':>15} | "
      f"{'unanch parse':>15} | {'KL parse':>15}")
for m in MS:
    def e2(d, c=None):
        e = (d[m]["pretrain"]["eval"]["per_layer_eta2"] if c is None
             else cond(d, m, c)["per_layer_eta2"])
        b0 = e["post_block0"]["level_5"]["feature_eta2"]
        b5 = e["post_block5"]["level_5"]["feature_eta2"]
        return f"{b0:.3f}->{b5:.3f}"
    print(f"{m:>2} | {e2(B):>15} | {e2(B,'pretrain_rl_exact'):>15} | "
          f"{e2(K,'pretrain_rl_kl_exact'):>15} | "
          f"{e2(B,'pretrain_rl_parse'):>15} | {e2(K,'pretrain_rl_kl_parse'):>15}")

print()
print("KL ARM exact-arm sampled vs greedy (sharpening check) + vs ceilings")
for m in MS:
    r = B[m]["references"]["exact_match"]
    kl = cond(K, m, "pretrain_rl_kl_exact")
    g = kl["generation"]["greedy"]["exact_overall"]
    s_ = kl["generation"]["sampled"]["exact_overall"]
    print(f"  m={m}: KL-RL greedy={g:.3f} sampled={s_:.3f} | "
          f"ceil greedy={r['greedy_ceiling']['overall']:.3f} "
          f"sampled={r['sampled_ceiling']['overall']:.3f} "
          f"blind={r['blind_floor']['overall']:.3f}")
