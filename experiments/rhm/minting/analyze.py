"""Re-derive the minting contrasts from saved arm JSONs, without re-running anything.

Fetch first (from experiments/):
  modal volume get rhm-scaling-data rhm_minting/<run_dir> /tmp/mint --force
Then:
  python3 rhm/minting/analyze.py /tmp/mint/<run_dir>

Prints the four readouts the loop reports in-flight, plus the contrasts §7's falsifiers are
stated in: verifier - narrow-static (is minting an expansion channel at all), mirror -
verifier (does the acceptance signal's TYPE matter), and mirror/random - verifier (is any
apparent mirror failure just the mint-and-accept drift §7 warns about, rather than the
mirror geometry).
"""

import json
import pathlib
import sys


def _load(run_dir):
    from rhm.minting.mint_loop import ARM_ORDER
    found = {}
    for p in sorted(pathlib.Path(run_dir).glob("*.json")):
        if p.name == "all.json":
            continue
        r = json.loads(p.read_text())
        found[f"{r['breadth']}_{r['acceptance']}"] = r
    if not found:
        raise SystemExit(f"no arm JSONs in {run_dir}")
    # print along the mirror -> non-mirror axis, not alphabetically
    arms = {k: found.pop(k) for k in ARM_ORDER if k in found}
    arms.update(found)
    return arms


def _row(label, cells, fmt="{:>8.3f}", width=22):
    out = f"{label:<{width}}"
    for c in cells:
        out += fmt.format(c) if c is not None else f"{'-':>8}"
    return out


def main(run_dir):
    arms = _load(run_dir)
    any_arm = next(iter(arms.values()))
    cfg = any_arm["config"]
    L, R = cfg["depth"], len(any_arm["rounds"])
    dl = [f"d{i}" for i in range(1, L + 1)]
    rhdr = [f"r{i}" for i in range(R)]
    print(f"run: {run_dir}")
    print(f"regime v{cfg['v']}_s{cfg['s']}_L{L}_m{cfg['m']}  T={cfg['s']**L}  "
          f"seed_pool={cfg['seed_pool']}  rounds={R}x{cfg['steps_per_round']}  "
          f"candidates={cfg['n_candidates']}  quota={cfg['accept_quota']}  "
          f"prefix_frac={cfg['prefix_frac']}  temp={cfg['temperature']}")

    def table(title, note, get):
        print(f"\n{'='*100}\n{title}\n  {note}")
        print(_row("arm", []) + "".join(f"{h:>8}" for h in rhdr))
        for k, r in arms.items():
            print(_row(k, [get(rd) for rd in r["rounds"]]))

    print(f"\n{'='*100}\nDEPTH RECOVERY on the frozen FULL-TREE probe (MLP best-over-blocks)"
          f"   chance=1/{cfg['v']}={1/cfg['v']:.3f}")
    print(f"  external anchors (Exp 1, unmetered pool): NTP floor root 0.08 / oracle-aux 0.80")
    print(f"{'arm':<22}" + "".join(f"{d:>8}" for d in dl) +
          f"{'val_full':>10}{'val_comp':>10}{'pool':>8}")
    for k, r in arms.items():
        mlp = r["recovery"]["full"]["mlp_best"]
        vl = r["rounds"][-1]["val"]
        print(f"{k:<22}" + "".join(f"{mlp[d]:>8.3f}" for d in dl) +
              f"{vl['full']:>10.4f}{vl['complement']:>10.4f}{r['final_pool_size']:>8}")

    base = arms.get("narrow_static")
    if base:
        print(f"\n  contrasts vs narrow_static (the §7 falsifiers):")
        bm = base["recovery"]["full"]["mlp_best"]
        for k, r in arms.items():
            if k == "narrow_static":
                continue
            m = r["recovery"]["full"]["mlp_best"]
            print(_row(f"  {k} - static", [m[d] - bm[d] for d in dl], "{:>+8.3f}", 24))
        ver = arms.get("narrow_verifier")
        if ver:
            print(f"\n  contrasts vs narrow_verifier (does acceptance TYPE matter):")
            vm = ver["recovery"]["full"]["mlp_best"]
            for k in ("narrow_mirror", "narrow_mirror_dedup", "narrow_peer",
                      "narrow_verifier@4", "narrow_random", "narrow_real_data"):
                if k in arms:
                    m = arms[k]["recovery"]["full"]["mlp_best"]
                    print(_row(f"  {k} - verifier", [m[d] - vm[d] for d in dl],
                               "{:>+8.3f}", 26))

    table("RULE-VIOLATION RATE in the accepted pool (§5's mirror degeneration)",
          "0.000 by construction for the verifier arm; the measurement is of mirror/random",
          lambda rd: (1 - rd["accepted"]["valid_frac"]) if "accepted" in rd else None)
    table("GENERATIVE VALIDITY on frozen full-tree prefixes (arm-comparable)",
          "fraction of the learner's own completions that parse all the way to a root",
          lambda rd: rd["gen_probe"]["full"]["valid_frac"])
    table("MEAN PARSE DEPTH on frozen full-tree prefixes",
          f"levels composed before the first violation, of {L}",
          lambda rd: rd["gen_probe"]["full"]["parse_level_mean"])
    table("SUPPORT EXPANSION: accepted whose root set lies ENTIRELY outside the seed's S",
          "0.000 for the narrow seed by construction; ~0.66 for a full-tree corpus",
          lambda rd: rd["accepted"]["root_only_outside_S"] if "accepted" in rd else None)
    table("MODE COLLAPSE: aligned-pair entropy (nats) of the accepted pool",
          "real full-tree data sits at ~3.90; falling = collapse, rising = drift to junk",
          lambda rd: rd["accepted"]["pair_entropy_nats"] if "accepted" in rd else None)
    table("DUPLICATION: accepted already present in the pool (is minting just replay?)",
          "high = the mint is regurgitating its own seed rather than making new content",
          lambda rd: rd["accepted"].get("dup_with_pool_frac")
          if "accepted" in rd else None)
    table("ACCEPTANCE SHARPENING: mean own-logprob of accepted minus that of candidates",
          "the mirror arm should be most positive -- it selects on exactly this quantity",
          lambda rd: (rd["accepted"]["mean_logprob"] - rd["candidates"]["mean_logprob"])
          if "accepted" in rd and "candidates" in rd else None)

    print(f"\n{'='*100}\nVERIFIER YIELD (accepted / candidates before the quota binds)")
    print(_row("arm", []) + "".join(f"{h:>8}" for h in rhdr))
    for k, r in arms.items():
        print(_row(k, [rd["accepted"].get("yield_", 0) / cfg["n_candidates"]
                       if "accepted" in rd else None for rd in r["rounds"]], "{:>8.4f}"))
    print(f"  quota met every round: "
          f"{ {k: all(rd['accepted'].get('quota_met') for rd in r['rounds'] if 'accepted' in rd) for k, r in arms.items()} }")
    print(f"{'='*100}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
