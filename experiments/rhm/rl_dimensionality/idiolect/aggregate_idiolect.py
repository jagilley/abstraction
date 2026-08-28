"""Cross-m tables for the idiolect sweep.

Usage (from this directory). The committed result files are already here;
to regenerate them, re-run the sweep and pull:
  modal volume get rhm-scaling-data \
      rl_dimensionality/idiolect/idiolect_results_seed43.json .
  python3 aggregate_idiolect.py idiolect_results_seed43.json
"""

import json
import sys

M_ORDER = ["1", "2", "3", "4", "6"]

# label -> display name, in report order
MECHANISMS = [
    ("dgp", "DGP floor"),
    ("pretrained", "pretrained"),
    ("pretrain_only", "+ continued NTP"),
    ("reinforce_exact", "REINFORCE (exact)"),
    ("reinforce_parse", "REINFORCE (parse)"),
    ("kl_exact", "+KL (exact)"),
    ("kl_parse", "+KL (parse)"),
    ("ei_exact", "EI (exact)"),
    ("ei_parse", "EI (parse)"),
    ("ei_exact_rerun", "EI (exact) rerun"),
    ("ei_parse_rerun", "EI (parse) rerun"),
    ("ei_exact_seed43", "EI (exact) seed 43"),
    ("ei_parse_seed43", "EI (parse) seed 43"),
]


def get(d, m, label, mode="sampled", field="kl_cond"):
    c = d["by_m"].get(m, {}).get("conditions", {}).get(label)
    if not c or mode not in c:
        return None
    o = c[mode]
    if field in ("root_valid",):
        return o.get(field)
    return o["overall"].get(field)


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def table(d, field, title, mode="sampled", nd=4):
    print(f"\n### {title}")
    hdr = "| mechanism | " + " | ".join(f"m={m}" for m in M_ORDER) + " |"
    print(hdr)
    print("|" + "---|" * (len(M_ORDER) + 1))
    for label, name in MECHANISMS:
        row = [fmt(get(d, m, label, mode, field), nd) for m in M_ORDER]
        if all(v == "—" for v in row):
            continue
        print(f"| {name} | " + " | ".join(row) + " |")
    ceil = [d["by_m"].get(m, {}).get("log2_m") for m in M_ORDER]
    if field.startswith("kl"):
        print("| *max possible (log2 m)* | "
              + " | ".join(fmt(c, nd) for c in ceil) + " |")


def rounds_table(d, arm="parse", seed_sfx=""):
    print(f"\n### EI round-by-round ({arm} verifier{seed_sfx or ''}, sampled): "
          f"drift and grammaticality climb together")
    print("| m | round | kl_cond (bits) | root valid | node valid |")
    print("|---|---|---|---|---|")
    for m in M_ORDER:
        for r in range(1, 7):
            lab = f"ei_{arm}{seed_sfx}_r{r}"
            k = get(d, m, lab, "sampled", "kl_cond")
            if k is None:
                continue
            print(f"| {m} | {r} | {fmt(k)} | "
                  f"{fmt(get(d, m, lab, 'sampled', 'root_valid'))} | "
                  f"{fmt(get(d, m, lab, 'sampled', 'valid_frac'))} |")


def js_table(d, pairs):
    print("\n### Is the drift private? (JS between synonym-choice "
          "distributions, bits)")
    hdr = "| pair | " + " | ".join(f"m={m}" for m in M_ORDER) + " |"
    print(hdr)
    print("|" + "---|" * (len(M_ORDER) + 1))
    for a, b, name in pairs:
        row = []
        for m in M_ORDER:
            js = d["by_m"].get(m, {}).get("js_sampled", {})
            val = js.get(f"{a}|{b}", js.get(f"{b}|{a}"))
            row.append(fmt(val))
        if all(v == "—" for v in row):
            continue
        print(f"| {name} | " + " | ".join(row) + " |")


def main(path):
    with open(path) as f:
        d = json.load(f)
    print(f"# Idiolect drift under reward optimization\n")
    print(f"n_eval = {d['config']['n_eval']} sequences per checkpoint, "
          f"temperature {d['config']['temperature']}, "
          f"prefix_len = {d['config']['prefix_len']}.")
    print("kl_cond = KL(model || DGP) on synonym choice, conditioned on the "
          "latent feature, in bits. 0 = speaks the corpus's language; "
          "log2(m) = always picks one synonym.")

    table(d, "kl_cond", "Synonym-choice drift, kl_cond (bits, sampled)")
    table(d, "kl_cond", "Synonym-choice drift, kl_cond (bits, greedy)",
          mode="greedy")
    table(d, "valid_frac", "Grammaticality of the same generations "
          "(fraction of model-chosen nodes that parse)")
    table(d, "root_valid", "Root validity (the reward-relevant number)")
    rounds_table(d, "parse")
    rounds_table(d, "exact")
    rounds_table(d, "parse", "_seed43")
    js_table(d, [
        ("ei_parse", "ei_parse_rerun", "EI parse vs same-seed rerun (noise floor)"),
        ("ei_exact", "ei_exact_rerun", "EI exact vs same-seed rerun (noise floor)"),
        ("dgp", "pretrained", "DGP vs pretrained"),
        ("pretrained", "ei_parse", "pretrained vs EI (parse)"),
        ("pretrained", "ei_exact", "pretrained vs EI (exact)"),
        ("pretrained", "kl_parse", "pretrained vs +KL (parse)"),
        ("ei_exact", "ei_parse", "EI exact vs EI parse (different verifier)"),
        ("ei_parse", "ei_parse_seed43", "** EI parse: seed 42 vs seed 43 (privacy test) **"),
        ("ei_exact", "ei_exact_seed43", "** EI exact: seed 42 vs seed 43 (privacy test) **"),
        ("pretrained", "ei_parse_seed43", "pretrained vs EI parse (seed 43)"),
    ])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "idiolect_results_seed43.json")
