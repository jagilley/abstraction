"""Checkpoint probes for `terminal_only`. Trains nothing.

The main script already emits the two structural readouts that matter most — `stage_all` (the
tied map applied once to a bounded-quotient query it never trained on) and the per-stage
diagnostics. This file adds the three that need a checkpoint and a different pool.

**`reprobe`** runs all three on one loaded arm:

  1. **`depth`** — is the stage map an *operator*? Pad the dividend with `k` extra leading
     radix digits (all zero) and run `S + k` stages. A genuine long-division stage satisfies
     `r ← (r·R + 0) mod N` with `r = 0`, so the extra stages are exact no-ops and accuracy is
     unchanged. A model that has learned an `S`-specific circuit — a positional schedule
     rather than a tied map — degrades. This is `ballistic_depth`'s composition-horizon
     question one level down, and it is free.
  2. **`restart`** — where does the chain lose it? Inject the *true* partial remainder at
     stage `j` and run stages `j … S-1` natively. `j = 0` is the ordinary chain, `j = S-1`
     is one native stage from a true state. The profile separates a poor per-stage rate
     (flat and low) from error propagation (rising steeply in `j`), which is
     `ballistic_depth` §8's cold-start probe at the stage seam.
  3. **`stagefn`** — what function did the stage map learn? Exhaustive over `y' ∈ [0, R·N)`
     for every training and held-out modulus, broken down by the true quotient `y' // N`.
     Long division needs every quotient digit `0 … R-1`; a model that only ever sees small
     quotients on its own chain trajectories will be good at some and not others, and the
     `q = 0` column is the no-reduction freebie the whole program floors against.

**`crossprobe`** scores a `terminal_only` arm and a donor `staged_reduce` arm on one shared
pool, so "did the terminal-only arm find the same decomposition" is answered on identical
draws rather than on two independently built pools.

Usage:
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/reprobe.py::reprobe \\
    --tag to3c --arms "to3_st10_many"
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/reprobe.py::crossprobe \\
    --tag to3c --arm to3_st10_many --ref-tag srb1 --ref-arm sr3_r10_many
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

GPU = os.environ.get("OLD_GPU", "L4")


def _build(tag, arm_name, seed, device, node="terminal_only"):
    """Load a checkpoint and rebuild everything the training run had.

    `node="terminal_only"` reads this node's arms; `node="staged_reduce"` reads the donor's,
    so `crossprobe` can put both models on one pool with one implementation of the chain.
    """
    import numpy as np
    import torch

    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    if node == "terminal_only":
        from one_layer_deeper.rule_acquisition.staged_reduce.terminal_only.terminal_only import (
            ARMS, BASE_ARM, DIGIT_OFFSET, DIV_TOK, HASH_M, _make_model, stages_for,
        )
        ck_path = (Path(DATA_DIR) / "staged_reduce" / "terminal_only" / str(tag)
                   / "ckpt" / f"{arm_name}_seed{seed}.pt")
    else:
        from one_layer_deeper.rule_acquisition.staged_reduce.staged_reduce import (
            ARMS, BASE_ARM, DIGIT_OFFSET, DIV_TOK, HASH_M, _make_model, stages_for,
        )
        ck_path = (Path(DATA_DIR) / "staged_reduce" / str(tag)
                   / "ckpt" / f"{arm_name}_seed{seed}.pt")
    if not ck_path.exists():
        raise FileNotFoundError(str(ck_path))
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    arm = {**BASE_ARM, **ARMS[arm_name]}
    w = arm["n_digits"]
    radix_eff, S = stages_for(arm["radix"], w)
    mode = arm.get("stage_mode", "st")

    family = ModulusFamily(
        n_digits=w, min_margin=arm["min_margin"], min_factor=cfg["min_factor"],
        heldout_fraction=cfg["heldout_mod_fraction"], seed=cfg["family_seed"],
    )
    fam_train, fam_held = family.split()
    if arm["max_moduli"] and len(fam_train) > arm["max_moduli"]:
        pick = np.random.default_rng(cfg["family_seed"] + 1).choice(
            len(fam_train), arm["max_moduli"], replace=False)
        fam_train = sorted(fam_train[i] for i in pick)
    if arm["max_moduli"] and len(fam_held) > 4 * arm["max_moduli"]:
        pick = np.random.default_rng(cfg["family_seed"] + 2).choice(
            len(fam_held), 4 * arm["max_moduli"], replace=False)
        fam_held = sorted(fam_held[i] for i in pick)
    train_mods = [m[0] for m in fam_train]
    held_mods = [m[0] for m in fam_held]

    w_x, n_ans = 2 * w, w
    max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
    read_pos = max_len - 1
    x_start = 4 + w
    import math as _math
    m_c = int(round(_math.log10(arm["radix"]))) if arm["radix"] else 2 * w
    r_slot0 = x_start + (w_x - w - m_c)
    r_slot1 = r_slot0 + w
    pow10 = torch.tensor([10 ** (w - 1 - k) for k in range(w)], device=device)
    thr = int(round(arm["train_frac"] * HASH_M))

    def digits_of(v, width):
        return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

    def value_of(dig):
        return (dig * pow10).sum(-1)

    def is_train(n_vals, x_vals):
        return ((x_vals * 2654435761 + n_vals * 40503) % HASH_M) < thr

    def prompts_for(n_vals, x_vals):
        rows = x_vals.shape[0]
        col = lambda t: torch.full((rows, 1), t, dtype=torch.long, device=device)
        return torch.cat([
            col(TOKEN_IDS["BOS"]), col(DIV_TOK), col(TOKEN_IDS["N"]),
            digits_of(n_vals, w) + DIGIT_OFFSET, col(TOKEN_IDS["X"]),
            digits_of(x_vals, w_x) + DIGIT_OFFSET, col(TOKEN_IDS["ANS"]),
        ], dim=1)

    model = _make_model(cfg, arm, max_len, n_ans, device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    @torch.no_grad()
    def stage_once(n_vals, inp):
        """One hard application of the tied map to the dividend `inp`."""
        h = model.roll(model.enc(prompts_for(n_vals, inp), read_pos))
        return value_of(model.dec(h).argmax(-1))

    @torch.no_grad()
    def chain(n_vals, y_vals, n_stages=None, restart_at=0):
        """The hard chain. `restart_at=j` injects the true partial remainder before stage j.

        The donor's arms were only ever chained hard, and this node's `st` arms are hard by
        construction; a `soft` arm's *native* forward is scored by the training script, so
        everything here is the argmax chain and the two are directly comparable.
        """
        ns = n_stages or S
        r = torch.zeros_like(y_vals)
        for j, i in enumerate(range(ns - 1, -1, -1)):
            if j == restart_at and restart_at > 0:
                r = (y_vals // radix_eff ** (i + 1)) % n_vals
            c = (y_vals // radix_eff**i) % radix_eff
            r = stage_once(n_vals, r * radix_eff + c)
        return r

    return dict(model=model, cfg=cfg, arm=arm, w=w, S=S, radix_eff=radix_eff, mode=mode,
                train_mods=train_mods, held_mods=held_mods, is_train=is_train,
                stage_once=stage_once, chain=chain, digits_of=digits_of, value_of=value_of,
                prompts_for=prompts_for, read_pos=read_pos, n_ans=n_ans, ck=ck)


def _chain_pool(mods_np, want_train, total, is_train, device, seed=777):
    """The donor's `_chain_pool`, verbatim, so pools stay bit-identical across nodes."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    n_mods = int(mods_np.size)
    per = max(64, total // max(1, n_mods))
    mods = torch.tensor(mods_np, dtype=torch.long, device=device)
    out_n, out_y = [], []
    for _ in range(64):
        nv = mods.repeat_interleave(per)
        u = torch.rand(nv.numel(), device=device, generator=gen, dtype=torch.float64)
        yv = torch.minimum((u * (nv * nv)).long(), nv * nv - 1)
        if want_train is not None:
            keep = is_train(nv, yv) == want_train
            nv, yv = nv[keep], yv[keep]
        out_n.append(nv)
        out_y.append(yv)
        if sum(t.numel() for t in out_n) >= total:
            break
    nv, yv = torch.cat(out_n), torch.cat(out_y)
    pick = torch.randperm(nv.numel(), generator=gen, device=device)[:total]
    return nv[pick], yv[pick]


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=7200, memory=32768)
def reprobe(tag: str = "to3c", arms: str = "to3_st10_many", seed: int = 0,
            n_chain: int = 65536, extra_depths: str = "0,1,2,4", chunk: int = 8192):
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extras = [int(x) for x in extra_depths.split(",") if x.strip()]
    out = {"tag": tag, "seed": seed, "arms": {}}

    for arm_name in [a.strip() for a in arms.split(",") if a.strip()]:
        P = _build(tag, arm_name, seed, device)
        S, R = P["S"], P["radix_eff"]
        mods_tr = np.array(P["train_mods"], dtype=np.int64)
        mods_he = np.array(P["held_mods"], dtype=np.int64)
        rec = {"S": S, "radix": R, "mode": P["mode"],
               "n_train_moduli": int(mods_tr.size), "step": P["ck"].get("step")}
        pools = {
            "chain_heldout_y": _chain_pool(mods_tr, False, n_chain, P["is_train"], device),
            "chain_heldout_n": _chain_pool(mods_he, None, n_chain, P["is_train"], device),
        }
        for pname, (nv, yv) in pools.items():
            print(f"[{arm_name}] pool {pname}: n={nv.numel()} floor="
                  f"{(yv < nv).float().mean():.4f}", flush=True)

        # 1. depth — extra leading zero stages must be no-ops for a genuine operator
        rec["depth"] = {}
        for pname, (nv, yv) in pools.items():
            row = {}
            for k in extras:
                ok = 0
                for i0 in range(0, nv.numel(), chunk):
                    s = slice(i0, min(i0 + chunk, nv.numel()))
                    r = P["chain"](nv[s], yv[s], n_stages=S + k)
                    ok += int((r == (yv[s] % nv[s])).sum())
                row[f"S+{k}"] = ok / nv.numel()
                print(f"[{arm_name}] depth {pname} S+{k}: {row[f'S+{k}']:.4f}", flush=True)
            rec["depth"][pname] = row

        # 2. restart — inject the true partial remainder before stage j
        rec["restart"] = {}
        for pname, (nv, yv) in pools.items():
            row = {}
            for j in range(S):
                ok = 0
                for i0 in range(0, nv.numel(), chunk):
                    s = slice(i0, min(i0 + chunk, nv.numel()))
                    r = P["chain"](nv[s], yv[s], restart_at=j)
                    ok += int((r == (yv[s] % nv[s])).sum())
                row[f"j={j}"] = ok / nv.numel()
            rec["restart"][pname] = row
            print(f"[{arm_name}] restart {pname}: "
                  + " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)

        # 3. stagefn — the tied map, exhaustive over [0, R*N), by true quotient digit
        rec["stagefn"] = {}
        if P["arm"]["radix"]:
            for label, mods in (("train_N", mods_tr), ("heldout_N", mods_he)):
                tot = np.zeros(R, dtype=np.int64)
                hit = np.zeros(R, dtype=np.int64)
                for n_val in mods.tolist():
                    hi = P["arm"]["radix"] * n_val
                    yv = torch.arange(hi, device=device, dtype=torch.long)
                    nv = torch.full_like(yv, n_val)
                    for i0 in range(0, yv.numel(), chunk):
                        s = slice(i0, min(i0 + chunk, yv.numel()))
                        pred = P["stage_once"](nv[s], yv[s])
                        q = (yv[s] // n_val).cpu().numpy()
                        good = (pred == (yv[s] % n_val)).cpu().numpy()
                        np.add.at(tot, q, 1)
                        np.add.at(hit, q, good.astype(np.int64))
                rec["stagefn"][label] = {
                    "overall": float(hit.sum() / max(1, tot.sum())),
                    "n": int(tot.sum()),
                    "by_quotient": {str(q): float(hit[q] / t) if t else None
                                    for q, t in enumerate(tot.tolist())},
                }
                print(f"[{arm_name}] stagefn {label}: overall="
                      f"{rec['stagefn'][label]['overall']:.4f} by-q "
                      + " ".join(f"{q}:{v:.3f}" for q, v in
                                 rec["stagefn"][label]["by_quotient"].items()
                                 if v is not None), flush=True)
        out["arms"][arm_name] = rec

    d = Path(DATA_DIR) / "staged_reduce" / "terminal_only" / str(tag)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"reprobe_seed{seed}.json").write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"[saved] /staged_reduce/terminal_only/{tag}/reprobe_seed{seed}.json", flush=True)
    return out


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=7200, memory=32768)
def crossprobe(tag: str = "to3c", arm: str = "to3_st10_many",
               ref_tag: str = "srb1", ref_arm: str = "sr3_r10_many",
               seed: int = 0, n_chain: int = 65536, chunk: int = 8192):
    """Score a terminal-only arm and a donor arm on one shared pool, per modulus."""
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = _build(tag, arm, seed, device, node="terminal_only")
    B = _build(ref_tag, ref_arm, seed, device, node="staged_reduce")
    assert A["S"] == B["S"] and A["radix_eff"] == B["radix_eff"], "radix/stage mismatch"
    mods_tr = np.array(A["train_mods"], dtype=np.int64)
    mods_he = np.array(A["held_mods"], dtype=np.int64)
    assert sorted(mods_tr.tolist()) == sorted(B["train_mods"]), "modulus sets differ"

    out = {"tag": tag, "arm": arm, "ref_tag": ref_tag, "ref_arm": ref_arm, "pools": {}}
    for pname, mods, want in (("chain_heldout_y", mods_tr, False),
                              ("chain_heldout_n", mods_he, None)):
        nv, yv = _chain_pool(mods, want, n_chain, A["is_train"], device)
        rows = {}
        for label, P in (("terminal_only", A), ("donor", B)):
            okv = torch.zeros(nv.numel(), dtype=torch.bool, device=device)
            for i0 in range(0, nv.numel(), chunk):
                s = slice(i0, min(i0 + chunk, nv.numel()))
                okv[s] = P["chain"](nv[s], yv[s]) == (yv[s] % nv[s])
            per = {int(m): float(okv[nv == m].float().mean()) for m in torch.unique(nv)}
            vals = sorted(per.values())
            rows[label] = {"acc": float(okv.float().mean()), "n": int(nv.numel()),
                           "per_modulus_min": vals[0], "per_modulus_max": vals[-1],
                           "per_modulus_median": vals[len(vals) // 2],
                           "per_modulus": per if len(per) <= 64 else None}
            print(f"[cross] {pname} {label}: {rows[label]['acc']:.4f} "
                  f"(per-mod {vals[0]:.3f}-{vals[-1]:.3f})", flush=True)
        rows["floor"] = float((yv < nv).float().mean())
        out["pools"][pname] = rows

    d = Path(DATA_DIR) / "staged_reduce" / "terminal_only" / str(tag)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"crossprobe_seed{seed}.json").write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"[saved] /staged_reduce/terminal_only/{tag}/crossprobe_seed{seed}.json", flush=True)
    return out
