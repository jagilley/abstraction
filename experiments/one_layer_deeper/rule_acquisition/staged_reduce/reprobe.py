"""Checkpoint probes the training-time instrument could not supply. Trains nothing.

**`reprobe` — the seen-stage problem.** A staged arm's chain readout is on held-out
*composite* `(N, y)`, but held-out-ness is defined at the level of the composite problem, not
per stage query. On the way down the chain, roughly half of the `S` stage queries land in the
arm's own training half (measured: `stage_seen_frac ~ 0.49`). That is genuinely ambiguous
rather than simply a confound — if the claim is "decompose until every atom's input space is
coverable", stage queries landing inside the covered set *is the mechanism*, not a cheat. But
it means "staged beats monolithic on held-out `y`" is not clean as stated.

This resolves it by stratifying on `k`, the number of stage queries on the **true** stage path
that fall in the training half. `k` is a property of `(N, y)` and the hash alone —
model-independent, and identical for any two arms sharing a radix and a modulus set.

  1. `acc_by_k` — chain accuracy as a function of `k`, over the ordinary held-out pool. If it
     is flat in `k`, the seen/unseen distinction does not carry the result.
  2. `k0` — a pool built by **rejection sampling to `k = 0`**, i.e. composite problems none of
     whose stage queries was ever trained on. For the monolithic arm `S = 1` and this is
     exactly its ordinary held-out pool, so the comparison is apples to apples. `P(k=0) =
     2^-S`, so this is reachable for `S <= 8` and not for `S = 20`. NOTE: conditioning on
     `k = 0` is **not** distribution-free — a dividend whose leading radix chunks are zero
     re-queries `(N, 0)` every time, so small `y` is enriched whenever `is_train(N, 0)` is
     false. The `k = 0` pool's own no-reduction floor is reported for exactly this reason and
     bounds how much of the cell that enrichment can explain.

**`dense_probe` — which moduli does the dense-`N` arm fail on?** `sr3_r10_dense` reads 0.2907
on its 126 held-out moduli with median 0.0097 and max 1.000: a bimodal population, not a
uniformly mediocre one. `staged_reduce.py` only stores the full per-modulus dict when there
are `<= 16` moduli, so that cap is worked around here rather than edited there (the finished
runs must stay reproducible). Reports per-modulus accuracy for all 126, broken down by
modulus class, and scores **both** the dense-trained and the semiprime-trained model on one
shared pool — including the 36-semiprime subset `sr3_r10_many` was scored on, which is a
subset selection rather than a new pool.

Both entrypoints share `_build`, so there is exactly one implementation of the chain
recurrence `r <- (r*R + c_i) mod N` in this node.

Usage:
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/staged_reduce/reprobe.py::reprobe \\
    --tag sr3a --arms "sr3_r10,sr3_r100"
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/staged_reduce/reprobe.py::dense_probe \\
    --tag srb2r --arm sr3_r10_dense --ref-tag srb1 --ref-arm sr3_r10_many
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

GPU = os.environ.get("OLD_GPU", "L4")


class _Probe:
    """A loaded checkpoint plus the chain machinery, rebuilt exactly as training had it."""

    def __init__(self, tag, arm_name, seed, device, max_draws=134_217_728):
        import numpy as np
        import torch

        from one_layer_deeper.rule_acquisition.staged_reduce.staged_reduce import (
            ARMS, BASE_ARM, DIGIT_OFFSET, DIV_TOK, HASH_M, _make_model, stages_for,
        )
        from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

        ck_path = Path(DATA_DIR) / "staged_reduce" / str(tag) / "ckpt" / f"{arm_name}_seed{seed}.pt"
        if not ck_path.exists():
            raise FileNotFoundError(str(ck_path))
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        arm = {**BASE_ARM, **ARMS[arm_name]}
        w = arm["n_digits"]
        radix_eff, S = stages_for(arm["radix"], w)
        self.ck, self.cfg, self.arm, self.w = ck, cfg, arm, w
        self.radix_eff, self.S, self.device = radix_eff, S, device
        self.arm_name, self.tag = arm_name, tag

        # --- modulus set, rebuilt exactly as training did -----------------------------
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
        self.train_mods = [m[0] for m in fam_train]
        self.held_mods = [m[0] for m in fam_held]
        dense_all = np.arange(max(2, 10 ** (w - 1)), 10**w, dtype=np.int64)
        dense_held_mask = (
            ((dense_all * 2246822519) % 10 == 0) | np.isin(dense_all, self.held_mods)
        ) & ~np.isin(dense_all, self.train_mods)
        self.dense_train = dense_all[~dense_held_mask]
        self.dense_held = dense_all[dense_held_mask]
        self.mods_tr = torch.tensor(
            self.dense_train if arm["dense_n"] else np.array(self.train_mods),
            dtype=torch.long, device=device)

        # --- prompt machinery, identical to training ----------------------------------
        w_x, n_ans = 2 * w, w
        max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1
        thr = int(round(arm["train_frac"] * HASH_M))
        pow10 = torch.tensor([10 ** (w - 1 - k) for k in range(w)], device=device)

        def digits_of(v, width):
            return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

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
        self.model, self.is_train = model, is_train

        def true_seen_count(nv, y):
            """`k`: how many of the S stage queries on the **true** path were trained on."""
            k = torch.zeros_like(y)
            for i in range(S - 1, -1, -1):
                inp = ((y // radix_eff ** (i + 1)) % nv) * radix_eff + (y // radix_eff**i) % radix_eff
                k += is_train(nv, inp).long()
            return k

        @torch.no_grad()
        def chain_ok(nv, y, chunk=8192):
            """THE chain. One implementation, shared by every probe in this node."""
            ok = torch.zeros(nv.numel(), dtype=torch.bool, device=device)
            for i0 in range(0, nv.numel(), chunk):
                s = slice(i0, min(i0 + chunk, nv.numel()))
                n_, y_ = nv[s], y[s]
                r = torch.zeros_like(y_)
                for i in range(S - 1, -1, -1):
                    inp = r * radix_eff + (y_ // radix_eff**i) % radix_eff
                    h = model.roll(model.enc(prompts_for(n_, inp), read_pos))
                    r = (model.dec(h).argmax(-1) * pow10).sum(-1)
                ok[s] = r == (y_ % n_)
            return ok

        self.true_seen_count, self.chain_ok = true_seen_count, chain_ok
        self.gen = torch.Generator(device=device).manual_seed(4242)
        self.max_draws = max_draws

    def draw(self, total, want_k0):
        """Modulus-uniform `(N, y)` on the held-out composite half of the *training* moduli."""
        import torch

        got_n, got_y, drawn = [], [], 0
        per = 1 << 20
        while drawn < self.max_draws and sum(t.numel() for t in got_n) < total:
            nv = self.mods_tr[torch.randint(0, self.mods_tr.numel(), (per,),
                                            device=self.device, generator=self.gen)]
            u = torch.rand(per, device=self.device, generator=self.gen, dtype=torch.float64)
            y = torch.minimum((u * (nv * nv)).long(), nv * nv - 1)
            keep = ~self.is_train(nv, y)
            if want_k0:
                keep &= self.true_seen_count(nv, y) == 0
            got_n.append(nv[keep]); got_y.append(y[keep])
            drawn += per
        if not got_n:
            return None, None, drawn
        return torch.cat(got_n)[:total], torch.cat(got_y)[:total], drawn


def _certifiable_T(eps, m=768):
    import math
    return float("inf") if eps <= 0 else math.log(2.0) / (m * eps)


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=7200, memory=32768)
def reprobe(
    tag: str = "sr3a",
    arms: str = "sr3_r10",
    seed: int = 0,
    n_k0: int = 65536,
    n_strat: int = 262144,
    max_draws: int = 134_217_728,
):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"tag": tag, "arms": {}}

    for arm_name in [a.strip() for a in arms.split(",") if a.strip()]:
        try:
            P = _Probe(tag, arm_name, seed, device, max_draws)
        except FileNotFoundError as e:
            print(f"[missing] {e}", flush=True)
            continue
        S = P.S
        print(f"\n===== {arm_name} (step {P.ck.get('step')}) radix={P.arm['radix']} S={S} =====",
              flush=True)
        rec: dict = {"n_stages": S, "radix": P.arm["radix"], "step": P.ck.get("step")}

        # (1) stratified: chain accuracy as a function of k, on the ordinary held-out pool
        nv, y, _ = P.draw(n_strat, want_k0=False)
        k = P.true_seen_count(nv, y)
        ok = P.chain_ok(nv, y)
        rec["acc_by_k"] = {
            int(kk): dict(n=int((k == kk).sum()),
                          acc=float(ok[k == kk].float().mean()) if int((k == kk).sum()) else None)
            for kk in range(S + 1) if int((k == kk).sum()) > 0
        }
        rec["acc_all"] = float(ok.float().mean())
        rec["n_strat"] = int(nv.numel())
        print(f"  [{arm_name}] all-k acc={rec['acc_all']:.5f} on n={nv.numel()}", flush=True)
        for kk, v in rec["acc_by_k"].items():
            print(f"      k={kk:>2}  n={v['n']:>8}  acc={v['acc']:.5f}", flush=True)

        # (2) the k=0 cell
        if 2**S <= 4096:
            nv0, y0, drawn = P.draw(n_k0, want_k0=True)
            if nv0 is not None and nv0.numel() >= 1024:
                ok0 = P.chain_ok(nv0, y0)
                acc0 = float(ok0.float().mean())
                rec["k0"] = dict(n=int(nv0.numel()), acc=acc0, eps=1 - acc0,
                                 resolution=1.0 / int(nv0.numel()),
                                 certifiable_T=_certifiable_T(1 - acc0),
                                 floor=float((y0 < nv0).float().mean()), draws=drawn)
                print(f"  [{arm_name}] k=0 acc={acc0:.5f} eps={1-acc0:.3e} "
                      f"(n={nv0.numel()}, res {1/nv0.numel():.1e}, {drawn} draws)", flush=True)
            else:
                rec["k0"] = dict(n=int(nv0.numel()) if nv0 is not None else 0, note="pool too small")
        else:
            rec["k0"] = dict(note=f"P(k=0)=2^-{S} unreachable by rejection sampling")

        # (3) the independent-failure prediction from the arm's own stage held-out rate
        res_path = Path(DATA_DIR) / "staged_reduce" / str(tag) / f"results_seed{seed}.json"
        if res_path.exists():
            blob = json.loads(res_path.read_text())
            fin = (blob["arms"].get(arm_name) or {}).get("final") or {}
            p = (fin.get("heldout_y") or {}).get("acc")
            if p is not None:
                rec["stage_heldout_acc"] = p
                rec["stage_heldout_acc_pow_S"] = p**S
                print(f"  [{arm_name}] stage held-out p={p:.5f} -> p^S={p**S:.5f}", flush=True)

        out["arms"][arm_name] = rec

    op = Path(DATA_DIR) / "staged_reduce" / str(tag) / f"reprobe_seed{seed}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] /staged_reduce/{tag}/reprobe_seed{seed}.json", flush=True)
    return out


def _factor(n: int):
    f, d, m = [], 2, n
    while d * d <= m:
        while m % d == 0:
            f.append(d); m //= d
        d += 1
    if m > 1:
        f.append(m)
    return f


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=7200, memory=32768)
def dense_probe(
    tag: str = "srb2r",
    arm: str = "sr3_r10_dense",
    ref_tag: str = "srb1",
    ref_arm: str = "sr3_r10_many",
    seed: int = 0,
    n_per: int = 4096,
):
    """Per-modulus chain accuracy for every held-out modulus, by modulus class.

    Both models are scored on **one shared pool** built from the dense arm's held-out modulus
    set, so the dense-vs-semiprime comparison is a subset selection rather than a new
    sampling. Every modulus in that set is unseen by *both* models: the dense held-out mask
    excludes the semiprime train moduli by construction, and the semiprime-trained arm only
    ever saw those 142.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = _Probe(tag, arm, seed, device)
    R = None
    try:
        R = _Probe(ref_tag, ref_arm, seed, device)
    except FileNotFoundError as e:
        print(f"[no reference model] {e}", flush=True)

    mods = [int(m) for m in P.dense_held]
    semi36 = set(int(m) for m in P.held_mods)  # the 36 `sr3_r10_many` was scored on
    print(f"[dense_probe] {arm}@{tag} S={P.S} | {len(mods)} held-out moduli, "
          f"{len(semi36)} of them the semiprime held-out set | ref={ref_arm if R else None}",
          flush=True)
    if R is not None:
        overlap = set(mods) & set(int(m) for m in R.train_mods)
        print(f"[check] moduli in the reference model's TRAIN set: {len(overlap)} "
              f"(must be 0){' <-- LEAK' if overlap else ''}", flush=True)

    # one shared pool: n_per uniform y in [0, N^2) per held-out modulus
    g = torch.Generator(device=device).manual_seed(99991)
    nv = torch.tensor(mods, dtype=torch.long, device=device).repeat_interleave(n_per)
    u = torch.rand(nv.numel(), device=device, generator=g, dtype=torch.float64)
    yv = torch.minimum((u * (nv * nv)).long(), nv * nv - 1)

    ok_d = P.chain_ok(nv, yv)
    ok_r = R.chain_ok(nv, yv) if R is not None else None
    free = (yv < nv).float()

    rows = []
    for i, m in enumerate(mods):
        s = slice(i * n_per, (i + 1) * n_per)
        fac = _factor(m)
        rows.append(dict(
            N=m, acc_dense=float(ok_d[s].float().mean()),
            acc_ref=float(ok_r[s].float().mean()) if ok_r is not None else None,
            floor=float(free[s].mean()),
            even=(m % 2 == 0), div5=(m % 5 == 0), min_factor=min(fac),
            n_prime_factors=len(fac), n_distinct=len(set(fac)),
            kind=("prime" if len(fac) == 1 else
                  "prime_power" if len(set(fac)) == 1 else
                  "semiprime" if len(fac) == 2 else "composite"),
            in_semi36=(m in semi36),
        ))

    def agg(pred, label):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        d = sum(r["acc_dense"] for r in sel) / len(sel)
        rr = ([r["acc_ref"] for r in sel] if ok_r is not None else None)
        r_ = (sum(rr) / len(rr)) if rr else float("nan")
        fl = sum(r["floor"] for r in sel) / len(sel)
        med = sorted(r["acc_dense"] for r in sel)[len(sel) // 2]
        hi = sum(1 for r in sel if r["acc_dense"] > 0.5)
        print(f"  {label:<34} n={len(sel):>4}  dense={d:.4f} (med {med:.4f}, "
              f">0.5: {hi}/{len(sel)})  ref={r_:.4f}  floor={fl:.4f}", flush=True)
        return dict(label=label, n=len(sel), acc_dense=d, acc_dense_median=med,
                    n_above_half=hi, acc_ref=r_, floor=fl)

    print(f"\n=== breakdown over {len(mods)} held-out moduli "
          f"({n_per} problems each, resolution {1/n_per:.1e} per modulus) ===", flush=True)
    summary = [
        agg(lambda r: True, "ALL held-out moduli"),
        agg(lambda r: r["in_semi36"], "the 36 semiprimes (apples-to-apples)"),
        agg(lambda r: not r["in_semi36"], "the rest"),
        agg(lambda r: r["even"], "even N"),
        agg(lambda r: not r["even"], "odd N"),
        agg(lambda r: r["div5"], "N divisible by 5"),
        agg(lambda r: r["min_factor"] == 2, "min prime factor = 2"),
        agg(lambda r: r["min_factor"] == 3, "min prime factor = 3"),
        agg(lambda r: r["min_factor"] == 5, "min prime factor = 5"),
        agg(lambda r: r["min_factor"] == 7, "min prime factor = 7"),
        agg(lambda r: r["min_factor"] >= 11, "min prime factor >= 11"),
        agg(lambda r: r["kind"] == "prime", "prime N"),
        agg(lambda r: r["kind"] == "prime_power", "prime power N"),
        agg(lambda r: r["kind"] == "semiprime", "semiprime N (2 factors)"),
        agg(lambda r: r["kind"] == "composite", "composite N (3+ factors)"),
        agg(lambda r: r["N"] < 400, "N < 400"),
        agg(lambda r: 400 <= r["N"] < 700, "400 <= N < 700"),
        agg(lambda r: r["N"] >= 700, "N >= 700"),
    ]

    print("\n=== per-modulus, sorted by dense accuracy ===", flush=True)
    for r in sorted(rows, key=lambda r: r["acc_dense"]):
        ref_s = "n/a" if r["acc_ref"] is None else f"{r['acc_ref']:.4f}"
        parity = "even" if r["even"] else "odd"
        print(f"  N={r['N']:>4} {parity:<5} minf={r['min_factor']:>3} "
              f"{r['kind']:<12} semi36={'Y' if r['in_semi36'] else '.'}  "
              f"dense={r['acc_dense']:.4f}  ref={ref_s}", flush=True)

    out = dict(tag=tag, arm=arm, ref_tag=ref_tag, ref_arm=ref_arm, n_per=n_per,
               n_moduli=len(mods), per_modulus=rows,
               summary=[s for s in summary if s])
    op = Path(DATA_DIR) / "staged_reduce" / str(tag) / f"dense_probe_seed{seed}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] /staged_reduce/{tag}/dense_probe_seed{seed}.json", flush=True)
    return out
