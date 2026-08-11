"""Checkpoint probes the training-time instrument could not supply. Trains nothing.

**The seen-stage problem.** A staged arm's chain readout is on held-out *composite* `(N, y)`,
but held-out-ness is defined at the level of the composite problem, not per stage query. On
the way down the chain, roughly half of the `S` stage queries land in the arm's own training
half (measured: `stage_seen_frac ~ 0.49`). That is genuinely ambiguous rather than simply a
confound — if the claim is "decompose until every atom's input space is coverable", stage
queries landing inside the covered set *is the mechanism*, not a cheat. But it means
"staged beats monolithic on held-out `y`" is not clean as stated.

This probe resolves it by stratifying on `k`, the number of stage queries on the **true**
stage path that fall in the training half. `k` is a property of `(N, y)` and the hash alone —
model-independent, and identical for any two arms sharing a radix and a modulus set. Two
readouts:

  1. `acc_by_k` — chain accuracy as a function of `k`, over the ordinary held-out pool. If it
     is flat in `k`, the seen/unseen distinction does not carry the result and the ambiguity
     dissolves.
  2. `k0` — a pool built by **rejection sampling to `k = 0`**, i.e. composite problems none of
     whose stage queries was ever trained on. For the monolithic arm `S = 1` and this is
     exactly its ordinary held-out pool, so the comparison is apples to apples. `P(k=0) =
     2^-S`, so this is reachable for `S <= 8` and not for `S = 20`; the probe reports the
     achieved pool size and its resolution and skips the cell when it is too small to read.

Also reported: the arm's stage-level held-out accuracy `p` and the independent-failure
prediction `p^S`, which is what the `k = 0` cell should equal if the stages fail
independently and nothing about the chain is special.

Usage:
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/staged_reduce/reprobe.py::reprobe \\
    --tag sr3a --arms "sr3_r10,sr3_r100"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

GPU = os.environ.get("OLD_GPU", "L4")


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=7200, memory=32768)
def reprobe(
    tag: str = "sr3a",
    arms: str = "sr3_r10",
    seed: int = 0,
    n_k0: int = 65536,
    n_strat: int = 262144,
    max_draws: int = 134_217_728,
):
    import numpy as np
    import torch

    # Imported inside the function, as `exact_atom/reprobe.py` does: a module-level import
    # would also register the training entrypoint on the same Modal app.
    from one_layer_deeper.rule_acquisition.staged_reduce.staged_reduce import (
        ARMS, BASE_ARM, DIGIT_OFFSET, DIV_TOK, HASH_M, _make_model, certifiable_T, stages_for,
    )
    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_dir = Path(DATA_DIR) / "staged_reduce" / str(tag) / "ckpt"
    out: dict = {"tag": tag, "arms": {}}

    for arm_name in [a.strip() for a in arms.split(",") if a.strip()]:
        ck_path = ck_dir / f"{arm_name}_seed{seed}.pt"
        if not ck_path.exists():
            print(f"[missing] {ck_path}", flush=True)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        arm = {**BASE_ARM, **ARMS[arm_name]}
        w = arm["n_digits"]
        radix_eff, S = stages_for(arm["radix"], w)
        print(f"\n===== {arm_name} (step {ck.get('step')}) radix={arm['radix']} S={S} =====",
              flush=True)

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
        train_mods = [m[0] for m in fam_train]
        held_mods = [m[0] for m in fam_held]
        dense_all = np.arange(max(2, 10 ** (w - 1)), 10**w, dtype=np.int64)
        dense_held_mask = (
            ((dense_all * 2246822519) % 10 == 0) | np.isin(dense_all, held_mods)
        ) & ~np.isin(dense_all, train_mods)
        mods_tr = torch.tensor(
            dense_all[~dense_held_mask] if arm["dense_n"] else np.array(train_mods),
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

        def true_seen_count(nv, y):
            """`k`: how many of the S stage queries on the **true** path were trained on.

            Depends only on `(N, y)` and the hash, so it partitions the pool identically for
            any model and is not contaminated by the model's own errors.
            """
            k = torch.zeros_like(y)
            for i in range(S - 1, -1, -1):
                inp = ((y // radix_eff ** (i + 1)) % nv) * radix_eff + (y // radix_eff**i) % radix_eff
                k += is_train(nv, inp).long()
            return k

        @torch.no_grad()
        def chain_ok(nv, y, chunk=8192):
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

        gen = torch.Generator(device=device).manual_seed(4242)

        def draw(total, want_k0):
            """Uniform modulus-uniform `(N, y)` on the held-out composite half.

            `want_k0` additionally rejects anything whose true stage path touches a trained
            query. The hash is arbitrary with respect to the arithmetic, so conditioning on
            `k = 0` selects an unbiased subpopulation of dividends.
            """
            got_n, got_y, drawn = [], [], 0
            per = 1 << 20
            while drawn < max_draws and sum(t.numel() for t in got_n) < total:
                nv = mods_tr[torch.randint(0, mods_tr.numel(), (per,), device=device, generator=gen)]
                u = torch.rand(per, device=device, generator=gen, dtype=torch.float64)
                y = torch.minimum((u * (nv * nv)).long(), nv * nv - 1)
                keep = ~is_train(nv, y)
                if want_k0:
                    keep &= true_seen_count(nv, y) == 0
                got_n.append(nv[keep]); got_y.append(y[keep])
                drawn += per
            if not got_n:
                return None, None, drawn
            return torch.cat(got_n)[:total], torch.cat(got_y)[:total], drawn

        rec: dict = {"n_stages": S, "radix": arm["radix"], "step": ck.get("step")}

        # (1) stratified: chain accuracy as a function of k, on the ordinary held-out pool
        nv, y, _ = draw(n_strat, want_k0=False)
        k = true_seen_count(nv, y)
        ok = chain_ok(nv, y)
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

        # (2) the k=0 cell: composite problems none of whose stage queries was trained on
        if 2**S <= 4096:  # otherwise rejection sampling cannot reach a readable pool
            nv0, y0, drawn = draw(n_k0, want_k0=True)
            if nv0 is not None and nv0.numel() >= 1024:
                ok0 = chain_ok(nv0, y0)
                acc0 = float(ok0.float().mean())
                rec["k0"] = dict(n=int(nv0.numel()), acc=acc0, eps=1 - acc0,
                                 resolution=1.0 / int(nv0.numel()),
                                 certifiable_T=certifiable_T(1 - acc0),
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
