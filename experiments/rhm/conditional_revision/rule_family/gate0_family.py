"""Gate 0: does the phenomenon exist at all -- does the model infer the rule set
in-context, and does its within-context loss decline track the oracle's rule-posterior
concentration?

Two arms, identical in every respect except the DGP:

  family  the R rule sets of the chosen design; each context window is K whole
          sequences drawn from ONE of them, so the model must infer which.
  floor   `differ_levels=[]` -- the same code path with all R rule sets bit-identical,
          i.e. the existing single-rule-set substrate. Same steps, same batch size, same
          tokens, same window construction, same seed. This is the load-bearing control:
          in the floor arm nothing can be learned in-context, so ANY decline of loss with
          context depth there is a positional artifact and must be subtracted.

THE PRIMARY INSTRUMENT IS A MATCHED DEPTH SWAP, not a raw loss-vs-position curve.
Gate -1 says the whole rule-revision budget is `ln R` nats spread over the window, so the
available signal is a few hundredths of a nat against a per-token NLL sd of ~1.2. An
unmatched curve cannot resolve that without enormous eval sets. Instead, ONE probe
sequence P is read at EVERY context depth:

    depth k :   [ f_1 ... f_k , P , f_{k+1} ... f_{K-1} ]

so P's tokens, its parse, and its rule set are identical across conditions and the only
thing that changes is how many same-rule sequences precede it. This is the `arity_torque`
idiom -- identical data, only the conditioning differs -- and it makes the contrast a
within-subject one. The fillers are nested (depth k+1's context contains depth k's), so
the oracle's information is monotone in k by construction.

Three readouts, in decreasing order of how much they gate the rest of the programme:

  1. MATCHED ICL: loss on P at depth 0 vs depth k, family vs floor, against the exact
     oracle's own matched contrast. Reported as a FRACTION of the available gap.
     P's FIRST token is scored separately: it is predicted purely from the preceding
     fillers, so it is the position where rule information is the only information.
  2. RULE DECODABILITY from activations by context depth, against the exact Bayes
     accuracy E[max_r w_r]. Uses a position grid and a large eval set, because an R-way
     probe needs many windows per class when R is large.
  3. PER-LEVEL ancestor recovery in both arms, against this substrate's reference lines
     (d1 0.979 / d3 0.836 / root 0.088). Whether the family regime costs belief depth is
     load-bearing for Gate 1: the incumbent's measurement was bounded by belief depth,
     not by the conditioning gap.

Pre-registered kills (decide before the numbers land):

  * matched ICL fraction < 0.10 at every depth AND rule decodability at chance
      -> no in-context inference. STOP. RHM_META_LEARNING's thin-signal failure
         reappearing on the ICL axis, which is worth knowing precisely.
  * matched ICL fraction > 0.10 but rule identity NOT decodable
      -> the model exploits the family without representing r. Gate 1 has no probe to
         build on; report and rethink the readout rather than the regime.
  * the floor arm shows the same matched decline as the family arm
      -> the decline is positional, not in-context. The instrument is wrong, not the DGP.

    modal run --detach -m rhm.conditional_revision.rule_family.gate0_family::gate0 \
        --design d2_R128_nF4 --k-seqs 8 --tag g0
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-rule-family", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=65536)
def gate0(
    design: str = "d2_R128_nF4",
    # DGP -- conditional_revision's regime so the reference lines transfer
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 8, phase: str = "aligned",
    # Model
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    deep_block: str = "post_block6",
    # Training
    base_steps: int = 12000, batch_size: int = 32, lr: float = 3e-4,
    weight_decay: float = 0.01,
    pool_tokens: int = 40_000_000, data_seed: int = 7,
    # Measurement
    n_pairs: int = 3072, n_oracle_pairs: int = 64,
    n_rule_windows: int = 4096, n_grid: int = 64,
    eval_seed: int = 999,
    probe_steps: int = 800, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    rule_probe_steps: int = 3000, rule_probe_lr: float = 3e-3,
    log_interval: int = 1000,
    arms: str = "family,floor",
    icl_trace_every: int = 0, n_trace_pairs: int = 512,
    ckpt_every: int = 0,
    heldout: bool = True, force_retrain: bool = False,
    save_ckpt: bool = True,
    seed: int = 42, tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_latent_loop import _probe_acc
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS, _top_level
    from rhm.conditional_revision.rule_family.oracle_mixture import (
        family_predictive, mixture_profiles,
    )

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = tb_key(v, s, L, m)
    R, dl, nF, mode = DESIGNS[design]

    print(f"{'=' * 84}\nRULE FAMILY -- GATE 0   design={design}   {key}   "
          f"{n_layer}L/{n_head}H/{n_embd}D")
    print(f"  R={R}  differ_levels={dl}  n_differ_features={nF}  mode={mode}")
    print(f"  window = {k_seqs} sequences x {T} tokens = {G} tokens   "
          f"batch={batch_size}  steps={base_steps}")
    print(f"{'=' * 84}", flush=True)

    families = {}
    families["family"], fam_meta = make_family(v, s, L, m, R=R, differ_levels=dl,
                                               n_differ_features=nF, seed=family_seed,
                                               mode=mode)
    families["floor"], _ = make_family(v, s, L, m, R=R, differ_levels=[],
                                       n_differ_features=nF, seed=family_seed,
                                       mode=mode)
    arm_names = [a for a in arms.split(",") if a]

    # ---------------- training data ----------------
    per_rule = max(1, pool_tokens // (R * T))
    print(f"  pool: {per_rule:,} sequences per rule set x {R} = "
          f"{per_rule * R * T / 1e6:.1f}M tokens", flush=True)
    pools = {}
    for arm in arm_names:
        pool = np.empty((R, per_rule, T), dtype=np.int64)
        for r in range(R):
            one, _, _ = generate_windows([families[arm][r]], per_rule, 1,
                                         seed=data_seed + 1000 * r + 1)
            pool[r] = one[:, 0]
        pools[arm] = torch.from_numpy(pool)

    arange_G = torch.arange(G)

    def get_batch(arm, gen):
        pool = pools[arm]
        rid = torch.randint(0, R, (batch_size,), generator=gen)
        if phase == "aligned":
            idx = torch.randint(0, per_rule, (batch_size, k_seqs), generator=gen)
            win = pool[rid[:, None], idx]                   # (B, K, T)
            return win.reshape(batch_size, G).to(device), rid.to(device)
        if phase != "hidden":
            raise ValueError(f"phase must be 'aligned' or 'hidden' (got {phase!r})")
        idx = torch.randint(0, per_rule, (batch_size, k_seqs + 1), generator=gen)
        stream = pool[rid[:, None], idx].reshape(batch_size, (k_seqs + 1) * T)
        off = torch.randint(0, T, (batch_size, 1), generator=gen)
        return (stream.gather(1, off + arange_G[None, :]).to(device),
                rid.to(device))

    # ---------------- matched depth-swap eval set ----------------
    # For each pair-window: one probe sequence P and K-1 fillers, all from one rule set.
    # Depth k places P after the first k fillers. Fillers are NESTED across k.
    def build_pairs(arm, n, seed_):
        sq, rid, lf = generate_windows(families[arm], n, k_seqs, seed=seed_)
        P, fill = sq[:, 0], sq[:, 1:]                       # (n,T), (n,K-1,T)
        wins = np.empty((k_seqs, n, k_seqs, T), dtype=np.int64)
        for k in range(k_seqs):
            wins[k] = np.concatenate([fill[:, :k], P[:, None], fill[:, k:]], axis=1)
        return wins, rid, lf, P, fill

    pair_pack = {arm: build_pairs(arm, n_pairs, eval_seed) for arm in arm_names}

    # ---------------- oracle on the matched construction ----------------
    print("\n--- oracle on the matched depth-swap set (exact BP mixture) ---", flush=True)
    n_or = min(n_oracle_pairs, n_pairs)
    o_wins, o_rid = pair_pack["family"][0][:, :n_or], pair_pack["family"][1][:n_or]
    # every sequence appears in every depth condition, so BP runs once on the union
    uniq = np.concatenate([pair_pack["family"][3][:n_or][:, None],
                           pair_pack["family"][4][:n_or]], axis=1)      # (n_or, K, T)
    upost = family_predictive(families["family"], uniq.reshape(-1, T), chunk=2048)
    upost = upost.reshape(R, n_or, k_seqs, T, v)                        # slot 0 = P
    # slot index of each window position, per depth: depth k -> [1..k, 0, k+1..K-1]
    slot = np.empty((k_seqs, k_seqs), dtype=np.int64)
    for k in range(k_seqs):
        slot[k] = np.concatenate([np.arange(1, k + 1), [0], np.arange(k + 1, k_seqs)])

    oracle_depth_mix, oracle_depth_true, oracle_first_mix, oracle_bayes_acc = [], [], [], []
    for k in range(k_seqs):
        pk = upost[:, :, slot[k]]                                       # (R,n_or,K,T,v)
        prof = mixture_profiles(pk, o_wins[k], o_rid)
        nm = prof["nll_mix"].reshape(n_or, k_seqs, T)
        nt = prof["nll_true"].reshape(n_or, k_seqs, T)
        ba = prof["w_pre"].max(axis=2).reshape(n_or, k_seqs, T)
        oracle_depth_mix.append(float(nm[:, k, 1:].mean()))    # P's matched positions
        oracle_depth_true.append(float(nt[:, k, 1:].mean()))
        oracle_first_mix.append(float(nm[:, k, 0].mean()))     # P's first token
        oracle_bayes_acc.append(float(ba[:, k, 0].mean()))
    del upost
    oracle = {"depth_nll_mix": oracle_depth_mix, "depth_nll_true": oracle_depth_true,
              "depth_first_token_nll_mix": oracle_first_mix,
              "depth_bayes_rule_acc": oracle_bayes_acc,
              "n_oracle_pairs": n_or}
    print(f"  oracle nll_mix on P (matched positions), by depth: "
          + " ".join(f"{x:.4f}" for x in oracle_depth_mix))
    print(f"  oracle nll_true (known rules)                    : "
          + " ".join(f"{x:.4f}" for x in oracle_depth_true))
    print(f"  oracle nll_mix on P's FIRST token                : "
          + " ".join(f"{x:.4f}" for x in oracle_first_mix))
    print(f"  oracle Bayes rule accuracy at P's start           : "
          + " ".join(f"{x:.4f}" for x in oracle_bayes_acc), flush=True)

    top = _top_level(T, L, s)
    grid = np.linspace(0, G - 2, min(n_grid, G - 1)).astype(int)
    results = {"config": {"design": design, "R": R, "differ_levels": dl,
                          "n_differ_features": nF, "mode": mode,
                          "v": v, "s": s, "L": L, "m": m, "k_seqs": k_seqs, "G": G,
                          "phase": phase,
                          "n_layer": n_layer, "n_head": n_head,
                          "n_embd": n_embd,
                          "base_steps": base_steps, "batch_size": batch_size,
                          "lr": lr, "pool_tokens": pool_tokens,
                          "per_rule_sequences": per_rule, "n_pairs": n_pairs,
                          "n_rule_windows": n_rule_windows,
                          "seed": seed, "tag": tag},
               "family_meta": fam_meta, "oracle": oracle, "arms": {}}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    trace_by_arm = {}

    def depth_losses(model, wins_np, npairs):
        """Mean loss on the probe sequence P at each context depth."""
        out = np.zeros(k_seqs)
        with torch.no_grad():
            for k in range(k_seqs):
                W = torch.from_numpy(
                    wins_np[k][:npairs].reshape(-1, G)).to(device)
                tot, nb = 0.0, 0
                for i in range(0, W.shape[0], 64):
                    xb = W[i:i + 64]
                    xin, ytg = xb[:, :-1].contiguous(), xb[:, 1:].contiguous()
                    logits, _ = model(xin, ytg)
                    nll = F.cross_entropy(
                        logits.reshape(-1, v), ytg.reshape(-1),
                        reduction='none').reshape(xb.shape[0], G - 1)
                    tot += float(nll[:, k * T:k * T + T - 1].mean(1).sum())
                    nb += xb.shape[0]
                out[k] = tot / nb
        return out

    # held-out rule sets: SAME tuple pool, DISJOINT partitions. make_family draws
    # partitions sequentially with dedup from one rng, so requesting 2R and taking
    # the tail gives rule sets the model never trained on while holding every other
    # aspect of the construction fixed -- the Student-2 question (did it learn the
    # family's structure or these particular tables) with nothing else varying.
    ho_wins = None
    if heldout and dl:
        # a design can EXHAUST its own rule space: with nF differing features and m
        # rules there are only (nF*m)!/(m!)^nF distinguishable partitions per level,
        # which is 70 for nF=2/m=4. d2_R64_nF2 therefore uses 64 of 70 and leaves
        # only 6 unseen -- so held-out transfer is measurable but on a much smaller
        # family, and that limit is itself a fact about the design.
        from math import factorial
        n_part = (factorial(nF * m) // factorial(m) ** nF) ** len(dl)
        R_tot = min(2 * R, n_part)
        print(f"  held-out: {n_part} distinguishable rule sets exist, {R} used in "
              f"training, {R_tot - R} available held-out", flush=True)
        big, _ = make_family(v, s, L, m, R=R_tot, differ_levels=dl,
                             n_differ_features=nF, seed=family_seed, mode=mode)
        for r in range(R):
            assert all(np.array_equal(a, b)
                       for a, b in zip(big[r], families['family'][r])), \
                'held-out construction perturbed the training family'
        ho_fam = big[R:]
        if not ho_fam:
            ho_fam = None
        ho_sq, _, _ = generate_windows(ho_fam, n_trace_pairs * 2, k_seqs,
                                       seed=eval_seed + 31)
        hoP, hofill = ho_sq[:, 0], ho_sq[:, 1:]
        ho_wins = np.empty((k_seqs, ho_sq.shape[0], k_seqs, T), dtype=np.int64)
        for k in range(k_seqs):
            ho_wins[k] = np.concatenate(
                [hofill[:, :k], hoP[:, None], hofill[:, k:]], axis=1)

    for arm in arm_names:
        print(f"\n{'=' * 84}\nARM: {arm}\n{'=' * 84}", flush=True)
        torch.manual_seed(seed)
        model = GPT(v, G, n_layer, n_head, n_embd).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed)

        ph = "" if phase == "aligned" else f"_{phase}"
        ckpt = (f"{DATA_DIR}/{key}/rule_family/{design}_K{k_seqs}{ph}_{arm}_"
                f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
        if os.path.exists(ckpt) and not force_retrain:
            print(f"  loading cached {ckpt}", flush=True)
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        else:
            trace = []
            for step in range(base_steps):
                model.train()
                x, _ = get_batch(arm, gen)
                _, loss = model(x[:, :-1].contiguous(), x[:, 1:].contiguous())
                opt.zero_grad(); loss.backward(); opt.step()
                if step % log_interval == 0 or step == base_steps - 1:
                    print(f"  {arm} {step:6d}  ntp {loss.item():.4f}", flush=True)
                if icl_trace_every and (step % icl_trace_every == 0
                                        or step == base_steps - 1):
                    model.eval()
                    dl_ = depth_losses(model, pair_pack[arm][0], n_trace_pairs)
                    trace.append({'step': step,
                                  'ntp': float(loss.item()),
                                  'depth_loss': dl_.tolist(),
                                  'decline': float(dl_[0] - dl_[-1])})
                    print(f"    trace step {step:6d}  depth0 {dl_[0]:.4f}  "
                          f"depth{k_seqs - 1} {dl_[-1]:.4f}  "
                          f"decline {dl_[0] - dl_[-1]:+.5f}", flush=True)
                    model.train()
                if ckpt_every and (step + 1) % ckpt_every == 0:
                    ip = ckpt.replace('.pt', f'_at{step + 1}.pt')
                    os.makedirs(os.path.dirname(ip), exist_ok=True)
                    torch.save({'model': model.state_dict(),
                                'config': results['config'],
                                'step': step + 1}, ip)
                    volume.commit()
                    print(f'    checkpoint -> {ip}', flush=True)
            trace_by_arm[arm] = trace
            if save_ckpt:
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                torch.save({"model": model.state_dict(), "config": results["config"]},
                           ckpt)
                volume.commit()
                print(f"  saved -> {ckpt}", flush=True)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # ---- (1) matched depth swap ----
        wins = pair_pack[arm][0]
        depth_loss, depth_first, depth_loss_by_level = [], [], []
        per_pair = np.zeros((k_seqs, n_pairs))
        with torch.no_grad():
            for k in range(k_seqs):
                W = torch.from_numpy(wins[k].reshape(-1, G)).to(device)
                tot = torch.zeros(T - 1, device=device)
                first = 0.0
                nb = 0
                for i in range(0, W.shape[0], 64):
                    xb = W[i:i + 64]
                    xin, ytg = xb[:, :-1].contiguous(), xb[:, 1:].contiguous()
                    logits, _ = model(xin, ytg)
                    nll = F.cross_entropy(logits.reshape(-1, v), ytg.reshape(-1),
                                          reduction="none").reshape(xb.shape[0], G - 1)
                    # P at depth k occupies global positions kT..kT+T-1; predicting P's
                    # token j sits at loss index kT+j-1
                    seg = nll[:, k * T:k * T + T - 1]
                    tot += seg.sum(0)
                    per_pair[k, i:i + xb.shape[0]] = seg.mean(1).cpu().numpy()
                    if k > 0:
                        first += float(nll[:, k * T - 1].sum())
                    nb += xb.shape[0]
                per_pos = (tot / nb).cpu().numpy()            # j = 1..T-1
                depth_loss.append(float(per_pos.mean()))
                depth_first.append(float(first / nb) if k > 0 else float("nan"))
                depth_loss_by_level.append(
                    [float(per_pos[top[1:] == l].mean()) if (top[1:] == l).any()
                     else float("nan") for l in range(L + 1)])
                del W

        # ---- (2) rule decodability ----
        rseqs, rrid, _ = generate_windows(families[arm], n_rule_windows, k_seqs,
                                          seed=eval_seed + 7)
        rx = torch.from_numpy(rseqs.reshape(n_rule_windows, G))
        gidx = torch.from_numpy(grid).to(device)
        A = torch.empty(n_rule_windows, len(grid), n_embd)
        with torch.no_grad():
            for i in range(0, n_rule_windows, 64):
                xb = rx[i:i + 64].to(device)
                _, _, inter = model(xb[:, :-1].contiguous(), return_intermediates=True)
                A[i:i + 64] = inter[deep_block][:, gidx, :].float().cpu()
        rid_t = torch.from_numpy(rrid).long()
        n_tr = int(0.7 * n_rule_windows)
        rule_acc_by_grid, rule_acc_by_seq = [], []
        if R > 1:
            Xtr, Xte = A[:n_tr].to(device), A[n_tr:].to(device)
            ytr, yte = rid_t[:n_tr].to(device), rid_t[n_tr:].to(device)
            mu = Xtr.reshape(-1, n_embd).mean(0, keepdim=True)
            sd = Xtr.reshape(-1, n_embd).std(0, keepdim=True) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            clf = nn.Linear(n_embd, R).to(device)
            o = torch.optim.Adam(clf.parameters(), lr=rule_probe_lr, weight_decay=1e-4)
            gsub = torch.Generator().manual_seed(seed + 3)
            for _ in range(rule_probe_steps):
                sel = torch.randint(0, Xtr.shape[0], (256,), generator=gsub).to(device)
                pos = torch.randint(0, len(grid), (256,), generator=gsub).to(device)
                l = F.cross_entropy(clf(Xtr[sel, pos]), ytr[sel])
                o.zero_grad(); l.backward(); o.step()
            with torch.no_grad():
                hit = (clf(Xte).argmax(-1) == yte[:, None]).float().mean(0).cpu().numpy()
            rule_acc_by_grid = hit.tolist()
            gseq = grid // T
            rule_acc_by_seq = [float(hit[gseq == k].mean()) if (gseq == k).any()
                               else float("nan") for k in range(k_seqs)]
            del Xtr, Xte
        del A

        # ---- (3) per-level ancestor recovery, last position of the window ----
        lseqs, _, llf = generate_windows(families[arm], 2048, k_seqs, seed=eval_seed + 11)
        lx = torch.from_numpy(lseqs.reshape(2048, G)).to(device)
        probe_seq = max(0, k_seqs - 2)
        probe_pos = probe_seq * T + T - 1
        last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
        acts_last = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, lx.shape[0], 64):
                _, _, inter = model(lx[i:i + 64, :-1].contiguous(),
                                    return_intermediates=True)
                for b in block_names:
                    acts_last[b].append(inter[b][:, probe_pos, :].float())
        acts_last = {b: torch.cat(vs) for b, vs in acts_last.items()}
        levels = {}
        for ell in range(L):
            y = torch.from_numpy(
                llf[ell][:, probe_seq, last_anc[ell]].astype(np.int64)).to(device)
            levels[f"d{L - ell}"] = max(
                max(_probe_acc(acts_last[b], y, v, device, probe_steps, probe_lr),
                    _probe_acc(acts_last[b], y, v, device, mlp_steps, probe_lr,
                               hidden=mlp_hidden))
                for b in block_names)

        results["arms"][arm] = {
            "depth_loss": depth_loss,
            "depth_paired_diff": [float((per_pair[0] - per_pair[k]).mean())
                                  for k in range(k_seqs)],
            "depth_paired_sem": [float((per_pair[0] - per_pair[k]).std(ddof=1)
                                       / np.sqrt(n_pairs)) for k in range(k_seqs)],
            "probe_position": int(probe_pos),
            "icl_trace": trace_by_arm.get(arm),
            "heldout_depth_loss": (depth_losses(model, ho_wins,
                                                ho_wins.shape[1]).tolist()
                                   if ho_wins is not None else None),
            "depth_first_token_loss": depth_first,
            "depth_loss_by_level": depth_loss_by_level,
            "rule_acc_by_seq": rule_acc_by_seq,
            "rule_acc_by_grid": rule_acc_by_grid,
            "grid": grid.tolist(),
            "levels": levels,
        }
        print(f"\n  {arm}: matched loss on P by depth: "
              + " ".join(f"{x:.4f}" for x in depth_loss))
        print(f"  {arm}: P's first-token loss by depth: "
              + " ".join(f"{x:.4f}" for x in depth_first))
        if rule_acc_by_seq:
            print(f"  {arm}: rule probe acc by seq: "
                  + " ".join(f"{x:.4f}" for x in rule_acc_by_seq)
                  + f"   (chance {1 / R:.4f})")
        ho = results["arms"][arm]["heldout_depth_loss"]
        if ho:
            print(f"  {arm}: HELD-OUT rule sets, loss on P by depth: "
                  + " ".join(f"{x:.4f}" for x in ho)
                  + f"   decline {ho[0] - ho[-1]:+.5f}")
        print(f"  {arm}: per-level recovery {levels}")
        print(f"  reference (single rule set): d1 0.979  d3 0.836  root 0.088",
              flush=True)
        del model, acts_last
        torch.cuda.empty_cache()

    # ---------------- verdict ----------------
    if "family" in results["arms"]:
        omix = oracle["depth_nll_mix"]
        avail = [omix[0] - omix[k] for k in range(k_seqs)]
        got = results["arms"]["family"]["depth_paired_diff"]
        sem = results["arms"]["family"]["depth_paired_sem"]
        flo_got = results["arms"].get("floor", {}).get("depth_paired_diff")
        flo_sem = results["arms"].get("floor", {}).get("depth_paired_sem")
        net = ([g - f for g, f in zip(got, flo_got)] if flo_got else got)
        icl_frac = [n / a if abs(a) > 1e-9 else float("nan")
                    for n, a in zip(net, avail)]
        cc = float(np.corrcoef(avail[1:], net[1:])[0, 1]) if k_seqs > 2 else float("nan")
        results["icl"] = {"available_by_depth": avail, "family_by_depth": got,
                          "family_sem": sem, "floor_by_depth": flo_got,
                          "floor_sem": flo_sem, "net_by_depth": net,
                          "icl_fraction_by_depth": icl_frac,
                          "corr_net_vs_available": cc}
        print(f"\n{'=' * 84}\nGATE 0 VERDICT (PROVISIONAL -- interpretation is Jasper's "
              f"call)\n{'=' * 84}")
        print(f"  matched depth swap: loss on the SAME probe sequence, "
              f"depth 0 minus depth k")
        print(f"  {'depth':<7}{'oracle avail':>14}{'family+-sem':>19}"
              f"{'floor+-sem':>19}{'net':>9}{'ICLfrac':>9}{'ruleacc':>9}"
              f"{'Bayes':>8}")
        ra = results["arms"]["family"].get("rule_acc_by_seq") or [float("nan")] * k_seqs
        for k in range(k_seqs):
            fg = f"{flo_got[k]:+.4f}+-{flo_sem[k]:.4f}" if flo_got else "n/a"
            print(f"  {k:<7}{avail[k]:>14.4f}"
                  f"{got[k]:>+12.4f}+-{sem[k]:.4f}{fg:>19}"
                  f"{net[k]:>9.4f}{icl_frac[k]:>9.3f}{ra[k]:>9.4f}"
                  f"{oracle['depth_bayes_rule_acc'][k]:>8.4f}")
        print(f"\n  chance rule acc {1 / R:.4f}   "
              f"corr(net decline, oracle available) across depths = {cc:+.3f}",
              flush=True)

    out_dir = f"{DATA_DIR}/{key}/rule_family"
    os.makedirs(out_dir, exist_ok=True)
    ph = "" if phase == "aligned" else f"_{phase}"
    name = f"gate0_{design}_K{k_seqs}{ph}{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
