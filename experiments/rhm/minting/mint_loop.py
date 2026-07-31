"""Minting against a non-mirror verifier: does the acceptance signal's TYPE matter?

Builds §7 of `ideas/breadth_as_grader_heterogeneity.md`. A learner is seeded on a bounded
corpus, then repeatedly MINTS candidate sequences and folds the accepted ones back into its
own training pool -- a self-curated curriculum from newly-minted data (§2's third cell).
The load-bearing variable is WHAT ACCEPTS:

  verifier : the RHM rule table. Non-mirror -- it is the exact generative process, not made
             of the model. "Does the code run."
  mirror   : the model's own mean token log-likelihood. §5 predicts this degenerates, because
             an acceptance signal derived from the learner is "the same grader in a mirror,
             blind in the same places" -- i.e. model autophagy.
  random   : accept uniformly at random. The control §7's own confound note demands: mint-and-
             accept drifts toward the easy part of the band REGARDLESS of grader type, which
             would mimic the mirror arm's predicted failure and mimic away the verifier arm's
             predicted success. Without this arm the headline contrast is uninterpretable.
  mirror_dedup : mirror, but exact repeats are dropped BEFORE the top-k. The plain mirror
             arm collapses to accepting what it already memorised, which confounds its
             failure with replay; this variant separates the two.
  real_data: not minting at all -- genuine held-out sequences at the same quota. The
             CEILING, and the control that makes "verifier beats static" interpretable,
             since the random arm shows most of that gap is raw data volume.
  static   : no minting, matched compute. The seed-only control.

crossed with SEED BREADTH -- narrow (Exp 1's SUBTREE, roots {0,1,2,3}) vs broad (FULL).

WHAT IS CONTROLLED. Every arm shares one rule table, one initialisation, one warm checkpoint
per breadth (arms fork from it), one total step budget (warmup + rounds*steps_per_round), one
candidate budget per round, and one acceptance quota per round -- so the pools grow by exactly
the same number of sequences and only their CONTENTS differ. The generative probe is run on a
FROZEN held-out prefix set, so it cannot move just because an arm's pool moved.

WHY THE SEED POOL IS BOUNDED (a departure from Exp 1, deliberate). Exp 1 draws from a 200k-
sequence pool, which is effectively unmetered; under an unmetered seed, minting cannot be an
expansion channel by construction (§2's cap is about FIXED content, §3's meter is about
scarcity), so "does the verifier arm exceed the static control" would be untestable. The seed
here is bounded and the static controls are re-run inside this harness at matched compute
rather than read off Exp 1. Exp 1's anchors (NTP floor root 0.08, oracle-aux ceiling 0.80)
remain the external reference, not an internal control.

READOUTS
  1. Rule-violation rate in the ACCEPTED pool, per round. §5's predicted mirror degeneration
     with a number attached. Zero by construction for the verifier arm; the measurement is of
     the mirror and random arms.
  2. Generative validity on FROZEN prefixes -- what fraction of the learner's own completions
     parse, and how many levels they survive. Arm-comparable, pool-independent.
  3. Coverage / diversity per round (§7's mandated confound instrument): root-set support,
     fraction of accepted sequences whose root set lies ENTIRELY OUTSIDE the seed's roots
     (unambiguous support expansion), per-level feature union, distinct-sequence fraction,
     duplication against the existing pool, aligned-pair entropy, and the log-likelihood
     sharpening that acceptance induces.
  4. Per-level ancestor recovery d1..d6 on a frozen held-out probe drawn from the FULL tree
     (plus S and complement), and held-out NTP loss on frozen full/S/complement corpora.

Run (from experiments/, MODAL_PROFILE=chromatic):
  modal run rhm/minting/mint_loop.py::prep --breadth narrow --sweep   # diagnostic first
  modal run --detach rhm/minting/mint_loop.py::minting --tag v1
"""

import json
import os

import modal

from rhm.rhm_data import generate_rules_distinct, possible_set_parse
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-minting", image=image)

# Ordered along the MIRROR -> NON-MIRROR axis, which is the thing under test. §5 states the
# criterion as a dichotomy and asks whether it admits degrees; this is that axis made
# explicit, with `peer` as the only rung that is both learned and unprivileged.
ARMS = [("narrow", "static"),        # floor: bounded seed, no minting
        ("narrow", "mirror"),        # perfect mirror: the learner's own likelihood
        ("narrow", "mirror_dedup"),  # mirror minus replay
        ("narrow", "peer"),          # learned, unprivileged, sighted on the complement
        ("narrow", "verifier@4"),    # graded non-mirror: parse only to level 4
        ("narrow", "verifier"),      # exact rule table -- privileged, it IS the DGP
        ("narrow", "random"),        # matched volume, no grader at all
        ("narrow", "peer_verifier"), # LEGALITY (rules) composed with LOCATION (peer)
        ("narrow", "real_data"),     # ceiling: genuine held-out data at the same quota
        ("broad", "static")]         # the breadth reference

ARM_ORDER = [f"{b}_{a}" for b, a in ARMS]


# ======================================================================
# Data + probe helpers (copied from specialization/rhm_subtree_specialization.py
# so this sub-experiment stays self-contained and Exp 1 stays untouched)
# ======================================================================

def _generate_with_traces(rules, n_sequences, seed, roots=None):
    """Aligned leaf sequences + true latent feature at every node.
    level_features[ell]: (n_seq, s^ell) parent features; [0]=roots, [L]=leaves.
    roots=None -> uniform over all v (FULL); a list -> root-restricted (SUBTREE)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features = []
    if roots is None:
        current = rng.integers(0, v, size=(n_sequences, 1))
    else:
        current = rng.choice(np.asarray(roots, dtype=np.int64), size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        nxt = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
    level_features.append(current.copy())
    return current, level_features


def _probe_acc(X, y, v, device, steps, lr, hidden=0, wd=1e-4, train_frac=0.7):
    """Decodability of label y from X. hidden=0 -> linear; hidden>0 -> 1-layer MLP."""
    import torch
    import torch.nn as nn
    n = X.shape[0]
    split = int(train_frac * n)
    mu, sd = X[:split].mean(0, keepdim=True), X[:split].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr, Xte, yte = Xn[:split], y[:split], Xn[split:], y[split:]
    if hidden:
        clf = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, v)).to(device)
    else:
        clf = nn.Linear(X.shape[1], v).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (clf(Xte).argmax(1) == yte).float().mean().item()


def _probe_model(model, eval_x, y_level, block_names, L, v, device,
                 probe_steps, probe_lr, mlp_hidden, mlp_steps, label):
    """Per-level ancestor recovery (linear + MLP, best-over-blocks), reported as d{L-ell}."""
    import torch
    model.eval()
    acc = {b: [] for b in block_names}
    with torch.no_grad():
        for i in range(0, len(eval_x), 256):
            _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
            for b in block_names:
                acc[b].append(inter[b][:, -1, :].float())
    acts = {b: torch.cat(acc[b], 0) for b in block_names}
    lin_best, mlp_best = {}, {}
    for ell in range(L):
        lin_best[ell] = max(
            _probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr)
            for b in block_names)
        if mlp_hidden:
            mlp_best[ell] = max(
                _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, 1e-3,
                           hidden=mlp_hidden, wd=1e-3) for b in block_names)
    del acts
    torch.cuda.empty_cache()
    print(f"  [{label}] MLP best:  " +
          "  ".join(f"d{L-ell}:{mlp_best[ell]:.3f}" for ell in range(L)))
    return {"linear_best": {f"d{L-ell}": lin_best[ell] for ell in range(L)},
            "mlp_best": {f"d{L-ell}": mlp_best[ell] for ell in range(L)}}


# ======================================================================
# The mint: generation, acceptance, and the coverage instruments
# ======================================================================

def _sample_completions(model, prefixes, T, temperature, device, gen, batch=1024):
    """Complete each prefix to length T by ancestral sampling from the model.

    Returns (sequences (B,T) int64 numpy, mean own-log-likelihood of the GENERATED span).
    The score is always computed from the UNTEMPERED log-softmax, so the mirror arm's
    acceptance signal is the model's own likelihood even if sampling is tempered.

    Prefix-conditioned rather than unconditional because training is phase-diverse (random
    windows over the concatenated corpus, exactly as Exp 1), so the model has no way to know
    it is at sequence position 0; the prefix supplies the phase. `prefix_frac` is a controlled
    variable shared by every arm.
    """
    import torch
    import torch.nn.functional as F
    model.eval()
    p = prefixes.shape[1]
    outs, scores = [], []
    with torch.no_grad():
        for lo in range(0, prefixes.shape[0], batch):
            x = prefixes[lo:lo + batch].to(device)
            lp = torch.zeros(x.shape[0], device=device)
            for _ in range(p, T):
                logits, _ = model(x)
                logits = logits[:, -1, :].float()
                nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1,
                                        generator=gen)
                lp += F.log_softmax(logits, dim=-1).gather(1, nxt).squeeze(1)
                x = torch.cat([x, nxt], dim=1)
            outs.append(x.cpu())
            scores.append((lp / max(1, T - p)).cpu())
    return torch.cat(outs).numpy().astype("int64"), torch.cat(scores).numpy()


def _score_seqs(model, seqs, p, device, batch=1024):
    """Mean log-likelihood of each sequence's GENERATED span (positions p..T-1) under
    `model`, teacher-forced. Used to score candidates under the frozen peer grader."""
    import torch
    import torch.nn.functional as F
    model.eval()
    x = torch.from_numpy(seqs)
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            b = x[lo:lo + batch].to(device)
            logits, _ = model(b[:, :-1])
            lp = F.log_softmax(logits.float(), dim=-1).gather(
                2, b[:, 1:, None]).squeeze(2)          # logits[:, i] predicts token i+1
            out.append(lp[:, p - 1:].mean(1).cpu())
    return torch.cat(out).numpy()


def _seq_stats(seqs, rules, S, v, L, s, pool_keys=None, logp=None):
    """Everything §7 asks to be instrumented per round, on one set of sequences."""
    import numpy as np
    par = possible_set_parse(seqs, rules)
    rs, valid = par["root_sets"], par["valid"]
    comp = [r for r in range(v) if r not in S]
    keys = [bytes(row) for row in seqs.astype("uint8")]
    uniq = set(keys)
    pairs = (seqs[:, 0::s] * v + seqs[:, 1::s]).reshape(-1)
    hist = np.bincount(pairs, minlength=v ** s).astype(float)
    hist /= max(1.0, hist.sum())
    nz = hist[hist > 0]
    out = {
        "n": int(len(seqs)),
        # (1) rule-violation rate -- the §5 measurement
        "valid_frac": float(valid.mean()),
        "parse_level_mean": float(par["parse_level"].mean()),
        "parse_level_hist": np.bincount(par["parse_level"], minlength=L + 1).tolist(),
        # (3) coverage / diversity -- the confound instrument
        "root_union_size": int(rs.any(0).sum()),
        "root_set_size_mean": float(rs.sum(1)[valid].mean()) if valid.any() else 0.0,
        "root_only_outside_S": float((rs[:, comp].any(1) & ~rs[:, S].any(1)).mean()),
        "root_any_outside_S": float(rs[:, comp].any(1).mean()),
        "level_union_size": par["level_union"].sum(1).tolist(),   # index 0 = root level
        "distinct_frac": float(len(uniq) / max(1, len(seqs))),
        "pair_entropy_nats": float(-(nz * np.log(nz)).sum()),
    }
    if pool_keys is not None:
        out["dup_with_pool_frac"] = float(sum(k in pool_keys for k in keys) / max(1, len(keys)))
    if logp is not None:
        out["mean_logprob"] = float(np.mean(logp))
    return out


def _accept(acceptance, cands, logp, rules, quota, rng, pool_keys=None, peer_logp=None):
    """Return the indices this acceptance rule folds into the pool.

    Every rule returns at most `quota` items, so pool growth is identical across arms and
    the single controlled variable is WHICH sequences are kept, not how many.

    The acceptance rules span an axis, not a dichotomy -- which is §5's own open question
    ("does the mirror-grader criterion admit degrees?"). Ordered from perfect mirror to
    perfect non-mirror:
        mirror      -- the learner's own likelihood. Zero information beyond the dense loss.
        peer        -- a frozen model trained on a DISJOINT slice of the tree. Learned, so
                       its blind spots are real; unprivileged, so it is not the DGP; and
                       sighted where the learner is blind, because it saw the complement.
                       This is §4's "distance => decorrelated blind spots" made measurable
                       instead of asserted.
        verifier@k  -- the rule table, but only required to parse to level k. A graded
                       non-mirror grader; k=L is the full verifier.
        verifier    -- the exact rule table to the root. Privileged: it IS the DGP.
    """
    import numpy as np
    if acceptance == "verifier" or acceptance.startswith("verifier@"):
        par = possible_set_parse(cands, rules)
        need = len(rules) if acceptance == "verifier" else int(acceptance.split("@")[1])
        ok = np.flatnonzero(par["parse_level"] >= need)
        yielded = len(ok)
        if len(ok) > quota:
            ok = rng.choice(ok, size=quota, replace=False)
        return np.sort(ok), yielded
    if acceptance == "peer":
        order = np.argsort(-peer_logp)[:quota]      # top-k by the PEER's likelihood
        return np.sort(order), len(cands)
    if acceptance == "peer_verifier":
        # Two differently-typed graders composed. The verifier gates LEGALITY, the peer
        # gates LOCATION (is this from the region I do not know) -- the m=2/m=3 crossover
        # showed these are separate functions, not two points on one mirror-ness axis.
        # This is heterogeneous_graders §10's redundancy-vs-heterogeneity falsifier, which
        # §7 names: a second differently-typed grader should beat doubling the first's budget.
        ok = np.flatnonzero(possible_set_parse(cands, rules)["valid"])
        if len(ok) == 0:
            return ok, 0
        order = ok[np.argsort(-peer_logp[ok])[:quota]]
        return np.sort(order), len(ok)
    if acceptance == "mirror":
        order = np.argsort(-logp)[:quota]          # top-k by own likelihood
        return np.sort(order), len(cands)
    if acceptance == "mirror_dedup":
        # The plain mirror arm collapses to accepting sequences already in its pool (top
        # likelihood == memorised), so its failure is confounded with duplication. This
        # variant gives the mirror grader the benefit of the doubt: drop exact repeats
        # first, THEN take top-k by own likelihood. If it still fails, the failure is the
        # mirror geometry; if it recovers, the failure was only replay.
        fresh = np.array([i for i, row in enumerate(cands.astype("uint8"))
                          if bytes(row) not in pool_keys], dtype=np.int64)
        if len(fresh) == 0:
            return np.zeros(0, dtype=np.int64), 0
        order = fresh[np.argsort(-logp[fresh])[:quota]]
        return np.sort(order), len(fresh)
    if acceptance == "random":
        idx = rng.choice(len(cands), size=min(quota, len(cands)), replace=False)
        return np.sort(idx), len(cands)
    raise ValueError(f"unknown acceptance rule {acceptance!r}")


# ======================================================================
# Shared setup: one rule table, one init, deterministic pools and frozen probes
# ======================================================================

def _setup(cfg):
    """Everything that must be bit-identical across arms and across containers."""
    import numpy as np
    v, s, L = cfg["v"], cfg["s"], cfg["depth"]
    T = s ** L
    rules = generate_rules_distinct(v, s, L, cfg["m"], seed=cfg["rule_seed"])
    S = sorted({int(x) for x in cfg["subtree_roots"].split(",") if x.strip()})
    comp = [r for r in range(v) if r not in S]

    seed_pool = {
        "narrow": _generate_with_traces(rules, cfg["seed_pool"], cfg["data_seed"] + 100, S)[0],
        "broad": _generate_with_traces(rules, cfg["seed_pool"], cfg["data_seed"])[0],
        # the peer grader's corpus: roots DISJOINT from the narrow seed, so a peer trained
        # here is sighted exactly where a narrow-seeded learner is blind
        "peer": _generate_with_traces(rules, cfg["seed_pool"],
                                      cfg["data_seed"] + 200, comp)[0],
    }
    # frozen held-out sets: ancestor probes, NTP val corpora, generative-probe prefixes
    evals, val_corpora = {}, {}
    for i, (name, r) in enumerate((("S", S), ("complement", comp), ("full", None))):
        sq, lf = _generate_with_traces(rules, cfg["n_eval"], cfg["eval_seed"] + i, r)
        evals[name] = (sq, lf)
        val_corpora[name] = _generate_with_traces(
            rules, cfg["n_val"], cfg["eval_seed"] + 100 + i, r)[0]
    gen_prefix = {
        "full": _generate_with_traces(rules, cfg["n_gen_probe"], cfg["eval_seed"] + 200)[0],
        "S": _generate_with_traces(rules, cfg["n_gen_probe"], cfg["eval_seed"] + 201, S)[0],
    }
    return dict(rules=rules, S=S, comp=comp, T=T, seed_pool=seed_pool, evals=evals,
                val_corpora=val_corpora, gen_prefix=gen_prefix)


def _prefix_len(cfg):
    """Prefix length in tokens, snapped down to a whole s-tuple boundary."""
    T = cfg["s"] ** cfg["depth"]
    return max(cfg["s"], (int(cfg["prefix_frac"] * T) // cfg["s"]) * cfg["s"])


def _default_cfg(**kw):
    cfg = dict(
        # DGP -- defaults are Exp 1's canonical frontier-stall regime. depth/m are
        # parameters because the yield diagnostic sweeps them: a learner stalled at ~d3.5
        # may be structurally unable to mint anything that parses all the way to a root,
        # and specialization/README.md's standing control requires a regime where the
        # target level is demonstrably reachable under plain NTP.
        v=16, s=2, depth=6, m=4, rule_seed=0, subtree_roots="0,1,2,3",
        # model -- Exp 1's, unchanged (~6.34M)
        n_layer=8, n_head=8, n_embd=256,
        # the metered seed, and the matched step budget (8000 + 8*1500 = 20000 = Exp 1's)
        seed_pool=8192, warmup_steps=8000, rounds=8, steps_per_round=1500,
        batch_size=64, lr=3e-4, weight_decay=0.01, data_seed=7,
        # the mint. prefix is a FRACTION of T so it means the same thing across regimes
        # (see _prefix_len); it supplies the phase that phase-diverse training withholds.
        n_candidates=8192, accept_quota=1024, prefix_frac=0.5, temperature=1.0,
        mint_weight=0.5,
        # measurement
        n_eval=8000, n_val=2000, n_gen_probe=2048, eval_seed=999,
        probe_steps=600, probe_lr=1e-2, mlp_hidden=128, mlp_steps=800,
        seed=42, quick=False,
    )
    cfg.update(kw)
    if cfg["quick"]:
        cfg.update(seed_pool=1024, warmup_steps=400, rounds=2, steps_per_round=200,
                   n_candidates=1024, accept_quota=256, n_eval=800, n_val=400,
                   n_gen_probe=512, probe_steps=120, mlp_steps=120)
    return cfg


def _make_batcher(pool, T, batch_size, device):
    """Phase-diverse NTP over the concatenated pool -- Exp 1's objective verbatim."""
    import numpy as np
    import torch
    corpus = torch.from_numpy(np.ascontiguousarray(pool.reshape(-1)))
    n = corpus.shape[0]
    ar = torch.arange(T)

    def get_batch(gen):
        ix = torch.randint(0, n - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + ar[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)
    return get_batch


def _make_mixed_batcher(seed_pool, mint_pool, T, batch_size, mint_weight, device):
    """Phase-diverse NTP drawn from TWO pools at a fixed gradient share.

    Growing the pool is the obvious way to let minted data matter, but it couples the dose
    to the generation budget: at quota 1024 x 8 rounds against a 65536-sequence seed the
    minted data is ~11% of the pool, and the first m=3 run showed that even PERFECT added
    data (the real_data ceiling) then moves d5 by only +0.034. Nothing about acceptance
    TYPE can show through an 11% dose.

    So the dose becomes an explicit controlled variable: `mint_weight` of every batch is
    drawn from the minted pool regardless of its size. This is also the more faithful
    reading of §2's "self-curating an external curriculum" -- the claim is about what the
    learner's curriculum consists of, not how many sequences it has filed away.
    """
    import numpy as np
    import torch
    seed_gb = _make_batcher(seed_pool, T, batch_size, device)
    if mint_pool is None or len(mint_pool) == 0:
        return seed_gb
    if mint_weight < 0:
        # POOLED: sample uniformly over seed+mint, so the minted share is its SIZE share.
        # This is the correct mechanism once the quota is large enough for the minted pool
        # to rival the seed. Fixing the share instead (mint_weight >= 0) concentrates the
        # gradient on a small pool, and the dose run showed that damages every arm equally
        # -- real held-out data included (d5 -0.177, d6 -0.302) -- so the harm was the
        # concentration, not the content, and acceptance type could not show through it.
        return _make_batcher(np.concatenate([seed_pool, mint_pool], axis=0), T,
                             batch_size, device)
    if mint_weight <= 0:
        return seed_gb
    n_mint = int(round(batch_size * mint_weight))
    if n_mint <= 0:
        return seed_gb
    n_seed = batch_size - n_mint
    mint_gb = _make_batcher(mint_pool, T, n_mint, device)
    if n_seed == 0:
        return mint_gb
    seed_part = _make_batcher(seed_pool, T, n_seed, device)

    def get_batch(gen):
        xs, ys = seed_part(gen)
        xm, ym = mint_gb(gen)
        return torch.cat([xs, xm], 0), torch.cat([ys, ym], 0)
    return get_batch


def _val_loss(model, pool, T, batch_size, device, seed, n=10):
    import torch
    gb = _make_batcher(pool, T, batch_size, device)
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            x, y = gb(gen)
            tot += model(x, y)[1].item()
    return tot / n


def _train(model, opt, get_batch, steps, gen, device, label, log_every=500):
    import torch
    for step in range(steps):
        model.train()
        x, y = get_batch(gen)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"    [{label}] step {step:6d}: ntp={loss.item():.4f}", flush=True)


def _regime_key(cfg):
    return f"v{cfg['v']}_s{cfg['s']}_L{cfg['depth']}_m{cfg['m']}"


def _warm_path(cfg, breadth):
    return (f"{DATA_DIR}/rhm_minting/warm/{_regime_key(cfg)}"
            f"_pool{cfg['seed_pool']}_w{cfg['warmup_steps']}_s{cfg['seed']}_{breadth}"
            f"{'_quick' if cfg['quick'] else ''}.pt")


# ======================================================================
# prep -- one warm checkpoint per breadth, forked by every arm of that breadth
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def prep(breadth: str = "narrow", sweep: bool = False, force: bool = False,
         quick: bool = False, seed_pool: int = 8192, warmup_steps: int = 8000,
         depth: int = 6, m: int = 4, v: int = 16, seed: int = 42,
         sweep_prefix: str = "0.125,0.25,0.5,0.75", sweep_temp: str = "1.0,0.8"):
    import numpy as np
    import torch
    from rhm.model import GPT

    volume.reload()
    cfg = _default_cfg(quick=quick, seed_pool=seed_pool, warmup_steps=warmup_steps,
                       depth=depth, m=m, v=v, seed=seed)
    su = _setup(cfg)
    device, T = "cuda", su["T"]
    path = _warm_path(cfg, breadth)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.manual_seed(cfg["seed"])
    model = GPT(cfg["v"], T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)

    if os.path.exists(path) and not force:
        print(f"warm checkpoint exists -> {path}")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"{'='*74}\nWARMUP  breadth={breadth}  seed_pool={cfg['seed_pool']} seqs "
              f"({cfg['seed_pool']*T:,} tokens)  steps={cfg['warmup_steps']}\n{'='*74}")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        gb = _make_batcher(su["seed_pool"][breadth], T, cfg["batch_size"], device)
        _train(model, opt, gb, cfg["warmup_steps"],
               torch.Generator().manual_seed(cfg["seed"] + 1), device, f"warm/{breadth}")
        torch.save(model.state_dict(), path)
        volume.commit()
        print(f"saved -> {path}")

    for name, corp in su["val_corpora"].items():
        print(f"  warm val NTP [{name}]: "
              f"{_val_loss(model, corp, T, cfg['batch_size'], device, cfg['seed']):.4f}")

    if not sweep:
        return {"path": path}

    # Diagnostic: the mint's yield is the experiment's main unknown. Sweep the two knobs
    # that set it (how much phase/context the prefix supplies, and sampling temperature)
    # before committing a step budget to arms that might accept nothing.
    print(f"\n{'='*74}\nMINT YIELD SWEEP  {_regime_key(cfg)}  T={T}  "
          f"constraints={sum(T // cfg['s']**k for k in range(1, cfg['depth']+1))}  "
          f"(frozen full-tree prefixes, n={cfg['n_gen_probe']})")
    print(f"{'prefix_len':>10} {'temp':>6} {'valid':>8} {'parse_lvl':>10} {'root_out_S':>11} "
          f"{'distinct':>9} {'pair_H':>8}")
    real = _seq_stats(su["gen_prefix"]["full"], su["rules"], su["S"], cfg["v"],
                      cfg["depth"], cfg["s"])
    print(f"{'REAL data':>10} {'-':>6} {real['valid_frac']:>8.4f} "
          f"{real['parse_level_mean']:>10.3f} {real['root_only_outside_S']:>11.3f} "
          f"{real['distinct_frac']:>9.3f} {real['pair_entropy_nats']:>8.3f}")
    rows = []
    fracs = [float(x) for x in sweep_prefix.split(",")]
    for p in sorted({_prefix_len({**cfg, "prefix_frac": f}) for f in fracs}):
        for temp in [float(x) for x in sweep_temp.split(",")]:
            g = torch.Generator(device=device).manual_seed(cfg["seed"] + 7)
            pre = torch.from_numpy(su["gen_prefix"]["full"][:, :p])
            cands, lp = _sample_completions(model, pre, T, temp, device, g)
            st = _seq_stats(cands, su["rules"], su["S"], cfg["v"], cfg["depth"], cfg["s"],
                            logp=lp)
            st.update(prefix_len=p, temperature=temp)
            rows.append(st)
            print(f"{p:>10} {temp:>6.2f} {st['valid_frac']:>8.4f} "
                  f"{st['parse_level_mean']:>10.3f} {st['root_only_outside_S']:>11.3f} "
                  f"{st['distinct_frac']:>9.3f} {st['pair_entropy_nats']:>8.3f}", flush=True)
    return {"path": path, "sweep": rows, "real": real}


# ======================================================================
# arm -- one (breadth, acceptance) cell
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def arm(breadth: str, acceptance: str, tag: str = "", quick: bool = False,
        seed_pool: int = 8192, warmup_steps: int = 8000, rounds: int = 8,
        steps_per_round: int = 1500, n_candidates: int = 8192, accept_quota: int = 1024,
        prefix_frac: float = 0.5, temperature: float = 1.0, mint_weight: float = 0.5,
        depth: int = 6, m: int = 4, v: int = 16, seed: int = 42):
    import numpy as np
    import torch
    from rhm.model import GPT

    volume.reload()
    cfg = _default_cfg(quick=quick, seed_pool=seed_pool, warmup_steps=warmup_steps,
                       rounds=rounds, steps_per_round=steps_per_round,
                       n_candidates=n_candidates, accept_quota=accept_quota,
                       prefix_frac=prefix_frac, temperature=temperature,
                       mint_weight=mint_weight, depth=depth, m=m, v=v, seed=seed)
    su = _setup(cfg)
    device, T, L, v, s = "cuda", su["T"], cfg["depth"], cfg["v"], cfg["s"]
    rules, S = su["rules"], su["S"]

    print(f"{'='*74}\nARM  breadth={breadth}  acceptance={acceptance}  tag={tag}")
    print(f"  seed_pool={cfg['seed_pool']}  rounds={cfg['rounds']}x{cfg['steps_per_round']} "
          f"steps  candidates={cfg['n_candidates']}  quota={cfg['accept_quota']}  "
          f"prefix={_prefix_len(cfg)}/{su['T']}  temp={cfg['temperature']}\n{'='*74}", flush=True)

    torch.manual_seed(cfg["seed"])
    model = GPT(v, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    model.load_state_dict(torch.load(_warm_path(cfg, breadth), map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    peer = None
    if acceptance in ("peer", "peer_verifier"):
        peer = GPT(v, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
        peer.load_state_dict(torch.load(_warm_path(cfg, "peer"), map_location=device))
        peer.eval()
        for q in peer.parameters():
            q.requires_grad_(False)
        print(f"  peer grader loaded (trained on roots {su['comp']}) -- frozen", flush=True)

    seed_data = su["seed_pool"][breadth]
    mint_data = np.zeros((0, T), dtype=np.int64)     # kept SEPARATE so the dose is a knob
    pool_keys = {bytes(r) for r in seed_data.astype("uint8")}
    rng = np.random.default_rng(cfg["seed"] + 11)
    train_gen = torch.Generator().manual_seed(cfg["seed"] + 2)
    cuda_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 3)
    seed_stats = _seq_stats(seed_data, rules, S, v, L, s)
    rounds = []

    for rnd in range(cfg["rounds"]):
        get_batch = _make_mixed_batcher(seed_data, mint_data, T, cfg["batch_size"],
                                        cfg["mint_weight"], device)
        _train(model, opt, get_batch, cfg["steps_per_round"], train_gen, device,
               f"{breadth}/{acceptance} r{rnd}")

        rec = {"round": rnd, "pool_size": int(len(seed_data) + len(mint_data)),
               "mint_pool_size": int(len(mint_data)),
               "mint_gradient_share": (cfg["mint_weight"] if len(mint_data) else 0.0)}
        rec["val"] = {n: _val_loss(model, c, T, cfg["batch_size"], device, cfg["seed"])
                      for n, c in su["val_corpora"].items()}
        # (2) generative validity on FROZEN prefixes -- arm-comparable, pool-independent.
        #     Its own generator, reseeded per round, so the sampling noise is identical
        #     across arms given the same model and does not depend on how much minting
        #     RNG an arm consumed earlier.
        rec["gen_probe"] = {}
        probe_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 500 + rnd)
        for pname, pre in su["gen_prefix"].items():
            gp = torch.from_numpy(np.ascontiguousarray(pre[:, :_prefix_len(cfg)]))
            cs, lp = _sample_completions(model, gp, T, cfg["temperature"], device, probe_gen)
            rec["gen_probe"][pname] = _seq_stats(cs, rules, S, v, L, s, logp=lp)

        if acceptance == "real_data":
            # The CEILING for minting: genuine held-out sequences from this arm's own seed
            # distribution, at exactly the minting quota. Without it, "verifier beats
            # static" is uninterpretable -- the random arm shows most of that gap is raw
            # data volume, so the question is how much of REAL data's value minting recovers.
            acc = _generate_with_traces(rules, cfg["accept_quota"],
                                        cfg["eval_seed"] + 9000 + rnd,
                                        S if breadth == "narrow" else None)[0]
            rec["accepted"] = _seq_stats(acc, rules, S, v, L, s, pool_keys)
            rec["accepted"].update(yield_=cfg["accept_quota"], quota_met=True)
            mint_data = np.concatenate([mint_data, acc], axis=0)
            pool_keys |= {bytes(r) for r in acc.astype("uint8")}
            print(f"  [r{rnd}] mint pool ->{len(mint_data)}  (+{len(acc)} REAL "
                  f"held-out sequences)", flush=True)
        elif acceptance != "static":
            src = seed_data if len(mint_data) == 0 else np.concatenate(
                [seed_data, mint_data], axis=0)
            idx = rng.choice(len(src), size=cfg["n_candidates"], replace=True)
            pre = torch.from_numpy(np.ascontiguousarray(src[idx][:, :_prefix_len(cfg)]))
            cands, lp = _sample_completions(model, pre, T, cfg["temperature"], device,
                                            cuda_gen)
            rec["candidates"] = _seq_stats(cands, rules, S, v, L, s, pool_keys, lp)
            plp = (_score_seqs(peer, cands, _prefix_len(cfg), device)
                   if peer is not None else None)
            if plp is not None:
                rec["candidates"]["peer_mean_logprob"] = float(plp.mean())
            keep, yielded = _accept(acceptance, cands, lp, rules, cfg["accept_quota"], rng,
                                    pool_keys, plp)
            acc = cands[keep]
            rec["accepted"] = _seq_stats(acc, rules, S, v, L, s, pool_keys, lp[keep])
            rec["accepted"]["yield_"] = int(yielded)
            rec["accepted"]["quota_met"] = bool(len(keep) >= cfg["accept_quota"])
            mint_data = np.concatenate([mint_data, acc], axis=0)
            pool_keys |= {bytes(r) for r in acc.astype("uint8")}
            a = rec["accepted"]
            print(f"  [r{rnd}] mint pool ->{len(mint_data)}  yield {yielded}/"
                  f"{cfg['n_candidates']} accepted {len(keep)}  valid {a['valid_frac']:.3f} "
                  f"parse_lvl {a['parse_level_mean']:.2f}  outS {a['root_only_outside_S']:.3f} "
                  f"distinct {a['distinct_frac']:.3f}  pairH {a['pair_entropy_nats']:.3f}",
                  flush=True)
        g = rec["gen_probe"]["full"]
        print(f"  [r{rnd}] val(full)={rec['val']['full']:.4f}  gen_probe(full) valid "
              f"{g['valid_frac']:.4f} parse_lvl {g['parse_level_mean']:.3f}", flush=True)
        rounds.append(rec)

    # (4) final ancestor probes on the frozen held-out sets
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(cfg["n_layer"])]
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    recovery = {}
    for name, (sq, lf) in su["evals"].items():
        ex = torch.from_numpy(sq).to(device)
        yl = {ell: torch.from_numpy(lf[ell][:, last_anc[ell]]).to(device) for ell in range(L)}
        recovery[name] = _probe_model(model, ex, yl, block_names, L, v, device,
                                      cfg["probe_steps"], cfg["probe_lr"],
                                      cfg["mlp_hidden"], cfg["mlp_steps"],
                                      f"{breadth}/{acceptance}|{name}")
        del ex, yl
        torch.cuda.empty_cache()

    res = {"config": cfg, "breadth": breadth, "acceptance": acceptance,
           "seed_pool_stats": seed_stats, "rounds": rounds, "recovery": recovery,
           "final_pool_size": int(len(seed_data) + len(mint_data)),
           "final_mint_pool_size": int(len(mint_data)),
           "final_mint_stats": (_seq_stats(mint_data, rules, S, v, L, s)
                                if len(mint_data) else None),
           "final_pool_stats": _seq_stats(
               np.concatenate([seed_data, mint_data], axis=0) if len(mint_data)
               else seed_data, rules, S, v, L, s)}
    out_dir = (f"{DATA_DIR}/rhm_minting/v{v}_s{s}_L{L}_m{cfg['m']}_S{len(S)}of{v}"
               f"_pool{cfg['seed_pool']}{('_' + tag) if tag else ''}"
               f"{'_quick' if cfg['quick'] else ''}")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{breadth}_{acceptance}.json", "w") as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"Saved -> {out_dir}/{breadth}_{acceptance}.json")
    return res


# ======================================================================
# orchestrator
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=8192)
def minting(tag: str = "v1", arms: str = "", quick: bool = False,
            seed_pool: int = 8192, warmup_steps: int = 8000, rounds: int = 8,
            steps_per_round: int = 1500, n_candidates: int = 8192,
            accept_quota: int = 1024, prefix_frac: float = 0.5,
            temperature: float = 1.0, mint_weight: float = 0.5,
            depth: int = 6, m: int = 4, v: int = 16, seed: int = 42):
    kw = dict(quick=quick, seed_pool=seed_pool, warmup_steps=warmup_steps,
              depth=depth, m=m, v=v, seed=seed)
    cfg = _default_cfg(rounds=rounds, steps_per_round=steps_per_round,
                       n_candidates=n_candidates, accept_quota=accept_quota,
                       prefix_frac=prefix_frac, temperature=temperature,
                       mint_weight=mint_weight, **kw)
    want = ARMS if not arms else [tuple(a.split(":")) for a in arms.split(",")]
    breadths = sorted({b for b, _ in want})
    if any(a in ("peer", "peer_verifier") for _, a in want):
        breadths.append("peer")          # trained on the complement, used frozen as grader

    print(f"prep: warm checkpoints for {breadths}")
    for h in [prep.spawn(breadth=b, **kw) for b in breadths]:
        h.get()

    print(f"spawning {len(want)} arms: {want}")
    handles = [((b, a), arm.spawn(
        breadth=b, acceptance=a, tag=tag, rounds=rounds,
        steps_per_round=steps_per_round, n_candidates=n_candidates,
        accept_quota=accept_quota, prefix_frac=prefix_frac, temperature=temperature,
        mint_weight=mint_weight, **kw))
        for b, a in want]
    out = {}
    for (b, a), h in handles:
        try:
            out[f"{b}_{a}"] = h.get()
        except Exception as e:                      # one dead arm must not lose the rest
            print(f"  ARM FAILED {b}/{a}: {e}")

    _report(out, cfg)
    out_dir = (f"{DATA_DIR}/rhm_minting/v{cfg['v']}_s{cfg['s']}_L{cfg['depth']}"
               f"_m{cfg['m']}_S4of{cfg['v']}_pool{cfg['seed_pool']}"
               f"{('_' + tag) if tag else ''}{'_quick' if quick else ''}")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/all.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"Saved -> {out_dir}/all.json")
    return out


def _report(out, cfg):
    L = cfg["depth"]
    dl = [f"d{i}" for i in range(1, L + 1)]
    print(f"\n{'='*100}\nDEPTH RECOVERY on the FROZEN FULL-TREE probe (MLP best-over-blocks)"
          f"   chance=1/{cfg['v']}={1/cfg['v']:.3f}")
    print(f"{'arm':<22}" + "".join(f"{d:>8}" for d in dl) + f"{'val(full)':>11}{'val(comp)':>11}")
    for k, r in out.items():
        mlp = r["recovery"]["full"]["mlp_best"]
        v_ = r["rounds"][-1]["val"]
        print(f"{k:<22}" + "".join(f"{mlp[d]:>8.3f}" for d in dl) +
              f"{v_['full']:>11.4f}{v_['complement']:>11.4f}")

    print(f"\n{'='*100}\nTHE MIRROR TEST: rule-violation rate in the ACCEPTED pool, per round")
    print(f"{'arm':<22}" + "".join(f"{'r'+str(i):>8}" for i in range(cfg["rounds"])))
    for k, r in out.items():
        cells = [rd.get("accepted", {}).get("valid_frac") for rd in r["rounds"]]
        print(f"{k:<22}" + "".join(f"{(1-c):>8.3f}" if c is not None else f"{'-':>8}"
                                   for c in cells))

    print(f"\n{'='*100}\nGENERATIVE VALIDITY on FROZEN full-tree prefixes (arm-comparable)")
    print(f"{'arm':<22}" + "".join(f"{'r'+str(i):>8}" for i in range(cfg["rounds"])))
    for k, r in out.items():
        print(f"{k:<22}" + "".join(f"{rd['gen_probe']['full']['valid_frac']:>8.4f}"
                                   for rd in r["rounds"]))

    print(f"\n{'='*100}\nSUPPORT EXPANSION: accepted sequences whose root set lies ENTIRELY "
          f"outside the seed's roots S")
    print(f"{'arm':<22}" + "".join(f"{'r'+str(i):>8}" for i in range(cfg["rounds"])) +
          f"{'pool_end':>10}")
    for k, r in out.items():
        cells = [rd.get("accepted", {}).get("root_only_outside_S") for rd in r["rounds"]]
        print(f"{k:<22}" + "".join(f"{c:>8.3f}" if c is not None else f"{'-':>8}"
                                   for c in cells) +
              f"{r['final_pool_stats']['root_only_outside_S']:>10.3f}")

    print(f"\n{'='*100}\nMODE COLLAPSE: aligned-pair entropy (nats) / distinct-sequence "
          f"fraction of the ACCEPTED pool")
    print(f"{'arm':<22}" + "".join(f"{'r'+str(i):>8}" for i in range(cfg["rounds"])))
    for k, r in out.items():
        cells = [rd.get("accepted", {}).get("pair_entropy_nats") for rd in r["rounds"]]
        print(f"{k:<22}" + "".join(f"{c:>8.3f}" if c is not None else f"{'-':>8}"
                                   for c in cells))
    print(f"{'='*100}")


@app.local_entrypoint()
def smoke(breadth: str = "narrow"):
    """2-min end-to-end shape check (tiny everything). Not a result."""
    prep.remote(breadth=breadth, quick=True, sweep=True)
    arm.remote(breadth=breadth, acceptance="verifier", tag="smoke", quick=True)
