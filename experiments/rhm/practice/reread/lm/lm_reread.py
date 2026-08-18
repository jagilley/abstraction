"""reread/lm — the LM twin: is a FIXED corpus renewable to an NTP learner whose READER climbs?

Parent: `../` (the sculpting round, `rr_s0`). Machinery donor:
[`../../../conditional_revision/`](../../../conditional_revision/README.md) — imported, never
modified. Substrate conventions: `rhm/CLAUDE.md`.

WHY THIS EXISTS. `rr_s0` found the frozen archive renewable in COMPETENCE but not in MINING
YIELD, and the diagnosis was structural: on the sculpting substrate the reader is exogenous
(handed over, pinned at 1.000 block accuracy) and mining reads the agent's OWN repair
(`mine_from="chosen"`, output canonicalised through `canon`), so the archive is rewritten by
the very act of reading it and vocabulary-gated PERCEPTION cannot be expressed. The book-reread
claim's mechanism — you cannot parse level-ℓ until you own level-(ℓ−1) — lives in the reader.
So test it where the reader itself climbs: autoregressive NTP on RHM, where the only
representation the model has is the one it learned, and per-level structure recovery is read
out against exact BP oracles.

THE CONTRAST. A corpus of C sequences drawn ONCE and re-presented for the whole token budget,
against a matched-token stream of FRESH draws from the same DGP. Identical model, identical
optimiser, identical batch sampler, identical number of gradient steps and tokens consumed —
the arms differ ONLY in whether the tokens have been seen before.

  reread signature      deep-level recovery on the frozen corpus keeps climbing long after
                        val loss and shallow-level recovery saturate, and tracks `fresh`
  novelty-bound         the frozen corpus stalls at the point fresh data keeps going

READOUTS, per checkpoint on a log-spaced token ladder:

  1. PER-LEVEL RECOVERY. `conditional_revision`'s own probe (`rhm_latent_loop._probe_acc`) on
     last-position activations, decoding each level's last ancestor; repo convention d1 =
     shallowest, d6 = root. Charted against the EXACT BP ceiling P(z_ℓ | x_{1..T}) from
     `oracle.prefix_beliefs`, so "the probe is weak" and "the model cannot know this" stay
     distinguishable (`probe_diag`'s discipline).
  2. BP-REFERENCED EXCESS LOSS. The model's per-position NLL minus the exact Bayes-optimal
     surprisal from `oracle.revision_and_entropy`, bucketed by the hierarchy level the arriving
     token CLOSES (`pos_top_level`, `conditional_revision`'s convention). Model-independent
     floor, computed once. This is the per-level extraction readout that costs nothing per
     checkpoint.
  3. THE MEMORISATION GAP. Val NLL on held-out fresh windows against train NLL on the arm's own
     corpus. The frozen arm's gap is what "the archive is consumed" would look like.
  4. THE ORDERING READOUT. Tokens-to-reach-level-ℓ (first checkpoint at which recovery crosses
     a declared fraction of that level's Bayes ceiling), per arm. Whether t_ℓ is monotone in ℓ
     IS the vocabulary-gating mechanism — worth having as its own number either way — and
     `t_ℓ(frozen) / t_ℓ(fresh)` per level is the "effectively unlimited data" quantification.

THE CORPUS-SIZE COORDINATE. `rr_s0`'s depth surprise transposes here as corpus size: how many
distinct tokens the archive holds, against how many are consumed. `frozen_2048` is 131k tokens
re-read ~600×; `frozen_200000` is 12.8M tokens re-read ~6×.

FIDELITY GATE. `frozen_200000` at step 12000 with (v16, s2, L6, m4, rule_seed 0, data_seed 7,
seed 42, 8L/8H/256D) IS `conditional_revision`'s cached base — same DGP, same pool size, same
sampler, same optimiser, same steps. Its per-level recovery must reproduce the published
d1 0.979 / d3 0.836 / d6 0.088 before anything from a later checkpoint is trusted.

Run from experiments/:
  modal run -m rhm.practice.reread.lm.lm_reread::gate
  modal run -m rhm.practice.reread.lm.lm_reread::lm_reread --quick --tag smoke0
  python3 rhm/practice/reread/lm/launch_detached.py --fn lm_reread --tag lm0 ...
"""

import hashlib
import json
import os
import time

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, setting_key, volume


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)
app = modal.App("rhm-practice-reread-lm", image=image)

REMOTE = "rhm_practice_reread_lm"

# `conditional_revision`'s published Gate-0 base, the fidelity anchor
CR_REF = {"d1": 0.979, "d3": 0.836, "d6": 0.088}
CR_REF_STEP = 12000


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

def parse_arms(spec):
    """"fresh,frozen_2048" -> [(label, kind, n_seqs)]. `frozen_<n>` freezes a corpus of
    <n> RHM sequences; `fresh` redraws often enough that no token is ever seen twice."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "fresh":
            out.append((part, "fresh", 0))
        elif part.startswith("frozen_"):
            out.append((part, "frozen", int(part.split("_")[1])))
        else:
            raise ValueError(f"unknown arm {part}")
    return out


def ladder(max_steps, extra=(CR_REF_STEP,)):
    """Log-spaced checkpoints, with the fidelity-gate step forced in."""
    import numpy as np
    base = np.unique(np.round(np.geomspace(250, max_steps, 11)).astype(int))
    pts = sorted(set(base.tolist()) | {int(x) for x in extra if x <= max_steps}
                 | {max_steps})
    return [int(x) for x in pts]


# --------------------------------------------------------------------------- #
# the main entrypoint
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def lm_reread(
    tag: str = "smoke",
    arms: str = "fresh,frozen_200000,frozen_16384,frozen_2048,frozen_512",
    # DGP — `conditional_revision`'s regime, so its reference lines transfer
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    # model — `conditional_revision`'s Gate-0 base
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # training — identical protocol
    max_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, data_seed: int = 7, seed: int = 42,
    fresh_every: int = 500,
    # measurement
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    n_oracle: int = 512, oracle_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7",
    thresh_frac: float = 0.5, log_interval: int = 1000, quick: bool = False,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.conditional_revision.oracle import prefix_beliefs, revision_and_entropy

    if quick:
        max_steps, n_eval_sequences, n_oracle = 600, 600, 48
        probe_steps, mlp_steps, fresh_every = 150, 150, 100
        if len(parse_arms(arms)) > 2:
            arms = "fresh,frozen_2048"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T = depth, s ** depth
    key = f"{setting_key(v, s, L, m)}_distinct"
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    arm_list = parse_arms(arms)
    ckpts = [250, max_steps] if quick else ladder(max_steps)
    all_blocks = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    pblocks = [b for b in probe_blocks.split(",") if b in all_blocks] or all_blocks
    started = time.time()

    print(f"{'=' * 78}\nreread/lm  tag={tag}  {key}  {n_layer}L/{n_head}H/{n_embd}D  T={T}")
    print(f"  arms={[a[0] for a in arm_list]}  max_steps={max_steps} "
          f"(tokens={max_steps * batch_size * T:,})  checkpoints={ckpts}")
    print(f"  probe blocks {pblocks}; full best-over-all-blocks at "
          f"{{{CR_REF_STEP}, {max_steps}}}\n{'=' * 78}", flush=True)

    # ---------------- fixed evaluation apparatus, shared by every arm ----------------
    eval_seqs, eval_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(eval_lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    # a held-out FRESH flat corpus: the val set every arm is scored on, never trained on
    val_seqs, _, _ = _generate_with_traces(rules, max(2000, n_eval_sequences // 2),
                                           eval_seed + 1)
    val_corpus = torch.from_numpy(val_seqs.astype(np.int64)).reshape(-1)
    arangeT = torch.arange(T)
    # position -> HIGHEST hierarchy node completed there (`conditional_revision`'s convention)
    pos_top_level = np.array(
        [min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0)
         for p in range(T)], dtype=np.int64)

    # ---- the exact BP references (model-independent, computed once) -----------------
    print("[oracle] exact per-level ceiling P(z_l | x_1..T) ...", flush=True)
    onodes, _, _ = prefix_beliefs(rules, eval_seqs[:min(n_eval_sequences, 2000)], T, L - 1)
    ceiling = {f"d{L - ell}": float(onodes[ell][:, last_anc[ell], :].max(-1).mean())
               for ell in range(L)}
    print(f"[oracle] probe ceilings {ceiling}", flush=True)
    print(f"[oracle] exact Bayes surprisal on {n_oracle} sequences ...", flush=True)
    osq, olf, _ = _generate_with_traces(rules, n_oracle, oracle_seed)
    ores = revision_and_entropy(rules, osq, olf, Ds=[0], chunk=256, verbose=False)
    bayes_pos = ores["surprisal"].mean(0)                       # (T-1,) per position
    oracle_x = torch.from_numpy(osq.astype(np.int64)).to(device)
    lvl_of_arrival = pos_top_level[1:]                          # class of x_{t+1}
    bayes_by_level = {int(k): float(bayes_pos[lvl_of_arrival == k].mean())
                      for k in np.unique(lvl_of_arrival)}
    print(f"[oracle] Bayes surprisal per arrival level {bayes_by_level}", flush=True)

    # ---------------- readout helpers ----------------
    def acts_last(model, blocks):
        out = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, eval_x.shape[0], 256):
                _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
                for b in blocks:
                    out[b].append(inter[b][:, -1, :].float())
        return {b: torch.cat(vs) for b, vs in out.items()}

    def per_level(model, blocks, full):
        a = acts_last(model, blocks)
        res = {}
        for ell in range(L):
            best = 0.0
            for b in blocks:
                best = max(best, _probe_acc(a[b], y_level[ell], v, device,
                                            probe_steps, probe_lr))
                if full:
                    best = max(best, _probe_acc(a[b], y_level[ell], v, device,
                                                mlp_steps, probe_lr, hidden=mlp_hidden))
            res[f"d{L - ell}"] = float(best)
        return res

    def flat_nll(model, corpus, n_batches=20, gseed=0):
        g = torch.Generator().manual_seed(gseed)
        n = corpus.shape[0]
        tot = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                ix = torch.randint(0, n - T - 1, (batch_size,), generator=g)
                idx = ix[:, None] + arangeT[None, :]
                _, loss = model(corpus[idx].to(device), corpus[idx + 1].to(device))
                tot += loss.item()
        return tot / n_batches

    def excess_over_bayes(model):
        """Model per-position NLL on the ALIGNED oracle sequences minus the exact
        Bayes-optimal surprisal, bucketed by the level the arriving token closes."""
        tot = torch.zeros(T - 1, device=device)
        with torch.no_grad():
            for i in range(0, oracle_x.shape[0], 256):
                xb = oracle_x[i:i + 256]
                logits, _ = model(xb[:, :-1].contiguous())
                nll = F.cross_entropy(logits.reshape(-1, v),
                                      xb[:, 1:].contiguous().reshape(-1),
                                      reduction="none").reshape(xb.shape[0], T - 1)
                tot += nll.sum(0)
        mpos = (tot / oracle_x.shape[0]).cpu().numpy()
        return {int(k): float((mpos - bayes_pos)[lvl_of_arrival == k].mean())
                for k in np.unique(lvl_of_arrival)}, mpos.tolist()

    # ---------------- the arm loop ----------------
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    setup = {"config": {"v": v, "s": s, "L": L, "m": m, "rule_seed": rule_seed,
                        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
                        "max_steps": max_steps, "batch_size": batch_size, "lr": lr,
                        "weight_decay": weight_decay, "data_seed": data_seed, "seed": seed,
                        "fresh_every": fresh_every, "T": T, "thresh_frac": thresh_frac,
                        "n_eval_sequences": n_eval_sequences, "n_oracle": n_oracle,
                        "probe_blocks": pblocks},
             "arms": [a[0] for a in arm_list], "ckpts": ckpts, "ceiling": ceiling,
             "bayes_by_level": bayes_by_level, "bayes_pos": bayes_pos.tolist(),
             "cr_ref": CR_REF, "cr_ref_step": CR_REF_STEP}
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump(setup, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, kind, n_seqs in arm_list:
        print(f"\n===== arm {label} ({kind}, {n_seqs} sequences = "
              f"{n_seqs * T:,} tokens) =====", flush=True)
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed)

        if kind == "frozen":
            # DRAWN ONCE. The identical tokens, in the identical concatenation, for the
            # whole budget. `data_seed` is `conditional_revision`'s, so `frozen_200000`
            # IS its Gate-0 base corpus.
            cseqs, _, _ = _generate_with_traces(rules, n_seqs, data_seed)
            corpus = torch.from_numpy(cseqs.astype(np.int64)).reshape(-1)
            corpus_tokens = int(corpus.shape[0])
            csha = hashlib.sha1(cseqs.astype(np.int64).tobytes()).hexdigest()[:16]
            print(f"  frozen corpus sha={csha} tokens={corpus_tokens:,}", flush=True)
        else:
            corpus, corpus_tokens, csha = None, 0, "-"

        log = []
        fresh_pool_tokens = 0
        for step in range(max_steps + 1):
            if step in ckpts:
                model.eval()
                full = step in (CR_REF_STEP, max_steps)
                blocks = all_blocks if full else pblocks
                lv = per_level(model, blocks, full)
                exc, mpos = excess_over_bayes(model)
                rec = {"step": step, "tokens": step * batch_size * T,
                       "epochs": (step * batch_size * T / corpus_tokens
                                  if corpus_tokens else None),
                       "levels": lv, "full_probe": full,
                       "val_nll": flat_nll(model, val_corpus, gseed=eval_seed + 5),
                       "train_nll": (flat_nll(model, corpus, gseed=seed + 5)
                                     if corpus is not None else None),
                       "excess_by_level": exc, "nll_pos": mpos}
                log.append(rec)
                print(f"[{label:16s} s{step:6d} tok {rec['tokens'] / 1e6:6.2f}M "
                      f"ep {('%.1f' % rec['epochs']) if rec['epochs'] else '  inf':>6s}] "
                      f"val {rec['val_nll']:.4f} "
                      f"train {('%.4f' % rec['train_nll']) if corpus is not None else '  -  '} "
                      f"levels " + " ".join(f"{k}={lv[k]:.3f}" for k in
                                            sorted(lv, key=lambda z: int(z[1:])))
                      + " excess {" + " ".join(f"{k}:{x:+.3f}" for k, x in sorted(exc.items()))
                      + "}", flush=True)
                with open(os.path.join(outdir, f"{label}.json"), "w") as fh:
                    json.dump({"arm": label, "kind": kind, "n_seqs": n_seqs,
                               "corpus_tokens": corpus_tokens, "corpus_sha": csha,
                               "log": log, "complete": step == max_steps},
                              fh, indent=2, cls=NumpyEncoder)
                volume.commit()
                model.train()
            if step == max_steps:
                break

            if kind == "fresh" and step % fresh_every == 0:
                # a genuinely new draw, sized so the pool is consumed about ONCE before
                # the next redraw: batch*T tokens per step * fresh_every steps
                need = max(64, batch_size * fresh_every)
                fseqs, _, _ = _generate_with_traces(rules, need,
                                                    data_seed + 100_003 * (step // fresh_every + 1))
                corpus = torch.from_numpy(fseqs.astype(np.int64)).reshape(-1)
                fresh_pool_tokens += int(corpus.shape[0])

            n = corpus.shape[0]
            ix = torch.randint(0, n - T - 1, (batch_size,), generator=gen)
            idx = ix[:, None] + arangeT[None, :]
            x, y = corpus[idx].to(device), corpus[idx + 1].to(device)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % log_interval == 0:
                print(f"  {label} {step:6d}  ntp {loss.item():.4f}", flush=True)

        results[label] = log
        if kind == "fresh":
            print(f"  {label}: {fresh_pool_tokens:,} distinct tokens drawn for "
                  f"{max_steps * batch_size * T:,} consumed", flush=True)

    # ---------------- the fidelity gate, printed where it cannot be missed -----------
    fid = None
    for label, kind, n_seqs in arm_list:
        if label == "frozen_200000":
            for r in results[label]:
                if r["step"] == CR_REF_STEP:
                    fid = {k: r["levels"].get(k) for k in CR_REF}
    print(f"\n{'=' * 78}\nFIDELITY GATE (frozen_200000 @ step {CR_REF_STEP} IS "
          f"conditional_revision's Gate-0 base)")
    print(f"  published : {CR_REF}")
    print(f"  measured  : {fid}")
    if fid and all(fid.get(k) is not None for k in CR_REF):
        print("  delta     : " + " ".join(f"{k}={fid[k] - CR_REF[k]:+.3f}" for k in CR_REF))
    print(f"{'=' * 78}", flush=True)

    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"elapsed": time.time() - started, "fidelity": fid}


# --------------------------------------------------------------------------- #
# structural gate (CPU): the DGP, the frozen corpus, and the oracle self-check
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=3600, memory=16384)
def gate(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
         data_seed: int = 7, n_oracle: int = 64):
    """C-F  the frozen corpus is byte-identical on redraw at the same seed, and a `fresh`
            draw at a different seed is not.
       C-O  `oracle.self_check` — E[B_D] == E[H_tot − H_irr_D] — so the BP reference this
            node reads its Bayes floor from is the same instrument `conditional_revision`
            validated.
       C-C  the exact per-level ceiling P(z_l | x_1..T): the denominator every recovery
            number is charted against, and the reason a level that never rises can be
            distinguished from a probe that cannot read it."""
    import hashlib
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision.oracle import prefix_beliefs, revision_and_entropy, self_check

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    out = {"setting": setting_key(v, s, L, m), "T": T}

    a, _, _ = _generate_with_traces(rules, 2048, data_seed)
    b, _, _ = _generate_with_traces(rules, 2048, data_seed)
    c, _, _ = _generate_with_traces(rules, 2048, data_seed + 100_003)
    sha = lambda z: hashlib.sha1(z.astype(np.int64).tobytes()).hexdigest()[:16]
    out["CF_frozen"] = {"sha": sha(a), "redraw_identical": bool(sha(a) == sha(b)),
                        "fresh_differs": bool(sha(a) != sha(c))}
    assert out["CF_frozen"]["redraw_identical"] and out["CF_frozen"]["fresh_differs"], "C-F"

    sq, lf, _ = _generate_with_traces(rules, n_oracle, 999)
    res = revision_and_entropy(rules, sq, lf, Ds=[0, 2], chunk=64, verbose=False)
    out["CO_oracle_self_check"] = self_check(res, tol=0.05)
    out["CO_bayes_surprisal_mean"] = float(res["surprisal"].mean())

    nodes, _, _ = prefix_beliefs(rules, sq, T, L - 1)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    out["CC_ceiling"] = {f"d{L - ell}": float(nodes[ell][:, last_anc[ell], :].max(-1).mean())
                         for ell in range(L)}
    # what a corpus of C sequences holds, in tokens, against the budgets the arms consume
    out["corpus_tokens"] = {str(n): n * T for n in (2048, 16384, 200000)}
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    lm_reread.remote(quick=quick, tag="smoke_local")
