"""Q2: the corpus wall from the inside.

`practice/reread/lm`'s `lm0` found extraction strictly level-ordered, a 6.4x re-read
corpus fully renewable, and a hard corpus-size wall: the small-corpus arms stall at the
next level and never move, while their memorisation gap grows. To a DERIVATIVE, a level
that stalls because the archive is exhausted goes as quiet as a level that was absorbed
(`ideas/calibration_and_violation_are_one_object.md` §10.5). That node logged probes,
`flat_nll` and the ORACLE-referenced excess per arrival level, but not the model's own
`H(q)`, and it saved no checkpoints. This one re-runs its arms with the endogenous panel.

WHAT IS RE-RUN VERBATIM. The training loop is `lm_reread`'s, itself
`conditional_revision.gate0`'s base loop: same GPT(8L/8H/256D), same AdamW(3e-4, wd 0.01),
batch 64, the same flat-window sampler `corpus[ix + arange(T)]` off a concatenated corpus,
the same `torch.manual_seed(seed)` before `GPT(...)` and the same dedicated
`torch.Generator().manual_seed(seed)` for `ix`. The model has `Dropout(0.0)` and every
readout runs under `no_grad` with its own generator, so nothing the panel does touches the
training RNG stream and the trajectory is `lm0`'s replayed. The arms differ ONLY in
whether the corpus tensor is redrawn.

THE FIDELITY GATE is `lm0` itself: `val_nll`, `train_nll` and the oracle-referenced
`excess_by_level` are recomputed exactly as `lm_reread` computes them (same held-out
corpus, same generator seeds, same 512 aligned oracle sequences) and compared against
`lm0`'s logged values at the 12 checkpoints it shares. That is a tighter check than the
`conditional_revision` Gate-0 anchor for this purpose -- it pins the loop against the run
being extended -- and it costs nothing. `lm0`'s probe ladder (d1..d6) therefore transfers
to these arms unchanged, which is why the probes are not re-run here (they were the
expensive instrument and the panel does not use them).

THE PANEL, per checkpoint, on two venues:

  V  held-out FRESH windows: `lm_reread`'s own held-out val corpus, 2048 windows of T+1
     tokens at a fixed generator seed, shared by every arm and checkpoint. For a frozen
     arm this is OFF its training distribution.
  C  the arm's OWN corpus windows (for `fresh`, the pool currently loaded). This is the
     distribution the identity of altitude Q1 applies to.

  per level on each venue:  H(q), NLL, excess = NLL - H(q)
  memorisation gap per level:  NLL_V - NLL_C
  the period detector (nested and unconstrained) on each venue
  the oracle residual `KL(p_L || q)` by level, on V -- the ground truth

LEVEL LABELS ON A REREAD CORPUS. `lm_reread`'s corpus is a concatenation of ALIGNED
length-T sequences, so corpus index i carries leaf index `i % T` and a window drawn at
offset `ix` has phase `ix % T`. The phase is therefore exact and free on both venues; no
new instrument is needed. (The oracle's `p_L` on venue V is `flat_oracle.flat_predictive`
with `rule_w=None`, the uniform-synonym generator `lm_reread` uses.)

Run from experiments/:
  modal run -m rhm.logit_reading.frontier.wall::build_refs --n-panel 256 --tag smoke
  modal run -m rhm.logit_reading.frontier.wall::wall_sweep --tag smoke --quick
  modal run --detach -m rhm.logit_reading.frontier.wall::wall_sweep --tag w0
"""

import json
import os
import time

import modal
import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key
from rhm.logit_reading.frontier.common import (
    _ignore, log_softmax, ent, xent, detrend, leaf_templates, detector_block,
    endo_levels, oracle_levels, level_panel, confusion)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-logit-frontier-wall", image=image)

OUT = "/data/v16_s2_L6_m4_distinct/logit_reading/frontier"
LM0 = "/data/rhm_practice_reread_lm/lm0"
# `lm0`'s ladder, kept verbatim so the fidelity comparison is exact
LM0_CKPTS = [250, 387, 601, 931, 1443, 2236, 3466, 5372, 8326, 12000, 12904, 20000]
PANEL_GEN_SEED = 20260917


def ladder(max_steps, n_extra=21):
    """`lm0`'s log-spaced ladder, densified, with step 0 (the honest floor) added.
    Denser around both arms' turns: `frozen_2048`'s val NLL bottoms between steps 931 and
    2236 in `lm0`, `frozen_16384`'s between 3466 and 8326, so both stalls sit inside the
    ladder with several checkpoints on each side."""
    base = np.unique(np.round(np.geomspace(250, max_steps, n_extra)).astype(int)).tolist()
    pts = {0, int(max_steps)} | set(base) | {c for c in LM0_CKPTS if c <= max_steps}
    return sorted(int(x) for x in pts)


# --------------------------------------------------------------------------- #
# the shared, model-independent references: computed once, read by every arm
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=32768)
def build_refs(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
               n_eval_sequences: int = 8000, eval_seed: int = 999,
               n_oracle: int = 512, oracle_seed: int = 999,
               n_panel: int = 2048, chunk: int = 8, n_phase: int = 512,
               tag: str = "w0"):
    """Everything model-independent the arms need, computed once:
      - the held-out val corpus `lm_reread` builds, and the fixed venue-V panel windows
        drawn from it (with their exact phase)
      - every observer's predictive on those windows: `p_L` for the oracle residual and
        `H_k` for the detector templates
      - the aligned-oracle Bayes surprisal per position (`lm_reread`'s `excess_over_bayes`
        floor) and the 512 oracle sequences it is measured on
      - observer k's OWN posterior over the window phase mod s^k: the ceiling altitude
        Q2 A charts the period detector against
    """
    import resource
    import torch
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision.oracle import revision_and_entropy
    from rhm.logit_reading.flat_oracle import flat_predictive

    volume.reload()
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    t0 = time.time()

    # --- `lm_reread`'s held-out val corpus, verbatim ---
    val_seqs, _, _ = _generate_with_traces(rules, max(2000, n_eval_sequences // 2),
                                           eval_seed + 1)
    val_corpus = val_seqs.astype(np.int64).reshape(-1)
    g = torch.Generator().manual_seed(PANEL_GEN_SEED)
    ix = torch.randint(0, len(val_corpus) - T - 2, (n_panel,), generator=g).numpy()
    windows = val_corpus[ix[:, None] + np.arange(T + 1)[None, :]]
    phase = (ix % T).astype(np.int64)
    print(f"[refs] val corpus {len(val_corpus):,} tokens; {n_panel} panel windows; "
          f"phase hist head {np.bincount(phase, minlength=T)[:8].tolist()}", flush=True)

    # --- every observer's predictive on the panel windows ---
    H_k = np.empty((n_panel, T, L + 1), dtype=np.float32)
    p_L = None
    for k in range(L + 1):
        tk = time.time()
        pr = flat_predictive(windows, rules, k, device="cuda",
                             chunk=(chunk if k == L else max(chunk, 64)),
                             rule_w=None).numpy()[:, 1:, :]
        H_k[:, :, k] = (-(pr * np.log(np.clip(pr, 1e-300, None))).sum(-1)).astype(np.float32)
        if k == L:
            p_L = pr.astype(np.float32)
        print(f"  observer k={k}  {time.time() - tk:.1f}s", flush=True)

    # --- the aligned-oracle Bayes floor `lm_reread`'s excess_over_bayes subtracts ---
    osq, olf, _ = _generate_with_traces(rules, n_oracle, oracle_seed)
    ores = revision_and_entropy(rules, osq, olf, Ds=[0], chunk=256, verbose=False)
    bayes_pos = ores["surprisal"].mean(0).astype(np.float64)
    print(f"[refs] Bayes surprisal mean {bayes_pos.mean():.4f}", flush=True)

    # --- the observer's own phase-posterior ceiling (altitude Q2 A's ceiling row) ---
    ceiling = {}
    for k in range(1, L + 1):
        S = s ** k
        acc = np.zeros((n_phase, T + 1))
        for c0 in range(0, n_phase, 4):
            w = windows[c0:c0 + 4]
            _, logpi, _ = flat_predictive(w, rules, k, device="cuda", chunk=4,
                                          return_phase=True, rule_w=None)
            lp = logpi.numpy()
            acc[c0:c0 + len(w)] = (lp.argmax(-1) == (phase[c0:c0 + len(w)] % S)[:, None])
        ceiling[k] = float(acc[:, 32:65].mean())
        print(f"  ceiling k={k}  {ceiling[k]:.3f}  (chance {1 / S:.3f})", flush=True)

    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/refs_{tag}.npz"
    np.savez_compressed(dst, windows=windows.astype(np.int64), phase=phase, p_L=p_L,
                        H_k=H_k, val_corpus=val_corpus.astype(np.int16),
                        bayes_pos=bayes_pos, oracle_seqs=osq.astype(np.int64),
                        ceiling=np.array([ceiling[k] for k in range(1, L + 1)]))
    meta = {"config": {"v": v, "s": s, "L": L, "m": m, "rule_seed": rule_seed,
                       "n_eval_sequences": n_eval_sequences, "eval_seed": eval_seed,
                       "n_oracle": n_oracle, "oracle_seed": oracle_seed,
                       "n_panel": n_panel, "panel_gen_seed": PANEL_GEN_SEED, "T": T},
            "ceiling_phase_posterior": ceiling,
            "bayes_pos_mean": float(bayes_pos.mean()),
            "elapsed": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    with open(f"{OUT}/refs_{tag}.json", "w") as f:
        json.dump(meta, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"[refs] {json.dumps(meta['ceiling_phase_posterior'])}", flush=True)
    print(f"[refs] done in {meta['elapsed']:.0f}s  peak RSS "
          f"{meta['peak_rss_gb']:.2f} GB -> {dst}", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=8 * 3600, memory=12288,
              max_containers=6)
def wall_arm(arm: str, tag: str = "w0", max_steps: int = 20000, batch_size: int = 64,
             lr: float = 3e-4, weight_decay: float = 0.01, data_seed: int = 7,
             seed: int = 42, fresh_every: int = 500, eval_seed: int = 999,
             v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
             n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
             n_panel: int = 2048, j_lo: int = 0, save_ckpts: str = "",
             log_interval: int = 2000, quick: bool = False):
    import hashlib
    import resource
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.reread.lm.lm_reread import parse_arms

    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    (label, kind, n_seqs), = parse_arms(arm)

    R = np.load(f"{OUT}/refs_{tag}.npz")
    windows = R["windows"]
    phase = R["phase"]
    p_L = R["p_L"].astype(np.float64)
    H_k = R["H_k"].astype(np.float64)
    val_corpus = torch.from_numpy(R["val_corpus"].astype(np.int64))
    bayes_pos = R["bayes_pos"]
    oracle_x = torch.from_numpy(R["oracle_seqs"]).to(device)
    if n_panel and n_panel < windows.shape[0]:
        windows, phase, p_L, H_k = (windows[:n_panel], phase[:n_panel],
                                    p_L[:n_panel], H_k[:n_panel])
    if quick:
        max_steps, fresh_every = 600, 100
        windows, phase, p_L, H_k = (windows[:128], phase[:128], p_L[:128], H_k[:128])

    ckpts = ladder(max_steps) if not quick else [0, 250, max_steps]
    save_steps = ({int(x) for x in save_ckpts.split(",") if x.strip()} if save_ckpts
                  else ({0, int(max_steps)} | {c for c in LM0_CKPTS if c <= max_steps}))
    arangeT = torch.arange(T)
    arangeT1 = np.arange(T)

    # oracle labels + detector templates on venue V (fixed, shared by every checkpoint)
    leaf_V = (phase[:, None] + 1 + arangeT1[None, :]) % (s ** L)
    lev_V = oracle_levels(phase, T, L, s)
    tmpls = leaf_templates(H_k, leaf_V, L, s)
    H_pL = ent(p_L)
    pos_top_level = np.array(
        [min(l for l in range(L + 1) if (p + 1) % (s ** (L - l)) == 0) for p in range(T)],
        dtype=np.int64)
    lvl_of_arrival = pos_top_level[1:]

    outdir = f"{OUT}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    started = time.time()
    print(f"arm {label} ({kind}, {n_seqs} seqs)  ckpts {ckpts}\n  save at "
          f"{sorted(save_steps)}  panel {windows.shape[0]} windows", flush=True)

    # ---------------- the model and the corpus: `lm_reread`'s loop, verbatim -----------
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    if kind == "frozen":
        cseqs, _, _ = _generate_with_traces(rules, n_seqs, data_seed)
        corpus = torch.from_numpy(cseqs.astype(np.int64)).reshape(-1)
        corpus_tokens = int(corpus.shape[0])
        csha = hashlib.sha1(cseqs.astype(np.int64).tobytes()).hexdigest()[:16]
        print(f"  frozen corpus sha={csha} tokens={corpus_tokens:,}", flush=True)
    else:
        corpus, corpus_tokens, csha = None, 0, "-"

    # ---------------- readouts ----------------
    def logits_on(x_int64):
        with torch.no_grad():
            out = []
            for i in range(0, x_int64.shape[0], 512):
                out.append(model(x_int64[i:i + 512])[0].double().cpu())
            return torch.cat(out).numpy()

    def venue_panel(win, ph, with_oracle):
        """The endogenous panel on one window population (n, T+1) with known phase."""
        z = logits_on(torch.from_numpy(np.ascontiguousarray(win[:, :T])).to(device))
        logq = log_softmax(z)
        H_q = ent(np.exp(logq))
        NLL = -np.take_along_axis(logq, win[:, 1:][..., None], -1)[..., 0]
        lev = oracle_levels(ph, T, L, s)
        CE = KL = None
        if with_oracle:
            CE = xent(p_L, logq)
            KL = CE - H_pL
        d = {"overall": {"H_q": float(H_q.mean()), "NLL": float(NLL.mean()),
                         "excess_realised": float((NLL - H_q).mean())},
             "by_level_oracle": level_panel(lev, L, H_q, NLL, CE, KL,
                                            H_pL if with_oracle else None)}
        if with_oracle:
            d["overall"]["CE"] = float(CE.mean())
            d["overall"]["KL_pL_q"] = float(KL.mean())
            d["overall"]["excess_oracle"] = float((CE - H_q).mean())
        det = detector_block(detrend(H_q), tmpls, ph, L, s, j_lo)
        d["detector"] = {k: det[k] for k in ("acc_unconstrained_template",
                                             "acc_unconstrained_meanprof", "chance")}
        d["by_level_endo"] = {}
        d["confusion"] = {}
        for mode in ("template", "meanprof"):
            elev = endo_levels(det[mode]["r_hat_L"], T, L, s)
            d["by_level_endo"][mode] = level_panel(elev, L, H_q, NLL, CE, KL, None)
            d["confusion"][mode] = confusion(lev, elev, L)
            d["detector"][mode] = {"acc_nested": det[mode]["acc_nested"],
                                   "frac_exact_field": float((elev == lev).mean())}
        # per-level NLL keyed by leaf level, kept raw so the reduction can difference it
        return d

    def flat_nll(corp, n_batches=20, gseed=0):
        """`lm_reread.flat_nll`, verbatim -- the fidelity column."""
        gg = torch.Generator().manual_seed(gseed)
        n = corp.shape[0]
        tot = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                ixb = torch.randint(0, n - T - 1, (batch_size,), generator=gg)
                idx = ixb[:, None] + arangeT[None, :]
                _, loss = model(corp[idx].to(device), corp[idx + 1].to(device))
                tot += loss.item()
        return tot / n_batches

    def excess_over_bayes():
        """`lm_reread.excess_over_bayes`, verbatim -- the other fidelity column."""
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
                for k in np.unique(lvl_of_arrival)}

    def corpus_windows(corp, n):
        gg = torch.Generator().manual_seed(PANEL_GEN_SEED + 1)
        ixc = torch.randint(0, corp.shape[0] - T - 2, (n,), generator=gg).numpy()
        cnp = corp.numpy()
        return cnp[ixc[:, None] + np.arange(T + 1)[None, :]], (ixc % T).astype(np.int64)

    # ---------------- the loop ----------------
    log = []
    fresh_pool_tokens = 0
    for step in range(max_steps + 1):
        if step in ckpts:
            model.eval()
            t0 = time.time()
            rec = {"step": step, "tokens": step * batch_size * T,
                   "epochs": (step * batch_size * T / corpus_tokens
                              if corpus_tokens else None),
                   "val_nll": flat_nll(val_corpus, gseed=eval_seed + 5),
                   "train_nll": (flat_nll(corpus, gseed=seed + 5)
                                 if corpus is not None else None),
                   "excess_by_level": excess_over_bayes(),
                   "venue_V": venue_panel(windows, phase, True)}
            if corpus is not None:
                cw, cph = corpus_windows(corpus, windows.shape[0])
                rec["venue_C"] = venue_panel(cw, cph, False)
            rec["panel_secs"] = time.time() - t0
            log.append(rec)
            vV = rec["venue_V"]
            print(f"[{label:14s} s{step:6d} tok {rec['tokens'] / 1e6:6.2f}M] "
                  f"val {rec['val_nll']:.4f} "
                  f"train {('%.4f' % rec['train_nll']) if corpus is not None else '  -   '} "
                  f"| V: NLL {vV['overall']['NLL']:.4f} Hq {vV['overall']['H_q']:.4f} "
                  f"exc {vV['overall']['excess_realised']:+.4f} "
                  f"KL {vV['overall']['KL_pL_q']:.4f} "
                  f"nest " + "".join(f"{vV['detector']['template']['acc_nested'][k]:.2f} "
                                     for k in range(1, L + 1))
                  + f"({rec['panel_secs']:.0f}s)", flush=True)
            with open(f"{outdir}/{label}.json", "w") as fh:
                json.dump({"arm": label, "kind": kind, "n_seqs": n_seqs,
                           "corpus_tokens": corpus_tokens, "corpus_sha": csha,
                           "ckpts": ckpts, "log": log,
                           "complete": step == max_steps}, fh, cls=NumpyEncoder, indent=1)
            if step in save_steps and not quick:
                torch.save({"model": model.state_dict(), "step": step,
                            "config": {"v": v, "s": s, "L": L, "m": m,
                                       "rule_seed": rule_seed, "n_layer": n_layer,
                                       "n_head": n_head, "n_embd": n_embd, "arm": label,
                                       "n_seqs": n_seqs, "seed": seed,
                                       "data_seed": data_seed}},
                           f"{outdir}/{label}_step{step:06d}.pt")
            volume.commit()
            model.train()
        if step == max_steps:
            break

        if kind == "fresh" and step % fresh_every == 0:
            need = max(64, batch_size * fresh_every)
            fseqs, _, _ = _generate_with_traces(
                rules, need, data_seed + 100_003 * (step // fresh_every + 1))
            corpus = torch.from_numpy(fseqs.astype(np.int64)).reshape(-1)
            fresh_pool_tokens += int(corpus.shape[0])

        n = corpus.shape[0]
        ix2 = torch.randint(0, n - T - 1, (batch_size,), generator=gen)
        idx = ix2[:, None] + arangeT[None, :]
        x, y = corpus[idx].to(device), corpus[idx + 1].to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_interval == 0:
            print(f"  {label} {step:6d}  ntp {loss.item():.4f}", flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"DONE {label} in {time.time() - started:.0f}s  peak RSS {rss:.2f} GB", flush=True)
    volume.commit()
    return {"arm": label, "elapsed": time.time() - started, "peak_rss_gb": rss,
            "n_ckpts": len(log), "final_val_nll": log[-1]["val_nll"]}


# --------------------------------------------------------------------------- #
# the coordinator, and the fidelity gate against lm0
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=10 * 3600, memory=4096)
def wall_sweep(tag: str = "w0",
               arms: str = "fresh,frozen_200000,frozen_16384,frozen_2048",
               max_steps: int = 20000, n_panel: int = 2048, quick: bool = False,
               save_ckpts: str = ""):
    """CPU coordinator: one container per arm (same GPU-hours, 1/n the wall clock)."""
    volume.reload()
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    kw = dict(tag=tag, max_steps=max_steps, n_panel=n_panel, quick=quick,
              save_ckpts=save_ckpts)
    rows = list(wall_arm.map(arm_list, kwargs=kw))
    volume.reload()                     # the arms wrote from their own containers
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    for r in rows:
        print(f"  {r['arm']:14s} {r['elapsed']:6.0f}s  peak RSS {r['peak_rss_gb']:.2f} GB  "
              f"{r['n_ckpts']} ckpts  final val {r['final_val_nll']:.4f}", flush=True)
    gate = fidelity_gate(tag, arm_list)
    with open(f"{OUT}/{tag}/summary.json", "w") as f:
        json.dump({"arms": rows, "fidelity_vs_lm0": gate}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"arms": rows, "fidelity_vs_lm0": gate}


def fidelity_gate(tag, arm_list):
    """Every quantity `lm0` logged that this node recomputes, at the shared checkpoints."""
    out = {}
    for a in arm_list:
        try:
            with open(f"{LM0}/{a}.json") as f:
                ref = {r["step"]: r for r in json.load(f)["log"]}
            with open(f"{OUT}/{tag}/{a}.json") as f:
                new = {r["step"]: r for r in json.load(f)["log"]}
        except FileNotFoundError as e:
            out[a] = {"error": str(e)}
            continue
        shared = sorted(set(ref) & set(new))
        d_val = [abs(new[st]["val_nll"] - ref[st]["val_nll"]) for st in shared]
        d_tr = [abs(new[st]["train_nll"] - ref[st]["train_nll"]) for st in shared
                if ref[st]["train_nll"] is not None and new[st]["train_nll"] is not None]
        d_ex = [abs(new[st]["excess_by_level"][str(k)] - ref[st]["excess_by_level"][str(k)])
                for st in shared for k in range(7)
                if str(k) in ref[st]["excess_by_level"] and str(k) in new[st]["excess_by_level"]]
        out[a] = {"n_shared_ckpts": len(shared),
                  "max_abs_d_val_nll": max(d_val) if d_val else None,
                  "max_abs_d_train_nll": max(d_tr) if d_tr else None,
                  "max_abs_d_excess_by_level": max(d_ex) if d_ex else None,
                  "lm0_final_val": ref[max(shared)]["val_nll"] if shared else None,
                  "new_final_val": new[max(shared)]["val_nll"] if shared else None}
        print(f"  FIDELITY {a}: {out[a]}", flush=True)
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=1800, memory=4096)
def gate_only(tag: str = "w0",
              arms: str = "fresh,frozen_200000,frozen_16384,frozen_2048"):
    volume.reload()
    return fidelity_gate(tag, [a.strip() for a in arms.split(",") if a.strip()])
