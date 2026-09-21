"""shaped: a trunk whose state is shaped by the goal, in place of the frozen NTP trunk.

Every value-side round so far (`striatum/`, `junction/`, `norm/`) read an outcome-trained
critic on a trunk that only ever saw next-token loss.  striatum's "does not establish"
list puts this first: *"Anything about a trunk that has the goal.  The frozen-trunk
readout is the cortex-to-value-system broadcast copy; a trunk fine-tuned on the task,
whose state is shaped by value, is the deferred arm."*

THE SHAPING.  From `traj_a1_s42/step064000.pt`, continue training with the ACTOR's own
objective: 16-way level-l structural query heads (l = 1..6) on the trunk's final residual
stream (`post_block7`, the block striatum's actor reads), trained JOINTLY with the trunk
on fresh clean windows and their parses -- the same query task, the same levels and the
same position range (t >= T_LO = 8) as striatum's actor.  Three arms, matched steps and
matched data (the data stream is a function of `data_seed` only, so the three arms see
bit-identical windows):

  task      the query loss alone.  The next-token head is left alone -- with no NTP loss
            `ln_f` and `lm_head` receive no gradient at all, and AdamW skips a parameter
            whose `.grad` is None, so nothing decays them either.
  task_ntp  query loss + `w_ntp` * next-token loss: the predictive kept alive.
  ntp       next-token loss alone, for the same steps on the same windows -- the control
            that separates "the trunk moved" from "the trunk moved toward the goal".  Its
            query heads are trained on DETACHED states, so they read the trunk without
            shaping it and the shaping-time head accuracy stays comparable across arms.

Checkpoints are written in `train_trajectory`'s format (`{"model": trunk state dict,
"config": {...}}`), so `calibration.load_trajectory_ckpt` rebuilds the grammar from the
config and every downstream cell -- `striatum.task.striatum_ckpt`, `junction.task
.junction_ckpt`, `striatum.addendum.hexcess_ckpt`, `calibration.calibration_ckpt`,
`altitude.identity.identity_sweep` -- reads them unchanged.  The query heads are saved
BESIDE the checkpoint (`heads_stepNNNNNN.pt`), never inside it, so the trunk's state dict
still loads `strict=True` into a plain `GPT`.

Also here, because both are cheap and both belong to "what did shaping cost the
predictive": `probe_ckpt` (Part 2a's per-level violation detection, the surprisal
percentile within token-level x window-position cells, plus a clean next-token CE) and
`ntp_ce` inside the shaping log.

Run:
  # smoke (attached, ~4 min)
  modal run -m rhm.logit_reading.orbitofrontal.shaped.shape::shape_ckpt \
      --src-ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --arm task --steps 300 --ckpt-steps 300 --tag smoke
  # the three arms, one container each
  modal run --detach -m rhm.logit_reading.orbitofrontal.shaped.shape::shape_sweep
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree -- another
    agent writing its launch log there makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-striatum-shaped", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
T_LO = 8                      # striatum's actor position floor
HIDDEN = 256                  # striatum's `train_actor` head width
ARMS = ("task", "task_ntp", "ntp")


# ---------------------------------------------------------------------------
# the query heads (striatum's `train_actor` architecture, trained jointly)
# ---------------------------------------------------------------------------

def make_heads(d, v, device, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.ModuleList([
        nn.Sequential(nn.Linear(d, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, v))
        for _ in LEVELS]).to(device)


def query_batch(rules, rule_w, n, T, seed):
    """`n` fresh clean windows of T+1 tokens plus the level-l answer at every index."""
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    L = len(rules)
    s = rules[0].shape[2]
    wins, phase, _, fs = windows_with_parse(rules, rule_w, n, T + 1, seed)
    y = clean_answers(fs, phase, T + 1, L, s, 3)[:, :, :T]        # (6, n, T) int16
    return wins.astype(np.uint8), y.astype(np.int64)


# ---------------------------------------------------------------------------
# the fine-tune
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=8192,
              max_containers=4)
def shape_ckpt(src_ckpt: str, arm: str = "task", steps: int = 8000,
               ckpt_steps: str = "2000,8000", w_ntp: float = 1.0, lr: float = 3e-4,
               weight_decay: float = 0.01, batch_size: int = 64,
               chunk_windows: int = 16384, chunk_steps: int = 200,
               head_warmup: int = 300, data_seed: int = 31337, seed: int = 42,
               eval_seed: int = 90909, n_eval: int = 1024, log_interval: int = 250,
               tag: str = "s42", out_dir: str = ""):
    import torch
    import torch.nn.functional as F
    volume.reload()
    assert arm in ARMS, arm
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(src_ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    d = model.transformer.wte.weight.shape[1]
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    odir = out_dir or f"{DATA_DIR}/{key}/logit_reading/shape_{arm}_{tag}"
    os.makedirs(odir, exist_ok=True)
    cks = sorted({int(x) for x in ckpt_steps.split(",") if x.strip()})
    print(f"SHAPE arm={arm} src={src_ckpt} step={cfg.get('step')} -> {odir}\n"
          f"  steps={steps} ckpts={cks} lr={lr} w_ntp={w_ntp} bs={batch_size} "
          f"warmup={head_warmup}", flush=True)

    # the checkpoint we write must load back into a plain GPT, so keep the source's
    # config verbatim and only overwrite `step` and the shaping record.
    src_sd = torch.load(src_ckpt, map_location="cpu")
    base_cfg = dict(src_sd["config"])
    shaping = dict(src_ckpt=src_ckpt, src_step=base_cfg.get("step"), arm=arm,
                   shape_steps=steps, w_ntp=w_ntp, lr=lr, weight_decay=weight_decay,
                   batch_size=batch_size, chunk_windows=chunk_windows,
                   chunk_steps=chunk_steps, head_warmup=head_warmup,
                   data_seed=data_seed, seed=seed, t_lo=T_LO, hidden=HIDDEN,
                   levels=LEVELS, block="post_block7")

    heads = make_heads(d, v, dev, seed=seed)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # ---- held-out evaluation set (fixed, never in the training stream) -------
    we, ye = query_batch(rules, rule_w, n_eval, T, eval_seed)
    Xe = torch.as_tensor(we[:, :T].astype(np.int64), device=dev)
    Ye = torch.as_tensor(ye, device=dev)                       # (6, n_eval, T)
    Ne = torch.as_tensor(we[:, 1:T + 1].astype(np.int64), device=dev)
    pos = torch.arange(T_LO, T, device=dev)

    def evaluate():
        model.eval(); heads.eval()
        accs, ce = [], 0.0
        with torch.no_grad():
            for c0 in range(0, len(Xe), 256):
                sl = slice(c0, min(c0 + 256, len(Xe)))
                lg, _, inter = model(Xe[sl], return_intermediates=True)
                ce += float(F.cross_entropy(lg.reshape(-1, v), Ne[sl].reshape(-1),
                                            reduction="sum"))
                h = inter["post_block7"][:, pos]
                accs.append(torch.stack([
                    (heads[li](h).argmax(-1) == Ye[li][sl][:, pos]).float().mean(1)
                    for li in range(len(LEVELS))], 0).cpu().numpy())
        model.train(); heads.train()
        return (np.concatenate(accs, 1).mean(1).tolist(),
                ce / (len(Xe) * T))

    a0, ce0 = evaluate()
    print(f"  [pre]  ntp_ce {ce0:.4f}  head acc "
          f"{' '.join(f'{x:.3f}' for x in a0)}", flush=True)

    def save(step):
        model.eval()
        torch.save({"model": {k: t.detach().cpu() for k, t in model.state_dict().items()},
                    "config": {**base_cfg, "step": step, "shaping": shaping}},
                   f"{odir}/step{step:06d}.pt")
        # NOT `step*_heads.pt`: `calibration.calibration_sweep` globs `step*.pt` in a
        # trajectory dir and would try to load the head file as a trunk checkpoint.
        torch.save({"heads": {k: t.detach().cpu() for k, t in heads.state_dict().items()},
                    "config": {"hidden": HIDDEN, "levels": LEVELS, "block": "post_block7",
                               "d": d, "v": v, "step": step, **shaping}},
                   f"{odir}/heads_step{step:06d}.pt")
        volume.commit()
        model.train()

    # ---- the loop ------------------------------------------------------------
    g = torch.Generator().manual_seed(seed + 5)
    log = [{"step": 0, "head_acc": a0, "ntp_ce": ce0, "elapsed_s": 0.0}]
    Wc = Yc = None
    run_ntp = arm in ("task_ntp", "ntp")
    run_task_on_trunk = arm in ("task", "task_ntp")
    losses = []
    for step in range(steps + 1):
        if step in cks:
            save(step)
        if step == steps:
            break
        if step % chunk_steps == 0:
            t0 = time.time()
            wc, yc = query_batch(rules, rule_w, chunk_windows, T,
                                 data_seed + 1000 * (step // chunk_steps))
            Wc = torch.as_tensor(wc.astype(np.int64), device=dev)
            Yc = torch.as_tensor(yc, device=dev)
            if step == 0:
                print(f"  chunk ({chunk_windows} windows) {time.time() - t0:.1f}s",
                      flush=True)
        ix = torch.randint(0, Wc.shape[0], (batch_size,), generator=g).to(dev)
        x = Wc[ix][:, :T]
        lg, _, inter = model(x, return_intermediates=True)
        detach = (arm == "ntp") or (step < head_warmup)
        h = inter["post_block7"][:, pos]
        h = h.detach() if detach else h
        q = 0.0
        for li in range(len(LEVELS)):
            q = q + F.cross_entropy(heads[li](h).reshape(-1, v),
                                    Yc[li][ix][:, pos].reshape(-1))
        q = q / len(LEVELS)
        loss = q
        if run_ntp:
            n_loss = F.cross_entropy(lg.reshape(-1, v), Wc[ix][:, 1:T + 1].reshape(-1))
            loss = (q + w_ntp * n_loss) if run_task_on_trunk else (q + n_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append([float(q), float(n_loss) if run_ntp else float("nan")])
        if step % log_interval == 0 and step:
            acc, ce = evaluate()
            rec = {"step": step, "head_acc": acc, "ntp_ce": ce,
                   "train_query_ce": float(np.mean([a[0] for a in losses[-log_interval:]])),
                   "train_ntp_ce": float(np.mean([a[1] for a in losses[-log_interval:]])),
                   "elapsed_s": time.time() - t00}
            log.append(rec)
            print(f"  step {step:6d}  q {rec['train_query_ce']:.4f}  "
                  f"ntp {rec['train_ntp_ce']:.4f}  eval_ce {ce:.4f}  acc "
                  f"{' '.join(f'{x:.3f}' for x in acc)}  {rec['elapsed_s']:.0f}s", flush=True)

    with open(f"{odir}/shape_log.json", "w") as f:
        json.dump({"config": shaping, "base_config": base_cfg, "ckpt_steps": cks,
                   "log": log}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"done {time.time() - t00:.0f}s  peak RSS {rss:.2f} GB -> {odir}", flush=True)
    return {"arm": arm, "dir": odir, "final": log[-1], "peak_rss_gb": rss}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def shape_sweep(src_ckpt: str = "", steps: int = 8000, ckpt_steps: str = "2000,8000",
                w_ntp: float = 1.0, arms: str = "task,task_ntp,ntp", tag: str = "s42",
                chunk_windows: int = 16384, head_warmup: int = 300):
    """CPU coordinator: one container per arm, matched steps and matched data."""
    volume.reload()
    src = src_ckpt or f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt"
    assert os.path.exists(src), src
    args = [(src, a, steps, ckpt_steps, w_ntp, 3e-4, 0.01, 64, chunk_windows, 200,
             head_warmup, 31337, 42, 90909, 1024, 250, tag, "")
            for a in arms.split(",") if a.strip()]
    print(f"{len(args)} arms: {[a[1] for a in args]}", flush=True)
    outs = list(shape_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:600], flush=True)
    return [str(o)[:600] for o in outs]


# ---------------------------------------------------------------------------
# what shaping cost the predictive: Part 2a detection, and a clean next-token CE
# ---------------------------------------------------------------------------

T_BUCKETS = [(1, 16), (16, 32), (32, 48), (48, 64)]


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288,
              max_containers=6)
def probe_ckpt(ckpt: str, stim_tag: str = "a1", n_ctrl: int = 4, seed: int = 0,
               n_clean: int = 2048, clean_seed: int = 77777, tag: str = ""):
    """Part 2a's per-level violation detection on one checkpoint, plus a clean NTP CE.

    The detection statistic is the parent's `detect_auc_stratified`: the violating
    token's model-surprisal percentile among legal control tokens drawn from the same
    (token level x window-position bucket) cell, exactly as `violation.readouts` builds
    it -- same `build_events` control draw, same strata, same percentile convention.
    Written here rather than run through `violation_ckpt` because only this column is
    wanted and the probes in that function are ~all of its cost.
    """
    import torch
    from rhm.logit_reading.flat_oracle import leaf_levels
    from rhm.logit_reading.violation import build_events
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    T = model.block_size
    L, s = cfg["L"], cfg["s"]
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    S = np.load(f"{DATA_DIR}/{key}/logit_reading/stimuli_{stim_tag}.npz")
    We = S["windows_edit"]
    n = len(We)
    rng = np.random.default_rng(seed)
    ev_w, ev_t, ev_kind, ev_k = build_events(S, rng, n_ctrl)
    order = np.argsort(ev_w, kind="stable")
    ev_w, ev_t, ev_kind, ev_k = ev_w[order], ev_t[order], ev_kind[order], ev_k[order]

    nll = np.zeros((n, T), np.float32)
    with torch.no_grad():
        for c0 in range(0, n, 256):
            sl = slice(c0, min(c0 + 256, n))
            x = torch.as_tensor(We[sl, :T], device="cuda")
            lg, _ = model(x)
            lsm = torch.log_softmax(lg.float(), -1)
            nxt = torch.as_tensor(We[sl, 1:T + 1], device="cuda")
            nll[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
    # surprisal of the token AT index t is read from the prediction made at t-1
    surpr = nll[ev_w, ev_t - 1]
    level = leaf_levels((S["phase"][ev_w] + ev_t) % T, s, L)
    tb = np.digitize(ev_t, [b[0] for b in T_BUCKETS[1:]])
    y = ev_kind == 0
    lvl_tb = level * 10 + tb
    pct = np.full(len(y), np.nan)
    ctrl = ~y
    for kk in np.unique(lvl_tb):
        cs = np.sort(surpr[ctrl & (lvl_tb == kk)])
        vi = np.where(y & (lvl_tb == kk))[0]
        if len(cs) and len(vi):
            lo = np.searchsorted(cs, surpr[vi], "left")
            hi = np.searchsorted(cs, surpr[vi], "right")
            pct[vi] = (lo + hi) / 2 / len(cs)
    det = {}
    for k in range(1, L + 1):
        sel = y & (ev_k == k)
        if sel.sum() < 5:
            continue
        det[str(k)] = {"n": int(sel.sum()),
                       "detect_auc_stratified": float(np.nanmean(pct[sel])),
                       "mean_surprisal": float(surpr[sel].mean())}
    det["controls"] = {"mean_surprisal": float(surpr[ctrl].mean()),
                       "rare_mean_surprisal": float(surpr[ev_kind == 1].mean())}

    # clean next-token CE on fresh windows from the same grammar
    from rhm.logit_reading.altitude.units import windows_with_parse
    wc, _, _, _ = windows_with_parse(rules, rule_w, n_clean, T + 1, clean_seed)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for c0 in range(0, n_clean, 256):
            sl = slice(c0, min(c0 + 256, n_clean))
            x = torch.as_tensor(wc[sl, :T].astype(np.int64), device="cuda")
            lg, _ = model(x)
            lsm = torch.log_softmax(lg.float(), -1)
            nxt = torch.as_tensor(wc[sl, 1:T + 1].astype(np.int64), device="cuda")
            tot += float((-lsm.gather(-1, nxt[..., None])[..., 0]).sum())
            cnt += int(x.numel())
    res = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag,
           "shaping": cfg.get("shaping"), "detection_by_kstar": det,
           "clean_ntp_ce": tot / cnt, "n_clean": n_clean,
           "n_events": {"viol": int(y.sum()), "ctrl": int(ctrl.sum())}}
    stem = ckpt[:-3]
    with open(f"{stem}_probe_{stim_tag}" + (f"_{tag}" if tag else "") + ".json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(json.dumps({k: (v if k != "detection_by_kstar" else
                          {kk: round(vv["detect_auc_stratified"], 3)
                           for kk, vv in v.items() if kk != "controls"})
                      for k, v in res.items() if k in ("step", "clean_ntp_ce",
                                                       "detection_by_kstar")}), flush=True)
    return res


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=2048)
def probe_sweep(ckpts: str, stim_tag: str = "a1", n_clean: int = 2048):
    volume.reload()
    ck = [c for c in ckpts.split(",") if c and os.path.exists(c)]
    print(f"{len(ck)} checkpoints", flush=True)
    outs = list(probe_ckpt.starmap([(c, stim_tag, 4, 0, n_clean, 77777, "") for c in ck],
                                   return_exceptions=True))
    for o in outs:
        print(str(o)[:300], flush=True)
    return [str(o)[:300] for o in outs]
