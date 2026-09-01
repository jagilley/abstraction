"""tuning/wledger — the weight-ledger typer check. Offline, on the banked checkpoints.
NO NEW TRAINING.

QUESTION (queued in `README.md` "Next steps", stated in
`ideas/absorption_blinds_the_evaluator.md` §4.3). Gate 1G found the three event types occupy
three disjoint weight subspaces (rotation: wte; burst: ln_f; drift: h1-h2). Gate 1 finding 2
found the *stream* ledger drains at the learner's own absorption rate, so the movement triple
T* — which types all three events perfectly offline — sits at or under floors online. The
predicted-but-unbuilt instrument is a typer read off the ledger absorption *creates* rather
than the one it drains: **per-layer-group update-norm ratios**, which a learner already
computes at train time (per-group grad norms) and which need no counterfactual pass.

This module is the checkpoint-side proxy for that instrument. It computes, for every banked
interval on every banked arm, the per-layer relative weight change

    rel_L(a, b) = || theta_L(b) - theta_L(a) ||_F / || theta_L(a) ||_F

and retains the flat delta for a few groups so a directional read (cosine to a stored
event-type template) can be graded too.

WHAT IS AND IS NOT ONLINE-AVAILABLE, stated up front (see the module docstring of
`analyze_wledger.py` for how each caveat is cashed):

  * A checkpoint-to-checkpoint delta over n steps is || sum_t g_t ||, not sum_t || g_t ||.
    An online per-group grad-norm meter reads the latter. The two agree only for a
    coherent update; the quiet-interval scaling exponent alpha (fit here) measures exactly
    that coherence (0.5 = random walk, 1.0 = fully coherent), so the gap is measured, not
    assumed.
  * The stride quantises lag. The banked grids give:
        rotation   {onset, onset+125}                     (both waves)
        burst      {onset, onset+25}, {onset, onset+125}   (both waves)
        drift      {onset, onset+125} (wave 1);
                   {onset, onset+25} as well (wave 2 added the drift micro-grid)
    So the fastest read is 25 steps for burst/drift and 125 for rotation. The op maps that
    Gate 1 actually ran fire at +50...+125 steps after onset.
  * A ratio between groups is scale-free within an interval, so the read needs no
    counterfactual pass and no oracle — only the arm's own running per-group norms, which
    is the online EWMA analogue of the quiet baseline fit here.

INTERVALS COMPUTED.  Every adjacent-checkpoint pair (the natural trailing-window read), plus
every event-anchored pair (onset, b) with onset < b <= onset + 400 that is banked (this is
what supplies the matched 125-step unit when the micro-grid puts a checkpoint in between,
e.g. bursts at 5000 -> 5025 -> 5125). Each interval carries its own label: the most recent
event onset at or before its left edge, that event's kind, and the lag of the left edge from
that onset. Quiet intervals are labelled by `gate1g.quiet_pairs`' rule verbatim so the null
here is the same object Gate 1G used (with the same documented flaw: it admits absorption
tails, so every event excess is a lower bound).

ARMS. All eight banked arms across three tags:
    g1a   track (never merges, absorbs everything: the primary arm)
          pair_parse (kind="none", rot_period=0 -> ROTATION-BLIND: the event-matched
              negative control for the rotation column -- same world, same steps, same
              bursts and drifts, but the rotating wall token is never in its input)
          self, pair_key (merge at 10050 -> post-merge rotations are not events for them)
    g2a   self_rekey, rekey_dead (never merge), self_v2 (merges at 8050)
    g2c   self_fm (merges at 6025)

Run from experiments/:
  modal run -m rhm.practice.tuning.wledger::deltas --tag wl0
"""

import json
import os

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-practice-tuning-wledger", image=image)

REMOTE = "rhm_practice_tuning_wl"

# the wave-1 / wave-2 schedule (identical in all three tags' setup.json)
ROTS = [8000, 10000, 12000, 14000, 16000, 18000]
BURSTS = [5000, 11000, 17000]
DRIFTS = [6000, 13000, 19000]
EVENTS = ([("rotation", s) for s in ROTS] + [("burst", s) for s in BURSTS]
          + [("drift", s) for s in DRIFTS])
LAG = 125                 # the matched unit every event shares, and the op map's window
BURST_LEN = 250
ANCHOR_MAX = 400          # event-anchored pairs (onset, b) with b - onset <= this

SRCS = ("rhm_practice_tuning_g1/g1a,"
        "rhm_practice_tuning_g2/g2a,"
        "rhm_practice_tuning_g2/g2c")

LAYERS = ["wte", "wpe"] + [f"h{i}" for i in range(8)] + ["ln_f", "lm_head"]
# groups whose flat delta is retained, for the directional (template-cosine) column
DIRGROUPS = ["wte", "wpe", "ln_f", "lm_head", "h0", "h1", "h2"]


def layer_of(k):
    """gate1g.layer_of verbatim -- same grouping, so the two reductions are comparable."""
    if k.startswith("transformer.wte"):
        return "wte"
    if k.startswith("transformer.wpe"):
        return "wpe"
    if k.startswith("transformer.ln_f"):
        return "ln_f"
    if k.startswith("lm_head"):
        return "lm_head"
    if k.startswith("transformer.h."):
        return "h" + k.split(".")[2]
    return "other"


def busy_windows(burst_len=BURST_LEN):
    """gate1g.busy_windows verbatim."""
    return [(e, e + (burst_len if kind == "burst" else LAG)) for kind, e in EVENTS]


def is_quiet(a, b, burst_len=BURST_LEN):
    """gate1g.quiet_pairs' admission rule, per interval."""
    return not any(a < hi and b > lo for lo, hi in busy_windows(burst_len))


def label(a, b):
    """{kind, onset, lag} of the most recent event onset at or before `a`; kind None if the
    interval is admissible as quiet under gate1g's rule."""
    prev = [(k, e) for k, e in EVENTS if e <= a]
    kind, onset = max(prev, key=lambda t: t[1]) if prev else (None, None)
    quiet = is_quiet(a, b) or onset is None
    return {"quiet": bool(quiet),
            "kind": None if quiet else kind,
            "onset": None if quiet else onset,
            "lag": None if quiet else int(a - onset)}


@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=32768, cpu=4.0)
def deltas(tag: str = "wl0", srcs: str = SRCS, arms: str = ""):
    import numpy as np
    import torch

    want = {a.strip() for a in arms.split(",") if a.strip()}
    out = {"lag": LAG, "anchor_max": ANCHOR_MAX, "layers": LAYERS,
           "dirgroups": DIRGROUPS, "events": EVENTS, "arms": {}}

    for src in [s.strip() for s in srcs.split(",") if s.strip()]:
        ck = f"{DATA_DIR}/{src}/ckpt"
        if not os.path.isdir(ck):
            print(f"!! no ckpt dir at {ck}", flush=True)
            continue
        avail = {}
        for f in sorted(os.listdir(ck)):
            if not f.endswith(".pt"):
                continue
            arm, st = f[:-3].rsplit("_s", 1)
            avail.setdefault(arm, {})[int(st)] = os.path.join(ck, f)
        print(f"[{src}] {[(a, len(v)) for a, v in avail.items()]}", flush=True)

        for arm, files in avail.items():
            if want and arm not in want:
                continue
            steps = sorted(files)

            # ---- load every checkpoint of this arm once, flattened per group ----
            # fp16 on disk; cast to fp32 for the arithmetic. ~25 MB/ckpt, <=43 ckpts.
            flat, norm = {}, {}
            order = None
            for st in steps:
                sd = torch.load(files[st], map_location="cpu")["model"]
                if order is None:
                    order = {}
                    for k in sd:
                        order.setdefault(layer_of(k), []).append(k)
                    for L in order:
                        order[L] = sorted(order[L])
                flat[st] = {L: torch.cat([sd[k].reshape(-1).float() for k in order[L]])
                            for L in order}
                norm[st] = {L: float(flat[st][L].norm()) for L in flat[st]}
                del sd
            groups = sorted(order)

            # ---- the interval set: adjacent pairs + event-anchored pairs ----
            pairs = {(steps[i], steps[i + 1]) for i in range(len(steps) - 1)}
            for _k, e in EVENTS:
                if e not in files:
                    continue
                for b in steps:
                    if e < b <= e + ANCHOR_MAX:
                        pairs.add((e, b))
            pairs = sorted(pairs)

            recs, dvec = [], {}
            for a, b in pairs:
                d = {L: flat[b][L] - flat[a][L] for L in groups}
                rec = {"a": a, "b": b, "n": b - a,
                       "abs": {L: float(d[L].norm()) for L in groups},
                       "rel": {L: float(d[L].norm()) / max(norm[a][L], 1e-12)
                               for L in groups},
                       "theta_norm": {L: norm[a][L] for L in groups},
                       **label(a, b)}
                recs.append(rec)
                dvec[(a, b)] = {L: d[L] for L in DIRGROUPS if L in d}
            del flat

            # ---- pairwise cosines between interval deltas, per retained group ----
            # (the directional column: cos to a stored per-type template, and the
            #  interval-to-interval coherence that grades it)
            names = [f"{a}_{b}" for a, b in pairs]
            cos = {}
            for i, (a, b) in enumerate(pairs):
                for j in range(i + 1, len(pairs)):
                    c, dd = pairs[j]
                    key = f"{names[i]}|{names[j]}"
                    cos[key] = {}
                    for L in DIRGROUPS:
                        u, w = dvec[(a, b)].get(L), dvec[(c, dd)].get(L)
                        if u is None or w is None:
                            continue
                        cos[key][L] = float(
                            torch.dot(u, w) / (u.norm() * w.norm() + 1e-12))
            # random-vector floor at the matched per-group dimension
            g = torch.Generator().manual_seed(0)
            floor = {}
            for L in DIRGROUPS:
                nd = next((v[L].numel() for v in dvec.values() if L in v), None)
                if nd is None:
                    continue
                cs = [float(torch.dot(u := torch.randn(nd, generator=g),
                                      w := torch.randn(nd, generator=g))
                            / (u.norm() * w.norm())) for _ in range(64)]
                floor[L] = {"d": int(nd), "analytic": float(1.0 / np.sqrt(nd)),
                            "measured_p99": float(np.percentile(np.abs(cs), 99))}
            del dvec

            out["arms"][f"{src.split('/')[-1]}:{arm}"] = {
                "src": src, "arm": arm, "steps": steps, "groups": groups,
                "intervals": recs, "cos": cos, "cos_floor": floor}
            nq = sum(1 for r in recs if r["quiet"])
            print(f"  [{arm}] {len(recs)} intervals ({nq} quiet), "
                  f"{len(cos)} cos pairs", flush=True)

    od = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "wledger.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder)
    volume.commit()
    print("wrote wledger.json for", list(out["arms"]), flush=True)
    return {"arms": list(out["arms"])}
