"""tuning/gate1g — Gate 1G: offline weight- and representation-side analytics on the
checkpoints banked by Gate 1 wave 1. NO NEW TRAINING.

SPEC: `SPEC.md` + Addendum §4 (checkpoint the readers "so hidden-state geometry reads
(RSA / Procrustes angle) never again need a replay"). This is that read, cashed.

WHAT IS BANKED, AND THE ONE CONSTRAINT IT IMPOSES. `g1a` saved fp16 reader+FM state dicts on
a stride-16 grid plus {onset, +25, +125} at every event, for `self`, `pair_key`, `pair_parse`,
`track` — 39 checkpoints per arm. Wave 1's `cheap` grid had micro-points after bursts and
rotations but NOT after drifts, and rotations got +50/+100 rather than +25, so the
intersection leaves:

    every event      {onset, +125}          <- the one interval length common to all twelve
    bursts also      {onset, +25}
    quiet            stride only, >= 1250 steps apart

So **125 steps is the matched unit** for every cross-event contrast here, and there is NO
quiet 125-step pair to serve as a direct null. The null is therefore built, not found:
`quiet_scaling` fits ||dtheta|| ~ n^alpha over the quiet stride intervals and extrapolates to
n = 125, with alpha reported so the extrapolation is auditable. Cross-event-type contrast at
matched n does the rest of the work.

PART 1 (CPU, on state dicts). Per-layer relative weight change ||dtheta||_F / ||theta||_F over
each event's [onset, onset+125], for every checkpointed arm: where does absorption live?
Then the RE-MAP ALIGNMENT TEST — per-layer cos(dtheta_rot_i, dtheta_rot_j) across the six
rotations on `track` (and on `self` pre-merge). If successive re-maps increasingly share a
subspace, the learner has carved a reusable rotation-class organ, i.e. a native rotate op in
gradient form — which would sharpen why the supplied re-key lost on timing (it was competing
with a specialised organ, not with generic plasticity). Floor: cos of independent random
vectors of the same per-layer dimension, both analytic (1/sqrt(d)) and measured.

PART 2 (GPU, minutes). On the panel's own fixed eval batches, per event type:
  * RSA — RDM over the 16 latent classes at h6 and a mid block, correlated pre vs post.
  * linear alignment — ridge map pre -> post, R^2, plus principal angles between top-k PCA
    subspaces (the Procrustes-style read).
  * class separability — between/within class variance, which is what separates DEGRADE
    (structure lost) from REORIENT (structure preserved, axes moved).
  * THE SHARP ONE, for rotations only: post-absorption h6 under the NEW key map against
    pre-rotation h6 under the OLD map, for the same latent z. If `track` is pure re-indexing
    the two coincide; if it is a rebuild they do not. Bracketed by two controls at the same
    checkpoint — same weights/different map (how much the map alone moves h6) and the quiet
    stride pair (how much time alone moves it).

Run from experiments/:
  modal run -m rhm.practice.tuning.gate1g::weights --tag g1g
  modal run -m rhm.practice.tuning.gate1g::reps    --tag g1g
"""

import json
import os
import time

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-practice-tuning-g1g", image=image)

REMOTE = "rhm_practice_tuning_g1g"
SRC = "rhm_practice_tuning_g1"          # read-only: wave 1's banked checkpoints

ROTS = [8000, 10000, 12000, 14000, 16000, 18000]
BURSTS = [5000, 11000, 17000]
DRIFTS = [6000, 13000, 19000]
EVENTS = ([("rotation", s) for s in ROTS] + [("burst", s) for s in BURSTS]
          + [("drift", s) for s in DRIFTS])
LAG = 125                                # the interval length every event shares
BURST_LEN = 250                          # a burst's own duration -> its own guard


def busy_windows(burst_len=BURST_LEN):
    """[onset, onset+guard] per event: the stretch an interval must not touch to be
    called quiet. The guard is the event's OWN duration -- 125 for the instantaneous
    events (one checkpoint of absorption), burst_len for a burst, which is a window."""
    out = []
    for kind, e in EVENTS:
        out.append((e, e + (burst_len if kind == "burst" else LAG)))
    return out


def quiet_pairs(steps, nmin=125, burst_len=BURST_LEN):
    """Adjacent-checkpoint intervals disjoint from every busy window.

    NOTE the design choice. The strict reading -- "no event within 400 either side" --
    leaves exactly ONE usable interval on this bank (2000->4000), which cannot support a
    scaling fit. So an interval is admitted as quiet if it merely does not OVERLAP any
    event's guard: it may begin exactly at onset+guard. That is deliberately conservative
    in the direction that matters -- the absorption tail is counted as quiet, which
    inflates the null and shrinks every event's excess. It also buys a handful of short
    intervals (and, on `track`, one at exactly n=125: 18125->18250), so the extrapolation
    to the matched unit is short or unnecessary rather than heroic."""
    busy = busy_windows(burst_len)
    out = []
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        if b - a < nmin:
            continue
        if any(a < hi and b > lo for lo, hi in busy):
            continue
        out.append((a, b, b - a))
    return out


def quiet_scaling(quiet, layers):
    """Per layer, fit log rel = c + alpha log n over the quiet intervals and predict at
    n = LAG. Reported with alpha, r2 and n so the extrapolation is auditable; where a
    quiet interval of exactly LAG exists it is reported alongside as `direct`."""
    import numpy as np
    out = {}
    for L in layers:
        xs = [(q["n"], q["rel"][L]) for q in quiet
              if L in q["rel"] and q["rel"][L] > 0]
        if len(xs) < 3:
            continue
        x = np.log(np.array([a for a, _ in xs], float))
        y = np.log(np.array([b for _, b in xs], float))
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss = float(((y - pred) ** 2).sum())
        tot = float(((y - y.mean()) ** 2).sum())
        direct = [b for a, b in xs if a == LAG]
        out[L] = {"alpha": float(coef[1]), "intercept": float(coef[0]),
                  "r2": 1.0 - ss / max(tot, 1e-12), "n_pairs": len(xs),
                  "pred_at_lag": float(np.exp(coef[0] + coef[1] * np.log(LAG))),
                  "direct_at_lag": float(np.mean(direct)) if direct else None}
    return out


def layer_of(k):
    """Group a parameter name into the layer whose absorption we want to localise."""
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


LAYERS = ["wte", "wpe"] + [f"h{i}" for i in range(8)] + ["ln_f", "lm_head"]


# --------------------------------------------------------------------------- #
# PART 1 — weight side (CPU)
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=32768)
def weights(tag: str = "g1g", src_tag: str = "g1a",
            arms: str = "track,self,pair_key,pair_parse"):
    import numpy as np
    import torch

    ck = f"{DATA_DIR}/{SRC}/{src_tag}/ckpt"
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    avail = {}
    for f in os.listdir(ck):
        if not f.endswith(".pt"):
            continue
        arm, st = f[:-3].rsplit("_s", 1)
        avail.setdefault(arm, {})[int(st)] = os.path.join(ck, f)
    print({a: len(v) for a, v in avail.items()}, flush=True)

    def load(arm, step):
        sd = torch.load(avail[arm][step], map_location="cpu")["model"]
        return {k: v.float() for k, v in sd.items()}

    def delta_by_layer(a, b):
        """{layer: (||dtheta||_F, ||theta_a||_F, flat dtheta)}"""
        acc = {}
        for k in a:
            L = layer_of(k)
            d = (b[k] - a[k]).reshape(-1)
            acc.setdefault(L, [[], 0.0])
            acc[L][0].append(d)
            acc[L][1] += float((a[k].reshape(-1) ** 2).sum())
        out = {}
        for L, (parts, n2) in acc.items():
            v = torch.cat(parts)
            out[L] = (float(v.norm()), float(np.sqrt(n2)), v)
        return out

    out = {"src_tag": src_tag, "lag": LAG, "arms": {}}

    for arm in arm_list:
        if arm not in avail:
            continue
        steps = sorted(avail[arm])
        rec = {"steps": steps, "events": {}, "quiet": [], "align": {}}

        # ---- the built null: ||dtheta|| ~ n^alpha over quiet stride intervals ----
        qp = quiet_pairs(steps)
        print(f"[{arm}] quiet pairs ({len(qp)}): {qp}", flush=True)
        for a_, b_, n in qp:
            A, B = load(arm, a_), load(arm, b_)
            d = delta_by_layer(A, B)
            rec["quiet"].append({"a": a_, "b": b_, "n": n,
                                 "rel": {L: d[L][0] / max(d[L][1], 1e-12)
                                         for L in d}})
            del A, B, d
        rec["quiet_fit"] = quiet_scaling(rec["quiet"], LAYERS)

        # ---- per-event, per-layer relative weight change over [onset, onset+LAG] ----
        for kind, e in EVENTS:
            if e not in avail[arm] or (e + LAG) not in avail[arm]:
                continue
            A, B = load(arm, e), load(arm, e + LAG)
            d = delta_by_layer(A, B)
            rel = {L: d[L][0] / max(d[L][1], 1e-12) for L in d}
            fit = rec["quiet_fit"]
            rec["events"][f"{kind}_{e}"] = {
                "kind": kind, "onset": e, "rel": rel,
                "abs": {L: d[L][0] for L in d},
                # how many times ordinary 125-step plasticity this event bought
                "excess": {L: rel[L] / fit[L]["pred_at_lag"]
                           for L in rel if L in fit and fit[L]["pred_at_lag"] > 0}}
            # keep the flat delta only for rotations/bursts/drifts we align later
            rec.setdefault("_vec", {})[f"{kind}_{e}"] = {L: d[L][2] for L in d}
            del A, B

        # ---- the re-map alignment test ----
        vecs = rec.pop("_vec", {})
        names = sorted(vecs)
        cos = {}
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                key = f"{ni}|{nj}"
                cos[key] = {}
                for L in LAYERS:
                    if L not in vecs[ni] or L not in vecs[nj]:
                        continue
                    u, w = vecs[ni][L], vecs[nj][L]
                    cos[key][L] = float(
                        torch.dot(u, w) / (u.norm() * w.norm() + 1e-12))
        rec["align"] = cos
        # floor: cos of independent random vectors at the same per-layer dimension
        g = torch.Generator().manual_seed(0)
        floor = {}
        for L in LAYERS:
            any_n = next((vecs[n][L].numel() for n in names if L in vecs[n]), None)
            if any_n is None:
                continue
            cs = []
            for _ in range(64):
                u = torch.randn(any_n, generator=g)
                w = torch.randn(any_n, generator=g)
                cs.append(float(torch.dot(u, w) / (u.norm() * w.norm())))
            floor[L] = {"d": int(any_n), "analytic": float(1.0 / np.sqrt(any_n)),
                        "measured_p99": float(np.percentile(np.abs(cs), 99))}
        rec["cos_floor"] = floor
        out["arms"][arm] = rec
        print(f"[{arm}] events {len(rec['events'])}  align pairs {len(cos)}", flush=True)

    od = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "weights.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print("wrote weights.json", flush=True)
    return {"arms": list(out["arms"])}


# --------------------------------------------------------------------------- #
# PART 2 — representation side (GPU)
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def reps(tag: str = "g1g", src_tag: str = "g1a", arms: str = "track,self",
         n_panel: int = 512, blocks: str = "post_block3,post_block6"):
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import drift_ev as DR

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    setup = json.load(open(f"{DATA_DIR}/{SRC}/{src_tag}/setup.json"))
    c = setup["config"]
    v, s, L, m = c["v"], c["s"], c["depth"], c["m"]
    T, V = s ** L, W.vocab_size(v)
    kl, kn = c["key_level"], c["key_node"]
    key_lo, key_hi = W.key_span(L, s, kl, kn)
    rot_step, phase1, rot_period = c["rot_step"], c["phase1"], c["rot_period"]
    blk = [b.strip() for b in blocks.split(",")]

    base = generate_rules_distinct(v, s, L, m, seed=c["rule_seed"])
    dsteps = DR.drift_steps(c["drift_steps"], c["max_steps"])
    traj, _ = DR.rules_trajectory(base, L, c["drift_level"], c["drift_cells"],
                                  dsteps, c["drift_seed"])
    # THE PANEL'S OWN BATCH, literally: gate1_lm draws n_eval sequences at eval_seed per
    # epoch and the T* panel reads its first n_panel rows (`pn_leaf = ev_leaf[:n_panel]`).
    # One deviation, stated: gate1_lm binds pn_leaf ONCE at epoch 0 and never rebinds it
    # on the epoch turn, so its panel keeps reading pre-drift sequences after a drift.
    # Here each epoch is read on its OWN grammar's draw -- otherwise a drift contrast
    # would compare two models on sequences neither of them currently lives in.
    EP = []
    for r_e in traj:
        el, elf, _ = _generate_with_traces(r_e, c["n_eval"], c["eval_seed"])
        EP.append((torch.from_numpy(el[:n_panel].astype(np.int64)),
                   elf[kl][:n_panel, kn].astype(np.int64)))

    ck = f"{DATA_DIR}/{SRC}/{src_tag}/ckpt"
    avail = {}
    for f in os.listdir(ck):
        if f.endswith(".pt"):
            arm, st = f[:-3].rsplit("_s", 1)
            avail.setdefault(arm, {})[int(st)] = os.path.join(ck, f)

    model = GPT(V, T, c["n_layer"], c["n_head"], c["n_embd"]).to(dev).eval()

    def put(arm, step):
        sd = torch.load(avail[arm][step], map_location="cpu")["model"]
        model.load_state_dict({k: t.float() for k, t in sd.items()})

    @torch.no_grad()
    def acts(leaf_t, wcol, chunk=256):
        """{block: (n, span, d)} on the indexed span."""
        out = {b: [] for b in blk}
        for i in range(0, leaf_t.shape[0], chunk):
            x = torch.cat([wcol[i:i + chunk, None],
                           leaf_t[i:i + chunk, :-1]], 1).to(dev)
            _, _, inter = model(x, return_intermediates=True)
            for b in blk:
                out[b].append(inter[b][:, key_lo:key_hi, :].float().cpu())
        return {b: torch.cat(vv) for b, vv in out.items()}

    def wall_col(z, q):
        return torch.from_numpy(W.wall_tok((z + q) % v, v).astype(np.int64))

    def neutral(n):
        return torch.full((n,), W.neutral_tok(v), dtype=torch.long)

    # ---------------- metrics ----------------
    def class_means(A, z):
        """(16, d) mean over sequences of each latent class, span-averaged."""
        X = A.mean(1)                                    # (n, d) span-mean
        return torch.stack([X[z == a].mean(0) if (z == a).sum() > 0
                            else torch.zeros(X.shape[1]) for a in range(v)])

    def rdm(M):
        Mc = M - M.mean(0, keepdim=True)
        Mn = Mc / (Mc.norm(dim=1, keepdim=True) + 1e-12)
        C = Mn @ Mn.T
        iu = torch.triu_indices(C.shape[0], C.shape[0], offset=1)
        return (1 - C)[iu[0], iu[1]]

    def rsa(A, B, z):
        ra, rb = rdm(class_means(A, z)), rdm(class_means(B, z))
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))

    def separability(A, z):
        """between-class / within-class variance of the span-mean activation."""
        X = A.mean(1)
        gm = X.mean(0)
        bw, wi, n = 0.0, 0.0, 0
        for a in range(v):
            sel = X[z == a]
            if sel.shape[0] < 2:
                continue
            cm = sel.mean(0)
            bw += sel.shape[0] * float(((cm - gm) ** 2).sum())
            wi += float(((sel - cm) ** 2).sum())
            n += sel.shape[0]
        return float(bw / max(wi, 1e-12))

    def align(A, B, k=16, ridge=1e-2):
        """ridge R^2 for a linear map A->B, and mean principal angle of top-k PCs."""
        X = A.mean(1); Y = B.mean(1)
        Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
        G = Xc.T @ Xc + ridge * torch.eye(Xc.shape[1]) * float(Xc.shape[0])
        Wm = torch.linalg.solve(G, Xc.T @ Yc)
        r2 = 1.0 - float(((Yc - Xc @ Wm) ** 2).sum() / max(float((Yc ** 2).sum()), 1e-12))
        ua = torch.linalg.svd(Xc, full_matrices=False)[2][:k]
        ub = torch.linalg.svd(Yc, full_matrices=False)[2][:k]
        sv = torch.linalg.svdvals(ua @ ub.T).clamp(-1, 1)
        return r2, float(torch.rad2deg(torch.arccos(sv)).mean())

    def reldist(A, B):
        return float((B - A).norm() / (A.norm() + 1e-12))

    cache = {}

    def get(arm, step, q, ep):
        ck_ = (arm, step, q, ep)
        if ck_ not in cache:
            lt, z = EP[ep]
            w = neutral(lt.shape[0]) if q is None else wall_col(z, q)
            put(arm, step)
            cache[ck_] = acts(lt, w)
            if len(cache) > 24:                      # bounded; each is ~2 x 512x16x256
                cache.pop(next(iter(cache)))
        return cache[ck_]

    def pair(arm, sa, sb, qa=None, qb=None, ea=None, eb=None):
        """all metrics between checkpoint sa (map qa) and sb (map qb)."""
        ea = DR.epoch_of(sa, dsteps) if ea is None else ea
        eb = DR.epoch_of(sb, dsteps) if eb is None else eb
        # every pair here is within one epoch (a drift onset and onset+125 share an
        # epoch, since epoch_of is >=), so the two sides are row-matched and rel_dist /
        # ridge alignment are meaningful. Guard it rather than assume it.
        assert ea == eb, f"cross-epoch pair {sa}->{sb}: rows would not correspond"
        z_a = EP[ea][1]
        A = get(arm, sa, qa, ea)
        B = get(arm, sb, qb, eb)
        o = {}
        for b in blk:
            r2, ang = align(A[b], B[b])
            o[b] = {"rsa": rsa(A[b], B[b], z_a), "align_r2": r2,
                    "pc_angle_deg": ang, "rel_dist": reldist(A[b], B[b]),
                    "sep_pre": separability(A[b], z_a),
                    "sep_post": separability(B[b], EP[eb][1])}
        return o

    out = {"src_tag": src_tag, "lag": LAG, "blocks": blk, "n_panel": n_panel, "arms": {}}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        if arm not in avail:
            continue
        rec = {"events": {}, "quiet": {}, "rotation_reindex": {}}
        steps = sorted(avail[arm])
        t0 = time.time()

        # ---- the built null: quiet intervals, same map on both sides ----
        for a_, b_, n in quiet_pairs(steps, nmin=LAG, burst_len=c["burst_len"]):
            q = W.q_at(a_, phase1, rot_period, rot_step, v)
            q2 = W.q_at(b_, phase1, rot_period, rot_step, v)
            if q != q2:
                continue
            # a quiet interval may END exactly on a drift onset; the two sides would then
            # be read on different grammars (different sequences), so rows would not
            # correspond. Those are not admissible nulls here.
            if DR.epoch_of(a_, dsteps) != DR.epoch_of(b_, dsteps):
                continue
            rec["quiet"][f"{a_}_{b_}"] = {"n": n, **pair(arm, a_, b_, q, q2)}

        # ---- per event, [onset, onset+LAG], each side under its OWN consumed map ----
        for kind, e in EVENTS:
            if e not in avail[arm] or (e + LAG) not in avail[arm]:
                continue
            qa = W.q_at(e, phase1, rot_period, rot_step, v)
            qb = W.q_at(e + LAG, phase1, rot_period, rot_step, v)
            rec["events"][f"{kind}_{e}"] = {"kind": kind, "onset": e,
                                            "q_pre": int(qa), "q_post": int(qb),
                                            **pair(arm, e, e + LAG, qa, qb)}

        # ---- THE SHARP ONE: is `track` pure re-indexing? ----
        for e in ROTS:
            if e not in avail[arm] or (e + LAG) not in avail[arm]:
                continue
            q_old = W.q_prev_at(e, phase1, rot_period, rot_step, v)
            q_new = W.q_at(e, phase1, rot_period, rot_step, v)
            rec["rotation_reindex"][str(e)] = {
                "q_old": int(q_old), "q_new": int(q_new),
                # the test: pre-rotation weights under the OLD map vs post-absorption
                # weights under the NEW map, same latent z
                "test": pair(arm, e, e + LAG, q_old, q_new),
                # control A: same weights, map swapped -- how much the map alone moves h
                "ctrl_map_only": pair(arm, e, e, q_old, q_new),
                # control B: the naive read -- both sides under the NEW map
                "ctrl_same_map": pair(arm, e, e + LAG, q_new, q_new),
            }
        print(f"[{arm}] done in {time.time() - t0:.0f}s", flush=True)
        out["arms"][arm] = rec

    od = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "reps.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print("wrote reps.json", flush=True)
    return {"arms": list(out["arms"])}
