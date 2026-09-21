"""[duplex] DOES THE OUTCOME ERROR HAVE TO REACH THE TRUNK'S WEIGHTS? — offline, on banked
checkpoints, before the paid in-loop version.

`aliquot` put a linear readout of the practice plant's state in the grader's seat and found the
pooled hiddens do not linearly carry the world's verdict: held-out AUC 0.73-0.76 on the live
trunk against 0.72-0.74 on a never-trained trunk, where the mirror — a tree net trained end to
end on outcomes — reads 0.99 on the same rows. The reading in discussion: a plant trained only
on masked infilling represents what PREDICTION needs and not what the OUTCOME needs, so the
outcome error has to reach the trunk's weights and not just the readout's.
`logit_reading/orbitofrontal/shaped/` measured exactly that on the logit-reading trunk (task +
next-token widened the state's consequence headroom for +0.008 nats of predictive cost; task
alone paid +0.61 nats and left the Bayesian-observer family). Nothing analogous exists on the
practice plant.

THIS IS NOT THE PAID VERSION. It is a directional read on banked checkpoints: the rows were
drawn under the FROZEN plant's behaviour, so nothing here can say what a loop whose plant is
shaped would go on to write. What it can say is whether the outcome error, applied to the
trunk's weights on rows the loop already has, makes the state linearly carry the verdict — and
what that costs the world model.

THE PROTOCOL IS `overtone/analyze_dump.py::post_write_probe`'s, reproduced as the gate (§G1
below) and then re-read after a shaping step. Per seed:

    frozen        the banked trunk, untouched                       (= overtone's `post_slot`)
    rand          a never-trained trunk of the same architecture    (= overtone's `rand_slot`,
                  minted under overtone's own `manual_seed(20260915)` so the gate is exact)
    both@N        N steps of  outcome BCE + masked-infill CE        (shaped, both losses)
    out@N         N steps of  outcome BCE alone                     (shaped, outcome only)
    infill@N      N steps of  masked-infill CE alone                (matched steps, so "the
                  trunk moved" is separated from "the trunk moved toward the outcome")

and on every one of the eight trunks: the linear probe's held-out AUC per slot and diet, the
world-model diagnostics (held-out infill CE, the block head's level-1 parse accuracy, the
nested DP parse at every rung), and the transfer split (fit at L2/L3, score at L4/L5).

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep
    modal run rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep --smoke 1
"""

import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-duplex", image=image)

REMOTE = "rhm_practice_duplex"                 # where this node writes
BANK = "rhm_practice_voicing"                  # where the banked dumps live

# The two banked seeds, and nothing beyond them (SPEC).
SEEDS = {
    "s0": {"tag": "ov_s0b", "arm": "ovt_comp_pr_sh"},     # uniform probe draw
    "s2": {"tag": "ov_s2", "arm": "ovt_comp_pr_dis"},     # 3/4 disagreement-drawn, 1/4 uniform
}

# `overtone::post_write_probe`'s own control-two seed. Kept verbatim so the `rand` row IS
# overtone's `rand_slot` and the reproduction gate is exact. (`aliquot`'s in-loop twin uses
# 20260918; that is a different object and is not what is being reproduced here.)
OV_RAND_SEED = 20260915

# The probe of record, `_fit_probe`'s literals, unchanged.
PROBE_RIDGE, PROBE_ITERS = 1e-2, 40

# Fresh-corpus seeds. Distinct from the run's own `train_seed` (1) and from the probe-clean
# stream (`seed + 31337`), so every window the shaping and the diagnostics see is a fresh draw
# from the same DGP.
CORPUS_SEED_TRAIN = 777001
CORPUS_SEED_EVAL = 777002
N_CORPUS_TRAIN = 32768
N_CORPUS_EVAL = 8192

# The shaping. ONE learning rate, the loop's own plant continuation rate (`gen_lr = 1e-4`), and
# three step counts. The paid arm continues its plant for `gen_steps = 20` per cycle over ~200
# cycles ~= 4000 steps, so 100 / 400 / 1600 are 2.5% / 10% / 40% of its step budget at its own
# rate. Three rather than two because the smoke showed 20 steps of outcome-only already moved
# the level-1 parse by -0.12, so the interesting range starts low.
SHAPE_LR = 1e-4
SHAPE_STEPS = (100, 400, 1600)
SHAPE_SLOT_BATCH = 32          # outcome rows drawn per slot per step (the critic draws 64)
SHAPE_INFILL_BATCH = 256       # `batch_size` in the run's own config
SHAPE_CLIP = 1.0               # `_train_generator`'s clip, applied to every arm alike

# The transfer split's fit cap, matched across the three splits so L2/L3 and L4/L5 fits are
# read at the same n.
XFER_CAP = 20000

# The world-model diagnostic's fixed evaluation. Identical windows and identical masks on every
# trunk, so a difference is the trunk.
WM_INFILL_BATCHES = 16
WM_INFILL_BATCH = 512
WM_PARSE_N = 4096
WM_DP_N = 1024
WM_SEED = 909


# --------------------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------------------- #

def irls_t(A, y, ridge=PROBE_RIDGE, iters=PROBE_ITERS):
    """`voicing::ov_irls` in torch float64 — same Newton step, same clip, same early stop, same
    ridge on every column including the intercept. Gate G0 asserts it against the numpy
    estimator on real rows; nothing here is a second estimator by intent."""
    import torch
    w = torch.zeros(A.shape[1], dtype=torch.float64, device=A.device)
    I = torch.eye(A.shape[1], dtype=torch.float64, device=A.device)
    for _ in range(int(iters)):
        p = torch.sigmoid((A @ w).clamp(-30.0, 30.0))
        g = A.T @ (p - y) + ridge * w
        sw = p * (1.0 - p) + 1e-6
        H = (A * sw[:, None]).T @ A + ridge * I
        try:
            stp = torch.linalg.solve(H, g)
        except Exception:
            break
        w = w - stp
        if float(stp.abs().max()) < 1e-9:
            break
    return w


def fit_probe_t(F, y, tr, ho, ridge=PROBE_RIDGE, iters=PROBE_ITERS):
    """`overtone::_fit_probe` verbatim, in torch: standardise on the TRAINING rows, append an
    unpenalised-by-nothing intercept column (the donor penalises it too and that is kept),
    solve, score the held-out rows. Returns the held-out scores or None."""
    import torch
    if int(tr.sum()) < 64 or int(ho.sum()) < 16:
        return None
    yt = y[tr]
    if float(yt.min()) == float(yt.max()):
        return None
    Ftr = F[tr].double()
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = torch.where(sd < 1e-9, torch.ones_like(sd), sd)
    A = torch.cat([(Ftr - mu) / sd, torch.ones(Ftr.shape[0], 1, dtype=torch.float64,
                                               device=F.device)], 1)
    B = F[ho].double()
    B = torch.cat([(B - mu) / sd, torch.ones(B.shape[0], 1, dtype=torch.float64,
                                             device=F.device)], 1)
    w = irls_t(A, yt.double(), ridge=ridge, iters=iters)
    return B @ w


def auc_t(x, y):
    """`voicing::vo_auc`'s rank identity with ties averaged, on torch. None when degenerate."""
    import torch
    x = x.double().flatten()
    y = y.double().flatten()
    m = torch.isfinite(x)
    x, y = x[m], y[m]
    n1 = float((y > 0.5).sum())
    n0 = float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    o = torch.argsort(x, stable=True)
    r = torch.empty_like(x)
    r[o] = torch.arange(1, x.numel() + 1, dtype=torch.float64, device=x.device)
    xs = x[o]
    # average the ranks of tied runs
    uniq, inv, cnt = torch.unique(xs, return_inverse=True, return_counts=True)
    csum = torch.cumsum(cnt, 0).double()
    start = csum - cnt.double()
    meanr = (start + csum + 1.0) / 2.0
    r[o] = meanr[inv]
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------------------- #
# the trunk read
# --------------------------------------------------------------------------------------- #

def span_mask(blk0, span, n_blocks, device):
    import torch
    m = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    m[blk0:blk0 + span] = 1.0
    return m


def render_write(obs, wr, canon, blk0, span, s):
    """The post-write configuration: the candidate rendered into the masked context at the slot.
    `overtone::post_write_probe`'s `x2`, verbatim."""
    import torch
    b = obs.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=obs.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=obs.device)[None, :]).reshape(-1)
    return obs.clone().scatter_(1, pos[None, :].expand(b, -1),
                                canon[wr].reshape(b, -1))


def mask_outside(x, blk0, span, n_blocks, s):
    """`ratchet/macros.py::parse_features`'s `mask_block` idiom: mask the FIRST BLOCK OUTSIDE
    the span. The plant is trained with `mask_min = 1` and has never seen a fully unmasked
    input; `aliquot` DESIGN §2.1 records that this mattered in the loop (holdAUC 0.759 masked
    against 0.747 unmasked on the live trunk, 0.716 against 0.641 on the random one)."""
    import torch
    mb = (blk0 + span) % n_blocks
    if blk0 <= mb < blk0 + span:                 # only reachable if span == n_blocks
        mb = (blk0 - 1) % n_blocks
    pos = (mb * s + torch.arange(s, device=x.device))[None, :]
    return x.clone().scatter_(1, pos.expand(x.shape[0], -1),
                              torch.full((x.shape[0], s), -1, dtype=x.dtype, device=x.device))


def trunk_features(core, obs, wr, canon, blk0, span, s, n_blocks, device, batch=4096,
                   critic=None, slot_id=0):
    """The four feature blocks the probe reads, on one buffer, through one trunk.

        pre    [pooled(obs).mean ; pooled(obs)[span].mean]              — the write is NOT in it
        pmean   pooled(obs+write).mean
        post   [pooled(obs+write).mean ; pooled(obs+write)[span].mean]  — the read of record
        postm   the same with one block outside the span masked

    Also the banked critic's logit on `pooled(obs)`, if a critic is passed, so "the critic read
    through a trunk that moved under it" is on the record too.
    """
    import torch
    import rhm.practice.native.span.span_net as SN
    out = {"pre": [], "pmean": [], "post": [], "postm": [], "mlp": []}
    with torch.no_grad():
        for a in range(0, obs.shape[0], batch):
            ob = obs[a:a + batch].to(device).long()
            wb = wr[a:a + batch].to(device).long()
            x2 = render_write(ob, wb, canon, blk0, span, s)
            x2m = mask_outside(x2, blk0, span, n_blocks, s)
            p0, _ = SN.trunk(core, ob)
            p1, _ = SN.trunk(core, x2)
            p1m, _ = SN.trunk(core, x2m)
            out["pre"].append(torch.cat([p0.mean(1), p0[:, blk0:blk0 + span, :].mean(1)], 1))
            out["pmean"].append(p1.mean(1))
            out["post"].append(torch.cat([p1.mean(1), p1[:, blk0:blk0 + span, :].mean(1)], 1))
            out["postm"].append(torch.cat([p1m.mean(1),
                                           p1m[:, blk0:blk0 + span, :].mean(1)], 1))
            if critic is not None:
                sid = torch.full((ob.shape[0],), int(slot_id), dtype=torch.long, device=device)
                out["mlp"].append(critic(p0, blk0, span, sid, wb))
    return {k: (torch.cat(v) if v else None) for k, v in out.items()}


# --------------------------------------------------------------------------------------- #
# the shaping
# --------------------------------------------------------------------------------------- #

def build_out_head(dim, n_slots, sdim=32, hidden=256, seed=0, device=None):
    """THE OUTCOME HEAD: an MLP over the SAME pooled features the linear probe reads, plus a
    slot embedding.

    The brief allowed either this or the critic's own form. This form was chosen because the
    shaping gradient then flows through exactly the representation the probe is going to read —
    the post-write pooled mean and the span pool — with nothing else in the path. In particular
    it carries NO candidate embedding: the candidate is already rendered into the input, so its
    content reaches the head only through the trunk, which is the object under test. The slot
    embedding is the critic's own (`Critic.slot`) and is what lets one head serve 30 slots.
    """
    import torch
    import torch.nn as nn

    class OutHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.se = nn.Embedding(int(n_slots), int(sdim))
            self.net = nn.Sequential(nn.Linear(2 * int(dim) + int(sdim), int(hidden)),
                                     nn.GELU(), nn.Linear(int(hidden), 1))

        def forward(self, pooled, smask, span, sid):
            f = torch.cat([pooled.mean(1),
                           (pooled * smask[:, :, None]).sum(1) / span[:, None],
                           self.se(sid)], 1)
            return self.net(f).squeeze(-1)

    st = torch.get_rng_state()
    h = OutHead()
    torch.set_rng_state(st)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in h.parameters():
            if p.dim() >= 2:
                p.copy_(torch.empty(p.shape, dtype=p.dtype).normal_(0.0, 0.02, generator=g))
            else:
                p.zero_()
    return h.to(device)


def fingerprint(core):
    """`aliquot`'s P-3r: sum of |parameter| in float64, so "the trunk moved" is a number."""
    import torch
    with torch.no_grad():
        return float(sum(p.detach().double().abs().sum() for p in core.parameters()))


def shape_trunk(core, bufs, layout, corpus, mode, steps, lr, device, cfg, log):
    """Continue the banked trunk for `steps` steps at `lr` on one of three losses.

    `mode`  "both"    outcome BCE + masked-infill CE, weight 1 each — the run's own convention
                      (`vo_critic_terms` adds the critic's term to the plant's loss in the SAME
                      optimizer step, `span_lam = 1.0`)
            "out"     outcome BCE alone
            "infill"  masked-infill CE alone, matched steps

    MATCHED STREAMS. The outcome row draw is one numpy generator seeded on `steps` alone, so
    "both" and "out" see the SAME rows in the same order; the infill window/mask draw is one
    torch generator seeded on `steps` alone, so "both" and "infill" see the SAME windows and the
    SAME masks. A difference between the arms is therefore the loss and not the data.
    """
    import torch
    import torch.nn.functional as F
    import rhm.practice.native.span.span_net as SN

    v, s, n_blocks = int(cfg["v"]), int(cfg["s"]), int(layout["n_blocks"])
    canon = layout["canon"]
    keys = layout.get("shape_keys") or layout["keys"]      # [du2] the outcome loss's slots
    core.train()
    params = list(core.parameters())
    head = None
    if mode in ("both", "out"):
        head = build_out_head(int(cfg["state_dim"]), layout["n_slots"], seed=0, device=device)
        params = params + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)
    rng = np.random.default_rng(10_000 + int(steps))
    g = torch.Generator(device=device).manual_seed(20_000 + int(steps))

    # the outcome batch's constant parts, built once
    per = SHAPE_SLOT_BATCH
    smask = torch.cat([span_mask(layout[k]["blk0"], layout[k]["span"], n_blocks,
                                 device).expand(per, -1) for k in keys]).float()
    spans = torch.cat([torch.full((per,), float(layout[k]["span"]), device=device)
                       for k in keys])
    sids = torch.cat([torch.full((per,), int(layout[k]["slot_id"]), dtype=torch.long,
                                 device=device) for k in keys])

    powers = (v ** torch.arange(s, device=device))
    bottom = corpus["bottom_map"]
    pool = corpus["train"]
    hist = []
    for step in range(1, int(steps) + 1):
        loss = None
        o_bce = o_auc = float("nan")
        if head is not None:
            xs, ys = [], []
            for k in keys:
                b = bufs[k]
                idx = b["tr_idx"][rng.integers(0, b["tr_idx"].shape[0], size=per)]
                ob = b["obs"][idx].to(device).long()
                wb = b["write"][idx].to(device).long()
                xs.append(render_write(ob, wb, canon, layout[k]["blk0"],
                                       layout[k]["span"], s))
                ys.append(b["y"][idx].to(device).double())
            x = torch.cat(xs)
            y = torch.cat(ys).float()
            pooled, _ = SN.trunk(core, x)
            lg = head(pooled, smask, spans, sids)
            ot = F.binary_cross_entropy_with_logits(lg, y)
            o_bce = float(ot.item())
            a_ = auc_t(lg.detach(), y.detach())
            o_auc = float("nan") if a_ is None else a_
            loss = ot if loss is None else loss + ot
        i_ce = i_acc = float("nan")
        if mode in ("both", "infill"):
            bi = torch.randint(0, pool.shape[0], (SHAPE_INFILL_BATCH,), generator=g,
                               device=device)
            leaves = pool[bi].long()
            tf = bottom[(leaves.view(-1, n_blocks, s) * powers).sum(-1)]
            n_mask = int(torch.randint(1, n_blocks + 1, (), generator=g, device=device).item())
            order = torch.rand(SHAPE_INFILL_BATCH, n_blocks, generator=g,
                               device=device).argsort(dim=1)
            mb = order[:, :n_mask]
            pos = mb[:, :, None] * s + torch.arange(s, device=device)
            ob = leaves.clone()
            ob.scatter_(1, pos.reshape(SHAPE_INFILL_BATCH, -1),
                        torch.full((SHAPE_INFILL_BATCH, n_mask * s), -1, device=device,
                                   dtype=leaves.dtype))
            lg2 = core.block_logits(ob)
            mk = torch.zeros(SHAPE_INFILL_BATCH, n_blocks, dtype=torch.bool, device=device)
            mk.scatter_(1, mb, True)
            it = F.cross_entropy(lg2[mk], tf[mk])
            i_ce = float(it.item())
            i_acc = float((lg2[mk].argmax(-1) == tf[mk]).double().mean().item())
            loss = it if loss is None else loss + it
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), SHAPE_CLIP)
        opt.step()
        if step % max(1, int(steps) // 8) == 0 or step == 1:
            hist.append({"step": step, "out_bce": o_bce, "out_auc": o_auc,
                         "infill_ce": i_ce, "infill_acc": i_acc})
            log(f"      {mode}@{steps} step {step:5d}  out_bce {o_bce:.4f} out_auc {o_auc:.3f}"
                f"  infill_ce {i_ce:.4f} acc {i_acc:.3f}")
    core.eval()
    return core, hist


# --------------------------------------------------------------------------------------- #
# the world-model diagnostics
# --------------------------------------------------------------------------------------- #

def wm_diag(core, corpus, layout, cfg, device, log):
    """What the shaping cost the world model. `shaped`'s pair was the trunk's clean next-token
    CE and the altitude instrument; here:

      infill      HELD-OUT masked-infill CE and fill accuracy on FRESH corpus windows, at a
                  FIXED window/mask set identical on every trunk, split by how many blocks are
                  masked (1, 2-4, 5-8, 9-16, 17-32)
      parse       the block head's level-1 parse accuracy against the exact features — an
                  ORACLE instrument, fine here — read with the `mask_block` idiom (mask block 0,
                  score the other 31) and, beside it, unmasked
      dp          the NESTED parse: `macros.macro_features`' max-sum DP over the TRUE table at
                  level 2..max_macro_level, span masked, recovered level-1 features against the
                  exact ones. This is the practice-side analogue of `shaped`'s altitude
                  instrument: how deep the trunk's generative grasp of the nested grammar
                  survives. (§D3 in DESIGN records why `frontier/common.py::nested_phase` is
                  NOT that analogue on this plant.)
    """
    import torch
    import torch.nn.functional as F
    import rhm.practice.ratchet.macros as MC

    v, s = int(cfg["v"]), int(cfg["s"])
    n_blocks = int(layout["n_blocks"])
    out = {}
    powers = (v ** torch.arange(s, device=device))
    bottom = corpus["bottom_map"]
    # ---- held-out infill, on the fixed plan ------------------------------------------- #
    tot, num, den, buckets = 0.0, 0.0, 0, {}
    with torch.no_grad():
        for pl in corpus["infill_plan"]:
            leaves = corpus["eval"][pl["idx"]].long()
            b = leaves.shape[0]
            tf = bottom[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
            mb = pl["mb"]
            pos = mb[:, :, None] * s + torch.arange(s, device=device)
            ob = leaves.clone()
            ob.scatter_(1, pos.reshape(b, -1),
                        torch.full((b, mb.shape[1] * s), -1, device=device,
                                   dtype=leaves.dtype))
            lg = core.block_logits(ob)
            mk = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
            mk.scatter_(1, mb, True)
            ce = float(F.cross_entropy(lg[mk], tf[mk]).item())
            ac = float((lg[mk].argmax(-1) == tf[mk]).double().mean().item())
            n = int(mk.sum())
            tot += ce * n
            num += ac * n
            den += n
            k = pl["bucket"]
            d = buckets.setdefault(k, [0.0, 0.0, 0])
            d[0] += ce * n
            d[1] += ac * n
            d[2] += n
    out["infill_ce"] = tot / max(den, 1)
    out["infill_acc"] = num / max(den, 1)
    out["infill_by_nmask"] = {k: {"ce": d[0] / d[2], "acc": d[1] / d[2], "n": d[2]}
                              for k, d in sorted(buckets.items())}
    # ---- the level-1 parse against the exact features --------------------------------- #
    x = corpus["parse_x"]
    ex = corpus["parse_exact"]
    with torch.no_grad():
        f_m = MC.parse_features(core, x, s=s, mask_block=0)
        f_u = MC.parse_features(core, x)
    keep = torch.ones(n_blocks, dtype=torch.bool, device=device)
    keep[0] = False
    out["parse_acc_mask1"] = float((f_m[:, keep] == ex[:, keep]).double().mean().item())
    out["parse_acc_nomask"] = float((f_u == ex).double().mean().item())
    out["parse_acc_nomask_offblk0"] = float((f_u[:, keep] == ex[:, keep]).double().mean().item())
    # ---- the nested DP parse at every rung -------------------------------------------- #
    dp = {}
    for level, moves in corpus["dp_moves"].items():
        accs, ns = [], 0
        bs = max(8, min(256, int(16_000_000 // max(1, moves["n_entries"]))))
        for mv in moves["moves"]:
            span = int(mv["span"])
            blk0 = int(mv["blk0"])
            got, n = 0.0, 0
            try:
                with torch.no_grad():
                    for a in range(0, corpus["dp_x"].shape[0], bs):
                        xb = corpus["dp_x"][a:a + bs].long()
                        fe, _ = MC.macro_features(core, xb, mv, s, v)
                        tgt = corpus["dp_exact"][a:a + bs, blk0:blk0 + span]
                        got += float((fe == tgt).double().sum().item())
                        n += int(fe.numel())
            except RuntimeError as e:                    # e.g. OOM at the top rung
                log(f"      dp level {level} node {mv['node']}: skipped ({e.__class__.__name__})")
                continue
            if n:
                accs.append(got / n)
                ns += n
        if accs:
            dp[int(level)] = {"acc": float(np.mean(accs)), "n": ns, "nodes": len(accs)}
    out["dp_parse"] = dp
    return out


# --------------------------------------------------------------------------------------- #
# the per-seed job
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=12288)
def probe_seed(seed_key, out_tag, smoke=False, dedup=False,
               shape_levels="", modes="", steps=""):
    import resource
    import torch
    import rhm.practice.native.span.span_net as SN
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
    from rhm.rhm_sculpt_planner import _sample_pool
    from rhm.practice.voicing.voicing import build_critic, ov_irls

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    spec = SEEDS[seed_key]
    root = os.path.join(DATA_DIR, BANK, spec["tag"], spec["arm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 100)
    log(f"[duplex] seed {seed_key}: {spec['tag']}/{spec['arm']}   device {device}"
        f"   smoke={bool(smoke)}  dedup={bool(dedup)}")
    log("=" * 100)

    blob = torch.load(os.path.join(root, "vo_heads.pt"), map_location="cpu", weights_only=True)
    cfgh = blob["cfg"]
    cfgr = json.load(open(os.path.join(root, "results.json")))["config"]
    v, s, depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
    dim, maxl = int(cfgh["state_dim"]), int(cfgh["max_macro_level"])
    length = s ** depth
    n_blocks = length // s
    thr = int(round(float(cfgh["vo_critic_hold"]) * 100))
    cfg = {"v": v, "s": s, "depth": depth, "state_dim": dim, "max_macro_level": maxl,
           "m": int(cfgr["m"]), "rule_seed": int(cfgr["rule_seed"]),
           "train_seed": int(cfgr["train_seed"]), "seed": int(cfgr["seed"]),
           "gen_lr": float(cfgr["gen_lr"]), "vo_critic_hold": float(cfgh["vo_critic_hold"])}
    log(f"  cfg v{v} s{s} L{depth} m{cfg['m']} rule_seed {cfg['rule_seed']} dim {dim} "
        f"maxl {maxl}  hold thr {thr}  gen_lr {cfg['gen_lr']}")

    rules = generate_rules_distinct(v, s, depth, cfg["m"], seed=cfg["rule_seed"])
    inverse_maps = build_inverse_maps(rules)
    inverse_bottom = inverse_maps[-1]
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    bottom_map = torch.from_numpy(inverse_bottom).to(device)

    def mk_core():
        return GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                     root_conditioned=False).to(device)

    frozen = mk_core()
    frozen.load_state_dict(blob["core"])
    frozen.eval()
    # CONTROL TWO, overtone's own: the same architecture, never trained, minted under
    # `manual_seed(20260915)` AFTER the trained core is loaded — the order matters for the
    # draw, and it is overtone's order.
    torch.manual_seed(OV_RAND_SEED)
    rand = mk_core()
    rand.eval()
    hm = int(cfgh.get("ov_critic_hidden", -1))
    critic = build_critic(SN.slot_count(s, depth, maxl), v, dim, s ** (maxl - 1), seed=0,
                          device=device,
                          hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
    critic.load_state_dict(blob["critic"])
    critic.eval()
    log(f"  trunk fingerprints: frozen {fingerprint(frozen):.6f}  rand {fingerprint(rand):.6f}")

    # ---- the banked rows -------------------------------------------------------------- #
    z = np.load(os.path.join(root, "vo_rows.npz"))
    meta = json.load(open(os.path.join(root, "vo_rows_meta.json")))
    keys = sorted(meta, key=lambda k: (k.split(":")[0], int(k.split(":")[1]),
                                       int(k.split(":")[2])))
    if smoke:
        keys = [k for k in keys if k.split(":")[1] in ("2", "5") and
                k.split(":")[2] in ("0", "1")]
    layout = {"n_blocks": n_blocks, "canon": canon,
              "n_slots": SN.slot_count(s, depth, maxl)}
    bufs, use_keys = {}, []
    for k in keys:
        mt = meta[k]
        which = k.split(":", 1)[0]
        obs = torch.from_numpy(z[f"{k}|obs"].astype(np.int64))
        n = int(obs.shape[0])
        y = torch.from_numpy(z[f"{k}|y"].astype(np.float64))
        code = z[f"{k}|code"].astype(np.int64)
        ho = torch.from_numpy(code < thr)
        if n < 128 or int(ho.sum()) < 16 or float(y.min()) == float(y.max()):
            continue
        uh = None
        uk = f"{k}|unif"
        if which == "probe" and uk in z:
            um = z[uk] > 0.5
            if um.any() and (~um).any():
                uh = torch.from_numpy(um)
        bufs[k] = {"obs": obs, "write": torch.from_numpy(z[f"{k}|write"].astype(np.int64)),
                   "y": y, "ho": ho, "tr": ~ho, "unif": uh,
                   "dp": torch.from_numpy(z[f"{k}|dp"].astype(np.float64)),
                   "tr_idx": torch.nonzero(~ho).flatten()}
        layout[k] = {"blk0": int(mt["blk0"]), "span": int(mt["span"]),
                     "slot_id": int(mt["slot_id"]), "level": int(mt["level"]),
                     "node": int(mt["node"]), "which": which, "slot": k.split(":", 1)[1]}
        use_keys.append(k)
    layout["keys"] = use_keys
    log(f"  buffers {len(use_keys)} of {len(keys)}   "
        f"rows {sum(int(bufs[k]['obs'].shape[0]) for k in use_keys)}")

    # ---- THE EXACT-DUPLICATE CHANNEL, measured and optionally closed ------------------ #
    # For a FILED row the post-write configuration IS the trajectory's final configuration,
    # and `hold_code` hashes the PRE-write context (which differs by which span is masked).
    # So one trajectory that wrote at several slots contributes several rows carrying the SAME
    # post-write input and the SAME verdict, and the audit's split can put them on opposite
    # sides. Anything trained on the post-write input therefore sees some held-out inputs
    # verbatim. This is a property of the arc's own split, inherited, not introduced here —
    # the in-loop critic's filed audit has it too — so it is MEASURED on every run and CLOSED
    # only when `dedup` is on, where the filter drops from the shaping diet AND from the
    # probe's own fit every training row whose post-write configuration appears among the
    # held-out rows of any buffer.
    hashes, ho_set = {}, set()
    for k in use_keys:
        b, lay = bufs[k], layout[k]
        pos = ((np.arange(lay["blk0"], lay["blk0"] + lay["span"])[:, None] * s)
               + np.arange(s)[None, :]).reshape(-1)
        x2 = b["obs"].numpy().copy()
        x2[:, pos] = np.ascontiguousarray(
            rules[depth - 1][:, 0, :])[b["write"].numpy()].reshape(x2.shape[0], -1)
        hh = [r.tobytes() for r in x2.astype(np.int8)]
        hashes[k] = hh
        hoa = b["ho"].numpy()
        ho_set.update(h for i, h in enumerate(hh) if hoa[i])
    ov_stat = {}
    for which in ("filed", "probe"):
        n_ho = n_hit = 0
        tr_all = set()
        for k in use_keys:
            if layout[k]["which"] != which:
                continue
            tra = bufs[k]["tr"].numpy()
            tr_all.update(h for i, h in enumerate(hashes[k]) if tra[i])
        for k in use_keys:
            if layout[k]["which"] != which:
                continue
            hoa = bufs[k]["ho"].numpy()
            for i, h in enumerate(hashes[k]):
                if hoa[i]:
                    n_ho += 1
                    n_hit += int(h in tr_all)
        ov_stat[which] = {"n_hold": n_ho, "n_in_train": n_hit,
                          "share": n_hit / max(n_ho, 1)}
        log(f"  overlap {which}: {n_hit}/{n_ho} held-out rows ({n_hit / max(n_ho, 1):.3%}) "
            f"have their exact post-write configuration among the training rows")
    n_drop = 0
    for k in use_keys:
        b = bufs[k]
        keepm = np.array([h not in ho_set for h in hashes[k]], bool)
        trf = b["tr"].numpy() & keepm
        n_drop += int(b["tr"].numpy().sum() - trf.sum())
        b["trf"] = torch.from_numpy(trf)
        b["trf_idx"] = torch.nonzero(b["trf"]).flatten()
    log(f"  dedup filter would drop {n_drop} training rows "
        f"({n_drop / max(sum(int(bufs[k]['tr'].sum()) for k in use_keys), 1):.3%}); "
        f"dedup={bool(dedup)}")
    if dedup:
        for k in use_keys:
            bufs[k]["tr"] = bufs[k]["trf"]
            bufs[k]["tr_idx"] = bufs[k]["trf_idx"]

    # [duplex/du2] THE SHAPING DIET'S OWN SLOT RESTRICTION. With `shape_levels` empty the
    # outcome loss hears every slot, which is `du0`/`du1`: the trunk has then already seen L4/L5
    # verdicts by the time the L2/L3-fit readout is scored at L4/L5, so section (d) tests the
    # READOUT's direction and not the representation's. With `shape_levels = "2,3"` the outcome
    # loss hears the L2/L3 slots ONLY — the levels every seed reaches — and (d) becomes the test
    # of whether an outcome heard there teaches a direction that carries to the frontier.
    # The probe's fit, the held-out rows, the infill stream and every diagnostic are untouched.
    lv_want = {int(x) for x in shape_levels.split(",") if x.strip()}
    layout["shape_keys"] = ([k for k in use_keys if layout[k]["level"] in lv_want]
                            if lv_want else list(use_keys))
    log(f"  shaping diet: {len(layout['shape_keys'])} of {len(use_keys)} buffers"
        + (f"  (levels {sorted(lv_want)})" if lv_want else "  (every slot)"))

    # ---- fresh corpus, the diagnostics' fixed plan ------------------------------------ #
    n_tr = 2048 if smoke else N_CORPUS_TRAIN
    n_ev = 1024 if smoke else N_CORPUS_EVAL
    _, lv_tr = _sample_pool(rules, n_tr, s, CORPUS_SEED_TRAIN)
    _, lv_ev = _sample_pool(rules, n_ev, s, CORPUS_SEED_EVAL)
    corpus = {"train": torch.from_numpy(lv_tr).to(device),
              "eval": torch.from_numpy(lv_ev).to(device),
              "bottom_map": bottom_map}
    grng = torch.Generator(device=device).manual_seed(WM_SEED)
    nb_batches = 4 if smoke else WM_INFILL_BATCHES
    bsz = 128 if smoke else WM_INFILL_BATCH
    plan = []
    for i in range(nb_batches):
        idx = torch.randint(0, corpus["eval"].shape[0], (bsz,), generator=grng, device=device)
        nm = int(torch.randint(1, n_blocks + 1, (), generator=grng, device=device).item())
        order = torch.rand(bsz, n_blocks, generator=grng, device=device).argsort(dim=1)
        bucket = ("1" if nm == 1 else "2-4" if nm <= 4 else "5-8" if nm <= 8 else
                  "9-16" if nm <= 16 else "17-32")
        plan.append({"idx": idx, "mb": order[:, :nm], "n_mask": nm, "bucket": bucket})
    corpus["infill_plan"] = plan
    npar = 512 if smoke else WM_PARSE_N
    corpus["parse_x"] = corpus["eval"][:npar]
    corpus["parse_exact"] = torch.from_numpy(
        MC.exact_features(lv_ev[:npar], inverse_bottom, v, s)).to(device)
    ndp = 256 if smoke else WM_DP_N
    corpus["dp_x"] = corpus["eval"][:ndp]
    corpus["dp_exact"] = torch.from_numpy(
        MC.exact_features(lv_ev[:ndp], inverse_bottom, v, s)).to(device)
    tt = MC.true_tables(rules, depth, s, v, cfg["m"], maxl)
    corpus["dp_moves"] = {}
    for level in range(2, maxl + 1):
        mvs = []
        for node in (0, 1):
            mv = MC.make_macro(level, node, s, tt[level])
            mvs.append(MC.to_device(mv, device))
        corpus["dp_moves"][level] = {"moves": mvs,
                                     "n_entries": int(tt[level]["child"].shape[0])}
    log("  true-table rows per level: " + "  ".join(
        f"L{l}:{corpus['dp_moves'][l]['n_entries']}" for l in sorted(corpus["dp_moves"])))
    log(f"  fresh corpus: train {n_tr}  eval {n_ev}  infill plan "
        f"{[p['n_mask'] for p in plan]}")

    # ---- G0: the estimator gate ------------------------------------------------------- #
    k0 = use_keys[0]
    b0 = bufs[k0]
    ft = trunk_features(frozen, b0["obs"], b0["write"], canon, layout[k0]["blk0"],
                        layout[k0]["span"], s, n_blocks, device)
    Fp = ft["post"]
    trm, hom = b0["tr"].to(device), b0["ho"].to(device)
    yv = b0["y"].to(device)
    sc_t = fit_probe_t(Fp, yv, trm, hom)
    Fn = Fp.detach().cpu().numpy().astype(np.float64)
    yn = b0["y"].numpy()
    hn = b0["ho"].numpy()
    trn = b0["tr"].numpy()          # the SAME rows the torch fit uses (dedup-aware)
    mu, sd = Fn[trn].mean(0), Fn[trn].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(Fn[trn] - mu) / sd, np.ones((int(trn.sum()), 1))], 1)
    B = np.concatenate([(Fn[hn] - mu) / sd, np.ones((int(hn.sum()), 1))], 1)
    w_np = ov_irls(A, yn[trn], ridge=PROBE_RIDGE, iters=PROBE_ITERS)
    sc_np = B @ w_np
    gate = {"key": k0,
            "max_abs_dscore": float(np.abs(sc_t.detach().cpu().numpy() - sc_np).max()),
            "auc_torch": auc_t(sc_t, yv[hom]),
            "auc_numpy": auc_t(torch.from_numpy(sc_np), torch.from_numpy(yn[hn])),
            "max_abs_dw": float(np.abs(
                w_np - irls_t(torch.from_numpy(A), torch.from_numpy(yn[trn])
                              ).numpy()).max())}
    gate["d_auc"] = abs(gate["auc_torch"] - gate["auc_numpy"])
    log(f"  [G0] estimator gate on {k0}: max |Δscore| {gate['max_abs_dscore']:.3e}  "
        f"AUC torch {gate['auc_torch']:.6f} numpy {gate['auc_numpy']:.6f} "
        f"(Δ {gate['d_auc']:.2e})")
    del ft, Fp, Fn, A, B

    # ---- the trunks ------------------------------------------------------------------- #
    steps_list = ((20,) if smoke else
                  (tuple(int(x) for x in steps.split(",") if x.strip()) or SHAPE_STEPS))
    mode_list = tuple(x.strip() for x in modes.split(",") if x.strip()) or \
        ("both", "out", "infill")
    trunks = [("frozen", frozen), ("rand", rand)]
    shape_hist = {}
    for mode in mode_list:
        for st_ in steps_list:
            c = mk_core()
            c.load_state_dict(blob["core"])
            t0 = time.time()
            c, hist = shape_trunk(c, bufs, layout, corpus, mode, st_, SHAPE_LR, device, cfg,
                                  log)
            nm = f"{mode}@{st_}"
            shape_hist[nm] = {"hist": hist, "sec": time.time() - t0}
            log(f"    shaped {nm}: {time.time() - t0:.1f}s  fingerprint {fingerprint(c):.6f}")
            trunks.append((nm, c))

    # ---- the probe, every trunk, identical rows ---------------------------------------- #
    res = {"seed": seed_key, "tag": spec["tag"], "arm": spec["arm"], "cfg": cfg, "thr": thr,
           "gate": gate, "shape": {"lr": SHAPE_LR, "steps": list(steps_list),
                                   "slot_batch": SHAPE_SLOT_BATCH,
                                   "infill_batch": SHAPE_INFILL_BATCH, "hist": shape_hist},
           "trunks": {}, "ref": {}, "dedup": bool(dedup),
           "overlap": ov_stat, "n_dedup_drop": int(n_drop)}
    xf_cap = 2000 if smoke else XFER_CAP
    for nm, core in trunks:
        t0 = time.time()
        core.eval()
        per_buf = []
        groups = {}
        want_mlp = (nm != "rand")
        for k in use_keys:
            b = bufs[k]
            lay = layout[k]
            ft = trunk_features(core, b["obs"], b["write"], canon, lay["blk0"], lay["span"],
                                s, n_blocks, device,
                                critic=(critic if want_mlp else None),
                                slot_id=lay["slot_id"])
            trm = b["tr"].to(device)
            hom = b["ho"].to(device)
            yv = b["y"].to(device)
            sc = {}
            for rd in ("pre", "pmean", "post", "postm"):
                sc[rd] = fit_probe_t(ft[rd], yv, trm, hom)
            if want_mlp:
                sc["mlp"] = ft["mlp"][hom]
            sc["dp"] = b["dp"].to(device)[hom]
            yh = yv[hom]
            rec = {"key": k, "which": lay["which"], "slot": lay["slot"], "level": lay["level"],
                   "node": lay["node"], "n_tr": int(trm.sum()), "n_hold": int(hom.sum()),
                   "base": float(yh.mean()), "auc": {}}
            for rd, v_ in sc.items():
                rec["auc"][rd] = (None if v_ is None else auc_t(v_, yh))
            if b["unif"] is not None:
                uh = b["unif"].to(device)[hom]
                for tag_, sub in (("auc_u", uh), ("auc_d", ~uh)):
                    if int(sub.sum()) >= 16:
                        rec[tag_] = {rd: (None if v_ is None else auc_t(v_[sub], yh[sub]))
                                     for rd, v_ in sc.items()}
            per_buf.append(rec)
            # the transfer split's pools: the READ OF RECORD (post, unmasked), pooled by rung
            grp = "L23" if lay["level"] <= 3 else "L45"
            gk = (lay["which"], grp)
            d = groups.setdefault(gk, {"tr_x": [], "tr_y": [], "ho_x": [], "ho_y": []})
            ti = torch.nonzero(trm).flatten()
            nbuf = sum(1 for kk in use_keys
                       if layout[kk]["which"] == lay["which"] and
                       ("L23" if layout[kk]["level"] <= 3 else "L45") == grp)
            cap = max(64, xf_cap // max(1, nbuf))
            if ti.numel() > cap:
                gsel = torch.Generator(device="cpu").manual_seed(31337 + lay["slot_id"])
                ti = ti[torch.randperm(ti.numel(), generator=gsel)[:cap].to(ti.device)]
            d["tr_x"].append(ft["post"][ti].double())
            d["tr_y"].append(yv[ti])
            d["ho_x"].append(ft["post"][hom].double())
            d["ho_y"].append(yh)
            del ft, sc
        # the transfer split
        xfer = {}
        for which in ("filed", "probe"):
            g23, g45 = groups.get((which, "L23")), groups.get((which, "L45"))
            if g23 is None or g45 is None:
                continue
            X23 = torch.cat(g23["tr_x"]); y23 = torch.cat(g23["tr_y"])
            H23 = torch.cat(g23["ho_x"]); v23 = torch.cat(g23["ho_y"])
            X45 = torch.cat(g45["tr_x"]); y45 = torch.cat(g45["tr_y"])
            H45 = torch.cat(g45["ho_x"]); v45 = torch.cat(g45["ho_y"])
            n_fit = min(X23.shape[0], X45.shape[0])
            pg = torch.Generator(device="cpu").manual_seed(4242)
            i23 = torch.randperm(X23.shape[0], generator=pg)[:n_fit].to(X23.device)
            i45 = torch.randperm(X45.shape[0], generator=pg)[:n_fit].to(X45.device)
            X23, y23 = X23[i23], y23[i23]
            X45, y45 = X45[i45], y45[i45]

            def _fit_score(Xf, yf, evals):
                mu_, sd_ = Xf.mean(0), Xf.std(0)
                sd_ = torch.where(sd_ < 1e-9, torch.ones_like(sd_), sd_)
                one = torch.ones(Xf.shape[0], 1, dtype=torch.float64, device=Xf.device)
                A_ = torch.cat([(Xf - mu_) / sd_, one], 1)
                w_ = irls_t(A_, yf)
                out_ = {}
                for lbl, (Xe, ye) in evals.items():
                    Be = torch.cat([(Xe - mu_) / sd_,
                                    torch.ones(Xe.shape[0], 1, dtype=torch.float64,
                                               device=Xe.device)], 1)
                    out_[lbl] = {"auc": auc_t(Be @ w_, ye), "n": int(Xe.shape[0]),
                                 "base": float(ye.mean())}
                return out_

            a = _fit_score(X23, y23, {"L23": (H23, v23), "L45": (H45, v45)})
            b_ = _fit_score(X45, y45, {"L45": (H45, v45)})
            xfer[which] = {"n_fit": int(n_fit),
                           "fit23_on23": a["L23"], "fit23_on45": a["L45"],
                           "fit45_on45": b_["L45"]}
        wm = wm_diag(core, corpus, layout, cfg, device, log)
        res["trunks"][nm] = {"fingerprint": fingerprint(core), "per_buf": per_buf,
                             "xfer": xfer, "wm": wm, "sec": time.time() - t0}
        med = {}
        for rec in per_buf:
            for lbl, dd in (("", rec["auc"]), ("/u", rec.get("auc_u")),
                            ("/d", rec.get("auc_d"))):
                if dd is None:
                    continue
                for rd, val in dd.items():
                    if val is not None:
                        med.setdefault(rec["which"] + lbl, {}).setdefault(rd, []).append(val)
        res["trunks"][nm]["median"] = {w: {rd: float(np.median(vals)) for rd, vals in d.items()}
                                       for w, d in med.items()}
        log(f"  [{nm}] {time.time() - t0:.1f}s  " + "  ".join(
            f"{w}: " + " ".join(f"{rd} {vv:.3f}" for rd, vv in sorted(d.items()))
            for w, d in sorted(res["trunks"][nm]["median"].items())))
        log(f"        wm: infill_ce {wm['infill_ce']:.4f} acc {wm['infill_acc']:.3f}  "
            f"parse(mask1) {wm['parse_acc_mask1']:.3f} parse(nomask) "
            f"{wm['parse_acc_nomask']:.3f}  dp " + " ".join(
                f"L{l}:{d['acc']:.3f}" for l, d in sorted(wm["dp_parse"].items())))
        if nm != "rand":
            del core

    res["sec"] = time.time() - t00
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    res["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log(f"  done in {res['sec']:.1f}s   peak RSS {res['peak_rss_mb']:.0f} MB   "
        f"peak GPU {res['peak_gpu_mb']:.0f} MB")
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{seed_key}.json"), "w") as fh:
        json.dump(res, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{seed_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    volume.commit()
    return {"seed": seed_key, "sec": res["sec"], "rss_mb": res["peak_rss_mb"],
            "gate": gate}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=12600, memory=2048)
def sweep(out_tag="du0", smoke=False, seeds="", dedup=False,
          shape_levels="", modes="", steps=""):
    """CPU coordinator: the two banked seeds across two containers. No GPU-hours are added by
    the fan-out — the seeds are independent and were always going to be two passes."""
    want = [x for x in seeds.split(",") if x] or list(SEEDS)
    args = [(k, out_tag, bool(smoke), bool(dedup), shape_levels, modes, steps)
            for k in want]
    outs = list(probe_seed.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"seeds": want, "smoke": bool(smoke),
                             "dedup": bool(dedup), "shape_levels": shape_levels,
                             "modes": modes, "steps": steps}) + "\n")
    volume.commit()
    return outs
