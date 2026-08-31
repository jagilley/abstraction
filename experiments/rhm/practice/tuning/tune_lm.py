"""tuning/tune_lm — Gate 0: the OOD-typing shadow panel, both factors, uncharged.

SPEC: `SPEC.md` in this folder. This module builds ONLY Gate 0.

WHAT THIS RUNS. `wall` and `no_wall` trained in LOCKSTEP on a burst-augmented stream, with a
panel of candidate gauges computed at every 125-step checkpoint and ACTED ON BY NOTHING:

    Factor T (the typing gauge, read at the event instant)
        nll        the keyed reader's own indexed-span NLL          [the I/O-public twin]
        D_pair     KL( p_key || p_parse ), two readers              [two models]
        D_self     KL( p_key || p_key-with-the-key-neutralised )    [one model, two conditions]

    Factor M (the meter, read over the window after)
        e_online   || h6 - FM(h0) || for a FM learning alongside the reader
        e_frozen   the same for a FM frozen one checkpoint back, and for a FM frozen at the
                   last checkpoint before the current event
        b, delta   b = EWMA(e); delta = b - e, the performance-error object

THE COUNTERFACTUAL-INSTANT DESIGN (this node's main design choice). The spec's Factor-T
question is whether the gauges separate a rotation from a burst AT MATCHED indexed-span
surprisal. Rather than compare a rotation checkpoint against a burst checkpoint — different
steps, different reader maturity, different confounds — every checkpoint evaluates BOTH
events counterfactually on the SAME weights:

    rotation instant   the eval wall token rotated by +rot_step (the donor's `perm` mode)
    burst instant      the eval leaves corrupted at rate rho, for a LADDER of rho

The donor validates this: at fwlm0/`wall` s7750 the counterfactual `misleading_idx` reads
+0.5377 and the realised rotation spike 250 steps later is +0.5485 — the counterfactual IS
the event instant, to 0.011 nats. The ladder then lets the reduction interpolate, per
checkpoint, the rho whose indexed-span spike EQUALS that checkpoint's rotation spike, and
read D_pair / D_self there. Matched surprisal by construction, at every checkpoint, with no
dependence on the consumed burst rate being perfectly calibrated.

The CONSUMED burst rate (`burst_rho`, calibrated by `calib`) is what the Factor-M rows need:
a burst the learner actually trains on. Shadow bursts (evaluated, never consumed) supply the
input-invariance half.

TRANSPARENCY. Three gates, all run before the main launch:
  (a) the panel moves nothing      `--shadow` vs `--no-shadow`, max|delta| = 0.0
  (b) the attached FM moves nothing `--fm` vs `--no-fm`, max|delta| = 0.0
  (c) donor fidelity                pre-burst checkpoints vs the fwlm0 / fwlm1 / tsdB twins
Mechanically: the FM reads activations under `no_grad` (detached before the reader's
backward), owns its optimiser, and its CONSTRUCTION is wrapped in a global-RNG save/restore
so it consumes zero net global torch RNG. Two readers share one worker, so each carries its
OWN global-RNG state, swapped in around every operation that consumes it (model construction
and probe fitting — nothing else in the donor's loop touches the global stream). That is
what makes co-residency bit-identical to the donor's one-worker-per-arm design, and gate (c)
is the arbiter.

Run from experiments/:
  modal run -m rhm.practice.tuning.tune_lm::gate
  modal run -m rhm.practice.tuning.tune_lm::calib
  modal run -m rhm.practice.tuning.tune_lm::tune_lm --quick --tag tn_smoke
  python3 rhm/practice/tuning/launch_detached.py --fn tune_lm --tag tn0 --log-dir /tmp
"""

import json
import os
import time

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, setting_key, volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-practice-tuning", image=image)

REMOTE = "rhm_practice_tuning"
DONOR_REMOTE = "rhm_practice_fourwall_lm"          # read-only: refs + the gate twins
REF_KEYS = ("v", "s", "depth", "m", "rule_seed", "key_level", "key_node",
            "n_eval", "eval_seed", "n_probe", "probe_seed")

# reader -> (wall kind, phase1 override, rot_period override, seed offset, merge step,
#            skips bursts, donor index panel + probes, how many FM lr variants)
READER_SPECS = {
    "wall":      ("wall", None, None, 0, None, False, True,  4),
    "no_wall":   ("none", None, 0,    0, None, False, True,  1),
    # the "burst -> skip" row: identical to `wall` in every respect except that the
    # optimiser takes no step at all inside a burst window (lr 0, no Adam state update).
    # The batch index is still drawn, so its sampler stream stays aligned with `wall`'s
    # and the two are bit-identical up to the first burst.
    "wall_skip": ("wall", None, None, 0, None, True,  False, 4),
}
WORKER_SPECS = {"pair": ["wall", "no_wall"], "skip": ["wall_skip"]}

FM_LRS = (3e-4, 1e-3, 3e-3, 1e-2)      # the one FM sweep the spec asks for
FM_PRIMARY = 1e-3                      # conditional_revision's `fwd_lr`


def reader_schedule(reader, phase1, rot_period, max_steps):
    kind, p1, rp, soff, mg, skip, panel, n_fm = READER_SPECS[reader]
    p1 = phase1 if p1 is None else p1
    rp = rot_period if rp is None else rp
    if rp <= 0:
        p1 = max_steps + 1                                  # never rotates
    return {"kind": kind, "phase1": int(p1), "rot_period": int(rp),
            "seed_off": int(soff), "merge_at": int(max_steps + 1 if mg is None else mg),
            "skip_bursts": bool(skip), "donor_panel": bool(panel), "n_fm": int(n_fm)}


def load_refs(cfg, donor_tag="fwlm1", verbose=True):
    """Reuse the donor's exact BP references verbatim when the config matches.

    `excess` is measured against `bayes_none`, so sharing the donor's array is what makes
    the `excess` instruments bit-comparable with the twins (endo_yield's idiom).
    """
    path = f"{DATA_DIR}/{DONOR_REMOTE}/{donor_tag}/setup.json"
    if os.path.exists(path):
        setup = json.load(open(path))
        d = setup["config"]
        if all(d.get(k) == cfg[k] for k in REF_KEYS):
            if verbose:
                print(f"[refs] loaded verbatim from {DONOR_REMOTE}/{donor_tag}/setup.json",
                      flush=True)
            return setup["refs"], f"{donor_tag}:setup.json"
        if verbose:
            print(f"[refs] donor config mismatch: "
                  f"{[k for k in REF_KEYS if d.get(k) != cfg[k]]} -> recomputing", flush=True)
    from rhm.practice.fourwall.lm.wall_lm import build_refs
    refs = build_refs(cfg["v"], cfg["s"], cfg["depth"], cfg["m"], cfg["rule_seed"],
                      cfg["key_level"], cfg["key_node"], cfg["n_eval"], cfg["eval_seed"],
                      cfg["n_probe"], cfg["probe_seed"], verbose=verbose)
    return refs, "recomputed"


# --------------------------------------------------------------------------- #
# one worker = one or more CO-RESIDENT readers, each with its own global-RNG state
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=32400, memory=32768)
def run_worker(tag: str, worker: str, scheds: dict, refs: dict, cfg: dict):
    import copy

    import numpy as np
    import torch
    import torch.nn.functional as F
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    T = s ** L
    V = W.vocab_size(v)
    rot_step, key_level, key_node = cfg["rot_step"], cfg["key_level"], cfg["key_node"]
    B = cfg["batch_size"]
    key_lo, key_hi = refs["key_lo"], refs["key_hi"]
    key_plen, last_plen = refs["key_plen"], refs["last_plen"]
    max_steps = cfg["max_steps"]
    shadow, use_fm = bool(cfg["shadow"]), bool(cfg["fm"])
    span = key_hi - key_lo

    rules = generate_rules_distinct(v, s, L, m, seed=cfg["rule_seed"])
    bayes_none = np.array(refs["bayes_none"])
    ptl = np.array(refs["pos_top_level"])
    P0_exact = np.array(refs["exact_leaf0"])

    # ---- evaluation apparatus, identical in every arm (the donor's, verbatim) ----
    ev_leaf, ev_lf, _ = _generate_with_traces(rules, cfg["n_eval"], cfg["eval_seed"])
    ev_z = ev_lf[key_level][:, key_node]
    ev_leaf_t = torch.from_numpy(ev_leaf.astype(np.int64))
    pr_leaf, pr_lf, _ = _generate_with_traces(rules, cfg["n_probe"], cfg["probe_seed"])
    pr_z = pr_lf[key_level][:, key_node]
    pr_leaf_t = torch.from_numpy(pr_leaf.astype(np.int64))
    anc = {"key": W.anc_index(key_plen, L, s), "last": W.anc_index(last_plen, L, s)}
    y_lvl = {a: {ell: torch.from_numpy(pr_lf[ell][:, anc[a][ell]].astype(np.int64)).to(device)
                 for ell in range(L)} for a in anc}
    MODES = ["true", "rand", "perm", "old", "none"]

    all_blocks = ["post_embed"] + [f"post_block{i}" for i in range(cfg["n_layer"])]
    pblocks = [b for b in cfg["probe_blocks"].split(",") if b in all_blocks] or all_blocks

    # ---- the burst schedule and the shadow ladder ----
    starts = [int(x) for x in str(cfg["burst_steps"]).split(",") if str(x).strip()]
    windows = BU.burst_windows(starts, cfg["burst_len"], max_steps)
    onsets = BU.burst_onsets(windows)
    # ONE RATE PER WINDOW. A rotation's spike grows with the scaffold's age (+0.085 at
    # s2000, +0.32 at s5000, +0.55 at s8000 on the donor), so a single fixed rate cannot
    # be magnitude-matched at three different onsets. `calib` measures the matched rate at
    # each onset's maturity and they are hard-coded here.
    rhos_c = [float(x) for x in str(cfg["burst_rho"]).split(",") if str(x).strip()]
    if len(rhos_c) == 1 and len(windows) > 1:
        rhos_c = rhos_c * len(windows)
    assert len(rhos_c) >= len(windows), "burst_rho must give one rate per window"
    rho_ref = rhos_c[0]
    ladder = [float(x) for x in str(cfg["burst_ladder"]).split(",") if str(x).strip()]

    # the panel's own (small) eval subset, and its FIXED shadow corruptions
    n_panel = min(cfg["n_panel"], ev_leaf.shape[0])
    pn_leaf = ev_leaf[:n_panel]
    pn_z = ev_z[:n_panel]
    pn_leaf_t = torch.from_numpy(pn_leaf.astype(np.int64))
    pn_burst_t, pn_burst_rate = [], []
    for i, rho in enumerate(ladder):
        cor, mask = BU.corrupt(pn_leaf, key_lo, key_hi, rho, v, BU.shadow_burst_rng(
            cfg["eval_seed"], i))
        pn_burst_t.append(torch.from_numpy(cor.astype(np.int64)))
        pn_burst_rate.append(float(mask.mean()))
    # the consumed rate's own shadow copy (index -1 in the ladder namespace)
    cor_c, mask_c = BU.corrupt(pn_leaf, key_lo, key_hi, rho_ref, v,
                               BU.shadow_burst_rng(cfg["eval_seed"], 991))
    pn_bc_t = torch.from_numpy(cor_c.astype(np.int64))

    # --------------------------------------------------------------------- #
    # helpers (the donor's, lifted so several co-resident readers can share them)
    # --------------------------------------------------------------------- #

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    def wcol(kind, seed_off, leaf_np, z_np, q, q_prev, mode, mode_i):
        """The donor's `wcol`, verbatim in behaviour."""
        if mode == "none":
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        rng = np.random.default_rng(cfg["eval_seed"] + 7717 * (mode_i + 1)
                                    + 131 * seed_off)
        w = W.wall_values(kind, z_np, q, v, rng=rng, mode=mode,
                          q_prev=q_prev, delta=rot_step)
        if w is None:
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        return torch.from_numpy((W.wall_tok(w, v)).astype(np.int64))

    @torch.no_grad()
    def nll_per_pos(model, leaf_t, wc, chunk=256):
        tot = torch.zeros(T, device=device)
        for i in range(0, leaf_t.shape[0], chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            tot += nll.sum(0)
        return (tot / leaf_t.shape[0]).cpu().numpy()

    @torch.no_grad()
    def wall_conditionals(model, kind):
        if kind == "none":
            ids = torch.full((1, 1), W.neutral_tok(v), dtype=torch.long, device=device)
        else:
            ids = (torch.arange(v, device=device) + v)[:, None]
        logits, _ = model(ids)
        p = torch.softmax(logits[:, 0, :v].float(), -1).cpu().numpy()
        return np.repeat(p, v, axis=0) if kind == "none" else p

    z_groups = [np.where(ev_z == a)[0] for a in range(v)]
    live = np.array([len(g) >= cfg["min_cell_n"] for g in z_groups])
    card_sel = torch.from_numpy(np.arange(min(cfg["n_card"], ev_leaf.shape[0])))

    @torch.no_grad()
    def transfer_and_cardinality(model, kind, chunk=256):
        """The donor's forced-transfer + MDL sweep.

        DEDUPE, exact: when `kind == "none"` every one of the 16 wall ids maps to the SAME
        neutral token, so all 16 passes are literally the same forward on the same input.
        Row 0 is computed and broadcast; every downstream number (Ei, Eo, D) is therefore
        bit-identical to the donor's 16-pass version, at 1/16 the cost. Verified by the
        fidelity gate against fwlm0/fwlm1/ey0 `no_wall`.
        """
        n = ev_leaf_t.shape[0]
        nc = card_sel.shape[0]
        Ei = np.zeros((v, v)); Eo = np.zeros((v, v))
        Pc = torch.zeros(v, nc, key_hi - key_lo, v, device=device)
        ids = [0] if kind == "none" else list(range(v))
        for i in ids:
            tok = W.neutral_tok(v) if kind == "none" else W.wall_tok(i, v)
            wc = torch.full((n,), tok, dtype=torch.long)
            si = torch.zeros(n); so = torch.zeros(n)
            for j in range(0, n, chunk):
                x = make_x(ev_leaf_t[j:j + chunk], wc[j:j + chunk]).to(device)
                y = ev_leaf_t[j:j + chunk].to(device)
                logits, _ = model(x)
                nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                      reduction="none").reshape(y.shape)
                si[j:j + chunk] = nll[:, key_lo:key_hi].mean(1).cpu()
                so[j:j + chunk] = nll[:, key_hi:].mean(1).cpu()
                lo_c, hi_c = j, min(j + chunk, nc)
                if lo_c < nc:
                    Pc[i, lo_c:hi_c] = torch.softmax(
                        logits[:hi_c - lo_c, key_lo:key_hi, :v].float(), -1)
            sin, son = si.numpy(), so.numpy()
            for a in range(v):
                g = z_groups[a]
                if len(g):
                    Ei[i, a] = sin[g].mean(); Eo[i, a] = son[g].mean()
        if kind == "none":
            for i in range(1, v):
                Ei[i] = Ei[0]; Eo[i] = Eo[0]; Pc[i] = Pc[0]
        D = np.zeros((v, v))
        lP = torch.log(Pc.clamp_min(1e-12))
        for i in range(v):
            M = 0.5 * (Pc[i][None] + Pc)
            lM = torch.log(M.clamp_min(1e-12))
            d = 0.5 * ((Pc[i][None] * (lP[i][None] - lM)).sum(-1)
                       + (Pc * (lP - lM)).sum(-1))
            D[i] = d.mean(dim=(1, 2)).cpu().numpy()
        return Ei, Eo, np.maximum(D, 0.0)

    def acts_at(model, leaf_t, wc, pos, blocks, chunk=256):
        out = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, leaf_t.shape[0], chunk):
                x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
                _, _, inter = model(x, return_intermediates=True)
                for b in blocks:
                    out[b].append(inter[b][:, pos, :].float())
        return {b: torch.cat(vs) for b, vs in out.items()}

    def probe_levels(model, wc, anchor, pos, blocks, full):
        a = acts_at(model, pr_leaf_t, wc, pos, blocks)
        res = {}
        for ell in range(L):
            best = 0.0
            for b in blocks:
                best = max(best, _probe_acc(a[b], y_lvl[anchor][ell], v, device,
                                            cfg["probe_steps"], cfg["probe_lr"]))
                if full:
                    best = max(best, _probe_acc(a[b], y_lvl[anchor][ell], v, device,
                                                cfg["mlp_steps"], cfg["probe_lr"],
                                                hidden=cfg["mlp_hidden"]))
            res[f"d{L - ell}"] = float(best)
        return res

    # ---------------- the new panel's primitives ---------------- #

    @torch.no_grad()
    def panel_read(model, leaf_t, wc, chunk=256):
        """(nll_idx, nll_out, probs (n, span, V)) in one pass."""
        n = leaf_t.shape[0]
        P = torch.empty(n, span, V, device=device)
        si = torch.zeros(n); so = torch.zeros(n)
        for i in range(0, n, chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            si[i:i + chunk] = nll[:, key_lo:key_hi].mean(1).cpu()
            so[i:i + chunk] = nll[:, key_hi:].mean(1).cpu()
            P[i:i + chunk] = torch.softmax(logits[:, key_lo:key_hi, :].float(), -1)
        return float(si.mean()), float(so.mean()), P

    def kl(P, Q):
        lp = torch.log(P.clamp_min(1e-12))
        lq = torch.log(Q.clamp_min(1e-12))
        return float((P * (lp - lq)).sum(-1).mean())

    def jsd(P, Q):
        M = 0.5 * (P + Q)
        return float(0.5 * (kl(P, M) + kl(Q, M)))

    @torch.no_grad()
    def fm_residual(model, fms, leaf_t, wc, chunk=256):
        """{name: (mse, rel)} — || h6 - FM(h0) ||^2 on the indexed span, and the same
        divided by mean h6^2 (the reader's activation scale drifts during training, so a
        raw residual that rose only because ||h6|| grew would be a confound)."""
        acc = {k: 0.0 for k in fms}
        tgt, cnt = 0.0, 0
        for i in range(0, leaf_t.shape[0], chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            _, _, inter = model(x, return_intermediates=True)
            h0 = inter[cfg["fm_shallow"]]
            h6 = inter[cfg["fm_deep"]][:, key_lo:key_hi, :]
            for k, fm in fms.items():
                pred = fm(h0)[:, key_lo:key_hi, :]
                acc[k] += float(((pred - h6) ** 2).sum())
            tgt += float((h6 ** 2).sum())
            cnt += h6.numel()
        return {k: {"mse": acc[k] / cnt, "rel": acc[k] / max(tgt, 1e-12)} for k in acc}

    # --------------------------------------------------------------------- #
    # the readers
    # --------------------------------------------------------------------- #

    class Reader:
        """One arm. Carries its OWN global-RNG state so co-residency is invisible."""

        def __init__(self, name, sched):
            self.name = name
            self.sched = sched
            self.kind = sched["kind"]
            self.phase1, self.rot_period = sched["phase1"], sched["rot_period"]
            self.seed_off = sched["seed_off"]
            self.merge_at = sched["merge_at"]
            self.skip_bursts = sched["skip_bursts"]
            self.donor_panel = sched["donor_panel"]
            self.seed = cfg["seed"] + self.seed_off
            ambient = torch.get_rng_state()
            torch.manual_seed(self.seed)
            self.model = GPT(V, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
            self.rng_state = torch.get_rng_state()      # exactly the donor's stream position
            torch.set_rng_state(ambient)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"])
            self.gen = torch.Generator().manual_seed(self.seed)
            self.train_rng = np.random.default_rng(self.seed + 5551)
            self.log = []
            # --- the FMs: constructed OUTSIDE the reader's stream, RNG-neutral ---
            self.fms, self.fopts = {}, {}
            self.fm_prev, self.fm_event = None, None
            if use_fm:
                lrs = FM_LRS[:sched["n_fm"]] if sched["n_fm"] > 1 else (FM_PRIMARY,)
                amb = torch.get_rng_state()
                proto = TransformerForwardModel(
                    d_model=cfg["n_embd"], d_head=cfg["fm_d_head"],
                    n_head=cfg["fm_n_head"], n_layer=cfg["fm_n_layer"],
                    mlp_mult=cfg["fm_mlp_mult"], block_size=T).to(device)
                torch.set_rng_state(amb)                # zero net global RNG consumed
                for lr in lrs:
                    k = f"lr{lr:g}"
                    self.fms[k] = copy.deepcopy(proto)  # identical init; lr is the only lever
                    self.fopts[k] = torch.optim.AdamW(self.fms[k].parameters(), lr=lr,
                                                      weight_decay=cfg["weight_decay"])
                del proto
                self.primary = f"lr{FM_PRIMARY:g}"
                self.fm_prev = copy.deepcopy(self.fms[self.primary])
                self.fm_event = copy.deepcopy(self.fms[self.primary])
            self.bench = {}                              # EWMA benchmarks b_t

        def rng_in(self):
            self._amb = torch.get_rng_state()
            torch.set_rng_state(self.rng_state)

        def rng_out(self):
            self.rng_state = torch.get_rng_state()
            torch.set_rng_state(self._amb)

    names = [n for n in WORKER_SPECS[worker]]
    readers = {n: Reader(n, scheds[n]) for n in names}
    keyed = next((r for r in readers.values() if r.kind != "none"), None)
    parse = next((r for r in readers.values() if r.kind == "none"), None)

    # ---- checkpoint grids ----
    rots = {n: W.rotation_steps(r.phase1, r.rot_period, max_steps)
            for n, r in readers.items()}
    cheap = set(range(0, max_steps + 1, cfg["ckpt_every"])) | {max_steps}
    for n, r in readers.items():
        if r.rot_period >= 4 * cfg["post_rot_b"]:
            for x in rots[n]:
                cheap |= {x + d for d in (cfg["post_rot_a"], cfg["post_rot_b"])
                          if x + d <= max_steps}
    for a, b in windows:                       # resolve each burst's own transient
        cheap |= {a + d for d in (0, 25, 50, 75) if a + d <= max_steps}
        cheap |= {b + d for d in (0, 25, 50, 75) if b + d <= max_steps}
    cheap = sorted(cheap)
    index_ck = sorted(set(range(0, max_steps + 1, cfg["index_every"])) | {max_steps})
    probe_ck = sorted({int(x) for x in cfg["probe_ckpts"].split(",")
                       if x.strip() and int(x) <= max_steps} | {max_steps})
    full_ck = {int(x) for x in cfg["full_ckpts"].split(",")
               if x.strip() and int(x) <= max_steps}
    if cfg.get("full_at_end", True):
        full_ck |= {max_steps}

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    print(f"===== worker {worker}  readers {names}  bursts {windows} rho={rhos_c} "
          f"ladder={ladder}  shadow={shadow} fm={use_fm} =====", flush=True)

    panel_log, started = [], time.time()
    pool_leaf = pool_z = None
    ew_a = 1.0 - 0.5 ** (1.0 / max(cfg["bench_halflife"], 1e-9))    # EWMA weight on the new e

    for step in range(max_steps + 1):
        if step in cheap:
            t_ck = time.time()
            do_index = step in index_ck
            do_probe = step in probe_ck
            bi = BU.burst_index(step, windows)

            # ------------- the donor's per-reader record (bit-identical) -------------
            for n, rd in readers.items():
                model = rd.model
                model.eval()
                q = W.q_at(step, rd.phase1, rd.rot_period, rot_step, v)
                qp = W.q_prev_at(step, rd.phase1, rd.rot_period, rot_step, v)
                merged = step >= rd.merge_at
                rec = {"step": step, "tokens": step * B * T, "q": int(q), "q_prev": int(qp),
                       "era": int(W.era_at(step, rd.phase1, rd.rot_period)),
                       "merged": bool(merged),
                       "consumed": "none" if (merged or rd.kind == "none") else "true",
                       "burst": int(bi), "skipped": bool(rd.skip_bursts and bi >= 0)}
                npos = {}
                if rd.kind == "none":
                    # every mode's wall column is the SAME neutral token: one pass, exact
                    base = nll_per_pos(model, ev_leaf_t,
                                       wcol(rd.kind, rd.seed_off, ev_leaf, ev_z, q, qp,
                                            "none", MODES.index("none")))
                    for mo in MODES:
                        npos[mo] = base
                else:
                    for mi, mode in enumerate(MODES):
                        npos[mode] = nll_per_pos(model, ev_leaf_t,
                                                 wcol(rd.kind, rd.seed_off, ev_leaf, ev_z,
                                                      q, qp, mode, mi))
                rec["nll"] = {
                    mo: {"idx": float(npos[mo][key_lo:key_hi].mean()),
                         "out": float(npos[mo][key_hi:].mean()),
                         "all": float(npos[mo].mean()),
                         "pos0": float(npos[mo][0])} for mo in MODES}
                rec["excess"] = {
                    mo: {"idx": float((npos[mo] - bayes_none)[key_lo:key_hi].mean()),
                         "out": float((npos[mo] - bayes_none)[key_hi:].mean()),
                         "all": float((npos[mo] - bayes_none).mean())} for mo in MODES}
                rec["excess_by_level"] = {
                    str(int(k)): float((npos["true"] - bayes_none)[ptl == k].mean())
                    for k in np.unique(ptl)}
                rec["excess_by_level_rand"] = {
                    str(int(k)): float((npos["rand"] - bayes_none)[ptl == k].mean())
                    for k in np.unique(ptl)}
                rec["nll_pos_true"] = npos["true"].tolist()
                rec["binding"] = {
                    "idx": float(npos["rand"][key_lo:key_hi].mean()
                                 - npos["true"][key_lo:key_hi].mean()),
                    "out": float(npos["rand"][key_hi:].mean() - npos["true"][key_hi:].mean()),
                    "pos0": float(npos["rand"][0] - npos["true"][0]),
                    "misleading_idx": float(npos["perm"][key_lo:key_hi].mean()
                                            - npos["true"][key_lo:key_hi].mean()),
                    "stale_idx": float(npos["true"][key_lo:key_hi].mean()
                                       - npos["old"][key_lo:key_hi].mean()),
                    "vs_none_idx": float(npos["none"][key_lo:key_hi].mean()
                                         - npos["true"][key_lo:key_hi].mean()),
                    "vs_none_out": float(npos["none"][key_hi:].mean()
                                         - npos["true"][key_hi:].mean()),
                }
                if rd.donor_panel and do_index:
                    Ei, Eo, D = transfer_and_cardinality(model, rd.kind)
                    off = ~np.eye(v, dtype=bool)
                    ar = np.arange(v)
                    implied = Ei.argmin(0)
                    cur, old, ph1 = (ar + q) % v, (ar + qp) % v, ar
                    diag_cur, diag_old = Ei[cur, ar], Ei[old, ar]
                    offmean = np.array([Ei[[i for i in range(v) if i != cur[a]], a].mean()
                                        for a in range(v)])
                    offmean_o = np.array([Eo[[i for i in range(v) if i != cur[a]], a].mean()
                                          for a in range(v)])
                    rec["index"] = {
                        "n_live": int(live.sum()),
                        "transfer_gap": float((offmean - diag_cur)[live].mean()),
                        "e_cur": float(diag_cur[live].mean()),
                        "e_old": float(diag_old[live].mean()),
                        "e_off": float(offmean[live].mean()),
                        "e_best": float(Ei.min(0)[live].mean()),
                        "map_match_cur": float((implied == cur)[live].mean()),
                        "map_match_old": float((implied == old)[live].mean()),
                        "map_match_phase1": float((implied == ph1)[live].mean()),
                        "match_cur_per_latent": (implied == cur).astype(int).tolist(),
                        "adv_cur_per_latent": (offmean - diag_cur).tolist(),
                        "out_span_transfer_gap": float((offmean_o - Eo[cur, ar])[live].mean()),
                        "jsd_mean": float(D[off].mean()),
                        "card": {f"tau{t}": int(W.cluster_count(D, t))
                                 for t in (0.0002, 0.001, 0.005, 0.02)},
                        "E_idx": Ei.tolist(),
                    }
                    Pm = wall_conditionals(model, rd.kind)
                    D0m = W.jsd_matrix(Pm)
                    rec["index"]["jsd0_mean"] = float(D0m[off].mean())
                    rec["index"]["jsd0_ratio"] = float(
                        D0m[off].mean() / max(refs["exact_jsd_mean"], 1e-12))
                if rd.donor_panel and do_probe:
                    full = step in full_ck
                    blocks = all_blocks if full else pblocks
                    rd.rng_in()
                    lv = {}
                    for mode in ("true", "rand", "none"):
                        wc = wcol(rd.kind, rd.seed_off, pr_leaf, pr_z, q, qp,
                                  mode, MODES.index(mode))
                        lv[mode] = {
                            "key": probe_levels(model, wc, "key", key_plen, blocks, full),
                            "last": probe_levels(model, wc, "last", last_plen, blocks, full)}
                    rd.rng_out()
                    rec["levels"], rec["full_probe"] = lv, bool(full)
                rd.log.append(rec)
                with open(os.path.join(outdir, f"{n}.json"), "w") as fh:
                    json.dump({"arm": n, "sched": rd.sched, "log": rd.log,
                               "complete": step == max_steps}, fh, indent=2,
                              cls=NumpyEncoder)

            t_donor = time.time() - t_ck
            # ------------------------- the SHADOW PANEL -------------------------
            if shadow and keyed is not None:
                t_pan = time.time()
                q = W.q_at(step, keyed.phase1, keyed.rot_period, rot_step, v) if keyed else 0
                qp = (W.q_prev_at(step, keyed.phase1, keyed.rot_period, rot_step, v)
                      if keyed else 0)
                prec = {"step": step, "q": int(q), "burst": int(bi),
                        "rho_c": rho_ref, "rhos_c": rhos_c, "ladder": ladder,
                        "ladder_realised": pn_burst_rate}
                # the wall columns the panel uses (deterministic; no RNG)
                wtrue = (wcol(keyed.kind, keyed.seed_off, pn_leaf, pn_z, q, qp, "true", 0)
                         if keyed else None)
                wperm = (wcol(keyed.kind, keyed.seed_off, pn_leaf, pn_z, q, qp, "perm", 2)
                         if keyed else None)
                wnone = torch.full((n_panel,), W.neutral_tok(v), dtype=torch.long)

                conds = [("clean", pn_leaf_t, wtrue), ("rot", pn_leaf_t, wperm)]
                for i, rho in enumerate(ladder):
                    conds.append((f"burst{i}", pn_burst_t[i], wtrue))
                conds.append(("burst_c", pn_bc_t, wtrue))

                # p_key-with-key-neutralised and p_parse are unchanged by the ROTATION, so
                # the clean-condition passes are shared between `clean` and `rot`.
                cache_self, cache_parse = {}, {}
                for cname, lt, wc in conds:
                    ck = "clean" if cname in ("clean", "rot") else cname
                    ki, ko, Pk = panel_read(keyed.model, lt, wc) if keyed else (0, 0, None)
                    if ck not in cache_self and keyed is not None:
                        cache_self[ck] = panel_read(keyed.model, lt, wnone)
                    if ck not in cache_parse and parse is not None:
                        cache_parse[ck] = panel_read(parse.model, lt, wnone)
                    si, so_, Ps = cache_self.get(ck, (None, None, None))
                    pi, po, Pp = cache_parse.get(ck, (None, None, None))
                    row = {"nll_key_idx": ki, "nll_key_out": ko,
                           "nll_self_idx": si, "nll_parse_idx": pi, "nll_parse_out": po}
                    if Pk is not None and Ps is not None:
                        row["D_self"] = kl(Pk, Ps)
                        row["D_self_rev"] = kl(Ps, Pk)
                        row["JS_self"] = jsd(Pk, Ps)
                    if Pk is not None and Pp is not None:
                        row["D_pair"] = kl(Pk, Pp)
                        row["D_pair_rev"] = kl(Pp, Pk)
                        row["JS_pair"] = jsd(Pk, Pp)
                    if Ps is not None and Pp is not None:
                        row["D_selfparse"] = kl(Ps, Pp)      # the mirror control: how far
                        # apart are the two PARSE bases (one hollow, one real)?
                    prec[cname] = row
                    del Pk

                prec["ms_panel"] = round(1000 * (time.time() - t_pan))
                t_fmr = time.time()
                # ---------------------------- Factor M ----------------------------
                if use_fm:
                    fmrec = {}
                    for n, rd in readers.items():
                        wcl = wtrue if rd.kind != "none" else wnone
                        wpm = wperm if rd.kind != "none" else wnone
                        # ONE reader forward per condition; every FM variant (the lr sweep
                        # and both frozen snapshots) rides the same activations.
                        fset = dict(rd.fms)
                        fset["frozen_prev"] = rd.fm_prev
                        fset["frozen_event"] = rd.fm_event
                        cset = {"clean": (pn_leaf_t, wcl), "burst_c": (pn_bc_t, wcl)}
                        if rd.kind != "none":
                            cset["rot"] = (pn_leaf_t, wpm)
                        e_out, bd = {}, {}
                        for cname, (lt, wc) in cset.items():
                            e_out[cname] = fm_residual(rd.model, fset, lt, wc)
                            for k, val in e_out[cname].items():
                                key = f"{cname}|{k}"
                                e = val["mse"]
                                prev = rd.bench.get(key)
                                b = e if prev is None else prev
                                bd[key] = {"b": b, "delta": b - e}
                                rd.bench[key] = (e if prev is None
                                                 else (1 - ew_a) * prev + ew_a * e)
                        fmrec[n] = {"e": e_out, "bench": bd}
                        # roll the frozen snapshots (AFTER reading them)
                        rd.fm_prev.load_state_dict(rd.fms[rd.primary].state_dict())
                        if any(step <= o < step + cfg["ckpt_every"] + 1
                               for o in onsets + rots.get(n, [])):
                            rd.fm_event.load_state_dict(rd.fms[rd.primary].state_dict())
                    prec["fm"] = fmrec
                prec["ms_fm"] = round(1000 * (time.time() - t_fmr))
                prec["ms_donor"] = round(1000 * t_donor)
                prec["ms"] = round(1000 * (time.time() - t_ck))
                panel_log.append(prec)
                with open(os.path.join(outdir, f"panel_{worker}.json"), "w") as fh:
                    json.dump({"worker": worker, "readers": names, "windows": windows,
                               "rho_c": rho_ref, "rhos_c": rhos_c, "ladder": ladder,
                               "ladder_realised": pn_burst_rate,
                               "log": panel_log, "complete": step == max_steps},
                              fh, indent=2, cls=NumpyEncoder)

            # ------------------------------ the line ------------------------------
            for n, rd in readers.items():
                r = rd.log[-1]
                cons = r["consumed"]
                lvtxt = ""
                if "levels" in r:
                    lt = r["levels"]["true"]["key"]; lr_ = r["levels"]["rand"]["key"]
                    lvtxt = f"  d4 t/r {lt['d4']:.3f}/{lr_['d4']:.3f}"
                pt = ""
                if shadow and panel_log:
                    pl = panel_log[-1]
                    if "D_pair" in pl["clean"]:
                        pt = (f" | Dp {pl['clean']['D_pair']:.4f}->{pl['rot']['D_pair']:.4f}"
                              f"/{pl['burst_c']['D_pair']:.4f}")
                    if "D_self" in pl["clean"]:
                        pt += (f" Ds {pl['clean']['D_self']:.4f}->"
                               f"{pl['rot']['D_self']:.4f}/{pl['burst_c']['D_self']:.4f}")
                print(f"[{n:10s} s{step:6d} q{r['q']:2d}{'B' if bi >= 0 else ' '}"
                      f"{'S' if r['skipped'] else ' '}] "
                      f"nll idx {r['nll'][cons]['idx']:.4f} out {r['nll'][cons]['out']:.4f} "
                      f"| bind {r['binding']['idx']:+.4f} "
                      f"mislead {r['binding']['misleading_idx']:+.4f}" + pt + lvtxt,
                      flush=True)
            volume.commit()
            for rd in readers.values():
                rd.model.train()

        if step == max_steps:
            break

        if step % cfg["fresh_every"] == 0:
            need = max(64, B * cfg["fresh_every"])
            fl, flf, _ = _generate_with_traces(
                rules, need, cfg["data_seed"] + 100_003 * (step // cfg["fresh_every"] + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))

        bi = BU.burst_index(step, windows)
        for n, rd in readers.items():
            # `ix` is drawn FIRST and unconditionally (the donor's rule), and it is drawn
            # even on a SKIPPED step, so a skipping arm's sampler stream stays aligned.
            ix = torch.randint(0, pool_leaf.shape[0], (B,), generator=rd.gen)
            leaf_b, z_b = pool_leaf[ix], pool_z[ix]
            if rd.skip_bursts and bi >= 0:
                continue                                   # THE SKIP OP
            if bi >= 0 and rhos_c[bi] > 0:
                cor, _ = BU.corrupt(leaf_b.numpy(), key_lo, key_hi, rhos_c[bi], v,
                                    BU.train_burst_rng(cfg["burst_seed"], step))
                leaf_b = torch.from_numpy(cor)
            q = W.q_at(step, rd.phase1, rd.rot_period, rot_step, v)
            if step >= rd.merge_at or rd.kind == "none":
                wc = torch.full((B,), W.neutral_tok(v), dtype=torch.long)
            elif rd.kind == "dead":
                wc = torch.from_numpy(
                    (W.wall_tok(rd.train_rng.integers(0, v, size=B), v)).astype(np.int64))
            else:
                wc = W.wall_tok((z_b + q) % v, v)
            x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
            if use_fm:
                _, loss, inter = rd.model(x, y, return_intermediates=True)
                h0 = inter[cfg["fm_shallow"]].detach()
                h6 = inter[cfg["fm_deep"]].detach()[:, key_lo:key_hi, :]
                rd.opt.zero_grad(); loss.backward(); rd.opt.step()
                for k, fm in rd.fms.items():
                    fl_ = F.mse_loss(fm(h0)[:, key_lo:key_hi, :], h6)
                    rd.fopts[k].zero_grad(); fl_.backward(); rd.fopts[k].step()
            else:
                _, loss = rd.model(x, y)
                rd.opt.zero_grad(); loss.backward(); rd.opt.step()
            if step % cfg["log_interval"] == 0:
                print(f"  {n} {step:6d} ntp {loss.item():.4f}", flush=True)

    print(f"[{worker}] DONE in {time.time() - started:.0f}s", flush=True)
    return {"worker": worker, "readers": names, "elapsed": time.time() - started,
            "n_ckpt": len(next(iter(readers.values())).log)}


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=16384)
def tune_lm(
    tag: str = "smoke",
    workers: str = "pair,skip",
    # DGP / model / training — the donor's, verbatim
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    key_level: int = 2, key_node: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    max_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, data_seed: int = 7, seed: int = 42,
    fresh_every: int = 500,
    phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
    # the burst schedule
    burst_steps: str = "5000,11000,17000", burst_len: int = 250,
    burst_rho: str = "0.0242,0.0410,0.0377", burst_seed: int = 31337,
    burst_ladder: str = "0.002,0.005,0.01,0.02,0.05,0.10",
    # the FM
    fm_shallow: str = "post_block0", fm_deep: str = "post_block6",
    fm_n_layer: int = 1, fm_d_head: int = 16, fm_n_head: int = 8,
    fm_mlp_mult: float = 1.0, bench_halflife: float = 8.0,
    # measurement
    n_eval: int = 4096, eval_seed: int = 999, n_panel: int = 512,
    n_probe: int = 4000, probe_seed: int = 1001, n_card: int = 256,
    min_cell_n: int = 24,
    ckpt_every: int = 125, index_every: int = 500,
    post_rot_a: int = 50, post_rot_b: int = 100,
    probe_ckpts: str = ("250,1000,2000,4000,5000,5250,6000,8000,8250,10000,"
                        "11000,11250,12000,14000,16000,17000,17250,18000"),
    full_ckpts: str = "8000",
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    full_at_end: bool = True,
    probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7",
    log_interval: int = 2000,
    shadow: bool = True, fm: bool = True, quick: bool = False,
):
    if quick:
        max_steps, n_eval, n_probe, n_panel = 900, 768, 384, 256
        phase1, rot_period, fresh_every = 300, 300, 100
        ckpt_every, index_every, probe_steps, mlp_steps = 100, 300, 100, 100
        probe_ckpts, full_ckpts, log_interval = "300,600", "900", 300
        n_card, min_cell_n = 128, 8
        burst_steps, burst_len = "400,700", 100
        burst_ladder = "0.01,0.05,0.20"

    wlist = [w.strip() for w in workers.split(",") if w.strip()]
    for w in wlist:
        if w not in WORKER_SPECS:
            raise ValueError(f"unknown worker {w}; known: {sorted(WORKER_SPECS)}")

    cfg = dict(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, key_level=key_level,
               key_node=key_node, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               max_steps=max_steps, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, data_seed=data_seed, seed=seed,
               fresh_every=fresh_every, rot_step=rot_step, phase1=phase1,
               rot_period=rot_period,
               burst_steps=burst_steps, burst_len=burst_len, burst_rho=burst_rho,
               burst_seed=burst_seed, burst_ladder=burst_ladder,
               fm_shallow=fm_shallow, fm_deep=fm_deep, fm_n_layer=fm_n_layer,
               fm_d_head=fm_d_head, fm_n_head=fm_n_head, fm_mlp_mult=fm_mlp_mult,
               bench_halflife=bench_halflife,
               n_eval=n_eval, eval_seed=eval_seed, n_panel=n_panel, n_probe=n_probe,
               probe_seed=probe_seed, n_card=n_card, min_cell_n=min_cell_n,
               ckpt_every=ckpt_every, index_every=index_every, post_rot_a=post_rot_a,
               post_rot_b=post_rot_b, probe_ckpts=probe_ckpts, full_ckpts=full_ckpts,
               probe_steps=probe_steps, probe_lr=probe_lr, mlp_hidden=mlp_hidden,
               mlp_steps=mlp_steps, probe_blocks=probe_blocks, log_interval=log_interval,
               full_at_end=bool(full_at_end),
               shadow=bool(shadow), fm=bool(fm))

    print(f"{'=' * 78}\ntuning/Gate 0  tag={tag}  {setting_key(v, s, depth, m)}  "
          f"{n_layer}L/{n_head}H/{n_embd}D  T={s ** depth}")
    print(f"  workers={wlist}  max_steps={max_steps}  shadow={shadow} fm={fm}")
    print(f"  bursts at {burst_steps} x {burst_len} steps, rho(s)={burst_rho}; "
          f"ladder {burst_ladder}\n{'=' * 78}", flush=True)

    refs, refs_src = load_refs(cfg)
    scheds = {r: reader_schedule(r, phase1, rot_period, max_steps) for r in READER_SPECS}
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "workers": wlist, "scheds": scheds, "refs": refs,
                   "refs_source": refs_src, "fm_lrs": list(FM_LRS),
                   "fm_primary": FM_PRIMARY}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    started = time.time()
    handles = [(w, run_worker.spawn(tag, w, scheds, refs, cfg)) for w in wlist]
    out = {}
    for w, h in handles:
        try:
            out[w] = h.get()
        except Exception as exc:                              # noqa: BLE001
            out[w] = {"worker": w, "error": repr(exc)}
            print(f"[driver] worker {w} FAILED: {exc!r}", flush=True)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n{json.dumps(out, indent=2)}\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}\n{json.dumps(out, indent=2)}")
    return out


# --------------------------------------------------------------------------- #
# the burst-rate calibration (GPU, short): match the MODEL's burst spike to the
# MODEL's rotation spike, at the maturity the first burst will meet
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def calib(tag: str = "calib0", steps: str = "2000,5000,8000",
          rhos: str = "0.002,0.005,0.01,0.02,0.05,0.10,0.20",
          v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
          key_level: int = 2, key_node: int = 0,
          n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
          batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
          data_seed: int = 7, seed: int = 42, fresh_every: int = 500,
          phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
          n_eval: int = 2048, eval_seed: int = 999):
    """Train the keyed reader on the DONOR's schedule (no bursts, no panel, no FM — so
    this is the donor's own trajectory) and, at each ladder checkpoint, read

        rotation instant   NLL(perm) - NLL(true)     on the indexed span
        burst instant      NLL(true, corrupted) - NLL(true, clean)   for every rho

    on the SAME weights and the same eval draw. The calibrated rate is the rho whose burst
    spike equals the rotation spike at step 5000 — the maturity the first consumed burst
    will actually meet.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T, V = depth, s ** depth, W.vocab_size(v)
    ck = sorted({int(x) for x in steps.split(",")})
    rr = [float(x) for x in rhos.split(",")]
    max_steps = max(ck)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    key_lo, key_hi = W.key_span(L, s, key_level, key_node)

    ev_leaf, ev_lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    ev_z = ev_lf[key_level][:, key_node]
    ev_t = torch.from_numpy(ev_leaf.astype(np.int64))
    burst_t, realised = [], []
    for i, rho in enumerate(rr):
        cor, mask = BU.corrupt(ev_leaf, key_lo, key_hi, rho, v,
                               BU.shadow_burst_rng(eval_seed, i))
        burst_t.append(torch.from_numpy(cor.astype(np.int64)))
        realised.append(float(mask.mean()))

    torch.manual_seed(seed)
    model = GPT(V, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    @torch.no_grad()
    def nll_idx_out(leaf_t, wc, chunk=256):
        si = so = 0.0
        n = leaf_t.shape[0]
        for i in range(0, n, chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            si += float(nll[:, key_lo:key_hi].sum()); so += float(nll[:, key_hi:].sum())
        return si / (n * (key_hi - key_lo)), so / (n * (T - key_hi))

    out = {"steps": ck, "rhos": rr, "realised": realised, "rows": []}
    pool_leaf = pool_z = None
    for step in range(max_steps + 1):
        if step in ck:
            model.eval()
            q = W.q_at(step, phase1, rot_period, rot_step, v)
            wt = torch.from_numpy(W.wall_tok((ev_z + q) % v, v).astype(np.int64))
            wp = torch.from_numpy(W.wall_tok((ev_z + q + rot_step) % v, v).astype(np.int64))
            wn = torch.full((n_eval,), W.neutral_tok(v), dtype=torch.long)
            ci, co = nll_idx_out(ev_t, wt)
            ri, ro = nll_idx_out(ev_t, wp)
            ni, no_ = nll_idx_out(ev_t, wn)
            row = {"step": step, "clean_idx": ci, "clean_out": co,
                   "rot_idx": ri, "rot_out": ro, "rot_spike": ri - ci,
                   "rot_out_leak": ro - co,
                   "none_idx": ni, "vs_none": ni - ci, "burst": []}
            for i, rho in enumerate(rr):
                bi_, bo_ = nll_idx_out(burst_t[i], wt)
                row["burst"].append({"rho": rho, "realised": realised[i],
                                     "idx": bi_, "out": bo_,
                                     "spike": bi_ - ci, "out_leak": bo_ - co})
            rho_star, brack = BU.match_rate(
                [{"rho": b["rho"], "d_naive": b["spike"]} for b in row["burst"]],
                row["rot_spike"])
            row["rho_star"], row["bracketed"] = rho_star, brack
            out["rows"].append(row)
            print(f"[calib s{step:6d}] clean {ci:.4f}  rot +{row['rot_spike']:.4f} "
                  f"(out +{row['rot_out_leak']:.4f})  vs_none +{row['vs_none']:.4f}",
                  flush=True)
            for b in row["burst"]:
                print(f"    rho {b['rho']:<7.4g} realised {b['realised']:.4f}  "
                      f"idx {b['idx']:.4f} (+{b['spike']:.4f})  "
                      f"out +{b['out_leak']:.4f}", flush=True)
            print(f"    -> rho* matching the rotation spike: {rho_star:.5f} "
                  f"({'bracketed' if brack else 'EXTRAPOLATED'})", flush=True)
            model.train()
        if step == max_steps:
            break
        if step % fresh_every == 0:
            need = max(64, batch_size * fresh_every)
            fl, flf, _ = _generate_with_traces(
                rules, need, data_seed + 100_003 * (step // fresh_every + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))
        ix = torch.randint(0, pool_leaf.shape[0], (batch_size,), generator=gen)
        leaf_b, z_b = pool_leaf[ix], pool_z[ix]
        q = W.q_at(step, phase1, rot_period, rot_step, v)
        wc = W.wall_tok((z_b + q) % v, v)
        x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "calib.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print("\n=== CALIBRATION SUMMARY ===")
    for r in out["rows"]:
        print(f"  s{r['step']:6d}  rotation spike +{r['rot_spike']:.4f}  ->  rho* "
              f"{r['rho_star']:.5f}  ({'bracketed' if r['bracketed'] else 'extrapolated'})")
    return out


# --------------------------------------------------------------------------- #
# gates (CPU: structural + the model-free irreducibility calibration)
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=7200, memory=16384)
def gate(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
         key_level: int = 2, key_node: int = 0, n_eval: int = 2048, eval_seed: int = 999,
         phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
         max_steps: int = 20000, burst_steps: str = "5000,11000,17000",
         burst_len: int = 250, burst_seed: int = 31337,
         rhos: str = "0.005,0.01,0.02,0.05,0.10,0.20", n_oracle: int = 192,
         rot_spike_target: float = 0.63):
    """B-1  the burst is confined to the indexed span, at the nominal rate, and carries
            ZERO information about the keyed latent (the irreducibility precondition).
       B-2  the burst never occupies a rotation instant, and every window is disjoint.
       B-3  RNG neutrality: the burst draws are independent of the batch sampler, and the
            SHADOW draw is fixed across checkpoints while the CONSUMED draw varies by step.
       B-4  the exact, model-free irreducibility curve (belief propagation, no model): the
            noise-aware optimum's excess is a floor no learner can lower, and the naive
            excess is what a reader that does not know the noise exists pays.
       B-5  the leaf/wall/neutral token layout still holds under corruption.
       G-*  the donor's own structural gates, re-asserted on this config.
    """
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    leaf, lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    z = lf[key_level][:, key_node]
    key_lo, key_hi = W.key_span(L, s, key_level, key_node)
    rr = [float(x) for x in rhos.split(",")]
    starts = [int(x) for x in burst_steps.split(",")]
    windows = BU.burst_windows(starts, burst_len, max_steps)
    rots = W.rotation_steps(phase1, rot_period, max_steps)
    out = {"setting": setting_key(v, s, L, m), "T": T, "key_span": [key_lo, key_hi],
           "windows": windows, "rotations": rots}

    # --- B-1: locality, rate, and irreducibility's precondition ---
    rng = BU.train_burst_rng(burst_seed, 5000)
    cor, mask = BU.corrupt(leaf, key_lo, key_hi, 0.10, v, rng)
    outside_touched = int((cor[:, key_hi:] != leaf[:, key_hi:]).sum())
    changed = cor[:, key_lo:key_hi] != leaf[:, key_lo:key_hi]
    # the information check runs at rho = 1 and POOLED over the span, so the plug-in MI
    # estimator's bias ((v-1)^2 / 2N) is ~0.003 nats rather than swamping the reading; a
    # label permutation supplies the matched null.
    full, _ = BU.corrupt(leaf, key_lo, key_hi, 1.0, v,
                         BU.train_burst_rng(burst_seed, 5001))
    zt = np.repeat(z[:, None], key_hi - key_lo, axis=1).ravel()
    ft = full[:, key_lo:key_hi].ravel()
    mi_corrupt = float(_mutinf(ft, zt, v))
    prng = np.random.default_rng(0)
    nulls = [float(_mutinf(ft, prng.permutation(zt), v)) for _ in range(20)]
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    # scale reference: the marginal MI a SINGLE clean leaf carries is itself tiny on this
    # world (the key's 0.152 nats/token lives in the joint, not the per-token marginal), so
    # the meaningful comparison is against the matched permutation null, not against it.
    mi_clean = float(_mutinf(leaf[:, key_lo:key_hi].ravel(), zt, v))
    out["B1"] = {"outside_span_touched": outside_touched,
                 "nominal_rate": 0.10, "realised_rate": float(mask.mean()),
                 "changed_given_masked": float(changed[mask].mean()),
                 "n_pairs": int(zt.size),
                 "MI_fully_corrupted_span_z_nats": mi_corrupt,
                 "MI_permutation_null_mean": mu, "MI_permutation_null_sd": sd,
                 "MI_permutation_null_max": float(np.max(nulls)),
                 "z_score_vs_null": (mi_corrupt - mu) / max(sd, 1e-12),
                 "MI_clean_per_token_marginal_nats": mi_clean}
    assert outside_touched == 0, "B-1 corruption escaped the indexed span"
    assert abs(mask.mean() - 0.10) < 0.01, "B-1 rate off"
    assert mi_corrupt <= mu + 4 * sd, \
        "B-1 corrupted tokens still carry the latent (above the permutation null)"

    # --- B-2 ---
    out["B2"] = {"overlaps_rotation": bool(BU.overlaps(windows, rots)),
                 "overlaps_rotation_guard100": bool(BU.overlaps(windows, rots, guard=100)),
                 "disjoint": all(windows[i][1] <= windows[i + 1][0]
                                 for i in range(len(windows) - 1))}
    assert not out["B2"]["overlaps_rotation_guard100"], "B-2 a burst sits on a rotation"
    assert out["B2"]["disjoint"], "B-2 windows overlap"

    # --- B-3 ---
    a1, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 100))
    a2, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 100))
    a3, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 101))
    b1, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.shadow_burst_rng(eval_seed, 0))
    b2, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.shadow_burst_rng(eval_seed, 0))
    out["B3"] = {"consumed_same_step_reproducible": bool((a1 == a2).all()),
                 "consumed_differs_across_steps": bool((a1 != a3).any()),
                 "shadow_fixed": bool((b1 == b2).all())}
    assert out["B3"]["consumed_same_step_reproducible"] and out["B3"]["shadow_fixed"], "B-3"
    assert out["B3"]["consumed_differs_across_steps"], "B-3 consumed draw is not per-step"

    # --- B-4: the exact, model-free irreducibility curve ---
    print("\n--- B-4  exact irreducibility curve (BP, no model) ---", flush=True)
    sub, subz = leaf[:n_oracle], z[:n_oracle]
    curve = BU.oracle_burst_curve(rules, sub, subz, key_level, key_node, key_lo, key_hi,
                                  rr, v, eval_seed=eval_seed, keyed=True)
    out["B4"] = curve
    per_tok = [r["d_robust"] / max(r["realised_rate"], 1e-9) for r in curve["rows"]]
    out["B4"]["irreducible_nats_per_corrupted_token"] = per_tok
    assert all(r["d_robust"] > 0 for r in curve["rows"]), "B-4 no irreducible excess"
    assert all(curve["rows"][i]["d_robust"] <= curve["rows"][i + 1]["d_robust"] + 1e-9
               for i in range(len(curve["rows"]) - 1)), "B-4 floor not monotone in rho"

    # --- B-5 ---
    out["B5"] = {"vocab": W.vocab_size(v), "leaf_max_corrupted": int(cor.max()),
                 "wall_range": [v, 2 * v - 1], "neutral": W.neutral_tok(v),
                 "disjoint": bool(int(cor.max()) < v)}
    assert out["B5"]["disjoint"], "B-5"

    # --- the donor's structural gates, re-asserted ---
    zz = np.arange(v)
    out["G1"] = {"bijection": bool(all(len(set(((zz + q) % v).tolist())) == v
                                       for q in range(v))),
                 "derangements": bool(all(W.is_derangement(k * rot_step, v)
                                          for k in range(1, v // max(1, np.gcd(rot_step, v)))))}
    assert out["G1"]["bijection"] and out["G1"]["derangements"], "G-1"
    out["G2"] = {"rotations": rots,
                 "q_at_0": W.q_at(0, phase1, rot_period, rot_step, v),
                 "q_at_phase1_minus_1": W.q_at(phase1 - 1, phase1, rot_period, rot_step, v)}
    assert out["G2"]["q_at_0"] == 0 and out["G2"]["q_at_phase1_minus_1"] == 0, "G-2"

    print("\n" + json.dumps({k: out[k] for k in out if k != "B4"}, indent=2,
                            cls=NumpyEncoder))
    print("\nB-4 rows:")
    for r in out["B4"]["rows"]:
        print(f"  rho {r['rho']:<7.4g} realised {r['realised_rate']:.4f}  "
              f"irreducible +{r['d_robust']:.4f}  naive +{r['d_naive']:.4f}")
    print(f"\n  rho whose IRREDUCIBLE floor equals a rotation's {rot_spike_target} nats: "
          f"{BU.match_rate(out['B4']['rows'], rot_spike_target, key='d_robust')}")
    print("\nGATE: PASS")
    return out


def _mutinf(a, b, v):
    import numpy as np
    j = np.zeros((v, v))
    np.add.at(j, (a, b), 1.0)
    j /= max(j.sum(), 1e-12)
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return (j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum()


@app.local_entrypoint()
def main(quick: bool = True):
    tune_lm.remote(quick=quick, tag="smoke_local")
