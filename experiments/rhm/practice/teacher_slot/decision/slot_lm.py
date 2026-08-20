"""teacher_slot/decision (Rung A) — the decision round: will a second loop CHOOSE the merge?

Donor (imported, never modified): `../../fourwall/lm/` — `wall.py`'s primitives, schedules
and exact BP oracles; `wall_lm.py`'s `build_refs`, arm-schedule shape and per-arm-worker
design. Spec: `../SPEC.md`, "Rung A — the decision round".

WHY. `fourwall/lm` measured that the gradient reader supplies *track* and not *merge*: it
takes a free spurious key instantly, re-maps after every rotation, never quotients, and
ends at Bayes-level task NLL with its token-derived inference pathway threefold suppressed
(key-anchor d4 0.27-0.30 keyed vs 0.83-0.84 controls, exact ceiling 0.921). Supplied
exogenously the merge op repairs exactly that, at a +0.10-0.17 nat transient that decays
in ~25-125 steps. Two facts set up this round:

  (a) AT THE INSTANT OF THE OP, task NLL is spiked while the pathway readout is already
      climbing -- a quantity exists that improves inside the dip;
  (b) the within-level lifetime task integral REWARDS never merging
      (true_wall 1.4202 < wall 1.4619 < merge_13000 1.4729 < merge_8000 1.4991
       < merge_1000 1.5304 ~ no_wall 1.5311).

WHAT'S NEW. An outer loop that at each checkpoint chooses the input condition for the next
interval -- `w` present (at the CURRENT era's map) or collapsed to the neutral filler --
REVERSIBLY. `fourwall/lm`'s `merge_s` arms are the step-function special case, which is
also the fidelity gate. The loops differ in ONE respect: what they read.

ARMS
    wall             policy hard-coded "never collapse"      [fidelity gate vs fwlm0]
    no_wall          no wall token at all                    [fidelity gate vs fwlm0/1]
    merge_8000       hard-coded "collapse from 8000"         [fidelity gate vs fwlm1]
    outer_task       reads indexed-span NLL under the CONSUMED condition (free)
    outer_path       reads the probe-free pathway readout (indexed-span excess over exact
                     Bayes under the NEUTRAL wall) -- priced
    outer_path_lp    same read, two-timescale learning-progress reward (`two_timescale`)
    outer_path_true  the unrotated control: the `true_wall` schedule under `outer_path`.
                     Does it merge with no rotation ever occurring, on pathway evidence
                     alone?

FIDELITY. The three gate arms must reproduce their `fourwall/lm` twins BIT-FOR-BIT at
every shared checkpoint (the donor's 9a discipline, max|delta| = 0.0). This is structural,
not hoped-for: a `FixedPolicy` probes nothing, consumes no RNG, and reduces the condition
rule to exactly the donor's `step >= merge_at`; `ix` is still drawn first and
unconditionally; training consumes zero global torch RNG; and `refs` is LOADED from the
donor's own `setup.json` on the volume rather than recomputed, so `excess` shares the
donor's exact Bayes array to the last bit.

Run from experiments/:
  modal run -m rhm.practice.teacher_slot.decision.slot_lm::bench
  modal run -m rhm.practice.teacher_slot.decision.slot_lm::slot --quick --tag tsd_smoke
  python3 rhm/practice/teacher_slot/decision/launch_detached.py --fn slot --tag tsdA ...
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
)
app = modal.App("rhm-practice-teacher-slot-decision", image=image)

REMOTE = "rhm_practice_teacher_slot_decision"
DONOR_REMOTE = "rhm_practice_fourwall_lm"          # read-only: refs + the gate twins

# arm -> (wall kind, phase1 override, rot_period override, seed offset, policy spec).
# `None` = take the run-level default.
ARM_SPECS_R1 = {
    # --- the exogenous bracket / unassisted arms, re-run in-tag: the FIDELITY GATES ---
    "wall":            ("wall", None, None, 0, {"kind": "fixed", "merge_at": None}),
    "no_wall":         ("none", None, 0,    0, {"kind": "fixed", "merge_at": None}),
    "merge_8000":      ("wall", None, None, 0, {"kind": "fixed", "merge_at": 8000}),
    "merge_1000":      ("wall", None, None, 0, {"kind": "fixed", "merge_at": 1000}),
    # --- the decision arms: identical in every respect except WHAT THE LOOP READS ---
    "outer_task":      ("wall", None, None, 0, {"kind": "bandit", "read": "task",
                                                "mode": "delta"}),
    "outer_path":      ("wall", None, None, 0, {"kind": "bandit", "read": "path",
                                                "mode": "delta"}),
    "outer_path_lp":   ("wall", None, None, 0, {"kind": "bandit", "read": "path",
                                                "mode": "lp"}),
    # --- the unrotated control: `true_wall`'s schedule under the `outer_path` policy ---
    "outer_path_true": ("wall", None, 0,    0, {"kind": "bandit", "read": "path",
                                                "mode": "delta"}),
}


# ROUND 2 (`tsdB`). The decision arms move to the ABBA PAIRED trial (see
# `policy.PairedPolicy` for why round 1's bandit measured its own baseline decay), and
# Rung A1/2's endogenous cross-level drive joins as `outer_yield*`. Round 1's specs are
# kept verbatim as `ARM_SPECS_R1` so `tsdA` stays reproducible (`--policy-round 1`).
#
# `span` is the two-timescale axis and `trial_every` is set so that EVERY decision arm
# spends the same 40 intervals inside trials and the same 20 intervals under the
# non-greedy condition (12.5% of the budget) -- the measurement horizon is varied, the
# perturbation is held fixed.
ARM_SPECS = {
    # --- the exogenous bracket / unassisted arms: the FIDELITY GATES (unchanged) ---
    "wall":            ("wall", None, None, 0, {"kind": "fixed", "merge_at": None}),
    "no_wall":         ("none", None, 0,    0, {"kind": "fixed", "merge_at": None}),
    "merge_8000":      ("wall", None, None, 0, {"kind": "fixed", "merge_at": 8000}),
    "merge_1000":      ("wall", None, None, 0, {"kind": "fixed", "merge_at": 1000}),
    # --- Rung A: identical except WHAT THE LOOP READS ---
    "outer_task":      ("wall", None, None, 0, {"kind": "paired", "read": "task",
                                                "span": 1, "trial_every": 16}),
    "outer_path":      ("wall", None, None, 0, {"kind": "paired", "read": "path",
                                                "span": 1, "trial_every": 16}),
    "outer_path_lp":   ("wall", None, None, 0, {"kind": "paired", "read": "path",
                                                "span": 2, "trial_every": 32}),
    "outer_path_true": ("wall", None, 0,    0, {"kind": "paired", "read": "path",
                                                "span": 1, "trial_every": 16}),
    # --- Rung A-1/2: the endogenous cross-level drive. The read is the learner's own
    #     NEXT-LEVEL YIELD -- the d5 probe at the key anchor under the neutral wall, one
    #     level ABOVE the keyed latent (d4). Not a probe we wrote to the answer: the
    #     arc's most seed-stable fact is that the value of level-k practice is expressed
    #     in level k+1's currency. d6 is at chance in this world and is never read. ---
    "outer_yield":     ("wall", None, None, 0, {"kind": "paired", "read": "yield",
                                                "span": 1, "trial_every": 32}),
    "outer_yield_lp":  ("wall", None, None, 0, {"kind": "paired", "read": "yield",
                                                "span": 2, "trial_every": 32}),
}


def specs_for(round_):
    return ARM_SPECS_R1 if int(round_) == 1 else ARM_SPECS


def arm_schedule(arm, phase1, rot_period, max_steps, merge_override=0, round_=2):
    """The donor's `wall_lm.arm_schedule`, extended with the policy spec. `merge_at` is
    kept in the dict (and equals max_steps + 1 when the fixed policy never collapses) so
    the donor's analyzer conventions still read this node's records."""
    kind, p1, rp, soff, pol = specs_for(round_)[arm]
    p1 = phase1 if p1 is None else p1
    rp = rot_period if rp is None else rp
    if rp <= 0:
        p1 = max_steps + 1                       # never rotates
    pol = dict(pol)
    mg = pol.get("merge_at")
    if pol["kind"] == "fixed" and mg is not None and merge_override > 0:   # smoke only
        mg = merge_override
        pol["merge_at"] = mg
    return {"kind": kind, "phase1": int(p1), "rot_period": int(rp), "seed_off": int(soff),
            "merge_at": int(max_steps + 1 if (pol["kind"] != "fixed" or mg is None) else mg),
            "policy": pol}


# --------------------------------------------------------------------------- #
# references: LOADED from the donor's tag, not recomputed
# --------------------------------------------------------------------------- #

REF_KEYS = ("v", "s", "depth", "m", "rule_seed", "key_level", "key_node",
            "n_eval", "eval_seed", "n_probe", "probe_seed")


def load_refs(cfg, donor_tag="fwlm1", verbose=True):
    """Reuse the donor's exact BP references verbatim when its config matches ours.

    Two reasons, both load-bearing: (1) `excess` is measured against `bayes_none`, so
    sharing the donor's array is what makes the `excess` instruments bit-comparable, and
    (2) the exact predictive sweep is minutes of CPU we do not need to repeat.
    """
    path = f"{DATA_DIR}/{DONOR_REMOTE}/{donor_tag}/setup.json"
    if os.path.exists(path):
        setup = json.load(open(path))
        d = setup["config"]
        if all(d.get(k) == cfg[k] for k in REF_KEYS):
            if verbose:
                print(f"[refs] loaded verbatim from {DONOR_REMOTE}/{donor_tag}/setup.json "
                      f"(config matches on {list(REF_KEYS)})", flush=True)
            return setup["refs"], f"{donor_tag}:setup.json"
        if verbose:
            print(f"[refs] donor config mismatch: "
                  f"{[k for k in REF_KEYS if d.get(k) != cfg[k]]} -> recomputing",
                  flush=True)
    from rhm.practice.fourwall.lm.wall_lm import build_refs
    refs = build_refs(cfg["v"], cfg["s"], cfg["depth"], cfg["m"], cfg["rule_seed"],
                      cfg["key_level"], cfg["key_node"], cfg["n_eval"], cfg["eval_seed"],
                      cfg["n_probe"], cfg["probe_seed"], verbose=verbose)
    return refs, "recomputed"


# --------------------------------------------------------------------------- #
# one arm = one worker = one GPU  (the donor's `run_arm`, plus the outer loop)
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=32400, memory=32768)
def run_arm(tag: str, arm: str, sched: dict, refs: dict, cfg: dict):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.teacher_slot.decision import policy as P

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    T = s ** L
    V = W.vocab_size(v)
    kind = sched["kind"]
    phase1, rot_period = sched["phase1"], sched["rot_period"]
    rot_step, key_level, key_node = cfg["rot_step"], cfg["key_level"], cfg["key_node"]
    seed = cfg["seed"] + sched["seed_off"]
    B = cfg["batch_size"]
    key_lo, key_hi = refs["key_lo"], refs["key_hi"]
    key_plen, last_plen = refs["key_plen"], refs["last_plen"]

    rules = generate_rules_distinct(v, s, L, m, seed=cfg["rule_seed"])
    bayes_none = np.array(refs["bayes_none"])
    ptl = np.array(refs["pos_top_level"])

    # ---- evaluation apparatus, identical in every arm (donor-verbatim) ----
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
    max_steps = cfg["max_steps"]

    all_blocks = ["post_embed"] + [f"post_block{i}" for i in range(cfg["n_layer"])]
    pblocks = [b for b in cfg["probe_blocks"].split(",") if b in all_blocks] or all_blocks

    def wcol(leaf_np, z_np, q, q_prev, mode, mode_i, n=None):
        """The wall column for an eval condition — donor-verbatim, so `rand`'s draw stays
        bit-identical to fwlm0/fwlm1's. `n` restricts to the policy's probe subset."""
        if mode == "none":
            return torch.full((leaf_np.shape[0] if n is None else n,),
                              W.neutral_tok(v), dtype=torch.long)
        rng = np.random.default_rng(cfg["eval_seed"] + 7717 * (mode_i + 1)
                                    + 131 * sched["seed_off"])
        w = W.wall_values(kind, z_np, q, v, rng=rng, mode=mode,
                          q_prev=q_prev, delta=rot_step)
        if w is None:
            return torch.full((leaf_np.shape[0] if n is None else n,),
                              W.neutral_tok(v), dtype=torch.long)
        out = torch.from_numpy((W.wall_tok(w, v)).astype(np.int64))
        return out if n is None else out[:n]

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    @torch.no_grad()
    def nll_per_pos(leaf_t, wc, chunk=256):
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
    def wall_conditionals():
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
    def transfer_and_cardinality(chunk=256):
        n = ev_leaf_t.shape[0]
        nc = card_sel.shape[0]
        Ei = np.zeros((v, v)); Eo = np.zeros((v, v))
        Pc = torch.zeros(v, nc, key_hi - key_lo, v, device=device)
        for i in range(v):
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
        D = np.zeros((v, v))
        lP = torch.log(Pc.clamp_min(1e-12))
        for i in range(v):
            M = 0.5 * (Pc[i][None] + Pc)
            lM = torch.log(M.clamp_min(1e-12))
            d = 0.5 * ((Pc[i][None] * (lP[i][None] - lM)).sum(-1)
                       + (Pc * (lP - lM)).sum(-1))
            D[i] = d.mean(dim=(1, 2)).cpu().numpy()
        return Ei, Eo, np.maximum(D, 0.0)

    def acts_at(leaf_t, wc, pos, blocks, chunk=256):
        out = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, leaf_t.shape[0], chunk):
                x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
                _, _, inter = model(x, return_intermediates=True)
                for b in blocks:
                    out[b].append(inter[b][:, pos, :].float())
        return {b: torch.cat(vs) for b, vs in out.items()}

    def probe_levels(wc, anchor, pos, blocks, full):
        a = acts_at(pr_leaf_t, wc, pos, blocks)
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

    # ------------------------------------------------------------------ #
    # THE OUTER LOOP's own instrument: the policy's priced read
    # ------------------------------------------------------------------ #
    n_pol = min(cfg["n_policy"], ev_leaf.shape[0])
    pol_leaf_t = ev_leaf_t[:n_pol]
    pol_z = ev_z[:n_pol]
    bayes_idx = float(bayes_none[key_lo:key_hi].mean())

    @torch.no_grad()
    def pol_nll_idx(mode, q, qp):
        """Indexed-span NLL on the policy's probe subset, under one wall condition."""
        wc = wcol(ev_leaf, pol_z, q, qp, mode, MODES.index(mode), n=n_pol)
        tot = torch.zeros(T, device=device)
        for i in range(0, n_pol, 256):
            x = make_x(pol_leaf_t[i:i + 256], wc[i:i + 256]).to(device)
            y = pol_leaf_t[i:i + 256].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            tot += nll.sum(0)
        return float((tot / n_pol).cpu().numpy()[key_lo:key_hi].mean())

    def probe_levels_at(wc, anchor, pos, blocks, levels):
        """`probe_levels` restricted to named levels — the policy needs d5, not all six."""
        a = acts_at(pr_leaf_t, wc, pos, blocks)
        res = {}
        for ell in levels:
            best = 0.0
            for b in blocks:
                best = max(best, _probe_acc(a[b], y_lvl[anchor][ell], v, device,
                                            cfg["probe_steps"], cfg["probe_lr"]))
            res[f"d{L - ell}"] = float(best)
        return res

    def yield_read():
        """RUNG A-1/2's drive: the learner's NEXT-LEVEL YIELD — probe recovery of the
        latent one level ABOVE the keyed one (d5, the keyed node's parent) at the key
        anchor, under the NEUTRAL wall. d4 (the keyed latent itself) is fitted in the
        same pass and logged for science but is NOT what the policy reads or is charged
        for. d6 is at chance in this world and is never read.

        `_probe_acc` initialises a fresh classifier, i.e. it draws from the GLOBAL torch
        RNG. Two consequences handled here: (1) the draw is pinned to a fixed seed so the
        read is a deterministic function of the model — which matters because d5's
        dynamic range is small and probe-init noise would otherwise swamp the paired
        contrast; (2) the global RNG state is saved and restored around the call, so the
        policy consumes no global randomness at all and the donor's zero-global-RNG
        design survives intact."""
        st = torch.get_rng_state()
        torch.manual_seed(cfg["seed"] + 90001)
        try:
            wc = wcol(pr_leaf, pr_z, 0, 0, "none", MODES.index("none"))
            return probe_levels_at(wc, "key", key_plen, pblocks, [L - 5, L - 4])
        finally:
            torch.set_rng_state(st)

    # ---- the checkpoint / decision grids ----
    rots = W.rotation_steps(phase1, rot_period, max_steps)
    merge_at = sched.get("merge_at", max_steps + 1)
    cheap = set(range(0, max_steps + 1, cfg["ckpt_every"])) | {max_steps}
    if rot_period >= 4 * cfg["post_rot_b"]:
        for r in rots:
            cheap |= {r + d for d in (cfg["post_rot_a"], cfg["post_rot_b"])
                      if r + d <= max_steps}
    probe_ck = {int(x) for x in cfg["probe_ckpts"].split(",")
                if x.strip() and int(x) <= max_steps} | {max_steps}
    full_ck = {int(x) for x in cfg["full_ckpts"].split(",")
               if x.strip() and int(x) <= max_steps} | {max_steps}
    decide_at = set(range(0, max_steps + 1, cfg["decide_every"]))
    n_decisions = len(decide_at)

    pol = P.build_policy(sched["policy"], n_decisions)
    budget = P.Budget(cfg["budget_total"], cfg["probe_price"],
                      prices={"path": cfg["probe_price"],
                              "yield": cfg["yield_price"]})
    condition = False                      # False = KEEP the wall; True = COLLAPSE it
    n_switch, t_dec = 0, 0
    trace = []

    torch.manual_seed(seed)
    model = GPT(V, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    train_rng = np.random.default_rng(seed + 5551)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    print(f"===== arm {arm}  kind={kind} phase1={phase1} rot_period={rot_period} "
          f"seed={seed}  policy={sched['policy']}  budget={budget.total}"
          f"@{budget.price}/probe  rotations at {rots[:8]}"
          f"{'...' if len(rots) > 8 else ''} =====", flush=True)

    log, pool_leaf, pool_z, started = [], None, None, time.time()
    for step in range(max_steps + 1):
        stop = step >= max_steps or budget.exhausted(step)
        if step in cheap or stop:
            model.eval()
            q = W.q_at(step, phase1, rot_period, rot_step, v)
            qp = W.q_prev_at(step, phase1, rot_period, rot_step, v)

            # ---- task error + the index instruments (FREE: identical across arms) ----
            npos = {}
            for mi, mode in enumerate(MODES):
                npos[mode] = nll_per_pos(ev_leaf_t, wcol(ev_leaf, ev_z, q, qp, mode, mi))

            # ---- THE DECISION for the interval [step, step + decide_every) ----
            dec = None
            if step in decide_at and not stop:
                reads, yinfo = {}, None
                nd = pol.needs_at(t_dec)
                if "task" in nd:
                    mo = "none" if (condition or kind == "none") else "true"
                    reads["task"] = pol_nll_idx(mo, q, qp)
                if "path" in nd:
                    reads["path"] = pol_nll_idx("none", q, qp) - bayes_idx
                if "yield" in nd:
                    yinfo = yield_read()
                    # as an ERROR, so every read shares the "lower is better" convention
                    reads["yield"] = 1.0 - yinfo["d5"]
                spent = budget.charge(nd)
                action, info = pol.decide(t_dec, step, reads)
                new_cond = action == P.COLLAPSE
                switched = new_cond != condition
                dec = {"t": t_dec, "step": step, "action": action, "switched": switched,
                       "charged": spent, "spend": budget.spend,
                       "n_probes": budget.n_probes, "train_cap": budget.train_cap,
                       "reads": reads, **({} if yinfo is None else
                                          {"yield_d5": yinfo["d5"],
                                           "yield_d4": yinfo["d4"]}), **info}
                trace.append(dec)
                t_dec += 1
                if switched and n_switch < cfg["micro_cap"]:
                    # resolve the op's own transient wherever the policy fires it — the
                    # same micro-grid the donor hard-codes around `merge_at`
                    cheap |= {step + d for d in (25, 50, 75) if step + d <= max_steps}
                    probe_ck |= {step}
                if switched:
                    n_switch += 1
                condition = new_cond

            merged = bool(condition)
            rec = {"step": step, "tokens": step * B * T, "q": int(q), "q_prev": int(qp),
                   "era": int(W.era_at(step, phase1, rot_period)),
                   "merged": merged,
                   "consumed": "none" if (merged or kind == "none") else "true",
                   "spend": budget.spend, "n_probes": budget.n_probes}
            if dec is not None:
                rec["decision"] = dec

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

            Ei, Eo, D = transfer_and_cardinality()
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
                "e_cur": float(diag_cur[live].mean()), "e_old": float(diag_old[live].mean()),
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
            Pm = wall_conditionals()
            D0m = W.jsd_matrix(Pm)
            rec["index"]["jsd0_mean"] = float(D0m[off].mean())
            rec["index"]["jsd0_ratio"] = float(
                D0m[off].mean() / max(refs["exact_jsd_mean"], 1e-12))

            if (step in probe_ck or stop) and cfg["probe_steps"] > 0:
                full = step in full_ck or stop
                blocks = all_blocks if full else pblocks
                lv = {}
                for mode in ("true", "rand", "none"):
                    wc = wcol(pr_leaf, pr_z, q, qp, mode, MODES.index(mode))
                    lv[mode] = {"key": probe_levels(wc, "key", key_plen, blocks, full),
                                "last": probe_levels(wc, "last", last_plen, blocks, full)}
                rec["levels"], rec["full_probe"] = lv, bool(full)

            log.append(rec)
            lvtxt = ""
            if "levels" in rec:
                lt = rec["levels"]["true"]["key"]; lr = rec["levels"]["rand"]["key"]
                ln = rec["levels"]["none"]["key"]
                lvtxt = ("  key-anchor d4 true/rand/none "
                         f"{lt['d4']:.3f}/{lr['d4']:.3f}/{ln['d4']:.3f}")
            dtxt = ""
            if dec is not None and dec.get("rule") in ("bandit", "paired"):
                # NB: Q is None for an action the policy has never observed (an untried
                # action is never chosen greedily), and `reward` is None at the first
                # decision point — so every field here must be None-safe.
                def _f(x, w=7):
                    return f"{'nan':>{w}s}" if x is None else format(x, f"+{w}.4f")
                dtxt = (f" | POL {dec['action'][:4]}({dec['why'][:4]}) "
                        f"e {_f(dec.get('e'))} D {_f(dec.get('contrast'))}"
                        f" V {_f(dec.get('V'))} sp {budget.spend}")
                if dec.get("yield_d5") is not None:
                    dtxt += f" d5 {dec['yield_d5']:.3f} d4 {dec['yield_d4']:.3f}"
            cons = rec["consumed"]
            print(f"[{arm:15s} s{step:6d} q{q:2d}{'M' if merged else ' '}] "
                  f"nll idx {rec['nll'][cons]['idx']:.4f} "
                  f"out {rec['nll'][cons]['out']:.4f} | bind {rec['binding']['idx']:+.4f} "
                  f"vsnone {rec['binding']['vs_none_idx']:+.4f} | gap "
                  f"{rec['index']['transfer_gap']:+.4f} jsd {rec['index']['jsd_mean']:.4f} "
                  f"card {rec['index']['card']['tau0.001']:2d} "
                  f"map {rec['index']['map_match_cur']:.2f}/{rec['index']['map_match_old']:.2f}"
                  + lvtxt + dtxt, flush=True)
            with open(os.path.join(outdir, f"{arm}.json"), "w") as fh:
                json.dump({"arm": arm, "sched": sched, "log": log, "trace": trace,
                           "budget": {"total": budget.total, "price": budget.price,
                                      "spend": budget.spend, "n_probes": budget.n_probes,
                                      "by_read": budget.by_read,
                                      "train_cap": budget.train_cap},
                           "summary": P.summarise_trace(trace, step),
                           "terminal_step": step,
                           "complete": bool(stop)}, fh, indent=2, cls=NumpyEncoder)
            volume.commit()
            model.train()

        if stop:
            break

        if step % cfg["fresh_every"] == 0:
            need = max(64, B * cfg["fresh_every"])
            fl, flf, _ = _generate_with_traces(
                rules, need, cfg["data_seed"] + 100_003 * (step // cfg["fresh_every"] + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))

        # `ix` is drawn FIRST and unconditionally, so the condition branch below cannot
        # move the batch-sampler stream (the donor's bit-identity property, preserved).
        ix = torch.randint(0, pool_leaf.shape[0], (B,), generator=gen)
        leaf_b, z_b = pool_leaf[ix], pool_z[ix]
        q = W.q_at(step, phase1, rot_period, rot_step, v)
        if condition or kind == "none":
            # the merge op, for one interval: sixteen addresses collapse to one, no
            # parameter touched, batches token-identical to `no_wall`'s while it holds
            wc = torch.full((B,), W.neutral_tok(v), dtype=torch.long)
        elif kind == "dead":
            wc = torch.from_numpy(
                (W.wall_tok(train_rng.integers(0, v, size=B), v)).astype(np.int64))
        else:
            # RESTORE SEMANTICS: the wall map is the CURRENT era's, computed from the
            # global step — never a stale snapshot taken when the wall was last held.
            wc = W.wall_tok((z_b + q) % v, v)
        x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg["log_interval"] == 0:
            print(f"  {arm} {step:6d} ntp {loss.item():.4f}", flush=True)

    print(f"[{arm}] DONE in {time.time() - started:.0f}s  terminal_step={log[-1]['step']} "
          f"spend={budget.spend} probes={budget.n_probes}", flush=True)
    return {"arm": arm, "elapsed": time.time() - started, "n_ckpt": len(log),
            "terminal_step": log[-1]["step"], "spend": budget.spend,
            "n_probes": budget.n_probes,
            "summary": P.summarise_trace(trace, log[-1]["step"])}


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=64800, memory=16384)
def slot(
    tag: str = "smoke",
    arms: str = ("wall,no_wall,merge_8000,outer_task,outer_path,outer_path_lp,"
                 "outer_path_true"),
    # DGP / model / training — the donor's, verbatim
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    key_level: int = 2, key_node: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    max_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, data_seed: int = 7, seed: int = 42,
    fresh_every: int = 500,
    phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
    # measurement
    n_eval: int = 4096, eval_seed: int = 999,
    n_probe: int = 4000, probe_seed: int = 1001, n_card: int = 256,
    min_cell_n: int = 24,
    ckpt_every: int = 125, post_rot_a: int = 50, post_rot_b: int = 100,
    probe_ckpts: str = ("250,1000,2000,3000,4000,5000,6000,8000,8250,8500,9000,10000,"
                        "12000,13000,14000,14250,16000,18000,19000"),
    full_ckpts: str = "8000",
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7",
    log_interval: int = 2000,
    # the outer loop
    decide_every: int = 125, n_policy: int = 1024, probe_price: int = 6,
    yield_price: int = 77, budget_total: int = 0, micro_cap: int = 6,
    policy_round: int = 2, policy_burn: int = 0, policy_trial_every: int = 0,
    donor_tag: str = "fwlm1", merge_override: int = 0, quick: bool = False,
):
    if quick:
        max_steps, n_eval, n_probe = 1400, 768, 384
        phase1, rot_period, fresh_every = 300, 300, 100
        ckpt_every, decide_every, probe_steps, mlp_steps = 100, 100, 100, 100
        probe_ckpts, full_ckpts, log_interval = "300,600", "1200", 200
        policy_burn, policy_trial_every = 2, 6
        n_card, min_cell_n, n_policy = 128, 8, 128
        merge_override = 300

    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    specs = specs_for(policy_round)
    for a in arm_list:
        if a not in specs:
            raise ValueError(f"unknown arm {a}; known: {sorted(specs)}")
    if decide_every % ckpt_every != 0:
        raise ValueError("decide_every must be a multiple of ckpt_every "
                         "(decisions must land on checkpoints)")

    cfg = dict(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, key_level=key_level,
               key_node=key_node, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               max_steps=max_steps, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, data_seed=data_seed, seed=seed,
               fresh_every=fresh_every, rot_step=rot_step,
               phase1=phase1, rot_period=rot_period, n_eval=n_eval,
               eval_seed=eval_seed, n_probe=n_probe, probe_seed=probe_seed, n_card=n_card,
               min_cell_n=min_cell_n,
               ckpt_every=ckpt_every, post_rot_a=post_rot_a, post_rot_b=post_rot_b,
               probe_ckpts=probe_ckpts, full_ckpts=full_ckpts, probe_steps=probe_steps,
               probe_lr=probe_lr, mlp_hidden=mlp_hidden, mlp_steps=mlp_steps,
               probe_blocks=probe_blocks, log_interval=log_interval,
               decide_every=decide_every, n_policy=n_policy, probe_price=probe_price,
               yield_price=yield_price, policy_round=policy_round,
               budget_total=budget_total or max_steps, micro_cap=micro_cap)

    print(f"{'=' * 78}\nteacher_slot/decision  tag={tag}  {setting_key(v, s, depth, m)}  "
          f"{n_layer}L/{n_head}H/{n_embd}D  T={s ** depth}")
    print(f"  arms={arm_list}  max_steps={max_steps} "
          f"(tokens={max_steps * batch_size * s ** depth:,})")
    print(f"  key = level {key_level} node {key_node}; phase1={phase1} "
          f"rot_period={rot_period} rot_step={rot_step}")
    print(f"  decision cadence {decide_every}; budget {cfg['budget_total']} steps; "
          f"pathway probe {probe_price} steps on {n_policy} sequences; "
          f"yield (d5) probe {yield_price} steps; task read free  [policy round "
          f"{policy_round}]"
          f"\n{'=' * 78}", flush=True)

    refs, refs_src = load_refs(cfg, donor_tag)
    print(f"\n--- GATE 0 (exact, model-free), source `{refs_src}` ---")
    for k, val in refs["gate0"].items():
        print(f"  {k:28s} {val:+.4f}")
    print(f"  probe ceilings, key anchor  no-wall {refs['ceil']['key_none']}")
    print(f"  probe ceilings, key anchor  wall    {refs['ceil']['key_wall']}\n", flush=True)

    scheds = {a: arm_schedule(a, phase1, rot_period, max_steps, merge_override,
                              round_=policy_round) for a in arm_list}
    for a, sc in scheds.items():          # smoke-only: make trials fire on a short grid
        if policy_burn:
            sc["policy"]["burn"] = policy_burn
        if policy_trial_every:
            sc["policy"]["trial_every"] = policy_trial_every
    for a in arm_list:
        print(f"  {a:16s} {scheds[a]}")
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # MERGE, don't clobber: re-running a subset of a tag's arms (e.g. after one arm
    # crashed) must leave the arms that already completed listed in setup.json.
    prev = {}
    sp = os.path.join(outdir, "setup.json")
    if os.path.exists(sp):
        try:
            prev = json.load(open(sp))
        except Exception:                                     # noqa: BLE001
            prev = {}
    all_arms = list(dict.fromkeys(list(prev.get("arms", [])) + arm_list))
    all_scheds = {**prev.get("scheds", {}), **scheds}
    with open(sp, "w") as fh:
        json.dump({"config": cfg, "arms": all_arms, "scheds": all_scheds, "refs": refs,
                   "refs_source": refs_src, "arms_this_launch": arm_list,
                   "rotations": {a: _rots(all_scheds[a], max_steps) for a in all_arms}},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    started = time.time()
    handles = [(a, run_arm.spawn(tag, a, scheds[a], refs, cfg)) for a in arm_list]
    out = {}
    for a, h in handles:
        try:
            out[a] = h.get()
        except Exception as exc:                              # noqa: BLE001
            out[a] = {"arm": a, "error": repr(exc)}
            print(f"[driver] arm {a} FAILED: {exc!r}", flush=True)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n{json.dumps(out, indent=2)}\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}\n"
          f"{json.dumps(out, indent=2)}")
    return out


def _rots(sched, max_steps):
    from rhm.practice.fourwall.lm import wall as W
    return W.rotation_steps(sched["phase1"], sched["rot_period"], max_steps)


# --------------------------------------------------------------------------- #
# the probe price: measured, not assumed
# --------------------------------------------------------------------------- #

@app.function(gpu="L4", timeout=1800, memory=16384)
def bench(v: int = 16, s: int = 2, depth: int = 6, n_layer: int = 8, n_head: int = 8,
          n_embd: int = 256, batch_size: int = 64, n_policy: int = 1024,
          n_eval: int = 4096, reps: int = 40, n_probe: int = 4000,
          probe_steps: int = 600,
          probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7"):
    """What a pathway probe actually costs, in units of one training step, on this GPU.

    The ledger's price should be the probe's TRUE compute cost, not a guess: one
    forward-only pass over `n_policy` sequences vs one forward+backward on a batch of
    `batch_size`. Reported for the full `n_eval` instrument sweep too, so the choice of a
    1024-sequence policy probe is visible as a design decision rather than a fudge.
    """
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.practice.fourwall.lm import wall as W

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, V = s ** depth, W.vocab_size(v)
    torch.manual_seed(0)
    model = GPT(V, T, n_layer, n_head, n_embd).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    xb = torch.randint(0, V, (batch_size, T), device=dev)
    yb = torch.randint(0, v, (batch_size, T), device=dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        _, loss = model(xb, yb); opt.zero_grad(); loss.backward(); opt.step()
    sync(); t0 = time.time()
    for _ in range(reps):
        _, loss = model(xb, yb); opt.zero_grad(); loss.backward(); opt.step()
    sync(); t_train = (time.time() - t0) / reps

    model.eval()

    @torch.no_grad()
    def evalpass(n, chunk=256):
        x = torch.randint(0, V, (n, T), device=dev)
        y = torch.randint(0, v, (n, T), device=dev)
        for i in range(0, n, chunk):
            logits, _ = model(x[i:i + chunk])
            F.cross_entropy(logits.reshape(-1, V), y[i:i + chunk].reshape(-1),
                            reduction="none")

    evalpass(n_policy); sync(); t0 = time.time()
    for _ in range(5):
        evalpass(n_policy)
    sync(); t_pol = (time.time() - t0) / 5
    evalpass(n_eval); sync(); t0 = time.time()
    for _ in range(3):
        evalpass(n_eval)
    sync(); t_full = (time.time() - t0) / 3

    # --- the RUNG A-1/2 yield read: activations at the key anchor for `probe_blocks`,
    #     then one linear probe fit per block for d5 (and, logged free, d4) ---
    from rhm.rhm_latent_loop import _probe_acc
    nb = len(probe_blocks.split(","))
    blocks = [b.strip() for b in probe_blocks.split(",")]

    def yieldpass(levels=1, chunk=256):
        acts = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, n_probe, chunk):
                x = torch.randint(0, V, (min(chunk, n_probe - i), T), device=dev)
                _, _, inter = model(x, return_intermediates=True)
                for b in blocks:
                    acts[b].append(inter[b][:, T - 1, :].float())
        A = {b: torch.cat(vs) for b, vs in acts.items()}
        y = torch.randint(0, v, (n_probe,), device=dev)
        for _ in range(levels):
            for b in blocks:
                _probe_acc(A[b], y, v, dev, probe_steps, 1e-2)

    yieldpass(1); sync(); t0 = time.time()
    for _ in range(2):
        yieldpass(1)
    sync(); t_y5 = (time.time() - t0) / 2
    sync(); t0 = time.time()
    for _ in range(2):
        yieldpass(2)
    sync(); t_y54 = (time.time() - t0) / 2

    out = {"t_train_step_s": t_train, "t_policy_probe_s": t_pol,
           "t_yield_probe_d5_s": t_y5, "t_yield_probe_d5_and_d4_s": t_y54,
           "price_yield_probe_steps": t_y5 / t_train,
           "price_yield_d5_and_d4_steps": t_y54 / t_train,
           "n_probe": n_probe, "probe_blocks": nb, "probe_steps": probe_steps,
           "t_full_eval_pass_s": t_full,
           "price_policy_probe_steps": t_pol / t_train,
           "price_full_eval_steps": t_full / t_train,
           "flop_ratio_policy": n_policy / (3.0 * batch_size),
           "flop_ratio_full": n_eval / (3.0 * batch_size)}
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# offline gate: the policy machinery, with no GPU and no substrate
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=1800, memory=8192)
def policy_gate(max_steps: int = 20000, decide_every: int = 125):
    return policy_gate_local(max_steps, decide_every)


def policy_gate_local(max_steps=20000, decide_every=125, verbose=True):
    """P-1  a FixedPolicy reproduces the donor's condition rule `step >= merge_at`
             exactly, at every step of the grid, for every merge step tested.
       P-2  the bandit is symmetric: relabelling the two actions relabels the choices
             (nothing in the rule prefers collapse).
       P-3  both actions are sampled, and every forced trial is tagged.
       P-4  the ledger: task reads are free, pathway reads are priced, and the training
             cap falls by exactly the spend.
       P-5  the trace reduction separates a greedy bail-out from a scheduled trial.
       P-6  the LP mode responds to a sustained fall in the error and is smoother than
             the one-step mode (they are distinct rules, not the same rule twice).
       P-7  ROUND 2: an ABBA block cancels a linear improvement trend EXACTLY -- the
             failure that voided round 1's contrast cannot recur.
       P-8  the paired rule is exactly label-symmetric, mirror world included.
       P-9  it recovers the true per-interval effect size, not just its sign.
       P-10 the ledger: reads happen only at block boundaries, the two horizons are
             matched on perturbation, and the yield read is priced apart from the
             pathway read.
    """
    from rhm.practice.teacher_slot.decision import policy as P
    out = {}
    grid = list(range(0, max_steps + 1, decide_every))
    n = len(grid)

    # P-1
    worst = 0
    for mg in (None, 1000, 8000, 13000):
        pol = P.FixedPolicy(mg)
        for t, st in enumerate(grid):
            a, _ = pol.decide(t, st, {})
            want = (mg is not None and st >= mg)
            worst = max(worst, int((a == P.COLLAPSE) != want))
    out["P1_fixed_matches_donor_rule"] = {"mismatches": worst}
    assert worst == 0, "P-1"

    # P-2 EXACT symmetry under relabelling: run the same rule in a world where COLLAPSE
    # is the good action (starting from KEEP) and in its exact mirror (KEEP good, starting
    # from COLLAPSE). Relabelling one must reproduce the other decision for decision --
    # nothing in the rule, including the status-quo tie-break, prefers collapse.
    def run(spec, good, init, n_dec=60):
        pol = P.build_policy(dict(spec, init_action=init), n)
        acts, trace = [], []
        for t, st in enumerate(grid[:n_dec]):
            e = 1.0 - 0.02 * sum(1 for x in acts if x == good)
            a, info = pol.decide(t, st, {spec["read"]: e})
            acts.append(a); trace.append(info)
        return acts, trace

    spec = {"kind": "bandit", "read": "path", "mode": "delta"}
    a1, t1 = run(spec, P.COLLAPSE, P.KEEP)
    a2, t2 = run(spec, P.KEEP, P.COLLAPSE)
    flip = {P.KEEP: P.COLLAPSE, P.COLLAPSE: P.KEEP}
    n_diff = sum(1 for x, y in zip(a1, a2) if x != flip[y])
    out["P2_symmetry"] = {"n_decisions": len(a1), "n_mismatch_after_relabel": n_diff,
                          "collapse_good_tail": a1[-8:], "keep_good_mirror_tail": a2[-8:],
                          "first_greedy_collapse_t": next(
                              (i for i, d in enumerate(t1)
                               if d["greedy"] == P.COLLAPSE), None)}
    assert n_diff == 0, f"P-2 the rule is not label-symmetric ({n_diff} mismatches)"
    assert a1[-1] == P.COLLAPSE and a2[-1] == P.KEEP, "P-2 the bandit must follow reward"

    # P-3
    pol = P.build_policy(spec, n)
    tr, acts = [], []
    for t, st in enumerate(grid):
        e = 1.0 - 0.02 * sum(1 for x in acts if x == P.KEEP)
        a, info = pol.decide(t, st, {"path": e})
        acts.append(a); tr.append({"t": t, "step": st, "action": a, **info})
    both = next((d["step"] for i, d in enumerate(tr)
                 if len({x["action"] for x in tr[:i + 1]}) == 2), None)
    out["P3_exploration"] = {
        "forced_trial_indices": sorted(pol.forced), "n_forced": len(pol.forced),
        "burn_in_decisions": pol.burn,
        "both_actions_sampled_by_step": both,
        "untried_never_chosen_greedily": all(
            d["why"] != "greedy"
            or d["q_keep" if d["action"] == P.KEEP else "q_collapse"] is not None
            for d in tr),
        "n_untagged": sum(1 for d in tr if d["why"] not in
                          ("burn_in", "greedy", "forced_trial"))}
    assert out["P3_exploration"]["n_untagged"] == 0, "P-3"
    assert both is not None, "P-3 both actions must be sampled"
    assert out["P3_exploration"]["untried_never_chosen_greedily"], "P-3 untried chosen"

    # P-4
    b = P.Budget(max_steps, 6)
    b.charge(("task",)); free_spend = b.spend
    b.charge(("path",)); b.charge(("path",))
    out["P4_ledger"] = {"after_task_read": free_spend, "after_two_probes": b.spend,
                        "n_probes": b.n_probes, "train_cap": b.train_cap,
                        "exhausted_at_cap": b.exhausted(b.train_cap)}
    assert free_spend == 0 and b.spend == 12 and b.train_cap == max_steps - 12, "P-4"

    # P-5
    sm = P.summarise_trace(tr, max_steps)
    out["P5_trace_summary_keep_favoured"] = sm
    assert sm["terminal_action"] == P.KEEP, "P-5"
    tr2, acts2 = [], []
    pol2 = P.build_policy(spec, n)
    for t, st in enumerate(grid):
        e = 1.0 - 0.02 * sum(1 for x in acts2 if x == P.COLLAPSE)
        a, info = pol2.decide(t, st, {"path": e})
        acts2.append(a); tr2.append({"t": t, "step": st, "action": a, **info})
    sm2 = P.summarise_trace(tr2, max_steps)
    out["P5_trace_summary_collapse_favoured"] = sm2
    assert sm2["commit_step"] is not None and sm2["terminal_action"] == P.COLLAPSE, "P-5b"

    # P-6 the two modes are distinct rules
    e_series = [1.0 - 0.02 * min(t, 20) for t in range(n)]
    r_delta, r_lp = [], []
    pd = P.BanditPolicy("path", mode="delta", n_decisions=n)
    pl = P.BanditPolicy("path", mode="lp", n_decisions=n)
    for t in range(40):
        _, i1 = pd.decide(t, grid[t], {"path": e_series[t]})
        _, i2 = pl.decide(t, grid[t], {"path": e_series[t]})
        r_delta.append(i1["reward"]); r_lp.append(i2["reward"])
    tail_d = [x for x in r_delta[22:30] if x is not None]
    tail_l = [x for x in r_lp[22:30] if x is not None]
    out["P6_modes_distinct"] = {
        "delta_reward_after_error_plateaus": tail_d,
        "lp_reward_after_error_plateaus": tail_l,
        "lp_has_memory": bool(max(tail_l) > 1e-6 and max(tail_d) < 1e-9)}
    assert out["P6_modes_distinct"]["lp_has_memory"], "P-6 modes collapsed to one rule"

    # ---------------- round 2: the ABBA paired trial ---------------- #
    spec2 = {"kind": "paired", "read": "path", "span": 1, "trial_every": 16}

    def run_paired(spec, e_fn, n_dec=161):
        pol = P.build_policy(spec, n_dec)
        acts, tr = [], []
        for t in range(n_dec - 1):
            rd = {}
            if pol.needs_at(t):
                rd[spec["read"]] = e_fn(t, acts)
            a, info = pol.decide(t, t * 125, rd)
            acts.append(a); tr.append({"t": t, "step": t * 125, "action": a, **info})
        return acts, tr

    # P-7 EXACT trend cancellation. Feed an error whose improvement rate is linear in
    # time and carries NO action effect at all. Round 1's rule read a contrast of -0.43
    # vs -0.76 out of exactly this; the ABBA block must read 0.
    g0, dl = 0.20, 0.0011
    def e_trend(t, acts):
        return 1.0 - sum(g0 - dl * k for k in range(t))
    _, tr7 = run_paired(spec2, e_trend)
    Ds = [d["contrast"] for d in tr7 if d.get("contrast") is not None]
    out["P7_linear_trend_cancels"] = {
        "n_trials": len(Ds), "max_abs_contrast": max(abs(x) for x in Ds),
        "ambient_improvement_per_interval": g0,
        "round1_style_contrast_on_same_series": abs(
            (g0 - dl * 4) - (g0 - dl * 6)) + 0.0,
        "final_V": tr7[-1]["V"], "n_collapse_chosen": sum(
            1 for d in tr7 if d["action"] == P.COLLAPSE and d["why"] == "greedy")}
    assert out["P7_linear_trend_cancels"]["max_abs_contrast"] < 1e-12, "P-7"
    assert out["P7_linear_trend_cancels"]["n_collapse_chosen"] == 0, "P-7 drifted"

    # P-8 exact label symmetry, including in the mirror world
    def e_act(good, amp):
        def f(t, acts):
            return 1.0 - amp * sum(1 for x in acts if x == good) - 0.02 * t
        return f
    # The trial POLARITY is an exogenous schedule and is not itself relabelled, so the
    # claim under test is about the rule: in the mirror world the estimated contrast must
    # be the exact negation, and every CHOSEN (non-trial) action must be the mirror image.
    a1, t1 = run_paired(spec2, e_act(P.COLLAPSE, 0.01))
    a2, t2 = run_paired(dict(spec2, init_action=P.COLLAPSE), e_act(P.KEEP, 0.01))
    flip = {P.KEEP: P.COLLAPSE, P.COLLAPSE: P.KEEP}
    vpairs = [(d1["V"], d2["V"]) for d1, d2 in zip(t1, t2)
              if d1["V"] is not None and d2["V"] is not None]
    chosen = [(d1, d2) for d1, d2 in zip(t1, t2) if d1["why"] != "trial"]
    out["P8_paired_symmetry"] = {
        "n_chosen_decisions": len(chosen),
        "n_mismatch_after_relabel": sum(1 for d1, d2 in chosen
                                        if d1["action"] != flip[d2["action"]]),
        "max_abs_V_plus_mirror_V": max(abs(x + y) for x, y in vpairs),
        "collapse_good_tail": a1[-6:], "keep_good_mirror_tail": a2[-6:]}
    assert out["P8_paired_symmetry"]["n_mismatch_after_relabel"] == 0, "P-8"
    assert out["P8_paired_symmetry"]["max_abs_V_plus_mirror_V"] < 1e-15, "P-8 V"
    assert a1[-1] == P.COLLAPSE and a2[-1] == P.KEEP, "P-8 must follow the contrast"

    # P-9 the contrast is recovered at the right MAGNITUDE and sign, on top of a trend
    amp = 0.01
    _, tr9 = run_paired(spec2, e_act(P.COLLAPSE, amp))
    D9 = [d["contrast"] for d in tr9 if d.get("contrast") is not None]
    out["P9_contrast_recovered"] = {"true_effect_per_interval": amp,
                                    "estimated": D9[:4], "mean": sum(D9) / len(D9)}
    assert all(abs(x - amp) < 1e-9 for x in D9), "P-9 biased contrast"

    # P-10 the ledger: reads only at block boundaries; matched perturbation across the
    # two horizons; the yield read is priced separately from the pathway read
    rows = {}
    for lab, sp, te in (("span1", 1, 16), ("span2", 2, 32)):
        pol = P.build_policy({"kind": "paired", "read": "path", "span": sp,
                              "trial_every": te}, 161)
        blk = 4 * sp
        rows[lab] = {"n_trials": len(pol.trials), "trial_intervals": len(pol.trials) * blk,
                     "non_greedy_intervals": len(pol.trials) * blk // 2,
                     "reads": len(pol.read_at),
                     "needs_off_boundary": sum(1 for t in range(160)
                                               if t not in pol.read_at and pol.needs_at(t))}
    b = P.Budget(20000, 6, prices={"path": 6, "yield": 40})
    b.charge(("task",)); b.charge(("path",)); b.charge(("yield",))
    rows["ledger"] = {"spend": b.spend, "by_read": b.by_read, "n_probes": b.n_probes}
    out["P10_paired_ledger"] = rows
    assert rows["span1"]["trial_intervals"] == rows["span2"]["trial_intervals"], "P-10"
    assert rows["span1"]["needs_off_boundary"] == 0 and \
        rows["span2"]["needs_off_boundary"] == 0, "P-10 reads off boundary"
    assert b.spend == 46 and b.by_read["task"] == 0, "P-10 ledger"

    if verbose:
        print(json.dumps(out, indent=2, default=str))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    slot.remote(quick=quick, tag="smoke_local")
