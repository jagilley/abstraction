"""Curiosity v1 — the keystone: does a reducible-DISAGREEMENT outer loop shape the inner
FM and re-open the frontier under non-stationarity, on a CONTROL task?  (the merged
Step 1+2 of ideas/two_timescale_value_loop.md; parent CURIOSITY_DRIVE_README.md + the
reaching arc REACHING_LOOKAHEAD_README.md).

Everything before this ran on the active-vision TELEPORT toy, where acting == planning
(the world model just predicts glimpse pixels), so FM *shaping* (veridicality vs
value-relevance) is unmeasurable. This ports the drive to the reaching CONTROLLER, where
acting != planning, and asks the disc-4 question the toy structurally can't:

  Under engineered non-stationarity of the CONTROLLED DYNAMICS, does an intrinsic
  reducible-disagreement drive that allocates EXPLORATION (which action-effects to probe)
  re-shape the inner FM toward the currently-reducible / control-relevant directions in
  a way a stationary extrinsic (goal-distance) value provably cannot -- and does goal-
  reaching sawtooth-and-recover across drifts rather than saturating once?

--- Why drift the DYNAMICS, not the content (the crux) --------------------------------
The reaching FM could predict two things: (a) the reveal CONTENT (digit pixels under the
fovea) -- a distractor, answer-IRRELEVANT (forecasting it buys zero control; this is why
the ACTIVE_VISION causal arm was null); (b) the DYNAMICS ("if I command up, where does the
fovea land?") -- the controlled variable, whose prediction IS what lets you plan. Learning
progress only earns its keep on (b). So we drift (b).

--- Visuomotor rotation: intended vs realized (preserves the arity substrate) ----------
The existing efference marker places its cue at the TRUE next cell -> the FM is handed
where the action lands, so drifting the map would be INVISIBLE to it. Fix: split the map.
  - CANONICAL map (fixed): what the efference marker encodes -- the INTENDED movement.
  - ENV map (drifts): what actually happens -- the REALIZED movement (a remapping /
    "visuomotor rotation" of the controls, redrawn each drift_period).
The FM must learn the canonical->realized correction; a drift re-scrambles it; re-learning
it is exactly what restores goal-reaching. This is the prism-adaptation / visuomotor-
rotation paradigm (textbook cerebellar learning-to-recalibrate), and it leaves the
spatial-efference-marker substrate the "arity, not resolution" arc rests on untouched.

--- Design (see DRIFTING_REACHING_SPEC.md) --------------------------------------------
  * Drive = position-decoded ensemble DISAGREEMENT (variance ACROSS FMs of the predicted
    next-POSITION), the control-relevant / value-relevant slice -- so the explore tap
    won't chase the answer-irrelevant content distractor. Content stays ON (it is the
    value-irrelevant axis disc-4 needs to dissociate veridicality from value-relevance).
    Magnitude, not the LP derivative: Step 0 (curiosity_scarcity swap-vs-morph) showed the
    derivative is drift-fragile and the disagreement magnitude is drift-robust.
  * noisy-TV control = a "slip" action whose realized effect is RANDOM: the ensemble
    AGREES it is unpredictable -> low disagreement (epistemic != aleatoric) -> the drive
    ignores it. The reaching-native noisy-TV, in the CONTROL channel (curiosity's was in
    pixels).
  * Frozen, briefly-pretrained controller (position-aware rep) so disc-4 isolates FM
    shaping, not representation drift. Scarcity is free from an OPEN arena: greedy reaching
    only ever exercises the ~2 goalward actions, so the rest of the action-effect map is
    under-sampled exactly as the theory needs.
  * Arms (each runs its OWN online FM ensemble on the SHARED drift + data stream; the only
    difference is the action policy):
      extrinsic : greedy on goal-value (no exploration)          -- passive re-learning
      eps       : eps-greedy on goal-value (UNDIRECTED explore)  -- fair explore control
      two_tap   : goal-value + disagreement (DIRECTED explore)   -- the headline agent
      explore   : disagreement only (ignores goal)               -- pure dynamics-ID ref

Run:
  modal run a2a_forward/reaching/curiosity_reaching_control.py::curiosity_reaching_control --quick
  modal run --detach a2a_forward/reaching/curiosity_reaching_control.py::curiosity_reaching_control --dataset mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def curiosity_reaching_control(
    dataset: str = "mnist",
    patch_size: int = 4,
    n_embd: int = 128,
    n_head: int = 4,
    n_loop_layers: int = 1,
    glimpse_grid: int = 3,
    horizon: int = 12,                 # episode length
    with_content: bool = True,         # content distractor = the value-IRRELEVANT axis (keep ON)
    # --- controller pretrain (canonical, stationary) then FREEZE ---
    pretrain_steps: int = 1500,
    probe_steps: int = 1500,
    pretrain_lr: float = 3e-4,
    # --- online FM ensemble ---
    ensemble_k: int = 4,
    fm_d_head: int = 32,
    fm_n_head: int = 4,                # FM capacity: learning a position-conditional spatial
    fm_n_layer: int = 2,              # transform needs more than 1 layer/1 head
    fm_mlp_mult: float = 2.0,
    fwd_lr: float = 1e-3,
    bootstrap_frac: float = 0.6,
    # --- non-stationarity (visuomotor remap of the controls) ---
    drift_period: int = 120,
    n_iters: int = 960,
    with_slip: bool = True,            # noisy-TV action (irreducible realized effect)
    # --- drive ---
    eff_scale: float = 1.0,            # efference magnitude (state is O(1); too-small -> arity-1 collapse)
    pos_aux_w: float = 1.0,            # position-legibility aux during pretrain (make the frozen rep decodable)
    explore_w: float = 0.5,            # two_tap blend weight on disagreement (in [0,1])
    eps_explore: float = 0.25,         # undirected-explore rate for the `eps` arm
    arms: str = "extrinsic,eps,two_tap,explore",
    batch_size: int = 128,
    eval_interval: int = 20,
    eval_batch: int = 128,
    seed: int = 42,
    quick: bool = False,
):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from a2a_forward.reaching.reaching_vit import ReachingLoopedViT
    from a2a_forward.forward_model import TransformerForwardModel

    if quick:
        pretrain_steps, probe_steps, n_iters = 200, 200, 240
        drift_period, batch_size, ensemble_k = 60, 64, 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = 28 // patch_size
    n_patches = grid ** 2
    n_positions = n_patches + 1
    K = horizon
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]

    # Action set: stay, up, down, left, right (+ slip). `slip` is the noisy-TV action.
    CANON_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    CARDINALS = [1, 2, 3, 4]            # indices of the 4 remappable directions
    SLIP = None
    if with_slip:
        SLIP = len(CANON_DELTAS)        # index 5
        CANON_DELTAS = CANON_DELTAS + [(0, 0)]   # canonical cue for slip == stay (disambiguated by act_id)
    n_actions = len(CANON_DELTAS)
    drow = torch.tensor([d[0] for d in CANON_DELTAS], device=device)
    dcol = torch.tensor([d[1] for d in CANON_DELTAS], device=device)

    print(f"CURIOSITY REACHING-CONTROL on {dataset}, {device}. grid={grid}x{grid} "
          f"K={K} n_actions={n_actions} slip={SLIP} arms={arm_list} K_ens={ensemble_k} "
          f"drift@{drift_period} iters={n_iters}")

    # ------------------------------------------------------------------
    # Open-arena kinematics.  canonical (fixed) vs env (drifting remap).
    # ------------------------------------------------------------------
    def canonical_apply(p, a):
        """The INTENDED move (fixed map). Open arena: clamp at edges."""
        pr, pc = p // grid, p % grid
        nr = (pr + drow[a]).clamp(0, grid - 1)
        nc = (pc + dcol[a]).clamp(0, grid - 1)
        return nr * grid + nc

    # A "control remap" = a permutation of the 4 cardinal deltas (24 possible). Redrawn
    # abruptly each drift segment (visuomotor rotation is the cyclic subset).
    rng_np = np.random.RandomState(seed + 7)
    n_segments = n_iters // max(drift_period, 1) + 2
    perms = []
    prev = None
    for _ in range(n_segments):
        if not perms:
            perm = list(CARDINALS)                 # seg 0 = identity (canonical==env): clean initial learning
        else:
            while True:
                q = list(CARDINALS)
                rng_np.shuffle(q)
                if q != prev:
                    break
            perm = q
        perms.append(perm)
        prev = perm
    # env delta table per segment: action -> (dr,dc). cardinals permuted; stay/slip fixed.
    env_drow = torch.zeros(n_segments, n_actions, dtype=torch.long, device=device)
    env_dcol = torch.zeros(n_segments, n_actions, dtype=torch.long, device=device)
    for si, perm in enumerate(perms):
        for a in range(n_actions):
            if a in CARDINALS:
                src = perm[CARDINALS.index(a)]     # this action now realizes cardinal `src`'s delta
                env_drow[si, a] = CANON_DELTAS[src][0]
                env_dcol[si, a] = CANON_DELTAS[src][1]
            else:
                env_drow[si, a] = CANON_DELTAS[a][0]
                env_dcol[si, a] = CANON_DELTAS[a][1]

    def env_apply(p, a, seg, slip_gen):
        """The REALIZED move under the current segment's remap; `slip` -> random cell."""
        pr, pc = p // grid, p % grid
        nr = (pr + env_drow[seg, a]).clamp(0, grid - 1)
        nc = (pc + env_dcol[seg, a]).clamp(0, grid - 1)
        nxt = nr * grid + nc
        if SLIP is not None:
            is_slip = (a == SLIP)
            rand_cell = torch.randint(n_patches, (p.shape[0],), generator=slip_gen).to(device)
            nxt = torch.where(is_slip, rand_cell, nxt)
        return nxt

    # Manhattan value table (open arena -> Manhattan is exact geodesic).
    ar = torch.arange(n_patches, device=device)
    rr, ccol = ar // grid, ar % grid
    manh = ((rr[:, None] - rr[None, :]).abs() + (ccol[:, None] - ccol[None, :]).abs()).float()

    def sample_pg(B, gen):
        """start p0, goal g among all cells with a minimum separation."""
        p = torch.randint(n_patches, (B,), generator=gen).to(device)
        g = torch.randint(n_patches, (B,), generator=gen).to(device)
        for _ in range(8):
            bad = manh[p, g] < 3
            if not bad.any():
                break
            g = torch.where(bad, torch.randint(n_patches, (B,), generator=gen).to(device), g)
        return p, g

    # --- content distractor (value-irrelevant axis) ---
    train_images = None
    if with_content:
        from datasets import load_dataset
        ds_name = {"mnist": "ylecun/mnist", "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
        ds = load_dataset(ds_name)
        tr = np.stack([np.array(im) for im in ds["train"]["image"]])
        train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
        print(f"  loaded {len(train_images)} content images")

    def sample_patch_emb(model, B, gen):
        if not with_content:
            return None
        idx = torch.randint(len(train_images), (B,), generator=gen)   # CPU idx for CPU tensor
        return model._patch_embeds(train_images[idx].to(device))

    # ------------------------------------------------------------------
    # Controller: pretrain (imitate the canonical oracle) then FREEZE.
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    model = ReachingLoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_embd=n_embd, n_head=n_head,
        n_loop_layers=n_loop_layers, n_steps=K, glimpse_grid=glimpse_grid,
        with_content=with_content, n_actions=n_actions).to(device)

    def canon_oracle(p, g):
        """greedy-Manhattan first action under the CANONICAL map (open arena optimal)."""
        succ = torch.stack([canonical_apply(p, torch.full_like(p, a)) for a in range(n_actions)], 0)
        md = manh[succ, g.unsqueeze(0).expand(n_actions, -1)]
        pen = torch.zeros(n_actions, 1, device=device); pen[0] = 0.01        # tiny stay penalty
        if SLIP is not None:
            pen[SLIP] = 0.02                                                  # never imitate slip
        return (md + pen).argmin(0)

    def roll_canon(m, p0, g, patch, action_fn):
        s = m.init_state(p0.shape[0], device)
        p = p0.clone()
        states, poss = [], []
        for t in range(K):
            s = m.step(s, p, g, patch)
            states.append(s); poss.append(p.clone())
            a = action_fn(s, p, g)
            p = canonical_apply(p, a)
        return states, poss

    # MLP position probe; jointly trained during pretrain so the FROZEN operator is
    # position-LEGIBLE (a fixed substrate property, shared across arms -- proprioception).
    probe = nn.Sequential(nn.Linear(n_embd, n_embd), nn.GELU(), nn.Linear(n_embd, 1)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()),
                            lr=pretrain_lr, weight_decay=0.01)
    g_pg = torch.Generator().manual_seed(seed + 100)
    g_img = torch.Generator().manual_seed(seed + 101)
    print("=== pretrain controller (canonical imitation + position-legibility aux) ===")
    for step in range(pretrain_steps):
        model.train()
        p0, g = sample_pg(batch_size, g_pg)
        patch = sample_patch_emb(model, batch_size, g_img)
        s = model.init_state(batch_size, device)
        p = p0.clone(); imit = 0.0; posaux = 0.0
        for t in range(K):
            s = model.step(s, p, g, patch)
            imit = imit + F.cross_entropy(model.policy_logits(s), canon_oracle(p, g))
            posaux = posaux + F.cross_entropy(probe(s[:, 1:, :]).squeeze(-1), p)
            with torch.no_grad():
                a = canon_oracle(p, g)
            p = canonical_apply(p, a)
        loss = (imit + pos_aux_w * posaux) / K
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0); opt.step()
        if step % max(pretrain_steps // 5, 1) == 0 or step == pretrain_steps - 1:
            print(f"  [pretrain] step {step:5d}: imit={imit.item()/K:.4f} posaux={posaux.item()/K:.4f}")
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.eval()

    # refine the probe on the FROZEN operator.
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    g_pg2 = torch.Generator().manual_seed(seed + 200)
    g_img2 = torch.Generator().manual_seed(seed + 201)
    g_act2 = torch.Generator().manual_seed(seed + 202)
    print("=== train position probe (frozen operator) ===")
    for step in range(probe_steps):
        p0, g = sample_pg(batch_size, g_pg2)
        patch = sample_patch_emb(model, batch_size, g_img2)
        with torch.no_grad():
            states, poss = roll_canon(
                model, p0, g, patch,
                lambda s, p, gg: torch.randint(n_actions, (p.shape[0],), generator=g_act2).to(device))
        t = torch.randint(K, (1,)).item()
        logits = probe(states[t][:, 1:, :]).squeeze(-1)
        loss = F.cross_entropy(logits, poss[t])
        popt.zero_grad(); loss.backward(); popt.step()
    probe.eval()
    with torch.no_grad():
        p0, g = sample_pg(eval_batch, g_pg2)
        patch = sample_patch_emb(model, eval_batch, g_img2)
        states, poss = roll_canon(
            model, p0, g, patch,
            lambda s, p, gg: torch.randint(n_actions, (p.shape[0],), generator=g_act2).to(device))
        probe_acc = float(np.mean([
            (probe(states[t][:, 1:, :]).squeeze(-1).argmax(-1) == poss[t]).float().mean().item()
            for t in range(K)]))
    print(f"  probe_acc={probe_acc:.3f}")

    # ------------------------------------------------------------------
    # Efference (fixed): spatial marker at the CANONICAL target + fixed per-action code.
    # ------------------------------------------------------------------
    torch.manual_seed(seed + 555)
    # Efference = the action COMMAND (identity), broadcast to all tokens. Deliberately NO
    # spatial cue at a target cell: an intended-cell marker traps the FM into COPYING it
    # (predicting the intended cell), so under a control remap it never learns the REALIZED
    # effect -- validated: with a cell marker, permuted-segment pos_acc stalls at the
    # chance-of-intended==realized floor (~0.31). With only the command, the FM must LEARN
    # action->effect and can RE-learn it when the map drifts -- the dynamics-learning we want.
    act_id = torch.randn(n_actions, n_embd, device=device) * eff_scale

    def efference(p, a):
        """(B, n_pos, n_embd): action-identity command broadcast to all tokens (arity-2 cue)."""
        return act_id[a].unsqueeze(1).expand(-1, n_positions, -1)

    def value_of(s_hat, g):
        """-E[Manhattan(decoded pos, goal)] read through the probe (exploit tap)."""
        probs = F.softmax(probe(s_hat[:, 1:, :]).squeeze(-1), dim=-1)   # (B, n_patches)
        return -(probs * manh[:, g].t()).sum(-1)

    def build_ensemble(off):
        fms = []
        for k in range(ensemble_k):
            torch.manual_seed(seed + 900 + off * 17 + k)
            fm = TransformerForwardModel(
                d_model=n_embd, d_head=fm_d_head, n_head=fm_n_head, n_layer=fm_n_layer,
                mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
            fms.append(fm)
        opt = torch.optim.AdamW([p for fm in fms for p in fm.parameters()],
                                lr=fwd_lr, weight_decay=0.01)
        return fms, opt

    def action_scores(fms, s, p, g):
        """Per-action exploit value (ensemble-mean FM) and explore disagreement
        (variance across FMs of the decoded next-position distribution)."""
        B = s.shape[0]
        vals, disag = [], []
        for a in range(n_actions):
            av = torch.full((B,), a, device=device)
            inp = s + efference(p, av)
            preds = torch.stack([fm(inp) for fm in fms], 0)            # (K, B, n_pos, n_embd)
            s_hat = s.unsqueeze(0) + preds
            # decoded next-position distribution per ensemble member
            pp = F.softmax(probe(s_hat[:, :, 1:, :]).squeeze(-1), dim=-1)  # (K, B, n_patches)
            vals.append(value_of(s + preds.mean(0), g))                   # exploit on mean FM
            disag.append(pp.var(0).sum(-1))                                # (B,) cross-model variance
        return torch.stack(vals, -1), torch.stack(disag, -1)              # (B, n_actions) each

    def make_action_fn(fms, arm, gen):
        def fn(s, p, g):
            V, D = action_scores(fms, s, p, g)
            B = s.shape[0]
            if arm == "explore":
                score = D
            elif arm == "extrinsic":
                score = V
            elif arm == "eps":
                score = V
            elif arm == "two_tap":
                def nrm(x):
                    lo = x.min(-1, keepdim=True).values
                    hi = x.max(-1, keepdim=True).values
                    return (x - lo) / (hi - lo + 1e-8)
                score = (1 - explore_w) * nrm(V) + explore_w * nrm(D)
            else:
                raise ValueError(arm)
            score = score + 1e-6 * torch.randn_like(score)
            a = score.argmax(-1)
            if arm == "eps":
                rnd = torch.randint(n_actions, (B,), generator=gen).to(device)
                take = (torch.rand(B, generator=gen).to(device) < eps_explore)
                a = torch.where(take, rnd, a)
            return a
        return fn

    def greedy_plan_fn(fms):
        """Common exploit-only planner used to EVAL every arm's FM on equal footing
        (measures FM quality for reaching, independent of the arm's explore policy)."""
        def fn(s, p, g):
            V, _ = action_scores(fms, s, p, g)
            return (V + 1e-6 * torch.randn_like(V)).argmax(-1)
        return fn

    # ------------------------------------------------------------------
    # Metrics on the CURRENT map.
    # ------------------------------------------------------------------
    def eval_metrics(fms, seg, gen_seed):
        g_p = torch.Generator().manual_seed(gen_seed)
        g_i = torch.Generator().manual_seed(gen_seed + 1)
        g_a = torch.Generator().manual_seed(gen_seed + 2)
        g_s = torch.Generator().manual_seed(gen_seed + 3)
        with torch.no_grad():
            # realistic states from a random rollout under the current map
            p0, g = sample_pg(eval_batch, g_p)
            patch = sample_patch_emb(model, eval_batch, g_i)
            s = model.init_state(eval_batch, device); p = p0.clone()
            S_list, P_list = [], []
            for t in range(K):
                s = model.step(s, p, g, patch)
                S_list.append(s); P_list.append(p.clone())
                a = torch.randint(n_actions, (eval_batch,), generator=g_a).to(device)
                p = env_apply(p, a, seg, g_s)
            # FM pos-acc + full-cos over all actions from these states
            per_act_hit = torch.zeros(n_actions); per_act_n = torch.zeros(n_actions)
            hit = tot = 0; coss = []
            for t in range(K):
                s_t, p_t = S_list[t], P_list[t]
                for a in range(n_actions):
                    av = torch.full((eval_batch,), a, device=device)
                    inp = s_t + efference(p_t, av)
                    delta = torch.stack([fm(inp) for fm in fms], 0).mean(0)
                    s_hat = s_t + delta
                    pred_pos = probe(s_hat[:, 1:, :]).squeeze(-1).argmax(-1)
                    true_next = env_apply(p_t, av, seg, torch.Generator().manual_seed(gen_seed + 99))
                    if a == SLIP:
                        continue                                           # slip is random: skip in pos-acc
                    h = (pred_pos == true_next).float()
                    hit += h.sum().item(); tot += h.numel()
                    per_act_hit[a] += h.sum().item(); per_act_n[a] += h.numel()
                    # full-state veridicality: cos of predicted vs true delta
                    with torch.no_grad():
                        true_s = model.step(s_t, true_next, g, patch)
                        coss.append(F.cosine_similarity(delta, true_s - s_t, dim=-1).mean().item())
            pos_acc = hit / max(tot, 1)
            full_cos = float(np.mean(coss)) if coss else 0.0
            per_act = {int(a): (per_act_hit[a] / per_act_n[a]).item() if per_act_n[a] > 0 else None
                       for a in range(n_actions)}
            # mean disagreement level (learnable-frontier signal)
            _, D = action_scores(fms, S_list[K // 2], P_list[K // 2], g)
            mean_disagree = D.mean().item()
        return pos_acc, full_cos, per_act, mean_disagree

    def eval_goal(fms, seg, gen_seed):
        """goal-progress under the COMMON greedy planner (measures FM reach-quality on
        equal footing, independent of the arm's explore policy)."""
        g_p = torch.Generator().manual_seed(gen_seed + 50)
        g_i = torch.Generator().manual_seed(gen_seed + 51)
        g_s = torch.Generator().manual_seed(gen_seed + 52)
        afn = greedy_plan_fn(fms)
        with torch.no_grad():
            p0, g = sample_pg(eval_batch, g_p)
            patch = sample_patch_emb(model, eval_batch, g_i)
            s = model.init_state(eval_batch, device); p = p0.clone()
            d0 = manh[p0, g]
            for t in range(K):
                s = model.step(s, p, g, patch)
                a = afn(s, p, g)
                p = env_apply(p, a, seg, g_s)
            df = manh[p, g]
        return float(((d0 - df) / d0.clamp(min=1)).mean().item())

    # ------------------------------------------------------------------
    # Online loop, per arm (shared drift schedule + data stream).
    # ------------------------------------------------------------------
    results = {}
    for ai, arm in enumerate(arm_list):
        print(f"\n=== ARM: {arm} ===")
        fms, fmopt = build_ensemble(ai)
        act_fn = make_action_fn(fms, arm, torch.Generator().manual_seed(seed + 3000 + ai))
        g_pg_a = torch.Generator().manual_seed(seed + 400)     # SHARED across arms (matched data)
        g_img_a = torch.Generator().manual_seed(seed + 401)
        g_slip_a = torch.Generator().manual_seed(seed + 402)
        tc = []                                                 # time-course
        act_hist = torch.zeros(n_actions)                       # behavior-policy action counts (per eval window)
        for it in range(n_iters):
            seg = it // drift_period
            p0, g = sample_pg(batch_size, g_pg_a)
            patch = sample_patch_emb(model, batch_size, g_img_a)
            # roll an episode with this arm's policy; collect transitions
            s = model.init_state(batch_size, device); p = p0.clone()
            trans = []
            with torch.no_grad():
                for t in range(K):
                    s = model.step(s, p, g, patch)
                    a = act_fn(s, p, g)
                    act_hist += torch.bincount(a.cpu(), minlength=n_actions).float()
                    p_next = env_apply(p, a, seg, g_slip_a)
                    s_next = model.step(s, p_next, g, patch)
                    trans.append((s.detach(), p.detach(), a.detach(), s_next.detach()))
                    p = p_next
            # train each FM on a bootstrapped subset of the episode's transitions
            s_c = torch.cat([tr[0] for tr in trans], 0)
            p_c = torch.cat([tr[1] for tr in trans], 0)
            a_c = torch.cat([tr[2] for tr in trans], 0)
            s_n = torch.cat([tr[3] for tr in trans], 0)
            target = s_n - s_c
            inp_all = s_c + efference(p_c, a_c)
            N = s_c.shape[0]
            loss = 0.0
            for fm in fms:
                idx = torch.randperm(N, device=device)[:int(bootstrap_frac * N)]
                loss = loss + F.mse_loss(fm(inp_all[idx]), target[idx])
            loss = loss / len(fms)
            fmopt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for fm in fms for p in fm.parameters()], 1.0)
            fmopt.step()

            if it % eval_interval == 0 or it == n_iters - 1:
                pos_acc, full_cos, per_act, mdis = eval_metrics(fms, seg, seed + 6000 + it)
                goal = eval_goal(fms, seg, seed + 6000 + it)
                cov = (act_hist / act_hist.sum().clamp(min=1)).tolist()
                slip_occ = cov[SLIP] if SLIP is not None else 0.0
                act_hist = torch.zeros(n_actions)
                rec = {"it": it, "seg": seg, "pos_acc": pos_acc, "full_cos": full_cos,
                       "per_act_pos_acc": per_act, "mean_disagree": mdis,
                       "goal_progress_greedyFM": goal, "action_coverage": cov,
                       "slip_occ": slip_occ, "fm_loss": loss.item()}
                tc.append(rec)
                if it % (eval_interval * 4) == 0 or it == n_iters - 1:
                    print(f"  it {it:4d} seg{seg} | pos_acc={pos_acc:.3f} full_cos={full_cos:.3f} "
                          f"goal={goal:+.3f} disagree={mdis:.4f} slip={slip_occ:.3f} loss={loss.item():.4f}")
        results[arm] = {"time_course": tc}

    # ------------------------------------------------------------------
    # Post-hoc: per-segment recovery of pos_acc (re-opening + compounding trend).
    # ------------------------------------------------------------------
    for arm in arm_list:
        tc = results[arm]["time_course"]
        seg_peak = {}
        for r in tc:
            seg_peak[r["seg"]] = max(seg_peak.get(r["seg"], 0.0), r["pos_acc"])
        results[arm]["seg_peak_pos_acc"] = {int(k): v for k, v in sorted(seg_peak.items())}

    # ------------------------------------------------------------------
    tag = f"{dataset}_g{grid}_K{K}_ke{ensemble_k}_drift{drift_period}_ew{explore_w}" + ("_quick" if quick else "")
    save_dir = f"{DATA_DIR}/a2a_forward/curiosity_reaching_control/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {"config": {"dataset": dataset, "grid": grid, "K": K, "n_actions": n_actions,
                      "slip": SLIP, "ensemble_k": ensemble_k, "drift_period": drift_period,
                      "n_iters": n_iters, "explore_w": explore_w, "eps_explore": eps_explore,
                      "arms": arm_list, "with_content": with_content, "seed": seed,
                      "probe_acc": probe_acc, "perms": [list(map(int, pm)) for pm in perms]},
           "results": results}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    # console summary: seg-peak pos_acc trend per arm
    for arm in arm_list:
        sp = results[arm]["seg_peak_pos_acc"]
        s = " ".join(f"s{k}={v:.2f}" for k, v in sp.items())
        print(f"  [{arm}] seg-peak pos_acc: {s}")
    for arm in arm_list:
        tc = results[arm]["time_course"]
        its = " ".join(f"{r['it']}" for r in tc)
        pa = " ".join(f"{r['pos_acc']:.2f}" for r in tc)
        gp = " ".join(f"{r['goal_progress_greedyFM']:+.2f}" for r in tc)
        dg = " ".join(f"{r['mean_disagree']:.3f}" for r in tc)
        print(f"\n  [{arm}] it:       {its}")
        print(f"  [{arm}] pos_acc:  {pa}")
        print(f"  [{arm}] goalprog: {gp}")
        print(f"  [{arm}] disagree: {dg}")
    return out
