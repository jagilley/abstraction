"""Control-regime port of the "missing u" test: is an efference-copy-conditioned
forward self-model (arity-2) CAUSALLY USABLE for control, where an arity-1 one is
structurally not -- at any capacity?

Why this task (see ACTIVE_VISION_README.md, "Where this points"). The active-vision
causal arm nulled: the command is content-free, so on a *perception* task its
forecast is answer-irrelevant ("the average effect of looking here" is identical
across images). The diagnosis: the command matters for the TRAJECTORY, not the
ANSWER. This experiment makes the trajectory the answer -- a foveal-REACHING control
task where the agent steers its fovea to a goal patch, so predicting the consequence
of your own action is the whole point (the cerebellar regime).

Three phases, all on a shared frozen controller (Run-5 scaling-sweep discipline):

  Phase 0 (build the controller). Train ReachingLoopedViT's policy head by imitation
    of a shortest-path oracle over RANDOM-action rollouts (broad state coverage that
    matches phase 1). This yields (a) a competent model-free controller = the CEILING,
    and (b) a recurrent state that encodes fovea position + goal.

  Phase 1 (observational -- replicate the arity gap, now answer-relevant). Freeze the
    controller. Train a linear fovea-position probe (the planner's value), then train
    FM_state (arity-1, f(s)) and FM_eff (arity-2, f(s, a); action delivered as a
    learned per-action embedding placed spatially at the current fovea) across a
    capacity sweep to predict the loop update Delta_t = s_{t+1}-s_t. Metrics:
    cmd_rel_spread (how much Delta depends on the action), command-conditional cos
    (FM_eff recovers it; FM_state = 0 structurally), arity-beats-capacity.

  Phase 2 (causal / behavioral -- the new result). Freeze the FMs. A one-step
    model-based PLANNER picks each action by rolling the forward self-model:
        a_t = argmax_a  value( s_t + FM(s_t, a) ),   value(s) = -E_pos[ dist(pos, g) ]
    with `value` read from the frozen position probe. Evaluate control performance
    (success rate / final distance / steps-to-goal) for:
        planner{FM_eff}       (arity-2)         -> should approach the ceiling
        planner{FM_state}     (arity-1, SWEPT)  -> pinned at the random floor, always
        planner{FM_eff-shuffled-action}         -> the inert-command control (~floor)
        model-free policy head (ceiling)  |  oracle (upper bound)  |  random (floor)
    FM_state's value is constant in a (no action input) -> the planner is a random
    walk regardless of capacity. That is the impossibility result.

Run:
  modal run --detach a2a_forward/mnist_reaching.py::reaching --with-content False   # clean nav
  modal run --detach a2a_forward/mnist_reaching.py::reaching --with-content True    # distractor
  modal run --detach a2a_forward/mnist_reaching.py::reaching --dataset fashion_mnist --with-content True
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def reaching(
    dataset: str = "mnist",              # only used when with_content=True
    with_content: bool = False,          # blank-canvas nav (False) vs glimpse distractor
    glimpse_grid: int = 3,
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    horizon: int = 14,                   # episode length (>= max Manhattan dist = 12)
    min_goal_dist: int = 2,              # goal at least this far from start
    batch_size: int = 128,
    lr: float = 3e-4,
    policy_steps: int = 3000,            # phase 0 imitation steps
    eval_interval: int = 250,
    # forward-model / probe training (phase 1, on the frozen controller)
    fwd_lr: float = 1e-3,
    probe_steps: int = 1500,
    fm_steps: int = 3000,
    fm_d_head_sweep: str = "4,8,16,32,64",
    fm_mlp_mult: float = 2.0,
    match_d_head: int = 16,              # FM size used in the planner eval
    # phase 2 eval
    n_eval_episodes: int = 40,           # batches of `batch_size` episodes
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.reaching.reaching_vit import ReachingLoopedViT, ACTION_DELTAS
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = 28 // patch_size
    n_patches = grid ** 2
    n_positions = n_patches + 1
    n_actions = len(ACTION_DELTAS)
    K = horizon
    sweep = [int(x) for x in fm_d_head_sweep.split(",")]
    mode = "content" if with_content else "blank"
    print(f"REACHING [{mode}] on {dataset}, {device}. grid={grid} K={K} "
          f"d={n_embd} sweep={sweep}")

    # --- Data (only needed for the content distractor) ---
    train_images = test_images = None
    if with_content:
        from datasets import load_dataset
        ds_name = {"mnist": "ylecun/mnist",
                   "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
        ds = load_dataset(ds_name)
        tr = np.stack([np.array(im) for im in ds["train"]["image"]])
        train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
        te = np.stack([np.array(im) for im in ds["test"]["image"]])
        test_images = torch.from_numpy(te).float().unsqueeze(1) / 255.0

    # --- Reproducible RNG ---
    g_pg = torch.Generator().manual_seed(seed)          # (start, goal) sampling
    g_act = torch.Generator().manual_seed(seed + 1)     # random-action policy
    g_img = torch.Generator().manual_seed(seed + 2)     # image sampling
    g_eval = torch.Generator().manual_seed(seed + 3)    # eval

    # --- Grid / dynamics helpers (env lives OUTSIDE the model) ---
    drow = torch.tensor([d[0] for d in ACTION_DELTAS], device=device)
    dcol = torch.tensor([d[1] for d in ACTION_DELTAS], device=device)
    # manhattan distance table between every pair of patch indices (n_patches^2)
    ar = torch.arange(n_patches, device=device)
    rr, cc = ar // grid, ar % grid
    dist_table = ((rr[:, None] - rr[None, :]).abs()
                  + (cc[:, None] - cc[None, :]).abs()).float()  # (n_patches, n_patches)

    def apply_action(p, a):
        pr, pc = p // grid, p % grid
        nr = (pr + drow[a]).clamp(0, grid - 1)
        nc = (pc + dcol[a]).clamp(0, grid - 1)
        return nr * grid + nc

    def eff_marker(p, a, reveal_vec):
        """Spatial efference copy for the forward model: a learned marker placed at
        the position the fovea is COMMANDED to next (p_{t+1} = clip(p + delta(a))) --
        the location about to be attended, exactly the glimpse arm's design. The
        commanded endpoint IS the motor command; the FM's job is to predict the
        activation consequence of arriving there. The action identity is implicit in
        *where* the marker sits, so an arity-1 FM (no marker) has no such signal."""
        tgt = apply_action(p, a)
        onehot = F.one_hot(tgt + 1, n_positions).to(reveal_vec.dtype)  # (N, n_pos)
        return onehot.unsqueeze(-1) * reveal_vec                       # (N, n_pos, E)

    def oracle_action(p, g):
        pr, pc = p // grid, p % grid
        gr, gc = g // grid, g % grid
        dr, dc = gr - pr, gc - pc
        a = torch.zeros_like(p)  # stay
        use_v = (dr.abs() >= dc.abs()) & (dr != 0)
        use_h = (~use_v) & (dc != 0)
        a = torch.where(use_v & (dr < 0), torch.ones_like(a) * 1, a)
        a = torch.where(use_v & (dr > 0), torch.ones_like(a) * 2, a)
        a = torch.where(use_h & (dc < 0), torch.ones_like(a) * 3, a)
        a = torch.where(use_h & (dc > 0), torch.ones_like(a) * 4, a)
        return a

    def sample_pg(B, gen):
        p = torch.randint(n_patches, (B,), generator=gen).to(device)
        g = torch.randint(n_patches, (B,), generator=gen).to(device)
        # resample goals that are too close (a few passes suffice)
        for _ in range(6):
            close = dist_table[p, g] < min_goal_dist
            if not close.any():
                break
            g = torch.where(close,
                            torch.randint(n_patches, (B,), generator=gen).to(device), g)
        return p, g

    def sample_patch_emb(B, gen):
        if not with_content:
            return None
        idx = torch.randint(len(train_images), (B,), generator=gen)
        return model._patch_embeds(train_images[idx].to(device))

    # --- Main model ---
    torch.manual_seed(seed)
    model = ReachingLoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_embd=n_embd,
        n_head=n_head, n_loop_layers=n_loop_layers, n_steps=K,
        glimpse_grid=glimpse_grid, with_content=with_content, n_actions=n_actions,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def roll(p0, g, patch_emb, action_fn, K_steps=None):
        """Roll an episode. action_fn(s, p) -> action (B,). Returns per-step
        post-observation states, positions, actions, and the final position.
        states[t] = s_{t+1} (post-obs at positions[t]); action[t] moves to the next."""
        Ke = K_steps or K
        B = p0.shape[0]
        s = model.init_state(B, device)
        p = p0.clone()
        states, poss, acts = [], [], []
        for t in range(Ke):
            s = model.step(s, p, g, patch_emb)
            a = action_fn(s, p)
            states.append(s)
            poss.append(p.clone())
            acts.append(a)
            p = apply_action(p, a)
        return states, poss, acts, p

    def random_action_fn(s, p):
        return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

    # =============================================
    # Phase 0: imitation-train the policy head (the ceiling controller)
    # =============================================
    print("\n--- phase 0: imitation (policy head) ---")
    hist = {"imit_loss": [], "imit_acc": []}
    for step in range(policy_steps):
        model.train()
        p0, g = sample_pg(batch_size, g_pg)
        patch_emb = sample_patch_emb(batch_size, g_img)
        states, poss, acts, _ = roll(p0, g, patch_emb, random_action_fn)
        loss = 0.0
        correct = tot = 0
        for t in range(K):
            logits = model.policy_logits(states[t])
            tgt = oracle_action(poss[t], g)
            loss = loss + F.cross_entropy(logits, tgt)
            correct += (logits.argmax(-1) == tgt).float().sum().item()
            tot += tgt.numel()
        loss = loss / K
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == policy_steps - 1:
            hist["imit_loss"].append((step, loss.item()))
            hist["imit_acc"].append((step, correct / tot))
            print(f"  step {step:5d}: imit_loss={loss.item():.4f} "
                  f"oracle_acc={correct / tot:.4f}")

    for pth in model.parameters():
        pth.requires_grad_(False)
    model.eval()

    # =============================================
    # Phase 1a: frozen fovea-position probe (planner value)
    # linear score per patch position -> softmax over positions = predicted fovea.
    # =============================================
    print("\n--- phase 1a: position probe ---")
    pos_probe = nn.Linear(n_embd, 1).to(device)
    opt_probe = torch.optim.AdamW(pos_probe.parameters(), lr=1e-3, weight_decay=0.0)
    g_pp = torch.Generator().manual_seed(seed + 10)
    g_ppi = torch.Generator().manual_seed(seed + 11)
    for step in range(probe_steps):
        pos_probe.train()
        p0, g = sample_pg(batch_size, g_pp)
        patch_emb = sample_patch_emb(batch_size, g_ppi)
        with torch.no_grad():
            states, poss, _, _ = roll(p0, g, patch_emb, random_action_fn)
        t = torch.randint(K, (1,), generator=g_pp).item()
        logits = pos_probe(states[t][:, 1:, :]).squeeze(-1)  # (B, n_patches)
        loss = F.cross_entropy(logits, poss[t])
        opt_probe.zero_grad(); loss.backward(); opt_probe.step()
    pos_probe.eval()
    # probe accuracy on held-out states
    with torch.no_grad():
        p0, g = sample_pg(batch_size, g_eval)
        patch_emb = sample_patch_emb(batch_size, g_eval)
        states, poss, _, _ = roll(p0, g, patch_emb, random_action_fn)
        probe_acc = np.mean([
            (pos_probe(states[t][:, 1:, :]).squeeze(-1).argmax(-1) == poss[t])
            .float().mean().item() for t in range(K)])
    print(f"  position-probe acc = {probe_acc:.4f}")

    def value_of(s_hat, g):
        """value(s) = -E_pos[dist(pos, g)] under the probe's position distribution."""
        logits = pos_probe(s_hat[:, 1:, :]).squeeze(-1)   # (B, n_patches)
        probs = F.softmax(logits, dim=-1)
        dist_to_g = dist_table[:, g].t()                  # (B, n_patches)
        return -(probs * dist_to_g).sum(-1)               # (B,)

    # =============================================
    # Phase 1b: FM capacity sweep (state vs eff) predicting Delta_t = s_{t+1}-s_t
    # =============================================
    print("\n--- phase 1b: forward-model sweep (state vs eff) ---")

    def make_transitions(action_fn, gen_pg, gen_img):
        """Roll random-action episodes; return consecutive transitions
        (s_{t+1}, p_t, a_t, s_{t+2}) flattened over t and batch."""
        p0, g = sample_pg(batch_size, gen_pg)
        patch_emb = sample_patch_emb(batch_size, gen_img)
        with torch.no_grad():
            states, poss, acts, _ = roll(p0, g, patch_emb, action_fn)
        s_cur = torch.cat(states[:-1], 0)                 # s_{t+1}
        s_nxt = torch.cat(states[1:], 0)                  # s_{t+2}
        p_cur = torch.cat(poss[:-1], 0)
        a_cur = torch.cat(acts[:-1], 0)
        return s_cur, p_cur, a_cur, s_nxt

    def train_fm(use_a, d_head):
        """Train a forward model to predict Delta = s_next - s_cur. use_a => arity-2:
        a single learned marker (reveal_vec) placed at the commanded next fovea
        p_{t+1} (the efference copy). Arity-1 (use_a=False) sees s only."""
        torch.manual_seed(seed + 300 + d_head + (1000 if use_a else 0))
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        reveal_vec = None
        if use_a:
            reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
            with torch.no_grad():
                reveal_vec.normal_(std=0.02)
        params = list(fm.parameters()) + ([reveal_vec] if use_a else [])
        opt_fm = torch.optim.AdamW(params, lr=fwd_lr, weight_decay=0.01)
        g_fm = torch.Generator().manual_seed(seed + 400 + d_head)
        g_fmi = torch.Generator().manual_seed(seed + 450 + d_head)

        def fm_input(s, p, a):
            return s if not use_a else s + eff_marker(p, a, reveal_vec)

        for _ in range(fm_steps):
            fm.train()
            s, p, a, tgt = make_transitions(random_action_fn, g_fm, g_fmi)
            pred = fm(fm_input(s, p, a))
            floss = F.mse_loss(pred, tgt - s)
            opt_fm.zero_grad(); floss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt_fm.step()
        # held-out cos + mse on Delta
        fm.eval()
        cos = mse = 0.0
        nb = 8
        with torch.no_grad():
            for _ in range(nb):
                s, p, a, tgt = make_transitions(random_action_fn, g_eval, g_eval)
                pred = fm(fm_input(s, p, a))
                target = tgt - s
                cos += F.cosine_similarity(pred, target, dim=-1).mean().item()
                mse += F.mse_loss(pred, target).item()
        return fm, reveal_vec, {"cos": cos / nb, "mse": mse / nb}

    sweep_results = {"state": {}, "eff": {}}
    matched = {}
    for d_head in sweep:
        fm_s, _, rs = train_fm(use_a=False, d_head=d_head)
        fm_e, ae_e, re = train_fm(use_a=True, d_head=d_head)
        sweep_results["state"][str(d_head)] = rs
        sweep_results["eff"][str(d_head)] = re
        print(f"  d_head={d_head:3d}: FM_state cos={rs['cos']:.4f} | "
              f"FM_eff cos={re['cos']:.4f}  (Δ={re['cos'] - rs['cos']:+.4f})")
        if d_head == match_d_head:
            matched = {"state": (fm_s, None), "eff": (fm_e, ae_e)}
        # keep swept models for the phase-2 capacity curves (eff rises with capacity,
        # state stays at the random floor -- arity beats resolution, behaviorally)
        sweep_results["state"][str(d_head)]["_model"] = fm_s
        sweep_results["eff"][str(d_head)]["_model"] = fm_e
        sweep_results["eff"][str(d_head)]["_rvec"] = ae_e
    if not matched:
        fm_s, _, _ = train_fm(use_a=False, d_head=match_d_head)
        fm_e, ae_e, _ = train_fm(use_a=True, d_head=match_d_head)
        matched = {"state": (fm_s, None), "eff": (fm_e, ae_e)}

    fm_state_matched = matched["state"][0]
    fm_eff, reveal_vec_eff = matched["eff"]

    def fm_eff_input(s, p, a):
        return s + eff_marker(p, a, reveal_vec_eff)

    # --- observational arity metrics (control regime) ---
    print("\n--- phase 1c: observational arity metrics ---")
    cmd_rel_spread, cc_cos_eff = [], []
    ontraj_cos_state, ontraj_cos_eff = [], []
    all_actions = torch.arange(n_actions, device=device)
    with torch.no_grad():
        for _ in range(12):
            p0, g = sample_pg(batch_size, g_eval)
            patch_emb = sample_patch_emb(batch_size, g_eval)
            states, poss, acts, _ = roll(p0, g, patch_emb, random_action_fn)
            for t in range(K):
                s_t = states[t]
                p_t = poss[t]
                B = s_t.shape[0]
                # true consequence of every action a from (s_t, p_t)
                nexts = torch.stack([
                    model.step(s_t, apply_action(p_t, all_actions[m].expand(B)), g,
                               patch_emb) for m in range(n_actions)], 0)  # (A,B,P,E)
                updates = nexts - s_t.unsqueeze(0)
                mean_upd = updates.mean(0)
                spread = (updates - mean_upd).norm(dim=-1).mean()
                denom = mean_upd.norm(dim=-1).mean() + 1e-8
                cmd_rel_spread.append((spread / denom).item())
                # command-conditional cos: FM_eff recovers Delta(a)-mean_a Delta
                eff_all = torch.stack([
                    fm_eff(fm_eff_input(s_t, p_t, all_actions[m].expand(B)))
                    for m in range(n_actions)], 0)
                true_cc = updates - mean_upd
                eff_cc = eff_all - eff_all.mean(0, keepdim=True)
                cc_cos_eff.append(
                    F.cosine_similarity(eff_cc, true_cc, dim=-1).mean().item())
                # on-traj cos on the ACTUAL action taken
                a_t = acts[t]
                true_upd = model.step(s_t, apply_action(p_t, a_t), g, patch_emb) - s_t
                p_state = fm_state_matched(s_t)
                p_eff = fm_eff(fm_eff_input(s_t, p_t, a_t))
                ontraj_cos_state.append(
                    F.cosine_similarity(p_state, true_upd, dim=-1).mean().item())
                ontraj_cos_eff.append(
                    F.cosine_similarity(p_eff, true_upd, dim=-1).mean().item())
    cf = {
        "cmd_rel_spread": float(np.mean(cmd_rel_spread)),
        "cmd_cond_cos_eff": float(np.mean(cc_cos_eff)),
        "cmd_cond_cos_state": 0.0,
        "ontraj_cos_state": float(np.mean(ontraj_cos_state)),
        "ontraj_cos_eff": float(np.mean(ontraj_cos_eff)),
    }
    big_state_cos = sweep_results["state"][str(max(sweep))]["cos"]
    small_eff_cos = sweep_results["eff"][str(min(sweep))]["cos"]
    arity_beats_capacity = small_eff_cos > big_state_cos
    print(f"  cmd_rel_spread={cf['cmd_rel_spread']:.4f}  "
          f"cmd-cond cos eff={cf['cmd_cond_cos_eff']:.4f} (state=0)")
    print(f"  on-traj cos state/eff = {cf['ontraj_cos_state']:.4f} / "
          f"{cf['ontraj_cos_eff']:.4f}")
    print(f"  ARITY vs RESOLUTION: smallest FM_eff (d={min(sweep)}) "
          f"{small_eff_cos:.4f} vs largest FM_state (d={max(sweep)}) "
          f"{big_state_cos:.4f} -> {arity_beats_capacity}")

    # =============================================
    # Phase 2: model-based planner (the causal / behavioral test)
    # =============================================
    print("\n--- phase 2: planner control performance ---")

    def planner_fn(fm, reveal_vec, shuffle=False):
        """One-step MB planner: pick argmax_a value(s + FM(s, a)). If reveal_vec is
        None (FM_state), value is constant in a -> random walk (tie-break noise).
        shuffle=True places the efference marker at the target of a DIFFERENT action
        (inert-command control): the forecast no longer matches the scored action."""
        perm = torch.tensor([1, 2, 3, 4, 0], device=device)  # fixed derangement

        def fn(s, p):
            B = s.shape[0]
            vals = []
            for a in range(n_actions):
                av = torch.full((B,), a, device=device)
                if reveal_vec is None:
                    s_hat = s + fm(s)
                else:
                    a_eff = perm[av] if shuffle else av
                    s_hat = s + fm(s + eff_marker(p, a_eff, reveal_vec))
                vals.append(value_of(s_hat, g_cur[0]))
            V = torch.stack(vals, 0)                       # (A, B)
            V = V + 1e-6 * torch.randn_like(V)             # tie-break -> random floor
            return V.argmax(0)
        return fn

    # closures over the current goal for value_of (set per eval batch)
    g_cur = [None]

    def policy_fn(s, p):
        return model.policy_logits(s).argmax(-1)

    def oracle_fn(s, p):
        return oracle_action(p, g_cur[0])

    def random_fn(s, p):
        return torch.randint(n_actions, (p.shape[0],), generator=g_eval).to(device)

    def eval_controller(action_fn):
        # Episode-level control metrics. success = reached the goal patch; final_dist =
        # mean Manhattan distance to goal at the end; steps = steps-to-first-hit (K if
        # never). Net progress (final_dist) is the clean discriminator -- per-step
        # oracle-agreement is confounded by path multiplicity and near-goal stalling.
        succ = dist = steps = norm_prog = 0.0
        n = 0
        for _ in range(n_eval_episodes):
            p0, g = sample_pg(batch_size, g_eval)
            g_cur[0] = g
            patch_emb = sample_patch_emb(batch_size, g_eval)
            with torch.no_grad():
                _, poss, _, p_final = roll(p0, g, patch_emb, action_fn)
            succ += (p_final == g).float().mean().item()
            d0 = dist_table[p0, g]
            df = dist_table[p_final, g]
            dist += df.mean().item()
            # normalized progress toward goal: 1 = reached, 0 = no progress, <0 = worse
            norm_prog += ((d0 - df) / d0.clamp(min=1)).mean().item()
            reached = torch.full((batch_size,), K, device=device)
            for t in range(K):
                hit = (poss[t] == g) & (reached == K)
                reached = torch.where(hit, torch.full_like(reached, t), reached)
            steps += reached.float().mean().item()
            n += 1
        return {"success": succ / n, "final_dist": dist / n, "steps": steps / n,
                "norm_progress": norm_prog / n}

    controllers = {}
    controllers["oracle"] = eval_controller(oracle_fn)
    controllers["random"] = eval_controller(random_fn)
    controllers["model_free"] = eval_controller(policy_fn)
    controllers["planner_eff"] = eval_controller(
        planner_fn(fm_eff, reveal_vec_eff))
    controllers["planner_eff_shuffled"] = eval_controller(
        planner_fn(fm_eff, reveal_vec_eff, shuffle=True))
    # capacity curves: planner success/progress as FM capacity grows, for BOTH
    # arities. Prediction: FM_eff rises with capacity (better model -> better control);
    # FM_state stays pinned at the random floor -- resolution cannot buy arity.
    planner_state_by_cap, planner_eff_by_cap = {}, {}
    for d_head in sweep:
        fm_s = sweep_results["state"][str(d_head)]["_model"]
        fm_e = sweep_results["eff"][str(d_head)]["_model"]
        rvec = sweep_results["eff"][str(d_head)]["_rvec"]
        planner_state_by_cap[str(d_head)] = eval_controller(planner_fn(fm_s, None))
        planner_eff_by_cap[str(d_head)] = eval_controller(planner_fn(fm_e, rvec))
    mkey = str(match_d_head if match_d_head in sweep else sweep[len(sweep) // 2])
    controllers["planner_state_matched"] = planner_state_by_cap[mkey]

    print(f"  {'controller':24s} {'success':>8s} {'fin_dist':>8s} {'steps':>7s} "
          f"{'progress':>8s}")
    for name, r in controllers.items():
        print(f"  {name:24s} {r['success']:8.3f} {r['final_dist']:8.3f} "
              f"{r['steps']:7.2f} {r['norm_progress']:8.3f}")
    print(f"  planner by capacity (success | progress):")
    print(f"    FM_eff  : " + " ".join(
        f"d{d}={planner_eff_by_cap[str(d)]['success']:.3f}/"
        f"{planner_eff_by_cap[str(d)]['norm_progress']:+.2f}" for d in sweep))
    print(f"    FM_state: " + " ".join(
        f"d{d}={planner_state_by_cap[str(d)]['success']:.3f}/"
        f"{planner_state_by_cap[str(d)]['norm_progress']:+.2f}" for d in sweep))

    # strip non-serialisable model handles from sweep_results before saving
    for cap in sweep_results["state"].values():
        cap.pop("_model", None)
    for cap in sweep_results["eff"].values():
        cap.pop("_model", None)
        cap.pop("_rvec", None)

    # =============================================
    # Save
    # =============================================
    tag = f"{dataset if with_content else 'nav'}_{mode}_g{grid}_K{K}_{n_head}H{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_reaching/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    result = {
        "dataset": dataset, "mode": mode, "with_content": with_content,
        "config": {
            "grid": grid, "K": K, "min_goal_dist": min_goal_dist, "n_embd": n_embd,
            "n_head": n_head, "n_loop_layers": n_loop_layers, "patch_size": patch_size,
            "policy_steps": policy_steps, "fm_steps": fm_steps,
            "fm_d_head_sweep": sweep, "match_d_head": match_d_head, "seed": seed,
        },
        "history": hist,
        "probe_acc": probe_acc,
        "final_oracle_acc": hist["imit_acc"][-1][1],
        "fm_sweep": sweep_results,
        "counterfactual": cf,
        "arity_beats_capacity": bool(arity_beats_capacity),
        "small_eff_cos": small_eff_cos, "big_state_cos": big_state_cos,
        "controllers": controllers,
        "planner_state_by_capacity": planner_state_by_cap,
        "planner_eff_by_capacity": planner_eff_by_cap,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'=' * 66}\n  REACHING [{mode}] {dataset} SUMMARY\n{'=' * 66}")
    print(f"  imitation oracle_acc = {result['final_oracle_acc']:.4f}  "
          f"probe_acc = {probe_acc:.4f}")
    print(f"  cmd_rel_spread = {cf['cmd_rel_spread']:.4f}  "
          f"cmd-cond cos eff = {cf['cmd_cond_cos_eff']:.4f} (state=0)")
    print(f"  arity_beats_capacity = {arity_beats_capacity}")
    best = str(max(sweep))
    print(f"  CONTROL success | progress:")
    print(f"    model_free={controllers['model_free']['success']:.3f}/"
          f"{controllers['model_free']['norm_progress']:+.2f}  "
          f"planner_eff(d{best})={planner_eff_by_cap[best]['success']:.3f}/"
          f"{planner_eff_by_cap[best]['norm_progress']:+.2f}  "
          f"planner_state(d{best})={planner_state_by_cap[best]['success']:.3f}/"
          f"{planner_state_by_cap[best]['norm_progress']:+.2f}  "
          f"random={controllers['random']['success']:.3f}/"
          f"{controllers['random']['norm_progress']:+.2f}")
    print(f"\nSaved to {save_dir}")
    return result
