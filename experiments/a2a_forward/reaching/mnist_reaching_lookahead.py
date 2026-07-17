"""Multi-step lookahead: does the transferable one-step self-forecast COMPOSE into a
runnable simulator?  (the live next step from REACHING_INTERNAL_README.md)

Parent: mnist_reaching_internal.py (the one-step endogenous self-forecast that came out
a transferable, task-relevant self-model) + reaching_vit.py.
Idea:  ideas/self_model_needs_a_loop.md ("run a rough forward pass on yourself, off to
the side" == exactly an N-step FM rollout; the central open question is whether the
one-step fixed-point self-consistency is SUFFICIENT for usable near-manifold
counterfactuals or only the seed).

Plain reaching is greedily solvable, so it can't test composition. This script adds
three LOOKAHEAD-FORCING geometries where a MYOPIC (one-step) planner provably fails and
only rolling the forecast forward multiple steps succeeds:

  obstacles -- a wall with a gap between start and goal. The Manhattan-straight action
               walks into the wall; a >=2-step detour is required.
  maze      -- a serpentine corridor (alternating horizontal walls). Deep detours ->
               large lookahead depth.
  occluded  -- NO walls, but the goal is HIDDEN until the fovea visits a cue patch that
               reveals it. Tests composition through a LATENT reveal event (the FM must
               simulate that visiting the cue changes the observation), not geometry.

The discriminator is the MPC planner: an ENDOGENOUS runnable simulator that rolls the
forward model N steps IN STATE SPACE, decoding its own believed fovea position off each
simulated state (probe argmax) to place the next action's efference marker -- NO env
access during the rollout. If composition holds (outcome A), rolling the one-step FM
routes around walls / through the reveal and planning progress rises with horizon; if it
degrades (outcome B) the one-step self-model is not composable; if it composes in a
value-consistent-but-not-full-state-veridical space (outcome C) -> the division-of-labor
reframe. We read this two ways, which are the SAME underlying question here:
  (a) BEHAVIORAL   -- MPC progress vs lookahead depth N (does deeper planning win?)
  (b) VERIDICALITY -- composed-position accuracy + composed Delta-cos of FM^n vs the true
                      n-step state, vs n (does the simulator still know where it is?)

Kinematics note: unlike textbook MPC, the planner is NOT given the env's transition
function. It decodes position from the state via a probe and predicts state transitions
via the FM -- so a good rollout REQUIRES the self-forecast to compose. That is the whole
point (faithful to the decoupled EXT-planner discipline of the parent arc).

Run (one detached job per geometry, in parallel):
  modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry obstacles
  modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry maze
  modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry occluded
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def lookahead_reaching(
    geometry: str = "obstacles",        # obstacles | maze | occluded
    dataset: str = "mnist",
    with_content: bool = True,          # content-glimpse distractor (primary regime, as parent)
    glimpse_grid: int = 3,
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    horizon: int = 16,                  # episode length (>= longest detour)
    batch_size: int = 128,
    lr: float = 3e-4,
    train_steps: int = 3000,
    eval_interval: int = 500,
    fwd_lr: float = 1e-3,
    fm_d_head: int = 16,
    fm_mlp_mult: float = 2.0,
    fm_aux_lambda: float = 1.0,
    plan_tau: float = 1.0,
    probe_steps: int = 2000,
    fm_steps: int = 3000,
    n_eval_episodes: int = 24,
    eval_batch: int = 64,               # smaller batch for the (heavier) MPC eval
    mpc_samples: int = 64,              # random-shooting sequences per MPC decision
    depths: str = "1,2,3,4,6",          # lookahead horizon sweep
    compose_ns: str = "1,2,3,4,6,8",    # composed-veridicality horizon sweep
    conditions: str = "mf,int_plan",
    fm_compose_horizons: str = "",      # extra fresh FMs trained w/ multi-step consistency
    fm_compose_lambda: float = 1.0,     # weight on the composition-consistency loss
    env_mpc_only: bool = False,         # skip all training; only the perfect-sim MPC control
    tag_suffix: str = "",
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from collections import deque
    from a2a_forward.reaching.reaching_vit import ReachingLoopedViT, ACTION_DELTAS
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = 28 // patch_size
    n_patches = grid ** 2
    n_positions = n_patches + 1
    n_actions = len(ACTION_DELTAS)
    K = horizon
    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    depth_list = [int(d) for d in depths.split(",") if d.strip()]
    compose_n_list = [int(d) for d in compose_ns.split(",") if d.strip()]
    compose_h_list = [int(h) for h in fm_compose_horizons.split(",") if h.strip()]
    mode = "content" if with_content else "blank"
    print(f"LOOKAHEAD REACHING [{geometry}/{mode}] on {dataset}, {device}. "
          f"grid={grid} K={K} d={n_embd} conditions={cond_list} depths={depth_list}")

    # ============================================================
    # Geometry: walls + geodesic (BFS, wall-aware) + Manhattan (wall-blind) tables
    # ============================================================
    drow = torch.tensor([d[0] for d in ACTION_DELTAS], device=device)
    dcol = torch.tensor([d[1] for d in ACTION_DELTAS], device=device)

    def rc(idx):
        return idx // grid, idx % grid

    # Fixed (seeded/deterministic) wall layout per geometry.
    walls = set()
    cue_patch = None
    if geometry == "obstacles":
        # vertical wall on column grid//2, gap only at the bottom two rows.
        col = grid // 2
        for r in range(grid - 2):
            walls.add(r * grid + col)
    elif geometry == "maze":
        # serpentine: horizontal walls on odd rows, gap alternating end.
        for r in range(1, grid, 2):
            gap = (grid - 1) if ((r // 2) % 2 == 0) else 0
            for c in range(grid):
                if c != gap:
                    walls.add(r * grid + c)
    elif geometry == "occluded":
        cue_patch = (grid // 2) * grid + (grid // 2)  # center cue reveals the goal
    else:
        raise ValueError(f"unknown geometry {geometry}")
    free = [i for i in range(n_patches) if i not in walls]
    wall_mask = torch.zeros(n_patches, dtype=torch.bool, device=device)
    for w in walls:
        wall_mask[w] = True
    print(f"  walls={len(walls)} free={len(free)} cue={cue_patch}")

    # Manhattan (wall-blind) distance table -- the MISLEADING value heuristic.
    ar = torch.arange(n_patches, device=device)
    rr, ccol = ar // grid, ar % grid
    manh = ((rr[:, None] - rr[None, :]).abs()
            + (ccol[:, None] - ccol[None, :]).abs()).float()  # (P, P)

    # Geodesic (wall-aware BFS) distance table -- oracle + honest progress metric.
    INF = 10 ** 6
    geo = np.full((n_patches, n_patches), INF, dtype=np.float64)
    adj = [[] for _ in range(n_patches)]
    for i in range(n_patches):
        if i in walls:
            continue
        r, c = i // grid, i % grid
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid and 0 <= nc < grid:
                j = nr * grid + nc
                if j not in walls:
                    adj[i].append(j)
    for src in free:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        for v, dd in d.items():
            geo[src, v] = dd
    geo_t = torch.tensor(geo, device=device)  # (P, P), INF where unreachable

    def apply_action(p, a):
        """Wall-aware kinematics (env, outside the model): into a wall -> stay."""
        pr, pc = p // grid, p % grid
        nr = (pr + drow[a]).clamp(0, grid - 1)
        nc = (pc + dcol[a]).clamp(0, grid - 1)
        tgt = nr * grid + nc
        blocked = wall_mask[tgt]
        return torch.where(blocked, p, tgt)

    def oracle_action(p, vt):
        """BFS-optimal first action toward the (visible) target vt: pick the action whose
        wall-aware successor minimizes geodesic distance to vt. Manhattan-greedy is NOT
        this in a maze -- that is what makes lookahead necessary."""
        B = p.shape[0]
        succ = torch.stack([apply_action(p, torch.full((B,), a, device=device))
                            for a in range(n_actions)], 0)   # (A, B)
        gd = geo_t[succ, vt.unsqueeze(0).expand(n_actions, B)]  # (A, B)
        # tiny stay penalty so the oracle keeps moving off ties
        gd = gd + torch.tensor([[0.01], [0], [0], [0], [0]], device=device)
        return gd.argmin(0)

    def greedy_manh_action(p, vt):
        """Myopic Manhattan floor: descend wall-BLIND Manhattan distance. Gets trapped."""
        B = p.shape[0]
        succ = torch.stack([apply_action(p, torch.full((B,), a, device=device))
                            for a in range(n_actions)], 0)
        md = manh[succ, vt.unsqueeze(0).expand(n_actions, B)]
        md = md + torch.tensor([[0.01], [0], [0], [0], [0]], device=device)
        return md.argmin(0)

    # --- Data (content distractor) ---
    train_images = None
    if with_content:
        from datasets import load_dataset
        ds_name = {"mnist": "ylecun/mnist",
                   "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
        ds = load_dataset(ds_name)
        tr = np.stack([np.array(im) for im in ds["train"]["image"]])
        train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0

    def make_gens(s):
        return (torch.Generator().manual_seed(s),
                torch.Generator().manual_seed(s + 1),
                torch.Generator().manual_seed(s + 2))

    free_t = torch.tensor(free, device=device)

    def sample_pg(B, gen):
        """Sample start p0 and goal g among FREE, mutually-reachable patches with a
        minimum geodesic separation. For obstacles/maze also bias to opposite sides of
        the wall so the detour actually bites."""
        def draw(nn_):
            idx = torch.randint(len(free), (nn_,), generator=gen)
            return free_t[idx.to(device)]
        p = draw(B)
        g = draw(B)
        for _ in range(12):
            d = geo_t[p, g]
            bad = (d < 3) | (d >= INF)
            if geometry in ("obstacles", "maze"):
                # want start/goal to straddle the barrier: geodesic >> Manhattan
                bad = bad | (d < manh[p, g] + 1)
            if not bad.any():
                break
            g = torch.where(bad, draw(B), g)
        return p, g

    def sample_patch_emb(model, B, gen):
        if not with_content:
            return None
        idx = torch.randint(len(train_images), (B,), generator=gen)
        return model._patch_embeds(train_images[idx].to(device))

    def eff_marker(p, a, reveal_vec):
        tgt = apply_action(p, a)
        onehot = F.one_hot(tgt + 1, n_positions).to(reveal_vec.dtype)
        return onehot.unsqueeze(-1) * reveal_vec

    # ---- Occlusion: visible target = cue until the fovea has visited it, then goal ----
    def new_model():
        torch.manual_seed(seed)
        return ReachingLoopedViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_embd=n_embd,
            n_head=n_head, n_loop_layers=n_loop_layers, n_steps=K,
            glimpse_grid=glimpse_grid, with_content=with_content, n_actions=n_actions,
        ).to(device)

    def roll(model, p0, g, patch_emb, action_fn, inject_fn=None):
        """Roll an episode with wall-aware kinematics and (for occluded) latent reveal.
        action_fn(s, p, vt) -> a. Returns states s_{t+1}, positions, visible-targets,
        actions, final position."""
        B = p0.shape[0]
        s = model.init_state(B, device)
        p = p0.clone()
        visited_cue = torch.zeros(B, dtype=torch.bool, device=device)
        inject = None
        states, poss, vts, acts = [], [], [], []
        for t in range(K):
            if geometry == "occluded":
                visited_cue = visited_cue | (p == cue_patch)
                vt = torch.where(visited_cue, g, torch.full_like(g, cue_patch))
            else:
                vt = g
            s = model.step(s, p, vt, patch_emb, inject=inject)
            a = action_fn(s, p, vt)
            states.append(s); poss.append(p.clone()); vts.append(vt); acts.append(a)
            inject = inject_fn(s, p, a) if inject_fn is not None else None
            p = apply_action(p, a)
        return states, poss, vts, acts, p

    def value_of(s_hat, vt, probe):
        """Value = -expected (wall-blind Manhattan) distance to the visible target, read
        through the position probe. Manhattan (not geodesic) so it is a MISLEADING local
        heuristic near walls -- the planner must look ahead to route around."""
        logits = probe(s_hat[:, 1:, :]).squeeze(-1)      # (N, n_patches)
        probs = F.softmax(logits, dim=-1)
        dist = manh[:, vt].t()                            # (N, n_patches)
        return -(probs * dist).sum(-1)

    # ============================================================
    # Condition trainers (one-step planner, matched to the parent int_plan recipe)
    # ============================================================
    def train_mf():
        model = new_model()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100)

        def rand_fn(s, p, vt):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(train_steps):
            model.train()
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            states, poss, vts, _, _ = roll(model, p0, g, patch, rand_fn)
            loss = sum(F.cross_entropy(model.policy_logits(states[t]),
                                       oracle_action(poss[t], vts[t])) for t in range(K)) / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % eval_interval == 0 or step == train_steps - 1:
                print(f"  [mf] step {step:5d}: imit_loss={loss.item():.4f}")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        return {"model": model, "fm": None, "value_head": None, "reveal_vec": None}

    def train_int_plan():
        """One-step differentiable planner (arity-2), the parent's transferable
        self-model, trained on the lookahead-forcing geometry. Note: a one-step planner
        with a Manhattan value CANNOT imitate the BFS oracle at wall states (it can only
        descend Manhattan) -- so it is myopic BY CONSTRUCTION; composition is tested at
        eval by rolling this FM N steps."""
        model = new_model()
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
        value_head = nn.Linear(n_embd, 1).to(device)
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            reveal_vec.normal_(std=0.02)
        fm_params = list(fm.parameters()) + list(value_head.parameters()) + [reveal_vec]
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": lr},
             {"params": fm_params, "lr": fwd_lr}], weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100)
        all_a = torch.arange(n_actions, device=device)

        def rand_fn(s, p, vt):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(train_steps):
            model.train(); fm.train()
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            states, poss, vts, acts, _ = roll(model, p0, g, patch, rand_fn)
            loss = 0.0
            for t in range(K):
                s, p, vt = states[t], poss[t], vts[t]
                B = s.shape[0]
                vals = []
                for m in range(n_actions):
                    av = all_a[m].expand(B)
                    inp = s + eff_marker(p, av, reveal_vec)
                    vals.append(value_of(s + fm(inp), vt, value_head))
                logits = torch.stack(vals, -1) / plan_tau
                loss = loss + F.cross_entropy(logits, oracle_action(p, vt))
                if fm_aux_lambda > 0:
                    a = acts[t]
                    with torch.no_grad():
                        # the FM's target folds in the NEXT obs (incl. any reveal), so a
                        # composed rollout simulates the full loop w/o env access.
                        if geometry == "occluded":
                            vc_next = (p == cue_patch)
                            vt_next = torch.where(vc_next | (vt == g), g,
                                                  torch.full_like(g, cue_patch))
                        else:
                            vt_next = vt
                        true_d = model.step(s, apply_action(p, a), vt_next, patch) - s
                    inp_a = s + eff_marker(p, a, reveal_vec)
                    loss = loss + fm_aux_lambda * F.mse_loss(fm(inp_a), true_d)
            loss = loss / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + fm_params, 1.0); opt.step()
            if step % eval_interval == 0 or step == train_steps - 1:
                print(f"  [int_plan] step {step:5d}: loss={loss.item():.4f}")
        for p in list(model.parameters()) + list(fm.parameters()) + \
                list(value_head.parameters()):
            p.requires_grad_(False)
        reveal_vec.requires_grad_(False)
        model.eval(); fm.eval()
        return {"model": model, "fm": fm, "value_head": value_head,
                "reveal_vec": reveal_vec}

    # ============================================================
    # Readout helpers on a FROZEN operator
    # ============================================================
    def train_pos_probe(model):
        probe = nn.Linear(n_embd, 1).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        g_pg, g_act, g_img = make_gens(seed + 200)

        def rand_fn(s, p, vt):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(probe_steps):
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                states, poss, _, _, _ = roll(model, p0, g, patch, rand_fn)
            t = torch.randint(K, (1,)).item()
            logits = probe(states[t][:, 1:, :]).squeeze(-1)
            loss = F.cross_entropy(logits, poss[t])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        g_pg, g_act, g_img = make_gens(seed + 999)
        accs = []
        with torch.no_grad():
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            states, poss, _, _, _ = roll(
                model, p0, g, patch,
                lambda s, p, vt: torch.randint(
                    n_actions, (p.shape[0],), generator=g_act).to(device))
            for t in range(K):
                accs.append((probe(states[t][:, 1:, :]).squeeze(-1).argmax(-1)
                             == poss[t]).float().mean().item())
        return probe, float(np.mean(accs))

    def train_frozen_fm(model, d_head, compose_horizon=0, compose_lambda=0.0):
        """Fresh arity-2 FM trained on this frozen operator's transitions.
        compose_horizon>0 adds a MULTI-STEP CONSISTENCY loss: free-run the FM's own state
        prediction n steps (teacher-forcing the actions/positions of a real random rollout)
        and penalize the composed state vs the true n-step state. This directly pressures
        the FM to COMPOSE (the A-vs-C test) rather than only be one-step veridical, and it
        is trained on RANDOM rollouts -- the same off-policy distribution the MPC planner
        queries -- so it targets composability where the planner actually reads it."""
        tagn = 700 + d_head + 13 * compose_horizon
        torch.manual_seed(seed + tagn)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
        rvec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            rvec.normal_(std=0.02)
        params = list(fm.parameters()) + [rvec]
        opt = torch.optim.AdamW(params, lr=fwd_lr, weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100 + tagn)

        def rand_fn(s, p, vt):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for _ in range(fm_steps):
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                states, poss, _, acts, _ = roll(model, p0, g, patch, rand_fn)
            s_cur = torch.cat(states[:-1], 0); s_nxt = torch.cat(states[1:], 0)
            p_cur = torch.cat(poss[:-1], 0); a_cur = torch.cat(acts[:-1], 0)
            loss = F.mse_loss(fm(s_cur + eff_marker(p_cur, a_cur, rvec)), s_nxt - s_cur)
            if compose_horizon > 0 and compose_lambda > 0:
                maxn = min(compose_horizon, K - 1)
                t0 = torch.randint(0, K - maxn, (1,)).item()
                S = states[t0]
                closs = 0.0
                for n in range(1, maxn + 1):
                    a, p = acts[t0 + n - 1], poss[t0 + n - 1]
                    S = S + fm(S + eff_marker(p, a, rvec))       # free-run state
                    closs = closs + F.mse_loss(S, states[t0 + n])  # vs true n-step state
                loss = loss + compose_lambda * closs / maxn
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        fm.eval()
        cos = 0.0
        with torch.no_grad():
            for _ in range(8):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                states, poss, _, acts, _ = roll(model, p0, g, patch, rand_fn)
                s_cur = torch.cat(states[:-1], 0); s_nxt = torch.cat(states[1:], 0)
                p_cur = torch.cat(poss[:-1], 0); a_cur = torch.cat(acts[:-1], 0)
                cos += F.cosine_similarity(fm(s_cur + eff_marker(p_cur, a_cur, rvec)),
                                           s_nxt - s_cur, dim=-1).mean().item()
        return fm, rvec, cos / 8

    # ============================================================
    # The MPC planner: ENDOGENOUS N-step FM rollout (random shooting)
    # ============================================================
    def mpc_action_fn(fm, rvec, probe, depth, gen):
        """Return an action_fn(s, p, vt) that plans by rolling `fm` `depth` steps in state
        space for `mpc_samples` random action sequences, DECODING its own believed
        position each step (no env access), and executing the first action of the best
        sequence. depth==1 reduces to the one-step greedy planner."""
        def fn(s, p, vt):
            B = s.shape[0]
            ns = mpc_samples
            # sample action sequences (ns, depth), shared across the batch
            seqs = torch.randint(n_actions, (ns, depth), generator=gen).to(device)
            S = s.unsqueeze(0).expand(ns, -1, -1, -1).reshape(ns * B, n_positions, n_embd)
            vt_e = vt.unsqueeze(0).expand(ns, B).reshape(-1)          # (ns*B,)
            # believed position decoded from the (real) start state
            p_hat = probe(S[:, 1:, :]).squeeze(-1).argmax(-1)         # (ns*B,)
            for d in range(depth):
                a = seqs[:, d].unsqueeze(1).expand(ns, B).reshape(-1)  # (ns*B,)
                S = S + fm(S + eff_marker(p_hat, a, rvec))
                p_hat = probe(S[:, 1:, :]).squeeze(-1).argmax(-1)
            val = value_of(S, vt_e, probe).reshape(ns, B)              # terminal value
            val = val + 1e-6 * torch.randn_like(val)
            best = val.argmax(0)                                       # (B,)
            return seqs[best, 0]
        return fn

    def env_mpc_action_fn(depth, gen):
        """PERFECT-SIMULATOR control: same random-shooting planner + same Manhattan value,
        but rolling the TRUE kinematics (exact positions, no FM, no probe) instead of the
        forecast. Isolates the planner/value: if this also floors on the maze, the maze is
        planner-bound (a misleading value + random shooting), not simulator-bound."""
        def fn(s, p, vt):
            B = p.shape[0]; ns = mpc_samples
            seqs = torch.randint(n_actions, (ns, depth), generator=gen).to(device)
            P = p.unsqueeze(0).expand(ns, B).reshape(-1)          # exact positions
            vt_e = vt.unsqueeze(0).expand(ns, B).reshape(-1)
            for d in range(depth):
                a = seqs[:, d].unsqueeze(1).expand(ns, B).reshape(-1)
                P = apply_action(P, a)
            val = (-manh[P, vt_e]).reshape(ns, B)                 # exact Manhattan terminal
            val = val + 1e-6 * torch.randn_like(val)
            return seqs[val.argmax(0), 0]
        return fn

    g_holder = [None]

    def eval_ctrl(model, action_fn, inject_fn=None, batch=None, episodes=None):
        b = batch or eval_batch
        ep = episodes or n_eval_episodes
        g_pg, g_act, g_img = make_gens(seed + 900)
        succ = norm_prog = 0.0
        for _ in range(ep):
            p0, g = sample_pg(b, g_pg)
            g_holder[0] = g
            patch = sample_patch_emb(model, b, g_img)
            with torch.no_grad():
                _, _, _, _, p_final = roll(model, p0, g, patch, action_fn, inject_fn)
            succ += (p_final == g).float().mean().item()
            d0 = geo_t[p0, g]; df = geo_t[p_final, g]
            norm_prog += ((d0 - df) / d0.clamp(min=1)).mean().item()
        return {"success": succ / ep, "norm_progress": norm_prog / ep}

    def compose_veridicality(model, fm, rvec, probe, ns_list):
        """Runnable-simulator readout. Along real oracle trajectories, take s_t and roll
        the FM forward n steps using the ACTUAL actions taken, decoding believed position
        each step (endogenous). Report, vs n: (i) composed-position accuracy of the
        simulated state (does it still know where it is?), (ii) composed Delta-cos of the
        n-step FM prediction vs the true n-step state delta."""
        g_pg, g_act, g_img = make_gens(seed + 950)
        maxn = max(ns_list)
        pos_hits = {n: [] for n in ns_list}
        dcos = {n: [] for n in ns_list}
        with torch.no_grad():
            for _ in range(8):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                # roll a real OPTIMAL trajectory to get true states/actions/positions
                states, poss, vts, acts, _ = roll(
                    model, p0, g, patch, lambda s, p, vt: oracle_action(p, vt))
                for t0 in range(0, K - maxn):
                    S = states[t0].clone()
                    p_hat = probe(S[:, 1:, :]).squeeze(-1).argmax(-1)
                    for n in range(1, maxn + 1):
                        a = acts[t0 + n - 1]                      # the ACTUAL action taken
                        S = S + fm(S + eff_marker(p_hat, a, rvec))
                        p_hat = probe(S[:, 1:, :]).squeeze(-1).argmax(-1)
                        if n in ns_list:
                            true_pos = poss[t0 + n]
                            pos_hits[n].append((p_hat == true_pos).float().mean().item())
                            true_delta = states[t0 + n] - states[t0]
                            pred_delta = S - states[t0]
                            dcos[n].append(F.cosine_similarity(
                                pred_delta, true_delta, dim=-1).mean().item())
        return ({n: float(np.mean(pos_hits[n])) for n in ns_list},
                {n: float(np.mean(dcos[n])) for n in ns_list})

    # ============================================================
    # Fast path: perfect-simulator MPC control only (no training needed -- the env-MPC
    # action ignores the operator state, so a throwaway model suffices to drive eval_ctrl).
    # ============================================================
    if env_mpc_only:
        model0 = new_model()
        gen_env = torch.Generator().manual_seed(seed + 1234)
        oracle_prog = eval_ctrl(model0, lambda s, p, vt: oracle_action(p, vt))
        greedy_prog = eval_ctrl(model0, lambda s, p, vt: greedy_manh_action(p, vt))
        env_sweep = {}
        for d in depth_list:
            gen_env.manual_seed(seed + 1234)
            env_sweep[d] = eval_ctrl(model0, env_mpc_action_fn(d, gen_env))
        print(f"[{geometry}] ENV-MPC (perfect sim) control | "
              f"oracle={oracle_prog['norm_progress']:+.3f} "
              f"greedy={greedy_prog['norm_progress']:+.3f}")
        s = "  ".join(f"d{d}={env_sweep[d]['norm_progress']:+.3f}" for d in depth_list)
        print(f"  ENV-MPC depth sweep: {s}")
        tag = f"{geometry}_{dataset if with_content else 'nav'}_g{grid}_K{K}_{n_head}H{n_embd}D{tag_suffix}"
        save_dir = f"{DATA_DIR}/a2a_forward/mnist_reaching_lookahead/{tag}"
        os.makedirs(save_dir, exist_ok=True)
        out = {"geometry": geometry, "env_mpc_only": True,
               "oracle_progress": oracle_prog, "greedy_manh_progress": greedy_prog,
               "env_mpc_depth_sweep": env_sweep,
               "config": {"grid": grid, "K": K, "depths": depth_list,
                          "mpc_samples": mpc_samples, "seed": seed}}
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(out, f, indent=2, cls=NumpyEncoder)
        volume.commit()
        print(f"\nSaved to {save_dir}")
        return out

    # ============================================================
    # Run
    # ============================================================
    trained = {}
    for c in cond_list:
        print(f"\n=== training condition: {c} ===")
        trained[c] = train_mf() if c == "mf" else train_int_plan()

    results = {}
    ref_done = {}
    for c, info in trained.items():
        print(f"\n=== readouts: {c} ===")
        model = info["model"]
        probe, probe_acc = train_pos_probe(model)
        fm_fresh, rv_fresh, cos_fresh = train_frozen_fm(model, fm_d_head)
        gen_mpc = torch.Generator().manual_seed(seed + 1234)

        # reference behaviors (env-exact, no FM): ceiling + myopic floor
        oracle_prog = eval_ctrl(model, lambda s, p, vt: oracle_action(p, vt))
        greedy_prog = eval_ctrl(model, lambda s, p, vt: greedy_manh_action(p, vt))

        # native controller
        if c == "mf":
            native = eval_ctrl(model, lambda s, p, vt: model.policy_logits(s).argmax(-1))
        else:
            fm, vh, rvec = info["fm"], info["value_head"], info["reveal_vec"]

            def plan1(s, p, vt):
                B = s.shape[0]; vals = []
                for a in range(n_actions):
                    av = torch.full((B,), a, device=device)
                    vals.append(value_of(s + fm(s + eff_marker(p, av, rvec)), vt, vh))
                V = torch.stack(vals, 0) + 1e-6 * torch.randn(n_actions, B, device=device)
                return V.argmax(0)
            native = eval_ctrl(model, plan1)

        # Per-FM readout: MPC depth sweep (a) + composed-veridicality sweep (b). gen_mpc
        # is reseeded before each FM so every FM is compared on IDENTICAL shooting samples.
        def fm_readout(fm_r, rv_r):
            gen_mpc.manual_seed(seed + 1234)
            sweep = {d: eval_ctrl(model, mpc_action_fn(fm_r, rv_r, probe, d, gen_mpc))
                     for d in depth_list}
            pos, dcos = compose_veridicality(model, fm_r, rv_r, probe, compose_n_list)
            return sweep, {"pos_acc": pos, "delta_cos": dcos}

        # FRESH frozen FM (one-step Delta-MSE) -- decoupled, tests operator plannability.
        sw_fresh, cv_fresh = fm_readout(fm_fresh, rv_fresh)
        depth_sweep = {"fresh_fm": sw_fresh}
        compose = {"fresh_fm": cv_fresh}
        fm_cos = {"fresh_fm": cos_fresh}
        # CO-TRAINED FM (for int_plan) -- composition of the actual self-model.
        if info["fm"] is not None:
            sw_ct, cv_ct = fm_readout(info["fm"], info["reveal_vec"])
            depth_sweep["cotrained_fm"] = sw_ct
            compose["cotrained_fm"] = cv_ct
        # MULTI-STEP-TRAINED fresh FMs (the A-vs-C sharpening): same operator, FM trained
        # with explicit multi-step consistency -> is DEEP composition achievable at all?
        for h in compose_h_list:
            fm_ms, rv_ms, cos_ms = train_frozen_fm(
                model, fm_d_head, compose_horizon=h, compose_lambda=fm_compose_lambda)
            sw_ms, cv_ms = fm_readout(fm_ms, rv_ms)
            key = f"multistep_fm_h{h}"
            depth_sweep[key] = sw_ms
            compose[key] = cv_ms
            fm_cos[key] = cos_ms

        entry = {
            "probe_acc": probe_acc,
            "frozen_fm_cos": fm_cos,
            "oracle_progress": oracle_prog,
            "greedy_manh_progress": greedy_prog,
            "native_progress": native,
            "mpc_depth_sweep": depth_sweep,
            "compose_veridicality": compose,
        }
        results[c] = entry
        print(f"  probe_acc={probe_acc:.3f} fm_cos="
              + " ".join(f"{k}={v:.3f}" for k, v in fm_cos.items()))
        print(f"  oracle={oracle_prog['norm_progress']:+.3f} "
              f"greedy_manh={greedy_prog['norm_progress']:+.3f} "
              f"native={native['norm_progress']:+.3f}")
        for tag, sw in depth_sweep.items():
            if sw:
                s = "  ".join(f"d{d}={sw[d]['norm_progress']:+.3f}" for d in depth_list)
                print(f"  MPC[{tag}] depth sweep: {s}")
        for tag, cv in compose.items():
            s = "  ".join(f"n{n}={cv['pos_acc'][n]:.2f}" for n in compose_n_list)
            print(f"  compose[{tag}] pos-acc: {s}")

    # --- Save ---
    tag = f"{geometry}_{dataset if with_content else 'nav'}_g{grid}_K{K}_{n_head}H{n_embd}D{tag_suffix}"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_reaching_lookahead/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "geometry": geometry, "dataset": dataset, "mode": mode,
        "config": {"grid": grid, "K": K, "n_embd": n_embd, "fm_d_head": fm_d_head,
                   "train_steps": train_steps, "conditions": cond_list,
                   "depths": depth_list, "compose_ns": compose_n_list,
                   "mpc_samples": mpc_samples, "n_walls": len(walls), "seed": seed},
        "results": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return out
