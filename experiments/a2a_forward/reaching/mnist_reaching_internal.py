"""Internalized forecasting on the foveal-reaching control task.

Parent: mnist_reaching.py (the EXTERNAL-planner control result) + reaching_vit.py.
Idea:  ideas/self_model_needs_a_loop.md (internalization; map-vs-model).

The reaching positive was produced by a DECOUPLED apparatus (Run-5 discipline): the
operator G is trained model-free and FROZEN; the forward model FM is trained on
frozen transitions (pure Delta-prediction MSE); an EXTERNAL argmax planner reads a
frozen FM + frozen probe. Nothing about G's representation is ever shaped by the fact
that a forecast will be read off it, and the FM is never shaped by being USED.

This script removes that decoupling: it makes the self-forecast ENDOGENOUS and
co-trains it with the operator, then measures the representational delta vs the
decoupled operator. Two coupling mechanisms (chosen as siblings, matched everything):

  INT_PLAN   -- the forecast SELECTS actions. A differentiable one-step planner scores
                every action by rolling the FM (a_t ~ argmax_a value(s + FM(s,a))) and
                is trained end-to-end by imitation of the oracle. Direct upgrade of the
                external argmax; makes the forecast behaviorally load-bearing.
  INT_INJECT -- the forecast is a fed-back INPUT modality. Each step injects
                gate*FM(s,a_t) into the recurrent carry (the a2a closed-loop analog on a
                control task); the action is still chosen by the policy head. Tests the
                fixed-point self-consistency / "new modality" claim head-on.

Controls: MF (model-free head only == the decoupled EXT operator) and INT_PLAN_STATE
(arity-1 internal planner: co-trained but command-blind -> should stay at the floor).

All conditions share init seed and identical random-rollout coverage, so the ONLY
variable is the coupling. Readout battery (applied identically to every operator):
  1. behavior (native controller + EXTERNAL planner on a fresh frozen FM of this G)
  2. FM veridicality (co-trained FM Delta-cos vs a fresh frozen FM's)
  3. self-consistency (on-traj cos: forecast vs actual next state at the taken action)
  4. state reorganization: CKA(MF-G, cond-G), cmd_rel_spread, position-probe acc
  5. MAP-vs-MODEL (the headline): train a linear policy readout on FROZEN cond-G and
     compare its unaided control to MF's -- did the plan AMORTIZE into the weights (a
     map) or stay a separable, ablation-fragile forecast (a model)?  Plus the direct
     ablation (zero the forecast / injection off) -> dependency.

Run:
  modal run --detach a2a_forward/mnist_reaching_internal.py::internal_reaching --dataset mnist
  modal run --detach a2a_forward/mnist_reaching_internal.py::internal_reaching --dataset fashion_mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def internal_reaching(
    dataset: str = "mnist",
    with_content: bool = True,          # content-glimpse footprint (the primary regime)
    glimpse_grid: int = 3,
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    horizon: int = 14,
    min_goal_dist: int = 2,
    batch_size: int = 128,
    lr: float = 3e-4,
    train_steps: int = 3000,            # per-condition operator training
    eval_interval: int = 500,
    fwd_lr: float = 1e-3,
    fm_d_head: int = 16,                # fixed FM capacity (internalization, not the sweep)
    fm_mlp_mult: float = 2.0,
    fm_aux_lambda: float = 1.0,         # anchor co-trained FM to real Delta (veridicality)
    plan_tau: float = 1.0,              # planner softmax temperature (train)
    probe_steps: int = 1500,
    fm_steps: int = 3000,               # fresh frozen-FM training (readouts / EXT planner)
    n_eval_episodes: int = 40,
    conditions: str = "mf,int_plan,int_inject,int_plan_state",
    tag_suffix: str = "",                # append to save dir (use for smoke tests)
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
    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    mode = "content" if with_content else "blank"
    print(f"INTERNAL REACHING [{mode}] on {dataset}, {device}. grid={grid} K={K} "
          f"d={n_embd} conditions={cond_list}")

    # --- Data (content distractor) ---
    train_images = None
    if with_content:
        from datasets import load_dataset
        ds_name = {"mnist": "ylecun/mnist",
                   "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
        ds = load_dataset(ds_name)
        tr = np.stack([np.array(im) for im in ds["train"]["image"]])
        train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0

    # --- Grid / dynamics (env lives OUTSIDE the model) ---
    drow = torch.tensor([d[0] for d in ACTION_DELTAS], device=device)
    dcol = torch.tensor([d[1] for d in ACTION_DELTAS], device=device)
    ar = torch.arange(n_patches, device=device)
    rr, cc = ar // grid, ar % grid
    dist_table = ((rr[:, None] - rr[None, :]).abs()
                  + (cc[:, None] - cc[None, :]).abs()).float()

    def apply_action(p, a):
        pr, pc = p // grid, p % grid
        nr = (pr + drow[a]).clamp(0, grid - 1)
        nc = (pc + dcol[a]).clamp(0, grid - 1)
        return nr * grid + nc

    def eff_marker(p, a, reveal_vec):
        tgt = apply_action(p, a)
        onehot = F.one_hot(tgt + 1, n_positions).to(reveal_vec.dtype)
        return onehot.unsqueeze(-1) * reveal_vec

    def oracle_action(p, g):
        pr, pc = p // grid, p % grid
        gr, gc = g // grid, g % grid
        dr, dc = gr - pr, gc - pc
        a = torch.zeros_like(p)
        use_v = (dr.abs() >= dc.abs()) & (dr != 0)
        use_h = (~use_v) & (dc != 0)
        a = torch.where(use_v & (dr < 0), torch.ones_like(a) * 1, a)
        a = torch.where(use_v & (dr > 0), torch.ones_like(a) * 2, a)
        a = torch.where(use_h & (dc < 0), torch.ones_like(a) * 3, a)
        a = torch.where(use_h & (dc > 0), torch.ones_like(a) * 4, a)
        return a

    def make_gens(s):
        return (torch.Generator().manual_seed(s),        # (start, goal)
                torch.Generator().manual_seed(s + 1),    # random action
                torch.Generator().manual_seed(s + 2))    # image

    def sample_pg(B, gen):
        p = torch.randint(n_patches, (B,), generator=gen).to(device)
        g = torch.randint(n_patches, (B,), generator=gen).to(device)
        for _ in range(6):
            close = dist_table[p, g] < min_goal_dist
            if not close.any():
                break
            g = torch.where(close,
                            torch.randint(n_patches, (B,), generator=gen).to(device), g)
        return p, g

    def sample_patch_emb(model, B, gen):
        if not with_content:
            return None
        idx = torch.randint(len(train_images), (B,), generator=gen)
        return model._patch_embeds(train_images[idx].to(device))

    def new_model():
        torch.manual_seed(seed)
        return ReachingLoopedViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_embd=n_embd,
            n_head=n_head, n_loop_layers=n_loop_layers, n_steps=K,
            glimpse_grid=glimpse_grid, with_content=with_content, n_actions=n_actions,
        ).to(device)

    def roll(model, p0, g, patch_emb, action_fn, inject_fn=None):
        """Roll an episode. action_fn(s,p)->a. Optional inject_fn(s,p,a)->inject tensor
        carried into the NEXT step (efference feedback). Returns post-obs states s_{t+1},
        positions, actions, final position."""
        B = p0.shape[0]
        s = model.init_state(B, device)
        p = p0.clone()
        inject = None
        states, poss, acts = [], [], []
        for t in range(K):
            s = model.step(s, p, g, patch_emb, inject=inject)
            a = action_fn(s, p)
            states.append(s); poss.append(p.clone()); acts.append(a)
            inject = inject_fn(s, p, a) if inject_fn is not None else None
            p = apply_action(p, a)
        return states, poss, acts, p

    def value_of(s_hat, g, probe):
        logits = probe(s_hat[:, 1:, :]).squeeze(-1)      # (B, n_patches)
        probs = F.softmax(logits, dim=-1)
        dist_to_g = dist_table[:, g].t()                  # (B, n_patches)
        return -(probs * dist_to_g).sum(-1)

    # ============================================================
    # Condition trainers -- each returns a dict with the trained operator + parts
    # ============================================================
    def train_mf():
        """Model-free: policy head only. == the decoupled EXT operator."""
        model = new_model()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(train_steps):
            model.train()
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            states, poss, _, _ = roll(model, p0, g, patch, rand_fn)
            loss = sum(F.cross_entropy(model.policy_logits(states[t]),
                                       oracle_action(poss[t], g)) for t in range(K)) / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % eval_interval == 0 or step == train_steps - 1:
                print(f"  [mf] step {step:5d}: imit_loss={loss.item():.4f}")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        return {"model": model}

    def train_int_plan(arity2=True):
        """Differentiable internal planner: action logits = value(s+FM(s,a))/tau,
        imitation-trained end-to-end. Co-trains G + FM + value head (+ reveal marker)."""
        model = new_model()
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
        value_head = nn.Linear(n_embd, 1).to(device)
        reveal_vec = None
        fm_params = list(fm.parameters()) + list(value_head.parameters())
        if arity2:
            reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
            with torch.no_grad():
                reveal_vec.normal_(std=0.02)
            fm_params += [reveal_vec]
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": lr},
             {"params": fm_params, "lr": fwd_lr}], weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100)
        all_a = torch.arange(n_actions, device=device)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(train_steps):
            model.train(); fm.train()
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            states, poss, acts, _ = roll(model, p0, g, patch, rand_fn)
            loss = 0.0
            for t in range(K):
                s, p = states[t], poss[t]
                B = s.shape[0]
                vals = []
                for m in range(n_actions):
                    av = all_a[m].expand(B)
                    inp = s + eff_marker(p, av, reveal_vec) if arity2 else s
                    vals.append(value_of(s + fm(inp), g, value_head))
                logits = torch.stack(vals, -1) / plan_tau       # (B, A)
                loss = loss + F.cross_entropy(logits, oracle_action(p, g))
                if fm_aux_lambda > 0:                            # keep FM veridical
                    a = acts[t]
                    with torch.no_grad():
                        true_d = model.step(s, apply_action(p, a), g, patch) - s
                    inp_a = s + eff_marker(p, a, reveal_vec) if arity2 else s
                    loss = loss + fm_aux_lambda * F.mse_loss(fm(inp_a), true_d)
            loss = loss / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + fm_params, 1.0); opt.step()
            if step % eval_interval == 0 or step == train_steps - 1:
                print(f"  [int_plan{'' if arity2 else '_state'}] step {step:5d}: "
                      f"loss={loss.item():.4f}")
        for p in list(model.parameters()) + list(fm.parameters()) + \
                list(value_head.parameters()):
            p.requires_grad_(False)
        if reveal_vec is not None:
            reveal_vec.requires_grad_(False)
        model.eval(); fm.eval()
        return {"model": model, "fm": fm, "value_head": value_head,
                "reveal_vec": reveal_vec, "arity2": arity2}

    def train_int_inject(gate_type="scalar", forecast_readout=False):
        """Injection: s_{t+1}=G(s_t+obs+GATE(FM(s_t,a_t))); action from policy head.
        Co-trains G + policy head + FM + gate (a2a closed-loop analog, control task).

        gate_type — the "reading ladder" of how the forecast is consumed:
          'scalar' : inject = raw_scalar * fm_pred  (minimal read; the original int_inject)
          'proj'   : inject = CerebellarGate(fm_pred) = zero-init LEARNED PROJECTION
                     (the "thalamic relay" gate used in the feedforward closed-loop --
                     a learned read-then-reinject that selects/reshapes the forecast)
        forecast_readout — additionally train the forecast to be position-decodable by an
          EXPLICIT head (full read-to-interpretable-scalar), still injected & action still
          from the policy head. Isolates "is the forecast read as a distinct object"."""
        from a2a_forward.forward_model import CerebellarGate
        model = new_model()
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=fm_d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            reveal_vec.normal_(std=0.02)
        if gate_type == "proj":
            gate_mod = CerebellarGate(n_embd).to(device)
            gate_params = list(gate_mod.parameters())
        else:
            gate_scalar = torch.zeros(1, device=device, requires_grad=True)
            gate_params = [gate_scalar]
        pos_ro = nn.Linear(n_embd, 1).to(device) if forecast_readout else None
        ro_params = list(pos_ro.parameters()) if forecast_readout else []
        fm_params = list(fm.parameters()) + [reveal_vec] + gate_params + ro_params
        opt = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": lr},
             {"params": fm_params, "lr": fwd_lr}], weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 100)

        def apply_gate(fm_pred):
            return gate_mod(fm_pred) if gate_type == "proj" else gate_scalar * fm_pred

        for step in range(train_steps):
            model.train(); fm.train()
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            B = p0.shape[0]
            s = model.init_state(B, device); p = p0.clone(); inject = None
            loss = 0.0
            for t in range(K):
                s = model.step(s, p, g, patch, inject=inject)
                loss = loss + F.cross_entropy(model.policy_logits(s),
                                              oracle_action(p, g))
                a = torch.randint(n_actions, (B,), generator=g_act).to(device)
                p_next = apply_action(p, a)
                with torch.no_grad():
                    true_d = model.step(s, p_next, g, patch) - s
                fm_pred = fm(s + eff_marker(p, a, reveal_vec))
                loss = loss + fm_aux_lambda * F.mse_loss(fm_pred, true_d)
                if forecast_readout:                         # explicit read of the forecast
                    ro_logits = pos_ro(s + fm_pred)[:, 1:, :].squeeze(-1)
                    loss = loss + F.cross_entropy(ro_logits, p_next)
                inject = apply_gate(fm_pred)                 # NOT detached: learn to use
                p = p_next
            loss = loss / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + fm_params, 1.0); opt.step()
            if step % eval_interval == 0 or step == train_steps - 1:
                gn = gate_mod.injection_norm() if gate_type == "proj" \
                    else float(gate_scalar.item())
                print(f"  [int_inject:{gate_type}{'+ro' if forecast_readout else ''}] "
                      f"step {step:5d}: loss={loss.item():.4f} gate_norm={gn:.4f}")
        for pth in list(model.parameters()) + list(fm.parameters()) + gate_params + ro_params:
            pth.requires_grad_(False)
        reveal_vec.requires_grad_(False)
        model.eval(); fm.eval()
        gate_val = (gate_mod.injection_norm() if gate_type == "proj"
                    else float(gate_scalar.item()))
        gate_mod_ret = gate_mod if gate_type == "proj" else None
        return {"model": model, "fm": fm, "reveal_vec": reveal_vec,
                "gate": gate_val, "gate_type": gate_type, "gate_mod": gate_mod_ret,
                "gate_scalar": (None if gate_type == "proj" else float(gate_scalar.item()))}

    # ============================================================
    # Readout helpers (applied to a FROZEN operator)
    # ============================================================
    def train_pos_probe(model):
        probe = nn.Linear(n_embd, 1).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        g_pg, g_act, g_img = make_gens(seed + 200)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(probe_steps):
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                states, poss, _, _ = roll(model, p0, g, patch, rand_fn)
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
            states, poss, _, _ = roll(model, p0, g, patch,
                                      lambda s, p: torch.randint(
                                          n_actions, (p.shape[0],),
                                          generator=g_act).to(device))
            for t in range(K):
                accs.append((probe(states[t][:, 1:, :]).squeeze(-1).argmax(-1)
                             == poss[t]).float().mean().item())
        return probe, float(np.mean(accs))

    def train_frozen_fm(model, use_a, d_head):
        """Fresh FM trained on this frozen operator's transitions (readout / EXT)."""
        torch.manual_seed(seed + 700 + d_head + (1000 if use_a else 0))
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False).to(device)
        rvec = None
        if use_a:
            rvec = torch.zeros(n_embd, device=device, requires_grad=True)
            with torch.no_grad():
                rvec.normal_(std=0.02)
        params = list(fm.parameters()) + ([rvec] if use_a else [])
        opt = torch.optim.AdamW(params, lr=fwd_lr, weight_decay=0.01)
        g_pg, g_act, g_img = make_gens(seed + 800 + d_head)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        def batch():
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                states, poss, acts, _ = roll(model, p0, g, patch, rand_fn)
            s_cur = torch.cat(states[:-1], 0); s_nxt = torch.cat(states[1:], 0)
            return s_cur, torch.cat(poss[:-1], 0), torch.cat(acts[:-1], 0), s_nxt

        def fm_in(s, p, a):
            return s if not use_a else s + eff_marker(p, a, rvec)

        for _ in range(fm_steps):
            s, p, a, tgt = batch()
            pred = fm(fm_in(s, p, a))
            loss = F.mse_loss(pred, tgt - s)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        fm.eval()
        cos = 0.0
        with torch.no_grad():
            for _ in range(8):
                s, p, a, tgt = batch()
                cos += F.cosine_similarity(fm(fm_in(s, p, a)), tgt - s,
                                           dim=-1).mean().item()
        return fm, rvec, cos / 8

    def fm_delta_cos(model, fm, reveal_vec):
        """Held-out Delta-cos of a given (co-trained) FM on this operator."""
        g_pg, g_act, g_img = make_gens(seed + 850)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        cos = 0.0
        with torch.no_grad():
            for _ in range(8):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                states, poss, acts, _ = roll(model, p0, g, patch, rand_fn)
                s_cur = torch.cat(states[:-1], 0); s_nxt = torch.cat(states[1:], 0)
                p_cur = torch.cat(poss[:-1], 0); a_cur = torch.cat(acts[:-1], 0)
                inp = s_cur if reveal_vec is None else \
                    s_cur + eff_marker(p_cur, a_cur, reveal_vec)
                cos += F.cosine_similarity(fm(inp), s_nxt - s_cur, dim=-1).mean().item()
        return cos / 8

    def phase1c(model, fm_state, fm_eff, rvec_eff):
        g_pg, g_act, g_img = make_gens(seed + 860)
        all_a = torch.arange(n_actions, device=device)
        spread, cc_eff, ot_state, ot_eff = [], [], [], []

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        with torch.no_grad():
            for _ in range(8):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                states, poss, acts, _ = roll(model, p0, g, patch, rand_fn)
                for t in range(K):
                    s_t, p_t = states[t], poss[t]
                    B = s_t.shape[0]
                    nexts = torch.stack([
                        model.step(s_t, apply_action(p_t, all_a[m].expand(B)), g, patch)
                        for m in range(n_actions)], 0)
                    upd = nexts - s_t.unsqueeze(0)
                    mean_upd = upd.mean(0)
                    spread.append(((upd - mean_upd).norm(dim=-1).mean()
                                   / (mean_upd.norm(dim=-1).mean() + 1e-8)).item())
                    eff_all = torch.stack([
                        fm_eff(s_t + eff_marker(p_t, all_a[m].expand(B), rvec_eff))
                        for m in range(n_actions)], 0)
                    cc_eff.append(F.cosine_similarity(
                        eff_all - eff_all.mean(0, keepdim=True),
                        upd - mean_upd, dim=-1).mean().item())
                    a_t = acts[t]
                    true_upd = model.step(s_t, apply_action(p_t, a_t), g, patch) - s_t
                    ot_state.append(F.cosine_similarity(
                        fm_state(s_t), true_upd, dim=-1).mean().item())
                    ot_eff.append(F.cosine_similarity(
                        fm_eff(s_t + eff_marker(p_t, a_t, rvec_eff)),
                        true_upd, dim=-1).mean().item())
        return {"cmd_rel_spread": float(np.mean(spread)),
                "cmd_cond_cos_eff": float(np.mean(cc_eff)),
                "ontraj_cos_state": float(np.mean(ot_state)),
                "ontraj_cos_eff": float(np.mean(ot_eff))}

    def subspace_delta_cos(model, fm, reveal_vec, wdir):
        """DIAGNOSTIC 1 (collusion vs selectivity). Split the co-trained FM's Delta
        prediction into the value-relevant direction `wdir` (a FRESH INDEPENDENT probe's
        weight -- the direction the value readout actually uses) vs its orthogonal
        complement. Selectivity (FM predicts what matters, drops the rest) => on-dir cos
        high, off-dir cos ~0, even when full-vector cos is low. Collusion (FM output is a
        value-shaped signal untethered from real state) => on-dir cos ALSO low."""
        what = wdir / (wdir.norm() + 1e-8)
        g_pg, g_act, g_img = make_gens(seed + 870)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        sp_all, st_all, off = [], [], []
        with torch.no_grad():
            for _ in range(8):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                states, poss, acts, _ = roll(model, p0, g, patch, rand_fn)
                s_cur = torch.cat(states[:-1], 0); s_nxt = torch.cat(states[1:], 0)
                p_cur = torch.cat(poss[:-1], 0); a_cur = torch.cat(acts[:-1], 0)
                inp = s_cur if reveal_vec is None else \
                    s_cur + eff_marker(p_cur, a_cur, reveal_vec)
                dpred = fm(inp); dtrue = s_nxt - s_cur
                sp = (dpred * what).sum(-1); st = (dtrue * what).sum(-1)  # scalar along w
                sp_all.append(sp.flatten()); st_all.append(st.flatten())
                dpred_perp = dpred - sp.unsqueeze(-1) * what
                dtrue_perp = dtrue - st.unsqueeze(-1) * what
                off.append(F.cosine_similarity(dpred_perp, dtrue_perp, dim=-1).mean().item())
        sp = torch.cat(sp_all); st = torch.cat(st_all)
        on_cos = F.cosine_similarity(sp.unsqueeze(0), st.unsqueeze(0), dim=-1).item()
        return {"on_dir_cos": on_cos, "off_dir_cos": float(np.mean(off))}

    def cka_states(model):
        """Stacked CLS states over a FIXED (p,g,random-action) eval set (matched
        across models because the generators are reseeded identically)."""
        g_pg, g_act, g_img = make_gens(seed + 500)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        chunks = []
        with torch.no_grad():
            for _ in range(4):
                p0, g = sample_pg(batch_size, g_pg)
                patch = sample_patch_emb(model, batch_size, g_img)
                states, _, _, _ = roll(model, p0, g, patch, rand_fn)
                chunks.append(torch.stack(states, 0)[:, :, 0, :].reshape(-1, n_embd))
        return torch.cat(chunks, 0)

    def linear_cka(X, Y):
        X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
        xy = (X.t() @ Y).norm() ** 2
        xx = (X.t() @ X).norm(); yy = (Y.t() @ Y).norm()
        return (xy / (xx * yy + 1e-12)).item()

    g_holder = [None]   # current-batch goal, exposed to planner/value action_fns

    def eval_ctrl_g(model, action_fn, inject_fn=None):
        g_pg, g_act, g_img = make_gens(seed + 900)
        succ = dist = norm_prog = 0.0
        for _ in range(n_eval_episodes):
            p0, g = sample_pg(batch_size, g_pg)
            g_holder[0] = g
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                _, _, _, p_final = roll(model, p0, g, patch, action_fn, inject_fn)
            succ += (p_final == g).float().mean().item()
            d0 = dist_table[p0, g]; df = dist_table[p_final, g]
            dist += df.mean().item()
            norm_prog += ((d0 - df) / d0.clamp(min=1)).mean().item()
        n = n_eval_episodes
        return {"success": succ / n, "final_dist": dist / n, "norm_progress": norm_prog / n}

    def ext_planner_fn(model, fm, rvec, probe, g_holder, zero_fm=False):
        def fn(s, p):
            B = s.shape[0]; vals = []
            for a in range(n_actions):
                av = torch.full((B,), a, device=device)
                if rvec is None:
                    fmp = fm(s)
                else:
                    fmp = fm(s + eff_marker(p, av, rvec))
                if zero_fm:
                    fmp = torch.zeros_like(fmp)
                vals.append(value_of(s + fmp, g_holder[0], probe))
            V = torch.stack(vals, 0) + 1e-6 * torch.randn(n_actions, B, device=device)
            return V.argmax(0)
        return fn

    def train_policy_readout(model):
        """Linear policy readout on the FROZEN operator (the amortization probe):
        can the oracle action be decoded from the state WITHOUT the FM channel?"""
        head = nn.Linear(n_embd, n_actions).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        g_pg, g_act, g_img = make_gens(seed + 600)

        def rand_fn(s, p):
            return torch.randint(n_actions, (p.shape[0],), generator=g_act).to(device)

        for step in range(probe_steps):
            p0, g = sample_pg(batch_size, g_pg)
            patch = sample_patch_emb(model, batch_size, g_img)
            with torch.no_grad():
                states, poss, _, _ = roll(model, p0, g, patch, rand_fn)
            t = torch.randint(K, (1,)).item()
            logits = head(model.ln_f(states[t])[:, 0])
            loss = F.cross_entropy(logits, oracle_action(poss[t], g))
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        return head

    # ============================================================
    # Run conditions
    # ============================================================
    trained = {}
    for c in cond_list:
        print(f"\n=== training condition: {c} ===")
        if c == "mf":
            trained[c] = train_mf()
        elif c == "int_plan":
            trained[c] = train_int_plan(arity2=True)
        elif c == "int_plan_state":
            trained[c] = train_int_plan(arity2=False)
        elif c == "int_inject":
            trained[c] = train_int_inject(gate_type="scalar")
        elif c == "int_inject_proj":
            trained[c] = train_int_inject(gate_type="proj")
        elif c == "int_inject_readout":
            trained[c] = train_int_inject(gate_type="scalar", forecast_readout=True)
        else:
            print(f"  (unknown condition {c}, skipping)")

    mf_states = cka_states(trained["mf"]["model"]) if "mf" in trained else None

    def native_fn(c, info, model):
        """Action function for the condition's OWN controller (+ inject_fn if any)."""
        if c == "mf":
            return (lambda s, p: model.policy_logits(s).argmax(-1)), None
        if c.startswith("int_inject"):
            fm, rvec = info["fm"], info["reveal_vec"]
            gmod, gscalar = info.get("gate_mod"), info.get("gate_scalar")

            def inj(s, p, a):
                fp = fm(s + eff_marker(p, a, rvec))
                return gmod(fp) if gmod is not None else gscalar * fp
            return (lambda s, p: model.policy_logits(s).argmax(-1)), inj
        # int_plan / int_plan_state -> internal planner argmax
        fm, vh, rvec = info["fm"], info["value_head"], info["reveal_vec"]

        def plan(s, p):
            B = s.shape[0]; vals = []
            for a in range(n_actions):
                av = torch.full((B,), a, device=device)
                inp = s + eff_marker(p, av, rvec) if rvec is not None else s
                vals.append(value_of(s + fm(inp), g_holder[0], vh))
            V = torch.stack(vals, 0) + 1e-6 * torch.randn(n_actions, B, device=device)
            return V.argmax(0)
        return plan, None

    results = {}
    for c, info in trained.items():
        print(f"\n=== readouts: {c} ===")
        model = info["model"]
        probe, probe_acc = train_pos_probe(model)
        # fresh frozen FMs on THIS operator (decoupled EXT planner + veridicality base)
        fm_s, _, cos_s = train_frozen_fm(model, use_a=False, d_head=fm_d_head)
        fm_e, rv_e, cos_e = train_frozen_fm(model, use_a=True, d_head=fm_d_head)
        p1c = phase1c(model, fm_s, fm_e, rv_e)

        # native controller
        nfn, inj = native_fn(c, info, model)
        native = eval_ctrl_g(model, nfn, inject_fn=inj)

        # EXT decoupled planner on fresh frozen FM_eff + fresh probe
        ext = eval_ctrl_g(model, ext_planner_fn(model, fm_e, rv_e, probe, g_holder))

        # amortization: linear policy readout on frozen operator (no FM channel)
        ro_head = train_policy_readout(model)
        amort = eval_ctrl_g(model, lambda s, p: ro_head(model.ln_f(s)[:, 0]).argmax(-1))

        cka = linear_cka(mf_states, cka_states(model)) if mf_states is not None else None

        entry = {
            "probe_acc": probe_acc,
            "frozen_fm_cos_state": cos_s, "frozen_fm_cos_eff": cos_e,
            "phase1c": p1c,
            "native_progress": native,
            "ext_planner_progress": ext,
            "amortization_readout_progress": amort,
            "cka_vs_mf": cka,
        }
        if "fm" in info and info.get("value_head") is not None:
            entry["cotrained_fm_cos"] = fm_delta_cos(
                model, info["fm"], info.get("reveal_vec"))
        # DIAGNOSTICS (collusion vs task-relevant selectivity), for any co-trained FM:
        if info.get("fm") is not None:
            fmc, rvc = info["fm"], info.get("reveal_vec")
            wdir = probe.weight.detach().squeeze(0)          # fresh independent direction
            entry["subspace_cos"] = subspace_delta_cos(model, fmc, rvc, wdir)
            # DIAGNOSTIC 2: co-trained FM read by a FRESH INDEPENDENT probe (external
            # argmax). If ~native -> FM is veridical for an un-colluded consumer
            # (selectivity); if ~floor -> it only means something to its own value head.
            entry["cotrained_fm_freshprobe_progress"] = eval_ctrl_g(
                model, ext_planner_fn(model, fmc, rvc, probe, g_holder))
        if c.startswith("int_inject"):
            entry["gate"] = info["gate"]
            entry["gate_type"] = info.get("gate_type", "scalar")
            entry["cotrained_fm_cos"] = fm_delta_cos(model, info["fm"], info["reveal_vec"])

            def pol_noinj(s, p):
                return model.policy_logits(s).argmax(-1)
            entry["ablate_injection_progress"] = eval_ctrl_g(model, pol_noinj)
        if c in ("int_plan", "int_plan_state"):
            fm, vh, rvec = info["fm"], info["value_head"], info["reveal_vec"]

            def plan_zero(s, p):
                B = s.shape[0]; vals = []
                for a in range(n_actions):
                    av = torch.full((B,), a, device=device)
                    inp = s + eff_marker(p, av, rvec) if rvec is not None else s
                    fmp = torch.zeros_like(fm(inp))
                    vals.append(value_of(s + fmp, g_holder[0], vh))
                V = torch.stack(vals, 0) + 1e-6 * torch.randn(n_actions, B, device=device)
                return V.argmax(0)
            entry["ablate_forecast_progress"] = eval_ctrl_g(model, plan_zero)

        results[c] = entry
        print(f"  probe_acc={probe_acc:.3f} cka_vs_mf={cka} "
              f"native_prog={native['norm_progress']:+.3f} "
              f"ext_prog={ext['norm_progress']:+.3f} "
              f"amort_prog={amort['norm_progress']:+.3f}")
        print(f"  frozen FM cos state/eff={cos_s:.3f}/{cos_e:.3f}  "
              f"p1c: spread={p1c['cmd_rel_spread']:.3f} "
              f"cc_eff={p1c['cmd_cond_cos_eff']:.3f} "
              f"ontraj s/e={p1c['ontraj_cos_state']:.3f}/{p1c['ontraj_cos_eff']:.3f}")
        if "subspace_cos" in entry:
            sc = entry["subspace_cos"]
            print(f"  DIAG: subspace cos on/off={sc['on_dir_cos']:.3f}/"
                  f"{sc['off_dir_cos']:.3f}  cotrained-FM+freshprobe progress="
                  f"{entry['cotrained_fm_freshprobe_progress']['norm_progress']:+.3f} "
                  f"(native={native['norm_progress']:+.3f})")

    # --- Save ---
    tag = f"{dataset if with_content else 'nav'}_{mode}_g{grid}_K{K}_{n_head}H{n_embd}D{tag_suffix}"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_reaching_internal/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "dataset": dataset, "mode": mode, "with_content": with_content,
        "config": {"grid": grid, "K": K, "n_embd": n_embd, "fm_d_head": fm_d_head,
                   "fm_aux_lambda": fm_aux_lambda, "train_steps": train_steps,
                   "conditions": cond_list, "seed": seed},
        "results": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")
    return out
