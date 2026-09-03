"""jacobian_teacher/core.py — the substrate fork, the Jacobian instruments, and the teachers.

Node: `experiments/mjc/jacobian_teacher/` (see [`SPEC.md`](SPEC.md)). Parent:
[`../README.md`](../README.md). Reading: Garibbo, Filipe, Aitchison & Costa 2026,
*Unifying error and reward action learning: a cerebello-basal ganglia theory* — eqs. 1–2.

WHAT THIS MODULE IS. `../ballistic/arm/arm_readapt.py` (Cut 4c-arm) is written as one Modal
function whose helpers are all closures over `cfg`. Every one of them is needed here, and needed
by BOTH entrypoints of this node (`phase_a.py`, `jacobian_teacher.py`), so they are hoisted onto
`ArmSubstrate` verbatim in arithmetic — same expressions, same seed offsets, same normalization,
same CEM. `phase_a.py::gate_fork_fidelity` is the gate that this transcription did not drift:
it re-derives Cut 4c-arm's entrypoint defaults from its SOURCE with `ast` and compares them key
by key, then hashes a fixed-command plant trajectory. Nothing here is an improvement on Cut
4c-arm; improvements would break the comparison this node exists to make.

WHAT IS NEW HERE, in three layers.

1. THE COMPOSED ENDPOINT MAP AND ITS JACOBIAN. Cut 4c-arm uses the one-step FM `f(s,u) -> Δs`
   forward (roll it, score it, pick a plan) and as an error (forecast minus outcome). Neither
   use differentiates it with respect to the command. `fm_endpoint` composes it H times from a
   true start state along a GIVEN command sequence and applies `fk_torch`, giving a
   differentiable `ŷ(u): R^{H x n_act} -> R^2`. `endpoint_vjp` takes `eᵀ ∂ŷ/∂u` in one backward
   pass; `endpoint_jac` materialises the full (2, H, n_act) tensor in two. This is Garibbo's
   middle factor: `∂e/∂φ = (∂e/∂y)(∂y/∂u)(∂u/∂φ)`, sensor supplies the first, policy the third.

2. THE ORACLE. `plant_endpoint_jac_fd` and `plant_onestep_jac_fd` obtain the SAME derivatives
   from the simulator by central finite differences, using `set_state` as the experimenter's
   ruler exactly as every exogenous-axis cut in this directory does (MuJoCo is memoryless, so a
   teleport-and-perturb query is exact). `direction_metrics` then scores an FM's Jacobian
   against the plant's on the axis Garibbo's derivation actually cares about — DIRECTION, not
   magnitude: "successful EBL requires only the correct sign of ∂y/∂a, whereas its magnitude
   merely scales the rate of learning".

3. THE TEACHERS. `action_gradient` returns, for one batch of executed reaches, the action-space
   teaching vector `g` of Eq. 2 under a named teacher. Every teacher returns an object of the
   same type and shape, (B, H, n_act), and is consumed by the same one line
   `(g.detach() * mu).sum()` — so `mixed@β` is literally `β·g_ebl + (1-β)·g_rbl`, the paper's
   weighted sum, and no arm has a private update path that could be doing the work.

SIGN CONVENTION, stated once. Everything in `action_gradient` is a DESCENT direction on a loss:
`g_ebl = (∂E/∂y)ᵀ ∂ŷ/∂u` with `E = ½‖y − y*‖²`, and `g_rbl = −δ (u − μ)/σ²` so that stepping
against it ascends reward. The surrogate `(g.detach() * μ_φ(s,goal)).sum()` handed to a
minimising optimiser therefore reproduces Eq. 1 for `β=0` (it is algebraically the standard
`−δ ∇_φ log π`) and Eq. 2 in general.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from mjc.arm_env import ArmEnv, collect_pool, fk

TEACHERS = ("rbl", "ebl_sensory", "ebl_imagined", "mixed",
            "ebl_committee", "ebl_committee_gated", "mixed_agree")
REWARDS = ("continuous", "binary")


# Configuration lives in `defaults.py`, which imports nothing — a Modal LOCAL ENTRYPOINT runs
# on the submitting machine, where `mjc/shared.py`'s rule says torch/mujoco may be absent, and
# this module imports torch at the top. Re-exported here so a Modal function body can keep
# importing them from `core`.
from mjc.jacobian_teacher.defaults import fork_defaults, teacher_defaults        # noqa: F401

# ===================================================================================== #
#  the substrate — Cut 4c-arm's closures, hoisted
# ===================================================================================== #
class ArmSubstrate:
    """Every helper `run_arm_readapt` defines, as methods. Same arithmetic, same seed
    offsets. The only additions are marked NEW."""

    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fs = cfg["frame_skip"]
        self.H = cfg["plan_H"]
        self.n = cfg["n_links"]
        self.SD, self.AD = 2 * self.n, self.n
        self.qc = np.array(cfg["q_center"][: self.n], dtype=np.float64)
        self.Ls = np.asarray(cfg["link_lengths"][: self.n], dtype=np.float64)
        self.Lt = torch.tensor(self.Ls, device=self.device, dtype=torch.float32)
        self.B = cfg["n_eval"]
        self.huber = nn.HuberLoss(delta=1.0)
        self.norm = None
        self.band_oversample = int(cfg.get("band_oversample", 40))
        self.last_band_stats = None

    # ---------------------------------------------------------------- env
    def make_env(self, b) -> ArmEnv:
        c = self.cfg
        return ArmEnv(dict(n_links=self.n, link_lengths=c["link_lengths"][: self.n],
                           link_masses=c["link_masses"][: self.n],
                           joint_damping=c["joint_damping"], gear=c["gear"],
                           curl_field={"b": float(b)}))

    # ---------------------------------------------------------------- kinematics
    def fk_torch(self, q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(self.Lt * torch.cos(ang)).sum(1),
                            (self.Lt * torch.sin(ang)).sum(1)], 1)

    def fk_jac(self, q):
        """NEW. Analytic planar ∂fk/∂q, (B, 2, n). Gate 2's kinematic half checks it against
        MuJoCo's own `mj_jacSite`; `fk` itself is already verified to ~1e-12 (arm_substrate P0)."""
        q = np.asarray(q, dtype=np.float64)
        ang = np.cumsum(q, axis=-1)                              # (B, n)
        s, c = np.sin(ang), np.cos(ang)
        # ∂y_x/∂q_j = −Σ_{i>=j} L_i sin(ang_i);  ∂y_y/∂q_j = +Σ_{i>=j} L_i cos(ang_i)
        sx = np.cumsum((self.Ls * s)[..., ::-1], axis=-1)[..., ::-1]
        cx = np.cumsum((self.Ls * c)[..., ::-1], axis=-1)[..., ::-1]
        return np.stack([-sx, cx], axis=-2)                       # (B, 2, n)

    # ---------------------------------------------------------------- FM
    def mlp(self, seed):
        c = self.cfg
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = c["fm_hidden"], c["fm_layers"]
        lyr = [nn.Linear(self.SD + self.AD, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, self.SD)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                nn.init.zeros_(m.bias)
        return net.to(self.device)

    def set_norm(self, S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32)
        Y = (S2 - S).astype(np.float32)
        self.norm = {k: torch.tensor(v, device=self.device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}
        return self.norm

    def train_steps(self, net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=self.device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=self.device)
        Xn = (X - self.norm["mx"]) / self.norm["sx"]
        Yn = (Y - self.norm["my"]) / self.norm["sy"]
        bs = min(self.cfg["fm_batch"], len(S))
        net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=self.device)
            opt.zero_grad()
            self.huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta(self, net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32),
                             device=self.device)
            return (net((X - self.norm["mx"]) / self.norm["sx"]) * self.norm["sy"]
                    + self.norm["my"]).cpu().numpy()

    def fm_step_t(self, net, s, u):
        """One differentiable FM step, in torch. `s` (B, SD), `u` (B, AD) -> s' (B, SD)."""
        x = torch.cat([s, u], 1)
        return s + (net((x - self.norm["mx"]) / self.norm["sx"]) * self.norm["sy"]
                    + self.norm["my"])

    def pool(self, cenv, nn_, seed):
        c = self.cfg
        return collect_pool(cenv, nn_, np.random.default_rng(seed), self.fs,
                            self.qc, c["q_range"], c["v_explore"])

    # ---------------------------------------------------------------- CEM
    def mpc_plan(self, net, states, goals, rng, horizon=None, terminal_only=False):
        """The fork's CEM. `terminal_only=True` swaps the running tip cost + terminal velocity
        penalty for the TERMINAL endpoint distance alone — the objective `ebl_imagined`
        descends — so Phase A gate 4 can compare gradient planning against sampling planning on
        the SAME objective rather than on two different ones. The default path is unchanged."""
        c = self.cfg
        Hh = horizon or self.H
        Bn = states.shape[0]
        Kc, ne_el = c["k_shoot"], c["cem_elite"]
        mu = np.zeros((Bn, Hh, self.AD), np.float32)
        sig = np.full((Bn, Hh, self.AD), c["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=self.device, dtype=torch.float32).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=self.device, dtype=torch.float32).repeat_interleave(Kc, 0)
        for _ in range(c["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, Hh, self.AD)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone()
                seqs_t = torch.tensor(seqs.reshape(Bn * Kc, Hh, self.AD), device=self.device)
                cost = torch.zeros(Bn * Kc, device=self.device)
                for h in range(Hh):
                    s = self.fm_step_t(net, s, seqs_t[:, h, :])
                    if not terminal_only:
                        cost = cost + (self.fk_torch(s[:, :self.n]) - g_t).norm(dim=1)
                if terminal_only:
                    cost = (self.fk_torch(s[:, :self.n]) - g_t).norm(dim=1)
                else:
                    cost = cost + c["vel_pen"] * s[:, self.n:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1)
            sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    # ---------------------------------------------------------------- reach geometry
    def eval_geometry(self, seed, B=None, dir_center=None, dir_halfwidth=None):
        """Cut 4c-arm's `eval_geometry`, plus (NEW) an optional TIP-SPACE DIRECTION BAND.

        `dir_center=None` reproduces the fork exactly: joint-space goal proposals rejected
        until the induced tip displacement lands in [reach_lo, reach_hi]. With a band set, a
        proposal must ALSO point within `dir_halfwidth` degrees of `dir_center` — this is the
        generalization axis (Izawa & Shadmehr train at one target direction and probe
        -30..+30 degrees away; Garibbo Fig. 2e)."""
        c = self.cfg
        B = B or self.B
        rng = np.random.default_rng(seed)
        n, Ls = self.n, self.Ls
        q0 = self.qc[None, :] + rng.uniform(-c["q_jit"], c["q_jit"], (B, n))
        t0 = fk(q0, Ls)
        qg = np.empty_like(q0)
        lo, hi = c["reach_lo"], c["reach_hi"]
        if dir_center is None:
            # THE FORK'S PATH, unchanged: `reach_tries` sequential proposals, keep the least
            # penalised. Kept verbatim so an unbanded call reproduces Cut 4c-arm exactly.
            for b in range(B):
                best, best_pen = None, np.inf
                for _ in range(c["reach_tries"]):
                    d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                    cand = q0[b] + c["reach_amp"] * d
                    dist = float(np.linalg.norm(fk(cand, Ls) - t0[b]))
                    pen = max(0.0, lo - dist) + max(0.0, dist - hi)
                    if pen < best_pen:
                        best, best_pen = cand, pen
                    if pen == 0.0:
                        break
                qg[b] = best
        else:
            # BANDED: a direction band and the reach ring together admit only a few percent of
            # random joint deltas, so `reach_tries` sequential draws would silently return a
            # mostly OFF-BAND probe set — the failure mode `eval_geometry` was written to stop,
            # one level up. Over-sample vectorised instead (`fk` is batched) and draw uniformly
            # from the feasible candidates, falling back to least-penalised only if none exist.
            M = c["reach_tries"] * self.band_oversample
            off_all = np.empty(B)
            for b in range(B):
                d = rng.normal(0, 1, (M, n))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                cand = q0[b][None, :] + c["reach_amp"] * d
                dvec = fk(cand, Ls) - t0[b]
                dist = np.linalg.norm(dvec, axis=1)
                ang = np.degrees(np.arctan2(dvec[:, 1], dvec[:, 0]))
                off = np.abs((ang - dir_center + 180.0) % 360.0 - 180.0)
                ok = (dist >= lo) & (dist <= hi) & (off <= dir_halfwidth)
                if ok.any():
                    pick = rng.choice(np.flatnonzero(ok))
                else:
                    pen = (np.maximum(0.0, lo - dist) + np.maximum(0.0, dist - hi)
                           + np.maximum(0.0, off - dir_halfwidth) / 90.0)
                    pick = int(np.argmin(pen))
                qg[b] = cand[pick]
                off_all[b] = off[pick]
            self.last_band_stats = {"center": float(dir_center),
                                    "halfwidth": float(dir_halfwidth),
                                    "in_band_frac": float((off_all <= dir_halfwidth).mean()),
                                    "mean_abs_offset": float(off_all.mean())}
        goals = fk(qg, Ls).astype(np.float32)
        starts = np.concatenate([q0, rng.normal(0, c["v0_std"], (B, n))], 1).astype(np.float32)
        return starts, goals, t0.astype(np.float32)

    # ---------------------------------------------------------------- execution
    def execute(self, cenv, starts, seqs):
        """NEW (the open-loop half of Cut 4c-arm's `rollout`, factored out). Executes a GIVEN
        (B, H, AD) command sequence from `starts` in `cenv` and returns the final states.
        Cut 4c-arm's `rollout` re-queries a plan every `replan_every` steps; a committed motor
        program is the `replan_every == H` case, and every teacher here trains one, so the
        sequence is fixed before the first step and this is exact."""
        states = starts.copy()
        Hh = seqs.shape[1]
        for step in range(Hh):
            a = seqs[:, step, :]
            for b in range(len(states)):
                cenv.set_state(states[b, :self.n].astype(np.float64),
                               states[b, self.n:].astype(np.float64))
                states[b], _ = cenv.step(a[b], self.fs)
        return states

    def rollout(self, plan_fn, replan_every, cenv, starts, goals, t0, record=None):
        """Cut 4c-arm's `rollout`, with the eval set passed in rather than closed over.
        Returns (median tip-goal distance, signed LATERAL deviation, signed RADIAL deviation).

        `lateral` is the fork's, sign taken against the curl: F = b·[[0,-1],[1,0]]·v is a +90°
        rotation of the velocity, so `perp` is the direction the field pushes the hand and a
        model over-compensating an absent field deviates the other way. `radial` is NEW and is
        the one line this node needs from the fork: Garibbo's dysmetria triple separates into
        hypermetria (overshoot, radial > 0), hypometria (undershoot, radial < 0) and
        displacement (lateral), and only a SIGNED along-reach readout can tell the first two
        apart from each other or from a plain miss."""
        states = starts.copy()
        plan = None
        for step in range(self.H):
            if step % replan_every == 0:
                plan = plan_fn(states, goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(len(states)):
                cenv.set_state(states[b, :self.n].astype(np.float64),
                               states[b, self.n:].astype(np.float64))
                s0b = cenv.get_state()
                s2, _ = cenv.step(acts[b], self.fs)
                if record is not None:
                    record.append((s0b, acts[b].astype(np.float32), s2))
                states[b] = s2
        tips = fk(states[:, :self.n].astype(np.float64), self.Ls).astype(np.float32)
        return self.readout(tips, goals, t0)

    def readout(self, tips, goals, t0):
        """NEW (factored). (median distance, median signed lateral, median signed radial)."""
        dist = float(np.median(np.linalg.norm(tips - goals, axis=1)))
        rd = goals - t0
        rd = rd / np.maximum(np.linalg.norm(rd, axis=1, keepdims=True), 1e-9)
        perp = np.stack([-rd[:, 1], rd[:, 0]], 1)
        err = tips - goals
        return (dist,
                float(np.median((err * perp).sum(1))),
                float(np.median((err * rd).sum(1))))

    # ---------------------------------------------------------------- BC motor program
    def build_policy(self):
        c = self.cfg
        h, L = c["pol_hidden"], c["pol_layers"]
        lyr = [nn.Linear(self.SD + 2, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, self.AD * self.H), nn.Tanh()]
        return nn.Sequential(*lyr).to(self.device)

    def clone_policy(self, net_fm, seed_off=55, dir_center=None, dir_halfwidth=None,
                     verbose=True):
        """Cut 4c-arm's `ballistic_bc` training half: clone the CEM planner's committed
        programs under `net_fm` into π(s, goal) -> H×n_act. Returns (policy, pol_norm).
        The band arguments are NEW and restrict the CLONING distribution the same way
        `eval_geometry` restricts the eval distribution, so a policy can be born inside the
        training band and probed outside it."""
        c = self.cfg
        rng = np.random.default_rng(self.cfg["seed"] + seed_off)
        n, Ls = self.n, self.Ls
        # The reach-band filter keeps only ~25% of joint-space candidates, so `bc_tuples` must
        # be sized to the KEPT count (Cut 4c-arm's gotcha (iv): a BC arm cloned from 78 kept
        # tuples visibly under-fits). A DIRECTION band multiplies that again by roughly
        # 2·halfwidth/360, so the draw is grown until `bc_tuples` survive rather than fixed.
        draws = c["bc_tuples"] if dir_center is None else c["bc_tuples"] * self.band_oversample
        q0 = self.qc[None, :] + rng.uniform(-c["q_jit"], c["q_jit"], (draws, n))
        t0 = fk(q0, Ls)
        d = rng.normal(0, 1, (draws, n))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        gp = fk(q0 + c["reach_amp"] * d, Ls)
        dvec = gp - t0
        dist = np.linalg.norm(dvec, axis=1)
        keep = (dist >= c["reach_lo"]) & (dist <= c["reach_hi"])
        if dir_center is not None:
            ang = np.degrees(np.arctan2(dvec[:, 1], dvec[:, 0]))
            off = np.abs((ang - dir_center + 180.0) % 360.0 - 180.0)
            keep &= off <= dir_halfwidth
        keep = np.flatnonzero(keep)[: c["bc_tuples"]]
        q0, gp = q0[keep], gp[keep]
        nt = len(keep)
        if verbose:
            print(f"[bc] reach{'' if dir_center is None else '+direction'}-band filter kept "
                  f"{nt} of {draws} candidates ({100.0 * nt / max(draws, 1):.1f}%), "
                  f"target {c['bc_tuples']}", flush=True)
        assert nt >= 64, (f"BC kept only {nt} tuples — size `bc_tuples` to the KEPT count "
                          f"(Cut 4c-arm's warning; a direction band tightens it further)")
        s0 = np.concatenate([q0, rng.normal(0, c["v0_std"], (nt, n))], 1).astype(np.float32)
        gp = gp.astype(np.float32)
        plans = np.empty((nt, self.H, self.AD), np.float32)
        for i in range(0, nt, 128):
            j = min(i + 128, nt)
            plans[i:j] = self.mpc_plan(net_fm, s0[i:j], gp[i:j], rng)
        X = np.concatenate([s0, gp], 1).astype(np.float32)
        pn = {"mu": torch.tensor(X.mean(0), device=self.device),
              "sd": torch.tensor(X.std(0) + 1e-6, device=self.device)}
        Xt = (torch.tensor(X, device=self.device) - pn["mu"]) / pn["sd"]
        Yt = torch.tensor(plans.reshape(nt, self.AD * self.H), device=self.device)
        pol = self.build_policy()
        optp = torch.optim.Adam(pol.parameters(), lr=c["pol_lr"])
        lossf = nn.MSELoss()
        brng = np.random.default_rng(c["seed"] + seed_off + 1)
        pol.train()
        bs = min(512, nt)
        for _ in range(c["pol_steps"]):
            idx = torch.tensor(brng.integers(0, nt, size=bs), device=self.device)
            optp.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); optp.step()
        pol.eval()
        if verbose:
            print(f"[bc] cloned {nt} (state, goal) -> {self.AD * self.H}-dim motor programs",
                  flush=True)
        return pol, pn

    def policy_mu(self, pol, pn, states, goals):
        """Differentiable μ_φ(s, goal) -> (B, H, AD)."""
        x = torch.tensor(np.concatenate([states, goals], 1).astype(np.float32),
                         device=self.device)
        return pol((x - pn["mu"]) / pn["sd"]).reshape(len(states), self.H, self.AD)

    def policy_plan_fn(self, pol, pn):
        def plan_fn(states, goals):
            with torch.no_grad():
                return self.policy_mu(pol, pn, states, goals).cpu().numpy().astype(np.float32)
        return plan_fn


# ===================================================================================== #
#  the committee (SPEC Phase C) — K forward models in the random-prior idiom
# ===================================================================================== #
class RPFMember(nn.Module):
    """`../curiosity_control/`'s ensemble member (Osband et al. 2018): a trainable net plus a
    FROZEN random prior, prediction = net + β·prior. Where data covers a region the net
    compensates the prior and members AGREE; where data is sparse the distinct frozen priors
    make them DISAGREE. That is the property this node wants a direction reading of: the
    members' agreement about the SIGN of ∂y/∂u is a reward-free, oracle-free proxy for the
    thing Phase B can only measure against a finite-difference plant Jacobian.

    Drop-in for a plain FM: it is an `nn.Module` with the same signature, so `fm_step_t`,
    `mpc_plan`, `endpoint_vjp` and `endpoint_jac` all take a member unchanged."""

    def __init__(self, net, prior, rpf_beta):
        super().__init__()
        self.net, self.prior, self.rpf_beta = net, prior, float(rpf_beta)
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if self.rpf_beta <= 0.0:
            return self.net(x)
        return self.net(x) + self.rpf_beta * self.prior(x)


class CommitteeMean(nn.Module):
    """The ensemble MEAN as a single forward model — what `curiosity_control` plans through.
    Kept distinct from averaging the members' teaching vectors, because the endpoint map
    composes the one-step model H times: the Jacobian of the mean model is not the mean of the
    members' Jacobians, and which of the two a committee should teach with is the question."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], 0).mean(0)


def build_committee(sub: ArmSubstrate, seed, K, rpf_beta):
    return [RPFMember(sub.mlp(seed + j), sub.mlp(seed + j + 99991), rpf_beta).to(sub.device)
            for j in range(K)]


def train_committee(sub: ArmSubstrate, members, S, U, S2, steps, seed, bootstrap=True):
    """Each member on its own bootstrap resample and its own minibatch stream, so
    disagreement off-data is a real property of the fit and not just of the initialisation."""
    for j, m in enumerate(members):
        rng = np.random.default_rng(seed + 7000 * (j + 1))
        if bootstrap:
            idx = rng.integers(0, len(S), size=len(S))
            Sj, Uj, S2j = S[idx], U[idx], S2[idx]
        else:
            Sj, Uj, S2j = S, U, S2
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                               lr=sub.cfg["fm_lr"])
        sub.train_steps(m, opt, Sj, Uj, S2j, steps, rng)
    return members


def committee_direction_agreement(sub: ArmSubstrate, members, s0, seq):
    """The reward-free, oracle-free direction reading: how much the members agree on the SIGN
    of each entry of the composed endpoint Jacobian. Returns the per-entry agreement in [0,1]
    (1 = unanimous) alongside the committee-mean Jacobian."""
    Js = np.stack([endpoint_jac(sub, m, s0, seq) for m in members], 0)   # (K, B, 2, H, AD)
    return np.abs(np.sign(Js).mean(0)), Js.mean(0), Js


# ===================================================================================== #
#  the composed endpoint map, and its Jacobian both ways
# ===================================================================================== #
def fm_endpoint(sub: ArmSubstrate, net, s0_t, seq_t):
    """ŷ(u): roll the FM H times from `s0_t` (B, SD) along `seq_t` (B, H, AD), then `fk_torch`.
    Differentiable in `seq_t`. This composition is IDENTICAL to `mpc_plan`'s inner loop, which
    is what makes Phase A gate 4 (`ebl_imagined` at the matched FM ≈ `ballistic_cem`) a real
    check on the composition rather than on two different rollouts."""
    s = s0_t
    for h in range(seq_t.shape[1]):
        s = sub.fm_step_t(net, s, seq_t[:, h, :])
    return sub.fk_torch(s[:, :sub.n])


def endpoint_vjp(sub: ArmSubstrate, net, s0, seq, e, sign_mask=None):
    """`eᵀ ∂ŷ/∂u` in ONE backward pass. `e` (B, 2) is the DIRECTED sensory error `∂E/∂y`,
    detached — the FM supplies only the coordinate transform and never sees the error.

    `sign_mask` (2, AD) is the dysmetria intervention: it flips the FM's BELIEVED sign of
    ∂y_k/∂u_j uniformly over the horizon, the direct generalization of Garibbo Fig. 3g–j's
    per-component flip of the 2×2 sensitivity-derivative matrix to our 2×n_act. It is applied
    inside the product rather than to the FM's weights, so the model is otherwise untouched and
    the only thing that changed is the direction it teaches in."""
    s0_t = torch.tensor(s0, device=sub.device, dtype=torch.float32)
    seq_t = torch.tensor(seq, device=sub.device, dtype=torch.float32).requires_grad_(True)
    y = fm_endpoint(sub, net, s0_t, seq_t)
    e_t = torch.as_tensor(e, device=sub.device, dtype=torch.float32)
    if sign_mask is None:
        scalar = (e_t * y).sum()
        (g,) = torch.autograd.grad(scalar, seq_t)
        return g.detach().cpu().numpy()
    # per sensory dimension, so the mask can be applied to each row separately
    M = torch.as_tensor(np.asarray(sign_mask, np.float32), device=sub.device)   # (2, AD)
    out = torch.zeros_like(seq_t)
    for k in range(y.shape[1]):
        (gk,) = torch.autograd.grad((e_t[:, k] * y[:, k]).sum(), seq_t, retain_graph=True)
        out = out + gk * M[k].view(1, 1, -1)
    return out.detach().cpu().numpy()


def endpoint_jac(sub: ArmSubstrate, net, s0, seq):
    """The full ∂ŷ/∂u as (B, 2, H, AD), in two backward passes (one per sensory dimension)."""
    s0_t = torch.tensor(s0, device=sub.device, dtype=torch.float32)
    seq_t = torch.tensor(seq, device=sub.device, dtype=torch.float32).requires_grad_(True)
    y = fm_endpoint(sub, net, s0_t, seq_t)
    rows = []
    for k in range(y.shape[1]):
        (gk,) = torch.autograd.grad(y[:, k].sum(), seq_t, retain_graph=(k == 0))
        rows.append(gk.detach())
    return torch.stack(rows, 1).cpu().numpy()


def onestep_jac(sub: ArmSubstrate, net, s, u):
    """The FM's one-step ∂f(s,u)/∂u as (B, SD, AD), in SD backward passes."""
    s_t = torch.tensor(s, device=sub.device, dtype=torch.float32)
    u_t = torch.tensor(u, device=sub.device, dtype=torch.float32).requires_grad_(True)
    ds = sub.fm_step_t(net, s_t, u_t) - s_t
    rows = []
    for k in range(ds.shape[1]):
        (gk,) = torch.autograd.grad(ds[:, k].sum(), u_t, retain_graph=(k < ds.shape[1] - 1))
        rows.append(gk.detach())
    return torch.stack(rows, 1).cpu().numpy()


def plant_endpoint_jac_fd(sub: ArmSubstrate, cenv, s0, seq, eps):
    """THE ORACLE for the composed map: ∂y/∂u from the SIMULATOR by central differences.

    2·H·n_act open-loop rollouts per start state. `set_state` makes each one an exact,
    independent query (MuJoCo is memoryless) — the same experimenter's ruler every
    exogenous-axis cut in this directory uses. Cost is why `jac_n` subsamples the eval set."""
    B, Hh, AD = seq.shape
    J = np.zeros((B, 2, Hh, AD), np.float64)
    for h in range(Hh):
        for j in range(AD):
            sp = seq.copy(); sp[:, h, j] += eps
            sm = seq.copy(); sm[:, h, j] -= eps
            yp = fk(sub.execute(cenv, s0, sp)[:, :sub.n].astype(np.float64), sub.Ls)
            ym = fk(sub.execute(cenv, s0, sm)[:, :sub.n].astype(np.float64), sub.Ls)
            J[:, :, h, j] = (yp - ym) / (2.0 * eps)
    return J


def plant_onestep_jac_fd(sub: ArmSubstrate, cenv, s, u, eps):
    """THE ORACLE for the one-step map: ∂s'/∂u from the simulator, (B, SD, AD)."""
    B, AD = u.shape
    J = np.zeros((B, sub.SD, AD), np.float64)
    for j in range(AD):
        for sgn, acc in ((+1.0, 0), (-1.0, 1)):
            up = u.copy(); up[:, j] += sgn * eps
            out = np.empty((B, sub.SD), np.float64)
            for b in range(B):
                cenv.set_state(s[b, :sub.n].astype(np.float64), s[b, sub.n:].astype(np.float64))
                s2, _ = cenv.step(up[b], sub.fs)
                out[b] = s2
            J[:, :, j] += (out if acc == 0 else -out) / (2.0 * eps)
    return J


def direction_metrics(Jhat, Jtrue, weight_by_true=True):
    """Score a predicted Jacobian against the oracle on DIRECTION, which is the axis Garibbo's
    derivation depends on ("successful EBL requires only the correct sign of ∂y/∂a, whereas its
    magnitude merely scales the rate of learning"). Both arrays share shape (B, 2, ...).

    Returned:
      `cos`            — cosine between the flattened per-sample Jacobians (the whole operator)
      `cos_row`        — the same, per sensory dimension, then averaged
      `sign_agree`     — fraction of entries whose signs match
      `sign_agree_w`   — the same, weighted by |J_true| so entries the plant is insensitive on
                         cannot pad the score
      `scale`          — median ‖Ĵ‖/‖J‖, reported because Garibbo say magnitude sets the RATE:
                         a direction-accurate but badly-scaled Jacobian is a learning-rate
                         rescale, not a wrong teacher, and the two must not be conflated
    """
    A = Jhat.reshape(Jhat.shape[0], -1).astype(np.float64)
    Bm = Jtrue.reshape(Jtrue.shape[0], -1).astype(np.float64)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-12
    cos = (A * Bm).sum(1) / den
    rows = []
    for k in range(Jhat.shape[1]):
        a = Jhat[:, k].reshape(Jhat.shape[0], -1).astype(np.float64)
        b = Jtrue[:, k].reshape(Jtrue.shape[0], -1).astype(np.float64)
        d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        rows.append((a * b).sum(1) / d)
    agree = (np.sign(Jhat) == np.sign(Jtrue)).astype(np.float64)
    w = np.abs(Jtrue) if weight_by_true else np.ones_like(Jtrue)
    return {
        "cos": float(np.mean(cos)),
        "cos_median": float(np.median(cos)),
        "cos_row": [float(np.mean(r)) for r in rows],
        "sign_agree": float(agree.mean()),
        "sign_agree_w": float((agree * w).sum() / (w.sum() + 1e-12)),
        "scale": float(np.median(np.linalg.norm(A, axis=1) / (np.linalg.norm(Bm, axis=1) + 1e-12))),
    }


def vjp_direction_metrics(Jhat, Jtrue, e):
    """The teacher-relevant reading: the cosine between the TEACHING VECTORS the two Jacobians
    produce for the SAME realized error, `eᵀĴ` vs `eᵀJ`. A Jacobian can be poor entrywise and
    still teach in the right direction for the errors that actually occur (and vice versa), so
    this — not `direction_metrics['cos']` — is the quantity the policy update sees."""
    gh = np.einsum("bk,bkha->bha", e, Jhat)
    gt = np.einsum("bk,bkha->bha", e, Jtrue)
    a = gh.reshape(len(gh), -1); b = gt.reshape(len(gt), -1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    cos = (a * b).sum(1) / den
    return {"vjp_cos": float(np.mean(cos)), "vjp_cos_median": float(np.median(cos)),
            "vjp_sign_agree": float((np.sign(gh) == np.sign(gt)).mean())}


# ===================================================================================== #
#  the teachers — Garibbo eqs. 1–2, in one function
# ===================================================================================== #
def action_gradient(sub: ArmSubstrate, net_fm, teacher, s0, goals, mu_np, act, tips,
                    tcfg, baseline):
    """The action-space teaching vector `g` (B, H, AD) for one batch of EXECUTED reaches.

    Inputs are the same for every teacher: the start states, the goals, the policy mean
    `mu_np` at those inputs, the command sequence `act` actually executed (μ + σε, clipped),
    and the realized endpoints `tips`. What differs is only how `g` is formed.

    Returns (g, diagnostics). `g` is a DESCENT direction (see the module docstring).
    """
    sig = tcfg["sigma"]
    diag = {}

    def rbl_gradient():
        if tcfg["rbl_reward"] == "binary":
            r = (np.linalg.norm(tips - goals, axis=1) < tcfg["reward_radius"]).astype(np.float32)
        else:
            r = -np.linalg.norm(tips - goals, axis=1).astype(np.float32)
        # EMA baseline, updated BEFORE the advantage is formed — dynamics_shift §5's order
        # exactly (`baseline = R.mean() if None else 0.9*baseline + 0.1*R.mean()`, then
        # `adv = R - baseline`). Porting the order matters: updating after would make the
        # first batch's advantage identically zero.
        ema = tcfg["rbl_baseline_ema"]
        baseline[0] = (float(r.mean()) if baseline[0] is None
                       else ema * baseline[0] + (1.0 - ema) * float(r.mean()))
        delta = r - baseline[0]                                    # δ = r − v, Eq. 1
        diag["reward_mean"] = float(r.mean())
        diag["baseline"] = float(baseline[0])
        diag["delta_abs"] = float(np.abs(delta).mean())
        # −δ (u − μ)/σ²  : stepping AGAINST this ascends reward
        return -(delta[:, None, None] * (act - mu_np) / (sig ** 2)).astype(np.float32)

    def ebl_gradient(err):
        return endpoint_vjp(sub, net_fm, s0, act, err,
                            sign_mask=tcfg.get("sign_mask")).astype(np.float32)

    if teacher == "rbl":
        g = rbl_gradient()
    elif teacher == "ebl_sensory":
        # ∂E/∂y with E = ½‖y − y*‖²  ->  the REALIZED endpoint error
        g = ebl_gradient((tips - goals).astype(np.float32))
    elif teacher == "ebl_imagined":
        # the same gradient with the MODEL-PREDICTED error in place of the realized one: no
        # sensory feedback enters at all, so this is gradient planning amortised into π.
        with torch.no_grad():
            yhat = fm_endpoint(sub,
                               net_fm,
                               torch.tensor(s0, device=sub.device, dtype=torch.float32),
                               torch.tensor(act, device=sub.device, dtype=torch.float32)
                               ).cpu().numpy()
        diag["imagined_err"] = float(np.linalg.norm(yhat - goals, axis=1).mean())
        diag["forecast_err"] = float(np.linalg.norm(yhat - tips, axis=1).mean())
        g = ebl_gradient((yhat - goals).astype(np.float32))
    elif teacher in ("ebl_committee", "ebl_committee_gated", "mixed_agree"):
        # `net_fm` is a LIST of members here. Each supplies its own teaching vector for the
        # same realized error; what differs between the three variants is how they are pooled.
        err = (tips - goals).astype(np.float32)
        gs = np.stack([endpoint_vjp(sub, m, s0, act, err, sign_mask=tcfg.get("sign_mask"))
                       for m in net_fm], 0).astype(np.float32)          # (K, B, H, AD)
        g_mean = gs.mean(0)
        agree = np.abs(np.sign(gs).mean(0)).astype(np.float32)          # (B, H, AD) in [0,1]
        diag["dir_agree"] = float(agree.mean())
        diag["g_member_spread"] = float(gs.std(0).mean())
        if teacher == "ebl_committee":
            g = g_mean
        elif teacher == "ebl_committee_gated":
            # each action-space component scaled by the members' sign agreement on it: the
            # committee is allowed to teach only where it knows which way to push
            g = g_mean * agree
        else:
            # β read OFF the committee rather than swept — the trained-head seat in this cut.
            # Per component, so a committee that is unanimous about one joint and split about
            # another mixes differently on each rather than on one scalar.
            g_r = rbl_gradient()
            ms = tcfg["mix_scale"]
            if ms <= 0.0:
                ms = (float(np.linalg.norm(g_mean.reshape(len(g_mean), -1), axis=1).mean())
                      / max(float(np.linalg.norm(g_r.reshape(len(g_r), -1), axis=1).mean()),
                            1e-12))
                tcfg["mix_scale"] = ms
            diag["mix_scale"] = ms
            diag["beta_eff"] = float(agree.mean())
            g = agree * g_mean + (1.0 - agree) * ms * g_r
    elif teacher == "mixed":
        g_e = ebl_gradient((tips - goals).astype(np.float32))
        g_r = rbl_gradient()
        beta = tcfg["beta"]
        diag["g_ebl_norm"] = float(np.linalg.norm(g_e.reshape(len(g_e), -1), axis=1).mean())
        diag["g_rbl_norm"] = float(np.linalg.norm(g_r.reshape(len(g_r), -1), axis=1).mean())
        ms = tcfg["mix_scale"]
        if ms <= 0.0:
            ms = diag["g_ebl_norm"] / max(diag["g_rbl_norm"], 1e-12)
            tcfg["mix_scale"] = ms          # measured once, on the first batch, then held
        diag["mix_scale"] = ms
        g = beta * g_e + (1.0 - beta) * ms * g_r                     # Eq. 2
    else:
        raise ValueError(f"unknown teacher {teacher!r}; expected one of {TEACHERS}")

    if tcfg["respect_clip"]:
        g = g * (np.abs(act) < 1.0 - 1e-6).astype(np.float32)
    diag["g_norm"] = float(np.linalg.norm(g.reshape(len(g), -1), axis=1).mean())
    diag["clip_frac"] = float((np.abs(act) >= 1.0 - 1e-6).mean())
    return g, diag


# ===================================================================================== #
#  the training loop every teacher shares
# ===================================================================================== #
def make_reach_pool(sub: ArmSubstrate, seed, n_pairs, dir_center=None, dir_halfwidth=None):
    """A fixed pool of (start, goal, t0) training reaches, drawn once and SHARED by every
    teacher. Two reasons it is a pool rather than fresh draws per update: the rejection loop
    in `eval_geometry` is the expensive part of a batch, and — the load-bearing one — every
    arm then trains on the same reaches in the same order, so the executed-reach budget is
    matched pair-for-pair and not merely in count."""
    return sub.eval_geometry(seed, B=n_pairs, dir_center=dir_center,
                             dir_halfwidth=dir_halfwidth)


def evaluate_policy(sub: ArmSubstrate, pol, pn, cenv, starts, goals, t0):
    """The committed motor program executed open-loop (replan_every = H), deterministic mean
    — the exploration noise is a property of TRAINING, not of the policy being graded."""
    return sub.rollout(sub.policy_plan_fn(pol, pn), sub.H, cenv, starts, goals, t0)


def train_teacher(sub: ArmSubstrate, net_fm, pol, pn, env_train, teacher, tcfg,
                  pool, eval_sets, seed, verbose=True, tag=""):
    """Adapt ONE policy under ONE teacher for `reach_budget` EXECUTED reaches.

    Every arm gets: the same initial policy (the caller passes a fresh copy of the same
    behavior-cloned motor program), the same `pool` of training reaches drawn in the same
    order from the same generator, and the same exploration noise draws. What differs is the
    single line that forms `g`. So a difference in the adaptation curve is a difference in the
    teaching signal and cannot be a difference in what was executed or when.

    `eval_sets` is `{name: (starts, goals, t0, env)}`; every entry is graded at each readout.

    Returned `curve` rows carry, besides the endpoint readouts:
      `grad_snr`  — the cosine between the two half-batch mean action gradients. Garibbo:
                    "the RBL action gradient is known to have a higher variance (i.e., noisier)
                    than the equivalent EBL gradient", and their motor-variability prediction
                    (Fig. 2b–c) rests on it. This is that quantity, measured directly, and it
                    is the honest form of the Izawa & Shadmehr variability readout HERE: σ is
                    FIXED and shared across arms in this port, so executed trial-to-trial
                    variability is identical by construction and the teacher's noise can only
                    show up in the gradient and in the residual jitter of the learned mean.
      `mu_jitter` — ‖μ_t − μ_{t−1}‖ on the fixed eval set between consecutive readouts: how
                    much the committed program is still moving. At plateau this is the learned
                    half of trial-to-trial variability.
    """
    P_starts, P_goals, P_t0 = pool
    npool = len(P_starts)
    Bt = tcfg["train_batch"]
    lr = tcfg["lr_rbl"] if teacher == "rbl" else tcfg["lr_ebl"]
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    # SAME draws for every teacher — the control that makes the budget matched pair-for-pair
    brng = np.random.default_rng(seed + 900)
    erng = np.random.default_rng(seed + 901)
    baseline = [None]
    t_start = time.time()
    reaches = 0
    # LOG-SPACED readouts by default. The smoke showed `ebl_sensory` reaching 90% of its gain
    # inside a few hundred executed reaches while `rbl` is still climbing at twenty thousand —
    # a uniform grid coarse enough to cover the reward arm cannot resolve the error arm at all,
    # and "does it teach faster" is the headline. `eval_every` remains the fallback.
    sched = sorted({int(x) for x in (tcfg.get("eval_schedule") or
                                     range(tcfg["eval_every"], tcfg["reach_budget"] + 1,
                                           tcfg["eval_every"]))
                    if 0 < int(x) <= tcfg["reach_budget"]})
    si = 0
    curve = []
    prev_mu = None

    def readout(nr):
        nonlocal prev_mu
        row = {"reaches": int(nr)}
        for name, (st, gl, t0, cenv) in eval_sets.items():
            d, lat, rad = evaluate_policy(sub, pol, pn, cenv, st, gl, t0)
            row[name] = d
            row[name + "_lat"] = lat
            row[name + "_rad"] = rad
        st, gl, _, _ = next(iter(eval_sets.values()))
        with torch.no_grad():
            mu_now = sub.policy_mu(pol, pn, st, gl).cpu().numpy()
        row["mu_jitter"] = (float(np.linalg.norm((mu_now - prev_mu).reshape(len(mu_now), -1),
                                                 axis=1).mean()) if prev_mu is not None
                            else float("nan"))
        prev_mu = mu_now
        return row

    curve.append(readout(0))
    while reaches < tcfg["reach_budget"]:
        idx = brng.integers(0, npool, size=Bt)
        st, gl = P_starts[idx], P_goals[idx]
        mu_t = sub.policy_mu(pol, pn, st, gl)                       # (B, H, AD), with grad
        mu_np = mu_t.detach().cpu().numpy()
        eps = erng.normal(0, 1, mu_np.shape).astype(np.float32)
        act = np.clip(mu_np + tcfg["sigma"] * eps, -1, 1).astype(np.float32)
        fin = sub.execute(env_train, st, act)
        tips = fk(fin[:, :sub.n].astype(np.float64), sub.Ls).astype(np.float32)
        g, diag = action_gradient(sub, net_fm, teacher, st, gl, mu_np, act, tips,
                                  tcfg, baseline)
        half = Bt // 2
        ga, gb = g[:half].mean(0).ravel(), g[half:].mean(0).ravel()
        snr = float((ga * gb).sum()
                    / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-12))
        gt = torch.tensor(g, device=sub.device)
        opt.zero_grad()
        (gt * mu_t).sum().div(len(g)).backward()                    # Eq. 2's downstream term
        opt.step()
        reaches += Bt
        if (si < len(sched) and reaches >= sched[si]) or reaches >= tcfg["reach_budget"]:
            while si < len(sched) and reaches >= sched[si]:
                si += 1
            row = readout(reaches)
            row.update({k: v for k, v in diag.items()})
            row["grad_snr"] = snr
            curve.append(row)
            if verbose:
                extra = ""
                if "imagined_err" in row:
                    extra = (f" imag={row['imagined_err']:.4f} "
                             f"fcast={row['forecast_err']:.4f}")
                print(f"  [{tag}{teacher:13s} r={reaches:6d}] "
                      + "  ".join(f"{k}={row[k]:.4f}" for k in eval_sets if k in row)
                      + f"  |g|={row['g_norm']:.3g} snr={snr:+.3f} "
                        f"jit={row['mu_jitter']:.4f} clip={row['clip_frac']:.2f}{extra}",
                      flush=True)
    if verbose:
        print(f"  [{tag}{teacher:13s} done] {reaches} executed reaches in "
              f"{time.time() - t_start:.1f}s", flush=True)
    return curve


def summarize_curve(curve, eval_names, plateau_evals=4):
    """Adaptation-curve summary: where it started, where it ended, how fast it got there, and
    how much it still moves at plateau."""
    out = {}
    tail = curve[-plateau_evals:] if len(curve) >= plateau_evals else curve[1:]
    for name in eval_names:
        rows = [r for r in curve if name in r]
        vals = [r[name] for r in rows]
        if not vals:
            continue
        out[name + "_start"] = vals[0]
        out[name + "_final"] = vals[-1]
        out[name + "_best"] = float(np.min(vals))
        out[name + "_gain"] = vals[0] - vals[-1]
        for suf in ("_lat", "_rad"):
            if name + suf in rows[-1]:
                out[name + suf + "_start"] = rows[0][name + suf]
                out[name + suf + "_final"] = rows[-1][name + suf]
        tv = [r[name] for r in tail if name in r]
        out[name + "_plateau_sd"] = float(np.std(tv)) if len(tv) > 1 else float("nan")
        # reaches to reach 90% of the total gain achieved by this arm (its OWN scale) — a
        # speed readout that does not presume the arms share a ceiling
        target = vals[0] - 0.9 * (vals[0] - float(np.min(vals)))
        hit = next((r["reaches"] for r, v in zip(rows, vals) if v <= target), None)
        out[name + "_reaches_to_90"] = int(hit) if hit is not None else None
    jit = [r["mu_jitter"] for r in tail if not np.isnan(r.get("mu_jitter", np.nan))]
    snr = [r["grad_snr"] for r in curve if "grad_snr" in r]
    out["mu_jitter_plateau"] = float(np.mean(jit)) if jit else float("nan")
    out["grad_snr_mean"] = float(np.mean(snr)) if snr else float("nan")
    out["g_norm_mean"] = float(np.mean([r["g_norm"] for r in curve if "g_norm" in r]))
    return out


# ===================================================================================== #
#  the two FM readings, per rung
# ===================================================================================== #
def fm_readings(sub: ArmSubstrate, net, env_true, probe, jac_set, tcfg, pol=None, pn=None):
    """The pair of readings the SPEC asks for on every FM: the record's SCALAR (forecast error
    on the task probe) and this node's DIRECTION reading (Jacobian accuracy against the plant
    oracle), one-step and composed-endpoint.

    `arm_substrate` P5/P7 standing rule — "report damage/gain on this substrate, never a slope
    against `fm_err`" — is why both readings are reported per rung against the ladder's own
    transition axis, rather than one being regressed on the other."""
    pS, pU, pT = probe
    out = {}
    out["fm_err"] = float(np.linalg.norm(sub.fm_delta(net, pS, pU) - pT, axis=1).mean())

    # --- one-step direction accuracy on the task probe ---
    k = min(tcfg["jac_probe_n"], len(pS))
    sS, sU = pS[:k], pU[:k]
    J1_fm = onestep_jac(sub, net, sS, sU)
    J1_pl = plant_onestep_jac_fd(sub, env_true, sS.astype(np.float64),
                                 sU.astype(np.float64), tcfg["fd_eps"])
    out["jac1"] = direction_metrics(J1_fm, J1_pl)

    # --- composed endpoint Jacobian, on the sequences the POLICY actually emits ---
    st, gl, t0 = jac_set
    m = min(tcfg["jac_n"], len(st))
    st, gl, t0 = st[:m], gl[:m], t0[:m]
    if pol is not None:
        with torch.no_grad():
            seq = sub.policy_mu(pol, pn, st, gl).cpu().numpy().astype(np.float32)
    else:
        seq = np.zeros((m, sub.H, sub.AD), np.float32)
    JE_fm = endpoint_jac(sub, net, st, seq)
    JE_pl = plant_endpoint_jac_fd(sub, env_true, st, seq, tcfg["fd_eps"])
    out["jacE"] = direction_metrics(JE_fm, JE_pl)
    # the teaching direction itself, for the errors this policy actually makes
    tips = fk(sub.execute(env_true, st, seq)[:, :sub.n].astype(np.float64), sub.Ls)
    e = (tips - gl).astype(np.float64)
    out["jacE"].update(vjp_direction_metrics(JE_fm, JE_pl, e))
    out["endpoint_err_at_jac_set"] = float(np.linalg.norm(e, axis=1).mean())
    return out
