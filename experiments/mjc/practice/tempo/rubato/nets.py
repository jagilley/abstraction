"""rubato Phase 2 — the LEARNED cerebellum.

Phase 1's executor read the body's true constants. This file replaces it with a model fitted on
the body's OWN EXECUTED DATA at the slow tempi only, and asks how much of the oracle's in-band
set survives.

WHAT IS FITTED, AND WHY THE TARGET IS EXACTLY THIS. The plant holds its command constant across a
control step, so its exact discrete map is `v_{n+1} = alpha v_n + (1 - alpha) v_term u_n`, and
inverting it gives

    u_n = c_a * a_n + c_v * v_n,   a_n = (v_{n+1} - v_n) / dt,
    c_a = dt / ((1 - alpha) v_term) = 0.008717,   c_v = 1 / v_term = 0.200000

on this body. So the TRUE inverse is LINEAR in `(v, a)` and the minimal family has exactly the
right functional form and two free numbers. That is not a convenience: it is what makes the
inductive-bias question sharp. A linear model fitted anywhere on the `(v, a)` plane extrapolates
to the whole plane by construction; an MLP fitted on the slow tempi' corner of it has no reason
to. `accelerando`'s tempo axis moves `|v|` by `s` and `|a|` by `s^2`, so a faster tempo is
literally a request for a region of `(v, a)` that slow practice never visited — and the two
families are the two answers to "what does a cerebellum do when asked".
The recovered coefficients are reported against the analytic ones at every fit, so the linear
family's success or failure can be read as a number and not inferred from its execution.

THE DATA IS THE PRACTICE TRAVERSALS, RE-EXECUTED FOR THEIR VELOCITIES. `build_ladder`'s harvest
runs the reflex law with motor noise and records `(s0, cmds)` per drilled seam; `World.rollout_
path` replays exactly those on the plant and returns the full `(x, v)` at every control step. The
triples are then `(v_n, (v_{n+1} - v_n)/dt, u_n)` with `u_n` the command actually issued — the
body's own executed data, nothing modelled. The replay is deterministic and bit-identical to the
harvest, so this adds a grounding and no new information beyond the velocities.

NOTHING HERE IS HANDED TO THE INCUMBENT. The fitted model is an EXECUTOR: it converts a stored
kinematic path into commands on the learner's side of the meter. It is never a planner, it never
proposes a decision, and the reflex arm stays model-free and re-fit per tempo.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# the data: the body's own executed (v, a, u) triples
# --------------------------------------------------------------------------- #

def triples_from_pool(W, pool, dt, led=None, max_rend=None, rng=None):
    """Replay a harvest pool on the plant and return `(v, a, u)` for every control step.

    `pool` is `{seam: dict(s0=(n,4), cmds=(n,L,2))}` as `build_ladder`'s `harvest` produces it.
    Charged as a grounding: replaying a rendition IS a trial, even though the trial has been run
    before and the result is bit-identical.
    """
    V, A, U = [], [], []
    for k in sorted(pool):
        s0 = np.asarray(pool[k]["s0"], np.float32)
        cm = np.asarray(pool[k]["cmds"], np.float32)
        if max_rend is not None and len(s0) > int(max_rend):
            idx = (rng or np.random.default_rng(0)).permutation(len(s0))[:int(max_rend)]
            idx = np.sort(idx)
            s0, cm = s0[idx], cm[idx]
        st = W.rollout_path(s0, cm, led=led, charge=True)          # (n, L+1, 4)
        v = st[:, :-1, 2:]
        vn = st[:, 1:, 2:]
        V.append(v.reshape(-1, 2))
        A.append(((vn - v) / float(dt)).reshape(-1, 2))
        U.append(np.asarray(cm, np.float64).reshape(-1, 2))
    return (np.concatenate(V, 0), np.concatenate(A, 0), np.concatenate(U, 0))


def split_triples(n, frac_hold=0.2, seed=0):
    """A deterministic train/hold split by index, asserted disjoint by gate P-FH rather than
    assumed. Held out at the level of the SAMPLE, which is the right unit here: the question is
    whether the map generalises over `(v, a)`, not over renditions."""
    rng = np.random.default_rng(int(seed) + 90210)
    perm = rng.permutation(int(n))
    nh = int(round(float(frac_hold) * int(n)))
    return np.sort(perm[nh:]), np.sort(perm[:nh])


def feats(v, a):
    """The executor's input: velocity and acceleration, both channels, no piece, no tempo, no
    phase and no posture. A cerebellum that needed the piece would not be a body model."""
    return np.concatenate([np.asarray(v, np.float64), np.asarray(a, np.float64)], axis=-1)


# --------------------------------------------------------------------------- #
# family 1 — linear in (v, a). The true inverse's own functional form.
# --------------------------------------------------------------------------- #

def fit_linear(v, a, u, ridge=0.0):
    """Least squares `u = W [v; a] + b`, the GENERAL 2x4 map with a bias.

    Deliberately not given the body's isotropy or the knowledge that `u_x` depends only on
    `(v_x, a_x)`: handing the family the answer's structure would make its success a statement
    about the experimenter. What is reported instead is how close the fitted map lands to the
    analytic one — the diagonal entries against `c_v` and `c_a`, the off-diagonals against zero,
    and the bias against zero.
    """
    X = np.concatenate([feats(v, a), np.ones((len(v), 1))], 1)
    Y = np.asarray(u, np.float64)
    G = X.T @ X
    if ridge:
        G = G + float(ridge) * np.eye(G.shape[0])
    Wm = np.linalg.solve(G, X.T @ Y)                                    # (5, 2)
    return dict(kind="linear", W=Wm)


def linear_report(fit, bc):
    Wm = fit["W"]
    return dict(
        c_v_fit=[float(Wm[0, 0]), float(Wm[1, 1])],
        c_a_fit=[float(Wm[2, 0]), float(Wm[3, 1])],
        c_v_true=float(1.0 / bc["v_term"]),
        c_a_true=float(bc["dt"] / ((1.0 - bc["alpha"]) * bc["v_term"])),
        off_diag_max=float(max(abs(Wm[0, 1]), abs(Wm[1, 0]), abs(Wm[2, 1]), abs(Wm[3, 0]))),
        bias_max=float(np.max(np.abs(Wm[4]))),
        W=[[float(x) for x in r] for r in Wm])


# --------------------------------------------------------------------------- #
# family 2 — a generic MLP. The same data, no functional form given.
# --------------------------------------------------------------------------- #

def fit_mlp(v, a, u, seed=0, hidden=128, layers=3, epochs=400, batch=8192, lr=1e-3,
            device="cpu", log=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    X = torch.tensor(feats(v, a), dtype=torch.float32)
    Y = torch.tensor(np.asarray(u, np.float64), dtype=torch.float32)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-8
    Xn = (X - mu) / sd
    mods, d = [], Xn.shape[1]
    for _ in range(int(layers)):
        mods += [nn.Linear(d, int(hidden)), nn.SiLU()]
        d = int(hidden)
    mods += [nn.Linear(d, 2)]
    net = nn.Sequential(*mods).to(device)
    Xn, Y = Xn.to(device), Y.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    n = len(Xn)
    hist = []
    for ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, int(batch)):
            j = perm[i:i + int(batch)]
            opt.zero_grad()
            loss = ((net(Xn[j]) - Y[j]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sch.step()
        if ep % max(1, int(epochs) // 8) == 0 or ep == int(epochs) - 1:
            hist.append(dict(epoch=int(ep), mse=float(tot / n)))
    if log is not None:
        log(f"      [mlp] {int(epochs)} epochs, final train mse {hist[-1]['mse']:.3e} "
            f"(n = {n})")
    return dict(kind="mlp", net=net, mu=mu.cpu().numpy(), sd=sd.cpu().numpy(), device=device,
                hist=hist)


# --------------------------------------------------------------------------- #
# using a fitted model as the EXECUTOR
# --------------------------------------------------------------------------- #

def predict(fit, v, a):
    if fit["kind"] == "linear":
        X = np.concatenate([feats(v, a), np.ones(np.shape(v)[:-1] + (1,))], -1)
        return X @ fit["W"]
    import torch
    X = feats(v, a)
    shp = X.shape[:-1]
    with torch.no_grad():
        t = torch.tensor((X.reshape(-1, X.shape[-1]) - fit["mu"]) / fit["sd"],
                         dtype=torch.float32, device=fit["device"])
        y = fit["net"](t).cpu().numpy()
    return y.reshape(shp + (2,))


def inv_fitted(fit, dt):
    """Wrap a fitted model as an inverse-dynamics scheme with `inv_zoh`'s signature.

    The acceleration handed to the model is the DESTINATION grid's own one-step velocity
    difference — exactly the feature the model was fitted on — so nothing about the deployment
    differs from training except where in `(v, a)` the query lands.
    """
    def f(bc, v, v_next):
        a = (np.asarray(v_next, np.float64) - np.asarray(v, np.float64)) / float(dt)
        return np.clip(predict(fit, v, a), -1.0, 1.0)
    return f


def heldout_mse(fit, v, a, u):
    p = np.clip(predict(fit, v, a), -1.0, 1.0)
    d = p - np.asarray(u, np.float64)
    return dict(mse=float((d ** 2).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                max_abs=float(np.max(np.abs(d))), n=int(len(u)),
                u_ms=float((np.asarray(u, np.float64) ** 2).mean()))


def coverage(v_tr, a_tr, v_q, a_q):
    """WHERE THE QUERY LANDS RELATIVE TO WHERE THE FIT LIVED, per channel, so "extrapolation" is
    a measured quantity and not an adjective. Reports the training range and the fraction of
    query points outside it, plus how far outside in units of the training spread."""
    out = {}
    for nm, tr, q in (("v", v_tr, v_q), ("a", a_tr, a_q)):
        tr = np.abs(np.asarray(tr, np.float64)).max(-1)
        q = np.abs(np.asarray(q, np.float64)).max(-1)
        hi = float(np.percentile(tr, 99.5))
        out[nm] = dict(train_p995=hi, query_p995=float(np.percentile(q, 99.5)),
                       query_max=float(q.max()),
                       frac_outside=float(np.mean(q > hi)),
                       reach=float(q.max() / hi) if hi > 0 else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# the FORWARD model, derived from the fitted INVERSE — one body model, not two
# --------------------------------------------------------------------------- #

def fwd_from_linear(fit, dt):
    """The forward step implied by the fitted LINEAR inverse. **They share parameters exactly**:
    there is one body model and two directions of use, which is the whole claim the executor is
    making when it says it holds a model of the body.

    The fitted inverse is `u = W_v v + W_a a + b` with `a = (v' - v)/dt`. Solving for `a`,

        a  = W_a^{-1} (u - W_v v - b)
        v' = v + dt * a
        x' = x + dt * (v + v') / 2

    The velocity channel is an algebraic inversion of the fitted map and introduces NO new
    parameters. The position channel is the trapezoidal integral of it — that is kinematics, not
    dynamics: a learner that did not know position is the integral of velocity would not have a
    kinematic program to execute in the first place. The exact pair (`inv_zoh`, `fwd_zoh`) has
    the same property by construction, so the oracle and the fitted model are the same KIND of
    object and the comparison between them is a comparison of one body model's accuracy.

    `W_a` must be invertible; on a fit that has seen any acceleration at all it is (the analytic
    value is `0.008717 * I`, well conditioned). A singular fit raises rather than silently
    returning a pseudo-inverse.
    """
    Wm = np.asarray(fit["W"], np.float64)
    Wv, Wa, b = Wm[:2], Wm[2:4], Wm[4]
    Wa_inv = np.linalg.inv(Wa)
    dt = float(dt)

    def f(states, u):
        st = np.asarray(states, np.float64)
        v = st[..., 2:]
        a = (np.asarray(u, np.float64) - v @ Wv - b[None, :]) @ Wa_inv
        v1 = v + dt * a
        x1 = st[..., :2] + dt * 0.5 * (v + v1)
        return np.concatenate([x1, v1], axis=-1)
    return f


def pd_analytic(bc, note_s):
    """THE PRE-FIXED PD RULE'S CENTRE, from the body and the tempo and nothing else.

    With a correct forecast the correction loop carries no delay, so the position-error dynamics
    are `e_ddot + (A kd + 1/tau) e_dot + A kp e = 0`. Ask for a natural frequency of one NOTE —
    the correction settles inside the unit's own smallest musical division, which is the only
    time-scale the piece supplies — and for critical damping:

        omega = 2 pi / T_note,  kp = omega^2 / A,  kd = (2 omega - 1/tau) / A

    Apparatus only: `A`, `tau` and the note duration. No arm is read, nothing is trained on it,
    and it is computed before any treatment number exists. The grid the rule is actually searched
    over is CENTRED on this value (x1/4 .. x4 on kp, x1/2 .. x4 on kd) so that an edge means
    "the body wanted something four times away from the analytic value", which is a legible
    statement rather than a grid artefact.
    """
    A, tau = bc["accel"], bc["tau"]
    w = 2.0 * np.pi / float(note_s)
    return dict(omega=float(w), kp=float(w * w / A), kd=float(max(1e-6, (2 * w - 1 / tau) / A)))


def pd_deadbeat(bc, rho):
    """DECISION 36's PD RULE: place both eigenvalues of the EXACT discrete error system at `rho`.

    WHY THIS RULE, AND WHY NOT THE PREVIOUS ONE — stated before its grid is read.

    With the exact inverse as feedforward and a state estimate `s_hat`, the tracking error
    `e = (x* - x, v* - v)` obeys a two-state LINEAR DISCRETE system, exactly and not to first
    order. Writing `g = (1-alpha) v_term` and `h = v_term (dt - tau (1-alpha))`,

        e_{n+1} = [[1 - h kp,  tau(1-alpha) - h kd],
                   [  -g kp,        alpha - g kd  ]] e_n

    because the plant's own velocity update and the feedforward's definition cancel the tempo,
    the piece and the path out of the error dynamics entirely. Placing both eigenvalues at `rho`
    means `trace = 2 rho` and `det = rho^2`; the `kp*kd` terms cancel in the determinant, so BOTH
    conditions are linear in `(kp, kd)` and the solution is closed-form:

        kp = (1 - rho)^2 / ((1 - alpha) v_term dt)          [using h + g tau = v_term dt exactly]
        kd = (1 + alpha - 2 rho - h kp) / g

    Four reasons this replaces `pd_analytic`, all of them properties of the two rules rather than
    of any outcome:
      1. It is DISCRETE. The previous rule was a continuous-time critical-damping condition and
         had no notion of the control period at all, on a body where `dt/tau = 0.8` — the same
         reason `inv_ct` is not `inv_zoh` (decision 4).
      2. It is EXACT. The error system above is the real one, so eigenvalue placement is a
         placement and not an approximation.
      3. It is BODY-ONLY and TEMPO-FREE, which is what the error dynamics are. The previous rule
         tied `omega` to the note duration, which made the gains tempo-dependent by construction
         — and that is precisely what made `kd = (2 omega - 1/tau)/A` go NEGATIVE and clamp to
         zero at every tempo slower than `T_note > 4 pi tau = 377 ms` (`kc1`: H = 32 and H = 16
         both degenerate) while demanding 4x the gain at H = 8 and 16x too much at H = 2.
      4. The free parameter is ONE NUMBER with an operational meaning — the per-control-step
         decay of the tracking error — so the search is a 1-D grid over `rho` and an edge says
         something specific ("the body wanted a faster/softer correction than the grid spans")
         instead of naming a corner of a 2-D multiplier box.

    `rho = 0` is deadbeat: any error is annihilated in two control steps. It is deliberately NOT
    the rule, because deadbeat has no robustness to model error and this world contains a command
    rotation the body model is denied by decision 5; `rho` is gridded and the margin is measured.
    Beyond `rho ~ 0.75` the placement needs `kd < 0`; that is a valid placement and is allowed and
    reported rather than clipped, so the grid's top end is not silently a different rule.
    """
    al, vt, dt, tau = bc["alpha"], bc["v_term"], bc["dt"], bc["tau"]
    g = (1.0 - al) * vt
    h = vt * (dt - tau * (1.0 - al))
    kp = (1.0 - float(rho)) ** 2 / ((1.0 - al) * vt * dt)
    kd = (1.0 + al - 2.0 * float(rho) - h * kp) / g
    return dict(rho=float(rho), kp=float(kp), kd=float(kd), g=float(g), h=float(h))
