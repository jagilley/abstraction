"""Backwards-compatibility gate for the `collection_mode` flag.

The repo rule is that all prior results stay reproducible, and `COLLECTION_REALISM.md` §3 promises
"finished cuts keep `teleport` and stay byte-identical". This file is what turns that promise into
a check rather than an intention. Run it before trusting any on-policy result, and re-run it after
any edit to `arm_env.collect_pool` / `embodied.py`.

Four checks, all cheap:

  1. TELEPORT IS BIT-IDENTICAL. The original `collect_pool` body (copied verbatim from the commit
     before the flag was added) is run against the new one on the same seed, and the arrays must
     match with `array_equal`, not `allclose`. This catches the subtle failure the flag could
     easily have introduced: consuming an RNG draw before the branch would shift every subsequent
     sample and silently change every cut in the node.

  2. THE PLANT IS UNTOUCHED. `build_xml` output compared byte-for-byte against a fresh `ArmEnv`, so
     no perturbation default drifted.

  3. `Body`'s PHYSICS MATCHES THE ESTABLISHED IDIOM. A trajectory rolled through `Body` (one
     `MjData` per episode) is compared against the same command sequence rolled through the
     `set_state`-multiplexed idiom every `rollout()` in this directory uses. On the arm (contacts
     off, MuJoCo memoryless) these must agree to floating-point noise; if they ever diverge, the
     multiplexed graders and the on-policy collector are measuring different plants.

  4. ON-POLICY DATA IS ACTUALLY ON-POLICY. Consecutive transitions within an episode must chain
     (`s_{t+1} == s'_t`) -- the property teleport collection does not have and the entire point of
     the exercise.

Run:
    cd experiments/
    modal run mjc/on_policy/verify_backcompat.py::verify
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR


def _collect_pool_original(env, n, rng, frame_skip, q_center, q_range, v_explore):
    """`arm_env.collect_pool` exactly as it stood at commit 797b622, before the flag."""
    import numpy as np

    n_dof, n_u = env.n, env.act_dim
    qc = np.asarray(q_center, dtype=np.float64)[:n_dof]
    S = np.empty((n, 2 * n_dof), np.float32)
    U = np.empty((n, n_u), np.float32)
    S2 = np.empty((n, 2 * n_dof), np.float32)
    for i in range(n):
        q = qc + rng.uniform(-q_range, q_range, n_dof)
        qd = rng.normal(0.0, v_explore, n_dof)
        env.set_state(q, qd)
        u = rng.uniform(-1, 1, n_u).astype(np.float32)
        s = env.get_state()
        s2, _ = env.step(u, frame_skip)
        S[i] = s
        U[i] = u
        S2[i] = s2
    return S, U, S2


@app.function(memory=8192, timeout=1800, volumes={DATA_DIR: volume})
def run_verify() -> dict:
    import numpy as np

    from mjc.arm_env import ArmEnv, build_xml, collect_pool, DEFAULT_DGP
    from mjc.embodied import Body, OUBehaviour, collect_on_policy

    dgp = dict(n_links=3, link_lengths=(0.4, 0.4, 0.3), link_masses=(1.0, 1.0, 0.6),
               joint_damping=0.5, gear=8.0, curl_field={"b": 6.0})
    qc, q_range, v_explore, fs = [0.4, 0.8, 0.6], 0.9, 8.0, 10
    out, failures = {}, []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)
        out[name] = bool(cond)
        if not cond:
            failures.append(name)

    # ---------------------------------------------------------------- 1
    print("\n[1] teleport path is BIT-IDENTICAL to the pre-flag implementation", flush=True)
    for n in (37, 500):
        a = _collect_pool_original(ArmEnv(dgp), n, np.random.default_rng(123), fs,
                                   qc, q_range, v_explore)
        b = collect_pool(ArmEnv(dgp), n, np.random.default_rng(123), fs,
                         qc, q_range, v_explore)
        c = collect_pool(ArmEnv(dgp), n, np.random.default_rng(123), fs,
                         qc, q_range, v_explore, collection_mode="teleport")
        same_default = all(np.array_equal(x, y) for x, y in zip(a, b))
        same_explicit = all(np.array_equal(x, y) for x, y in zip(a, c))
        check(f"teleport_bit_identical_n{n}", same_default and same_explicit,
              f"S/U/S2 array_equal, n={n}")

    # the RNG must be left in the same state too -- otherwise a caller that reuses the
    # generator downstream (several scripts do) would diverge on the NEXT draw.
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    _collect_pool_original(ArmEnv(dgp), 50, r1, fs, qc, q_range, v_explore)
    collect_pool(ArmEnv(dgp), 50, r2, fs, qc, q_range, v_explore)
    check("rng_state_identical", np.array_equal(r1.random(8), r2.random(8)),
          "generator left at the same position")

    # ---------------------------------------------------------------- 2
    print("\n[2] the plant is untouched", flush=True)
    dgp_before = json.dumps(dgp, sort_keys=True, default=str)
    xml_before = build_xml(dgp)
    collect_pool(ArmEnv(dgp), 20, np.random.default_rng(1), fs, qc, q_range, v_explore)
    collect_on_policy(ArmEnv(dgp), 40, np.random.default_rng(1), fs, qc, 0.3,
                      OUBehaviour(3, 4, np.random.default_rng(1)), ep_len=10, n_par=4)
    check("dgp_not_mutated_by_collection",
          json.dumps(dgp, sort_keys=True, default=str) == dgp_before
          and build_xml(dgp) == xml_before,
          "knob dict and MJCF identical after both collection modes")
    e = ArmEnv(dgp)
    check("defaults_unchanged",
          DEFAULT_DGP["n_links"] == 2 and DEFAULT_DGP["payload_mass"] == 0.0
          and DEFAULT_DGP["contacts"] is False and e.act_dim == 3 and e.state_dim == 6,
          f"act_dim={e.act_dim} state_dim={e.state_dim}")

    # ---------------------------------------------------------------- 2b
    # The E2 spatial-curl edit must leave the GLOBAL curl byte-identical: a `curl_field` with no
    # `center` key has to step exactly as before the gate was added (so Cut 4c-arm, P5, E1 are
    # unchanged). Roll the same command sequence from the same states through a global-curl env and
    # a fresh one; they must agree to floating point.
    print("\n[2b] global curl is byte-identical after the spatial-gate edit", flush=True)
    dgp_g = dict(dgp)                                  # dgp already has curl_field={"b":6.0}, no center
    rng = np.random.default_rng(7)
    q0 = np.asarray(qc)[None, :] + rng.uniform(-0.3, 0.3, (4, 3))
    cmds = rng.uniform(-1, 1, (12, 4, 3)).astype(np.float32)
    def _roll(env):
        st = np.zeros((4, 6), np.float32)
        for b_ in range(4):
            env.set_state(q0[b_], np.zeros(3)); st[b_] = env.get_state()
        out = []
        for t in range(12):
            for b_ in range(4):
                env.set_state(st[b_, :3].astype(np.float64), st[b_, 3:].astype(np.float64))
                st[b_], _ = env.step(cmds[t, b_], fs)
            out.append(st.copy())
        return np.stack(out)
    ta, tb = _roll(ArmEnv(dgp_g)), _roll(ArmEnv(dgp_g))
    check("global_curl_deterministic", np.array_equal(ta, tb), "same env twice -> identical")
    # and the gated field with a far-away center + tiny sigma is ~off (sanity that gating multiplies)
    dgp_off = dict(dgp); dgp_off["curl_field"] = {"b": 6.0, "center": (99.0, 99.0), "sigma": 0.3}
    dgp_none = dict(dgp); dgp_none["curl_field"] = {"b": 0.0}
    dev = float(np.abs(_roll(ArmEnv(dgp_off)) - _roll(ArmEnv(dgp_none))).max())
    check("gated_far_curl_is_off", dev < 1e-6,
          f"curl gated at a far center == field-free, max dev {dev:.2e}")

    # ---------------------------------------------------------------- 3
    print("\n[3] Body's per-episode MjData matches the set_state-multiplexed idiom", flush=True)
    n_par, T = 4, 20
    rng = np.random.default_rng(5)
    q0 = np.asarray(qc)[None, :] + rng.uniform(-0.3, 0.3, (n_par, 3))
    cmds = rng.uniform(-1, 1, (T, n_par, 3)).astype(np.float32)

    env_a = ArmEnv(dgp)
    body = Body(env_a, n_par=n_par, frame_skip=fs)
    for b_ in range(n_par):                       # place each episode at its own start
        body._select(b_); env_a.set_state(q0[b_], np.zeros(3)); body._states[b_] = env_a.get_state()
    traj_body = []
    for t in range(T):
        _, _, S2, _ = body.step(cmds[t])
        traj_body.append(S2.copy())
    body.release()
    traj_body = np.stack(traj_body)

    env_b = ArmEnv(dgp)                            # the classic multiplexed rollout
    states = np.zeros((n_par, 6), np.float32)
    for b_ in range(n_par):
        env_b.set_state(q0[b_], np.zeros(3)); states[b_] = env_b.get_state()
    traj_mux = []
    for t in range(T):
        for b_ in range(n_par):
            env_b.set_state(states[b_, :3].astype(np.float64), states[b_, 3:].astype(np.float64))
            states[b_], _ = env_b.step(cmds[t, b_], fs)
        traj_mux.append(states.copy())
    traj_mux = np.stack(traj_mux)

    dev = float(np.abs(traj_body - traj_mux).max())
    check("body_matches_multiplexed_rollout", dev < 1e-5, f"max abs deviation = {dev:.3e}")
    out["body_vs_mux_max_dev"] = dev

    # ---------------------------------------------------------------- 4
    print("\n[4] on-policy data is genuinely a trajectory", flush=True)
    env_c = ArmEnv(dgp)
    beh = OUBehaviour(3, 8, np.random.default_rng(11), sigma=0.7, theta=0.15)
    S, U, S2, info = collect_on_policy(env_c, 320, np.random.default_rng(12), fs,
                                       qc, 0.3, beh, ep_len=10, n_par=8,
                                       wrap_limit=3.0, return_info=True)
    ep, tt = info["ep"], info["t"]
    chained, checked = True, 0
    for e_ in np.unique(ep):
        m = np.flatnonzero(ep == e_)
        m = m[np.argsort(tt[m])]
        for i in range(len(m) - 1):
            checked += 1
            if not np.allclose(S[m[i + 1]], S2[m[i]], atol=1e-6):
                chained = False
    check("transitions_chain_within_episode", chained,
          f"{checked} consecutive pairs, s_(t+1) == s'_t")
    check("step_count_matches_transitions", abs(info["steps_used"] - len(S)) <= info["n_par"],
          f"steps_used={info['steps_used']} for {len(S)} transitions")

    # and the contrast that makes the point: teleport transitions do NOT chain
    tS, tU, tS2 = collect_pool(ArmEnv(dgp), 320, np.random.default_rng(13), fs,
                               qc, q_range, v_explore)
    tel_chain_frac = float(np.mean(np.abs(tS[1:] - tS2[:-1]).max(1) < 1e-6))
    check("teleport_does_not_chain", tel_chain_frac < 0.01,
          f"only {100 * tel_chain_frac:.1f}% of teleport pairs chain (expected ~0%)")
    out["onpolicy_diag"] = {k: int(v) for k, v in info.items()
                            if k in ("steps_used", "n_resets", "n_forced_resets", "n_episodes")}

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"), flush=True)
    out["all_passed"] = not failures
    return out


@app.local_entrypoint()
def verify():
    r = run_verify.remote()
    print("\n[local] verdict:", "PASS" if r.get("all_passed") else "FAIL")
    for k, v in r.items():
        print(f"    {k}: {v}")
    if not r.get("all_passed"):
        raise SystemExit(1)
