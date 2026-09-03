"""jacobian_teacher/defaults.py — configuration, importable WITHOUT numpy/torch/mujoco.

Modal local entrypoints run on the submitting machine, where `mjc/shared.py`'s standing rule
applies: "Keep top-level imports to modal/json/numpy only. `mujoco`, `torch`, and `matplotlib`
are imported INSIDE Modal function bodies so the app can be submitted from a machine that
doesn't have them installed." `core.py` imports torch at module scope, so the defaults an
entrypoint needs to BUILD a cfg live here, in a module with no third-party imports at all.
`core.py` re-exports both functions, so `from ...core import fork_defaults` still works inside
a Modal function body.
"""


def fork_defaults() -> dict:
    """EXACTLY `../ballistic/arm/arm_readapt.py::arm_readapt`'s signature defaults, in the
    `cfg` spelling that function builds. `phase_a.py::gate_fork_fidelity` re-derives these
    from that file's source and fails the run on any mismatch, so this dict may not be
    "improved" — a divergence here silently changes the substrate the comparison rests on."""
    return dict(
        b0=0.0, b1=6.0,
        milestones=[0, 400, 1000, 2500, 6000, 14000],
        n_links=3,
        link_lengths=[0.4, 0.4, 0.3],
        link_masses=[1.0, 1.0, 0.6],
        q_center=[0.4, 0.8, 0.6],
        joint_damping=0.5, gear=8.0, frame_skip=10,
        q_range=0.9, v_explore=8.0,
        fm_hidden=256, fm_layers=3, fm_lr=1e-3, fm_batch=512,
        fm_steps=6000, finetune_steps=1500, pool_n=14000, probe_n=1500,
        n_eval=48, plan_H=14,
        k_shoot=1024, cem_iters=8, cem_elite=32, cem_init_sigma=0.8, vel_pen=0.5,
        q_jit=0.25, v0_std=0.0,
        reach_amp=1.2, reach_lo=0.25, reach_hi=0.50, reach_tries=40,
        bc_tuples=6000, pol_hidden=256, pol_layers=3, pol_lr=1e-3, pol_steps=4000,
    )


def teacher_defaults() -> dict:
    """This node's own knobs. None of these exist in Cut 4c-arm."""
    return dict(
        # --- the executed-reach budget, identical for every teacher ---
        train_batch=128,           # reaches per policy update (dynamics_shift §5: rl_batch=128)
        reach_budget=20000,        # total EXECUTED reaches per teacher run
        eval_every=2000,           # fallback readout spacing when `eval_schedule` is empty
        # The adaptation curve's x-axis. Log-spaced because the arms live on different
        # timescales by hypothesis: an error-based teacher that converges in hundreds of
        # executed reaches and a reward-based one that is still climbing at twenty thousand
        # cannot both be resolved on one uniform grid.
        eval_schedule=[128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20000],
        # --- the shared stochastic policy (Garibbo: π(a|h)=N(μ_φ, σ²), σ fixed here) ---
        sigma=0.15,                # dynamics_shift §5's rl_sigma; SHARED by every teacher, so
                                   # the executed action distribution is identical across arms
                                   # and only the teaching signal differs
        respect_clip=False,        # zero the action gradient where μ+σε left [-1,1]. dynamics_
                                   # shift §5 ignores the clip in its log-prob; the default
                                   # matches it, and the same treatment applies to BOTH arms.
        # --- learning rates. Reported beside the measured gradient norms (phase_a gate 3) so
        #     the choice is auditable rather than assumed.
        lr_rbl=5e-4,               # dynamics_shift §5's rl_lr
        lr_ebl=5e-4,
        rbl_reward="continuous",   # or "binary" (Izawa & Shadmehr's reward zone)
        reward_radius=0.06,        # metres; the binary reward zone
        rbl_baseline_ema=0.9,      # dynamics_shift §5
        beta=1.0,                  # Eq. 2's mixing weight (1 = pure EBL, 0 = pure RBL)
        mix_scale=1.0,             # optional rescale of g_rbl before the β-sum; 1.0 = the
                                   # paper's raw sum. Phase A reports the norm ratio that
                                   # would make β an interpolating axis.
        sign_mask=None,            # dysmetria: a (2, n_act) list of ±1 applied to the FM's
                                   # believed ∂y_k/∂u_j. None = healthy cerebellum.
        # --- direction-accuracy instruments ---
        jac_n=16,                  # start states used for the composed endpoint Jacobian
        jac_probe_n=512,           # (s,u) pairs used for the one-step Jacobian
        fd_eps=1e-3,               # central-difference step on the command
        # --- generalization: train on a direction band, test on the rest ---
        train_dir_halfwidth=30.0,  # degrees; training goals lie within ±this of `train_dir`
        train_dir=0.0,             # degrees, measured in tip space from +x
        gen_offsets=[0.0, 30.0, 60.0, 90.0, 150.0, 180.0],   # test-band centres, degrees
        gen_halfwidth=15.0,
        band_oversample=40,        # candidate over-draw for a direction-banded sample
        # --- plateau variability: how many trailing eval points define "plateau" ---
        plateau_evals=4,
    )
