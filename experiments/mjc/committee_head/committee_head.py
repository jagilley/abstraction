"""committee_head -- a COMMITTEE of forward models as the READER, a trained HEAD as the JUDGE.

Spec: [`SPEC.md`](SPEC.md). Parent: [`../README.md`](../README.md) (mjc).
Substrate: E3 [`../on_policy/directed_on_policy/directed_on_policy.py`](../on_policy/directed_on_policy/directed_on_policy.py),
forked. Program: `ideas/two_timescale_value_loop.md`; the untested atom is
`ideas/activation_to_activation_forward.md` §"Learned valence tagging" (May) -- *a small auxiliary
classifier predicting whether a given error will prove reducible over subsequent training* -- which
has never been built. Every composition in the record so far was hand-specified.

THE QUESTION. Can a head trained on the SECOND-MOMENT SIGNATURE of a committee of small forward
models learn the three-way split -- mastered / novel-and-aleatoric / novel-and-learnable -- together
with relevance, and allocate a metered collection budget at least as well as the hand-composed
`lprog x visits` signal?

WHY A COMMITTEE, AND WHY A HEAD (what the record already established, so this cut does not re-ask it):
  * NO SINGLE FM READS THE MIDDLE LEG. The FM's first moment is the unbiased conditional mean
    (Wiener gain ~1); all the structure is in the error's SECOND moment (`REPRESENTATIONAL_DIVERGENCE`,
    15x across directions). A single FM's error magnitude IS the reader's output entropy (R^2 0.90)
    with revision R^2 0.0001 (`a2a/conditional_revision` Gate D) -- it cannot separate aleatoric from
    learnable. So the reader must pool over repeats: a committee.
  * EVERY POOLED READER THAT WORKED IS ALSO PARTIALLY BLIND. Committee disagreement rejects
    high-dimensional well-sampled noise but CHASES scarce low-dim noise (`curiosity_control` Finding 3)
    and is blind to staleness under a confident prior (`ballistic/directed` S1, `directed_readapt`) --
    which the RPF prior term is the known fix for. The benchmark net `b(s)-e` reads the noisy TV as
    statistically zero (`benchmark_vs_cost`) but is a GAIN, not an allocation score: at re-opening it
    is transiently, deeply NEGATIVE, i.e. actively repulsive if used as a score. A held-out trial of
    learning (the metered survey's `lprog`) inverts under data starvation (`metered_repair`).
    Each channel is individually wrong somewhere. That is precisely the case for LEARNING the
    composition off all of them at once instead of hand-writing one more product.
  * RELEVANCE IS NOT IN THE EPISTEMIC SIGNATURE (`curiosity_control` Finding 2). It has to come from
    rolling the plan through the FM (`fm_visits`, the p-tap). So the head predicts TWO things.

THE DESIGN, in one paragraph. E3's substrate verbatim (arm; one on-reach reducible target A; three
off-reach reducible distractors; two off-reach noise regions; reflecting OU walk on the curl gains so
nothing stays repaired; on-policy metered collection AND survey through `Body`; sighted grader =
region-A FM error; ballistic control as the cash-out). The single `net` becomes a K-member
random-prior (RPF) committee whose MEAN plans, so `fm_visits` and control are unchanged in kind.
Per region per round the loop reads a reward-free SIGNATURE off what it already computes -- committee
error `e`, disagreement `d`, benchmarked error `b(s)-e`, agency gap `g`, plan occupancy `v`, and the
last few rounds of each -- and a small shared-across-regions head maps that signature to
(predicted reducibility, predicted relevance). Allocation is their product, with an explicit no-op
floor (S2's `value-floor` lesson: a multiplicative signal needs one).

THE HEAD'S TARGETS ARE DENSE AND ALLOCATION-INDEPENDENT, which is the whole reason shape (1) is
supervised and not RL. The metered survey visits EVERY region EVERY round, so `lprog_j(t+1)` and
`visits_j(t+1)` are logged for every region regardless of where the budget went. The head at round t
predicts them from the round-t signature:

      y_red[j] = lprog_j(t+1)   -- "will collecting here pay", the valence tag
      y_rel[j] = visits_j(t+1)  -- "will I be here"

There is no bandit problem, no propensity correction, no credit assignment. `--head-target
realized_drop` switches y_red to the realized survey-error drop `max(e_j(t) - e_j(t+1), 0)` instead;
that reading of "the realized held-out error drop after collecting there" is the causally direct one
but it is only informative where budget was actually spent, so `lprog_next` is the default and the
other is logged alongside for both to be read off one run.

WHAT MAKES THIS A MEASUREMENT AND NOT AN INFERENCE FROM ALLOCATION. A head is trained on EVERY arm
(`--shadow-head`, default on) but only USED for allocation on the `head_*` arms. So split quality --
does the head predict ~0 reducibility on the noise regions and >0 on the reducible ones, per region
per round -- is measured under a fixed, known-good allocation (`value`) as well as under the head's
own, decoupling "did it learn the split" from "did its allocation help". Shadow heads cost GPU time
only: the metered `mon_steps`/`coll_steps` accounting is untouched.

THE ABLATION TABLE IS THE PRODUCT OF THIS CUT, since it says WHICH READER CARRIES THE SPLIT. Arms
`head_sup-nod` / `-nobme` / `-nov` / `-nohist` / `-lp` drop a channel or add the survey's own
counterfactual `lprog` as an input. The `-lp` arm is the honest ceiling: if the head only works when
handed `lprog`, it has added nothing beyond what the survey already measures.

FORK FIDELITY. At `--k 1 --rpf-beta 0` this file's `value` arm must reproduce E3's `value` arm
BIT-IDENTICALLY. Every added computation (disagreement, the benchmark net, the arity-1 net, the head)
uses its own dedicated RNG streams and never touches the shared ones; every committee stream is
`stream(k) = E3_stream + 7919*k`, so member 0 is E3's; `mem_forward` reduces to `net(Xn)` exactly at
beta=0 and no prior net is constructed. `check_fidelity.py` is the gate. (The one deliberate
non-fork: E3's `--render-rounds` render pack is dropped -- it is a communication artifact whose
replayer does not know about committees, and it is off by default, so no E3 result depends on it.)

Run:
    cd experiments/            # MODAL_PROFILE=chromatic
    modal run mjc/committee_head/committee_head.py::committee_head --quick               # smoke
    modal run mjc/committee_head/committee_head.py::committee_head --phase-a --tag pa0   # Phase A
    modal run --detach mjc/committee_head/committee_head.py::committee_head \
        --tag ladder_s0 --seed 0 --policies uniform,error-only,disagree-only,value,oracle
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# The Phase-B ladder. `head_*` arms allocate with the trained head; every other arm still TRAINS a
# shadow head (split quality is measured under its allocation too) but does not consume it.
_CORE = "uniform,error-only,disagree-only,value,oracle"
_HEADS = "head_sup,head_hetero"
_ABL = "head_sup-nod,head_sup-nobme,head_sup-nov,head_sup-nohist,head_sup-lp"
_ALL_POLICIES = _CORE + "," + _HEADS
# every policy that needs the counterfactual survey (as an allocation input, or as a head label)
_LP_POLICIES = ("value", "value-floor", "lprog-only", "error-only")

# per-member RNG stream offset: member 0 reuses E3's stream verbatim (the fidelity gate)
_MSTRIDE = 7919


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_committee_head(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.stats import rankdata, spearmanr

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from mjc.embodied import ReachBehaviour

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    B = cfg["n_eval"]; T = cfg["rounds"]
    NM = int(cfg["k_members"]); beta = float(cfg["rpf_beta"])
    LPK = NM if int(cfg["lp_k"]) <= 0 else min(int(cfg["lp_k"]), NM)
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    # ================================================================= #
    # GEOMETRY -- verbatim from E3. Regions in tip space, on/off-reach set by ACTUAL PATH
    # VISITATION (gate-occupancy over the FK'd joint-space reach sweep), all in the extended/tame
    # radius band, well-separated. E3's three geometry gotchas are load-bearing; see its README.
    # ================================================================= #
    def sample_reach_endpoints(n_samp, rng):
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (n_samp, n))
        t0 = fk(q0, Ls)
        eps = np.empty((n_samp, 2)); qg = np.empty((n_samp, n))
        lo, hi = cfg["reach_lo"], cfg["reach_hi"]
        for i in range(n_samp):
            best, best_pen = None, np.inf
            for _ in range(cfg["reach_tries"]):
                d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                cand = q0[i] + cfg["reach_amp"] * d
                dist = float(np.linalg.norm(fk(cand, Ls) - t0[i]))
                pen = max(0.0, lo - dist) + max(0.0, dist - hi)
                if pen < best_pen:
                    best, best_pen = cand, pen
                if pen == 0.0:
                    break
            eps[i] = fk(best, Ls); qg[i] = best
        return q0, qg, eps, t0.mean(0)

    geo_rng = np.random.default_rng(cfg["seed"] + 20)
    q0_cloud, qg_cloud, cloud, P0 = sample_reach_endpoints(cfg["geom_samples"], geo_rng)
    radius = np.linalg.norm(cloud, axis=1)
    ext = radius > cfg["rad_min"]
    ang_all = np.arctan2(cloud[:, 1] - P0[1], cloud[:, 0] - P0[0])
    hcnt, edges = np.histogram(ang_all[ext], bins=24, range=(-np.pi, np.pi))
    theta_pref = 0.5 * (edges[np.argmax(hcnt)] + edges[np.argmax(hcnt) + 1])
    dth = np.angle(np.exp(1j * (ang_all - theta_pref)))
    ev_mask = (np.abs(dth) < cfg["pref_width"]) & ext
    G = cloud[ev_mask].mean(0)

    ev_q0, ev_qg = q0_cloud[ev_mask], qg_cloud[ev_mask]
    sig2 = 2.0 * cfg["region_sigma"] ** 2

    def path_visitation(cands):
        occ = np.zeros(len(cands))
        for a in np.linspace(0.0, 1.0, cfg["vis_steps"]):
            tips = fk(ev_q0 * (1 - a) + ev_qg * a, Ls)
            dd = ((cands[:, None, 0] - tips[None, :, 0]) ** 2
                  + (cands[:, None, 1] - tips[None, :, 1]) ** 2)
            occ += np.exp(-dd / sig2).sum(1)
        return occ / max(occ.max(), 1e-9)

    vis = path_visitation(cloud)
    G = cloud[ev_mask].mean(0)

    def _nearest(pt):
        return float(np.linalg.norm(cloud - pt[None, :], axis=1).min())

    def _farthest_points(cands, m, seed):
        if len(cands) == 0 or m <= 0:
            return np.zeros((0, 2))
        rng = np.random.default_rng(seed)
        idx = [int(rng.integers(len(cands)))]
        while len(idx) < m and len(idx) < len(cands):
            d = np.min(np.stack([np.linalg.norm(cands - cands[i], axis=1) for i in idx]), 0)
            idx.append(int(np.argmax(d)))
        return cands[idx]

    ext_idx = np.flatnonzero(ext)
    A_c = cloud[ext_idx[np.argmax(vis[ext_idx])]]
    far_from_A = np.linalg.norm(cloud - A_c[None, :], axis=1) > cfg["min_sep"]
    REG = [dict(name="A", center=A_c, kind="curl", reducible=True, on_reach=True)]
    off_cands = cloud[ext & (vis < cfg["off_vis_max"]) & far_from_A]

    # ================================================================= #
    # OFF-REACH SELECTION -- "enterable but off the eval path" (opt-in, `--reachable-off`).
    #
    # THE DEFECT, measured on E3's OWN published ladder_s0 `value` arm:
    #
    #     region   eval-path visitation   collection in_share
    #     A                    0.833               0.462
    #     Boff1                0.084               0.413   <- what an off-reach distractor SHOULD be
    #     Boff2                0.010               0.000
    #     Boff3                0.000               0.000
    #     Doff1                0.000               0.000
    #     Doff2                0.072               0.000
    #
    # Four of six regions are NEVER ENTERED. Their metered survey therefore falls through the
    # `min_in` guard onto out-of-region transitions, so every per-region quantity for them --
    # error, disagreement, b-e, and the counterfactual `lprog` -- describes somewhere else.
    #
    # E3's headline survives this, which is why it was never caught there: `lprog x visits` is
    # carried by `visits`, computed by rolling the FM itself, and needs no in-region measurement.
    # A PER-REGION SIGNATURE does not survive it. The head would be asked to learn the three-way
    # split from features, and from a label, that are not measurements of the regions they are
    # attached to. Phase A#4's stop condition fires here for a reason that is about the substrate.
    #
    # This is `metered_repair` §7.2's named hole ("either place off-reach regions inside the
    # dynamic reach band by construction ... or accept that off-reach means unsurveyable"), and
    # `Boff1` is the proof the two properties are SEPARABLE rather than one property: off the eval
    # path AND enterable when aimed at. The cause is `_farthest_points`, which maximises mutual
    # separation and so selects *for* unenterability.
    #
    # THE FIX applies this node's own gotcha #1 discipline -- classify by ACTUAL visitation, never
    # by geometry -- to COLLECTION reachability, which the geometric selector never tested: reach
    # toward each candidate with an INDEPENDENT, FIELD-FREE instrument (its own plant, its own
    # normalisation, its own FM, none of them the agent's) and keep only candidates the body
    # actually enters. Deliberately self-contained rather than reusing the loop's machinery: that
    # machinery closes over a normalisation that does not exist yet at this point in the setup,
    # and rebuilding the ordering around it would put the E3-forked path -- the thing the fidelity
    # gate protects -- at risk to save fifty lines.
    #
    # OFF BY DEFAULT: with `reachable_off=False` nothing below runs and the geometry is E3's.
    # ================================================================= #
    def make_fixed_goal_sampler_geom(center, jit):
        c = np.asarray(center, np.float32)

        def sample(states, rng):
            mm = len(np.atleast_2d(np.asarray(states)))
            return (c[None, :] + rng.uniform(-jit, jit, (mm, 2))).astype(np.float32)
        return sample

    reach_frac_log = {}
    if cfg["reachable_off"] and len(off_cands) > 0:
        pr_env = ArmEnv(dict(n_links=n, link_lengths=cfg["link_lengths"][:n],
                             link_masses=cfg["link_masses"][:n],
                             joint_damping=cfg["joint_damping"], gear=cfg["gear"]))
        pS, pU, pS2 = collect_pool(pr_env, cfg["reach_pool_n"],
                                   np.random.default_rng(cfg["seed"] + 24), fs, qc,
                                   cfg["q_range"], cfg["v_explore"])
        pX = np.concatenate([pS, pU], 1)
        pn = {k_: torch.tensor(vv, device=device) for k_, vv in dict(
            mx=pX.mean(0), sx=pX.std(0) + 1e-6,
            my=(pS2 - pS).mean(0), sy=(pS2 - pS).std(0) + 1e-6).items()}
        ph = cfg["reach_hidden"]
        pnet = nn.Sequential(nn.Linear(SD + AD, ph), nn.SiLU(), nn.Linear(ph, ph), nn.SiLU(),
                             nn.Linear(ph, SD)).to(device)
        popt = torch.optim.Adam(pnet.parameters(), lr=cfg["fm_lr"])
        pXt = torch.tensor(((pX - pX.mean(0)) / (pX.std(0) + 1e-6)).astype(np.float32), device=device)
        pYt = torch.tensor((((pS2 - pS) - (pS2 - pS).mean(0)) /
                            ((pS2 - pS).std(0) + 1e-6)).astype(np.float32), device=device)
        prng = np.random.default_rng(cfg["seed"] + 25)
        hub = nn.HuberLoss(delta=1.0)
        for _ in range(cfg["reach_fm_steps"]):
            bi = torch.tensor(prng.integers(0, len(pXt), size=min(512, len(pXt))), device=device)
            popt.zero_grad(); hub(pnet(pXt[bi]), pYt[bi]).backward(); popt.step()
        pnet.eval()

        def pr_plan(states, goals):
            Bn = states.shape[0]; ks = cfg["collect_k_shoot"]
            mu = np.zeros((Bn, H, AD), np.float32)
            sg = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(np.asarray(goals, np.float32), device=device).repeat_interleave(ks, 0)
            s0 = torch.tensor(np.asarray(states, np.float32), device=device).repeat_interleave(ks, 0)
            for _ in range(cfg["collect_cem_iters"]):
                e_ = prng.standard_normal((Bn, ks, H, AD)).astype(np.float32)
                sq = np.clip(mu[:, None] + sg[:, None] * e_, -1, 1)
                with torch.no_grad():
                    st = s0.clone()
                    sqt = torch.tensor(sq.reshape(Bn * ks, H, AD), device=device)
                    cost = torch.zeros(Bn * ks, device=device)
                    for hh in range(H):
                        xx = torch.cat([st, sqt[:, hh, :]], 1)
                        st = st + (pnet((xx - pn["mx"]) / pn["sx"]) * pn["sy"] + pn["my"])
                        cost = cost + (fk_torch(st[:, :n]) - g_t).norm(dim=1)
                    ei = torch.topk(-cost.reshape(Bn, ks), cfg["cem_elite"], dim=1).indices.cpu().numpy()
                el = np.take_along_axis(sq, ei[:, :, None, None], axis=1)
                mu = el.mean(1); sg = el.std(1) + 1e-3
            return mu.astype(np.float32)

        def reach_frac(c, i):
            """fraction of transitions the body actually spends INSIDE the region when aimed at
            it -- the same quantity the loop later reports as `in_share`, measured before the
            region exists rather than discovered afterwards."""
            rr = np.random.default_rng(cfg["seed"] + 26 + i)
            beh = ReachBehaviour(pr_plan, np.zeros((1, 2), np.float32), rr,
                                 sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
            gs = make_fixed_goal_sampler_geom(c, cfg["goal_jit"])
            # n_par=1 so `reach_probe_n` transitions are that many SEQUENTIAL steps, i.e. whole
            # reaches. At n_par=16 this probe measured "enterable within the first
            # ceil(reach_probe_n/16) steps of a reach" -- the same travel-vs-arrival bias it is
            # supposed to be screening for, which would silently select regions near the start
            # posture and call them reachable.
            S_, _, _ = collect_pool(pr_env, cfg["reach_probe_n"], rr, fs, qc, cfg["op_q_range"], None,
                                    collection_mode="on_policy", behaviour=beh, goal_sampler=gs,
                                    ep_len=cfg["ep_len"], n_par=1, v0_std=cfg["v0_std"],
                                    wrap_limit=cfg["wrap_limit"])
            tips = fk(S_[:, :n].astype(np.float64), Ls)
            return float((np.linalg.norm(tips - np.asarray(c), axis=1)
                          < cfg["region_k"] * cfg["region_sigma"]).mean())

        # shortlist by separation first (cheap), then keep the ones the body can actually enter
        shortlist = _farthest_points(off_cands, cfg["reach_shortlist"], cfg["seed"] + 30)
        fr = np.array([reach_frac(c, i) for i, c in enumerate(shortlist)])
        reach_frac_log = {f"cand{i}": {"center": c.tolist(), "reach_frac": float(fr[i])}
                          for i, c in enumerate(shortlist)}
        keep = shortlist[fr >= cfg["off_reach_min"]]
        print(f"[geom] reachable-off probe: {len(shortlist)} candidates, "
              f"reach_frac=[" + " ".join(f"{x:.2f}" for x in np.sort(fr)[::-1]) + "]  "
              f"-> {len(keep)} pass >= {cfg['off_reach_min']}", flush=True)
        need = cfg["n_off_red"] + cfg["n_off_noise"]
        if len(keep) >= need:
            off_cands = keep
        else:
            # not enough enterable candidates: fall back to the best-ranked ones rather than
            # silently shrinking the region set, and say so -- a thinner ladder is a result about
            # the substrate, but an undeclared one is a bug.
            off_cands = shortlist[np.argsort(-fr)][:need]
            print(f"[geom] WARNING only {len(keep)} candidates cleared the reachability floor; "
                  f"taking the {need} best-ranked. Off-reach distractors may still be "
                  f"unsurveyable -- check `n_in` in the logs before reading any per-region "
                  f"signature.", flush=True)
    off_pts = _farthest_points(off_cands, cfg["n_off_red"] + cfg["n_off_noise"], cfg["seed"] + 30)
    for i, pt in enumerate(off_pts):
        if i < cfg["n_off_red"]:
            REG.append(dict(name=f"Boff{i + 1}", center=pt, kind="curl", reducible=True, on_reach=False))
        else:
            REG.append(dict(name=f"Doff{i - cfg['n_off_red'] + 1}", center=pt, kind="noise",
                            reducible=False, on_reach=False))
    on_cands = cloud[ext & (vis > cfg["on_vis_min"]) & far_from_A]
    for i, pt in enumerate(_farthest_points(on_cands, cfg["n_on_noise"], cfg["seed"] + 31)):
        REG.append(dict(name=f"Con{i + 1}", center=pt, kind="noise", reducible=False, on_reach=True))
    K = len(REG)
    for r in REG:
        r["sigma"] = cfg["region_sigma"]
    names = [r["name"] for r in REG]
    reducible = [j for j in range(K) if REG[j]["reducible"]]
    noise_idx = [j for j in range(K) if not REG[j]["reducible"]]
    a_idx_ = names.index("A") if "A" in names else 0        # the sighted grader's region
    reg_vis = path_visitation(np.array([np.asarray(r["center"]) for r in REG]))
    print(f"[geom] P0={P0.round(3).tolist()} (r={np.linalg.norm(P0):.2f}) "
          f"A(most-visited)={A_c.round(3).tolist()} (r={np.linalg.norm(A_c):.2f}) "
          f"K={K} regions ({len(reducible)} reducible)", flush=True)
    for j, r in enumerate(REG):
        c = np.asarray(r["center"])
        print(f"[geom]   {r['name']}: center={c.round(3).tolist()} r={np.linalg.norm(c):.2f} "
              f"visitation={reg_vis[j]:.2f} "
              f"kind={r['kind']} reducible={r['reducible']} on_reach={r['on_reach']} "
              f"nearest-cloud={_nearest(c):.3f}m", flush=True)

    def gate_np(tips, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((tips[..., 0] - c[0]) ** 2 + (tips[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    def in_region(states, j):
        tips = fk(np.asarray(states)[:, :n].astype(np.float64), Ls)
        return np.linalg.norm(tips - REG[j]["center"], axis=1) < cfg["region_k"] * REG[j]["sigma"]

    # ================================================================= #
    # ENV -- verbatim from E3.
    # ================================================================= #
    def env_dgp(b_state):
        curls, noises = [], []
        for j, r in enumerate(REG):
            if r["kind"] == "curl":
                curls.append({"b": float(b_state[j]), "center": tuple(np.asarray(r["center"]).tolist()),
                              "sigma": r["sigma"]})
            else:
                noises.append({"amp": float(cfg["noise_amp"]), "center": tuple(np.asarray(r["center"]).tolist()),
                               "sigma": r["sigma"]})
        dgp = dict(n_links=n, link_lengths=cfg["link_lengths"][:n], link_masses=cfg["link_masses"][:n],
                   joint_damping=cfg["joint_damping"], gear=cfg["gear"], noise_seed=cfg["seed"] + 999)
        if curls:
            dgp["curl_fields"] = curls
        if noises:
            dgp["noise_fields"] = noises
        return dgp

    def make_env(b_state):
        return ArmEnv(env_dgp(b_state))

    # ================================================================= #
    # THE READER -- a RANDOM-PRIOR COMMITTEE of forward models f(s,u) -> Δs.
    #
    # The member idiom is `curiosity_control.py`'s `build_member` (Osband et al. 2018): a trainable
    # net plus a FROZEN random prior, prediction = net + beta*prior. Where data covers a region the
    # net compensates the prior and members AGREE; where data is sparse the distinct frozen priors
    # make them DISAGREE. That is the documented fix for the confident-prior staleness blindness a
    # plain warm-started ensemble suffers (`directed_readapt`) -- the failure mode in which every
    # member confidently agrees on a stale prediction, so disagreement collapses to zero exactly
    # where the drifted frontier now needs re-learning.
    #
    # This is also the microzone reading in `beliefs/trees/cerebellum_and_cognitive_architecture.md`:
    # many bands, one teacher. The committee MEAN plans, so control and `fm_visits` are unchanged in
    # kind from E3 -- only the reader's second moment is new.
    #
    # FIDELITY: at NM=1, beta=0 no prior net is built and `mem_forward` is literally `net(Xn)`.
    # ================================================================= #
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(SD + AD, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, SD)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def _mseed(k):
        """member k's init seed. k=0 is E3's `cfg["seed"]+40` exactly."""
        return cfg["seed"] + 40 + _MSTRIDE * k

    def build_member(k):
        net = _mlp(_mseed(k))
        if beta <= 0.0:
            return {"net": net, "prior": None}
        prior = _mlp(_mseed(k) + 99991)
        for p in prior.parameters():
            p.requires_grad_(False)
        return {"net": net, "prior": prior}

    def mem_forward(m, Xn):
        return m["net"](Xn) if m["prior"] is None else m["net"](Xn) + beta * m["prior"](Xn)

    huber = nn.HuberLoss(delta=1.0)

    def train_member(m, opt, S, U, S2, steps, brng):
        """E3's `train_steps`, on one member. Each member draws its minibatch indices WITH
        REPLACEMENT from its own stream, which is the bootstrap; the RPF term supplies the
        off-data disagreement the bootstrap alone does not (`curiosity_control` §build_member)."""
        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); m["net"].train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(mem_forward(m, Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(m["net"].parameters(), 1.0)
            opt.step()
        m["net"].eval()

    def train_committee(ms, opts, S, U, S2, steps, base_seed):
        for k, (m, o) in enumerate(zip(ms, opts)):
            train_member(m, o, S, U, S2, steps, np.random.default_rng(base_seed + _MSTRIDE * k))

    def ens_delta_t(ms, x):
        """committee-MEAN Δs prediction, in torch, for a raw (state,cmd) batch. At NM=1 this is
        E3's `net((x-mx)/sx)*sy+my` expression character for character."""
        xn = (x - norm["mx"]) / norm["sx"]
        if len(ms) == 1:
            return mem_forward(ms[0], xn) * norm["sy"] + norm["my"]
        return torch.stack([mem_forward(m, xn) for m in ms], 0).mean(0) * norm["sy"] + norm["my"]

    def ens_stack(ms, S, U):
        """(NM, N, SD) per-member Δs predictions in raw space -- the second moment lives here."""
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
            xn = (X - norm["mx"]) / norm["sx"]
            return torch.stack([mem_forward(m, xn) * norm["sy"] + norm["my"] for m in ms], 0)

    def fm_delta(ms, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32), device=device)
            return ens_delta_t(ms, X).cpu().numpy()

    def fm_err(ms, S, U, S2):
        if len(S) == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(ms, S, U) - (S2 - S).astype(np.float32), axis=1).mean())

    def fm_err_per(ms, S, U, S2):
        """per-sample committee-mean error -- the stream the benchmark net b(s) is fitted to."""
        if len(S) == 0:
            return np.zeros(0, np.float32)
        return np.linalg.norm(fm_delta(ms, S, U) - (S2 - S).astype(np.float32),
                              axis=1).astype(np.float32)

    def fm_disagree(ms, S, U):
        """predictive variance across members, mean over state dims (`curiosity_control`
        `score_disagree`). This is the reader's SECOND MOMENT and the only channel that can, in
        principle, tell 'nobody knows' from 'nothing to know'."""
        if len(ms) < 2 or len(S) == 0:
            return float("nan")
        return float(ens_stack(ms, S, U).var(0).mean().item())

    def _ensemble_cos(residuals):
        """Input-centered pairwise cosine between members' residuals on the same batch
        (`a2a_forward/confabulation/confabulation.py`). HIGH => the residual is determined by the
        input, i.e. an FM-invariant gap every member misses identically (aleatoric, or a shared
        computational hole); LOW => it is determined by the particular member, i.e. genuine
        epistemic spread. The committee-health guard: a committee whose members leave the SAME
        residual is not a committee, it is one model in K copies."""
        import torch.nn.functional as F
        if len(residuals) < 2 or residuals[0].shape[0] < 2:
            return float("nan")
        cent = [r - r.mean(dim=0, keepdim=True) for r in residuals]
        sims = []
        for i in range(len(cent)):
            for j in range(i + 1, len(cent)):
                sims.append(float(F.cosine_similarity(cent[i], cent[j], dim=-1).mean()))
        return float(sum(sims) / len(sims)) if sims else float("nan")

    def ens_cos(ms, S, U, S2):
        if len(ms) < 2 or len(S) < 2:
            return float("nan")
        true = torch.tensor((S2 - S).astype(np.float32), device=device)
        return _ensemble_cos([p - true for p in ens_stack(ms, S, U)])

    def teleport_pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs, qc, cfg["q_range"], cfg["v_explore"])

    b0_state = np.array([cfg["b1"] if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    env_drift0 = make_env(b0_state)
    nS, nU, nS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k_: torch.tensor(v, device=device) for k_, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ================================================================= #
    # TWO AUXILIARY READERS that make the signature the full δ_perf object rather than a
    # convenience subset. Neither ever sees a reward or an oracle.
    #
    #  b(s)  -- the CONTEXT-CONDITIONAL BENCHMARK NET (`plasticity_gain.py` make_bench/bench_step):
    #           a state-only regressor trained online on the committee's OWN error stream, so
    #           `b(s) - e` is "am I doing better than I have recently done HERE". This is the one
    #           channel the record shows reads a noisy TV as statistically zero once habituated
    #           (`benchmark_vs_cost`, +0.0001 +- 0.0024) -- and also the one that is transiently
    #           DEEPLY NEGATIVE at re-opening, i.e. actively repulsive as a raw allocation score.
    #           A learned head can use it as a feature with a sign and a lag; a hand-composed
    #           product cannot. That asymmetry is a large part of why this cut exists.
    #
    #  FM1(s) -- the ARITY-1 net, giving the AGENCY GAP g = ||FM2_mean(s,u) - FM1(s)||
    #           (`agency_gate.py`): how much of the next state my own command explains. Here every
    #           survey transition is self-produced, so g is expected near-inert -- carried anyway
    #           so the signature is the full δ_perf = (b-e)*sigma((g-g0)/theta) object and a
    #           playback control stays available without re-deriving the substrate.
    # ================================================================= #
    def _mlp_io(seed, din, dout, h, L):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, dout)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def make_bench(seed):
        return _mlp_io(seed, SD, 1, cfg["bench_hidden"], cfg["bench_layers"])

    def bench_pred(bnet, S):
        with torch.no_grad():
            X = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:SD]) / norm["sx"][:SD]
            return bnet(X).squeeze(-1).cpu().numpy().astype(np.float32)

    def bench_step(bnet, bopt, S, e):
        X = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:SD]) / norm["sx"][:SD]
        t = torch.tensor(e, device=device, dtype=torch.float32)
        bnet.train(); bopt.zero_grad()
        huber(bnet(X).squeeze(-1), t).backward(); bopt.step(); bnet.eval()

    def make_fm1(seed):
        return _mlp_io(seed, SD, SD, cfg["fm_hidden"], cfg["fm_layers"])

    def fm1_delta(net1, S):
        with torch.no_grad():
            X = (torch.tensor(S, device=device, dtype=torch.float32) - norm["mx"][:SD]) / norm["sx"][:SD]
            return (net1(X) * norm["sy"] + norm["my"]).cpu().numpy()

    def train_fm1(net1, opt1, S, S2, steps, brng):
        X = (torch.tensor(S.astype(np.float32), device=device) - norm["mx"][:SD]) / norm["sx"][:SD]
        Y = ((torch.tensor((S2 - S).astype(np.float32), device=device) - norm["my"]) / norm["sy"])
        bs = min(cfg["fm_batch"], len(S)); net1.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt1.zero_grad(); huber(net1(X[idx]), Y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net1.parameters(), 1.0)
            opt1.step()
        net1.eval()

    def agency_gap(ms, net1, S, U):
        if len(S) == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(ms, S, U) - fm1_delta(net1, S), axis=1).mean())

    # ================================================================= #
    # CEM planner over the COMMITTEE MEAN + eval rollout -- E3's, with `net` -> `ms`.
    # ================================================================= #
    def make_plan_fn(ms, k_shoot, cem_iters, rng):
        def plan(states, goals):
            Bn = states.shape[0]
            mu = np.zeros((Bn, H, AD), np.float32)
            sig = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(np.asarray(goals, np.float32), device=device).repeat_interleave(k_shoot, 0)
            s0 = torch.tensor(np.asarray(states, np.float32), device=device).repeat_interleave(k_shoot, 0)
            for _ in range(cem_iters):
                e = rng.standard_normal((Bn, k_shoot, H, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * k_shoot, H, AD), device=device)
                    cost = torch.zeros(Bn * k_shoot, device=device)
                    for h in range(H):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + ens_delta_t(ms, x)
                        cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, k_shoot), cfg["cem_elite"], dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)
        return plan

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        _, _, eps, _ = sample_reach_endpoints(cfg["n_eval"] * 6, rng)
        a = np.arctan2(eps[:, 1] - P0[1], eps[:, 0] - P0[0])
        d = np.abs(np.angle(np.exp(1j * (a - theta_pref))))
        pick = np.argsort(d)[:B]
        goals = eps[pick].astype(np.float32)
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (B, n))
        starts = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (B, n))], 1).astype(np.float32)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every, cenv):
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                cenv.set_state(states[b, :n].astype(np.float64), states[b, n:].astype(np.float64))
                s2, _ = cenv.step(acts[b], fs)
                states[b] = s2
        tips = fk(states[:, :n].astype(np.float64), Ls).astype(np.float32)
        return float(np.median(np.linalg.norm(tips - ev_goals, axis=1)))

    def fm_visits(ms):
        """Region occupancy of the ballistic eval plan rolled through the COMMITTEE MEAN (no env,
        no budget) -- the relevance signal stays purely internal, and it is the one thing the
        epistemic signature demonstrably does not contain (`curiosity_control` Finding 2)."""
        plan = make_plan_fn(ms, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 7001))(ev_starts, ev_goals)
        v = np.zeros(K)
        with torch.no_grad():
            s = torch.tensor(ev_starts, device=device)
            for h in range(H):
                tips = fk_torch(s[:, :n]).cpu().numpy()
                for j in range(K):
                    v[j] += float(gate_np(tips, j).sum())
                x = torch.cat([s, torch.tensor(plan[:, h, :], device=device)], 1)
                s = s + ens_delta_t(ms, x)
        return v / max(v.sum(), 1e-9)

    # ================================================================= #
    # ON-POLICY reach-toward-a-region collection -- verbatim (the metered acquisition primitive;
    # one env.step per transition, so `need` == steps charged).
    # ================================================================= #
    def make_fixed_goal_sampler(center, jit):
        c = np.asarray(center, np.float32)

        def sample(states, rng):
            m = len(np.atleast_2d(np.asarray(states)))
            return (c[None, :] + rng.uniform(-jit, jit, (m, 2))).astype(np.float32)
        return sample

    def collect_toward(cenv, center, need, ms, rng_buf, k, n_par=None):
        npar = cfg["n_par"] if n_par is None else int(n_par)
        plan_fn = make_plan_fn(ms, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                               np.random.default_rng(cfg["seed"] + 8000 + k))
        beh = ReachBehaviour(plan_fn, np.zeros((npar, 2), np.float32), rng_buf,
                             sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
        gs = make_fixed_goal_sampler(center, cfg["goal_jit"])
        S, U, S2, info = collect_pool(cenv, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                      collection_mode="on_policy", behaviour=beh, goal_sampler=gs,
                                      ep_len=cfg["ep_len"], n_par=npar, v0_std=cfg["v0_std"],
                                      wrap_limit=cfg["wrap_limit"], return_info=True)
        return S, U, S2, info

    # ================================================================= #
    # EPISODE-DENOMINATED COLLECTION (`--episode-budget`).
    #
    # THE DEFECT. `collect_on_policy` fills `need` transitions by stepping `n_par` parallel
    # episodes FORWARD IN TIME, so it samples timesteps 0 .. ceil(need/n_par) of an `ep_len`
    # reach. At the shipped n_par=16 NOTHING completes a reach: the metered survey sees the first
    # 4 steps of 14, and a region allotted 120/6 = 20 transitions sees the first 2 -- the body is
    # still at the start posture, nowhere near the region it is aiming at. So a SMALL allocation
    # structurally cannot sample the region it names, in-region YIELD becomes a function of how
    # concentrated the allocation happens to be, and the ladder ranks arms by yield rather than by
    # targeting. Measured on the transition-denominated ladder: corr(A-err, in-region transitions)
    # = -0.940, and the all-or-nothing `burst` timing control came first. Charged steps were not
    # matched either (uniform 6*ceil(20/16)*16 = 192 vs burst 1*ceil(120/16)*16 = 128, exactly as
    # predicted). This is inherited from E3 and applies to its published ladder too.
    #
    # THE FIX. The atomic unit of embodied collection is an EPISODE, not a transition: you cannot
    # half-do a reach, and half a reach is the travel rather than the destination. So denominate
    # the budget in complete reaches. With n_par=1, `need = n_eps*ep_len` runs exactly `n_eps`
    # whole episodes, so
    #   * charged steps = budget_eps*ep_len, IDENTICAL for every policy by construction;
    #   * every transition comes from a completed reach, so the travel:arrival ratio is constant
    #     and in-region yield depends only on WHERE you aimed -- the quantity being measured.
    # `--gate-invariance` is the check that this holds; see the gate below.
    # ================================================================= #
    EPB = bool(cfg["episode_budget"])
    EP = int(cfg["ep_len"])

    def collect_eps(cenv, center, n_eps, ms, rng_buf, k):
        """exactly `n_eps` COMPLETE reaches toward `center` (n_par=1 => need is a whole
        number of episodes, so collect_on_policy never stops mid-reach)."""
        return collect_toward(cenv, center, int(n_eps) * EP, ms, rng_buf, k, n_par=1)

    # ================================================================= #
    # BASE (stale) COMMITTEE -- trained off-budget on the PRE-drift world. Plus the pre-calibrated
    # benchmark net and arity-1 net, which enter the loop already fitted to the clean world exactly
    # as `plasticity_gain` pretrains b(s) ("the bird enters calibrated").
    # ================================================================= #
    b_pre = np.zeros(K, float)
    env_pre = make_env(b_pre)
    S0_, U0_, S20_ = teleport_pool(env_pre, cfg["pool_n"], cfg["seed"] + 10)
    base_members = [build_member(k) for k in range(NM)]
    train_committee(base_members, [torch.optim.Adam(m["net"].parameters(), lr=cfg["fm_lr"])
                                   for m in base_members],
                    S0_, U0_, S20_, cfg["fm_steps"], cfg["seed"] + 300)
    base_state = [copy.deepcopy(m["net"].state_dict()) for m in base_members]

    fm1_base = make_fm1(cfg["seed"] + 41)
    train_fm1(fm1_base, torch.optim.Adam(fm1_base.parameters(), lr=cfg["fm_lr"]),
              S0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 3100))
    fm1_base_state = copy.deepcopy(fm1_base.state_dict())

    bench_base = make_bench(cfg["seed"] + 42)
    bopt_base = torch.optim.Adam(bench_base.parameters(), lr=cfg["bench_lr"])
    e_pool = fm_err_per(base_members, S0_, U0_, S20_)
    brng0 = np.random.default_rng(cfg["seed"] + 3200)
    for _ in range(cfg["bench_pretrain_steps"]):
        idx = brng0.integers(0, len(S0_), size=min(cfg["fm_batch"], len(S0_)))
        bench_step(bench_base, bopt_base, S0_[idx], e_pool[idx])
    bench_base_state = copy.deepcopy(bench_base.state_dict())
    print(f"[pretrain] committee NM={NM} rpf_beta={beta} trained on pre-drift pool; "
          f"clean-world committee err={float(e_pool.mean()):.4f} "
          f"b(s) MAE={float(np.abs(bench_pred(bench_base, S0_[:4000]) - e_pool[:4000]).mean()):.4f}",
          flush=True)

    gmax = np.max(np.stack([gate_np(fk(S0_[:, :n].astype(np.float64), Ls), j) for j in range(K)]), 0)
    oidx = np.flatnonzero(gmax < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = S0_[rp], U0_[rp], S20_[rp]
    print(f"[pretrain] replay={len(rp)} out-of-region transitions", flush=True)

    # ================================================================= #
    # CONTINUOUS DRIFT -- verbatim. Precomputed ONCE and SHARED across every policy (and, since it
    # is a pure function of the seed, across JOBS too -- so the ladder may be split over several
    # Modal runs at the same seed and still be a controlled comparison).
    # ================================================================= #
    drift_rng = np.random.default_rng(cfg["seed"] + 77)
    b_ref = np.array([cfg["b1"] if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    b_traj = np.zeros((T, K))
    b_cur = b_ref.copy()
    schedule = {}
    for t in range(T):
        b_traj[t] = b_cur.copy()
        if cfg["drift_mode"] == "ou":
            for j in reducible:
                nb = b_cur[j] + cfg["ou_sigma"] * drift_rng.standard_normal()
                if nb < cfg["b_lo"]:
                    nb = 2 * cfg["b_lo"] - nb
                if nb > cfg["b_hi"]:
                    nb = 2 * cfg["b_hi"] - nb
                b_cur[j] = float(np.clip(nb, cfg["b_lo"], cfg["b_hi"]))
        else:
            if t > 0 and t % cfg["drift_every"] == 0:
                cyc = (t // cfg["drift_every"] - 1)
                cycle = cfg["drift_cycle"] or reducible
                j = cycle[cyc % len(cycle)]
                b_cur[j] = float(drift_rng.choice([-1.0, 1.0]) * cfg["b1"])
                schedule[t] = (j, b_cur[j])
    print(f"[setup] drift_mode={cfg['drift_mode']} b_traj[A] over rounds: "
          + " ".join(f"{b_traj[t, 0]:.1f}" for t in range(min(T, 12))), flush=True)

    def _norm_w(v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return v / v.sum() if v.sum() > 1e-12 else np.full(K, 1.0 / K)

    # ================================================================= #
    # THE JUDGE -- a small head, SHARED ACROSS REGIONS, mapping one region's signature to
    # (predicted reducibility, predicted relevance).
    #
    # Shared-across-regions is a design commitment, not an implementation convenience: the head
    # maps a FEATURE VECTOR to a value, so it cannot memorise "region 3 is the noisy one" and
    # nothing in it scales with K. What it can learn is a rule about signatures -- which is the
    # only thing that could transfer to Phase C's positions, or anywhere else.
    #
    # CHANNELS (per region j, round t; all reward-free, all already computed by the loop):
    #   e    committee-mean survey error, in-region
    #   d    committee disagreement (predictive variance across members)
    #   bme  b(s) - e, the benchmarked error
    #   g    agency gap ||FM2_mean - FM1||
    #   v    plan occupancy (fm_visits)
    #   a    (history only) share of last round's budget spent here
    #   lp   (only on the `-lp` arm) the survey's own counterfactual learning progress
    # plus `hist_len` lags of each, so the head can form a derivative if it needs one without being
    # handed the derivative's known blindness at re-opening.
    # ================================================================= #
    CH = ["e", "d", "bme", "g", "v"]

    def head_spec(pname):
        """`head_sup-nod` -> drop d; `-nobme`, `-nov`, `-nohist`, `-lp` (add survey lprog)."""
        parts = pname.split("-")
        drop, add_lp, hl = set(), False, int(cfg["hist_len"])
        for p in parts[1:]:
            if p == "nod":
                drop.add("d")
            elif p == "nobme":
                drop.add("bme")
            elif p == "nov":
                drop.add("v")
            elif p == "nog":
                drop.add("g")
            elif p == "nohist":
                hl = 0
            elif p == "lp":
                add_lp = True
            else:
                raise ValueError(f"unknown head ablation {p!r} in {pname!r}")
        return drop, add_lp, hl

    def feat_dim(drop, add_lp, hl):
        cur = len([c for c in CH if c not in drop]) + (1 if add_lp else 0)
        return cur + hl * (cur + 1)          # +1: the lagged allocation share

    def build_feat(sig_hist, alloc_hist, j, drop, add_lp, hl):
        """sig_hist: list (oldest..newest) of dicts of per-region arrays; alloc_hist likewise."""
        chans = [c for c in CH if c not in drop] + (["lp"] if add_lp else [])
        cur = sig_hist[-1]
        f = [float(cur[c][j]) for c in chans]
        for L in range(1, hl + 1):
            past = sig_hist[-1 - L] if len(sig_hist) > L else None
            f += [float(past[c][j]) if past is not None else 0.0 for c in chans]
            pa = alloc_hist[-L] if len(alloc_hist) >= L else None
            f.append(float(pa[j]) if pa is not None else 0.0)
        # a channel can be legitimately undefined (disagreement at NM=1; ensemble cosine on a
        # thin in-region subset). Zero-fill rather than propagate: a missing channel should mute
        # itself, not poison the whole row.
        return np.nan_to_num(np.array(f, np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def make_head(seed, d_in):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h = cfg["head_hidden"]
        # head_hidden=0 => LINEAR. The MLP is ~1850 params fitted on K*T ~ 180 rows, which is
        # absurdly overparameterised; the linear head is the control that says whether capacity
        # (rather than the target) is what is wrong.
        net = (nn.Linear(d_in, 2) if h <= 0 else
               nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                             nn.Linear(h, 2)))
        if isinstance(net, nn.Linear):
            nn.init.kaiming_uniform_(net.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(net.bias)
            return net.to(device)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def head_fit(head, hopt, Xs, Ys, groups=None):
        """Full-batch refit on the whole (tiny: K*T rows) labelled set each round. Full-batch is
        deliberate -- it consumes no RNG, so a shadow head cannot perturb the loop it is watching,
        which is what makes the fidelity gate meaningful with `--shadow-head 1`."""
        X = np.asarray(Xs, np.float32); Y = np.asarray(Ys, np.float32)
        mx, sx = X.mean(0), X.std(0) + 1e-6
        my, sy = Y.mean(0), Y.std(0) + 1e-6
        Xt = torch.tensor((X - mx) / sx, device=device, dtype=torch.float32)
        Yt = torch.tensor((Y - my) / sy, device=device, dtype=torch.float32)
        gi = None
        if cfg["head_loss"] == "rank" and groups is not None:
            gi = [np.flatnonzero(np.asarray(groups) == g_) for g_ in sorted(set(groups))]
            gi = [torch.tensor(ix, device=device) for ix in gi if len(ix) > 1]
        head.train()
        for _ in range(cfg["head_steps"]):
            hopt.zero_grad()
            P = head(Xt)
            if gi:
                # RANKNET on the reducibility output. The allocation consumes only the ORDERING of
                # regions within a round, never the magnitudes, so optimise the ordering directly:
                # for every within-round pair the target ranks i above j, push s_i above s_j. This
                # is far more robust to a noisy target than regressing its value, which is the
                # diagnosed failure (lprog's round-to-round autocorrelation is +0.007, i.e. the
                # per-round realisation is noise around a per-region mean).
                loss = 0.0
                for ix in gi:
                    si = P[ix, 0]; yi = Yt[ix, 0]
                    dl = si[:, None] - si[None, :]
                    dy = yi[:, None] - yi[None, :]
                    m = dy > 0
                    if m.any():
                        loss = loss + nn.functional.softplus(-dl[m]).mean()
                loss = loss / max(len(gi), 1) + huber(P[:, 1], Yt[:, 1])   # relevance stays regressed
            else:
                loss = huber(P, Yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            hopt.step()
        head.eval()
        return (mx, sx, my, sy)

    def head_predict(head, stats, F):
        mx, sx, my, sy = stats
        with torch.no_grad():
            Xt = torch.tensor((np.asarray(F, np.float32) - mx) / sx, device=device, dtype=torch.float32)
            P = head(Xt).cpu().numpy() * sy + my
        return P[:, 0], P[:, 1]                       # (pred reducibility, pred relevance)

    # ================================================================= #
    # one policy's full T-round loop
    # ================================================================= #
    steady_share = {}                                 # filled by `value`, consumed by burst/steady

    def run_policy(pname):
        is_head = pname.startswith("head_")
        drop, add_lp, hl = head_spec(pname) if is_head else (set(), False, int(cfg["hist_len"]))
        use_head = is_head and pname != "head_rl"
        members = [build_member(k) for k in range(NM)]
        for m, st in zip(members, base_state):
            m["net"].load_state_dict(st)
        opts = [torch.optim.Adam(m["net"].parameters(), lr=cfg["finetune_lr"]) for m in members]
        net1 = make_fm1(cfg["seed"] + 41); net1.load_state_dict(fm1_base_state)
        opt1 = torch.optim.Adam(net1.parameters(), lr=cfg["finetune_lr"])
        bnet = make_bench(cfg["seed"] + 42); bnet.load_state_dict(bench_base_state)
        bopt = torch.optim.Adam(bnet.parameters(), lr=cfg["bench_lr"])
        brng = np.random.default_rng(cfg["seed"] + 3300)

        head = make_head(cfg["seed"] + 50, feat_dim(drop, add_lp, hl)) if (is_head or cfg["shadow_head"]) else None
        hopt = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"],
                                weight_decay=cfg["head_wd"]) if head is not None else None
        hX, hYred, hYrel, hstats = [], [], [], None
        hRound = []                                    # round each row came from (rank grouping)
        feat_hist = []                                 # features per round, for lagged labels
        rl_base = None                                # EWMA baseline for the head_rl arm
        rl_prev = None                                # last round's sighted-teacher level
        rl_feat = None                                # the features the last action was taken on
        rl_share = None                               # the action itself (realised budget shares)

        bufs = {j: None for j in range(K)}
        sig_hist, alloc_hist = [], []
        hist = []
        # every arm runs the counterfactual survey: `value`/`lprog-only` consume it as their
        # allocation signal, every other arm needs it as the head's LABEL. It costs GPU time, not
        # `Body` steps, so the metered monitor:collect accounting is unchanged either way.
        uses_lp = (pname in _LP_POLICIES) or is_head or cfg["shadow_head"]
        for t in range(T):
            b_state = b_traj[t]
            if t in schedule:
                bufs[schedule[t][0]] = None
            env = make_env(b_state)
            mon_steps = 0; coll_steps = 0

            # ---------- METERED on-policy survey: reach each region, read the SIGNATURE ----------
            per_err = np.zeros(K); lprog = np.zeros(K)
            disag = np.full(K, np.nan); bme = np.full(K, np.nan); agap = np.full(K, np.nan)
            ceil_e = np.full(K, np.nan)
            ecos = np.full(K, np.nan); n_in = np.zeros(K, int)
            bench_S, bench_e = [], []
            cur = [copy.deepcopy(m["net"].state_dict()) for m in members] if uses_lp else None
            for j in range(K):
                srng = np.random.default_rng(cfg["seed"] + 9000 + 41 * t + j)
                if EPB:
                    # the SURVEY matters more than the collection here: the signature is read off
                    # it, so a survey that never arrives is a signature of somewhere else.
                    mS, mU, mS2, minfo = collect_eps(env, REG[j]["center"], cfg["mon_eps"],
                                                     members, srng, 100 + j)
                else:
                    mS, mU, mS2, minfo = collect_toward(env, REG[j]["center"], cfg["mon_n"],
                                                        members, srng, 100 + j)
                mon_steps += int(minfo["steps_used"])
                m_in = in_region(mS, j)
                if m_in.sum() >= cfg["min_in"]:
                    iS, iU, iS2 = mS[m_in], mU[m_in], mS2[m_in]
                else:
                    iS, iU, iS2 = mS, mU, mS2
                n_in[j] = int(m_in.sum())
                per_err[j] = fm_err(members, iS, iU, iS2)
                # the matched-FM error on the SAME in-region survey subset: region j's own
                # irreducible floor, measured by the same instrument on the same data. The
                # REDUCIBLE part of the current error is the excess over it. Logged for every arm
                # as a readout; consumed only by `oracle --oracle-mode excess`.
                ceil_e[j] = fm_err(ceil_member, iS, iU, iS2)
                # --- the second-moment signature, all read off the SAME in-region subset ---
                e_per = fm_err_per(members, iS, iU, iS2)
                disag[j] = fm_disagree(members, iS, iU)
                bme[j] = float((bench_pred(bnet, iS) - e_per).mean())   # PREDICT, then update below
                agap[j] = agency_gap(members, net1, iS, iU)
                ecos[j] = ens_cos(members, iS, iU, iS2)
                bench_S.append(iS); bench_e.append(e_per)
                if uses_lp:
                    h = len(iS) // 2
                    if h >= 2:
                        # the counterfactual fit is the loop's most expensive per-round item
                        # (NM probe committees x K regions). `lp_k` runs it on a SUB-committee;
                        # `eb` is then read off the same sub-committee so the two halves of the
                        # difference are the same estimator. lp_k >= NM (the default) is the
                        # full committee, and at NM=1 it is E3's single-net probe exactly.
                        lp_ms = members[:LPK]
                        eb = fm_err(lp_ms, iS[h:], iU[h:], iS2[h:])
                        probes = [{"net": _mlp(_mseed(k)), "prior": lp_ms[k]["prior"]} for k in range(LPK)]
                        for k in range(LPK):
                            probes[k]["net"].load_state_dict(cur[k])
                        train_committee(probes, [torch.optim.Adam(p["net"].parameters(), lr=cfg["finetune_lr"])
                                                 for p in probes],
                                        np.concatenate([rS, iS[:h]]), np.concatenate([rU, iU[:h]]),
                                        np.concatenate([rS2, iS2[:h]]), cfg["lp_steps"],
                                        cfg["seed"] + 9500 + 41 * t + j)
                        ea = fm_err(probes, iS[h:], iU[h:], iS2[h:])
                        lprog[j] = max(eb - ea, 0.0)
                        del probes
            visits = fm_visits(members)
            var_replay = fm_disagree(members, rS, rU)      # committee health: in-data variance
            stale_mask = np.array([1.0 if (REG[j]["reducible"] and per_err[j] > cfg["err_floor"])
                                   else 0.0 for j in range(K)])
            sig = {"e": per_err, "d": disag, "bme": bme, "g": agap, "v": visits, "lp": lprog}
            sig_hist.append(sig)

            # ---------- the head: label yesterday's features with today's outcome, then refit ----
            # The label for round t-1 is available NOW and for EVERY region, because the survey is
            # exhaustive. That density is what makes the valence tag supervised rather than a bandit.
            # LABELLING. The target is formed at a lag of LW rounds. For `lprog_smooth` the label
            # is the MEAN of lprog over the next LW rounds, which is the fix for the diagnosed
            # failure: a single round's lprog has autocorrelation +0.007 (white noise about a
            # per-region mean), so no reader can predict it, while averaging it over ~8 rounds
            # restores a real relationship to the signature (rho(d, target): -0.06 at W=1 ->
            # +0.25 at W=8, measured on logged data before this was built).
            LW = int(cfg["head_window"]) if cfg["head_target"] == "lprog_smooth" else 1
            if head is not None and t >= LW and len(feat_hist) > t - LW:
                t0 = t - LW
                for j in range(K):
                    if cfg["head_target"] == "lprog_smooth":
                        y_red = float(np.nanmean([sig_hist[r]["lp"][j] for r in range(t0 + 1, t + 1)]))
                    elif cfg["head_target"] == "realized_drop":
                        y_red = float(max(sig_hist[t0]["e"][j] - sig_hist[t0 + 1]["e"][j], 0.0))
                    else:
                        y_red = float(sig_hist[t0 + 1]["lp"][j])
                    if not np.isfinite(y_red):
                        continue
                    hX.append(feat_hist[t0][j]); hYred.append(y_red)
                    hYrel.append(float(sig_hist[t0 + 1]["v"][j])); hRound.append(t0)
                if len(hX) >= cfg["head_min_rows"]:
                    hstats = head_fit(head, hopt, hX, np.stack([hYred, hYrel], 1), groups=hRound)
            feats = ([build_feat(sig_hist, alloc_hist, j, drop, add_lp, hl) for j in range(K)]
                     if head is not None else None)
            feat_hist.append(feats)
            p_red = p_rel = None
            if head is not None and hstats is not None and np.isfinite(np.stack(feats)).all():
                p_red, p_rel = head_predict(head, hstats, feats)

            # ---------- the policy: WHERE to spend this round's collection budget ----------
            head_ready = (p_red is not None) and (t >= cfg["head_warmup"] + (
                int(cfg["head_window"]) - 1 if cfg["head_target"] == "lprog_smooth" else 0))
            if pname == "uniform":
                w = None
            elif pname == "oracle":
                if cfg["oracle_mode"] == "excess":
                    # E3's target `stale_mask * per_err * visits` is ERROR-SCALE SENSITIVE. Once
                    # regions differ in intrinsic difficulty -- here off-reach regions carry ~10x
                    # region A's error (1.133 vs 0.105) -- raw per_err swamps `visits` and the
                    # oracle spends 2/3 of its budget off-reach, landing barely ahead of uniform
                    # (spread 0.018) and destroying the gap-closed normalisation. What an oracle
                    # should chase is the REDUCIBLE part: the excess over the region's own
                    # matched-FM floor, which is privileged information an oracle is entitled to.
                    tgt = stale_mask * np.maximum(per_err - np.nan_to_num(ceil_e, nan=0.0), 0.0) * visits
                else:
                    tgt = stale_mask * per_err * visits
                w = _norm_w(tgt) if tgt.sum() > 1e-9 else None
            elif pname == "value":
                tv = lprog * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None
            elif pname == "lprog-only":
                w = _norm_w(lprog) if lprog.sum() > 1e-12 else None
            elif pname == "visits-only":
                w = _norm_w(visits)
            elif pname == "error-only":
                w = _norm_w(per_err)
            elif pname == "disagree-only":
                # the honest HOMOGENEOUS baseline for the heterogeneous-grader arm, and the
                # `curiosity_control` Finding-3 test: does plain disagreement chase these two
                # scarce, low-dimensional noise regions the way it did there?
                w = _norm_w(np.nan_to_num(disag, nan=0.0))
            elif pname == "steady":
                w = _norm_w(steady_share["value"])
            elif pname == "burst":
                w = np.zeros(K); w[burst_pick[t]] = 1.0
            elif pname == "head_rl":
                # ---- REINFORCE update for the action taken LAST round, now that its outcome is in.
                # The only signal is the SIGHTED teacher -- the round-over-round drop in region-A FM
                # error (`drift_value_loop` Cut 3: control is a near-blind grader). The record says
                # REINFORCE is noisy on a shallow bowl, so: an EWMA baseline, a scale-normalised
                # advantage, and this arm is meant to be read only AFTER shape (1) has been.
                if rl_feat is not None and rl_share is not None and hstats is not None:
                    R = float(rl_prev - per_err[a_idx_])
                    rl_base = R if rl_base is None else (1 - cfg["rl_beta"]) * rl_base + cfg["rl_beta"] * R
                    adv = (R - rl_base) / (abs(rl_base) + 1e-6)
                    mx_, sx_, _, _ = hstats
                    Xt = torch.tensor((np.asarray(rl_feat, np.float32) - mx_) / sx_,
                                      device=device, dtype=torch.float32)
                    logp = torch.log_softmax(head(Xt)[:, 0] / cfg["rl_temp"], 0)
                    loss = -adv * (torch.tensor(rl_share, device=device, dtype=torch.float32) * logp).sum()
                    hopt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); hopt.step()
                rl_prev = float(per_err[a_idx_])
                if head_ready:
                    with torch.no_grad():
                        sc = torch.tensor(np.asarray(p_red, np.float32) / cfg["rl_temp"])
                        w = torch.softmax(sc, 0).numpy().astype(float)
                    rl_feat = feats
                else:
                    w = None; rl_feat = None
            elif is_head:
                if not head_ready:
                    w = None                                  # cold start: the head knows nothing
                elif pname.startswith("head_hetero"):
                    # heterogeneous_graders §9: allocate where the DENSE grader (committee
                    # disagreement) and the EVALUATIVE grader (the head, trained on realised
                    # outcomes) disagree -- the named fix for staleness-blindness, since that is
                    # exactly the cell where the head says "this will pay" and the confident
                    # committee says "nothing to see". Gated by predicted relevance so it does not
                    # simply become a frontier-tracker that pays nothing (Finding 2).
                    rs = _norm_w(np.clip(p_red, 0, None)); ds = _norm_w(np.nan_to_num(disag, nan=0.0))
                    tv = np.maximum(rs - ds, 0.0) * np.clip(p_rel, 0, None)
                    w = _norm_w(tv) if tv.max() >= cfg["head_floor"] else None
                else:
                    tv = np.clip(p_red, 0, None) * np.clip(p_rel, 0, None)
                    w = _norm_w(tv) if tv.max() >= cfg["head_floor"] else None   # the S2 no-op floor
            else:
                raise ValueError(pname)

            rng_c = np.random.default_rng(cfg["seed"] + 11000 + 97 * t)
            in_share = np.full(K, np.nan)
            B_ = cfg["budget_eps"] if EPB else cfg["budget"]      # episodes, or transitions
            if w is None:
                counts = np.full(K, B_ // K, int); counts[0] += B_ - counts.sum()
            else:
                counts = np.floor(w * B_).astype(int)
                counts[int(np.argmax(w))] += B_ - counts.sum()
            alloc = counts.astype(float)
            for j in range(K):
                if counts[j] <= 0:
                    continue
                if EPB:
                    cS, cU, cS2, cinfo = collect_eps(env, REG[j]["center"], int(counts[j]),
                                                     members, rng_c, 200 + j)
                else:
                    cS, cU, cS2, cinfo = collect_toward(env, REG[j]["center"], int(counts[j]),
                                                        members, rng_c, 200 + j)
                coll_steps += int(cinfo["steps_used"])
                in_share[j] = float(in_region(cS, j).mean())
                d = (cS, cU, cS2)
                bufs[j] = d if bufs[j] is None else tuple(
                    np.concatenate([bufs[j][i], d[i]])[-cfg["buf_cap"]:] for i in range(3))

            # ---------- fine-tune the committee (+ b(s), + FM1) on everything believed valid ----
            parts = [(rS, rU, rS2)] + [b for b in bufs.values() if b is not None]
            aS = np.concatenate([p[0] for p in parts]); aU = np.concatenate([p[1] for p in parts])
            aS2 = np.concatenate([p[2] for p in parts])
            train_committee(members, opts, aS, aU, aS2, cfg["finetune_steps"], cfg["seed"] + 12000 + t)
            train_fm1(net1, opt1, aS, aS2, cfg["fm1_steps"], np.random.default_rng(cfg["seed"] + 13000 + t))
            bS = np.concatenate(bench_S); be = np.concatenate(bench_e)
            for _ in range(cfg["bench_steps"]):                 # b(s) tracks the committee's OWN error
                idx = brng.integers(0, len(bS), size=min(cfg["fm_batch"], len(bS)))
                bench_step(bnet, bopt, bS[idx], be[idx])

            # ---------- grade (off-budget, the experimenter's fixed instrument) ----------
            snap = [{"net": copy.deepcopy(m["net"]), "prior": m["prior"]} for m in members]
            reg_err = [fm_err(snap, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                       for j in range(K)]
            bal = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                       np.random.default_rng(cfg["seed"] + 7001)), H, env)
            rec = {"round": t, "alloc": alloc.tolist(), "in_share": in_share.tolist(),
                   "per_err_mon": per_err.tolist(), "reg_err": reg_err, "visits": visits.tolist(),
                   "lprog": lprog.tolist(), "stale_mask": stale_mask.tolist(), "b_state": b_state.tolist(),
                   "ballistic_cem": bal, "mon_steps": mon_steps, "coll_steps": coll_steps,
                   "disagree": disag.tolist(), "bme": bme.tolist(), "agap": agap.tolist(),
                   "ens_cos": ecos.tolist(), "n_in": n_in.tolist(), "var_replay": var_replay,
                   "ceil_e": ceil_e.tolist(),
                   "excess": np.maximum(per_err - np.nan_to_num(ceil_e, nan=0.0), 0.0).tolist(),
                   "head_used": bool(head_ready and use_head)}
            if p_red is not None:
                rec["pred_red"] = np.asarray(p_red).tolist(); rec["pred_rel"] = np.asarray(p_rel).tolist()
            if (t % cfg["reactive_every"] == 0) or (t == T - 1):
                rec["reactive"] = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                                       np.random.default_rng(cfg["seed"] + 7000)), 1, env)
            hist.append(rec)
            alloc_hist.append(alloc / max(alloc.sum(), 1.0))
            if pname == "head_rl":
                rl_share = (alloc / max(alloc.sum(), 1.0)) if rl_feat is not None else None

            astr = "/".join(str(int(x)) for x in alloc)
            hd = ""
            if p_red is not None:
                hd = ("  pred_red=[" + " ".join(f"{x:.3f}" for x in p_red) + "]")
            print(f"[{pname:>16s} r{t:02d}] alloc={astr:>16s} ball={bal:.4f}"
                  + (f" reac={rec['reactive']:.4f}" if "reactive" in rec else "")
                  + f"  mon/coll={mon_steps}/{coll_steps}"
                  + "  regERR=[" + " ".join(f"{e:.3f}" for e in reg_err) + "]"
                  + "  d=[" + " ".join(f"{x:.1e}" for x in disag) + "]" + hd, flush=True)
        if pname == "value":
            steady_share["value"] = np.mean(np.stack([np.asarray(r["alloc"]) / max(sum(r["alloc"]), 1.0)
                                                      for r in hist]), 0)
        return hist

    # ---- fixed per-region PROBE sets + matched-FM ceiling (off-budget grader) -- verbatim from E3,
    # with the reference/ceiling FMs kept as SINGLE nets: they are the experimenter's instrument,
    # not the agent's reader, and holding them fixed keeps the grader comparable to E3's.
    print("[probe] training ref FM (drifted world) + building per-region grader probes ...", flush=True)
    # NOTE these are SINGLE nets with no prior term even when the agent's committee has one. They
    # are the experimenter's instrument, not the agent's reader, so holding them at E3's exact form
    # keeps the grader and the ceiling comparable across the two nodes.
    ref_member = [{"net": _mlp(cfg["seed"] + 43), "prior": None}]
    dS, dU, dS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 400)
    train_member(ref_member[0], torch.optim.Adam(ref_member[0]["net"].parameters(), lr=cfg["fm_lr"]),
                 dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 401))
    PB = []
    probeS, probeU, probeS2, _ = collect_toward(env_drift0, G, cfg["probe_pool_n"], ref_member,
                                                np.random.default_rng(cfg["seed"] + 610), 900)
    for j in range(1, K):
        oS, oU, oS2, _ = collect_toward(env_drift0, REG[j]["center"], cfg["probe_pool_n"] // 2, ref_member,
                                        np.random.default_rng(cfg["seed"] + 901 + j), 901 + j)
        probeS = np.concatenate([probeS, oS]); probeU = np.concatenate([probeU, oU])
        probeS2 = np.concatenate([probeS2, oS2])
    for j in range(K):
        m = in_region(probeS, j)
        PB.append((probeS[m], probeU[m], probeS2[m]))
        print(f"[probe]   region {names[j]}: {int(m.sum())} probe transitions", flush=True)
    ceilS, ceilU, ceilS2, _ = collect_toward(env_drift0, G, cfg["ceil_pool_n"], ref_member,
                                             np.random.default_rng(cfg["seed"] + 620), 910)
    if cfg["ceil_all_regions"]:
        # E3 trains the matched ceiling on reaches toward G only, so it is "undertrained by
        # construction" off-reach and its off-reach ceilings are caveated (and often nan). That is
        # tolerable when the ceiling only normalises region A, but NOT when the oracle consumes a
        # per-region floor: an undertrained floor makes a hard region look reducible. So gather
        # competent data in EVERY region and let the ceiling be competent everywhere.
        for j in range(K):
            gS, gU, gS2, _ = collect_toward(env_drift0, REG[j]["center"], cfg["ceil_pool_n"] // 2,
                                            ref_member, np.random.default_rng(cfg["seed"] + 630 + j),
                                            920 + j)
            ceilS = np.concatenate([ceilS, gS]); ceilU = np.concatenate([ceilU, gU])
            ceilS2 = np.concatenate([ceilS2, gS2])
        print(f"[probe] ceiling FM trained on ALL-region reaches ({len(ceilS)} transitions)",
              flush=True)
    ceil_member = [{"net": _mlp(cfg["seed"] + 44), "prior": None}]
    train_member(ceil_member[0], torch.optim.Adam(ceil_member[0]["net"].parameters(), lr=cfg["fm_lr"]),
                 ceilS, ceilU, ceilS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 421))
    ceil_err = [fm_err(ceil_member, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                for j in range(K)]
    print("[probe] matched-ceiling per-region err: "
          + " ".join(f"{names[j]}={ceil_err[j]:.3f}" for j in range(K)), flush=True)

    # the scheduled-burst control (metered_repair §7.1) needs `value`'s realised mean share, so
    # `value` is forced to run first when `burst`/`steady` are requested (see the entrypoint).
    burst_pick = None
    policies = list(cfg["policies"])
    # ================================================================= #
    # PHASE A GATE -- ALLOCATION-SIZE INVARIANCE (`--gate-invariance`).
    #
    # The one check that says whether allocation and YIELD are separable, i.e. whether the ladder
    # measures where a policy aimed or merely how concentrated it was. Reach toward ONE region at
    # several allocation sizes and measure the in-region share. Under episode denomination it must
    # be FLAT: every size is a whole number of completed reaches, so the travel:arrival ratio does
    # not depend on the size. Under transition denomination it RISES with size, which is the bug.
    # Both are measured here so the gate documents the defect it fixes rather than asserting it.
    # Off-budget: a fresh env and dedicated RNG, nothing is trained, no loop state is touched.
    # ================================================================= #
    invariance = {}
    if cfg["gate_invariance"]:
        genv = make_env(b_traj[0])
        gms = [build_member(k) for k in range(NM)]
        for m, st in zip(gms, base_state):
            m["net"].load_state_dict(st)
        print(f"\n=== Phase A gate: allocation-size invariance (region {names[a_idx_]}) ===",
              flush=True)
        print(f"    {'mode':>12s} {'size':>6s} {'in-region share':>15s}", flush=True)
        for mode in ("episode", "transition"):
            row = {}
            allx, ally = [], []
            for i, size in enumerate(cfg["gate_sizes"]):
                # REPEATS. In-region share at size 1 is estimated from ep_len transitions, at
                # size 8 from 8*ep_len -- comparing single point estimates across sizes confuses
                # sampling noise with a size effect (the first cut of this gate did exactly that
                # and read SIZE-DEPENDENT for both modes). Independent repeats per size make the
                # comparison a trend test with error bars instead.
                sh = []
                for rep in range(cfg["gate_reps"]):
                    grng = np.random.default_rng(cfg["seed"] + 31000 + 131 * i + 17 * rep
                                                 + (7 if mode == "episode" else 0))
                    if mode == "episode":
                        gS, _, _, gi = collect_eps(genv, REG[a_idx_]["center"], size, gms, grng,
                                                   300 + 10 * i + rep)
                    else:
                        gS, _, _, gi = collect_toward(genv, REG[a_idx_]["center"], size * EP, gms,
                                                      grng, 400 + 10 * i + rep, n_par=cfg["n_par"])
                    if len(gS):
                        sh.append(float(in_region(gS, a_idx_).mean()))
                        allx.append(size); ally.append(sh[-1])
                spt = gi["steps_used"] / max(len(gS), 1)
                m = float(np.mean(sh)) if sh else float("nan")
                se = float(np.std(sh, ddof=1) / max(len(sh) ** 0.5, 1)) if len(sh) > 1 else 0.0
                row[str(size)] = {"in_share": m, "sem": se, "steps_per_txn": float(spt),
                                  "reps": len(sh)}
                print(f"    {mode:>12s} {size:6d} {m:9.3f} +- {se:.3f}   steps/txn {spt:.3f}",
                      flush=True)
            v = [row[str(x)]["in_share"] for x in cfg["gate_sizes"]]
            st_ = [row[str(x)]["steps_per_txn"] for x in cfg["gate_sizes"]]
            rho = float(spearmanr(allx, ally).statistic) if len(set(allx)) > 1 else float("nan")
            row["spread"] = float(np.nanmax(v) - np.nanmin(v))
            row["steps_per_txn_spread"] = float(np.nanmax(st_) - np.nanmin(st_))
            row["rho_size_yield"] = rho
            invariance[mode] = row
            # the verdict is the TREND, not the spread: a monotone rise of in-region share with
            # allocation size is the confound; scatter around a flat line is sampling noise.
            verdict = ("FLAT (allocation and yield separable)" if abs(rho) < cfg["gate_tol"]
                       else "SIZE-DEPENDENT (yield rises with allocation)" if rho > 0
                       else "SIZE-DEPENDENT (inverted)")
            print(f"    {mode:>12s}  rho(size, in-share) = {rho:+.3f}   spread {row['spread']:.3f}"
                  f"   steps/txn spread {row['steps_per_txn_spread']:.3f}   <- {verdict}",
                  flush=True)
        del gms

    results = {}
    for pname in policies:
        if pname == "burst":
            # matched-mean-share bursty schedule: the SAME long-run share as `value`, delivered
            # all-or-nothing one region per round. If bursty beats steady here, the ladder has been
            # measuring TIMING rather than reducibility -- which would re-scope E3's headline too.
            p = steady_share["value"]
            cnt = np.floor(p * T).astype(int)
            cnt[int(np.argmax(p))] += T - cnt.sum()
            seq = np.concatenate([np.full(c, j) for j, c in enumerate(cnt) if c > 0])
            burst_pick = np.random.default_rng(cfg["seed"] + 555).permutation(seq)
        print(f"\n=== policy: {pname} ===", flush=True)
        results[pname] = run_policy(pname)

    # ================================================================= #
    # summary -- E3's, plus the split-quality and committee-health readouts this cut is for
    # ================================================================= #
    def auc(pname, key):
        v = [r[key] for r in results[pname] if key in r and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    def reg_auc(pname, j):
        v = [r["reg_err"][j] for r in results[pname] if r["reg_err"][j] == r["reg_err"][j]]
        return float(np.mean(v)) if v else float("nan")

    a_idx = names.index("A") if "A" in names else 0
    print("\n=== summary: mean over rounds (lower=better) ===", flush=True)
    print(f"    {'policy':>16s}  {'ballistic':>9s} {'reactive':>9s} {'A-err':>7s} "
          + " ".join(f"{nm}-err" for nm in names), flush=True)
    summary = {}
    for pname in policies:
        row = {"ballistic_auc": auc(pname, "ballistic_cem"), "reactive_auc": auc(pname, "reactive"),
               "regA_err_auc": reg_auc(pname, a_idx),
               "reg_err_auc": [reg_auc(pname, j) for j in range(K)],
               "mon_steps_total": int(sum(r["mon_steps"] for r in results[pname])),
               "coll_steps_total": int(sum(r["coll_steps"] for r in results[pname])),
               "final_reg_err": results[pname][-1]["reg_err"]}
        row["monitor_collect_ratio"] = row["mon_steps_total"] / max(row["coll_steps_total"], 1)
        sh = np.mean(np.stack([np.asarray(r["alloc"]) / max(sum(r["alloc"]), 1.0)
                               for r in results[pname]]), 0)
        row["mean_share"] = sh.tolist()
        row["share_noise"] = float(sh[noise_idx].sum()) if noise_idx else 0.0
        row["share_A"] = float(sh[a_idx])
        row["share_offred"] = float(sum(sh[j] for j in reducible if j != a_idx))
        summary[pname] = row
        print(f"    {pname:>16s}  {row['ballistic_auc']:9.4f} {row['reactive_auc']:9.4f} "
              f"{row['regA_err_auc']:7.4f} " + " ".join(f"{e:6.3f}" for e in row["reg_err_auc"]), flush=True)

    mr = np.nanmean([summary[p]["monitor_collect_ratio"] for p in policies])
    print(f"\n[anti-subsidy] mean monitor:collect step ratio = {mr:.2f}  (must stay O(1))", flush=True)

    print("\n=== budget shares (A / off-reach reducible / noise) ===", flush=True)
    for p in policies:
        print(f"    {p:>16s}  A={summary[p]['share_A']:.2f} offred={summary[p]['share_offred']:.2f} "
              f"noise={summary[p]['share_noise']:.2f}", flush=True)

    off_idx = [j for j in range(K) if not REG[j]["on_reach"] and REG[j]["reducible"]]
    if off_idx:
        def off_err_auc(pname):
            return float(np.nanmean([reg_auc(pname, j) for j in off_idx]))
        print("\n=== relevance test (E3's retracted-then-recovered claim) ===", flush=True)
        for p in policies:
            summary[p]["off_err_auc"] = off_err_auc(p)
            print(f"    {p:>16s}: A-err={summary[p]['regA_err_auc']:.4f}  "
                  f"off-err={off_err_auc(p):.4f}  ballistic={summary[p]['ballistic_auc']:.4f}", flush=True)

    if "uniform" in results and "oracle" in results:
        for key, lbl in (("ballistic_cem", "ballistic"), ("regA_err_auc", "A-FM-err")):
            u = auc("uniform", "ballistic_cem") if key == "ballistic_cem" else summary["uniform"]["regA_err_auc"]
            o = auc("oracle", "ballistic_cem") if key == "ballistic_cem" else summary["oracle"]["regA_err_auc"]
            den = u - o
            print(f"\n  [{lbl}] uniform={u:.4f} oracle={o:.4f} spread={den:+.4f}", flush=True)
            for pname in policies:
                val = auc(pname, "ballistic_cem") if key == "ballistic_cem" else summary[pname]["regA_err_auc"]
                g = (u - val) / den if abs(den) > 1e-9 else float("nan")
                summary[pname][lbl + "_gap_closed"] = g
                print(f"      {pname:>16s}  gap-closed={g:+.2f}", flush=True)

    if "A" in names:
        cA = ceil_err[a_idx]
        for pname in policies:
            st = results[pname][0]["reg_err"][a_idx]
            den = st - cA
            rc = [((st - r["reg_err"][a_idx]) / den if abs(den) > 1e-9 else float("nan"))
                  for r in results[pname]]
            summary[pname]["regA_recovery_auc"] = float(np.nanmean(rc))
        print(f"\n=== region-A ceiling-normalised recovery (ceil={ceil_err[a_idx]:.3f}, "
              f"higher=better) ===", flush=True)
        for pname in policies:
            print(f"    {pname:>16s}  {summary[pname]['regA_recovery_auc']:+.2f}", flush=True)

    # ---------------------------------------------------------------- #
    # PHASE A #4 -- SIGNATURE SEPARABILITY, read-only. Before asking whether a head helps, ask
    # whether anything in the signature separates the classes at all. AUROC of each raw channel,
    # pooled over rounds, for (a) reducible vs noise and (b) A vs everything else. 0.5 = blind.
    # ---------------------------------------------------------------- #
    def _auroc(pos, neg):
        pos = np.asarray(pos, float)[np.isfinite(pos)]
        neg = np.asarray(neg, float)[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def _chan(pname, key, js):
        return np.concatenate([[r[key][j] for j in js] for r in results[pname]])

    sep = {}
    # `excess*` is PRIVILEGED (it consumes the matched-FM ceiling) and is NOT a head input --
    # CH above is the head's channel list and does not contain it. It is reported here as the
    # reference line: what a reader WITH the answer achieves, i.e. the ceiling on reducibility
    # discrimination that the reward-free channels are being asked to approach.
    chan_keys = [("per_err_mon", "e"), ("disagree", "d"), ("bme", "b-e"), ("agap", "g"),
                 ("visits", "v"), ("lprog", "lprog"), ("excess", "excess*")]
    print("\n=== Phase A#4: signature separability, read-only (AUROC; 0.5 = blind) ===", flush=True)
    print(f"    {'policy':>16s} {'channel':>7s}  {'red-vs-noise':>12s}  {'A-vs-rest':>10s}", flush=True)
    for pname in policies:
        sep[pname] = {}
        for key, lbl in chan_keys:
            a1 = _auroc(_chan(pname, key, reducible), _chan(pname, key, noise_idx)) if noise_idx else float("nan")
            a2 = _auroc(_chan(pname, key, [a_idx]), _chan(pname, key, [j for j in range(K) if j != a_idx]))
            sep[pname][lbl] = {"red_vs_noise": a1, "A_vs_rest": a2}
            print(f"    {pname:>16s} {lbl:>7s}  {a1:12.3f}  {a2:10.3f}", flush=True)

    # ---------------------------------------------------------------- #
    # SPLIT QUALITY -- the measurement this cut exists for, and the reason a shadow head runs on
    # every arm: "did the head learn the split" is then separable from "did its allocation help".
    #   * predicted reducibility vs REALISED next-round lprog (corr, per region per round)
    #   * predicted relevance vs REALISED next-round visits
    #   * AUROC of the head's own predicted reducibility for reducible-vs-noise
    # ---------------------------------------------------------------- #
    split = {}
    print("\n=== split quality: head prediction vs realised outcome (lag-1) ===", flush=True)
    print(f"    {'policy':>16s}  {'r(red,lprog+1)':>14s} {'r(rel,visits+1)':>15s} "
          f"{'AUROC red:r-vs-n':>16s}", flush=True)
    for pname in policies:
        rows = results[pname]
        pr, tr, pl, tl, cls = [], [], [], [], []
        for i in range(len(rows) - 1):
            if "pred_red" not in rows[i]:
                continue
            for j in range(K):
                pr.append(rows[i]["pred_red"][j]); tr.append(rows[i + 1]["lprog"][j])
                pl.append(rows[i]["pred_rel"][j]); tl.append(rows[i + 1]["visits"][j])
                cls.append(1 if REG[j]["reducible"] else 0)
        if len(pr) < 4:
            split[pname] = None
            print(f"    {pname:>16s}  {'--':>14s} {'--':>15s} {'--':>16s}", flush=True)
            continue
        cls = np.array(cls); pr_ = np.array(pr)
        c1 = float(spearmanr(pr, tr).statistic) if np.std(tr) > 0 else float("nan")
        c2 = float(spearmanr(pl, tl).statistic) if np.std(tl) > 0 else float("nan")
        au = _auroc(pr_[cls == 1], pr_[cls == 0])
        split[pname] = {"rho_red_lprog": c1, "rho_rel_visits": c2, "auroc_red": au, "n": len(pr)}
        print(f"    {pname:>16s}  {c1:14.3f} {c2:15.3f} {au:16.3f}", flush=True)

    # ---------------------------------------------------------------- #
    # PHASE A#2 (committee health) and heterogeneous_graders §9's DISCRIMINATOR.
    # §9: the thing that tells a genuine frontier from a noisy TV is not the level of disagreement
    # but whether it CLOSES when you collect there. So regress Δd_j(t->t+1) on the share of budget
    # spent at j at round t, separately for reducible and noise regions, pooled over arms.
    # ---------------------------------------------------------------- #
    health = {}
    for pname in policies:
        rows = results[pname]
        health[pname] = {
            "ens_cos_mean": float(np.nanmean([r["ens_cos"] for r in rows])),
            "ens_cos_by_class": {
                "reducible": float(np.nanmean([[r["ens_cos"][j] for j in reducible] for r in rows])),
                "noise": (float(np.nanmean([[r["ens_cos"][j] for j in noise_idx] for r in rows]))
                          if noise_idx else float("nan"))},
            "var_replay_mean": float(np.nanmean([r["var_replay"] for r in rows])),
            "d_by_class": {
                "reducible": float(np.nanmean([[r["disagree"][j] for j in reducible] for r in rows])),
                "noise": (float(np.nanmean([[r["disagree"][j] for j in noise_idx] for r in rows]))
                          if noise_idx else float("nan"))},
            "bme_by_class": {
                "reducible": float(np.nanmean([[r["bme"][j] for j in reducible] for r in rows])),
                "noise": (float(np.nanmean([[r["bme"][j] for j in noise_idx] for r in rows]))
                          if noise_idx else float("nan"))},
        }
    print("\n=== Phase A#2/#3: committee health + benchmark estimability ===", flush=True)
    print(f"    {'policy':>16s} {'ens_cos':>8s} {'var_replay':>11s} {'d(red)':>9s} {'d(noise)':>9s} "
          f"{'b-e(red)':>9s} {'b-e(noise)':>11s}", flush=True)
    for pname in policies:
        h = health[pname]
        print(f"    {pname:>16s} {h['ens_cos_mean']:8.3f} {h['var_replay_mean']:11.2e} "
              f"{h['d_by_class']['reducible']:9.2e} {h['d_by_class']['noise']:9.2e} "
              f"{h['bme_by_class']['reducible']:9.4f} {h['bme_by_class']['noise']:11.4f}", flush=True)

    closes = {"reducible": {"x": [], "y": []}, "noise": {"x": [], "y": []}}
    for pname in policies:
        rows = results[pname]
        for i in range(len(rows) - 1):
            tot = max(sum(rows[i]["alloc"]), 1.0)
            for j in range(K):
                cl = "reducible" if REG[j]["reducible"] else "noise"
                dd = rows[i + 1]["disagree"][j] - rows[i]["disagree"][j]
                if np.isfinite(dd):
                    closes[cl]["x"].append(rows[i]["alloc"][j] / tot); closes[cl]["y"].append(dd)
    hetero = {}
    print("\n=== heterogeneous_graders §9 discriminator: does disagreement CLOSE when you collect? ===",
          flush=True)
    for cl, dd in closes.items():
        if len(dd["x"]) > 4 and np.std(dd["x"]) > 0:
            rho = float(spearmanr(dd["x"], dd["y"]).statistic)
            slope = float(np.polyfit(dd["x"], dd["y"], 1)[0])
        else:
            rho = slope = float("nan")
        hetero[cl] = {"rho_alloc_dDisagree": rho, "slope": slope, "n": len(dd["x"])}
        print(f"    {cl:>10s}: rho(alloc share, Δdisagree) = {rho:+.3f}  slope={slope:+.3e} "
              f"(n={len(dd['x'])})   [negative = collecting there CLOSES it]", flush=True)

    out = {"config": cfg, "region_names": names,
           "region_kind": [r["kind"] for r in REG],
           "region_reducible": [bool(r["reducible"]) for r in REG],
           "region_on_reach": [bool(r["on_reach"]) for r in REG],
           "region_centers": {r["name"]: np.asarray(r["center"]).tolist() for r in REG},
           "schedule": {str(k_): v for k_, v in schedule.items()},
           "P0": P0.tolist(), "G": G.tolist(), "ceil_err": ceil_err,
           "reach_frac_probe": reach_frac_log, "invariance_gate": invariance,
           "results": results, "summary": summary, "separability": sep, "split_quality": split,
           "committee_health": health, "hetero_discriminator": hetero}
    outdir = os.path.join(DATA_DIR, "committee_head", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def committee_head(
    quick: bool = False,
    phase_a: bool = False,
    spawn: bool = False,          # queue the call server-side and return (survives client death)
    tag: str = "",
    seed: int = 0,
    policies: str = _ALL_POLICIES,
    rounds: int = 30,
    # --- the committee (the reader) ---
    k_members: int = 4,                       # K=1 + rpf_beta=0 == E3 exactly (the fidelity gate)
    lp_k: int = 0,                            # members in the counterfactual LP probe (0 = all)
    rpf_beta: float = 0.6,                    # random-prior scale (curiosity_control's design point)
    # --- the head (the judge) ---
    head_hidden: int = 32,
    head_lr: float = 3e-3,
    head_wd: float = 1e-4,
    head_steps: int = 300,                    # full-batch refits per round (dataset is K*T rows)
    head_warmup: int = 3,                     # rounds of uniform before the head is trusted
    head_min_rows: int = 8,
    head_floor: float = 1e-4,                 # the S2 no-op floor on the multiplicative signal
    head_target: str = "lprog_next",          # "realized_drop" | "lprog_smooth" (see docstring)
    head_window: int = 8,                     # rounds averaged by the `lprog_smooth` target
    head_loss: str = "huber",                 # "rank" = RankNet on the reducibility output
    hist_len: int = 3,                        # lags of each channel handed to the head
    shadow_head: bool = True,                 # train (not use) a head on every arm
    # --- the auxiliary readers ---
    bench_hidden: int = 64,
    bench_layers: int = 2,
    bench_lr: float = 1e-3,
    bench_pretrain_steps: int = 1500,
    bench_steps: int = 60,                    # online b(s) updates per round
    fm1_steps: int = 300,
    # --- head_rl (shape 2; off the default ladder) ---
    rl_temp: float = 0.02,
    rl_beta: float = 0.3,
    # --- drift / substrate (E3's defaults, unchanged) ---
    drift_mode: str = "ou",
    ou_sigma: float = 1.6,
    b_lo: float = 2.0,
    b_hi: float = 10.0,
    drift_every: int = 8,
    drift_cycle: str = "0,0,1",
    b1: float = 6.0,
    noise_amp: float = 3.0,
    region_sigma: float = 0.12,
    region_k: float = 1.5,
    geom_samples: int = 1500,
    rad_min: float = 0.80,
    pref_width: float = 0.5,
    vis_steps: int = 12,
    off_vis_max: float = 0.12,
    on_vis_min: float = 0.40,
    min_sep: float = 0.30,
    n_off_red: int = 3,
    n_off_noise: int = 2,
    # --- off-reach selection: enterable-but-off-the-eval-path (see the block in the geometry
    #     section). OFF by default, so the default geometry is E3's and the fidelity gate holds. ---
    reachable_off: bool = False,
    off_reach_min: float = 0.10,       # min measured in-region share for an off-reach distractor
    reach_shortlist: int = 12,         # separated candidates probed before filtering
    reach_probe_n: int = 120,          # on-policy transitions per candidate probe (off-budget)
    reach_pool_n: int = 4000,
    reach_fm_steps: int = 2000,
    reach_hidden: int = 128,
    n_on_noise: int = 0,
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    ep_len: int = 14,
    op_q_range: float = 0.25,
    n_par: int = 16,
    sigma_u: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14,
    wrap_limit: float = 3.0,
    goal_jit: float = 0.05,
    budget: int = 120,                        # transitions/round (transition denomination)
    mon_n: int = 90,                          # survey transitions/region (transition denomination)
    # --- episode denomination + its gate + the oracle repair. All OFF by default, so the default
    #     path is E3's and the bit-identity fidelity gate is untouched. ---
    episode_budget: bool = False,
    budget_eps: int = 12,                     # COMPLETE reaches/round, split across regions
    mon_eps: int = 3,                         # COMPLETE reaches/region/round for the survey
    gate_invariance: bool = False,
    gate_sizes: str = "1,2,4,8",              # allocation sizes (episodes) probed by the gate
    gate_tol: float = 0.35,                   # |rho(size, in-share)| below this reads FLAT
    gate_reps: int = 5,                       # independent repeats per allocation size
    oracle_mode: str = "raw",                 # "excess" = reducible part only (see run_policy)
    ceil_all_regions: bool = False,           # ceiling FM competent in EVERY region
    min_in: int = 8,
    buf_cap: int = 260,
    lp_steps: int = 400,
    replay_n: int = 2500,
    finetune_lr: float = 3e-4,
    finetune_steps: int = 700,
    reactive_every: int = 4,
    err_floor: float = 0.05,
    value_floor: float = 1e-4,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    probe_pool_n: int = 2000,
    ceil_pool_n: int = 4000,
    n_eval: int = 40,
    plan_h: int = 14,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    q_jit: float = 0.25,
    v0_std: float = 0.0,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    reach_tries: int = 40,
):
    import os

    pol = [p for p in policies.split(",") if p.strip()]
    if phase_a:
        # Phase A is READ-ONLY: a neutral allocation (uniform) plus the incumbent, so the
        # separability / committee-health / benchmark-estimability tables are read before any
        # head is allowed to touch the budget.
        if policies == _ALL_POLICIES:
            pol = ["uniform", "value"]
        rounds = rounds if rounds != 30 else 16
        tag = tag or "phase_a"
    if quick:
        # every knob below is E3's `--quick` verbatim, so `--quick --policies value --k-members 1
        # --rpf-beta 0` here and `--quick --policies value` there are the SAME experiment and the
        # fidelity gate is a bit-for-bit comparison rather than an approximate one. `k_members` is
        # deliberately NOT set here: the gate needs to pass it explicitly.
        rounds = 14; pool_n = 3000; fm_steps = 1500; finetune_steps = 300
        lp_steps = 150; replay_n = 1200; fm_hidden = 128; fm_layers = 2; n_eval = 12
        k_shoot = 256; cem_iters = 4; collect_k_shoot = 128; collect_cem_iters = 3
        budget = 120; mon_n = 60; probe_pool_n = 800; ceil_pool_n = 1500; geom_samples = 600; n_par = 8
        bench_pretrain_steps = 500; fm1_steps = 150; head_steps = 200
        if policies == _ALL_POLICIES:
            pol = ["uniform", "value", "disagree-only", "head_sup"]
        tag = tag or "smoke"
    tag = tag or "default"
    # the scheduled-burst control is defined RELATIVE to `value`'s realised mean share, so `value`
    # must run first in the same job (metered_repair §7.1).
    if ("burst" in pol or "steady" in pol) and "value" not in pol:
        pol = ["value"] + pol
    if ("burst" in pol or "steady" in pol) and pol.index("value") > min(
            [pol.index(x) for x in ("burst", "steady") if x in pol]):
        pol.remove("value"); pol = ["value"] + pol

    cfg = dict(
        tag=tag, seed=seed, policies=pol, rounds=rounds,
        k_members=k_members, rpf_beta=rpf_beta, lp_k=lp_k,
        head_hidden=head_hidden, head_lr=head_lr, head_wd=head_wd, head_steps=head_steps,
        head_warmup=head_warmup, head_min_rows=head_min_rows, head_floor=head_floor,
        head_target=head_target, head_window=head_window, head_loss=head_loss,
        hist_len=hist_len, shadow_head=bool(shadow_head),
        bench_hidden=bench_hidden, bench_layers=bench_layers, bench_lr=bench_lr,
        bench_pretrain_steps=bench_pretrain_steps, bench_steps=bench_steps, fm1_steps=fm1_steps,
        rl_temp=rl_temp, rl_beta=rl_beta,
        drift_mode=drift_mode, ou_sigma=ou_sigma, b_lo=b_lo, b_hi=b_hi, drift_every=drift_every,
        drift_cycle=[int(x) for x in drift_cycle.split(",") if x.strip()], b1=b1, noise_amp=noise_amp,
        region_sigma=region_sigma, region_k=region_k, geom_samples=geom_samples, rad_min=rad_min,
        pref_width=pref_width, vis_steps=vis_steps, off_vis_max=off_vis_max, on_vis_min=on_vis_min,
        min_sep=min_sep, n_off_red=n_off_red, n_off_noise=n_off_noise, n_on_noise=n_on_noise,
        reachable_off=bool(reachable_off), off_reach_min=off_reach_min,
        reach_shortlist=reach_shortlist, reach_probe_n=reach_probe_n, reach_pool_n=reach_pool_n,
        reach_fm_steps=reach_fm_steps, reach_hidden=reach_hidden,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, q_range=q_range,
        v_explore=v_explore, ep_len=ep_len, op_q_range=op_q_range, n_par=n_par, sigma_u=sigma_u,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        collect_replan_every=collect_replan_every, wrap_limit=wrap_limit, goal_jit=goal_jit,
        budget=budget, mon_n=mon_n, min_in=min_in,
        episode_budget=bool(episode_budget), budget_eps=budget_eps, mon_eps=mon_eps,
        gate_invariance=bool(gate_invariance),
        gate_sizes=[int(x) for x in gate_sizes.split(",") if x.strip()], gate_tol=gate_tol,
        gate_reps=gate_reps,
        oracle_mode=oracle_mode, ceil_all_regions=bool(ceil_all_regions), buf_cap=buf_cap, lp_steps=lp_steps,
        replay_n=replay_n, finetune_lr=finetune_lr, finetune_steps=finetune_steps,
        reactive_every=reactive_every, err_floor=err_floor, value_floor=value_floor,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pool_n=pool_n, probe_pool_n=probe_pool_n, ceil_pool_n=ceil_pool_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        reach_tries=reach_tries,
    )
    if spawn:
        # SERVER-SIDE detach. `modal run --detach <file>::<local entrypoint>` does NOT reliably
        # survive the client dying: a container restart here cancelled four in-flight Phase B jobs
        # mid-round ("Received a cancellation signal"), which is exactly what
        # `/run-experiment-on-modal` warns about for local entrypoints. `.spawn()` queues the call
        # server-side and returns immediately, so nothing is tied to this client's lifetime.
        # The local results.json mirror is skipped by construction -- pull the durable copy from
        # the volume instead:
        #     modal volume get mujoco-control-data /committee_head/<tag>/results.json <dest>
        call = run_committee_head.spawn(cfg)
        print(f"[spawn] call {call.object_id} queued for tag={tag!r} "
              f"({len(pol)} policies x {rounds} rounds)")
        print(f"[spawn] results -> /data/committee_head/{tag}/results.json on mujoco-control-data")
        return
    out = run_committee_head.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "committee_head_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
