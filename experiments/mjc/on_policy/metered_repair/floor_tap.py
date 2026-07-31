"""E4 -- the REDUCIBILITY TAP, REPAIRED AND CERTIFIED, and the visited-but-irreducible cell PLACED.

Direct parent: [`../directed_on_policy/`](../directed_on_policy/README.md) (E3, PR #4) -- this is a
copy-and-modify of its loop, per `CLAUDE.md`'s "copy to a new folder rather than pile
backwards-compatible mods on a published script". E3's `ladder_s{0,1,2}` are untouched and still
reproduce.

WHY. Two facts from `rhm/directed_sculpting/full_loop/` (PRs #15/#16) land directly on E3:

  (1) E3's learning-progress tap is a FIXED-BUDGET COUNTERFACTUAL FIT
      (`directed_on_policy.py:559-567`: fine-tune a fork on half the in-region monitor batch, read
      the held-out drop). RHM ran exactly that estimator and found it ordered its channels
      *backwards* -- lowest on the target (+0.044), highest on the irreducible channels (+0.066) --
      because a fixed-budget fit measures MARGINAL RETURN, and early marginal return is dominated by
      DATA STARVATION, not by reducibility. E3's geometry has the same hazard with a different
      cause: the OFF-REACH regions are the data-starved ones, and they are exactly the distractors.
      E3's own caveat is the signature -- `value` still pours **29%** of its budget into the
      irreducible regions against `oracle`'s 0%, and `lprog-only` sends 35% against uniform's 33%,
      i.e. the tap is barely filtering at all.
      RHM's repair: stop measuring a FLOW (marginal return) and measure a STOCK --
          reducible_fraction = (err - aleatoric_floor) / err
      with the floor MEASURED rather than inferred. Worth -0.054 +- 0.009 there (3/3 seeds), with an
      8x drop in allocation variance and the irreducible leak falling 14.2% -> 7.3%.

  (2) The two substrates have MIRROR-IMAGE DEGENERATE GEOMETRIES. RHM has no
      visited-but-IRREDUCIBLE cell, so its `value_red` can only TIE `visits_only` (+0.0019 +- 0.0030)
      *by construction* -- "this geometry has no visited-but-irreducible cell, so the channels the
      planner visits are also the reducible ones". E3 has no visited-but-IRRELEVANT cell ("you only
      go where you reach"), so `visits-only` sits naturally close to `value`. Between them, the
      conjunction `lprog x visits` has NEVER been shown to need BOTH terms on either substrate.

WHAT THIS CHANGES (four things, each a knob, each reported):

  A. THE FLOOR, MEASURED. `matched_pairs_floor` estimates the aleatoric floor of the FM's own target
     (Delta s | s,u) from the agent's OWN metered in-region monitor samples, by k-nearest-neighbour
     matching in normalised (s,u) space: residual of each sample's Delta s around its local
     neighbourhood mean, bias-corrected for the finite neighbourhood. RHM measured its floor by
     RE-EXECUTING the same (x,k) four times -- exact re-executability the RHM README correctly notes
     "a physical substrate does not" hand us, and which `Body` forbids outright (no `set_state`).
     Matched pairs is the honest substitute: it needs no repeats, no labels, and no teleport.
     It is BIASED UP by local curvature of the true dynamics over the neighbourhood, which is why
     `certify_floor` exists (below) and why every arm gets the same estimator.

  B. THE STOCK TAP. `red = clip((err - floor)/err, 0, 1)`, EMA-smoothed over rounds (RHM's variance
     half: it also added EMA smoothing and frozen probes). Two new arms: `reducible-only` (the tap
     alone) and `value-red` (`red x visits`). BOTH taps are computed EVERY round for EVERY arm, so
     the metered monitor charge is identical and the ONLY thing that differs between arms is the
     allocation rule -- RHM's own protocol, and what makes `value` vs `value-red` a within-run
     measurement of the repair rather than a cross-run comparison.

  C. THE MISSING CELL. E3's eval task is a NARROW angular cone (`pref_width=0.5` half-width, and
     `eval_geometry` takes the B *most* on-reach endpoints), which is *why* `on_vis_min=0.40` finds
     no candidate and `n_on_noise` comes out empty. Widen the reach-goal distribution
     (`pref_width=1.2`, `eval_spread=True` samples goals ACROSS the sector instead of at its centre)
     and a visited-but-irreducible region C becomes placeable. Now `visits-only` must split A/C while
     `value-red` lands on A alone -- the first test on either substrate of whether the conjunction
     needs both terms.

  D. WHAT IS *NOT* CHANGED, AND THE ONE TRADE THAT IS. E3's drift rate (`ou_sigma=1.6`) and collection
     budget (120/round) are KEPT, so E4 sits in the same scarcity regime whose ladder resolved. A
     slower walk -- which would give the near-saturated control instrument some dynamic range, since
     E3 reports "no policy fully repairs A" -- is a real follow-up and a one-knob sweep, but halving
     the damage while the tap also changes would confound the repair with the regime, which is the
     exact confound RHM's necessity sweep had to apologise for.
     The trade that IS made: K drops 6 -> 5 and `mon_n` 40 -> 60. Every region costs CHARGED survey
     steps, so K multiplies the monitor bill; the floor tap needs a denser in-region batch than the
     flow tap does, so the freed steps buy batch density. Monitor:collect goes 1.84x -> ~2.5x, which
     is reported. The anti-subsidy property is that looking is charged and O(1) -- S2's pathology was
     22x and FREE -- and the ratio is identical across arms by construction, so it cannot favour one.

THE CERTIFICATION GATE (`certify_floor`, cheap, run it FIRST). A measured floor is only worth
having if it measures the floor, so certify the estimator against three independent references on
the off-budget per-region probe sets, before any ladder runs:
  * the KNOWN ground truth -- `noise_amp` is 0 for every curl region by construction, so the
    estimator's output there IS its curvature bias, and that is the number to report;
  * the MATCHED-FM CEILING (`ceil_err`, E3's existing privileged instrument) -- a matched FM cannot
    beat the floor, so `floor_hat <= ceil_err` per region is a hard sanity bound;
  * the ORDERING -- noise regions must come out strictly above curl regions, which is the only
    property the tap actually consumes.

READOUT. E3's, plus: `floor`/`red` per region per round; budget share to the irreducible regions
(the leak); and per-arm allocation-share sd (RHM's "tree-share sd" column, where the repair showed
up as 0.396 -> 0.047).

Run:
    cd experiments/            # MODAL_PROFILE=chromatic
    modal run mjc/on_policy/verify_backcompat.py::verify                       # the env gate
    modal run mjc/on_policy/metered_repair/floor_tap.py::floor_tap \
        --quick --certify-only --tag cert          # the two GATES (geometry + floor), ~minutes
    modal run mjc/on_policy/metered_repair/floor_tap.py::floor_tap --quick     # smoke
    for s in 0 1 2; do          # launch each seed as its OWN client (siblings evict each other)
      modal run --detach mjc/on_policy/metered_repair/floor_tap.py::floor_tap --tag tap_s$s --seed $s
    done
    python3 mjc/on_policy/metered_repair/floor_tap_agg.py --tags tap_s0 tap_s1 tap_s2
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# the full ladder, and the `--policies` default. Named so `--quick` can tell "the user asked for
# a specific policy" from "the user took the default" without changing either behaviour.
# `reducible-only` and `value-red` are the two new arms (the repaired STOCK tap); `lprog-only` and
# `value` are E3's FLOW tap, kept in the ladder so the repair is a WITHIN-RUN measurement.
_ALL_POLICIES = ("uniform,oracle,error-only,lprog-only,reducible-only,reddelta-only,"
                 "visits-only,value,value-red,value-reddelta")


@app.function(gpu="L4", memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_floor_tap(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from mjc.embodied import ReachBehaviour

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    B = cfg["n_eval"]; T = cfg["rounds"]
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    # ================================================================= #
    # GEOMETRY -- place the 2x2 regions in tip space, relative to the arm's reachable fan, and
    # bias the eval task toward the ON-reach direction so on/off-reach is a real distinction.
    # Everything reachable-by-construction (drawn from actual reach endpoints), verified by the
    # printed diagnostics (in-region collection share + eval visitation per region).
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
    radius = np.linalg.norm(cloud, axis=1)              # each endpoint's distance from the base
    ext = radius > cfg["rad_min"]                        # EXTENDED endpoints only -- see below
    # THE ON-REACH DIRECTION MUST BE TAME. Folded/inward reaches (small radius) are fast and
    # high-inertia-swing -- their baseline Δs and FM error are huge (~2.4), which SWAMPS the curl
    # drift signal (the thing the whole experiment measures) and makes learning-progress dominated by
    # fitting baseline dynamics rather than the drift. E2's `curl_center` sat at radius ~0.95 for
    # exactly this reason. So place every region in the EXTENDED band (moderate velocity), and choose
    # the on-reach direction as the densest angular sector AMONG EXTENDED endpoints.
    ang_all = np.arctan2(cloud[:, 1] - P0[1], cloud[:, 0] - P0[0])
    hcnt, edges = np.histogram(ang_all[ext], bins=24, range=(-np.pi, np.pi))
    theta_pref = 0.5 * (edges[np.argmax(hcnt)] + edges[np.argmax(hcnt) + 1])
    dth = np.angle(np.exp(1j * (ang_all - theta_pref)))
    ev_mask = (np.abs(dth) < cfg["pref_width"]) & ext
    G = cloud[ev_mask].mean(0)                          # eval-goal centroid (reaches END near here)

    # ACTUAL PATH VISITATION (FM-free). The reach tip PATH is curved (FK of a near-linear joint sweep
    # is curved in tip space), so a region's relevance is set by how much the eval reaches PASS
    # THROUGH it, not by its distance to the straight P0->G segment (which mislabelled swung-through
    # regions as off-reach). Approximate each on-reach eval reach by interpolating q0->qg in JOINT
    # space and FK-ing to a tip trajectory, then gate-sum occupancy at each candidate. This is the
    # same quantity the live `fm_visits` measures, computed here without a trained model.
    ev_q0, ev_qg = q0_cloud[ev_mask], qg_cloud[ev_mask]
    sig2 = 2.0 * cfg["region_sigma"] ** 2

    def path_visitation(cands):
        occ = np.zeros(len(cands))
        for a in np.linspace(0.0, 1.0, cfg["vis_steps"]):
            tips = fk(ev_q0 * (1 - a) + ev_qg * a, Ls)                  # (M, 2)
            dd = ((cands[:, None, 0] - tips[None, :, 0]) ** 2
                  + (cands[:, None, 1] - tips[None, :, 1]) ** 2)        # (C, M)
            occ += np.exp(-dd / sig2).sum(1)
        return occ / max(occ.max(), 1e-9)

    vis = path_visitation(cloud)
    G = cloud[ev_mask].mean(0)                          # eval-goal centroid = far end of on-reach

    def _nearest(pt):
        return float(np.linalg.norm(cloud - pt[None, :], axis=1).min())

    def _farthest_points(cands, m, seed):
        """Greedy farthest-point sampling: m well-separated points from a candidate set."""
        if len(cands) == 0 or m <= 0:
            return np.zeros((0, 2))
        rng = np.random.default_rng(seed)
        idx = [int(rng.integers(len(cands)))]
        while len(idx) < m and len(idx) < len(cands):
            d = np.min(np.stack([np.linalg.norm(cands - cands[i], axis=1) for i in idx]), 0)
            idx.append(int(np.argmax(d)))
        return cands[idx]

    # THE REGION SET -- SCARCITY AMONG DISTRACTORS, the on-policy S1/S2 ladder. On this arm the curl
    # is easy to learn (~a round's worth of data repairs a region), so with only one target the budget
    # is never binding and allocation is not a lever. S1 / curiosity Phase 2b: the value signal beats
    # uniform only when the reducible frontier is SMALL among MANY distractors. So:
    #   * one ON-REACH reducible target A (the genuinely MOST-visited extended point, where control
    #     depends on the model) -- the right place to spend budget;
    #   * `n_off_red` OFF-REACH REDUCIBLE distractors (low visitation, learnable-but-irrelevant) --
    #     catch lprog-only, which chases reducible structure without asking whether it matters;
    #   * `n_off_noise` OFF-REACH IRREDUCIBLE-NOISE distractors (low visitation, high but UNLEARNABLE
    #     error) -- the noisy-TV trap; catch error-only, which chases raw prediction error.
    # E4's CHANGE (C): the ON-REACH noise decoy -- S1/S2's region C, the visited-but-IRREDUCIBLE cell
    # that separates `value-red` from `visits-only` -- IS placeable here, and E3's report that it is
    # not was a fact about E3's *task*, not about embodiment. E3's eval sector is a narrow cone
    # (`pref_width=0.5`, and `eval_geometry` takes the B *most* on-reach endpoints), so visitation
    # collapses onto A itself and `on_vis_min=0.40` has no candidate. Widen the reach-goal
    # distribution (`pref_width`, `eval_spread`) and the visited set becomes a band that supports a
    # second well-separated high-visitation site. C is placed BEFORE the off-reach distractors so it
    # gets the best remaining visited spot, and every later region must clear `min_sep` from BOTH A
    # and C -- E3 gotcha (iii): noise that bleeds into A destroys A's reducibility.
    # All regions extended (tame), classified by ACTUAL path visitation, and REACHABLE.
    # Reachability is not optional and E5's calibration gate found out the hard way. Candidates come
    # from `sample_reach_endpoints`, whose tip-displacement band [reach_lo, reach_hi] is enforced only
    # BEST-EFFORT (`reach_tries` samples, least-bad kept even at non-zero penalty), so the cloud has a
    # tail far outside the band -- and the off-reach picks, being the LEAST-visited, select that tail
    # preferentially. In E5 that placed a region 0.81 m from the start tip against a 0.50 m band, the
    # body could not get there, its in-region probe set came back 0/500, and every downstream number
    # would have been a NaN reading as "no effect". E3's own gotcha (i) is the mirror of this.
    d_P0 = np.linalg.norm(cloud - P0[None, :], axis=1)
    # The binding constraint is DYNAMIC reach, not kinematic. Every cloud point is kinematically
    # reachable by construction (it IS the FK of some `qg`), but `collect_toward` plans ONE 14-step
    # open-loop CEM sequence from a jittered home posture, so what matters is how far the tip can
    # actually be driven in `ep_len` steps at this gear. cert_v2 measured the cutoff directly:
    # regions at 0.22 / 0.30 / 0.33 / 0.43 m from the start tip returned 272-1308 in-region probe
    # transitions; the one at 0.55 m returned ZERO. So the band is calibrated to that observation
    # (`reach_pad` 0.90 -> 0.45 m) rather than to the nominal `reach_hi`.
    reachable = d_P0 <= cfg["reach_hi"] * cfg["reach_pad"]
    # ...and a DENSITY floor, which guards a different failure the distance filter cannot see: a point
    # in a sparse tail of the endpoint cloud has few postures that land in its gate, so the planner has
    # few solutions even at a comfortable distance. Count how many reach endpoints fall inside each
    # candidate's own gate.
    gate_r = cfg["region_k"] * cfg["region_sigma"]
    dens = np.array([int((np.linalg.norm(cloud - cloud[i][None, :], axis=1) < gate_r).sum())
                     for i in range(len(cloud))])
    dense = dens >= max(2, int(cfg["min_density_frac"] * len(cloud)))
    reachable = reachable & dense
    print(f"[geom] reachable candidates: {int((ext & reachable).sum())}/{len(cloud)} "
          f"(dist<={cfg['reach_hi'] * cfg['reach_pad']:.2f}m, density>="
          f"{max(2, int(cfg['min_density_frac'] * len(cloud)))})", flush=True)
    ext_idx = np.flatnonzero(ext & reachable)
    A_c = cloud[ext_idx[np.argmax(vis[ext_idx])]]      # the genuinely most-visited reachable point
    REG = [dict(name="A", center=A_c, kind="curl", reducible=True, on_reach=True)]
    placed = [A_c]

    def _sep_mask(pts):
        """Candidates at least `min_sep` from every already-placed region centre."""
        m = np.ones(len(cloud), bool)
        for p in pts:
            m &= np.linalg.norm(cloud - np.asarray(p)[None, :], axis=1) > cfg["min_sep"]
        return m

    def _pick_by_vis(mask, m, highest: bool, label: str):
        """The m most- (or least-) visited candidates under `mask`, each re-separated from the ones
        already picked. Returns fewer than m only if the workspace genuinely cannot supply them --
        which is reported rather than silently absorbed."""
        out = []
        for _ in range(m):
            cand = mask & _sep_mask(placed + out)
            if not cand.any():
                print(f"[geom] WARNING: only {len(out)}/{m} {label} regions placeable "
                      f"(min_sep={cfg['min_sep']}) -- the geometry is degenerate, read with care",
                      flush=True)
                break
            ci = np.flatnonzero(cand)
            pick = ci[np.argmax(vis[ci])] if highest else ci[np.argmin(vis[ci])]
            out.append(cloud[pick])
        return out

    # (C) the visited-but-irreducible cell: most-visited site that is not A
    for i, pt in enumerate(_pick_by_vis(ext & reachable, cfg["n_on_noise"], True,
                                        "on-reach noise")):
        REG.append(dict(name=f"Con{i + 1}", center=pt, kind="noise", reducible=False, on_reach=True))
        placed.append(pt)
    # off-reach distractors: least-visited extended sites. `off_vis_max` is now a REPORTED
    # requirement rather than a hard filter -- a wider eval sector visits more of the workspace, so a
    # hard threshold can empty the candidate set; taking the least-visited sites and printing what
    # visitation they actually achieved fails loudly instead of silently.
    n_off = cfg["n_off_red"] + cfg["n_off_noise"]
    for i, pt in enumerate(_pick_by_vis(ext & reachable, n_off, False, "off-reach")):
        if i < cfg["n_off_red"]:
            REG.append(dict(name=f"Boff{i + 1}", center=pt, kind="curl", reducible=True, on_reach=False))
        else:
            REG.append(dict(name=f"Doff{i - cfg['n_off_red'] + 1}", center=pt, kind="noise",
                            reducible=False, on_reach=False))
        placed.append(pt)
    K = len(REG)
    for r in REG:
        r["sigma"] = cfg["region_sigma"]
    names = [r["name"] for r in REG]
    reducible = [j for j in range(K) if REG[j]["reducible"]]
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

    # E4's geometry is only valid if the 2x2 is actually a 2x2: the two on-reach cells must both be
    # genuinely visited and the off-reach cells genuinely not. Report it as a gate rather than trust
    # it -- this is the whole reason E3 could not place C, and it is checked in the smoke.
    geom_gate = {"A_vis": float(reg_vis[0]),
                 "dist_from_P0": [float(np.linalg.norm(np.asarray(r["center"]) - P0)) for r in REG],
                 "on_noise_vis": [float(reg_vis[j]) for j in range(K) if REG[j]["kind"] == "noise"
                                  and REG[j]["on_reach"]],
                 "off_vis": [float(reg_vis[j]) for j in range(K) if not REG[j]["on_reach"]]}
    geom_gate["pass_A"] = geom_gate["A_vis"] >= cfg["on_vis_min"]
    geom_gate["pass_on_noise"] = (len(geom_gate["on_noise_vis"]) == cfg["n_on_noise"]
                                  and all(v >= cfg["on_vis_min"] for v in geom_gate["on_noise_vis"]))
    geom_gate["pass_off"] = all(v <= cfg["off_vis_max"] for v in geom_gate["off_vis"])
    geom_gate["off_vis_ratio"] = [float(geom_gate["A_vis"] / max(v, 1e-6)) for v in geom_gate["off_vis"]]
    print(f"[geom] GATE  A_vis={geom_gate['A_vis']:.2f}(>={cfg['on_vis_min']}) -> "
          f"{'PASS' if geom_gate['pass_A'] else 'FAIL'} | "
          f"on-noise vis={[round(v, 2) for v in geom_gate['on_noise_vis']]} -> "
          f"{'PASS' if geom_gate['pass_on_noise'] else 'FAIL'} | "
          f"off vis max={max(geom_gate['off_vis']) if geom_gate['off_vis'] else 0:.2f}"
          f"(<={cfg['off_vis_max']}) -> {'PASS' if geom_gate['pass_off'] else 'FAIL'}  "
          f"| A:off visitation ratio={[round(r, 1) for r in geom_gate['off_vis_ratio']]}", flush=True)

    def gate_np(tips, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((tips[..., 0] - c[0]) ** 2 + (tips[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    def in_region(states, j):
        tips = fk(np.asarray(states)[:, :n].astype(np.float64), Ls)
        return np.linalg.norm(tips - REG[j]["center"], axis=1) < cfg["region_k"] * REG[j]["sigma"]

    # ================================================================= #
    # ENV -- one plant, current per-region drift gains folded into curl_fields / noise_fields.
    # ================================================================= #
    def env_dgp(b_state):
        """The fully-resolved DGP knob dict for a given per-region drift state. Split out of
        `make_env` so a render pack can carry the EXACT plant it was recorded on (no
        re-derivation, no drift between the loop and the renderer)."""
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
    # FM  f(s,u) -> Δs   (identical architecture/training to E2 readapt_local.py)
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

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32), device=device)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def fm_err(net, S, U, S2):
        if len(S) == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(net, S, U) - (S2 - S).astype(np.float32), axis=1).mean())

    # ================================================================= #
    # E4's CHANGE (A) -- THE MEASURED ALEATORIC FLOOR.
    #
    # RHM measured its floor by RE-EXECUTING the same (x, k) four times and reading the residual
    # around the conditional mean. `Body` forbids that outright (no `set_state`), and the RHM README
    # says so itself: "exact re-executability is something RHM hands us that a physical substrate does
    # not". MATCHED PAIRS is the honest substitute -- no repeats, no labels, no teleport, computed
    # from the agent's OWN metered in-region monitor batch:
    #
    #   for each sample i, take its k nearest neighbours in normalised (s,u) space; the residual of
    #   i's target around that neighbourhood's mean is noise plus whatever the true map does across
    #   the neighbourhood. Bias-correct for the finite neighbourhood (p = k+1 draws around their own
    #   mean carry only sqrt((p-1)/p) of the population spread) and average.
    #
    # RUN IT ON THE FM'S RESIDUAL, NOT ON Δs. This is the one design choice that makes the estimator
    # usable on a continuous substrate. The curvature bias of matched pairs is the local variation of
    # whatever field you difference; on raw Δs that is the FULL arm dynamics (large), on the residual
    # Δs − f(s,u) it is only the part f has NOT already captured (small, and smooth in (s,u), since
    # both the curl and the model are). The limits are then exactly right by construction: in a
    # well-fit region the residual IS the noise, the local mean is ~0, floor -> err and red -> 0; in a
    # region whose curl has drifted the residual has a large smoothly-varying systematic part, the
    # local mean absorbs it, and red -> 1.
    def matched_pairs_floor(net, S, U, S2, k=None, on=None):
        """(floor_estimate, mean_neighbour_distance, n_used). NaN if the batch is too small.

        `on="residual"` (default) differences the FM residual; `on="delta"` differences raw Δs and is
        kept only so `certify_floor` can show what the residual version buys."""
        k = int(cfg["floor_k"]) if k is None else int(k)
        on = cfg["floor_on"] if on is None else on
        m = len(S)
        if m < k + 2:
            return float("nan"), float("nan"), m
        cap = int(cfg["floor_max_n"])
        if m > cap:                                        # bound the O(m^2) distance matrix
            sel = np.random.default_rng(cfg["seed"] + 7777).permutation(m)[:cap]
            S, U, S2 = S[sel], U[sel], S2[sel]
            m = cap
        X = np.concatenate([S, U], 1).astype(np.float64)
        mx = norm["mx"].cpu().numpy().astype(np.float64)
        sx = norm["sx"].cpu().numpy().astype(np.float64)
        Z = (X - mx) / sx
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        nb = np.argsort(D, axis=1)[:, :k]
        Y = (S2 - S).astype(np.float64)
        if on == "residual":
            Y = Y - fm_delta(net, S, U).astype(np.float64)
        grp = np.concatenate([Y[:, None, :], Y[nb]], 1)    # (m, k+1, SD): self + its neighbours
        resid = Y - grp.mean(1)
        p = k + 1
        floor = float(np.linalg.norm(resid, axis=1).mean() / np.sqrt((p - 1) / p))
        nbd = float(np.sqrt(np.take_along_axis(D, nb, 1)).mean())
        return floor, nbd, m

    def reducible_fraction(net, S, U, S2):
        """The STOCK tap: what fraction of the current in-region error is NOT irreducible noise.
        `err` here is the same `fm_err` the rest of the loop reports, so `red` is on a scale the
        allocation trace and the grader share."""
        err = fm_err(net, S, U, S2)
        floor, nbd, nu = matched_pairs_floor(net, S, U, S2)
        if not (err == err) or not (floor == floor) or err <= 1e-12:
            return float("nan"), err, floor, nbd
        return float(np.clip((err - floor) / err, 0.0, 1.0)), err, floor, nbd

    def teleport_pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs, qc, cfg["q_range"], cfg["v_explore"])

    # norm stats from a broad teleport pool of the DRIFTED world (off-budget: the experimenter's
    # ruler, exactly as E2). b_state at round 0 = reducible regions drifted.
    b0_state = np.array([cfg["b1"] if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    env_drift0 = make_env(b0_state)
    nS, nU, nS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ================================================================= #
    # CEM planner + eval rollout (reach toward a Cartesian goal; cost = tip-to-goal)
    # ================================================================= #
    def make_plan_fn(net, k_shoot, cem_iters, rng):
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
                        s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
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
        dsig = np.angle(np.exp(1j * (a - theta_pref)))    # SIGNED offset from the preferred heading
        d = np.abs(dsig)
        if cfg["eval_spread"]:
            # E4's CHANGE (C), the task half: SPREAD the eval goals across the on-reach sector instead
            # of piling them at its centre. Stratified over the signed angle so coverage is even and
            # deterministic (no extra seed dependence). This is what turns the visited set from a tube
            # into a band, which is what makes a second visited site -- region C -- exist at all.
            insec = np.flatnonzero(d < cfg["pref_width"])
            if len(insec) >= B:
                order = insec[np.argsort(dsig[insec])]
                pick = order[np.linspace(0, len(order) - 1, B).round().astype(int)]
            else:                                          # sector too thin: fall back to E3's rule
                pick = np.argsort(d)[:B]
        else:
            pick = np.argsort(d)[:B]                       # E3: the B most on-reach endpoints
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

    def fm_visits(net):
        """Region occupancy of the ballistic eval plan rolled through the FM ITSELF (no env, no
        budget) -- the relevance signal is purely internal (S1: the FM tells you where you'll be)."""
        plan = make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 7001))(ev_starts, ev_goals)
        v = np.zeros(K)
        with torch.no_grad():
            s = torch.tensor(ev_starts, device=device)
            for h in range(H):
                tips = fk_torch(s[:, :n]).cpu().numpy()
                for j in range(K):
                    v[j] += float(gate_np(tips, j).sum())
                x = torch.cat([s, torch.tensor(plan[:, h, :], device=device)], 1)
                s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
        return v / max(v.sum(), 1e-9)

    # ================================================================= #
    # RENDER PACK (off by default) -- everything `render_ladder.py` needs to replay and draw
    # this round's ballistic reaches, including the FM's OWN forecast of them.
    # ================================================================= #
    render_packs = []
    render_rounds = set(cfg.get("render_rounds", []))
    render_pols = set(cfg.get("render_policies", []))

    def render_pack(pname, t, net, b_state):
        """Record the ballistic reaches this FM plans, what the body actually does, and what the
        FM *thought* would happen -- the predicted-vs-actual divergence IS the thing that shrinks
        as the model is re-calibrated, and it is invisible in any scalar we already log.

        SIDE-EFFECT-FREE BY CONSTRUCTION, which is the whole reason this is safe to bolt onto a
        published loop: it uses a FRESH `ArmEnv` (hence its own `_noise_rng`, so the graded
        rollout's noise draws are untouched) and FRESH plan generators (`make_plan_fn` seeds its
        own), and it never trains. With `--render-rounds ""` (the default) it is not called at
        all, so `ladder_s{0,1,2}` reproduce exactly.

        The replay is ONE CONTINUOUS EPISODE per reach rather than `rollout()`'s multiplexed
        `set_state`-per-step -- exact for the contactless arm (`../README.md` §machinery), modulo
        the float32 state round-trip the multiplexed idiom incurs (~1e-7).
        """
        # Record EVERY eval reach by default (`render_n=0`). The point is that the pack's `miss` is
        # then the same 40-reach median the loop grades as `ballistic_cem`, not a 4-reach subsample
        # of it -- those disagree badly (at one round, 9.1 cm over 4 reaches vs 4.9 cm over 40), and
        # a headline number in a video must be the real metric. The video renders only the first
        # few; recording the rest is nearly free.
        R = B if int(cfg["render_n"]) <= 0 else min(int(cfg["render_n"]), B)
        renv = make_env(b_state)
        plan = make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 7001))(ev_starts, ev_goals)
        starts, goals, acts = ev_starts[:R], ev_goals[:R], plan[:R]

        actual = np.zeros((R, H + 1, SD), np.float64)          # what the BODY does
        for b in range(R):
            renv.set_state(starts[b, :n].astype(np.float64), starts[b, n:].astype(np.float64))
            actual[b, 0] = starts[b]
            for h in range(H):
                actual[b, h + 1], _ = renv.step(acts[b, h], fs)

        with torch.no_grad():                                   # what the MODEL expected
            s = torch.tensor(starts, device=device)
            pred = [s.cpu().numpy().copy()]
            for h in range(H):
                x = torch.cat([s, torch.tensor(acts[:, h, :], device=device)], 1)
                s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                pred.append(s.cpu().numpy().copy())
            pred = np.stack(pred, 1).astype(np.float64)

        a_tip = fk(actual[..., :n], Ls)
        p_tip = fk(pred[..., :n], Ls)
        per_miss = np.linalg.norm(a_tip[:, -1] - goals, axis=1)
        per_div = np.linalg.norm(p_tip - a_tip, axis=2).mean(1)
        miss = float(np.median(per_miss))
        div = float(per_div.mean())
        print(f"[render] pack {pname} r{t}: R={R} miss={miss:.4f} model-vs-body tip div={div:.4f}",
              flush=True)
        return {"policy": pname, "round": int(t), "b_state": np.asarray(b_state).tolist(),
                "dgp": env_dgp(b_state), "starts": starts.tolist(), "goals": goals.tolist(),
                "actions": acts.tolist(), "actual_states": actual.tolist(),
                "actual_tips": a_tip.tolist(), "pred_tips": p_tip.tolist(),
                "miss": miss, "tip_divergence": div,
                "per_miss": per_miss.tolist(), "per_div": per_div.tolist()}

    # ================================================================= #
    # ON-POLICY reach-toward-a-region collection (the metered acquisition primitive)
    # ================================================================= #
    def make_fixed_goal_sampler(center, jit):
        c = np.asarray(center, np.float32)

        def sample(states, rng):
            m = len(np.atleast_2d(np.asarray(states)))
            return (c[None, :] + rng.uniform(-jit, jit, (m, 2))).astype(np.float32)
        return sample

    def collect_toward(cenv, center, need, fm, rng_buf, k):
        """Reach toward `center` under the current FM; return the transitions the body lived +
        info (steps_used). One env.step per transition, so `need` == steps charged."""
        plan_fn = make_plan_fn(fm, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                               np.random.default_rng(cfg["seed"] + 8000 + k))
        beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng_buf,
                             sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
        gs = make_fixed_goal_sampler(center, cfg["goal_jit"])
        S, U, S2, info = collect_pool(cenv, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                      collection_mode="on_policy", behaviour=beh, goal_sampler=gs,
                                      ep_len=cfg["ep_len"], n_par=cfg["n_par"], v0_std=cfg["v0_std"],
                                      wrap_limit=cfg["wrap_limit"], return_info=True)
        return S, U, S2, info

    # ================================================================= #
    # BASE (stale) FM -- trained off-budget on the PRE-drift world (b=0 in reducible regions; the
    # noise regions are permanent). This is the innate/pre-episode model every policy starts from.
    # ================================================================= #
    b_pre = np.array([0.0 if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    env_pre = make_env(b_pre)
    S0_, U0_, S20_ = teleport_pool(env_pre, cfg["pool_n"], cfg["seed"] + 10)
    fm_base = _mlp(cfg["seed"] + 40)
    train_steps(fm_base, torch.optim.Adam(fm_base.parameters(), lr=cfg["fm_lr"]),
                S0_, U0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    base_state = copy.deepcopy(fm_base.state_dict())
    # replay buffer: out-of-EVERY-region base transitions (valid post-drift; keeps the counterfactual
    # fit from smearing a local rotation globally -- S0's lesson).
    gmax = np.max(np.stack([gate_np(fk(S0_[:, :n].astype(np.float64), Ls), j) for j in range(K)]), 0)
    oidx = np.flatnonzero(gmax < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = S0_[rp], U0_[rp], S20_[rp]
    print(f"[pretrain] base FM trained on pre-drift pool; replay={len(rp)} out-of-region transitions",
          flush=True)

    # ================================================================= #
    # CONTINUOUS DRIFT (COLLECTION_REALISM.md §4): a reflecting random walk on each reducible
    # region's curl gain, so the field is NEVER permanently repaired and allocation stays zero-sum
    # round after round -- the regime cut 4a named as the only one where a value loop earns anything.
    # The trajectory is precomputed ONCE and SHARED across every policy, so the only thing that
    # varies between policies is where they spend the budget (controlled). A single scheduled drift
    # (the S2 style) is available via drift_mode="schedule" but converges once repaired.
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
                    nb = 2 * cfg["b_lo"] - nb                    # reflecting bounds keep it moving
                if nb > cfg["b_hi"]:
                    nb = 2 * cfg["b_hi"] - nb
                b_cur[j] = float(np.clip(nb, cfg["b_lo"], cfg["b_hi"]))
        else:                                                    # "schedule": re-drift periodically
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
    # one policy's full T-round loop
    # ================================================================= #
    def run_policy(pname):
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"])
        bufs = {j: None for j in range(K)}
        red_ema = np.full(K, np.nan)                             # E4: per-policy stock-tap EMA state
        err_prev = np.full(K, np.nan)                            # last round's per-region monitor err
        floor_hist = {j: [] for j in range(K)}                   # per-region (floor, n) window
        hist = []
        # round "-1" = the STALE base FM, before this policy has collected anything, graded against
        # the round-0 drifted world. The strongest "early" frame there is: every round >= 0 has
        # already been fine-tuned once on that round's collection.
        if pname in render_pols and -1 in render_rounds:
            render_packs.append(render_pack(pname, -1, net, b_traj[0]))
        for t in range(T):
            b_state = b_traj[t]                                  # shared drift; only alloc varies
            if t in schedule:
                bufs[schedule[t][0]] = None                     # (schedule mode) voided on re-drift
            env = make_env(b_state)
            mon_steps = 0; coll_steps = 0

            # ---------- METERED on-policy survey: reach each region, gather signals ----------
            # E4's CHANGE (B): BOTH taps are computed EVERY round for EVERY arm. The metered monitor
            # charge is then identical across the ladder and the ONLY difference between arms is the
            # allocation rule -- RHM's protocol, and what makes `value` vs `value-red` a within-run
            # measurement of the repair instead of a cross-run comparison. The extra cost is the
            # counterfactual fit's GPU time (off-budget), not steps.
            per_err = np.zeros(K); lprog = np.zeros(K)
            red = np.full(K, np.nan); floor = np.full(K, np.nan); nbdist = np.full(K, np.nan)

            n_in = np.zeros(K, int); fell_back = 0
            cur = copy.deepcopy(net.state_dict())
            for j in range(K):
                mS, mU, mS2, minfo = collect_toward(env, REG[j]["center"], cfg["mon_n"], net,
                                                    np.random.default_rng(cfg["seed"] + 9000 + 41 * t + j), 100 + j)
                mon_steps += int(minfo["steps_used"])
                m_in = in_region(mS, j)                         # signal read on the IN-region subset
                n_in[j] = int(m_in.sum())
                if n_in[j] >= cfg["min_in"]:
                    iS, iU, iS2 = mS[m_in], mU[m_in], mS2[m_in]
                    clean = True
                else:                                           # too few in-region samples: fall back
                    iS, iU, iS2 = mS, mU, mS2
                    clean = False; fell_back += 1
                per_err[j] = fm_err(net, iS, iU, iS2)
                # --- E3's FLOW tap: fixed-budget counterfactual fit (the one RHM found inverted) ---
                h = len(iS) // 2
                if h >= 2:
                    eb = fm_err(net, iS[h:], iU[h:], iS2[h:])
                    pr = _mlp(cfg["seed"] + 40); pr.load_state_dict(cur)
                    train_steps(pr, torch.optim.Adam(pr.parameters(), lr=cfg["finetune_lr"]),
                                np.concatenate([rS, iS[:h]]), np.concatenate([rU, iU[:h]]),
                                np.concatenate([rS2, iS2[:h]]), cfg["lp_steps"],
                                np.random.default_rng(cfg["seed"] + 9500 + 41 * t + j))
                    ea = fm_err(pr, iS[h:], iU[h:], iS2[h:])
                    lprog[j] = max(eb - ea, 0.0)
                # --- E4's STOCK tap: (err - measured floor)/err.
                # The floor is POOLED ACROSS ROUNDS and the error is not, which is the right split:
                # the aleatoric floor is a STATIONARY property of the region (the noise amplitude
                # never moves) while the error tracks the drift. Neighbour matching still happens
                # strictly WITHIN a round, so pooling adds pairs without ever matching across drift
                # states; only the per-round estimates are averaged, weighted by their sample counts.
                # This is what makes a thin region measurable at all: the smoke had mean in-region
                # counts of 36/7/1/21 against `min_in`, so two regions produced no estimate in most
                # rounds and fell back -- and a fallback on the noise cell is precisely the failure
                # the tap exists to prevent.
                if n_in[j] >= cfg["floor_k"] + 2:
                    f_, nb_, nu_ = matched_pairs_floor(net, iS, iU, iS2)
                    if f_ == f_:
                        floor_hist[j].append((f_, nu_))
                        del floor_hist[j][:-cfg["floor_window"]]
                        nbdist[j] = nb_
                if floor_hist[j] and sum(nn_ for _, nn_ in floor_hist[j]) >= cfg["min_in_total"]:
                    wsum = sum(f_ * nn_ for f_, nn_ in floor_hist[j])
                    nsum = sum(nn_ for _, nn_ in floor_hist[j])
                    floor[j] = wsum / nsum
                    if per_err[j] > 1e-12:
                        red[j] = float(np.clip((per_err[j] - floor[j]) / per_err[j], 0.0, 1.0))
            # EMA the stock tap over rounds (RHM's variance half), then fill the un-measured regions
            # with the MEAN OF THE MEASURED ONES.
            # The previous fallback was 1.0 -- "treat an unlooked-at region as fully reducible" -- and
            # the smoke showed why that is wrong: the two thin regions read 1.00, i.e. MAXIMALLY
            # reducible, and one of them is the pure-noise cell. A fallback of 1.0 is not neutral, it
            # is an attractor pointing at exactly the regions we cannot measure, which is the noisy-TV
            # trap wearing the tap's clothes. The cross-region mean asserts nothing: an unmeasured
            # region is assumed to look like the measured ones, so it neither attracts nor repels.
            for j in range(K):
                if red[j] == red[j]:
                    red_ema[j] = (red[j] if red_ema[j] != red_ema[j]
                                  else cfg["tap_ema"] * red_ema[j] + (1 - cfg["tap_ema"]) * red[j])
            meas = red_ema[red_ema == red_ema]
            red_use = np.where(red_ema == red_ema, red_ema,
                               float(meas.mean()) if len(meas) else 0.5)
            # THE RATE TAP, built from the SAME honest estimator as the level tap.
            # The full ladder produced a dissociation worth chasing: the stock tap is the better
            # MEASUREMENT of reducibility (separation +0.108 against the flow tap's -0.010, i.e. the
            # flow tap is inverted here exactly as RHM found) and allocates far better by every
            # process metric (leak 57.8% -> 19.7%), yet `value` running on the INVERTED flow tap
            # matched the privileged oracle while `value-red` did not. The hypothesis: the two taps
            # measure different quantities that share a name. A fixed-budget counterfactual fit is a
            # RATE -- how fast can error fall here right now -- and (err - floor)/err is a LEVEL --
            # how much of the error here is reducible in principle. Under continuous drift what pays
            # is spending where damage JUST happened, which is a rate.
            # THE TAP IS NEWLY-APPEARED REDUCIBLE ERROR: how much did error rise here since last
            # round, times how much of the error here is reducible at all.
            #
            #     red_delta[j] = max(err_j(t) - err_j(t-1), 0) * red_use[j]
            #
            # A FIRST VERSION USED THE DEVIATION OF `red` ABOVE ITS OWN EMA and was degenerate: it
            # fired in roughly one round in six -- `red` sits below its own smoothed baseline about
            # half the time, and is unmeasured in some rounds besides -- so `_norm_w` fell through to
            # uniform in most rounds and BOTH new arms produced identical numbers (0.3723 each). A
            # tap that is silent most of the time is not a rate, it is a sparse spike detector, and it
            # cannot be compared against a flow tap that emits a value every round.
            # The multiplication by `red_use` is what keeps the floor correction in play: the floor
            # CANCELS from a plain difference of errors (Δ(err - floor) = Δerr), so a bare error-change
            # detector would chase the noise regions, whose error fluctuates most. This form is dense
            # like the flow tap and reducibility-aware like the stock tap, which is exactly the
            # combination the hypothesis says should win.
            red_delta = np.zeros(K)
            for j in range(K):
                if err_prev[j] == err_prev[j] and per_err[j] == per_err[j]:
                    red_delta[j] = max(float(per_err[j] - err_prev[j]), 0.0) * float(red_use[j])
            err_prev = per_err.copy()
            visits = fm_visits(net)
            # ground-truth "stale AND reducible" for the ORACLE (privileged) only
            stale_mask = np.array([1.0 if (REG[j]["reducible"] and per_err[j] > cfg["err_floor"])
                                   else 0.0 for j in range(K)])

            # ---------- the policy: WHERE to spend this round's collection budget ----------
            if pname == "uniform":
                w = None
            elif pname == "oracle":
                tgt = stale_mask * per_err * visits
                w = _norm_w(tgt) if tgt.sum() > 1e-9 else None
            elif pname == "value":                              # E3's arm: FLOW tap x relevance
                tv = lprog * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None    # no-op floor (S2 lesson)
            elif pname == "value-red":                          # E4's arm: STOCK tap x relevance
                tv = red_use * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None
            elif pname == "lprog-only":
                w = _norm_w(lprog) if lprog.sum() > 1e-12 else None
            elif pname == "reducible-only":                     # the STOCK tap alone (a LEVEL)
                w = _norm_w(red_use) if red_use.sum() > 1e-12 else None
            elif pname == "reddelta-only":                      # the same tap as a RATE
                w = _norm_w(red_delta) if red_delta.sum() > 1e-12 else None
            elif pname == "value-reddelta":                     # RATE x relevance
                tv = red_delta * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None
            elif pname == "visits-only":
                w = _norm_w(visits)
            elif pname == "error-only":
                w = _norm_w(per_err)
            else:
                raise ValueError(pname)

            rng_c = np.random.default_rng(cfg["seed"] + 11000 + 97 * t)
            in_share = np.full(K, np.nan)
            if w is None:                                       # uniform: spread over all regions
                counts = np.full(K, cfg["budget"] // K, int); counts[0] += cfg["budget"] - counts.sum()
            else:
                counts = np.floor(w * cfg["budget"]).astype(int)
                counts[int(np.argmax(w))] += cfg["budget"] - counts.sum()
            alloc = counts.astype(float)
            for j in range(K):
                if counts[j] <= 0:
                    continue
                cS, cU, cS2, cinfo = collect_toward(env, REG[j]["center"], int(counts[j]), net, rng_c, 200 + j)
                coll_steps += int(cinfo["steps_used"])
                in_share[j] = float(in_region(cS, j).mean())
                d = (cS, cU, cS2)
                bufs[j] = d if bufs[j] is None else tuple(
                    np.concatenate([bufs[j][i], d[i]])[-cfg["buf_cap"]:] for i in range(3))

            # ---------- fine-tune the FM on everything currently believed valid ----------
            parts = [(rS, rU, rS2)] + [b for b in bufs.values() if b is not None]
            train_steps(net, opt, np.concatenate([p[0] for p in parts]),
                        np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
                        cfg["finetune_steps"], np.random.default_rng(cfg["seed"] + 12000 + t))

            # ---------- grade (off-budget, the experimenter's fixed instrument) ----------
            snap = copy.deepcopy(net)
            # per-region FM error on a fixed matched-reach probe set (built once, below)
            reg_err = [fm_err(snap, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                       for j in range(K)]
            if pname in render_pols and t in render_rounds:
                render_packs.append(render_pack(pname, t, snap, b_state))
            bal = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                       np.random.default_rng(cfg["seed"] + 7001)), H, env)
            rec = {"round": t, "alloc": alloc.tolist(), "in_share": in_share.tolist(),
                   "per_err_mon": per_err.tolist(), "reg_err": reg_err, "visits": visits.tolist(),
                   "lprog": lprog.tolist(), "stale_mask": stale_mask.tolist(), "b_state": b_state.tolist(),
                   "ballistic_cem": bal, "mon_steps": mon_steps, "coll_steps": coll_steps,
                   # E4's new per-round instrument trace
                   "red": red.tolist(), "red_ema": red_ema.tolist(), "floor": floor.tolist(),
                   "red_delta": red_delta.tolist(),
                   "nbdist": nbdist.tolist(), "n_in": n_in.tolist(), "fell_back": int(fell_back)}
            if (t % cfg["reactive_every"] == 0) or (t == T - 1):
                rec["reactive"] = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                                       np.random.default_rng(cfg["seed"] + 7000)), 1, env)
            hist.append(rec)
            astr = "/".join(str(int(x)) for x in alloc)
            print(f"[{pname:>14s} r{t:02d}] alloc={astr:>18s} ball={bal:.4f}"
                  + (f" reac={rec['reactive']:.4f}" if "reactive" in rec else "")
                  + f"  mon/coll={mon_steps}/{coll_steps}"
                  + "  regERR=[" + " ".join(f"{e:.3f}" for e in reg_err) + "]"
                  + "  red=[" + " ".join(f"{v:.2f}" for v in red_use) + "]"
                  + "  n_in=[" + " ".join(str(int(v)) for v in n_in) + "]", flush=True)
        return hist

    # ---- fixed per-region PROBE sets (off-budget grader): matched-reach transitions in each region,
    # gathered with a ref FM trained on the DRIFTED world (so the reaches are competent, not deflected
    # by a stale model) and partitioned by region. Plus a matched-FM CEILING (trained on abundant
    # competent on-reach reaches) to normalise the value-relevant region-A error.
    print("[probe] training ref FM (drifted world) + building per-region grader probes ...", flush=True)
    ref_net = _mlp(cfg["seed"] + 43)
    dS, dU, dS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 401))
    PB = []
    probeS, probeU, probeS2, _ = collect_toward(env_drift0, G, cfg["probe_pool_n"], ref_net,
                                                np.random.default_rng(cfg["seed"] + 610), 900)
    # populate the off-reach / distractor probes by reaching toward each non-A region centre too
    for j in range(1, K):
        oS, oU, oS2, _ = collect_toward(env_drift0, REG[j]["center"], cfg["probe_pool_n"] // 2, ref_net,
                                        np.random.default_rng(cfg["seed"] + 901 + j), 901 + j)
        probeS = np.concatenate([probeS, oS]); probeU = np.concatenate([probeU, oU])
        probeS2 = np.concatenate([probeS2, oS2])
    for j in range(K):
        m = in_region(probeS, j)
        PB.append((probeS[m], probeU[m], probeS2[m]))
        print(f"[probe]   region {names[j]}: {int(m.sum())} probe transitions", flush=True)
    # A HARD GATE. An empty in-region probe set makes `fm_err` NaN, which propagates into `reg_err`,
    # the ladder's summary and the floor certification -- and a table of NaNs reads as "no effect"
    # rather than "no instrument". E5's calibration hit exactly this (0/500 in one region), so the
    # check is explicit here rather than left to a reader noticing a blank column.
    # ADAPTIVE TOP-UP for any region the fixed pool under-fills. Seed 2 of the first full ladder
    # aborted here with probe counts [2365, 2292, 10, 174]: the off-reach reducible region is off the
    # eval path by design, which is exactly what makes it hard to AIM at, and the hit rate varies
    # enough between seeds that a fixed pool size cannot be tuned once for all of them. Collect extra
    # chunks aimed at the thin region until it clears `min_probe` or the tries run out. Off-budget
    # (the experimenter's ruler), so this costs wall-clock and not the meter.
    for j in range(K):
        tries = 0
        while len(PB[j][0]) < cfg["min_probe"] and tries < cfg["probe_max_tries"]:
            tries += 1
            oS, oU, oS2, _ = collect_toward(env_drift0, REG[j]["center"], cfg["probe_pool_n"],
                                            ref_net,
                                            np.random.default_rng(cfg["seed"] + 9600 + 31 * j + tries),
                                            9600 + 31 * j + tries)
            m = in_region(oS, j)
            PB[j] = (np.concatenate([PB[j][0], oS[m]]), np.concatenate([PB[j][1], oU[m]]),
                     np.concatenate([PB[j][2], oS2[m]]))
            print(f"[probe]   top-up {names[j]} (try {tries}): +{int(m.sum())} -> {len(PB[j][0])}",
                  flush=True)
    geom_gate["probe_n"] = [int(len(PB[j][0])) for j in range(K)]
    geom_gate["pass_probe"] = all(nn_ >= cfg["min_probe"] for nn_ in geom_gate["probe_n"])
    print(f"[probe] GATE  in-region probe counts={geom_gate['probe_n']} (>={cfg['min_probe']} each) "
          f"-> {'PASS' if geom_gate['pass_probe'] else 'FAIL'}", flush=True)
    if not geom_gate["pass_probe"] and not cfg.get("certify_only"):
        raise RuntimeError(f"empty/thin in-region probe set: {geom_gate['probe_n']} -- the graded "
                           f"readout would be NaN. Fix the geometry before running the ladder.")
    # matched ceiling: FM trained on abundant competent ON-REACH reaches in the drifted world. Valid
    # floor for the on-reach regions (A, C); undertrained by construction for the off-reach regions
    # (B, D) -- you never reach there -- so those are reported but caveated (E2's lesson).
    ceilS, ceilU, ceilS2, _ = collect_toward(env_drift0, G, cfg["ceil_pool_n"], ref_net,
                                             np.random.default_rng(cfg["seed"] + 620), 910)
    ceil_net = _mlp(cfg["seed"] + 44)
    train_steps(ceil_net, torch.optim.Adam(ceil_net.parameters(), lr=cfg["fm_lr"]),
                ceilS, ceilU, ceilS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 421))
    ceil_err = [fm_err(ceil_net, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                for j in range(K)]
    print("[probe] matched-ceiling per-region err: "
          + " ".join(f"{names[j]}={ceil_err[j]:.3f}" for j in range(K)), flush=True)

    # ================================================================= #
    # THE FLOOR ESTIMATOR'S CERTIFICATION GATE (in-run, on the off-budget probe sets).
    # A measured floor is worth having only if it measures the floor, and matched pairs is biased UP
    # by local curvature. Three independent references, none of which the agent's tap may use:
    #   * `noise_amp` is 0 for every CURL region by construction -- so the estimator's output there IS
    #     its curvature bias. That number is the instrument's noise level.
    #   * a MATCHED FM cannot beat the floor, so floor_hat <= ceil_err must hold per region.
    #   * ORDERING: noise regions strictly above curl regions is the only property the tap consumes.
    # Also reports the raw-Δs variant, to show what running on the residual buys.
    # ================================================================= #
    floor_cert = {"region": names, "kind": [r["kind"] for r in REG], "ceil_err": ceil_err}
    for tag_, net_ in (("ceil", ceil_net), ("ref", ref_net)):
        fr, fd, nb, ferr = [], [], [], []
        for j in range(K):
            if len(PB[j][0]) < cfg["floor_k"] + 2:
                fr.append(float("nan")); fd.append(float("nan")); nb.append(float("nan"))
                ferr.append(float("nan")); continue
            a, d1, _ = matched_pairs_floor(net_, *PB[j], on="residual")
            b, _, _ = matched_pairs_floor(net_, *PB[j], on="delta")
            fr.append(a); fd.append(b); nb.append(d1); ferr.append(fm_err(net_, *PB[j]))
        floor_cert[f"floor_residual_{tag_}"] = fr
        floor_cert[f"floor_delta_{tag_}"] = fd
        floor_cert[f"nbdist_{tag_}"] = nb
        floor_cert[f"fm_err_{tag_}"] = ferr
    fr = floor_cert["floor_residual_ceil"]
    cur_idx = [j for j in range(K) if REG[j]["reducible"]]
    nz_idx = [j for j in range(K) if not REG[j]["reducible"]]

    # THE BATCH-SIZE SWEEP, and it is the gate that actually matters. Everything above is measured on
    # ~600-transition probe sets, but the LOOP reads the tap from an in-region monitor batch of ~18.
    # Matched-pairs bias grows as the batch thins (neighbours get further away, so the "local" mean
    # drifts toward the batch mean, floor -> err and red -> 0 for EVERY region). A certification that
    # passes at m=600 and is never checked at m=18 would certify an instrument the loop does not use.
    # So: re-estimate at the sizes the loop will actually see, and report the SEPARATION at each.
    sw_rng = np.random.default_rng(cfg["seed"] + 7778)
    floor_cert["batch_sweep_m"] = list(cfg["floor_batch_sweep"])
    floor_cert["batch_sweep"] = {}
    for j in range(K):
        row = []
        for m_ in cfg["floor_batch_sweep"]:
            if len(PB[j][0]) < max(cfg["floor_k"] + 2, m_):
                row.append(float("nan")); continue
            sel = sw_rng.permutation(len(PB[j][0]))[:m_]
            f_, _, _ = matched_pairs_floor(ceil_net, PB[j][0][sel], PB[j][1][sel], PB[j][2][sel],
                                           on="residual")
            row.append(f_)
        floor_cert["batch_sweep"][names[j]] = row
    seps = []
    for i, m_ in enumerate(cfg["floor_batch_sweep"]):
        cu = [v for v in (floor_cert["batch_sweep"][names[j]][i] for j in cur_idx) if v == v]
        nz = [v for v in (floor_cert["batch_sweep"][names[j]][i] for j in nz_idx) if v == v]
        seps.append(float(np.mean(nz) - np.mean(cu)) if cu and nz else float("nan"))
    floor_cert["batch_sweep_separation"] = seps
    print("\n[floor-cert] BATCH-SIZE SWEEP -- the loop reads ~`mon_n * in_share` samples, not 600",
          flush=True)
    print("    " + "region".rjust(8) + "kind".rjust(9)
          + "".join(f"m={m_}".rjust(10) for m_ in cfg["floor_batch_sweep"]), flush=True)
    for j in range(K):
        print("    " + names[j].rjust(8) + REG[j]["kind"].rjust(9)
              + "".join(f"{v:.3f}".rjust(10) for v in floor_cert["batch_sweep"][names[j]]), flush=True)
    print("    " + "SEPARATION".rjust(17)
          + "".join(f"{v:+.3f}".rjust(10) for v in seps), flush=True)
    floor_cert["pass_batch_sweep"] = bool(
        all((v != v) or v > cfg["min_separation"] for v in seps))
    print(f"    all sweep separations > {cfg['min_separation']} -> "
          f"{'PASS' if floor_cert['pass_batch_sweep'] else 'FAIL'}   "
          f"(if this fails at the small end, the tap is dead IN THE LOOP however well it "
          f"certifies on dense probes -- raise `mon_n` or cut K further)", flush=True)
    floor_cert["bias_curl_regions"] = float(np.nanmean([fr[j] for j in cur_idx])) if cur_idx else float("nan")
    floor_cert["floor_noise_regions"] = float(np.nanmean([fr[j] for j in nz_idx])) if nz_idx else float("nan")
    floor_cert["separation"] = floor_cert["floor_noise_regions"] - floor_cert["bias_curl_regions"]
    floor_cert["pass_ordering"] = bool(floor_cert["separation"] > 0)
    floor_cert["pass_below_ceiling"] = bool(all(
        (fr[j] != fr[j]) or (ceil_err[j] != ceil_err[j])
        or fr[j] <= ceil_err[j] * cfg["ceiling_tol"]
        for j in range(K)))
    print("\n[floor-cert] per-region matched-pairs floor (graded with the MATCHED-CEILING FM)",
          flush=True)
    print(f"    {'region':>8s} {'kind':>8s} {'floor(resid)':>12s} {'floor(Δs)':>10s} "
          f"{'ceil_err':>9s} {'nb-dist':>8s}", flush=True)
    for j in range(K):
        print(f"    {names[j]:>8s} {REG[j]['kind']:>8s} {fr[j]:12.4f} "
              f"{floor_cert['floor_delta_ceil'][j]:10.4f} {ceil_err[j]:9.4f} "
              f"{floor_cert['nbdist_ceil'][j]:8.3f}", flush=True)
    print(f"    GATE  bias(curl, known noise-free)={floor_cert['bias_curl_regions']:.4f}  "
          f"floor(noise)={floor_cert['floor_noise_regions']:.4f}  "
          f"separation={floor_cert['separation']:+.4f} -> "
          f"{'PASS' if floor_cert['pass_ordering'] else 'FAIL'} | "
          f"floor<=ceiling -> {'PASS' if floor_cert['pass_below_ceiling'] else 'FAIL'}", flush=True)

    # `certify_only` stops here: the two gates (geometry + floor estimator) are exactly the ones the
    # ladder would use, so certifying them is a strict prefix of the run rather than a separate
    # re-implementation that could drift from it.
    if cfg.get("certify_only"):
        cert_out = {"config": cfg, "region_names": names,
                    "region_kinds": [r["kind"] for r in REG],
                    "region_reducible": [bool(r["reducible"]) for r in REG],
                    "region_on_reach": [bool(r["on_reach"]) for r in REG],
                    "region_centers": {r["name"]: np.asarray(r["center"]).tolist() for r in REG},
                    "region_visitation": reg_vis.tolist(), "geom_gate": geom_gate,
                    "ceil_err": ceil_err, "floor_certification": floor_cert}
        outdir = os.path.join(DATA_DIR, "metered_repair", "certify", cfg["tag"])
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(cert_out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()
        ok = (geom_gate["pass_A"] and geom_gate["pass_on_noise"] and geom_gate["pass_off"]
              and geom_gate["pass_probe"] and floor_cert["pass_ordering"]
              and floor_cert["pass_below_ceiling"] and floor_cert["pass_batch_sweep"])
        print(f"\n[certify] ALL GATES {'PASS' if ok else 'FAIL'} -- wrote {outdir}", flush=True)
        return {"results": cert_out, "render": None}

    results = {}
    for pname in cfg["policies"]:
        print(f"\n=== policy: {pname} ===", flush=True)
        results[pname] = run_policy(pname)

    # ================================================================= #
    # summary
    # ================================================================= #
    def auc(pname, key):
        v = [r[key] for r in results[pname] if key in r and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    def reg_auc(pname, j):
        v = [r["reg_err"][j] for r in results[pname] if r["reg_err"][j] == r["reg_err"][j]]
        return float(np.mean(v)) if v else float("nan")

    a_idx = names.index("A") if "A" in names else 0
    print("\n=== summary: mean over rounds (lower=better) ===", flush=True)
    print(f"    {'policy':>12s}  {'ballistic':>9s} {'reactive':>9s} {'A-err':>7s} "
          + " ".join(f"{nm}-err" for nm in names), flush=True)
    # E4: the allocation columns the repair is supposed to move. `leak` is the fraction of the
    # collection budget spent on the IRREDUCIBLE regions (RHM's 14.2% -> 7.3%, E3's 29%); `A_share_sd`
    # is the round-to-round sd of the share sent to A (RHM's "tree-share sd", 0.396 -> 0.047, the 8x
    # variance drop that was most of the repair's value).
    noise_idx = [j for j in range(K) if not REG[j]["reducible"]]

    def alloc_shares(pname):
        A = np.array([r["alloc"] for r in results[pname]], float)
        tot = np.maximum(A.sum(1, keepdims=True), 1e-9)
        sh = A / tot
        return sh

    summary = {}
    for pname in cfg["policies"]:
        sh = alloc_shares(pname)
        row = {"ballistic_auc": auc(pname, "ballistic_cem"), "reactive_auc": auc(pname, "reactive"),
               "regA_err_auc": reg_auc(pname, a_idx),
               "reg_err_auc": [reg_auc(pname, j) for j in range(K)],
               "mon_steps_total": int(sum(r["mon_steps"] for r in results[pname])),
               "coll_steps_total": int(sum(r["coll_steps"] for r in results[pname])),
               "final_reg_err": results[pname][-1]["reg_err"],
               "share_mean": sh.mean(0).tolist(),
               "leak_irreducible": float(sh[:, noise_idx].sum(1).mean()) if noise_idx else 0.0,
               "A_share": float(sh[:, a_idx].mean()),
               "A_share_sd": float(sh[:, a_idx].std()),
               "red_mean": np.nanmean(np.array([r["red_ema"] for r in results[pname]], float),
                                      axis=0).tolist(),
               "red_delta_mean": np.nanmean(np.array([r["red_delta"] for r in results[pname]], float),
                                            axis=0).tolist(),
               "lprog_mean": np.nanmean(np.array([r["lprog"] for r in results[pname]], float),
                                        axis=0).tolist(),
               "floor_mean": np.nanmean(np.array([r["floor"] for r in results[pname]], float),
                                        axis=0).tolist(),
               "n_in_mean": np.mean(np.array([r["n_in"] for r in results[pname]], float),
                                    axis=0).tolist()}
        row["monitor_collect_ratio"] = row["mon_steps_total"] / max(row["coll_steps_total"], 1)
        summary[pname] = row
        print(f"    {pname:>12s}  {row['ballistic_auc']:9.4f} {row['reactive_auc']:9.4f} "
              f"{row['regA_err_auc']:7.4f} " + " ".join(f"{e:6.3f}" for e in row["reg_err_auc"]), flush=True)

    # the anti-subsidy headline: monitor:collect ratio is O(1), not the 22x it was under teleport
    mr = np.nanmean([summary[p]["monitor_collect_ratio"] for p in cfg["policies"]])
    print(f"\n[anti-subsidy] mean monitor:collect step ratio = {mr:.2f}  "
          f"(S2 teleport was 22x; on-policy survey is metered)", flush=True)

    # ================================================================= #
    # E4's two headline tables
    # ================================================================= #
    print("\n=== (1) THE REPAIR: flow tap vs stock tap, everything else identical ===", flush=True)
    print(f"    {'policy':>14s} {'A-err':>7s} {'ballistic':>9s} {'leak(irred)':>11s} "
          f"{'A-share':>8s} {'A-share sd':>10s}", flush=True)
    for pname in cfg["policies"]:
        r = summary[pname]
        print(f"    {pname:>14s} {r['regA_err_auc']:7.4f} {r['ballistic_auc']:9.4f} "
              f"{r['leak_irreducible']:11.1%} {r['A_share']:8.1%} {r['A_share_sd']:10.3f}", flush=True)
    for a, b, lbl in (("value-red", "value", "STOCK - FLOW  (the repair)"),
                      ("value-reddelta", "value-red", "RATE - LEVEL (same estimator)"),
                      ("value-reddelta", "value", "RATE(honest) - RATE(counterfactual fit)"),
                      ("value-red", "visits-only", "CONJUNCTION - relevance alone"),
                      ("value-red", "reducible-only", "CONJUNCTION - reducibility alone"),
                      ("value-red", "oracle", "vs the privileged oracle")):
        if a in summary and b in summary:
            d = summary[a]["regA_err_auc"] - summary[b]["regA_err_auc"]
            print(f"    [{lbl:>32s}]  ΔA-err = {d:+.4f}  (negative = {a} better)", flush=True)

    # The tap's own ordering is the thing the whole cut turns on: reducibility must come out HIGH on
    # the reducible regions and LOW on the irreducible ones. Reported from the arm that uses it.
    ref_pol = "value-red" if "value-red" in summary else cfg["policies"][0]
    print(f"\n=== (2) THE TAPS, per region (from `{ref_pol}`; kind in brackets) ===", flush=True)
    print("    region      kind   mean red (stock)   mean lprog (flow)   mean floor   mean n_in",
          flush=True)
    for j in range(K):
        print(f"    {names[j]:>8s} {REG[j]['kind']:>8s} {summary[ref_pol]['red_mean'][j]:14.3f} "
              f"{summary[ref_pol]['lprog_mean'][j]:19.4f} {summary[ref_pol]['floor_mean'][j]:12.4f} "
              f"{summary[ref_pol]['n_in_mean'][j]:11.1f}", flush=True)
    red_red = [summary[ref_pol]["red_mean"][j] for j in range(K) if REG[j]["reducible"]]
    red_irr = [summary[ref_pol]["red_mean"][j] for j in range(K) if not REG[j]["reducible"]]
    lp_red = [summary[ref_pol]["lprog_mean"][j] for j in range(K) if REG[j]["reducible"]]
    lp_irr = [summary[ref_pol]["lprog_mean"][j] for j in range(K) if not REG[j]["reducible"]]
    tap_order = {"red_reducible": float(np.nanmean(red_red)), "red_irreducible": float(np.nanmean(red_irr)),
                 "lprog_reducible": float(np.nanmean(lp_red)), "lprog_irreducible": float(np.nanmean(lp_irr))}
    tap_order["red_separation"] = tap_order["red_reducible"] - tap_order["red_irreducible"]
    tap_order["lprog_separation"] = tap_order["lprog_reducible"] - tap_order["lprog_irreducible"]
    print(f"    STOCK separation (reducible - irreducible) = {tap_order['red_separation']:+.3f}   "
          f"FLOW separation = {tap_order['lprog_separation']:+.4f}   "
          f"(RHM's flow tap came out NEGATIVE here; positive = the tap points the right way)",
          flush=True)

    # the relevance test: does concentrating on the value-relevant region A beat spreading, and does
    # a value/relevance-aware policy ABANDON the off-reach distractors (leave their error high while
    # keeping A low)? off-err = mean FM error over the off-reach reducible distractors.
    off_idx = [j for j in range(K) if not REG[j]["on_reach"] and REG[j]["reducible"]]
    if off_idx:
        def off_err_auc(pname):
            return float(np.nanmean([reg_auc(pname, j) for j in off_idx]))
        print("\n=== relevance test (the retracted S2 claim, now metered) ===", flush=True)
        print(f"    (off-reach reducible distractors: {[names[j] for j in off_idx]})", flush=True)
        for p in cfg["policies"]:
            summary[p]["off_err_auc"] = off_err_auc(p)
            print(f"    {p:>12s}: A-err={summary[p]['regA_err_auc']:.4f}  "
                  f"off-err={off_err_auc(p):.4f}  ballistic={summary[p]['ballistic_auc']:.4f}", flush=True)

    if "uniform" in results and "oracle" in results:
        for key, lbl in (("ballistic_cem", "ballistic"), ("regA_err_auc", "A-FM-err")):
            u = auc("uniform", "ballistic_cem") if key == "ballistic_cem" else summary["uniform"]["regA_err_auc"]
            o = auc("oracle", "ballistic_cem") if key == "ballistic_cem" else summary["oracle"]["regA_err_auc"]
            den = u - o
            print(f"\n  [{lbl}] uniform={u:.4f} oracle={o:.4f} spread={den:+.4f}", flush=True)
            for pname in cfg["policies"]:
                val = auc(pname, "ballistic_cem") if key == "ballistic_cem" else summary[pname]["regA_err_auc"]
                g = (u - val) / den if abs(den) > 1e-9 else float("nan")
                summary[pname][lbl + "_gap_closed"] = g
                print(f"      {pname:>12s}  gap-closed={g:+.2f}", flush=True)

    # ceiling-normalised region-A recovery (the value-relevant, sighted readout): fraction of the
    # stale->matched gap closed on region A, averaged over rounds. 1.0 = matched-FM competence.
    if "A" in names:
        cA = ceil_err[a_idx]
        for pname in cfg["policies"]:
            st = results[pname][0]["reg_err"][a_idx]
            den = st - cA
            rec = [((st - r["reg_err"][a_idx]) / den if abs(den) > 1e-9 else float("nan"))
                   for r in results[pname]]
            summary[pname]["regA_recovery_auc"] = float(np.nanmean(rec))
        print(f"\n=== region-A ceiling-normalised recovery (ceil={ceil_err[a_idx]:.3f}, "
              f"higher=better) ===", flush=True)
        for pname in cfg["policies"]:
            print(f"    {pname:>12s}  {summary[pname]['regA_recovery_auc']:+.2f}", flush=True)

    out = {"config": cfg, "region_names": names,
           "region_kinds": [r["kind"] for r in REG],
           "region_reducible": [bool(r["reducible"]) for r in REG],
           "region_on_reach": [bool(r["on_reach"]) for r in REG],
           "region_centers": {r["name"]: np.asarray(r["center"]).tolist() for r in REG},
           "region_visitation": reg_vis.tolist(), "geom_gate": geom_gate,
           "schedule": {str(k): v for k, v in schedule.items()},
           "P0": P0.tolist(), "G": G.tolist(), "ceil_err": ceil_err,
           "floor_certification": floor_cert, "tap_order": tap_order,
           "results": results, "summary": summary}
    outdir = os.path.join(DATA_DIR, "metered_repair", "floor_tap", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)

    # render packs go to their OWN file so `results.json` stays byte-comparable with the
    # published ladder_s{0,1,2} artifacts (`render_ladder.py` reads this one).
    render = None
    if render_packs:
        render = {"meta": {"tag": cfg["tag"], "seed": cfg["seed"], "frame_skip": fs, "plan_H": H,
                           "n_links": n, "link_lengths": Ls.tolist(), "region_k": cfg["region_k"],
                           "P0": P0.tolist(), "G": G.tolist(), "ceil_err": ceil_err,
                           "region_names": names, "ceil_regA": ceil_err[a_idx],
                           "regions": [{"name": r["name"], "center": np.asarray(r["center"]).tolist(),
                                        "sigma": r["sigma"], "kind": r["kind"],
                                        "reducible": bool(r["reducible"]),
                                        "on_reach": bool(r["on_reach"])} for r in REG]},
                  "packs": render_packs}
        with open(os.path.join(outdir, "render_pack.json"), "w") as fh:
            json.dump(render, fh, cls=NumpyEncoder)
        print(f"[save] wrote {len(render_packs)} render packs to {outdir}/render_pack.json", flush=True)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out, "render": render}


@app.local_entrypoint()
def floor_tap(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    policies: str = _ALL_POLICIES,
    rounds: int = 26,
    # --- E4's new knobs ---
    # `floor_k` is small ON PURPOSE. The matched-pairs bias is the curvature of the differenced field
    # over the neighbourhood, so a TIGHT neighbourhood is what keeps the estimator honest; the variance
    # that costs is then averaged away over the batch and over rounds (`tap_ema`). With an in-region
    # batch of ~20 a large k would make the "local" mean nearly the batch mean, which drives floor->err
    # and red->0 for every region -- killing the separation, not just adding noise.
    floor_k: int = 4,                         # matched-pairs neighbourhood size
    floor_on: str = "residual",               # "residual" (the FM's own error) | "delta" (raw Δs)
    floor_max_n: int = 600,                   # cap on the O(m^2) distance matrix
    tap_ema: float = 0.5,                     # EMA weight on the previous stock-tap value
    eval_spread: bool = True,                 # spread eval goals ACROSS the sector (places region C)
    drift_mode: str = "ou",                   # "ou" = continuous walk (default); "schedule" = S2-style
    # E3's drift rate, KEPT. The slower walk that would give the near-saturated control instrument
    # dynamic range is a real follow-up, but halving the damage while the tap ALSO changes would
    # confound the repair with the regime -- the exact confound RHM's own necessity sweep had to
    # apologise for ("the drive changed at the same time as the sweep"). One variable at a time.
    ou_sigma: float = 1.6,
    b_lo: float = 2.0,
    b_hi: float = 10.0,
    drift_every: int = 8,                     # (schedule mode only)
    drift_cycle: str = "0,0,1",              # (schedule mode only) region indices among `reducible`
    b1: float = 6.0,                          # curl gain reference (E2's design point)
    noise_amp: float = 3.0,                   # aleatoric torque noise in the C/D regions (< gear=8)
    # --- geometry ---
    region_sigma: float = 0.12,
    region_k: float = 1.5,                    # in-region iff ‖tip-center‖ < region_k*sigma
    geom_samples: int = 1500,
    rad_min: float = 0.80,                     # regions must sit beyond this radius (extended/tame)
    pref_width: float = 1.2,                  # half-width (rad) of the on-reach eval sector (E3: 0.5)
    vis_steps: int = 12,                      # tip-path interpolation steps for path-visitation
    # 0.25, not E3's 0.12, and the reason is E4's own change (C). Widening the eval sector to make a
    # second visited site exist necessarily raises occupancy EVERYWHERE off the old narrow cone, so a
    # threshold calibrated against a +-0.5 rad cone is not the same test against +-1.2 rad. What the
    # design needs is that `visits` be much SMALLER at the distractors than at A, and cert_v4 measures
    # that ratio at 1.00 vs 0.01/0.18 -- 5.5x even for the worst one. The ratio is reported below.
    off_vis_max: float = 0.25,                # off-reach distractors: REPORTED visitation ceiling
    on_vis_min: float = 0.40,                 # on-reach regions: REPORTED visitation floor
    min_sep: float = 0.30,                    # min tip-space separation between any two regions
    # K = 5, down from E3's 6, and it is the anti-subsidy meter that sets it. Every region costs
    # `mon_n` CHARGED survey steps per round, so K multiplies the monitor bill while the collection
    # budget stays scarce by design. E3 ran K=6 x mon_n=40 against budget=120 (ratio 1.84x). The floor
    # tap needs a DENSER in-region batch than the flow tap does (see `floor_k`), so the trade is made
    # explicitly: drop to the minimum K that still supports every discrimination -- A, C (the new
    # visited-but-irreducible cell), two off-reach reducible distractors to catch `lprog-only`, one
    # off-reach noise distractor to catch `error-only` -- and spend the freed monitor steps on batch
    # density instead. That is exactly S1/S2's original A/B/C/D 2x2 plus one extra B.
    n_off_red: int = 1,                       # OFF-reach REDUCIBLE distractor (catches lprog-only)
    n_off_noise: int = 1,                     # OFF-reach NOISE distractor (catches error-only)
    n_on_noise: int = 1,                      # ON-reach NOISE = region C, the visited-but-irreducible
                                              # cell (E3: 0, unplaceable in its narrow eval cone)
    # --- arm geometry (E2 / 4c-arm design point) ---
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- on-policy collection ---
    ep_len: int = 14,
    op_q_range: float = 0.25,
    n_par: int = 16,
    sigma_u: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14,
    wrap_limit: float = 3.0,
    goal_jit: float = 0.05,
    # --- per-round loop budget (SCARCE, on purpose: repair must be incomplete so allocation is
    #     zero-sum; monitoring is CHARGED separately) ---
    budget: int = 120,                        # directed collection transitions per round (E3's, kept)
    # 5 regions x 60 = 300 charged survey steps against a 120-step collection budget -> ratio ~2.5x,
    # above E3's 1.84x and reported as such. The anti-subsidy claim is that looking is CHARGED and
    # O(1), not that the constant is 1.84 -- S2's pathology was 22x and free. The ratio is identical
    # across arms by construction (both taps are computed for every arm), so it cannot favour one.
    mon_n: int = 60,                          # survey transitions per region per round (metered)
    min_in: int = 6,                          # min in-region survey samples to trust the signal
    floor_window: int = 6,                    # rounds of floor estimates to pool per region
    min_in_total: int = 14,                   # pooled in-region samples before trusting a floor
    buf_cap: int = 260,                       # ring buffer per region: recent data tracks the walk
    lp_steps: int = 400,
    replay_n: int = 2500,
    finetune_lr: float = 3e-4,
    finetune_steps: int = 700,
    reactive_every: int = 4,
    err_floor: float = 0.05,
    value_floor: float = 1e-4,
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    probe_pool_n: int = 2000,
    ceil_pool_n: int = 4000,
    # --- control eval ---
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
    reach_pad: float = 0.90,                  # reach-distance filter, calibrated in cert_v2
    min_density_frac: float = 0.01,           # min share of reach endpoints inside a region's gate
    min_probe: int = 40,                      # min in-region probe transitions per region (HARD gate)
    probe_max_tries: int = 6,                 # adaptive top-up chunks for a thin region
    floor_batch_sweep: str = "600,200,60,24,16",  # batch sizes to re-certify the floor at
    min_separation: float = 0.05,             # min reducible-vs-irreducible floor separation
    # For an IRREDUCIBLE region the floor and the matched-FM error ARE the same quantity, so
    # floor ~= ceil_err is expected there and a small overshoot is estimator noise in the SAFE
    # direction: `red` clips to 0, which is the correct answer for a noise region. cert_v1 came
    # in at 1.068 on the on-reach noise region against a 1.05 tolerance -- too tight, not broken.
    ceiling_tol: float = 1.20,
    # --- rendering (OFF by default: with render_rounds="" nothing is called and every prior
    #     result reproduces exactly). Rounds are 0-indexed; -1 = the stale base FM. ---
    render_rounds: str = "",
    render_policies: str = "value-red",
    render_n: int = 0,                        # 0 = every eval reach (so `miss` == the graded median)
    certify_only: bool = False,               # stop after the geometry + floor gates (see `certify_floor`)
):
    import os

    pol = [p for p in policies.split(",") if p.strip()]
    if quick:
        rounds = 10; pool_n = 3000; fm_steps = 1500; finetune_steps = 300
        lp_steps = 150; replay_n = 1200; fm_hidden = 128; fm_layers = 2; n_eval = 12
        k_shoot = 256; cem_iters = 4; collect_k_shoot = 128; collect_cem_iters = 3
        budget = 120; mon_n = 60; probe_pool_n = 800; ceil_pool_n = 1500; geom_samples = 600; n_par = 8
        if policies == _ALL_POLICIES:                            # only if not explicitly overridden
            # the smoke's job is the two GATES (geometry places C; the floor estimator separates) plus
            # the one contrast the whole cut turns on -- flow vs stock at matched everything.
            pol = ["uniform", "value", "value-red", "visits-only"]
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, policies=pol, rounds=rounds, drift_mode=drift_mode,
        floor_k=floor_k, floor_on=floor_on, floor_max_n=floor_max_n, tap_ema=tap_ema,
        eval_spread=eval_spread,
        ou_sigma=ou_sigma, b_lo=b_lo, b_hi=b_hi, drift_every=drift_every,
        drift_cycle=[int(x) for x in drift_cycle.split(",") if x.strip()], b1=b1, noise_amp=noise_amp,
        region_sigma=region_sigma, region_k=region_k, geom_samples=geom_samples, rad_min=rad_min,
        pref_width=pref_width, vis_steps=vis_steps, off_vis_max=off_vis_max, on_vis_min=on_vis_min,
        min_sep=min_sep, n_off_red=n_off_red, n_off_noise=n_off_noise, n_on_noise=n_on_noise,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, q_range=q_range,
        v_explore=v_explore, ep_len=ep_len, op_q_range=op_q_range, n_par=n_par, sigma_u=sigma_u,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        collect_replan_every=collect_replan_every, wrap_limit=wrap_limit, goal_jit=goal_jit,
        budget=budget, mon_n=mon_n, min_in=min_in, floor_window=floor_window,
        min_in_total=min_in_total, buf_cap=buf_cap, lp_steps=lp_steps,
        replay_n=replay_n, finetune_lr=finetune_lr, finetune_steps=finetune_steps,
        reactive_every=reactive_every, err_floor=err_floor, value_floor=value_floor,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pool_n=pool_n, probe_pool_n=probe_pool_n, ceil_pool_n=ceil_pool_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        reach_tries=reach_tries, reach_pad=reach_pad, min_probe=min_probe,
        min_density_frac=min_density_frac,
        probe_max_tries=probe_max_tries,
        floor_batch_sweep=[int(x) for x in floor_batch_sweep.split(',') if x.strip()],
        min_separation=min_separation, ceiling_tol=ceiling_tol,
        render_rounds=[int(x) for x in render_rounds.split(",") if x.strip()],
        render_policies=[p for p in render_policies.split(",") if p.strip()],
        render_n=render_n,
        certify_only=certify_only,
    )
    out = run_floor_tap.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "floor_tap_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
    if out.get("render"):
        with open(os.path.join(localdir, "render_pack.json"), "w") as fh:
            json.dump(out["render"], fh, cls=NumpyEncoder)
        print(f"[local] wrote render_pack.json ({len(out['render']['packs'])} packs) to {localdir}")
