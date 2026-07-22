# The fully-online two-timescale value loop: why it's obstructed, and why that confirms the premise

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) — this cut attacks the keystone the doc names verbatim (interface next-piece #2 / [mjc README](../README.md) next-steps): the **fully-online single two-timescale loop** — "reward continuously shaping a *live* FM, vs an outer optimization over retrained FMs." We built it two ways and found it **obstructed on both levers, for two distinct deep reasons** — and the obstruction turns out to be a *positive confirmation* that value must live on the slow timescale.
**Parents / lineage**: afferent lever ← [../curiosity_control/README.md](../curiosity_control/README.md) (the drive atom on control; self-tuning `b` was its flagged next step). Efferent lever ← [../meta_adapt/README.md](../meta_adapt/README.md) §#4d/#4e (`../meta_adapt/meta_value_shaping.py` / `../meta_adapt/meta_value_learn.py` — value-shaping *caused by reward*, but **offline**: an outer CEM over FMs retrained from scratch per candidate weight). Re-adaptation substrate for the proposed next step ← [../dynamics_shift/README.md](../dynamics_shift/README.md) (`../dynamics_shift/dynamics_shift.py`).
**Code** (lives in this folder): `meta_curiosity_loop.py` (afferent — self-tune *where to collect*), `meta_value_online.py` (efferent — self-tune *what the FM models*). File index: [FILES.md](FILES.md). **Status**: done; both are **instructive negatives + one conceptual reframe**. Mechanism metrics clean; control-level results are negatives with an understood cause. Single family (drifting/capacity-competition pusher); 1–3 seeds/regime. **Date**: 2026-07-21.
**Builds on this**: [`drift_value_loop/`](../drift_value_loop/README.md) (runs the experiment this cut named — re-adaptation compounding across a sequence of Type-2 drifts — and finds the obstruction was the *teacher*)

---

## Conceptual background (what "the loop" is, and what we learned it *is*)

### The frame

The idea doc's bet: intelligence runs on **two learning systems at two speeds** —
- a **fast forward model (FM)** — *"if I do X, what happens next?"* — the cerebellar/supervised axis, which adapts in seconds; and
- a **slow value system** — *"what is worth attending to / what is good?"* — the basal-ganglia/dopaminergic axis, consolidated over many experiences.

The contribution the doc claims is missing: **put a reward signal into the *slow* loop and let it steer the fast one**, as *one continuously-running system* (not the offline batch jobs of #4d/#4e). That is what turns a self-consistent *predictor* into an *agent*.

The value can steer the FM through **two interface taps** (the doc's "two afferent taps + one efferent gain"):
1. **Afferent — *where to collect*** (explore/exploit balance): as the world drifts, should the agent go where its predictions are *wrong-but-learnable* (curiosity, the `e`-tap) or where the *task needs it accurate* (exploit, the `p`-tap)? The slow loop tunes the balance `b`.
2. **Efferent — *what to model well*** (capacity allocation): the FM has limited capacity; should it spend it veridically on *everything* or *re-allocate* toward the value-relevant subspace? The slow loop tunes the per-dim weight.

Prior cuts landed both taps **offline / hand-set**. This cut asked: **can a single online reward loop *discover* the right setting of either tap, in one live FM?**

### The headline finding: value is a *slow, committed, offline-estimable* quantity

**Both levers' fast-online discovery is obstructed — for two different deep reasons — and the obstruction confirms the two-timescale premise itself.** We set out to build "the fast online loop that discovers the value," and the structure of the problem refused, twice:

| lever | why fast-online discovery is obstructed |
|---|---|
| **afferent** (explore/exploit balance) | The drive's effect on the FM is real (pure-explore tracks a drifting frontier ~3× worse and gets seduced by aleatoric noise — the noisy-TV pathology). **But CEM-MPC control is *robust* to it**: a replanning controller tolerates a stale model (Cut #3's "search-substitutes-for-the-model"), and the task/exploit signal already localizes the frontier where control cares. So the *control-reward landscape over `b` is flat* → the loop has no gradient and wanders by seed. This *deepens* the value-relevance dissociation ([CURIOSITY_CONTROL](../curiosity_control/README.md) Finding 2): the explore tap's distinctive contribution is exactly the part control is robust to. |
| **efferent** (capacity allocation) | The value-shaping control benefit is a **slow, long-horizon, commit-dependent** property of *training*. It (i) **washes out at convergence** — the gain shrinks/reverses as the FM trains (see table below) — and (ii) is **invisible to fast local probes** — a re-allocation that takes thousands of steps to manifest cannot be read off a short antithetic probe. A probe long enough to see it *is* a full retrain, degenerating back to #4e's offline fresh-FM-per-candidate. A single live FM cannot be *reset* to compare committed candidates, and a fast reward gradient cannot see a commit-dependent benefit. **This is *why* #4e had to go offline.** |

So the "failure" is really this: **the value — both the drive's relevance and the FM's allocation — is not the kind of thing you can fast-tune with a local reward gradient.** You can only estimate it by *committing* for a long time (or comparing fully-trained alternatives offline). The idea doc *asserts* "value belongs on the slow timescale"; these experiments show **why it is forced there.** The two-timescale split is not a design choice — it is imposed by the structure of value. (This maps onto biology: dopaminergic value is consolidated slowly over many experiences — even in sleep — not fast-tuned per step.)

### The reframe that resolves it: adaptation *speed*, not converged competence

Every place we looked, the value signal's benefit was a **sample-efficiency / adaptation-speed** effect, **not** a peak-performance effect:
- value-shaping (dropping the value-irrelevant subspace) helps the FM get good *faster*, and stops mattering once it's fully trained;
- curiosity helps *find the reducible frontier sooner*, and stops mattering once the task signal alone gets you there.

**None of it raises the ceiling on a fixed task — all of it accelerates adaptation.** This matches the human pattern (a narrow system beats us at any *converged* skill; we are unmatched at picking up *new* tasks fast and robustly), and it *rescues* the efferent "washout" finding: **the washout only bites if you reach convergence, and under perpetually-drifting dynamics you never do.** Every new context (Jasper's "picking up a cup in a new context") is a fresh pre-convergence sprint, so the regime where value-shaping helps is not a window you pass through once — it is the *permanent operating regime*. Under that frame the value's job is sharp: **carve the FM down to the core generalizable invariants — "nothing more, nothing less" — so that when the world drifts, only a thin part needs re-fitting.**

### Architecture refinement (the fast system is cortex **+** cerebellum, FM as bridge)

The "fast system" is not the FM alone. Mapped onto Cut #3: the **planner** that does the task is cortex; the **FM** is the cerebellum providing fast self-calibration; and the cerebellum is *also the bridge to value* — the FM's **prediction error is the messenger** the value system reads to decide what is worth modeling (the idea doc's literal cerebellum→VTA pathway). So the loop is **fast (cortex does the task + cerebellum calibrates) ⟷ slow (value selects the invariants to model)**, with the FM's prediction error carrying signal both ways. That is a cleaner fast/slow split than "cerebellum = fast": *task-doing + self-calibration is fast; invariant-selection is slow; the FM is the interface* — which is exactly why it kept refusing to sit cleanly on either timescale in our experiments.

---

## Cut A — Afferent online loop (`meta_curiosity_loop.py`): a control-negative with a known cause

**What it is.** One continuous run on [CURIOSITY_CONTROL](../curiosity_control/README.md)'s substrate (RPF-ensemble FM continually re-fit on a recency FIFO; teleport-region collection scored by `grounded@b = b·reducible + (1−b)·exploit`). A **slow reward-driven outer loop** (advantage-normalized REINFORCE on `θ`, `b=σ(θ)`, updated every `outer_m` rounds) self-tunes the explore/exploit balance from **control reward only** (−CEM-MPC goal-distance). Reference arms hold `b` fixed (`b1.0/b0.5/b0.0`) or collect at random. No wireheading: `b∈(0,1)` is bounded/on-manifold and the reward is real-env rollouts.

**The 2×2 was designed to show:** reward discovers an *intermediate* `b` without noise, and *drops* `b` under a noisy-TV patch (grounding itself), matching the swept optimum — plus tracking a *moving* optimum when noise onsets partway through.

**What happened.**
- **Mechanism works at small scale.** In the smoke, when a noise patch switched on mid-run the loop drove `b` 0.5 → 0.22 (θ −0.33 → −1.32), kept noise-frac at 0.00, and beat pure-explore `b=1` by +0.046 — the noisy-TV-grounding story, live.
- **But at full capacity the *control* landscape over `b` is flat**, so the loop has no gradient to climb. Learned `b` came out **seed-determined, not condition-determined** (seeds 0/1 → high `b≈0.7–0.9`; seed 2 → low `b≈0.22`), and neither noise nor onset bent it.
- **The drive's effect is real at the *tracking* level.** Fixed-arm frontier-tracking error: pure-explore `b=1` tracks the needle **3× worse under noise (0.042 vs 0.015)** and is noise-sensitive; exploit `b≤0.5` is noise-immune (~0.015 either way). The noisy-TV pathology is present and sharp.
- **It just doesn't transmit to control.** That 3× tracking gap moved final control by only ~0.01 (b1.0 0.17 vs b0.0 0.16). Characterization confirmed there is **no sensitive window**: weak needle (amp 3.5, replan 8) → everyone controls *well* (~0.16, flat); strong needle (amp 7, replan 12) → everyone controls *badly* (~0.40 ≈ random floor, flat); the ballistic edge (amp 5, replan 10) → ~0.27, with `b0.0 ≈ random ≈ best`. The CEM-MPC replanner is robust to exactly the tracking differences the drive produces.

**Conclusion.** A clean control-negative: on a single tracked frontier, the exploit/task signal already suffices where control cares, and the controller tolerates the residual staleness — so the afferent value loop has no control gradient to discover. (See "the fragility is the finding" — this is the afferent face of it.)

## Cut B — Efferent online loop (`meta_value_online.py`): the value-shaping signal is slow & commit-dependent

**What it is.** #4e's substrate reused verbatim (context-FM `f(s,u,z)` over a `push_rot` actuator family; puck + optional pusher force-field = capacity competition; CEM-MPC control cost that ignores the puck). The change vs #4e: instead of retraining an FM from scratch per candidate weight (offline), **one live FM trains continuously** while a **slow reward-driven loop tunes its scalar puck-weight** `w_puck` online. Two update modes:
- **`single`** — advantage-normalized REINFORCE on `θ` (`w_puck=σ(θ)`), one perturbed `w` per epoch. Needs the FM warmed to near-convergence first, else a training-progress **trend confound** dominates (the loop chases the FM's raw improvement, not the `w` effect).
- **`fork`** — antithetic ES: each epoch, probe `w±ε` on two branches forked from the *same* live snapshot with **common random numbers** (identical minibatches), so the trend + minibatch noise cancel and `cost(w−ε)−cost(w+ε)` is *purely* the `w` effect. Operates pre-convergence.

**What happened.**
- **The two-timescale machinery works** (live FM + slow reward loop, responsive updates, no crash).
- **The re-allocation is real, in one live FM.** At marginal competition (hidden 64, amp 2.0, single mode, seed 0) the loop drove `w_puck` **0.5 → 0.26**, cost **0.305 → 0.205**, and **pusher-vel R² 0.69 → 0.90** — reward, flowing back, dropped the value-irrelevant puck and re-allocated capacity to the pusher.
- **But the control payoff is a pre-convergence transient.** The value-shaping gain **shrinks or reverses as the FM converges** — the whole story in one table:

  | setting | drop-vs-keep gap at **warmup** (1.5k steps) | gap at **convergence** (5.5k steps) |
  |---|---|---|
  | hidden 64, amp 2.0 | w0.0 0.243 vs w1.0 0.259 → **+0.016 (drop better)** | 0.210 vs 0.214 → **+0.004 (barely)** |
  | hidden 40, amp 3.0 | 0.427 vs 0.468 → **+0.040 (drop better)** | 0.360 vs 0.335 → **−0.025 (keep better — reversed)** |

  So the converged online loop (which needs convergence to de-trend) operates where the signal is gone. And the marginal competition is **seed-dependent** (seed 1's FM already modeled the pusher fine at `w=0.5` → no competition → `w_puck` wandered to 0.54). On the **easy pusher** (no competition) the landscape is flat across all 3 seeds and the loop is correctly *indifferent* (`w_puck` wanders 0.49–0.82).
- **Fork mode exposed the deeper obstruction.** Pre-convergence, the fixed-arm landscape *does* show the real gradient (w0.0 **0.226** ≪ w1.0 **0.269**, gap 0.043). But the online fork loop drove `w_puck` the **wrong way** (up to 0.74) and landed worse than veridical — because the fork `diff` signals were tiny and sign-inconsistent (|diff| ~0.002–0.018) while the true gap is 0.043. **A 150-step probe is too short to reveal a benefit that takes thousands of steps of re-allocation to manifest.** Long-enough probes would work but *are* full retrains (≈ offline #4e).

**Conclusion.** The efferent value (allocation) is a **slow, long-horizon, commit-dependent** quantity: single-mode washes it out at the convergence it needs; fork-mode can't see it in a local probe. Fast-online discovery of it is structurally obstructed — which is exactly why #4e discovered it offline.

---

## What this establishes / caveats

- **Establishes.** (1) The two-timescale online loop *machinery* works on both levers (live inner FM + slow reward-driven outer loop, no wireheading). (2) **Fast-online discovery of the value is obstructed on both levers, for two distinct deep reasons** — afferent: CEM-MPC control is robust to the drive's tracking effect; efferent: the value-shaping benefit is a long-horizon, commit-dependent, pre-convergence transient. (3) Therefore **value is a slow / committed / offline-estimable quantity** — a *positive, mechanistic confirmation* that the two-timescale split is forced, not chosen. (4) The value's benefit is an **adaptation-speed** effect, not a converged-competence effect, which (5) dissolves the efferent "washout" concern under perpetual drift (you never converge).
- **Caveats / does-not-show.**
  - **These are negatives + a reframe, not a clean positive control win.** We did *not* land "the online loop wins on converged control" — because (the finding is) it structurally can't on a fixed task.
  - **Substrate-bound.** The afferent control-robustness is a property of CEM-MPC with replanning on this pusher; a more ballistic / less-re-groundable controller *might* transmit the tracking gap (we found no sensitive window here, but did not exhaust it). The efferent washout was measured on the #4e puck+field family; the *reversal* at hidden 40/amp 3.0 (keeping the "irrelevant" puck *helps* — likely shared-feature regularization) is its own un-chased thread.
  - **Seeds.** Afferent full matrix effectively 3 seeds (control seed-variable, as the parent warned); efferent 1–3 seeds/regime. Mechanism metrics (frontier-err, per-dim R², `w_puck`/`b` trajectories) are the trustworthy readouts; control is the noisy one.
  - **The reframe is a prediction, not yet a result.** "Value earns its keep on adaptation speed / compounds across drifts" is argued, not shown — see Next steps.
  - **A known trap, for the record.** The efferent single-mode loop is only meaningful *after* the trend confound is killed (warmup-to-convergence); the smoke that skipped it chased raw training progress. The afferent `random`-arm needs `b=None` guarding (fixed early; the first 9-run matrix crashed at save on it — trajectories survived in logs).

## Reproduce

```bash
cd experiments/
# --- Cut A: afferent online loop (self-tune where to collect) ---
modal run mjc/online_value_loop/meta_curiosity_loop.py::meta_curiosity_loop --quick                 # smoke
# 2x2 (reward-discovers-the-setpoint) + moving-optimum onset, 3 seeds each:
for s in 0 1 2; do
  modal run --detach mjc/online_value_loop/meta_curiosity_loop.py::meta_curiosity_loop --tag disc_clean_s$s --seed $s
  modal run --detach mjc/online_value_loop/meta_curiosity_loop.py::meta_curiosity_loop --tag disc_noise_s$s --seed $s --noise
  modal run --detach mjc/online_value_loop/meta_curiosity_loop.py::meta_curiosity_loop --tag onset_s$s     --seed $s --noise --noise-onset 0.4
done
# landscape characterizations (no sensitive window): --patch-amp {3.5|5|7} --replan-every {8|10|12}

# --- Cut B: efferent online loop (self-tune what the FM models) ---
modal run mjc/online_value_loop/meta_value_online.py::meta_value_online --quick --field-pusher-amp 2.0   # smoke
# the 2x2 (competition vs easy), 3 seeds each:
for s in 0 1 2; do
  modal run --detach mjc/online_value_loop/meta_value_online.py::meta_value_online --tag comp_s$s --seed $s --field-pusher-amp 2.0 --outer-mode single --warmup-steps 1500 --outer-every 250
  modal run --detach mjc/online_value_loop/meta_value_online.py::meta_value_online --tag easy_s$s --seed $s --field-pusher-amp 0.0 --outer-mode single --warmup-steps 1500 --outer-every 250
done
# fork mode (pre-convergence, exposes the long-horizon obstruction):
modal run --detach mjc/online_value_loop/meta_value_online.py::meta_value_online --tag comp_fork_s0 --seed 0 --field-pusher-amp 2.0 --outer-mode fork
```
Use `--detach`; launch runs as **independent** background processes (parallel `&` + `--detach` can lose all-but-the-last on a client disconnect — the experiments-CLAUDE gotcha bit us repeatedly here). Results at `/data/{meta_curiosity_loop,meta_value_online}/<tag>/`, mirrored to `figures/…` when the client survives; otherwise pull from the volume.

## Figures
- `meta_curiosity_loop_<tag>/`: `fig1_learned_b` (the learned set-point trajectory + onset marker), `fig2_control` (control + frontier-tracking per arm), `fig3_occupancy`, `fig4_orderparams`.
- `meta_value_online_<tag>/`: `fig1_learned_w` (learned `w_puck` trajectory vs fixed references), **`fig2_control_realloc`** (control + the pusher-vel↑/puck-vel↓ re-allocation in one live FM), `fig3_orderparams`.

## Next steps — the experiment the whole thing points at

**Measure adaptation *speed* under perpetual drift, and look for *compounding* — the meta-layer signature the program has never isolated.** This is where, by everything above, the value should finally earn its keep, and it's built directly on the substrate Cut #3 already provides.

- **Substrate**: Cut #3 (`../dynamics_shift/dynamics_shift.py`) already shows the *fast half* — a model-based agent re-fits its FM from ~50 reward-free transitions after a *single* dynamics shift (drag→ice), while a model-free policy is flat / needs ~240× more. But it re-fits the **whole** FM, on **one** shift.
- **The experiment**: a **sequence of related Type-2 drifts** (the generator itself changing — e.g. the actuator-rotation `push_rot` conflict that opened the meta gap in [META_ADAPT](../meta_adapt/README.md) Cut #4, *not* mere noise, which [RHM_META_LEARNING](../../rhm/ratchet/RHM_META_LEARNING_README.md) showed collapses meta-learning to multitask). Compare **transitions-to-recover per drift** for:
  - a **veridical FM** (models everything → re-learns everything each drift), vs
  - a **value-carved FM** — a stable invariant core + a thin adaptable part. The program already has the architecture: the **context-latent FM `f(s,u,z)`** from [META_ADAPT §#4b](../meta_adapt/README.md) (adapt = infer a small `z`, not re-fit the net). The value's job is to decide *what lives in the frozen invariant `f` vs the fast-adapting `z`* — and *that split is the slow/committed quantity* this cut showed value has to be.
- **The prediction (the payoff)**: the value-carved FM re-adapts **faster**, and the gap **compounds** — each new drift is cheaper because the invariant core is already paid for. That compounding-across-drifts is the meta-layer signature (idea-doc "strict compounding not isolated"; [CURIOSITY_CONTROL](../curiosity_control/README.md) caveats), and it is precisely "humans pick up new tasks fast and robustly."
- **Why it respects this cut's finding**: the value-carving (which invariants) is established **slowly** — over the drift sequence / a lifetime, or offline — consistent with "value is slow/committed." It then **pays off in fast re-adaptation**. The slow-value → fast-adaptation split is preserved; we stop measuring converged control (where value is *designed* not to help) and start measuring re-adaptation speed (where it should).
- **Secondary threads**: (i) the afferent tap in a genuinely *ballistic, non-re-groundable* controller (does the tracking gap ever transmit to control?); (ii) the hidden-40/amp-3.0 *reversal* (when does modeling the "irrelevant" subspace *regularize* the relevant one?); (iii) back-translate the "value is slow/committed — fast-online discovery is obstructed" finding into [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) and the cerebellum belief tree.
