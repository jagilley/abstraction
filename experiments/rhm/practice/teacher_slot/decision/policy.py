"""teacher_slot/decision — the outer loops, kept out of the Modal app so every rule is
auditable and testable with no GPU and no substrate (the donor's `wall.py` convention).

WHAT AN OUTER LOOP IS HERE. `../../fourwall/lm/` measured that the inner (gradient) loop
supplies exactly one index op — *track* — and that the merge op, supplied exogenously,
repairs the withheld inference pathway at a +0.10-0.17 nat transient that decays in
~25-125 steps. Two facts make the decision round well-posed:

  (a) at the instant of the op TASK NLL IS SPIKED WHILE THE PATHWAY READOUT IS ALREADY
      CLIMBING -- a quantity exists that improves inside the dip;
  (b) the within-level lifetime task integral REWARDS never merging.

So: give a second loop the choice, at every checkpoint, of the input condition for the
next interval -- `w` present (at the CURRENT era's map) or collapsed to the neutral filler
-- and vary only WHAT IT READS. `fourwall/lm`'s `merge_s` arms are the step-function
special case of this policy class, which is also the fidelity gate.

THE ACTIONS are reversible per decision point, which is what makes "hold through the dip"
a decision rather than a foregone conclusion: a loop reading task NLL *can* bail out at
M+125 when the spike is at its worst.

  KEEP      the wall token, mapped through the era's current offset q (never a stale
            snapshot -- `restore` re-enters the schedule where the global clock is)
  COLLAPSE  the neutral filler: `fourwall/lm`'s merge op, applied for one interval

DESIGN CHOICES, stated because they are the experiment's degrees of freedom:

  * The learner is minimal ON PURPOSE -- a two-action value with recency-weighted
    estimates of a baseline-subtracted reward. The point is what it READS, not how clever
    it is. It is also fully DETERMINISTIC (a fixed geometric try-the-other-action
    schedule instead of epsilon-greedy), so no policy consumes any RNG at all and the
    donor's zero-global-RNG design is preserved exactly.
  * The reward is BASELINE-SUBTRACTED. Raw per-interval improvement is dominated by the
    learning curve's own slope (everything improves fast early and slowly late), so an
    un-baselined bandit would just credit whichever action it happened to try first. The
    baseline is a recency-weighted average of the improvement rate REGARDLESS of action,
    so Q[a] estimates the ADVANTAGE of a over the ambient rate. This is the standard
    gradient-bandit baseline and it is symmetric across the two actions -- nothing in the
    rule prefers collapse.
  * Exploration is unavoidable (a policy that never explores never observes the untried
    condition) and it is NOT free here: 125 steps under the other condition really does
    perturb the learner. It is therefore DETERMINISTIC, geometric and decaying in density,
    and it is the ONLY way an untried action is ever reached. Every forced trial is tagged
    in the trace so the reduction can separate a GREEDY restore (a bail-out) from a
    scheduled one. Because every arm shares the same schedule, differences in WHEN the
    arms commit are differences in what they read, not in when they got to look.
  * The first `burn` decision points are a BURN-IN: the loop holds the status quo and its
    rewards seed the baseline without setting any Q. See `BanditPolicy` for why -- the
    smoke run showed the alternative silently hands the run to whichever action the
    warm-up happened to sample second.
"""

KEEP, COLLAPSE = "keep", "collapse"
ACTIONS = (KEEP, COLLAPSE)


def forced_trial_indices(n_decisions, start=6, growth=1.5):
    """Deterministic 'try the other action' schedule: geometric, decaying in density.

    Front-loading is deliberate -- an exploratory interval costs the learner something
    (the abandoned wall circuitry evaporates fast, and rebuilding it takes 50-250 steps),
    so trials are cheapest early and are thinned out as the run matures.
    """
    out, t = [], int(start)
    while t < n_decisions:
        out.append(t)
        nxt = int(round(t * growth))
        t = nxt if nxt > t else t + 1
    return out


class Policy:
    """Base: `decide(t, step, reads) -> (action, info)`; `needs` names the readouts whose
    cost the ledger charges (an empty tuple = this policy buys nothing)."""

    needs = ()
    kind = "base"

    def needs_at(self, t):
        """Which readouts the policy's input stream contains at decision point `t`.
        Dynamic because a PAIRED policy only reads at its trial boundaries."""
        return self.needs

    def decide(self, t, step, reads):
        raise NotImplementedError


class FixedPolicy(Policy):
    """The step function: KEEP until `merge_at`, COLLAPSE forever after.

    `merge_at=None` -> never collapse (`wall`); `merge_at=0` -> always collapse
    (`no_wall`'s behaviour, though that arm carries no wall token at all). This class is
    the FIDELITY GATE: run under it, an arm must reproduce its `fourwall/lm` twin
    bit-for-bit, because it probes nothing, consumes no RNG, and reduces the condition
    rule to exactly the donor's `step >= merge_at`.
    """

    kind = "fixed"

    def __init__(self, merge_at=None):
        self.merge_at = merge_at

    def decide(self, t, step, reads):
        collapse = self.merge_at is not None and step >= self.merge_at
        a = COLLAPSE if collapse else KEEP
        return a, {"rule": "fixed", "why": "fixed", "greedy": a}


class BanditPolicy(Policy):
    """A two-action value over ONE scalar read.

    read_key   which readout the policy's input stream contains. This is the ONLY thing
               that differs between `outer_task`, `outer_path` and `outer_path_lp`.
    mode       'delta'  reward = the one-interval improvement in the read error
               'lp'     reward = two-timescale learning progress on the read error,
                        EMA_slow(e) - EMA_fast(e) (`two_timescale_value_loop`'s form):
                        positive while the error is falling, with memory, so credit for a
                        real improvement persists for several intervals.

    All readouts are passed in as ERRORS (lower is better) so the two modes share a sign
    convention: improvement = a fall in e.

    TWO RULES THAT THE SMOKE RUN FORCED, both about the same hazard -- the learning
    curve's own slope is enormous in the first few hundred steps and swamps any
    action-differential:

      BURN-IN. For the first `burn` decision points the loop holds the STATUS QUO and its
      rewards update the baseline ONLY. Without this, the first interval's reward
      (random-init -> trained, ~1.6 nats) becomes the baseline and hands whichever action
      happens to be sampled second an advantage of about -1.6, i.e. the ORDER of the
      warm-up decides the run. Burning in fixes that without building in a preference:
      the rule is "hold what you were handed while your baseline matures", and it applies
      identically whichever condition the arm starts in.

      AN UNTRIED ACTION IS NEVER CHOSEN GREEDILY. Q is `None` until an action has actually
      been observed, and an untried action is reached ONLY through the deterministic
      forced-trial schedule. Otherwise the first post-burn-in decision would be settled by
      the sign of a noise-level advantage on the one action that had been tried.

    Ties (including "only one action has ever been tried") go to the STATUS QUO -- the
    action currently in force -- which is symmetric as a rule, though on a wall-carrying
    arm it reads as "do not disturb the conditions of learning without evidence".
    """

    kind = "bandit"

    def __init__(self, read_key, mode="delta", alpha=0.5, beta=0.3,
                 fast=0.5, slow=0.15, n_decisions=161, burn=4,
                 explore_start=5, explore_growth=1.5, priced=True,
                 init_action=KEEP):
        self.read_key = read_key
        self.mode = mode
        self.alpha, self.beta = alpha, beta
        self.fast_a, self.slow_a = fast, slow
        self.burn = burn
        self.forced = set(forced_trial_indices(n_decisions, explore_start, explore_growth))
        self.needs = (read_key,) if priced else ()
        self.Q = {KEEP: None, COLLAPSE: None}
        self.seen = {KEEP: 0, COLLAPSE: 0}
        self.baseline = None
        self.e_prev = None
        self.ema_f = None
        self.ema_s = None
        self.init_action = init_action
        self.last_action = None

    # -- the read -> reward map ------------------------------------------------ #
    def _reward(self, e):
        """`e` is the current value of the read ERROR. Returns (reward, detail)."""
        if self.mode == "lp":
            self.ema_f = e if self.ema_f is None else self.ema_f + self.fast_a * (e - self.ema_f)
            self.ema_s = e if self.ema_s is None else self.ema_s + self.slow_a * (e - self.ema_s)
            r = self.ema_s - self.ema_f          # > 0 while the error is falling
            return r, {"ema_fast": self.ema_f, "ema_slow": self.ema_s}
        r = None if self.e_prev is None else (self.e_prev - e)
        return r, {"e_prev": self.e_prev}

    def decide(self, t, step, reads):
        e = reads[self.read_key]
        r, detail = self._reward(e)
        info = {"rule": "bandit", "e": e, "reward": None, "advantage": None,
                "baseline": self.baseline, "credited": self.last_action, **detail}

        # credit the interval that just closed to the action that was in force during it
        counted = False
        if r is not None and self.last_action is not None:
            b = r if self.baseline is None else self.baseline
            adv = r - b
            a = self.last_action
            if t >= self.burn:                     # Q updates start after the burn-in
                self.Q[a] = adv if self.Q[a] is None else \
                    self.Q[a] + self.alpha * (adv - self.Q[a])
                self.seen[a] += 1
                counted = True
            self.baseline = r if self.baseline is None else \
                self.baseline + self.beta * (r - self.baseline)
            info.update({"reward": r, "advantage": adv, "baseline": self.baseline})
        self.e_prev = e

        # choose
        sq = self.init_action if self.last_action is None else self.last_action
        tried = [x for x in ACTIONS if self.Q[x] is not None]
        if not tried:
            greedy = sq
        else:
            best = max(self.Q[x] for x in tried)
            greedy = sq if (sq in tried and self.Q[sq] >= best) else \
                next(x for x in tried if self.Q[x] == best)
        if t < self.burn:
            action, why = sq, "burn_in"
        elif t in self.forced:
            action = COLLAPSE if greedy == KEEP else KEEP
            why = "forced_trial"
        else:
            action, why = greedy, "greedy"
        info.update({"why": why, "q_keep": self.Q[KEEP], "q_collapse": self.Q[COLLAPSE],
                     "greedy": greedy, "status_quo": sq, "counted": counted,
                     "n_keep": self.seen[KEEP], "n_collapse": self.seen[COLLAPSE]})
        self.last_action = action
        return action, info


class PairedPolicy(Policy):
    """ROUND 2. The same two-action decision, estimated by an ABBA PAIRED TRIAL.

    WHY THIS REPLACES `BanditPolicy`. Round 1 (`tsdA`) measured its own instrument
    instead of the world. Its reward was a one-interval improvement compared against a
    recency-weighted baseline, and the first interval's improvement is 92-97x the ambient
    rate; the baseline needs ~13 decision points to come within 2x of ambient, so an
    action estimated at t=4 got advantage -0.76 and one estimated at t=6 got -0.43 for no
    reason but the decay. Since an untried action is only ever reached by a scheduled
    trial, the untried action is ALWAYS sampled second, so the decaying baseline
    systematically favoured it. Three arms reading three different quantities committed
    at the identical step with near-identical values. That is the signature, and no
    amount of burn-in removes it -- the fix has to be structural.

    THE FIX. Never compare an action to a baseline; compare it to the OTHER ACTION,
    measured in adjacent intervals, in an ABBA block:

        polarity 0:  C K K C          polarity 1:  K C C K

    with the contrast  D = mean(improvement over the C intervals)
                         - mean(improvement over the K intervals).

    Under any improvement rate that is linear in time over the block, u_i = g - i*delta,
    the trend cancels EXACTLY within a single trial: (u1+u4)/2 - (u2+u3)/2 = 0. That is
    the whole point -- the estimate no longer depends on when in the run it was taken, on
    which action was tried first, or on the scale of the ambient learning rate. The block
    polarity alternates across trials so any residual curvature cancels on average too.

    D is a signed contrast in nats (or in probe accuracy) per interval, and the decision
    rule is sign(V) where V is a recency-weighted average of the D's -- so the rule is
    SCALE-FREE by construction and manifestly antisymmetric under relabelling the two
    actions. There is no baseline, no optimism, no tie-break that leans anywhere except
    the status quo.

    span   how many decision intervals each letter of the block occupies. This is the
           `two_timescale_value_loop` axis, and it is the ONLY difference between a
           `delta` arm and an `lp` arm in round 2: span=1 measures the contrast over a
           one-interval (125-step) exposure to each condition, span=2 over a
           two-interval (250-step) exposure. Under pairing the round-1 delta-vs-LP
           distinction genuinely collapses (an EMA's memory leaks credit ACROSS the
           block and destroys the very cancellation the block is for), so it is
           re-derived here as the measurement HORIZON, which is what the two-timescale
           reward was actually buying. Trial cadence is set per arm so that every arm
           spends the SAME number of intervals inside trials and the same number under
           the non-greedy condition -- the horizon is varied, the perturbation is not.

    Between trials the loop simply plays sign(V) and reads nothing, so a paired policy
    also PROBES FAR LESS than round 1's: it needs a read only at the block's 5 boundary
    points, not at all 160 decision points.
    """

    kind = "paired"

    # A DEAD ZONE at the instrument's own measured noise floor. `sign(V)` alone will
    # flip on arbitrarily small values -- the offline gate caught it flipping on 2.2e-16
    # of floating-point residue when the true contrast was exactly zero -- and more to
    # the point, a loop should not rearrange the conditions of learning on evidence
    # smaller than its instrument can resolve. Values are the donor's measured floors,
    # not free parameters: 0.0012 nats is fwlm's seed floor on indexed-span NLL, and
    # 0.005 is the dead-pair spread on the d5 probe.
    V_TOL = {"task": 0.0012, "path": 0.0012, "yield": 0.005}

    def __init__(self, read_key, span=1, burn=8, trial_every=16, n_decisions=161,
                 alpha=0.5, init_action=KEEP, priced=True, v_tol=None):
        self.read_key = read_key
        self.span, self.burn, self.alpha = int(span), int(burn), float(alpha)
        self.v_tol = float(self.V_TOL.get(read_key, 0.0) if v_tol is None else v_tol)
        self.needs = (read_key,) if priced else ()
        self.init_action = init_action
        self.last_action = None
        # the status quo is the last CHOSEN action, never the last letter of a trial
        # block: a scheduled trial is not a choice, so it must not move the default.
        self.last_chosen = None
        self.V = None
        self.n_trials_done = 0
        blk = 4 * self.span
        last = n_decisions - 2                    # the final index is the stop step
        self.trials = [t for t in range(self.burn, n_decisions, int(trial_every))
                       if t + blk <= last]
        # decision index -> (trial i, letter); and the block-boundary read points
        self.plan, self.read_at, self.reads_of = {}, set(), {}
        for i, t0 in enumerate(self.trials):
            letters = "CKKC" if i % 2 == 0 else "KCCK"
            for k in range(blk):
                self.plan[t0 + k] = (i, letters[k // self.span])
            for k in range(5):
                self.read_at.add(t0 + k * self.span)
                self.reads_of.setdefault(i, []).append(t0 + k * self.span)
        self.buf = {}

    def needs_at(self, t):
        return self.needs if t in self.read_at else ()

    def decide(self, t, step, reads):
        info = {"rule": "paired", "e": None, "contrast": None, "V": self.V,
                "n_trials": self.n_trials_done}
        if t in self.read_at:
            info["e"] = self.buf[t] = reads[self.read_key]

        # close out any trial whose fifth read point is now in hand
        for i, pts in self.reads_of.items():
            if pts[-1] != t or any(q not in self.buf for q in pts):
                continue
            e = [self.buf[q] for q in pts]
            u = [e[k] - e[k + 1] for k in range(4)]      # improvement per block quarter
            letters = "CKKC" if i % 2 == 0 else "KCCK"
            uc = [u[k] for k in range(4) if letters[k] == "C"]
            uk = [u[k] for k in range(4) if letters[k] == "K"]
            D = sum(uc) / len(uc) - sum(uk) / len(uk)
            self.V = D if self.V is None else self.V + self.alpha * (D - self.V)
            self.n_trials_done += 1
            info.update({"contrast": D, "V": self.V, "polarity": letters,
                         "u": u, "n_trials": self.n_trials_done})

        sq = self.init_action if self.last_chosen is None else self.last_chosen
        greedy = sq if self.V is None or abs(self.V) <= self.v_tol else \
            (COLLAPSE if self.V > 0 else KEEP)
        if t in self.plan:
            action = COLLAPSE if self.plan[t][1] == "C" else KEEP
            why = "trial"
        elif t < self.burn:
            action, why = sq, "burn_in"
        else:
            action, why = greedy, "greedy"
        info.update({"why": why, "greedy": greedy, "status_quo": sq,
                     "v_tol": self.v_tol,
                     "q_keep": None if self.V is None else -self.V,
                     "q_collapse": None if self.V is None else self.V,
                     "trial": self.plan.get(t, (None, None))[0]})
        self.last_action = action
        if why != "trial":
            self.last_chosen = action
        return action, info


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

class Budget:
    """A shared step-budget. Training steps cost 1; a PATHWAY probe costs `price`.

    THE ASYMMETRY IS THE POINT (ear's convention: evaluation is priced). Task NLL under
    the arm's consumed condition is the learner's own experience and is charged nothing.
    The pathway readout is a COUNTERFACTUAL evaluation -- the model run under the neutral
    wall, which for a wall-carrying arm is a condition it is not living in -- and that is
    what costs. `price` is calibrated to the probe's true compute cost relative to one
    training step (`slot_lm.bench` measures it on the same GPU).

    Instruments logged for science at every checkpoint are free and identical across
    arms; the ledger charges only what the POLICY's input stream contains.
    """

    FREE = ("task",)

    def __init__(self, total, price, prices=None):
        self.total, self.price = int(total), int(price)
        # per-read prices; anything unlisted falls back to `price`. Round 2 needs this
        # because the NEXT-LEVEL YIELD read (a d5 probe fit) costs an order of magnitude
        # more than the pathway read (one forward pass), and that asymmetry is part of
        # the measurement rather than something to average away.
        self.prices = dict(prices or {})
        self.spend = 0
        self.n_probes = 0
        self.by_read = {}

    def price_of(self, n):
        return 0 if n in self.FREE else int(self.prices.get(n, self.price))

    def charge(self, needs):
        c = 0
        for n in needs:
            pc = self.price_of(n)
            c += pc
            self.by_read[n] = self.by_read.get(n, 0) + pc
            if pc:
                self.n_probes += 1
        self.spend += c
        return c

    @property
    def train_cap(self):
        """The last training step the arm can afford: budget minus what it has spent."""
        return self.total - self.spend

    def exhausted(self, step):
        return step >= self.train_cap


# --------------------------------------------------------------------------- #
# trace reduction (pure; used by the analyzer and by the offline gate)
# --------------------------------------------------------------------------- #

def summarise_trace(trace, max_steps):
    """The committed decision readouts, computed from the decision trace alone."""
    if not trace:
        return {}
    acts = [d["action"] for d in trace]
    steps_ = [d["step"] for d in trace]
    why = [d.get("why", "fixed") for d in trace]
    greedy = [d.get("greedy") for d in trace]
    n = len(acts)

    def _first(pred):
        for i in range(n):
            if pred(i):
                return steps_[i]
        return None

    first_collapse = _first(lambda i: acts[i] == COLLAPSE)
    first_greedy_collapse = _first(lambda i: greedy[i] == COLLAPSE)
    CHOSEN = ("greedy", "fixed")          # `fixed` = a hard-coded step-function arm
    EXPLORATORY = ("forced_trial", "trial")   # scheduled, not chosen
    first_chosen_collapse = _first(lambda i: acts[i] == COLLAPSE and why[i] in CHOSEN)

    # commitment: the start of the last maximal run in which every NON-forced decision is
    # COLLAPSE (forced trials are scheduled, not chosen, so they do not break a commitment)
    commit = None
    i = n - 1
    while i >= 0 and (acts[i] == COLLAPSE or why[i] in EXPLORATORY):
        if acts[i] == COLLAPSE and why[i] in CHOSEN:
            commit = steps_[i]
        i -= 1

    # Two restore counts, because they answer different halves of "does it hold?".
    #   decline  the interval just past was COLLAPSE (from any cause) and the policy
    #            greedily chose KEEP: it had the chance to continue and did not.
    #   bail-out the interval just past was a COLLAPSE the policy CHOSE greedily, and it
    #            greedily chose KEEP: a genuine retreat out of a merge it committed to --
    #            the failure mode the spec predicts for a loop reading task NLL.
    decline_steps, bail_steps, hold_steps = [], [], []
    for i in range(1, n):
        if acts[i - 1] != COLLAPSE or why[i] in EXPLORATORY:
            continue
        if acts[i] == KEEP and why[i] in CHOSEN:
            decline_steps.append(steps_[i])
            if why[i - 1] in CHOSEN:
                bail_steps.append(steps_[i])
        else:
            hold_steps.append(steps_[i])

    switches = sum(1 for i in range(1, n) if acts[i] != acts[i - 1])
    coll = [i for i in range(n) if acts[i] == COLLAPSE]
    dt = [steps_[i + 1] - steps_[i] for i in range(n - 1)] + [max_steps - steps_[-1]]
    collapsed_steps = sum(dt[i] for i in coll)
    forced_collapse_steps = sum(dt[i] for i in coll if why[i] in EXPLORATORY)
    warm_collapse_steps = sum(dt[i] for i in coll if why[i] == "burn_in")
    return {
        "n_decisions": n,
        "first_collapse_step": first_collapse,
        "first_greedy_collapse_step": first_greedy_collapse,
        "first_chosen_collapse_step": first_chosen_collapse,
        "commit_step": commit,
        "n_bailouts": len(bail_steps),
        "bailout_steps": bail_steps[:32],
        "n_declines": len(decline_steps),
        "decline_steps": decline_steps[:32],
        "n_holds": len(hold_steps),
        "n_switches": switches,
        "frac_decisions_collapsed": len(coll) / n,
        "collapsed_steps": collapsed_steps,
        "exploratory_collapsed_steps": forced_collapse_steps + warm_collapse_steps,
        "terminal_action": acts[-1],
    }


def build_policy(spec, n_decisions):
    """`spec` is the JSON-safe policy description carried in the arm's schedule."""
    k = spec["kind"]
    if k == "fixed":
        return FixedPolicy(spec.get("merge_at"))
    if k == "paired":
        return PairedPolicy(
            spec["read"], span=spec.get("span", 1), burn=spec.get("burn", 8),
            trial_every=spec.get("trial_every", 16), n_decisions=n_decisions,
            alpha=spec.get("alpha", 0.5), v_tol=spec.get("v_tol"),
            init_action=spec.get("init_action", KEEP))
    if k == "bandit":
        return BanditPolicy(
            spec["read"], mode=spec.get("mode", "delta"),
            alpha=spec.get("alpha", 0.5), beta=spec.get("beta", 0.3),
            fast=spec.get("fast", 0.5), slow=spec.get("slow", 0.15),
            n_decisions=n_decisions, burn=spec.get("burn", 4),
            explore_start=spec.get("explore_start", 5),
            init_action=spec.get("init_action", KEEP),
            explore_growth=spec.get("explore_growth", 1.5))
    raise ValueError(k)
