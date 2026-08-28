"""maestro/policy — A2's outer loop. FORKED from `../conductor/policy.py`; every addition is
marked `# [maestro]` and the donor is untouched. `SchedulePolicy`, `YokePolicy`, `QuietPolicy`,
`ReadLedger`, `summarise_trace` and `null_abba` are the donor's, byte-for-byte, so A1's
thermostat arm replicates in-tag by running the same code object.

WHAT A2 ADDS: `LearnedPolicy` — the rule as a small LEARNED policy over (gauge readings, level
state), fitted offline (`fit.py`) with NEXT-LEVEL YIELD as the reward, against a twin of the same
class fitted with a WITHIN-LEVEL reward. Read `LearnedPolicy`'s docstring for the form and
`fit.py`'s for the fitting procedure and its measured n.

conductor/policy — the outer loop, kept OUT of the Modal app so every rule is auditable and
testable with no GPU and no substrate (`../teacher_slot/decision/policy.py`'s convention, and
`../fourwall/lm/wall.py`'s before it).

WHAT IS PORTED, AND WHAT IS NOT. The donor is `teacher_slot/decision/policy.py`'s
`PairedPolicy`: an ABBA paired trial (CKKC / KCCK) whose contrast `D = mean(u over C) -
mean(u over K)` cancels any improvement rate that is linear in time EXACTLY within one trial,
decided by `sign(V)` for a recency-weighted `V`, with a DEAD ZONE at a measured floor, fully
deterministic, and a `Budget` that charges only what the policy's input stream contains.

The donor's two conditions are REVERSIBLE and are lived in adjacent intervals: the wall token is
kept or collapsed, and a loop that reads task NLL *can* bail out of a merge at M+125. This
round's actions are not reversible. `commit` freezes a table into the ratchet (`committed[l]` is
never unset, and the next level can only ever be built over what was frozen); `era advance` moves
the world's damage cell one level deeper and there is no way back. So the ABBA block cannot be
run over the ACTIONS: there is no C-K-K-C over "committed / not committed".

Two ports were available and the second was taken:

  (a) TRIAL-COMMIT. Make commit temporarily reversible — install the candidate table for the C
      quarters of a block, remove it afterwards — and keep the donor's contrast verbatim. This is
      structurally faithful (the donor accepts that an exploratory interval really does perturb
      the learner) but it is not sound HERE: during a C quarter the arm mines the next level over
      a committed lower table and the plant trains on trajectories that used the macro, and
      neither observation can be taken back when the table is removed. The block would leave a
      permanent residue of exactly the treatment it is trying to measure reversibly.

  (b) WHAT THIS FILE DOES. Keep everything in the donor that is the DISCIPLINE and replace the
      estimator, because the estimator's job here is different. The donor's block exists to
      cancel the ambient improvement trend, which is a NUISANCE when the question is "which of
      two conditions is better". For an absorbing action the question is "has the currency this
      level mints stopped moving" — the ambient trend IS the signal, and there is no second
      condition to cancel it against. What ports unchanged:

        * one SCALAR read per arm, in ERROR convention (lower is better), so arms differ in
          nothing but the read;
        * a recency-weighted `V` over block estimates, and a decision by SIGN against a dead
          zone whose value is MEASURED on the statistic the policy actually thresholds
          (`floors.py`, the null-ABBA method) and never defaulted;
        * the donor's 5-read block geometry (`W` quarters of `span` decision points each), so
          the ledger charges a block-shaped input stream and the arms' read cadence is common;
        * full determinism — no policy consumes one draw of any RNG, so the donor's
          zero-global-RNG design survives;
        * "positive and then flat": the loop may not act on a gauge that has never moved. The
          donor's certificate spells this `lp_min_drop`; here it is expressed in FLOOR UNITS and
          so carries no new constant — the gauge must have been observed above its own dead zone
          at least once since the last action.

      The estimator becomes `D = mean(u)` over the block's quarters — the paired-interval
      improvement RATE — and the rule is HOLD while `V > v_tol`, ACT when `V <= v_tol`.

  THE DONOR'S CONTRAST IS STILL COMPUTED, at every block, in every arm, and it is never driven.
  On a fixed-condition series its true value is exactly zero, so the live `N` series IS the
  instrument's own noise floor measured in-tag — the same null-ABBA statistic `floors.py`
  derives offline from `as_s0`/`cs_s0`, now measured on the run's own data. The exact variance
  relation between the two statistics is the reason one calibrates the other: for iid quarter
  improvements of variance sigma^2, Var(N) = sigma^2 and Var(D) = sigma^2 / W, so at W = 4 the
  dead zone for `D` is `sd(N) / 2`.

SAID PLAINLY, so no reader has to infer it: THIS IS NOT AN ABBA PAIRED TRIAL. It is a
thermostat on a paired-interval slope with a measured dead zone, and its nearest ancestor in
this repo is the census's G-A rule ("quiet iff the level admitted no more than theta new tuples
over the trailing W cycles"), which `census/README.md` finding 4 measured to be indistinguishable
from a clock. That history is exactly why every gauge-driven arm in this round carries a
YOKED-CLOCK control, and why the reduction reports gauge-vs-yoke divergence cycles first.
"""

HOLD, COMMIT, ADVANCE = "hold", "commit", "advance"
ACTIONS = (HOLD, COMMIT, ADVANCE)


class Policy:
    """Base. `step(cyc, reads) -> info` pushes one decision point; `quiet` is the licence to
    act; `acted(kind, cyc)` tells the policy an absorbing action was taken and re-arms it.
    `needs` names the readouts whose cost the ledger charges (empty = this policy buys
    nothing)."""

    needs = ()
    kind = "base"
    read_key = None

    def needs_at(self, cyc):
        return self.needs

    @property
    def quiet(self):
        return False

    def step(self, cyc, reads):
        return {}

    def acted(self, kind, cyc, why=""):
        pass


class SchedulePolicy(Policy):
    """The fixed ladder: never acts. This is the FIDELITY GATE — run under it the fork must
    reproduce its `assay.py` twin bit for bit, because it reads nothing that is charged, probes
    nothing that consumes RNG, and leaves the commit rule as the donor's `delta_prov`."""

    kind = "schedule"

    def step(self, cyc, reads):
        return {"rule": "schedule", "quiet": False}


class YokePolicy(Policy):
    """The census's timing control (`yoked_delay`), generalised to both actions: replay a gauge
    arm's MEASURED action cycles by clock and by nothing else. The plan is not hard-coded — the
    entrypoint runs the gauge arm first in the same job and passes its realised actions in.

    Every era advance is in the plan, including the ones the gauge arm took because it hit its
    cap, so the yoke never has to agree with the caps to reproduce the schedule."""

    kind = "yoke"

    def __init__(self, plan):
        self.plan = [dict(p) for p in (plan or [])]
        self.commit_at = {int(p["cycle"]) for p in self.plan if p["kind"] == COMMIT}
        self.advance_at = {int(p["cycle"]) for p in self.plan if p["kind"] == ADVANCE}
        self._cyc = None

    def step(self, cyc, reads):
        self._cyc = int(cyc)
        return {"rule": "yoke", "quiet": False,
                "yoke_commit": int(cyc) in self.commit_at,
                "yoke_advance": int(cyc) in self.advance_at}

    def commit_now(self, cyc):
        return int(cyc) in self.commit_at

    def advance_now(self, cyc):
        return int(cyc) in self.advance_at


class QuietPolicy(Policy):
    """THE THERMOSTAT. One scalar read, in error convention; hold while it still moves, act when
    it quiets inside its own measured dead zone.

    read_key   which readout the policy's input stream contains. This is the ONLY thing that
               differs between `outer_ledger`, `outer_yield` and `outer_endo`.
    span       decision points per block quarter (the donor's measurement HORIZON axis).
    W          quarters per block; 4 keeps the donor's CKKC geometry so the null contrast is
               the donor's statistic exactly.
    burn       decision points held after every action (and at every era start) before the loop
               may act again. The analogue of the donor's burn-in and of the certificate's
               `sil_min_cycle`: a level whose regime has just changed has not been observed yet.
    alpha      recency weight on the block estimates.
    v_tol      THE DEAD ZONE. Required — there is no default, because a defaulted floor is a
               chosen one. `floors.py` derives it; `v_tol_by_level` supplies a per-read-level
               value for `yield`, whose instrument (a distinct-tuple count) has a different
               noise scale at level 3 than at level 4.

    The buffer, `V`, and the `moved` latch are all reset on every action and at every era start,
    because each of those changes the regime the gauge is measured in. Nothing here draws from
    any RNG, and `step` is a pure function of the reads it has been given.
    """

    kind = "quiet"

    def __init__(self, read_key, v_tol=None, v_tol_by_level=None, span=1, W=4, burn=4,
                 alpha=0.5, priced=False):
        if v_tol is None and not v_tol_by_level:
            raise ValueError(
                f"QuietPolicy({read_key!r}) needs a MEASURED dead zone: pass v_tol or "
                f"v_tol_by_level. See floors.py — a defaulted floor is a chosen one.")
        self.read_key = read_key
        self.v_tol_default = None if v_tol is None else float(v_tol)
        self.v_tol_by_level = {int(k): float(x) for k, x in (v_tol_by_level or {}).items()}
        self.span, self.W, self.burn = int(span), int(W), int(burn)
        self.alpha = float(alpha)
        self.priced = bool(priced)
        self.needs = (read_key,) if priced else ()
        self.n_reads = 0
        self.n_blocks = 0
        self.trace = []
        self._bucket = None
        self.reset("init")

    # -- the dead zone ---------------------------------------------------------------- #
    def tol_for(self, level):
        if level is not None and int(level) in self.v_tol_by_level:
            return self.v_tol_by_level[int(level)]
        if self.v_tol_default is None:
            raise KeyError(f"no measured dead zone for read {self.read_key} at level {level}")
        return self.v_tol_default

    # -- state ------------------------------------------------------------------------ #
    def reset(self, why):
        self.buf = []
        self.V = None
        self.moved = False
        self.n_since = 0
        self._quiet = False
        self.last_reset = why

    @property
    def quiet(self):
        return bool(self._quiet)

    def acted(self, kind, cyc, why=""):
        self.reset(f"{kind}@c{cyc}" + (f":{why}" if why else ""))

    # -- the read -> decision map ------------------------------------------------------ #
    def step(self, cyc, reads):
        e = float(reads[self.read_key])
        level = reads.get(f"{self.read_key}_level")
        tol = self.tol_for(level)
        self.buf.append(float(e))
        self.n_since += 1
        self.n_reads += 1
        info = {"rule": "quiet", "cycle": int(cyc), "read": self.read_key, "e": e,
                "read_level": None if level is None else int(level), "v_tol": tol,
                "D": None, "N": None, "V": self.V, "polarity": None,
                "n_since": self.n_since, "moved": self.moved, "quiet": False,
                "in_burn": self.n_since <= self.burn}
        need = self.W * self.span + 1
        if len(self.buf) >= need:
            pts = [self.buf[-1 - (self.W - k) * self.span] for k in range(self.W + 1)]
            u = [pts[k] - pts[k + 1] for k in range(self.W)]      # >0 == the error fell
            D = sum(u) / len(u)
            # the DONOR'S CONTRAST, computed and never driven: on a fixed-condition series its
            # true value is exactly zero, so its running spread is the in-tag noise floor.
            letters = ("CKKC" if self.n_blocks % 2 == 0 else "KCCK")[:self.W]
            uc = [u[k] for k in range(self.W) if letters[k] == "C"]
            uk = [u[k] for k in range(self.W) if letters[k] == "K"]
            # the null contrast needs BOTH letters in the block, so it is undefined below W=2.
            # It is a logged instrument and never drives anything, so an undefined N simply
            # means this geometry carries no in-tag floor meter — which `preflight`'s collapsed
            # W=1 geometry is the only thing that ever does.
            N = (sum(uc) / len(uc) - sum(uk) / len(uk)) if (uc and uk) else None
            self.n_blocks += 1
            self.V = D if self.V is None else self.V + self.alpha * (D - self.V)
            if self.V > tol:
                self.moved = True        # "positive and then flat", in floor units
            info.update({"D": D, "N": N, "V": self.V, "u": list(u), "polarity": letters,
                         "moved": self.moved})
        self._quiet = bool(self.V is not None and self.moved and self.n_since > self.burn
                           and self.V <= tol)
        info["quiet"] = self._quiet
        info["v_mult"] = None if self.V is None else self.V / tol if tol else None
        self.trace.append(info)
        return info


# --------------------------------------------------------------------------- #
# [maestro] THE LEARNED RULE
# --------------------------------------------------------------------------- #

GAUGES = ("ledger", "yield", "endo_excess")


class LearnedPolicy(Policy):
    """A2's rule: the same thermostat SHAPE with a LEARNED GAUGE MIXTURE, indexed by level
    state, fitted offline with next-level yield as the reward (`fit.py`).

    THE FORM, and why it is this one. A1's thermostat is `hold while V > v_tol`, where `V` is
    one hand-picked read's recency-weighted paired-interval slope and `v_tol` is that read's
    MEASURED dead zone. Write the slope in FLOOR UNITS, `Vhat_k = V_k / tol_k`, and A1's rule is
    exactly `hold while Vhat_yield > 1`. So the natural learned generalisation is a MIXTURE over
    the floor-normalised slopes:

        Vhat(t) = sum_k a[s(t)]_k * V_k(t) / tol_k(t)          decision statistic
        theta(t) = theta[s(t)]                                  its own MEASURED dead zone
        rule:  HOLD while Vhat > theta ;  ACT when Vhat <= theta

    with `a` a unit-norm weight vector per level-state bucket `s`, and `theta` measured on a
    fixed-condition reference series by the DONOR'S OWN null-ABBA method applied to the same
    mixture (`Nhat = sum_k a_k N_k / tol_k`, whose true value is zero under any linear trend, so
    `theta = sd(Nhat) / 2` at W=4 exactly as A1's `v_tol = sd(N)/2`).

    THREE PROPERTIES THIS FORM BUYS, all of them load-bearing:

      * IT CONTAINS THE THERMOSTAT EXACTLY. `a = e_yield` gives `Vhat = V_yield / tol_yield` and
        `theta = sd(N_yield)/(2*tol_yield) = 1`, i.e. `V_yield <= tol_yield` — A1's rule, decision
        for decision. Gate L-1 asserts this against the donor class on live series, and L-2
        asserts the floor construction reproduces A1's measured number on A1's own reference arm.
        So "does learning the rule buy anything" is asked INSIDE one hypothesis class, with the
        hand-written rule as a named point in it.

      * IT IS SCALE-FREE, so the fit cannot make the policy degenerate. `Vhat` and `theta` are
        both linear in `a`, so the decision depends only on `a`'s DIRECTION. Only the direction
        is learned; the scale is set by the measured floor. (The first form tried — a fitted
        VALUE `b + w.x` thresholded at its own noise — was rejected offline for exactly this:
        ridge shrinkage pulls the prediction to its mean, ~0.86 tuples/cycle, which never
        reaches any noise floor, so the arm would never have acted. Recorded because the failure
        is a property of the form, not of the corpus.)

      * IT READS LEVEL STATE. `a` and `theta` are a TABLE over a two-cell discretisation of level
        state: `will_commit = (active <= max_macro_level) and (committed[active] is None)` — i.e.
        whether a firing would INSTALL A TABLE or MOVE THE WORLD. A1's thermostat runs one rule
        for both actions and cannot tell them apart; this is the smallest level-state split that
        is semantically the action's own type. The bucket is read off the panel at the decision
        instant, before block (g), exactly as the offline fit reads it.

    WHAT IS SHARED WITH THE DONOR BY IDENTITY, NOT BY COPY: the per-gauge estimator is the
    donor's `QuietPolicy`, instantiated once per gauge and consumed for its `V` and `N` only.
    The block geometry, the EWMA, the burn-in, the "positive and then flat" latch and the reset
    on every action and era start are therefore A1's, not a re-implementation of A1's.

    THE ROUND-1 POSTMORTEM (`../teacher_slot/decision/policy.py::PairedPolicy`) is answered
    STRUCTURALLY rather than by burn-in: there is no within-run learner here. The policy is
    fitted on a FIXED offline corpus and FROZEN before the run starts, so there is no learning
    curve whose slope can swamp an action differential, no baseline to decay, and no sampling
    order to bias — the three mechanisms that made round 1 measure its own instrument. Nothing
    in `step` consumes an RNG draw or updates a parameter.
    """

    kind = "learned"

    def __init__(self, mixes, thetas, floors, gauges=GAUGES, span=1, W=4, burn=4,
                 alpha=0.5, priced=True, reward=None, fit_id=None):
        if not mixes or not thetas:
            raise ValueError("LearnedPolicy needs a fitted mixture and a MEASURED theta per "
                             "bucket; see fit.py — a defaulted floor is a chosen one.")
        self.gauges = tuple(gauges)
        self.mixes = {str(k): [float(x) for x in v] for k, v in mixes.items()}
        self.thetas = {str(k): float(x) for k, x in thetas.items()}
        for k, v in self.mixes.items():
            if len(v) != len(self.gauges):
                raise ValueError(f"mix[{k}] has {len(v)} weights for {len(self.gauges)} gauges")
            if k not in self.thetas:
                raise ValueError(f"no measured theta for bucket {k}")
            if self.thetas[k] <= 0:
                raise ValueError(f"theta[{k}] = {self.thetas[k]} is not positive")
        self.floors = dict(floors or {})
        self.span, self.W, self.burn = int(span), int(W), int(burn)
        self.alpha = float(alpha)
        self.priced = bool(priced)
        self.reward = reward
        self.fit_id = fit_id
        self.read_key = f"learned:{reward}" if reward else "learned"
        # the INPUT STREAM: every gauge the mixture consumes. `ReadLedger.FREE` makes the
        # learner's own experience free and prices only the counterfactual forward pass.
        self.needs = tuple(self.gauges)
        # the donor's estimator, one per gauge, used for its V and N ONLY.
        self._est = {}
        for g in self.gauges:
            by_level = self.floors.get(f"{g}_by_level")
            tol = self.floors.get(g)
            self._est[g] = QuietPolicy(g, v_tol=tol, v_tol_by_level=by_level, span=self.span,
                                       W=self.W, burn=self.burn, alpha=self.alpha)
        self.n_reads = 0
        self.n_blocks = 0
        self.trace = []
        self._bucket = None
        self.reset("init")

    # -- the normalising unit: the gauge's own MEASURED dead zone --------------------- #
    def tol_for(self, gauge, level):
        return self._est[gauge].tol_for(level)

    def bucket(self, reads):
        wc = reads.get("will_commit")
        if wc is None:
            raise KeyError("LearnedPolicy reads level state: the panel must carry "
                           "`will_commit` at the decision instant (before block (g))")
        return "1" if int(wc) else "0"

    # -- state ------------------------------------------------------------------------ #
    def reset(self, why):
        self.V = None
        self.moved = False
        self.n_since = 0
        self._quiet = False
        self.last_reset = why
        for e in self._est.values():
            e.reset(why)

    @property
    def quiet(self):
        return bool(self._quiet)

    def acted(self, kind, cyc, why=""):
        self.reset(f"{kind}@c{cyc}" + (f":{why}" if why else ""))

    # -- the read -> decision map ------------------------------------------------------ #
    def step(self, cyc, reads):
        b = self.bucket(reads)
        # A BUCKET CHANGE IS A REGIME CHANGE, so it resets exactly as an action and an era start
        # do -- the donor's stated reason ("each of those changes the regime the gauge is
        # measured in"), applied to the one piece of state A1's rule did not have. Without it a
        # statistic armed under the commit mixture could licence an action under the advance
        # mixture, which is not "positive and then flat" but "positive in one regime, then flat
        # in another". Live this fires only where an action or an era start already reset (the
        # bucket can only flip at an era boundary or at this policy's own commit), so it changes
        # no live trajectory; it is what makes the OFFLINE REPLAY over another arm's series
        # honest, where the commits in the series are not this policy's own.
        if self._bucket is not None and b != self._bucket:
            self.reset(f"bucket{self._bucket}->{b}@c{cyc}")
        self._bucket = b
        self.n_since += 1
        self.n_reads += 1
        a = self.mixes[b]
        tol_theta = self.thetas[b]
        per, Vs, Ns = {}, [], []
        for g in self.gauges:
            val = reads.get(g)
            if val is None:
                per[g] = {"e": None, "V": None, "N": None, "tol": None}
                Vs.append(None)
                Ns.append(None)
                continue
            info = self._est[g].step(cyc, reads)
            t = info["v_tol"]
            v = None if info["V"] is None else info["V"] / t
            n = None if info["N"] is None else info["N"] / t
            per[g] = {"e": info["e"], "V": v, "N": n, "tol": t,
                      "read_level": info.get("read_level")}
            Vs.append(v)
            Ns.append(n)
        ready = all(v is not None for v in Vs)
        if ready:
            self.V = float(sum(a[i] * Vs[i] for i in range(len(a))))
            self.n_blocks += 1
            if self.V > tol_theta:
                self.moved = True                 # "positive and then flat", in floor units
        Nhat = (float(sum(a[i] * Ns[i] for i in range(len(a))))
                if all(n is not None for n in Ns) else None)
        self._quiet = bool(self.V is not None and self.moved and self.n_since > self.burn
                           and self.V <= tol_theta)
        info = {"rule": "learned", "cycle": int(cyc), "read": self.read_key,
                "reward": self.reward, "bucket": b, "a": list(a),
                "V": self.V, "N": Nhat, "D": None, "v_tol": tol_theta,
                "v_mult": None if self.V is None else self.V / tol_theta,
                "per_gauge": per, "n_since": self.n_since, "moved": self.moved,
                "in_burn": self.n_since <= self.burn, "quiet": self._quiet,
                "read_level": per.get("yield", {}).get("read_level")}
        self.trace.append(info)
        return info


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

class ReadLedger:
    """ear's convention, transplanted: THE LEARNER'S OWN EXPERIENCE IS FREE, A COUNTERFACTUAL
    READ IS PRICED, and instruments logged for science in every arm are free and identical
    across arms — the ledger charges only what the POLICY's input stream contains.

      ledger  the arm's own metering error on the era's own cell. Already computed by the cycle
              the policy reads it; the arm would have paid for it whether or not a loop existed.
              FREE, exactly as the donor's `task` read is free.
      yield   distinct level-(l+1) tuples at support in the arm's own chosen trajectories.
              Mined from trajectories the agent produced anyway, by a numpy counter. FREE.
      endo    the plant's own masked-infill NLL over the span one level above the era's damage
              cell. This is a forward pass the arm does NOT otherwise make, so it is PRICED, at
              `price` grounding-equivalents per read (measured by `endo_bench`, not assumed).

    `spend` is in grounding-equivalents and is folded into the arm's own `counts["ground"]`, so
    it shows up in `t_cum` — the run's priced-time readout — and nowhere else. Nothing in the
    cycle loop reads `t_cum`, so pricing a read cannot change what any arm does; the ledger is a
    readout, not a governor. That is asserted by the yoked arms, which pay nothing and must
    reproduce their gauge arm's trajectory exactly."""

    FREE = ("ledger", "yield")

    def __init__(self, prices=None):
        self.prices = {k: int(x) for k, x in (prices or {}).items()}
        self.spend = 0
        self.n_priced = 0
        self.n_reads = {}
        self.by_read = {}

    def price_of(self, key):
        return 0 if key in self.FREE else int(self.prices.get(key, 0))

    def charge(self, needs):
        c = 0
        for k in needs:
            p = self.price_of(k)
            c += p
            self.n_reads[k] = self.n_reads.get(k, 0) + 1
            self.by_read[k] = self.by_read.get(k, 0) + p
            if p:
                self.n_priced += 1
        self.spend += c
        return c

    def state(self):
        return {"spend_g": int(self.spend), "n_priced_reads": int(self.n_priced),
                "n_reads": dict(self.n_reads), "by_read": dict(self.by_read),
                "prices": dict(self.prices)}


# --------------------------------------------------------------------------- #
# trace reduction (pure; used by the analyzer and by the offline gate)
# --------------------------------------------------------------------------- #

def summarise_trace(trace, actions, n_cycles):
    """The committed decision readouts, from the decision trace and the realised actions alone.

    `trace` is the policy's per-cycle info list; `actions` the realised
    `[{cycle, kind, level, why, era}]`. The donor's `summarise_trace` counted bail-outs, which
    only exist for a reversible action; the absorbing analogue is HOW LONG THE LOOP HELD and
    WHAT ENDED EACH HOLD — a chosen action, or the cap running out under it."""
    acts = [dict(a) for a in (actions or [])]
    # A CANCELLED attempt is not a commit. The rule licensed one and the substrate refused it
    # (the mined table was empty), which counts as an action for the loop's clock but installs
    # nothing — so it must never appear in `commit_cycles`. Caught on `cd_smoke`, where at quick
    # scale nothing is minable and every arm's licensed commits were cancelled: reported as
    # commits, they would have been read as the loop committing where no commit occurred.
    cancelled = [a for a in acts if a["kind"] == COMMIT and a.get("cancelled")]
    commits = [a for a in acts if a["kind"] == COMMIT and not a.get("cancelled")]
    advs = [a for a in acts if a["kind"] == ADVANCE]
    chosen = [a for a in acts if a.get("why") == "quiet" and not a.get("cancelled")]
    capped = [a for a in acts if a.get("why") == "cap"]
    out = {
        "n_decisions": len(trace or []),
        "n_actions": len(acts),
        "n_commits": len(commits),
        "commit_cycles": [a["cycle"] for a in commits],
        "commit_levels": [a.get("level") for a in commits],
        "n_commits_cancelled": len(cancelled),
        "cancelled_cycles": [a["cycle"] for a in cancelled],
        "cancelled_levels": [a.get("level") for a in cancelled],
        "n_advances": len(advs),
        "advance_cycles": [a["cycle"] for a in advs],
        "n_chosen": len(chosen), "n_capped": len(capped),
        "chosen_cycles": [a["cycle"] for a in chosen],
        "capped_cycles": [a["cycle"] for a in capped],
        "rode_cap_to_end": bool(acts and acts[-1].get("why") == "cap"),
        "n_cycles": int(n_cycles),
    }
    if trace:
        V = [d.get("V") for d in trace if d.get("V") is not None]
        N = [d.get("N") for d in trace if d.get("N") is not None]
        mult = [d.get("v_mult") for d in trace if d.get("v_mult") is not None]
        out.update({
            "n_blocks": len(N),
            "frac_quiet": sum(1 for d in trace if d.get("quiet")) / len(trace),
            "frac_armed": sum(1 for d in trace if d.get("moved")) / len(trace),
            "V_mean": (sum(V) / len(V)) if V else None,
            "V_min": min(V) if V else None, "V_max": max(V) if V else None,
            "v_mult_mean": (sum(mult) / len(mult)) if mult else None,
            "null_N_mean": (sum(N) / len(N)) if N else None,
            "null_N_sd": _sd(N), "in_tag_floor": (_sd(N) / 2.0) if N else None,
            "v_tol": trace[-1].get("v_tol"),
        })
    return out


def _sd(xs):
    if not xs or len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def null_abba(series, skip=(), span=1, W=4):
    """THE NULL-ABBA FLOOR METHOD (`teacher_slot/endo_yield/SPEC.md` §4), as a pure function.

    On a fixed-condition series the true contrast is exactly zero, so every fabricated block is
    a draw from the instrument's own noise. Returns `(N, D)` — the donor's contrast and this
    file's decision statistic — over every window of `W*span+1` points that contains no index in
    `skip` (era boundaries and commit cycles are regime changes, not noise)."""
    Ns, Ds = [], []
    n = len(series)
    skip = set(skip)
    for t0 in range(0, n - W * span):
        idx = [t0 + k * span for k in range(W + 1)]
        if any(j in skip for j in idx):
            continue
        vals = [float(series[j]) for j in idx]
        u = [vals[k] - vals[k + 1] for k in range(W)]
        letters = ("CKKC" if len(Ns) % 2 == 0 else "KCCK")[:W]
        uc = [u[k] for k in range(W) if letters[k] == "C"]
        uk = [u[k] for k in range(W) if letters[k] == "K"]
        if not (uc and uk):
            continue
        Ns.append(sum(uc) / len(uc) - sum(uk) / len(uk))
        Ds.append(sum(u) / len(u))
    return Ns, Ds


def build_policy(spec, floors=None, yoke_plan=None):
    """`spec` is the JSON-safe policy description carried in the arm's schedule."""
    import json
    k = spec.get("kind")
    if k in (None, "schedule"):
        return SchedulePolicy()
    if k == "yoke":
        return YokePolicy(yoke_plan)
    if k == "quiet":
        f = dict(floors or {})
        read = spec["read"]
        by_level = f.get(f"{read}_by_level")
        tol = spec.get("v_tol", f.get(read))
        return QuietPolicy(read, v_tol=tol, v_tol_by_level=by_level,
                           span=spec.get("span", 1), W=spec.get("W", 4),
                           burn=spec.get("burn", 4), alpha=spec.get("alpha", 0.5),
                           priced=spec.get("priced", False))
    # [maestro] the learned rule. `fit` is the JSON-safe fitted object `fit.py` writes: the
    # per-bucket mixture and the per-bucket MEASURED theta, plus the fit's own provenance. The
    # per-gauge FLOORS come from the run's config (the same measured dead zones A1 was governed
    # by), because they are the mixture's normalising units and must not drift from A1's.
    if k == "learned":
        fitobj = spec.get("fit") or {}
        if isinstance(fitobj, str):
            fitobj = json.loads(fitobj)
        if not fitobj:
            raise ValueError("a learned arm needs its fitted policy (--fit / spec['fit']); "
                             "see fit.py — an unfitted learned rule is a chosen one.")
        return LearnedPolicy(fitobj["mixes"], fitobj["thetas"], floors or {},
                             gauges=tuple(fitobj.get("gauges", GAUGES)),
                             span=spec.get("span", 1), W=spec.get("W", 4),
                             burn=spec.get("burn", 4), alpha=spec.get("alpha", 0.5),
                             priced=spec.get("priced", True),
                             reward=fitobj.get("reward"), fit_id=fitobj.get("fit_id"))
    raise ValueError(k)


# --------------------------------------------------------------------------- #
# THE OFFLINE GATE SUITE — no GPU, no substrate, no Modal.
# --------------------------------------------------------------------------- #

def policy_gate(verbose=True):
    """The donor's own gate suite (`teacher_slot/decision`'s P-1..P-10), re-asked of the ported
    rule. Every check is on the statistic the policy actually thresholds."""
    out = {}

    def _say(k, ok, detail):
        out[k] = {"pass": bool(ok), **detail}
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")
        return ok

    ok_all = True

    # P-1  TREND CANCELLATION. Under an improvement rate linear in time the donor's contrast is
    #      exactly zero. This is the property the null floor rests on, so it is checked first.
    e, val = [], 100.0
    for i in range(60):
        e.append(val)
        val -= (5.0 - 0.05 * i)                      # improvement linear in time
    N, D = null_abba(e, span=1, W=4)
    ok_all &= _say("P-1 trend cancellation", max(abs(x) for x in N) < 1e-9,
                   {"max_abs_N": max(abs(x) for x in N), "n": len(N),
                    "mean_D": sum(D) / len(D)})

    # P-2  THE VARIANCE RELATION the dead zone is derived through: for iid quarter improvements,
    #      sd(D) == sd(N) / W**0.5 at W quarters, i.e. sd(N)/2 at W=4.
    import random as _r
    _r.seed(0)
    noise = [_r.gauss(0, 1) for _ in range(20000)]
    ser = [0.0]
    for x in noise:
        ser.append(ser[-1] - x)                      # u == the iid draws
    N2, D2 = null_abba(ser, span=1, W=4)
    ratio = _sd(N2) / _sd(D2)
    ok_all &= _say("P-2 variance relation sd(N)/sd(D) == 2", abs(ratio - 2.0) < 0.05,
                   {"sd_N": _sd(N2), "sd_D": _sd(D2), "ratio": ratio})

    # P-3  MIRROR SYMMETRY UNDER RELABELLING. Negating the series negates V exactly, so nothing
    #      in the rule prefers one direction of the read; only the dead-zone comparison is
    #      asymmetric, and that asymmetry is the RULE ("act when it quiets"), not a bias.
    p = QuietPolicy("x", v_tol=0.0)
    q = QuietPolicy("x", v_tol=0.0)
    mism = 0
    for i, val in enumerate(ser[:200]):
        a = p.step(i, {"x": val})
        b = q.step(i, {"x": -val})
        if a["V"] is not None and abs(a["V"] + b["V"]) > 1e-12:
            mism += 1
    ok_all &= _say("P-3 mirror symmetry under relabelling", mism == 0, {"mismatches": mism})

    # P-4  DEAD ZONE ON A NULL SERIES. A flat series with float residue must never license an
    #      action: the donor caught `sign(V)` flipping on 2.2e-16 of floating-point dust.
    p = QuietPolicy("x", v_tol=1e-6, burn=2)
    fired = 0
    for i in range(50):
        p.step(i, {"x": 1.0 + (1e-16 if i % 2 else -1e-16)})
        fired += int(p.quiet)
    ok_all &= _say("P-4 dead zone on a null series", fired == 0,
                   {"n_quiet": fired, "note": "never moved, so never armed"})

    # P-5  "POSITIVE AND THEN FLAT". A gauge that has never risen above its own floor cannot
    #      license an action, however quiet it is; one that moves and then quiets can.
    p = QuietPolicy("x", v_tol=0.5, burn=2)
    for i in range(20):
        p.step(i, {"x": 10.0})
    never = p.quiet
    q = QuietPolicy("x", v_tol=0.5, burn=2)
    seq = [10.0 - 2.0 * i for i in range(10)] + [(-10.0)] * 20      # falls fast, then flat
    fires_at = None
    for i, val in enumerate(seq):
        q.step(i, {"x": val})
        if q.quiet and fires_at is None:
            fires_at = i
    ok_all &= _say("P-5 positive-then-flat precondition",
                   (not never) and fires_at is not None,
                   {"dead_gauge_quiet": never, "moved_gauge_first_quiet_at": fires_at})

    # P-6  RESET ON ACTION. After an action the loop must re-arm from scratch: no V, no latch,
    #      and a fresh burn-in, so one quiet reading cannot fire two absorbing actions.
    q.acted(COMMIT, 99)
    ok_all &= _say("P-6 reset on action",
                   (q.V is None) and (not q.moved) and (q.n_since == 0) and (not q.quiet),
                   {"V": q.V, "moved": q.moved, "n_since": q.n_since})

    # P-7  DETERMINISM AND RNG-NEUTRALITY. Two instances fed the same reads agree bit for bit,
    #      and the policy consumes no draw of the global streams it shares with the substrate.
    st = _r.getstate()
    a = QuietPolicy("x", v_tol=0.1)
    b = QuietPolicy("x", v_tol=0.1)
    va, vb = [], []
    for i, val in enumerate(ser[:300]):
        va.append(a.step(i, {"x": val})["V"])
        vb.append(b.step(i, {"x": val})["V"])
    ok_all &= _say("P-7 determinism + RNG neutrality",
                   va == vb and _r.getstate() == st,
                   {"identical": va == vb, "rng_untouched": _r.getstate() == st})

    # P-8  THE READ KEY IS A PURE LABEL. Identical numbers under different read names give
    #      identical decisions — the arms differ in what they read and in nothing else.
    a = QuietPolicy("yield", v_tol=0.1)
    b = QuietPolicy("endo", v_tol=0.1)
    same = True
    for i, val in enumerate(ser[:300]):
        same &= (a.step(i, {"yield": val})["quiet"] == b.step(i, {"endo": val})["quiet"])
    ok_all &= _say("P-8 read key is a pure label", same, {"identical_decisions": same})

    # P-9  LEDGER ARITHMETIC. A free read charges nothing; a priced read charges its measured
    #      cost and increments the priced-read counter.
    L = ReadLedger({"endo": 77})
    L.charge(("ledger",)); L.charge(("yield",)); L.charge(("endo",)); L.charge(("endo",))
    st_ = L.state()
    ok_all &= _say("P-9 ledger arithmetic",
                   st_["spend_g"] == 154 and st_["n_priced_reads"] == 2
                   and st_["by_read"]["ledger"] == 0 and st_["by_read"]["yield"] == 0,
                   st_)

    # P-10 NO DEFAULTED FLOOR. Constructing a driven policy without a measured dead zone is an
    #      error, not a silent zero.
    try:
        QuietPolicy("yield")
        raised = False
    except ValueError:
        raised = True
    ok_all &= _say("P-10 no defaulted dead zone", raised, {"raised": raised})

    # P-11 THE YOKE REPLAYS BY CLOCK AND BY NOTHING ELSE.
    plan = [{"cycle": 7, "kind": COMMIT, "level": 2}, {"cycle": 9, "kind": ADVANCE},
            {"cycle": 20, "kind": ADVANCE}]
    y = YokePolicy(plan)
    hits = [(c, y.commit_now(c), y.advance_now(c)) for c in (6, 7, 8, 9, 20, 21)]
    ok_all &= _say("P-11 yoke replays by clock",
                   hits == [(6, False, False), (7, True, False), (8, False, False),
                            (9, False, True), (20, False, True), (21, False, False)],
                   {"hits": hits})

    # P-12 SPAN IS THE HORIZON AXIS: at span s the block spans W*s decision points and the
    #      estimate is the mean improvement per span, so a slower series is readable at a
    #      longer horizon and the geometry is otherwise unchanged.
    p1 = QuietPolicy("x", v_tol=0.0, span=1)
    p2 = QuietPolicy("x", v_tol=0.0, span=2)
    lin = [100.0 - 1.0 * i for i in range(40)]
    for i, val in enumerate(lin):
        p1.step(i, {"x": val}); p2.step(i, {"x": val})
    ok_all &= _say("P-12 span is the horizon axis",
                   abs(p1.V - 1.0) < 1e-9 and abs(p2.V - 2.0) < 1e-9,
                   {"V_span1": p1.V, "V_span2": p2.V})

    # ------------------------------------------------------------------ #
    # [maestro] THE LEARNED RULE'S OWN GATES, L-1..L-7. Every check is on the statistic the
    # learned policy actually thresholds, and the first one is the round's structural claim.
    # ------------------------------------------------------------------ #
    FL = {"ledger": 0.02944712566990095, "endo_excess": 0.01091575129919287,
          "yield_by_level": {3: 0.46127129019246205, 4: 0.5166900731510206}}

    def _mk(mixes, thetas, **kw):
        return LearnedPolicy(mixes, thetas, FL, **kw)

    def _reads(i, x, wc=1, lvl=3):
        return {"ledger": x, "yield": x, "endo_excess": x, "yield_level": lvl,
                "will_commit": wc}

    # L-1  THE LEARNED CLASS CONTAINS THE THERMOSTAT, EXACTLY. With the mixture on `yield`
    #      alone and theta at 1 floor unit, `LearnedPolicy` must agree with the DONOR CLASS
    #      `QuietPolicy('yield')` decision for decision on a live series. This is what makes
    #      "does learning the rule buy anything" a question inside one hypothesis class.
    _r.seed(1)
    ser2 = [0.0]
    for _ in range(400):
        ser2.append(ser2[-1] - _r.gauss(0.02, 1.0))
    lp = _mk({"1": [0.0, 1.0, 0.0], "0": [0.0, 1.0, 0.0]}, {"1": 1.0, "0": 1.0})
    qp = QuietPolicy("yield", v_tol_by_level=FL["yield_by_level"])
    dis, nq = 0, 0
    for i, x in enumerate(ser2):
        rd = _reads(i, x)
        A = lp.step(i, rd)["quiet"]
        B = qp.step(i, rd)["quiet"]
        dis += int(A != B)
        nq += int(A)
        if A:
            lp.acted(COMMIT, i)
            qp.acted(COMMIT, i)
    ok_all &= _say("L-1 learned class contains the thermostat exactly",
                   dis == 0 and nq > 0,
                   {"disagreements": dis, "n_decisions": len(ser2), "n_quiet": nq})

    # L-2  THE FLOOR CONSTRUCTION REDUCES TO THE DONOR'S. `theta = sd(Nhat)/2` with the
    #      mixture on one gauge and NO floor normalisation must be `null_abba`'s own
    #      `sd(N)/2` on the same series — i.e. the new code path recovers A1's number.
    Nser, _D = null_abba(ser2, span=1, W=4)
    donor_floor = _sd(Nser) / 2.0
    lp2 = _mk({"1": [0.0, 1.0, 0.0], "0": [0.0, 1.0, 0.0]},
              {"1": 1.0, "0": 1.0},
              gauges=("ledger", "yield", "endo_excess"))
    got = []
    for i, x in enumerate(ser2):
        inf = lp2.step(i, _reads(i, x))
        if inf["N"] is not None:
            got.append(inf["N"] * inf["per_gauge"]["yield"]["tol"])   # undo floor units
    mixed_floor = _sd(got) / 2.0
    ok_all &= _say("L-2 floor construction reduces to the donor's null-ABBA",
                   abs(mixed_floor - donor_floor) / max(donor_floor, 1e-12) < 0.02,
                   {"donor_sd(N)/2": donor_floor, "learned_sd(Nhat)/2": mixed_floor,
                    "rel_diff": abs(mixed_floor - donor_floor) / max(donor_floor, 1e-12)})

    # L-3  SCALE-FREENESS. Multiplying the mixture and its theta by any positive constant is
    #      the same policy. This is what stops a fit's shrinkage from making the arm inert.
    base = {"1": [0.3, 0.9, -0.2], "0": [0.5, -0.4, 0.7]}
    th = {"1": 1.4, "0": 0.8}
    for scale in (0.01, 7.0):
        a = _mk(base, th)
        c = _mk({k: [scale * x for x in v] for k, v in base.items()},
                {k: scale * x for k, x in th.items()})
        same = True
        for i, x in enumerate(ser2[:200]):
            rd = _reads(i, x + 0.3 * (i % 5), wc=(i // 37) % 2)
            same &= (a.step(i, rd)["quiet"] == c.step(i, rd)["quiet"])
        ok_all &= _say(f"L-3 scale-free in the mixture (x{scale})", same, {"identical": same})

    # L-4  THE BUCKET IS LEVEL STATE AND NOTHING ELSE. Two policies whose tables differ only
    #      in the cell that is never visited decide identically.
    a = _mk({"1": [0.0, 1.0, 0.0], "0": [1.0, 0.0, 0.0]}, {"1": 1.0, "0": 1.0})
    b2 = _mk({"1": [0.0, 1.0, 0.0], "0": [-9.0, 3.0, 5.0]}, {"1": 1.0, "0": 4.0})
    same = True
    for i, x in enumerate(ser2[:200]):
        rd = _reads(i, x, wc=1)
        same &= (a.step(i, rd)["quiet"] == b2.step(i, rd)["quiet"])
    ok_all &= _say("L-4 unvisited table cell is inert", same, {"identical": same})

    # L-5  RESET ON ACTION reaches every sub-estimator, so one quiet reading cannot fire two
    #      absorbing actions and the re-arm is A1's.
    a = _mk(base, th)
    for i, x in enumerate(ser2[:60]):
        a.step(i, _reads(i, x))
    a.acted(COMMIT, 99)
    ok_all &= _say("L-5 reset on action reaches every sub-estimator",
                   (a.V is None) and (not a.moved) and (a.n_since == 0) and (not a.quiet)
                   and all(e.V is None and e.n_since == 0 for e in a._est.values()),
                   {"V": a.V, "moved": a.moved, "n_since": a.n_since,
                    "sub_V": {g: e.V for g, e in a._est.items()}})

    # L-6  DETERMINISM AND RNG-NEUTRALITY, the donor's P-7 re-asked of the learned rule: two
    #      instances agree bit for bit and no global RNG draw is consumed.
    st = _r.getstate()
    a = _mk(base, th)
    b2 = _mk(base, th)
    va, vb = [], []
    for i, x in enumerate(ser2[:300]):
        rd = _reads(i, x, wc=(i // 23) % 2)
        va.append(a.step(i, rd)["V"])
        vb.append(b2.step(i, rd)["V"])
    ok_all &= _say("L-6 determinism + RNG neutrality (learned)",
                   va == vb and _r.getstate() == st,
                   {"identical": va == vb, "rng_untouched": _r.getstate() == st})

    # L-7  NO DEFAULTED FIT AND NO DEFAULTED FLOOR: an unfitted or non-positive-theta learned
    #      rule is an error, not a silent zero.
    raised = 0
    for bad in ({}, None):
        try:
            _mk(bad, th)
        except ValueError:
            raised += 1
    try:
        _mk(base, {"1": 0.0, "0": 1.0})
    except ValueError:
        raised += 1
    try:
        build_policy({"kind": "learned"}, floors=FL)
    except ValueError:
        raised += 1
    ok_all &= _say("L-7 no defaulted fit / floor", raised == 4, {"n_raised": raised})

    # L-8  A BUCKET CHANGE RESETS. A statistic armed under one level-state cell must not
    #      licence an action under the other -- "positive and then flat" has to mean flat in
    #      the SAME regime it was positive in.
    a = _mk({"1": [0.0, 1.0, 0.0], "0": [0.0, -1.0, 0.0]}, {"1": 1.0, "0": 1.0})
    fired_at_flip = False
    for i, x in enumerate(ser2[:80]):
        a.step(i, _reads(i, x, wc=1))
    armed = a.moved
    inf = a.step(80, _reads(80, ser2[80], wc=0))
    fired_at_flip = inf["quiet"]
    ok_all &= _say("L-8 bucket change resets the latch",
                   armed and (not fired_at_flip) and a.V is None and a.n_since == 1,
                   {"armed_before_flip": armed, "fired_on_flip": fired_at_flip,
                    "V_after_flip": a.V, "n_since": a.n_since,
                    "last_reset": a.last_reset})

    out["ALL"] = bool(ok_all)
    if verbose:
        print(f"\npolicy_gate: {'ALL PASS' if ok_all else 'FAILURES ABOVE'}")
    return out


if __name__ == "__main__":
    import json
    import sys
    r = policy_gate()
    print(json.dumps({k: v for k, v in r.items() if k != "ALL"}, indent=2)[:0] or "", end="")
    sys.exit(0 if r["ALL"] else 1)
