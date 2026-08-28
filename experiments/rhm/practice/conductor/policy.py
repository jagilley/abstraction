"""conductor/policy — the outer loop, kept OUT of the Modal app so every rule is auditable and
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
