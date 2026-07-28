"""Repair cost per drift event -- the homeostatic instrument.

The claim under test (`ideas/adaptive_core_and_hierarchy_climb.md` §6): if hierarchy
climbing is real, drift event #8 costs less to repair than drift event #1 AT MATCHED DRIFT
MAGNITUDE; if it is not, repair cost is flat forever. Magnitude matching is exact here --
see `rhm_drift.drift_kl` / `calibrate_sigma`.

WHY RAW CROSS-ENTROPY CANNOT BE THE READOUT
-------------------------------------------
The obvious instrument -- "how much budget until held-out CE returns to its pre-drift
value" -- is a null BY CONSTRUCTION under support-fixed drift, and the null looks like a
result. For a mixture drifting away from uniform, the per-node KL is exactly the entropy
it destroys:

    KL(w || uniform) = log m - H(w)

so summing over nodes, H(p_old) - H(p_new) = KL(p_new || p_old) exactly. Therefore a model
that perfectly learned the pre-drift DGP scores

    CE(p_new, q=p_old) = H(p_new) + KL(p_new || p_old) = H(p_old)

which is *precisely* its pre-drift CE. Verified numerically at 0.000e+00 difference. The
drift made the world easier by exactly as much as it made the model wrong, and raw CE
cannot see either half.

(Between two non-uniform states the cancellation is partial rather than exact, so the raw
signal is a small residual of a large cancellation -- still a bad instrument, and now
sign-ambiguous too.)

THE READOUT THAT WORKS
----------------------
Measure the gap to a matched optimum, which is E3's `gap_closed` / `regA_recovery_auc`
idiom (`mjc/on_policy/directed_on_policy/directed_on_policy.py:736,748` -- a `ceil_net`
trained on abundant current-world data, a stale `fm_base`, and progress scored as the
fraction of the stale->matched gap closed):

    progress(b) = (ce_stale - ce(b)) / (ce_stale - ce_matched)
    repair_cost = min{ b : progress(b) >= threshold }

`ce_stale` is the model frozen at the moment of drift, evaluated on POST-drift data;
`ce_matched` is a reference given an abundant budget on the post-drift state. Both are
evaluated on the same held-out post-drift set, so the entropy change cancels between them
and what survives is the model's mismatch.

Budget is metered in sequences consumed -- the RHM analog of E3's `Body` step charge, and
the reason "where should I spend" is a question at all.

READ IT AGAINST THE DEPTH PROBE, NOT ALONE
------------------------------------------
Falling repair cost is confoundable with ordinary continued training (§6). The
discriminator is two instruments of different type: climbing predicts repair cost falls AND
per-level ancestor recovery rises; ordinary continued training predicts cost falls and
depth stays put. Support-fixed drift makes the probe genuinely freezable -- the rule tables
never move, so a fixed held-out probe set keeps valid ancestor labels for the whole run
(asserted by `verify_backcompat.py` B5). Note that sign agreement between the two does not
APPORTION the effect; that remains open and may need a matched-data arm.
"""

import numpy as np

from rhm.rhm_drift import node_counts


def derivation_entropy(rules, s, weights, v=None):
    """Entropy (nats/sequence) of the derivation process under `weights`.

    H = H(root) + sum over nodes of E[H(rule choice at that node)].

    This is an UPPER bound on the achievable leaf-token CE: ambiguous rule tables make the
    derivation -> leaves map non-injective, so H(leaves) <= H(derivation). Use it for the
    analytic blindness argument (which is exact at the derivation level) and as a sanity
    scale, NOT as the matched-model floor -- for that, train a matched reference.
    """
    v = rules[0].shape[0] if v is None else v
    counts = node_counts(rules, s, weights)
    total = float(np.log(v))
    for ell in range(len(rules)):
        p = weights[ell]
        h = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(-1)
        total += float((counts[ell] * h).sum())
    return total


def raw_ce_blindness(rules, s, w_old, w_new):
    """The size of the trap: how much of the drift's KL is cancelled by the entropy drop.

    Returns a dict with the KL, the entropy change, and their sum -- which is what a
    perfectly-pre-drift-fit model's raw CE would move by. Near zero means raw CE is blind.
    """
    from rhm.rhm_drift import drift_kl
    kl, _per_level = drift_kl(rules, s, w_old, w_new)
    h_old = derivation_entropy(rules, s, w_old)
    h_new = derivation_entropy(rules, s, w_new)
    return {"kl_nats_per_seq": kl, "entropy_old": h_old, "entropy_new": h_new,
            "entropy_drop": h_old - h_new, "raw_ce_shift": (h_new + kl) - h_old,
            "blindness_ratio": abs((h_new + kl) - h_old) / max(kl, 1e-12)}


class Meter:
    """Charges sequences consumed. The RHM analog of `mjc/embodied.py`'s `Body._charge`.

    `monitor` and `collect` are tallied separately so the monitor:collect ratio can be
    reported -- E3's headline anti-subsidy number (1.84x, against S2's fatal 22x).
    """

    def __init__(self, budget=None):
        self.budget = budget
        self.collect = 0
        self.monitor = 0

    def charge_collect(self, n):
        self.collect += int(n)
        self._check()
        return n

    def charge_monitor(self, n):
        self.monitor += int(n)
        self._check()
        return n

    @property
    def total(self):
        return self.collect + self.monitor

    @property
    def ratio(self):
        return self.monitor / max(self.collect, 1)

    def _check(self):
        if self.budget is not None and self.total > self.budget:
            raise BudgetExhausted(f"spent {self.total} of {self.budget}")


class BudgetExhausted(RuntimeError):
    pass


def repair_progress(ce, ce_stale, ce_matched):
    """Fraction of the stale->matched gap closed. 0 = no repair, 1 = fully matched.

    Returns nan when the gap is degenerate (|stale - matched| too small to normalise by),
    which is the honest answer: with no gap there is no repair to measure. Callers should
    treat nan as "instrument out of range" rather than as 0 or 1 -- E3's own lesson that a
    readout degenerate at its ends must be reported, not clipped.
    """
    gap = ce_stale - ce_matched
    if not np.isfinite(gap) or abs(gap) < 1e-6:
        return float("nan")
    return float((ce_stale - ce) / gap)


def attributable_progress(ce, ce_stale, gap_total, ce_ctrl, ce_ctrl_stale, gap_nodrift,
                          min_gap=1e-6):
    """Repair progress with the ordinary-continued-training component differenced out.

    A matched reference given extra budget improves for TWO reasons: it adapts to the drift
    (what we want to measure) and it is simply further along in training (what we do not).
    Measured on a smoke at L=5: with ZERO drift the stale->matched gap was still +0.0143
    nats, against +0.0354 at KL=0.68 -- so ~40% of the apparent "repair" was compute, not
    adaptation. That is §6's confound reappearing inside the instrument itself, and it
    would inflate every event's repair number.

    The fix is a difference-in-differences against a paired arm that receives the SAME
    budget on UNDRIFTED data:

        attributable(b) = [(ce_stale - ce(b)) - (ce_ctrl_stale - ce_ctrl(b))] / gap_attributable
        gap_attributable = gap_total - gap_nodrift

    Each half is an improvement relative to its own stale baseline on its own eval set, so
    the differing entropies of the two distributions cancel. This is the matched-data arm
    `ideas/adaptive_core_and_hierarchy_climb.md` flags as possibly needed to APPORTION a
    falling cost curve; building it in from the start makes the apportionment available
    rather than retrofitted.

    `min_gap` is the smallest attributable gap worth dividing by. It must be set from the
    NULL arm's measured gap, not left at its 1e-6 default: on the smoke the zero-drift
    attributable gap was -0.0044 -- statistically zero but a thousandfold above 1e-6 -- and
    dividing by it produced a confident-looking "progress 1.052" on a drift that never
    happened. Same self-calibration trick as the distractor verifier's LP floor: let the
    null arm set the resolution.
    """
    gap = gap_total - gap_nodrift
    if not np.isfinite(gap) or abs(gap) < max(min_gap, 1e-12):
        return float("nan")
    return float(((ce_stale - ce) - (ce_ctrl_stale - ce_ctrl)) / gap)


def cost_from_progress(progress_curve, threshold=0.9):
    """Budget at which a PROGRESS curve first reaches `threshold`. (budget, progress) pairs.

    Prefer this over `repair_cost` when the primary readout is `attributable_progress` --
    `repair_cost` recomputes raw progress internally, so mixing them reports a cost for a
    curve the verdicts never used (it produced a confident 187k-sequence cost on the
    zero-drift arm whose attributable progress was correctly nan).

    NOTE ON WHAT THIS MEASURES. Cost-to-a-FRACTION-of-the-gap is self-normalising, so it
    does NOT increase with drift magnitude and should not be expected to: measured at L=5,
    KL 0.335/0.677/1.372 gave 48449/12225/12135 sequences. A small drift sits near the
    noise floor, so closing 90% of its (small) gap takes more samples, not fewer. The
    quantity that tracks magnitude is the GAP itself (0.0096/0.0203/0.0421, ~linear in KL).
    For "bigger drift costs more work", integrate to a fixed number of nats instead.

    This is fine for the homeostatic claim, which compares events at MATCHED magnitude --
    but it means a cross-magnitude cost comparison is not the sensitivity check it looks
    like.
    """
    prev_b, prev_p = None, None
    for b, p in progress_curve:
        if not np.isfinite(p):
            continue
        if p >= threshold:
            if prev_p is None or p == prev_p:
                return float(b)
            return float(prev_b + (threshold - prev_p) / (p - prev_p) * (b - prev_b))
        prev_b, prev_p = b, p
    return None


def repair_cost(curve, ce_stale, ce_matched, threshold=0.9):
    """Budget at which the repair curve first reaches `threshold` of the gap.

    `curve` -- iterable of (budget, ce), ascending in budget. Returns None if the threshold
    is never reached within the curve (report it; do not silently treat as the max budget).
    Linearly interpolates between the bracketing checkpoints.
    """
    prev_b, prev_p = None, None
    for b, ce in curve:
        p = repair_progress(ce, ce_stale, ce_matched)
        if not np.isfinite(p):
            continue
        if p >= threshold:
            if prev_p is None or p == prev_p:
                return float(b)
            frac = (threshold - prev_p) / (p - prev_p)
            return float(prev_b + frac * (b - prev_b))
        prev_b, prev_p = b, p
    return None
