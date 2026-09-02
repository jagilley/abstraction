"""questions.py — THE QUESTION PORT'S SELECTORS, and the difficulty quota that pins them.

Out of the Modal app on purpose (`../conductor/policy.py`'s pattern): every rule here is
auditable, and gate-able, with no GPU. `antiphon.py` imports this module and calls exactly one
function per cycle.

WHAT A QUESTION IS. Each cycle the world offers a MENU of `K` candidate repair episodes drawn
from the era's own damage cell — the same distribution the donor's `context_instances` draws
from, at K times the count. The arm's selector picks the `n_pr` it will actually practise on.
The cell is the world's; the selection is the arm's. That is the whole port.

THE THREE THINGS THE MENU MUST NOT MOVE, and how each is held:

  * VOLUME       every selector returns exactly `n_pr` indices. Asserted here.
  * DIFFICULTY   `quota` is the d* histogram of the menu's FIRST `n_pr` candidates — i.e. of the
                 donor's own draw, which is the head of the menu and is drawn by the donor's own
                 RNG call. Every quota-pinned selector fills that histogram bin-for-bin, so
                 selection moves WHICH INSTANCE WITHIN A DIFFICULTY STRATUM and never the
                 difficulty. Asserted here. (`comp_free` is the one deliberate exception; it
                 passes `quota=None`, and its realised histogram is logged every cycle so the
                 deviation is measured rather than absorbed.)
  * BUDGET       nothing here touches the beam; `n_pr` in, `n_pr` out.

THE FEASIBILITY OF THE QUOTA IS FREE. The head is itself a subset of the menu, so the menu
contains at least `quota[t]` candidates of every stratum `t` by construction — the quota can
always be filled, whatever the scores.

BREADTH BEFORE DEPTH. A selector that simply took the `n_pr` highest-scoring candidates would
take 64 copies of one key. `_round_robin` instead takes one candidate per distinct key in
priority order, then a second, and so on — so a cycle's 64 questions span as many distinct
targets as the quota allows. This matters because the mining subsample keeps only
`mine_cap = 8` of the solved answers.

CONTAINMENT. The selectors take DISJOINT argument bundles. `select_bisect` is the only one that
receives `oracle=` (the clean derivation's exact key, the true table, the arm's frozen tables);
`select_endo` / `select_novel` / `select_comp` / `select_comp_free` receive `agent=` (the arm's
own reader parse of the UNDAMAGED part of the span, its own miner counts, its own value head).
`containment_gate` asserts an agent bundle carries no truth object, and `antiphon.py::preflight`
runs it.

KEY FORMAT. Every key and half-key in this module is a plain tuple of ints — the level-1 feature
tuple over a span, exactly `Miner`'s own key format (`macros.parse_features` output, sliced).
The delivery ledger's posed/landed keys are therefore the same primitive the provenance lane's
efference→table reconstruction uses, at a coarser grain (a span rather than an execution).
"""

import collections

import numpy as np

SUPPORT_DEFAULT = 3
SUBW_FLOOR = 1e-4            # [trap] a sub-half the arm has never committed
DELIV_ALPHA = 0.3            # EWMA rate of the delivery ledger
DELIV_INIT = 1.0             # optimistic init: an unseen half-key is worth trying once


# --------------------------------------------------------------------------- #
# geometry: which span is mined, and which part of it the damage does NOT touch
# --------------------------------------------------------------------------- #

def target_geometry(era_level, era_node, maxl, s):
    """The level the cycle's questions are aimed at, the span the miner reads, and which blocks
    of that span the era's damage cell overwrites.

    `run_arm` mines levels 2..min(maxl, era_level+1) at
    `node = (era_node * s**(era_level-1)) // s**(level-1)`; the TARGET is the deepest of those,
    which is the level being earned. Returns a dict, or `level=None` when there is no mined
    level (never happens on this ladder).
    """
    level = min(int(maxl), int(era_level) + 1)
    if level < 2:
        return {"level": None}
    span = s ** (level - 1)
    node = (era_node * s ** (era_level - 1)) // span
    blk0 = node * span                              # first block of the mined span
    cell_span = s ** (era_level - 1)                # blocks the damage cell covers
    cell0 = era_node * cell_span
    dmg = [b for b in range(blk0, blk0 + span) if cell0 <= b < cell0 + cell_span]
    clean = [b - blk0 for b in range(blk0, blk0 + span) if not (cell0 <= b < cell0 + cell_span)]
    return {"level": int(level), "span": int(span), "node": int(node), "blk0": int(blk0),
            "dmg_blocks": [b - blk0 for b in dmg], "clean_offsets": clean,
            "has_clean": bool(clean)}


def keys_at(feats, geom):
    """The (n, span) key block of a level-1 feature array at the mined span."""
    b0 = geom["blk0"]
    return feats[:, b0:b0 + geom["span"]]


def half_keys(feats, geom):
    """The part of the key the QUESTION fixes and the repair cannot touch — the mined span's
    blocks that the damage cell does not cover. Empty (all-zero-width) in the consumption eras
    whose cell swallows the whole span; callers fall back to index order there."""
    if not geom["has_clean"]:
        return np.zeros((feats.shape[0], 0), dtype=np.int64)
    off = np.asarray(geom["clean_offsets"], dtype=np.int64)
    return feats[:, geom["blk0"] + off]


# --------------------------------------------------------------------------- #
# the quota
# --------------------------------------------------------------------------- #

def quota_of(d, n_head):
    """The difficulty histogram every quota-pinned arm must reproduce: the d* mix of the menu's
    head, which IS the donor's own draw."""
    return dict(collections.Counter(int(x) for x in np.asarray(d)[:n_head]))


def check_quota(d, sel, quota):
    got = dict(collections.Counter(int(x) for x in np.asarray(d)[sel]))
    return got == {int(k): int(v) for k, v in quota.items()}, got


# --------------------------------------------------------------------------- #
# the selection kernel
# --------------------------------------------------------------------------- #

def _round_robin(prio, keys, d, quota, n_pr):
    """Pick `n_pr` indices maximising `prio`, subject to the d* histogram `quota` (or free if
    `quota is None`), taking BREADTH FIRST: one candidate per distinct key in priority order,
    then a second, and so on.

    Fully deterministic — ties break on candidate index — so an arm's selection is a function
    of its own state and the menu, and of no RNG at all.
    """
    n = len(prio)
    keys = [tuple(int(x) for x in k) if hasattr(k, "__iter__") else (int(k),)
            for k in keys]
    need = None if quota is None else {int(k): int(v) for k, v in quota.items()}
    order = sorted(range(n), key=lambda i: (-float(prio[i]), i))
    taken = collections.Counter()
    chosen, seen = [], set()
    rounds = 0
    while len(chosen) < n_pr and rounds <= n_pr:
        for i in order:
            if len(chosen) >= n_pr:
                break
            if i in seen or taken[keys[i]] != rounds:
                continue
            if need is not None:
                st = int(d[i])
                if need.get(st, 0) <= 0:
                    continue
                need[st] -= 1
            chosen.append(i)
            seen.add(i)
            taken[keys[i]] += 1
        rounds += 1
    # the quota is feasible by construction (the head is in the menu), so this cannot bind;
    # kept as a fail-loud rather than a silent short batch.
    assert len(chosen) == n_pr, (
        f"selector returned {len(chosen)} of {n_pr} — quota {quota} could not be filled from "
        f"this menu, which should be impossible while the head is part of it")
    return np.array(sorted(chosen), dtype=np.int64)


# --------------------------------------------------------------------------- #
# the delivery ledger — the aleatoric guard, in the SPEC's own currency
# --------------------------------------------------------------------------- #

class DeliveryLedger:
    """Delta-`at_support`-per-priced-sample, attributed to the half-key the question fixed.

    Posed half-key h, then: did the observation that came back actually move a key that was not
    already banked? A half-key whose questions keep coming back as parse-ambiguous junk — the
    channel Phase 0 measured at 0.58 of the L3 stream and 0.82 of the L4 stream — decays toward
    zero and stops being asked. That is the noisy-TV discriminator: noise-disagreement does not
    CLOSE when you collect there.
    """

    def __init__(self, alpha=DELIV_ALPHA, init=DELIV_INIT):
        self.alpha, self.init = float(alpha), float(init)
        self.w = {}
        self.posed = collections.Counter()
        self.landed = collections.Counter()

    def weight(self, h):
        return self.w.get(tuple(int(x) for x in h), self.init)

    def update(self, posed_halves, credits):
        """`posed_halves[i]` is the half-key of the i-th MINED instance; `credits[i]` is 1 when
        that instance's observation advanced a key that was below support."""
        for h, c in zip(posed_halves, credits):
            h = tuple(int(x) for x in h)
            self.posed[h] += 1
            self.landed[h] += int(c)
            self.w[h] = (1 - self.alpha) * self.w.get(h, self.init) + self.alpha * float(c)

    def state(self):
        return {"n_halves": len(self.w),
                "posed": int(sum(self.posed.values())), "landed": int(sum(self.landed.values())),
                "mean_w": float(np.mean(list(self.w.values()))) if self.w else None}


# --------------------------------------------------------------------------- #
# the five selectors
# --------------------------------------------------------------------------- #

def select_exo(n_pr, **_):
    """THE EXOGENOUS LADDER — the menu's head, which is the donor's own `context_instances`
    draw at the donor's own RNG position. In this parameterization the era ladder's demand IS a
    uniform draw over instances of the era's cell, so this arm is simultaneously the incumbent,
    the "random questions" arm, and the bit-identical replay of the donor."""
    return np.arange(int(n_pr), dtype=np.int64)


def select_bisect(n_pr, d, quota, oracle):
    """THE ORACLE CEILING. `oracle` carries the clean derivation's EXACT key at the mined span,
    the true table for that level, the arm's own miner counts, and the flat set of its own
    operative lower table. Withheld from every other arm by construction.

    Priority, in order: never target a key the grammar cannot actually produce (junk is
    worthless and is the trap `novel` walks into); never target a key already banked at support;
    among the rest prefer keys whose BOTH HALVES are already rows of the operative lower table,
    because only those survive `Miner.build`'s ratchet — this is the r-squared wall, made the
    selector's objective rather than its ceiling; then prefer the nearly-there over the
    untouched, which finishes keys instead of scattering.
    """
    keys = [tuple(int(x) for x in k) for k in oracle["keys"]]
    truth = oracle["truth_flat"]
    counts = oracle["counts"]
    lower = oracle.get("lower_flat")
    support = int(oracle.get("support", SUPPORT_DEFAULT))
    half = len(keys[0]) // 2 if keys and len(keys[0]) >= 2 else 0
    prio = np.zeros(len(keys), float)
    for i, k in enumerate(keys):
        if k not in truth:
            prio[i] = 0.0                      # junk: never the target
            continue
        c = int(counts.get(k, 0))
        if c >= support:
            prio[i] = 0.1                      # banked: worth nothing, still beats junk
            continue
        buildable = (lower is None or half == 0
                     or (k[:half] in lower and k[half:] in lower))
        prio[i] = 2.0 + (1.0 if buildable else 0.0) + c / max(support, 1)
    return _round_robin(prio, keys, d, quota, n_pr)


def select_endo(n_pr, d, quota, agent):
    """THE LEARNER'S OWN JUDGE. `agent` carries only the arm's own state: the half-keys its own
    reader parsed off the UNDAMAGED part of the mined span, the set of half-keys covered by its
    own at-support keys, and its own delivery ledger. No truth, no exact map, no clean
    derivation.

    Score = novelty of the half-key against its own banked vocabulary, times the ledger's
    running estimate of whether questions with that half-key actually deliver anything. The
    first factor is prospective Delta-`at_support`; the second is the aleatoric guard.
    """
    halves = [tuple(int(x) for x in h) for h in agent["halves"]]
    covered = agent["covered"]                 # half-key -> how many banked keys carry it
    ledger = agent.get("ledger")
    if not halves or len(halves[0]) == 0:      # consumption eras: the cell swallows the span
        return _round_robin(np.zeros(len(d)), [(i,) for i in range(len(d))],
                            d, quota, n_pr)
    prio = np.array([(1.0 / (1.0 + covered.get(h, 0)))
                     * (ledger.weight(h) if ledger is not None else 1.0)
                     for h in halves], float)
    return _round_robin(prio, halves, d, quota, n_pr)


def select_novel(n_pr, d, quota, agent):
    """THE NERDSNIPE CONTROL — `select_endo` with the delivery ledger removed. Phase 0 measured
    the aleatoric channel this walks into: the grammar's own parse ambiguity makes 104 junk keys
    available at L3 and 7573 at L4, and junk is MAXIMALLY novel, forever. A3', with no burst
    machinery needed."""
    return select_endo(n_pr, d, quota, {**agent, "ledger": None})


def select_comp(n_pr, d, quota, agent):
    """THE KNOWN NEGATIVE, difficulty-pinned. Competence-coupled demand: prefer the questions
    the arm's own value head expects to get right. `merge`/`fourwall` beta=2 manufactured 30-70x
    starved holes this way. Here the d* quota holds the difficulty mix, so this arm tests
    COMFORT IN CONTENT at matched difficulty — the weaker half of the comfort pole."""
    val = np.asarray(agent["value"], float)
    return _round_robin(val, [(i,) for i in range(len(val))], d, quota, n_pr)


def select_trust(n_pr, d, quota, agent):
    """[trap] THE GUARD AT THE RIGHT DEPTH — novelty times the arm's OWN USE RECORD.

    `phase0_trap.py` [9]/[9b]: at the level the clean half lives on, the arm's committed table is
    0.25-0.50 precise and hard membership in it is no guard at all (it prefers the arm's own
    committed junk rows). But a clean half is itself a pair of lower-level rows, and at THAT
    level the arm has a beam use record — `log['entry']['hist']['beam']`, the priced beam's own
    per-entry selection count, already recorded and already paid for in every arm. The beam's
    selection mass sits on TRUE rows at 0.73-0.93 against a table precision of 0.25-0.50 (lift
    1.8-3.2x; `census`'s junk-filter result, on this node's own logs).

    So: score a candidate by the product of its two sub-halves' use shares. `agent["subw"]` is
    that map (flat tuple -> share, from `trap_menu.trust_weights`); a sub-half the arm has never
    committed scores `SUBW_FLOOR`, which keeps the quota fillable and keeps the arm from simply
    refusing everything early, when its table is small.

    No truth, no exact map, no clean derivation — only the arm's own history of which of its own
    entries its own executor chose.
    """
    halves = [tuple(int(x) for x in h) for h in agent["halves"]]
    covered = agent["covered"]
    subw = agent.get("subw") or {}
    if not halves or len(halves[0]) < 2:
        return select_endo(n_pr, d, quota, {**agent, "ledger": None})
    half = len(halves[0]) // 2
    prio = np.array([(1.0 / (1.0 + covered.get(h, 0)))
                     * (subw.get(h[:half], SUBW_FLOOR) * subw.get(h[half:], SUBW_FLOOR))
                     for h in halves], float)
    return _round_robin(prio, halves, d, quota, n_pr)


def select_comp_free(n_pr, d, quota, agent):
    """THE KNOWN NEGATIVE, unpinned — the honest beta=2 pole. Same rule, no quota: the arm may
    move the difficulty mix as well as the content, which is the axis the pinned arm gives up.
    Volume, priced budget and lifetime stay pinned. Its realised d* histogram is logged every
    cycle: the deviation IS the arm's definition, so it is measured, never absorbed."""
    val = np.asarray(agent["value"], float)
    return _round_robin(val, [(i,) for i in range(len(val))], d, None, n_pr)


SELECTORS = {
    "exo": select_exo, "bisect": select_bisect, "endo": select_endo,
    "novel": select_novel, "comp": select_comp, "comp_free": select_comp_free,
    "trust": select_trust,                      # [trap]
}
NEEDS_ORACLE = ("bisect",)
QUOTA_FREE = ("comp_free",)


def select(mode, n_pr, d=None, quota=None, oracle=None, agent=None):
    """The single entry point `antiphon.py` calls. Routes on `mode` and enforces containment:
    a non-oracle mode never receives the oracle bundle."""
    if mode == "exo":
        return select_exo(n_pr)
    fn = SELECTORS[mode]
    if mode in NEEDS_ORACLE:
        assert oracle is not None, f"mode {mode} needs the oracle bundle"
        return fn(n_pr, d, quota, oracle)
    assert agent is not None, f"mode {mode} needs the agent bundle"
    ok, why = containment_gate(agent)
    assert ok, f"containment FAILED for mode {mode}: {why}"
    return fn(n_pr, d, (None if mode in QUOTA_FREE else quota), agent)


# --------------------------------------------------------------------------- #
# gates — all offline, all free
# --------------------------------------------------------------------------- #

_FORBIDDEN = ("truth", "truth_flat", "clean", "keys", "inverse", "inverse_bottom",
              "exact", "rules", "lower_flat", "oracle",
              # [trap] T-4: the world's own distractor tag never reaches an agent bundle.
              # `subw` is the arm's OWN use record over its OWN committed rows and is allowed;
              # anything naming the trap, the substitution mask or the bank is not.
              "trap", "distractor", "is_d", "bank")


def containment_gate(agent):
    """Q-1: an agent bundle carries nothing the exact oracle owns. Keyed on NAMES, which is
    the auditable form — the bundle is built one line away from here, in `antiphon.py`."""
    bad = [k for k in agent if any(f in str(k).lower() for f in _FORBIDDEN)]
    return (not bad), (f"agent bundle carries {bad}" if bad else "clean")


def question_gate(verbose=False):
    """The offline suite. Every check is pure arithmetic on synthetic menus."""
    out = {}
    rng = np.random.default_rng(0)
    K, n_pr = 512, 64
    d = rng.integers(0, 6, size=K)
    quota = quota_of(d, n_pr)

    # Q-1 containment
    ok1, _ = containment_gate({"halves": 1, "covered": 2, "value": 3, "ledger": 4})
    ok1b, _ = containment_gate({"halves": 1, "truth_flat": 2})
    out["Q-1_containment"] = bool(ok1 and not ok1b)

    # Q-2 the head fills its own quota, and `exo` IS the head in order
    sel = select("exo", n_pr)
    out["Q-2_exo_is_head"] = bool(list(sel) == list(range(n_pr))
                                  and check_quota(d, sel, quota)[0])

    # Q-3 every pinned selector reproduces the quota exactly
    keys = [tuple(int(x) for x in r) for r in rng.integers(0, 8, size=(K, 4))]
    truth = set(keys[:200])
    counts = {k: int(rng.integers(0, 5)) for k in keys[:100]}
    orc = {"keys": keys, "truth_flat": truth, "counts": counts,
           "lower_flat": {k[:2] for k in keys[:150]} | {k[2:] for k in keys[:150]},
           "support": 3}
    halves = [k[2:] for k in keys]
    covered = collections.Counter(h for h in halves[:80])
    ag = {"halves": halves, "covered": dict(covered), "value": rng.random(K),
          "ledger": DeliveryLedger()}
    pinned = {}
    for mode in ("bisect", "endo", "novel", "comp"):
        s = select(mode, n_pr, d, quota, oracle=orc, agent=ag)
        pinned[mode] = bool(len(s) == n_pr and len(set(s.tolist())) == n_pr
                            and check_quota(d, s, quota)[0])
    out["Q-3_quota_pinned"] = bool(all(pinned.values()))
    out["Q-3_detail"] = pinned

    # Q-4 comp_free returns n_pr, is quota-FREE, and demonstrably deviates
    sf = select("comp_free", n_pr, d, quota, agent=ag)
    out["Q-4_comp_free_unpinned"] = bool(len(sf) == n_pr and len(set(sf.tolist())) == n_pr
                                         and not check_quota(d, sf, quota)[0])

    # Q-5 determinism: same inputs, same output, twice
    a = select("endo", n_pr, d, quota, agent=ag)
    b = select("endo", n_pr, d, quota, agent=ag)
    out["Q-5_deterministic"] = bool(np.array_equal(a, b))

    # Q-6 breadth: the oracle spans more distinct target keys than a plain top-n_pr would
    prio = np.array([2.0 + (1.0 if keys[i] in truth else -2.0) for i in range(K)])
    top = np.array(sorted(sorted(range(K), key=lambda i: (-prio[i], i))[:n_pr]))
    sb = select("bisect", n_pr, d, quota, oracle=orc)
    out["Q-6_breadth"] = bool(len({keys[i] for i in sb}) >= len({keys[i] for i in top}))

    # Q-7 the oracle never targets junk while a needy true key is available in-stratum
    junk_picked = sum(1 for i in sb if keys[i] not in truth)
    out["Q-7_oracle_avoids_junk"] = bool(junk_picked < n_pr // 2)
    out["Q-7_detail"] = {"junk_selected": int(junk_picked), "of": n_pr}

    # Q-8 the ledger decays a half-key that never lands and spares one that does
    led = DeliveryLedger()
    h_bad, h_good = (1, 1), (2, 2)
    for _ in range(10):
        led.update([h_bad, h_good], [0, 1])
    out["Q-8_ledger_discriminates"] = bool(led.weight(h_bad) < 0.2 < led.weight(h_good))

    # Q-9 geometry: the clean part of the mined span is exactly the half the cell does not cover
    g = {}
    for lv, nd in ((1, 25), (2, 12), (3, 6), (4, 3), (5, 1)):
        gg = target_geometry(lv, nd, 4, 2)
        g[f"L{lv}n{nd}"] = {"level": gg["level"], "node": gg["node"],
                            "dmg": gg["dmg_blocks"], "clean": gg["clean_offsets"]}
    out["Q-9_geometry"] = bool(
        g["L1n25"]["clean"] == [0] and g["L1n25"]["dmg"] == [1]
        and g["L2n12"]["clean"] == [2, 3] and g["L2n12"]["dmg"] == [0, 1]
        and g["L3n6"]["clean"] == [4, 5, 6, 7] and g["L3n6"]["dmg"] == [0, 1, 2, 3]
        and g["L4n3"]["clean"] == [] and g["L5n1"]["clean"] == [])
    out["Q-9_detail"] = g

    # Q-10 empty-clean eras fall through to a quota-legal, deterministic selection
    ag0 = {"halves": [() for _ in range(K)], "covered": {}, "value": rng.random(K),
           "ledger": DeliveryLedger()}
    s0 = select("endo", n_pr, d, quota, agent=ag0)
    out["Q-10_empty_clean_ok"] = bool(len(s0) == n_pr and check_quota(d, s0, quota)[0])

    # [trap] Q-12/Q-13 need REAL sub-half structure, so they build their own 4-wide halves
    # (at era 3 a clean half is a level-3 key and its own halves are level-2 keys).
    h4 = [tuple(int(x) for x in r) for r in rng.integers(0, 8, size=(K, 4))]
    subs = sorted({h[:2] for h in h4} | {h[2:] for h in h4})
    ag4 = {"halves": h4, "covered": {}, "value": rng.random(K), "ledger": DeliveryLedger()}

    used = set(subs[:len(subs) // 4])                      # the rows the beam actually used
    w = {r: (1.0 if r in used else 0.0) for r in subs}
    tot = sum(w.values()) or 1.0
    subw = {k: v / tot for k, v in w.items()}

    st = select("trust", n_pr, d, quota, agent={**ag4, "subw": subw})
    st2 = select("trust", n_pr, d, quota, agent={**ag4, "subw": subw})
    out["Q-12_trust_quota"] = bool(len(st) == n_pr and len(set(st.tolist())) == n_pr
                                   and check_quota(d, st, quota)[0]
                                   and np.array_equal(st, st2))

    def _hits(sel):
        return sum(1 for i in sel if h4[i][:2] in used and h4[i][2:] in used)
    base = _hits(select("novel", n_pr, d, quota, agent=ag4))
    out["Q-13_trust_prefers_used"] = bool(_hits(st) > base)
    out["Q-13_detail"] = {"trust_hits": int(_hits(st)), "novel_hits": int(base), "of": n_pr,
                          "n_sub_rows": len(subs), "n_used": len(used)}
    sn = select("trust", n_pr, d, quota, agent={**ag4, "subw": {}})
    out["Q-13_no_record_ok"] = bool(len(sn) == n_pr and check_quota(d, sn, quota)[0])

    out["ALL"] = all(bool(v) for k, v in out.items() if not k.endswith("_detail"))
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    r = question_gate(verbose=True)
    print(f"\nquestion gate: {'ALL PASS' if r['ALL'] else 'FAIL'}")
