"""Chronicle -- controlled English with an exact belief oracle.

The RHM move, applied to actual English. A short story is generated from a
latent scene `z = (person, object, place, hour)`, each slot uniform over 4
single-token values, so `|Z| = 256`. The story is a sequence of clauses; the
*clause plan* is sampled independently of `z` and the *emissions* are the only
z-dependent part. That factorisation is what makes the oracle exact and cheap:
the posterior `P(z | x_<=t)` stays a product of per-slot uniforms over a
shrinking live set, so belief revision

    B_t = KL( P(z | x_<=t+1) || P(z | x_<=t) )

is a closed form, not an inference problem.

Why bother building a substrate instead of using real text: the whole cut turns
on separating *reducible* from *irreducible* surprise, and on real text neither
term is knowable. Here both are, exactly -- and unlike RHM we can also break the
confound the RHM tracking audit surfaced (there, prefix uncertainty *is* nearly
the family label; see `../../rhm/conditional_revision/README.md` appendix section 2).

Analysis positions come in families. Oracle surprisal is exactly `log 4` for all
of them except where noted:

    synonym        stylistic choice among 4 equiprobable words.  B = 0.
    offtopic       a contentful 4-way fact about a variable OUTSIDE z.  B = 0.
    mention        a slot's value word, emitted uniformly and carrying no
                   information about z at all.  B = 0.
    reveal         states z_k while the slot is fully live (L = 4).  B = log 4.
    reveal_narrow  states z_k with L < 4 live.  B = log L = surprisal.
    deduced        states z_k after the live set was already down to 1.
                   B = 0 and oracle surprisal 0 -- but the model has to have
                   done the deduction, so its own revision need not be 0.
    negate_live    "... was not <v>" for a v still live.  B = log(L/(L-1)).
    negate_dead    "... was not <v>" for a v already eliminated.  B = 0.
    echo           restates an already-revealed z_k.  B = 0, surprisal 0.
    narrow         "narrowed it to either <a> or <b>" -- NOT an analysis
                   position (see below), but the load-bearing clause.

Two construction details do all the work.

**1. The negation clause emits uniformly over all four values whatever the
posterior state.** It names a *dead* value with probability `p = d/4` and
otherwise a live non-true one; `_negation_emission` shows that this makes the
marginal emission exactly `1/4` per value for every `(L, d)`. So `negate_live`
vs `negate_dead` is a contrast inside one identical template, at one identical
clause position, with oracle surprisal pinned to `log 4` on both sides and only
`B` differing. RHM has no analogue of it.

**2. The narrow clause eliminates values without naming them.** A first version
of this substrate had only negation as an elimination mechanism, which made
"dead" perfectly equivalent to "named earlier in this story". A frozen
Llama-3.2-1B then separated `negate_live` from `negate_dead` at **AUC 0.96 on
token surprisal alone** and at **1.00** from a linear readout of its own state:
both were reading an induction head, not a belief, and the contrast was
degenerate before it started. `narrow` names the two *survivors*, so the values
it eliminates are never mentioned -- which fills in the two missing cells of
(dead/live) x (mentioned/not) and makes `n_prior_mentions` a matchable
nuisance variable instead of a synonym for the label. It is also what creates
the doc's actual claim in the data: a *never-mentioned dead* value is a
high-surprisal token that revises nothing, and a *previously-mentioned live*
value is a quiet token that revises a lot.

`narrow` positions are excluded from analysis (`analysis=False`). Between its
two value tokens the true posterior is momentarily non-uniform -- reading
"either the ledger" puts 1/2 on the ledger -- which would break the
uniform-on-a-live-set invariant. After the clause completes the posterior is
exactly uniform on the named pair again (verified by the brute-force test), so
every analysis position remains exact. Do not add analysis positions inside a
narrow clause without generalising the posterior representation.

Everything here is pure Python; `python3 -m a2a_forward.conditional_revision.corpus` runs the
brute-force self-tests (no Modal, no GPU, no tokenizer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

LN = math.log

# --------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------
# Pools are larger than needed; `build_lexicon` keeps the first 4 entries of
# each that are a single token under the reader's tokenizer, so the analysis
# position is always exactly one token and offset alignment is unambiguous.

SLOT_POOLS: dict[str, list[str]] = {
    "person": [" Anna", " Peter", " Sarah", " David", " Julia", " Martin",
               " Elena", " Thomas", " Clara", " Simon"],
    "object": [" ledger", " compass", " locket", " telegram", " diary",
               " violin", " camera", " passport", " bracelet", " painting"],
    "place":  [" attic", " cellar", " garden", " kitchen", " library",
               " chapel", " harbor", " tower", " stable", " gallery"],
    "hour":   [" dawn", " noon", " dusk", " midnight", " sunrise", " sunset",
               " morning", " nightfall", " daybreak", " twilight"],
}

DECOY_POOLS: dict[str, list[str]] = {
    "weather": [" rain", " fog", " snow", " wind", " frost", " drizzle"],
    "animal":  [" dog", " cat", " horse", " bird", " goat", " fox"],
}

SYNSETS: dict[str, list[str]] = {
    "notice":  [" noticed", " observed", " spotted", " remarked", " reported"],
    "quiet":   [" quiet", " silent", " still", " calm", " hushed"],
    "strange": [" strange", " odd", " peculiar", " curious", " unusual"],
    "quick":   [" quickly", " swiftly", " rapidly", " briskly", " sharply"],
    "small":   [" small", " little", " tiny", " modest", " slight"],
    "old":     [" old", " ancient", " aged", " worn", " faded"],
    "walk":    [" walked", " strolled", " wandered", " drifted", " stepped"],
    "began":   [" began", " started", " proceeded", " continued", " resumed"],
    "begin":   [" begin", " start", " proceed", " resume", " continue"],
}

SLOTS = list(SLOT_POOLS)
N_VAL = 4                      # values per slot
N_SYN = 4                      # words per synonym set (must equal N_VAL so the
                               # primary contrast is exactly surprisal-matched)

# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------
# Segments:  ("lit", text) | ("syn", setname) | ("val", slotname)
#            | ("dec", decoyname)
# Two carriers per slot per polarity, so a redundant negation is never a
# verbatim repeat of the clause that made the value dead.

CARRIERS: dict[str, dict[str, list]] = {
    "person": {
        "pos": [
            [("lit", " The inspector"), ("syn", "notice"),
             ("lit", " that the visitor that night was"), ("val", "person"),
             ("lit", ".")],
            [("lit", " According to the porter, the person at the door had been"),
             ("val", "person"), ("lit", ".")],
        ],
        "neg": [
            [("lit", " The inspector"), ("syn", "notice"),
             ("lit", " that the visitor that night was not"), ("val", "person"),
             ("lit", ".")],
            [("lit", " According to the porter, the person at the door had not been"),
             ("val", "person"), ("lit", ".")],
        ],
        "mention": [
            [("lit", " The file held an"), ("syn", "old"),
             ("lit", " statement from"), ("val", "person"),
             ("lit", ", dated years earlier.")],
        ],
        "narrow": [
            [("lit", " The porter narrowed the visitor down to either"),
             ("val", "person"), ("lit", " or"), ("val", "person"), ("lit", ".")],
        ],
    },
    "object": {
        "pos": [
            [("lit", " A"), ("syn", "small"),
             ("lit", " note said the missing item was the"), ("val", "object"),
             ("lit", ".")],
            [("lit", " The catalogue confirmed that what had gone missing was the"),
             ("val", "object"), ("lit", ".")],
        ],
        "neg": [
            [("lit", " A"), ("syn", "small"),
             ("lit", " note said the missing item was not the"), ("val", "object"),
             ("lit", ".")],
            [("lit", " The catalogue confirmed that what had gone missing was not the"),
             ("val", "object"), ("lit", ".")],
        ],
        "mention": [
            [("lit", " Someone had once written the word"), ("val", "object"),
             ("lit", " in the margin of the file.")],
        ],
        "narrow": [
            [("lit", " The catalogue narrowed the missing item to either the"),
             ("val", "object"), ("lit", " or the"), ("val", "object"),
             ("lit", ".")],
        ],
    },
    "place": {
        "pos": [
            [("lit", " Someone"), ("syn", "walk"),
             ("lit", " past and said the search would"), ("syn", "begin"),
             ("lit", " in the"), ("val", "place"), ("lit", ".")],
            [("lit", " The sergeant's order was that the search"), ("syn", "begin"),
             ("lit", " in the"), ("val", "place"), ("lit", ".")],
        ],
        "neg": [
            [("lit", " Someone"), ("syn", "walk"),
             ("lit", " past and said the search would not"), ("syn", "begin"),
             ("lit", " in the"), ("val", "place"), ("lit", ".")],
            [("lit", " The sergeant's order was that the search not"),
             ("syn", "begin"), ("lit", " in the"), ("val", "place"),
             ("lit", ".")],
        ],
        "mention": [
            [("lit", " An"), ("syn", "old"), ("lit", " floor plan of the"),
             ("val", "place"), ("lit", " lay unrolled on the desk.")],
        ],
        "narrow": [
            [("lit", " The sergeant narrowed the search to either the"),
             ("val", "place"), ("lit", " or the"), ("val", "place"),
             ("lit", ".")],
        ],
    },
    "hour": {
        "pos": [
            [("lit", " The clerk"), ("syn", "notice"),
             ("lit", " that the door was locked at"), ("val", "hour"),
             ("lit", ".")],
            [("lit", " The logbook recorded the door as locked at"),
             ("val", "hour"), ("lit", ".")],
        ],
        "neg": [
            [("lit", " The clerk"), ("syn", "notice"),
             ("lit", " that the door was not locked at"), ("val", "hour"),
             ("lit", ".")],
            [("lit", " The logbook recorded the door as not locked at"),
             ("val", "hour"), ("lit", ".")],
        ],
        "mention": [
            [("lit", " The kitchen clock had stopped at"), ("val", "hour"),
             ("lit", " some weeks before.")],
        ],
        "narrow": [
            [("lit", " The logbook narrowed the locking to either"),
             ("val", "hour"), ("lit", " or"), ("val", "hour"), ("lit", ".")],
        ],
    },
}

FILLERS: list[list] = [
    [("lit", " The hallway was"), ("syn", "quiet"), ("lit", " and"),
     ("syn", "old"), ("lit", ".")],
    [("lit", " Outside, the light moved"), ("syn", "quick"),
     ("lit", " across the floor.")],
    [("lit", " It was a"), ("syn", "strange"), ("lit", " evening, and the house"),
     ("syn", "began"), ("lit", " to settle.")],
    [("lit", " A"), ("syn", "small"), ("lit", " lamp burned in the corner, and the"),
     ("syn", "old"), ("lit", " clock ticked on.")],
]

OFFTOPICS: dict[str, list] = {
    "weather": [("lit", " All evening the"), ("dec", "weather"),
                ("lit", " had not let up.")],
    "animal":  [("lit", " A"), ("dec", "animal"),
                ("lit", " had been asleep beside the stairs.")],
}

OPENING = ("Late that night the house on Wexley Street stood empty, and the "
           "case notes were still open on the desk.")

CLAUSE_WEIGHTS = {"reveal": 0.15, "negate": 0.25, "narrow": 0.13,
                  "mention": 0.14, "echo": 0.06, "filler": 0.16,
                  "offtopic": 0.11}

# --------------------------------------------------------------------------
# lexicon construction
# --------------------------------------------------------------------------


@dataclass
class Lexicon:
    slot_values: dict[str, list[str]]
    decoy_values: dict[str, list[str]]
    synsets: dict[str, list[str]]

    def check(self) -> None:
        for d in (self.slot_values, self.decoy_values, self.synsets):
            for k, v in d.items():
                assert len(v) == N_VAL, f"{k}: {len(v)} != {N_VAL}"
                assert len(set(v)) == N_VAL, f"{k}: duplicates"


def build_lexicon(is_single_token=None) -> Lexicon:
    """Keep the first `N_VAL` entries of each pool that tokenize to one token.

    `is_single_token` is `lambda word -> bool`. Passing None skips the check and
    takes the first four -- used by the tokenizer-free self-test.
    """
    def pick(pool, n):
        if is_single_token is None:
            return pool[:n]
        out = [w for w in pool if is_single_token(w)]
        if len(out) < n:
            raise ValueError(f"only {len(out)} single-token words in {pool}")
        return out[:n]

    lex = Lexicon(
        slot_values={k: pick(v, N_VAL) for k, v in SLOT_POOLS.items()},
        decoy_values={k: pick(v, N_VAL) for k, v in DECOY_POOLS.items()},
        synsets={k: pick(v, N_SYN) for k, v in SYNSETS.items()},
    )
    lex.check()
    return lex


# --------------------------------------------------------------------------
# the exact oracle
# --------------------------------------------------------------------------


def _negation_emission(live: frozenset, true_v: int) -> dict[int, float]:
    """P(emitted value | z_k = true_v) for a negation clause.

    The generator names a **dead** value with probability `p = d/4` (uniform
    over the dead ones) and otherwise a live non-true value (uniform). With
    `d = 4 - L` that makes the *marginal* emission exactly uniform over all
    four values whatever the state:

        v dead : p/d                        = 1/4
        v live : (1-p) * P(z != v) / (L-1)
               = (1 - d/4) * ((L-1)/L) / (L-1) = 1/4

    so `negate_live` and `negate_dead` sit at oracle surprisal exactly `log 4`
    and differ only in `B`. This is the whole point of the negation clause; do
    not "simplify" the value choice to uniform-over-live.
    """
    L = len(live)
    dead = [v for v in range(N_VAL) if v not in live]
    d = len(dead)
    assert L >= 2, "negation requires at least two live values"
    p = d / N_VAL
    out = {v: 0.0 for v in range(N_VAL)}
    for v in dead:
        out[v] = p / d
    for v in live:
        if v != true_v:
            out[v] += (1.0 - p) / (L - 1)
    return out


def _kl_uniform(live_after: frozenset, live_before: frozenset) -> float:
    """KL( Unif(live_after) || Unif(live_before) ), nested supports."""
    assert live_after <= live_before and live_after
    return LN(len(live_before) / len(live_after))


def _entropy(probs) -> float:
    return -sum(p * LN(p) for p in probs if p > 0)


@dataclass
class Position:
    """One emitted lexical choice, with everything the gates need."""
    word: str
    char_start: int          # offset of the first non-space char in `text`
    family: str
    slot: Optional[str]
    value_idx: Optional[int]
    analysis: bool           # False for the two tokens inside a narrow clause
    constraint: Optional[frozenset]   # what this token tells you about z_slot
    B: float                 # oracle belief revision, joint over z (nats)
    gen_surprisal: float     # oracle -log P(x_t | x_<t)
    H_post_before: float     # oracle H(z | x_<t), nats
    H_post_after: float
    n_live_before: int
    n_live_after: int
    n_prior_mentions: int    # times this exact word already occurred, this story
    kill_mech: Optional[str] # for negate_dead: was the value killed by an
                             # earlier negation (which NAMED it) or by a narrow
                             # clause (which did not)? The single most
                             # diagnostic split for "is this just induction?"
    clause_idx: int
    clause_kind: str
    slot_in_clause: int
    frame: str               # (clause template, slot-in-clause)
    live_before: tuple       # per-slot posterior support, SLOTS order
    live_after: tuple
    E_B: float               # E_x[ B ] under the oracle emission distribution
    H_emit: float            # H(x_t | x_<t)
    H_emit_given_z: float    # E_z[ H(x_t | z, x_<t) ]


@dataclass
class Story:
    text: str
    z: dict[str, int]
    decoy: dict[str, int]
    positions: list[Position]
    n_clauses: int


def generate_story(rng, lex: Lexicon, n_clauses: int = 14) -> Story:
    """Sample one story together with its exact oracle annotations."""
    z = {k: rng.randrange(N_VAL) for k in SLOTS}
    decoy = {k: rng.randrange(N_VAL) for k in DECOY_POOLS}

    live = {k: frozenset(range(N_VAL)) for k in SLOTS}
    revealed = {k: False for k in SLOTS}
    used_decoy: set[str] = set()
    seen: dict[str, int] = {}          # word -> occurrences so far
    killed_by: dict[tuple, str] = {}   # (slot, value) -> "negate" | "narrow"
    last_filler = -1

    text = OPENING
    positions: list[Position] = []

    def H_post() -> float:
        return sum(LN(len(live[k])) for k in SLOTS)

    def emit(word, fam, sl, vi, analysis, constraint, B, emis_marg,
             H_emit_given_z, E_B, lb, la, ci, kind, sic, tpl_id):
        nonlocal text
        Hb = H_post()
        lv_before = tuple(tuple(sorted(live[k])) for k in SLOTS)
        char_start = len(text) + (1 if word.startswith(" ") else 0)
        text += word
        if sl is not None and la is not None:
            live[sl] = la
        Ha = H_post()
        positions.append(Position(
            word=word, char_start=char_start, family=fam, slot=sl,
            value_idx=vi, analysis=analysis, constraint=constraint, B=B,
            gen_surprisal=(-LN(emis_marg[vi]) if emis_marg.get(vi, 0) > 0
                           else float("inf")),
            H_post_before=Hb, H_post_after=Ha,
            n_live_before=len(lb) if lb is not None else N_VAL,
            n_live_after=len(la) if la is not None else N_VAL,
            n_prior_mentions=seen.get(word, 0),
            kill_mech=killed_by.get((sl, vi)) if sl is not None else None,
            clause_idx=ci, clause_kind=kind, slot_in_clause=sic,
            frame=f"{tpl_id}#{sic}",
            live_before=lv_before,
            live_after=tuple(tuple(sorted(live[k])) for k in SLOTS),
            E_B=E_B, H_emit=_entropy(emis_marg.values()),
            H_emit_given_z=H_emit_given_z))
        seen[word] = seen.get(word, 0) + 1

    def choice(seq):
        return seq[rng.randrange(len(seq))]

    for ci in range(n_clauses):
        # ---- clause plan: eligibility is prefix-determined only, so the plan
        # is independent of z given the prefix. That is what keeps the oracle
        # exact and what makes the identity self-test pass.
        rev_ok = [k for k in SLOTS if not revealed[k]]
        neg_ok = [k for k in SLOTS if not revealed[k] and len(live[k]) >= 2]
        nar_ok = [k for k in SLOTS if not revealed[k] and len(live[k]) >= 3]
        echo_ok = [k for k in SLOTS if revealed[k]]
        dec_ok = [k for k in DECOY_POOLS if k not in used_decoy]
        elig = {"filler": CLAUSE_WEIGHTS["filler"],
                "mention": CLAUSE_WEIGHTS["mention"]}
        if rev_ok:
            elig["reveal"] = CLAUSE_WEIGHTS["reveal"]
        if neg_ok:
            elig["negate"] = CLAUSE_WEIGHTS["negate"]
        if nar_ok:
            elig["narrow"] = CLAUSE_WEIGHTS["narrow"]
        if echo_ok:
            elig["echo"] = CLAUSE_WEIGHTS["echo"]
        if dec_ok:
            elig["offtopic"] = CLAUSE_WEIGHTS["offtopic"]
        r, kind = rng.random() * sum(elig.values()), "filler"
        for k_, w_ in elig.items():
            r -= w_
            if r <= 0:
                kind = k_
                break

        # ------------------------------------------------ pick the template
        if kind == "filler":
            fi = rng.randrange(len(FILLERS))
            if fi == last_filler:
                fi = (fi + 1 + rng.randrange(len(FILLERS) - 1)) % len(FILLERS)
            last_filler = fi
            segs, slot, tpl_id = FILLERS[fi], None, f"filler{fi}"
        elif kind == "offtopic":
            dk = choice(dec_ok)
            used_decoy.add(dk)
            segs, slot, tpl_id = OFFTOPICS[dk], dk, f"offtopic_{dk}"
        else:
            pool = {"reveal": rev_ok, "negate": neg_ok, "narrow": nar_ok,
                    "echo": echo_ok, "mention": SLOTS}[kind]
            slot = choice(pool)
            sub = {"reveal": "pos", "echo": "pos", "negate": "neg",
                   "narrow": "narrow", "mention": "mention"}[kind]
            ti = rng.randrange(len(CARRIERS[slot][sub]))
            segs = CARRIERS[slot][sub][ti]
            tpl_id = f"carrier_{slot}_{sub}{ti}"

        # ------------------------------------------------ render the clause
        sic = 0
        narrow_pair: list[int] = []
        for seg in segs:
            if seg[0] == "lit":
                text += seg[1]
                continue

            if seg[0] == "syn":
                words = lex.synsets[seg[1]]
                vi = rng.randrange(N_SYN)
                emit(words[vi], "synonym", None, vi, True, None, 0.0,
                     {i: 1.0 / N_SYN for i in range(N_SYN)}, LN(N_SYN), 0.0,
                     None, None, ci, kind, sic, tpl_id)
                sic += 1
                continue

            if seg[0] == "dec":
                words = lex.decoy_values[seg[1]]
                vi = decoy[seg[1]]
                emit(words[vi], "offtopic", None, vi, True, None, 0.0,
                     {i: 1.0 / N_VAL for i in range(N_VAL)}, LN(N_VAL), 0.0,
                     None, None, ci, kind, sic, tpl_id)
                sic += 1
                continue

            # ("val", slotname)
            sl = seg[1]
            words = lex.slot_values[sl]
            lb = live[sl]

            if kind == "mention":
                # names a value, says nothing about z: B = 0 by construction,
                # and it is what gives *live* values a mention history.
                vi = rng.randrange(N_VAL)
                emit(words[vi], "mention", sl, vi, True, None, 0.0,
                     {i: 1.0 / N_VAL for i in range(N_VAL)}, LN(N_VAL), 0.0,
                     lb, lb, ci, kind, sic, tpl_id)

            elif kind == "narrow":
                if not narrow_pair:                    # first disjunct
                    d_ = choice([v for v in sorted(lb) if v != z[sl]])
                    pair = [z[sl], d_]
                    if rng.random() < 0.5:
                        pair = pair[::-1]
                    narrow_pair = pair
                    emit(words[pair[0]], "narrow", sl, pair[0], False, None,
                         0.0, {v: 1.0 / len(lb) for v in lb}, 0.0, 0.0,
                         lb, lb, ci, kind, sic, tpl_id)
                else:                                  # second disjunct
                    la = frozenset(narrow_pair)
                    vi = narrow_pair[1]
                    for v_ in sorted(lb):
                        if v_ not in la:
                            killed_by[(sl, v_)] = "narrow"
                    emit(words[vi], "narrow", sl, vi, False, la,
                         _kl_uniform(la, lb),
                         {v: 1.0 / max(len(lb) - 1, 1)
                          for v in lb if v != narrow_pair[0]}, 0.0, 0.0,
                         lb, la, ci, kind, sic, tpl_id)

            elif kind in ("reveal", "echo"):
                vi = z[sl]
                emis = {v: 1.0 / len(lb) for v in lb}
                la = frozenset({vi})
                fam = ("echo" if kind == "echo" else
                       "deduced" if len(lb) == 1 else
                       "reveal" if len(lb) == N_VAL else "reveal_narrow")
                emit(words[vi], fam, sl, vi, True, la, _kl_uniform(la, lb),
                     emis, 0.0,
                     sum(emis[v] * _kl_uniform(frozenset({v}), lb) for v in lb),
                     lb, la, ci, kind, sic, tpl_id)
                revealed[sl] = True

            else:                                      # negate
                emis_cond = _negation_emission(lb, z[sl])
                emis_marg = {v: 0.0 for v in range(N_VAL)}
                per_z = []
                for zz in lb:
                    e_z = _negation_emission(lb, zz)
                    per_z.append(_entropy(e_z.values()))
                    for v_, q_ in e_z.items():
                        emis_marg[v_] += q_ / len(lb)
                rr, acc, vi = rng.random(), 0.0, None
                for v_ in range(N_VAL):
                    acc += emis_cond[v_]
                    if rr <= acc:
                        vi = v_
                        break
                vi = vi if vi is not None else max(emis_cond, key=emis_cond.get)
                la = lb - {vi} if vi in lb else lb
                was_live = vi in lb
                emit(words[vi], "negate_live" if was_live else "negate_dead",
                     sl, vi, True, frozenset(v for v in range(N_VAL) if v != vi),
                     _kl_uniform(la, lb), emis_marg,
                     sum(per_z) / len(per_z),
                     sum(emis_marg[v] * _kl_uniform(lb - {v} if v in lb else lb, lb)
                         for v in range(N_VAL)),
                     lb, la, ci, kind, sic, tpl_id)
                if was_live:
                    killed_by[(sl, vi)] = "negate"
            sic += 1

    return Story(text=text, z=z, decoy=decoy, positions=positions,
                 n_clauses=n_clauses)


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------


def _brute_force_support(story: Story, upto: int):
    """Enumerate all 4^K scenes and keep those satisfying every constraint the
    first `upto` positions impose. Re-derives the posterior support from the
    text semantics rather than from the incremental bookkeeping, so it catches
    bookkeeping bugs. RHM's `oracle.py` hypertree bug is the precedent: a
    plausible factorisation was simply wrong and only brute force found it."""
    import itertools
    cands = []
    for combo in itertools.product(range(N_VAL), repeat=len(SLOTS)):
        zz = dict(zip(SLOTS, combo))
        if all(zz[p.slot] in p.constraint
               for p in story.positions[:upto]
               if p.slot is not None and p.constraint is not None):
            cands.append(zz)
    return {k: frozenset(c[k] for c in cands) for k in SLOTS}, len(cands)


def self_test(n_stories: int = 150, seed: int = 0, verbose: bool = True) -> dict:
    """Three exact checks (no Monte Carlo):

    1. the incremental factorised posterior support == brute-force enumeration;
    2. `H_post` == log |support| at every analysis position, i.e. the posterior
       really is uniform there;
    3. the per-position identity
           E_x[ B ] = H(x_t | x_<t) - E_z[ H(x_t | z, x_<t) ]
       which is the language instance of the idea doc's section 4 identity.

    Also reports the (family x prior-mention) support table -- the cell counts
    Gate B1's matching depends on.
    """
    import random
    lex = build_lexicon(None)
    rng = random.Random(seed)
    max_post = max_ident = max_hpost = 0.0
    fam_counts: dict[str, int] = {}
    surp_by_fam: dict[str, list] = {}
    mention_tab: dict[tuple, int] = {}
    for _ in range(n_stories):
        st = generate_story(rng, lex)
        for p in st.positions:
            if not p.analysis:
                continue
            fam_counts[p.family] = fam_counts.get(p.family, 0) + 1
            surp_by_fam.setdefault(p.family, []).append(p.gen_surprisal)
            max_ident = max(max_ident, abs(p.E_B - (p.H_emit - p.H_emit_given_z)))
            if p.family in ("negate_live", "negate_dead"):
                key = (f"{p.family}/{p.kill_mech or 'live'}",
                       min(p.n_prior_mentions, 2))
                mention_tab[key] = mention_tab.get(key, 0) + 1
        for upto in range(len(st.positions) + 1):
            bf, n_c = _brute_force_support(st, upto)
            inc = {k: frozenset(range(N_VAL)) for k in SLOTS}
            for p in st.positions[:upto]:
                if p.slot is not None and p.constraint is not None:
                    inc[p.slot] = inc[p.slot] & p.constraint
            for k in SLOTS:
                max_post = max(max_post, float(inc[k] != bf[k]))
            if upto < len(st.positions):
                p = st.positions[upto]
                if p.analysis and p.slot is not None:
                    max_hpost = max(max_hpost, abs(
                        sum(LN(len(inc[k])) for k in SLOTS) - p.H_post_before))

    out = {
        "max_posterior_mismatch": max_post,
        "max_identity_err": max_ident,
        "max_H_post_err": max_hpost,
        "family_counts": fam_counts,
        "mean_gen_surprisal_by_family": {
            k: sum(v) / len(v) for k, v in sorted(surp_by_fam.items())},
        "negation_by_prior_mentions": {f"{a}|{b}": c
                                       for (a, b), c in sorted(mention_tab.items())},
    }
    if verbose:
        print("Chronicle oracle self-test")
        print(f"  support vs brute force (0 = exact) : {max_post}")
        print(f"  H_post vs log|support|             : {max_hpost:.3e}")
        print(f"  identity E[B] = H - H|z            : {max_ident:.3e}")
        print("  family counts / mean oracle surprisal (nats):")
        for k in sorted(fam_counts):
            print(f"    {k:16s} n={fam_counts[k]:7d}  "
                  f"surprisal={out['mean_gen_surprisal_by_family'][k]:.4f}")
        print(f"  ln 4 = {LN(4):.4f}")
        print("  Gate B1 support, negation family x prior mentions of the word:")
        for k, v in out["negation_by_prior_mentions"].items():
            print(f"    {k:22s} {v}")
    assert max_post == 0.0, "factorised support disagrees with enumeration"
    assert max_ident < 1e-12, "per-position identity violated"
    assert max_hpost < 1e-12, "H_post is not log|support| at an analysis position"
    return out


if __name__ == "__main__":
    self_test()
