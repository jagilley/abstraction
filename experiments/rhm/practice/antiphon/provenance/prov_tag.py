#!/usr/bin/env python3
"""[antiphon shape P] The provenance primitive, offline, over the banked practice logs.

CPU-only. Reads nothing but already-fetched `figures/<tag>/<arm>/results.json` mirrors and
their `setup.json` under the practice arc's nodes. No Modal, no GPU, no network.

WHAT THE EFFERENCE COPY IS ON THIS SUBSTRATE
--------------------------------------------
The agent's outgoing act is a rollout: it proposes macro slots, executes, and (when the
rollout SOLVES) its own generator parses the corrected configuration into level-1 features.
`macros.Miner.observe` counts those parses; `Miner.state()["keys_at_support"]`
(the identity instrument installed since `tall/`) is the sorted list of level-l flat keys the
agent has itself emitted at least `mine_support` times.

    E_l(c) = { level-l flat keys the agent's OWN solved rollouts have produced,
               at support >= mine_support, by cycle c }

`E_l` is the efference-copy log: one entry per question the agent posed and got an answer to,
in its own hand. It is cumulative and monotone (asserted, gate G0).

WHAT "AN ARRIVING DATUM MATCHING NO OUTGOING QUESTION" COMPUTES TO
-----------------------------------------------------------------
The arriving data are the entries of the committed macro table — the vocabulary the executor
DP serves from. Entry i at level l carries a flat key k_i. At cycle c:

    self(i, c)  <=>  k_i in E_l(c)          reafference: I have emitted this spelling myself
    other(i, c) <=>  k_i not in E_l(c)      exafference: someone else's answer

Two tags, and the difference between them is the whole point:

  * `arr`  — frozen at the arrival cycle (commit, or c1 for a gift). What was exafferent when
             it landed.
  * `live` — re-evaluated every cycle. Exafference DECAYS: a gifted entry becomes self the
             moment the agent independently re-derives its spelling. This is provenance of
             *re-derivation*, not of origin, and it is the arity-2 reading — the datum joins
             the agent's causal history when the agent's own act reproduces it.

Two weightings, both reported:

  * by HOLDING  — over table entries (what I have).
  * by USE      — weighted by `log["entry"]["hist"][phase][l][i]`, the per-cycle count of
                  executions the max-sum DP actually served from entry i (`assay`'s
                  entry-identity instrument). This is the provenance of what the agent
                  actually leans on, which is the quantity `census`/`assay` interpretation
                  (a) ("use is only earnable") is about.

KEY MAPS (index -> flat key), per arm per level, all from banked bytes
---------------------------------------------------------------------
  * surgery event with `flat_after`  (`complete`, `junk_dose`, `strip`) -> verbatim
  * surgery mode `exact`             -> `setup.json["true_tables"][l]["flat"]` (verbatim table)
  * gift at c1 (`vocab == "true"`)   -> same
  * earned (no surgery)              -> RECONSTRUCTED from E_l(commit) by replaying
                                        `Miner.build`: sorted(E_l) filtered by buildability
                                        over the arm's own committed lower table.

GATES
-----
G0  `keys_at_support` monotone in c, every arm, every level.
G1  earned tables: the reconstruction from the efference-copy log must reproduce the logged
    `vocab[l]` size AND the logged `true_mask` element-wise. This is the match rule verified
    against an independent artifact rather than assumed; it also forces self_frac == 1.000 on
    every earned arm, which is the construction check.
G2  gifted tables: logged `true_mask` must be all-ones with length == |true_tables[l]|.
G3  surgical tables: `flat_after` length == `true_mask` length == `vocab[l]`, and the truth
    mask recomputed from the keys must equal the logged one.

USAGE
-----
    python3 rhm/practice/antiphon/provenance/prov_tag.py [--figures] [--tags as_s0,wd_s1]
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANTIPHON = os.path.dirname(HERE)
PRACTICE = os.path.dirname(ANTIPHON)
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(PRACTICE, "woodshed"))

import reduce_trust as RT  # noqa: E402  (the donor instrument; imported, not re-implemented)

# (node, tag, family, note).  The provenance cells live in the tags that carry BOTH
# `setup.json["true_tables"]` (the index->key map for gifted tables) and `entry_rec`.
MANIFEST = [
    # [antiphon_p] the proposal-grain round; same substrate, wd_s1's arms and order
    ("antiphon/provenance", "ap_s0", "P'", "wd_s1 replayed with the proposal-grain join"),
    ("assay", "as_s0", "F0", "the surgery six: arrival x content"),
    ("assay", "as_s1", "F0", "stream-displaced twins of given_c1 / exact"),
    ("woodshed", "wd_s0", "F1", "rehearsal at near-null dose"),
    ("woodshed", "wd_s1", "F1", "rehearsal at corrected dose"),
]
# Earned-only tags: no gift anywhere, so every entry must tag self. These are the population
# G1 runs over — the match rule's construction check, at scale.
EARNED_MANIFEST = [
    ("conductor", "cd_s0", "A1", "thermostat round"),
    ("maestro", "ma_s0", "A2", "learned-policy round"),
    ("maestro", "ma_s1", "A2", "A2 stream twins"),
    ("crescendo", "cr3_s0", "A3", "L4 opened, draw A"),
    ("crescendo", "cr3_s1", "A3", "L4 opened, draw B"),
    ("audiation", "au_s1", "E1b", "per-datum credit vs pooled anchor"),
    ("intonation", "in_s0", "bonus", "delta_perf on the crescendo stack"),
    ("intonation", "ma_s0", "bonus", "delta_perf on the maestro stack"),
    ("caesura", "ca_s0", "bonus", "delta_silence pacing"),
    ("tacet", "tc_s0", "bonus", "the gate family"),
    ("ostinato", "os_s0", "bonus", "the rho ladder"),
]

LEVELS = (2, 3)  # the levels a macro table is committed at on this substrate


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def tag_root(node, tag):
    return os.path.join(PRACTICE, node, "figures", tag)


def load_setup(node, tag):
    p = os.path.join(tag_root(node, tag), "setup.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def load_raw(node, tag, arm):
    p = os.path.join(tag_root(node, tag), arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def arms_of(node, tag):
    root = tag_root(node, tag)
    if not os.path.isdir(root):
        return []
    return [a for a in sorted(os.listdir(root))
            if os.path.isfile(os.path.join(root, a, "results.json"))]


# --------------------------------------------------------------------------- #
# the efference-copy log
# --------------------------------------------------------------------------- #

def efference_log(raw, levels=LEVELS):
    """E_l(c) as a list (per cycle) of frozensets of keys, per level.  Gate G0 inline."""
    mn = raw["log"]["miner"]
    out, viol = {}, []
    for ell in levels:
        seq, prev = [], frozenset()
        for c, rec in enumerate(mn):
            ks = frozenset(tuple(int(x) for x in k)
                           for k in ((rec or {}).get(str(ell)) or {}).get("keys_at_support", []))
            if not prev <= ks:
                viol.append((ell, c + 1, len(prev - ks)))
            seq.append(ks)
            prev = ks
        out[ell] = seq
    return out, viol


# --------------------------------------------------------------------------- #
# index -> key maps for the committed table
# --------------------------------------------------------------------------- #

def build_from_efference(keys, lower_flat, s, span):
    """Replay `macros.Miner.build`'s ordering: sorted keys, halves resolvable in `lower`."""
    half = span // s
    lut = {tuple(int(x) for x in row) for row in lower_flat}
    rows = []
    for k in sorted(keys):
        kids = [tuple(k[i * half:(i + 1) * half]) for i in range(s)]
        if any(kk not in lut for kk in kids):
            continue
        rows.append(tuple(k))
    return rows


def key_maps(raw, setup, eff, levels=LEVELS):
    """Per level, a list over cycles of (keys or None).  keys[i] is entry i's flat key."""
    cfg = raw["config"]
    s, depth = int(cfg["s"]), int(cfg["depth"])
    n_cycles = len(raw["log"]["cycle"])
    vocab = raw["log"]["vocab"]
    truth = {int(k): [tuple(int(x) for x in r) for r in v["flat"]]
             for k, v in (setup.get("true_tables") or {}).items()}
    ev = raw.get("events") or []

    # surgery events, per level, in cycle order
    surg = {}
    for e in ev:
        if e.get("kind") == "surgery":
            surg.setdefault(int(e["level"]), []).append(e)
    for v in surg.values():
        v.sort(key=lambda e: int(e["cycle"]))

    # a gift at c1 leaves no events at all: the table simply exists at cycle 1
    voc0 = vocab[0] or {}
    gift_c1 = {ell: voc0.get(str(ell)) is not None for ell in levels}

    maps, prov_kind = {}, {}
    for ell in levels:
        maps[ell] = [None] * n_cycles
        prov_kind[ell] = [None] * n_cycles
    # level 1 flat = single features 0..v-1 (Miner's halves at level 2)
    base_flat = [(i,) for i in range(int(cfg["v"]))]

    # `census`'s extension op APPENDS admitted keys (logged verbatim in the event), so an
    # extending arm's index order is commit-order-then-admission-order, not sorted order.
    adm = {}
    for e in ev:
        if e.get("kind") == "extend" and e.get("admitted"):
            adm.setdefault(int(e["level"]), []).append(
                (int(e["cycle"]),
                 [tuple(int(x) for x in (k["key"] if isinstance(k, dict) else k))
                  for k in e["admitted"]]))
    for v in adm.values():
        v.sort(key=lambda t: t[0])

    for ell in levels:
        span = s ** (ell - 1)
        prev_n, cur = None, None
        for c in range(n_cycles):
            n = (vocab[c] or {}).get(str(ell))
            if n is None:
                prev_n = None
                continue
            # the committed table is a FROZEN snapshot: rebuild only when its size moves
            # (commit, a recert that re-mines) — otherwise carry it forward, appending the
            # keys any extension admitted at this cycle.
            if n != prev_n or cur is None:
                se = None
                for e in surg.get(ell, []):
                    if int(e["cycle"]) <= c + 1:
                        se = e
                new_adm = [k for cy, ks in adm.get(ell, []) if cy == c + 1 for k in ks]
                if se is not None and se.get("flat_after") is not None:
                    cur = ([tuple(int(x) for x in r) for r in se["flat_after"]],
                           "surgery:" + se["mode"])
                elif se is not None and se["mode"] == "exact":
                    cur = (list(truth.get(ell, [])), "surgery:exact")
                elif gift_c1[ell]:
                    cur = (list(truth.get(ell, [])), "gift@c1")
                elif cur is not None and new_adm and len(cur[0]) + len(new_adm) == n:
                    cur = (cur[0] + new_adm, "earned+extend")
                else:
                    lower = base_flat if ell == 2 else (
                        maps[ell - 1][c][0] if maps[ell - 1][c] else None)
                    if lower is None:
                        prev_n = n
                        continue
                    cur = (build_from_efference(eff[ell][c], lower, s, span), "earned")
            maps[ell][c] = cur
            prov_kind[ell][c] = cur[1]
            prev_n = n
    # unwrap: maps[ell][c] -> key list
    for ell in levels:
        maps[ell] = [(m[0] if m else None) for m in maps[ell]]
    return maps, prov_kind, truth


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def run_gates(raw, maps, prov_kind, truth, levels=LEVELS):
    """G1/G2/G3: every cycle where a table exists, the key map must reproduce the logged
    `true_mask` element-wise and `vocab[l]`'s size."""
    log = raw["log"]
    rows = []
    for ell in levels:
        n_ck, n_bad, first_bad = 0, 0, None
        for c in range(len(log["cycle"])):
            n = (log["vocab"][c] or {}).get(str(ell))
            if n is None:
                continue
            km = maps[ell][c]
            tm = ((log["entry"][c] or {}).get("true_mask") or {}).get(str(ell))
            if km is None or tm is None:
                n_bad += 1
                first_bad = first_bad or (c + 1, "missing map/mask")
                continue
            n_ck += 1
            tset = set(truth.get(ell, []))
            mine = [int(k in tset) for k in km]
            ok = (len(km) == n == len(tm)) and mine == list(tm)
            if not ok:
                n_bad += 1
                if first_bad is None:
                    first_bad = (c + 1, f"n={n} |km|={len(km)} |tm|={len(tm)} "
                                        f"mask_eq={mine == list(tm)}")
        rows.append({"level": ell, "kind": next((k for k in prov_kind[ell] if k), None),
                     "n_checked": n_ck, "n_bad": n_bad, "first_bad": first_bad})
    return rows


# --------------------------------------------------------------------------- #
# the provenance tags and their reductions
# --------------------------------------------------------------------------- #

def tag_arm(raw, setup, levels=LEVELS):
    eff, viol = efference_log(raw, levels)
    maps, prov_kind, truth = key_maps(raw, setup, eff, levels)
    gates = run_gates(raw, maps, prov_kind, truth, levels)
    log = raw["log"]
    n_cycles = len(log["cycle"])

    # arrival cycle per level: first cycle the table is non-None
    arrival = {}
    for ell in levels:
        for c in range(n_cycles):
            if (log["vocab"][c] or {}).get(str(ell)) is not None:
                arrival[ell] = c + 1
                break

    out = {"eff": eff, "maps": maps, "prov_kind": prov_kind, "truth": truth,
           "gates": gates, "g0_viol": viol, "arrival": arrival, "series": {}}

    for ell in levels:
        a = arrival.get(ell)
        if a is None:
            continue
        arr_keys = maps[ell][a - 1]
        arr_other = None
        if arr_keys is not None:
            arr_other = frozenset(k for k in arr_keys if k not in eff[ell][a - 1])
        ser = {k: np.full(n_cycles, np.nan) for k in
               ("n_ent", "n_self", "self_frac", "n_self_arr", "self_frac_arr",
                "conv", "use_beam", "use_probe",
                "self_use_beam", "self_use_probe",
                "self_use_beam_arr", "self_use_probe_arr",
                "n_eff", "n_eff_true", "n_used", "ess_use")}
        cum = None   # cumulative per-entry execution counts (beam), for the concentration read
        for c in range(n_cycles):
            km = maps[ell][c]
            if km is None:
                continue
            E = eff[ell][c]
            selfmask = np.array([1 if k in E else 0 for k in km], float)
            ser["n_ent"][c] = len(km)
            ser["n_self"][c] = selfmask.sum()
            ser["self_frac"][c] = selfmask.mean() if len(km) else np.nan
            ser["n_eff"][c] = len(E)
            ser["n_eff_true"][c] = sum(1 for k in E if k in set(truth.get(ell, [])))
            # frozen-at-arrival tag, carried forward on the arrival key list
            if arr_keys is not None and len(arr_keys) == len(km):
                arrmask = np.array([1 if k in eff[ell][a - 1] else 0 for k in km], float)
                ser["n_self_arr"][c] = arrmask.sum()
                ser["self_frac_arr"][c] = arrmask.mean()
                if arr_other:
                    conv = sum(1 for k in arr_other if k in E) / len(arr_other)
                    ser["conv"][c] = conv
                else:
                    ser["conv"][c] = np.nan
            hist = ((log["entry"][c] or {}).get("hist") or {})
            hb = np.array((hist.get("beam") or {}).get(str(ell), []) or [], float)
            if hb.size == len(km):
                cum = hb.copy() if (cum is None or cum.size != hb.size) else cum + hb
                ser["n_used"][c] = float((cum > 0).sum())
                tot_c = cum.sum()
                if tot_c > 0:
                    q = cum[cum > 0] / tot_c
                    ser["ess_use"][c] = float(np.exp(-(q * np.log(q)).sum()))
            for phase, tgt, tgt_arr in (("beam", "self_use_beam", "self_use_beam_arr"),
                                        ("probe", "self_use_probe", "self_use_probe_arr")):
                h = np.array((hist.get(phase) or {}).get(str(ell), []) or [], float)
                if h.size != len(km):
                    continue
                tot = h.sum()
                ser["use_" + phase][c] = tot
                if tot > 0:
                    ser[tgt][c] = float((h * selfmask).sum() / tot)
                    if arr_keys is not None and len(arr_keys) == len(km):
                        ser[tgt_arr][c] = float((h * arrmask).sum() / tot)
        out["series"][ell] = ser
    return out


def fmt(x, w=8, p=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return " " * (w - 1) + "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):>{w}d}"
    return f"{float(x):>{w}.{p}f}"


def last_valid(a):
    idx = np.where(np.isfinite(a))[0]
    return (float(a[idx[-1]]), int(idx[-1])) if idx.size else (float("nan"), -1)


def at_cycle(a, c):
    return float(a[c - 1]) if 0 <= c - 1 < len(a) else float("nan")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tags", default="")
    args = ap.parse_args()
    want = set(t for t in args.tags.split(",") if t)

    os.makedirs(OUT, exist_ok=True)
    L, P = [], print

    def emit(sline=""):
        L.append(sline)
        P(sline)

    emit("=" * 100)
    emit("[antiphon shape P] PROVENANCE, OFFLINE — efference copies vs the banked practice logs")
    emit("=" * 100)
    emit("")
    emit("efference copy  E_l(c) = level-l flat keys the agent's OWN solved rollouts emitted")
    emit("                        at support >= mine_support, by cycle c "
         "(log['miner'][c][l]['keys_at_support'])")
    emit("match rule      entry i of the committed table is SELF at c iff key_i in E_l(c);")
    emit("                otherwise OTHER (exafference: an arriving datum matching no")
    emit("                outgoing question).  `arr` freezes the tag at the arrival cycle;")
    emit("                `live` re-evaluates it every cycle (re-derivation converts).")
    emit("")

    store = OrderedDict()
    all_manifest = [(n, t, f, note, True) for (n, t, f, note) in MANIFEST] + \
                   [(n, t, f, note, False) for (n, t, f, note) in EARNED_MANIFEST]
    for node, tag, fam, note, is_prov in all_manifest:
        if want and tag not in want:
            continue
        setup = load_setup(node, tag)
        if setup is None or "true_tables" not in setup:
            continue
        for arm in arms_of(node, tag):
            raw = load_raw(node, tag, arm)
            if raw is None or not (raw.get("log") or {}).get("entry"):
                continue
            try:
                res = tag_arm(raw, setup)
            except Exception as e:  # noqa: BLE001
                emit(f"  !! {node}/{tag}/{arm}: {type(e).__name__}: {e}")
                continue
            store[(node, tag, arm)] = dict(res=res, raw=raw, fam=fam, is_prov=is_prov,
                                           note=note)

    # ------------------------------------------------------------------ (0) gates
    emit("-" * 100)
    emit("(0) GATES — the match rule verified against independent artifacts")
    emit("-" * 100)
    emit("")
    emit("G0  keys_at_support monotone in cycle (the efference log only ever grows)")
    emit("G1/G2/G3  the index->key map reproduces the logged per-cycle true_mask and vocab size")
    emit("")
    emit(f"{'node/tag/arm':<38} {'lvl':>3} {'source':<16} {'checked':>8} {'bad':>5}  first_bad")
    n_pass = n_fail = 0
    g0_bad = 0
    for (node, tag, arm), rec in store.items():
        res = rec["res"]
        g0_bad += len(res["g0_viol"])
        for g in res["gates"]:
            if g["n_checked"] == 0 and g["n_bad"] == 0:
                continue
            ok = g["n_bad"] == 0
            n_pass += ok
            n_fail += (not ok)
            if not ok or arm in ("anchor", "exact", "given_c1", "complete", "junk_dose",
                                 "strip", "exact_reh", "exact_exp"):
                emit(f"{node}/{tag}/{arm:<18.18} {g['level']:>3} {str(g['kind'] or '-'):<16} "
                     f"{g['n_checked']:>8} {g['n_bad']:>5}  {g['first_bad'] or ''}")
    emit("")
    emit(f"G0 violations: {g0_bad}   |   key-map cells PASS {n_pass} / FAIL {n_fail}")
    emit("")

    # ------------------------------------------------------------------ (1) the provenance table
    emit("-" * 100)
    emit("(1) PROVENANCE OF THE COMMITTED VOCABULARY — holding-weighted")
    emit("-" * 100)
    emit("")
    emit("  arrival     the cycle the table first exists (commit, or 1 for a gift)")
    emit("  n_ent       entries at arrival / at end")
    emit("  self@arr    fraction of the arriving table already in the efference log")
    emit("  self@end    fraction in the efference log at the last cycle (live tag)")
    emit("  conv        of the entries EXAFFERENT at arrival, the fraction the agent has")
    emit("              since re-derived itself  (reafference conversion)")
    emit("  |E|@end     size of the efference log at the last cycle")
    emit("")
    hdr = (f"{'tag':<8} {'arm':<14} {'l':>2} {'arrival':>7} {'n_ent0':>7} {'n_entN':>7} "
           f"{'self@arr':>9} {'self@end':>9} {'conv':>7} {'|E|@end':>8} {'|E∩T|':>7}")
    for is_prov in (True, False):
        emit(hdr if is_prov else "")
        if not is_prov:
            emit("--- earned-only tags (no gift anywhere): the construction check ---")
            emit(hdr)
        for (node, tag, arm), rec in store.items():
            if rec["is_prov"] != is_prov:
                continue
            res = rec["res"]
            for ell in LEVELS:
                ser = res["series"].get(ell)
                if ser is None:
                    continue
                a = res["arrival"][ell]
                nend, _ = last_valid(ser["n_ent"])
                sfe, _ = last_valid(ser["self_frac"])
                cv, _ = last_valid(ser["conv"])
                ne, _ = last_valid(ser["n_eff"])
                net, _ = last_valid(ser["n_eff_true"])
                emit(f"{tag:<8} {arm:<14} {ell:>2} {a:>7d} "
                     f"{fmt(at_cycle(ser['n_ent'], a), 7, 0)} {fmt(nend, 7, 0)} "
                     f"{fmt(at_cycle(ser['self_frac'], a), 9)} {fmt(sfe, 9)} "
                     f"{fmt(cv, 7)} {fmt(ne, 8, 0)} {fmt(net, 7, 0)}")
        if is_prov:
            emit("")

    # ------------------------------------------------------------------ (2) use-weighted
    emit("")
    emit("-" * 100)
    emit("(2) PROVENANCE OF USE — execution-weighted (assay's entry-identity instrument)")
    emit("-" * 100)
    emit("")
    emit("  self_use    share of macro EXECUTIONS at this level served by a self-tagged entry")
    emit("  *_arr       the same with the tag frozen at arrival (origin, not re-derivation)")
    emit("  beam = the practice rollouts; probe = the metering rollouts")
    emit("")
    emit("  n_used / n_ent  distinct entries ever served, out of the table's size")
    emit("  ess_use         exp(entropy) of the cumulative per-entry use distribution — the")
    emit("                  EFFECTIVE number of entries the agent actually leans on")
    emit("")
    emit(f"{'tag':<8} {'arm':<14} {'l':>2} "
         f"{'beam@end':>9} {'beamarr':>9} {'probe@end':>10} {'probearr':>9} "
         f"{'beam_tot':>10} {'mean_beam':>10} {'n_used':>7} {'n_ent':>6} {'ess_use':>8}")
    for (node, tag, arm), rec in store.items():
        if not rec["is_prov"]:
            continue
        res = rec["res"]
        for ell in LEVELS:
            ser = res["series"].get(ell)
            if ser is None:
                continue
            b, _ = last_valid(ser["self_use_beam"])
            ba, _ = last_valid(ser["self_use_beam_arr"])
            p_, _ = last_valid(ser["self_use_probe"])
            pa, _ = last_valid(ser["self_use_probe_arr"])
            tot = np.nansum(ser["use_beam"])
            wm = ser["self_use_beam"]
            w = ser["use_beam"]
            msk = np.isfinite(wm) & np.isfinite(w)
            mean_b = float((wm[msk] * w[msk]).sum() / w[msk].sum()) if msk.any() and \
                w[msk].sum() > 0 else float("nan")
            nu, _ = last_valid(ser["n_used"])
            eu, _ = last_valid(ser["ess_use"])
            ne_, _ = last_valid(ser["n_ent"])
            emit(f"{tag:<8} {arm:<14} {ell:>2} "
                 f"{fmt(b, 9)} {fmt(ba, 9)} {fmt(p_, 10)} {fmt(pa, 9)} "
                 f"{fmt(tot, 10, 0)} {fmt(mean_b, 10)} {fmt(nu, 7, 0)} {fmt(ne_, 6, 0)} "
                 f"{fmt(eu, 8, 2)}")

    # ------------------------------------------------------------------ (3) the two clocks
    emit("")
    emit("-" * 100)
    emit("(3) THE TWO CLOCKS — how fast exafference decays vs how fast trust forms")
    emit("-" * 100)
    emit("")
    emit("  The provenance clock runs on the CYCLE grid (entry.hist is per-cycle); the trust")
    emit("  clock runs on the PROBE grid (probe_every = 8), so any provenance time constant")
    emit("  below ~8 cycles is not resolvable by the trust instrument it is being compared to.")
    emit("")
    emit("  su@0/8/16   use-weighted self share at tau = 0 / 8 / 16 cycles after arrival")
    emit("  t_su.5/.9   first tau at which the use-weighted self share reaches 0.50 / 0.90")
    emit("  cross1      woodshed: first tau at/after the lift trough where pi reaches chance")
    emit("  t_half      woodshed: cycles to half the arm's own final pi mass")
    emit("")
    emit(f"{'tag':<8} {'arm':<14} {'l':>2} {'su@0':>7} {'su@8':>7} {'su@16':>7} "
         f"{'t_su.5':>7} {'t_su.9':>7} | {'cross1':>7} {'t_half':>7} {'liftmin':>8} "
         f"{'mass@N':>7} {'amax@N':>7}")
    cross_rows = []
    for (node, tag, arm), rec in store.items():
        if not rec["is_prov"]:
            continue
        res = rec["res"]
        ra = RT.load_arm(os.path.join(tag_root(node, tag), arm, "results.json"))
        if ra is None:
            continue
        for ell in LEVELS:
            ser = res["series"].get(ell)
            if ser is None or ell not in ra["mass"]:
                continue
            st = RT.rate_stats(ra, ell) or {}
            a = res["arrival"][ell]
            su = ser["self_use_beam"]
            tau = np.arange(len(su), dtype=float) - (a - 1)
            ok = np.isfinite(su) & (tau >= 0)
            t, y = tau[ok], su[ok]

            def _at(x):
                return float(np.interp(x, t, y)) if len(t) and t[0] <= x <= t[-1] else np.nan

            def _cross(thr):
                idx = np.where(y >= thr)[0]
                if not idx.size:
                    return float("nan")
                i = int(idx[0])
                if i == 0:
                    return float(t[0])
                y0, y1 = y[i - 1], y[i]
                return float(t[i - 1] + (t[i] - t[i - 1]) * (thr - y0) / (y1 - y0)) \
                    if y1 != y0 else float(t[i])

            sfe, _ = last_valid(ser["self_frac"])
            cv, _ = last_valid(ser["conv"])
            sue, _ = last_valid(ser["self_use_beam"])
            wm, w = ser["self_use_beam"], ser["use_beam"]
            msk = np.isfinite(wm) & np.isfinite(w) & (w > 0)
            mean_b = float((wm[msk] * w[msk]).sum() / w[msk].sum()) if msk.any() else np.nan
            n_self, _ = last_valid(ser["n_self"])
            row = dict(node=node, tag=tag, arm=arm, ell=ell, arrival=a,
                       mass=float(ra["mass"][ell][-1]), amax=float(ra["amax"][ell][-1]),
                       lift_end=float(st.get("lift_end", np.nan)),
                       lift_min=float(st.get("lift_min", np.nan)),
                       tau_cross1=float(st.get("tau_cross1", np.nan)),
                       t_half=float(st.get("t_half", np.nan)),
                       self_frac_end=sfe, n_self_end=n_self, conv=cv,
                       self_use_end=sue, self_use_mean=mean_b,
                       su0=_at(0.0), su8=_at(8.0), su16=_at(16.0),
                       t_su50=_cross(0.5), t_su90=_cross(0.9),
                       conc=(sue / sfe if (sfe and np.isfinite(sfe) and sfe > 0) else np.nan))
            cross_rows.append(row)
            emit(f"{tag:<8} {arm:<14} {ell:>2} {fmt(row['su0'], 7)} {fmt(row['su8'], 7)} "
                 f"{fmt(row['su16'], 7)} {fmt(row['t_su50'], 7, 1)} {fmt(row['t_su90'], 7, 1)} | "
                 f"{fmt(row['tau_cross1'], 7, 1)} {fmt(row['t_half'], 7, 1)} "
                 f"{fmt(row['lift_min'], 8)} {fmt(row['mass'], 7)} {fmt(row['amax'], 7)}")

    # ------------------------------------------------------------------ (4) the cross
    emit("")
    emit("-" * 100)
    emit("(4) DOES COMPUTED PROVENANCE REPRODUCE WHAT THE ARM LABELS ESTABLISHED?")
    emit("-" * 100)
    R = {(r["tag"], r["arm"], r["ell"]): r for r in cross_rows}

    def show(title, cells, cols):
        emit("")
        emit(f"  {title}")
        emit("  " + f"{'arm':<16}" + "".join(f"{c:>14}" for c in cols))
        for tag, arm, ell in cells:
            r = R.get((tag, arm, ell))
            if r is None:
                continue
            emit("  " + f"{arm + '@L' + str(ell):<16}"
                 + "".join(fmt(r.get(c), 14) for c in cols))

    cols = ["amax", "mass", "self_frac_end", "n_self_end", "self_use_end",
            "self_use_mean", "conv"]
    emit("")
    emit("  amax/mass      = pi's L3 argmax share and proposal mass at the last probe (TRUST)")
    emit("  self_frac_end  = share of the committed table the agent has re-derived (HOLDING)")
    emit("  n_self_end     = the same as a count of entries")
    emit("  self_use_end   = share of L3 executions served by a re-derived entry (USE, last cycle)")
    emit("  self_use_mean  = the same, execution-weighted over the whole run")
    emit("  conv           = of the entries exafferent AT ARRIVAL, the share since re-derived")
    show("(a) assay finding 6, ARRIVAL DOMINATES (as_s0): gift@c1 vs commit-time gift",
         [("as_s0", "given_c1", 3), ("as_s0", "exact", 3), ("as_s0", "complete", 3),
          ("as_s0", "junk_dose", 3), ("as_s0", "anchor", 3), ("as_s0", "strip", 3)], cols)
    show("(b) woodshed wd_s1, CREDIT > EXPOSURE > NONE (all three are `exact` tables at L3)",
         [("wd_s1", "exact_reh", 3), ("wd_s1", "exact_exp", 3), ("wd_s1", "exact", 3)], cols)
    show("(c) the same three arms at L2, where rehearsal never fired — the in-tag handle",
         [("wd_s1", "exact_reh", 2), ("wd_s1", "exact_exp", 2), ("wd_s1", "exact", 2)], cols)
    show("(d) wd_s0, the near-null dose (rehearsal solve rate 0.003)",
         [("wd_s0", "exact_reh", 3), ("wd_s0", "exact_exp", 3), ("wd_s0", "exact", 3)], cols)
    show("(e) as_s1 stream-displaced twins — the only floor available for these statistics",
         [("as_s1", "given_c1_j", 3), ("as_s0", "given_c1", 3),
          ("as_s1", "exact_j", 3), ("as_s0", "exact", 3)], cols)

    # rank agreement on the arrival cross
    from itertools import combinations
    cellsA = [("as_s0", a, 3) for a in
              ("given_c1", "exact", "complete", "junk_dose", "anchor", "strip")]
    emit("")
    emit("  (a') rank agreement with pi's argmax share over the six as_s0 arms at L3")
    emit("       (Kendall tau_b; +1 = the provenance statistic orders the arms exactly as")
    emit("        trust does, -1 = exactly inverted)")
    ref = [R[k]["amax"] for k in cellsA if k in R]
    line = []
    for c in cols[2:]:
        v = [R[k].get(c) for k in cellsA if k in R]
        n_c = n_d = 0
        for i, j in combinations(range(len(ref)), 2):
            if v[i] is None or not np.isfinite(v[i]) or not np.isfinite(v[j]):
                continue
            a_, b_ = np.sign(ref[i] - ref[j]), np.sign(v[i] - v[j])
            if a_ * b_ > 0:
                n_c += 1
            elif a_ * b_ < 0:
                n_d += 1
        line.append(fmt((n_c - n_d) / max(n_c + n_d, 1), 14, 2))
    emit("  " + f"{'kendall tau_b vs amax':<16}" + fmt(1.0, 14, 2) + fmt(np.nan, 14)
         + "".join(line))

    emit("")
    emit("  FLOORS AND HANDLES (spreads, on each statistic)")
    emit("  " + f"{'handle':<44}" + "".join(f"{c:>14}" for c in cols))

    def spread(cells, label):
        vals = {c: [R[k].get(c) for k in cells if k in R] for c in cols}
        out, keep = [], {}
        for c in cols:
            v = [x for x in vals[c] if x is not None and np.isfinite(x)]
            keep[c] = (max(v) - min(v)) if len(v) >= 2 else np.nan
            out.append(fmt(keep[c], 14))
        emit("  " + f"{label:<44}" + "".join(out))
        return keep

    spread([("as_s0", "given_c1", 3), ("as_s1", "given_c1_j", 3)],
           "stream-twin floor, given family @L3 (n=1 pair)")
    f_str_e = spread([("as_s0", "exact", 3), ("as_s1", "exact_j", 3)],
                     "stream-twin floor, exact family @L3 (n=1 pair)")
    h_l2 = spread([("wd_s1", "exact_reh", 2), ("wd_s1", "exact_exp", 2),
                   ("wd_s1", "exact", 2)], "in-tag handle, wd_s1 unrehearsed L2 (n=3)")
    h_s0 = spread([("wd_s0", "exact_reh", 3), ("wd_s0", "exact_exp", 3),
                   ("wd_s0", "exact", 3)],
                  "NULL-DOSE handle, wd_s0 L3 (n=3, dose 0.003)")
    e_l3 = spread([("wd_s1", "exact_reh", 3), ("wd_s1", "exact_exp", 3),
                   ("wd_s1", "exact", 3)], "wd_s1 REHEARSED L3 spread (the effect)")

    emit("")
    emit("  the effect in floor-multiples  (>1 = the effect is larger than the handle)")
    emit("  " + f"{'ratio':<44}" + "".join(f"{c:>14}" for c in cols))
    for lab, den in (("L3 effect / exact-family stream floor", f_str_e),
                     ("L3 effect / in-tag unrehearsed-L2 handle", h_l2),
                     ("L3 effect / wd_s0 NULL-DOSE L3 handle", h_s0)):
        emit("  " + f"{lab:<44}"
             + "".join(fmt(e_l3[c] / den[c] if den.get(c) and np.isfinite(den[c])
                           and den[c] > 0 else np.nan, 14, 2) for c in cols))

    emit("")
    emit("-" * 100)
    emit("(5) CAVEATS")
    emit("-" * 100)
    emit("")
    emit("  * The efference log is DOWNSTREAM of use: an entry enters E_l only by being")
    emit("    served often enough that the agent's own solved rollouts spell it out at")
    emit("    support >= 3. Self-tagging is therefore a near-deterministic consequence of")
    emit("    an entry being used, not an independent variable — section (2)'s concentration")
    emit("    column is mechanism, not effect.")
    emit("  * No floor for pi mass exists anywhere in the arc (woodshed FILES.md), and no")
    emit("    floor for ANY provenance statistic exists either. The only handles are the")
    emit("    as_s1 stream twins (n=1 pair per family) and wd_s1's unrehearsed L2 (n=3).")
    emit("  * Single seed everywhere; rehearsal is not compute-matched (~2x practice compute).")
    emit("  * mine_support = 3 is a free parameter of the match rule (a key must be emitted")
    emit("    three times before it counts as posed) and it is NOT sweepable offline: the")
    emit("    banked logs carry `keys_at_support` only at that one threshold. `n_at_support`")
    emit("    below shows how much the log SIZE would move; the membership does not exist on")
    emit("    disk at any other threshold. A machinery limit, not a measured invariance.")
    emit("")
    emit(f"    {'tag/arm':<24} {'l':>2}" + "".join(f"{'sup>=' + str(t):>10}"
                                                   for t in (1, 2, 3, 5, 10)))
    for (node, tag, arm), rec in store.items():
        if not rec["is_prov"] or tag not in ("as_s0",):
            continue
        for ell in LEVELS:
            st = (rec["raw"]["log"]["miner"][-1] or {}).get(str(ell)) or {}
            nas = st.get("n_at_support") or {}
            emit(f"    {tag + '/' + arm:<24} {ell:>2}"
                 + "".join(fmt(nas.get(str(t)), 10, 0) for t in (1, 2, 3, 5, 10)))
    emit("  * `beam` counts practice-rollout executions and `probe` metering executions; the")
    emit("    two agree to <0.01 everywhere, so only `beam` is carried into (3)-(4).")

    txt = os.path.join(OUT, "reduction.txt")
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote {txt}")
    json.dump([{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in r.items()} for r in cross_rows],
              open(os.path.join(OUT, "cross_rows.json"), "w"), indent=1)
    if args.figures:
        make_figures(store, cross_rows)
    return store, cross_rows



# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(store, cross_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ser_of(tag, arm, ell, key):
        for (node, t, a), rec in store.items():
            if t == tag and a == arm:
                s = rec["res"]["series"].get(ell)
                return (None if s is None else s[key],
                        rec["res"]["arrival"].get(ell))
        return None, None

    # ---- f1: the exafference decay curves ---------------------------------- #
    fams = [("as_s0", ["anchor", "strip", "junk_dose", "complete", "exact", "given_c1"],
             "as_s0 — the surgery six"),
            ("wd_s1", ["exact", "exact_exp", "exact_reh", "given_c1"],
             "wd_s1 — the rehearsal one-bit pair")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for r, (tag, arms, title) in enumerate(fams):
        for c, ell in enumerate(LEVELS):
            ax = axes[r][c]
            for arm in arms:
                y, a = ser_of(tag, arm, ell, "self_use_beam")
                if y is None:
                    continue
                x = np.arange(1, len(y) + 1)
                ax.plot(x, y, lw=1.6, label=arm)
                if a:
                    ax.axvline(a, color="0.8", lw=0.7, zorder=0)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{title} · L{ell}", fontsize=10)
            ax.set_ylabel("share of executions served by a\nself-tagged (re-derived) entry"
                          if c == 0 else "")
            if r == 1:
                ax.set_xlabel("cycle")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("[antiphon P] provenance of USE. A commit-time gift lands ~85-95% "
                 "re-derived already; the c1 gift stays 100% exafferent for 47 cycles\n"
                 "and still wins the trust bracket — the tag and the arm labels disagree",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_exafference_decay.png"), dpi=130)
    plt.close(fig)

    # ---- f2: holding vs use, and the two clocks ---------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    rows = [r for r in cross_rows if r["ell"] == 3 and r["tag"] in ("as_s0", "wd_s1")]
    for r in rows:
        ax.scatter(r["self_frac_end"], r["self_use_end"], s=44)
        ax.annotate(f"{r['arm']}", (r["self_frac_end"], r["self_use_end"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="holding = use")
    ax.set_xlabel("self share of HOLDINGS (table entries)")
    ax.set_ylabel("self share of USE (executions)")
    ax.set_title("L3, end of run: use concentrates on\nthe entries the agent re-derives",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[1]
    seen = set()
    for r in sorted(cross_rows, key=lambda q: q["arm"]):
        if not (np.isfinite(r["t_su50"]) and np.isfinite(r["tau_cross1"])):
            continue
        k = (round(r["t_su50"], 1), round(r["tau_cross1"], 1))
        m = "o" if r["ell"] == 3 else "^"
        ax.scatter(r["t_su50"], r["tau_cross1"], s=40, marker=m)
        if k not in seen:
            ax.annotate(f"{r['arm']}@L{r['ell']}", (r["t_su50"], r["tau_cross1"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
        seen.add(k)
    lim = 90
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.axvspan(0, 8, color="0.9", zorder=0)
    ax.set_xlabel("tau at which USE is 50% self-tagged  (cycle grid)")
    ax.set_ylabel("tau at which pi reaches chance share (probe grid)")
    ax.set_title("the two clocks. commit-time gifts land already re-derived\n(shaded = the "
                 "trust instrument's 8-cycle sampling limit); the c1 gift does not",
                 fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    labs = ["amax", "mass", "self_frac_end", "n_self_end", "self_use_end", "self_use_mean"]
    R = {(r["tag"], r["arm"], r["ell"]): r for r in cross_rows}
    trio = ("exact_reh", "exact_exp", "exact")   # credit > exposure > none, per woodshed

    def spr(cells, c):
        v = [R[k][c] for k in cells if k in R and np.isfinite(R[k][c])]
        return (max(v) - min(v)) if len(v) >= 2 else np.nan
    eff = [spr([("wd_s1", a, 3) for a in trio], c) for c in labs]
    hans = [[spr([("wd_s1", a, 2) for a in trio], c) for c in labs],
            [spr([("wd_s0", a, 3) for a in trio], c) for c in labs]]
    # does the statistic ORDER the three arms as trust does (credit > exposure > none)?
    ordok = []
    for c in labs:
        v = [R[("wd_s1", a, 3)][c] for a in trio]
        ordok.append(bool(v[0] > v[1] > v[2]))
    w = 0.38
    for k, (han, lab) in enumerate(zip(hans, ("unrehearsed-L2 handle (n=3)",
                                              "wd_s0 null-dose handle (n=3)"))):
        rat = [e / h if h and np.isfinite(h) and h > 0 else np.nan
               for e, h in zip(eff, han)]
        ax.bar(np.arange(len(labs)) + (k - 0.5) * w, rat, w,
               color=["#4c72b0" if o else "#c44e52" for o in ordok],
               alpha=1.0 if k == 0 else 0.55, edgecolor="k", lw=0.4, label=lab)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    top = np.nanmax([e / h if h and np.isfinite(h) and h > 0 else np.nan
                     for han in hans for e, h in zip(eff, han)])
    ax.set_ylim(0, top * 1.22)
    for i, o in enumerate(ordok):
        ax.text(i, top * 1.06, "order ok" if o else "order X", ha="center", fontsize=6.5,
                color="#4c72b0" if o else "#c44e52")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("wd_s1 L3 spread / handle")
    ax.set_title("no provenance statistic both ORDERS the one-bit\npair as trust does and "
                 "clears its own handle", fontsize=9.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f2_two_clocks.png"), dpi=130)
    plt.close(fig)

    # ---- f3: the credit contrast, in provenance terms ---------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, (key, lab) in enumerate((("self_use_beam", "self share of L3 executions"),
                                    ("n_self", "L3 entries re-derived by the agent"),
                                    ("use_beam", "L3 executions per cycle"))):
        ax = axes[j]
        for arm, col in (("exact_reh", "#c44e52"), ("exact_exp", "#dd8452"),
                         ("exact", "#4c72b0"), ("given_c1", "#55a868")):
            y, a = ser_of("wd_s1", arm, 3, key)
            if y is None:
                continue
            ax.plot(np.arange(1, len(y) + 1), y, lw=1.6, color=col, label=arm)
        ax.axvline(70, color="0.8", lw=0.8, zorder=0)
        ax.axvline(71, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("cycle"); ax.set_ylabel(lab); ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        if key == "use_beam":
            ax.set_yscale("log")
    fig.suptitle("[antiphon P] wd_s1 · L3 arrives at c70, rehearsal fires from c71", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f3_credit_provenance.png"), dpi=130)
    plt.close(fig)
    print("wrote figures to", OUT)

if __name__ == "__main__":
    main()
