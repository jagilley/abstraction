"""[preplay/pp3] THE LEARNER'S OWN OPERATIVE TABLES, RECONSTRUCTED OFFLINE FROM `results.json`.

`preplay` (pp1) priced CONSTRUCTED candidate entries: true entries drawn from the grammar's own
table over the grammar's own lower table, wrong entries random pairs of true lower rows. The
realistic question is the learner's OWN mined entries, whose false entries are the ones its own
miner produced. The loop never banks a table as an object -- but `soundboard.py` 13729 says an
offline replay of any build is exact with `build.picks` beside it, and this file is that replay.

WHAT THE BUILD IS (`enharmonic/quotient.py::ClassMiner.build`, 220-316). For each class-pair key
at support, the ratchet emits the CROSS-PRODUCT of the lower table's rows in the first half's
class with its rows in the second half's class, capped at `spell_cap` spellings per class:

    want       = span // s                              the half width
    lower_flat = [row for row in lower["flat"] if len(row) == want]
    rows       = for key in sorted(counts) with count >= support:
                     picks_i = pick(key_i) for each half; skip the key if any is empty
                     extend with the cross-product of the picks, first half major

WHAT THE DUMP CARRIES, and therefore what this replay needs:

  `log["miner"][-1][l]["keys_at_support"]`    the at-support class-pair keys, ALREADY SORTED by
        `ClassMiner.state` with `sup = 3` hard-coded -- which is `cfg["mine_support"]` on these
        arms, asserted below. `sorted` over the JSONABLE keys (tuples -> lists) is the same
        order as `build`'s `sorted(self.counts.items())` over the raw ones, because the map is
        elementwise and order-preserving on homogeneous keys. Gate R-3 is what checks it.
  `quotient["last_build"][l]["picks"]`        class id (as `str` of the tuple) -> the surviving
        lower-row indices, which IS the build up to the cross-product. Indices are into
        `lower_flat`, which this replay reconstructs itself -- so level l needs the
        reconstructed level-(l-1) table, and the recursion closes at L2 on `MC.base_table(v)`.
  `quotient["last_build"][l]["n_entries"]`    the row count to gate against.

WHY THIS IS THE **OPERATIVE** TABLE AND NOT THE CYCLE'S OWN BUILD. `soundboard.py` 13653-13676
computes `omask` through `operative(ell)`, which calls `miners[ell].build(operative(ell-1), ...)`
bottom-up -- and `build` ASSIGNS `self.last_build`. The `_lb_save`/restore around that block puts
the cycle's own `last_build` back into `log["quot"]`, but the top-level `quotient` dict is
serialised from the miner AFTER it, so `quotient["last_build"]` is the OPERATIVE chain. The
evidence is arithmetic and is gated: `n_lower_rows` reads 8 / 18 / 84 at L2 / L3 / L4, which is
exactly `len(base_table)` and this replay's own L2 and L3 row counts, and `n_entries` reads
18 / 84 / 204, which is exactly the length of `entry.json.gz[-1]["operative_mask"][l]`.

L5 IS OUT OF REACH and is not reconstructed. Its last build has `n_keys_built = 0` (the L4 book
blocks all 14 of its at-support keys) while the operative L5 table still holds the 128 rows of an
EARLIER build, whose keys and picks are not in the dump. The recursion stops at L4.

THE FROZEN COMMITTED TABLES (13 / 59 / 88 / 64 rows) are not reconstructible at all: only their
sizes (`events[kind=commit].n_entries`) and their truth masks (`entry.json.gz[-1]["true_mask"]`)
are banked, never their keys or their picks.

Run locally (no GPU, no Modal):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/learner_tables.py --all
"""

import hashlib
import json
import os

import numpy as np

import rhm.practice.ratchet.macros as MC

LEVELS_OK = (2, 3, 4)           # L5 is out of reach; see the module docstring

# `preplay.ARMS`' keys, repeated here so this file imports without `modal` and can be run
# locally with nothing but numpy and the `rhm` package on the path.
ARM_MAP = {
    "s0_sv": ("sb_s1", "sb_sv_yk"), "s2_sv": ("sb_s2", "sb_sv_yk"),
    "s0_so": ("sb_s1", "sb_so_yk"), "s2_so": ("sb_s2", "sb_so_yk"),
    "s0_yd": ("sb_s1", "sb_yd_yk"), "s2_yd": ("sb_s2", "sb_yd_yk"),
}


# R-4's payload. `entry.json.gz` is NOT on the volume -- only the local mirror carries it --
# so gate R-3 (the per-row truth vector against the banked `operative_mask`) can only be closed
# HERE, on this machine. R-4 carries that closure into every paid container: a digest of the
# exact table R-3 was closed on, so a container that rebuilds a different table from the
# volume's `results.json` cannot price it. Regenerate with `--emit-digests` after R-3 passes,
# never by hand.
DIGESTS = {
    "s0_sv": {
        "2": "f7df730af51afa66eff9256dd373925a104e681cfdb277713a28e56449449eab",
        "3": "c5450f21227727328dffc28e3bb2dd98715c4c9f2586c02ad3f21853b30c726f",
        "4": "11cb7e257f9ad57a95b5b17d4b990f7134fee2c61b74d6fbe63d1fea2a924ee1"
    },
    "s2_sv": {
        "2": "ba16fd65d7b1925384e1d00a3c89e03df1f3d6dc353dabe11d5222876ba643e0",
        "3": "97862d8d8bd5179b0207427e63f95f8c252c241e80bb565c469aa42395bf5ddd",
        "4": "3f75d41e14fc42df40609e574856619cd18f028b48be7af675f4b20c887ae1dd"
    },
    "s0_so": {
        "2": "bb539a7dc2bce5f6e35ab4deb0594f3af9d1c7b982814a0d1ef33bd27f0b79a2",
        "3": "e9d270ec4b5da79c6e554fefb1ba4225231ec3a61163793078e87f049d267699",
        "4": "0997ffa6553329fe71e146cabcb48e2951bb3bb9bb17939a79ee5ceba00c120c"
    },
    "s2_so": {
        "2": "0bda07f442fb96a136db08af26c686c7f188c4872ba984e9b9cc585db7cee3c0",
        "3": "888b4f85d178c93fb736eb9132be154c90677975b5a3f516d4edf92082657990",
        "4": "094d9f83defacbb1ae475801bb5714170d6b0be919b0d6a08b88ede65fbffd68"
    },
    "s0_yd": {
        "2": "79691cd4c32b579164e7c4a5453a1db846def963beda2c343aee6c58abd94ff7",
        "3": "14f755a4e4fb5047299ce42bb77638aa119f804e12f017f935f099c6a09c3ed7",
        "4": "46c080736e635cc80159af7aaa71cb9e438ff103f391c36ca18621c7d3f91413"
    },
    "s2_yd": {
        "2": "12e5db36d88a7d81603df56272d9e62166ca55ac017c76404d8f7081ab36e74a",
        "3": "81e5c4197121700f198095b573cc78b826a628a6e8d0e4395061103ad4ebc2f5",
        "4": "2c18cff0c83f184505b002aad1e6e8d941eb4472f929a70c49488ddc3869191c"
    }
}


def table_digest(tbl, truth_row):
    """sha256 over the child rows, the flat level-1 spellings and the truth flag, in order."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tbl["child"], np.int64).tobytes())
    h.update(np.ascontiguousarray(tbl["flat"], np.int64).tobytes())
    h.update(np.ascontiguousarray(np.asarray(truth_row, np.int64)).tobytes())
    return h.hexdigest()


def _key_strs(key):
    """A jsonable class-pair key -> the `str(class id)` strings `picks` is keyed by."""
    out = []
    for half in key:
        c = tuple(int(x) for x in half) if isinstance(half, (list, tuple)) else int(half)
        out.append(str(c))
    return out


def build_level(ell, keys_at_support, picks, lower, s):
    """`ClassMiner.build`'s emission, replayed. Returns `(table, diag)`."""
    want = (s ** (ell - 1)) // s
    lower_flat = [tuple(int(x) for x in row) for row in lower["flat"] if len(row) == want]
    rows, n_keys_built, inv, n_oob = [], 0, [], 0
    for key in keys_at_support:                 # already sorted by `ClassMiner.state`
        ks = _key_strs(key)
        ix = [picks.get(k) for k in ks]
        if any(q is None or len(q) == 0 for q in ix):
            continue
        # A pick index out of range for the reconstructed lower table means the RECURSION is
        # wrong, not that the build was: `picks` indexes `lower_flat`, so an out-of-range index
        # says this replay's level-(l-1) table is not the one the arm built over. Counted and
        # gated (R-1c) rather than allowed to raise, so the failure is legible.
        if any(int(j) >= len(lower_flat) for q in ix for j in q):
            n_oob += 1
            continue
        n_keys_built += 1
        combos = [[]]
        for q in ix:
            combos = [pre + [int(j)] for pre in combos for j in q]
        inv.append(len(combos))
        rows.extend(combos)
    child = np.array(rows, np.int64).reshape(len(rows), s)
    tbl = MC.make_table(ell, child, lower, s)
    diag = {"n_keys_built": int(n_keys_built), "n_entries": int(len(rows)),
            "n_lower_rows": int(len(lower_flat)), "n_oob": int(n_oob),
            "inventory_max": int(max(inv) if inv else 0),
            "inventory_mean": float(np.mean(inv)) if inv else 0.0}
    return tbl, diag


def reconstruct(res_j, ent_last, truth=None, levels=LEVELS_OK, log=print, arm_key=None):
    """The whole chain, with every gate shown beside it. `ent_last` is
    `entry.json.gz[-1]`; `truth` is `MC.true_tables(...)` or None to skip gate R-2.

    Gates, all reported whether they pass or fail (the caller decides):
      R-0  `mine_support == 3`, the support `ClassMiner.state` hard-codes for `keys_at_support`
      R-1  the reconstructed row count == `last_build[l]["n_entries"]`
      R-1b the reconstructed `n_keys_built`, `n_lower_rows`, `inventory_*` == the logged ones
      R-1c no `picks` index falls outside the reconstructed lower table -- the recursion's own
           consistency check, which fires when level l-1 is not the table the arm built over
      R-2  `MC.grade_table` against the true table == the logged `tab_n_correct` / `tab_precision`
      R-3  the per-row truth vector == `operative_mask[l]`, ROW FOR ROW (the strongest)
      R-3s the same as SETS, so a pure ordering failure is distinguishable from a set failure
      R-4  with `arm_key`: the table digest == the one R-3 was closed on locally (`DIGESTS`),
           which is how the paid containers inherit a gate the volume cannot serve
    """
    cfg = res_j["config"]
    v, s = int(cfg["v"]), int(cfg["s"])
    sup = int(cfg["mine_support"])
    miner_last = res_j["log"]["miner"][-1]
    lb_all = res_j["quotient"]["last_build"]
    aud_last = res_j["log"]["aud"][-1]
    om = ent_last["operative_mask"]

    gates = [{"gate": "R-0", "what": "mine_support == 3 (the support `state` hard-codes)",
              "got": sup, "want": 3, "pass": sup == 3}]
    tables, diags = {1: MC.base_table(v)}, {}
    for ell in levels:
        lb = lb_all.get(str(ell), {})
        kas = miner_last[str(ell)]["keys_at_support"]
        picks = lb.get("picks", {})
        lower = tables[ell - 1]
        tbl, diag = build_level(ell, kas, picks, lower, s)
        tables[ell] = tbl
        diags[ell] = diag
        g = []
        g.append({"gate": f"R-1:L{ell}", "what": "row count == last_build n_entries",
                  "got": diag["n_entries"], "want": int(lb.get("n_entries", -1)),
                  "pass": diag["n_entries"] == int(lb.get("n_entries", -1))})
        g.append({"gate": f"R-1c:L{ell}", "what": "no pick index out of the lower table's range",
                  "got": diag["n_oob"], "want": 0, "pass": diag["n_oob"] == 0})
        for f in ("n_keys_built", "n_lower_rows", "inventory_max"):
            g.append({"gate": f"R-1b:L{ell}:{f}", "what": f"replayed {f} == logged",
                      "got": diag[f], "want": int(lb.get(f, -1)),
                      "pass": diag[f] == int(lb.get(f, -1))})
        if truth is not None:
            gr = MC.grade_table(tbl, truth[ell])
            want_n = int(aud_last[str(ell)]["tab_n_correct"])
            want_l = int(aud_last[str(ell)]["tab_n_learned"])
            g.append({"gate": f"R-2:L{ell}", "what": "grade_table n_correct/n_learned == logged",
                      "got": [gr["n_correct"], gr["n_learned"]], "want": [want_n, want_l],
                      "pass": (gr["n_correct"] == want_n and gr["n_learned"] == want_l)})
            diag["grade"] = gr
        # R-3: the per-row truth vector, row for row. `truth_row` is computed whenever the
        # true tables are in hand; the GATE needs the banked mask beside it, and is simply
        # absent (never silently passed) when `entry.json.gz` is not available.
        if truth is not None:
            tset = {tuple(int(x) for x in r) for r in truth[ell]["flat"]}
            mine = [int(tuple(int(x) for x in r) in tset) for r in tbl["flat"]]
            diag["truth_row"] = mine
            if str(ell) in om:
                mask_log = [int(q) for q in om[str(ell)]]
                g.append({"gate": f"R-3:L{ell}",
                          "what": "per-row truth vector == operative_mask",
                          "got": [len(mine), int(sum(mine))],
                          "want": [len(mask_log), int(sum(mask_log))],
                          "pass": mine == mask_log})
                g.append({"gate": f"R-3s:L{ell}", "what": "...as multisets (order-blind)",
                          "got": sorted(mine), "want": sorted(mask_log),
                          "pass": sorted(mine) == sorted(mask_log), "quiet": True})
        gates.extend(g)
    if arm_key is not None and truth is not None:
        want_d = DIGESTS.get(arm_key, {})
        for ell in levels:
            d_ = table_digest(tables[ell], diags[ell]["truth_row"])
            gates.append({"gate": f"R-4:L{ell}",
                          "what": "table digest == the one R-3 closed on locally",
                          "got": d_[:16], "want": str(want_d.get(str(ell)))[:16],
                          "pass": d_ == want_d.get(str(ell))})
    return tables, diags, gates


def arm_root(repo, tag, arm):
    return os.path.join(repo, "rhm/practice/voicing/sotto_voce/aliquot/soundboard/figures",
                        tag, arm)


def load_arm(root):
    import gzip
    res_j = json.load(open(os.path.join(root, "results.json")))
    ent = json.load(gzip.open(os.path.join(root, "entry.json.gz"), "rt"))
    return res_j, ent[-1]


def tables_payload(res_j, tables, diags, levels=LEVELS_OK):
    """The object written to `figures/<arm>_operative_tables.json`: child rows, flat level-1
    spellings, and the truth flag per row."""
    cfg = res_j["config"]
    out = {"arm": res_j["arm"], "seed": int(cfg["seed"]), "rule_seed": int(cfg["rule_seed"]),
           "v": int(cfg["v"]), "s": int(cfg["s"]), "depth": int(cfg["depth"]),
           "spell_cap": int(res_j["quotient"]["spell_cap"]),
           "mine_support": int(cfg["mine_support"]), "levels": {}}
    for ell in levels:
        t = tables[ell]
        d = diags[ell]
        out["levels"][str(ell)] = {
            "n_entries": int(t["child"].shape[0]),
            "child": [[int(x) for x in r] for r in t["child"]],
            "flat": [[int(x) for x in r] for r in t["flat"]],
            "truth": d.get("truth_row"),
            "grade": d.get("grade"),
            "n_keys_built": d["n_keys_built"], "n_lower_rows": d["n_lower_rows"],
            "inventory_max": d["inventory_max"], "inventory_mean": d["inventory_mean"]}
    return out


def falsify(res_j, ent_last, truth, log=print, arm_key=None):
    """Every gate above, shown to FAIL on a deliberately broken input before it is reported.
    The repo's rule: a gate that has never been seen to fail is not a gate."""
    import copy
    cases = []

    def run(name, mutate, gate_prefix, key=arm_key, blind=False):
        r2 = copy.deepcopy(res_j)
        e2 = copy.deepcopy(ent_last) if not blind else {"operative_mask": {}}
        mutate(r2, e2)
        try:
            _, _, gs = reconstruct(r2, e2, truth=truth, arm_key=key)
            hit = [g for g in gs if g["gate"].startswith(gate_prefix) and not g["pass"]]
            ok = len(hit) > 0
            why = hit[0]["gate"] if hit else "NO GATE FIRED"
        except Exception as ex:                     # a crash is also a refusal to report
            ok, why = True, f"raised {type(ex).__name__}"
        cases.append({"case": name, "targets": gate_prefix, "caught": ok, "by": why})
        log(f"  [{'CAUGHT' if ok else 'MISSED'}] {name:<46} -> {why}")

    def drop_key(r2, e2):
        """One at-support key removed at L3 -- but a key whose halves BOTH have picks, since
        dropping an unbuildable key changes no row and would make the case vacuous (it did, on
        s2_sv, on the first pass; see NOTES_own.md defect 1)."""
        kas = r2["log"]["miner"][-1]["3"]["keys_at_support"]
        pk = r2["quotient"]["last_build"]["3"]["picks"]
        for i, key in enumerate(kas):
            if all(pk.get(k) for k in _key_strs(key)):
                kas.pop(i)
                return
        raise AssertionError("no buildable key to drop")

    def truncate_picks(r2, e2):
        pk = r2["quotient"]["last_build"]["3"]["picks"]
        k = sorted(pk, key=lambda q: -len(pk[q]))[0]
        pk[k] = pk[k][:1]

    def flip_support(r2, e2):
        r2["config"]["mine_support"] = 2

    def reverse_rows(r2, e2):
        """The ORDER failure specifically: the same rows, emitted back to front. R-3 must
        catch it and R-3s (the order-blind twin) must not, which is what makes R-3 a
        statement about ORDER and not merely about the set."""
        e2["operative_mask"]["3"] = list(reversed(e2["operative_mask"]["3"]))

    def wrong_lower(r2, e2):
        """The recursion broken at the bottom: L3 built over a lower table that is not the
        one the arm ran. Emulated by relabelling L2's picks so the L2 table permutes."""
        pk = r2["quotient"]["last_build"]["2"]["picks"]
        ks = sorted(pk)
        pk[ks[0]], pk[ks[1]] = pk[ks[1]], pk[ks[0]]

    run("one at-support key dropped at L3", drop_key, "R-1:L3")
    run("a class's picks truncated to one row", truncate_picks, "R-1")
    run("mine_support relabelled 2", flip_support, "R-0")
    have_mask = bool((ent_last or {}).get("operative_mask"))
    if have_mask:
        run("operative_mask reversed at L3 (ORDER only)", reverse_rows, "R-3:L3")
        run("L2's picks permuted (the recursion's floor)", wrong_lower, "R-3")
    else:
        # `entry.json.gz` is not on the volume; R-3's cases cannot be posed there. R-4 carries
        # the closure instead, and its two cases below are posed either way.
        run("L2's picks permuted (the recursion's floor)", wrong_lower, "R-4")
        log("  [--] R-3's own cases need `entry.json.gz`, which this copy does not carry")

    def reverse_keys(r2, e2):
        """The ROW ORDER changed with the row SET intact: `build` emits in `sorted(counts)`
        order, so a replay that read the keys back to front would produce the same table as a
        set and a different one as a sequence."""
        for lv in ("2", "3", "4"):
            r2["log"]["miner"][-1][lv]["keys_at_support"] = list(
                reversed(r2["log"]["miner"][-1][lv]["keys_at_support"]))

    if arm_key is not None:
        # R-4 with the banked mask in hand: it must fire beside R-3.
        run("at-support keys reversed (ORDER, set intact)", reverse_keys, "R-4")
        # ...and WITHOUT it, which is the paid container's own situation: `entry.json.gz` is
        # not on the volume, so R-3 cannot run there and R-4 is the only thing standing
        # between a mis-replayed table and a priced one.
        run("...the same, with `entry.json.gz` ABSENT (the container's case)",
            reverse_keys, "R-4", blind=True)

    # the order-blind twin must SURVIVE the pure-order break, or R-3 proves nothing about order
    if not have_mask:
        return cases
    import copy as _c
    e3 = _c.deepcopy(ent_last)
    e3["operative_mask"]["3"] = list(reversed(e3["operative_mask"]["3"]))
    _, _, gs = reconstruct(_c.deepcopy(res_j), e3, truth=truth)
    r3s = [g for g in gs if g["gate"] == "R-3s:L3"]
    surv = bool(r3s and r3s[0]["pass"])
    log(f"  [{'OK' if surv else 'BAD'}] R-3s survives the pure-order break"
        f"{'' if surv else ' -- R-3 is then only a set gate'}")
    cases.append({"case": "R-3s survives a pure-order break", "targets": "R-3s",
                  "caught": surv, "by": "order-blind by construction"})
    return cases


def main():
    import argparse
    from rhm.rhm_data import generate_rules_distinct
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__),
                                                   "..", "..", "..", "..", "..", ".."))
    ap.add_argument("--arms", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--falsify", action="store_true")
    ap.add_argument("--emit-digests", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    keys = list(ARM_MAP) if (a.all or a.falsify) else ["s0_sv", "s2_sv"]
    if a.arms:
        keys = [k for k in a.arms.split(",") if k]
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figdir, exist_ok=True)
    n_fail = n_missed = 0
    emitted = {}
    for key in keys:
        tag, arm = ARM_MAP[key]
        root = arm_root(repo, tag, arm)
        res_j, ent_last = load_arm(root)
        cfg = res_j["config"]
        rules = generate_rules_distinct(int(cfg["v"]), int(cfg["s"]), int(cfg["depth"]),
                                        int(cfg["m"]), seed=int(cfg["rule_seed"]))
        truth = MC.true_tables(rules, int(cfg["depth"]), int(cfg["s"]), int(cfg["v"]),
                               int(cfg["m"]), max(LEVELS_OK))
        tables, diags, gates = reconstruct(res_j, ent_last, truth=truth,
                                          arm_key=(None if a.emit_digests else key))
        if a.emit_digests:
            r3 = [g for g in gates if g["gate"].startswith("R-3:")]
            assert r3 and all(g["pass"] for g in r3), (
                f"{key}: R-3 did not pass; no digest may be emitted")
            emitted[key] = {str(e): table_digest(tables[e], diags[e]["truth_row"])
                            for e in LEVELS_OK}
        print("=" * 100)
        print(f"{key}  =  {tag}/{arm}   seed {cfg['seed']}  rule_seed {cfg['rule_seed']}")
        for g in gates:
            if g.get("quiet") and g["pass"]:
                continue
            mark = "PASS" if g["pass"] else "FAIL"
            got, want = g["got"], g["want"]
            if isinstance(got, list) and len(got) > 6:
                got, want = f"<{len(got)} rows>", f"<{len(want)} rows>"
            print(f"  [{mark}] {g['gate']:<18} {g['what']:<52} got {got}  want {want}")
            n_fail += int(not g["pass"])
        for ell in LEVELS_OK:
            d = diags[ell]
            gr = d.get("grade", {})
            tr = sum(d.get("truth_row") or [])
            print(f"    L{ell}: {d['n_entries']:>4} rows  {d['n_keys_built']:>3} keys  "
                  f"lower {d['n_lower_rows']:>3}  inv max {d['inventory_max']:>2} "
                  f"mean {d['inventory_mean']:>5.2f}   TRUE {tr:>3} / FALSE "
                  f"{d['n_entries'] - tr:>3}   distinct-flat precision "
                  f"{(gr.get('precision') if gr else float('nan')):.4f}")
        if a.falsify:
            print("  -- falsification --")
            for c in falsify(res_j, ent_last, truth, arm_key=key):
                n_missed += int(not c["caught"])
        if a.write:
            p = os.path.join(figdir, f"{key}_operative_tables.json")
            with open(p, "w") as fh:
                json.dump(tables_payload(res_j, tables, diags), fh, separators=(",", ":"))
            print(f"    wrote {p}")
    if a.emit_digests:
        print("\nDIGESTS = " + json.dumps(emitted, indent=4).replace('": {', '": {'))
    print("=" * 100)
    print(f"gate failures: {n_fail}   falsification misses: {n_missed}")
    return 0 if (n_fail == 0 and n_missed == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
