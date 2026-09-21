"""shaped -- the cross-arm reduction.

Reads, for each of the four trunks (the frozen `traj_a1_s42/step064000.pt` reference and
the three shaped arms), whatever the banked pipelines wrote beside its checkpoint:

  step*_striatum_{a1,swap65k}.{json,npz}   `striatum.task.striatum_ckpt`
  step*_striatum_*_hexcess.npz             `striatum.addendum.hexcess_ckpt`
  step*_junction_a1.{json,npz}             `junction.task.junction_ckpt`
  step*_calibration.{json,npz} + altitude_identity.json   Part 1 / `altitude.identity`
  step*_probe_a1.json                      `shaped.shape.probe_ckpt` (Part 2a + clean CE)
  shape_log.json                           `shaped.shape.shape_ckpt`

and writes one `results/tables.md` with every number side by side across the arms, plus
the figures.  The per-arm tables the banked analyzers already write are generated
unchanged by running those analyzers on each arm's directory (see the reproduction header
in the output); this file only does the cross-arm join and the two reductions the banked
analyzers cannot do on a shaped trunk:

  * `addendum_hx` -- striatum's §5 addendum with the model's REALISED horizon excess in
    the role of the surprise-shaped readout.  The banked `coeruleus/` head is a readout
    fitted to the FROZEN trunk's states; it does not transfer to a trunk whose states have
    moved, and no head was refitted per arm, so the arm-comparable surprise score is the
    realised horizon excess -- which striatum's own addendum found reads damage BETTER
    than the head does (0.55-0.61 against 0.50-0.57).  The banked column is still printed
    for the frozen reference.
  * `legality_matched` -- junction's matched legality contrast (exact strata on
    (j, anchor-position bucket), random 1:1, the model's surprisal left in) computed from
    the *striatum* per-episode npz, so every arm has it whether or not a junction cell was
    run, with striatum's own oracle legality probe on the identical rows as the ceiling.

Run (local, after fetching the volume artefacts):
  python -m rhm.logit_reading.orbitofrontal.shaped.analyze --root <local mirror of
      /data/v16_s2_L6_m4_distinct/logit_reading>
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import (fmt, tbl, rematch, _auc, _matched_rows,
                                                _cond_auc, _oof_auc, LEVELS, A_LIST)
from rhm.logit_reading.striatum.junction.analyze import match_random, Rows, order_diets

# arm key -> (volume sub-directory, step)
ARMS = [("frozen", "traj_a1_s42", 64000),
        ("task", "shape_task_s42", 3000),
        ("task_ntp", "shape_task_ntp_s42", 3000),
        ("ntp", "shape_ntp_s42", 3000)]

ARM_NOTE = {
    "frozen": "the banked next-token trunk every value-side round has read",
    "task": "query loss alone; the next-token head left alone",
    "task_ntp": "query loss + next-token loss (w_ntp = 1)",
    "ntp": "next-token loss alone for the same steps on the same windows (the control)",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class Arm:
    def __init__(self, root, name, sub, step):
        self.name, self.sub, self.step = name, sub, step
        self.dir = os.path.join(root, sub)
        self.stem = os.path.join(self.dir, f"step{step:06d}")
        self.striatum = {t: self._json(f"{self.stem}_striatum_{t}.json")
                         for t in ("a1", "swap65k")}
        self.npz = {t: f"{self.stem}_striatum_{t}.npz" for t in ("a1", "swap65k")}
        self.hexcess = {t: f"{self.stem}_striatum_{t}_hexcess.npz"
                        for t in ("a1", "swap65k")}
        self.junction = self._json(f"{self.stem}_junction_a1.json")
        self.jnpz = f"{self.stem}_junction_a1.npz"
        self.probe = self._json(f"{self.stem}_probe_a1.json")
        self.shape_log = self._json(os.path.join(self.dir, "shape_log.json"))
        self.identity = self._json(os.path.join(self.dir, "altitude_identity.json"))
        cs = self._json(os.path.join(self.dir, "calibration_sweep.json"))
        self.calib = {r["step"]: r for r in cs} if cs else {}
        if not self.calib:                       # the frozen arm: per-checkpoint files
            for f in glob.glob(os.path.join(self.dir, "step*_calibration.json")):
                r = self._json(f)
                if r:
                    self.calib[int(os.path.basename(f)[4:10])] = r
        self._rm = {}

    @staticmethod
    def _json(p):
        return json.load(open(p)) if os.path.exists(p) else None

    def cb(self, tag="swap65k"):
        r = self.striatum.get(tag)
        return r["critic_block"] if r else None

    def rm(self, tag="swap65k"):
        """striatum's local re-match: strata (k*, j, position), sign-balanced surprisal."""
        if tag not in self._rm:
            self._rm[tag] = (rematch(self.npz[tag], self.cb(tag))
                             if os.path.exists(self.npz[tag]) else {})
        return self._rm[tag]

    def ident(self, step=None):
        if not self.identity:
            return None
        step = self.step if step is None else step
        for r in self.identity:
            if r.get("step") == step:
                return r
        return None


def have(a, tag="swap65k"):
    return a.striatum.get(tag) is not None and os.path.exists(a.npz[tag])


# ---------------------------------------------------------------------------
# the reductions the banked analyzers cannot do on a shaped trunk
# ---------------------------------------------------------------------------

def addendum_hx(npz_path, cb, hexcess_path=None, anchor="tv", label="flip", a=0, lmax=4,
                nbin=5, seed=0):
    """striatum's addendum with `excess_horizon` in the banked head's place.

    Identical matched rows (exact (k*, j, position bucket) strata, sign-balanced
    nearest-neighbour pairs on the model's surprisal at the anchor) and identical
    conditional/out-of-fold machinery; the surprise-shaped readout is the model's realised
    horizon excess `sum_{u=t0}^{t0+8}(NLL_u - H(q_u))` rather than a head's prediction of
    it.  The banked head's columns are added where the file exists."""
    Z = np.load(npz_path)
    pre = f"{anchor}__"
    if pre + "t0" not in Z:
        return {}
    HX = np.load(hexcess_path) if hexcess_path and os.path.exists(hexcess_path) else None
    out = {}
    for l in [x for x in LEVELS if x <= lmax]:
        got = _matched_rows(Z, pre, l, a, label, seed=seed)
        if got is None:
            continue
        mm, lab = got
        g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
        w, t0 = g("w"), g("t0")
        sc = {"R": -g(f"R_{cb}_l{l}_a{a}"), "R_clean": -g(f"R_clean_l{l}_a{a}"),
              "V": -g(f"V_l{l}_a{a}"), "excess_1pos": g("excess_e"),
              "banked": g("banked_excess")}
        if g(f"R_mlp_l{l}_a{a}") is not None:
            sc["R_mlp"] = -g(f"R_mlp_l{l}_a{a}")
        if HX is not None:
            he = HX["hexcess_edit"]
            sc["excess_horizon"] = np.nan_to_num(he[w, np.clip(t0, 0, he.shape[1] - 1)],
                                                 nan=0.0)
        sc = {k: np.asarray(v)[mm] for k, v in sc.items() if v is not None}
        y = lab[mm]
        e = {"n": int(len(mm)), "base_rate": float(y.mean()),
             "marginal": {k: _auc(v, y) for k, v in sc.items()}}
        for surp in ("excess_horizon", "banked"):
            if surp not in sc:
                continue
            for tgt in ("R", "V", "R_clean", "R_mlp"):
                if tgt not in sc:
                    continue
                e[f"{tgt}|{surp}"] = _cond_auc(sc[tgt], sc[surp], y, nbin)
                e[f"{surp}|{tgt}"] = _cond_auc(sc[surp], sc[tgt], y, nbin)
                e[f"oof[{surp},{tgt}]"] = _oof_auc(np.stack([sc[surp], sc[tgt]], 1), y,
                                                   seed=seed)
            e[f"oof[{surp}]"] = _oof_auc(sc[surp][:, None], y, seed=seed)
        for tgt in ("R", "V", "R_clean", "R_mlp"):
            if tgt in sc:
                e[f"oof[{tgt}]"] = _oof_auc(sc[tgt][:, None], y, seed=seed)
        out[f"l{l}"] = e
    return out


def legality_matched(npz_path, cb, anchor="fd", a=0, tbucket=4, seed=0, lmax=3):
    """junction's Q1a contrast, computed from a striatum per-episode npz.

    Swap vs rare among CONSEQUENTIAL episodes, exact strata on (j, anchor-position
    bucket), random 1:1 inside each cell (nearest-neighbour matching on surprisal
    collapses the cell -- junction's `match_random`).  `j` and the position bucket are
    pinned by construction; the model's own surprisal is left in as the input-level cue a
    state readout could use, and reported as a reference row rather than a guard."""
    Z = np.load(npz_path)
    pre = f"{anchor}__"
    if pre + "t0" not in Z:
        return {}
    g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
    et, jj, t0, nll = g("etype"), g("j"), g("t0"), g("nll_e")
    keys = jj * 1000 + (t0 // tbucket)
    out = {}
    for l in [x for x in LEVELS if x <= lmax]:
        cons = g(f"cons_l{l}_a{a}")
        if cons is None:
            continue
        cons = cons.astype(bool)
        ok = g(f"ok_l{l}_a{a}").astype(bool)
        base = ok & (et < 2) & cons
        lab = et == 0
        mi = np.where(base)[0]
        if len(mi) < 50 or not 0 < lab[mi].mean() < 1:
            continue
        sel = match_random(keys[mi], lab[mi], np.random.default_rng(seed))
        if len(sel) < 40:
            continue
        mm = mi[sel]
        sc = {"R": -g(f"R_{cb}_l{l}_a{a}"), "V": -g(f"V_l{l}_a{a}"),
              "R_clean": -g(f"R_clean_l{l}_a{a}"), "Rshuf": -g(f"Rsh_l{l}_a{a}"),
              "probe_illegal": g("probe_illegal"), "nll": nll,
              "excess_1pos": g("excess_e"), "banked_excess": g("banked_excess")}
        if g(f"R_mlp_l{l}_a{a}") is not None:
            sc["R_mlp"] = -g(f"R_mlp_l{l}_a{a}")
        e = {"n": int(len(mm)), "n_raw": int(base.sum()),
             "guard_j": _auc(jj.astype(float)[mm], lab[mm]),
             "guard_t0": _auc(t0.astype(float)[mm], lab[mm])}
        for k, v in sc.items():
            if v is not None:
                e[k] = _auc(np.asarray(v)[mm], lab[mm])
                e[f"raw_{k}"] = _auc(np.asarray(v)[base], lab[base])
        out[f"l{l}"] = e
    return out


# ---------------------------------------------------------------------------
# cross-arm tables
# ---------------------------------------------------------------------------

def sec_glance(arms):
    """The four rows the round turns on, every arm side by side."""
    rows = []
    for arm in arms:
        tv = arm.rm("swap65k").get("tv", {})
        lm = (legality_matched(arm.npz["a1"], arm.cb("a1")) if have(arm, "a1") else {})
        st = arm.striatum.get("swap65k")
        pb = st["primary_block"] if st else None
        g = lambda lab, key, l: tv.get(f"{lab}/l{l}_a0", {}).get(key)
        rows.append([arm.name,
                     fmt(st["actor_clean_acc"][f"{pb}/l4"]) if st else "--",
                     fmt(st["actor_clean_acc"][f"{pb}/l6"]) if st else "--",
                     fmt(g("flip", "R", 2)), fmt(g("flip", "V", 2)),
                     fmt(g("cons", "R", 2)), fmt(g("cons", "probe_cons", 2)),
                     fmt(g("cons", "R", 4)), fmt(g("cons", "probe_cons", 4)),
                     fmt(lm.get("l2", {}).get("R")),
                     fmt(lm.get("l2", {}).get("probe_illegal"))])
    return tbl(rows, ["arm", "actor l=4", "actor l=6", "damage R", "damage V",
                      "cons R", "cons oracle", "cons R (l4)", "cons oracle (l4)",
                      "legality R", "legality oracle"])


def sec_shaping(arms):
    rows = []
    for a in arms:
        sl = a.shape_log
        if not sl:
            rows.append([a.name, "--", "--", "--", "--"] + ["--"] * 6)
            continue
        log = sl["log"]
        pre, fin = log[0], [r for r in log if r["step"] <= a.step][-1]
        rows.append([a.name, sl["config"]["arm"], fin["step"],
                     fmt(pre["ntp_ce"], 4), fmt(fin["ntp_ce"], 4)]
                    + [fmt(x) for x in fin["head_acc"]])
    return tbl(rows, ["arm", "objective", "last logged step", "ntp CE before",
                      "ntp CE after"]
               + [f"head l={l}" for l in LEVELS])


def sec_ntp_traj(arms):
    steps = sorted({r["step"] for a in arms if a.shape_log for r in a.shape_log["log"]})
    rows = []
    for a in arms:
        if not a.shape_log:
            continue
        m = {r["step"]: r["ntp_ce"] for r in a.shape_log["log"]}
        rows.append([a.name] + [fmt(m.get(s), 4) if s in m else "--" for s in steps])
    return tbl(rows, ["arm"] + [str(s) for s in steps])


def sec_altitude(arms):
    rows = []
    for a in arms:
        steps = ([a.step] if (not a.identity or a.name == "frozen") else
                 sorted({r["step"] for r in a.identity}))
        for st in steps:
            r = a.ident(st)
            if not r:
                continue
            al, idn = r["altitude"], r["identity"]
            ce = (a.calib.get(st, {}).get("overall", {}) or {}).get("CE")
            rows.append([a.name, st, fmt(ce, 4),
                         fmt(idn["mean_gap"], 4), fmt(r["alpha_star_overall"], 3),
                         al["discrete"]["best_k"], fmt(al["convex"]["kappa_star"], 2),
                         fmt(al["convex"]["KL_at_star"], 4),
                         fmt(al["fit_quality"]["KL_pL_q"], 4),
                         fmt(al["convex"]["KL_at_star"] / max(al["fit_quality"]["KL_pL_q"],
                                                              1e-9), 2)])
    return tbl(rows, ["arm", "step", "CE", "CE-H(q)", "alpha*", "best_k", "kappa*",
                      "KL(p_k*||q)", "KL(p_L||q)", "ratio"])


def sec_detection(arms):
    rows = []
    for a in arms:
        if not a.probe:
            continue
        det = a.probe["detection_by_kstar"]
        rows.append([a.name, fmt(a.probe["clean_ntp_ce"], 4)]
                    + [fmt(det.get(str(k), {}).get("detect_auc_stratified"))
                       for k in range(1, 6)]
                    + [fmt(det.get(str(k), {}).get("mean_surprisal"), 2)
                       for k in (1, 5)])
    return tbl(rows, ["arm", "clean NTP CE"] + [f"k*={k}" for k in range(1, 6)]
               + ["surpr k*=1", "surpr k*=5"])


def sec_actor(arms, tag="swap65k"):
    rows = []
    for a in arms:
        r = a.striatum.get(tag)
        if not r:
            continue
        acc, pb = r["actor_clean_acc"], r["primary_block"]
        rows.append([a.name, pb] + [fmt(acc.get(f"{pb}/l{l}")) for l in LEVELS]
                    + [fmt(acc.get("post_embed/l1")), fmt(acc.get("post_embed/l3"))])
    return tbl(rows, ["arm", "block"] + [f"l={l}" for l in LEVELS] + ["emb l1", "emb l3"])


def sec_critic(arms, tag="swap65k"):
    rows = []
    for a in arms:
        r = a.striatum.get(tag)
        if not r:
            continue
        cb, v = r["critic_block"], r["critic_val_r2"]
        s, cl = r["critic_val_r2_shuf"], r.get("clean_critic_val_r2", {})
        rows.append([a.name, cb] + [fmt(v.get(f"{cb}/l{l}_d0"), 4) for l in LEVELS]
                    + [fmt(cl.get("l1_d0"), 4), fmt(v.get("post_embed/l1_d0"), 4),
                       fmt(s.get(f"{cb}/sh_l1_d0"), 4)])
    return tbl(rows, ["arm", "block"] + [f"V l{l} d0" for l in LEVELS]
               + ["clean l1", "emb l1", "shuf l1"])


SCORE_ROWS = [("R", "critic revision R, linear"), ("R_mlp", "critic revision, MLP"),
              ("R_clean", "clean-only critic's revision"),
              ("Rshuf", "shuffled-outcome floor"), ("V", "the critic's level V"),
              ("nll", "model surprisal (guard)"), ("dH", "dH"),
              ("excess", "single-position excess"),
              ("banked_excess", "banked coeruleus excess head"),
              ("probe_cons", "oracle consequence probe"),
              ("probe_illegal", "oracle legality probe")]


def sec_rm(arms, tag, an, lname, a=0, lmax=4):
    """One re-matched table, arms as row groups."""
    rows = []
    for arm in arms:
        e = arm.rm(tag).get(an, {})
        if not e:
            continue
        rows.append([arm.name, "n matched"]
                    + [e.get(f"{lname}/l{l}_a{a}", {}).get("n", "--")
                       for l in LEVELS if l <= lmax])
        for key, nice in SCORE_ROWS:
            vals = [e.get(f"{lname}/l{l}_a{a}", {}).get(key) for l in LEVELS if l <= lmax]
            if all(v is None for v in vals):
                continue
            rows.append([arm.name, nice] + [fmt(v) for v in vals])
        for gk in ("guard_nll", "guard_t0", "guard_kstar", "guard_j"):
            vals = [e.get(f"{lname}/l{l}_a{a}", {}).get(gk) for l in LEVELS if l <= lmax]
            if all(v is None for v in vals):
                continue
            rows.append([arm.name, gk] + [fmt(v) for v in vals])
    return tbl(rows, ["arm", "score"] + [f"l={l}" for l in LEVELS if l <= lmax])


def sec_headline_cross(arms, tag, an, lname, keys, a=0, lmax=4):
    """The compact cross-arm view: one row per (arm, score)."""
    rows = []
    for key, nice in keys:
        for arm in arms:
            e = arm.rm(tag).get(an, {})
            vals = [e.get(f"{lname}/l{l}_a{a}", {}).get(key) for l in LEVELS if l <= lmax]
            if all(v is None for v in vals):
                continue
            rows.append([nice, arm.name] + [fmt(v) for v in vals])
    return tbl(rows, ["score", "arm"] + [f"l={l}" for l in LEVELS if l <= lmax])


def sec_legality(arms, tag="a1", an="fd"):
    rows = []
    for arm in arms:
        if not have(arm, tag):
            continue
        lm = legality_matched(arm.npz[tag], arm.cb(tag), anchor=an)
        if not lm:
            continue
        ls = sorted(lm, key=lambda k: int(k[1:]))
        rows.append([arm.name, "n matched / raw"]
                    + [f"{lm[k]['n']} / {lm[k]['n_raw']}" for k in ls])
        for key, nice in (("R", "value revision R (the reading)"),
                          ("V", "the critic's level V"),
                          ("R_clean", "clean-only critic's revision"),
                          ("Rshuf", "shuffled-outcome floor"),
                          ("probe_illegal", "CEILING: oracle legality probe"),
                          ("nll", "reference: model surprisal"),
                          ("excess_1pos", "reference: single-position excess"),
                          ("banked_excess", "reference: banked excess head"),
                          ("guard_j", "guard: j"), ("guard_t0", "guard: position")):
            vals = [lm[k].get(key) for k in ls]
            if all(v is None for v in vals):
                continue
            rows.append([arm.name, nice] + [fmt(v) for v in vals])
        rows.append([arm.name, "headroom (probe - 0.5)"]
                    + [fmt(abs(lm[k]["probe_illegal"] - 0.5)) for k in ls])
        rows.append([arm.name, "used (|R - 0.5|)"]
                    + [fmt(abs(lm[k]["R"] - 0.5)) for k in ls])
    return tbl(rows, ["arm", "row"] + [f"l={l}" for l in (1, 2, 3)])


def sec_across(arms, tag, an, key="across_level", a_list=(0, 1)):
    rows = []
    for arm in arms:
        r = arm.striatum.get(tag)
        if not r:
            continue
        t = r["tables"].get(an, {}).get(key, {})
        for a in a_list:
            e = t.get(f"a{a}")
            if e:
                rows.append([arm.name, a, e["n"], fmt(e["mean_rank_cons"]),
                             fmt(e["mean_rank_incons"]), fmt(e["win_rate"])])
    return tbl(rows, ["arm", "a", "n", "rank cons", "rank incons", "win rate"])


ADD_ROWS = ["marginal:R", "marginal:R_mlp", "marginal:R_clean", "marginal:V",
            "marginal:excess_1pos", "marginal:excess_horizon", "marginal:banked",
            "V|excess_horizon", "excess_horizon|V", "R|excess_horizon",
            "excess_horizon|R", "R_clean|excess_horizon", "excess_horizon|R_clean",
            "oof[excess_horizon]", "oof[V]", "oof[excess_horizon,V]",
            "oof[R]", "oof[excess_horizon,R]",
            "V|banked", "banked|V", "oof[banked]", "oof[banked,V]"]


def sec_addendum(arms, tag="swap65k", an="tv", label="flip"):
    rows = []
    for arm in arms:
        if not have(arm, tag):
            continue
        add = addendum_hx(arm.npz[tag], arm.cb(tag), arm.hexcess[tag], anchor=an,
                          label=label)
        if not add:
            continue
        ks = sorted(add, key=lambda k: int(k[1:]))
        rows.append([arm.name, "n matched"] + [add[k]["n"] for k in ks])
        rows.append([arm.name, "base rate"] + [fmt(add[k]["base_rate"]) for k in ks])
        for nm in ADD_ROWS:
            if nm.startswith("marginal:"):
                f_ = nm.split(":")[1]
                vals = [add[k]["marginal"].get(f_) for k in ks]
            else:
                vals = [add[k].get(nm) for k in ks]
            if all(v is None for v in vals):
                continue
            rows.append([arm.name, nm] + [fmt(v) for v in vals])
    return tbl(rows, ["arm", "quantity"] + [f"l={l}" for l in (1, 2, 3, 4)])


# ---------------------------------------------------------------------------
# junction: the diet dimension, per arm
# ---------------------------------------------------------------------------

def sec_junction(arms, a=0):
    """Q1's matched legality reading per diet, per arm, with the span and the ceiling."""
    rows = []
    for arm in arms:
        r, npz = arm.junction, arm.jnpz
        if not r or not os.path.exists(npz):
            continue
        R = Rows(npz, "fd")
        if not R.ok:
            continue
        wd0 = [k for k in r["diet_spec"] if r["diet_spec"][k].get("kind") == "window"]
        wd = (["clean"] if r.get("clean_critic_val_r2") else []) + wd0
        per = {}
        for nm in order_diets(wd):
            vals = []
            for l in (1, 2, 3):
                got = R.matched("legal", l, a, "jt_rand")
                sc = R.scores(nm, l, a)
                vals.append(_auc(sc["R"][got[0]], got[1][got[0]])
                            if got and "R" in sc else None)
            per[nm] = vals
            rows.append([arm.name, nm,
                         fmt(r["diet_stats"].get(nm, {}).get("assoc"), 3),
                         fmt(r["diet_stats"].get(nm, {}).get("c_gap"), 2)]
                        + [fmt(v) for v in vals])
        for i, lbl in enumerate(("span l=1", "span l=2", "span l=3")):
            vv = [per[nm][i] for nm in per if per[nm][i] is not None]
            if vv:
                rows.append([arm.name, lbl, "", ""] + [""] * i
                            + [f"{min(vv):.3f}-{max(vv):.3f}"] + [""] * (2 - i))
        for key, nice in (("probe_swap_ed", "CEILING oracle legality probe, linear"),
                          ("probe_swap_mlp", "CEILING oracle legality probe, MLP"),
                          ("nll", "reference model surprisal"),
                          ("hexcess", "reference realised horizon excess")):
            vals = []
            for l in (1, 2, 3):
                got = R.matched("legal", l, a, "jt_rand")
                rf = R.refs(l, a)
                vals.append(_auc(rf[key][got[0]], got[1][got[0]])
                            if got and key in rf else None)
            if any(v is not None for v in vals):
                rows.append([arm.name, nice, "", ""] + [fmt(v) for v in vals])
    return tbl(rows, ["arm", "diet", "assoc", "c gap", "l=1", "l=2", "l=3"])


def sec_junction_control(arms, a=0):
    """The positive control: unconditional swap-vs-rare on V, over all edited rows."""
    rows = []
    for arm in arms:
        r, npz = arm.junction, arm.jnpz
        if not r or not os.path.exists(npz):
            continue
        R = Rows(npz, "fd")
        if not R.ok:
            continue
        wd0 = [k for k in r["diet_spec"] if r["diet_spec"][k].get("kind") == "window"]
        ed, sw = R.et < 2, R.et == 0
        cd = R.g("c_depth")
        for nm in order_diets(wd0):
            v1 = [R.scores(nm, l, a).get("V") for l in (1, 2, 3)]
            rows.append([arm.name, nm, fmt(r["diet_stats"].get(nm, {}).get("c_gap"), 2)]
                        + [fmt(_auc(v[ed], sw[ed])) if v is not None else "--" for v in v1]
                        + [fmt(_auc(v1[0][ed], (cd >= 2)[ed])) if v1[0] is not None
                           else "--"])
    return tbl(rows, ["arm", "diet", "c gap", "V:swap-vs-rare l1", "l2", "l3",
                      "V:consequence depth>=2 l1"])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

COL = {"frozen": "#444444", "task": "#c0392b", "task_ntp": "#1f77b4", "ntp": "#2ca02c"}


def figures(arms, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    # --- 1. what the shaping did -------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a in arms:
        if not a.shape_log:
            continue
        log = [r for r in a.shape_log["log"]]
        st = [r["step"] for r in log]
        ax[0].plot(st, [r["ntp_ce"] for r in log], "-o", ms=3, color=COL[a.name],
                   label=a.name)
        for li, l in enumerate(LEVELS):
            ax[1].plot(st, [r["head_acc"][li] for r in log], "-", color=COL[a.name],
                       alpha=0.3 + 0.1 * li, lw=1 + 0.3 * li)
    fr = [x for x in arms if x.name == "frozen"]
    if fr and fr[0].striatum.get("swap65k"):
        r = fr[0].striatum["swap65k"]
        pb = r["primary_block"]
        for l in LEVELS:
            b = r["actor_clean_acc"][f"{pb}/l{l}"]
            ax[1].plot([0, max(st)], [b, b], ":", color="k", lw=0.8)
    ax[0].set_xlabel("shaping step"); ax[0].set_ylabel("next-token CE (held-out clean)")
    ax[0].set_title("what shaping costs the predictive"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("shaping step"); ax[1].set_ylabel("query accuracy (shaping-time head)")
    ax[1].set_title("the goal the trunk is being shaped toward\n(dotted: frozen trunk's refit actor)")
    # refit actor accuracy per arm
    w = 0.2
    xs = np.arange(len(LEVELS))
    for i, a in enumerate(arms):
        r = a.striatum.get("swap65k")
        if not r:
            continue
        pb = r["primary_block"]
        ax[2].bar(xs + (i - 1.5) * w, [r["actor_clean_acc"][f"{pb}/l{l}"] for l in LEVELS],
                  w, color=COL[a.name], label=a.name)
    ax[2].set_xticks(xs); ax[2].set_xticklabels([f"l={l}" for l in LEVELS])
    ax[2].set_ylabel("clean held-out accuracy"); ax[2].legend(fontsize=8)
    ax[2].set_title("the refit actor on each trunk")
    fig.tight_layout(); fig.savefig(f"{outdir}/shaping.png", dpi=140); plt.close(fig)

    # --- 2. the headline: cost vs structure, and the legality headroom ------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ls = [1, 2, 3, 4]
    xs = np.arange(len(ls))
    for i, a in enumerate(arms):
        e = a.rm("swap65k").get("tv", {})
        if not e:
            continue
        ax[0].bar(xs + (i - 1.5) * w,
                  [e.get(f"flip/l{l}_a0", {}).get("R", np.nan) for l in ls], w,
                  color=COL[a.name], label=a.name)
        ax[1].bar(xs + (i - 1.5) * w,
                  [e.get(f"cons/l{l}_a0", {}).get("R", np.nan) for l in ls], w,
                  color=COL[a.name])
        ax[1].plot(xs + (i - 1.5) * w,
                   [e.get(f"cons/l{l}_a0", {}).get("probe_cons", np.nan) for l in ls],
                   "k_", ms=10, mew=2)
    for k in (0, 1):
        ax[k].axhline(0.5, color="k", lw=0.8, ls="--")
        ax[k].set_xticks(xs); ax[k].set_xticklabels([f"l={l}" for l in ls])
        ax[k].set_ylim(0.40, 0.75)
    ax[0].set_ylabel("AUC"); ax[0].set_title("realised damage at t_v (matched)")
    ax[0].legend(fontsize=8)
    ax[1].set_title("structural consequence at t_v (matched)\n(black bars: oracle probe)")
    # legality at matched consequence, a1 onset
    xs3 = np.arange(3)
    for i, a in enumerate(arms):
        if not have(a, "a1"):
            continue
        lm = legality_matched(a.npz["a1"], a.cb("a1"))
        if not lm:
            continue
        ax[2].bar(xs3 + (i - 1.5) * w,
                  [lm.get(f"l{l}", {}).get("R", np.nan) for l in (1, 2, 3)], w,
                  color=COL[a.name])
        ax[2].plot(xs3 + (i - 1.5) * w,
                   [lm.get(f"l{l}", {}).get("probe_illegal", np.nan) for l in (1, 2, 3)],
                   "k_", ms=10, mew=2)
    ax[2].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[2].set_ylim(0.40, 0.62)
    ax[2].set_xticks(xs3); ax[2].set_xticklabels([f"l={l}" for l in (1, 2, 3)])
    ax[2].set_title("legality at matched consequence (a1, onset)\n(black bars: oracle legality probe)")
    fig.tight_layout(); fig.savefig(f"{outdir}/headline.png", dpi=140); plt.close(fig)

    # --- 3. the structural cost -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for a in arms:
        if not a.probe:
            continue
        det = a.probe["detection_by_kstar"]
        ks = [k for k in range(1, 6) if str(k) in det]
        ax[0].plot(ks, [det[str(k)]["detect_auc_stratified"] for k in ks], "-o", ms=4,
                   color=COL[a.name], label=a.name)
    ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[0].set_xlabel("k* (levels you must know to be offended)")
    ax[0].set_ylabel("detection AUC"); ax[0].set_title("Part 2a on each trunk")
    ax[0].legend(fontsize=8)
    names, kap, ce = [], [], []
    for a in arms:
        r = a.ident()
        if not r:
            continue
        names.append(a.name)
        kap.append(r["altitude"]["convex"]["kappa_star"])
        ce.append((a.calib.get(a.step, {}).get("overall", {}) or {}).get("CE"))
    if names:
        ax[1].bar(np.arange(len(names)), kap, 0.5,
                  color=[COL[n] for n in names])
        ax[1].set_xticks(np.arange(len(names))); ax[1].set_xticklabels(names)
        ax[1].set_ylabel("kappa* (altitude)"); ax[1].set_title("Part 1 altitude on each trunk")
        for i, (k_, c_) in enumerate(zip(kap, ce)):
            ax[1].text(i, k_ + 0.05, f"CE {c_:.3f}" if c_ else "", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/structural.png", dpi=140); plt.close(fig)
    print("figures ->", outdir)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="rhm/logit_reading/orbitofrontal/shaped/results/data")
    ap.add_argument("--out", default="rhm/logit_reading/orbitofrontal/shaped/results")
    ap.add_argument("--figs", default="rhm/logit_reading/orbitofrontal/shaped/figs")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = [Arm(args.root, n, s, st) for n, s, st in ARMS]
    arms = [a for a in arms if have(a, "a1") or have(a, "swap65k") or a.shape_log]

    md = ["# shaped -- tables", "",
          "Facts only; the interpretation waits on a discussion.  Regenerated by",
          "`python -m rhm.logit_reading.orbitofrontal.shaped.analyze --root <local mirror>`.", "",
          "## Reproduction", "", "```bash", "cd experiments",
          "D=/data/v16_s2_L6_m4_distinct/logit_reading", "",
          "# 1. the fine-tune: three arms, matched steps and matched data, one container",
          "#    each (~2 min per arm on an L4, peak RSS 5.0 GB)",
          "modal run --detach -m rhm.logit_reading.orbitofrontal.shaped.shape::shape_sweep \\",
          "    --steps 3000 --ckpt-steps 1000,3000 --tag s42", "",
          "# 2. the value-side battery, identical rows, on every shaped trunk",
          "modal run --detach -m rhm.logit_reading.striatum.task::striatum_sweep \\",
          "    --ckpts \"$D/shape_task_s42/step003000.pt,"
          "$D/shape_task_ntp_s42/step003000.pt,$D/shape_ntp_s42/step003000.pt\" \\",
          "    --stim-tags a1,swap65k --n-clean 6144",
          "modal run --detach -m rhm.logit_reading.striatum.addendum::hexcess_sweep \\",
          "    --cells \"$D/shape_task_s42/step003000.pt:a1,"
          "$D/shape_task_s42/step003000.pt:swap65k,...\"   # 3 arms x 2 venues",
          "modal run --detach -m rhm.logit_reading.striatum.junction.task::junction_sweep \\",
          "    --cells \"$D/shape_task_s42/step003000.pt:a1:1:0:0,"
          "$D/shape_task_ntp_s42/step003000.pt:a1:1:0:0,"
          "$D/shape_ntp_s42/step003000.pt:a1:1:0:0\"   # window diets only", "",
          "# 3. the structural side: Part 1's altitude and Part 2a, on every arm",
          "for a in task task_ntp ntp; do",
          "  modal run --detach -m rhm.logit_reading.calibration::calibration_sweep \\",
          "      --traj-dir $D/shape_${a}_s42",
          "  modal run --detach -m rhm.logit_reading.altitude.identity::identity_sweep \\",
          "      --traj-dir $D/shape_${a}_s42        # after the calibration arrays land",
          "done",
          "modal run --detach -m rhm.logit_reading.orbitofrontal.shaped.shape::probe_sweep \\",
          "    --ckpts \"$D/traj_a1_s42/step064000.pt,"
          "$D/shape_task_s42/step003000.pt,...\"      # the frozen reference too", "",
          "# 4. the tables and figures (local)",
          "bash rhm/logit_reading/orbitofrontal/shaped/results/fetch.sh  <local-mirror>",
          "bash rhm/logit_reading/orbitofrontal/shaped/results/reduce.sh <local-mirror>",
          "```", "",
          "The frozen reference is read from the BANKED artefacts beside "
          "`traj_a1_s42/step064000.pt` -- the same code path, already run -- so no cell is "
          "re-run for it.  `results/per_arm/` holds each shaped arm's tables in the banked "
          "striatum and junction formats, written by those analyzers unchanged.", "",
          "## The arms", "",
          tbl([[n, ARM_NOTE[n], sub, st] for n, sub, st in ARMS],
              ["arm", "objective", "volume dir", "step"]), "",
          "All three shaped arms start from `traj_a1_s42/step064000.pt`, run the same "
          "number of steps on bit-identical windows (the data stream is a function of "
          "`data_seed` alone) and share every optimiser setting.  The query heads are "
          "trained on DETACHED states in the `ntp` arm, so that arm's trunk moves under "
          "next-token loss only.", "",
          "## At a glance", "",
          "`swap65k` at t_v for the damage and consequence columns (re-matched on "
          "(k*, j, position) with sign-balanced surprisal pairs), level 2 unless the "
          "header says otherwise; `a1` at the onset for the legality columns (swap vs "
          "rare among consequential episodes, strata (j, position), random 1:1).  "
          "`oracle` is a ridge probe of the SAME anchor states for the same label.", "",
          sec_glance(arms), "",
          "## 0. What the shaping did", "", sec_shaping(arms), "",
          "`head l=*` is the SHAPING-TIME head's held-out accuracy -- a weaker readout "
          "than the refit actor below (it is trained against a moving trunk), so it is "
          "the within-arm trajectory that is comparable, not its level against the "
          "banked actor.", "",
          "### Held-out clean next-token CE through the fine-tune", "",
          sec_ntp_traj(arms), "",
          "## 1. The structural side: what shaping cost the predictive", "",
          "### Part 1 -- the altitude fit (`altitude/` Q1, CPU over `calibration_ckpt`'s arrays)",
          "", sec_altitude(arms), "",
          "### Part 2a -- per-level violation detection, and the clean next-token CE", "",
          "Detection AUC is the violating token's model-surprisal percentile among legal "
          "control tokens in the same (token level x window-position bucket) cell, on "
          "`stimuli_a1` -- the parent's `detect_auc_stratified`.", "",
          sec_detection(arms), "",
          "## 2. The actor and the critic on each trunk", "",
          "### The refit actor: clean held-out accuracy by query level (chance 0.0625)",
          "", sec_actor(arms), "",
          "### The critic: held-out R^2 of V[l, d=0] on the realised outcome", "",
          sec_critic(arms), ""]

    for tag, an, anm in (("swap65k", "tv", "t_v (the Bayesian-detectable violation)"),
                         ("a1", "fd", "the edit onset (first_diff)")):
        md += [f"## 3. Venue `{tag}`, anchor {anm} -- striatum's headline rows", "",
               "Re-matched locally: exact strata on (k*, j, anchor-position bucket), "
               "sign-balanced nearest-neighbour pairs on the model's surprisal at the "
               "anchor (`striatum.analyze.rematch`, the banked recipe).", "",
               "### AUC(realised damage at the goal: right on orig, wrong on edit), a = 0",
               "", sec_rm(arms, tag, an, "flip"), "",
               "### AUC(structural consequence: the true answer changed), a = 0", "",
               sec_rm(arms, tag, an, "cons"), ""]

    md += ["## 4. Legality at matched consequence -- the direct test", "",
           "Swap vs rare among CONSEQUENTIAL episodes at the edit onset on `a1`, exact "
           "strata on (j, anchor-position bucket), random 1:1 inside each cell "
           "(junction's `match_random`; nearest-neighbour matching on surprisal collapses "
           "the cell).  `j` and position are pinned by construction; the model's own "
           "surprisal is left in and reported as a reference.  The oracle legality probe "
           "is a ridge readout of the SAME anchor states for the same label, fitted on "
           "the train split -- the headroom a state readout has.  (This is striatum's "
           "`probe_illegal`, a linear ridge on the anchor state.  junction's ceiling in "
           "section 7 is its own `probe_swap_ed` / `probe_swap_mlp`, which reads a little "
           "higher; both are reported and neither is moved by anything below it.)", "",
           sec_legality(arms), "",
           "## 5. The across-level contrast (identical state, different query level)", "",
           "### Trained critic, `swap65k` at t_v", "",
           sec_across(arms, "swap65k", "tv"), "",
           "### Clean-only critic (the exposure floor), `swap65k` at t_v", "",
           sec_across(arms, "swap65k", "tv", "across_level_clean"), "",
           "### Trained critic, `a1` at the onset", "", sec_across(arms, "a1", "fd"), "",
           "## 6. Addendum -- containment of the surprise-shaped readout by the critic's level",
           "",
           "`swap65k`, anchor t_v, label realised damage, the matched rows of section 3.  "
           "The surprise-shaped readout here is the model's REALISED horizon excess "
           "`sum_{u=t0}^{t0+8}(NLL_u - H(q_u))`, not the banked `coeruleus/` head: that "
           "head is a readout fitted to the FROZEN trunk's states and does not transfer "
           "to a trunk whose states moved.  striatum's addendum found the realised "
           "horizon excess reads damage better than the head does, so it is the stronger "
           "comparator.  The banked columns are printed for the frozen arm, which has the "
           "head beside it.  `X|Y` is the AUC of X inside five quantile bins of Y "
           "(weighted by n_pos*n_neg); `oof[...]` is a 5-fold least-squares combination.",
           "", sec_addendum(arms), ""]

    jr = [a for a in arms if a.junction and os.path.exists(a.jnpz)]
    if jr:
        md += ["## 7. junction -- does the diet reach the legality reading once the trunk "
               "was shaped?", "",
               "`a1`, the edit onset, the fixed held-out rows; AUC of the value revision "
               "R for swap vs rare among consequential episodes, exact strata on "
               "(j, position).  Every diet is read on identical episodes.", "",
               sec_junction(jr), "",
               "### The positive control -- does a diet reach the critic at all?", "",
               "Unconditional swap-vs-rare on the critic's level over every edited "
               "held-out episode, nothing matched.", "",
               sec_junction_control(jr), ""]

    open(os.path.join(args.out, "tables.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(args.out, "tables.md"))
    if not args.no_figs:
        figures(arms, args.figs)


if __name__ == "__main__":
    main()
