"""Drift-direction analysis: why do the seeds' drift directions converge as m grows?

Part A reads the committed sweep (`idiolect_results_seed43.json`) and
decomposes the seed-42 vs seed-43 drift angle: geometry check, per-level
angles, shared/private energy split, projection onto the pretrained model's
own bias, round-by-round trajectory, and a parametric bootstrap for the
sampling-noise CI (plus the pure-noise null, which is 60 degrees).

Part B reads `selection_differential.json` (from `drift_direction.py`) and
tests whether the shared component is the verifier selecting the synonyms the
shared pretrained checkpoint executes most reliably: alignment of the final
drift with the one-step selection differential at the pretrained checkpoint,
and of each seed's round-to-round change with its own selection differential.

All geometry is done in sqrt-probability (Hellinger) coordinates on the
per-feature conditional synonym distributions, feature- and node-weighted —
the same geometry as the sqrt-JS triangle in the README, and it reproduces
those angles to within a degree.

Usage (from this directory):
  python3 aggregate_direction.py [../idiolect_results_seed43.json] [selection_differential.json]
"""

import json
import math
import os
import sys

import numpy as np

MS = ["2", "3", "4", "6"]
LEVELS = ["L1", "L2", "L3", "L4", "L5"]


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------

def cond_dist(H):
    H = np.asarray(H, float)
    w = H.sum(axis=1)
    return w / max(w.sum(), 1e-30), H / np.clip(w[:, None], 1e-30, None)


def js_bits(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    wp, wq = p.sum(1), q.sum(1); w = 0.5 * (wp + wq)
    cp = p / np.clip(wp[:, None], 1e-30, None); cq = q / np.clip(wq[:, None], 1e-30, None)
    mid = 0.5 * (cp + cq)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(np.clip(a, 1e-30, None) / np.clip(b, 1e-30, None)), 0.0)
        return t.sum(1)
    return float((w * (0.5 * kl(cp, mid) + 0.5 * kl(cq, mid))).sum())


def js_overall(A, B, levels=LEVELS):
    tot = acc = 0.0
    for L in levels:
        w = A[L][1] + B[L][1]
        acc += w * js_bits(A[L][0], B[L][0]); tot += w
    return acc / tot


def tri_angle(a2, b2, c2):
    a, b = math.sqrt(a2), math.sqrt(b2)
    return math.degrees(math.acos(max(-1, min(1, (a2 + b2 - c2) / (2 * a * b)))))


def js_angle(P, A, B, levels=LEVELS):
    return tri_angle(js_overall(P, A, levels), js_overall(P, B, levels), js_overall(A, B, levels))


def drift_vec(P, A, levels=LEVELS):
    """A - P in sqrt-conditional coordinates, with feature x node weights."""
    vs, ws = [], []
    for L in levels:
        wp, cp = cond_dist(P[L][0]); wa, ca = cond_dist(A[L][0])
        w = 0.5 * (wp + wa) * (P[L][1] + A[L][1])
        vs.append((np.sqrt(ca) - np.sqrt(cp)).ravel()); ws.append(np.repeat(w, cp.shape[1]))
    return np.concatenate(vs), np.concatenate(ws)


def wcos(u, v, w):
    den = math.sqrt((w * u * u).sum() * (w * v * v).sum())
    return (w * u * v).sum() / den if den > 0 else float("nan")


def wnorm(u, w):
    return math.sqrt((w * u * u).sum() / w.sum())


def ang(c):
    return math.degrees(math.acos(max(-1, min(1, c))))


def cos_angle(P, A, B, levels=LEVELS):
    u, w = drift_vec(P, A, levels); v, _ = drift_vec(P, B, levels)
    return ang(wcos(u, v, w))


def uniform_like(P):
    return {L: (np.repeat(np.asarray(P[L][0]).sum(1, keepdims=True), np.asarray(P[L][0]).shape[1], 1)
                / np.asarray(P[L][0]).shape[1], P[L][1]) for L in P}


def resample(H, rng):
    out = {}
    for L, (h, n) in H.items():
        h = np.asarray(h, float)
        c = rng.multinomial(n, h.ravel() / h.sum()).reshape(h.shape).astype(float)
        out[L] = (c / c.sum(), n)
    return out


# ----------------------------------------------------------------------
# Part A: the committed sweep
# ----------------------------------------------------------------------

def part_a(d):
    def hist(m, cond):
        c = d["by_m"][m]["conditions"][cond]["sampled"]
        return {L: (np.asarray(c[L]["joint_hist"], float), c[L]["n_valid"]) for L in LEVELS if L in c}

    print("## A. Decomposing the seed-42 vs seed-43 drift angle (committed sweep)\n")
    print("### A1. Geometry check: README sqrt-JS triangle angle vs direct Hellinger cosine; same-seed rerun floor")
    print("| m | JS-angle (parse) | cos-angle (parse) | JS-angle (exact) | rerun angle (parse) | rerun angle (exact) |")
    print("|---|---|---|---|---|---|")
    for m in MS:
        P = hist(m, "pretrained")
        A, B = hist(m, "ei_parse"), hist(m, "ei_parse_seed43")
        Ae, Be = hist(m, "ei_exact"), hist(m, "ei_exact_seed43")
        print(f"| {m} | {js_angle(P,A,B):.1f} | {cos_angle(P,A,B):.1f} | {js_angle(P,Ae,Be):.1f} | "
              f"{cos_angle(P,A,hist(m,'ei_parse_rerun')):.1f} | {cos_angle(P,Ae,hist(m,'ei_exact_rerun')):.1f} |")

    print("\n### A2. Per-level angle (parse) and each level's share of the seed-42 drift energy")
    print("| m | " + " | ".join(LEVELS) + " |")
    print("|---|" + "---|" * len(LEVELS))
    for m in MS:
        P, A, B = hist(m, "pretrained"), hist(m, "ei_parse"), hist(m, "ei_parse_seed43")
        tot = sum((P[L][1] + A[L][1]) * js_overall(P, A, [L]) for L in LEVELS)
        cells = [f"{cos_angle(P,A,B,[L]):.0f}° ({100*(P[L][1]+A[L][1])*js_overall(P,A,[L])/tot:.0f}%)" for L in LEVELS]
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n### A3. Shared vs private energy, and the pretrained model's own bias b = pretrained − uniform")
    print("| m | shared frac (parse) | shared frac (exact) | cos(d42,b) | cos(d43,b) | cos(shared,b) | angle after projecting out b |")
    print("|---|---|---|---|---|---|---|")
    for m in MS:
        P = hist(m, "pretrained"); U = uniform_like(P)
        b, w = drift_vec(U, P)
        u, _ = drift_vec(P, hist(m, "ei_parse")); v, _ = drift_vec(P, hist(m, "ei_parse_seed43"))
        ue, _ = drift_vec(P, hist(m, "ei_exact")); ve, _ = drift_vec(P, hist(m, "ei_exact_seed43"))

        def frac(x, y):
            sh, pr = 0.5 * (x + y), 0.5 * (x - y)
            s2, p2 = (w * sh * sh).sum(), (w * pr * pr).sum()
            return s2 / (s2 + p2)

        def proj_out(x):
            return x - (w * x * b).sum() / (w * b * b).sum() * b
        print(f"| {m} | {frac(u,v):.2f} | {frac(ue,ve):.2f} | {wcos(u,b,w):+.2f} | {wcos(v,b,w):+.2f} | "
              f"{wcos(0.5*(u+v),b,w):+.2f} | {ang(wcos(proj_out(u),proj_out(v),w)):.1f}° |")

    print("\n### A4. Round-by-round angle between the two seeds' drifts (parse; seed-42 rounds from ei02)")
    print("| m | r1 | r2 | r3 | r4 | r5 | r6 |")
    print("|---|---|---|---|---|---|---|")
    for m in MS:
        P = hist(m, "pretrained")
        cells = [f"{cos_angle(P, hist(m, f'ei_parse_r{r}'), hist(m, f'ei_parse_seed43_r{r}')):.0f}°" for r in range(1, 7)]
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n### A5. Sampling-noise CI on the angle (parametric bootstrap, 300 draws) and the pure-noise null")
    print("| m | angle | 95% CI | null: three redraws of the pretrained histogram |")
    print("|---|---|---|---|")
    rng = np.random.default_rng(0)
    for m in MS:
        P, A, B = hist(m, "pretrained"), hist(m, "ei_parse"), hist(m, "ei_parse_seed43")
        bs, null = [], []
        for _ in range(300):
            bs.append(cos_angle(resample(P, rng), resample(A, rng), resample(B, rng)))
            null.append(cos_angle(resample(P, rng), resample(P, rng), resample(P, rng)))
        print(f"| {m} | {cos_angle(P,A,B):.1f}° | [{np.percentile(bs,2.5):.1f}°, {np.percentile(bs,97.5):.1f}°] | {np.mean(null):.1f}° |")


# ----------------------------------------------------------------------
# Part B: the replayed selection step
# ----------------------------------------------------------------------

def part_b(sd):
    def H(m, ck, key="sample_hist"):
        c = sd["by_m"][m]["checkpoints"][ck]
        n_key = "n_valid" if key == "sample_hist" else "n_valid_winners"
        return {L: (np.asarray(c[L][key], float), c[L][n_key]) for L in LEVELS if L in c}

    def sel_vec(m, ck, levels=LEVELS):
        """One-step selection differential: winners − samples at checkpoint ck."""
        return drift_vec(H(m, ck), H(m, ck, "winner_hist"), levels)

    print("\n\n## B. Replaying one selection step (valid-conditioned — see caveat)\n")
    print("Caveat: the selection differential below compares winners' synonym choice with samples' synonym choice")
    print("*among grammatical nodes on both sides*. It therefore cannot see selection acting through grammaticality")
    print("itself (attempts at a poorly-known synonym that fail to parse never enter either histogram). Part C measures")
    print("competence directly. B1 remains a clean replication of the angle on fresh samples.\n")
    print(f"(replayed best-of-{sd['config']['k']} parse selection at {sd['config']['n_prompts']} prompts per checkpoint)\n")

    print("### B1. Replication of the angle on fresh samples (16 per prompt), and its per-level pattern")
    print("| m | angle s42 vs s43 (final) | " + " | ".join(LEVELS) + " |")
    print("|---|---|" + "---|" * len(LEVELS))
    for m in MS:
        P, A, B = H(m, "pretrained"), H(m, "s42_r6"), H(m, "s43_r6")
        cells = [f"{cos_angle(P,A,B,[L]):.0f}°" for L in LEVELS]
        print(f"| {m} | {cos_angle(P,A,B):.1f}° | " + " | ".join(cells) + " |")

    print("\n### B2. Does the drift follow the selection differential measured at the *pretrained* checkpoint?")
    print("S0 = winners − samples at pretrained (same for both seeds). |S0| is the one-step selection strength on synonym choice.")
    print("| m | |S0| | |d42| | |d43| | cos(d42,S0) | cos(d43,S0) | cos(shared,S0) | cos(private,S0) | angle after projecting out S0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MS:
        P = H(m, "pretrained")
        s0, w = sel_vec(m, "pretrained")
        u, _ = drift_vec(P, H(m, "s42_r6")); v, _ = drift_vec(P, H(m, "s43_r6"))
        sh, pr = 0.5 * (u + v), 0.5 * (u - v)

        def proj_out(x):
            return x - (w * x * s0).sum() / (w * s0 * s0).sum() * s0
        print(f"| {m} | {wnorm(s0,w):.4f} | {wnorm(u,w):.4f} | {wnorm(v,w):.4f} | {wcos(u,s0,w):+.2f} | {wcos(v,s0,w):+.2f} | "
              f"{wcos(sh,s0,w):+.2f} | {wcos(pr,s0,w):+.2f} | {ang(wcos(proj_out(u),proj_out(v),w)):.1f}° |")

    print("\n   per level: cos(shared drift, S0)")
    print("| m | " + " | ".join(LEVELS) + " |")
    print("|---|" + "---|" * len(LEVELS))
    for m in MS:
        P = H(m, "pretrained")
        cells = []
        for L in LEVELS:
            s0, w = sel_vec(m, "pretrained", [L])
            u, _ = drift_vec(P, H(m, "s42_r6"), [L]); v, _ = drift_vec(P, H(m, "s43_r6"), [L])
            cells.append(f"{wcos(0.5*(u+v), s0, w):+.2f}")
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n### B3. One-step prediction within a seed: cos(change from round r to r+1, selection differential at round r)")
    print("(r=0 is the pretrained checkpoint; a positive, consistent value means each round moves the way its own selection step pointed)")
    print("| m | seed | r0→1 | r1→2 | r2→3 | r3→4 | r4→5 | r5→6 | mean |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MS:
        for seed in ("s42", "s43"):
            cks = ["pretrained"] + [f"{seed}_r{r}" for r in range(1, 7)]
            cs = []
            for i in range(6):
                s, w = sel_vec(m, cks[i])
                du, _ = drift_vec(H(m, cks[i]), H(m, cks[i + 1]))
                cs.append(wcos(du, s, w))
            print(f"| {m} | {seed} | " + " | ".join(f"{c:+.2f}" for c in cs) + f" | {np.mean(cs):+.2f} |")

    print("\n### B4. Is the selection differential the checkpoint's competence? cos(S0, per-rule upward validity deviation) and cos(S0, b)")
    print("| m | cos(S0, up-validity) | cos(S0, reward-by-rule) | cos(S0, b=pretrained−uniform) | sample→winner reward |")
    print("|---|---|---|---|---|")
    for m in MS:
        c = sd["by_m"][m]["checkpoints"]["pretrained"]
        s0, w = sel_vec(m, "pretrained")
        P = H(m, "pretrained"); U = uniform_like(P)
        b, _ = drift_vec(U, P)
        ups, rws = [], []
        for L in LEVELS:
            for key, acc in (("up_valid_by_rule", ups), ("reward_by_rule", rws)):
                x = np.asarray(c[L][key], float)
                x = np.where(np.isnan(x), np.nanmean(x, axis=1, keepdims=True), x)
                x = x - np.nanmean(x, axis=1, keepdims=True)      # deviation from the feature's mean
                acc.append(np.nan_to_num(x).ravel())
        ups, rws = np.concatenate(ups), np.concatenate(rws)
        print(f"| {m} | {wcos(s0,ups,w):+.2f} | {wcos(s0,rws,w):+.2f} | {wcos(s0,b,w):+.2f} | "
              f"{c['sample_reward_mean']:.3f}→{c['winner_reward_mean']:.3f} |")


# ----------------------------------------------------------------------
# Part C: competence (teacher-forced loss by synonym)
# ----------------------------------------------------------------------

def loss_vec(entry, key="subtree", levels=LEVELS):
    """Per-(feature, rule) loss deviation from the feature's mean, concatenated over levels."""
    out = []
    for L in levels:
        x = np.asarray(entry[L][key], float)
        out.append((x - x.mean(axis=1, keepdims=True)).ravel())
    return np.concatenate(out)


def part_c(sd, cp):
    def H(m, ck):
        c = sd["by_m"][m]["checkpoints"][ck]
        return {L: (np.asarray(c[L]["sample_hist"], float), c[L]["n_valid"]) for L in LEVELS}

    print("\n\n## C. Is the shared component the checkpoint's own competence?\n")
    print("Competence = −(teacher-forced per-token CE over a node's subtree on ground-truth sequences), per (level, feature, synonym),")
    print("deviation from the feature's mean. Measured at the shared pretrained checkpoint. Alignment is expected exactly where the")
    print("verifier filters hard (low validity): a poorly-known synonym fails to parse more often, so the winners over-represent the")
    print("well-known ones, and SFT on winners moves *both* seeds toward them.\n")
    print("### C1. Per level: pretrained validity, angle between the seeds' drifts, cos(shared drift, competence)")
    print("| m | " + " | ".join(LEVELS) + " |")
    print("|---|" + "---|" * len(LEVELS))
    for m in MS:
        P = H(m, "pretrained")
        comp = -loss_vec(cp["by_m"][m]["checkpoints"]["pretrained"])
        cells = []; off = 0
        for L in LEVELS:
            n = P[L][0].size; sl = slice(off, off + n); off += n
            u, w = drift_vec(P, H(m, "s42_r6"), [L]); v, _ = drift_vec(P, H(m, "s43_r6"), [L])
            vf = sd["by_m"][m]["checkpoints"]["pretrained"][L]["valid_frac"]
            cells.append(f"valid {vf:.2f}, {ang(wcos(u,v,w)):.0f}°, cos {wcos(0.5*(u+v), comp[sl], w):+.2f}")
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n### C2. Overall alignments (all levels, node-weighted)")
    print("| m | cos(shared, competence) | cos(d42, comp) | cos(d43, comp) | cos(b = pretrained−uniform, comp) | cos(d42, −Δloss42) | cos(d43, −Δloss43) |")
    print("|---|---|---|---|---|---|---|")
    for m in MS:
        P = H(m, "pretrained"); U = uniform_like(P)
        comp = -loss_vec(cp["by_m"][m]["checkpoints"]["pretrained"])
        b, w = drift_vec(U, P)
        u, _ = drift_vec(P, H(m, "s42_r6")); v, _ = drift_vec(P, H(m, "s43_r6"))
        dl42 = -(loss_vec(cp["by_m"][m]["checkpoints"]["s42_r6"]) - loss_vec(cp["by_m"][m]["checkpoints"]["pretrained"]))
        dl43 = -(loss_vec(cp["by_m"][m]["checkpoints"]["s43_r6"]) - loss_vec(cp["by_m"][m]["checkpoints"]["pretrained"]))
        print(f"| {m} | {wcos(0.5*(u+v),comp,w):+.2f} | {wcos(u,comp,w):+.2f} | {wcos(v,comp,w):+.2f} | {wcos(b,comp,w):+.2f} | "
              f"{wcos(u,dl42,w):+.2f} | {wcos(v,dl43,w):+.2f} |")


# ----------------------------------------------------------------------
# Part D: the causal controls
# ----------------------------------------------------------------------

def part_d(ct):
    def H(m, ck):
        c = ct["by_m"][m]["checkpoints"][ck]
        return {L: (np.asarray(c[L]["sample_hist"], float), c[L]["n_valid"]) for L in LEVELS}

    def has(m, ck):
        return ck in ct["by_m"].get(m, {}).get("checkpoints", {})

    print("\n\n## D. Causal controls: selection-free EI (random reward) and EI from a fresh pretraining seed\n")
    for m in ("2", "6"):
        if m not in ct["by_m"]:
            continue
        P42 = H(m, "pt42")
        d42, w = drift_vec(P42, H(m, "s42_r6")); d43, _ = drift_vec(P42, H(m, "s43_r6"))
        comp42 = -loss_vec(ct["by_m"][m]["checkpoints"]["pt42"]["loss"])
        print(f"### m = {m}")
        print("| arm | |drift| | angle vs s42 | angle vs s43 | cos(drift, competence of its own start) | per-level angle vs s42 | per-level cos(comp) |")
        print("|---|---|---|---|---|---|---|")

        def row(name, P, ck, comp, ref_P=None):
            d, wd = drift_vec(P, H(m, ck))
            # for the fresh-seed arm the drift lives in its own coordinates; compare directionally anyway
            per_ang, per_cos = [], []
            off = 0
            for L in LEVELS:
                n = P[L][0].size; sl = slice(off, off + n); off += n
                per_ang.append(f"{ang(wcos(d[sl], d42[sl], wd[sl])):.0f}°")
                per_cos.append(f"{wcos(d[sl], comp[sl], wd[sl]):+.2f}")
            print(f"| {name} | {wnorm(d,wd):.4f} | {ang(wcos(d,d42,wd)):.0f}° | {ang(wcos(d,d43,wd)):.0f}° | "
                  f"{wcos(d,comp,wd):+.2f} | {' '.join(per_ang)} | {' '.join(per_cos)} |")
        row("s43 (parse, same checkpoint)", P42, "s43_r6", comp42)
        if has(m, "rand_r6"):
            row("random selection (same checkpoint)", P42, "rand_r6", comp42)
        if has(m, "pt44_r6"):
            P44 = H(m, "pt44")
            comp44 = -loss_vec(ct["by_m"][m]["checkpoints"]["pt44"]["loss"])
            row("parse, fresh pretraining seed 44 (drift rel. its own pretrained)", P44, "pt44_r6", comp44)
            b42, _ = drift_vec(uniform_like(P42), P42); b44, _ = drift_vec(uniform_like(P44), P44)
            print(f"\n  competence profiles of the two checkpoints: cos(comp42, comp44) = {wcos(comp42, comp44, w):+.2f}; "
                  f"their synonym biases: cos(b42, b44) = {wcos(b42, b44, w):+.2f}")
            off = 0; cells = []
            for L in LEVELS:
                n = P42[L][0].size; sl = slice(off, off + n); off += n
                cells.append(f"{L}: {wcos(comp42[sl], comp44[sl], w[sl]):+.2f}")
            print("  per level cos(comp42, comp44): " + "  ".join(cells))
        if has(m, "rand_r6"):
            print("\n  round-by-round |drift| and angle vs s42 for the random-selection arm:")
            for r in range(1, 7):
                if has(m, f"rand_r{r}"):
                    d, wd = drift_vec(P42, H(m, f"rand_r{r}"))
                    print(f"    r{r}: |d|={wnorm(d,wd):.4f}  angle vs s42={ang(wcos(d,d42,wd)):.0f}°  parse reward="
                          f"{ct['by_m'][m]['checkpoints'][f'rand_r{r}']['sample_reward_mean']:.3f}  "
                          f"L1 valid={ct['by_m'][m]['checkpoints'][f'rand_r{r}']['L1']['valid_frac']:.3f}")
        print()


def main(sweep_path, sel_path, comp_path="competence.json", ctrl_path="controls.json"):
    with open(sweep_path) as f:
        part_a(json.load(f))
    sd = None
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            sd = json.load(f)
        part_b(sd)
    else:
        print(f"\n(no {sel_path}; skipping part B)")
    if sd is not None and os.path.exists(comp_path):
        with open(comp_path) as f:
            part_c(sd, json.load(f))
    if os.path.exists(ctrl_path):
        with open(ctrl_path) as f:
            part_d(json.load(f))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../idiolect_results_seed43.json",
         sys.argv[2] if len(sys.argv) > 2 else "selection_differential.json")
