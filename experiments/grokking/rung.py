"""[grokking] Node 2: rung -- freeze the minted K as the input alphabet and train fresh nets over it.

Same 30% split, same seed, same full-batch AdamW recipe (lr 1e-3, wd 1.0); only the input alphabet
and the learner change.

  Alphabets (features of a and of b, concatenated):
    raw     one-hot (a, b), 2p = 194 dims                          (the original)
    minted  [cos, sin](2 pi w a/p) for w in K, and the same for b, 4k dims
    all48   the same at all 48 characters, 192 dims                (the basis gift without the mint)
    all48n  (control, group `ctrl`) all48 scaled by sqrt(k/48) so its per-input norm equals minted's
            (per-input norms: raw and top1 sqrt 2, minted and lowE sqrt(2k), all48 sqrt 96)
    lowE    the k LOWEST-producer-score frequencies of the final net not in K, 4k dims
            (any nonzero character solves this task, so this may tie `minted`: the mint would then
            buy the count, not the identity)
    top1    the single highest-producer-score frequency of the final net, 4 dims (what the symbolic
            gate, A-LS under m3, kept: the minted alphabet in the strict sense)
  Learners:
    mlp       the [128, 128] ReLU GrokMLP with the input dim swapped (alphabet -> epochs-to-solve)
    bilinear  logits = W((U x_a) * (V x_b)) + b, hidden width m (default 4k, k = |K|, for EVERY
              alphabet): the minimal product net over the features (alphabet -> parameter count)
  Metrics: epochs to 99% and 100% test (eval every 10 epochs; the run stops at 100% or the cap),
  parameter count, params x epochs, and 6 * params * n_train * epochs as a FLOP count.

  Groups:
    main    {raw, minted, all48, lowE, top1} x {mlp, bilinear} at train fraction 0.3, plus top1 x
            bilinear at its own minimal width m = 4 (the matched-width arm uses m = 4|K|)
    sweep   train fraction {0.3, 0.2, 0.1, 0.05, 0.02} x {raw, minted, top1}, mlp (0.3 shared with main)
    add3    optional: (a+b+c) mod p at train fraction 0.02, raw vs minted, mlp (off by default)

K is read from the mint run's final snapshot (`--mint-tag`, `--k-form`, `--k-rule`), or given
explicitly (`--k-list 3,17,40`).

Commands (run from experiments/, MODAL_PROFILE=chromatic):
    modal run grokking/rung.py::rung --tag rsmoke --smoke 1
    modal run --detach grokking/rung.py::rung --tag r1 --mint-tag m1 --k-form B --k-rule lt1 \
        --cap-sweep 150000
    python3 grokking/reduce_rung.py --tag r1 --fetch
"""

import json
import os
import time

import numpy as np

from grokking.shared import (DATA_DIR, HIDDEN, LR, P, SEED, WEIGHT_DECAY, GrokMLP, app,
                                  char_features, n_params, nn, onehot, peak_rss_mb, split_pairs,
                                  torch, train_net, volume)

FRACS = (0.3, 0.2, 0.1, 0.05, 0.02)
EVAL_EVERY = 10
CURVE_KEEP_EVERY = 10          # keep every 10th eval (= every 100 epochs) in the stored curve


if nn is not None:

    class Bilinear(nn.Module):
        """logits = W((U x_a) * (V x_b)) + b. x = [x_a, x_b] split in half."""

        def __init__(self, d_half, m, p=P):
            super().__init__()
            self.d_half = d_half
            self.U = nn.Linear(d_half, m, bias=False)
            self.V = nn.Linear(d_half, m, bias=False)
            self.W = nn.Linear(m, p)

        def forward(self, x):
            return self.W(self.U(x[:, :self.d_half]) * self.V(x[:, self.d_half:]))


def features(alphabet, a, b, freqs, c=None):
    """Input tensor for pairs (a, b) (or triples with c) under an alphabet."""
    if alphabet == "raw":
        if c is None:
            return onehot(a, b)
        x = torch.zeros(len(a), 3 * P)
        idx = torch.arange(len(a))
        x[idx, torch.as_tensor(a)] = 1.0
        x[idx, P + torch.as_tensor(b)] = 1.0
        x[idx, 2 * P + torch.as_tensor(c)] = 1.0
        return x
    xab = char_features(a, b, freqs)
    if c is None:
        return xab
    xc = char_features(c, c, freqs)[:, :2 * len(freqs)]
    return torch.cat([xab, xc], 1)


def alphabet_freqs(K, lowK, top1):
    return {"raw": [], "minted": list(K), "all48": list(range(1, 49)), "lowE": list(lowK),
            "top1": [int(top1)], "all48n": list(range(1, 49))}


@app.function(cpu=8.0, memory=2048, timeout=4 * 3600, volumes={DATA_DIR: volume}, max_containers=4)
def rung_arm(tag: str, arm: dict, freqs_by_alph: dict, K: list, seed: int = SEED):
    t0 = time.time()
    freqs = list(freqs_by_alph[arm["alphabet"]])
    task = arm.get("task", "add2")
    if task == "add2":
        a_tr, b_tr, a_te, b_te = split_pairs(P, arm["frac"], seed)
        tr_x = features(arm["alphabet"], a_tr, b_tr, freqs)
        te_x = features(arm["alphabet"], a_te, b_te, freqs)
        if arm["alphabet"] == "all48n":     # norm-matched control: per-input norm equal to minted's
            sc = float(np.sqrt(len(K) / 48.0))
            tr_x, te_x = tr_x * sc, te_x * sc
        tr_y = torch.as_tensor((a_tr + b_tr) % P)
        te_y = torch.as_tensor((a_te + b_te) % P)
    else:  # add3: all p^3 triples, deterministic shuffle, test subsampled to 20k
        rng = np.random.RandomState(seed)
        idx = rng.permutation(P ** 3)
        n_tr = int(P ** 3 * arm["frac"])
        tr_i, te_i = idx[:n_tr], idx[n_tr:n_tr + 20000]

        def dec(i):
            return i // (P * P), (i // P) % P, i % P
        (a_tr, b_tr, c_tr), (a_te, b_te, c_te) = dec(tr_i), dec(te_i)
        tr_x = features(arm["alphabet"], a_tr, b_tr, freqs, c_tr)
        te_x = features(arm["alphabet"], a_te, b_te, freqs, c_te)
        tr_y = torch.as_tensor((a_tr + b_tr + c_tr) % P)
        te_y = torch.as_tensor((a_te + b_te + c_te) % P)
    D = tr_x.shape[1]
    torch.manual_seed(seed)
    if arm["learner"] == "mlp":
        model = GrokMLP(p=P, hidden_dims=HIDDEN, in_dim=D)
    else:
        m = int(arm.get("m") or 4 * max(1, len(K)))     # same width for every alphabet
        if task != "add2":
            raise ValueError("bilinear learner is add2-only")
        model = Bilinear(D // 2, m, P)
    npar = n_params(model)
    res = train_net(model, tr_x, tr_y, te_x, te_y, int(arm["cap"]), LR, WEIGHT_DECAY,
                    eval_every=EVAL_EVERY, stop_at_test=1.0, log_every=5000, tag=f"[{arm['name']}]")
    fh = res["first_hit"]
    n_tr = int(tr_x.shape[0])
    out = {"arm": arm, "freqs": freqs, "in_dim": int(D), "params": npar, "n_train": n_tr,
           "m_bilinear": int(m) if arm["learner"] == "bilinear" else None,
           "epochs_run": res["epochs_run"], "first_hit": fh, "seconds": res["seconds"],
           "final_test": res["curve"][-1][2], "final_train": res["curve"][-1][1],
           "curve": res["curve"][::CURVE_KEEP_EVERY] + [res["curve"][-1]],
           "peak_rss_mb": peak_rss_mb()}
    for thr in ("0.99", "1.0"):
        e = fh[thr]
        out[f"params_x_epochs_{thr}"] = None if e is None else npar * (e + 1)
        out[f"flops_{thr}"] = None if e is None else 6 * npar * n_tr * (e + 1)
    print(f"[{arm['name']}] D={D} params={npar} e99={fh['0.99']} e100={fh['1.0']} "
          f"final_test={out['final_test']:.4f} {time.time() - t0:.0f}s rss={peak_rss_mb():.0f}MB", flush=True)
    return out


def build_arms(groups, cap_main, cap_sweep, cap_add3, m_bilinear=None):
    arms = []
    if "main" in groups:
        for alph in ("raw", "minted", "all48", "lowE", "top1"):
            for learner in ("mlp", "bilinear"):
                arms.append({"name": f"main_{alph}_{learner}", "group": "main", "alphabet": alph,
                             "learner": learner, "frac": 0.3, "cap": cap_main, "m": m_bilinear})
        arms.append({"name": "main_top1_bilinear_m4", "group": "main", "alphabet": "top1",
                     "learner": "bilinear", "frac": 0.3, "cap": cap_main, "m": 4})
    if "sweep" in groups:
        for frac in FRACS:
            for alph in ("raw", "minted", "top1"):
                if frac == 0.3 and "main" in groups:
                    continue           # shared with main
                arms.append({"name": f"sweep_{alph}_f{frac}", "group": "sweep", "alphabet": alph,
                             "learner": "mlp", "frac": frac, "cap": cap_sweep})
    if "ctrl" in groups:                   # added after r1: all48 rescaled to minted-13's input norm
        for learner in ("mlp", "bilinear"):
            arms.append({"name": f"ctrl_all48n_{learner}", "group": "ctrl", "alphabet": "all48n",
                         "learner": learner, "frac": 0.3, "cap": cap_main, "m": m_bilinear})
    if "add3" in groups:
        for alph in ("raw", "minted"):
            arms.append({"name": f"add3_{alph}_f0.02", "group": "add3", "alphabet": alph,
                         "learner": "mlp", "frac": 0.02, "cap": cap_add3, "task": "add3"})
    return arms


def resolve_K(mint_tag, k_form, k_rule, K_str):
    """K and the lowE control from the mint run's final snapshot (or an explicit K)."""
    path = os.path.join(DATA_DIR, mint_tag, "mint.json")
    with open(path) as f:
        d = json.load(f)
    fin = max((r for r in d["records"].values() if r["run"] == "true"), key=lambda r: r["epoch"])
    K = [int(x) for x in K_str.split(",")] if K_str else list(fin["walks"][k_form][k_rule]["K"])
    score = np.asarray(fin["score"])
    w = np.arange(1, len(score) + 1)
    prod = [int(x) for x in w[np.lexsort((w, -score))]]         # descending producer score (mint's order)
    order = prod[::-1]                                          # ascending producer score
    lowK = [x for x in order if x not in K][:len(K)]
    top1 = prod[0]
    src = {"final_epoch": fin["epoch"], "score": fin["score"], "prod_order": prod,
           "K_is_prod_top_k": sorted(K) == sorted(prod[:len(K)]),
           "top1_is_ALS_m3_K": fin["walks"]["ALS"]["m3"]["K"] == [top1]}
    return K, lowK, top1, src


@app.function(cpu=1.0, memory=2048, timeout=12 * 3600, volumes={DATA_DIR: volume})
def rung(tag: str = "r1", mint_tag: str = "m1", k_form: str = "B", k_rule: str = "lt1", k_list: str = "",
         groups: str = "main,sweep", cap_main: int = 40000, cap_sweep: int = 100000,
         cap_add3: int = 40000, m_bilinear: int = 0, smoke: int = 0):
    """Coordinator (CPU): resolve K from the mint run, fan the arms out (<= 4 containers)."""
    t0 = time.time()
    volume.reload()
    Kl, lowK, top1, src = resolve_K(mint_tag, k_form, k_rule, k_list)
    fba = alphabet_freqs(Kl, lowK, top1)
    groups = groups.split(",")
    if smoke:
        cap_main, cap_sweep, cap_add3 = 2000, 2000, 500
    arms = build_arms(groups, cap_main, cap_sweep, cap_add3, m_bilinear or None)
    if smoke:
        arms = [a for a in arms if a["name"] in ("main_top1_mlp", "main_top1_bilinear_m4", "main_lowE_bilinear",
                                                 "sweep_top1_f0.02")] or arms[:3]
    print(f"K ({k_form}/{k_rule}, k={len(Kl)}) = {Kl}; K is producer top-{len(Kl)}: {src['K_is_prod_top_k']}; "
          f"lowE = {lowK}; top1 = {top1} (== ALS/m3 K: {src['top1_is_ALS_m3_K']}); {len(arms)} arms", flush=True)
    # longest arms first (raw one-hot, then the small-fraction sweep) so the <= 4 containers stay busy
    arms = sorted(arms, key=lambda a: (a["alphabet"] != "raw", a["group"] != "sweep", a["frac"]))
    results = list(rung_arm.starmap([(tag, a, fba, Kl, SEED) for a in arms]))
    out = {"tag": tag, "mint_tag": mint_tag, "k_form": k_form, "k_rule": k_rule, "K": Kl, "lowK": lowK,
           "top1": top1, "prod_order": src["prod_order"], "K_is_prod_top_k": src["K_is_prod_top_k"],
           "top1_is_ALS_m3_K": src["top1_is_ALS_m3_K"], "freqs_by_alph": fba,
           "src_final_epoch": src["final_epoch"], "groups": groups, "results": results, "smoke": smoke,
           "recipe": {"lr": LR, "wd": WEIGHT_DECAY, "seed": SEED, "eval_every": EVAL_EVERY}}
    d = os.path.join(DATA_DIR, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rung.json"), "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"rung done in {time.time() - t0:.0f}s -> {d}/rung.json", flush=True)
    return os.path.join(d, "rung.json")
