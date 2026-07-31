"""G1-G3 -- certify the partially-heterogeneous DGP, and measure what off-channel data is WORTH.

WHY THIS EXISTS
---------------
`ideas/meta_learning_under_metered_data.md` §5 shows the repo has only ever measured the two
DEGENERATE ends of the shared-structure axis, and that they answer "should I restrict to what's
relevant?" with opposite signs:

  specialization/   one shared ruleset, A and B are subtrees   -> INERT (breadth restriction Δ<=0.013)
  full_loop/        independent rule tables per channel        -> PAYS 0.063 (uniform -> oracle)

§9 names the missing middle as the realism condition: *superficially unrelated, deeply shared*.
`rhm_channels.share_rules_top` builds it -- structA is depth- and m-matched to the tree and takes
its top `share_top` rule tables from the tree, keeping its own surface tables. Sharing depth 0 is
full_loop's geometry; sharing depth L is specialization's; 1..L-1 is the middle nobody has run.

This module is the certification that has to pass BEFORE the loop runs on it, in the same
discipline as [`../env_check.py`](../env_check.py):

  G1  STRUCTURAL IRRELEVANCE SURVIVES.  The whole distractor construction rests on P1 -- a
      distractor's content cannot move the parsed root or the DP `d*`. Sharing rule tables is
      exactly the kind of change that could break it, so it is re-asserted at every sharing
      depth. (It should hold: each channel draws its OWN independent root and every grammar
      readout slices the tree alone. But "should" is what P1 was for.)

  G2  THE SURFACE STAYS DISTINCT.  "Superficially unrelated" has to be true or the geometry is
      just specialization with extra steps. Reported as the overlap between the tree's and
      structA's legal leaf tuples, plus which levels are literally identical, plus the unigram
      marginals (`check_marginals`) that the ladder's policies must not be able to read.

  G3  THE TRANSFER CURVE -- the load-bearing measurement.  Sharing DEEP tables while keeping
      DISTINCT surface tables might buy the block FM nothing at all: the FM predicts Δz for one
      acted block, and a block's rendering is governed by the BOTTOM table, which is never
      shared below full sharing. What sharing *can* buy is the contextual half -- which level-1
      feature the generator infers for a block is constrained by the levels above, over a shared
      feature alphabet. Whether that transfers is empirical, and it is the whole premise of the
      ladder sweep, so it is measured directly and without any loop:

        train a FRESH block FM from scratch under a fixed channel allocation, at matched total
        budget and a matched (uniform-over-blocks) STATE distribution, then read tree
        acted-block FM error.

      The headline comparison is COVERAGE-MATCHED, not `structA_only` -- see the `ALLOC_ARMS`
      comment for why the obvious arm cannot be read. `tree_structA - tree_half` holds tree-block
      transitions exactly fixed and varies only what fills the rest of the budget, against
      `tree_structB` (never-shared grammar), `tree_noise` (irreducible) and `tree_only`
      (a doubling of the tree data itself, the exchange-rate denominator).

  `fm_arch` selects the readout. "block" is the published FM the ladder actually runs on;
  "shared_marker" is the POSITIVE CONTROL added after the first pass came back flat -- see
  `_build_shared_marker_fm`. Whether the sharing knob is visible at all is a property of the
  readout, and the two arches separate that from a property of the geometry.

STATIC ON PURPOSE. No drift anywhere here: this measures a property of the DGP's rule tables, and
the ladder is where the same geometry meets a moving world and a metered budget.

RULE SEEDS. `--seed` here moves the RULE DRAW as well as the training draw (`rule_seed_offset`),
because G1-G3 are claims about the geometry itself. Seed 1 -> offset 0 -> the same rule tables the
ladder sweep uses, so the two nodes cross-reference exactly at that seed.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/partial_hetero/geometry_check.py::geometry_check --quick
  for s in 1 2 3; do
    modal run --detach rhm/directed_sculpting/full_loop/partial_hetero/geometry_check.py::geometry_check \
        --tag g_s$s --seed $s
  done
"""

import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import (check_marginals, check_structural_irrelevance, make_layout,
                              sample_pool)
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    block_latent_mean, blocks_of_channel, build_block_tables, build_channel_local_fm,
    corrupt_tree, make_spec, per_channel_error, regenerate_block, sample_states,
    train_generator_channels)


app = modal.App("rhm-ds-partial-hetero", image=image)


# The fixed allocations G3 trains under: a per-CHANNEL probability vector by name, so the same
# table works at any layout. `uniform` is the ladder's uniform rung (uniform over CHANNELS, not
# blocks) and `tree_structA` is the transfer-aware oracle's allocation.
#
# THE COVERAGE-MATCHED DESIGN, and why the obvious arm is not the headline. The block FM carries a
# per-block `action_embedding` and `block_position`, so an FM trained on structA blocks alone never
# updates the tree blocks' action embeddings at all: `structA_only` conflates GRAMMAR TRANSFER with
# PARAMETER COVERAGE and cannot be read as the value of off-channel data. It is kept because it
# still answers a narrower question (does the shared trunk alone carry anything?), flagged.
#
# The clean comparison holds TREE-BLOCK TRANSITIONS EXACTLY FIXED and varies only what fills the
# rest of the budget. `tree_half` spends half the budget entirely on the tree; each `tree_*` arm
# spends the same absolute amount on the tree and the other half on one distractor. So
#
#     tree_structA - tree_half   = what shared-grammar off-channel data adds  (negative = helps)
#     tree_structB - tree_half   = the same for a never-shared grammar        (the control)
#     tree_noise   - tree_half   = the same for irreducible data              (the null)
#     tree_only    - tree_half   = what DOUBLING tree data buys               (the exchange rate)
#
# and every one of those four arms trains the tree's action embeddings the identical number of
# times. The trunk does get 2x the optimizer steps in the full-budget arms, which is exactly why
# the structB and noise arms are run: they carry that generic effect without carrying any sharing.
ALLOC_ARMS = {
    "tree_half": {"tree": 1.0},                          # HALF budget -- the matched reference
    "tree_only": {"tree": 1.0},                          # ceiling: double the tree data
    "tree_structA": {"tree": 0.5, "structA": 0.5},       # + shared-grammar data
    "tree_structB": {"tree": 0.5, "structB": 0.5},       # + never-shared grammar (control)
    "tree_noise": {"tree": 0.5, "noiseA": 0.25, "noiseB": 0.25},   # + irreducible data (null)
    "uniform": None,                                     # the ladder's floor rung
    "structA_only": {"structA": 1.0},                    # pure transfer, coverage-confounded
    "structB_only": {"structB": 1.0},
}
HALF_BUDGET_ARMS = ("tree_half",)


def alloc_vector(arm, names):
    """Per-channel probability vector for `arm`, in channel order."""
    spec = ALLOC_ARMS[arm]
    if spec is None:
        return np.full(len(names), 1.0 / len(names))
    w = np.array([float(spec.get(n, 0.0)) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError(f"arm {arm!r} allocates nothing over channels {names}")
    return w / w.sum()


def surface_stats(layout):
    """G2: how distinct is each grammar channel's SURFACE from the tree's?

    `overlap_with_tree` is the fraction of a channel's legal leaf s-tuples that are also legal
    tree leaf tuples. At full sharing it is 1.0 by construction; below that it should sit near
    the chance overlap of two independent draws of v*m tuples from v**s, which is what makes
    "superficially unrelated" a fact rather than a hope.
    """
    v, s = layout["v"], layout["s"]
    powers = (v ** np.arange(s)).astype(np.int64)
    tree = layout["tree"]
    tree_codes = set(int(c) for c in (tree["rules"][-1] * powers).sum(-1).ravel())
    out = {}
    for ch in layout["channels"]:
        if ch["rules"] is None:
            continue
        codes = set(int(c) for c in (ch["rules"][-1] * powers).sum(-1).ravel())
        same_depth = ch["depth"] == tree["depth"]
        out[ch["name"]] = {
            "share_top": int(ch.get("share_top") or 0),
            "depth": ch["depth"], "m": ch["m"],
            "n_legal_leaf_tuples": len(codes),
            "overlap_with_tree": len(codes & tree_codes) / max(1, len(codes)),
            "bottom_table_identical": bool(np.array_equal(ch["rules"][-1], tree["rules"][-1])),
            "levels_identical_to_tree": (
                [bool(np.array_equal(a, b)) for a, b in zip(ch["rules"], tree["rules"])]
                if same_depth else None),
        }
    out["_chance_overlap"] = min(1.0, (tree["m"] * v) / float(v ** s))
    return out


def level_sibling_stats(layout, n=20000, seed=13):
    """G2b: is the deep sharing VISIBLE IN THE DATA, level by level, before any model sees it?

    Expands each grammar channel top-down from its own roots and records the feature array at
    every level, then histograms the SIBLING TUPLE at each level -- the s-tuple one rule
    produces. The sibling law at level `ell` is a functional of tables 0..ell-1 alone, so under
    `share_top=k` it is IDENTICAL to the tree's for every ell <= k and differs above. The TV
    vector is therefore a direct, exact readout of the sharing depth in the data's own currency,
    and it is what makes a null in G3 interpretable: flat transfer with a graded TV vector means
    the FM cannot use the sharing, not that the DGP does not have it.

    Exact, not parsed: the derivation is sampled forward, so nothing depends on
    `build_inverse_maps`' last-writer-wins ambiguity.
    """
    v, s = layout["v"], layout["s"]
    powers = (v ** np.arange(s)).astype(np.int64)
    per_channel = {}
    for ch in layout["channels"]:
        if ch["rules"] is None:
            continue
        rng = np.random.default_rng(seed + ch["blk0"])
        cur = rng.integers(0, v, size=(n, 1))
        hists = []
        for ell, layer in enumerate(ch["rules"]):
            choices = rng.integers(0, ch["m"], size=cur.shape)
            nxt = layer[cur, choices]                          # (n, width, s)
            codes = (nxt * powers).sum(-1).ravel()
            h = np.bincount(codes, minlength=v ** s).astype(np.float64)
            hists.append(h / h.sum())
            cur = nxt.reshape(n, -1)
        per_channel[ch["name"]] = hists
    tree_h = per_channel[layout["tree"]["name"]]
    out = {}
    for ch in layout["channels"]:
        if ch["rules"] is None:
            continue
        h = per_channel[ch["name"]]
        out[ch["name"]] = {
            "share_top": int(ch.get("share_top") or 0),
            # TV per level, level 1 (root's own expansion) first; None where depths disagree
            "sibling_tv_vs_tree": [float(0.5 * np.abs(a - b).sum())
                                   for a, b in zip(h, tree_h)],
            "n_levels": len(h),
        }
    return out


def _build_shared_marker_fm():
    """`rhm_sculpt_latent._build_block_fm` with the acted-block marker SHARED across blocks.

    THE POSITIVE CONTROL FOR THE READOUT, added after the first pass came back null. The published
    block FM already shares its trunk, its head and (via every sample) its block-position table --
    the ONLY parameters private to a block are `action_embedding[k]`, the vector that marks which
    block the command touched. That turns out to be fatal on its own: an FM trained on structA
    blocks alone leaves tree FM error at 0.998 (predicting nothing) even at `share_top=4`, where
    structA's rule tables are BIT-IDENTICAL to the tree's. With a random-init marker the acted
    block simply is not identifiable, so no amount of shared grammar can express itself.

    Replacing the per-block marker with ONE learned vector removes the last private parameter
    without changing anything else, so at full sharing -- identical grammars, identical dynamics --
    off-channel data should be worth what on-channel data is worth. If it still is not, the null is
    about the geometry rather than about the parameterisation, and that is the question.
    """
    import torch
    import torch.nn as nn

    class SharedMarkerBlockFM(nn.Module):
        def __init__(self, state_dim, n_blocks, n_head, n_layer):
            super().__init__()
            self.acted = nn.Parameter(torch.randn(state_dim) * 0.02)
            self.block_position = nn.Embedding(n_blocks, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm = nn.LayerNorm(state_dim)
            self.head = nn.Linear(state_dim, state_dim)
            self.register_buffer("positions", torch.arange(n_blocks), persistent=False)

        def forward(self, z_block, k):
            batch, n_blocks, dim = z_block.shape
            marker = torch.zeros_like(z_block)
            marker.scatter_(1, k[:, None, None].expand(-1, 1, dim),
                            self.acted.view(1, 1, dim).expand(batch, 1, dim))
            hidden = z_block + self.block_position(self.positions)[None] + marker
            return self.head(self.norm(self.encoder(hidden)))

    return SharedMarkerBlockFM


def controller_alignment(controller, layout, tb, device, *, n=8192, seed=17, steps=600):
    """G4 -- is the frozen ENCODER channel-position-aligned, before any forward model exists?

    This is the diagnostic that decides WHERE the weight sharing has to go, and it costs one
    linear probe. Decode a block's own level-1 feature from its latent: fit on TREE blocks, then
    read the same probe on structA blocks. At `share_top=4` the two channels' grammars are
    bit-identical, so

      * transfers  -> the encoder writes "feature f" the same way at both positions, and a
                      weight-shared FM (`build_channel_local_fm`) is all that is missing;
      * collapses  -> the ENCODER is the barrier. Its absolute `position_embedding` means no FM
                      architecture on top can align the two channels, and the fix has to move
                      into the state encoder itself.

    Reported raw and after per-block CENTERING, since an additive positional offset is the cheap
    failure mode and `build_channel_local_fm` already subtracts it.
    """
    import torch
    import torch.nn.functional as F

    v, s = layout["v"], layout["s"]
    powers = torch.from_numpy((v ** np.arange(s)).astype(np.int64)).to(device)
    leaves = torch.from_numpy(sample_pool(layout, n, seed)["leaves"]).to(device)
    with torch.no_grad():
        z = torch.cat([controller.block_state(leaves[i:i + 4096])
                       for i in range(0, n, 4096)], dim=0)               # (n, n_blocks, D)
        codes = (leaves.view(n, tb["n_blocks"], s) * powers[None, None]).sum(-1)
        feats = torch.gather(tb["bottom_blk"], 1,
                             codes.T.contiguous()).T                     # (n, n_blocks)

    tree_blk = tb["tree_blocks"]
    A = next((c for c in layout["channels"] if c["name"] == "structA"), None)
    if A is None:
        return None
    a_blk = torch.arange(A["blk0"], A["blk1"], device=device)
    dim = z.shape[-1]
    out = {}
    for tag, zz in (("raw", z), ("centered", z - z.mean(0, keepdim=True))):
        def flat(blks):
            X = zz[:, blks, :].reshape(-1, dim)
            y = feats[:, blks].reshape(-1)
            ok = y >= 0
            return X[ok].float(), y[ok]
        Xt, yt = flat(tree_blk)
        Xa, ya = flat(a_blk)
        cut = int(0.8 * Xt.shape[0])
        probe = torch.nn.Linear(dim, v).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            idx = torch.randint(0, cut, (4096,), device=device)
            loss = F.cross_entropy(probe(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = lambda X, y: float((probe(X).argmax(-1) == y).float().mean())
            out[tag] = {"fit_on_tree_eval_tree": acc(Xt[cut:], yt[cut:]),
                        "fit_on_tree_eval_structA": acc(Xa, ya),
                        "chance": 1.0 / v}
    return out


def train_block_fm_alloc(fm, controller, generator, layout, tb, *, chan_w, n_steps, batch_size,
                         n_corrupt, budget, lr, seed, render, gen, device,
                         pool_size=20_000, pool_refresh=250):
    """`channel_env.train_block_fm` with the ACTED block drawn from a fixed channel allocation.

    The two halves are separated on purpose, exactly as the ladder separates them:
      * the STATE distribution is untouched -- corruption is tree-only and the random pre-steps
        stay uniform over every block, so all arms see the same states;
      * only the block whose transition becomes a TRAINING TARGET is drawn from `chan_w`.
    So an arm is "which transitions did the budget buy", not "which world did the agent live in",
    which is what makes `structA_only - tree_only` a clean read on the data's worth.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    n_ch = tb["n_channels"]
    blocks_by_chan = [blocks_of_channel(tb, c) for c in range(n_ch)]
    live = [c for c in range(n_ch) if chan_w[c] > 0 and len(blocks_by_chan[c]) > 0]
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    fm.train()
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report = max(1, n_steps // 4)
    pool = None
    for step in range(1, n_steps + 1):
        if pool is None or step % pool_refresh == 1:
            pool = sample_pool(layout, pool_size, int(rng.integers(0, 2 ** 31)))["leaves"]
        idx = rng.integers(0, pool.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(corrupt_tree(pool[idx], tree_blocks, c, tb["v"], tb["s"],
                                          rng)).to(device)
        for _ in range(int(rng.integers(0, budget + 1))):
            kk = torch.randint(0, tb["n_blocks"], (batch_size,), device=device, generator=gen)
            x = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
        pick = rng.choice(n_ch, size=batch_size, p=chan_w)
        k = torch.empty(batch_size, dtype=torch.long, device=device)
        for cc in live:
            sel = torch.from_numpy(pick == cc).to(device)
            n_sel = int(sel.sum())
            if n_sel:
                blks = blocks_by_chan[cc]
                k[sel] = blks[torch.randint(0, len(blks), (n_sel,), device=device)]
        with torch.no_grad():
            z = controller.block_state(x)
            x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
            target = controller.block_state(x2) - z
        pred = fm(z, k)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            cos = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1).mean().item()
            print(f"    fm step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cos:.3f}")
    fm.eval()
    return fm


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=14400, memory=32768)
def share_probe(share_top: int, seed: int = 1, v: int = 8, s: int = 2, state_dim: int = 96,
                tree_depth: int = 4, struct_ms: str = "2,4", noise_blocks: str = "1,1",
                controller_steps: int = 12_000, generator_steps: int = 12_000,
                fm_steps: int = 12_000, batch_size: int = 256, n_corrupt: int = 3,
                edit_budget: int = 6, n_probe_states: int = 1024,
                arms: str = ",".join(ALLOC_ARMS), fm_arch: str = "block",
                quick: bool = False):
    """G3 at one sharing depth: train one FM per fixed allocation, read tree FM error.

    Controller and generator are trained ONCE and shared by every arm, so the arms differ only
    in which transitions their FM was fit on -- the same discipline as the ladder's single warm
    FM fork.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = fm_steps = 400
        n_probe_states = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    # structA is depth- and m-matched to the tree so that "level i" means the same scale in both
    # -- the precondition `share_rules_top` enforces. structB stays shallow and NEVER shared: it
    # is the deeply-unrelated control the sweep needs to attribute any effect to sharing.
    spec = make_spec(tree_depth=tree_depth, struct_depths=[tree_depth, 2],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")],
                     struct_shares=[int(share_top), 0], rule_seed_offset=seed - 1)
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    names = tb["channel_names"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)
    print(f"\n### share_top={share_top} seed={seed}: T={layout['total_len']} tokens / "
          f"{tb['n_blocks']} blocks, rule_seed_offset={seed - 1}, device={device}")
    for ch in layout["channels"]:
        print(f"  {ch['name']:9s} {ch['kind']:6s} blocks[{ch['blk0']:2d}:{ch['blk1']:2d}]"
              + (f" depth={ch['depth']} m={ch['m']}" if ch["rules"] is not None else "")
              + (f" share_top={ch['share_top']}" if ch.get("share_top") else ""))

    Controller, Generator = _build_rich_controller(), _build_generator()
    ARCHES = {"block": _build_block_fm, "shared_marker": _build_shared_marker_fm,
              "channel_local": build_channel_local_fm}
    if fm_arch not in ARCHES:
        raise ValueError(f"fm_arch must be one of {sorted(ARCHES)} (got {fm_arch!r})")
    BlockFM = ARCHES[fm_arch]()
    T, n_blocks = layout["total_len"], tb["n_blocks"]
    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))

    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    del train_leaves, train_roots, pool

    # one frozen probe set, shared by every arm at this sharing depth
    x_probe, _ = sample_states(layout, tb, generator, n=n_probe_states, n_corrupt=n_corrupt,
                               presteps=2, seed=seed + 41, device=device, render=render, gen=gen)

    align = controller_alignment(controller, layout, tb, device,
                                 n=2048 if quick else 8192, steps=100 if quick else 600)
    if align:
        print("  G4 encoder alignment (linear feature probe fit on TREE blocks):")
        for tag, r in align.items():
            print(f"    {tag:9s} eval tree {r['fit_on_tree_eval_tree']:.3f} | "
                  f"eval structA {r['fit_on_tree_eval_structA']:.3f} | "
                  f"chance {r['chance']:.3f}")
    # the channel-local FM subtracts a per-block latent mean; estimate it once, label-free
    center = (block_latent_mean(controller, layout, tb, n=4096 if quick else 20_000,
                                seed=seed + 61, device=device)
              if fm_arch == "channel_local" else None)

    out = {}
    for ai, arm in enumerate([a for a in arms.split(",") if a]):
        w = alloc_vector(arm, names)
        steps = fm_steps // 2 if arm in HALF_BUDGET_ARMS else fm_steps
        print(f"  arm {arm:14s} w=" + " ".join(f"{n[:6]}={w[i]:.2f}" for i, n in enumerate(names))
              + f"  steps={steps}")
        fm = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
        if fm_arch == "channel_local":
            fm.configure(tb).set_center(center)
        # arm-indexed (not hashed) data seed: PYTHONHASHSEED randomisation would make the
        # collection stream differ run to run, which is the one thing this comparison cannot have
        train_block_fm_alloc(fm, controller, generator, layout, tb, chan_w=w, n_steps=steps,
                             batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                             lr=1e-3, seed=seed + 51 + 977 * ai,
                             render=render, gen=gen, device=device)
        errs = per_channel_error(fm, controller, generator, tb, x_probe, render=render,
                                 gen=gen, seed=seed + 88)
        out[arm] = {
            "alloc": {names[c]: float(w[c]) for c in range(len(names))},
            "fm_steps": steps,
            "tree_fm_err": errs[tb["tree_channel"]]["nmse_acted"],
            "err_by_channel": {names[c]: errs[c]["nmse_acted"] for c in range(len(names))},
            "err_by_channel_all_blocks": {names[c]: errs[c]["nmse"] for c in range(len(names))},
        }
        print(f"    -> tree FM err {out[arm]['tree_fm_err']:.4f}")
        del fm

    return {"share_top": int(share_top), "seed": int(seed), "fm_arch": fm_arch,
            "encoder_alignment": align,
            "channels": [{k: ch[k] for k in ("name", "kind", "depth", "m", "blk0", "blk1",
                                             "share_top")} for ch in layout["channels"]],
            "total_len": T, "n_blocks": n_blocks,
            "arms": out, "elapsed_seconds": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=8192)
def geometry_check(shares: str = "0,1,2,3,4", seed: int = 1, v: int = 8, s: int = 2,
                   tree_depth: int = 4, struct_ms: str = "2,4", noise_blocks: str = "1,1",
                   arms: str = ",".join(ALLOC_ARMS), fm_arch: str = "block",
                   tag: str = "v1", quick: bool = False,
                   controller_steps: int = 12_000, generator_steps: int = 12_000,
                   fm_steps: int = 12_000, n_probe_states: int = 1024):
    """G1-G3 across the sharing-depth sweep. DGP checks locally, one GPU probe per depth."""
    share_list = [int(x) for x in shares.split(",") if x != ""]
    started = time.time()

    # ---- G1 / G2: pure-DGP checks, no model involved ------------------------------------
    dgp = {}
    for k in share_list:
        spec = make_spec(tree_depth=tree_depth, struct_depths=[tree_depth, 2],
                         struct_ms=[int(x) for x in struct_ms.split(",")],
                         noise_blocks=[int(x) for x in noise_blocks.split(",")],
                         struct_shares=[k, 0], rule_seed_offset=seed - 1)
        layout = make_layout(v, s, spec)
        irr = check_structural_irrelevance(layout, n=1024 if quick else 4096, seed=7)
        n_stat = 4000 if quick else 20000
        dgp[k] = {"irrelevance": irr, "surface": surface_stats(layout),
                  "levels": level_sibling_stats(layout, n=n_stat, seed=13),
                  "marginals": check_marginals(layout, n=n_stat, seed=11)}
        ok = irr["roots_identical"] and irr["valid_identical"] and irr["dp_identical"]
        sA = dgp[k]["surface"]["structA"]
        tv = dgp[k]["levels"]["structA"]["sibling_tv_vs_tree"]
        print(f"G1/G2 share_top={k}: P1 {'PASS' if ok else '*** FAIL ***'} "
              f"(d* mean {irr['mean_dp_cost']:.2f}) | structA surface overlap "
              f"{sA['overlap_with_tree']:.3f} (chance {dgp[k]['surface']['_chance_overlap']:.3f}), "
              f"bottom identical={sA['bottom_table_identical']} | sibling TV vs tree by level "
              + "[" + " ".join(f"{t:.3f}" for t in tv) + "]")
        if not ok:
            raise RuntimeError(f"P1 structural irrelevance BROKEN at share_top={k}: {irr}")

    # ---- G3: one GPU probe per sharing depth, in parallel --------------------------------
    handles = [(k, share_probe.spawn(share_top=k, seed=seed, v=v, s=s, tree_depth=tree_depth,
                                     struct_ms=struct_ms, noise_blocks=noise_blocks,
                                     arms=arms, fm_arch=fm_arch, quick=quick,
                                     controller_steps=controller_steps,
                                     generator_steps=generator_steps, fm_steps=fm_steps,
                                     n_probe_states=n_probe_states))
               for k in share_list]
    probes = {}
    for k, h in handles:
        probes[k] = h.get()
        print(f"G3 share_top={k} done in {probes[k]['elapsed_seconds']:.0f}s")

    # ---- the transfer curve --------------------------------------------------------------
    arm_list = [a for a in arms.split(",") if a]
    print(f"\n{'=' * 88}\n=== G3 -- tree FM error by fixed allocation, vs sharing depth "
          f"(seed {seed}) ===\n{'=' * 88}")
    print(f"{'share_top':>9s} " + " ".join(f"{a:>13s}" for a in arm_list))
    curve = {}

    def _d(a, x, y):
        """`x - y` in tree FM error, or None if either arm was not run."""
        return (a[x]["tree_fm_err"] - a[y]["tree_fm_err"]) if x in a and y in a else None

    for k in share_list:
        a = probes[k]["arms"]
        # every `_value` holds tree-block transitions FIXED; negative means the added data HELPED
        add_A = _d(a, "tree_structA", "tree_half")
        dbl = _d(a, "tree_only", "tree_half")
        curve[k] = {
            "tree_err": {arm: a[arm]["tree_fm_err"] for arm in arm_list},
            "structA_value": add_A,
            "structB_value": _d(a, "tree_structB", "tree_half"),
            "noise_value": _d(a, "tree_noise", "tree_half"),
            "doubling_value": dbl,
            # in units of "a doubling of tree data": 0 = worthless, 1 = as good as tree data
            "structA_exchange_rate": (add_A / dbl) if (add_A is not None and dbl) else None,
            # the coverage-confounded pure-transfer arms, kept for the narrower question
            "structA_only_cost": _d(a, "structA_only", "tree_only"),
            "structB_only_cost": _d(a, "structB_only", "tree_only"),
            "uniform_cost": _d(a, "uniform", "tree_only"),
        }
        print(f"{k:>9d} " + " ".join(f"{a[arm]['tree_fm_err']:>13.4f}" for arm in arm_list))

    if probes[share_list[0]].get("encoder_alignment"):
        print(f"\n  G4 -- encoder alignment: a linear feature probe FIT ON TREE BLOCKS, read on")
        print(f"  structA blocks. At share_top=4 the grammars are identical, so a probe that does")
        print(f"  not transfer means the ENCODER is the barrier, not the forward model.")
        print(f"{'share_top':>9s} {'tree (raw)':>12s} {'structA (raw)':>14s} "
              f"{'tree (ctr)':>12s} {'structA (ctr)':>14s}")
        for k in share_list:
            al = probes[k]["encoder_alignment"]
            print(f"{k:>9d} {al['raw']['fit_on_tree_eval_tree']:>12.3f} "
                  f"{al['raw']['fit_on_tree_eval_structA']:>14.3f} "
                  f"{al['centered']['fit_on_tree_eval_tree']:>12.3f} "
                  f"{al['centered']['fit_on_tree_eval_structA']:>14.3f}")

    print(f"\n  value ADDED to a fixed tree-data budget (negative = helped), and the exchange rate")
    print(f"  against doubling the tree data itself:")
    print(f"{'share_top':>9s} {'+structA':>12s} {'+structB':>12s} {'+noise':>12s} "
          f"{'2x tree':>12s} {'A rate':>9s}")
    for k in share_list:
        c = curve[k]
        cells = [c["structA_value"], c["structB_value"], c["noise_value"], c["doubling_value"]]
        print(f"{k:>9d} " + " ".join(f"{x:>+12.4f}" if x is not None else f"{'-':>12s}"
                                     for x in cells)
              + (f" {c['structA_exchange_rate']:>9.2f}"
                 if c["structA_exchange_rate"] is not None else f" {'-':>9s}"))

    metrics = {
        "config": {"shares": share_list, "seed": seed, "v": v, "s": s,
                   "tree_depth": tree_depth, "struct_depths": [tree_depth, 2],
                   "struct_ms": struct_ms, "noise_blocks": noise_blocks, "arms": arm_list,
                   "fm_arch": fm_arch, "rule_seed_offset": seed - 1, "quick": quick},
        "dgp": dgp, "probes": probes, "curve": curve,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/partial_hetero/geometry_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics
