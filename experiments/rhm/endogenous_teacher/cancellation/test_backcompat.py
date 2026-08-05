"""Regression test: rhm/model.py's cancellation additions are bit-identical on the old path.

`GPT.forward` gained `cerebellar_mode` / `cerebellar_readd_block` for the cancellation cut.
Every prior RHM result (rhm_latent_loop, residual_decomposition, all of ratchet/,
confabulation) goes through the summation path and passes neither argument, so this
asserts that path is UNCHANGED to the last bit -- not merely close.

`old_forward` below is a verbatim transcription of the pre-change loop (see
`git log -p experiments/rhm/model.py`). Comparisons use torch.equal, not allclose.

Run:
  modal run -m rhm.endogenous_teacher.cancellation.test_backcompat::test_backcompat
"""

import modal

from rhm.shared import volume, DATA_DIR

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-backcompat", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800)
def test_backcompat(v: int = 16, seq_len: int = 64, n_layer: int = 8, n_head: int = 8,
                    n_embd: int = 256, batch: int = 8, seed: int = 0):
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT

    device = "cuda"

    def old_forward(self, idx, targets=None, return_intermediates=False,
                    cerebellar_fn=None, cerebellar_input_block=0,
                    cerebellar_inject_block=1):
        """Verbatim pre-change body."""
        B, Tl = idx.size()
        assert Tl <= self.block_size
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(Tl, device=idx.device))
        x = self.transformer.drop(tok_emb + pos_emb)
        intermediates = {}
        if return_intermediates:
            intermediates["post_embed"] = x
        cerebellar_injection = None
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if return_intermediates:
                intermediates[f"post_block{i}"] = x
            if cerebellar_fn is not None and i == cerebellar_input_block:
                cerebellar_injection = cerebellar_fn(x)
            if cerebellar_injection is not None and i == cerebellar_inject_block:
                x = x + cerebellar_injection
                cerebellar_injection = None
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_intermediates:
            return logits, loss, intermediates
        return logits, loss

    torch.manual_seed(seed)
    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device).eval()
    torch.manual_seed(seed + 1)
    proj = torch.nn.Linear(n_embd, n_embd).to(device).eval()   # a non-trivial injection
    x = torch.randint(0, v, (batch, seq_len), device=device)
    y = torch.randint(0, v, (batch, seq_len), device=device)

    def cb(act):
        return proj(act)

    def same(a, b):
        return (a is None and b is None) or torch.equal(a, b)

    failures = []

    def check(name, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)

    print("\n[1] summation path, with injection, with intermediates")
    with torch.no_grad():
        lo, so, io = old_forward(model, x, y, True, cb, 0, 1)
        ln, sn, iN = model(x, y, return_intermediates=True, cerebellar_fn=cb,
                           cerebellar_input_block=0, cerebellar_inject_block=1)
    check("logits", same(lo, ln))
    check("loss", torch.equal(so, sn))
    check("intermediate keys", set(io) == set(iN))
    check("intermediate values", all(same(io[k], iN[k]) for k in io))

    print("\n[2] summation path, no injection (the plain forward)")
    with torch.no_grad():
        lo2, so2 = old_forward(model, x, y)
        ln2, sn2 = model(x, y)
    check("logits", same(lo2, ln2))
    check("loss", torch.equal(so2, sn2))

    print("\n[3] non-default inject/input blocks")
    with torch.no_grad():
        lo3, _, io3 = old_forward(model, x, y, True, cb, 2, 5)
        ln3, _, iN3 = model(x, y, return_intermediates=True, cerebellar_fn=cb,
                            cerebellar_input_block=2, cerebellar_inject_block=5)
    check("logits", same(lo3, ln3))
    check("intermediate values", all(same(io3[k], iN3[k]) for k in io3))

    print("\n[4] cancel with a ZERO injection == plain forward (zero-init gate at step 0)")
    with torch.no_grad():
        lz, _ = model(x, y, cerebellar_fn=lambda a: torch.zeros_like(a),
                      cerebellar_mode="cancel", cerebellar_readd_block=6)
    check("logits == plain", same(lz, ln2))

    print("\n[5] cancel is genuinely different from add, and exposes _eff")
    with torch.no_grad():
        lc, _, ic = model(x, y, return_intermediates=True, cerebellar_fn=cb,
                          cerebellar_mode="cancel", cerebellar_readd_block=6)
    check("differs from summation", not torch.equal(lc, ln))
    check("post_block6_eff present", "post_block6_eff" in ic)
    check("summation has no _eff key", not any(k.endswith("_eff") for k in iN))
    # the deviation stream really is (state - injection) at the re-add block
    with torch.no_grad():
        _, _, ip = model(x, y, return_intermediates=True)
        inj = cb(ip["post_block0"])
    check("_eff == deviation + injection",
          torch.allclose(ic["post_block6_eff"], ic["post_block6"] + inj, atol=1e-5))

    print("\n[6] misconfigured cancel raises rather than silently dropping the forecast")
    try:
        model(x, y, cerebellar_fn=cb, cerebellar_mode="cancel", cerebellar_readd_block=0)
        check("readd < inject asserts", False)
    except AssertionError:
        check("readd < inject asserts", True)
    try:
        model(x, y, cerebellar_fn=cb, cerebellar_mode="cancel",
              cerebellar_readd_block=n_layer)
        check("readd >= n_layer asserts", False)
    except AssertionError:
        check("readd >= n_layer asserts", True)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED: {failures}")
        raise SystemExit(1)
    print("ALL PASS -- the summation path is bit-identical; cancel is additive-only.")
    return {"failures": failures}
