"""The quantizer: a SHARED k-means codebook over 16x16-px patches.

`T[1]` on this substrate is a first-regime object -- a set of reconstruction-optimal patch
summaries built by an objective that knows nothing about parts (memo Sec.1 item 1, Sec.9 "the
alphabet may not factor"). Its reconstruction error is the floor everything downstream is
bounded by, so it is measured per style and reported beside every number.

k-means rather than a VQ-VAE, deliberately: the decoder is a LOOKUP (centroid -> pixels), so
"the quantizer's error" is exactly the error of the alphabet and not of a learned decoder that
could hide structure the tokens do not carry. A VQ-VAE would make the floor a property of two
things at once. The memo names both and calls k-means "the simplest form"; the neural form is
the sibling if the floor turns out to bind.

The codebook is fit on `plant_train` seeds ONLY, so every held-out swatch is held out from the
quantizer as well as from the token model.
"""

import numpy as np
import torch

PATCH = 16
GRID = 16


def to_patches(imgs, patch=PATCH):
    """(N, H, W, 3) uint8 -> (N, grid*grid, patch*patch*3) float32 in [0, 1]."""
    x = np.asarray(imgs, np.float32) / 255.0
    N, H, W, C = x.shape
    g = H // patch
    x = x.reshape(N, g, patch, g, patch, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(N, g * g, patch * patch * C)


def from_patches(p, patch=PATCH, grid=GRID):
    """(N, grid*grid, patch*patch*3) -> (N, H, W, 3) float32."""
    N = p.shape[0]
    x = p.reshape(N, grid, grid, patch, patch, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(N, grid * patch, grid * patch, 3)


@torch.no_grad()
def assign(X, C, bs=8192):
    """Nearest centroid. X (n, d), C (K, d) torch on device -> (n,) long."""
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for a in range(0, X.shape[0], bs):
        chunk = X[a:a + bs]
        d = cn[None] - 2.0 * (chunk @ C.T)
        out[a:a + bs] = d.argmin(1)
    return out


@torch.no_grad()
def kmeans(X, K, iters=40, seed=0, dev="cuda", verbose=True):
    """Lloyd with k-means++-lite init (uniform sample of distinct rows). X: (n, d) torch."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)[:K]
    C = X[perm.to(X.device)].clone()
    for t in range(iters):
        a = assign(X, C)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K, device=X.device)
        newC.index_add_(0, a, X)
        cnt.index_add_(0, a, torch.ones(n, device=X.device))
        dead = cnt == 0
        newC[~dead] = newC[~dead] / cnt[~dead][:, None]
        if dead.any():                                    # respawn on random rows
            ridx = torch.randperm(n, generator=g)[:int(dead.sum())].to(X.device)
            newC[dead] = X[ridx]
        shift = float((newC - C).pow(2).sum(1).mean())
        C = newC
        if verbose and (t % 10 == 0 or t == iters - 1):
            print(f"      kmeans K={K} it {t:3d} shift {shift:.3e}", flush=True)
        if shift < 1e-9:
            break
    return C


@torch.no_grad()
def encode(imgs, C, dev="cuda", bs=64, patch=PATCH, grid=GRID):
    """(N, H, W, 3) uint8 -> (N, grid*grid) int64 codes, and the per-image patch MSE."""
    codes, mses = [], []
    for a in range(0, len(imgs), bs):
        P = torch.as_tensor(to_patches(imgs[a:a + bs], patch), device=dev)
        n, g2, d = P.shape
        flat = P.reshape(-1, d)
        idx = assign(flat, C)
        rec = C[idx]
        mses.append(((rec - flat) ** 2).mean(1).reshape(n, g2).mean(1).cpu().numpy())
        codes.append(idx.reshape(n, g2).cpu().numpy())
    return np.concatenate(codes), np.concatenate(mses)


def psnr(mse):
    return float(10.0 * np.log10(1.0 / max(float(mse), 1e-12)))
