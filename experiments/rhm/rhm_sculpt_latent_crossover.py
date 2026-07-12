"""Plot the latent-vs-token crossover under two lossy channels (PO + stochastic).
Numbers taken directly from the full m=2 run logs (L=4, c=3, 1024 eval instances)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

widths = [1, 16, 64, 256]
# colorblind-safe (Okabe-Ito-ish), light->dark with width
colors = {1: "#E69F00", 16: "#56B4E9", 64: "#0072B2", 256: "#111111"}

# ---- Partial observability (rhm_sculpt_latent_po) : gap = latent - token ----
po_p = [0.0, 0.25, 0.5, 0.75, 0.9]
po_gap = {
    1:   [-0.204, -0.230, -0.119, -0.035, -0.009],
    16:  [-0.117, -0.098, -0.059, -0.009, +0.003],
    64:  [-0.087, -0.023, +0.021, +0.030, +0.029],
    256: [-0.045, +0.021, +0.037, +0.043, +0.040],
}
po256_token = [0.544, 0.460, 0.297, 0.149, 0.062]
po256_latent = [0.499, 0.480, 0.334, 0.192, 0.102]

# ---- Stochastic dynamics (rhm_sculpt_latent_stoch) : gap = latent - token ----
st_q = [0.0, 0.1, 0.25, 0.5, 0.75]
st_gap = {
    1:   [-0.204, -0.191, -0.161, -0.105, -0.066],
    16:  [-0.117, -0.003, +0.028, +0.009, -0.002],
    64:  [-0.087, +0.055, +0.036, +0.048, +0.024],
    256: [-0.045, +0.021, +0.075, +0.050, +0.034],
}
st256_token = [0.544, 0.256, 0.159, 0.103, 0.061]
st256_latent = [0.499, 0.276, 0.234, 0.152, 0.095]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Planning in latents beats token-space exactly when the token channel goes lossy\n"
             "RHM sculpting, L=4 m=2 c=3, single seed; gap = latent_success − token_success (paired, 1024 instances)",
             fontsize=11)

# top-left: PO gap
ax = axes[0, 0]
for w in widths:
    ax.plot(po_p, po_gap[w], "-o", color=colors[w], label=f"beam {w}", markersize=5)
ax.axhline(0, color="0.4", lw=1, ls="--")
ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.06, color="#0072B2", alpha=0.04)
ax.set_title("(a) Partial observability: latent carries belief", fontsize=10)
ax.set_xlabel("occlusion probability  p")
ax.set_ylabel("gap  (latent − token)")
ax.text(0.02, 0.9, "latent wins ↑", transform=ax.transAxes, fontsize=8, color="#0072B2")
ax.text(0.02, 0.04, "token wins ↓", transform=ax.transAxes, fontsize=8, color="#B00")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.2)

# top-right: stochastic gap
ax = axes[0, 1]
for w in widths:
    ax.plot(st_q, st_gap[w], "-o", color=colors[w], label=f"beam {w}", markersize=5)
ax.axhline(0, color="0.4", lw=1, ls="--")
ax.set_title("(b) Stochastic actuator: latent ranks by stable FM, not one noisy sample", fontsize=10)
ax.set_xlabel("slip probability  q")
ax.set_ylabel("gap  (latent − token)")
ax.text(0.02, 0.9, "latent wins ↑", transform=ax.transAxes, fontsize=8, color="#0072B2")
ax.text(0.02, 0.04, "token wins ↓", transform=ax.transAxes, fontsize=8, color="#B00")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)

# bottom-left: PO absolute levels at width 256 (mechanism)
ax = axes[1, 0]
ax.plot(po_p, po256_token, "-s", color="#B00", label="token beam", markersize=5)
ax.plot(po_p, po256_latent, "-o", color="#0072B2", label="latent beam", markersize=5)
ax.set_title("(c) PO, beam 256: token collapses, latent decays gently", fontsize=10)
ax.set_xlabel("occlusion probability  p")
ax.set_ylabel("success (possible-set)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# bottom-right: stochastic absolute levels at width 256 (mechanism)
ax = axes[1, 1]
ax.plot(st_q, st256_token, "-s", color="#B00", label="token beam", markersize=5)
ax.plot(st_q, st256_latent, "-o", color="#0072B2", label="latent beam", markersize=5)
ax.set_title("(d) Stochastic, beam 256: same divergence", fontsize=10)
ax.set_xlabel("slip probability  q")
ax.set_ylabel("success (possible-set)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "~/scratch/latent_crossover.png"
fig.savefig(out, dpi=140)
print("saved", out)
