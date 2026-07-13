"""Ceiling-vs-depth and ceiling-vs-m comparison from the per-level expansion grid."""
import numpy as np
import matplotlib.pyplot as plt

levels = np.arange(6)

# Capacity variants at v8/m2: depth sweep (fixed 256D) + depth x width control.
# 10 models spanning ~30K to ~6.3M params. chance = 1/8
capacity = {
    "2L/256D": [0.827, 0.440, 0.435, 0.281, 0.287, 0.241],
    "4L/256D": [0.830, 0.444, 0.435, 0.282, 0.293, 0.244],
    "6L/256D": [0.830, 0.444, 0.439, 0.281, 0.288, 0.240],
    "8L/256D": [0.831, 0.444, 0.440, 0.279, 0.294, 0.244],
    "2L/32D": [0.809, 0.415, 0.417, 0.244, 0.267, 0.231],
    "8L/32D": [0.823, 0.439, 0.432, 0.278, 0.288, 0.240],
    "2L/64D": [0.813, 0.428, 0.425, 0.267, 0.274, 0.245],
    "8L/64D": [0.828, 0.442, 0.438, 0.283, 0.291, 0.243],
    "2L/128D": [0.822, 0.435, 0.431, 0.272, 0.280, 0.238],
    "8L/128D": [0.830, 0.443, 0.438, 0.282, 0.293, 0.240],
}
# Synonymity sweep: v8, 8L/256D fixed. chance = 1/8
syn = {
    "m=2": [0.831, 0.444, 0.440, 0.279, 0.294, 0.244],
    "m=4": [0.592, 0.310, 0.209, 0.199, 0.200, 0.209],
    "m=6": [0.476, 0.193, 0.175, 0.160, 0.162, 0.167],
    "m=8": [0.343, 0.180, 0.179, 0.189, 0.195, 0.195],
}
# Root proof, plotted as accuracy / chance so v8 and v16 are comparable
root = {
    "v=8, m=2  (chance 1/8)":  (np.array([0.831,0.444,0.440,0.279,0.294,0.244]) / (1/8)),
    "v=16, m=2 (chance 1/16)": (np.array([0.895,0.416,0.371,0.340,0.252,0.216]) / (1/16)),
}

plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel A: capacity invariance (all 10 models overlap)
ax = axes[0]
for k, v in capacity.items():
    ax.plot(levels, v, "-o", color="#1f4e79", alpha=0.35, lw=1.5, ms=3)
ax.axhline(1/8, ls="--", c="0.5", lw=1)
ax.text(5, 1/8 + 0.01, "chance", ha="right", va="bottom", color="0.4", fontsize=9)
ax.set_title("Capacity is irrelevant (v=8, m=2)\n10 models, depth 2–8L × width 32–256D (~30K–6.3M params)", fontsize=10)
ax.set_xlabel("hierarchy level  (0 = leaves … 5 = root)")
ax.set_ylabel("next-token accuracy")
ax.set_ylim(0, 1)

# Panel B: m (fan out)
ax = axes[1]
oranges = plt.cm.YlOrRd(np.linspace(0.45, 0.9, 4))
for (k, v), c in zip(syn.items(), oranges):
    ax.plot(levels, v, "-o", color=c, label=k, lw=2, ms=5)
ax.axhline(1/8, ls="--", c="0.5", lw=1)
ax.text(5, 1/8 + 0.01, "chance", ha="right", va="bottom", color="0.4", fontsize=9)
ax.set_title("Synonymity SETS the ceiling\n(v=8, capacity fixed 8L/256D)", fontsize=11)
ax.set_xlabel("hierarchy level  (0 = leaves … 5 = root)")
ax.set_ylabel("next-token accuracy")
ax.set_ylim(0, 1); ax.legend(title="synonyms/rule", frameon=False)

# Panel C: reaches-root, accuracy relative to chance
ax = axes[2]
cols = {"v=8, m=2  (chance 1/8)": "#7a7a7a", "v=16, m=2 (chance 1/16)": "#1f77b4"}
for k, v in root.items():
    ax.plot(levels, v, "-o", color=cols[k], label=k, lw=2, ms=5)
ax.axhline(1.0, ls="--", c="0.5", lw=1)
ax.text(0, 1.0 + 0.2, "chance", ha="left", va="bottom", color="0.4", fontsize=9)
ax.set_title("Lower occupancy → wave reaches the root\n(8L/256D)", fontsize=11)
ax.set_xlabel("hierarchy level  (0 = leaves … 5 = root)")
ax.set_ylabel("accuracy / chance  (× above chance)")
ax.legend(frameon=False)

fig.suptitle("The composition ceiling is set by synonymity (m), not model depth", fontsize=13, y=1.02)
fig.tight_layout()
out = "~/scratch/ceiling_depth_vs_m.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
