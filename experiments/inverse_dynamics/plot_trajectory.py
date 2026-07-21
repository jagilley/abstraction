"""Plot: grokking induces inverse-self-model ill-posedness.

Reads results/p97_trajectory/results.json and makes a 2-panel figure:
  A: forward vs inverse test R^2 (widest self-model) vs training epoch, with
     test accuracy overlaid -- forward stays ~1.0, inverse collapses as the model
     groks.
  B: mechanism -- inverse R^2 (logits->h1) vs the readout's within-fiber variance
     fraction (how much pair info the representation has thrown away). Monotone.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "results", "p97_trajectory", "results.json")) as f:
    R = json.load(f)

traj = R["trajectory"]
order = sorted(traj, key=lambda k: traj[k]["epoch"])
epochs = [traj[k]["epoch"] for k in order]
test_acc = [traj[k]["test_acc"] for k in order]


def r2(k, tag, direction):
    # widest self-model = last width in the sweep
    return traj[k]["sweep"][tag][direction][-1]["test"]["r2"]


def within(k, rep):
    return traj[k]["fiber_diagnostics"][rep]["within_fiber_frac"]


fwd_log = [r2(k, "h1->logits", "forward") for k in order]
inv_log = [r2(k, "h1->logits", "inverse") for k in order]
fwd_h2 = [r2(k, "h1->h2", "forward") for k in order]
inv_h2 = [r2(k, "h1->h2", "inverse") for k in order]
logit_within = [within(k, "logits") for k in order]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# ---- Panel A ----
axA.plot(epochs, fwd_log, "o-", color="#1b7837", lw=2, label="forward  h1→logits")
axA.plot(epochs, fwd_h2, "o--", color="#5aae61", lw=2, label="forward  h1→h2")
axA.plot(epochs, inv_h2, "s--", color="#d6604d", lw=2, label="inverse  h2→h1")
axA.plot(epochs, inv_log, "s-", color="#b2182b", lw=2.5, label="inverse  logits→h1")
axA.set_xlabel("training epoch")
axA.set_ylabel("self-model test R²  (widest model)")
axA.set_ylim(-0.05, 1.05)
axA.set_title("Grokking induces inverse ill-posedness\n(forward stays perfect; inverse collapses)")
axA.legend(loc="center left", fontsize=9)
axA.grid(True, alpha=0.3)

axT = axA.twinx()
axT.plot(epochs, test_acc, ":", color="#777777", lw=1.5)
axT.fill_between(epochs, 0, test_acc, color="#999999", alpha=0.08)
axT.set_ylabel("main-model test accuracy (grokking)", color="#777777")
axT.set_ylim(-0.05, 1.05)
axT.tick_params(axis="y", labelcolor="#777777")

# ---- Panel B ----
axB.plot(logit_within, inv_log, "s-", color="#b2182b", lw=2.5)
for x, y, e in zip(logit_within, inv_log, epochs):
    axB.annotate(f"{e//1000}k", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
axB.set_xlabel("readout within-fiber variance fraction\n(← more compressed onto c = a+b)")
axB.set_ylabel("inverse R²  (logits→h1)")
axB.invert_xaxis()  # compression increases to the right
axB.set_title("Mechanism: inverse dies as the readout\ncompresses away the pair identity")
axB.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, "results", "p97_trajectory", "trajectory.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
