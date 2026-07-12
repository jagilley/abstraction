# RHM Sculpting: can editing (not revealing) turn RHM into a genuine control task?

*Stand-alone note — self-contained, deliberately **not** indexed in `FILES.md` or linked from the README system, so the finding can be evaluated on its own merits. Exploratory, single rule-seed-family (seeds 0 and 1). Date: 2026-07-11. Script: [`rhm_sculpt_precheck.py`](rhm_sculpt_precheck.py).*

> **Update (2026-07-11):** the learned experiment this note points to has been built and run — a learned beam planner captures the coordination prize (matches the strong reflex, monotone in beam width). See [RHM_SCULPTING_README.md](RHM_SCULPTING_README.md) for the full learned arc; this note remains the perfect-simulator pre-check it references.

---

## TL;DR

A "value-of-information" version of the Random Hierarchy Model (an agent reveals hidden blocks to guess a hidden root) is an **inference task in disguise**: choosing the best action *is* forecasting its value, so acting ≈ planning and there is no separate world-model to learn. This note asks whether a small change — let the agent **edit observed leaves toward a target root** instead of revealing hidden ones — creates a genuine **control** task where planning over a learned dynamics model beats a reflex (acting ≠ planning).

Using the known grammar as a **perfect simulator** (no learning), a coordinated planner solves 100% of instances while a myopic reflex — even a strong, target-aware one — fails on a large and **depth-scaling** fraction: up to **0.66** of solvable instances at depth L=5. The requirement for coordination is real and grows with the hierarchy. Two knobs control it: **rule multiplicity `m`** (local coupling) and **depth `L`** (coordination across levels). This validates the task design; it is a statement about the *task*, not yet about a learned model.

---

## Background (self-contained)

**The Random Hierarchy Model (RHM)** (Cagnetta & Wyart, 2024) generates a string of `s^L` leaf tokens from a single **root** symbol by expanding top-down through `L` levels: each symbol picks one of `m` production rules, each rule mapping the symbol to an `s`-tuple of child symbols; after `L` expansions you have the leaves. Rules are fixed once. Here `v` = vocabulary size, `s` = branching, so a "block" is `s` adjacent leaves produced by one bottom-level node. We use the *distinct*-tuple construction (the `m` productions of a feature are distinct `s`-tuples, but tuples can collide across features, so parsing leaves back to a root is genuinely ambiguous).

**Why this note exists.** A prior line turned RHM into an *active* task: the agent sees a masked sequence, **reveals** a budget of blocks, and predicts the hidden root. That task is a pure **sensing POMDP** — actions change only the agent's *knowledge* of a static hidden root. On such a task, acting optimally means judging which reveal is most informative, which is exactly the quantity a planner would forecast (value of information). So **acting ≈ planning**, and a companion result found that internalizing a forward model (co-training a self-forecast to reorganize the representation) has *no headroom* there: any competent belief already carries the plannable signal, and a raw policy plans as well as an explicit forecaster.

The open question this note takes up: **what would make RHM a control domain where acting ≠ planning** — where a forward/dynamics model is a genuinely separate thing a reflex policy can skip, so that internalizing it could reorganize the representation (as it does in motor-control tasks)?

---

## The hypothesis

Change what an action *does*: from **revealing a hidden block** (sensing a static latent) to **editing an observed leaf** toward a **target root `r*`** (transforming a persistent state). Three ingredients should then be present together:

1. **A persistent, agent-transformed state** — edits mutate the world, not just the belief.
2. **Content-independent, deterministic consequences** — editing an *observed* token is deterministic (unlike revealing a *hidden* one, whose consequence depends on unknown content), so a forward model can forecast it. The non-trivial thing the model must learn is the grammar's **bottom-up parse** (how a local edit changes the global root).
3. **Coupling that makes greedy suboptimal** — otherwise a myopic reflex solves the task and we are back to acting ≈ planning. RHM's hierarchy supplies coupling: a valid subtree needs its leaves to be a *legal production*, and a parent feature couples its whole subtree, so hitting a global target needs *coordinated* multi-edit changes.

Ingredient 3 is the one that can silently fail, so it must be **measured before any learning**. That is what this note does.

---

## The experiment: a perfect-simulator pre-check

Everything below uses the *known* grammar directly (exact combinatorics, no neural network). The point is to establish whether the *task* rewards planning over a reflex; only if it does is a learned version worth building.

**Instances.** Sample a root `r*`, generate one valid `r*`-derivation (the "clean" leaves), then **corrupt `c` randomly chosen blocks** (set their leaves to random symbols). The corrupted sequence is the start. The **budget** is `c·s` token edits (generous: enough to fully rewrite the corrupted blocks). By construction the true edit distance `d* ≤ c·s`.

**Planner (full-horizon optimal, perfect simulator).** An exact dynamic program over the grammar computes `d*` = the minimum number of token edits to turn the start leaves into **any** valid `r*`-derivation. Bottom-up: for each bottom node and candidate feature, the min Hamming distance from its block to that feature's productions; for each internal node and feature, `min over rules of sum over children` of child costs; the answer is the root's cost at `r*`. The planner **solves** an instance iff `d* ≤ budget` — which is ~100% here by construction.

**Reflex A — parse-consistency hill-climber (weak).** A myopic editor: at each step try every single-token edit, keep the one that most increases a *local* score (how many tree nodes parse to any feature), stop on success, budget exhaustion, or a **local optimum** (no strict improvement). It has no multi-step coordination — the best a plain model-free reflex could do. "Stuck" = it hit a local optimum with budget to spare.

**Reflex B — target-aware top-down editor (strong).** A stronger, `r*`-aware reflex: commit to an `r*`-derivation top-down with **1-level-lookahead** rule choices (at each node pick the rule whose `s` child-targets are most already-derivable in the child possible-sets), then pay the *true* edit cost of that committed derivation. It fails iff that cost exceeds budget. It differs from the optimal planner only in that it selects rules **myopically** instead of coordinating them across the whole subtree.

**Metric — lookahead prize.** The fraction of planner-solvable instances (`d* ≤ budget`) that the reflex **fails**. Since the planner solves ~100%, this is essentially the reflex's failure rate on the same instances. High prize ⇒ planning beats reflex ⇒ acting ≠ planning.

**Settings.** `v=8`, `s=2`, `m ∈ {2,3,4}`, `L ∈ {3,4,5}`, `c ∈ {1,2,3}`, ~400 instances/cell, rule seed 0 (seed 1 spot-checked). Sanity assertions pass: an uncorrupted derivation has `d*=0` and parses to its root.

---

## Results

### The lookahead prize survives the *strong* reflex and grows with depth

Strong target-aware reflex (Reflex B) — fraction of planner-solvable instances it fails. This is the conservative measure (Reflex B is a legitimate, fairly strong myopic policy). Rule seed 0; seed 1 in parentheses where spot-checked.

**m = 2** (tight coupling)
| depth | c=1 | c=2 | c=3 |
|---|---|---|---|
| L=3 (8 leaves)  | 0.08 | 0.20 | 0.09 |
| L=4 (16 leaves) | 0.12 | 0.39 (0.35) | 0.42 (0.44) |
| L=5 (32 leaves) | 0.25 | 0.47 (0.45) | **0.66 (0.57)** |

**m = 3** (looser)
| depth | c=1 | c=2 | c=3 |
|---|---|---|---|
| L=3 | 0.07 | 0.14 | 0.04 |
| L=4 | 0.17 | 0.35 | 0.34 |
| L=5 | 0.31 | 0.40 | 0.50 |

The robust trend is **depth**: for a fixed corruption level, the prize rises monotonically with `L` (e.g. m=2, c=3: 0.09 → 0.42 → 0.66). At L=3 the tree is so shallow that 1-level lookahead covers most of the coordination (hence the small, non-monotone L=3 row); by L=5 even the strong reflex leaves 25–66% of the prize on the table. **Coordinating a deep compositional derivation is what the reflex cannot shortcut, and it is what scales with the hierarchy.**

### The weak reflex fails hard and gets stuck at local optima

Full breakdown at **m=2, seed 0** (planner solves 1.00 everywhere by construction):

| L | c | d*_mean | parse-reflex solves (stuck at local opt) | strong reflex solves |
|---|---|---|---|---|
| 3 | 1 | 1.51 | 0.60 (0.27) | 0.92 |
| 3 | 2 | 2.86 | 0.35 (0.60) | 0.80 |
| 3 | 3 | 3.97 | 0.19 (0.80) | 0.92 |
| 4 | 1 | 1.46 | 0.68 (0.24) | 0.88 |
| 4 | 2 | 2.91 | 0.41 (0.52) | 0.61 |
| 4 | 3 | 4.21 | 0.24 (0.74) | 0.58 |
| 5 | 1 | 1.54 | 0.58 (0.27) | 0.75 |
| 5 | 2 | 2.98 | 0.35 (0.54) | 0.54 |
| 5 | 3 | 4.36 | 0.23 (0.73) | 0.34 |

The **stuck** column is the mechanism made visible: the weak reflex makes corrupted blocks locally valid (they parse to *some* feature) but the feature assignment doesn't chain up to `r*`, and no single edit both keeps validity and flips the root — so it halts at a local optimum with budget unspent (up to 80% of instances). Note `d*_mean` is comfortably below budget (e.g. 2.9 < 4), so these are genuine coordination failures, not budget shortfalls.

### Negative control: loosen the coupling and the prize collapses

At **m = 4** (many productions per feature ⇒ many valid `r*`-derivations ⇒ loose constraint), even the *weak* parse-reflex nearly matches the planner: prize 0.04 (L3,c1), 0.06 (L4,c1), 0.14 (L3,c2), 0.10 (L4,c2). Loose coupling ⇒ local repair usually chains to `r*` ⇒ back to acting ≈ planning. This is the control that confirms the effect at m=2/3 is coupling, not an artifact of the setup.

---

## Reading

- **Two independent knobs.** *Tightness* (`m`↓) sharpens **local** coupling — fewer productions per feature, so a local repair rarely chains to `r*`. *Depth* (`L`↑) adds **coordination across levels** — a myopic rule choice compounds errors down the tree. The strong reflex is defeated mostly by depth; the weak reflex by both.
- **What is genuinely control-like here.** The action's consequence for the *sequence* is trivial (apply the edit), but its consequence for the *root* requires modeling the whole bottom-up parse — a real dynamics model. Reaching a target root needs a sequence of edits whose *joint* effect is a valid deep derivation, which a reflex cannot assemble greedily. That is the acting ≠ planning structure the sensing task lacked.

---

## What this does and does not establish

**Does establish (on the evidence above):**
- With a *perfect simulator*, RHM Sculpting has a genuine lookahead prize: coordinated planning solves 100% while myopic reflexes fail on a large fraction.
- The prize **scales with hierarchical depth** and with **tightness** (`m`↓), and **collapses** when coupling is loose (m=4) — a clean, mechanistic dose-response, robust across two rule seeds and two reflex designs.

**Does not establish (be skeptical here):**
- **No learned model yet.** This is a property of the *task*, established by exact combinatorics. It does *not* show that a *learned* controller leaves forward-model-plannability headroom, nor that internalizing a learned forward model reorganizes the representation. That is the next experiment; a perfect-simulator prize is a *necessary* precondition for it, not proof of it.
- **Reflexes are specific.** The prize is defined relative to two hand-designed reflexes. Reflex B is a reasonable strong myopic policy (1-level lookahead), but a cleverer bounded-lookahead reflex would narrow the gap; the honest claim is "a myopic reflex" not "any sub-optimal policy."
- **One corruption regime.** Scattered-block corruption with budget `= c·s`. Other regimes (whole-subtree corruption, random starts, tighter budgets) are untested and could move the numbers.
- **Single rule-seed family.** Seeds 0 and 1 agree closely, but this is not a broad seed sweep; `v` and `s` are fixed at 8 and 2.
- **"Prize" ≠ "learned headroom."** The logical gap between "a reflex fails where a perfect-simulator planner succeeds" and "a *learned* reflex leaves headroom a *learned* forward model can capture" is exactly what the next phase must test.

**Threats to validity a reader should weigh:** (a) is Reflex B fair, or a straw reflex? (we argue it is a real myopic policy, and the depth-scaling is robust to using it); (b) is corrupt-then-repair a natural task or a contrived one that manufactures coordination? (c) does a perfect-simulator prize actually predict learned-model headroom, or could a learned policy internalize the coordination directly? These are open.

---

## Reproduce

```bash
cd experiments/
# strong + weak reflex, m sweep, depth sweep (each ~1-2 min, CPU on Modal)
modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --m 2 --depths "3,4,5" --corrupt-blocks "1,2,3" --n-instances 400
modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --m 3 --depths "3,4,5" --corrupt-blocks "1,2,3" --n-instances 400
modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --m 4 --depths "3,4"   --corrupt-blocks "1,2"   --n-instances 400
# seed robustness
modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --m 2 --rule-seed 1 --depths "4,5" --corrupt-blocks "2,3" --n-instances 400
```

Results JSON saved to the `rhm-scaling-data` volume under `rhm_sculpt_precheck/v{v}_s{s}_m{m}_rs{seed}.json`. A `--quick` flag runs a small smoke test with the built-in correctness assertions.

---

## If the reader is convinced: the natural next step

Build the *learned* version at a high-prize setting (e.g. `L=4, m=2, c=2–3`, keeping `seq_len=16` comparable to the sensing task): a model-free edit-policy (the reflex analog), a learned forward model over its state, and a planner — then test whether the model-free policy leaves forward-model-plannability headroom (predicted *yes* here, unlike the sensing task where it was *no*), and whether co-training an internalized forecast reorganizes the representation toward plannability. That experiment, not this one, is what would show internalization "firing" on an RHM control task.
