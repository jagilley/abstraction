# grokking/whittle — the grokked net, rewritten in its own minted coordinates and told to keep its answers while shedding weight, collapses to the one-frequency closed form: 18 numbers

**Up**: [`../README.md`](../README.md) (the grokking node; this writeup is its §4) · **Files**: [`FILES.md`](FILES.md) ·
**Decisions, gates, defects and run records**: [`NOTES.md`](NOTES.md) · **Conversation**: `CONVERSATION.md`[^private]
(the session behind both rounds; the parent's `../CONVERSATION.md`[^private] is the session before it).
**Motivation**: Jasper's ask (2026-09-23), after reading the parent's Mint / Basis / Rung: the point of extracting the Fourier
solution is not parameter count for its own sake but that the generalizing solution should not stay locked in one net's
idiosyncratic weights once found. Rung's fresh nets were a stranger trained over the minted features; he wanted the *same*
learner continuing, and proposed changing the objective from "learn modular addition" to "keep your modular addition, and zero
out as many weights as you can". The reading agreed on before the build: memorization stays representable in minted
coordinates (Rung measured it), so the basis does not forbid it; what the basis changes is the *price* of the generalizing
solution relative to the memorizing one, and the price should be on weight *count*, not magnitude, since weight decay is the
magnitude price that had stopped the kept count at 13.
**Runs**: 2026-09-23; `w1` (7 arms, 570 s wall, 7 CPU containers) and `w2` (8 arms, 847 s, 8 containers); app `grokking-mint`,
volume `grokking-mint-data`, profile `chromatic`; no fresh training, every arm starts from a banked net. **Counts in symbol
connections, the surviving wiring, signs and ranks are the claims.**
**Attribution**: the objective change, the continuing learner over a fresh one, the extraction-not-efficiency framing, the go
for one frequency and the license for the implementer to try several routes are Jasper's. The arm design (basis-committed
against one-hot, both sides minted, the full rotation, the weight-decay contrast), the count-price reading of why one-hot
memorizes, the channel move as the missing walk move, and the fair-claim decomposition are the orchestrator's. The QR
coordinates, the drift finding and its twin, the exact cascade, the group definitions, biases as groups, the plants, the
explicit tensor fit of the one-channel net, and every build, gate, reduction and withdrawal are the implementer's (one Opus
session, both rounds).

## One-liner

Take the grokked MLP of the parent node, rewrite its first layer (and, in one arm, its output layer) in the wave coordinates
Basis read off the net's own behaviour, and run a prune-and-retrain walk at weight decay 0 gated only on "do I still give the
answers I gave before on my train pairs", held-out logged and never consumed. **The change of basis is what lets the net go
sparse while staying general**: the same walk on the one-hot net keeps its train answers and falls to 8% held-out at 7,085
weights, while the basis-committed net reaches 817 weights at 99.2% and the both-sides net 162 at 99.8%. **The whittled wiring
is the equation's block structure**: every surviving first-layer unit reads one minted symbol, the same on both operands, and
the middle units read and write one symbol each, with no edge crossing frequencies. **With a move that removes a whole
frequency at once, the product net whittles to one frequency**: 3 product units, 18 weights, no biases, every one of the
9,409 pairs correct, and the three units' tensor fits A·Re(e^{iψ} z_a z_b z̄_c) at 0.6% residual with the frame phase
cancelled, which is cos ω(a+b−c) written as Gauss's three real multiplications. Any of the three surviving frequencies carries
the task alone. The ReLU net stops at two frequencies and 76 numbers under three optimizers. Against the original 53,985
parameters: 235× for the same ReLU net continuing from its own weights, 3,000× for the product net. Nothing about cosines,
97 or the closed form was handed over; the gifts are the eigendecomposition (Basis), the identification of output classes
with input symbols, the count-price machinery, and, for the 18-number result, multiplication.

## The question, and the vocabulary

The parent found that the vanilla net is Fourier over 13 frequencies, that the basis can be read off the net's own function
with a label-free certificate, and that the kept count is the minimum affordable under weight decay 1.0. What it did not do
is change the learner that found the solution. Its 34× was a fresh net over the minted alphabet.

Terms, as used below. A **minted symbol** is one of the 2-D subspaces Basis recovered (a conjugate eigenvector pair of the
net's own shuffle a ↦ net(a, g)), plus the recovered constant direction; the 13 the parent's gate kept are **K**. The
**minted coordinates** spell an input symbol as its 27 projections onto DC and K, at unit wave amplitude, so the first layer
becomes a 128×54 matrix A with A·x equal to form C's layer 0 exactly. **Committing** means the net keeps only these
coordinates, with A trainable: the subspace is fixed, the coefficients are not. **Both sides minted** rewrites the head the
same way: the net writes 27 numbers and the score for class c is their dot with c's own spelling. A **group** is the unit of
pruning: a unit's connection to one minted symbol is a 2-vector, pruned together, because the rotation inside a subspace is
arbitrary and a scalar zero inside a 2-vector is not a claim; elsewhere a group is a scalar. **Symbol connections** are the
group count, the currency of record; scalars are reported beside it. The **walk** is the parent's `census_walk` in reverse
over the net's own weights: per layer, cut the smallest groups by norm, retrain with weight decay 0 on the train pairs with
the *original net's train argmax* as the target (it equals the train labels), admit iff agreement is 100%, else restore and
halve the fraction, down to a floor. The **channel move** (round 2) removes one symbol's every read and write at once. A
**drift twin** is the arm's starting net retrained for the same epochs with nothing pruned, which separates what pruning did
from what retraining alone did. Held-out accuracy on the 6,587 test pairs is the **oracle**, logged at every round and never
consumed; DFT frequency labels appear in the tables only as the oracle's names for the symbols.

Every arm starts from a banked net: the parent's grokked MLP at epoch 39,999 (53,985 parameters, test 1.0) or, for the
product net, Rung's minted-13 bilinear at its solved state (`r2`, 7,845 parameters), rotated into the recovered coordinates.
Two facts from the parent's controls are load-bearing: one frequency's argmax margin is 0.002 per unit amplitude against
9.96 for the 13, and a ReLU MLP over one frequency's four features never passed 0.985 from scratch at any decay.

## 1. Round 1 (`w1`): the basis is what lets the net go sparse, and the wiring is the equation

Seven arms, all from the same walk. Parameters count biases, as the original's 53,985 does.

| arm | what it commits to | weight decay | weights → | parameters | held-out | drift twin |
|---|---|---|---|---|---|---|
| unrestricted | nothing (one-hot in, one-hot out) | 0 | 53,632 → 7,085 | 7,399 | 0.082 | 0.9997 |
| committed | input in K's 13 symbols | 0 | 35,712 → 817 | 1,017 | 0.992 | 0.995 |
| both_sides | input and output in K | 0 | 26,752 → 162 | 230 | 0.998 | 0.998 |
| all48 | input in all 48 symbols (a pure rotation) | 0 | 53,632 → 786 | 989 | 0.988 | 0.924 |
| committed_wd1 | as committed | 1.0 | 35,712 → 1,840 | 2,121 | 0.995 | 1.000 |
| committed_R2000 | as committed, 4× retrain budget | 0 | 35,712 → 796 | 1,012 | 0.943 | 0.983 |
| bilinear_both | product net, both sides in K | 0 | 4,108 → 54 | 81 | 1.000 | 1.000 |

**The one-hot net keeps its answers and loses the task.** Under the same gate and the same count price it falls
monotonically as it prunes: 0.978 held-out at 38,784 weights, 0.877 at 18,959, 0.505 at 14,104, 0.082 at the end, with its
worst test-pair margin going from +4.9 to −257. Its drift twin stays at 0.9997, so this is pruning, not the optimizer. The
surviving units keep a median 29 of 97 one-hot weights per operand, and read by projection onto the recovered basis they
put a median 0.47 of their energy on any one symbol. The mechanism is the price: in one-hot coordinates a wave is 97
scalars and in minted coordinates it is one 2-vector, so a count price makes waves unaffordable in the first and cheap in
the second. Memorization is representable in both (Rung's data sweep shows the minted-13 MLP memorizing at 940 pairs);
which of the two is cheaper is what the basis decides.

**The whittled wiring is per-frequency channels.** In the committed arm all 37 surviving first-layer units read exactly one
symbol, and the same symbol on a and on b; 11 of the 13 symbols survive (DFT 24 and 47 drop), at 3 units each except 4 for
one and 6 for another; 56 of 66 middle units take inputs from a single symbol, median fan-in 3. In both_sides all 28
first-layer units are single-symbol, 6 symbols are read and 5 written, and all 13 middle units write only symbols they read.
In all48 all 50 units are single-symbol. The one-hot arm has 19 of 122 middle units single-symbol. No arm keeps the constant
direction in layer 0. The logit table's energy on the frequency diagonal is 0.87 for both_sides and 0.97 for the bilinear
against 0.02 for the one-hot net.

**The count price finds the mint's set.** all48 commits to all 48 symbols, so nothing is truncated and the walk chooses.
It ends reading 14 symbols: 11 of the mint's 13 (not DFT 38 and 45) plus the three candidates the mint's gate had ranked
14th, 15th and 16th (DFT 13, 22, 12). Two different endogenous prices, one over frequencies at fixed weights and one over
weights at fixed coordinates, agree on the kept set to within the margin.

**The magnitude price keeps more.** At weight decay 1.0 the committed arm keeps all 13 symbols, 3–8 units each, at 2.3× the
weights of the decay-0 arm; its twin is flat at 1.0.

**The product net factors exactly.** 9 units, 3 frequencies (DFT 18, 24, 38) at exactly 3 units each, 6 weights per unit,
held-out 1.0 with minimum margin 2.1. Removing any single unit from 9 broke 80–358 train pairs, and a frequency's 3 units
never came up together, which set round 2.

**Drift, and what it does and does not touch.** At weight decay 0 with the loss near zero, Adam's steps are sign-like and
not rotation-equivariant (the parent's control A). With nothing pruned, a 2,000-epoch retrain moves all48's held-out to
0.92, and chained 500-epoch retrains move every MLP arm's; the committed twin's last chunk lost 88 train pairs. The
per-round budget was set to 500 epochs with early stop, and the twin is logged beside every walk. Held-out in all48 and
committed_R2000 is mixed with this drift; the counts are not, since a 4× budget moved committed from 817 to 796.

## 2. Round 2 (`w2`): one frequency, 18 numbers

The channel move offers frequencies in ascending total norm, removes one whole, retrains for up to 20,000 epochs (the
bilinear's twin is flat at 1.0 over that budget, and the MLP's stays ≥ 0.9997), and gates as before; biases become groups
too, including all 27 numbers of the minted head bias, which round 1 never put on the table. Phases alternate, channel then
group, until a full cycle admits nothing.

| arm | start | weights (scalars / groups) | biases | frequencies | units | held-out | min margin, all 9,409 pairs |
|---|---|---|---|---|---|---|---|
| bil_channel | round 1's 3-frequency bilinear | 18 / 9 | 0 | 1 (DFT 18) | 3 | 1.000 | 0.214 |
| bil_plant1 ×3 | one of its channels alone | 18 / 9 | 0 | 1 (38 / 18 / 24) | 3 | 1.000 | 0.007 / 0.006 / 0.005 |
| bil_from_start | Rung's 52-unit bilinear | 36 / 18 | 0 | 1 (38) | 6 | 1.000 | 0.036 |
| mlp_channel (AdamW 1e-3) | round 1's both_sides | 68 / 40 | 15 | 2 (39, 47) | 12 + 4 | 0.998 | |
| mlp_channel_sgd | same | 68 / 40 | 9 | 2 (39, 47) | 12 + 4 | 0.9995 | |
| mlp_channel_lr1e-2 | same | 68 / 40 | 8 | 2 (39, 47) | 12 + 4 | 0.9994 | |

**The one-frequency net.** The bilinear removed DFT 38 in 446 epochs and DFT 24 in 2,846, the surviving amplitude going
7.6 → 13.6 → 120; the group phase then removed all 14 head-bias groups and nothing else (every weight-group prune broke
about 2,500 train pairs). What is left, written out: three product units on symbol s9, each with a 2-vector on a, a
2-vector on b and a 2-vector on the head's two rows for that symbol.

| unit | U (on a) | V (on b) | W (to the head) |
|---|---|---|---|
| 14 | (−4.868, 4.107) | (−5.402, 1.623) | (−4.821, 1.874) |
| 31 | (−5.327, −2.795) | (4.260, 5.257) | (−0.119, 4.791) |
| 51 | (−0.792, −5.698) | (−1.553, 5.660) | (−5.004, −3.805) |

Their 2×2×2 tensor Σ U_i ⊗ V_i ⊗ W_i fits A·Re(e^{iψ} z_a z_b z̄_c) with relative residual 0.0059 at A = 146.2, ψ = −2.68687;
the frame's own phase at the net's identity symbol is +2.68672, so ψ + φ₀ = −0.00015 and the peak falls at c = a+b. The
symbol's eigenvalue phase at the reference generator, read from the net, is 2π·48.00000/97 (the oracle's name for it is
DFT 18: 18·62 mod 97 = 49, which folds to 48). Margins on all 9,409 pairs: minimum 0.214, median 0.278, maximum 0.319,
none non-positive; the exact form predicts A·(1 − cos 2π/97) = 0.307. Three real multiplications is the bilinear rank of a
complex product, so this is cos ω(a+b−c) in its minimal product spelling. The other four bilinear finals fit the same family
at residual 0.009–0.023 with ψ + φ₀ within 0.003.

**Any frequency carries it alone.** Each of the three channels of the round-1 net, planted alone with every bias and the
constant row zeroed, starts at 12–29% train agreement and passes the gate after one retrain of 1,125–2,057 epochs, at
held-out 1.0 and amplitudes 27–52. The amplitude a lone frequency needs is whatever makes its margin positive: with an exact
product the argmax is right at any positive amplitude. The parent's "about 500" was the amplitude at which a product stayed
exact against weight decay 1.0's erosion; at decay 0 the price goes with the meter.

**The path changes the count, not the form.** Given the channel move from the start, the 52-unit bilinear removes 12
frequencies in ascending norm order (eleven in 99 epochs each, agreement never breaking), leaving all 52 units on DFT 38,
and the group walk then thins them to 6, from which any single-unit removal fails at the group budget. Six rank-one terms
spell the same complex product (residual 0.009) that the round-1 walk had already brought to 3. Group-first then channel
reaches the minimum; channel-first stops at a local one.

**The ReLU net's floor under this walk is two frequencies.** All three optimizers remove the same four channels in the
same order (DFT 36, 45, 38, 18), the fourth taking 15,516 epochs under AdamW at 1e-3, 3,937 under SGD and 2,215 under AdamW
at 1e-2, and all three end in the identical wiring: per frequency, 6 first-layer units feeding 2 middle units of fan-in 3
that write only that frequency. Removing either remaining channel fails at the 20,000-epoch cap in every arm, with
1,758–2,101 train pairs wrong and held-out 0.27–0.35. The twins are flat, so drift does not explain it, and it matches the
parent's control B2. One confound stands: the walk never regrows a pruned connection, so the one-frequency attempt starts
from 8 units, and this is a floor for this sparse net rather than for ReLU nets.

## The update

1. **The change of basis is what makes a count price find the generalizing solution.** In one-hot coordinates the same
   price, the same gate and the same optimizer produce a sparse memorizer at 8% held-out; in the net's own minted coordinates
   they produce a 99.2–99.8% net at 50–235× fewer parameters. Memorization stays representable in both; the basis decides
   which solution is cheaper. This is the parent's "the kept count is set by the meter" with the meter changed from
   magnitude to count, and it answers Jasper's question in the form he asked it: once the first layer is in minted
   coordinates the rest of the net does go very sparse, not because memorizing is impossible but because it no longer pays.
2. **The whittled net's wiring is the equation's block structure.** Every surviving first-layer unit reads one minted
   symbol, the same on both operands; middle units read one symbol and, when the head is minted, write only that symbol.
   That is Nanda's sum over independent frequencies as a connectivity graph, found by a price that was told nothing about
   frequencies being separable. In the product net the inside of each block is also the formula: three multiplications.
3. **The solution is 18 numbers, and it is the closed form.** With the channel move and biases on the table, the product net
   whittles to one frequency: 3 units, 18 weights, no biases, every pair correct, the tensor fitting the complex product
   with the frame phase cancelled. The wave came from the net's own shuffle, the frequency from its eigenvalue, the angle
   addition is what the count price left standing, the readout is the minted head. No cosine, no 97 and no template were
   given; multiplication was. This is below the 20-number amplitude-fit compile the parent handed over *with* the template.
4. **Which frequency is irrelevant, and so is the amplitude scale.** Any of the three surviving channels carries the task
   alone, as Rung's identity result said it should; and a lone frequency needs only a positive margin, so the amplitude of
   about 500 the parent reported was the price of holding an exact product against weight decay 1.0, not a property of the
   solution.
5. **The count price recovers the mint.** From the full 48-symbol rotation the walk keeps 14 symbols, 11 shared with the
   gate's 13 and the other three from the gate's next three ranks. The mint's set is not an artefact of the gate.
6. **Two floors, one architecture-bound.** The ReLU continuing learner stops at two frequencies and 76 numbers, identically
   under three optimizers, consistent with the parent's finding that ReLU never makes one frequency exact; the product net
   reaches one. The walk's move set also sets a floor: the same product spread over six units cannot be thinned to three by
   single-group moves, so the path matters for the count and not for the form.
7. **The fair size claim.** Against the original 53,985 parameters: the same ReLU net, continuing from its own weights, with
   its inputs and outputs in its own minted symbols, holds the task at 99.8% with 230 parameters (235×), of which 2× is the
   change of variables itself (97-wide to 27-wide on each side) and 118× is the whittle; after the channel move, 76
   parameters at 99.94% (710×). The most conservative claim assumes nothing about the output side: 1,017 parameters at
   99.2% (53×). The product net's 18 (3,000×) is a claim about the solution's minimal spelling given multiplication, not
   about the learner that found it. Against the strongest non-symbolic compression at matched accuracy, SVD at 26,852
   parameters for 0.99, the whittled ReLU net is 26–350× smaller; the parent's Mint had found that pruning by frequency
   without retraining was no smaller than SVD.

## What this does not show

One grokked net, one modulus. The product net descends from Rung's fresh bilinear, not from the grokked MLP, so the
continuing-learner claims are the ReLU ones (230 and 76 numbers, 2 frequencies). The gate is train-only and the walk cannot
regrow, so held-out slips by 14–50 pairs on the ReLU arms and the ReLU floor of two frequencies is a fact about this sparse
net rather than about ReLU nets: a re-densified one-channel retrain would separate the two. At weight decay 0 Adam drifts
at zero loss, which the twins bound but do not remove from all48's and committed_R2000's held-out series. The walk's move set
defines what "minimum" means here (channel-first ends at six units for the same form), and the walk was not run on the ReLU
net with regrowth or with a differentiable count price. Nothing here carries the whittled net onto a second task; the
18-number net is the solution on Z₉₇ and the identity of its frequency does not transfer to another modulus, while the
procedure does. The identification of output classes with input symbols, made once by Basis to build its operator, is used
again here to mint the head.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic MODAL_BUILD_VALIDATION=warn; CPU containers; no fresh training
modal run grokking/whittle/whittle.py::whittle_gates --retrain 500            # G1–G4
modal run grokking/whittle/whittle.py::whittle_run --tag wsmoke --smoke 1
modal run --detach grokking/whittle/whittle.py::whittle_run --tag w1         # 7 arms, one container each
python3 grokking/whittle/reduce_whittle.py --tag w1 --fetch
modal run grokking/whittle/whittle2.py::whittle2_gates                        # G5–G9
modal run --detach grokking/whittle/whittle2.py::whittle2_run --tag w2       # 8 arms
python3 grokking/whittle/reduce_whittle2.py --tag w2 --fetch
```

The exact commands and app ids are in [`results/RUN_whittle.sh`](results/RUN_whittle.sh); the volume layout is
`/data/<tag>/` with `<arm>_final.npz` beside the JSON (the w2 finals carry every pair's margin). The banked inputs are
`/data/m1/true/snapshots.npz` (the grokked net) and `/data/r2/B3_minted13_bil_stop_final.npz` (the bilinear).

## Next steps (not started; to be discussed)

- **A one-channel ReLU net with regrowth**: re-densify one channel (all 128 units on one symbol) and retrain, to settle
  whether the two-frequency floor is ReLU's or the walk's.
- **The whittled net as the continuing learner on a sibling task**: subtraction on the same symbols keeps the first
  layer's shape, and the 76-number net's two channels are the right basis for it; against the unrestricted grokked net and a
  fresh one.
- **A second piece**, which is still what would separate this from feature extraction: three-term addition, or a different
  modulus where the procedure transfers and the 18 numbers do not.

## Files

[`FILES.md`](FILES.md) indexes every script, figure and result; [`NOTES.md`](NOTES.md) holds the decisions (the QR
coordinates, the groups, the cascade, the budget and the drift finding, the channel move, biases as groups, the explicit
read), the gates G1–G9, the defect W-D1 and both run records. Tables of record: `figures/whittle_w1_table.txt`,
`figures/whittle_w2_table.txt`; figures `figures/whittle_w1_survivors.png`, `figures/whittle_w2_walk.png`.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
