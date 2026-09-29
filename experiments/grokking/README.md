# grokking — symbolic minting on vanilla grokking: the basis mints itself from the net's own function, the kept count is the norm-affordable minimum, and a fresh learner over it solves at 34× less compute; whittled in its own minted coordinates under a count price, the same net collapses to the one-frequency closed form in 18 numbers

**Up**: [`experiments/`](../CLAUDE.md) · **Files**: [`FILES.md`](FILES.md) · **Decisions, gates, falsifiers and defects**:
[`NOTES.md`](NOTES.md), [`basis/NOTES.md`](basis/NOTES.md) · **Children**: [`basis/`](basis/README.md) (the endogenous basis
mint, written up in §2 below), [`whittle/`](whittle/README.md) (the continuing learner whittled to the closed form, its own
writeup, headlined in §4 below) · **Conversation**: `CONVERSATION.md`[^private].
**Motivation**: Jasper's ask (2026-09-22): the repo's abstraction-creation work falls under the practice lineage
([`rhm/practice/`](../rhm/practice/README.md)) and the value side ([`rhm/logit_reading/`](../rhm/logit_reading/README.md)),
and months ago it did substantial work on modular-arithmetic grokking
(`fer/experiments/zipfian_grokking/`[^private]), but nothing practice-flavored has ever run
on grokking itself. The idea: find the Fourier solution conventionally, then let a practice-shaped step exploit the fact that the
Fourier solution is far smaller than the memorizing one, so that continuing to solve the problem costs far less compute. Scoped by
Jasper to **vanilla grokking only**: no Zipfian weighting, no CNB self-regulation, no forward self-models.
**Runs**: 2026-09-22 → 23; `m1` (Mint, 674 s), `b1` (Basis, 564 s), `r1` + `r1ctrl` (Rung, 2,841 s + 240 s), `r2` (Rung controls,
682 s + 761 s + 473 s); all on Modal CPU containers under profile `chromatic`, app `grokking-mint`, volume `grokking-mint-data`;
one seed (42) throughout, one net grokked once and its snapshots banked for every later read. **Ranks, signs, kept sets and
per-snapshot counts are the claims.**
**Attribution**: the question, the scope to vanilla grokking, the endogenous-basis question ("the grokked learner clearly knows an
*example* of the cosine/sine functions; could it mint that basis endogenously?", relayed from a side chat), the question whether
retraining a fresh net over minted features is the arc's symbol creation or a hack, and the reading that turning weight decay off
is no more of a gift than fresh training are Jasper's. The three-node design, the port of the arc's try-and-keep gate with the
net's own predictions as the grade, the margin calculation, the reparameterization argument for the 48-frequency failure, the
control designs and the synthesis are the orchestrator's. The one-gate-per-form call, the frequency-filtered form C, the three
tolerance rules, the softmax-operator read and its phase reference, the transported init, the planted solution, and every build,
gate, reduction and withdrawal are the two implementers' (one Opus session each).

## One-liner

A 2-layer MLP grokked on (a+b) mod 97 is read with the practice arc's minting machinery and no labels: frequency units are read
off its own weights, offered in its own energy order, and kept only if a model built from them agrees with the net's own train
predictions. **The vanilla net is Fourier** (96% of its logit structure on the diagonal plus DC, over 13–15 frequencies; the
repo's old "the MLP is not Fourier" claim was a top-5 measurement artifact), and the gate keeps **13 frequencies** carrying ~100
of the 128 units. **The basis mints itself**: diagonalizing the net's own function a ↦ net(a, g) on unordered symbols recovers all
48 character pairs exactly at the final net, with a label-free cross-generator certificate that fires only once the function is
a group action, and the walk over the recovered basis keeps the same 13. **The hand-written closed form over any minted set is
correct for any nonempty set**, so its zero-parameter compression is task-determined, not the net's; the net's own spelling
pruned to its 13 frequencies is 99.7–99.9% and no smaller than SVD. **Frozen as an input alphabet, the 13 buy a fresh learner
34× less compute** (5.0e11 against 1.7e13 FLOPs to 100%) and one notch of data range, and 13 frequencies the net never used buy
the same: the mint buys the count, not the identity. **Why 13**: one frequency separates the answer from its neighbour by 0.002
per unit amplitude against 9.96 for the 13, weight decay 1.0 erodes even a planted exact one-frequency solution, and with decay
at 0.01 a 501-parameter net learns it from scratch. The count the gate found is the learnable minimum under the recipe's norm
budget. A full 48-frequency alphabet never groks under AdamW even from raw's exact initial function, and does under SGD: a fact
about the optimizer, recorded, not about the mint.

## The question, and the vocabulary

The known solution (Nanda et al., `reading/3386_progress_measures_for_grokking.pdf`[^private]):
a grokked net puts each residue a on a circle at angle 2πωa/p for a few whole-number frequencies ω, adds angles, and reads
out the c whose angle matches. Three ingredients: the **basis** (cosine and sine waves at the used frequencies), **angle
addition** (cos ω(a+b) from cos and sin of a and b separately), and the **readout** (compare to every c). For prime p one
frequency already does this exactly. In the MLP here each hidden unit is tuned to one frequency: its 97 input weights for a,
read in residue order, trace a cosine at that frequency, and so do its weights for b.

The repo's standing assessment ([`one_layer_deeper/README.md`](../one_layer_deeper/README.md) §"Would the practice arc help
here?") is that the arc's minting cannot *discover* this structure, because the hierarchy lives in the computation rather than
the token stream and `mine_from = chosen` can only mint what appears in the learner's own trajectories. This node does not
contest that. It isolates component 3 of the practice definition
([`ideas/practice_manufactures_its_own_credit.md`](../../ideas/practice_manufactures_its_own_credit.md) §1, "re-chunking: compile a
mastered trace into one committed unit; never built"): grokking is the descent that discovers the characters, and the question
is whether a practice-shaped step can read them off the learner, name them, freeze them, and make what follows cheaper.

Terms, as used below. A **frequency** ω ∈ {1..48} is the unit being minted (conjugate pairs folded). The **read** is a 1-D DFT of
each layer-0 weight row over the residue index, or in §2 the eigendecomposition of the net's own function. The **producer
order** is the net's own summed weight energy per frequency, descending. The **gate** is `census_walk`, the arc's try-and-keep
extension loop (forked verbatim from [`preplay/incremental.py`](../rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py)):
offer candidates in order, admit iff the compiled model's disagreement with the net's own argmax on the **train pairs** does not
rise. The true labels are the **oracle**: held-out label accuracy is logged beside every step and never consumed. A **compiled
form** is what is built from the kept set K. Four ran: **A**, the closed form Σ_{ω∈K} cos 2πω(a+b−c)/p with zero free
parameters; **A-LS**, A with per-frequency amplitudes fit by least squares to the net's train logits; **B**, the net with every
unit whose dominant frequency is outside K zeroed; **C**, the net with each layer-0 row projected onto DC plus K's waves. Three
tolerance rules: `le0` (tol 0, the arc's verbatim rule, admits ties), `lt1` (a candidate must fix at least one train prediction
net of any it breaks), `m3` (must beat 3σ of chance agreement, about 16 pairs).

Recipe, forked verbatim from [`inverse_dynamics/grokking_fwd_vs_inv.py`](../inverse_dynamics/grokking_fwd_vs_inv.py): p=97,
MLP [128,128] ReLU, one-hot (a,b) of 194 dims, 30% split (2,822 train / 6,587 held-out), full-batch AdamW lr 1e-3 weight decay
1.0, seed 42, 40k epochs, 53,985 parameters. The net fits its train set at epoch 300 and reaches test 0.5 / 0.99 / 1.0 at epochs
10,750 / 16,050 / 18,550. Snapshots every 250 epochs are banked and every later read runs on them.

## 1. Mint (`m1`): the net is Fourier, the gate keeps 13, and the closed form is task-determined

**The read.** At the final net 15 frequencies carry 90% of layer-0 weight energy, 21 have at least one unit whose dominant
frequency they are, 122 of 128 units have the same dominant frequency for a and for b, and the median unit puts 0.92 of its
energy on its dominant frequency. The logit table over all p² pairs has 0.734 of its energy on the (ω, ω) diagonal and 0.229 on
DC, 3.7% elsewhere; 13 frequencies carry 90% of the diagonal. The old measure
(`cnb_self_regulation`[^private], 0.374 on its net) reads 0.430 here: it
takes the top 5 of the *unfolded* diagonal, which covers about three distinct frequencies of a solution spread over thirteen.

**The walk, at the final net.** Under `lt1`, B and C both keep the same 13, the top 13 of the producer order:
K = {10, 11, 18, 24, 30, 33, 36, 38, 39, 44, 45, 46, 47}. The walk stops admitting when train disagreement reaches zero;
frequencies 13 and 22, ranked 14th and 15th, are rejected. Under `m3`, B drops 33 (12 kept, 92 units) and C swaps 30 for 13.
Random candidate orders keep more: C/`lt1` 26–27 against 13, B/`lt1` 14–17 against 13, so the producer's order is the right
order, which is what the preplay node found on RHM (`pp4`). Forms A and A-LS keep exactly 1 under `lt1` and `m3` (frequency 30,
the top producer score) and all 48 under `le0`.

| form, final net | size | held-out |
|---|---|---|
| original net | 53,985 params | 1.000 |
| A, closed form over K = {30} | 0 free params (388 as a lookup table) | 1.000 |
| A-LS, K = {30} | 1 frequency + 1 amplitude | 1.000 |
| B, unit-pruned to K, `lt1` | 45,264 params (101 of 128 units) | 0.9974 |
| C, layer-0 rows filtered to DC + K, `lt1` | 36,065 params | 0.9994 |
| SVD truncation of all three layers, rank 36 | 29,261 params | 0.9974 |
| SVD, rank 33 / 41 | 26,852 / 33,276 params | 0.99 / 1.0 |

**The degeneracy, and why it is the load-bearing fact.** Form A has argmax a+b for *every* nonempty K, because for prime p the
cosine at any single nonzero frequency peaks uniquely at the right residue. A gate on A can therefore only decide between an
empty K and a nonempty one, and its 100% held-out accuracy from epoch 250, while the net is still at 0.000 test, follows from
its construction: it asks only whether the net's train argmax equals a+b, which is true once the net has fit its labels. A
memorizing lookup table passes the same gate. The zero-parameter compression is real and it is Nanda's solution written down; the
net contributed only the choice of frequencies, and the task makes that choice irrelevant to the closed form. B and C do depend
on the net, keep 13, and reach 0.997–0.999; neither is smaller than SVD at matched accuracy, because a frequency read of layer 0
leaves layers 1 and the head untouched.

**The compile runs ahead of the net mid-transition, and does not finish first.** Form C under `m3`, the net restricted to its
own kept frequencies, reads 0.346 / 0.570 / 0.766 held-out at epochs 9k / 10k / 11k where the net reads 0.211 / 0.352 / 0.564.
First snapshot at which each series holds ≥ 0.5 / 0.9 / 0.99 / 1.0 (250-epoch grid): net 10,750 / 13,000 / 16,250 / 18,750;
C/`lt1` 10,500 / 12,250 / 16,500 / 18,500; C/`m3` 9,750 / 12,250 / 17,000 / never; B/`lt1` 11,000 / 13,250 / 18,750 / never.
Kept k falls from 43–46 during the memorized plateau to 13 through the transition
([`figures/mint_m1_curve.png`](figures/mint_m1_curve.png)).

**Falsifiers.** On a net trained on shuffled labels (memorizes to 100% train) and on five untrained inits: A admits nothing under
any rule; A-LS admits noise under `le0` (k 12–39) and `lt1` (k 1–4) with held-out 0.000, and nothing under `m3`; B and C admit
by construction (they are the net's own spelling; `lt1` B 36–44, C 41–47) with true-label held-out at chance, 0.007–0.014.
Instrument checks: B at K = all is bit-exact to the net at all 208 snapshots; C at K = all is within 1.7e-5 of every logit; the
read recovers planted frequencies; the A-LS closed form matches an explicit least-squares fit to 5e-17.

## 2. Basis (`b1`): the basis mints itself from the net's own function, with a label-free certificate

Section 1 handed the net the DFT, which knows about cosines and about 97. Jasper's question: the net holds *examples* of the
waves (each tuned unit's row is the full character, tabulated at all 97 residues) but not the family; could it mint the family?
The route built: for a fixed symbol g the net's own function a ↦ net(a, g) is a map on the 97 **unordered** symbols; if the net has
grokked it is a single 97-cycle, and the eigenvectors of a cycle are exactly the characters of Z_p in the cycle's own order. So
read the operator M_g[c, a] = softmax(net(a, g))[c] from the net's own predictions at six generators, eigendecompose the reference
one, pair complex-conjugate eigenvectors into real 2-D subspaces (the candidate units), and run §1's walk and compiles over them.
Nothing about 97, cosines or the integer names of the symbols is used. Two readings ride beside it: the **certificate**, the
best projector overlap of each candidate with the other five generators' candidates (endogenous: a group action must give the
same subspaces for every g), and the **oracle**, each candidate's overlap with the true DFT character pair and the phase of its
eigenvalue against the true character's value at g (logged, never consumed). One addition the route needed: an eigenvector is a
character only up to a complex scalar, so form A needs a phase reference, which is read from the net too, as the symbol ê for
which net(a, ê) = a; it reads 0 from epoch 250 on. Everything runs on the banked `m1` snapshots, no retraining.

**At the final net the read is exact.** All 48 subspaces overlap the DFT pairs at 1.0000, the eigenvalue phases match the
characters' values at every generator to 1.4e-5, and the certificate is 1.0000. Fed to the walk, the recovered basis gives the
same producer scores (to 2.8e-4 relative), the same 128 unit dominants, the same kept K and the same held-out at all 12
form-by-rule gates as `m1`. Over the 95 snapshots from epoch 16,500 on, the kept set matches `m1`'s at 95 / 95 / 93 (B, three
rules) and 93 / 92 / 90 (C); after 17,000 the mismatches are single swaps.

**The read turns on late and sharply, and the certificate lags the oracle.** Epoch from which each series stays ≥ 0.99:

| series | epoch |
|---|---|
| net test accuracy | 16,250 |
| DFT overlap of the recovered basis, mean / min | 17,250 / 17,500 |
| all six hard maps a ↦ argmax net(a, g) are single 97-cycles | 18,500 |
| cross-generator certificate, mean / min | 19,500 / 19,750 |

Before test accuracy reaches about 0.99 the read is near chance (certificate 0.03–0.27, overlap 0.06–0.62, no bijections). One
wrong prediction among 97 breaks the cycle and moves the eigenvectors, so the basis is readable only once the function is
essentially exact ([`basis/figures/basis_b1_read.png`](basis/figures/basis_b1_read.png)).

**What the certificate certifies.** Designed checks: an exact shift recovers the characters to 1e-15; a random permutation and a
random 97-cycle both read overlap 0.06 and certificate 0.05; a relabelled group action σ∘shift∘σ⁻¹ reads certificate 1.0 with
DFT overlap 0.07 and overlap 1.0 with the *relabelled* characters. So the certificate certifies "my function is a group
action", not the integer labels, and the minted basis is the basis of the net's own action.

**The degeneracy carries over only once the basis is exact.** Between epochs 16,500 and 25,000 the A and A-LS gates keep 2–12
candidates where `m1` keeps 1: a single recovered candidate at overlap 0.9998 reproduces only 27% of the net's train argmax at
20,000, because a single cosine's argmax margin is 1 − cos(2π/97) ≈ 0.002 and that small a basis error flips it. From 25,250 on
k = 1 as in `m1`. That margin is the thread §3 picks up.

**Falsifiers.** Shuffled-label net: certificate 0.008–0.033, DFT overlap 0.039–0.052, 0 of 6 bijections, ê wanders; untrained
inits: certificate 0.09–0.11, overlap 0.06, operator near uniform. The recovered candidates on those nets are junk, and the
closed-form walks do admit a few of them (unlike `m1`'s form A), with held-out at chance; the certificate is what would stop the
walk, and it is the piece §1 lacked.

## 3. Rung (`r1`) and its controls (`r2`): the count buys 34× compute, identity buys nothing, and the count is the norm-affordable minimum

Freeze the kept 13 as the input alphabet: a and b each arrive as 26 numbers (cosine and sine at 13 frequencies) instead of a
97-way one-hot. Train fresh learners over it under the unchanged recipe, on the same split and seed: the same [128,128] MLP with
its input dim swapped, and a bilinear net, logits = W((U x_a) ⊙ (V x_b)) + b, width 52. Alphabets: raw one-hot; the minted 13;
all 48 frequencies (the basis gift without the mint); the 13 *lowest*-energy frequencies the net did not use; and `top1`, the
single frequency 30 that A-LS kept. Epochs to 0.99 / 1.0 test, evaluated every 10, runs stop at 1.0 or a 40k cap.

| alphabet, train fraction 0.3 | learner | params | epochs to 0.99 / 1.0 | FLOPs to 1.0 |
|---|---|---|---|---|
| raw one-hot | MLP | 53,985 | 16,040 / 18,520 | 1.69e13 |
| raw one-hot | bilinear | 15,229 | 5,240 / 16,740 | 4.32e12 |
| minted 13 | MLP | 35,809 | 4,570 / 9,520 | 5.77e12 |
| minted 13 | bilinear | 7,845 | 2,630 / 3,760 | 5.00e11 |
| unused 13 | MLP | 35,809 | 4,450 / 6,520 | 3.95e12 |
| unused 13 | bilinear | 7,845 | 2,810 / 3,390 | 4.50e11 |
| all 48 | MLP / bilinear | 53,729 / 15,125 | never / never (test 0.0006 / 0.0038, train 1.0) | — |
| top1 | MLP / bilinear / bilinear width 4 | 29,665 / 5,349 / 501 | never (peak 0.714 / 0.337 / 0.127) | — |

**Data sweep** (MLP, 150k cap below 0.3): raw solves only at fraction 0.3; minted 13 at 0.3 and at 0.2 (1,881 pairs, epochs
7,660 / 8,830), and at 0.1 and below sits at chance; top1 sits above chance but never exact at every fraction (0.546 → 0.089).
Unsolved arms fit train to 1.0 and hold their test level flat from about 18k epochs to the cap
([`figures/rung_r1_sweep.png`](figures/rung_r1_sweep.png)). The three-term task (a+b+c) mod p was built and not run.

**Identity buys nothing; the count is what the mint bought.** Thirteen frequencies the net never used solve as fast as its own
thirteen, on both learners. On this task every nonzero character is equivalent under the automorphisms of Z_97, so this is the
expected null, and it says plainly what the minted content is here: a number.

**Why one frequency fails: the norm budget** (`r2`, control B). The argmax margin of Σ_{ω∈K} cos 2πωd/p, peak at d=0 minus the
runner-up, computed exactly: 0.002097 for {30}, 9.958 for the kept 13, 48.50 for all 48. So a one-frequency solution needs an
amplitude thousands of times larger than the 13-frequency one to reach the same cross-entropy. Planting the exact one-frequency
solution in the 501-parameter bilinear net at amplitudes 1 to 10⁴ and training under the recipe: every arm's amplitude at
frequency 30 falls about 4× per 500 epochs, test accuracy holds at 1.0 only while it stays above roughly 500–700, and every arm
ends near amplitude 2.4–3.1 with margin −0.001 to 0.004. Weight decay 1.0 erodes a correct solution. From random init, the same
501-parameter net solves at weight decay 0.1 / 0.01 / 0 in 14,550 / 7,940 / 7,640 epochs, settling at amplitude about 520 and
margin 1.08; at decay 1.0 it stalls at amplitude 2.65. The minted-13 bilinear arm of `r1`, re-run with probes, solves at
amplitude 0.80 per frequency (sum 10.4), diagonal margin 7.6 at its stop, 15.2 if run to 40k. Under the recipe's norm budget
one frequency is unaffordable and thirteen at unit amplitude are cheap. **The count the gate kept is the learnable minimum under
the meter**, weight decay being a price on weight norm; the exact symbolic minimum of one is reachable only with the meter
relaxed. The ReLU MLP over the four top1 features never passes 0.985 at any decay: it can only approximate the product, and the
0.002 gap punishes any inexactness, so the bilinear product unit is a genuine gift of the angle-addition ingredient.

**Why all 48 fail: AdamW, not the alphabet** (`r2`, control A). With DC included, the 48-frequency features are s·Q times the
one-hot vector with Q orthogonal, so the MLP over them is the same function class as raw and plain gradient descent with L2
decay from a matched init follows the same trajectory. Transporting raw's exact initial layer 0 into the 48-feature coordinates
(initial logits match to 4.5e-8), AdamW still fails: test 0.0003–0.0005, train 1.0, on both the norm-matched and the `r1`-scale
features. Under SGD with momentum 0.9 and L2 decay, raw and its transported twin both grok at all three learning rates tried
(0.99 at 27,850 / 28,060, 25,140 / 22,210, 42,350 / 34,650 epochs), diverging only in floating point mid-transition (relative
loss difference above 1e-4 from epoch 200–2,000; test accuracy differing by more than 0.01 from 11,900–18,000). Adam's
per-coordinate normalization is not equivariant under rotation of the input, and in Fourier coordinates it memorizes and stays.
This gives "the mint buys the count" its mechanism under this recipe: restricting to 13 frequencies restricts the function class
to one Adam solves in these coordinates, where the unrestricted class it does not. It is a fact about grokking under AdamW,
recorded here because the arm was run, and nothing downstream rests on it.

One cross-recipe number, not a like-for-like comparison:

| arm | recipe | params | FLOPs to 1.0 |
|---|---|---|---|
| raw one-hot MLP | wd 1.0 | 53,985 | 1.7e13 |
| minted 13, bilinear | wd 1.0 | 7,845 | 5.0e11 |
| minted 1 (frequency 30), bilinear width 4 | wd 0.01 | 501 | 6.7e10 |

## 4. Whittle (`w1`, `w2`): the same net, in its own minted coordinates, told to keep its answers and shed weight, collapses to the one-frequency closed form

Written up in full at [`whittle/README.md`](whittle/README.md) (2026-09-23, after the sections above). **Goal**: replace Rung's
fresh learner with the learner that found the solution. Rewrite the grokked net's first layer (and, in one arm, its head) in
the coordinates Basis read off its own behaviour, then change the objective from "learn modular addition" to "keep your
modular addition and zero out as many weights as you can": a prune-and-retrain walk at weight decay 0, priced on weight
*count* rather than magnitude, gated only on 100% agreement with the net's own former train answers, held-out logged and never
consumed. **Finding**: the change of basis is what lets a count price find the generalizing solution. The same walk on the
one-hot net keeps its train answers and falls to 8% held-out at 7,085 weights, because a wave costs 97 scalars there and one
2-vector in minted coordinates; the basis-committed net reaches 817 weights at 99.2%, and with the head minted too 162 weights
(230 parameters) at 99.8%, 235× under the original 53,985. The whittled wiring is the equation's block structure: every
surviving first-layer unit reads one minted symbol, the same on both operands, and the middle units read and write one
symbol each. From the full 48-symbol rotation the walk keeps 14 symbols, 11 of them the gate's 13. With a move that removes a
whole frequency at once, the product net whittles to **one frequency: 3 product units, 18 weights, no biases, every one of
the 9,409 pairs correct**, and the three units' tensor fits A·Re(e^{iψ} z_a z_b z̄_c) at 0.6% residual with the frame phase
cancelled, which is cos ω(a+b−c) as Gauss's three real multiplications; any of the three surviving frequencies carries the
task alone, at amplitudes 27–146 (the "about 500" of §3 was the price of an exact product under weight decay 1.0). The ReLU
net stops at two frequencies and 76 numbers under three optimizers, a floor for that sparse net since the walk cannot
regrow. No cosine, no 97 and no template were given; the gifts are the eigendecomposition, the output-class identification,
the count-price machinery and, for the 18-number result, multiplication.

## The update

1. **The vanilla MLP is Fourier**, spread over 13–15 frequencies, and the repo's earlier claim to the contrary was a top-5
   measure on the unfolded diagonal. Settled by the read, at no cost.
2. **With the basis gifted, the symbolic content is the gift.** The closed form over any nonempty set of characters is correct
   for prime p, so a gate on it tests only whether the net has fit its train labels. Jasper's original sketch, "mint symbolic
   representations for the Fourier solution and collapse to a 100× smaller net", is realizable as a zero-parameter formula and
   says nothing about the net when the DFT is handed over. The net's own spelling, pruned to its kept frequencies, is the net's,
   is 99.7–99.9%, and is no smaller than SVD.
3. **The basis is recoverable from the net's own function, exactly, with a label-free certificate.** The gift moves from
   "the Fourier basis, which knows p and trig" to "find the axes of the shuffle my own function defines", which is task-general
   and yields the characters of whatever group the function is an action of. The eigenvalue phases carry angle addition. What
   remains gifted is the eigendecomposition operator and the readout template; that is where the gifts live in the arc too, in
   operators rather than content.
4. **The compile only reads a mastered thing.** The operator read is near chance until the net's function is essentially exact,
   and the certificate fires about two thousand epochs after test accuracy crosses 0.99. That is component 3's precondition
   measured: re-chunking presupposes mastery.
5. **The kept count is set by the meter.** Thirteen is not overspend. Under weight decay 1.0 the one-frequency solution's margin
   is unaffordable and is eroded even when planted; thirteen at unit amplitude buy a margin of ten. The gate stopped at thirteen
   because train disagreement reached zero there, and margin is why it took thirteen. Relax the meter and the compiled learner
   needs one frequency and 501 parameters. The compiled representation cannot memorize, so the price that made the original net
   leave memorization is no longer needed by the compile, and left on, forbids the compile's minimal form.
6. **What Rung is and is not, in the arc's terms.** Mint plus Basis is minting: units read from the learner's own solved
   function, kept by its own agreement, with a certificate. The amplitude-fit closed form over the recovered basis is component 3
   done verbatim, about twenty numbers, nothing trained, everything read from the net. Rung's fresh learners are an *assay* of
   whether the minted content is useful to a learner that did not earn it (the `assay` node's "content is giftable" question), not
   the compile, and its 34× leans on the bilinear product unit; the fairer same-architecture number is 2.9× compute and one notch
   of data range. The arc's payoff, a deeper piece becoming affordable, needs a second piece and has not been run here.
7. **Two facts about grokking under this recipe**, recorded because the arms ran: the full Fourier alphabet does not grok under
   AdamW and does under SGD; and the frequency-filtered net runs ahead of the unfiltered one through the transition by up to 0.2
   held-out without finishing first.

## What this does not show

One piece, one modulus, one seed, one architecture. Nothing here shows the minted *family* matters over the sampled characters;
that needs a task where the net's own frequencies do not transfer but the diagonalization procedure does (a different modulus),
or one where frequency identity matters (a composite modulus, multiplication). Nothing here is the arc's "next rung": the payoffs
measured are the same piece cheaper and the same piece on less data. The 34× is against a bilinear learner whose product unit
is the angle-addition ingredient built in; the same-architecture MLP over the minted alphabet is 2.9×. The 501-parameter result is
at a different weight decay from every other arm. The certificate's floor depends on the net (untrained inits read about 0.10,
the memorizing true run 0.02–0.05). The random-order controls in Basis permute candidate indices and are not comparable to
Mint's. The three-term task was not run. The old grokking-era nodes under `fer/experiments/zipfian_grokking/` are not re-read
here beyond the one measurement corrected in §1.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic; app grokking-mint, volume grokking-mint-data; CPU containers throughout
# Mint: gates, smoke, the run of record, the reduction
modal run grokking/mint.py::gates
modal run grokking/mint.py::mint --tag smoke --smoke 1
modal run --detach grokking/mint.py::mint --tag m1
python3 grokking/reduce_mint.py --tag m1 --fetch
# Basis: on the banked m1 snapshots (MODAL_BUILD_VALIDATION=warn is needed while a launch log is being written; see basis/NOTES.md)
modal run grokking/basis/basis.py::basis_gates
modal run --detach grokking/basis/basis.py::basis_run --tag b1
python3 grokking/basis/reduce_basis.py --tag b1 --fetch
# Rung and its controls
modal run --detach grokking/rung.py::rung --tag r1 --mint-tag m1 --k-form B --k-rule lt1
python3 grokking/reduce_rung.py --tag r1 --fetch
modal run --detach grokking/rung_controls.py::controls --tag r2 --groups A1,B1,B2,B3
python3 grokking/reduce_rung_controls.py --tag r2 --fetch
```

The exact commands, flags and app ids of every run are in [`results/RUN_mint.sh`](results/RUN_mint.sh) and
[`basis/results/RUN_basis.sh`](basis/results/RUN_basis.sh); the volume layout is `/data/<tag>/` with the `m1` snapshots banked
beside `mint.json`. The node was built as `experiments/grokking_mint/` and renamed at close-out (NOTES.md §Rename); the launch logs
keep the old path.

## Next steps (not started; to be discussed)

- **The twenty-number compile, learned** → done, and below twenty: [`whittle/`](whittle/README.md) §2 reaches 18 numbers with nothing
  trained from scratch. As originally posed: the 501-parameter net with its readout also expressed in the recovered basis, both
  sides minted, at weight decay off; and the data sweep for the 501-parameter net, since a net that cannot memorize should solve
  from far fewer than 1,881 pairs, which is the renewable-extraction number this substrate can actually give.
- **A second piece**, which is what would separate this from feature extraction: three-term addition on the same modulus (the
  same characters, one more angle addition), and a different modulus, where the net's sampled waves do not transfer but the
  operator-diagonalization procedure does.
- **The same learner continuing**, rather than a fresh one → done on the same piece in [`whittle/`](whittle/README.md); still open
  on a second piece. As originally posed: the grokked net with its layer 0 committed to the minted basis
  (form C as the init) carried onto the second piece, against the unrestricted net.
- **Minting the template**: the readout is the one ingredient still hand-written; whether the projection onto the recovered
  basis can be read from the net's head rows the way the basis was read from its function.

## Files

[`FILES.md`](FILES.md) indexes every script, figure and result here; [`basis/FILES.md`](basis/FILES.md) the child's. Decisions,
gates, falsifiers and defects: [`NOTES.md`](NOTES.md), [`basis/NOTES.md`](basis/NOTES.md).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
