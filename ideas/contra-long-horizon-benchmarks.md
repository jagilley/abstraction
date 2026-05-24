# Contra Long-Horizon Benchmarks

*"It's not helpful to think of problems as of "hard".  It's better to think that we merely don't know how to solve them yet." — Ilya Sutskever*

## I. Intelligence as rotation

Copernicus didn't work harder than Ptolemy. He didn't collect more data, run more calculations, or spend more years on the problem. He re-described the same solar system in different coordinates, and his model was initially *worse* at predicting observations. Ptolemy's geocentric system, loaded with epicycles, had enough free parameters to fit the messy realities of planetary motion with impressive accuracy. Copernicus, by insisting on a heliocentric frame with simple circular orbits, threw away those degrees of freedom and got worse numbers. On any benchmark that scored predictive accuracy, Ptolemy wins.

But Copernicus had done something Ptolemy hadn't. He had found the right *basis* — the coordinate system in which planetary motion was not merely predictable but *legible*. The epicycles weren't a feature of Ptolemy's model. They were a symptom of the wrong coordinate system patching over its own inadequacy. Once Kepler replaced Copernicus's circles with ellipses — a relatively modest fix, available only *because* the heliocentric basis made the anomaly's shape visible — the whole edifice of epicyclic machinery became unnecessary. The problem hadn't been lack of effort or data. The problem had been a bad choice of coordinates, and the solution was a rotation into better ones.

Call this a rotation — meant semi-literally. To solve a problem is to find a basis over the problem's conceptual space in which the solution is legible. Before the rotation the answer is *there*, but encrypted by a bad frame. After the rotation it's obvious. This is what Sutskever means when he says "it's not helpful to think of problems as hard — it's better to think that we merely don't know how to solve them yet." A hard problem is a problem seen in the wrong basis. The contribution of intelligence is to find the rotation that makes it easy.

The history of science is largely a sequence of such rotations. Einstein's theory of general relativity rotated gravity into a basis in which the Sun's gravitational lensing was consistent with empirical validation of Newtonian gravity in most reference frames. Darwin rotated biology into variation-and-selection coordinates and the fossil record became a narrative. Shannon rotated communication into information-theoretic coordinates and noisy channels became engineering. In each case: no new data, no additional effort, just a change of basis that made structure legible.

Now decompose any cognitive task in light of this. Three parts emerge: an **information state** (what you know), a **rotation** (the reframing that makes the answer decodable), and an **execution** (the mechanical work of producing the answer once you see it). These are different things — and Ptolemy vs. Copernicus is a case study in what happens when you conflate them. Ptolemy's system was a triumph of execution: immense computational labor applied in the wrong basis. Copernicus's was a triumph of rotation: the right basis, imperfectly executed. We remember which one mattered.

## II. What makes a rotation difficult

If intelligence is rotation, the natural question is: what makes some rotations hard and others trivial?

The answer isn't complexity per se. It's **constraint**.

A rotation is easy when you're free to rearrange your entire representational space without consequence. Early in learning — whether you're a neural network in the first epochs of pre-training or a student encountering a field for the first time — almost any reorganization is cheap, because little existing structure is load-bearing. You can adopt radically new coordinate systems without breaking anything, because there's nothing to break.

A rotation is hard when your representational space is already crowded with commitments. You know many things. Those things are encoded as specific directions, specific relationships, specific computational pathways. A new insight doesn't just need to find a good representation of itself — it needs to find one that is *compatible* with everything you've already committed to. It needs to rotate the basis while preserving the structural invariants that your existing knowledge depends on.

This is the core difficulty of what machine learning calls "continual learning" and what ordinary life calls "changing your mind about something fundamental." The problem isn't a lack of compute or data. The problem is geometric: you need to find a rotation of a high-dimensional space that simultaneously opens up room for new structure and preserves the fidelity of existing structure. The more you know, the harder this gets.

There is a deep connection here to physics. The most profound theoretical advances have the structure of symmetry transformations — changes of coordinates that leave certain quantities invariant. General relativity nominally falsifies Newtonian gravity. But the constraint it respects is that for the domains where Newtonian gravity was working — weak fields, low velocities — the old abstraction is recovered as a limiting case. The new basis doesn't discard the old one; it *contains* it as a special case and extends it into previously inaccessible regimes. Einstein didn't invalidate Newton. He found a larger coordinate system in which Newton's coordinates are a well-defined local patch.

This is what makes Einstein's rotations qualitatively different from, say, a competent engineer solving a series of well-posed subproblems. The engineer performs many small rotations, each locally unconstrained — each subproblem is self-contained, and the solution to one doesn't need to be compatible with the internal structure of the others. Einstein performs one large rotation under maximal constraint — the new basis must be consistent with all of electrodynamics, all of classical mechanics in the appropriate limit, and the isotropy of the speed of light. The constraint is what makes it extraordinary.

## III. Long-horizon tasks are chains of unconstrained rotations

There's an apocryphal story about Gauss as a schoolboy. His teacher, hoping to keep the class busy, assigns the sum 1 + 2 + 3 + ... + 100. Gauss pairs the terms from opposite ends — 1 and 100, 2 and 99, 3 and 98 — notices that each pair sums to 101, counts 50 pairs, and writes down 5050 while his classmates are still grinding through the addition.

The classmates are performing a long-horizon task: a hundred sequential operations, each trivial, whose difficulty is entirely in the execution. Gauss performs a rotation: he finds a basis in which the problem collapses from O(n) to O(1). The classmates' work is not less correct. But it's a fundamentally different kind of cognitive act.

Yet this is roughly what long-horizon benchmarks do. They measure how well a model sums the series term by term — how reliably it executes 1 + 2 + 3 + ... without dropping a carry. They do not measure whether the model can see that the answer is 50 × 101.

With this framework in hand, the critique of long-horizon benchmarks becomes precise.

A long-horizon task — build this application, resolve these 50 GitHub issues, implement this feature across a large codebase — decomposes into a sequence of discrete subtasks. Each subtask involves a local rotation: understand the subproblem, see the solution, execute it. But critically, these rotations are **locally unconstrained**. Each sub-step is well-specified enough that its rotation is easy, and the steps are loosely coupled enough that the solution to step 23 doesn't need to respect deep structural invariants established by step 7.

The difficulty of long-horizon tasks, such as it is, comes from two sources that are *not* intelligence in the rotation sense:

1. **Information management**: keeping track of what you know, what you need to know, and where to find it across a large problem space. This is perception and memory, not rotation.

2. **Execution reliability**: producing correct outputs consistently across hundreds of steps, where a single error can propagate. This is labor quality control, not rotation.

Chaining 500 easy rotations is an execution feat, not an intelligence feat. The horizon is a counter of how many rotations got spawned, not a property of the rotations themselves.

This has a direct corollary: long-horizon performance should be radically improvable by engineering. Better scaffolding, better memory management, better error correction, better tool use. And indeed, this is exactly what the last year of AI development has demonstrated. The gap between frontier models on long-horizon coding tasks has narrowed dramatically, not because the models got fundamentally more intelligent, but because the harnesses got better. When the binding constraint is execution infrastructure rather than cognitive rotation, infrastructure improvements dominate.

## IV. The automation ratchet

There's a deeper structural argument against benchmarking intelligence on long-horizon tasks, and it has to do with how domains mature.

The progression of any domain follows a characteristic arc:

1. **Stage 1**: Humans do the whole thing with intelligence. The task is undifferentiated — perception, rotation, and execution are all tangled together. A medieval physician does diagnosis, theory, and treatment as one holistic act of expertise.

2. **Stage 2**: Someone identifies the rotation. The key insight is isolated, made explicit, sometimes formalized. Germ theory separates the rotation (understanding disease causation) from the surrounding perceptual and manual work.

3. **Stage 3**: Everything around the rotation gets automated. Once you know what the rotation is, the information-gathering and execution steps can be codified into protocols, instruments, and eventually code. Blood tests automate diagnosis. Antibiotics automate treatment.

4. **Stage 4**: Only the rotation remains as a job for intelligence, if anything. The domain is mature. The remaining hard problems are the ones where we don't yet know the right basis — antibiotic resistance, novel pathogens, autoimmune disorders.

Long-horizon benchmarks freeze domains at Stage 1 and measure intelligence against the entire undifferentiated blob. This rewards scale over insight. Worse, it actively disincentivizes the factoring work — the identification and isolation of the rotation — that is itself the mark of intelligence having been applied to a domain.

The engineer who looks at a 10,000-line implementation task and says "this is really a 200-line task with a bunch of boilerplate around it, and we should generate the boilerplate" has applied more intelligence to the problem than the agent that dutifully writes all 10,000 lines. But on a long-horizon benchmark, the second agent scores higher.

A mature understanding of intelligence would recognize that the impulse to *shorten* the horizon — to factor out the rotation and automate the rest — is itself the most intelligent response to a long-horizon task. We have no benchmarks that reward this.

## V. The pre-training spectrum

The way that neural networks learn can be described in the terminology of the rotation-under-constraint framework.

During early pre-training, a model's representational space is largely uncommitted. No direction is load-bearing. The optimizer has enormous freedom: it can take large steps in any direction, and even noisy gradients will find their way to a useful basin, because the basins of attraction for good solutions are large and there is no existing structure to interfere with. The model undergoes dramatic phase transitions — sudden reorganizations where representations restructure around newly crystallized abstractions. These are genuine changes of basis, but they're *unconstrained* changes of basis. The model can move its entire manifold at once because nothing depends on it staying put.

As training progresses, this changes. More directions become committed to representing specific features, relationships, and abstractions. Each new change of basis must now respect an increasingly dense web of existing structure. The optimizer rightfully becomes cautious — learning rates decay, steps get smaller, phase transitions become rarer and more localized. Late pre-training begins to resemble continual learning: the representational space is crowded, and finding a rotation that improves one thing without degrading another becomes the binding constraint.

This suggests a spectrum:

**Early pre-training** (unconstrained rotation: any basis is fine, messy gradients work) → **Late pre-training** (increasingly constrained: rotations must respect existing structure) → **Continual learning** (heavily constrained: finding a viable rotation *is* the problem)

The interesting implication: the capabilities that emerge in late pre-training — the ones that feel like genuine understanding rather than sophisticated pattern matching — may be precisely the ones that require constrained rotation. The model has to discover an abstraction that *unifies* previously disconnected structures without destroying them. This is structurally identical to what Einstein did with special relativity: find a change of coordinates that makes electrodynamics and mechanics mutually consistent.

If this is right, it means the quality of a model's intelligence — not its competence, but the depth of its understanding — is related to the *degree of constraint* under which its representations were forged. And it means that the path to more intelligent models may run not through more data or more compute per se, but through training regimes that intensify representational constraint — that force the model to find rotations that unify rather than rotations that merely extend.

## VI. What we should benchmark instead

If intelligence is constrained rotation, then benchmarking intelligence requires measuring the quality of rotations under constraint. Not: "can you complete a long sequence of well-specified steps?" But: "can you find a reframing that makes a hard problem simple while respecting existing structure?"

Consider what this would look like concretely:

- **Refactoring under invariants**: Here's a codebase with 50 tests and a feature request that's architecturally awkward given the current design. Restructure the codebase so the feature becomes trivial, without breaking any tests. The score isn't whether you implemented the feature — it's how much simpler the codebase is *after*.

- **Unification tasks**: Here are three established results from different subfields. Find a framing that makes all three corollaries of a single principle. Score on parsimony and correctness.

- **Compression under fidelity**: Here's a complex explanation of a phenomenon. Produce a shorter explanation that preserves all the predictive content. Intelligence is measured by the compression ratio achieved without loss of fidelity — i.e., how much of the apparent complexity was an artifact of a bad basis.

- **Paradigm transfer**: Here's a solved problem in domain A and an unsolved problem in domain B that has analogous structure. Find the mapping. This directly measures the ability to see past surface features to underlying basis-invariant structure.

These benchmarks would be short-horizon by construction — the rotation, once found, is quick to express. They'd be hard to game with scaffolding, because the difficulty is in the rotation itself, not in execution reliability. And they'd reward exactly the thing that is most expensive, most uniquely valuable, and least automatable about intelligence: the capacity to see a problem in new coordinates.

## VII.

None of this is an argument that long-horizon capability is unimportant. It's certainly enormously commercially valuable. A model that can reliably execute long sequences of well-specified steps is a better tool than one that can't, full stop.

But tool-competence and intelligence are different things, and benchmarking one while claiming to measure the other is actively misleading. It distorts research priorities toward execution infrastructure and away from the deep question: can the model find a basis you hadn't considered?

If you want to build a digital Einstein, don't benchmark it on whether it can sustain attention across a 50-file codebase. Benchmark it on whether, confronted with a problem that looks hard, it can make it look easy — and whether the way it achieves that is compatible with everything else it knows.

The real intelligence was the change of basis we made along the way.
