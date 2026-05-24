# Intelligence as Rotation

*"It's not helpful to think of problems as hard. It's better to think that we merely don't know how to solve them yet." — Ilya Sutskever*

## I.

Copernicus didn't work harder than Ptolemy. He didn't collect more data, run more calculations, or spend more years on the problem. He re-described the same solar system in different coordinates, and his model was initially worse at predicting observations. Ptolemy's geocentric system, loaded with epicycles, had enough free parameters to fit the messy realities of planetary motion with impressive accuracy. On any benchmark that scored predictive accuracy, Ptolemy wins. But Copernicus had found the right basis — the coordinate system in which planetary motion was not merely predictable but legible. Once Kepler replaced Copernicus's circles with ellipses — a relatively modest fix, available only because the heliocentric basis made the anomaly's shape visible — the whole edifice of epicyclic machinery became unnecessary. The problem hadn't been lack of effort or data. The problem had been a bad choice of coordinates, and the solution was a rotation into better ones.

Call this a rotation — meant semi-literally. To solve a problem is to find a basis over the problem's conceptual space in which the solution is legible. Before the rotation the answer is *there*, but encrypted by a bad frame. After the rotation it's obvious. This is what Sutskever means when he says "it's not helpful to think of problems as hard — it's better to think that we merely don't know how to solve them yet." A hard problem is a problem seen in the wrong basis. The contribution of intelligence is to find the rotation that makes it easy.

The history of science is largely a sequence of such rotations. Einstein's theory of general relativity rotated gravity into a basis in which the Sun's gravitational lensing was consistent with empirical validation of Newtonian gravity in most reference frames. Darwin rotated biology into variation-and-selection coordinates and the fossil record became a narrative. Shannon rotated communication into information-theoretic coordinates and noisy channels became engineering. In each case: no new data, no additional effort, just a change of basis that made structure legible.

Now decompose any cognitive task in light of this. Three parts emerge: an **information state** (what you know), a **rotation** (the reframing that makes the answer decodable), and an **execution** (the mechanical work of producing the answer once you see it). These are different things — and Ptolemy vs. Copernicus is a case study in what happens when you conflate them. Ptolemy's system was a triumph of execution: immense computational labor applied in the wrong basis. Copernicus's was a triumph of rotation: the right basis, imperfectly executed. The empirical success of Ptolemy's basis was, in a way, the thing holding astronomy back.

## II. What makes a rotation difficult

If intelligence is rotation, the natural question is: what makes some rotations hard and others trivial?

The answer isn't complexity per se. It's **constraint**.

A rotation is easy when you're free to rearrange your entire representational space without consequence. Early in learning — a student encountering a field for the first time, a scientist approaching a new domain — almost any reorganization is cheap, because little existing structure is load-bearing. You can adopt radically new coordinate systems without breaking anything, because there's nothing to break.

A rotation is hard when your representational space is already crowded with commitments. You know many things. Those things are encoded as specific directions, specific relationships, specific computational pathways. A new insight doesn't just need to find a good representation of itself — it needs to find one that is *compatible* with everything you've already committed to. It needs to rotate the basis while preserving the structural invariants that your existing knowledge depends on.

This is the core difficulty of changing your mind about something fundamental. The problem isn't a lack of effort or data. The problem is geometric: you need to find a rotation of a high-dimensional space that simultaneously opens up room for new structure and preserves the fidelity of existing structure. The more you know, the harder this gets.

There is a deep connection here to physics. The most profound theoretical advances have the structure of symmetry transformations — changes of coordinates that leave certain quantities invariant. General relativity nominally falsifies Newtonian gravity. But the constraint it respects is that for the domains where Newtonian gravity was working — weak fields, low velocities — the old abstraction is recovered as a limiting case. The new basis doesn't discard the old one; it *contains* it as a special case and extends it into previously inaccessible regimes. Einstein didn't invalidate Newton. He found a larger coordinate system in which Newton's coordinates are a well-defined local patch.

This is what makes Einstein's rotations qualitatively different from, say, a competent engineer solving a series of well-posed subproblems. The engineer performs many small rotations, each locally unconstrained — each subproblem is self-contained, and the solution to one doesn't need to be compatible with the internal structure of the others. Einstein performs one large rotation under maximal constraint — the new basis must be consistent with all of electrodynamics, all of classical mechanics in the appropriate limit, and the isotropy of the speed of light. The constraint is what makes it extraordinary.

## III. The pre-training spectrum

Neural network training offers a concrete — and surprisingly literal — instantiation of this framework.

During early pre-training, a model's representational space is largely uncommitted. No direction is load-bearing. The optimizer has enormous freedom: it can take large steps in any direction, and even noisy gradients will find their way to a useful basin, because the basins of attraction for good solutions are large. The model undergoes dramatic phase transitions — sudden reorganizations where representations restructure around newly crystallized abstractions. These are genuine changes of basis, but they're *unconstrained* changes of basis. The model can move its entire manifold at once because nothing depends on it staying put.

There is a geometric way to see why. Any data point, once embedded, can be decomposed into two components: its projection onto the model's current representational subspace — the part the model already captures — and the residual that's orthogonal to it. Early in training, that residual is enormous. The model can barely represent anything, so almost all of each data point is news. And critically, these residuals are *coherent* across data points: the shared structure of language (or images, or whatever the training domain) means that the orthogonal components point, roughly, in the same directions. The optimizer has no trouble finding useful rotations because the signal is loud and the space is empty.

As training progresses, this changes. More directions become committed to representing specific features, relationships, and abstractions. Each data point's orthogonal residual shrinks — but it doesn't just shrink. Much of the remaining structure isn't orthogonal to the existing subspace at all. It's *aliased* against existing features: a regularity about legal reasoning gets partially folded into the model's representation of formal register, because legal text is formal. The residual for legal reasoning isn't noise in the orthogonal complement. It's a misshapen remainder defined partly in opposition to structure the model has already committed to.

This is why the loss curve decelerates. The easy rotations — crystallizing new features in empty representational space from loud, coherent signal — are exhausted early. What remains requires coordinated restructuring: simultaneously adjusting existing features and crystallizing new ones, because the information the model needs to learn is entangled with directions that already serve other purposes. The optimizer becomes cautious — learning rates decay, steps get smaller, phase transitions become rarer and more localized. Finding a rotation that improves one thing without degrading another becomes the binding constraint.

This suggests a spectrum:

**Early pre-training** (unconstrained rotation: any basis is fine, messy gradients work) → **Late pre-training** (increasingly constrained: rotations must respect existing structure) → **Continual learning** (heavily constrained: finding a viable rotation *is* the problem)

The interesting implication: the capabilities that emerge in late pre-training — the ones that feel like genuine understanding rather than sophisticated pattern matching — may be precisely the ones that require constrained rotation. The model has to discover an abstraction that *unifies* previously disconnected structures — which often means revising existing features that have been encoding the right information in the wrong way. This is structurally identical to what Einstein did with special relativity: not adding new equations alongside Newton's, but reconceiving what mass, time, and simultaneity *meant* so that mechanics and electrodynamics could be seen as aspects of the same thing.

If this is right, it means the quality of a model's intelligence — not its competence, but the depth of its understanding — is related to the *degree of constraint* under which its representations were forged. And it means that the path to more intelligent models may run not through more data or more compute per se, but through training regimes that intensify representational constraint — that force the model to find rotations that revise and unify rather than rotations that merely extend.

## IV.

Intelligence, then, is not the ability to execute complex procedures. It is the ability to find — under constraint — the change of basis that makes complexity dissolve. The history of science is a record of such rotations. Neural network training, viewed geometrically, is a machine that produces them. The open question is whether we are building systems capable of the hard rotations — the constrained ones, the ones that restructure rather than merely extend — and whether our methods for evaluating intelligence would notice if they did.
