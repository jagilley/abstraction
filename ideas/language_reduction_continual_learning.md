# Why language reduction ~= continual learning

Builds on [ideas/language_reduction.md](ideas/language_reduction.md)

### The principle

The net result of the language reduction experiment produces a problem setup that's remarkably close to the continual learning problem more generally.

Consider what happens in the example where the token 'queen' has been removed from the pre-training distribution. We have good reason to believe that the model still represents a concept of "female royalty" quite expressively. See language_reduction/GEOMETRY_ANALYSIS.md for details. At least, it *can* represent this concept even if it doesn't have a word for it.

This is remarkably similar to a regular LLM that has arrived at some unexpectedly coherent but off-manifold concept through in-context learning. The core insight - whether you want to express it in the model activations or compress it down to a token - is off manifold from what the model would naively produce, but off manifold in a coherent, interesting direction.

Traditional LLMs are limited by the fact that they can arrive at such insights through forward passes, but can't rearrange their own representations to take advantage of the power that these insights have from an explanatory perspective. Additionally, their ability to introspect in serious recursive fashion may be limited by their not having any tools to directly experience their "signatures of cognition": they can output text, and they can look at their own outputs, but we're not giving them a privileged view into the cognitive processes that produced those outputs.

In principle, if you can get a denoised model that doesn't represent the token 'queen' natively to be capable of "representing this concept natively" rather than as an off-manifold direction that needs to be discovered de novo, you could apply those same processes to the problem of continual learning in production LLMs.

### Novelty is sparse in concept-space

A "completely novel" idea is never completely off the edge of the map. Real models have seen approximately all text ever produced. Even genuinely novel insights should be decomposable along semantic axes the model already represents well. What makes an idea novel is that it's off-manifold in a few small but important dimensions — a sparse perturbation against a backdrop of familiar semantics. This is exactly the queen case: the concept is just "king" perturbed along the gender axis. The model has all the infrastructure; it just hasn't composed it this way.

This means the GLP's job is tractable. You don't need to project arbitrary off-manifold directions back on. You need to identify the small number of dimensions that constitute the novelty, and make the model native in that specific subspace. The GLP's velocity field may be the right tool for this: the velocity at a point tells you which way the manifold wants to pull it, and if the velocity is large along a few dimensions and near-zero along the rest, you've isolated the novelty axes.

### Integration is structural, not additive

Learning "queen" doesn't just add a word — it restructures how the model represents king (king now has a gendered counterpart), monarchy (monarchy now has internal gender structure), and the entire gender-authority subspace. A word is a compressed composition, and integrating a new composition changes the relationships between its ingredients.

This is testable: if you fine-tune the τ=0.3 model on queen-containing text, does king's embedding move toward the τ=0.0 model's king? See `language_reduction/EXPERIMENT_CONCEPT_RECOVERY.md` for the experiment spec.

### Tokens as backprop handles

A core disanalogy between brains and LLMs: brains can backprop directly from latent representations (Hebbian learning operates on activation patterns), whereas LLMs need tokens to compute a loss against. The only way to create gradient pressure in a specific direction in representation space is to route it through a token.

This reframes what a token IS in the context of continual learning. A token is not just a unit of communication — it's a unit of learnability. It gives a concept a permanent address (the embedding vector) and a focal point through which gradients flow to restructure surrounding representations. Without a token, a concept is a transient activation pattern that emerges from composition and disappears when the context ends.

Minting a new token for a novel concept — even if you never emit it at inference time — could serve as a "backprop handle" that organizes the surrounding embedding space during training. The token's purpose is organization, not communication. After training, you could delete the token and keep the structural gains.

In the language reduction setting, the queen token already exists in the vocabulary (V=50257) but is untrained at τ=0.3. Fine-tuning on queen-containing text IS effectively minting a token: training a fresh embedding from scratch and establishing all its contextual routing. This makes the language reduction models a natural testbed for the minted-token idea without requiring vocabulary expansion.

A further possibility: if the GLP residual tells you the direction of the missing concept in activation space, you could initialize the minted token's embedding as the projection of that residual into embedding space. This warm initialization places the new token at approximately the right location, potentially making few-shot concept learning much more efficient.

### The RL connection

SFT on "The queen ruled wisely" produces a gradient that says "increase p(queen|The), increase p(ruled|The queen),increase p(wisely|The queen ruled)." Most of that gradient mass is about the context — learning what "ruled" and "wisely" mean in royal text. The signal about queen's compressive utility over the compositional alternative is a
tiny fraction, buried under context-learning gradients that are irrelevant to consolidation.

A/B RL comparing "The queen governed the kingdom" (preferred) against "The female ruler governed the kingdom" (dispreferred) produces a difference gradient: ∇ log p(B) - ∇ log p(A). Everything shared between the two
completions — the context, the grammar, the semantics of governing — cancels. What survives is purely the signal
about the choice to use queen vs. the compositional route. That's a gradient that points directly at "consolidate
around this abstraction."

This maps onto two different things the model needs to learn, which I think have been conflated:

- SFT teaches content: what queen means, where it goes in embedding space, what contexts it appears in.
- A/B RL teaches utility: that queen is worth using — that routing through queen is better than maintaining the
compositional approximation.

MDL reduction is about utility, not content. The model needs to discover that queen is a useful compression
primitive — that using it simplifies the representation of the gender-authority intersection. SFT doesn't provide
that signal. A/B RL does, and it provides it in isolation from everything else.

And the language reduction setup gives you something unusually clean for this: you have the natural A/B pairs
already. For every position where the denoiser replaced "queen" with a substitute, you have a matched pair — same
context, one version with queen, one with the compositional approximation. These are free DPO pairs. The gradient
from these pairs says precisely "prefer the version that uses the new token," which is precisely the consolidationsignal.

The pipeline would be:

1. Integration (already done): SFT on queen-containing text. Queen finds a coherent position.
2. Utility training (the new step): DPO on matched pairs from original vs. denoised corpus. The gradient teaches
the model that queen is worth using.
3. Monitor for consolidation: Watch king's embedding MDL proxies and context loss. The phase transition — if it
happens — would be when the model starts simplifying king's representation because it has a better tool (queen)
for encoding the female-authority intersection.

The connection to the GLP: warm-initializing queen from the residual tells the model WHERE the concept is. The A/B RL tells the model TO USE IT. Without the RL, the GLP-initialized embedding is a good address that nobody visits. With the RL, the model is actively pushed to route through it.

I think this is the missing piece between structural integration and MDL reduction. The gradient signal from SFT
is about the data. The gradient signal from A/B RL is about the decision to use the new abstraction. Those are
different learning problems, and conflating them is why SFT alone didn't produce consolidation.