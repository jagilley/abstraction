# Cerebellum and Cognitive Architecture

*Domain: neuroscience, cognitive architecture, AI implications*
*Last updated: 2026-08-16*


## The brain's cognitive power arises from multiple specialized subsystems teaching the cortex, not from cortical computation alone
*Confidence: strong*

- The cortex learns from at least five distinct teaching systems: hippocampus (episodic replay), cerebellum (predictive forward models), basal ganglia (reward prediction errors via dopamine), amygdala (salience/valence tagging), and neuromodulatory systems (plasticity regime gating) — overview[^private]
- The cortex also learns directly from sensory streams (Hebbian/STDP), but subcortical tutoring accelerates and structures what it learns — same source
- Each subsystem implements a distinct learning algorithm: cerebellum = supervised ("what will happen?"), basal ganglia = reinforcement ("was that good?"), hippocampus = fast one-shot encoding, cortex = slow statistical extraction — maps to Doya's framework, discussed here[^private]


### The teaching systems are graders of *different type*, and the architecture's distinctive payoff is the detectability of grader blindness
*Confidence: moderate*

- The subsystems are not modules that decompose a task into parts; they are **graders that disagree** about the same behaviour. A single-grader learner cannot ever discover that its grader is blind — the failure is invisible from the inside, by construction, in the same way and for the same reason as a spurious somatic marker ([two_timescale_value_loop §somatic marker](../../ideas/two_timescale_value_loop.md): the spuriousness is invisible in-distribution and manifests only under shift) — [ideas/heterogeneous_graders.md](../../ideas/heterogeneous_graders.md)
- **Our own research process is the strongest available evidence**, because it independently converged on the same structure: [`mjc/HISTORY.md`](../../experiments/mjc/HISTORY.md) records that most of that arc's nulls were *instrument failures*, and every one was caught by playing two differently-typed graders against each other — control-vs-FM-error in [`drift_value_loop`](../../experiments/mjc/drift_value_loop/README.md) Cut 3 (clean interior optimum at b=0.5 exactly where control is flat to 0.1%), in [E1](../../experiments/mjc/on_policy/README.md) (control at ceiling by m=50 while FM error improved to m=400), and in the [S2 retraction](../../experiments/mjc/ballistic/directed/README.md) (re-scored by the sighted grader, the privileged oracle moved from fourth to second)
- The cerebellum→VTA pathway is then not a curiosity but **the wire that lets the two graders disagree with each other** — the anatomical form of the same move (sharpens the "prediction errors modulate dopaminergic signaling" line below from a connectivity fact into a functional claim)
- Caveat on the same history: knowing which grader is blind did **not** stop the next node from reading a program-level belief update off the instrument it had just published as blind. Detectability is not automatic — which is the argument for building it as a mechanism rather than relying on discipline.

#### No single learning signal can be both dense and evaluative — so the two-teacher structure is derived, not designed
*Confidence: moderate*

- **Dense** requires being *free*: available every step, hence self-supervised on what actually happened (every `(s,u,s′)` is a label of the physics, no goal required). **Evaluative** requires referencing outcomes, and the informative outcomes are the ones you would rather not sample — in the wild, sampling rewards is dangerous. The two pull opposite ways and no known objective collapses them — [ideas/heterogeneous_graders.md](../../ideas/heterogeneous_graders.md)
- The motor case is the crisp one: reward alone cannot learn movement (too dangerous to sample failures); prediction error alone has no preference over movements (Friston's dark room). Neither is a motor learner; the **pair with an interface** is.
- This is why the reward-free / ballistic synergy is a good deal rather than an awkward one — the one asset maintainable *without* reward is exactly the asset committed movement *depends on*, the same object — [`mjc/ballistic/`](../../experiments/mjc/ballistic/README.md)
- Load-bearing because it makes the structure *forced* rather than a contingency of vertebrate anatomy, and therefore expected to reappear in any learner facing both conditions.
- **The same dichotomy from the representation side, and arguably the more intuitive statement of it: compression vs expansion.** Compression (`R_res → R_comp`) is drivable by a dense predictive signal; **expansion (`R_act ↑`) is not**, because opening a direction makes prediction *worse* before it makes it better — Copernicus got worse numbers than Ptolemy at the moment of the rotation — so its grader must tolerate deferred, initially-negative payoff, i.e. be evaluative. Consequence, measured behaviourally: compression alone terminates on fixed data — the ratchet stops compounding and hits activation-norm inflation by 16 cycles. *(The rank numbers originally cited here — wake-sleep 23.9 → 13.5, the internal route 27.8 → 30.2 — were withdrawn 2026-07-26: the `R_act`/`R_comp`/`R_res` decomposition they came from does not hold, so they do not license reading the phenomenon as a frontier draining. The phenomenon stands. **Resolved 2026-07-27**: the decomposition was falsified across three domains and the rank mechanism withdrawn outright — a wake-sleep 2x2 finds compression has no measurable effect on the triple (wake-sleep ~ open-loop, ΔR_act +52.7 vs +51.9) and `R_res` does not drain. Replacement instrument: [residual_decomposition](../../experiments/rhm/residual_decomposition/README.md).)* — [dimensionality expansion §What grades an expansion?](../dimensionality_expansion.md), [GATED_RATCHET](../../experiments/a2a_forward/GATED_RATCHET_README.md)
- Under that reading **learning progress is the expansion grader**, and its band-pass shape (~0 when mastered, ~0 when irreducible, peak at moderate-and-falling error) is a *specification* rather than an empirical curiosity — [two_timescale_value_loop](../../ideas/two_timescale_value_loop.md), [curiosity Phase 1](../../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md)
- Engineering corollary: LLM training does not lack an expansion loop — **it has one implemented in humans** (weight decay, width, depth, data mix, when to stop are all deferred-payoff expansion decisions on a wall-clock of weeks), which is the mechanism behind LLMs expanding only slowly, coarsely and exogenously — [ideas/heterogeneous_graders.md §4b](../../ideas/heterogeneous_graders.md)

#### Disagreement between *heterogeneous* graders is a different instrument from disagreement between *homogeneous* ensemble members — and only the former can see structural blindness
*Confidence: speculative — the distinction is argued and consistent with existing data, but the discriminating experiment is unbuilt*

- [`ballistic/directed/`](../../experiments/mjc/ballistic/directed/README.md) S1 established, on a structural argument, that ensemble disagreement **cannot detect a drift**: every member trained pre-drift agrees, and they are all wrong together. Disagreement finds where you *lack* data, not where your data went *stale*. Recorded there as a falsification of the instrument.
- But those members were **homogeneous** — same objective, different seeds. Blindness that is *structural* is shared by every member, so homogeneous disagreement is blind exactly where its members are. Heterogeneous graders do not inherit this: a grader blind for a structural reason stays blind while a differently-typed one does not.
- Existence proof from our own data: the control grader and the FM-error grader disagreed *precisely at* control's structural blind spot (the bullets in the parent node).
- **Open, and the proposed test**: make grader disagreement itself the allocation signal, with homogeneous seed-ensemble disagreement as the honest baseline, under a **local** drift ([E2](../../experiments/mjc/on_policy/README.md) — global drift leaves no place-dependent blindness to detect) and with the noisy-TV control retained, since irreducible noise also produces disagreement; the discriminator is that noise-disagreement does not *close* when you collect there — [ideas/heterogeneous_graders.md §9](../../ideas/heterogeneous_graders.md)


### Each learning organ consumes exactly one currency of world-change, so a learner needs as many organs as its niche has drift currencies
*Confidence: strong*

- The measured core is a crossed double dissociation on one substrate: drifting *what is true* (grammar rule cells, demand fixed) degrades the dense learner (plant parse 0.62→0.50) and leaves post-commit verification silent (1 recert swap in 62), while drifting *what is asked* (derivation distribution, truth pinned at precision 1.000 by gate) leaves the dense learner flat and fires verification decisively (9/62 swaps, all coverage-improving; ablating it costs Δ0.271 deep error, ≈8× measured stream noise) — each news type moved exactly its predicted organ and left the other flat, in both directions — [typed_gaps](../../experiments/rhm/practice/typed_gaps/README.md)
- Two further cells of the map are retrospective rather than newly measured: **interface-news ↔ repair/metering** (the mjc bridge/damage-recovery arc is where δ earned its keep — [mjc/practice](../../experiments/mjc/practice/README.md)), and **level-news ↔ the teacher** (pacing signals beyond the earnable range are undefined, not noisy, so that currency cannot be self-consumed — [recital](../../experiments/rhm/practice/recital/README.md), [tall](../../experiments/rhm/practice/tall/README.md))
- The gap must be **aimed** for any organ to be exercised at all: news in the executor's currency or at inadmissible loudness destroys the experiment rather than testing the component (level-2 grammar drift and σ=2.5 demand drift, both measured inadmissible by the same arms-stop-separating signature) — [typed_gaps](../../experiments/rhm/practice/typed_gaps/README.md)
- Corollary for monoliths: a training regime whose world drifts in a single currency (pretraining: fact-content) needs the single matching organ and is correct to have only it; where LLM practice becomes metered, the missing organs reappear piecemeal with humans in the role — consistent with this tree's "expansion loop implemented in humans" reading — [meta_learning_under_metered_data](../../ideas/meta_learning_under_metered_data.md), [heterogeneous_graders §4b](../../ideas/heterogeneous_graders.md)
- Scope: the crossed measurement is single-seed and single-substrate; the load-bearing contrasts are ≥8× measured stream noise and the dissociation spans two independent runs, but the four-organ map has two cells measured directly and two inherited — [typed_gaps §Caveats](../../experiments/rhm/practice/typed_gaps/README.md)

See also: [No single learning signal can be both dense and evaluative](#no-single-learning-signal-can-be-both-dense-and-evaluative--so-the-two-teacher-structure-is-derived-not-designed) — the signal-side derivation of the same multi-organ conclusion; this node is the world-side derivation, and their convergence is what makes the architecture look forced rather than contingent.

#### A committed chunk stores demand-concentration, not truth — so skill maintenance is demand-tracking, evaluative rather than entropic
*Confidence: strong*

- Truth-immunity measured directly: a frozen committed table with 36% of its entries invalidated (precision 1.000→0.643) tracks the *current* truth's performance to within ±0.05, and a stale mined table **beats the full current true table** in audition throughout — a falsification test run on content that turned out not to be propositional — [typed_gaps](../../experiments/rhm/practice/typed_gaps/README.md)
- The value sign structure, from an oracle bracket differing only in tracking: current concentration (earned-vs-given 1.00–1.24) > full coverage (1.0 by construction) > stale concentration (0.345–0.754) — concentration is a leveraged bet on demand with a decay time set by demand drift, and coverage is the unleveraged hedge — same source
- Maintenance mechanism: evaluative re-selection works (recert swaps all coverage-improving; the earned, maintained vocabulary matches a perfect demand-tracking oracle at ≈8% of feedback budget) while passive decay-forgetting *costs* coverage — selection, not averaging, at both ends of a chunk's life — same source
- Introspection splits accordingly: pre-commit certification stays demoted in every regime tested (three consecutive frontier refusals; abstinence never pays), while post-commit verification is load-bearing only under demand-drift — self-examination reconciles the frozen past with the moving present, it does not gate the future — same source
- Antecedent within the arc: the mined vocabulary was already an *empirical consumption prior* under a static world (a mined 8-entry table beats matched-size random subsets of the true table by ~0.20; value is concentration over coverage) — [ratchet](../../experiments/rhm/practice/ratchet/README.md)


### The cerebellum is a domain-general prediction engine, not merely a motor controller
*Confidence: strong*

- The cerebellum contains ~half the brain's neurons despite ~10% of its volume — discussed here[^private]
- Massive closed-loop circuits exist between cerebellum and PFC, routed through thalamus, topographically organized — same source
- Cerebellar damage produces cognitive deficits beyond motor: language fluency, working memory, emotional regulation — Schmahmann's "cerebellar cognitive affective syndrome" — same source
- Cerebellar activation tracks task novelty across cognitive tasks (verb generation, mental rotation, working memory) — high early in learning, decreasing as performance becomes automatic — same source
- The cerebellar microcircuit (granule → Purkinje, climbing fiber teaching signal) is remarkably uniform across all zones — the same algorithm wired to different cortical partners — discussed here[^private]

#### The cerebellar forward model's codomain — raw body state vs cortical activation — is set by afferent wiring, not by a different computation
*Confidence: strong*

- The uniform microcircuit computes on whatever the mossy fibers deliver, so one forward-model algorithm serves a *body-state* codomain in some regions and a *cortical-activation* codomain in others — the codomain is a wiring choice, not an algorithm change (sharpens the parent node's "uniform across all zones").
- Ancient **vestibulo-/spinocerebellum** (archi/paleocerebellum), fed by vestibular afferents and spinocerebellar tracts, forward-models *raw body/physical state* (spindle length/velocity, joint angle, load, head motion) — textbook motor cerebellum; the cleanest behavioral proof is central attenuation of self-generated tickle (Blakemore, Wolpert & Frith) and, in its absence, dysmetria.
- New **cerebrocerebellum** (neocerebellum, lateral hemispheres), fed by the cortico-ponto-cerebellar projection from association cortex, forward-models *cortical activation dynamics* — the moderate-confidence sibling belief below, and the phylogenetically newest part (neodentate), disproportionately expanded in humans.
- The two codomains occupy *physically distinct* territories along a phylogenetic gradient (body-state → cortical-state), so a "physics-state forward model" and an "activation forward model" are not rival theories of one organ but faithful descriptions of two cerebellar regions running one algorithm.
- Program relevance: this is biology performing the exact "swap the forward model's codomain by rewiring its input while holding the computation fixed" move we weighed — evidence that codomain is a free design knob and the algorithm is the invariant — [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md).

See also: [forward models of cortical dynamics](#the-cerebellum-builds-forward-models-of-cortical-dynamics-not-just-sensory-consequences-of-motor-commands) (the cortical-activation half in detail); [cerebellar expansion](#cerebellar-expansion-was-the-last-major-anatomical-change-before-behavioral-modernity) (the neodentate = the new, cortical-state territory).

**Cerebellar microzone modularity buys separability and forward modeling at once — by physically factoring, not by learning a clean distributed split.**
*Confidence: moderate*

- The cerebellum is built from hundreds of *microzones* — parasagittal Purkinje-cell bands each sharing one inferior-olive climbing-fiber signal, grouped into "multizonal microcomplexes" (Apps & Garwicz) — each a semi-independent module owning a slice of the input.
- So biology attains a variable-by-variable (e.g. value-relevant vs value-irrelevant) split without learning a disentangled distributed representation: it dedicates *separate hardware* per factor — the concrete biological form of an object-factored / slot architecture.
- Bears on the separable-vs-self-model tension in our control work: separability need not be learned inside one net if the substrate is physically modular — [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md).

#### The cerebellum builds forward models of cortical dynamics, not just sensory consequences of motor commands
*Confidence: moderate*

- Ramnani and others argue the cerebello-prefrontal circuit lets the cerebellum model PFC function itself — predicting prefrontal outputs, not just sensory consequences — discussed here[^private]
- The cerebellum receives efference copies of cortical state via the corticopontocerebellar tract, making it a conditional predictor: "given cortical state X, the cortex should transition to state Y" — discussed here[^private]
- The cerebellum's predictions are conditioned on current cortical activations (via pontine relay), not on cortical weights directly — it must *learn* cortical dynamics by observing them — same source
- This means the cerebellum's model necessarily lags after cortical learning, which may explain the brief post-insight "strangeness" where familiar things feel unfamiliar — same source

#### Cerebellar output serves two complementary computational roles: dynamical bias (forward) and learning signal (backward)
*Confidence: moderate*

- The ccRNN paper (Pemberton et al. 2021) frames the cerebellum as a backward DNI predicting future loss gradients; the BP(λ) paper (Pemberton & Ponte Costa 2024) extends this with eligibility traces. Both acknowledge the forward DNI variant also works — the cerebellum outputs activations that the cortex can use either way — ccRNN[^private], BP(λ)[^private]
- In our A2A experiments, the same FM prediction serves both roles with empirically separable effects: injection (forward role) produces robustness (0.24× OL sensitivity) and self-knowledge (R²=0.63–0.73) but not learning speed; local loss (backward role) produces learning speed (+1.4pp, 31% lower val loss) but 13× brittleness and zero self-knowledge — [local loss](../../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md)
- The local loss gradient `∂post_block3/∂θ_early · (post_block3 − FM_pred)` is literally a synthetic gradient — the FM prediction error projected through the local Jacobian — functionally equivalent to what ccRNN's cerebellar module provides — same source
- CL_LL (combining both roles) produces the best self-knowledge (R²=0.76–0.79) with 63% less dependency than injection alone, suggesting the biological system likely uses both simultaneously — same source

See also: [Self-prediction and self-knowledge — injection vs local loss dissociation](self_prediction_and_self_knowledge.md#the-forward-activation-preview-and-backward-synthetic-gradient-roles-produce-distinct-representational-effects)

#### Cerebellar prediction errors function as metacognitive monitoring signals
*Confidence: moderate*

- The motor analogy: without cerebellar forward models, movement is reactive and ataxic; with them, it's smooth and anticipatory. The cognitive analog: without cognitive forward models, thinking is halting and deliberative; with them, it's fluid and self-monitoring — developed here[^private]
- Pre-reflective "something is off" feelings (intuition that an argument is wrong before identifying the flaw, feeling of knowing, tip-of-the-tongue states) look like cerebellar prediction error about cognitive trajectories — same source
- Cerebellar patients show impaired metacognitive calibration — less accurate at judging their own performance — discussed here[^private]
- Cognitive "flow" may be the phenomenological signature of low cerebellar prediction error — the cerebellum has successfully predicted what the cortex is about to do — same source
- Cerebellar prediction errors may modulate dopaminergic signaling via VTA projections, linking the supervised learning system (cerebellum) to the reinforcement learning system (basal ganglia) — same source

#### The cerebellar error signaling pathway has rich structure
*Confidence: moderate*

- Climbing fibers from the inferior olive carry the canonical error signal — complex spikes in Purkinje cells, rare (~1-4 Hz) and therefore sparse and highly informative — detailed here[^private]
- Complex spikes trigger LTD at parallel fiber–Purkinje synapses that contributed to the wrong prediction; absence of complex spikes triggers LTP at synapses that contributed to correct predictions — same source
- Deep cerebellar nuclei receive both Purkinje output (inhibitory) and mossy fiber collaterals (excitatory), potentially performing a comparison operation — discussed here[^private]
- The thalamic relay applies temporal derivative filtering (burst vs tonic mode), amplifying error signals while attenuating steady-state — discussed here[^private]
- Cerebellar output lands in cortical middle layers (layer 4 / deep layer 3) — the same laminar target as feedforward sensory input — meaning the cortex treats it as *data*, not top-down feedback — same source


### The cortex is a general-purpose pattern integrator with representational homogeneity across areas
*Confidence: strong*

- The neocortex uses remarkably uniform architecture everywhere: six layers, similar columnar microcircuitry, similar population coding — developed here[^private]
- This means cortical areas speak the same representational language — a pattern from one region can be projected to another and processed using the same operations, regardless of source — same source
- The distinction between intermediate and final cortical activations is dynamic (transient vs attractor state), not structural — the representation is the same kind of thing at every moment — same source

#### The cortex can ingest cerebellar predictions as native input — the grounding IS the processing
*Confidence: strong*

- The cortex doesn't need cerebellar signals pre-formatted in "cortical language" — it just needs them at the right laminar address, and cortical circuits handle the rest — developed here[^private]
- The cortex and cerebellum co-develop, so cerebellar outputs and cortical weights co-adapt until the cerebellum's output space is one the cortex can use — same source
- Analogy: the cortex handles retinal input the same way — retinal ganglion firing isn't "in cortical format" but becomes meaningful by being processed through V1's weights — same source
- The cortex is constitutively unable to distinguish input source by format alone — it can only distinguish by content and context (which pathways are active, what else is co-active) — discussed here[^private]
- This architectural property — unified representational format — may be a precondition for the unified, seamless character of conscious experience — same source


### The cerebellum and cortex co-training produces an abstraction ratchet
*Confidence: moderate*

- The ratchet cycle: each day the cerebellum models the current cortex → each night those models get distilled into cortex → the next day the cerebellum encounters a cortex with new capabilities and can now model higher-order regularities — developed here[^private]
- This is structurally parallel to hippocampal → cortical consolidation (fast system trains slow system offline), but distills *predictions* rather than *episodes* — same source
- Well-documented that early motor skill acquisition heavily engages the cerebellum, then activity shifts toward cortical and striatal circuits as skill becomes well-practiced — same source
- This should produce power-law learning curves (rapid early gains, progressively subtler refinement) and sleep-dependent discrete jumps in skill — behavioral evidence exists for both — same source

#### Each ratchet cycle compresses multi-step computation into a single cortical primitive
*Confidence: moderate*

- The cerebellar prediction is necessarily lower-dimensional than the full cortical computation it predicts — it captures the input-output mapping but not intermediate dynamics — developed here[^private]
- When the cortex accepts a cerebellar prediction (pre-activated end-state consistent with its own evolving dynamics), full elaboration becomes partially redundant — the compressed prediction becomes a new stable association — same source
- Once operation A is compressed to a primitive, it serves as input to higher-order operation B, which then gets compressed in turn — this is how you get hierarchical abstraction without anyone designing the hierarchy — same source
- Maps precisely onto the expertise literature: novice chess players evaluate pieces, intermediates recognize tactical patterns (chunks), grandmasters recognize strategic configurations (chunks of chunks) — same source
- Experts genuinely cannot decompose their chunks back into primitives — the intermediate steps are no longer explicitly represented — same source
- The Maya Angelou observation ("people will forget what you said... but never forget how you made them feel") may be literally correct: the felt quality is the cerebellar model's compressed summary of the trajectory shape, persisting after content details decay — same source

#### Sleep is a key window for cerebellar-to-cortical distillation
*Confidence: moderate*

- The prediction: cortex in learning mode (high plasticity, NREM up-states favorable for LTP, spindles gating cortical plasticity), cerebellum in inference mode (generating outputs for cortex to learn from, but not itself updating) — developed here[^private]
- Sleep spindles are thalamocortical events generated in thalamus including VL thalamus, which is the relay for cerebellar output — so there's an existing mechanism for cerebellar signals to arrive at cortex timed to maximal plasticity windows — same source
- Cerebellar-cortical distillation is cross-architecture (cerebellum's feedforward granule-Purkinje architecture → cortex's recurrent columnar architecture) — this works fine; cross-architecture distillation is well-established in ML — same source

**Dream phenomenology maps to the student side of distillation.** Dorsolateral PFC (metacognitive monitoring) is deactivated during REM — the cortex is absorbing predictions without the metacognitive critic, exactly like a student network tracking the teacher's output distribution without questioning it. Dream bizarreness may result from the cerebellum being queried across a broad input distribution for better generalization — analogous to out-of-distribution training in ML distillation. — developed here[^private]

**Cerebellar-cortical distillation may map better onto REM than NREM**, differentiating it from hippocampal consolidation (NREM-focused). If it's about learning generalizable forward models rather than stabilizing specific episodes, REM's role in integration and generalization would be a better fit. — same source

#### Co-training gives the cortex a self-index of its own competencies
*Confidence: moderate*

- Because the cerebellum constantly sends back predictions about cortical states, and the cortex processes these through its own weights, the cortex develops representations that are *about its own internal states* — a learned self-model, not introspection — developed here[^private]
- This explains fast metacognitive fluency: people can sense "I know something relevant to this" before retrieval — too fast for full prediction-error loops, but consistent with querying a learned self-index — same source
- The self-index is approximate, learned, and lossy — matching the phenomenology of rough, imperfect access to what we know — same source
- Creativity may be self-indexed recombination: querying the self-index combinatorially to compose knowledge in novel ways without an external orchestrator — same source


### The brain uses precision-weighted prediction errors to distinguish informative surprise from noise
*Confidence: strong*

- Predictive coding framework (Friston, Rao & Ballard, Hohwy, Andy Clark): every prediction comes with estimated precision (inverse variance), and errors are weighted by precision before driving updates — discussed here[^private]
- The *structure* of the error matters: low-rank structured errors (correlated across time/features) indicate the model is systematically wrong about something specific; diffuse high-dimensional errors are likely noise — same source
- Expected uncertainty (ACh-mediated, known unknowns: boost sensory processing but don't restructure the model) is qualitatively different from unexpected uncertainty (NE-mediated, model is wrong: trigger network reset and exploration) — Yu & Dayan framework — same source
- The amygdala provides fast contextual valence tagging: "good novelty" (learnable, structured) vs "bad novelty" (noise), grounded in both learned associations and evolutionary priors — same source
- Good/bad novelty discrimination feels near-instant, suggesting it's either a well-trained cortical pattern classifier or involves fast subcortical valence computation (likely both) — same source


### Cerebellar expansion was the last major anatomical change before behavioral modernity
*Confidence: moderate*

- Brain volume was already near-modern by 300,000 years ago; what changed between 100,000–35,000 years ago was internal reorganization — cerebellar and parietal bulging, cranial base flexion — discussed here[^private]
- Only fossils younger than 35,000 years show the same globular shape as present-day humans — same source
- The dentate nucleus (cerebellar output to PFC) is disproportionately expanded in humans compared to other primates, even relative to overall brain size — same source
- Parietal cortex (spatial reasoning, tool-use planning, causal simulation) + cerebellum expanding together suggests enhanced ability to simulate complex physical/causal sequences with increasing abstraction — same source
- The 100k–35k window overlaps with the archaeological explosion: symbolic art, composite tools, long-distance trade, rapid technological turnover — same source

**Neanderthals had the cortical hardware but lacked the cerebellar infrastructure.** They had larger brains than Sapiens on average but elongated braincases without cerebellar/parietal bulging, and their archaeological record shows less recursive complexity and symbolic explosion. — same source

**Behavioral modernity's diverse signatures may share a single bottleneck**: deep recursive prediction over cortical dynamics via the cerebellum. Art, language, complex tools, and trade networks are functionally diverse but would all emerge together if they all depend on the same domain-general computational infrastructure. — same source

**The prolonged maturation of the human cerebellum** (not reaching adult organization until late adolescence) enables more ratchet cycles and deeper representational hierarchies than a faster-maturing system could achieve. Modern human infants go through a postnatal "globularization phase" including cerebellar expansion — a pattern not seen in Neanderthal development. — same source


---


## Engineering implications

*Derived from the neuroscience beliefs above; these are our guesses about what follows for AI architecture.*

### Current LLMs lack a cerebellar analog and therefore lack the substrate for metacognitive calibration
*Confidence: moderate*

- LLMs have no subsystem that builds a predictive model of the network's own processing dynamics and generates error signals when internal operations deviate — discussed here[^private]
- The cerebellar patient parallel: LLMs can reason but have poor calibration, can't catch themselves mid-generation, lack the "something feels off" signal — same source
- Metacognitive calibration may require a structurally distinct monitoring component, not just scale — same source
- See also: [Why single models are not enough](../why_single_models_are_not_enough.md)

**Sharpening (2026-07-25): LLMs did not fail to acquire the subcortical systems — they were trained on the transcript of all of them.** Text is *"the whole world as projected by people onto text"* (Ilya, transcript[^private] §00:08:31), i.e. the frozen output channel of finished multi-grader systems, which is why the same interview finds *"no human analog to pre-training."* So "LLM ≈ cortex" is wrong in both directions: too stingy (in-context learning is a real fast loop, and looks more hippocampal than cortical) and too generous (the cortex learns from five teachers; the LLM has one whose signal is *already the other four's output*). Better: an LLM is a **fossilized synthesis of full brains that cannot play its inherited parts against each other in a first-class manner** — and correspondingly a *cortex alone* is closer to a raw-data/image model than to an LLM. Completeness bifurcates: cortex-alone is content-poor but process-complete; an LLM is content-rich but process-empty. — [ideas/heterogeneous_graders.md §3](../../ideas/heterogeneous_graders.md)

#### An LLM-as-judge is not a second grader; it is the same grader in a mirror
*Confidence: speculative — a structural argument about training setups, not a measurement on any model we have run*

- Grader conflict is present in the corpus **as a fossil**, so an LLM can emit the surface form of a system catching itself without a second grader to run. Under this reading **chain-of-thought is a simulation of grader conflict inside the one channel available** — which predicts its profile: large gains, plus an unreliability no amount of additional CoT fixes, because the check and the checked come from one distribution and go blind together — [ideas/heterogeneous_graders.md §8](../../ideas/heterogeneous_graders.md)
- The criterion that follows: a genuine second grader must be **sighted where the first is blind**. Same corpus + same failure geometry = redundancy, not heterogeneity.
- **Prediction**: RL post-training pays in proportion to how sighted-where-the-first-is-blind its grader is — verifiable rewards (does the code run, does the proof check) ≫ LLM-as-judge — and scaling LLM-as-judge specifically does not close the generalization gap.
- Note where Ilya locates the same intuition (§01:29:23): diversity *between* agents (self-play, debate, prover-verifier). The brain obtains it *within* one agent via organs with genuinely different objectives — cheaper, and on the timescale of a single action rather than a population.

### A looped transformer + cerebellar-style fast predictor is a viable architecture
*Confidence: speculative*

- A fast auxiliary module predicts activations via online-updated weight decomposition, injects a "draft" into the looped transformer's recurrence — discussed here[^private]
- Easy inputs (good prediction) converge in 1-2 loops; hard inputs (bad prediction) need many — adaptive compute almost for free — same source
- Implementable via low-rank linear predictors, fast weight systems, or key-value memories — same source
- Hard parts: credit assignment between the two systems, stability of online updates, dimensionality of the prediction target — same source

### Error signals should be precision-weighted and structurally decomposed, not raw residual magnitude
*Confidence: contested*

- The neuroscience suggests: normalize errors by estimated variance, decompose residuals (treat low-rank structured errors differently from diffuse noise), track temporal persistence, gate by model confidence — developed here[^private]
- The right stack is probably hardcoded structural priors (information-theoretic, like the genome's conserved value function) + learned error classification (domain-specific, like amygdala/cortical valence learning) — same source
- **Negative result**: precision weighting the local loss (Mahalanobis distance, weighting each dimension by inverse FM error variance) reduced brittleness only 29% (9.5× vs 13.4×) and made no difference when combined with injection (CL_PW ≈ CL_LL on every metric). The FM's error structure is unrelated to task relevance — precision weighting addresses a non-bottleneck — [precision weighting](../../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md#precision-weighted-local-loss-2026-06-16)
- **Positive result**: a bilevel-optimized learning gate (task-informed, not FM-informed weighting) reduced brittleness 35% and produced record self-knowledge (R²=0.83). The gate's selectivity doesn't correlate with FM error variance or digit discrimination — it discovers a more abstract task-relevant criterion via bilevel optimization — [learning gate](../../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md#learning-gate-bilevel-optimized-local-loss-2026-06-16)
- See also: [Abstraction supervision as metacognitive control](../abstraction_supervision_as_metacognitive_control.md), [Metacognitive novelty learning](../metacognitive_novelty_learning.md)

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
