# `committee_head/` — file index

Up: [`../README.md`](../README.md) (mjc) · Design: [SPEC.md](SPEC.md) · Node files: this file.
Substrate parent: [`../on_policy/directed_on_policy/`](../on_policy/directed_on_policy/README.md) (E3).

## Code files

| file | purpose |
|---|---|
| `committee_head.py` | **Phases A + B.** E3's directed-collection loop forked, with the single forward model replaced by a **K-member random-prior (RPF) committee** whose *mean* plans (so `fm_visits` and the ballistic control are unchanged in kind). Per region per round it reads a reward-free **signature** off what the loop already computes — committee-mean error `e`, disagreement `d`, benchmarked error `b(s)−e` (an online benchmark net fitted to the committee's own error stream, the `plasticity_gain` idiom), the agency gap `g` (an arity-1 net, the `agency_gate` idiom), plan occupancy `v`, and `hist_len` lags of each plus the lagged allocation share — and a small **head shared across regions** maps one region's signature to (predicted reducibility, predicted relevance), whose product allocates the budget behind S2's no-op floor. Policies: `uniform,error-only,disagree-only,lprog-only,visits-only,value,oracle,head_sup[-nod\|-nobme\|-nov\|-nog\|-nohist\|-lp],head_hetero,head_rl,steady,burst`. |
| `check_fidelity.py` | **The Phase A#1 gate.** Local comparator: `committee_head`'s `value` arm at `--k-members 1 --rpf-beta 0` must reproduce E3's `value` arm **bit-identically** (`array_equal`, not `allclose`) on every per-round quantity plus the setup artifacts. Reads the two locally mirrored `results.json`; the two `modal run` commands that produce them are in its docstring. |
| `committee_head_rhm.py` | **Phase C — calibration on RHM, the oracle scoring but never feeding.** The same head shape and the same reward-free training, on the frozen [`conditional_revision`](../../rhm/conditional_revision/README.md) reader with a committee of **temporal** activation FMs (one-token conditioning gap, so the residual has a genuine aleatoric leg), scored afterwards against [`oracle.py`](../../rhm/conditional_revision/oracle.py)'s exact reducible/irreducible split. Head fit on one split of sequences, scored on a disjoint one. Never retrains the base model. |
| `committee_head_agg.py` | Cross-tag / cross-seed aggregator. Four tables: the ladder (A-err, ballistic, budget shares, monitor:collect), split quality (head prediction vs realised outcome, per region per round), the ablation table, and Phase A (raw-channel separability AUROC, committee health, benchmark estimability, the §9 discriminator). Writes `fig_ladder.png` and `fig_signature.png`. |
| `__init__.py` | Package marker, so `.add_local_python_source("mjc")` ships this folder. |

## Auxiliary docs

| file | purpose |
|---|---|
| [`README.md`](README.md) | The writeup — the head learns the three-way split (0.857, leave-one-region-out 0.882), the two substrate defects and their gates, the E3 recheck under matched steps, and Phase C as an identified orthogonality. |
| [`SPEC.md`](SPEC.md) | The design doc this node was built from (2026-09-02, pre-run). |
| `CONVERSATION.md`[^private] | Session record — Jasper's prompts verbatim, assistant turns summarised. Carries the *motivations* and the four refuted hypotheses, which the README compresses out. |

## What is deliberately NOT forked from E3

`directed_on_policy.py`'s **render pack** (`--render-rounds`) and its replayer `render_ladder.py`. It is
a communication artifact whose replayer does not know about committees, and it is off by default, so no
E3 result depends on it and nothing here needs it. Everything else in E3's loop is forked as written,
including its three geometry gotchas and its off-reach measurement hole (regions the body cannot reach
get zero grader probes and report `nan` — reproduced here, see any run's `[probe]` block).

## The fidelity contract (why the fork is checkable rather than asserted)

Every addition uses its own RNG streams and never touches the shared ones:

- committee member `k` draws from **E3's stream + 7919·k**, so member 0 *is* E3;
- at `rpf_beta = 0` no prior net is constructed and `mem_forward` reduces to `net(Xn)` exactly;
- the benchmark net, the arity-1 net and the counterfactual LP probes each hold dedicated generators;
- the head refits **full-batch**, so it consumes no RNG at all — which is what lets a *shadow* head
  train on every arm without perturbing the arm it is watching.

`--quick` is E3's `--quick` verbatim, so the gate is a bit-for-bit comparison and not an approximate one.

## Modal volume layout

```
/data/committee_head/<tag>/results.json                      # Phases A+B  (mujoco-control-data)
/data/<rhm_key>/committee_head_phase_c/<tag>/results.json     # Phase C     (rhm-scaling-data)
```

Local mirrors + figures: `figures/committee_head_<tag>/`, `figures/phase_c_<tag>/`,
`figures/agg_<tags>/` (PNGs regenerable, git-ignored per the repo convention).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
