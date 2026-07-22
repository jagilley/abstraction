# STRUCTURE.md — Canonical repo structure

Companion to [`experiments/CLAUDE.md`](experiments/CLAUDE.md), which specifies the *content* conventions for a README/`FILES.md` pair. This file specifies the *shape* of the tree those docs live in.

## The one principle

**The filesystem is the abstraction hierarchy.** Every meaningful unit of work is a **folder**, and every folder carries its own `README.md` written at *that folder's altitude*. The goal: point an agent at the README of any folder, at any depth, and it lands with exactly the context that layer needs — a headline map of what's below it, and a pointer to climb to what's above.

## A node = a folder

Each node folder contains:

- **`README.md`** — the index/writeup *for this node, at this node's altitude*. It is the default context injected into a Claude Code session rooted here, so it stays under the ~25K-token injection cap (see [`experiments/CLAUDE.md`](experiments/CLAUDE.md)). It carries: the goal, the architecture/setup, a one-headline-per-child map of the children below, a pointer **up** to the parent README, and pointers **down** to each child README.
- **`FILES.md`** *(when the node has its own code and/or several children)* — the complete file-by-file index that would otherwise bloat the README: a `## Code files` table (every `.py` living directly in this folder → one-line purpose) and a `## Children` table (every child folder → one-line summary + link to its `README.md`). A leaf node (a lone writeup) doesn't need one.
- **child folders** — each is itself a node, recursively.

"Experiment" and "sub-experiment" are just names for nodes at different depths; the structure is uniform all the way down.

## Writeups are folders, not sibling files

**This is the migration.** The old convention put an auxiliary writeup as a flat `TOPIC_README.md` sitting at the *same* layer as its parent `README.md` — the parent/child relationship only implied by the `_README` suffix. The new convention gives each writeup **its own folder**, with the writeup inside as `README.md`, so the tree *is* the relationship:

```
# OLD (flat — relationship only implied)          # NEW (nested — the tree is the relationship)
a2a_forward/                                       a2a_forward/
  README.md                                          README.md                   <- a2a_forward index
  MIRROR_TEST_README.md                              FILES.md
  reaching/                                          mirror_test/
    README.md                                          README.md                 <- the mirror-test writeup
    CURIOSITY_CONTROL_README.md                      reaching/
                                                       README.md                 <- reaching-arc index
                                                       FILES.md
                                                       curiosity_control/
                                                         README.md               <- the curiosity-control writeup
```

`reaching/` was already this pattern — a folder grouping several child writeups. The change is to make *every* writeup a folder the same way, so `CURIOSITY_CONTROL_README.md` becomes `curiosity_control/README.md`.

Naming: the folder is the `snake_case` of the topic; the writeup inside is always literally `README.md`; the full index is always `FILES.md`.

## Summaries halve as you climb

Full detail lives once, in the writeup's own `README.md`. Every level *above* it carries a *summary* of it, and that summary roughly **halves in length at each step up**: the immediate parent gets a headline paragraph, the grandparent a sentence, the great-grandparent a clause. So the leaf holds everything, mid-levels hold paragraphs, and the top-level README holds one-liners — which is exactly what keeps each README scoped to its altitude and under the injection cap. The `/writeup`[^private] skill writes the leaf and propagates these halving summaries upward.

## Code files

Code files are also semantic. Files that are shared or imported by the auxiliary READMEs' code files (e.g., `experiments/a2a_forward/shared.py`) should be kept at the parent level, to be properly imported. Files associated with auxiliary READMEs should be in the child folder.

## Mechanics that keep it from breaking

- **Pointers both ways.** Every README links up to its parent and down to each child. This is what lets an agent enter at any layer and still climb or descend for context.
- **Link depth.** A writeup nested one level deeper gains a `../` on every relative link to a shared dir (`../../ideas/` → `../../../ideas/`) and to sibling code. Fix these when you nest — a "moved" doc with dead links is strictly worse than an un-moved one.
- **Code placement.** Code lives at the *lowest node that shares it*: a script only one leaf uses can live in that leaf's folder; a module several children import stays at their common-ancestor node. `FILES.md` at each node indexes only the code that lives directly there.
- **Migrate slowly, reproducibly.** Turning a `TOPIC_README.md` into `topic/README.md` is a semantic reorganization — use `/semantic-reorg`[^private]: `git mv`, fix inbound links repo-wide, fix outbound link depth, and never touch `/data/...` result paths (they're independent of source location). The two forms may coexist during the transition; there is no rush.

## See also
- [`experiments/CLAUDE.md`](experiments/CLAUDE.md) — content conventions for the README/`FILES.md` split and the injection-cap rationale.
- `.claude/skills/writeup/SKILL.md`[^private] — the `/writeup` flow: write a leaf README, then propagate halving summaries upward.
- `.claude/skills/semantic-reorg/SKILL.md`[^private] — how to move a doc into its new folder without breaking references.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
