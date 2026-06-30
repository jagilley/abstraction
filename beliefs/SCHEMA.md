# Belief System

## What this is

A structured representation of our research beliefs as a traversable hierarchy. Inspired by cognitive schema theory (Piaget, Bartlett): beliefs form trees where abstract principles sit at the root and specific empirical claims sit at the leaves.

## Why it exists

When new evidence arrives, we need to quickly assess:
1. Which existing beliefs does it bear on?
2. Does it support, refine, or contradict those beliefs?
3. How far up the tree does the disturbance propagate?

A leaf being wrong is cheap. A root being wrong means rethinking everything beneath it.

## Structure

Belief trees live in `beliefs/trees/`. Each file is a single domain's belief hierarchy.

Existing essay-format beliefs in `beliefs/*.md` remain as deeper explorations. Trees reference them where relevant.

## Format

The hierarchy uses markdown headings for structure. Each belief node has:
- **Statement**: A single clear sentence, as a heading or bold text.
- **Confidence**: `established` | `strong` | `moderate` | `speculative` | `contested`
- **Evidence**: Bullet list of positive evidence — factual claims, observations, or reasoning that support the belief. Each item should link to a file in the repo where an agent can read more.

```markdown
## The belief statement goes here
*Confidence: moderate*

- Evidence point one — [source](path/to/file.md)
- Evidence point two — [source](path/to/other/file.md)
- A prior knowledge fact that supports this, no link needed if it's textbook-level

### A child belief, more specific
*Confidence: speculative*

- Evidence for this child — [source](path/to/file.md)
```

Heading depth encodes tree depth: `##` is a root belief, `###` is a child, `####` is a grandchild. For deeper nesting within a section, use **bold text** for the belief statement and an indented evidence list.

## Operations

Two Claude skills operate on this system:

- `/add-belief` — Integrates a new belief or piece of evidence into the appropriate tree. Finds the right location in the hierarchy, checks for consistency, and creates or modifies nodes.

- `/update-beliefs` — Takes new evidence (a paper, a result, a conversation) and traverses the existing trees to assess what's supported, what's threatened, and how far up the tree the impact goes.

## Schema dynamics

Following Piaget:
- **Assimilation**: New evidence fits an existing belief node. Confidence may increase; evidence is appended.
- **Accommodation**: New evidence forces restructuring. Nodes are modified, split, or reorganized.
- **Disequilibrium**: Evidence contradicts a belief but we haven't resolved how. Flagged with `contested` confidence and an explicit note.
